"""Benchmark latency-aware LIBERO task scheduling with real env.step calls."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

REQUIRED_TASK_COLUMNS = {"task_id", "task_name", "mean_latency_ms", "njnt", "ngeom"}


@dataclass(frozen=True)
class TaskRecord:
    task_id: int
    task_name: str
    mean_latency_ms: float
    njnt: int
    ngeom: int
    estimated_latency_score: float = 0.0
    extra: dict[str, str] = field(default_factory=dict)


def _parse_int(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer column {key!r}: {row.get(key)!r}") from exc


def _parse_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid float column {key!r}: {row.get(key)!r}") from exc


def load_task_records(path: str | Path) -> list[TaskRecord]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_TASK_COLUMNS - fieldnames)
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")
        records = []
        for row in reader:
            extra = {
                key: value
                for key, value in row.items()
                if key not in REQUIRED_TASK_COLUMNS and value is not None
            }
            records.append(
                TaskRecord(
                    task_id=_parse_int(row, "task_id"),
                    task_name=row["task_name"],
                    mean_latency_ms=_parse_float(row, "mean_latency_ms"),
                    njnt=_parse_int(row, "njnt"),
                    ngeom=_parse_int(row, "ngeom"),
                    extra=extra,
                )
            )
    return records


def sample_task_records(
    records: list[TaskRecord],
    *,
    num_envs: int,
    seed: int,
) -> list[TaskRecord]:
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")
    if num_envs > len(records):
        raise ValueError(
            f"num_envs={num_envs} exceeds available task records={len(records)}"
        )
    rng = random.Random(seed)
    return rng.sample(records, num_envs)


def _z_scores(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    std = float(np.std(array))
    if std == 0.0:
        return [0.0 for _ in values]
    mean = float(np.mean(array))
    return [float((value - mean) / std) for value in array]


def estimate_latency_scores(
    records: list[TaskRecord],
    *,
    weight_jnt: float = 0.45,
    weight_geom: float = 0.55,
) -> list[TaskRecord]:
    jnt_scores = _z_scores([float(record.njnt) for record in records])
    geom_scores = _z_scores([float(record.ngeom) for record in records])
    scored = []
    for record, jnt_score, geom_score in zip(records, jnt_scores, geom_scores):
        score = weight_jnt * jnt_score + weight_geom * geom_score
        scored.append(replace(record, estimated_latency_score=float(score)))
    return scored


TASK_ID_BASELINE = "task_id_baseline"
RANDOM_BASELINE = "random_baseline"
TRAPEZOID_PIPELINE = "trapezoid_pipeline"


@dataclass(frozen=True)
class ScheduleItem:
    schedule_name: str
    task: TaskRecord
    core_index: int
    cpu_id: int
    layer_index: int
    order_index: int
    side: str = "baseline"


@dataclass(frozen=True)
class StepEvent:
    schedule_name: str
    round_index: int
    core_index: int
    cpu_id: int
    task_id: int
    task_name: str
    task_step_index: int
    latency_s: float
    round_wall_time_s: float
    idle_time_s: float
    cpu_affinity_applied: bool


def _require_cpu_ids(cpu_ids: list[int]) -> None:
    if not cpu_ids:
        raise ValueError("cpu_ids must not be empty")


def _layered_plan(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
    schedule_name: str,
    side: str = "baseline",
    order_offset: int = 0,
) -> list[ScheduleItem]:
    _require_cpu_ids(cpu_ids)
    items = []
    for index, record in enumerate(records):
        core_index = index % len(cpu_ids)
        layer_index = index // len(cpu_ids)
        items.append(
            ScheduleItem(
                schedule_name=schedule_name,
                task=record,
                core_index=core_index,
                cpu_id=cpu_ids[core_index],
                layer_index=layer_index,
                order_index=order_offset + index,
                side=side,
            )
        )
    return items


def build_task_id_baseline_plan(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
) -> list[ScheduleItem]:
    ordered = sorted(records, key=lambda record: record.task_id)
    return _layered_plan(
        ordered,
        cpu_ids=cpu_ids,
        schedule_name=TASK_ID_BASELINE,
    )


def build_random_baseline_plan(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
    seed: int,
) -> list[ScheduleItem]:
    ordered = list(records)
    random.Random(seed).shuffle(ordered)
    return _layered_plan(
        ordered,
        cpu_ids=cpu_ids,
        schedule_name=RANDOM_BASELINE,
    )


def build_trapezoid_pipeline_plan(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
) -> list[ScheduleItem]:
    _require_cpu_ids(cpu_ids)
    if len(records) % 2 != 0:
        raise ValueError("trapezoid_pipeline requires an even number of tasks")
    ordered = sorted(
        records,
        key=lambda record: (-record.estimated_latency_score, record.task_id),
    )
    split = len(ordered) // 2
    long_half = ordered[:split]
    short_half = list(reversed(ordered[split:]))
    long_items = _layered_plan(
        long_half,
        cpu_ids=cpu_ids,
        schedule_name=TRAPEZOID_PIPELINE,
        side="long",
    )
    short_items = _layered_plan(
        short_half,
        cpu_ids=cpu_ids,
        schedule_name=TRAPEZOID_PIPELINE,
        side="short",
        order_offset=len(long_items),
    )
    return long_items + short_items


def _items_by_core(plan: list[ScheduleItem]) -> dict[int, list[ScheduleItem]]:
    grouped: dict[int, list[ScheduleItem]] = {}
    for item in sorted(plan, key=lambda value: (value.core_index, value.order_index)):
        grouped.setdefault(item.core_index, []).append(item)
    return grouped


def _next_item_for_core(
    items: list[ScheduleItem],
    per_task_counts: dict[int, int],
    *,
    steps_per_env: int,
    cursor: int,
) -> tuple[ScheduleItem | None, int]:
    if not items:
        return None, cursor
    for offset in range(len(items)):
        index = (cursor + offset) % len(items)
        item = items[index]
        if per_task_counts.get(item.task.task_id, 0) < steps_per_env:
            return item, (index + 1) % len(items)
    return None, cursor


def run_schedule_with_step_function(
    plan: list[ScheduleItem],
    *,
    steps_per_env: int,
    step_fn: Any,
    cpu_affinity_by_core: dict[int, bool] | None = None,
) -> list[StepEvent]:
    if steps_per_env < 1:
        raise ValueError("steps_per_env must be >= 1")
    grouped = _items_by_core(plan)
    per_task_counts = {item.task.task_id: 0 for item in plan}
    cursors = {core_index: 0 for core_index in grouped}
    events: list[StepEvent] = []
    round_index = 0
    while any(count < steps_per_env for count in per_task_counts.values()):
        round_results = []
        for core_index, items in grouped.items():
            item, next_cursor = _next_item_for_core(
                items,
                per_task_counts,
                steps_per_env=steps_per_env,
                cursor=cursors[core_index],
            )
            cursors[core_index] = next_cursor
            if item is None:
                continue
            task_step_index = per_task_counts[item.task.task_id]
            latency_s = float(step_fn(item, task_step_index))
            per_task_counts[item.task.task_id] = task_step_index + 1
            round_results.append((item, task_step_index, max(latency_s, 0.0)))
        if not round_results:
            break
        round_wall_time_s = max(latency for _, _, latency in round_results)
        for item, task_step_index, latency_s in round_results:
            affinity = True
            if cpu_affinity_by_core is not None:
                affinity = bool(cpu_affinity_by_core.get(item.core_index, True))
            events.append(
                StepEvent(
                    schedule_name=item.schedule_name,
                    round_index=round_index,
                    core_index=item.core_index,
                    cpu_id=item.cpu_id,
                    task_id=item.task.task_id,
                    task_name=item.task.task_name,
                    task_step_index=task_step_index,
                    latency_s=latency_s,
                    round_wall_time_s=round_wall_time_s,
                    idle_time_s=max(round_wall_time_s - latency_s, 0.0),
                    cpu_affinity_applied=affinity,
                )
            )
        round_index += 1
    return events


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def compute_schedule_summary(
    schedule_name: str,
    events: list[StepEvent],
    *,
    failed: bool = False,
) -> dict[str, Any]:
    if failed:
        status = "failed"
    elif any(not event.cpu_affinity_applied for event in events):
        status = "degraded"
    else:
        status = "completed"
    if not events:
        return {
            "schedule_name": schedule_name,
            "status": status,
            "total_steps": 0,
            "makespan_s": 0.0,
            "steps_per_second": 0.0,
            "mean_core_idle_ratio": None,
            "cpu_affinity_success_rate": None,
        }
    round_wall_times = {
        event.round_index: event.round_wall_time_s for event in events
    }
    makespan_s = float(sum(round_wall_times.values()))
    latencies = [event.latency_s for event in events]
    idle_ratios = [
        0.0 if event.round_wall_time_s == 0.0 else event.idle_time_s / event.round_wall_time_s
        for event in events
    ]
    affinity_rate = sum(1 for event in events if event.cpu_affinity_applied) / len(events)
    return {
        "schedule_name": schedule_name,
        "status": status,
        "total_steps": len(events),
        "makespan_s": makespan_s,
        "steps_per_second": 0.0 if makespan_s == 0.0 else len(events) / makespan_s,
        "mean_step_latency_s": float(np.mean(np.asarray(latencies))),
        "median_step_latency_s": float(np.median(np.asarray(latencies))),
        "p90_step_latency_s": _percentile(latencies, 90),
        "p95_step_latency_s": _percentile(latencies, 95),
        "p99_step_latency_s": _percentile(latencies, 99),
        "mean_core_idle_ratio": float(np.mean(np.asarray(idle_ratios))),
        "p90_round_idle_ratio": _percentile(idle_ratios, 90),
        "p99_round_idle_ratio": _percentile(idle_ratios, 99),
        "cpu_affinity_success_rate": float(affinity_rate),
    }
