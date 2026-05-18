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
