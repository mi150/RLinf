"""Benchmark latency-aware LIBERO task scheduling with real env.step calls."""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import queue
import random
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class ProcessRunResult:
    events: list[StepEvent]
    errors: list[dict[str, Any]]


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


def _validate_schedule_inputs(plan: list[ScheduleItem], *, steps_per_env: int) -> None:
    if not plan:
        raise ValueError("plan must not be empty")
    if steps_per_env < 1:
        raise ValueError("steps_per_env must be >= 1")
    task_ids = [item.task.task_id for item in plan]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("duplicate task_id in schedule plan")


def apply_cpu_affinity(cpu_id: int | None) -> bool:
    if cpu_id is None or not hasattr(os, "sched_setaffinity"):
        return False
    try:
        os.sched_setaffinity(0, {cpu_id})
    except OSError:
        return False
    return True


def build_worker_plans(plan: list[ScheduleItem]) -> dict[int, list[ScheduleItem]]:
    return _items_by_core(plan)


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


def _next_round_commands(
    grouped: dict[int, list[ScheduleItem]],
    per_task_counts: dict[int, int],
    cursors: dict[int, int],
    *,
    steps_per_env: int,
) -> list[tuple[int, ScheduleItem]]:
    commands = []
    for core_index in sorted(grouped):
        item, next_cursor = _next_item_for_core(
            grouped[core_index],
            per_task_counts,
            steps_per_env=steps_per_env,
            cursor=cursors[core_index],
        )
        cursors[core_index] = next_cursor
        if item is None:
            continue
        per_task_counts[item.task.task_id] += 1
        commands.append((core_index, item))
    return commands


def _worker_loop(
    *,
    core_index: int,
    items: list[ScheduleItem],
    steps_per_env: int,
    env_factory: Any,
    dummy_action: list[float],
    command_queue: Any,
    result_queue: Any,
) -> None:
    del steps_per_env
    cpu_id = items[0].cpu_id if items else None
    affinity_applied = apply_cpu_affinity(cpu_id)
    envs: dict[int, Any] = {}
    task_counts = {item.task.task_id: 0 for item in items}
    current_item: ScheduleItem | None = None
    current_step_index: int | None = None
    try:
        for item in items:
            current_item = item
            if item.task.task_id not in envs:
                envs[item.task.task_id] = env_factory(item)
        current_item = None
        while True:
            command = command_queue.get()
            if command == "stop":
                break
            round_index, item = command
            current_item = item
            env = envs[item.task.task_id]
            task_step_index = task_counts[item.task.task_id]
            current_step_index = task_step_index
            start = time.perf_counter()
            env.step(np.asarray(dummy_action, dtype=np.float32))
            latency_s = max(float(time.perf_counter() - start), 0.0)
            task_counts[item.task.task_id] = task_step_index + 1
            result_queue.put(
                {
                    "event": "step",
                    "round_index": round_index,
                    "core_index": core_index,
                    "cpu_id": cpu_id,
                    "task_id": item.task.task_id,
                    "task_name": item.task.task_name,
                    "task_step_index": task_step_index,
                    "latency_s": latency_s,
                    "cpu_affinity_applied": affinity_applied,
                }
            )
            current_item = None
            current_step_index = None
    except Exception as exc:
        error = {
            "event": "error",
            "core_index": core_index,
            "cpu_id": cpu_id,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        if current_item is not None:
            error.update(
                {
                    "task_id": current_item.task.task_id,
                    "task_name": current_item.task.task_name,
                    "task_step_index": current_step_index,
                }
            )
        result_queue.put(error)
    finally:
        for env in envs.values():
            close = getattr(env, "close", None)
            if close is not None:
                close()


def _coerce_worker_latency(raw_result: dict[str, Any], *, schedule_name: str) -> float:
    raw_latency = raw_result.get("latency_s")
    try:
        latency_s = float(raw_latency)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "invalid worker latency for "
            f"schedule_name={schedule_name}, round_index={raw_result.get('round_index')}, "
            f"core_index={raw_result.get('core_index')}, "
            f"task_id={raw_result.get('task_id')}: {raw_latency!r}"
        ) from exc
    if not np.isfinite(latency_s) or latency_s < 0.0:
        raise ValueError(
            "invalid worker latency for "
            f"schedule_name={schedule_name}, round_index={raw_result.get('round_index')}, "
            f"core_index={raw_result.get('core_index')}, "
            f"task_id={raw_result.get('task_id')}: {latency_s!r}"
        )
    return latency_s


def _worker_latency_error(exc: Exception, *, schedule_name: str) -> dict[str, Any]:
    return {
        "event": "error",
        "schedule_name": schedule_name,
        "error_type": exc.__class__.__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def run_schedule_with_process_workers(
    plan: list[ScheduleItem],
    *,
    steps_per_env: int,
    env_factory: Any,
    dummy_action: list[float],
    subprocess_timeout_s: float = 300.0,
    mp_context: Any | None = None,
) -> ProcessRunResult:
    _validate_schedule_inputs(plan, steps_per_env=steps_per_env)
    schedule_name = plan[0].schedule_name
    grouped = build_worker_plans(plan)
    per_task_counts = {item.task.task_id: 0 for item in plan}
    cursors = dict.fromkeys(grouped, 0)
    ctx = mp_context or mp.get_context("spawn")
    result_queue = ctx.Queue()
    command_queues: dict[int, Any] = {}
    processes = []
    events: list[StepEvent] = []
    try:
        for core_index, items in sorted(grouped.items()):
            command_queue = ctx.Queue()
            command_queues[core_index] = command_queue
            process = ctx.Process(
                target=_worker_loop,
                kwargs={
                    "core_index": core_index,
                    "items": items,
                    "steps_per_env": steps_per_env,
                    "env_factory": env_factory,
                    "dummy_action": dummy_action,
                    "command_queue": command_queue,
                    "result_queue": result_queue,
                },
            )
            process.start()
            processes.append(process)

        round_index = 0
        while any(count < steps_per_env for count in per_task_counts.values()):
            commands = _next_round_commands(
                grouped,
                per_task_counts,
                cursors,
                steps_per_env=steps_per_env,
            )
            if not commands:
                break
            for core_index, item in commands:
                command_queues[core_index].put((round_index, item))

            round_results = []
            deadline = time.monotonic() + subprocess_timeout_s
            for _ in commands:
                remaining_s = max(deadline - time.monotonic(), 0.0)
                try:
                    result = result_queue.get(timeout=remaining_s)
                except queue.Empty:
                    return ProcessRunResult(
                        events=events,
                        errors=[
                            {
                                "event": "timeout",
                                "schedule_name": schedule_name,
                                "round_index": round_index,
                                "error_type": "TimeoutError",
                                "error": (
                                    "timed out waiting for process worker result "
                                    f"after {subprocess_timeout_s:.3f}s"
                                ),
                            }
                        ],
                    )
                if result.get("event") == "error":
                    result.setdefault("schedule_name", schedule_name)
                    return ProcessRunResult(events=events, errors=[result])
                if result.get("event") != "step":
                    return ProcessRunResult(
                        events=events,
                        errors=[
                            {
                                "event": "error",
                                "schedule_name": schedule_name,
                                "round_index": round_index,
                                "error_type": "ValueError",
                                "error": f"unexpected worker result: {result!r}",
                            }
                        ],
                    )
                round_results.append(result)

            try:
                round_wall_time_s = max(
                    _coerce_worker_latency(result, schedule_name=schedule_name)
                    for result in round_results
                )
            except ValueError as exc:
                return ProcessRunResult(
                    events=events,
                    errors=[_worker_latency_error(exc, schedule_name=schedule_name)],
                )

            for result in round_results:
                latency_s = _coerce_worker_latency(result, schedule_name=schedule_name)
                events.append(
                    StepEvent(
                        schedule_name=schedule_name,
                        round_index=int(result["round_index"]),
                        core_index=int(result["core_index"]),
                        cpu_id=int(result["cpu_id"]),
                        task_id=int(result["task_id"]),
                        task_name=str(result["task_name"]),
                        task_step_index=int(result["task_step_index"]),
                        latency_s=latency_s,
                        round_wall_time_s=round_wall_time_s,
                        idle_time_s=max(round_wall_time_s - latency_s, 0.0),
                        cpu_affinity_applied=bool(result["cpu_affinity_applied"]),
                    )
                )
            round_index += 1
        return ProcessRunResult(events=events, errors=[])
    finally:
        for command_queue in command_queues.values():
            command_queue.put("stop")
        for process in processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)


def run_schedule_with_step_function(
    plan: list[ScheduleItem],
    *,
    steps_per_env: int,
    step_fn: Any,
    cpu_affinity_by_core: dict[int, bool] | None = None,
) -> list[StepEvent]:
    _validate_schedule_inputs(plan, steps_per_env=steps_per_env)
    grouped = _items_by_core(plan)
    per_task_counts = {item.task.task_id: 0 for item in plan}
    cursors = dict.fromkeys(grouped, 0)
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
            raw_latency = step_fn(item, task_step_index)
            try:
                latency_s = float(raw_latency)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "invalid latency for "
                    f"task_id={item.task.task_id}, core_index={item.core_index}, "
                    f"step_index={task_step_index}: {raw_latency!r}"
                ) from exc
            if not np.isfinite(latency_s) or latency_s < 0.0:
                raise ValueError(
                    "invalid latency for "
                    f"task_id={item.task.task_id}, core_index={item.core_index}, "
                    f"step_index={task_step_index}: {latency_s!r}"
                )
            per_task_counts[item.task.task_id] = task_step_index + 1
            round_results.append((item, task_step_index, latency_s))
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
    rounds: dict[int, list[StepEvent]] = {}
    for event in events:
        rounds.setdefault(event.round_index, []).append(event)
    total_cores = len({event.core_index for event in events})
    makespan_s = float(
        sum(max(event.round_wall_time_s for event in round_events) for round_events in rounds.values())
    )
    latencies = [event.latency_s for event in events]
    idle_ratios: list[float] = []
    for round_events in rounds.values():
        round_wall_time_s = max(event.round_wall_time_s for event in round_events)
        per_round_idle_ratios = [
            0.0 if round_wall_time_s == 0.0 else event.idle_time_s / round_wall_time_s
            for event in round_events
        ]
        missing_cores = total_cores - len({event.core_index for event in round_events})
        if missing_cores > 0:
            per_round_idle_ratios.extend([1.0 if round_wall_time_s > 0.0 else 0.0] * missing_cores)
        idle_ratios.append(float(np.mean(np.asarray(per_round_idle_ratios))))
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


@dataclass
class LiberoEnvFactory:
    suite: str
    camera_height: int
    camera_width: int
    libero_type: str
    seed: int
    warmup_steps: int
    dummy_action: list[float]
    task_spec_cache: dict[int, Any] = field(default_factory=dict)
    init_state_cache: dict[int, Any] = field(default_factory=dict)

    def __call__(self, item: ScheduleItem) -> Any:
        from toolkits.profile_libero_step_latency import (
            ProfileConfig,
            build_task_trial_specs,
        )
        from toolkits.profile_libero_step_latency import (
            make_libero_env_factory as make_profile_env_factory,
        )

        if item.task.task_id not in self.task_spec_cache:
            config = ProfileConfig(
                suite=self.suite,
                task_ids=str(item.task.task_id),
                trials_per_task=1,
                specific_trial_ids=None,
                warmup_steps=self.warmup_steps,
                measure_steps=1,
                cpu_id=item.cpu_id,
                cpu_ids=None,
                camera_height=self.camera_height,
                camera_width=self.camera_width,
                libero_type=self.libero_type,
                seed=self.seed,
                output_dir=Path("."),
                dummy_action=self.dummy_action,
                stop_on_done=False,
                subprocess_timeout_s=None,
                mujoco_profiler=False,
            )
            specs, init_states = build_task_trial_specs(config)
            self.task_spec_cache[item.task.task_id] = (config, specs[0])
            self.init_state_cache[item.task.task_id] = init_states[0]
        config, spec = self.task_spec_cache[item.task.task_id]
        env = make_profile_env_factory(config, spec)()
        if hasattr(env, "seed"):
            env.seed(spec.seed)
        env.reset()
        init_state = self.init_state_cache[item.task.task_id]
        if init_state is not None and hasattr(env, "set_init_state"):
            env.set_init_state(init_state)
        for _ in range(self.warmup_steps):
            env.step(np.asarray(self.dummy_action, dtype=np.float32))
        env.reset()
        if init_state is not None and hasattr(env, "set_init_state"):
            env.set_init_state(init_state)
        return env


def make_libero_env_factory(
    *,
    suite: str,
    camera_height: int,
    camera_width: int,
    libero_type: str,
    seed: int,
    warmup_steps: int,
    dummy_action: list[float],
) -> Any:
    return LiberoEnvFactory(
        suite=suite,
        camera_height=camera_height,
        camera_width=camera_width,
        libero_type=libero_type,
        seed=seed,
        warmup_steps=warmup_steps,
        dummy_action=dummy_action,
    )


@dataclass(frozen=True)
class BenchmarkResult:
    schedule_name: str
    events: list[StepEvent]
    summary: dict[str, Any]
    errors: list[dict[str, Any]]


class BenchmarkRunner:
    def __init__(
        self,
        *,
        steps_per_env: int,
        step_fn: Any | None = None,
        env_factory: Any | None = None,
        dummy_action: list[float] | None = None,
        subprocess_timeout_s: float = 300.0,
    ) -> None:
        self.steps_per_env = steps_per_env
        self.step_fn = step_fn
        self.env_factory = env_factory
        self.dummy_action = dummy_action
        self.subprocess_timeout_s = subprocess_timeout_s

    def run(self, schedule_name: str, plan: list[ScheduleItem]) -> BenchmarkResult:
        try:
            if self.step_fn is not None:
                events = run_schedule_with_step_function(
                    plan,
                    steps_per_env=self.steps_per_env,
                    step_fn=self.step_fn,
                )
                errors: list[dict[str, Any]] = []
            else:
                if self.env_factory is None:
                    raise RuntimeError("env_factory is required when step_fn is not provided")
                if self.dummy_action is None:
                    raise RuntimeError("dummy_action is required when step_fn is not provided")
                process_result = run_schedule_with_process_workers(
                    plan,
                    steps_per_env=self.steps_per_env,
                    env_factory=self.env_factory,
                    dummy_action=self.dummy_action,
                    subprocess_timeout_s=self.subprocess_timeout_s,
                )
                events = process_result.events
                errors = process_result.errors
            summary = compute_schedule_summary(schedule_name, events, failed=bool(errors))
            return BenchmarkResult(
                schedule_name=schedule_name,
                events=events,
                summary=summary,
                errors=errors,
            )
        except Exception as exc:
            summary = compute_schedule_summary(schedule_name, [], failed=True)
            return BenchmarkResult(
                schedule_name=schedule_name,
                events=[],
                summary=summary,
                errors=[
                    {
                        "event": "error",
                        "schedule_name": schedule_name,
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                ],
            )
