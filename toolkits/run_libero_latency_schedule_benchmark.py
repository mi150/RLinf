"""Benchmark latency-aware LIBERO task scheduling with real env.step calls."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import queue
import random
import time
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass, replace
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
PHASE_SHIFTED_TRAPEZOID = "phase_shifted_trapezoid"


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


def build_phase_shifted_trapezoid_plan(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
) -> list[ScheduleItem]:
    plan = build_trapezoid_pipeline_plan(records, cpu_ids=cpu_ids)
    long_items_by_core = {
        item.core_index: item
        for item in plan
        if item.side == "long" and item.layer_index == 0
    }
    sorted_long_cores = sorted(
        long_items_by_core,
        key=lambda core_index: (
            -long_items_by_core[core_index].task.estimated_latency_score,
            long_items_by_core[core_index].task.task_id,
            core_index,
        ),
    )
    short_first_count = 0 if len(sorted_long_cores) == 1 else len(sorted_long_cores) // 2
    short_first_cores = set(sorted_long_cores[:short_first_count])
    shifted = []
    for item in plan:
        schedule_name = PHASE_SHIFTED_TRAPEZOID
        if item.core_index not in short_first_cores:
            order_index = item.order_index
        elif item.side == "long":
            order_index = item.order_index + len(plan)
        else:
            order_index = item.order_index - len(plan)
        shifted.append(
            replace(
                item,
                schedule_name=schedule_name,
                order_index=order_index,
            )
        )
    return sorted(shifted, key=lambda item: item.order_index)


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
    invalid_cpu_ids = [item.cpu_id for item in plan if item.cpu_id < 0]
    if invalid_cpu_ids:
        raise ValueError(f"cpu_id must be >= 0: {invalid_cpu_ids[0]}")
    task_ids = [item.task.task_id for item in plan]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("duplicate task_id in schedule plan")


def apply_cpu_affinity(cpu_id: int | None) -> bool:
    if cpu_id is None or not hasattr(os, "sched_setaffinity"):
        return False
    try:
        os.sched_setaffinity(0, {cpu_id})
    except (OSError, ValueError):
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
    affinity_applied = False
    envs: dict[int, Any] = {}
    task_counts = {item.task.task_id: 0 for item in items}
    current_item: ScheduleItem | None = None
    current_step_index: int | None = None
    current_phase = "affinity"
    try:
        affinity_applied = apply_cpu_affinity(cpu_id)
        current_phase = "env_init"
        for item in items:
            current_item = item
            current_step_index = task_counts[item.task.task_id]
            if item.task.task_id not in envs:
                envs[item.task.task_id] = env_factory(item)
        current_item = None
        current_phase = "command_wait"
        result_queue.put(
            {
                "event": "ready",
                "core_index": core_index,
                "cpu_id": cpu_id,
                "cpu_affinity_applied": affinity_applied,
            }
        )
        while True:
            command = command_queue.get()
            if command == "stop":
                break
            round_index, item = command
            current_item = item
            env = envs[item.task.task_id]
            task_step_index = task_counts[item.task.task_id]
            current_step_index = task_step_index
            current_phase = "step"
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
            current_phase = "command_wait"
    except Exception as exc:
        error = {
            "event": "error",
            "phase": current_phase,
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


def _command_diagnostic(command: tuple[int, ScheduleItem]) -> dict[str, Any]:
    core_index, item = command
    return {
        "core_index": core_index,
        "cpu_id": item.cpu_id,
        "task_id": item.task.task_id,
        "task_name": item.task.task_name,
    }


def _process_exitcode_diagnostics(
    processes: list[tuple[int, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "pid": process.pid,
            "core_index": core_index,
            "exitcode": process.exitcode,
            "is_alive": process.is_alive(),
        }
        for core_index, process in processes
    ]


def _close_mp_queue(mp_queue: Any) -> None:
    close = getattr(mp_queue, "close", None)
    if close is not None:
        try:
            close()
        except Exception:
            pass
    join_thread = getattr(mp_queue, "join_thread", None)
    if join_thread is not None:
        try:
            join_thread()
        except Exception:
            pass


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


def _worker_startup_diagnostic(core_index: int, items: list[ScheduleItem]) -> dict[str, Any]:
    return {
        "core_index": core_index,
        "cpu_id": items[0].cpu_id if items else None,
        "task_ids": [item.task.task_id for item in items],
        "task_names": [item.task.task_name for item in items],
    }


def _startup_timeout_error(
    *,
    schedule_name: str,
    pending_workers: list[dict[str, Any]],
    processes: list[tuple[int, Any]],
    startup_timeout_s: float,
) -> dict[str, Any]:
    return {
        "event": "timeout",
        "phase": "startup",
        "schedule_name": schedule_name,
        "pending_workers": pending_workers,
        "process_exitcodes": _process_exitcode_diagnostics(processes),
        "error_type": "TimeoutError",
        "error": (
            "timed out waiting for process worker startup "
            f"after {startup_timeout_s:.3f}s"
        ),
    }


def run_schedule_with_process_workers(
    plan: list[ScheduleItem],
    *,
    steps_per_env: int,
    env_factory: Any,
    dummy_action: list[float],
    subprocess_timeout_s: float = 300.0,
    startup_timeout_s: float | None = None,
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
    processes: list[tuple[int, Any]] = []
    events: list[StepEvent] = []
    startup_timeout_s = subprocess_timeout_s if startup_timeout_s is None else startup_timeout_s
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
            processes.append((core_index, process))

        ready_cores: set[int] = set()
        startup_pending_workers = [
            _worker_startup_diagnostic(core_index, items)
            for core_index, items in sorted(grouped.items())
        ]
        startup_deadline = time.monotonic() + startup_timeout_s
        while len(ready_cores) < len(grouped):
            remaining_s = max(startup_deadline - time.monotonic(), 0.0)
            if remaining_s <= 0.0:
                return ProcessRunResult(
                    events=events,
                    errors=[
                        _startup_timeout_error(
                            schedule_name=schedule_name,
                            pending_workers=[
                                worker
                                for worker in startup_pending_workers
                                if worker["core_index"] not in ready_cores
                            ],
                            processes=processes,
                            startup_timeout_s=startup_timeout_s,
                        )
                    ],
                )
            try:
                result = result_queue.get(timeout=remaining_s)
            except queue.Empty:
                return ProcessRunResult(
                    events=events,
                    errors=[
                        _startup_timeout_error(
                            schedule_name=schedule_name,
                            pending_workers=[
                                worker
                                for worker in startup_pending_workers
                                if worker["core_index"] not in ready_cores
                            ],
                            processes=processes,
                            startup_timeout_s=startup_timeout_s,
                        )
                    ],
                )
            event_type = result.get("event")
            if event_type == "ready":
                ready_cores.add(int(result["core_index"]))
                continue
            if event_type == "error":
                result.setdefault("schedule_name", schedule_name)
                return ProcessRunResult(events=events, errors=[result])
            return ProcessRunResult(
                events=events,
                errors=[
                    {
                        "event": "error",
                        "phase": "startup",
                        "schedule_name": schedule_name,
                        "error_type": "ValueError",
                        "error": f"unexpected worker result during startup: {result!r}",
                    }
                ],
            )

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
            pending_commands_by_core = {
                core_index: _command_diagnostic((core_index, item))
                for core_index, item in commands
            }
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
                                "pending_commands": list(pending_commands_by_core.values()),
                                "process_exitcodes": _process_exitcode_diagnostics(processes),
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
                pending_commands_by_core.pop(int(result["core_index"]), None)

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
            try:
                command_queue.put("stop")
            except Exception:
                pass
        for _, process in processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        for command_queue in command_queues.values():
            _close_mp_queue(command_queue)
        _close_mp_queue(result_queue)


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
        startup_timeout_s: float | None = None,
    ) -> None:
        self.steps_per_env = steps_per_env
        self.step_fn = step_fn
        self.env_factory = env_factory
        self.dummy_action = dummy_action
        self.subprocess_timeout_s = subprocess_timeout_s
        self.startup_timeout_s = startup_timeout_s

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
                    startup_timeout_s=self.startup_timeout_s,
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


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("integer list must not be empty")
    return [int(item) for item in items]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_to_jsonable(value), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_selected_tasks(path: Path, records: list[TaskRecord]) -> None:
    fixed_fieldnames = [
        "task_id",
        "task_name",
        "mean_latency_ms",
        "njnt",
        "ngeom",
        "estimated_latency_score",
    ]
    optional_fieldnames = sorted({key for record in records for key in record.extra})
    fieldnames = fixed_fieldnames + optional_fieldnames
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: getattr(record, key) for key in fixed_fieldnames}
            row.update({key: record.extra.get(key, "") for key in optional_fieldnames})
            writer.writerow(row)


def write_schedule_plan(path: Path, plan: list[ScheduleItem]) -> None:
    fieldnames = [
        "schedule_name",
        "order_index",
        "core_index",
        "cpu_id",
        "layer_index",
        "side",
        "task_id",
        "task_name",
        "estimated_latency_score",
        "mean_latency_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in plan:
            writer.writerow(
                {
                    "schedule_name": item.schedule_name,
                    "order_index": item.order_index,
                    "core_index": item.core_index,
                    "cpu_id": item.cpu_id,
                    "layer_index": item.layer_index,
                    "side": item.side,
                    "task_id": item.task.task_id,
                    "task_name": item.task.task_name,
                    "estimated_latency_score": item.task.estimated_latency_score,
                    "mean_latency_ms": item.task.mean_latency_ms,
                }
            )


def write_step_events(path: Path, events: list[StepEvent]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(_to_jsonable(event), sort_keys=True) + "\n")


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for summary in summaries for key in summary})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _is_random_baseline_name(schedule_name: str) -> bool:
    return schedule_name == RANDOM_BASELINE or schedule_name.startswith(
        f"{RANDOM_BASELINE}_"
    )


def compute_comparison_metrics(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(
        (
            summary
            for summary in summaries
            if summary.get("schedule_name") == TASK_ID_BASELINE
        ),
        None,
    )
    baseline_sps = _as_optional_float(
        baseline.get("steps_per_second") if baseline is not None else None
    )
    baseline_idle = _as_optional_float(
        baseline.get("mean_core_idle_ratio") if baseline is not None else None
    )
    baseline_comparison: dict[str, dict[str, float | None]] = {}
    for summary in summaries:
        schedule_name = str(summary.get("schedule_name", ""))
        if schedule_name == TASK_ID_BASELINE:
            continue
        steps_per_second = _as_optional_float(summary.get("steps_per_second"))
        idle_ratio = _as_optional_float(summary.get("mean_core_idle_ratio"))
        bubble_reduction = None
        if (
            baseline_idle is not None
            and baseline_idle != 0.0
            and idle_ratio is not None
        ):
            bubble_reduction = (baseline_idle - idle_ratio) / baseline_idle
        baseline_comparison[schedule_name] = {
            "speedup_vs_task_id_baseline": _safe_ratio(
                steps_per_second,
                baseline_sps,
            ),
            "bubble_reduction_vs_task_id_baseline": bubble_reduction,
        }

    random_summaries = [
        summary
        for summary in summaries
        if _is_random_baseline_name(str(summary.get("schedule_name", "")))
    ]
    random_steps_per_second = [
        value
        for value in (
            _as_optional_float(summary.get("steps_per_second"))
            for summary in random_summaries
        )
        if value is not None
    ]
    random_idle_ratios = [
        value
        for value in (
            _as_optional_float(summary.get("mean_core_idle_ratio"))
            for summary in random_summaries
        )
        if value is not None
    ]
    random_aggregate = None
    if random_summaries:
        random_aggregate = {
            "count": len(random_summaries),
            "mean_steps_per_second": (
                float(np.mean(np.asarray(random_steps_per_second)))
                if random_steps_per_second
                else None
            ),
            "median_steps_per_second": (
                float(np.median(np.asarray(random_steps_per_second)))
                if random_steps_per_second
                else None
            ),
            "best_steps_per_second": (
                max(random_steps_per_second) if random_steps_per_second else None
            ),
            "worst_steps_per_second": (
                min(random_steps_per_second) if random_steps_per_second else None
            ),
            "mean_core_idle_ratio": (
                float(np.mean(np.asarray(random_idle_ratios)))
                if random_idle_ratios
                else None
            ),
        }
    return {
        "baseline_comparison": baseline_comparison,
        "random_aggregate": random_aggregate,
    }


def add_comparison_metrics_to_summaries(summaries: list[dict[str, Any]]) -> None:
    comparison = compute_comparison_metrics(summaries)["baseline_comparison"]
    for summary in summaries:
        schedule_metrics = comparison.get(str(summary.get("schedule_name", "")))
        if schedule_metrics is None:
            continue
        summary.update(schedule_metrics)


def _format_optional_float(value: Any) -> str:
    number = _as_optional_float(value)
    if number is None:
        return ""
    return f"{number:.6f}"


def _task_ids_for_side(items: list[ScheduleItem], side: str) -> str:
    task_ids = [
        str(item.task.task_id)
        for item in sorted(items, key=lambda value: value.order_index)
        if item.side == side
    ]
    return ", ".join(task_ids)


def _append_trapezoid_mapping_evidence(
    lines: list[str],
    plans: dict[str, list[ScheduleItem]] | None,
) -> None:
    if not plans or TRAPEZOID_PIPELINE not in plans:
        return
    lines.extend(["", "## Trapezoid Mapping Evidence", ""])
    lines.append("| core_index | cpu_id | long task ids | short task ids |")
    lines.append("|---:|---:|---|---|")
    grouped = _items_by_core(plans[TRAPEZOID_PIPELINE])
    for core_index, items in sorted(grouped.items()):
        cpu_id = items[0].cpu_id if items else ""
        lines.append(
            "| {core_index} | {cpu_id} | {long_tasks} | {short_tasks} |".format(
                core_index=core_index,
                cpu_id=cpu_id,
                long_tasks=_task_ids_for_side(items, "long"),
                short_tasks=_task_ids_for_side(items, "short"),
            )
        )


def write_comparison_report(
    path: Path,
    summaries: list[dict[str, Any]],
    *,
    plans: dict[str, list[ScheduleItem]] | None = None,
) -> None:
    metrics = compute_comparison_metrics(summaries)
    lines = ["# LIBERO Latency Schedule Benchmark", "", "## Raw Schedule Metrics", ""]
    lines.append("| schedule | status | steps/sec | idle ratio |")
    lines.append("|---|---:|---:|---:|")
    for summary in summaries:
        lines.append(
            "| {schedule} | {status} | {sps:.6f} | {idle} |".format(
                schedule=summary["schedule_name"],
                status=summary["status"],
                sps=float(summary.get("steps_per_second") or 0.0),
                idle=summary.get("mean_core_idle_ratio"),
            )
        )
    lines.extend(["", "## Baseline Comparison", ""])
    lines.append(
        "| schedule | speedup_vs_task_id_baseline | "
        "bubble_reduction_vs_task_id_baseline |"
    )
    lines.append("|---|---:|---:|")
    for schedule_name, values in metrics["baseline_comparison"].items():
        lines.append(
            "| {schedule} | {speedup} | {bubble} |".format(
                schedule=schedule_name,
                speedup=_format_optional_float(
                    values.get("speedup_vs_task_id_baseline")
                ),
                bubble=_format_optional_float(
                    values.get("bubble_reduction_vs_task_id_baseline")
                ),
            )
        )
    random_aggregate = metrics["random_aggregate"]
    if random_aggregate is not None:
        lines.extend(["", "## Random Baseline Aggregate", ""])
        lines.append(
            "| count | mean steps/sec | median steps/sec | best steps/sec | "
            "worst steps/sec | mean idle ratio |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|")
        lines.append(
            "| {count} | {mean_sps} | {median_sps} | {best_sps} | {worst_sps} | "
            "{mean_idle} |".format(
                count=random_aggregate["count"],
                mean_sps=_format_optional_float(
                    random_aggregate["mean_steps_per_second"]
                ),
                median_sps=_format_optional_float(
                    random_aggregate["median_steps_per_second"]
                ),
                best_sps=_format_optional_float(
                    random_aggregate["best_steps_per_second"]
                ),
                worst_sps=_format_optional_float(
                    random_aggregate["worst_steps_per_second"]
                ),
                mean_idle=_format_optional_float(
                    random_aggregate["mean_core_idle_ratio"]
                ),
            )
        )
    _append_trapezoid_mapping_evidence(lines, plans)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-csv", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-ids", required=True)
    parser.add_argument("--steps-per-env", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-baseline-repeats", type=int, default=1)
    parser.add_argument("--suite", default="libero_90")
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument(
        "--libero-type",
        choices=["standard", "pro", "plus"],
        default="standard",
    )
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--subprocess-timeout-s", type=float, default=300.0)
    parser.add_argument("--startup-timeout-s", type=float, default=None)
    parser.add_argument("--dummy-action", default="0,0,0,0,0,0,-1")
    parser.add_argument(
        "--fake-latency-from-csv",
        action="store_true",
        help="Use CSV mean_latency_ms as fake latency for unit/local smoke tests.",
    )
    return parser


def _dummy_action_from_arg(value: str) -> list[float]:
    try:
        action = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--dummy-action must be a comma-separated list of floats") from exc
    if not action:
        raise ValueError("--dummy-action must not be empty")
    return action


def _fake_step_fn(item: ScheduleItem, step_index: int) -> float:
    del step_index
    return item.task.mean_latency_ms / 1000.0


def _parse_cpu_ids_or_exit(parser: argparse.ArgumentParser, value: str) -> list[int]:
    try:
        cpu_ids = parse_int_list(value)
    except ValueError as exc:
        parser.error(f"--cpu-ids: {exc}")
    invalid_cpu_ids = [cpu_id for cpu_id in cpu_ids if cpu_id < 0]
    if invalid_cpu_ids:
        parser.error(f"--cpu-ids must be >= 0: {invalid_cpu_ids[0]}")
    if len(set(cpu_ids)) != len(cpu_ids):
        parser.error("--cpu-ids must not contain duplicates")
    return cpu_ids


def _dummy_action_or_exit(parser: argparse.ArgumentParser, value: str) -> list[float]:
    try:
        return _dummy_action_from_arg(value)
    except ValueError as exc:
        parser.error(str(exc))


def _validate_cli_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.num_envs < 1:
        parser.error("--num-envs must be >= 1")
    if args.num_envs % 2 != 0:
        parser.error("--num-envs must be even for trapezoid_pipeline")
    if args.steps_per_env < 1:
        parser.error("--steps-per-env must be >= 1")
    if args.random_baseline_repeats < 0:
        parser.error("--random-baseline-repeats must be >= 0")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be >= 0")
    if args.subprocess_timeout_s <= 0.0:
        parser.error("--subprocess-timeout-s must be > 0")
    if args.startup_timeout_s is not None and args.startup_timeout_s <= 0.0:
        parser.error("--startup-timeout-s must be > 0")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _validate_cli_args(parser, args)
    cpu_ids = _parse_cpu_ids_or_exit(parser, args.cpu_ids)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    errors_path = output_dir / "errors.jsonl"
    errors_path.unlink(missing_ok=True)
    records = estimate_latency_scores(
        sample_task_records(
            load_task_records(args.task_csv),
            num_envs=args.num_envs,
            seed=args.seed,
        )
    )
    run_config = {
        key: _to_jsonable(value) for key, value in vars(args).items()
    }
    run_config["cpu_ids"] = cpu_ids
    write_json(output_dir / "run_config.json", run_config)
    write_selected_tasks(output_dir / "selected_tasks.csv", records)

    plans: list[list[ScheduleItem]] = [
        build_task_id_baseline_plan(records, cpu_ids=cpu_ids),
        build_trapezoid_pipeline_plan(records, cpu_ids=cpu_ids),
        build_phase_shifted_trapezoid_plan(records, cpu_ids=cpu_ids),
    ]
    for repeat in range(args.random_baseline_repeats):
        plans.append(
            build_random_baseline_plan(
                records,
                cpu_ids=cpu_ids,
                seed=args.seed + repeat + 1,
            )
        )

    if args.fake_latency_from_csv:
        runner = BenchmarkRunner(
            steps_per_env=args.steps_per_env,
            step_fn=_fake_step_fn,
        )
    else:
        dummy_action = _dummy_action_or_exit(parser, args.dummy_action)
        env_factory = make_libero_env_factory(
            suite=args.suite,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            libero_type=args.libero_type,
            seed=args.seed,
            warmup_steps=args.warmup_steps,
            dummy_action=dummy_action,
        )
        runner = BenchmarkRunner(
            steps_per_env=args.steps_per_env,
            env_factory=env_factory,
            dummy_action=dummy_action,
            subprocess_timeout_s=args.subprocess_timeout_s,
            startup_timeout_s=args.startup_timeout_s,
        )

    summaries = []
    errors = []
    plans_by_name: dict[str, list[ScheduleItem]] = {}
    for plan in plans:
        schedule_name = plan[0].schedule_name
        if schedule_name == RANDOM_BASELINE and args.random_baseline_repeats > 1:
            random_index = sum(
                1
                for summary in summaries
                if summary["schedule_name"].startswith(RANDOM_BASELINE)
            )
            schedule_name = f"{RANDOM_BASELINE}_{random_index}"
            plan = [replace(item, schedule_name=schedule_name) for item in plan]
        plans_by_name[schedule_name] = plan
        write_schedule_plan(output_dir / f"schedule_plan_{schedule_name}.csv", plan)
        result = runner.run(schedule_name, plan)
        write_step_events(output_dir / f"step_events_{schedule_name}.jsonl", result.events)
        summaries.append(result.summary)
        errors.extend(result.errors)

    add_comparison_metrics_to_summaries(summaries)
    write_summary_csv(output_dir / "schedule_summary.csv", summaries)
    write_json(output_dir / "schedule_summary.json", summaries)
    write_comparison_report(
        output_dir / "comparison_report.md",
        summaries,
        plans=plans_by_name,
    )
    if errors:
        with errors_path.open("w", encoding="utf-8") as handle:
            for error in errors:
                handle.write(json.dumps(_to_jsonable(error), sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
