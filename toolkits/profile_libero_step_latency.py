"""Profile LIBERO simulator step latency without Ray or policy models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


@dataclass(frozen=True)
class ProfileConfig:
    suite: str
    task_ids: str
    trials_per_task: int
    specific_trial_ids: list[int] | None
    warmup_steps: int
    measure_steps: int
    cpu_id: int | None
    cpu_ids: list[int] | None
    camera_height: int
    camera_width: int
    libero_type: str
    seed: int
    output_dir: Path
    dummy_action: list[float]
    stop_on_done: bool


@dataclass(frozen=True)
class TaskTrialSpec:
    suite_name: str
    task_id: int
    trial_id: int
    task_name: str
    task_language: str
    bddl_file: str
    seed: int


def parse_int_list(value: str, *, allow_all: bool = False) -> list[int] | str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("empty integer list")
    if stripped.lower() == "all":
        if allow_all:
            return "all"
        raise ValueError("'all' is not allowed here")
    result: list[int] = []
    for part in stripped.split(","):
        item = part.strip()
        if not item:
            raise ValueError(f"empty item in integer list: {value!r}")
        result.append(int(item))
    return result


def parse_task_ids(value: str, *, num_tasks: int) -> list[int]:
    parsed = parse_int_list(value, allow_all=True)
    if parsed == "all":
        return list(range(num_tasks))
    task_ids = parsed
    for task_id in task_ids:
        if task_id < 0 or task_id >= num_tasks:
            raise ValueError(f"task id {task_id} out of range [0, {num_tasks})")
    return task_ids


def select_trial_ids(
    *,
    num_trials: int,
    trials_per_task: int,
    specific_trial_ids: list[int] | None,
    seed: int,
    task_id: int,
) -> list[int]:
    if num_trials <= 0:
        return []
    if specific_trial_ids is not None:
        for trial_id in specific_trial_ids:
            if trial_id < 0 or trial_id >= num_trials:
                raise ValueError(f"trial id {trial_id} out of range [0, {num_trials})")
        return list(specific_trial_ids)
    if trials_per_task >= num_trials:
        return list(range(num_trials))
    rng = np.random.default_rng(seed + task_id)
    selected = rng.choice(num_trials, size=trials_per_task, replace=False)
    return [int(item) for item in selected.tolist()]


def parse_dummy_action(value: str | None) -> list[float]:
    if value is None:
        return list(DEFAULT_DUMMY_ACTION)
    stripped = value.strip()
    if not stripped:
        raise ValueError("empty dummy action")
    return [float(part.strip()) for part in stripped.split(",")]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def compute_latency_summary(latencies: list[float]) -> dict[str, float | int | None]:
    if not latencies:
        return {
            "step_count": 0,
            "mean_latency_s": None,
            "median_latency_s": None,
            "p90_latency_s": None,
            "p95_latency_s": None,
            "p99_latency_s": None,
            "min_latency_s": None,
            "max_latency_s": None,
            "std_latency_s": None,
            "tail_ratio_p99_to_median": None,
        }
    array = np.asarray(latencies, dtype=np.float64)
    median = float(np.median(array))
    p99 = _percentile(latencies, 99)
    return {
        "step_count": int(array.size),
        "mean_latency_s": float(np.mean(array)),
        "median_latency_s": median,
        "p90_latency_s": _percentile(latencies, 90),
        "p95_latency_s": _percentile(latencies, 95),
        "p99_latency_s": p99,
        "min_latency_s": float(np.min(array)),
        "max_latency_s": float(np.max(array)),
        "std_latency_s": float(np.std(array)),
        "tail_ratio_p99_to_median": (
            None if median == 0.0 or p99 is None else float(p99 / median)
        ),
    }
