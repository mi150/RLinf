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
