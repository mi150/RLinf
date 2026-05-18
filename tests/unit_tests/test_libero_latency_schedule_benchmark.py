import csv
from pathlib import Path

import numpy as np
import pytest

from toolkits.run_libero_latency_schedule_benchmark import (
    TaskRecord,
    estimate_latency_scores,
    load_task_records,
    sample_task_records,
)


def _write_task_csv(path: Path) -> None:
    rows = [
        {
            "task_id": "3",
            "task_name": "task_c",
            "mean_latency_ms": "13.0",
            "njnt": "12",
            "ngeom": "100",
            "scene_type": "study",
        },
        {
            "task_id": "1",
            "task_name": "task_a",
            "mean_latency_ms": "21.0",
            "njnt": "17",
            "ngeom": "200",
            "scene_type": "kitchen",
        },
        {
            "task_id": "2",
            "task_name": "task_b",
            "mean_latency_ms": "17.0",
            "njnt": "15",
            "ngeom": "150",
            "scene_type": "living_room",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_load_task_records_preserves_required_and_extra_fields(tmp_path: Path):
    csv_path = tmp_path / "tasks.csv"
    _write_task_csv(csv_path)

    records = load_task_records(csv_path)

    assert [record.task_id for record in records] == [3, 1, 2]
    assert records[1].task_name == "task_a"
    assert records[1].mean_latency_ms == 21.0
    assert records[1].njnt == 17
    assert records[1].ngeom == 200
    assert records[1].extra["scene_type"] == "kitchen"


def test_load_task_records_rejects_missing_required_columns(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("task_id,task_name,njnt,ngeom\n1,t,2,3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_task_records(csv_path)


def test_sample_task_records_is_seeded_without_replacement(tmp_path: Path):
    csv_path = tmp_path / "tasks.csv"
    _write_task_csv(csv_path)
    records = load_task_records(csv_path)

    first = sample_task_records(records, num_envs=2, seed=123)
    second = sample_task_records(records, num_envs=2, seed=123)

    assert [record.task_id for record in first] == [record.task_id for record in second]
    assert len({record.task_id for record in first}) == 2


def test_sample_task_records_rejects_oversized_request(tmp_path: Path):
    csv_path = tmp_path / "tasks.csv"
    _write_task_csv(csv_path)
    records = load_task_records(csv_path)

    with pytest.raises(ValueError, match="num_envs"):
        sample_task_records(records, num_envs=4, seed=0)


def test_estimate_latency_scores_uses_z_scored_njnt_and_ngeom():
    records = [
        TaskRecord(task_id=0, task_name="low_jnt", mean_latency_ms=1.0, njnt=10, ngeom=300),
        TaskRecord(task_id=1, task_name="mid", mean_latency_ms=2.0, njnt=20, ngeom=100),
        TaskRecord(task_id=2, task_name="high_jnt", mean_latency_ms=3.0, njnt=30, ngeom=200),
    ]

    scored = estimate_latency_scores(records, weight_jnt=0.25, weight_geom=0.75)

    njnt = np.asarray([record.njnt for record in records], dtype=np.float64)
    ngeom = np.asarray([record.ngeom for record in records], dtype=np.float64)
    expected = 0.25 * ((njnt - np.mean(njnt)) / np.std(njnt)) + 0.75 * (
        (ngeom - np.mean(ngeom)) / np.std(ngeom)
    )
    np.testing.assert_allclose(
        [record.estimated_latency_score for record in scored],
        expected,
    )
