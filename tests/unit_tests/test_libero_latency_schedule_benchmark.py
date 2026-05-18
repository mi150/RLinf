import csv
from pathlib import Path

import numpy as np
import pytest

from toolkits.run_libero_latency_schedule_benchmark import (
    StepEvent,
    TaskRecord,
    compute_schedule_summary,
    build_random_baseline_plan,
    build_task_id_baseline_plan,
    build_trapezoid_pipeline_plan,
    estimate_latency_scores,
    load_task_records,
    sample_task_records,
    run_schedule_with_step_function,
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


def _records_for_schedule() -> list[TaskRecord]:
    return [
        TaskRecord(
            task_id=4,
            task_name="t4",
            mean_latency_ms=4.0,
            njnt=14,
            ngeom=140,
            estimated_latency_score=4.0,
        ),
        TaskRecord(
            task_id=1,
            task_name="t1",
            mean_latency_ms=1.0,
            njnt=11,
            ngeom=110,
            estimated_latency_score=1.0,
        ),
        TaskRecord(
            task_id=3,
            task_name="t3",
            mean_latency_ms=3.0,
            njnt=13,
            ngeom=130,
            estimated_latency_score=3.0,
        ),
        TaskRecord(
            task_id=2,
            task_name="t2",
            mean_latency_ms=2.0,
            njnt=12,
            ngeom=120,
            estimated_latency_score=2.0,
        ),
    ]


def test_task_id_baseline_assigns_sorted_tasks_to_core_columns():
    plan = build_task_id_baseline_plan(_records_for_schedule(), cpu_ids=[10, 11])

    assert [item.task.task_id for item in plan] == [1, 2, 3, 4]
    assert [(item.core_index, item.cpu_id, item.layer_index) for item in plan] == [
        (0, 10, 0),
        (1, 11, 0),
        (0, 10, 1),
        (1, 11, 1),
    ]
    assert {item.schedule_name for item in plan} == {"task_id_baseline"}


def test_random_baseline_is_seeded_and_static():
    first = build_random_baseline_plan(_records_for_schedule(), cpu_ids=[0, 1], seed=7)
    second = build_random_baseline_plan(_records_for_schedule(), cpu_ids=[0, 1], seed=7)

    assert [item.task.task_id for item in first] == [item.task.task_id for item in second]
    assert [item.task.task_id for item in first] == [2, 1, 4, 3]
    assert [(item.core_index, item.cpu_id, item.layer_index) for item in first] == [
        (0, 0, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 1, 1),
    ]
    assert {item.schedule_name for item in first} == {"random_baseline"}


def test_trapezoid_pipeline_keeps_long_short_pairs_on_same_core():
    plan = build_trapezoid_pipeline_plan(_records_for_schedule(), cpu_ids=[20, 21])

    long_items = [item for item in plan if item.side == "long"]
    short_items = [item for item in plan if item.side == "short"]
    long_by_core = {item.core_index: item for item in long_items}
    short_by_core = {item.core_index: item for item in short_items}

    assert {
        core_index: (long_by_core[core_index].task.task_id, short_by_core[core_index].task.task_id)
        for core_index in sorted(long_by_core)
    } == {
        0: (4, 1),
        1: (3, 2),
    }
    assert [
        (item.core_index, item.cpu_id, item.layer_index, item.order_index)
        for item in long_items
    ] == [
        (0, 20, 0, 0),
        (1, 21, 0, 1),
    ]
    assert [
        (item.core_index, item.cpu_id, item.layer_index, item.order_index)
        for item in short_items
    ] == [
        (0, 20, 0, 2),
        (1, 21, 0, 3),
    ]
    assert {item.schedule_name for item in plan} == {"trapezoid_pipeline"}


def test_trapezoid_pipeline_rejects_odd_task_count():
    records = _records_for_schedule()[:3]

    with pytest.raises(ValueError, match="even"):
        build_trapezoid_pipeline_plan(records, cpu_ids=[0, 1])


def test_run_schedule_with_step_function_completes_equal_steps_per_task():
    records = _records_for_schedule()
    plan = build_task_id_baseline_plan(records, cpu_ids=[0, 1])
    latency_by_task = {1: 0.01, 2: 0.02, 3: 0.03, 4: 0.04}

    events = run_schedule_with_step_function(
        plan,
        steps_per_env=2,
        step_fn=lambda item, step_index: latency_by_task[item.task.task_id],
    )

    assert len(events) == 8
    counts = {}
    for event in events:
        counts[event.task_id] = counts.get(event.task_id, 0) + 1
    assert counts == {1: 2, 2: 2, 3: 2, 4: 2}
    assert all(event.round_wall_time_s >= event.latency_s for event in events)


def test_compute_schedule_summary_reports_throughput_and_idle():
    events = [
        StepEvent(
            schedule_name="s",
            round_index=0,
            core_index=0,
            cpu_id=0,
            task_id=1,
            task_name="a",
            task_step_index=0,
            latency_s=0.01,
            round_wall_time_s=0.02,
            idle_time_s=0.01,
            cpu_affinity_applied=True,
        ),
        StepEvent(
            schedule_name="s",
            round_index=0,
            core_index=1,
            cpu_id=1,
            task_id=2,
            task_name="b",
            task_step_index=0,
            latency_s=0.02,
            round_wall_time_s=0.02,
            idle_time_s=0.0,
            cpu_affinity_applied=True,
        ),
    ]

    summary = compute_schedule_summary("s", events)

    assert summary["schedule_name"] == "s"
    assert summary["status"] == "completed"
    assert summary["total_steps"] == 2
    assert summary["makespan_s"] == 0.02
    assert summary["steps_per_second"] == 100.0
    assert summary["mean_core_idle_ratio"] == 0.25
    assert summary["cpu_affinity_success_rate"] == 1.0


def test_compute_schedule_summary_marks_degraded_affinity():
    event = StepEvent(
        schedule_name="s",
        round_index=0,
        core_index=0,
        cpu_id=0,
        task_id=1,
        task_name="a",
        task_step_index=0,
        latency_s=0.01,
        round_wall_time_s=0.01,
        idle_time_s=0.0,
        cpu_affinity_applied=False,
    )

    summary = compute_schedule_summary("s", [event])

    assert summary["status"] == "degraded"
    assert summary["cpu_affinity_success_rate"] == 0.0
