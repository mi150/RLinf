import csv
import json
import time
from pathlib import Path

import numpy as np
import pytest

from toolkits.run_libero_latency_schedule_benchmark import (
    BenchmarkRunner,
    ScheduleItem,
    StepEvent,
    TaskRecord,
    apply_cpu_affinity,
    build_phase_shifted_trapezoid_plan,
    build_random_baseline_plan,
    build_task_id_baseline_plan,
    build_trapezoid_pipeline_plan,
    build_worker_plans,
    compute_comparison_metrics,
    compute_schedule_summary,
    estimate_latency_scores,
    load_task_records,
    main,
    run_schedule_with_process_workers,
    run_schedule_with_step_function,
    sample_task_records,
    write_comparison_report,
    write_selected_tasks,
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


def test_write_selected_tasks_preserves_sorted_optional_columns(tmp_path: Path):
    path = tmp_path / "selected_tasks.csv"
    records = [
        TaskRecord(
            task_id=1,
            task_name="a",
            mean_latency_ms=1.0,
            njnt=2,
            ngeom=3,
            estimated_latency_score=0.5,
            extra={"scene_type": "kitchen", "suite": "libero_10"},
        ),
        TaskRecord(
            task_id=2,
            task_name="b",
            mean_latency_ms=4.0,
            njnt=5,
            ngeom=6,
            estimated_latency_score=0.6,
            extra={"difficulty": "hard"},
        ),
    ]

    write_selected_tasks(path, records)

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0].keys() == {
        "task_id",
        "task_name",
        "mean_latency_ms",
        "njnt",
        "ngeom",
        "estimated_latency_score",
        "difficulty",
        "scene_type",
        "suite",
    }
    assert rows[0]["scene_type"] == "kitchen"
    assert rows[0]["suite"] == "libero_10"
    assert rows[0]["difficulty"] == ""
    assert rows[1]["difficulty"] == "hard"


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


def _records_for_four_core_phase_shift() -> list[TaskRecord]:
    return [
        TaskRecord(
            task_id=10,
            task_name="long_100",
            mean_latency_ms=100.0,
            njnt=100,
            ngeom=100,
            estimated_latency_score=100.0,
        ),
        TaskRecord(
            task_id=11,
            task_name="long_90",
            mean_latency_ms=90.0,
            njnt=90,
            ngeom=90,
            estimated_latency_score=90.0,
        ),
        TaskRecord(
            task_id=12,
            task_name="long_80",
            mean_latency_ms=80.0,
            njnt=80,
            ngeom=80,
            estimated_latency_score=80.0,
        ),
        TaskRecord(
            task_id=13,
            task_name="long_70",
            mean_latency_ms=70.0,
            njnt=70,
            ngeom=70,
            estimated_latency_score=70.0,
        ),
        TaskRecord(
            task_id=14,
            task_name="short_60",
            mean_latency_ms=60.0,
            njnt=60,
            ngeom=60,
            estimated_latency_score=60.0,
        ),
        TaskRecord(
            task_id=15,
            task_name="short_50",
            mean_latency_ms=50.0,
            njnt=50,
            ngeom=50,
            estimated_latency_score=50.0,
        ),
        TaskRecord(
            task_id=16,
            task_name="short_40",
            mean_latency_ms=40.0,
            njnt=40,
            ngeom=40,
            estimated_latency_score=40.0,
        ),
        TaskRecord(
            task_id=17,
            task_name="short_30",
            mean_latency_ms=30.0,
            njnt=30,
            ngeom=30,
            estimated_latency_score=30.0,
        ),
    ]


class FakeProcessEnv:
    def __init__(self, latency_s: float):
        self.latency_s = latency_s

    def step(self, action):
        del action
        return {}, 0.0, False, {}


class FailingProcessEnv:
    def step(self, action):
        del action
        raise RuntimeError("process step failed")


class SleepingProcessEnv:
    def __init__(self, sleep_s: float):
        self.sleep_s = sleep_s

    def step(self, action):
        del action
        time.sleep(self.sleep_s)
        return {}, 0.0, False, {}


class SlowInitProcessEnv:
    def __init__(self, sleep_s: float):
        time.sleep(sleep_s)

    def step(self, action):
        del action
        return {}, 0.0, False, {}


def fake_process_env_factory(item: ScheduleItem):
    return FakeProcessEnv(latency_s=item.task.mean_latency_ms / 1000.0)


def failing_process_env_factory(item: ScheduleItem):
    del item
    return FailingProcessEnv()


def failing_process_env_init_factory(item: ScheduleItem):
    raise RuntimeError(f"env init failed for task {item.task.task_id}")


def sleeping_process_env_factory(item: ScheduleItem):
    del item
    return SleepingProcessEnv(sleep_s=0.2)


def slow_init_process_env_factory(item: ScheduleItem):
    del item
    return SlowInitProcessEnv(sleep_s=0.2)


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


def test_phase_shifted_trapezoid_staggers_long_and_short_rounds_by_core():
    records = _records_for_four_core_phase_shift()
    trapezoid_plan = build_trapezoid_pipeline_plan(records, cpu_ids=[20, 21, 22, 23])
    plan = build_phase_shifted_trapezoid_plan(records, cpu_ids=[20, 21, 22, 23])
    side_by_task = {item.task.task_id: item.side for item in plan}

    long_items = [item for item in plan if item.side == "long"]
    short_items = [item for item in plan if item.side == "short"]
    long_by_core = {item.core_index: item for item in long_items}
    short_by_core = {item.core_index: item for item in short_items}

    assert {
        core_index: (long_by_core[core_index].task.task_id, short_by_core[core_index].task.task_id)
        for core_index in sorted(long_by_core)
    } == {
        0: (10, 17),
        1: (11, 16),
        2: (12, 15),
        3: (13, 14),
    }
    assert {
        (item.task.task_id, item.core_index, item.cpu_id, item.layer_index, item.side)
        for item in plan
    } == {
        (item.task.task_id, item.core_index, item.cpu_id, item.layer_index, item.side)
        for item in trapezoid_plan
    }

    events = run_schedule_with_step_function(
        plan,
        steps_per_env=1,
        step_fn=lambda item, step_index: item.task.estimated_latency_score / 1000.0,
    )
    trapezoid_events = run_schedule_with_step_function(
        trapezoid_plan,
        steps_per_env=1,
        step_fn=lambda item, step_index: item.task.estimated_latency_score / 1000.0,
    )
    round0_sides = {
        event.core_index: side_by_task[event.task_id]
        for event in events
        if event.round_index == 0
    }
    round0_task_ids = {event.task_id for event in events if event.round_index == 0}
    phase_round_wall_times = {
        round_index: max(event.latency_s for event in events if event.round_index == round_index)
        for round_index in [0, 1]
    }
    trapezoid_round_wall_times = {
        round_index: max(
            event.latency_s
            for event in trapezoid_events
            if event.round_index == round_index
        )
        for round_index in [0, 1]
    }

    assert round0_sides == {0: "short", 1: "short", 2: "long", 3: "long"}
    assert not {10, 11}.issubset(round0_task_ids)
    assert abs(phase_round_wall_times[0] - phase_round_wall_times[1]) < abs(
        trapezoid_round_wall_times[0] - trapezoid_round_wall_times[1]
    )
    assert {item.schedule_name for item in plan} == {"phase_shifted_trapezoid"}


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


def test_run_schedule_with_step_function_rejects_empty_plan():
    with pytest.raises(ValueError, match="plan must not be empty"):
        run_schedule_with_step_function([], steps_per_env=1, step_fn=lambda item, step_index: 0.0)


def test_run_schedule_with_step_function_rejects_duplicate_task_ids():
    records = _records_for_schedule()
    plan = build_task_id_baseline_plan(records, cpu_ids=[0, 1])
    duplicate_plan = plan + [plan[0]]

    with pytest.raises(ValueError, match="duplicate task_id"):
        run_schedule_with_step_function(
            duplicate_plan,
            steps_per_env=1,
            step_fn=lambda item, step_index: 0.0,
        )


def test_run_schedule_with_step_function_rejects_invalid_latency_values():
    records = _records_for_schedule()
    plan = build_task_id_baseline_plan(records, cpu_ids=[0, 1])

    with pytest.raises(ValueError, match="invalid latency"):
        run_schedule_with_step_function(
            plan,
            steps_per_env=1,
            step_fn=lambda item, step_index: -0.1 if item.task.task_id == 1 else 0.0,
        )

    with pytest.raises(ValueError, match="invalid latency"):
        run_schedule_with_step_function(
            plan,
            steps_per_env=1,
            step_fn=lambda item, step_index: float("nan") if item.task.task_id == 1 else 0.0,
        )


def test_build_worker_plans_groups_items_by_core():
    records = _records_for_schedule()
    plan = build_task_id_baseline_plan(records, cpu_ids=[10, 11])

    worker_plans = build_worker_plans(plan)

    assert sorted(worker_plans) == [0, 1]
    assert [item.cpu_id for item in worker_plans[0]] == [10, 10]
    assert [item.cpu_id for item in worker_plans[1]] == [11, 11]


def test_run_schedule_with_process_workers_completes_equal_steps_per_task():
    records = _records_for_schedule()
    plan = build_task_id_baseline_plan(records, cpu_ids=[0, 1])

    result = run_schedule_with_process_workers(
        plan,
        steps_per_env=1,
        env_factory=fake_process_env_factory,
        dummy_action=[0.0] * 7,
        subprocess_timeout_s=10.0,
    )

    assert result.errors == []
    assert len(result.events) == 4
    assert {event.task_id for event in result.events} == {1, 2, 3, 4}
    assert all(event.round_wall_time_s >= event.latency_s for event in result.events)


def test_run_schedule_with_process_workers_reports_step_error_context():
    records = _records_for_schedule()
    plan = build_task_id_baseline_plan(records, cpu_ids=[0])

    result = run_schedule_with_process_workers(
        plan,
        steps_per_env=1,
        env_factory=failing_process_env_factory,
        dummy_action=[0.0] * 7,
        subprocess_timeout_s=10.0,
    )

    assert result.events == []
    assert len(result.errors) == 1
    error = result.errors[0]
    assert error["event"] == "error"
    assert error["core_index"] == 0
    assert error["cpu_id"] == 0
    assert error["task_id"] == 1
    assert error["task_name"] == "t1"
    assert error["task_step_index"] == 0
    assert error["error_type"] == "RuntimeError"
    assert "process step failed" in error["error"]
    assert "Traceback" in error["traceback"]


def test_run_schedule_with_process_workers_reports_env_init_error_context():
    records = _records_for_schedule()
    plan = build_task_id_baseline_plan(records, cpu_ids=[0])

    result = run_schedule_with_process_workers(
        plan,
        steps_per_env=1,
        env_factory=failing_process_env_init_factory,
        dummy_action=[0.0] * 7,
        subprocess_timeout_s=10.0,
    )

    assert result.events == []
    assert len(result.errors) == 1
    error = result.errors[0]
    assert error["event"] == "error"
    assert error["phase"] == "env_init"
    assert error["core_index"] == 0
    assert error["cpu_id"] == 0
    assert error["task_id"] == 1
    assert error["task_name"] == "t1"
    assert error["task_step_index"] == 0
    assert error["error_type"] == "RuntimeError"
    assert "env init failed for task 1" in error["error"]
    assert "Traceback" in error["traceback"]


def test_run_schedule_with_process_workers_waits_for_startup_before_step_timeout():
    records = _records_for_schedule()
    plan = build_task_id_baseline_plan(records, cpu_ids=[0, 1])

    result = run_schedule_with_process_workers(
        plan,
        steps_per_env=1,
        env_factory=slow_init_process_env_factory,
        dummy_action=[0.0] * 7,
        subprocess_timeout_s=0.01,
        startup_timeout_s=1.0,
    )

    assert result.errors == []
    assert len(result.events) == 4
    assert {event.task_id for event in result.events} == {1, 2, 3, 4}


def test_run_schedule_with_process_workers_reports_startup_timeout_diagnostics():
    records = _records_for_schedule()
    plan = build_task_id_baseline_plan(records, cpu_ids=[0, 1])

    result = run_schedule_with_process_workers(
        plan,
        steps_per_env=1,
        env_factory=slow_init_process_env_factory,
        dummy_action=[0.0] * 7,
        subprocess_timeout_s=10.0,
        startup_timeout_s=0.01,
    )

    assert result.events == []
    assert len(result.errors) == 1
    error = result.errors[0]
    assert error["event"] == "timeout"
    assert error["phase"] == "startup"
    assert error["schedule_name"] == "task_id_baseline"
    assert error["pending_workers"]
    assert {
        "core_index",
        "cpu_id",
        "task_ids",
        "task_names",
    } <= set(error["pending_workers"][0])
    assert error["process_exitcodes"]
    assert error["error_type"] == "TimeoutError"
    assert "startup" in error["error"]


def test_run_schedule_with_process_workers_timeout_includes_diagnostics():
    records = _records_for_schedule()
    plan = build_task_id_baseline_plan(records, cpu_ids=[0, 1])

    result = run_schedule_with_process_workers(
        plan,
        steps_per_env=1,
        env_factory=sleeping_process_env_factory,
        dummy_action=[0.0] * 7,
        subprocess_timeout_s=0.01,
        startup_timeout_s=10.0,
    )

    assert result.events == []
    assert len(result.errors) == 1
    error = result.errors[0]
    assert error["event"] == "timeout"
    assert error["schedule_name"] == "task_id_baseline"
    assert error["round_index"] == 0
    assert error["pending_commands"]
    assert {
        "core_index",
        "cpu_id",
        "task_id",
        "task_name",
    } <= set(error["pending_commands"][0])
    assert error["process_exitcodes"]
    assert {
        "pid",
        "core_index",
        "exitcode",
        "is_alive",
    } <= set(error["process_exitcodes"][0])


def test_run_schedule_with_process_workers_rejects_empty_plan():
    with pytest.raises(ValueError, match="plan must not be empty"):
        run_schedule_with_process_workers(
            [],
            steps_per_env=1,
            env_factory=fake_process_env_factory,
            dummy_action=[0.0] * 7,
            subprocess_timeout_s=10.0,
        )


def test_run_schedule_with_process_workers_rejects_duplicate_task_ids():
    records = _records_for_schedule()
    plan = build_task_id_baseline_plan(records, cpu_ids=[0, 1])
    duplicate_plan = plan + [plan[0]]

    with pytest.raises(ValueError, match="duplicate task_id"):
        run_schedule_with_process_workers(
            duplicate_plan,
            steps_per_env=1,
            env_factory=fake_process_env_factory,
            dummy_action=[0.0] * 7,
            subprocess_timeout_s=10.0,
        )


def test_apply_cpu_affinity_returns_false_when_affinity_unavailable(monkeypatch):
    monkeypatch.delattr("os.sched_setaffinity", raising=False)

    assert apply_cpu_affinity(0) is False


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


def test_compute_schedule_summary_includes_missing_core_idle():
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
        StepEvent(
            schedule_name="s",
            round_index=1,
            core_index=0,
            cpu_id=0,
            task_id=1,
            task_name="a",
            task_step_index=1,
            latency_s=0.03,
            round_wall_time_s=0.03,
            idle_time_s=0.0,
            cpu_affinity_applied=True,
        ),
    ]

    summary = compute_schedule_summary("s", events)

    assert summary["makespan_s"] == 0.05
    assert summary["mean_core_idle_ratio"] == pytest.approx(0.375)


def test_compute_comparison_metrics_compares_and_aggregates_random():
    summaries = [
        {
            "schedule_name": "task_id_baseline",
            "steps_per_second": 10.0,
            "mean_core_idle_ratio": 0.4,
        },
        {
            "schedule_name": "trapezoid_pipeline",
            "steps_per_second": 15.0,
            "mean_core_idle_ratio": 0.1,
        },
        {
            "schedule_name": "random_baseline_0",
            "steps_per_second": 8.0,
            "mean_core_idle_ratio": 0.5,
        },
        {
            "schedule_name": "random_baseline_1",
            "steps_per_second": 12.0,
            "mean_core_idle_ratio": 0.3,
        },
    ]

    metrics = compute_comparison_metrics(summaries)

    comparison = metrics["baseline_comparison"]
    assert comparison["trapezoid_pipeline"]["speedup_vs_task_id_baseline"] == 1.5
    assert (
        comparison["trapezoid_pipeline"]["bubble_reduction_vs_task_id_baseline"]
        == pytest.approx(0.75)
    )
    assert comparison["random_baseline_0"]["speedup_vs_task_id_baseline"] == 0.8
    random_aggregate = metrics["random_aggregate"]
    assert random_aggregate == {
        "count": 2,
        "mean_steps_per_second": 10.0,
        "median_steps_per_second": 10.0,
        "best_steps_per_second": 12.0,
        "worst_steps_per_second": 8.0,
        "mean_core_idle_ratio": 0.4,
    }


def test_write_comparison_report_includes_comparison_random_and_mapping_evidence(
    tmp_path: Path,
):
    report_path = tmp_path / "comparison_report.md"
    records = _records_for_schedule()
    trapezoid_plan = build_trapezoid_pipeline_plan(records, cpu_ids=[20, 21])
    summaries = [
        {
            "schedule_name": "task_id_baseline",
            "status": "completed",
            "steps_per_second": 10.0,
            "mean_core_idle_ratio": 0.4,
        },
        {
            "schedule_name": "trapezoid_pipeline",
            "status": "completed",
            "steps_per_second": 15.0,
            "mean_core_idle_ratio": 0.1,
        },
        {
            "schedule_name": "random_baseline",
            "status": "completed",
            "steps_per_second": 8.0,
            "mean_core_idle_ratio": 0.5,
        },
    ]

    write_comparison_report(
        report_path,
        summaries,
        plans={"trapezoid_pipeline": trapezoid_plan},
    )

    report = report_path.read_text(encoding="utf-8")
    assert "## Baseline Comparison" in report
    assert "speedup_vs_task_id_baseline" in report
    assert "bubble_reduction_vs_task_id_baseline" in report
    assert "| trapezoid_pipeline | 1.500000 | 0.750000 |" in report
    assert "## Random Baseline Aggregate" in report
    assert (
        "| count | mean steps/sec | median steps/sec | best steps/sec | "
        "worst steps/sec | mean idle ratio |"
    ) in report
    assert "## Trapezoid Mapping Evidence" in report
    assert "| core_index | cpu_id | long task ids | short task ids |" in report
    assert "| 0 | 20 | 4 | 1 |" in report
    assert "| 1 | 21 | 3 | 2 |" in report


def test_benchmark_runner_uses_injected_step_function_for_unit_tests():
    records = _records_for_schedule()
    plan = build_task_id_baseline_plan(records, cpu_ids=[0, 1])
    runner = BenchmarkRunner(
        steps_per_env=1,
        step_fn=lambda item, step_index: 0.01 * item.task.task_id,
    )

    result = runner.run("task_id_baseline", plan)

    assert result.summary["schedule_name"] == "task_id_baseline"
    assert result.summary["total_steps"] == 4
    assert len(result.events) == 4
    assert result.errors == []


def test_benchmark_runner_uses_process_workers_without_step_function():
    records = _records_for_schedule()
    plan = build_task_id_baseline_plan(records, cpu_ids=[0, 1])
    runner = BenchmarkRunner(
        steps_per_env=1,
        env_factory=fake_process_env_factory,
        dummy_action=[0.0] * 7,
        subprocess_timeout_s=10.0,
    )

    result = runner.run("task_id_baseline", plan)

    assert result.summary["schedule_name"] == "task_id_baseline"
    assert result.summary["total_steps"] == 4
    assert len(result.events) == 4
    assert result.errors == []


def test_benchmark_runner_reports_failed_schedule_from_step_exception():
    records = _records_for_schedule()
    plan = build_task_id_baseline_plan(records, cpu_ids=[0, 1])

    def fail_on_task(item: ScheduleItem, step_index: int) -> float:
        if item.task.task_id == 2:
            raise RuntimeError("step failed")
        return 0.01

    runner = BenchmarkRunner(steps_per_env=1, step_fn=fail_on_task)

    result = runner.run("task_id_baseline", plan)

    assert result.summary["status"] == "failed"
    assert result.errors
    assert "step failed" in result.errors[0]["error"]


def test_main_fake_mode_writes_outputs(tmp_path: Path):
    csv_path = tmp_path / "tasks.csv"
    _write_task_csv(csv_path)
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            "--task-csv",
            str(csv_path),
            "--num-envs",
            "2",
            "--cpu-ids",
            "0,1",
            "--steps-per-env",
            "2",
            "--output-dir",
            str(output_dir),
            "--fake-latency-from-csv",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "run_config.json").exists()
    assert (output_dir / "selected_tasks.csv").exists()
    assert (output_dir / "schedule_plan_task_id_baseline.csv").exists()
    assert (output_dir / "step_events_task_id_baseline.jsonl").exists()
    assert (output_dir / "schedule_summary.csv").exists()
    assert (output_dir / "schedule_summary.json").exists()
    assert (output_dir / "comparison_report.md").exists()
    summaries = json.loads((output_dir / "schedule_summary.json").read_text())
    assert {item["schedule_name"] for item in summaries} >= {
        "task_id_baseline",
        "trapezoid_pipeline",
    }


def _base_main_args(csv_path: Path, output_dir: Path) -> list[str]:
    return [
        "--task-csv",
        str(csv_path),
        "--num-envs",
        "2",
        "--cpu-ids",
        "0,1",
        "--steps-per-env",
        "2",
        "--output-dir",
        str(output_dir),
    ]


@pytest.mark.parametrize("cpu_ids", ["-1,0", "0,0"])
def test_main_rejects_invalid_cpu_ids(tmp_path: Path, cpu_ids: str):
    csv_path = tmp_path / "tasks.csv"
    _write_task_csv(csv_path)
    args = _base_main_args(csv_path, tmp_path / "out")
    args[args.index("--cpu-ids") + 1] = cpu_ids
    args.append("--fake-latency-from-csv")

    with pytest.raises(SystemExit):
        main(args)


def test_main_fake_mode_ignores_invalid_dummy_action_and_clears_stale_errors(
    tmp_path: Path,
):
    csv_path = tmp_path / "tasks.csv"
    _write_task_csv(csv_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    errors_path = output_dir / "errors.jsonl"
    errors_path.write_text('{"old": true}\n', encoding="utf-8")

    exit_code = main(
        [
            *_base_main_args(csv_path, output_dir),
            "--dummy-action",
            "bad",
            "--fake-latency-from-csv",
        ]
    )

    assert exit_code == 0
    assert not errors_path.exists()


def test_main_real_mode_rejects_invalid_dummy_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    csv_path = tmp_path / "tasks.csv"
    _write_task_csv(csv_path)

    def fail_if_called(**kwargs):
        raise AssertionError("env factory should not be created for invalid CLI input")

    monkeypatch.setattr(
        "toolkits.run_libero_latency_schedule_benchmark.make_libero_env_factory",
        fail_if_called,
    )

    with pytest.raises(SystemExit):
        main(
            [
                *_base_main_args(csv_path, tmp_path / "out"),
                "--dummy-action",
                "bad",
            ]
        )


def test_main_random_repeats_are_named_and_run_config_is_json_safe(tmp_path: Path):
    csv_path = tmp_path / "tasks.csv"
    _write_task_csv(csv_path)
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            *_base_main_args(csv_path, output_dir),
            "--random-baseline-repeats",
            "2",
            "--fake-latency-from-csv",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "schedule_plan_random_baseline_0.csv").exists()
    assert (output_dir / "schedule_plan_random_baseline_1.csv").exists()
    summaries = json.loads((output_dir / "schedule_summary.json").read_text())
    assert {item["schedule_name"] for item in summaries} >= {
        "random_baseline_0",
        "random_baseline_1",
    }
    summary_by_name = {item["schedule_name"]: item for item in summaries}
    assert summary_by_name["trapezoid_pipeline"][
        "speedup_vs_task_id_baseline"
    ] is not None
    assert summary_by_name["trapezoid_pipeline"][
        "bubble_reduction_vs_task_id_baseline"
    ] is not None
    run_config = json.loads((output_dir / "run_config.json").read_text())
    assert run_config["task_csv"] == str(csv_path)
    assert run_config["output_dir"] == str(output_dir)
    assert run_config["cpu_ids"] == [0, 1]
