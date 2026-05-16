import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from toolkits.profile_libero_step_latency import (
    ProfileConfig,
    ProfileResult,
    TaskTrialSpec,
    _profile_subprocess_entry,
    append_jsonl,
    build_arg_parser,
    compute_latency_summary,
    config_from_args,
    parse_bddl_metadata,
    parse_dummy_action,
    parse_int_list,
    parse_task_ids,
    profile_task_trial,
    profile_task_trial_in_subprocess,
    select_trial_ids,
    write_run_config,
    write_summary_files,
)
from toolkits.profile_libero_step_latency import (
    main as profile_main,
)

SAMPLE_BDDL = """
(define (problem LIBERO_Kitchen_Tabletop_Manipulation)
  (:domain robosuite)
  (:language turn on the stove and put the frying pan on it)
  (:regions
    (flat_stove_init_region
      (:target kitchen_table)
      (:ranges ((-0.21 0.19 -0.19 0.21)))
    )
    (cook_region
      (:target flat_stove_1)
    )
  )
  (:fixtures
    kitchen_table - kitchen_table
    flat_stove_1 - flat_stove
  )
  (:objects
    chefmate_8_frypan_1 - chefmate_8_frypan
    moka_pot_1 - moka_pot
  )
  (:obj_of_interest
    chefmate_8_frypan_1
    flat_stove_1
  )
  (:init
    (On flat_stove_1 kitchen_table_flat_stove_init_region)
    (On chefmate_8_frypan_1 kitchen_table_frypan_init_region)
  )
  (:goal
    (And (Turnon flat_stove_1) (On chefmate_8_frypan_1 flat_stove_1_cook_region))
  )
)
"""


def test_parse_bddl_metadata_counts_sections(tmp_path: Path):
    bddl_path = tmp_path / "KITCHEN_SCENE3_turn_on_the_stove.bddl"
    bddl_path.write_text(SAMPLE_BDDL)

    metadata = parse_bddl_metadata(bddl_path)

    assert metadata["problem_name"] == "LIBERO_Kitchen_Tabletop_Manipulation"
    assert metadata["domain_name"] == "robosuite"
    assert metadata["task_language"] == "turn on the stove and put the frying pan on it"
    assert metadata["scene_type"] == "kitchen"
    assert metadata["scene_name"] == "KITCHEN_SCENE3"
    assert metadata["region_names"] == ["flat_stove_init_region", "cook_region"]
    assert metadata["num_regions"] == 2
    assert metadata["num_fixtures"] == 2
    assert metadata["fixture_categories"] == ["flat_stove", "kitchen_table"]
    assert metadata["num_objects"] == 2
    assert metadata["object_categories"] == ["chefmate_8_frypan", "moka_pot"]
    assert metadata["obj_of_interest"] == ["chefmate_8_frypan_1", "flat_stove_1"]
    assert metadata["num_obj_of_interest"] == 2
    assert metadata["init_predicates"] == ["On", "On"]
    assert metadata["num_init_predicates"] == 2
    assert metadata["goal_predicates"] == ["Turnon", "On"]
    assert metadata["num_goal_predicates"] == 2


def test_parse_bddl_metadata_infers_living_room_scene_name(tmp_path: Path):
    bddl_path = (
        tmp_path
        / "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl"
    )
    bddl_path.write_text(SAMPLE_BDDL)

    metadata = parse_bddl_metadata(bddl_path)

    assert metadata["scene_name"] == "LIVING_ROOM_SCENE2"
    assert metadata["scene_type"] == "living_room"


def test_parse_int_list_accepts_all_and_numbers():
    assert parse_int_list("all", allow_all=True) == "all"
    assert parse_int_list("0,3,7", allow_all=True) == [0, 3, 7]
    assert parse_int_list(" 1 , 2 ") == [1, 2]


def test_parse_int_list_rejects_empty_and_all_when_disallowed():
    with pytest.raises(ValueError, match="empty"):
        parse_int_list("")
    with pytest.raises(ValueError, match="not allowed"):
        parse_int_list("all", allow_all=False)


def test_parse_task_ids_all_uses_task_count():
    assert parse_task_ids("all", num_tasks=4) == [0, 1, 2, 3]
    assert parse_task_ids("0,2", num_tasks=4) == [0, 2]


def test_parse_task_ids_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        parse_task_ids("4", num_tasks=4)


def test_select_trial_ids_specific_ids_win():
    assert select_trial_ids(
        num_trials=10,
        trials_per_task=2,
        specific_trial_ids=[5, 1],
        seed=0,
        task_id=3,
    ) == [5, 1]


def test_select_trial_ids_rejects_specific_ids_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        select_trial_ids(
            num_trials=10,
            trials_per_task=2,
            specific_trial_ids=[10],
            seed=0,
            task_id=3,
        )


def test_select_trial_ids_deterministic_prefix_when_under_limit():
    assert select_trial_ids(
        num_trials=3,
        trials_per_task=10,
        specific_trial_ids=None,
        seed=123,
        task_id=0,
    ) == [0, 1, 2]


def test_select_trial_ids_seeded_sample_is_stable():
    first = select_trial_ids(
        num_trials=20,
        trials_per_task=4,
        specific_trial_ids=None,
        seed=123,
        task_id=2,
    )
    second = select_trial_ids(
        num_trials=20,
        trials_per_task=4,
        specific_trial_ids=None,
        seed=123,
        task_id=2,
    )
    assert first == second
    assert len(first) == 4
    assert len(set(first)) == 4


def test_parse_dummy_action_default_and_override():
    np.testing.assert_allclose(parse_dummy_action(None), [0, 0, 0, 0, 0, 0, -1])
    np.testing.assert_allclose(parse_dummy_action("1,2,3"), [1, 2, 3])


def test_compute_latency_summary_percentiles_and_tail_ratio():
    summary = compute_latency_summary([0.01, 0.02, 0.03, 0.10])
    assert summary["step_count"] == 4
    assert math.isclose(summary["mean_latency_s"], 0.04)
    assert math.isclose(summary["median_latency_s"], 0.025)
    assert summary["p99_latency_s"] > summary["p95_latency_s"]
    assert math.isclose(
        summary["tail_ratio_p99_to_median"],
        summary["p99_latency_s"] / summary["median_latency_s"],
    )


def test_compute_latency_summary_empty_latency_list():
    summary = compute_latency_summary([])
    assert summary["step_count"] == 0
    assert summary["mean_latency_s"] is None
    assert summary["tail_ratio_p99_to_median"] is None


class FakeModel:
    nbody = 4
    ngeom = 5
    njnt = 2
    nq = 9
    nv = 8
    nu = 7
    ncam = 2
    camera_names = ["agentview", "robot0_eye_in_hand"]


class FakeSim:
    model = FakeModel()


class FakeEnv:
    def __init__(self):
        self.sim = FakeSim()
        self.steps = 0
        self.closed = False

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        self.steps = 0
        return {"obs": 1}

    def set_init_state(self, init_state):
        self.init_state = init_state
        return {"obs": 2}

    def step(self, action):
        self.steps += 1
        done = self.steps >= 3
        return {"obs": self.steps}, float(done), done, {"success": done}

    def check_success(self):
        return self.steps >= 3

    def close(self):
        self.closed = True


class IncrementingClock:
    def __init__(self, *, step: float):
        self.value = 0.0
        self.step = step

    def __call__(self):
        current = self.value
        self.value += self.step
        return current


def _profile_config(
    tmp_path: Path,
    *,
    measure_steps: int = 3,
    stop_on_done: bool = False,
) -> ProfileConfig:
    return ProfileConfig(
        suite="libero_90",
        task_ids="0",
        trials_per_task=1,
        specific_trial_ids=None,
        warmup_steps=1,
        measure_steps=measure_steps,
        cpu_id=None,
        cpu_ids=None,
        camera_height=64,
        camera_width=64,
        libero_type="standard",
        seed=11,
        output_dir=tmp_path,
        dummy_action=[0.0] * 7,
        stop_on_done=stop_on_done,
        subprocess_timeout_s=30.0,
    )


def _task_trial_spec(bddl_path: Path) -> TaskTrialSpec:
    return TaskTrialSpec(
        suite_name="libero_90",
        task_id=0,
        trial_id=1,
        task_name="KITCHEN_SCENE3_task",
        task_language="turn on the stove",
        bddl_file=str(bddl_path),
        seed=11,
    )


def test_config_from_args_parses_cli_values(tmp_path: Path):
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--suite",
            "libero_90",
            "--task-ids",
            "0,1",
            "--trials-per-task",
            "2",
            "--specific-trial-ids",
            "3,4",
            "--warmup-steps",
            "5",
            "--measure-steps",
            "6",
            "--cpu-id",
            "0",
            "--subprocess-timeout-s",
            "123.5",
            "--camera-height",
            "128",
            "--camera-width",
            "96",
            "--output-dir",
            str(tmp_path),
            "--dummy-action",
            "0,0,0,0,0,0,-1",
            "--stop-on-done",
        ]
    )
    config = config_from_args(args)
    assert config.suite == "libero_90"
    assert config.task_ids == "0,1"
    assert config.trials_per_task == 2
    assert config.specific_trial_ids == [3, 4]
    assert config.warmup_steps == 5
    assert config.measure_steps == 6
    assert config.cpu_id == 0
    assert config.cpu_ids is None
    assert config.subprocess_timeout_s == 123.5
    assert config.camera_height == 128
    assert config.camera_width == 96
    assert config.output_dir == tmp_path
    assert config.dummy_action == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    assert config.stop_on_done is True


def test_config_from_args_parses_cpu_ids_and_default_timeout(tmp_path: Path):
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--suite",
            "libero_90",
            "--output-dir",
            str(tmp_path),
            "--cpu-ids",
            "0,1",
        ]
    )

    config = config_from_args(args)

    assert config.cpu_id is None
    assert config.cpu_ids == [0, 1]
    assert config.subprocess_timeout_s is not None
    assert config.subprocess_timeout_s > 0


def test_config_from_args_rejects_cpu_id_with_cpu_ids(tmp_path: Path):
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--suite",
            "libero_90",
            "--output-dir",
            str(tmp_path),
            "--cpu-id",
            "0",
            "--cpu-ids",
            "1,2",
        ]
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        config_from_args(args)


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--camera-height", "-1", "--camera-height must be > 0"),
        ("--cpu-id", "-1", "--cpu-id must be >= 0"),
        ("--cpu-ids", "0,-1", "--cpu-ids must be >= 0"),
        (
            "--subprocess-timeout-s",
            "-1",
            "--subprocess-timeout-s must be > 0, 0, or none",
        ),
    ],
)
def test_config_from_args_rejects_invalid_cli_values(
    tmp_path: Path, flag: str, value: str, message: str
):
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--suite",
            "libero_90",
            "--output-dir",
            str(tmp_path),
            flag,
            value,
        ]
    )

    with pytest.raises(ValueError, match=message):
        config_from_args(args)


def test_main_reports_startup_errors_without_traceback(
    monkeypatch, capsys, tmp_path: Path
):
    def raise_import_error(config):
        raise ImportError("missing libero")

    monkeypatch.setattr(
        "toolkits.profile_libero_step_latency.run_profile",
        raise_import_error,
    )

    with pytest.raises(SystemExit) as exc_info:
        profile_main(
            [
                "--suite",
                "libero_90",
                "--output-dir",
                str(tmp_path),
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "missing libero" in captured.err
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_profile_task_trial_with_mock_env(tmp_path: Path):
    bddl_path = tmp_path / "KITCHEN_SCENE3_task.bddl"
    bddl_path.write_text(SAMPLE_BDDL)
    config = _profile_config(tmp_path)
    spec = _task_trial_spec(bddl_path)

    result = profile_task_trial(
        config=config,
        spec=spec,
        env_factory=FakeEnv,
        init_state=np.zeros(3),
        clock=IncrementingClock(step=0.25),
    )

    assert result.error is None
    assert len(result.events) == 3
    assert result.events[0]["event"] == "libero_step_latency"
    assert result.events[0]["suite_name"] == "libero_90"
    assert result.events[0]["num_objects"] == 2
    assert result.events[0]["nbody"] == 4
    assert result.events[0]["latency_s"] == 0.25
    assert result.summary["step_count"] == 3
    assert result.summary["mean_latency_s"] == 0.25
    assert result.summary["median_latency_s"] == 0.25
    assert result.summary["min_latency_s"] == 0.25
    assert result.summary["max_latency_s"] == 0.25
    assert result.summary["success_seen"] is True
    assert result.summary["done_seen_step"] == 2


def test_profile_task_trial_stop_on_done_stops_measurement(tmp_path: Path):
    bddl_path = tmp_path / "KITCHEN_SCENE3_task.bddl"
    bddl_path.write_text(SAMPLE_BDDL)
    config = _profile_config(tmp_path, measure_steps=10, stop_on_done=True)
    spec = _task_trial_spec(bddl_path)

    result = profile_task_trial(
        config=config,
        spec=spec,
        env_factory=FakeEnv,
        init_state=np.zeros(3),
        clock=IncrementingClock(step=0.25),
    )

    assert result.error is None
    assert len(result.events) == 3
    assert len(result.events) < config.measure_steps
    assert result.summary["step_count"] == 3
    assert result.summary["step_count"] < config.measure_steps
    assert result.summary["done_seen_step"] == 2


def test_profile_task_trial_error_includes_stage_and_traceback(tmp_path: Path):
    bddl_path = tmp_path / "KITCHEN_SCENE3_task.bddl"
    bddl_path.write_text(SAMPLE_BDDL)
    config = _profile_config(tmp_path)
    spec = _task_trial_spec(bddl_path)

    def raise_env_factory():
        raise RuntimeError("factory failed")

    result = profile_task_trial(
        config=config,
        spec=spec,
        env_factory=raise_env_factory,
        init_state=np.zeros(3),
        clock=IncrementingClock(step=0.25),
    )

    assert result.events == []
    assert result.summary is None
    assert result.error["error_type"] == "RuntimeError"
    assert result.error["error"] == "factory failed"
    assert result.error["stage"] == "env_factory"
    assert "RuntimeError: factory failed" in result.error["traceback"]


def test_profile_task_trial_continues_when_bddl_metadata_fails(tmp_path: Path):
    missing_bddl_path = tmp_path / "KITCHEN_SCENE3_missing.bddl"
    config = _profile_config(tmp_path, measure_steps=1)
    spec = _task_trial_spec(missing_bddl_path)

    result = profile_task_trial(
        config=config,
        spec=spec,
        env_factory=FakeEnv,
        init_state=np.zeros(3),
        clock=IncrementingClock(step=0.25),
    )

    assert result.error is None
    assert len(result.warnings) == 1
    assert result.warnings[0]["event"] == "warning"
    assert result.warnings[0]["stage"] == "parse_bddl_metadata"
    assert result.summary["step_count"] == 1
    assert result.events[0]["scene_type"] is None
    assert result.events[0]["num_objects"] is None
    assert result.summary["num_objects"] is None


def test_profile_task_trial_continues_when_runtime_metadata_fails(
    monkeypatch, tmp_path: Path
):
    bddl_path = tmp_path / "KITCHEN_SCENE3_task.bddl"
    bddl_path.write_text(SAMPLE_BDDL)
    config = _profile_config(tmp_path, measure_steps=1)
    spec = _task_trial_spec(bddl_path)

    def raise_runtime_metadata(env, config):
        raise RuntimeError("runtime metadata failed")

    monkeypatch.setattr(
        "toolkits.profile_libero_step_latency.collect_runtime_metadata",
        raise_runtime_metadata,
    )

    result = profile_task_trial(
        config=config,
        spec=spec,
        env_factory=FakeEnv,
        init_state=np.zeros(3),
        clock=IncrementingClock(step=0.25),
    )

    assert result.error is None
    assert len(result.warnings) == 1
    assert result.warnings[0]["event"] == "warning"
    assert result.warnings[0]["stage"] == "collect_runtime_metadata"
    assert result.summary["step_count"] == 1
    assert result.events[0]["camera_names"] is None
    assert result.events[0]["nbody"] is None
    assert result.summary["nbody"] is None


def test_profile_task_trial_returns_all_best_effort_warnings(
    monkeypatch, tmp_path: Path
):
    missing_bddl_path = tmp_path / "KITCHEN_SCENE3_missing.bddl"
    config = _profile_config(tmp_path, measure_steps=1)
    spec = _task_trial_spec(missing_bddl_path)

    def raise_runtime_metadata(env, config):
        raise RuntimeError("runtime metadata failed")

    monkeypatch.setattr(
        "toolkits.profile_libero_step_latency.collect_runtime_metadata",
        raise_runtime_metadata,
    )

    result = profile_task_trial(
        config=config,
        spec=spec,
        env_factory=FakeEnv,
        init_state=np.zeros(3),
        clock=IncrementingClock(step=0.25),
    )

    assert result.error is None
    assert [warning["stage"] for warning in result.warnings] == [
        "parse_bddl_metadata",
        "collect_runtime_metadata",
    ]
    assert [warning["event"] for warning in result.warnings] == ["warning", "warning"]
    assert result.summary["step_count"] == 1
    assert len(result.events) == 1


def test_output_writers_create_jsonl_csv_json(tmp_path: Path):
    class CustomValue:
        def __repr__(self):
            return "CustomValue(7)"

    event_path = tmp_path / "step_latency_events.jsonl"
    append_jsonl(
        event_path,
        [
            {"event": "a", "value": 1, "path": tmp_path, "nan": float("nan")},
            {
                "event": "b",
                "value": np.int64(2),
                "array": np.asarray([1, 2]),
                "inf": np.float64("inf"),
                "custom": CustomValue(),
            },
        ],
    )
    assert "NaN" not in event_path.read_text()
    assert "Infinity" not in event_path.read_text()
    records = [json.loads(line) for line in event_path.read_text().splitlines()]
    assert records == [
        {"event": "a", "nan": None, "path": str(tmp_path), "value": 1},
        {
            "array": [1, 2],
            "custom": {"__type__": "CustomValue", "repr": "CustomValue(7)"},
            "event": "b",
            "inf": None,
            "value": 2,
        },
    ]

    summaries = [
        {
            "suite_name": "libero_90",
            "task_id": np.int64(0),
            "step_count": 2,
            "error": None,
            "bddl_file": tmp_path / "task0.bddl",
        },
        {
            "suite_name": "libero_90",
            "task_id": 1,
            "step_count": 2,
            "error": None,
            "latencies": np.asarray([0.1, 0.2]),
        },
    ]
    write_summary_files(tmp_path, summaries)
    assert (tmp_path / "step_latency_summary.csv").exists()
    summary_json = json.loads((tmp_path / "step_latency_summary.json").read_text())
    assert summary_json == [
        {
            "bddl_file": str(tmp_path / "task0.bddl"),
            "error": None,
            "step_count": 2,
            "suite_name": "libero_90",
            "task_id": 0,
        },
        {
            "error": None,
            "latencies": [0.1, 0.2],
            "step_count": 2,
            "suite_name": "libero_90",
            "task_id": 1,
        },
    ]


def test_write_run_config_serializes_paths(tmp_path: Path):
    config = ProfileConfig(
        suite="libero_90",
        task_ids="all",
        trials_per_task=1,
        specific_trial_ids=None,
        warmup_steps=1,
        measure_steps=1,
        cpu_id=0,
        cpu_ids=None,
        camera_height=64,
        camera_width=64,
        libero_type="standard",
        seed=0,
        output_dir=tmp_path,
        dummy_action=[0.0] * 7,
        stop_on_done=False,
        subprocess_timeout_s=30.0,
    )
    write_run_config(tmp_path, config)
    data = json.loads((tmp_path / "run_config.json").read_text())
    assert data["output_dir"] == str(tmp_path)
    assert data["suite"] == "libero_90"
    assert data["subprocess_timeout_s"] == 30.0


def test_run_profile_uses_rotating_cpu_ids_and_timeout(monkeypatch, tmp_path: Path):
    config = replace(
        _profile_config(tmp_path, measure_steps=1),
        cpu_ids=[2, 4],
        subprocess_timeout_s=12.0,
    )
    specs = [
        _task_trial_spec(tmp_path / f"KITCHEN_SCENE3_task_{index}.bddl")
        for index in range(3)
    ]
    seen_cpu_ids = []
    seen_timeouts = []

    def fake_build_task_trial_specs(config):
        return specs, [np.zeros(3), np.zeros(3), np.zeros(3)]

    def fake_profile_task_trial_in_subprocess(
        *, config, spec, init_state, timeout_s=None
    ):
        seen_cpu_ids.append(config.cpu_id)
        seen_timeouts.append(timeout_s)
        return ProfileResult(
            events=[],
            summary={"task_id": spec.task_id, "trial_id": spec.trial_id},
            error=None,
        )

    monkeypatch.setattr(
        "toolkits.profile_libero_step_latency.build_task_trial_specs",
        fake_build_task_trial_specs,
    )
    monkeypatch.setattr(
        "toolkits.profile_libero_step_latency.profile_task_trial_in_subprocess",
        fake_profile_task_trial_in_subprocess,
    )

    from toolkits.profile_libero_step_latency import run_profile

    run_profile(config)

    assert seen_cpu_ids == [2, 4, 2]
    assert seen_timeouts == [12.0, 12.0, 12.0]


def test_run_profile_writes_all_result_warnings(monkeypatch, tmp_path: Path):
    config = _profile_config(tmp_path, measure_steps=1)
    spec = _task_trial_spec(tmp_path / "KITCHEN_SCENE3_task.bddl")

    def fake_build_task_trial_specs(config):
        return [spec], [np.zeros(3)]

    def fake_profile_task_trial_in_subprocess(
        *, config, spec, init_state, timeout_s=None
    ):
        return ProfileResult(
            events=[],
            summary={"task_id": spec.task_id, "trial_id": spec.trial_id},
            error=None,
            warnings=[
                {"event": "warning", "stage": "parse_bddl_metadata"},
                {"event": "warning", "stage": "collect_runtime_metadata"},
            ],
        )

    monkeypatch.setattr(
        "toolkits.profile_libero_step_latency.build_task_trial_specs",
        fake_build_task_trial_specs,
    )
    monkeypatch.setattr(
        "toolkits.profile_libero_step_latency.profile_task_trial_in_subprocess",
        fake_profile_task_trial_in_subprocess,
    )

    from toolkits.profile_libero_step_latency import run_profile

    run_profile(config)

    records = [
        json.loads(line)
        for line in (tmp_path / "errors.jsonl").read_text().splitlines()
    ]
    assert [record["stage"] for record in records] == [
        "parse_bddl_metadata",
        "collect_runtime_metadata",
    ]


def test_profile_task_trial_in_subprocess_reports_nonzero_exit(
    monkeypatch, tmp_path: Path
):
    bddl_path = tmp_path / "KITCHEN_SCENE3_task.bddl"
    bddl_path.write_text(SAMPLE_BDDL)
    config = ProfileConfig(
        suite="libero_90",
        task_ids="0",
        trials_per_task=1,
        specific_trial_ids=None,
        warmup_steps=0,
        measure_steps=1,
        cpu_id=None,
        cpu_ids=None,
        camera_height=64,
        camera_width=64,
        libero_type="standard",
        seed=11,
        output_dir=tmp_path,
        dummy_action=[0.0] * 7,
        stop_on_done=False,
        subprocess_timeout_s=30.0,
    )
    spec = TaskTrialSpec(
        suite_name="libero_90",
        task_id=0,
        trial_id=1,
        task_name="KITCHEN_SCENE3_task",
        task_language="turn on the stove",
        bddl_file=str(bddl_path),
        seed=11,
    )

    class FakeParentConn:
        def poll(self, timeout):
            return True

        def recv(self):
            raise EOFError

    class FakeProcess:
        exitcode = 9

        def __init__(self, target, args):
            self.target = target
            self.args = args

        def start(self):
            return None

        def join(self):
            return None

    class FakeContext:
        Process = FakeProcess

        def Pipe(self, duplex=False):
            return FakeParentConn(), object()

    monkeypatch.setattr(
        "toolkits.profile_libero_step_latency.mp.get_context",
        lambda method: FakeContext(),
    )

    result = profile_task_trial_in_subprocess(
        config=config,
        spec=spec,
        init_state=np.zeros(3),
    )

    assert result.events == []
    assert result.summary is None
    assert result.error["error_type"] == "SubprocessError"
    assert "exited with code 9" in result.error["error"]


def test_profile_task_trial_in_subprocess_receives_before_join(
    monkeypatch, tmp_path: Path
):
    bddl_path = tmp_path / "KITCHEN_SCENE3_task.bddl"
    bddl_path.write_text(SAMPLE_BDDL)
    config = _profile_config(tmp_path, measure_steps=1)
    spec = _task_trial_spec(bddl_path)
    calls = []

    class FakeParentConn:
        def poll(self, timeout):
            calls.append("poll")
            return True

        def recv(self):
            calls.append("recv")
            return ProfileResult(
                events=[{"event": "libero_step_latency"}],
                summary={"step_count": 1},
                error=None,
            )

    class FakeProcess:
        exitcode = 0

        def __init__(self, target, args):
            self.target = target
            self.args = args

        def start(self):
            calls.append("start")

        def join(self):
            calls.append("join")

    class FakeContext:
        Process = FakeProcess

        def Pipe(self, duplex=False):
            calls.append(f"pipe:{duplex}")
            return FakeParentConn(), object()

    monkeypatch.setattr(
        "toolkits.profile_libero_step_latency.mp.get_context",
        lambda method: FakeContext(),
    )

    result = profile_task_trial_in_subprocess(
        config=config,
        spec=spec,
        init_state=np.zeros(3),
    )

    assert result.error is None
    assert calls.index("recv") < calls.index("join")


def test_profile_task_trial_in_subprocess_timeout_terminates_child(
    monkeypatch, tmp_path: Path
):
    bddl_path = tmp_path / "KITCHEN_SCENE3_task.bddl"
    bddl_path.write_text(SAMPLE_BDDL)
    config = _profile_config(tmp_path, measure_steps=1)
    spec = _task_trial_spec(bddl_path)
    calls = []

    class FakeParentConn:
        def poll(self, timeout):
            calls.append(("poll", timeout))
            return False

        def recv(self):
            calls.append(("recv", None))
            raise AssertionError("recv should not be called after timeout")

        def close(self):
            calls.append(("parent_close", None))

    class FakeChildConn:
        def close(self):
            calls.append(("child_close", None))

    class FakeProcess:
        exitcode = None

        def __init__(self, target, args):
            self.target = target
            self.args = args

        def start(self):
            calls.append(("start", None))

        def terminate(self):
            calls.append(("terminate", None))
            self.exitcode = -15

        def join(self, timeout=None):
            calls.append(("join", timeout))

    class FakeContext:
        Process = FakeProcess

        def Pipe(self, duplex=False):
            calls.append(("pipe", duplex))
            return FakeParentConn(), FakeChildConn()

    monkeypatch.setattr(
        "toolkits.profile_libero_step_latency.mp.get_context",
        lambda method: FakeContext(),
    )

    result = profile_task_trial_in_subprocess(
        config=config,
        spec=spec,
        init_state=np.zeros(3),
        timeout_s=0.01,
    )

    assert result.events == []
    assert result.summary is None
    assert result.error["error_type"] == "SubprocessError"
    assert result.error["stage"] == "subprocess_timeout"
    assert "timed out after 0.01s" in result.error["error"]
    assert ("terminate", None) in calls
    assert ("join", 5.0) in calls
    assert ("recv", None) not in calls


def test_profile_task_trial_in_subprocess_timeout_kills_stubborn_child(
    monkeypatch, tmp_path: Path
):
    bddl_path = tmp_path / "KITCHEN_SCENE3_task.bddl"
    bddl_path.write_text(SAMPLE_BDDL)
    config = _profile_config(tmp_path, measure_steps=1)
    spec = _task_trial_spec(bddl_path)
    calls = []

    class FakeParentConn:
        def poll(self, timeout):
            calls.append(("poll", timeout))
            return False

        def close(self):
            calls.append(("parent_close", None))

    class FakeChildConn:
        def close(self):
            calls.append(("child_close", None))

    class FakeProcess:
        exitcode = None

        def __init__(self, target, args):
            self.target = target
            self.args = args
            self.kill_called = False

        def start(self):
            calls.append(("start", None))

        def terminate(self):
            calls.append(("terminate", None))

        def join(self, timeout=None):
            calls.append(("join", timeout))

        def is_alive(self):
            calls.append(("is_alive", None))
            return not self.kill_called

        def kill(self):
            calls.append(("kill", None))
            self.kill_called = True
            self.exitcode = -9

    class FakeContext:
        Process = FakeProcess

        def Pipe(self, duplex=False):
            calls.append(("pipe", duplex))
            return FakeParentConn(), FakeChildConn()

    monkeypatch.setattr(
        "toolkits.profile_libero_step_latency.mp.get_context",
        lambda method: FakeContext(),
    )

    result = profile_task_trial_in_subprocess(
        config=config,
        spec=spec,
        init_state=np.zeros(3),
        timeout_s=0.01,
    )

    assert result.error["error_type"] == "SubprocessError"
    assert result.error["stage"] == "subprocess_timeout"
    assert "cleanup attempted" in result.error["error"]
    assert ("terminate", None) in calls
    assert ("kill", None) in calls
    assert ("join", 5.0) in calls


def test_profile_subprocess_entry_sends_factory_errors(monkeypatch, tmp_path: Path):
    bddl_path = tmp_path / "KITCHEN_SCENE3_task.bddl"
    bddl_path.write_text(SAMPLE_BDDL)
    config = _profile_config(tmp_path, measure_steps=1)
    spec = _task_trial_spec(bddl_path)
    sent_results = []

    class FakeChildConn:
        def send(self, result):
            sent_results.append(result)

        def close(self):
            return None

    def raise_factory(config, spec):
        raise RuntimeError("libero import failed")

    monkeypatch.setattr(
        "toolkits.profile_libero_step_latency.make_libero_env_factory",
        raise_factory,
    )

    _profile_subprocess_entry(
        FakeChildConn(),
        config=config,
        spec=spec,
        init_state=np.zeros(3),
    )

    assert len(sent_results) == 1
    result = sent_results[0]
    assert result.events == []
    assert result.summary is None
    assert result.error["error_type"] == "RuntimeError"
    assert result.error["stage"] == "subprocess_entry"
    assert "libero import failed" in result.error["error"]
    assert "RuntimeError: libero import failed" in result.error["traceback"]
