from __future__ import annotations

from toolkits.rollout_eval.benchmark.run import build_request, parse_args
from toolkits.rollout_eval.benchmark.scenarios import expand_cases
from toolkits.rollout_eval.benchmark.types import BenchmarkRequest, EnvModelPreset


def test_benchmark_cli_defaults() -> None:
    args = parse_args(["--config-path", "examples/embodiment/config", "--config-name", "x"])
    assert args.pipeline == "process"
    assert args.scenario_set == (
        "concurrent_mps,concurrent_mig,env_only_mps,model_only_mps,env_only_mig,model_only_mig"
    )
    assert args.model_only_input == "dummy_from_env_reset"
    assert args.env_only_action == "random"
    assert args.pipeline_queue_timeout_s == 5.0
    assert args.num_envs is None


def test_benchmark_build_request_defaults() -> None:
    args = parse_args(["--config-path", "examples/embodiment/config", "--config-name", "x"])
    request = build_request(args)

    assert request.scenario_set == (
        "concurrent_mps",
        "concurrent_mig",
        "env_only_mps",
        "model_only_mps",
        "env_only_mig",
        "model_only_mig",
    )
    assert request.mps_sm == (20, 40, 60)
    assert request.pipeline == "process"
    assert request.pipeline_queue_timeout_s == 5.0
    assert request.num_envs_override is None


def test_expand_all_scenarios_generates_expected_classes_and_unique_ids() -> None:
    request = BenchmarkRequest(
        config_path="c",
        config_name="n",
        override=(),
        output_dir="o",
        scenario_set=(
            "concurrent_mps",
            "concurrent_mig",
            "env_only_mps",
            "model_only_mps",
            "env_only_mig",
            "model_only_mig",
        ),
        pipeline="process",
        mps_sm=(20, 40),
        mig_devices=("MIG-A", "MIG-B"),
        presets=(
            EnvModelPreset("p1", "maniskill", "openvla_oft"),
            EnvModelPreset("p2", "behavior", "openpi"),
        ),
        model_only_input="dummy_from_env_reset",
        env_only_action="random",
        warmup_steps=10,
        measure_steps=100,
    )

    cases = expand_cases(request)

    assert {case.scenario for case in cases} == {
        "concurrent_mps",
        "concurrent_mig",
        "env_only_mps",
        "model_only_mps",
        "env_only_mig",
        "model_only_mig",
    }
    assert len(cases) == 24
    assert len({case.case_id for case in cases}) == len(cases)


def test_expand_cases_is_deterministic() -> None:
    request = BenchmarkRequest(
        config_path="c",
        config_name="n",
        override=(),
        output_dir="o",
        scenario_set=("model_only_mig", "concurrent_mps"),
        pipeline="process",
        mps_sm=(40, 20),
        mig_devices=("MIG-B", "MIG-A"),
        presets=(
            EnvModelPreset("preset-b", "behavior", "openpi"),
            EnvModelPreset("preset-a", "maniskill", "openvla_oft"),
        ),
        model_only_input="dummy_from_env_reset",
        env_only_action="random",
        warmup_steps=1,
        measure_steps=2,
    )

    first = expand_cases(request)
    second = expand_cases(request)

    assert [case.case_id for case in first] == [case.case_id for case in second]
