from __future__ import annotations

import pytest

from toolkits.rollout_eval.benchmark.run import build_request, parse_args
from toolkits.rollout_eval.benchmark.scenarios import expand_cases
from toolkits.rollout_eval.benchmark.types import BenchmarkRequest, EnvModelPreset


def test_benchmark_cli_defaults() -> None:
    args = parse_args(["--config-path", "examples/embodiment/config", "--config-name", "x"])
    assert args.pipeline == "process"
    assert args.scenario_set == (
        "concurrent_mps,concurrent_mig,env_only_mps,model_only_mps,"
        "env_only_mig,model_only_mig,env_only_cpu_core,concurrent_cpu_core"
    )
    assert args.model_only_input == "dummy_from_env_reset"
    assert args.env_only_action == "random"
    assert args.pipeline_queue_timeout_s == 5.0
    assert args.num_envs is None
    assert args.cpu_bind_cores == ""
    assert args.cpu_bind_strategy == "even_split"
    assert args.cpu_bind_config is None
    assert args.cpu_bind_strict is True


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
    assert request.cpu_bind_cores is None
    assert request.cpu_bind_strategy == "even_split"
    assert request.cpu_bind_config is None
    assert request.cpu_bind_strict is True


def test_benchmark_build_request_parses_cpu_bind_options() -> None:
    args = parse_args(
        [
            "--config-path",
            "examples/embodiment/config",
            "--config-name",
            "x",
            "--scenario-set",
            "env_only_cpu_core,concurrent_cpu_core",
            "--cpu-bind-cores",
            "0-7",
            "--cpu-bind-strategy",
            "even_split",
        ]
    )
    request = build_request(args)

    assert request.scenario_set == ("env_only_cpu_core", "concurrent_cpu_core")
    assert request.cpu_bind_cores == "0-7"
    assert request.cpu_bind_strategy == "even_split"
    assert request.cpu_bind_config is None
    assert request.cpu_bind_strict is True


def test_benchmark_build_request_allows_default_cpu_strategy_without_cores() -> None:
    args = parse_args(
        [
            "--config-path",
            "examples/embodiment/config",
            "--config-name",
            "x",
            "--scenario-set",
            "env_only_cpu_core,concurrent_cpu_core",
            "--cpu-bind-strategy",
            "default",
        ]
    )

    request = build_request(args)

    assert request.scenario_set == ("env_only_cpu_core", "concurrent_cpu_core")
    assert request.cpu_bind_cores is None
    assert request.cpu_bind_strategy == "none"
    assert request.cpu_bind_config is None


def test_build_request_rejects_explicit_cpu_scenarios_without_cpu_bind_cores() -> None:
    args = parse_args(
        [
            "--config-path",
            "examples/embodiment/config",
            "--config-name",
            "x",
            "--scenario-set",
            "env_only_cpu_core,concurrent_cpu_core",
        ]
    )

    with pytest.raises(ValueError, match="cpu_core scenarios require --cpu-bind-cores"):
        build_request(args)


def test_build_request_allows_explicit_cpu_scenarios_without_cores_for_none_mode() -> None:
    args = parse_args(
        [
            "--config-path",
            "examples/embodiment/config",
            "--config-name",
            "x",
            "--scenario-set",
            "env_only_cpu_core,concurrent_cpu_core",
            "--cpu-bind-strategy",
            "none",
        ]
    )

    request = build_request(args)

    assert request.scenario_set == ("env_only_cpu_core", "concurrent_cpu_core")
    assert request.cpu_bind_cores is None
    assert request.cpu_bind_strategy == "none"


def test_build_request_supports_robocasa_openpi_preset() -> None:
    args = parse_args(
        [
            "--config-path",
            "examples/embodiment/config",
            "--config-name",
            "x",
            "--env-model-preset",
            "robocasa_openpi",
        ]
    )

    request = build_request(args)
    assert request.presets == (
        EnvModelPreset(
            name="robocasa_openpi",
            env_type="robocasa",
            model_type="openpi",
        ),
    )


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


def test_expand_cpu_core_cases_generates_deterministic_case_ids() -> None:
    request = BenchmarkRequest(
        config_path="c",
        config_name="n",
        override=(),
        output_dir="o",
        scenario_set=("env_only_cpu_core", "concurrent_cpu_core"),
        pipeline="process",
        mps_sm=(),
        mig_devices=(),
        presets=(EnvModelPreset("p1", "maniskill", "openvla_oft"),),
        model_only_input="dummy_from_env_reset",
        env_only_action="random",
        warmup_steps=1,
        measure_steps=2,
        cpu_bind_cores="0-7",
        cpu_bind_strategy="even_split",
        cpu_bind_config=None,
        cpu_bind_strict=True,
    )

    cases = expand_cases(request)

    assert [case.scenario for case in cases] == [
        "concurrent_cpu_core",
        "env_only_cpu_core",
    ]
    assert len({case.case_id for case in cases}) == 2
    assert all(case.cpu_binding_mode == "even_split" for case in cases)
    assert all(case.cpu_available_cores == tuple(range(8)) for case in cases)
    assert all("cpu-even-split-" in case.case_id for case in cases)


def test_expand_cpu_core_cases_support_none_binding_mode_without_cores() -> None:
    request = BenchmarkRequest(
        config_path="c",
        config_name="n",
        override=(),
        output_dir="o",
        scenario_set=("env_only_cpu_core", "concurrent_cpu_core"),
        pipeline="process",
        mps_sm=(),
        mig_devices=(),
        presets=(EnvModelPreset("p1", "maniskill", "openvla_oft"),),
        model_only_input="dummy_from_env_reset",
        env_only_action="random",
        warmup_steps=1,
        measure_steps=2,
        cpu_bind_cores=None,
        cpu_bind_strategy="none",
        cpu_bind_config=None,
        cpu_bind_strict=True,
    )

    cases = expand_cases(request)

    assert [case.scenario for case in cases] == [
        "concurrent_cpu_core",
        "env_only_cpu_core",
    ]
    assert all(case.cpu_binding_mode == "none" for case in cases)
    assert all(case.cpu_available_cores is None for case in cases)
    assert all("cpu-none-default-sched" in case.case_id for case in cases)


def test_expand_cpu_core_case_ids_include_normalized_cpu_selection() -> None:
    base_kwargs = {
        "config_path": "c",
        "config_name": "n",
        "override": (),
        "output_dir": "o",
        "scenario_set": ("env_only_cpu_core",),
        "pipeline": "process",
        "mps_sm": (),
        "mig_devices": (),
        "presets": (EnvModelPreset("p1", "maniskill", "openvla_oft"),),
        "model_only_input": "dummy_from_env_reset",
        "env_only_action": "random",
        "warmup_steps": 1,
        "measure_steps": 2,
        "cpu_bind_strategy": "even_split",
        "cpu_bind_config": None,
        "cpu_bind_strict": True,
    }
    request_a = BenchmarkRequest(**base_kwargs, cpu_bind_cores="0-3")
    request_b = BenchmarkRequest(**base_kwargs, cpu_bind_cores="4-7")

    cases_a = expand_cases(request_a)
    cases_b = expand_cases(request_b)

    assert len(cases_a) == 1
    assert len(cases_b) == 1
    assert cases_a[0].case_id != cases_b[0].case_id
    assert len(cases_a[0].case_id) < 80
    assert len(cases_b[0].case_id) < 80
    assert "-n4-h" in cases_a[0].case_id
    assert "-n4-h" in cases_b[0].case_id
    assert cases_a[0].cpu_available_cores == (0, 1, 2, 3)
    assert cases_b[0].cpu_available_cores == (4, 5, 6, 7)


def test_expand_cpu_core_cases_fail_fast_for_config_only_input_in_task1() -> None:
    request = BenchmarkRequest(
        config_path="c",
        config_name="n",
        override=(),
        output_dir="o",
        scenario_set=("env_only_cpu_core", "concurrent_cpu_core"),
        pipeline="process",
        mps_sm=(),
        mig_devices=(),
        presets=(EnvModelPreset("p1", "maniskill", "openvla_oft"),),
        model_only_input="dummy_from_env_reset",
        env_only_action="random",
        warmup_steps=1,
        measure_steps=2,
        cpu_bind_cores=None,
        cpu_bind_strategy="even_split",
        cpu_bind_config="cpu-bind.yaml",
        cpu_bind_strict=True,
    )

    with pytest.raises(ValueError, match="cpu_bind_cores"):
        expand_cases(request)


def test_expand_cpu_core_cases_fail_fast_for_blank_or_delimiter_only_cpu_bind_cores() -> None:
    request = BenchmarkRequest(
        config_path="c",
        config_name="n",
        override=(),
        output_dir="o",
        scenario_set=("env_only_cpu_core", "concurrent_cpu_core"),
        pipeline="process",
        mps_sm=(),
        mig_devices=(),
        presets=(EnvModelPreset("p1", "maniskill", "openvla_oft"),),
        model_only_input="dummy_from_env_reset",
        env_only_action="random",
        warmup_steps=1,
        measure_steps=2,
        cpu_bind_cores=" , , ",
        cpu_bind_strategy="even_split",
        cpu_bind_config=None,
        cpu_bind_strict=True,
    )

    with pytest.raises(ValueError, match="cpu_bind_cores"):
        expand_cases(request)
