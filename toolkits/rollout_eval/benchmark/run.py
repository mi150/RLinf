"""CLI entry for rollout_eval benchmark orchestration."""

from __future__ import annotations

import argparse
import json

from toolkits.rollout_eval.benchmark.orchestrator import run_benchmark_orchestrator
from toolkits.rollout_eval.benchmark.types import BenchmarkRequest, EnvModelPreset

DEFAULT_SCENARIO_SET = (
    "concurrent_mps",
    "concurrent_mig",
    "env_only_mps",
    "model_only_mps",
    "env_only_mig",
    "model_only_mig",
    "env_only_cpu_core",
    "concurrent_cpu_core",
)
DEFAULT_PRESET_MAP = {
    "maniskill_openvlaoft": EnvModelPreset(
        name="maniskill_openvlaoft", env_type="maniskill", model_type="openvla_oft"
    ),
    "behavior_openpi": EnvModelPreset(
        name="behavior_openpi", env_type="behavior", model_type="openpi"
    ),
    "libero_gr00t": EnvModelPreset(
        name="libero_gr00t", env_type="libero", model_type="gr00t"
    ),
    "metaworld_openpi": EnvModelPreset(
        name="metaworld_openpi", env_type="metaworld", model_type="openpi"
    ),
    "robocasa_openpi": EnvModelPreset(
        name="robocasa_openpi", env_type="robocasa", model_type="openpi"
    ),
}


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse benchmark CLI arguments only."""
    parser = argparse.ArgumentParser(description="Rollout eval benchmark matrix runner")
    parser.add_argument("--config-path", required=True, help="Hydra config directory")
    parser.add_argument("--config-name", required=True, help="Hydra config name")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override, repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        default="./rollout_eval_output/benchmark",
        help="Output directory for benchmark reports",
    )
    parser.add_argument(
        "--scenario-set",
        default=",".join(DEFAULT_SCENARIO_SET),
        help="Comma-separated scenario classes",
    )
    parser.add_argument(
        "--pipeline",
        default="process",
        choices=["process"],
        help="Concurrent execution model",
    )
    parser.add_argument(
        "--mps-sm",
        default="20,40,60",
        help="Comma-separated MPS SM share percentages",
    )
    parser.add_argument(
        "--mig-devices",
        default="",
        help="Comma-separated MIG UUID list",
    )
    parser.add_argument(
        "--env-model-preset",
        default="maniskill_openvlaoft,behavior_openpi",
        help="Comma-separated preset names",
    )
    parser.add_argument(
        "--model-only-input",
        default="dummy_from_env_reset",
        choices=["dummy_from_env_reset"],
        help="Model-only input source",
    )
    parser.add_argument(
        "--env-only-action",
        default="random",
        choices=["random", "zero"],
        help="Action generation policy for env-only profile",
    )
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override env.eval.total_num_envs to increase/decrease benchmark batch size",
    )
    parser.add_argument(
        "--pipeline-queue-timeout-s",
        type=float,
        default=5.0,
        help="Per-queue operation timeout for concurrent pipeline workers",
    )
    parser.add_argument(
        "--pipeline-run-timeout-s",
        type=float,
        default=None,
        help="Optional end-to-end timeout for one concurrent case",
    )
    parser.add_argument(
        "--cpu-bind-cores",
        default="",
        help="CPU pool for env binding, e.g. 0-127 or 0-31,64-95",
    )
    parser.add_argument(
        "--cpu-bind-strategy",
        default="even_split",
        choices=["even_split", "none", "default"],
        help="CPU binding strategy for cpu_core scenarios",
    )
    parser.add_argument(
        "--cpu-bind-config",
        default=None,
        help="Optional YAML file with explicit env_core_groups override",
    )
    parser.add_argument(
        "--no-cpu-bind-strict",
        action="store_false",
        dest="cpu_bind_strict",
        help="Allow non-strict CPU binding validation",
    )
    parser.set_defaults(cpu_bind_strict=True)
    return parser.parse_args(argv)


def build_request(args: argparse.Namespace) -> BenchmarkRequest:
    """Convert parsed args into a typed benchmark request."""
    cpu_bind_strategy = args.cpu_bind_strategy
    if cpu_bind_strategy == "default":
        cpu_bind_strategy = "none"

    raw_scenario_set = args.scenario_set
    scenario_set = _parse_csv(raw_scenario_set)
    cpu_scenarios = {"env_only_cpu_core", "concurrent_cpu_core"}
    has_cpu_scenario = any(scenario in cpu_scenarios for scenario in scenario_set)
    has_cpu_bind_cores = bool((args.cpu_bind_cores or "").strip())
    has_cpu_bind_config = bool((args.cpu_bind_config or "").strip())
    cpu_binding_disabled = cpu_bind_strategy == "none"
    if (
        has_cpu_scenario
        and not cpu_binding_disabled
        and not has_cpu_bind_cores
        and not has_cpu_bind_config
    ):
        if raw_scenario_set == ",".join(DEFAULT_SCENARIO_SET):
            scenario_set = tuple(
                scenario for scenario in scenario_set if scenario not in cpu_scenarios
            )
        else:
            raise ValueError(
                "cpu_core scenarios require --cpu-bind-cores in Task 1"
            )

    presets: list[EnvModelPreset] = []
    for name in _parse_csv(args.env_model_preset):
        preset = DEFAULT_PRESET_MAP.get(name)
        if preset is None:
            raise ValueError(
                f"Unknown preset '{name}'. Known presets: {sorted(DEFAULT_PRESET_MAP)}"
            )
        presets.append(preset)

    return BenchmarkRequest(
        config_path=args.config_path,
        config_name=args.config_name,
        override=tuple(args.override),
        output_dir=args.output_dir,
        scenario_set=scenario_set,
        pipeline=args.pipeline,
        mps_sm=_parse_int_csv(args.mps_sm),
        mig_devices=_parse_csv(args.mig_devices),
        presets=tuple(presets),
        model_only_input=args.model_only_input,
        env_only_action=args.env_only_action,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        num_envs_override=args.num_envs,
        pipeline_queue_timeout_s=args.pipeline_queue_timeout_s,
        pipeline_run_timeout_s=args.pipeline_run_timeout_s,
        cpu_bind_cores=args.cpu_bind_cores or None,
        cpu_bind_strategy=cpu_bind_strategy,
        cpu_bind_config=args.cpu_bind_config,
        cpu_bind_strict=bool(args.cpu_bind_strict),
    )


def main(argv: list[str] | None = None) -> None:
    """Parse CLI, run benchmark orchestrator, and print summary."""
    args = parse_args(argv)
    request = build_request(args)
    summary = run_benchmark_orchestrator(request)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
