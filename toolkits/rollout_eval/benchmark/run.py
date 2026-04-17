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
    )
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
    return parser.parse_args(argv)


def build_request(args: argparse.Namespace) -> BenchmarkRequest:
    """Convert parsed args into a typed benchmark request."""
    scenario_set = _parse_csv(args.scenario_set)
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
    )


def main(argv: list[str] | None = None) -> None:
    """Parse CLI, run benchmark orchestrator, and print summary."""
    args = parse_args(argv)
    request = build_request(args)
    summary = run_benchmark_orchestrator(request)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
