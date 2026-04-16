"""Entrypoint for lightweight non-Ray rollout evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from rlinf.config import validate_cfg
from toolkits.rollout_eval.adapters import build_env_adapter, build_model_adapter, build_null_model_adapter
from toolkits.rollout_eval.config_bridge import build_eval_runtime_config
from toolkits.rollout_eval.engine import run_rollout_loop
from toolkits.rollout_eval.reporting import (
    dump_report_json,
    dump_report_markdown,
    dump_batch_sweep_json,
    dump_batch_sweep_markdown,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight non-Ray rollout eval")
    parser.add_argument("--config-path", required=True, help="Hydra config directory")
    parser.add_argument("--config-name", required=True, help="Hydra config name")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override, repeatable. Example: --override env.eval.total_num_envs=8",
    )
    parser.add_argument(
        "--output-dir",
        default="./rollout_eval_output",
        help="Output directory for json/markdown reports",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print normalized runtime config and exit",
    )
    parser.add_argument(
        "--skip-validate-cfg",
        action="store_true",
        help="Skip rlinf.config.validate_cfg",
    )
    parser.add_argument(
        "--disable-profiler",
        action="store_true",
        help="Disable torch profiler collection",
    )
    parser.add_argument(
        "--disable-stage-split",
        action="store_true",
        help="Disable backbone/action-head stage split hooks",
    )
    parser.add_argument(
        "--profile-batch-sizes",
        default=None,
        help=(
            "Comma-separated list of batch sizes (num_envs) to sweep over for "
            "inference profiling. Example: --profile-batch-sizes 1,4,8,16. "
            "The model is loaded once and reused; the environment is rebuilt per batch size."
        ),
    )
    parser.add_argument(
        "--env-only",
        action="store_true",
        help=(
            "Profile environment steps only, without loading any model. "
            "Returns zero actions each step. Useful for measuring pure env throughput. "
            "Also pass --skip-validate-cfg to skip model path checks in config."
        ),
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=None,
        help=(
            "Action dimension for --env-only mode. "
            "Inferred from actor.model.action_dim in config if not provided."
        ),
    )
    return parser.parse_args()



def _load_cfg(config_path: str, config_name: str, overrides: list[str]):
    abs_config_path = str(Path(config_path).resolve())
    with initialize_config_dir(version_base="1.1", config_dir=abs_config_path):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg



def _build_single_report(
    cfg,
    runtime,
    model_adapter,
    output_dir: Path,
    *,
    batch_size: int | None = None,
) -> dict:
    """Run one rollout loop and return the report dict."""
    if batch_size is not None:
        from omegaconf import OmegaConf

        eval_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        with open_dict(eval_cfg):
            eval_cfg.env.eval.total_num_envs = batch_size
        runtime = build_eval_runtime_config(eval_cfg)
        with open_dict(eval_cfg):
            eval_cfg.rollout_eval = OmegaConf.to_container(
                cfg.get("rollout_eval", {}), resolve=True
            )
    else:
        eval_cfg = cfg

    env_adapter = build_env_adapter(eval_cfg, split="eval", profile_output_dir=runtime.profiler_output_dir)
    result = run_rollout_loop(
        env_adapter=env_adapter,
        model_adapter=model_adapter,
        runtime=runtime,
    )

    report = {
        "status": "pass",
        "env_type": runtime.env_type,
        "model_type": runtime.model_type,
        "model_path": runtime.model_path,
        "num_envs": runtime.num_envs,
        "total_steps": result.total_steps,
        "warmup_steps": result.warmup_steps,
        "measure_steps": result.measure_steps,
        "latency": {
            "model_infer_seconds": result.latency.model_infer_seconds,
            "env_step_seconds": result.latency.env_step_seconds,
            "model_infer_count": result.latency.model_infer_count,
            "env_step_count": result.latency.env_step_count,
        },
        "torch_profile": result.profile_metrics,
    }
    return report


def main() -> None:
    args = _parse_args()

    # Clear sys.argv so that spawned subprocesses (e.g. IsaacSim/OmniGibson)
    # don't pick up our CLI args and pass them to their kit application.
    sys.argv = sys.argv[:1]

    cfg = _load_cfg(args.config_path, args.config_name, args.override)

    # For behavior env, inject omnigibson_cfg even when skipping validation
    if args.skip_validate_cfg:
        env_type = cfg.get("env", {}).get("eval", {}).get("env_type", "")
        if env_type == "behavior":
            try:
                import omnigibson as og
                import yaml
                from omegaconf import OmegaConf

                config_filename = os.path.join(og.example_config_path, "r1pro_behavior.yaml")
                omnigibson_cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
                omnigibson_cfg = OmegaConf.create(omnigibson_cfg)
                with open_dict(omnigibson_cfg):
                    omnigibson_cfg.robots[0].obs_modalities = ["rgb", "depth", "proprio"]
                with open_dict(cfg):
                    cfg.env.train.omnigibson_cfg = omnigibson_cfg
                    cfg.env.eval.omnigibson_cfg = omnigibson_cfg
            except Exception as e:
                print(f"Warning: Failed to inject omnigibson_cfg for behavior env: {e}")

    if not args.skip_validate_cfg:
        cfg = validate_cfg(cfg)

    if "rollout_eval" not in cfg:
        with open_dict(cfg):
            cfg.rollout_eval = {}

    output_dir = Path(args.output_dir)

    with open_dict(cfg):
        cfg.rollout_eval.enable_torch_profiler = not args.disable_profiler
        cfg.rollout_eval.split_model_stages = not args.disable_stage_split
        cfg.rollout_eval.profiler_output_dir = str(output_dir / "profiler")

    runtime = build_eval_runtime_config(cfg)

    if args.dry_run:
        print(json.dumps(runtime.__dict__, indent=2, ensure_ascii=False))
        return

    # Batch-size sweep mode
    if args.profile_batch_sizes is not None:
        try:
            batch_sizes = [int(x.strip()) for x in args.profile_batch_sizes.split(",") if x.strip()]
        except ValueError as exc:
            raise SystemExit(f"--profile-batch-sizes must be comma-separated integers: {exc}")

        if args.env_only:
            model_adapter = build_null_model_adapter(cfg, action_dim_override=args.action_dim)
        else:
            model_adapter = build_model_adapter(
                cfg, split_model_stages=runtime.split_model_stages
            )

        sweep_results = []
        for bs in batch_sizes:
            print(f"\n[batch-sweep] num_envs={bs} ...")
            report = _build_single_report(cfg, runtime, model_adapter, output_dir, batch_size=bs)
            sweep_results.append(report)
            dump_report_json(report, str(output_dir / f"report_bs{bs}.json"))
            dump_report_markdown(report, str(output_dir / f"report_bs{bs}.md"))

        dump_batch_sweep_json(sweep_results, str(output_dir / "batch_sweep.json"))
        dump_batch_sweep_markdown(sweep_results, str(output_dir / "batch_sweep.md"))
        print(json.dumps(sweep_results, indent=2, ensure_ascii=False))
        return

    # Single-run mode
    env_adapter = build_env_adapter(cfg, split="eval", profile_output_dir=runtime.profiler_output_dir)
    if args.env_only:
        model_adapter = build_null_model_adapter(cfg, action_dim_override=args.action_dim)
    else:
        model_adapter = build_model_adapter(
            cfg, split_model_stages=runtime.split_model_stages
        )
    result = run_rollout_loop(
        env_adapter=env_adapter,
        model_adapter=model_adapter,
        runtime=runtime,
    )

    report = {
        "status": "pass",
        "env_type": runtime.env_type,
        "model_type": runtime.model_type,
        "model_path": runtime.model_path,
        "num_envs": runtime.num_envs,
        "total_steps": result.total_steps,
        "warmup_steps": result.warmup_steps,
        "measure_steps": result.measure_steps,
        "latency": {
            "model_infer_seconds": result.latency.model_infer_seconds,
            "env_step_seconds": result.latency.env_step_seconds,
            "model_infer_count": result.latency.model_infer_count,
            "env_step_count": result.latency.env_step_count,
        },
        "torch_profile": result.profile_metrics,
    }

    dump_report_json(report, str(output_dir / "report.json"))
    dump_report_markdown(report, str(output_dir / "report.md"))

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
