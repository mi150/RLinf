"""Entrypoint for lightweight non-Ray rollout evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from rlinf.config import validate_cfg
from toolkits.rollout_eval.adapters import build_env_adapter, build_model_adapter
from toolkits.rollout_eval.config_bridge import build_eval_runtime_config
from toolkits.rollout_eval.engine import run_rollout_loop
from toolkits.rollout_eval.reporting import dump_report_json, dump_report_markdown


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
    return parser.parse_args()



def _load_cfg(config_path: str, config_name: str, overrides: list[str]):
    abs_config_path = str(Path(config_path).resolve())
    with initialize_config_dir(version_base="1.1", config_dir=abs_config_path):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg



def main() -> None:
    args = _parse_args()
    cfg = _load_cfg(args.config_path, args.config_name, args.override)
    if not args.skip_validate_cfg:
        cfg = validate_cfg(cfg)

    if "rollout_eval" not in cfg:
        with open_dict(cfg):
            cfg.rollout_eval = {}

    with open_dict(cfg):
        cfg.rollout_eval.enable_torch_profiler = not args.disable_profiler
        cfg.rollout_eval.split_model_stages = not args.disable_stage_split
        cfg.rollout_eval.profiler_output_dir = str(
            Path(args.output_dir) / "profiler"
        )

    runtime = build_eval_runtime_config(cfg)

    if args.dry_run:
        print(json.dumps(runtime.__dict__, indent=2, ensure_ascii=False))
        return

    env_adapter = build_env_adapter(cfg, split="eval")
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

    output_dir = Path(args.output_dir)
    dump_report_json(report, str(output_dir / "report.json"))
    dump_report_markdown(report, str(output_dir / "report.md"))

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
