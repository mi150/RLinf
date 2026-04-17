"""Pipeline orchestrator for progressive rollout evaluation experiments.

Usage:
    python -m toolkits.rollout_eval.experiment.run_experiment \
        --config-path /path/to/hydra/config \
        --config-name libero_spatial_ppo_gr00t \
        --phases baseline,cache_eval,action_replace \
        --seeds 42,43,44 \
        --num-runs-per-seed 3 \
        --output-dir ./experiment_output
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from toolkits.rollout_eval.experiment.types import (
    EpisodeTrajectory,
    ExperimentConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def _run_phase_baseline(
    cfg: ExperimentConfig,
    env_adapter: Any,
    model_adapter: Any,
) -> dict[int, list[EpisodeTrajectory]]:
    """Phase 1: Baseline collection with determinism verification."""
    from toolkits.rollout_eval.experiment.determinism import verify_determinism
    from toolkits.rollout_eval.experiment.recording_loop import run_recording_loop
    from toolkits.rollout_eval.experiment.reporting import dump_baseline_report

    all_trajectories: dict[int, list[EpisodeTrajectory]] = {}

    for seed in cfg.seeds:
        seed_trajs: list[EpisodeTrajectory] = []
        for run_idx in range(cfg.num_runs_per_seed):
            logger.info("Phase 1: seed=%d run=%d/%d", seed, run_idx + 1, cfg.num_runs_per_seed)
            _, trajs = run_recording_loop(
                env_adapter, model_adapter, cfg.eval_runtime, seed=seed,
            )
            seed_trajs.extend(trajs)
        all_trajectories[seed] = seed_trajs

    # Determinism check
    determinism = verify_determinism(all_trajectories)
    for seed, dr in determinism.items():
        if not dr.action_match:
            logger.warning("Seed %d: non-deterministic (max L2=%.6f)", seed, dr.max_action_l2)

    # Save trajectories
    traj_dir = Path(cfg.output_dir) / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    for seed, trajs in all_trajectories.items():
        for i, traj in enumerate(trajs):
            path = traj_dir / f"seed{seed}_run{i}.pkl"
            with open(path, "wb") as f:
                pickle.dump(traj, f)

    # Report
    dump_baseline_report(
        output_dir=cfg.output_dir,
        seeds=cfg.seeds,
        num_runs_per_seed=cfg.num_runs_per_seed,
        determinism=determinism,
        trajectories=all_trajectories,
    )
    logger.info("Phase 1 complete. Report: %s/reports/phase1_baseline.json", cfg.output_dir)
    return all_trajectories


def _run_phase_cache_eval(
    cfg: ExperimentConfig,
    env_adapter: Any,
    model_adapter: Any,
) -> None:
    """Phase 2: Feature cache two-pass evaluation."""
    from rlinf.models.embodiment.feature_cache import FeatureCacheConfig
    from toolkits.rollout_eval.experiment.cache_eval import (
        CacheAwareModelAdapter,
        run_cache_eval,
    )
    from toolkits.rollout_eval.experiment.reporting import dump_cache_report

    cache_config = cfg.cache_config or FeatureCacheConfig(enabled=True, mode="naive")
    cache_adapter = CacheAwareModelAdapter(model_adapter, cache_config)

    logger.info("Phase 2: cache mode=%s", cache_config.mode)
    result, _, _ = run_cache_eval(
        env_adapter, cache_adapter, cfg.eval_runtime, cfg.seeds,
    )

    dump_cache_report(
        output_dir=cfg.output_dir,
        cache_result=result,
        cache_config={"mode": cache_config.mode},
    )
    logger.info(
        "Phase 2 complete. Hit rate=%.1f%%, latency savings=%.1f%%",
        result.hit_rate * 100, result.latency_savings_pct,
    )


def _run_phase_action_replace(
    cfg: ExperimentConfig,
    env_adapter: Any,
    model_adapter: Any,
    baseline_trajectories: dict[int, list[EpisodeTrajectory]] | None = None,
) -> None:
    """Phase 3: Action replacement in bottleneck zone."""
    from toolkits.rollout_eval.experiment.action_replacer import run_action_replace_eval
    from toolkits.rollout_eval.experiment.bottleneck_detector import detect_bottleneck_k_b
    from toolkits.rollout_eval.experiment.reporting import dump_action_replace_report
    from toolkits.rollout_eval.experiment.trajectory_loader import scan_and_pair_trajectories

    # Determine K_B
    k_b_source = "static"
    k_b = cfg.bottleneck_k_b or 0

    if cfg.bottleneck_detect_online and baseline_trajectories:
        # Flatten all trajectories for detection
        all_trajs = [t for trajs in baseline_trajectories.values() for t in trajs]
        k_b = detect_bottleneck_k_b(all_trajs)
        k_b_source = "online"
        logger.info("Phase 3: online K_B detected = %d", k_b)

    if k_b == 0:
        logger.warning("Phase 3: K_B=0, no bottleneck zone to replace")
        return

    # Determine action source
    source_trajs: dict[int, EpisodeTrajectory] = {}

    if cfg.action_source == "external" and cfg.external_trajectory_dir:
        loaded = scan_and_pair_trajectories(
            cfg.external_trajectory_dir, cfg.external_trajectory_seeds,
        )
        # Convert LoadedTrajectory → EpisodeTrajectory for the replacer
        from toolkits.rollout_eval.experiment.types import StepRecord
        for seed, lt_list in loaded.items():
            if lt_list:
                lt = lt_list[0]
                steps = [
                    StepRecord(
                        step=i, obs={}, action=a, reward=r,
                        terminated=False, truncated=False, info={},
                        model_latency_ms=0, env_latency_ms=0,
                    )
                    for i, (a, r) in enumerate(zip(lt.actions, lt.rewards))
                ]
                source_trajs[seed] = EpisodeTrajectory(
                    seed=seed, env_index=0, steps=steps,
                    success=lt.success, total_reward=sum(lt.rewards),
                )
    elif baseline_trajectories:
        # Use first trajectory per seed from Phase 1
        for seed, trajs in baseline_trajectories.items():
            if trajs:
                source_trajs[seed] = trajs[0]

    logger.info("Phase 3: K_B=%d, source=%s, seeds=%s", k_b, cfg.action_source, cfg.seeds)

    if cfg.action_source == "external":
        # Pure open-loop replay — no model inference at all
        from toolkits.rollout_eval.experiment.action_replacer import OpenLoopReplayAdapter
        from toolkits.rollout_eval.experiment.recording_loop import run_recording_loop
        import dataclasses

        replaced: list[EpisodeTrajectory] = []

        # Reload ALL trajectories from dir (no seed filter) for individual replay
        all_loaded = scan_and_pair_trajectories(cfg.external_trajectory_dir)

        for seed, lt_list in all_loaded.items():
            for ep_idx, lt in enumerate(lt_list):
                actions = lt.actions
                replay_adapter = OpenLoopReplayAdapter(lt)

                replay_runtime = dataclasses.replace(
                    cfg.eval_runtime,
                    total_steps=len(actions),
                    warmup_steps=min(cfg.eval_runtime.warmup_steps, len(actions)),
                )

                video_path = None
                if cfg.record_video:
                    # Name: phase3_sid{seed}_rank{rank}_env{env}_ep{ep}.mp4
                    video_name = (
                        f"phase3_sid{lt.seed}_rank{lt.rank}"
                        f"_env{lt.env_index}_ep{lt.episode}.mp4"
                    )
                    video_path = str(Path(cfg.output_dir) / "videos" / video_name)

                _, trajs = run_recording_loop(
                    env_adapter, replay_adapter, replay_runtime, seed=seed,
                    video_path=video_path, video_fps=cfg.video_fps, k_b=k_b,
                )
                replaced.extend(trajs)
    else:
        replaced = run_action_replace_eval(
            env_adapter, model_adapter, cfg.eval_runtime,
            baseline_trajectories=source_trajs,
            k_b=k_b,
            seeds=cfg.seeds,
        )

    # Build baseline dict for reporting
    baseline_for_report = {
        seed: [source_trajs[seed]] for seed in source_trajs
    }

    dump_action_replace_report(
        output_dir=cfg.output_dir,
        k_b=k_b,
        k_b_source=k_b_source,
        action_source=cfg.action_source,
        total_steps=cfg.eval_runtime.total_steps,
        baseline_trajectories=baseline_for_report,
        replaced_trajectories=replaced,
    )
    logger.info("Phase 3 complete. Report: %s/reports/phase3_action_replace.json", cfg.output_dir)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_experiment(
    cfg: ExperimentConfig,
    env_adapter: Any,
    model_adapter: Any,
) -> None:
    """Run the full experiment pipeline."""
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    config_path = Path(cfg.output_dir) / "experiment_config.json"
    with open(config_path, "w") as f:
        json.dump(_safe_asdict(cfg), f, indent=2, default=str)

    baseline_trajectories = None

    for phase in cfg.phases:
        if phase == "baseline":
            baseline_trajectories = _run_phase_baseline(cfg, env_adapter, model_adapter)
        elif phase == "cache_eval":
            _run_phase_cache_eval(cfg, env_adapter, model_adapter)
        elif phase == "action_replace":
            _run_phase_action_replace(cfg, env_adapter, model_adapter, baseline_trajectories)
        else:
            logger.warning("Unknown phase: %s", phase)

    logger.info("Experiment complete. Output: %s", cfg.output_dir)


def _safe_asdict(cfg: ExperimentConfig) -> dict:
    """Convert config to dict, handling non-serialisable fields."""
    d: dict[str, Any] = {
        "phases": cfg.phases,
        "seeds": cfg.seeds,
        "num_runs_per_seed": cfg.num_runs_per_seed,
        "record_video": cfg.record_video,
        "video_fps": cfg.video_fps,
        "bottleneck_k_b": cfg.bottleneck_k_b,
        "bottleneck_detect_online": cfg.bottleneck_detect_online,
        "action_source": cfg.action_source,
        "external_trajectory_dir": cfg.external_trajectory_dir,
        "output_dir": cfg.output_dir,
    }
    return d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rollout eval experiment pipeline")
    parser.add_argument("--config-path", required=True, help="Hydra config directory")
    parser.add_argument("--config-name", required=True, help="Hydra config name")
    parser.add_argument("--override", nargs="*", default=[], help="Hydra overrides")
    parser.add_argument("--phases", default="baseline", help="Comma-separated phases")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds")
    parser.add_argument("--num-runs-per-seed", type=int, default=2)
    parser.add_argument("--cache-mode", default=None, help="Cache mode for Phase 2")
    parser.add_argument("--bottleneck-k-b", default=None, help="Static K_B or 'auto'")
    parser.add_argument("--action-source", default="pipeline",
                        choices=["pipeline", "cross_run", "external"])
    parser.add_argument("--external-trajectory-dir", default=None)
    parser.add_argument("--external-trajectory-seeds", default=None)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--output-dir", default="./experiment_output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    # Build Hydra config
    from toolkits.rollout_eval.config_bridge import build_eval_runtime_config
    from toolkits.rollout_eval.run import _load_cfg

    hydra_cfg = _load_cfg(args.config_path, args.config_name, args.override)

    # validate_cfg fills in derived fields like control_mode for ManiSkill
    # (same as run.py does unless --skip-validate-cfg is passed)
    try:
        from rlinf.config import validate_cfg
        hydra_cfg = validate_cfg(hydra_cfg)
    except Exception as e:
        logger.warning("validate_cfg skipped: %s", e)

    eval_runtime = build_eval_runtime_config(hydra_cfg)

    # Build adapters
    from toolkits.rollout_eval.adapters.env_adapter import build_env_adapter
    from toolkits.rollout_eval.experiment.seedable_env_adapter import SeedableEnvAdapter

    seeds = [int(s) for s in args.seeds.split(",")]
    phases = [p.strip() for p in args.phases.split(",")]

    inner_env = build_env_adapter(hydra_cfg, split="eval")
    env_adapter = SeedableEnvAdapter(inner_env, seeds=seeds)

    # Skip model loading when all phases use open-loop external replay
    _needs_model = not (
        phases == ["action_replace"] and args.action_source == "external"
    )
    if _needs_model:
        from toolkits.rollout_eval.adapters.model_adapter import build_model_adapter
        model_adapter = build_model_adapter(hydra_cfg, split_model_stages=False)
    else:
        logger.info("action_source=external: skipping model load")
        model_adapter = None

    # Build cache config if needed
    cache_config = None
    if args.cache_mode and "cache_eval" in phases:
        from rlinf.models.embodiment.feature_cache import FeatureCacheConfig
        cache_config = FeatureCacheConfig(enabled=True, mode=args.cache_mode)

    # K_B
    k_b = None
    detect_online = False
    if args.bottleneck_k_b == "auto":
        detect_online = True
    elif args.bottleneck_k_b is not None:
        k_b = int(args.bottleneck_k_b)

    ext_seeds = None
    if args.external_trajectory_seeds:
        ext_seeds = [int(s) for s in args.external_trajectory_seeds.split(",")]

    experiment_cfg = ExperimentConfig(
        eval_runtime=eval_runtime,
        phases=phases,
        num_runs_per_seed=args.num_runs_per_seed,
        seeds=seeds,
        record_video=args.record_video,
        cache_config=cache_config,
        bottleneck_k_b=k_b,
        bottleneck_detect_online=detect_online,
        action_source=args.action_source,
        external_trajectory_dir=args.external_trajectory_dir,
        external_trajectory_seeds=ext_seeds,
        output_dir=args.output_dir,
    )

    run_experiment(experiment_cfg, env_adapter, model_adapter)


if __name__ == "__main__":
    main()
