"""Report generation for experiment pipeline phases."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from toolkits.rollout_eval.experiment.types import (
    CacheEvalResult,
    DeterminismResult,
    EpisodeTrajectory,
)


def _serialisable(obj: Any) -> Any:
    """Make an object JSON-serialisable."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialisable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialisable(v) for v in obj]
    if hasattr(obj, "item"):  # numpy / torch scalar
        return obj.item()
    if hasattr(obj, "tolist"):  # tensor
        return obj.tolist()
    return obj


def _write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_serialisable(data), f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Phase 1: Baseline
# ---------------------------------------------------------------------------

def dump_baseline_report(
    output_dir: str | Path,
    seeds: list[int],
    num_runs_per_seed: int,
    determinism: dict[int, DeterminismResult],
    trajectories: dict[int, list[EpisodeTrajectory]],
    video_paths: list[str] | None = None,
) -> Path:
    """Write Phase 1 baseline report."""
    report: dict[str, Any] = {
        "phase": "baseline",
        "seeds": seeds,
        "num_runs_per_seed": num_runs_per_seed,
        "determinism": {},
        "per_seed_summary": {},
        "video_paths": video_paths or [],
    }

    for seed, dr in determinism.items():
        report["determinism"][seed] = {
            "action_match": dr.action_match,
            "max_action_l2": dr.max_action_l2,
            "obs_match": dr.obs_match,
            "reward_match": dr.reward_match,
        }

    for seed, trajs in trajectories.items():
        if not trajs:
            continue
        successes = sum(1 for t in trajs if t.success)
        rewards = [t.total_reward for t in trajs]
        lengths = [len(t.steps) for t in trajs]
        report["per_seed_summary"][seed] = {
            "num_episodes": len(trajs),
            "success_rate": successes / len(trajs),
            "avg_reward": sum(rewards) / len(rewards),
            "avg_episode_length": sum(lengths) / len(lengths),
        }

    path = Path(output_dir) / "reports" / "phase1_baseline.json"
    _write_json(report, path)
    return path


# ---------------------------------------------------------------------------
# Phase 2: Cache eval
# ---------------------------------------------------------------------------

def dump_cache_report(
    output_dir: str | Path,
    cache_result: CacheEvalResult,
    cache_config: dict[str, Any] | None = None,
    video_paths: list[str] | None = None,
) -> Path:
    """Write Phase 2 cache evaluation report."""
    report: dict[str, Any] = {
        "phase": "cache_eval",
        "cache_mode": cache_result.cache_mode,
        "cache_config": cache_config or {},
        "pass1": {
            "hit_rate": 0.0,
            "avg_model_latency_ms": cache_result.latency_without_cache_ms,
            "success_rate": cache_result.success_rate_pass1,
        },
        "pass2": {
            "hit_rate": cache_result.hit_rate,
            "avg_model_latency_ms": cache_result.latency_with_cache_ms,
            "success_rate": cache_result.success_rate_pass2,
        },
        "latency_savings_pct": cache_result.latency_savings_pct,
        "action_divergence": {
            "l2_mean": cache_result.action_divergence_l2_mean,
            "l2_max": cache_result.action_divergence_l2_max,
        },
        "video_paths": video_paths or [],
    }

    path = Path(output_dir) / "reports" / "phase2_cache_eval.json"
    _write_json(report, path)
    return path


def dump_cache_report_unsupported(
    output_dir: str | Path,
    reason: str,
) -> Path:
    """Write Phase 2 report when cache evaluation cannot run."""
    report: dict[str, Any] = {
        "phase": "cache_eval",
        "status": "unsupported",
        "reason": reason,
    }

    path = Path(output_dir) / "reports" / "phase2_cache_eval.json"
    _write_json(report, path)
    return path


# ---------------------------------------------------------------------------
# Phase 3: Action replacement
# ---------------------------------------------------------------------------

def dump_action_replace_report(
    output_dir: str | Path,
    k_b: int,
    k_b_source: str,
    action_source: str,
    total_steps: int,
    baseline_trajectories: dict[int, list[EpisodeTrajectory]],
    replaced_trajectories: list[EpisodeTrajectory],
    video_paths: list[str] | None = None,
) -> Path:
    """Write Phase 3 action replacement report."""
    report: dict[str, Any] = {
        "phase": "action_replace",
        "k_b": k_b,
        "k_b_source": k_b_source,
        "action_source": action_source,
        "replacement_zone": {
            "start_step": total_steps - k_b,
            "end_step": total_steps,
        },
        "per_seed": {},
        "aggregate": {},
        "video_paths": video_paths or [],
    }

    # Per-seed metrics
    baseline_success_count = 0
    replaced_success_count = 0
    all_l2 = []

    for traj in replaced_trajectories:
        seed = traj.seed
        baseline_trajs = baseline_trajectories.get(seed, [])
        baseline_success = any(t.success for t in baseline_trajs) if baseline_trajs else False

        # Collect replacement L2 from step metadata
        step_l2s = [
            s.meta.get("replacement_l2", 0.0)
            for s in traj.steps
            if s.meta.get("replaced", False)
        ]

        report["per_seed"][seed] = {
            "baseline_success": baseline_success,
            "replaced_success": traj.success,
            "replacement_l2_mean": sum(step_l2s) / max(len(step_l2s), 1),
            "replacement_l2_max": max(step_l2s) if step_l2s else 0.0,
            "reward_delta": traj.total_reward - (
                baseline_trajs[0].total_reward if baseline_trajs else 0.0
            ),
        }

        if baseline_success:
            baseline_success_count += 1
        if traj.success:
            replaced_success_count += 1
        all_l2.extend(step_l2s)

    total = max(len(replaced_trajectories), 1)
    baseline_sr = baseline_success_count / total
    replaced_sr = replaced_success_count / total

    report["aggregate"] = {
        "success_rate_baseline": baseline_sr,
        "success_rate_replaced": replaced_sr,
        "success_rate_delta": replaced_sr - baseline_sr,
        "avg_replacement_l2": sum(all_l2) / max(len(all_l2), 1),
    }

    path = Path(output_dir) / "reports" / "phase3_action_replace.json"
    _write_json(report, path)
    return path
