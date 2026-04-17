"""Data types for the experiment pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from rlinf.models.embodiment.feature_cache import FeatureCacheConfig
from toolkits.rollout_eval.config_bridge import EvalRuntimeConfig


# ---------------------------------------------------------------------------
# Per-step / per-episode recording
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """Single step captured during a recording loop."""

    step: int
    obs: dict[str, Any]
    action: torch.Tensor
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    model_latency_ms: float
    env_latency_ms: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeTrajectory:
    """Full episode trajectory for cross-run comparison."""

    seed: int
    env_index: int
    steps: list[StepRecord]
    success: bool
    total_reward: float
    video_path: str | None = None


# ---------------------------------------------------------------------------
# External trajectory (loaded from pkl)
# ---------------------------------------------------------------------------

@dataclass
class LoadedTrajectory:
    """Trajectory loaded from an external pkl file."""

    seed: int
    rank: int
    env_index: int
    episode: int
    success: bool
    actions: list[torch.Tensor]
    observations: list[Any]
    rewards: list[float]
    source_path: str


# ---------------------------------------------------------------------------
# Determinism verification
# ---------------------------------------------------------------------------

@dataclass
class DeterminismResult:
    """Result of comparing multiple runs with the same seed."""

    seed: int
    action_match: bool
    max_action_l2: float
    obs_match: bool
    reward_match: bool


# ---------------------------------------------------------------------------
# Cache evaluation
# ---------------------------------------------------------------------------

@dataclass
class CacheEvalResult:
    """Metrics from a two-pass cache evaluation."""

    cache_mode: str
    hit_rate: float
    same_step_hit_rate: float
    latency_with_cache_ms: float
    latency_without_cache_ms: float
    latency_savings_pct: float
    action_divergence_l2_mean: float
    action_divergence_l2_max: float
    success_rate_pass1: float
    success_rate_pass2: float


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Top-level config for the experiment pipeline."""

    eval_runtime: EvalRuntimeConfig

    # Phase control
    phases: list[str] = field(default_factory=lambda: ["baseline"])
    num_runs_per_seed: int = 2
    seeds: list[int] = field(default_factory=lambda: [42])

    # Video
    record_video: bool = True
    video_fps: int = 30

    # Phase 2: Cache
    cache_config: FeatureCacheConfig | None = None

    # Phase 3: Action replacement
    bottleneck_k_b: int | None = None
    bottleneck_detect_online: bool = False
    action_source: str = "pipeline"  # "pipeline" | "cross_run" | "external"
    external_trajectory_dir: str | None = None
    external_trajectory_seeds: list[int] | None = None

    # Output
    output_dir: str = "./experiment_output"
