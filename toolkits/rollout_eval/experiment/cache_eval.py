"""Cache-aware model adapter and two-pass cache evaluation."""

from __future__ import annotations

import importlib
from dataclasses import asdict, is_dataclass
from types import ModuleType
from typing import Any

import torch

from toolkits.rollout_eval.adapters.model_adapter import GenericModelAdapter
from toolkits.rollout_eval.config_bridge import EvalRuntimeConfig
from toolkits.rollout_eval.experiment.recording_loop import run_recording_loop
from toolkits.rollout_eval.experiment.seedable_env_adapter import SeedableEnvAdapter
from toolkits.rollout_eval.experiment.types import (
    CacheEvalResult,
    EpisodeTrajectory,
    ExperimentCacheConfig,
)

_FEATURE_CACHE_MODULE = "rlinf.models.embodiment.feature_cache"
_FEATURE_CACHE_CONFIG_FIELDS = {
    "enabled",
    "mode",
    "similarity_metric",
    "similarity_threshold",
    "invalidate_on_weight_update",
    "max_cache_seeds",
    "max_entries",
    "debug_log",
    "debug_log_max_events",
}
_FEATURE_CACHE_UNAVAILABLE_MESSAGE = (
    "Feature cache runtime is unavailable: "
    "feature cache runtime is unavailable in this installation"
)


def _load_feature_cache_runtime() -> ModuleType | None:
    """Import the runtime feature cache module if it is installed."""
    try:
        return importlib.import_module(_FEATURE_CACHE_MODULE)
    except Exception:
        return None


def _is_feature_cache_runtime_available() -> bool:
    """Return whether the runtime feature cache implementation can be used."""
    runtime = _load_feature_cache_runtime()
    return runtime is not None and all(
        hasattr(runtime, attr)
        for attr in ("CacheStats", "FeatureCache", "FeatureCacheConfig")
    )


def _cache_config_kwargs(cache_config: Any) -> dict[str, Any]:
    """Extract fields accepted by the runtime FeatureCacheConfig."""
    if is_dataclass(cache_config):
        data = asdict(cache_config)
    elif isinstance(cache_config, dict):
        data = dict(cache_config)
    else:
        data = {
            field_name: getattr(cache_config, field_name)
            for field_name in _FEATURE_CACHE_CONFIG_FIELDS
            if hasattr(cache_config, field_name)
        }
    return {
        key: value
        for key, value in data.items()
        if key in _FEATURE_CACHE_CONFIG_FIELDS
    }


def _to_runtime_cache_config(runtime: ModuleType, cache_config: Any) -> Any:
    """Convert toolkit or duck-typed cache config to runtime config."""
    FeatureCacheConfig = runtime.FeatureCacheConfig
    if isinstance(cache_config, FeatureCacheConfig):
        return cache_config
    return FeatureCacheConfig(**_cache_config_kwargs(cache_config))


class CacheAwareModelAdapter:
    """Wraps GenericModelAdapter to drive the existing FeatureCache.

    Before each episode the orchestrator must call ``set_seed(seed)`` so that
    the cache can key on ``(seed, step)``.
    """

    def __init__(
        self,
        inner: GenericModelAdapter,
        cache_config: ExperimentCacheConfig | Any,
    ):
        if not _is_feature_cache_runtime_available():
            raise RuntimeError(_FEATURE_CACHE_UNAVAILABLE_MESSAGE)
        self.inner = inner
        self.cache_config = cache_config
        self._step_counter = 0
        self._current_seed = 0
        self._configure_cache()

    def _configure_cache(self) -> None:
        runtime = _load_feature_cache_runtime()
        if runtime is None:
            raise RuntimeError(_FEATURE_CACHE_UNAVAILABLE_MESSAGE)
        FeatureCache = runtime.FeatureCache
        self.cache_config = _to_runtime_cache_config(runtime, self.cache_config)
        model = self.inner.model
        model.feature_cache_config = self.cache_config
        if not hasattr(model, "feature_cache") or model.feature_cache is None:
            model.feature_cache = FeatureCache(self.cache_config)

    def set_seed(self, seed: int) -> None:
        self._current_seed = seed
        self._step_counter = 0

    def infer(
        self, obs_batch: dict[str, Any], mode: str = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        self.inner.model.current_seed = self._current_seed
        self.inner.model.current_step = self._step_counter
        self._step_counter += 1
        return self.inner.infer(obs_batch=obs_batch, mode=mode)

    def get_cache_stats(self) -> Any:
        return self.inner.model.feature_cache.get_stats()

    def reset_cache_stats(self) -> None:
        self.inner.model.feature_cache.reset_stats()

    def invalidate_cache(self) -> None:
        self.inner.model.feature_cache.invalidate_all()


# ---------------------------------------------------------------------------
# Two-pass evaluation
# ---------------------------------------------------------------------------

def run_cache_eval(
    env_adapter: SeedableEnvAdapter,
    model_adapter: CacheAwareModelAdapter,
    runtime: EvalRuntimeConfig,
    seeds: list[int],
) -> tuple[CacheEvalResult, list[EpisodeTrajectory], list[EpisodeTrajectory]]:
    """Run two-pass cache evaluation.

    Pass 1 populates the cache (all misses).
    Pass 2 measures cache hits and action divergence.

    Returns:
        (CacheEvalResult, pass1_trajectories, pass2_trajectories)
    """
    # Pass 1: populate cache
    model_adapter.invalidate_cache()
    model_adapter.reset_cache_stats()

    pass1_trajectories: list[EpisodeTrajectory] = []
    pass1_latencies: list[float] = []

    for seed in seeds:
        model_adapter.set_seed(seed)
        result, trajs = run_recording_loop(env_adapter, model_adapter, runtime, seed=seed)
        pass1_trajectories.extend(trajs)
        if result.latency.model_infer_count > 0:
            pass1_latencies.append(
                result.latency.model_infer_seconds / result.latency.model_infer_count * 1000
            )

    # Pass 2: measure cache hits
    model_adapter.reset_cache_stats()

    pass2_trajectories: list[EpisodeTrajectory] = []
    pass2_latencies: list[float] = []

    for seed in seeds:
        model_adapter.set_seed(seed)
        result, trajs = run_recording_loop(env_adapter, model_adapter, runtime, seed=seed)
        pass2_trajectories.extend(trajs)
        if result.latency.model_infer_count > 0:
            pass2_latencies.append(
                result.latency.model_infer_seconds / result.latency.model_infer_count * 1000
            )

    pass2_stats = model_adapter.get_cache_stats()

    # Compute metrics
    total_pass2 = pass2_stats.hits + pass2_stats.misses
    hit_rate = pass2_stats.hits / max(total_pass2, 1)
    same_step_hit_rate = pass2_stats.same_step_hits / max(total_pass2, 1)

    avg_lat_no_cache = sum(pass1_latencies) / max(len(pass1_latencies), 1)
    avg_lat_with_cache = sum(pass2_latencies) / max(len(pass2_latencies), 1)
    savings = (
        (avg_lat_no_cache - avg_lat_with_cache) / max(avg_lat_no_cache, 1e-9) * 100
    )

    # Action divergence
    l2_values = _compute_action_divergence(pass1_trajectories, pass2_trajectories)
    l2_mean = sum(l2_values) / max(len(l2_values), 1)
    l2_max = max(l2_values) if l2_values else 0.0

    # Success rates
    sr1 = _success_rate(pass1_trajectories)
    sr2 = _success_rate(pass2_trajectories)

    eval_result = CacheEvalResult(
        cache_mode=model_adapter.cache_config.mode,
        hit_rate=hit_rate,
        same_step_hit_rate=same_step_hit_rate,
        latency_with_cache_ms=avg_lat_with_cache,
        latency_without_cache_ms=avg_lat_no_cache,
        latency_savings_pct=savings,
        action_divergence_l2_mean=l2_mean,
        action_divergence_l2_max=l2_max,
        success_rate_pass1=sr1,
        success_rate_pass2=sr2,
    )
    return eval_result, pass1_trajectories, pass2_trajectories


def _compute_action_divergence(
    pass1: list[EpisodeTrajectory],
    pass2: list[EpisodeTrajectory],
) -> list[float]:
    """Compute per-step L2 divergence between pass1 and pass2 actions."""
    values: list[float] = []
    for t1, t2 in zip(pass1, pass2):
        min_len = min(len(t1.steps), len(t2.steps))
        for i in range(min_len):
            a1 = t1.steps[i].action.float().flatten()
            a2 = t2.steps[i].action.float().flatten()
            if a1.shape == a2.shape:
                values.append(torch.norm(a1 - a2).item())
    return values


def _success_rate(trajectories: list[EpisodeTrajectory]) -> float:
    if not trajectories:
        return 0.0
    return sum(1 for t in trajectories if t.success) / len(trajectories)
