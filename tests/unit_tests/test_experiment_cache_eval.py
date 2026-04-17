"""Tests for cache evaluation module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from toolkits.rollout_eval.experiment.cache_eval import (
    CacheAwareModelAdapter,
    _compute_action_divergence,
    _success_rate,
)
from toolkits.rollout_eval.experiment.types import (
    EpisodeTrajectory,
    ExperimentCacheConfig,
    StepRecord,
)

# -----------------------------------------------------------------------
# CacheAwareModelAdapter
# -----------------------------------------------------------------------

def _make_mock_inner_model():
    """Create a mock GenericModelAdapter with a mock model."""
    inner = MagicMock()
    model = MagicMock()
    model.feature_cache = None
    model.feature_cache_config = None
    inner.model = model
    inner.infer.return_value = (torch.zeros(1, 7), {})
    return inner


def _require_feature_cache_runtime() -> None:
    """Skip cache-positive tests when runtime cache support is absent."""
    import toolkits.rollout_eval.experiment.cache_eval as cache_eval_module

    if not cache_eval_module._is_feature_cache_runtime_available():
        pytest.skip("feature cache runtime unavailable")


class TestCacheAwareModelAdapter:
    def test_configure_cache(self):
        _require_feature_cache_runtime()
        inner = _make_mock_inner_model()
        cfg = ExperimentCacheConfig(enabled=True, mode="naive")
        adapter = CacheAwareModelAdapter(inner, cfg)

        assert adapter.cache_config.mode == cfg.mode
        assert inner.model.feature_cache_config.mode == cfg.mode
        assert inner.model.feature_cache is not None

    def test_set_seed_resets_step(self):
        _require_feature_cache_runtime()
        inner = _make_mock_inner_model()
        cfg = ExperimentCacheConfig(enabled=True, mode="naive")
        adapter = CacheAwareModelAdapter(inner, cfg)

        adapter.set_seed(42)
        assert adapter._current_seed == 42
        assert adapter._step_counter == 0

    def test_infer_increments_step(self):
        _require_feature_cache_runtime()
        inner = _make_mock_inner_model()
        cfg = ExperimentCacheConfig(enabled=True, mode="naive")
        adapter = CacheAwareModelAdapter(inner, cfg)
        adapter.set_seed(10)

        adapter.infer({"img": torch.zeros(1, 3, 64, 64)})
        assert adapter._step_counter == 1
        assert inner.model.current_seed == 10
        assert inner.model.current_step == 0

        adapter.infer({"img": torch.zeros(1, 3, 64, 64)})
        assert adapter._step_counter == 2
        assert inner.model.current_step == 1

    def test_get_cache_stats(self):
        _require_feature_cache_runtime()
        inner = _make_mock_inner_model()
        cfg = ExperimentCacheConfig(enabled=True, mode="naive")
        adapter = CacheAwareModelAdapter(inner, cfg)

        stats = adapter.get_cache_stats()
        assert stats.hits == 0
        assert stats.misses == 0

    def test_invalidate_cache(self):
        _require_feature_cache_runtime()
        inner = _make_mock_inner_model()
        cfg = ExperimentCacheConfig(enabled=True, mode="naive")
        adapter = CacheAwareModelAdapter(inner, cfg)
        adapter.invalidate_cache()
        # Should not raise

    def test_runtime_cache_unavailable(self, monkeypatch):
        import toolkits.rollout_eval.experiment.cache_eval as cache_eval_module

        inner = _make_mock_inner_model()
        cfg = ExperimentCacheConfig(enabled=True, mode="naive")

        monkeypatch.setattr(
            cache_eval_module,
            "_is_feature_cache_runtime_available",
            lambda: False,
            raising=False,
        )

        with pytest.raises(RuntimeError, match="feature cache runtime is unavailable"):
            cache_eval_module.CacheAwareModelAdapter(inner, cache_config=cfg)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_traj(actions: list[torch.Tensor], success: bool = True) -> EpisodeTrajectory:
    steps = [
        StepRecord(step=i, obs={}, action=a, reward=1.0,
                   terminated=False, truncated=False, info={},
                   model_latency_ms=0, env_latency_ms=0)
        for i, a in enumerate(actions)
    ]
    return EpisodeTrajectory(seed=42, env_index=0, steps=steps,
                             success=success, total_reward=float(len(actions)))


class TestActionDivergence:
    def test_identical(self):
        a = [torch.tensor([1.0, 2.0])] * 3
        t1 = _make_traj(a)
        t2 = _make_traj(a)
        vals = _compute_action_divergence([t1], [t2])
        assert len(vals) == 3
        assert all(v == 0.0 for v in vals)

    def test_different(self):
        a1 = [torch.tensor([1.0, 0.0])]
        a2 = [torch.tensor([0.0, 1.0])]
        vals = _compute_action_divergence([_make_traj(a1)], [_make_traj(a2)])
        assert len(vals) == 1
        assert vals[0] > 0


class TestSuccessRate:
    def test_all_success(self):
        trajs = [_make_traj([], success=True)] * 3
        assert _success_rate(trajs) == 1.0

    def test_mixed(self):
        trajs = [_make_traj([], success=True), _make_traj([], success=False)]
        assert _success_rate(trajs) == 0.5

    def test_empty(self):
        assert _success_rate([]) == 0.0
