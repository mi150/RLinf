"""Tests for experiment pipeline types and seedable env adapter."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
import pytest

from toolkits.rollout_eval.experiment.types import (
    CacheEvalResult,
    DeterminismResult,
    EpisodeTrajectory,
    ExperimentConfig,
    LoadedTrajectory,
    StepRecord,
)
from toolkits.rollout_eval.experiment.seedable_env_adapter import SeedableEnvAdapter
from toolkits.rollout_eval.config_bridge import EvalRuntimeConfig
from toolkits.rollout_eval.rollout_types import EnvStepResult


# -----------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------

class TestStepRecord:
    def test_fields(self):
        rec = StepRecord(
            step=0,
            obs={"img": torch.zeros(1)},
            action=torch.ones(7),
            reward=1.0,
            terminated=False,
            truncated=False,
            info={},
            model_latency_ms=10.0,
            env_latency_ms=5.0,
        )
        assert rec.step == 0
        assert rec.reward == 1.0
        assert rec.meta == {}


class TestEpisodeTrajectory:
    def test_basic(self):
        traj = EpisodeTrajectory(
            seed=42, env_index=0, steps=[], success=True, total_reward=10.0
        )
        assert traj.seed == 42
        assert traj.video_path is None


class TestLoadedTrajectory:
    def test_basic(self):
        lt = LoadedTrajectory(
            seed=1, rank=0, env_index=0, episode=0, success=True,
            actions=[torch.zeros(7)], observations=[{}], rewards=[1.0],
            source_path="/tmp/test.pkl",
        )
        assert len(lt.actions) == 1


class TestDeterminismResult:
    def test_basic(self):
        dr = DeterminismResult(seed=42, action_match=True, max_action_l2=0.0,
                               obs_match=True, reward_match=True)
        assert dr.action_match


class TestCacheEvalResult:
    def test_basic(self):
        cr = CacheEvalResult(
            cache_mode="naive", hit_rate=0.8, same_step_hit_rate=0.8,
            latency_with_cache_ms=10.0, latency_without_cache_ms=40.0,
            latency_savings_pct=75.0, action_divergence_l2_mean=0.001,
            action_divergence_l2_max=0.01, success_rate_pass1=0.5,
            success_rate_pass2=0.5,
        )
        assert cr.latency_savings_pct == 75.0


class TestExperimentConfig:
    def test_defaults(self):
        rt = EvalRuntimeConfig(
            env_type="maniskill", model_type="openvla_oft",
            model_path="/tmp/model", num_envs=1, group_size=1,
            num_action_chunks=1, total_steps=10, warmup_steps=2, seed=42,
        )
        cfg = ExperimentConfig(eval_runtime=rt)
        assert cfg.phases == ["baseline"]
        assert cfg.num_runs_per_seed == 2
        assert cfg.record_video is True
        assert cfg.action_source == "pipeline"


# -----------------------------------------------------------------------
# SeedableEnvAdapter
# -----------------------------------------------------------------------

def _make_mock_inner(seed_attr: bool = True):
    """Create a mock GenericEnvAdapter with a mock env."""
    inner = MagicMock()
    env = MagicMock()
    if seed_attr:
        env.seed = 0
    else:
        del env.seed
    inner.env = env
    inner.reset.return_value = ({"img": torch.zeros(1, 3, 64, 64)}, {})
    inner.step.return_value = EnvStepResult(
        obs={"img": torch.zeros(1, 3, 64, 64)},
        reward=torch.tensor([1.0]),
        terminated=torch.tensor([False]),
        truncated=torch.tensor([False]),
        info={},
    )
    return inner


class TestSeedableEnvAdapter:
    def test_reset_with_explicit_seed(self):
        inner = _make_mock_inner()
        adapter = SeedableEnvAdapter(inner, seeds=[10, 20])
        obs, info = adapter.reset(seed=99)
        assert inner.env.seed == 99
        assert adapter.current_seed == 99

    def test_reset_cycles_through_seeds(self):
        inner = _make_mock_inner()
        adapter = SeedableEnvAdapter(inner, seeds=[10, 20, 30])
        adapter.reset()
        assert adapter.current_seed == 10
        adapter.reset()
        assert adapter.current_seed == 20
        adapter.reset()
        assert adapter.current_seed == 30
        # wraps around
        adapter.reset()
        assert adapter.current_seed == 10

    def test_step_delegates(self):
        inner = _make_mock_inner()
        adapter = SeedableEnvAdapter(inner)
        action = torch.zeros(1, 7)
        result = adapter.step(action)
        inner.step.assert_called_once_with(action)
        assert isinstance(result, EnvStepResult)

    def test_close_delegates(self):
        inner = _make_mock_inner()
        adapter = SeedableEnvAdapter(inner)
        adapter.close()
        inner.close.assert_called_once()

    def test_env_property(self):
        inner = _make_mock_inner()
        adapter = SeedableEnvAdapter(inner)
        assert adapter.env is inner.env

    def test_set_seed_via_set_seed_method(self):
        inner = _make_mock_inner(seed_attr=False)
        inner.env.set_seed = MagicMock()
        adapter = SeedableEnvAdapter(inner, seeds=[42])
        adapter.reset()
        inner.env.set_seed.assert_called_once_with(42)
