"""Tests for recording loop and determinism verification."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
import pytest

from toolkits.rollout_eval.config_bridge import EvalRuntimeConfig
from toolkits.rollout_eval.experiment.determinism import verify_determinism
from toolkits.rollout_eval.experiment.recording_loop import run_recording_loop
from toolkits.rollout_eval.experiment.seedable_env_adapter import SeedableEnvAdapter
from toolkits.rollout_eval.experiment.types import (
    DeterminismResult,
    EpisodeTrajectory,
    StepRecord,
)
from toolkits.rollout_eval.rollout_types import EnvStepResult


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_runtime(total_steps: int = 5) -> EvalRuntimeConfig:
    return EvalRuntimeConfig(
        env_type="mock", model_type="mock", model_path="/tmp/mock",
        num_envs=1, group_size=1, num_action_chunks=1,
        total_steps=total_steps, warmup_steps=1, seed=42,
    )


def _make_mock_env_adapter(
    total_steps: int = 5,
    terminate_at: int | None = None,
) -> SeedableEnvAdapter:
    """Build a SeedableEnvAdapter backed by mocks."""
    inner = MagicMock()
    inner.env = MagicMock()
    inner.env.seed = 0

    obs = {"img": torch.zeros(1, 3, 64, 64)}
    inner.reset.return_value = (obs, {})

    step_counter = {"n": 0}

    def _step(actions):
        idx = step_counter["n"]
        step_counter["n"] += 1
        terminated = terminate_at is not None and idx == terminate_at
        return EnvStepResult(
            obs={"img": torch.zeros(1, 3, 64, 64)},
            reward=torch.tensor([1.0]),
            terminated=torch.tensor([terminated]),
            truncated=torch.tensor([False]),
            info={"success": terminated},
        )

    inner.step.side_effect = _step
    return SeedableEnvAdapter(inner, seeds=[42])


def _make_mock_model():
    model = MagicMock()
    model.infer.return_value = (torch.zeros(1, 7), {})
    return model


# -----------------------------------------------------------------------
# Recording loop
# -----------------------------------------------------------------------

class TestRecordingLoop:
    def test_basic_run(self):
        env = _make_mock_env_adapter(total_steps=3)
        model = _make_mock_model()
        runtime = _make_runtime(total_steps=3)

        result, trajectories = run_recording_loop(env, model, runtime, seed=42)

        assert result.total_steps == 3
        # No termination → single partial trajectory
        assert len(trajectories) == 1
        assert len(trajectories[0].steps) == 3
        assert trajectories[0].seed == 42

    def test_episode_boundary(self):
        env = _make_mock_env_adapter(total_steps=5, terminate_at=2)
        model = _make_mock_model()
        runtime = _make_runtime(total_steps=5)

        result, trajectories = run_recording_loop(env, model, runtime, seed=42)

        # Step 2 terminates → episode 1 has 3 steps (0,1,2), then re-reset
        assert len(trajectories) >= 2
        assert trajectories[0].success is True  # terminated with success=True
        assert len(trajectories[0].steps) == 3

    def test_step_records_have_latency(self):
        env = _make_mock_env_adapter(total_steps=2)
        model = _make_mock_model()
        runtime = _make_runtime(total_steps=2)

        _, trajectories = run_recording_loop(env, model, runtime, seed=42)
        for step in trajectories[0].steps:
            assert step.model_latency_ms >= 0
            assert step.env_latency_ms >= 0


# -----------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------

def _make_trajectory(seed: int, actions: list[torch.Tensor], rewards: list[float]) -> EpisodeTrajectory:
    steps = []
    for i, (a, r) in enumerate(zip(actions, rewards)):
        steps.append(StepRecord(
            step=i, obs={"x": torch.tensor([float(i)])}, action=a,
            reward=r, terminated=False, truncated=False, info={},
            model_latency_ms=0.0, env_latency_ms=0.0,
        ))
    return EpisodeTrajectory(seed=seed, env_index=0, steps=steps,
                             success=True, total_reward=sum(rewards))


class TestDeterminism:
    def test_identical_runs(self):
        actions = [torch.tensor([1.0, 2.0, 3.0])] * 3
        rewards = [1.0, 1.0, 1.0]
        t1 = _make_trajectory(42, actions, rewards)
        t2 = _make_trajectory(42, actions, rewards)

        results = verify_determinism({42: [t1, t2]})
        assert results[42].action_match is True
        assert results[42].max_action_l2 == 0.0
        assert results[42].reward_match is True

    def test_divergent_actions(self):
        a1 = [torch.tensor([1.0, 0.0])] * 3
        a2 = [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
        t1 = _make_trajectory(42, a1, [1.0] * 3)
        t2 = _make_trajectory(42, a2, [1.0] * 3)

        results = verify_determinism({42: [t1, t2]})
        assert results[42].action_match is False
        assert results[42].max_action_l2 > 0

    def test_single_run_trivially_deterministic(self):
        t1 = _make_trajectory(42, [torch.zeros(3)], [1.0])
        results = verify_determinism({42: [t1]})
        assert results[42].action_match is True

    def test_reward_mismatch(self):
        actions = [torch.tensor([1.0])] * 2
        t1 = _make_trajectory(42, actions, [1.0, 2.0])
        t2 = _make_trajectory(42, actions, [1.0, 3.0])

        results = verify_determinism({42: [t1, t2]})
        assert results[42].reward_match is False
        assert results[42].action_match is True
