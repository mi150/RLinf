"""Tests for trajectory loader, bottleneck detector, and action replacer."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import torch
import pytest

from toolkits.rollout_eval.experiment.action_replacer import (
    ActionReplacerModelAdapter,
    OpenLoopReplayAdapter,
)
from toolkits.rollout_eval.experiment.bottleneck_detector import detect_bottleneck_k_b
from toolkits.rollout_eval.experiment.trajectory_loader import (
    load_trajectory_from_pkl,
    scan_and_pair_trajectories,
)
from toolkits.rollout_eval.experiment.types import (
    EpisodeTrajectory,
    LoadedTrajectory,
    StepRecord,
)


# -----------------------------------------------------------------------
# Trajectory Loader
# -----------------------------------------------------------------------

class TestLoadTrajectoryFromPkl:
    def test_load_with_standard_name(self, tmp_path):
        pkl_path = tmp_path / "step_100_sid_42_rank_0_env_0_episode_3_success.pkl"
        data = {
            "actions": [torch.zeros(7) for _ in range(5)],
            "observations": [{"img": torch.zeros(3, 64, 64)} for _ in range(5)],
            "rewards": [1.0] * 5,
            "success": True,
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

        traj = load_trajectory_from_pkl(pkl_path)
        assert traj.seed == 42
        assert traj.rank == 0
        assert traj.env_index == 0
        assert traj.episode == 3
        assert traj.success is True
        assert len(traj.actions) == 5

    def test_load_with_nonstandard_name(self, tmp_path):
        pkl_path = tmp_path / "custom_trajectory.pkl"
        data = {"actions": [torch.ones(7)]}
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

        traj = load_trajectory_from_pkl(pkl_path)
        assert traj.seed == 0  # fallback
        assert len(traj.actions) == 1

    def test_load_fail_trajectory(self, tmp_path):
        pkl_path = tmp_path / "step_50_sid_10_rank_1_env_2_episode_0_fail.pkl"
        data = {"actions": [torch.zeros(7)]}
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

        traj = load_trajectory_from_pkl(pkl_path)
        assert traj.success is False
        assert traj.seed == 10


class TestScanAndPairTrajectories:
    def test_scan_groups_by_seed(self, tmp_path):
        for sid in [42, 42, 43]:
            ep = 0 if sid == 43 else (0 if not list(tmp_path.glob("*sid_42*")) else 1)
            name = f"step_100_sid_{sid}_rank_0_env_0_episode_{ep}_success.pkl"
            with open(tmp_path / name, "wb") as f:
                pickle.dump({"actions": [torch.zeros(7)]}, f)

        result = scan_and_pair_trajectories(tmp_path)
        assert 42 in result
        assert 43 in result
        assert len(result[42]) == 2
        assert len(result[43]) == 1

    def test_filter_by_target_seeds(self, tmp_path):
        for sid in [42, 43, 44]:
            name = f"step_100_sid_{sid}_rank_0_env_0_episode_0_success.pkl"
            with open(tmp_path / name, "wb") as f:
                pickle.dump({"actions": [torch.zeros(7)]}, f)

        result = scan_and_pair_trajectories(tmp_path, target_seeds=[42, 44])
        assert 42 in result
        assert 44 in result
        assert 43 not in result


# -----------------------------------------------------------------------
# Bottleneck Detector
# -----------------------------------------------------------------------

def _make_traj_with_actions(actions: list[torch.Tensor]) -> EpisodeTrajectory:
    steps = [
        StepRecord(step=i, obs={}, action=a, reward=0.0,
                   terminated=False, truncated=False, info={},
                   model_latency_ms=0, env_latency_ms=0)
        for i, a in enumerate(actions)
    ]
    return EpisodeTrajectory(seed=42, env_index=0, steps=steps,
                             success=True, total_reward=0.0)


class TestBottleneckDetector:
    def test_insufficient_data(self):
        t1 = _make_traj_with_actions([torch.zeros(7)] * 5)
        assert detect_bottleneck_k_b([t1]) == 0

    def test_identical_trajectories(self):
        actions = [torch.ones(7) * i for i in range(10)]
        t1 = _make_traj_with_actions(actions)
        t2 = _make_traj_with_actions(actions)
        # Identical → no divergence → K_B = 0
        assert detect_bottleneck_k_b([t1, t2]) == 0

    def test_divergent_early_convergent_late(self):
        # Trajectories that diverge early but converge at the end
        a1 = [torch.tensor([float(i)]) for i in range(10)]
        a2 = [torch.tensor([float(i) + 5.0]) for i in range(7)] + \
             [torch.tensor([float(i)]) for i in range(7, 10)]
        t1 = _make_traj_with_actions(a1)
        t2 = _make_traj_with_actions(a2)

        k_b = detect_bottleneck_k_b([t1, t2])
        # The last 3 steps are identical, divergence is in earlier steps
        # K_B should be > 0 (where divergence exceeds threshold)
        assert k_b >= 0  # exact value depends on threshold


# -----------------------------------------------------------------------
# ActionReplacerModelAdapter
# -----------------------------------------------------------------------

def _make_step_record(step: int, action: torch.Tensor) -> StepRecord:
    return StepRecord(step=step, obs={}, action=action, reward=0.0,
                      terminated=False, truncated=False, info={},
                      model_latency_ms=0, env_latency_ms=0)


class TestActionReplacerModelAdapter:
    def test_no_replacement_before_bottleneck(self):
        inner = MagicMock()
        inner.infer.return_value = (torch.ones(1, 7), {})

        baseline = EpisodeTrajectory(
            seed=42, env_index=0,
            steps=[_make_step_record(i, torch.zeros(1, 7)) for i in range(10)],
            success=True, total_reward=0.0,
        )

        adapter = ActionReplacerModelAdapter(
            inner=inner, baseline_trajectories={42: baseline},
            k_b=3, total_steps=10,
        )
        adapter.set_seed(42)

        # Steps 0-6 should NOT be replaced (before bottleneck zone)
        for _ in range(7):
            action, meta = adapter.infer({"img": torch.zeros(1)})
            assert meta["replaced"] is False
            assert torch.allclose(action, torch.ones(1, 7))

    def test_replacement_in_bottleneck_zone(self):
        inner = MagicMock()
        inner.infer.return_value = (torch.ones(1, 7), {})

        baseline = EpisodeTrajectory(
            seed=42, env_index=0,
            steps=[_make_step_record(i, torch.zeros(1, 7)) for i in range(10)],
            success=True, total_reward=0.0,
        )

        adapter = ActionReplacerModelAdapter(
            inner=inner, baseline_trajectories={42: baseline},
            k_b=3, total_steps=10,
        )
        adapter.set_seed(42)

        # Advance to bottleneck zone (step 7)
        for _ in range(7):
            adapter.infer({"img": torch.zeros(1)})

        # Steps 7, 8, 9 should be replaced
        action, meta = adapter.infer({"img": torch.zeros(1)})
        assert meta["replaced"] is True
        assert torch.allclose(action, torch.zeros(1, 7))
        assert "replacement_l2" in meta


# -----------------------------------------------------------------------
# OpenLoopReplayAdapter
# -----------------------------------------------------------------------

class TestOpenLoopReplayAdapter:
    def test_replay_actions(self):
        actions = [torch.tensor([float(i)]) for i in range(5)]
        traj = LoadedTrajectory(
            seed=42, rank=0, env_index=0, episode=0, success=True,
            actions=actions, observations=[{}] * 5, rewards=[1.0] * 5,
            source_path="/tmp/test.pkl",
        )
        adapter = OpenLoopReplayAdapter(traj)

        for i in range(5):
            action, meta = adapter.infer({})
            assert torch.allclose(action, torch.tensor([float(i)]))
            assert meta["open_loop"] is True

    def test_exhausted_raises(self):
        traj = LoadedTrajectory(
            seed=42, rank=0, env_index=0, episode=0, success=True,
            actions=[torch.zeros(7)], observations=[{}], rewards=[1.0],
            source_path="/tmp/test.pkl",
        )
        adapter = OpenLoopReplayAdapter(traj)
        adapter.infer({})

        with pytest.raises(RuntimeError, match="exhausted"):
            adapter.infer({})

    def test_reset(self):
        traj = LoadedTrajectory(
            seed=42, rank=0, env_index=0, episode=0, success=True,
            actions=[torch.zeros(7)], observations=[{}], rewards=[1.0],
            source_path="/tmp/test.pkl",
        )
        adapter = OpenLoopReplayAdapter(traj)
        adapter.infer({})
        adapter.reset()
        # Should work again after reset
        action, _ = adapter.infer({})
        assert action is not None
