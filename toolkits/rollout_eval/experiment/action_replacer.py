"""Action replacement adapters for bottleneck zone evaluation."""

from __future__ import annotations

from typing import Any

import torch

from toolkits.rollout_eval.adapters.model_adapter import ModelAdapterProtocol
from toolkits.rollout_eval.config_bridge import EvalRuntimeConfig
from toolkits.rollout_eval.experiment.recording_loop import run_recording_loop
from toolkits.rollout_eval.experiment.seedable_env_adapter import SeedableEnvAdapter
from toolkits.rollout_eval.experiment.types import (
    EpisodeTrajectory,
    LoadedTrajectory,
)


class ActionReplacerModelAdapter:
    """Substitutes cached actions in the bottleneck zone.

    For steps >= total_steps - k_b, the adapter returns the baseline action
    instead of the model's action. The model is still called for comparison.
    """

    def __init__(
        self,
        inner: ModelAdapterProtocol | Any,
        baseline_trajectories: dict[int, EpisodeTrajectory],
        k_b: int,
        total_steps: int,
    ):
        self.inner = inner
        self.baseline = baseline_trajectories
        self.k_b = k_b
        self.total_steps = total_steps
        self._step = 0
        self._current_seed = 0

    def set_seed(self, seed: int) -> None:
        self._current_seed = seed
        self._step = 0

    def infer(
        self, obs_batch: dict[str, Any], mode: str = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Always run model for comparison
        model_action, meta = self.inner.infer(obs_batch=obs_batch, mode=mode)
        if not isinstance(meta, dict):
            meta = {}

        # Check if in bottleneck zone
        if self._step >= self.total_steps - self.k_b:
            baseline = self.baseline.get(self._current_seed)
            if baseline and self._step < len(baseline.steps):
                cached_action = baseline.steps[self._step].action
                # Ensure same device
                if torch.is_tensor(model_action) and torch.is_tensor(cached_action):
                    cached_action = cached_action.to(model_action.device)
                meta["replaced"] = True
                meta["model_action"] = model_action
                meta["replacement_l2"] = torch.norm(
                    model_action.float() - cached_action.float()
                ).item()
                self._step += 1
                return cached_action, meta

        meta["replaced"] = False
        self._step += 1
        return model_action, meta


class OpenLoopReplayAdapter:
    """Feeds pre-recorded actions from external trajectories.

    No model inference is performed. Actions are replayed step-by-step.
    """

    def __init__(self, trajectory: LoadedTrajectory):
        self.trajectory = trajectory
        self._step = 0

    def infer(
        self, obs_batch: dict[str, Any], mode: str = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self._step >= len(self.trajectory.actions):
            raise RuntimeError(
                f"OpenLoopReplayAdapter exhausted: step {self._step} "
                f">= trajectory length {len(self.trajectory.actions)}"
            )
        action = self.trajectory.actions[self._step]
        meta = {
            "open_loop": True,
            "source": self.trajectory.source_path,
            "original_success": self.trajectory.success,
        }
        self._step += 1
        return action, meta

    def reset(self) -> None:
        self._step = 0


# ---------------------------------------------------------------------------
# Phase 3 runner
# ---------------------------------------------------------------------------

def run_action_replace_eval(
    env_adapter: SeedableEnvAdapter,
    model_adapter: ModelAdapterProtocol | Any,
    runtime: EvalRuntimeConfig,
    baseline_trajectories: dict[int, EpisodeTrajectory],
    k_b: int,
    seeds: list[int],
) -> list[EpisodeTrajectory]:
    """Run action replacement evaluation.

    Wraps the model adapter with ActionReplacerModelAdapter and runs
    the recording loop for each seed.

    Returns:
        List of trajectories with replacement metadata in step.meta.
    """
    replacer = ActionReplacerModelAdapter(
        inner=model_adapter,
        baseline_trajectories=baseline_trajectories,
        k_b=k_b,
        total_steps=runtime.total_steps,
    )

    all_trajectories: list[EpisodeTrajectory] = []
    for seed in seeds:
        replacer.set_seed(seed)
        _, trajs = run_recording_loop(env_adapter, replacer, runtime, seed=seed)
        all_trajectories.extend(trajs)

    return all_trajectories
