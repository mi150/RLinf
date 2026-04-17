"""Seedable environment adapter for deterministic evaluation."""

from __future__ import annotations

from typing import Any

import torch

from toolkits.rollout_eval.adapters.env_adapter import GenericEnvAdapter
from toolkits.rollout_eval.rollout_types import EnvStepResult


class SeedableEnvAdapter:
    """Wraps GenericEnvAdapter to inject fixed seeds on reset.

    The orchestrator calls ``reset(seed=...)`` before each episode to ensure
    deterministic environment initialisation.
    """

    def __init__(self, inner: GenericEnvAdapter, seeds: list[int] | None = None):
        self.inner = inner
        self.seeds = seeds or []
        self._current_seed_idx = 0
        self._current_seed: int | None = None

    # ------------------------------------------------------------------
    # Seed injection
    # ------------------------------------------------------------------

    def _inject_seed(self, seed: int) -> None:
        """Push *seed* into the underlying env through available APIs."""
        env = self.inner.env
        if hasattr(env, "seed") and not callable(getattr(env, "seed")):
            env.seed = seed
        if hasattr(env, "set_seed") and callable(env.set_seed):
            env.set_seed(seed)
        # ManiSkill / Gymnasium style
        if hasattr(env, "_seed") and not callable(getattr(env, "_seed")):
            env._seed = seed

    # ------------------------------------------------------------------
    # Public API (mirrors EnvAdapterProtocol)
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment with a deterministic seed.

        If *seed* is ``None``, the next seed from the configured list is used.
        """
        if seed is None and self.seeds:
            seed = self.seeds[self._current_seed_idx % len(self.seeds)]
            self._current_seed_idx += 1

        if seed is not None:
            self._inject_seed(seed)
        self._current_seed = seed

        obs, info = self.inner.reset()
        return obs, info

    def step(self, actions: torch.Tensor) -> EnvStepResult:
        return self.inner.step(actions)

    def close(self) -> None:
        self.inner.close()

    @property
    def current_seed(self) -> int | None:
        return self._current_seed

    @property
    def env(self) -> Any:
        """Expose underlying env for RecordVideo wrapping etc."""
        return self.inner.env
