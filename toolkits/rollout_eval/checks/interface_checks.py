"""Interface-level contract checks for observations and actions."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from toolkits.rollout_eval.rollout_types import infer_batch_size


def _is_finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())



def assert_obs_batch(obs_batch: dict[str, Any], expected_batch: int | None = None) -> None:
    """Validate observation payload structure and tensor sanity."""
    if not isinstance(obs_batch, dict) or not obs_batch:
        raise ValueError("obs_batch must be a non-empty dict")

    batch = infer_batch_size(obs_batch)
    if expected_batch is not None and batch != expected_batch:
        raise ValueError(
            f"Observation batch mismatch: expected {expected_batch}, got {batch}"
        )

    for key, value in obs_batch.items():
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                raise ValueError(f"Observation tensor '{key}' must be at least 1D")
            if not _is_finite_tensor(value):
                raise ValueError(f"Observation tensor '{key}' contains NaN/Inf")
        elif isinstance(value, np.ndarray):
            if value.ndim == 0:
                raise ValueError(f"Observation array '{key}' must be at least 1D")
            if not np.isfinite(value).all():
                raise ValueError(f"Observation array '{key}' contains NaN/Inf")



def assert_action_batch(actions: torch.Tensor, expected_batch: int | None = None) -> None:
    """Validate model action tensor produced for environment stepping."""
    if not isinstance(actions, torch.Tensor):
        raise TypeError(f"actions must be torch.Tensor, got {type(actions)}")

    if actions.ndim < 1:
        raise ValueError("actions tensor must be at least 1D")

    if expected_batch is not None and int(actions.shape[0]) != expected_batch:
        raise ValueError(
            f"Action batch mismatch: expected {expected_batch}, got {actions.shape[0]}"
        )

    if not torch.isfinite(actions).all():
        raise ValueError("actions contain NaN/Inf")
