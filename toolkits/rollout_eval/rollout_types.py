"""Common lightweight types for rollout eval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class EnvStepResult:
    """Container for one environment transition step."""

    obs: dict[str, Any]
    reward: torch.Tensor | np.ndarray | None = None
    terminated: torch.Tensor | np.ndarray | None = None
    truncated: torch.Tensor | np.ndarray | None = None
    info: dict[str, Any] | None = None


@dataclass
class LatencyStats:
    """Latency counters for model/env operations."""

    model_infer_seconds: float = 0.0
    env_step_seconds: float = 0.0
    model_infer_count: int = 0
    env_step_count: int = 0


@dataclass
class RolloutLoopResult:
    """Summary from a rollout loop execution."""

    total_steps: int
    warmup_steps: int
    measure_steps: int
    latency: LatencyStats
    profile_metrics: dict[str, float] = field(default_factory=dict)



def infer_batch_size(obs_batch: dict[str, Any]) -> int:
    """Infer batch size from observation payload.

    Args:
        obs_batch: Observation dictionary.

    Returns:
        Inferred leading batch dimension.

    Raises:
        ValueError: If batch size cannot be inferred.
    """
    for value in obs_batch.values():
        if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim >= 1:
            return int(value.shape[0])
        if isinstance(value, list) and len(value) > 0:
            return len(value)
    raise ValueError("Cannot infer batch size from observation payload")
