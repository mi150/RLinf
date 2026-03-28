"""Determinism and stability checks for rollout outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DeterminismCheckResult:
    """Result of two-run deterministic consistency check."""

    mean_abs_diff: float
    passed: bool



def run_determinism_check(
    actions_first: torch.Tensor,
    actions_second: torch.Tensor,
    tolerance: float = 1e-5,
) -> DeterminismCheckResult:
    """Compare two action tensors and check stability threshold."""
    if actions_first.shape != actions_second.shape:
        return DeterminismCheckResult(mean_abs_diff=float("inf"), passed=False)

    diff = torch.mean(torch.abs(actions_first.float() - actions_second.float())).item()
    return DeterminismCheckResult(mean_abs_diff=float(diff), passed=diff <= tolerance)
