"""Bottleneck K_B detection from trajectory data."""

from __future__ import annotations

import torch

from toolkits.rollout_eval.experiment.types import EpisodeTrajectory


def detect_bottleneck_k_b(
    trajectories: list[EpisodeTrajectory],
    sigma_multiplier: float = 2.0,
) -> int:
    """Detect bottleneck boundary K_B from trajectory data.

    Algorithm (reverse-aligned L2 divergence):
    1. Reverse-align all trajectories (step -1, -2, ..., -T)
    2. Compute pairwise L2 distance at each reverse step
    3. K_B = first step from end where divergence > mu + sigma_multiplier * sigma

    Args:
        trajectories: List of episode trajectories (ideally same seed, different episodes).
        sigma_multiplier: Threshold multiplier (default 2.0).

    Returns:
        K_B: number of steps from end that define the bottleneck zone.
        Returns 0 if no bottleneck detected or insufficient data.
    """
    if len(trajectories) < 2:
        return 0

    # Extract action sequences and reverse-align
    action_seqs = []
    for traj in trajectories:
        actions = [s.action.float().flatten() for s in traj.steps]
        if actions:
            action_seqs.append(actions)

    if len(action_seqs) < 2:
        return 0

    # Find minimum trajectory length
    min_len = min(len(seq) for seq in action_seqs)
    if min_len < 2:
        return 0

    # Reverse-align: index 0 = last step, index k = step -(k+1)
    reversed_seqs = []
    for seq in action_seqs:
        reversed_seqs.append(list(reversed(seq[-min_len:])))

    # Compute mean pairwise L2 at each reverse step
    divergences = []
    for k in range(min_len):
        l2_values = []
        for i in range(len(reversed_seqs)):
            for j in range(i + 1, len(reversed_seqs)):
                a_i = reversed_seqs[i][k]
                a_j = reversed_seqs[j][k]
                if a_i.shape == a_j.shape:
                    l2_values.append(torch.norm(a_i - a_j).item())
        if l2_values:
            divergences.append(sum(l2_values) / len(l2_values))
        else:
            divergences.append(0.0)

    if not divergences:
        return 0

    # Compute threshold: mu + sigma_multiplier * sigma
    div_tensor = torch.tensor(divergences)
    mu = div_tensor.mean().item()
    sigma = div_tensor.std().item() if len(divergences) > 1 else 0.0
    threshold = mu + sigma_multiplier * sigma

    # K_B = first reverse step where divergence exceeds threshold
    for k, d in enumerate(divergences):
        if d > threshold:
            return k

    # No bottleneck detected — all steps are within threshold
    return 0
