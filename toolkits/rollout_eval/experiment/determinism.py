"""Determinism verification across repeated runs."""

from __future__ import annotations

import torch

from toolkits.rollout_eval.experiment.types import DeterminismResult, EpisodeTrajectory


def verify_determinism(
    trajectories_by_seed: dict[int, list[EpisodeTrajectory]],
) -> dict[int, DeterminismResult]:
    """Compare trajectories across runs for the same seed.

    Args:
        trajectories_by_seed: Mapping from seed to list of episode trajectories
            collected across multiple runs.

    Returns:
        Per-seed determinism results.
    """
    results: dict[int, DeterminismResult] = {}

    for seed, runs in trajectories_by_seed.items():
        if len(runs) < 2:
            results[seed] = DeterminismResult(
                seed=seed, action_match=True, max_action_l2=0.0,
                obs_match=True, reward_match=True,
            )
            continue

        ref = runs[0]
        action_match = True
        obs_match = True
        reward_match = True
        max_l2 = 0.0

        for other in runs[1:]:
            min_steps = min(len(ref.steps), len(other.steps))
            if len(ref.steps) != len(other.steps):
                action_match = False
                obs_match = False

            for i in range(min_steps):
                # Action comparison
                a_ref = _to_float_tensor(ref.steps[i].action)
                a_other = _to_float_tensor(other.steps[i].action)
                if a_ref.shape == a_other.shape:
                    l2 = torch.norm(a_ref - a_other).item()
                    max_l2 = max(max_l2, l2)
                    if l2 > 1e-6:
                        action_match = False
                else:
                    action_match = False
                    max_l2 = float("inf")

                # Reward comparison
                if abs(ref.steps[i].reward - other.steps[i].reward) > 1e-6:
                    reward_match = False

                # Observation comparison (check tensor keys)
                obs_match = obs_match and _obs_equal(
                    ref.steps[i].obs, other.steps[i].obs
                )

        results[seed] = DeterminismResult(
            seed=seed,
            action_match=action_match,
            max_action_l2=max_l2,
            obs_match=obs_match,
            reward_match=reward_match,
        )

    return results


def _to_float_tensor(val: torch.Tensor | object) -> torch.Tensor:
    if torch.is_tensor(val):
        return val.float().flatten()
    return torch.tensor(val, dtype=torch.float32).flatten()


def _obs_equal(obs_a: dict, obs_b: dict) -> bool:
    """Shallow equality check on observation dicts (tensor keys only)."""
    if set(obs_a.keys()) != set(obs_b.keys()):
        return False
    for key in obs_a:
        va, vb = obs_a[key], obs_b[key]
        if torch.is_tensor(va) and torch.is_tensor(vb):
            if va.shape != vb.shape:
                return False
            if not torch.allclose(va.float(), vb.float(), atol=1e-6):
                return False
        elif torch.is_tensor(va) != torch.is_tensor(vb):
            return False
    return True
