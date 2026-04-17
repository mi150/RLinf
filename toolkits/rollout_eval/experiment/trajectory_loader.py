"""Load trajectories from external pkl files."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import torch

from toolkits.rollout_eval.experiment.types import LoadedTrajectory

# File naming convention from training data collection:
# step_{step}_sid_{sid}_rank_{rank}_env_{env}_episode_{episode}_{success|fail}.pkl
_PKL_PATTERN = re.compile(
    r"step_(?P<step>\d+)_sid_(?P<sid>\d+)_rank_(?P<rank>\d+)"
    r"_env_(?P<env>\d+)_episode_(?P<episode>\d+)_(?P<outcome>success|fail)\.pkl$"
)


def load_trajectory_from_pkl(path: str | Path) -> LoadedTrajectory:
    """Load a single trajectory from a pkl file.

    Expected pkl contents:
        - actions: list of T tensors [action_dim]
        - observations: list of observation dicts (optional)
        - rewards: list of floats (optional)
        - success: bool (optional, inferred from filename if absent)
    """
    path = Path(path)
    with open(path, "rb") as f:
        data: dict[str, Any] = pickle.load(f)

    actions = data["actions"]
    if isinstance(actions, torch.Tensor):
        actions = list(actions)
    else:
        actions = [a if torch.is_tensor(a) else torch.tensor(a) for a in actions]

    observations = data.get("observations", [{}] * len(actions))
    rewards = data.get("rewards", [0.0] * len(actions))
    success = data.get("success", False)

    # Parse metadata from filename
    match = _PKL_PATTERN.search(path.name)
    if match:
        seed = int(match.group("sid"))
        rank = int(match.group("rank"))
        env_index = int(match.group("env"))
        episode = int(match.group("episode"))
        success = match.group("outcome") == "success"
    else:
        seed = 0
        rank = 0
        env_index = 0
        episode = 0

    return LoadedTrajectory(
        seed=seed,
        rank=rank,
        env_index=env_index,
        episode=episode,
        success=success,
        actions=actions,
        observations=observations,
        rewards=rewards,
        source_path=str(path),
    )


def scan_and_pair_trajectories(
    directory: str | Path,
    target_seeds: list[int] | None = None,
) -> dict[int, list[LoadedTrajectory]]:
    """Scan directory for pkl files and group by seed.

    Args:
        directory: Path to scan for pkl files.
        target_seeds: If given, only load trajectories matching these seeds.

    Returns:
        Mapping from seed to list of LoadedTrajectory, sorted by episode.
    """
    directory = Path(directory)
    result: dict[int, list[LoadedTrajectory]] = {}

    for pkl_path in sorted(directory.glob("*.pkl")):
        match = _PKL_PATTERN.search(pkl_path.name)
        if not match:
            continue

        seed = int(match.group("sid"))
        if target_seeds is not None and seed not in target_seeds:
            continue

        traj = load_trajectory_from_pkl(pkl_path)
        result.setdefault(seed, []).append(traj)

    # Sort each seed's trajectories by episode
    for seed in result:
        result[seed].sort(key=lambda t: t.episode)

    return result
