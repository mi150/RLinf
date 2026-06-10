# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared helpers for chunk-step environment execution."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

CHUNK_STEP_MODES = {
    "sync_time_major",
    "parallel_shard",
    "latency_balanced_pair",
    "latency_bin_packing",
}


@dataclass
class ChunkShard:
    """Mapping from one shard worker to the global env ids it owns."""

    env_indices: torch.Tensor
    worker: Any


@dataclass
class LocalChunkResult:
    """Chunk-step result produced by one local env shard."""

    obs_list: list[Any]
    infos_list: list[Any]
    rewards: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor


@dataclass
class GlobalChunkResult:
    """Chunk-step result after restoring global env order."""

    obs_list: list[Any]
    infos_list: list[Any]
    rewards: torch.Tensor
    raw_terminations: torch.Tensor
    raw_truncations: torch.Tensor


def split_env_indices(
    num_envs: int, num_shards: int, *, device: torch.device | str | None = None
) -> list[torch.Tensor]:
    """Split global env indices into near-even contiguous shards."""
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if num_shards > num_envs:
        raise ValueError(f"num_shards({num_shards}) must be <= num_envs({num_envs})")

    indices = torch.arange(num_envs, device=device)
    return [chunk for chunk in torch.tensor_split(indices, num_shards) if len(chunk)]


def select_local_chunk_actions(
    chunk_actions: torch.Tensor | np.ndarray, env_indices: torch.Tensor
) -> torch.Tensor | np.ndarray:
    """Select the action chunk for one shard."""
    if torch.is_tensor(chunk_actions):
        return chunk_actions.index_select(0, env_indices.to(chunk_actions.device))
    return chunk_actions[np.asarray(env_indices.cpu())]


def build_chunk_done_outputs(
    raw_terminations: torch.Tensor,
    raw_truncations: torch.Tensor,
    *,
    collapse_to_last_step: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build public chunk done tensors from raw per-step done tensors."""
    past_terminations = raw_terminations.any(dim=1)
    past_truncations = raw_truncations.any(dim=1)
    past_dones = torch.logical_or(past_terminations, past_truncations)

    if collapse_to_last_step:
        chunk_terminations = torch.zeros_like(raw_terminations)
        chunk_terminations[:, -1] = past_terminations
        chunk_truncations = torch.zeros_like(raw_truncations)
        chunk_truncations[:, -1] = past_truncations
    else:
        chunk_terminations = raw_terminations.clone()
        chunk_truncations = raw_truncations.clone()

    return (
        chunk_terminations,
        chunk_truncations,
        past_terminations,
        past_truncations,
        past_dones,
    )


def maybe_apply_ignore_terminations(
    raw_terminations: torch.Tensor, ignore_terminations: bool
) -> torch.Tensor:
    """Zero terminations when an env maps success handling to truncation."""
    if ignore_terminations:
        return torch.zeros_like(raw_terminations)
    return raw_terminations


def stack_vector_chunk_returns(env_results: list[tuple]) -> tuple[list[Any], ...]:
    """Convert per-env chunk returns into per-step vector-env returns.

    Each worker returns ``tuple(zip(*local_step_results))``. This helper restores
    the same shape callers get from repeated ``VectorEnv.step`` calls, but the
    workers have already executed their full local chunks independently.
    """
    if not env_results:
        raise ValueError("env_results must not be empty")

    chunk_size = len(env_results[0][0])
    per_step_returns = []
    for step_idx in range(chunk_size):
        step_returns = [
            tuple(return_part[step_idx] for return_part in env_result)
            for env_result in env_results
        ]
        return_lists = tuple(zip(*step_returns))
        per_step_returns.append(tuple(_stack(values) for values in return_lists))
    return tuple(list(values) for values in zip(*per_step_returns))


def _stack(values: tuple[Any, ...]) -> Any:
    try:
        return np.stack(values)
    except ValueError:
        return np.array(values, dtype=object)
