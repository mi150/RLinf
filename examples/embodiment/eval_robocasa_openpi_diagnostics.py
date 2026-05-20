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

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def to_jsonable(value: Any) -> Any:
    """Convert numpy and torch values into JSON-serializable Python values."""
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def build_episode_record(
    *,
    episode_id: int,
    env_id: int,
    task_name: str,
    seed: int,
    task_description: str,
    actions: list[Any],
    step_records: list[dict[str, Any]],
    success: bool,
    termination_reason: str,
) -> dict[str, Any]:
    """Build a JSON-compatible diagnostics record for one evaluation episode."""
    return to_jsonable(
        {
            "episode_id": episode_id,
            "env_id": env_id,
            "task_name": task_name,
            "seed": seed,
            "success": bool(success),
            "num_steps": len(step_records),
            "termination_reason": termination_reason,
            "task_description": task_description,
            "actions": actions,
            "steps": step_records,
        }
    )
