from __future__ import annotations

import json

import numpy as np
import torch

from examples.embodiment.eval_robocasa_openpi_diagnostics import (
    build_episode_record,
    to_jsonable,
)


def test_to_jsonable_converts_numpy_and_torch_values() -> None:
    value = {
        "array": np.array([[1, 2], [3, 4]], dtype=np.int64),
        "float": np.float32(1.5),
        "bool": np.bool_(True),
        "tensor": torch.tensor([1.0, 2.0]),
    }

    result = to_jsonable(value)

    assert result == {
        "array": [[1, 2], [3, 4]],
        "float": 1.5,
        "bool": True,
        "tensor": [1.0, 2.0],
    }
    json.dumps(result)


def test_build_episode_record_contains_required_fields() -> None:
    record = build_episode_record(
        episode_id=3,
        env_id=0,
        task_name="CloseDrawer",
        seed=42,
        task_description="close the drawer",
        actions=[np.array([0.1, 0.2])],
        step_records=[
            {
                "step": 0,
                "reward": np.float32(0.0),
                "success": np.bool_(False),
                "terminated": False,
                "truncated": False,
                "diagnostics": {"ncon": 1, "qvel": np.array([0.0])},
            }
        ],
        success=True,
        termination_reason="success",
    )

    assert record["episode_id"] == 3
    assert record["env_id"] == 0
    assert record["task_name"] == "CloseDrawer"
    assert record["seed"] == 42
    assert record["success"] is True
    assert record["num_steps"] == 1
    assert record["termination_reason"] == "success"
    assert record["task_description"] == "close the drawer"
    assert record["actions"] == [[0.1, 0.2]]
    assert record["steps"][0]["diagnostics"]["qvel"] == [0.0]
    json.dumps(record)
