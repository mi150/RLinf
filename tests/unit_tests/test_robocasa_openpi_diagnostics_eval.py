from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from examples.embodiment.eval_robocasa_openpi_diagnostics import (
    build_episode_record,
    to_jsonable,
    validate_diagnostics_cfg,
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


def test_validate_diagnostics_cfg_rejects_non_robocasa_openpi() -> None:
    cfg = OmegaConf.create(
        {
            "env": {"eval": {"env_type": "libero"}},
            "actor": {
                "model": {"model_type": "openpi", "model_path": "/tmp/model"}
            },
            "diagnostics": {"output_path": "/tmp/out.jsonl", "num_episodes": 1},
        }
    )

    with pytest.raises(ValueError, match="RoboCasa"):
        validate_diagnostics_cfg(cfg)


def test_validate_diagnostics_cfg_sets_defaults(tmp_path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    cfg = OmegaConf.create(
        {
            "env": {
                "eval": {
                    "env_type": "robocasa",
                    "total_num_envs": 1,
                    "max_episode_steps": 3,
                }
            },
            "actor": {
                "model": {"model_type": "openpi", "model_path": str(model_path)}
            },
            "diagnostics": {"output_path": str(tmp_path / "out.jsonl")},
        }
    )

    validate_diagnostics_cfg(cfg)

    assert cfg.diagnostics.num_episodes == 1
    assert cfg.diagnostics.max_contacts == 32
    assert cfg.diagnostics.include_model_names is True
    assert cfg.diagnostics.flush_every == 1
