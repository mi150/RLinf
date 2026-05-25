from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import examples.embodiment.eval_robocasa_openpi_diagnostics as diagnostics_eval
from examples.embodiment.eval_robocasa_openpi_diagnostics import (
    _first_task_name,
    build_episode_record,
    run_diagnostics_eval,
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


def test_first_task_name_supports_plain_string() -> None:
    cfg = OmegaConf.create({"env": {"eval": {"task_names": "CloseDrawer"}}})

    assert _first_task_name(cfg) == "CloseDrawer"


def test_validate_diagnostics_cfg_rejects_invalid_flush_every(tmp_path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    cfg = OmegaConf.create(
        {
            "env": {"eval": {"env_type": "robocasa", "total_num_envs": 1}},
            "actor": {
                "model": {"model_type": "openpi", "model_path": str(model_path)}
            },
            "diagnostics": {
                "output_path": str(tmp_path / "out.jsonl"),
                "flush_every": 0,
            },
        }
    )

    with pytest.raises(ValueError, match="flush_every"):
        validate_diagnostics_cfg(cfg)


def _diagnostics_cfg(tmp_path) -> OmegaConf:
    model_path = tmp_path / "model"
    model_path.mkdir()
    return OmegaConf.create(
        {
            "env": {
                "eval": {
                    "env_type": "robocasa",
                    "task_names": "CloseDrawer",
                    "total_num_envs": 1,
                    "max_episode_steps": 3,
                    "seed": 42,
                    "action_space": "12d",
                }
            },
            "actor": {
                "model": {
                    "model_type": "openpi",
                    "model_path": str(model_path),
                    "num_action_chunks": 1,
                    "action_dim": 2,
                    "policy_setup": "12d",
                }
            },
            "diagnostics": {
                "output_path": str(tmp_path / "out.jsonl"),
                "num_episodes": 1,
                "max_contacts": 2,
                "include_model_names": False,
                "flush_every": 1,
            },
        }
    )


class _FakeModel:
    def __init__(self, raw_actions):
        self.raw_actions = raw_actions

    def predict_action_batch(self, env_obs, mode="eval"):
        return self.raw_actions


class _FakeEnv:
    def __init__(self, success_at_end=False):
        self.success_at_end = success_at_end
        self.actions = []
        self.closed = False

    def reset(self):
        return {"task_descriptions": ["close the drawer"]}, {}

    def step(self, action, auto_reset=False):
        self.actions.append(action)
        infos = {}
        if self.success_at_end:
            infos["episode"] = {"success_at_end": torch.tensor([True])}
        return (
            {"task_descriptions": ["close the drawer"]},
            torch.tensor([1.0]),
            torch.tensor([False]),
            torch.tensor([False]),
            infos,
        )

    def get_mujoco_diagnostics(self, max_contacts=None, include_model_names=True):
        return [{"ncon": 0, "max_contacts": max_contacts}]

    def close(self):
        self.closed = True


def test_run_diagnostics_eval_reads_success_at_end(monkeypatch, tmp_path) -> None:
    cfg = _diagnostics_cfg(tmp_path)
    cfg.env.eval.max_episode_steps = 1
    env = _FakeEnv(success_at_end=True)
    monkeypatch.setattr(diagnostics_eval, "create_robocasa_eval_env", lambda cfg: env)
    monkeypatch.setattr(
        diagnostics_eval,
        "load_openpi_model",
        lambda cfg: _FakeModel(np.array([[[0.1, 0.2]]])),
    )
    monkeypatch.setattr(
        diagnostics_eval,
        "prepare_actions",
        lambda **kwargs: kwargs["raw_chunk_actions"],
        raising=False,
    )

    run_diagnostics_eval(cfg)

    record = json.loads((tmp_path / "out.jsonl").read_text().strip())
    assert record["success"] is True
    assert record["termination_reason"] == "success"
    assert record["steps"][0]["success"] is True
    assert record["steps"][0]["terminated"] is False


def test_run_diagnostics_eval_prepares_action_chunk(monkeypatch, tmp_path) -> None:
    cfg = _diagnostics_cfg(tmp_path)
    cfg.env.eval.max_episode_steps = 1
    env = _FakeEnv()
    prepare_calls = []

    def _prepare_actions(**kwargs):
        prepare_calls.append(kwargs)
        return np.array([[[9.0, 8.0, 7.0]]])

    monkeypatch.setattr(diagnostics_eval, "create_robocasa_eval_env", lambda cfg: env)
    monkeypatch.setattr(
        diagnostics_eval,
        "load_openpi_model",
        lambda cfg: _FakeModel(np.array([[[0.1, 0.2]]])),
    )
    monkeypatch.setattr(
        diagnostics_eval,
        "prepare_actions",
        _prepare_actions,
        raising=False,
    )

    run_diagnostics_eval(cfg)

    assert prepare_calls
    assert prepare_calls[0]["env_type"] == "robocasa"
    assert prepare_calls[0]["model_type"] == "openpi"
    assert prepare_calls[0]["num_action_chunks"] == 1
    assert prepare_calls[0]["action_dim"] == 2
    assert prepare_calls[0]["policy"] == "12d"
    np.testing.assert_allclose(env.actions[0], np.array([[9.0, 8.0, 7.0]]))
