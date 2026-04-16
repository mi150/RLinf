from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from toolkits.rollout_eval.adapters.env_adapter import GenericEnvAdapter
from toolkits.rollout_eval.adapters.model_adapter import (
    GenericModelAdapter,
    _select_stage_modules,
    _validate_model_path_or_raise,
)


class _OpenVLALike(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = torch.nn.Module()
        self.base_model.model = torch.nn.Module()
        self.base_model.model.vision_backbone = torch.nn.Linear(4, 4)
        self.base_model.model.language_model = torch.nn.Module()
        self.base_model.model.language_model.lm_head = torch.nn.Linear(4, 4)


class _OpenPILike(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.paligemma_with_expert = torch.nn.Module()
        self.paligemma_with_expert.paligemma = torch.nn.Linear(4, 4)
        self.action_out_proj = torch.nn.Linear(4, 4)


class _GR00TLike(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.action_head = torch.nn.Linear(4, 4)



def test_select_stage_modules_for_openvla_oft() -> None:
    model = _OpenVLALike()
    backbone, action_head = _select_stage_modules(model, "openvla_oft")

    assert backbone is not None
    assert action_head is not None
    assert backbone[0] == "base_model.model.vision_backbone"
    assert action_head[0] == "base_model.model.language_model.lm_head"



def test_select_stage_modules_for_openpi() -> None:
    model = _OpenPILike()
    backbone, action_head = _select_stage_modules(model, "openpi")

    assert backbone is not None
    assert action_head is not None
    assert backbone[0] == "paligemma_with_expert"
    assert action_head[0] == "action_out_proj"



def test_select_stage_modules_for_gr00t() -> None:
    model = _GR00TLike()
    backbone, action_head = _select_stage_modules(model, "gr00t")

    assert backbone is not None
    assert action_head is not None
    assert backbone[0] == "backbone"
    assert action_head[0] == "action_head"



def test_validate_model_path_raises_for_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing_model"
    with pytest.raises(RuntimeError, match="does not exist"):
        _validate_model_path_or_raise(str(missing))



def test_validate_model_path_raises_for_adapter_only_dir(tmp_path: Path) -> None:
    model_dir = tmp_path / "adapter_only"
    model_dir.mkdir()
    (model_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="LoRA adapter-only"):
        _validate_model_path_or_raise(str(model_dir))



def test_validate_model_path_raises_when_config_missing(tmp_path: Path) -> None:
    model_dir = tmp_path / "incomplete"
    model_dir.mkdir()
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing required config.json"):
        _validate_model_path_or_raise(str(model_dir))



def test_validate_model_path_accepts_dir_with_config(tmp_path: Path) -> None:
    model_dir = tmp_path / "full"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    _validate_model_path_or_raise(str(model_dir))



def test_validate_model_path_accepts_openpi_with_model_safetensors(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "openpi_safetensors"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_text("", encoding="utf-8")

    _validate_model_path_or_raise(str(model_dir), model_type="openpi")



def test_validate_model_path_rejects_openpi_without_weights(tmp_path: Path) -> None:
    model_dir = tmp_path / "openpi_empty"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="OpenPI model directory must contain"):
        _validate_model_path_or_raise(str(model_dir), model_type="openpi")



def test_build_model_adapter_raises_when_distributed_initialized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from toolkits.rollout_eval.adapters import model_adapter as module

    model_dir = tmp_path / "base_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "model_type": "openvla_oft",
                    "model_path": str(model_dir),
                    "precision": "bf16",
                }
            },
            "rollout": {"model": {"model_path": str(model_dir), "precision": "bf16"}},
        }
    )

    class _DummyModel:
        def eval(self):
            return None

    monkeypatch.setattr(module, "get_model", lambda _: _DummyModel())
    monkeypatch.setattr(module.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.distributed, "is_initialized", lambda: True)

    with pytest.raises(RuntimeError, match="single-process non-distributed mode"):
        module.build_model_adapter(cfg)



def test_generic_env_adapter_fills_task_descriptions_from_wrapper_attr() -> None:
    class _DummyEnv:
        def get_wrapper_attr(self, name: str):
            if name == "instruction":
                return ["pick cube"]
            return None

        def reset(self):
            obs = {"main_images": torch.zeros(2, 64, 64, 3, dtype=torch.uint8)}
            return obs, {}

        def step(self, _actions):
            obs = {"main_images": torch.zeros(2, 64, 64, 3, dtype=torch.uint8)}
            return (
                obs,
                torch.zeros(2),
                torch.zeros(2, dtype=torch.bool),
                torch.zeros(2, dtype=torch.bool),
                {},
            )

    adapter = GenericEnvAdapter(env=_DummyEnv())
    obs, _ = adapter.reset()
    assert obs["task_descriptions"] == ["pick cube", "pick cube"]



def test_generic_env_adapter_fills_placeholder_when_no_instruction() -> None:
    class _DummyEnv:
        def reset(self):
            return {"states": torch.zeros(3, 10)}, {}

        def step(self, _actions):
            return (
                {"states": torch.zeros(3, 10)},
                torch.zeros(3),
                torch.zeros(3, dtype=torch.bool),
                torch.zeros(3, dtype=torch.bool),
                {},
            )

    adapter = GenericEnvAdapter(env=_DummyEnv())
    obs, _ = adapter.reset()
    assert obs["task_descriptions"] == ["complete the task"] * 3


def test_generic_model_adapter_adds_openpi_optional_obs_keys() -> None:
    class _DummyOpenPiModel:
        def __init__(self) -> None:
            self.last_env_obs = None

        def predict_action_batch(self, env_obs, mode="eval", **kwargs):
            self.last_env_obs = env_obs
            return torch.zeros(1, 4), {}

    model = _DummyOpenPiModel()
    adapter = GenericModelAdapter(model=model, model_type="openpi")
    obs = {
        "main_images": torch.zeros(1, 64, 64, 3, dtype=torch.uint8),
        "states": torch.zeros(1, 8),
        "task_descriptions": ["task"],
        "extra_view_images": None,
    }
    adapter.infer(obs_batch=obs, mode="eval")

    assert model.last_env_obs is not None
    assert "wrist_images" in model.last_env_obs
    assert model.last_env_obs["wrist_images"] is None


def test_openpi_backbone_profile_when_forward_called_directly() -> None:
    class _DummyOpenPiDirectForward(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.paligemma_with_expert = torch.nn.Linear(4, 4)
            self.action_out_proj = torch.nn.Linear(4, 4)

        def predict_action_batch(self, env_obs, mode="eval", **kwargs):
            x = env_obs["states"]
            # Simulate OpenPI-style direct module.forward() call.
            hidden = self.paligemma_with_expert.forward(x)
            actions = self.action_out_proj(hidden)
            return actions, {}

    model = _DummyOpenPiDirectForward()
    adapter = GenericModelAdapter(model=model, model_type="openpi")
    obs = {
        "states": torch.randn(2, 4),
        "main_images": torch.zeros(2, 64, 64, 3, dtype=torch.uint8),
        "task_descriptions": ["task", "task"],
        "extra_view_images": None,
    }

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        adapter.infer(obs_batch=obs, mode="eval")
    keys = [evt.key for evt in prof.key_averages()]

    assert any("model.backbone.openpi" in key for key in keys)
    assert any("model.action_head.openpi" in key for key in keys)


def test_gr00t_logic_stage_profile_hooks_capture_backbone_and_action_head() -> None:
    class _DummyActionHead:
        def get_rl_action(self, backbone_outputs, action_inputs, mode="eval"):
            bsz = backbone_outputs.shape[0]
            actions = backbone_outputs[:, None, :]
            return backbone_outputs, {
                "actions": actions,
                "chains": torch.zeros(bsz, 2, 1, 4),
                "denoise_inds": torch.zeros(bsz, 1, dtype=torch.long),
                "prev_logprobs": torch.zeros(bsz, 1, 1, 4),
                "prev_values": torch.zeros(bsz, 1),
            }

    class _DummyGr00tModel:
        def __init__(self) -> None:
            self.backbone = torch.nn.Linear(4, 4).to(torch.bfloat16)
            self.action_head = _DummyActionHead()
            self.padding_value = 8
            self.image_nums = 1
            self.output_action_chunks = 1

        def obs_convert_fn(self, env_obs):
            return {"state": env_obs["states"]}

        def _check_state_is_batched(self, obs):
            return True

        def apply_transforms(self, obs):
            state = torch.from_numpy(obs["state"]).float()
            bsz = state.shape[0]
            return {
                "state": state,
                "eagle_input_ids": torch.ones(bsz, 4, dtype=torch.long),
                "eagle_attention_mask": torch.ones(bsz, 4, dtype=torch.long),
                "eagle_pixel_values": torch.zeros(bsz, 3),
                "eagle_image_sizes": torch.zeros(bsz, 2),
            }

        def prepare_input(self, normalized_input):
            return normalized_input["state"], {"state": normalized_input["state"]}

        def validate_data(self, action_head_outputs, backbone_outputs, is_training=False):
            return None

        def _get_unnormalized_action(self, normalized_action):
            return {"action": normalized_action.cpu()}

        def action_convert_fn(self, unnormalized_action, chunk_size):
            return unnormalized_action["action"].detach().numpy()

    model = _DummyGr00tModel()
    adapter = GenericModelAdapter(model=model, model_type="gr00t")
    obs = {"states": torch.randn(2, 4)}

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        adapter.infer(obs_batch=obs, mode="eval")
    keys = [evt.key for evt in prof.key_averages()]

    assert any("model.backbone.gr00t.backbone" in key for key in keys)
    assert any("model.action_head.gr00t.action_head.get_rl_action" in key for key in keys)


def test_openvla_oft_logic_stage_profile_capture_backbone_and_action_head() -> None:
    class _DummyLanguageModel:
        def forward(self, x):
            return x + 1

    class _DummyOpenVLA:
        def __init__(self) -> None:
            self.language_model = _DummyLanguageModel()

        def _unnormalize_actions(self, actions, unnorm_key):
            return actions

        def predict_action_batch(self, env_obs=None, mode="eval", **kwargs):
            states = env_obs["states"]
            hidden = self.language_model.forward(states)
            actions = self._unnormalize_actions(hidden, "dummy")
            return actions, {}

    model = _DummyOpenVLA()
    adapter = GenericModelAdapter(model=model, model_type="openvla_oft")
    obs = {"states": torch.randn(2, 4)}

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        adapter.infer(obs_batch=obs, mode="eval")
    keys = [evt.key for evt in prof.key_averages()]

    assert any("model.backbone.openvla_oft.language_model.forward" in key for key in keys)
    assert any("model.action_head.openvla_oft._unnormalize_actions" in key for key in keys)
