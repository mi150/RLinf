# Copyright 2026 The RLinf Authors.
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

import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from rlinf.utils.logging import get_logger


def _promote_scalar_params_to_1d(model):
    """FSDP does not support 0-d parameters, so we promote scalar Parameters to shape=[1]."""
    scalar_param_names = [name for name, p in model.named_parameters() if p.ndim == 0]
    for full_name in scalar_param_names:
        if "." in full_name:
            module_name, param_name = full_name.rsplit(".", 1)
            module = model.get_submodule(module_name)
        else:
            module = model
            param_name = full_name

        old_p = getattr(module, param_name)
        new_p = nn.Parameter(
            old_p.detach().reshape(1),
            requires_grad=old_p.requires_grad,
        )
        setattr(module, param_name, new_p)


def _dreamzero_disable_torch_compile() -> bool:
    return os.getenv("DREAMZERO_DISABLE_TORCH_COMPILE", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _disable_torch_compile_for_dreamzero():
    """Turn torch.compile into a no-op before importing DreamZero Wan modules."""

    if getattr(torch.compile, "_rlinf_dreamzero_noop", False):
        return

    def _compile_noop(model=None, *args, **kwargs):
        if model is None:
            def decorator(fn):
                return fn

            return decorator
        return model

    _compile_noop._rlinf_dreamzero_noop = True
    torch.compile = _compile_noop


def get_model(cfg: DictConfig, torch_dtype=None):
    """Load DreamZero policy from checkpoint."""

    started_at = time.monotonic()
    logger = get_logger()

    def log_stage(message: str, *args):
        logger.info("[DreamZero load %.1fs] " + message, time.monotonic() - started_at, *args)

    log_stage(
        "start model_path=%s lora=%s dtype=%s",
        cfg.get("model_path"),
        cfg.get("lora_weights_path", None) or os.getenv("DREAMZERO_LORA_PATH", ""),
        torch_dtype,
    )

    if _dreamzero_disable_torch_compile():
        log_stage("disable torch.compile wrappers")
        _disable_torch_compile_for_dreamzero()

    from groot.vla.data.transform import ComposedModalityTransform
    from safetensors.torch import load_file

    from rlinf.models.embodiment.dreamzero.dreamzero_policy import (
        DreamZeroConfig,
        DreamZeroPolicy,
    )
    from rlinf.data.datasets.dreamzero.data_transforms import (
        build_dreamzero_composed_transform,
        build_dreamzero_observation_transform,
        load_dreamzero_dataset_metadata,
    )
    from rlinf.utils.patcher import Patcher
    from rlinf.models.embodiment.dreamzero.patch.wan_policy_head_action_only import (
        flow_unipc_step,
        lazy_joint_video_action,
    )

    Patcher.clear()
    Patcher.add_patch(
        "groot.vla.model.dreamzero.modules.wan_video_vae.WanVideoVAE",
        "rlinf.models.embodiment.dreamzero.patch.wan_video_vae.WanVideoVAE",
    )
    Patcher.add_patch(
        "groot.vla.model.dreamzero.modules.wan_video_vae.WanVideoVAE38",
        "rlinf.models.embodiment.dreamzero.patch.wan_video_vae.WanVideoVAE38",
    )
    Patcher.add_patch(
        "groot.vla.model.dreamzero.modules.wan_video_vae.WanVideoVAEStateDictConverter",
        "rlinf.models.embodiment.dreamzero.patch.wan_video_vae.WanVideoVAEStateDictConverter",
    )
    _dit_chunk = "groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk"
    if not _dreamzero_disable_torch_compile():
        Patcher.add_wrapper(
            f"{_dit_chunk}.CausalWanSelfAttention._process_clean_image_only",
            torch.compile(mode="reduce-overhead"),
        )
        Patcher.add_wrapper(
            f"{_dit_chunk}.CausalWanSelfAttention._process_state_blocks",
            torch.compile(mode="reduce-overhead"),
        )
        Patcher.add_wrapper(
            f"{_dit_chunk}.CausalWanSelfAttention._process_noisy_image_blocks",
            torch.compile(mode="reduce-overhead"),
        )
        Patcher.add_wrapper(
            f"{_dit_chunk}.CausalWanSelfAttention._process_noisy_action_blocks",
            torch.compile(mode="reduce-overhead"),
        )
    Patcher.add_patch(
        f"{_dit_chunk}.CausalWanModel._forward_train",
        "rlinf.models.embodiment.dreamzero.patch.wan_causal_model_forward_train._forward_train",
    )
    Patcher.add_patch(
        f"{_dit_chunk}.CausalWanModel._forward_blocks",
        "rlinf.models.embodiment.dreamzero.patch.wan_causal_model_forward_inference._forward_blocks",
    )
    Patcher.add_patch(
        f"{_dit_chunk}.CausalWanModel._forward_inference",
        "rlinf.models.embodiment.dreamzero.patch.wan_causal_model_forward_inference._forward_inference",
    )
    _wan_policy_head = "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf"
    Patcher.add_patch(
        f"{_wan_policy_head}.WANPolicyHead._run_diffusion_steps",
        "rlinf.models.embodiment.dreamzero.patch.wan_policy_head_action_only._run_diffusion_steps",
    )
    Patcher.add_wrapper(
        f"{_wan_policy_head}.WANPolicyHead.lazy_joint_video_action",
        lazy_joint_video_action,
    )
    Patcher.add_wrapper(
        "groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler.FlowUniPCMultistepScheduler.step",
        flow_unipc_step,
    )
    Patcher.apply()
    log_stage("patches applied")

    model_path = Path(cfg.get("model_path"))
    if not model_path.exists():
        raise FileNotFoundError(f"DreamZero model_path does not exist: {model_path}")

    tokenizer_path = cfg.get("tokenizer_path", "google/umt5-xxl")

    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config_dict = json.load(f)

    dreamzero_config = DreamZeroConfig(**config_dict)
    log_stage("config loaded")

    st = model_path / "model.safetensors"
    st_index = model_path / "model.safetensors.index.json"
    has_full_model_weights = st.exists() or st_index.exists()

    if "config" in dreamzero_config.action_head_cfg and isinstance(
        dreamzero_config.action_head_cfg["config"], dict
    ):
        # If full DreamZero safetensors are absent, fall back to component loading from
        # WAN paths in config.json (diffusion_model_pretrained_path / text / image / vae).
        dreamzero_config.action_head_cfg["config"]["skip_component_loading"] = (
            has_full_model_weights
        )

    dreamzero_config.env_action_dim = cfg.get("env_action_dim", 7)
    dreamzero_config.gradient_checkpointing = cfg.get("gradient_checkpointing", False)

    embodiment_tag = cfg.get("embodiment_tag", "libero_sim")
    metadata = load_dreamzero_dataset_metadata(cfg)
    data_transforms = build_dreamzero_composed_transform(cfg, tokenizer_path)
    assert isinstance(data_transforms, ComposedModalityTransform), f"{data_transforms=}"
    data_transforms.set_metadata(metadata)
    data_transforms.eval()
    log_stage("metadata/transforms ready for embodiment_tag=%s", embodiment_tag)

    dreamzero_config.data_transforms = data_transforms
    dreamzero_config.observation_transform = build_dreamzero_observation_transform(
        embodiment_tag, cfg
    )
    dreamzero_config.embodiment_tag = str(embodiment_tag)
    dreamzero_config.relative_action = bool(cfg.get("relative_action", False))
    dreamzero_config.relative_action_per_horizon = bool(
        cfg.get("relative_action_per_horizon", False)
    )
    dreamzero_config.relative_action_keys = list(cfg.get("relative_action_keys") or [])

    model = DreamZeroPolicy(
        config=dreamzero_config,
    )
    log_stage("DreamZeroPolicy constructed")

    # Load DreamZero full weights if available; otherwise keep component-initialized model.
    if has_full_model_weights:
        state_dict = {}
        if st_index.exists():
            with open(st_index, "r") as f:
                index = json.load(f)
            shard_files = sorted(set(index["weight_map"].values()))
            log_stage("loading %d safetensors shards", len(shard_files))
            for shard_idx, shard_file in enumerate(shard_files, 1):
                log_stage("loading shard %d/%d: %s", shard_idx, len(shard_files), shard_file)
                state_dict.update(load_file(str(model_path / shard_file)))
        elif st.exists():
            log_stage("loading single safetensors file: %s", st.name)
            state_dict.update(load_file(str(st)))
        log_stage("loaded state_dict with %d tensors", len(state_dict))
        if any(".base_layer." in k for k in state_dict):
            state_dict = {
                k.replace(".base_layer.", "."): v for k, v in state_dict.items()
            }
        model_keys = set(model.state_dict().keys())
        if any(k.startswith("action_head.model.base_model.model.") for k in model_keys):
            remapped_state_dict = {}
            for key, value in state_dict.items():
                mapped_key = (
                    key.replace(
                        "action_head.model.",
                        "action_head.model.base_model.model.",
                        1,
                    )
                    if key.startswith("action_head.model.")
                    and not key.startswith("action_head.model.base_model.model.")
                    else key
                )
                if mapped_key not in model_keys:
                    for suffix in (".weight", ".bias"):
                        if mapped_key.endswith(suffix):
                            base_layer_key = (
                                mapped_key[: -len(suffix)]
                                + ".base_layer"
                                + suffix
                            )
                            if base_layer_key in model_keys:
                                mapped_key = base_layer_key
                            break
                remapped_state_dict[mapped_key] = value
            state_dict = remapped_state_dict
        missing, unexpected = model.load_state_dict(
            state_dict, strict=False, assign=True
        )
        log_stage(
            "load_state_dict finished missing=%d unexpected=%d",
            len(missing),
            len(unexpected),
        )
        action_head = getattr(model, "action_head", None)
        action_head_config = getattr(action_head, "config", None)
        if (
            action_head is not None
            and hasattr(action_head, "inject_lora_after_loading")
            and bool(getattr(action_head_config, "defer_lora_injection", False))
        ):
            action_head.inject_lora_after_loading()
            log_stage("deferred action_head LoRA injected")
    else:
        get_logger().warning(
            "No model.safetensors under %s; initializing DreamZero from component weights "
            "configured in config.json (WAN diffusion/text/image/vae paths).",
            model_path,
        )

    lora_weights_path = cfg.get("lora_weights_path", None) or os.getenv(
        "DREAMZERO_LORA_PATH", ""
    )
    if lora_weights_path:
        lora_weights_path = Path(lora_weights_path)
        if not lora_weights_path.exists():
            raise FileNotFoundError(
                f"DreamZero lora_weights_path does not exist: {lora_weights_path}"
            )
        get_logger().info("Loading DreamZero LoRA weights from %s", lora_weights_path)
        model.load_lora_weight(str(lora_weights_path))
        log_stage("LoRA weights loaded from %s", lora_weights_path)

    _promote_scalar_params_to_1d(model)
    log_stage("scalar parameters promoted")
    model = model.to(dtype=torch_dtype)
    log_stage("model dtype/device conversion finished")

    return model
