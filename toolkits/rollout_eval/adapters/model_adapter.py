"""Model adapters for lightweight rollout evaluation."""

from __future__ import annotations

import types
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from omegaconf import DictConfig, open_dict

from rlinf.models import get_model
from toolkits.rollout_eval.profiling.torch_profiler import TARGET_SPLIT_MODELS


class ModelAdapterProtocol(Protocol):
    """Protocol for lightweight model inference."""

    def infer(
        self, obs_batch: dict[str, Any], mode: str = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]: ...


@contextmanager
def _temporary_profile_wrapper(target: Any, attr_name: str, stage_name: str):
    original = getattr(target, attr_name, None)
    if original is None:
        yield
        return

    if isinstance(original, types.MethodType):
        original_func = original.__func__

        def _wrapped(self_obj, *args, **kwargs):
            with torch.profiler.record_function(stage_name):
                return original_func(self_obj, *args, **kwargs)

        wrapped = types.MethodType(_wrapped, target)
    else:

        def _wrapped(*args, **kwargs):
            with torch.profiler.record_function(stage_name):
                return original(*args, **kwargs)

        wrapped = _wrapped

    setattr(target, attr_name, wrapped)
    try:
        yield
    finally:
        setattr(target, attr_name, original)


@dataclass
class _StageSpec:
    backbone_exact: tuple[str, ...] = ()
    backbone_contains: tuple[str, ...] = ()
    action_head_exact: tuple[str, ...] = ()
    action_head_contains: tuple[str, ...] = ()


MODEL_STAGE_SPECS: dict[str, _StageSpec] = {
    "openvla_oft": _StageSpec(
        backbone_exact=(
            "base_model.model.vision_backbone",
            "vision_backbone",
            "backbone",
        ),
        backbone_contains=("vision_backbone", "backbone"),
        action_head_exact=(
            "base_model.model.language_model.lm_head",
            "language_model.lm_head",
            "lm_head",
            "value_head",
        ),
        action_head_contains=(
            "lm_head",
            "value_head",
            "action_head",
            "policy_head",
            "projector",
        ),
    ),
    "openpi": _StageSpec(
        backbone_exact=(
            "paligemma_with_expert",
            "paligemma_with_expert.paligemma",
            "paligemma_with_expert.gemma_expert",
        ),
        backbone_contains=("paligemma_with_expert", "gemma_expert", "paligemma"),
        action_head_exact=("action_out_proj", "noise_head", "value_head", "q_head"),
        action_head_contains=(
            "action_out_proj",
            "noise_head",
            "value_head",
            "q_head",
            "action_head",
            "policy_head",
        ),
    ),
    "gr00t": _StageSpec(
        backbone_exact=("backbone",),
        backbone_contains=("backbone",),
        action_head_exact=("action_head",),
        action_head_contains=("action_head", "value_head", "action_decoder"),
    ),
}


@dataclass
class GenericModelAdapter:
    """Adapter that normalizes predict_action_batch outputs."""

    model: Any
    model_type: str
    split_model_stages: bool = True
    sampling_defaults: dict[str, Any] = field(default_factory=dict)
    _hook_handles: list[Any] = field(default_factory=list)
    _wrapped_forwards: list[tuple[torch.nn.Module, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.split_model_stages and self.model_type not in TARGET_SPLIT_MODELS:
            self._register_stage_profile_hooks()

    def _register_stage_profile_hooks(self) -> None:
        if not isinstance(self.model, torch.nn.Module):
            return
        if self.model_type == "gr00t":
            # GR00T stage profiling follows explicit eval-side logic in `infer`.
            return

        def _install_forward_wrapper(module: torch.nn.Module, stage_name: str) -> None:
            original_forward = getattr(module, "forward", None)
            if original_forward is None:
                return
            marker = "_rollout_eval_profile_wrapped"
            if getattr(module, marker, False):
                return

            def _wrapped_forward(*args, **kwargs):
                with torch.profiler.record_function(stage_name):
                    return original_forward(*args, **kwargs)

            setattr(module, marker, True)
            self._wrapped_forwards.append((module, original_forward))
            setattr(module, "forward", _wrapped_forward)

        def _install_hook(module: torch.nn.Module, stage_name: str) -> None:
            stack: list[Any] = []

            def _pre_hook(_module, _inputs):
                ctx = torch.profiler.record_function(stage_name)
                ctx.__enter__()
                stack.append(ctx)

            def _post_hook(_module, _inputs, _outputs):
                if stack:
                    ctx = stack.pop()
                    ctx.__exit__(None, None, None)

            self._hook_handles.append(module.register_forward_pre_hook(_pre_hook))
            self._hook_handles.append(module.register_forward_hook(_post_hook))

        backbone, action_head = _select_stage_modules(self.model, self.model_type)

        if backbone is not None:
            backbone_stage = f"model.backbone.{self.model_type}.{backbone[0]}"
            # OpenPI frequently calls paligemma_with_expert.forward(...) directly,
            # which bypasses nn.Module forward hooks. Wrap forward to ensure capture.
            if self.model_type == "openpi":
                _install_forward_wrapper(backbone[1], backbone_stage)
            else:
                _install_hook(backbone[1], backbone_stage)
        if action_head is not None:
            _install_hook(
                action_head[1],
                f"model.action_head.{self.model_type}.{action_head[0]}",
            )

    def infer(
        self, obs_batch: dict[str, Any], mode: str = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.model_type == "gr00t":
            return self._infer_gr00t(obs_batch=obs_batch, mode=mode)
        if self.model_type == "openpi":
            return self._infer_openpi(obs_batch=obs_batch, mode=mode)
        if self.model_type == "openvla_oft":
            return self._infer_openvla_oft(obs_batch=obs_batch, mode=mode)

        obs_batch = self._normalize_obs_for_model(obs_batch)
        if hasattr(self.model, "predict_action_batch"):
            kwargs = dict(self.sampling_defaults)
            actions, extra = self.model.predict_action_batch(
                env_obs=obs_batch,
                mode=mode,
                **kwargs,
            )
        else:
            actions = self.model(obs_batch)
            extra = {}

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)

        return actions, extra if isinstance(extra, dict) else {"raw_extra": extra}

    def _infer_gr00t(
        self, obs_batch: dict[str, Any], mode: str = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        from rlinf.models.embodiment.gr00t.utils import (
            squeeze_dict_values,
            unsqueeze_dict_values,
        )

        env_obs = dict(obs_batch)
        states = env_obs.get("states")
        if torch.is_tensor(states):
            env_obs["states"] = states.to(torch.bfloat16).cpu().float()

        observations = self.model.obs_convert_fn(env_obs)
        obs_copy = observations.copy()

        is_batch = self.model._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        for key, value in obs_copy.items():
            if not isinstance(value, np.ndarray):
                obs_copy[key] = np.array(value)

        normalized_input = self.model.apply_transforms(obs_copy)
        for key in normalized_input:
            if getattr(normalized_input[key], "dtype", None) == torch.float32:
                normalized_input[key] = normalized_input[key].to(torch.bfloat16)

        if (
            "eagle_input_ids" in normalized_input
            and hasattr(self.model, "padding_value")
            and normalized_input["eagle_input_ids"].shape[-1] < self.model.padding_value
        ):
            normalized_input["eagle_input_ids"] = torch.nn.functional.pad(
                normalized_input["eagle_input_ids"],
                pad=(
                    0,
                    self.model.padding_value - normalized_input["eagle_input_ids"].shape[-1],
                ),
                mode="constant",
                value=0,
            )
        if (
            "eagle_attention_mask" in normalized_input
            and hasattr(self.model, "padding_value")
            and normalized_input["eagle_attention_mask"].shape[-1]
            < self.model.padding_value
        ):
            normalized_input["eagle_attention_mask"] = torch.nn.functional.pad(
                normalized_input["eagle_attention_mask"],
                pad=(
                    0,
                    self.model.padding_value
                    - normalized_input["eagle_attention_mask"].shape[-1],
                ),
                mode="constant",
                value=0,
            )

        with torch.inference_mode():
            backbone_inputs, action_inputs = self.model.prepare_input(normalized_input)
            with torch.profiler.record_function("model.backbone.gr00t.backbone"):
                backbone_outputs = self.model.backbone(backbone_inputs)
            with torch.profiler.record_function(
                "model.action_head.gr00t.action_head.get_rl_action"
            ):
                action_head_outputs, rlinf_outputs = self.model.action_head.get_rl_action(
                    backbone_outputs, action_inputs, mode=mode
                )

        actions = rlinf_outputs["actions"]
        self.model.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        actions = actions.detach().float()

        forward_inputs = {
            "chains": rlinf_outputs["chains"],
            "denoise_inds": rlinf_outputs["denoise_inds"],
            **normalized_input,
        }
        if (
            "state" in normalized_input
            and "eagle_pixel_values" in normalized_input
            and "eagle_image_sizes" in normalized_input
            and hasattr(self.model, "image_nums")
        ):
            bsize = normalized_input["state"].shape[0]
            forward_inputs["eagle_pixel_values"] = normalized_input[
                "eagle_pixel_values"
            ].reshape(
                bsize, self.model.image_nums, *normalized_input["eagle_pixel_values"].shape[1:]
            )
            forward_inputs["eagle_image_sizes"] = normalized_input[
                "eagle_image_sizes"
            ].reshape(
                bsize, self.model.image_nums, *normalized_input["eagle_image_sizes"].shape[1:]
            )

        result = {
            "prev_logprobs": rlinf_outputs["prev_logprobs"],
            "prev_values": rlinf_outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }

        unnormalized_action = self.model._get_unnormalized_action(actions)
        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)

        raw_action = self.model.action_convert_fn(
            unnormalized_action,
            chunk_size=self.model.output_action_chunks,
        )
        return torch.from_numpy(raw_action), result

    def _infer_openpi(
        self, obs_batch: dict[str, Any], mode: str = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        obs_batch = self._normalize_obs_for_model(obs_batch)
        kwargs = dict(self.sampling_defaults)
        with ExitStack() as stack:
            if self.split_model_stages:
                backbone = getattr(self.model, "paligemma_with_expert", None)
                if backbone is not None:
                    stack.enter_context(
                        _temporary_profile_wrapper(
                            backbone,
                            "forward",
                            "model.backbone.openpi.paligemma_with_expert.forward",
                        )
                    )
                action_head = getattr(self.model, "action_out_proj", None)
                if action_head is not None:
                    stack.enter_context(
                        _temporary_profile_wrapper(
                            action_head,
                            "forward",
                            "model.action_head.openpi.action_out_proj.forward",
                        )
                    )
            actions, extra = self.model.predict_action_batch(
                env_obs=obs_batch,
                mode=mode,
                **kwargs,
            )
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        return actions, extra if isinstance(extra, dict) else {"raw_extra": extra}

    def _infer_openvla_oft(
        self, obs_batch: dict[str, Any], mode: str = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        kwargs = dict(self.sampling_defaults)
        actions, extra = self.model.predict_action_batch(
            env_obs=obs_batch,
            mode=mode,
            **kwargs,
        )
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        return actions, extra if isinstance(extra, dict) else {"raw_extra": extra}

    def _normalize_obs_for_model(self, obs_batch: dict[str, Any]) -> dict[str, Any]:
        if self.model_type != "openpi" or not isinstance(obs_batch, dict):
            return obs_batch
        normalized = dict(obs_batch)
        normalized.setdefault("wrist_images", normalized.get("extra_view_images"))
        normalized.setdefault("extra_view_images", None)
        return normalized


def _first_exact(
    module_map: dict[str, torch.nn.Module], candidates: tuple[str, ...]
) -> tuple[str, torch.nn.Module] | None:
    for name in candidates:
        if name in module_map:
            return name, module_map[name]
    return None


def _first_contains(
    module_map: dict[str, torch.nn.Module], candidates: tuple[str, ...]
) -> tuple[str, torch.nn.Module] | None:
    for name, module in module_map.items():
        if not name:
            continue
        lname = name.lower()
        if any(token in lname for token in candidates):
            return name, module
    return None


def _fallback_stage_modules(
    module_map: dict[str, torch.nn.Module], model: torch.nn.Module
) -> tuple[tuple[str, torch.nn.Module] | None, tuple[str, torch.nn.Module] | None]:
    backbone_candidates = []
    action_head_candidates = []

    for name, module in module_map.items():
        lname = name.lower()
        if any(k in lname for k in ["backbone", "vision_backbone"]):
            backbone_candidates.append((name, module))
        if any(k in lname for k in ["action_head", "policy_head", "lm_head"]):
            action_head_candidates.append((name, module))

    if not backbone_candidates and hasattr(model, "backbone"):
        backbone_candidates.append(("backbone", getattr(model, "backbone")))
    if not action_head_candidates and hasattr(model, "action_head"):
        action_head_candidates.append(("action_head", getattr(model, "action_head")))

    return (
        backbone_candidates[0] if backbone_candidates else None,
        action_head_candidates[0] if action_head_candidates else None,
    )


def _select_stage_modules(
    model: torch.nn.Module, model_type: str
) -> tuple[tuple[str, torch.nn.Module] | None, tuple[str, torch.nn.Module] | None]:
    module_map = dict(model.named_modules())
    spec = MODEL_STAGE_SPECS.get(model_type)
    if spec is None:
        return _fallback_stage_modules(module_map, model)

    backbone = _first_exact(module_map, spec.backbone_exact)
    if backbone is None:
        backbone = _first_contains(module_map, spec.backbone_contains)

    action_head = _first_exact(module_map, spec.action_head_exact)
    if action_head is None:
        action_head = _first_contains(module_map, spec.action_head_contains)

    if backbone is None or action_head is None:
        fb_backbone, fb_action_head = _fallback_stage_modules(module_map, model)
        if backbone is None:
            backbone = fb_backbone
        if action_head is None:
            action_head = fb_action_head

    return backbone, action_head



def _resolve_model_path(cfg: DictConfig) -> str:
    if "rollout" in cfg and "model" in cfg.rollout and "model_path" in cfg.rollout.model:
        return str(cfg.rollout.model.model_path)
    return str(cfg.actor.model.model_path)



def _validate_model_path_or_raise(model_path: str, model_type: str | None = None) -> None:
    path = Path(model_path)
    if not path.exists():
        raise RuntimeError(f"Model path does not exist: {model_path}")

    if path.is_dir():
        if model_type == "openpi":
            has_pt_checkpoint = (
                (path / "model_state_dict" / "full_weights.pt").exists()
                or (path / "actor" / "model_state_dict" / "full_weights.pt").exists()
            )
            has_safetensors = any(path.glob("*.safetensors")) or (
                path / "model.safetensors"
            ).exists()
            if not (has_pt_checkpoint or has_safetensors):
                raise RuntimeError(
                    "OpenPI model directory must contain either checkpoint weights "
                    "(model_state_dict/full_weights.pt or actor/model_state_dict/full_weights.pt) "
                    f"or safetensors files (*.safetensors / model.safetensors): {model_path}"
                )
            return

        has_config = (path / "config.json").exists()
        has_adapter_only = (path / "adapter_config.json").exists() and not has_config
        if has_adapter_only:
            raise RuntimeError(
                "Model path points to a LoRA adapter-only directory (has adapter_config.json "
                f"but no config.json): {model_path}. Please provide the base model path in "
                "actor.model.model_path/rollout.model.model_path and keep LoRA path in actor.model.lora_path."
            )
        if not has_config:
            raise RuntimeError(
                f"Model directory missing required config.json: {model_path}"
            )



@dataclass
class NullModelAdapter:
    """Model adapter that returns zero actions without loading any model.

    Useful for profiling environment step performance in isolation.
    """

    action_dim: int = 7
    num_action_chunks: int = 1

    def infer(
        self, obs_batch: dict[str, Any], mode: str = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        batch_size = _infer_batch_size_from_obs(obs_batch)
        if self.num_action_chunks > 1:
            actions = torch.zeros(batch_size, self.num_action_chunks, self.action_dim)
        else:
            actions = torch.zeros(batch_size, self.action_dim)
        return actions, {}


def _infer_batch_size_from_obs(obs_batch: dict[str, Any]) -> int:
    import numpy as np

    for value in obs_batch.values():
        if isinstance(value, (torch.Tensor, np.ndarray)) and hasattr(value, "shape"):
            return int(value.shape[0])
        if isinstance(value, list) and len(value) > 0:
            return len(value)
    return 1


def build_null_model_adapter(
    cfg: DictConfig,
    action_dim_override: int | None = None,
    num_action_chunks_override: int | None = None,
) -> NullModelAdapter:
    """Build a null model adapter (no model loaded) for env-only profiling.

    Args:
        cfg: Full RLinf Hydra config (used to infer action shape when not overridden).
        action_dim_override: Override for action dimension. If None, inferred from config.
        num_action_chunks_override: Override for number of action chunks.

    Returns:
        A NullModelAdapter that produces zero actions.
    """
    actor_model = cfg.get("actor", {}).get("model", {})

    if action_dim_override is not None:
        action_dim = action_dim_override
    else:
        action_dim = int(actor_model.get("action_dim", 7))

    if num_action_chunks_override is not None:
        num_action_chunks = num_action_chunks_override
    else:
        num_action_chunks = int(actor_model.get("num_action_chunks", 1))

    print(
        f"[NullModelAdapter] action_dim={action_dim} num_action_chunks={num_action_chunks} "
        f"(env-only mode, no model loaded)"
    )
    return NullModelAdapter(action_dim=action_dim, num_action_chunks=num_action_chunks)


def build_model_adapter(
    cfg: DictConfig, split_model_stages: bool = True
) -> ModelAdapterProtocol:
    """Build model adapter from RLinf config.

    Raises:
        RuntimeError: If model path or model construction is invalid.
    """
    model_path = _resolve_model_path(cfg)
    _validate_model_path_or_raise(model_path, model_type=str(cfg.actor.model.model_type))

    model_cfg = cfg.actor.model.copy()
    with open_dict(model_cfg):
        if "openpi_data" in cfg:
            model_cfg.openpi_data = cfg.openpi_data
        if "rollout" in cfg and "model" in cfg.rollout:
            if "precision" in cfg.rollout.model:
                model_cfg.precision = cfg.rollout.model.precision
            if "model_path" in cfg.rollout.model:
                model_cfg.model_path = cfg.rollout.model.model_path

    try:
        model = get_model(model_cfg)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to build model adapter from path '{model_path}': "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    model.eval()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError(
            "Distributed process group is initialized, but rollout_eval expects single-process non-distributed mode."
        )

    sampling_cfg = cfg.algorithm.get("sampling_params", {})
    temp_eval = float(sampling_cfg.get("temperature_eval", -1))
    do_sample = bool(temp_eval > 0)
    sampling_defaults = {
        "do_sample": do_sample,
        "temperature": temp_eval if do_sample else 1.0,
        "top_k": int(sampling_cfg.get("top_k", 0)),
    }

    return GenericModelAdapter(
        model=model,
        model_type=str(cfg.actor.model.model_type),
        split_model_stages=split_model_stages,
        sampling_defaults=sampling_defaults,
    )
