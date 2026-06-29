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

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from groot.vla.data.transform import ComposedModalityTransform
from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig
from tianshou.data import Batch
from transformers.configuration_utils import PretrainedConfig

from rlinf.algorithms.dreamzero import set_dreamzero_loss_payload
from rlinf.data.datasets.dreamzero.data_transforms import (
    DreamZeroObservationTransform,
)
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.dreamzero.world_model import DreamZeroWorldModel
from rlinf.utils.logging import get_logger


@dataclass
class DreamZeroConfig(VLAConfig):
    model_type = "dreamzero"
    backbone_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Backbone configuration."}
    )

    action_head_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Action head configuration."}
    )

    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})

    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    env_action_dim: int = field(
        default=None, metadata={"help": "Environment action dimension."}
    )
    num_action_chunks: int = field(
        default=16, metadata={"help": "Number of action chunks."}
    )

    relative_action: bool = field(default=False, metadata={"help": "Relative action."})
    relative_action_per_horizon: bool = field(
        default=False, metadata={"help": "Relative action per horizon."}
    )
    relative_action_keys: list = field(
        default_factory=list, metadata={"help": "Relative action keys."}
    )

    data_transforms: ComposedModalityTransform = field(
        default=None,
        metadata={
            "help": "Transforming data modalities, e.g. video frame augmentation or action normalization."
        },
    )

    gradient_checkpointing: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class DreamZeroPolicy(VLA, BasePolicy):
    """Lightweight DreamZero action model: IdentityBackbone + WANPolicyHead."""

    _no_split_modules = [
        "T5SelfAttention",  # text encoder
        "AttentionBlock",  # vae
        "CausalWanAttentionBlock",  # action head
    ]

    def __init__(
        self,
        config: DreamZeroConfig,
    ):
        super().__init__(config)
        self.config = config
        try:
            diffusion_model = getattr(getattr(self, "action_head", None), "model", None)
            enabled = self.config.gradient_checkpointing
            if diffusion_model is not None:
                if hasattr(diffusion_model, "_set_gradient_checkpointing"):
                    diffusion_model._set_gradient_checkpointing(
                        diffusion_model, enabled
                    )
                elif hasattr(diffusion_model, "gradient_checkpointing"):
                    diffusion_model.gradient_checkpointing = enabled
        except Exception:
            pass
        self.world_model: DreamZeroWorldModel | None = None
        self.observation_transform: DreamZeroObservationTransform | None = getattr(
            config, "observation_transform", None
        )

    def apply(self, batch: Batch, **kwargs) -> Batch:
        """Normalize inputs"""
        obs = batch.obs
        normalized_input = self.config.data_transforms(obs)
        batch.normalized_obs = normalized_input
        return batch

    def unapply(self, batch: Batch, obs: Optional[dict] = None, **kwargs):
        """Unnormalize actions and convert relative actions to absolute if needed"""
        unnormalized_action = self.config.data_transforms.unapply(
            {"action": batch.normalized_action.cpu()}
        )

        # Check if relative_action is enabled and convert relative to absolute
        relative_action = self.config.relative_action
        relative_action_per_horizon = self.config.relative_action_per_horizon
        relative_action_keys = self.config.relative_action_keys
        if (
            (relative_action or relative_action_per_horizon)
            and relative_action_keys
            and obs is not None
        ):
            for key in relative_action_keys:
                action_key = f"action.{key}"
                state_key = f"state.{key}"

                if action_key not in unnormalized_action:
                    continue

                # Try to find the state data - check multiple possible key formats
                last_state = None

                # Format 1: Direct key like "state.joint_position"
                if state_key in obs:
                    last_state = obs[state_key]
                else:
                    # Format 2: Search for keys containing both "state" and the key name
                    for obs_key in obs.keys():
                        if "state" in obs_key and key in obs_key:
                            last_state = obs[obs_key]
                            break

                    # Format 3: If key is "joint_position" and obs has "state" key directly
                    # This handles cases where the observation uses modality-level keys
                    if last_state is None and "state" in obs:
                        state_data = obs["state"]
                        # Check if the state data shape matches the action shape
                        action_dim = unnormalized_action[action_key].shape[-1]
                        if torch.is_tensor(state_data):
                            state_dim = state_data.shape[-1]
                        elif isinstance(state_data, np.ndarray):
                            state_dim = state_data.shape[-1]
                        else:
                            state_dim = None

                        if state_dim == action_dim:
                            last_state = state_data

                if last_state is None:
                    continue

                if torch.is_tensor(last_state):
                    last_state = last_state.cpu().numpy()

                # Shape is (B, T, D) or (T, D), we want the last timestep
                # After indexing: (B, D) or (D,)
                if len(last_state.shape) >= 2:
                    last_state = last_state[..., -1, :]  # Get the last timestep

                # Action shape is (horizon, D) or (B, horizon, D)
                # Expand dims to broadcast: (D,) -> (1, D) or (B, D) -> (B, 1, D)
                if len(unnormalized_action[action_key].shape) > len(last_state.shape):
                    last_state = np.expand_dims(
                        last_state, axis=-2
                    )  # Add horizon dimension

                # Add state to relative action to get absolute action
                unnormalized_action[action_key] = (
                    unnormalized_action[action_key] + last_state
                )

        batch.act = unnormalized_action
        return batch

    def _process_batch(self, batch: Batch) -> Batch:
        """Process batch."""
        self._sync_action_head_device()
        # Normalize / transform
        batch = self.apply(batch)
        normalized_input = batch.normalized_obs
        # If the normalized input is still a Batch, flatten it into a pure dict
        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()
        # Do dtype cast if needed
        target_dtype = next(self.parameters()).dtype
        for k, v in normalized_input.items():
            if (
                torch.is_tensor(v)
                and v.dtype == torch.float32
                and target_dtype != torch.float32
            ):
                normalized_input[k] = v.to(dtype=target_dtype)
        return normalized_input

    def _sync_action_head_device(self) -> None:
        action_head = getattr(self, "action_head", None)
        if action_head is None:
            return
        try:
            device = next(self.parameters()).device
        except StopIteration:
            return
        current = getattr(action_head, "_device", None)
        if current is None or torch.device(current) != device:
            action_head._device = str(device)
            if hasattr(action_head, "_vae_device_ready"):
                action_head._vae_device_ready = False
        if not hasattr(action_head, "trt_engine"):
            action_head.trt_engine = None
        if not hasattr(action_head, "trt_context"):
            action_head.trt_context = None

    @staticmethod
    def _to_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        return tensor

    @staticmethod
    def _action_debug_stats(value: Any) -> dict[str, Any]:
        arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        arr = np.asarray(arr)
        flat = arr.astype(np.float32, copy=False).reshape(-1)
        if flat.size == 0:
            return {"shape": tuple(arr.shape), "empty": True}
        return {
            "shape": tuple(arr.shape),
            "min": float(np.nanmin(flat)),
            "max": float(np.nanmax(flat)),
            "mean": float(np.nanmean(flat)),
            "std": float(np.nanstd(flat)),
            "sample": arr.reshape(-1, arr.shape[-1])[:2].tolist()
            if arr.ndim > 0
            else arr.tolist(),
        }

    def _debug_action_payload(
        self,
        *,
        normalized_action: torch.Tensor,
        unnormalized_action: dict[str, Any],
        env_actions: np.ndarray,
    ) -> None:
        if os.getenv("DREAMZERO_DEBUG_ACTIONS", "0").lower() not in (
            "1",
            "true",
            "yes",
            "on",
        ):
            return
        if getattr(self, "_debug_action_logged", 0) >= int(
            os.getenv("DREAMZERO_DEBUG_ACTIONS_LIMIT", "3")
        ):
            return
        self._debug_action_logged = getattr(self, "_debug_action_logged", 0) + 1
        logger = get_logger()
        logger.info(
            "[DreamZero action debug] normalized_action=%s",
            self._action_debug_stats(normalized_action),
        )
        for key, value in sorted(unnormalized_action.items()):
            if hasattr(value, "shape"):
                logger.info(
                    "[DreamZero action debug] unnormalized %s=%s",
                    key,
                    self._action_debug_stats(value),
                )
        logger.info(
            "[DreamZero action debug] env_actions=%s gripper_counts=%s",
            self._action_debug_stats(env_actions),
            {
                str(k): int(v)
                for k, v in zip(
                    *np.unique(env_actions[..., -1].reshape(-1), return_counts=True)
                )
            },
        )

    @staticmethod
    def _map_gripper_for_env(actions: np.ndarray) -> np.ndarray:
        mode = os.getenv("DREAMZERO_GRIPPER_MODE", "pm1").lower()
        raw = actions[..., -1].copy()
        if mode in ("pm1", "plus_minus_one", "default"):
            actions[..., -1] = np.where(raw > 0, 1.0, -1.0).astype(actions.dtype)
        elif mode in ("zero_one", "01"):
            actions[..., -1] = np.where(raw > 0.5, 1.0, 0.0).astype(actions.dtype)
        elif mode in ("flip_pm1", "pm1_flip"):
            actions[..., -1] = np.where(raw > 0, -1.0, 1.0).astype(actions.dtype)
        elif mode in ("raw", "none"):
            actions[..., -1] = raw.astype(actions.dtype)
        else:
            raise ValueError(
                "Unsupported DREAMZERO_GRIPPER_MODE="
                f"{mode!r}; use pm1, zero_one, flip_pm1, or raw."
            )
        return actions

    def _flatten_rl_tensor_payload(
        self, normalized_input: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        flat: dict[str, torch.Tensor] = {}
        for key, value in normalized_input.items():
            if value is None:
                continue
            tensor = self._to_tensor(value)
            flat[f"dreamzero_rl.{key}"] = tensor.detach().cpu().contiguous()
        return flat

    @staticmethod
    def _expand_video_for_action_head(
        images: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if images.ndim < 5:
            return images
        action_steps = actions.shape[-2] if actions.ndim >= 3 else 0
        if action_steps <= 0:
            return images
        num_action_per_block = 24
        num_frame_per_block = 2
        blocks = max(1, action_steps // num_action_per_block)
        target_frames = blocks * num_frame_per_block * 4 + 1
        if images.shape[1] == target_frames:
            return images
        if images.shape[1] > target_frames:
            return images[:, -target_frames:]
        repeat_shape = (images.shape[0], target_frames - images.shape[1], *images.shape[2:])
        pad = images[:, :1].expand(repeat_shape)
        return torch.cat([pad, images], dim=1)

    def _normalize_forward_payload_for_rollout(
        self, forward_inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        images = forward_inputs.get("dreamzero_rl.images")
        actions = forward_inputs.get("dreamzero_rl.action")
        if torch.is_tensor(images) and torch.is_tensor(actions):
            forward_inputs = dict(forward_inputs)
            forward_inputs["dreamzero_rl.images"] = (
                self._expand_video_for_action_head(images, actions)
                .detach()
                .cpu()
                .contiguous()
            )
        return forward_inputs

    @staticmethod
    def _dreamzero_action_logprob_std(default: float = 0.02) -> float:
        raw = os.getenv("DREAMZERO_ACTION_LOGPROB_STD", str(default))
        std = float(raw)
        if std <= 0:
            raise ValueError("DREAMZERO_ACTION_LOGPROB_STD must be positive.")
        return std

    @staticmethod
    def _gaussian_action_logprob(
        action: torch.Tensor,
        mean: torch.Tensor,
        *,
        std: float,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        action = action.to(device=mean.device, dtype=torch.float32)
        mean = mean.float()
        if action.shape != mean.shape:
            action = action.reshape_as(mean)
        var = std * std
        log_scale = float(np.log(std))
        log_two_pi = float(np.log(2.0 * np.pi))
        logprob = -0.5 * ((action - mean).pow(2) / var + 2.0 * log_scale + log_two_pi)
        if mask is not None:
            mask = mask.to(device=logprob.device, dtype=torch.bool)
            if mask.shape != logprob.shape:
                mask = mask.reshape_as(logprob)
            logprob = torch.where(mask, logprob, torch.zeros_like(logprob))
        return logprob.reshape(logprob.shape[0], -1).sum(dim=-1)

    def _build_action_chain_payload(
        self,
        *,
        normalized_action: torch.Tensor,
        mean_action: torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        std = self._dreamzero_action_logprob_std()
        old_logprob = self._gaussian_action_logprob(
            normalized_action,
            mean_action,
            std=std,
            mask=action_mask,
        )
        timesteps = torch.zeros(
            normalized_action.shape[0], 1, dtype=torch.long, device=normalized_action.device
        )
        return {
            "dreamzero_action_chain": torch.stack(
                [mean_action.detach(), normalized_action.detach()], dim=1
            )
            .cpu()
            .contiguous(),
            "dreamzero_action_timesteps": timesteps.detach().cpu().contiguous(),
            "dreamzero_old_action_logprob": old_logprob.detach().cpu().contiguous(),
            "dreamzero_action_logprob_std": torch.full(
                (normalized_action.shape[0],),
                std,
                dtype=torch.float32,
                device=normalized_action.device,
            )
            .cpu()
            .contiguous(),
        }

    @staticmethod
    def _restore_rl_tensor_payload(forward_inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            key.removeprefix("dreamzero_rl."): value
            for key, value in forward_inputs.items()
            if key.startswith("dreamzero_rl.")
        }

    def _reset_action_chain_inference_state(self) -> None:
        action_head = getattr(self, "action_head", None)
        if action_head is None:
            return
        if hasattr(action_head, "release_inference_cache"):
            action_head.release_inference_cache()
            return
        for name in (
            "kv_cache1",
            "kv_cache_neg",
            "crossattn_cache",
            "crossattn_cache_neg",
            "clip_feas",
            "ys",
            "language",
        ):
            if hasattr(action_head, name):
                setattr(action_head, name, None)
        if hasattr(action_head, "current_start_frame"):
            action_head.current_start_frame = 0
        if hasattr(action_head, "skip_countdown"):
            action_head.skip_countdown = 0

    def release_inference_cache(self) -> None:
        self._reset_action_chain_inference_state()

    def _infer_real_action_dim(self, fallback: int) -> int:
        transforms = getattr(getattr(self.config, "data_transforms", None), "transforms", [])
        for transform in reversed(transforms):
            action_order = getattr(transform, "action_concat_order", None)
            if not action_order:
                continue
            try:
                return sum(
                    transform.get_state_action_dims_post_transform(key)
                    for key in action_order
                )
            except Exception:
                return fallback
        return fallback

    def _extract_env_actions(self, act: dict[str, Any]) -> np.ndarray:
        if "action.actions" in act:
            actions = act["action.actions"]
        elif "action" in act:
            actions = act["action"]
        else:
            action_keys = [
                key
                for key in ("action.joint_position", "action.gripper_position")
                if key in act
            ]
            if not action_keys:
                action_keys = sorted(
                    key
                    for key, value in act.items()
                    if key.startswith("action.") and hasattr(value, "shape")
                )
            if not action_keys:
                raise KeyError(f"No action tensors found in DreamZero output: {list(act.keys())}")
            tensors = [act[key] for key in action_keys]
            if any(torch.is_tensor(tensor) for tensor in tensors):
                tensors = [
                    tensor if torch.is_tensor(tensor) else torch.as_tensor(tensor)
                    for tensor in tensors
                ]
                actions = torch.cat(tensors, dim=-1)
            else:
                actions = np.concatenate(tensors, axis=-1)
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        env_action_dim = self.config.env_action_dim
        if (
            env_action_dim is not None
            and actions.ndim == 3
            and actions.shape[-2] == env_action_dim
            and actions.shape[-1] != env_action_dim
        ):
            actions = np.swapaxes(actions, -1, -2)
        if env_action_dim is not None and actions.shape[-1] != env_action_dim:
            if actions.shape[-1] > env_action_dim and env_action_dim >= 2:
                actions = np.concatenate(
                    [actions[..., : env_action_dim - 1], actions[..., -1:]], axis=-1
                )
            elif actions.shape[-1] > env_action_dim:
                actions = actions[..., :env_action_dim]
            else:
                padded = np.zeros((*actions.shape[:-1], env_action_dim), dtype=actions.dtype)
                padded[..., : actions.shape[-1]] = actions
                actions = padded
        num_action_chunks = self.config.num_action_chunks
        if num_action_chunks is not None and actions.ndim >= 3:
            actions = actions[:, :num_action_chunks, :]
        return actions

    def _observation_convert(self, env_obs: dict) -> dict:
        """Convert environment observation to DreamZero model input."""
        if self.observation_transform is None:
            raise RuntimeError(
                "DreamZeroPolicy requires config.observation_transform for env "
                "observation conversion."
            )
        converted_obs = self.observation_transform.convert(env_obs)
        prompts = converted_obs.get("annotation.language.task_description", [])
        if os.getenv("DREAMZERO_DEBUG_LANGUAGE", "0").lower() in ("1", "true", "yes"):
            preview = list(prompts)[: min(2, len(prompts))]
            get_logger().info("[DreamZero language] task_descriptions=%s", preview)
        return converted_obs

    def predict_action_batch(self, env_obs, mode, **kwargs) -> np.ndarray:
        """
        input:
            env_obs:
                - main_images: [B,H,W,C] uint8
                - extra_view_images: [B,H,W,C]
                - states: [B,D]
                - task_descriptions: list[str] or None
        output:
            actions: np.ndarray [B, num_action_chunks, 8]  # 6ee + 1 gripper
            result: dict  # compatible with rollout interface"""

        converted_obs = self._observation_convert(env_obs)
        batch = Batch(obs=converted_obs)
        # ---------- DreamZero inference ----------
        normalized_input = self._process_batch(batch)
        with torch.no_grad():
            model_pred = self.lazy_joint_video_action_causal(normalized_input)

        normalized_action = model_pred["action_pred"].float()

        # Unnormalize actions (pass obs for relative action normalization)
        unnormalized_action = self.config.data_transforms.unapply(
            {"action": normalized_action.cpu()}
        )
        batch.act = unnormalized_action

        actions = self._extract_env_actions(batch.act)
        actions = self._map_gripper_for_env(actions)
        self._debug_action_payload(
            normalized_action=normalized_action,
            unnormalized_action=batch.act,
            env_actions=actions,
        )

        assert actions.shape[-1] == self.config.env_action_dim, (
            f"Action shape mismatch: {actions.shape} != {self.config.env_action_dim}"
        )

        env_action_tensor = torch.as_tensor(actions, dtype=torch.float32).cpu()
        flat_env_action = env_action_tensor.reshape(actions.shape[0], -1)
        real_action_dim = self._infer_real_action_dim(
            min(normalized_action.shape[-1], actions.shape[-1] + 1)
        )
        action_mask = torch.zeros_like(normalized_action, dtype=torch.bool)
        action_mask[..., :real_action_dim] = True
        action_chain_payload = self._build_action_chain_payload(
            normalized_action=normalized_action.cpu(),
            mean_action=normalized_action.detach().cpu(),
            action_mask=action_mask.cpu(),
        )
        rl_input = dict(normalized_input)
        rl_input["action"] = normalized_action.cpu()
        rl_input["action_mask"] = action_mask.cpu()
        rl_input["has_real_action"] = torch.ones(
            normalized_action.shape[0], dtype=torch.bool
        )
        forward_inputs = {
            "action": flat_env_action,
            "model_action": normalized_action.cpu()
            .reshape(normalized_action.shape[0], -1)
            .contiguous(),
        }
        forward_inputs.update(self._flatten_rl_tensor_payload(rl_input))
        forward_inputs.update(action_chain_payload)
        forward_inputs = self._normalize_forward_payload_for_rollout(forward_inputs)
        result = {
            "prev_logprobs": action_chain_payload[
                "dreamzero_old_action_logprob"
            ].to(dtype=torch.float32),
            "prev_values": torch.zeros((flat_env_action.shape[0], 1), dtype=torch.float32),
            "forward_inputs": forward_inputs,
        }
        return actions, result

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        elif forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        else:
            raise NotImplementedError

    def sft_forward(self, data=None, **kwargs):
        if data is None:
            data = kwargs.get("data")
        if data is None:
            raise ValueError("sft_forward requires `data` from the SFT dataloader.")
        outputs = super().forward(data)
        if "loss" not in outputs:
            raise ValueError("sft_forward requires `loss` in the outputs.")
        return outputs

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        """Default forward pass."""
        if forward_inputs is None:
            raise ValueError("DreamZero default_forward requires `forward_inputs`.")
        if any(key.startswith("dreamzero_rl.") for key in forward_inputs):
            return self.rl_forward(forward_inputs=forward_inputs, **kwargs)

        world_inputs = self._prepare_world_model_inputs(forward_inputs)
        world_model = self._get_world_model(
            obs_dim=world_inputs["curr_obs"]["states"].shape[-1],
            action_dim=world_inputs["actions"].shape[-1],
        )
        outputs = world_model(**world_inputs)
        set_dreamzero_loss_payload(outputs["losses"], outputs["metrics"])

        actions = world_inputs["actions"]
        logprobs = torch.zeros_like(actions, dtype=torch.float32)
        result: dict[str, Any] = {
            "logprobs": logprobs,
            "dreamzero_losses": outputs["losses"],
            "dreamzero_metrics": outputs["metrics"],
            "dreamzero_outputs": outputs,
        }
        if kwargs.get("compute_entropy", False):
            result["entropy"] = torch.zeros_like(actions, dtype=torch.float32)
        if kwargs.get("compute_values", False):
            values = world_model.value_model(outputs["posterior_features"].detach())
            result["values"] = values
        return result

    def rl_forward(
        self,
        forward_inputs: dict[str, Any],
        **kwargs,
    ) -> dict[str, Any]:
        rl_input = self._restore_rl_tensor_payload(forward_inputs)
        if not rl_input:
            raise KeyError(
                "DreamZero RL forward requires tensor payload keys prefixed with "
                "'dreamzero_rl.'."
            )

        actions = rl_input.get("action")
        if actions is None:
            actions = forward_inputs.get("model_action")
        if actions is None:
            raise KeyError(
                "DreamZero RL forward requires normalized sampled actions under "
                "'dreamzero_rl.action' or 'model_action'."
            )
        actions = self._ensure_batch_time(actions, "actions")
        if actions.ndim > 3:
            actions = actions.reshape(actions.shape[0], -1, actions.shape[-1])
        if actions.is_floating_point():
            actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
            actions = actions.clamp(-1.0, 1.0)
        rl_input["action"] = actions
        images = rl_input.get("images")
        if torch.is_tensor(images) and images.ndim >= 5 and images.shape[1] == 1:
            diffusion_model = getattr(getattr(self, "action_head", None), "model", None)
            num_action_per_block = int(
                getattr(diffusion_model, "num_action_per_block", actions.shape[1])
            )
            num_frame_per_block = int(
                getattr(getattr(self, "action_head", None), "num_frame_per_block", 1)
            )
            blocks = max(1, actions.shape[1] // max(1, num_action_per_block))
            target_frames = blocks * max(1, num_frame_per_block) * 4 + 1
            rl_input["images"] = images.expand(
                images.shape[0], target_frames, *images.shape[2:]
            ).contiguous()

        logprob_mode = kwargs.get("dreamzero_logprob_mode", "action_chain")
        use_action_chain_logprob = (
            logprob_mode == "action_chain"
            and "dreamzero_old_action_logprob" in forward_inputs
        )

        action_logprobs = None
        extra_metrics: dict[str, torch.Tensor] = {}
        if use_action_chain_logprob:
            self._reset_action_chain_inference_state()
            mean_pred = self.lazy_joint_video_action_causal(
                rl_input, return_video=False
            )["action_pred"]
            action_logprob_std = forward_inputs.get("dreamzero_action_logprob_std", None)
            if torch.is_tensor(action_logprob_std):
                std = float(action_logprob_std.float().reshape(-1)[0].item())
            else:
                std = self._dreamzero_action_logprob_std()
            action_logprobs = self._gaussian_action_logprob(
                actions,
                mean_pred.to(device=actions.device),
                std=std,
                mask=rl_input.get("action_mask"),
            )
            action_loss = action_logprobs.detach().new_zeros(action_logprobs.shape)
        else:
            action_losses = []
            for index in range(actions.shape[0]):
                sample_input = self._slice_rl_sample(rl_input, index)
                outputs = VLA.forward(self, sample_input)
                if "action_loss" not in outputs:
                    raise KeyError(
                        "DreamZero action-head forward did not return 'action_loss'; "
                        "cannot compute a real RL update."
                    )
                action_losses.append(outputs["action_loss"].reshape(()))
                if index == 0:
                    if "dynamics_loss" in outputs:
                        extra_metrics["dreamzero/dynamics_loss"] = outputs[
                            "dynamics_loss"
                        ].detach()
                    if "loss" in outputs:
                        extra_metrics["dreamzero/base_loss"] = outputs["loss"].detach()

            action_loss = torch.stack(action_losses)

        losses = {"action_loss": action_loss}
        if action_logprobs is not None:
            losses["action_logprobs"] = action_logprobs
            old_action_logprobs = forward_inputs["dreamzero_old_action_logprob"].to(
                device=actions.device, dtype=torch.float32
            )
            losses["old_action_logprobs"] = old_action_logprobs.reshape_as(
                action_logprobs
            )
        metrics = {
            "dreamzero/raw_action_loss": action_loss.detach().mean(),
            "dreamzero/rl_batch_size": torch.tensor(
                actions.shape[0], device=actions.device, dtype=torch.float32
            ),
        }
        if action_logprobs is not None:
            metrics["dreamzero/action_chain_logprob_forward"] = (
                action_logprobs.detach().mean()
            )
        metrics.update(extra_metrics)
        set_dreamzero_loss_payload(losses, metrics)

        if action_logprobs is None:
            flat_logprobs = torch.zeros(
                actions.shape[0],
                actions.shape[1] * actions.shape[2],
                dtype=torch.float32,
                device=actions.device,
            )
        else:
            flat_logprobs = action_logprobs.float()
        result: dict[str, Any] = {
            "logprobs": flat_logprobs,
            "dreamzero_losses": losses,
            "dreamzero_metrics": metrics,
        }
        if kwargs.get("compute_entropy", False):
            result["entropy"] = torch.zeros_like(flat_logprobs)
        return result

    @staticmethod
    def _slice_rl_sample(inputs: dict[str, Any], index: int) -> dict[str, Any]:
        sample: dict[str, Any] = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                sample[key] = value[index : index + 1].contiguous()
            else:
                sample[key] = value
        return sample

    def _get_world_model(self, obs_dim: int, action_dim: int) -> DreamZeroWorldModel:
        if self.world_model is not None:
            return self.world_model

        cfg = self.config
        self.world_model = DreamZeroWorldModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            stochastic_dim=getattr(cfg, "rssm_stochastic_dim", 32),
            deterministic_dim=getattr(cfg, "rssm_deterministic_dim", 128),
            hidden_dim=getattr(cfg, "world_model_hidden_dim", 256),
            imagination_horizon=getattr(cfg, "imagination_horizon", 15),
            gamma=getattr(cfg, "gamma", 0.99),
            lambda_=getattr(cfg, "lambda_", 0.95),
            kl_scale=getattr(cfg, "kl_scale", 1.0),
            free_nats=getattr(cfg, "free_nats", 1.0),
        )
        self.world_model.to(next(self.parameters()).device)
        self.world_model.to(dtype=next(self.parameters()).dtype)
        return self.world_model

    def _prepare_world_model_inputs(
        self, forward_inputs: dict[str, Any]
    ) -> dict[str, Any]:
        curr_obs = forward_inputs.get("curr_obs")
        if curr_obs is None:
            states = forward_inputs.get(
                "states", forward_inputs.get("state", forward_inputs.get("curr_states"))
            )
            if states is None:
                raise KeyError(
                    "DreamZero default_forward requires forward_inputs['curr_obs'] "
                    "or a state tensor under 'states'/'state'/'curr_states'."
                )
            curr_obs = {"states": states}

        next_obs = forward_inputs.get("next_obs")
        if next_obs is None and "next_states" in forward_inputs:
            next_obs = {"states": forward_inputs["next_states"]}

        actions = forward_inputs.get("actions", forward_inputs.get("action"))
        if actions is None:
            raise KeyError("DreamZero default_forward requires actions.")
        actions = self._restore_world_model_actions(actions)

        rewards = forward_inputs.get("rewards")
        if rewards is None:
            rewards = torch.zeros(
                *actions.shape[:2],
                1,
                dtype=actions.dtype,
                device=actions.device,
            )

        dones = forward_inputs.get("dones")
        if dones is None:
            dones = torch.zeros(
                *actions.shape[:2],
                1,
                dtype=torch.bool,
                device=actions.device,
            )

        curr_obs = {"states": self._ensure_batch_time(curr_obs["states"], "curr_obs")}
        if next_obs is not None:
            next_obs = {
                "states": self._ensure_batch_time(next_obs["states"], "next_obs")
            }
        actions = self._ensure_batch_time(actions, "actions")
        curr_obs["states"] = self._flatten_world_model_time_axes(
            curr_obs["states"], "curr_obs"
        )
        if next_obs is not None:
            next_obs["states"] = self._flatten_world_model_time_axes(
                next_obs["states"], "next_obs"
            )
        actions = self._flatten_world_model_time_axes(actions, "actions")
        curr_obs["states"] = self._align_world_model_states_to_actions(
            curr_obs["states"], actions, "curr_obs"
        )
        if next_obs is not None:
            next_obs["states"] = self._align_world_model_states_to_actions(
                next_obs["states"], actions, "next_obs"
            )
        if curr_obs["states"].shape[1] == 1 and actions.shape[1] > 1:
            curr_obs["states"] = curr_obs["states"].expand(
                curr_obs["states"].shape[0],
                actions.shape[1],
                *curr_obs["states"].shape[2:],
            )
        if (
            next_obs is not None
            and next_obs["states"].shape[1] == 1
            and actions.shape[1] > 1
        ):
            next_obs["states"] = next_obs["states"].expand(
                next_obs["states"].shape[0],
                actions.shape[1],
                *next_obs["states"].shape[2:],
            )
        rewards = self._ensure_batch_time(rewards, "rewards")
        dones = self._ensure_batch_time(dones, "dones")
        rewards = self._flatten_world_model_time_axes(rewards, "rewards")
        dones = self._flatten_world_model_time_axes(dones, "dones")

        return {
            "curr_obs": curr_obs,
            "next_obs": next_obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
        }

    @staticmethod
    def _ensure_batch_time(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.ndim < 2:
            raise ValueError(f"{name} must have at least batch/time dims.")
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(1)
        return tensor

    def _restore_world_model_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 2:
            return actions
        env_action_dim = getattr(self.config, "env_action_dim", None)
        if not env_action_dim:
            return actions
        if actions.shape[-1] == env_action_dim:
            return actions
        if actions.shape[-1] % env_action_dim != 0:
            return actions
        return actions.reshape(actions.shape[0], -1, env_action_dim)

    @staticmethod
    def _flatten_world_model_time_axes(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.ndim < 3:
            return tensor
        if tensor.ndim == 3:
            return tensor
        return tensor.reshape(tensor.shape[0], -1, tensor.shape[-1])

    @staticmethod
    def _align_world_model_states_to_actions(
        states: torch.Tensor,
        actions: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if states.ndim != 3 or actions.ndim != 3:
            return states
        action_time = actions.shape[1]
        if states.shape[1] == action_time:
            return states
        if states.shape[1] == action_time * actions.shape[-1]:
            return states.reshape(
                states.shape[0],
                action_time,
                actions.shape[-1],
                states.shape[-1],
            )[:, :, 0, :].contiguous()
        if states.shape[-1] == action_time:
            return states.transpose(1, 2).contiguous()
        return states
