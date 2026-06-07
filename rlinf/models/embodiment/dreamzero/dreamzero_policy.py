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

import logging
from typing import Any, Optional

import numpy as np
import torch
from groot.vla.model.dreamzero.base_vla import VLA
from tianshou.data import Batch

from rlinf.algorithms.dreamzero import set_dreamzero_loss_payload
from rlinf.data.datasets.dreamzero.data_transforms import (
    collect_dreamzero_dataset_keys,
    convert_rollout_env_obs,
    rollout_obs_layout_for_embodiment,
)
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.dreamzero.dreamzero_config import DreamZeroConfig
from rlinf.models.embodiment.dreamzero.world_model import DreamZeroWorldModel


class DreamZeroPolicy(VLA, BasePolicy):
    """Lightweight DreamZero action model: IdentityBackbone + WANPolicyHead."""

    # CausalWanModel has to be wrapped to avoid a FSDP2 bug
    # when using with gradient checkpointing
    _no_split_modules = [
        "T5SelfAttention",  # text encoder
        "AttentionBlock",  # vae
        "CausalWanModel",  # action head
        "CausalWanAttentionBlock",  # action head layer
    ]

    def __init__(
        self,
        config: DreamZeroConfig,
    ):
        super().__init__(config)
        self.config = config
        embodiment_tag = config.embodiment_tag
        if embodiment_tag is None:
            raise ValueError(
                "DreamZeroPolicy requires config.embodiment_tag (set in get_model)."
            )
        self._rollout_obs_layout = rollout_obs_layout_for_embodiment(embodiment_tag)
        _, _, action_keys, _ = collect_dreamzero_dataset_keys(
            config.data_transforms, embodiment_tag
        )
        self._action_keys = tuple(action_keys)
        self.world_model: DreamZeroWorldModel | None = None

    @staticmethod
    def _tree_to_device(value: Any, device: torch.device) -> Any:
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, list):
            return [DreamZeroPolicy._tree_to_device(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(DreamZeroPolicy._tree_to_device(item, device) for item in value)
        if isinstance(value, dict):
            return {
                key: DreamZeroPolicy._tree_to_device(item, device)
                for key, item in value.items()
            }
        return value

    def _patch_action_head_cache_device(self) -> None:
        action_head = getattr(self, "action_head", None)
        if action_head is None or getattr(
            action_head, "_rlinf_cache_device_patch_applied", False
        ):
            return
        run_diffusion_steps = getattr(action_head, "_run_diffusion_steps", None)
        if run_diffusion_steps is None:
            return

        def _run_diffusion_steps_device_guard(*args, **kwargs):
            noisy_input = kwargs.get("noisy_input")
            if noisy_input is None and args:
                noisy_input = args[0]
            if torch.is_tensor(noisy_input):
                device = noisy_input.device
                for cache_key in ("kv_caches", "crossattn_caches"):
                    if cache_key in kwargs:
                        kwargs[cache_key] = self._tree_to_device(
                            kwargs[cache_key], device
                        )
            return run_diffusion_steps(*args, **kwargs)

        action_head._run_diffusion_steps = _run_diffusion_steps_device_guard
        action_head._rlinf_cache_device_patch_applied = True

    # This method is called in FSDPModelManager.setup_model_and_optimizer
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={}):
        try:
            diffusion_model = getattr(getattr(self, "action_head", None), "model", None)
            enabled = True
            use_reentrant = gradient_checkpointing_kwargs.get("use_reentrant", True)

            if diffusion_model is None:
                raise ValueError("DreamZero policy must have action_head.")

            if hasattr(diffusion_model, "_set_gradient_checkpointing"):
                diffusion_model._set_gradient_checkpointing(diffusion_model, enabled)
            elif hasattr(diffusion_model, "gradient_checkpointing"):
                diffusion_model.gradient_checkpointing = enabled

            setattr(
                diffusion_model, "gradient_checkpointing_use_reentrant", use_reentrant
            )

            logging.warning(
                "DreamZero gradient checkpointing is enabled. If you encounter errors "
                "or memory leaks, consider: (1) upgrading to PyTorch 2.10 or later; "
                "(2) using use_reentrant=True to avoid issues when CUDA graphs and "
                "gradient checkpointing are used together."
            )

        except Exception:
            pass

    def apply(self, batch: Batch, **kwargs) -> Batch:
        """Run the forward modality pipeline on rollout observations.

        Input ``batch.obs`` is already in DreamZero modality keys (e.g.
        ``video.image``, ``state.state``, language key) from
        ``_observation_convert``. This method delegates to
        ``config.data_transforms``, built in ``get_model`` from Hydra cfg and
        ``metadata.json`` (via ``load_dreamzero_dataset_metadata`` +
        ``data_transforms.set_metadata``).

        Pipeline (libero_sim example, see ``libero_sim._build_composed_transform``):

        1. Video / state / action preprocessing and normalization
           (``StateActionTransform`` uses q99 stats from metadata).
        2. ``ConcatTransform.apply``: concat per-key tensors into flat
           ``state`` / ``action`` vectors. Per-key widths come from metadata
           (e.g. ``action.actions`` shape ``[7]`` for Libero).
        3. ``DreamTransform.apply``: pad state/action to ``max_state_dim`` /
           ``max_action_dim`` (typically 32 from yaml) so the WAN action head
           always sees a fixed width. Extra padded dims are zeros and masked
           during training; at inference the model still outputs width 32.

        The returned ``batch.normalized_obs`` is the dict consumed by
        ``lazy_joint_video_action_causal`` (tokens, video, padded actions, etc.).
        """
        obs = batch.obs
        normalized_input = self.config.data_transforms(obs)
        batch.normalized_obs = normalized_input
        return batch

    def unapply(self, batch: Batch, obs: Optional[dict] = None, **kwargs):
        """Invert model actions back to environment-scale per-modality tensors.

        ``batch.normalized_action`` is ``action_pred`` from the WAN head, shape
        ``[..., max_action_dim]`` (e.g. 32), matching the padded width from
        ``DreamTransform.apply``. Environment DOF is smaller (e.g. Libero 7);
        that width is **not** taken from Hydra ``action_dim`` on the policy—it
        comes from ``metadata.json`` loaded at build time:

        - ``get_model`` calls ``data_transforms.set_metadata(metadata)``.
        - ``ConcatTransform.set_metadata`` sets ``action_dims["action.actions"]``
          from ``metadata.modalities.action.<key>.shape[0]`` (7 for libero_sim).
        - On ``unapply``, transforms run in reverse order:
          ``DreamTransform.unapply`` (passthrough) →
          ``ConcatTransform.unapply`` slices ``[..., 0:env_dim]`` per
          ``action_concat_order`` → ``StateActionTransform.unapply`` reverses
          q99 normalization.

        Output is a dict like ``{"action.actions": tensor}`` with **env** width
        (7 for Libero). ``predict_action_batch`` then merges keys via
        ``_actions_from_unapply`` for the sim.

        If ``relative_action`` / ``relative_action_per_horizon`` is enabled,
        optionally adds the last ``state.*`` from ``obs`` (converted rollout
        obs passed from ``predict_action_batch``) to obtain absolute actions.
        """
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

    def _process_batch(self, batch: Batch) -> dict[str, Any]:
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
        """Keep DreamZero action-head helper tensors on this worker's device."""
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
        self._patch_action_head_cache_device()

    @staticmethod
    def _to_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        return tensor

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
        repeat_shape = (
            images.shape[0],
            target_frames - images.shape[1],
            *images.shape[2:],
        )
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
    def _restore_rl_tensor_payload(forward_inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            key.removeprefix("dreamzero_rl."): value
            for key, value in forward_inputs.items()
            if key.startswith("dreamzero_rl.")
        }

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

    def _observation_convert(self, env_obs: dict) -> dict:
        """Map RLinf rollout observations to DreamZero modality keys."""
        return convert_rollout_env_obs(self.config.embodiment_tag, env_obs)

    def _actions_from_unapply(self, act_dict: dict[str, Any]) -> np.ndarray:
        """Concatenate per-key unnormalized actions in dataset concat order."""
        parts: list[np.ndarray] = []
        for key in self._action_keys:
            if key not in act_dict:
                raise KeyError(
                    f"Unnormalized action missing {key!r}; "
                    f"available keys: {sorted(act_dict)}."
                )
            value = act_dict[key]
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            parts.append(np.asarray(value))
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=-1)

    def predict_action_batch(self, env_obs, mode, **kwargs) -> np.ndarray:
        """
        input:
            env_obs:
                - main_images: [B,H,W,C] uint8
                - wrist_images: [B,H,W,C] (optional, embodiment-specific)
                - extra_view_images: [B,N,H,W,C] (optional, e.g. oxe_droid)
                - states: [B,D]
                - task_descriptions: list[str] or None
        output:
            actions: np.ndarray [B, num_action_chunks, action_dim]
            result: dict  # compatible with rollout interface"""

        converted_obs = self._observation_convert(env_obs)
        batch = Batch(obs=converted_obs)
        # ---------- DreamZero inference ----------
        normalized_input = self._process_batch(batch)
        with torch.no_grad():
            model_pred = self.lazy_joint_video_action_causal(normalized_input)

        normalized_action = model_pred["action_pred"].float()

        batch = self.unapply(
            Batch(normalized_action=normalized_action),
            obs=converted_obs,
        )
        actions = self._actions_from_unapply(batch.act)

        if self._rollout_obs_layout.binarize_gripper:
            actions[..., -1] = np.where(actions[..., -1] > 0, 1.0, -1.0).astype(
                actions.dtype
            )

        flat = (
            torch.as_tensor(actions, dtype=torch.float32)
            .reshape(actions.shape[0], -1)
            .cpu()
        )
        real_action_dim = self._infer_real_action_dim(
            min(normalized_action.shape[-1], actions.shape[-1])
        )
        action_mask = torch.zeros_like(normalized_action, dtype=torch.bool)
        action_mask[..., :real_action_dim] = True
        rl_input = dict(normalized_input)
        rl_input["action"] = normalized_action.cpu()
        rl_input["action_mask"] = action_mask.cpu()
        rl_input["has_real_action"] = torch.ones(
            normalized_action.shape[0], dtype=torch.bool
        )
        forward_inputs = {
            "action": flat,
            "model_action": normalized_action.cpu()
            .reshape(normalized_action.shape[0], -1)
            .contiguous(),
        }
        forward_inputs.update(self._flatten_rl_tensor_payload(rl_input))
        forward_inputs = self._normalize_forward_payload_for_rollout(forward_inputs)
        result = {
            "prev_logprobs": torch.zeros_like(flat, dtype=torch.float32),
            "prev_values": torch.zeros((flat.shape[0], 1), dtype=torch.float32),
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
        # Mark the start of each training iteration so PyTorch knows when
        # to reclaim memory held by CUDA graphs from the previous iteration.
        torch.compiler.cudagraph_mark_step_begin()

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
            result["values"] = world_model.value_model(outputs["posterior_features"].detach())
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

        action_losses = []
        extra_metrics: dict[str, torch.Tensor] = {}
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
        metrics = {
            "dreamzero/raw_action_loss": action_loss.detach().mean(),
            "dreamzero/rl_batch_size": torch.tensor(
                actions.shape[0], device=actions.device, dtype=torch.float32
            ),
        }
        metrics.update(extra_metrics)
        set_dreamzero_loss_payload(losses, metrics)

        flat_logprobs = torch.zeros(
            actions.shape[0],
            actions.shape[1] * actions.shape[2],
            dtype=torch.float32,
            device=actions.device,
        )
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
            curr_obs["states"], actions
        )
        if next_obs is not None:
            next_obs["states"] = self._align_world_model_states_to_actions(
                next_obs["states"], actions
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
        if tensor.ndim <= 3:
            return tensor
        return tensor.reshape(tensor.shape[0], -1, tensor.shape[-1])

    @staticmethod
    def _align_world_model_states_to_actions(
        states: torch.Tensor,
        actions: torch.Tensor,
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
