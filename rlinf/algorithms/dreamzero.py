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

"""DreamZero algorithms aligned with RLinf registries."""

from typing import Literal, Optional

import torch

from rlinf.algorithms.losses import compute_ppo_actor_loss
from rlinf.algorithms.registry import register_advantage, register_policy_loss
from rlinf.algorithms.utils import safe_normalize

_DREAMZERO_LOSS_PAYLOAD: tuple[dict[str, torch.Tensor], dict] | None = None


def set_dreamzero_loss_payload(
    losses: dict[str, torch.Tensor], metrics: Optional[dict] = None
) -> None:
    """Store the latest DreamZero loss payload for the next loss call.

    FSDP actor workers pass a fixed PPO-style kwargs set into the loss registry.
    This narrow handoff lets DreamZeroPolicy.default_forward provide either
    world-model proxy losses or a differentiable action-head loss without
    changing runner/channel control flow.
    """
    global _DREAMZERO_LOSS_PAYLOAD
    _DREAMZERO_LOSS_PAYLOAD = (losses, metrics or {})


def pop_dreamzero_loss_payload() -> tuple[dict[str, torch.Tensor], dict] | None:
    """Consume and clear the latest DreamZero loss payload."""
    global _DREAMZERO_LOSS_PAYLOAD
    payload = _DREAMZERO_LOSS_PAYLOAD
    _DREAMZERO_LOSS_PAYLOAD = None
    return payload


@register_advantage("dreamzero")
def compute_dreamzero_advantages_and_returns(
    rewards: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    values: Optional[torch.Tensor] = None,
    normalize_advantages: bool = False,
    normalize_returns: bool = False,
    loss_mask: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute DreamZero advantages through the standard advantage interface.

    This placeholder intentionally reuses GAE-style return propagation so the
    algorithm can run end-to-end while DreamZero-specific world-model math is
    developed behind the same registry contract.
    """
    batch_size = kwargs.get("batch_size", rewards.shape[-1])
    n_steps = kwargs.get("n_steps", rewards.shape[0])
    if (
        batch_size is not None
        and n_steps is not None
        and (rewards.ndim != 2 or rewards.shape != (n_steps, batch_size))
    ):
        score_rewards = rewards.reshape(-1)
        if score_rewards.numel() != batch_size:
            raise ValueError(
                "DreamZero placeholder advantages expected one score per "
                f"environment after preprocessing, got {score_rewards.numel()} "
                f"scores for batch_size={batch_size}."
            )
        step_rewards = torch.zeros(
            (n_steps, batch_size), dtype=rewards.dtype, device=rewards.device
        )
        step_rewards[-1] = score_rewards
        rewards = step_rewards

    rewards = rewards.reshape(n_steps, batch_size)
    if dones is None:
        dones = torch.zeros(
            (n_steps + 1, batch_size), dtype=torch.bool, device=rewards.device
        )
        dones[-1] = True
    else:
        dones = dones.reshape(dones.shape[0], -1).to(torch.bool)
        if dones.shape[1] != batch_size:
            dones = dones[:, :batch_size]

    returns = torch.zeros_like(rewards)
    running_return = torch.zeros(batch_size, dtype=rewards.dtype, device=rewards.device)
    for step in reversed(range(n_steps)):
        running_return = rewards[step] + gamma * running_return * (~dones[step + 1])
        returns[step] = running_return

    if values is not None:
        values = values.reshape(values.shape[0], -1)[:n_steps, :batch_size]
        advantages = returns - values
    else:
        advantages = returns.clone()

    if normalize_advantages and loss_mask is not None:
        advantages = safe_normalize(
            advantages, loss_mask=loss_mask.reshape_as(advantages)
        )
    if normalize_returns and loss_mask is not None:
        returns = safe_normalize(returns, loss_mask=loss_mask.reshape_as(returns))

    return advantages, returns


def _resolve_dreamzero_payload(
    kwargs: dict,
) -> tuple[dict[str, torch.Tensor] | None, dict]:
    dreamzero_losses = kwargs.get("dreamzero_losses", None)
    dreamzero_metrics = kwargs.get("dreamzero_metrics", {})
    if dreamzero_losses is None:
        payload = pop_dreamzero_loss_payload()
        if payload is not None:
            dreamzero_losses, dreamzero_metrics = payload
    return dreamzero_losses, dreamzero_metrics


def _compute_ppo_fallback_loss(**kwargs) -> tuple[torch.Tensor, dict]:
    policy_loss, metrics = compute_ppo_actor_loss(**kwargs)
    metrics.update({"dreamzero/ppo_fallback_loss": policy_loss.detach()})
    return policy_loss, metrics


def _align_vector_like(
    value: torch.Tensor,
    target: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    value = value.to(device=target.device, dtype=dtype or target.dtype)
    if value.shape == target.shape:
        return value
    if value.numel() == target.numel():
        return value.reshape_as(target)
    if value.ndim > 0 and value.shape[0] == target.shape[0]:
        return value.reshape(value.shape[0], -1).mean(dim=-1).reshape_as(target)
    if value.numel() == 1:
        return value.expand_as(target)
    raise ValueError(
        f"DreamZero PPO proxy could not align {name} shape "
        f"{tuple(value.shape)} to action-loss shape {tuple(target.shape)}."
    )


def _as_sample_logprobs(
    value: torch.Tensor,
    target: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    value = value.to(device=target.device, dtype=torch.float32)
    if value.ndim == 0:
        value = value.reshape(1)
    if value.shape == target.shape:
        return value
    if value.numel() == target.numel():
        return value.reshape_as(target)
    if value.ndim > 0 and value.shape[0] == target.shape[0]:
        return value.reshape(value.shape[0], -1).sum(dim=-1).reshape_as(target)
    if value.numel() == 1:
        return value.expand_as(target)
    raise ValueError(
        f"DreamZero action-chain logprob could not align {name} shape "
        f"{tuple(value.shape)} to sample shape {tuple(target.shape)}."
    )


def _reduce_sample_values_like_action_loss(
    value: torch.Tensor,
    target: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype | None = None,
    mode: Literal["mean", "last_valid"] = "mean",
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    value = value.to(device=target.device, dtype=dtype or target.dtype)
    if value.shape == target.shape:
        return value
    if value.numel() == target.numel():
        return value.reshape_as(target)
    if value.ndim > 0 and value.shape[0] == target.shape[0]:
        flat = value.reshape(value.shape[0], -1)
        if loss_mask is not None:
            mask = loss_mask.to(device=target.device, dtype=torch.bool)
            mask = mask.reshape(mask.shape[0], -1)
            if mask.shape != flat.shape:
                if mask.shape[0] == flat.shape[0] and mask.numel() != flat.numel():
                    mask = mask.expand_as(flat)
                elif mask.numel() == flat.numel():
                    mask = mask.reshape_as(flat)
                else:
                    raise ValueError(
                        f"DreamZero RL loss could not align loss_mask shape "
                        f"{tuple(loss_mask.shape)} to {name} shape {tuple(value.shape)}."
                    )
        else:
            mask = torch.ones_like(flat, dtype=torch.bool)

        if mode == "last_valid":
            valid = mask.any(dim=-1)
            reversed_indices = mask.flip(dims=[-1]).float().argmax(dim=-1)
            last_indices = flat.shape[-1] - 1 - reversed_indices
            reduced = flat.gather(1, last_indices.to(torch.long).unsqueeze(-1)).squeeze(-1)
            reduced = torch.where(valid, reduced, torch.zeros_like(reduced))
        elif mode == "mean":
            denom = mask.sum(dim=-1).clamp_min(1).to(flat.dtype)
            reduced = (flat * mask.to(flat.dtype)).sum(dim=-1) / denom
        else:
            raise ValueError(f"Unsupported DreamZero reduction mode: {mode!r}")
        return reduced.reshape_as(target)
    if value.numel() == 1:
        return value.expand_as(target)
    raise ValueError(
        f"DreamZero RL loss could not align {name} shape "
        f"{tuple(value.shape)} to action-loss shape {tuple(target.shape)}."
    )


def compute_dreamzero_world_model_proxy_loss(**kwargs) -> tuple[torch.Tensor, dict]:
    """Compute the DreamZero world-model proxy loss used by 18e6ef2.

    This path consumes the lightweight DreamZeroWorldModel payload containing
    ``model_loss``, ``actor_loss``, and ``value_loss``. It is intentionally kept
    separate from the real action-head RL loss so proxy training is not confused
    with native DreamZero action model finetuning.
    """
    dreamzero_losses, dreamzero_metrics = _resolve_dreamzero_payload(kwargs)

    if dreamzero_losses is None:
        return _compute_ppo_fallback_loss(**kwargs)

    required = ("model_loss", "actor_loss", "value_loss")
    missing = [name for name in required if name not in dreamzero_losses]
    if missing:
        raise KeyError(f"DreamZero loss payload missing required keys: {missing}.")

    model_scale = kwargs.get("dreamzero_model_loss_scale", 1.0)
    actor_scale = kwargs.get("dreamzero_actor_loss_scale", 1.0)
    value_scale = kwargs.get("dreamzero_value_loss_scale", 1.0)

    model_loss = dreamzero_losses["model_loss"]
    actor_loss = dreamzero_losses["actor_loss"]
    value_loss = dreamzero_losses["value_loss"]
    total_loss = (
        model_scale * model_loss + actor_scale * actor_loss + value_scale * value_loss
    )

    metrics = {
        "dreamzero/model_loss": model_loss.detach(),
        "dreamzero/actor_loss": actor_loss.detach(),
        "dreamzero/value_loss": value_loss.detach(),
        "dreamzero/total_loss": total_loss.detach(),
    }
    metrics.update(dreamzero_metrics)
    return total_loss, metrics


@register_policy_loss("dreamzero_action_head_rl")
def compute_dreamzero_action_head_rl_loss(**kwargs) -> tuple[torch.Tensor, dict]:
    """Compute PPO over a DreamZero action-head log-probability proxy.

    DreamZeroPolicy.default_forward returns an ``action_loss`` tensor produced by
    the real DreamZero action model on rollout observations/actions. DreamZero
    can either expose an action-chain log-probability for PPO or fall back to the
    historical ``-action_loss`` differentiable proxy when explicitly requested.
    """
    dreamzero_losses, dreamzero_metrics = _resolve_dreamzero_payload(kwargs)

    if dreamzero_losses is None:
        return _compute_ppo_fallback_loss(**kwargs)

    if "action_loss" not in dreamzero_losses:
        raise KeyError(
            "DreamZero RL loss payload missing required key 'action_loss'. "
            "World-model-only losses are not a real DreamZero policy RL update."
        )

    action_loss = dreamzero_losses["action_loss"]
    if action_loss.ndim == 0:
        action_loss = action_loss.reshape(1)

    action_loss = action_loss.float()

    advantages = kwargs.get("advantages", None)
    if advantages is None:
        raise KeyError("DreamZero RL loss requires environment advantages.")

    loss_mask = kwargs.get("loss_mask", None)
    reduction_mode = kwargs.get(
        "dreamzero_advantage_reduction",
        "last_valid" if kwargs.get("reward_type") == "chunk_level" else "mean",
    )
    sample_advantages = _reduce_sample_values_like_action_loss(
        advantages,
        action_loss,
        name="advantages",
        dtype=torch.float32,
        mode=reduction_mode,
        loss_mask=loss_mask,
    )

    ppo_loss_mask = None
    if loss_mask is not None and kwargs.get("dreamzero_use_loss_mask", False):
        ppo_loss_mask = _reduce_sample_values_like_action_loss(
            loss_mask,
            action_loss,
            name="loss_mask",
            dtype=torch.float32,
            mode="mean",
            loss_mask=loss_mask,
        ).bool()

    logprob_mode = kwargs.get("dreamzero_logprob_mode", "action_chain")
    if logprob_mode == "action_chain":
        if "action_logprobs" not in dreamzero_losses:
            raise KeyError(
                "DreamZero action-chain mode requires dreamzero_losses['action_logprobs']."
            )
        logprobs = _as_sample_logprobs(
            dreamzero_losses["action_logprobs"],
            action_loss,
            name="action_logprobs",
        )
        old_logprobs_input = kwargs.get("old_logprobs", None)
        if old_logprobs_input is None:
            old_logprobs_input = dreamzero_losses.get("old_action_logprobs", None)
        if old_logprobs_input is None:
            raise KeyError("DreamZero action-chain mode requires old_logprobs.")
        old_logprobs = _as_sample_logprobs(
            old_logprobs_input,
            logprobs,
            name="old_logprobs",
        )
        mode_metric = 1.0
        extra_logprob_metrics = {
            "dreamzero/action_chain_logprob": logprobs.detach().mean(),
            "dreamzero/old_action_chain_logprob": old_logprobs.detach().mean(),
        }
    elif logprob_mode in ("action_loss_proxy", "proxy"):
        proxy_scale = kwargs.get(
            "dreamzero_action_logprob_proxy_scale",
            kwargs.get("dreamzero_action_loss_scale", 1.0),
        )
        logprobs = -proxy_scale * action_loss
        old_logprobs = kwargs.get("old_logprobs", None)
        use_rollout_old_proxy = kwargs.get(
            "dreamzero_use_rollout_old_logprob_proxy", False
        )
        if (
            use_rollout_old_proxy
            and old_logprobs is not None
            and old_logprobs.numel() == logprobs.numel()
        ):
            old_logprobs = _align_vector_like(
                old_logprobs,
                logprobs,
                name="old_logprobs",
                dtype=torch.float32,
            )
        else:
            old_logprobs = logprobs.detach()
        mode_metric = 0.0
        extra_logprob_metrics = {
            "dreamzero/action_logprob_proxy": logprobs.detach().mean(),
            "dreamzero/old_action_logprob_proxy": old_logprobs.detach().mean(),
            "dreamzero/action_logprob_proxy_scale": torch.tensor(
                float(proxy_scale), device=action_loss.device
            ),
        }
    else:
        raise ValueError(
            "dreamzero_logprob_mode must be 'action_chain' or "
            f"'action_loss_proxy', got {logprob_mode!r}."
        )

    ppo_kwargs = dict(kwargs)
    if "dreamzero_ppo_clip_ratio_c" in kwargs:
        ppo_kwargs["clip_ratio_c"] = kwargs["dreamzero_ppo_clip_ratio_c"]
    else:
        ppo_kwargs["clip_ratio_c"] = None
    ppo_kwargs.update(
        {
            "logprobs": logprobs,
            "old_logprobs": old_logprobs,
            "advantages": sample_advantages,
            "loss_mask": ppo_loss_mask,
            "loss_mask_sum": None,
            "max_episode_steps": None,
        }
    )
    ppo_loss, metrics = compute_ppo_actor_loss(**ppo_kwargs)

    total_scale = kwargs.get("dreamzero_ppo_loss_scale", 1.0)
    total_loss = total_scale * ppo_loss
    metrics.update(
        {
            "dreamzero/action_loss": total_loss.detach(),
            "dreamzero/ppo_loss": ppo_loss.detach(),
            "dreamzero/logprob_mode": torch.tensor(
                mode_metric, device=action_loss.device
            ),
            "dreamzero/ppo_loss_scale": torch.tensor(
                float(total_scale), device=action_loss.device
            ),
            "dreamzero/action_loss_unweighted": action_loss.detach().mean(),
            "dreamzero/sample_advantage_mean": sample_advantages.detach().mean(),
            "dreamzero/loss_mask_enabled": torch.tensor(
                float(ppo_loss_mask is not None), device=action_loss.device
            ),
            "dreamzero/total_loss": total_loss.detach(),
        }
    )
    metrics.update(extra_logprob_metrics)
    metrics.update(dreamzero_metrics)
    return total_loss, metrics


register_policy_loss("dreamzero_world_model_proxy")(
    compute_dreamzero_world_model_proxy_loss
)

# Historical compatibility: commits 120d374 -> 18e6ef2 used `loss_type:
# dreamzero` for the world-model proxy path. Keep that meaning stable and use
# `dreamzero_action_head_rl` for native DreamZero action-head RL.
register_policy_loss("dreamzero")(compute_dreamzero_world_model_proxy_loss)
