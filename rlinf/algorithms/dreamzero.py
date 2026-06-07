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

from typing import Optional

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
    """Compute an advantage-weighted DreamZero action-head RL loss.

    DreamZeroPolicy.default_forward returns an ``action_loss`` tensor produced by
    the real DreamZero action model on rollout observations/actions. The
    environment advantage scales that loss, so positive-return samples reinforce
    their sampled action chunks while zero/negative samples do not masquerade as
    a successful policy update.
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

    advantages = kwargs.get("advantages", None)
    if advantages is None:
        raise KeyError("DreamZero RL loss requires environment advantages.")
    advantages = advantages.to(device=action_loss.device, dtype=action_loss.dtype)
    while advantages.ndim > action_loss.ndim:
        advantages = advantages.mean(dim=-1)
    while advantages.ndim < action_loss.ndim:
        advantages = advantages.unsqueeze(-1)
    if advantages.shape != action_loss.shape:
        advantages = advantages.expand_as(action_loss)

    weight_mode = kwargs.get("dreamzero_advantage_weight_mode", "positive")
    if weight_mode == "positive":
        weights = advantages.clamp_min(0)
    elif weight_mode == "signed":
        weights = advantages
    else:
        raise ValueError(
            "dreamzero_advantage_weight_mode must be 'positive' or 'signed', "
            f"got {weight_mode!r}."
        )

    loss_mask = kwargs.get("loss_mask", None)
    if loss_mask is not None:
        loss_mask = loss_mask.to(device=action_loss.device, dtype=action_loss.dtype)
        while loss_mask.ndim > action_loss.ndim:
            loss_mask = loss_mask.mean(dim=-1)
        while loss_mask.ndim < action_loss.ndim:
            loss_mask = loss_mask.unsqueeze(-1)
        if loss_mask.shape != action_loss.shape:
            loss_mask = loss_mask.expand_as(action_loss)
        weights = weights * loss_mask

    weight_sum = weights.detach().abs().sum().clamp_min(1.0)
    scale = kwargs.get("dreamzero_action_loss_scale", 1.0)
    total_loss = scale * (action_loss * weights.detach()).sum() / weight_sum

    metrics = {
        "dreamzero/action_loss": total_loss.detach(),
        "dreamzero/action_loss_unweighted": action_loss.detach().mean(),
        "dreamzero/advantage_weight_mean": weights.detach().mean(),
        "dreamzero/total_loss": total_loss.detach(),
    }
    metrics.update(dreamzero_metrics)
    return total_loss, metrics


register_policy_loss("dreamzero_world_model_proxy")(
    compute_dreamzero_world_model_proxy_loss
)

# Historical compatibility: commits 120d374 -> 18e6ef2 used `loss_type:
# dreamzero` for the world-model proxy path. Keep that meaning stable and use
# `dreamzero_action_head_rl` for native DreamZero action-head RL.
register_policy_loss("dreamzero")(compute_dreamzero_world_model_proxy_loss)
