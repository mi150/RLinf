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

"""Lightweight DreamZero world-model components.

These modules implement the algorithmic core used by the RL registry path:
observation encoding, RSSM posterior/prior rollout, latent imagination, and
world-model/actor/value losses. The implementation intentionally operates on
plain tensors so it can be exercised without the external DreamZero runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal


def _mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    layers: int = 2,
    activation: type[nn.Module] = nn.ELU,
) -> nn.Sequential:
    modules: list[nn.Module] = []
    last_dim = input_dim
    for _ in range(layers):
        modules.extend([nn.Linear(last_dim, hidden_dim), activation()])
        last_dim = hidden_dim
    modules.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*modules)


def _extract_state_tensor(obs: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(obs):
        return obs
    for key in ("states", "state", "obs", "observations"):
        value = obs.get(key)
        if torch.is_tensor(value):
            return value
    raise KeyError(
        "DreamZeroWorldModel expects observation tensors under one of "
        "'states', 'state', 'obs', or 'observations'."
    )


def _as_batch_time(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    if tensor.ndim < 2:
        raise ValueError(
            f"{name} must include batch and time dimensions, got {tensor.shape}."
        )
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(1)
    return tensor


@dataclass
class RSSMState:
    """Latent RSSM state."""

    deter: torch.Tensor
    stoch: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

    def detach(self) -> "RSSMState":
        """Return a state detached from the current computation graph."""
        return RSSMState(
            deter=self.deter.detach(),
            stoch=self.stoch.detach(),
            mean=self.mean.detach(),
            std=self.std.detach(),
        )


class Encoder(nn.Module):
    """Encode state observations into compact embeddings."""

    def __init__(self, obs_dim: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = _mlp(obs_dim, embed_dim, hidden_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Decoder(nn.Module):
    """Reconstruct observations from RSSM features."""

    def __init__(self, feature_dim: int, obs_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = _mlp(feature_dim, obs_dim, hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class RSSM(nn.Module):
    """Recurrent state-space model with Gaussian prior and posterior."""

    def __init__(
        self,
        action_dim: int,
        embed_dim: int,
        stochastic_dim: int,
        deterministic_dim: int,
        hidden_dim: int,
        min_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.stochastic_dim = stochastic_dim
        self.deterministic_dim = deterministic_dim
        self.min_std = min_std

        recurrent_input_dim = stochastic_dim + action_dim
        self.recurrent = nn.GRUCell(recurrent_input_dim, deterministic_dim)
        self.prior_net = _mlp(
            deterministic_dim, stochastic_dim * 2, hidden_dim, layers=1
        )
        self.posterior_net = _mlp(
            deterministic_dim + embed_dim, stochastic_dim * 2, hidden_dim, layers=1
        )

    @property
    def feature_dim(self) -> int:
        return self.deterministic_dim + self.stochastic_dim

    def initial(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> RSSMState:
        deter = torch.zeros(
            batch_size, self.deterministic_dim, device=device, dtype=dtype
        )
        stoch = torch.zeros(batch_size, self.stochastic_dim, device=device, dtype=dtype)
        mean = torch.zeros_like(stoch)
        std = torch.ones_like(stoch)
        return RSSMState(deter=deter, stoch=stoch, mean=mean, std=std)

    def _dist_stats(self, raw_stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_std = raw_stats.chunk(2, dim=-1)
        std = F.softplus(raw_std) + self.min_std
        return mean, std

    def observe(
        self,
        embeds: torch.Tensor,
        actions: torch.Tensor,
        dones: torch.Tensor | None = None,
    ) -> tuple[RSSMState, RSSMState]:
        batch_size, time_steps = embeds.shape[:2]
        state = self.initial(batch_size, embeds.device, embeds.dtype)
        prior_states: list[RSSMState] = []
        posterior_states: list[RSSMState] = []

        for step in range(time_steps):
            if dones is not None and step > 0:
                not_done = (~dones[:, step - 1].bool()).to(dtype=state.stoch.dtype)
                while not_done.ndim < state.stoch.ndim:
                    not_done = not_done.unsqueeze(-1)
                state = RSSMState(
                    deter=state.deter * not_done,
                    stoch=state.stoch * not_done,
                    mean=state.mean * not_done,
                    std=state.std,
                )

            prior = self.imagine_step(state, actions[:, step])
            post_input = torch.cat([prior.deter, embeds[:, step]], dim=-1)
            post_mean, post_std = self._dist_stats(self.posterior_net(post_input))
            post_stoch = post_mean + post_std * torch.randn_like(post_std)
            posterior = RSSMState(
                deter=prior.deter, stoch=post_stoch, mean=post_mean, std=post_std
            )
            prior_states.append(prior)
            posterior_states.append(posterior)
            state = posterior

        return self.stack_states(prior_states), self.stack_states(posterior_states)

    def imagine_step(self, prev_state: RSSMState, action: torch.Tensor) -> RSSMState:
        recurrent_input = torch.cat([prev_state.stoch, action], dim=-1)
        deter = self.recurrent(recurrent_input, prev_state.deter)
        mean, std = self._dist_stats(self.prior_net(deter))
        stoch = mean + std * torch.randn_like(std)
        return RSSMState(deter=deter, stoch=stoch, mean=mean, std=std)

    def get_features(self, state: RSSMState) -> torch.Tensor:
        return torch.cat([state.deter, state.stoch], dim=-1)

    @staticmethod
    def stack_states(states: list[RSSMState]) -> RSSMState:
        return RSSMState(
            deter=torch.stack([state.deter for state in states], dim=1),
            stoch=torch.stack([state.stoch for state in states], dim=1),
            mean=torch.stack([state.mean for state in states], dim=1),
            std=torch.stack([state.std for state in states], dim=1),
        )


class RewardPredictor(nn.Module):
    """Predict scalar rewards from latent features."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = _mlp(feature_dim, 1, hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class ValueModel(nn.Module):
    """Predict state values from latent features."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = _mlp(feature_dim, 1, hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class ActorModel(nn.Module):
    """Latent policy used for DreamZero imagination."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int,
        init_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.net = _mlp(feature_dim, action_dim, hidden_dim)
        self.log_std = nn.Parameter(
            torch.ones(action_dim) * torch.log(torch.tensor(init_std))
        )

    def forward(self, features: torch.Tensor) -> Independent:
        mean = torch.tanh(self.net(features))
        std = F.softplus(self.log_std).expand_as(mean) + 1e-4
        return Independent(Normal(mean, std), 1)

    def sample(self, features: torch.Tensor) -> torch.Tensor:
        return self(features).rsample()


class DreamZeroWorldModel(nn.Module):
    """Dreamer-style world model and imagination loss module."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        stochastic_dim: int = 32,
        deterministic_dim: int = 128,
        hidden_dim: int = 256,
        embed_dim: int | None = None,
        imagination_horizon: int = 15,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        kl_scale: float = 1.0,
        free_nats: float = 1.0,
    ) -> None:
        super().__init__()
        embed_dim = embed_dim or hidden_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.imagination_horizon = imagination_horizon
        self.gamma = gamma
        self.lambda_ = lambda_
        self.kl_scale = kl_scale
        self.free_nats = free_nats

        self.encoder = Encoder(obs_dim, embed_dim, hidden_dim)
        self.rssm = RSSM(
            action_dim=action_dim,
            embed_dim=embed_dim,
            stochastic_dim=stochastic_dim,
            deterministic_dim=deterministic_dim,
            hidden_dim=hidden_dim,
        )
        self.decoder = Decoder(self.rssm.feature_dim, obs_dim, hidden_dim)
        self.reward_predictor = RewardPredictor(self.rssm.feature_dim, hidden_dim)
        self.value_model = ValueModel(self.rssm.feature_dim, hidden_dim)
        self.actor = ActorModel(self.rssm.feature_dim, action_dim, hidden_dim)

    def forward(
        self,
        *,
        curr_obs: dict[str, torch.Tensor] | torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: dict[str, torch.Tensor] | torch.Tensor | None = None,
        dones: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        obs_tensor = _as_batch_time(_extract_state_tensor(curr_obs), name="curr_obs")
        actions = _as_batch_time(actions, name="actions")
        rewards = _as_batch_time(rewards, name="rewards")
        if rewards.ndim == 2:
            rewards = rewards.unsqueeze(-1)
        if dones is not None:
            dones = _as_batch_time(dones, name="dones").bool()
            if dones.ndim == 2:
                dones = dones.unsqueeze(-1)

        if obs_tensor.shape[:2] != actions.shape[:2]:
            raise ValueError(
                "DreamZero requires batch-time aligned observations and actions, "
                f"got obs={obs_tensor.shape[:2]} actions={actions.shape[:2]}."
            )
        if rewards.shape[:2] != actions.shape[:2]:
            raise ValueError(
                "DreamZero requires batch-time aligned rewards and actions, "
                f"got rewards={rewards.shape[:2]} actions={actions.shape[:2]}."
            )
        if actions.shape[-1] != self.action_dim:
            actions = actions.reshape(*actions.shape[:2], -1)[..., : self.action_dim]

        target_dtype = next(self.parameters()).dtype
        if obs_tensor.is_floating_point():
            obs_tensor = obs_tensor.to(dtype=target_dtype)
        if actions.is_floating_point():
            actions = actions.to(dtype=target_dtype)
        if rewards.is_floating_point():
            rewards = rewards.to(dtype=target_dtype)

        reconstruction_target = obs_tensor
        if next_obs is not None:
            next_obs_tensor = _as_batch_time(
                _extract_state_tensor(next_obs), name="next_obs"
            )
            if next_obs_tensor.shape == obs_tensor.shape:
                if next_obs_tensor.is_floating_point():
                    next_obs_tensor = next_obs_tensor.to(dtype=target_dtype)
                reconstruction_target = next_obs_tensor

        embeds = self.encoder(obs_tensor.reshape(-1, obs_tensor.shape[-1])).reshape(
            obs_tensor.shape[0], obs_tensor.shape[1], -1
        )
        prior, posterior = self.rssm.observe(embeds, actions, dones=dones)
        posterior_features = self.rssm.get_features(posterior)

        reconstructions = self.decoder(posterior_features)
        reward_predictions = self.reward_predictor(posterior_features)

        recon_loss = F.mse_loss(reconstructions, reconstruction_target)
        reward_loss = F.mse_loss(reward_predictions, rewards)
        kl_loss = self._kl_loss(posterior, prior)
        model_loss = recon_loss + reward_loss + self.kl_scale * kl_loss

        imagined = self.imagine(posterior)
        imagined_features = imagined["features"]
        imagined_rewards = self.reward_predictor(imagined_features.detach())
        imagined_values = self.value_model(imagined_features.detach())
        lambda_returns = self.lambda_return(
            imagined_rewards, imagined_values, self.gamma, self.lambda_
        )
        actor_loss = self.actor_loss_from_imagination(
            imagined["log_probs"], lambda_returns.detach()
        )
        value_loss = F.mse_loss(imagined_values, lambda_returns.detach())

        return {
            "losses": {
                "model_loss": model_loss,
                "actor_loss": actor_loss,
                "value_loss": value_loss,
            },
            "metrics": {
                "dreamzero/reconstruction_loss": recon_loss.detach(),
                "dreamzero/reward_loss": reward_loss.detach(),
                "dreamzero/kl_loss": kl_loss.detach(),
                "dreamzero/model_loss": model_loss.detach(),
                "dreamzero/actor_loss": actor_loss.detach(),
                "dreamzero/value_loss": value_loss.detach(),
            },
            "posterior_features": posterior_features,
            "reconstructions": reconstructions,
            "reward_predictions": reward_predictions,
            "imagined_features": imagined_features,
        }

    def _kl_loss(self, posterior: RSSMState, prior: RSSMState) -> torch.Tensor:
        posterior_dist = Independent(Normal(posterior.mean, posterior.std), 1)
        prior_dist = Independent(Normal(prior.mean, prior.std), 1)
        kl = torch.distributions.kl_divergence(posterior_dist, prior_dist)
        free_nats = torch.full_like(kl, self.free_nats)
        return torch.maximum(kl, free_nats).mean()

    def imagine(self, posterior: RSSMState) -> dict[str, torch.Tensor]:
        state = RSSMState(
            deter=posterior.deter[:, -1].detach(),
            stoch=posterior.stoch[:, -1].detach(),
            mean=posterior.mean[:, -1].detach(),
            std=posterior.std[:, -1].detach(),
        )
        features: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        log_probs: list[torch.Tensor] = []
        for _ in range(self.imagination_horizon):
            feature = self.rssm.get_features(state).detach()
            dist = self.actor(feature)
            action = dist.rsample()
            log_probs.append(dist.log_prob(action.detach()))
            state = self.rssm.imagine_step(state, action).detach()
            features.append(self.rssm.get_features(state))
            actions.append(action)
        return {
            "features": torch.stack(features, dim=1),
            "actions": torch.stack(actions, dim=1),
            "log_probs": torch.stack(log_probs, dim=1),
        }

    @staticmethod
    def actor_loss_from_imagination(
        log_probs: torch.Tensor, returns: torch.Tensor
    ) -> torch.Tensor:
        advantages = returns.squeeze(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        return -(log_probs * advantages.detach()).mean()

    @staticmethod
    def lambda_return(
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float,
        lambda_: float,
    ) -> torch.Tensor:
        bootstrap = values[:, -1]
        next_values = torch.cat([values[:, 1:], bootstrap[:, None]], dim=1)
        returns = []
        next_return = bootstrap
        for step in reversed(range(rewards.shape[1])):
            target = rewards[:, step] + gamma * (
                (1.0 - lambda_) * next_values[:, step] + lambda_ * next_return
            )
            returns.append(target)
            next_return = target
        returns.reverse()
        return torch.stack(returns, dim=1)
