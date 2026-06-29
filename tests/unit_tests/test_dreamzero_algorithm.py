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

"""Tests for DreamZero algorithm registry integration."""

import asyncio
import importlib
import types

import pytest
import torch
from omegaconf import OmegaConf
from transformers.feature_extraction_utils import BatchFeature

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.dreamzero import (
    pop_dreamzero_loss_payload,
    set_dreamzero_loss_payload,
)
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.utils.nested_dict_process import stack_list_of_dict_tensor
from rlinf.models.embodiment.dreamzero.world_model import DreamZeroWorldModel
from rlinf.workers.actor.fsdp_actor_worker import (
    build_dreamzero_forward_inputs,
    ensure_dreamzero_world_model_forward_inputs,
    get_dreamzero_train_rollout_size,
    get_dreamzero_loss_action_dim,
    process_nested_dict_for_train,
    should_log_actor_training_progress,
    strip_dreamzero_action_head_payload,
)


def test_dreamzero_policy_loss_uses_ppo_with_action_loss_proxy():
    action_loss = torch.tensor([0.1, 0.5], dtype=torch.float32, requires_grad=True)
    advantages = torch.tensor([2.0, -1.0], dtype=torch.float32)

    loss, metrics = policy_loss(
        task_type="embodied",
        loss_type="dreamzero_action_head_rl",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=3,
        logprobs=torch.zeros(2, 1, 3, dtype=torch.float32),
        old_logprobs=torch.zeros(2, dtype=torch.float32),
        advantages=advantages,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=None,
        dreamzero_logprob_mode="action_loss_proxy",
        dreamzero_losses={
            "action_loss": action_loss,
        },
        dreamzero_metrics={"dreamzero/raw_action_loss": action_loss.mean()},
    )

    loss.backward()

    logprobs = -action_loss
    old_logprobs = logprobs.detach()
    ratio = torch.exp(logprobs - old_logprobs)
    clipped_ratio = ratio.clamp(0.8, 1.2)
    expected = torch.max(-advantages * ratio, -advantages * clipped_ratio).mean()
    assert torch.allclose(loss, expected)
    assert action_loss.grad[0] > 0
    assert action_loss.grad[1] < 0
    assert metrics["actor/policy_loss"] == expected.item()
    assert metrics["dreamzero/action_logprob_proxy"] == logprobs.mean().item()
    assert metrics["dreamzero/total_loss"] == loss.item()
    assert "dreamzero/model_loss" not in metrics


def test_dreamzero_action_chain_mode_uses_recomputed_logprobs_not_action_loss():
    action_loss = torch.tensor([5.0, 7.0], dtype=torch.float32, requires_grad=True)
    logprobs = torch.tensor([-1.0, -0.4], dtype=torch.float32, requires_grad=True)
    old_logprobs = torch.tensor([-1.05, -0.35], dtype=torch.float32)
    advantages = torch.tensor([2.0, -1.0], dtype=torch.float32)

    loss, metrics = policy_loss(
        task_type="embodied",
        loss_type="dreamzero_action_head_rl",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=3,
        logprobs=torch.zeros(2, dtype=torch.float32),
        old_logprobs=old_logprobs,
        advantages=advantages,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=None,
        dreamzero_logprob_mode="action_chain",
        dreamzero_losses={
            "action_loss": action_loss,
            "action_logprobs": logprobs,
        },
    )

    loss.backward()

    ratio = torch.exp(logprobs - old_logprobs)
    clipped_ratio = ratio.clamp(0.8, 1.2)
    expected = torch.max(-advantages * ratio, -advantages * clipped_ratio).mean()
    assert torch.allclose(loss, expected)
    assert action_loss.grad is None
    assert torch.all(logprobs.grad != 0)
    assert metrics["dreamzero/logprob_mode"] == 1.0
    assert metrics["dreamzero/action_chain_logprob"] == logprobs.mean().item()
    assert metrics["dreamzero/old_action_chain_logprob"] == old_logprobs.mean().item()


def test_dreamzero_action_chain_mode_requires_chain_logprobs():
    with pytest.raises(KeyError, match="action_logprobs"):
        policy_loss(
            task_type="embodied",
            loss_type="dreamzero_action_head_rl",
            logprob_type="chunk_level",
            reward_type="chunk_level",
            single_action_dim=3,
            logprobs=torch.zeros(1, dtype=torch.float32),
            old_logprobs=torch.zeros(1, dtype=torch.float32),
            advantages=torch.ones(1, dtype=torch.float32),
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
            clip_ratio_c=None,
            dreamzero_logprob_mode="action_chain",
            dreamzero_losses={
                "action_loss": torch.tensor([1.0], dtype=torch.float32),
            },
        )


def test_dreamzero_policy_loss_zeroes_all_zero_advantages():
    action_loss = torch.tensor([1.0, 3.0], dtype=torch.float32, requires_grad=True)

    loss, metrics = policy_loss(
        task_type="embodied",
        loss_type="dreamzero_action_head_rl",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=3,
        logprobs=torch.zeros(2, 1, 3, dtype=torch.float32),
        old_logprobs=torch.zeros(2, dtype=torch.float32),
        advantages=torch.zeros(2, 1, dtype=torch.float32),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=None,
        dreamzero_logprob_mode="action_loss_proxy",
        dreamzero_losses={
            "action_loss": action_loss,
        },
    )

    loss.backward()

    assert torch.allclose(loss, torch.tensor(0.0))
    assert torch.equal(action_loss.grad, torch.zeros_like(action_loss))
    assert metrics["actor/policy_loss"] == 0.0


def test_dreamzero_policy_loss_uses_last_valid_timestep_advantage():
    action_loss = torch.tensor([2.0], dtype=torch.float32, requires_grad=True)
    advantages = torch.zeros(1, 4, 1, dtype=torch.float32)
    advantages[0, -1, 0] = 3.0
    loss_mask = torch.ones_like(advantages, dtype=torch.bool)

    loss, metrics = policy_loss(
        task_type="embodied",
        loss_type="dreamzero_action_head_rl",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=3,
        logprobs=torch.zeros(1, 1, 3, dtype=torch.float32),
        old_logprobs=torch.zeros(1, dtype=torch.float32),
        advantages=advantages,
        loss_mask=loss_mask,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=None,
        dreamzero_logprob_mode="action_loss_proxy",
        dreamzero_losses={
            "action_loss": action_loss,
        },
    )

    loss.backward()

    assert torch.allclose(loss, torch.tensor(-3.0))
    assert torch.allclose(action_loss.grad, torch.tensor([3.0]))
    assert metrics["dreamzero/sample_advantage_mean"] == 3.0


def test_dreamzero_policy_loss_keeps_sample_gradient_when_rollout_mask_is_zero():
    action_loss = torch.tensor([0.25], dtype=torch.float32, requires_grad=True)

    loss, metrics = policy_loss(
        task_type="embodied",
        loss_type="dreamzero_action_head_rl",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=3,
        logprobs=torch.zeros(1, 1, 3, dtype=torch.float32),
        old_logprobs=torch.zeros(1, dtype=torch.float32),
        advantages=torch.tensor([1.0], dtype=torch.float32),
        loss_mask=torch.zeros(1, 1, 1, dtype=torch.bool),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        dreamzero_logprob_mode="action_loss_proxy",
        dreamzero_losses={
            "action_loss": action_loss,
        },
    )

    loss.backward()

    assert torch.allclose(loss, torch.tensor(-1.0))
    assert action_loss.grad.item() > 0.0
    assert metrics["dreamzero/loss_mask_enabled"] == 0.0


def test_dreamzero_policy_loss_consumes_model_side_rl_payload_once():
    action_loss = torch.tensor([2.0, 4.0])
    set_dreamzero_loss_payload(
        {
            "action_loss": action_loss,
        },
        {"dreamzero/raw_action_loss": torch.tensor(3.0)},
    )
    old_logprobs = torch.zeros(2, dtype=torch.float32)
    advantages = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

    loss, metrics = policy_loss(
        task_type="embodied",
        loss_type="dreamzero_action_head_rl",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=3,
        logprobs=torch.zeros(2, 1, 3, dtype=torch.float32),
        old_logprobs=old_logprobs,
        advantages=advantages,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=None,
        dreamzero_logprob_mode="action_loss_proxy",
    )

    logprobs = -action_loss
    old_logprobs = logprobs.detach()
    ratio = torch.exp(logprobs - old_logprobs)
    clipped_ratio = ratio.clamp(0.8, 1.2)
    expected = torch.max(
        -advantages.reshape(-1) * ratio,
        -advantages.reshape(-1) * clipped_ratio,
    ).mean()
    assert torch.allclose(loss, expected)
    assert metrics["dreamzero/raw_action_loss"] == 3.0


def test_dreamzero_action_head_rl_skips_generic_logprob_preprocess():
    loss, metrics = policy_loss(
        task_type="embodied",
        loss_type="dreamzero_action_head_rl",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=7,
        logprobs=torch.zeros(1, 768, dtype=torch.float32),
        old_logprobs=torch.zeros(1, dtype=torch.float32),
        advantages=torch.tensor([2.0], dtype=torch.float32),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=None,
        dreamzero_logprob_mode="action_loss_proxy",
        dreamzero_losses={
            "action_loss": torch.tensor([3.0], dtype=torch.float32),
        },
    )

    assert torch.allclose(loss, torch.tensor(-2.0))
    assert metrics["dreamzero/action_logprob_proxy"] == -3.0


def test_dreamzero_action_head_rl_rejects_world_model_only_payload():
    with pytest.raises(KeyError, match="action_loss"):
        policy_loss(
            task_type="embodied",
            loss_type="dreamzero_action_head_rl",
            logprob_type="chunk_level",
            reward_type="chunk_level",
            single_action_dim=3,
            logprobs=torch.zeros(1, 1, 3, dtype=torch.float32),
            old_logprobs=torch.zeros(1, 1, 3, dtype=torch.float32),
            advantages=torch.ones(1, 1, dtype=torch.float32),
            dreamzero_losses={
                "model_loss": torch.tensor(1.0),
                "actor_loss": torch.tensor(1.0),
                "value_loss": torch.tensor(1.0),
            },
        )


def test_dreamzero_world_model_proxy_loss_consumes_world_model_payload():
    model_loss = torch.tensor(1.5, dtype=torch.float32)
    actor_loss = torch.tensor(2.0, dtype=torch.float32)
    value_loss = torch.tensor(0.25, dtype=torch.float32)

    loss, metrics = policy_loss(
        task_type="embodied",
        loss_type="dreamzero_world_model_proxy",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=3,
        logprobs=torch.zeros(1, 1, 3, dtype=torch.float32),
        old_logprobs=torch.zeros(1, 1, 3, dtype=torch.float32),
        advantages=torch.ones(1, 1, dtype=torch.float32),
        dreamzero_losses={
            "model_loss": model_loss,
            "actor_loss": actor_loss,
            "value_loss": value_loss,
        },
        dreamzero_metrics={"dreamzero/reward_loss": torch.tensor(0.125)},
    )

    assert torch.allclose(loss, model_loss + actor_loss + value_loss)
    assert metrics["dreamzero/model_loss"] == model_loss.item()
    assert metrics["dreamzero/actor_loss"] == actor_loss.item()
    assert metrics["dreamzero/value_loss"] == value_loss.item()
    assert metrics["dreamzero/reward_loss"] == 0.125
    assert metrics["dreamzero/total_loss"] == loss.item()


def test_dreamzero_world_model_proxy_skips_generic_logprob_preprocess():
    loss, metrics = policy_loss(
        task_type="embodied",
        loss_type="dreamzero_world_model_proxy",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=7,
        logprobs=torch.zeros(1, 512, dtype=torch.float32),
        old_logprobs=torch.zeros(1, 16, 7, dtype=torch.float32),
        advantages=torch.ones(1, 1, dtype=torch.float32),
        dreamzero_losses={
            "model_loss": torch.tensor(1.0),
            "actor_loss": torch.tensor(0.5),
            "value_loss": torch.tensor(0.25),
        },
    )

    assert torch.allclose(loss, torch.tensor(1.75))
    assert metrics["dreamzero/model_loss"] == 1.0


def test_dreamzero_world_model_proxy_uses_env_action_dim_for_loss_reshape():
    model_cfg = {
        "model_type": "dreamzero",
        "action_dim": 32,
        "env_action_dim": 7,
    }

    assert (
        get_dreamzero_loss_action_dim(
            model_cfg,
            loss_type="dreamzero_world_model_proxy",
        )
        == 7
    )
    assert (
        get_dreamzero_loss_action_dim(
            model_cfg,
            loss_type="dreamzero",
        )
        == 7
    )
    assert (
        get_dreamzero_loss_action_dim(
            model_cfg,
            loss_type="dreamzero_action_head_rl",
        )
        == 32
    )


def test_dreamzero_legacy_loss_type_aliases_world_model_proxy():
    loss, metrics = policy_loss(
        task_type="embodied",
        loss_type="dreamzero",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=3,
        logprobs=torch.zeros(1, 1, 3, dtype=torch.float32),
        old_logprobs=torch.zeros(1, 1, 3, dtype=torch.float32),
        advantages=torch.ones(1, 1, dtype=torch.float32),
        dreamzero_losses={
            "model_loss": torch.tensor(1.0),
            "actor_loss": torch.tensor(0.5),
            "value_loss": torch.tensor(0.25),
        },
    )

    assert torch.allclose(loss, torch.tensor(1.75))
    assert metrics["dreamzero/model_loss"] == 1.0


def test_dreamzero_world_model_forward_uses_batch_time_layout():
    torch.manual_seed(0)
    model = DreamZeroWorldModel(
        obs_dim=6,
        action_dim=4,
        stochastic_dim=5,
        deterministic_dim=7,
        hidden_dim=16,
        imagination_horizon=3,
    )
    batch_size = 2
    time_steps = 4
    curr_obs = {"states": torch.randn(batch_size, time_steps, 6)}
    next_obs = {"states": torch.randn(batch_size, time_steps, 6)}
    actions = torch.randn(batch_size, time_steps, 4)
    rewards = torch.randn(batch_size, time_steps, 1)
    dones = torch.zeros(batch_size, time_steps, 1, dtype=torch.bool)

    outputs = model(
        curr_obs=curr_obs,
        next_obs=next_obs,
        actions=actions,
        rewards=rewards,
        dones=dones,
    )

    assert set(outputs["losses"]) == {"model_loss", "actor_loss", "value_loss"}
    assert outputs["posterior_features"].shape == (batch_size, time_steps, 12)
    assert outputs["reconstructions"].shape == (batch_size, time_steps, 6)
    assert outputs["imagined_features"].shape == (batch_size, 3, 12)

    total_loss = sum(outputs["losses"].values())
    total_loss.backward()
    assert model.encoder.net[0].weight.grad is not None
    assert model.actor.net[0].weight.grad is not None


def test_dreamzero_world_model_forward_accepts_single_step_vectors():
    torch.manual_seed(0)
    model = DreamZeroWorldModel(
        obs_dim=112,
        action_dim=16,
        stochastic_dim=5,
        deterministic_dim=7,
        hidden_dim=16,
        imagination_horizon=3,
    )

    outputs = model(
        curr_obs={"states": torch.randn(1, 112)},
        next_obs={"states": torch.randn(1, 112)},
        actions=torch.randn(1, 16),
        rewards=torch.randn(1, 1),
        dones=torch.zeros(1, 1, dtype=torch.bool),
    )

    assert outputs["posterior_features"].shape == (1, 1, 12)
    assert outputs["reconstructions"].shape == (1, 1, 112)


def test_dreamzero_default_forward_uses_main_action_head_rl_loss():
    dreamzero_policy_module = pytest.importorskip(
        "rlinf.models.embodiment.dreamzero.dreamzero_policy"
    )
    DreamZeroPolicy = dreamzero_policy_module.DreamZeroPolicy

    class FakeDreamZeroPolicy(DreamZeroPolicy):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.called_inputs = None
            self.losses = [
                torch.tensor(2.0, requires_grad=True),
                torch.tensor(4.0, requires_grad=True),
            ]

        def forward(self, inputs):
            self.called_inputs = inputs
            return BatchFeature(
                data={
                    "loss": torch.tensor(5.0, requires_grad=True),
                    "action_loss": torch.tensor([2.0, 4.0], requires_grad=True),
                    "dynamics_loss": torch.tensor(7.0),
                }
            )

        def _slice_rl_sample(self, inputs, index):
            return {
                key: value[index : index + 1] if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }

        def _forward_sample_loss(self, inputs):
            raise AssertionError("unused")

        def __call__(self, inputs):
            return self.forward(inputs)

        def _mock_vla_forward(self, inputs):
            self.called_inputs = inputs
            return BatchFeature(
                data={
                    "loss": self.losses.pop(0),
                    "action_loss": self.called_inputs["action"].sum() + 2.0,
                    "dynamics_loss": torch.tensor(7.0),
                }
            )

        def rl_forward(self, forward_inputs, **kwargs):
            rl_input = self._restore_rl_tensor_payload(forward_inputs)
            action_losses = []
            for index in range(rl_input["action"].shape[0]):
                sample_input = self._slice_rl_sample(rl_input, index)
                outputs = self._mock_vla_forward(sample_input)
                action_losses.append(outputs["action_loss"].reshape(()))
            action_loss = torch.stack(action_losses)
            losses = {"action_loss": action_loss}
            metrics = {"dreamzero/raw_action_loss": action_loss.detach().mean()}
            from rlinf.algorithms.dreamzero import set_dreamzero_loss_payload

            set_dreamzero_loss_payload(losses, metrics)
            return {
                "logprobs": torch.zeros(action_loss.shape[0], 28),
                "dreamzero_losses": losses,
                "dreamzero_metrics": metrics,
            }

    policy = FakeDreamZeroPolicy()
    outputs = policy.default_forward(
        forward_inputs={
            "action": torch.zeros(2, 28),
            "model_action": torch.zeros(2, 4, 7),
            "dreamzero_rl.action": torch.ones(2, 4, 7),
            "dreamzero_rl.images": torch.zeros(
                2, 1, 1, 256, 256, 3, dtype=torch.uint8
            ),
        }
    )

    assert policy.called_inputs is not None
    assert "dreamzero_outputs" not in outputs
    assert torch.equal(outputs["dreamzero_losses"]["action_loss"], torch.full((2,), 30.0))
    payload = pop_dreamzero_loss_payload()
    assert payload is not None
    losses, metrics = payload
    assert torch.equal(losses["action_loss"], torch.full((2,), 30.0))
    assert metrics["dreamzero/raw_action_loss"] == torch.tensor(30.0)


def test_dreamzero_policy_syncs_action_head_device_helpers():
    dreamzero_policy_module = pytest.importorskip(
        "rlinf.models.embodiment.dreamzero.dreamzero_policy"
    )
    DreamZeroPolicy = dreamzero_policy_module.DreamZeroPolicy

    class FakeActionHead:
        _device = "meta"
        _vae_device_ready = True

    class FakeDreamZeroPolicy(DreamZeroPolicy):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.action_head = FakeActionHead()

    policy = FakeDreamZeroPolicy()
    policy._sync_action_head_device()

    assert policy.action_head._device == str(policy.weight.device)
    assert policy.action_head._vae_device_ready is False
    assert policy.action_head.trt_engine is None
    assert policy.action_head.trt_context is None


def test_dreamzero_prepare_world_model_inputs_accepts_flat_state_keys():
    dreamzero_policy_module = pytest.importorskip(
        "rlinf.models.embodiment.dreamzero.dreamzero_policy"
    )
    DreamZeroPolicy = dreamzero_policy_module.DreamZeroPolicy

    class FakeDreamZeroPolicy(DreamZeroPolicy):
        def __init__(self):
            torch.nn.Module.__init__(self)

    policy = FakeDreamZeroPolicy()
    inputs = policy._prepare_world_model_inputs(
        {
            "curr_states": torch.zeros(2, 3, 4, 8),
            "next_states": torch.ones(2, 3, 4, 8),
            "actions": torch.zeros(2, 3, 4, 7),
            "rewards": torch.zeros(2, 3, 4, 1),
            "dones": torch.zeros(2, 3, 4, 1, dtype=torch.bool),
        }
    )

    assert inputs["curr_obs"]["states"].shape == (2, 12, 8)
    assert inputs["next_obs"]["states"].shape == (2, 12, 8)
    assert inputs["actions"].shape == (2, 12, 7)


def test_dreamzero_prepare_world_model_inputs_flattens_action_chunk_time_axis():
    dreamzero_policy_module = pytest.importorskip(
        "rlinf.models.embodiment.dreamzero.dreamzero_policy"
    )
    DreamZeroPolicy = dreamzero_policy_module.DreamZeroPolicy

    class FakeDreamZeroPolicy(DreamZeroPolicy):
        def __init__(self):
            torch.nn.Module.__init__(self)

    policy = FakeDreamZeroPolicy()
    inputs = policy._prepare_world_model_inputs(
        {
            "curr_states": torch.zeros(2, 3, 4, 8),
            "next_states": torch.ones(2, 3, 4, 8),
            "actions": torch.zeros(2, 3, 4, 7),
            "rewards": torch.zeros(2, 3, 4, 1),
            "dones": torch.zeros(2, 3, 4, 1, dtype=torch.bool),
        }
    )

    assert inputs["curr_obs"]["states"].shape == (2, 12, 8)
    assert inputs["next_obs"]["states"].shape == (2, 12, 8)
    assert inputs["actions"].shape == (2, 12, 7)
    assert inputs["rewards"].shape == (2, 12, 1)
    assert inputs["dones"].shape == (2, 12, 1)


def test_dreamzero_prepare_world_model_inputs_restores_flat_env_actions():
    dreamzero_policy_module = pytest.importorskip(
        "rlinf.models.embodiment.dreamzero.dreamzero_policy"
    )
    DreamZeroPolicy = dreamzero_policy_module.DreamZeroPolicy

    class FakeConfig:
        env_action_dim = 7

    class FakeDreamZeroPolicy(DreamZeroPolicy):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.config = FakeConfig()

    policy = FakeDreamZeroPolicy()
    inputs = policy._prepare_world_model_inputs(
        {
            "curr_states": torch.zeros(2, 112, 8),
            "next_states": torch.ones(2, 112, 8),
            "action": torch.zeros(2, 112 * 7),
        }
    )

    assert inputs["actions"].shape == (2, 112, 7)
    assert inputs["curr_obs"]["states"].shape == (2, 112, 8)
    assert inputs["rewards"].shape == (2, 112, 1)


def test_dreamzero_prepare_world_model_inputs_restores_transposed_state_time_axis():
    dreamzero_policy_module = pytest.importorskip(
        "rlinf.models.embodiment.dreamzero.dreamzero_policy"
    )
    DreamZeroPolicy = dreamzero_policy_module.DreamZeroPolicy

    class FakeConfig:
        env_action_dim = 7

    class FakeDreamZeroPolicy(DreamZeroPolicy):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.config = FakeConfig()

    policy = FakeDreamZeroPolicy()
    curr_states = torch.randn(1, 8, 16)
    next_states = torch.randn(1, 8, 16)
    inputs = policy._prepare_world_model_inputs(
        {
            "curr_states": curr_states,
            "next_states": next_states,
            "actions": torch.zeros(1, 16, 7),
            "rewards": torch.zeros(1, 16, 1),
            "dones": torch.zeros(1, 16, 1, dtype=torch.bool),
        }
    )

    assert torch.equal(inputs["curr_obs"]["states"], curr_states.transpose(1, 2))
    assert torch.equal(inputs["next_obs"]["states"], next_states.transpose(1, 2))
    assert inputs["actions"].shape == (1, 16, 7)


def test_dreamzero_rl_forward_clamps_actions_before_action_head():
    dreamzero_policy_module = pytest.importorskip(
        "rlinf.models.embodiment.dreamzero.dreamzero_policy"
    )
    DreamZeroPolicy = dreamzero_policy_module.DreamZeroPolicy

    class FakeDreamZeroPolicy(DreamZeroPolicy):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.seen_actions = []

        def _slice_rl_sample(self, inputs, index):
            return {
                key: value[index : index + 1] if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }

    policy = FakeDreamZeroPolicy()

    original_forward = dreamzero_policy_module.VLA.forward

    def fake_vla_forward(self, sample_input):
        self.seen_actions.append(sample_input["action"].detach().clone())
        return BatchFeature(data={"action_loss": sample_input["action"].sum()})

    dreamzero_policy_module.VLA.forward = fake_vla_forward
    try:
        outputs = policy.rl_forward(
            forward_inputs={
                "dreamzero_rl.action": torch.tensor(
                    [[[[-2.0, -0.5], [0.5, 2.0]]]], dtype=torch.float32
                )
            }
        )
    finally:
        dreamzero_policy_module.VLA.forward = original_forward

    seen_action = policy.seen_actions[0]
    assert seen_action.min() >= -1.0
    assert seen_action.max() <= 1.0
    assert torch.allclose(outputs["dreamzero_losses"]["action_loss"], torch.tensor([0.0]))


def test_dreamzero_rl_forward_recomputes_action_chain_logprob(monkeypatch):
    dreamzero_policy_module = pytest.importorskip(
        "rlinf.models.embodiment.dreamzero.dreamzero_policy"
    )
    DreamZeroPolicy = dreamzero_policy_module.DreamZeroPolicy

    class FakeDreamZeroPolicy(DreamZeroPolicy):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def _slice_rl_sample(self, inputs, index):
            return {
                key: value[index : index + 1] if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }

        def lazy_joint_video_action_causal(self, inputs, *, return_video=True):
            return BatchFeature(data={"action_pred": inputs["action"] + self.weight})

    policy = FakeDreamZeroPolicy()
    original_forward = dreamzero_policy_module.VLA.forward

    def fake_vla_forward(self, sample_input):
        raise AssertionError("action-chain logprob path should not run VLA.forward")

    monkeypatch.setenv("DREAMZERO_ACTION_LOGPROB_STD", "1.0")
    dreamzero_policy_module.VLA.forward = fake_vla_forward
    try:
        outputs = policy.rl_forward(
            forward_inputs={
                "dreamzero_rl.action": torch.tensor(
                    [[[0.25, -0.25], [0.5, -0.5]]], dtype=torch.float32
                ),
                "dreamzero_rl.action_mask": torch.ones(1, 2, 2, dtype=torch.bool),
                "dreamzero_old_action_logprob": torch.tensor([-4.0]),
                "dreamzero_action_logprob_std": torch.tensor([1.0]),
            }
        )
    finally:
        dreamzero_policy_module.VLA.forward = original_forward

    assert outputs["logprobs"].shape == (1,)
    assert outputs["dreamzero_losses"]["action_logprobs"].shape == (1,)
    assert torch.equal(
        outputs["dreamzero_losses"]["old_action_logprobs"], torch.tensor([-4.0])
    )
    outputs["logprobs"].sum().backward()
    assert policy.weight.grad is not None


def test_dreamzero_rl_forward_action_loss_proxy_ignores_action_chain_payload(
    monkeypatch,
):
    dreamzero_policy_module = pytest.importorskip(
        "rlinf.models.embodiment.dreamzero.dreamzero_policy"
    )
    DreamZeroPolicy = dreamzero_policy_module.DreamZeroPolicy

    class FakeDreamZeroPolicy(DreamZeroPolicy):
        def __init__(self):
            torch.nn.Module.__init__(self)

        def _slice_rl_sample(self, inputs, index):
            return {
                key: value[index : index + 1] if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }

        def lazy_joint_video_action_causal(self, inputs, *, return_video=True):
            raise AssertionError("PPO proxy path should not recompute action-chain logprob")

    policy = FakeDreamZeroPolicy()

    def fake_vla_forward(self, sample_input):
        return BatchFeature(data={"action_loss": sample_input["action"].sum() + 1.0})

    monkeypatch.setattr(dreamzero_policy_module.VLA, "forward", fake_vla_forward)
    outputs = policy.rl_forward(
        forward_inputs={
            "dreamzero_rl.action": torch.tensor(
                [[[0.25, -0.25], [0.5, -0.5]]], dtype=torch.float32
            ),
            "dreamzero_rl.action_mask": torch.ones(1, 2, 2, dtype=torch.bool),
            "dreamzero_old_action_logprob": torch.tensor([-4.0]),
            "dreamzero_action_logprob_std": torch.tensor([1.0]),
        },
        dreamzero_logprob_mode="action_loss_proxy",
    )

    assert torch.allclose(outputs["dreamzero_losses"]["action_loss"], torch.tensor([1.0]))
    assert "action_logprobs" not in outputs["dreamzero_losses"]
    assert outputs["logprobs"].shape == (1, 4)


def test_dreamzero_rl_forward_requests_action_only_recompute(monkeypatch):
    dreamzero_policy_module = pytest.importorskip(
        "rlinf.models.embodiment.dreamzero.dreamzero_policy"
    )
    DreamZeroPolicy = dreamzero_policy_module.DreamZeroPolicy

    class FakeDreamZeroPolicy(DreamZeroPolicy):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.return_video_values = []

        def lazy_joint_video_action_causal(self, inputs, *, return_video=True):
            self.return_video_values.append(return_video)
            return BatchFeature(data={"action_pred": inputs["action"]})

    monkeypatch.setenv("DREAMZERO_ACTION_LOGPROB_STD", "1.0")
    policy = FakeDreamZeroPolicy()
    outputs = policy.rl_forward(
        forward_inputs={
            "dreamzero_rl.action": torch.tensor(
                [[[0.25, -0.25], [0.5, -0.5]]], dtype=torch.float32
            ),
            "dreamzero_rl.action_mask": torch.ones(1, 2, 2, dtype=torch.bool),
            "dreamzero_old_action_logprob": torch.tensor([-4.0]),
            "dreamzero_action_logprob_std": torch.tensor([1.0]),
        }
    )

    assert policy.return_video_values == [False]
    assert "action_logprobs" in outputs["dreamzero_losses"]


def test_dreamzero_wan_forward_blocks_skips_video_head_for_action_only():
    patch_module = importlib.import_module(
        "rlinf.models.embodiment.dreamzero.patch.wan_causal_model_forward_inference"
    )

    class FakeBlock:
        def __call__(self, *, x, **kwargs):
            return x + 1.0, torch.ones(1)

    class FakeWanModel:
        def __init__(self):
            self.dim = 4
            self.freq_dim = 4
            self.freqs_action = torch.zeros(1)
            self.freqs_state = torch.zeros(1)
            self.gradient_checkpointing = False
            self.blocks = [FakeBlock()]
            self.head_calls = 0

        def action_encoder(self, action, timestep_action, embodiment_id):
            del timestep_action, embodiment_id
            return torch.ones(action.shape[0], action.shape[1], self.dim)

        def state_encoder(self, state, embodiment_id):
            del embodiment_id
            return torch.ones(state.shape[0], state.shape[1], self.dim)

        def action_decoder(self, action_tokens, embodiment_id):
            del embodiment_id
            return action_tokens

        def time_embedding(self, timestep_embedding):
            return torch.zeros(
                timestep_embedding.shape[0],
                self.dim,
                dtype=timestep_embedding.dtype,
                device=timestep_embedding.device,
            )

        def time_projection(self, embeddings):
            return torch.zeros(
                *embeddings.shape[:-1],
                6 * self.dim,
                dtype=embeddings.dtype,
                device=embeddings.device,
            )

        def text_embedding(self, context):
            return context

        def head(self, x_video, e_video):
            del e_video
            self.head_calls += 1
            return x_video

    model = FakeWanModel()
    x_video, action_noise_pred, updated_kv_caches = patch_module._forward_blocks(
        model,
        x=torch.zeros(1, 4, 1, 1, 2),
        seq_len=2,
        freqs=torch.zeros(1),
        timestep=torch.zeros(1, 1, dtype=torch.int64),
        context=torch.zeros(1, 2, 4),
        clip_feature=None,
        embodiment_id=torch.zeros(1, dtype=torch.long),
        action=torch.zeros(1, 1, 3),
        timestep_action=torch.zeros(1, 1, dtype=torch.int64),
        state=torch.zeros(1, 1, 3),
        kv_cache=[torch.zeros(1)],
        current_start_frame=0,
        return_video_pred=False,
    )

    assert x_video is None
    assert model.head_calls == 0
    assert action_noise_pred is not None
    assert len(updated_kv_caches) == 1


def test_dreamzero_wan_forward_inference_skips_unpatchify_for_action_only():
    patch_module = importlib.import_module(
        "rlinf.models.embodiment.dreamzero.patch.wan_causal_model_forward_inference"
    )

    class FakeWanModel:
        model_type = "t2v"
        concat_first_frame_latent = False
        text_len = 2

        def __init__(self):
            self.unpatchify_calls = 0
            self._forward_blocks = types.MethodType(patch_module._forward_blocks, self)

        def patch_embedding(self, x):
            return x

        def _create_freqs(self, *, grid_size, start_frame):
            del grid_size, start_frame
            return torch.zeros(1)

        def unpatchify(self, x_video, grid_size):
            del x_video, grid_size
            self.unpatchify_calls += 1
            raise AssertionError("unpatchify should be skipped in action-only inference")

        dim = 4
        freq_dim = 4
        freqs_action = torch.zeros(1)
        freqs_state = torch.zeros(1)
        gradient_checkpointing = False
        blocks = []

        def action_encoder(self, action, timestep_action, embodiment_id):
            del timestep_action, embodiment_id
            return torch.ones(action.shape[0], action.shape[1], self.dim)

        def state_encoder(self, state, embodiment_id):
            del embodiment_id
            return torch.ones(state.shape[0], state.shape[1], self.dim)

        def action_decoder(self, action_tokens, embodiment_id):
            del embodiment_id
            return action_tokens

        def time_embedding(self, timestep_embedding):
            return torch.zeros(
                timestep_embedding.shape[0],
                self.dim,
                dtype=timestep_embedding.dtype,
                device=timestep_embedding.device,
            )

        def time_projection(self, embeddings):
            return torch.zeros(
                *embeddings.shape[:-1],
                6 * self.dim,
                dtype=embeddings.dtype,
                device=embeddings.device,
            )

        def text_embedding(self, context):
            return context

        def head(self, x_video, e_video):
            del e_video
            return x_video

    model = FakeWanModel()
    video_noise_pred, action_noise_pred, updated_kv_caches = patch_module._forward_inference(
        model,
        x=torch.zeros(1, 4, 1, 1, 2),
        timestep=torch.zeros(1, 1, dtype=torch.int64),
        context=torch.zeros(1, 2, 4),
        seq_len=2,
        kv_cache=[],
        crossattn_cache=[],
        current_start_frame=0,
        y=None,
        clip_feature=None,
        action=torch.zeros(1, 1, 3),
        timestep_action=torch.zeros(1, 1, dtype=torch.int64),
        state=torch.zeros(1, 1, 3),
        embodiment_id=torch.zeros(1, dtype=torch.long),
        return_video_pred=False,
    )

    assert video_noise_pred is None
    assert model.unpatchify_calls == 0
    assert action_noise_pred is not None
    assert updated_kv_caches == []


def test_dreamzero_run_diffusion_steps_passes_action_only_flag_to_model():
    patch_module = importlib.import_module(
        "rlinf.models.embodiment.dreamzero.patch.wan_policy_head_action_only"
    )

    class FakePolicyHead:
        trt_engine = None
        ip_size = 1

        def __init__(self):
            self.seen_return_video_flags = []

        def model(self, *args, **kwargs):
            del args
            self.seen_return_video_flags.append(kwargs["return_video_pred"])
            return None, kwargs["action"] + 1.0, [torch.ones(1)]

        def _exchange_predictions(self, predictions):
            return predictions

    head = FakePolicyHead()
    with patch_module._action_only_video_context(enabled=True):
        predictions = patch_module._run_diffusion_steps(
            head,
            noisy_input=torch.zeros(1, 1, 1, 1, 2),
            timestep=torch.zeros(1, 1, dtype=torch.int64),
            action=torch.zeros(1, 2, 3),
            timestep_action=torch.zeros(1, 2, dtype=torch.int64),
            state=torch.zeros(1, 1, 3),
            embodiment_id=torch.zeros(1, dtype=torch.long),
            context=[torch.zeros(1, 2, 4)],
            seq_len=2,
            y=torch.zeros(1, 1, 1, 1, 2),
            clip_feature=torch.zeros(1, 4),
            kv_caches=[[torch.zeros(1)]],
            crossattn_caches=[[torch.zeros(1)]],
            kv_cache_metadata={"start_frame": 0, "update_kv_cache": False},
        )

    assert head.seen_return_video_flags == [False]
    assert torch.equal(predictions[0][0], torch.zeros(1, 1, 1, 1, 2))
    assert torch.equal(predictions[0][1], torch.ones(1, 2, 3))


def test_dreamzero_action_only_video_scheduler_step_is_noop():
    patch_module = importlib.import_module(
        "rlinf.models.embodiment.dreamzero.patch.wan_policy_head_action_only"
    )
    called = {"value": False}

    def original_step(self, *, model_output, timestep, sample, step_index, return_dict=True):
        del self, model_output, timestep, sample, step_index, return_dict
        called["value"] = True
        raise AssertionError("video scheduler step should be skipped in action-only mode")

    wrapped_step = patch_module.flow_unipc_step(original_step)
    sample = torch.zeros(1, 1, 1, 1, 2)

    with patch_module._action_only_video_context(enabled=True):
        result = wrapped_step(
            object(),
            model_output=torch.ones_like(sample),
            timestep=torch.tensor(0),
            sample=sample,
            step_index=0,
            return_dict=False,
        )

    assert called["value"] is False
    assert result == (sample,)


def test_dreamzero_get_model_registers_action_only_video_postprocess_patches(monkeypatch):
    dreamzero_module = importlib.import_module("rlinf.models.embodiment.dreamzero")
    patcher_module = importlib.import_module("rlinf.utils.patcher")
    original_apply = patcher_module.Patcher.apply

    def stop_after_patch_registration():
        raise RuntimeError("stop after patch registration")

    monkeypatch.setattr(patcher_module.Patcher, "apply", stop_after_patch_registration)

    with pytest.raises(RuntimeError, match="stop after patch registration"):
        dreamzero_module.get_model(OmegaConf.create({"model_path": "/tmp/not-needed"}))

    mappings = patcher_module.Patcher._mappings_dict
    wrappers = patcher_module.Patcher._wrappers_dict
    assert (
        mappings[
            "groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk.CausalWanModel._forward_inference"
        ]
        == "rlinf.models.embodiment.dreamzero.patch.wan_causal_model_forward_inference._forward_inference"
    )
    assert (
        mappings[
            "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf.WANPolicyHead._run_diffusion_steps"
        ]
        == "rlinf.models.embodiment.dreamzero.patch.wan_policy_head_action_only._run_diffusion_steps"
    )
    assert (
        "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf.WANPolicyHead.lazy_joint_video_action"
        in wrappers
    )
    assert (
        "groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler.FlowUniPCMultistepScheduler.step"
        in wrappers
    )

    patcher_module.Patcher.apply = original_apply
    patcher_module.Patcher.clear()


def test_rollout_generate_releases_dreamzero_inference_cache():
    rollout_module = pytest.importorskip(
        "rlinf.workers.rollout.hf.huggingface_worker"
    )
    MultiStepRolloutWorker = rollout_module.MultiStepRolloutWorker

    class FakeModel:
        def __init__(self):
            self.release_count = 0

        def release_inference_cache(self):
            self.release_count += 1

    class FakeWorker(MultiStepRolloutWorker):
        def __init__(self):
            self.enable_offload = False
            self.rollout_epoch = 2
            self._rank = 0
            self.hf_model = FakeModel()
            self.generated = 0
            self.empty_cache_count = 0

            class FakeTorchPlatform:
                def __init__(inner_self, outer):
                    inner_self.outer = outer

                def empty_cache(inner_self):
                    inner_self.outer.empty_cache_count += 1

            self.torch_platform = FakeTorchPlatform(self)

        async def generate_one_epoch(self, input_channel, output_channel):
            self.generated += 1

    worker = FakeWorker()
    asyncio.run(worker.generate(input_channel=None, output_channel=None))

    assert worker.generated == 2
    assert worker.hf_model.release_count == 1
    assert worker.empty_cache_count == 1


def test_rollout_sync_filters_dreamzero_lora_receiver_state_dict():
    rollout_module = pytest.importorskip(
        "rlinf.workers.rollout.hf.huggingface_worker"
    )
    MultiStepRolloutWorker = rollout_module.MultiStepRolloutWorker

    class FakeModel:
        def state_dict(self):
            return {
                "action_head.model.base_model.model.blocks.0.attn.q.lora_A.default.weight": torch.ones(1),
                "action_head.model.base_model.model.action_decoder.weight": torch.ones(1),
                "action_head.model.base_model.model.action_encoder.weight": torch.ones(1),
                "action_head.model.base_model.model.state_encoder.weight": torch.ones(1),
                "action_head.model.base_model.model.blocks.0.attn.q.base_layer.weight": torch.ones(1),
            }

        def set_global_step(self, step):
            self.global_step = step

    class FakeWeightSyncer:
        def __init__(self):
            self.receiver_state_keys = None

        def receiver_initialized(self):
            return False

        async def init_receiver(self, *, state_dict, recv, send):
            del recv, send
            self.receiver_state_keys = set(state_dict)

        async def apply(self, model, recv):
            del model, recv
            return 0

    class FakeWorker(MultiStepRolloutWorker):
        def __init__(self):
            self.cfg = OmegaConf.create(
                {
                    "actor": {
                        "model": {"model_type": "dreamzero", "is_lora": True},
                    },
                }
            )
            self.actor_group_name = "ActorGroup"
            self.actor_weight_src_rank = 0
            self._group_name = "RolloutGroup"
            self._weight_sync_rollout_ranks = [0]
            self._weight_sync_is_sender = False
            self._sync_weight_comm_options = None
            self.weight_syncer = FakeWeightSyncer()
            self.hf_model = FakeModel()
            self.finished_episodes = 0

            class FakeTorchPlatform:
                def empty_cache(inner_self):
                    return None

            self.torch_platform = FakeTorchPlatform()

        def broadcast(self, *args, **kwargs):
            raise AssertionError("syncer should not call recv in this unit test")

    worker = FakeWorker()
    asyncio.run(worker.sync_model_from_actor())

    assert worker.weight_syncer.receiver_state_keys == {
        "action_head.model.base_model.model.blocks.0.attn.q.lora_A.default.weight",
        "action_head.model.base_model.model.action_decoder.weight",
        "action_head.model.base_model.model.action_encoder.weight",
        "action_head.model.base_model.model.state_encoder.weight",
    }


def test_actor_rollout_state_dict_filters_dreamzero_lora_sync_names():
    actor_module = pytest.importorskip("rlinf.workers.actor.fsdp_actor_worker")
    EmbodiedFSDPActor = actor_module.EmbodiedFSDPActor

    class FakeActor(EmbodiedFSDPActor):
        def __init__(self):
            self.cfg = OmegaConf.create(
                {
                    "actor": {
                        "model": {"model_type": "dreamzero", "is_lora": True},
                    },
                }
            )
            self.param_names_need_sync = [
                "action_head.model.base_model.model.blocks.0.attn.q.lora_A.default.weight",
                "action_head.model.base_model.model.blocks.0.attn.q.base_layer.weight",
                "action_head.model.base_model.model.action_decoder.weight",
                "action_head.model.base_model.model.action_encoder.weight",
                "action_head.model.base_model.model.state_encoder.weight",
                "world_model.rssm.weight",
            ]

        def get_model_state_dict(self, *, cpu_offload, full_state_dict):
            assert cpu_offload is False
            assert full_state_dict is False
            return {
                key: torch.ones(1)
                for key in self.param_names_need_sync
            }

    actor = FakeActor()
    state_dict = actor.get_rollout_state_dict()

    assert set(state_dict) == {
        "action_head.model.base_model.model.blocks.0.attn.q.lora_A.default.weight",
        "action_head.model.base_model.model.action_decoder.weight",
        "action_head.model.base_model.model.action_encoder.weight",
        "action_head.model.base_model.model.state_encoder.weight",
    }
    assert actor.param_names_need_sync == list(state_dict.keys())


def test_actor_training_progress_logging_samples_long_runs():
    logged = [
        idx for idx in range(64) if should_log_actor_training_progress(idx, total=64)
    ]

    assert logged[0] == 0
    assert logged[-1] == 63
    assert len(logged) <= 18
    assert 3 in logged


def test_dreamzero_rl_forward_does_not_reuse_action_chain_graph_between_microbatches(
    monkeypatch,
):
    dreamzero_policy_module = pytest.importorskip(
        "rlinf.models.embodiment.dreamzero.dreamzero_policy"
    )
    DreamZeroPolicy = dreamzero_policy_module.DreamZeroPolicy

    class FakeActionHead:
        def __init__(self):
            self.kv_cache1 = None
            self.kv_cache_neg = None
            self.crossattn_cache = None
            self.crossattn_cache_neg = None
            self.clip_feas = None
            self.ys = None
            self.language = "cached"
            self.current_start_frame = 1
            self.skip_countdown = 1

    class FakeDreamZeroPolicy(DreamZeroPolicy):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
            self.action_head = FakeActionHead()

        def lazy_joint_video_action_causal(self, inputs, *, return_video=True):
            action_pred = (inputs["action"] + self.weight).pow(2)
            if self.action_head.kv_cache1 is not None:
                action_pred = action_pred + self.action_head.kv_cache1
            self.action_head.kv_cache1 = action_pred
            return BatchFeature(data={"action_pred": action_pred})

    def forward_inputs(action_value):
        return {
            "dreamzero_rl.action": torch.full(
                (1, 2, 2), action_value, dtype=torch.float32
            ),
            "dreamzero_rl.action_mask": torch.ones(1, 2, 2, dtype=torch.bool),
            "dreamzero_old_action_logprob": torch.tensor([-4.0]),
            "dreamzero_action_logprob_std": torch.tensor([1.0]),
        }

    monkeypatch.setenv("DREAMZERO_ACTION_LOGPROB_STD", "1.0")
    policy = FakeDreamZeroPolicy()

    first = policy.rl_forward(forward_inputs(0.25))
    first["logprobs"].sum().backward()
    policy.weight.grad = None
    second = policy.rl_forward(forward_inputs(0.5))
    second["logprobs"].sum().backward()

    assert policy.weight.grad is not None


def test_dreamzero_advantage_uses_unified_registry_entry():
    rewards = torch.tensor(
        [
            [[1.0], [0.0]],
            [[0.5], [0.25]],
        ],
        dtype=torch.float32,
    )
    dones = torch.zeros(3, 2, 1, dtype=torch.bool)
    dones[-1] = True
    loss_mask = torch.ones_like(rewards, dtype=torch.bool)

    result = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="dreamzero",
        rewards=rewards,
        dones=dones,
        gamma=1.0,
        gae_lambda=1.0,
        group_size=1,
        reward_type="chunk_level",
        loss_mask=loss_mask,
    )

    assert set(result) == {"advantages", "returns"}
    assert result["advantages"].shape == rewards.shape
    assert result["returns"].shape == rewards.shape
    assert torch.all(result["advantages"][loss_mask] >= 0)


def test_dreamzero_loader_disables_torch_compile_from_env(monkeypatch):
    dreamzero_model_module = pytest.importorskip("rlinf.models.embodiment.dreamzero")

    monkeypatch.setenv("DREAMZERO_DISABLE_TORCH_COMPILE", "1")

    def fn(x):
        return x

    original_compile = torch.compile
    try:
        dreamzero_model_module._disable_torch_compile_for_dreamzero()
        assert torch.compile(fn) is fn
        assert torch.compile()(fn) is fn
        assert dreamzero_model_module._dreamzero_disable_torch_compile() is True
    finally:
        monkeypatch.setattr(torch, "compile", original_compile)


def test_dreamzero_wan_policy_head_uses_configured_inference_timesteps():
    wan_module = pytest.importorskip(
        "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf"
    )

    cfg = wan_module.WANPolicyHeadConfig(
        skip_component_loading=True,
        train_architecture="none",
        tune_diffusion_model=False,
        text_encoder_cfg={"_target_": "torch.nn.Identity"},
        image_encoder_cfg={"_target_": "torch.nn.Identity"},
        vae_cfg={"_target_": "torch.nn.Identity"},
        diffusion_model_cfg={
            "_target_": "torch.nn.Linear",
            "in_features": 1,
            "out_features": 1,
        },
        action_dim=2,
        action_horizon=2,
        num_frames=1,
        num_inference_timesteps=4,
    )

    head = wan_module.WANPolicyHead(cfg)

    assert head.num_inference_timesteps == 4
    assert head.num_inference_steps == 4


def test_dreamzero_causal_inference_blocks_use_gradient_checkpointing(monkeypatch):
    import torch.utils.checkpoint

    wan_module = pytest.importorskip(
        "groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk"
    )

    checkpoint_calls = []

    def fake_checkpoint(function, *args, **kwargs):
        checkpoint_calls.append(kwargs.pop("use_reentrant", None))
        return function(*args, **kwargs)

    class FakeBlock(torch.nn.Module):
        def forward(self, x, **kwargs):
            return x + 1, kwargs.get("kv_cache")

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", fake_checkpoint)

    model = wan_module.CausalWanModel(
        model_type="t2v",
        patch_size=(1, 1, 1),
        frame_seqlen=1,
        text_len=2,
        in_dim=4,
        dim=4,
        ffn_dim=8,
        freq_dim=4,
        text_dim=4,
        out_dim=4,
        num_heads=2,
        num_layers=2,
        max_chunk_size=-1,
        cross_attn_norm=False,
        concat_first_frame_latent=False,
    )
    model.blocks = torch.nn.ModuleList([FakeBlock(), FakeBlock()])
    model.gradient_checkpointing = True

    x = torch.ones(1, 4, 1, 1, 1, requires_grad=True)
    timestep = torch.zeros(1, 1, dtype=torch.long)
    context = torch.zeros(1, 2, 4)
    freqs = torch.zeros(1, 1, 2)

    output, action_pred, caches = model._forward_blocks(
        x=x,
        seq_len=1,
        freqs=freqs,
        timestep=timestep,
        context=context,
        clip_feature=None,
        embodiment_id=None,
        action=None,
        timestep_action=None,
        state=None,
        kv_cache=[None, None],
        current_start_frame=1,
    )

    assert output.requires_grad
    assert action_pred is None
    assert caches == [None, None]
    assert checkpoint_calls == [False, False]


def test_dreamzero_libero_observation_transform_builds_inference_modalities():
    transforms = pytest.importorskip(
        "rlinf.data.datasets.dreamzero.data_transforms.observation"
    )

    transform = transforms.DreamZeroLiberoObservationTransform(num_history_frames=4)
    env_obs = {
        "main_images": torch.zeros(2, 256, 256, 3, dtype=torch.uint8),
        "wrist_images": torch.ones(2, 256, 256, 3, dtype=torch.uint8),
        "states": torch.zeros(2, 8, dtype=torch.float32),
        "task_descriptions": ["pick up the bowl", "open the drawer"],
    }

    converted = transform.convert(env_obs)

    assert converted["video.image"].shape == (2, 1, 256, 256, 3)
    assert converted["video.wrist_image"].shape == (2, 1, 256, 256, 3)
    assert converted["state.state"].shape == (2, 1, 8)
    assert converted["state.joint_position"].shape == (2, 1, 7)
    assert converted["state.gripper_position"].shape == (2, 1, 1)
    assert converted["annotation.language.task_description"] == [
        "pick up the bowl",
        "open the drawer",
    ]


def test_build_dreamzero_forward_inputs_preserves_batch_time_layout():
    rollout_batch = {
        "curr_obs": {
            "states": torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
        },
        "next_obs": {
            "states": (100 + torch.arange(2 * 3 * 8, dtype=torch.float32)).reshape(
                2, 3, 8
            )
        },
        "actions": torch.arange(2 * 3 * 4 * 7, dtype=torch.float32).reshape(2, 3, 4, 7),
        "rewards": torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4),
        "dones": torch.zeros(3, 3, 4, dtype=torch.bool),
        "forward_inputs": {
            "action": torch.zeros(2, 3, 28),
            "model_action": torch.zeros(2, 3, 4, 32),
            "states": torch.zeros(2, 3, 8),
            "dreamzero_rl.action": torch.zeros(2, 3, 24, 32),
            "dreamzero_rl.images": torch.zeros(2, 3, 1, 8, 8, 3),
        },
    }

    result = build_dreamzero_forward_inputs(rollout_batch)

    assert result["curr_states"].shape == (2, 3, 4, 8)
    assert result["next_states"].shape == (2, 3, 4, 8)
    assert result["actions"].shape == (2, 3, 4, 7)
    assert result["model_action"].shape == (2, 3, 128)
    assert result["rewards"].shape == (2, 3, 4, 1)
    assert result["dones"].shape == (2, 3, 4, 1)
    assert result["dreamzero_rl.action"].shape == (2, 3, 24, 32)
    assert result["dreamzero_rl.images"].shape == (2, 3, 9, 8, 8, 3)
    assert torch.equal(
        result["curr_states"][0, 0],
        rollout_batch["curr_obs"]["states"][0, 0].expand(4, 8),
    )
    assert torch.equal(result["actions"][0, 0], rollout_batch["actions"][0, 0])


def test_build_dreamzero_forward_inputs_clamps_model_actions_for_action_head():
    rollout_batch = {
        "curr_obs": {"states": torch.zeros(1, 1, 8)},
        "next_obs": {"states": torch.zeros(1, 1, 8)},
        "actions": torch.full((1, 1, 2, 7), 10.0),
        "rewards": torch.zeros(1, 1, 2),
        "dones": torch.zeros(1, 1, 2, dtype=torch.bool),
        "forward_inputs": {
            "model_action": torch.tensor([[[[-2.0, -0.5], [0.25, 2.0]]]]),
            "dreamzero_rl.action": torch.tensor([[[[-1.5, 0.5], [2.5, -0.25]]]]),
        },
    }

    result = build_dreamzero_forward_inputs(rollout_batch)

    assert result["dreamzero_rl.action"].min() >= -1.0
    assert result["dreamzero_rl.action"].max() <= 1.0
    assert result["model_action"].min() >= -1.0
    assert result["model_action"].max() <= 1.0
    assert torch.equal(result["actions"], rollout_batch["actions"])


def test_build_dreamzero_forward_inputs_expands_single_frame_dreamzero_video():
    rollout_batch = {
        "curr_obs": {"states": torch.zeros(1, 1, 8)},
        "next_obs": {"states": torch.zeros(1, 1, 8)},
        "actions": torch.zeros(1, 1, 24, 7),
        "rewards": torch.zeros(1, 1, 24),
        "dones": torch.zeros(1, 1, 24, dtype=torch.bool),
        "forward_inputs": {
            "dreamzero_rl.action": torch.zeros(1, 1, 24, 32),
            "dreamzero_rl.images": torch.arange(8 * 8 * 3, dtype=torch.uint8).reshape(
                1, 1, 1, 8, 8, 3
            ),
        },
    }

    result = build_dreamzero_forward_inputs(rollout_batch)

    assert result["dreamzero_rl.images"].shape == (1, 1, 9, 8, 8, 3)
    assert torch.equal(
        result["dreamzero_rl.images"][:, :, 0],
        result["dreamzero_rl.images"][:, :, -1],
    )


def test_build_dreamzero_forward_inputs_can_keep_only_action_head_payload():
    rollout_batch = {
        "curr_obs": {"states": torch.zeros(1, 1, 8)},
        "next_obs": {"states": torch.zeros(1, 1, 8)},
        "actions": torch.zeros(1, 1, 2, 7),
        "rewards": torch.zeros(1, 1, 2),
        "dones": torch.zeros(1, 1, 2, dtype=torch.bool),
        "forward_inputs": {
            "action": torch.zeros(1, 1, 14),
            "model_action": torch.full((1, 1, 2, 32), 2.0),
            "dreamzero_rl.action": torch.zeros(1, 1, 2, 32),
            "dreamzero_rl.images": torch.zeros(1, 1, 1, 8, 8, 3),
            "dreamzero_old_action_logprob": torch.zeros(1, 1),
            "dreamzero_action_logprob_std": torch.ones(1, 1),
        },
    }

    result = build_dreamzero_forward_inputs(
        rollout_batch,
        action_head_only=True,
    )

    assert set(result) == {
        "model_action",
        "dreamzero_rl.action",
        "dreamzero_rl.images",
        "dreamzero_old_action_logprob",
        "dreamzero_action_logprob_std",
    }
    assert result["model_action"].min() >= -1.0
    assert result["model_action"].max() <= 1.0
    assert result["dreamzero_rl.images"].shape == (1, 1, 9, 8, 8, 3)


def test_dreamzero_forward_payload_images_stack_across_online_history_lengths():
    from rlinf.models.embodiment.dreamzero.dreamzero_policy import DreamZeroPolicy

    policy = object.__new__(DreamZeroPolicy)

    first = {
        "dreamzero_rl.images": torch.zeros(2, 1, 8, 8, 3, dtype=torch.uint8),
        "dreamzero_rl.action": torch.zeros(2, 16, 32),
    }
    second = {
        "dreamzero_rl.images": torch.ones(2, 4, 8, 8, 3, dtype=torch.uint8),
        "dreamzero_rl.action": torch.zeros(2, 16, 32),
    }

    first = policy._normalize_forward_payload_for_rollout(first)
    second = policy._normalize_forward_payload_for_rollout(second)
    stacked = stack_list_of_dict_tensor([first, second])

    assert stacked["dreamzero_rl.images"].shape == (2, 2, 9, 8, 8, 3)
    assert torch.equal(stacked["dreamzero_rl.images"][0, :, 0], stacked["dreamzero_rl.images"][0, :, -1])


def test_build_dreamzero_forward_inputs_can_strip_action_head_payload_for_proxy():
    rollout_batch = {
        "curr_obs": {"states": torch.zeros(1, 1, 8)},
        "next_obs": {"states": torch.zeros(1, 1, 8)},
        "actions": torch.zeros(1, 1, 2, 7),
        "rewards": torch.zeros(1, 1, 2),
        "dones": torch.zeros(1, 1, 2, dtype=torch.bool),
        "forward_inputs": {
            "model_action": torch.zeros(1, 1, 2, 32),
            "dreamzero_rl.action": torch.zeros(1, 1, 2, 32),
            "dreamzero_rl.images": torch.zeros(1, 1, 1, 8, 8, 3),
        },
    }

    result = build_dreamzero_forward_inputs(
        rollout_batch,
        preserve_action_head_payload=False,
    )

    assert "model_action" not in result
    assert not any(key.startswith("dreamzero_rl.") for key in result)
    assert result["curr_states"].shape == (1, 1, 2, 8)
    assert result["actions"].shape == (1, 1, 2, 7)


def test_strip_dreamzero_action_head_payload_removes_rl_keys_and_model_action():
    forward_inputs = {
        "actions": torch.zeros(1, 1, 2, 7),
        "model_action": torch.zeros(1, 1, 64),
        "dreamzero_rl.action": torch.zeros(1, 1, 2, 32),
        "dreamzero_rl.images": torch.zeros(1, 1, 1, 8, 8, 3),
    }

    result = strip_dreamzero_action_head_payload(forward_inputs)

    assert set(result) == {"actions"}


def test_ensure_dreamzero_world_model_forward_inputs_rebuilds_from_batch():
    batch = {
        "curr_obs": {"states": torch.zeros(2, 8)},
        "next_obs": {"states": torch.ones(2, 8)},
        "actions": torch.zeros(2, 4, 7),
        "rewards": torch.zeros(2, 4),
        "dones": torch.zeros(2, 4, dtype=torch.bool),
    }
    forward_inputs = {
        "dreamzero_rl.action": torch.zeros(2, 4, 32),
        "model_action": torch.zeros(2, 128),
    }

    result = ensure_dreamzero_world_model_forward_inputs(batch, forward_inputs)

    assert "dreamzero_rl.action" not in result
    assert "model_action" not in result
    assert result["curr_states"].shape == (2, 4, 8)
    assert result["next_states"].shape == (2, 4, 8)
    assert result["actions"].shape == (2, 4, 7)
    assert result["rewards"].shape == (2, 4, 1)
    assert result["dones"].shape == (2, 4, 1)


def test_ensure_dreamzero_world_model_forward_inputs_normalizes_flat_payload():
    forward_inputs = {
        "curr_states": torch.zeros(1, 112),
        "next_states": torch.ones(1, 112),
        "actions": torch.zeros(1, 112),
        "rewards": torch.zeros(1, 1),
        "dones": torch.zeros(1, 1, dtype=torch.bool),
    }

    result = ensure_dreamzero_world_model_forward_inputs(
        {},
        forward_inputs,
        env_action_dim=7,
    )

    assert result["curr_states"].shape == (1, 16, 7)
    assert result["next_states"].shape == (1, 16, 7)
    assert result["actions"].shape == (1, 16, 7)
    assert result["rewards"].shape == (1, 16, 1)
    assert result["dones"].shape == (1, 16, 1)


def test_ensure_dreamzero_world_model_forward_inputs_rebuilds_misaligned_flat_payload_from_batch():
    batch = {
        "curr_obs": {"states": torch.zeros(1, 7, 16)},
        "next_obs": {"states": torch.ones(1, 7, 16)},
        "actions": torch.zeros(1, 16, 7),
        "rewards": torch.zeros(1, 16),
        "dones": torch.zeros(1, 16, dtype=torch.bool),
    }
    forward_inputs = {
        "curr_states": torch.zeros(1, 112),
        "next_states": torch.ones(1, 112),
        "actions": torch.zeros(1, 16),
        "rewards": torch.zeros(1, 1),
        "dones": torch.zeros(1, 1, dtype=torch.bool),
    }

    result = ensure_dreamzero_world_model_forward_inputs(batch, forward_inputs)

    assert result["curr_states"].shape == (1, 16, 7)
    assert result["next_states"].shape == (1, 16, 7)
    assert result["actions"].shape == (1, 16, 7)
    assert result["rewards"].shape == (1, 16, 1)
    assert result["dones"].shape == (1, 16, 1)


def test_dreamzero_prepare_world_model_inputs_collapses_env_action_axis_from_state_time():
    dreamzero_policy_module = pytest.importorskip(
        "rlinf.models.embodiment.dreamzero.dreamzero_policy"
    )
    DreamZeroPolicy = dreamzero_policy_module.DreamZeroPolicy

    class FakeConfig:
        env_action_dim = 7

    class FakeDreamZeroPolicy(DreamZeroPolicy):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.config = FakeConfig()

    policy = FakeDreamZeroPolicy()
    inputs = policy._prepare_world_model_inputs(
        {
            "curr_states": torch.zeros(1, 112, 8),
            "next_states": torch.ones(1, 112, 8),
            "actions": torch.zeros(1, 16, 7),
            "rewards": torch.zeros(1, 16, 1),
            "dones": torch.zeros(1, 16, 1, dtype=torch.bool),
        }
    )

    assert inputs["curr_obs"]["states"].shape == (1, 16, 8)
    assert inputs["next_obs"]["states"].shape == (1, 16, 8)
    assert inputs["actions"].shape == (1, 16, 7)


def test_world_model_proxy_uses_transition_payload_size_before_shuffle():
    rollout_batch = {
        "prev_logprobs": torch.zeros(5, 4, 16, 7),
        "advantages": torch.zeros(5, 4, 16),
        "returns": torch.zeros(5, 4, 16),
        "actions": torch.zeros(2, 4, 16, 7),
        "forward_inputs": {
            "curr_states": torch.zeros(2, 4, 16, 8),
            "next_states": torch.zeros(2, 4, 16, 8),
            "actions": torch.zeros(2, 4, 16, 7),
            "rewards": torch.zeros(2, 4, 16, 1),
            "dones": torch.zeros(2, 4, 16, 1, dtype=torch.bool),
            "dreamzero_rl.action": torch.zeros(2, 4, 16, 32),
        },
    }
    shuffle_id = torch.randperm(
        rollout_batch["prev_logprobs"].shape[0]
        * rollout_batch["prev_logprobs"].shape[1]
    )

    with pytest.raises(IndexError):
        process_nested_dict_for_train(rollout_batch, shuffle_id)

    rollout_size = get_dreamzero_train_rollout_size(
        rollout_batch,
        loss_type="dreamzero_world_model_proxy",
    )
    transition_shuffle_id = torch.randperm(rollout_size)
    processed = process_nested_dict_for_train(rollout_batch, transition_shuffle_id)
    flattened_rollout_size = get_dreamzero_train_rollout_size(
        processed,
        loss_type="dreamzero_world_model_proxy",
        is_flattened=True,
    )

    assert rollout_size == 8
    assert flattened_rollout_size == 8
    assert processed["forward_inputs"]["curr_states"].shape == (8, 16, 8)
    assert processed["forward_inputs"]["actions"].shape == (8, 16, 7)
    assert processed["prev_logprobs"].shape == (8, 16, 7)
