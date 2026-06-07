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

import pytest
import torch
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
    strip_dreamzero_action_head_payload,
)


def test_dreamzero_policy_loss_weights_action_loss_by_environment_advantage():
    action_loss = torch.tensor([1.0, 3.0], dtype=torch.float32)
    advantages = torch.tensor([[2.0], [0.5]], dtype=torch.float32)

    loss, metrics = policy_loss(
        task_type="embodied",
        loss_type="dreamzero_action_head_rl",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=3,
        logprobs=torch.zeros(2, 1, 3, dtype=torch.float32),
        old_logprobs=torch.zeros(2, 1, 3, dtype=torch.float32),
        advantages=advantages,
        dreamzero_losses={
            "action_loss": action_loss,
        },
        dreamzero_metrics={"dreamzero/raw_action_loss": action_loss.mean()},
    )

    weights = advantages.reshape(-1).clamp_min(0)
    expected = (action_loss * weights).sum() / weights.abs().sum().clamp_min(1.0)
    assert torch.allclose(loss, expected)
    assert metrics["dreamzero/action_loss"] == expected.item()
    assert metrics["dreamzero/total_loss"] == loss.item()
    assert "dreamzero/model_loss" not in metrics


def test_dreamzero_policy_loss_positive_mode_zeroes_all_zero_advantages():
    action_loss = torch.tensor([1.0, 3.0], dtype=torch.float32, requires_grad=True)

    loss, metrics = policy_loss(
        task_type="embodied",
        loss_type="dreamzero_action_head_rl",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=3,
        logprobs=torch.zeros(2, 1, 3, dtype=torch.float32),
        old_logprobs=torch.zeros(2, 1, 3, dtype=torch.float32),
        advantages=torch.zeros(2, 1, dtype=torch.float32),
        dreamzero_losses={
            "action_loss": action_loss,
        },
    )

    loss.backward()

    assert torch.allclose(loss, torch.tensor(0.0))
    assert torch.equal(action_loss.grad, torch.zeros_like(action_loss))
    assert metrics["dreamzero/advantage_weight_mean"] == 0.0


def test_dreamzero_policy_loss_consumes_model_side_rl_payload_once():
    set_dreamzero_loss_payload(
        {
            "action_loss": torch.tensor([2.0, 4.0]),
        },
        {"dreamzero/raw_action_loss": torch.tensor(3.0)},
    )

    loss, metrics = policy_loss(
        task_type="embodied",
        loss_type="dreamzero_action_head_rl",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=3,
        logprobs=torch.zeros(2, 1, 3, dtype=torch.float32),
        old_logprobs=torch.zeros(2, 1, 3, dtype=torch.float32),
        advantages=torch.tensor([[1.0], [0.0]], dtype=torch.float32),
    )

    assert torch.allclose(loss, torch.tensor(2.0))
    assert metrics["dreamzero/raw_action_loss"] == 3.0


def test_dreamzero_action_head_rl_skips_generic_logprob_preprocess():
    loss, metrics = policy_loss(
        task_type="embodied",
        loss_type="dreamzero_action_head_rl",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=7,
        logprobs=torch.zeros(1, 768, dtype=torch.float32),
        old_logprobs=torch.zeros(1, 768, dtype=torch.float32),
        advantages=torch.tensor([2.0], dtype=torch.float32),
        dreamzero_losses={
            "action_loss": torch.tensor([3.0], dtype=torch.float32),
        },
    )

    assert torch.allclose(loss, torch.tensor(3.0))
    assert metrics["dreamzero/action_loss"] == 3.0


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
