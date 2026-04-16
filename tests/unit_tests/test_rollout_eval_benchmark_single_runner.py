from __future__ import annotations

import torch

from toolkits.rollout_eval.benchmark.single_runner import (
    run_env_only_case,
    run_model_only_case,
)
from toolkits.rollout_eval.rollout_types import EnvStepResult


class _FakeEnvAdapter:
    class _ActionSpace:
        def __init__(self, action_dim: int) -> None:
            self.action_dim = action_dim

        def sample(self):
            return torch.rand(self.action_dim).numpy()

    class _InnerEnv:
        def __init__(self, action_dim: int) -> None:
            self.action_space = _FakeEnvAdapter._ActionSpace(action_dim)

    def __init__(self, batch_size: int = 2, obs_dim: int = 3, action_dim: int = 2) -> None:
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.step_calls = 0
        self.last_actions: list[torch.Tensor] = []
        self.closed = False
        self.env = _FakeEnvAdapter._InnerEnv(action_dim)

    def reset(self):
        obs = {
            "states": torch.ones(self.batch_size, self.obs_dim, dtype=torch.float32),
            "task_descriptions": ["task"] * self.batch_size,
        }
        return obs, {}

    def step(self, actions: torch.Tensor) -> EnvStepResult:
        self.step_calls += 1
        self.last_actions.append(actions.detach().clone())
        obs = {
            "states": torch.full(
                (self.batch_size, self.obs_dim),
                float(self.step_calls + 1),
                dtype=torch.float32,
            ),
            "task_descriptions": ["task"] * self.batch_size,
        }
        return EnvStepResult(obs=obs)

    def close(self) -> None:
        self.closed = True


class _FakeModelAdapter:
    def __init__(self) -> None:
        self.calls = 0
        self.last_obs = None

    def infer(self, obs_batch, mode: str = "eval"):
        self.calls += 1
        self.last_obs = obs_batch
        batch = int(obs_batch["states"].shape[0])
        return torch.zeros(batch, 2), {}


def test_run_env_only_case_uses_random_actions_and_env_metrics() -> None:
    env = _FakeEnvAdapter(batch_size=3, obs_dim=4, action_dim=2)
    result = run_env_only_case(env_adapter=env, warmup_steps=1, measure_steps=4)

    assert env.step_calls == 5
    assert env.closed is True
    assert result.metrics.env_steps_per_sec > 0.0
    assert result.metrics.model_infers_per_sec == 0.0
    assert result.metrics.pipeline_samples_per_sec == 0.0
    assert result.metrics.env_step_latency_ms is not None
    assert result.metrics.env_step_latency_ms.avg_ms >= 0.0
    assert result.metrics.env_step_latency_ms.p95_ms >= result.metrics.env_step_latency_ms.p50_ms
    assert all(torch.any(action > 0) for action in env.last_actions)
    assert all(tuple(action.shape) == (3, 2) for action in env.last_actions)


def test_run_env_only_case_honors_action_dim_override() -> None:
    env = _FakeEnvAdapter(batch_size=3, obs_dim=4, action_dim=2)
    result = run_env_only_case(
        env_adapter=env,
        warmup_steps=1,
        measure_steps=2,
        action_dim_override=7,
    )

    assert result.metrics.env_steps_per_sec > 0.0
    assert all(tuple(action.shape) == (3, 7) for action in env.last_actions)


def test_run_model_only_case_builds_dummy_obs_from_reset_template() -> None:
    env = _FakeEnvAdapter(batch_size=2, obs_dim=5)
    model = _FakeModelAdapter()

    result = run_model_only_case(
        env_adapter=env,
        model_adapter=model,
        warmup_steps=2,
        measure_steps=3,
    )

    assert model.calls == 5
    assert env.step_calls == 0
    assert env.closed is True
    assert model.last_obs is not None
    assert torch.equal(model.last_obs["states"], torch.zeros(2, 5))
    assert result.metrics.model_infers_per_sec > 0.0
    assert result.metrics.env_steps_per_sec == 0.0
    assert result.metrics.model_infer_latency_ms is not None
    assert result.metrics.model_infer_latency_ms.avg_ms >= 0.0
    assert result.metrics.model_infer_latency_ms.p95_ms >= result.metrics.model_infer_latency_ms.p50_ms
