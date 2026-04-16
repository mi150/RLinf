from __future__ import annotations

from dataclasses import dataclass

import torch

from toolkits.rollout_eval.config_bridge import EvalRuntimeConfig
from toolkits.rollout_eval.engine.loop import run_rollout_loop


@dataclass
class DummyStepResult:
    obs: dict
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: dict


class DummyEnvAdapter:
    def __init__(self) -> None:
        self.counter = 0

    def reset(self) -> tuple[dict, dict]:
        return {"states": torch.zeros(4, 3)}, {}

    def step(self, actions: torch.Tensor) -> DummyStepResult:
        self.counter += 1
        obs = {"states": torch.ones(4, 3) * self.counter}
        return DummyStepResult(
            obs=obs,
            reward=torch.ones(4),
            terminated=torch.zeros(4, dtype=torch.bool),
            truncated=torch.zeros(4, dtype=torch.bool),
            info={},
        )

    def close(self) -> None:
        return None


class DummyModelAdapter:
    def infer(self, obs_batch: dict, mode: str = "eval") -> tuple[torch.Tensor, dict]:
        batch = obs_batch["states"].shape[0]
        return torch.zeros(batch, 2), {"mode": mode}



def test_run_rollout_loop_tracks_warmup_and_measure_steps() -> None:
    runtime = EvalRuntimeConfig(
        env_type="maniskill",
        model_type="openvla_oft",
        model_path="/tmp/model",
        num_envs=4,
        group_size=1,
        num_action_chunks=1,
        total_steps=8,
        warmup_steps=3,
        seed=1,
    )

    result = run_rollout_loop(
        env_adapter=DummyEnvAdapter(),
        model_adapter=DummyModelAdapter(),
        runtime=runtime,
    )

    assert result.total_steps == 8
    assert result.warmup_steps == 3
    assert result.measure_steps == 5
    assert result.latency.model_infer_count == 8
    assert result.latency.env_step_count == 8
