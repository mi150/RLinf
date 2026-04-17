"""Single-side benchmark runners for env-only and model-only profiling."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch

from toolkits.rollout_eval.benchmark.metrics import aggregate_case_metrics
from toolkits.rollout_eval.benchmark.types import CaseMetrics
from toolkits.rollout_eval.rollout_types import infer_batch_size


@dataclass(frozen=True)
class SingleRunnerResult:
    """Result object returned by single-side benchmark runners."""

    metrics: CaseMetrics
    warmup_steps: int
    measure_steps: int


def _make_dummy_obs_from_template(template: Any) -> Any:
    """Create a dummy payload preserving structure and tensor/array shapes."""
    if torch.is_tensor(template):
        return torch.zeros_like(template)
    if isinstance(template, np.ndarray):
        return np.zeros_like(template)
    if isinstance(template, dict):
        return {key: _make_dummy_obs_from_template(value) for key, value in template.items()}
    if isinstance(template, list):
        return [_make_dummy_obs_from_template(value) for value in template]
    if isinstance(template, tuple):
        return tuple(_make_dummy_obs_from_template(value) for value in template)
    return template


def _infer_action_dim_from_obs(obs_batch: dict[str, Any]) -> int:
    states = obs_batch.get("states")
    if torch.is_tensor(states) and states.ndim >= 2:
        return int(states.shape[-1])
    if isinstance(states, np.ndarray) and states.ndim >= 2:
        return int(states.shape[-1])
    return 1


def _extract_action_space(env_adapter: Any) -> Any:
    action_space = getattr(env_adapter, "action_space", None)
    if action_space is not None:
        return action_space
    env = getattr(env_adapter, "env", None)
    if env is None:
        return None
    action_space = getattr(env, "action_space", None)
    if action_space is not None:
        return action_space
    getter = getattr(env, "get_wrapper_attr", None)
    if callable(getter):
        try:
            return getter("action_space")
        except Exception:
            return None
    return None


def _sample_random_actions(
    obs_batch: dict[str, Any],
    env_adapter: Any,
    action_dim_override: int | None = None,
) -> torch.Tensor:
    batch_size = infer_batch_size(obs_batch)
    if action_dim_override is not None and action_dim_override > 0:
        return torch.rand(batch_size, action_dim_override, dtype=torch.float32)

    action_space = _extract_action_space(env_adapter)
    if action_space is not None and hasattr(action_space, "sample"):
        samples = [action_space.sample() for _ in range(batch_size)]
        sample_array = np.asarray(samples)
        return torch.as_tensor(sample_array, dtype=torch.float32)

    action_dim = _infer_action_dim_from_obs(obs_batch)
    return torch.rand(batch_size, action_dim, dtype=torch.float32)


def run_env_only_case(
    *,
    env_adapter: Any,
    warmup_steps: int,
    measure_steps: int,
    action_dim_override: int | None = None,
) -> SingleRunnerResult:
    """Run env-only benchmark loop using random actions."""
    total_steps = warmup_steps + measure_steps
    obs_batch, _ = env_adapter.reset()

    env_step_latency_ms: list[float] = []
    measured_step_count = 0
    measured_env_seconds = 0.0

    for step_idx in range(total_steps):
        actions = _sample_random_actions(
            obs_batch, env_adapter, action_dim_override=action_dim_override
        )
        start = perf_counter()
        step_result = env_adapter.step(actions)
        elapsed = perf_counter() - start
        obs_batch = step_result.obs

        if step_idx >= warmup_steps:
            measured_step_count += 1
            measured_env_seconds += elapsed
            env_step_latency_ms.append(elapsed * 1000.0)

    if hasattr(env_adapter, "close"):
        env_adapter.close()

    metrics = aggregate_case_metrics(
        env_step_count=measured_step_count,
        env_step_seconds=measured_env_seconds,
        env_step_latency_ms=env_step_latency_ms,
    )
    return SingleRunnerResult(
        metrics=metrics,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
    )


def run_model_only_case(
    *,
    env_adapter: Any,
    model_adapter: Any,
    warmup_steps: int,
    measure_steps: int,
) -> SingleRunnerResult:
    """Run model-only benchmark loop with dummy observations from reset template."""
    total_steps = warmup_steps + measure_steps
    reset_obs, _ = env_adapter.reset()
    dummy_obs = _make_dummy_obs_from_template(reset_obs)

    model_infer_latency_ms: list[float] = []
    model_infer_gpu_time_ms: list[float] = []
    measured_infer_count = 0
    measured_model_seconds = 0.0

    for step_idx in range(total_steps):
        start_event = end_event = None
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        start = perf_counter()
        _actions, _meta = model_adapter.infer(obs_batch=dummy_obs, mode="eval")
        if start_event is not None and end_event is not None:
            end_event.record()
            torch.cuda.synchronize()
        elapsed = perf_counter() - start

        if step_idx >= warmup_steps:
            measured_infer_count += 1
            measured_model_seconds += elapsed
            model_infer_latency_ms.append(elapsed * 1000.0)
            if start_event is not None and end_event is not None:
                model_infer_gpu_time_ms.append(float(start_event.elapsed_time(end_event)))

    if hasattr(env_adapter, "close"):
        env_adapter.close()

    metrics = aggregate_case_metrics(
        model_infer_count=measured_infer_count,
        model_infer_seconds=measured_model_seconds,
        model_infer_latency_ms=model_infer_latency_ms,
        model_infer_gpu_time_ms=model_infer_gpu_time_ms if torch.cuda.is_available() else None,
    )
    return SingleRunnerResult(
        metrics=metrics,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
    )
