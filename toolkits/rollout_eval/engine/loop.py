"""Core rollout loop for lightweight non-Ray evaluation."""

from __future__ import annotations

from typing import Any

from toolkits.rollout_eval.checks.interface_checks import (
    assert_action_batch,
    assert_obs_batch,
)
from toolkits.rollout_eval.config_bridge import EvalRuntimeConfig
from toolkits.rollout_eval.profiling import (
    RolloutTorchProfiler,
    aggregate_profile_metrics,
)
from toolkits.rollout_eval.profiling.collector import LatencyCollector
from toolkits.rollout_eval.rollout_types import RolloutLoopResult, infer_batch_size


def run_rollout_loop(
    env_adapter: Any,
    model_adapter: Any,
    runtime: EvalRuntimeConfig,
) -> RolloutLoopResult:
    """Run the non-Ray rollout loop.

    Args:
        env_adapter: Environment adapter implementing reset/step.
        model_adapter: Model adapter implementing infer.
        runtime: Normalized runtime configuration.

    Returns:
        Rollout loop summary including measured latency counters.
    """
    obs_batch, _ = env_adapter.reset()
    expected_batch = infer_batch_size(obs_batch)

    latency = LatencyCollector()
    profiler = RolloutTorchProfiler(
        enabled=runtime.enable_torch_profiler,
        output_dir=runtime.profiler_output_dir,
    )

    with profiler:
        for _step in range(runtime.total_steps):
            assert_obs_batch(obs_batch, expected_batch=expected_batch)

            with profiler.trace("model.inference"):
                with latency.timed("model_infer"):
                    actions, _meta = model_adapter.infer(obs_batch=obs_batch, mode="eval")

            assert_action_batch(actions, expected_batch=expected_batch)

            with profiler.trace("env.simulation"):
                with latency.timed("env_step"):
                    step_result = env_adapter.step(actions)

            obs_batch = step_result.obs
            profiler.step()

    profile_metrics = aggregate_profile_metrics(profiler.key_averages())

    if hasattr(env_adapter, "close"):
        env_adapter.close()

    return RolloutLoopResult(
        total_steps=runtime.total_steps,
        warmup_steps=runtime.warmup_steps,
        measure_steps=max(0, runtime.total_steps - runtime.warmup_steps),
        latency=latency.stats,
        profile_metrics=profile_metrics,
    )
