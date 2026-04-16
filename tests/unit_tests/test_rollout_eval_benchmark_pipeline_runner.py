"""Unit tests for rollout_eval benchmark dual-process pipeline runner."""

from __future__ import annotations

import time

import pytest

from toolkits.rollout_eval.benchmark.pipeline_runner import (
    PipelineRunnerConfig,
    run_dual_process_pipeline,
)


class FakeSimAdapter:
    """Simple sim adapter with deterministic reset/step behavior."""

    def __init__(self) -> None:
        self._value = 0

    def reset(self) -> dict[str, int]:
        self._value = 0
        return {"value": self._value}

    def step(self, action: int) -> dict[str, int]:
        time.sleep(0.001)
        self._value += int(action)
        return {"value": self._value}


class FakeModelAdapter:
    """Simple model adapter producing incremental positive action."""

    def infer(self, observation: dict[str, int]) -> int:
        time.sleep(0.001)
        return int(observation["value"]) + 1


class SlowModelAdapter:
    """Slow model adapter used to validate timeout-safe stop behavior."""

    def infer(self, observation: dict[str, int]) -> int:
        time.sleep(0.5)
        return int(observation["value"]) + 1


def _sim_factory() -> FakeSimAdapter:
    return FakeSimAdapter()


def _model_factory() -> FakeModelAdapter:
    return FakeModelAdapter()


def _slow_model_factory() -> SlowModelAdapter:
    return SlowModelAdapter()


def _preferred_start_method() -> str:
    try:
        import multiprocessing as mp

        mp.get_context("fork")
        return "fork"
    except Exception:
        return "spawn"


def test_run_dual_process_pipeline_liveness_and_non_zero_throughput() -> None:
    metrics = run_dual_process_pipeline(
        sim_factory=_sim_factory,
        model_factory=_model_factory,
        config=PipelineRunnerConfig(
            warmup_steps=2,
            measure_steps=20,
            queue_size=1,
            queue_timeout_s=0.5,
            join_timeout_s=1.0,
            run_timeout_s=5.0,
            start_method=_preferred_start_method(),
        ),
    )

    assert metrics.env_steps_per_sec > 0.0
    assert metrics.model_infers_per_sec > 0.0
    assert metrics.pipeline_samples_per_sec > 0.0
    assert metrics.env_step_latency_ms is not None
    assert metrics.model_infer_latency_ms is not None
    assert metrics.pipeline_step_latency_ms is not None
    assert metrics.env_step_latency_ms.avg_ms > 0.0
    assert metrics.model_infer_latency_ms.avg_ms > 0.0
    assert metrics.pipeline_step_latency_ms.avg_ms > 0.0


def test_run_dual_process_pipeline_timeout_safe() -> None:
    with pytest.raises(TimeoutError, match="timed out"):
        run_dual_process_pipeline(
            sim_factory=_sim_factory,
            model_factory=_slow_model_factory,
            config=PipelineRunnerConfig(
                warmup_steps=0,
                measure_steps=10,
                queue_size=1,
                queue_timeout_s=0.05,
                join_timeout_s=0.5,
                run_timeout_s=0.4,
                start_method=_preferred_start_method(),
            ),
        )
