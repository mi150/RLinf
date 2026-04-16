from __future__ import annotations

import pytest

from toolkits.rollout_eval.benchmark.metrics import (
    aggregate_case_metrics,
    summarize_latency_ms,
    throughput,
)


def test_throughput_handles_zero_seconds() -> None:
    assert throughput(10, 0.0) == 0.0


def test_throughput_basic_math() -> None:
    assert throughput(25, 5.0) == pytest.approx(5.0)


def test_latency_summary_avg_p50_p95() -> None:
    summary = summarize_latency_ms([1.0, 2.0, 10.0])
    assert summary.avg_ms == pytest.approx(13.0 / 3)
    assert summary.p50_ms == pytest.approx(2.0)
    assert summary.p95_ms >= summary.p50_ms


def test_aggregate_case_metrics_builds_expected_outputs() -> None:
    metrics = aggregate_case_metrics(
        env_step_count=200,
        env_step_seconds=10.0,
        model_infer_count=150,
        model_infer_seconds=6.0,
        pipeline_sample_count=120,
        pipeline_seconds=8.0,
        env_step_latency_ms=[1.0, 2.0, 3.0],
        model_infer_latency_ms=[4.0, 5.0],
        model_infer_gpu_time_ms=[1.2, 1.4, 1.6],
        pipeline_step_latency_ms=[2.0, 2.5, 3.0],
    )

    assert metrics.env_steps_per_sec == pytest.approx(20.0)
    assert metrics.model_infers_per_sec == pytest.approx(25.0)
    assert metrics.pipeline_samples_per_sec == pytest.approx(15.0)
    assert metrics.env_step_latency_ms is not None
    assert metrics.env_step_latency_ms.avg_ms == pytest.approx(2.0)
    assert metrics.model_infer_latency_ms is not None
    assert metrics.model_infer_latency_ms.p50_ms == pytest.approx(4.5)
    assert metrics.model_infer_gpu_time_ms is not None
    assert metrics.model_infer_gpu_time_ms.avg_ms == pytest.approx(1.4)
    assert metrics.pipeline_step_latency_ms is not None
    assert metrics.pipeline_step_latency_ms.p95_ms >= metrics.pipeline_step_latency_ms.p50_ms
