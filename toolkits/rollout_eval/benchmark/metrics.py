"""Metric aggregation utilities for rollout_eval benchmark."""

from __future__ import annotations

from math import ceil, floor

from toolkits.rollout_eval.benchmark.types import CaseMetrics, LatencySummary


def throughput(count: int, seconds: float) -> float:
    """Compute per-second throughput with non-positive guard."""
    if seconds <= 0:
        return 0.0
    return float(count) / seconds


def _percentile(sorted_samples: list[float], percentile: float) -> float:
    if not sorted_samples:
        return 0.0
    if len(sorted_samples) == 1:
        return float(sorted_samples[0])

    rank = (len(sorted_samples) - 1) * percentile / 100.0
    lower = floor(rank)
    upper = ceil(rank)
    if lower == upper:
        return float(sorted_samples[lower])
    weight = rank - lower
    return float(sorted_samples[lower] * (1.0 - weight) + sorted_samples[upper] * weight)


def summarize_latency_ms(samples: list[float]) -> LatencySummary:
    """Summarize latency samples into avg/p50/p95 milliseconds."""
    if not samples:
        return LatencySummary(avg_ms=0.0, p50_ms=0.0, p95_ms=0.0)

    sorted_samples = sorted(float(sample) for sample in samples)
    avg_ms = sum(sorted_samples) / len(sorted_samples)
    p50_ms = _percentile(sorted_samples, 50.0)
    p95_ms = _percentile(sorted_samples, 95.0)
    return LatencySummary(avg_ms=avg_ms, p50_ms=p50_ms, p95_ms=p95_ms)


def aggregate_case_metrics(
    *,
    env_step_count: int = 0,
    env_step_seconds: float = 0.0,
    model_infer_count: int = 0,
    model_infer_seconds: float = 0.0,
    pipeline_sample_count: int = 0,
    pipeline_seconds: float = 0.0,
    env_step_latency_ms: list[float] | None = None,
    model_infer_latency_ms: list[float] | None = None,
    model_infer_gpu_time_ms: list[float] | None = None,
    pipeline_step_latency_ms: list[float] | None = None,
) -> CaseMetrics:
    """Build a case metric object from raw counters and latency samples."""
    return CaseMetrics(
        env_steps_per_sec=throughput(env_step_count, env_step_seconds),
        model_infers_per_sec=throughput(model_infer_count, model_infer_seconds),
        pipeline_samples_per_sec=throughput(pipeline_sample_count, pipeline_seconds),
        env_step_latency_ms=summarize_latency_ms(env_step_latency_ms or []),
        model_infer_latency_ms=summarize_latency_ms(model_infer_latency_ms or []),
        model_infer_gpu_time_ms=(
            summarize_latency_ms(model_infer_gpu_time_ms or [])
            if model_infer_gpu_time_ms is not None
            else None
        ),
        pipeline_step_latency_ms=(
            summarize_latency_ms(pipeline_step_latency_ms or [])
            if pipeline_step_latency_ms is not None
            else None
        ),
    )
