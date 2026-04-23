"""Type definitions for rollout_eval benchmark matrix and metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvModelPreset:
    """Named env/model preset used by benchmark scenario expansion."""

    name: str
    env_type: str
    model_type: str


@dataclass(frozen=True)
class BenchmarkRequest:
    """Top-level benchmark request parsed from CLI."""

    config_path: str
    config_name: str
    override: tuple[str, ...]
    output_dir: str
    scenario_set: tuple[str, ...]
    pipeline: str
    mps_sm: tuple[int, ...]
    mig_devices: tuple[str, ...]
    presets: tuple[EnvModelPreset, ...]
    model_only_input: str
    env_only_action: str
    warmup_steps: int
    measure_steps: int
    num_envs_override: int | None = None
    pipeline_queue_timeout_s: float = 5.0
    pipeline_run_timeout_s: float | None = None
    cpu_bind_cores: str | None = None
    cpu_bind_strategy: str | None = None
    cpu_bind_config: str | None = None
    cpu_bind_strict: bool = True


@dataclass(frozen=True)
class BenchmarkCase:
    """One expanded benchmark case with fixed resource binding."""

    case_id: str
    scenario: str
    preset_name: str
    env_type: str
    model_type: str
    mps_sm: int | None = None
    mig_device: str | None = None
    cpu_binding_mode: str | None = None
    cpu_available_cores: tuple[int, ...] | None = None
    cpu_env_core_groups: tuple[tuple[int, ...], ...] | None = None


@dataclass(frozen=True)
class LatencySummary:
    """Latency summary with mean and key percentiles in milliseconds."""

    avg_ms: float
    p50_ms: float
    p95_ms: float


@dataclass(frozen=True)
class CaseMetrics:
    """Aggregated throughput and latency outputs for a benchmark case."""

    env_steps_per_sec: float = 0.0
    model_infers_per_sec: float = 0.0
    pipeline_samples_per_sec: float = 0.0
    env_step_latency_ms: LatencySummary | None = None
    model_infer_latency_ms: LatencySummary | None = None
    model_infer_gpu_time_ms: LatencySummary | None = None
    pipeline_step_latency_ms: LatencySummary | None = None
