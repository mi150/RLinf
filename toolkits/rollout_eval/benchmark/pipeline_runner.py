"""Dual-process pipeline runner for rollout_eval benchmark cases."""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from queue import Empty, Full
from typing import Any, Callable, Literal, Protocol

import torch

from toolkits.rollout_eval.benchmark.metrics import aggregate_case_metrics
from toolkits.rollout_eval.benchmark.resource_binding import apply_cpu_affinity
from toolkits.rollout_eval.benchmark.types import CaseMetrics


class SimAdapter(Protocol):
    """Minimal simulation adapter interface used by the pipeline runner."""

    def reset(self) -> Any:
        """Reset environment and return initial observation payload."""

    def step(self, action: Any) -> Any:
        """Apply an action and return the next observation payload."""


class ModelAdapter(Protocol):
    """Minimal model adapter interface used by the pipeline runner."""

    def infer(self, observation: Any) -> Any:
        """Infer action from observation payload."""


@dataclass(frozen=True)
class PipelineRunnerConfig:
    """Runtime controls for dual-process pipeline execution."""

    warmup_steps: int
    measure_steps: int
    queue_size: int = 1
    queue_timeout_s: float = 5.0
    join_timeout_s: float = 3.0
    run_timeout_s: float | None = None
    start_method: Literal["spawn", "fork", "forkserver"] = "spawn"
    sim_cpu_affinity: tuple[int, ...] | None = None


def _validate_config(config: PipelineRunnerConfig) -> None:
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if config.measure_steps <= 0:
        raise ValueError("measure_steps must be > 0")
    if config.queue_size <= 0:
        raise ValueError("queue_size must be > 0")
    if config.queue_timeout_s <= 0:
        raise ValueError("queue_timeout_s must be > 0")
    if config.join_timeout_s <= 0:
        raise ValueError("join_timeout_s must be > 0")
    if config.run_timeout_s is not None and config.run_timeout_s <= 0:
        raise ValueError("run_timeout_s must be > 0 when provided")


def _sim_worker_main(
    *,
    sim_factory: Callable[[], SimAdapter],
    total_steps: int,
    warmup_steps: int,
    obs_queue: mp.Queue,
    action_queue: mp.Queue,
    result_queue: mp.Queue,
    stop_event: mp.Event,
    queue_timeout_s: float,
    cpu_affinity: tuple[int, ...] | None = None,
) -> None:
    try:
        if cpu_affinity:
            apply_cpu_affinity(cpu_affinity)
        sim = sim_factory()
        observation = sim.reset()

        env_latencies_ms: list[float] = []
        pipeline_latencies_ms: list[float] = []
        env_measure_start: float | None = None
        env_measure_end: float | None = None

        for step_idx in range(total_steps):
            if stop_event.is_set():
                break

            sent_ts = time.perf_counter()
            try:
                obs_queue.put((step_idx, observation, sent_ts), timeout=queue_timeout_s)
            except Full:
                raise TimeoutError("sim worker timed out while writing observation queue")

            try:
                action_step_idx, action = action_queue.get(timeout=queue_timeout_s)
            except Empty:
                raise TimeoutError("sim worker timed out while waiting action queue")

            if action_step_idx != step_idx:
                raise RuntimeError(
                    f"step index mismatch: expected {step_idx}, got {action_step_idx}"
                )

            received_ts = time.perf_counter()
            if step_idx >= warmup_steps:
                if env_measure_start is None:
                    env_measure_start = received_ts
                pipeline_latencies_ms.append((received_ts - sent_ts) * 1000.0)

            env_step_begin = time.perf_counter()
            observation = sim.step(action)
            env_step_end = time.perf_counter()

            if step_idx >= warmup_steps:
                env_latencies_ms.append((env_step_end - env_step_begin) * 1000.0)
                env_measure_end = env_step_end

        env_steps = max(0, len(env_latencies_ms))
        env_seconds = (
            max(0.0, (env_measure_end or 0.0) - (env_measure_start or 0.0))
            if env_measure_start is not None and env_measure_end is not None
            else 0.0
        )
        result_queue.put(
            {
                "worker": "sim",
                "status": "ok",
                "env_step_count": env_steps,
                "env_step_seconds": env_seconds,
                "pipeline_sample_count": len(pipeline_latencies_ms),
                "pipeline_seconds": env_seconds,
                "env_step_latency_ms": env_latencies_ms,
                "pipeline_step_latency_ms": pipeline_latencies_ms,
            }
        )
    except Exception as exc:  # pragma: no cover - validated by parent behavior tests
        stop_event.set()
        result_queue.put({"worker": "sim", "status": "error", "error": str(exc)})


def _model_worker_main(
    *,
    model_factory: Callable[[], ModelAdapter],
    total_steps: int,
    warmup_steps: int,
    obs_queue: mp.Queue,
    action_queue: mp.Queue,
    result_queue: mp.Queue,
    stop_event: mp.Event,
    queue_timeout_s: float,
) -> None:
    try:
        model = model_factory()

        model_latencies_ms: list[float] = []
        model_gpu_latencies_ms: list[float] = []
        model_measure_start: float | None = None
        model_measure_end: float | None = None

        for _ in range(total_steps):
            if stop_event.is_set():
                break

            try:
                step_idx, observation, _sent_ts = obs_queue.get(timeout=queue_timeout_s)
            except Empty:
                raise TimeoutError("model worker timed out while waiting observation queue")

            start_event = end_event = None
            should_measure_step = step_idx >= warmup_steps
            if torch.cuda.is_available() and should_measure_step:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            infer_begin = time.perf_counter()
            action = model.infer(observation)
            infer_end = time.perf_counter()
            if start_event is not None and end_event is not None:
                end_event.record()
                torch.cuda.synchronize()

            try:
                action_queue.put((step_idx, action), timeout=queue_timeout_s)
            except Full:
                raise TimeoutError("model worker timed out while writing action queue")

            if should_measure_step:
                if model_measure_start is None:
                    model_measure_start = infer_begin
                model_latencies_ms.append((infer_end - infer_begin) * 1000.0)
                if start_event is not None and end_event is not None:
                    model_gpu_latencies_ms.append(float(start_event.elapsed_time(end_event)))
                model_measure_end = infer_end

        model_infers = max(0, len(model_latencies_ms))
        model_seconds = (
            max(0.0, (model_measure_end or 0.0) - (model_measure_start or 0.0))
            if model_measure_start is not None and model_measure_end is not None
            else 0.0
        )
        result_queue.put(
            {
                "worker": "model",
                "status": "ok",
                "model_infer_count": model_infers,
                "model_infer_seconds": model_seconds,
                "model_infer_latency_ms": model_latencies_ms,
                "model_infer_gpu_time_ms": model_gpu_latencies_ms if torch.cuda.is_available() else None,
            }
        )
    except Exception as exc:  # pragma: no cover - validated by parent behavior tests
        stop_event.set()
        result_queue.put({"worker": "model", "status": "error", "error": str(exc)})


def _terminate_if_alive(process: mp.Process, timeout_s: float) -> None:
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(timeout_s)


def _default_run_timeout(config: PipelineRunnerConfig) -> float:
    total_steps = config.warmup_steps + config.measure_steps
    return max(5.0, (total_steps * config.queue_timeout_s * 4.0) + 2.0)


def run_dual_process_pipeline(
    *,
    sim_factory: Callable[[], SimAdapter],
    model_factory: Callable[[], ModelAdapter],
    config: PipelineRunnerConfig,
) -> CaseMetrics:
    """Run a sim+model dual-process pipeline and aggregate throughput/latency metrics."""
    _validate_config(config)
    total_steps = config.warmup_steps + config.measure_steps

    context = mp.get_context(config.start_method)
    obs_queue: mp.Queue = context.Queue(maxsize=config.queue_size)
    action_queue: mp.Queue = context.Queue(maxsize=config.queue_size)
    result_queue: mp.Queue = context.Queue()
    stop_event: mp.Event = context.Event()

    sim_process = context.Process(
        target=_sim_worker_main,
        kwargs={
            "sim_factory": sim_factory,
            "total_steps": total_steps,
            "warmup_steps": config.warmup_steps,
            "obs_queue": obs_queue,
            "action_queue": action_queue,
            "result_queue": result_queue,
            "stop_event": stop_event,
            "queue_timeout_s": config.queue_timeout_s,
            "cpu_affinity": config.sim_cpu_affinity,
        },
        name="rollout_eval_sim_worker",
    )
    model_process = context.Process(
        target=_model_worker_main,
        kwargs={
            "model_factory": model_factory,
            "total_steps": total_steps,
            "warmup_steps": config.warmup_steps,
            "obs_queue": obs_queue,
            "action_queue": action_queue,
            "result_queue": result_queue,
            "stop_event": stop_event,
            "queue_timeout_s": config.queue_timeout_s,
        },
        name="rollout_eval_model_worker",
    )

    sim_process.start()
    model_process.start()

    results: dict[str, dict[str, Any]] = {}
    deadline = time.perf_counter() + (config.run_timeout_s or _default_run_timeout(config))

    try:
        while len(results) < 2 and time.perf_counter() < deadline:
            try:
                message = result_queue.get(timeout=0.1)
            except Empty:
                continue
            worker_name = str(message.get("worker", "unknown"))
            results[worker_name] = message

        if len(results) < 2:
            stop_event.set()
            raise TimeoutError("pipeline runner timed out before receiving worker results")

        for worker_name in ("sim", "model"):
            status = str(results[worker_name].get("status"))
            if status != "ok":
                raise RuntimeError(
                    f"{worker_name} worker failed: {results[worker_name].get('error', 'unknown error')}"
                )

        return aggregate_case_metrics(
            env_step_count=int(results["sim"]["env_step_count"]),
            env_step_seconds=float(results["sim"]["env_step_seconds"]),
            model_infer_count=int(results["model"]["model_infer_count"]),
            model_infer_seconds=float(results["model"]["model_infer_seconds"]),
            pipeline_sample_count=int(results["sim"]["pipeline_sample_count"]),
            pipeline_seconds=float(results["sim"]["pipeline_seconds"]),
            env_step_latency_ms=list(results["sim"]["env_step_latency_ms"]),
            model_infer_latency_ms=list(results["model"]["model_infer_latency_ms"]),
            model_infer_gpu_time_ms=(
                list(results["model"]["model_infer_gpu_time_ms"])
                if results["model"].get("model_infer_gpu_time_ms") is not None
                else None
            ),
            pipeline_step_latency_ms=list(results["sim"]["pipeline_step_latency_ms"]),
        )
    finally:
        stop_event.set()
        _terminate_if_alive(sim_process, config.join_timeout_s)
        _terminate_if_alive(model_process, config.join_timeout_s)
