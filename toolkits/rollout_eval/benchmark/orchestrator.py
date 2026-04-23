"""Benchmark orchestrator for rollout_eval MPS/MIG scenario matrix."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from queue import Empty
from typing import Callable

from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from rlinf.config import validate_cfg
from toolkits.rollout_eval.adapters import build_env_adapter, build_model_adapter
from toolkits.rollout_eval.benchmark.pipeline_runner import (
    PipelineRunnerConfig,
    run_dual_process_pipeline,
)
from toolkits.rollout_eval.benchmark.reporting import (
    write_case_report,
    write_summary_reports,
)
from toolkits.rollout_eval.benchmark.resource_binding import (
    apply_cpu_affinity,
    build_even_split_cpu_groups,
    build_process_env,
    effective_process_affinity,
    load_cpu_groups_from_yaml,
)
from toolkits.rollout_eval.benchmark.scenarios import expand_cases
from toolkits.rollout_eval.benchmark.single_runner import (
    run_env_only_case,
    run_model_only_case,
)
from toolkits.rollout_eval.benchmark.types import (
    BenchmarkCase,
    BenchmarkRequest,
    CaseMetrics,
    LatencySummary,
)


class SkipCase(RuntimeError):
    """Raised when a benchmark case should be marked skipped."""


def _resolve_cpu_groups(
    request: BenchmarkRequest,
    case: BenchmarkCase,
    env_count: int,
) -> tuple[tuple[int, ...], ...]:
    if case.scenario not in {"env_only_cpu_core", "concurrent_cpu_core"}:
        return ()
    if case.cpu_binding_mode == "none":
        return ()
    if case.cpu_env_core_groups:
        return case.cpu_env_core_groups
    if request.cpu_bind_config:
        return load_cpu_groups_from_yaml(request.cpu_bind_config, env_count)
    if not case.cpu_available_cores:
        raise SkipCase("cpu_core scenario requires --cpu-bind-cores or --cpu-bind-config")
    return build_even_split_cpu_groups(case.cpu_available_cores, env_count)


def _prepare_case_cpu_groups(
    request: BenchmarkRequest,
    case: BenchmarkCase,
) -> BenchmarkCase:
    if case.scenario not in {"env_only_cpu_core", "concurrent_cpu_core"}:
        return case
    if case.cpu_binding_mode == "none":
        return case
    if case.cpu_env_core_groups:
        return case

    cfg = _load_cfg_for_case(request, case)
    cpu_groups = _resolve_cpu_groups(request, case, int(cfg.env.eval.total_num_envs))
    return replace(case, cpu_env_core_groups=cpu_groups)


def _ensure_cpu_affinity_supported(case: BenchmarkCase) -> None:
    if (
        case.scenario.endswith("_cpu_core")
        and case.cpu_binding_mode != "none"
        and not hasattr(os, "sched_setaffinity")
    ):
        raise SkipCase("CPU affinity is unavailable on this platform")


def _load_cfg_for_case(request: BenchmarkRequest, case: BenchmarkCase):
    abs_config_path = str(Path(request.config_path).resolve())
    with initialize_config_dir(version_base="1.1", config_dir=abs_config_path):
        cfg = compose(config_name=request.config_name, overrides=list(request.override))

    with open_dict(cfg):
        if "env" in cfg:
            if "train" in cfg.env:
                cfg.env.train.env_type = case.env_type
            if "eval" in cfg.env:
                cfg.env.eval.env_type = case.env_type

        if "model" in cfg and "model_type" in cfg.model:
            cfg.model.model_type = case.model_type

        if "actor" in cfg and "model" in cfg.actor and "model_type" in cfg.actor.model:
            cfg.actor.model.model_type = case.model_type

        if "rollout" in cfg and "model" in cfg.rollout and "model_type" in cfg.rollout.model:
            cfg.rollout.model.model_type = case.model_type
        if request.num_envs_override is not None:
            cfg.env.eval.total_num_envs = int(request.num_envs_override)

    try:
        return validate_cfg(cfg)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to validate cfg for env '{case.env_type}' and model '{case.model_type}': {exc}"
        ) from exc


def _build_case_env(case: BenchmarkCase) -> dict[str, str]:
    mps_percentage = case.mps_sm if case.scenario.endswith("_mps") else None
    mig_uuid = case.mig_device if case.scenario.endswith("_mig") else None

    merged = build_process_env(
        base_env=os.environ,
        mig_device_uuid=mig_uuid,
        mps_active_thread_percentage=mps_percentage,
    )

    updates: dict[str, str] = {}
    for key, value in merged.items():
        current = os.environ.get(key)
        if current != value:
            updates[key] = value
    return updates


@contextmanager
def _temporary_environ(updates: dict[str, str]):
    if not updates:
        yield
        return

    previous: dict[str, str | None] = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


class _PipelineSimAdapter:
    """Adapter exposing reset/step APIs expected by pipeline_runner."""

    def __init__(self, env_adapter) -> None:
        self._env_adapter = env_adapter

    def reset(self):
        obs_batch, _meta = self._env_adapter.reset()
        return obs_batch

    def step(self, action):
        step_result = self._env_adapter.step(action)
        return step_result.obs


class _PipelineModelAdapter:
    """Adapter exposing infer(observation) API expected by pipeline_runner."""

    def __init__(self, model_adapter) -> None:
        self._model_adapter = model_adapter

    def infer(self, observation):
        action, _meta = self._model_adapter.infer(obs_batch=observation, mode="eval")
        return action


def _execute_case(request: BenchmarkRequest, case: BenchmarkCase) -> CaseMetrics:
    if case.scenario not in {
        "concurrent_mps",
        "concurrent_mig",
        "concurrent_cpu_core",
        "env_only_mps",
        "env_only_cpu_core",
        "model_only_mps",
        "env_only_mig",
        "model_only_mig",
    }:
        raise SkipCase(f"Unsupported scenario: {case.scenario}")

    case_env_updates = _build_case_env(case)
    _ensure_cpu_affinity_supported(case)

    with _temporary_environ(case_env_updates):
        if case.scenario.startswith("env_only_"):
            cfg = _load_cfg_for_case(request, case)
            cpu_groups = _resolve_cpu_groups(
                request, case, int(cfg.env.eval.total_num_envs)
            )
            if case.scenario == "env_only_cpu_core" and cpu_groups:
                apply_cpu_affinity(effective_process_affinity(cpu_groups))
            env_adapter = build_env_adapter(cfg, split="eval", profile_output_dir=None)
            action_dim_override = None
            try:
                action_dim_override = int(cfg.actor.model.get("action_dim", 0))
            except Exception:
                action_dim_override = None
            return run_env_only_case(
                env_adapter=env_adapter,
                warmup_steps=request.warmup_steps,
                measure_steps=request.measure_steps,
                action_dim_override=action_dim_override,
            ).metrics

        if case.scenario.startswith("model_only_"):
            cfg = _load_cfg_for_case(request, case)
            env_adapter = build_env_adapter(cfg, split="eval", profile_output_dir=None)
            model_adapter = build_model_adapter(cfg, split_model_stages=False)
            return run_model_only_case(
                env_adapter=env_adapter,
                model_adapter=model_adapter,
                warmup_steps=request.warmup_steps,
                measure_steps=request.measure_steps,
            ).metrics

        if case.scenario.startswith("concurrent_"):
            start_method = "fork" if os.name != "nt" else "spawn"
            cfg = None
            sim_cpu_affinity = None
            if case.scenario == "concurrent_cpu_core":
                cfg = _load_cfg_for_case(request, case)
                cpu_groups = _resolve_cpu_groups(
                    request, case, int(cfg.env.eval.total_num_envs)
                )
                if cpu_groups:
                    sim_cpu_affinity = effective_process_affinity(cpu_groups)

            def _sim_factory():
                local_cfg = cfg or _load_cfg_for_case(request, case)
                env_adapter = build_env_adapter(
                    local_cfg, split="eval", profile_output_dir=None
                )
                return _PipelineSimAdapter(env_adapter)

            def _model_factory():
                local_cfg = cfg or _load_cfg_for_case(request, case)
                model_adapter = build_model_adapter(
                    local_cfg, split_model_stages=False
                )
                return _PipelineModelAdapter(model_adapter)

            return run_dual_process_pipeline(
                sim_factory=_sim_factory,
                model_factory=_model_factory,
                config=PipelineRunnerConfig(
                    warmup_steps=request.warmup_steps,
                    measure_steps=request.measure_steps,
                    queue_timeout_s=request.pipeline_queue_timeout_s,
                    run_timeout_s=request.pipeline_run_timeout_s,
                    start_method=start_method,
                    sim_cpu_affinity=sim_cpu_affinity,
                ),
            )

    raise SkipCase(f"No execution path for scenario: {case.scenario}")


def _metrics_to_dict(metrics: CaseMetrics) -> dict:
    return asdict(metrics)


def _metrics_from_dict(payload: dict) -> CaseMetrics:
    def _latency(value):
        if value is None:
            return None
        return LatencySummary(**value)

    return CaseMetrics(
        env_steps_per_sec=float(payload.get("env_steps_per_sec", 0.0)),
        model_infers_per_sec=float(payload.get("model_infers_per_sec", 0.0)),
        pipeline_samples_per_sec=float(payload.get("pipeline_samples_per_sec", 0.0)),
        env_step_latency_ms=_latency(payload.get("env_step_latency_ms")),
        model_infer_latency_ms=_latency(payload.get("model_infer_latency_ms")),
        model_infer_gpu_time_ms=_latency(payload.get("model_infer_gpu_time_ms")),
        pipeline_step_latency_ms=_latency(payload.get("pipeline_step_latency_ms")),
    )


def _run_case_worker(
    result_queue: mp.Queue,
    request: BenchmarkRequest,
    case: BenchmarkCase,
) -> None:
    try:
        metrics = _execute_case(request, case)
        result_queue.put({"status": "pass", "metrics": _metrics_to_dict(metrics)})
    except SkipCase as exc:
        result_queue.put({"status": "skipped", "skip_reason": str(exc)})
    except Exception as exc:  # noqa: BLE001
        error_type = type(exc).__name__
        error_message = str(exc) or repr(exc)
        result_queue.put(
            {
                "status": "failed",
                "error_message": (
                    f"{error_type}: {error_message}\n"
                    f"{traceback.format_exc()}"
                ),
            }
        )


def _default_case_timeout_s(request: BenchmarkRequest) -> float:
    total_steps = request.warmup_steps + request.measure_steps
    return max(
        30.0,
        (total_steps * max(1.0, request.pipeline_queue_timeout_s) * 8.0) + 10.0,
    )


def _execute_case_in_subprocess(request: BenchmarkRequest, case: BenchmarkCase) -> dict:
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_run_case_worker,
        kwargs={
            "result_queue": result_queue,
            "request": request,
            "case": case,
        },
        name=f"rollout_eval_case_{case.case_id}",
    )

    process.start()

    timeout_s = _default_case_timeout_s(request)
    deadline = time.perf_counter() + timeout_s
    message = None

    try:
        while time.perf_counter() < deadline:
            timeout = min(0.1, max(0.0, deadline - time.perf_counter()))
            try:
                message = result_queue.get(timeout=timeout)
                break
            except Empty:
                if not process.is_alive():
                    break
                continue
    finally:
        process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)

    if message is None:
        try:
            message = result_queue.get(timeout=0.2)
        except Empty:
            message = None

    if message is None:
        if process.exitcode not in (0, None):
            return {
                "status": "failed",
                "error_message": f"case subprocess exited with code {process.exitcode}",
            }
        return {
            "status": "failed",
            "error_message": f"case subprocess timed out after {timeout_s:.1f}s",
        }
    return message


def run_benchmark_orchestrator(
    request: BenchmarkRequest,
    *,
    case_executor: Callable[[BenchmarkRequest, BenchmarkCase], CaseMetrics] | None = None,
) -> dict:
    """Run full benchmark matrix; failures are isolated at case level."""
    executor = case_executor or _execute_case
    cases = expand_cases(request)

    output_dir = Path(request.output_dir)
    case_records: list[dict] = []

    for raw_case in cases:
        case = raw_case
        process_env = _build_case_env(raw_case)
        status = "pass"
        metrics = None
        error_message = None
        skip_reason = None

        try:
            case = _prepare_case_cpu_groups(request, raw_case)
            process_env = _build_case_env(case)

            if case_executor is None:
                result = _execute_case_in_subprocess(request, case)
                status = str(result.get("status", "failed"))
                if status == "pass":
                    metrics_payload = result.get("metrics")
                    if isinstance(metrics_payload, dict):
                        metrics = _metrics_from_dict(metrics_payload)
                    else:
                        status = "failed"
                        error_message = "missing metrics payload from case subprocess"
                elif status == "skipped":
                    skip_reason = str(result.get("skip_reason", "skipped"))
                else:
                    status = "failed"
                    error_message = str(
                        result.get("error_message", "unknown case subprocess error")
                    )
            else:
                metrics = executor(request, case)
        except SkipCase as exc:
            status = "skipped"
            skip_reason = str(exc)
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error_message = str(exc)

        record = write_case_report(
            output_dir=output_dir,
            request=request,
            case=case,
            status=status,
            metrics=metrics,
            process_env=process_env,
            error_message=error_message,
            skip_reason=skip_reason,
        )
        case_records.append(record)

    return write_summary_reports(output_dir=output_dir, case_records=case_records)
