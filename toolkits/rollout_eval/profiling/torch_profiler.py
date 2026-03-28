"""Torch profiler utilities for rollout eval."""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

TARGET_SPLIT_MODELS = {"openvla_oft", "openpi", "gr00t"}


class RolloutTorchProfiler:
    """Thin wrapper over ``torch.profiler.profile`` with stage aggregation."""

    def __init__(
        self,
        enabled: bool,
        output_dir: str,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
    ) -> None:
        self.enabled = enabled
        self.output_dir = output_dir
        self._ctx = None
        self._prof = None

        if not self.enabled:
            return

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self._prof = profile(
            activities=activities,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
        )

    def __enter__(self):
        if self.enabled and self._prof is not None:
            self._ctx = self._prof.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self._prof is not None:
            self._prof.__exit__(exc_type, exc, tb)
            trace_dir = Path(self.output_dir)
            chrome_trace_path = trace_dir / "rollout_trace.json"
            tb_trace_path = trace_dir / f"rollout_eval.{int(time.time() * 1000)}.pt.trace.json"
            try:
                self._prof.export_chrome_trace(str(chrome_trace_path))
                # TensorBoard torch_tb_profiler expects '<worker>.<timestamp>.pt.trace.json'.
                tb_trace_path.write_bytes(chrome_trace_path.read_bytes())
            except Exception:
                pass

    def step(self) -> None:
        if self.enabled and self._prof is not None:
            self._prof.step()

    def trace(self, name: str):
        if not self.enabled:
            return _NullContext()
        return torch.profiler.record_function(name)

    def key_averages(self):
        if not self.enabled or self._prof is None:
            return []
        return self._prof.key_averages()


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _event_cpu_us(event: Any) -> float:
    total = float(getattr(event, "cpu_time_total", 0.0))
    if total > 0:
        return total
    return float(getattr(event, "self_cpu_time_total", 0.0))


def _event_cuda_us(event: Any) -> float:
    total = float(getattr(event, "cuda_time_total", 0.0))
    if total > 0:
        return total
    total = float(getattr(event, "device_time_total", 0.0))
    if total > 0:
        return total
    total = float(getattr(event, "self_cuda_time_total", 0.0))
    if total > 0:
        return total
    return float(getattr(event, "self_device_time_total", 0.0))


def aggregate_profile_metrics(events: Iterable[Any]) -> dict[str, float]:
    """Aggregate profile events into env/model/backbone/action_head metrics."""
    metrics = {
        "env_sim_cpu_us": 0.0,
        "model_infer_cpu_us": 0.0,
        "model_backbone_cpu_us": 0.0,
        "model_action_head_cpu_us": 0.0,
        "env_sim_cuda_us": 0.0,
        "model_infer_cuda_us": 0.0,
        "model_backbone_cuda_us": 0.0,
        "model_action_head_cuda_us": 0.0,
    }

    for event in events:
        key = getattr(event, "key", "")
        cpu_t = _event_cpu_us(event)
        cuda_t = _event_cuda_us(event)

        if "env.simulation" in key:
            metrics["env_sim_cpu_us"] += cpu_t
            metrics["env_sim_cuda_us"] += cuda_t
        if "model.inference" in key:
            metrics["model_infer_cpu_us"] += cpu_t
            metrics["model_infer_cuda_us"] += cuda_t
        if "model.backbone" in key:
            metrics["model_backbone_cpu_us"] += cpu_t
            metrics["model_backbone_cuda_us"] += cuda_t
        if "model.action_head" in key:
            metrics["model_action_head_cpu_us"] += cpu_t
            metrics["model_action_head_cuda_us"] += cuda_t

    return metrics
