"""Latency and resource collection utilities."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

from toolkits.rollout_eval.rollout_types import LatencyStats


@dataclass
class _TimerToken:
    name: str
    start: float


class LatencyCollector:
    """Collects per-stage latency metrics for rollout eval."""

    def __init__(self) -> None:
        self._stats = LatencyStats()

    @contextmanager
    def timed(self, name: str):
        token = _TimerToken(name=name, start=time.perf_counter())
        try:
            yield
        finally:
            elapsed = time.perf_counter() - token.start
            if token.name == "model_infer":
                self._stats.model_infer_seconds += elapsed
                self._stats.model_infer_count += 1
            elif token.name == "env_step":
                self._stats.env_step_seconds += elapsed
                self._stats.env_step_count += 1

    @property
    def stats(self) -> LatencyStats:
        return self._stats
