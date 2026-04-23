"""Unit tests for rollout_eval benchmark dual-process pipeline runner."""

from __future__ import annotations

import time
from queue import Empty, Queue
from threading import Event

import pytest

import toolkits.rollout_eval.benchmark.pipeline_runner as pipeline_runner_module
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


def test_sim_worker_main_applies_cpu_affinity_before_sim_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_order: list[tuple[str, tuple[int, ...] | None]] = []
    monkeypatch.setattr(
        pipeline_runner_module,
        "apply_cpu_affinity",
        lambda cpus: call_order.append(("affinity", tuple(cpus))),
        raising=False,
    )

    def _affinity_aware_sim_factory() -> FakeSimAdapter:
        call_order.append(("factory", None))
        return FakeSimAdapter()

    obs_queue: Queue = Queue(maxsize=1)
    action_queue: Queue = Queue(maxsize=1)
    result_queue: Queue = Queue()
    stop_event = Event()
    action_queue.put((0, 1))

    pipeline_runner_module._sim_worker_main(
        sim_factory=_affinity_aware_sim_factory,
        total_steps=1,
        warmup_steps=0,
        obs_queue=obs_queue,
        action_queue=action_queue,
        result_queue=result_queue,
        stop_event=stop_event,
        queue_timeout_s=0.1,
        cpu_affinity=(0, 2),
    )

    result = result_queue.get_nowait()
    assert result["status"] == "ok"
    assert call_order == [("affinity", (0, 2)), ("factory", None)]


def test_run_dual_process_pipeline_passes_cpu_affinity_to_sim_process_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processes: list[_FakeProcess] = []

    class _FakeQueue:
        def __init__(self) -> None:
            self._items: list[object] = []

        def put(self, item, timeout: float | None = None) -> None:
            del timeout
            self._items.append(item)

        def get(self, timeout: float | None = None):
            del timeout
            if not self._items:
                raise Empty
            return self._items.pop(0)

    class _FakeEvent:
        def __init__(self) -> None:
            self._is_set = False

        def set(self) -> None:
            self._is_set = True

        def is_set(self) -> bool:
            return self._is_set

    class _FakeProcess:
        def __init__(self, *, target, kwargs, name: str) -> None:
            del target
            self.kwargs = kwargs
            self.name = name
            processes.append(self)

        def start(self) -> None:
            result_queue = self.kwargs["result_queue"]
            if self.name == "rollout_eval_sim_worker":
                result_queue.put(
                    {
                        "worker": "sim",
                        "status": "ok",
                        "env_step_count": 2,
                        "env_step_seconds": 1.0,
                        "pipeline_sample_count": 2,
                        "pipeline_seconds": 1.0,
                        "env_step_latency_ms": [1.0, 1.0],
                        "pipeline_step_latency_ms": [2.0, 2.0],
                    }
                )
                return
            result_queue.put(
                {
                    "worker": "model",
                    "status": "ok",
                    "model_infer_count": 2,
                    "model_infer_seconds": 1.0,
                    "model_infer_latency_ms": [1.0, 1.0],
                    "model_infer_gpu_time_ms": None,
                }
            )

        def join(self, timeout: float | None = None) -> None:
            del timeout

        def is_alive(self) -> bool:
            return False

        def terminate(self) -> None:
            return None

    class _FakeContext:
        def Queue(self, maxsize: int | None = None) -> _FakeQueue:
            del maxsize
            return _FakeQueue()

        def Event(self) -> _FakeEvent:
            return _FakeEvent()

        def Process(self, *, target, kwargs, name: str) -> _FakeProcess:
            return _FakeProcess(target=target, kwargs=kwargs, name=name)

    monkeypatch.setattr(
        pipeline_runner_module.mp,
        "get_context",
        lambda _start_method: _FakeContext(),
    )

    metrics = run_dual_process_pipeline(
        sim_factory=_sim_factory,
        model_factory=_model_factory,
        config=PipelineRunnerConfig(
            warmup_steps=0,
            measure_steps=2,
            queue_timeout_s=0.5,
            run_timeout_s=5.0,
            start_method=_preferred_start_method(),
            sim_cpu_affinity=(0, 2),
        ),
    )

    assert metrics.pipeline_samples_per_sec > 0.0
    sim_process = next(
        process for process in processes if process.name == "rollout_eval_sim_worker"
    )
    model_process = next(
        process for process in processes if process.name == "rollout_eval_model_worker"
    )
    assert sim_process.kwargs["cpu_affinity"] == (0, 2)
    assert "cpu_affinity" not in model_process.kwargs
