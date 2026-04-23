from __future__ import annotations

import json
from queue import Empty
from types import SimpleNamespace

import pytest

import toolkits.rollout_eval.benchmark.orchestrator as orchestrator_module
from toolkits.rollout_eval.benchmark.orchestrator import (
    SkipCase,
    run_benchmark_orchestrator,
)
from toolkits.rollout_eval.benchmark.types import (
    BenchmarkCase,
    BenchmarkRequest,
    CaseMetrics,
    EnvModelPreset,
)


def _make_request(output_dir: str) -> BenchmarkRequest:
    return BenchmarkRequest(
        config_path="examples/embodiment/config",
        config_name="dummy",
        override=(),
        output_dir=output_dir,
        scenario_set=("concurrent_mps",),
        pipeline="process",
        mps_sm=(20,),
        mig_devices=(),
        presets=(EnvModelPreset("p", "maniskill", "openvla_oft"),),
        model_only_input="dummy_from_env_reset",
        env_only_action="random",
        warmup_steps=1,
        measure_steps=2,
    )


def test_orchestrator_isolates_case_failures_and_continues(tmp_path, monkeypatch) -> None:
    cases = [
        BenchmarkCase(
            case_id="case-pass",
            scenario="env_only_mps",
            preset_name="p",
            env_type="maniskill",
            model_type="openvla_oft",
            mps_sm=20,
        ),
        BenchmarkCase(
            case_id="case-fail",
            scenario="model_only_mps",
            preset_name="p",
            env_type="maniskill",
            model_type="openvla_oft",
            mps_sm=20,
        ),
        BenchmarkCase(
            case_id="case-skip",
            scenario="concurrent_mig",
            preset_name="p",
            env_type="maniskill",
            model_type="openvla_oft",
            mig_device="MIG-A",
        ),
    ]

    monkeypatch.setattr(
        "toolkits.rollout_eval.benchmark.orchestrator.expand_cases",
        lambda _request: list(cases),
    )

    visited: list[str] = []

    def _executor(request: BenchmarkRequest, case: BenchmarkCase) -> CaseMetrics:
        del request
        visited.append(case.case_id)
        if case.case_id == "case-fail":
            raise RuntimeError("boom")
        if case.case_id == "case-skip":
            raise SkipCase("resource unavailable")
        return CaseMetrics(
            env_steps_per_sec=10.0,
            model_infers_per_sec=0.0,
            pipeline_samples_per_sec=0.0,
        )

    request = _make_request(str(tmp_path))
    summary = run_benchmark_orchestrator(request, case_executor=_executor)

    assert visited == ["case-pass", "case-fail", "case-skip"]
    assert summary["counts"] == {"total": 3, "pass": 1, "failed": 1, "skipped": 1}

    pass_report = tmp_path / "env_only_mps" / "case-pass" / "case_report.json"
    fail_report = tmp_path / "model_only_mps" / "case-fail" / "case_report.json"
    skip_report = tmp_path / "concurrent_mig" / "case-skip" / "case_report.json"

    assert pass_report.exists()
    assert fail_report.exists()
    assert skip_report.exists()
    assert json.loads(fail_report.read_text(encoding="utf-8"))["status"] == "failed"
    assert json.loads(skip_report.read_text(encoding="utf-8"))["status"] == "skipped"


def test_orchestrator_writes_summary_json_and_markdown(tmp_path, monkeypatch) -> None:
    case = BenchmarkCase(
        case_id="single-case",
        scenario="env_only_mps",
        preset_name="p",
        env_type="maniskill",
        model_type="openvla_oft",
        mps_sm=40,
    )
    monkeypatch.setattr(
        "toolkits.rollout_eval.benchmark.orchestrator.expand_cases",
        lambda _request: [case],
    )

    def _executor(request: BenchmarkRequest, run_case: BenchmarkCase) -> CaseMetrics:
        del request, run_case
        return CaseMetrics(
            env_steps_per_sec=12.5,
            model_infers_per_sec=0.0,
            pipeline_samples_per_sec=0.0,
        )

    request = _make_request(str(tmp_path))
    summary = run_benchmark_orchestrator(request, case_executor=_executor)

    assert summary["counts"] == {"total": 1, "pass": 1, "failed": 0, "skipped": 0}
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.md").exists()


def test_execute_case_env_only_cpu_core_applies_affinity_before_env_adapter(
    tmp_path, monkeypatch
) -> None:
    case = BenchmarkCase(
        case_id="cpu-env-only",
        scenario="env_only_cpu_core",
        preset_name="p",
        env_type="maniskill",
        model_type="openvla_oft",
        cpu_binding_mode="even_split",
        cpu_available_cores=(0, 1, 2, 3),
    )
    request = _make_request(str(tmp_path))
    request = BenchmarkRequest(
        **{
            **request.__dict__,
            "scenario_set": ("env_only_cpu_core",),
            "cpu_bind_cores": "0-3",
            "cpu_bind_strategy": "even_split",
            "cpu_bind_config": None,
            "cpu_bind_strict": True,
        }
    )

    cfg = SimpleNamespace(env=SimpleNamespace(eval=SimpleNamespace(total_num_envs=2)))
    monkeypatch.setattr(
        orchestrator_module,
        "_load_cfg_for_case",
        lambda _request, _case: cfg,
    )

    call_order: list[tuple[str, tuple[int, ...] | None]] = []
    monkeypatch.setattr(
        orchestrator_module,
        "apply_cpu_affinity",
        lambda cpus: call_order.append(("affinity", tuple(cpus))),
        raising=False,
    )

    def _build_env_adapter(_cfg, split: str, profile_output_dir):
        del _cfg, profile_output_dir
        call_order.append(("env_adapter", None))
        assert split == "eval"
        return object()

    monkeypatch.setattr(
        orchestrator_module,
        "build_env_adapter",
        _build_env_adapter,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "run_env_only_case",
        lambda **_kwargs: SimpleNamespace(metrics=CaseMetrics(env_steps_per_sec=1.0)),
    )

    metrics = orchestrator_module._execute_case(request, case)

    assert metrics.env_steps_per_sec == 1.0
    assert call_order == [("affinity", (0, 1, 2, 3)), ("env_adapter", None)]


def test_execute_case_env_only_cpu_core_none_mode_skips_affinity(
    tmp_path, monkeypatch
) -> None:
    case = BenchmarkCase(
        case_id="cpu-env-only-none",
        scenario="env_only_cpu_core",
        preset_name="p",
        env_type="maniskill",
        model_type="openvla_oft",
        cpu_binding_mode="none",
        cpu_available_cores=None,
    )
    request = _make_request(str(tmp_path))
    request = BenchmarkRequest(
        **{
            **request.__dict__,
            "scenario_set": ("env_only_cpu_core",),
            "cpu_bind_cores": None,
            "cpu_bind_strategy": "none",
            "cpu_bind_config": None,
            "cpu_bind_strict": True,
        }
    )

    cfg = SimpleNamespace(env=SimpleNamespace(eval=SimpleNamespace(total_num_envs=2)))
    monkeypatch.setattr(
        orchestrator_module,
        "_load_cfg_for_case",
        lambda _request, _case: cfg,
    )

    call_order: list[str] = []
    monkeypatch.setattr(
        orchestrator_module,
        "apply_cpu_affinity",
        lambda cpus: call_order.append(f"affinity:{cpus}"),
        raising=False,
    )

    def _build_env_adapter(_cfg, split: str, profile_output_dir):
        del _cfg, profile_output_dir
        call_order.append("env_adapter")
        assert split == "eval"
        return object()

    monkeypatch.setattr(
        orchestrator_module,
        "build_env_adapter",
        _build_env_adapter,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "run_env_only_case",
        lambda **_kwargs: SimpleNamespace(metrics=CaseMetrics(env_steps_per_sec=1.0)),
    )

    metrics = orchestrator_module._execute_case(request, case)

    assert metrics.env_steps_per_sec == 1.0
    assert call_order == ["env_adapter"]


def test_execute_case_concurrent_cpu_core_loads_yaml_groups_and_passes_sim_affinity(
    tmp_path, monkeypatch
) -> None:
    case = BenchmarkCase(
        case_id="cpu-concurrent",
        scenario="concurrent_cpu_core",
        preset_name="p",
        env_type="maniskill",
        model_type="openvla_oft",
        cpu_binding_mode="even_split",
        cpu_available_cores=(0, 1, 2, 3, 4, 5),
    )
    request = _make_request(str(tmp_path))
    request = BenchmarkRequest(
        **{
            **request.__dict__,
            "scenario_set": ("concurrent_cpu_core",),
            "num_envs_override": 2,
            "cpu_bind_cores": "0-5",
            "cpu_bind_strategy": "even_split",
            "cpu_bind_config": "/tmp/cpu-groups.yaml",
            "cpu_bind_strict": True,
        }
    )

    cfg = SimpleNamespace(env=SimpleNamespace(eval=SimpleNamespace(total_num_envs=2)))
    monkeypatch.setattr(
        orchestrator_module,
        "_load_cfg_for_case",
        lambda _request, _case: cfg,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "build_even_split_cpu_groups",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("even split should not be used when cpu_bind_config is set")
        ),
        raising=False,
    )

    captured: dict[str, object] = {}

    def _load_cpu_groups(path: str, env_count: int) -> tuple[tuple[int, ...], ...]:
        captured["yaml_path"] = path
        captured["env_count"] = env_count
        return ((0, 1), (4, 5))

    monkeypatch.setattr(
        orchestrator_module,
        "load_cpu_groups_from_yaml",
        _load_cpu_groups,
        raising=False,
    )

    def _run_pipeline(*, sim_factory, model_factory, config):
        del sim_factory, model_factory
        captured["sim_cpu_affinity"] = config.sim_cpu_affinity
        return CaseMetrics(pipeline_samples_per_sec=1.0)

    monkeypatch.setattr(
        orchestrator_module,
        "run_dual_process_pipeline",
        _run_pipeline,
    )

    metrics = orchestrator_module._execute_case(request, case)

    assert metrics.pipeline_samples_per_sec == 1.0
    assert captured == {
        "yaml_path": "/tmp/cpu-groups.yaml",
        "env_count": 2,
        "sim_cpu_affinity": (0, 1, 4, 5),
    }


def test_execute_case_concurrent_cpu_core_none_mode_passes_no_affinity(
    tmp_path, monkeypatch
) -> None:
    case = BenchmarkCase(
        case_id="cpu-concurrent-none",
        scenario="concurrent_cpu_core",
        preset_name="p",
        env_type="maniskill",
        model_type="openvla_oft",
        cpu_binding_mode="none",
        cpu_available_cores=None,
    )
    request = _make_request(str(tmp_path))
    request = BenchmarkRequest(
        **{
            **request.__dict__,
            "scenario_set": ("concurrent_cpu_core",),
            "num_envs_override": 2,
            "cpu_bind_cores": None,
            "cpu_bind_strategy": "none",
            "cpu_bind_config": None,
            "cpu_bind_strict": True,
        }
    )

    cfg = SimpleNamespace(env=SimpleNamespace(eval=SimpleNamespace(total_num_envs=2)))
    monkeypatch.setattr(
        orchestrator_module,
        "_load_cfg_for_case",
        lambda _request, _case: cfg,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "build_even_split_cpu_groups",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("none mode should not build cpu groups")
        ),
        raising=False,
    )

    captured: dict[str, object] = {}

    def _run_pipeline(*, sim_factory, model_factory, config):
        del sim_factory, model_factory
        captured["sim_cpu_affinity"] = config.sim_cpu_affinity
        return CaseMetrics(pipeline_samples_per_sec=1.0)

    monkeypatch.setattr(
        orchestrator_module,
        "run_dual_process_pipeline",
        _run_pipeline,
    )

    metrics = orchestrator_module._execute_case(request, case)

    assert metrics.pipeline_samples_per_sec == 1.0
    assert captured == {"sim_cpu_affinity": None}


def test_execute_case_concurrent_cpu_core_uses_even_split_with_final_env_count(
    tmp_path, monkeypatch
) -> None:
    case = BenchmarkCase(
        case_id="cpu-concurrent-even-split",
        scenario="concurrent_cpu_core",
        preset_name="p",
        env_type="maniskill",
        model_type="openvla_oft",
        cpu_binding_mode="even_split",
        cpu_available_cores=(0, 1, 2, 3, 4, 5),
    )
    request = _make_request(str(tmp_path))
    request = BenchmarkRequest(
        **{
            **request.__dict__,
            "scenario_set": ("concurrent_cpu_core",),
            "num_envs_override": 3,
            "cpu_bind_cores": "0-5",
            "cpu_bind_strategy": "even_split",
            "cpu_bind_config": None,
            "cpu_bind_strict": True,
        }
    )

    cfg = SimpleNamespace(env=SimpleNamespace(eval=SimpleNamespace(total_num_envs=3)))
    monkeypatch.setattr(
        orchestrator_module,
        "_load_cfg_for_case",
        lambda _request, _case: cfg,
    )

    captured: dict[str, object] = {}

    def _build_even_split(
        cores: tuple[int, ...], env_count: int
    ) -> tuple[tuple[int, ...], ...]:
        captured["cores"] = cores
        captured["env_count"] = env_count
        return ((0, 1), (2, 3), (4, 5))

    monkeypatch.setattr(
        orchestrator_module,
        "build_even_split_cpu_groups",
        _build_even_split,
        raising=False,
    )

    def _run_pipeline(*, sim_factory, model_factory, config):
        del sim_factory, model_factory
        captured["sim_cpu_affinity"] = config.sim_cpu_affinity
        return CaseMetrics(pipeline_samples_per_sec=1.0)

    monkeypatch.setattr(
        orchestrator_module,
        "run_dual_process_pipeline",
        _run_pipeline,
    )

    metrics = orchestrator_module._execute_case(request, case)

    assert metrics.pipeline_samples_per_sec == 1.0
    assert captured == {
        "cores": (0, 1, 2, 3, 4, 5),
        "env_count": 3,
        "sim_cpu_affinity": (0, 1, 2, 3, 4, 5),
    }


def test_execute_case_cpu_core_skips_when_affinity_unsupported(
    tmp_path, monkeypatch
) -> None:
    case = BenchmarkCase(
        case_id="cpu-env-only-unsupported",
        scenario="env_only_cpu_core",
        preset_name="p",
        env_type="maniskill",
        model_type="openvla_oft",
        cpu_binding_mode="even_split",
        cpu_available_cores=(0, 1),
    )
    request = _make_request(str(tmp_path))
    request = BenchmarkRequest(
        **{
            **request.__dict__,
            "scenario_set": ("env_only_cpu_core",),
            "cpu_bind_cores": "0-1",
            "cpu_bind_strategy": "even_split",
            "cpu_bind_config": None,
            "cpu_bind_strict": True,
        }
    )

    monkeypatch.delattr(orchestrator_module.os, "sched_setaffinity", raising=False)

    with pytest.raises(SkipCase, match="unavailable"):
        orchestrator_module._execute_case(request, case)


def test_orchestrator_prepares_cpu_groups_for_reporting(tmp_path, monkeypatch) -> None:
    case = BenchmarkCase(
        case_id="cpu-groups-report",
        scenario="env_only_cpu_core",
        preset_name="p",
        env_type="maniskill",
        model_type="openvla_oft",
        cpu_binding_mode="even_split",
        cpu_available_cores=(0, 1, 2, 3),
    )
    monkeypatch.setattr(
        "toolkits.rollout_eval.benchmark.orchestrator.expand_cases",
        lambda _request: [case],
    )

    cfg = SimpleNamespace(env=SimpleNamespace(eval=SimpleNamespace(total_num_envs=2)))
    monkeypatch.setattr(
        orchestrator_module,
        "_load_cfg_for_case",
        lambda _request, _case: cfg,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "build_even_split_cpu_groups",
        lambda cores, env_count: ((0, 1), (2, 3)),
        raising=False,
    )

    captured_case: dict[str, BenchmarkCase] = {}

    def _executor(_request: BenchmarkRequest, prepared_case: BenchmarkCase) -> CaseMetrics:
        captured_case["case"] = prepared_case
        return CaseMetrics(env_steps_per_sec=1.0)

    request = _make_request(str(tmp_path))
    request = BenchmarkRequest(
        **{
            **request.__dict__,
            "scenario_set": ("env_only_cpu_core",),
            "cpu_bind_cores": "0-3",
            "cpu_bind_strategy": "even_split",
            "cpu_bind_config": None,
            "cpu_bind_strict": True,
        }
    )

    summary = run_benchmark_orchestrator(request, case_executor=_executor)

    assert summary["counts"]["pass"] == 1
    assert captured_case["case"].cpu_env_core_groups == ((0, 1), (2, 3))


def test_orchestrator_writes_cpu_binding_metadata(tmp_path, monkeypatch) -> None:
    case = BenchmarkCase(
        case_id="cpu-report-case",
        scenario="env_only_cpu_core",
        preset_name="p",
        env_type="maniskill",
        model_type="openvla_oft",
        cpu_binding_mode="even_split",
        cpu_available_cores=(0, 1, 2, 3),
        cpu_env_core_groups=((0, 1), (2, 3)),
    )
    monkeypatch.setattr(
        "toolkits.rollout_eval.benchmark.orchestrator.expand_cases",
        lambda _request: [case],
    )

    request = _make_request(str(tmp_path))
    summary = run_benchmark_orchestrator(
        request,
        case_executor=lambda _request, _case: CaseMetrics(env_steps_per_sec=12.0),
    )

    report = json.loads(
        (tmp_path / "env_only_cpu_core" / "cpu-report-case" / "case_report.json").read_text(
            encoding="utf-8"
        )
    )
    metadata = json.loads(
        (tmp_path / "env_only_cpu_core" / "cpu-report-case" / "case_meta.json").read_text(
            encoding="utf-8"
        )
    )
    summary_md = (tmp_path / "summary.md").read_text(encoding="utf-8")

    assert summary["counts"]["pass"] == 1
    assert report["resource"]["cpu_binding_mode"] == "even_split"
    assert report["resource"]["cpu_available_cores"] == [0, 1, 2, 3]
    assert report["resource"]["cpu_env_core_groups"] == [[0, 1], [2, 3]]
    assert metadata["resource"]["cpu_effective_affinity"] == [0, 1, 2, 3]
    assert "| cpu-report-case | env_only_cpu_core | pass | 12.000000 | 0.000000 | 0.000000 | even_split | 4 |" in summary_md
    assert "[[0, 1], [2, 3]]" not in summary_md


def test_orchestrator_isolates_prepare_case_failures(tmp_path, monkeypatch) -> None:
    cases = [
        BenchmarkCase(
            case_id="cpu-case-fail",
            scenario="env_only_cpu_core",
            preset_name="p",
            env_type="maniskill",
            model_type="openvla_oft",
            cpu_binding_mode="even_split",
            cpu_available_cores=(0, 1),
        ),
        BenchmarkCase(
            case_id="plain-case-pass",
            scenario="env_only_mps",
            preset_name="p",
            env_type="maniskill",
            model_type="openvla_oft",
            mps_sm=20,
        ),
    ]
    monkeypatch.setattr(
        "toolkits.rollout_eval.benchmark.orchestrator.expand_cases",
        lambda _request: list(cases),
    )

    def _prepare_case(_request: BenchmarkRequest, case: BenchmarkCase) -> BenchmarkCase:
        if case.case_id == "cpu-case-fail":
            raise RuntimeError("bad cpu config")
        return case

    monkeypatch.setattr(
        orchestrator_module,
        "_prepare_case_cpu_groups",
        _prepare_case,
    )

    request = _make_request(str(tmp_path))
    summary = run_benchmark_orchestrator(
        request,
        case_executor=lambda _request, _case: CaseMetrics(env_steps_per_sec=1.0),
    )

    assert summary["counts"] == {"total": 2, "pass": 1, "failed": 1, "skipped": 0}
    failed_report = tmp_path / "env_only_cpu_core" / "cpu-case-fail" / "case_report.json"
    pass_report = tmp_path / "env_only_mps" / "plain-case-pass" / "case_report.json"
    assert json.loads(failed_report.read_text(encoding="utf-8"))["status"] == "failed"
    assert json.loads(pass_report.read_text(encoding="utf-8"))["status"] == "pass"


def test_execute_case_in_subprocess_uses_queue_get_without_empty(monkeypatch) -> None:
    class _FakeQueue:
        def __init__(self) -> None:
            self._items: list[dict[str, object]] = []

        def put(self, item: dict[str, object]) -> None:
            self._items.append(item)

        def get(self, timeout: float | None = None) -> dict[str, object]:
            del timeout
            if not self._items:
                raise Empty
            return self._items.pop(0)

    class _FakeProcess:
        def __init__(self, *, kwargs, name: str) -> None:
            self.kwargs = kwargs
            self.name = name
            self.exitcode = 0
            self._alive = False

        def start(self) -> None:
            queue = self.kwargs["result_queue"]
            queue.put({"status": "pass", "metrics": {"env_steps_per_sec": 1.0}})
            self._alive = False

        def is_alive(self) -> bool:
            return self._alive

        def join(self, timeout: float | None = None) -> None:
            del timeout

        def terminate(self) -> None:
            self._alive = False

    class _FakeContext:
        def Queue(self, maxsize: int | None = None) -> _FakeQueue:
            del maxsize
            return _FakeQueue()

        def Process(self, *, target, kwargs, name: str) -> _FakeProcess:
            del target
            return _FakeProcess(kwargs=kwargs, name=name)

    request = _make_request("/tmp/out")
    case = BenchmarkCase(
        case_id="subprocess-case",
        scenario="env_only_mps",
        preset_name="p",
        env_type="maniskill",
        model_type="openvla_oft",
        mps_sm=20,
    )

    monkeypatch.setattr(orchestrator_module.mp, "get_context", lambda _method: _FakeContext())

    result = orchestrator_module._execute_case_in_subprocess(request, case)

    assert result == {"status": "pass", "metrics": {"env_steps_per_sec": 1.0}}


def test_execute_case_in_subprocess_drains_late_queue_message(monkeypatch) -> None:
    class _LateMessageQueue:
        def __init__(self) -> None:
            self._calls = 0

        def get(self, timeout: float | None = None) -> dict[str, object]:
            del timeout
            self._calls += 1
            if self._calls == 1:
                raise Empty
            return {"status": "pass", "metrics": {"env_steps_per_sec": 2.0}}

    class _FakeProcess:
        def __init__(self, *, kwargs, name: str) -> None:
            del kwargs, name
            self.exitcode = 0

        def start(self) -> None:
            return None

        def is_alive(self) -> bool:
            return False

        def join(self, timeout: float | None = None) -> None:
            del timeout

        def terminate(self) -> None:
            return None

    class _FakeContext:
        def Queue(self, maxsize: int | None = None) -> _LateMessageQueue:
            del maxsize
            return _LateMessageQueue()

        def Process(self, *, target, kwargs, name: str) -> _FakeProcess:
            del target
            return _FakeProcess(kwargs=kwargs, name=name)

    request = _make_request("/tmp/out")
    case = BenchmarkCase(
        case_id="late-message-case",
        scenario="env_only_mps",
        preset_name="p",
        env_type="maniskill",
        model_type="openvla_oft",
        mps_sm=20,
    )

    monkeypatch.setattr(orchestrator_module.mp, "get_context", lambda _method: _FakeContext())

    result = orchestrator_module._execute_case_in_subprocess(request, case)

    assert result == {"status": "pass", "metrics": {"env_steps_per_sec": 2.0}}


def test_run_case_worker_includes_exception_type_and_traceback() -> None:
    class _EmptyMessageError(RuntimeError):
        def __str__(self) -> str:
            return ""

    class _FakeQueue:
        def __init__(self) -> None:
            self.items: list[dict[str, object]] = []

        def put(self, item: dict[str, object]) -> None:
            self.items.append(item)

    queue = _FakeQueue()
    request = _make_request("/tmp/out")
    case = BenchmarkCase(
        case_id="failing-case",
        scenario="env_only_mps",
        preset_name="p",
        env_type="maniskill",
        model_type="openvla_oft",
        mps_sm=20,
    )

    original_execute_case = orchestrator_module._execute_case
    try:
        orchestrator_module._execute_case = lambda _request, _case: (_ for _ in ()).throw(  # type: ignore[assignment]
            _EmptyMessageError()
        )
        orchestrator_module._run_case_worker(queue, request, case)
    finally:
        orchestrator_module._execute_case = original_execute_case  # type: ignore[assignment]

    assert len(queue.items) == 1
    payload = queue.items[0]
    assert payload["status"] == "failed"
    assert "EmptyMessageError" in str(payload["error_message"])
    assert "Traceback" in str(payload["error_message"])
