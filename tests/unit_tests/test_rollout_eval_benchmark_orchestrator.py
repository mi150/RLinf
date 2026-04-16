from __future__ import annotations

import json

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
