"""Reporting utilities for rollout_eval benchmark per-case and summary outputs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from toolkits.rollout_eval.benchmark.types import (
    BenchmarkCase,
    BenchmarkRequest,
    CaseMetrics,
)


def _dump_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resource_binding(case: BenchmarkCase) -> dict[str, object]:
    cpu_groups = [list(group) for group in case.cpu_env_core_groups] if case.cpu_env_core_groups else None
    cpu_effective_affinity = (
        sorted({cpu for group in case.cpu_env_core_groups for cpu in group})
        if case.cpu_env_core_groups
        else None
    )
    return {
        "mps_sm": case.mps_sm,
        "mig_device": case.mig_device,
        "cpu_binding_mode": case.cpu_binding_mode,
        "cpu_available_cores": list(case.cpu_available_cores) if case.cpu_available_cores else None,
        "cpu_env_core_groups": cpu_groups,
        "cpu_effective_affinity": cpu_effective_affinity,
    }


def write_case_report(
    *,
    output_dir: Path,
    request: BenchmarkRequest,
    case: BenchmarkCase,
    status: str,
    metrics: CaseMetrics | None,
    process_env: dict[str, str],
    error_message: str | None,
    skip_reason: str | None,
) -> dict:
    """Write case-level report and metadata files, return report dict."""
    case_dir = output_dir / case.scenario / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "case_id": case.case_id,
        "scenario": case.scenario,
        "preset_name": case.preset_name,
        "env_type": case.env_type,
        "model_type": case.model_type,
        "resource": _resource_binding(case),
        "status": status,
        "metrics": asdict(metrics) if metrics is not None else None,
        "error_message": error_message,
        "skip_reason": skip_reason,
    }

    metadata = {
        "case_id": case.case_id,
        "scenario": case.scenario,
        "request": {
            "config_path": request.config_path,
            "config_name": request.config_name,
            "override": list(request.override),
            "pipeline": request.pipeline,
            "warmup_steps": request.warmup_steps,
            "measure_steps": request.measure_steps,
            "num_envs_override": request.num_envs_override,
            "pipeline_queue_timeout_s": request.pipeline_queue_timeout_s,
            "pipeline_run_timeout_s": request.pipeline_run_timeout_s,
        },
        "process_env": process_env,
        "resource": _resource_binding(case),
    }

    _dump_json(report, case_dir / "case_report.json")
    _dump_json(metadata, case_dir / "case_meta.json")
    return report


def write_summary_reports(*, output_dir: Path, case_records: list[dict]) -> dict:
    """Write summary json + markdown reports for all benchmark cases."""
    counts = {
        "total": len(case_records),
        "pass": sum(1 for record in case_records if record.get("status") == "pass"),
        "failed": sum(1 for record in case_records if record.get("status") == "failed"),
        "skipped": sum(1 for record in case_records if record.get("status") == "skipped"),
    }

    summary = {
        "counts": counts,
        "cases": case_records,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _dump_json(summary, output_dir / "summary.json")

    lines = [
        "# Rollout Eval Benchmark Summary",
        "",
        f"- Total: {counts['total']}",
        f"- Pass: {counts['pass']}",
        f"- Failed: {counts['failed']}",
        f"- Skipped: {counts['skipped']}",
        "",
        "| case_id | scenario | status | env_steps/s | infer/s | pipeline/s | cpu_mode | cpu_cores | infer_gpu_ms(avg) |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]

    for record in case_records:
        metrics = record.get("metrics") or {}
        resource = record.get("resource") or {}
        cpu_cores = len(resource.get("cpu_effective_affinity") or [])
        lines.append(
            "| {case_id} | {scenario} | {status} | {env:.6f} | {infer:.6f} | {pipe:.6f} | {cpu_mode} | {cpu_cores} | {gpu:.3f} |".format(
                case_id=record.get("case_id", ""),
                scenario=record.get("scenario", ""),
                status=record.get("status", ""),
                env=float(metrics.get("env_steps_per_sec", 0.0)),
                infer=float(metrics.get("model_infers_per_sec", 0.0)),
                pipe=float(metrics.get("pipeline_samples_per_sec", 0.0)),
                cpu_mode=resource.get("cpu_binding_mode") or "",
                cpu_cores=cpu_cores,
                gpu=float((metrics.get("model_infer_gpu_time_ms") or {}).get("avg_ms", 0.0)),
            )
        )

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
