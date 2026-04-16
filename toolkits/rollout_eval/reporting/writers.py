"""Writers for rollout eval report artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def dump_report_json(report: dict[str, Any], output_path: str) -> None:
    """Write machine-readable report JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")



def dump_report_markdown(report: dict[str, Any], output_path: str) -> None:
    """Write human-readable report markdown."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tp = report.get("torch_profile", {})

    lines = [
        "# Rollout Eval Report",
        "",
        "## Summary",
        f"- status: {report.get('status', 'unknown')}",
        f"- total_steps: {report.get('total_steps', 0)}",
        f"- warmup_steps: {report.get('warmup_steps', 0)}",
        f"- measure_steps: {report.get('measure_steps', 0)}",
        "",
        "## Latency",
        f"- model_infer_seconds: {report.get('latency', {}).get('model_infer_seconds', 0.0):.6f}",
        f"- env_step_seconds: {report.get('latency', {}).get('env_step_seconds', 0.0):.6f}",
        "",
        "## Torch Profile (us)",
        f"- env_sim_cpu_us: {tp.get('env_sim_cpu_us', 0.0):.3f}",
        f"- model_infer_cpu_us: {tp.get('model_infer_cpu_us', 0.0):.3f}",
        f"- model_backbone_cpu_us: {tp.get('model_backbone_cpu_us', 0.0):.3f}",
        f"- model_action_head_cpu_us: {tp.get('model_action_head_cpu_us', 0.0):.3f}",
        f"- env_sim_cuda_us: {tp.get('env_sim_cuda_us', 0.0):.3f}",
        f"- model_infer_cuda_us: {tp.get('model_infer_cuda_us', 0.0):.3f}",
        f"- model_backbone_cuda_us: {tp.get('model_backbone_cuda_us', 0.0):.3f}",
        f"- model_action_head_cuda_us: {tp.get('model_action_head_cuda_us', 0.0):.3f}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def dump_batch_sweep_json(reports: list[dict[str, Any]], output_path: str) -> None:
    """Write batch-size sweep summary as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in reports:
        lat = r.get("latency", {})
        tp = r.get("torch_profile", {})
        num_envs = r.get("num_envs", 0)
        infer_s = lat.get("model_infer_seconds", 0.0)
        infer_count = lat.get("model_infer_count", 1) or 1
        avg_infer_ms = infer_s / infer_count * 1000.0
        throughput = num_envs / avg_infer_ms * 1000.0 if avg_infer_ms > 0 else 0.0
        rows.append({
            "num_envs": num_envs,
            "model_infer_seconds_total": infer_s,
            "model_infer_count": infer_count,
            "avg_infer_ms_per_call": avg_infer_ms,
            "throughput_envs_per_sec": throughput,
            "model_backbone_cuda_us": tp.get("model_backbone_cuda_us", 0.0),
            "model_action_head_cuda_us": tp.get("model_action_head_cuda_us", 0.0),
            "model_infer_cuda_us": tp.get("model_infer_cuda_us", 0.0),
        })

    path.write_text(json.dumps({"batch_sweep": rows}, indent=2), encoding="utf-8")


def dump_batch_sweep_markdown(reports: list[dict[str, Any]], output_path: str) -> None:
    """Write batch-size sweep summary as a Markdown comparison table."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "| num_envs | avg_infer_ms | throughput (envs/s) "
        "| backbone_cuda_us | action_head_cuda_us | total_infer_cuda_us |"
    )
    sep = "|---|---|---|---|---|---|"

    rows = [header, sep]
    for r in reports:
        lat = r.get("latency", {})
        tp = r.get("torch_profile", {})
        num_envs = r.get("num_envs", 0)
        infer_s = lat.get("model_infer_seconds", 0.0)
        infer_count = lat.get("model_infer_count", 1) or 1
        avg_ms = infer_s / infer_count * 1000.0
        throughput = num_envs / avg_ms * 1000.0 if avg_ms > 0 else 0.0
        rows.append(
            f"| {num_envs} | {avg_ms:.2f} | {throughput:.1f} "
            f"| {tp.get('model_backbone_cuda_us', 0.0):.1f} "
            f"| {tp.get('model_action_head_cuda_us', 0.0):.1f} "
            f"| {tp.get('model_infer_cuda_us', 0.0):.1f} |"
        )

    lines = [
        "# Batch-Size Sweep Report",
        "",
        "Model is loaded once; environment is rebuilt for each batch size.",
        "",
        *rows,
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
