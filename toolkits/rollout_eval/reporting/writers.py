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
