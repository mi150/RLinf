"""Tensor-parallel model-only OpenPI benchmark under MPS SM quotas.

This entrypoint is intentionally separate from ``benchmark.run``.  The regular
benchmark matrix is single-process and single-GPU; this file owns distributed
process-group setup so TP profiling can use one rank per GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from toolkits.rollout_eval.adapters.model_adapter import GenericModelAdapter
from toolkits.rollout_eval.benchmark.metrics import aggregate_case_metrics
from toolkits.rollout_eval.benchmark.orchestrator import _make_random_model_obs
from toolkits.rollout_eval.benchmark.types import BenchmarkCase


@dataclass(frozen=True)
class TPProfileCase:
    """One tensor-parallel model-only case."""

    case_id: str
    tp_size: int
    mps_sm: int
    num_envs: int
    gpu_ids: tuple[int, ...]


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _parse_csv(value))


def _free_tcp_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for TP model-only MPS profiling."""
    parser = argparse.ArgumentParser(
        description="Profile OpenPI model-only tensor parallel inference under MPS"
    )
    parser.add_argument("--config-path", required=True, help="Hydra config directory")
    parser.add_argument("--config-name", required=True, help="Hydra config name")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override, repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        default="./rollout_eval_output/tp_model_profile",
        help="Output directory for TP benchmark reports",
    )
    parser.add_argument(
        "--gpus",
        required=True,
        help="Comma-separated physical GPU ids. Each TP case uses the first tp_size ids.",
    )
    parser.add_argument(
        "--tp-sizes",
        default="2",
        help="Comma-separated tensor parallel sizes to profile.",
    )
    parser.add_argument(
        "--mps-sm",
        default="20,40,60,80,100",
        help="Comma-separated MPS SM active thread percentages.",
    )
    parser.add_argument(
        "--num-envs-list",
        default="1,4,8,16,32",
        help="Comma-separated random observation batch sizes.",
    )
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument(
        "--model-only-input",
        default="random",
        choices=["random"],
        help="Only random input is supported for TP model-only profiling.",
    )
    parser.add_argument(
        "--skip-validate-cfg",
        action="store_true",
        help="Kept for CLI symmetry; this script does not call validate_cfg.",
    )
    parser.add_argument(
        "--case-timeout-s",
        type=float,
        default=None,
        help="Optional timeout per TP case.",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--tp-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mps-sm-current", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num-envs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def build_tp_cases(args: argparse.Namespace) -> list[TPProfileCase]:
    """Build the deterministic TP/MPS/batch matrix."""
    gpu_ids = _parse_int_csv(args.gpus)
    cases: list[TPProfileCase] = []
    for tp_size in sorted(set(_parse_int_csv(args.tp_sizes))):
        if tp_size <= 0:
            raise ValueError("--tp-sizes values must be positive")
        if tp_size > len(gpu_ids):
            raise ValueError(
                f"tp_size={tp_size} requires at least {tp_size} GPUs, "
                f"got {len(gpu_ids)} from --gpus"
            )
        for mps_sm in sorted(set(_parse_int_csv(args.mps_sm))):
            if mps_sm <= 0 or mps_sm > 100:
                raise ValueError("--mps-sm values must be in [1, 100]")
            for num_envs in sorted(set(_parse_int_csv(args.num_envs_list))):
                if num_envs <= 0:
                    raise ValueError("--num-envs-list values must be positive")
                cases.append(
                    TPProfileCase(
                        case_id=f"tp{tp_size}-mps-sm{mps_sm}-bs{num_envs}",
                        tp_size=tp_size,
                        mps_sm=mps_sm,
                        num_envs=num_envs,
                        gpu_ids=gpu_ids[:tp_size],
                    )
                )
    return cases


def _load_cfg(args: argparse.Namespace):
    abs_config_path = str(Path(args.config_path).resolve())
    with initialize_config_dir(version_base="1.1", config_dir=abs_config_path):
        cfg = compose(config_name=args.config_name, overrides=list(args.override))
    with open_dict(cfg):
        cfg.env.eval.env_type = "libero"
        cfg.env.eval.total_num_envs = int(args.num_envs)
        cfg.actor.model.model_type = "openpi"
        cfg.rollout.model.model_type = "openpi"
    return cfg


def _resolve_model_path(cfg) -> str:
    if "rollout" in cfg and "model" in cfg.rollout and "model_path" in cfg.rollout.model:
        return str(cfg.rollout.model.model_path)
    return str(cfg.actor.model.model_path)


def _build_openpi_model(cfg):
    from rlinf.models import get_model
    from toolkits.rollout_eval.adapters.model_adapter import _validate_model_path_or_raise

    model_path = _resolve_model_path(cfg)
    _validate_model_path_or_raise(model_path, model_type="openpi")
    model_cfg = cfg.actor.model.copy()
    with open_dict(model_cfg):
        if "openpi_data" in cfg:
            model_cfg.openpi_data = cfg.openpi_data
        if "rollout" in cfg and "model" in cfg.rollout:
            if "precision" in cfg.rollout.model:
                model_cfg.precision = cfg.rollout.model.precision
            if "model_path" in cfg.rollout.model:
                model_cfg.model_path = cfg.rollout.model.model_path
    model = get_model(model_cfg)
    model.eval()
    return model


def _openpi_tp_plan(model) -> dict[str, Any]:
    from torch.distributed.tensor import Replicate, Shard
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    module_names = dict(model.named_modules())
    plan: dict[str, Any] = {}
    linear_suffixes = (
        "mlp.gate_proj",
        "mlp.up_proj",
    )
    rowwise_suffixes = (
        "mlp.down_proj",
    )
    for name in module_names:
        if name.endswith(linear_suffixes):
            plan[name] = ColwiseParallel(output_layouts=Shard(-1))
        elif name.endswith(rowwise_suffixes):
            plan[name] = RowwiseParallel(input_layouts=Shard(-1), output_layouts=Replicate())

    # Keep attention projections replicated.  OpenPI uses a custom attention
    # path with explicit q/k/v reshapes, so sharding those modules requires a
    # deeper rewrite of the forward path.
    if not plan:
        raise RuntimeError("No OpenPI modules matched the TP parallelization plan")
    return plan


def _parallelize_openpi_model(model, tp_size: int) -> tuple[Any, int]:
    if tp_size == 1:
        return model, 0

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import parallelize_module

    mesh = init_device_mesh("cuda", (tp_size,))
    plan = _openpi_tp_plan(model)
    model = parallelize_module(model, device_mesh=mesh, parallelize_plan=plan)
    return model, len(plan)


def _sampling_defaults(cfg) -> dict[str, Any]:
    sampling_cfg = cfg.algorithm.get("sampling_params", {})
    temp_eval = float(sampling_cfg.get("temperature_eval", -1))
    do_sample = bool(temp_eval > 0)
    return {
        "do_sample": do_sample,
        "temperature": temp_eval if do_sample else 1.0,
        "top_k": int(sampling_cfg.get("top_k", 0)),
    }


def _worker_result_path(output_dir: Path, case_id: str, rank: int) -> Path:
    return output_dir / "cases" / case_id / f"rank{rank}.json"


def _write_worker_result(output_dir: Path, case_id: str, rank: int, payload: dict) -> None:
    path = _worker_result_path(output_dir, case_id, rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_worker(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    started = time.perf_counter()
    try:
        cfg = _load_cfg(args)
        model = _build_openpi_model(cfg).cuda(local_rank)
        model, tp_module_count = _parallelize_openpi_model(model, world_size)
        adapter = GenericModelAdapter(
            model=model,
            model_type="openpi",
            split_model_stages=False,
            sampling_defaults=_sampling_defaults(cfg),
        )
        obs_batch = _make_random_model_obs(
            cfg,
            BenchmarkCase(
                case_id=str(args.case_id),
                scenario="model_only_tp_mps",
                preset_name="libero_openpi",
                env_type="libero",
                model_type="openpi",
                num_envs=int(args.num_envs),
                mps_sm=int(args.mps_sm_current),
            ),
        )

        total_steps = int(args.warmup_steps) + int(args.measure_steps)
        latencies_ms: list[float] = []
        gpu_times_ms: list[float] = []
        measured_seconds = 0.0

        dist.barrier()
        for step_idx in range(total_steps):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            start = time.perf_counter()
            adapter.infer(obs_batch=obs_batch, mode="eval")
            end_event.record()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if step_idx >= int(args.warmup_steps):
                measured_seconds += elapsed
                latencies_ms.append(elapsed * 1000.0)
                gpu_times_ms.append(float(start_event.elapsed_time(end_event)))
        dist.barrier()

        metrics = aggregate_case_metrics(
            model_infer_count=int(args.measure_steps),
            model_infer_seconds=measured_seconds,
            model_infer_latency_ms=latencies_ms,
            model_infer_gpu_time_ms=gpu_times_ms,
        )
        status_payload = {
            "status": "pass",
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "tp_module_count": tp_module_count,
            "elapsed_s": time.perf_counter() - started,
            "metrics": asdict(metrics),
        }
        _write_worker_result(output_dir, str(args.case_id), rank, status_payload)
    except Exception as exc:  # noqa: BLE001
        _write_worker_result(
            output_dir,
            str(args.case_id),
            rank,
            {
                "status": "failed",
                "rank": rank,
                "error_message": (
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                ),
            },
        )
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_case(args: argparse.Namespace, case: TPProfileCase) -> dict:
    case_dir = Path(args.output_dir) / "cases" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(str(gpu_id) for gpu_id in case.gpu_ids),
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(case.mps_sm),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(_free_tcp_port()),
        }
    )
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={case.tp_size}",
        "-m",
        "toolkits.rollout_eval.benchmark.tp_model_profile",
        "--worker",
        "--config-path",
        args.config_path,
        "--config-name",
        args.config_name,
        "--output-dir",
        args.output_dir,
        "--gpus",
        ",".join(str(gpu_id) for gpu_id in case.gpu_ids),
        "--case-id",
        case.case_id,
        "--tp-size",
        str(case.tp_size),
        "--mps-sm-current",
        str(case.mps_sm),
        "--num-envs",
        str(case.num_envs),
        "--warmup-steps",
        str(args.warmup_steps),
        "--measure-steps",
        str(args.measure_steps),
    ]
    for override in args.override:
        cmd.extend(["--override", override])

    started = time.perf_counter()
    completed = subprocess.run(
        cmd,
        env=env,
        cwd=Path.cwd(),
        check=False,
        text=True,
        capture_output=True,
        timeout=args.case_timeout_s,
    )
    (case_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (case_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")

    rank_payloads = []
    for rank in range(case.tp_size):
        result_path = _worker_result_path(Path(args.output_dir), case.case_id, rank)
        if result_path.exists():
            rank_payloads.append(json.loads(result_path.read_text(encoding="utf-8")))

    failed_ranks = [p for p in rank_payloads if p.get("status") != "pass"]
    status = "pass" if completed.returncode == 0 and not failed_ranks else "failed"
    rank0 = next((p for p in rank_payloads if p.get("rank") == 0), None)
    metrics = rank0.get("metrics") if rank0 else None
    record = {
        "case_id": case.case_id,
        "status": status,
        "tp_size": case.tp_size,
        "mps_sm": case.mps_sm,
        "num_envs": case.num_envs,
        "gpu_ids": list(case.gpu_ids),
        "elapsed_s": time.perf_counter() - started,
        "returncode": completed.returncode,
        "metrics": metrics,
        "rank_results": rank_payloads,
        "error_message": None,
    }
    if status != "pass":
        if failed_ranks:
            record["error_message"] = failed_ranks[0].get("error_message")
        else:
            record["error_message"] = completed.stderr[-4000:]
    (case_dir / "case_report.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return record


def write_summary(output_dir: Path, records: list[dict]) -> dict:
    """Write JSON and Markdown summaries for TP profile cases."""
    counts = {
        "total": len(records),
        "pass": sum(1 for record in records if record.get("status") == "pass"),
        "failed": sum(1 for record in records if record.get("status") == "failed"),
    }
    summary = {"counts": counts, "cases": records}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# OpenPI TP Model-Only MPS Profile Summary",
        "",
        f"- Total: {counts['total']}",
        f"- Pass: {counts['pass']}",
        f"- Failed: {counts['failed']}",
        "",
        "| case_id | status | tp_size | mps_sm | num_envs | infer/s | step/s | ideal step/s | infer_gpu_ms(avg) | gpus |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        metrics = record.get("metrics") or {}
        infer_per_sec = float(metrics.get("model_infers_per_sec", 0.0))
        num_envs = int(record.get("num_envs") or 0)
        step_per_sec = infer_per_sec * float(num_envs)
        mps_sm = int(record.get("mps_sm") or 0)
        ideal_step_per_sec = (100.0 / mps_sm) * step_per_sec if mps_sm else 0.0
        gpu_avg = float((metrics.get("model_infer_gpu_time_ms") or {}).get("avg_ms", 0.0))
        lines.append(
            "| {case_id} | {status} | {tp_size} | {mps_sm} | {num_envs} | "
            "{infer:.6f} | {step:.6f} | {ideal:.6f} | {gpu_avg:.3f} | {gpus} |".format(
                case_id=record.get("case_id", ""),
                status=record.get("status", ""),
                tp_size=record.get("tp_size", ""),
                mps_sm=mps_sm or "",
                num_envs=num_envs or "",
                infer=infer_per_sec,
                step=step_per_sec,
                ideal=ideal_step_per_sec,
                gpu_avg=gpu_avg,
                gpus=",".join(str(gpu) for gpu in record.get("gpu_ids", [])),
            )
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run TP model-only benchmark matrix or a distributed worker."""
    args = parse_args(argv)
    if args.worker:
        _run_worker(args)
        return

    output_dir = Path(args.output_dir)
    records = []
    for case in build_tp_cases(args):
        records.append(_run_case(args, case))
        write_summary(output_dir, records)
    print(json.dumps(write_summary(output_dir, records), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
