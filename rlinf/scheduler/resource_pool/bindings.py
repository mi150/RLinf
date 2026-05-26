from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

RESOURCE_BINDING_ENV = "RLINF_RESOURCE_BINDING_JSON"
CPU_AFFINITY_ENV = "RLINF_CPU_AFFINITY"
ENV_CPU_CORE_GROUPS_ENV = "RLINF_ENV_CPU_CORE_GROUPS"
CUDA_VISIBLE_DEVICES_ENV = "CUDA_VISIBLE_DEVICES"
MPS_ACTIVE_THREAD_PERCENTAGE_ENV = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"


def _tuple_int(values: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


@dataclass(frozen=True)
class CpuBinding:
    """CPU binding metadata for one worker process."""

    process_cpu_cores: tuple[int, ...] = ()
    env_cpu_core_groups: tuple[tuple[int, ...], ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CpuBinding | None":
        """Build a CPU binding from JSON-compatible data."""
        if payload is None:
            return None
        return cls(
            process_cpu_cores=_tuple_int(payload.get("process_cpu_cores", ())),
            env_cpu_core_groups=tuple(
                _tuple_int(group) for group in payload.get("env_cpu_core_groups", ())
            ),
        )


@dataclass(frozen=True)
class GpuBinding:
    """GPU quota and visibility metadata for one worker process."""

    mode: Literal["mps", "mig"] | None = None
    sm_percent: int = 0
    visible_devices: tuple[str, ...] = ()
    mig_device_uuid: str | None = None
    parent_gpu: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GpuBinding | None":
        """Build a GPU binding from JSON-compatible data."""
        if payload is None:
            return None
        mode = payload.get("mode")
        if mode not in ("mps", "mig"):
            raise ValueError(f"mode must be one of 'mps' or 'mig', got {mode}")
        if "sm_percent" not in payload or payload.get("sm_percent") is None:
            raise ValueError("GPU binding requires sm_percent")
        mig_device_uuid = payload.get("mig_device_uuid")
        parent_gpu = payload.get("parent_gpu")
        if mode == "mig" and not mig_device_uuid:
            raise ValueError("MIG binding requires mig_device_uuid")
        if mode == "mig" and parent_gpu is None:
            raise ValueError("MIG binding requires parent GPU metadata")
        return cls(
            mode=mode,
            sm_percent=int(payload.get("sm_percent", 0)),
            visible_devices=tuple(str(v) for v in payload.get("visible_devices", ())),
            mig_device_uuid=str(mig_device_uuid)
            if mig_device_uuid is not None
            else None,
            parent_gpu=int(parent_gpu) if parent_gpu is not None else None,
        )


@dataclass(frozen=True)
class WorkerResourceBinding:
    """Resolved resource binding for a worker rank."""

    component: str
    rank: int
    cluster_node_rank: int
    node_group_label: str
    cpu: CpuBinding | None = None
    gpu: GpuBinding | None = None

    def to_json(self) -> str:
        """Serialize the binding as stable compact JSON."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "WorkerResourceBinding":
        """Deserialize a binding from JSON."""
        payload = json.loads(text)
        if "gpu" not in payload:
            raise ValueError("resource binding requires explicit gpu field")
        return cls(
            component=str(payload["component"]),
            rank=int(payload["rank"]),
            cluster_node_rank=int(payload["cluster_node_rank"]),
            node_group_label=str(payload["node_group_label"]),
            cpu=CpuBinding.from_dict(payload.get("cpu")),
            gpu=GpuBinding.from_dict(payload.get("gpu")),
        )

    def to_env_vars(self) -> dict[str, str]:
        """Convert the binding into environment variables for a worker."""
        from .gpu_binding import build_gpu_env_vars

        env = {RESOURCE_BINDING_ENV: self.to_json()}
        if self.cpu is not None and self.cpu.process_cpu_cores:
            env[CPU_AFFINITY_ENV] = ",".join(map(str, self.cpu.process_cpu_cores))
            if self.cpu.env_cpu_core_groups:
                env[ENV_CPU_CORE_GROUPS_ENV] = ";".join(
                    ",".join(map(str, group)) for group in self.cpu.env_cpu_core_groups
                )
        if self.gpu is not None:
            env.update(build_gpu_env_vars(self.gpu))
        return env
