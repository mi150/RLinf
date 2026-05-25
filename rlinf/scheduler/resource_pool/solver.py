from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from omegaconf import DictConfig

from .bindings import CpuBinding, GpuBinding, WorkerResourceBinding
from .config import ResourcePoolConfig
from .cpu_binding import (
    build_even_split_cpu_groups,
    effective_process_affinity,
    parse_cpu_core_set,
)


class ResourcePoolSolver:
    """Resolve fine-grained resource bindings for scheduler components."""

    def __init__(
        self,
        pool_cfg: ResourcePoolConfig,
        cfg: DictConfig,
        cluster,
        component_placement,
    ) -> None:
        """Initialize the resource pool solver."""
        self.pool_cfg = pool_cfg
        self.cfg = cfg
        self.cluster = cluster
        self.component_placement = component_placement
        self._used_mig_uuids: set[str] = set()
        self._summary: dict[str, object] = {
            "shared_cpu_cores": {},
            "shared_mig_uuids": {},
            "mps_gpu_totals": {},
        }

    @property
    def summary(self) -> dict[str, object]:
        """Return artifact summary for the most recent solve."""
        return self._summary

    def solve(self) -> dict[str, list[WorkerResourceBinding]]:
        """Resolve resource bindings from config or an explicit plan file."""
        if not self.pool_cfg.enabled:
            return {}
        if self.pool_cfg.allocation_mode == "plan_file":
            bindings = self._load_plan_file()
            self._refresh_summary(bindings)
            return bindings
        self._validate_default_mig_devices()
        components = sorted(
            set(self.pool_cfg.cpu.components) | set(self.pool_cfg.gpu.components)
        )
        bindings = {
            component: self._solve_component(component) for component in components
        }
        self._refresh_summary(bindings)
        return bindings

    def _component_local_env_count(self, worker_count: int) -> int:
        stage_num = int(self.cfg.rollout.get("pipeline_stage_num", 1))
        _ = stage_num
        if bool(getattr(self.cfg.runner, "only_eval", False)):
            total_envs = int(self.cfg.env.eval.total_num_envs)
        else:
            total_envs = int(self.cfg.env.train.total_num_envs)
        if total_envs % worker_count != 0:
            raise ValueError("env total_num_envs must be divisible by env worker count")
        return total_envs // worker_count

    def _solve_component(self, component: str) -> list[WorkerResourceBinding]:
        strategy = self.component_placement.get_strategy(component)
        placements = sorted(
            strategy.get_placement(self.cluster, isolate_accelerator=True),
            key=lambda placement: placement.rank,
        )
        cpu_bindings = self._solve_cpu_component(component, placements)
        bindings: list[WorkerResourceBinding] = []
        for index, placement in enumerate(placements):
            bindings.append(
                WorkerResourceBinding(
                    component=component,
                    rank=int(placement.rank),
                    cluster_node_rank=int(placement.cluster_node_rank),
                    node_group_label=str(placement.node_group_label),
                    cpu=cpu_bindings[index] if cpu_bindings else None,
                    gpu=self._solve_gpu_binding(component, placement),
                )
            )
        return bindings

    def _solve_cpu_component(self, component: str, placements) -> list[CpuBinding]:
        if (
            not self.pool_cfg.cpu.enabled
            or component not in self.pool_cfg.cpu.components
        ):
            return []

        request = self.pool_cfg.cpu.components[component]
        pool = self.pool_cfg.cpu.pools[request.pool]
        for placement in placements:
            if placement.node_group_label != pool.node_group:
                raise ValueError(
                    f"CPU pool '{request.pool}' is scoped to node group "
                    f"'{pool.node_group}', but {component}:{placement.rank} is on "
                    f"'{placement.node_group_label}'"
                )

        cores = parse_cpu_core_set(pool.cores)
        process_core_groups = build_even_split_cpu_groups(cores, len(placements))
        bindings: list[CpuBinding] = []
        for process_cpu_cores in process_core_groups:
            env_cpu_core_groups: tuple[tuple[int, ...], ...] = ()
            if request.granularity == "per_env":
                if component != "env":
                    raise ValueError(
                        "per_env CPU binding is only valid for env component"
                    )
                env_cpu_core_groups = build_even_split_cpu_groups(
                    process_cpu_cores,
                    self._component_local_env_count(len(placements)),
                )
                process_cpu_cores = effective_process_affinity(env_cpu_core_groups)
            bindings.append(
                CpuBinding(
                    process_cpu_cores=process_cpu_cores,
                    env_cpu_core_groups=env_cpu_core_groups,
                )
            )
        return bindings

    def _solve_gpu_binding(self, component: str, placement) -> GpuBinding | None:
        if (
            not self.pool_cfg.gpu.enabled
            or component not in self.pool_cfg.gpu.components
        ):
            return None

        request = self.pool_cfg.gpu.components[component]
        if request.sm_percent == 0:
            return None

        pool = self.pool_cfg.gpu.pools[request.pool]
        if placement.node_group_label != pool.node_group:
            raise ValueError(
                f"GPU pool '{request.pool}' is scoped to node group "
                f"'{pool.node_group}', but {component}:{placement.rank} is on "
                f"'{placement.node_group_label}'"
            )

        if self.pool_cfg.gpu.mode == "mps":
            devices = parse_cpu_core_set(pool.devices or "")
            if placement.local_accelerator_rank not in devices:
                raise ValueError(
                    f"GPU {placement.local_accelerator_rank} for {component}:"
                    f"{placement.rank} is not in pool '{request.pool}' devices"
                )
            return GpuBinding(
                mode="mps",
                sm_percent=request.sm_percent,
                visible_devices=tuple(placement.visible_accelerators),
                parent_gpu=placement.local_accelerator_rank,
            )

        for device in pool.mig_devices:
            if (
                device.uuid not in self._used_mig_uuids
                and device.parent_gpu == placement.local_accelerator_rank
                and device.sm_percent >= request.sm_percent
            ):
                self._used_mig_uuids.add(device.uuid)
                return GpuBinding(
                    mode="mig",
                    sm_percent=request.sm_percent,
                    mig_device_uuid=device.uuid,
                    parent_gpu=device.parent_gpu,
                )
        raise ValueError(
            f"No MIG device in pool '{request.pool}' can satisfy {component}:"
            f"{placement.rank} on GPU {placement.local_accelerator_rank}"
        )

    def _validate_default_mig_devices(self) -> None:
        if not self.pool_cfg.gpu.enabled or self.pool_cfg.gpu.mode != "mig":
            return

        seen: set[str] = set()
        for pool in self.pool_cfg.gpu.pools.values():
            for device in pool.mig_devices:
                if device.uuid in seen:
                    raise ValueError(
                        f"Duplicate MIG UUID in resource pool: {device.uuid}"
                    )
                seen.add(device.uuid)

    def _load_plan_file(self) -> dict[str, list[WorkerResourceBinding]]:
        if self.pool_cfg.allocation_plan_path is None:
            raise ValueError("allocation_plan_path is required for plan_file mode")

        payload = json.loads(
            Path(self.pool_cfg.allocation_plan_path).read_text(encoding="utf-8")
        )
        bindings: dict[str, list[WorkerResourceBinding]] = defaultdict(list)
        for item in payload.get("bindings", []):
            binding = WorkerResourceBinding.from_json(json.dumps(item))
            bindings[binding.component].append(binding)
        return {
            component: sorted(component_bindings, key=lambda binding: binding.rank)
            for component, component_bindings in bindings.items()
        }

    def _refresh_summary(
        self, bindings: dict[str, list[WorkerResourceBinding]]
    ) -> None:
        cpu_owners: dict[str, list[str]] = defaultdict(list)
        mig_owners: dict[str, list[str]] = defaultdict(list)
        mps_totals: dict[str, int] = defaultdict(int)

        for component, component_bindings in bindings.items():
            for binding in component_bindings:
                owner = f"{component}:{binding.rank}"
                if binding.cpu is not None:
                    for core in binding.cpu.process_cpu_cores:
                        cpu_owners[str(core)].append(owner)
                if binding.gpu is None:
                    continue
                if binding.gpu.mode == "mig" and binding.gpu.mig_device_uuid:
                    mig_owners[binding.gpu.mig_device_uuid].append(owner)
                if (
                    binding.gpu.mode == "mps"
                    and binding.gpu.parent_gpu is not None
                    and binding.gpu.sm_percent > 0
                ):
                    mps_totals[str(binding.gpu.parent_gpu)] += binding.gpu.sm_percent

        self._summary = {
            "shared_cpu_cores": {
                core: owners for core, owners in cpu_owners.items() if len(owners) > 1
            },
            "shared_mig_uuids": {
                uuid: owners for uuid, owners in mig_owners.items() if len(owners) > 1
            },
            "mps_gpu_totals": dict(mps_totals),
        }
