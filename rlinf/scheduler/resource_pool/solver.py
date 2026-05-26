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
from .gpu_binding import validate_sm_percent


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
            self._validate_plan_file_bindings(bindings)
            self._refresh_summary(bindings)
            return bindings
        self._used_mig_uuids = set()
        self._validate_default_cpu_pool_exclusivity()
        self._validate_default_mig_devices()
        components = sorted(
            set(self.pool_cfg.cpu.components) | set(self.pool_cfg.gpu.components)
        )
        bindings = {
            component: self._solve_component(component) for component in components
        }
        self._refresh_summary(bindings)
        return bindings

    def _validate_default_cpu_pool_exclusivity(self) -> None:
        if not self.pool_cfg.cpu.enabled:
            return

        pool_components: dict[str, list[str]] = defaultdict(list)
        for component, request in self.pool_cfg.cpu.components.items():
            pool_components[request.pool].append(component)

        for pool_name, components in pool_components.items():
            if len(components) > 1:
                raise ValueError(
                    f"CPU pool '{pool_name}' is exclusive in default allocation "
                    "mode and cannot be shared by multiple components: "
                    f"{sorted(components)}. Use allocation_mode='plan_file' for "
                    "explicit CPU sharing."
                )

        core_owners: dict[tuple[str, int], str] = {}
        for pool_name, pool in self.pool_cfg.cpu.pools.items():
            for core in parse_cpu_core_set(pool.cores):
                key = (pool.node_group, core)
                previous_owner = core_owners.get(key)
                if previous_owner is not None:
                    raise ValueError(
                        f"CPU core {core} in node group '{pool.node_group}' is "
                        f"claimed by both pools '{previous_owner}' and "
                        f"'{pool_name}'. Default allocation mode is exclusive; "
                        "use allocation_mode='plan_file' for explicit sharing."
                    )
                core_owners[key] = pool_name

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
        cpu_bindings_by_rank = self._solve_cpu_component(component, placements)
        bindings: list[WorkerResourceBinding] = []
        for placement in placements:
            bindings.append(
                WorkerResourceBinding(
                    component=component,
                    rank=int(placement.rank),
                    cluster_node_rank=int(placement.cluster_node_rank),
                    node_group_label=str(placement.node_group_label),
                    cpu=cpu_bindings_by_rank.get(int(placement.rank)),
                    gpu=self._solve_gpu_binding(component, placement),
                )
            )
        return bindings

    def _solve_cpu_component(self, component: str, placements) -> dict[int, CpuBinding]:
        if (
            not self.pool_cfg.cpu.enabled
            or component not in self.pool_cfg.cpu.components
        ):
            return {}

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
        placements_by_node: dict[int, list] = defaultdict(list)
        for placement in placements:
            placements_by_node[int(placement.cluster_node_rank)].append(placement)

        bindings: dict[int, CpuBinding] = {}
        local_env_count = (
            self._component_local_env_count(len(placements))
            if request.granularity == "per_env"
            else 0
        )
        for node_rank in sorted(placements_by_node):
            node_placements = sorted(
                placements_by_node[node_rank], key=lambda placement: placement.rank
            )
            process_core_groups = build_even_split_cpu_groups(
                cores, len(node_placements)
            )
            for placement, process_cpu_cores in zip(
                node_placements, process_core_groups, strict=True
            ):
                env_cpu_core_groups: tuple[tuple[int, ...], ...] = ()
                if request.granularity == "per_env":
                    if component != "env":
                        raise ValueError(
                            "per_env CPU binding is only valid for env component"
                        )
                    env_cpu_core_groups = build_even_split_cpu_groups(
                        process_cpu_cores,
                        local_env_count,
                    )
                    process_cpu_cores = effective_process_affinity(env_cpu_core_groups)
                bindings[int(placement.rank)] = CpuBinding(
                    process_cpu_cores=process_cpu_cores,
                    env_cpu_core_groups=env_cpu_core_groups,
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
            visible_devices = {int(device) for device in placement.visible_accelerators}
            if not visible_devices.issubset(devices):
                raise ValueError(
                    f"GPU pool '{request.pool}' devices {sorted(devices)} do not "
                    f"include all visible devices {sorted(visible_devices)} for "
                    f"{component}:{placement.rank}"
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
            for component, component_bindings in sorted(bindings.items())
        }

    def _validate_plan_file_bindings(
        self, bindings: dict[str, list[WorkerResourceBinding]]
    ) -> None:
        configured_components = set(self.pool_cfg.cpu.components) | set(
            self.pool_cfg.gpu.components
        )
        missing_components = configured_components - set(bindings)
        if missing_components:
            raise ValueError(
                "plan file is missing components configured in resource_pool: "
                f"{sorted(missing_components)}"
            )

        placement_by_component = {}
        for component in bindings:
            try:
                strategy = self.component_placement.get_strategy(component)
            except AssertionError as exc:
                raise ValueError(
                    f"plan file binding references unknown component '{component}'"
                ) from exc
            placement_by_component[component] = {
                placement.rank: placement
                for placement in strategy.get_placement(
                    self.cluster, isolate_accelerator=True
                )
            }

        mig_devices = {
            device.uuid: device
            for pool in self.pool_cfg.gpu.pools.values()
            for device in pool.mig_devices
        }

        seen_owner_ranks: set[tuple[str, int]] = set()
        for component, component_bindings in bindings.items():
            component_placements = placement_by_component[component]
            binding_ranks = {binding.rank for binding in component_bindings}
            placement_ranks = set(component_placements)
            if binding_ranks != placement_ranks:
                missing = sorted(placement_ranks - binding_ranks)
                extra = sorted(binding_ranks - placement_ranks)
                detail = []
                if missing:
                    detail.append(f"missing ranks {missing}")
                if extra:
                    detail.append(f"extra ranks {extra}")
                raise ValueError(
                    f"plan file bindings for component '{component}' must cover "
                    f"all placement ranks; {', '.join(detail)}"
                )
            for binding in component_bindings:
                owner = f"{component}:{binding.rank}"
                owner_key = (component, binding.rank)
                if owner_key in seen_owner_ranks:
                    raise ValueError(f"duplicate plan file binding for {owner}")
                seen_owner_ranks.add(owner_key)

                placement = component_placements.get(binding.rank)
                if placement is None:
                    raise ValueError(
                        f"plan file binding for {owner} targets nonexistent "
                        f"rank {binding.rank}; valid ranks are "
                        f"{sorted(component_placements)}"
                    )
                if binding.cluster_node_rank != placement.cluster_node_rank:
                    raise ValueError(
                        f"plan file binding for {owner} targets cluster node "
                        f"{binding.cluster_node_rank}, but placement uses "
                        f"{placement.cluster_node_rank}"
                    )
                if binding.node_group_label != placement.node_group_label:
                    raise ValueError(
                        f"plan file binding for {owner} targets node group "
                        f"{binding.node_group_label}, but placement uses "
                        f"{placement.node_group_label}"
                    )

                self._validate_plan_file_cpu_binding(
                    owner, binding, len(component_placements)
                )
                self._validate_plan_file_gpu_binding(
                    owner, binding, placement, mig_devices
                )

    def _validate_plan_file_cpu_binding(
        self,
        owner: str,
        binding: WorkerResourceBinding,
        component_worker_count: int,
    ) -> None:
        if binding.cpu is None:
            if binding.component in self.pool_cfg.cpu.components:
                raise ValueError(
                    f"plan file binding for {owner} requires CPU binding because "
                    "the component is configured in resource_pool.cpu.components"
                )
            return

        if not binding.cpu.process_cpu_cores:
            raise ValueError(
                f"plan file CPU binding for {owner} must include process CPU cores"
            )
        self._validate_cpu_core_tuple(
            binding.cpu.process_cpu_cores,
            f"plan file CPU binding for {owner}",
        )

        request = self.pool_cfg.cpu.components.get(binding.component)
        if request is not None:
            pool = self.pool_cfg.cpu.pools[request.pool]
            if binding.node_group_label != pool.node_group:
                raise ValueError(
                    f"plan file CPU binding for {owner} targets node group "
                    f"'{binding.node_group_label}', but CPU pool '{request.pool}' "
                    f"is scoped to '{pool.node_group}'"
                )
            pool_cores = set(parse_cpu_core_set(pool.cores))
            missing = sorted(set(binding.cpu.process_cpu_cores) - pool_cores)
            if missing:
                raise ValueError(
                    f"plan file CPU binding for {owner} uses cores {missing} "
                    f"outside CPU pool '{request.pool}'"
                )

        expects_per_env = request is not None and request.granularity == "per_env"
        has_per_env_groups = bool(binding.cpu.env_cpu_core_groups)
        if not expects_per_env and not has_per_env_groups:
            return
        if binding.component != "env":
            raise ValueError(
                f"plan file per-env CPU binding for {owner} is only valid for env"
            )
        if request is not None and not expects_per_env:
            raise ValueError(
                f"plan file binding for {owner} has env CPU groups, but the "
                "component is not configured with granularity='per_env'"
            )

        expected_env_groups = self._component_local_env_count(component_worker_count)
        if len(binding.cpu.env_cpu_core_groups) != expected_env_groups:
            raise ValueError(
                f"plan file per-env CPU binding for {owner} has "
                f"{len(binding.cpu.env_cpu_core_groups)} env CPU groups, but "
                f"{expected_env_groups} are required"
            )

        process_core_set = set(binding.cpu.process_cpu_cores)
        for index, group in enumerate(binding.cpu.env_cpu_core_groups):
            self._validate_cpu_core_tuple(
                group,
                f"plan file per-env CPU binding for {owner} group {index}",
            )
            missing = sorted(set(group) - process_core_set)
            if missing:
                raise ValueError(
                    f"plan file per-env CPU binding for {owner} group {index} "
                    f"uses cores outside process CPU affinity: {missing}"
                )

    def _validate_cpu_core_tuple(self, cores: tuple[int, ...], context: str) -> None:
        if not cores:
            raise ValueError(f"{context} must not be empty")
        if any(core < 0 for core in cores):
            raise ValueError(f"{context} contains negative CPU core ids")
        if len(set(cores)) != len(cores):
            raise ValueError(f"{context} contains duplicate CPU core ids")

    def _validate_plan_file_gpu_binding(
        self,
        owner: str,
        binding: WorkerResourceBinding,
        placement,
        mig_devices,
    ) -> None:
        if binding.gpu is None:
            if binding.component in self.pool_cfg.gpu.components:
                raise ValueError(
                    f"plan file binding for {owner} requires GPU binding because "
                    "the component is configured in resource_pool.gpu.components"
                )
            return

        validate_sm_percent(binding.gpu.sm_percent)
        if binding.gpu.sm_percent == 0:
            if binding.gpu.mode != "mps":
                raise ValueError(
                    f"plan file binding for {owner} with sm_percent 0 is only "
                    "valid for MPS render-device visibility"
                )
            if not binding.gpu.visible_devices:
                raise ValueError(
                    f"MPS plan file binding for {owner} with sm_percent 0 has "
                    "no render device"
                )
            return

        if binding.gpu.mode == "mps":
            if not binding.gpu.visible_devices:
                raise ValueError(f"MPS plan file binding for {owner} has no devices")
            if tuple(binding.gpu.visible_devices) != tuple(
                placement.visible_accelerators
            ):
                raise ValueError(
                    f"MPS plan file binding for {owner} uses visible devices "
                    f"{binding.gpu.visible_devices}, but placement uses "
                    f"{tuple(placement.visible_accelerators)}"
                )
            if binding.gpu.parent_gpu is not None and str(
                binding.gpu.parent_gpu
            ) != str(placement.local_accelerator_rank):
                raise ValueError(
                    f"MPS plan file binding for {owner} targets parent GPU "
                    f"{binding.gpu.parent_gpu}, but placement uses "
                    f"{placement.local_accelerator_rank}"
                )
            return

        if binding.gpu.mode == "mig":
            if not binding.gpu.mig_device_uuid:
                raise ValueError(f"MIG plan file binding for {owner} has no UUID")
            if binding.gpu.parent_gpu is None:
                raise ValueError(
                    f"MIG plan file binding for {owner} requires parent GPU metadata"
                )
            device = mig_devices.get(binding.gpu.mig_device_uuid)
            if device is None:
                raise ValueError(
                    f"MIG plan file binding for {owner} references unknown UUID "
                    f"{binding.gpu.mig_device_uuid}"
                )
            if binding.gpu.parent_gpu is not None and (
                binding.gpu.parent_gpu != device.parent_gpu
            ):
                raise ValueError(
                    f"MIG plan file binding for {owner} targets parent GPU "
                    f"{binding.gpu.parent_gpu}, but metadata uses "
                    f"{device.parent_gpu}"
                )
            if str(device.parent_gpu) != str(placement.local_accelerator_rank):
                raise ValueError(
                    f"MIG plan file binding for {owner} targets parent GPU "
                    f"{device.parent_gpu}, but placement uses "
                    f"{placement.local_accelerator_rank}"
                )
            if binding.gpu.sm_percent > device.sm_percent:
                raise ValueError(
                    f"MIG plan file binding for {owner} requests sm_percent "
                    f"{binding.gpu.sm_percent}, but metadata for "
                    f"{binding.gpu.mig_device_uuid} provides {device.sm_percent}"
                )
            return

        raise ValueError(f"GPU plan file binding for {owner} has invalid mode")

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
                        cpu_owners[f"node{binding.cluster_node_rank}:cpu{core}"].append(
                            owner
                        )
                if binding.gpu is None:
                    continue
                if binding.gpu.mode == "mig" and binding.gpu.mig_device_uuid:
                    mig_owners[binding.gpu.mig_device_uuid].append(owner)
                if (
                    binding.gpu.mode == "mps"
                    and binding.gpu.parent_gpu is not None
                    and binding.gpu.sm_percent > 0
                ):
                    mps_totals[
                        f"node{binding.cluster_node_rank}:gpu{binding.gpu.parent_gpu}"
                    ] += binding.gpu.sm_percent

        self._summary = {
            "shared_cpu_cores": {
                core: owners for core, owners in cpu_owners.items() if len(owners) > 1
            },
            "shared_mig_uuids": {
                uuid: owners for uuid, owners in mig_owners.items() if len(owners) > 1
            },
            "mps_gpu_totals": dict(mps_totals),
        }
