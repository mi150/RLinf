from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from omegaconf import DictConfig, OmegaConf

from .gpu_binding import validate_sm_percent


@dataclass(frozen=True)
class CpuPoolConfig:
    """CPU resource pool definition."""

    node_group: str = "cluster"
    cores: str = ""


@dataclass(frozen=True)
class CpuComponentConfig:
    """CPU resource assignment for a scheduler component."""

    pool: str
    granularity: Literal["process", "per_env"] = "process"


@dataclass(frozen=True)
class CpuResourceConfig:
    """CPU resource pool configuration."""

    enabled: bool = False
    pools: dict[str, CpuPoolConfig] = field(default_factory=dict)
    components: dict[str, CpuComponentConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class MigDeviceConfig:
    """MIG device slice exposed to a GPU resource pool."""

    uuid: str
    parent_gpu: int
    sm_percent: int


@dataclass(frozen=True)
class GpuPoolConfig:
    """GPU resource pool definition."""

    node_group: str = "cluster"
    devices: str | None = None
    mig_devices: tuple[MigDeviceConfig, ...] = ()


@dataclass(frozen=True)
class GpuComponentConfig:
    """GPU resource assignment for a scheduler component."""

    pool: str
    sm_percent: int = 0


@dataclass(frozen=True)
class GpuResourceConfig:
    """GPU resource pool configuration."""

    enabled: bool = False
    mode: Literal["mps", "mig"] = "mps"
    pools: dict[str, GpuPoolConfig] = field(default_factory=dict)
    components: dict[str, GpuComponentConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class ResourcePoolConfig:
    """Top-level resource pool configuration."""

    enabled: bool = False
    allocation_mode: Literal["default", "plan_file"] = "default"
    allocation_plan_path: str | None = None
    cpu: CpuResourceConfig = field(default_factory=CpuResourceConfig)
    gpu: GpuResourceConfig = field(default_factory=GpuResourceConfig)

    @classmethod
    def from_cluster_cfg(cls, cluster_cfg: DictConfig) -> "ResourcePoolConfig":
        """Build resource pool config from a Hydra cluster config."""
        raw = OmegaConf.select(cluster_cfg, "resource_pool")
        if raw is None:
            return cls()

        enabled = bool(raw.get("enabled", False))
        if not enabled:
            return cls()

        payload = OmegaConf.to_container(raw, resolve=True)
        assert isinstance(payload, Mapping), (
            "cluster.resource_pool must be a mapping after OmegaConf conversion"
        )

        allocation_mode = str(payload.get("allocation_mode", "default"))
        if allocation_mode not in {"default", "plan_file"}:
            raise ValueError(
                "allocation_mode must be one of 'default' or 'plan_file', "
                f"got {allocation_mode}"
            )

        allocation_plan_path = payload.get("allocation_plan_path")
        if allocation_mode == "plan_file" and not allocation_plan_path:
            raise ValueError(
                "allocation_plan_path is required when allocation_mode is 'plan_file'"
            )

        cpu = _parse_cpu_resource(payload.get("cpu"))
        gpu = _parse_gpu_resource(payload.get("gpu"))

        return cls(
            enabled=True,
            allocation_mode=allocation_mode,
            allocation_plan_path=(
                None if allocation_plan_path is None else str(allocation_plan_path)
            ),
            cpu=cpu,
            gpu=gpu,
        )


def _as_mapping(payload: object, context: str) -> Mapping[str, object]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return payload


def _parse_cpu_resource(payload: object) -> CpuResourceConfig:
    config = _as_mapping(payload, "cpu resource config")
    enabled = bool(config.get("enabled", False))
    if not enabled:
        return CpuResourceConfig(enabled=False)

    pools: dict[str, CpuPoolConfig] = {}
    for pool_name, pool_payload in _as_mapping(
        config.get("pools"), "cpu pools"
    ).items():
        pool_config = _as_mapping(pool_payload, f"cpu pool '{pool_name}'")
        pools[str(pool_name)] = CpuPoolConfig(
            node_group=str(pool_config.get("node_group", "cluster")),
            cores=str(pool_config.get("cores", "")),
        )

    components: dict[str, CpuComponentConfig] = {}
    for component_name, component_payload in _as_mapping(
        config.get("components"), "cpu components"
    ).items():
        component_config = _as_mapping(
            component_payload, f"cpu component '{component_name}'"
        )
        pool_name = str(component_config.get("pool", ""))
        if pool_name not in pools:
            raise ValueError(
                f"cpu component '{component_name}' references unknown pool '{pool_name}'"
            )

        granularity = str(component_config.get("granularity", "process"))
        if granularity not in {"process", "per_env"}:
            raise ValueError(
                f"granularity must be one of 'process' or 'per_env', got {granularity}"
            )

        components[str(component_name)] = CpuComponentConfig(
            pool=pool_name,
            granularity=granularity,
        )

    return CpuResourceConfig(enabled=enabled, pools=pools, components=components)


def _parse_gpu_resource(payload: object) -> GpuResourceConfig:
    config = _as_mapping(payload, "gpu resource config")
    enabled = bool(config.get("enabled", False))
    if not enabled:
        return GpuResourceConfig(enabled=False)

    mode = str(config.get("mode", "mps"))
    if mode not in {"mps", "mig"}:
        raise ValueError(f"mode must be one of 'mps' or 'mig', got {mode}")

    pools: dict[str, GpuPoolConfig] = {}
    for pool_name, pool_payload in _as_mapping(
        config.get("pools"), "gpu pools"
    ).items():
        pool_config = _as_mapping(pool_payload, f"gpu pool '{pool_name}'")
        mig_devices_payload = pool_config.get("mig_devices", ())
        mig_devices: list[MigDeviceConfig] = []
        for mig_device_payload in mig_devices_payload or ():
            mig_device_config = _as_mapping(
                mig_device_payload, f"gpu pool '{pool_name}' mig device"
            )
            uuid = str(mig_device_config.get("uuid", ""))
            if not uuid:
                raise ValueError(f"gpu pool '{pool_name}' mig device requires uuid")
            if "parent_gpu" not in mig_device_config:
                raise ValueError(
                    f"gpu pool '{pool_name}' mig device requires parent_gpu"
                )
            if (
                "sm_percent" not in mig_device_config
                or mig_device_config.get("sm_percent") is None
            ):
                raise ValueError(
                    f"gpu pool '{pool_name}' mig device requires sm_percent"
                )
            mig_devices.append(
                MigDeviceConfig(
                    uuid=uuid,
                    parent_gpu=int(mig_device_config.get("parent_gpu", 0)),
                    sm_percent=validate_sm_percent(mig_device_config.get("sm_percent")),
                )
            )
        pools[str(pool_name)] = GpuPoolConfig(
            node_group=str(pool_config.get("node_group", "cluster")),
            devices=(
                None
                if pool_config.get("devices") is None
                else str(pool_config.get("devices"))
            ),
            mig_devices=tuple(mig_devices),
        )

    components: dict[str, GpuComponentConfig] = {}
    for component_name, component_payload in _as_mapping(
        config.get("components"), "gpu components"
    ).items():
        component_config = _as_mapping(
            component_payload, f"gpu component '{component_name}'"
        )
        pool_name = str(component_config.get("pool", ""))
        if pool_name not in pools:
            raise ValueError(
                f"gpu component '{component_name}' references unknown pool '{pool_name}'"
            )
        if (
            "sm_percent" not in component_config
            or component_config.get("sm_percent") is None
        ):
            raise ValueError(f"gpu component '{component_name}' requires sm_percent")

        sm_percent = validate_sm_percent(component_config.get("sm_percent"))
        components[str(component_name)] = GpuComponentConfig(
            pool=pool_name,
            sm_percent=sm_percent,
        )

    return GpuResourceConfig(
        enabled=enabled, mode=mode, pools=pools, components=components
    )
