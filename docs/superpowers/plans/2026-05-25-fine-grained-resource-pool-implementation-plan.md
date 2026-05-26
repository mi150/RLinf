# Fine-Grained Resource Pool 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为 embodied 训练入口实现可选资源池：CPU core 默认独占绑定到 worker，env worker 可把 core 继续绑定到本地子环境，GPU worker 可通过 MPS/MIG 使用 `0/20/40/60/80/100` SM 配额。

**架构：** 保留 `cluster.component_placement` 作为 worker 数量、node 和 accelerator placement 的来源。新增 `rlinf.scheduler.resource_pool` 生成 worker-level binding，`WorkerGroup.launch(resource_bindings=...)` 按 rank 注入 env vars，`Worker` 暴露 binding 元数据，`EnvWorker`/SubprocVectorEnv 系后端应用 CPU affinity。MPS/MIG 只做非破坏性进程环境绑定，不管理 daemon 或 MIG 生命周期。

**技术栈：** Python dataclasses、OmegaConf、Ray runtime env、Linux `os.sched_setaffinity`、CUDA MPS/MIG env vars、pytest、Ruff。

---

## 文件结构

- 创建：`rlinf/scheduler/resource_pool/__init__.py`
  - 导出 resource pool public API。
- 创建：`rlinf/scheduler/resource_pool/bindings.py`
  - 定义 binding dataclasses、JSON round-trip 和 env var 转换。
- 创建：`rlinf/scheduler/resource_pool/cpu_binding.py`
  - CPU core parser、独占切分、per-env group parser、affinity helper。
- 创建：`rlinf/scheduler/resource_pool/gpu_binding.py`
  - SM 百分比枚举校验、MPS/MIG env var builder。
- 创建：`rlinf/scheduler/resource_pool/config.py`
  - 解析 `cluster.resource_pool` 为轻量 dataclasses。
- 创建：`rlinf/scheduler/resource_pool/solver.py`
  - 默认 deterministic solver、plan-file loader、placement/pool 校验、artifact summary。
- 创建：`rlinf/scheduler/resource_pool/pool.py`
  - `FineGrainedResourcePool` 门面：disabled/enabled、component bindings 查询、artifact 写出。
- 修改：`rlinf/scheduler/__init__.py`
  - 导出 `FineGrainedResourcePool` 和 binding 类型。
- 修改：`rlinf/scheduler/worker/worker_group.py`
  - `launch()` 增加 `resource_bindings`，按 rank 合并 env vars。
- 修改：`rlinf/scheduler/worker/worker.py`
  - 解析 `RLINF_RESOURCE_BINDING_JSON`，提供 `resource_binding` property。
- 修改：`rlinf/scheduler/manager/worker_manager.py`
  - `WorkerInfo` 增加 `resource_binding` 字段。
- 修改：`rlinf/workers/env/env_worker.py`
  - EnvWorker/AsyncEnvWorker 进程级 CPU affinity 和 per-env backend 校验。
- 修改：`rlinf/envs/venv/venv.py`
  - 通用 SubprocVectorEnv 子进程在 env factory 前应用 per-env affinity。
- 修改：`rlinf/envs/libero/venv.py`
- 修改：`rlinf/envs/calvin/venv.py`
- 修改：`rlinf/envs/metaworld/venv.py`
- 修改：`rlinf/envs/robocasa/venv.py`
- 修改：`rlinf/envs/habitat/venv.py`
  - 自定义 SubprocVectorEnv worker 传递 local env index 并应用 affinity。
- 修改：`examples/embodiment/train_embodied_agent.py`
- 修改：`examples/embodiment/train_async.py`
  - 创建 resource pool，传入 worker launches，写 artifact。
- 创建：`tests/unit_tests/test_resource_pool_bindings.py`
- 创建：`tests/unit_tests/test_resource_pool_cpu_binding.py`
- 创建：`tests/unit_tests/test_resource_pool_gpu_binding.py`
- 创建：`tests/unit_tests/test_resource_pool_config.py`
- 创建：`tests/unit_tests/test_resource_pool_solver.py`
- 创建：`tests/unit_tests/test_resource_pool_worker_integration.py`
- 创建：`tests/unit_tests/test_resource_pool_env_binding.py`

## 任务 1：Binding 数据模型和 GPU env 契约

**文件：**
- 创建：`rlinf/scheduler/resource_pool/__init__.py`
- 创建：`rlinf/scheduler/resource_pool/bindings.py`
- 创建：`rlinf/scheduler/resource_pool/gpu_binding.py`
- 测试：`tests/unit_tests/test_resource_pool_bindings.py`
- 测试：`tests/unit_tests/test_resource_pool_gpu_binding.py`

- [ ] **步骤 1：编写失败测试：binding round-trip、MPS、zero quota、MIG env**

```python
# tests/unit_tests/test_resource_pool_bindings.py
import json

from rlinf.scheduler.resource_pool.bindings import (
    CPU_AFFINITY_ENV,
    CUDA_VISIBLE_DEVICES_ENV,
    ENV_CPU_CORE_GROUPS_ENV,
    MPS_ACTIVE_THREAD_PERCENTAGE_ENV,
    RESOURCE_BINDING_ENV,
    CpuBinding,
    GpuBinding,
    WorkerResourceBinding,
)


def test_worker_resource_binding_round_trips_and_builds_env() -> None:
    binding = WorkerResourceBinding(
        component="env",
        rank=0,
        cluster_node_rank=1,
        node_group_label="cluster",
        cpu=CpuBinding(
            process_cpu_cores=(0, 1, 2, 3),
            env_cpu_core_groups=((0,), (1,), (2,), (3,)),
        ),
        gpu=GpuBinding(
            mode="mps",
            sm_percent=40,
            visible_devices=("0",),
            mig_device_uuid=None,
            parent_gpu=0,
        ),
    )

    restored = WorkerResourceBinding.from_json(binding.to_json())

    assert restored == binding
    env = binding.to_env_vars()
    assert json.loads(env[RESOURCE_BINDING_ENV])["component"] == "env"
    assert env[CPU_AFFINITY_ENV] == "0,1,2,3"
    assert env[ENV_CPU_CORE_GROUPS_ENV] == "0;1;2;3"
    assert env[CUDA_VISIBLE_DEVICES_ENV] == "0"
    assert env[MPS_ACTIVE_THREAD_PERCENTAGE_ENV] == "40"
```

```python
# tests/unit_tests/test_resource_pool_gpu_binding.py
import pytest

from rlinf.scheduler.resource_pool.bindings import (
    CUDA_VISIBLE_DEVICES_ENV,
    MPS_ACTIVE_THREAD_PERCENTAGE_ENV,
    GpuBinding,
)
from rlinf.scheduler.resource_pool.gpu_binding import (
    build_gpu_env_vars,
    validate_sm_percent,
)


@pytest.mark.parametrize("value", [0, 20, 40, 60, 80, 100])
def test_validate_sm_percent_accepts_supported_values(value: int) -> None:
    assert validate_sm_percent(value) == value


def test_validate_sm_percent_rejects_unsupported_value() -> None:
    with pytest.raises(ValueError, match="sm_percent"):
        validate_sm_percent(25)


def test_mps_zero_quota_does_not_build_gpu_env() -> None:
    binding = GpuBinding(
        mode="mps",
        sm_percent=0,
        visible_devices=("0",),
        parent_gpu=0,
    )
    assert build_gpu_env_vars(binding) == {}


def test_mps_quota_builds_visible_device_and_percentage_env() -> None:
    binding = GpuBinding(
        mode="mps",
        sm_percent=60,
        visible_devices=("1",),
        parent_gpu=1,
    )
    env = build_gpu_env_vars(binding)
    assert env[CUDA_VISIBLE_DEVICES_ENV] == "1"
    assert env[MPS_ACTIVE_THREAD_PERCENTAGE_ENV] == "60"


def test_mig_quota_builds_uuid_visibility() -> None:
    binding = GpuBinding(
        mode="mig",
        sm_percent=20,
        visible_devices=(),
        mig_device_uuid="MIG-abc",
        parent_gpu=0,
    )
    assert build_gpu_env_vars(binding) == {CUDA_VISIBLE_DEVICES_ENV: "MIG-abc"}
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest -q tests/unit_tests/test_resource_pool_bindings.py tests/unit_tests/test_resource_pool_gpu_binding.py
```

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'rlinf.scheduler.resource_pool'`。

- [ ] **步骤 3：实现 binding dataclasses 和 GPU helper**

```python
# rlinf/scheduler/resource_pool/gpu_binding.py
from __future__ import annotations

from .bindings import (
    CUDA_VISIBLE_DEVICES_ENV,
    MPS_ACTIVE_THREAD_PERCENTAGE_ENV,
    GpuBinding,
)

ALLOWED_SM_PERCENTAGES = (0, 20, 40, 60, 80, 100)


def validate_sm_percent(value: int | None) -> int:
    percent = 0 if value is None else int(value)
    if percent not in ALLOWED_SM_PERCENTAGES:
        raise ValueError(
            f"sm_percent must be one of {ALLOWED_SM_PERCENTAGES}, got {percent}"
        )
    return percent


def build_gpu_env_vars(binding: GpuBinding) -> dict[str, str]:
    if binding.sm_percent == 0:
        return {}
    if binding.mode == "mps":
        if not binding.visible_devices:
            raise ValueError("MPS binding requires visible_devices")
        return {
            CUDA_VISIBLE_DEVICES_ENV: ",".join(binding.visible_devices),
            MPS_ACTIVE_THREAD_PERCENTAGE_ENV: str(binding.sm_percent),
        }
    if binding.mode == "mig":
        if not binding.mig_device_uuid:
            raise ValueError("MIG binding requires mig_device_uuid")
        return {CUDA_VISIBLE_DEVICES_ENV: binding.mig_device_uuid}
    return {}
```

```python
# rlinf/scheduler/resource_pool/bindings.py
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
    process_cpu_cores: tuple[int, ...] = ()
    env_cpu_core_groups: tuple[tuple[int, ...], ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CpuBinding | None":
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
    mode: Literal["mps", "mig"] | None = None
    sm_percent: int = 0
    visible_devices: tuple[str, ...] = ()
    mig_device_uuid: str | None = None
    parent_gpu: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GpuBinding | None":
        if payload is None:
            return None
        return cls(
            mode=payload.get("mode"),
            sm_percent=int(payload.get("sm_percent", 0)),
            visible_devices=tuple(str(v) for v in payload.get("visible_devices", ())),
            mig_device_uuid=payload.get("mig_device_uuid"),
            parent_gpu=payload.get("parent_gpu"),
        )


@dataclass(frozen=True)
class WorkerResourceBinding:
    component: str
    rank: int
    cluster_node_rank: int
    node_group_label: str
    cpu: CpuBinding | None = None
    gpu: GpuBinding | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "WorkerResourceBinding":
        payload = json.loads(text)
        return cls(
            component=str(payload["component"]),
            rank=int(payload["rank"]),
            cluster_node_rank=int(payload["cluster_node_rank"]),
            node_group_label=str(payload["node_group_label"]),
            cpu=CpuBinding.from_dict(payload.get("cpu")),
            gpu=GpuBinding.from_dict(payload.get("gpu")),
        )

    def to_env_vars(self) -> dict[str, str]:
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
```

```python
# rlinf/scheduler/resource_pool/__init__.py
from .bindings import CpuBinding, GpuBinding, WorkerResourceBinding

__all__ = ["CpuBinding", "GpuBinding", "WorkerResourceBinding"]
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest -q tests/unit_tests/test_resource_pool_bindings.py tests/unit_tests/test_resource_pool_gpu_binding.py
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/scheduler/resource_pool tests/unit_tests/test_resource_pool_bindings.py tests/unit_tests/test_resource_pool_gpu_binding.py
git commit -s -m "feat: add resource binding data model"
```

## 任务 2：CPU core parser、独占切分和 affinity helper

**文件：**
- 创建：`rlinf/scheduler/resource_pool/cpu_binding.py`
- 修改：`rlinf/scheduler/resource_pool/__init__.py`
- 测试：`tests/unit_tests/test_resource_pool_cpu_binding.py`

- [ ] **步骤 1：编写失败测试**

```python
# tests/unit_tests/test_resource_pool_cpu_binding.py
import pytest

from rlinf.scheduler.resource_pool.cpu_binding import (
    apply_process_cpu_affinity,
    build_even_split_cpu_groups,
    effective_process_affinity,
    get_env_core_group_from_env,
    parse_cpu_core_set,
    parse_env_cpu_core_groups,
)


def test_parse_cpu_core_set_supports_ranges_and_values() -> None:
    assert parse_cpu_core_set("0-3,8,10-11") == (0, 1, 2, 3, 8, 10, 11)


def test_parse_cpu_core_set_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_cpu_core_set("0-2,2")


def test_build_even_split_cpu_groups_distributes_remainder() -> None:
    assert build_even_split_cpu_groups(tuple(range(10)), partitions=3) == (
        (0, 1, 2, 3),
        (4, 5, 6),
        (7, 8, 9),
    )


def test_build_even_split_cpu_groups_requires_one_core_per_partition() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_even_split_cpu_groups((0, 1), partitions=3)


def test_parse_env_cpu_core_groups_and_lookup_by_index() -> None:
    assert parse_env_cpu_core_groups("0;1,2;3") == ((0,), (1, 2), (3,))
    assert get_env_core_group_from_env(
        {"RLINF_ENV_CPU_CORE_GROUPS": "0;1,2;3"}, local_env_index=1
    ) == (1, 2)


def test_effective_process_affinity_unions_groups() -> None:
    assert effective_process_affinity(((3, 1), (2,))) == (1, 2, 3)


def test_apply_process_cpu_affinity_calls_sched_setaffinity(monkeypatch) -> None:
    captured = {}

    def fake_sched_setaffinity(pid: int, cpus: set[int]) -> None:
        captured["pid"] = pid
        captured["cpus"] = cpus

    monkeypatch.setattr("os.sched_setaffinity", fake_sched_setaffinity, raising=False)
    apply_process_cpu_affinity((0, 2))

    assert captured == {"pid": 0, "cpus": {0, 2}}
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest -q tests/unit_tests/test_resource_pool_cpu_binding.py
```

预期：FAIL，报错包含 `cannot import name` 或缺少 `cpu_binding` 模块。

- [ ] **步骤 3：实现 CPU helper**

```python
# rlinf/scheduler/resource_pool/cpu_binding.py
from __future__ import annotations

import os
from collections.abc import Mapping

from .bindings import ENV_CPU_CORE_GROUPS_ENV


def parse_cpu_core_set(spec: str) -> tuple[int, ...]:
    text = str(spec).strip()
    if not text:
        raise ValueError("cpu core spec must not be empty")
    cores: list[int] = []
    for raw_token in text.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError("cpu core spec contains an empty token")
        if "-" in token:
            if token.count("-") != 1:
                raise ValueError(f"invalid cpu range '{token}'")
            start_text, end_text = token.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"invalid cpu range '{token}'")
            cores.extend(range(start, end + 1))
        else:
            core = int(token)
            if core < 0:
                raise ValueError(f"cpu core id must be >= 0, got {core}")
            cores.append(core)
    normalized = tuple(sorted(cores))
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate cpu cores are not allowed")
    return normalized


def build_even_split_cpu_groups(
    cores: tuple[int, ...], partitions: int
) -> tuple[tuple[int, ...], ...]:
    if partitions <= 0:
        raise ValueError("partitions must be > 0")
    if len(set(cores)) != len(cores):
        raise ValueError("duplicate cpu cores are not allowed")
    if len(cores) < partitions:
        raise ValueError("each partition must receive at least one cpu core")
    base = len(cores) // partitions
    remainder = len(cores) % partitions
    groups: list[tuple[int, ...]] = []
    cursor = 0
    for idx in range(partitions):
        size = base + (1 if idx < remainder else 0)
        groups.append(tuple(cores[cursor : cursor + size]))
        cursor += size
    return tuple(groups)


def effective_process_affinity(groups: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return tuple(sorted({core for group in groups for core in group}))


def parse_env_cpu_core_groups(spec: str) -> tuple[tuple[int, ...], ...]:
    text = str(spec).strip()
    if not text:
        return ()
    return tuple(parse_cpu_core_set(group) for group in text.split(";"))


def get_env_core_group_from_env(
    env: Mapping[str, str], local_env_index: int
) -> tuple[int, ...] | None:
    spec = env.get(ENV_CPU_CORE_GROUPS_ENV)
    if not spec:
        return None
    groups = parse_env_cpu_core_groups(spec)
    if local_env_index < 0 or local_env_index >= len(groups):
        raise ValueError(
            f"missing cpu core group for local env index {local_env_index}; "
            f"available groups: {len(groups)}"
        )
    return groups[local_env_index]


def apply_process_cpu_affinity(cpus: tuple[int, ...]) -> None:
    if not cpus:
        raise ValueError("cpu affinity set must not be empty")
    if not hasattr(os, "sched_setaffinity"):
        raise NotImplementedError("os.sched_setaffinity is unavailable")
    os.sched_setaffinity(0, set(cpus))
```

```python
# rlinf/scheduler/resource_pool/__init__.py
from .bindings import CpuBinding, GpuBinding, WorkerResourceBinding
from .cpu_binding import apply_process_cpu_affinity

__all__ = [
    "CpuBinding",
    "GpuBinding",
    "WorkerResourceBinding",
    "apply_process_cpu_affinity",
]
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest -q tests/unit_tests/test_resource_pool_cpu_binding.py
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/scheduler/resource_pool tests/unit_tests/test_resource_pool_cpu_binding.py
git commit -s -m "feat: add cpu resource binding helpers"
```

## 任务 3：Resource pool config parser

**文件：**
- 创建：`rlinf/scheduler/resource_pool/config.py`
- 测试：`tests/unit_tests/test_resource_pool_config.py`

- [ ] **步骤 1：编写失败测试**

```python
# tests/unit_tests/test_resource_pool_config.py
import pytest
from omegaconf import OmegaConf

from rlinf.scheduler.resource_pool.config import ResourcePoolConfig


def test_resource_pool_config_disabled_when_missing() -> None:
    cfg = OmegaConf.create({"cluster": {"num_nodes": 1}})
    parsed = ResourcePoolConfig.from_cluster_cfg(cfg.cluster)
    assert parsed.enabled is False


def test_resource_pool_config_parses_cpu_and_mps_components() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "resource_pool": {
                    "enabled": True,
                    "allocation_mode": "default",
                    "cpu": {
                        "enabled": True,
                        "pools": {"env_cpu": {"node_group": "node", "cores": "0-7"}},
                        "components": {
                            "env": {"pool": "env_cpu", "granularity": "per_env"}
                        },
                    },
                    "gpu": {
                        "enabled": True,
                        "mode": "mps",
                        "pools": {"gpu_pool": {"node_group": "cluster", "devices": "0-1"}},
                        "components": {"rollout": {"pool": "gpu_pool", "sm_percent": 40}},
                    },
                }
            }
        }
    )

    parsed = ResourcePoolConfig.from_cluster_cfg(cfg.cluster)

    assert parsed.enabled is True
    assert parsed.cpu.components["env"].granularity == "per_env"
    assert parsed.gpu.components["rollout"].sm_percent == 40


def test_resource_pool_config_rejects_invalid_sm_percent() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "resource_pool": {
                    "enabled": True,
                    "gpu": {
                        "enabled": True,
                        "mode": "mps",
                        "pools": {"gpu_pool": {"devices": "0"}},
                        "components": {"rollout": {"pool": "gpu_pool", "sm_percent": 25}},
                    },
                }
            }
        }
    )

    with pytest.raises(ValueError, match="sm_percent"):
        ResourcePoolConfig.from_cluster_cfg(cfg.cluster)


def test_plan_file_mode_requires_path() -> None:
    cfg = OmegaConf.create(
        {"cluster": {"resource_pool": {"enabled": True, "allocation_mode": "plan_file"}}}
    )
    with pytest.raises(ValueError, match="allocation_plan_path"):
        ResourcePoolConfig.from_cluster_cfg(cfg.cluster)
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest -q tests/unit_tests/test_resource_pool_config.py
```

预期：FAIL，报错包含 `No module named 'rlinf.scheduler.resource_pool.config'`。

- [ ] **步骤 3：实现 config dataclasses**

```python
# rlinf/scheduler/resource_pool/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf

from .gpu_binding import validate_sm_percent


@dataclass(frozen=True)
class CpuPoolConfig:
    node_group: str = "cluster"
    cores: str = ""


@dataclass(frozen=True)
class CpuComponentConfig:
    pool: str
    granularity: Literal["process", "per_env"] = "process"


@dataclass(frozen=True)
class CpuResourceConfig:
    enabled: bool = False
    pools: dict[str, CpuPoolConfig] = field(default_factory=dict)
    components: dict[str, CpuComponentConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class MigDeviceConfig:
    uuid: str
    parent_gpu: int
    sm_percent: int


@dataclass(frozen=True)
class GpuPoolConfig:
    node_group: str = "cluster"
    devices: str | None = None
    mig_devices: tuple[MigDeviceConfig, ...] = ()


@dataclass(frozen=True)
class GpuComponentConfig:
    pool: str
    sm_percent: int = 0


@dataclass(frozen=True)
class GpuResourceConfig:
    enabled: bool = False
    mode: Literal["mps", "mig"] = "mps"
    pools: dict[str, GpuPoolConfig] = field(default_factory=dict)
    components: dict[str, GpuComponentConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class ResourcePoolConfig:
    enabled: bool = False
    allocation_mode: Literal["default", "plan_file"] = "default"
    allocation_plan_path: str | None = None
    cpu: CpuResourceConfig = field(default_factory=CpuResourceConfig)
    gpu: GpuResourceConfig = field(default_factory=GpuResourceConfig)

    @classmethod
    def from_cluster_cfg(cls, cluster_cfg: DictConfig) -> "ResourcePoolConfig":
        raw = getattr(cluster_cfg, "resource_pool", None)
        if raw is None or not bool(raw.get("enabled", False)):
            return cls(enabled=False)
        payload = OmegaConf.to_container(raw, resolve=True)
        assert isinstance(payload, dict)
        allocation_mode = str(payload.get("allocation_mode", "default"))
        if allocation_mode not in {"default", "plan_file"}:
            raise ValueError(f"unsupported allocation_mode: {allocation_mode}")
        plan_path = payload.get("allocation_plan_path")
        if allocation_mode == "plan_file" and not plan_path:
            raise ValueError("allocation_plan_path is required for plan_file mode")
        return cls(
            enabled=True,
            allocation_mode=allocation_mode,
            allocation_plan_path=str(plan_path) if plan_path else None,
            cpu=_parse_cpu(payload.get("cpu") or {}),
            gpu=_parse_gpu(payload.get("gpu") or {}),
        )


def _parse_cpu(payload: dict[str, Any]) -> CpuResourceConfig:
    pools = {
        name: CpuPoolConfig(
            node_group=str(pool.get("node_group", "cluster")),
            cores=str(pool["cores"]),
        )
        for name, pool in (payload.get("pools") or {}).items()
    }
    components = {
        name: CpuComponentConfig(
            pool=str(component["pool"]),
            granularity=str(component.get("granularity", "process")),
        )
        for name, component in (payload.get("components") or {}).items()
    }
    for component_name, component in components.items():
        if component.pool not in pools:
            raise ValueError(
                f"CPU component {component_name} references unknown pool {component.pool}"
            )
        if component.granularity not in {"process", "per_env"}:
            raise ValueError(f"unsupported CPU granularity: {component.granularity}")
    return CpuResourceConfig(
        enabled=bool(payload.get("enabled", False)),
        pools=pools,
        components=components,
    )


def _parse_gpu(payload: dict[str, Any]) -> GpuResourceConfig:
    mode = str(payload.get("mode", "mps"))
    if mode not in {"mps", "mig"}:
        raise ValueError(f"unsupported gpu resource mode: {mode}")
    pools: dict[str, GpuPoolConfig] = {}
    for name, pool in (payload.get("pools") or {}).items():
        mig_devices = tuple(
            MigDeviceConfig(
                uuid=str(device["uuid"]),
                parent_gpu=int(device["parent_gpu"]),
                sm_percent=validate_sm_percent(device["sm_percent"]),
            )
            for device in pool.get("mig_devices", ())
        )
        pools[name] = GpuPoolConfig(
            node_group=str(pool.get("node_group", "cluster")),
            devices=str(pool["devices"]) if "devices" in pool else None,
            mig_devices=mig_devices,
        )
    components = {
        name: GpuComponentConfig(
            pool=str(component["pool"]),
            sm_percent=validate_sm_percent(component.get("sm_percent", 0)),
        )
        for name, component in (payload.get("components") or {}).items()
    }
    for component_name, component in components.items():
        if component.pool not in pools:
            raise ValueError(
                f"GPU component {component_name} references unknown pool {component.pool}"
            )
    return GpuResourceConfig(
        enabled=bool(payload.get("enabled", False)),
        mode=mode,
        pools=pools,
        components=components,
    )
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest -q tests/unit_tests/test_resource_pool_config.py
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/scheduler/resource_pool/config.py tests/unit_tests/test_resource_pool_config.py
git commit -s -m "feat: parse resource pool config"
```

## 任务 4：默认 solver、plan-file loader 和 artifact summary

**文件：**
- 创建：`rlinf/scheduler/resource_pool/solver.py`
- 创建：`rlinf/scheduler/resource_pool/pool.py`
- 修改：`rlinf/scheduler/resource_pool/__init__.py`
- 修改：`rlinf/scheduler/__init__.py`
- 测试：`tests/unit_tests/test_resource_pool_solver.py`

- [ ] **步骤 1：编写失败测试：默认 CPU per-env、MPS zero quota、MIG 独占、plan-file 共享**

```python
# tests/unit_tests/test_resource_pool_solver.py
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rlinf.scheduler.resource_pool.config import ResourcePoolConfig
from rlinf.scheduler.resource_pool.pool import FineGrainedResourcePool
from rlinf.scheduler.resource_pool.solver import ResourcePoolSolver
from rlinf.utils.placement import HybridComponentPlacement
from tests.unit_tests.test_placement import create_fake_cluster


def test_default_solver_builds_env_per_env_cpu_groups() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {"env": {"node_group": "node", "placement": "0:0-1"}},
                "resource_pool": {
                    "enabled": True,
                    "cpu": {
                        "enabled": True,
                        "pools": {"env_cpu": {"node_group": "node", "cores": "0-7"}},
                        "components": {"env": {"pool": "env_cpu", "granularity": "per_env"}},
                    },
                },
            },
            "env": {"train": {"total_num_envs": 4}, "eval": {"total_num_envs": 4}},
            "runner": {"only_eval": False, "val_check_interval": -1},
            "rollout": {"pipeline_stage_num": 1},
        }
    )
    cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=0)
    placement = HybridComponentPlacement(cfg, cluster)
    pool_cfg = ResourcePoolConfig.from_cluster_cfg(cfg.cluster)

    bindings = ResourcePoolSolver(pool_cfg, cfg, cluster, placement).solve()

    assert [binding.rank for binding in bindings["env"]] == [0, 1]
    assert bindings["env"][0].cpu.process_cpu_cores == (0, 1, 2, 3)
    assert bindings["env"][0].cpu.env_cpu_core_groups == ((0, 1), (2, 3))
    assert bindings["env"][1].cpu.process_cpu_cores == (4, 5, 6, 7)
    assert bindings["env"][1].cpu.env_cpu_core_groups == ((4, 5), (6, 7))


def test_default_solver_skips_gpu_binding_for_zero_quota() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {"env": "0"},
                "resource_pool": {
                    "enabled": True,
                    "gpu": {
                        "enabled": True,
                        "mode": "mps",
                        "pools": {"gpu_pool": {"node_group": "cluster", "devices": "0"}},
                        "components": {"env": {"pool": "gpu_pool", "sm_percent": 0}},
                    },
                },
            },
            "env": {"train": {"total_num_envs": 1}, "eval": {"total_num_envs": 1}},
            "runner": {"only_eval": False, "val_check_interval": -1},
            "rollout": {"pipeline_stage_num": 1},
        }
    )
    cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=1)
    placement = HybridComponentPlacement(cfg, cluster)
    pool_cfg = ResourcePoolConfig.from_cluster_cfg(cfg.cluster)

    bindings = ResourcePoolSolver(pool_cfg, cfg, cluster, placement).solve()

    assert bindings["env"][0].gpu is None


def test_default_solver_rejects_duplicate_mig_uuid() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {"rollout": "0-1"},
                "resource_pool": {
                    "enabled": True,
                    "gpu": {
                        "enabled": True,
                        "mode": "mig",
                        "pools": {
                            "mig_pool": {
                                "node_group": "cluster",
                                "mig_devices": [
                                    {"uuid": "MIG-A", "parent_gpu": 0, "sm_percent": 20},
                                    {"uuid": "MIG-A", "parent_gpu": 1, "sm_percent": 20},
                                ],
                            }
                        },
                        "components": {"rollout": {"pool": "mig_pool", "sm_percent": 20}},
                    },
                },
            },
            "env": {"train": {"total_num_envs": 1}, "eval": {"total_num_envs": 1}},
            "runner": {"only_eval": False, "val_check_interval": -1},
            "rollout": {"pipeline_stage_num": 1},
        }
    )
    cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=2)
    placement = HybridComponentPlacement(cfg, cluster)
    pool_cfg = ResourcePoolConfig.from_cluster_cfg(cfg.cluster)

    with pytest.raises(ValueError, match="Duplicate MIG UUID"):
        ResourcePoolSolver(pool_cfg, cfg, cluster, placement).solve()


def test_plan_file_mode_allows_explicit_cpu_sharing(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "bindings": [
                    {
                        "component": "env",
                        "rank": 0,
                        "cluster_node_rank": 0,
                        "node_group_label": "node",
                        "cpu": {"process_cpu_cores": [0, 1], "env_cpu_core_groups": []},
                        "gpu": None,
                    },
                    {
                        "component": "env",
                        "rank": 1,
                        "cluster_node_rank": 0,
                        "node_group_label": "node",
                        "cpu": {"process_cpu_cores": [0, 1], "env_cpu_core_groups": []},
                        "gpu": None,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {"env": {"node_group": "node", "placement": "0:0-1"}},
                "resource_pool": {
                    "enabled": True,
                    "allocation_mode": "plan_file",
                    "allocation_plan_path": str(plan_path),
                },
            },
            "env": {"train": {"total_num_envs": 2}, "eval": {"total_num_envs": 2}},
            "runner": {"only_eval": False, "val_check_interval": -1},
            "rollout": {"pipeline_stage_num": 1},
        }
    )
    cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=0)
    placement = HybridComponentPlacement(cfg, cluster)

    pool = FineGrainedResourcePool.from_config(cfg, cluster, placement)

    assert pool.get_component_bindings("env")[0].cpu.process_cpu_cores == (0, 1)
    assert pool.summary["shared_cpu_cores"] == {"0": ["env:0", "env:1"], "1": ["env:0", "env:1"]}
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest -q tests/unit_tests/test_resource_pool_solver.py
```

预期：FAIL，报错包含缺少 `solver` 或 `pool` 模块。

- [ ] **步骤 3：实现 solver 和 pool 门面**

创建 `rlinf/scheduler/resource_pool/solver.py`。文件开头和 `ResourcePoolSolver` 公共结构使用以下代码；本步骤下面列出的私有方法要求也必须在同一个文件中实现，不能改 public 方法名：

```python
# rlinf/scheduler/resource_pool/solver.py
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
    def __init__(
        self,
        pool_cfg: ResourcePoolConfig,
        cfg: DictConfig,
        cluster,
        component_placement,
    ) -> None:
        self.pool_cfg = pool_cfg
        self.cfg = cfg
        self.cluster = cluster
        self.component_placement = component_placement
        self._summary: dict[str, object] = {
            "shared_cpu_cores": {},
            "shared_mig_uuids": {},
            "mps_gpu_totals": {},
        }

    @property
    def summary(self) -> dict[str, object]:
        return self._summary

    def solve(self) -> dict[str, list[WorkerResourceBinding]]:
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
```

`_component_local_env_count()` 使用以下精确逻辑：

```python
def _component_local_env_count(self, worker_count: int) -> int:
    stage_num = int(self.cfg.rollout.get("pipeline_stage_num", 1))
    if bool(getattr(self.cfg.runner, "only_eval", False)):
        total_envs = int(self.cfg.env.eval.total_num_envs)
    else:
        total_envs = int(self.cfg.env.train.total_num_envs)
    if total_envs % worker_count != 0:
        raise ValueError("env total_num_envs must be divisible by env worker count")
    return total_envs // worker_count
```

`_component_local_env_count()` 返回一个 EnvWorker across all local stages 的子环境数量。后续 EnvWorker 会按 stage 切片 `env_cpu_core_groups`，每个 stage 只把自己的 group 子集传给当次创建的 VectorEnv。

`_solve_component(component)` 要按 rank 排序 placement，并为每个 rank 生成 `WorkerResourceBinding`：

```python
def _solve_component(self, component: str) -> list[WorkerResourceBinding]:
    placements = self.component_placement.get_strategy(component).get_placement(
        self.cluster, isolate_accelerator=True
    )
    placements = sorted(placements, key=lambda placement: placement.rank)
    cpu_bindings = self._solve_cpu_component(component, placements)
    result = []
    for index, placement in enumerate(placements):
        result.append(
            WorkerResourceBinding(
                component=component,
                rank=placement.rank,
                cluster_node_rank=placement.cluster_node_rank,
                node_group_label=placement.node_group_label,
                cpu=cpu_bindings[index] if cpu_bindings else None,
                gpu=self._solve_gpu_binding(component, placement),
            )
        )
    return result
```

`_solve_cpu_component(component, placements)` 要：

- 如果 CPU 未启用或 component 无 CPU request，返回 `[]`。
- 校验每个 placement 的 `node_group_label` 等于 CPU pool 的 `node_group`。
- 用 `parse_cpu_core_set(pool.cores)` 解析 pool。
- 用 `build_even_split_cpu_groups(cores, len(placements))` 给 worker 严格切分 core。
- 当 `component == "env"` 且 `granularity == "per_env"` 时，用 `_component_local_env_count(len(placements))` 计算本 worker 的 local env 数，并把 worker core group 再切为 `env_cpu_core_groups`。
- 当其他 component 使用 `granularity == "per_env"` 时抛出 `ValueError("per_env CPU binding is only valid for env component")`。

`_solve_gpu_binding(component, placement)` 要：

- 如果 GPU 未启用或 component 无 GPU request，返回 `None`。
- 如果 `request.sm_percent == 0`，返回 `None`。
- 校验 placement 的 `node_group_label` 等于 GPU pool 的 `node_group`。
- MPS 模式：解析 `pool.devices`，校验 `placement.local_accelerator_rank` 在 pool devices 内，返回 `GpuBinding(mode="mps", sm_percent=request.sm_percent, visible_devices=tuple(placement.visible_accelerators), parent_gpu=placement.local_accelerator_rank)`。
- MIG 模式：在 pool 的 `mig_devices` 中选择第一个未使用且 `device.parent_gpu == placement.local_accelerator_rank` 且 `device.sm_percent >= request.sm_percent` 的设备；找不到则抛出包含 `"No MIG device"` 的 `ValueError`；找到后返回 `GpuBinding(mode="mig", sm_percent=request.sm_percent, mig_device_uuid=device.uuid, parent_gpu=device.parent_gpu)`。

`_validate_default_mig_devices()` 要在默认模式下拒绝重复 MIG UUID。`_load_plan_file()` 要读取 JSON object 的 `bindings` list，用 `WorkerResourceBinding.from_json(json.dumps(item))` 还原，并按 component/rank 排序。`_refresh_summary(bindings)` 要统计：

- `shared_cpu_cores`: core 字符串到 `["component:rank", ...]`，只保留出现次数大于 1 的 core。
- `shared_mig_uuids`: UUID 到 `["component:rank", ...]`，只保留出现次数大于 1 的 UUID。
- `mps_gpu_totals`: parent GPU 字符串到非零 MPS `sm_percent` 总和。

```python
# rlinf/scheduler/resource_pool/pool.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .bindings import WorkerResourceBinding
from .config import ResourcePoolConfig
from .solver import ResourcePoolSolver


@dataclass
class FineGrainedResourcePool:
    enabled: bool = False
    bindings: dict[str, list[WorkerResourceBinding]] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)

    @classmethod
    def disabled(cls) -> "FineGrainedResourcePool":
        return cls(enabled=False)

    @classmethod
    def from_config(cls, cfg, cluster, component_placement) -> "FineGrainedResourcePool":
        pool_cfg = ResourcePoolConfig.from_cluster_cfg(cfg.cluster)
        if not pool_cfg.enabled:
            return cls.disabled()
        solver = ResourcePoolSolver(pool_cfg, cfg, cluster, component_placement)
        bindings = solver.solve()
        return cls(enabled=True, bindings=bindings, summary=solver.summary)

    def get_component_bindings(
        self, component: str
    ) -> list[WorkerResourceBinding] | None:
        if not self.enabled:
            return None
        return self.bindings.get(component)

    def write_plan(self, path: str | Path) -> None:
        payload = {
            "enabled": self.enabled,
            "bindings": [
                json.loads(binding.to_json())
                for component in sorted(self.bindings)
                for binding in self.bindings[component]
            ],
            "summary": self.summary,
        }
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
```

导出：

```python
# rlinf/scheduler/resource_pool/__init__.py
from .bindings import CpuBinding, GpuBinding, WorkerResourceBinding
from .cpu_binding import apply_process_cpu_affinity
from .pool import FineGrainedResourcePool

__all__ = [
    "CpuBinding",
    "FineGrainedResourcePool",
    "GpuBinding",
    "WorkerResourceBinding",
    "apply_process_cpu_affinity",
]
```

```python
# rlinf/scheduler/__init__.py
from .resource_pool import FineGrainedResourcePool

__all__ = [
    # keep existing names
    "FineGrainedResourcePool",
]
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest -q tests/unit_tests/test_resource_pool_solver.py
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/scheduler/resource_pool rlinf/scheduler/__init__.py tests/unit_tests/test_resource_pool_solver.py
git commit -s -m "feat: solve resource pool bindings"
```

## 任务 5：WorkerGroup 注入和 WorkerInfo 传播

**文件：**
- 修改：`rlinf/scheduler/worker/worker_group.py`
- 修改：`rlinf/scheduler/worker/worker.py`
- 修改：`rlinf/scheduler/manager/worker_manager.py`
- 测试：`tests/unit_tests/test_resource_pool_worker_integration.py`

- [ ] **步骤 1：编写失败测试**

```python
# tests/unit_tests/test_resource_pool_worker_integration.py
from dataclasses import asdict
from unittest import mock
from unittest.mock import Mock

from rlinf.scheduler.placement.placement import Placement
from rlinf.scheduler.resource_pool.bindings import (
    RESOURCE_BINDING_ENV,
    WorkerResourceBinding,
)
from rlinf.scheduler.worker.worker import Worker
from rlinf.scheduler.worker.worker_group import WorkerGroup


class DummyWorker:
    pass


class DummyPlacementStrategy:
    def get_placement(self, cluster, isolate_accelerator=True):
        return [
            Placement(
                rank=0,
                cluster_node_rank=0,
                placement_node_rank=0,
                local_accelerator_rank=-1,
                accelerator_type="NO_ACCEL",
                local_rank=0,
                local_world_size=1,
                visible_accelerators=[],
                isolate_accelerator=True,
                local_hardware_ranks=[],
                node_group_label="node",
            )
        ]


def test_worker_group_injects_resource_binding_env() -> None:
    cluster = Mock()
    cluster.get_node_ip.return_value = "127.0.0.1"
    cluster.get_node_info.return_value.accelerator_type = "NO_ACCEL"
    cluster.get_node_info.return_value.accelerator_model = ""
    cluster.get_node_group.return_value.get_node_env_vars.return_value = {}
    cluster.get_node_group.return_value.get_node_python_interpreter_path.return_value = None
    cluster.allocate.return_value = object()

    binding = WorkerResourceBinding(
        component="env",
        rank=0,
        cluster_node_rank=0,
        node_group_label="node",
    )
    group = WorkerGroup(DummyWorker, args=(), kwargs={})
    group._cluster = cluster
    group._placement_strategy = DummyPlacementStrategy()
    group._isolate_gpu = True
    group._catch_system_failure = False
    group._max_concurrency = None
    group._disable_distributed_log = False
    group._extra_env_vars = {}
    group._resource_bindings_by_rank = {0: binding}

    group._create_workers()

    env_vars = cluster.allocate.call_args.kwargs["env_vars"]
    assert RESOURCE_BINDING_ENV in env_vars


def test_worker_parses_resource_binding_from_env() -> None:
    binding = WorkerResourceBinding(
        component="env",
        rank=2,
        cluster_node_rank=0,
        node_group_label="node",
    )
    worker = object.__new__(Worker)

    with mock.patch.dict("os.environ", {RESOURCE_BINDING_ENV: binding.to_json()}):
        worker._setup_resource_binding()

    assert worker.resource_binding == binding
    assert worker._resource_binding_dict == asdict(binding)
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest -q tests/unit_tests/test_resource_pool_worker_integration.py
```

预期：FAIL，报错包含 `_setup_resource_binding` 或 `_resource_bindings_by_rank` 不存在。

- [ ] **步骤 3：修改 WorkerGroup**

在 `WorkerGroup.__init__()` 中初始化：

```python
self._resource_bindings_by_rank = {}
```

在 `launch()` 签名增加：

```python
resource_bindings: Optional[list[WorkerResourceBinding]] = None,
```

在 `launch()` 保存绑定：

```python
self._resource_bindings_by_rank = {}
if resource_bindings is not None:
    self._resource_bindings_by_rank = {
        binding.rank: binding for binding in resource_bindings
    }
    assert len(self._resource_bindings_by_rank) == len(resource_bindings), (
        "Duplicate resource binding ranks are not allowed."
    )
```

在 `_create_workers()` 获取 placements 后校验：

```python
placement_ranks = {placement.rank for placement in placements}
binding_ranks = set(self._resource_bindings_by_rank)
assert binding_ranks.issubset(placement_ranks), (
    f"Resource binding ranks {binding_ranks} are not a subset of "
    f"placement ranks {placement_ranks}."
)
```

在 `env_vars.update(AcceleratorUtil...)` 后、`self._extra_env_vars` 前合并 binding：

```python
binding = self._resource_bindings_by_rank.get(placement.rank)
if binding is not None:
    assert binding.cluster_node_rank == placement.cluster_node_rank, (
        f"Resource binding for rank {placement.rank} targets node "
        f"{binding.cluster_node_rank}, but placement uses node "
        f"{placement.cluster_node_rank}."
    )
    assert binding.node_group_label == placement.node_group_label, (
        f"Resource binding for rank {placement.rank} targets node group "
        f"{binding.node_group_label}, but placement uses "
        f"{placement.node_group_label}."
    )
    env_vars.update(binding.to_env_vars())
env_vars.update(self._extra_env_vars)
```

- [ ] **步骤 4：修改 Worker 和 WorkerInfo**

```python
# rlinf/scheduler/manager/worker_manager.py
resource_binding: dict | None = None
"""Fine-grained resource binding metadata for this worker."""
```

```python
# rlinf/scheduler/worker/worker.py
from dataclasses import asdict
```

在 `Worker.__init__()` 中 `_setup_worker_info()` 前加入：

```python
self._setup_resource_binding()
```

增加 property 和 helper：

```python
@property
def resource_binding(self):
    return self._resource_binding

def _setup_resource_binding(self) -> None:
    from ..resource_pool.bindings import RESOURCE_BINDING_ENV, WorkerResourceBinding

    binding_json = os.environ.get(RESOURCE_BINDING_ENV)
    self._resource_binding = (
        WorkerResourceBinding.from_json(binding_json) if binding_json else None
    )
    self._resource_binding_dict = (
        asdict(self._resource_binding) if self._resource_binding is not None else None
    )
```

在 `_setup_worker_info()` 传入：

```python
resource_binding=self._resource_binding_dict,
```

- [ ] **步骤 5：运行测试验证通过**

运行：

```bash
pytest -q tests/unit_tests/test_resource_pool_worker_integration.py
```

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add rlinf/scheduler/worker/worker_group.py rlinf/scheduler/worker/worker.py rlinf/scheduler/manager/worker_manager.py tests/unit_tests/test_resource_pool_worker_integration.py
git commit -s -m "feat: inject resource bindings into workers"
```

## 任务 6：EnvWorker 进程级 CPU affinity 和 per-env backend 校验

**文件：**
- 修改：`rlinf/workers/env/env_worker.py`
- 测试：`tests/unit_tests/test_resource_pool_env_binding.py`

- [ ] **步骤 1：编写失败测试**

```python
# tests/unit_tests/test_resource_pool_env_binding.py
from omegaconf import OmegaConf
import pytest

from rlinf.scheduler.resource_pool.bindings import CpuBinding, WorkerResourceBinding
from rlinf.workers.env.env_worker import EnvWorker


def _make_worker_with_binding(env_type: str, cpu: CpuBinding) -> EnvWorker:
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "env": {
                "train": {"env_type": env_type},
                "eval": {"env_type": env_type},
            },
            "runner": {"only_eval": False},
        }
    )
    worker._resource_binding = WorkerResourceBinding(
        component="env",
        rank=0,
        cluster_node_rank=0,
        node_group_label="node",
        cpu=cpu,
    )
    return worker


def test_env_worker_applies_process_cpu_affinity(monkeypatch) -> None:
    worker = _make_worker_with_binding(
        "maniskill", CpuBinding(process_cpu_cores=(0, 2))
    )
    captured = {}

    def fake_apply(cpus):
        captured["cpus"] = cpus

    monkeypatch.setattr(
        "rlinf.workers.env.env_worker.apply_process_cpu_affinity",
        fake_apply,
    )

    worker._apply_resource_pool_cpu_affinity()

    assert captured == {"cpus": (0, 2)}


def test_env_worker_rejects_unsupported_per_env_backend() -> None:
    worker = _make_worker_with_binding(
        "maniskill",
        CpuBinding(process_cpu_cores=(0, 1), env_cpu_core_groups=((0,), (1,))),
    )

    with pytest.raises(ValueError, match="per-env CPU binding"):
        worker._validate_env_resource_binding_supported()


def test_env_worker_accepts_supported_per_env_backend() -> None:
    worker = _make_worker_with_binding(
        "libero",
        CpuBinding(process_cpu_cores=(0, 1), env_cpu_core_groups=((0,), (1,))),
    )
    worker._validate_env_resource_binding_supported()
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest -q tests/unit_tests/test_resource_pool_env_binding.py -k "env_worker"
```

预期：FAIL，报错包含缺少 `_apply_resource_pool_cpu_affinity`。

- [ ] **步骤 3：实现 EnvWorker helper**

在 `rlinf/workers/env/env_worker.py` import 添加：

```python
import os
from contextlib import contextmanager

from rlinf.scheduler.resource_pool.bindings import ENV_CPU_CORE_GROUPS_ENV
from rlinf.scheduler.resource_pool.cpu_binding import apply_process_cpu_affinity
```

在 `EnvWorker` class 中添加：

```python
_PER_ENV_CPU_SUPPORTED_ENVS = {
    "calvin",
    "habitat",
    "libero",
    "metaworld",
    "robocasa",
}

def _apply_resource_pool_cpu_affinity(self) -> None:
    binding = getattr(self, "_resource_binding", None)
    if binding is None or binding.cpu is None:
        return
    if binding.cpu.process_cpu_cores:
        apply_process_cpu_affinity(binding.cpu.process_cpu_cores)
        self.log_info(
            f"Applied resource pool CPU affinity: {binding.cpu.process_cpu_cores}"
        )

def _validate_env_resource_binding_supported(self) -> None:
    binding = getattr(self, "_resource_binding", None)
    if binding is None or binding.cpu is None:
        return
    if not binding.cpu.env_cpu_core_groups:
        return
    env_types = set()
    if not getattr(self, "only_eval", False) and self.cfg.env.get("train", None) is not None:
        env_types.add(str(self.cfg.env.train.env_type).lower())
    if self.cfg.env.get("eval", None) is not None:
        env_types.add(str(self.cfg.env.eval.env_type).lower())
    unsupported = env_types - self._PER_ENV_CPU_SUPPORTED_ENVS
    if unsupported:
        raise ValueError(
            "per-env CPU binding is only supported for SubprocVectorEnv-style "
            f"env backends {sorted(self._PER_ENV_CPU_SUPPORTED_ENVS)}, "
            f"but got {sorted(env_types)}. Use granularity=process or a supported backend."
        )
```

在 `init_worker()` 开头、`_setup_dst_rank_map()` 前加入：

```python
self._validate_env_resource_binding_supported()
self._apply_resource_pool_cpu_affinity()
```

为任务 7 的 stage 切片预留 context manager：

```python
@contextmanager
def _stage_env_cpu_core_groups(self, stage_id: int, num_envs_per_stage: int):
    binding = getattr(self, "_resource_binding", None)
    groups = (
        binding.cpu.env_cpu_core_groups
        if binding is not None and binding.cpu is not None
        else ()
    )
    if not groups:
        yield
        return
    start = stage_id * num_envs_per_stage
    end = start + num_envs_per_stage
    stage_groups = groups[start:end]
    if len(stage_groups) != num_envs_per_stage:
        raise ValueError(
            f"resource pool has {len(groups)} per-env CPU groups, but stage "
            f"{stage_id} needs indexes [{start}, {end})"
        )
    old_value = os.environ.get(ENV_CPU_CORE_GROUPS_ENV)
    os.environ[ENV_CPU_CORE_GROUPS_ENV] = ";".join(
        ",".join(map(str, group)) for group in stage_groups
    )
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(ENV_CPU_CORE_GROUPS_ENV, None)
        else:
            os.environ[ENV_CPU_CORE_GROUPS_ENV] = old_value
```

在 `_setup_env_and_wrappers()` 中包住每个 stage 的 env 创建。保留当前 wrapper 添加逻辑，把现有 env constructor call 移动到 context manager 内：

```python
with self._stage_env_cpu_core_groups(stage_id, num_envs_per_stage):
    env = env_cls(
        cfg=env_cfg,
        num_envs=num_envs_per_stage,
        seed_offset=self._rank * self.stage_num + stage_id,
        total_num_processes=self._world_size * self.stage_num,
        worker_info=self.worker_info,
    )
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest -q tests/unit_tests/test_resource_pool_env_binding.py -k "env_worker"
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/workers/env/env_worker.py tests/unit_tests/test_resource_pool_env_binding.py
git commit -s -m "feat: apply env worker cpu affinity"
```

## 任务 7：通用 SubprocVectorEnv per-env affinity

**文件：**
- 修改：`rlinf/envs/venv/venv.py`
- 测试：`tests/unit_tests/test_resource_pool_env_binding.py`

- [ ] **步骤 1：追加失败测试**

```python
# append to tests/unit_tests/test_resource_pool_env_binding.py
from rlinf.envs.venv.venv import _apply_subproc_env_cpu_affinity


def test_subproc_env_affinity_uses_local_env_index(monkeypatch) -> None:
    captured = {}

    def fake_apply(cpus):
        captured["cpus"] = cpus

    monkeypatch.setattr(
        "rlinf.envs.venv.venv.apply_process_cpu_affinity",
        fake_apply,
    )
    monkeypatch.setenv("RLINF_ENV_CPU_CORE_GROUPS", "0;1,2;3")

    _apply_subproc_env_cpu_affinity(local_env_index=1)

    assert captured == {"cpus": (1, 2)}
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest -q tests/unit_tests/test_resource_pool_env_binding.py::test_subproc_env_affinity_uses_local_env_index
```

预期：FAIL，报错包含无法 import `_apply_subproc_env_cpu_affinity`。

- [ ] **步骤 3：修改通用 venv worker**

在 `rlinf/envs/venv/venv.py` import 添加：

```python
import os

from rlinf.scheduler.resource_pool.cpu_binding import (
    apply_process_cpu_affinity,
    get_env_core_group_from_env,
)
```

添加 helper：

```python
def _apply_subproc_env_cpu_affinity(local_env_index: int) -> None:
    core_group = get_env_core_group_from_env(os.environ, local_env_index)
    if core_group is not None:
        apply_process_cpu_affinity(core_group)
```

修改 `_worker()` 签名和 env 创建前逻辑：

```python
def _worker(parent, p, env_fn_wrapper, obs_bufs=None, local_env_index: int = -1):
    parent.close()
    if local_env_index >= 0:
        _apply_subproc_env_cpu_affinity(local_env_index)
    env = env_fn_wrapper.data()
```

修改 `SubprocEnvWorker.__init__()` 的签名和 `args` 构造。保留该方法现有的 Pipe、share_memory、dummy env、buffer、Process start、child close 和 `super().__init__(env_fn)` 逻辑，只把签名和 `args` 改成如下形状：

```python
def __init__(
    self,
    env_fn: Callable[[], gym.Env],
    share_memory: bool = False,
    local_env_index: int = -1,
) -> None:
    args = (
        self.parent_remote,
        self.child_remote,
        CloudpickleWrapper(env_fn),
        self.buffer,
        local_env_index,
    )
```

修改 `SubprocVectorEnv.__init__()`：

```python
env_index = {"value": 0}

def worker_fn(fn: Callable[[], gym.Env]) -> SubprocEnvWorker:
    local_env_index = env_index["value"]
    env_index["value"] += 1
    return SubprocEnvWorker(
        fn, share_memory=False, local_env_index=local_env_index
    )
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest -q tests/unit_tests/test_resource_pool_env_binding.py::test_subproc_env_affinity_uses_local_env_index
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/envs/venv/venv.py tests/unit_tests/test_resource_pool_env_binding.py
git commit -s -m "feat: bind subproc env cpu affinity"
```

## 任务 8：自定义 SubprocVectorEnv 后端 per-env affinity

**文件：**
- 修改：`rlinf/envs/libero/venv.py`
- 修改：`rlinf/envs/calvin/venv.py`
- 修改：`rlinf/envs/metaworld/venv.py`
- 修改：`rlinf/envs/robocasa/venv.py`
- 修改：`rlinf/envs/habitat/venv.py`
- 测试：`tests/unit_tests/test_resource_pool_env_binding.py`

- [ ] **步骤 1：追加失败测试：每个后端暴露 local env index 参数**

```python
# append to tests/unit_tests/test_resource_pool_env_binding.py
import inspect


def test_custom_subproc_workers_accept_local_env_index() -> None:
    from rlinf.envs.calvin.venv import ReconfigureSubprocEnvWorker as CalvinWorker
    from rlinf.envs.habitat.venv import ReconfigureSubprocEnvWorker as HabitatWorker
    from rlinf.envs.libero.venv import ReconfigureSubprocEnvWorker as LiberoWorker
    from rlinf.envs.metaworld.venv import ReconfigureSubprocEnvWorker as MetaWorldWorker
    from rlinf.envs.robocasa.venv import RobocasaSubprocEnvWorker

    for worker_cls in [
        CalvinWorker,
        HabitatWorker,
        LiberoWorker,
        MetaWorldWorker,
        RobocasaSubprocEnvWorker,
    ]:
        assert "local_env_index" in inspect.signature(worker_cls.__init__).parameters
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest -q tests/unit_tests/test_resource_pool_env_binding.py::test_custom_subproc_workers_accept_local_env_index
```

预期：FAIL，至少一个 worker 构造函数缺少 `local_env_index`。

- [ ] **步骤 3：修改五个自定义 venv 文件**

对每个文件执行同一模式：

```python
from rlinf.envs.venv.venv import _apply_subproc_env_cpu_affinity
```

修改 `_worker()` 签名并在创建 env 前应用：

```python
def _worker(parent, p, env_fn_wrapper, obs_bufs=None, local_env_index: int = -1):
    parent.close()
    if local_env_index >= 0:
        _apply_subproc_env_cpu_affinity(local_env_index)
    env = env_fn_wrapper.data()
```

修改自定义 worker constructor 的签名和 `args` 构造。保留每个文件当前 constructor 里的 multiprocessing context、Pipe、share_memory、buffer、Process start、child close 和 `EnvWorker.__init__(self, env_fn)` 逻辑，只把签名和 `args` 改成如下形状：

```python
def __init__(
    self,
    env_fn: Callable[[], gym.Env],
    share_memory: bool = False,
    local_env_index: int = -1,
):
    args = (
        self.parent_remote,
        self.child_remote,
        CloudpickleWrapper(env_fn),
        self.buffer,
        local_env_index,
    )
```

修改自定义 vector env constructor：

```python
env_index = {"value": 0}

def worker_fn(fn: Callable[[], gym.Env]) -> ReconfigureSubprocEnvWorker:
    local_env_index = env_index["value"]
    env_index["value"] += 1
    return ReconfigureSubprocEnvWorker(
        fn, share_memory=False, local_env_index=local_env_index
    )
```

RoboCasa 使用 `RobocasaSubprocEnvWorker` 和 `RobocasaSubprocEnv`，类型名按该文件现有类名替换。

- [ ] **步骤 4：运行 env binding 全量测试**

运行：

```bash
pytest -q tests/unit_tests/test_resource_pool_env_binding.py
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/envs/libero/venv.py rlinf/envs/calvin/venv.py rlinf/envs/metaworld/venv.py rlinf/envs/robocasa/venv.py rlinf/envs/habitat/venv.py tests/unit_tests/test_resource_pool_env_binding.py
git commit -s -m "feat: bind custom subproc env cpu affinity"
```

## 任务 9：Embodied entrypoint 集成和 artifact 写出

**文件：**
- 修改：`examples/embodiment/train_embodied_agent.py`
- 修改：`examples/embodiment/train_async.py`
- 测试：`tests/unit_tests/test_resource_pool_worker_integration.py`

- [ ] **步骤 1：追加失败测试：disabled pool helper 返回 None**

```python
# append to tests/unit_tests/test_resource_pool_worker_integration.py
from examples.embodiment.train_embodied_agent import _get_resource_bindings
from rlinf.scheduler.resource_pool.pool import FineGrainedResourcePool


def test_get_resource_bindings_returns_none_when_pool_disabled() -> None:
    assert _get_resource_bindings(FineGrainedResourcePool.disabled(), "actor") is None
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest -q tests/unit_tests/test_resource_pool_worker_integration.py::test_get_resource_bindings_returns_none_when_pool_disabled
```

预期：FAIL，报错包含无法 import `_get_resource_bindings`。

- [ ] **步骤 3：修改 sync entrypoint**

在 imports 添加：

```python
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from rlinf.scheduler import Cluster, FineGrainedResourcePool
```

添加 helpers：

```python
def _get_resource_bindings(
    resource_pool: FineGrainedResourcePool, component: str
):
    return resource_pool.get_component_bindings(component)


def _resource_pool_artifact_path() -> Path:
    return Path(HydraConfig.get().runtime.output_dir) / "resource_pool_plan.json"
```

在 `component_placement = HybridComponentPlacement(cfg, cluster)` 后加入：

```python
resource_pool = FineGrainedResourcePool.from_config(
    cfg=cfg,
    cluster=cluster,
    component_placement=component_placement,
)
```

每个 embodied `launch()` 加 `resource_bindings`：

```python
resource_bindings=_get_resource_bindings(resource_pool, "actor")
```

对应 rollout/env/reward 分别传 `"rollout"`、`"env"`、`"reward"`。

在 runner 创建前写 artifact：

```python
if resource_pool.enabled:
    resource_pool.write_plan(_resource_pool_artifact_path())
```

- [ ] **步骤 4：修改 async entrypoint**

在 `examples/embodiment/train_async.py` 做同样改动。该文件可以定义自己的 `_get_resource_bindings()` 和 `_resource_pool_artifact_path()`，避免两个 Hydra entrypoint 互相 import。

- [ ] **步骤 5：运行 helper 测试验证通过**

运行：

```bash
pytest -q tests/unit_tests/test_resource_pool_worker_integration.py::test_get_resource_bindings_returns_none_when_pool_disabled
```

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add examples/embodiment/train_embodied_agent.py examples/embodiment/train_async.py tests/unit_tests/test_resource_pool_worker_integration.py
git commit -s -m "feat: wire resource pool into embodied entrypoints"
```

## 任务 10：受影响测试、格式检查和文档同步

**文件：**
- 修改：`docs/superpowers/specs/2026-05-25-fine-grained-resource-pool-design.md`
- 修改：`docs/superpowers/plans/2026-05-25-fine-grained-resource-pool-implementation-plan.md`

- [ ] **步骤 1：运行新增 unit tests**

运行：

```bash
pytest -q \
  tests/unit_tests/test_resource_pool_bindings.py \
  tests/unit_tests/test_resource_pool_cpu_binding.py \
  tests/unit_tests/test_resource_pool_gpu_binding.py \
  tests/unit_tests/test_resource_pool_config.py \
  tests/unit_tests/test_resource_pool_solver.py \
  tests/unit_tests/test_resource_pool_worker_integration.py \
  tests/unit_tests/test_resource_pool_env_binding.py
```

预期：PASS。

- [ ] **步骤 2：运行受影响现有 tests**

运行：

```bash
pytest -q \
  tests/unit_tests/test_worker.py \
  tests/unit_tests/test_placement.py \
  tests/unit_tests/test_robocasa_env.py \
  tests/unit_tests/test_robocasa_mujoco_diagnostics.py
```

预期：PASS。

- [ ] **步骤 3：运行 Ruff**

运行：

```bash
ruff check \
  rlinf/scheduler/resource_pool \
  rlinf/scheduler/__init__.py \
  rlinf/scheduler/worker/worker_group.py \
  rlinf/scheduler/worker/worker.py \
  rlinf/scheduler/manager/worker_manager.py \
  rlinf/workers/env/env_worker.py \
  rlinf/envs/venv/venv.py \
  rlinf/envs/libero/venv.py \
  rlinf/envs/calvin/venv.py \
  rlinf/envs/metaworld/venv.py \
  rlinf/envs/robocasa/venv.py \
  rlinf/envs/habitat/venv.py \
  examples/embodiment/train_embodied_agent.py \
  examples/embodiment/train_async.py \
  tests/unit_tests/test_resource_pool_bindings.py \
  tests/unit_tests/test_resource_pool_cpu_binding.py \
  tests/unit_tests/test_resource_pool_gpu_binding.py \
  tests/unit_tests/test_resource_pool_config.py \
  tests/unit_tests/test_resource_pool_solver.py \
  tests/unit_tests/test_resource_pool_worker_integration.py \
  tests/unit_tests/test_resource_pool_env_binding.py
```

预期：PASS。

- [ ] **步骤 4：运行 Ruff format check**

运行：

```bash
ruff format --check \
  rlinf/scheduler/resource_pool \
  rlinf/scheduler/__init__.py \
  rlinf/scheduler/worker/worker_group.py \
  rlinf/scheduler/worker/worker.py \
  rlinf/scheduler/manager/worker_manager.py \
  rlinf/workers/env/env_worker.py \
  rlinf/envs/venv/venv.py \
  rlinf/envs/libero/venv.py \
  rlinf/envs/calvin/venv.py \
  rlinf/envs/metaworld/venv.py \
  rlinf/envs/robocasa/venv.py \
  rlinf/envs/habitat/venv.py \
  examples/embodiment/train_embodied_agent.py \
  examples/embodiment/train_async.py \
  tests/unit_tests/test_resource_pool_bindings.py \
  tests/unit_tests/test_resource_pool_cpu_binding.py \
  tests/unit_tests/test_resource_pool_gpu_binding.py \
  tests/unit_tests/test_resource_pool_config.py \
  tests/unit_tests/test_resource_pool_solver.py \
  tests/unit_tests/test_resource_pool_worker_integration.py \
  tests/unit_tests/test_resource_pool_env_binding.py
```

预期：PASS。

- [ ] **步骤 5：同步规格和计划中的最终命名**

如果实现中最终 public API 名称、env var 名称、artifact 路径或支持 env 类型集合与规格不同，只允许做最小文档修正。修正文档时写明最终名称，不保留替代名称。

- [ ] **步骤 6：Commit**

```bash
git add docs/superpowers/specs/2026-05-25-fine-grained-resource-pool-design.md docs/superpowers/plans/2026-05-25-fine-grained-resource-pool-implementation-plan.md
git commit -s -m "docs: update resource pool implementation notes"
```

如果步骤 5 没有文档变更，跳过 commit，并在最终汇报中说明没有文档差异。

---

## 规格覆盖自检

- Embodied-only 范围：任务 9 覆盖 sync/async embodied entrypoint；任务 10 回归 disabled 行为。
- CPU 默认独占：任务 2 和任务 4 覆盖 parser、切分、默认 solver。
- plan-file 显式共享：任务 4 覆盖 loader、summary 和共享允许语义。
- EnvWorker 进程级绑定：任务 6 覆盖 EnvWorker，AsyncEnvWorker 通过继承覆盖。
- per-env CPU 绑定：任务 6、7、8 覆盖 stage 切片、通用 venv 和五个自定义 venv。
- GPU `0/20/40/60/80/100`：任务 1、3、4 覆盖枚举、zero quota、MPS/MIG。
- pool 与 placement 匹配：任务 4 solver 校验覆盖 node group 和 device/MIG parent。
- Worker 注入和元数据：任务 5 覆盖 WorkerGroup、Worker、WorkerInfo。
- Artifact：任务 4 和任务 9 覆盖 summary 与写出。
- 验证：任务 10 覆盖新增测试、受影响现有测试、Ruff 和格式检查。
