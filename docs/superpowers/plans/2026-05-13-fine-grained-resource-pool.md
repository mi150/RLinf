# Fine-Grained Resource Pool 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为 embodied 训练路径添加一个可选的 Fine-Grained CPU/GPU Resource Pool，支持 CPU per-env core 绑定、MPS/MIG 非破坏性 GPU 绑定、默认确定性分配和外部 plan 输入。

**架构：** 保留现有 `cluster.component_placement` 作为组件 worker 数量和 node/GPU rank 放置来源。新增 `rlinf.scheduler.resource_pool` 在 embodied entrypoint 中根据 placement 生成 worker 级绑定计划，并通过 `WorkerGroup.launch(resource_bindings=...)` 注入 worker runtime env；EnvWorker 和 SubprocVectorEnv 系后端负责应用 CPU affinity。

**技术栈：** Python dataclasses、OmegaConf、Ray actor runtime env、Linux `os.sched_setaffinity`、CUDA/MPS/MIG 环境变量、pytest、Ruff。

---

## 文件结构图

- 创建：`rlinf/scheduler/resource_pool/__init__.py`
  - 导出 public API：`FineGrainedResourcePool`、binding dataclasses、helper constants。
- 创建：`rlinf/scheduler/resource_pool/bindings.py`
  - 定义 `CpuBinding`、`GpuBinding`、`WorkerResourceBinding`，提供 JSON 序列化和 env var 转换。
- 创建：`rlinf/scheduler/resource_pool/cpu_binding.py`
  - CPU core 表达式解析、per-env split、affinity 应用、binding env 解析。
- 创建：`rlinf/scheduler/resource_pool/gpu_binding.py`
  - MPS/MIG env var 构造和校验。
- 创建：`rlinf/scheduler/resource_pool/config.py`
  - 解析 `cluster.resource_pool`，生成轻量配置 dataclasses。
- 创建：`rlinf/scheduler/resource_pool/solver.py`
  - 默认 deterministic solver 和 plan-file loader/validator。
- 创建：`rlinf/scheduler/resource_pool/pool.py`
  - `FineGrainedResourcePool` 门面：enabled/disabled、plan artifact 写出、component bindings 查询。
- 修改：`rlinf/scheduler/__init__.py`
  - 导出 resource pool API。
- 修改：`rlinf/scheduler/worker/worker_group.py`
  - `launch()` 接收 `resource_bindings`，按 rank 注入 env vars。
- 修改：`rlinf/scheduler/worker/worker.py`
  - 解析 `RLINF_RESOURCE_BINDING_JSON`，提供 `resource_binding` property，并把 binding 放进 `WorkerInfo`。
- 修改：`rlinf/scheduler/manager/worker_manager.py`
  - `WorkerInfo` 增加 `resource_binding: dict | None` 字段。
- 修改：`rlinf/workers/env/env_worker.py`
  - 在 env 创建前应用 EnvWorker 进程 CPU affinity；严格校验 per-env 支持。
- 修改：`rlinf/envs/venv/venv.py`
  - 通用 `SubprocEnvWorker` 支持传入 local env index，并在 subprocess entry 早期应用 per-env affinity。
- 修改：`rlinf/envs/calvin/venv.py`
  - CALVIN 自定义 subproc worker 传递 local env index 并应用 affinity。
- 修改：`rlinf/envs/robocasa/venv.py`
  - RoboCasa 自定义 subproc worker 传递 local env index 并应用 affinity。
- 修改：`examples/embodiment/train_embodied_agent.py`
  - 创建 resource pool 并传入 actor/rollout/env launches。
- 修改：`examples/embodiment/train_async.py`
  - 同步 embodied async entrypoint。
- 创建：`tests/unit_tests/test_resource_pool_bindings.py`
  - binding JSON/env var 和 WorkerGroup rank validation 的单元测试。
- 创建：`tests/unit_tests/test_resource_pool_cpu_binding.py`
  - CPU parser/splitter/affinity/helper 测试。
- 创建：`tests/unit_tests/test_resource_pool_gpu_binding.py`
  - MPS/MIG env var 和校验测试。
- 创建：`tests/unit_tests/test_resource_pool_solver.py`
  - 默认 solver、plan-file 校验、冲突检测测试。
- 创建：`tests/unit_tests/test_resource_pool_worker_integration.py`
  - Worker/WorkerGroup env 注入和 WorkerInfo 解析测试。
- 创建：`tests/unit_tests/test_resource_pool_env_binding.py`
  - fake SubprocVectorEnv per-env affinity 和 unsupported env strict failure 测试。

## 任务 1：Binding 数据模型和 Env Var 契约

**文件：**
- 创建：`rlinf/scheduler/resource_pool/__init__.py`
- 创建：`rlinf/scheduler/resource_pool/bindings.py`
- 测试：`tests/unit_tests/test_resource_pool_bindings.py`

- [ ] **步骤 1：编写失败测试：binding 能 JSON round-trip 并生成 env vars**

```python
# tests/unit_tests/test_resource_pool_bindings.py
import json

from rlinf.scheduler.resource_pool.bindings import (
    CPU_AFFINITY_ENV,
    ENV_CPU_CORE_GROUPS_ENV,
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
        cpu=CpuBinding(
            process_cpu_cores=(0, 1, 2, 3),
            env_cpu_core_groups=((0, 1), (2, 3)),
        ),
        gpu=GpuBinding(
            mode="mps",
            visible_devices=("0",),
            mps_active_thread_percentage=40,
            memory_mb=None,
            mig_device_uuid=None,
            parent_gpu=0,
        ),
    )

    restored = WorkerResourceBinding.from_json(binding.to_json())
    assert restored == binding

    env = binding.to_env_vars()
    assert json.loads(env[RESOURCE_BINDING_ENV])["component"] == "env"
    assert env[CPU_AFFINITY_ENV] == "0,1,2,3"
    assert env[ENV_CPU_CORE_GROUPS_ENV] == "0,1;2,3"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "40"
```

- [ ] **步骤 2：运行测试确认失败**

运行：`pytest -q tests/unit_tests/test_resource_pool_bindings.py::test_worker_resource_binding_round_trips_and_builds_env`

预期：FAIL，报错 `ModuleNotFoundError: No module named 'rlinf.scheduler.resource_pool'`。

- [ ] **步骤 3：实现 binding dataclasses 和 env var 常量**

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
MPS_PINNED_DEVICE_MEM_LIMIT_ENV = "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"


def _tuple_int(values: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


@dataclass(frozen=True)
class CpuBinding:
    """CPU binding assigned to one worker process."""

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
    """GPU binding assigned to one worker process."""

    mode: Literal["mps", "mig"] | None = None
    visible_devices: tuple[str, ...] = ()
    mps_active_thread_percentage: int | None = None
    memory_mb: int | None = None
    mig_device_uuid: str | None = None
    parent_gpu: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GpuBinding | None":
        if payload is None:
            return None
        return cls(
            mode=payload.get("mode"),
            visible_devices=tuple(str(v) for v in payload.get("visible_devices", ())),
            mps_active_thread_percentage=payload.get("mps_active_thread_percentage"),
            memory_mb=payload.get("memory_mb"),
            mig_device_uuid=payload.get("mig_device_uuid"),
            parent_gpu=payload.get("parent_gpu"),
        )


@dataclass(frozen=True)
class WorkerResourceBinding:
    """Fine-grained resource binding for one component rank."""

    component: str
    rank: int
    cluster_node_rank: int
    cpu: CpuBinding | None = None
    gpu: GpuBinding | None = None

    def to_json(self) -> str:
        """Serialize binding to stable JSON for Ray runtime env injection."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "WorkerResourceBinding":
        """Deserialize binding JSON from worker environment."""
        payload = json.loads(text)
        return cls(
            component=str(payload["component"]),
            rank=int(payload["rank"]),
            cluster_node_rank=int(payload["cluster_node_rank"]),
            cpu=CpuBinding.from_dict(payload.get("cpu")),
            gpu=GpuBinding.from_dict(payload.get("gpu")),
        )

    def to_env_vars(self) -> dict[str, str]:
        """Build process env vars for this binding."""
        env = {RESOURCE_BINDING_ENV: self.to_json()}
        if self.cpu is not None and self.cpu.process_cpu_cores:
            env[CPU_AFFINITY_ENV] = ",".join(map(str, self.cpu.process_cpu_cores))
            if self.cpu.env_cpu_core_groups:
                env[ENV_CPU_CORE_GROUPS_ENV] = ";".join(
                    ",".join(map(str, group)) for group in self.cpu.env_cpu_core_groups
                )
        if self.gpu is not None:
            if self.gpu.mig_device_uuid:
                env[CUDA_VISIBLE_DEVICES_ENV] = self.gpu.mig_device_uuid
            elif self.gpu.visible_devices:
                env[CUDA_VISIBLE_DEVICES_ENV] = ",".join(self.gpu.visible_devices)
            if self.gpu.mps_active_thread_percentage is not None:
                env[MPS_ACTIVE_THREAD_PERCENTAGE_ENV] = str(
                    self.gpu.mps_active_thread_percentage
                )
            if self.gpu.memory_mb is not None and self.gpu.mode == "mps":
                env[MPS_PINNED_DEVICE_MEM_LIMIT_ENV] = str(self.gpu.memory_mb)
        return env
```

```python
# rlinf/scheduler/resource_pool/__init__.py
from .bindings import CpuBinding, GpuBinding, WorkerResourceBinding

__all__ = [
    "CpuBinding",
    "GpuBinding",
    "WorkerResourceBinding",
]
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest -q tests/unit_tests/test_resource_pool_bindings.py::test_worker_resource_binding_round_trips_and_builds_env`

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/scheduler/resource_pool/__init__.py rlinf/scheduler/resource_pool/bindings.py tests/unit_tests/test_resource_pool_bindings.py
git commit -s -m "feat: add resource binding data model"
```

## 任务 2：CPU Core 解析、切分和 Affinity Helper

**文件：**
- 创建：`rlinf/scheduler/resource_pool/cpu_binding.py`
- 修改：`rlinf/scheduler/resource_pool/__init__.py`
- 测试：`tests/unit_tests/test_resource_pool_cpu_binding.py`

- [ ] **步骤 1：编写失败测试：CPU core set、per-env split、env var 解析**

```python
# tests/unit_tests/test_resource_pool_cpu_binding.py
import pytest

from rlinf.scheduler.resource_pool.cpu_binding import (
    build_even_split_cpu_groups,
    effective_process_affinity,
    get_env_core_group_from_env,
    parse_cpu_core_set,
)


def test_parse_cpu_core_set_supports_ranges_and_discrete_values() -> None:
    assert parse_cpu_core_set("0-3,8,10-11") == (0, 1, 2, 3, 8, 10, 11)


def test_parse_cpu_core_set_rejects_duplicate_cores() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_cpu_core_set("0-2,2")


def test_build_even_split_cpu_groups_distributes_remainder() -> None:
    assert build_even_split_cpu_groups(tuple(range(10)), env_count=3) == (
        (0, 1, 2, 3),
        (4, 5, 6),
        (7, 8, 9),
    )


def test_effective_process_affinity_unions_and_sorts() -> None:
    assert effective_process_affinity(((3, 1), (2,))) == (1, 2, 3)


def test_get_env_core_group_from_env_reads_group_by_index() -> None:
    env = {"RLINF_ENV_CPU_CORE_GROUPS": "0,1;2,3"}
    assert get_env_core_group_from_env(env, local_env_index=1) == (2, 3)
```

- [ ] **步骤 2：运行测试确认失败**

运行：`pytest -q tests/unit_tests/test_resource_pool_cpu_binding.py`

预期：FAIL，报错 `ModuleNotFoundError` 或缺少函数。

- [ ] **步骤 3：实现 CPU helper**

```python
# rlinf/scheduler/resource_pool/cpu_binding.py
from __future__ import annotations

import os
from collections.abc import Mapping

from .bindings import ENV_CPU_CORE_GROUPS_ENV


def parse_cpu_core_set(spec: str) -> tuple[int, ...]:
    """Parse a CPU core list like ``0-3,8`` into sorted unique IDs."""
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
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError as exc:
                raise ValueError(f"invalid cpu range '{token}'") from exc
            if start < 0 or end < 0 or end < start:
                raise ValueError(f"invalid cpu range '{token}'")
            cores.extend(range(start, end + 1))
            continue
        try:
            core = int(token)
        except ValueError as exc:
            raise ValueError(f"invalid cpu core '{token}'") from exc
        if core < 0:
            raise ValueError(f"cpu core id must be >= 0, got '{token}'")
        cores.append(core)
    normalized = tuple(sorted(cores))
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate cpu cores are not allowed")
    return normalized


def build_even_split_cpu_groups(
    cores: tuple[int, ...], env_count: int
) -> tuple[tuple[int, ...], ...]:
    """Evenly split CPU cores into deterministic per-env groups."""
    if env_count <= 0:
        raise ValueError("env_count must be > 0")
    if len(set(cores)) != len(cores):
        raise ValueError("duplicate cpu cores are not allowed")
    if len(cores) < env_count:
        raise ValueError("each logical env must receive at least one core")
    base = len(cores) // env_count
    remainder = len(cores) % env_count
    groups: list[tuple[int, ...]] = []
    cursor = 0
    for env_idx in range(env_count):
        group_size = base + (1 if env_idx < remainder else 0)
        groups.append(tuple(cores[cursor : cursor + group_size]))
        cursor += group_size
    return tuple(groups)


def effective_process_affinity(groups: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    """Return sorted union of all per-env CPU groups."""
    return tuple(sorted({cpu for group in groups for cpu in group}))


def parse_env_cpu_core_groups(spec: str) -> tuple[tuple[int, ...], ...]:
    """Parse ``RLINF_ENV_CPU_CORE_GROUPS`` value into per-env groups."""
    if not spec.strip():
        return ()
    groups = []
    for raw_group in spec.split(";"):
        group = parse_cpu_core_set(raw_group)
        if not group:
            raise ValueError("every env core group must contain at least one cpu")
        groups.append(group)
    return tuple(groups)


def get_env_core_group_from_env(
    env: Mapping[str, str], local_env_index: int
) -> tuple[int, ...] | None:
    """Return per-env CPU group for a local subprocess env index."""
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
    """Apply CPU affinity for the current process."""
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

运行：`pytest -q tests/unit_tests/test_resource_pool_cpu_binding.py`

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/scheduler/resource_pool/__init__.py rlinf/scheduler/resource_pool/cpu_binding.py tests/unit_tests/test_resource_pool_cpu_binding.py
git commit -s -m "feat: add resource pool cpu binding helpers"
```

## 任务 3：GPU MPS/MIG Env Builder 和校验

**文件：**
- 创建：`rlinf/scheduler/resource_pool/gpu_binding.py`
- 修改：`rlinf/scheduler/resource_pool/__init__.py`
- 测试：`tests/unit_tests/test_resource_pool_gpu_binding.py`

- [ ] **步骤 1：编写失败测试：MPS 百分比和 MIG UUID 覆盖**

```python
# tests/unit_tests/test_resource_pool_gpu_binding.py
import pytest

from rlinf.scheduler.resource_pool.gpu_binding import (
    build_gpu_env_vars,
    validate_mps_percentage,
)


def test_build_gpu_env_vars_sets_mps_limit_and_visible_devices() -> None:
    env = build_gpu_env_vars(
        mode="mps",
        visible_devices=("0",),
        mps_active_thread_percentage=60,
        memory_mb=4096,
        mig_device_uuid=None,
    )
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "60"
    assert env["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] == "4096"


def test_build_gpu_env_vars_uses_mig_uuid_for_visibility() -> None:
    env = build_gpu_env_vars(
        mode="mig",
        visible_devices=("0",),
        mps_active_thread_percentage=None,
        memory_mb=None,
        mig_device_uuid="MIG-abc",
    )
    assert env == {"CUDA_VISIBLE_DEVICES": "MIG-abc"}


@pytest.mark.parametrize("value", [0, -1, 101])
def test_validate_mps_percentage_rejects_invalid_values(value: int) -> None:
    with pytest.raises(ValueError, match="MPS active thread percentage"):
        validate_mps_percentage(value)
```

- [ ] **步骤 2：运行测试确认失败**

运行：`pytest -q tests/unit_tests/test_resource_pool_gpu_binding.py`

预期：FAIL，缺少模块或函数。

- [ ] **步骤 3：实现 GPU helper**

```python
# rlinf/scheduler/resource_pool/gpu_binding.py
from __future__ import annotations

from typing import Literal

from .bindings import (
    CUDA_VISIBLE_DEVICES_ENV,
    MPS_ACTIVE_THREAD_PERCENTAGE_ENV,
    MPS_PINNED_DEVICE_MEM_LIMIT_ENV,
)


def validate_mps_percentage(value: int) -> int:
    """Validate MPS active thread percentage."""
    if value < 1 or value > 100:
        raise ValueError(
            "MPS active thread percentage must be in [1, 100], "
            f"got {value}."
        )
    return value


def build_gpu_env_vars(
    *,
    mode: Literal["mps", "mig"],
    visible_devices: tuple[str, ...],
    mps_active_thread_percentage: int | None,
    memory_mb: int | None,
    mig_device_uuid: str | None,
) -> dict[str, str]:
    """Build non-destructive GPU binding environment variables."""
    env: dict[str, str] = {}
    if mode == "mig":
        if not mig_device_uuid:
            raise ValueError("MIG mode requires mig_device_uuid")
        env[CUDA_VISIBLE_DEVICES_ENV] = mig_device_uuid
        return env
    if mode != "mps":
        raise ValueError(f"unsupported gpu binding mode: {mode}")
    if visible_devices:
        env[CUDA_VISIBLE_DEVICES_ENV] = ",".join(visible_devices)
    if mps_active_thread_percentage is not None:
        env[MPS_ACTIVE_THREAD_PERCENTAGE_ENV] = str(
            validate_mps_percentage(mps_active_thread_percentage)
        )
    if memory_mb is not None:
        if memory_mb <= 0:
            raise ValueError("MPS memory_mb must be > 0")
        env[MPS_PINNED_DEVICE_MEM_LIMIT_ENV] = str(memory_mb)
    return env
```

- [ ] **步骤 4：让 `WorkerResourceBinding.to_env_vars()` 调用 GPU helper**

把 `rlinf/scheduler/resource_pool/bindings.py` 中 GPU env 生成逻辑替换为：

```python
        if self.gpu is not None:
            from .gpu_binding import build_gpu_env_vars

            if self.gpu.mode is not None:
                env.update(
                    build_gpu_env_vars(
                        mode=self.gpu.mode,
                        visible_devices=self.gpu.visible_devices,
                        mps_active_thread_percentage=(
                            self.gpu.mps_active_thread_percentage
                        ),
                        memory_mb=self.gpu.memory_mb,
                        mig_device_uuid=self.gpu.mig_device_uuid,
                    )
                )
```

- [ ] **步骤 5：运行 binding 和 GPU 测试验证通过**

运行：`pytest -q tests/unit_tests/test_resource_pool_bindings.py tests/unit_tests/test_resource_pool_gpu_binding.py`

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add rlinf/scheduler/resource_pool/bindings.py rlinf/scheduler/resource_pool/gpu_binding.py tests/unit_tests/test_resource_pool_gpu_binding.py
git commit -s -m "feat: add resource pool gpu binding helpers"
```

## 任务 4：Resource Pool Config Parser

**文件：**
- 创建：`rlinf/scheduler/resource_pool/config.py`
- 测试：`tests/unit_tests/test_resource_pool_solver.py`

- [ ] **步骤 1：编写失败测试：disabled config 和基本 MPS config 解析**

```python
# tests/unit_tests/test_resource_pool_solver.py
from omegaconf import OmegaConf

from rlinf.scheduler.resource_pool.config import ResourcePoolConfig


def test_resource_pool_config_disabled_when_missing() -> None:
    cfg = OmegaConf.create({"cluster": {"num_nodes": 1}})
    parsed = ResourcePoolConfig.from_cluster_cfg(cfg.cluster)
    assert parsed.enabled is False


def test_resource_pool_config_parses_mps_component_requests() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "resource_pool": {
                    "enabled": True,
                    "allocation_mode": "default",
                    "cpu": {
                        "enabled": True,
                        "pools": {"env_cpu": {"node_group": "cluster", "cores": "0-7"}},
                        "components": {
                            "env": {
                                "pool": "env_cpu",
                                "granularity": "per_env",
                                "unsupported_env_policy": "error",
                            }
                        },
                    },
                    "gpu": {
                        "enabled": True,
                        "mode": "mps",
                        "pools": {
                            "gpu_pool": {"node_group": "cluster", "devices": "0-1"}
                        },
                        "components": {
                            "rollout": {
                                "pool": "gpu_pool",
                                "sm_percent": 40,
                                "memory_mb": None,
                                "separate_gpus": False,
                            }
                        },
                    },
                }
            }
        }
    )

    parsed = ResourcePoolConfig.from_cluster_cfg(cfg.cluster)
    assert parsed.enabled is True
    assert parsed.cpu.components["env"].granularity == "per_env"
    assert parsed.gpu.mode == "mps"
    assert parsed.gpu.components["rollout"].sm_percent == 40
```

- [ ] **步骤 2：运行测试确认失败**

运行：`pytest -q tests/unit_tests/test_resource_pool_solver.py::test_resource_pool_config_disabled_when_missing tests/unit_tests/test_resource_pool_solver.py::test_resource_pool_config_parses_mps_component_requests`

预期：FAIL，缺少 `ResourcePoolConfig`。

- [ ] **步骤 3：实现 config dataclasses**

```python
# rlinf/scheduler/resource_pool/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class CpuPoolConfig:
    """Named CPU pool."""

    node_group: str
    cores: str


@dataclass(frozen=True)
class CpuComponentConfig:
    """CPU request for one component."""

    pool: str
    granularity: Literal["process", "per_env"] = "process"
    unsupported_env_policy: Literal["error"] = "error"


@dataclass(frozen=True)
class CpuResourceConfig:
    """CPU resource-pool configuration."""

    enabled: bool = False
    pools: dict[str, CpuPoolConfig] = field(default_factory=dict)
    components: dict[str, CpuComponentConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class MigDeviceConfig:
    """Pre-created MIG device metadata."""

    uuid: str
    parent_gpu: int
    sm_percent: int
    memory_mb: int


@dataclass(frozen=True)
class GpuPoolConfig:
    """Named GPU pool."""

    node_group: str
    devices: str | None = None
    mig_devices: tuple[MigDeviceConfig, ...] = ()


@dataclass(frozen=True)
class GpuComponentConfig:
    """GPU request for one component."""

    pool: str
    sm_percent: int | None = None
    memory_mb: int | None = None
    separate_gpus: bool = False


@dataclass(frozen=True)
class GpuResourceConfig:
    """GPU resource-pool configuration."""

    enabled: bool = False
    mode: Literal["mps", "mig"] = "mps"
    pools: dict[str, GpuPoolConfig] = field(default_factory=dict)
    components: dict[str, GpuComponentConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class ResourcePoolConfig:
    """Parsed cluster.resource_pool configuration."""

    enabled: bool = False
    allocation_mode: Literal["default", "plan_file"] = "default"
    allocation_plan_path: str | None = None
    cpu: CpuResourceConfig = field(default_factory=CpuResourceConfig)
    gpu: GpuResourceConfig = field(default_factory=GpuResourceConfig)

    @classmethod
    def from_cluster_cfg(cls, cluster_cfg: DictConfig) -> "ResourcePoolConfig":
        """Parse resource_pool from cluster config."""
        raw = getattr(cluster_cfg, "resource_pool", None)
        if raw is None or not bool(raw.get("enabled", False)):
            return cls(enabled=False)
        payload = OmegaConf.to_container(raw, resolve=True)
        assert isinstance(payload, dict)
        allocation_mode = payload.get("allocation_mode", "default")
        if allocation_mode not in {"default", "plan_file"}:
            raise ValueError(f"unsupported allocation_mode: {allocation_mode}")
        cpu = _parse_cpu(payload.get("cpu") or {})
        gpu = _parse_gpu(payload.get("gpu") or {})
        if allocation_mode == "plan_file" and not payload.get("allocation_plan_path"):
            raise ValueError("allocation_plan_path is required for plan_file mode")
        return cls(
            enabled=True,
            allocation_mode=allocation_mode,
            allocation_plan_path=payload.get("allocation_plan_path"),
            cpu=cpu,
            gpu=gpu,
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
            granularity=component.get("granularity", "process"),
            unsupported_env_policy=component.get("unsupported_env_policy", "error"),
        )
        for name, component in (payload.get("components") or {}).items()
    }
    for component_name, component in components.items():
        if component.pool not in pools:
            raise ValueError(
                f"CPU component {component_name} references unknown pool {component.pool}"
            )
    return CpuResourceConfig(
        enabled=bool(payload.get("enabled", False)),
        pools=pools,
        components=components,
    )


def _parse_gpu(payload: dict[str, Any]) -> GpuResourceConfig:
    pools = {}
    for name, pool in (payload.get("pools") or {}).items():
        mig_devices = tuple(
            MigDeviceConfig(
                uuid=str(device["uuid"]),
                parent_gpu=int(device["parent_gpu"]),
                sm_percent=int(device["sm_percent"]),
                memory_mb=int(device["memory_mb"]),
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
            sm_percent=component.get("sm_percent"),
            memory_mb=component.get("memory_mb"),
            separate_gpus=bool(component.get("separate_gpus", False)),
        )
        for name, component in (payload.get("components") or {}).items()
    }
    for component_name, component in components.items():
        if component.pool not in pools:
            raise ValueError(
                f"GPU component {component_name} references unknown pool {component.pool}"
            )
    mode = payload.get("mode", "mps")
    if mode not in {"mps", "mig"}:
        raise ValueError(f"unsupported gpu resource mode: {mode}")
    return GpuResourceConfig(
        enabled=bool(payload.get("enabled", False)),
        mode=mode,
        pools=pools,
        components=components,
    )
```

- [ ] **步骤 4：运行 config 测试验证通过**

运行：`pytest -q tests/unit_tests/test_resource_pool_solver.py::test_resource_pool_config_disabled_when_missing tests/unit_tests/test_resource_pool_solver.py::test_resource_pool_config_parses_mps_component_requests`

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/scheduler/resource_pool/config.py tests/unit_tests/test_resource_pool_solver.py
git commit -s -m "feat: parse resource pool config"
```

## 任务 5：默认 Solver 和 External Plan Loader

**文件：**
- 创建：`rlinf/scheduler/resource_pool/solver.py`
- 修改：`rlinf/scheduler/resource_pool/__init__.py`
- 测试：`tests/unit_tests/test_resource_pool_solver.py`

- [ ] **步骤 1：扩展失败测试：默认 solver 为 env 生成 per-env CPU 绑定**

```python
# append to tests/unit_tests/test_resource_pool_solver.py
from rlinf.scheduler.resource_pool.solver import ResourcePoolSolver
from tests.unit_tests.test_placement import create_fake_cluster
from rlinf.utils.placement import HybridComponentPlacement


def test_default_solver_builds_env_cpu_groups() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {"env": {"node_group": "node", "placement": "0:0-1"}},
                "resource_pool": {
                    "enabled": True,
                    "allocation_mode": "default",
                    "cpu": {
                        "enabled": True,
                        "pools": {"env_cpu": {"node_group": "cluster", "cores": "0-7"}},
                        "components": {
                            "env": {
                                "pool": "env_cpu",
                                "granularity": "per_env",
                                "unsupported_env_policy": "error",
                            }
                        },
                    },
                },
            },
            "env": {"train": {"total_num_envs": 4}, "eval": {"total_num_envs": 4}},
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
```

- [ ] **步骤 2：扩展失败测试：MPS solver 按 placement visible device 生成 GPU binding**

```python
def test_default_solver_builds_mps_gpu_bindings() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {"rollout": "0-1"},
                "resource_pool": {
                    "enabled": True,
                    "gpu": {
                        "enabled": True,
                        "mode": "mps",
                        "pools": {"gpu_pool": {"node_group": "cluster", "devices": "0-1"}},
                        "components": {
                            "rollout": {
                                "pool": "gpu_pool",
                                "sm_percent": 50,
                                "memory_mb": None,
                                "separate_gpus": False,
                            }
                        },
                    },
                },
            },
            "env": {"train": {"total_num_envs": 2}, "eval": {"total_num_envs": 2}},
            "rollout": {"pipeline_stage_num": 1},
        }
    )
    cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=2)
    placement = HybridComponentPlacement(cfg, cluster)
    pool_cfg = ResourcePoolConfig.from_cluster_cfg(cfg.cluster)

    bindings = ResourcePoolSolver(pool_cfg, cfg, cluster, placement).solve()

    assert bindings["rollout"][0].gpu.visible_devices == ("0",)
    assert bindings["rollout"][0].gpu.mps_active_thread_percentage == 50
    assert bindings["rollout"][1].gpu.visible_devices == ("1",)
```

- [ ] **步骤 3：运行 solver 测试确认失败**

运行：`pytest -q tests/unit_tests/test_resource_pool_solver.py -k "default_solver"`

预期：FAIL，缺少 `ResourcePoolSolver`。

- [ ] **步骤 4：实现默认 solver 的最小可用版本**

```python
# rlinf/scheduler/resource_pool/solver.py
from __future__ import annotations

import json
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
    """Build worker-level bindings from component placement and resource config."""

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

    def solve(self) -> dict[str, list[WorkerResourceBinding]]:
        """Return bindings keyed by component name."""
        if not self.pool_cfg.enabled:
            return {}
        if self.pool_cfg.allocation_mode == "plan_file":
            return self._load_plan_file()
        components = set(self.pool_cfg.cpu.components) | set(
            self.pool_cfg.gpu.components
        )
        return {component: self._solve_component(component) for component in components}

    def _solve_component(self, component: str) -> list[WorkerResourceBinding]:
        placements = self.component_placement.get_strategy(component).get_placement(
            self.cluster, isolate_accelerator=True
        )
        placements = sorted(placements, key=lambda p: p.rank)
        cpu_bindings = self._solve_cpu_component(component, len(placements))
        result: list[WorkerResourceBinding] = []
        for idx, placement in enumerate(placements):
            result.append(
                WorkerResourceBinding(
                    component=component,
                    rank=placement.rank,
                    cluster_node_rank=placement.cluster_node_rank,
                    cpu=cpu_bindings[idx] if idx < len(cpu_bindings) else None,
                    gpu=self._solve_gpu_binding(component, placement),
                )
            )
        return result

    def _solve_cpu_component(
        self, component: str, worker_count: int
    ) -> list[CpuBinding]:
        request = self.pool_cfg.cpu.components.get(component)
        if request is None or not self.pool_cfg.cpu.enabled:
            return []
        pool = self.pool_cfg.cpu.pools[request.pool]
        cores = parse_cpu_core_set(pool.cores)
        if len(cores) < worker_count:
            raise ValueError(
                f"CPU pool {request.pool} has {len(cores)} cores for "
                f"{worker_count} {component} workers"
            )
        groups = build_even_split_cpu_groups(cores, worker_count)
        if component == "env" and request.granularity == "per_env":
            total_envs = int(self.cfg.env.train.total_num_envs)
            stage_num = int(self.cfg.rollout.get("pipeline_stage_num", 1))
            envs_per_worker = total_envs // worker_count // stage_num
            return [
                CpuBinding(
                    process_cpu_cores=effective_process_affinity(
                        build_even_split_cpu_groups(group, envs_per_worker)
                    ),
                    env_cpu_core_groups=build_even_split_cpu_groups(
                        group, envs_per_worker
                    ),
                )
                for group in groups
            ]
        return [CpuBinding(process_cpu_cores=group) for group in groups]

    def _solve_gpu_binding(self, component: str, placement) -> GpuBinding | None:
        request = self.pool_cfg.gpu.components.get(component)
        if request is None or not self.pool_cfg.gpu.enabled:
            return None
        if self.pool_cfg.gpu.mode == "mps":
            return GpuBinding(
                mode="mps",
                visible_devices=tuple(placement.visible_accelerators),
                mps_active_thread_percentage=request.sm_percent,
                memory_mb=request.memory_mb,
                parent_gpu=placement.local_accelerator_rank,
            )
        pool = self.pool_cfg.gpu.pools[request.pool]
        if not pool.mig_devices:
            raise ValueError(f"GPU pool {request.pool} has no MIG devices")
        device = pool.mig_devices[placement.rank % len(pool.mig_devices)]
        if request.sm_percent is not None and request.sm_percent > device.sm_percent:
            raise ValueError(
                f"{component} rank {placement.rank} requests SM {request.sm_percent} "
                f"but MIG device {device.uuid} exposes {device.sm_percent}"
            )
        if request.memory_mb is not None and request.memory_mb > device.memory_mb:
            raise ValueError(
                f"{component} rank {placement.rank} requests memory "
                f"{request.memory_mb} but MIG device {device.uuid} exposes "
                f"{device.memory_mb}"
            )
        return GpuBinding(
            mode="mig",
            visible_devices=(),
            memory_mb=request.memory_mb,
            mig_device_uuid=device.uuid,
            parent_gpu=device.parent_gpu,
        )

    def _load_plan_file(self) -> dict[str, list[WorkerResourceBinding]]:
        path = self.pool_cfg.allocation_plan_path
        assert path is not None
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        bindings: dict[str, list[WorkerResourceBinding]] = {}
        for item in payload.get("bindings", []):
            binding = WorkerResourceBinding.from_json(json.dumps(item))
            bindings.setdefault(binding.component, []).append(binding)
        return {
            component: sorted(component_bindings, key=lambda b: b.rank)
            for component, component_bindings in bindings.items()
        }
```

- [ ] **步骤 5：运行 solver 测试验证通过**

运行：`pytest -q tests/unit_tests/test_resource_pool_solver.py`

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add rlinf/scheduler/resource_pool/solver.py tests/unit_tests/test_resource_pool_solver.py
git commit -s -m "feat: add default resource pool solver"
```

## 任务 6：FineGrainedResourcePool 门面和 Plan Artifact

**文件：**
- 创建：`rlinf/scheduler/resource_pool/pool.py`
- 修改：`rlinf/scheduler/resource_pool/__init__.py`
- 修改：`rlinf/scheduler/__init__.py`
- 测试：`tests/unit_tests/test_resource_pool_solver.py`

- [ ] **步骤 1：编写失败测试：disabled pool 返回空 bindings，enabled pool 可写 plan**

```python
# append to tests/unit_tests/test_resource_pool_solver.py
from pathlib import Path

from rlinf.scheduler.resource_pool.pool import FineGrainedResourcePool


def test_disabled_resource_pool_returns_empty_bindings() -> None:
    cfg = OmegaConf.create({"cluster": {"num_nodes": 1}})
    pool = FineGrainedResourcePool.disabled()
    assert pool.enabled is False
    assert pool.get_component_bindings("env") is None


def test_resource_pool_writes_plan_artifact(tmp_path: Path) -> None:
    binding = WorkerResourceBinding(
        component="env",
        rank=0,
        cluster_node_rank=0,
        cpu=None,
        gpu=None,
    )
    pool = FineGrainedResourcePool(enabled=True, bindings={"env": [binding]})
    output = tmp_path / "resource_pool_plan.json"
    pool.write_plan(output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["bindings"][0]["component"] == "env"
```

- [ ] **步骤 2：运行测试确认失败**

运行：`pytest -q tests/unit_tests/test_resource_pool_solver.py -k "resource_pool"`

预期：FAIL，缺少 `FineGrainedResourcePool`。

- [ ] **步骤 3：实现 pool 门面**

```python
# rlinf/scheduler/resource_pool/pool.py
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from omegaconf import DictConfig

from .bindings import WorkerResourceBinding
from .config import ResourcePoolConfig
from .solver import ResourcePoolSolver


class FineGrainedResourcePool:
    """Facade for optional fine-grained resource binding plans."""

    def __init__(
        self,
        *,
        enabled: bool,
        bindings: dict[str, list[WorkerResourceBinding]] | None = None,
    ) -> None:
        self.enabled = enabled
        self._bindings = bindings or {}

    @classmethod
    def disabled(cls) -> "FineGrainedResourcePool":
        """Return a disabled resource pool."""
        return cls(enabled=False)

    @classmethod
    def from_config(
        cls,
        *,
        cfg: DictConfig,
        cluster,
        component_placement,
    ) -> "FineGrainedResourcePool":
        """Build a resource pool from global config."""
        pool_cfg = ResourcePoolConfig.from_cluster_cfg(cfg.cluster)
        if not pool_cfg.enabled:
            return cls.disabled()
        bindings = ResourcePoolSolver(
            pool_cfg=pool_cfg,
            cfg=cfg,
            cluster=cluster,
            component_placement=component_placement,
        ).solve()
        return cls(enabled=True, bindings=bindings)

    def get_component_bindings(
        self, component: str
    ) -> list[WorkerResourceBinding] | None:
        """Return bindings for one component, or None when disabled/unconfigured."""
        if not self.enabled:
            return None
        return self._bindings.get(component)

    def write_plan(self, path: str | Path) -> None:
        """Write final binding plan as JSON."""
        all_bindings = [
            asdict(binding)
            for component in sorted(self._bindings)
            for binding in sorted(self._bindings[component], key=lambda b: b.rank)
        ]
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"bindings": all_bindings}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
```

```python
# rlinf/scheduler/resource_pool/__init__.py
from .bindings import CpuBinding, GpuBinding, WorkerResourceBinding
from .cpu_binding import apply_process_cpu_affinity
from .pool import FineGrainedResourcePool

__all__ = [
    "CpuBinding",
    "GpuBinding",
    "WorkerResourceBinding",
    "FineGrainedResourcePool",
    "apply_process_cpu_affinity",
]
```

```python
# rlinf/scheduler/__init__.py
from .resource_pool import FineGrainedResourcePool

# Add to __all__
"FineGrainedResourcePool",
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest -q tests/unit_tests/test_resource_pool_solver.py`

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/scheduler/__init__.py rlinf/scheduler/resource_pool/__init__.py rlinf/scheduler/resource_pool/pool.py tests/unit_tests/test_resource_pool_solver.py
git commit -s -m "feat: add fine grained resource pool facade"
```

## 任务 7：WorkerGroup 注入 Binding Env Vars

**文件：**
- 修改：`rlinf/scheduler/worker/worker_group.py`
- 测试：`tests/unit_tests/test_resource_pool_worker_integration.py`

- [ ] **步骤 1：编写失败测试：WorkerGroup 根据 rank 合并 binding env vars**

```python
# tests/unit_tests/test_resource_pool_worker_integration.py
from unittest.mock import Mock

from rlinf.scheduler.placement.placement import Placement
from rlinf.scheduler.resource_pool.bindings import (
    RESOURCE_BINDING_ENV,
    WorkerResourceBinding,
)
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
                node_group_label="cluster",
            )
        ]


def test_worker_group_injects_resource_binding_env(monkeypatch) -> None:
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
        cpu=None,
        gpu=None,
    )
    group = WorkerGroup(DummyWorker, args=(), kwargs={})
    group._cluster = cluster
    group._placement_strategy = DummyPlacementStrategy()
    group._isolate_gpu = True
    group._max_concurrency = None
    group._disable_distributed_log = False
    group._resource_bindings_by_rank = {0: binding}

    group._create_workers()

    env_vars = cluster.allocate.call_args.kwargs["env_vars"]
    assert RESOURCE_BINDING_ENV in env_vars
```

- [ ] **步骤 2：运行测试确认失败**

运行：`pytest -q tests/unit_tests/test_resource_pool_worker_integration.py::test_worker_group_injects_resource_binding_env`

预期：FAIL，`WorkerGroup` 没有 `_resource_bindings_by_rank` 处理。

- [ ] **步骤 3：修改 `WorkerGroup.launch()` 签名和绑定校验**

在 `rlinf/scheduler/worker/worker_group.py` 中：

```python
from ..resource_pool import WorkerResourceBinding
```

把 `launch()` 签名增加：

```python
        resource_bindings: Optional[list[WorkerResourceBinding]] = None,
```

在 `launch()` 里保存：

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

如果需要兼容测试直接调用 `_create_workers()`，在 `__init__()` 初始化：

```python
        self._resource_bindings_by_rank = {}
```

- [ ] **步骤 4：在 `_create_workers()` 合并 binding env vars**

在 `env_vars.update(AcceleratorUtil...)` 之后、`cluster.allocate()` 之前加入：

```python
            binding = self._resource_bindings_by_rank.get(placement.rank)
            if binding is not None:
                assert binding.cluster_node_rank == placement.cluster_node_rank, (
                    f"Resource binding for rank {placement.rank} targets node "
                    f"{binding.cluster_node_rank}, but placement uses node "
                    f"{placement.cluster_node_rank}."
                )
                env_vars.update(binding.to_env_vars())
```

同时在 `_create_workers()` 获取 placements 后校验绑定 rank 覆盖：

```python
        placement_ranks = {placement.rank for placement in placements}
        binding_ranks = set(self._resource_bindings_by_rank)
        assert binding_ranks.issubset(placement_ranks), (
            f"Resource binding ranks {binding_ranks} are not a subset of "
            f"placement ranks {placement_ranks}."
        )
```

- [ ] **步骤 5：运行 WorkerGroup 注入测试验证通过**

运行：`pytest -q tests/unit_tests/test_resource_pool_worker_integration.py::test_worker_group_injects_resource_binding_env`

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add rlinf/scheduler/worker/worker_group.py tests/unit_tests/test_resource_pool_worker_integration.py
git commit -s -m "feat: inject resource bindings into worker launches"
```

## 任务 8：Worker 解析 Binding 并写入 WorkerInfo

**文件：**
- 修改：`rlinf/scheduler/manager/worker_manager.py`
- 修改：`rlinf/scheduler/worker/worker.py`
- 测试：`tests/unit_tests/test_resource_pool_worker_integration.py`

- [ ] **步骤 1：编写失败测试：Worker 能从 env 解析 resource binding**

```python
# append to tests/unit_tests/test_resource_pool_worker_integration.py
from unittest import mock

from rlinf.scheduler.resource_pool.bindings import RESOURCE_BINDING_ENV
from rlinf.scheduler.worker.worker import Worker


def test_worker_parses_resource_binding_from_env() -> None:
    binding = WorkerResourceBinding(
        component="env",
        rank=2,
        cluster_node_rank=0,
        cpu=None,
        gpu=None,
    )
    worker = object.__new__(Worker)

    with mock.patch.dict("os.environ", {RESOURCE_BINDING_ENV: binding.to_json()}):
        worker._setup_resource_binding()

    assert worker.resource_binding == binding
```

- [ ] **步骤 2：运行测试确认失败**

运行：`pytest -q tests/unit_tests/test_resource_pool_worker_integration.py::test_worker_parses_resource_binding_from_env`

预期：FAIL，缺少 `_setup_resource_binding` 或 property。

- [ ] **步骤 3：扩展 WorkerInfo dataclass**

在 `rlinf/scheduler/manager/worker_manager.py` 的 `WorkerInfo` 增加字段：

```python
    resource_binding: dict | None = None
    """Fine-grained resource binding metadata for this worker."""
```

- [ ] **步骤 4：实现 Worker 解析方法和 property**

在 `rlinf/scheduler/worker/worker.py` imports 附近使用局部 import，避免循环。增加 property：

```python
    @property
    def resource_binding(self):
        """Get fine-grained resource binding for this worker, if configured."""
        return self._resource_binding
```

在 `__init__` setup 流程中 `_setup_worker_info()` 之前加入：

```python
        self._setup_resource_binding()
```

增加方法：

```python
    def _setup_resource_binding(self):
        """Parse optional fine-grained resource binding from environment."""
        from ..resource_pool.bindings import RESOURCE_BINDING_ENV, WorkerResourceBinding

        binding_json = os.environ.get(RESOURCE_BINDING_ENV)
        self._resource_binding = (
            WorkerResourceBinding.from_json(binding_json) if binding_json else None
        )
```

在 `_setup_worker_info()` 传入：

```python
            resource_binding=asdict(self._resource_binding)
            if self._resource_binding is not None
            else None,
```

并在文件顶部已有 `dataclasses` 需求时添加：

```python
from dataclasses import asdict
```

- [ ] **步骤 5：运行 worker integration 测试**

运行：`pytest -q tests/unit_tests/test_resource_pool_worker_integration.py`

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add rlinf/scheduler/manager/worker_manager.py rlinf/scheduler/worker/worker.py tests/unit_tests/test_resource_pool_worker_integration.py
git commit -s -m "feat: expose worker resource binding metadata"
```

## 任务 9：EnvWorker 进程级 CPU Affinity 和 Unsupported Env 严格失败

**文件：**
- 修改：`rlinf/workers/env/env_worker.py`
- 测试：`tests/unit_tests/test_resource_pool_env_binding.py`

- [ ] **步骤 1：编写失败测试：unsupported env 在 per-env binding 下失败**

```python
# tests/unit_tests/test_resource_pool_env_binding.py
from unittest import mock

import pytest
from omegaconf import OmegaConf

from rlinf.scheduler.resource_pool.bindings import CpuBinding, WorkerResourceBinding
from rlinf.workers.env.env_worker import EnvWorker


def test_env_worker_rejects_per_env_binding_for_unsupported_env() -> None:
    cfg = OmegaConf.create(
        {
            "env": {
                "train": {"env_type": "maniskill"},
                "eval": {"env_type": "maniskill"},
            }
        }
    )
    worker = object.__new__(EnvWorker)
    worker.cfg = cfg
    worker._resource_binding = WorkerResourceBinding(
        component="env",
        rank=0,
        cluster_node_rank=0,
        cpu=CpuBinding(
            process_cpu_cores=(0, 1),
            env_cpu_core_groups=((0,), (1,)),
        ),
        gpu=None,
    )

    with pytest.raises(ValueError, match="per-env CPU binding"):
        worker._validate_env_resource_binding_supported()
```

- [ ] **步骤 2：编写失败测试：EnvWorker 应用进程 affinity**

```python
def test_env_worker_applies_process_cpu_affinity(monkeypatch) -> None:
    worker = object.__new__(EnvWorker)
    worker._resource_binding = WorkerResourceBinding(
        component="env",
        rank=0,
        cluster_node_rank=0,
        cpu=CpuBinding(process_cpu_cores=(0, 2), env_cpu_core_groups=()),
        gpu=None,
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
```

- [ ] **步骤 3：运行测试确认失败**

运行：`pytest -q tests/unit_tests/test_resource_pool_env_binding.py -k "env_worker"`

预期：FAIL，缺少方法。

- [ ] **步骤 4：修改 EnvWorker 引入 helper 并实现校验/应用方法**

在 `rlinf/workers/env/env_worker.py` imports 添加：

```python
from rlinf.scheduler.resource_pool.cpu_binding import apply_process_cpu_affinity
```

在 `EnvWorker` 类中增加：

```python
    _PER_ENV_CPU_SUPPORTED_ENVS = {
        "calvin",
        "libero",
        "metaworld",
        "robocasa",
    }

    def _apply_resource_pool_cpu_affinity(self) -> None:
        """Apply EnvWorker process-level CPU affinity from resource binding."""
        binding = getattr(self, "_resource_binding", None)
        if binding is None or binding.cpu is None:
            return
        if binding.cpu.process_cpu_cores:
            apply_process_cpu_affinity(binding.cpu.process_cpu_cores)
            self.log_info(
                f"Applied resource pool CPU affinity: "
                f"{binding.cpu.process_cpu_cores}"
            )

    def _validate_env_resource_binding_supported(self) -> None:
        """Fail fast when strict per-env CPU binding is requested for unsupported envs."""
        binding = getattr(self, "_resource_binding", None)
        if binding is None or binding.cpu is None:
            return
        if not binding.cpu.env_cpu_core_groups:
            return
        env_types = {
            str(self.cfg.env.train.env_type).lower(),
            str(self.cfg.env.eval.env_type).lower(),
        }
        unsupported = env_types - self._PER_ENV_CPU_SUPPORTED_ENVS
        if unsupported:
            raise ValueError(
                "per-env CPU binding is only supported for SubprocVectorEnv-style "
                f"env backends {sorted(self._PER_ENV_CPU_SUPPORTED_ENVS)}, "
                f"but got {sorted(env_types)}. Disable cluster.resource_pool or "
                "use a supported env backend."
            )
```

在 `init_worker()` 开头、任何 env class import/创建之前加入：

```python
        self._validate_env_resource_binding_supported()
        self._apply_resource_pool_cpu_affinity()
```

- [ ] **步骤 5：运行 EnvWorker 测试验证通过**

运行：`pytest -q tests/unit_tests/test_resource_pool_env_binding.py -k "env_worker"`

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add rlinf/workers/env/env_worker.py tests/unit_tests/test_resource_pool_env_binding.py
git commit -s -m "feat: apply env worker resource cpu affinity"
```

## 任务 10：SubprocVectorEnv Per-Env CPU Affinity

**文件：**
- 修改：`rlinf/envs/venv/venv.py`
- 修改：`rlinf/envs/calvin/venv.py`
- 修改：`rlinf/envs/robocasa/venv.py`
- 测试：`tests/unit_tests/test_resource_pool_env_binding.py`

- [ ] **步骤 1：编写失败测试：subproc worker entry 先应用 affinity 再创建 env**

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
    monkeypatch.setenv("RLINF_ENV_CPU_CORE_GROUPS", "0,1;2,3")

    _apply_subproc_env_cpu_affinity(local_env_index=1)

    assert captured == {"cpus": (2, 3)}
```

- [ ] **步骤 2：运行测试确认失败**

运行：`pytest -q tests/unit_tests/test_resource_pool_env_binding.py::test_subproc_env_affinity_uses_local_env_index`

预期：FAIL，缺少 `_apply_subproc_env_cpu_affinity`。

- [ ] **步骤 3：修改通用 `rlinf/envs/venv/venv.py` worker entry**

在 imports 添加：

```python
import os
from rlinf.scheduler.resource_pool.cpu_binding import (
    apply_process_cpu_affinity,
    get_env_core_group_from_env,
)
```

在模块级增加 helper：

```python
def _apply_subproc_env_cpu_affinity(local_env_index: int) -> None:
    """Apply per-env CPU affinity in subprocess env worker."""
    core_group = get_env_core_group_from_env(os.environ, local_env_index)
    if core_group is not None:
        apply_process_cpu_affinity(core_group)
```

把 `_worker(...)` 签名增加最后一个参数：

```python
def _worker(parent, p, env_fn_wrapper, obs_buffer, local_env_index: int = -1):
```

在 `_worker` 里创建 env 之前加入：

```python
    if local_env_index >= 0:
        _apply_subproc_env_cpu_affinity(local_env_index)
```

修改 `SubprocEnvWorker.__init__` 签名：

```python
    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        share_memory: bool = False,
        local_env_index: int = -1,
    ) -> None:
```

构造 args 时传入：

```python
            local_env_index,
```

修改 `SubprocVectorEnv.__init__`：

```python
        env_index = {"value": 0}

        def worker_fn(fn: Callable[[], gym.Env]) -> SubprocEnvWorker:
            local_env_index = env_index["value"]
            env_index["value"] += 1
            return SubprocEnvWorker(
                fn, share_memory=False, local_env_index=local_env_index
            )
```

- [ ] **步骤 4：修改 CALVIN 自定义 subproc worker 传入 index**

在 `rlinf/envs/calvin/venv.py` 中：

```python
    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        share_memory: bool = False,
        local_env_index: int = -1,
    ):
```

args 增加 `local_env_index`。`ReconfigureSubprocEnv.__init__` 使用同样的 `env_index` closure，把 `local_env_index` 传给 `ReconfigureSubprocEnvWorker`。

- [ ] **步骤 5：修改 RoboCasa 自定义 subproc worker 传入 index**

在 `rlinf/envs/robocasa/venv.py` 中对 `RobocasaSubprocEnvWorker` 和 `RobocasaSubprocEnv.__init__` 做与 CALVIN 相同的变更。

- [ ] **步骤 6：运行 per-env affinity 测试验证通过**

运行：`pytest -q tests/unit_tests/test_resource_pool_env_binding.py::test_subproc_env_affinity_uses_local_env_index`

预期：PASS。

- [ ] **步骤 7：运行 env binding 全量测试**

运行：`pytest -q tests/unit_tests/test_resource_pool_env_binding.py`

预期：PASS。

- [ ] **步骤 8：Commit**

```bash
git add rlinf/envs/venv/venv.py rlinf/envs/calvin/venv.py rlinf/envs/robocasa/venv.py tests/unit_tests/test_resource_pool_env_binding.py
git commit -s -m "feat: bind subproc env workers to cpu core groups"
```

## 任务 11：Embodied Entrypoint 集成

**文件：**
- 修改：`examples/embodiment/train_embodied_agent.py`
- 修改：`examples/embodiment/train_async.py`
- 测试：`tests/unit_tests/test_resource_pool_worker_integration.py`

- [ ] **步骤 1：编写失败测试：entrypoint helper 返回 disabled pool 时 launches 传 None**

为了避免直接运行 Hydra entrypoint，创建轻量 helper 后测试。先写测试：

```python
# append to tests/unit_tests/test_resource_pool_worker_integration.py
from examples.embodiment.train_embodied_agent import _get_resource_bindings
from rlinf.scheduler.resource_pool.pool import FineGrainedResourcePool


def test_get_resource_bindings_returns_none_when_pool_disabled() -> None:
    pool = FineGrainedResourcePool.disabled()
    assert _get_resource_bindings(pool, "actor") is None
```

- [ ] **步骤 2：运行测试确认失败**

运行：`pytest -q tests/unit_tests/test_resource_pool_worker_integration.py::test_get_resource_bindings_returns_none_when_pool_disabled`

预期：FAIL，缺少 `_get_resource_bindings`。

- [ ] **步骤 3：修改 sync embodied entrypoint**

在 `examples/embodiment/train_embodied_agent.py` imports 添加：

```python
from pathlib import Path

from rlinf.scheduler import FineGrainedResourcePool
```

添加 helper：

```python
def _get_resource_bindings(resource_pool: FineGrainedResourcePool, component: str):
    """Return optional resource bindings for a component."""
    return resource_pool.get_component_bindings(component)
```

在 `component_placement = HybridComponentPlacement(cfg, cluster)` 后加入：

```python
    resource_pool = FineGrainedResourcePool.from_config(
        cfg=cfg,
        cluster=cluster,
        component_placement=component_placement,
    )
```

每个 `launch()` 加参数：

```python
        resource_bindings=_get_resource_bindings(resource_pool, "actor"),
```

对应 rollout/env 分别传 `"rollout"`、`"env"`。

在 runner 创建前写 artifact：

```python
    if resource_pool.enabled:
        resource_pool.write_plan(
            Path(cfg.runner.logger.log_path)
            / cfg.runner.logger.experiment_name
            / "resource_pool_plan.json"
        )
```

- [ ] **步骤 4：修改 async embodied entrypoint**

对 `examples/embodiment/train_async.py` 做同样集成。可以从 sync entrypoint 导入 helper，也可以在 async 文件内定义同名 helper；优先在各自文件定义，避免 Hydra entrypoint 互相 import 造成副作用。

- [ ] **步骤 5：运行 entrypoint helper 测试验证通过**

运行：`pytest -q tests/unit_tests/test_resource_pool_worker_integration.py::test_get_resource_bindings_returns_none_when_pool_disabled`

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add examples/embodiment/train_embodied_agent.py examples/embodiment/train_async.py tests/unit_tests/test_resource_pool_worker_integration.py
git commit -s -m "feat: wire resource pool into embodied entrypoints"
```

## 任务 12：验证、文档补充和最终清理

**文件：**
- 修改：`docs/superpowers/specs/2026-05-13-fine-grained-resource-pool-design.md`（如实现中发现契约微调）
- 可选修改：`docs/source-en/rst_source/tutorials/user/placement.rst`
- 可选修改：`docs/source-zh/rst_source/tutorials/user/placement.rst`

- [ ] **步骤 1：运行资源池相关单元测试**

运行：

```bash
pytest -q \
  tests/unit_tests/test_resource_pool_bindings.py \
  tests/unit_tests/test_resource_pool_cpu_binding.py \
  tests/unit_tests/test_resource_pool_gpu_binding.py \
  tests/unit_tests/test_resource_pool_solver.py \
  tests/unit_tests/test_resource_pool_worker_integration.py \
  tests/unit_tests/test_resource_pool_env_binding.py
```

预期：PASS。

- [ ] **步骤 2：运行受影响的现有测试**

运行：

```bash
pytest -q \
  tests/unit_tests/test_placement.py \
  tests/unit_tests/test_worker.py \
  tests/unit_tests/test_rollout_eval_benchmark_resource_binding.py
```

预期：PASS。`test_worker.py` 需要本地 Ray 可初始化；如果环境无法启动 Ray，记录具体错误，不要声称通过。

- [ ] **步骤 3：运行 Ruff 检查受影响文件**

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
  rlinf/envs/calvin/venv.py \
  rlinf/envs/robocasa/venv.py \
  examples/embodiment/train_embodied_agent.py \
  examples/embodiment/train_async.py \
  tests/unit_tests/test_resource_pool_bindings.py \
  tests/unit_tests/test_resource_pool_cpu_binding.py \
  tests/unit_tests/test_resource_pool_gpu_binding.py \
  tests/unit_tests/test_resource_pool_solver.py \
  tests/unit_tests/test_resource_pool_worker_integration.py \
  tests/unit_tests/test_resource_pool_env_binding.py
```

预期：PASS。

- [ ] **步骤 4：运行格式检查**

运行：

```bash
ruff format --check \
  rlinf/scheduler/resource_pool \
  rlinf/scheduler/worker/worker_group.py \
  rlinf/scheduler/worker/worker.py \
  rlinf/scheduler/manager/worker_manager.py \
  rlinf/workers/env/env_worker.py \
  rlinf/envs/venv/venv.py \
  rlinf/envs/calvin/venv.py \
  rlinf/envs/robocasa/venv.py \
  examples/embodiment/train_embodied_agent.py \
  examples/embodiment/train_async.py \
  tests/unit_tests/test_resource_pool_bindings.py \
  tests/unit_tests/test_resource_pool_cpu_binding.py \
  tests/unit_tests/test_resource_pool_gpu_binding.py \
  tests/unit_tests/test_resource_pool_solver.py \
  tests/unit_tests/test_resource_pool_worker_integration.py \
  tests/unit_tests/test_resource_pool_env_binding.py
```

预期：PASS。

- [ ] **步骤 5：更新规格中的实现偏差**

如果实现中使用的最终 env var、支持 env 列表、错误策略或 plan artifact 路径与规格不一致，直接更新：

```bash
$EDITOR docs/superpowers/specs/2026-05-13-fine-grained-resource-pool-design.md
```

然后运行占位符扫描命令：

```bash
rg -n "<占位符扫描正则>" docs/superpowers/specs/2026-05-13-fine-grained-resource-pool-design.md
```

预期：无输出。

- [ ] **步骤 6：Commit 最终验证/文档调整**

如果步骤 5 修改了文档：

```bash
git add docs/superpowers/specs/2026-05-13-fine-grained-resource-pool-design.md
git commit -s -m "docs: align resource pool design with implementation"
```

如果没有文档修改，不需要 commit。

## 规格覆盖自检

- 架构边界：任务 4、5、6、11 覆盖 `cluster.resource_pool`、默认 solver、外部 plan 和 embodied-only 接入。
- CPU per-env：任务 2、9、10 覆盖 core 解析、EnvWorker 进程 affinity、SubprocVectorEnv 子进程 affinity 和 unsupported env strict failure。
- GPU MPS/MIG：任务 1、3、5 覆盖 env vars、MPS 百分比、MIG UUID、memory/SM metadata 校验。
- Worker 注入：任务 7、8 覆盖 `WorkerGroup.launch(resource_bindings=...)`、runtime env 注入、Worker/WorkerInfo 解析。
- Artifacts：任务 6、11 覆盖 plan JSON 写出。
- 测试和验证：任务 12 覆盖新增测试、受影响现有测试、Ruff 和格式检查。
