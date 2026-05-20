# RoboCasa OpenPI Diagnostics Eval 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 实现一个独立 RoboCasa + OpenPI eval 脚本，输出每个 episode 的 success、action 轨迹和每步 MuJoCo `mjData` 诊断 JSONL。

**架构：** 在 RoboCasa subprocess worker 内新增只读诊断快照接口，保证能从真实子进程里的 `env.sim.data` 读取数据。独立 eval 脚本不启动 Ray，直接复用 `RobocasaEnv` 和 OpenPI model factory，循环预测 action chunk、逐步 step、采集诊断并写 JSONL。

**技术栈：** Python、Hydra/OmegaConf、NumPy、PyTorch、RoboCasa/robosuite/MuJoCo、pytest。

---

## 文件结构

- 修改：`rlinf/envs/robocasa/venv.py`
  - 职责：提供 MuJoCo 诊断快照纯函数、worker 命令和 `RobocasaSubprocEnv.get_mujoco_diagnostics()` 聚合方法。
- 修改：`rlinf/envs/robocasa/robocasa_env.py`
  - 职责：给独立 eval 脚本提供 `RobocasaEnv.get_mujoco_diagnostics()` 透传方法。
- 创建：`examples/embodiment/eval_robocasa_openpi_diagnostics.py`
  - 职责：Hydra 入口、配置校验、OpenPI 模型加载、本地 RoboCasa eval loop、episode JSONL 记录构建和 JSON 兼容转换。
- 创建：`tests/unit_tests/test_robocasa_mujoco_diagnostics.py`
  - 职责：测试诊断快照纯函数、contact 截断、contact force fallback 和 worker 透传的可测试接口。
- 创建：`tests/unit_tests/test_robocasa_openpi_diagnostics_eval.py`
  - 职责：测试 JSON 兼容转换、episode record builder、配置校验。

---

### 任务 1：MuJoCo 诊断快照纯函数

**文件：**
- 修改：`rlinf/envs/robocasa/venv.py`
- 测试：`tests/unit_tests/test_robocasa_mujoco_diagnostics.py`

- [ ] **步骤 1：编写失败的诊断序列化测试**

创建 `tests/unit_tests/test_robocasa_mujoco_diagnostics.py`，写入 fake model/data。测试不导入真实 RoboCasa。

```python
from __future__ import annotations

import numpy as np

from rlinf.envs.robocasa.venv import build_mujoco_diagnostics_snapshot


class _FakeContact:
    def __init__(self, dist: float, geom1: int, geom2: int):
        self.dist = dist
        self.geom1 = geom1
        self.geom2 = geom2


class _FakeModel:
    nbody = 2
    ngeom = 3

    def body(self, idx: int):
        return type("Body", (), {"name": f"body_{idx}"})()

    def geom(self, idx: int):
        return type("Geom", (), {"name": f"geom_{idx}"})()


class _FakeData:
    ncon = 2
    contact = [_FakeContact(-0.1, 0, 1), _FakeContact(0.2, 1, 2)]
    qvel = np.array([1.0, 2.0])
    xpos = np.array([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]])
    xquat = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    subtree_linvel = np.array([[0.3, 0.4, 0.5], [0.6, 0.7, 0.8]])
    energy = np.array([3.0, 4.0])


def test_build_mujoco_diagnostics_snapshot_serializes_arrays_and_names() -> None:
    snapshot = build_mujoco_diagnostics_snapshot(
        model=_FakeModel(),
        data=_FakeData(),
        max_contacts=8,
        include_model_names=True,
        contact_force_fn=lambda model, data, idx: [idx, idx + 1, 0, 0, 0, 0],
    )

    assert snapshot["ncon"] == 2
    assert snapshot["contacts"][0] == {
        "dist": -0.1,
        "geom1": 0,
        "geom2": 1,
        "geom1_name": "geom_0",
        "geom2_name": "geom_1",
        "force": [0, 1, 0, 0, 0, 0],
    }
    assert snapshot["qvel"] == [1.0, 2.0]
    assert snapshot["xpos"][1] == [1.0, 1.1, 1.2]
    assert snapshot["xquat"][0] == [1.0, 0.0, 0.0, 0.0]
    assert snapshot["subtree_linvel"][0] == [0.3, 0.4, 0.5]
    assert snapshot["energy"] == [3.0, 4.0]
    assert snapshot["kinetic_energy"] == 3.0
    assert snapshot["potential_energy"] == 4.0
    assert snapshot["body_names"] == ["body_0", "body_1"]
    assert snapshot["geom_names"] == ["geom_0", "geom_1", "geom_2"]
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/unit_tests/test_robocasa_mujoco_diagnostics.py::test_build_mujoco_diagnostics_snapshot_serializes_arrays_and_names -v`

预期：FAIL，报错包含 `cannot import name 'build_mujoco_diagnostics_snapshot'`。

- [ ] **步骤 3：实现诊断快照纯函数**

在 `rlinf/envs/robocasa/venv.py` 顶部 imports 后添加：

```python
def _json_list(value: Any) -> list:
    return np.asarray(value).tolist()


def _named_items(model: Any, count_attr: str, accessor_name: str) -> list[str]:
    count = int(getattr(model, count_attr, 0))
    accessor = getattr(model, accessor_name)
    names = []
    for idx in range(count):
        item = accessor(idx)
        names.append(str(getattr(item, "name", "")))
    return names


def _default_contact_force(model: Any, data: Any, contact_id: int) -> list[float]:
    import mujoco

    force = np.zeros(6, dtype=np.float64)
    mujoco.mj_contactForce(model, data, contact_id, force)
    return force.tolist()


def build_mujoco_diagnostics_snapshot(
    model: Any,
    data: Any,
    max_contacts: int | None = None,
    include_model_names: bool = True,
    contact_force_fn: Callable[[Any, Any, int], list[float]] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable MuJoCo diagnostics snapshot."""
    force_fn = contact_force_fn or _default_contact_force
    ncon = int(getattr(data, "ncon", 0))
    contact_limit = ncon if max_contacts is None else min(ncon, int(max_contacts))
    contacts = []
    for contact_id in range(contact_limit):
        contact = data.contact[contact_id]
        contact_info = {
            "dist": float(contact.dist),
            "geom1": int(contact.geom1),
            "geom2": int(contact.geom2),
            "geom1_name": "",
            "geom2_name": "",
            "force": None,
        }
        if include_model_names:
            contact_info["geom1_name"] = str(model.geom(contact.geom1).name)
            contact_info["geom2_name"] = str(model.geom(contact.geom2).name)
        try:
            contact_info["force"] = force_fn(model, data, contact_id)
        except Exception as exc:
            contact_info["force_error"] = str(exc)
        contacts.append(contact_info)

    energy = _json_list(getattr(data, "energy", []))
    snapshot = {
        "ncon": ncon,
        "contacts": contacts,
        "qvel": _json_list(data.qvel),
        "xpos": _json_list(data.xpos),
        "xquat": _json_list(data.xquat),
        "subtree_linvel": _json_list(data.subtree_linvel),
        "energy": energy,
        "kinetic_energy": float(energy[0]) if len(energy) > 0 else None,
        "potential_energy": float(energy[1]) if len(energy) > 1 else None,
        "body_names": [],
        "geom_names": [],
    }
    if include_model_names:
        snapshot["body_names"] = _named_items(model, "nbody", "body")
        snapshot["geom_names"] = _named_items(model, "ngeom", "geom")
    return snapshot
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/unit_tests/test_robocasa_mujoco_diagnostics.py::test_build_mujoco_diagnostics_snapshot_serializes_arrays_and_names -v`

预期：PASS。

- [ ] **步骤 5：添加 contact 截断和 force fallback 测试**

在 `tests/unit_tests/test_robocasa_mujoco_diagnostics.py` 追加：

```python
def test_build_mujoco_diagnostics_snapshot_truncates_contacts() -> None:
    snapshot = build_mujoco_diagnostics_snapshot(
        model=_FakeModel(),
        data=_FakeData(),
        max_contacts=1,
        include_model_names=False,
        contact_force_fn=lambda model, data, idx: [0, 0, 0, 0, 0, 0],
    )

    assert snapshot["ncon"] == 2
    assert len(snapshot["contacts"]) == 1
    assert snapshot["contacts"][0]["geom1_name"] == ""
    assert snapshot["body_names"] == []
    assert snapshot["geom_names"] == []


def test_build_mujoco_diagnostics_snapshot_keeps_contact_when_force_fails() -> None:
    def _raise_force(model, data, idx):
        raise RuntimeError("force unavailable")

    snapshot = build_mujoco_diagnostics_snapshot(
        model=_FakeModel(),
        data=_FakeData(),
        max_contacts=1,
        include_model_names=True,
        contact_force_fn=_raise_force,
    )

    contact = snapshot["contacts"][0]
    assert contact["force"] is None
    assert contact["force_error"] == "force unavailable"
    assert contact["dist"] == -0.1
```

- [ ] **步骤 6：运行新增测试验证通过**

运行：`pytest tests/unit_tests/test_robocasa_mujoco_diagnostics.py -v`

预期：PASS，3 个测试通过。

- [ ] **步骤 7：Commit**

```bash
git add rlinf/envs/robocasa/venv.py tests/unit_tests/test_robocasa_mujoco_diagnostics.py
git commit -s -m "feat: add robocasa mujoco diagnostics snapshot"
```

---

### 任务 2：RoboCasa subprocess 诊断命令和 env 透传

**文件：**
- 修改：`rlinf/envs/robocasa/venv.py`
- 修改：`rlinf/envs/robocasa/robocasa_env.py`
- 测试：`tests/unit_tests/test_robocasa_mujoco_diagnostics.py`

- [ ] **步骤 1：编写失败的 worker 透传测试**

在 `tests/unit_tests/test_robocasa_mujoco_diagnostics.py` 追加：

```python
def test_robocasa_subproc_env_get_mujoco_diagnostics_calls_workers() -> None:
    from rlinf.envs.robocasa.venv import RobocasaSubprocEnv

    env = RobocasaSubprocEnv.__new__(RobocasaSubprocEnv)

    class _Worker:
        def __init__(self, value):
            self.value = value

        def get_mujoco_diagnostics(self, max_contacts, include_model_names):
            return {
                "value": self.value,
                "max_contacts": max_contacts,
                "include_model_names": include_model_names,
            }

    env.workers = [_Worker(1), _Worker(2)]

    assert env.get_mujoco_diagnostics(
        max_contacts=4,
        include_model_names=False,
    ) == [
        {"value": 1, "max_contacts": 4, "include_model_names": False},
        {"value": 2, "max_contacts": 4, "include_model_names": False},
    ]
```

在 `tests/unit_tests/test_robocasa_env.py` 追加：

```python
def test_robocasa_env_get_mujoco_diagnostics_delegates_to_vector_env() -> None:
    module = importlib.import_module("rlinf.envs.robocasa.robocasa_env")
    RobocasaEnv = module.RobocasaEnv
    env = RobocasaEnv.__new__(RobocasaEnv)

    class _VectorEnv:
        def get_mujoco_diagnostics(self, max_contacts, include_model_names):
            return [{"max_contacts": max_contacts, "names": include_model_names}]

    env.env = _VectorEnv()

    assert env.get_mujoco_diagnostics(
        max_contacts=7,
        include_model_names=True,
    ) == [{"max_contacts": 7, "names": True}]
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest \
  tests/unit_tests/test_robocasa_mujoco_diagnostics.py::test_robocasa_subproc_env_get_mujoco_diagnostics_calls_workers \
  tests/unit_tests/test_robocasa_env.py::test_robocasa_env_get_mujoco_diagnostics_delegates_to_vector_env \
  -v
```

预期：FAIL，报错包含 `has no attribute 'get_mujoco_diagnostics'`。

- [ ] **步骤 3：实现 worker 命令和 vector env 方法**

在 `rlinf/envs/robocasa/venv.py` 的 `_worker` 中，`elif cmd == "setattr":` 后添加：

```python
            elif cmd == "get_mujoco_diagnostics":
                p.send(
                    build_mujoco_diagnostics_snapshot(
                        model=env.sim.model,
                        data=env.sim.data,
                        max_contacts=data.get("max_contacts"),
                        include_model_names=data.get("include_model_names", True),
                    )
                )
```

在 `RobocasaSubprocEnvWorker` 类中添加：

```python
    def get_mujoco_diagnostics(
        self,
        max_contacts: int | None = None,
        include_model_names: bool = True,
    ) -> dict[str, Any]:
        self.parent_remote.send(
            [
                "get_mujoco_diagnostics",
                {
                    "max_contacts": max_contacts,
                    "include_model_names": include_model_names,
                },
            ]
        )
        return self.parent_remote.recv()
```

在 `RobocasaSubprocEnv` 类中添加：

```python
    def get_mujoco_diagnostics(
        self,
        max_contacts: int | None = None,
        include_model_names: bool = True,
    ) -> list[dict[str, Any]]:
        self._assert_is_not_closed()
        return [
            worker.get_mujoco_diagnostics(max_contacts, include_model_names)
            for worker in self.workers
        ]
```

- [ ] **步骤 4：实现 RobocasaEnv 透传方法**

在 `rlinf/envs/robocasa/robocasa_env.py` 的 `close()` 前添加：

```python
    def get_mujoco_diagnostics(
        self,
        max_contacts: int | None = None,
        include_model_names: bool = True,
    ) -> list[dict]:
        """Return MuJoCo diagnostics for each RoboCasa subprocess env."""
        return self.env.get_mujoco_diagnostics(
            max_contacts=max_contacts,
            include_model_names=include_model_names,
        )
```

如果当前 Python 版本或 lint 不接受 `int | None`，改用已导入的 `Optional[int]`。

- [ ] **步骤 5：运行测试验证通过**

运行：

```bash
pytest \
  tests/unit_tests/test_robocasa_mujoco_diagnostics.py \
  tests/unit_tests/test_robocasa_env.py \
  -v
```

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add rlinf/envs/robocasa/venv.py rlinf/envs/robocasa/robocasa_env.py tests/unit_tests/test_robocasa_mujoco_diagnostics.py tests/unit_tests/test_robocasa_env.py
git commit -s -m "feat: expose robocasa mujoco diagnostics"
```

---

### 任务 3：JSON 兼容转换和 episode record builder

**文件：**
- 创建：`examples/embodiment/eval_robocasa_openpi_diagnostics.py`
- 测试：`tests/unit_tests/test_robocasa_openpi_diagnostics_eval.py`

- [ ] **步骤 1：编写失败的 JSON 转换和 episode record 测试**

创建 `tests/unit_tests/test_robocasa_openpi_diagnostics_eval.py`：

```python
from __future__ import annotations

import json

import numpy as np
import torch

from examples.embodiment.eval_robocasa_openpi_diagnostics import (
    build_episode_record,
    to_jsonable,
)


def test_to_jsonable_converts_numpy_and_torch_values() -> None:
    value = {
        "array": np.array([[1, 2], [3, 4]], dtype=np.int64),
        "float": np.float32(1.5),
        "bool": np.bool_(True),
        "tensor": torch.tensor([1.0, 2.0]),
    }

    result = to_jsonable(value)

    assert result == {
        "array": [[1, 2], [3, 4]],
        "float": 1.5,
        "bool": True,
        "tensor": [1.0, 2.0],
    }
    json.dumps(result)


def test_build_episode_record_contains_required_fields() -> None:
    record = build_episode_record(
        episode_id=3,
        env_id=0,
        task_name="CloseDrawer",
        seed=42,
        task_description="close the drawer",
        actions=[np.array([0.1, 0.2])],
        step_records=[
            {
                "step": 0,
                "reward": np.float32(0.0),
                "success": np.bool_(False),
                "terminated": False,
                "truncated": False,
                "diagnostics": {"ncon": 1, "qvel": np.array([0.0])},
            }
        ],
        success=True,
        termination_reason="success",
    )

    assert record["episode_id"] == 3
    assert record["env_id"] == 0
    assert record["task_name"] == "CloseDrawer"
    assert record["seed"] == 42
    assert record["success"] is True
    assert record["num_steps"] == 1
    assert record["termination_reason"] == "success"
    assert record["task_description"] == "close the drawer"
    assert record["actions"] == [[0.1, 0.2]]
    assert record["steps"][0]["diagnostics"]["qvel"] == [0.0]
    json.dumps(record)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/unit_tests/test_robocasa_openpi_diagnostics_eval.py -v`

预期：FAIL，报错包含 `No module named` 或 `cannot import name`。

- [ ] **步骤 3：创建脚本基础和纯函数**

创建 `examples/embodiment/eval_robocasa_openpi_diagnostics.py`，先写入 imports、`to_jsonable()`、`build_episode_record()`：

```python
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def to_jsonable(value: Any) -> Any:
    """Convert common numeric containers into JSON-compatible Python values."""
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def build_episode_record(
    episode_id: int,
    env_id: int,
    task_name: str,
    seed: int,
    task_description: str,
    actions: list[Any],
    step_records: list[dict[str, Any]],
    success: bool,
    termination_reason: str,
) -> dict[str, Any]:
    """Build one JSONL episode record."""
    return to_jsonable(
        {
            "episode_id": episode_id,
            "env_id": env_id,
            "task_name": task_name,
            "seed": seed,
            "success": bool(success),
            "num_steps": len(step_records),
            "termination_reason": termination_reason,
            "task_description": task_description,
            "actions": actions,
            "steps": step_records,
        }
    )
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/unit_tests/test_robocasa_openpi_diagnostics_eval.py -v`

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add examples/embodiment/eval_robocasa_openpi_diagnostics.py tests/unit_tests/test_robocasa_openpi_diagnostics_eval.py
git commit -s -m "feat: add robocasa diagnostics episode records"
```

---

### 任务 4：配置校验、模型加载和本地 eval loop

**文件：**
- 修改：`examples/embodiment/eval_robocasa_openpi_diagnostics.py`
- 测试：`tests/unit_tests/test_robocasa_openpi_diagnostics_eval.py`

- [ ] **步骤 1：编写失败的配置校验测试**

在 `tests/unit_tests/test_robocasa_openpi_diagnostics_eval.py` 追加：

```python
import pytest
from omegaconf import OmegaConf

from examples.embodiment.eval_robocasa_openpi_diagnostics import validate_diagnostics_cfg


def test_validate_diagnostics_cfg_rejects_non_robocasa_openpi() -> None:
    cfg = OmegaConf.create(
        {
            "env": {"eval": {"env_type": "libero"}},
            "actor": {"model": {"model_type": "openpi", "model_path": "/tmp/model"}},
            "diagnostics": {"output_path": "/tmp/out.jsonl", "num_episodes": 1},
        }
    )

    with pytest.raises(ValueError, match="RoboCasa"):
        validate_diagnostics_cfg(cfg)


def test_validate_diagnostics_cfg_sets_defaults(tmp_path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    cfg = OmegaConf.create(
        {
            "env": {
                "eval": {
                    "env_type": "robocasa",
                    "total_num_envs": 1,
                    "max_episode_steps": 3,
                }
            },
            "actor": {"model": {"model_type": "openpi", "model_path": str(model_path)}},
            "diagnostics": {"output_path": str(tmp_path / "out.jsonl")},
        }
    )

    validate_diagnostics_cfg(cfg)

    assert cfg.diagnostics.num_episodes == 1
    assert cfg.diagnostics.max_contacts == 32
    assert cfg.diagnostics.include_model_names is True
    assert cfg.diagnostics.flush_every == 1
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/unit_tests/test_robocasa_openpi_diagnostics_eval.py::test_validate_diagnostics_cfg_rejects_non_robocasa_openpi tests/unit_tests/test_robocasa_openpi_diagnostics_eval.py::test_validate_diagnostics_cfg_sets_defaults -v`

预期：FAIL，报错包含 `cannot import name 'validate_diagnostics_cfg'`。

- [ ] **步骤 3：实现配置校验**

在 `examples/embodiment/eval_robocasa_openpi_diagnostics.py` 添加：

```python
def validate_diagnostics_cfg(cfg: DictConfig) -> None:
    """Validate and normalize diagnostics eval config."""
    if cfg.env.eval.env_type != "robocasa":
        raise ValueError("RoboCasa diagnostics eval requires env.eval.env_type=robocasa.")
    if cfg.actor.model.model_type != "openpi":
        raise ValueError("RoboCasa diagnostics eval requires actor.model.model_type=openpi.")
    model_path = Path(str(cfg.actor.model.model_path))
    if not model_path.exists():
        raise FileNotFoundError(f"OpenPI model path does not exist: {model_path}")
    if "diagnostics" not in cfg:
        cfg.diagnostics = OmegaConf.create({})
    cfg.diagnostics.output_path = str(
        cfg.diagnostics.get("output_path", "robocasa_openpi_eval.jsonl")
    )
    cfg.diagnostics.num_episodes = int(cfg.diagnostics.get("num_episodes", 1))
    cfg.diagnostics.max_contacts = int(cfg.diagnostics.get("max_contacts", 32))
    cfg.diagnostics.include_model_names = bool(
        cfg.diagnostics.get("include_model_names", True)
    )
    cfg.diagnostics.flush_every = int(cfg.diagnostics.get("flush_every", 1))
    if int(cfg.env.eval.total_num_envs) != 1:
        raise ValueError("Standalone diagnostics eval currently requires env.eval.total_num_envs=1.")
```

- [ ] **步骤 4：实现模型/env 创建 helper 和 eval loop**

在同一脚本中添加：

```python
def load_openpi_model(cfg: DictConfig) -> torch.nn.Module:
    """Load the OpenPI model used by RoboCasa eval."""
    from rlinf.models.embodiment.openpi import get_model

    model = get_model(cfg.actor.model)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model


def create_robocasa_eval_env(cfg: DictConfig):
    """Create a local one-env RoboCasa evaluator."""
    from rlinf.envs.robocasa.robocasa_env import RobocasaEnv

    return RobocasaEnv(
        cfg=cfg.env.eval,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )


def _tensor_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return bool(value.item())
    return bool(value)


def _tensor_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return float(value.item())
    return float(value)


def run_diagnostics_eval(cfg: DictConfig) -> None:
    """Run standalone RoboCasa + OpenPI diagnostics evaluation."""
    validate_diagnostics_cfg(cfg)
    output_path = Path(str(cfg.diagnostics.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = create_robocasa_eval_env(cfg)
    model = load_openpi_model(cfg)
    try:
        with output_path.open("a", encoding="utf-8") as output_file:
            for episode_id in range(int(cfg.diagnostics.num_episodes)):
                obs, _ = env.reset()
                actions: list[Any] = []
                step_records: list[dict[str, Any]] = []
                success = False
                termination_reason = "max_episode_steps"
                task_description = ""
                if obs.get("task_descriptions"):
                    task_description = obs["task_descriptions"][0]

                while len(step_records) < int(cfg.env.eval.max_episode_steps):
                    with torch.no_grad():
                        action_chunk, _ = model.predict_action_batch(
                            env_obs=obs,
                            mode="eval",
                        )
                    if isinstance(action_chunk, torch.Tensor):
                        action_chunk_np = action_chunk.detach().cpu().numpy()
                    else:
                        action_chunk_np = np.asarray(action_chunk)

                    for chunk_idx in range(action_chunk_np.shape[1]):
                        action = action_chunk_np[:, chunk_idx]
                        obs, reward, terminated, truncated, infos = env.step(
                            action,
                            auto_reset=False,
                        )
                        diagnostics = env.get_mujoco_diagnostics(
                            max_contacts=cfg.diagnostics.max_contacts,
                            include_model_names=cfg.diagnostics.include_model_names,
                        )[0]
                        step_success = _tensor_bool(terminated[0])
                        step_truncated = _tensor_bool(truncated[0])
                        actions.append(action[0])
                        step_records.append(
                            {
                                "step": len(step_records),
                                "reward": _tensor_float(reward[0]),
                                "success": step_success,
                                "terminated": step_success,
                                "truncated": step_truncated,
                                "diagnostics": diagnostics,
                            }
                        )
                        if step_success:
                            success = True
                            termination_reason = "success"
                            break
                        if step_truncated or len(step_records) >= int(cfg.env.eval.max_episode_steps):
                            termination_reason = "truncated"
                            break
                    if success or termination_reason == "truncated":
                        break

                task_name = str(cfg.env.eval.task_names[0])
                record = build_episode_record(
                    episode_id=episode_id,
                    env_id=0,
                    task_name=task_name,
                    seed=int(cfg.env.eval.seed),
                    task_description=task_description,
                    actions=actions,
                    step_records=step_records,
                    success=success,
                    termination_reason=termination_reason,
                )
                output_file.write(json.dumps(record) + "\n")
                if (episode_id + 1) % int(cfg.diagnostics.flush_every) == 0:
                    output_file.flush()
    finally:
        env.close()
```

- [ ] **步骤 5：添加 Hydra main**

在脚本末尾添加：

```python
@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="robocasa_closedrawer_ppo_openpi",
)
def main(cfg: DictConfig) -> None:
    from rlinf.config import validate_cfg

    cfg.runner.only_eval = True
    cfg = validate_cfg(cfg)
    run_diagnostics_eval(cfg)


if __name__ == "__main__":
    main()
```

- [ ] **步骤 6：运行单元测试验证通过**

运行：`pytest tests/unit_tests/test_robocasa_openpi_diagnostics_eval.py -v`

预期：PASS。

- [ ] **步骤 7：运行脚本帮助/配置解析冒烟检查**

运行：

```bash
EMBODIED_PATH=/data1/miliang/RLinf/examples/embodiment \
PYTHONPATH=/data1/miliang/RLinf:$PYTHONPATH \
python examples/embodiment/eval_robocasa_openpi_diagnostics.py --help
```

预期：显示 Hydra 帮助或配置帮助，不启动真实模型/环境。

- [ ] **步骤 8：Commit**

```bash
git add examples/embodiment/eval_robocasa_openpi_diagnostics.py tests/unit_tests/test_robocasa_openpi_diagnostics_eval.py
git commit -s -m "feat: add robocasa openpi diagnostics eval"
```

---

### 任务 5：最终验证和文档化运行命令

**文件：**
- 修改：`docs/superpowers/plans/2026-05-20-robocasa-openpi-diagnostics-eval.md`

- [ ] **步骤 1：运行相关单元测试**

运行：

```bash
pytest \
  tests/unit_tests/test_robocasa_mujoco_diagnostics.py \
  tests/unit_tests/test_robocasa_env.py \
  tests/unit_tests/test_robocasa_openpi_diagnostics_eval.py \
  -v
```

预期：PASS。

- [ ] **步骤 2：运行 lint/format 检查**

运行：

```bash
ruff check rlinf/envs/robocasa/venv.py rlinf/envs/robocasa/robocasa_env.py examples/embodiment/eval_robocasa_openpi_diagnostics.py tests/unit_tests/test_robocasa_mujoco_diagnostics.py tests/unit_tests/test_robocasa_openpi_diagnostics_eval.py tests/unit_tests/test_robocasa_env.py
```

预期：PASS。

如果失败，按 Ruff 输出修正对应文件，然后重新运行同一命令。

- [ ] **步骤 3：记录真实运行命令**

在最终回复中给出真实 eval 命令：

```bash
EMBODIED_PATH=/data1/miliang/RLinf/examples/embodiment \
PYTHONPATH=/data1/miliang/RLinf:$PYTHONPATH \
python examples/embodiment/eval_robocasa_openpi_diagnostics.py \
  actor.model.model_path=/path/to/RLinf-Pi0-RoboCasa \
  rollout.model.model_path=/path/to/RLinf-Pi0-RoboCasa \
  env.eval.total_num_envs=1 \
  diagnostics.output_path=/path/to/robocasa_openpi_eval.jsonl \
  diagnostics.num_episodes=10
```

- [ ] **步骤 4：Commit 验证修正**

如果步骤 1 或步骤 2 产生修正：

```bash
git add rlinf/envs/robocasa/venv.py rlinf/envs/robocasa/robocasa_env.py examples/embodiment/eval_robocasa_openpi_diagnostics.py tests/unit_tests/test_robocasa_mujoco_diagnostics.py tests/unit_tests/test_robocasa_openpi_diagnostics_eval.py tests/unit_tests/test_robocasa_env.py
git commit -s -m "test: cover robocasa diagnostics eval"
```

如果没有修正，不创建空提交。

---

## 自检

- 规格覆盖：入口脚本、JSONL 结构、success/action 轨迹、contact、qvel/xpos/xquat/subtree_linvel/energy、contact force fallback、测试范围均有对应任务。
- 范围控制：不支持 GR00T，不改 Ray eval runner，不保存 obs，不实现阈值判死。
- 类型一致性：诊断快照统一为 `dict[str, Any]`，episode record builder 输出 JSON-compatible dict，RoboCasa env 透传返回 `list[dict]`。
