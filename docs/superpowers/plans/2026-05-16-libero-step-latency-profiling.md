# LIBERO Step Latency Profiling 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 新增独立 toolkit 脚本，直接 profile LIBERO simulator `env.step()` latency，并输出每步 JSONL、task/trial 汇总和错误记录。

**架构：** `toolkits/profile_libero_step_latency.py` 包含 CLI、纯函数解析/统计、BDDL metadata 解析、worker subprocess runner 和文件输出。测试集中在 `tests/unit_tests/test_libero_step_latency_profiler.py`，用 mock env 覆盖不依赖 LIBERO/MuJoCo 的路径。真实 LIBERO 运行作为手动命令，不进 CI。

**技术栈：** Python dataclasses、argparse、csv、json/jsonl、multiprocessing、os CPU affinity、time.perf_counter、NumPy、pytest。

---

## 文件结构

- 创建：`toolkits/profile_libero_step_latency.py`
  - 职责：standalone CLI；不导入 Ray；提供可测试纯函数；运行真实 LIBERO profiling；写 JSONL/CSV/JSON 输出。
  - 主要单元：
    - `ProfileConfig`：CLI 解析后的配置。
    - `TaskTrialSpec`：一个 suite/task/trial measurement 单元。
    - `parse_task_ids()`、`parse_int_list()`、`parse_dummy_action()`：CLI 值解析。
    - `select_trial_ids()`：可复现 trial 选择。
    - `compute_latency_summary()`：latency 分位数和 tail ratio。
    - `parse_bddl_metadata()`：静态 BDDL metadata。
    - `collect_runtime_metadata()`：best-effort MuJoCo/env metadata。
    - `profile_task_trial()`：子进程中执行一个 task/trial。
    - `run_profile()`：主进程枚举任务并写输出文件。
- 创建：`tests/unit_tests/test_libero_step_latency_profiler.py`
  - 职责：覆盖纯函数、BDDL parser、mock env profiling、输出 schema，不要求真实 LIBERO。

## 任务 1：CLI 解析与 summary 纯函数

**文件：**
- 创建：`toolkits/profile_libero_step_latency.py`
- 创建：`tests/unit_tests/test_libero_step_latency_profiler.py`

- [ ] **步骤 1：编写失败的解析和统计测试**

在 `tests/unit_tests/test_libero_step_latency_profiler.py` 添加：

```python
import math

import numpy as np
import pytest

from toolkits.profile_libero_step_latency import (
    compute_latency_summary,
    parse_dummy_action,
    parse_int_list,
    parse_task_ids,
    select_trial_ids,
)


def test_parse_int_list_accepts_all_and_numbers():
    assert parse_int_list("all", allow_all=True) == "all"
    assert parse_int_list("0,3,7", allow_all=True) == [0, 3, 7]
    assert parse_int_list(" 1 , 2 ") == [1, 2]


def test_parse_int_list_rejects_empty_and_all_when_disallowed():
    with pytest.raises(ValueError, match="empty"):
        parse_int_list("")
    with pytest.raises(ValueError, match="not allowed"):
        parse_int_list("all", allow_all=False)


def test_parse_task_ids_all_uses_task_count():
    assert parse_task_ids("all", num_tasks=4) == [0, 1, 2, 3]
    assert parse_task_ids("0,2", num_tasks=4) == [0, 2]


def test_parse_task_ids_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        parse_task_ids("4", num_tasks=4)


def test_select_trial_ids_specific_ids_win():
    assert select_trial_ids(
        num_trials=10,
        trials_per_task=2,
        specific_trial_ids=[5, 1],
        seed=0,
        task_id=3,
    ) == [5, 1]


def test_select_trial_ids_deterministic_prefix_when_under_limit():
    assert select_trial_ids(
        num_trials=3,
        trials_per_task=10,
        specific_trial_ids=None,
        seed=123,
        task_id=0,
    ) == [0, 1, 2]


def test_select_trial_ids_seeded_sample_is_stable():
    first = select_trial_ids(
        num_trials=20,
        trials_per_task=4,
        specific_trial_ids=None,
        seed=123,
        task_id=2,
    )
    second = select_trial_ids(
        num_trials=20,
        trials_per_task=4,
        specific_trial_ids=None,
        seed=123,
        task_id=2,
    )
    assert first == second
    assert len(first) == 4
    assert len(set(first)) == 4


def test_parse_dummy_action_default_and_override():
    np.testing.assert_allclose(parse_dummy_action(None), [0, 0, 0, 0, 0, 0, -1])
    np.testing.assert_allclose(parse_dummy_action("1,2,3"), [1, 2, 3])


def test_compute_latency_summary_percentiles_and_tail_ratio():
    summary = compute_latency_summary([0.01, 0.02, 0.03, 0.10])
    assert summary["step_count"] == 4
    assert math.isclose(summary["mean_latency_s"], 0.04)
    assert math.isclose(summary["median_latency_s"], 0.025)
    assert summary["p99_latency_s"] > summary["p95_latency_s"]
    assert math.isclose(
        summary["tail_ratio_p99_to_median"],
        summary["p99_latency_s"] / summary["median_latency_s"],
    )


def test_compute_latency_summary_empty_latency_list():
    summary = compute_latency_summary([])
    assert summary["step_count"] == 0
    assert summary["mean_latency_s"] is None
    assert summary["tail_ratio_p99_to_median"] is None
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_libero_step_latency_profiler.py -q
```

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'toolkits.profile_libero_step_latency'`。

- [ ] **步骤 3：实现最少解析和统计代码**

创建 `toolkits/profile_libero_step_latency.py`，先加入：

```python
"""Profile LIBERO simulator step latency without Ray or policy models."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


@dataclass(frozen=True)
class ProfileConfig:
    suite: str
    task_ids: str
    trials_per_task: int
    specific_trial_ids: list[int] | None
    warmup_steps: int
    measure_steps: int
    cpu_id: int | None
    cpu_ids: list[int] | None
    camera_height: int
    camera_width: int
    libero_type: str
    seed: int
    output_dir: Path
    dummy_action: list[float]
    stop_on_done: bool


@dataclass(frozen=True)
class TaskTrialSpec:
    suite_name: str
    task_id: int
    trial_id: int
    task_name: str
    task_language: str
    bddl_file: str
    seed: int


def parse_int_list(value: str, *, allow_all: bool = False) -> list[int] | str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("empty integer list")
    if stripped.lower() == "all":
        if allow_all:
            return "all"
        raise ValueError("'all' is not allowed here")
    result: list[int] = []
    for part in stripped.split(","):
        item = part.strip()
        if not item:
            raise ValueError(f"empty item in integer list: {value!r}")
        result.append(int(item))
    return result


def parse_task_ids(value: str, *, num_tasks: int) -> list[int]:
    parsed = parse_int_list(value, allow_all=True)
    if parsed == "all":
        return list(range(num_tasks))
    task_ids = parsed
    for task_id in task_ids:
        if task_id < 0 or task_id >= num_tasks:
            raise ValueError(f"task id {task_id} out of range [0, {num_tasks})")
    return task_ids


def select_trial_ids(
    *,
    num_trials: int,
    trials_per_task: int,
    specific_trial_ids: list[int] | None,
    seed: int,
    task_id: int,
) -> list[int]:
    if num_trials <= 0:
        return []
    if specific_trial_ids is not None:
        for trial_id in specific_trial_ids:
            if trial_id < 0 or trial_id >= num_trials:
                raise ValueError(
                    f"trial id {trial_id} out of range [0, {num_trials})"
                )
        return list(specific_trial_ids)
    if trials_per_task >= num_trials:
        return list(range(num_trials))
    rng = np.random.default_rng(seed + task_id)
    selected = rng.choice(num_trials, size=trials_per_task, replace=False)
    return [int(item) for item in selected.tolist()]


def parse_dummy_action(value: str | None) -> list[float]:
    if value is None:
        return list(DEFAULT_DUMMY_ACTION)
    stripped = value.strip()
    if not stripped:
        raise ValueError("empty dummy action")
    return [float(part.strip()) for part in stripped.split(",")]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def compute_latency_summary(latencies: list[float]) -> dict[str, float | int | None]:
    if not latencies:
        return {
            "step_count": 0,
            "mean_latency_s": None,
            "median_latency_s": None,
            "p90_latency_s": None,
            "p95_latency_s": None,
            "p99_latency_s": None,
            "min_latency_s": None,
            "max_latency_s": None,
            "std_latency_s": None,
            "tail_ratio_p99_to_median": None,
        }
    array = np.asarray(latencies, dtype=np.float64)
    median = float(np.median(array))
    p99 = _percentile(latencies, 99)
    return {
        "step_count": int(array.size),
        "mean_latency_s": float(np.mean(array)),
        "median_latency_s": median,
        "p90_latency_s": _percentile(latencies, 90),
        "p95_latency_s": _percentile(latencies, 95),
        "p99_latency_s": p99,
        "min_latency_s": float(np.min(array)),
        "max_latency_s": float(np.max(array)),
        "std_latency_s": float(np.std(array)),
        "tail_ratio_p99_to_median": (
            None if median == 0.0 or p99 is None else float(p99 / median)
        ),
    }
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_libero_step_latency_profiler.py -q
```

预期：PASS，所有解析和统计测试通过。

- [ ] **步骤 5：格式检查**

运行：

```bash
ruff format toolkits/profile_libero_step_latency.py tests/unit_tests/test_libero_step_latency_profiler.py
ruff check toolkits/profile_libero_step_latency.py tests/unit_tests/test_libero_step_latency_profiler.py
```

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add toolkits/profile_libero_step_latency.py tests/unit_tests/test_libero_step_latency_profiler.py
git commit -s -m "feat: add libero profiling parsing helpers"
```

## 任务 2：BDDL metadata parser

**文件：**
- 修改：`toolkits/profile_libero_step_latency.py`
- 修改：`tests/unit_tests/test_libero_step_latency_profiler.py`

- [ ] **步骤 1：编写失败的 BDDL 解析测试**

追加到 `tests/unit_tests/test_libero_step_latency_profiler.py`：

```python
from pathlib import Path

from toolkits.profile_libero_step_latency import parse_bddl_metadata


SAMPLE_BDDL = """
(define (problem LIBERO_Kitchen_Tabletop_Manipulation)
  (:domain robosuite)
  (:language turn on the stove and put the frying pan on it)
  (:regions
    (flat_stove_init_region
      (:target kitchen_table)
      (:ranges ((-0.21 0.19 -0.19 0.21)))
    )
    (cook_region
      (:target flat_stove_1)
    )
  )
  (:fixtures
    kitchen_table - kitchen_table
    flat_stove_1 - flat_stove
  )
  (:objects
    chefmate_8_frypan_1 - chefmate_8_frypan
    moka_pot_1 - moka_pot
  )
  (:obj_of_interest
    chefmate_8_frypan_1
    flat_stove_1
  )
  (:init
    (On flat_stove_1 kitchen_table_flat_stove_init_region)
    (On chefmate_8_frypan_1 kitchen_table_frypan_init_region)
  )
  (:goal
    (And (Turnon flat_stove_1) (On chefmate_8_frypan_1 flat_stove_1_cook_region))
  )
)
"""


def test_parse_bddl_metadata_counts_sections(tmp_path: Path):
    bddl_path = tmp_path / "KITCHEN_SCENE3_turn_on_the_stove.bddl"
    bddl_path.write_text(SAMPLE_BDDL)

    metadata = parse_bddl_metadata(bddl_path)

    assert metadata["problem_name"] == "LIBERO_Kitchen_Tabletop_Manipulation"
    assert metadata["domain_name"] == "robosuite"
    assert metadata["task_language"] == "turn on the stove and put the frying pan on it"
    assert metadata["scene_type"] == "kitchen"
    assert metadata["scene_name"] == "KITCHEN_SCENE3"
    assert metadata["num_regions"] == 2
    assert metadata["num_fixtures"] == 2
    assert metadata["fixture_categories"] == ["flat_stove", "kitchen_table"]
    assert metadata["num_objects"] == 2
    assert metadata["object_categories"] == ["chefmate_8_frypan", "moka_pot"]
    assert metadata["num_obj_of_interest"] == 2
    assert metadata["num_init_predicates"] == 2
    assert metadata["num_goal_predicates"] == 2
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_libero_step_latency_profiler.py::test_parse_bddl_metadata_counts_sections -q
```

预期：FAIL，报错包含 `ImportError` 或 `AttributeError`，因为 `parse_bddl_metadata` 尚未实现。

- [ ] **步骤 3：实现 BDDL parser**

在 `toolkits/profile_libero_step_latency.py` 中加入：

```python
import re


BDDL_SCHEMA_KEYS = [
    "problem_name",
    "domain_name",
    "task_language",
    "scene_type",
    "scene_name",
    "region_names",
    "num_regions",
    "fixture_categories",
    "num_fixtures",
    "object_categories",
    "num_objects",
    "obj_of_interest",
    "num_obj_of_interest",
    "init_predicates",
    "num_init_predicates",
    "goal_predicates",
    "num_goal_predicates",
]


def _balanced_section(text: str, section_name: str) -> str:
    marker = f"(:{section_name}"
    start = text.find(marker)
    if start < 0:
        return ""
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def _extract_problem_name(text: str) -> str | None:
    match = re.search(r"\(define\s+\(problem\s+([^)]+)\)", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def _extract_simple_section_value(text: str, section_name: str) -> str | None:
    match = re.search(rf"\(:{section_name}\s+([^)]+)\)", text, flags=re.IGNORECASE)
    return " ".join(match.group(1).split()) if match else None


def _extract_category_lines(section: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("(:") or line == ")":
            continue
        if " - " not in line:
            continue
        instance, category = [part.strip() for part in line.split(" - ", 1)]
        category = category.rstrip(")")
        result.setdefault(category, []).append(instance)
    return result


def _extract_region_names(section: str) -> list[str]:
    names: list[str] = []
    body = section[len("(:regions") :].strip()
    depth = 0
    token_start: int | None = None
    for index, char in enumerate(body):
        if char == "(":
            if depth == 0:
                token_start = index + 1
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and token_start is not None:
                item = body[token_start:index].strip()
                if item and not item.startswith(":"):
                    name = item.split()[0]
                    if name and name not in names:
                        names.append(name)
                token_start = None
    return names


def _extract_predicate_names(section: str) -> list[str]:
    names = re.findall(r"\(([A-Za-z_][A-Za-z0-9_]*)\b", section)
    return [
        name
        for name in names
        if name.lower()
        not in {
            "init",
            "goal",
            "and",
        }
    ]


def _infer_scene_type(problem_name: str | None, bddl_path: Path) -> str | None:
    source = f"{problem_name or ''} {bddl_path.name}".lower()
    if "living_room" in source:
        return "living_room"
    if "coffee_table" in source:
        return "coffee_table"
    if "kitchen" in source:
        return "kitchen"
    if "study" in source:
        return "study"
    if "floor" in source:
        return "floor"
    if "tabletop" in source or "table" in source:
        return "table"
    return None


def _infer_scene_name(bddl_path: Path) -> str | None:
    match = re.match(r"([A-Z]+_SCENE\d+)", bddl_path.stem)
    return match.group(1) if match else None


def parse_bddl_metadata(bddl_path: str | Path) -> dict[str, Any]:
    path = Path(bddl_path)
    text = path.read_text()
    problem_name = _extract_problem_name(text)
    domain_name = _extract_simple_section_value(text, "domain")
    task_language = _extract_simple_section_value(text, "language")
    regions_section = _balanced_section(text, "regions")
    fixtures = _extract_category_lines(_balanced_section(text, "fixtures"))
    objects = _extract_category_lines(_balanced_section(text, "objects"))
    obj_of_interest_section = _balanced_section(text, "obj_of_interest")
    init_predicates = _extract_predicate_names(_balanced_section(text, "init"))
    goal_predicates = _extract_predicate_names(_balanced_section(text, "goal"))
    obj_of_interest = [
        line.strip()
        for line in obj_of_interest_section.splitlines()[1:]
        if line.strip() and line.strip() != ")"
    ]
    metadata: dict[str, Any] = {
        "problem_name": problem_name,
        "domain_name": domain_name,
        "task_language": task_language,
        "scene_type": _infer_scene_type(problem_name, path),
        "scene_name": _infer_scene_name(path),
        "region_names": _extract_region_names(regions_section),
        "fixture_categories": sorted(fixtures),
        "object_categories": sorted(objects),
        "obj_of_interest": obj_of_interest,
        "init_predicates": init_predicates,
        "goal_predicates": goal_predicates,
    }
    metadata["num_regions"] = len(metadata["region_names"])
    metadata["num_fixtures"] = sum(len(items) for items in fixtures.values())
    metadata["num_objects"] = sum(len(items) for items in objects.values())
    metadata["num_obj_of_interest"] = len(obj_of_interest)
    metadata["num_init_predicates"] = len(init_predicates)
    metadata["num_goal_predicates"] = len(goal_predicates)
    for key in BDDL_SCHEMA_KEYS:
        metadata.setdefault(key, None)
    return metadata
```

- [ ] **步骤 4：运行 BDDL 测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_libero_step_latency_profiler.py::test_parse_bddl_metadata_counts_sections -q
```

预期：PASS。

- [ ] **步骤 5：运行已有 profiler 单测**

运行：

```bash
pytest tests/unit_tests/test_libero_step_latency_profiler.py -q
```

预期：PASS。

- [ ] **步骤 6：格式检查**

运行：

```bash
ruff format toolkits/profile_libero_step_latency.py tests/unit_tests/test_libero_step_latency_profiler.py
ruff check toolkits/profile_libero_step_latency.py tests/unit_tests/test_libero_step_latency_profiler.py
```

预期：PASS。

- [ ] **步骤 7：Commit**

```bash
git add toolkits/profile_libero_step_latency.py tests/unit_tests/test_libero_step_latency_profiler.py
git commit -s -m "feat: parse libero bddl profiling metadata"
```

## 任务 3：单 task/trial profiling 核心与 mock env 测试

**文件：**
- 修改：`toolkits/profile_libero_step_latency.py`
- 修改：`tests/unit_tests/test_libero_step_latency_profiler.py`

- [ ] **步骤 1：编写失败的 mock profiling 测试**

追加到 `tests/unit_tests/test_libero_step_latency_profiler.py`：

```python
from toolkits.profile_libero_step_latency import ProfileConfig, TaskTrialSpec, profile_task_trial


class FakeModel:
    nbody = 4
    ngeom = 5
    njnt = 2
    nq = 9
    nv = 8
    nu = 7
    ncam = 2
    camera_names = ["agentview", "robot0_eye_in_hand"]


class FakeSim:
    model = FakeModel()


class FakeEnv:
    def __init__(self):
        self.sim = FakeSim()
        self.steps = 0
        self.closed = False

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        return {"obs": 1}

    def set_init_state(self, init_state):
        self.init_state = init_state
        return {"obs": 2}

    def step(self, action):
        self.steps += 1
        done = self.steps >= 3
        return {"obs": self.steps}, float(done), done, {"success": done}

    def check_success(self):
        return self.steps >= 3

    def close(self):
        self.closed = True


def test_profile_task_trial_with_mock_env(tmp_path: Path):
    bddl_path = tmp_path / "KITCHEN_SCENE3_task.bddl"
    bddl_path.write_text(SAMPLE_BDDL)
    config = ProfileConfig(
        suite="libero_90",
        task_ids="0",
        trials_per_task=1,
        specific_trial_ids=None,
        warmup_steps=1,
        measure_steps=3,
        cpu_id=None,
        cpu_ids=None,
        camera_height=64,
        camera_width=64,
        libero_type="standard",
        seed=11,
        output_dir=tmp_path,
        dummy_action=[0.0] * 7,
        stop_on_done=False,
    )
    spec = TaskTrialSpec(
        suite_name="libero_90",
        task_id=0,
        trial_id=1,
        task_name="KITCHEN_SCENE3_task",
        task_language="turn on the stove",
        bddl_file=str(bddl_path),
        seed=11,
    )

    result = profile_task_trial(
        config=config,
        spec=spec,
        env_factory=FakeEnv,
        init_state=np.zeros(3),
        clock=lambda: 1.0,
    )

    assert result.error is None
    assert len(result.events) == 3
    assert result.events[0]["event"] == "libero_step_latency"
    assert result.events[0]["suite_name"] == "libero_90"
    assert result.events[0]["num_objects"] == 2
    assert result.events[0]["nbody"] == 4
    assert result.summary["step_count"] == 3
    assert result.summary["success_seen"] is True
    assert result.summary["done_seen_step"] == 2
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_libero_step_latency_profiler.py::test_profile_task_trial_with_mock_env -q
```

预期：FAIL，报错包含 `ImportError` 或 `AttributeError`，因为 `profile_task_trial` 尚未实现。

- [ ] **步骤 3：实现 runtime metadata 和 profiling 核心**

在 `toolkits/profile_libero_step_latency.py` 中加入：

```python
@dataclass
class ProfileResult:
    events: list[dict[str, Any]]
    summary: dict[str, Any] | None
    error: dict[str, Any] | None


RUNTIME_METADATA_KEYS = [
    "camera_names",
    "camera_heights",
    "camera_widths",
    "renderer",
    "nbody",
    "ngeom",
    "njnt",
    "nq",
    "nv",
    "nu",
    "ncam",
]


def collect_runtime_metadata(env: Any, config: ProfileConfig) -> dict[str, Any]:
    model = getattr(getattr(env, "sim", None), "model", None)
    camera_names = getattr(model, "camera_names", None)
    metadata = {
        "camera_names": list(camera_names) if camera_names is not None else None,
        "camera_heights": config.camera_height,
        "camera_widths": config.camera_width,
        "renderer": "mujoco",
        "nbody": getattr(model, "nbody", None),
        "ngeom": getattr(model, "ngeom", None),
        "njnt": getattr(model, "njnt", None),
        "nq": getattr(model, "nq", None),
        "nv": getattr(model, "nv", None),
        "nu": getattr(model, "nu", None),
        "ncam": getattr(model, "ncam", None),
    }
    for key in RUNTIME_METADATA_KEYS:
        metadata.setdefault(key, None)
    return metadata


def _apply_cpu_affinity(cpu_id: int | None) -> bool:
    if cpu_id is None:
        return False
    if not hasattr(os, "sched_setaffinity"):
        return False
    try:
        os.sched_setaffinity(0, {cpu_id})
    except OSError:
        return False
    return True


def _step_env(env: Any, action: list[float]) -> tuple[Any, float, bool, dict[str, Any]]:
    obs, reward, done, info = env.step(np.asarray(action, dtype=np.float32))
    if info is None:
        info = {}
    return obs, float(reward), bool(done), dict(info)


def _event_base(
    *,
    config: ProfileConfig,
    spec: TaskTrialSpec,
    cpu_affinity_applied: bool,
    bddl_metadata: dict[str, Any],
    runtime_metadata: dict[str, Any],
) -> dict[str, Any]:
    base = {
        "suite_name": spec.suite_name,
        "task_id": spec.task_id,
        "trial_id": spec.trial_id,
        "task_name": spec.task_name,
        "task_language": spec.task_language,
        "cpu_id": config.cpu_id,
        "pid": os.getpid(),
        "seed": spec.seed,
        "cpu_affinity_applied": cpu_affinity_applied,
        "bddl_file": spec.bddl_file,
    }
    base.update(bddl_metadata)
    base.update(runtime_metadata)
    base["task_language"] = spec.task_language or bddl_metadata.get("task_language")
    return base


def _error_record(
    *,
    spec: TaskTrialSpec,
    message: str,
    exc_type: str,
) -> dict[str, Any]:
    return {
        "event": "error",
        "suite_name": spec.suite_name,
        "task_id": spec.task_id,
        "trial_id": spec.trial_id,
        "task_name": spec.task_name,
        "task_language": spec.task_language,
        "bddl_file": spec.bddl_file,
        "error_type": exc_type,
        "error": message,
    }


def profile_task_trial(
    *,
    config: ProfileConfig,
    spec: TaskTrialSpec,
    env_factory: Any,
    init_state: Any,
    clock: Any = time.perf_counter,
) -> ProfileResult:
    env = None
    cpu_affinity_applied = _apply_cpu_affinity(config.cpu_id)
    try:
        bddl_metadata = parse_bddl_metadata(spec.bddl_file)
        env = env_factory()
        if hasattr(env, "seed"):
            env.seed(spec.seed)
        env.reset()
        if init_state is not None and hasattr(env, "set_init_state"):
            env.set_init_state(init_state)
        runtime_metadata = collect_runtime_metadata(env, config)
        for _ in range(config.warmup_steps):
            _step_env(env, config.dummy_action)
        base = _event_base(
            config=config,
            spec=spec,
            cpu_affinity_applied=cpu_affinity_applied,
            bddl_metadata=bddl_metadata,
            runtime_metadata=runtime_metadata,
        )
        events: list[dict[str, Any]] = []
        latencies: list[float] = []
        done_seen_step: int | None = None
        success_seen = False
        for step_index in range(config.measure_steps):
            start = clock()
            _, reward, done, info = _step_env(env, config.dummy_action)
            end = clock()
            latency_s = max(float(end - start), 0.0)
            success = bool(info.get("success", False))
            if hasattr(env, "check_success"):
                success = success or bool(env.check_success())
            if done and done_seen_step is None:
                done_seen_step = step_index
            success_seen = success_seen or success
            latencies.append(latency_s)
            event = {
                "event": "libero_step_latency",
                **base,
                "step_index": step_index,
                "latency_s": latency_s,
                "reward": reward,
                "done": done,
                "success": success,
                "done_seen_step": done_seen_step,
            }
            events.append(event)
            if done and config.stop_on_done:
                break
        summary = {
            **base,
            **compute_latency_summary(latencies),
            "done_seen_step": done_seen_step,
            "success_seen": success_seen,
            "error": None,
        }
        return ProfileResult(events=events, summary=summary, error=None)
    except Exception as exc:
        return ProfileResult(
            events=[],
            summary=None,
            error=_error_record(
                spec=spec,
                message=str(exc),
                exc_type=exc.__class__.__name__,
            ),
        )
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()
```

- [ ] **步骤 4：运行 mock profiling 测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_libero_step_latency_profiler.py::test_profile_task_trial_with_mock_env -q
```

预期：PASS。

- [ ] **步骤 5：运行 profiler 单测**

运行：

```bash
pytest tests/unit_tests/test_libero_step_latency_profiler.py -q
```

预期：PASS。

- [ ] **步骤 6：格式检查**

运行：

```bash
ruff format toolkits/profile_libero_step_latency.py tests/unit_tests/test_libero_step_latency_profiler.py
ruff check toolkits/profile_libero_step_latency.py tests/unit_tests/test_libero_step_latency_profiler.py
```

预期：PASS。

- [ ] **步骤 7：Commit**

```bash
git add toolkits/profile_libero_step_latency.py tests/unit_tests/test_libero_step_latency_profiler.py
git commit -s -m "feat: profile libero task trial latency"
```

## 任务 4：真实 LIBERO suite 枚举与文件输出

**文件：**
- 修改：`toolkits/profile_libero_step_latency.py`
- 修改：`tests/unit_tests/test_libero_step_latency_profiler.py`

- [ ] **步骤 1：编写失败的输出测试**

追加到 `tests/unit_tests/test_libero_step_latency_profiler.py`：

```python
import json

from toolkits.profile_libero_step_latency import (
    append_jsonl,
    write_run_config,
    write_summary_files,
)


def test_output_writers_create_jsonl_csv_json(tmp_path: Path):
    event_path = tmp_path / "step_latency_events.jsonl"
    append_jsonl(event_path, [{"event": "a", "value": 1}, {"event": "b", "value": 2}])
    records = [json.loads(line) for line in event_path.read_text().splitlines()]
    assert records == [{"event": "a", "value": 1}, {"event": "b", "value": 2}]

    summaries = [
        {"suite_name": "libero_90", "task_id": 0, "step_count": 2, "error": None},
        {"suite_name": "libero_90", "task_id": 1, "step_count": 2, "error": None},
    ]
    write_summary_files(tmp_path, summaries)
    assert (tmp_path / "step_latency_summary.csv").exists()
    summary_json = json.loads((tmp_path / "step_latency_summary.json").read_text())
    assert summary_json == summaries


def test_write_run_config_serializes_paths(tmp_path: Path):
    config = ProfileConfig(
        suite="libero_90",
        task_ids="all",
        trials_per_task=1,
        specific_trial_ids=None,
        warmup_steps=1,
        measure_steps=1,
        cpu_id=0,
        cpu_ids=None,
        camera_height=64,
        camera_width=64,
        libero_type="standard",
        seed=0,
        output_dir=tmp_path,
        dummy_action=[0.0] * 7,
        stop_on_done=False,
    )
    write_run_config(tmp_path, config)
    data = json.loads((tmp_path / "run_config.json").read_text())
    assert data["output_dir"] == str(tmp_path)
    assert data["suite"] == "libero_90"
```

- [ ] **步骤 2：运行输出测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_libero_step_latency_profiler.py::test_output_writers_create_jsonl_csv_json tests/unit_tests/test_libero_step_latency_profiler.py::test_write_run_config_serializes_paths -q
```

预期：FAIL，报错包含 `ImportError`，因为 writer 函数尚未实现。

- [ ] **步骤 3：实现输出 writer**

在 `toolkits/profile_libero_step_latency.py` 中加入：

```python
def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _jsonable_config(config: ProfileConfig) -> dict[str, Any]:
    data = asdict(config)
    data["output_dir"] = str(config.output_dir)
    return data


def write_run_config(output_dir: Path, config: ProfileConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(_jsonable_config(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_summary_files(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "step_latency_summary.json"
    json_path.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = output_dir / "step_latency_summary.csv"
    fieldnames = sorted({key for summary in summaries for key in summary})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)
```

- [ ] **步骤 4：实现真实 LIBERO imports 和 suite 枚举**

在 `toolkits/profile_libero_step_latency.py` 中加入：

```python
def _configure_libero_type(libero_type: str) -> None:
    os.environ["LIBERO_TYPE"] = libero_type


def _import_libero_modules(libero_type: str) -> tuple[Any, Any, Any]:
    _configure_libero_type(libero_type)
    if libero_type == "pro":
        from liberopro.liberopro import benchmark, get_libero_path
        from liberopro.liberopro.envs import OffScreenRenderEnv
    elif libero_type == "plus":
        from liberoplus.liberoplus import benchmark, get_libero_path
        from liberoplus.liberoplus.envs import OffScreenRenderEnv
    else:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    return benchmark, get_libero_path, OffScreenRenderEnv


def _bddl_path_for_task(get_libero_path: Any, task: Any) -> str:
    return str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)


def build_task_trial_specs(config: ProfileConfig) -> tuple[list[TaskTrialSpec], list[Any]]:
    benchmark, get_libero_path, _ = _import_libero_modules(config.libero_type)
    bench = benchmark.get_benchmark(config.suite)()
    task_ids = parse_task_ids(config.task_ids, num_tasks=bench.get_num_tasks())
    specs: list[TaskTrialSpec] = []
    init_states: list[Any] = []
    for task_id in task_ids:
        task = bench.get_task(task_id)
        task_init_states = bench.get_task_init_states(task_id)
        trial_ids = select_trial_ids(
            num_trials=len(task_init_states),
            trials_per_task=config.trials_per_task,
            specific_trial_ids=config.specific_trial_ids,
            seed=config.seed,
            task_id=task_id,
        )
        for trial_id in trial_ids:
            specs.append(
                TaskTrialSpec(
                    suite_name=config.suite,
                    task_id=task_id,
                    trial_id=trial_id,
                    task_name=Path(task.bddl_file).stem,
                    task_language=task.language,
                    bddl_file=_bddl_path_for_task(get_libero_path, task),
                    seed=config.seed + task_id * 100000 + trial_id,
                )
            )
            init_states.append(task_init_states[trial_id])
    return specs, init_states


def make_libero_env_factory(config: ProfileConfig, spec: TaskTrialSpec) -> Any:
    _, _, OffScreenRenderEnv = _import_libero_modules(config.libero_type)

    def factory() -> Any:
        return OffScreenRenderEnv(
            bddl_file_name=spec.bddl_file,
            camera_heights=config.camera_height,
            camera_widths=config.camera_width,
        )

    return factory
```

- [ ] **步骤 5：实现 sequential run_profile**

在 `toolkits/profile_libero_step_latency.py` 中加入：

```python
def _profile_subprocess_entry(
    queue: Any,
    config: ProfileConfig,
    spec: TaskTrialSpec,
    init_state: Any,
) -> None:
    result = profile_task_trial(
        config=config,
        spec=spec,
        env_factory=make_libero_env_factory(config, spec),
        init_state=init_state,
    )
    queue.put(result)


def profile_task_trial_in_subprocess(
    *,
    config: ProfileConfig,
    spec: TaskTrialSpec,
    init_state: Any,
) -> ProfileResult:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_profile_subprocess_entry,
        args=(queue, config, spec, init_state),
    )
    process.start()
    process.join()
    if process.exitcode != 0:
        return ProfileResult(
            events=[],
            summary=None,
            error=_error_record(
                spec=spec,
                message=f"profiling subprocess exited with code {process.exitcode}",
                exc_type="SubprocessError",
            ),
        )
    try:
        return queue.get_nowait()
    except Exception:
        return ProfileResult(
            events=[],
            summary=None,
            error=_error_record(
                spec=spec,
                message="profiling subprocess produced no result",
                exc_type="SubprocessError",
            ),
        )


def run_profile(config: ProfileConfig) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(config.output_dir, config)
    events_path = config.output_dir / "step_latency_events.jsonl"
    errors_path = config.output_dir / "errors.jsonl"
    specs, init_states = build_task_trial_specs(config)
    summaries: list[dict[str, Any]] = []
    for spec, init_state in zip(specs, init_states):
        result = profile_task_trial_in_subprocess(
            config=config,
            spec=spec,
            init_state=init_state,
        )
        append_jsonl(events_path, result.events)
        if result.summary is not None:
            summaries.append(result.summary)
        if result.error is not None:
            append_jsonl(errors_path, [result.error])
    write_summary_files(config.output_dir, summaries)
    return 0
```

- [ ] **步骤 6：添加 subprocess wrapper 单测**

追加到 `tests/unit_tests/test_libero_step_latency_profiler.py`：

```python
from toolkits.profile_libero_step_latency import profile_task_trial_in_subprocess


def test_profile_task_trial_in_subprocess_reports_nonzero_exit(
    monkeypatch, tmp_path: Path
):
    bddl_path = tmp_path / "KITCHEN_SCENE3_task.bddl"
    bddl_path.write_text(SAMPLE_BDDL)
    config = ProfileConfig(
        suite="libero_90",
        task_ids="0",
        trials_per_task=1,
        specific_trial_ids=None,
        warmup_steps=0,
        measure_steps=1,
        cpu_id=None,
        cpu_ids=None,
        camera_height=64,
        camera_width=64,
        libero_type="standard",
        seed=11,
        output_dir=tmp_path,
        dummy_action=[0.0] * 7,
        stop_on_done=False,
    )
    spec = TaskTrialSpec(
        suite_name="libero_90",
        task_id=0,
        trial_id=1,
        task_name="KITCHEN_SCENE3_task",
        task_language="turn on the stove",
        bddl_file=str(bddl_path),
        seed=11,
    )

    class FakeQueue:
        def __init__(self, maxsize=1):
            self.maxsize = maxsize

        def get_nowait(self):
            raise RuntimeError("empty queue")

    class FakeProcess:
        exitcode = 9

        def __init__(self, target, args):
            self.target = target
            self.args = args

        def start(self):
            return None

        def join(self):
            return None

    class FakeContext:
        Queue = FakeQueue
        Process = FakeProcess

    monkeypatch.setattr(
        "toolkits.profile_libero_step_latency.mp.get_context",
        lambda method: FakeContext(),
    )

    result = profile_task_trial_in_subprocess(
        config=config,
        spec=spec,
        init_state=np.zeros(3),
    )

    assert result.events == []
    assert result.summary is None
    assert result.error["error_type"] == "SubprocessError"
    assert "exited with code 9" in result.error["error"]
```

- [ ] **步骤 7：运行输出测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_libero_step_latency_profiler.py::test_output_writers_create_jsonl_csv_json tests/unit_tests/test_libero_step_latency_profiler.py::test_write_run_config_serializes_paths -q
```

预期：PASS。

- [ ] **步骤 8：运行 subprocess wrapper 单测**

运行：

```bash
pytest tests/unit_tests/test_libero_step_latency_profiler.py::test_profile_task_trial_in_subprocess_reports_nonzero_exit -q
```

预期：PASS。

- [ ] **步骤 9：运行 profiler 单测**

运行：

```bash
pytest tests/unit_tests/test_libero_step_latency_profiler.py -q
```

预期：PASS。

- [ ] **步骤 10：格式检查**

运行：

```bash
ruff format toolkits/profile_libero_step_latency.py tests/unit_tests/test_libero_step_latency_profiler.py
ruff check toolkits/profile_libero_step_latency.py tests/unit_tests/test_libero_step_latency_profiler.py
```

预期：PASS。

- [ ] **步骤 11：Commit**

```bash
git add toolkits/profile_libero_step_latency.py tests/unit_tests/test_libero_step_latency_profiler.py
git commit -s -m "feat: write libero profiling outputs"
```

## 任务 5：CLI main 与手动运行文档

**文件：**
- 修改：`toolkits/profile_libero_step_latency.py`
- 修改：`docs/superpowers/specs/2026-05-16-libero-step-latency-profiling-design.md`

- [ ] **步骤 1：编写 CLI parser 代码**

在 `toolkits/profile_libero_step_latency.py` 中加入：

```python
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-ids", default="all")
    parser.add_argument("--trials-per-task", type=int, default=1)
    parser.add_argument("--specific-trial-ids", default=None)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=200)
    parser.add_argument("--cpu-id", type=int, default=None)
    parser.add_argument("--cpu-ids", default=None)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument(
        "--libero-type",
        choices=["standard", "pro", "plus"],
        default="standard",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dummy-action", default=None)
    parser.add_argument("--stop-on-done", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> ProfileConfig:
    specific_trial_ids = (
        None
        if args.specific_trial_ids is None
        else parse_int_list(args.specific_trial_ids, allow_all=False)
    )
    cpu_ids = None if args.cpu_ids is None else parse_int_list(args.cpu_ids)
    if args.trials_per_task < 1:
        raise ValueError("--trials-per-task must be >= 1")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")
    if args.measure_steps < 1:
        raise ValueError("--measure-steps must be >= 1")
    return ProfileConfig(
        suite=args.suite,
        task_ids=args.task_ids,
        trials_per_task=args.trials_per_task,
        specific_trial_ids=specific_trial_ids,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        cpu_id=args.cpu_id,
        cpu_ids=cpu_ids,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        libero_type=args.libero_type,
        seed=args.seed,
        output_dir=args.output_dir,
        dummy_action=parse_dummy_action(args.dummy_action),
        stop_on_done=args.stop_on_done,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return run_profile(config)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **步骤 2：添加 CLI config 单测**

追加到 `tests/unit_tests/test_libero_step_latency_profiler.py`：

```python
from toolkits.profile_libero_step_latency import build_arg_parser, config_from_args


def test_config_from_args_parses_cli_values(tmp_path: Path):
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--suite",
            "libero_90",
            "--task-ids",
            "0,1",
            "--trials-per-task",
            "2",
            "--specific-trial-ids",
            "3,4",
            "--warmup-steps",
            "5",
            "--measure-steps",
            "6",
            "--cpu-id",
            "0",
            "--cpu-ids",
            "0,1",
            "--camera-height",
            "128",
            "--camera-width",
            "96",
            "--output-dir",
            str(tmp_path),
            "--dummy-action",
            "0,0,0,0,0,0,-1",
            "--stop-on-done",
        ]
    )
    config = config_from_args(args)
    assert config.suite == "libero_90"
    assert config.specific_trial_ids == [3, 4]
    assert config.cpu_ids == [0, 1]
    assert config.camera_height == 128
    assert config.camera_width == 96
    assert config.stop_on_done is True
```

- [ ] **步骤 3：运行 CLI 单测**

运行：

```bash
pytest tests/unit_tests/test_libero_step_latency_profiler.py::test_config_from_args_parses_cli_values -q
```

预期：PASS。

- [ ] **步骤 4：更新设计文档的手动命令说明**

在 `docs/superpowers/specs/2026-05-16-libero-step-latency-profiling-design.md` 的 `Success Criteria` 前加入：

```markdown
## Manual Validation Command

After installing LIBERO dependencies, a small manual run can be started with:

```bash
python toolkits/profile_libero_step_latency.py \
  --suite libero_spatial \
  --task-ids 0 \
  --trials-per-task 1 \
  --warmup-steps 2 \
  --measure-steps 5 \
  --cpu-id 0 \
  --output-dir results/libero_step_latency_smoke
```

Expected output files:

- `results/libero_step_latency_smoke/step_latency_events.jsonl`
- `results/libero_step_latency_smoke/step_latency_summary.csv`
- `results/libero_step_latency_smoke/step_latency_summary.json`
- `results/libero_step_latency_smoke/run_config.json`
```

- [ ] **步骤 5：运行完整 profiler 单测**

运行：

```bash
pytest tests/unit_tests/test_libero_step_latency_profiler.py -q
```

预期：PASS。

- [ ] **步骤 6：运行格式和 lint**

运行：

```bash
ruff format toolkits/profile_libero_step_latency.py tests/unit_tests/test_libero_step_latency_profiler.py
ruff check toolkits/profile_libero_step_latency.py tests/unit_tests/test_libero_step_latency_profiler.py
```

预期：PASS。

- [ ] **步骤 7：Commit**

```bash
git add toolkits/profile_libero_step_latency.py tests/unit_tests/test_libero_step_latency_profiler.py docs/superpowers/specs/2026-05-16-libero-step-latency-profiling-design.md
git commit -s -m "feat: add libero step latency profiling cli"
```

## 任务 6：最终验证

**文件：**
- 验证：`toolkits/profile_libero_step_latency.py`
- 验证：`tests/unit_tests/test_libero_step_latency_profiler.py`
- 验证：`docs/superpowers/specs/2026-05-16-libero-step-latency-profiling-design.md`

- [ ] **步骤 1：运行目标单测**

运行：

```bash
pytest tests/unit_tests/test_libero_step_latency_profiler.py -q
```

预期：PASS。

- [ ] **步骤 2：运行 lint**

运行：

```bash
ruff check toolkits/profile_libero_step_latency.py tests/unit_tests/test_libero_step_latency_profiler.py
```

预期：PASS。

- [ ] **步骤 3：运行格式检查**

运行：

```bash
ruff format --check toolkits/profile_libero_step_latency.py tests/unit_tests/test_libero_step_latency_profiler.py
```

预期：PASS。

- [ ] **步骤 4：检查 CLI help**

运行：

```bash
python toolkits/profile_libero_step_latency.py --help
```

预期：PASS，输出包含 `--suite`、`--measure-steps`、`--cpu-id`、`--output-dir`。

- [ ] **步骤 5：检查工作区只包含本任务变更**

运行：

```bash
git status --short
```

预期：只显示本 profiler 相关文件，或显示用户已有的未跟踪 config 文件但不包含未提交的 profiler 变更。

- [ ] **步骤 6：最终 commit 检查**

运行：

```bash
git log -5 --oneline
```

预期：最近提交包含本计划中的 profiler helper、BDDL metadata、profiling core、output writer、CLI 提交。

## 规格覆盖自检

- 独立 toolkit 脚本：任务 1 到任务 5 覆盖。
- 不启动 Ray、不加载模型：任务 4 的真实 LIBERO factory 只创建 `OffScreenRenderEnv`，没有 Ray 或 policy import。
- CPU affinity：任务 3 的 `_apply_cpu_affinity()` 覆盖。
- 每步 JSONL：任务 3 生成 event，任务 4 写 `step_latency_events.jsonl`。
- summary CSV/JSON：任务 1 统计，任务 4 写文件。
- BDDL metadata：任务 2 覆盖。
- MuJoCo runtime metadata：任务 3 覆盖。
- 错误隔离：任务 3 的 `ProfileResult.error` 和任务 4 的 `errors.jsonl` 覆盖。
- CLI：任务 5 覆盖。
- 单元测试不依赖真实 LIBERO：任务 1 到任务 5 的测试均使用纯函数或 mock env。
