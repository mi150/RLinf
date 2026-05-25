# Embodied Rollout Profiling 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为同步和 async embodied rollout 添加默认关闭的结构化 profiling，记录 env chunk 耗时、可用的 subenv step 耗时、每个任务首次成功步数及任务级摘要。

**架构：** 新增 `rlinf/utils/embodied_rollout_profiler.py` 作为小而独立的 JSONL profiler，`EnvWorker` 在共享的 `_run_interact_once()` 与 `env_interact_step()` 周围调用它。默认关闭时只做一次布尔判断；开启时每个 Ray env worker 写自己的 rank 文件，避免并发写同一文件。

**技术栈：** Python dataclasses、JSONL、PyTorch tensor 处理、OmegaConf DictConfig、pytest、现有 `EnvWorker`/`AsyncEnvWorker`/LIBERO subprocess venv。

---

## 文件结构

- 创建：`rlinf/utils/embodied_rollout_profiler.py`
  - 负责读取配置、构造路径、JSONL 写入、env chunk event、任务首次成功状态、任务 metadata 提取、summary 聚合。
- 修改：`rlinf/workers/env/env_worker.py`
  - 初始化 profiler，并在同步和 async 共用的 `_run_interact_once()` 中记录 chunk timing 和 epoch 结束的 task step records。
- 修改：`rlinf/envs/libero/venv.py`
  - 对 subprocess vector env 增加可选的 per-subenv step timing 缓存与读取方法，不改变默认 step 返回值。
- 修改：`rlinf/envs/libero/libero_env.py`
  - 将 LIBERO subprocess timing 转成 profiler 可消费的 `subenv_step_timing` 事件源。
- 创建：`tests/unit_tests/test_embodied_rollout_profiler.py`
  - 测试 profiler 纯逻辑、JSONL 输出、summary 计算、metadata fallback。
- 创建：`tests/unit_tests/test_env_worker_rollout_profiler.py`
  - 用轻量 fake env 验证 chunk 内首次成功步数、失败 max step、缺 metadata 不报错。
- 修改：`docs/superpowers/specs/2026-05-10-embodied-rollout-profiling-design.md`
  - 如实现中字段命名有小调整，保持规格同步。

## 任务 1：Profiler 纯单元与 JSONL 写入

**文件：**
- 创建：`rlinf/utils/embodied_rollout_profiler.py`
- 创建：`tests/unit_tests/test_embodied_rollout_profiler.py`

- [ ] **步骤 1：编写失败测试**

在 `tests/unit_tests/test_embodied_rollout_profiler.py` 添加：

```python
import json

import torch
from omegaconf import OmegaConf

from rlinf.utils.embodied_rollout_profiler import EmbodiedRolloutProfiler


def _cfg(tmp_path, enabled=True):
    return OmegaConf.create(
        {
            "runner": {
                "logger": {
                    "log_path": str(tmp_path),
                    "experiment_name": "exp",
                }
            },
            "rollout": {"profiling": {"enabled": enabled}},
            "env": {
                "train": {
                    "env_type": "fake_env",
                    "total_num_envs": 4,
                    "max_steps_per_rollout_epoch": 6,
                }
            },
        }
    )


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_disabled_profiler_writes_no_files(tmp_path):
    profiler = EmbodiedRolloutProfiler(_cfg(tmp_path, enabled=False), rank=0)
    profiler.record_env_chunk_timing(
        stage_id=0,
        rollout_epoch=0,
        chunk_index=0,
        global_step=0,
        wall_time_s=1.0,
        dones=torch.zeros((2, 2), dtype=torch.bool),
        terminations=torch.zeros((2, 2), dtype=torch.bool),
        truncations=torch.zeros((2, 2), dtype=torch.bool),
        env=None,
        obs=None,
        subenv_step_timings=None,
    )
    assert not (tmp_path / "exp" / "profiling").exists()


def test_enabled_profiler_writes_env_chunk_jsonl(tmp_path):
    profiler = EmbodiedRolloutProfiler(_cfg(tmp_path), rank=3)
    terminations = torch.tensor([[False, True], [False, False]])
    truncations = torch.tensor([[False, False], [False, True]])
    dones = terminations | truncations

    profiler.record_env_chunk_timing(
        stage_id=1,
        rollout_epoch=2,
        chunk_index=4,
        global_step=5,
        wall_time_s=0.25,
        dones=dones,
        terminations=terminations,
        truncations=truncations,
        env=None,
        obs=None,
        subenv_step_timings=None,
    )

    events_path = tmp_path / "exp" / "profiling" / "embodied_rollout_events_rank_3.jsonl"
    records = _read_jsonl(events_path)
    assert records == [
        {
            "event": "env_chunk_timing",
            "global_step": 5,
            "rollout_epoch": 2,
            "chunk_index": 4,
            "rank": 3,
            "stage_id": 1,
            "env_type": "fake_env",
            "batch_size": 2,
            "chunk_size": 2,
            "wall_time_s": 0.25,
            "timing_granularity": "batch",
            "done_count": 2,
            "success_count": 1,
            "truncation_count": 1,
        }
    ]
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/unit_tests/test_embodied_rollout_profiler.py -q`

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'rlinf.utils.embodied_rollout_profiler'`。

- [ ] **步骤 3：实现最少 profiler 骨架和 env chunk 写入**

创建 `rlinf/utils/embodied_rollout_profiler.py`：

```python
import json
import os
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig


def _to_bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _as_python(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


@dataclass
class EmbodiedRolloutProfiler:
    cfg: DictConfig
    rank: int

    def __post_init__(self) -> None:
        profiling_cfg = self.cfg.rollout.get("profiling", {})
        self.enabled = _to_bool(profiling_cfg.get("enabled", False))
        self.env_type = str(self.cfg.env.train.env_type)
        logger_cfg = self.cfg.runner.logger
        self.output_dir = os.path.join(
            str(logger_cfg.get("log_path", "logs")),
            str(logger_cfg.get("experiment_name", "default")),
            "profiling",
        )
        self.events_path = os.path.join(
            self.output_dir, f"embodied_rollout_events_rank_{self.rank}.jsonl"
        )
        self.summary_path = os.path.join(
            self.output_dir, f"task_step_summary_rank_{self.rank}.jsonl"
        )

    def _write_jsonl(self, path: str, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def record_env_chunk_timing(
        self,
        *,
        stage_id: int,
        rollout_epoch: int,
        chunk_index: int,
        global_step: int,
        wall_time_s: float,
        dones: torch.Tensor | None,
        terminations: torch.Tensor | None,
        truncations: torch.Tensor | None,
        env: Any,
        obs: dict[str, Any] | None,
        subenv_step_timings: list[dict[str, Any]] | None,
    ) -> None:
        if not self.enabled:
            return
        if dones is None:
            return

        timing_granularity = "subenv" if subenv_step_timings else "batch"
        record = {
            "event": "env_chunk_timing",
            "global_step": int(global_step),
            "rollout_epoch": int(rollout_epoch),
            "chunk_index": int(chunk_index),
            "rank": int(self.rank),
            "stage_id": int(stage_id),
            "env_type": self.env_type,
            "batch_size": int(dones.shape[0]),
            "chunk_size": int(dones.shape[1]) if dones.dim() > 1 else 1,
            "wall_time_s": float(wall_time_s),
            "timing_granularity": timing_granularity,
            "done_count": int(dones.to(torch.bool).sum().item()),
            "success_count": int(terminations.to(torch.bool).sum().item())
            if terminations is not None
            else 0,
            "truncation_count": int(truncations.to(torch.bool).sum().item())
            if truncations is not None
            else 0,
        }
        self._write_jsonl(self.events_path, record)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/unit_tests/test_embodied_rollout_profiler.py -q`

预期：2 passed。

- [ ] **步骤 5：Commit**

运行：

```bash
git add rlinf/utils/embodied_rollout_profiler.py tests/unit_tests/test_embodied_rollout_profiler.py
git commit -s -m "feat: add embodied rollout profiler writer"
```

## 任务 2：任务首次成功步数与 summary 聚合

**文件：**
- 修改：`rlinf/utils/embodied_rollout_profiler.py`
- 修改：`tests/unit_tests/test_embodied_rollout_profiler.py`

- [ ] **步骤 1：编写失败测试**

追加到 `tests/unit_tests/test_embodied_rollout_profiler.py`：

```python
class FakeEnv:
    task_ids = [7, 8]
    task_names = ["Pick", "Place"]
    task_descriptions = ["pick object", "place object"]


def test_task_episode_steps_record_first_success_and_failures(tmp_path):
    profiler = EmbodiedRolloutProfiler(_cfg(tmp_path), rank=0)
    env = FakeEnv()

    profiler.start_rollout_epoch(
        stage_id=0,
        rollout_epoch=0,
        global_step=2,
        env=env,
        obs={"task_descriptions": ["pick object", "place object"]},
        batch_size=2,
    )
    profiler.update_task_success(
        stage_id=0,
        rollout_epoch=0,
        chunk_start_step=0,
        terminations=torch.tensor([[False, True], [False, False]]),
    )
    profiler.update_task_success(
        stage_id=0,
        rollout_epoch=0,
        chunk_start_step=2,
        terminations=torch.tensor([[True, True], [False, False]]),
    )
    profiler.finish_rollout_epoch(
        stage_id=0,
        rollout_epoch=0,
        global_step=2,
        env=env,
        obs={"task_descriptions": ["pick object", "place object"]},
    )

    events_path = tmp_path / "exp" / "profiling" / "embodied_rollout_events_rank_0.jsonl"
    records = _read_jsonl(events_path)
    task_records = [record for record in records if record["event"] == "task_episode_steps"]
    assert task_records[0]["success"] is True
    assert task_records[0]["first_success_step"] == 2
    assert task_records[0]["recorded_step"] == 2
    assert task_records[0]["task_id"] == 7
    assert task_records[0]["task_name"] == "Pick"
    assert task_records[1]["success"] is False
    assert task_records[1]["first_success_step"] is None
    assert task_records[1]["recorded_step"] == 6

    summary_path = tmp_path / "exp" / "profiling" / "task_step_summary_rank_0.jsonl"
    summaries = _read_jsonl(summary_path)
    assert summaries[0]["task_id"] == 7
    assert summaries[0]["success_rate"] == 1.0
    assert summaries[0]["mean_recorded_step"] == 2.0
    assert summaries[1]["task_id"] == 8
    assert summaries[1]["success_rate"] == 0.0
    assert summaries[1]["mean_first_success_step"] is None
    assert summaries[1]["mean_recorded_step"] == 6.0


def test_task_metadata_missing_values_are_null(tmp_path):
    profiler = EmbodiedRolloutProfiler(_cfg(tmp_path), rank=0)
    profiler.start_rollout_epoch(
        stage_id=0,
        rollout_epoch=0,
        global_step=0,
        env=object(),
        obs={},
        batch_size=1,
    )
    profiler.finish_rollout_epoch(
        stage_id=0,
        rollout_epoch=0,
        global_step=0,
        env=object(),
        obs={},
    )
    records = _read_jsonl(
        tmp_path / "exp" / "profiling" / "embodied_rollout_events_rank_0.jsonl"
    )
    assert records[0]["task_id"] is None
    assert records[0]["task_name"] is None
    assert records[0]["task_description"] is None
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/unit_tests/test_embodied_rollout_profiler.py -q`

预期：FAIL，报错包含 `AttributeError: 'EmbodiedRolloutProfiler' object has no attribute 'start_rollout_epoch'`。

- [ ] **步骤 3：实现 epoch 状态、metadata 和 summary**

在 `rlinf/utils/embodied_rollout_profiler.py` 中加入：

```python
from collections import defaultdict
from dataclasses import field


def _safe_sequence_get(value: Any, index: int) -> Any:
    value = _as_python(value)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    if isinstance(value, dict):
        return None
    try:
        return value[index]
    except (TypeError, IndexError, KeyError):
        return value


@dataclass
class _TaskProgress:
    metadata: dict[str, Any]
    success: bool = False
    first_success_step: int | None = None
```

修改 `EmbodiedRolloutProfiler` dataclass：

```python
    _epoch_state: dict[tuple[int, int], list[_TaskProgress]] = field(
        default_factory=dict, init=False
    )
```

添加方法：

```python
    def _global_env_id(self, stage_id: int, local_env_index: int) -> int:
        total_num_envs = int(self.cfg.env.train.total_num_envs)
        stage_num = int(self.cfg.rollout.get("pipeline_stage_num", 1))
        num_envs_per_stage = total_num_envs // max(stage_num, 1)
        return int(self.rank * num_envs_per_stage + stage_id * (num_envs_per_stage // max(stage_num, 1)) + local_env_index)

    def _extract_metadata(
        self,
        env: Any,
        obs: dict[str, Any] | None,
        local_env_index: int,
        stage_id: int,
    ) -> dict[str, Any]:
        obs = obs or {}
        task_ids = getattr(env, "task_ids", None)
        task_names = getattr(env, "task_names", None)
        task_descriptions = getattr(env, "task_descriptions", None)
        if task_descriptions is None:
            task_descriptions = obs.get("task_descriptions")
        current_task = getattr(env, "current_task", None)
        env_names = getattr(env, "env_names", getattr(env, "env_names_all", None))

        return {
            "local_env_index": int(local_env_index),
            "global_env_id": self._global_env_id(stage_id, local_env_index),
            "env_type": self.env_type,
            "task_id": _safe_sequence_get(task_ids, local_env_index),
            "task_name": _safe_sequence_get(task_names, local_env_index)
            or _safe_sequence_get(env_names, local_env_index),
            "task_description": _safe_sequence_get(task_descriptions, local_env_index),
            "task_type": _safe_sequence_get(current_task, local_env_index),
        }

    def start_rollout_epoch(
        self,
        *,
        stage_id: int,
        rollout_epoch: int,
        global_step: int,
        env: Any,
        obs: dict[str, Any] | None,
        batch_size: int,
    ) -> None:
        if not self.enabled:
            return
        key = (int(stage_id), int(rollout_epoch))
        self._epoch_state[key] = [
            _TaskProgress(
                metadata=self._extract_metadata(env, obs, env_index, stage_id)
            )
            for env_index in range(int(batch_size))
        ]

    def update_task_success(
        self,
        *,
        stage_id: int,
        rollout_epoch: int,
        chunk_start_step: int,
        terminations: torch.Tensor | None,
    ) -> None:
        if not self.enabled or terminations is None:
            return
        key = (int(stage_id), int(rollout_epoch))
        progresses = self._epoch_state.get(key)
        if not progresses:
            return
        terminations_bool = terminations.detach().cpu().to(torch.bool)
        for env_index, progress in enumerate(progresses):
            if progress.success:
                continue
            hit_steps = torch.nonzero(terminations_bool[env_index], as_tuple=False)
            if hit_steps.numel() == 0:
                continue
            first_offset = int(hit_steps[0].item())
            progress.success = True
            progress.first_success_step = int(chunk_start_step + first_offset + 1)

    def finish_rollout_epoch(
        self,
        *,
        stage_id: int,
        rollout_epoch: int,
        global_step: int,
        env: Any,
        obs: dict[str, Any] | None,
    ) -> None:
        if not self.enabled:
            return
        key = (int(stage_id), int(rollout_epoch))
        progresses = self._epoch_state.pop(key, [])
        if not progresses:
            return
        max_steps = int(self.cfg.env.train.max_steps_per_rollout_epoch)
        task_records = []
        for progress in progresses:
            metadata = progress.metadata
            recorded_step = (
                progress.first_success_step if progress.success else max_steps
            )
            record = {
                "event": "task_episode_steps",
                "global_step": int(global_step),
                "rollout_epoch": int(rollout_epoch),
                "rank": int(self.rank),
                "stage_id": int(stage_id),
                **metadata,
                "success": bool(progress.success),
                "first_success_step": progress.first_success_step,
                "max_steps": max_steps,
                "recorded_step": int(recorded_step),
            }
            task_records.append(record)
            self._write_jsonl(self.events_path, record)
        self._write_task_summary(global_step, rollout_epoch, task_records)

    def _write_task_summary(
        self,
        global_step: int,
        rollout_epoch: int,
        task_records: list[dict[str, Any]],
    ) -> None:
        grouped: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
        for record in task_records:
            grouped[
                (record["task_id"], record["task_name"], record["task_type"])
            ].append(record)
        for (task_id, task_name, task_type), records in grouped.items():
            recorded_steps = torch.tensor(
                [record["recorded_step"] for record in records], dtype=torch.float32
            )
            success_steps = [
                record["first_success_step"]
                for record in records
                if record["first_success_step"] is not None
            ]
            summary = {
                "global_step": int(global_step),
                "rollout_epoch": int(rollout_epoch),
                "rank": int(self.rank),
                "env_type": self.env_type,
                "task_id": task_id,
                "task_name": task_name,
                "task_type": task_type,
                "count": len(records),
                "success_count": int(sum(bool(record["success"]) for record in records)),
                "success_rate": float(
                    sum(bool(record["success"]) for record in records) / len(records)
                ),
                "mean_recorded_step": float(recorded_steps.mean().item()),
                "mean_first_success_step": float(torch.tensor(success_steps, dtype=torch.float32).mean().item())
                if success_steps
                else None,
                "p50_recorded_step": float(torch.quantile(recorded_steps, 0.5).item()),
                "p95_recorded_step": float(torch.quantile(recorded_steps, 0.95).item()),
            }
            self._write_jsonl(self.summary_path, summary)
```

- [ ] **步骤 4：修正 global env id 公式**

将 `_global_env_id()` 改成需要 `EnvWorker` 传入的 `num_envs_per_stage` 更准确：

```python
    def set_layout(self, *, num_envs_per_stage: int, stage_num: int) -> None:
        self.num_envs_per_stage = int(num_envs_per_stage)
        self.stage_num = int(stage_num)

    def _global_env_id(self, stage_id: int, local_env_index: int) -> int:
        num_envs_per_worker = int(
            getattr(self, "num_envs_per_stage", 0)
        ) * int(getattr(self, "stage_num", 1))
        return int(
            self.rank * num_envs_per_worker
            + stage_id * int(getattr(self, "num_envs_per_stage", 0))
            + local_env_index
        )
```

在 `__post_init__()` 末尾加默认：

```python
        self.num_envs_per_stage = 0
        self.stage_num = 1
```

- [ ] **步骤 5：运行测试验证通过**

运行：`pytest tests/unit_tests/test_embodied_rollout_profiler.py -q`

预期：4 passed。

- [ ] **步骤 6：Commit**

运行：

```bash
git add rlinf/utils/embodied_rollout_profiler.py tests/unit_tests/test_embodied_rollout_profiler.py
git commit -s -m "feat: record embodied task completion steps"
```

## 任务 3：EnvWorker 集成同步与 async 共享路径

**文件：**
- 修改：`rlinf/workers/env/env_worker.py`
- 创建：`tests/unit_tests/test_env_worker_rollout_profiler.py`

- [ ] **步骤 1：编写失败测试**

创建 `tests/unit_tests/test_env_worker_rollout_profiler.py`：

```python
import json
import time

import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.workers.env.env_worker import EnvWorker


class MinimalEnvWorker(EnvWorker):
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_video_cnt = 0
        self.eval_video_cnt = 0
        self.should_stop = False
        self.env_list = []
        self.eval_env_list = []
        self.last_obs_list = []
        self.last_intervened_info_list = []
        self.rollout_epoch = cfg.algorithm.get("rollout_epoch", 1)
        self.collect_transitions = cfg.rollout.get("collect_transitions", False)
        self.collect_prev_infos = cfg.rollout.get("collect_prev_infos", True)
        self.stage_num = cfg.rollout.pipeline_stage_num
        self.train_num_envs_per_stage = cfg.env.train.total_num_envs
        self._rank = 0
        self._world_size = 1
        self._init_rollout_profiler()


class FakeStageEnv:
    task_ids = [11, 12]
    task_names = ["A", "B"]
    task_descriptions = ["task A", "task B"]

    def chunk_step(self, actions):
        time.sleep(0.001)
        obs = {
            "states": torch.zeros((2, 1)),
            "task_descriptions": self.task_descriptions,
        }
        rewards = torch.zeros((2, 2))
        terminations = torch.tensor([[False, True], [False, False]])
        truncations = torch.tensor([[False, False], [False, True]])
        infos = {"final_observation": obs}
        return [obs], rewards, terminations, truncations, [infos]


def _cfg(tmp_path):
    return OmegaConf.create(
        {
            "algorithm": {"rollout_epoch": 1},
            "actor": {"model": {"num_action_chunks": 2, "action_dim": 1}},
            "runner": {
                "logger": {
                    "log_path": str(tmp_path),
                    "experiment_name": "exp",
                }
            },
            "rollout": {
                "pipeline_stage_num": 1,
                "collect_transitions": False,
                "collect_prev_infos": True,
                "profiling": {"enabled": True},
            },
            "env": {
                "train": {
                    "env_type": "fake_env",
                    "total_num_envs": 2,
                    "max_steps_per_rollout_epoch": 4,
                    "max_episode_steps": 4,
                    "auto_reset": True,
                    "ignore_terminations": True,
                    "video_cfg": {"save_video": False},
                },
                "eval": {
                    "env_type": "fake_env",
                    "total_num_envs": 2,
                    "max_steps_per_rollout_epoch": 4,
                    "max_episode_steps": 4,
                    "auto_reset": True,
                    "video_cfg": {"save_video": False},
                },
            },
        }
    )


def test_env_worker_profiler_records_chunk_and_epoch(tmp_path):
    worker = MinimalEnvWorker(_cfg(tmp_path))
    worker.env_list = [FakeStageEnv()]
    worker._profiler_start_epoch(
        stage_id=0,
        rollout_epoch=0,
        global_step=9,
        env_output=EnvOutput(
            obs={
                "states": torch.zeros((2, 1)),
                "task_descriptions": ["task A", "task B"],
            }
        ),
    )
    env_output, _ = worker.env_interact_step(torch.zeros((2, 2, 1)), stage_id=0)
    worker._profiler_record_chunk(
        stage_id=0,
        rollout_epoch=0,
        chunk_index=0,
        global_step=9,
        env_output=env_output,
        wall_time_s=0.1,
    )
    worker._profiler_finish_epoch(
        stage_id=0,
        rollout_epoch=0,
        global_step=9,
        env_output=env_output,
    )

    records_path = (
        tmp_path / "exp" / "profiling" / "embodied_rollout_events_rank_0.jsonl"
    )
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "env_chunk_timing",
        "task_episode_steps",
        "task_episode_steps",
    ]
    assert records[0]["success_count"] == 1
    assert records[1]["first_success_step"] == 2
    assert records[1]["recorded_step"] == 2
    assert records[2]["success"] is False
    assert records[2]["recorded_step"] == 4
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/unit_tests/test_env_worker_rollout_profiler.py -q`

预期：FAIL，报错包含 `AttributeError: 'MinimalEnvWorker' object has no attribute '_init_rollout_profiler'`。

- [ ] **步骤 3：在 EnvWorker 初始化 profiler**

在 `rlinf/workers/env/env_worker.py` import 区加入：

```python
import time

from rlinf.utils.embodied_rollout_profiler import EmbodiedRolloutProfiler
```

如果文件已 import `time`，不要重复。

在 `EnvWorker.__init__()` 的配置字段初始化后添加：

```python
        self._init_rollout_profiler()
```

添加方法：

```python
    def _init_rollout_profiler(self) -> None:
        self.rollout_profiler = EmbodiedRolloutProfiler(self.cfg, rank=self._rank)
        num_envs_per_stage = getattr(self, "train_num_envs_per_stage", 0)
        stage_num = getattr(self, "stage_num", 1)
        self.rollout_profiler.set_layout(
            num_envs_per_stage=num_envs_per_stage,
            stage_num=stage_num,
        )
```

- [ ] **步骤 4：添加 EnvWorker profiler helper 方法**

在 `EnvWorker` 中添加：

```python
    def _profiler_start_epoch(
        self,
        *,
        stage_id: int,
        rollout_epoch: int,
        global_step: int,
        env_output: EnvOutput,
    ) -> None:
        env = self.env_list[stage_id]
        self.rollout_profiler.start_rollout_epoch(
            stage_id=stage_id,
            rollout_epoch=rollout_epoch,
            global_step=global_step,
            env=env,
            obs=env_output.obs,
            batch_size=self.train_num_envs_per_stage,
        )

    def _profiler_record_chunk(
        self,
        *,
        stage_id: int,
        rollout_epoch: int,
        chunk_index: int,
        global_step: int,
        env_output: EnvOutput,
        wall_time_s: float,
    ) -> None:
        subenv_step_timings = None
        env = self.env_list[stage_id]
        if hasattr(env, "pop_subenv_step_timings"):
            subenv_step_timings = env.pop_subenv_step_timings()
        chunk_start_step = chunk_index * self.cfg.actor.model.num_action_chunks
        self.rollout_profiler.record_env_chunk_timing(
            stage_id=stage_id,
            rollout_epoch=rollout_epoch,
            chunk_index=chunk_index,
            global_step=global_step,
            wall_time_s=wall_time_s,
            dones=env_output.dones,
            terminations=env_output.terminations,
            truncations=env_output.truncations,
            env=env,
            obs=env_output.obs,
            subenv_step_timings=subenv_step_timings,
        )
        self.rollout_profiler.update_task_success(
            stage_id=stage_id,
            rollout_epoch=rollout_epoch,
            chunk_start_step=chunk_start_step,
            terminations=env_output.terminations,
        )

    def _profiler_finish_epoch(
        self,
        *,
        stage_id: int,
        rollout_epoch: int,
        global_step: int,
        env_output: EnvOutput,
    ) -> None:
        env = self.env_list[stage_id]
        self.rollout_profiler.finish_rollout_epoch(
            stage_id=stage_id,
            rollout_epoch=rollout_epoch,
            global_step=global_step,
            env=env,
            obs=env_output.obs,
        )
```

- [ ] **步骤 5：把 helper 接入 `_run_interact_once()`**

在 `for epoch in range(self.rollout_epoch):` 内，`env_outputs = self.bootstrap_step()` 后，每个 stage 发送 bootstrap obs 前调用：

```python
                self._profiler_start_epoch(
                    stage_id=stage_id,
                    rollout_epoch=epoch,
                    global_step=getattr(self, "global_step", 0),
                    env_output=env_output,
                )
```

在调用 `self.env_interact_step()` 的地方改成显式计时：

```python
                    step_start_time = time.perf_counter()
                    env_output, env_info = self.env_interact_step(
                        rollout_result.actions, stage_id
                    )
                    step_wall_time_s = time.perf_counter() - step_start_time
                    self._profiler_record_chunk(
                        stage_id=stage_id,
                        rollout_epoch=epoch,
                        chunk_index=chunk_index,
                        global_step=getattr(self, "global_step", 0),
                        env_output=env_output,
                        wall_time_s=step_wall_time_s,
                    )
```

把当前循环 `for _ in range(self.n_train_chunk_steps):` 改为：

```python
            for chunk_index in range(self.n_train_chunk_steps):
```

在 `self.store_last_obs_and_intervened_info(env_outputs)` 前，为每个 stage 调用：

```python
                self._profiler_finish_epoch(
                    stage_id=stage_id,
                    rollout_epoch=epoch,
                    global_step=getattr(self, "global_step", 0),
                    env_output=env_output,
                )
```

- [ ] **步骤 6：确保 global_step 存在**

在 `EnvWorker.__init__()` 添加：

```python
        self.global_step = 0
```

在 `EnvWorker` 中添加或更新 `set_global_step` 方法。如果已有同名方法，合并逻辑：

```python
    def set_global_step(self, global_step: int):
        self.global_step = int(global_step)
```

- [ ] **步骤 7：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_env_worker_rollout_profiler.py tests/unit_tests/test_embodied_rollout_profiler.py -q
```

预期：全部 PASS。

- [ ] **步骤 8：Commit**

运行：

```bash
git add rlinf/workers/env/env_worker.py tests/unit_tests/test_env_worker_rollout_profiler.py
git commit -s -m "feat: integrate embodied profiler with env worker"
```

## 任务 4：LIBERO subprocess per-subenv timing

**文件：**
- 修改：`rlinf/envs/libero/venv.py`
- 修改：`rlinf/envs/libero/libero_env.py`
- 修改：`tests/unit_tests/test_embodied_rollout_profiler.py`

- [ ] **步骤 1：编写失败测试**

在 `tests/unit_tests/test_embodied_rollout_profiler.py` 追加：

```python
def test_subenv_step_timing_records_are_written(tmp_path):
    profiler = EmbodiedRolloutProfiler(_cfg(tmp_path), rank=0)
    profiler.set_layout(num_envs_per_stage=2, stage_num=1)
    profiler.record_env_chunk_timing(
        stage_id=0,
        rollout_epoch=0,
        chunk_index=1,
        global_step=3,
        wall_time_s=0.5,
        dones=torch.zeros((2, 2), dtype=torch.bool),
        terminations=torch.zeros((2, 2), dtype=torch.bool),
        truncations=torch.zeros((2, 2), dtype=torch.bool),
        env=FakeEnv(),
        obs={"task_descriptions": ["pick object", "place object"]},
        subenv_step_timings=[
            {"local_env_index": 0, "step_index": 2, "wall_time_s": 0.11},
            {"local_env_index": 1, "step_index": 2, "wall_time_s": 0.22},
        ],
    )
    records = _read_jsonl(
        tmp_path / "exp" / "profiling" / "embodied_rollout_events_rank_0.jsonl"
    )
    assert records[0]["timing_granularity"] == "subenv"
    assert records[1]["event"] == "subenv_step_timing"
    assert records[1]["local_env_index"] == 0
    assert records[1]["global_env_id"] == 0
    assert records[1]["task_name"] == "Pick"
    assert records[2]["wall_time_s"] == 0.22
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/unit_tests/test_embodied_rollout_profiler.py::test_subenv_step_timing_records_are_written -q`

预期：FAIL，因为 `record_env_chunk_timing()` 尚未写 `subenv_step_timing` 明细。

- [ ] **步骤 3：profiler 写 subenv records**

在 `record_env_chunk_timing()` 写完 `env_chunk_timing` 后添加：

```python
        for timing in subenv_step_timings or []:
            local_env_index = int(timing["local_env_index"])
            metadata = self._extract_metadata(env, obs, local_env_index, stage_id)
            subenv_record = {
                "event": "subenv_step_timing",
                "global_step": int(global_step),
                "rollout_epoch": int(rollout_epoch),
                "chunk_index": int(chunk_index),
                "rank": int(self.rank),
                "stage_id": int(stage_id),
                "step_index": int(timing["step_index"]),
                "wall_time_s": float(timing["wall_time_s"]),
                **metadata,
            }
            self._write_jsonl(self.events_path, subenv_record)
```

- [ ] **步骤 4：LIBERO venv 缓存每个 worker step 时间**

在 `rlinf/envs/libero/venv.py` import 区添加：

```python
import time
```

在 `_worker()` 的 `cmd == "step"` 分支改为：

```python
            if cmd == "step":
                step_start = time.perf_counter()
                env_return = env.step(data)
                step_wall_time_s = time.perf_counter() - step_start
                if obs_bufs is not None:
                    _encode_obs(env_return[0], obs_bufs)
                    env_return = (None, *env_return[1:])
                p.send((env_return, step_wall_time_s))
```

在 `ReconfigureSubprocEnvWorker` 中添加 last timing：

```python
        self.last_step_wall_time_s = None
```

并覆盖 step 方法。如果父类 step 方法名不同，先读取 `rlinf/envs/venv/venv.py` 中 `SubprocEnvWorker`，按现有签名实现同名覆盖：

```python
    def send_action(self, action: Any) -> None:
        self.parent_remote.send(["step", action])

    def get_result(self) -> Any:
        env_return, step_wall_time_s = self.parent_remote.recv()
        self.last_step_wall_time_s = float(step_wall_time_s)
        return env_return
```

如果父类使用 `send()`/`recv()` 而不是 `send_action()`/`get_result()`，按父类实际方法名覆盖，保持对 `SubprocVectorEnv.step()` 透明。

- [ ] **步骤 5：LIBERO env 暴露并清空 timing**

在 `rlinf/envs/libero/libero_env.py` 的 `LiberoEnv.__init__()` 添加：

```python
        self._subenv_step_timings = []
```

在 `LiberoEnv.step()` 的 `self.env.step(actions)` 后添加：

```python
        if hasattr(self.env, "workers"):
            self._subenv_step_timings.extend(
                {
                    "local_env_index": env_index,
                    "step_index": int(self._elapsed_steps[env_index]),
                    "wall_time_s": float(worker.last_step_wall_time_s),
                }
                for env_index, worker in enumerate(self.env.workers)
                if getattr(worker, "last_step_wall_time_s", None) is not None
            )
```

在 `LiberoEnv` 添加：

```python
    def pop_subenv_step_timings(self):
        timings = self._subenv_step_timings
        self._subenv_step_timings = []
        return timings
```

- [ ] **步骤 6：运行相关测试**

运行：

```bash
pytest tests/unit_tests/test_embodied_rollout_profiler.py::test_subenv_step_timing_records_are_written -q
pytest tests/unit_tests/test_import_rlinf_package.py -q
```

预期：指定 profiler 测试 PASS；import 测试 PASS 或因环境缺少可选仿真依赖而按现有测试策略跳过/报告。

- [ ] **步骤 7：Commit**

运行：

```bash
git add rlinf/utils/embodied_rollout_profiler.py rlinf/envs/libero/venv.py rlinf/envs/libero/libero_env.py tests/unit_tests/test_embodied_rollout_profiler.py
git commit -s -m "feat: record libero subenv step timing"
```

## 任务 5：配置兼容、文档和默认关闭验证

**文件：**
- 修改：`docs/superpowers/specs/2026-05-10-embodied-rollout-profiling-design.md`
- 修改：`tests/unit_tests/test_env_worker_rollout_profiler.py`

- [ ] **步骤 1：编写默认关闭集成测试**

在 `tests/unit_tests/test_env_worker_rollout_profiler.py` 追加：

```python
def test_env_worker_profiler_default_disabled(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.rollout.profiling.enabled = False
    worker = MinimalEnvWorker(cfg)
    worker.env_list = [FakeStageEnv()]
    output = EnvOutput(
        obs={"states": torch.zeros((2, 1)), "task_descriptions": ["A", "B"]},
        dones=torch.zeros((2, 2), dtype=torch.bool),
        terminations=torch.zeros((2, 2), dtype=torch.bool),
        truncations=torch.zeros((2, 2), dtype=torch.bool),
    )
    worker._profiler_start_epoch(
        stage_id=0,
        rollout_epoch=0,
        global_step=0,
        env_output=output,
    )
    worker._profiler_record_chunk(
        stage_id=0,
        rollout_epoch=0,
        chunk_index=0,
        global_step=0,
        env_output=output,
        wall_time_s=0.01,
    )
    worker._profiler_finish_epoch(
        stage_id=0,
        rollout_epoch=0,
        global_step=0,
        env_output=output,
    )
    assert not (tmp_path / "exp" / "profiling").exists()
```

- [ ] **步骤 2：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_env_worker_rollout_profiler.py tests/unit_tests/test_embodied_rollout_profiler.py -q
```

预期：全部 PASS。

- [ ] **步骤 3：更新规格中的实现状态说明**

在 `docs/superpowers/specs/2026-05-10-embodied-rollout-profiling-design.md` 的 `Configuration` 后添加：

```markdown
Implementation note: the initial implementation writes per-rank JSONL files from env workers. Global cross-rank aggregation is intentionally left to offline post-processing because Ray actors should not contend on one output file.
```

- [ ] **步骤 4：运行格式和目标测试**

运行：

```bash
ruff format rlinf/utils/embodied_rollout_profiler.py tests/unit_tests/test_embodied_rollout_profiler.py tests/unit_tests/test_env_worker_rollout_profiler.py
ruff check rlinf/utils/embodied_rollout_profiler.py rlinf/workers/env/env_worker.py rlinf/envs/libero/venv.py rlinf/envs/libero/libero_env.py tests/unit_tests/test_embodied_rollout_profiler.py tests/unit_tests/test_env_worker_rollout_profiler.py
pytest tests/unit_tests/test_embodied_rollout_profiler.py tests/unit_tests/test_env_worker_rollout_profiler.py tests/unit_tests/test_comm_mapper.py tests/unit_tests/test_embodied_io_struct_feature_cache.py -q
```

预期：format 成功；ruff check PASS；pytest 全部 PASS。

- [ ] **步骤 5：Commit**

运行：

```bash
git add docs/superpowers/specs/2026-05-10-embodied-rollout-profiling-design.md tests/unit_tests/test_env_worker_rollout_profiler.py
git commit -s -m "test: verify embodied profiling default off"
```

## 任务 6：最终验证与交付记录

**文件：**
- 不新增文件，检查当前分支状态。

- [ ] **步骤 1：运行最终验证命令**

运行：

```bash
ruff check rlinf/utils/embodied_rollout_profiler.py rlinf/workers/env/env_worker.py rlinf/envs/libero/venv.py rlinf/envs/libero/libero_env.py tests/unit_tests/test_embodied_rollout_profiler.py tests/unit_tests/test_env_worker_rollout_profiler.py
pytest tests/unit_tests/test_embodied_rollout_profiler.py tests/unit_tests/test_env_worker_rollout_profiler.py tests/unit_tests/test_comm_mapper.py tests/unit_tests/test_embodied_io_struct_feature_cache.py -q
```

预期：全部 PASS。

- [ ] **步骤 2：检查 git 状态**

运行：`git status --short`

预期：只允许存在用户已有的 `requirements/install.sh` 未提交改动；实现相关文件应全部提交。

- [ ] **步骤 3：给用户交付摘要**

交付内容包含：

```text
Implemented:
- Default-off embodied rollout profiling under rollout.profiling.enabled.
- Per-rank JSONL env chunk timing and task completion records.
- Per-rank task step summaries.
- Optional LIBERO subprocess subenv timing.

Validation:
- ruff check ...
- pytest ...

Notes:
- Existing requirements/install.sh was left untouched.
- Vectorized envs such as ManiSkill report batch timing only.
```
