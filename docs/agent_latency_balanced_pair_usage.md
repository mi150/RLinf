# Agent 改动说明：延迟均衡 Env Chunk Step

本文总结本次 env rollout 相关改动，包括新增了什么能力、涉及哪些模块、如何配置运行，以及如何看运行结果。

## 一句话总结

本次新增了 `latency_balanced_pair` chunk step 模式，用于 LIBERO 和 RoboCasa 这类子进程 env。它会根据每个 env 最近的 chunk 执行耗时，把慢 env 和快 env 动态配到同一个 CPU slot，减少一次 rollout 中最慢 env 拖尾造成的等待。

## 为什么改

原来的 env chunk step 主要有两类执行方式：

- `sync_time_major`：所有 env 同步执行第 1 步，再同步执行第 2 步，直到 chunk 结束。
- `parallel_shard`：把 env 分成多个 shard，并行跑 shard 内部的 chunk。

在 LIBERO/RoboCasa 这类场景里，不同 env 的仿真耗时会有明显差异。一次 rollout 需要等所有 env 的 chunk 都完成，慢 env 会拉长整体等待时间。本次改动的目标是：

- 让每个 env 子进程连续执行自己的完整 action chunk。
- 统计每个 env 的 chunk 级耗时。
- 用 EMA 维护 env 的预测耗时。
- 每轮 chunk 前重新分组，让慢 env 和快 env 尽量配在一起。
- 在开启 CPU resource pool 时，把同一组 env 动态绑定到同一个 CPU core/core group 上。

## 新增能力

### 1. 新增 chunk step 模式

新增模式名：

```yaml
env:
  train:
    chunk_step_mode: latency_balanced_pair
```

这个模式已经注册到 `CHUNK_STEP_MODES`，配置后可以被校验和分发。

### 2. 延迟均衡配对

核心逻辑在 `BaseVectorEnv.latency_balanced_pair_chunk_step()`。

执行流程：

1. 输入 `chunk_actions`，形状为 `[num_envs, chunk_step, action_dim]`。
2. 按预测延迟把 env 分到多个 CPU slot。
3. 每个 slot 先 dispatch 一个 env 子进程，子进程收到完整 action chunk。
4. 某个 slot 的 env 完成后，立即 dispatch 同 slot 的下一个 env，不等待其他 slot。
5. 父进程记录每个 env 完成 chunk 的耗时，并使用 EMA 更新预测延迟。
6. 下一轮 chunk 前，根据预测延迟重新构建分组。
7. 返回结果时按原始 env 顺序恢复，外部 API 不变。

配对策略是一个简单的 greedy LPT：

1. 按预测耗时从慢到快排序 env。
2. 依次把 env 放进当前累计耗时最低、且还没满的 slot。
3. 每个 slot 最多放 `envs_per_core` 个 env。

当 `envs_per_core: 2` 时，效果通常是慢 env 配快 env。

### 3. 动态 CPU affinity

如果启用了 CPU resource pool，并且 env CPU 粒度是 `per_env`，每个 env 子进程会有自己的 CPU core group。

在 `latency_balanced_pair` 模式下，调度器会按 slot 来分配 CPU core group：

- 一个 stage 中有多少个 slot，就需要多少个 CPU group。
- 同一 slot 内的多个 env 会绑定到同一个 CPU group。
- 下一轮重新配对后，env 子进程的 affinity 可以动态更新。

这样可以表达“一个 CPU slot 上以 slot-local pipeline 依次跑多个 env”的调度方式。

### 4. EnvWorker 时间戳日志

新增了 env chunk step 的开始/结束事件记录。

开启方式：

```yaml
env:
  train:
    log_sim_timestamps: true
    log_sim_affinity_interval: 1
```

输出位置：

```text
<runner.logger.log_path>/env_sim_timestamps/env_rank_<rank>.jsonl
```

每条记录包含：

- `event`: `start` 或 `end`
- `rank`: EnvWorker rank
- `epoch`
- `chunk_step`
- `stage`
- `local_envs`
- `wall_ns`
- `duration_s`，只在 `end` 事件中出现
- 当前 EnvWorker 进程 affinity
- 可选的子 env 进程 affinity 采样

`duration_s` 可以用来分析每个 EnvWorker 的真实仿真耗时。

现在同一文件还会记录子 env 粒度事件：

- `event`: `subenv_start` 或 `subenv_end`
- `local_env`: 当前 rank/stage 内的子 env 下标
- `global_env`: 按 `rank/stage/local_env` 展开的全局 env 下标
- `operation`: `step`、`chunk_step` 或 `latency_balanced_pair_chunk_step`
- `vector_step`: 当前 chunk 内的 vector step 序号
- `action_chunk_steps`: 子 env 连续执行的 action chunk 长度
- `pair_slot` / `pair_offset`，只在 `latency_balanced_pair` 中出现
- `child_pid` 和 `cpu_affinity`
- `duration_s`，只在 `subenv_end` 事件中出现

子 env 事件是 EnvWorker 父进程观测到的 dispatch 到 worker ready/recv
耗时；同步 step/chunk step 会按完成顺序记录 `subenv_end`，再按原 env
顺序恢复返回值，因此可用于分析 rank 内各 local env 的拖尾、配对和等待关系。

## 主要代码改动

### `rlinf/envs/chunk_runner.py`

新增 `latency_balanced_pair` 到合法 chunk step mode 集合。

### `rlinf/config.py`

新增 `latency_balanced_pair` 参数校验：

- `envs_per_core >= 1`
- `envs_per_core <= local_num_envs`
- `local_num_envs % envs_per_core == 0`
- `ema_alpha` 必须在 `(0, 1]`
- `initial_latency_ms` 如果设置，必须大于 0

校验后会把配置规范化为：

```yaml
latency_balanced_pair:
  envs_per_core: 2
  ema_alpha: 0.3
  initial_latency_ms: null
  dynamic_affinity: true
```

### `rlinf/envs/venv/venv.py`

新增底层调度实现：

- `latency_balanced_pair_chunk_step()`
- `_build_latency_balanced_groups()`
- `_get_slot_cpu_core_groups()`

同时扩展了 env worker 协议：

- `set_cpu_affinity(cpus)`
- `get_cpu_affinity()`

子进程 worker 支持接收：

- `set_cpu_affinity`
- `get_cpu_affinity`

### `rlinf/envs/libero/libero_env.py`

LIBERO 的 `chunk_step()` 新增分支：

```python
if self.chunk_step_mode == "latency_balanced_pair":
    return self._chunk_step_latency_balanced_pair(chunk_actions)
```

新增 `_chunk_step_latency_balanced_pair()`，负责：

- 调用底层 vector env 的 `latency_balanced_pair_chunk_step()`
- 保持原来的 reward / done / truncation / auto reset 语义
- 返回原来的 RLinf chunk step API 形状

### `rlinf/envs/robocasa/robocasa_env.py`

RoboCasa 接入方式和 LIBERO 一致，也新增：

- `chunk_step()` 分支
- `_chunk_step_latency_balanced_pair()`

### `rlinf/envs/libero/venv.py`

LIBERO 的定制 subproc worker 增加 CPU affinity 指令支持。

### `rlinf/envs/robocasa/venv.py`

RoboCasa 的定制 subproc worker 增加 CPU affinity 指令支持。

### `rlinf/scheduler/resource_pool/solver.py`

resource pool solver 新增对 `latency_balanced_pair` 的 CPU slot 分配逻辑。

普通 `per_env` 模式下，CPU core group 按 env 数均分。

`latency_balanced_pair` 模式下，CPU core group 按 slot 数均分：

```text
per_stage_env_count = local_env_count / pipeline_stage_num
slot_count = per_stage_env_count / envs_per_core
```

每个 stage 复用同一组 slot CPU groups。

同时，`sm_percent: 0` 的 GPU binding 现在会保留 visible GPU 信息，方便 env 渲染进程仍然能看到指定 GPU。

### `rlinf/workers/env/env_worker.py`

新增：

- EnvWorker 初始化前后打印 CPU binding 状态。
- 初始化后校验 EnvWorker 进程和子 env 进程的 CPU affinity。
- `env_interact_step()` 支持写 `env_sim_timestamps/*.jsonl`。
- 在 chunk step 调用中透传 `epoch` 和 `chunk_step_idx`，用于日志定位。

### 测试

新增或更新了这些测试覆盖：

- `latency_balanced_pair` 模式是否注册。
- 底层 vector env 是否能按 step-major 形状恢复结果。
- 非法 `envs_per_core` 是否报错。
- 慢快 env 是否能被配到预期 slot。
- resource pool 在 pairing 模式下是否按 slot 分配 CPU groups。
- pipeline stage 下 slot CPU groups 是否正确复用。
- `sm_percent: 0` 时 GPU visible device 信息是否保留。

## 推荐配置

推荐先用：

```text
examples/embodiment/config/0libero_pairing_test.yaml
```

关键配置：

```yaml
cluster:
  num_nodes: 1
  component_placement:
    actor: 0-7
    rollout: 0-7
    env: 0-7
  resource_pool:
    enabled: true
    allocation_mode: default
    cpu:
      enabled: true
      pools:
        env_cpu:
          node_group: cluster
          cores: ${oc.env:RLINF_ENV_CPU_CORES,0-55}
      components:
        env:
          pool: env_cpu
          granularity: per_env
    gpu:
      enabled: true
      mode: mps
      pools:
        env_render_gpu:
          node_group: cluster
          devices: ${oc.env:RLINF_ENV_RENDER_GPUS,0-7}
      components:
        env:
          pool: env_render_gpu
          sm_percent: 0

env:
  train:
    total_num_envs: 224
    chunk_step_mode: latency_balanced_pair
    log_sim_timestamps: true
    log_sim_affinity_interval: 1
    latency_balanced_pair:
      envs_per_core: 2
      ema_alpha: 0.3
      initial_latency_ms: null
      dynamic_affinity: true

rollout:
  pipeline_stage_num: 2
```

这份配置下：

- env worker 数是 8。
- 总 env 数是 224。
- 每个 env worker 本地有 28 个 env。
- pipeline stage 数是 2。
- 每个 stage 有 14 个 env。
- `envs_per_core: 2`，所以每个 stage 有 7 个 CPU slot。

## 执行方式

准备常用环境变量：

```bash
export RLINF_ENV_CPU_CORES=0-55
export RLINF_ENV_RENDER_GPUS=0-7
export LIBERO_TYPE=standard
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

启动 Ray：

```bash
ray start --head
```

执行 pairing 配置：

```bash
bash examples/embodiment/run_embodiment.sh 0libero_pairing_test LIBERO
```

如果模型路径需要覆盖：

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path examples/embodiment/config/ \
  --config-name 0libero_pairing_test \
  actor.model.model_path=/path/to/RLinf-Pi05-LIBERO-SFT \
  rollout.model.model_path=/path/to/RLinf-Pi05-LIBERO-SFT \
  runner.logger.log_path=logs/0libero_pairing_test
```

做 baseline 对比可以执行：

```bash
bash examples/embodiment/run_embodiment.sh 0libero_pipeline_test LIBERO
```

`0libero_pipeline_test.yaml` 使用 `sync_time_major`，适合和
`0libero_pairing_test.yaml` 对比。

## 运行后看什么

### 1. 主日志

在 `run_embodiment.log` 中搜索：

```text
latency_balanced_pair enabled
Env CPU binding before env setup
Env CPU binding after train env setup
```

如果看到 `latency_balanced_pair enabled`，说明 env 已经进入新的 chunk step 路径。

### 2. 时间戳日志

查看：

```text
<runner.logger.log_path>/env_sim_timestamps/env_rank_*.jsonl
```

重点看 `event=end` 的 `duration_s` 字段。这个值表示一次 env chunk step 的耗时，可用于比较 baseline 和 pairing 模式。

如果要分析 rank 内 local env 差异，重点看 `event=subenv_end`：

- `operation=latency_balanced_pair_chunk_step` 表示 pairing 模式下某个子 env 完整 action chunk 的耗时。
- `operation=step` 表示 sync-time-major baseline 中某个子 env 的单步耗时；可按 `chunk_step` + `vector_step` 聚合成一个 action chunk。
- `pair_slot` 相同的子 env 会被安排到同一个 CPU slot；`pair_offset` 表示同 slot 内第几个被 dispatch。当前实现是 slot-local pipeline，同 slot 上一个 env 完成后会立即 dispatch 下一个 env，不再等待其他 slot 的同 offset env。

### 3. resource pool 计划

resource pool 开启时会输出资源绑定计划。重点看 env 组件是否包含：

```text
env_cpu_core_groups
```

这个字段表示每个子 env 进程预期绑定到哪些 CPU cores。

## 常见问题

### 没有进入 pairing 模式

检查：

- `env.train.chunk_step_mode` 是否是 `latency_balanced_pair`。
- LIBERO/RoboCasa 的 env 是否走到了对应的 `chunk_step()` 分支。
- 配置是否通过了 `validate_cfg()`。

### 没有动态绑核

检查：

- `cluster.resource_pool.enabled` 是否为 `true`。
- `cluster.resource_pool.cpu.enabled` 是否为 `true`。
- `cluster.resource_pool.cpu.components.env.granularity` 是否是 `per_env`。
- `latency_balanced_pair.dynamic_affinity` 是否为 `true`。
- `RLINF_ENV_CPU_CORES` 是否覆盖了足够 CPU cores。

### 配置校验失败

常见原因：

- `total_num_envs / env_worker_num` 不能被 `envs_per_core` 整除。
- 开启 pipeline 后，`local_env_count / pipeline_stage_num` 不能被 `envs_per_core` 整除。
- `ema_alpha` 不在 `(0, 1]`。
- `initial_latency_ms` 设置成了非正数。

### 日志开销太大

`log_sim_affinity_interval: 1` 会每个 chunk 都采样子 env 进程 affinity，有额外开销。

正式测速时建议：

```yaml
env:
  train:
    log_sim_timestamps: true
    log_sim_affinity_interval: 0
```

这样仍然保留 `duration_s`，但不频繁采样子进程 affinity。

## 验证命令

运行相关单测：

```bash
pytest -q \
  tests/unit_tests/test_chunk_step_parallel.py \
  tests/unit_tests/test_resource_pool_solver.py
```

只跑新增逻辑相关测试：

```bash
pytest -q \
  tests/unit_tests/test_chunk_step_parallel.py::test_latency_balanced_pair_mode_is_registered \
  tests/unit_tests/test_chunk_step_parallel.py::test_subproc_vector_env_latency_balanced_pair_restores_step_major_shape \
  tests/unit_tests/test_chunk_step_parallel.py::test_latency_balanced_pair_groups_slow_and_fast_envs \
  tests/unit_tests/test_resource_pool_solver.py::test_default_solver_shares_env_cpu_groups_for_latency_balanced_pair \
  tests/unit_tests/test_resource_pool_solver.py::test_default_solver_uses_per_stage_slots_for_latency_balanced_pair
```

## 后续扩展建议

- 如果要接入其他 env，优先复用 `BaseVectorEnv.latency_balanced_pair_chunk_step()`。
- 不要在每个 env 文件里复制调度算法，只在 env wrapper 层做返回值整理。
- 对单进程 vectorized env，无法做到 per-env CPU affinity，只能做 EnvWorker 级别绑核。
- 如果正式做性能对比，建议固定 `total_num_envs`、`pipeline_stage_num`、CPU core 列表和 GPU 列表，只改变 `chunk_step_mode`。
