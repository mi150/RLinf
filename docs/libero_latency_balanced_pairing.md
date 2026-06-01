# LIBERO Latency-Balanced Pairing Chunk Step

> 历史说明：本文档记录早期 `latency_balanced_pair` / `envs_per_core=2`
> 实验设计。当前代码已收敛为三种保留方案：不绑定 CPU 不调度、绑定 CPU 不调度、
> 以及 `envs_per_core=1 + dynamic_affinity=true + core_donation_enabled=true`
> 的 core donation v2。当前使用方式请参考
> `docs/agent_latency_balanced_pair_usage.md`。

本文档说明 `examples/embodiment/config/0libero_pairing_test.yaml` 中使用的
`latency_balanced_pair` chunk step 模式。这个模式已经在当前代码中实现，目标是把
每个 env 的 action chunk 当作调度单元，根据历史 chunk step 耗时把 env 动态拼成
若干 core slot，使同一个 slot 上的 env 尽量形成“慢 + 快”或“中 + 中”的组合。

## 结论

当前实现包含两部分：

- 延迟感知配对调度：已实现。配置
  `env.train.chunk_step_mode: latency_balanced_pair` 后，LIBERO 会走这个调度路径。
- 同 core 绑核：代码已支持，但需要打开 CPU resource pool。当前
  `0libero_pairing_test.yaml` 里 `cluster.resource_pool.cpu.enabled: false`，
  所以这份配置默认只做配对调度，不会真的把配到同一 slot 的 env 子进程绑到同一个
  CPU core/core group。

如果要实现“一个核心上依次跑两个 env”，需要同时满足：

```yaml
env:
  train:
    chunk_step_mode: latency_balanced_pair
    latency_balanced_pair:
      envs_per_core: 2
      dynamic_affinity: true

cluster:
  resource_pool:
    cpu:
      enabled: true
      components:
        env:
          granularity: per_env
```

其中 `envs_per_core: 2` 表示每个 core slot 放 2 个 env。当前测试配置里这个字段被注释，
因此使用代码默认值 `2`。

## 配置入口

测试配置位置：

```text
examples/embodiment/config/0libero_pairing_test.yaml
```

关键字段：

```yaml
cluster:
  component_placement:
    actor: 0-7
    rollout: 0-7
    env: 0-7
  resource_pool:
    enabled: true
    cpu:
      enabled: false
      pools:
        env_cpu:
          node_group: cluster
          cores: ${oc.env:RLINF_ENV_CPU_CORES,0-111}
      components:
        env:
          pool: env_cpu
          granularity: per_env

env:
  train:
    total_num_envs: 224
    chunk_step_mode: latency_balanced_pair
    log_sim_timestamps: true
    log_sim_affinity_interval: 1
    latency_balanced_pair:
      ema_alpha: 0.3
      initial_latency_ms: null
      dynamic_affinity: true

rollout:
  pipeline_stage_num: 2
```

按这份配置计算：

- env worker 数是 8，因为 `cluster.component_placement.env: 0-7`。
- 全局训练 env 数是 224。
- 每个 env worker 本地管理 `224 / 8 = 28` 个 env。
- pipeline stage 数是 2。
- 每个 stage 管理 `28 / 2 = 14` 个 env。
- 默认 `envs_per_core = 2`。
- 每个 stage 需要 `14 / 2 = 7` 个 core slot。

如果启用 CPU resource pool，每个 env worker 会为每个 stage 分配 7 个 slot，
每个 slot 对应一个 CPU core group，slot 内的 2 个 env 会动态绑定到同一个 core group。

## 执行路径

LIBERO 的 `chunk_step()` 会根据配置选择执行模式：

```text
rlinf/envs/libero/libero_env.py
  LiberoEnv.chunk_step()
    if chunk_step_mode == "latency_balanced_pair":
      _chunk_step_latency_balanced_pair()
```

`_chunk_step_latency_balanced_pair()` 会把参数转发到底层 vector env：

```text
rlinf/envs/libero/libero_env.py
  self.env.latency_balanced_pair_chunk_step(
      chunk_actions,
      envs_per_core=...,
      ema_alpha=...,
      initial_latency_ms=...,
      dynamic_affinity=...,
  )
```

真正的配对算法在：

```text
rlinf/envs/venv/venv.py
  BaseVectorEnv.latency_balanced_pair_chunk_step()
  BaseVectorEnv._build_latency_balanced_groups()
```

LIBERO 的子进程 worker 支持 chunk step 和 affinity 指令：

```text
rlinf/envs/libero/venv.py
  _worker()
    cmd == "chunk_step"
    cmd == "set_cpu_affinity"
    cmd == "get_cpu_affinity"
```

## 调度算法

### 1. 以 chunk 为耗时统计单位

输入动作形状是：

```text
chunk_actions: [num_envs, chunk_step, action_dim]
```

每个 env 子进程一次收到自己的完整 action chunk，然后在子进程内部连续执行：

```text
env_i: action[i, 0] -> action[i, 1] -> ... -> action[i, K - 1]
```

父进程统计这个完整 chunk 的 wall time。调度目标不是平衡单个 step，而是平衡每个 env
完成一个 chunk 的耗时，因为 rollout 等待的是 chunk 级结果。

### 2. 维护每个 env 的 EMA 延迟

每个 vector env 内部维护：

```text
_balanced_pair_predicted_latency_s[env_id]
```

初始值：

- 如果配置了 `initial_latency_ms`，使用该值转换成秒。
- 否则默认每个 env 的预测耗时都是 `1.0s`。

每次 env 完成一个 chunk 后更新：

```text
new_latency = ema_alpha * actual_latency
            + (1 - ema_alpha) * old_latency
```

配置项：

```yaml
latency_balanced_pair:
  ema_alpha: 0.3
  initial_latency_ms: null
```

`ema_alpha` 越大，配对越快响应最新耗时；越小，配对越稳定。

### 3. 构建慢快配对组

每次 chunk step 前，调度器会根据当前预测延迟重新构建 groups：

```text
slot_count = local_env_count / envs_per_core
```

核心逻辑：

1. 按预测延迟从慢到快排序 env。
2. 依次把 env 放到当前累计负载最低、且未满的 slot。
3. 每个 slot 最多放 `envs_per_core` 个 env。

当 `envs_per_core = 2` 时，这个 greedy LPT 策略会倾向于：

- 先把慢 env 分散到不同 slot。
- 后续把快 env 填到已有慢 slot 中。
- 如果 env 耗时都接近，则形成“中 + 中”的组合。

示例：

```text
预测耗时:
env0 = 10
env1 = 1
env2 = 8
env3 = 2

envs_per_core = 2
slot_count = 2

排序后: env0, env2, env3, env1

分配:
slot0 <- env0
slot1 <- env2
slot1 <- env3
slot0 <- env1

结果:
slot0 = env0 + env1 = 11
slot1 = env2 + env3 = 10
```

这就是“慢 + 快”拼图。

### 4. slot 内依次执行，slot 间并发执行

调度器按 `env_offset` 分轮发送 chunk：

```text
round 0: 每个 slot 的第 0 个 env 并发执行
round 1: 每个 slot 的第 1 个 env 并发执行
...
```

当 `envs_per_core = 2` 时，一个 slot 上的两个 env 不是同时发起，而是分两轮依次跑。
如果开启动态 affinity，它们会被绑定到同一个 CPU core group 上，因此符合“一个核心上
依次跑两个 env”的执行模型。

返回结果会被恢复到输入 env 顺序，外部看到的 API 仍然保持：

```text
obs_list:           list length = chunk_step
chunk_rewards:      [num_envs, chunk_step]
chunk_terminations: [num_envs, chunk_step]
chunk_truncations:  [num_envs, chunk_step]
infos_list:         list length = chunk_step
```

## CPU 绑核机制

### 静态资源分配

CPU core group 由 fine-grained resource pool 生成。关键代码：

```text
rlinf/scheduler/resource_pool/solver.py
  ResourcePoolSolver._solve_cpu_component()
```

当满足以下条件时：

- `cluster.resource_pool.cpu.enabled: true`
- env component 配置了 `granularity: per_env`
- `env.train.chunk_step_mode: latency_balanced_pair`

resource pool 会按 `envs_per_core` 计算 slot 数，并生成：

```text
RLINF_ENV_CPU_CORE_GROUPS
```

这个环境变量是分号分隔的 per-env core group，例如：

```text
0;1;0;1
```

对于 `envs_per_core = 2`，这表示 env0/env2 使用 core 0，env1/env3 使用 core 1。
在 latency-balanced 模式下，后续动态配对会按 slot 选取这些 core group。

### 子进程初始 affinity

LIBERO 子进程启动时会读取当前 env 对应的 core group：

```text
rlinf/envs/libero/venv.py
  _apply_subproc_env_cpu_affinity(local_env_index)
```

这一步提供初始绑核。

### 动态 affinity

每次重新构建 latency-balanced groups 后，如果：

- `dynamic_affinity: true`
- 当前 vector env 能看到 `_env_cpu_core_groups`

调度器会把同一个 slot 内的 env 子进程绑定到同一个 core group：

```text
for slot_index, group in enumerate(pair_groups):
    core_group = slot_cpu_core_groups[slot_index]
    for local_pos in group:
        worker.set_cpu_affinity(core_group)
```

子进程内部最终调用 Linux affinity API：

```text
os.sched_setaffinity(0, set(cpus))
```

注意：如果 `cluster.resource_pool.cpu.enabled: false`，就不会注入
`RLINF_ENV_CPU_CORE_GROUPS`，因此 `dynamic_affinity: true` 也没有 core group 可用。
这时仍会做延迟配对，但 OS 层不会执行同 core 绑定。

## 推荐启用配置

如果目标是明确实现“每个 core slot 依次跑两个 env”，建议把测试配置改成：

```yaml
cluster:
  resource_pool:
    enabled: true
    allocation_mode: default
    cpu:
      enabled: true
      pools:
        env_cpu:
          node_group: cluster
          cores: ${oc.env:RLINF_ENV_CPU_CORES,0-111}
      components:
        env:
          pool: env_cpu
          granularity: per_env

env:
  train:
    chunk_step_mode: latency_balanced_pair
    log_sim_timestamps: true
    log_sim_affinity_interval: 1
    latency_balanced_pair:
      envs_per_core: 2
      ema_alpha: 0.3
      initial_latency_ms: null
      dynamic_affinity: true
```

如果机器有 112 个可用 CPU core，当前配置的训练部分需要：

```text
env workers = 8
local envs per worker = 28
pipeline stages = 2
per-stage envs = 14
envs_per_core = 2
slots per stage = 7
slots per worker = 14
total slots = 8 * 14 = 112
```

这和默认 `RLINF_ENV_CPU_CORES=0-111` 正好匹配。

## 校验和日志

### resource pool plan

启动脚本会在 Hydra 输出目录写：

```text
resource_pool_plan.json
```

检查 env binding 中是否包含 CPU 字段：

```json
{
  "component": "env",
  "cpu": {
    "process_cpu_cores": [0, 1, ...],
    "env_cpu_core_groups": [[0], [1], ...]
  }
}
```

如果 `cpu` 是 `null`，说明 CPU resource pool 没有启用或没有给 env component 生成绑定。

### sim timestamp 日志

配置中已经打开：

```yaml
log_sim_timestamps: true
log_sim_affinity_interval: 1
```

`EnvWorker.env_interact_step()` 会记录：

- chunk step start/end 时间。
- env worker 进程 affinity。
- 子 env affinity sample。

如果 CPU 绑核生效，日志里的 `child_affinity_sample` 应该能看到子 env 被限制到较小的
core group，而不是整个机器的 CPU 集合。

### 运行时日志

第一次启用 latency-balanced pair 时，vector env 会输出类似信息：

```text
latency_balanced_pair enabled: local_envs=14, envs_per_core=2,
slot_count=7, cpu_groups=7, first_groups=...
```

其中：

- `local_envs` 是当前 stage 的 env 数。
- `envs_per_core` 是每个 slot 的 env 数。
- `slot_count` 是当前 stage 需要的 slot 数。
- `cpu_groups` 大于 0 表示看到了 CPU core group。

如果 `cpu_groups=0`，说明只启用了配对调度，没有启用实际绑核。

## 限制和注意事项

- 该模式要求当前 stage 的本地 env 数能被 `envs_per_core` 整除。
- `envs_per_core` 必须大于等于 1，且不能超过本地 env 数。
- 该模式依赖每个 env 有独立子进程，适合 LIBERO/RoboCasa 这类
  `SubprocVectorEnv` 风格后端。
- 单进程 batched simulator 只能做 per-worker 绑核，不能做真正的 per-env 绑核。
- 绑核只控制 OS 调度，不会限制 MuJoCo、BLAS、OpenMP 等库自己开的线程。必要时建议设置：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

## 相关单测

当前已有相关单测覆盖：

```text
tests/unit_tests/test_chunk_step_parallel.py
  test_latency_balanced_pair_mode_is_registered
  test_subproc_vector_env_latency_balanced_pair_restores_step_major_shape
  test_latency_balanced_pair_rejects_invalid_group_size
  test_latency_balanced_pair_groups_slow_and_fast_envs

tests/unit_tests/test_resource_pool_solver.py
  test_default_solver_shares_env_cpu_groups_for_latency_balanced_pair
  test_default_solver_uses_per_stage_slots_for_latency_balanced_pair
  test_default_solver_reuses_per_stage_slots_for_one_env_per_core
```

建议验证命令：

```bash
pytest -q \
  tests/unit_tests/test_chunk_step_parallel.py \
  tests/unit_tests/test_resource_pool_solver.py \
  tests/unit_tests/test_resource_pool_cpu_binding.py \
  tests/unit_tests/test_resource_pool_env_binding.py
```

当前环境如果没有安装 `pytest`，需要先进入项目对应 Python 环境再执行。

## 快速判断是否真的启用了完整能力

检查顺序：

1. `env.train.chunk_step_mode` 是否是 `latency_balanced_pair`。
2. `latency_balanced_pair.envs_per_core` 是否是期望值，默认为 `2`。
3. `cluster.resource_pool.cpu.enabled` 是否是 `true`。
4. env component 是否配置了 `granularity: per_env`。
5. `resource_pool_plan.json` 里 env binding 是否有 `env_cpu_core_groups`。
6. sim timestamp 日志里 `child_affinity_sample` 是否显示子 env 只绑定到 slot 的 core group。

如果 1 和 2 成立，但 3 到 6 不成立，说明只启用了“按延迟拼图调度”，没有启用“同 core
依次执行”的 OS 绑核部分。
