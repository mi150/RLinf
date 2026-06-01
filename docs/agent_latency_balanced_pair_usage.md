# Agent 改动说明：Env Chunk Step 保留方案

当前 env rollout 只保留三类可对比方案：

1. **不绑定 CPU，不调度**：`sync_time_major`，关闭/不配置 env CPU resource pool。
2. **绑定 CPU，不调度**：`sync_time_major`，启用 env `per_env` CPU resource pool。
3. **core donation v2**：`latency_balanced_pair`，要求 `envs_per_core=1`、`dynamic_affinity=true`、`core_donation_enabled=true`，并且必须存在 per-env CPU core groups。

旧实验形态不再作为可运行方案保留，包括 `envs_per_core=2`、`dynamic_affinity=false`、以及只动态重绑但不启用 core donation 的 pairing 变体。

## 配置方式

### 1. 不绑定 CPU，不调度

使用 `sync_time_major`，不要给 env 配 CPU resource pool：

```yaml
env:
  train:
    chunk_step_mode: sync_time_major
```

### 2. 绑定 CPU，不调度

仍使用 `sync_time_major`，但启用 env `per_env` CPU binding：

```yaml
cluster:
  resource_pool:
    enabled: true
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
    chunk_step_mode: sync_time_major
```

### 3. Core Donation V2

`latency_balanced_pair` 只支持这一种形态：

```yaml
cluster:
  resource_pool:
    enabled: true
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
    latency_balanced_pair:
      envs_per_core: 1
      ema_alpha: 0.3
      initial_latency_ms: null
      dynamic_affinity: true
      core_donation_enabled: true
      core_donation_max_extra_groups: 1
```

如果 `latency_balanced_pair` 没有 per-env CPU core groups，运行时会报错。无 CPU 绑定场景请使用 `sync_time_major` baseline。

## Core Donation V2 行为

每个子 env 都先绑定到自己的 base CPU core group。父进程 dispatch 每个 env 的完整 action chunk；当某个 env 完成后，如果还有慢 env 在执行，父进程会把已完成 env 的 core group 临时 donation 给慢 env。chunk 结束后恢复到 base affinity。

运行中的 donation/restore 使用子进程 pid 直接 `sched_setaffinity`，不经过 env worker command pipe，避免和正在返回的 `chunk_step` 结果串包。

## Profile 字段

开启：

```yaml
env:
  train:
    log_sim_timestamps: true
    log_sim_affinity_interval: 0
```

输出位置：

```text
<runner.logger.log_path>/env_sim_timestamps/env_rank_<rank>.jsonl
```

重点字段：

- `event=end` 的 `duration_s`：EnvWorker 视角的一次 chunk interaction 总耗时。
- `event=end.chunk_profile.total_s`：父进程 chunk step 总耗时。
- `wait_recv_s`：父进程等待子 env 完成并接收结果的总耗时，通常是主要开销。
- `recv_call_s`、`dispatch_call_s`、`stack_s`：父进程非仿真等待开销。
- `core_donation_count`：当前 chunk 发生 donation 的次数。
- `core_donation_s` / `core_donation_restore_s`：donation 与恢复 affinity 的开销。
- `event=subenv_end.duration_s`：单个子 env 完整 action chunk 的耗时。

## 运行示例

准备环境变量：

```bash
export RLINF_ENV_CPU_CORES=0-111
export RLINF_ENV_RENDER_GPUS=0-7
export LIBERO_TYPE=standard
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

绑定 CPU baseline：

```bash
bash examples/embodiment/run_embodiment.sh 0libero_pipeline_test LIBERO
```

Core donation v2：

```bash
bash examples/embodiment/run_embodiment.sh 0libero_pairing_test LIBERO
```

RoboCasa profile 配置：

```text
examples/embodiment/config/robocasa_profile_pairing_dynamic_core_donation.yaml
```

## 校验规则

`validate_cfg()` 会拒绝非 v2 的 `latency_balanced_pair` 配置：

- `envs_per_core` 必须为 `1`。
- `dynamic_affinity` 必须为 `true`。
- `core_donation_enabled` 必须为 `true`。
- `ema_alpha` 必须在 `(0, 1]`。
- `initial_latency_ms` 如果设置，必须大于 `0`。
- `core_donation_max_extra_groups` 必须大于等于 `0`。

## 验证命令

```bash
pytest -q tests/unit_tests/test_chunk_step_parallel.py
pytest -q tests/unit_tests/test_resource_pool_solver.py
ruff check \
  rlinf/envs/venv/venv.py \
  rlinf/envs/robocasa/robocasa_env.py \
  rlinf/envs/libero/libero_env.py \
  rlinf/config.py \
  rlinf/scheduler/resource_pool/solver.py \
  tests/unit_tests/test_chunk_step_parallel.py \
  tests/unit_tests/test_resource_pool_solver.py
```
