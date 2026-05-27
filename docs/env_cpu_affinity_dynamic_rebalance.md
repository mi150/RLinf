# Env CPU Affinity and Dynamic Rebalance Design

本文总结一种让多个 env 绑定到 CPU core，并根据运行耗时动态调整绑定关系的方法。这个方案主要面向每个 env 有独立 OS 进程的后端，例如 `SubprocVectorEnv` 类环境。对于单进程向量化仿真，只能绑定整个 env worker 进程，无法做到每个 env 单独绑核。

## 目标

Env worker 通常会管理多个 env。如果所有 env 自由竞争 CPU，慢 env 会拖慢整个 chunk action 的完成时间；如果静态把 env 绑定到 core，又会遇到不同 env 耗时不均的问题。目标是：

- 每个 env 子进程限制在指定 CPU core 上运行。
- 每个 core 可以承载多个 env。
- 运行若干个 chunk action 后，根据每个 env 的实际耗时重新分配 env 到 core。
- 降低最慢 core 的总耗时，减少 rollout 等待尾部 env 的时间。

## 前置条件

动态重排依赖静态绑核能力。也就是说，系统必须先能做到：

1. 知道每个 env 对应哪个 OS 进程。
2. 能拿到该进程的 `pid`。
3. 能在安全时机调用 `os.sched_setaffinity(pid, {core_id})`。
4. 能记录每个 env 完成一个 chunk action 的耗时。

如果 env 绑核还没有实现，动态重排也暂时不好落地，因为动态重排的本质就是不断更新 `env_id -> core_id` 映射，并把这个映射应用到 env 子进程。

## 适用范围

### 适合支持

以下模式适合做 per-env 绑核和动态重排：

- 每个 env 是独立 `multiprocessing.Process`。
- env worker 里能访问子进程对象，例如 `SubprocEnvWorker.process.pid`。
- chunk step 结束后，env worker 能收集每个 env 的耗时。

相关代码位置：

- `rlinf/workers/env/env_worker.py`
  - `EnvWorker._setup_env_and_wrappers()` 创建每个 pipeline stage 的 env 实例。
  - `EnvWorker.env_interact_step()` 调用 env 的 step/chunk step，是收集耗时和触发 rebalance 的候选位置。
- `rlinf/envs/venv/venv.py`
  - `SubprocEnvWorker` 启动子进程。
  - `_worker()` 是子进程入口，适合在 env 创建前设置初始 affinity。
  - `BaseVectorEnv.workers` 保存每个子 env worker。
- `rlinf/envs/libero/venv.py`、`rlinf/envs/metaworld/venv.py`、`rlinf/envs/robocasa/venv.py`、`rlinf/envs/habitat/venv.py`、`rlinf/envs/calvin/venv.py`
  - 这些文件有各自的 reconfigurable subproc worker，需要同步接入通用 affinity 逻辑。

### 不适合直接支持

以下模式不能做到 per-env 绑核：

- 单进程内向量化仿真，例如多个 env 只是同一个进程里的 batched object。
- GPU/仿真引擎内部统一调度多个 env，外部看不到每个 env 的 OS 进程。

这类环境最多支持 per-worker 绑核，也就是把整个 Ray actor/EnvWorker 进程限制到一组 CPU core。不能对单个 env 动态迁移 CPU core。

TODO(agent): 如果后续要支持 ManiSkill/IsaacLab 这类单进程向量化 env，需要先确认是否能拆成多个 env 子进程，或者仿真引擎是否提供内部 thread/core affinity 接口。

## 配置建议

建议先加一个显式配置，不默认开启：

```yaml
env:
  train:
    cpu_affinity:
      enabled: true
      mode: per_env
      core_ids: null
      cores_per_env: 1
      rebalance:
        enabled: true
        interval_chunks: 2
        ewma_alpha: 0.3
        min_improvement_ratio: 0.1
        max_migrations_per_round: 4
        cooldown_chunks: 4
  eval:
    cpu_affinity:
      enabled: false
```

字段含义：

- `enabled`: 是否开启 CPU affinity。
- `mode`: `per_env` 表示对子 env 进程绑核；`per_worker` 表示只绑 EnvWorker 进程。
- `core_ids`: 可用 CPU core 列表。为 `null` 时使用 `os.sched_getaffinity(0)` 返回的当前可见 CPU 集合。
- `cores_per_env`: 一般先用 `1`。如果 env 内部会稳定多线程，可以设置为大于 1，但调度会更复杂。
- `rebalance.enabled`: 是否开启动态重排。
- `interval_chunks`: 每多少个 chunk action 重排一次。
- `ewma_alpha`: env 耗时的 EWMA 平滑系数。
- `min_improvement_ratio`: 只有预计最慢 core 负载下降超过该比例才迁移。
- `max_migrations_per_round`: 每次最多迁移多少 env，避免扰动过大。
- `cooldown_chunks`: 单个 env 两次迁移之间至少间隔多少个 chunk。

## 绑核实现方式

Linux 下切换进程 CPU affinity 很简单：

```python
os.sched_setaffinity(pid, {core_id})
```

如果在子进程内部设置当前进程：

```python
os.sched_setaffinity(0, {core_id})
```

建议分两层做：

1. 子进程启动时设置初始 affinity。
2. chunk 结束后，如果 rebalance 决定迁移，再由父进程对目标 `pid` 调用 `os.sched_setaffinity(pid, {new_core_id})`。

这样不需要重启 env，也不需要搬运 env 状态。env 子进程继续运行，只是后续被 OS 调度到新的 CPU core。

需要注意：

- `core_id` 必须在当前进程可见 CPU 集合里，尤其是在容器/cgroup/Ray 环境下。
- 推荐设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`，避免 env 内部库开线程抢核。
- 不要在 env 正在 step 时迁移。推荐在 chunk step 完成后、下一批 action 发送前迁移。

## 动态重排算法

每个 env 维护一个平滑耗时：

```text
cost[env] = alpha * latest_chunk_time[env] + (1 - alpha) * cost[env]
```

每隔 `interval_chunks` 个 chunk 执行一次重排：

1. 读取当前 `env_id -> core_id`。
2. 用 `cost[env]` 估算每个 core 的负载：

   ```text
   load[core] = sum(cost[env] for env assigned to core)
   ```

3. 使用 LPT greedy 重新分配：

   ```text
   envs = sort_by_cost_desc(envs)
   for env in envs:
       core = argmin(load)
       assign env to core
       load[core] += cost[env]
   ```

4. 对比旧分配和新分配的预计 `max(load)`。
5. 如果预计收益小于 `min_improvement_ratio`，放弃迁移。
6. 应用 cooldown 和 `max_migrations_per_round` 限制。
7. 对需要迁移的 env 子进程调用 `os.sched_setaffinity(pid, {new_core})`。

这个算法不是最优 bin packing，但实现简单、稳定，适合在线调度。不要每一步都重排，否则会引入调度抖动和缓存局部性损失。

## 耗时统计

建议统计 chunk 级别耗时，而不是单步耗时：

- 单步耗时噪声大。
- rollout 等待的是 chunk action 完成。
- 动态重排的目标是减少 chunk 尾部等待。

统计方式有两种：

### 父进程 wall time

父进程在 `send(action)` 和 `recv()` 周围统计每个 env 的完成时间。实现简单，但会包含排队和 CPU 竞争影响。

### 子进程内部执行时间

在子进程 `_worker()` 内部围绕 `env.step()` 统计时间，并把耗时随 step 结果返回。这个更接近 env 自身执行成本，推荐优先使用。

如果子进程内部统计需要改返回协议，可以先把耗时塞进 `info`：

```python
info["env_step_time_s"] = elapsed
```

但要确认不同 env 的 `info` 类型和结构一致，否则需要在 vector env 层做兼容。

## 需要修改的文件

一个最小可用实现大概需要改这些位置：

- `rlinf/config.py`
  - 增加 `env.*.cpu_affinity` 的默认配置和校验。
  - 校验 `core_ids`、`mode`、`rebalance` 参数。
- `rlinf/workers/env/env_worker.py`
  - 计算每个 EnvWorker 可用的 core 池。
  - 给每个 stage/env 分配初始 core。
  - 在 chunk step 后收集 env 耗时。
  - 调用 rebalance scheduler 并应用新的 affinity。
- `rlinf/envs/venv/venv.py`
  - 给 `SubprocEnvWorker` 传入 `env_id` 和初始 `core_id`。
  - 子进程启动时设置 affinity。
  - 暴露 `pid` 和 `set_affinity(core_id)` 方法。
  - 可选：在子进程内部统计 step/chunk step 耗时。
- `rlinf/envs/*/venv.py`
  - 对复制了 subproc worker 的环境同步接入，避免只有基础 `SubprocVectorEnv` 生效。
- `tests/unit_tests/`
  - 加 CPU affinity scheduler 的纯单元测试。
  - 用 fake process/fake pid mock `os.sched_setaffinity`，不要依赖真实 CPU 拓扑。

如果只想先验证方法，可以先只支持 `rlinf/envs/venv/venv.py` 的通用 `SubprocVectorEnv`，然后逐步扩展到 LIBERO/MetaWorld/Robocasa/Habitat/Calvin 的定制 venv。

## 为什么没有 env 绑核时动态重排不好做

动态重排需要把调度结果落到系统层：

```text
new_assignment = {env_0: core_3, env_1: core_7, ...}
apply(new_assignment)
```

如果没有 env 绑核能力，`apply()` 不存在。此时即使知道哪个 env 快、哪个 env 慢，也只能得到一张计划表，无法让 OS 按计划调度进程。

另外，如果 env 不是独立进程，就没有 per-env 的 `pid`。这种情况下 `env_id -> core_id` 不是可执行的系统操作，只能变成 `env_worker_pid -> core_set`，粒度太粗，解决不了单个 env 的拼图问题。

因此推荐实施顺序是：

1. 先实现 per-env 静态绑核。
2. 再实现耗时统计。
3. 最后实现动态 rebalance。

## 风险和限制

- 动态迁移可能降低 CPU cache locality，所以需要 cooldown 和收益阈值。
- wall time 统计会受到当前绑定方案影响，可能把排队时间误判为 env 固有耗时。
- 多个 Ray actor 不能共享同一批 core，否则 worker 之间会互相干扰。最好在 EnvWorker 启动时分配专属 core pool。
- 对单进程向量化 env 不支持 per-env 绑核。
- 在容器/cgroup 环境中，`core_ids` 必须遵守当前进程可见 CPU 集合。
- macOS/Windows 没有同样的 `os.sched_setaffinity` 接口，该方案按 Linux 优先设计。

## 推荐结论

这个方案可以做，但需要先把 per-env CPU affinity 作为基础能力补上。动态重排本身不难，核心是一个在线负载均衡器；真正需要投入的是把不同 env 后端的进程模型统一起来，并保证每个 env 子进程都能被定位、统计和重新设置 affinity。

如果短期只需要验证收益，建议先选一个 `SubprocVectorEnv` 后端做最小闭环：

1. 每个 env 子进程启动时静态绑核。
2. 记录每个 env 的 chunk 耗时。
3. 每 2 到 5 个 chunk 用 LPT greedy 重排。
4. 只在预计收益明显时调用 `os.sched_setaffinity` 迁移。

验证有效后，再扩展到各个定制 venv。
