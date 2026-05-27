# LIBERO CPU Affinity Switch Overhead

本文记录一次小样本实验，用于估计 LIBERO env 运行时切换 CPU affinity 的开销。背景设计见
`docs/env_cpu_affinity_dynamic_rebalance.md`。

## 结论

在本机测试里，`os.sched_setaffinity` 的切核开销是微秒级，明显小于 LIBERO
`env.step()` 的耗时。

- 单个进程在 core 0 和 core 1 之间切换 affinity：平均约 `7.11 us`。
- 父进程切一个子进程 pid 的 affinity：平均约 `1.24 us`。
- 两个 LIBERO env 子进程分别绑定 core 0 / core 1，然后互换绑定：一轮互换两个
  pid 平均约 `14.15 us`，也就是单个 env 迁移约 `7 us` 量级。
- 同一轮测试里，LIBERO `env.step()` 是 `ms` 到 `s` 级；因此动态 rebalance 的主要
  风险不是 syscall 本身，而是切换频率、cache locality、多 env 之间的资源竞争。

## 测试环境

- Python 环境：`/data1/gaobowen/RLinf/.venv-libero-openpi`
- 仓库：`/data1/gaobowen/RLinf_mmm`
- LIBERO suite：`libero_spatial`
- 任务：
  - 单 env 测试：task 0, trial 0
  - 双 env 互换测试：task 0 和 task 1, trial 0
- Camera：`64x64`
- CPU core：`0,1`
- 环境变量：

```bash
HOME=/tmp/rlinf-libero-home
XDG_CACHE_HOME=/tmp/rlinf-xdg-cache
MUJOCO_GL=egl
NUMBA_DISABLE_JIT=1
PYTHONPATH=/data1/gaobowen/RLinf_mmm
```

这里设置 `HOME=/tmp/rlinf-libero-home` 是因为当前默认 `~/.libero` 位置不可写；
`NUMBA_DISABLE_JIT=1` 是为了绕过 robosuite 在该 venv 形态下的 numba cache locator
问题。这个设置会影响绝对 step 性能，所以本实验只看 affinity 切换开销量级。

## 怎么切换

### 单 env 交替切换 core

这个测试在一个 LIBERO env 进程里，每个 `env.step()` 前把当前进程 affinity 在
core 0 和 core 1 之间交替切换：

```python
cores = [0, 1]

for step_index in range(num_steps):
    target_core = cores[step_index % 2]
    start_ns = time.perf_counter_ns()
    os.sched_setaffinity(0, {target_core})
    affinity_s = (time.perf_counter_ns() - start_ns) / 1e9

    start = time.perf_counter()
    env.step(dummy_action)
    step_s = time.perf_counter() - start
```

这个路径测的是“env 自己切换当前进程 core”的成本。

### 两个 LIBERO env 互换 core

这个测试更接近动态 rebalance 的应用方式：父进程持有两个 LIBERO env 子进程 pid。
初始状态：

```text
env0 pid -> core 0
env1 pid -> core 1
```

每轮 step 前，父进程把两个 pid 的绑定互换：

```python
current = [0, 1]
pids = [env0_pid, env1_pid]

for round_index in range(num_rounds):
    current = [current[1], current[0]]

    start_ns = time.perf_counter_ns()
    os.sched_setaffinity(pids[0], {current[0]})
    os.sched_setaffinity(pids[1], {current[1]})
    pair_affinity_s = (time.perf_counter_ns() - start_ns) / 1e9

    command_env0_step()
    command_env1_step()
    wait_both_done()
```

也就是：

```text
round 0: env0 -> core 1, env1 -> core 0
round 1: env0 -> core 0, env1 -> core 1
round 2: env0 -> core 1, env1 -> core 0
...
```

这个路径测的是“父进程在 chunk/step 边界迁移子 env 进程”的成本，和
`docs/env_cpu_affinity_dynamic_rebalance.md` 里的动态重排应用方式一致。

## 结果

### 纯 affinity syscall

| 操作 | 样本数 | 平均 | p95 | p99 | 最大 |
|---|---:|---:|---:|---:|---:|
| 当前进程重复设置同一个 core | 20000 | `0.75 us` | `0.77 us` | `0.99 us` | `10.22 us` |
| 当前进程在 core 0/1 间切换 | 20000 | `7.11 us` | `7.67 us` | `8.64 us` | `237.97 us` |
| 父进程切子进程 pid，在 core 0/1 间切换 | 20000 | `1.24 us` | `1.24 us` | `1.34 us` | `17.71 us` |

### 单个 LIBERO env 每步前切换 core

每种模式各测 30 个 step。

| 模式 | affinity 平均开销 | step 平均耗时 | 总耗时平均 |
|---|---:|---:|---:|
| 固定 core，不调用 affinity | `0 us` | `299.29 ms` | `299.29 ms` |
| 每步前重复设置同一个 core | `11.13 us` | `298.65 ms` | `298.66 ms` |
| 每步前在 core 0/1 间切换 | `32.11 us` | `298.31 ms` | `298.34 ms` |

这组结果里，切换 core 没有带来可测的端到端变慢；差异小于 step 本身波动。

### 两个 LIBERO env 互换 core

每轮父进程连续切两个 env 子进程 pid，然后两个 env 并发 step。共测 10 轮。

| 模式 | 一轮切两个 pid 的 affinity 平均开销 | 一轮切换 p95 | round wall time 平均 |
|---|---:|---:|---:|
| 不互换 | `0 us` | `0 us` | `1.379 s` |
| 每轮互换 core 0/1 | `14.15 us` | `15.98 us` | `1.060 s` |

`round wall time` 在互换模式下更低，不能解读为切核会加速 LIBERO；这只是小样本下任务、
渲染和 CPU 竞争波动。这里可靠的信息是：父进程完成“一次两个 pid 的互换”只需要十几微秒。

原始输出：

- `/tmp/rlinf-libero-affinity-switch/summary.json`
- `/tmp/rlinf-libero-affinity-switch/events.csv`
- `/tmp/rlinf-libero-two-env-affinity-swap/summary.json`
- `/tmp/rlinf-libero-two-env-affinity-swap/events.csv`

## 对动态 rebalance 的含义

如果动态 rebalance 只在 chunk 边界触发，并且每轮迁移数量有限，`sched_setaffinity`
本身不是瓶颈。例如迁移 4 个 env，按本次两 env 互换测试估计，系统调用总开销大约是
几十微秒级。

后续更需要验证的是：

- `256x256` 或训练实际 camera 配置下的 step 尾延迟。
- 更多 env 并发时，迁移是否破坏 cache locality。
- Ray actor / EnvWorker 是否共享同一批 core，避免 worker 之间互相抢核。
- 真实 chunk 边界迁移，而不是每个 step 都迁移。

