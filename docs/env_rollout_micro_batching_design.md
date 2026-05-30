# Env-Rollout Observation Micro-Batching 设计与实现说明

本文说明 `rollout.observation_micro_batching` 的设计、配置和当前实现。这个功能
的目标不是做 DGQ，而是解决一个更直接的问题：

```text
只启动一个 EnvWorker
  -> 这个 EnvWorker 管理全部 env subprocess
  -> EnvWorker 产生 observation 后放入中间 buffer
  -> 多个 RolloutWorker 谁空闲谁从 buffer 取一批 observation 推理
  -> RolloutWorker 把 action chunk result 放回 result buffer
  -> EnvWorker 按 env_ids 恢复原始 env 顺序
```

也就是说，这个方案的核心是 **一个 EnvWorker 对多个 RolloutWorker 的推理负载
分发**。它和 `dynamic_global_queue` / `windowed_dynamic_global_queue` 是不同
层面的功能：

- `observation_micro_batching`：管 EnvWorker 和 RolloutWorker 之间怎么传
  observation / action result。
- `dynamic_global_queue`：管单个 EnvWorker 内部的 env subprocess 怎么执行
  action chunk。
- `windowed_dynamic_global_queue`：管单个 EnvWorker 内部的 window / lane
  streaming 调度。

当前文档只说明第一项。

## 一句话总结

EnvWorker 不再把一个完整 env batch 固定发给某个 RolloutWorker，而是把
observation 按 `batch_size` 切成 micro-batch，放入 shared obs buffer。多个
RolloutWorker 从这个 buffer 抢任务；推理完成后把带 `env_ids` 的 result 放回
shared result buffer。EnvWorker 根据 `env_ids` 把 result 回填到原始 env 行。

当前实现仍然是在 chunk 边界工作：

- EnvWorker 先得到一批 observation。
- 达到 `rollout.observation_micro_batching.batch_size` 后切成 micro-batch。
- micro-batch 被空闲 RolloutWorker 消费。
- EnvWorker 收齐当前需要的 rollout result 后，再进入下一次 env chunk 执行。

还没有实现“单个 env chunk 一完成就立刻进入 obs buffer”的完全异步 per-env
action-ready 流程。

## 配置形态

推荐配置形态：

```yaml
cluster:
  component_placement:
    env: 0
    rollout: 0-7

rollout:
  pipeline_stage_num: 1
  observation_micro_batching:
    enabled: true
    batch_size: 8
```

含义：

- `env: 0`：只启动一个 EnvWorker rank。
- `rollout: 0-7`：启动多个 RolloutWorker rank。
- `pipeline_stage_num: 1`：普通 micro-batching 当前不再使用固定 stage
  pipeline。
- `observation_micro_batching.enabled: true`：开启 EnvWorker 和 RolloutWorker
  之间的 obs/result buffer。
- `observation_micro_batching.batch_size: 8`：EnvWorker 每凑到 8 条
  observation 就形成一个 micro-batch，放入 obs buffer，等待空闲
  RolloutWorker 推理。

## 可选参数配置

### Rollout 侧参数

| 参数 | 示例值 | 含义 |
| --- | --- | --- |
| `rollout.observation_micro_batching.enabled` | `true` | 启用 EnvWorker 到 RolloutWorker 的 observation micro-batching。 |
| `rollout.observation_micro_batching.batch_size` | `8` | 每个 observation micro-batch 包含多少条 env observation。该值影响 rollout 推理任务粒度。 |
| `rollout.observation_micro_batching.mode` | `windowed_dgq` | 可选标记字段。普通 micro-batching 不需要配置；windowed DGQ 配置中可以写成 `windowed_dgq` 方便区分语义。 |
| `rollout.pipeline_stage_num` | `1` | 普通 observation micro-batching 当前要求为 `1`。如果选择 windowed DGQ，`pipeline_stage_num` 才表示 window 数。 |

### Placement 参数

| 参数 | 示例值 | 含义 |
| --- | --- | --- |
| `cluster.component_placement.env` | `0` | 推荐形态：只启动一个 EnvWorker rank。 |
| `cluster.component_placement.env` | `"0-7:0"` | 仍然只启动一个 EnvWorker rank 0，只是把资源 rank 0 到 7 都分配给这个 EnvWorker。仅靠这个配置不会把 env subprocess 均分到 8 张卡上，也不会启动 8 个 EnvWorker。 |
| `cluster.component_placement.rollout` | `0-7` | 启动多个 RolloutWorker，让它们从同一个 obs buffer 抢 micro-batch。 |

### Env 侧 GPU 轮转参数

| 参数 | 示例值 | 含义 |
| --- | --- | --- |
| `env.train.subproc_gpu_round_robin` | `true` | 可选。用于 LIBERO / RoboCasa。启用后，一个 EnvWorker 下的 env subprocess 会按 local env index 在该 EnvWorker 可见 GPU 之间 round-robin 绑定。 |

重点说明 `env: "0-7:0"`：

```text
resource ranks: 0,1,2,3,4,5,6,7
process ranks:  0
```

它表达的是“这些资源都归 EnvWorker rank 0 使用”。它不是：

```text
env 0 -> card 0
env 1 -> card 1
...
env 7 -> card 7
```

所以单靠 `env: "0-7:0"` 不会把所有 env 自动均分到 8 张卡上。env subprocess
的数量和归属仍然由单个 EnvWorker 创建和管理；多个 RolloutWorker 的并行来自
`cluster.component_placement.rollout: 0-7`，不是来自 `env: "0-7:0"`。

如果希望保持一个 EnvWorker，同时让它创建的 env subprocess 分散使用多张 GPU，
需要额外打开：

```yaml
env:
  train:
    subproc_gpu_round_robin: true
```

这时子进程绑定规则是：

```text
gpu_id = visible_gpus[(global_env_index) % len(visible_gpus)]
```

例如 EnvWorker 可见 GPU 为 `0,1,2,3,4,5,6,7` 时，env subprocess 会按
`0,1,2,3,4,5,6,7,0,1,...` 轮转。每个子进程在创建 MuJoCo / Robosuite env 前
会把 `MUJOCO_EGL_DEVICE_ID` / `EGL_VISIBLE_DEVICES` 设成对应的物理 GPU id。
这样 MuJoCo/EGL 不会继续默认落到 GPU0。

## 运行链路

当前实现的链路是：

```text
EnvWorker bootstrap / chunk_step 后得到 observation batch
  -> 按 batch_size 切成 observation micro-batch
  -> obs buffer: train_obs_micro_batches
  -> 空闲 RolloutWorker 抢一个 micro-batch
  -> RolloutWorker 推理 action chunk
  -> result buffer: train_rollout_micro_results
  -> EnvWorker 按 env_ids 回填 rollout result
  -> EnvWorker 得到完整 action batch
  -> EnvWorker 执行下一次 env chunk_step
```

这条链路解决的是 rollout inference 的负载分配问题。以前固定 batch 协议下，
EnvWorker 会按固定 rank 映射把 observation 发给固定 RolloutWorker；现在多个
RolloutWorker 可以共享一个任务队列，谁空闲谁推理。

## Buffer 消息

当前实现了两个逻辑 buffer，底层复用 RLinf 现有 `Channel`：

```text
obs buffer
  channel key: train_obs_micro_batches
  producer: EnvWorker
  consumer: 多个 RolloutWorker

result buffer
  channel key: train_rollout_micro_results
  producer: RolloutWorker
  consumer: EnvWorker
```

obs buffer 消息包含：

- `type`
- `mode`
- `stage_id`
- `env_ids`
- `env_output`

result buffer 消息包含：

- `type`
- `stage_id`
- `env_ids`
- `rollout_result`

`env_ids` 是关键字段。RolloutWorker 返回 result 的顺序可能和 EnvWorker 发送
micro-batch 的顺序不同，所以 EnvWorker 不依赖返回顺序，而是按 `env_ids`
回填到原始 env batch 行。

## 当前约束

当前普通 observation micro-batching 有这些约束：

- 只支持一个 EnvWorker。
- `rollout.pipeline_stage_num` 必须为 1。
- 主要用于 train rollout。
- 当前仍然是 chunk 边界上的 micro-batching，不是单 env 完成事件驱动的全异步
  buffer。
- 它不负责 env subprocess 的 CPU 调度；env subprocess 怎么执行 action chunk
  由当前 env 的 `chunk_step_mode` 决定。
- `subproc_gpu_round_robin` 当前只支持 LIBERO / RoboCasa。

## 和 DGQ / Windowed DGQ 的关系

这三个开关不要混成一个功能：

```text
rollout.observation_micro_batching.enabled
  -> 开启 EnvWorker/RolloutWorker 中间 buffer

env.train.chunk_step_mode: dynamic_global_queue
  -> 选择普通 DGQ env chunk 执行

env.train.chunk_step_mode: windowed_dynamic_global_queue
  -> 选择 windowed DGQ pipeline
```

普通 micro-batching 可以和 `dynamic_global_queue` 搭配，也可以作为
windowed DGQ 的 rollout 通信部分被复用。但它本身不是 DGQ。

## 已修改文件

- `rlinf/workers/env/env_worker.py`
  - 新增 obs micro-batch 发送逻辑。
  - 新增 indexed rollout result 接收逻辑。
  - 按 `env_ids` 将 rollout result 重排成原始 env 顺序。
  - micro-batching 结束后向 RolloutWorker 发送 stop 消息。

- `rlinf/workers/rollout/hf/huggingface_worker.py`
  - 新增 shared-queue micro-batch generate 模式。
  - 多个 RolloutWorker 通过同一个 obs queue key 抢占式消费 observation。
  - 推理完成后把 result 带 `env_ids` 发回 result queue。

- `rlinf/data/embodied_io_struct.py`
  - `RolloutResult` 新增可选 `env_ids`、`stage_ids` metadata。

- `rlinf/config.py`
  - 校验启用 micro-batching 时当前只允许一个 EnvWorker。
  - 校验普通 micro-batching 当前要求 `rollout.pipeline_stage_num == 1`。
  - 校验 `rollout.observation_micro_batching.batch_size >= 1`。

- `tests/unit_tests/test_chunk_step_parallel.py`
  - 覆盖 obs micro-batch 切分。
  - 覆盖 numpy / tuple / 嵌套 observation 与 `final_obs` 的 micro-batch 切分。
  - 覆盖 rollout result 按 `env_ids` 重排。
  - 覆盖非法 `env_ids` 和仅含 metadata/result flag 时的批大小推断。
  - 覆盖不同 stage result 乱序时的 pending buffer。

## 后续计划

后续如果要做完全异步的 observation/action buffer，目标形态是：

```text
某个 env subprocess chunk 完成
  -> 立即产生 observation
  -> observation 进入 obs buffer
  -> 空闲 RolloutWorker 推理
  -> action 进入 action buffer
  -> 该 env 进入 action-ready queue
```

这需要把 LIBERO / RoboCasa 的 chunk 后处理拆成按 env_id 增量更新，包括：

- `current_raw_obs`
- reward
- termination / truncation / done
- auto reset
- `final_observation`
- `final_info`
- intervene action / flag
- trajectory 写入

## 验证方式

运行单测：

```bash
/data1/gaobowen/RLinf/.venv-libero-openpi/bin/python \
  -m pytest tests/unit_tests/test_chunk_step_parallel.py
```
