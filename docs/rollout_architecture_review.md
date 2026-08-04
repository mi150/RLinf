# RLinf Rollout 架构 Review：Env Worker 与 Rollout Worker

> 本文基于 RLinf main 分支源码阅读整理，聚焦 embodied 场景下 `EnvWorker` 与
> `MultiStepRolloutWorker` 的底层实现与交互机制。所有结论均标注了对应的代码位置。

## 目录

- [结论摘要](#结论摘要)
- [1. 底层 Ray 的用法](#1-底层-ray-的用法)
- [2. Placement：把 component 映射到物理资源](#2-placement把-component-映射到物理资源)
- [3. Channel：跨 worker group 的异步数据管道](#3-channel跨-worker-group-的异步数据管道)
- [4. Placement 与 Channel 的职责划分](#4-placement-与-channel-的职责划分)
- [5. Env Worker ↔ Rollout Worker 的完整交互](#5-env-worker--rollout-worker-的完整交互)
- [6. 四种 rollout 策略变体](#6-四种-rollout-策略变体)
- [7. Review 意见](#7-review-意见)
- [附录：关键代码位置索引](#附录关键代码位置索引)

---

## 结论摘要

**是 Ray，但只用了 Ray 的"进程管理 + 控制面 RPC"，数据面完全自己实现。**

Env Worker 和 Rollout Worker 都是普通的 Ray actor（`ray.remote(cls)`），但它们之间传观测和
动作时**不走 Ray object store**，而是走 RLinf 自己维护的 torch.distributed 通信组
（GLOO / NCCL / CUDA IPC）。中间的 `Channel` 只承担"排队 + 定序"的角色，Ray RPC 仅用来传递
一个"我要 put/get"的信号。

两者的交互本质是一个 **per-chunk-step 的 ping-pong**：

```
env 送观测 → rollout 推理 → 送回动作 → env chunk_step → 送新观测 → ...
```

循环 `n_train_chunk_steps` 次。

调度层由两根正交的支柱支撑：**Placement 决定"谁在哪里"，Channel 决定"数据怎么流"**。前者是
启动期的一次性决策，后者是运行期的持续行为。

---

## 1. 底层 Ray 的用法

### 1.1 Worker → Ray actor

`WorkerGroup.launch()` → `_create_workers()` → `Cluster.allocate()`
（`rlinf/scheduler/cluster/cluster.py:665-790`）：

```python
remote_cls = ray.remote(cls)
options = {
    "runtime_env": {
        "py_executable": python_interpreter_path,
        "env_vars": merged_env_vars,
    },
    "name": worker_name,
    "scheduling_strategy": NodeAffinitySchedulingStrategy(
        node_id=node.ray_id, soft=False
    ),
}
actor = remote_cls.options(**options).remote(*cls_args, **cls_kwargs)
```

几个关键设计选择：

**不申请 `num_gpus` / `num_cpus`。** 整个 `rlinf/scheduler/` 下没有任何 `num_gpus=` 的
actor option。GPU 归属靠两件事实现：

1. `NodeAffinitySchedulingStrategy(soft=False)` 硬绑节点；
2. 注入 `CUDA_VISIBLE_DEVICES` 和 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`
   （`rlinf/scheduler/hardware/accelerators/nvidia_gpu.py:254-256`）。

**代价与收益：** Ray 完全不知道 GPU 被谁占了，配置写错只会在 OOM 时暴露；但换来的是可以把
actor / env / rollout 三个进程 collocate 在同一张卡上，从而启用 CUDA IPC 传张量。

**环境变量自举。** `WorkerGroup._create_workers` 注入 `RANK` / `WORLD_SIZE` /
`MASTER_ADDR` / `LOCAL_ACCELERATOR_RANK` 等，`Worker.__new__` 在远端进程里读取它们完成
自举（`rlinf/scheduler/worker/worker.py:323-376`）。这段有个
`"ActorClass(" not in cls.__name__` 判断，用来跳过 Ray 在本地先 new 一次 wrapper 的情况。

### 1.2 方法调用与句柄

`_attach_cls_func()` 把 worker 类的每个 public 方法挂成 `WorkerGroupFunc`，调用时对所有
rank fan-out `.remote()`，返回 `WorkerGroupFuncResult`。

该 result 对象在构造时就起了一个 daemon 线程跑 `ray.get`
（`rlinf/scheduler/worker/worker_group.py:465-490`），所以 runner 侧的
`env.interact(...)` 是**立即返回的非阻塞句柄** —— 这正是 env 和 rollout 能并发跑起来的原因。

失败处理比较暴力：任一 rank 抛异常，等待线程直接 `os.kill(self._pid, signal.SIGUSR1)`
然后 `exit(-1)`，整个 job 挂掉。没有细粒度重试或容错。

---

## 2. Placement：把 component 映射到物理资源

### 2.1 它解决什么问题

RLinf 一个 job 里有多个异构 component（actor / rollout / env / reward / critic），每个有
不同的资源需求：actor 要大显存做 FSDP，rollout 要跑推理引擎，env 可能只需要 CPU、也可能要
GPU 仿真、甚至要绑定真实机械臂。

Placement 回答的就是：**每个 component 起多少个进程、每个进程落在哪个节点的哪几张卡上。**

### 2.2 两层结构

**第一层 `PlacementStrategy`** —— 底层机制，三种实现
（`rlinf/scheduler/placement/__init__.py`）：

| 策略 | 语义 |
| --- | --- |
| `PackedPlacementStrategy` | 按硬件 rank 区间紧凑排布，支持 `num_hardware_per_process`（一进程多卡，用于 TP）和 `stride`（跨步取卡，用于 cudaIPC 权重同步） |
| `NodePlacementStrategy` | 只按节点排布，不涉及加速卡（`Channel` 自己用的就是这个） |
| `FlexiblePlacementStrategy` | 显式指定每个进程的资源列表 |

`get_placement()` 输出一组 `Placement` dataclass
（`rlinf/scheduler/placement/placement.py:163-197`），字段包括：

```python
rank                    # 进程在 worker group 内的全局 rank
cluster_node_rank       # 所在节点在集群中的 rank
local_accelerator_rank  # 节点内的 GPU 编号
visible_accelerators    # 该进程可见的加速卡列表 → CUDA_VISIBLE_DEVICES
local_hardware_ranks    # 分配到的本地硬件 rank（机器人等非 GPU 硬件）
node_group_label        # 所属 node group 标签
isolate_accelerator     # 是否隔离，决定 local rank 是否归零
```

**第二层 `ComponentPlacement`** —— 解析 YAML 的 `cluster.component_placement`，把声明式
配置翻译成上面的策略对象（`rlinf/scheduler/placement/placement.py:228-674`）：

```yaml
# 全 collocate：三个 component 各自每卡一进程 → 同一张 GPU 上 3 个 Ray actor 进程
cluster:
  component_placement:
    actor,env,rollout: all
```

```yaml
# 异构多机：不同 component 落在不同 node group，env 绑机器人硬件
cluster:
  num_nodes: 2
  component_placement:
    actor:
      node_group: a800
      placement: 0-7
    rollout:
      node_group: "4090"
      placement: 0-7
    env:
      node_group: robot
      placement: 0-3:0-7      # 4 台机器人，每台 2 个进程共享
```

`rlinf/utils/placement.py` 在此之上提供了业务层封装：

- `HybridComponentPlacement`（86-96 行）—— embodied 用，任意 component 可落任意卡；
- `ModelParallelComponentPlacement` —— reasoning 用，会判定 collocated /
  disaggregated / auto 模式，并据此选择权重同步方式（同卡 CUDA IPC，跨卡 NCCL）。

### 2.3 Placement 的下游影响

这是关键 —— Placement 不只是"排布"，它决定了三件事：

1. **`visible_accelerators` → `CUDA_VISIBLE_DEVICES`。** 前面提到 RLinf 不申请 Ray 的
   `num_gpus`，卡的归属完全由 Placement 计算后注入环境变量实现。
2. **通信路径。** env 和 rollout 是否在同一张 GPU 上，直接决定张量走 CUDA IPC 还是 NCCL
   （见 [5.4](#54-数据面三条不同速度的路径)）。
3. **world_size → 数据分片。** `ComponentPlacement.get_world_size("env")` 和
   `get_world_size("rollout")` 被 `CommMapper` 用来算 M:N batch 重分片方案
   （见 [5.3](#53-路由mn-rank-映射)）。

> 所以改一行 `component_placement`，通信拓扑和分片方案会跟着全变 —— 这是 RLinf
> "改配置不改代码就能切换执行模式"的机制来源。

---

## 3. Channel：跨 worker group 的异步数据管道

### 3.1 它解决什么问题

Placement 排好进程后，进程之间要传数据。裸用 `Worker.send/recv` 是**同步点对点**的，两边
必须严格配对、同时到场，否则死锁。但 RL 流水线的现实是：

- env 步进和 rollout 推理速度不一样，需要缓冲；
- 异步模式下 env / rollout / actor 各跑自己的 `while True`，根本没有全局节拍；
- decoupled 模式下"哪个 rollout 处理哪批数据"是运行时才确定的。

Channel 就是为此提供的**解耦层**：一个带 key 分桶的分布式 FIFO 队列。

### 3.2 核心特性

`ChannelWorker` 内部是 `dict[key -> asyncio.Queue]`
（`rlinf/scheduler/channel/channel_worker.py:230-420`），提供：

- **`put` / `get` 语义** —— 生产者消费者完全解耦，谁先到都不会死锁；
- **key 路由** —— `put(item, key=...)` 分桶。RLinf 用它承载路由信息：静态模式下 key 是
  七元组（见 [5.3](#53-路由mn-rank-映射)），pipeline 模式下 key 是目标 actor rank；
- **weight + `get_batch(target_weight=N)`** —— 按权重聚批，用于变长序列场景；
- **`maxsize` 背压** —— 队列满了 put 会阻塞，天然限流；
- **`distributed=True`** —— 每节点一个副本，按 key 就近路由（embodied runner 目前
  未启用，见 [7.2 P1](#p1--channel-是单点)）；
- **`local=True`** —— 进程内 `LocalChannel`，不跨进程。

### 3.3 实现：控制面走 Ray，数据面绕开 Ray

`Channel` 背后的 `ChannelWorker` 自己也是一个 `Worker` 子类（即 Ray actor），用
`max_concurrency=2**31-1` 启动以避免 get 阻塞 put。

`Channel.put()` 的实现（`rlinf/scheduler/channel/channel.py:361-420`）是理解整套机制的
关键：

```python
# 1) 一次 Ray RPC：只告诉 ChannelWorker "有人要 put，源地址是 X"
async_channel_work = AsyncChannelWork(
    channel_name=self._channel_name,
    channel_key=key,
    channel_actor=target_actor,
    method="put",
    src_addr=self._current_worker.worker_address,
)

# 2) 真正的数据通过 collective group（torch.distributed）传输
self._current_worker.send(
    item, self._channel_name, target_rank,
    async_op=True, piggyback_payload=(key, weight),
)
```

ChannelWorker 侧 `put()` 里对应地调 `self.recv(src_addr...)` 把数据收下来入队。

`get()` 是镜像的：`target_actor.get.remote(dst_addr, query_id, key)` 触发 ChannelWorker
出队并 `send` 给请求方，请求方 `self._current_worker.recv(...)` 接收。

> **大张量从不进 Ray object store，也不经过 plasma 序列化。**
> 这是 RLinf 相对朴素 Ray 实现的核心优化。

这也解释了为什么 Channel 是单点却还能用：它转发的是元数据量级的 RPC，不是几百 MB 的图像
batch。

此外 `Channel` 还提供 `put_via_ray` / `get_via_ray` 分支，用于在**非 worker 上下文**
（例如 runner 进程）中操作 channel，此时才回退到 Ray 通信。

---

## 4. Placement 与 Channel 的职责划分

回到 embodied 的 rollout 链路，两者的分工是这样衔接的：

```
Placement（启动期，一次）
  └─ 算出 env world_size=8、rollout world_size=8、都在 GPU 0-7
       ├─→ 注入 CUDA_VISIBLE_DEVICES，决定同卡 → 张量走 CUDA IPC
       └─→ 喂给 CommMapper，算出 8:8 的 batch 分片方案
                                    │
Channel（运行期，每 chunk step）      ▼
  └─ send_to/recv_from 按分片方案生成七元组 key
       ├─ Rollout channel: env → rollout（观测）
       ├─ Env channel:     rollout → env（动作）
       └─ Actor channel:   env → actor（轨迹）
```

| | Placement | Channel |
| --- | --- | --- |
| 时机 | 启动期，一次性 | 运行期，持续 |
| 关注 | 空间（进程 ↔ 硬件） | 时间（数据 ↔ 顺序 / 缓冲） |
| 产物 | `list[Placement]` → env vars | `asyncio.Queue` per key |
| 用 Ray 做 | actor 调度（NodeAffinity） | 控制面 RPC |
| 不用 Ray 做 | 资源账本（不报 `num_gpus`） | 数据搬运（走 collective） |

---

## 5. Env Worker ↔ Rollout Worker 的完整交互

### 5.1 通道拓扑

`EmbodiedRunner.__init__` 建了三个 channel（`rlinf/runners/embodied_runner.py:92-99`），
命名容易误导，实际方向是：

| Channel | 方向 | 载荷 |
| --- | --- | --- |
| `Rollout` | env → rollout | `{"obs", "final_obs"}` |
| `Env` | rollout → env | `RolloutResult`（动作、logprob、value、version） |
| `Actor` | env → actor | `Trajectory` / micro-batch |
| `Reward` | env → reward | 终局观测（启用 reward model 时） |

Runner 启动时（`rlinf/runners/embodied_runner.py:527-545`）：

```python
env_handle = self.env.interact(
    input_channel=self.env_channel,        # 收动作
    rollout_channel=self.rollout_channel,  # 发观测
    reward_channel=self.reward_channel,
    actor_channel=self.actor_channel,
)
rollout_handle = self.rollout.generate(
    input_channel=self.rollout_channel,    # 收观测
    output_channel=self.env_channel,       # 发动作
)
```

两个句柄都是非阻塞的，两组 actor 就此并发运行，靠 channel 的阻塞队列自然同步。

### 5.2 时序（同步模式，`pipeline_stage_num=1`）

Env 侧（`rlinf/workers/env/env_worker.py:1013-1247`）：

```
bootstrap_step()                                   # reset，拿 obs_0
_send_train_bootstrap()                            # 先"喂"一发，打破死锁
for chunk_step in range(n_train_chunk_steps):
    recv_from(rollout, tag="train_rollout_results") # ← 阻塞等动作
    compute_bootstrap_rewards(...)
    rollout_results[stage].append_step_result(...)  # 攒轨迹
    env_interact_step(actions)                      # 真正 chunk_step 仿真
    send_to(rollout, {"obs", "final_obs"})          # → 发新观测
# 循环外还有一轮 recv，用于拿最后一步的 bootstrap value
```

Rollout 侧（`rlinf/workers/rollout/hf/huggingface_worker.py:673-758`）：

```
for _ in range(n_train_chunk_steps):
    env_output = await recv_from(env, tag="train_rollout_results")
    actions, result = _predict_rollout_actions(env_output["obs"], ...)
    send_to(env, RolloutResult(...), split_fn=_split_rollout_result)
# 同样多一轮收尾
```

> 循环外那个 bootstrap send 很重要：如果 env 进循环就先 recv，两边会互等死锁。

### 5.3 路由：M:N rank 映射

`send_to` / `recv_from` 不是简单的 rank-to-rank，而是按 batch 维度做重分片
（`rlinf/scheduler/worker/routing.py`）：

- `CommMapper.get_dst_ranks()` 按全局 batch 切分，算出"我这份数据要拆成几块、分别给哪个
  目标 rank"；
- `CommMapper.get_src_ranks()` 是对偶操作。

这允许 env 和 rollout 的 world_size 不相等。

channel key 是七元组（`build_route_channel_key`）：

```python
(ROUTING_KEY_PREFIX, src_group_name, dst_group_name, tag, route_key, src_rank, dst_rank)
```

`route_key=stage_id` 用来隔离 pipeline stage 的独立流。

**可读性陷阱：** env 用 `mode="train", tag="rollout_results"` 拼成
`"train_rollout_results"`，rollout 回传时直接 `tag="train_rollout_results"`。两个方向
tag 字符串完全相同，只靠 key 里的 `src_group_name` / `dst_group_name` 区分。能工作，但读
代码时很容易误判成同一个流。

### 5.4 数据面：三条不同速度的路径

`CollectiveGroup._get_object_info()`
（`rlinf/scheduler/collective/collective_group.py:1300-1377`）按类型分派：

| 类型 | 条件 | 传输方式 |
| --- | --- | --- |
| `TENSOR` / `TENSOR_LIST` / `TENSOR_DICT` | 纯张量，或值全是张量的 list/dict | NCCL（GPU）/ GLOO（CPU） |
| `DATACLASS_WITH_TENSORS` | dataclass，抽出张量字段单独传 | 同上，元数据走 CPU |
| `OBJECT` | 其他一切 | pickle → CPU uint8 tensor → GLOO |

同卡还有额外优化：`_send_tensor_list_to_uncertain_peer` 会先交换 device identity，同一张
GPU 上的张量走 **CUDA IPC**，跨卡才走 NCCL
（`collective_group.py:1978-2000`、`2033-2062`）。

于是两个方向落在了不同的路径上：

- **rollout → env** 送 `RolloutResult`，是 `@dataclass(kw_only=True)`
  （`rlinf/data/embodied_io_struct.py:283`）→ 走 `DATACLASS_WITH_TENSORS` →
  IPC / NCCL。**快路径。**
- **env → rollout** 送的是 `_build_rollout_input_data()` 返回的
  `{"obs": {...}, "final_obs": {...}}`（`rlinf/workers/env/env_worker.py:923-931`）。
  值是嵌套 dict 而非张量 → 落到 `OBJECT` → **pickle + GLOO**。

> 这是本次 review 发现的最值得关注的一点：**图像观测是整条链路上最大的负载，却走了最慢的
> 那条路。**

---

## 6. 四种 rollout 策略变体

### 6.1 同步 ping-pong

入口：`EmbodiedRunner.run`。

最基础形态。env 和 rollout 严格交替，GPU 利用率取决于仿真与推理的耗时比。配
`enable_offload: True` 时两边还要反复搬模型 / 环境。

### 6.2 `pipeline_stage_num > 1`

把 `total_num_envs` 切成 N 个 stage，内层循环 `for stage_id in range(stage_num)`，每个
stage 独立 `route_key`。

效果是 env 在仿真 stage 0 时，rollout 已经能处理队列里 stage 1 的观测 —— 软件流水。

代价是每次推理 batch 变小（`train_batch_size = total_num_envs // stage_num`），需要权衡。

### 6.3 `use_training_pipeline: True`

入口：`EmbodiedRunner.run_pipeline`，actor 类为 `PipelineEmbodiedFSDPActor`。

env 不再把原始轨迹丢给 actor，而是自己完成：

1. `compute_advantages_and_returns`；
2. shuffle + 切 micro-batch（`pack_pipeline_micro_batches`）；
3. 按 actor rank 作为 channel key 推进 `actor_channel`
   （`rlinf/workers/env/env_worker.py:1440-1488`）。

actor 一边收一边训。advantage 归一化需要跨 env rank 同步，用 `self.broadcast(...)` 在
env group 内完成。

### 6.4 全异步

入口：`AsyncEmbodiedRunner`，worker 为 `AsyncEnvWorker` / `AsyncMultiStepRolloutWorker`。

两边都是 `while True` 的 asyncio task：

- 权重同步靠 `request_actor_sync_model()` 在后台协程里做
  （`_poll_background_weight_sync`）；
- off-policy 程度由 `algorithm.staleness_threshold` 控制；
- `RolloutResult.versions` 记录每条数据的模型版本，`wait_if_stale()` 做背压；
- `_run_interact_once(cooperative_yield=True)` 会在 stage 边界 `await asyncio.sleep(0)`
  让出控制权。

### 6.5 `enable_decoupled_mode`

放弃静态 rank 映射，改用携带回程路由的动态模式：

```
batch_index = "{send_rank}_{batch_idx}_{mode}_{tag}"
```

配合 `self.batch_router` 记录来源。rollout 用
`recv_from_and_record_batch_routes_with_timeout` 做超时聚批（任何 env 的数据都能收），
处理完再按 `batch_router` 发回原主（`send_to_recorded_batch_routes`）。

这实质上是给 rollout 加了 **dynamic batching**，要求
`env_world_size >= rollout_world_size`（`env_worker.py:__init__` 末尾有断言）。

---

## 7. Review 意见

### 7.1 值得肯定的设计

- **数据面绕开 Ray object store，同卡 CUDA IPC。** 这两点让 collocated 模式的通信开销
  接近手写 NCCL 方案。
- **Placement 与 Channel 职责正交。** 空间维度和时间维度分离，使得改一行 YAML 就能切换
  collocated / disaggregated / 异构多机，业务代码零改动。
- **`send_to` / `recv_from` 的 batch 级重分片抽象很干净。** M:N 解耦让 env 数和 GPU 数
  可以独立调整。
- **埋点覆盖完整。** `Worker.timer` 装饰器 + `consume_durations(return_per_rank=True)`
  让性能问题好定位。

### 7.2 问题与建议

#### P0 — env → rollout 的观测走 pickle + GLOO

`_build_rollout_input_data` 返回嵌套 dict，导致降级到 `OBJECT` 路径。

**建议：** 改成一个带张量字段的 dataclass（对齐 `RolloutResult` 的做法），或者把 obs 扁平
成 `dict[str, Tensor]`，即可吃到 IPC / NCCL。收益随图像分辨率和 env 数放大。

**影响范围可控：** 仅涉及 `_build_rollout_input_data`，以及 rollout 侧的
`_merge_obs_batches` / `_infer_env_batch_size`。

#### P1 — Channel 是单点

`Channel.create("Env")` 用默认 `distributed=False`，所有流量汇聚到 node_rank=0 的一个
ChannelWorker actor。

payload 不经它转发（走 collective），但每次 put/get 都要一次 Ray RPC + 一次 asyncio
入出队，控制面开销是 `O(n_chunk_steps × stage_num × world_size)`。

多机场景下 `distributed=True` 能按 node 就近路由，但 embodied runner 没启用。

#### P2 — `timeout_time=0.02` 硬编码

位置：`rlinf/workers/rollout/hf/async_huggingface_worker.py` 的
`decoupled_generate_one_epoch` 调用处。

这是 decoupled 模式下聚批延迟与吞吐的核心旋钮，应该走 config 暴露出来。

#### P2 — 双向复用同一 tag 字符串

见 [5.3](#53-路由mn-rank-映射)。建议改成 `"env2rollout_obs"` /
`"rollout2env_actions"` 之类的显式命名，语义不变但可读性大幅提升。

#### P2 — `Worker.send/recv` 的线程安全约束只写在文档里

`Worker.send` / `recv` 的 docstring 明确标注 *not thread safe*，而 ChannelWorker 用了
极大的 `max_concurrency`。

目前异步路径都在单个 asyncio event loop 里所以安全，但这是个隐性约束，未来若引入线程池
会踩坑。建议在 `CollectiveGroup` 层面加断言，而不只是文档注释。

#### P3 — 容错粒度

单 rank 失败 → SIGUSR1 → 全 job 退出。对实机场景（`RealWorld` env、硬件抖动）来说偏
脆弱，可以考虑对 env group 做 rank 级别的隔离降级。

---

## 附录：关键代码位置索引

| 主题 | 文件 | 行号 |
| --- | --- | --- |
| Ray actor 创建 | `rlinf/scheduler/cluster/cluster.py` | 665-790 |
| Worker 环境自举 | `rlinf/scheduler/worker/worker.py` | 323-376 |
| `send_to` 路由发送 | `rlinf/scheduler/worker/worker.py` | 842-1031 |
| `recv_from` 路由接收 | `rlinf/scheduler/worker/worker.py` | 1033-1230 |
| WorkerGroup fan-out | `rlinf/scheduler/worker/worker_group.py` | 220-470 |
| M:N batch 重分片 | `rlinf/scheduler/worker/routing.py` | 70-200 |
| `Placement` dataclass | `rlinf/scheduler/placement/placement.py` | 163-197 |
| `ComponentPlacement` 解析 | `rlinf/scheduler/placement/placement.py` | 228-674 |
| `PackedPlacementStrategy` | `rlinf/scheduler/placement/packed.py` | 22-335 |
| `HybridComponentPlacement` | `rlinf/utils/placement.py` | 86-96 |
| Channel put/get | `rlinf/scheduler/channel/channel.py` | 361-520 |
| ChannelWorker 队列 | `rlinf/scheduler/channel/channel_worker.py` | 230-420 |
| 载荷类型分派 | `rlinf/scheduler/collective/collective_group.py` | 1300-1377 |
| CUDA IPC / NCCL 分流 | `rlinf/scheduler/collective/collective_group.py` | 1978-2062 |
| Env rollout 主循环 | `rlinf/workers/env/env_worker.py` | 1013-1247 |
| Env 观测打包 | `rlinf/workers/env/env_worker.py` | 923-931 |
| Env pipeline 送轨迹 | `rlinf/workers/env/env_worker.py` | 1440-1488 |
| Rollout 主循环 | `rlinf/workers/rollout/hf/huggingface_worker.py` | 673-758 |
| Rollout decoupled 聚批 | `rlinf/workers/rollout/hf/huggingface_worker.py` | 234-453 |
| 同步 runner 编排 | `rlinf/runners/embodied_runner.py` | 92-99, 527-545 |
| 异步 runner 编排 | `rlinf/runners/async_embodied_runner.py` | 156-185 |
