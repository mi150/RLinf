# Chunk Step 并行执行改造方案

本文档记录 `rlinf/envs/maniskill`、`rlinf/envs/robocasa`、`rlinf/envs/libero`、`rlinf/envs/behavior` 中 `chunk_step()` 的现状、目标行为和建议改造路径。

## 背景

当前这几个 env 的 `chunk_step(chunk_actions)` 都以时间维为外层循环：

```python
for i in range(chunk_size):
    actions = chunk_actions[:, i]
    obs, reward, terminations, truncations, infos = self.step(
        actions, auto_reset=False
    )
```

输入约定是：

```text
chunk_actions: [num_envs, chunk_step, action_dim]
```

当前执行顺序是：

```text
t = 0: env0 action[0,0], env1 action[1,0], ..., envN action[N,0]
t = 1: env0 action[0,1], env1 action[1,1], ..., envN action[N,1]
...
```

也就是所有 env 同步推进第一个 action，完成后再同步推进第二个 action。

用户期望改成“所有 env 并行做 action，但是各做各的”。更准确地说，执行单位应从“全局时间步”改成“每个 env 自己的 action chunk”：

```text
worker/env0: action[0,0] -> action[0,1] -> ... -> action[0,K-1]
worker/env1: action[1,0] -> action[1,1] -> ... -> action[1,K-1]
...
worker/envN: action[N,0] -> action[N,1] -> ... -> action[N,K-1]
```

父进程只负责并发派发、等待结果、按原始 env 顺序拼回返回值。

## 关键结论

不能只把 `for i in range(chunk_size)` 改成 `for env_id in range(num_envs)`。

原因是当前 `self.step(actions)` 的语义是“推进整个 vector env 一步”。在 ManiSkill、RoboCasa、LIBERO 这些实现里，一个 wrapper 通常持有一个 vectorized env；每次 step 都会推进这一组 env 的公共仿真时钟。如果仍然用同一个 vector env 对象，就很难让 env0 先连续走完整个 chunk，同时 env1 保持在原来的时间点。

正确方向是把并行粒度下沉到 env 实例或 env shard：

- 每个 worker 持有 1 个 env，最符合“各做各的”语义。
- 每个 worker 持有一小组 env shard，兼顾并行和 vectorized simulator 吞吐。
- 父 wrapper 对外仍暴露相同的 `chunk_step()` API 和返回形状。

## 改造目标

外部接口保持不变：

```python
obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = env.chunk_step(
    chunk_actions
)
```

返回形状保持不变：

```text
obs_list:            list length = chunk_step, each obs has num_envs batch dim
infos_list:          list length = chunk_step
chunk_rewards:       [num_envs, chunk_step]
chunk_terminations:  [num_envs, chunk_step]
chunk_truncations:   [num_envs, chunk_step]
```

内部执行顺序改成：

```text
parent chunk_step()
  split chunk_actions by env/shard
  dispatch each env/shard chunk to worker asynchronously
  each worker locally runs its own chunk loop
  gather results
  restore original env order
  apply done / truncation / auto_reset policy
  return same API shape
```

## 推荐设计

### 1. 新增通用的 shard chunk runner

建议新增一个通用辅助层，而不是在四个 env 里复制复杂逻辑。可以放在：

```text
rlinf/envs/utils/chunk_runner.py
```

建议职责：

- 管理多个 env worker 或 shard worker。
- 把全局 `chunk_actions` 切成局部 `local_chunk_actions`。
- 并发调用每个 worker 的 `chunk_step_local()`。
- 将 worker 返回的局部结果按 `env_indices` 拼回全局顺序。
- 统一处理 `chunk_rewards`、`terminations`、`truncations` 的 stack 和 scatter。

建议数据结构：

```python
@dataclass
class ChunkShard:
    env_indices: torch.Tensor  # global env ids in this shard
    worker: Any


@dataclass
class LocalChunkResult:
    obs_list: list
    infos_list: list
    rewards: torch.Tensor       # [local_num_envs, chunk_step]
    terminations: torch.Tensor  # [local_num_envs, chunk_step]
    truncations: torch.Tensor   # [local_num_envs, chunk_step]
```

### 2. 每个 worker 内部保留原来的时间循环

单个 worker 内部仍然可以这样执行：

```python
for i in range(chunk_size):
    actions = local_chunk_actions[:, i]
    obs, reward, terminations, truncations, infos = local_env.step(
        actions,
        auto_reset=False,
    )
```

区别是 `local_env` 只包含本 worker 负责的 env 或 shard，不再是全局所有 env。

这样可以保留原先的 done 语义，又能让不同 worker 并发运行。

### 3. 父进程并发派发

如果使用 Python 子进程，可以用 `multiprocessing.Pipe` / `Connection` 的非阻塞派发模式：

```python
for shard in shards:
    local_actions = chunk_actions[shard.env_indices]
    shard.conn.send(("chunk_step", {"chunk_actions": local_actions}))

local_results = []
for shard in shards:
    local_results.append(shard.conn.recv())
```

如果已有 Ray worker，也可以把每个 shard 做成 Ray actor：

```python
refs = [
    shard.worker.chunk_step_local.remote(chunk_actions[shard.env_indices])
    for shard in shards
]
local_results = ray.get(refs)
```

Ray 版本和 RLinf 的整体调度模型更一致，但会引入 actor 生命周期、序列化和资源声明的额外改动。子进程版本更接近 Behavior 当前实现。

## 共同改动和差异边界

四类 env 都要改一部分共同文件，也都要在各自 env 文件里接入新路径。共同文件里的逻辑应尽量完全复用；env 文件内部的改法只能保持接口一致，不能强行写成完全一样。

### 四类 env 都要改的文件

```text
rlinf/envs/utils/chunk_runner.py
  新增。放 shard 切分、结果拼接、done 聚合等纯数据逻辑。

rlinf/config.py
  新增或校验 chunk_step_mode / chunk_step_num_shards。

tests/unit_tests/test_chunk_step_parallel.py
  新增。用 dummy env / fake shard result 测公共逻辑。

examples/embodiment/config/*.yaml
tests/e2e_tests/embodied/*.yaml
  可选。只有需要显式启用 parallel_shard 的配置才加字段。
```

这几个文件里的改动对四类 env 来说应当是一套逻辑，不应该为 Behavior、RoboCasa、LIBERO、ManiSkill 各写一套。

### 公共逻辑应该完全一样的部分

这些逻辑建议只在 `rlinf/envs/utils/chunk_runner.py` 里实现一次，四类 env 直接调用：

```text
split_env_indices(num_envs, num_shards)
  输入全局 env 数和 shard 数。
  输出每个 shard 负责的全局 env id。

select_local_chunk_actions(chunk_actions, env_indices)
  从 [num_envs, chunk_step, action_dim] 里切出某个 shard 的动作。

scatter_chunk_results(local_results, shard_indices, num_envs)
  把多个局部结果按原始 env id 拼回全局 batch。

build_chunk_done_outputs(raw_terminations, raw_truncations, mode flags)
  复用当前 past_dones / only-last-step-done 的语义。

maybe_apply_ignore_terminations(raw_terminations, ignore_terminations)
  保留 RoboCasa / LIBERO 当前的 ignore_terminations 行为。
```

这些函数只处理 tensor/list/dict 的形状和顺序，不应该知道底层是 Behavior、RoboCasa、LIBERO 还是 ManiSkill。

公共逻辑的输入输出建议固定为：

```python
@dataclass
class LocalChunkResult:
    obs_list: list
    infos_list: list
    rewards: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor


@dataclass
class GlobalChunkResult:
    obs_list: list
    infos_list: list
    rewards: torch.Tensor
    raw_terminations: torch.Tensor
    raw_truncations: torch.Tensor
```

这样四类 env 的 `chunk_step()` 都可以长成类似结构：

```python
if self.chunk_step_mode == "sync_time_major":
    return self._chunk_step_sync_time_major(chunk_actions)

local_results = self._run_chunk_shards(chunk_actions)
global_result = scatter_chunk_results(
    local_results=local_results,
    shard_indices=self._chunk_shard_indices,
    num_envs=self.num_envs,
)
return self._finalize_chunk_result(global_result)
```

这里的 `_run_chunk_shards(...)` 是 env-specific，`scatter_chunk_results(...)` 和 `_finalize_chunk_result(...)` 尽量公共化。

### 各 env 文件里共同要接的入口

四个 env 文件都需要类似这几个接入点：

```text
<EnvClass>.__init__(...)
  读取 chunk_step_mode / chunk_step_num_shards。
  初始化 shard indices。
  需要并发时初始化 shard worker。

<EnvClass>.chunk_step(...)
  增加 mode 分支。
  sync_time_major 走旧实现。
  parallel_shard 走新 shard runner。

<EnvClass>._handle_auto_reset(...)
  尽量保留现有函数。
  父 wrapper 拼回全局 done 后再调用。

<EnvClass>.close(...)
  如果新增了子进程 / Ray actor / shard env，需要统一释放。
```

`step(...)` 是否改要谨慎。建议第一版只改 `chunk_step(...)`，让普通 `step(...)` 保持现状。这样 blast radius 小，已有同步 step 语义不会被无意改变。

### 各 env 文件里不能完全一样的部分

这些部分只能保持“接口一致”，实现要按 env 自己的底层结构写：

```text
worker 创建方式
  Behavior 已经有 _behavior_worker + Pipe。
  RoboCasa / LIBERO 可能需要新增通用 subprocess worker。
  ManiSkill 要考虑 GPU vectorized simulation 和 CUDA/SAPIEN 资源。

local env 初始化
  每个 shard 的 cfg、seed_offset、worker_info、task 分配规则不一样。

obs/info 包装
  Behavior 有 _wrap_obs / _record_metrics。
  RoboCasa / LIBERO 有自己的 obs dict、metric 和 done 处理。
  ManiSkill 多数 obs 可能已经在 GPU tensor 上，不能随便搬到 CPU。

auto_reset 细节
  四类 env 都有 _handle_auto_reset，但 final_observation / final_info 的结构和 device 可能不同。

close / cleanup
  子进程、simulator、renderer、GPU context 的释放方式不同。
```

所以结论是：

- 公共数据逻辑：应当完全一样，写在 `chunk_runner.py`。
- 配置校验：应当完全一样，写在 `rlinf/config.py`。
- `chunk_step()` 的外部 API：四类 env 必须一样。
- worker 生命周期、local env 初始化、obs/info 包装、资源释放：各 env 分别实现。

## 四类 env 的建议落点

这一节按“要改哪些文件、哪些函数”展开。核心原则是：`chunk_step()` 的对外返回不变，新增 shard/worker 并发执行路径；旧的时间维同步路径保留为默认路径。

### Behavior

Behavior 已经有子进程入口 `_behavior_worker()`，当前是在子进程里对一个 `VectorEnvironment` 做 chunk 时间循环。

建议优先从 Behavior 开始改，因为它已经有子进程 RPC 结构。

需要改的文件和函数：

```text
rlinf/envs/behavior/behavior_env.py
  _behavior_worker(...)
  BehaviorEnv.__init__(...)
  BehaviorEnv._call_subproc(...)
  BehaviorEnv.step(...)
  BehaviorEnv.chunk_step(...)
  BehaviorEnv._handle_auto_reset(...)
  BehaviorEnv.close(...)

rlinf/envs/utils/chunk_runner.py
  ChunkShard
  LocalChunkResult
  split_env_indices(...)
  scatter_chunk_results(...)
  build_chunk_done_outputs(...)

tests/unit_tests/test_chunk_step_parallel.py
  dummy shard runner tests
```

具体改法：

1. 创建多个 Behavior 子进程，每个子进程只负责一部分 `env_indices`。
2. 在 `BehaviorEnv.__init__()` 里根据 `chunk_step_num_shards` 初始化 `self._chunk_shards`。旧的单子进程字段可以保留，`chunk_step_mode == sync_time_major` 时仍走旧路径。
3. 把当前 `_call_subproc(cmd, payload)` 保留为单 worker 调用，再新增 `_call_chunk_shards(cmd, shard_payloads)` 或类似 helper，用于先向所有 shard `send()`，再统一 `recv()`。
4. 修改 `_behavior_worker()` 的 `"chunk_step"` 分支，让它返回 `LocalChunkResult` 语义的数据：局部 `raw_obs_list`、`chunk_rewards`、`raw_chunk_terminations`、`raw_chunk_truncations`、`infos_list`。worker 内部仍保留 `for i in range(chunk_size)`，因为此时它只负责自己的 local envs。
5. 修改 `BehaviorEnv.chunk_step()`：如果 `chunk_step_mode == sync_time_major`，走旧逻辑；如果是 `parallel_shard`，按 shard 切 `chunk_actions`，并发派发，收集后调用通用 `scatter_chunk_results(...)` 拼回全局 batch。
6. `BehaviorEnv.step()` 可以先不改，保持普通 step 的全局同步语义。后续如果希望 step/chunk_step 同一套执行模型，再让 `step()` 复用 shard runner。
7. `BehaviorEnv._handle_auto_reset()` 不建议下放到 worker。推荐父进程拼出全局 `past_dones` 后仍调用这个函数，这样 `final_observation`、`final_info`、mask 字段的语义不变。
8. `BehaviorEnv.close()` 要关闭所有 shard 子进程，不能只关原来的单个子进程。

注意：如果 Behavior 的任务分配依赖 `task_idx` 或 `num_envs`，初始化 shard 时要保证每个 worker 的 `local_num_envs` 和全局 `env_indices` 映射明确。不要让每个 shard 都误以为自己拥有完整的 `num_envs`。

### RoboCasa / LIBERO

RoboCasa 和 LIBERO 的实现非常相似，通常底层环境较重，单个进程里同步 vector step 容易成为瓶颈。

建议二者一起改，但分两个 env 文件落地；公共逻辑放到 utils，避免两个文件复制一份并发拼接代码。

需要改的文件和函数：

```text
rlinf/envs/robocasa/robocasa_env.py
  RobocasaEnv.__init__(...)
  RobocasaEnv.step(...)
  RobocasaEnv.chunk_step(...)
  RobocasaEnv._handle_auto_reset(...)
  RobocasaEnv.close(...)

rlinf/envs/libero/libero_env.py
  LiberoEnv.__init__(...)
  LiberoEnv.step(...)
  LiberoEnv.chunk_step(...)
  LiberoEnv._handle_auto_reset(...)

rlinf/envs/utils/chunk_runner.py
  same common shard/scatter helpers used by Behavior

rlinf/envs/utils/subproc_env_worker.py  # 可选
  generic subprocess worker for RoboCasa/LIBERO shards

tests/unit_tests/test_chunk_step_parallel.py
  shared scatter/done tests
tests/unit_tests/test_robocasa_env.py
  add mode-switch regression tests if mocking is feasible
```

具体改法：

1. 抽出单 shard wrapper：每个 shard 初始化自己的 env 实例，`num_envs = local_num_envs`。
2. 在 `RobocasaEnv.__init__()` 和 `LiberoEnv.__init__()` 中读取 `chunk_step_mode`、`chunk_step_num_shards`，创建 shard worker 列表。旧的单 wrapper 初始化仍保留给默认路径使用。
3. 新增一个本地函数或 worker 方法，例如 `_chunk_step_local(local_chunk_actions)`，内容基本就是当前 `chunk_step()` 的主体，但作用在 local env 上。
4. 修改 `RobocasaEnv.chunk_step()` 和 `LiberoEnv.chunk_step()`：默认走旧逻辑；`parallel_shard` 时，把 `chunk_actions[env_indices]` 发给各 shard，收集局部结果后 scatter 回全局结果。
5. `RobocasaEnv.step()` / `LiberoEnv.step()` 初期建议不改。因为普通 step 的输入是 `[num_envs, action_dim]`，它不需要“每个 env 独立跑完整 chunk”的语义。
6. `_handle_auto_reset()` 仍由父 wrapper 调用。RoboCasa/LIBERO 现在对 `ignore_terminations` 有特殊分支，拼回 raw done 后再套用当前逻辑，最不容易影响训练行为。
7. `RobocasaEnv.close()` 需要补充关闭所有 shard worker。`LiberoEnv` 如果后面新增 worker，也要同步加 close/cleanup；如果当前没有显式 close，需要新增一个对称清理函数。

注意事项：

- `ignore_terminations` 的逻辑要保持现状。
- `auto_reset` 仍然应在 chunk 结束后基于 `past_dones` 统一处理，或者在 local worker 内处理后把 `final_observation` / `final_info` 带回。
- `infos` 通常是嵌套 dict/list，拼接时要用现有 obs/info 工具，避免手写浅拷贝导致 batch 维错乱。
- RoboCasa/LIBERO 的 env 初始化通常比较重。不要在每次 `chunk_step()` 里临时创建 worker，必须在 `__init__()` 建好并复用。

### ManiSkill

ManiSkill 常见优势是 GPU vectorized simulation。当前“所有 env 同步推进一个 action”其实非常适合 GPU 并行。

如果强行改成每个 env 独立 worker：

- 可能降低 GPU vectorization 吞吐。
- 可能增加进程间通信和 CUDA context 开销。
- 多个进程同时持有 GPU simulator 可能不稳定，具体取决于 ManiSkill/SAPIEN 版本和渲染配置。

建议先支持 shard 级并发，而不是默认每 env 一个 worker：

```text
num_envs = 64, num_shards = 4
shard0: env 0..15
shard1: env 16..31
shard2: env 32..47
shard3: env 48..63
```

每个 shard 内仍使用 ManiSkill vectorized step。这样可以在“各 shard 独立跑 chunk”和“保留一定 GPU batch 吞吐”之间折中。

需要改的文件和函数：

```text
rlinf/envs/maniskill/maniskill_env.py
  ManiskillEnv.__init__(...)
  ManiskillEnv.step(...)
  ManiskillEnv.chunk_step(...)
  ManiskillEnv._handle_auto_reset(...)

rlinf/envs/utils/chunk_runner.py
  same common shard/scatter helpers

tests/unit_tests/test_chunk_step_parallel.py
  common logic tests
```

具体改法：

1. 在 `ManiskillEnv.__init__()` 里新增 `chunk_step_mode`、`chunk_step_num_shards` 的读取和校验。默认仍为 `sync_time_major`。
2. `ManiskillEnv.chunk_step()` 先加模式分支：默认保持当前实现；`parallel_shard` 时调用 shard runner。
3. shard runner 不建议默认每 env 一个进程。优先实现 `num_shards` 个 ManiSkill vector env，每个 shard 内部仍 vectorized step。
4. `ManiskillEnv.step()` 初期不改。ManiSkill 的普通 step 正是 GPU vectorized env 最擅长的路径。
5. `_handle_auto_reset()` 仍在父 wrapper 上基于全局 `past_dones` 调用，避免多个 CUDA/SAPIEN worker 各自 reset 后再拼 final obs 造成语义混乱。
6. 如果后续发现多进程 ManiSkill + GPU simulator 不稳定，可以只支持 `chunk_step_num_shards=1` 或只允许同进程 shard，并在 config validation 中报错/降级。

额外建议：

- ManiSkill 的并行改造要先做 benchmark，再决定是否开放给默认训练配置。
- 如果 env 使用 GPU tensor obs，scatter 工具必须保持 device，不要无意中 `.cpu()`。
- 如果多 shard 共享同一块 GPU，必须明确 CUDA context、渲染后端和资源释放策略。

## 配置和公共入口

四类 env 都需要相同的配置入口，建议集中改：

```text
rlinf/config.py
  validate_cfg(...)
  env train/eval config defaults if present

examples/embodiment/config/*.yaml
tests/e2e_tests/embodied/*.yaml
  only add explicit examples/tests when enabling parallel_shard
```

建议新增字段：

```yaml
env:
  train:
    chunk_step_mode: sync_time_major
    chunk_step_num_shards: 1
  eval:
    chunk_step_mode: sync_time_major
    chunk_step_num_shards: 1
```

`validate_cfg(...)` 中建议检查：

- `chunk_step_mode` 只能是 `sync_time_major` 或 `parallel_shard`。
- `chunk_step_num_shards >= 1`。
- `chunk_step_num_shards <= num_envs`。
- 对 ManiSkill 可以先限制 `parallel_shard` 必须显式开启，并对 GPU 多进程模式给 warning 或报错。

## Done / auto-reset 语义

现有逻辑大致是：

```python
past_terminations = raw_chunk_terminations.any(dim=1)
past_truncations = raw_chunk_truncations.any(dim=1)
past_dones = torch.logical_or(past_terminations, past_truncations)

if past_dones.any() and self.auto_reset:
    obs_list[-1], infos_list[-1] = self._handle_auto_reset(...)

chunk_terminations = torch.zeros_like(raw_chunk_terminations)
chunk_terminations[:, -1] = past_terminations

chunk_truncations = torch.zeros_like(raw_chunk_truncations)
chunk_truncations[:, -1] = past_truncations
```

并行改造后建议保留这个对外语义：

- worker 只返回 raw per-step terminations/truncations。
- 父 wrapper 拼成全局 raw tensor 后再计算 `past_dones`。
- 父 wrapper 统一生成对外的 `chunk_terminations` / `chunk_truncations`。

这样对 runner、replay buffer、advantage 计算的影响最小。

如果某个底层 env 必须在 worker 内 reset，则 worker 需要额外返回：

```text
final_observation
final_info
reset_observation
reset_info
done_mask
```

父 wrapper 再把这些字段合并进全局 `infos_list[-1]`。

## 结果拼接要求

拼接时必须保证 batch 维顺序等于原始 env id 顺序。建议流程：

```python
global_rewards = torch.empty(num_envs, chunk_size, ...)
for shard, result in zip(shards, results):
    global_rewards[shard.env_indices] = result.rewards
```

`obs_list` 要按时间维组织，不能按 shard 组织：

```python
global_obs_list = []
for t in range(chunk_size):
    obs_t = scatter_obs_by_env_id(
        [result.obs_list[t] for result in results],
        [shard.env_indices for shard in shards],
    )
    global_obs_list.append(obs_t)
```

`infos_list` 同理：

```python
global_infos_list = []
for t in range(chunk_size):
    infos_t = scatter_infos_by_env_id(...)
    global_infos_list.append(infos_t)
```

如果当前项目没有通用的 `scatter_obs_by_env_id()`，需要新增并为 dict / tensor / numpy / list 结构写单测。

## 配置建议

配置字段见上面的“配置和公共入口”。这里强调默认值和兼容性：

- `sync_time_major`: 当前行为，所有 env 同步走每个 chunk 时间步。
- `parallel_shard`: 新行为，不同 shard 并发执行自己的 chunk。
- `chunk_step_num_shards`: shard 数量。`1` 等价于当前单 worker 路径，`num_envs` 表示每 env 一个 worker。

默认值建议先保持 `sync_time_major`，不要直接改变已有训练配置的行为。只有在 config 显式设置 `parallel_shard` 时，才进入新执行路径。

## 测试计划

建议至少补以下测试：

1. 构造 dummy vector env，`num_envs=4`，`chunk_step=3`，记录每个 env 的 action 执行序列。
2. 验证 `parallel_shard` 下每个 env 收到的 action 顺序是 `[env_id, 0] -> [env_id, 1] -> [env_id, 2]`。
3. 验证返回形状和当前 `chunk_step()` 完全一致。
4. 验证 shard 结果乱序返回时，最终 batch 维仍按原始 env id 排列。
5. 验证某个 env 中途 done 时，`chunk_terminations[:, -1]` 的聚合语义不变。
6. 验证 `ignore_terminations=True` 时，RoboCasa / LIBERO 的行为不变。
7. 如果开启 `auto_reset`，验证 `final_observation`、`final_info` 和 `_final_observation` mask 没有丢。

## 建议实施顺序

1. 先写 dummy env 单测，固定目标语义。
2. 新增通用 shard runner 和 obs/info scatter 工具。
3. 先改 Behavior，因为它已有子进程 RPC。
4. 再改 RoboCasa / LIBERO，二者共用同一套 shard runner。
5. 最后评估 ManiSkill 是否启用 `parallel_shard`，并用性能测试决定默认 shard 大小。
6. 加配置开关，默认保持旧行为。
7. 跑现有 e2e，确认旧行为未回归。

## 最小化改动方案

如果四个 env 都要改，但不想动太多公共代码，建议用这个版本。

一句话：四个 env 都加一个 `chunk_step_num_shards` 开关，默认是 `1`，所以旧行为完全不变；只有手动设成 `>1` 时，才走新的并发 chunk 路径。

这个版本先不做大抽象：

- 不新增通用 `chunk_runner.py`。
- 不改 `step()`。
- 不改 `_handle_auto_reset()`。
- 不改默认行为。
- 不新增公共 worker。
- 不新增公共 obs/info 拼接工具。
- 每个 env 文件自己处理自己的 shard 和结果拼接。

### 四个 env 共用的最小代码形状

四个 env 都用同一个简单模式：

```python
def chunk_step(self, chunk_actions):
    if getattr(self, "chunk_step_num_shards", 1) <= 1:
        # Old code path. Keep the current implementation unchanged.
        ...

    # New code path. Only runs when explicitly enabled.
    ...
```

也就是说，不需要第一版就改 `rlinf/config.py`。可以先在 env 里直接读：

```python
self.chunk_step_num_shards = getattr(cfg, "chunk_step_num_shards", 1)
```

后面确认要正式支持，再补 config 默认值和校验。

### 每个 env 改什么

#### Behavior

改文件：

```text
rlinf/envs/behavior/behavior_env.py
```

怎么改：

- `__init__()` 里读 `chunk_step_num_shards`。
- `chunk_step_num_shards == 1` 时保持旧逻辑。
- `chunk_step_num_shards > 1` 时启动多个已有的 `_behavior_worker()`。
- 每个 worker 只管一部分 env。
- `chunk_step()` 里把 `chunk_actions` 按 env 分给 worker。
- 收回来以后按原 env 顺序拼回去。
- `close()` 里关闭所有 worker。

Behavior 最好改，因为它已经有子进程 worker。

#### RoboCasa

改文件：

```
rlinf/envs/robocasa/robocasa_env.py
```

怎么改：

- `__init__()` 里读 `chunk_step_num_shards`。
- `chunk_step_num_shards == 1` 时保持旧逻辑。
- 在 `robocasa_env.py` 里新增一个本地 worker，比如 `_robocasa_chunk_worker()`。
- `chunk_step_num_shards > 1` 时启动多个 RoboCasa worker。
- 每个 worker 创建自己的 `RobocasaEnv(local_num_envs)`。
- 子 worker 内部强制 `chunk_step_num_shards = 1`，避免递归创建 worker。
- 父 env 负责切 action、发 worker、收结果、拼结果。
- 当前 `ignore_terminations` 和 `_handle_auto_reset()` 逻辑继续放在父 env 里处理。

#### LIBERO

改文件：

```text
rlinf/envs/libero/libero_env.py
```

怎么改：

- 和 RoboCasa 一样。
- 在 `libero_env.py` 里新增本地 worker，比如 `_libero_chunk_worker()`。
- `chunk_step_num_shards == 1` 时保持旧逻辑。
- `chunk_step_num_shards > 1` 时多个 worker 并发跑自己的 action chunk。
- 父 env 负责拼回全局结果。
- `ignore_terminations` 和 `_handle_auto_reset()` 继续放在父 env 里。

#### ManiSkill

改文件：

```
rlinf/envs/maniskill/maniskill_env.py
```

怎么改：

- `__init__()` 里读 `chunk_step_num_shards`。
- `chunk_step_num_shards == 1` 时保持旧逻辑。
- 不建议第一版就做多进程 ManiSkill，因为 ManiSkill 常用 GPU vectorized simulation，多进程可能不稳定。
- 最小安全做法：`chunk_step_num_shards > 1` 时先明确报错，告诉用户 ManiSkill 需要单独的 GPU shard 设计。
- 如果一定要支持，也只建议做 shard 级 vector env，不建议每 env 一个进程。

### 为什么不能只改循环顺序

这个不能这么写：

```python
for env_id in range(num_envs):
    for i in range(chunk_size):
        self.step(...)
```

这会继续推进整个 vector env，而不是只推进某一个 env。结果要么行为错误，要么需要构造假 action 填给其他 env，训练语义会变得不清楚。

### 最小方案总结

最小版本改动如下：

- 改 4 个 env 文件。
- 不新增公共 runtime 文件。
- 第一版可以不改 `rlinf/config.py`。
- 四个 env 都加 `chunk_step_num_shards`。
- 默认 `chunk_step_num_shards = 1`，旧逻辑不变。
- Behavior / RoboCasa / LIBERO 真实支持 `>1`。
- ManiSkill 先只加入口和保护，`>1` 暂时显式报错，后面单独设计。

推荐顺序：

1. 先改 Behavior。
2. 再改 RoboCasa。
3. 再改 LIBERO。
4. 最后改 ManiSkill 的入口和保护。
5. 跑通后再考虑是否抽公共工具。

## 风险

- 对 GPU vectorized env，per-env 并行不一定更快，可能明显更慢。
- 多进程 env 会增加内存、显存、初始化时间和序列化开销。
- obs/info 结构复杂，拼接工具写错会造成静默数据错位。
- auto-reset 如果在 worker 和 parent 两边都处理，容易重复 reset。
- 并行执行会改变 env 间相对推进时机，可能影响依赖全局同步时钟的底层 simulator。
