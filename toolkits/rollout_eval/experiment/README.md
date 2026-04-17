# Rollout Eval Experiment Pipeline

渐进式 3 阶段实验管线，用于在 `rollout_eval` 基础上进行固定 seed 基线采集、FeatureCache 性能评估和瓶颈区动作替换验证。

## 快速开始

```bash
# 无缓存基线 + 动作替换管线
python -m toolkits.rollout_eval.experiment.run_experiment \
    --config-path examples/embodiment/config \
    --config-name libero_spatial_ppo_gr00t \
    --phases baseline,action_replace \
    --seeds 42,43,44 \
    --num-runs-per-seed 3 \
    --cache-mode similarity_gated \
    --bottleneck-k-b auto \
    --record-video \
    --output-dir ./experiment_output
```

## 三个阶段

### Phase 1: Baseline Collection

使用固定 env seed 多次运行 eval，验证确定性并采集基线轨迹。

```bash
python -m toolkits.rollout_eval.experiment.run_experiment \
    --config-path examples/embodiment/config \
    --config-name libero_spatial_ppo_gr00t \
    --phases baseline \
    --seeds 42,43 \
    --num-runs-per-seed 3 \
    --record-video \
    --output-dir ./experiment_output
```

输出:
- `reports/phase1_baseline.json` — 确定性验证结果、每 seed 成功率/奖励/步数统计
- `trajectories/seed{N}_run{M}.pkl` — 保存的轨迹数据（供后续阶段使用）
- `videos/phase1_seed{N}_run{M}.mp4` — 仿真视频

### Phase 2: Feature Cache Evaluation

在固定 seed 下测试 FeatureCache 的性能表现。采用两遍法：第一遍填充缓存（全 miss），第二遍测量命中率、延迟节省和动作偏差。

```bash
python -m toolkits.rollout_eval.experiment.run_experiment \
    --config-path examples/embodiment/config \
    --config-name libero_spatial_ppo_gr00t_feature_cache \
    --phases cache_eval \
    --seeds 42 \
    --cache-mode naive \
    --output-dir ./experiment_output
```

支持的 cache mode: `naive`, `similarity_gated`, `cross_step_naive`, `cross_step_similarity`, `cross_global_same_step`

输出:
- `reports/phase2_cache_eval.json` — 命中率、延迟节省百分比、动作 L2 偏差

### Phase 3: Action Replacement

在瓶颈区（step >= T - K_B）替换动作，验证复用瓶颈的可行性。

```bash
# 使用 Phase 1 的基线轨迹作为替换来源
python -m toolkits.rollout_eval.experiment.run_experiment \
    --config-path examples/embodiment/config \
    --config-name libero_spatial_ppo_gr00t \
    --phases baseline,action_replace \
    --seeds 42,43 \
    --bottleneck-k-b auto \
    --output-dir ./experiment_output

# 使用外部 pkl 轨迹
python -m toolkits.rollout_eval.experiment.run_experiment \
    --config-path examples/embodiment/config \
    --config-name maniskill_ppo_openvlaoft \
    --phases action_replace \
    --action-source external \
    --external-trajectory-dir /path/to/collected_data/ \
    --external-trajectory-seeds 42,43 \
    --bottleneck-k-b 8 \
    --output-dir ./experiment_output
```

K_B 检测方式:
- `--bottleneck-k-b <int>` — 静态指定
- `--bottleneck-k-b auto` — 从 Phase 1 轨迹自动检测（反向对齐 L2 散度算法）

动作来源 (`--action-source`):
- `pipeline` — 使用同次管线 Phase 1 采集的基线轨迹
- `cross_run` — 使用其他运行保存的轨迹
- `external` — 加载外部 pkl 文件（训练数据采集产生的轨迹）

输出:
- `reports/phase3_action_replace.json` — 替换区 L2 偏差、成功率变化、奖励差异

## CLI 参数一览

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config-path` | Hydra 配置目录 | (必填) |
| `--config-name` | Hydra 配置名 | (必填) |
| `--override` | Hydra 覆盖参数 | `[]` |
| `--phases` | 逗号分隔的阶段列表 | `baseline` |
| `--seeds` | 逗号分隔的 seed 列表 | `42` |
| `--num-runs-per-seed` | 每个 seed 重复运行次数 | `2` |
| `--cache-mode` | Phase 2 缓存模式 | `None` |
| `--bottleneck-k-b` | 静态 K_B 值或 `auto` | `None` |
| `--action-source` | 动作替换来源 | `pipeline` |
| `--external-trajectory-dir` | 外部轨迹 pkl 目录 | `None` |
| `--external-trajectory-seeds` | 外部轨迹 seed 过滤 | `None` |
| `--record-video` | 录制仿真视频 | `False` |
| `--output-dir` | 输出目录 | `./experiment_output` |

## 输出目录结构

```
experiment_output/
├── experiment_config.json        # 完整配置快照
├── reports/
│   ├── phase1_baseline.json
│   ├── phase2_cache_eval.json
│   └── phase3_action_replace.json
├── trajectories/
│   ├── seed42_run0.pkl
│   └── seed42_run1.pkl
└── videos/
    ├── phase1_seed42_run0.mp4
    └── phase3_replace_pipeline_seed42.mp4
```

## 外部轨迹 pkl 格式

文件命名: `step_{step}_sid_{sid}_rank_{rank}_env_{env}_episode_{episode}_{success|fail}.pkl`

pkl 内容:
```python
{
    "actions": list[torch.Tensor],     # T 个 [action_dim] 张量
    "observations": list[dict],        # (可选) 观测序列
    "rewards": list[float],            # (可选) 奖励序列
    "success": bool,                   # (可选, 可从文件名推断)
}
```

## 模块结构

```
toolkits/rollout_eval/experiment/
├── __init__.py
├── run_experiment.py          # 管线编排器 + CLI
├── types.py                   # 数据类型定义
├── seedable_env_adapter.py    # 固定 seed 环境适配器
├── recording_loop.py          # 带轨迹录制的 rollout 循环
├── cache_eval.py              # CacheAwareModelAdapter + no-cache baseline 下的占位/兼容逻辑
├── action_replacer.py         # ActionReplacer + OpenLoopReplay 适配器
├── trajectory_loader.py       # 外部 pkl 轨迹加载
├── bottleneck_detector.py     # K_B 瓶颈检测
├── determinism.py             # 确定性验证
└── reporting.py               # 各阶段报告生成
```

## 设计原则

- 包装模式: 所有新适配器包装现有 `GenericEnvAdapter` / `GenericModelAdapter`，不修改已有代码
- 组合优于继承: `ExperimentConfig` 包含 `EvalRuntimeConfig` 而非继承
- 复用生产代码: Phase 2 直接使用 `rlinf/models/embodiment/feature_cache.py` 中的 `FeatureCache`
- 阶段独立: 每个阶段可单独运行，也可作为完整管线的一部分

## 测试

```bash
# 运行所有实验管线测试
python -m pytest tests/unit_tests/test_experiment_*.py -v
```

设计文档: `docs/superpowers/specs/2026-03-31-rollout-eval-experiment-pipeline-design.md`
