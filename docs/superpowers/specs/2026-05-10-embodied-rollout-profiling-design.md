# Embodied Rollout Profiling Design

Date: 2026-05-10

## Goal

Add optional profiling for embodied RL rollout execution. The profiling should answer two questions:

1. In multi-environment rollout, whether environment observations are batched before one parallel VLA or policy generation call, and whether slow environments create a tail-latency effect.
2. During training, how many environment steps each task needs before its first success in each rollout epoch. If a task never succeeds in the epoch, record the maximum rollout step count instead.

The feature must cover both synchronous embodied runners and async embodied runners. It must be disabled by default.

## Current Rollout Flow

The synchronous and async embodied paths share the same core worker methods:

- `EnvWorker._run_interact_once()`
- `EnvWorker.env_interact_step()`
- `MultiStepRolloutWorker.generate_one_epoch()`
- `MultiStepRolloutWorker.recv_env_output()`
- `MultiStepRolloutWorker.predict()`

The observed flow is:

1. The env worker resets or bootstraps and sends current observations to the env channel.
2. The rollout worker receives observations from one or more mapped env ranks.
3. The rollout worker merges those observations into one batch.
4. The rollout worker calls `hf_model.predict_action_batch(env_obs=...)`.
5. The rollout worker splits the action batch back by env rank.
6. The env worker receives rollout results and calls `env.chunk_step(chunk_actions)`.

Therefore the user's hypothesis is correct for this path: within each rollout rank's mapped input set, model generation waits for the mapped environment observations, then computes actions as one batch. Async runners keep env and rollout workers running continuously, but each obs-to-action exchange still has this local synchronization pattern.

## Configuration

Add a default-off config switch under rollout:

```yaml
rollout:
  profiling:
    enabled: false
```

The initial design uses one switch for both env timing and task step records. More granular switches can be added later if needed.

## Output Files

Write structured files under the experiment directory:

```text
<runner.logger.log_path>/<runner.logger.experiment_name>/profiling/
```

Use per-worker files to avoid concurrent writes from multiple Ray actors:

```text
embodied_rollout_events_rank_<rank>.jsonl
task_step_summary_rank_<rank>.jsonl
```

JSONL is preferred because it supports incremental writes during long training runs and is easy to post-process.

## Event Records

### `env_chunk_timing`

Record one event for each `EnvWorker.env_interact_step()` call.

Required fields:

- `event`: `"env_chunk_timing"`
- `global_step`
- `rollout_epoch`
- `chunk_index`
- `rank`
- `stage_id`
- `env_type`
- `batch_size`
- `chunk_size`
- `wall_time_s`
- `timing_granularity`: `"batch"` or `"subenv"`
- `done_count`
- `success_count`
- `truncation_count`

For vectorized environments such as ManiSkill, batch-level timing is the supported generic signal. For subprocess-backed environments, optional per-subenv timing may be added as separate records.

### `subenv_step_timing`

Record only where the environment backend naturally exposes per-subenv execution timing, such as subprocess vector env implementations.

Required fields:

- `event`: `"subenv_step_timing"`
- `global_step`
- `rollout_epoch`
- `chunk_index`
- `rank`
- `stage_id`
- `local_env_index`
- `global_env_id`
- `step_index`
- `wall_time_s`
- `env_type`
- `task_id`
- `task_name`
- `task_type`

If the backend cannot provide meaningful per-subenv timing, do not fake it. Emit batch timing instead and set `timing_granularity` to `"batch"`.

### `task_episode_steps`

At the end of each rollout epoch, record one event per local env.

Required fields:

- `event`: `"task_episode_steps"`
- `global_step`
- `rollout_epoch`
- `rank`
- `stage_id`
- `local_env_index`
- `global_env_id`
- `env_type`
- `task_id`
- `task_name`
- `task_description`
- `task_type`
- `success`
- `first_success_step`
- `max_steps`
- `recorded_step`

`recorded_step` is `first_success_step` for successful tasks and `max_steps` for unsuccessful tasks.

The step counter is based on actual environment action steps inside each chunk. If `num_action_chunks > 1`, success at the second action inside the third chunk must be recorded as the corresponding real step, not only the chunk index.

This record must be computed from termination or success signals before the loss mask removes post-success steps. The mask should not affect profiling results.

## Task Metadata

Use a best-effort metadata extractor that does not require all environments to implement the same interface immediately.

Read from these common sources when present:

- `task_ids`
- `task_names`
- `task_descriptions`
- `current_task`
- `env_names`
- observation field `task_descriptions`

Missing values should be written as `null`. Missing task metadata must not stop training.

`global_env_id` should be derived from rank, stage, and local env index so records remain stable across distributed workers.

## Aggregation

Write per-rank task summaries to:

```text
task_step_summary_rank_<rank>.jsonl
```

Append one summary record per global step, rollout epoch, and task identity.

Required summary fields:

- `global_step`
- `rollout_epoch`
- `rank`
- `env_type`
- `task_id`
- `task_name`
- `task_type`
- `count`
- `success_count`
- `success_rate`
- `mean_recorded_step`
- `mean_first_success_step`
- `p50_recorded_step`
- `p95_recorded_step`

`mean_first_success_step` should only include successful samples. `mean_recorded_step` includes failures using `max_steps`.

## Implementation Boundaries

Add a small profiling utility module, likely `rlinf/utils/embodied_rollout_profiler.py`, responsible for:

- Reading profiling config.
- Building output paths.
- Writing JSONL events.
- Tracking per-rank, per-stage, per-rollout-epoch state.
- Updating first-success step from chunk-level termination or success tensors.
- Extracting task metadata best-effort.
- Writing per-task summaries.

Keep the main integration in `EnvWorker`, because sync and async paths share `EnvWorker._run_interact_once()`.

Rollout-side timing should stay optional and small. Existing worker timers already expose coarse `predict` timing; the new profiling can add `model_predict_timing` later if needed, but the initial scope focuses on environment timing and task completion steps.

## Compatibility

The feature must not alter rollout behavior, loss masks, rewards, dones, reset logic, or actor training data.

When profiling is disabled, overhead should be limited to a boolean check.

When profiling is enabled:

- JSONL writes should be append-only.
- Each Ray actor writes its own file to avoid write contention.
- File IO failures should be logged with worker logging and should not silently corrupt training state.

## Testing

Add focused tests:

1. Profiler unit tests:
   - Disabled profiler writes no files.
   - Enabled profiler writes valid JSONL.
   - Per-rank file names do not collide.
   - Summary aggregation handles success and failure correctly.

2. EnvWorker/profiler logic tests:
   - Simulate multiple envs and multiple action chunks.
   - Verify first success inside a chunk maps to the correct real step.
   - Verify failed tasks record `max_steps`.
   - Verify missing task metadata writes `null` instead of raising.

3. Regression tests:
   - Existing embodied tests should pass with profiling disabled.
   - If heavyweight simulation dependencies are unavailable locally, run the new unit tests and document skipped integration coverage.

## Non-Goals

- Do not force per-subenv timing for GPU vectorized envs where it is not meaningful.
- Do not add token-level or model-internal generation profiling.
- Do not change training semantics, masking, or rollout scheduling.
- Do not require all envs to implement a new metadata interface in the first pass.
