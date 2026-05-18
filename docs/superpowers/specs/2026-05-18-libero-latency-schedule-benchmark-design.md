# LIBERO Latency Schedule Benchmark Design

## Purpose

This toolkit benchmark validates whether latency-aware task placement can reduce
long-tail `env.step()` stalls in multi-environment LIBERO simulation. It compares
two static baselines against a proposed trapezoid pipeline schedule using real
LIBERO `env.step()` execution.

The benchmark reads tasks from an existing profiling CSV. The CSV provides the
task pool and structural features such as `njnt` and `ngeom`. The benchmark uses
those features only to estimate relative step latency for scheduling. It measures
actual benefit from real concurrent environment steps.

## Scope

The implementation adds a standalone toolkit script, expected at:

`toolkits/run_libero_latency_schedule_benchmark.py`

It does not change RLinf runtime scheduling, workers, Ray placement, or training
logic. It is an experimental benchmark for measuring the potential benefit of a
stable latency-aware schedule.

## Inputs

The primary input is a profiling CSV with one row per task. It must include:

- `task_id`
- `task_name`
- `mean_latency_ms`
- `njnt`
- `ngeom`

Optional columns such as `scene_type`, `scene_name`, and `task_language` are
preserved in outputs when present.

Each run samples tasks without replacement from this CSV:

- `--num-envs` selects the task count for one run, such as 32 or 64.
- `--seed` controls deterministic sampling.
- The command runs one selected `num_envs` value at a time.

The benchmark also accepts normal LIBERO execution settings such as suite,
trial selection, camera size, warmup steps, dummy action, CPU core list, and
steps per environment.

## Latency Estimation

The proposed schedule estimates relative task latency from `njnt` and `ngeom`.
The default estimator should be deterministic and simple:

```text
score = w_jnt * z(njnt) + w_geom * z(ngeom)
```

The default weights are:

- `w_jnt = 0.45`
- `w_geom = 0.55`

`ngeom` receives slightly higher weight because the prior LIBERO-90 profiling
showed stronger correlation between `ngeom` and measured step latency.

Sorting is stable and reproducible:

1. Estimated latency score descending.
2. `task_id` ascending as the tie-breaker.

The measured `mean_latency_ms` from the CSV is not used to score the proposed
schedule by default. It may be reported for analysis and sanity checks.

## Schedules

Each schedule receives the same sampled tasks, CPU cores, trial/init states,
dummy action, camera configuration, warmup steps, and `steps_per_env`.

### Task ID Baseline

`task_id_baseline` sorts sampled tasks by `task_id` ascending. Tasks are then
assigned statically to core columns in round-robin layer order. Workers execute
only the tasks assigned to their core.

### Random Baseline

`random_baseline` shuffles the sampled tasks using a deterministic baseline seed.
Tasks are assigned with the same static layer order as the task ID baseline.

The CLI should allow repeated random baselines. Summaries report per-repeat
results plus aggregate statistics when repeats are used.

### Trapezoid Pipeline

`trapezoid_pipeline` sorts sampled tasks by estimated latency descending, then
splits the sorted list into two halves:

- The first half is the long trapezoid.
- The second half is the short trapezoid.

The long half is filled into core columns in descending latency order. The short
half is filled so that corresponding positions map to the same core columns.
Runtime executes the long side in forward order and the short side in reverse
order. This makes the shortest group at the tail of the long trapezoid connect
to the longest group at the head of the short trapezoid, reducing pipeline
bubble while preserving a fixed core mapping.

Each worker only owns the environments assigned to its core. Long and short
tasks mapped to the same core stay on that core for the full schedule. There is
no runtime migration or dynamic reassignment.

## Runtime Execution Model

The benchmark uses real concurrent execution:

1. The main process builds a schedule plan for each schedule.
2. It starts one worker process per configured CPU core.
3. Each worker applies CPU affinity with `os.sched_setaffinity` when available.
4. Each worker creates and holds its assigned LIBERO environments.
5. Environment creation, reset, init-state setup, and warmup are excluded from
   measured makespan.
6. The measured phase uses synchronized rounds controlled by the main process.
7. In each round, workers perform real `env.step(dummy_action)` on their current
   assigned environment and return measured latency.
8. The main process records round wall time as the slowest worker completion.
9. Other workers' waiting time is counted as bubble/idle.

The fair execution target is `--steps-per-env`. Every sampled task must complete
the same number of measured steps for every schedule. For example, 64 tasks and
100 steps per environment means each schedule completes 6400 measured steps.

## Metrics

For each schedule, the benchmark reports:

- `status`
- total measured steps
- measured makespan
- steps per second
- mean step latency
- median step latency
- p90, p95, and p99 step latency
- mean core idle ratio
- p90 and p99 round idle ratio
- CPU affinity success rate
- speedup versus baselines
- bubble reduction versus baselines

Random baseline repeats should be reported individually and as aggregate
statistics.

## Outputs

The benchmark writes one output directory per run. Expected files:

- `run_config.json`
- `selected_tasks.csv`
- `schedule_plan_<schedule_name>.csv`
- `step_events_<schedule_name>.jsonl`
- `schedule_summary.csv`
- `schedule_summary.json`
- `comparison_report.md`
- `errors.jsonl` when warnings or failures occur

`selected_tasks.csv` records sampled tasks and estimated latency scores.

`schedule_plan_<schedule_name>.csv` records the core mapping, schedule order,
task IDs, task names, and whether each task belongs to the long or short side
for the trapezoid schedule.

`step_events_<schedule_name>.jsonl` records measured runtime events with fields
such as schedule name, round index, worker/core ID, task ID, step index for that
task, latency, round wall time, idle time, and CPU affinity status.

`comparison_report.md` summarizes the experiment in human-readable form and
includes the mapping evidence needed to verify that corresponding long and short
tasks stayed on the same core.

## Example Commands

Unit-test smoke mode:

```bash
python toolkits/run_libero_latency_schedule_benchmark.py \
  --task-csv results/libero90_step_latency_all_tasks_10steps/task_latency_ranked.csv \
  --num-envs 32 \
  --cpu-ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --steps-per-env 10 \
  --output-dir results/libero90_latency_schedule_smoke \
  --fake-latency-from-csv
```

Real LIBERO benchmark:

```bash
python toolkits/run_libero_latency_schedule_benchmark.py \
  --task-csv results/libero90_step_latency_all_tasks_10steps/task_latency_ranked.csv \
  --num-envs 64 \
  --cpu-ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
  --steps-per-env 100 \
  --warmup-steps 20 \
  --output-dir results/libero90_latency_schedule_real
```

## Error Handling

Worker initialization failure, environment creation failure, step exceptions,
IPC failures, or timeouts fail the current schedule and write structured records
to `errors.jsonl`.

CPU affinity failure marks a schedule as degraded but does not automatically
stop the run. The summary and report must make the degraded status visible
because affinity failure affects the credibility of core-utilization results.

Schedules run independently. A failed schedule must not corrupt output from
completed schedules.

## Testing

Default unit tests should not require LIBERO, MuJoCo, GPUs, or display access.
They should use fake task records and fake worker/environment hooks to verify:

- deterministic CSV loading and sampling
- `njnt`/`ngeom` estimator behavior
- stable tie-breaking by `task_id`
- task ID baseline plan construction
- seeded random baseline plan construction
- trapezoid long/short split and fixed core mapping
- per-task measured step counts
- summary metrics including throughput and idle ratio
- degraded status when CPU affinity is reported as failed
- failed status and error records when a worker reports an exception

Real LIBERO execution is a manual benchmark path and should not be part of the
default CI test suite.

## Non-Goals

This benchmark does not implement dynamic runtime task reordering, online
latency learning, Ray integration, training-loop integration, or RL policy
execution. It also does not attempt to prove final training throughput. Its goal
is to measure whether the proposed stable task placement reduces simulator
long-tail bubbles under controlled concurrent step execution.
