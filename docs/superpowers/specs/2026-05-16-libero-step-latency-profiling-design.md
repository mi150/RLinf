# LIBERO Step Latency Profiling Design

Date: 2026-05-16

## Goal

Build an independent profiling tool that demonstrates step-level tail latency in
embodied RL simulators, using LIBERO as the first target. The tool should measure
how long a single simulator `env.step()` takes for different LIBERO tasks and
trials, while minimizing non-simulator interference.

The first implementation must not start Ray, load a policy model, or enter the
RLinf training loop. It should produce structured records that connect latency
statistics to task, scene, object, texture, rendering, and MuJoCo model metadata.

## Motivation

RLinf's training stack batches environment observations and model actions, so a
slow simulator step can become a tail-latency source for rollout throughput. This
experiment isolates the simulator itself. By binding each measured environment
process to one CPU and avoiding model/Ray overhead, the output should provide a
clean motivation artifact for the claim that different embodied tasks and scenes
have different per-step costs.

## Recommended Approach

Add a standalone toolkit script:

```text
toolkits/profile_libero_step_latency.py
```

The script creates LIBERO `OffScreenRenderEnv` instances directly, applies a
fixed dummy action, and measures warmup-excluded step latency with
`time.perf_counter()`.

Default execution is sequential and isolated:

1. Select a LIBERO suite.
2. Enumerate selected task IDs and trial IDs.
3. Spawn one subprocess for one task/trial profile run.
4. Bind the subprocess to the configured CPU with `os.sched_setaffinity` when
   available.
5. Create one `OffScreenRenderEnv`.
6. Reset it to the selected initial state.
7. Run warmup steps that are excluded from statistics.
8. Run measured steps and write per-step JSONL events.
9. Write one summary row per task/trial.

This design intentionally favors clean attribution over throughput. A future
parallel mode can run multiple CPU-bound subprocesses at once, but the initial
tool should make the single-process path the default.

## CLI

Example:

```bash
python toolkits/profile_libero_step_latency.py \
  --suite libero_90 \
  --task-ids all \
  --trials-per-task 3 \
  --warmup-steps 20 \
  --measure-steps 200 \
  --cpu-id 0 \
  --output-dir results/libero_step_latency
```

Required or primary options:

- `--suite`: LIBERO suite name, such as `libero_spatial`, `libero_object`,
  `libero_goal`, `libero_10`, `libero_90`, or `libero_130`.
- `--task-ids`: `all` or a comma-separated list such as `0,3,7`.
- `--trials-per-task`: number of initial states to profile for each selected
  task when explicit trial IDs are not provided.
- `--specific-trial-ids`: optional comma-separated trial IDs for reproducible
  selection.
- `--warmup-steps`: number of unmeasured steps before recording latency.
- `--measure-steps`: number of measured steps per task/trial.
- `--cpu-id`: CPU used for sequential isolated profiling.
- `--cpu-ids`: optional future-compatible CPU list. The initial implementation
  may accept it as an alias but should still run sequentially unless parallel
  mode is explicitly added later.
- `--camera-height` and `--camera-width`: LIBERO camera resolution.
- `--libero-type`: `standard`, `pro`, or `plus`; default is `standard`.
- `--seed`: base seed for task/trial selection and environment seeding.
- `--output-dir`: output directory for JSONL and summary files.
- `--dummy-action`: optional comma-separated action override. The default is
  seven dimensions with zeros and gripper value `-1`.
- `--stop-on-done`: optional flag. Default is false so each task/trial records
  the same number of measured steps. When false, the tool keeps stepping after
  the first done signal and records `done_seen_step`.

## Output Files

Write files under `--output-dir`:

```text
step_latency_events.jsonl
step_latency_summary.csv
step_latency_summary.json
errors.jsonl
run_config.json
```

`step_latency_events.jsonl` contains one measured step per line. `errors.jsonl`
contains task/trial failures. Summary files contain one row/object per
task/trial.

## Per-Step Event Schema

Each measured step record should include:

- `event`: `"libero_step_latency"`.
- `suite_name`.
- `task_id`.
- `trial_id`.
- `task_name`.
- `task_language`.
- `step_index`.
- `latency_s`.
- `reward`.
- `done`.
- `success`.
- `done_seen_step`.
- `cpu_id`.
- `pid`.
- `seed`.
- `cpu_affinity_applied`.
- `bddl_file`.
- `scene_type`.
- `scene_name`.
- `camera_names`.
- `camera_heights`.
- `camera_widths`.
- `renderer`.
- `num_objects`.
- `object_categories`.
- `num_fixtures`.
- `fixture_categories`.
- `num_regions`.
- `num_obj_of_interest`.
- `num_init_predicates`.
- `num_goal_predicates`.
- `nbody`.
- `ngeom`.
- `njnt`.
- `nq`.
- `nv`.
- `nu`.
- `ncam`.

Fields that cannot be collected should be written as `null`, not omitted, so
downstream analysis can distinguish unavailable metadata from schema changes.

## Summary Schema

Each task/trial summary should include all stable task and complexity metadata
plus:

- `step_count`.
- `mean_latency_s`.
- `median_latency_s`.
- `p90_latency_s`.
- `p95_latency_s`.
- `p99_latency_s`.
- `min_latency_s`.
- `max_latency_s`.
- `std_latency_s`.
- `tail_ratio_p99_to_median`.
- `done_seen_step`.
- `success_seen`.
- `error`.

If a task/trial fails before measurement starts, it should not appear in the
summary as a successful measurement. It should appear in `errors.jsonl` with the
same task identity fields and a concise error message.

## LIBERO Complexity Metadata

The tool should collect two categories of metadata.

### Static BDDL Metadata

Parse the selected BDDL file and record:

- Problem name and domain.
- Language instruction.
- Scene name and scene type inferred from the problem name or BDDL path, such
  as `kitchen`, `table`, `living_room`, `study`, `coffee_table`, or `floor`.
- Region count and region names.
- Fixture count and fixture categories.
- Movable object count and object categories.
- Object-of-interest count and names.
- Initial predicate count and predicate names.
- Goal predicate count and predicate names.

These fields are the primary explanatory variables for task and scene
complexity.

### Runtime Environment Metadata

After the environment is constructed, collect best-effort metadata from the
actual robosuite/MuJoCo model:

- `sim.model.nbody`, `ngeom`, `njnt`, `nq`, `nv`, `nu`, and `ncam`.
- Camera names and resolution.
- Renderer name and offscreen renderer configuration.
- `scene_xml` and `scene_properties` when available.
- Floor and wall style when available.
- Texture/material counts if they are available through the model or parsed XML.

Texture metadata is best-effort. Standard LIBERO problem classes define default
scene properties, such as floor and wall style, but users may override them
through environment init parameters. The profiler should prefer actual init
parameters and runtime attributes when available, then fall back to static
inference.

The tool should not fail if texture or MuJoCo metadata cannot be read.

## Measurement Semantics

The measurement target is one call to `env.step(action)` in a single LIBERO
environment. Timing includes MuJoCo simulation, reward/success checks,
observation generation, and offscreen rendering configured by LIBERO. Timing
does not include Ray scheduling, RLinf worker communication, model inference, or
action batching.

Warmup steps are excluded from summary statistics. Measured steps use a fixed
dummy action unless the user overrides it. The default dummy action is designed
to keep the robot mostly stationary while opening the gripper, matching the
LIBERO action dimension used by existing evaluation code.

By default, the profiler records a fixed number of measured steps even if the
environment reports done. This keeps step counts comparable across tasks.
`done_seen_step` and `success_seen` preserve episode outcome information.

## Error Handling

- If LIBERO cannot be imported or the suite cannot be built, fail at startup
  with an actionable message.
- If one task/trial fails during construction, reset, or step, write an error
  event and continue to the next task/trial.
- If CPU affinity is unavailable or rejected by the OS, continue profiling and
  mark `cpu_affinity_applied: false`.
- If BDDL metadata parsing fails for one task, continue with `null` metadata and
  include a warning field in the error log.
- If MuJoCo runtime metadata cannot be read, continue with `null` values.
- Close each environment in a `finally` block inside the worker subprocess.

## Tests

Add focused unit tests that do not require a real LIBERO/MuJoCo installation:

- BDDL parsing from a small fixture string or fixture file.
- Task ID parsing for `all` and comma-separated lists.
- Trial selection with `--trials-per-task` and `--specific-trial-ids`.
- Summary statistics, including percentile and `p99 / median` tail ratio.
- Error record formatting.
- A smoke path with a mock environment that returns deterministic step results.

Real LIBERO profiling should be documented as a manual command, not required in
CI.

## Out of Scope

- Ray worker or RLinf training-loop integration.
- Policy/model inference profiling.
- Parallel multi-process profiling as the default mode.
- GPU profiling.
- Full OBJ/MTL/PNG asset graph analysis. The initial tool records texture and
  material metadata only when it is available through simple static or runtime
  inspection.

## Success Criteria

The implementation is complete when:

1. A user can run the standalone script against a LIBERO suite and produce
   per-step JSONL plus task/trial summary files.
2. Each measured row contains task identity, step latency, CPU affinity status,
   BDDL metadata, render configuration, and best-effort MuJoCo model metadata.
3. Failures in individual task/trial runs are isolated and written to
   `errors.jsonl`.
4. Unit tests cover parsing, summary statistics, CLI selection logic, and mock
   profiling behavior.
5. The script remains independent of Ray and policy models.
