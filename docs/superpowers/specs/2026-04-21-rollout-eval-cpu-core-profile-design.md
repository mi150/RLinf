# Rollout Eval CPU Core Profile Design

## Overview

This document defines a benchmark-only extension for `toolkits/rollout_eval/benchmark` to profile environment throughput under fine-grained CPU core allocation. The current benchmark supports GPU-side resource settings via MPS and MIG. This design adds a third resource setting, CPU core binding, focused on the environment side only.

The goal is to benchmark how environment throughput changes when CPU cores are partitioned across logical evaluation environments. For example, with `cfg.env.eval.total_num_envs = 32` and an available CPU pool of 128 cores, the benchmark should be able to assign 4 non-overlapping cores to each logical environment by default, then profile throughput under that setting.

This feature is intentionally scoped to the benchmark toolkit. It does not change RLinf training, scheduler placement, or environment runtime behavior outside `toolkits/rollout_eval/benchmark`.

## Goals

- Add a CPU core resource profile to rollout eval benchmark cases.
- Support automatic even-split CPU core assignment from CLI input.
- Treat `cfg.env.eval.total_num_envs` as the number of logical environments used for partitioning.
- Apply CPU affinity only to the environment side:
  - `env_only_cpu_core`: bind the benchmark case process that runs the env loop.
  - `concurrent_cpu_core`: bind only the sim worker process.
- Persist CPU binding metadata into benchmark reports for throughput analysis.

## Non-Goals

- No CPU binding for model-only scenarios.
- No combined resource matrix such as `mps + cpu_core` or `mig + cpu_core` in this iteration.
- No thread-level or per-logical-env runtime pinning inside env implementations.
- No changes to main RLinf runtime, Ray scheduling, or production cluster placement.
- No attempt to make this fully cross-platform; Linux is the target platform.

## Current State

The benchmark currently models resources at the case level with:

- `mps_sm`
- `mig_device`

Scenario expansion is driven by `toolkits/rollout_eval/benchmark/scenarios.py`, and process-level GPU resource settings are injected by `toolkits/rollout_eval/benchmark/resource_binding.py`. Execution flows through:

- `orchestrator.py` for case loading and dispatch
- `single_runner.py` for `env_only_*` and `model_only_*`
- `pipeline_runner.py` for `concurrent_*`
- `reporting.py` for case-level and summary outputs

There is no CPU resource abstraction today.

## User-Facing Behavior

### Scenario Model

Add two new benchmark scenarios:

- `env_only_cpu_core`
- `concurrent_cpu_core`

These are independent scenario families, parallel to the current MPS and MIG scenarios. `model_only_cpu_core` is not introduced.

### CPU Binding Input

Primary entry is CLI-driven automatic partitioning.

New CLI arguments:

- `--cpu-bind-cores`
  - Defines the available CPU pool.
  - Accepts compact CPU list syntax such as `0-127` or `0-31,64-95`.
- `--cpu-bind-strategy`
  - First iteration supports only `even_split`.
- `--cpu-bind-config`
  - Optional YAML file path for explicit per-env core groups.
  - Supported as an override path, but not the primary workflow.
- `--cpu-bind-strict`
  - Defaults to enabled behavior.
  - Invalid CPU sets, insufficient cores, overlapping YAML assignments, or incomplete explicit mappings fail the case.

### Default Partitioning Semantics

When `--cpu-bind-strategy even_split` is active:

1. Resolve the final `cfg.env.eval.total_num_envs` for the case.
2. Parse the available CPU pool from `--cpu-bind-cores`.
3. Partition that pool across the logical env count.
4. Produce non-overlapping core groups.
5. If the core count is not divisible by the env count, assign the remainder deterministically to the first groups.
6. If the total available core count is smaller than the env count, fail the case.

Example:

- `total_num_envs = 32`
- `cpu_bind_cores = 0-127`

Result:

- 32 groups
- 4 cores per group
- no overlap

This partitioning is based on logical env count, not OS process count and not vectorized worker count.

## Execution Semantics

### Environment-Side Affinity Only

CPU core profile applies only to the environment side.

- `env_only_cpu_core`
  - Affinity is applied to the case subprocess before creating the env adapter.
- `concurrent_cpu_core`
  - Affinity is applied only to the sim worker process in `pipeline_runner.py`.
- `model_only_*`
  - No affinity changes.

### Process-Level Binding

The first iteration uses process-level affinity, not thread-level affinity.

This means the benchmark will compute a per-logical-env partition for reporting and future extension, but the actual applied affinity for the current runtime will be the union of all logical env core groups for the env-side process.

Rationale:

- The benchmark currently controls env execution at process boundaries.
- It does not have stable hooks for pinning individual logical env instances or internal threads.
- Process-level affinity is enough to profile the throughput impact of restricting env execution to a selected CPU set while keeping implementation scope contained.

### Platform Behavior

Implementation uses Linux CPU affinity via `os.sched_setaffinity`.

Behavior:

- If Linux affinity is available, apply the computed CPU set.
- If affinity is unavailable on the running platform, mark the case as skipped with a clear reason.

## Architecture Changes

### `types.py`

Extend benchmark request and case metadata with CPU binding fields.

Suggested additions:

- `BenchmarkRequest`
  - `cpu_bind_cores: str | None`
  - `cpu_bind_strategy: str | None`
  - `cpu_bind_config: str | None`
  - `cpu_bind_strict: bool`
- `BenchmarkCase`
  - `cpu_binding_mode: str | None`
  - `cpu_available_cores: tuple[int, ...] | None`
  - `cpu_env_core_groups: tuple[tuple[int, ...], ...] | None`

The request stores raw intent. The expanded case stores normalized, deterministic resource bindings for execution and reporting.

### `run.py`

Add CLI options and parse them into `BenchmarkRequest`.

Update defaults:

- `DEFAULT_SCENARIO_SET` includes:
  - `env_only_cpu_core`
  - `concurrent_cpu_core`

CPU arguments should remain optional so existing MPS/MIG-only usage keeps working.

### `scenarios.py`

Extend `SCENARIOS` with the two CPU core scenarios.

`expand_cases()` should:

- create CPU core cases only when CPU input is present and valid for the request
- compute or load CPU core groups during expansion so case IDs and reports remain deterministic
- encode CPU resource identity into the case ID, for example:
  - `env-only-cpu-core-maniskill-openvlaoft-cpu-even-split`

### New CPU Binding Helpers

Add CPU-specific helpers in `resource_binding.py` or a nearby benchmark-only helper module.

Required responsibilities:

- parse CPU list expressions such as `0-127,160-191`
- validate uniqueness and ordering
- build even-split groups from `(available_cores, total_num_envs)`
- load optional YAML explicit assignments
- validate:
  - no overlap
  - complete group coverage when explicit config is used
  - enough cores for the env count in strict mode
- apply process affinity

Suggested helper API shape:

- `parse_cpu_core_set(spec: str) -> tuple[int, ...]`
- `build_even_split_cpu_groups(cores: tuple[int, ...], env_count: int) -> tuple[tuple[int, ...], ...]`
- `load_cpu_groups_from_yaml(path: str, env_count: int) -> tuple[tuple[int, ...], ...]`
- `effective_process_affinity(groups: tuple[tuple[int, ...], ...]) -> tuple[int, ...]`
- `apply_cpu_affinity(cpus: tuple[int, ...]) -> None`

### `orchestrator.py`

Extend case execution logic:

- detect CPU core scenarios
- resolve CPU binding inputs into case-local metadata
- for `env_only_cpu_core`, apply CPU affinity before env adapter construction
- pass sim worker affinity into `run_dual_process_pipeline()` for `concurrent_cpu_core`

Validation must use the final case config after `num_envs_override` is applied, because CPU partitioning depends on the actual logical env count.

### `pipeline_runner.py`

Add optional sim-side CPU affinity support.

Suggested config extension:

- `PipelineRunnerConfig.sim_cpu_affinity: tuple[int, ...] | None = None`

Then in the sim worker startup path:

- apply affinity once at worker entry if configured
- leave model worker untouched

Applying affinity inside `_sim_worker_main()` is preferred over parent-side mutation because it is explicit, local to the worker, and easier to test.

### `reporting.py`

Extend case report resource metadata with CPU information:

- `cpu_binding_mode`
- `cpu_available_cores`
- `cpu_env_core_groups`
- `cpu_effective_affinity`

Summary markdown should include only compact CPU information such as:

- binding mode
- available core count
- env count
- average cores per env

Do not print the full core-group matrix in the markdown summary, because it will make the report noisy for large env counts.

## YAML Override Shape

YAML remains optional in this iteration. A simple explicit format is sufficient.

Example:

```yaml
env_core_groups:
  - [0, 1, 2, 3]
  - [4, 5, 6, 7]
  - [8, 9, 10, 11]
  - [12, 13, 14, 15]
```

Rules:

- number of groups must equal `cfg.env.eval.total_num_envs`
- groups must not overlap
- each group must contain at least one CPU core
- invalid or incomplete files fail the case in strict mode

## Error Handling

Case-level failure or skip behavior:

- invalid CPU core syntax: fail case
- duplicate cores after parsing: fail case
- available core count less than logical env count: fail case
- explicit YAML with overlap or missing groups: fail case
- Linux affinity API unavailable: skip case
- affinity application failure from OS: fail case

This follows the existing benchmark principle of isolating case failures without aborting the whole matrix.

## Testing Plan

Add unit tests covering:

- scenario expansion for CPU scenarios
- CPU list parsing:
  - range syntax
  - mixed range/list syntax
  - duplicate detection
  - malformed input
- even-split partitioning:
  - divisible case
  - remainder case
  - insufficient cores
- explicit YAML loading and validation
- affinity application:
  - mocked `os.sched_setaffinity`
  - env-only path applies affinity
  - concurrent path applies affinity only to sim worker
  - model-only path does not apply affinity
- report serialization of CPU metadata

If the current benchmark test suite includes orchestrator or pipeline behavior tests, add a minimal CPU scenario path test to confirm the sim worker receives the configured affinity.

## Rollout Plan

Implementation should proceed in this order:

1. Extend request/case types and CLI surface.
2. Add CPU parsing, partitioning, validation, and affinity helpers.
3. Add CPU scenario expansion.
4. Wire affinity into `orchestrator.py` and `pipeline_runner.py`.
5. Extend reporting output.
6. Add unit tests for parsing, expansion, execution, and reporting.
7. Update benchmark documentation or CLI help text if needed.

## Future Extensions

The design intentionally preserves room for later work:

- combine `cpu_core` with `mps` or `mig`
- move from scenario-family modeling to a generic multi-resource matrix
- add per-thread or per-subenv pinning if env implementations expose stable hooks

Those extensions are explicitly out of scope for this document.
