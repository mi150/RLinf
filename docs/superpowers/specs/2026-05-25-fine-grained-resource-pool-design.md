# Fine-Grained Resource Pool Design

Date: 2026-05-25

## Goal

Add an optional fine-grained CPU/GPU resource pool for embodied training. The
feature lets RLinf bind CPU cores to worker processes, optionally bind CPU cores
to individual subprocess environments inside env workers, and assign GPU SM
quota profiles to workers through non-destructive MPS or pre-created MIG
bindings.

The primary goals are:

- Prevent unintended CPU interference between embodied workers by default.
- Allow env workers to split their assigned CPU cores across local environments,
  including one-core-per-env layouts.
- Allow worker-level GPU quota profiles at `0`, `20`, `40`, `60`, `80`, or
  `100` percent.
- Preserve existing behavior when `cluster.resource_pool` is omitted or
  disabled.

## Scope

Version 1 only integrates with embodied training entrypoints:

- `examples/embodiment/train_embodied_agent.py`
- `examples/embodiment/train_async.py`
- `EnvWorker` and `AsyncEnvWorker`
- embodied actor, rollout, env, and optional reward worker groups launched from
  those entrypoints

Reasoning, coding, agent, SGLang, vLLM, generic scheduler-wide resource
binding, and automatic cluster resource scheduling are outside the v1 scope.

The existing `cluster.component_placement` remains the source of truth for
component worker counts, node placement, and accelerator placement. The resource
pool runs after component placement and adds stricter process-level binding
metadata for the already-placed worker ranks.

## Configuration

`cluster.resource_pool` is optional. When omitted or disabled, RLinf behaves as
it does today.

### CPU Configuration

```yaml
cluster:
  resource_pool:
    enabled: true
    allocation_mode: default
    cpu:
      enabled: true
      pools:
        env_cpu:
          node_group: cluster
          cores: 0-63
      components:
        env:
          pool: env_cpu
          granularity: per_env
```

Rules:

- CPU core specs support comma-separated IDs and ranges, for example
  `0-7,16,20-23`.
- `allocation_mode: default` performs strict exclusive allocation inside each
  CPU pool. The same CPU core is not assigned to multiple workers.
- `granularity: process` binds the worker process to its assigned cores.
- `granularity: per_env` is valid only for the `env` component. It first assigns
  exclusive cores to each env worker, then splits that worker-local core set
  across its local environment subprocesses.
- Every worker and every supported local env must receive at least one CPU core.
  Invalid or undersized pools fail before workers are launched.
- Explicit CPU sharing is only supported through `allocation_mode: plan_file`,
  where the user supplies final worker-level bindings intentionally.

### GPU Configuration

```yaml
cluster:
  resource_pool:
    enabled: true
    allocation_mode: default
    gpu:
      enabled: true
      mode: mps
      pools:
        train_gpu:
          node_group: cluster
          devices: 0-7
      components:
        rollout:
          pool: train_gpu
          sm_percent: 40
        env:
          pool: train_gpu
          sm_percent: 0
```

Rules:

- `sm_percent` must be one of `0`, `20`, `40`, `60`, `80`, or `100`.
- `sm_percent: 0` means the resource pool does not inject a GPU quota for that
  worker. The worker keeps the normal `component_placement` GPU visibility. This
  is intended for simulation environments that do not use GPU quota binding.
- In MPS mode, RLinf injects process environment variables such as
  `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`. It does not start, stop, or validate the
  MPS control daemon lifecycle.
- In MIG mode, RLinf binds workers to pre-created MIG UUIDs. It does not create
  or destroy MIG instances.
- GPU pool `node_group` and `devices` or `mig_devices` must match the worker
  placements produced by `component_placement`. Mismatches fail before launch.

Example MIG pool:

```yaml
cluster:
  resource_pool:
    enabled: true
    allocation_mode: default
    gpu:
      enabled: true
      mode: mig
      pools:
        train_gpu:
          node_group: cluster
          mig_devices:
            - uuid: MIG-aaaaaaaa
              parent_gpu: 0
              sm_percent: 20
            - uuid: MIG-bbbbbbbb
              parent_gpu: 0
              sm_percent: 20
      components:
        rollout:
          pool: train_gpu
          sm_percent: 20
```

## Architecture

Add a main-package scheduler resource-pool module:

```text
rlinf/scheduler/resource_pool/
  __init__.py
  bindings.py
  config.py
  cpu_binding.py
  gpu_binding.py
  solver.py
  pool.py
```

Responsibilities:

- Parse and validate `cluster.resource_pool`.
- Build worker-level `WorkerResourceBinding` records from
  `HybridComponentPlacement`.
- Load and validate external plan files.
- Convert binding records to Ray runtime environment variables.
- Provide CPU affinity helpers shared by env workers and subprocess env workers.
- Provide MPS/MIG environment helpers and validation.

Entrypoint flow:

1. Validate the Hydra config.
2. Create `Cluster`.
3. Create `HybridComponentPlacement`.
4. Create `FineGrainedResourcePool.from_config(cfg, cluster, component_placement)`.
5. Pass `resource_pool.get_component_bindings(component)` into each embodied
   `WorkerGroup.launch()`.
6. `WorkerGroup` merges binding env vars by worker rank before actor creation.
7. `Worker` parses `RLINF_RESOURCE_BINDING_JSON` and exposes the parsed binding
   through `worker.resource_binding` and `WorkerInfo`.
8. `EnvWorker` and `AsyncEnvWorker` apply process CPU affinity before creating
   envs.
9. Supported SubprocVectorEnv-style env subprocesses apply per-env affinity
   before constructing the actual env instance.

## Binding Model

The solver outputs one record per bound worker rank:

```python
WorkerResourceBinding(
    component="env",
    rank=0,
    cluster_node_rank=0,
    node_group_label="cluster",
    cpu=CpuBinding(
        process_cpu_cores=(0, 1, 2, 3),
        env_cpu_core_groups=((0,), (1,), (2,), (3,)),
    ),
    gpu=GpuBinding(
        mode="mps",
        sm_percent=20,
        visible_devices=("0",),
        mig_device_uuid=None,
        parent_gpu=0,
    ),
)
```

Environment variables derived from bindings:

- `RLINF_RESOURCE_BINDING_JSON`
- `RLINF_CPU_AFFINITY`
- `RLINF_ENV_CPU_CORE_GROUPS`
- `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`
- `CUDA_VISIBLE_DEVICES` for MIG UUID binding or normal placement visibility

The binding JSON is also written to the resource-pool artifact.

## CPU Binding Behavior

Process-level binding applies to all embodied env backends on Linux hosts that
support `os.sched_setaffinity`. The binding is applied inside the worker process
before env class imports or env construction that may start subprocesses.

Per-env binding is supported in v1 for SubprocVectorEnv-style backends:

- common `rlinf.envs.venv.SubprocVectorEnv`
- LIBERO
- CALVIN
- MetaWorld
- RoboCasa
- Habitat

Each supported subprocess worker receives a local environment index and calls
the shared CPU affinity helper before `env_fn_wrapper.data()`.

ManiSkill, Behavior, IsaacLab, RealWorld, and other envs with separate process
models or external runtime lifecycles support process-level EnvWorker binding in
v1, but not per-env binding. If `granularity: per_env` is configured for an
unsupported backend, RLinf fails before env creation and tells the user to use
`granularity: process` or a supported backend.

For `pipeline_stage_num > 1`, the local env count used for splitting is the
number of subprocess environments created by that EnvWorker across all local
stages. The generated `env_cpu_core_groups` must cover every local environment
subprocess exactly once.

## GPU Binding Behavior

MPS mode is a process-environment binding layer:

- RLinf injects `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` for nonzero quotas.
- RLinf does not manage MPS daemon lifecycle.
- MPS quotas are soft scheduling limits, not hard exclusive SM partitions.
- Default mode allows the sum of MPS percentages on one physical GPU to exceed
  `100`; the artifact reports the per-GPU total so users can see oversubscription.

MIG mode consumes pre-created devices:

- Each configured MIG UUID must be unique in default mode.
- Each nonzero worker quota binds to a MIG UUID via `CUDA_VISIBLE_DEVICES`.
- A worker request must be less than or equal to the MIG device metadata
  `sm_percent`; exact matching is recommended for the `20/40/60/80/100` profiles.
- A MIG UUID is assigned to at most one worker in default mode.
- Sharing a MIG UUID is only allowed through an explicit plan file.

## Default Solver

The default solver is deterministic:

- Iterate components in sorted order for stable artifacts.
- Sort placements by worker rank.
- Validate each component's placements against the selected CPU/GPU pool
  `node_group`.
- Validate GPU placement devices against pool devices or MIG parent devices.
- Split CPU cores evenly by worker count, distributing remainders to lower ranks.
- For `env` per-env binding, split each worker's assigned core set across local
  env subprocesses, again deterministically.
- Generate no GPU binding for `sm_percent: 0`.

If the solver cannot produce a non-overlapping default CPU plan, it raises an
error before any worker group is launched.

## Plan File Mode

`allocation_mode: plan_file` loads final worker-level bindings from
`allocation_plan_path`.

Plan file mode is intentionally more permissive:

- CPU core sharing is allowed when multiple bindings explicitly reference the
  same core.
- MIG UUID sharing is allowed only if explicitly present in the file.
- The loaded plan is still validated against component ranks, cluster node
  ranks, node group labels, allowed GPU quota values, and known MIG metadata.

The artifact marks shared CPU cores or shared MIG UUIDs so the user can
distinguish intentional sharing from the default exclusive mode.

## Artifact

When enabled, RLinf writes `resource_pool_plan.json` under the run output
directory. The implementation should prefer Hydra's runtime output directory;
if an embodied runner output directory is already canonical for the run, the
implementation may place the artifact there consistently.

The artifact includes:

- feature config summary
- one binding entry per bound worker
- CPU shared-core summary
- MIG shared-UUID summary
- MPS per-physical-GPU quota totals
- validation mode: `default` or `plan_file`

## Error Handling

Fail before worker launch for:

- invalid CPU core specs
- duplicate or out-of-range CPU cores in a default exclusive pool
- insufficient CPU cores for requested worker or per-env counts
- `per_env` requested for an unsupported env backend
- invalid GPU `sm_percent`
- GPU pool node group or device mismatch with placement
- missing or duplicate MIG UUID metadata in default mode
- plan file bindings for nonexistent component ranks

MPS daemon availability is not checked by default because daemon lifecycle is
outside RLinf's scope. Documentation and logs must state this clearly.

## Tests

Unit coverage:

- config parser for disabled, CPU-only, MPS, MIG, and plan-file modes
- CPU core spec parser and exclusive splitter
- per-env CPU group generation and validation
- plan-file explicit sharing behavior
- MPS env var injection and zero-quota no-op behavior
- MIG UUID binding and duplicate validation
- pool-vs-placement mismatch errors
- `WorkerGroup.launch(resource_bindings=...)` rank injection
- `Worker` parsing and `WorkerInfo` propagation
- `EnvWorker` and `AsyncEnvWorker` process affinity
- supported SubprocVectorEnv-style backends applying affinity before env factory

Regression coverage:

- embodied entrypoints with `resource_pool.enabled: false` preserve existing
  launch behavior
- unsupported per-env backend fails with an actionable error

Manual validation on prepared hosts:

- MPS host with daemon already running
- MIG host with pre-created UUIDs
- one supported SubprocVectorEnv-style embodied environment
- one process-only embodied environment

## Non-Goals

- No scheduler-wide resource-pool integration in v1.
- No Ray `num_cpus`, `num_gpus`, or custom resource scheduling changes.
- No MPS daemon lifecycle management.
- No MIG instance creation or deletion.
- No hard-isolation claim for MPS quotas.
- No per-env binding for ManiSkill, Behavior, IsaacLab, RealWorld, or external
  process env backends in v1.

## Success Criteria

- Existing embodied training runs unchanged when `cluster.resource_pool` is
  disabled or omitted.
- Default CPU allocation prevents unintended core sharing across workers.
- Explicit plan-file CPU sharing is possible and visible in the artifact.
- Supported env subprocesses can receive independent CPU core groups.
- GPU quotas accept only `0/20/40/60/80/100`.
- MPS and MIG bindings are non-destructive and limited to process visibility or
  process quota environment variables.
- Invalid configs fail before workers are launched.
