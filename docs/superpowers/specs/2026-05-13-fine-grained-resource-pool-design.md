# Fine-Grained Resource Pool Design

Date: 2026-05-13

## Goal

Add a fine-grained CPU/GPU resource pool for the embodied training path. The
first version targets:

- `examples/embodiment/train_embodied_agent.py`
- `examples/embodiment/train_async.py`
- `EnvWorker` / `AsyncEnvWorker`
- HuggingFace rollout workers
- FSDP actor workers

The feature should let RLinf derive a worker-level resource binding plan from a
cluster-level resource pool. CPU resources are managed at core granularity. GPU
resources are managed through non-destructive MPS or pre-created MIG bindings.

## Confirmed Scope

Version 1 is intentionally scoped to embodied training. Reasoning, coding,
agent, SGLang, vLLM, reward, and generic scheduler-wide resource binding are
outside this implementation.

The current `cluster.component_placement` remains the source of truth for which
components exist, how many worker ranks each component has, and which nodes or
accelerators they are placed on. The new resource pool runs after component
placement and produces more specific CPU/GPU binding metadata for those worker
ranks.

## Current State

RLinf currently uses Ray primarily for process lifecycle and node affinity:

- `Cluster.allocate()` starts Ray actors with `NodeAffinitySchedulingStrategy`.
- Worker placement injects runtime environment variables such as
  `CUDA_VISIBLE_DEVICES`, `RANK`, `WORLD_SIZE`, `LOCAL_HARDWARE_RANKS`, and
  `NODE_GROUP_LABEL`.
- `WorkerGroup.launch()` does not request Ray `num_cpus`, `num_gpus`, or custom
  resource quantities.
- Placement strategies map component ranks to node or hardware ranks, but do not
  model CPU core ranges, MPS percentages, MIG UUIDs, or GPU memory limits.

There is benchmark-only code in `toolkits/rollout_eval/benchmark` for CPU
affinity and MPS/MIG environment-variable injection. The production feature
should use the same non-destructive philosophy, but the implementation should
live under the main package rather than importing benchmark internals.

## User-Facing Configuration

Add an optional `cluster.resource_pool` section. When omitted or disabled,
behavior must remain unchanged.

Example MPS configuration:

```yaml
cluster:
  resource_pool:
    enabled: true
    allocation_mode: default
    allocation_plan_path: null

    cpu:
      enabled: true
      pools:
        train_env_cpu:
          node_group: cluster
          cores: 0-63
      components:
        env:
          pool: train_env_cpu
          granularity: per_env
          unsupported_env_policy: error

    gpu:
      enabled: true
      mode: mps
      pools:
        train_gpu:
          node_group: cluster
          devices: 0-7
      components:
        env:
          pool: train_gpu
          sm_percent: 20
          memory_mb: null
          separate_gpus: false
        rollout:
          pool: train_gpu
          sm_percent: 40
          memory_mb: null
          separate_gpus: false
        actor:
          pool: train_gpu
          sm_percent: 80
          memory_mb: null
          separate_gpus: true
```

Example MIG configuration:

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
            - uuid: MIG-xxxxxxxx
              parent_gpu: 0
              sm_percent: 25
              memory_mb: 10240
            - uuid: MIG-yyyyyyyy
              parent_gpu: 0
              sm_percent: 25
              memory_mb: 10240
      components:
        rollout:
          pool: train_gpu
          sm_percent: 25
          memory_mb: 10240
          separate_gpus: false
```

`allocation_mode` supports:

- `default`: use the built-in deterministic solver.
- `plan_file`: load final worker-level bindings from `allocation_plan_path`.

## Architecture

Add a resource-pool package under the main scheduler area:

```text
rlinf/scheduler/resource_pool/
  __init__.py
  config.py
  pool.py
  solver.py
  bindings.py
  cpu_binding.py
  gpu_binding.py
```

Responsibilities:

- Parse and validate `cluster.resource_pool`.
- Build worker-level CPU/GPU binding records from `HybridComponentPlacement`.
- Load and validate external allocation plans.
- Convert binding records into Ray runtime environment variables.
- Provide CPU affinity helpers shared by EnvWorker and subprocess env workers.
- Provide GPU environment helpers for MPS/MIG binding.

The embodied entrypoints should create the resource pool after
`HybridComponentPlacement`:

```python
component_placement = HybridComponentPlacement(cfg, cluster)
resource_pool = FineGrainedResourcePool.from_config(
    cfg=cfg,
    cluster=cluster,
    component_placement=component_placement,
)
```

Each worker group launch then receives the binding plan for that component:

```python
actor_group = actor_worker_cls.create_group(cfg).launch(
    cluster,
    name=cfg.actor.group_name,
    placement_strategy=actor_placement,
    resource_bindings=resource_pool.get_component_bindings("actor"),
)
```

`WorkerGroup.launch()` accepts this optional argument. When it is absent, the
existing launch path is unchanged.

## Binding Data Model

The solver outputs one binding record per worker rank:

```python
WorkerResourceBinding(
    component="env",
    rank=0,
    cluster_node_rank=0,
    cpu=CpuBinding(
        process_cpu_cores=(0, 1, 2, 3, 4, 5, 6, 7),
        env_cpu_core_groups=((0, 1), (2, 3), (4, 5), (6, 7)),
    ),
    gpu=GpuBinding(
        mode="mps",
        visible_devices=("0",),
        mps_active_thread_percentage=20,
        memory_mb=None,
        mig_device_uuid=None,
        parent_gpu=0,
    ),
)
```

The record is serializable to JSON so it can be injected into worker
environment variables and written to an experiment artifact.

Binding-derived environment variables:

- `RLINF_RESOURCE_BINDING_JSON`
- `RLINF_CPU_AFFINITY`
- `RLINF_ENV_CPU_CORE_GROUPS`
- `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`
- `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` when `memory_mb` is configured for MPS
- `CUDA_VISIBLE_DEVICES` for MIG UUID binding or normal GPU visibility

## Default Solver

The default solver is deterministic and conservative.

CPU:

- Parse CPU pool core expressions such as `0-63` or `0-31,64-95`.
- Partition cores by component and worker rank based on final worker placement.
- For `env.granularity: per_env`, split each EnvWorker rank's process cores into
  non-overlapping per-env groups.
- Bind the EnvWorker process to the union of its per-env core groups.
- Reject overlapping core assignments and insufficient core counts.

GPU:

- Start from the existing component placement result for `actor`, `rollout`, and
  `env`.
- In MPS mode, bind each worker to a visible GPU device and inject the requested
  MPS limits.
- In MIG mode, bind each worker to a pre-created MIG UUID and validate requested
  SM/memory metadata against the MIG device metadata in the pool config.
- Satisfy `separate_gpus: true` before colocated requests.
- Default ordering is actor/training workers first, then rollout, then env.
- If the constraints cannot be satisfied, fail before launching workers and
  report the conflicting component and rank.

External plan mode:

- The plan file provides the final worker-level binding records.
- RLinf validates component names, ranks, node consistency, CPU overlap, GPU
  device references, and required fields.
- A valid external plan fully overrides the default solver.

## CPU Per-Env Binding

Version 1 supports strict per-env CPU binding only for SubprocVectorEnv-style
environment backends where each logical environment has a subprocess worker.
This includes the common `rlinf.envs.venv.SubprocVectorEnv` path and
SubprocVectorEnv-like derivatives such as CALVIN and RoboCasa.

The binding sequence is:

1. `WorkerGroup` injects the EnvWorker's binding record.
2. EnvWorker applies process-level affinity to the union of assigned cores at
   initialization time.
3. EnvWorker passes the binding metadata into environment construction through
   `worker_info` or an equivalent resource-binding helper.
4. Subprocess env workers apply their per-env CPU affinity at the beginning of
   the subprocess entry function, before creating the actual env instance.

Unsupported environment backends must fail when `granularity: per_env` is
configured. There is no silent fallback to process-level binding in v1.

The failure message includes:

- requested env type
- requested granularity
- supported v1 backend family
- how to disable resource_pool or change the env backend

## GPU Binding Semantics

GPU binding is non-destructive.

MPS:

- RLinf does not start or stop the MPS control daemon.
- RLinf assumes the host has been prepared for MPS before training starts.
- `sm_percent` maps to `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`.
- `memory_mb`, when configured, maps to a CUDA MPS client memory limit
  environment variable if supported by the target stack.
- MPS percentages are limits, not exclusive reservations. The documentation and
  logs must not describe MPS as hard SM partitioning.

MIG:

- RLinf does not create or destroy MIG instances.
- MIG devices must be created before training starts and listed in
  `cluster.resource_pool.gpu.pools`.
- Worker visibility is bound by setting `CUDA_VISIBLE_DEVICES` to the MIG UUID.
- MIG provides hard isolation according to the pre-created profile. The
  `sm_percent` and `memory_mb` fields are used for validation, metadata, and
  solver decisions, not for changing the device at runtime.

## Integration Points

### Embodied Entrypoints

Modify only the embodied launch scripts in v1:

- `examples/embodiment/train_embodied_agent.py`
- `examples/embodiment/train_async.py`

Both should:

- construct the resource pool when enabled
- pass component bindings to actor, rollout, and env launches
- write the final plan artifact after successful validation

### WorkerGroup and Cluster

`WorkerGroup.launch()` gains an optional `resource_bindings` parameter. It
validates that one binding exists for each placement rank. During `_create_workers`
it merges binding-derived environment variables after normal placement-derived
variables, so MIG UUIDs can intentionally override the normal
`CUDA_VISIBLE_DEVICES`.

`Cluster.allocate()` can remain unchanged if all required binding data is passed
through `env_vars`.

### Worker and EnvWorker

The base `Worker` should expose a parsed resource binding from
`RLINF_RESOURCE_BINDING_JSON` so workers can log or inspect their assignments.

EnvWorker applies process-level CPU affinity before environment instances
are created. Subprocess env worker implementations use a shared helper to
retrieve and apply the per-env core group for their local env index.

## Artifacts and Logging

When resource pool is enabled, write the validated plan to:

```text
<runner.logger.log_path>/<runner.logger.experiment_name>/resource_pool_plan.json
```

Each worker should log a compact summary:

- component
- rank
- node rank
- CPU process affinity
- per-env CPU group count when relevant
- GPU mode
- visible device or MIG UUID
- MPS active thread percentage when relevant

## Error Handling

Fail before worker launch for:

- malformed CPU core expressions
- CPU core overlap across simultaneous bindings
- insufficient CPU cores for per-env groups
- CPU pool node group mismatches
- `per_env` CPU binding requested for unsupported env backend
- `os.sched_setaffinity` unavailable while CPU binding is enabled
- missing or malformed GPU pool config
- MPS percentage outside `[1, 100]`
- MIG mode without UUID metadata
- external plan component/rank mismatch
- external plan binding a worker to a different node than component placement

Fail during worker initialization for:

- OS affinity application failure
- invalid binding JSON
- subprocess env worker missing a required per-env core group

Runtime failures should be logged through RLinf worker logging and should stop
training rather than continue with untrusted resource isolation.

## Testing Strategy

Unit tests:

- resource pool config parsing and validation
- CPU core expression parsing and deterministic splitting
- solver rank-to-binding determinism
- CPU overlap and insufficient-core failures
- MPS environment variable injection
- MIG UUID visibility override
- external plan validation
- `WorkerGroup.launch()` binding count validation

Integration-style tests:

- fake embodied entrypoint path with `resource_pool.enabled: false` remains
  unchanged
- enabled resource pool passes binding env vars into fake workers
- fake SubprocVectorEnv worker applies per-env affinity before env factory call

Manual validation:

- MPS host with MPS daemon already running
- MIG host with pre-created MIG UUIDs
- one supported SubprocVectorEnv-style embodied environment
- one unsupported env backend to verify strict failure behavior

## Non-Goals

- No automatic MPS daemon lifecycle management.
- No automatic MIG instance creation or deletion.
- No scheduler-wide support for all RLinf runners in v1.
- No Ray `num_cpus`, `num_gpus`, placement group, or custom resource scheduling
  changes in v1.
- No silent fallback from per-env CPU binding to process-level binding.
- No guarantee that MPS provides hard SM or memory isolation.

## Done Criteria

- Existing embodied training still works when `cluster.resource_pool.enabled` is
  absent or false.
- When enabled, actor, rollout, and env workers receive deterministic binding
  records.
- Supported SubprocVectorEnv-style envs bind subprocess workers to per-env CPU
  core groups.
- Unsupported envs fail clearly when per-env CPU binding is requested.
- MPS and MIG modes inject only non-destructive process environment bindings.
- The final resource pool plan is persisted as a JSON artifact for
  reproducibility.
