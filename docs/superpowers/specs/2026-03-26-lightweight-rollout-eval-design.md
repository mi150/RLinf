# Lightweight Non-Ray Rollout Eval Design

## 1. Background and Goal

RLinf currently evaluates embodied policies through runner/worker flows built on Ray. This is suitable for distributed training/evaluation, but it is heavy for rapid local validation and profiling.

The goal of this design is to add a lightweight rollout-eval tool that:

- does not depend on Ray process orchestration;
- can initialize multiple embodied simulation environments;
- can load multiple VLA models;
- executes the core path `env -> obs -> model inference -> action -> env.step`;
- profiles performance;
- validates correctness of interfaces and runtime behavior.

This tool is intended for local debugging, performance characterization, and regression checks in a single-machine setup.

## 2. Confirmed Scope and Constraints

The following constraints are explicitly confirmed:

- Coverage target: broad coverage across most currently supported embodied env/model combinations where interfaces are compatible.
- Config source: existing Hydra configs only.
- Correctness checks included:
  - interface/contract checks;
  - determinism/stability checks under fixed seeds.
- Correctness checks excluded:
  - no golden/baseline artifact comparison in this phase.
- Runtime topology: single-machine multi-GPU support.
- Orchestration: no Ray, no distributed scheduler.
- Location: `toolkits/rollout_eval/`.
- Batch control: environment count (`num_envs`) must be controlled from Hydra config only; no CLI override.

## 3. High-Level Architecture

Implementation location:

- `toolkits/rollout_eval/run.py` (entrypoint with Hydra integration)
- `toolkits/rollout_eval/config_bridge.py`
- `toolkits/rollout_eval/adapters/env_adapter.py`
- `toolkits/rollout_eval/adapters/model_adapter.py`
- `toolkits/rollout_eval/engine/loop.py`
- `toolkits/rollout_eval/checks/interface_checks.py`
- `toolkits/rollout_eval/checks/determinism_checks.py`
- `toolkits/rollout_eval/profiling/collector.py`
- `toolkits/rollout_eval/reporting/` (json + markdown output)

Design principles:

- Keep the loop minimal and explicit.
- Reuse existing RLinf registries/utilities where possible.
- Isolate env/model differences in adapters.
- Avoid coupling to existing Ray worker channels.

## 4. Data Flow and Runtime Stages

### Stage A: Config normalize

`config_bridge.py` reads an existing Hydra config and derives a normalized eval runtime view, without mutating user-visible training fields.

Derived runtime fields include:

- `num_steps`
- `warmup_steps`
- `seed`
- `profiling flags`
- `check flags`
- `device policy`
- `num_envs` and related rollout shape metadata

### Stage B: Environment boot

`env_adapter.py` constructs environments via `rlinf.envs.get_env_cls(env_type, env_cfg)`.

Responsibilities:

- instantiate env with config-derived init params;
- run `reset()` smoke check;
- produce normalized observation structures (`ObsBatch`);
- expose environment metadata (`obs_spec`, `action_spec`, `num_envs`, etc.).

### Stage C: Model boot

`model_adapter.py` constructs models through existing model-loading paths (`rlinf.models.get_model` and model-specific helpers as needed).

Responsibilities:

- load configured VLA checkpoint/model variant;
- place model on single-machine multi-GPU policy;
- expose a stable `infer(obs_batch) -> action_batch` interface.

No Ray worker abstraction is used in this layer.

### Stage D: Rollout loop

`engine/loop.py` runs:

1. read/normalize obs from env
2. preprocess for model
3. model forward/inference
4. action postprocess/shape guard
5. `env.step(action)`
6. record metrics/check state

Loop phases:

- warmup phase (excluded from final profiling summary);
- measurement phase (used for throughput/latency/memory stats).

### Stage E: Checks and reporting

`checks/` validates:

- interface contracts (key presence, shape, dtype, device consistency, finite values);
- action validity constraints (bounds/type/shape expected by env);
- deterministic stability across fixed-seed reruns with tolerated thresholds.

`reporting/` writes:

- machine-readable JSON (`report.json`);
- human-readable markdown summary (`report.md`).

## 5. Batch Size Control via Environment Count

A required behavior is direct control of inference batch through environment multiplicity.

Rules:

- source of truth for env count is Hydra config only;
- no command-line override for `num_envs`;
- effective inference batch is derived from env parallelism semantics (for example `num_envs`, and environment/group semantics if configured);
- runtime asserts ensure `obs_batch` and `action_batch` leading dimensions remain consistent with derived effective batch.

Output reporting includes throughput and latency tied to the configured env count, enabling apples-to-apples profiling across config variants.

## 6. Multi-GPU Strategy (Single Machine)

This tool supports multi-GPU placement on one host, while avoiding distributed process groups.

Strategy:

- prefer local model placement/partitioning supported by the selected model stack (for example automatic device mapping when available);
- use simple local fallbacks where applicable;
- do not initialize cross-process distributed orchestration.

Out of scope:

- multi-node execution;
- Ray actor/channel placement semantics.

## 7. Error Handling Policy

Startup failures are fail-fast:

- invalid env config or env construction failure;
- missing/incompatible model artifacts;
- unsupported runtime device policy.

Runtime errors are classified:

- `fatal`: unrecoverable interface mismatch, persistent invalid tensor states, repeated critical step failures;
- `warn`: transient step anomalies eligible for bounded retry.

All errors are captured in JSON report artifacts with enough context to reproduce (config summary + stack trace snippet + stage identifier).

## 8. Compatibility and Extensibility

Compatibility approach:

- broad support for env/model pairs that follow shared interfaces;
- adapter-specific branches for known outliers;
- avoid spreading env/model-specific conditionals into core loop logic.

Extensibility:

- add new env/model support by extending adapter registries;
- keep checker modules independent and composable;
- maintain reporting schema stability for downstream automation.

## 9. Non-Goals (Current Phase)

- replacing current Ray-based production evaluation pipelines;
- baseline/golden output artifact comparison;
- distributed, cross-node evaluation orchestration;
- broad refactoring of existing runner/worker internals.

## 10. MVP Acceptance Criteria

MVP is complete when all conditions are met:

- non-Ray end-to-end rollout eval runs successfully from existing Hydra config;
- broad env/model coverage path exists via adapters, with graceful unsupported-path errors;
- env-count-driven batch control works from Hydra config;
- correctness checks include interface contracts + fixed-seed stability;
- profiling outputs include warmup-separated measurement metrics;
- reports are generated as JSON + markdown.

## 11. Risks and Mitigations

Risk: hidden env/model interface differences break a fully generic path.

Mitigation:

- strict adapter boundaries;
- explicit capability checks at startup;
- clear unsupported-combination diagnostics.

Risk: profiling comparability is noisy due to startup and cache effects.

Mitigation:

- mandatory warmup phase;
- repeatable measurement windows;
- fixed-seed deterministic mode in stability checks.

Risk: multi-GPU behavior differs by model backend.

Mitigation:

- backend-aware placement in model adapter;
- standardized runtime telemetry exposing actual device placement.

## 12. Proposed Next Step

After this design document is approved, create an implementation plan using the writing-plans workflow, then execute in phases:

1. scaffold package and entrypoint;
2. implement config bridge and loop skeleton;
3. add adapters for initial broad set of env/model interfaces;
4. add checks and profiling collector;
5. add tests and docs.
