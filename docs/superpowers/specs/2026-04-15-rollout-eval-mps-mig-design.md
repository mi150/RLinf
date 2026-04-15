# Rollout Eval MPS/MIG Fine-Grained GPU Profiling Design

## 1. Context and Goal

This design extends `toolkits/rollout_eval` with fine-grained GPU resource benchmarking under both NVIDIA MPS and MIG, targeting performance characterization across environment/model combinations.

Scope is limited to the lightweight non-Ray rollout evaluation toolkit (`toolkits/rollout_eval`).

The benchmark must support:
- Concurrent env+model profiling under MPS.
- Single-side profiling under MPS (env-only or model-only).
- Concurrent env+model profiling under MIG (pre-created MIG devices).
- Single-side profiling under MIG (env-only or model-only).
- Concurrent mode implemented as a pipeline with process-level separation.

Primary metric priority is throughput. Latency metrics are secondary but required.

## 2. Confirmed Product Decisions

- Execution layer: `toolkits/rollout_eval` only.
- MIG lifecycle: consume pre-created MIG instances only (no create/destroy automation).
- Concurrent execution model: dual-process pipeline (`sim process` + `model process`).
- Initial sweep presets: 2 built-in env×model combinations.
- Metric priority: throughput first.
- Resource configuration: CLI-driven (no matrix YAML in v1).
- Additional requirement: explicit env-only and model-only profiling as first-class scenarios.
- `model_only` input source: dummy observations generated from one real env `reset()` template.
- `env_only` action source: random actions.
- Latency outputs include `avg`, `p50`, `p95`.

## 3. Architecture

### 3.1 High-Level Structure

Keep existing single-run path intact and add a dedicated benchmark orchestration layer.

- Keep: `python -m toolkits.rollout_eval.run`
  - Single run execution, existing report behavior remains stable.
- Add: `python -m toolkits.rollout_eval.benchmark.run`
  - Matrix expansion, scenario scheduling, result aggregation and summary reporting.

### 3.2 New Modules

- `toolkits/rollout_eval/benchmark/scenarios.py`
  - Scenario definitions, case schema, and matrix expansion logic.
- `toolkits/rollout_eval/benchmark/orchestrator.py`
  - End-to-end case scheduling, status handling, and summary generation.
- `toolkits/rollout_eval/benchmark/pipeline_runner.py`
  - Dual-process pipeline runtime for concurrent env+model benchmarks.
- `toolkits/rollout_eval/benchmark/single_runner.py`
  - Env-only and model-only execution runtimes.
- `toolkits/rollout_eval/benchmark/reporting.py`
  - Case-level and summary-level JSON/Markdown output generation.
- `toolkits/rollout_eval/benchmark/types.py`
  - Dataclasses for scenario config, execution result, and metrics.

### 3.3 Non-Goals (v1)

- No MIG partition create/delete automation.
- No runner/worker integration into training mainline.
- No YAML matrix authoring (CLI only in v1).

## 4. Scenario Set and Experiment Matrix

### 4.1 First-Class Scenarios

The benchmark matrix includes exactly six scenario classes:

1. `concurrent_mps`
2. `concurrent_mig`
3. `env_only_mps`
4. `model_only_mps`
5. `env_only_mig`
6. `model_only_mig`

### 4.2 Expansion Dimensions

- Preset dimension: fixed 2 env×model built-ins.
- MPS dimension:
  - Concurrent: SM allocation pairs for env/model.
  - Single-side: target allocation list for env-only or model-only.
- MIG dimension:
  - Concurrent: env/model MIG device pair.
  - Single-side: one MIG device bound to env-only or model-only runtime.

Each expanded case receives a deterministic `case_id` and isolated output folder.

## 5. Data Flow and Runtime Design

### 5.1 Concurrent Pipeline (Dual Process)

- `sim process`
  - Owns env adapter.
  - Produces observations (`obs_queue`) and consumes actions (`action_queue`).
  - Records env throughput and env latency metrics.
- `model process`
  - Owns model adapter.
  - Consumes observations and produces actions.
  - Records inference throughput and inference latency metrics.
- `orchestrator` (main process)
  - Manages warmup/measurement windows and stop conditions.
  - Aggregates counters/time windows into final metrics.

Default queue depth is `1` to avoid hiding bottlenecks by excessive buffering.

### 5.2 Single-Side Runtime

- `env_only_*`
  - Runs env stepping with random action generator.
  - Outputs env throughput + latency (`avg/p50/p95`).
- `model_only_*`
  - Performs one env `reset()` to capture observation template.
  - Generates dummy observations conforming to this template.
  - Runs inference loop without env stepping.
  - Outputs model throughput + latency (`avg/p50/p95`).

## 6. Resource Binding Strategy

### 6.1 MPS Cases

- Benchmark does not manage MPS daemon lifecycle.
- Per-case process environment carries MPS-related settings (e.g. active thread percentage or equivalent configured knobs).
- Case metadata captures the exact resource knobs used.

### 6.2 MIG Cases

- MIG devices must already exist before benchmark execution.
- Bind process visibility via `CUDA_VISIBLE_DEVICES=<MIG-UUID>`.
- Concurrent MIG case binds env/model process to different MIG UUIDs.

## 7. CLI Contract (v1)

Proposed benchmark CLI:

```bash
python -m toolkits.rollout_eval.benchmark.run \
  --config-path examples/embodiment/config \
  --config-name <name> \
  --scenario-set concurrent_mps,concurrent_mig,env_only_mps,model_only_mps,env_only_mig,model_only_mig \
  --mps-sm 20,40,60 \
  --mig-devices MIG-uuid-a,MIG-uuid-b \
  --pipeline process \
  --env-model-preset preset_a,preset_b \
  --model-only-input dummy_from_env_reset \
  --env-only-action random \
  --warmup-steps 20 \
  --measure-steps 200 \
  --output-dir ./rollout_eval_output/benchmark
```

Notes:
- `--pipeline process` is default and only supported mode in v1.
- Existing `toolkits.rollout_eval.run` CLI remains backward compatible.

## 8. Metrics and Reporting

### 8.1 Throughput Metrics (Primary)

- `env_steps_per_sec`
- `model_infers_per_sec`
- `pipeline_samples_per_sec` (concurrent scenarios only)

### 8.2 Latency Metrics (Secondary)

For relevant stages, output:
- `avg`
- `p50`
- `p95`

Stages:
- `env_step_latency`
- `model_infer_latency`
- `pipeline_step_latency` (concurrent scenarios)

### 8.3 Report Artifacts

Per case:
- `case_report.json`
- `case_report.md`
- `case_meta.json` (command, env vars, git commit, timestamp, scenario/resource config)

Global:
- `summary.json`
- `summary.md`

Directory convention:
- `<output>/<scenario>/<case_id>/...`

## 9. Error Handling and Reproducibility

- Preflight validation checks and case-level fail/skip handling:
  - MPS unavailable/invalid knobs.
  - MIG UUID missing or mismatched.
  - Preset dependency missing (env/model path/runtime requirements).
- One case failure does not abort whole matrix run.
- Failed/skipped cases are included in summary with structured fields:
  - `status`
  - `error_type`
  - `error_message`
- Reproducibility controls:
  - capture seed, warmup/measure settings, env var snapshots.
  - fixed output structure and deterministic case IDs.

## 10. Testing Strategy and Done Criteria

### 10.1 Unit Tests

Add benchmark-focused unit tests under `tests/unit_tests/`:
- Matrix expansion for all six scenarios.
- Throughput and latency (`avg/p50/p95`) aggregation correctness.
- Case status transitions and failure schema completeness.

### 10.2 Integration Tests (Lightweight)

- Smoke tests for `env_only_mps` and `model_only_mps` on minimal steps.
- `model_only` dummy-from-reset path validation.
- Dual-process pipeline liveness and count consistency checks.

### 10.3 Manual Validation (MPS/MIG Host)

- MPS concurrent + single-side full matrix execution.
- MIG concurrent + single-side execution with pre-created UUIDs.
- Summary report integrity and cross-case comparability.

### 10.4 Done Criteria

- All six scenario classes executable (allowing resource-based skips).
- Unified summary report generated with throughput-first ordering.
- Env-only and model-only scenarios independently runnable.
- Existing `toolkits.rollout_eval.run` behavior unaffected.

## 11. Rollout Plan (Implementation-Oriented)

1. Add benchmark type/CLI skeleton and scenario expansion.
2. Implement single-side runners (env-only/model-only) and metrics aggregator.
3. Implement dual-process concurrent pipeline runtime.
4. Implement MPS/MIG resource binding injection.
5. Implement reporting and summary ranking.
6. Add unit/integration tests and smoke scripts.

