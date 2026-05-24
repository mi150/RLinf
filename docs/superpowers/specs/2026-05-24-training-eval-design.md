# Training Eval Design

## 1. Context and Goal

This design adds a new `training_eval` capability under `toolkits/` for profiling the embodied FSDP training update path.

The goal is to measure how different `model + env` combinations behave under:

- different `micro_batch_size` values
- MPS quota changes per GPU rank
- pre-created MIG device layouts

This benchmark is training-only. It does not run rollout collection, environment stepping, or reward computation during measurement. It reuses the real embodied FSDP update logic so the measured path matches production training as closely as possible.

## 2. Confirmed Product Decisions

- Add a separate `toolkits/training_eval` package.
- Keep rollout benchmark behavior unchanged.
- Support embodied FSDP only in v1.
- Do not add TP or Megatron support.
- Measure single micro-batch update performance, not full epoch throughput.
- Use synthetic random trajectories whose tensor shapes and nested fields match real training input.
- Keep the measured path on the real training update chain: trajectory batch conversion, preprocessing, forward, backward, optimizer step.
- Profile GPU memory and trajectory throughput as the primary outputs.

## 3. Scope

### In Scope

- Training-only benchmark entrypoint and case matrix.
- Synthetic trajectory generation for embodied training schemas.
- Real FSDP actor update execution.
- MPS and MIG resource binding for rank-local GPU workers.
- Per-case and summary reporting.

### Non-Goals

- No rollout collection.
- No environment stepping during measurement.
- No TP, PP, or Megatron support.
- No changes to `toolkits/rollout_eval` runtime or CLI behavior.
- No attempt to simulate environment physics or reward signals.
- No new model architecture work.

## 4. Architecture

### 4.1 High-Level Structure

Add a new benchmark stack that mirrors the rollout benchmark shape, but stays isolated:

- `toolkits/training_eval/run.py`
  - CLI entrypoint.
- `toolkits/training_eval/scenarios.py`
  - case matrix expansion.
- `toolkits/training_eval/trajectory_generator.py`
  - synthetic trajectory and batch template generation.
- `toolkits/training_eval/runner.py`
  - case execution, worker launch, measurement, aggregation.
- `toolkits/training_eval/resource_binding.py`
  - MPS/MIG process env preparation.
- `toolkits/training_eval/reporting.py`
  - per-case and summary reports.
- `toolkits/training_eval/types.py`
  - request, case, metrics, and schema dataclasses.

The implementation should use existing embodied training code paths rather than reimplementing the update logic in the benchmark package.

### 4.2 Execution Model

Each case:

1. Loads a real RLinf embodied training config.
2. Applies the case-specific `env_type`, `model_type`, `micro_batch_size`, and resource binding.
3. Builds a synthetic trajectory pool whose shapes match the target model/env schema.
4. Converts trajectories into the same batch format used by embodied FSDP training.
5. Runs the real FSDP actor update path.
6. Records wall time and GPU memory peaks.

## 5. Supported Case Matrix

The benchmark matrix is defined by:

- `preset`:
  - `model + env` pair
  - optionally a schema adapter for trajectory generation
- `micro_batch_size`
- `resource profile`
  - MPS quota profile
  - MIG device profile

The first version keeps the resource families simple and explicit. It does not add TP as an extra dimension.

## 6. Synthetic Trajectory Design

### 6.1 Data Contract

The generator must produce trajectory objects that can flow through the same conversion path used by embodied training:

- `Trajectory`
- `convert_trajectories_to_batch(...)`
- `EmbodiedFSDPActor.recv_rollout_trajectories(...)`
- `EmbodiedFSDPActor.run_training()`

The synthetic payload must match the real training shape conventions:

- tensor ranks and batch axes
- nested dict keys
- dtype expectations
- optional fields that may be `None`

### 6.2 Shape Semantics

The benchmark should generate one fixed synthetic global batch per measured update step.

That batch size should be chosen so a measured update corresponds to one micro-batch worth of work per rank in benchmark mode. This keeps the measurement aligned with the requested single-micro performance view.

The generator should preserve the same leading-dimension rules as real trajectory conversion, including the `T x B x ...` layout used by embodied training batches.

### 6.3 Template Source

The generator may derive template shapes from:

- the loaded config
- one real `env.reset()` call when an env observation schema is needed
- model-specific schema adapters for extra fields used by the actor

The measured run itself must not step the environment.

## 7. Training Execution Semantics

### 7.1 Real Update Path

The benchmark should not call a simplified loss-only microbenchmark.

Instead it must execute the real embodied FSDP actor update path, including:

- batch preprocessing
- advantage / return handling when required by the model path
- forward and backward
- optimizer step
- gradient scaler / FSDP handling when enabled by config

### 7.2 Micro Batch Control

`micro_batch_size` is the primary sweep dimension.

For benchmark mode:

- keep the update path real
- avoid sweeping TP
- keep the measurement focused on one micro-batch update shape
- do not broaden the benchmark into full training epoch throughput

## 8. Resource Binding

### 8.1 MPS

- Use per-rank MPS quota injection.
- Do not manage the MPS daemon lifecycle.
- Allow the case to specify either a uniform quota or an explicit per-rank quota list.
- Capture the exact quota used in the case metadata.

### 8.2 MIG

- Consume pre-created MIG UUIDs only.
- Bind one rank to one visible MIG device.
- Support explicit rank-to-device mapping.
- Do not create or destroy MIG instances.

### 8.3 Failure Policy

If the requested resource profile is unavailable, invalid, or incomplete, the case should be marked skipped with a clear reason rather than crashing the whole matrix.

## 9. Metrics and Reporting

### 9.1 Primary Metrics

- `trajectories_per_sec`
- `peak_gpu_memory_allocated_gb`
- `peak_gpu_memory_reserved_gb`

### 9.2 Secondary Metrics

- step wall time
- per-rank peak memory
- optional latency percentiles for the update step

### 9.3 Aggregation Rules

- Throughput is computed from measured synthetic trajectories processed per wall-clock second.
- Peak GPU memory is tracked per rank and reported both per-rank and as aggregate maxima.
- Summary tables should sort by throughput first, with memory shown next.

### 9.4 Report Artifacts

Per case:

- `case_report.json`
- `case_report.md`
- `case_meta.json`

Global:

- `summary.json`
- `summary.md`

## 10. Testing Strategy

### Unit Tests

- case matrix expansion
- trajectory schema generation
- MPS and MIG resource binding normalization
- report aggregation and summary sorting
- shape compatibility between generated trajectories and the embodied batch converter

### Lightweight Integration Tests

- one minimal FSDP training smoke case with synthetic trajectories
- one resource-bound case for MPS or MIG, guarded by GPU availability

### Done Criteria

- `training_eval` runs independently of rollout evaluation.
- embodied FSDP training updates run on synthetic trajectories with real batch shapes.
- micro batch sweeps report throughput and GPU memory.
- MPS and MIG cases are supported without TP.
- existing rollout benchmark behavior remains unchanged.
