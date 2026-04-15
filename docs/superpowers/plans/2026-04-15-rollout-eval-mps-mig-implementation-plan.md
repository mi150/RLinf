# Rollout Eval MPS/MIG Profiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a benchmark layer in `toolkits/rollout_eval` to profile throughput/latency across MPS and MIG resource partition scenarios, including concurrent, env-only, and model-only modes.

**Architecture:** Keep `toolkits.rollout_eval.run` backward compatible as single-run executor, and add `toolkits.rollout_eval.benchmark` as a new orchestration package. The benchmark package expands scenario matrices, runs dual-process pipeline or single-side loops, binds MPS/MIG resources per case, and emits per-case plus summary reports.

**Tech Stack:** Python 3, PyTorch, existing Hydra config bridge, pytest, Ruff.

---

## File Structure Map

- Create: `toolkits/rollout_eval/benchmark/__init__.py`
- Create: `toolkits/rollout_eval/benchmark/run.py` (CLI entry)
- Create: `toolkits/rollout_eval/benchmark/types.py` (dataclasses for case/metrics/result)
- Create: `toolkits/rollout_eval/benchmark/scenarios.py` (scenario set + matrix expansion)
- Create: `toolkits/rollout_eval/benchmark/metrics.py` (throughput + avg/p50/p95 aggregation)
- Create: `toolkits/rollout_eval/benchmark/single_runner.py` (env-only/model-only runtimes)
- Create: `toolkits/rollout_eval/benchmark/pipeline_runner.py` (dual-process concurrent runtime)
- Create: `toolkits/rollout_eval/benchmark/resource_binding.py` (MPS/MIG env injection)
- Create: `toolkits/rollout_eval/benchmark/orchestrator.py` (case scheduling, status control)
- Create: `toolkits/rollout_eval/benchmark/reporting.py` (case + summary writers)
- Modify: `toolkits/rollout_eval/adapters/__init__.py` (safe exports reused by benchmark)
- Modify: `toolkits/rollout_eval/run.py` (no behavior change; optional utility extraction only if needed)
- Create: `tests/unit_tests/test_rollout_eval_benchmark_scenarios.py`
- Create: `tests/unit_tests/test_rollout_eval_benchmark_metrics.py`
- Create: `tests/unit_tests/test_rollout_eval_benchmark_resource_binding.py`
- Create: `tests/unit_tests/test_rollout_eval_benchmark_orchestrator.py`
- Create: `tests/unit_tests/test_rollout_eval_benchmark_single_runner.py`
- Create: `tests/unit_tests/test_rollout_eval_benchmark_pipeline_runner.py`
- Modify: `toolkits/rollout_eval/profile.sh` (add benchmark invocation examples)

### Task 1: Scaffold Benchmark Package and CLI

**Files:**
- Create: `toolkits/rollout_eval/benchmark/__init__.py`
- Create: `toolkits/rollout_eval/benchmark/run.py`
- Create: `toolkits/rollout_eval/benchmark/types.py`
- Test: `tests/unit_tests/test_rollout_eval_benchmark_scenarios.py`

- [ ] **Step 1: Write the failing test for CLI argument parsing and defaults**

```python
def test_benchmark_cli_defaults():
    args = parse_args(["--config-path", "examples/embodiment/config", "--config-name", "x"])
    assert args.pipeline == "process"
    assert "env_only_mps" in args.scenario_set
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/unit_tests/test_rollout_eval_benchmark_scenarios.py -k cli_defaults`
Expected: FAIL with import or missing parser errors.

- [ ] **Step 3: Implement minimal CLI and type stubs**

```python
# run.py
parser.add_argument("--scenario-set", default="concurrent_mps,concurrent_mig,env_only_mps,model_only_mps,env_only_mig,model_only_mig")
parser.add_argument("--pipeline", default="process", choices=["process"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest -q tests/unit_tests/test_rollout_eval_benchmark_scenarios.py -k cli_defaults`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add toolkits/rollout_eval/benchmark/__init__.py toolkits/rollout_eval/benchmark/run.py toolkits/rollout_eval/benchmark/types.py tests/unit_tests/test_rollout_eval_benchmark_scenarios.py
git commit -s -m "feat: scaffold rollout eval benchmark cli"
```

### Task 2: Implement Scenario Matrix Expansion

**Files:**
- Create: `toolkits/rollout_eval/benchmark/scenarios.py`
- Modify: `toolkits/rollout_eval/benchmark/types.py`
- Test: `tests/unit_tests/test_rollout_eval_benchmark_scenarios.py`

- [ ] **Step 1: Write failing tests for six scenario classes and deterministic case IDs**

```python
def test_expand_all_scenarios_generates_expected_classes():
    cases = expand_cases(...)
    assert {c.scenario for c in cases} == {
        "concurrent_mps", "concurrent_mig", "env_only_mps",
        "model_only_mps", "env_only_mig", "model_only_mig",
    }
    assert len({c.case_id for c in cases}) == len(cases)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest -q tests/unit_tests/test_rollout_eval_benchmark_scenarios.py -k expand`
Expected: FAIL.

- [ ] **Step 3: Implement `expand_cases` with preset × resource dimensions**

```python
def expand_cases(req: BenchmarkRequest) -> list[BenchmarkCase]:
    # parse scenario_set, mps_sm, mig_devices, preset list
    # return deterministic sorted case list
```

- [ ] **Step 4: Run scenario tests**

Run: `pytest -q tests/unit_tests/test_rollout_eval_benchmark_scenarios.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add toolkits/rollout_eval/benchmark/scenarios.py toolkits/rollout_eval/benchmark/types.py tests/unit_tests/test_rollout_eval_benchmark_scenarios.py
git commit -s -m "feat: add benchmark scenario matrix expansion"
```

### Task 3: Implement Metrics Aggregation (Throughput + avg/p50/p95)

**Files:**
- Create: `toolkits/rollout_eval/benchmark/metrics.py`
- Modify: `toolkits/rollout_eval/benchmark/types.py`
- Test: `tests/unit_tests/test_rollout_eval_benchmark_metrics.py`

- [ ] **Step 1: Write failing tests for throughput math and latency summary**

```python
def test_latency_summary_avg_p50_p95():
    s = summarize_latency_ms([1.0, 2.0, 10.0])
    assert s.avg_ms == pytest.approx(13.0 / 3)
    assert s.p50_ms == pytest.approx(2.0)
    assert s.p95_ms >= s.p50_ms
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest -q tests/unit_tests/test_rollout_eval_benchmark_metrics.py`
Expected: FAIL.

- [ ] **Step 3: Implement metric utilities**

```python
def throughput(count: int, seconds: float) -> float: ...
def summarize_latency_ms(samples: list[float]) -> LatencySummary: ...
```

- [ ] **Step 4: Run tests and verify pass**

Run: `pytest -q tests/unit_tests/test_rollout_eval_benchmark_metrics.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add toolkits/rollout_eval/benchmark/metrics.py toolkits/rollout_eval/benchmark/types.py tests/unit_tests/test_rollout_eval_benchmark_metrics.py
git commit -s -m "feat: add benchmark throughput and latency aggregation"
```

### Task 4: Implement Resource Binding (MPS/MIG)

**Files:**
- Create: `toolkits/rollout_eval/benchmark/resource_binding.py`
- Test: `tests/unit_tests/test_rollout_eval_benchmark_resource_binding.py`

- [ ] **Step 1: Write failing tests for env var injection and MIG visibility**

```python
def test_build_mig_env_sets_cuda_visible_devices():
    env = build_process_env(base={}, mig_uuid="MIG-abc", mps_pct=None)
    assert env["CUDA_VISIBLE_DEVICES"] == "MIG-abc"
```

- [ ] **Step 2: Run tests and confirm fail**

Run: `pytest -q tests/unit_tests/test_rollout_eval_benchmark_resource_binding.py`
Expected: FAIL.

- [ ] **Step 3: Implement non-destructive binding helpers**

```python
def build_process_env(base: dict[str, str], mig_uuid: str | None, mps_pct: int | None) -> dict[str, str]:
    # no daemon or MIG lifecycle control
```

- [ ] **Step 4: Run tests and confirm pass**

Run: `pytest -q tests/unit_tests/test_rollout_eval_benchmark_resource_binding.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add toolkits/rollout_eval/benchmark/resource_binding.py tests/unit_tests/test_rollout_eval_benchmark_resource_binding.py
git commit -s -m "feat: add benchmark mps mig resource binding helpers"
```

### Task 5: Implement Single-Side Runners (env-only/model-only)

**Files:**
- Create: `toolkits/rollout_eval/benchmark/single_runner.py`
- Modify: `toolkits/rollout_eval/adapters/__init__.py`
- Test: `tests/unit_tests/test_rollout_eval_benchmark_single_runner.py`

- [ ] **Step 1: Write failing tests for env-only random-action and model-only dummy-from-reset**

```python
def test_model_only_builds_dummy_obs_from_reset_template(...):
    result = run_model_only_case(...)
    assert result.metrics.model_infers_per_sec > 0
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest -q tests/unit_tests/test_rollout_eval_benchmark_single_runner.py`
Expected: FAIL.

- [ ] **Step 3: Implement single-side loops with sample capture**

```python
def run_env_only_case(...): ...
def run_model_only_case(...): ...
```

- [ ] **Step 4: Run tests and verify pass**

Run: `pytest -q tests/unit_tests/test_rollout_eval_benchmark_single_runner.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add toolkits/rollout_eval/benchmark/single_runner.py toolkits/rollout_eval/adapters/__init__.py tests/unit_tests/test_rollout_eval_benchmark_single_runner.py
git commit -s -m "feat: add env-only and model-only benchmark runners"
```

### Task 6: Implement Dual-Process Pipeline Runner

**Files:**
- Create: `toolkits/rollout_eval/benchmark/pipeline_runner.py`
- Test: `tests/unit_tests/test_rollout_eval_benchmark_pipeline_runner.py`

- [ ] **Step 1: Write failing tests for queue liveness and measurement counts**

```python
def test_pipeline_runner_records_env_and_model_counts(...):
    result = run_concurrent_case(...)
    assert result.metrics.pipeline_samples_per_sec > 0
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest -q tests/unit_tests/test_rollout_eval_benchmark_pipeline_runner.py`
Expected: FAIL.

- [ ] **Step 3: Implement `sim process` + `model process` + orchestrated windows**

```python
def run_concurrent_case(...):
    # multiprocessing queues, warmup, measure, stop
```

- [ ] **Step 4: Run tests and verify pass**

Run: `pytest -q tests/unit_tests/test_rollout_eval_benchmark_pipeline_runner.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add toolkits/rollout_eval/benchmark/pipeline_runner.py tests/unit_tests/test_rollout_eval_benchmark_pipeline_runner.py
git commit -s -m "feat: add dual-process concurrent benchmark runner"
```

### Task 7: Implement Orchestrator and Reporting

**Files:**
- Create: `toolkits/rollout_eval/benchmark/orchestrator.py`
- Create: `toolkits/rollout_eval/benchmark/reporting.py`
- Test: `tests/unit_tests/test_rollout_eval_benchmark_orchestrator.py`
- Modify: `toolkits/rollout_eval/benchmark/run.py`

- [ ] **Step 1: Write failing tests for pass/failed/skipped status and summary outputs**

```python
def test_orchestrator_keeps_running_after_case_failure(tmp_path):
    summary = run_benchmark(...)
    assert summary.total_cases == 3
    assert summary.failed_cases == 1
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest -q tests/unit_tests/test_rollout_eval_benchmark_orchestrator.py`
Expected: FAIL.

- [ ] **Step 3: Implement case execution loop and report writers**

```python
def run_benchmark(...):
    # preflight, dispatch runner by scenario, write case_report/case_meta, write summary
```

- [ ] **Step 4: Run tests and verify pass**

Run: `pytest -q tests/unit_tests/test_rollout_eval_benchmark_orchestrator.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add toolkits/rollout_eval/benchmark/orchestrator.py toolkits/rollout_eval/benchmark/reporting.py toolkits/rollout_eval/benchmark/run.py tests/unit_tests/test_rollout_eval_benchmark_orchestrator.py
git commit -s -m "feat: add benchmark orchestrator and reports"
```

### Task 8: Docs/Examples and End-to-End Verification

**Files:**
- Modify: `toolkits/rollout_eval/profile.sh`
- Modify: `toolkits/rollout_eval/benchmark/run.py` (help text polish)
- Optional Create: `docs/rollout_eval_benchmark_mps_mig.md` (if team prefers dedicated doc)

- [ ] **Step 1: Write failing smoke test command list (script-level validation)**

Run target commands (expect no argument parsing errors):
- `python -m toolkits.rollout_eval.benchmark.run --help`
- minimal `env_only_mps` invocation
- minimal `model_only_mps` invocation

- [ ] **Step 2: Add examples to `profile.sh` and CLI help**

```bash
# include one MPS concurrent and one MIG env-only example
```

- [ ] **Step 3: Run full benchmark unit test suite**

Run: `pytest -q tests/unit_tests/test_rollout_eval_benchmark_*.py`
Expected: PASS.

- [ ] **Step 4: Run lint checks on touched files**

Run: `ruff check toolkits/rollout_eval/benchmark tests/unit_tests/test_rollout_eval_benchmark_*.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add toolkits/rollout_eval/profile.sh toolkits/rollout_eval/benchmark/run.py tests/unit_tests/test_rollout_eval_benchmark_*.py
git commit -s -m "docs: add benchmark usage examples and finalize test coverage"
```

## Global Verification Gate (before merge)

- [ ] Run: `pytest -q tests/unit_tests/test_rollout_eval_benchmark_*.py`
- [ ] Run: `pytest -q tests/unit_tests/test_rollout_eval_loop.py tests/unit_tests/test_rollout_eval_profiler.py`
- [ ] Run: `ruff check toolkits/rollout_eval tests/unit_tests`
- [ ] Validate that `python -m toolkits.rollout_eval.run --help` still works unchanged.

## Risks and Mitigations

- Process deadlock in concurrent pipeline.
  - Mitigation: bounded queue + timeout + explicit stop sentinels + unit test for liveness.
- Metric skew from warmup leakage.
  - Mitigation: strict warmup/measure phase separation and independent counters.
- Environment-specific runtime dependencies (Behavior/Isaac).
  - Mitigation: unit tests rely on stubs/mocks; hardware/manual matrix validation documented separately.

## Rollback Plan

If issues appear after integration:
- Keep `toolkits.rollout_eval.run` as stable fallback.
- Disable benchmark entry by feature flag or stop invoking `toolkits.rollout_eval.benchmark.run` in scripts.
- Revert benchmark package commits independently (isolated path under `toolkits/rollout_eval/benchmark`).
