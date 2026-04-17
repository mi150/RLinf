# Remove Feature Cache While Preserving Toolkits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all feature-cache runtime behavior from RLinf while keeping `toolkits/rollout_eval` non-cache workflows functional and making `cache_eval` explicitly unsupported.

**Architecture:** Runtime files are restored to the pre-feature-cache baseline (`17bba2407552e710579acecd9c2baf3df57268bd`) to ensure true removal instead of soft-disable. `toolkits/rollout_eval/experiment` gets an explicit capability gate and unsupported report path for `cache_eval`, so baseline/action-replace phases continue to work. Regression is controlled by focused unit tests plus search-based guards.

**Tech Stack:** Python 3.11, pytest, Hydra/OmegaConf, git, Ruff.

---

## File Structure Map

### Runtime rollback set (restore from `17bba2407552e710579acecd9c2baf3df57268bd`)

- Modify: `rlinf/models/embodiment/gr00t/gr00t_action_model.py` (remove cache integration)
- Modify: `rlinf/models/embodiment/openpi/openpi_action_model.py` (remove cache integration)
- Modify: `rlinf/models/embodiment/openvla_oft/official/openvla_oft_action_model.py` (remove cache integration)
- Modify: `rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py` (remove cache integration)
- Modify: `rlinf/workers/rollout/hf/huggingface_worker.py` (remove cache config/logging/invalidation paths)
- Modify: `rlinf/workers/rollout/hf/async_huggingface_worker.py` (remove cache metrics aggregation path)
- Modify: `examples/embodiment/config/maniskill_ppo_openpi.yaml` (remove feature-cache config block)
- Delete: `rlinf/models/embodiment/feature_cache.py`
- Delete: `tests/unit_tests/test_feature_cache.py`
- Delete: `tests/unit_tests/test_feature_cache_integration.py`
- Delete: `examples/embodiment/config/libero_spatial_ppo_gr00t_feature_cache_similarity.yaml`

### Toolkits compatibility set

- Modify: `toolkits/rollout_eval/experiment/types.py` (cache config type decoupled from runtime module)
- Modify: `toolkits/rollout_eval/experiment/cache_eval.py` (no direct runtime cache class dependency)
- Modify: `toolkits/rollout_eval/experiment/run_experiment.py` (capability detection + unsupported path)
- Modify: `toolkits/rollout_eval/experiment/reporting.py` (unsupported cache report writer)
- Modify: `toolkits/rollout_eval/experiment/README.md` (document no-cache baseline behavior)

### Tests

- Modify: `tests/unit_tests/test_experiment_cache_eval.py`
- Modify: `tests/unit_tests/test_experiment_orchestrator.py`
- Create: `tests/unit_tests/test_no_feature_cache_runtime.py`

---

### Task 1: Add Failing Tests for No-Cache Toolkits Behavior

**Files:**
- Modify: `tests/unit_tests/test_experiment_orchestrator.py`
- Modify: `tests/unit_tests/test_experiment_cache_eval.py`

- [ ] **Step 1: Write failing test for unsupported `cache_eval` report path**

```python
# tests/unit_tests/test_experiment_orchestrator.py

def test_cache_eval_phase_reports_unsupported_when_feature_cache_absent(
    tmp_path,
    monkeypatch,
):
    import json
    from toolkits.rollout_eval.experiment.run_experiment import run_experiment
    from toolkits.rollout_eval.experiment.types import ExperimentConfig

    cfg = ExperimentConfig(
        eval_runtime=_make_runtime(total_steps=3),
        phases=["cache_eval", "baseline"],
        seeds=[42],
        output_dir=str(tmp_path),
    )

    monkeypatch.setattr(
        "toolkits.rollout_eval.experiment.run_experiment._is_feature_cache_available",
        lambda: False,
    )

    run_experiment(cfg, _make_mock_env(), _make_mock_model())

    cache_report = tmp_path / "reports" / "phase2_cache_eval.json"
    baseline_report = tmp_path / "reports" / "phase1_baseline.json"
    assert cache_report.exists()
    assert baseline_report.exists()

    payload = json.loads(cache_report.read_text())
    assert payload["phase"] == "cache_eval"
    assert payload["status"] == "unsupported"
    assert "feature cache is not available" in payload["reason"].lower()
```

- [ ] **Step 2: Write failing test for cache adapter construction without runtime cache module**

```python
# tests/unit_tests/test_experiment_cache_eval.py

def test_cache_adapter_raises_when_runtime_cache_unavailable(monkeypatch):
    inner = _make_mock_inner_model()

    monkeypatch.setattr(
        "toolkits.rollout_eval.experiment.cache_eval._is_feature_cache_runtime_available",
        lambda: False,
    )

    with pytest.raises(RuntimeError, match="feature cache runtime is unavailable"):
        CacheAwareModelAdapter(inner, cache_config={"enabled": True, "mode": "naive"})
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest -q tests/unit_tests/test_experiment_orchestrator.py -k unsupported_when_feature_cache_absent`  
Expected: FAIL (`_is_feature_cache_available` missing / report schema mismatch).

Run: `pytest -q tests/unit_tests/test_experiment_cache_eval.py -k runtime_cache_unavailable`  
Expected: FAIL (`cache_config` type/path mismatch).

- [ ] **Step 4: Commit failing tests**

```bash
git add tests/unit_tests/test_experiment_orchestrator.py tests/unit_tests/test_experiment_cache_eval.py
git commit -s -m "test: add failing no-cache experiment compatibility tests"
```

### Task 2: Implement Toolkits No-Cache Compatibility

**Files:**
- Modify: `toolkits/rollout_eval/experiment/types.py`
- Modify: `toolkits/rollout_eval/experiment/cache_eval.py`
- Modify: `toolkits/rollout_eval/experiment/run_experiment.py`
- Modify: `toolkits/rollout_eval/experiment/reporting.py`
- Modify: `tests/unit_tests/test_experiment_orchestrator.py`
- Modify: `tests/unit_tests/test_experiment_cache_eval.py`

- [ ] **Step 1: Replace direct `FeatureCacheConfig` import in experiment types**

```python
# toolkits/rollout_eval/experiment/types.py
from typing import Any

@dataclass
class ExperimentConfig:
    ...
    cache_config: dict[str, Any] | None = None
```

- [ ] **Step 2: Add runtime capability check + unsupported report function**

```python
# toolkits/rollout_eval/experiment/run_experiment.py

def _is_feature_cache_available() -> bool:
    try:
        from rlinf.models.embodiment.feature_cache import FeatureCache  # noqa: F401
        return True
    except Exception:
        return False


def _run_phase_cache_eval(...):
    from toolkits.rollout_eval.experiment.reporting import dump_cache_report_unsupported

    if not _is_feature_cache_available():
        dump_cache_report_unsupported(
            output_dir=cfg.output_dir,
            reason="Feature cache is not available in this RLinf baseline.",
        )
        logger.warning("Phase 2 skipped: feature cache is not available in this baseline")
        return
```

```python
# toolkits/rollout_eval/experiment/reporting.py

def dump_cache_report_unsupported(
    output_dir: str | Path,
    reason: str,
) -> Path:
    report = {
        "phase": "cache_eval",
        "status": "unsupported",
        "reason": reason,
    }
    path = Path(output_dir) / "reports" / "phase2_cache_eval.json"
    _write_json(report, path)
    return path
```

- [ ] **Step 3: Decouple cache adapter from runtime dataclasses and protect construction**

```python
# toolkits/rollout_eval/experiment/cache_eval.py

def _is_feature_cache_runtime_available() -> bool:
    try:
        from rlinf.models.embodiment.feature_cache import FeatureCache  # noqa: F401
        return True
    except Exception:
        return False


class CacheAwareModelAdapter:
    def __init__(self, inner: GenericModelAdapter, cache_config: dict[str, Any]):
        if not _is_feature_cache_runtime_available():
            raise RuntimeError("Feature cache runtime is unavailable in current RLinf build")
        ...
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest -q tests/unit_tests/test_experiment_orchestrator.py -k unsupported_when_feature_cache_absent`  
Expected: PASS.

Run: `pytest -q tests/unit_tests/test_experiment_cache_eval.py -k runtime_cache_unavailable`  
Expected: PASS.

Run: `pytest -q tests/unit_tests/test_experiment_*.py`  
Expected: PASS.

- [ ] **Step 5: Commit toolkits compatibility implementation**

```bash
git add toolkits/rollout_eval/experiment/types.py toolkits/rollout_eval/experiment/cache_eval.py toolkits/rollout_eval/experiment/run_experiment.py toolkits/rollout_eval/experiment/reporting.py tests/unit_tests/test_experiment_orchestrator.py tests/unit_tests/test_experiment_cache_eval.py
git commit -s -m "fix(toolkits): support no-feature-cache baseline with explicit cache_eval unsupported path"
```

### Task 3: Remove Runtime Feature-Cache Code by Restoring Pre-Cache Baseline

**Files:**
- Modify: `rlinf/models/embodiment/gr00t/gr00t_action_model.py`
- Modify: `rlinf/models/embodiment/openpi/openpi_action_model.py`
- Modify: `rlinf/models/embodiment/openvla_oft/official/openvla_oft_action_model.py`
- Modify: `rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py`
- Modify: `rlinf/workers/rollout/hf/huggingface_worker.py`
- Modify: `rlinf/workers/rollout/hf/async_huggingface_worker.py`
- Modify: `examples/embodiment/config/maniskill_ppo_openpi.yaml`
- Delete: `rlinf/models/embodiment/feature_cache.py`
- Delete: `examples/embodiment/config/libero_spatial_ppo_gr00t_feature_cache_similarity.yaml`

- [ ] **Step 1: Add failing guard test to enforce no runtime cache references**

```python
# tests/unit_tests/test_no_feature_cache_runtime.py
from pathlib import Path

RUNTIME_FILES = [
    "rlinf/models/embodiment/openpi/openpi_action_model.py",
    "rlinf/models/embodiment/gr00t/gr00t_action_model.py",
    "rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py",
    "rlinf/workers/rollout/hf/huggingface_worker.py",
    "rlinf/workers/rollout/hf/async_huggingface_worker.py",
]


def test_runtime_files_do_not_reference_feature_cache():
    repo = Path(__file__).resolve().parents[2]
    for rel in RUNTIME_FILES:
        text = (repo / rel).read_text(encoding="utf-8")
        assert "feature_cache" not in text
```

- [ ] **Step 2: Run guard test to verify failure**

Run: `pytest -q tests/unit_tests/test_no_feature_cache_runtime.py`  
Expected: FAIL (current files still contain `feature_cache`).

- [ ] **Step 3: Restore runtime files to pre-cache baseline and remove cache-only files**

```bash
BASE=17bba2407552e710579acecd9c2baf3df57268bd
git restore --source "$BASE" -- \
  rlinf/models/embodiment/gr00t/gr00t_action_model.py \
  rlinf/models/embodiment/openpi/openpi_action_model.py \
  rlinf/models/embodiment/openvla_oft/official/openvla_oft_action_model.py \
  rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  rlinf/workers/rollout/hf/async_huggingface_worker.py \
  examples/embodiment/config/maniskill_ppo_openpi.yaml

git rm -f rlinf/models/embodiment/feature_cache.py
git rm -f examples/embodiment/config/libero_spatial_ppo_gr00t_feature_cache_similarity.yaml
```

- [ ] **Step 4: Run runtime guard + focused unit tests**

Run: `pytest -q tests/unit_tests/test_no_feature_cache_runtime.py`  
Expected: PASS.

Run: `pytest -q tests/unit_tests/test_rollout_eval_benchmark_*.py`  
Expected: PASS.

- [ ] **Step 5: Commit runtime rollback**

```bash
git add tests/unit_tests/test_no_feature_cache_runtime.py rlinf/models/embodiment/gr00t/gr00t_action_model.py rlinf/models/embodiment/openpi/openpi_action_model.py rlinf/models/embodiment/openvla_oft/official/openvla_oft_action_model.py rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py rlinf/workers/rollout/hf/huggingface_worker.py rlinf/workers/rollout/hf/async_huggingface_worker.py examples/embodiment/config/maniskill_ppo_openpi.yaml
git commit -s -m "refactor(runtime): remove feature cache integrations from models and rollout workers"
```

### Task 4: Remove Cache-Specific Tests and Fix Remaining Test Surface

**Files:**
- Delete: `tests/unit_tests/test_feature_cache.py`
- Delete: `tests/unit_tests/test_feature_cache_integration.py`
- Modify: `tests/unit_tests/test_experiment_cache_eval.py`

- [ ] **Step 1: Remove cache-specific tests that no longer apply**

```bash
git rm -f tests/unit_tests/test_feature_cache.py
git rm -f tests/unit_tests/test_feature_cache_integration.py
```

- [ ] **Step 2: Rewrite cache-eval tests to cover unsupported/no-cache contract**

```python
# tests/unit_tests/test_experiment_cache_eval.py

def test_run_cache_eval_unavailable_runtime_returns_unsupported(monkeypatch, tmp_path):
    from toolkits.rollout_eval.experiment.reporting import dump_cache_report_unsupported

    path = dump_cache_report_unsupported(
        output_dir=tmp_path,
        reason="Feature cache is not available in this RLinf baseline.",
    )
    assert path.exists()
```

- [ ] **Step 3: Run target tests**

Run: `pytest -q tests/unit_tests/test_experiment_cache_eval.py tests/unit_tests/test_experiment_orchestrator.py tests/unit_tests/test_no_feature_cache_runtime.py`  
Expected: PASS.

- [ ] **Step 4: Commit test cleanup**

```bash
git add tests/unit_tests/test_experiment_cache_eval.py tests/unit_tests/test_experiment_orchestrator.py tests/unit_tests/test_no_feature_cache_runtime.py
git commit -s -m "test: align experiment and runtime tests with no-feature-cache baseline"
```

### Task 5: Update Toolkits Documentation for No-Cache Baseline

**Files:**
- Modify: `toolkits/rollout_eval/experiment/README.md`

- [ ] **Step 1: Add explicit compatibility note and behavior contract**

```md
## Feature Cache Phase Compatibility

This RLinf baseline does not include runtime feature-cache support.

- `baseline` and `action_replace` phases are fully supported.
- `cache_eval` phase writes `reports/phase2_cache_eval.json` with:
  - `status: "unsupported"`
  - a human-readable `reason`
```

- [ ] **Step 2: Remove/adjust cache-mode claims that imply support**

```md
# replace
支持的 cache mode: ...

# with
`cache_eval` requires a cache-enabled RLinf build. In this baseline it is reported as unsupported.
```

- [ ] **Step 3: Run doc/link sanity checks (lightweight)**

Run: `rg -n "cache mode|FeatureCache|cache_eval" toolkits/rollout_eval/experiment/README.md`  
Expected: shows updated unsupported wording, no contradictory claims.

- [ ] **Step 4: Commit docs update**

```bash
git add toolkits/rollout_eval/experiment/README.md
git commit -s -m "docs(toolkits): document cache_eval unsupported behavior in no-cache baseline"
```

### Task 6: Final Verification and Handoff

**Files:**
- Verify repository state only.

- [ ] **Step 1: Run complete verification set**

Run: `pytest -q tests/unit_tests/test_experiment_*.py`  
Expected: PASS.

Run: `pytest -q tests/unit_tests/test_rollout_eval_benchmark_*.py`  
Expected: PASS.

Run: `pytest -q tests/unit_tests/test_no_feature_cache_runtime.py`  
Expected: PASS.

Run: `rg -n "feature_cache" rlinf/models/embodiment rlinf/workers/rollout/hf examples/embodiment/config | head -n 20`  
Expected: no runtime/config execution-path references.

- [ ] **Step 2: Review and summarize resulting API/behavior changes**

```bash
git status --short
git log --oneline -n 8
```

Expected: only planned files changed; commits reflect test-first progression.

- [ ] **Step 3: Prepare PR summary block**

```md
## Summary
- Remove feature-cache runtime integrations from embodied models and rollout workers.
- Keep rollout_eval toolkits operational in no-cache baseline.
- Make cache_eval phase explicit unsupported with structured report artifact.

## Validation
- pytest -q tests/unit_tests/test_experiment_*.py
- pytest -q tests/unit_tests/test_rollout_eval_benchmark_*.py
- pytest -q tests/unit_tests/test_no_feature_cache_runtime.py
```
