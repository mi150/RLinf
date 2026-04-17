# Remove Feature Cache While Preserving Toolkits Functionality

**Date:** 2026-04-17  
**Status:** Proposed  
**Scope:** Remove feature-cache behavior from RLinf runtime paths, while keeping `toolkits/` usable and correct.

## 1. Objective

Restore RLinf to a no-feature-cache runtime baseline. Any previous feature-cache execution path in model/workers/config should be removed. `toolkits/rollout_eval` must continue working, with graceful degradation for cache-only phases.

## 2. Non-Goals

- No redesign of benchmark/experiment UX beyond compatibility messaging.
- No unrelated refactors in scheduler/env/model architecture.
- No performance tuning outside removal fallout fixes.

## 3. Baseline Definition

"No-feature-cache baseline" means:

- No cache lookup/put/invalidate behavior in model inference and rollout workers.
- No feature-cache-only config fields required for normal run.
- No cache-dependent assertions in runtime metrics path.
- Cache-specific experiment phase is explicitly unsupported (skip/fail-fast with clear reason), not partially functional.

## 4. Impacted Areas

- Runtime/model integration
  - `rlinf/models/embodiment/openpi/openpi_action_model.py`
  - `rlinf/models/embodiment/gr00t/gr00t_action_model.py`
  - `rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py`
  - `rlinf/models/embodiment/openvla_oft/official/openvla_oft_action_model.py`
- Rollout workers / metrics
  - `rlinf/workers/rollout/hf/huggingface_worker.py`
  - `rlinf/workers/rollout/hf/async_huggingface_worker.py`
- Cache utility/tests/config
  - `rlinf/models/embodiment/feature_cache.py`
  - `tests/unit_tests/test_feature_cache.py`
  - `tests/unit_tests/test_feature_cache_integration.py`
  - cache-specific config YAMLs under `examples/embodiment/config/`
- Toolkits compatibility
  - `toolkits/rollout_eval/experiment/cache_eval.py`
  - `toolkits/rollout_eval/experiment/run_experiment.py`
  - `toolkits/rollout_eval/experiment/types.py`
  - `toolkits/rollout_eval/experiment/README.md`

## 5. Design Approach (Recommended)

### 5.1 Runtime Removal

- Remove feature-cache runtime integration from model/worker code paths.
- Remove or neutralize `FeatureCacheConfig` imports in core runtime code.
- Ensure model inference behavior is direct-compute only.

### 5.2 Toolkits Degradation Contract

- `toolkits` remains importable/executable without feature cache.
- `cache_eval` phase behavior:
  - detect feature-cache unavailability at startup,
  - produce structured report status `unsupported` with reason,
  - do not crash other phases.
- `baseline` and `action_replace` phases remain unchanged.

### 5.3 Metrics Compatibility

- Remove hard dependency on `same_step_hits` and cache-only counters from runtime paths.
- In toolkits reports, cache metrics fields become optional or reported as `null`/`unsupported` when cache is absent.

## 6. Error Handling

- If user requests `cache_eval` in no-cache baseline:
  - log one explicit warning,
  - write report artifact with unsupported status,
  - continue remaining phases when configured.
- No silent fallback to fake cache behavior.

## 7. Testing Strategy

### 7.1 Unit

- Remove/adjust feature-cache unit tests to match no-cache baseline.
- Add/adjust toolkit tests to verify:
  - `cache_eval` phase unsupported path is deterministic and non-crashing,
  - `baseline` and `action_replace` phases still pass.

### 7.2 Integration / Smoke

- Smoke run `toolkits/rollout_eval/benchmark` matrix unit tests.
- Smoke run one embodied eval config for OpenPI and OpenVLA-OFT (minimal steps).

### 7.3 Regression Guard

- Search-based check in runtime critical files for remaining `feature_cache` execution references.
- Ensure no required YAML points to removed cache-only mode.

## 8. Rollout Order

1. Remove runtime cache integration (models/workers).
2. Make toolkits cache phase explicitly unsupported.
3. Update docs/examples/tests.
4. Run focused tests then broader toolkits suite.

## 9. Risks and Mitigations

- Risk: latent references to removed cache fields cause runtime `AttributeError`.  
  Mitigation: grep-based sweep + unit/smoke tests.
- Risk: toolkits cache phase becomes ambiguous for users.  
  Mitigation: explicit unsupported report + README note + CLI log.
- Risk: accidental breakage in non-cache rollout metrics.  
  Mitigation: keep metric keys stable where possible; only remove cache-only semantics.

## 10. Acceptance Criteria

- Runtime does not execute any feature-cache path.
- `toolkits/rollout_eval` remains usable for non-cache workflows.
- `cache_eval` phase handles no-cache baseline explicitly and safely.
- Relevant unit/smoke tests pass.
