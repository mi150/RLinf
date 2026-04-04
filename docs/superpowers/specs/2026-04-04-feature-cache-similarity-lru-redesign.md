# Feature Cache Redesign: Similarity-LRU Mode

**Date:** 2026-04-04
**Status:** Draft
**Scope:** `rlinf/models/embodiment/feature_cache.py`, model integration layers, rollout worker

## Problem Statement

The current feature cache uses a `dict[seed -> dict[step -> CacheEntry]]` storage structure that requires matching on `(seed, step)` pairs. The desired behavior is:

1. Cache lookup based purely on observation similarity (ignore seed and step)
2. Cross-global-step cache retention (backbone is frozen)
3. LRU eviction when cache exceeds a configurable max size
4. Fix existing bugs and integrate cache into OpenPI model

## Current Issues

1. **OpenPI has no cache integration** — `sample_actions()` accepts `env_seeds`/`step_indices`/`current_obs` but never uses them for cache lookup. The VLM prefix computation runs every time.
2. **`feature_cosine` is ineffective for GR00T and OpenVLA-OFT** — Both models' `get_vision_features()` return `None`, causing fallback to `obs_ssim`.
3. **Storage structure mismatch** — `dict[seed -> dict[step -> CacheEntry]]` cannot support global similarity search across all entries.
4. **OpenVLA-OFT duplicate concatenation bug** — `predict_action_batch()` concatenates `mm_embeddings`/`mm_attention_mask`/`multimodal_position_ids` twice: once inside the `else` branch (lines 478-486) and once unconditionally outside (lines 502-510).
5. **Modes that skip similarity checks** — `naive`, `cross_global_same_step`, and `cross_step_naive` modes never call `_is_similarity_hit()`, so `similarity_threshold` has no effect.

## Design

### New Cache Mode: `similarity_lru`

#### Storage Structure

Replace the nested dict with a flat list:

```python
@dataclass
class CacheEntry:
    features: dict[str, Any]              # Cached backbone features (CPU pinned)
    ref_obs: torch.Tensor | None          # Reference obs image (CPU)
    ref_vision_feat: torch.Tensor | None  # Vision encoder output (CPU)
    last_access: int                      # Monotonically increasing access counter
```

```python
class FeatureCache:
    # New fields for similarity_lru mode
    _entries: list[CacheEntry]            # Flat list of all cache entries
    _access_counter: int                  # Monotonically increasing, incremented on get/put

    # Legacy fields (retained for backward compatibility)
    _storage: dict[int, dict[int, CacheEntry]]
    _seed_order: list[int]
```

#### get() Logic (similarity_lru mode)

1. Iterate over all entries in `_entries`
2. For each entry, compute `similarity(current_obs, entry.ref_obs)` using the configured metric
3. Find the entry with the highest similarity score
4. If highest similarity >= `similarity_threshold`: hit — update `last_access`, return features
5. Otherwise: miss
6. If `_entries` is empty or `current_obs` has no extractable image tensor: miss

#### put() Logic (similarity_lru mode)

1. Create new entry with `last_access = _access_counter++`
2. Append to `_entries`
3. If `len(_entries) > max_entries`: evict the entry with the smallest `last_access`

#### Configuration

```python
@dataclass
class FeatureCacheConfig:
    enabled: bool = False
    mode: str = "disabled"              # "similarity_lru" | "disabled" | legacy modes
    similarity_metric: str = "obs_ssim" # "obs_ssim" | "obs_cosine" | "feature_cosine"
    similarity_threshold: float = 0.90
    max_entries: int = 256              # New: max entries for similarity_lru mode
    invalidate_on_weight_update: bool = True
    debug_log: bool = False
    debug_log_max_events: int = 1000
    # Legacy (retained for backward compatibility)
    max_cache_seeds: int = -1
```

#### Backward Compatibility

- Legacy modes (`naive`, `similarity_gated`, `cross_step_naive`, `cross_step_similarity`, `cross_global_same_step`) retain their existing behavior using `_storage`
- `similarity_lru` mode uses `_entries` exclusively
- `get()` and `put()` signatures unchanged — `seed` and `step` parameters are accepted but ignored in `similarity_lru` mode
- `invalidate_all()` clears both `_entries` and `_storage`

### Model Integration

#### GR00T (no code changes needed)

Current `_get_rl_action()` calls `feature_cache.get(seed, step, current_obs, ...)`. In `similarity_lru` mode, the cache ignores `seed`/`step` internally. No model code changes required.

#### OpenPI (new cache integration)

Integration point: `sample_actions()`, between observation preprocessing and VLM prefix computation.

**Cache content:** `past_key_values` (KV cache tuple) + `prefix_output` (if `use_vlm_value=True`)

**Flow:**
1. Before `embed_prefix()` + `paligemma_with_expert.forward()`, check cache
2. If hit: skip prefix computation, use cached `past_key_values` and `prefix_output`
3. If miss: compute normally, then `put()` results into cache
4. Per-sample cache lookup (same pattern as GR00T): iterate batch, split hits/misses, only compute backbone for misses

#### OpenVLA-OFT (bug fix only)

Remove duplicate concatenation in `predict_action_batch()` (lines 478-486 inside `else` branch). The unconditional concatenation at lines 502-510 handles all cases correctly.

### Rollout Worker

No changes needed. `_build_step_indices` / `_advance_step_indices` / `_reset_done_step_indices` continue to run; step indices are passed to models but ignored by cache in `similarity_lru` mode.

### Bug Fixes

1. **OpenVLA-OFT duplicate concatenation** — Remove lines 478-486 (redundant `torch.cat` inside `else` branch)
2. **OpenPI missing cache integration** — Add per-sample cache lookup in `sample_actions()`

### Error Handling

- If `max_entries` is -1 or unset in `similarity_lru` mode, default to 256
- If obs has no extractable image tensor (`_extract_obs_tensor` returns None): `put()` stores entry with `ref_obs=None`; `get()` treats entries with `ref_obs=None` as non-matchable (miss)
- Empty cache: `get()` returns miss immediately without iteration

### Memory Budget (similarity_lru, max_entries=256)

| Model | Per Entry | Total (256 entries) | Notes |
|-------|-----------|---------------------|-------|
| GR00T N1.5 | ~14 MB | ~3.5 GB | `vl_embs` tensor |
| OpenVLA-OFT 7B | ~15 MB | ~3.8 GB | Multimodal embeddings |
| OpenPI Pi0 | ~115 MB | ~7.4 GB (64 entries) | Large KV-cache; use smaller max_entries |
| OpenPI Pi0.5 | ~257 MB | ~4.1 GB (16 entries) | Use max_entries=16 |

### Testing

New unit tests for `similarity_lru` mode:
- `test_similarity_lru_basic_hit_miss` — put entry, get with same obs (hit), get with different obs (miss)
- `test_similarity_lru_ignores_seed_step` — different seed/step but same obs → hit
- `test_similarity_lru_lru_eviction` — exceed max_entries, oldest-accessed entry evicted
- `test_similarity_lru_global_best_match` — multiple entries, returns highest similarity match
- `test_similarity_lru_cross_global_step` — cache persists across simulated global steps

### What This Design Does NOT Change

- Training forward path (`default_forward`) — cache is rollout-only
- Weight sync logic (`invalidate_on_weight_update` retained)
- Env worker or seed propagation
- Evaluation rollout (cache not used during eval)
