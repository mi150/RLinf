# Feature Cache Similarity-LRU Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `similarity_lru` cache mode that does global obs-similarity lookup with LRU eviction, integrate cache into OpenPI, and fix the OpenVLA-OFT duplicate-concatenation bug.

**Architecture:** The `FeatureCache` class gains a flat `_entries` list used exclusively by the new `similarity_lru` mode. Legacy modes keep using the existing `_storage` dict. OpenPI's `sample_actions()` gets per-sample cache lookup (same pattern as GR00T). OpenVLA-OFT's duplicate `torch.cat` block is removed.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `rlinf/models/embodiment/feature_cache.py` | Add `max_entries` to config, `_entries`/`_access_counter` to cache, `similarity_lru` branch in `get()`/`put()` |
| Modify | `rlinf/workers/rollout/hf/huggingface_worker.py:135-150` | Parse `max_entries` from config |
| Modify | `rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py:56-72` | Parse `max_entries` in `_build_feature_cache_from_config` |
| Modify | `rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py:478-510` | Fix duplicate concatenation bug |
| Modify | `rlinf/models/embodiment/openpi/openpi_action_model.py:579-704` | Add per-sample cache lookup in `sample_actions()` |
| Modify | `tests/unit_tests/test_feature_cache.py` | Add `similarity_lru` unit tests |

---

### Task 1: Add `max_entries` to `FeatureCacheConfig` and new storage fields

**Files:**
- Modify: `rlinf/models/embodiment/feature_cache.py:23-31` (FeatureCacheConfig)
- Modify: `rlinf/models/embodiment/feature_cache.py:34-38` (CacheEntry)
- Modify: `rlinf/models/embodiment/feature_cache.py:129-135` (FeatureCache.__init__)
- Test: `tests/unit_tests/test_feature_cache.py`

- [ ] **Step 1: Write failing test for `max_entries` config field**

Add to `tests/unit_tests/test_feature_cache.py`:

```python
def test_similarity_lru_config_has_max_entries():
    cfg = FeatureCacheConfig(enabled=True, mode="similarity_lru", max_entries=128)
    assert cfg.max_entries == 128
    cache = FeatureCache(cfg)
    assert cache.config.max_entries == 128
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/miliang/RLinf && python -m pytest tests/unit_tests/test_feature_cache.py::test_similarity_lru_config_has_max_entries -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'max_entries'`

- [ ] **Step 3: Add `max_entries` to `FeatureCacheConfig`, `last_access` to `CacheEntry`, new fields to `FeatureCache.__init__`**

In `rlinf/models/embodiment/feature_cache.py`, edit the `FeatureCacheConfig` dataclass (line 23-31):

```python
@dataclass
class FeatureCacheConfig:
    enabled: bool = False
    mode: str = "disabled"
    similarity_metric: str = "obs_ssim"
    similarity_threshold: float = 0.90
    invalidate_on_weight_update: bool = True
    max_cache_seeds: int = -1
    max_entries: int = 256
    debug_log: bool = False
    debug_log_max_events: int = 1000
```

Edit the `CacheEntry` dataclass (line 34-38) to add `last_access`:

```python
@dataclass
class CacheEntry:
    features: dict[str, Any]
    ref_obs: torch.Tensor | None = None
    ref_vision_feat: torch.Tensor | None = None
    last_access: int = 0
```

Edit `FeatureCache.__init__` (line 130-135) to add new fields:

```python
class FeatureCache:
    def __init__(self, config: FeatureCacheConfig):
        self.config = config
        self._storage: dict[int, dict[int, CacheEntry]] = {}
        self._seed_order: list[int] = []
        self._entries: list[CacheEntry] = []
        self._access_counter: int = 0
        self._stats = CacheStats()
        self._debug_log_count = 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/miliang/RLinf && python -m pytest tests/unit_tests/test_feature_cache.py::test_similarity_lru_config_has_max_entries -v`
Expected: PASS

- [ ] **Step 5: Run all existing tests to verify no regressions**

Run: `cd /mnt/miliang/RLinf && python -m pytest tests/unit_tests/test_feature_cache.py -v`
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
cd /mnt/miliang/RLinf
git add rlinf/models/embodiment/feature_cache.py tests/unit_tests/test_feature_cache.py
git commit -m "feat(feature_cache): add max_entries config and similarity_lru storage fields"
```

---

### Task 2: Implement `similarity_lru` branch in `get()`

**Files:**
- Modify: `rlinf/models/embodiment/feature_cache.py:219-301` (get method)
- Test: `tests/unit_tests/test_feature_cache.py`

- [ ] **Step 1: Write failing tests for similarity_lru get behavior**

Add to `tests/unit_tests/test_feature_cache.py`:

```python
def test_similarity_lru_basic_hit():
    cache = FeatureCache(
        FeatureCacheConfig(
            enabled=True,
            mode="similarity_lru",
            similarity_metric="obs_cosine",
            similarity_threshold=0.99,
            max_entries=10,
        )
    )
    features = {"x": torch.ones(1, 4)}
    obs = _make_obs(0)
    cache.put(seed=1, step=0, features=features, obs=obs)
    loaded, hit = cache.get(seed=999, step=999, current_obs=obs)
    assert hit is True
    assert loaded is not None
    assert torch.allclose(loaded["x"], features["x"])


def test_similarity_lru_miss_on_different_obs():
    cache = FeatureCache(
        FeatureCacheConfig(
            enabled=True,
            mode="similarity_lru",
            similarity_metric="obs_cosine",
            similarity_threshold=0.99,
            max_entries=10,
        )
    )
    cache.put(seed=1, step=0, features={"x": torch.ones(1, 4)}, obs=_make_obs(0))
    _, hit = cache.get(seed=1, step=0, current_obs=_make_obs(100000))
    assert hit is False


def test_similarity_lru_ignores_seed_step():
    cache = FeatureCache(
        FeatureCacheConfig(
            enabled=True,
            mode="similarity_lru",
            similarity_metric="obs_cosine",
            similarity_threshold=0.99,
            max_entries=10,
        )
    )
    obs = _make_obs(0)
    cache.put(seed=1, step=0, features={"x": torch.ones(1, 4)}, obs=obs)
    # Different seed and step, same obs → should hit
    loaded, hit = cache.get(seed=42, step=77, current_obs=obs)
    assert hit is True
    assert loaded is not None


def test_similarity_lru_returns_best_match():
    cache = FeatureCache(
        FeatureCacheConfig(
            enabled=True,
            mode="similarity_lru",
            similarity_metric="obs_cosine",
            similarity_threshold=0.5,
            max_entries=10,
        )
    )
    obs_a = _make_obs(0)
    obs_b = _make_obs(1)  # very similar to obs_a
    obs_far = _make_obs(100000)
    cache.put(seed=1, step=0, features={"x": torch.full((1, 4), 1.0)}, obs=obs_a)
    cache.put(seed=2, step=0, features={"x": torch.full((1, 4), 2.0)}, obs=obs_far)
    # Query with obs_b (close to obs_a) should return obs_a's features
    loaded, hit = cache.get(seed=99, step=99, current_obs=obs_b)
    assert hit is True
    assert loaded is not None
    assert torch.allclose(loaded["x"], torch.full((1, 4), 1.0))


def test_similarity_lru_empty_cache_miss():
    cache = FeatureCache(
        FeatureCacheConfig(
            enabled=True,
            mode="similarity_lru",
            similarity_metric="obs_cosine",
            similarity_threshold=0.5,
            max_entries=10,
        )
    )
    _, hit = cache.get(seed=1, step=0, current_obs=_make_obs(0))
    assert hit is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/miliang/RLinf && python -m pytest tests/unit_tests/test_feature_cache.py::test_similarity_lru_basic_hit tests/unit_tests/test_feature_cache.py::test_similarity_lru_miss_on_different_obs tests/unit_tests/test_feature_cache.py::test_similarity_lru_ignores_seed_step tests/unit_tests/test_feature_cache.py::test_similarity_lru_returns_best_match tests/unit_tests/test_feature_cache.py::test_similarity_lru_empty_cache_miss -v`
Expected: FAIL — `similarity_lru` mode falls through to the `else` branch returning miss

- [ ] **Step 3: Implement `similarity_lru` branch in `get()`**

In `rlinf/models/embodiment/feature_cache.py`, in the `get()` method, add a new branch **before** the final `else` (before line 290). Insert between the `cross_step_similarity` block and the `else`:

```python
        elif self.config.mode == "similarity_lru":
            if not self._entries:
                self._stats.misses += 1
                self._log_debug(
                    f"[FeatureCache][GET][MISS] seed={seed} step={step} reason=empty_entries mode=similarity_lru"
                )
                return None, False
            current_obs_tensor = self._extract_obs_tensor(current_obs)
            if current_obs_tensor is None:
                self._stats.misses += 1
                self._log_debug(
                    f"[FeatureCache][GET][MISS] seed={seed} step={step} reason=no_obs_tensor mode=similarity_lru"
                )
                return None, False
            best_sim = -1.0
            best_entry: CacheEntry | None = None
            metric = self.config.similarity_metric
            threshold = self.config.similarity_threshold
            curr_obs_cpu = current_obs_tensor.detach().cpu()
            for candidate in self._entries:
                if candidate.ref_obs is None:
                    continue
                if metric == "feature_cosine" and vision_encoder_fn is not None:
                    current_feat = vision_encoder_fn(current_obs)
                    if current_feat is not None and candidate.ref_vision_feat is not None:
                        sim = _compute_cosine_similarity(
                            current_feat.detach().cpu(),
                            candidate.ref_vision_feat.detach().cpu(),
                        )
                    else:
                        sim = _compute_ssim(curr_obs_cpu, candidate.ref_obs)
                elif metric == "obs_ssim":
                    sim = _compute_ssim(curr_obs_cpu, candidate.ref_obs)
                elif metric == "obs_cosine":
                    sim = _compute_cosine_similarity(curr_obs_cpu, candidate.ref_obs)
                else:
                    sim = _compute_ssim(curr_obs_cpu, candidate.ref_obs)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = candidate
            if best_entry is not None and best_sim >= threshold:
                self._access_counter += 1
                best_entry.last_access = self._access_counter
                self._stats.hits += 1
                device = target_device if target_device is not None else self._resolve_device(current_obs)
                self._log_debug(
                    f"[FeatureCache][GET][HIT] seed={seed} step={step} sim={best_sim:.4f} mode=similarity_lru"
                )
                return _recursive_to_device(best_entry.features, device), True
            self._stats.misses += 1
            self._log_debug(
                f"[FeatureCache][GET][MISS] seed={seed} step={step} best_sim={best_sim:.4f} threshold={threshold} mode=similarity_lru"
            )
            return None, False
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `cd /mnt/miliang/RLinf && python -m pytest tests/unit_tests/test_feature_cache.py::test_similarity_lru_basic_hit tests/unit_tests/test_feature_cache.py::test_similarity_lru_miss_on_different_obs tests/unit_tests/test_feature_cache.py::test_similarity_lru_ignores_seed_step tests/unit_tests/test_feature_cache.py::test_similarity_lru_returns_best_match tests/unit_tests/test_feature_cache.py::test_similarity_lru_empty_cache_miss -v`
Expected: PASS

- [ ] **Step 5: Run all tests**

Run: `cd /mnt/miliang/RLinf && python -m pytest tests/unit_tests/test_feature_cache.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
cd /mnt/miliang/RLinf
git add rlinf/models/embodiment/feature_cache.py tests/unit_tests/test_feature_cache.py
git commit -m "feat(feature_cache): implement similarity_lru get() with global similarity search"
```

---

### Task 3: Implement `similarity_lru` branch in `put()` with LRU eviction

**Files:**
- Modify: `rlinf/models/embodiment/feature_cache.py:303-336` (put method)
- Modify: `rlinf/models/embodiment/feature_cache.py:338-355` (invalidate methods)
- Test: `tests/unit_tests/test_feature_cache.py`

- [ ] **Step 1: Write failing test for LRU eviction**

Add to `tests/unit_tests/test_feature_cache.py`:

```python
def test_similarity_lru_lru_eviction():
    cache = FeatureCache(
        FeatureCacheConfig(
            enabled=True,
            mode="similarity_lru",
            similarity_metric="obs_cosine",
            similarity_threshold=0.99,
            max_entries=2,
        )
    )
    obs_a = _make_obs(0)
    obs_b = _make_obs(10000)
    obs_c = _make_obs(20000)
    cache.put(seed=1, step=0, features={"x": torch.full((1, 1), 1.0)}, obs=obs_a)
    cache.put(seed=2, step=0, features={"x": torch.full((1, 1), 2.0)}, obs=obs_b)
    # Access obs_a to make it more recently used
    cache.get(seed=1, step=0, current_obs=obs_a)
    # Add obs_c — should evict obs_b (least recently accessed)
    cache.put(seed=3, step=0, features={"x": torch.full((1, 1), 3.0)}, obs=obs_c)
    # obs_a should still be in cache
    _, hit_a = cache.get(seed=1, step=0, current_obs=obs_a)
    assert hit_a is True
    # obs_b should have been evicted
    _, hit_b = cache.get(seed=2, step=0, current_obs=obs_b)
    assert hit_b is False
    # obs_c should be in cache
    _, hit_c = cache.get(seed=3, step=0, current_obs=obs_c)
    assert hit_c is True


def test_similarity_lru_invalidate_all_clears_entries():
    cache = FeatureCache(
        FeatureCacheConfig(
            enabled=True,
            mode="similarity_lru",
            similarity_metric="obs_cosine",
            similarity_threshold=0.99,
            max_entries=10,
        )
    )
    cache.put(seed=1, step=0, features={"x": torch.ones(1, 1)}, obs=_make_obs(0))
    cache.invalidate_all()
    _, hit = cache.get(seed=1, step=0, current_obs=_make_obs(0))
    assert hit is False
    assert len(cache._entries) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/miliang/RLinf && python -m pytest tests/unit_tests/test_feature_cache.py::test_similarity_lru_lru_eviction tests/unit_tests/test_feature_cache.py::test_similarity_lru_invalidate_all_clears_entries -v`
Expected: FAIL — `put()` doesn't populate `_entries` yet

- [ ] **Step 3: Implement `similarity_lru` branch in `put()` and update `invalidate` methods**

In `rlinf/models/embodiment/feature_cache.py`, edit the `put()` method. Replace the body (lines 311-336) with:

```python
    def put(
        self,
        seed: int,
        step: int,
        features: dict[str, Any],
        obs: dict[str, Any] | torch.Tensor | None = None,
        vision_feat: torch.Tensor | None = None,
    ) -> None:
        if not self.config.enabled or self.config.mode == "disabled":
            return

        ref_obs = self._extract_obs_tensor(obs)
        pinned_features = _recursive_to_cpu_pinned(features)
        pinned_ref_obs = (
            ref_obs.detach().cpu().contiguous().pin_memory()
            if ref_obs is not None
            else None
        )
        pinned_vision_feat = (
            vision_feat.detach().cpu().contiguous().pin_memory()
            if vision_feat is not None
            else None
        )

        if self.config.mode == "similarity_lru":
            self._access_counter += 1
            entry = CacheEntry(
                features=pinned_features,
                ref_obs=pinned_ref_obs,
                ref_vision_feat=pinned_vision_feat,
                last_access=self._access_counter,
            )
            self._entries.append(entry)
            max_entries = self.config.max_entries if self.config.max_entries > 0 else 256
            while len(self._entries) > max_entries:
                min_idx = min(range(len(self._entries)), key=lambda i: self._entries[i].last_access)
                self._entries.pop(min_idx)
            self._log_debug(
                f"[FeatureCache][PUT] seed={seed} step={step} mode=similarity_lru "
                f"cached_entries={len(self._entries)}"
            )
            return

        # Legacy modes
        if seed not in self._storage:
            self._storage[seed] = {}
            self._seed_order.append(seed)

        self._storage[seed][step] = CacheEntry(
            features=pinned_features,
            ref_obs=pinned_ref_obs,
            ref_vision_feat=pinned_vision_feat,
        )
        self._evict_if_needed()
        self._log_debug(
            f"[FeatureCache][PUT] seed={seed} step={step} mode={self.config.mode} "
            f"cached_seeds={self.num_cached_seeds()} cached_entries={self.num_cached_entries()}"
        )
```

Edit `invalidate_all()` to also clear `_entries`:

```python
    def invalidate_all(self) -> None:
        cleared = bool(self._storage or self._entries)
        if cleared:
            self._storage.clear()
            self._seed_order.clear()
            self._entries.clear()
            self._stats.invalidations += 1
            self._log_debug("[FeatureCache][INVALIDATE_ALL] cache_cleared=True")
```

Edit `num_cached_entries()` to include `_entries`:

```python
    def num_cached_entries(self) -> int:
        legacy = sum(len(seed_store) for seed_store in self._storage.values())
        return legacy + len(self._entries)
```

- [ ] **Step 4: Run the new tests**

Run: `cd /mnt/miliang/RLinf && python -m pytest tests/unit_tests/test_feature_cache.py::test_similarity_lru_lru_eviction tests/unit_tests/test_feature_cache.py::test_similarity_lru_invalidate_all_clears_entries -v`
Expected: PASS

- [ ] **Step 5: Run all tests**

Run: `cd /mnt/miliang/RLinf && python -m pytest tests/unit_tests/test_feature_cache.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
cd /mnt/miliang/RLinf
git add rlinf/models/embodiment/feature_cache.py tests/unit_tests/test_feature_cache.py
git commit -m "feat(feature_cache): implement similarity_lru put() with LRU eviction"
```

---

### Task 4: Parse `max_entries` in rollout worker and OpenVLA-OFT config builder

**Files:**
- Modify: `rlinf/workers/rollout/hf/huggingface_worker.py:135-150`
- Modify: `rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py:56-72`

- [ ] **Step 1: Update rollout worker config parsing**

In `rlinf/workers/rollout/hf/huggingface_worker.py`, edit the `FeatureCacheConfig(...)` constructor call (lines 135-149) to add `max_entries`:

```python
            cache_config = FeatureCacheConfig(
                enabled=feature_cache_cfg.get("enabled", False),
                mode=feature_cache_cfg.get("mode", "disabled"),
                similarity_metric=feature_cache_cfg.get(
                    "similarity_metric", "obs_ssim"
                ),
                similarity_threshold=feature_cache_cfg.get(
                    "similarity_threshold", 0.90
                ),
                invalidate_on_weight_update=invalidate_on_weight_update,
                max_cache_seeds=feature_cache_cfg.get("max_cache_seeds", -1),
                max_entries=feature_cache_cfg.get("max_entries", 256),
                debug_log=feature_cache_cfg.get("debug_log", False),
                debug_log_max_events=feature_cache_cfg.get(
                    "debug_log_max_events", 1000
                ),
            )
```

Also update the config print (lines 153-162) to include `max_entries`:

```python
            print(
                "[FeatureCache][CONFIG] "
                f"enabled={cache_config.enabled} "
                f"mode={cache_config.mode} "
                f"similarity_metric={cache_config.similarity_metric} "
                f"similarity_threshold={cache_config.similarity_threshold} "
                f"invalidate_on_weight_update={cache_config.invalidate_on_weight_update} "
                f"max_cache_seeds={cache_config.max_cache_seeds} "
                f"max_entries={cache_config.max_entries} "
                f"debug_log={cache_config.debug_log} "
                f"debug_log_max_events={cache_config.debug_log_max_events}"
            , flush=True)
```

- [ ] **Step 2: Update OpenVLA-OFT config builder**

In `rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py`, edit `_build_feature_cache_from_config` (lines 56-72) to add `max_entries`:

```python
        cache_config = FeatureCacheConfig(
            enabled=_cfg_get(feature_cache_cfg, "enabled", False),
            mode=_cfg_get(feature_cache_cfg, "mode", "disabled"),
            similarity_metric=_cfg_get(
                feature_cache_cfg, "similarity_metric", "obs_ssim"
            ),
            similarity_threshold=_cfg_get(
                feature_cache_cfg, "similarity_threshold", 0.90
            ),
            invalidate_on_weight_update=_cfg_get(
                feature_cache_cfg, "invalidate_on_weight_update", True
            ),
            max_cache_seeds=_cfg_get(feature_cache_cfg, "max_cache_seeds", -1),
            max_entries=_cfg_get(feature_cache_cfg, "max_entries", 256),
            debug_log=_cfg_get(feature_cache_cfg, "debug_log", False),
            debug_log_max_events=_cfg_get(feature_cache_cfg, "debug_log_max_events", 1000),
        )
```

- [ ] **Step 3: Run existing tests to verify no regressions**

Run: `cd /mnt/miliang/RLinf && python -m pytest tests/unit_tests/test_feature_cache.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
cd /mnt/miliang/RLinf
git add rlinf/workers/rollout/hf/huggingface_worker.py rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py
git commit -m "feat(feature_cache): parse max_entries in rollout worker and OpenVLA-OFT config"
```

---

### Task 5: Fix OpenVLA-OFT duplicate concatenation bug

**Files:**
- Modify: `rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py:478-500`

- [ ] **Step 1: Remove the duplicate `torch.cat` block inside the `else` branch**

In `rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py`, delete lines 478-486 (the first `torch.cat` block inside the `else` branch). These lines are:

```python
            mm_embeddings = torch.cat(
                [entry["mm_embeddings"] for entry in final_entries if entry is not None], dim=0
            )
            mm_attention_mask = torch.cat(
                [entry["mm_attention_mask"] for entry in final_entries if entry is not None], dim=0
            )
            multimodal_position_ids = torch.cat(
                [entry["multimodal_position_ids"] for entry in final_entries if entry is not None], dim=0
            )
```

Keep the identical block at lines 502-510 (outside the `if/else`) which handles all cases correctly.

After deletion, the `else` branch should end with the `[FeatureCache][PUT_OK]` print block, then fall through to the unconditional `torch.cat` at the former line 502.

- [ ] **Step 2: Verify the file structure is correct**

Read the modified area to confirm:
1. The `else` branch ends after the `[FeatureCache][PUT_OK]` print
2. The unconditional `torch.cat` block follows immediately after the `if/else`
3. No duplicate concatenation remains

- [ ] **Step 3: Run existing tests**

Run: `cd /mnt/miliang/RLinf && python -m pytest tests/unit_tests/test_feature_cache.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
cd /mnt/miliang/RLinf
git add rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py
git commit -m "fix(openvla_oft): remove duplicate torch.cat in predict_action_batch"
```

---

### Task 6: Integrate feature cache into OpenPI `sample_actions()`

**Files:**
- Modify: `rlinf/models/embodiment/openpi/openpi_action_model.py:579-620`
- Test: `tests/unit_tests/test_feature_cache.py`

This is the largest task. OpenPI's `sample_actions()` currently computes the VLM prefix (lines 601-619) for every call. We add per-sample cache lookup following the same pattern as GR00T.

**What gets cached:** `past_key_values` (KV cache tuple) + `prefix_output` (for `use_vlm_value`) + `prefix_pad_masks` (needed by denoising loop).

- [ ] **Step 1: Write a unit test verifying OpenPI cache integration pattern**

Add to `tests/unit_tests/test_feature_cache.py`:

```python
def test_similarity_lru_with_nested_tuple_features():
    """Verify cache handles KV-cache-like nested tuple structures (OpenPI pattern)."""
    cache = FeatureCache(
        FeatureCacheConfig(
            enabled=True,
            mode="similarity_lru",
            similarity_metric="obs_cosine",
            similarity_threshold=0.99,
            max_entries=10,
        )
    )
    kv_cache = (
        (torch.randn(1, 4, 8, 16), torch.randn(1, 4, 8, 16)),
        (torch.randn(1, 4, 8, 16), torch.randn(1, 4, 8, 16)),
    )
    features = {
        "past_key_values": kv_cache,
        "prefix_output": torch.randn(1, 32, 1024),
        "prefix_pad_masks": torch.ones(1, 32, dtype=torch.bool),
    }
    obs = _make_obs(0)
    cache.put(seed=1, step=0, features=features, obs=obs)
    loaded, hit = cache.get(seed=99, step=99, current_obs=obs)
    assert hit is True
    assert loaded is not None
    assert isinstance(loaded["past_key_values"], tuple)
    assert isinstance(loaded["past_key_values"][0], tuple)
    assert loaded["past_key_values"][0][0].shape == (1, 4, 8, 16)
    assert loaded["prefix_output"].shape == (1, 32, 1024)
```

- [ ] **Step 2: Run test to verify it passes** (should pass with existing implementation)

Run: `cd /mnt/miliang/RLinf && python -m pytest tests/unit_tests/test_feature_cache.py::test_similarity_lru_with_nested_tuple_features -v`
Expected: PASS (the recursive transfer functions already handle nested tuples)

- [ ] **Step 3: Implement cache integration in OpenPI `sample_actions()`**

In `rlinf/models/embodiment/openpi/openpi_action_model.py`, replace lines 601-619 (from `images, img_masks...` through `use_cache=True,` closing paren) with the cache-aware version:

```python
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )

        cache_enabled = (
            self.feature_cache is not None and self.feature_cache.config.enabled
        )
        backbone_device = state.device

        # --- Per-sample cache lookup ---
        cached_entries: list[dict[str, Any] | None] = [None] * bsize
        hit_indices: list[int] = []
        miss_indices: list[int] = []

        def _slice_obs_one_sample(
            obs: dict[str, Any] | None, idx: int, total: int
        ) -> dict[str, Any] | None:
            if obs is None:
                return None
            obs_b: dict[str, Any] = {}
            for key, value in obs.items():
                if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == total:
                    obs_b[key] = value[idx : idx + 1]
                elif isinstance(value, (list, tuple)) and len(value) == total:
                    obs_b[key] = value[idx : idx + 1]
                else:
                    obs_b[key] = value
            return obs_b

        if cache_enabled and env_seeds is not None and step_indices is not None:
            for b in range(bsize):
                current_obs_b = _slice_obs_one_sample(current_obs, b, bsize)
                cached_data, hit = self.feature_cache.get(
                    seed=int(env_seeds[b].item()),
                    step=int(step_indices[b].item()),
                    current_obs=current_obs_b,
                    target_device=backbone_device,
                    vision_encoder_fn=self.get_vision_features,
                )
                if hit:
                    cached_entries[b] = cached_data
                    hit_indices.append(b)
                else:
                    miss_indices.append(b)
        else:
            miss_indices = list(range(bsize))

        if cache_enabled:
            print(
                f"[FeatureCache][QUERY_RESULT][OpenPI] "
                f"batch_size={bsize} hit_count={len(hit_indices)} miss_count={len(miss_indices)}",
                flush=True,
            )

        # --- Compute prefix for misses ---
        if miss_indices:
            if len(miss_indices) == bsize:
                # All miss — compute for full batch (common path)
                miss_images = images
                miss_img_masks = img_masks
                miss_lang_tokens = lang_tokens
                miss_lang_masks = lang_masks
            else:
                miss_idx_t = torch.as_tensor(miss_indices, device=device, dtype=torch.int64)
                miss_images = [img.index_select(0, miss_idx_t) for img in images]
                miss_img_masks = [m.index_select(0, miss_idx_t) for m in img_masks]
                miss_lang_tokens = lang_tokens.index_select(0, miss_idx_t)
                miss_lang_masks = lang_masks.index_select(0, miss_idx_t)

            miss_prefix_embs, miss_prefix_pad_masks, miss_prefix_att_masks = self.embed_prefix(
                miss_images, miss_img_masks, miss_lang_tokens, miss_lang_masks
            )
            miss_prefix_att_2d_masks = make_att_2d_masks(miss_prefix_pad_masks, miss_prefix_att_masks)
            miss_prefix_position_ids = torch.cumsum(miss_prefix_pad_masks, dim=1) - 1
            miss_prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(miss_prefix_att_2d_masks)
            self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

            (miss_prefix_output, _), miss_past_key_values = self.paligemma_with_expert.forward(
                attention_mask=miss_prefix_att_2d_masks_4d,
                position_ids=miss_prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[miss_prefix_embs, None],
                use_cache=True,
            )

            # Store per-sample results and populate cache
            for miss_slot, b in enumerate(miss_indices):
                entry_kv = self._slice_kv_cache(miss_past_key_values, miss_slot)
                entry_data = {
                    "past_key_values": entry_kv,
                    "prefix_output": miss_prefix_output[miss_slot : miss_slot + 1],
                    "prefix_pad_masks": miss_prefix_pad_masks[miss_slot : miss_slot + 1],
                }
                cached_entries[b] = entry_data
                if cache_enabled and env_seeds is not None and step_indices is not None:
                    self.feature_cache.put(
                        seed=int(env_seeds[b].item()),
                        step=int(step_indices[b].item()),
                        features=entry_data,
                        obs=_slice_obs_one_sample(current_obs, b, bsize),
                    )

        # --- Reassemble full batch from hits + misses ---
        all_kv_caches = [cached_entries[b]["past_key_values"] for b in range(bsize)]
        past_key_values = self._reassemble_kv_cache(all_kv_caches)
        prefix_output = torch.cat(
            [cached_entries[b]["prefix_output"] for b in range(bsize)], dim=0
        )
        prefix_pad_masks = torch.cat(
            [cached_entries[b]["prefix_pad_masks"] for b in range(bsize)], dim=0
        )
```

The rest of `sample_actions()` (from `x_t = noise` onward, line 621+) stays unchanged — it uses `past_key_values`, `prefix_output`, and `prefix_pad_masks` which are now populated from cache or fresh computation.

- [ ] **Step 4: Run all unit tests**

Run: `cd /mnt/miliang/RLinf && python -m pytest tests/unit_tests/test_feature_cache.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /mnt/miliang/RLinf
git add rlinf/models/embodiment/openpi/openpi_action_model.py tests/unit_tests/test_feature_cache.py
git commit -m "feat(openpi): integrate feature cache into sample_actions() with per-sample lookup"
```

---

### Task 7: Update example config and final validation

**Files:**
- Modify: `examples/embodiment/config/libero_spatial_ppo_gr00t_feature_cache_similarity.yaml:86-94`

- [ ] **Step 1: Update the example config to use `similarity_lru` mode**

Edit `examples/embodiment/config/libero_spatial_ppo_gr00t_feature_cache_similarity.yaml`, replace the `feature_cache` block (lines 86-94):

```yaml
  feature_cache:
    enabled: true
    mode: "similarity_lru"
    similarity_metric: "obs_cosine"
    similarity_threshold: 0.90
    invalidate_on_weight_update: false
    max_entries: 256
    debug_log: False
    debug_log_max_events: 20000
```

- [ ] **Step 2: Run all unit tests one final time**

Run: `cd /mnt/miliang/RLinf && python -m pytest tests/unit_tests/test_feature_cache.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
cd /mnt/miliang/RLinf
git add examples/embodiment/config/libero_spatial_ppo_gr00t_feature_cache_similarity.yaml
git commit -m "chore: update example config to use similarity_lru cache mode"
```
