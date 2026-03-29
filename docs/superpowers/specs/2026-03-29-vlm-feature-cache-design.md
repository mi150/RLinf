# VLM Backbone Feature Cache for VLA RL Training

**Date:** 2026-03-29
**Status:** Approved

## Problem

In VLA (Vision-Language-Action) RL training, the VLM backbone (vision encoder + language model) is the most expensive component of each forward pass. When environment seeds are fixed across rollout epochs, observations in early steps are identical or similar across epochs. Re-computing backbone features for these overlapping observations is wasteful.

## Goal

Cache the intermediate features passed from the VLM backbone to the action head during rollout, keyed by environment seed. Reuse these cached features in subsequent rollout epochs over the same environments, avoiding redundant backbone computation.

Two caching modes:
1. **Naive**: Always reuse cached features for the same seed/step, ignoring observation divergence.
2. **Similarity-gated**: Reuse cached features only when current observation similarity exceeds a configurable threshold; otherwise re-run the backbone.

## Architecture

**Approach:** Cache layer inside each model class, with a shared `FeatureCache` utility.

Each VLA model (OpenPI, GR00T, OpenVLA-OFT) integrates the cache at its natural backbone-to-action-head boundary. A shared `FeatureCache` class handles storage, retrieval, similarity computation, and invalidation.

**Important:** The cache is used **only during rollout inference** (`predict_action_batch`). During **training forward passes** (`default_forward`), the cache is never consulted — training requires gradients through the backbone.

## Components

### 1. `FeatureCache` Utility

Location: `rlinf/models/embodiment/feature_cache.py`

A shared class used by all model implementations.

```python
class FeatureCache:
    def __init__(self, config: FeatureCacheConfig):
        # mode: "naive" | "similarity_gated" | "disabled"
        # similarity_metric: "obs_ssim" | "obs_cosine" | "feature_cosine"
        # similarity_threshold: float (default 0.90)
        # invalidate_on_weight_update: bool
        # max_cache_seeds: int (-1 = unlimited)
        self._storage: dict[int, dict[int, CacheEntry]] = {}
        # _storage[seed][step] = CacheEntry
        self._stats = CacheStats()  # hits, misses, invalidations counters

    def get(self, seed: int, step: int, current_obs: dict | None = None,
            vision_encoder_fn: Callable | None = None) -> tuple[dict[str, Any] | None, bool]:
        """Retrieve cached features.
        Returns (features_dict_or_None, cache_hit_bool).
        - naive mode: returns cached if exists
        - similarity_gated: returns cached only if similarity > threshold
        Increments hit/miss counters.
        """

    def put(self, seed: int, step: int, features: dict[str, Any],
            obs: dict | None = None, vision_feat: Tensor | None = None) -> None:
        """Store features in CPU pinned memory for a given seed/step.
        Also stores reference obs/vision_feat for similarity comparison.
        For nested structures (e.g., KV-cache tuples), recursively moves
        leaf tensors to CPU pinned memory."""

    def invalidate(self, seed: int | None = None) -> None:
        """Clear cache for a specific seed, or all seeds if None."""

    def invalidate_all(self) -> None:
        """Called when backbone weights change (if configured)."""

    def get_stats(self) -> CacheStats:
        """Return hit/miss/invalidation counters for logging."""
```

**Storage format:**

```python
@dataclass
class CacheEntry:
    features: dict[str, Any]      # model-specific cached data (may contain tuples of tensors
                                   # e.g., HF KV-cache is tuple[tuple[Tensor, ...], ...])
    ref_obs: Tensor | None        # reference observation image for obs-level similarity
    ref_vision_feat: Tensor | None  # reference vision features for feature-level similarity

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    invalidations: int = 0
```

**Note on KV-cache storage:** HuggingFace `past_key_values` is a `tuple[tuple[Tensor, Tensor], ...]` (one pair per layer). The `put()` method recursively traverses this structure, calling `.cpu().pin_memory()` on each leaf tensor. The `get()` method reconstructs the same tuple structure with `.to(device)` on each leaf.

### 2. Similarity Metrics

Three configurable similarity metrics:

| Metric | Method | Cost | Accuracy |
|--------|--------|------|----------|
| `obs_ssim` | SSIM on raw observation images | Negligible | Coarse |
| `obs_cosine` | Cosine similarity on flattened image pixels | Negligible | Coarse |
| `feature_cosine` | Cosine similarity on vision encoder embeddings | Low (see per-model cost below) | High |

**`feature_cosine` cost per model:**

| Model | Vision Encoder | Approx % of Full Backbone | Notes |
|-------|---------------|--------------------------|-------|
| OpenPI | SigLIP | ~5-10% | SigLIP is small relative to PaliGemma expert LM |
| GR00T | Eagle2.5 ViT | ~15-20% | Vision encoder is moderate vs full LM transformer |
| OpenVLA-OFT | SigLIP + DINOv2 + Projector | ~25-35% | Dual vision backbone is a larger fraction; net caching benefit is reduced |

For `feature_cosine`, each model exposes a lightweight `get_vision_features(obs)` method that runs only its vision encoder:
- **OpenPI**: SigLIP vision encoder only (before PaliGemma expert LM)
- **GR00T**: Eagle2.5 vision encoder only (before full LM transformer)
- **OpenVLA-OFT**: SigLIP/DINOv2 vision backbone + projector (before language model)

`BasePolicy` provides a default `get_vision_features()` that returns `None`. When `None` is returned and `feature_cosine` is configured, the cache falls back to `obs_ssim`.

### 3. Per-Model Cache Integration

#### OpenPI (`openpi_action_model.py`)

**Cache point:** Inside `sample_actions()`, after the prefix forward (line ~609), before the denoising loop (line ~611).

**Cached tensors:**
- `past_key_values`: Full VLM KV-cache — `tuple[tuple[Tensor, Tensor], ...]`, one (key, value) pair per transformer layer
- `prefix_output`: `Tensor [B, seq_len, hidden_dim]` — **required when `use_vlm_value=True`**, as the value head calls `self.get_value_from_vlm(prefix_output)`

```python
# In sample_actions():
cached, hit = self.feature_cache.get(seed, step, current_obs, self.get_vision_features)
if hit:
    past_key_values = cached["past_key_values"]  # CPU pinned → GPU (recursive)
    prefix_output = cached.get("prefix_output")   # for value head if needed
else:
    (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    cache_data = {"past_key_values": past_key_values}
    if self.use_vlm_value:
        cache_data["prefix_output"] = prefix_output
    self.feature_cache.put(seed, step,
        features=cache_data,
        obs=current_obs, vision_feat=vision_features)
# Denoising loop proceeds with past_key_values (cached or fresh)
```

**Approx memory per step per env:**
- Pi0 (PaliGemma 2B, 18 layers, hidden_dim=2048, ~816 prefix tokens, bfloat16): 2 * 18 * 816 * 2048 * 2 bytes ≈ **115 MB**
- Pi0.5 (PaliGemma 3B, 26 layers, hidden_dim=2560, ~968 tokens, bfloat16): 2 * 26 * 968 * 2560 * 2 bytes ≈ **257 MB**

#### GR00T (`gr00t_action_model.py`)

**Cache point:** Inside `GR00T_N1_5_ForRLActionPrediction._get_rl_action()` (line ~647), after `self.backbone(backbone_inputs)` (line ~655) and before `self.action_head.get_rl_action()` (line ~656). The cache is placed in the **main model class**, not inside the action head.

**Cached tensors:** `vl_embs` [B, ~1000, 3584] — the `backbone_output.backbone_features` tensor.

**NOT cached:** `state_features` (derived from proprioceptive state which changes every step), `embodiment_id` (fixed, passed through directly).

```python
# In GR00T_N1_5_ForRLActionPrediction._get_rl_action():
cached, hit = self.feature_cache.get(seed, step, current_obs, self.get_vision_features)
if hit:
    vl_embs = cached["vl_embs"]  # CPU pinned → GPU
    # Construct a minimal backbone_output with cached vl_embs
else:
    backbone_output = self.backbone(backbone_inputs)
    vl_embs = backbone_output.backbone_features
    self.feature_cache.put(seed, step,
        features={"vl_embs": vl_embs},
        obs=current_obs, vision_feat=vision_features)
# Action head proceeds with vl_embs (cached or fresh)
# state_features are always computed fresh from current proprioceptive state
```

**Approx memory:** ~14 MB per step per env (bfloat16, [1, 1000, 3584]).

#### OpenVLA-OFT (`openvla_oft_action_model.py`)

**Cache point:** Inside `predict_action_batch()`, after `_build_embedding()` (line ~513-573) which produces multimodal embeddings, and **before** the language model forward pass. This is the true VLM backbone boundary — the vision encoder + projector output. We do NOT cache the language model output (`action_logits`) because RL training requires re-sampling different actions with updated policy logits.

**Cached tensors:**
- `multimodal_embeddings`: `[B, total_seq_len, hidden_size]` — concatenation of projected vision patches + text embeddings + optional proprioception
- `multimodal_attention_mask`: `[B, total_seq_len]`
- `multimodal_position_ids`: `[B, total_seq_len]`

```python
# In predict_action_batch():
cached, hit = self.feature_cache.get(seed, step, current_obs, self.get_vision_features)
if hit:
    multimodal_embeddings = cached["multimodal_embeddings"]
    multimodal_attention_mask = cached["multimodal_attention_mask"]
    multimodal_position_ids = cached["multimodal_position_ids"]
else:
    multimodal_embeddings, multimodal_attention_mask, multimodal_position_ids = \
        self._build_embedding(pixel_values, input_ids, attention_mask, proprio)
    self.feature_cache.put(seed, step,
        features={
            "multimodal_embeddings": multimodal_embeddings,
            "multimodal_attention_mask": multimodal_attention_mask,
            "multimodal_position_ids": multimodal_position_ids,
        },
        obs=current_obs, vision_feat=vision_features)
# Language model forward pass always runs (needed for fresh action logits + value head)
lm_output = self.language_model(multimodal_embeddings, multimodal_attention_mask, ...)
```

**Note:** This caches the **vision encoder + projector** output, not the full LM. The language model still runs each step to produce fresh action logits for RL sampling. The speedup comes from skipping the expensive vision encoding (SigLIP + DINOv2 + projector).

**Approx memory:** ~10-20 MB per step per env (depends on seq_len and hidden_size, bfloat16).

### 4. Seed Management & Rollout Worker Plumbing

#### Fixed Seeds Across Rollout Epochs

New config option `use_fixed_rollout_seeds: bool` in env config. When enabled, the env worker resets environments with the same seeds each rollout epoch. Seeds within one epoch are different across environments.

**Per-env-class changes:** Env implementations (e.g., `libero_env.py`, `maniskill_env.py`) currently re-randomize seeds via `self._generator` on each reset. When `use_fixed_rollout_seeds=True`, the env class must store the initial seed list and reuse it on each `reset()` call instead of advancing the RNG. This interacts with existing `update_reset_state_ids()` — when fixed seeds are active, `reset_state_ids` are derived from the fixed seeds rather than re-randomized.

#### Metadata Propagation

1. **EnvOutput** (`embodied_io_struct.py`): Add `env_seeds: torch.Tensor | None = None` field.
   - Shape: `[B]` (one seed per env in the batch), dtype: `torch.int64`
   - Integrated with `to_dict()`: included if not None
   - Integrated with `merge_env_outputs()`: concatenated along batch dim
   - Integrated with `split_env_batch()`: split along batch dim
   - In `__post_init__`: moved to CPU and `.contiguous()` like other tensors

2. **EnvWorker** (`env_worker.py`): Populates `env_seeds` on each `EnvOutput` from the known per-env seed assignments.

3. **RolloutWorker** (`huggingface_worker.py`): Extracts `env_seeds` from the received `EnvOutput`. Maintains a per-env `step_counter` (see below). Passes `(seed, step)` to `model.predict_action_batch()` via kwargs.

#### Step Counter

The rollout worker maintains a `step_counter: Tensor` of shape `[num_envs]`, initialized to zero.

- **Incremented** by 1 after each `predict_action_batch()` call (per env in the batch).
- **Reset to 0** for an env when that env's `done` flag is True (detected via `EnvOutput.dones` or `EnvOutput.terminations | EnvOutput.truncations`).
- **Action chunking:** Each `predict_action_batch()` call corresponds to one "step" in the cache, regardless of how many sub-actions the chunk contains. A chunk of 8 actions = 1 cache step.
- **Auto-reset envs:** When different envs in the batch finish at different times, only the finished env's counter resets. Others continue incrementing.

### 5. Cache Invalidation

Config-driven invalidation:

- **Backbone frozen** (`freeze_backbone: true`): `invalidate_on_weight_update` defaults to `false`. Cache persists indefinitely across epochs.
- **Backbone trainable** (`freeze_backbone: false`): `invalidate_on_weight_update` defaults to `true`. Cache is invalidated after each training step that updates backbone weights.
- **Manual invalidation**: `cache.invalidate(seed)` or `cache.invalidate_all()` can be called explicitly.

Invalidation is triggered by the rollout worker after it syncs model weights from the actor (`sync_model_from_actor()`). If backbone weights changed (detectable via version tracking), it calls `model.feature_cache.invalidate_all()`.

**Dtype and device compatibility:** Cached tensors are stored in their original compute dtype (typically bfloat16). On cache hit, tensors are moved to the model's compute device via `.to(device, non_blocking=True)`. No dtype conversion is needed as long as the cache dtype matches the model's precision config.

### 6. Cache Hit Rate Logging

The `FeatureCache` maintains hit/miss/invalidation counters via `CacheStats`. The rollout worker logs these at the end of each rollout epoch:

```
[FeatureCache] Epoch 3: hits=48, misses=16, invalidations=0, hit_rate=75.0%
```

This is logged alongside existing rollout metrics for monitoring.

## Configuration

```yaml
algorithm:
  feature_cache:
    enabled: true
    mode: "naive"                    # "naive" | "similarity_gated" | "disabled"
    similarity_metric: "obs_ssim"    # "obs_ssim" | "obs_cosine" | "feature_cosine"
    similarity_threshold: 0.90       # reuse if similarity >= threshold
    invalidate_on_weight_update: null  # null = auto (based on freeze_backbone)
    max_cache_seeds: -1              # -1 = unlimited, or cap to limit memory

env:
  train:
    use_fixed_rollout_seeds: true    # reuse same seeds across rollout epochs
```

**When `use_fixed_rollout_seeds: false`:** The cache will have near-zero hit rate since seeds differ each epoch. In this case, it is recommended to set `feature_cache.enabled: false` to avoid the overhead of cache put operations. If `feature_cache.enabled: true` with non-fixed seeds, the system will still function correctly but without benefit.

## Data Flow

```
Rollout Epoch N, Step T, Env with Seed S:

1. EnvWorker sends (obs, env_seed=S) to RolloutWorker
2. RolloutWorker passes (obs, seed=S, step=T) to model.predict_action_batch()
3. Model checks feature_cache.get(seed=S, step=T, obs):

   [Cache HIT]                          [Cache MISS]
   +-- Load features from CPU pinned    +-- Run full VLM backbone
   |   memory -> GPU                    +-- Store features -> CPU pinned memory
   +-- Skip backbone forward            |   (feature_cache.put)
   +-- Run action head only             +-- Run action head

4. Model returns (actions, logprobs, values, forward_inputs)
5. On weight sync:
   - If invalidate_on_weight_update & backbone updated -> cache.invalidate_all()
   - Else cache persists for next epoch

Training Forward Pass (default_forward):
   - Cache is NEVER consulted during training
   - Full backbone forward runs with gradients enabled
```

## Memory Budget

Estimates for 4 envs, 16 steps per epoch, bfloat16:

| Model | Variant | Per Step | 4 envs x 16 steps | Notes |
|-------|---------|----------|-------------------|-------|
| GR00T | N1.5 (3584 hidden) | ~14 MB | ~900 MB | Moderate |
| OpenVLA-OFT | 7B | ~15 MB | ~960 MB | Only vision embeddings, not LM output |
| OpenPI | Pi0 (2B, 18 layers) | ~115 MB | ~7.4 GB | KV-cache is large |
| OpenPI | Pi0.5 (3B, 26 layers) | ~257 MB | ~16.4 GB | Use max_cache_seeds to cap |

## Files Changed

| File | Change |
|------|--------|
| `rlinf/models/embodiment/feature_cache.py` | **New.** Shared `FeatureCache` utility class, `CacheEntry`, `CacheStats`, similarity metrics |
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | Integrate cache in `sample_actions()`, add `get_vision_features()` |
| `rlinf/models/embodiment/gr00t/gr00t_action_model.py` | Integrate cache in `_get_rl_action()` on main model class, add `get_vision_features()` |
| `rlinf/models/embodiment/openvla_oft/official/openvla_oft_action_model.py` | Integrate cache in `predict_action_batch()` after `_build_embedding()`, add `get_vision_features()` |
| `rlinf/models/embodiment/base_policy.py` | Add optional `feature_cache` attribute and default `get_vision_features() -> None` |
| `rlinf/workers/env/env_worker.py` | Pass env_seed with observations; support fixed rollout seeds |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | Extract seed/step metadata; maintain step_counter; pass to model; trigger invalidation on weight sync; log cache stats |
| `rlinf/data/embodied_io_struct.py` | Add `env_seeds: Tensor | None` field to `EnvOutput` (shape [B], int64), integrate with to_dict/merge/split |
| Config YAML schemas | Add `algorithm.feature_cache` and `env.train.use_fixed_rollout_seeds` |

## Testing Strategy

1. **Unit tests for `FeatureCache`**: Test put/get, naive vs similarity-gated, invalidation, memory pinning, KV-cache tuple handling, stats counters.
2. **Per-model integration tests**: Verify cache hit skips backbone, cache miss runs backbone, outputs match within tolerance.
3. **Similarity metric tests**: Verify SSIM, cosine, and feature_cosine produce correct scores and thresholds work. Test fallback from feature_cosine to obs_ssim when get_vision_features returns None.
4. **End-to-end test**: Run 2 rollout epochs with fixed seeds, verify cache hit rate and output consistency.
5. **Memory tests**: Verify features are in CPU pinned memory, max_cache_seeds cap works, dtype preserved.
6. **Step counter tests**: Verify counter resets on done, handles mixed-done batches, action chunking = 1 step.
