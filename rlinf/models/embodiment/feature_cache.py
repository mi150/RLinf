# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

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


@dataclass
class CacheEntry:
    features: dict[str, Any]
    ref_obs: torch.Tensor | None = None
    ref_vision_feat: torch.Tensor | None = None
    last_access: int = 0


@dataclass
class CacheStats:
    hits: int = 0
    same_step_hits: int = 0
    misses: int = 0
    invalidations: int = 0


def _recursive_to_cpu_pinned(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().contiguous().pin_memory()
    if isinstance(obj, dict):
        return {k: _recursive_to_cpu_pinned(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_recursive_to_cpu_pinned(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_recursive_to_cpu_pinned(v) for v in obj)
    return obj


def _recursive_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device=device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _recursive_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_recursive_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_recursive_to_device(v, device) for v in obj)
    return obj


def _compute_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.float().reshape(1, -1)
    b_flat = b.float().reshape(1, -1)
    return float(F.cosine_similarity(a_flat, b_flat, dim=-1).item())


def _gaussian_kernel2d(
    kernel_size: int = 11, sigma: float = 1.5, device: torch.device | None = None
) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
    coords = coords - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel2d = torch.outer(g, g)
    return kernel2d


def _compute_ssim(img_a: torch.Tensor, img_b: torch.Tensor) -> float:
    """Compute SSIM for tensors in shape [B, C, H, W]."""
    if img_a.dim() == 3:
        img_a = img_a.unsqueeze(0)
    if img_b.dim() == 3:
        img_b = img_b.unsqueeze(0)
    if img_a.shape != img_b.shape:
        return 0.0

    img_a = img_a.float()
    img_b = img_b.float()
    if img_a.max() > 1.0 or img_b.max() > 1.0:
        img_a = img_a / 255.0
        img_b = img_b / 255.0

    c = img_a.shape[1]
    kernel = _gaussian_kernel2d(device=img_a.device).view(1, 1, 11, 11)
    kernel = kernel.repeat(c, 1, 1, 1)
    padding = 5

    mu_a = F.conv2d(img_a, kernel, padding=padding, groups=c)
    mu_b = F.conv2d(img_b, kernel, padding=padding, groups=c)

    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a2 = F.conv2d(img_a * img_a, kernel, padding=padding, groups=c) - mu_a2
    sigma_b2 = F.conv2d(img_b * img_b, kernel, padding=padding, groups=c) - mu_b2
    sigma_ab = F.conv2d(img_a * img_b, kernel, padding=padding, groups=c) - mu_ab

    c1 = 0.01**2
    c2 = 0.03**2
    num = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    ssim_map = num / (den + 1e-12)
    return float(ssim_map.mean().item())


class FeatureCache:
    def __init__(self, config: FeatureCacheConfig):
        self.config = config
        self._storage: dict[int, dict[int, CacheEntry]] = {}
        self._seed_order: list[int] = []
        self._stats = CacheStats()
        self._debug_log_count = 0
        self._entries: list[CacheEntry] = []
        self._access_counter: int = 0

    def _log_debug(self, message: str) -> None:
        if not self.config.debug_log:
            return
        max_events = self.config.debug_log_max_events
        if max_events >= 0 and self._debug_log_count >= max_events:
            if self._debug_log_count == max_events:
                print(
                    "[FeatureCache][DEBUG] reached debug_log_max_events="
                    f"{max_events}, suppressing further logs.",
                    flush=True,
                )
            self._debug_log_count += 1
            return
        print(message, flush=True)
        self._debug_log_count += 1

    def _extract_obs_tensor(self, obs: dict[str, Any] | torch.Tensor | None) -> torch.Tensor | None:
        if obs is None:
            return None
        if torch.is_tensor(obs):
            return obs
        if not isinstance(obs, dict):
            return None
        for key in ("observation/image", "main_images", "pixel_values"):
            value = obs.get(key)
            if torch.is_tensor(value):
                return value
        for value in obs.values():
            if torch.is_tensor(value):
                return value
        return None

    def _resolve_device(self, current_obs: dict[str, Any] | torch.Tensor | None) -> torch.device:
        obs_tensor = self._extract_obs_tensor(current_obs)
        if obs_tensor is not None:
            return obs_tensor.device
        return torch.device("cpu")

    def _is_similarity_hit(
        self,
        entry: CacheEntry,
        current_obs: dict[str, Any] | torch.Tensor | None,
        vision_encoder_fn: Callable[[dict[str, Any] | torch.Tensor], torch.Tensor | None]
        | None,
    ) -> bool:
        metric = self.config.similarity_metric
        threshold = self.config.similarity_threshold
        current_obs_tensor = self._extract_obs_tensor(current_obs)

        if metric == "feature_cosine" and vision_encoder_fn is not None:
            current_feat = vision_encoder_fn(current_obs)  # type: ignore[arg-type]
            if current_feat is not None and entry.ref_vision_feat is not None:
                sim = _compute_cosine_similarity(
                    current_feat.detach().cpu(),
                    entry.ref_vision_feat.detach().cpu(),
                )
                return sim >= threshold
            # fallback to obs-level metric when model does not support vision-only features
            metric = "obs_ssim"

        if current_obs_tensor is None or entry.ref_obs is None:
            return False
        ref_obs = entry.ref_obs
        curr_obs = current_obs_tensor.detach().cpu()

        if metric == "obs_ssim":
            sim = _compute_ssim(curr_obs, ref_obs)
            return sim >= threshold
        if metric == "obs_cosine":
            sim = _compute_cosine_similarity(curr_obs, ref_obs)
            return sim >= threshold
        return False

    def _evict_if_needed(self) -> None:
        max_cache_seeds = self.config.max_cache_seeds
        if max_cache_seeds is None or max_cache_seeds < 0:
            return
        while len(self._seed_order) > max_cache_seeds:
            evict_seed = self._seed_order.pop(0)
            if evict_seed in self._storage:
                del self._storage[evict_seed]

    def get(
        self,
        seed: int,
        step: int,
        current_obs: dict[str, Any] | torch.Tensor | None = None,
        target_device: torch.device | None = None,
        vision_encoder_fn: Callable[[dict[str, Any] | torch.Tensor], torch.Tensor | None]
        | None = None,
    ) -> tuple[dict[str, Any] | None, bool]:
        if not self.config.enabled or self.config.mode == "disabled":
            self._stats.misses += 1
            self._log_debug(
                f"[FeatureCache][GET][MISS] seed={seed} step={step} reason=cache_disabled mode={self.config.mode}"
            )
            return None, False

        seed_store = self._storage.get(seed)
        if seed_store is None and self.config.mode != "similarity_lru":
            self._stats.misses += 1
            self._log_debug(
                f"[FeatureCache][GET][MISS] seed={seed} step={step} reason=seed_not_found mode={self.config.mode}"
            )
            return None, False
        entry = None
        resolved_step: int | None = None

        if self.config.mode in (
            "naive",
            "similarity_gated",
            "cross_global_same_step",
        ):
            if step not in seed_store:
                self._stats.misses += 1
                self._log_debug(
                    f"[FeatureCache][GET][MISS] seed={seed} step={step} reason=step_not_found mode={self.config.mode}"
                )
                return None, False
            entry = seed_store[step]
            resolved_step = step
            if self.config.mode == "similarity_gated" and not self._is_similarity_hit(
                entry, current_obs, vision_encoder_fn
            ):
                self._stats.misses += 1
                self._log_debug(
                    f"[FeatureCache][GET][MISS] seed={seed} step={step} reason=similarity_check_failed mode={self.config.mode}"
                )
                return None, False
        elif self.config.mode == "cross_step_naive":
            # Step-agnostic lookup: always reuse the latest cached entry for this seed.
            if not seed_store:
                self._stats.misses += 1
                self._log_debug(
                    f"[FeatureCache][GET][MISS] seed={seed} step={step} reason=empty_seed_store mode={self.config.mode}"
                )
                return None, False
            latest_step = max(seed_store.keys())
            entry = seed_store[latest_step]
            resolved_step = latest_step
        elif self.config.mode == "cross_step_similarity":
            # Step-agnostic lookup: find a semantically similar cached entry within this seed.
            for candidate_step, candidate in reversed(list(seed_store.items())):
                if self._is_similarity_hit(candidate, current_obs, vision_encoder_fn):
                    entry = candidate
                    resolved_step = candidate_step
                    break
            if entry is None:
                self._stats.misses += 1
                self._log_debug(
                    f"[FeatureCache][GET][MISS] seed={seed} step={step} reason=no_similar_entry mode={self.config.mode}"
                )
                return None, False
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
        else:
            self._stats.misses += 1
            self._log_debug(
                f"[FeatureCache][GET][MISS] seed={seed} step={step} reason=unsupported_mode mode={self.config.mode}"
            )
            return None, False

        self._stats.hits += 1
        if resolved_step == step:
            self._stats.same_step_hits += 1
        device = target_device if target_device is not None else self._resolve_device(current_obs)
        return _recursive_to_device(entry.features, device), True

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
            ref_obs.detach().cpu().contiguous().pin_memory() if ref_obs is not None else None
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

    def invalidate(self, seed: int | None = None) -> None:
        if seed is None:
            self.invalidate_all()
            return
        if seed in self._storage:
            del self._storage[seed]
            self._seed_order = [s for s in self._seed_order if s != seed]
            self._stats.invalidations += 1
            self._log_debug(
                f"[FeatureCache][INVALIDATE] seed={seed} cached_seeds={self.num_cached_seeds()} cached_entries={self.num_cached_entries()}"
            )

    def invalidate_all(self) -> None:
        cleared = bool(self._storage or self._entries)
        if cleared:
            self._storage.clear()
            self._seed_order.clear()
            self._entries.clear()
            self._stats.invalidations += 1
            self._log_debug("[FeatureCache][INVALIDATE_ALL] cache_cleared=True")

    def get_stats(self) -> CacheStats:
        return CacheStats(
            hits=self._stats.hits,
            same_step_hits=self._stats.same_step_hits,
            misses=self._stats.misses,
            invalidations=self._stats.invalidations,
        )

    def num_cached_seeds(self) -> int:
        return len(self._storage)

    def num_cached_entries(self) -> int:
        legacy = sum(len(seed_store) for seed_store in self._storage.values())
        return legacy + len(self._entries)

    def reset_stats(self) -> None:
        self._stats = CacheStats()
