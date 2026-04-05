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

from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from prismatic.extern.hf.configuration_prismatic import (
    OpenVLAConfig as OpenVLAOFTConfig,
)
from prismatic.extern.hf.modeling_prismatic import (
    OpenVLAForActionPrediction as OpenVLAOFTForActionPrediction,
)
from prismatic.vla.constants import (
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    STOP_INDEX,
    NormalizationType,
)
from transformers.generation import TopKLogitsWarper

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.feature_cache import FeatureCache, FeatureCacheConfig
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.utils import (
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
)


class OpenVLAOFTForRLActionPrediction(OpenVLAOFTForActionPrediction, BasePolicy):
    @staticmethod
    def _build_feature_cache_from_config(config: OpenVLAOFTConfig):
        feature_cache_cfg = getattr(config, "feature_cache", None)
        if feature_cache_cfg is None:
            return None

        def _cfg_get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            if hasattr(obj, "get"):
                return obj.get(key, default)
            return getattr(obj, key, default)

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
        return FeatureCache(cache_config)

    def __init__(
        self,
        config: OpenVLAOFTConfig,
        action_dim,
        num_action_chunks,
        add_value_head,
        max_prompt_length,
    ) -> None:
        super().__init__(config)

        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks

        self.unnorm_key = config.unnorm_key
        if (
            self.unnorm_key not in self.norm_stats
            and f"{self.unnorm_key}_no_noops" in self.norm_stats
        ):
            self.unnorm_key = f"{self.unnorm_key}_no_noops"
        assert self.unnorm_key in self.norm_stats, (
            f"Action un-norm key {self.unnorm_key} not found in VLA `norm_stats`!"
        )

        if add_value_head:
            self.hidden_size = self.config.hidden_size
            output_dim = (
                1 if self.config.value_type == "chunk_level" else self.num_action_chunks
            )
            self.value_head = ValueHead(
                input_dim=self.hidden_size,
                hidden_sizes=(512, 128),
                output_dim=output_dim,
                activation="gelu",
                bias_last=False,
            )

        self.max_prompt_length = max_prompt_length
        self.feature_cache = self._build_feature_cache_from_config(config)

    def _build_embedding(self, input_ids, attention_mask, pixel_values):
        assert torch.all(input_ids[:, -1] == STOP_INDEX)
        assert input_ids.shape[0] == attention_mask.shape[0]
        assert input_ids.shape[1] == attention_mask.shape[1]

        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]

        n_patch_tokens = (
            self.vision_backbone.get_num_patches()
            * self.vision_backbone.get_num_images_in_input()
        )

        # llm label & mask & embedding
        all_actions_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        all_actions_mask[:, -self.action_dim * self.num_action_chunks :] = (
            True  # [B, L + act + 1], [many x 0; act x 1; 0]
        )

        input_embeddings = self.get_input_embeddings()(input_ids)  # [B, L + act + 1, D]
        input_embeddings = input_embeddings * (~all_actions_mask.unsqueeze(-1))

        # vision
        projected_patch_embeddings = self._process_vision_features(
            pixel_values, None, use_film=False
        )
        # [B, 256 * num_images, D]
        assert projected_patch_embeddings.shape[1] == n_patch_tokens

        # multimodal embeddings
        projected_patch_embeddings = projected_patch_embeddings.reshape(
            input_embeddings.shape[0], -1, *projected_patch_embeddings.shape[2:]
        )
        multimodal_embeddings, multimodal_attention_mask = (
            self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )
        )
        assert (
            multimodal_embeddings.shape[1]
            == input_embeddings.shape[1] + projected_patch_embeddings.shape[1]
        )
        assert (
            multimodal_attention_mask.shape[1]
            == attention_mask.shape[1] + projected_patch_embeddings.shape[1]
        )

        return multimodal_embeddings, multimodal_attention_mask

    def _get_action_stats(self) -> dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, self.unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

    def _prepare_input_for_action_prediction(self, input_ids, attention_mask):
        """Prepares input for action prediction by adding necessary tokens"""
        # Add (ACTION_DIM * NUM_ACTIONS_CHUNK) placeholder tokens to input_ids to simulate action tokens
        placeholder_action_token_ids = (
            torch.ones((input_ids.shape[0], self.action_dim * self.num_action_chunks))
            .to(input_ids.device)
            .to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        # Add stop token to sequence (needed in non-causal bi-directional self-attention, as it appears at train time)
        stop_token_id = (
            torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype)
            * STOP_INDEX
        )
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        # Extend the attention mask to fit the new shape of input
        # Note: Only batch size == 1 supported right now
        mask_extension = (
            torch.ones(
                (
                    attention_mask.shape[0],
                    input_ids.shape[-1] - attention_mask.shape[-1],
                )
            )
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        return input_ids, attention_mask

    def _unnormalize_actions(self, normalized_actions, unnorm_key=None):
        """Unnormalize actions using dataset statistics"""
        action_norm_stats = self.get_action_stats(unnorm_key)

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = action_norm_stats.get(
                "mask", np.ones_like(action_norm_stats["min"], dtype=bool)
            )
            action_high, action_low = (
                np.array(action_norm_stats["max"]),
                np.array(action_norm_stats["min"]),
            )
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = action_norm_stats.get(
                "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
            )
            action_high, action_low = (
                np.array(action_norm_stats["q99"]),
                np.array(action_norm_stats["q01"]),
            )
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")

        action_dim = normalized_actions.shape[-1]
        repeat_factor = action_dim // action_high.shape[0]
        action_high = action_high.repeat(repeat_factor)
        action_low = action_low.repeat(repeat_factor)
        mask = mask * repeat_factor

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8)
            + action_low,
            normalized_actions,
        )

        return actions

    @torch.no_grad()
    def predict_action_batch(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.FloatTensor = None,
        env_obs=None,
        calculate_logprobs=True,
        calculate_values=True,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        do_sample = kwargs.pop("do_sample")
        env_seeds = kwargs.pop("env_seeds", None)
        step_indices = kwargs.pop("step_indices", None)
        cache_enabled = (
            self.feature_cache is not None and self.feature_cache.config.enabled
        )
        print(
            "[FeatureCache][INFO] Predicting action batch. "
            f"cache_enabled={cache_enabled} "
            f"has_env_seeds={env_seeds is not None} "
            f"has_step_indices={step_indices is not None} "
            "and use Rlinf model.",
            flush=True,
        )
        if env_obs is not None:
            task_descriptions = [
                f"In: What action should the robot take to {t.lower()}?\nOut: "
                for t in env_obs["task_descriptions"]
            ]
            if env_obs["main_images"].ndim == 4:
                env_obs["main_images"] = env_obs["main_images"].unsqueeze(1)
            assert env_obs["main_images"].ndim == 5

            all_images = [
                env_obs["main_images"].permute(0, 1, 4, 2, 3)
            ]  # [B, 1, H, W, C] -> [B, 1, C, H, W]
            if self.vision_backbone.get_num_images_in_input() > 1:
                if env_obs["wrist_images"].ndim == 4:
                    env_obs["wrist_images"] = env_obs["wrist_images"].unsqueeze(1)
                assert env_obs["wrist_images"].ndim == 5
                wrist_imgs = env_obs["wrist_images"].permute(
                    0, 1, 4, 2, 3
                )  # [B, N_IMG, H, W, C] -> [B, N_IMG, C, H, W]
                all_images.extend(
                    [wrist_imgs[:, i] for i in range(wrist_imgs.shape[1])]
                )

            max_length = self.max_prompt_length
            device = next(self.parameters()).device
            precision = next(self.parameters()).dtype

            primary_image = all_images.pop(0)
            images = {"images": primary_image}
            inputs = self.input_processor(
                text=task_descriptions,
                images=images,
                proprio_states=env_obs["states"],
                padding="max_length",
                max_length=max_length,
            )

            if all_images:
                all_wrist_inputs = [
                    self.input_processor(
                        text=task_descriptions,
                        images={"images": wrist_image.unsqueeze(1)},
                        proprio_states=env_obs["states"],
                        padding="max_length",
                        max_length=max_length,
                    )
                    for wrist_image in all_images
                ]

                # Concatenate all images
                primary_pixel_values = inputs["pixel_values"]
                all_wrist_pixel_values = [
                    wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs
                ]
                inputs["pixel_values"] = torch.cat(
                    [primary_pixel_values] + all_wrist_pixel_values, dim=1
                )

            input_ids = inputs["input_ids"].to(device=device, dtype=torch.long)
            attention_mask = inputs["attention_mask"].to(
                device=device, dtype=torch.bool
            )
            pixel_values = inputs["pixel_values"].to(device=device, dtype=precision)

            B, N, C, H, W = pixel_values.shape
            pixel_values = pixel_values.reshape(B, N * C, H, W)

        forward_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

        # assert first token is 1
        assert torch.all(input_ids[:, 0] == 1)
        assert torch.all(attention_mask[:, 0] == 1)
        # last token is space ` `
        assert torch.all(input_ids[:, -1] == 29871)
        assert torch.all(attention_mask[:, -1] == 1)

        n_prompt_tokens = input_ids.shape[-1] - 1
        # Calculate number of patches (including proprio token and/or diffusion timestep embedding if present)
        n_patches = (
            self.vision_backbone.get_num_patches()
            * self.vision_backbone.get_num_images_in_input()
        )

        # llm inputs
        input_ids, attention_mask = self._prepare_input_for_action_prediction(
            input_ids, attention_mask
        )
        assert torch.all(input_ids[:, -1] == STOP_INDEX)  # [B, L + act + 1, D]
        assert torch.all(
            attention_mask[:, -1 - self.action_dim * self.num_action_chunks :] == 1
        )  # [B, L + act + 1]
        # multimodal
        batch_size = input_ids.shape[0]
        cached_entries: list[dict[str, Any] | None] = [None] * batch_size
        cache_trace_enabled = bool(
            cache_enabled and getattr(self.feature_cache.config, "debug_log", False)
        )
        miss_reason = "not_checked"
        miss_sample: tuple[int, int, int] | None = None  # (batch_idx, seed, step)
        hit_indices: list[int] = []
        miss_indices: list[int] = []
        hit_steps: list[int] = []
        miss_steps: list[int] = []
        if (
            cache_enabled
            and env_seeds is not None
            and step_indices is not None
            and env_seeds.shape[0] == batch_size
            and step_indices.shape[0] == batch_size
        ):
            print(
                "[FeatureCache][QUERY_BATCH] "
                f"batch_size={batch_size} "
                f"seeds_preview={env_seeds.tolist()[:8]} "
                f"steps_preview={step_indices.tolist()[:8]}",
                flush=True,
            )
            for b in range(batch_size):
                seed_b = int(env_seeds[b].item())
                step_b = int(step_indices[b].item())
                cached_data, hit = self.feature_cache.get(
                    seed=seed_b,
                    step=step_b,
                    current_obs=env_obs,
                    target_device=input_ids.device,
                    vision_encoder_fn=self.get_vision_features,
                )
                if hit:
                    cached_entries[b] = cached_data
                    hit_indices.append(b)
                    hit_steps.append(step_b)
                else:
                    miss_indices.append(b)
                    miss_steps.append(step_b)
                    if miss_sample is None:
                        miss_reason = "cache_miss"
                        miss_sample = (b, seed_b, step_b)
            print(
                "[FeatureCache][QUERY_RESULT] "
                f"batch_size={batch_size} "
                f"hit_count={len(hit_indices)} miss_count={len(miss_indices)} "
                f"hit_steps_preview={hit_steps[:8]} "
                f"miss_steps_preview={miss_steps[:8]}",
                flush=True,
            )
        elif cache_enabled:
            miss_reason = "seed_or_step_missing_or_shape_mismatch"
            miss_indices = list(range(batch_size))
            if step_indices is not None and step_indices.shape[0] == batch_size:
                miss_steps = [int(step_indices[i].item()) for i in miss_indices]
            print(
                "[FeatureCache][QUERY_RESULT] "
                f"batch_size={batch_size} hit_count=0 miss_count={len(miss_indices)} "
                f"reason={miss_reason} miss_steps_preview={miss_steps[:8]}",
                flush=True,
            )
        else:
            miss_reason = "cache_disabled"
            miss_indices = list(range(batch_size))

        final_entries: list[dict[str, Any] | None] = [None] * batch_size
        for b in hit_indices:
            final_entries[b] = cached_entries[b]

        if not miss_indices:
            if cache_trace_enabled:
                print(
                    "[FeatureCache][REUSE] "
                    f"batch_size={batch_size} "
                    f"seeds={env_seeds.tolist() if env_seeds is not None else None} "
                    f"steps={step_indices.tolist() if step_indices is not None else None}"
                , flush=True)
        else:
            if cache_trace_enabled:
                print(
                    "[FeatureCache][OBTAIN] recompute_backbone "
                    f"batch_size={batch_size} reason={miss_reason} miss_sample={miss_sample}"
                , flush=True)
            miss_index_tensor = torch.as_tensor(
                miss_indices, device=input_ids.device, dtype=torch.int64
            )
            miss_input_ids = input_ids.index_select(0, miss_index_tensor)
            miss_attention_mask = attention_mask.index_select(0, miss_index_tensor)
            miss_pixel_values = pixel_values.index_select(0, miss_index_tensor)
            miss_mm_embeddings, miss_mm_attention_mask = self._build_embedding(
                miss_input_ids, miss_attention_mask, miss_pixel_values
            )
            miss_multimodal_position_ids = miss_mm_attention_mask.cumsum(dim=1) - 1

            for miss_slot, b in enumerate(miss_indices):
                miss_entry = {
                    "mm_embeddings": miss_mm_embeddings[miss_slot : miss_slot + 1],
                    "mm_attention_mask": miss_mm_attention_mask[miss_slot : miss_slot + 1],
                    "multimodal_position_ids": miss_multimodal_position_ids[
                        miss_slot : miss_slot + 1
                    ],
                }
                final_entries[b] = miss_entry
                if (
                    cache_enabled
                    and env_seeds is not None
                    and step_indices is not None
                    and env_seeds.shape[0] == batch_size
                    and step_indices.shape[0] == batch_size
                ):
                    self.feature_cache.put(
                        seed=int(env_seeds[b].item()),
                        step=int(step_indices[b].item()),
                        features=miss_entry,
                        obs=env_obs,
                    )
            if (
                cache_enabled
                and env_seeds is not None
                and step_indices is not None
                and env_seeds.shape[0] == batch_size
                and step_indices.shape[0] == batch_size
            ):
                print(
                    "[FeatureCache][PUT_OK] "
                    f"saved_entries={len(miss_indices)} "
                    f"miss_steps_preview={miss_steps[:8]} "
                    f"miss_indices_preview={miss_indices[:8]}",
                    flush=True,
                )

        mm_embeddings = torch.cat(
            [entry["mm_embeddings"] for entry in final_entries if entry is not None], dim=0
        )
        mm_attention_mask = torch.cat(
            [entry["mm_attention_mask"] for entry in final_entries if entry is not None], dim=0
        )
        multimodal_position_ids = torch.cat(
            [entry["multimodal_position_ids"] for entry in final_entries if entry is not None], dim=0
        )

        # Forward pass through language model
        import time as _time
        _t0 = _time.perf_counter()
        outputs = self.language_model(
            input_ids=None,
            attention_mask=mm_attention_mask,
            position_ids=multimodal_position_ids,
            past_key_values=None,
            inputs_embeds=mm_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        _lm_ms = (_time.perf_counter() - _t0) * 1000
        _total_tokens = mm_embeddings.shape[1]
        _n_action_tokens = self.action_dim * self.num_action_chunks
        _n_prefix_tokens = _total_tokens - _n_action_tokens
        # 按 token 数线性估计 action head 耗时（粗估）
        _action_est_ms = _lm_ms * _n_action_tokens / _total_tokens
        print(
            f"[OpenVLAOFT][PROFILE] total_seq={_total_tokens} "
            f"prefix_tokens={_n_prefix_tokens} action_tokens={_n_action_tokens} "
            f"lm_forward_ms={_lm_ms:.1f} "
            f"action_token_est_ms={_action_est_ms:.1f} "
            f"prefix_est_ms={_lm_ms - _action_est_ms:.1f}",
            flush=True,
        )

        # Extract hidden states for action tokens
        last_hidden_states = outputs.hidden_states[-1]  # (B, seq_len, D)
        assert last_hidden_states.shape[1] == mm_embeddings.shape[1]

        logits_tensor = outputs.logits[
            :,
            n_patches + n_prompt_tokens : n_patches
            + n_prompt_tokens
            + self.action_dim * self.num_action_chunks,
            :,
        ]  # [B, act, vocab_size + 64]

        last_hidden_states = last_hidden_states[
            :, -self.action_dim * self.num_action_chunks - 1 : -1
        ]

        logits_tensor[..., : self.vocab_size - self.config.n_action_bins] = -torch.inf
        logits_tensor[..., self.vocab_size :] = -torch.inf

        if do_sample:
            processed_logits_tensor = logits_tensor / kwargs["temperature"]
            top_k = min(
                kwargs["top_k"], processed_logits_tensor.size(-1)
            )  # Safety check
            if top_k > 0:
                logits_warper = TopKLogitsWarper(
                    top_k
                )  # since here is logprob instead of logits, we use 0 instead of -inf
                processed_logits_tensor = logits_warper(None, processed_logits_tensor)
            processed_logprob_tensor = F.log_softmax(
                processed_logits_tensor, dim=-1
            )  # [B, act, vocab_size + 64]

            probs_tensor = torch.exp(
                processed_logprob_tensor
            )  # [B, act, vocab_size + 64]
            probs_flat = probs_tensor.view(
                -1, processed_logprob_tensor.shape[-1]
            )  # [B * act, vocab_size + 64]

            sample_flat = torch.multinomial(
                probs_flat, num_samples=1, replacement=True
            )  # [B * act, 1]
            idxs = sample_flat.view(
                processed_logprob_tensor.shape[0], processed_logprob_tensor.shape[1]
            )  # [B, act]
        else:
            processed_logits_tensor = logits_tensor
            idxs = processed_logits_tensor.argmax(dim=-1)  # [B, act]

        # assert torch.all(idxs >= 0) and torch.all(idxs < self.config.n_action_bins)
        # generated_ids = idxs + (self.vocab_size - self.config.n_action_bins)
        assert torch.all(
            idxs >= self.vocab_size - self.config.n_action_bins
        ) and torch.all(idxs < self.vocab_size)

        chunk_action_tokens = idxs.reshape(-1, self.action_dim)
        predicted_action_token_ids = chunk_action_tokens.cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1
        )
        # normalized_actions = self.bin_centers[discretized_actions]
        normalized_actions = np.asarray(
            [self.bin_centers[da] for da in discretized_actions]
        )  # [B, dim]
        normalized_actions = normalized_actions.reshape(-1, self.action_dim)

        # Unnormalize predicted actions
        actions = self._unnormalize_actions(normalized_actions, self.unnorm_key)
        actions = actions.reshape(idxs.shape)

        action_logits = processed_logits_tensor
        action_logits[..., : self.vocab_size - self.config.n_action_bins] = -torch.inf
        action_logits[..., self.vocab_size :] = -torch.inf

        chunk_logprobs = compute_logprobs_from_logits(logits=action_logits, target=idxs)

        if hasattr(self, "value_head") and calculate_values:
            hidden_features = last_hidden_states[
                :, -self.action_dim * self.num_action_chunks
            ]  # [batch_size, hidden_dim]

            chunk_values = self.value_head(hidden_features)  # [batch_size, 1]
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        chunk_actions = torch.as_tensor(
            actions.reshape(-1, self.num_action_chunks, self.action_dim)
        )
        chunk_action_tokens = idxs.reshape(-1, self.num_action_chunks, self.action_dim)

        forward_inputs["action_tokens"] = chunk_action_tokens

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }

        return chunk_actions, result

    @torch.no_grad()
    def predict_action_batch_staged(
        self,
        env_obs=None,
        calculate_values=True,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Two-stage inference with profiler annotations.

        Stage 1 (backbone): vision encoding + text prefill, saved as past_key_values.
        Stage 2 (action head): action token prefill using KV cache from stage 1.

        Sequence layout after _build_embedding:
          [BOS(1)] + [patch(n_patches)] + [text(n_prompt_tokens-1)] + [action(n_action_tokens)]
        """
        do_sample = kwargs.pop("do_sample")

        assert env_obs is not None, "predict_action_batch_staged requires env_obs"

        task_descriptions = [
            f"In: What action should the robot take to {t.lower()}?\nOut: "
            for t in env_obs["task_descriptions"]
        ]
        if env_obs["main_images"].ndim == 4:
            env_obs["main_images"] = env_obs["main_images"].unsqueeze(1)
        assert env_obs["main_images"].ndim == 5

        all_images = [env_obs["main_images"].permute(0, 1, 4, 2, 3)]
        if self.vision_backbone.get_num_images_in_input() > 1:
            if env_obs["wrist_images"].ndim == 4:
                env_obs["wrist_images"] = env_obs["wrist_images"].unsqueeze(1)
            assert env_obs["wrist_images"].ndim == 5
            wrist_imgs = env_obs["wrist_images"].permute(0, 1, 4, 2, 3)
            all_images.extend([wrist_imgs[:, i] for i in range(wrist_imgs.shape[1])])

        device = next(self.parameters()).device
        precision = next(self.parameters()).dtype

        primary_image = all_images.pop(0)
        inputs = self.input_processor(
            text=task_descriptions,
            images={"images": primary_image},
            proprio_states=env_obs["states"],
            padding="max_length",
            max_length=self.max_prompt_length,
        )

        if all_images:
            all_wrist_pixel_values = [
                self.input_processor(
                    text=task_descriptions,
                    images={"images": wrist_image.unsqueeze(1)},
                    proprio_states=env_obs["states"],
                    padding="max_length",
                    max_length=self.max_prompt_length,
                )["pixel_values"]
                for wrist_image in all_images
            ]
            inputs["pixel_values"] = torch.cat(
                [inputs["pixel_values"]] + all_wrist_pixel_values, dim=1
            )

        input_ids = inputs["input_ids"].to(device=device, dtype=torch.long)
        attention_mask = inputs["attention_mask"].to(device=device, dtype=torch.bool)
        pixel_values = inputs["pixel_values"].to(device=device, dtype=precision)

        B, N, C, H, W = pixel_values.shape
        pixel_values = pixel_values.reshape(B, N * C, H, W)

        forward_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

        n_action_tokens = self.action_dim * self.num_action_chunks

        input_ids, attention_mask = self._prepare_input_for_action_prediction(
            input_ids, attention_mask
        )

        # --- Build full multimodal embeddings ---
        # Layout: [BOS(1)] + [patch(n_patches)] + [text(n_prompt_tokens-1)] + [action(n_action_tokens)]
        with torch.profiler.record_function("model.backbone.openvla_oft._build_embedding"):
            mm_embeddings, mm_attention_mask = self._build_embedding(
                input_ids, attention_mask, pixel_values
            )

        # Split at action boundary
        prefix_len = mm_embeddings.shape[1] - n_action_tokens
        prefix_embeddings = mm_embeddings[:, :prefix_len, :]
        prefix_attention_mask = mm_attention_mask[:, :prefix_len]
        prefix_position_ids = prefix_attention_mask.long().cumsum(dim=1) - 1

        action_embeddings = mm_embeddings[:, prefix_len:, :]
        # Stage 2 需要完整的 attention_mask（长度 = prefix_len + n_action_tokens），
        # 这样 _update_causal_mask 里 target_length 才能正确设为 342，
        # 生成 causal_mask 形状 (n_action_tokens, 342) 而不是 (n_action_tokens, n_action_tokens+1)。
        full_attention_mask = mm_attention_mask
        action_position_ids = (
            torch.arange(prefix_len, prefix_len + n_action_tokens, device=device)
            .unsqueeze(0)
            .expand(B, -1)
        )

        # --- Stage 1: prefix prefill (vision backbone + text tokens) ---
        with torch.profiler.record_function("model.backbone.openvla_oft.prefix_prefill"):
            prefix_outputs = self.language_model(
                input_ids=None,
                attention_mask=prefix_attention_mask,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=prefix_embeddings,
                labels=None,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        past_key_values = prefix_outputs.past_key_values

        # --- Stage 2: action token prefill using KV cache ---
        # use_cache=True 使 transformers 从 DynamicCache 读取正确的 past_seen_tokens(286)，
        # 从而 cache_position=arange(286,342)，causal_mask 形状为 (56,342)，与 KV 维度匹配。
        # full_attention_mask(shape [B,342]) 告知 _update_causal_mask target_length=342。
        with torch.profiler.record_function("model.action_head.openvla_oft.action_prefill"):
            action_outputs = self.language_model(
                input_ids=None,
                attention_mask=full_attention_mask,
                position_ids=action_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=action_embeddings,
                labels=None,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

        # action_outputs.logits: (B, n_action_tokens, vocab_size)
        logits_tensor = action_outputs.logits
        last_hidden_states = action_outputs.hidden_states[-1]  # (B, n_action_tokens, D)

        logits_tensor[..., : self.vocab_size - self.config.n_action_bins] = -torch.inf
        logits_tensor[..., self.vocab_size :] = -torch.inf

        if do_sample:
            processed_logits_tensor = logits_tensor / kwargs["temperature"]
            top_k = min(kwargs["top_k"], processed_logits_tensor.size(-1))
            if top_k > 0:
                logits_warper = TopKLogitsWarper(top_k)
                processed_logits_tensor = logits_warper(None, processed_logits_tensor)
            processed_logprob_tensor = F.log_softmax(processed_logits_tensor, dim=-1)
            probs_flat = torch.exp(processed_logprob_tensor).view(-1, processed_logprob_tensor.shape[-1])
            idxs = torch.multinomial(probs_flat, num_samples=1, replacement=True).view(
                processed_logprob_tensor.shape[0], processed_logprob_tensor.shape[1]
            )
        else:
            processed_logits_tensor = logits_tensor
            idxs = processed_logits_tensor.argmax(dim=-1)

        assert torch.all(
            idxs >= self.vocab_size - self.config.n_action_bins
        ) and torch.all(idxs < self.vocab_size)

        chunk_action_tokens = idxs.reshape(-1, self.action_dim)
        predicted_action_token_ids = chunk_action_tokens.cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1
        )
        normalized_actions = np.asarray(
            [self.bin_centers[da] for da in discretized_actions]
        ).reshape(-1, self.action_dim)

        actions = self._unnormalize_actions(normalized_actions, self.unnorm_key)
        actions = actions.reshape(idxs.shape)

        action_logits = processed_logits_tensor
        action_logits[..., : self.vocab_size - self.config.n_action_bins] = -torch.inf
        action_logits[..., self.vocab_size :] = -torch.inf

        chunk_logprobs = compute_logprobs_from_logits(logits=action_logits, target=idxs)

        if hasattr(self, "value_head") and calculate_values:
            # use the last action token's hidden state for value estimation
            hidden_features = last_hidden_states[:, -1]
            chunk_values = self.value_head(hidden_features)
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        chunk_actions = torch.as_tensor(
            actions.reshape(-1, self.num_action_chunks, self.action_dim)
        )
        forward_inputs["action_tokens"] = idxs.reshape(-1, self.num_action_chunks, self.action_dim)

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }

        return chunk_actions, result

    def get_vision_features(self, _obs: dict) -> torch.Tensor | None:
        # Fallback to obs-level similarity when vision-only extraction is unavailable.
        return None

    def preprocess_for_train(self, data):
        # action-token: [bsz, chunk-step, action-dim] -> [bsz, chunk-step x action-dim]
        for key in ["action_tokens"]:
            value = data[key]
            data[key] = value.reshape(
                value.shape[0],
                self.action_dim * self.num_action_chunks,
                *value.shape[3:],
            )
        return data

    def setup_config_and_processor(self, model_config, input_processor):
        self.vocab_size = (
            model_config.text_config.vocab_size - model_config.pad_to_multiple_of
        )
        self.bins = np.linspace(-1, 1, model_config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        action_norm_stats = self._get_action_stats()
        self.min_action = np.array(action_norm_stats["q01"])
        self.max_action = np.array(action_norm_stats["q99"])
        self.action_scale = 1.0

        self.input_processor = input_processor

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def default_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.FloatTensor = None,
        output_hidden_states: bool = False,
        forward_inputs: Optional[dict[str, torch.Tensor]] = None,
        compute_logprobs: bool = False,
        compute_entropy: bool = False,
        compute_values: bool = False,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        if forward_inputs is not None:
            forward_inputs = self.preprocess_for_train(forward_inputs)
            input_ids = forward_inputs["input_ids"]
            attention_mask = forward_inputs["attention_mask"]
            pixel_values = forward_inputs["pixel_values"]

            action_tokens = forward_inputs["action_tokens"]

        assert torch.all(input_ids[:, 0] == 1)
        assert torch.all(attention_mask[:, 0] == 1)
        # last token is space ` `
        assert torch.all(input_ids[:, -1] == 29871)
        assert torch.all(attention_mask[:, -1] == 1)

        attention_mask = attention_mask.to(torch.long)
        # llm inputs
        input_ids, attention_mask = self._prepare_input_for_action_prediction(
            input_ids, attention_mask
        )
        assert torch.all(input_ids[:, -1] == STOP_INDEX)  # [B, L + act + 1, D]
        assert torch.all(
            input_ids[:, -self.action_dim * self.num_action_chunks - 2] == 29871
        )
        assert torch.all(
            attention_mask[:, -2 - self.action_dim * self.num_action_chunks :] == 1
        )  # [B, L + act + 1]

        # multimodal
        mm_embeddings, mm_attention_mask = self._build_embedding(
            input_ids, attention_mask, pixel_values
        )
        multimodal_position_ids = mm_attention_mask.cumsum(dim=1) - 1

        if compute_values:
            output_hidden_states = True

        # Forward pass through language model
        outputs = self.language_model(
            input_ids=None,
            attention_mask=mm_attention_mask,
            position_ids=multimodal_position_ids,
            past_key_values=None,
            inputs_embeds=mm_embeddings,
            labels=None,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if not compute_logprobs and not compute_values:
            return outputs

        if compute_logprobs:
            logits = outputs.logits[
                :, -self.action_dim * self.num_action_chunks - 1 : -1
            ]  # [B, action-dim, vocab-size]

            processed_logits_tensor = logits / kwargs["temperature"]
            top_k = min(
                kwargs["top_k"], processed_logits_tensor.size(-1)
            )  # Safety check
            if top_k > 0:
                logits_warper = TopKLogitsWarper(
                    top_k
                )  # since here is logprob instead of logits, we use 0 instead of -inf
                processed_logits_tensor = logits_warper(None, processed_logits_tensor)

            action_logits = processed_logits_tensor
            action_logits[
                ..., : self.vocab_size - self.config.n_action_bins
            ] = -torch.inf
            action_logits[..., self.vocab_size :] = -torch.inf

            logprobs = compute_logprobs_from_logits(
                logits=action_logits, target=action_tokens
            )

            entropy = None
            if compute_entropy:
                entropy = compute_entropy_from_logits(logits=action_logits)

        if hasattr(self, "value_head") and compute_values:
            last_hidden_state = outputs.hidden_states[-1]
            hidden_features = last_hidden_state[
                :, -self.action_dim * self.num_action_chunks - 1
            ]  # [batch_size, hidden_dim]
            values = self.value_head(hidden_features)
        else:
            values = None

        result = {
            "logprobs": logprobs,
            "entropy": entropy,
            "values": values,
        }

        return result
