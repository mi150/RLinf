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

import copy
import gc
from typing import Any, Literal

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.config import SupportedModel
from rlinf.data.embodied_io_struct import (
    RolloutResult,
)
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.models.embodiment.feature_cache import FeatureCache, FeatureCacheConfig
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.placement import HybridComponentPlacement


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.should_stop = False

        self.actor_group_name = cfg.actor.group_name
        self.device = self.torch_platform.current_device()

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

        self.placement = HybridComponentPlacement(cfg, Cluster())

        actor_world_size = self.placement.get_world_size("actor")
        self.actor_weight_src_rank = self._rank % actor_world_size
        self.rollout_epoch = cfg.algorithm.get("rollout_epoch", 1)
        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.expert_model = None

        # Sync weight comm options
        max_ctas = cfg.rollout.get("sync_weight_nccl_max_ctas", None)
        min_ctas = cfg.rollout.get("sync_weight_nccl_min_ctas", None)
        self._sync_weight_comm_options = CollectiveGroupOptions(
            accel_max_ctas=max_ctas, accel_min_ctas=min_ctas
        )
        self.total_num_train_envs = cfg.env.train.total_num_envs
        self.total_num_eval_envs = cfg.env.eval.total_num_envs
        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num

        self.train_batch_size = (
            self.total_num_train_envs // self._world_size // self.num_pipeline_stages
        )
        self.eval_batch_size = (
            self.total_num_eval_envs // self._world_size // self.num_pipeline_stages
        )
        self.enable_cuda_graph = cfg.rollout.get("enable_cuda_graph", False)
        self.enable_eval = cfg.runner.val_check_interval > 0 or cfg.runner.only_eval

        self.n_train_chunk_steps = (
            cfg.env.train.max_steps_per_rollout_epoch
            // cfg.actor.model.num_action_chunks
        )
        self.n_eval_chunk_steps = (
            cfg.env.eval.max_steps_per_rollout_epoch
            // cfg.actor.model.num_action_chunks
        )
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.version = 0
        self.finished_episodes = None
        self._step_counter_by_seed: dict[int, int] = {}
        self._warned_missing_env_seeds = False
        self._rollout_seed_update_interval = int(
            self.cfg.env.train.get("rollout_seed_update_interval_global_steps", 0)
        )
        self._rollout_seed_group: int | None = None

    def init_worker(self):
        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path
            rollout_model_config.feature_cache = self.cfg.algorithm.get(
                "feature_cache", None
            )

        self.hf_model: BasePolicy = get_model(rollout_model_config)

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            self.hf_model.load_state_dict(model_dict)

        if self.cfg.rollout.get("expert_model", None):
            expert_model_config = copy.deepcopy(self.cfg.actor.model)
            with open_dict(expert_model_config):
                expert_model_config.precision = self.cfg.rollout.expert_model.precision
                expert_model_config.model_path = (
                    self.cfg.rollout.expert_model.model_path
                )
            self.expert_model = get_model(expert_model_config)

            if self.cfg.runner.get("expert_ckpt_path", None):
                expert_model_dict = torch.load(self.cfg.runner.expert_ckpt_path)
                self.expert_model.load_state_dict(expert_model_dict)

        self.hf_model.eval()
        if self.expert_model is not None:
            self.expert_model.eval()

        feature_cache_cfg = self.cfg.algorithm.get("feature_cache", None)
        if feature_cache_cfg is not None:
            invalidate_on_weight_update = feature_cache_cfg.get(
                "invalidate_on_weight_update", None
            )
            if invalidate_on_weight_update is None:
                invalidate_on_weight_update = not self.cfg.actor.model.get(
                    "freeze_backbone", False
                )
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
            self.hf_model.feature_cache = FeatureCache(cache_config)
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
        else:
            self.hf_model.feature_cache = None
            print("[FeatureCache][CONFIG] feature_cache_cfg_missing=True", flush=True)

        if self.cfg.rollout.get("enable_torch_compile", False):
            mode = self.cfg.rollout.get(
                "torch_compile_mode", "max-autotune-no-cudagraphs"
            )
            self.hf_model.enable_torch_compile(mode=mode)
        if self.enable_cuda_graph and not self.enable_offload:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
            )

        self.dst_ranks = {
            "train": self._setup_dst_ranks(
                self.total_num_train_envs // self.num_pipeline_stages
            ),
        }
        self.src_ranks = {
            "train": self._setup_src_ranks(
                self.total_num_train_envs // self.num_pipeline_stages
            ),
        }
        if self.enable_eval:
            self.dst_ranks["eval"] = self._setup_dst_ranks(
                self.total_num_eval_envs // self.num_pipeline_stages
            )
            self.src_ranks["eval"] = self._setup_src_ranks(
                self.total_num_eval_envs // self.num_pipeline_stages
            )

        self.log_info(f"Rollout worker initialized with dst_ranks: {self.dst_ranks}")
        self.log_info(f"Rollout worker initialized with src_ranks: {self.src_ranks}")
        self.setup_sample_params()
        if self.enable_offload:
            self.offload_model()

    def setup_sample_params(self):
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_train"]
            if self._sampling_params["do_sample"]
            else 1.0,
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

        self._eval_sampling_params = {
            "do_sample": True
            if self._sampling_params.get("temperature_eval", -1) > 0
            else False,
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

        if self.expert_model is not None:
            self._dagger_sampling_params = {
                "beta": self.cfg.algorithm.get("dagger", {}).get("init_beta", 0.5),
                "beta_schedule": self.cfg.algorithm.get("dagger", {}).get(
                    "beta_schedule", "exponential"
                ),
                "beta_min": self.cfg.algorithm.get("dagger", {}).get("beta_min", 0.05),
                "beta_decay": self.cfg.algorithm.get("dagger", {}).get(
                    "beta_decay", 0.99
                ),
            }

    def update_dagger_beta(self):
        if self.expert_model is None:
            return

        if self._dagger_sampling_params["beta_schedule"] == "exponential":
            self._dagger_sampling_params["beta"] = max(
                self._dagger_sampling_params["beta_min"],
                self._dagger_sampling_params["beta"]
                * self._dagger_sampling_params["beta_decay"],
            )
        else:
            raise NotImplementedError(
                f"Beta schedule {self._dagger_sampling_params['beta_schedule']} is not implemented"
            )

    def _setup_dst_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        """Compute env peer ranks for this rollout worker.

        This mapping supports both one-to-many and many-to-one env/rollout layouts.
        The returned ranks are used as communication counterparts for receiving env
        outputs and sending action chunks.

        Args:
            batch_size: Total env batch size per pipeline stage across all workers.

        Returns:
            Ordered ``(env_rank, batch_size)`` tuples this rollout worker should
            send action chunks to.
        """
        env_world_size = self.placement.get_world_size("env")
        rollout_world_size = self.placement.get_world_size("rollout")
        return CommMapper.get_dst_ranks(
            batch_size=batch_size,
            src_world_size=rollout_world_size,
            dst_world_size=env_world_size,
            src_rank=self._rank,
        )

    def _setup_src_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        """Compute env source ranks and sizes for receiving env outputs."""
        env_world_size = self.placement.get_world_size("env")
        rollout_world_size = self.placement.get_world_size("rollout")
        return CommMapper.get_src_ranks(
            batch_size=batch_size,
            src_world_size=env_world_size,
            dst_world_size=rollout_world_size,
            dst_rank=self._rank,
        )

    @Worker.timer("predict")
    def predict(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
        env_seeds: torch.Tensor | None = None,
        step_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.CNN_POLICY,
        ]:
            if self.cfg.algorithm.loss_type == "embodied_dagger":
                kwargs = {"mode": "eval"}
            else:
                kwargs = {"mode": mode}

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.CNN_POLICY,
            SupportedModel.FLOW_POLICY,
            SupportedModel.MLP_POLICY,
        ]:
            kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        only_save_expert = self.cfg.algorithm.get("dagger", {}).get(
            "only_save_expert", True
        )

        if mode == "train" and self.expert_model is not None:
            # training with expert model. Beta-probability acting.
            use_expert = torch.rand(1).item() < self._dagger_sampling_params["beta"]
        else:
            use_expert = False

        with torch.no_grad():
            expert_label_flag = False
            # Decide which model to act via use_expert
            if use_expert:
                actions, result = self.expert_model.predict_action_batch(
                    env_obs=env_obs,
                    env_seeds=env_seeds,
                    step_indices=step_indices,
                    **kwargs,
                )
                expert_label_flag = True
            else:
                print("Using HF model for action prediction.")
                actions, result = self.hf_model.predict_action_batch(
                    env_obs=env_obs,
                    env_seeds=env_seeds,
                    step_indices=step_indices,
                    **kwargs,
                )

            # Decide re-label or not
            if (
                not only_save_expert  # only re-label in classic dagger mode
                and not use_expert  # only re-label if not using expert
                and self.expert_model is not None  # only re-label if expert exists
                and mode == "train"  # only re-label in train mode
            ):
                _, expert_result = self.expert_model.predict_action_batch(
                    env_obs=env_obs,
                    env_seeds=env_seeds,
                    step_indices=step_indices,
                    **kwargs,
                )
                expert_forward_inputs = expert_result["forward_inputs"]
                expert_target = expert_forward_inputs.get(
                    "model_action", expert_forward_inputs.get("action")
                )
                if expert_target is not None:
                    result["forward_inputs"]["model_action"] = expert_target
                expert_label_flag = True

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        result["expert_label_flag"] = bool(expert_label_flag)
        return actions, result

    def get_bootstrap_values(
        self, final_obs: dict[str, Any] | None
    ) -> torch.Tensor | None:
        if final_obs is None:
            return None
        if not (
            hasattr(self.hf_model, "value_head") or hasattr(self.hf_model, "q_head")
        ):
            return None
        with torch.no_grad():
            actions, result = self.predict(final_obs)
            if "prev_values" in result and result["prev_values"] is not None:
                final_values = result["prev_values"]
            else:
                final_values = torch.zeros_like(actions[:, :1], dtype=torch.float32)
        return final_values[:, :1].cpu().contiguous()

    async def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""
        param_state_dict = await self.recv(
            self.actor_group_name,
            src_rank=self.actor_weight_src_rank,
            async_op=True,
            options=self._sync_weight_comm_options,
        ).async_wait()
        self.hf_model.load_state_dict(param_state_dict)
        feature_cache = getattr(self.hf_model, "feature_cache", None)
        if (
            feature_cache is not None
            and feature_cache.config.invalidate_on_weight_update
        ):
            feature_cache.invalidate_all()

        del param_state_dict
        gc.collect()
        self.torch_platform.empty_cache()

    def _build_step_indices(self, env_seeds: torch.Tensor | None) -> torch.Tensor | None:
        if env_seeds is None:
            return None
        step_indices = []
        for seed in env_seeds.tolist():
            step_indices.append(self._step_counter_by_seed.get(int(seed), 0))
        return torch.as_tensor(step_indices, dtype=torch.int64)

    def _advance_step_indices(self, env_seeds: torch.Tensor | None) -> None:
        if env_seeds is None:
            return
        for seed in env_seeds.tolist():
            seed_int = int(seed)
            self._step_counter_by_seed[seed_int] = (
                self._step_counter_by_seed.get(seed_int, 0) + 1
            )

    def _reset_done_step_indices(
        self, env_seeds: torch.Tensor | None, dones: torch.Tensor | None
    ) -> None:
        if env_seeds is None or dones is None:
            return
        done_mask = dones[:, -1] if dones.dim() > 1 else dones
        for seed, done in zip(env_seeds.tolist(), done_mask.tolist()):
            if bool(done):
                self._step_counter_by_seed[int(seed)] = 0

    @Worker.timer("generate_one_epoch")
    async def generate_one_epoch(self, input_channel: Channel, output_channel: Channel):
        self.update_dagger_beta()
        for _ in range(self.n_train_chunk_steps):
            for _ in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel)
                env_seeds = env_output.get("env_seeds")
                if (
                    not self._warned_missing_env_seeds
                    and getattr(self.hf_model, "feature_cache", None) is not None
                    and self.hf_model.feature_cache.config.enabled
                    and env_seeds is None
                ):
                    print(
                        "Feature cache enabled but env_seeds are missing from env output; "
                        "cache keys cannot be aligned across steps."
                    , flush=True)
                    self._warned_missing_env_seeds = True
                self._reset_done_step_indices(env_seeds, env_output.get("dones"))
                step_indices = self._build_step_indices(env_seeds)
                feature_cache = getattr(self.hf_model, "feature_cache", None)
                if (
                    feature_cache is not None
                    and feature_cache.config.enabled
                    and env_seeds is not None
                    and step_indices is not None
                ):
                    seeds_preview = env_seeds.tolist()[:8]
                    steps_preview = step_indices.tolist()[:8]
                    print(
                        "[FeatureCache][QUERY_ATTEMPT] "
                        f"global_step={self.version} "
                        f"batch={env_seeds.shape[0]} "
                        f"seeds_preview={seeds_preview} "
                        f"steps_preview={steps_preview}",
                        flush=True,
                    )
                actions, result = self.predict(
                    env_output["obs"],
                    env_seeds=env_seeds,
                    step_indices=step_indices,
                )
                self._advance_step_indices(env_seeds)

                save_flags = None
                if result.get("expert_label_flag", False):
                    save_flags = torch.full(
                        (actions.shape[0], self.cfg.actor.model.num_action_chunks),
                        True,
                        dtype=torch.bool,
                        device=actions.device,
                    )
                rollout_result = RolloutResult(
                    actions=actions,
                    prev_logprobs=result["prev_logprobs"]
                    if self.collect_prev_infos
                    else None,
                    prev_values=result["prev_values"]
                    if self.collect_prev_infos
                    else None,
                    bootstrap_values=self.get_bootstrap_values(
                        env_output.get("final_obs", None)
                    ),
                    save_flags=save_flags,
                    forward_inputs=result["forward_inputs"],
                    versions=torch.full_like(
                        result["prev_logprobs"],
                        float(self.version),
                        dtype=torch.float32,
                    ),
                )
                self.send_rollout_result(output_channel, rollout_result, mode="train")
        for _ in range(self.num_pipeline_stages):
            env_output = await self.recv_env_output(input_channel)
            env_seeds = env_output.get("env_seeds")
            if (
                not self._warned_missing_env_seeds
                and getattr(self.hf_model, "feature_cache", None) is not None
                and self.hf_model.feature_cache.config.enabled
                and env_seeds is None
            ):
                print(
                    "Feature cache enabled but env_seeds are missing from env output; "
                    "cache keys cannot be aligned across steps."
                , flush=True)
                self._warned_missing_env_seeds = True
            self._reset_done_step_indices(env_seeds, env_output.get("dones"))
            step_indices = self._build_step_indices(env_seeds)
            feature_cache = getattr(self.hf_model, "feature_cache", None)
            if (
                feature_cache is not None
                and feature_cache.config.enabled
                and env_seeds is not None
                and step_indices is not None
            ):
                seeds_preview = env_seeds.tolist()[:8]
                steps_preview = step_indices.tolist()[:8]
                print(
                    "[FeatureCache][QUERY_ATTEMPT] "
                    f"global_step={self.version} "
                    f"batch={env_seeds.shape[0]} "
                    f"seeds_preview={seeds_preview} "
                    f"steps_preview={steps_preview}",
                    flush=True,
                )
            actions, result = self.predict(
                env_output["obs"],
                env_seeds=env_seeds,
                step_indices=step_indices,
            )
            self._advance_step_indices(env_seeds)

            rollout_result = RolloutResult(
                actions=actions,
                prev_values=result["prev_values"] if self.collect_prev_infos else None,
                bootstrap_values=self.get_bootstrap_values(
                    env_output.get("final_obs", None)
                ),
            )
            self.send_rollout_result(output_channel, rollout_result, mode="train")

    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
    ):
        if self.enable_offload:
            self.reload_model()

        feature_cache = getattr(self.hf_model, "feature_cache", None)
        cache_enabled = bool(
            feature_cache is not None and feature_cache.config.enabled
        )
        if cache_enabled and feature_cache.config.mode == "cross_global_same_step":
            # Enforce per-generate step alignment (step0, step1, ...) across global steps.
            self._step_counter_by_seed.clear()
            print(
                "[FeatureCache][STEP_ALIGN] reset step_counter_by_seed at generate start "
                "for mode=cross_global_same_step",
                flush=True,
            )
        aggregate_cache_metrics = {
            "feature_cache/hits": 0.0,
            "feature_cache/same_step_hits": 0.0,
            "feature_cache/misses": 0.0,
            "feature_cache/invalidations": 0.0,
            "feature_cache/cached_seeds": 0.0,
            "feature_cache/cached_entries": 0.0,
            "feature_cache/enabled": 1.0 if cache_enabled else 0.0,
        }
        for _ in tqdm(
            range(self.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            await self.generate_one_epoch(input_channel, output_channel)
            cache_metrics = self._collect_feature_cache_metrics(log_to_stdout=True)
            if cache_metrics:
                aggregate_cache_metrics["feature_cache/hits"] += cache_metrics[
                    "feature_cache/hits"
                ]
                aggregate_cache_metrics["feature_cache/same_step_hits"] += cache_metrics[
                    "feature_cache/same_step_hits"
                ]
                aggregate_cache_metrics["feature_cache/misses"] += cache_metrics[
                    "feature_cache/misses"
                ]
                aggregate_cache_metrics["feature_cache/invalidations"] += cache_metrics[
                    "feature_cache/invalidations"
                ]
                aggregate_cache_metrics["feature_cache/cached_seeds"] = cache_metrics[
                    "feature_cache/cached_seeds"
                ]
                aggregate_cache_metrics["feature_cache/cached_entries"] = cache_metrics[
                    "feature_cache/cached_entries"
                ]

        if self.enable_offload:
            self.offload_model()

        total = (
            aggregate_cache_metrics["feature_cache/hits"]
            + aggregate_cache_metrics["feature_cache/misses"]
        )
        aggregate_cache_metrics["feature_cache/hit_rate"] = (
            100.0 * aggregate_cache_metrics["feature_cache/hits"] / total
            if total > 0
            else 0.0
        )
        if (
            not cache_enabled
            and total == 0
            and aggregate_cache_metrics["feature_cache/invalidations"] == 0
        ):
            return {}
        return aggregate_cache_metrics

    def _collect_feature_cache_metrics(self, log_to_stdout: bool = False) -> dict[str, float]:
        feature_cache = getattr(self.hf_model, "feature_cache", None)
        if feature_cache is None:
            return {}
        stats = feature_cache.get_stats()
        total = stats.hits + stats.misses
        hit_rate = (100.0 * stats.hits / total) if total > 0 else 0.0
        if log_to_stdout:
            print(
                f"[FeatureCache] hits={stats.hits} same_step_hits={stats.same_step_hits} "
                f"misses={stats.misses} "
                f"invalidations={stats.invalidations} hit_rate={hit_rate:.2f}% "
                f"cached_seeds={feature_cache.num_cached_seeds()} "
                f"cached_entries={feature_cache.num_cached_entries()}"
            , flush=True)
        feature_cache.reset_stats()
        return {
            "feature_cache/hits": float(stats.hits),
            "feature_cache/same_step_hits": float(stats.same_step_hits),
            "feature_cache/misses": float(stats.misses),
            "feature_cache/invalidations": float(stats.invalidations),
            "feature_cache/hit_rate": float(hit_rate),
            "feature_cache/cached_seeds": float(feature_cache.num_cached_seeds()),
            "feature_cache/cached_entries": float(feature_cache.num_cached_entries()),
        }

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.reload_model()
        for _ in tqdm(
            range(self.cfg.algorithm.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(self.n_eval_chunk_steps):
                for _ in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel, mode="eval")
                    actions, _ = self.predict(env_output["obs"], mode="eval")
                    self.send_chunk_actions(output_channel, actions, mode="eval")

        if self.enable_offload:
            self.offload_model()

    def offload_model(self):
        if self.enable_cuda_graph:
            self.hf_model.release_cuda_graph()
        self.hf_model.to("cpu")
        self.torch_platform.empty_cache()

    def reload_model(self):
        self.hf_model.to(self.device)
        if self.enable_cuda_graph:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
            )

    async def recv_env_output(
        self, input_channel: Channel, mode: Literal["train", "eval"] = "train"
    ) -> dict[str, Any]:
        """Receive env outputs from mapped env ranks and merge if needed.

        Args:
            input_channel: Channel carrying env->rollout outputs.
            mode: Rollout mode, either ``"train"`` or ``"eval"``.

        Returns:
            A single env output dict. When multiple env ranks are mapped to this
            rollout worker, outputs are merged on batch dimension.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_ranks[mode]
        obs_batches = []
        for src_rank, expected_size in src_ranks_and_sizes:
            obs_batch = await input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_obs"
                ),
                async_op=True,
            ).async_wait()
            actual_size = self._infer_env_batch_size(obs_batch)
            assert actual_size == expected_size, (
                f"Expected env output batch size {expected_size} from env rank {src_rank}, "
                f"got {actual_size}."
            )
            obs_batches.append(obs_batch)
        return self._merge_obs_batches(obs_batches)

    def _split_actions(
        self, actions: torch.Tensor | np.ndarray, sizes: list[int]
    ) -> list[torch.Tensor | np.ndarray]:
        """Split rollout actions into size-specified shards along dim-0.

        Args:
            actions: Model-predicted action chunk batch (tensor or ndarray).
            sizes: Batch sizes for each destination env rank.

        Returns:
            A list of action shards aligned with destination rank order.
        """
        assert sum(sizes) == actions.shape[0], (
            f"Number of actions ({actions.shape[0]}) must equal split sizes sum ({sum(sizes)})."
        )
        if isinstance(actions, np.ndarray):
            split_indices = np.cumsum(sizes[:-1]).tolist()
            return list(np.split(actions, split_indices, axis=0))
        return list(torch.split(actions, sizes, dim=0))

    @staticmethod
    def _infer_env_batch_size(obs_batch: dict[str, Any]) -> int:
        obs = obs_batch["obs"] if "obs" in obs_batch else obs_batch
        for key in ("states", "main_images", "task_descriptions"):
            value = obs.get(key)
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, list):
                return len(value)
        raise ValueError("Cannot infer batch size from env obs.")

    @staticmethod
    def _merge_obs_batches(obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not obs_batches:
            return {}
        obs_dicts = [
            obs_batch["obs"] if "obs" in obs_batch else obs_batch
            for obs_batch in obs_batches
        ]
        final_obs_list = [obs_batch.get("final_obs", None) for obs_batch in obs_batches]

        def _merge_obs_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
            merged: dict[str, Any] = {}
            for key in dicts[0].keys():
                values = [obs_dict[key] for obs_dict in dicts]
                first_non_none = next(
                    (value for value in values if value is not None), None
                )
                if first_non_none is None:
                    merged[key] = None
                elif isinstance(first_non_none, torch.Tensor):
                    merged[key] = torch.cat(values, dim=0)
                elif isinstance(first_non_none, list):
                    merged[key] = [item for sublist in values for item in sublist]
                else:
                    merged[key] = values
            return merged

        merged_obs = _merge_obs_dicts(obs_dicts)
        merged_final_obs = None
        if any(final_obs is not None for final_obs in final_obs_list):
            final_obs_or_obs = [
                final_obs if final_obs is not None else obs_dict
                for obs_dict, final_obs in zip(obs_dicts, final_obs_list)
            ]
            merged_final_obs = _merge_obs_dicts(final_obs_or_obs)
        merged_env_seeds = None
        env_seeds_list = [obs_batch.get("env_seeds", None) for obs_batch in obs_batches]
        if all(seeds is not None for seeds in env_seeds_list):
            merged_env_seeds = torch.cat(env_seeds_list, dim=0)  # type: ignore[arg-type]
        merged_dones = None
        dones_list = [obs_batch.get("dones", None) for obs_batch in obs_batches]
        if all(dones is not None for dones in dones_list):
            merged_dones = torch.cat(dones_list, dim=0)  # type: ignore[arg-type]
        return {
            "obs": merged_obs,
            "final_obs": merged_final_obs,
            "dones": merged_dones,
            "env_seeds": merged_env_seeds,
        }

    def send_chunk_actions(
        self,
        output_channel: Channel,
        chunk_actions: torch.Tensor | np.ndarray,
        mode: Literal["train", "eval"] = "train",
    ):
        """Send action shards to mapped env ranks.

        Args:
            output_channel: Channel carrying rollout->env action chunks.
            chunk_actions: Predicted action chunk batch (tensor or ndarray).
            mode: Rollout mode, either ``"train"`` or ``"eval"``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_ranks[mode]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        chunk_actions_split = self._split_actions(chunk_actions, split_sizes)
        for (dst_rank, _), chunk_action_i in zip(
            dst_ranks_and_sizes, chunk_actions_split
        ):
            if isinstance(chunk_action_i, torch.Tensor):
                chunk_action_i = (
                    chunk_action_i.detach().cpu().contiguous()
                )  # for evaluation
            output_channel.put(
                chunk_action_i,
                key=CommMapper.build_channel_key(
                    self._rank, dst_rank, extra=f"{mode}_actions"
                ),
                async_op=True,
            )

    def _split_rollout_result(
        self, rollout_result: RolloutResult, sizes: list[int]
    ) -> list[RolloutResult]:
        def _split_optional_tensor(
            tensor: torch.Tensor | None,
        ) -> tuple[torch.Tensor | None, ...]:
            if tensor is None:
                return tuple(None for _ in sizes)
            return tuple(torch.split(tensor, sizes, dim=0))

        split_actions = _split_optional_tensor(rollout_result.actions)
        split_prev_logprobs = _split_optional_tensor(rollout_result.prev_logprobs)
        split_prev_values = _split_optional_tensor(rollout_result.prev_values)
        split_bootstrap_values = _split_optional_tensor(rollout_result.bootstrap_values)
        split_save_flags = _split_optional_tensor(rollout_result.save_flags)
        split_versions = _split_optional_tensor(rollout_result.versions)
        split_forward_inputs = (
            [{} for _ in sizes]
            if not rollout_result.forward_inputs
            else [
                {
                    key: torch.split(value, sizes, dim=0)[idx]
                    for key, value in rollout_result.forward_inputs.items()
                }
                for idx in range(len(sizes))
            ]
        )

        return [
            RolloutResult(
                actions=split_actions[idx],
                prev_logprobs=split_prev_logprobs[idx],
                prev_values=split_prev_values[idx],
                bootstrap_values=split_bootstrap_values[idx],
                save_flags=split_save_flags[idx],
                forward_inputs=split_forward_inputs[idx],
                versions=split_versions[idx],
            )
            for idx in range(len(sizes))
        ]

    def send_rollout_result(
        self,
        output_channel: Channel,
        rollout_result: RolloutResult,
        mode: Literal["train", "eval"] = "train",
    ):
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_ranks[mode]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        split_rollout_results = self._split_rollout_result(rollout_result, split_sizes)
        for (dst_rank, _), rollout_result_i in zip(
            dst_ranks_and_sizes, split_rollout_results
        ):
            output_channel.put(
                rollout_result_i,
                key=CommMapper.build_channel_key(
                    self._rank, dst_rank, extra=f"{mode}_rollout_results"
                ),
                async_op=True,
            )

    def set_global_step(self, global_step: int):
        self.version = global_step
        print(
            f"[FeatureCache][GLOBAL_STEP] rollout_worker_rank={self._rank} set_global_step={global_step}",
            flush=True,
        )
        if (
            self.cfg.env.train.get("use_fixed_rollout_seeds", False)
            and self._rollout_seed_update_interval > 0
        ):
            new_seed_group = global_step // self._rollout_seed_update_interval
            if self._rollout_seed_group is None:
                self._rollout_seed_group = new_seed_group
            elif new_seed_group != self._rollout_seed_group:
                self._rollout_seed_group = new_seed_group
                self._step_counter_by_seed.clear()
                feature_cache = getattr(self.hf_model, "feature_cache", None)
                if feature_cache is not None and feature_cache.config.enabled:
                    feature_cache.invalidate_all()
                    print(
                        "[FeatureCache][ENV_SEED_ROTATE_INVALIDATE] "
                        f"rollout_worker_rank={self._rank} global_step={global_step} "
                        f"seed_group={new_seed_group} cache_cleared=True",
                        flush=True,
                    )
        if self.finished_episodes is None:
            self.finished_episodes = (
                self.version * self.total_num_train_envs * self.rollout_epoch
            )
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
