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

import asyncio
import gc

import torch
from omegaconf.omegaconf import DictConfig

from rlinf.scheduler import Channel
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class AsyncMultiStepRolloutWorker(MultiStepRolloutWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._generate_task: asyncio.Task = None
        self.staleness_threshold = cfg.algorithm.get("staleness_threshold", None)
        self.num_envs_per_stage = (
            self.cfg.env.train.total_num_envs
            // self._world_size
            // self.num_pipeline_stages
        )
        assert not self.enable_offload, (
            "Offload not supported in AsyncMultiStepRolloutWorker"
        )

        self._background_weight_sync_active = self.cfg.actor.get(
            "sync_weight_no_wait", False
        )
        self._weight_sync_requested = False
        self._weight_sync_work = None
        self._weight_sync_apply_total = 0
        self._weight_sync_coalesced_total = 0
        self._weight_sync_request_total = 0

    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        metric_channel: Channel,
    ):
        assert self._generate_task is None, (
            "generate task is not None but generate function is called."
        )
        self._generate_task = asyncio.create_task(
            self._generate(input_channel, output_channel, metric_channel)
        )
        try:
            await self._generate_task
        except asyncio.CancelledError:
            pass

    async def _generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        metric_channel: Channel,
    ):
        while True:
            if self._background_weight_sync_active:
                await self._poll_background_weight_sync()
            await self.wait_if_stale()
            aggregate_cache_metrics = {
                "rollout/feature_cache/hits": 0.0,
                "rollout/feature_cache/same_step_hits": 0.0,
                "rollout/feature_cache/misses": 0.0,
                "rollout/feature_cache/invalidations": 0.0,
                "rollout/feature_cache/cached_seeds": 0.0,
                "rollout/feature_cache/cached_entries": 0.0,
                "rollout/feature_cache/enabled": float(
                    bool(
                        getattr(self.hf_model, "feature_cache", None) is not None
                        and self.hf_model.feature_cache.config.enabled
                    )
                ),
            }
            for _ in range(self.rollout_epoch):
                await self.generate_one_epoch(input_channel, output_channel)
                cache_metrics = self._collect_feature_cache_metrics(log_to_stdout=True)
                if cache_metrics:
                    aggregate_cache_metrics["rollout/feature_cache/hits"] += cache_metrics[
                        "feature_cache/hits"
                    ]
                    aggregate_cache_metrics[
                        "rollout/feature_cache/same_step_hits"
                    ] += cache_metrics["feature_cache/same_step_hits"]
                    aggregate_cache_metrics["rollout/feature_cache/misses"] += cache_metrics[
                        "feature_cache/misses"
                    ]
                    aggregate_cache_metrics[
                        "rollout/feature_cache/invalidations"
                    ] += cache_metrics["feature_cache/invalidations"]
                    aggregate_cache_metrics["rollout/feature_cache/cached_seeds"] = (
                        cache_metrics["feature_cache/cached_seeds"]
                    )
                    aggregate_cache_metrics["rollout/feature_cache/cached_entries"] = (
                        cache_metrics["feature_cache/cached_entries"]
                    )
            if self.finished_episodes is not None:
                self.finished_episodes += self.total_num_train_envs * self.rollout_epoch
            rollout_metrics = self.pop_execution_times()
            rollout_metrics = {
                f"time/rollout/{k}": v for k, v in rollout_metrics.items()
            }
            total = (
                aggregate_cache_metrics["rollout/feature_cache/hits"]
                + aggregate_cache_metrics["rollout/feature_cache/misses"]
            )
            if total > 0:
                aggregate_cache_metrics["rollout/feature_cache/hit_rate"] = (
                    100.0 * aggregate_cache_metrics["rollout/feature_cache/hits"] / total
                )
            else:
                aggregate_cache_metrics["rollout/feature_cache/hit_rate"] = 0.0
            metric_channel.put(
                {"rank": self._rank, "time": rollout_metrics, "rollout": aggregate_cache_metrics},
                async_op=True,
            )

    async def wait_if_stale(self) -> None:
        if self.staleness_threshold is None:
            return
        assert self.finished_episodes is not None, (
            "finished_episodes should be initialized."
        )
        while True:
            capacity = (
                (self.staleness_threshold + self.version + 1)
                * self.total_num_train_envs
                * self.rollout_epoch
            )
            if (
                self.finished_episodes + self.total_num_train_envs * self.rollout_epoch
                <= capacity
            ):
                break
            await asyncio.sleep(0.01)

    def stop(self):
        if self._generate_task is not None and not self._generate_task.done():
            self._generate_task.cancel()

    def _start_background_weight_sync_if_needed(self):
        if (
            not self._background_weight_sync_active
            or not self._weight_sync_requested
            or self._weight_sync_work is not None
        ):
            return

        self._weight_sync_requested = False
        self._weight_sync_work = self.recv(
            self.actor_group_name,
            src_rank=self.actor_weight_src_rank,
            async_op=True,
            options=self._sync_weight_comm_options,
        )

    def _apply_synced_model_weights(self, param_state_dict):
        self.hf_model.load_state_dict(param_state_dict)
        feature_cache = getattr(self.hf_model, "feature_cache", None)
        if (
            feature_cache is not None
            and feature_cache.config.invalidate_on_weight_update
        ):
            feature_cache.invalidate_all()

        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    async def _poll_background_weight_sync(self):
        self._start_background_weight_sync_if_needed()
        if self._weight_sync_work is None:
            return

        if not self._weight_sync_work.done():
            return

        param_state_dict = await self._weight_sync_work.async_wait()
        self._weight_sync_work = None
        self._apply_synced_model_weights(param_state_dict)
        self._weight_sync_apply_total += 1

        self._start_background_weight_sync_if_needed()

    async def request_actor_sync_model(self):
        self._weight_sync_request_total += 1
        if self._weight_sync_requested or self._weight_sync_work is not None:
            self._weight_sync_coalesced_total += 1
        self._weight_sync_requested = True
        self._start_background_weight_sync_if_needed()
