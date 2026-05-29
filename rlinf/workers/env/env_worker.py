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
import os
import re
from collections import defaultdict
from typing import Any, Literal

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    EnvOutput,
    RolloutResult,
    Trajectory,
)
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.wrappers import RecordVideo
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.nested_dict_process import (
    copy_dict_tensor,
    split_dict,
    update_nested_cfg,
)
from rlinf.utils.placement import HybridComponentPlacement


class EnvWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.train_video_cnt = 0
        self.eval_video_cnt = 0
        self.should_stop = False

        self.env_list = []
        self.eval_env_list = []

        self.last_obs_list = []
        self.last_intervened_info_list = []
        self.rollout_epoch = self.cfg.algorithm.get("rollout_epoch", 1)
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.stage_num = self.cfg.rollout.pipeline_stage_num

        self.reward_mode = self.cfg.get("reward", {}).get("reward_mode", "per_step")
        self._interact_step_count = 0
        self._probe_warmup_steps = self.cfg.env.train.get("probe_cfg", {}).get(
            "warmup_steps", 0
        )
        self._probe_resume_pending = False
        # If resuming, schedule probe state loading after envs are initialized
        _resume_dir = self.cfg.runner.get("resume_dir", None)
        if _resume_dir:
            _probe_state_dir = os.path.join(str(_resume_dir), "probe_state")
            if os.path.isdir(_probe_state_dir):
                self._probe_resume_dir = _probe_state_dir
                self._probe_resume_pending = True
                # Also skip warmup since probe already has data from before resume
                # Extract step number from resume_dir path (e.g. global_step_2 → 2)
                _step_match = re.search(r"global_step_(\d+)", str(_resume_dir))
                if _step_match:
                    self._interact_step_count = int(_step_match.group(1))
                    print(
                        f"[probe-resume] will load probe state from {_probe_state_dir}, "
                        f"setting step_count={self._interact_step_count}"
                    )
            else:
                print(
                    f"[probe-resume] no probe_state dir found at {_probe_state_dir}, starting fresh"
                )
        if self.cfg.get("reward", {}).get("use_reward_model", False):
            self.reward_weight = self.cfg.reward.get("reward_weight", 1.0)
            self.env_reward_weight = self.cfg.reward.get("env_reward_weight", 0.0)

        # Env configurations
        self.enable_offload = self.cfg.env.train.get("enable_offload", False)
        self.only_eval = getattr(self.cfg.runner, "only_eval", False)
        self.enable_eval = self.cfg.runner.val_check_interval > 0 or self.only_eval
        if not self.only_eval:
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs // self._world_size // self.stage_num
            )
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs // self._world_size // self.stage_num
            )
        self.n_train_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        self.n_eval_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        self.actor_split_num = self.get_actor_split_num()

    def init_worker(self):
        self.dst_rank_map = self._setup_dst_rank_map()
        self.src_rank_map = self._setup_src_rank_map()
        self.log_info(f"Env worker initialized with dst_rank_map: {self.dst_rank_map}")
        self.log_info(f"Env worker initialized with src_rank_map: {self.src_rank_map}")
        train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
        eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)

        # This is a barrier to ensure all envs' initial setup upon import is done
        # Essential for RealWorld env to ensure initial ROS node setup is done
        self.broadcast(
            True,
            groups=[(self._group_name, list(range(self._world_size)))],
        )

        self.update_env_cfg()

        train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
        eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)

        if not self.only_eval:
            self.env_list = self._setup_env_and_wrappers(
                env_cls=train_env_cls,
                env_cfg=self.cfg.env.train,
                num_envs_per_stage=self.train_num_envs_per_stage,
            )
        if self.enable_eval:
            self.eval_env_list = self._setup_env_and_wrappers(
                env_cls=eval_env_cls,
                env_cfg=self.cfg.env.eval,
                num_envs_per_stage=self.eval_num_envs_per_stage,
            )

        # Resume probe state if available
        if self._probe_resume_pending:
            for stage_id in range(self.stage_num):
                env_obj = self.env_list[stage_id]
                while hasattr(env_obj, "env") and not hasattr(env_obj, "probe"):
                    env_obj = env_obj.env
                if hasattr(env_obj, "probe") and env_obj.probe is not None:
                    _pkl = os.path.join(
                        self._probe_resume_dir,
                        f"probe_rank{self._rank}_stage{stage_id}.pkl",
                    )
                    if os.path.exists(_pkl):
                        env_obj.probe._load_probe_state(_pkl)
                        print(
                            f"[probe-resume] rank={self._rank} stage={stage_id}: loaded {_pkl}"
                        )
                    else:
                        print(
                            f"[probe-resume] rank={self._rank} stage={stage_id}: {_pkl} not found, using initial probe"
                        )
            self._probe_resume_pending = False

        if not self.only_eval:
            self._init_env()

    def update_env_cfg(self):
        # train env
        train_override_cfgs = self.cfg.env.train.get("override_cfgs", None)
        if train_override_cfgs is not None:
            assert len(train_override_cfgs) > self._rank, (
                f"{len(train_override_cfgs)=} > {self._rank=}"
            )

            general_train_override_cfg = OmegaConf.to_container(
                self.cfg.env.train.get("override_cfg", {}), resolve=True
            )
            override_cfg = OmegaConf.to_container(
                train_override_cfgs[self._rank], resolve=True
            ).copy()

            base_cfg = {}
            base_cfg = update_nested_cfg(base_cfg, general_train_override_cfg)
            base_cfg = update_nested_cfg(base_cfg, override_cfg)
            setattr(self.cfg.env.train, "override_cfg", OmegaConf.create(base_cfg))

        eval_override_cfgs = self.cfg.env.eval.get("override_cfgs", None)
        if eval_override_cfgs is not None:
            assert len(eval_override_cfgs) > self._rank, (
                f"{len(eval_override_cfgs)=} > {self._rank=}"
            )

            general_eval_override_cfg = OmegaConf.to_container(
                self.cfg.env.eval.get("override_cfg", {}), resolve=True
            )
            eval_override_cfg = OmegaConf.to_container(
                eval_override_cfgs[self._rank], resolve=True
            ).copy()
            base_eval_cfg = {}
            base_eval_cfg = update_nested_cfg(base_eval_cfg, general_eval_override_cfg)
            base_eval_cfg = update_nested_cfg(base_eval_cfg, eval_override_cfg)
            setattr(self.cfg.env.eval, "override_cfg", OmegaConf.create(base_eval_cfg))

    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage: int):
        env_list = []

        for stage_id in range(self.stage_num):
            env = env_cls(
                cfg=env_cfg,
                num_envs=num_envs_per_stage,
                seed_offset=self._rank * self.stage_num + stage_id,
                total_num_processes=self._world_size * self.stage_num,
                worker_info=self.worker_info,
            )
            if env_cfg.video_cfg.save_video:
                env = RecordVideo(env, env_cfg.video_cfg)
            if env_cfg.get("data_collection", None) and getattr(
                env_cfg.data_collection, "enabled", False
            ):
                from rlinf.envs.wrappers import CollectEpisode

                env = CollectEpisode(
                    env,
                    save_dir=env_cfg.data_collection.save_dir,
                    rank=self._rank,
                    num_envs=num_envs_per_stage,
                    export_format=getattr(
                        env_cfg.data_collection, "export_format", "pickle"
                    ),
                    robot_type=getattr(env_cfg.data_collection, "robot_type", "panda"),
                    fps=getattr(env_cfg.data_collection, "fps", 10),
                    only_success=getattr(
                        env_cfg.data_collection, "only_success", False
                    ),
                    stats_sample_ratio=getattr(
                        env_cfg.data_collection, "stats_sample_ratio", 0.1
                    ),
                    finalize_interval=getattr(
                        env_cfg.data_collection, "finalize_interval", 100
                    ),
                )
            env_list.append(env)
        return env_list

    def _setup_dst_rank_map(self) -> dict[str, list[tuple[int, int]]]:
        """Compute destination rank map for this env worker.

        This mapping supports both one-to-many and many-to-one env/rollout/reward layouts.
        The returned ranks are used as communication counterparts for both sending
        env outputs and receiving results from rollout and reward workers.

        Returns:
            Destination rank map for this env worker.
            The key is the channel name (e.g. "rollout_train", "reward_train", "rollout_eval"), and the value is a ordered list of tuples of (dst_rank, batch_size).
        """
        dst_rank_map = {
            "rollout_train": CommMapper.get_dst_ranks(
                batch_size=self.cfg.env.train.total_num_envs // self.stage_num,
                src_world_size=self._component_placement.get_world_size("env"),
                dst_world_size=self._component_placement.get_world_size("rollout"),
                src_rank=self._rank,
            ),
        }
        if self.cfg.get("reward", {}).get("use_reward_model", False):
            dst_rank_map.update(
                {
                    "reward_train": CommMapper.get_dst_ranks(
                        batch_size=self.cfg.env.train.total_num_envs // self.stage_num,
                        src_world_size=self._component_placement.get_world_size("env"),
                        dst_world_size=self._component_placement.get_world_size(
                            "reward"
                        ),
                        src_rank=self._rank,
                    ),
                }
            )

        if self.enable_eval:
            dst_rank_map.update(
                {
                    "rollout_eval": CommMapper.get_dst_ranks(
                        batch_size=self.cfg.env.eval.total_num_envs // self.stage_num,
                        src_world_size=self._component_placement.get_world_size("env"),
                        dst_world_size=self._component_placement.get_world_size(
                            "rollout"
                        ),
                        src_rank=self._rank,
                    ),
                }
            )
        return dst_rank_map

    def _setup_src_rank_map(self) -> dict[str, list[tuple[int, int]]]:
        """Compute source rank map for this env worker.

        This mapping supports both one-to-many and many-to-one env/rollout/reward layouts.
        The returned ranks are used as communication counterparts for both receiving results from rollout and reward workers and sending action chunks.

        Returns:
            Source rank map for this env worker.
            The key is the channel name (e.g. "rollout_train", "reward_train", "rollout_eval"), and the value is a ordered list of tuples of (src_rank, batch_size).
        """
        src_rank_map = {
            "rollout_train": CommMapper.get_src_ranks(
                batch_size=self.cfg.env.train.total_num_envs // self.stage_num,
                src_world_size=self._component_placement.get_world_size("rollout"),
                dst_world_size=self._component_placement.get_world_size("env"),
                dst_rank=self._rank,
            ),
        }
        if self.cfg.get("reward", {}).get("use_reward_model", False):
            src_rank_map.update(
                {
                    "reward_train": CommMapper.get_src_ranks(
                        batch_size=self.cfg.env.train.total_num_envs // self.stage_num,
                        src_world_size=self._component_placement.get_world_size(
                            "reward"
                        ),
                        dst_world_size=self._component_placement.get_world_size("env"),
                        dst_rank=self._rank,
                    ),
                }
            )
        if self.enable_eval:
            src_rank_map.update(
                {
                    "rollout_eval": CommMapper.get_src_ranks(
                        batch_size=self.cfg.env.eval.total_num_envs // self.stage_num,
                        src_world_size=self._component_placement.get_world_size(
                            "rollout"
                        ),
                        dst_world_size=self._component_placement.get_world_size("env"),
                        dst_rank=self._rank,
                    ),
                }
            )
        return src_rank_map

    def _init_env(self):
        for i in range(self.stage_num):
            if self.cfg.env.train.auto_reset:
                extracted_obs, _ = self.env_list[i].reset()
                self.last_obs_list.append(extracted_obs)
                self.last_intervened_info_list.append((None, None))
            if self.enable_offload and hasattr(self.env_list[i], "offload"):
                self.env_list[i].offload()

    @Worker.timer("env_interact_step")
    def env_interact_step(
        self,
        chunk_actions: torch.Tensor,
        stage_id: int,
        forward_inputs=None,
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to interact with the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=chunk_actions,
            env_type=self.cfg.env.train.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=self.cfg.env.train.get("wm_env_type", None),
        )
        env_info = {}

        # Compute denoising_curvature from forward_inputs if available
        denoising_curvature = None
        if forward_inputs is not None and "chains" in forward_inputs:
            try:
                chains = forward_inputs["chains"]  # (B, K, chunk_size, action_dim)
                if isinstance(chains, torch.Tensor):
                    chains = chains.float()
                    deltas = chains[:, 1:] - chains[:, :-1]
                    delta_diffs = deltas[:, 1:] - deltas[:, :-1]
                    B, S = delta_diffs.shape[:2]
                    norms = torch.norm(delta_diffs.reshape(B, S, -1), dim=-1)
                    denoising_curvature = norms.mean(dim=1).cpu().numpy()
            except Exception:
                pass

        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
            self.env_list[stage_id].chunk_step(
                chunk_actions, denoising_curvature=denoising_curvature
            )
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        final_obs = (
            self._build_chunk_final_obs(obs_list, infos_list)
            if self.cfg.get("reward", {}).get("use_reward_model", False)
            else infos["final_observation"]
            if isinstance(infos, dict) and "final_observation" in infos
            else None
        )
        if not self.cfg.env.train.auto_reset:
            if self.cfg.env.train.ignore_terminations:
                if chunk_truncations[:, -1].any():
                    assert chunk_truncations[:, -1].all()
                    if "episode" in infos:
                        for key in infos["episode"]:
                            env_info[key] = infos["episode"][key].cpu()
            else:
                if "episode" in infos:
                    for key in infos["episode"]:
                        env_info[key] = infos["episode"][key].cpu()
        elif chunk_dones.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        intervene_actions = (
            infos["intervene_action"] if "intervene_action" in infos else None
        )
        intervene_flags = infos["intervene_flag"] if "intervene_flag" in infos else None
        if self.cfg.env.train.auto_reset and chunk_dones.any():
            if "intervene_action" in infos["final_info"]:
                intervene_actions = infos["final_info"]["intervene_action"]
                intervene_flags = infos["final_info"]["intervene_flag"]

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
            rewards=chunk_rewards,
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            intervene_actions=intervene_actions,
            intervene_flags=intervene_flags,
        )
        return env_output, env_info

    def env_evaluate_step(
        self, raw_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to evaluate the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.cfg.env.eval.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
        )
        env_info = {}

        obs_list, _, chunk_terminations, chunk_truncations, infos_list = (
            self.eval_env_list[stage_id].chunk_step(chunk_actions)
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        final_obs = (
            self._build_chunk_final_obs(obs_list, infos_list)
            if self.cfg.get("reward", {}).get("use_reward_model", False)
            else infos["final_observation"]
            if isinstance(infos, dict) and "final_observation" in infos
            else None
        )

        # Only record metrics for newly done envs (avoid double-counting in auto_reset=False)
        if not hasattr(self, "_eval_env_done"):
            self._eval_env_done = [
                np.zeros(self.eval_num_envs_per_stage, dtype=bool)
                for _ in range(self.stage_num)
            ]
        newly_done = chunk_dones[:, -1].cpu().numpy() & ~self._eval_env_done[stage_id]
        self._eval_env_done[stage_id] |= chunk_dones[:, -1].cpu().numpy()

        if newly_done.any():
            if "episode" in infos:
                for key in infos["episode"]:
                    val = infos["episode"][key].cpu()
                    env_info[key] = (
                        val[newly_done] if val.shape[0] == len(newly_done) else val
                    )
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][newly_done].cpu()

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
        )
        return env_output, env_info

    def _build_chunk_final_obs(self, obs_list, infos_list):
        """Build per-env terminal observations for a whole chunk.

        Matches the old wrapper semantics:
        - default to the last rollout observation for each env
        - if an env terminated earlier in the chunk, replace that env's observation
          with the true `final_observation` captured at that substep
        """
        if not isinstance(obs_list, (list, tuple)) or len(obs_list) == 0:
            return None

        last_obs = obs_list[-1]
        if not isinstance(last_obs, dict):
            return None

        merged_final_obs = copy_dict_tensor(last_obs)

        if not isinstance(infos_list, (list, tuple)):
            return merged_final_obs

        for step_infos in infos_list:
            if not isinstance(step_infos, dict):
                continue
            if (
                "final_observation" not in step_infos
                or "_final_observation" not in step_infos
            ):
                continue

            final_obs = step_infos["final_observation"]
            reset_mask = step_infos["_final_observation"]
            if final_obs is None or reset_mask is None:
                continue
            reset_mask = (
                reset_mask.detach().cpu().numpy()
                if isinstance(reset_mask, torch.Tensor)
                else np.asarray(reset_mask)
            )
            done_mask = (
                reset_mask.any(axis=-1)
                if reset_mask.ndim > 1
                else reset_mask.astype(bool)
            )
            if not done_mask.any():
                continue

            for key, value in merged_final_obs.items():
                if key not in final_obs:
                    continue

                final_value = final_obs[key]
                if isinstance(value, torch.Tensor) and isinstance(
                    final_value, torch.Tensor
                ):
                    dst_mask = torch.as_tensor(done_mask, device=value.device)
                    src_mask = dst_mask.to(device=final_value.device)
                    merged_final_obs[key][dst_mask] = final_value[src_mask]
                elif isinstance(value, np.ndarray) and isinstance(
                    final_value, np.ndarray
                ):
                    merged_final_obs[key][done_mask] = final_value[done_mask]

        return merged_final_obs

    def recv_chunk_actions(self, input_channel: Channel, mode="train") -> np.ndarray:
        """Receive and merge chunked actions for the current env worker.

        The method fetches one action shard from each mapped rollout source rank
        under a deterministic channel key pattern and concatenates them on the
        batch dimension.

        Args:
            input_channel: Channel carrying rollout->env action chunks.
            mode: Rollout mode, either ``"train"`` or ``"eval"``.

        Returns:
            Concatenated action chunk array with shape ``[num_envs_per_stage, ...]``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_rank_map[f"rollout_{mode}"]
        chunk_action = []
        for src_rank, expected_size in src_ranks_and_sizes:
            action_i = input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_actions"
                ),
            )
            if isinstance(action_i, torch.Tensor):
                action_i = action_i.detach().cpu().numpy()
            else:
                action_i = np.asarray(action_i)
            assert action_i.shape[0] == expected_size, (
                f"Expected action shard size {expected_size} from rollout rank {src_rank}, "
                f"got shape {action_i.shape}."
            )
            chunk_action.append(action_i)
        chunk_action = np.concatenate(chunk_action, axis=0)
        expected_total_size = sum(size for _, size in src_ranks_and_sizes)
        assert chunk_action.shape[0] == expected_total_size, (
            f"Expected concatenated action size {expected_total_size}, got {chunk_action.shape[0]}."
        )
        return chunk_action

    @Worker.timer("recv_rollout_results")
    def recv_rollout_results(
        self, input_channel: Channel, mode="train"
    ) -> RolloutResult:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_rank_map[f"rollout_{mode}"]
        rollout_results: list[RolloutResult] = []

        def _infer_rollout_batch_size(rollout_result: RolloutResult) -> int:
            for field_name in (
                "actions",
                "prev_logprobs",
                "prev_values",
                "bootstrap_values",
                "versions",
            ):
                value = getattr(rollout_result, field_name, None)
                if isinstance(value, torch.Tensor):
                    return value.shape[0]
            if rollout_result.forward_inputs:
                first_tensor = next(iter(rollout_result.forward_inputs.values()))
                if isinstance(first_tensor, torch.Tensor):
                    return first_tensor.shape[0]
            raise ValueError("Cannot infer batch size from rollout result.")

        for src_rank, expected_size in src_ranks_and_sizes:
            rollout_result = input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_rollout_results"
                ),
            )

            actual_size = _infer_rollout_batch_size(rollout_result)
            assert actual_size == expected_size, (
                f"Expected rollout result size {expected_size} from rollout rank {src_rank}, "
                f"got batch size {actual_size}."
            )

            rollout_results.append(rollout_result)

        return RolloutResult.merge_rollout_results(rollout_results)

    @Worker.timer("compute_bootstrap_rewards")
    def compute_bootstrap_rewards(
        self,
        env_output: EnvOutput,
        bootstrap_values: torch.Tensor | None,
        reward_model_output: torch.Tensor | None,
    ) -> torch.Tensor | None:
        rewards = env_output.rewards
        if rewards is None:
            return None

        if reward_model_output is not None:
            reward_model_output = reward_model_output.to(rewards.dtype)
            rewards = (
                self.env_reward_weight * rewards
                + self.reward_weight * reward_model_output
            )

        adjusted_rewards = rewards.clone()
        if (
            bootstrap_values is None
            or not self.cfg.env.train.auto_reset
            or env_output.dones is None
        ):
            return adjusted_rewards

        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        if bootstrap_type == "standard":
            last_step_truncations = env_output.truncations[:, -1]
        else:
            last_step_truncations = env_output.dones[:, -1]

        if not last_step_truncations.any():
            return adjusted_rewards

        final_values = torch.zeros_like(adjusted_rewards[:, -1], dtype=torch.float32)
        final_values[last_step_truncations] = (
            bootstrap_values[last_step_truncations].reshape(-1).to(torch.float32)
        )
        adjusted_rewards[:, -1] += self.cfg.algorithm.gamma * final_values
        return adjusted_rewards

    def finish_rollout(self, mode="train"):
        # reset
        if mode == "train":
            for i in range(self.stage_num):
                if self.cfg.env.train.video_cfg.save_video and isinstance(
                    self.env_list[i], RecordVideo
                ):
                    self.env_list[i].flush_video()
                self.env_list[i].update_reset_state_ids()
        elif mode == "eval":
            for i in range(self.stage_num):
                if self.cfg.env.eval.video_cfg.save_video and isinstance(
                    self.eval_env_list[i], RecordVideo
                ):
                    self.eval_env_list[i].flush_video()
                if not self.cfg.env.eval.auto_reset:
                    self.eval_env_list[i].update_reset_state_ids()

    def send_env_batch(
        self,
        rollout_channel: Channel,
        env_batch: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
    ) -> None:
        """Send split env batches to mapped rollout ranks.

        Each destination rank receives one split batch via a stable key built from
        ``src_rank``, ``dst_rank`` and ``mode``.

        Args:
            rollout_channel: Channel carrying env->rollout outputs.
            env_batch: Env output dictionary for one pipeline stage.
            mode: Rollout mode, either ``"train"`` or ``"eval"``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_rank_map[f"rollout_{mode}"]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        env_batches = split_dict(env_batch, split_sizes)
        for (rank, _), env_batch_i in zip(dst_ranks_and_sizes, env_batches):
            rollout_channel.put(
                item=env_batch_i,
                key=CommMapper.build_channel_key(self._rank, rank, extra=f"{mode}_obs"),
            )

    def send_reward_input(
        self,
        send_channel: Channel,
        reward_input: dict[str, torch.Tensor],
        mode: Literal["train", "eval"] = "train",
    ):
        dst_ranks_and_sizes = self.dst_rank_map[f"reward_{mode}"]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        reward_input_batches = split_dict(reward_input, split_sizes)
        for (rank, _), reward_input_i in zip(dst_ranks_and_sizes, reward_input_batches):
            send_channel.put(
                item=reward_input_i,
                key=CommMapper.build_channel_key(
                    self._rank, rank, extra=f"{mode}_reward_input"
                ),
                async_op=True,
            )

    @Worker.timer("recv_reward_results")
    def recv_reward_results(self, recv_channel: Channel) -> torch.Tensor:
        reward_results: list[torch.Tensor] = []
        src_ranks_and_sizes = self.src_rank_map["reward_train"]
        for src_rank, expected_size in src_ranks_and_sizes:
            rewards = recv_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra="reward_output"
                ),
            )
            actual_size = rewards.shape[0]
            assert actual_size == expected_size, (
                f"Expected reward result size {expected_size} from reward rank {src_rank}, "
                f"got batch size {actual_size}."
            )
            reward_results.append(rewards)
        return torch.cat(reward_results, dim=0)

    @Worker.timer("get_reward_model_output")
    def get_reward_model_output(
        self,
        env_output: EnvOutput,
        send_channel: Channel,
        recv_channel: Channel,
        last_run: bool = False,
    ):
        if self.reward_mode == "per_step":
            reward_input_obs = (
                env_output.final_obs
                if env_output.final_obs is not None
                else env_output.obs
            )
        elif self.reward_mode == "terminal" and env_output.final_obs is not None:
            reward_input_obs = env_output.final_obs
        else:
            return None

        reward_input = {"images": reward_input_obs["main_images"]}
        if last_run:
            reward_input.update(
                {
                    "last_run": torch.ones(
                        (self.train_num_envs_per_stage, 1), dtype=torch.bool
                    )
                }
            )
        self.send_reward_input(send_channel=send_channel, reward_input=reward_input)
        reward_output = self.recv_reward_results(recv_channel=recv_channel)
        if self.reward_mode != "terminal" or reward_output is None:
            return reward_output
        return self._scatter_terminal_reward_output(
            env_output=env_output, reward_output=reward_output
        )

    def _scatter_terminal_reward_output(
        self,
        env_output: EnvOutput,
        reward_output: torch.Tensor,
    ) -> torch.Tensor:
        if env_output.rewards is None or env_output.dones is None:
            return reward_output

        done_envs = env_output.dones.any(dim=1)
        sparse_rewards = torch.zeros_like(env_output.rewards, dtype=reward_output.dtype)
        if not done_envs.any():
            return sparse_rewards

        done_steps = env_output.dones.to(torch.int64).argmax(dim=1)
        sparse_rewards[done_envs, done_steps[done_envs]] = (
            reward_output[done_envs].reshape(-1).to(sparse_rewards.dtype)
        )
        return sparse_rewards

    def bootstrap_step(self) -> list[EnvOutput]:
        def get_zero_dones() -> torch.Tensor:
            return (
                torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                .unsqueeze(1)
                .repeat(1, self.cfg.actor.model.num_action_chunks)
            )

        v17_enabled = self.cfg.env.train.get("v17_continuous_collect", {}).get(
            "enabled", False
        )
        env_outputs: list[EnvOutput] = []
        if not self.cfg.env.train.auto_reset or v17_enabled:
            # baseline (auto_reset=False): always reset at epoch start
            # v17: also reset so every episode starts fresh (no leftover from previous PPO step)
            for stage_id in range(self.stage_num):
                self.env_list[stage_id].is_start = True
                extracted_obs, infos = self.env_list[stage_id].reset()
                dones = get_zero_dones()
                terminations = dones.clone()
                truncations = dones.clone()

                env_output = EnvOutput(
                    obs=extracted_obs,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    final_obs=infos["final_observation"]
                    if "final_observation" in infos
                    else None,
                    intervene_actions=None,
                    intervene_flags=None,
                )
                env_outputs.append(env_output)
        else:
            dones = get_zero_dones()
            terminations = dones.clone()
            truncations = dones.clone()

            for stage_id in range(self.stage_num):
                env_output = EnvOutput(
                    obs=self.last_obs_list[stage_id],
                    rewards=None,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    intervene_actions=self.last_intervened_info_list[stage_id][0],
                    intervene_flags=self.last_intervened_info_list[stage_id][1],
                )
                env_outputs.append(env_output)

        return env_outputs

    def record_env_metrics(
        self, env_metrics: dict[str, list], env_info: dict[str, Any], epoch: int
    ):
        for key, value in env_info.items():
            if (
                not self.cfg.env.train.auto_reset
                and not self.cfg.env.train.ignore_terminations
            ):
                if key in env_metrics and len(env_metrics[key]) > epoch:
                    env_metrics[key][epoch] = value
                else:
                    env_metrics[key].append(value)
            else:
                env_metrics[key].append(value)

    def store_last_obs_and_intervened_info(self, env_output_list: list[EnvOutput]):
        self.last_obs_list = [env_output.obs for env_output in env_output_list]
        self.last_intervened_info_list = [
            (env_output.intervene_actions, env_output.intervene_flags)
            for env_output in env_output_list
        ]

    async def send_rollout_trajectories(
        self, rollout_result: EmbodiedRolloutResult, channel: Channel
    ):
        trajectories: Trajectory = rollout_result.to_splited_trajectories(
            self.actor_split_num
        )
        for trajectory in trajectories:
            channel.put(trajectory, async_op=True)

    @Worker.timer("run_interact_once")
    async def _run_interact_once(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None,
        *,
        cooperative_yield: bool,
    ) -> dict[str, torch.Tensor]:
        self.rollout_results: list[EmbodiedRolloutResult] = [
            EmbodiedRolloutResult(
                max_episode_length=self.cfg.env.train.max_episode_steps,
            )
            for _ in range(self.stage_num)
        ]
        env_metrics = defaultdict(list)

        # v10 dynamic early stop config
        v10_enabled = self.cfg.env.train.get("v10_dynamic_stop", {}).get(
            "enabled", False
        )

        # v17 continuous collection config
        v17_cfg = self.cfg.env.train.get("v17_continuous_collect", {})
        v17_enabled = v17_cfg.get("enabled", False)
        if v17_enabled:
            v17_target = v17_cfg.get("target_trajectories", 256)
            v17_traj_count = [
                0 for _ in range(self.stage_num)
            ]  # completed trajectories per stage
            v17_all_collected = [False for _ in range(self.stage_num)]
            v17_valid_chunks = [
                None for _ in range(self.stage_num)
            ]  # chunk index where target reached
            # Per-env episode start tracking (for loss_mask)
            n_envs = self.train_num_envs_per_stage
            v17_env_ep_start = [
                np.zeros(n_envs, dtype=np.int32) for _ in range(self.stage_num)
            ]
            # loss_mask: [n_chunk_steps, n_envs, num_action_chunks] per epoch, built at epoch end
            v17_epoch_loss_masks = []  # list of per-epoch masks
            v17_global_chunk_idx = [
                0 for _ in range(self.stage_num)
            ]  # total chunk index across epochs
            # Per-env episode tracking: list of (env_id, ep_num, outcome, chunk_start, chunk_end)
            # outcome: 'success', 'timeout', 'probe_cut', 'force_term'
            v17_env_episodes = [[] for _ in range(self.stage_num)]
            v17_env_ep_count = [
                np.zeros(n_envs, dtype=np.int32) for _ in range(self.stage_num)
            ]
            v17_probe_immune = [
                np.zeros(n_envs, dtype=bool) for _ in range(self.stage_num)
            ]  # spared envs immune to probe cut
            v17_probe_ever_flagged = [
                np.zeros(n_envs, dtype=bool) for _ in range(self.stage_num)
            ]  # ever flagged by probe in current episode

        # v17: collect in 1 long epoch, split later; actor uses rollout_epoch for reshape
        actual_rollout_epoch = 1 if v17_enabled else self.rollout_epoch
        import time as _time

        _timing_bootstrap_time = 0.0
        _timing_chunk_loop_time = 0.0
        _timing_bootstrap_value_time = 0.0
        _timing_send_actor_time = 0.0
        if v17_enabled:
            # v17: separate chunks with auto_reset vs without
            _v17_step_with_reset_time = 0.0
            _v17_step_with_reset_count = 0
            _v17_step_no_reset_time = 0.0
            _v17_step_no_reset_count = 0
            _v17_reset_env_count = 0
        else:
            # baseline: all chunks are the same (no auto_reset), track per-epoch
            _bl_step_time = 0.0
            _bl_step_count = 0
            _bl_per_epoch_step_time = []
        for epoch in range(actual_rollout_epoch):
            _bs_t0 = _time.time()
            env_outputs = self.bootstrap_step()
            _timing_bootstrap_time += _time.time() - _bs_t0
            for stage_id in range(self.stage_num):
                env_output: EnvOutput = env_outputs[stage_id]
                env_batch = env_output.to_dict()
                init_send_dict = {
                    "obs": env_batch["obs"],
                    "final_obs": env_batch["final_obs"],
                }
                if v10_enabled or v17_enabled:
                    init_send_dict["env_all_done"] = torch.zeros(
                        self.train_num_envs_per_stage, dtype=torch.int32
                    )
                self.send_env_batch(rollout_channel, init_send_dict)

            # v17: reset probe state at epoch start (if probe enabled)
            if v17_enabled:
                for sid in range(self.stage_num):
                    _env_obj = self.env_list[sid]
                    while hasattr(_env_obj, "env") and not hasattr(_env_obj, "probe"):
                        _env_obj = _env_obj.env
                    if hasattr(_env_obj, "probe") and _env_obj.probe is not None:
                        _env_obj.probe.reset_all()

            # v17: per-epoch loss_mask init
            if v17_enabled:
                num_action_chunks = self.cfg.actor.model.num_action_chunks
                v17_epoch_mask = [
                    torch.ones(
                        self.n_train_chunk_steps,
                        n_envs,
                        num_action_chunks,
                        dtype=torch.bool,
                    )
                    for _ in range(self.stage_num)
                ]
                last_env_output_v17 = dict.fromkeys(range(self.stage_num))

            # v10: per-env done tracking
            if v10_enabled:
                n_envs = self.train_num_envs_per_stage
                env_ever_done = [
                    np.zeros(n_envs, dtype=bool) for _ in range(self.stage_num)
                ]
                env_done_step = [
                    np.full(n_envs, -1, dtype=np.int32) for _ in range(self.stage_num)
                ]
                env_natural_done = [
                    np.zeros(n_envs, dtype=bool) for _ in range(self.stage_num)
                ]  # only natural done, not probe-cut
                probe_predicted_fail = [None for _ in range(self.stage_num)]
                last_env_output = [None for _ in range(self.stage_num)]
                # Reset probe state at epoch start
                for sid in range(self.stage_num):
                    env_obj = self.env_list[sid]
                    while hasattr(env_obj, "env") and not hasattr(env_obj, "probe"):
                        env_obj = env_obj.env
                    if hasattr(env_obj, "probe") and env_obj.probe is not None:
                        env_obj.probe.reset_all()

            _chunk_loop_t0 = _time.time()
            for chunk_step_idx in range(self.n_train_chunk_steps):
                for stage_id in range(self.stage_num):
                    if cooperative_yield:
                        await asyncio.sleep(0)

                    env_output = env_outputs[stage_id]
                    curr_obs = env_output.obs
                    if env_output.intervene_actions is not None:
                        self.rollout_results[stage_id].update_last_actions(
                            env_output.intervene_actions,
                            env_output.intervene_flags,
                        )

                    reward_model_output = None
                    if reward_channel is not None and chunk_step_idx != 0:
                        reward_model_output = self.get_reward_model_output(
                            env_output,
                            send_channel=reward_channel,
                            recv_channel=input_channel,
                        )
                        if reward_model_output is not None:
                            env_metrics["reward_model_output"].append(
                                reward_model_output.detach().float().reshape(-1).cpu()
                            )

                    rollout_result = self.recv_rollout_results(
                        input_channel, mode="train"
                    )
                    rewards = self.compute_bootstrap_rewards(
                        env_output, rollout_result.bootstrap_values, reward_model_output
                    )
                    _v17_mask = (
                        v17_epoch_mask[stage_id][chunk_step_idx]
                        if v17_enabled
                        else None
                    )
                    chunk_step_result = ChunkStepResult(
                        actions=rollout_result.forward_inputs.get("action", None),
                        prev_logprobs=rollout_result.prev_logprobs
                        if self.collect_prev_infos
                        else None,
                        prev_values=rollout_result.prev_values
                        if self.collect_prev_infos
                        else None,
                        forward_inputs=rollout_result.forward_inputs,
                        versions=rollout_result.versions,
                        dones=env_output.dones,
                        truncations=env_output.truncations,
                        terminations=env_output.terminations,
                        rewards=rewards,
                        loss_mask=_v17_mask,
                    )
                    self.rollout_results[stage_id].append_step_result(chunk_step_result)
                    if rollout_result.save_flags is not None:
                        self.rollout_results[stage_id].mark_last_step_with_flags(
                            rollout_result.save_flags
                        )

                    # ── v17: continuous collection with auto_reset ──
                    if v17_enabled and v17_all_collected[stage_id]:
                        # Target reached → dummy mode: skip env.step
                        env_output = last_env_output_v17[stage_id]
                        dummy_dones = torch.ones_like(env_output.dones)
                        env_output = EnvOutput(
                            obs=env_output.obs,
                            final_obs=env_output.final_obs,
                            rewards=torch.zeros_like(env_output.rewards)
                            if env_output.rewards is not None
                            else None,
                            dones=dummy_dones,
                            terminations=torch.zeros_like(env_output.terminations),
                            truncations=dummy_dones,
                            intervene_actions=None,
                            intervene_flags=None,
                        )
                        v17_epoch_mask[stage_id][chunk_step_idx] = (
                            False  # mask out dummy steps
                        )
                        v10_all_done_flag = True
                    # ── v10: skip env.step for done envs ──
                    elif v10_enabled and env_ever_done[stage_id].all():
                        # All envs done → send last obs as dummy, skip env.step
                        env_output = last_env_output[stage_id]
                        # Override dones to True for all envs
                        dummy_dones = torch.ones_like(env_output.dones)
                        env_output = EnvOutput(
                            obs=env_output.obs,
                            final_obs=env_output.final_obs,
                            rewards=torch.zeros_like(env_output.rewards)
                            if env_output.rewards is not None
                            else None,
                            dones=dummy_dones,
                            terminations=torch.zeros_like(env_output.terminations),
                            truncations=dummy_dones,
                            intervene_actions=None,
                            intervene_flags=None,
                        )
                        v10_all_done_flag = True
                    else:
                        v10_all_done_flag = False
                        _step_t0 = _time.time()
                        env_output, env_info = self.env_interact_step(
                            rollout_result.actions,
                            stage_id,
                            forward_inputs=rollout_result.forward_inputs,
                        )
                        _step_dt = _time.time() - _step_t0
                        if v17_enabled:
                            _had_reset = env_output.dones[:, -1].any().item()
                            if _had_reset:
                                _v17_step_with_reset_time += _step_dt
                                _v17_step_with_reset_count += 1
                                _v17_reset_env_count += int(
                                    env_output.dones[:, -1].sum().item()
                                )
                            else:
                                _v17_step_no_reset_time += _step_dt
                                _v17_step_no_reset_count += 1
                        else:
                            _bl_step_time += _step_dt
                            _bl_step_count += 1

                        # ── v17: trajectory counting + probe cut ──
                        if v17_enabled:
                            chunk_dones_last = env_output.dones[:, -1].cpu().numpy()
                            chunk_term_last = (
                                env_output.terminations[:, -1].cpu().numpy()
                            )
                            # Unwrap to get probe (used for both collection and cut)
                            env_obj = self.env_list[stage_id]
                            while hasattr(env_obj, "env") and not hasattr(
                                env_obj, "probe"
                            ):
                                env_obj = env_obj.env
                            _has_probe = (
                                hasattr(env_obj, "probe") and env_obj.probe is not None
                            )
                            # Count naturally completed episodes
                            for ei in range(n_envs):
                                if chunk_dones_last[ei]:
                                    _is_succ = bool(chunk_term_last[ei])
                                    _was_immune = bool(v17_probe_immune[stage_id][ei])
                                    _was_flagged = bool(
                                        v17_probe_ever_flagged[stage_id][ei]
                                    )
                                    outcome = "success" if _is_succ else "timeout"
                                    ep_start = v17_env_ep_start[stage_id][ei]
                                    ep_len = (
                                        v17_global_chunk_idx[stage_id] - ep_start + 1
                                    )
                                    v17_env_episodes[stage_id].append(
                                        (
                                            ei,
                                            int(v17_env_ep_count[stage_id][ei]),
                                            outcome,
                                            int(ep_start),
                                            int(v17_global_chunk_idx[stage_id]),
                                            int(ep_len),
                                            _was_immune,
                                            _was_flagged,
                                        )
                                    )
                                    v17_env_ep_count[stage_id][ei] += 1
                                    v17_traj_count[stage_id] += 1
                                    v17_env_ep_start[stage_id][ei] = (
                                        v17_global_chunk_idx[stage_id] + 1
                                    )
                                    # Clear immunity and flagged on natural done (episode completed)
                                    v17_probe_immune[stage_id][ei] = False
                                    v17_probe_ever_flagged[stage_id][ei] = False
                                    # Collect episode for probe online training + reset probe state for this env
                                    if _has_probe:
                                        env_obj.probe.collect_episode(
                                            ei, is_success=bool(chunk_term_last[ei])
                                        )
                                        env_obj.probe.window_buf[ei] = []
                                        env_obj.probe.predicted_fail[ei] = False
                                        env_obj.probe.trigger_step[ei] = -1
                                        env_obj.probe.reset_env_hidden(ei)

                            # Track ever-flagged status for probe accuracy stats
                            if _has_probe:
                                v17_probe_ever_flagged[stage_id] |= (
                                    env_obj.probe.predicted_fail
                                )

                            # Probe cut: force-reset predicted-fail envs
                            if hasattr(env_obj, "probe") and env_obj.probe is not None:
                                pred_fail = env_obj.probe.predicted_fail.copy()
                                # Only cut envs that are predicted fail AND not already done AND not immune
                                cut_mask = (
                                    pred_fail
                                    & ~chunk_dones_last.astype(bool)
                                    & ~v17_probe_immune[stage_id]
                                )
                                cut_candidates = np.where(cut_mask)[0]
                                # Randomly spare some for online retrain data (cut_ratio < 1.0)
                                _cut_ratio = self.cfg.env.train.get(
                                    "probe_cfg", {}
                                ).get("cut_ratio", 1.0)
                                if len(cut_candidates) > 0 and _cut_ratio < 1.0:
                                    _rng_mask = (
                                        np.random.rand(len(cut_candidates)) < _cut_ratio
                                    )
                                    # Ensure at least 1 is spared if we have enough candidates
                                    if _rng_mask.all() and len(cut_candidates) >= 2:
                                        _rng_mask[
                                            np.random.randint(len(cut_candidates))
                                        ] = False
                                    cut_indices = cut_candidates[_rng_mask]
                                    spared_indices = cut_candidates[~_rng_mask]
                                    # Set immunity: spared envs run to natural timeout (240 steps)
                                    for ei in spared_indices:
                                        v17_probe_immune[stage_id][ei] = True
                                        env_obj.probe.predicted_fail[ei] = False
                                        env_obj.probe.window_buf[ei] = []
                                        env_obj.probe.trigger_step[ei] = -1
                                else:
                                    cut_indices = cut_candidates
                                if (
                                    len(cut_indices) > 0
                                    and self._interact_step_count
                                    > self._probe_warmup_steps
                                ):
                                    # Force cut: reset envs, mark as termination
                                    reset_obs = env_obj.force_cut_envs(
                                        cut_indices, do_reset=True
                                    )
                                    if reset_obs is not None:
                                        # Update env_output obs for cut envs
                                        for key in env_output.obs:
                                            if isinstance(
                                                env_output.obs[key], torch.Tensor
                                            ) and isinstance(
                                                reset_obs.get(key), torch.Tensor
                                            ):
                                                env_output.obs[key][cut_indices] = (
                                                    reset_obs[key][
                                                        range(len(cut_indices))
                                                    ]
                                                )
                                    # Mark as termination (V=0)
                                    env_output.terminations[cut_indices, -1] = True
                                    env_output.dones[cut_indices, -1] = True
                                    for ei in cut_indices:
                                        ep_start = v17_env_ep_start[stage_id][ei]
                                        ep_len = (
                                            v17_global_chunk_idx[stage_id]
                                            - ep_start
                                            + 1
                                        )
                                        v17_env_episodes[stage_id].append(
                                            (
                                                int(ei),
                                                int(v17_env_ep_count[stage_id][ei]),
                                                "probe_cut",
                                                int(ep_start),
                                                int(v17_global_chunk_idx[stage_id]),
                                                int(ep_len),
                                                False,
                                                True,
                                            )
                                        )  # immune=False (was cut), flagged=True
                                        v17_env_ep_count[stage_id][ei] += 1
                                        v17_traj_count[stage_id] += 1
                                        v17_env_ep_start[stage_id][ei] = (
                                            v17_global_chunk_idx[stage_id] + 1
                                        )
                                        v17_probe_ever_flagged[stage_id][ei] = (
                                            False  # reset for next episode
                                        )
                                        # Save score history for cut episode
                                        if env_obj.probe._score_history[ei]:
                                            env_obj.probe._completed_score_histories.append(
                                                {
                                                    "scores": list(
                                                        env_obj.probe._score_history[ei]
                                                    ),
                                                    "outcome": "probe_cut",
                                                    "length": len(
                                                        env_obj.probe._score_history[ei]
                                                    ),
                                                }
                                            )
                                        env_obj.probe._score_history[ei] = []
                                        env_obj.probe._env_ep_chunk[ei] = 0
                                        # Clear feat_buf for cut env (prevent contaminating next episode)
                                        env_obj.probe._ou_feat_buf[ei] = []
                                        env_obj.probe.window_buf[ei] = []
                                        env_obj.probe.predicted_fail[ei] = False
                                        env_obj.probe.trigger_step[ei] = -1
                                        env_obj.probe.reset_env_hidden(ei)
                                    pass  # per-chunk probe cut log removed; see [actprobe] step summary

                            # Check if target reached
                            if (
                                v17_traj_count[stage_id] >= v17_target
                                and not v17_all_collected[stage_id]
                            ):
                                v17_all_collected[stage_id] = True
                                v17_valid_chunks[stage_id] = chunk_step_idx
                                # Force-terminate remaining active envs + mask their incomplete episodes
                                active_envs = np.where(
                                    ~env_output.dones[:, -1].cpu().numpy()
                                )[0]
                                if len(active_envs) > 0:
                                    env_output.terminations[active_envs, -1] = True
                                    env_output.dones[active_envs, -1] = True
                                    # Log force-terminated envs + clear probe feat buf (label unknown)
                                    for ei in active_envs:
                                        ep_start = v17_env_ep_start[stage_id][ei]
                                        ep_len = (
                                            v17_global_chunk_idx[stage_id]
                                            - ep_start
                                            + 1
                                        )
                                        v17_env_episodes[stage_id].append(
                                            (
                                                int(ei),
                                                int(v17_env_ep_count[stage_id][ei]),
                                                "force_term",
                                                int(ep_start),
                                                int(v17_global_chunk_idx[stage_id]),
                                                int(ep_len),
                                                bool(v17_probe_immune[stage_id][ei]),
                                                bool(
                                                    v17_probe_ever_flagged[stage_id][ei]
                                                ),
                                            )
                                        )
                                        v17_env_ep_count[stage_id][ei] += 1
                                        if _has_probe and env_obj.probe._ou_enabled:
                                            env_obj.probe._ou_feat_buf[ei] = []
                                    # Mask out these incomplete episodes
                                    for ei in active_envs:
                                        ep_start_local = v17_env_ep_start[stage_id][
                                            ei
                                        ] - (epoch * self.n_train_chunk_steps)
                                        ep_start_local = max(0, ep_start_local)
                                        v17_epoch_mask[stage_id][
                                            ep_start_local : chunk_step_idx + 1, ei
                                        ] = False
                                # Mask all remaining chunks in this epoch
                                v17_epoch_mask[stage_id][chunk_step_idx + 1 :] = False
                                print(
                                    f"[v17] epoch={epoch} chunk={chunk_step_idx}: "
                                    f"target {v17_target} reached (actual={v17_traj_count[stage_id]}), "
                                    f"masked {len(active_envs)} incomplete envs, switching to dummy"
                                )

                            last_env_output_v17 = {stage_id: env_output}
                            v17_global_chunk_idx[stage_id] += 1

                        if v10_enabled:
                            # Update per-env done tracking
                            chunk_dones = (
                                env_output.dones[:, -1].cpu().numpy()
                            )  # last step of chunk
                            chunk_terminations = (
                                env_output.terminations[:, -1].cpu().numpy()
                            )  # actual task success
                            for ei in range(n_envs):
                                if chunk_dones[ei] and not env_ever_done[stage_id][ei]:
                                    env_ever_done[stage_id][ei] = True
                                    env_natural_done[stage_id][ei] = bool(
                                        chunk_terminations[ei]
                                    )  # True only for real success, not timeout
                                    env_done_step[stage_id][ei] = chunk_step_idx

                            # Get probe predictions from env infos
                            # (probe runs inside chunk_step via libero_env)
                            # Check if probe flagged any env as fail
                            env_obj = self.env_list[stage_id]
                            # Unwrap if needed (e.g., RecordVideo wrapper)
                            while hasattr(env_obj, "env") and not hasattr(
                                env_obj, "probe"
                            ):
                                env_obj = env_obj.env
                            if hasattr(env_obj, "probe") and env_obj.probe is not None:
                                probe_predicted_fail[stage_id] = (
                                    env_obj.probe.predicted_fail.copy()
                                )

                                # Check: all predicted-success envs done → cut predicted-fail envs
                                # Skip cutting during warmup (probe still runs for data collection)
                                if self._interact_step_count > self._probe_warmup_steps:
                                    pred_fail = probe_predicted_fail[stage_id]
                                    pred_success_mask = ~pred_fail
                                    if pred_success_mask.any():
                                        all_pred_success_done = env_ever_done[stage_id][
                                            pred_success_mask
                                        ].all()
                                        if (
                                            all_pred_success_done
                                            and not env_ever_done[stage_id].all()
                                        ):
                                            # Cut remaining (predicted-fail) envs
                                            newly_cut = ~env_ever_done[stage_id]
                                            env_ever_done[stage_id][:] = True
                                            env_done_step[stage_id][newly_cut] = (
                                                chunk_step_idx
                                            )
                                            n_cut = newly_cut.sum()
                                            print(
                                                f"[probe] epoch={epoch} chunk={chunk_step_idx}: "
                                                f"cutting {n_cut} predicted-fail envs (all pred-success done)"
                                            )
                                elif epoch == 0 and chunk_step_idx == 0:
                                    print(
                                        f"[probe] step={self._interact_step_count}: warmup (no cutting)"
                                    )

                            last_env_output[stage_id] = env_output

                        self.record_env_metrics(env_metrics, env_info, epoch)

                    env_batch = env_output.to_dict()  # noqa: this line starts the common path
                    send_dict = {
                        "obs": env_batch["obs"],
                        "final_obs": env_batch["final_obs"],
                    }
                    if v10_enabled or v17_enabled:
                        # Always include env_all_done flag (int32 tensor, batch-sized for split_dict compat)
                        val = 1 if v10_all_done_flag else 0
                        send_dict["env_all_done"] = torch.full(
                            (env_output.dones.shape[0],), val, dtype=torch.int32
                        )
                    self.send_env_batch(rollout_channel, send_dict)
                    if self.collect_transitions:
                        next_obs = (
                            env_output.final_obs
                            if env_output.dones.any() and self.cfg.env.train.auto_reset
                            else env_output.obs
                        )
                        self.rollout_results[stage_id].append_transitions(
                            curr_obs, next_obs
                        )

                    env_outputs[stage_id] = env_output

            _timing_chunk_loop_time += _time.time() - _chunk_loop_t0

            # ── v10: epoch-end logging ──
            if v10_enabled:
                for stage_id in range(self.stage_num):
                    nd = env_natural_done[stage_id]
                    n_natural = nd.sum()
                    n_probe_cut = 0
                    n_probe_fp = 0  # flagged as fail but naturally succeeded before cut
                    n_probe_flagged = 0
                    if probe_predicted_fail[stage_id] is not None:
                        pf = probe_predicted_fail[stage_id]
                        n_probe_flagged = int(pf.sum())
                        n_probe_cut = int(
                            (pf & ~nd).sum()
                        )  # flagged AND not naturally done = actually cut
                        n_probe_fp = int(
                            (pf & nd).sum()
                        )  # flagged but had already naturally succeeded
                    skipped_chunks = 0
                    for ei in range(n_envs):
                        ds = env_done_step[stage_id][ei]
                        if ds >= 0:
                            skipped_chunks += self.n_train_chunk_steps - 1 - ds
                    total_chunks = n_envs * self.n_train_chunk_steps
                    pct_saved = (
                        100.0 * skipped_chunks / total_chunks if total_chunks > 0 else 0
                    )
                    # Count undetected fails: not flagged AND not natural_done
                    n_undetected_fail = 0
                    if probe_predicted_fail[stage_id] is not None:
                        pf = probe_predicted_fail[stage_id]
                        n_undetected_fail = int((~pf & ~nd).sum())
                    # Find cut trigger chunk (earliest chunk where all pred-success were done)
                    cut_chunk = -1
                    for ei in range(n_envs):
                        if (
                            probe_predicted_fail[stage_id] is not None
                            and not probe_predicted_fail[stage_id][ei]
                        ):
                            # predicted success env - its done_step is when it finished
                            ds = env_done_step[stage_id][ei]
                            cut_chunk = max(cut_chunk, ds)
                    # Get current tau from probe
                    probe_tau = -1
                    env_obj_log = self.env_list[stage_id]
                    while hasattr(env_obj_log, "env") and not hasattr(
                        env_obj_log, "probe"
                    ):
                        env_obj_log = env_obj_log.env
                    if hasattr(env_obj_log, "probe") and env_obj_log.probe is not None:
                        probe_tau = env_obj_log.probe.tau
                    # Success env done chunks (compact distribution)
                    succ_done_chunks = sorted(
                        [
                            int(env_done_step[stage_id][ei])
                            for ei in range(n_envs)
                            if nd[ei] and env_done_step[stage_id][ei] >= 0
                        ]
                    )
                    succ_chunks_str = (
                        ",".join(str(c) for c in succ_done_chunks)
                        if succ_done_chunks
                        else "-"
                    )
                    print(
                        f"[probe] epoch={epoch} stage={stage_id}: "
                        f"done={n_natural}/{n_envs}, flagged={n_probe_flagged}, cut={n_probe_cut}, "
                        f"fp={n_probe_fp}, missed={n_undetected_fail}, "
                        f"cut@={cut_chunk}, tau={probe_tau:.4f}, "
                        f"saved={skipped_chunks}/{total_chunks}({pct_saved:.1f}%), "
                        f"succ_chunks={succ_chunks_str}"
                    )

                    # ── Online update: collect labeled episodes ──
                    env_obj = self.env_list[stage_id]
                    while hasattr(env_obj, "env") and not hasattr(env_obj, "probe"):
                        env_obj = env_obj.env
                    if hasattr(env_obj, "probe") and env_obj.probe is not None:
                        _is_warmup = (
                            self._interact_step_count <= self._probe_warmup_steps
                        )
                        env_obj.probe.finalize_epoch(
                            nd, env_done_step[stage_id], is_warmup=_is_warmup
                        )

            # ── v17: epoch-end per-env episode summary ──
            if v17_enabled:
                for stage_id in range(self.stage_num):
                    eps = v17_env_episodes[stage_id]
                    n_success = sum(1 for e in eps if e[2] == "success")
                    n_timeout = sum(1 for e in eps if e[2] == "timeout")
                    n_probe_cut = sum(1 for e in eps if e[2] == "probe_cut")
                    n_force_term = sum(1 for e in eps if e[2] == "force_term")
                    succ_lens = [e[5] for e in eps if e[2] == "success"]
                    fail_lens = [e[5] for e in eps if e[2] in ("timeout", "probe_cut")]
                    force_lens = [e[5] for e in eps if e[2] == "force_term"]
                    avg_succ = sum(succ_lens) / len(succ_lens) if succ_lens else 0
                    avg_fail = sum(fail_lens) / len(fail_lens) if fail_lens else 0
                    avg_force = sum(force_lens) / len(force_lens) if force_lens else 0
                    # Per-env episode count
                    ep_counts = v17_env_ep_count[stage_id]
                    # (autoreset-env summary merged into actprobe below)
                    # ── Probe performance stats (v14-style) ──
                    # missed = timeout (probe didn't catch), cut = probe_cut (probe caught)
                    # cut_rate = how many fails probe caught vs total fails
                    n_fail_total = n_timeout + n_probe_cut
                    cut_rate = (
                        n_probe_cut / n_fail_total * 100 if n_fail_total > 0 else 0
                    )
                    cut_lens = [e[5] for e in eps if e[2] == "probe_cut"]
                    timeout_lens = [e[5] for e in eps if e[2] == "timeout"]
                    avg_cut_len = sum(cut_lens) / len(cut_lens) if cut_lens else 0
                    avg_timeout_len = (
                        sum(timeout_lens) / len(timeout_lens) if timeout_lens else 0
                    )
                    # Get tau from probe
                    _env_obj = self.env_list[stage_id]
                    while hasattr(_env_obj, "env") and not hasattr(_env_obj, "probe"):
                        _env_obj = _env_obj.env
                    _tau = (
                        _env_obj.probe.tau
                        if hasattr(_env_obj, "probe") and _env_obj.probe is not None
                        else -1
                    )
                    # Chunks saved by probe cut vs letting all fail envs timeout
                    # Each probe_cut saved (48 - cut_len) chunks compared to full timeout
                    max_chunks = 48
                    saved_chunks = sum(max_chunks - l for l in cut_lens)
                    potential_chunks = n_fail_total * max_chunks
                    pct_saved = (
                        saved_chunks / potential_chunks * 100
                        if potential_chunks > 0
                        else 0
                    )
                    # Immune env analysis: episodes with was_immune=True (tuple index 6)
                    immune_succ = sum(
                        1 for e in eps if len(e) > 6 and e[6] and e[2] == "success"
                    )  # spared → succeeded (probe fp)
                    immune_timeout = sum(
                        1 for e in eps if len(e) > 6 and e[6] and e[2] == "timeout"
                    )  # spared → timeout (correct, kept for data)
                    # Never-flagged timeout: timeout without ever being flagged (tuple index 7)
                    never_flagged_timeout = sum(
                        1 for e in eps if len(e) > 7 and e[2] == "timeout" and not e[7]
                    )
                    # Flagged but succeeded (not immune, probe flagged but env succeeded before cut)
                    flagged_succ = sum(
                        1
                        for e in eps
                        if len(e) > 7 and e[2] == "success" and e[7] and not e[6]
                    )
                    print(
                        f"[actprobe] stage={stage_id}: "
                        f"succ={n_success}, missed={n_timeout}, cut={n_probe_cut}, "
                        f"cut_rate={cut_rate:.1f}%, tau={_tau:.4f}, "
                        f"avg_len: succ={avg_succ:.1f}, cut={avg_cut_len:.1f}, timeout={avg_timeout_len:.1f}, "
                        f"saved={saved_chunks}/{potential_chunks}({pct_saved:.1f}%)"
                    )
                    print(
                        f"[actprobe-detail] stage={stage_id}: "
                        f"immune_succ={immune_succ}(fp), immune_timeout={immune_timeout}(data), "
                        f"never_flagged_timeout={never_flagged_timeout}(blind), flagged_succ={flagged_succ}(near_fp)"
                    )

            _bv_t0 = _time.time()
            for stage_id in range(self.stage_num):
                env_output = env_outputs[stage_id]
                if env_output.intervene_actions is not None:
                    self.rollout_results[stage_id].update_last_actions(
                        env_output.intervene_actions,
                        env_output.intervene_flags,
                    )

                reward_model_output = None
                if reward_channel is not None:
                    last_run = epoch == self.rollout_epoch - 1
                    reward_model_output = self.get_reward_model_output(
                        env_output,
                        send_channel=reward_channel,
                        recv_channel=input_channel,
                        last_run=last_run,
                    )
                    if reward_model_output is not None:
                        env_metrics["reward_model_output"].append(
                            reward_model_output.detach().float().reshape(-1).cpu()
                        )
                rollout_result = self.recv_rollout_results(input_channel, mode="train")
                rewards = self.compute_bootstrap_rewards(
                    env_output, rollout_result.bootstrap_values, reward_model_output
                )
                chunk_step_result = ChunkStepResult(
                    prev_values=rollout_result.prev_values
                    if self.collect_prev_infos
                    else None,
                    dones=env_output.dones,
                    truncations=env_output.truncations,
                    terminations=env_output.terminations,
                    rewards=rewards,
                )
                self.rollout_results[stage_id].append_step_result(chunk_step_result)
            _timing_bootstrap_value_time += _time.time() - _bv_t0

            # baseline: record per-epoch env_step time
            if not v17_enabled:
                epoch_step_time = _bl_step_time - sum(_bl_per_epoch_step_time)
                _bl_per_epoch_step_time.append(epoch_step_time)

            self.store_last_obs_and_intervened_info(env_outputs)
            self.finish_rollout()

        # ── Timing summary ──
        if v17_enabled:
            avg_with = (
                _v17_step_with_reset_time / _v17_step_with_reset_count
                if _v17_step_with_reset_count
                else 0
            )
            avg_no = (
                _v17_step_no_reset_time / _v17_step_no_reset_count
                if _v17_step_no_reset_count
                else 0
            )
            avg_reset_per_chunk = (
                _v17_reset_env_count / _v17_step_with_reset_count
                if _v17_step_with_reset_count
                else 0
            )
            print(
                f"[v17-timing] bootstrap_reset={_timing_bootstrap_time:.1f}s (1 epoch), "
                f"step_with_reset: {_v17_step_with_reset_count} calls, total={_v17_step_with_reset_time:.1f}s, avg={avg_with:.3f}s/call, "
                f"avg_reset_envs={avg_reset_per_chunk:.1f}"
            )
            print(
                f"[v17-timing] step_no_reset: {_v17_step_no_reset_count} calls, total={_v17_step_no_reset_time:.1f}s, avg={avg_no:.3f}s/call"
            )
            print(
                f"[v17-timing] total_env_step={_v17_step_with_reset_time + _v17_step_no_reset_time:.1f}s, "
                f"reset_overhead={_v17_step_with_reset_time - _v17_step_with_reset_count * avg_no:.1f}s (estimated)"
            )
            # Read precise auto_reset timing from libero_env
            for stage_id in range(self.stage_num):
                env_obj = self.env_list[stage_id]
                while hasattr(env_obj, "env"):
                    env_obj = env_obj.env
                if hasattr(env_obj, "_auto_reset_total_time"):
                    avg_ar = (
                        env_obj._auto_reset_total_time / env_obj._auto_reset_total_count
                        if env_obj._auto_reset_total_count
                        else 0
                    )
                    avg_envs = (
                        env_obj._auto_reset_total_envs / env_obj._auto_reset_total_count
                        if env_obj._auto_reset_total_count
                        else 0
                    )
                    print(
                        f"[v17-timing] stage={stage_id} auto_reset: {env_obj._auto_reset_total_count} calls, "
                        f"total={env_obj._auto_reset_total_time:.1f}s, avg={avg_ar:.3f}s/call, "
                        f"avg_envs={avg_envs:.1f}, total_envs={env_obj._auto_reset_total_envs}"
                    )
                    # Reset for next PPO step
                    env_obj._auto_reset_total_time = 0.0
                    env_obj._auto_reset_total_count = 0
                    env_obj._auto_reset_total_envs = 0
        else:
            avg_step = _bl_step_time / _bl_step_count if _bl_step_count else 0
            avg_bs = (
                _timing_bootstrap_time / actual_rollout_epoch
                if actual_rollout_epoch
                else 0
            )
            print(
                f"[bl-timing] bootstrap_reset={_timing_bootstrap_time:.1f}s ({actual_rollout_epoch} epochs, avg={avg_bs:.1f}s/epoch)"
            )
            print(
                f"[bl-timing] env_step: {_bl_step_count} calls, total={_bl_step_time:.1f}s, avg={avg_step:.3f}s/call"
            )
            print(
                f"[bl-timing] per_epoch_step_time: {[round(t, 1) for t in _bl_per_epoch_step_time]}"
            )
        # Common timers
        _env_step_total = (
            (_v17_step_with_reset_time + _v17_step_no_reset_time)
            if v17_enabled
            else _bl_step_time
        )
        _accounted = (
            _timing_bootstrap_time
            + _timing_chunk_loop_time
            + _timing_bootstrap_value_time
        )
        tag = "v17-timing" if v17_enabled else "bl-timing"
        print(
            f"[{tag}] chunk_loop={_timing_chunk_loop_time:.1f}s, bootstrap_value={_timing_bootstrap_value_time:.1f}s, "
            f"send_actor={_timing_send_actor_time:.1f}s"
        )
        print(
            f"[{tag}] TOTAL: bootstrap={_timing_bootstrap_time:.1f} + chunk_loop={_timing_chunk_loop_time:.1f} + "
            f"bootstrap_value={_timing_bootstrap_value_time:.1f} + send_actor={_timing_send_actor_time:.1f} = "
            f"{_accounted + _timing_send_actor_time:.1f}s"
        )

        # ── Online update: trigger probe retrain at step end ──
        if v17_enabled:
            for stage_id in range(self.stage_num):
                env_obj = self.env_list[stage_id]
                while hasattr(env_obj, "env") and not hasattr(env_obj, "probe"):
                    env_obj = env_obj.env
                if hasattr(env_obj, "probe") and env_obj.probe is not None:
                    env_obj.probe.trim_buffer()
                    n_buf = (
                        len(env_obj.probe._ou_buffer)
                        if env_obj.probe._ou_enabled
                        else 0
                    )
                    n_succ = (
                        sum(1 for e in env_obj.probe._ou_buffer if e["success"])
                        if n_buf > 0
                        else 0
                    )
                    print(
                        f"[v17-probe] stage={stage_id}: buffer={n_buf} eps ({n_succ}S/{n_buf - n_succ}F), "
                        f"step={env_obj.probe._ou_step}"
                    )
                    env_obj.probe.maybe_retrain()
            print(f"[v17] step done: total trajectories={v17_traj_count}")

            # Auto-save probe state every step (into checkpoint dir for resume)
            _ckpt_base = os.path.join(
                "/workspace/RLinf",
                self.cfg.runner.logger.get("log_path", "../results"),
                self.cfg.runner.logger.get("experiment_name", "default"),
                "checkpoints",
                f"global_step_{self._interact_step_count}",
                "probe_state",
            )
            os.makedirs(_ckpt_base, exist_ok=True)
            for stage_id in range(self.stage_num):
                env_obj = self.env_list[stage_id]
                while hasattr(env_obj, "env") and not hasattr(env_obj, "probe"):
                    env_obj = env_obj.env
                if hasattr(env_obj, "probe") and env_obj.probe is not None:
                    if (
                        hasattr(env_obj.probe, "_ou_thread")
                        and env_obj.probe._ou_thread is not None
                    ):
                        env_obj.probe._ou_thread.join(timeout=60)
                    env_obj.probe.apply_pending()
                    _save_path = (
                        f"{_ckpt_base}/probe_rank{self._rank}_stage{stage_id}.pkl"
                    )
                    env_obj.probe.save_probe_state(_save_path)
                    # Save and plot real-time score histories
                    histories = env_obj.probe._completed_score_histories
                    if histories:
                        n_s = sum(1 for h in histories if h["outcome"] == "success")
                        n_f = sum(1 for h in histories if h["outcome"] == "timeout")
                        n_c = sum(1 for h in histories if h["outcome"] == "probe_cut")
                        print(
                            f"[probe-realtime] step={self._interact_step_count} rank={self._rank} stage={stage_id}: "
                            f"{len(histories)} eps ({n_s}S/{n_f}F/{n_c}C), tau={env_obj.probe.tau:.4f}"
                        )
                        # Plot
                        try:
                            import matplotlib

                            matplotlib.use("Agg")
                            import matplotlib.pyplot as plt

                            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                            for h in histories:
                                color = {
                                    "success": "blue",
                                    "timeout": "red",
                                    "probe_cut": "orange",
                                }[h["outcome"]]
                                alpha = 0.15
                                ax.plot(
                                    range(1, len(h["scores"]) + 1),
                                    h["scores"],
                                    color=color,
                                    alpha=alpha,
                                    linewidth=0.8,
                                )
                            ax.axhline(
                                y=env_obj.probe.tau,
                                color="green",
                                linestyle="--",
                                linewidth=2,
                                label=f"tau={env_obj.probe.tau:.3f}",
                            )
                            ax.set_xlabel("Chunk (within episode)")
                            ax.set_ylabel("P(fail) score")
                            ax.set_title(
                                f"Step {self._interact_step_count} rank{self._rank}: real-time scores ({n_s}S/{n_f}F/{n_c}C)"
                            )
                            ax.set_ylim(-0.05, 1.05)
                            ax.set_xlim(0, 50)
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            _plot_path = f"{_ckpt_base}/realtime_scores_rank{self._rank}_stage{stage_id}.png"
                            plt.savefig(_plot_path, dpi=150)
                            plt.close()
                            print(f"[probe-realtime] plot saved: {_plot_path}")
                        except Exception as e:
                            print(f"[probe-realtime] plot failed: {e}")
                        env_obj.probe._completed_score_histories = []

        if v10_enabled:
            for stage_id in range(self.stage_num):
                env_obj = self.env_list[stage_id]
                while hasattr(env_obj, "env") and not hasattr(env_obj, "probe"):
                    env_obj = env_obj.env
                if hasattr(env_obj, "probe") and env_obj.probe is not None:
                    env_obj.probe.maybe_retrain()

            # Auto-save probe state at end of warmup (for v16 resume)
            if (
                self._interact_step_count == self._probe_warmup_steps
                and self._probe_warmup_steps > 0
            ):
                # Use experiment log dir if available, fallback to default
                log_base = self.cfg.runner.logger.get("log_path", "../results")
                exp_name = self.cfg.runner.logger.get("experiment_name", "default")
                save_dir = os.path.join(
                    "/workspace/RLinf", log_base, "probe_state", exp_name
                )
                os.makedirs(save_dir, exist_ok=True)
                for stage_id in range(self.stage_num):
                    env_obj = self.env_list[stage_id]
                    while hasattr(env_obj, "env") and not hasattr(env_obj, "probe"):
                        env_obj = env_obj.env
                    if hasattr(env_obj, "probe") and env_obj.probe is not None:
                        # Wait for retrain thread to finish before saving
                        if (
                            hasattr(env_obj.probe, "_ou_thread")
                            and env_obj.probe._ou_thread is not None
                        ):
                            env_obj.probe._ou_thread.join(timeout=60)
                        save_path = f"{save_dir}/probe_rank{self._rank}_stage{stage_id}_step{self._interact_step_count}.pkl"
                        env_obj.probe.save_probe_state(save_path)

        _sa_t0 = _time.time()
        if actor_channel is not None:
            for stage_id in range(self.stage_num):
                # v17: log loss_mask stats before sending
                if v17_enabled and len(self.rollout_results[stage_id].loss_mask) > 0:
                    mask_stack = torch.stack(
                        self.rollout_results[stage_id].loss_mask, dim=0
                    )
                    total = mask_stack.numel()
                    valid = mask_stack.sum().item()
                    n_chunks = mask_stack.shape[0]
                    n_envs = mask_stack.shape[1]
                    # Count complete episodes from dones in valid region
                    dones_stack = torch.stack(
                        self.rollout_results[stage_id].dones, dim=0
                    )
                    valid_chunk_end = (
                        v17_valid_chunks[stage_id]
                        if v17_valid_chunks[stage_id] is not None
                        else n_chunks
                    )
                    n_complete_eps = dones_stack[:valid_chunk_end, :, -1].sum().item()
                    print(
                        f"[v17-nosplit] stage={stage_id}: shape=[{n_chunks}, {n_envs}], "
                        f"valid={valid}/{total} ({100 * valid / total:.1f}%), "
                        f"valid_chunks={valid_chunk_end}/{n_chunks}, "
                        f"complete_episodes={int(n_complete_eps)}"
                    )
                await self.send_rollout_trajectories(
                    self.rollout_results[stage_id], actor_channel
                )

        _timing_send_actor_time = _time.time() - _sa_t0

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return env_metrics

    @Worker.timer("interact")
    async def interact(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None = None,
    ):
        self._interact_step_count += 1

        # Apply pending probe retrain before step starts (avoid 1-step lag)
        v10_cfg = self.cfg.env.train.get("v10_dynamic_stop", {})
        if v10_cfg.get("enabled", False):
            for stage_id in range(self.stage_num):
                env_obj = self.env_list[stage_id]
                while hasattr(env_obj, "env") and not hasattr(env_obj, "probe"):
                    env_obj = env_obj.env
                if hasattr(env_obj, "probe") and env_obj.probe is not None:
                    env_obj.probe.apply_pending()

        env_metrics = await self._run_interact_once(
            input_channel,
            rollout_channel,
            reward_channel,
            actor_channel,
            cooperative_yield=False,
        )

        for env in self.env_list:
            if self.enable_offload and hasattr(env, "offload"):
                env.offload()

        return env_metrics

    def evaluate(self, input_channel: Channel, rollout_channel: Channel):
        eval_metrics = defaultdict(list)
        # Reset per-env done tracker for eval (avoid double-counting)
        self._eval_env_done = [
            np.zeros(self.eval_num_envs_per_stage, dtype=bool)
            for _ in range(self.stage_num)
        ]

        for eval_rollout_epoch in range(self.cfg.algorithm.eval_rollout_epoch):
            if not self.cfg.env.eval.auto_reset or eval_rollout_epoch == 0:
                for stage_id in range(self.stage_num):
                    self.eval_env_list[stage_id].is_start = True
                    extracted_obs, infos = self.eval_env_list[stage_id].reset()
                    env_output = EnvOutput(
                        obs=extracted_obs,
                        final_obs=infos["final_observation"]
                        if "final_observation" in infos
                        else None,
                    )
                    env_batch = env_output.to_dict()
                    self.send_env_batch(
                        rollout_channel,
                        {
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                        },
                        mode="eval",
                    )

            for eval_step in range(self.n_eval_chunk_steps):
                for stage_id in range(self.stage_num):
                    raw_chunk_actions = self.recv_chunk_actions(
                        input_channel, mode="eval"
                    )
                    env_output, env_info = self.env_evaluate_step(
                        raw_chunk_actions, stage_id
                    )

                    for key, value in env_info.items():
                        eval_metrics[key].append(value)

                    if self.cfg.env.eval.auto_reset:
                        if (
                            eval_rollout_epoch
                            == self.cfg.algorithm.eval_rollout_epoch - 1
                            and eval_step == self.n_eval_chunk_steps - 1
                        ):
                            continue
                    else:
                        if eval_step == self.n_eval_chunk_steps - 1:
                            continue
                    env_batch = env_output.to_dict()
                    self.send_env_batch(
                        rollout_channel,
                        {
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                        },
                        mode="eval",
                    )

            self.finish_rollout(mode="eval")
        for stage_id in range(self.stage_num):
            if self.cfg.env.eval.get("enable_offload", False) and hasattr(
                self.eval_env_list[stage_id], "offload"
            ):
                self.eval_env_list[stage_id].offload()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return eval_metrics

    def get_actor_split_num(self):
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num
