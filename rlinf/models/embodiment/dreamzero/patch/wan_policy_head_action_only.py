# Copyright 2026 The RLinf Authors.
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

from contextlib import contextmanager
import functools

from diffusers.schedulers.scheduling_utils import SchedulerOutput
import torch


_ACTION_ONLY_VIDEO_CONTEXT: list[bool] = []


def _is_action_only_video_context() -> bool:
    return bool(_ACTION_ONLY_VIDEO_CONTEXT and _ACTION_ONLY_VIDEO_CONTEXT[-1])


@contextmanager
def _action_only_video_context(enabled: bool):
    _ACTION_ONLY_VIDEO_CONTEXT.append(enabled)
    try:
        yield
    finally:
        _ACTION_ONLY_VIDEO_CONTEXT.pop()


def lazy_joint_video_action(original_func):
    @functools.wraps(original_func)
    def wrapped(self, *args, **kwargs):
        return_video = args[3] if len(args) >= 4 else kwargs.get("return_video", True)
        with _action_only_video_context(enabled=not return_video):
            return original_func(self, *args, **kwargs)

    return wrapped


def flow_unipc_step(original_func):
    @functools.wraps(original_func)
    def wrapped(self, *args, **kwargs):
        if len(args) >= 3:
            model_output, timestep, sample = args[:3]
            remaining_args = args[3:]
        else:
            model_output = kwargs["model_output"]
            timestep = kwargs["timestep"]
            sample = kwargs["sample"]
            remaining_args = ()

        if _is_action_only_video_context() and sample.ndim == 5:
            if kwargs.get("return_dict", True) is False:
                return (sample,)
            return SchedulerOutput(prev_sample=sample)

        if len(args) >= 3:
            return original_func(self, model_output, timestep, sample, *remaining_args, **kwargs)
        return original_func(self, **kwargs)

    return wrapped


def _run_diffusion_steps(
    self,
    noisy_input: torch.Tensor,
    timestep: torch.Tensor,
    action: torch.Tensor,
    timestep_action: torch.Tensor,
    state: torch.Tensor,
    embodiment_id: torch.Tensor,
    context: torch.Tensor,
    seq_len: int,
    y: torch.Tensor,
    clip_feature: torch.Tensor,
    kv_caches: list,
    crossattn_caches: list,
    kv_cache_metadata: dict[str, bool | int],
    return_video_pred: bool | None = None,
) -> list[tuple[torch.Tensor | None, torch.Tensor]]:
    if return_video_pred is None:
        return_video_pred = not _is_action_only_video_context()

    predictions = []
    for index, prompt_emb in enumerate(context):
        kv_cache = kv_caches[index]
        crossattn_cache = crossattn_caches[index]
        if not kv_cache_metadata["update_kv_cache"] and self.trt_engine is not None:
            obs_noise_pred, action_noise_pred = self.trt_engine(
                noisy_input,
                timestep,
                action=action,
                timestep_action=timestep_action,
                state=state,
                context=prompt_emb,
                y=y,
                clip_feature=clip_feature,
                kv_cache=kv_cache,
            )
        else:
            obs_noise_pred, action_noise_pred, updated_kv_caches = self.model(
                noisy_input,
                timestep,
                action=action,
                timestep_action=timestep_action,
                state=state,
                embodiment_id=embodiment_id,
                context=prompt_emb,
                seq_len=seq_len,
                y=y,
                clip_feature=clip_feature,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start_frame=kv_cache_metadata["start_frame"],
                return_video_pred=return_video_pred,
            )
            if kv_cache_metadata["update_kv_cache"]:
                for block_index, updated_kv_cache in enumerate(updated_kv_caches):
                    kv_cache[block_index] = updated_kv_cache.clone()
        if obs_noise_pred is not None:
            obs_noise_pred = obs_noise_pred.clone()
        elif not return_video_pred:
            obs_noise_pred = torch.zeros_like(noisy_input)
        if action_noise_pred is not None:
            action_noise_pred = action_noise_pred.clone()
        else:
            action_noise_pred = torch.tensor(0.0, device=obs_noise_pred.device)
        predictions.append((obs_noise_pred, action_noise_pred))
    return self._exchange_predictions(predictions)
