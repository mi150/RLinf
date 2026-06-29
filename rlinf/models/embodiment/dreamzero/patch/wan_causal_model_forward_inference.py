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

import torch
from groot.vla.model.dreamzero.modules.wan2_1_submodule import sinusoidal_embedding_1d


def _forward_blocks(
    self,
    x: torch.Tensor,
    seq_len: int,
    freqs: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    clip_feature: torch.Tensor | None,
    embodiment_id: torch.Tensor | None,
    action: torch.Tensor | None,
    timestep_action: torch.Tensor | None,
    state: torch.Tensor | None,
    kv_cache: list[torch.Tensor],
    current_start_frame: int,
    return_video_pred: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, list[torch.Tensor]]:
    x = x.flatten(start_dim=2).transpose(1, 2)

    B = x.shape[0]
    F = timestep.shape[1]

    if action is not None:
        embodiment_id = torch.tensor([0], device=x.device).repeat(x.shape[0])
        action_features = self.action_encoder(action, timestep_action, embodiment_id)
        state_features = self.state_encoder(state, embodiment_id)
        action_register = torch.cat([action_features, state_features], dim=1)
        action_length = action_features.shape[1]
        action_register_length = action_register.shape[1]
        x = torch.cat([x, action_register], dim=1)
    else:
        state_features = None
        action_length = 0
        action_register_length = None

    if F <= seq_len:
        repeat = (seq_len + F - 1) // F
        timestep = timestep.repeat_interleave(repeat, dim=1)[:, :seq_len]
    else:
        indices = torch.linspace(0, F - 1, seq_len, device=timestep.device, dtype=torch.long)
        timestep = timestep[:, indices]

    if action is not None:
        assert timestep_action is not None
        assert state_features is not None
        stride = timestep_action.shape[1] // state_features.shape[1]
        timestep_state = timestep_action[:, ::stride]
        timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x)
    )
    e = e.unflatten(dim=0, sizes=(B, -1))
    e0 = self.time_projection(e)
    e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

    context = self.text_embedding(context)
    if clip_feature is not None:
        clip_embedding = self.img_emb(clip_feature)
        context = torch.cat([clip_embedding, context], dim=1)

    def create_custom_forward(module):
        def custom_forward(*inputs, **kwargs):
            outputs, updated_kv_cache = module(*inputs, **kwargs)
            return outputs, updated_kv_cache

        return custom_forward

    updated_kv_caches: list[torch.Tensor] = []
    for block_index, block in enumerate(self.blocks):
        kwargs = dict(
            e=e0,
            freqs=freqs,
            freqs_action=self.freqs_action,
            freqs_state=self.freqs_state,
            context=context,
            action_register_length=action_register_length,
            kv_cache=kv_cache[block_index],
            current_start_frame=current_start_frame,
        )
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            x, updated_kv_cache = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x,
                **kwargs,
                use_reentrant=False,
            )
        else:
            x, updated_kv_cache = block(x=x, **kwargs)
        updated_kv_caches.append(updated_kv_cache)

    if action is not None:
        action_noise_pred = x[:, seq_len : seq_len + action_length]
        action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
    else:
        action_noise_pred = None

    if not return_video_pred:
        return None, action_noise_pred, updated_kv_caches

    x_video = x[:, :seq_len]
    e_video = e[:, :seq_len]
    x_video = self.head(x_video, e_video.unsqueeze(2))
    return x_video, action_noise_pred, updated_kv_caches


def _forward_inference(
    self,
    x,
    timestep,
    context,
    seq_len,
    kv_cache: list[torch.Tensor],
    crossattn_cache: list[torch.Tensor],
    current_start_frame: int,
    y=None,
    clip_feature=None,
    action=None,
    timestep_action=None,
    state=None,
    embodiment_id=None,
    return_video_pred: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, list[torch.Tensor]]:
    del crossattn_cache

    if self.model_type == "i2v":
        assert clip_feature is not None and y is not None
    assert context.shape[1] == self.text_len

    if y is not None and self.concat_first_frame_latent:
        x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

    x = self.patch_embedding(x)
    grid_size = torch.tensor(x.shape[2:], dtype=torch.long)

    freqs = self._create_freqs(
        grid_size=grid_size,
        start_frame=current_start_frame,
    )

    x_video, action_noise_pred, updated_kv_caches = self._forward_blocks(
        x=x,
        seq_len=seq_len,
        freqs=freqs,
        timestep=timestep,
        context=context,
        clip_feature=clip_feature,
        embodiment_id=embodiment_id,
        action=action,
        timestep_action=timestep_action,
        state=state,
        kv_cache=kv_cache,
        current_start_frame=current_start_frame,
        return_video_pred=return_video_pred,
    )

    if action_noise_pred is not None:
        action_noise_pred = action_noise_pred.clone()

    if not return_video_pred:
        return None, action_noise_pred, updated_kv_caches

    assert x_video is not None
    x_video = x_video.clone()
    video_noise_pred = self.unpatchify(x_video, grid_size)
    return video_noise_pred, action_noise_pred, updated_kv_caches
