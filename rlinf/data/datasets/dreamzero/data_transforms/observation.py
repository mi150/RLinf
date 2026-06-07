from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.utils.logging import get_logger

logger = get_logger()


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _resize_bt_hwc_uint8(x: Any, h: int, w: int) -> np.ndarray:
    import cv2

    arr = _to_numpy(x)
    if arr.ndim == 3:
        arr = arr[None, ...]
    out = np.empty((arr.shape[0], h, w, 3), dtype=np.uint8)
    for idx, frame in enumerate(arr):
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        out[idx] = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    return out


def _ensure_bt(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x[None, ...]
    return x


@dataclass
class DreamZeroObservationTransform:
    embodiment_tag: str = "libero_sim"

    def convert(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class _LiberoBaseTransform(DreamZeroObservationTransform):
    def _convert_state_fields(self, states: np.ndarray) -> dict[str, np.ndarray]:
        states = _ensure_bt(states.astype(np.float32))
        if states.shape[-1] >= 7:
            eef_pos = states[..., :3]
            ee_ori = states[..., 3:6]
            gripper = states[..., 6:8]
            joint = states[..., :7]
            grip_pos = states[..., -1:]
        else:
            eef_pos = np.zeros((states.shape[0], 3), dtype=np.float32)
            ee_ori = np.zeros((states.shape[0], 3), dtype=np.float32)
            gripper = np.zeros((states.shape[0], 2), dtype=np.float32)
            joint = np.zeros((states.shape[0], 7), dtype=np.float32)
            grip_pos = np.zeros((states.shape[0], 1), dtype=np.float32)
            eef_pos[..., : min(states.shape[-1], 3)] = states[
                ..., : min(states.shape[-1], 3)
            ]
            joint[..., : states.shape[-1]] = states
        return {
            "state.state": states[:, None, :],
            "state.eef_pos": eef_pos[:, None, :],
            "state.ee_ori": ee_ori[:, None, :],
            "state.gripper": gripper[:, None, :],
            "state.joint_position": joint[:, None, :],
            "state.gripper_position": grip_pos[:, None, :],
        }


class DreamZeroLiberoObservationTransform(_LiberoBaseTransform):
    def __init__(self, num_history_frames: int = 4) -> None:
        super().__init__(embodiment_tag="libero_sim")
        self.num_history_frames = max(1, int(num_history_frames))
        self._main_history: list[deque[np.ndarray]] = []
        self._wrist_history: list[deque[np.ndarray]] = []
        self._last_prompts: list[str] | None = None

    def reset(self) -> None:
        self._main_history = []
        self._wrist_history = []
        self._last_prompts = None

    def _ensure_history(self, batch_size: int, prompts: list[str]) -> None:
        if (
            len(self._main_history) != batch_size
            or len(self._wrist_history) != batch_size
            or self._last_prompts != prompts
        ):
            self._main_history = [
                deque(maxlen=self.num_history_frames) for _ in range(batch_size)
            ]
            self._wrist_history = [
                deque(maxlen=self.num_history_frames) for _ in range(batch_size)
            ]
        self._last_prompts = list(prompts)

    def _append_history(
        self,
        histories: list[deque[np.ndarray]],
        frames: np.ndarray,
    ) -> np.ndarray:
        stacked = []
        for idx, frame in enumerate(frames):
            histories[idx].append(frame.copy())
            frames_to_use = list(histories[idx])
            if len(frames_to_use) > 1:
                while len(frames_to_use) < self.num_history_frames:
                    frames_to_use.insert(0, frames_to_use[0])
            stacked.append(np.stack(frames_to_use, axis=0))
        return np.stack(stacked, axis=0)

    def convert(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        main = _resize_bt_hwc_uint8(env_obs["main_images"], 256, 256)
        wrist = _resize_bt_hwc_uint8(env_obs.get("wrist_images", main), 256, 256)
        states = _to_numpy(env_obs.get("states", np.zeros((main.shape[0], 8))))
        prompts = env_obs.get("task_descriptions", [""] * main.shape[0])
        if isinstance(prompts, str):
            prompts = [prompts] * main.shape[0]
        prompts = list(prompts)
        self._ensure_history(main.shape[0], prompts)
        main_video = self._append_history(self._main_history, main)
        wrist_video = self._append_history(self._wrist_history, wrist)
        state_fields = self._convert_state_fields(states)
        return {
            "video.image": main_video,
            "video.wrist_image": wrist_video,
            **state_fields,
            "annotation.task": prompts,
            "annotation.language.action_text": prompts,
            "annotation.language.task_description": prompts,
            "annotation.language.language_instruction": prompts,
            "annotation.language.language_instruction_2": prompts,
            "annotation.language.language_instruction_3": prompts,
        }


class DreamZeroDroidObservationTransform(DreamZeroObservationTransform):
    def __init__(self, target_hw: tuple[int, int] = (176, 320)) -> None:
        super().__init__(embodiment_tag="oxe_droid")
        self.target_hw = target_hw

    def convert(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        main = _resize_bt_hwc_uint8(env_obs["main_images"], *self.target_hw)
        wrist = _resize_bt_hwc_uint8(env_obs.get("wrist_images", main), *self.target_hw)
        prompts = env_obs.get("task_descriptions", [""] * main.shape[0])
        if isinstance(prompts, str):
            prompts = [prompts] * main.shape[0]
        states = _to_numpy(env_obs.get("states", np.zeros((main.shape[0], 8))))
        if states.ndim == 1:
            states = states[None, :]
        return {
            "video.exterior_image_1_left": main,
            "video.exterior_image_2_left": main,
            "video.wrist_image_left": wrist,
            "state.state": states[:, None, :].astype(np.float32),
            "annotation.task": list(prompts),
            "annotation.language.action_text": list(prompts),
            "annotation.language.task_description": list(prompts),
            "annotation.language.language_instruction": list(prompts),
            "annotation.language.language_instruction_2": list(prompts),
            "annotation.language.language_instruction_3": list(prompts),
        }


def _infer_droid_view_hw_from_model_cfg(model_cfg: Any) -> tuple[int, int]:
    h_override = model_cfg.get("dreamzero_droid_view_height", None)
    w_override = model_cfg.get("dreamzero_droid_view_width", None)
    if h_override is not None and w_override is not None:
        return int(h_override), int(w_override)
    model_path = Path(str(model_cfg.get("model_path", ""))).expanduser()
    cfg_path = model_path / "config.json"
    if not cfg_path.is_file():
        return (176, 320)
    try:
        with open(cfg_path) as f:
            cfg_json = json.load(f)
        ah_cfg = cfg_json.get("action_head_cfg", {}).get("config", {})
        dcfg = ah_cfg.get("diffusion_model_cfg", {})
        in_dim = int(dcfg.get("in_dim", -1))
        model_type = str(dcfg.get("model_type", "")).lower()
        frame_seqlen = int(dcfg.get("frame_seqlen", -1))
        if in_dim == 48 or model_type == "ti2v" or frame_seqlen == 50:
            return (160, 320)
    except Exception:
        logger.exception("Failed to infer DROID resize from %s", cfg_path)
    return (176, 320)


def _infer_libero_history_frames_from_model_cfg(model_cfg: Any) -> int:
    override = model_cfg.get("dreamzero_libero_history_frames", None)
    if override is not None:
        return int(override)
    return 4


def build_dreamzero_observation_transform(
    embodiment_tag: str, model_cfg: Any | None = None
) -> DreamZeroObservationTransform:
    tag = str(embodiment_tag).lower()
    if tag in {"libero", "libero_sim", "libero_spatial", "libero_object", "libero_goal"}:
        return DreamZeroLiberoObservationTransform(
            num_history_frames=_infer_libero_history_frames_from_model_cfg(
                model_cfg or {}
            )
        )
    if tag in {"oxe_droid", "droid"}:
        return DreamZeroDroidObservationTransform(
            target_hw=_infer_droid_view_hw_from_model_cfg(model_cfg or {})
        )
    return DreamZeroLiberoObservationTransform()
