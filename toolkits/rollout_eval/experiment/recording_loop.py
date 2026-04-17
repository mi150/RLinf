"""Recording loop that captures full trajectories alongside standard metrics."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from toolkits.rollout_eval.adapters.model_adapter import ModelAdapterProtocol
from toolkits.rollout_eval.checks.interface_checks import (
    assert_action_batch,
    assert_obs_batch,
)
from toolkits.rollout_eval.config_bridge import EvalRuntimeConfig
from toolkits.rollout_eval.experiment.seedable_env_adapter import SeedableEnvAdapter
from toolkits.rollout_eval.experiment.types import EpisodeTrajectory, StepRecord
from toolkits.rollout_eval.profiling.collector import LatencyCollector
from toolkits.rollout_eval.rollout_types import RolloutLoopResult, infer_batch_size


def run_recording_loop(
    env_adapter: SeedableEnvAdapter,
    model_adapter: ModelAdapterProtocol | Any,
    runtime: EvalRuntimeConfig,
    seed: int | None = None,
    video_path: str | None = None,
    video_fps: int = 10,
    k_b: int | None = None,
) -> tuple[RolloutLoopResult, list[EpisodeTrajectory]]:
    """Run rollout loop with full trajectory recording.

    Args:
        env_adapter: Seedable environment adapter.
        model_adapter: Model adapter (or any object with ``infer``).
        runtime: Eval runtime config.
        seed: If given, reset env with this seed before starting.
        video_path: If given, write an MP4 video to this path.
            Frames are extracted from ``obs_batch["main_images"]``.
        video_fps: Frames per second for the output video.
        k_b: If given, draw a marker on frames at the bottleneck boundary
            (step >= total_steps - k_b) using a red border overlay.

    Returns:
        Tuple of (RolloutLoopResult, list of EpisodeTrajectory).
    """
    obs_batch, _ = env_adapter.reset(seed=seed)
    expected_batch = infer_batch_size(obs_batch)

    latency = LatencyCollector()
    current_seed = env_adapter.current_seed or seed or 0

    # Per-episode accumulators
    episode_steps: list[StepRecord] = []
    episode_reward = 0.0
    trajectories: list[EpisodeTrajectory] = []
    episode_env_idx = 0

    # Video frame buffer
    frames: list[np.ndarray] = []

    for step_idx in range(runtime.total_steps):
        assert_obs_batch(obs_batch, expected_batch=expected_batch)

        t0 = time.perf_counter()
        with latency.timed("model_infer"):
            actions, meta = model_adapter.infer(obs_batch=obs_batch, mode="eval")
        model_ms = (time.perf_counter() - t0) * 1000.0

        # Broadcast single-action tensors to match env batch size.
        # OpenLoopReplayAdapter returns shape [action_dim]; env expects [B, action_dim].
        if torch.is_tensor(actions) and actions.ndim == 1 and expected_batch > 1:
            actions = actions.unsqueeze(0).expand(expected_batch, -1)

        assert_action_batch(actions, expected_batch=expected_batch)

        t0 = time.perf_counter()
        with latency.timed("env_step"):
            step_result = env_adapter.step(actions)
        env_ms = (time.perf_counter() - t0) * 1000.0

        reward_val = _scalar(step_result.reward)
        terminated = _bool(step_result.terminated)
        truncated = _bool(step_result.truncated)

        # Collect video frame: capture the state AFTER env.step so every
        # action's outcome is recorded. On the first step we prepend the
        # pre-action obs so the video starts from the initial state.
        if video_path is not None:
            in_bottleneck = k_b is not None and step_idx >= runtime.total_steps - k_b
            if step_idx == 0:
                # Initial frame (before any action)
                frame0 = _extract_frame(obs_batch, env_idx=0, in_bottleneck=False)
                if frame0 is not None:
                    frames.append(frame0)
            # Frame after this action executed
            frame = _extract_frame(step_result.obs, env_idx=0, in_bottleneck=in_bottleneck)
            if frame is not None:
                frames.append(frame)

        record = StepRecord(
            step=step_idx,
            obs=obs_batch,
            action=(actions[0].detach().cpu() if torch.is_tensor(actions) and actions.ndim > 1
                    else actions.detach().cpu() if torch.is_tensor(actions) else actions),
            reward=reward_val,
            terminated=terminated,
            truncated=truncated,
            info=step_result.info or {},
            model_latency_ms=model_ms,
            env_latency_ms=env_ms,
            meta=meta if isinstance(meta, dict) else {},
        )
        episode_steps.append(record)
        episode_reward += reward_val

        obs_batch = step_result.obs

        # Episode boundary
        if terminated or truncated:
            success = _extract_success(step_result.info)
            trajectories.append(
                EpisodeTrajectory(
                    seed=current_seed,
                    env_index=episode_env_idx,
                    steps=list(episode_steps),
                    success=success,
                    total_reward=episode_reward,
                )
            )
            episode_steps = []
            episode_reward = 0.0
            episode_env_idx += 1
            # Reset stateful adapters (e.g. OpenLoopReplayAdapter) on episode boundary
            if hasattr(model_adapter, "reset"):
                model_adapter.reset()
            # Re-reset with same seed for next episode
            obs_batch, _ = env_adapter.reset(seed=current_seed)

    # Flush any remaining partial episode
    if episode_steps:
        trajectories.append(
            EpisodeTrajectory(
                seed=current_seed,
                env_index=episode_env_idx,
                steps=list(episode_steps),
                success=False,
                total_reward=episode_reward,
            )
        )

    # Write video
    if video_path is not None and frames:
        _write_video(frames, video_path, fps=video_fps)

    result = RolloutLoopResult(
        total_steps=runtime.total_steps,
        warmup_steps=runtime.warmup_steps,
        measure_steps=max(0, runtime.total_steps - runtime.warmup_steps),
        latency=latency.stats,
        profile_metrics={},
    )
    return result, trajectories


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def _extract_frame(
    obs_batch: dict[str, Any],
    env_idx: int = 0,
    in_bottleneck: bool = False,
) -> np.ndarray | None:
    """Extract a single RGB frame from obs_batch for the given env index."""
    img = None
    for key in ("main_images", "images", "rgb", "main_image"):
        val = obs_batch.get(key)
        if val is None:
            continue
        if torch.is_tensor(val):
            val = val.cpu().numpy()
        if isinstance(val, np.ndarray) and val.ndim >= 3:
            # [B, H, W, C] or [B, C, H, W] or [H, W, C]
            if val.ndim == 4:
                frame = val[min(env_idx, val.shape[0] - 1)]
            else:
                frame = val
            # CHW → HWC
            if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
                frame = frame.transpose(1, 2, 0)
            # Normalise to uint8
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8) if frame.max() <= 1.0 \
                    else frame.astype(np.uint8)
            img = frame[..., :3]  # drop alpha if present
            break

    if img is None:
        return None

    # Draw a thin red border on bottleneck-zone frames
    if in_bottleneck and img.shape[0] > 4 and img.shape[1] > 4:
        img = img.copy()
        img[:3, :] = [255, 0, 0]
        img[-3:, :] = [255, 0, 0]
        img[:, :3] = [255, 0, 0]
        img[:, -3:] = [255, 0, 0]

    return img


def _write_video(frames: list[np.ndarray], path: str, fps: int = 10) -> None:
    """Write frames to an MP4 file using imageio."""
    import imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=7)


# ---------------------------------------------------------------------------
# Scalar/bool helpers
# ---------------------------------------------------------------------------

def _scalar(val: Any) -> float:
    if val is None:
        return 0.0
    if torch.is_tensor(val):
        return float(val.sum().item())
    if hasattr(val, "item"):
        return float(val.item())
    return float(val)


def _bool(val: Any) -> bool:
    if val is None:
        return False
    if torch.is_tensor(val):
        return bool(val.any().item())
    if hasattr(val, "any"):
        return bool(val.any())
    return bool(val)


def _extract_success(info: dict[str, Any] | None) -> bool:
    if info is None:
        return False
    if "success" in info:
        return bool(info["success"])
    if "is_success" in info:
        return bool(info["is_success"])
    return False


