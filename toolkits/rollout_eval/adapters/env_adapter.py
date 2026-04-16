"""Environment adapters for lightweight rollout evaluation."""

from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from omegaconf import DictConfig

from rlinf.envs import get_env_cls
from toolkits.rollout_eval.rollout_types import EnvStepResult


class EnvAdapterProtocol(Protocol):
    """Protocol for lightweight environment interaction."""

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def step(self, actions: torch.Tensor) -> EnvStepResult: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Profiling-aware behavior env worker (monkey-patch target)
# ---------------------------------------------------------------------------

def _profiled_behavior_env_worker(conn, cfg_dict, num_envs, task_idx):
    """Drop-in replacement for _behavior_env_worker with per-step GPU profiling.

    Uses OmniGibson's pynvml_utils to collect SM utilization, VRAM, and
    splits each step into pre_step / sim_step (IsaacSim physics+render) / post_step.
    Profiling data is attached to every IPC response under the "profile" key.
    """
    # Prevent nsys/cupti injection — IsaacSim manages its own CUDA context.
    os.environ["CUDA_INJECTION64_PATH"] = ""
    os.environ["CUDA_TOOL_EXIT_AFTER_DETACH"] = "1"

    env = None
    _profile_output_dir = cfg_dict.pop("__profile_output_dir__", None)

    try:
        import copy
        import time as _time

        import omnigibson as og
        from omnigibson.learning.utils.eval_utils import TASK_INDICES_TO_NAMES
        from omnigibson.macros import gm
        import omnigibson.utils.pynvml_utils as pynvml

        gm.HEADLESS = True
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_TRANSITION_RULES = True

        cfg_dict["task"]["activity_name"] = TASK_INDICES_TO_NAMES[task_idx]

        class _ProfilingVecEnv:
            """VectorEnvironment with per-step timing and pynvml GPU stats."""

            def __init__(self, n_envs, config):
                self.num_envs = n_envs
                if og.sim is not None:
                    og.sim.stop()
                from tqdm import trange
                self.envs = [
                    og.Environment(configs=copy.deepcopy(config), in_vec_env=True)
                    for _ in trange(n_envs, desc="Loading environments")
                ]
                og.sim.play()
                for e in self.envs:
                    e.post_play_load()
                try:
                    pynvml.nvmlInit()
                    self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    self._pynvml_ok = True
                except Exception:
                    self._pynvml_ok = False
                self.last_profile = {}

            def _gpu_stats(self):
                if not self._pynvml_ok:
                    return {}
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                    return {
                        "gpu_sm_util_pct": util.gpu,
                        "gpu_mem_util_pct": util.memory,
                        "gpu_vram_used_gb": round(mem.used / 1024 ** 3, 3),
                        "gpu_vram_total_gb": round(mem.total / 1024 ** 3, 3),
                    }
                except Exception:
                    return {}

            def step(self, actions):
                t0 = _time.perf_counter()
                for i, action in enumerate(actions):
                    self.envs[i]._pre_step(action)
                t_pre = _time.perf_counter()
                og.sim.step()
                t_sim = _time.perf_counter()
                observations, rewards, terminates, truncates, infos = [], [], [], [], []
                for i, action in enumerate(actions):
                    obs, reward, terminated, truncated, info = self.envs[i]._post_step(action)
                    observations.append(obs)
                    rewards.append(reward)
                    terminates.append(terminated)
                    truncates.append(truncated)
                    infos.append(info)
                t_post = _time.perf_counter()
                self.last_profile = {
                    "pre_step_ms": round((t_pre - t0) * 1e3, 2),
                    "sim_step_ms": round((t_sim - t_pre) * 1e3, 2),
                    "post_step_ms": round((t_post - t_sim) * 1e3, 2),
                    "total_step_ms": round((t_post - t0) * 1e3, 2),
                    **self._gpu_stats(),
                }
                return observations, rewards, terminates, truncates, infos

            def reset(self, get_obs=True, **kwargs):
                t0 = _time.perf_counter()
                if get_obs:
                    observations, infos = [], []
                    for e in self.envs:
                        obs, info = e.reset(get_obs=get_obs, **kwargs)
                        observations.append(obs)
                        infos.append(info)
                    self.last_profile = {
                        "reset_ms": round((_time.perf_counter() - t0) * 1e3, 2),
                        **self._gpu_stats(),
                    }
                    return observations, infos
                else:
                    for e in self.envs:
                        e.reset(get_obs=get_obs, **kwargs)

            def close(self):
                if self._pynvml_ok:
                    try:
                        pynvml.nvmlShutdown()
                    except Exception:
                        pass

        env = _ProfilingVecEnv(num_envs, cfg_dict)
        conn.send({"type": "ready", "activity_name": cfg_dict["task"]["activity_name"]})

        from rlinf.envs.utils import to_tensor

        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                raw_obs, infos = env.reset()
                conn.send({"type": "ok", "result": (raw_obs, infos),
                           "profile": env.last_profile})
            elif cmd == "step":
                result = env.step(payload)
                conn.send({"type": "ok", "result": result,
                           "profile": env.last_profile})
            elif cmd == "chunk_step":
                chunk_actions = payload["chunk_actions"]
                chunk_size = chunk_actions.shape[1]
                raw_obs_list, chunk_rewards = [], []
                raw_chunk_terminations, raw_chunk_truncations, infos_list = [], [], []
                profiles = []
                for i in range(chunk_size):
                    raw_obs, step_rewards, terminations, truncations, infos = env.step(
                        chunk_actions[:, i]
                    )
                    raw_obs_list.append(raw_obs)
                    chunk_rewards.append(to_tensor(step_rewards))
                    raw_chunk_terminations.append(to_tensor(terminations))
                    raw_chunk_truncations.append(to_tensor(truncations))
                    infos_list.append(infos)
                    profiles.append(dict(env.last_profile))
                conn.send({
                    "type": "ok",
                    "result": (raw_obs_list, chunk_rewards, raw_chunk_terminations,
                               raw_chunk_truncations, infos_list),
                    "profile": profiles,
                })
            elif cmd == "close":
                try:
                    env.close()
                finally:
                    conn.send({"type": "ok", "result": None})
                    break
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except Exception:
        conn.send({"type": "error", "traceback": traceback.format_exc()})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        conn.close()


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _patch_behavior_env_for_profiling(profile_output_dir: str) -> None:
    """Monkey-patch BehaviorEnv to use profiled subprocess worker and collect profile data."""
    import rlinf.envs.behavior.behavior_env as _mod

    def _patched_init_env(self):
        from multiprocessing import get_context
        from omegaconf import OmegaConf

        self._ctx = get_context("spawn")
        self._parent_conn, child_conn = self._ctx.Pipe()
        self._env_profile_log = []  # accumulate per-step profile dicts

        cfg_dict = OmegaConf.to_container(self.cfg.omnigibson_cfg, resolve=True)
        cfg_dict["__profile_output_dir__"] = profile_output_dir

        self._env_process = self._ctx.Process(
            target=_profiled_behavior_env_worker,
            args=(child_conn, cfg_dict, self.num_envs, self.cfg.task_idx),
            daemon=True,
        )
        self._env_process.start()
        child_conn.close()

        msg = self._parent_conn.recv()
        if msg.get("type") != "ready":
            raise RuntimeError(
                f"Failed to initialize behavior subprocess env: {msg.get('traceback', msg)}"
            )
        self._load_tasks_cfg(msg["activity_name"])

    def _patched_call_subproc(self, cmd: str, payload=None):
        self._parent_conn.send((cmd, payload))
        msg = self._parent_conn.recv()
        if msg.get("type") == "error":
            raise RuntimeError(
                f"Behavior subprocess env failed on command '{cmd}':\n{msg['traceback']}"
            )
        # Collect profile data from subprocess
        profile = msg.get("profile")
        if profile is not None and hasattr(self, "_env_profile_log"):
            if isinstance(profile, list):
                self._env_profile_log.extend(profile)
            else:
                self._env_profile_log.append(profile)
        return msg["result"]

    _mod.BehaviorEnv._init_env = _patched_init_env
    _mod.BehaviorEnv._call_subproc = _patched_call_subproc


@dataclass
class GenericEnvAdapter:
    """Wrapper that normalizes env APIs into EnvStepResult."""

    env: Any

    def _infer_batch_size(self, obs: dict[str, Any]) -> int:
        for key in ("main_images", "states", "extra_view_images"):
            value = obs.get(key)
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return int(value.shape[0])
        return 1

    def _read_env_attr(self, name: str) -> Any:
        if hasattr(self.env, "get_wrapper_attr"):
            try:
                value = self.env.get_wrapper_attr(name)
                if value is not None:
                    return value
            except Exception:
                pass

        cur = self.env
        for _ in range(8):
            if hasattr(cur, name):
                try:
                    value = getattr(cur, name)
                    if value is not None:
                        return value
                except Exception:
                    pass
            if hasattr(cur, "unwrapped") and cur.unwrapped is not cur:
                cur = cur.unwrapped
                continue
            if hasattr(cur, "env") and cur.env is not cur:
                cur = cur.env
                continue
            break
        return None

    def _resolve_task_descriptions(self, obs: dict[str, Any]) -> list[str]:
        candidates = []
        for name in ("task_descriptions", "instruction"):
            candidates.append(self._read_env_attr(name))
        func = self._read_env_attr("get_language_instruction")
        if callable(func):
            try:
                candidates.append(func())
            except Exception:
                pass

        batch_size = self._infer_batch_size(obs)
        for value in candidates:
            if value is None:
                continue
            if isinstance(value, str):
                return [value] * batch_size
            if isinstance(value, (list, tuple)):
                items = [str(v) for v in value]
                if len(items) == batch_size:
                    return items
                if len(items) == 1:
                    return items * batch_size
                return (items + [items[-1]])[:batch_size]

        return ["complete the task"] * batch_size

    def _normalize_obs(self, obs: Any) -> Any:
        if isinstance(obs, dict) and "task_descriptions" not in obs:
            obs = dict(obs)
            obs["task_descriptions"] = self._resolve_task_descriptions(obs)
        return obs

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = self.env.reset()
        obs = self._normalize_obs(obs)
        return obs, info

    def step(self, actions: torch.Tensor) -> EnvStepResult:
        if hasattr(self.env, "chunk_step") and actions.ndim >= 3:
            obs_list, rewards, terminations, truncations, infos_list = self.env.chunk_step(
                actions
            )
            obs = obs_list[-1] if isinstance(obs_list, (list, tuple)) else obs_list
            info = infos_list[-1] if isinstance(infos_list, (list, tuple)) else infos_list
            obs = self._normalize_obs(obs)
            return EnvStepResult(
                obs=obs,
                reward=rewards,
                terminated=terminations,
                truncated=truncations,
                info=info,
            )

        obs, reward, terminated, truncated, info = self.env.step(actions)
        obs = self._normalize_obs(obs)
        return EnvStepResult(
            obs=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()
        # Save behavior env profile log if available
        profile_log = getattr(self.env, "_env_profile_log", None)
        if profile_log:
            import json
            from pathlib import Path
            out_dir = Path(getattr(self.env, "_profile_output_dir", "/tmp"))
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "env_subprocess_profile.json"
            with open(out_path, "w") as f:
                json.dump(profile_log, f, indent=2)
            # Print summary
            sim_times = [p["sim_step_ms"] for p in profile_log if "sim_step_ms" in p]
            total_times = [p["total_step_ms"] for p in profile_log if "total_step_ms" in p]
            vrams = [p["gpu_vram_used_gb"] for p in profile_log if "gpu_vram_used_gb" in p]
            sm_utils = [p["gpu_sm_util_pct"] for p in profile_log if "gpu_sm_util_pct" in p]
            if sim_times:
                import statistics
                print(
                    f"[env-profiler] {len(sim_times)} steps | "
                    f"sim_step_ms mean={statistics.mean(sim_times):.1f} "
                    f"p50={statistics.median(sim_times):.1f} | "
                    f"total_step_ms mean={statistics.mean(total_times):.1f} | "
                    f"SM util mean={statistics.mean(sm_utils):.1f}% | "
                    f"VRAM mean={statistics.mean(vrams):.2f}GB"
                )
            print(f"[env-profiler] Full log saved to {out_path}")



def build_env_adapter(
    cfg: DictConfig, split: str = "eval", profile_output_dir: str | None = None,
) -> EnvAdapterProtocol:
    """Build environment adapter from RLinf config.

    Args:
        cfg: RLinf Hydra config.
        split: One of ``train`` or ``eval``.
        profile_output_dir: If set and env is behavior, patch subprocess worker
            to run torch profiler and save traces to this directory.

    Returns:
        Normalized env adapter.

    Raises:
        RuntimeError: If environment construction fails with contextual hints.
    """
    env_cfg = cfg.env[split]
    env_cls = get_env_cls(env_cfg.env_type, env_cfg)
    num_envs = int(env_cfg.total_num_envs)

    # Apply subprocess profiling patch for behavior envs
    if str(env_cfg.env_type) == "behavior" and profile_output_dir is not None:
        _patch_behavior_env_for_profiling(profile_output_dir)

    try:
        env = env_cls(
            cfg=env_cfg,
            num_envs=num_envs,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Failed to build env '{env_cfg.env_type}' due to missing local assets: {exc}. "
            "Please check env asset paths/seeds in Hydra config."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to build env '{env_cfg.env_type}': {type(exc).__name__}: {exc}"
        ) from exc

    # Store profile_output_dir on env for use in GenericEnvAdapter.close()
    if profile_output_dir is not None:
        env._profile_output_dir = profile_output_dir

    return GenericEnvAdapter(env=env)
