"""Environment adapters for lightweight rollout evaluation."""

from __future__ import annotations

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



def build_env_adapter(cfg: DictConfig, split: str = "eval") -> EnvAdapterProtocol:
    """Build environment adapter from RLinf config.

    Args:
        cfg: RLinf Hydra config.
        split: One of ``train`` or ``eval``.

    Returns:
        Normalized env adapter.

    Raises:
        RuntimeError: If environment construction fails with contextual hints.
    """
    env_cfg = cfg.env[split]
    env_cls = get_env_cls(env_cfg.env_type, env_cfg)
    num_envs = int(env_cfg.total_num_envs)

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

    return GenericEnvAdapter(env=env)
