"""Hydra config bridge for lightweight rollout evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig


@dataclass
class EvalRuntimeConfig:
    """Normalized runtime config for non-Ray rollout eval."""

    env_type: str
    model_type: str
    model_path: str
    num_envs: int
    group_size: int
    num_action_chunks: int
    total_steps: int
    warmup_steps: int
    seed: int
    enable_torch_profiler: bool = True
    profiler_output_dir: str = "./rollout_eval_output/profiler"
    split_model_stages: bool = True


def _require(cfg: DictConfig, path: str):
    cur = cfg
    for part in path.split("."):
        if part not in cur:
            raise KeyError(f"Missing required config field: {path}")
        cur = cur[part]
    return cur


def build_eval_runtime_config(cfg: DictConfig) -> EvalRuntimeConfig:
    """Build normalized runtime config from existing Hydra config.

    Args:
        cfg: Full RLinf Hydra config.

    Returns:
        A minimal runtime config used by the lightweight eval loop.
    """
    env_eval = _require(cfg, "env.eval")
    actor_model = _require(cfg, "actor.model")

    rollout_model_path = (
        cfg.rollout.model.model_path
        if "rollout" in cfg
        and "model" in cfg.rollout
        and "model_path" in cfg.rollout.model
        else actor_model.model_path
    )

    total_steps = int(env_eval.get("max_steps_per_rollout_epoch", 0))
    if total_steps <= 0:
        total_steps = int(actor_model.get("num_action_chunks", 1))

    warmup_steps = max(1, min(10, total_steps // 5 if total_steps > 1 else 1))

    return EvalRuntimeConfig(
        env_type=str(env_eval.env_type),
        model_type=str(actor_model.model_type),
        model_path=str(rollout_model_path),
        num_envs=int(env_eval.total_num_envs),
        group_size=int(env_eval.get("group_size", 1)),
        num_action_chunks=int(actor_model.get("num_action_chunks", 1)),
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        seed=int(cfg.actor.get("seed", 0)),
        enable_torch_profiler=bool(
            cfg.get("rollout_eval", {}).get("enable_torch_profiler", True)
        ),
        profiler_output_dir=str(
            cfg.get("rollout_eval", {}).get(
                "profiler_output_dir", "./rollout_eval_output/profiler"
            )
        ),
        split_model_stages=bool(
            cfg.get("rollout_eval", {}).get("split_model_stages", True)
        ),
    )
