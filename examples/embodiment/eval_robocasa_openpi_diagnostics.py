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

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def to_jsonable(value: Any) -> Any:
    """Convert numpy and torch values into JSON-serializable Python values."""
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def build_episode_record(
    *,
    episode_id: int,
    env_id: int,
    task_name: str,
    seed: int,
    task_description: str,
    actions: list[Any],
    step_records: list[dict[str, Any]],
    success: bool,
    termination_reason: str,
) -> dict[str, Any]:
    """Build a JSON-compatible diagnostics record for one evaluation episode."""
    return to_jsonable(
        {
            "episode_id": episode_id,
            "env_id": env_id,
            "task_name": task_name,
            "seed": seed,
            "success": bool(success),
            "num_steps": len(step_records),
            "termination_reason": termination_reason,
            "task_description": task_description,
            "actions": actions,
            "steps": step_records,
        }
    )


def validate_diagnostics_cfg(cfg: DictConfig) -> None:
    """Validate and fill defaults for RoboCasa OpenPI diagnostics eval."""
    env_type = OmegaConf.select(cfg, "env.eval.env_type")
    if env_type != "robocasa":
        raise ValueError("RoboCasa diagnostics eval requires env.eval.env_type=robocasa")

    model_type = OmegaConf.select(cfg, "actor.model.model_type")
    if model_type != "openpi":
        raise ValueError("Diagnostics eval requires actor.model.model_type=openpi")

    model_path = OmegaConf.select(cfg, "actor.model.model_path")
    if model_path is None or not Path(str(model_path)).exists():
        raise FileNotFoundError(f"OpenPI model_path does not exist: {model_path}")

    total_num_envs = OmegaConf.select(cfg, "env.eval.total_num_envs")
    if total_num_envs != 1:
        raise ValueError("RoboCasa diagnostics eval requires env.eval.total_num_envs=1")

    if OmegaConf.select(cfg, "diagnostics") is None:
        OmegaConf.update(cfg, "diagnostics", {}, merge=False, force_add=True)

    defaults = {
        "output_path": "robocasa_openpi_eval.jsonl",
        "num_episodes": 1,
        "max_contacts": 32,
        "include_model_names": True,
        "flush_every": 1,
    }
    for key, value in defaults.items():
        if OmegaConf.select(cfg, f"diagnostics.{key}") is None:
            OmegaConf.update(
                cfg,
                f"diagnostics.{key}",
                value,
                merge=False,
                force_add=True,
            )


def load_openpi_model(cfg: DictConfig) -> torch.nn.Module:
    """Load the configured OpenPI model for local evaluation."""
    from rlinf.models.embodiment.openpi import get_model

    model = get_model(cfg.actor.model)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return model


def create_robocasa_eval_env(cfg: DictConfig):
    """Create a single-process local RoboCasa eval environment."""
    from rlinf.envs.robocasa.robocasa_env import RobocasaEnv

    return RobocasaEnv(
        cfg=cfg.env.eval,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )


def _tensor_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return False
        return bool(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
        return bool(value.reshape(-1)[0].item())
    return bool(value)


def _tensor_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0
        return float(value.reshape(-1)[0].item())
    return float(value)


def _first_task_name(cfg: DictConfig) -> str:
    task_names = OmegaConf.select(cfg, "env.eval.task_names")
    if task_names is None:
        return ""
    task_names = OmegaConf.to_container(task_names, resolve=True)
    if isinstance(task_names, list):
        return str(task_names[0]) if task_names else ""
    return str(task_names)


def _first_task_description(obs: dict[str, Any]) -> str:
    descriptions = obs.get("task_descriptions", "")
    if isinstance(descriptions, (list, tuple)):
        return str(descriptions[0]) if descriptions else ""
    return str(descriptions)


def _iter_single_env_actions(action_chunk: Any):
    if isinstance(action_chunk, tuple):
        action_chunk = action_chunk[0]
    if isinstance(action_chunk, torch.Tensor):
        action_chunk = action_chunk.detach().cpu().numpy()
    action_chunk = np.asarray(action_chunk)

    if action_chunk.ndim == 3:
        for step_idx in range(action_chunk.shape[1]):
            yield action_chunk[:, step_idx]
    elif action_chunk.ndim == 2:
        if action_chunk.shape[0] == 1:
            yield action_chunk
        else:
            for step_idx in range(action_chunk.shape[0]):
                yield action_chunk[step_idx : step_idx + 1]
    else:
        yield action_chunk.reshape(1, -1)


def run_diagnostics_eval(cfg: DictConfig) -> None:
    """Run local RoboCasa OpenPI diagnostics evaluation and write JSONL records."""
    validate_diagnostics_cfg(cfg)
    output_path = Path(str(cfg.diagnostics.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = create_robocasa_eval_env(cfg)
    model = load_openpi_model(cfg)
    try:
        with output_path.open("w", encoding="utf-8") as output_file:
            for episode_id in range(int(cfg.diagnostics.num_episodes)):
                obs, _ = env.reset()
                actions: list[Any] = []
                step_records: list[dict[str, Any]] = []
                success = False
                termination_reason = "max_episode_steps"
                max_episode_steps = int(cfg.env.eval.max_episode_steps)
                task_description = _first_task_description(obs)

                while len(step_records) < max_episode_steps:
                    with torch.no_grad():
                        action_chunk = model.predict_action_batch(
                            env_obs=obs,
                            mode="eval",
                        )

                    should_stop = False
                    for action in _iter_single_env_actions(action_chunk):
                        if len(step_records) >= max_episode_steps:
                            should_stop = True
                            break

                        obs, reward, terminated, truncated, _ = env.step(
                            action,
                            auto_reset=False,
                        )
                        diagnostics = env.get_mujoco_diagnostics(
                            max_contacts=int(cfg.diagnostics.max_contacts),
                            include_model_names=bool(
                                cfg.diagnostics.include_model_names
                            ),
                        )[0]

                        is_success = _tensor_bool(terminated)
                        is_truncated = _tensor_bool(truncated)
                        actions.append(action)
                        step_records.append(
                            {
                                "step": len(step_records),
                                "reward": _tensor_float(reward),
                                "success": is_success,
                                "terminated": is_success,
                                "truncated": is_truncated,
                                "diagnostics": diagnostics,
                            }
                        )

                        if is_success:
                            success = True
                            termination_reason = "success"
                            should_stop = True
                            break
                        if is_truncated:
                            termination_reason = "truncated"
                            should_stop = True
                            break

                    if should_stop:
                        break

                record = build_episode_record(
                    episode_id=episode_id,
                    env_id=0,
                    task_name=_first_task_name(cfg),
                    seed=int(cfg.env.eval.seed),
                    task_description=task_description,
                    actions=actions,
                    step_records=step_records,
                    success=success,
                    termination_reason=termination_reason,
                )
                output_file.write(json.dumps(record) + "\n")
                if (episode_id + 1) % int(cfg.diagnostics.flush_every) == 0:
                    output_file.flush()
    finally:
        env.close()


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="robocasa_closedrawer_ppo_openpi",
)
def main(cfg: DictConfig) -> None:
    from rlinf.config import validate_cfg

    cfg.runner.only_eval = True
    cfg = validate_cfg(cfg)
    run_diagnostics_eval(cfg)


if __name__ == "__main__":
    main()
