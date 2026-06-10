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

"""Subprocess vectorized environment for Robocasa.

Based on metaworld/venv.py implementation, adapted for Robocasa/Robosuite environments.
"""

import json
import os
import time
from multiprocessing import Pipe, connection
from multiprocessing.context import Process
from typing import Any, Callable, Optional, Union

import gymnasium as gym
import numpy as np

from rlinf.envs.venv import (
    BaseVectorEnv,
    CloudpickleWrapper,
    EnvWorker,
    ShArray,
    SubprocEnvWorker,
    SubprocVectorEnv,
    _setup_buf,
)
from rlinf.envs.venv.venv import _apply_subproc_env_cpu_affinity, _to_jsonable
from rlinf.scheduler.resource_pool.cpu_binding import (
    apply_process_cpu_affinity,
    get_env_core_group_from_env,
)


def _json_list(value: Any) -> list:
    return np.asarray(value).tolist()


def _named_items(model: Any, count_attr: str, accessor_name: str) -> list[str]:
    count = int(getattr(model, count_attr, 0))
    accessor = getattr(model, accessor_name)
    return [accessor(idx).name for idx in range(count)]


def _default_contact_force(model: Any, data: Any, contact_id: int) -> list[float]:
    import mujoco

    force = np.zeros(6, dtype=np.float64)
    mujoco.mj_contactForce(model, data, contact_id, force)
    return _json_list(force)


def build_mujoco_diagnostics_snapshot(
    model: Any,
    data: Any,
    max_contacts: Optional[int] = None,
    include_model_names: bool = True,
    contact_force_fn: Optional[Callable[[Any, Any, int], list[float]]] = None,
) -> dict[str, Any]:
    """Build a JSON-serializable MuJoCo diagnostics snapshot."""
    ncon = int(getattr(data, "ncon", 0))
    contact_limit = ncon if max_contacts is None else min(ncon, max_contacts)
    force_fn = contact_force_fn or _default_contact_force

    contacts = []
    for contact_id in range(contact_limit):
        contact = data.contact[contact_id]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        contact_snapshot = {
            "dist": float(contact.dist),
            "geom1": geom1,
            "geom2": geom2,
            "geom1_name": model.geom(geom1).name if include_model_names else "",
            "geom2_name": model.geom(geom2).name if include_model_names else "",
            "force": None,
        }
        try:
            contact_snapshot["force"] = _json_list(force_fn(model, data, contact_id))
        except Exception as exc:
            contact_snapshot["force_error"] = str(exc)
        contacts.append(contact_snapshot)

    energy = _json_list(getattr(data, "energy", []))
    return {
        "ncon": ncon,
        "contacts": contacts,
        "qvel": _json_list(getattr(data, "qvel", [])),
        "xpos": _json_list(getattr(data, "xpos", [])),
        "xquat": _json_list(getattr(data, "xquat", [])),
        "subtree_linvel": _json_list(getattr(data, "subtree_linvel", [])),
        "energy": energy,
        "kinetic_energy": energy[0] if len(energy) > 0 else None,
        "potential_energy": energy[1] if len(energy) > 1 else None,
        "body_names": _named_items(model, "nbody", "body")
        if include_model_names
        else [],
        "geom_names": _named_items(model, "ngeom", "geom")
        if include_model_names
        else [],
    }


def _worker(
    parent: connection.Connection,
    p: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    obs_bufs: Optional[Union[dict, tuple, ShArray]] = None,
    local_env_index: int = -1,
) -> None:
    """Worker function for robocasa subprocess environment.

    Based on metaworld's _worker function, adapted for robosuite environments.
    """

    def _encode_obs(
        obs: Union[dict, tuple, np.ndarray], buffer: Union[dict, tuple, ShArray]
    ) -> None:
        if isinstance(obs, np.ndarray) and isinstance(buffer, ShArray):
            buffer.save(obs)
        elif isinstance(obs, tuple) and isinstance(buffer, tuple):
            for o, b in zip(obs, buffer):
                _encode_obs(o, b)
        elif isinstance(obs, dict) and isinstance(buffer, dict):
            for k in obs.keys():
                _encode_obs(obs[k], buffer[k])
        return None

    def _check_success(env, env_return):
        success = env._check_success()
        env_return = list(env_return)
        info = env_return[-1]
        info["success"] = success
        env_return[-1] = info
        env_return = tuple(env_return)
        return env_return

    def get_ep_meta(env, env_return):
        ep_meta = env.get_ep_meta()
        env_return = list(env_return)
        info = env_return[-1]
        info["ep_meta"] = ep_meta
        env_return[-1] = info
        env_return = tuple(env_return)
        return env_return

    def _step_with_timing(env, action, *, chunk_action_index=None, repeat_index=None):
        wall_start_ns = time.time_ns()
        perf_start = time.perf_counter()
        env_return = env.step(action)
        duration_s = max(time.perf_counter() - perf_start, 0.0)
        wall_end_ns = time.time_ns()
        env_return = list(env_return)
        info = env_return[-1]
        timing = {
            "local_env": int(local_env_index),
            "duration_s": duration_s,
            "wall_start_ns": wall_start_ns,
            "wall_end_ns": wall_end_ns,
        }
        if chunk_action_index is not None:
            timing["chunk_action_index"] = int(chunk_action_index)
        if repeat_index is not None:
            timing["repeat_index"] = int(repeat_index)
        info.setdefault("robocasa_step_timings", []).append(timing)
        env_return[-1] = info
        return tuple(env_return)

    parent.close()
    if local_env_index >= 0:
        _apply_subproc_env_cpu_affinity(local_env_index)
    env = env_fn_wrapper.data()
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:  # the pipe has been closed
                p.close()
                break
            if cmd == "step":
                # Robosuite returns (obs, reward, done, info), not 5 values like gymnasium
                env_return = _step_with_timing(env, data)
                if obs_bufs is not None:
                    _encode_obs(env_return[0], obs_bufs)
                    env_return = (None, *env_return[1:])
                # RoboCasa step can't record success in info, _check_success() must be called
                if hasattr(env, "_check_success"):
                    env_return = _check_success(env, env_return)
                # call get_ep_meta() to get the RoboCasa env meta, includes prompt & layout_id, etcs
                if hasattr(env, "get_ep_meta"):
                    env_return = get_ep_meta(env, env_return)
                p.send(env_return)
            elif cmd == "chunk_step":
                if obs_bufs is not None:
                    raise NotImplementedError(
                        "chunk_step does not support shared-memory observations"
                    )
                action_repeat = 1
                if isinstance(data, tuple):
                    data, action_repeat = data
                action_repeat = int(action_repeat)
                if action_repeat < 1:
                    raise ValueError(
                        "action_repeat_per_chunk_step must be >= 1, "
                        f"got {action_repeat}"
                    )
                env_returns = []
                for chunk_action_index, action in enumerate(data):
                    step_timings = []
                    for repeat_index in range(action_repeat):
                        env_return = _step_with_timing(
                            env,
                            action,
                            chunk_action_index=chunk_action_index,
                            repeat_index=repeat_index,
                        )
                        step_timings.extend(
                            env_return[-1].get("robocasa_step_timings", [])
                        )
                        if hasattr(env, "_check_success"):
                            env_return = _check_success(env, env_return)
                        if hasattr(env, "get_ep_meta"):
                            env_return = get_ep_meta(env, env_return)
                    env_return = list(env_return)
                    info = env_return[-1]
                    info["robocasa_step_timings"] = step_timings
                    env_return[-1] = info
                    env_return = tuple(env_return)
                    env_returns.append(env_return)
                p.send(tuple(zip(*env_returns)))
            elif cmd == "set_cpu_affinity":
                apply_process_cpu_affinity(tuple(data))
                p.send(tuple(sorted(os.sched_getaffinity(0))))
            elif cmd == "get_cpu_affinity":
                p.send(tuple(sorted(os.sched_getaffinity(0))))
            elif cmd == "reset":
                # Robosuite reset can return just obs or (obs, info)
                retval = env.reset(**data)
                reset_returns_info = (
                    isinstance(retval, (tuple, list))
                    and len(retval) == 2
                    and isinstance(retval[1], dict)
                )
                if reset_returns_info:
                    obs, info = retval
                else:
                    obs = retval
                    info = {}
                if obs_bufs is not None:
                    _encode_obs(obs, obs_bufs)
                    obs = None
                # call get_ep_meta() to get the RoboCasa env meta, includes prompt & layout_id, etcs
                if hasattr(env, "get_ep_meta"):
                    info = get_ep_meta(env, (info,))[-1]
                # return obs + info other than mere obs
                p.send((obs, info))
            elif cmd == "close":
                p.send(env.close())
                p.close()
                break
            elif cmd == "render":
                p.send(env.render(**data) if hasattr(env, "render") else None)
            elif cmd == "seed":
                if hasattr(env, "seed"):
                    p.send(env.seed(data))
                else:
                    env.reset(seed=data)
                    p.send(None)
            elif cmd == "getattr":
                p.send(getattr(env, data) if hasattr(env, data) else None)
            elif cmd == "setattr":
                setattr(env.unwrapped, data["key"], data["value"])
            elif cmd == "get_mujoco_diagnostics":
                p.send(
                    build_mujoco_diagnostics_snapshot(
                        model=env.sim.model,
                        data=env.sim.data,
                        max_contacts=data.get("max_contacts"),
                        include_model_names=data.get("include_model_names", True),
                    )
                )
            else:
                p.close()
                raise NotImplementedError(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        p.close()


class RobocasaSubprocEnvWorker(SubprocEnvWorker):
    """Subprocess environment worker for Robocasa.

    Based on metaworld's ReconfigureSubprocEnvWorker, but without the reconfigure
    functionality since robocasa doesn't need it.
    """

    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        share_memory: bool = False,
        local_env_index: int = -1,
    ):
        self.parent_remote, self.child_remote = Pipe()
        self.share_memory = share_memory
        self.buffer: Optional[Union[dict, tuple, ShArray]] = None
        self._cpu_affinity = get_env_core_group_from_env(os.environ, local_env_index)
        if self.share_memory:
            dummy = env_fn()
            obs_space = dummy.observation_space
            dummy.close()
            del dummy
            self.buffer = _setup_buf(obs_space)
        args = (
            self.parent_remote,
            self.child_remote,
            CloudpickleWrapper(env_fn),
            self.buffer,
            local_env_index,
        )
        # Use our custom _worker function
        self.process = Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()
        EnvWorker.__init__(self, env_fn)

    def get_mujoco_diagnostics(
        self,
        max_contacts: Optional[int] = None,
        include_model_names: bool = True,
    ) -> dict[str, Any]:
        self.parent_remote.send(
            [
                "get_mujoco_diagnostics",
                {
                    "max_contacts": max_contacts,
                    "include_model_names": include_model_names,
                },
            ]
        )
        return self.parent_remote.recv()


class RobocasaSubprocEnv(SubprocVectorEnv):
    """Subprocess vectorized environment for Robocasa/Robosuite.

    Based on metaworld's ReconfigureSubprocEnv, adapted for robocasa environments.
    Uses subprocess isolation to avoid OpenGL context sharing issues in MuJoCo.
    """

    def __init__(self, env_fns: list[Callable[[], gym.Env]], **kwargs: Any) -> None:
        env_index = {"value": 0}

        def worker_fn(fn: Callable[[], gym.Env]) -> RobocasaSubprocEnvWorker:
            # Use our custom worker with shared memory disabled
            # Robosuite dict observations work better without shared memory
            local_env_index = env_index["value"]
            env_index["value"] += 1
            return RobocasaSubprocEnvWorker(
                fn, share_memory=False, local_env_index=local_env_index
            )

        BaseVectorEnv.__init__(self, env_fns, worker_fn, **kwargs)

    def get_mujoco_diagnostics(
        self,
        max_contacts: Optional[int] = None,
        include_model_names: bool = True,
    ) -> list[dict[str, Any]]:
        if not hasattr(self, "is_closed"):
            self.is_closed = False
        self._assert_is_not_closed()
        return [
            worker.get_mujoco_diagnostics(max_contacts, include_model_names)
            for worker in self.workers
        ]

    def record_robocasa_step_timing_events(
        self,
        info_lists: list[dict[str, Any]],
        *,
        vector_step: int,
    ) -> None:
        """Write one timestamp row per real RoboCasa env.step call."""
        context = getattr(self, "_sim_timestamp_context", None)
        if context is None:
            return
        handle = self._get_sim_timestamp_file()
        if handle is None:
            return
        for info in info_lists:
            for timing in info.get("robocasa_step_timings", []):
                local_env = int(timing.get("local_env", -1))
                record = {
                    "event": "robocasa_env_step",
                    "rank": int(context["rank"]),
                    "pid": int(context["pid"]),
                    "epoch": context.get("epoch"),
                    "chunk_step": context.get("chunk_step"),
                    "stage": context.get("stage"),
                    "local_env": local_env,
                    "global_env": self._global_env_id(local_env),
                    "operation": "robocasa_step",
                    "vector_step": int(vector_step),
                    "chunk_action_index": timing.get("chunk_action_index"),
                    "repeat_index": timing.get("repeat_index"),
                    "duration_s": timing.get("duration_s"),
                    "wall_start_ns": timing.get("wall_start_ns"),
                    "wall_end_ns": timing.get("wall_end_ns"),
                }
                handle.write(json.dumps(_to_jsonable(record), sort_keys=True) + "\n")
