# Copyright 2025 The LIBERO project and The RLinf Authors.
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

import ctypes
import json
import os
import time
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from multiprocessing import Array, Pipe, connection
from multiprocessing.context import Process
from typing import Any, Callable, Optional, Union

import cloudpickle
import gym
import numpy as np

from rlinf.envs.chunk_runner import stack_vector_chunk_returns
from rlinf.scheduler.resource_pool.cpu_binding import (
    apply_process_cpu_affinity,
    get_env_core_group_from_env,
    parse_env_cpu_core_groups,
)
from rlinf.utils.logging import get_logger

logger = get_logger()

gym_old_venv_step_type = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
gym_new_venv_step_type = tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]
warnings.simplefilter("once", DeprecationWarning)
_NP_TO_CT = {
    np.bool_: ctypes.c_bool,
    np.uint8: ctypes.c_uint8,
    np.uint16: ctypes.c_uint16,
    np.uint32: ctypes.c_uint32,
    np.uint64: ctypes.c_uint64,
    np.int8: ctypes.c_int8,
    np.int16: ctypes.c_int16,
    np.int32: ctypes.c_int32,
    np.int64: ctypes.c_int64,
    np.float32: ctypes.c_float,
    np.float64: ctypes.c_double,
}


def _to_jsonable(value: Any) -> Any:
    """Convert nested numpy-heavy values into JSON-serializable data."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, set):
        return [_to_jsonable(item) for item in sorted(value, key=repr)]
    return value


def deprecation(msg: str) -> None:
    """Deprecation warning wrapper."""
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)


class CloudpickleWrapper(object):
    """A cloudpickle wrapper used in SubprocVectorEnv."""

    def __init__(self, data: Any) -> None:
        self.data = data

    def __getstate__(self) -> str:
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data: str) -> None:
        self.data = cloudpickle.loads(data)


GYM_RESERVED_KEYS = [
    "metadata",
    "reward_range",
    "spec",
    "action_space",
    "observation_space",
]


################################################################################
#
# Workers
#
################################################################################


class EnvWorker(ABC):
    """An abstract worker for an environment."""

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        self._env_fn = env_fn
        self.is_closed = False
        self.result: Union[
            gym_old_venv_step_type,
            gym_new_venv_step_type,
            tuple[np.ndarray, dict],
            np.ndarray,
        ]
        # self.action_space = self.get_env_attr("action_space")  # noqa: B009
        self.is_reset = False

    @abstractmethod
    def get_env_attr(self, key: str) -> Any:
        pass

    @abstractmethod
    def set_env_attr(self, key: str, value: Any) -> None:
        pass

    def set_cpu_affinity(self, cpus: tuple[int, ...]) -> None:
        raise NotImplementedError

    def get_cpu_affinity(self) -> tuple[int, ...]:
        raise NotImplementedError

    def send(self, action: Optional[np.ndarray]) -> None:
        """Send action signal to low-level worker.

        When action is None, it indicates sending "reset" signal; otherwise
        it indicates "step" signal. The paired return value from "recv"
        function is determined by such kind of different signal.
        """
        if hasattr(self, "send_action"):
            deprecation(
                "send_action will soon be deprecated. "
                "Please use send and recv for your own EnvWorker."
            )
            if action is None:
                self.is_reset = True
                self.result = self.reset()
            else:
                self.is_reset = False
                self.send_action(action)

    def recv(
        self,
    ) -> Union[
        gym_old_venv_step_type,
        gym_new_venv_step_type,
        tuple[np.ndarray, dict],
        np.ndarray,
    ]:  # noqa:E125
        """Receive result from low-level worker.

        If the last "send" function sends a NULL action, it only returns a
        single observation; otherwise it returns a tuple of (obs, rew, done,
        info) or (obs, rew, terminated, truncated, info), based on whether
        the environment is using the old step API or the new one.
        """
        if hasattr(self, "get_result"):
            deprecation(
                "get_result will soon be deprecated. "
                "Please use send and recv for your own EnvWorker."
            )
            if not self.is_reset:
                self.result = self.get_result()
        return self.result

    @abstractmethod
    def reset(self, **kwargs: Any) -> Union[np.ndarray, tuple[np.ndarray, dict]]:
        pass

    def step(
        self, action: np.ndarray
    ) -> Union[gym_old_venv_step_type, gym_new_venv_step_type]:
        """Perform one timestep of the environment's dynamic.

        "send" and "recv" are coupled in sync simulation, so users only call
        "step" function. But they can be called separately in async
        simulation, i.e. someone calls "send" first, and calls "recv" later.
        """
        self.send(action)
        return self.recv()  # type: ignore

    @staticmethod
    def wait(
        workers: list["EnvWorker"], wait_num: int, timeout: Optional[float] = None
    ) -> list["EnvWorker"]:
        """Given a list of workers, return those ready ones."""
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None) -> Optional[list[int]]:
        # return self.action_space.seed(seed)  # issue 299
        pass

    @abstractmethod
    def render(self, **kwargs: Any) -> Any:
        """Render the environment."""
        pass

    @abstractmethod
    def close_env(self) -> None:
        pass

    def close(self) -> None:
        if self.is_closed:
            return None
        self.is_closed = True
        self.close_env()


class ShArray:
    """Wrapper of multiprocessing Array."""

    def __init__(self, dtype: np.generic, shape: tuple[int]) -> None:
        self.arr = Array(_NP_TO_CT[dtype.type], int(np.prod(shape)))  # type: ignore
        self.dtype = dtype
        self.shape = shape

    def save(self, ndarray: np.ndarray) -> None:
        assert isinstance(ndarray, np.ndarray)
        dst = self.arr.get_obj()
        dst_np = np.frombuffer(dst, dtype=self.dtype).reshape(self.shape)  # type: ignore
        np.copyto(dst_np, ndarray)

    def get(self) -> np.ndarray:
        obj = self.arr.get_obj()
        return np.frombuffer(obj, dtype=self.dtype).reshape(self.shape)  # type: ignore


def _setup_buf(space: gym.Space) -> Union[dict, tuple, ShArray]:
    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict)
        return {k: _setup_buf(v) for k, v in space.spaces.items()}
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(space.spaces, tuple)
        return tuple([_setup_buf(t) for t in space.spaces])
    else:
        return ShArray(space.dtype, space.shape)  # type: ignore


def _apply_subproc_env_cpu_affinity(local_env_index: int) -> None:
    core_group = get_env_core_group_from_env(os.environ, local_env_index)
    if core_group is not None:
        apply_process_cpu_affinity(core_group)


def _worker(
    parent: connection.Connection,
    p: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    obs_bufs: Optional[Union[dict, tuple, ShArray]] = None,
    local_env_index: int = -1,
) -> None:
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
                env_return = env.step(data)
                if obs_bufs is not None:
                    _encode_obs(env_return[0], obs_bufs)
                    env_return = (None, *env_return[1:])
                p.send(env_return)
            elif cmd == "chunk_step":
                if obs_bufs is not None:
                    raise NotImplementedError(
                        "chunk_step does not support shared-memory observations"
                    )
                env_returns = [env.step(action) for action in data]
                p.send(tuple(zip(*env_returns)))
            elif cmd == "set_cpu_affinity":
                apply_process_cpu_affinity(tuple(data))
                p.send(tuple(sorted(os.sched_getaffinity(0))))
            elif cmd == "get_cpu_affinity":
                p.send(tuple(sorted(os.sched_getaffinity(0))))
            elif cmd == "reset":
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
                if obs_bufs is not None:
                    _encode_obs(obs, obs_bufs)
                    obs = None
                if reset_returns_info:
                    p.send((obs, info))
                else:
                    p.send(obs)
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
            elif cmd == "check_success":
                p.send(env.check_success())
            elif cmd == "get_segmentation_of_interest":
                p.send(env.get_segmentation_of_interest(data))
            elif cmd == "get_sim_state":
                p.send(env.get_sim_state())
            elif cmd == "set_init_state":
                obs = env.set_init_state(data)
                p.send(obs)
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class DummyEnvWorker(EnvWorker):
    """Dummy worker used in sequential vector environments."""

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        self.env = env_fn()
        super().__init__(env_fn)

    def get_env_attr(self, key: str) -> Any:
        return getattr(self.env, key)

    def set_env_attr(self, key: str, value: Any) -> None:
        setattr(self.env.unwrapped, key, value)

    def set_cpu_affinity(self, cpus: tuple[int, ...]) -> None:
        _ = cpus

    def get_cpu_affinity(self) -> tuple[int, ...]:
        if not hasattr(os, "sched_getaffinity"):
            return ()
        return tuple(sorted(os.sched_getaffinity(0)))

    def reset(self, **kwargs: Any) -> Union[np.ndarray, tuple[np.ndarray, dict]]:
        if "seed" in kwargs:
            super().seed(kwargs["seed"])
        return self.env.reset(**kwargs)

    @staticmethod
    def wait(  # type: ignore
        workers: list["DummyEnvWorker"], wait_num: int, timeout: Optional[float] = None
    ) -> list["DummyEnvWorker"]:
        # Sequential EnvWorker objects are always ready
        return workers

    def send(self, action: Optional[np.ndarray], **kwargs: Any) -> None:
        if action is None:
            self.result = self.env.reset(**kwargs)
        else:
            self.result = self.env.step(action)  # type: ignore

    def send_chunk_step(self, chunk_action: np.ndarray) -> None:
        env_returns = [self.env.step(action) for action in chunk_action]
        self.result = tuple(zip(*env_returns))  # type: ignore

    def seed(self, seed: Optional[int] = None) -> Optional[list[int]]:
        super().seed(seed)
        try:
            return self.env.seed(seed)  # type: ignore
        except (AttributeError, NotImplementedError):
            self.env.reset(seed=seed)
            return [seed]  # type: ignore

    def render(self, **kwargs: Any) -> Any:
        return self.env.render(**kwargs)

    def close_env(self) -> None:
        self.env.close()

    def check_success(self):
        return self.env.check_success()

    def get_segmentation_of_interest(self, segmentation_image):
        return self.env.get_segmentation_of_interest(segmentation_image)

    def get_sim_state(self):
        return self.env.get_sim_state()

    def set_init_state(self, init_state):
        return self.env.set_init_state(init_state)


class SubprocEnvWorker(EnvWorker):
    """Subprocess worker used in SubprocVectorEnv and ShmemVectorEnv."""

    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        share_memory: bool = False,
        local_env_index: int = -1,
    ) -> None:
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
        self.process = Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()
        super().__init__(env_fn)

    def get_env_attr(self, key: str) -> Any:
        self.parent_remote.send(["getattr", key])
        return self.parent_remote.recv()

    def set_env_attr(self, key: str, value: Any) -> None:
        self.parent_remote.send(["setattr", {"key": key, "value": value}])

    def set_cpu_affinity(self, cpus: tuple[int, ...]) -> None:
        cpus = tuple(cpus)
        if self._cpu_affinity == cpus:
            return
        self.parent_remote.send(["set_cpu_affinity", cpus])
        self._cpu_affinity = tuple(self.parent_remote.recv())

    def get_cpu_affinity(self) -> tuple[int, ...]:
        self.parent_remote.send(["get_cpu_affinity", None])
        return tuple(self.parent_remote.recv())

    def _decode_obs(self) -> Union[dict, tuple, np.ndarray]:
        def decode_obs(
            buffer: Optional[Union[dict, tuple, ShArray]],
        ) -> Union[dict, tuple, np.ndarray]:
            if isinstance(buffer, ShArray):
                return buffer.get()
            elif isinstance(buffer, tuple):
                return tuple([decode_obs(b) for b in buffer])
            elif isinstance(buffer, dict):
                return {k: decode_obs(v) for k, v in buffer.items()}
            else:
                raise NotImplementedError

        return decode_obs(self.buffer)

    @staticmethod
    def wait(  # type: ignore
        workers: list["SubprocEnvWorker"],
        wait_num: int,
        timeout: Optional[float] = None,
    ) -> list["SubprocEnvWorker"]:
        remain_conns = conns = [x.parent_remote for x in workers]
        ready_conns: list[connection.Connection] = []
        remain_time, t1 = timeout, time.time()
        while len(remain_conns) > 0 and len(ready_conns) < wait_num:
            if timeout:
                remain_time = timeout - (time.time() - t1)
                if remain_time <= 0:
                    break
            # connection.wait hangs if the list is empty
            new_ready_conns = connection.wait(remain_conns, timeout=remain_time)
            ready_conns.extend(new_ready_conns)  # type: ignore
            remain_conns = [conn for conn in remain_conns if conn not in ready_conns]
        return [workers[conns.index(con)] for con in ready_conns]

    def send(self, action: Optional[np.ndarray], **kwargs: Any) -> None:
        if action is None:
            if "seed" in kwargs:
                super().seed(kwargs["seed"])
            self.parent_remote.send(["reset", kwargs])
        else:
            self.parent_remote.send(["step", action])

    def send_chunk_step(self, chunk_action: np.ndarray) -> None:
        self.parent_remote.send(["chunk_step", chunk_action])

    def recv(
        self,
    ) -> Union[
        gym_old_venv_step_type,
        gym_new_venv_step_type,
        tuple[np.ndarray, dict],
        np.ndarray,
    ]:  # noqa:E125
        result = self.parent_remote.recv()
        if isinstance(result, tuple):
            if len(result) == 2:
                obs, info = result
                if self.share_memory:
                    obs = self._decode_obs()
                return obs, info
            obs = result[0]
            if self.share_memory:
                obs = self._decode_obs()
            return (obs, *result[1:])  # type: ignore
        else:
            obs = result
            if self.share_memory:
                obs = self._decode_obs()
            return obs

    def reset(self, **kwargs: Any) -> Union[np.ndarray, tuple[np.ndarray, dict]]:
        if "seed" in kwargs:
            super().seed(kwargs["seed"])
        self.parent_remote.send(["reset", kwargs])

        result = self.parent_remote.recv()
        if isinstance(result, tuple):
            obs, info = result
            if self.share_memory:
                obs = self._decode_obs()
            return obs, info
        else:
            obs = result
            if self.share_memory:
                obs = self._decode_obs()
            return obs

    def seed(self, seed: Optional[int] = None) -> Optional[list[int]]:
        super().seed(seed)
        self.parent_remote.send(["seed", seed])
        ret = self.parent_remote.recv()
        return ret

    def render(self, **kwargs: Any) -> Any:
        self.parent_remote.send(["render", kwargs])
        return self.parent_remote.recv()

    def close_env(self) -> None:
        try:
            self.parent_remote.send(["close", None])
            # mp may be deleted so it may raise AttributeError
            self.parent_remote.recv()
            self.process.join()
        except (BrokenPipeError, EOFError, AttributeError):
            pass
        # ensure the subproc is terminated
        self.process.terminate()

    def check_success(self):
        self.parent_remote.send(["check_success", None])
        return self.parent_remote.recv()

    def get_segmentation_of_interest(self, segmentation_image):
        self.parent_remote.send(["get_segmentation_of_interest", segmentation_image])
        return self.parent_remote.recv()

    def get_sim_state(self):
        self.parent_remote.send(["get_sim_state", None])
        return self.parent_remote.recv()

    def set_init_state(self, init_state):
        self.parent_remote.send(["set_init_state", init_state])
        obs = self.parent_remote.recv()
        if self.share_memory:
            obs = self._decode_obs()
        return obs


################################################################################
#
# VecEnvs
#
################################################################################


class BaseVectorEnv(object):
    """Base class for vectorized environments.

    Usage:
    ::

        env_num = 8
        envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(env_num)])
        assert len(envs) == env_num

    It accepts a list of environment generators. In other words, an environment
    generator ``efn`` of a specific task means that ``efn()`` returns the
    environment of the given task, for example, ``gym.make(task)``.

    All of the VectorEnv must inherit :class:`~tianshou.env.BaseVectorEnv`.
    Here are some other usages:
    ::

        envs.seed(2)  # which is equal to the next line
        envs.seed([2, 3, 4, 5, 6, 7, 8, 9])  # set specific seed for each env
        obs = envs.reset()  # reset all environments
        obs = envs.reset([0, 5, 7])  # reset 3 specific environments
        obs, rew, done, info = envs.step([1] * 8)  # step synchronously
        envs.render()  # render all environments
        envs.close()  # close all environments

    .. warning::

        If you use your own environment, please make sure the ``seed`` method
        is set up properly, e.g.,
        ::

            def seed(self, seed):
                np.random.seed(seed)

        Otherwise, the outputs of these envs may be the same with each other.

    :param env_fns: a list of callable envs, ``env_fns[i]()`` generates the i-th env.
    :param worker_fn: a callable worker, ``worker_fn(env_fns[i])`` generates a
        worker which contains the i-th env.
    :param int wait_num: use in asynchronous simulation if the time cost of
        ``env.step`` varies with time and synchronously waiting for all
        environments to finish a step is time-wasting. In that case, we can
        return when ``wait_num`` environments finish a step and keep on
        simulation in these environments. If ``None``, asynchronous simulation
        is disabled; else, ``1 <= wait_num <= env_num``.
    :param float timeout: use in asynchronous simulation same as above, in each
        vectorized step it only deal with those environments spending time
        within ``timeout`` seconds.
    """

    def __init__(
        self,
        env_fns: list[Callable[[], gym.Env]],
        worker_fn: Callable[[Callable[[], gym.Env]], EnvWorker],
        wait_num: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._env_fns = env_fns
        # A VectorEnv contains a pool of EnvWorkers, which corresponds to
        # interact with the given envs (one worker <-> one env).
        self.workers = [worker_fn(fn) for fn in env_fns]
        self.worker_class = type(self.workers[0])
        assert issubclass(self.worker_class, EnvWorker)
        assert all(isinstance(w, self.worker_class) for w in self.workers)

        self.env_num = len(env_fns)
        self.wait_num = wait_num or len(env_fns)
        assert 1 <= self.wait_num <= len(env_fns), (
            f"wait_num should be in [1, {len(env_fns)}], but got {wait_num}"
        )
        self.timeout = timeout
        assert self.timeout is None or self.timeout > 0, (
            f"timeout is {timeout}, it should be positive if provided!"
        )
        self.is_async = self.wait_num != len(env_fns) or timeout is not None
        self.waiting_conn: list[EnvWorker] = []
        # environments in self.ready_id is actually ready
        # but environments in self.waiting_id are just waiting when checked,
        # and they may be ready now, but this is not known until we check it
        # in the step() function
        self.waiting_id: list[int] = []
        # all environments are ready in the beginning
        self.ready_id = list(range(self.env_num))
        self.is_closed = False
        self._balanced_pair_predicted_latency_s: list[float] | None = None
        self._env_cpu_core_groups = parse_env_cpu_core_groups(
            os.environ.get("RLINF_ENV_CPU_CORE_GROUPS", "")
        )
        self._balanced_pair_logged = False
        self._sim_timestamp_context: dict[str, Any] | None = None
        self._sim_timestamp_file = None
        self._sim_vector_step_index = 0
        self._sim_async_step_starts: dict[int, tuple[float, dict[str, Any]]] = {}
        self._last_chunk_profile: dict[str, Any] | None = None

    def _assert_is_not_closed(self) -> None:
        assert not self.is_closed, (
            f"Methods of {self.__class__.__name__} cannot be called after close."
        )

    def __len__(self) -> int:
        """Return len(self), which is the number of environments."""
        return self.env_num

    def set_sim_timestamp_context(self, context: dict[str, Any] | None) -> None:
        """Set per-sub-env timestamp context for the next vector env call."""
        self._sim_timestamp_context = dict(context) if context is not None else None
        self._sim_vector_step_index = 0
        self._sim_async_step_starts.clear()

    def get_last_chunk_profile(self) -> dict[str, Any] | None:
        """Return the latest vector chunk-step timing breakdown."""
        return dict(self._last_chunk_profile) if self._last_chunk_profile else None

    def _get_sim_timestamp_file(self):
        context = self._sim_timestamp_context
        if context is None:
            return None
        if self._sim_timestamp_file is None:
            output_dir = str(context["output_dir"])
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"env_rank_{int(context['rank'])}.jsonl")
            self._sim_timestamp_file = open(path, "a", encoding="utf-8", buffering=1)
        return self._sim_timestamp_file

    def _global_env_id(self, local_env_id: int) -> int | None:
        context = self._sim_timestamp_context
        if context is None:
            return None
        try:
            return (
                int(context["rank"]) * int(context["stage_num"]) + int(context["stage"])
            ) * int(context["local_envs"]) + int(local_env_id)
        except (KeyError, TypeError, ValueError):
            return None

    def _worker_pid(self, worker: EnvWorker) -> int | None:
        process = getattr(worker, "process", None)
        pid = getattr(process, "pid", None)
        return int(pid) if pid is not None else None

    def _worker_cpu_affinity(self, worker: EnvWorker) -> tuple[int, ...]:
        affinity = getattr(worker, "_cpu_affinity", None)
        return tuple(affinity) if affinity is not None else ()

    def _set_worker_process_cpu_affinity(
        self, worker: EnvWorker, cpus: tuple[int, ...]
    ) -> bool:
        """Set a subprocess worker's affinity without using its command pipe."""
        process = getattr(worker, "process", None)
        pid = getattr(process, "pid", None)
        if pid is None or not hasattr(os, "sched_setaffinity"):
            return False
        os.sched_setaffinity(int(pid), set(cpus))
        if hasattr(worker, "_cpu_affinity"):
            worker._cpu_affinity = tuple(cpus)
        return True

    def _write_subenv_timestamp_event(
        self,
        event: str,
        *,
        local_env_id: int,
        worker: EnvWorker,
        operation: str,
        perf_start: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        context = self._sim_timestamp_context
        if context is None:
            return
        handle = self._get_sim_timestamp_file()
        if handle is None:
            return

        wall_ns = time.time_ns()
        record: dict[str, Any] = {
            "event": event,
            "rank": int(context["rank"]),
            "pid": int(context["pid"]),
            "child_pid": self._worker_pid(worker),
            "epoch": context.get("epoch"),
            "chunk_step": context.get("chunk_step"),
            "stage": context.get("stage"),
            "local_envs": context.get("local_envs"),
            "local_env": int(local_env_id),
            "global_env": self._global_env_id(int(local_env_id)),
            "operation": operation,
            "wall_ns": wall_ns,
            "cpu_affinity": self._worker_cpu_affinity(worker),
        }
        if extra:
            record.update(_to_jsonable(extra))
        if perf_start is not None:
            record["duration_s"] = max(time.perf_counter() - perf_start, 0.0)
        handle.write(json.dumps(_to_jsonable(record), sort_keys=True) + "\n")

    def __getattribute__(self, key: str) -> Any:
        """Switch the attribute getter depending on the key.

        Any class who inherits ``gym.Env`` will inherit some attributes, like
        ``action_space``. However, we would like the attribute lookup to go straight
        into the worker (in fact, this vector env's action_space is always None).
        """
        if key in GYM_RESERVED_KEYS:  # reserved keys in gym.Env
            return self.get_env_attr(key)
        else:
            return super().__getattribute__(key)

    def get_env_attr(
        self,
        key: str,
        id: Optional[Union[int, list[int], np.ndarray]] = None,
    ) -> list[Any]:
        """Get an attribute from the underlying environments.

        If id is an int, retrieve the attribute denoted by key from the environment
        underlying the worker at index id. The result is returned as a list with one
        element. Otherwise, retrieve the attribute for all workers at indices id and
        return a list that is ordered correspondingly to id.

        :param str key: The key of the desired attribute.
        :param id: Indice(s) of the desired worker(s). Default to None for all env_id.

        :return list: The list of environment attributes.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        return [self.workers[j].get_env_attr(key) for j in id]

    def set_env_attr(
        self,
        key: str,
        value: Any,
        id: Optional[Union[int, list[int], np.ndarray]] = None,
    ) -> None:
        """Set an attribute in the underlying environments.

        If id is an int, set the attribute denoted by key from the environment
        underlying the worker at index id to value.
        Otherwise, set the attribute for all workers at indices id.

        :param str key: The key of the desired attribute.
        :param Any value: The new value of the attribute.
        :param id: Indice(s) of the desired worker(s). Default to None for all env_id.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)
        for j in id:
            self.workers[j].set_env_attr(key, value)

    def _wrap_id(
        self,
        id: Optional[Union[int, list[int], np.ndarray]] = None,
    ) -> Union[list[int], np.ndarray]:
        if id is None:
            return list(range(self.env_num))
        return [id] if np.isscalar(id) else id  # type: ignore

    def _assert_id(self, id: Union[list[int], np.ndarray]) -> None:
        for i in id:
            assert i not in self.waiting_id, (
                f"Cannot interact with environment {i} which is stepping now."
            )
            assert i in self.ready_id, (
                f"Can only interact with ready environments {self.ready_id}."
            )

    def reset(
        self,
        id: Optional[Union[int, list[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, tuple[np.ndarray, Union[dict, list[dict]]]]:
        """Reset the state of some envs and return initial observations.

        If id is None, reset the state of all the environments and return
        initial observations, otherwise reset the specific environments with
        the given id, either an int or a list.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        # send(None) == reset() in worker
        for i in id:
            self.workers[i].send(None, **kwargs)
        ret_list = [self.workers[i].recv() for i in id]

        reset_returns_info = (
            isinstance(ret_list[0], (tuple, list))
            and len(ret_list[0]) == 2
            and isinstance(ret_list[0][1], dict)
        )
        if reset_returns_info:
            obs_list = [r[0] for r in ret_list]
        else:
            obs_list = ret_list

        if isinstance(obs_list[0], tuple):
            raise TypeError(
                "Tuple observation space is not supported. ",
                "Please change it to array or dict space",
            )
        try:
            obs = np.stack(obs_list)
        except ValueError:  # different len(obs)
            obs = np.array(obs_list, dtype=object)

        if reset_returns_info:
            infos = [r[1] for r in ret_list]
            return obs, infos  # type: ignore
        else:
            return obs

    def step(
        self,
        action: np.ndarray,
        id: Optional[Union[int, list[int], np.ndarray]] = None,
    ) -> Union[gym_old_venv_step_type, gym_new_venv_step_type]:
        """Run one timestep of some environments' dynamics.

        If id is None, run one timestep of all the environments’ dynamics;
        otherwise run one timestep for some environments with given id,  either
        an int or a list. When the end of episode is reached, you are
        responsible for calling reset(id) to reset this environment’s state.

        Accept a batch of action and return a tuple (batch_obs, batch_rew,
        batch_done, batch_info) in numpy format.

        :param numpy.ndarray action: a batch of action provided by the agent.

        :return: A tuple consisting of either:

            * ``obs`` a numpy.ndarray, the agent's observation of current environments
            * ``rew`` a numpy.ndarray, the amount of rewards returned after \
                previous actions
            * ``done`` a numpy.ndarray, whether these episodes have ended, in \
                which case further step() calls will return undefined results
            * ``info`` a numpy.ndarray, contains auxiliary diagnostic \
                information (helpful for debugging, and sometimes learning)

            or:

            * ``obs`` a numpy.ndarray, the agent's observation of current environments
            * ``rew`` a numpy.ndarray, the amount of rewards returned after \
                previous actions
            * ``terminated`` a numpy.ndarray, whether these episodes have been \
                terminated
            * ``truncated`` a numpy.ndarray, whether these episodes have been truncated
            * ``info`` a numpy.ndarray, contains auxiliary diagnostic \
                information (helpful for debugging, and sometimes learning)

            The case distinction is made based on whether the underlying environment
            uses the old step API (first case) or the new step API (second case).

        For the async simulation:

        Provide the given action to the environments. The action sequence
        should correspond to the ``id`` argument, and the ``id`` argument
        should be a subset of the ``env_id`` in the last returned ``info``
        (initially they are env_ids of all the environments). If action is
        None, fetch unfinished step() calls instead.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if not self.is_async:
            assert len(action) == len(id)
            vector_step_index = self._sim_vector_step_index
            start_times: dict[int, float] = {}
            for i, j in enumerate(id):
                env_id = int(j)
                worker = self.workers[env_id]
                self._write_subenv_timestamp_event(
                    "subenv_start",
                    local_env_id=env_id,
                    worker=worker,
                    operation="step",
                    extra={"vector_step": vector_step_index},
                )
                start_times[env_id] = time.perf_counter()
                self.workers[j].send(action[i])
            results: list[Any | None] = [None for _ in id]
            in_flight = {
                self.workers[int(env_id)]: (index, int(env_id))
                for index, env_id in enumerate(id)
            }
            while in_flight:
                ready_workers = self.worker_class.wait(list(in_flight), 1, self.timeout)
                if not ready_workers:
                    continue
                for worker in ready_workers:
                    index, env_id = in_flight.pop(worker)
                    env_return = worker.recv()
                    self._write_subenv_timestamp_event(
                        "subenv_end",
                        local_env_id=env_id,
                        worker=worker,
                        operation="step",
                        perf_start=start_times.get(env_id),
                        extra={"vector_step": vector_step_index},
                    )
                    env_return[-1]["env_id"] = id[index]
                    results[index] = env_return
            result = [env_return for env_return in results if env_return is not None]
            if len(result) != len(id):
                raise RuntimeError("step missed env results")
            self._sim_vector_step_index += 1
        else:
            if action is not None:
                self._assert_id(id)
                assert len(action) == len(id)
                vector_step_index = self._sim_vector_step_index
                for act, env_id in zip(action, id):
                    local_env_id = int(env_id)
                    worker = self.workers[local_env_id]
                    extra = {"vector_step": vector_step_index}
                    self._write_subenv_timestamp_event(
                        "subenv_start",
                        local_env_id=local_env_id,
                        worker=worker,
                        operation="step",
                        extra=extra,
                    )
                    self._sim_async_step_starts[local_env_id] = (
                        time.perf_counter(),
                        extra,
                    )
                    worker.send(act)
                    self.waiting_conn.append(worker)
                    self.waiting_id.append(local_env_id)
                self.ready_id = [x for x in self.ready_id if x not in id]
                self._sim_vector_step_index += 1
            ready_conns: list[EnvWorker] = []
            while not ready_conns:
                ready_conns = self.worker_class.wait(
                    self.waiting_conn, self.wait_num, self.timeout
                )
            result = []
            for conn in ready_conns:
                waiting_index = self.waiting_conn.index(conn)
                self.waiting_conn.pop(waiting_index)
                env_id = self.waiting_id.pop(waiting_index)
                # env_return can be (obs, reward, done, info) or
                # (obs, reward, terminated, truncated, info)
                env_return = conn.recv()
                start_info = self._sim_async_step_starts.pop(int(env_id), None)
                perf_start = start_info[0] if start_info is not None else None
                extra = start_info[1] if start_info is not None else None
                self._write_subenv_timestamp_event(
                    "subenv_end",
                    local_env_id=int(env_id),
                    worker=conn,
                    operation="step",
                    perf_start=perf_start,
                    extra=extra,
                )
                env_return[-1]["env_id"] = env_id  # Add `env_id` to info
                result.append(env_return)
                self.ready_id.append(env_id)
        return_lists = tuple(zip(*result))
        obs_list = return_lists[0]
        try:
            obs_stack = np.stack(obs_list)
        except ValueError:  # different len(obs)
            obs_stack = np.array(obs_list, dtype=object)
        other_stacks = map(np.stack, return_lists[1:])
        return (obs_stack, *other_stacks)  # type: ignore

    def chunk_step(
        self,
        chunk_action: np.ndarray,
        id: Optional[Union[int, list[int], np.ndarray]] = None,
    ) -> tuple[list[Any], ...]:
        """Run full local action chunks in each worker before gathering results."""
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)
        assert len(chunk_action) == len(id)

        start_times: dict[int, float] = {}
        profile_start = time.perf_counter()
        send_start = profile_start
        for i, j in enumerate(id):
            env_id = int(j)
            worker = self.workers[env_id]
            extra = {
                "action_chunk_steps": int(chunk_action[i].shape[0]),
                "vector_step": self._sim_vector_step_index,
            }
            self._write_subenv_timestamp_event(
                "subenv_start",
                local_env_id=env_id,
                worker=worker,
                operation="chunk_step",
                extra=extra,
            )
            start_times[env_id] = time.perf_counter()
            worker.send_chunk_step(chunk_action[i])
        send_end = time.perf_counter()
        env_results: list[Any | None] = [None for _ in id]
        in_flight = {
            self.workers[int(env_id)]: (index, int(env_id))
            for index, env_id in enumerate(id)
        }
        wait_recv_start = time.perf_counter()
        first_ready_time: float | None = None
        last_recv_time: float | None = None
        while in_flight:
            ready_workers = self.worker_class.wait(list(in_flight), 1, self.timeout)
            if not ready_workers:
                continue
            ready_time = time.perf_counter()
            if first_ready_time is None:
                first_ready_time = ready_time
            for worker in ready_workers:
                index, env_id = in_flight.pop(worker)
                env_results[index] = worker.recv()
                last_recv_time = time.perf_counter()
                self._write_subenv_timestamp_event(
                    "subenv_end",
                    local_env_id=env_id,
                    worker=worker,
                    operation="chunk_step",
                    perf_start=start_times.get(env_id),
                    extra={
                        "action_chunk_steps": int(chunk_action[index].shape[0]),
                        "vector_step": self._sim_vector_step_index,
                    },
                )
        wait_recv_end = time.perf_counter()
        self._sim_vector_step_index += 1
        ordered_env_results = [
            env_result for env_result in env_results if env_result is not None
        ]
        if len(ordered_env_results) != len(id):
            raise RuntimeError("chunk_step missed env results")
        stack_start = time.perf_counter()
        stacked = stack_vector_chunk_returns(ordered_env_results)
        stack_end = time.perf_counter()
        self._last_chunk_profile = {
            "operation": "chunk_step",
            "env_count": len(id),
            "dispatch_s": send_end - send_start,
            "wait_recv_s": wait_recv_end - wait_recv_start,
            "time_to_first_ready_s": (
                first_ready_time - wait_recv_start
                if first_ready_time is not None
                else None
            ),
            "first_ready_to_last_recv_s": (
                last_recv_time - first_ready_time
                if first_ready_time is not None and last_recv_time is not None
                else None
            ),
            "stack_s": stack_end - stack_start,
            "total_s": stack_end - profile_start,
        }
        return stacked

    def latency_balanced_pair_chunk_step(
        self,
        chunk_action: np.ndarray,
        id: Optional[Union[int, list[int], np.ndarray]] = None,
        *,
        envs_per_core: int = 1,
        ema_alpha: float = 0.3,
        initial_latency_ms: Optional[float] = None,
        dynamic_affinity: bool = True,
        core_donation_enabled: bool = True,
        core_donation_max_extra_groups: int = 1,
    ) -> tuple[list[Any], ...]:
        """Run local chunks with core donation v2 scheduling.

        This is the only supported latency-balanced mode. Each env has its own
        base CPU core group. When an env finishes, its group can be temporarily
        donated to a slower in-flight env, then restored before returning.
        """
        self._assert_is_not_closed()
        id = list(self._wrap_id(id))
        if self.is_async:
            self._assert_id(id)
        assert len(chunk_action) == len(id)
        if len(id) == 0:
            raise ValueError(
                "latency_balanced_pair_chunk_step requires at least one env"
            )

        chunk_size = int(chunk_action.shape[1])
        if chunk_size <= 0:
            raise ValueError(
                f"chunk_action must contain at least one step, got {chunk_size}"
            )

        envs_per_core = int(envs_per_core)
        if (
            envs_per_core != 1
            or not bool(dynamic_affinity)
            or not bool(core_donation_enabled)
        ):
            raise ValueError(
                "latency_balanced_pair only supports core donation v2: "
                "envs_per_core=1, dynamic_affinity=True, "
                "core_donation_enabled=True"
            )
        if not self._env_cpu_core_groups:
            raise ValueError(
                "latency_balanced_pair core donation v2 requires per-env CPU "
                "core groups. Use sync_time_major for the no-CPU-binding "
                "baseline."
            )
        if not 0.0 < float(ema_alpha) <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        core_donation_max_extra_groups = int(core_donation_max_extra_groups)
        if core_donation_max_extra_groups < 0:
            raise ValueError(
                "core_donation_max_extra_groups must be >= 0, "
                f"got {core_donation_max_extra_groups}"
            )

        initial_latency = (
            float(initial_latency_ms) / 1000.0
            if initial_latency_ms is not None
            else 1.0
        )
        if initial_latency <= 0.0:
            raise ValueError(
                "initial_latency_ms must be positive when set, "
                f"got {initial_latency_ms}"
            )
        if (
            self._balanced_pair_predicted_latency_s is None
            or len(self._balanced_pair_predicted_latency_s) != self.env_num
        ):
            self._balanced_pair_predicted_latency_s = [
                initial_latency for _ in range(self.env_num)
            ]

        slot_count = len(id) // envs_per_core
        profile_start = time.perf_counter()
        group_start = profile_start
        pair_groups = self._build_latency_balanced_groups(id, envs_per_core)
        group_end = time.perf_counter()
        affinity_start = group_end
        slot_cpu_core_groups: tuple[tuple[int, ...], ...] = ()
        base_affinity_by_worker: dict[EnvWorker, tuple[int, ...]] = {}
        active_affinity_by_worker: dict[EnvWorker, tuple[int, ...]] = {}
        extra_groups_by_worker: dict[EnvWorker, int] = {}
        running_workers_by_core_group: dict[tuple[int, ...], set[EnvWorker]] = {}
        if dynamic_affinity and self._env_cpu_core_groups:
            slot_cpu_core_groups = self._get_slot_cpu_core_groups(slot_count)
            if len(slot_cpu_core_groups) < slot_count:
                raise ValueError(
                    "latency_balanced_pair needs at least one CPU core group per "
                    f"slot, got {len(slot_cpu_core_groups)} groups for "
                    f"{slot_count} slots"
                )
            for slot_index, group in enumerate(pair_groups):
                core_group = slot_cpu_core_groups[slot_index]
                for local_pos in group:
                    worker = self.workers[id[local_pos]]
                    worker.set_cpu_affinity(core_group)
                    base_affinity_by_worker[worker] = core_group
                    active_affinity_by_worker[worker] = core_group
                    extra_groups_by_worker[worker] = 0
                    running_workers_by_core_group.setdefault(core_group, set()).add(
                        worker
                    )
        affinity_end = time.perf_counter()
        if not self._balanced_pair_logged:
            self._balanced_pair_logged = True
            logger.info(
                "latency_balanced_pair enabled: local_envs=%s, envs_per_core=%s, "
                "slot_count=%s, cpu_groups=%s, first_groups=%s",
                len(id),
                envs_per_core,
                slot_count,
                len(self._env_cpu_core_groups),
                self._env_cpu_core_groups[: min(8, len(self._env_cpu_core_groups))],
            )

        env_step_results: list[Any | None] = [None for _ in id]
        in_flight: dict[EnvWorker, tuple[int, int, int, float, dict[str, Any]]] = {}
        dispatch_call_time_s = 0.0
        wait_call_time_s = 0.0
        recv_call_time_s = 0.0
        core_donation_time_s = 0.0
        core_donation_restore_time_s = 0.0
        core_donation_count = 0
        first_ready_time: float | None = None
        last_recv_time: float | None = None

        donation_enabled = (
            bool(core_donation_enabled)
            and dynamic_affinity
            and bool(slot_cpu_core_groups)
            and core_donation_max_extra_groups > 0
        )

        def restore_donated_core_groups() -> None:
            nonlocal core_donation_restore_time_s
            if not donation_enabled:
                return
            restore_start = time.perf_counter()
            for worker, base_affinity in base_affinity_by_worker.items():
                if (
                    active_affinity_by_worker.get(worker, base_affinity)
                    != base_affinity
                ):
                    if not self._set_worker_process_cpu_affinity(worker, base_affinity):
                        worker.set_cpu_affinity(base_affinity)
                    active_affinity_by_worker[worker] = base_affinity
            core_donation_restore_time_s = time.perf_counter() - restore_start

        def donate_finished_core_group(finished_worker: EnvWorker) -> None:
            nonlocal core_donation_count, core_donation_time_s
            if not donation_enabled or not in_flight:
                return
            donated_group = base_affinity_by_worker.get(finished_worker, ())
            if not donated_group:
                return
            running_workers = running_workers_by_core_group.get(donated_group)
            if running_workers:
                running_workers.discard(finished_worker)
                if running_workers:
                    return
            donation_targets = sorted(
                in_flight.items(),
                key=lambda item: (
                    -self._balanced_pair_predicted_latency_s[id[item[1][0]]],
                    item[1][0],
                ),
            )
            for target_worker, (local_pos, *_rest) in donation_targets:
                if extra_groups_by_worker.get(target_worker, 0) >= (
                    core_donation_max_extra_groups
                ):
                    continue
                current_affinity = active_affinity_by_worker.get(
                    target_worker,
                    base_affinity_by_worker.get(target_worker, ()),
                )
                new_affinity = tuple(sorted(set(current_affinity) | set(donated_group)))
                if new_affinity == current_affinity:
                    continue
                donation_start = time.perf_counter()
                if not self._set_worker_process_cpu_affinity(
                    target_worker, new_affinity
                ):
                    target_worker.set_cpu_affinity(new_affinity)
                donation_end = time.perf_counter()
                core_donation_time_s += donation_end - donation_start
                active_affinity_by_worker[target_worker] = new_affinity
                extra_groups_by_worker[target_worker] = (
                    extra_groups_by_worker.get(target_worker, 0) + 1
                )
                core_donation_count += 1
                return

        def dispatch_slot_env(slot_index: int, pair_offset: int) -> None:
            nonlocal dispatch_call_time_s
            group = pair_groups[slot_index]
            if pair_offset >= len(group):
                return
            local_pos = group[pair_offset]
            env_id = id[local_pos]
            worker = self.workers[env_id]
            extra = {
                "action_chunk_steps": chunk_size,
                "pair_offset": pair_offset,
                "pair_slot": slot_index,
                "predicted_latency_s": self._balanced_pair_predicted_latency_s[env_id],
                "vector_step": self._sim_vector_step_index,
            }
            self._write_subenv_timestamp_event(
                "subenv_start",
                local_env_id=int(env_id),
                worker=worker,
                operation="latency_balanced_pair_chunk_step",
                extra=extra,
            )
            send_start = time.perf_counter()
            worker.send_chunk_step(chunk_action[local_pos])
            send_end = time.perf_counter()
            dispatch_call_time_s += send_end - send_start
            in_flight[worker] = (
                local_pos,
                slot_index,
                pair_offset,
                time.perf_counter(),
                extra,
            )

        initial_dispatch_start = time.perf_counter()
        for slot_index in range(slot_count):
            dispatch_slot_env(slot_index, 0)
        initial_dispatch_end = time.perf_counter()

        wait_recv_start = time.perf_counter()
        try:
            while in_flight:
                wait_start = time.perf_counter()
                ready_workers = self.worker_class.wait(list(in_flight), 1, self.timeout)
                wait_end = time.perf_counter()
                wait_call_time_s += wait_end - wait_start
                if not ready_workers:
                    continue
                if first_ready_time is None:
                    first_ready_time = wait_end
                for worker in ready_workers:
                    (
                        local_pos,
                        slot_index,
                        pair_offset,
                        start_time,
                        extra,
                    ) = in_flight.pop(worker)
                    recv_start = time.perf_counter()
                    env_step_results[local_pos] = worker.recv()
                    recv_end = time.perf_counter()
                    recv_call_time_s += recv_end - recv_start
                    last_recv_time = recv_end
                    actual_latency = max(time.perf_counter() - start_time, 0.0)
                    env_id = id[local_pos]
                    old_latency = self._balanced_pair_predicted_latency_s[env_id]
                    self._balanced_pair_predicted_latency_s[env_id] = (
                        float(ema_alpha) * actual_latency
                        + (1.0 - float(ema_alpha)) * old_latency
                    )
                    end_extra = dict(extra)
                    end_extra["updated_predicted_latency_s"] = (
                        self._balanced_pair_predicted_latency_s[env_id]
                    )
                    self._write_subenv_timestamp_event(
                        "subenv_end",
                        local_env_id=int(env_id),
                        worker=worker,
                        operation="latency_balanced_pair_chunk_step",
                        perf_start=start_time,
                        extra=end_extra,
                    )
                    donate_finished_core_group(worker)
                    dispatch_slot_env(slot_index, pair_offset + 1)
        finally:
            restore_donated_core_groups()
        wait_recv_end = time.perf_counter()

        env_results = [result for result in env_step_results if result is not None]
        if len(env_results) != len(id):
            raise RuntimeError("latency-balanced pair chunk_step missed env results")
        self._sim_vector_step_index += 1
        stack_start = time.perf_counter()
        stacked = stack_vector_chunk_returns(env_results)
        stack_end = time.perf_counter()
        self._last_chunk_profile = {
            "operation": "latency_balanced_pair_chunk_step",
            "env_count": len(id),
            "envs_per_core": envs_per_core,
            "slot_count": slot_count,
            "group_s": group_end - group_start,
            "affinity_s": affinity_end - affinity_start,
            "initial_dispatch_s": initial_dispatch_end - initial_dispatch_start,
            "dispatch_call_s": dispatch_call_time_s,
            "wait_recv_s": wait_recv_end - wait_recv_start,
            "wait_call_s": wait_call_time_s,
            "recv_call_s": recv_call_time_s,
            "core_donation_enabled": donation_enabled,
            "core_donation_count": core_donation_count,
            "core_donation_s": core_donation_time_s,
            "core_donation_restore_s": core_donation_restore_time_s,
            "time_to_first_ready_s": (
                first_ready_time - wait_recv_start
                if first_ready_time is not None
                else None
            ),
            "first_ready_to_last_recv_s": (
                last_recv_time - first_ready_time
                if first_ready_time is not None and last_recv_time is not None
                else None
            ),
            "stack_s": stack_end - stack_start,
            "total_s": stack_end - profile_start,
        }
        return stacked

    def _get_slot_cpu_core_groups(self, slot_count: int) -> tuple[tuple[int, ...], ...]:
        unique_groups: list[tuple[int, ...]] = []
        seen_groups: set[tuple[int, ...]] = set()
        for group in self._env_cpu_core_groups:
            if group in seen_groups:
                continue
            unique_groups.append(group)
            seen_groups.add(group)
            if len(unique_groups) == slot_count:
                return tuple(unique_groups)
        return self._env_cpu_core_groups[:slot_count]

    def _build_latency_balanced_groups(
        self, env_ids: list[int], envs_per_core: int
    ) -> list[list[int]]:
        if self._balanced_pair_predicted_latency_s is None:
            raise RuntimeError("latency predictions are not initialized")
        slot_count = len(env_ids) // envs_per_core
        groups: list[list[int]] = [[] for _ in range(slot_count)]
        loads = [0.0 for _ in range(slot_count)]
        local_positions = list(range(len(env_ids)))
        local_positions.sort(
            key=lambda pos: (
                -self._balanced_pair_predicted_latency_s[env_ids[pos]],
                pos,
            )
        )

        for local_pos in local_positions:
            target_slot = min(
                (
                    slot
                    for slot in range(slot_count)
                    if len(groups[slot]) < envs_per_core
                ),
                key=lambda slot: (loads[slot], len(groups[slot]), slot),
            )
            groups[target_slot].append(local_pos)
            loads[target_slot] += self._balanced_pair_predicted_latency_s[
                env_ids[local_pos]
            ]
        return groups

    def seed(
        self,
        seed: Optional[Union[int, list[int]]] = None,
    ) -> list[Optional[list[int]]]:
        """Set the seed for all environments.

        Accept ``None``, an int (which will extend ``i`` to
        ``[i, i + 1, i + 2, ...]``) or a list.

        :return: The list of seeds used in this env's random number generators.
            The first value in the list should be the "main" seed, or the value
            which a reproducer pass to "seed".
        """
        self._assert_is_not_closed()
        seed_list: Union[list[None], list[int]]
        if seed is None:
            seed_list = [seed] * self.env_num
        elif isinstance(seed, int):
            seed_list = [seed + i for i in range(self.env_num)]
        else:
            seed_list = seed
        return [w.seed(s) for w, s in zip(self.workers, seed_list)]

    def render(self, **kwargs: Any) -> list[Any]:
        """Render all of the environments."""
        self._assert_is_not_closed()
        if self.is_async and len(self.waiting_id) > 0:
            raise RuntimeError(
                f"Environments {self.waiting_id} are still stepping, cannot "
                "render them now."
            )
        return [w.render(**kwargs) for w in self.workers]

    def close(self) -> None:
        """Close all of the environments.

        This function will be called only once (if not, it will be called during
        garbage collected). This way, ``close`` of all workers can be assured.
        """
        self._assert_is_not_closed()
        for w in self.workers:
            w.close()
        if self._sim_timestamp_file is not None:
            self._sim_timestamp_file.close()
            self._sim_timestamp_file = None
        self.is_closed = True


class DummyVectorEnv(BaseVectorEnv):
    """Dummy vectorized environment wrapper, implemented in for-loop.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for other APIs' usage.
    """

    def __init__(self, env_fns: list[Callable[[], gym.Env]], **kwargs: Any) -> None:
        super().__init__(env_fns, DummyEnvWorker, **kwargs)

    def check_success(self):
        return [w.check_success() for w in self.workers]

    def get_segmentation_of_interest(self, segmentation_images):
        return [
            w.get_segmentation_of_interest(img)
            for w, img in zip(self.workers, segmentation_images)
        ]

    def get_sim_state(self):
        return [w.get_sim_state() for w in self.workers]

    def set_init_state(
        self,
        init_state: Optional[Union[int, list[int], np.ndarray]] = None,
        id: Optional[Union[int, list[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, tuple[np.ndarray, Union[dict, list[dict]]]]:
        """Reset the state of some envs and return initial observations.
        If id is None, reset the state of all the environments and return
        initial observations, otherwise reset the specific environments with
        the given id, either an int or a list.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        # send(None) == reset() in worker
        obs_list = []
        for j, i in enumerate(id):
            obs = self.workers[i].set_init_state(init_state[j])
            obs_list.append(obs)
        obs = np.stack(obs_list)
        return obs


class SubprocVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on subprocess.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for other APIs' usage.
    """

    def __init__(self, env_fns: list[Callable[[], gym.Env]], **kwargs: Any) -> None:
        env_index = {"value": 0}

        def worker_fn(fn: Callable[[], gym.Env]) -> SubprocEnvWorker:
            local_env_index = env_index["value"]
            env_index["value"] += 1
            return SubprocEnvWorker(
                fn, share_memory=False, local_env_index=local_env_index
            )

        super().__init__(env_fns, worker_fn, **kwargs)

    def check_success(self):
        return [w.check_success() for w in self.workers]

    def get_segmentation_of_interest(self, segmentation_images):
        return [
            w.get_segmentation_of_interest(img)
            for w, img in zip(self.workers, segmentation_images)
        ]

    def get_sim_state(self):
        return [w.get_sim_state() for w in self.workers]

    def set_init_state(
        self,
        init_state: Optional[Union[int, list[int], np.ndarray]] = None,
        id: Optional[Union[int, list[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, tuple[np.ndarray, Union[dict, list[dict]]]]:
        """Reset the state of some envs and return initial observations.
        If id is None, reset the state of all the environments and return
        initial observations, otherwise reset the specific environments with
        the given id, either an int or a list.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        # send(None) == reset() in worker
        obs_list = []
        for j, i in enumerate(id):
            obs = self.workers[i].set_init_state(init_state[j])
            obs_list.append(obs)
        obs = np.stack(obs_list)
        return obs
