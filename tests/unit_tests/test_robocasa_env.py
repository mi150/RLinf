from __future__ import annotations

import builtins
import importlib
import json
import sys
from types import ModuleType

import numpy as np
import torch
from omegaconf import OmegaConf


def test_get_env_fns_imports_robocasa_before_robosuite_make(
    monkeypatch,
) -> None:
    fake_gymnasium = ModuleType("gymnasium")
    fake_gymnasium.Env = object
    fake_imageio = ModuleType("imageio")
    fake_pil = ModuleType("PIL")
    fake_pil_image = ModuleType("PIL.Image")
    fake_pil_image_draw = ModuleType("PIL.ImageDraw")
    fake_pil_image_font = ModuleType("PIL.ImageFont")
    fake_venv = ModuleType("rlinf.envs.robocasa.venv")
    fake_venv.RobocasaSubprocEnv = object
    monkeypatch.delitem(
        sys.modules,
        "rlinf.envs.robocasa.robocasa_env",
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "gymnasium", fake_gymnasium)
    monkeypatch.setitem(sys.modules, "imageio", fake_imageio)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil_image)
    monkeypatch.setitem(sys.modules, "PIL.ImageDraw", fake_pil_image_draw)
    monkeypatch.setitem(sys.modules, "PIL.ImageFont", fake_pil_image_font)
    monkeypatch.setitem(sys.modules, "rlinf.envs.robocasa.venv", fake_venv)

    module = importlib.import_module("rlinf.envs.robocasa.robocasa_env")
    RobocasaEnv = module.RobocasaEnv

    env = RobocasaEnv.__new__(RobocasaEnv)
    env.cfg = OmegaConf.create(
        {
            "init_params": {
                "camera_widths": 224,
                "camera_heights": 224,
            },
            "robot_name": "PandaOmron",
            "image_space": ["observation/image"],
        }
    )
    env.num_envs = 1
    env.task_ids = np.array([0])
    env.task_names = ["CloseDrawer"]
    env.env_seeds = np.array([7])

    import_state = {"robocasa_imported": False}
    fake_env = object()

    fake_robosuite = ModuleType("robosuite")

    def _fake_make(**kwargs):
        assert import_state["robocasa_imported"] is True
        assert kwargs["env_name"] == "CloseDrawer"
        assert kwargs["robots"] == "PandaOmron"
        return fake_env

    fake_robosuite.make = _fake_make

    fake_controllers = ModuleType("robosuite.controllers")
    fake_controllers.load_composite_controller_config = lambda controller, robot: {
        "controller": controller,
        "robot": robot,
    }

    fake_robocasa = ModuleType("robocasa")

    monkeypatch.setitem(sys.modules, "robosuite", fake_robosuite)
    monkeypatch.setitem(sys.modules, "robosuite.controllers", fake_controllers)
    monkeypatch.setitem(sys.modules, "robocasa", fake_robocasa)

    original_import = builtins.__import__

    def _tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "robocasa":
            import_state["robocasa_imported"] = True
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _tracking_import)

    env_fn = env.get_env_fns()[0]

    assert env_fn() is fake_env


def test_robocasa_env_get_mujoco_diagnostics_delegates_to_vector_env() -> None:
    module = importlib.import_module("rlinf.envs.robocasa.robocasa_env")
    RobocasaEnv = module.RobocasaEnv
    env = RobocasaEnv.__new__(RobocasaEnv)

    class _VectorEnv:
        def get_mujoco_diagnostics(self, max_contacts, include_model_names):
            return [{"max_contacts": max_contacts, "names": include_model_names}]

    env.env = _VectorEnv()

    assert env.get_mujoco_diagnostics(max_contacts=7, include_model_names=True) == [
        {"max_contacts": 7, "names": True}
    ]


def test_robocasa_sync_chunk_step_repeats_actions_without_expanding_output(
    monkeypatch,
) -> None:
    module = importlib.import_module("rlinf.envs.robocasa.robocasa_env")
    RobocasaEnv = module.RobocasaEnv
    env = RobocasaEnv.__new__(RobocasaEnv)
    env.cfg = OmegaConf.create(
        {
            "action_repeat_per_chunk_step": 3,
            "max_episode_steps": 100,
        }
    )
    env.num_envs = 1
    env.auto_reset = False
    env.ignore_terminations = False
    env.chunk_step_mode = "sync_time_major"
    env._elapsed_steps = np.zeros(1, dtype=np.int32)
    recorded_actions = []

    def fake_step(actions, auto_reset=True):
        del auto_reset
        recorded_actions.append(np.asarray(actions).copy())
        env._elapsed_steps += 1
        obs = {"state": torch.tensor([[float(env._elapsed_steps[0])]])}
        reward = torch.tensor([float(env._elapsed_steps[0])])
        termination = torch.tensor([False])
        truncation = torch.tensor([False])
        infos = {
            "episode": {
                "return": torch.tensor([float(env._elapsed_steps[0])]),
                "episode_len": torch.tensor([env._elapsed_steps[0]]),
            }
        }
        return obs, reward, termination, truncation, infos

    monkeypatch.setattr(env, "step", fake_step)

    chunk_actions = np.array([[[1.0, 0.0], [2.0, 0.0]]], dtype=np.float32)
    obs_list, rewards, terminations, truncations, infos_list = env.chunk_step(
        chunk_actions,
        denoising_curvature=np.array([0.0]),
    )

    assert len(recorded_actions) == 6
    assert [actions[0, 0] for actions in recorded_actions] == [
        1.0,
        1.0,
        1.0,
        2.0,
        2.0,
        2.0,
    ]
    assert len(obs_list) == 2
    assert len(infos_list) == 2
    assert rewards.shape == (1, 2)
    assert terminations.shape == (1, 2)
    assert truncations.shape == (1, 2)
    assert rewards.tolist() == [[3.0, 6.0]]


def test_robocasa_subproc_chunk_step_repeats_without_expanding_output(
    monkeypatch,
) -> None:
    module = importlib.import_module("rlinf.envs.robocasa.venv")
    worker = module._worker

    class _Pipe:
        def __init__(self):
            self.sent = None
            self._closed = False

        def close(self):
            self._closed = True

        def recv(self):
            if self.sent is None:
                return [
                    "chunk_step",
                    (
                        np.array([[1.0], [2.0]], dtype=np.float32),
                        3,
                    ),
                ]
            raise EOFError

        def send(self, value):
            self.sent = value

    class _Wrapper:
        def data(self):
            return fake_env

    class _FakeEnv:
        def __init__(self):
            self.actions = []

        def step(self, action):
            self.actions.append(float(np.asarray(action)[0]))
            obs = {"state": np.array([len(self.actions)], dtype=np.float32)}
            return obs, 0.0, False, {}

    fake_env = _FakeEnv()
    parent = _Pipe()
    pipe = _Pipe()

    monkeypatch.setattr(module, "_apply_subproc_env_cpu_affinity", lambda idx: None)

    worker(parent, pipe, _Wrapper(), local_env_index=-1)

    assert fake_env.actions == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    assert len(pipe.sent) == 4
    assert len(pipe.sent[0]) == 2


def test_robocasa_subproc_chunk_step_records_repeated_step_timings(
    monkeypatch,
) -> None:
    module = importlib.import_module("rlinf.envs.robocasa.venv")
    worker = module._worker

    class _Pipe:
        def __init__(self):
            self.sent = None
            self._closed = False

        def close(self):
            self._closed = True

        def recv(self):
            if self.sent is None:
                return [
                    "chunk_step",
                    (
                        np.array([[1.0], [2.0]], dtype=np.float32),
                        2,
                    ),
                ]
            raise EOFError

        def send(self, value):
            self.sent = value

    class _Wrapper:
        def data(self):
            return fake_env

    class _FakeEnv:
        def __init__(self):
            self.actions = []

        def step(self, action):
            self.actions.append(float(np.asarray(action)[0]))
            obs = {"state": np.array([len(self.actions)], dtype=np.float32)}
            return obs, 0.0, False, {}

    fake_env = _FakeEnv()
    parent = _Pipe()
    pipe = _Pipe()

    monkeypatch.setattr(module, "_apply_subproc_env_cpu_affinity", lambda idx: None)

    worker(parent, pipe, _Wrapper(), local_env_index=3)

    info_returns = pipe.sent[-1]
    timings = [
        timing for info in info_returns for timing in info["robocasa_step_timings"]
    ]
    assert fake_env.actions == [1.0, 1.0, 2.0, 2.0]
    assert len(timings) == 4
    assert [timing["local_env"] for timing in timings] == [3, 3, 3, 3]
    assert [timing["chunk_action_index"] for timing in timings] == [0, 0, 1, 1]
    assert [timing["repeat_index"] for timing in timings] == [0, 1, 0, 1]
    assert all(timing["duration_s"] >= 0.0 for timing in timings)


def test_robocasa_vector_env_writes_step_timing_events(tmp_path) -> None:
    module = importlib.import_module("rlinf.envs.robocasa.venv")
    RobocasaSubprocEnv = module.RobocasaSubprocEnv
    env = RobocasaSubprocEnv.__new__(RobocasaSubprocEnv)
    env._sim_timestamp_context = {
        "output_dir": str(tmp_path),
        "rank": 2,
        "pid": 1234,
        "epoch": 0,
        "chunk_step": 1,
        "stage": 0,
        "stage_num": 1,
        "local_envs": 1,
    }
    env._sim_timestamp_file = None
    info_lists = [
        {
            "robocasa_step_timings": [
                {
                    "local_env": 0,
                    "chunk_action_index": 1,
                    "repeat_index": 2,
                    "duration_s": 0.25,
                    "wall_start_ns": 10,
                    "wall_end_ns": 20,
                }
            ]
        }
    ]

    env.record_robocasa_step_timing_events(info_lists, vector_step=3)

    records = [
        json.loads(line)
        for line in (tmp_path / "env_rank_2.jsonl").read_text().splitlines()
    ]
    assert records == [
        {
            "chunk_action_index": 1,
            "chunk_step": 1,
            "duration_s": 0.25,
            "epoch": 0,
            "event": "robocasa_env_step",
            "global_env": 2,
            "local_env": 0,
            "operation": "robocasa_step",
            "pid": 1234,
            "rank": 2,
            "repeat_index": 2,
            "stage": 0,
            "vector_step": 3,
            "wall_end_ns": 20,
            "wall_start_ns": 10,
        }
    ]
