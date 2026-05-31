import json
import time

import numpy as np
import pytest
import torch

from rlinf.envs.chunk_runner import (
    CHUNK_STEP_MODES,
    build_chunk_done_outputs,
    select_local_chunk_actions,
    split_env_indices,
)
from rlinf.envs.venv.venv import DummyVectorEnv, SubprocVectorEnv


class CountingEnv:
    def __init__(self, env_id: int):
        self.env_id = env_id
        self.local_step = 0

    def reset(self, **kwargs):
        self.local_step = 0
        return np.array([self.env_id, self.local_step])

    def step(self, action):
        self.local_step += 1
        action_value = int(np.asarray(action).reshape(-1)[0])
        obs = np.array([self.env_id, self.local_step, action_value])
        reward = float(self.env_id * 10 + action_value)
        done = action_value < 0
        info = {
            "env_id": self.env_id,
            "local_step": self.local_step,
            "action": action_value,
        }
        return obs, reward, done, info

    def close(self):
        pass


class BarrierEnv:
    def __init__(self, env_id: int, arrivals, num_envs: int, timeout_s: float):
        self.env_id = env_id
        self.arrivals = arrivals
        self.num_envs = num_envs
        self.timeout_s = timeout_s
        self.local_step = 0

    def reset(self, **kwargs):
        self.local_step = 0
        return np.array([self.env_id, self.local_step])

    def step(self, action):
        self.local_step += 1
        step_id = self.local_step
        offset = (step_id - 1) * self.num_envs
        with self.arrivals.get_lock():
            self.arrivals[offset + self.env_id] = 1

        deadline = time.monotonic() + self.timeout_s
        barrier_timeout = False
        while True:
            with self.arrivals.get_lock():
                arrived = sum(
                    self.arrivals[offset + env_id]
                    for env_id in range(self.num_envs)
                )
            if arrived == self.num_envs:
                break
            if time.monotonic() >= deadline:
                barrier_timeout = True
                break
            time.sleep(0.01)

        action_value = int(np.asarray(action).reshape(-1)[0])
        obs = np.array([self.env_id, step_id, action_value])
        return obs, float(action_value), False, {"barrier_timeout": barrier_timeout}

    def close(self):
        pass


class SleepEnv(CountingEnv):
    def __init__(self, env_id: int, sleep_s: float):
        super().__init__(env_id)
        self.sleep_s = sleep_s

    def step(self, action):
        time.sleep(self.sleep_s)
        return super().step(action)


def test_split_env_indices_and_select_local_actions():
    shards = split_env_indices(5, 2)

    assert [shard.tolist() for shard in shards] == [[0, 1, 2], [3, 4]]

    actions = torch.arange(5 * 3 * 2).reshape(5, 3, 2)
    local_actions = select_local_chunk_actions(actions, shards[1])

    assert torch.equal(local_actions, actions[3:5])


def test_split_env_indices_rejects_too_many_shards():
    with pytest.raises(ValueError, match="must be <= num_envs"):
        split_env_indices(2, 3)


def test_latency_balanced_pair_mode_is_registered():
    assert "latency_balanced_pair" in CHUNK_STEP_MODES


def test_build_chunk_done_outputs_collapses_to_last_step():
    raw_terminations = torch.tensor(
        [[False, True, False], [False, False, False]]
    )
    raw_truncations = torch.tensor([[False, False, False], [True, False, True]])

    (
        chunk_terminations,
        chunk_truncations,
        past_terminations,
        past_truncations,
        past_dones,
    ) = build_chunk_done_outputs(raw_terminations, raw_truncations)

    assert past_terminations.tolist() == [True, False]
    assert past_truncations.tolist() == [False, True]
    assert past_dones.tolist() == [True, True]
    assert chunk_terminations.tolist() == [
        [False, False, True],
        [False, False, False],
    ]
    assert chunk_truncations.tolist() == [
        [False, False, False],
        [False, False, True],
    ]


def test_dummy_vector_env_chunk_step_restores_step_major_shape():
    env = DummyVectorEnv([lambda i=i: CountingEnv(i) for i in range(2)])
    chunk_actions = np.array([[[1], [2], [3]], [[4], [5], [6]]])

    obs_list, rewards_list, dones_list, infos_list = env.chunk_step(chunk_actions)

    assert len(obs_list) == 3
    assert obs_list[0].tolist() == [[0, 1, 1], [1, 1, 4]]
    assert obs_list[2].tolist() == [[0, 3, 3], [1, 3, 6]]
    assert rewards_list[1].tolist() == [2.0, 15.0]
    assert dones_list[0].tolist() == [False, False]
    assert infos_list[2][0]["local_step"] == 3
    assert infos_list[2][1]["action"] == 6

    env.close()


def test_vector_env_chunk_step_logs_subenv_timestamps(tmp_path):
    env = DummyVectorEnv([lambda i=i: CountingEnv(i) for i in range(2)])
    chunk_actions = np.array([[[1], [2]], [[3], [4]]])
    output_dir = tmp_path / "env_sim_timestamps"
    env.set_sim_timestamp_context(
        {
            "output_dir": str(output_dir),
            "rank": 3,
            "pid": 12345,
            "epoch": 5,
            "chunk_step": 7,
            "stage": 1,
            "stage_num": 2,
            "local_envs": 2,
        }
    )

    try:
        env.chunk_step(chunk_actions)
    finally:
        env.set_sim_timestamp_context(None)
        env.close()

    events = [
        json.loads(line)
        for line in (output_dir / "env_rank_3.jsonl").read_text().splitlines()
    ]
    starts = [event for event in events if event["event"] == "subenv_start"]
    ends = [event for event in events if event["event"] == "subenv_end"]

    assert [event["local_env"] for event in starts] == [0, 1]
    assert [event["global_env"] for event in starts] == [14, 15]
    assert all(event["operation"] == "chunk_step" for event in starts + ends)
    assert all(event["action_chunk_steps"] == 2 for event in starts + ends)
    assert all(event["vector_step"] == 0 for event in starts + ends)
    assert all(event["duration_s"] >= 0.0 for event in ends)
    profile = env.get_last_chunk_profile()
    assert profile is not None
    assert profile["operation"] == "chunk_step"
    assert profile["env_count"] == 2
    assert profile["dispatch_s"] >= 0.0
    assert profile["wait_recv_s"] >= 0.0
    assert profile["stack_s"] >= 0.0


def test_subproc_vector_env_chunk_step_dispatches_before_recv():
    mp = pytest.importorskip("multiprocessing")
    num_envs = 2
    chunk_steps = 2
    arrivals = mp.Array("b", num_envs * chunk_steps, lock=True)
    timeout_s = 1.0
    env = SubprocVectorEnv(
        [
            lambda i=i: BarrierEnv(i, arrivals, num_envs, timeout_s)
            for i in range(num_envs)
        ]
    )
    chunk_actions = np.array([[[1], [2]], [[3], [4]]])

    try:
        obs_list, _rewards_list, _dones_list, infos_list = env.chunk_step(
            chunk_actions
        )
    finally:
        env.close()

    assert obs_list[0].tolist() == [[0, 1, 1], [1, 1, 3]]
    assert obs_list[1].tolist() == [[0, 2, 2], [1, 2, 4]]
    assert all(
        not info["barrier_timeout"]
        for step_infos in infos_list
        for info in step_infos
    )


def test_subproc_vector_env_latency_balanced_pair_restores_step_major_shape():
    env = SubprocVectorEnv([lambda i=i: CountingEnv(i) for i in range(4)])
    chunk_actions = np.array([[[1], [2]], [[3], [4]], [[5], [6]], [[7], [8]]])

    try:
        obs_list, rewards_list, dones_list, infos_list = (
            env.latency_balanced_pair_chunk_step(
                chunk_actions,
                envs_per_core=2,
                ema_alpha=0.5,
                dynamic_affinity=False,
            )
        )
    finally:
        env.close()

    assert len(obs_list) == 2
    assert obs_list[0].tolist() == [[0, 1, 1], [1, 1, 3], [2, 1, 5], [3, 1, 7]]
    assert obs_list[1].tolist() == [[0, 2, 2], [1, 2, 4], [2, 2, 6], [3, 2, 8]]
    assert rewards_list[0].tolist() == [1.0, 13.0, 25.0, 37.0]
    assert rewards_list[1].tolist() == [2.0, 14.0, 26.0, 38.0]
    assert dones_list[0].tolist() == [False, False, False, False]
    assert infos_list[1][3]["local_step"] == 2
    assert infos_list[1][3]["action"] == 8


def test_latency_balanced_pair_runs_slot_local_pipeline(tmp_path):
    env = SubprocVectorEnv(
        [
            lambda: SleepEnv(0, 0.02),
            lambda: SleepEnv(1, 0.02),
            lambda: SleepEnv(2, 0.30),
            lambda: SleepEnv(3, 0.02),
        ]
    )
    env._balanced_pair_predicted_latency_s = [10.0, 1.0, 8.0, 2.0]
    chunk_actions = np.array([[[1]], [[2]], [[3]], [[4]]])
    output_dir = tmp_path / "env_sim_timestamps"
    env.set_sim_timestamp_context(
        {
            "output_dir": str(output_dir),
            "rank": 0,
            "pid": 12345,
            "epoch": 0,
            "chunk_step": 0,
            "stage": 0,
            "stage_num": 1,
            "local_envs": 4,
        }
    )

    try:
        env.latency_balanced_pair_chunk_step(
            chunk_actions,
            envs_per_core=2,
            ema_alpha=0.5,
            dynamic_affinity=False,
        )
    finally:
        env.set_sim_timestamp_context(None)
        env.close()

    events = [
        json.loads(line)
        for line in (output_dir / "env_rank_0.jsonl").read_text().splitlines()
    ]
    starts = {
        event["local_env"]: event
        for event in events
        if event["event"] == "subenv_start"
    }
    ends = {
        event["local_env"]: event
        for event in events
        if event["event"] == "subenv_end"
    }

    assert starts[1]["pair_slot"] == starts[0]["pair_slot"]
    assert starts[1]["pair_offset"] == 1
    assert starts[2]["pair_slot"] != starts[0]["pair_slot"]
    assert starts[1]["wall_ns"] < ends[2]["wall_ns"]
    profile = env.get_last_chunk_profile()
    assert profile is not None
    assert profile["operation"] == "latency_balanced_pair_chunk_step"
    assert profile["env_count"] == 4
    assert profile["envs_per_core"] == 2
    assert profile["slot_count"] == 2
    assert profile["wait_call_s"] >= 0.0
    assert profile["recv_call_s"] >= 0.0


def test_latency_balanced_pair_rejects_invalid_group_size():
    env = DummyVectorEnv([lambda i=i: CountingEnv(i) for i in range(2)])
    chunk_actions = np.array([[[1]], [[2]]])

    try:
        with pytest.raises(ValueError, match="envs_per_core"):
            env.latency_balanced_pair_chunk_step(chunk_actions, envs_per_core=3)
    finally:
        env.close()


def test_latency_balanced_pair_groups_slow_and_fast_envs():
    env = DummyVectorEnv([lambda i=i: CountingEnv(i) for i in range(4)])
    env._balanced_pair_predicted_latency_s = [10.0, 1.0, 8.0, 2.0]

    try:
        groups = env._build_latency_balanced_groups([0, 1, 2, 3], envs_per_core=2)
    finally:
        env.close()

    assert groups == [[0, 1], [2, 3]]
