import json
import os
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
                    self.arrivals[offset + env_id] for env_id in range(self.num_envs)
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


class ScriptedWorker:
    ready_queue = []

    def __init__(self, env_id: int):
        self.env_id = env_id
        self.result = (
            [np.array([env_id, 1])],
            [float(env_id)],
            [False],
            [{"env_id": env_id}],
        )
        self.affinity_calls = []

    def send_chunk_step(self, _chunk_action):
        ScriptedWorker.ready_queue.append(self)

    def recv(self):
        return self.result

    def set_cpu_affinity(self, cpus):
        self.affinity_calls.append(tuple(cpus))

    @staticmethod
    def wait(workers, _wait_num, _timeout=None):
        for worker in reversed(ScriptedWorker.ready_queue):
            if worker in workers:
                ScriptedWorker.ready_queue.remove(worker)
                return [worker]
        return []


class FailingRecvWorker(ScriptedWorker):
    def recv(self):
        if self.env_id == 0:
            raise RuntimeError("scripted recv failure")
        return super().recv()


class PidBackedScriptedWorker(ScriptedWorker):
    def __init__(self, env_id: int, pid: int):
        super().__init__(env_id)
        self.process = type("FakeProcess", (), {"pid": pid})()


class OrderedCompletionWorker(ScriptedWorker):
    completion_order = []
    send_log = []

    def send_chunk_step(self, _chunk_action):
        OrderedCompletionWorker.send_log.append(self.env_id)

    @staticmethod
    def wait(workers, _wait_num, _timeout=None):
        worker_ids = {worker.env_id: worker for worker in workers}
        for env_id in list(OrderedCompletionWorker.completion_order):
            if env_id in worker_ids:
                OrderedCompletionWorker.completion_order.remove(env_id)
                return [worker_ids[env_id]]
        return []


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
    raw_terminations = torch.tensor([[False, True, False], [False, False, False]])
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


def test_vector_env_chunk_timestamp_events_are_json_serializable(tmp_path):
    env = DummyVectorEnv([lambda i=i: CountingEnv(i) for i in range(2)])
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
        fake_worker = type(
            "FakeWorker",
            (),
            {
                "process": type("FakeProcess", (), {"pid": 4321})(),
                "_cpu_affinity": (7, 8),
            },
        )()
        env._write_subenv_timestamp_event(
            "end",
            local_env_id=0,
            worker=fake_worker,
            operation="chunk_step",
            extra={
                "cpu_affinity": np.array([1, 2]),
                "nested": {"items": [np.array([3]), np.int64(4)]},
            },
        )
    finally:
        env.set_sim_timestamp_context(None)
        env.close()

    line = (output_dir / "env_rank_3.jsonl").read_text().strip()
    record = json.loads(line)
    assert record["cpu_affinity"] == [1, 2]
    assert record["nested"]["items"][0] == [3]
    assert record["nested"]["items"][1] == 4


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
        obs_list, _rewards_list, _dones_list, infos_list = env.chunk_step(chunk_actions)
    finally:
        env.close()

    assert obs_list[0].tolist() == [[0, 1, 1], [1, 1, 3]]
    assert obs_list[1].tolist() == [[0, 2, 2], [1, 2, 4]]
    assert all(
        not info["barrier_timeout"] for step_infos in infos_list for info in step_infos
    )


def test_subproc_vector_env_latency_balanced_pair_restores_step_major_shape():
    env = SubprocVectorEnv([lambda i=i: CountingEnv(i) for i in range(2)])
    env._env_cpu_core_groups = ((0,), (1,))
    chunk_actions = np.array([[[1], [2]], [[3], [4]]])

    try:
        obs_list, rewards_list, dones_list, infos_list = (
            env.latency_balanced_pair_chunk_step(
                chunk_actions,
                envs_per_core=1,
                ema_alpha=0.5,
                dynamic_affinity=True,
                core_donation_enabled=True,
            )
        )
    finally:
        env.close()

    assert len(obs_list) == 2
    assert obs_list[0].tolist() == [[0, 1, 1], [1, 1, 3]]
    assert obs_list[1].tolist() == [[0, 2, 2], [1, 2, 4]]
    assert rewards_list[0].tolist() == [1.0, 13.0]
    assert rewards_list[1].tolist() == [2.0, 14.0]
    assert dones_list[0].tolist() == [False, False]
    assert infos_list[1][1]["local_step"] == 2
    assert infos_list[1][1]["action"] == 4


def test_latency_balanced_pair_runs_core_donation_v2_path(tmp_path):
    env = SubprocVectorEnv(
        [
            lambda: SleepEnv(0, 0.02),
            lambda: SleepEnv(1, 0.02),
            lambda: SleepEnv(2, 0.30),
            lambda: SleepEnv(3, 0.02),
        ]
    )
    env._env_cpu_core_groups = ((0,), (1,), (2,), (3,))
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
            envs_per_core=1,
            ema_alpha=0.5,
            dynamic_affinity=True,
            core_donation_enabled=True,
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
        event["local_env"]: event for event in events if event["event"] == "subenv_end"
    }

    assert all(starts[env_id]["pair_offset"] == 0 for env_id in range(4))
    assert {starts[env_id]["pair_slot"] for env_id in range(4)} == {0, 1, 2, 3}
    assert all(
        event["operation"] == "latency_balanced_pair_chunk_step"
        for event in ends.values()
    )
    profile = env.get_last_chunk_profile()
    assert profile is not None
    assert profile["operation"] == "latency_balanced_pair_chunk_step"
    assert profile["env_count"] == 4
    assert profile["envs_per_core"] == 1
    assert profile["slot_count"] == 4
    assert profile["wait_call_s"] >= 0.0
    assert profile["recv_call_s"] >= 0.0


def test_latency_balanced_pair_donates_finished_core_groups():
    env = SubprocVectorEnv.__new__(SubprocVectorEnv)
    env.is_async = False
    env.env_num = 2
    env.workers = [ScriptedWorker(0), ScriptedWorker(1)]
    env.worker_class = ScriptedWorker
    env.timeout = None
    env.is_closed = False
    env._sim_vector_step_index = 0
    env._sim_timestamp_context = None
    env._last_chunk_profile = None
    env._balanced_pair_predicted_latency_s = [10.0, 1.0]
    env._env_cpu_core_groups = ((0,), (1,))
    env._balanced_pair_logged = True
    ScriptedWorker.ready_queue = []
    chunk_actions = np.array([[[1]], [[2]]])

    env.latency_balanced_pair_chunk_step(
        chunk_actions,
        envs_per_core=1,
        ema_alpha=0.5,
        dynamic_affinity=True,
        core_donation_enabled=True,
        core_donation_max_extra_groups=1,
    )

    assert env.workers[0].affinity_calls == [(0,), (0, 1), (0,)]
    assert env.workers[1].affinity_calls == [(1,)]
    profile = env.get_last_chunk_profile()
    assert profile is not None
    assert profile["core_donation_count"] == 1
    assert profile["core_donation_restore_s"] >= 0.0


def test_latency_balanced_pair_restores_donated_cores_on_error():
    env = SubprocVectorEnv.__new__(SubprocVectorEnv)
    env.is_async = False
    env.env_num = 2
    env.workers = [FailingRecvWorker(0), FailingRecvWorker(1)]
    env.worker_class = FailingRecvWorker
    env.timeout = None
    env.is_closed = False
    env._sim_vector_step_index = 0
    env._sim_timestamp_context = None
    env._last_chunk_profile = None
    env._balanced_pair_predicted_latency_s = [10.0, 1.0]
    env._env_cpu_core_groups = ((0,), (1,))
    env._balanced_pair_logged = True
    ScriptedWorker.ready_queue = []
    chunk_actions = np.array([[[1]], [[2]]])

    with pytest.raises(RuntimeError, match="scripted recv failure"):
        env.latency_balanced_pair_chunk_step(
            chunk_actions,
            envs_per_core=1,
            ema_alpha=0.5,
            dynamic_affinity=True,
            core_donation_enabled=True,
            core_donation_max_extra_groups=1,
        )

    assert env.workers[0].affinity_calls == [(0,), (0, 1), (0,)]


def test_latency_balanced_pair_donation_uses_pid_affinity_without_pipe(monkeypatch):
    affinity_calls = []

    def fake_sched_setaffinity(pid, cpus):
        affinity_calls.append((pid, tuple(sorted(cpus))))

    monkeypatch.setattr(os, "sched_setaffinity", fake_sched_setaffinity)

    env = SubprocVectorEnv.__new__(SubprocVectorEnv)
    env.is_async = False
    env.env_num = 2
    env.workers = [PidBackedScriptedWorker(0, 1000), PidBackedScriptedWorker(1, 1001)]
    env.worker_class = PidBackedScriptedWorker
    env.timeout = None
    env.is_closed = False
    env._sim_vector_step_index = 0
    env._sim_timestamp_context = None
    env._last_chunk_profile = None
    env._balanced_pair_predicted_latency_s = [10.0, 1.0]
    env._env_cpu_core_groups = ((0,), (1,))
    env._balanced_pair_logged = True
    ScriptedWorker.ready_queue = []
    chunk_actions = np.array([[[1]], [[2]]])

    env.latency_balanced_pair_chunk_step(
        chunk_actions,
        envs_per_core=1,
        ema_alpha=0.5,
        dynamic_affinity=True,
        core_donation_enabled=True,
        core_donation_max_extra_groups=1,
    )

    assert env.workers[0].affinity_calls == [(0,)]
    assert env.workers[1].affinity_calls == [(1,)]
    assert affinity_calls == [(1000, (0, 1)), (1000, (0,))]


def test_latency_bin_packing_dispatches_k_bins_with_serial_envs():
    env = SubprocVectorEnv.__new__(SubprocVectorEnv)
    env.is_async = False
    env.env_num = 8
    env.workers = [OrderedCompletionWorker(i) for i in range(8)]
    env.worker_class = OrderedCompletionWorker
    env.timeout = None
    env.is_closed = False
    env._sim_vector_step_index = 0
    env._sim_timestamp_context = None
    env._last_chunk_profile = None
    env._bin_packing_predicted_latency_s = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    env._env_cpu_core_groups = ((0,), (1,), (2,), (3,))
    env._bin_packing_logged = True
    OrderedCompletionWorker.completion_order = [0, 1, 2, 3, 7, 6, 5, 4]
    OrderedCompletionWorker.send_log = []
    chunk_actions = np.arange(8).reshape(8, 1, 1)

    obs_list, rewards_list, dones_list, infos_list = env.latency_bin_packing_chunk_step(
        chunk_actions,
        bin_count=4,
        ema_alpha=0.5,
    )

    assert OrderedCompletionWorker.send_log[:4] == [0, 1, 2, 3]
    assert OrderedCompletionWorker.send_log[4:] == [7, 6, 5, 4]
    assert env.workers[0].affinity_calls == [(0,)]
    assert env.workers[7].affinity_calls == [(0,)]
    assert env.workers[1].affinity_calls == [(1,)]
    assert env.workers[6].affinity_calls == [(1,)]
    assert obs_list[0].tolist() == [[env_id, 1] for env_id in range(8)]
    assert rewards_list[0].tolist() == [float(env_id) for env_id in range(8)]
    assert dones_list[0].tolist() == [False for _ in range(8)]
    assert infos_list[0][4]["env_id"] == 4
    profile = env.get_last_chunk_profile()
    assert profile is not None
    assert profile["operation"] == "latency_bin_packing_chunk_step"
    assert profile["bin_count"] == 4
    assert profile["bin_groups"] == [[0, 7], [1, 6], [2, 5], [3, 4]]
    assert profile["bin_loads_predicted_s"] == [9.0, 9.0, 9.0, 9.0]


def test_latency_balanced_pair_rejects_non_core_donation_v2_modes():
    env = DummyVectorEnv([lambda i=i: CountingEnv(i) for i in range(2)])
    chunk_actions = np.array([[[1]], [[2]]])

    try:
        with pytest.raises(ValueError, match="core donation v2"):
            env.latency_balanced_pair_chunk_step(chunk_actions, envs_per_core=3)
        with pytest.raises(ValueError, match="core donation v2"):
            env.latency_balanced_pair_chunk_step(
                chunk_actions,
                envs_per_core=1,
                dynamic_affinity=False,
                core_donation_enabled=True,
            )
        with pytest.raises(ValueError, match="core donation v2"):
            env.latency_balanced_pair_chunk_step(
                chunk_actions,
                envs_per_core=1,
                dynamic_affinity=True,
                core_donation_enabled=False,
            )
        with pytest.raises(ValueError, match="requires per-env CPU core groups"):
            env.latency_balanced_pair_chunk_step(
                chunk_actions,
                envs_per_core=1,
                dynamic_affinity=True,
                core_donation_enabled=True,
            )
    finally:
        env.close()
