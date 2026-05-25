import os

import pytest
from omegaconf import OmegaConf

from rlinf.envs.venv.venv import _apply_subproc_env_cpu_affinity
from rlinf.scheduler.resource_pool.bindings import (
    ENV_CPU_CORE_GROUPS_ENV,
    CpuBinding,
    WorkerResourceBinding,
)
from rlinf.workers.env.env_worker import EnvWorker


def _make_worker_with_binding(
    env_type: str = "libero",
    cpu: CpuBinding | None = None,
    *,
    train_env_type: str | None = None,
    eval_env_type: str | None = None,
    only_eval: bool = False,
    enable_eval: bool = False,
    stage_num: int = 1,
) -> EnvWorker:
    worker = object.__new__(EnvWorker)
    train_env_type = train_env_type or env_type
    eval_env_type = eval_env_type or env_type
    worker.cfg = OmegaConf.create(
        {
            "env": {
                "train": {"env_type": train_env_type},
                "eval": {"env_type": eval_env_type},
            },
            "runner": {"only_eval": only_eval},
        }
    )
    worker.only_eval = only_eval
    worker.enable_eval = enable_eval
    worker.stage_num = stage_num
    worker._rank = 0
    worker._world_size = 1
    worker._worker_info = None
    worker._resource_binding = WorkerResourceBinding(
        component="env",
        rank=0,
        cluster_node_rank=0,
        node_group_label="node",
        cpu=cpu or CpuBinding(),
    )
    return worker


def test_env_worker_applies_process_cpu_affinity(monkeypatch) -> None:
    worker = _make_worker_with_binding(
        "maniskill", CpuBinding(process_cpu_cores=(0, 2))
    )
    captured = {}

    def fake_apply(cpus):
        captured["cpus"] = cpus

    monkeypatch.setattr(
        "rlinf.workers.env.env_worker.apply_process_cpu_affinity",
        fake_apply,
    )

    worker._apply_resource_pool_cpu_affinity()

    assert captured == {"cpus": (0, 2)}


def test_env_worker_rejects_unsupported_per_env_backend() -> None:
    worker = _make_worker_with_binding(
        "maniskill",
        CpuBinding(process_cpu_cores=(0, 1), env_cpu_core_groups=((0,), (1,))),
    )

    with pytest.raises(ValueError, match="per-env CPU binding"):
        worker._validate_env_resource_binding_supported()


def test_env_worker_accepts_supported_per_env_backend() -> None:
    worker = _make_worker_with_binding(
        "libero",
        CpuBinding(process_cpu_cores=(0, 1), env_cpu_core_groups=((0,), (1,))),
    )
    worker._validate_env_resource_binding_supported()


def test_env_worker_ignores_disabled_eval_backend_for_validation() -> None:
    worker = _make_worker_with_binding(
        train_env_type="libero",
        eval_env_type="maniskill",
        cpu=CpuBinding(process_cpu_cores=(0, 1), env_cpu_core_groups=((0,), (1,))),
        only_eval=False,
        enable_eval=False,
    )

    worker._validate_env_resource_binding_supported()


def test_env_worker_validates_enabled_eval_backend() -> None:
    worker = _make_worker_with_binding(
        train_env_type="libero",
        eval_env_type="maniskill",
        cpu=CpuBinding(process_cpu_cores=(0, 1), env_cpu_core_groups=((0,), (1,))),
        only_eval=False,
        enable_eval=True,
    )

    with pytest.raises(ValueError, match="per-env CPU binding"):
        worker._validate_env_resource_binding_supported()


def test_stage_env_cpu_core_groups_slices_and_restores_env_var(monkeypatch) -> None:
    worker = _make_worker_with_binding(
        cpu=CpuBinding(env_cpu_core_groups=((0,), (1,), (2,), (3,))),
    )
    monkeypatch.setenv(ENV_CPU_CORE_GROUPS_ENV, "old")

    with worker._stage_env_cpu_core_groups(stage_id=1, num_envs_per_stage=2):
        assert os.environ[ENV_CPU_CORE_GROUPS_ENV] == "2;3"

    assert os.environ[ENV_CPU_CORE_GROUPS_ENV] == "old"


def test_stage_env_cpu_core_groups_removes_env_var_after_exception(monkeypatch) -> None:
    worker = _make_worker_with_binding(
        cpu=CpuBinding(env_cpu_core_groups=((0,), (1,), (2,), (3,))),
    )
    monkeypatch.delenv(ENV_CPU_CORE_GROUPS_ENV, raising=False)

    with pytest.raises(RuntimeError, match="boom"):
        with worker._stage_env_cpu_core_groups(stage_id=0, num_envs_per_stage=2):
            assert os.environ[ENV_CPU_CORE_GROUPS_ENV] == "0;1"
            raise RuntimeError("boom")

    assert ENV_CPU_CORE_GROUPS_ENV not in os.environ


def test_setup_env_and_wrappers_injects_stage_cpu_groups() -> None:
    seen_cpu_groups = []

    class FakeEnv:
        def __init__(
            self,
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            worker_info,
        ) -> None:
            seen_cpu_groups.append(os.environ.get(ENV_CPU_CORE_GROUPS_ENV))

    worker = _make_worker_with_binding(
        cpu=CpuBinding(env_cpu_core_groups=((0,), (1,), (2,), (3,))),
        stage_num=2,
    )
    env_cfg = OmegaConf.create({"video_cfg": {"save_video": False}})

    envs = worker._setup_env_and_wrappers(
        FakeEnv,
        env_cfg,
        num_envs_per_stage=2,
    )

    assert seen_cpu_groups == ["0;1", "2;3"]
    assert len(envs) == 2


def test_subproc_env_affinity_uses_local_env_index(monkeypatch) -> None:
    captured = {}

    def fake_apply(cpus):
        captured["cpus"] = cpus

    monkeypatch.setattr(
        "rlinf.envs.venv.venv.apply_process_cpu_affinity",
        fake_apply,
    )
    monkeypatch.setenv("RLINF_ENV_CPU_CORE_GROUPS", "0;1,2;3")

    _apply_subproc_env_cpu_affinity(local_env_index=1)

    assert captured == {"cpus": (1, 2)}
