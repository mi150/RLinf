import pytest
from omegaconf import OmegaConf

from rlinf.scheduler.resource_pool.bindings import CpuBinding, WorkerResourceBinding
from rlinf.workers.env.env_worker import EnvWorker


def _make_worker_with_binding(env_type: str, cpu: CpuBinding) -> EnvWorker:
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "env": {
                "train": {"env_type": env_type},
                "eval": {"env_type": env_type},
            },
            "runner": {"only_eval": False},
        }
    )
    worker.only_eval = False
    worker._resource_binding = WorkerResourceBinding(
        component="env",
        rank=0,
        cluster_node_rank=0,
        node_group_label="node",
        cpu=cpu,
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
