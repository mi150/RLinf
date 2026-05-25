from dataclasses import asdict
from unittest import mock
from unittest.mock import Mock

import pytest

from examples.embodiment.train_embodied_agent import _get_resource_bindings
from rlinf.scheduler.placement.placement import Placement
from rlinf.scheduler.resource_pool.bindings import (
    RESOURCE_BINDING_ENV,
    WorkerResourceBinding,
)
from rlinf.scheduler.resource_pool.pool import FineGrainedResourcePool
from rlinf.scheduler.worker.worker import Worker
from rlinf.scheduler.worker.worker_group import WorkerGroup


class DummyWorker:
    pass


class DummyPlacementStrategy:
    def __init__(self, placements: list[Placement] | None = None):
        self._placements = placements or [
            Placement(
                rank=0,
                cluster_node_rank=0,
                placement_node_rank=0,
                local_accelerator_rank=-1,
                accelerator_type="NO_ACCEL",
                local_rank=0,
                local_world_size=1,
                visible_accelerators=[],
                isolate_accelerator=True,
                local_hardware_ranks=[],
                node_group_label="node",
            )
        ]

    def get_placement(self, cluster, isolate_accelerator=True):
        return self._placements


def _make_cluster() -> Mock:
    cluster = Mock()
    cluster.get_node_ip.return_value = "127.0.0.1"
    cluster.get_node_info.return_value.accelerator_type = "NO_ACCEL"
    cluster.get_node_info.return_value.accelerator_model = ""
    cluster.get_node_group.return_value.get_node_env_vars.return_value = {}
    cluster.get_node_group.return_value.get_node_python_interpreter_path.return_value = None
    cluster.allocate.return_value = object()
    return cluster


def _make_binding(
    *,
    rank: int = 0,
    cluster_node_rank: int = 0,
    node_group_label: str = "node",
) -> WorkerResourceBinding:
    return WorkerResourceBinding(
        component="env",
        rank=rank,
        cluster_node_rank=cluster_node_rank,
        node_group_label=node_group_label,
    )


def _attach_ready_noop(group: WorkerGroup) -> None:
    group._is_ready = lambda: None


def test_nvidia_visible_devices_allows_mig_uuid(monkeypatch) -> None:
    from rlinf.scheduler.hardware.accelerators.nvidia_gpu import NvidiaGPUManager

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-abc")
    assert NvidiaGPUManager.get_visible_devices() == []


def test_nvidia_visible_devices_rejects_mixed_gpu_ids_and_mig_uuid(
    monkeypatch,
) -> None:
    from rlinf.scheduler.hardware.accelerators.nvidia_gpu import NvidiaGPUManager

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,MIG-abc")
    with pytest.raises(ValueError, match="MIG"):
        NvidiaGPUManager.get_visible_devices()


def test_worker_group_injects_resource_binding_env() -> None:
    cluster = _make_cluster()
    binding = _make_binding()
    group = WorkerGroup(DummyWorker, args=(), kwargs={})

    with mock.patch.object(WorkerGroup, "_attach_cls_func", _attach_ready_noop):
        group.launch(
            cluster=cluster,
            placement_strategy=DummyPlacementStrategy(),
            resource_bindings=[binding],
        )

    env_vars = cluster.allocate.call_args.kwargs["env_vars"]
    assert RESOURCE_BINDING_ENV in env_vars


def test_worker_group_rejects_duplicate_resource_binding_ranks() -> None:
    cluster = _make_cluster()
    binding = _make_binding()
    duplicate = _make_binding()
    group = WorkerGroup(DummyWorker, args=(), kwargs={})

    with (
        mock.patch.object(WorkerGroup, "_attach_cls_func", _attach_ready_noop),
        pytest.raises(AssertionError, match="Duplicate resource binding ranks"),
    ):
        group.launch(
            cluster=cluster,
            placement_strategy=DummyPlacementStrategy(),
            resource_bindings=[binding, duplicate],
        )


def test_worker_group_rejects_resource_binding_rank_outside_placement() -> None:
    cluster = _make_cluster()
    binding = _make_binding(rank=1)
    group = WorkerGroup(DummyWorker, args=(), kwargs={})

    with (
        mock.patch.object(WorkerGroup, "_attach_cls_func", _attach_ready_noop),
        pytest.raises(AssertionError, match="subset of placement ranks"),
    ):
        group.launch(
            cluster=cluster,
            placement_strategy=DummyPlacementStrategy(),
            resource_bindings=[binding],
        )


def test_worker_group_rejects_resource_binding_node_mismatch() -> None:
    cluster = _make_cluster()
    binding = _make_binding(cluster_node_rank=1)
    group = WorkerGroup(DummyWorker, args=(), kwargs={})

    with (
        mock.patch.object(WorkerGroup, "_attach_cls_func", _attach_ready_noop),
        pytest.raises(AssertionError, match="targets cluster node"),
    ):
        group.launch(
            cluster=cluster,
            placement_strategy=DummyPlacementStrategy(),
            resource_bindings=[binding],
        )


def test_worker_group_rejects_resource_binding_node_group_mismatch() -> None:
    cluster = _make_cluster()
    binding = _make_binding(node_group_label="other")
    group = WorkerGroup(DummyWorker, args=(), kwargs={})

    with (
        mock.patch.object(WorkerGroup, "_attach_cls_func", _attach_ready_noop),
        pytest.raises(AssertionError, match="targets node group"),
    ):
        group.launch(
            cluster=cluster,
            placement_strategy=DummyPlacementStrategy(),
            resource_bindings=[binding],
        )


def test_worker_parses_resource_binding_from_env() -> None:
    binding = _make_binding(rank=2)
    worker = object.__new__(Worker)

    with mock.patch.dict("os.environ", {RESOURCE_BINDING_ENV: binding.to_json()}):
        worker._setup_resource_binding()

    assert worker.resource_binding == binding
    assert worker._resource_binding_dict == asdict(binding)


def test_get_resource_bindings_returns_none_when_pool_disabled() -> None:
    assert _get_resource_bindings(FineGrainedResourcePool.disabled(), "actor") is None
