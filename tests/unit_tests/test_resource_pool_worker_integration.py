from dataclasses import asdict
from unittest import mock
from unittest.mock import Mock

from rlinf.scheduler.placement.placement import Placement
from rlinf.scheduler.resource_pool.bindings import (
    RESOURCE_BINDING_ENV,
    WorkerResourceBinding,
)
from rlinf.scheduler.worker.worker import Worker
from rlinf.scheduler.worker.worker_group import WorkerGroup


class DummyWorker:
    pass


class DummyPlacementStrategy:
    def get_placement(self, cluster, isolate_accelerator=True):
        return [
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


def test_worker_group_injects_resource_binding_env() -> None:
    cluster = Mock()
    cluster.get_node_ip.return_value = "127.0.0.1"
    cluster.get_node_info.return_value.accelerator_type = "NO_ACCEL"
    cluster.get_node_info.return_value.accelerator_model = ""
    cluster.get_node_group.return_value.get_node_env_vars.return_value = {}
    cluster.get_node_group.return_value.get_node_python_interpreter_path.return_value = None
    cluster.allocate.return_value = object()

    binding = WorkerResourceBinding(
        component="env",
        rank=0,
        cluster_node_rank=0,
        node_group_label="node",
    )
    group = WorkerGroup(DummyWorker, args=(), kwargs={})
    group._cluster = cluster
    group._placement_strategy = DummyPlacementStrategy()
    group._isolate_gpu = True
    group._catch_system_failure = False
    group._max_concurrency = None
    group._disable_distributed_log = False
    group._extra_env_vars = {}
    group._resource_bindings_by_rank = {0: binding}

    group._create_workers()

    env_vars = cluster.allocate.call_args.kwargs["env_vars"]
    assert RESOURCE_BINDING_ENV in env_vars


def test_worker_parses_resource_binding_from_env() -> None:
    binding = WorkerResourceBinding(
        component="env",
        rank=2,
        cluster_node_rank=0,
        node_group_label="node",
    )
    worker = object.__new__(Worker)

    with mock.patch.dict("os.environ", {RESOURCE_BINDING_ENV: binding.to_json()}):
        worker._setup_resource_binding()

    assert worker.resource_binding == binding
    assert worker._resource_binding_dict == asdict(binding)
