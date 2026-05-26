import json

import pytest

from rlinf.scheduler.resource_pool.bindings import (
    ACCELERATOR_TYPE_ENV,
    CPU_AFFINITY_ENV,
    CUDA_VISIBLE_DEVICES_ENV,
    ENV_CPU_CORE_GROUPS_ENV,
    MPS_ACTIVE_THREAD_PERCENTAGE_ENV,
    MUJOCO_EGL_DEVICE_ID_ENV,
    RESOURCE_BINDING_ENV,
    CpuBinding,
    GpuBinding,
    WorkerResourceBinding,
)


def test_worker_resource_binding_round_trips_and_builds_env() -> None:
    binding = WorkerResourceBinding(
        component="env",
        rank=0,
        cluster_node_rank=1,
        node_group_label="cluster",
        cpu=CpuBinding(
            process_cpu_cores=(0, 1, 2, 3),
            env_cpu_core_groups=((0,), (1,), (2,), (3,)),
        ),
        gpu=GpuBinding(
            mode="mps",
            sm_percent=40,
            visible_devices=("0",),
            mig_device_uuid=None,
            parent_gpu=0,
        ),
    )

    restored = WorkerResourceBinding.from_json(binding.to_json())

    assert restored == binding
    env = binding.to_env_vars()
    assert json.loads(env[RESOURCE_BINDING_ENV])["component"] == "env"
    assert env[CPU_AFFINITY_ENV] == "0,1,2,3"
    assert env[ENV_CPU_CORE_GROUPS_ENV] == "0;1;2;3"
    assert env[CUDA_VISIBLE_DEVICES_ENV] == "0"
    assert env[MPS_ACTIVE_THREAD_PERCENTAGE_ENV] == "40"


def test_worker_resource_binding_without_gpu_hides_cuda_devices() -> None:
    binding = WorkerResourceBinding(
        component="env",
        rank=0,
        cluster_node_rank=0,
        node_group_label="node",
        cpu=CpuBinding(process_cpu_cores=(0,)),
        gpu=None,
    )

    env = binding.to_env_vars()

    assert env[CUDA_VISIBLE_DEVICES_ENV] == ""
    assert env[MUJOCO_EGL_DEVICE_ID_ENV] == ""
    assert env[MPS_ACTIVE_THREAD_PERCENTAGE_ENV] == ""
    assert env[ACCELERATOR_TYPE_ENV] == "NO_ACCEL"


def test_zero_quota_gpu_binding_preserves_render_device_without_accelerator() -> None:
    binding = WorkerResourceBinding(
        component="env",
        rank=0,
        cluster_node_rank=0,
        node_group_label="node",
        cpu=CpuBinding(process_cpu_cores=(0,)),
        gpu=GpuBinding(
            mode="mps",
            sm_percent=0,
            visible_devices=("3",),
            parent_gpu=3,
        ),
    )

    env = binding.to_env_vars()

    assert env[ACCELERATOR_TYPE_ENV] == "NO_ACCEL"
    assert env[CUDA_VISIBLE_DEVICES_ENV] == "3"
    assert env[MUJOCO_EGL_DEVICE_ID_ENV] == "3"
    assert env[MPS_ACTIVE_THREAD_PERCENTAGE_ENV] == ""


def test_gpu_binding_from_dict_normalizes_optional_types() -> None:
    binding = GpuBinding.from_dict(
        {
            "mode": "mig",
            "sm_percent": "20",
            "visible_devices": (),
            "mig_device_uuid": 123,
            "parent_gpu": "0",
        }
    )

    assert binding == GpuBinding(
        mode="mig",
        sm_percent=20,
        visible_devices=(),
        mig_device_uuid="123",
        parent_gpu=0,
    )


def test_gpu_binding_from_dict_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        GpuBinding.from_dict({"mode": "exclusive", "sm_percent": 20})


def test_worker_resource_binding_requires_explicit_gpu_key() -> None:
    payload = {
        "component": "env",
        "rank": 0,
        "cluster_node_rank": 0,
        "node_group_label": "node",
        "cpu": None,
    }

    with pytest.raises(ValueError, match="gpu"):
        WorkerResourceBinding.from_json(json.dumps(payload))


@pytest.mark.parametrize("payload", [{}, {"mode": None, "sm_percent": 0}])
def test_gpu_binding_from_dict_rejects_missing_mode(payload: dict) -> None:
    with pytest.raises(ValueError, match="mode"):
        GpuBinding.from_dict(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"mode": "mps"},
        {"mode": "mps", "sm_percent": None},
    ],
)
def test_gpu_binding_from_dict_requires_sm_percent(payload: dict) -> None:
    with pytest.raises(ValueError, match="sm_percent"):
        GpuBinding.from_dict(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"mode": "mig", "sm_percent": 0, "parent_gpu": 0},
        {"mode": "mig", "sm_percent": 0, "mig_device_uuid": "MIG-A"},
    ],
)
def test_gpu_binding_from_dict_requires_mig_metadata(payload: dict) -> None:
    with pytest.raises(ValueError, match="MIG"):
        GpuBinding.from_dict(payload)
