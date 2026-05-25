import pytest

from rlinf.scheduler.resource_pool.bindings import (
    CUDA_VISIBLE_DEVICES_ENV,
    MPS_ACTIVE_THREAD_PERCENTAGE_ENV,
    GpuBinding,
)
from rlinf.scheduler.resource_pool.gpu_binding import (
    build_gpu_env_vars,
    validate_sm_percent,
)


@pytest.mark.parametrize("value", [0, 20, 40, 60, 80, 100])
def test_validate_sm_percent_accepts_supported_values(value: int) -> None:
    assert validate_sm_percent(value) == value


def test_validate_sm_percent_rejects_unsupported_value() -> None:
    with pytest.raises(ValueError, match="sm_percent"):
        validate_sm_percent(25)


def test_mps_zero_quota_does_not_build_gpu_env() -> None:
    binding = GpuBinding(
        mode="mps",
        sm_percent=0,
        visible_devices=("0",),
        parent_gpu=0,
    )
    assert build_gpu_env_vars(binding) == {}


def test_mps_quota_builds_visible_device_and_percentage_env() -> None:
    binding = GpuBinding(
        mode="mps",
        sm_percent=60,
        visible_devices=("1",),
        parent_gpu=1,
    )
    env = build_gpu_env_vars(binding)
    assert env[CUDA_VISIBLE_DEVICES_ENV] == "1"
    assert env[MPS_ACTIVE_THREAD_PERCENTAGE_ENV] == "60"


def test_mig_quota_builds_uuid_visibility() -> None:
    binding = GpuBinding(
        mode="mig",
        sm_percent=20,
        visible_devices=(),
        mig_device_uuid="MIG-abc",
        parent_gpu=0,
    )
    assert build_gpu_env_vars(binding) == {CUDA_VISIBLE_DEVICES_ENV: "MIG-abc"}
