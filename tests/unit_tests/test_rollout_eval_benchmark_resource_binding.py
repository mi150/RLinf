from __future__ import annotations

import pytest

from toolkits.rollout_eval.benchmark.resource_binding import (
    CUDA_VISIBLE_DEVICES_ENV,
    MPS_ACTIVE_THREAD_PERCENTAGE_ENV,
    build_process_env,
)


def test_build_process_env_is_non_destructive_and_merges_inputs() -> None:
    base_env = {
        "PATH": "/usr/bin",
        "EXISTING": "1",
    }
    extra_env = {
        "EXTRA_A": "a",
        "EXTRA_B": "b",
    }

    built = build_process_env(base_env=base_env, extra_env=extra_env)

    assert built == {
        "PATH": "/usr/bin",
        "EXISTING": "1",
        "EXTRA_A": "a",
        "EXTRA_B": "b",
    }
    assert base_env == {
        "PATH": "/usr/bin",
        "EXISTING": "1",
    }


def test_build_process_env_overrides_cuda_visible_devices_with_mig_uuid() -> None:
    built = build_process_env(
        base_env={CUDA_VISIBLE_DEVICES_ENV: "0"},
        mig_device_uuid="MIG-7b2f4b7f-03f2-58fd-a0a4-123456789abc",
    )

    assert built[CUDA_VISIBLE_DEVICES_ENV] == "MIG-7b2f4b7f-03f2-58fd-a0a4-123456789abc"


def test_build_process_env_sets_and_overrides_mps_percentage() -> None:
    built = build_process_env(
        base_env={MPS_ACTIVE_THREAD_PERCENTAGE_ENV: "25"},
        mps_active_thread_percentage=60,
    )

    assert built[MPS_ACTIVE_THREAD_PERCENTAGE_ENV] == "60"


@pytest.mark.parametrize("invalid", [0, -1, 101])
def test_build_process_env_rejects_invalid_mps_percentage(invalid: int) -> None:
    with pytest.raises(ValueError, match="mps_active_thread_percentage"):
        build_process_env(mps_active_thread_percentage=invalid)
