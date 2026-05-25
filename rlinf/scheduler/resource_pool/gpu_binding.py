from __future__ import annotations

from .bindings import (
    CUDA_VISIBLE_DEVICES_ENV,
    MPS_ACTIVE_THREAD_PERCENTAGE_ENV,
    GpuBinding,
)

ALLOWED_SM_PERCENTAGES = (0, 20, 40, 60, 80, 100)


def validate_sm_percent(value: int | None) -> int:
    percent = 0 if value is None else int(value)
    if percent not in ALLOWED_SM_PERCENTAGES:
        raise ValueError(
            f"sm_percent must be one of {ALLOWED_SM_PERCENTAGES}, got {percent}"
        )
    return percent


def build_gpu_env_vars(binding: GpuBinding) -> dict[str, str]:
    if binding.sm_percent == 0:
        return {}
    if binding.mode == "mps":
        if not binding.visible_devices:
            raise ValueError("MPS binding requires visible_devices")
        return {
            CUDA_VISIBLE_DEVICES_ENV: ",".join(binding.visible_devices),
            MPS_ACTIVE_THREAD_PERCENTAGE_ENV: str(binding.sm_percent),
        }
    if binding.mode == "mig":
        if not binding.mig_device_uuid:
            raise ValueError("MIG binding requires mig_device_uuid")
        return {CUDA_VISIBLE_DEVICES_ENV: binding.mig_device_uuid}
    return {}
