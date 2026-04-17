"""Helpers for non-destructive process-level GPU resource binding.

This module only prepares environment variables for child processes.
It does not manage CUDA MPS daemon lifecycle and does not create or destroy MIG
instances.
"""

from __future__ import annotations

from collections.abc import Mapping

MPS_ACTIVE_THREAD_PERCENTAGE_ENV = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
CUDA_VISIBLE_DEVICES_ENV = "CUDA_VISIBLE_DEVICES"


def _validate_mps_percentage(value: int) -> int:
    if value < 1 or value > 100:
        raise ValueError(
            "mps_active_thread_percentage must be in [1, 100], "
            f"got {value}."
        )
    return value


def build_process_env(
    *,
    base_env: Mapping[str, str] | None = None,
    extra_env: Mapping[str, str] | None = None,
    mig_device_uuid: str | None = None,
    mps_active_thread_percentage: int | None = None,
) -> dict[str, str]:
    """Build child-process env with optional MIG/MPS binding.

    Args:
        base_env: Optional source environment. The mapping is never mutated.
        extra_env: Additional environment variables to merge.
        mig_device_uuid: If provided, overrides ``CUDA_VISIBLE_DEVICES`` with a
            MIG device UUID (for example ``MIG-xxxxxxxx``).
        mps_active_thread_percentage: Optional MPS SM share hint in percentage.
            This maps to ``CUDA_MPS_ACTIVE_THREAD_PERCENTAGE``.

    Returns:
        A new dictionary suitable for ``subprocess`` or multiprocessing child
        process env injection.
    """
    merged: dict[str, str] = dict(base_env or {})

    if extra_env:
        merged.update({k: str(v) for k, v in extra_env.items()})

    if mig_device_uuid:
        merged[CUDA_VISIBLE_DEVICES_ENV] = mig_device_uuid

    if mps_active_thread_percentage is not None:
        percentage = _validate_mps_percentage(mps_active_thread_percentage)
        merged[MPS_ACTIVE_THREAD_PERCENTAGE_ENV] = str(percentage)

    return merged
