"""Helpers for non-destructive process-level GPU resource binding.

This module only prepares environment variables for child processes.
It does not manage CUDA MPS daemon lifecycle and does not create or destroy MIG
instances.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import yaml

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


def parse_cpu_core_set(spec: str) -> tuple[int, ...]:
    """Parse a CPU core set string into sorted unique core IDs."""
    text = spec.strip()
    if not text:
        raise ValueError("cpu core spec must not be empty")

    cores: list[int] = []
    for raw_token in text.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError("cpu core spec contains an empty token")

        if "-" in token:
            if token.count("-") != 1:
                raise ValueError(f"invalid cpu range '{token}'")
            start_text, end_text = token.split("-", maxsplit=1)
            if not start_text or not end_text:
                raise ValueError(f"invalid cpu range '{token}'")
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError as exc:
                raise ValueError(f"invalid cpu range '{token}'") from exc
            if start < 0 or end < 0:
                raise ValueError(f"cpu core id must be >= 0, got '{token}'")
            if end < start:
                raise ValueError(f"invalid cpu range '{token}'")
            cores.extend(range(start, end + 1))
            continue

        try:
            core = int(token)
        except ValueError as exc:
            raise ValueError(f"invalid cpu core '{token}'") from exc
        if core < 0:
            raise ValueError(f"cpu core id must be >= 0, got '{token}'")
        cores.append(core)

    normalized = tuple(sorted(cores))
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate cpu cores are not allowed")
    return normalized


def build_even_split_cpu_groups(
    cores: tuple[int, ...], env_count: int
) -> tuple[tuple[int, ...], ...]:
    """Evenly split CPU cores into deterministic per-env groups."""
    if env_count <= 0:
        raise ValueError("env_count must be > 0")
    if len(set(cores)) != len(cores):
        raise ValueError("duplicate cpu cores are not allowed")
    if len(cores) < env_count:
        raise ValueError("each logical env must receive at least one core")

    base = len(cores) // env_count
    remainder = len(cores) % env_count
    groups: list[tuple[int, ...]] = []
    cursor = 0
    for env_idx in range(env_count):
        group_size = base + (1 if env_idx < remainder else 0)
        groups.append(tuple(cores[cursor : cursor + group_size]))
        cursor += group_size
    return tuple(groups)


def load_cpu_groups_from_yaml(path: str, env_count: int) -> tuple[tuple[int, ...], ...]:
    """Load explicit CPU groups from YAML and validate shape and overlap."""
    if env_count <= 0:
        raise ValueError("env_count must be > 0")

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("cpu bind yaml must define a mapping with env_core_groups")
    groups = payload.get("env_core_groups")
    if not isinstance(groups, list):
        raise ValueError("cpu bind yaml must define env_core_groups as a list")
    if len(groups) != env_count:
        raise ValueError("env_count does not match env_core_groups length")

    normalized: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for idx, group in enumerate(groups):
        if not isinstance(group, (list, tuple)):
            raise ValueError(f"env_core_groups[{idx}] must be a list")
        if len(group) == 0:
            raise ValueError("every env core group must contain at least one cpu")
        parsed_group: list[int] = []
        for value in group:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"invalid cpu id '{value}' in env_core_groups[{idx}]: "
                    "must be integer"
                )
            cpu = value
            if cpu < 0:
                raise ValueError(f"cpu core id must be >= 0, got '{cpu}'")
            if cpu in seen:
                raise ValueError("cpu bind yaml groups must not overlap")
            seen.add(cpu)
            parsed_group.append(cpu)
        normalized.append(tuple(parsed_group))

    return tuple(normalized)


def effective_process_affinity(groups: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    """Compute process-level affinity as the sorted union of groups."""
    return tuple(sorted({cpu for group in groups for cpu in group}))


def apply_cpu_affinity(cpus: tuple[int, ...]) -> None:
    """Apply CPU affinity for current process."""
    if not cpus:
        raise ValueError("cpu affinity set must not be empty")
    if not hasattr(os, "sched_setaffinity"):
        raise NotImplementedError(
            "os.sched_setaffinity is unavailable on this platform"
        )
    os.sched_setaffinity(0, set(cpus))
