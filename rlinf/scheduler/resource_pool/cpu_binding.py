from __future__ import annotations

import os
from collections.abc import Mapping

from .bindings import ENV_CPU_CORE_GROUPS_ENV


def parse_cpu_core_set(spec: str) -> tuple[int, ...]:
    text = str(spec).strip()
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
            start = int(start_text)
            end = int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"invalid cpu range '{token}'")
            cores.extend(range(start, end + 1))
            continue
        core = int(token)
        if core < 0:
            raise ValueError(f"cpu core id must be >= 0, got {core}")
        cores.append(core)

    normalized = tuple(sorted(cores))
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate cpu cores are not allowed")
    return normalized


def build_even_split_cpu_groups(
    cores: tuple[int, ...], partitions: int
) -> tuple[tuple[int, ...], ...]:
    if partitions <= 0:
        raise ValueError("partitions must be > 0")
    if len(set(cores)) != len(cores):
        raise ValueError("duplicate cpu cores are not allowed")
    if len(cores) < partitions:
        raise ValueError("each partition must receive at least one cpu core")

    base = len(cores) // partitions
    remainder = len(cores) % partitions
    groups: list[tuple[int, ...]] = []
    cursor = 0
    for index in range(partitions):
        size = base + (1 if index < remainder else 0)
        groups.append(tuple(cores[cursor : cursor + size]))
        cursor += size
    return tuple(groups)


def effective_process_affinity(groups: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return tuple(sorted({core for group in groups for core in group}))


def parse_env_cpu_core_groups(spec: str) -> tuple[tuple[int, ...], ...]:
    text = str(spec).strip()
    if not text:
        return ()
    return tuple(parse_cpu_core_set(group) for group in text.split(";"))


def get_env_core_group_from_env(
    env: Mapping[str, str], local_env_index: int
) -> tuple[int, ...] | None:
    spec = env.get(ENV_CPU_CORE_GROUPS_ENV)
    if not spec:
        return None

    groups = parse_env_cpu_core_groups(spec)
    if local_env_index < 0 or local_env_index >= len(groups):
        raise ValueError(
            f"missing cpu core group for local env index {local_env_index}; "
            f"available groups: {len(groups)}"
        )
    return groups[local_env_index]


def apply_process_cpu_affinity(cpus: tuple[int, ...]) -> None:
    if not cpus:
        raise ValueError("cpu affinity set must not be empty")
    if not hasattr(os, "sched_setaffinity"):
        raise NotImplementedError("os.sched_setaffinity is unavailable")
    os.sched_setaffinity(0, set(cpus))
