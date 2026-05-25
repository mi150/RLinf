import pytest

from rlinf.scheduler.resource_pool import (
    apply_process_cpu_affinity,
    build_even_split_cpu_groups,
    effective_process_affinity,
    get_env_core_group_from_env,
    parse_cpu_core_set,
    parse_env_cpu_core_groups,
)


def test_parse_cpu_core_set_supports_ranges_and_values() -> None:
    assert parse_cpu_core_set("0-3,8,10-11") == (0, 1, 2, 3, 8, 10, 11)


def test_parse_cpu_core_set_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_cpu_core_set("0-2,2")


def test_build_even_split_cpu_groups_distributes_remainder() -> None:
    assert build_even_split_cpu_groups(tuple(range(10)), partitions=3) == (
        (0, 1, 2, 3),
        (4, 5, 6),
        (7, 8, 9),
    )


def test_build_even_split_cpu_groups_requires_one_core_per_partition() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_even_split_cpu_groups((0, 1), partitions=3)


def test_parse_env_cpu_core_groups_and_lookup_by_index() -> None:
    assert parse_env_cpu_core_groups("0;1,2;3") == ((0,), (1, 2), (3,))
    assert get_env_core_group_from_env(
        {"RLINF_ENV_CPU_CORE_GROUPS": "0;1,2;3"}, local_env_index=1
    ) == (1, 2)


def test_get_env_core_group_from_env_returns_none_when_unconfigured() -> None:
    assert get_env_core_group_from_env({}, local_env_index=0) is None


def test_get_env_core_group_from_env_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError, match="local env index 2"):
        get_env_core_group_from_env(
            {"RLINF_ENV_CPU_CORE_GROUPS": "0;1"}, local_env_index=2
        )


def test_parse_env_cpu_core_groups_allows_explicit_shared_cores() -> None:
    assert parse_env_cpu_core_groups("0;0,1") == ((0,), (0, 1))


def test_effective_process_affinity_unions_groups() -> None:
    assert effective_process_affinity(((3, 1), (2,))) == (1, 2, 3)


def test_apply_process_cpu_affinity_calls_sched_setaffinity(monkeypatch) -> None:
    captured = {}

    def fake_sched_setaffinity(pid: int, cpus: set[int]) -> None:
        captured["pid"] = pid
        captured["cpus"] = cpus

    monkeypatch.setattr("os.sched_setaffinity", fake_sched_setaffinity, raising=False)
    apply_process_cpu_affinity((0, 2))

    assert captured == {"pid": 0, "cpus": {0, 2}}
