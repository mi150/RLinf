from __future__ import annotations

from pathlib import Path

import pytest

from toolkits.rollout_eval.benchmark.resource_binding import (
    CUDA_VISIBLE_DEVICES_ENV,
    MPS_ACTIVE_THREAD_PERCENTAGE_ENV,
    apply_cpu_affinity,
    build_even_split_cpu_groups,
    build_process_env,
    effective_process_affinity,
    load_cpu_groups_from_yaml,
    parse_cpu_core_set,
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


def test_parse_cpu_core_set_supports_ranges_and_discrete_values() -> None:
    assert parse_cpu_core_set("0-3,8,10-11") == (0, 1, 2, 3, 8, 10, 11)


@pytest.mark.parametrize(
    "spec",
    [
        "3-1",
        "1-",
        "-1",
        "a-b",
        "2-3-4",
    ],
)
def test_parse_cpu_core_set_rejects_malformed_ranges(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_cpu_core_set(spec)


def test_parse_cpu_core_set_rejects_duplicate_cores() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_cpu_core_set("0-2,2-4")


def test_build_even_split_cpu_groups_distributes_remainder() -> None:
    groups = build_even_split_cpu_groups(tuple(range(10)), env_count=3)
    assert groups == (
        (0, 1, 2, 3),
        (4, 5, 6),
        (7, 8, 9),
    )


def test_build_even_split_cpu_groups_rejects_insufficient_cores() -> None:
    with pytest.raises(ValueError, match="at least one core"):
        build_even_split_cpu_groups((0, 1), env_count=3)


def test_build_even_split_cpu_groups_rejects_duplicate_core_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_even_split_cpu_groups((0, 1, 1, 2), env_count=2)


@pytest.mark.parametrize("env_count", [0, -1])
def test_build_even_split_cpu_groups_rejects_invalid_env_count(env_count: int) -> None:
    with pytest.raises(ValueError, match="env_count"):
        build_even_split_cpu_groups((0, 1, 2), env_count=env_count)


def test_load_cpu_groups_from_yaml_validates_overlap(tmp_path: Path) -> None:
    config_path = tmp_path / "cpu_bind.yaml"
    config_path.write_text(
        "env_core_groups:\n"
        "  - [0, 1]\n"
        "  - [1, 2]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlap"):
        load_cpu_groups_from_yaml(str(config_path), env_count=2)


def test_load_cpu_groups_from_yaml_validates_env_count(tmp_path: Path) -> None:
    config_path = tmp_path / "cpu_bind.yaml"
    config_path.write_text(
        "env_core_groups:\n"
        "  - [0, 1]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="env_count"):
        load_cpu_groups_from_yaml(str(config_path), env_count=2)


def test_load_cpu_groups_from_yaml_validates_non_empty_groups(tmp_path: Path) -> None:
    config_path = tmp_path / "cpu_bind.yaml"
    config_path.write_text(
        "env_core_groups:\n"
        "  - [0]\n"
        "  - []\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least one cpu"):
        load_cpu_groups_from_yaml(str(config_path), env_count=2)


def test_load_cpu_groups_from_yaml_success(tmp_path: Path) -> None:
    config_path = tmp_path / "cpu_bind.yaml"
    config_path.write_text(
        "env_core_groups:\n"
        "  - [0, 2]\n"
        "  - [4, 6]\n",
        encoding="utf-8",
    )

    groups = load_cpu_groups_from_yaml(str(config_path), env_count=2)
    assert groups == ((0, 2), (4, 6))


def test_load_cpu_groups_from_yaml_rejects_boolean_cpu_values(tmp_path: Path) -> None:
    config_path = tmp_path / "cpu_bind.yaml"
    config_path.write_text(
        "env_core_groups:\n"
        "  - [0, true]\n"
        "  - [2, false]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="integer"):
        load_cpu_groups_from_yaml(str(config_path), env_count=2)


def test_effective_process_affinity_unions_and_sorts() -> None:
    assert effective_process_affinity(((3, 1), (2,))) == (1, 2, 3)


def test_apply_cpu_affinity_delegates_to_os(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def _fake_sched_setaffinity(pid: int, cpus: set[int]) -> None:
        captured["pid"] = pid
        captured["cpus"] = cpus

    monkeypatch.setattr(
        "toolkits.rollout_eval.benchmark.resource_binding.os.sched_setaffinity",
        _fake_sched_setaffinity,
    )
    apply_cpu_affinity((0, 2, 4))

    assert captured == {"pid": 0, "cpus": {0, 2, 4}}


def test_apply_cpu_affinity_raises_when_platform_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(
        "toolkits.rollout_eval.benchmark.resource_binding.os.sched_setaffinity",
        raising=False,
    )

    with pytest.raises(NotImplementedError, match="unavailable"):
        apply_cpu_affinity((0,))


def test_apply_cpu_affinity_rejects_empty_cpu_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"value": False}

    def _fake_sched_setaffinity(_pid: int, _cpus: set[int]) -> None:
        called["value"] = True

    monkeypatch.setattr(
        "toolkits.rollout_eval.benchmark.resource_binding.os.sched_setaffinity",
        _fake_sched_setaffinity,
    )
    with pytest.raises(ValueError, match="must not be empty"):
        apply_cpu_affinity(())
    assert called["value"] is False
