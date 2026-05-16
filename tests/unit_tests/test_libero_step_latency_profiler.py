import math

import numpy as np
import pytest

from toolkits.profile_libero_step_latency import (
    compute_latency_summary,
    parse_dummy_action,
    parse_int_list,
    parse_task_ids,
    select_trial_ids,
)


def test_parse_int_list_accepts_all_and_numbers():
    assert parse_int_list("all", allow_all=True) == "all"
    assert parse_int_list("0,3,7", allow_all=True) == [0, 3, 7]
    assert parse_int_list(" 1 , 2 ") == [1, 2]


def test_parse_int_list_rejects_empty_and_all_when_disallowed():
    with pytest.raises(ValueError, match="empty"):
        parse_int_list("")
    with pytest.raises(ValueError, match="not allowed"):
        parse_int_list("all", allow_all=False)


def test_parse_task_ids_all_uses_task_count():
    assert parse_task_ids("all", num_tasks=4) == [0, 1, 2, 3]
    assert parse_task_ids("0,2", num_tasks=4) == [0, 2]


def test_parse_task_ids_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        parse_task_ids("4", num_tasks=4)


def test_select_trial_ids_specific_ids_win():
    assert select_trial_ids(
        num_trials=10,
        trials_per_task=2,
        specific_trial_ids=[5, 1],
        seed=0,
        task_id=3,
    ) == [5, 1]


def test_select_trial_ids_deterministic_prefix_when_under_limit():
    assert select_trial_ids(
        num_trials=3,
        trials_per_task=10,
        specific_trial_ids=None,
        seed=123,
        task_id=0,
    ) == [0, 1, 2]


def test_select_trial_ids_seeded_sample_is_stable():
    first = select_trial_ids(
        num_trials=20,
        trials_per_task=4,
        specific_trial_ids=None,
        seed=123,
        task_id=2,
    )
    second = select_trial_ids(
        num_trials=20,
        trials_per_task=4,
        specific_trial_ids=None,
        seed=123,
        task_id=2,
    )
    assert first == second
    assert len(first) == 4
    assert len(set(first)) == 4


def test_parse_dummy_action_default_and_override():
    np.testing.assert_allclose(parse_dummy_action(None), [0, 0, 0, 0, 0, 0, -1])
    np.testing.assert_allclose(parse_dummy_action("1,2,3"), [1, 2, 3])


def test_compute_latency_summary_percentiles_and_tail_ratio():
    summary = compute_latency_summary([0.01, 0.02, 0.03, 0.10])
    assert summary["step_count"] == 4
    assert math.isclose(summary["mean_latency_s"], 0.04)
    assert math.isclose(summary["median_latency_s"], 0.025)
    assert summary["p99_latency_s"] > summary["p95_latency_s"]
    assert math.isclose(
        summary["tail_ratio_p99_to_median"],
        summary["p99_latency_s"] / summary["median_latency_s"],
    )


def test_compute_latency_summary_empty_latency_list():
    summary = compute_latency_summary([])
    assert summary["step_count"] == 0
    assert summary["mean_latency_s"] is None
    assert summary["tail_ratio_p99_to_median"] is None
