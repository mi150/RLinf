"""Tests for reporting and orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import pytest

from toolkits.rollout_eval.config_bridge import EvalRuntimeConfig
from toolkits.rollout_eval.experiment.reporting import (
    dump_action_replace_report,
    dump_baseline_report,
    dump_cache_report,
)
from toolkits.rollout_eval.experiment.run_experiment import (
    parse_args,
    run_experiment,
)
from toolkits.rollout_eval.experiment.types import (
    CacheEvalResult,
    DeterminismResult,
    EpisodeTrajectory,
    ExperimentConfig,
    StepRecord,
)
from toolkits.rollout_eval.rollout_types import EnvStepResult


# -----------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------

def _make_step(step: int, replaced: bool = False) -> StepRecord:
    return StepRecord(
        step=step, obs={}, action=torch.zeros(7), reward=1.0,
        terminated=False, truncated=False, info={},
        model_latency_ms=10.0, env_latency_ms=5.0,
        meta={"replaced": replaced, "replacement_l2": 0.01} if replaced else {},
    )


def _make_traj(seed: int, n_steps: int = 5, success: bool = True) -> EpisodeTrajectory:
    return EpisodeTrajectory(
        seed=seed, env_index=0,
        steps=[_make_step(i) for i in range(n_steps)],
        success=success, total_reward=float(n_steps),
    )


class TestDumpBaselineReport:
    def test_writes_json(self, tmp_path):
        det = {42: DeterminismResult(seed=42, action_match=True, max_action_l2=0.0,
                                     obs_match=True, reward_match=True)}
        trajs = {42: [_make_traj(42)]}

        path = dump_baseline_report(
            output_dir=str(tmp_path), seeds=[42], num_runs_per_seed=1,
            determinism=det, trajectories=trajs,
        )
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["phase"] == "baseline"
        assert "42" in data["determinism"]


class TestDumpCacheReport:
    def test_writes_json(self, tmp_path):
        result = CacheEvalResult(
            cache_mode="naive", hit_rate=0.8, same_step_hit_rate=0.8,
            latency_with_cache_ms=10.0, latency_without_cache_ms=40.0,
            latency_savings_pct=75.0, action_divergence_l2_mean=0.001,
            action_divergence_l2_max=0.01, success_rate_pass1=0.5,
            success_rate_pass2=0.5,
        )
        path = dump_cache_report(output_dir=str(tmp_path), cache_result=result)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["phase"] == "cache_eval"
        assert data["latency_savings_pct"] == 75.0


class TestDumpActionReplaceReport:
    def test_writes_json(self, tmp_path):
        baseline = {42: [_make_traj(42)]}
        replaced_steps = [_make_step(i, replaced=(i >= 7)) for i in range(10)]
        replaced = [EpisodeTrajectory(
            seed=42, env_index=0, steps=replaced_steps,
            success=True, total_reward=10.0,
        )]

        path = dump_action_replace_report(
            output_dir=str(tmp_path), k_b=3, k_b_source="static",
            action_source="pipeline", total_steps=10,
            baseline_trajectories=baseline, replaced_trajectories=replaced,
        )
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["phase"] == "action_replace"
        assert data["k_b"] == 3


# -----------------------------------------------------------------------
# CLI arg parsing
# -----------------------------------------------------------------------

class TestParseArgs:
    def test_basic(self):
        args = parse_args([
            "--config-path", "/tmp/cfg",
            "--config-name", "test",
            "--phases", "baseline,cache_eval",
            "--seeds", "42,43",
        ])
        assert args.phases == "baseline,cache_eval"
        assert args.seeds == "42,43"

    def test_defaults(self):
        args = parse_args(["--config-path", "/tmp", "--config-name", "x"])
        assert args.num_runs_per_seed == 2
        assert args.action_source == "pipeline"


# -----------------------------------------------------------------------
# Orchestrator (mock-based integration)
# -----------------------------------------------------------------------

def _make_runtime(total_steps: int = 5) -> EvalRuntimeConfig:
    return EvalRuntimeConfig(
        env_type="mock", model_type="mock", model_path="/tmp/mock",
        num_envs=1, group_size=1, num_action_chunks=1,
        total_steps=total_steps, warmup_steps=1, seed=42,
    )


def _make_mock_env():
    env = MagicMock()
    obs = {"img": torch.zeros(1, 3, 64, 64)}
    env.reset.return_value = (obs, {})
    env.step.return_value = EnvStepResult(
        obs=obs, reward=torch.tensor([1.0]),
        terminated=torch.tensor([False]), truncated=torch.tensor([False]),
        info={},
    )
    env.current_seed = 42
    env.env = MagicMock()
    return env


def _make_mock_model():
    model = MagicMock()
    model.infer.return_value = (torch.zeros(1, 7), {})
    return model


class TestRunExperiment:
    def test_baseline_only(self, tmp_path):
        cfg = ExperimentConfig(
            eval_runtime=_make_runtime(total_steps=3),
            phases=["baseline"],
            seeds=[42],
            num_runs_per_seed=2,
            output_dir=str(tmp_path),
        )
        env = _make_mock_env()
        model = _make_mock_model()

        run_experiment(cfg, env, model)

        assert (tmp_path / "reports" / "phase1_baseline.json").exists()
        assert (tmp_path / "experiment_config.json").exists()

    def test_unsupported_when_feature_cache_absent(self, tmp_path, monkeypatch):
        cfg = ExperimentConfig(
            eval_runtime=_make_runtime(total_steps=3),
            phases=["cache_eval", "baseline"],
            seeds=[42],
            output_dir=str(tmp_path),
        )
        env = _make_mock_env()
        model = _make_mock_model()

        monkeypatch.setattr(
            "toolkits.rollout_eval.experiment.run_experiment._is_feature_cache_available",
            lambda: False,
            raising=False,
        )

        run_experiment(cfg, env, model)

        cache_report = tmp_path / "reports" / "phase2_cache_eval.json"
        baseline_report = tmp_path / "reports" / "phase1_baseline.json"
        assert cache_report.exists()
        assert baseline_report.exists()

        payload = json.loads(cache_report.read_text())
        assert payload["phase"] == "cache_eval"
        assert payload["status"] == "unsupported"
        assert "feature cache unavailable" in payload["reason"].lower()

    def test_unknown_phase_warns(self, tmp_path, caplog):
        import logging
        cfg = ExperimentConfig(
            eval_runtime=_make_runtime(total_steps=2),
            phases=["unknown_phase"],
            seeds=[42],
            output_dir=str(tmp_path),
        )
        with caplog.at_level(logging.WARNING):
            run_experiment(cfg, _make_mock_env(), _make_mock_model())
        assert "Unknown phase" in caplog.text
