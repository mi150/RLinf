"""Scenario matrix expansion for rollout_eval benchmark."""

from __future__ import annotations

import re

from toolkits.rollout_eval.benchmark.types import BenchmarkCase, BenchmarkRequest

SCENARIOS = (
    "concurrent_mps",
    "concurrent_mig",
    "env_only_mps",
    "model_only_mps",
    "env_only_mig",
    "model_only_mig",
)


def _normalize_token(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return normalized or "x"


def _make_case_id(scenario: str, preset_name: str, resource_token: str) -> str:
    return f"{_normalize_token(scenario)}-{_normalize_token(preset_name)}-{_normalize_token(resource_token)}"


def expand_cases(request: BenchmarkRequest) -> list[BenchmarkCase]:
    """Expand request to deterministic case matrix across scenarios/resources."""
    cases: list[BenchmarkCase] = []

    scenarios = [scenario for scenario in SCENARIOS if scenario in request.scenario_set]
    presets = sorted(request.presets, key=lambda preset: preset.name)
    mps_values = sorted(set(request.mps_sm))
    mig_values = sorted(set(request.mig_devices))

    for scenario in scenarios:
        if scenario.endswith("_mps"):
            for preset in presets:
                for sm in mps_values:
                    resource_token = f"mps-sm{sm}"
                    cases.append(
                        BenchmarkCase(
                            case_id=_make_case_id(scenario, preset.name, resource_token),
                            scenario=scenario,
                            preset_name=preset.name,
                            env_type=preset.env_type,
                            model_type=preset.model_type,
                            mps_sm=sm,
                        )
                    )
            continue

        if scenario.endswith("_mig"):
            for preset in presets:
                for mig_device in mig_values:
                    resource_token = f"mig-{mig_device}"
                    cases.append(
                        BenchmarkCase(
                            case_id=_make_case_id(scenario, preset.name, resource_token),
                            scenario=scenario,
                            preset_name=preset.name,
                            env_type=preset.env_type,
                            model_type=preset.model_type,
                            mig_device=mig_device,
                        )
                    )

    return cases
