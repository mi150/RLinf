"""Scenario matrix expansion for rollout_eval benchmark."""

from __future__ import annotations

import hashlib
import re

from toolkits.rollout_eval.benchmark.types import BenchmarkCase, BenchmarkRequest

SCENARIOS = (
    "concurrent_mps",
    "concurrent_mig",
    "concurrent_cpu_core",
    "env_only_mps",
    "model_only_mps",
    "env_only_mig",
    "model_only_mig",
    "env_only_cpu_core",
)


def _normalize_token(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return normalized or "x"


def _make_case_id(scenario: str, preset_name: str, resource_token: str) -> str:
    return f"{_normalize_token(scenario)}-{_normalize_token(preset_name)}-{_normalize_token(resource_token)}"


def _parse_cpu_core_set(spec: str) -> tuple[int, ...]:
    if not spec.strip():
        return ()
    values: list[int] = []
    for token in spec.split(","):
        item = token.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid cpu core range '{item}'")
            values.extend(range(start, end + 1))
            continue
        values.append(int(item))
    return tuple(sorted(set(values)))


def expand_cases(request: BenchmarkRequest) -> list[BenchmarkCase]:
    """Expand request to deterministic case matrix across scenarios/resources."""
    cases: list[BenchmarkCase] = []

    scenarios = [scenario for scenario in SCENARIOS if scenario in request.scenario_set]
    presets = sorted(request.presets, key=lambda preset: preset.name)
    mps_values = sorted(set(request.mps_sm))
    mig_values = sorted(set(request.mig_devices))
    cpu_binding_mode = request.cpu_bind_strategy or "even_split"
    cpu_binding_disabled = cpu_binding_mode == "none"
    parsed_cpu_cores = _parse_cpu_core_set(request.cpu_bind_cores or "")
    cpu_available_cores = parsed_cpu_cores if parsed_cpu_cores else None
    cpu_scenarios_requested = any(
        scenario in request.scenario_set
        for scenario in ("env_only_cpu_core", "concurrent_cpu_core")
    )
    if cpu_scenarios_requested and cpu_available_cores is None and not cpu_binding_disabled:
        raise ValueError(
            "cpu_bind_cores must provide at least one core when cpu_core "
            "scenarios are requested in Task 1"
        )
    if cpu_binding_disabled:
        cpu_resource_token = "cpu-none-default-sched"
    else:
        cpu_core_digest = hashlib.sha1(
            ",".join(str(core) for core in cpu_available_cores or ()).encode("utf-8")
        ).hexdigest()[:8]
        cpu_resource_token = (
            "cpu-"
            f"{cpu_binding_mode}-"
            f"n{len(cpu_available_cores)}-"
            f"h{cpu_core_digest}"
            if cpu_available_cores is not None
            else None
        )

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
            continue

        if scenario.endswith("_cpu_core"):
            if cpu_resource_token is None:
                continue
            for preset in presets:
                cases.append(
                    BenchmarkCase(
                        case_id=_make_case_id(
                            scenario, preset.name, cpu_resource_token
                        ),
                        scenario=scenario,
                        preset_name=preset.name,
                        env_type=preset.env_type,
                        model_type=preset.model_type,
                        cpu_binding_mode=cpu_binding_mode,
                        cpu_available_cores=cpu_available_cores,
                    )
                )

    return cases
