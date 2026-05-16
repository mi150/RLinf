"""Profile LIBERO simulator step latency without Ray or policy models."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
BDDL_SCHEMA_KEYS = [
    "problem_name",
    "domain_name",
    "task_language",
    "scene_type",
    "scene_name",
    "region_names",
    "num_regions",
    "fixture_categories",
    "num_fixtures",
    "object_categories",
    "num_objects",
    "obj_of_interest",
    "num_obj_of_interest",
    "init_predicates",
    "num_init_predicates",
    "goal_predicates",
    "num_goal_predicates",
]


@dataclass(frozen=True)
class ProfileConfig:
    suite: str
    task_ids: str
    trials_per_task: int
    specific_trial_ids: list[int] | None
    warmup_steps: int
    measure_steps: int
    cpu_id: int | None
    cpu_ids: list[int] | None
    camera_height: int
    camera_width: int
    libero_type: str
    seed: int
    output_dir: Path
    dummy_action: list[float]
    stop_on_done: bool


@dataclass(frozen=True)
class TaskTrialSpec:
    suite_name: str
    task_id: int
    trial_id: int
    task_name: str
    task_language: str
    bddl_file: str
    seed: int


def parse_int_list(value: str, *, allow_all: bool = False) -> list[int] | str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("empty integer list")
    if stripped.lower() == "all":
        if allow_all:
            return "all"
        raise ValueError("'all' is not allowed here")
    result: list[int] = []
    for part in stripped.split(","):
        item = part.strip()
        if not item:
            raise ValueError(f"empty item in integer list: {value!r}")
        result.append(int(item))
    return result


def parse_task_ids(value: str, *, num_tasks: int) -> list[int]:
    parsed = parse_int_list(value, allow_all=True)
    if parsed == "all":
        return list(range(num_tasks))
    task_ids = parsed
    for task_id in task_ids:
        if task_id < 0 or task_id >= num_tasks:
            raise ValueError(f"task id {task_id} out of range [0, {num_tasks})")
    return task_ids


def select_trial_ids(
    *,
    num_trials: int,
    trials_per_task: int,
    specific_trial_ids: list[int] | None,
    seed: int,
    task_id: int,
) -> list[int]:
    if num_trials <= 0:
        return []
    if specific_trial_ids is not None:
        for trial_id in specific_trial_ids:
            if trial_id < 0 or trial_id >= num_trials:
                raise ValueError(f"trial id {trial_id} out of range [0, {num_trials})")
        return list(specific_trial_ids)
    if trials_per_task >= num_trials:
        return list(range(num_trials))
    rng = np.random.default_rng(seed + task_id)
    selected = rng.choice(num_trials, size=trials_per_task, replace=False)
    return [int(item) for item in selected.tolist()]


def parse_dummy_action(value: str | None) -> list[float]:
    if value is None:
        return list(DEFAULT_DUMMY_ACTION)
    stripped = value.strip()
    if not stripped:
        raise ValueError("empty dummy action")
    return [float(part.strip()) for part in stripped.split(",")]


def _balanced_section(text: str, section_name: str) -> str:
    marker = f"(:{section_name}"
    start = text.find(marker)
    if start < 0:
        return ""
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def _extract_problem_name(text: str) -> str | None:
    match = re.search(r"\(define\s+\(problem\s+([^)]+)\)", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def _extract_simple_section_value(text: str, section_name: str) -> str | None:
    match = re.search(rf"\(:{section_name}\s+([^)]+)\)", text, flags=re.IGNORECASE)
    return " ".join(match.group(1).split()) if match else None


def _extract_category_lines(section: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("(:") or line == ")":
            continue
        if " - " not in line:
            continue
        instance, category = [part.strip() for part in line.split(" - ", 1)]
        category = category.rstrip(")")
        result.setdefault(category, []).append(instance)
    return result


def _extract_region_names(section: str) -> list[str]:
    names: list[str] = []
    body = section[len("(:regions") :].strip()
    depth = 0
    token_start: int | None = None
    for index, char in enumerate(body):
        if char == "(":
            if depth == 0:
                token_start = index + 1
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and token_start is not None:
                item = body[token_start:index].strip()
                if item and not item.startswith(":"):
                    name = item.split()[0]
                    if name and name not in names:
                        names.append(name)
                token_start = None
    return names


def _extract_predicate_names(section: str) -> list[str]:
    names = re.findall(r"\(([A-Za-z_][A-Za-z0-9_]*)\b", section)
    return [
        name
        for name in names
        if name.lower()
        not in {
            "init",
            "goal",
            "and",
        }
    ]


def _infer_scene_type(problem_name: str | None, bddl_path: Path) -> str | None:
    source = f"{problem_name or ''} {bddl_path.name}".lower()
    if "living_room" in source:
        return "living_room"
    if "coffee_table" in source:
        return "coffee_table"
    if "kitchen" in source:
        return "kitchen"
    if "study" in source:
        return "study"
    if "floor" in source:
        return "floor"
    if "tabletop" in source or "table" in source:
        return "table"
    return None


def _infer_scene_name(bddl_path: Path) -> str | None:
    match = re.match(r"([A-Z]+_SCENE\d+)", bddl_path.stem)
    return match.group(1) if match else None


def parse_bddl_metadata(bddl_path: str | Path) -> dict[str, Any]:
    path = Path(bddl_path)
    text = path.read_text()
    problem_name = _extract_problem_name(text)
    domain_name = _extract_simple_section_value(text, "domain")
    task_language = _extract_simple_section_value(text, "language")
    regions_section = _balanced_section(text, "regions")
    fixtures = _extract_category_lines(_balanced_section(text, "fixtures"))
    objects = _extract_category_lines(_balanced_section(text, "objects"))
    obj_of_interest_section = _balanced_section(text, "obj_of_interest")
    init_predicates = _extract_predicate_names(_balanced_section(text, "init"))
    goal_predicates = _extract_predicate_names(_balanced_section(text, "goal"))
    obj_of_interest = [
        line.strip()
        for line in obj_of_interest_section.splitlines()[1:]
        if line.strip() and line.strip() != ")"
    ]
    metadata: dict[str, Any] = {
        "problem_name": problem_name,
        "domain_name": domain_name,
        "task_language": task_language,
        "scene_type": _infer_scene_type(problem_name, path),
        "scene_name": _infer_scene_name(path),
        "region_names": _extract_region_names(regions_section),
        "fixture_categories": sorted(fixtures),
        "object_categories": sorted(objects),
        "obj_of_interest": obj_of_interest,
        "init_predicates": init_predicates,
        "goal_predicates": goal_predicates,
    }
    metadata["num_regions"] = len(metadata["region_names"])
    metadata["num_fixtures"] = sum(len(items) for items in fixtures.values())
    metadata["num_objects"] = sum(len(items) for items in objects.values())
    metadata["num_obj_of_interest"] = len(obj_of_interest)
    metadata["num_init_predicates"] = len(init_predicates)
    metadata["num_goal_predicates"] = len(goal_predicates)
    for key in BDDL_SCHEMA_KEYS:
        metadata.setdefault(key, None)
    return metadata


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def compute_latency_summary(latencies: list[float]) -> dict[str, float | int | None]:
    if not latencies:
        return {
            "step_count": 0,
            "mean_latency_s": None,
            "median_latency_s": None,
            "p90_latency_s": None,
            "p95_latency_s": None,
            "p99_latency_s": None,
            "min_latency_s": None,
            "max_latency_s": None,
            "std_latency_s": None,
            "tail_ratio_p99_to_median": None,
        }
    array = np.asarray(latencies, dtype=np.float64)
    median = float(np.median(array))
    p99 = _percentile(latencies, 99)
    return {
        "step_count": int(array.size),
        "mean_latency_s": float(np.mean(array)),
        "median_latency_s": median,
        "p90_latency_s": _percentile(latencies, 90),
        "p95_latency_s": _percentile(latencies, 95),
        "p99_latency_s": p99,
        "min_latency_s": float(np.min(array)),
        "max_latency_s": float(np.max(array)),
        "std_latency_s": float(np.std(array)),
        "tail_ratio_p99_to_median": (
            None if median == 0.0 or p99 is None else float(p99 / median)
        ),
    }
