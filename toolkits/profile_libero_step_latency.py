"""Profile LIBERO simulator step latency without Ray or policy models."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import time
import traceback
from dataclasses import asdict, dataclass, is_dataclass
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
RUNTIME_METADATA_KEYS = [
    "camera_names",
    "camera_heights",
    "camera_widths",
    "renderer",
    "nbody",
    "ngeom",
    "njnt",
    "nq",
    "nv",
    "nu",
    "ncam",
]
DEFAULT_SUBPROCESS_TIMEOUT_S: float | None = None
SUBPROCESS_CLEANUP_TIMEOUT_S = 5.0


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


@dataclass
class ProfileResult:
    events: list[dict[str, Any]]
    summary: dict[str, Any] | None
    error: dict[str, Any] | None


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


def _configure_libero_type(libero_type: str) -> None:
    os.environ["LIBERO_TYPE"] = libero_type


def _import_libero_modules(libero_type: str) -> tuple[Any, Any, Any]:
    _configure_libero_type(libero_type)
    if libero_type == "pro":
        from liberopro.liberopro import benchmark, get_libero_path
        from liberopro.liberopro.envs import OffScreenRenderEnv
    elif libero_type == "plus":
        from liberoplus.liberoplus import benchmark, get_libero_path
        from liberoplus.liberoplus.envs import OffScreenRenderEnv
    else:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    return benchmark, get_libero_path, OffScreenRenderEnv


def _bddl_path_for_task(get_libero_path: Any, task: Any) -> str:
    return str(
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )


def build_task_trial_specs(
    config: ProfileConfig,
) -> tuple[list[TaskTrialSpec], list[Any]]:
    benchmark, get_libero_path, _ = _import_libero_modules(config.libero_type)
    bench = benchmark.get_benchmark(config.suite)()
    task_ids = parse_task_ids(config.task_ids, num_tasks=bench.get_num_tasks())
    specs: list[TaskTrialSpec] = []
    init_states: list[Any] = []
    for task_id in task_ids:
        task = bench.get_task(task_id)
        task_init_states = bench.get_task_init_states(task_id)
        trial_ids = select_trial_ids(
            num_trials=len(task_init_states),
            trials_per_task=config.trials_per_task,
            specific_trial_ids=config.specific_trial_ids,
            seed=config.seed,
            task_id=task_id,
        )
        for trial_id in trial_ids:
            specs.append(
                TaskTrialSpec(
                    suite_name=config.suite,
                    task_id=task_id,
                    trial_id=trial_id,
                    task_name=Path(task.bddl_file).stem,
                    task_language=task.language,
                    bddl_file=_bddl_path_for_task(get_libero_path, task),
                    seed=config.seed + task_id * 100000 + trial_id,
                )
            )
            init_states.append(task_init_states[trial_id])
    return specs, init_states


def make_libero_env_factory(config: ProfileConfig, spec: TaskTrialSpec) -> Any:
    _, _, OffScreenRenderEnv = _import_libero_modules(config.libero_type)

    def factory() -> Any:
        return OffScreenRenderEnv(
            bddl_file_name=spec.bddl_file,
            camera_heights=config.camera_height,
            camera_widths=config.camera_width,
        )

    return factory


def parse_dummy_action(value: str | None) -> list[float]:
    if value is None:
        return list(DEFAULT_DUMMY_ACTION)
    stripped = value.strip()
    if not stripped:
        raise ValueError("empty dummy action")
    return [float(part.strip()) for part in stripped.split(",")]


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_to_jsonable(record), sort_keys=True) + "\n")


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    return {"__type__": type(value).__name__, "repr": repr(value)}


def _jsonable_config(config: ProfileConfig) -> dict[str, Any]:
    data = _to_jsonable(config)
    return dict(data)


def write_run_config(output_dir: Path, config: ProfileConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(_jsonable_config(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_summary_files(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "step_latency_summary.json"
    jsonable_summaries = _to_jsonable(summaries)
    json_path.write_text(
        json.dumps(jsonable_summaries, indent=2, sort_keys=True), encoding="utf-8"
    )
    csv_path = output_dir / "step_latency_summary.csv"
    fieldnames = sorted({key for summary in jsonable_summaries for key in summary})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in jsonable_summaries:
            writer.writerow(summary)


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
    match = re.match(r"([A-Z]+(?:_[A-Z]+)*_SCENE\d+)", bddl_path.stem)
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


def collect_runtime_metadata(env: Any, config: ProfileConfig) -> dict[str, Any]:
    model = getattr(getattr(env, "sim", None), "model", None)
    camera_names = getattr(model, "camera_names", None)
    metadata = {
        "camera_names": list(camera_names) if camera_names is not None else None,
        "camera_heights": config.camera_height,
        "camera_widths": config.camera_width,
        "renderer": "mujoco",
        "nbody": getattr(model, "nbody", None),
        "ngeom": getattr(model, "ngeom", None),
        "njnt": getattr(model, "njnt", None),
        "nq": getattr(model, "nq", None),
        "nv": getattr(model, "nv", None),
        "nu": getattr(model, "nu", None),
        "ncam": getattr(model, "ncam", None),
    }
    for key in RUNTIME_METADATA_KEYS:
        metadata.setdefault(key, None)
    return metadata


def _apply_cpu_affinity(cpu_id: int | None) -> bool:
    if cpu_id is None:
        return False
    if not hasattr(os, "sched_setaffinity"):
        return False
    try:
        os.sched_setaffinity(0, {cpu_id})
    except OSError:
        return False
    return True


def _step_env(env: Any, action: list[float]) -> tuple[Any, float, bool, dict[str, Any]]:
    obs, reward, done, info = env.step(np.asarray(action, dtype=np.float32))
    if info is None:
        info = {}
    return obs, float(reward), bool(done), dict(info)


def _event_base(
    *,
    config: ProfileConfig,
    spec: TaskTrialSpec,
    cpu_affinity_applied: bool,
    bddl_metadata: dict[str, Any],
    runtime_metadata: dict[str, Any],
) -> dict[str, Any]:
    base = {
        "suite_name": spec.suite_name,
        "task_id": spec.task_id,
        "trial_id": spec.trial_id,
        "task_name": spec.task_name,
        "task_language": spec.task_language,
        "cpu_id": config.cpu_id,
        "pid": os.getpid(),
        "seed": spec.seed,
        "cpu_affinity_applied": cpu_affinity_applied,
        "bddl_file": spec.bddl_file,
    }
    base.update(bddl_metadata)
    base.update(runtime_metadata)
    base["task_language"] = spec.task_language or bddl_metadata.get("task_language")
    return base


def _error_record(
    *,
    spec: TaskTrialSpec,
    message: str,
    exc_type: str,
    stage: str | None = None,
    traceback_text: str | None = None,
) -> dict[str, Any]:
    record = {
        "event": "error",
        "suite_name": spec.suite_name,
        "task_id": spec.task_id,
        "trial_id": spec.trial_id,
        "task_name": spec.task_name,
        "task_language": spec.task_language,
        "bddl_file": spec.bddl_file,
        "error_type": exc_type,
        "error": message,
    }
    if stage is not None:
        record["stage"] = stage
    if traceback_text is not None:
        record["traceback"] = traceback_text
    return record


def profile_task_trial(
    *,
    config: ProfileConfig,
    spec: TaskTrialSpec,
    env_factory: Any,
    init_state: Any,
    clock: Any = time.perf_counter,
) -> ProfileResult:
    env = None
    cpu_affinity_applied = _apply_cpu_affinity(config.cpu_id)
    stage = "apply_cpu_affinity"
    try:
        stage = "parse_bddl_metadata"
        bddl_metadata = parse_bddl_metadata(spec.bddl_file)
        stage = "env_factory"
        env = env_factory()
        if hasattr(env, "seed"):
            env.seed(spec.seed)
        stage = "reset"
        env.reset()
        if init_state is not None and hasattr(env, "set_init_state"):
            stage = "set_init_state"
            env.set_init_state(init_state)
        stage = "collect_runtime_metadata"
        runtime_metadata = collect_runtime_metadata(env, config)
        for warmup_step in range(config.warmup_steps):
            stage = f"warmup_step:{warmup_step}"
            _step_env(env, config.dummy_action)
        stage = "reset"
        env.reset()
        if init_state is not None and hasattr(env, "set_init_state"):
            stage = "set_init_state"
            env.set_init_state(init_state)
        base = _event_base(
            config=config,
            spec=spec,
            cpu_affinity_applied=cpu_affinity_applied,
            bddl_metadata=bddl_metadata,
            runtime_metadata=runtime_metadata,
        )
        events: list[dict[str, Any]] = []
        latencies: list[float] = []
        done_seen_step: int | None = None
        success_seen = False
        for step_index in range(config.measure_steps):
            stage = f"measure_step:{step_index}"
            start = clock()
            _, reward, done, info = _step_env(env, config.dummy_action)
            end = clock()
            latency_s = max(float(end - start), 0.0)
            success = bool(info.get("success", False))
            if hasattr(env, "check_success"):
                success = success or bool(env.check_success())
            if done and done_seen_step is None:
                done_seen_step = step_index
            success_seen = success_seen or success
            latencies.append(latency_s)
            event = {
                "event": "libero_step_latency",
                **base,
                "step_index": step_index,
                "latency_s": latency_s,
                "reward": reward,
                "done": done,
                "success": success,
                "done_seen_step": done_seen_step,
            }
            events.append(event)
            if done and config.stop_on_done:
                break
        summary = {
            **base,
            **compute_latency_summary(latencies),
            "done_seen_step": done_seen_step,
            "success_seen": success_seen,
            "error": None,
        }
        return ProfileResult(events=events, summary=summary, error=None)
    except Exception as exc:
        return ProfileResult(
            events=[],
            summary=None,
            error=_error_record(
                spec=spec,
                message=str(exc),
                exc_type=exc.__class__.__name__,
                stage=stage,
                traceback_text=traceback.format_exc(),
            ),
        )
    finally:
        if env is not None and hasattr(env, "close"):
            try:
                stage = "close"
                env.close()
            except Exception:
                pass


def _profile_subprocess_entry(
    child_conn: Any,
    config: ProfileConfig,
    spec: TaskTrialSpec,
    init_state: Any,
) -> None:
    try:
        result = profile_task_trial(
            config=config,
            spec=spec,
            env_factory=make_libero_env_factory(config, spec),
            init_state=init_state,
        )
    except Exception as exc:
        result = ProfileResult(
            events=[],
            summary=None,
            error=_error_record(
                spec=spec,
                message=str(exc),
                exc_type=exc.__class__.__name__,
                stage="subprocess_entry",
                traceback_text=traceback.format_exc(),
            ),
        )
    try:
        child_conn.send(result)
    finally:
        child_conn.close()


def _cleanup_timed_out_subprocess(process: Any) -> None:
    terminate = getattr(process, "terminate", None)
    if terminate is not None:
        terminate()
    process.join(SUBPROCESS_CLEANUP_TIMEOUT_S)
    is_alive = getattr(process, "is_alive", None)
    if is_alive is not None and is_alive():
        kill = getattr(process, "kill", None)
        if kill is not None:
            kill()
            process.join(SUBPROCESS_CLEANUP_TIMEOUT_S)


def profile_task_trial_in_subprocess(
    *,
    config: ProfileConfig,
    spec: TaskTrialSpec,
    init_state: Any,
    timeout_s: float | None = DEFAULT_SUBPROCESS_TIMEOUT_S,
) -> ProfileResult:
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_profile_subprocess_entry,
        args=(child_conn, config, spec, init_state),
    )
    process.start()
    child_close = getattr(child_conn, "close", None)
    if child_close is not None:
        child_close()
    try:
        if timeout_s is not None and not parent_conn.poll(timeout_s):
            _cleanup_timed_out_subprocess(process)
            return ProfileResult(
                events=[],
                summary=None,
                error=_error_record(
                    spec=spec,
                    message=(
                        f"profiling subprocess timed out after {timeout_s}s; "
                        "cleanup attempted"
                    ),
                    exc_type="SubprocessError",
                    stage="subprocess_timeout",
                ),
            )
        result = parent_conn.recv()
    except EOFError:
        process.join()
        return ProfileResult(
            events=[],
            summary=None,
            error=_error_record(
                spec=spec,
                message=(
                    "profiling subprocess produced no result "
                    f"and exited with code {process.exitcode}"
                ),
                exc_type="SubprocessError",
                stage="subprocess_result",
            ),
        )
    finally:
        parent_close = getattr(parent_conn, "close", None)
        if parent_close is not None:
            parent_close()
    process.join()
    if process.exitcode != 0 and result.error is None:
        return ProfileResult(
            events=[],
            summary=None,
            error=_error_record(
                spec=spec,
                message=f"profiling subprocess exited with code {process.exitcode}",
                exc_type="SubprocessError",
                stage="subprocess_exit",
            ),
        )
    return result


def run_profile(config: ProfileConfig) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(config.output_dir, config)
    events_path = config.output_dir / "step_latency_events.jsonl"
    errors_path = config.output_dir / "errors.jsonl"
    specs, init_states = build_task_trial_specs(config)
    summaries: list[dict[str, Any]] = []
    for spec, init_state in zip(specs, init_states):
        result = profile_task_trial_in_subprocess(
            config=config,
            spec=spec,
            init_state=init_state,
        )
        append_jsonl(events_path, result.events)
        if result.summary is not None:
            summaries.append(result.summary)
        if result.error is not None:
            append_jsonl(errors_path, [result.error])
    write_summary_files(config.output_dir, summaries)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-ids", default="all")
    parser.add_argument("--trials-per-task", type=int, default=1)
    parser.add_argument("--specific-trial-ids", default=None)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=200)
    parser.add_argument("--cpu-id", type=int, default=None)
    parser.add_argument("--cpu-ids", default=None)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument(
        "--libero-type",
        choices=["standard", "pro", "plus"],
        default="standard",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dummy-action", default=None)
    parser.add_argument("--stop-on-done", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> ProfileConfig:
    specific_trial_ids = (
        None
        if args.specific_trial_ids is None
        else parse_int_list(args.specific_trial_ids, allow_all=False)
    )
    cpu_ids = None if args.cpu_ids is None else parse_int_list(args.cpu_ids)
    if args.trials_per_task < 1:
        raise ValueError("--trials-per-task must be >= 1")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")
    if args.measure_steps < 1:
        raise ValueError("--measure-steps must be >= 1")
    if args.camera_height <= 0:
        raise ValueError("--camera-height must be > 0")
    if args.camera_width <= 0:
        raise ValueError("--camera-width must be > 0")
    if args.cpu_id is not None and args.cpu_id < 0:
        raise ValueError("--cpu-id must be >= 0")
    if cpu_ids is not None and any(cpu_id < 0 for cpu_id in cpu_ids):
        raise ValueError("--cpu-ids must be >= 0")
    return ProfileConfig(
        suite=args.suite,
        task_ids=args.task_ids,
        trials_per_task=args.trials_per_task,
        specific_trial_ids=specific_trial_ids,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        cpu_id=args.cpu_id,
        cpu_ids=cpu_ids,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        libero_type=args.libero_type,
        seed=args.seed,
        output_dir=args.output_dir,
        dummy_action=parse_dummy_action(args.dummy_action),
        stop_on_done=args.stop_on_done,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        return run_profile(config)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
