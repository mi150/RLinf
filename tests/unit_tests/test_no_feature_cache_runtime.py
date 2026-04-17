"""Guard against reintroducing runtime feature-cache integrations."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

RUNTIME_FILES = (
    "rlinf/models/embodiment/gr00t/gr00t_action_model.py",
    "rlinf/models/embodiment/openpi/openpi_action_model.py",
    "rlinf/models/embodiment/openvla_oft/official/openvla_oft_action_model.py",
    "rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py",
    "rlinf/workers/rollout/hf/huggingface_worker.py",
    "rlinf/workers/rollout/hf/async_huggingface_worker.py",
    "examples/embodiment/config/maniskill_ppo_openpi.yaml",
)

CACHE_ONLY_FILES = (
    "rlinf/models/embodiment/feature_cache.py",
    "examples/embodiment/config/libero_spatial_ppo_gr00t_feature_cache_similarity.yaml",
)


def test_runtime_files_do_not_reference_feature_cache():
    offenders = []
    for relative_path in RUNTIME_FILES:
        path = REPO_ROOT / relative_path
        if "feature_cache" in path.read_text(encoding="utf-8"):
            offenders.append(relative_path)

    assert offenders == []


def test_cache_only_files_are_removed():
    existing_files = [
        relative_path
        for relative_path in CACHE_ONLY_FILES
        if (REPO_ROOT / relative_path).exists()
    ]

    assert existing_files == []
