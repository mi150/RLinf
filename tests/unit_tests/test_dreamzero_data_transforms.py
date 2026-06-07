import numpy as np

from rlinf.data.datasets.dreamzero.dreamzero import DreamZeroCollator
from rlinf.data.datasets.dreamzero.data_transforms import (
    convert_rollout_env_obs,
    rollout_obs_layout_for_embodiment,
)
from rlinf.models.embodiment.dreamzero.dreamzero_policy import DreamZeroPolicy


def _obs(value: int, prompts=None):
    prompts = prompts or ["task a", "task b"]
    return {
        "main_images": np.full((2, 256, 256, 3), value, dtype=np.uint8),
        "wrist_images": np.full((2, 256, 256, 3), value + 1, dtype=np.uint8),
        "states": np.zeros((2, 8), dtype=np.float32),
        "task_descriptions": prompts,
    }


def test_convert_libero_rollout_obs_maps_images_state_and_language():
    converted = convert_rollout_env_obs("libero_sim", _obs(3))

    assert converted["video.image"].shape == (2, 1, 256, 256, 3)
    assert converted["video.wrist_image"].shape == (2, 1, 256, 256, 3)
    assert converted["state.state"].shape == (2, 1, 8)
    assert converted["annotation.task"] == ["task a", "task b"]
    assert np.all(converted["video.image"][:, 0] == 3)
    assert np.all(converted["video.wrist_image"][:, 0] == 4)


def test_convert_libero_rollout_obs_accepts_channel_first_images():
    obs = {
        "main_images": np.ones((2, 3, 16, 32), dtype=np.float32),
        "wrist_images": np.zeros((2, 3, 16, 32), dtype=np.float32),
        "states": np.zeros((2, 8), dtype=np.float32),
        "task_descriptions": "pick up the mug",
    }

    converted = convert_rollout_env_obs("libero_sim", obs)

    assert converted["video.image"].shape == (2, 1, 16, 32, 3)
    assert converted["video.image"].dtype == np.uint8
    assert converted["annotation.task"] == ["pick up the mug", "pick up the mug"]


def test_libero_rollout_layout_binarizes_gripper():
    layout = rollout_obs_layout_for_embodiment("libero_sim")

    assert layout.binarize_gripper is True
    assert ("main_images", "video.image") in layout.video_fields
    assert ("wrist_images", "video.wrist_image") in layout.video_fields


def test_dreamzero_collator_static_batch_supports_libero_sim():
    class DummyTokenizer:
        def __call__(self, texts, return_mask=False, add_special_tokens=True):
            assert "two horizontal views" in texts[0]
            ids = np.ones((len(texts), 4), dtype=np.int64)
            mask = np.ones((len(texts), 4), dtype=np.int64)
            return ids, mask

    features = [
        {
            "images": np.zeros((1, 2, 3, 4, 4), dtype=np.uint8),
            "state": np.zeros((1, 64), dtype=np.float32),
            "state_mask": np.ones((1, 64), dtype=bool),
            "action": np.zeros((16, 32), dtype=np.float32),
            "action_mask": np.ones((16, 32), dtype=bool),
            "embodiment_id": np.int64(21),
            "text": "pick up the mug",
        }
    ]

    batch = DreamZeroCollator.collate_batch(
        features,
        DummyTokenizer(),
        {"libero_sim": 21},
    )

    assert batch["images"].shape == (1, 1, 2, 3, 4, 4)
    assert batch["text"].shape == (1, 4)
    assert batch["text_attention_mask"].shape == (1, 4)


def test_dreamzero_policy_concatenates_unapplied_actions_in_dataset_order():
    policy = object.__new__(DreamZeroPolicy)
    policy._action_keys = ("action.arm", "action.gripper")

    arm = np.full((2, 16, 6), 0.25, dtype=np.float32)
    gripper = np.full((2, 16, 1), -0.5, dtype=np.float32)

    actions = policy._actions_from_unapply(
        {"action.arm": arm, "action.gripper": gripper}
    )

    assert actions.shape == (2, 16, 7)
    assert np.all(actions[..., :6] == 0.25)
    assert np.all(actions[..., -1] == -0.5)
