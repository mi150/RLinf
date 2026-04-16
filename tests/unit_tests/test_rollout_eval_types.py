import torch

from toolkits.rollout_eval.rollout_types import infer_batch_size


def test_infer_batch_size_from_tensor_dict() -> None:
    batch = infer_batch_size({"states": torch.zeros(6, 4), "images": torch.zeros(6, 8, 8, 3)})
    assert batch == 6


def test_infer_batch_size_raises_for_empty_obs() -> None:
    try:
        infer_batch_size({})
    except ValueError as exc:
        assert "Cannot infer batch size" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty observation")
