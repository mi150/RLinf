import torch

from rlinf.data.embodied_io_struct import ChunkStepResult, EmbodiedRolloutResult


def _all_tensors_contiguous_in_dict(data: dict) -> bool:
    for value in data.values():
        if isinstance(value, torch.Tensor):
            if not value.is_contiguous():
                return False
        elif isinstance(value, dict):
            if not _all_tensors_contiguous_in_dict(value):
                return False
    return True


def test_to_splited_trajectories_returns_contiguous_tensors():
    rollout = EmbodiedRolloutResult(max_episode_length=8)
    bsz = 4

    # Create non-contiguous tensors via transpose/view-like operations.
    actions = torch.randn(bsz, 6).transpose(0, 1).transpose(0, 1)
    prev_logprobs = torch.randn(bsz, 2, 3).transpose(1, 2)
    prev_values = torch.randn(bsz, 2).transpose(0, 1).transpose(0, 1)
    rewards = torch.randn(bsz, 2).transpose(1, 0).transpose(1, 0)
    dones = torch.zeros(bsz, 2, dtype=torch.bool).transpose(1, 0).transpose(1, 0)
    versions = torch.ones(bsz, 2).transpose(0, 1).transpose(0, 1)

    forward_inputs = {
        "action": torch.randn(bsz, 6).transpose(0, 1).transpose(0, 1),
        "model_action": torch.randn(bsz, 6).transpose(0, 1).transpose(0, 1),
    }
    chunk = ChunkStepResult(
        actions=actions,
        prev_logprobs=prev_logprobs,
        prev_values=prev_values,
        rewards=rewards,
        dones=dones,
        terminations=dones,
        truncations=dones,
        versions=versions,
        forward_inputs=forward_inputs,
    )
    rollout.append_step_result(chunk)

    splited = rollout.to_splited_trajectories(split_size=2)
    assert len(splited) == 2
    for traj in splited:
        assert traj.actions is None or traj.actions.is_contiguous()
        assert traj.prev_logprobs is None or traj.prev_logprobs.is_contiguous()
        assert traj.prev_values is None or traj.prev_values.is_contiguous()
        assert traj.rewards is None or traj.rewards.is_contiguous()
        assert traj.versions is None or traj.versions.is_contiguous()
        if traj.forward_inputs:
            assert _all_tensors_contiguous_in_dict(traj.forward_inputs)
