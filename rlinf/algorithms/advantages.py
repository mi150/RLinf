# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch

from rlinf.algorithms.registry import register_advantage
from rlinf.algorithms.utils import kl_penalty, safe_normalize
from rlinf.utils.utils import masked_mean


@register_advantage("gae")
def compute_gae_advantages_and_returns(
    rewards: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    values: Optional[torch.Tensor] = None,
    normalize_advantages: bool = True,
    normalize_returns: bool = False,
    loss_mask: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate advantages and returns for Proximal Policy Optimization (PPO).
    NOTE: currently this function does not support auto-reset.

    This function implements Generalized Advantage Estimation (GAE) to compute
    advantages and returns for PPO training. The advantages are normalized
    using mean and standard deviation for stable training.

    Args:
        rewards (torch.Tensor): Rewards per timestep. Shape: [seq_len, bsz].
        values (torch.Tensor): Value function estimates. Shape: [seq_len, bsz].
        dones (torch.Tensor): Done flags (1 if episode ended, else 0).
        gamma (float, optional): Discount factor. Defaults to 1.0.
        gae_lambda (float, optional): GAE smoothing factor. Defaults to 1.0.
        normalize_advantages (bool, optional): Whether to normalize advantages. Defaults to True.
        normalize_returns (bool, optional): Whether to normalize returns. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (advantages, returns)
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0

    critic_free = values is None
    if critic_free:
        gae_lambda = 1
        gamma = 1

    # Log episode count and length distribution seen by GAE
    if dones is not None and loss_mask is not None:
        dones_last = dones[:T, :, -1] if dones.dim() == 3 else dones[:T]
        mask_last = loss_mask[:, :, -1] if loss_mask.dim() == 3 else loss_mask
        valid_ep_ends = dones_last & mask_last
        n_gae_eps = valid_ep_ends.sum().item()
        B = rewards.shape[1] if rewards.dim() >= 2 else 1
        # Compute per-episode lengths
        ep_lens = []
        for ei in range(B):
            start = 0
            for t in range(T):
                if dones_last[t, ei] and mask_last[t, ei]:
                    ep_lens.append(t - start + 1)
                    start = t + 1
                elif dones_last[t, ei] and not mask_last[t, ei]:
                    start = t + 1
        avg_len = sum(ep_lens) / len(ep_lens) if ep_lens else 0
        # Separate success (reward > 0) and fail episodes
        # Use rewards summed per episode
        succ_lens, fail_lens = [], []
        for ei in range(B):
            start = 0
            for t in range(T):
                if dones_last[t, ei] and mask_last[t, ei]:
                    ep_reward = (
                        (
                            rewards[start : t + 1, ei]
                            * mask_last[start : t + 1, ei].unsqueeze(-1).float()
                        )
                        .sum()
                        .item()
                        if rewards.dim() == 3
                        else rewards[start : t + 1, ei].sum().item()
                    )
                    ep_len = t - start + 1
                    if ep_reward > 0:
                        succ_lens.append(ep_len)
                    else:
                        fail_lens.append(ep_len)
                    start = t + 1
                elif dones_last[t, ei] and not mask_last[t, ei]:
                    start = t + 1
        print(
            f"[GAE] T={T}, B={B}, valid_episodes={int(n_gae_eps)}, avg_len={avg_len:.1f}"
        )
        print(
            f"[GAE] success: n={len(succ_lens)}, avg_len={sum(succ_lens) / len(succ_lens):.1f}"
            if succ_lens
            else "[GAE] success: n=0"
        )
        print(
            f"[GAE] fail: n={len(fail_lens)}, avg_len={sum(fail_lens) / len(fail_lens):.1f}"
            if fail_lens
            else "[GAE] fail: n=0"
        )
        # (GAE-detail removed — see [GAE] summary above)

    for step in reversed(range(T)):
        if critic_free:
            delta = rewards[step]
        else:
            delta = (
                rewards[step]
                + gamma * values[step + 1] * (~dones[step + 1])
                - values[step]
            )

        gae = delta + gamma * gae_lambda * (~dones[step + 1]) * gae
        returns[step] = gae if critic_free else gae + values[step]

    advantages = returns - values[:-1] if not critic_free else returns

    # ── Failure advantage reweighting + diagnostic ──
    if loss_mask is not None and not critic_free:
        try:
            _T, _B = rewards.shape
            env_reward_sum = (rewards * loss_mask.float()).sum(dim=0)  # [B]
            succ_envs = env_reward_sum > 0
            fail_envs = ~succ_envs

            # Reweight cut-failure advantages: scale up to simulate full-length episode
            # A failure cut at step t has active_steps < T; full timeout would have T steps.
            # Scale factor = T / active_steps compensates the missing steps' gradient.
            reweight_enabled = kwargs.get("failure_reweight", False)
            n_reweighted = 0
            if fail_envs.any():
                fail_active_per_env = (
                    loss_mask[:, fail_envs].float().sum(dim=0)
                )  # [n_fail]
                full_length = float(_T)
                # Debug: show active_steps distribution for fail envs
                active_list = fail_active_per_env.tolist()
                n_short = sum(1 for a in active_list if 0 < a < full_length)
                print(
                    f"[reweight-debug] T={_T}, B={_B}, n_fail={fail_envs.sum().item()}, "
                    f"full_length={full_length}, n_short={n_short}, "
                    f"active_steps: min={min(active_list):.0f}, max={max(active_list):.0f}, "
                    f"mean={sum(active_list) / len(active_list):.1f}, "
                    f"dist={sorted({int(a) for a in active_list})[:10]}"
                )
            if reweight_enabled and fail_envs.any():
                for j in range(fail_envs.sum().item()):
                    actual = fail_active_per_env[j].item()
                    if 0 < actual < full_length:
                        scale = full_length / actual
                        env_idx = fail_envs.nonzero(as_tuple=True)[0][j]
                        advantages[:, env_idx] *= scale
                        returns[:, env_idx] = (
                            advantages[:, env_idx] + values[:-1, env_idx]
                        )
                        n_reweighted += 1

            # Diagnostic logging (after reweighting)
            masked_adv = advantages * loss_mask.float()
            if succ_envs.any():
                succ_adv = masked_adv[:, succ_envs]
                succ_mask = loss_mask[:, succ_envs]
                succ_active = succ_mask.sum().item()
                succ_abs_sum = succ_adv.abs().sum().item()
                succ_mean = succ_adv.sum().item() / max(succ_active, 1)
            else:
                succ_active, succ_abs_sum, succ_mean = 0, 0, 0

            if fail_envs.any():
                fail_adv = masked_adv[:, fail_envs]
                fail_mask = loss_mask[:, fail_envs]
                fail_active = fail_mask.sum().item()
                fail_abs_sum = fail_adv.abs().sum().item()
                fail_mean = fail_adv.sum().item() / max(fail_active, 1)
            else:
                fail_active, fail_abs_sum, fail_mean = 0, 0, 0

            n_succ = succ_envs.sum().item()
            n_fail = fail_envs.sum().item()
            v_mean = values[:-1][loss_mask].mean().item() if loss_mask.any() else 0
            rw_str = f", reweighted={n_reweighted}" if reweight_enabled else ""

            # ── Deep probe: 5-segment breakdown for failure & success ──
            N_SEG = 5
            seg_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]

            def _segment_stats(env_indices):
                """Per-segment stats with mean, std, and per-env distribution."""
                # Collect per-env per-segment values for distribution analysis
                per_env_seg_abs = [
                    [] for _ in range(N_SEG)
                ]  # |adv| mean per env per seg
                per_env_seg_v = [[] for _ in range(N_SEG)]
                per_env_seg_delta = [[] for _ in range(N_SEG)]
                per_env_seg_adv = [[] for _ in range(N_SEG)]

                for env_idx in env_indices:
                    env_mask = loss_mask[:, env_idx]
                    active = env_mask.sum().item()
                    if active < N_SEG:
                        continue
                    active_idx = env_mask.nonzero(as_tuple=True)[0]
                    seg_size = len(active_idx) // N_SEG
                    for s in range(N_SEG):
                        start = s * seg_size
                        end = (s + 1) * seg_size if s < N_SEG - 1 else len(active_idx)
                        idx = active_idx[start:end]
                        adv_seg = advantages[idx, env_idx]
                        val_seg = values[idx, env_idx]
                        r_seg = rewards[idx, env_idx]
                        v_next = values[idx + 1, env_idx]
                        d_next = dones[idx + 1, env_idx].float()
                        delta_seg = r_seg + gamma * v_next * (1 - d_next) - val_seg

                        per_env_seg_adv[s].append(adv_seg.mean().item())
                        per_env_seg_abs[s].append(adv_seg.abs().mean().item())
                        per_env_seg_v[s].append(val_seg.mean().item())
                        per_env_seg_delta[s].append(delta_seg.abs().mean().item())

                n_env = len(per_env_seg_abs[0]) if per_env_seg_abs[0] else 0
                result = {}
                for s in range(N_SEG):
                    if not per_env_seg_abs[s]:
                        result[s] = {
                            "adv": 0,
                            "abs": 0,
                            "abs_std": 0,
                            "abs_p90": 0,
                            "v": 0,
                            "v_std": 0,
                            "delta": 0,
                        }
                        continue
                    import numpy as _np

                    abs_arr = _np.array(per_env_seg_abs[s])
                    adv_arr = _np.array(per_env_seg_adv[s])
                    v_arr = _np.array(per_env_seg_v[s])
                    d_arr = _np.array(per_env_seg_delta[s])
                    result[s] = {
                        "adv": float(adv_arr.mean()),
                        "abs": float(abs_arr.mean()),
                        "abs_std": float(abs_arr.std()),
                        "abs_p90": float(_np.percentile(abs_arr, 90)),
                        "v": float(v_arr.mean()),
                        "v_std": float(v_arr.std()),
                        "delta": float(d_arr.mean()),
                    }
                return result, n_env

            print(
                f"[adv-diag] envs={n_succ}S/{n_fail}F, V_mean={v_mean:.4f}, "
                f"succ: steps={succ_active:.0f} adv_mean={succ_mean:+.4f} |adv|_sum={succ_abs_sum:.2f}, "
                f"fail: steps={fail_active:.0f} adv_mean={fail_mean:+.4f} |adv|_sum={fail_abs_sum:.2f}"
                f"{rw_str}"
            )

            if fail_envs.any():
                f_idx = fail_envs.nonzero(as_tuple=True)[0].tolist()
                fr, fn = _segment_stats(f_idx)
                parts = [
                    f"{seg_labels[s]}:adv={fr[s]['adv']:+.4f}|adv|={fr[s]['abs']:.4f}±{fr[s]['abs_std']:.4f}"
                    f"(p90={fr[s]['abs_p90']:.4f})V={fr[s]['v']:.4f}±{fr[s]['v_std']:.4f}|δ|={fr[s]['delta']:.4f}"
                    for s in range(N_SEG)
                ]
                print(f"[adv-probe-fail] n={fn}, " + ", ".join(parts))

            # Success segment probe omitted — focus on failure tail analysis
        except Exception:
            pass

    if normalize_advantages:
        advantages = safe_normalize(advantages, loss_mask=loss_mask)
    if normalize_returns:
        returns = safe_normalize(returns, loss_mask=loss_mask)

    return advantages, returns


@register_advantage("grpo")
def compute_grpo_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    **kwargs,
):
    """
    Compute GRPO advantages.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups, group_size]
        loss_mask (torch.Tensor): Loss mask for valid entries. Shape: [num_groups, group_size]
        group_size (int): Group size for advantage computation.

    Returns:
        torch.Tensor: advantages
    """
    grouped_rewards = rewards.view(-1, group_size)

    grouped_reward_mean = grouped_rewards.mean(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )
    grouped_reward_std = grouped_rewards.std(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )

    advantages = grouped_rewards - grouped_reward_mean
    advantages = advantages / (grouped_reward_std + 1e-6)

    advantages = (torch.zeros_like(loss_mask) + advantages.view(1, -1)) * loss_mask

    return advantages, None


@register_advantage("grpo_dynamic")
def compute_grpo_dynamic_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    idx_to_traj: list[int],
    advantage_mode: str = "turn",  # "trajectory" or "turn"
    **kwargs,
):
    """
    Compute GRPO advantages for multi-turn multi-agent scenarios.

    IMPORTANT: This function computes advantages PER QUESTION, not globally.
    - idx_to_traj maps turn_idx -> global_traj_idx (e.g., [0,0,1,1,2,2,3,3,4,4,...,15,15])
    - Trajectories 0-3 belong to question 0, 4-7 to question 1, etc.
    - We must compute GRPO separately for each question's group_size trajectories

    Two advantage computation modes:
    1. "trajectory": Trajectory-level GRPO (Method 1)
       - Compute mean/std over group_size trajectory rewards per question
       - Broadcast same advantage to all turns in a trajectory
       - Example: Q0 has 4 trajs with 1,2,3,4 turns. Compute GRPO over 4 traj rewards,
                  then assign traj0_adv to its 1 turn, traj1_adv to its 2 turns, etc.

    2. "turn": Turn-level GRPO (Method 2)
       - Compute mean/std over all turns within each question
       - Example: Q0 has 4 trajs with 1,2,3,4 turns = 10 turns total.
                  Compute GRPO over these 10 turn rewards (currently all same within traj).
       - Future-proof: works when turns have different rewards within same trajectory

    Args:
        rewards: Shape [num_sequence, 1] after preprocessing (num_sequence = total turns)
        loss_mask: Shape [seq_len, num_sequence] after preprocessing
        group_size: Number of trajectories per question (e.g., 4)
        idx_to_traj: List mapping turn_idx -> global_traj_idx
        advantage_mode: "trajectory" or "turn"

    Returns:
        advantages: Shape [seq_len, num_sequence]
    """
    num_sequence = len(idx_to_traj)

    rewards_flat = rewards.squeeze(-1)

    assert rewards_flat.numel() == num_sequence, (
        f"Rewards size mismatch: {rewards_flat.numel()} != {num_sequence}"
    )

    num_trajectories = max(idx_to_traj) + 1
    num_questions = num_trajectories // group_size
    assert num_trajectories % group_size == 0, (
        f"num_trajectories {num_trajectories} not divisible by group_size {group_size}"
    )

    turn_advantages = torch.zeros(
        num_sequence, dtype=rewards.dtype, device=rewards.device
    )

    if advantage_mode == "trajectory":
        # Aggregate turn rewards into per-trajectory rewards first.
        trajectory_rewards = torch.zeros(
            num_trajectories, dtype=rewards.dtype, device=rewards.device
        )
        trajectory_counts = torch.zeros(
            num_trajectories, dtype=torch.long, device=rewards.device
        )

        for turn_idx, traj_idx in enumerate(idx_to_traj):
            trajectory_rewards[traj_idx] += rewards_flat[turn_idx]
            trajectory_counts[traj_idx] += 1

        # Step 1: Average rewards per trajectory.
        trajectory_rewards = trajectory_rewards / trajectory_counts.clamp(min=1).float()

        # Step 2: reshape to [num_questions, group_size] for per-question GRPO.
        trajectory_rewards_grouped = trajectory_rewards.view(num_questions, group_size)

        # Step 3: compute per-question mean and std.
        per_question_mean = trajectory_rewards_grouped.mean(
            dim=-1, keepdim=True
        )  # [num_questions, 1]
        per_question_std = trajectory_rewards_grouped.std(
            dim=-1, keepdim=True
        )  # [num_questions, 1]

        # Step 4: normalize within each question group.
        normalized_trajectory_rewards = (
            trajectory_rewards_grouped - per_question_mean
        ) / (per_question_std + 1e-6)  # [num_questions, group_size]

        # Step 5: flatten back to [num_trajectories].
        normalized_trajectory_rewards = normalized_trajectory_rewards.view(-1)

        # Step 6: broadcast trajectory advantages to all turns in that trajectory.
        for turn_idx, traj_idx in enumerate(idx_to_traj):
            turn_advantages[turn_idx] = normalized_trajectory_rewards[traj_idx]

    elif advantage_mode == "turn":
        # Step 1: map each turn to its owning question.
        turn_to_question = torch.tensor(
            [idx_to_traj[i] // group_size for i in range(num_sequence)],
            dtype=torch.long,
            device=rewards.device,
        )

        # Step 2: normalize turn rewards within each question group.
        for question_idx in range(num_questions):
            question_mask = turn_to_question == question_idx
            question_turn_rewards = rewards_flat[question_mask]

            # Step 3: compute mean and std for all turns in this question.
            question_mean = question_turn_rewards.mean()
            question_std = question_turn_rewards.std()

            # Step 4: normalize turn rewards within the question.
            normalized_question_rewards = (question_turn_rewards - question_mean) / (
                question_std + 1e-6
            )

            # Step 5: write normalized turn-level advantages back.
            turn_advantages[question_mask] = normalized_question_rewards

    else:
        raise ValueError(
            f"Invalid advantage_mode: {advantage_mode}. Must be 'trajectory' or 'turn'"
        )

    advantages = torch.zeros_like(
        loss_mask, dtype=rewards.dtype
    ) + turn_advantages.view(1, -1)
    advantages = advantages * loss_mask

    return advantages, None


@register_advantage("reinpp")
def compute_reinpp_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    use_reinpp_baseline: bool = False,
    kl_beta: float = 0.0,
    logprob=None,
    ref_logprob=None,
    kl_penalty_type: str = "",
    **kwargs,
):
    """
    Compute advantages for reinforce++ and reinforce++ baseline.

    Args:
        rewards (torch.Tensor): The reward or score values.
        loss_mask (torch.Tensor): The loss mask for valid entries.
        group_size (int): The group size for advantage computation.
        use_reinpp_baseline (bool, optional): Whether to use reinforce++ baseline.
        kl_beta (float, optional): KL penalty coefficient.
        logprob (optional): Log probability of current policy.
        ref_logprob (optional): Log probability of reference policy.
        kl_penalty_type (str, optional): Type of KL penalty.

    Returns:
        torch.Tensor: advantages
    """
    # first group baseline for reinforce++ baseline
    if use_reinpp_baseline:
        grouped_rewards = rewards.view(-1, group_size)  # [num_prompt, group_size]
        grouped_rewards -= grouped_rewards.mean(dim=1, keepdims=True)
        rewards = grouped_rewards.view(-1)  # [B]

    # build the reward matrix
    r_matrix = torch.zeros_like(loss_mask).float()  # [L, B]
    seq_length = loss_mask.size(0)
    mask_flipped = loss_mask.long().fliplr()
    eos_positions = mask_flipped.argmax(
        dim=0, keepdim=True
    )  # position of last True in original mask
    eos_indices = seq_length - 1 - eos_positions  # [1, B]

    r_matrix = r_matrix.scatter_(dim=0, index=eos_indices, src=rewards)  # [L, B]

    # add kl penalty
    if kl_beta > 0:
        kld = kl_penalty(logprob, ref_logprob, kl_penalty=kl_penalty_type)  # [L, B]
        r_matrix -= kl_beta * kld

    # compute return
    ret_matrix = torch.cumsum(r_matrix.flip(dims=[0]), dim=0).flip(dims=[0])

    # normalize
    advantages = ret_matrix.clone()

    mean = masked_mean(advantages, loss_mask)
    var = masked_mean((advantages - mean).pow(2), loss_mask)
    rstd = var.clamp(min=1e-8).rsqrt()

    advantages = (advantages - mean) * rstd

    return advantages, None


@register_advantage("raw")
def compute_raw_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    normalize_advantages: bool = False,
    **kwargs,
):
    """
    Return raw rewards or normalized rewards.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups, group_size]
        loss_mask (torch.Tensor): Loss mask for valid entries. Shape: [num_groups, group_size]
        normalize_advantages (bool): Whether to normalize advantages.

    Returns:
        torch.Tensor: advantages
    """
    if rewards.ndim == 2:
        rewards = rewards.reshape(-1)
    advantages = rewards.unsqueeze(0).expand_as(loss_mask) * loss_mask

    # Simple baseline subtraction (mean of valid advantages)
    if normalize_advantages:
        valid = advantages[loss_mask.bool()]
        if valid.numel() > 0:
            advantages = (advantages - valid.mean()) / (valid.std() + 1e-5)

    return advantages, None
