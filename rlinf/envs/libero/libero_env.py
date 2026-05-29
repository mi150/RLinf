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

import copy
import glob
import importlib
import logging
import os
import sys
import threading
from typing import Optional, Union

import gym
import numpy as np
import torch
from omegaconf.omegaconf import OmegaConf

logger = logging.getLogger(__name__)


# ── Whole Probe Model (LSTM+MLP, ~24K params) ─────────────────────────
class _WholeModelWithLang(torch.nn.Module):
    """Standard Whole: language-conditioned LSTM h0/c0 + dropout.
    Architecture: Qwen3-Embedding → LangProject → LSTM(h0,c0) → dropout → MLP → sigmoid.
    """

    def __init__(
        self,
        input_dim=11,
        hidden_dim=32,
        mlp_dims=(16, 8),
        lang_dim=1024,
        bottleneck=16,
        dropout=0.4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lang_proj = torch.nn.Sequential(
            torch.nn.Linear(lang_dim, bottleneck),
            torch.nn.ReLU(),
        )
        self.h0_proj = torch.nn.Linear(bottleneck, hidden_dim)
        self.c0_proj = torch.nn.Linear(bottleneck, hidden_dim)
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + input_dim, mlp_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_dims[0], mlp_dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_dims[1], 1),
        )

    def _init_hidden(self, batch_size, lang_emb=None):
        if lang_emb is not None:
            proj = self.lang_proj(lang_emb)
            h0 = self.h0_proj(proj).unsqueeze(0)
            c0 = self.c0_proj(proj).unsqueeze(0)
            return (h0, c0)
        device = next(self.parameters()).device
        return (
            torch.zeros(1, batch_size, self.hidden_dim, device=device),
            torch.zeros(1, batch_size, self.hidden_dim, device=device),
        )

    def forward(self, x, hc=None, lang_emb=None):
        if hc is None:
            hc = self._init_hidden(x.shape[0], lang_emb)
        out, hc = self.lstm(x, hc)
        h = self.dropout(out[:, -1, :])
        combined = torch.cat([h, x[:, -1, :]], dim=-1)
        logit = self.head(combined)
        prob = torch.sigmoid(logit).squeeze(-1)
        return prob, hc

    def forward_sequence(self, x, lengths, lang_emb=None):
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

        B = x.shape[0]
        hc = self._init_hidden(B, lang_emb)
        packed = pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed, hc)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.dropout(out)
        T_pad = out.shape[1]
        combined = torch.cat([out, x[:, :T_pad, :]], dim=-1)
        logit = self.head(combined).squeeze(-1)
        return torch.sigmoid(logit)


class _WholeModel(torch.nn.Module):
    """Lightweight failure detector: 11 features (10 + t/T) → P(fail).
    Architecture matches run_whole_10feat_modelcall_nolang.py: LSTM(11→32) + MLP(32+11→16→8→1).
    """

    def __init__(self, input_dim=11, hidden_dim=32, mlp_dims=(16, 8)):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # MLP takes concatenation of LSTM hidden + input: (hidden_dim + input_dim)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + input_dim, mlp_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_dims[0], mlp_dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_dims[1], 1),
        )

    def forward(self, x, hc=None):
        # x: (B, 1, input_dim), hc: tuple of (1, B, H)
        out, hc = self.lstm(x, hc)
        h = out[:, -1, :]  # (B, H)
        combined = torch.cat([h, x[:, -1, :]], dim=-1)  # (B, H+input_dim)
        logit = self.head(combined)  # (B, 1)
        prob = torch.sigmoid(logit).squeeze(-1)  # (B,)
        return prob, hc

    def forward_sequence(self, x, lengths):
        """Full-sequence forward for training. x: (B, T, D), lengths: (B,)"""
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

        packed = pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        T_pad = out.shape[1]
        combined = torch.cat([out, x[:, :T_pad, :]], dim=-1)
        logit = self.head(combined).squeeze(-1)
        return torch.sigmoid(logit)


class ActProbe:
    """Manages Whole model inference + sliding window for per-env failure detection."""

    def __init__(self, cfg, num_envs, device="cpu"):
        self.num_envs = num_envs
        self.device = device
        self.K = cfg.get("K", 3)
        self.agg = cfg.get("agg", "median")
        self.max_chunk_steps = cfg.get("max_chunk_steps", 44)  # 220/5
        self._min_cut_chunk = cfg.get("min_cut_chunk", 0)

        # Load checkpoint
        ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_tau = ckpt.get("sw_threshold", ckpt.get("tau", 0.5))
        self.tau = cfg.get("initial_tau", ckpt_tau)  # override if specified
        self.norm_mean = np.asarray(ckpt["norm_mean"], dtype=np.float32)  # (11,) includes t/T
        self.norm_std = np.asarray(ckpt["norm_std"], dtype=np.float32)  # (11,)
        if self.tau != ckpt_tau:
            logger.info(
                f"[ActProbe] loaded checkpoint, tau overridden: {ckpt_tau:.4f} → {self.tau:.4f}"
            )
        else:
            logger.info(f"[ActProbe] loaded checkpoint, tau={self.tau:.4f}")

        # Build model
        lang_cfg = cfg.get("lang_encoder", None)
        self._use_lang = lang_cfg is not None and lang_cfg.get("enabled", False)
        self._lang_emb = None
        self._lang_dim = 1024
        self._dropout = cfg.get("dropout", 0.0)

        if self._use_lang:
            self._lang_dim = lang_cfg.get("lang_dim", 1024)
            self.model = _WholeModelWithLang(
                input_dim=11,
                hidden_dim=32,
                mlp_dims=(16, 8),
                lang_dim=self._lang_dim,
                bottleneck=16,
                dropout=self._dropout,
            )
            # Load language model and compute embedding (once, frozen)
            task_instruction = lang_cfg.get(
                "task_instruction",
                "pick up the salad dressing and place it in the basket",
            )
            lang_model_path = lang_cfg.get("model_path", "")
            try:
                from transformers import AutoModel, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    lang_model_path, trust_remote_code=True
                )
                lang_model = AutoModel.from_pretrained(
                    lang_model_path, trust_remote_code=True
                )
                lang_model.eval()
                inputs = tokenizer(
                    [task_instruction],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                with torch.no_grad():
                    outputs = lang_model(**inputs)
                    self._lang_emb = outputs.last_hidden_state.mean(
                        dim=1
                    )  # (1, lang_dim)
                del lang_model, tokenizer
                logger.info(
                    f"[ActProbe] lang encoder loaded from {lang_model_path}, emb dim={self._lang_emb.shape[1]}"
                )
            except Exception as e:
                logger.warning(
                    f"[ActProbe] lang encoder failed: {e}, using zero init"
                )
                self._lang_emb = None
        else:
            self.model = _WholeModel(input_dim=11, hidden_dim=32, mlp_dims=(16, 8))

        # Load checkpoint weights (skip mismatched keys for architecture switch)
        try:
            self.model.load_state_dict(ckpt["model_state_dict"])
        except RuntimeError:
            logger.info(
                "[ActProbe] checkpoint architecture mismatch, starting with random weights"
            )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # ── Online update (background retraining) ──
        ou_cfg = cfg.get("online_update", {})
        self._ou_enabled = ou_cfg.get("enabled", False) if ou_cfg else False
        if self._ou_enabled:
            self._ou_interval = ou_cfg.get("interval", 5)
            self._ou_buffer_size = ou_cfg.get("buffer_size", 400)
            self._ou_recent_steps = ou_cfg.get(
                "recent_steps", 0
            )  # 0 = use all (legacy)
            self._ou_freeze_after_step = ou_cfg.get(
                "freeze_after_step", 0
            )  # 0 = never freeze
            self._ou_buffer_frozen = False
            self._ou_epochs = ou_cfg.get("epochs", 200)
            self._ou_min_episodes = ou_cfg.get("min_episodes", 50)
            self._ou_buffer = []  # list of {"feat": (T,11), "length": int, "success": bool, "step": int}
            self._ou_thread = None
            self._ou_pending = None  # dict ready to swap in
            self._ou_step = 0
            self._ou_n_updates = 0
            # Load SFT base data to ensure fail examples are always available
            self._ou_max_base = ou_cfg.get("max_base_episodes", 0)  # 0 = use all
            self._ou_base_data = self._load_base_data(
                ou_cfg.get("base_data_path", None), self._ou_max_base
            )
            n_base_s = sum(1 for e in self._ou_base_data if e["success"])
            n_base_f = len(self._ou_base_data) - n_base_s
            logger.info(
                f"[ActProbe] online update ON: interval={self._ou_interval}, "
                f"buffer={self._ou_buffer_size}, recent_steps={self._ou_recent_steps}, "
                f"epochs={self._ou_epochs}, base_data={n_base_s}S/{n_base_f}F"
            )

        # ── Resume from saved probe state (v16+) ──
        resume_path = cfg.get("resume_state_path", None)
        if resume_path:
            self._load_probe_state(resume_path)

        self.reset_all()

    def save_probe_state(self, path):
        """Save probe state (model + buffer + tau) for resume after warmup."""
        import pickle

        # Wait for any pending retrain to finish
        if (
            self._ou_enabled
            and self._ou_thread is not None
            and self._ou_thread.is_alive()
        ):
            self._ou_thread.join(timeout=60)
        # Apply pending if available
        self.apply_pending()
        state = {
            "model_state_dict": self.model.state_dict(),
            "norm_mean": self.norm_mean,
            "norm_std": self.norm_std,
            "tau": self.tau,
            "ou_step": self._ou_step if self._ou_enabled else 0,
            "ou_n_updates": self._ou_n_updates if self._ou_enabled else 0,
            "ou_buffer": list(self._ou_buffer) if self._ou_enabled else [],
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        n_buf = len(state["ou_buffer"])
        n_s = sum(1 for e in state["ou_buffer"] if e["success"])
        logger.info(
            f"[ActProbe] state saved to {path}: tau={self.tau:.4f}, buffer={n_s}S/{n_buf - n_s}F"
        )

    def _load_probe_state(self, path):
        """Load saved probe state (model + buffer + tau) for resume."""
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)
        try:
            self.model.load_state_dict(state["model_state_dict"])
        except RuntimeError:
            logger.info(
                "[ActProbe] resume: architecture mismatch, keeping current weights (will retrain)"
            )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.norm_mean = np.asarray(state["norm_mean"], dtype=np.float32)
        self.norm_std = np.asarray(state["norm_std"], dtype=np.float32)
        self.tau = state["tau"]
        if self._ou_enabled:
            self._ou_step = state.get("ou_step", 0)
            self._ou_n_updates = state.get("ou_n_updates", 0)
            self._ou_buffer = state.get("ou_buffer", [])
        n_buf = len(state.get("ou_buffer", []))
        n_s = sum(1 for e in state.get("ou_buffer", []) if e["success"])
        logger.info(
            f"[ActProbe] resumed from {path}: tau={self.tau:.4f}, "
            f"ou_step={state.get('ou_step', 0)}, buffer={n_s}S/{n_buf - n_s}F"
        )

    def reset_env_hidden(self, env_idx):
        """Reset LSTM hidden state for a single env (after done/cut in auto_reset mode).
        For lang-conditioned model, re-initialize with lang embedding."""
        if self.hc is None:
            return
        if self._use_lang and self._lang_emb is not None:
            with torch.no_grad():
                proj = self.model.lang_proj(self._lang_emb[0:1])  # (1, bottleneck)
                h0 = self.model.h0_proj(proj)  # (1, H)
                c0 = self.model.c0_proj(proj)  # (1, H)
                self.hc[0][:, env_idx, :] = h0
                self.hc[1][:, env_idx, :] = c0
        else:
            self.hc[0][:, env_idx, :] = 0
            self.hc[1][:, env_idx, :] = 0

    def reset_all(self):
        """Reset all per-env state (call at epoch start)."""
        self.hc = None  # LSTM hidden: lazy init on first forward
        self.window_buf = [[] for _ in range(self.num_envs)]
        self.chunk_idx = 0
        self.predicted_fail = np.zeros(self.num_envs, dtype=bool)
        self.trigger_step = np.full(self.num_envs, -1, dtype=np.int32)
        self._env_ep_chunk = np.zeros(
            self.num_envs, dtype=np.int32
        )  # per-env episode chunk counter
        # Per-env score history for diagnostics
        self._score_history = [[] for _ in range(self.num_envs)]
        self._completed_score_histories = []  # (scores, outcome, length)
        # Per-env feature accumulator for online update
        if self._ou_enabled:
            self._ou_feat_buf = [[] for _ in range(self.num_envs)]

    def extract_features(self, chunk_actions, obs_list, denoising_curvature=None):
        """Extract 10 model-call level features from a single chunk.

        Args:
            chunk_actions: (num_envs, chunk_size, action_dim) numpy
            obs_list: list of extracted_obs dicts (len=chunk_size),
                      each obs has 'state' (num_envs, 8) = [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]
            denoising_curvature: (num_envs,) numpy or None
        Returns:
            features: (num_envs, 10) numpy
        """
        num_envs = chunk_actions.shape[0]
        chunk_size = chunk_actions.shape[1]

        # 1. action_norm_mean
        action_norms = np.linalg.norm(chunk_actions, axis=-1)  # (E, C)
        action_norm_mean = action_norms.mean(axis=1)  # (E,)

        # Get states from obs_list: each obs has 'states' key
        # states may be a list of tensors (E, 8) or a stacked tensor
        states_list = []
        for obs in obs_list:
            s = obs["states"]
            if isinstance(s, (list, tuple)):
                s = torch.stack(s) if isinstance(s[0], torch.Tensor) else np.stack(s)
            if isinstance(s, torch.Tensor):
                s = s.cpu().numpy()
            states_list.append(s)
        states = np.stack(states_list, axis=1)  # (E, C, 8)

        # 2-3. gripper_qpos_last (2 values)
        gripper_qpos_last = states[:, -1, 6:8]  # (E, 2)

        # 4. gripper_oscillation: sign flips in gripper diff
        gripper_diff = np.diff(states[:, :, 6], axis=1)  # (E, C-1)
        signs = np.sign(gripper_diff)
        sign_changes = np.abs(np.diff(signs, axis=1))  # (E, C-2)
        gripper_oscillation = (sign_changes > 0).sum(axis=1).astype(np.float32)  # (E,)

        # 5-7. eef_pos_last (3 values)
        eef_pos_last = states[:, -1, :3]  # (E, 3)

        # 8. eef_speed_mean (within-chunk diff)
        eef_pos = states[:, :, :3]  # (E, C, 3)
        eef_diff = np.diff(eef_pos, axis=1)  # (E, C-1, 3)
        eef_speed = np.linalg.norm(eef_diff, axis=-1)  # (E, C-1)
        eef_speed_mean = eef_speed.mean(axis=1)  # (E,)

        # 9. action_jerk_mean (within-chunk 2nd order diff)
        action_diff = np.diff(action_norms, axis=1)  # (E, C-1)
        action_jerk = np.abs(np.diff(action_diff, axis=1))  # (E, C-2)
        action_jerk_mean = (
            action_jerk.mean(axis=1) if action_jerk.shape[1] > 0 else np.zeros(num_envs)
        )

        # 10. denoising_curvature
        if denoising_curvature is None:
            denoising_curvature = np.zeros(num_envs, dtype=np.float32)

        features = np.stack(
            [
                action_norm_mean,
                gripper_qpos_last[:, 0],
                gripper_qpos_last[:, 1],
                gripper_oscillation,
                eef_pos_last[:, 0],
                eef_pos_last[:, 1],
                eef_pos_last[:, 2],
                eef_speed_mean,
                action_jerk_mean,
                denoising_curvature,
            ],
            axis=1,
        )  # (E, 10)

        return features

    def predict(self, features):
        """Run probe inference for one chunk step.

        Args:
            features: (num_envs, 10) numpy
        Returns:
            prob_fail: (num_envs,) numpy, P(fail) per env
            newly_flagged: (num_envs,) bool numpy, envs newly flagged as fail this step
        """
        # Store raw features for online update (before normalization, before t/T)
        if self._ou_enabled:
            for i in range(self.num_envs):
                self._ou_feat_buf[i].append(features[i].copy())

        self.chunk_idx += 1
        self._env_ep_chunk += 1  # increment before t/T to match collect_episode's t_frac=[1/48, 2/48, ...]
        # Use per-env episode chunk counter for t/T (not global chunk_idx)
        # This ensures t/T resets to 0 for each new episode in auto_reset mode
        t_over_T_per_env = self._env_ep_chunk / self.max_chunk_steps  # (num_envs,)
        # Append per-env t/T to features, then normalize all 11 dims together
        t_feat = t_over_T_per_env.reshape(-1, 1).astype(np.float32)  # (E, 1)
        features_with_t = np.concatenate([features, t_feat], axis=1)  # (E, 11)
        x = (features_with_t - self.norm_mean) / (self.norm_std + 1e-8)  # (E, 11)
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # (E, 1, 11)

        with torch.no_grad():
            if self._use_lang and self._lang_emb is not None and self.hc is None:
                _lang = self._lang_emb.expand(x_t.shape[0], -1)
                prob, self.hc = self.model(x_t, self.hc, lang_emb=_lang)
            else:
                prob, self.hc = self.model(x_t, self.hc)
        prob_np = prob.numpy()  # (E,)

        # Record score history
        for i in range(self.num_envs):
            self._score_history[i].append(float(prob_np[i]))

        # Update sliding window (skip flagging for envs below min_cut_chunk)
        _min_cut = getattr(self, "_min_cut_chunk", 0)
        newly_flagged = np.zeros(self.num_envs, dtype=bool)
        for i in range(self.num_envs):
            if self.predicted_fail[i]:
                continue  # already flagged
            if _min_cut > 0 and self._env_ep_chunk[i] < _min_cut:
                continue  # too early in episode, don't flag
            self.window_buf[i].append(prob_np[i])
            if len(self.window_buf[i]) > self.K:
                self.window_buf[i] = self.window_buf[i][-self.K :]
            if len(self.window_buf[i]) == self.K:
                if self.agg == "median":
                    val = np.median(self.window_buf[i])
                else:
                    val = np.mean(self.window_buf[i])
                if val > self.tau:
                    self.predicted_fail[i] = True
                    self.trigger_step[i] = self.chunk_idx
                    newly_flagged[i] = True

        return prob_np, newly_flagged

    # ── Online update methods ────────────────────────────────────

    def _load_base_data(self, path, max_episodes=0):
        """Load SFT rollout data as base for online retraining (ensures fail examples)."""
        if path is None:
            return []
        import pickle

        try:
            with open(path, "rb") as f:
                episodes = pickle.load(f)
        except Exception as e:
            logger.warning(f"[ActProbe] failed to load base data from {path}: {e}")
            return []
        CHUNK_SIZE = 10
        base = []
        for ep in episodes:
            T = ep["length"]
            an = np.array(ep["action_norm"], dtype=np.float32)[:T]
            gq = np.array(ep["gripper_qpos"], dtype=np.float32)[:T]
            eef = np.array(ep["eef_pos"], dtype=np.float32)[:T]
            dc = np.array(ep.get("denoising_curvature", []), dtype=np.float32)
            if gq.ndim == 1:
                gq = gq[:, None]
            M = len(ep.get("action_chunk", [0] * (T // CHUNK_SIZE)))
            feat = np.zeros((M, 10), dtype=np.float32)
            for i in range(M):
                s, e = i * CHUNK_SIZE, min((i + 1) * CHUNK_SIZE, T)
                if s >= T:
                    break
                feat[i, 0] = an[s:e].mean()
                feat[i, 1], feat[i, 2] = gq[e - 1, 0], gq[e - 1, 1]
                gc = gq[s:e, 0]
                if len(gc) >= 2:
                    sg = np.sign(np.diff(gc))
                    sg = sg[sg != 0]
                    if len(sg) >= 2:
                        feat[i, 3] = float(np.sum(sg[1:] != sg[:-1]))
                feat[i, 4:7] = eef[e - 1]
                if e - s >= 2:
                    feat[i, 7] = np.linalg.norm(
                        np.diff(eef[s:e], axis=0), axis=1
                    ).mean()
                cn = an[s:e]
                if len(cn) >= 3:
                    feat[i, 8] = np.abs(cn[2:] - 2 * cn[1:-1] + cn[:-2]).mean()
                if i < len(dc):
                    feat[i, 9] = dc[i]
            feat = feat[:M]
            t_frac = np.arange(1, M + 1, dtype=np.float32) / self.max_chunk_steps
            feat_with_t = np.concatenate([feat, t_frac[:, None]], axis=1)
            base.append({"feat": feat_with_t, "length": M, "success": ep["success"]})
        # Subsample if max_episodes is set (stratified: keep S/F ratio)
        if max_episodes > 0 and len(base) > max_episodes:
            rng = np.random.RandomState(42)
            succ = [e for e in base if e["success"]]
            fail = [e for e in base if not e["success"]]
            n_succ = max(1, int(max_episodes * len(succ) / len(base)))
            n_fail = max_episodes - n_succ
            rng.shuffle(succ)
            rng.shuffle(fail)
            base = succ[:n_succ] + fail[:n_fail]
        return base

    def collect_episode(self, env_idx, is_success):
        """Collect a single completed episode's features into the online buffer.

        Called immediately when an env finishes (success or timeout) in auto_reset mode.
        Clears the feature buffer for that env afterwards.

        Args:
            env_idx: int — which env just completed
            is_success: bool — True if episode ended with success (termination)
        """
        # Save score history for this episode
        if self._score_history[env_idx]:
            self._completed_score_histories.append(
                {
                    "scores": list(self._score_history[env_idx]),
                    "outcome": "success" if is_success else "timeout",
                    "length": len(self._score_history[env_idx]),
                }
            )
        self._score_history[env_idx] = []
        self._env_ep_chunk[env_idx] = 0

        if not self._ou_enabled:
            return
        # Skip collection if buffer is frozen (freeze_after_step reached)
        if self._ou_buffer_frozen:
            self._ou_feat_buf[env_idx] = []
            return
        feat_seq = self._ou_feat_buf[env_idx]
        if len(feat_seq) >= 3:
            feat_arr = np.stack(feat_seq, axis=0)  # (T, 10)
            T = len(feat_seq)
            t_frac = np.arange(1, T + 1, dtype=np.float32) / self.max_chunk_steps
            feat_with_t = np.concatenate([feat_arr, t_frac[:, None]], axis=1)  # (T, 11)
            self._ou_buffer.append(
                {
                    "feat": feat_with_t,
                    "length": T,
                    "success": bool(is_success),
                    "step": self._ou_step,
                }
            )
        self._ou_feat_buf[env_idx] = []

    def trim_buffer(self):
        """Trim the online buffer by recent_steps or buffer_size. Call at PPO step end."""
        if not self._ou_enabled:
            return
        # Check if buffer should be frozen after this step
        if (
            self._ou_freeze_after_step > 0
            and self._ou_step >= self._ou_freeze_after_step
            and not self._ou_buffer_frozen
        ):
            self._ou_buffer_frozen = True
            n_s = sum(1 for e in self._ou_buffer if e["success"])
            n_f = len(self._ou_buffer) - n_s
            print(
                f"[probe] buffer FROZEN at step {self._ou_step}: {len(self._ou_buffer)} eps ({n_s}S/{n_f}F)"
            )
            return  # don't trim, keep all data up to freeze point
        if self._ou_buffer_frozen:
            return  # buffer frozen, no trimming
        if self._ou_recent_steps > 0:
            min_step = max(0, self._ou_step - self._ou_recent_steps + 1)
            self._ou_buffer = [e for e in self._ou_buffer if e["step"] >= min_step]
        elif len(self._ou_buffer) > self._ou_buffer_size:
            self._ou_buffer = self._ou_buffer[-self._ou_buffer_size :]

    def finalize_epoch(self, natural_done, env_done_step=None, is_warmup=False):
        """Called at epoch end. Package per-env feature sequences + labels into buffer.

        Args:
            natural_done: (num_envs,) bool — True for envs that succeeded naturally.
            env_done_step: (num_envs,) int32 — chunk index where env finished (-1 if never).
                Used to truncate stale features after env is done.
            is_warmup: bool — if True, don't skip probe-flagged envs (no actual cutting happened).
        """
        if not self._ou_enabled:
            return
        for i in range(self.num_envs):
            feat_seq = self._ou_feat_buf[i]
            if len(feat_seq) < 3:
                continue
            # Skip envs cut by probe — we don't know their true label
            # But during warmup, no actual cutting happened, so all labels are valid
            if not is_warmup and not natural_done[i] and self.predicted_fail[i]:
                continue
            # Truncate to actual episode length (env keeps returning stale obs after done)
            if env_done_step is not None and env_done_step[i] >= 0:
                actual_len = env_done_step[i] + 1  # done_step is 0-indexed chunk idx
                feat_seq = feat_seq[:actual_len]
            if len(feat_seq) < 3:
                continue
            feat_arr = np.stack(feat_seq, axis=0)  # (T, 10)
            T = len(feat_seq)
            t_frac = np.arange(1, T + 1, dtype=np.float32) / self.max_chunk_steps
            feat_with_t = np.concatenate([feat_arr, t_frac[:, None]], axis=1)  # (T, 11)
            self._ou_buffer.append(
                {
                    "feat": feat_with_t,
                    "length": T,
                    "success": bool(natural_done[i]),
                    "step": self._ou_step,
                }
            )
        # Keep buffer bounded: by recent_steps if set, otherwise by buffer_size
        if self._ou_recent_steps > 0:
            min_step = max(0, self._ou_step - self._ou_recent_steps + 1)
            self._ou_buffer = [e for e in self._ou_buffer if e["step"] >= min_step]
        elif len(self._ou_buffer) > self._ou_buffer_size:
            self._ou_buffer = self._ou_buffer[-self._ou_buffer_size :]

    def apply_pending(self):
        """Apply pending retrain weights if available. Called at step start to avoid 1-step lag."""
        if not self._ou_enabled or self._ou_pending is None:
            return
        pending = self._ou_pending
        self._ou_pending = None
        self.model.load_state_dict(pending["model_state_dict"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.norm_mean = pending["norm_mean"]
        self.norm_std = pending["norm_std"]
        self.tau = pending["tau"]
        self._ou_n_updates += 1
        n_succ = pending["n_success"]
        n_fail = pending["n_fail"]
        print(
            f"[probe] online update #{self._ou_n_updates} applied @ step {self._ou_step}: "
            f"tau={self.tau:.4f}, data={n_succ}S/{n_fail}F"
        )

    def maybe_retrain(self):
        """Called at step end. Trigger retrain if due."""
        if not self._ou_enabled:
            return
        self._ou_step += 1

        # Apply any pending weights (in case apply_pending wasn't called)
        self.apply_pending()

        # Trigger new retrain if interval reached
        if self._ou_step % self._ou_interval == 0:
            if self._ou_thread is None or not self._ou_thread.is_alive():
                buf = list(self._ou_buffer)  # snapshot
                if len(buf) + len(self._ou_base_data) >= self._ou_min_episodes:
                    self._ou_thread = threading.Thread(
                        target=self._retrain_worker, args=(buf,), daemon=True
                    )
                    self._ou_thread.start()
                    n_s = sum(1 for e in buf if e["success"])
                    print(
                        f"[probe] retrain triggered @ step {self._ou_step}, "
                        f"{len(buf)} eps ({n_s}S/{len(buf) - n_s}F)"
                    )

    def _retrain_worker(self, online_episodes):
        """Background thread: retrain probe on base + online data (~2-4s on CPU)."""
        import copy as _copy
        import time as _time

        # Prepare lang embedding for training (expand to max batch size)
        _lang_emb = self._lang_emb if self._use_lang else None

        def _fwd_seq(m, f, l):
            """Wrapper: forward_sequence with optional lang_emb."""
            if _lang_emb is not None:
                _le = _lang_emb.expand(f.shape[0], -1)
                return m.forward_sequence(f, l, lang_emb=_le)
            return m.forward_sequence(f, l)

        t0 = _time.time()
        rng = np.random.RandomState(42)

        # Merge base (SFT) data + online buffer
        # Base data ensures fail examples are always present
        episodes = [
            {
                "feat": e["feat"].copy(),
                "length": e["length"],
                "success": e["success"],
                "_is_online": False,
            }
            for e in self._ou_base_data
        ]
        episodes += [
            {
                "feat": e["feat"].copy(),
                "length": e["length"],
                "success": e["success"],
                "_is_online": True,
            }
            for e in online_episodes
        ]

        # Compute normalization from raw features
        all_feat = np.concatenate([e["feat"] for e in episodes], axis=0)
        norm_mean = all_feat.mean(0).astype(np.float32)
        norm_std = all_feat.std(0).astype(np.float32)
        norm_std[norm_std < 1e-8] = 1.0

        # Normalize in place
        for e in episodes:
            e["feat"] = ((e["feat"] - norm_mean) / norm_std).astype(np.float32)

        # Split train / val
        rng.shuffle(episodes)
        n_val = max(2, int(0.15 * len(episodes)))
        val_eps = episodes[:n_val]
        train_eps = episodes[n_val:]

        # Train (match architecture used in __init__)
        if self._use_lang:
            model = _WholeModelWithLang(
                input_dim=11,
                hidden_dim=32,
                mlp_dims=(16, 8),
                lang_dim=self._lang_dim,
                bottleneck=16,
                dropout=self._dropout,
            )
        else:
            model = _WholeModel(input_dim=11, hidden_dim=32, mlp_dims=(16, 8))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_loss = float("inf")
        best_state = None
        patience_cnt = 0

        for epoch in range(self._ou_epochs):
            model.train()
            rng.shuffle(train_eps)
            for i in range(0, len(train_eps), 64):
                batch = train_eps[i : i + 64]
                feat, tgt, mask, lengths = self._collate(batch)
                scores = _fwd_seq(model, feat, lengths)
                sc = scores.clamp(1e-7, 1 - 1e-7)
                loss = (
                    -(tgt * torch.log(sc) + (1 - tgt) * torch.log(1 - sc))
                    * mask.float()
                )
                loss = loss.sum() / mask.float().sum()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            # Validation loss
            model.eval()
            with torch.no_grad():
                vf, vt, vm, vl = self._collate(val_eps)
                vs = _fwd_seq(model, vf, vl)
                vsc = vs.clamp(1e-7, 1 - 1e-7)
                vloss = (
                    -(vt * torch.log(vsc) + (1 - vt) * torch.log(1 - vsc)) * vm.float()
                )
                vloss = (vloss.sum() / vm.float().sum()).item()

            if vloss < best_loss - 1e-4:
                best_loss = vloss
                best_state = _copy.deepcopy(model.state_dict())
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= 40:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Calibrate threshold on val-split online success episodes
        # (avoid overfit: model trained on train_eps, so use val_eps for unbiased scores;
        #  avoid base distribution mismatch: only use online eps)
        model.eval()
        val_online_succ = [
            e for e in val_eps if e.get("_is_online", False) and e["success"]
        ]
        # Fallback: if too few online val eps, use all val success eps
        if len(val_online_succ) < 5:
            val_online_succ = [e for e in val_eps if e["success"]]
        succ_max_scores = []
        with torch.no_grad():
            for e in val_online_succ:
                f = torch.from_numpy(e["feat"]).float().unsqueeze(0)
                l = torch.tensor([e["length"]], dtype=torch.long)
                s = _fwd_seq(model, f, l)[0, : e["length"]].numpy()
                # Sliding window max score
                buf = []
                best_ws = 0.0
                for sv in s:
                    buf.append(float(sv))
                    if len(buf) > self.K:
                        buf.pop(0)
                    if len(buf) == self.K:
                        val = (
                            float(np.median(buf))
                            if self.agg == "median"
                            else float(np.mean(buf))
                        )
                        best_ws = max(best_ws, val)
                succ_max_scores.append(best_ws)

        margin = 0.02  # 小 margin 应对分布偏移
        if succ_max_scores:
            # P95 tau: tolerates 5% success outliers
            p95_succ = float(np.percentile(succ_max_scores, 95))
            max_succ = float(np.max(succ_max_scores))
            new_tau = min(p95_succ + margin, 0.99)
            max_succ_str = f"p95={p95_succ:.4f},max={max_succ:.4f}"
        else:
            new_tau = self.tau  # fallback
            max_succ_str = "N/A"

        # Score curve: average P(fail) at each chunk position for val success vs fail
        max_T = max(e["length"] for e in val_eps) if val_eps else 0
        succ_scores_by_chunk = [[] for _ in range(max_T)]
        fail_scores_by_chunk = [[] for _ in range(max_T)]
        with torch.no_grad():
            for e in val_eps:
                f = torch.from_numpy(e["feat"]).float().unsqueeze(0)
                l = torch.tensor([e["length"]], dtype=torch.long)
                s = _fwd_seq(model, f, l)[0, : e["length"]].numpy()
                target = succ_scores_by_chunk if e["success"] else fail_scores_by_chunk
                for t in range(len(s)):
                    target[t].append(float(s[t]))
        # Format: score(n_samples) for every chunk
        succ_points = []
        fail_points = []
        for i in range(max_T):
            sv = succ_scores_by_chunk[i]
            fv = fail_scores_by_chunk[i]
            succ_points.append(f"{np.mean(sv):.2f}({len(sv)})" if sv else "-")
            fail_points.append(f"{np.mean(fv):.2f}({len(fv)})" if fv else "-")
        n_succ = len([e for e in val_eps if e["success"]])
        n_fail = len([e for e in val_eps if not e["success"]])
        succ_lens = sorted([e["length"] for e in val_eps if e["success"]])
        fail_lens = sorted([e["length"] for e in val_eps if not e["success"]])
        print(
            f"[probe-curve] succ(n={n_succ}, lens={succ_lens[:5]}..{succ_lens[-3:]}): {' '.join(succ_points)}"
        )
        print(
            f"[probe-curve] fail(n={n_fail}, lens={fail_lens[:3]}..{fail_lens[-3:]}): {' '.join(fail_points)}"
        )

        n_succ = sum(1 for e in episodes if e["success"])
        n_online = len(online_episodes)
        n_base = len(self._ou_base_data)
        elapsed = _time.time() - t0
        print(
            f"[probe] retrain done {elapsed:.1f}s: "
            f"tau={new_tau:.4f} ({max_succ_str}+{margin}), "
            f"loss={best_loss:.4f}, ep={epoch + 1}, "
            f"{n_succ}S/{len(episodes) - n_succ}F (base={n_base}+online={n_online})"
        )

        # Store for hot-swap (main thread picks up in maybe_retrain)
        self._ou_pending = {
            "model_state_dict": model.state_dict(),
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "tau": new_tau,
            "n_success": n_succ,
            "n_fail": len(episodes) - n_succ,
        }

    @staticmethod
    def _collate(episodes):
        """Collate episodes into padded batch tensors for training."""
        B = len(episodes)
        T_max = max(e["length"] for e in episodes)
        feat = torch.zeros(B, T_max, 11)
        tgt = torch.zeros(B, T_max)
        mask = torch.zeros(B, T_max, dtype=torch.bool)
        lengths = torch.zeros(B, dtype=torch.long)
        for i, e in enumerate(episodes):
            T = e["length"]
            feat[i, :T] = torch.from_numpy(e["feat"]).float()
            tgt[i, :T] = 0.0 if e["success"] else 1.0
            mask[i, :T] = True
            lengths[i] = T
        return feat, tgt, mask, lengths


from rlinf.envs.libero.utils import (
    get_benchmark_overridden,
    get_libero_image,
    get_libero_type,
    get_libero_wrist_image,
    quat2axisangle,
)
from rlinf.envs.libero.venv import ReconfigureSubprocEnv
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor

libero_type = get_libero_type()

if libero_type in ["pro", "plus"]:
    sys.path[:] = [p for p in sys.path if "opt/libero" not in p]
    LIBERO_PKG_NAME = f"libero{libero_type}"
    LIBERO_MAIN_MODULE_PATH = f"{LIBERO_PKG_NAME}.{LIBERO_PKG_NAME}"
    try:
        real_libero_pkg = importlib.import_module(LIBERO_PKG_NAME)
        real_libero_core = importlib.import_module(LIBERO_MAIN_MODULE_PATH)

        try:
            real_libero_benchmark = importlib.import_module(
                f"{LIBERO_MAIN_MODULE_PATH}.benchmark"
            )
        except ImportError:
            real_libero_benchmark = importlib.import_module(
                f"{LIBERO_PKG_NAME}.benchmark"
            )

        try:
            real_libero_envs = importlib.import_module(
                f"{LIBERO_MAIN_MODULE_PATH}.envs"
            )
        except ImportError:
            real_libero_envs = importlib.import_module(f"{LIBERO_PKG_NAME}.envs")

        sys.modules["libero"] = real_libero_pkg
        sys.modules["libero.libero"] = real_libero_core
        sys.modules["libero.libero.benchmark"] = real_libero_benchmark
        sys.modules["libero.libero.envs"] = real_libero_envs
    except ImportError as e:
        print(
            f"[Main Process Routing Error] Failed to import '{LIBERO_MAIN_MODULE_PATH}'. Error: {e}"
        )

if libero_type == "pro":
    from liberopro.liberopro.benchmark import Benchmark
elif libero_type == "plus":
    from liberoplus.liberoplus.benchmark import Benchmark
else:
    from libero.libero.benchmark import Benchmark


class LiberoEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.seed_offset = seed_offset
        self.cfg = cfg
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.num_envs = num_envs
        self.group_size = self.cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.specific_reset_id = cfg.get("specific_reset_id", None)
        self.task_ids_filter = cfg.get("task_ids_filter", None)  # e.g. [0] to eval only task 0

        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset

        self._generator = np.random.default_rng(seed=self.seed)
        self._generator_ordered = np.random.default_rng(seed=0)
        self.start_idx = 0

        self.task_suite: Benchmark = get_benchmark_overridden(cfg.task_suite_name)()

        self._compute_total_num_group_envs()
        self.reset_state_ids_all = self.get_reset_state_ids_all()
        self.update_reset_state_ids()
        self._init_task_and_trial_ids()
        self._init_env()

        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward
        self.use_step_penalty = getattr(cfg, "use_step_penalty", False)

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

        self.video_cfg = cfg.video_cfg
        self.current_raw_obs = None

        # Whole probe (optional)
        probe_cfg = cfg.get("probe_cfg", None)
        if probe_cfg and probe_cfg.get("enabled", False):
            self.probe = ActProbe(probe_cfg, self.num_envs)
            logger.info(
                f"[LiberoEnv] ActProbe enabled, K={self.probe.K}, tau={self.probe.tau:.4f}"
            )
        else:
            self.probe = None

        # Probe data collection (optional, for generating base training data)
        collect_cfg = cfg.get("collect_probe_data", {})
        self._collecting = collect_cfg.get("enabled", False) if collect_cfg else False
        if self._collecting:
            self._collect_save_dir = collect_cfg.get("save_dir", "/tmp/probe_collect")
            os.makedirs(self._collect_save_dir, exist_ok=True)
            self._collect_max_eps = collect_cfg.get("max_episodes", 300)
            self._collect_rank = seed_offset  # use as rank identifier
            self._collect_buffers = {i: self._new_collect_buf() for i in range(self.num_envs)}
            self._collect_done = np.zeros(self.num_envs, dtype=bool)
            self._collect_episodes = []
            import atexit
            atexit.register(self._save_collect_data)
            logger.info(
                f"[LiberoEnv] Probe data collection ON: save_dir={self._collect_save_dir}, "
                f"max_episodes={self._collect_max_eps}, rank={self._collect_rank}"
            )

    def _init_env(self):
        env_fns = self.get_env_fns()
        self.env = ReconfigureSubprocEnv(env_fns)

    def get_env_fns(self):
        env_fn_params = self.get_env_fn_params()
        env_fns = []

        current_type_val = get_libero_type()

        for env_fn_param in env_fn_params:

            def env_fn(param=env_fn_param, _type_val=current_type_val):
                os.environ["LIBERO_TYPE"] = _type_val
                seed = param.pop("seed")

                if _type_val in ["pro", "plus"]:
                    sys.path[:] = [p for p in sys.path if "opt/libero" not in p]

                    pkg_name = f"libero{_type_val}"
                    core_name = f"{pkg_name}.{pkg_name}"

                    try:
                        real_pkg = importlib.import_module(pkg_name)
                        real_core = importlib.import_module(core_name)
                        real_bench = importlib.import_module(f"{core_name}.benchmark")
                        real_envs = importlib.import_module(f"{core_name}.envs")

                        sys.modules["libero"] = real_pkg
                        sys.modules["libero.libero"] = real_core
                        sys.modules["libero.libero.benchmark"] = real_bench
                        sys.modules["libero.libero.envs"] = real_envs

                        loaded_path = os.path.dirname(real_core.__file__)
                        os.environ["LIBERO_ASSET_ROOT"] = os.path.join(
                            loaded_path, "assets"
                        )
                        os.environ["LIBERO_BDDL_PATH"] = os.path.join(
                            loaded_path, "bddl_files"
                        )
                        os.environ["LIBERO_INIT_STATES_PATH"] = os.path.join(
                            loaded_path, "init_files"
                        )

                        WorkerEnv = real_envs.OffScreenRenderEnv

                    except ImportError as e:
                        print(f"[Worker Env Error] {e}")
                        raise e
                else:
                    from libero.libero.envs import OffScreenRenderEnv as WorkerEnv

                env = WorkerEnv(**param)
                env.seed(seed)
                return env

            env_fns.append(env_fn)
        return env_fns

    def get_env_fn_params(self, env_idx=None):
        env_fn_params = []
        base_env_args = OmegaConf.to_container(self.cfg.init_params, resolve=True)

        variant = os.environ.get(
            "LIBERO_TYPE",
            self.cfg.get("libero_variant", "standard")
            if hasattr(self.cfg, "get")
            else "standard",
        )
        raw_suffix = os.environ.get(
            "LIBERO_SUFFIX",
            os.environ.get(
                "LIBERO_PERTURBATION",
                self.cfg.get("perturbation_suffix", None)
                if hasattr(self.cfg, "get")
                else None,
            ),
        )
        if variant == "pro":
            import liberopro.liberopro as l_pro

            bddl_root = l_pro.get_libero_path("bddl_files")
        elif variant == "plus":
            import liberoplus.liberoplus as l_plus

            bddl_root = l_plus.get_libero_path("bddl_files")
        else:
            from libero.libero import get_libero_path

            bddl_root = get_libero_path("bddl_files")

        suite_name = self.cfg.task_suite_name.lower()
        suite_keyword = suite_name.replace("libero_", "").strip()

        task_descriptions = []
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        for env_id in range(self.num_envs):
            if env_id not in env_idx:
                task_descriptions.append(
                    self.task_descriptions[env_id]
                    if hasattr(self, "task_descriptions")
                    else ""
                )
                continue

            task = self.task_suite.get_task(self.task_ids[env_id])
            folder_name = task.problem_folder
            file_name = task.bddl_file
            original_path = os.path.join(bddl_root, folder_name, file_name)

            final_path = original_path

            if variant == "pro":
                pro_suffix = raw_suffix.replace(".bddl", "") if raw_suffix else None

                valid_perts = ["_lan", "_object", "_swap", "_task"]
                if pro_suffix == "all":
                    filter_perts = valid_perts
                elif pro_suffix is not None:
                    # Map bare name (e.g. "task") to directory suffix (e.g. "_task")
                    normalized = (
                        f"_{pro_suffix}"
                        if not pro_suffix.startswith("_")
                        else pro_suffix
                    )
                    filter_perts = [normalized] if normalized in valid_perts else []
                else:
                    filter_perts = []

                if filter_perts:
                    all_sub_dirs = [
                        d
                        for d in os.listdir(bddl_root)
                        if os.path.isdir(os.path.join(bddl_root, d))
                        and suite_keyword in d
                        and any(d.endswith(pert) for pert in filter_perts)
                    ]

                    core_task_name = file_name.replace(".bddl", "")
                    all_candidates = []

                    for sub_dir in all_sub_dirs:
                        target_dir_path = os.path.join(bddl_root, sub_dir)
                        matches = [
                            os.path.join(target_dir_path, f)
                            for f in os.listdir(target_dir_path)
                            if core_task_name in f and f.endswith(".bddl")
                        ]
                        all_candidates.extend(matches)

                    if all_candidates:
                        all_candidates.sort()
                        if getattr(self.cfg, "is_eval", False):
                            idx_offset = (
                                list(env_idx).index(env_id) if env_id in env_idx else 0
                            )
                            final_path = all_candidates[
                                (self.seed + idx_offset) % len(all_candidates)
                            ]
                        else:
                            final_path = self._generator.choice(all_candidates)

            elif variant == "plus":
                plus_suffix = raw_suffix.replace(".bddl", "") if raw_suffix else None
                if plus_suffix == "all":
                    clean_name = file_name.replace(".bddl", "")
                    for marker in [
                        "_view",
                        "_initstate",
                        "_noise",
                        "_sample",
                        "_light",
                        "_table",
                        "_add_1",
                        "_lan",
                        "_language",
                        "_copy",
                        "_level",
                        "_tb",
                    ]:
                        if marker in clean_name:
                            clean_name = clean_name.split(marker)[0]
                            break

                    suite_pattern = folder_name.replace("_", "").lower()
                    all_dirs = [
                        d
                        for d in os.listdir(bddl_root)
                        if os.path.isdir(os.path.join(bddl_root, d))
                    ]
                    search_dirs = [
                        os.path.join(bddl_root, d)
                        for d in all_dirs
                        if suite_pattern in d.lower().replace("_", "")
                    ]

                    if not search_dirs:
                        search_dirs = [os.path.join(bddl_root, folder_name)]

                    all_candidates = []
                    for target_dir in search_dirs:
                        matches = [
                            f
                            for f in glob.glob(os.path.join(target_dir, "*.bddl"))
                            if clean_name in os.path.basename(f)
                        ]
                        all_candidates.extend(matches)

                    if all_candidates:
                        all_candidates.sort()
                        if getattr(self.cfg, "is_eval", False):
                            idx_offset = (
                                list(env_idx).index(env_id) if env_id in env_idx else 0
                            )
                            final_path = all_candidates[
                                (self.seed + idx_offset) % len(all_candidates)
                            ]
                        else:
                            final_path = self._generator.choice(all_candidates)

            env_fn_params.append(
                {
                    **base_env_args,
                    "bddl_file_name": final_path,
                    "seed": self.seed,
                }
            )
            task_descriptions.append(task.language)

        self.task_descriptions = task_descriptions
        return env_fn_params

    def _compute_total_num_group_envs(self):
        self.total_num_group_envs = 0
        self.trial_id_bins = []
        task_ids = range(self.task_suite.get_num_tasks())
        if self.task_ids_filter is not None:
            task_ids = self.task_ids_filter
        for task_id in task_ids:
            task_num_trials = len(self.task_suite.get_task_init_states(task_id))
            self.trial_id_bins.append(task_num_trials)
            self.total_num_group_envs += task_num_trials
        self.cumsum_trial_id_bins = np.cumsum(self.trial_id_bins)

    def update_reset_state_ids(self):
        if self.cfg.is_eval or self.cfg.use_ordered_reset_state_ids:
            reset_state_ids = self._get_ordered_reset_state_ids(self.num_group)
        else:
            reset_state_ids = self._get_random_reset_state_ids(self.num_group)
        self.reset_state_ids = reset_state_ids.repeat(self.group_size)

    def _init_task_and_trial_ids(self):
        self.task_ids, self.trial_ids = (
            self._get_task_and_trial_ids_from_reset_state_ids(self.reset_state_ids)
        )

    def _get_random_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            reset_state_ids = self.specific_reset_id * np.ones(
                (num_reset_states,), dtype=int
            )
        else:
            reset_state_ids = self._generator.integers(
                low=0, high=self.total_num_group_envs, size=(num_reset_states,)
            )
        return reset_state_ids

    def get_reset_state_ids_all(self):
        reset_state_ids = np.arange(self.total_num_group_envs)
        valid_size = len(reset_state_ids) - (
            len(reset_state_ids) % self.total_num_processes
        )
        self._generator_ordered.shuffle(reset_state_ids)
        reset_state_ids = reset_state_ids[:valid_size]
        reset_state_ids = reset_state_ids.reshape(self.total_num_processes, -1)
        return reset_state_ids

    def _get_ordered_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            reset_state_ids = self.specific_reset_id * np.ones(
                (self.num_group,), dtype=int
            )
        else:
            if self.start_idx + num_reset_states > len(self.reset_state_ids_all[0]):
                self.reset_state_ids_all = self.get_reset_state_ids_all()
                self.start_idx = 0
            reset_state_ids = self.reset_state_ids_all[self.seed_offset][
                self.start_idx : self.start_idx + num_reset_states
            ]
            self.start_idx = self.start_idx + num_reset_states
        return reset_state_ids

    def _get_task_and_trial_ids_from_reset_state_ids(self, reset_state_ids):
        task_ids = []
        trial_ids = []
        # build ordered list of actual task ids (respects task_ids_filter)
        if self.task_ids_filter is not None:
            actual_task_ids = list(self.task_ids_filter)
        else:
            actual_task_ids = list(range(self.task_suite.get_num_tasks()))
        # get task id and trial id from reset state ids
        for reset_state_id in reset_state_ids:
            start_pivot = 0
            for idx, end_pivot in enumerate(self.cumsum_trial_id_bins):
                if reset_state_id < end_pivot and reset_state_id >= start_pivot:
                    task_ids.append(actual_task_ids[idx])
                    trial_ids.append(reset_state_id - start_pivot)
                    break
                start_pivot = end_pivot

        return np.array(task_ids), np.array(trial_ids)

    def _get_reset_states(self, env_idx):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        init_state = [
            self.task_suite.get_task_init_states(self.task_ids[env_id])[
                self.trial_ids[env_id]
            ]
            for env_id in env_idx
        ]
        return init_state

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0

    # ── Probe data collection helpers ──────────────────────────────

    def _new_collect_buf(self):
        return {"action_norms": [], "gripper_qpos": [], "eef_pos": [],
                "denoising_curvature": [], "action_chunks": [], "success": False}

    def _collect_chunk(self, chunk_actions, obs_list, dc, raw_terms, raw_truncs):
        """Record probe features from one chunk_step call."""
        if not self._collecting or len(self._collect_episodes) >= self._collect_max_eps:
            return
        chunk_actions_np = (chunk_actions.cpu().numpy()
                            if isinstance(chunk_actions, torch.Tensor) else chunk_actions)
        chunk_size = chunk_actions_np.shape[1]
        for ei in range(self.num_envs):
            if self._collect_done[ei]:
                continue
            buf = self._collect_buffers[ei]
            # Per env-step features
            for si in range(chunk_size):
                a = chunk_actions_np[ei, si]
                buf["action_norms"].append(float(np.linalg.norm(a)))
                obs = obs_list[si]
                s = obs["states"]
                if isinstance(s, torch.Tensor):
                    s = s[ei].cpu().numpy()
                elif isinstance(s, list):
                    s = s[ei].cpu().numpy() if isinstance(s[ei], torch.Tensor) else np.array(s[ei])
                else:
                    s = np.array(s[ei]) if hasattr(s, '__getitem__') else np.array(s)
                buf["gripper_qpos"].append(s[6:8].copy())
                buf["eef_pos"].append(s[:3].copy())
            # Per chunk-step features
            if dc is not None:
                buf["denoising_curvature"].append(float(dc[ei]) if not isinstance(dc[ei], float) else dc[ei])
            buf["action_chunks"].append(chunk_actions_np[ei].copy())
            # Episode boundary
            term = raw_terms[ei].any().item() if isinstance(raw_terms, torch.Tensor) else bool(raw_terms[ei].any())
            trunc = raw_truncs[ei].any().item() if isinstance(raw_truncs, torch.Tensor) else bool(raw_truncs[ei].any())
            if term:
                buf["success"] = True
            if term or trunc:
                self._finalize_collect_episode(ei)

    def _finalize_collect_episode(self, ei):
        """Save one completed episode and reset its buffer."""
        buf = self._collect_buffers[ei]
        length = len(buf["action_norms"])
        if length == 0:
            return
        eef_pos = np.array(buf["eef_pos"], dtype=np.float32)
        eef_vel = np.zeros_like(eef_pos)
        if length > 1:
            eef_vel[1:] = np.diff(eef_pos, axis=0)
        task_id = int(self.task_ids[ei]) if hasattr(self, "task_ids") else 0
        task_desc = self.task_descriptions[ei] if hasattr(self, "task_descriptions") else ""
        episode = {
            "episode_id": len(self._collect_episodes),
            "task_id": task_id,
            "task_name": task_desc.replace(" ", "_"),
            "task_description": task_desc,
            "seed": int(self.seed),
            "success": buf["success"],
            "length": length,
            "action_chunk": buf["action_chunks"],
            "action_norm": np.array(buf["action_norms"], dtype=np.float32),
            "gripper_qpos": np.array(buf["gripper_qpos"], dtype=np.float32),
            "eef_pos": eef_pos,
            "eef_vel": eef_vel,
            "denoising_curvature": np.array(buf["denoising_curvature"], dtype=np.float32),
            "failure_type": None if buf["success"] else "timeout",
        }
        self._collect_episodes.append(episode)
        self._collect_done[ei] = True
        self._collect_buffers[ei] = self._new_collect_buf()
        if len(self._collect_episodes) % 20 == 0:
            logger.info(f"[ProbeCollect] rank={self._collect_rank} collected {len(self._collect_episodes)}/{self._collect_max_eps} episodes")
        if len(self._collect_episodes) >= self._collect_max_eps:
            self._save_collect_data()

    def _save_collect_data(self):
        """Dump collected episodes to pickle."""
        if not self._collect_episodes:
            return
        import pickle
        path = os.path.join(self._collect_save_dir, f"probe_data_rank{self._collect_rank}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self._collect_episodes, f)
        n_s = sum(1 for e in self._collect_episodes if e["success"])
        logger.info(
            f"[ProbeCollect] Saved {len(self._collect_episodes)} episodes "
            f"({n_s}S/{len(self._collect_episodes)-n_s}F) to {path}"
        )

    def reset_collect_state(self):
        """Reset collection done flags for new rollout epoch (auto_reset=False)."""
        if self._collecting:
            self._collect_done[:] = False

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | terminations
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = to_tensor(episode_info)
        return infos

    def _extract_image_and_state(self, obs):
        return {
            "full_image": get_libero_image(obs),
            "wrist_image": get_libero_wrist_image(obs),
            "state": np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ]
            ),
        }

    def _wrap_obs(self, obs_list):
        images_and_states_list = []
        for obs in obs_list:
            images_and_states = self._extract_image_and_state(obs)
            images_and_states_list.append(images_and_states)

        images_and_states = to_tensor(
            list_of_dict_to_dict_of_list(images_and_states_list)
        )

        full_image_tensor = torch.stack(
            [value.clone() for value in images_and_states["full_image"]]
        )
        wrist_image_tensor = torch.stack(
            [value.clone() for value in images_and_states["wrist_image"]]
        )

        states = images_and_states["state"]

        obs = {
            "main_images": full_image_tensor,
            "wrist_images": wrist_image_tensor,
            "states": states,
            "task_descriptions": self.task_descriptions,
        }
        return obs

    def _reconfigure(self, reset_state_ids, env_idx):
        reconfig_env_idx = []
        task_ids, trial_ids = self._get_task_and_trial_ids_from_reset_state_ids(
            reset_state_ids
        )
        for j, env_id in enumerate(env_idx):
            task_changed = self.task_ids[env_id] != task_ids[j]
            self.task_ids[env_id] = task_ids[j]
            self.trial_ids[env_id] = trial_ids[j]
            if task_changed or not getattr(self.cfg, "is_eval", False):
                reconfig_env_idx.append(env_id)
        if reconfig_env_idx:
            env_fn_params = self.get_env_fn_params(reconfig_env_idx)
            self.env.reconfigure_env_fns(env_fn_params, reconfig_env_idx)
        self.env.seed(self.seed * len(env_idx))
        self.env.reset(id=env_idx)
        variant = os.environ.get(
            "LIBERO_TYPE",
            self.cfg.get("libero_variant", "standard")
            if hasattr(self.cfg, "get")
            else "standard",
        )
        if variant != "plus":
            init_state = self._get_reset_states(env_idx=env_idx)
            self.env.set_init_state(init_state=init_state, id=env_idx)

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids=None,
    ):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        if self.is_start:
            reset_state_ids = (
                self.reset_state_ids if self.use_fixed_reset_state_ids else None
            )
            self._is_start = False

        if reset_state_ids is None:
            num_reset_states = len(env_idx)
            reset_state_ids = self._get_random_reset_state_ids(num_reset_states)

        self._reconfigure(reset_state_ids, env_idx)
        for _ in range(15):
            zero_actions = np.zeros((len(env_idx), 7))
            if self.cfg.reset_gripper_open:
                zero_actions[:, -1] = -1
            raw_obs, _reward, terminations, info_lists = self.env.step(
                zero_actions, env_idx
            )
        if self.current_raw_obs is None:
            self.current_raw_obs = [None] * self.num_envs
        for i, idx in enumerate(env_idx):
            self.current_raw_obs[idx] = raw_obs[i]

        obs = self._wrap_obs(self.current_raw_obs)
        self._reset_metrics(env_idx)
        # Reset collection state for reset envs
        if self._collecting:
            for idx in env_idx:
                self._collect_done[idx] = False
                self._collect_buffers[idx] = self._new_collect_buf()
        infos = {}
        return obs, infos

    def step(self, actions=None, auto_reset=True):
        """Step the environment with the given actions."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, info_lists = self.env.step(actions)
        self.current_raw_obs = raw_obs
        infos = list_of_dict_to_dict_of_list(info_lists)
        truncations = self.elapsed_steps >= self.cfg.max_episode_steps
        obs = self._wrap_obs(raw_obs)

        step_reward = self._calc_step_reward(terminations)

        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions, denoising_curvature=None):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        # ── Probe data collection (before auto_reset modifies terminations) ──
        if self._collecting:
            self._collect_chunk(chunk_actions, obs_list, denoising_curvature,
                                raw_chunk_terminations, raw_chunk_truncations)

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones.cpu().numpy(), obs_list[-1], infos_list[-1]
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()

        # ── Probe inference (optional) ──
        # Results are NOT injected into infos (would break Channel serialization).
        # Instead, env_worker reads probe state directly from env.probe.
        if self.probe is not None:
            actions_np = (
                chunk_actions.cpu().numpy()
                if isinstance(chunk_actions, torch.Tensor)
                else chunk_actions
            )
            features = self.probe.extract_features(
                actions_np, obs_list, denoising_curvature
            )
            self.probe.predict(features)
            # probe.predicted_fail is updated in-place; env_worker reads it directly

        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        import time as _time

        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        if self.cfg.is_eval:
            self.update_reset_state_ids()
        _reset_t0 = _time.time()
        obs, infos = self.reset(
            env_idx=env_idx,
            reset_state_ids=self.reset_state_ids[env_idx]
            if self.use_fixed_reset_state_ids
            else None,
        )
        _reset_dt = _time.time() - _reset_t0
        if not hasattr(self, "_auto_reset_total_time"):
            self._auto_reset_total_time = 0.0
            self._auto_reset_total_count = 0
            self._auto_reset_total_envs = 0
        self._auto_reset_total_time += _reset_dt
        self._auto_reset_total_count += 1
        self._auto_reset_total_envs += len(env_idx)
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def force_cut_envs(self, env_indices, do_reset=True):
        """Force-cut specified envs (v17). Returns reset obs if do_reset=True.

        Args:
            env_indices: numpy array of env indices to cut
            do_reset: if True, reset the envs (probe cut). If False, just mark done (end-of-collection cut).
        Returns:
            reset_obs: dict of obs tensors after reset (only if do_reset=True), else None
        """
        if len(env_indices) == 0:
            return None
        if do_reset:
            obs, _infos = self.reset(
                env_idx=env_indices,
                reset_state_ids=self.reset_state_ids[env_indices]
                if self.use_fixed_reset_state_ids
                else None,
            )
            # Reset probe state for cut envs
            if self.probe is not None:
                for ei in env_indices:
                    self.probe.window_buf[ei] = []
                    self.probe.predicted_fail[ei] = False
                    self.probe.trigger_step[ei] = -1
                    if self.probe._ou_enabled:
                        self.probe._ou_feat_buf[ei] = []
                    if self.probe.hc is not None:
                        self.probe.hc[0][:, ei, :] = 0
                        self.probe.hc[1][:, ei, :] = 0
            return obs
        return None

    def _calc_step_reward(self, terminations):
        step_penalty = -1 if self.use_step_penalty else 0
        termination_bonus = self.cfg.reward_coef * terminations
        reward = step_penalty + termination_bonus

        if self.use_rel_reward:
            reward_diff = reward - self.prev_step_reward
            self.prev_step_reward = reward
            return reward_diff
        else:
            return reward
