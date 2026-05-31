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
"""ConfProbe: failure-detection probe for pi0.5 + RoboCasa (v19 early-cut).

Mirrors the WholeProbe interface used by libero_env so the v17/v19 env_worker
path can drive RoboCasa rollouts unchanged. Reads 2 action-space scalars per
chunk step (action_norm + chunk_mse) + relative timestep, runs an LSTM with
lang-conditioned init, and flags envs whose K-window median failure prob exceeds
tau. See checkpoints/probe_robocasa/HANDOFF_probe_robocasa.md.
"""

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Architecture constants (must match the trained ckpt — see handoff §3).
LANG_DIM = 1024
LANG_BOTTLENECK = 16
HIDDEN = 32
DROPOUT = 0.4


class WholeFlex(nn.Module):
    """2-feat lang-conditioned LSTM probe (WholeFlex). Supports both full-sequence
    (training-style) and single-step streaming (online cut) forward passes."""

    def __init__(self, n_feat=2):
        super().__init__()
        self.n_feat = n_feat
        self.lp = nn.Linear(LANG_DIM, LANG_BOTTLENECK)
        self.h0_lin = nn.Linear(LANG_BOTTLENECK, HIDDEN)
        self.c0_lin = nn.Linear(LANG_BOTTLENECK, HIDDEN)
        self.lstm = nn.LSTM(n_feat + 1, HIDDEN, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(DROPOUT)
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN + n_feat + 1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def init_hidden(self, lang_emb):
        """Return (h0, c0) each (1, B, HIDDEN) from lang embedding (B, LANG_DIM)."""
        z = torch.relu(self.lp(lang_emb))
        h0 = self.h0_lin(z).unsqueeze(0).contiguous()
        c0 = self.c0_lin(z).unsqueeze(0).contiguous()
        return h0, c0

    def forward_step(self, x_t, hc):
        """Single-step streaming forward.

        Args:
            x_t: (B, n_feat+1) current-step features [norm_feats..., ts]
            hc: (h, c) each (1, B, HIDDEN)
        Returns:
            prob: (B,) failure prob in [0, 1]
            hc: updated (h, c)
        """
        out, hc = self.lstm(x_t.unsqueeze(1), hc)  # out: (B, 1, HIDDEN)
        out = self.drop(out).squeeze(1)  # (B, HIDDEN)  (drop is no-op in eval)
        logit = self.mlp(torch.cat([out, x_t], dim=-1)).squeeze(-1)  # (B,)
        return torch.sigmoid(logit), hc

    def forward(self, x, lang_emb, lengths):
        """Full-sequence forward (matches training; used for tau calibration)."""
        h0, c0 = self.init_hidden(lang_emb)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed, (h0, c0))
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=x.shape[1]
        )
        out = self.drop(out)
        return torch.sigmoid(self.mlp(torch.cat([out, x], dim=-1)).squeeze(-1))


class ConfProbe:
    """Online failure-detection probe for RoboCasa rollouts.

    Interface mirrors libero_env.WholeProbe (predicted_fail, reset_all,
    reset_env_hidden, extract_features, predict) so the v17/v19 env_worker path
    works unchanged.
    """

    def __init__(self, cfg, num_envs, device="cpu"):
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg

        # Cut policy
        self.K = cfg.get("K", 3)
        self.agg = cfg.get("agg", "median")
        self.tau = cfg.get("initial_tau", 0.5)
        self.alpha = cfg.get("alpha", 0.15)
        self.min_cut_chunk = cfg.get("min_cut_chunk", 0)
        # Timestep mode (must match the ckpt's training):
        #   "abs": ts = chunk_step / ts_scale  (online-friendly, no episode length
        #          needed, zero approximation — recommended for RL early-cut)
        #   "rel": ts ≈ chunk_step / max_chunk_steps (approximates train-time
        #          linspace(0,1,T); biased online since T unknown)
        self.ts_mode = cfg.get("ts_mode", "abs")
        self.ts_scale = cfg.get("ts_scale", 100.0)  # abs: t/100 (handoff §9)
        self.max_chunk_steps = cfg.get("max_chunk_steps", 15)  # rel mode only

        # Feature params (must match probe training — handoff §4)
        self.exec_horizon = cfg.get("exec_horizon", 20)
        self.n_action_dims = cfg.get("n_action_dims", 7)

        # ── Load checkpoint ──
        ckpt_path = cfg.checkpoint_path
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # n_feat from ckpt: 2-feat [action_norm, chunk_mse] or 8-feat (+ eef
        # pos[x,y,z] and per-chunk velocity[vx,vy,vz]). Build the matching model.
        self.n_feat = int(ck.get("n_feat", 2))
        self.model = WholeFlex(n_feat=self.n_feat)
        self.model.load_state_dict(ck["model"])  # NOTE: key is "model"
        self.model.eval()
        self.norm_mean = np.asarray(ck["norm_mean"], dtype=np.float32)  # (n_feat,)
        self.norm_std = np.asarray(ck["norm_std"], dtype=np.float32)  # (n_feat,)
        if "ts_mode" in ck:  # the ckpt's ts convention wins over cfg
            self.ts_mode = ck["ts_mode"]
        if "tau" in ck and cfg.get("initial_tau", None) is None:
            self.tau = float(ck["tau"])
        logger.info(
            f"[ConfProbe] loaded {ckpt_path}: tau={self.tau:.4f}, K={self.K}, "
            f"exec_horizon={self.exec_horizon}, n_dims={self.n_action_dims}"
        )

        # ── Language conditioning ──
        # Two options: (a) precomputed embedding .npy (lang_emb_path) — no Qwen3
        # needed at runtime (preferred for fixed single-task); (b) load Qwen3 and
        # encode task_instruction online.
        self._lang_cfg = cfg.get("lang_encoder", {})
        self._lang_emb = None  # (1, LANG_DIM) tensor
        self._lang_model = None
        self._lang_tok = None
        if self._lang_cfg.get("enabled", True):
            emb_path = self._lang_cfg.get("lang_emb_path", None)
            if emb_path:
                emb = np.load(emb_path).astype(np.float32).reshape(1, -1)
                self._lang_emb = torch.from_numpy(emb)
                logger.info(f"[ConfProbe] lang emb loaded from {emb_path} {emb.shape}")
            else:
                self._init_lang_encoder()
                instr = self._lang_cfg.get("task_instruction", None)
                if instr:
                    self.set_lang(instr)

        # ── Online retrain (optional, mirrors libero WholeProbe) ──
        ou = cfg.get("online_update", {})
        self._ou_enabled = ou.get("enabled", False)
        self._ou_interval = ou.get("interval", 1)
        self._ou_buffer_size = ou.get("buffer_size", 400)
        self._ou_recent_steps = ou.get("recent_steps", 3)
        self._ou_freeze_after_step = ou.get("freeze_after_step", 0)
        self._ou_epochs = ou.get("epochs", 50)
        self._ou_min_episodes = ou.get("min_episodes", 50)
        self._ou_lr = ou.get("lr", 1e-3)
        self._ou_batch = ou.get("batch_size", 64)
        self._ou_margin = ou.get("tau_margin", 0.02)
        self._ou_buffer = []
        self._ou_pending = None
        self._ou_thread = None
        self._ou_step = 0
        self._ou_n_updates = 0
        self._ou_buffer_frozen = False
        self._ou_base_data = []
        if self._ou_enabled:
            self._ou_base_data = self._load_base_data(
                ou.get("base_data_path", None),
                max_episodes=ou.get("max_base_episodes", 0),
            )
            n_s = sum(1 for e in self._ou_base_data if e["success"])
            logger.info(
                f"[ConfProbe] online_update enabled: base={len(self._ou_base_data)} "
                f"({n_s}S/{len(self._ou_base_data) - n_s}F), interval={self._ou_interval}, "
                f"epochs={self._ou_epochs}"
            )

        self.reset_all()

    # ------------------------------------------------------------- base data
    def _load_base_data(self, path, max_episodes=0):
        """Load offline base episodes (feat already computed: [action_norm,
        chunk_mse, step_idx]). ts (col 2) is recomputed per ts_mode at retrain."""
        if not path:
            return []
        import pickle

        try:
            with open(path, "rb") as f:
                episodes = pickle.load(f)
        except Exception as e:
            logger.warning(f"[ConfProbe] failed to load base data {path}: {e}")
            return []
        base = []
        for ep in episodes:
            feat = np.asarray(ep["feat"], dtype=np.float32)  # (M, 3) raw
            base.append(
                {"feat": feat, "length": int(ep["length"]), "success": bool(ep["success"])}
            )
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

    # ------------------------------------------------------------------ lang
    def _init_lang_encoder(self):
        from transformers import AutoModel, AutoTokenizer

        path = self._lang_cfg["model_path"]
        self._lang_tok = AutoTokenizer.from_pretrained(path)
        self._lang_model = AutoModel.from_pretrained(path).eval()
        logger.info(f"[ConfProbe] Qwen3 lang encoder loaded from {path}")

    def ensure_lang(self, instruction):
        """Encode instruction once (idempotent). Called by env with the real
        ep_meta.lang so the probe's lang-cond matches what the policy received."""
        if self._lang_emb is not None or not instruction:
            return  # already set (via lang_emb_path or prior call)
        if self._lang_model is None:
            self._init_lang_encoder()
        self.set_lang(instruction)

    def set_lang(self, instruction):
        """Encode a task instruction → (1, LANG_DIM) embedding (mean-pooled)."""
        if self._lang_model is None:
            return
        with torch.no_grad():
            inp = self._lang_tok(
                [instruction],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            emb = self._lang_model(**inp).last_hidden_state.mean(dim=1)  # (1, 1024)
        self._lang_emb = emb.float()
        logger.info(f"[ConfProbe] lang set: '{instruction[:50]}...'")

    # ----------------------------------------------------------------- reset
    def reset_all(self):
        """Reset all per-env state (call at epoch start)."""
        self.hc = None  # LSTM hidden (lazy init on first predict)
        self.window_buf = [[] for _ in range(self.num_envs)]
        self.predicted_fail = np.zeros(self.num_envs, dtype=bool)
        self.trigger_step = np.full(self.num_envs, -1, dtype=np.int32)
        self.chunk_idx = 0
        self._env_ep_chunk = np.zeros(self.num_envs, dtype=np.int32)
        self._prev_full_chunk = [None] * self.num_envs  # for chunk_mse
        self._prev_eef = [None] * self.num_envs  # for eef velocity (8-feat)
        self._score_history = [[] for _ in range(self.num_envs)]
        self._completed_score_histories = []
        # per-env raw feature accumulator (action_norm, chunk_mse) for online retrain
        self._ou_feat_buf = [[] for _ in range(self.num_envs)]

    def reset_env_hidden(self, env_idx):
        """Reset LSTM hidden + per-env probe state after done/cut (auto_reset)."""
        for ei in np.atleast_1d(env_idx):
            self.window_buf[ei] = []
            self.predicted_fail[ei] = False
            self.trigger_step[ei] = -1
            self._env_ep_chunk[ei] = 0
            self._prev_full_chunk[ei] = None
            self._prev_eef[ei] = None
            self._ou_feat_buf[ei] = []
            if self.hc is not None:
                if self._lang_emb is not None:
                    h0, c0 = self.model.init_hidden(self._lang_emb)
                    self.hc[0][:, ei, :] = h0[:, 0, :]
                    self.hc[1][:, ei, :] = c0[:, 0, :]
                else:
                    self.hc[0][:, ei, :] = 0
                    self.hc[1][:, ei, :] = 0

    # -------------------------------------------------------------- features
    def extract_features(self, full_chunk, eef_pos=None):
        """Compute per-chunk features for one chunk step.

        Args:
            full_chunk: (num_envs, H, action_dim) — the *full* predicted action
                chunk (action_horizon, e.g. 50), not just the executed slice.
                chunk_mse needs the overlap between consecutive full predictions.
            eef_pos: (num_envs, >=3) chunk-start end-effector position; required
                for the 8-feat probe (eef xyz + per-chunk velocity). Ignored for
                2-feat.
        Returns:
            features: (num_envs, n_feat) numpy. 2-feat: [action_norm, chunk_mse].
                8-feat: [action_norm, chunk_mse, x, y, z, vx, vy, vz].
        """
        full = np.asarray(full_chunk, dtype=np.float32)
        E, H, _ = full.shape
        exec_h = min(self.exec_horizon, H)
        d = self.n_action_dims

        # action_norm[t] = RMS over first exec_h steps, first d dims
        action_norm = np.sqrt(
            np.mean(full[:, :exec_h, :d] ** 2, axis=(1, 2))
        )  # (E,)

        # chunk_mse[t] = MSE between prev[exec_h : exec_h+overlap] and cur[:overlap]
        overlap = max(H - exec_h, 0)
        chunk_mse = np.zeros(E, dtype=np.float32)
        for i in range(E):
            prev = self._prev_full_chunk[i]
            if prev is not None and overlap > 0:
                a = prev[exec_h : exec_h + overlap, :d]
                b = full[i, :overlap, :d]
                m = min(a.shape[0], b.shape[0])
                if m > 0:
                    chunk_mse[i] = np.mean((a[:m] - b[:m]) ** 2)
            self._prev_full_chunk[i] = full[i].copy()

        base = np.stack([action_norm, chunk_mse], axis=1)  # (E, 2)
        if self.n_feat <= 2:
            return base
        # 8-feat: append eef position [x,y,z] + per-chunk velocity [vx,vy,vz]
        # (velocity = eef_pos diff between consecutive chunk steps).
        eef = np.asarray(eef_pos, dtype=np.float32).reshape(E, -1)[:, :3]  # (E, 3)
        vel = np.zeros_like(eef)
        for i in range(E):
            prev = self._prev_eef[i]
            if prev is not None:
                vel[i] = eef[i] - prev
            self._prev_eef[i] = eef[i].copy()
        return np.concatenate([base, eef, vel], axis=1)  # (E, 8)

    # --------------------------------------------------------------- predict
    def predict(self, features):
        """Run probe inference for one chunk step; updates self.predicted_fail.

        Args:
            features: (num_envs, 2) numpy [action_norm, chunk_mse]
        Returns:
            prob_fail: (num_envs,) numpy
            newly_flagged: (num_envs,) bool numpy
        """
        # accumulate raw features (pre-norm) for online retrain
        if self._ou_enabled:
            for i in range(self.num_envs):
                self._ou_feat_buf[i].append(features[i].copy())

        self.chunk_idx += 1
        self._env_ep_chunk += 1

        # normalize features, append timestep
        feat = (features - self.norm_mean) / (self.norm_std + 1e-7)  # (E, 2)
        # _env_ep_chunk was incremented above → 0-indexed step = _env_ep_chunk-1
        step0 = (self._env_ep_chunk - 1).astype(np.float32)
        if self.ts_mode == "abs":
            ts = (step0 / self.ts_scale).reshape(-1, 1)  # t/100, no clip
        else:  # rel (approximation)
            ts = np.clip(step0 / max(self.max_chunk_steps - 1, 1), 0.0, 1.0).reshape(
                -1, 1
            )
        ts = ts.astype(np.float32)
        x = np.concatenate([feat, ts], axis=1).astype(np.float32)  # (E, 3)
        x_t = torch.from_numpy(x)

        with torch.no_grad():
            if self.hc is None:
                # lazy init hidden from lang (or zeros)
                if self._lang_emb is not None:
                    lang = self._lang_emb.expand(self.num_envs, -1)
                    h0, c0 = self.model.init_hidden(lang)
                    self.hc = (h0.contiguous(), c0.contiguous())
                else:
                    self.hc = (
                        torch.zeros(1, self.num_envs, HIDDEN),
                        torch.zeros(1, self.num_envs, HIDDEN),
                    )
            prob, self.hc = self.model.forward_step(x_t, self.hc)
        prob_np = prob.numpy()  # (E,)

        for i in range(self.num_envs):
            self._score_history[i].append(float(prob_np[i]))

        newly_flagged = np.zeros(self.num_envs, dtype=bool)
        for i in range(self.num_envs):
            if self.predicted_fail[i]:
                continue
            if self.min_cut_chunk > 0 and self._env_ep_chunk[i] < self.min_cut_chunk:
                continue
            self.window_buf[i].append(prob_np[i])
            if len(self.window_buf[i]) > self.K:
                self.window_buf[i] = self.window_buf[i][-self.K :]
            if len(self.window_buf[i]) == self.K:
                val = (
                    np.median(self.window_buf[i])
                    if self.agg == "median"
                    else np.mean(self.window_buf[i])
                )
                if val > self.tau:
                    self.predicted_fail[i] = True
                    self.trigger_step[i] = self.chunk_idx
                    newly_flagged[i] = True

        return prob_np, newly_flagged

    # ------------------------------------------------------------ tau calib
    def calibrate_tau(self, success_score_sequences):
        """Recalibrate tau on held-out successful rollouts at significance alpha.

        Args:
            success_score_sequences: list of per-episode score arrays (success only)
        """
        if not success_score_sequences:
            return
        per_ep_max = [np.max(s) for s in success_score_sequences if len(s) > 0]
        if per_ep_max:
            new_tau = float(np.quantile(per_ep_max, 1 - self.alpha))
            logger.info(f"[ConfProbe] tau recalibrated: {self.tau:.4f} → {new_tau:.4f}")
            self.tau = new_tau

    # ------------------------------------------------------------ save/load
    def save_probe_state(self, path):
        """Save probe state (model + buffer + tau) for resume (mirrors LIBERO)."""
        import pickle

        if self._ou_enabled and self._ou_thread is not None and self._ou_thread.is_alive():
            self._ou_thread.join(timeout=60)
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
            f"[ConfProbe] state saved to {path}: tau={self.tau:.4f}, buffer={n_s}S/{n_buf - n_s}F"
        )

    def _load_probe_state(self, path):
        """Load saved probe state for resume (mirrors LIBERO)."""
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)
        try:
            self.model.load_state_dict(state["model_state_dict"])
        except RuntimeError:
            logger.info("[ConfProbe] resume: arch mismatch, keeping current weights")
        self.model.eval()
        self.norm_mean = np.asarray(state["norm_mean"], dtype=np.float32)
        self.norm_std = np.asarray(state["norm_std"], dtype=np.float32)
        self.tau = state["tau"]
        if self._ou_enabled:
            self._ou_step = state.get("ou_step", 0)
            self._ou_n_updates = state.get("ou_n_updates", 0)
            self._ou_buffer = state.get("ou_buffer", [])
        logger.info(f"[ConfProbe] resumed from {path}: tau={self.tau:.4f}, ou_step={self._ou_step}")

    # =================================================================
    # Online retrain (mirrors libero WholeProbe; 2-feat WholeFlex)
    # =================================================================
    def _step_col_to_ts(self, step_col):
        """Map a step-index column (0..M-1) to ts per ts_mode."""
        step_col = np.asarray(step_col, dtype=np.float32)
        if self.ts_mode == "abs":
            return step_col / self.ts_scale
        M = len(step_col)
        return step_col / max(M - 1, 1)

    def _raw_to_xfeat(self, raw):
        """raw (T, >=n_feat) [feat_0..feat_{n_feat-1}, (step_idx)] → normalized
        x (T, n_feat+1) with ts computed per ts_mode. n_feat is 2 or 8."""
        raw = np.asarray(raw, dtype=np.float32)
        n = self.n_feat
        ts = (
            self._step_col_to_ts(raw[:, n])
            if raw.shape[1] > n
            else self._step_col_to_ts(np.arange(len(raw)))
        )
        feat = (raw[:, :n] - self.norm_mean) / (self.norm_std + 1e-7)
        return np.concatenate([feat, ts[:, None]], axis=1).astype(np.float32)

    def collect_episode(self, env_idx, is_success):
        """Called when an env finishes (auto_reset). Save score history + push the
        env's accumulated feature sequence into the online buffer with its label."""
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
        if not self._ou_enabled or self._ou_buffer_frozen:
            self._ou_feat_buf[env_idx] = []
            return
        feat_seq = self._ou_feat_buf[env_idx]
        if len(feat_seq) >= 3:
            arr = np.stack(feat_seq, axis=0)  # (T, 2) raw
            T = len(feat_seq)
            step_col = np.arange(T, dtype=np.float32)[:, None]
            raw = np.concatenate([arr, step_col], axis=1)  # (T, 3) raw+step
            self._ou_buffer.append(
                {"feat": raw, "length": T, "success": bool(is_success), "step": self._ou_step}
            )
        self._ou_feat_buf[env_idx] = []

    def finalize_epoch(self, natural_done, env_done_step=None, is_warmup=False):
        """Epoch end: package remaining per-env sequences. Skip probe-cut envs
        (unknown true label) unless warmup."""
        if not self._ou_enabled:
            return
        for i in range(self.num_envs):
            feat_seq = self._ou_feat_buf[i]
            if len(feat_seq) < 3:
                continue
            if not is_warmup and not natural_done[i] and self.predicted_fail[i]:
                continue
            if env_done_step is not None and env_done_step[i] >= 0:
                feat_seq = feat_seq[: env_done_step[i] + 1]
            if len(feat_seq) < 3:
                continue
            arr = np.stack(feat_seq, axis=0)
            T = len(feat_seq)
            step_col = np.arange(T, dtype=np.float32)[:, None]
            raw = np.concatenate([arr, step_col], axis=1)
            self._ou_buffer.append(
                {"feat": raw, "length": T, "success": bool(natural_done[i]), "step": self._ou_step}
            )
        self.trim_buffer()

    def trim_buffer(self):
        """Trim online buffer by recent_steps/buffer_size; freeze if past freeze step."""
        if self._ou_freeze_after_step > 0 and self._ou_step >= self._ou_freeze_after_step:
            if not self._ou_buffer_frozen:
                self._ou_buffer_frozen = True
            return
        if self._ou_buffer_frozen:
            return
        if self._ou_recent_steps > 0:
            min_step = max(0, self._ou_step - self._ou_recent_steps + 1)
            self._ou_buffer = [e for e in self._ou_buffer if e["step"] >= min_step]
        elif len(self._ou_buffer) > self._ou_buffer_size:
            self._ou_buffer = self._ou_buffer[-self._ou_buffer_size :]

    def apply_pending(self):
        """Hot-swap retrained weights (called at step start to avoid 1-step lag)."""
        if not self._ou_enabled or self._ou_pending is None:
            return
        p = self._ou_pending
        self._ou_pending = None
        self.model.load_state_dict(p["model_state_dict"])
        self.model.eval()
        self.norm_mean = p["norm_mean"]
        self.norm_std = p["norm_std"]
        self.tau = p["tau"]
        self._ou_n_updates += 1
        logger.info(
            f"[ConfProbe] online update #{self._ou_n_updates} @ step {self._ou_step}: "
            f"tau={self.tau:.4f}, data={p['n_success']}S/{p['n_fail']}F"
        )

    def maybe_retrain(self):
        """Step end: bump step, apply pending, trigger background retrain if due."""
        if not self._ou_enabled:
            return
        self._ou_step += 1
        if (
            self._ou_freeze_after_step > 0
            and self._ou_step >= self._ou_freeze_after_step
            and not self._ou_buffer_frozen
        ):
            self._ou_buffer_frozen = True
        self.apply_pending()
        if self._ou_step % self._ou_interval == 0:
            if self._ou_thread is None or not self._ou_thread.is_alive():
                buf = list(self._ou_buffer)
                if len(buf) + len(self._ou_base_data) >= self._ou_min_episodes:
                    import threading

                    self._ou_thread = threading.Thread(
                        target=self._retrain_worker, args=(buf,), daemon=True
                    )
                    self._ou_thread.start()
                    n_s = sum(1 for e in buf if e["success"])
                    logger.info(
                        f"[ConfProbe] retrain triggered @ step {self._ou_step}: "
                        f"{len(buf)} online ({n_s}S/{len(buf) - n_s}F) + {len(self._ou_base_data)} base"
                    )

    def _collate(self, episodes):
        """Pad a list of {feat(raw T,3), length, success} into batch tensors.
        Builds normalized x (T,3) with ts per ts_mode."""
        B = len(episodes)
        T_max = max(e["length"] for e in episodes)
        feat = torch.zeros(B, T_max, self.n_feat + 1)
        tgt = torch.zeros(B, T_max)
        mask = torch.zeros(B, T_max, dtype=torch.bool)
        lengths = torch.zeros(B, dtype=torch.long)
        for i, e in enumerate(episodes):
            T = e["length"]
            x = self._raw_to_xfeat(e["feat"][:T])
            feat[i, :T] = torch.from_numpy(x)
            tgt[i, :T] = 1.0 if e["success"] is False else 0.0  # label: fail=1
            mask[i, :T] = True
            lengths[i] = T
        return feat, tgt, mask, lengths

    def _retrain_worker(self, online_episodes):
        """Background: retrain WholeFlex on base + online (raw feats), recalibrate tau."""
        import copy as _copy
        import time as _time

        t0 = _time.time()
        rng = np.random.RandomState(42)
        lang = self._lang_emb if self._lang_emb is not None else None

        def _fwd(m, f, l):
            le = lang.expand(f.shape[0], -1) if lang is not None else torch.zeros(f.shape[0], LANG_DIM)
            return m(f, le, l)

        # merge base (raw feat) + online (raw feat); recompute norm from raw [an, mse]
        episodes = [
            {"feat": e["feat"].copy(), "length": e["length"], "success": e["success"], "_online": False}
            for e in self._ou_base_data
        ] + [
            {"feat": e["feat"].copy(), "length": e["length"], "success": e["success"], "_online": True}
            for e in online_episodes
        ]
        if len(episodes) < 4:
            return
        all_raw = np.concatenate(
            [e["feat"][:, : self.n_feat] for e in episodes], axis=0
        )
        norm_mean = all_raw.mean(0).astype(np.float32)
        norm_std = all_raw.std(0).astype(np.float32)
        norm_std[norm_std < 1e-8] = 1.0
        # temporarily use new norm for collation
        _saved_mean, _saved_std = self.norm_mean, self.norm_std
        self.norm_mean, self.norm_std = norm_mean, norm_std

        rng.shuffle(episodes)
        n_val = max(2, int(0.15 * len(episodes)))
        val_eps, train_eps = episodes[:n_val], episodes[n_val:]

        # train from scratch (strict alignment with LIBERO WholeProbe retrain)
        model = WholeFlex(n_feat=self.n_feat)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=self._ou_lr, weight_decay=1e-4)
        best_loss, best_state, patience = float("inf"), None, 0
        epoch = 0
        for epoch in range(self._ou_epochs):
            model.train()
            rng.shuffle(train_eps)
            for i in range(0, len(train_eps), self._ou_batch):
                batch = train_eps[i : i + self._ou_batch]
                f, tg, m, ln = self._collate(batch)
                sc = _fwd(model, f, ln).clamp(1e-7, 1 - 1e-7)
                loss = -(tg * torch.log(sc) + (1 - tg) * torch.log(1 - sc)) * m.float()
                loss = loss.sum() / m.float().sum()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                vf, vt, vm, vl = self._collate(val_eps)
                vsc = _fwd(model, vf, vl).clamp(1e-7, 1 - 1e-7)
                vloss = (-(vt * torch.log(vsc) + (1 - vt) * torch.log(1 - vsc)) * vm.float())
                vloss = (vloss.sum() / vm.float().sum()).item()
            if vloss < best_loss - 1e-4:
                best_loss, best_state, patience = vloss, _copy.deepcopy(model.state_dict()), 0
            else:
                patience += 1
                if patience >= 40:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)

        # recalibrate tau on val-split success episodes (sliding-window max, q (1-alpha))
        model.eval()
        val_succ = [e for e in val_eps if e.get("_online") and e["success"]]
        if len(val_succ) < 5:
            val_succ = [e for e in val_eps if e["success"]]
        succ_ws = []
        with torch.no_grad():
            for e in val_succ:
                f, _, _, l = self._collate([e])
                s = _fwd(model, f, l)[0, : e["length"]].numpy()
                buf, best = [], 0.0
                for sv in s:
                    buf.append(float(sv))
                    if len(buf) > self.K:
                        buf.pop(0)
                    if len(buf) == self.K:
                        v = float(np.median(buf)) if self.agg == "median" else float(np.mean(buf))
                        best = max(best, v)
                succ_ws.append(best)
        new_tau = (
            min(float(np.quantile(succ_ws, 1 - self.alpha)) + self._ou_margin, 0.99)
            if succ_ws
            else self.tau
        )
        self.norm_mean, self.norm_std = _saved_mean, _saved_std  # restore until apply
        n_succ = sum(1 for e in episodes if e["success"])
        logger.info(
            f"[ConfProbe] retrain done {_time.time() - t0:.1f}s: tau={new_tau:.4f}, "
            f"loss={best_loss:.4f}, ep={epoch + 1}, {n_succ}S/{len(episodes) - n_succ}F"
        )
        self._ou_pending = {
            "model_state_dict": model.state_dict(),
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "tau": new_tau,
            "n_success": n_succ,
            "n_fail": len(episodes) - n_succ,
        }
