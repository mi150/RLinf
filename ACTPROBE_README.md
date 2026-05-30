# ActProbe: Autoreset + Probe-based Early-Cut for Embodied RL (LIBERO + GR00T)

ActProbe accelerates embodied RL by detecting episodes that are going to fail
*during* the rollout and cutting them early, saving environment interactions
without losing success rate. It ships in two modes:

- **autoreset** — done envs immediately reset and keep collecting trajectories
  within one long rollout epoch (continuous collection). No probe.
- **autoreset + actprobe** — on top of autoreset, a lightweight failure-detection
  probe (LSTM + language conditioning) predicts failure mid-rollout and
  force-cuts those envs. The probe is **retrained online every PPO step** to
  track policy drift, and its threshold is recalibrated on success rollouts.

Verified on LIBERO-10 task 0 with GR00T-N1.5.

---

## How the probe trains *online during RL* (important)

This is the key detail that makes ActProbe work under RL. The probe is **not a
frozen detector** — if it were, it would go stale the moment PPO starts changing
the policy (the action/feature distribution shifts, and a fixed probe would
mis-fire). Instead, the probe is **retrained from scratch every PPO step**,
interleaved with the policy update, in a background thread so it doesn't block
the rollout.

The loop each PPO step:

1. **Collect labeled episodes during rollout.** As envs roll out, the probe
   extracts per-chunk features and, when an episode finishes, records its full
   feature sequence + its outcome (`success` from a real task termination, or
   `failure` from timeout). These go into an **online buffer**, which keeps only
   the most recent `recent_steps` steps of episodes (so it reflects the *current*
   policy, not stale ones).

2. **Retrain the probe on (online buffer + base data).** A background thread
   trains a fresh probe (random init, not warm-started) for `epochs` epochs on:
   - the **online buffer** (this step's success/failure episodes — the live
     distribution), plus
   - the **base data** (`merged_200eps.pkl`, episodes from the SFT policy) —
     this guarantees enough *failure* examples are always present, since early
     in RL the buffer can be almost all successes and the probe would have no
     failures to learn from.
   Feature normalization (mean/std) is recomputed from the merged set each time.

3. **Recalibrate the cut threshold `tau`.** After retraining, `tau` is set to the
   `(1 - alpha)` quantile (P95 by default) of the per-episode max scores on the
   held-out **success** episodes, plus a small margin. Semantics: tolerate ~alpha
   false-positive rate on successes. `tau` is recomputed from scratch each step
   (it tracks the new model's score scale; it does **not** smooth from the old
   `tau`).

4. **Hot-swap.** At the start of the next PPO step, the new weights + `norm` +
   `tau` replace the live probe (`apply_pending`), so the policy update and the
   probe update stay in lockstep with a one-step pipeline (no stall).

5. **Warmup.** For the first `warmup_steps` PPO steps the probe only *collects*
   data and is *not* allowed to cut — it needs a few steps of on-distribution
   data before its cuts are trustworthy.

Net effect: the probe continuously chases the drifting policy, so its failure
predictions stay calibrated throughout training, and the early-cuts keep saving
env interactions as the policy improves.

Relevant config (`env.train.probe_cfg.online_update`): `interval`, `epochs`,
`buffer_size`, `recent_steps`, `min_episodes`, `base_data_path`. Implementation:
`rlinf/envs/libero/libero_env.py` (`ActProbe.collect_episode` /
`maybe_retrain` / `_retrain_worker` / `apply_pending`) and the rollout/cut path
in `rlinf/workers/env/env_worker.py`.

---

## 1. Dependencies to download

| Item | Where | Used by |
|------|-------|---------|
| **RLinf-Gr00t-SFT-Long** (~7 GB) | https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Long | all modes (policy) |
| **Qwen3-Embedding-0.6B** (~1.2 GB) | https://huggingface.co/Qwen/Qwen3-Embedding-0.6B | actprobe (language conditioning) |
| probe_v19.pt (100 KB) | **bundled in this repo** (`checkpoints/probe_libero10_task0/`) | actprobe (offline-pretrained probe init) |
| merged_200eps.pkl (6.4 MB) | **bundled in this repo** | actprobe (base data for online retrain) |
| LIBERO datasets | installed with `libero` package | all modes (env) |

Download the two HF models, then set their paths in the config (search for
`/path/to/model/...` placeholders). The probe weights and base data are already
in the repo — no download needed.

---

## 2. How to run

All commands use `examples/embodiment/run_embodiment.sh <config> GR00T`.
Configs are under `examples/embodiment/config/`.

### Baseline (standard PPO, no autoreset)
```bash
bash examples/embodiment/run_embodiment.sh libero_10_ppo_gr00t_baseline_task0_8gpu GR00T
```

### Autoreset (continuous collection, no probe)
```bash
bash examples/embodiment/run_embodiment.sh libero_10_autoreset_task0_8gpu GR00T
```

### Autoreset + ActProbe (probe early-cut + online retrain)
```bash
bash examples/embodiment/run_embodiment.sh libero_10_autoreset_actprobe_task0_8gpu GR00T
```

### (Optional) Collect base data for the probe
To regenerate `merged_200eps.pkl` from your own SFT rollouts:
```bash
bash examples/embodiment/run_embodiment.sh libero_10_collect_probe_data GR00T
```

---

## 3. Key config knobs (actprobe)

In `libero_10_autoreset_actprobe_task0_8gpu.yaml` under `env.train.probe_cfg`:

| key | meaning |
|-----|---------|
| `checkpoint_path` | offline-pretrained probe weights (bundled, used at warmup) |
| `initial_tau` | starting cut threshold (recalibrated online) |
| `warmup_steps` | PPO steps before cutting starts (probe still collects) |
| `K`, `agg` | sliding-window size / aggregation for the cut decision |
| `online_update.*` | online retrain: interval, epochs, buffer, recent_steps, base data |

`env.train.v17_continuous_collect.target_trajectories` controls how many
trajectories to collect per rank before switching to dummy mode (per-rank).

---

## 4. What gets saved / logged

Per PPO step the env worker logs ActProbe metrics:
```
[actprobe] succ=.. missed=.. cut=.. cut_rate=..% tau=.. saved=../..(..%)
[actprobe-detail] immune_succ=..(fp) never_flagged_timeout=..(blind) flagged_succ=..(near_fp)
[probe] retrain done ..s: tau=.. (p95+margin), loss=.., NS/NF (base+online)
```
These give recall (cut + immune_timeout), blind (missed failures), false
positives, and env-step savings.
