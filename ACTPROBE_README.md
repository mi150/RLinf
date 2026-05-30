# ActProbe: Autoreset + Probe-based Early-Cut for Embodied RL (LIBERO + GR00T)

ActProbe speeds up embodied RL by predicting which rollout episodes are going to
**fail** and cutting them short, saving environment interactions without hurting
success rate. Verified on LIBERO-10 task 0 with GR00T-N1.5.

Two modes:

- **autoreset** — when an env finishes, it immediately resets and keeps
  collecting trajectories within one long rollout epoch (continuous collection).
  No probe.
- **autoreset + actprobe** — adds a lightweight failure-detection probe
  (LSTM + language conditioning) that predicts failure mid-rollout and force-cuts
  those envs. The probe is **retrained online every PPO step** and its cut
  threshold is recalibrated each step, so it tracks the policy as RL changes it.

---

## How ActProbe works during RL

The probe is **not a frozen detector**. A fixed probe would go stale the moment
PPO starts changing the policy (the action/feature distribution shifts and the
probe mis-fires). So the probe is **retrained from scratch every PPO step**, in a
background thread that overlaps the next rollout (no stall).

Each PPO step, interleaved with the policy update:

1. **Collect labeled episodes.** While envs roll out, the probe extracts
   per-chunk features; when an episode ends it stores the full feature sequence +
   outcome (`success` = real task termination, `failure` = timeout) into an
   **online buffer** that keeps only the last `recent_steps` steps of episodes
   (so it reflects the *current* policy).

2. **Retrain on online buffer + base data.** A fresh probe (random init) trains
   for `epochs` epochs on the online buffer **plus** the bundled base data
   (`merged_200eps.pkl`, SFT-policy episodes). The base data guarantees enough
   *failure* examples are always present — early in RL the buffer can be almost
   all successes. Feature normalization is recomputed from the merged set.

3. **Recalibrate `tau`.** The cut threshold is set to the `(1 - alpha)` quantile
   (P95 by default) of per-episode max scores on held-out **success** episodes,
   plus a small margin (tolerate ~`alpha` false positives on successes).
   Recomputed from scratch each step — it tracks the new model's score scale, it
   does not smooth from the old `tau`.

4. **Hot-swap.** At the next step's start the new weights / norm / `tau` replace
   the live probe, keeping policy and probe updates in a one-step pipeline.

**Warmup.** For the first `warmup_steps` PPO steps (default `2`) the probe
collects and retrains but is **not allowed to cut** — it needs a couple of steps
of on-distribution data first. Cutting begins at step `warmup_steps + 1`.

**Update frequency.** Retraining runs every `interval` PPO steps (default `1`,
i.e. once per step) for `epochs` epochs (default `50`). With `interval: 1` the
probe refreshes as often as the policy; raise it if retrain wall-clock becomes a
bottleneck (it's a background thread).

**Spare policy — don't cut every predicted failure.** Among the envs flagged as
failing, ActProbe cuts only a fraction `cut_ratio` (default `0.7`) at random and
**spares the rest** (they become *immune* and run to natural timeout; at least
one is always spared). Reason: **PPO still needs complete failure trajectories**.
If every predicted failure were cut (`cut_ratio = 1.0`), the policy-gradient batch
would contain only truncated failures — biasing value/advantage estimates and
dropping the failure signal the policy learns from. Sparing ~`1 - cut_ratio` of
the flagged envs keeps a steady fraction of intact failures in every PPO update
while still cutting the majority to save interactions. Spared episodes also feed
fully-labeled failures back into the probe's retrain buffer.

Implementation: `rlinf/envs/libero/libero_env.py` (`ActProbe`: feature
extraction, `collect_episode`, `maybe_retrain`, `_retrain_worker`,
`apply_pending`) and the rollout/cut path in `rlinf/workers/env/env_worker.py`.

---

## Dependencies

| Item | Where | Used by |
|------|-------|---------|
| **RLinf-Gr00t-SFT-Long** (~7 GB) | https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Long | all modes (policy) |
| **Qwen3-Embedding-0.6B** (~1.2 GB) | https://huggingface.co/Qwen/Qwen3-Embedding-0.6B | actprobe (language conditioning) |
| `probe_v19.pt` (100 KB) | **bundled** (`checkpoints/probe_libero10_task0/`) | actprobe (probe init at warmup) |
| `merged_200eps.pkl` (6.4 MB) | **bundled** | actprobe (base data for online retrain) |
| LIBERO datasets | installed with the `libero` package | all modes (env) |

Download the two HF models and set their paths in the config (search for
`/path/to/model/...` placeholders). The probe weights and base data ship in the
repo — no download needed.

---

## Running

`bash examples/embodiment/run_embodiment.sh <config> GR00T` (configs under
`examples/embodiment/config/`):

| Mode | config |
|------|--------|
| Baseline (standard PPO) | `libero_10_ppo_gr00t_baseline_task0_8gpu` |
| Autoreset (no probe) | `libero_10_autoreset_task0_8gpu` |
| Autoreset + ActProbe | `libero_10_autoreset_actprobe_task0_8gpu` |
| (Optional) collect base data | `libero_10_collect_probe_data` |

```bash
bash examples/embodiment/run_embodiment.sh libero_10_autoreset_actprobe_task0_8gpu GR00T
```

The collect config regenerates `merged_200eps.pkl` from your own SFT rollouts.

---

## Config knobs

`env.train.probe_cfg` (in `libero_10_autoreset_actprobe_task0_8gpu.yaml`):

| key | meaning | default |
|-----|---------|---------|
| `checkpoint_path` | offline-pretrained probe weights (used at warmup) | bundled |
| `initial_tau` | starting cut threshold (recalibrated online) | 0.95 |
| `warmup_steps` | steps before cutting starts (probe still collects) | 2 |
| `K`, `agg` | sliding-window size / aggregation for the cut decision | 1 / median |
| `cut_ratio` | fraction of flagged failures to actually cut (rest spared) | 0.7 |
| `min_cut_chunk` | earliest chunk a cut may happen (avoid cutting too early) | 10 |
| `online_update.interval` | retrain every N PPO steps | 1 |
| `online_update.epochs` | epochs per retrain | 50 |
| `online_update.recent_steps` | steps of episodes kept in the online buffer | 3 |
| `online_update.base_data_path` | base data for retrain | bundled |

`env.train.v17_continuous_collect.target_trajectories` — trajectories to collect
**per rank** before switching to dummy mode.

---

## Logged metrics

Per PPO step the env worker logs:
```
[actprobe] succ=.. missed=.. cut=.. cut_rate=..% tau=.. saved=../..(..%)
[actprobe-detail] immune_succ=..(fp) never_flagged_timeout=..(blind) flagged_succ=..(near_fp)
[probe] retrain done ..s: tau=.. (p95+margin), loss=.., NS/NF (base+online)
```
Giving recall (cut + immune_timeout), blind (missed failures), false positives,
and env-step savings.
