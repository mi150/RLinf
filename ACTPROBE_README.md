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

## Validation

A 10-step end-to-end run on **LIBERO-10 task 0** (GR00T-N1.5, 4 GPUs). This run
uses a reduced rollout (`target_trajectories=64`) for a fast functional check, so
late-step probe stats are noisy; the trends are what matter, not exact late values.

| step | task SR | recall | precision | FPR | env-steps saved |
|-----:|--------:|-------:|----------:|----:|----------------:|
| 1–2 (warmup) | 16→25% | →32% | ~90% | <10% | 0% (collect only, no cut) |
| 3  | 25% | 58% | 94% | 3% | **+16%** |
| 6  | 35% | 96% | 93% | 3% | +27% |
| 8  | 74% | 91% | 88% | 1% | +37% |
| 10 | 76% | 100% | 69%\* | 3% | **+42%** |

\* Late-step precision is small-sample noise: once SR is high there are very few
failures left, so the immune validation set is tiny. Recall stays high and the
absolute FP count is ~5. For clean paper-grade curves, run full
`target_trajectories=128` (slower; this run traded statistical resolution for speed).

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

The probe is a **binary failure detector** (positive class = "this episode will
fail"). Per PPO step (per env rank / `stage`) the env worker logs three lines,
named with standard classifier terms. Counts are over the episodes that finished
in that step; lengths are in **chunks** (1 chunk = `num_action_chunks` env steps,
so a 480-step episode = 96 chunks).

### `[actprobe]` — per-step summary
```
[actprobe] stage=N: success=.. timeout=.. cut=.. recall=..% precision=..% FPR=..% tau=..
           avg_len: success=.. cut=.. timeout=.. saved=S/P(..%)
```
| field | meaning | how it's computed |
|-------|---------|-------------------|
| `success` | episodes that solved the task | count of `outcome == success` (real env termination) |
| `timeout` | episodes that ran to natural timeout (not cut) | count of `outcome == timeout` (= spared-immune failures + never-flagged failures) |
| `cut` | episodes the probe force-cut early | count of `outcome == probe_cut` |
| **`recall`** | of all failures, the share the probe caught | `TP / (TP + FN)` (see `[actprobe-cm]`) |
| **`precision`** | of flagged episodes, the share that really failed | `TP_immune / (TP_immune + FP_immune)`, measured on the **spared (immune)** set — see note below |
| **`FPR`** | false-positive rate (winners wrongly flagged) | `FP / success` |
| `tau` | current cut threshold | `min(P95(held-out success scores) + 0.02, 0.99)`, recalibrated from scratch each step (capped at 0.99) |
| `avg_len` | mean episode length per group (chunks) | a natural timeout reaches `max_chunks = max_episode_steps / num_action_chunks` (= 96 here) |
| **`saved = S/P (%)`** | env-step budget saved by cutting failures | `S = Σ_cut (max_chunks − cut_len)`, `P = (cut + timeout) × max_chunks`. Positive ⇒ fewer interactions. `max_chunks` is config-derived — do **not** hardcode it (a wrong constant inverts the sign) |

### `[actprobe-cm]` — confusion matrix
```
[actprobe-cm] stage=N: TP=.. FP=.. FN=.. TN=..
              [TP=cut(..)+immune_fail(..), FP=on_success_immune(..)+on_success_precut(..), FN=blind_timeout(..)]
```
| term | meaning | composition |
|------|---------|-------------|
| **TP** | flagged, and it was a failure | `cut` (assumed failures) + `immune_fail` (flagged-but-spared episodes that did time out — verified) |
| **FP** | flagged, but it succeeded | `on_success_immune` (spared then succeeded) + `on_success_precut` (succeeded before the cut landed) |
| **FN** | failed, but never flagged | `blind_timeout` |
| **TN** | succeeded, never flagged | `success − FP` |

**Why precision uses the immune set.** Cut episodes are terminated early, so their
true success/failure is never observed — counting them as TP would be circular.
ActProbe therefore **spares a random `1 − cut_ratio` of flagged envs** (they run to
natural timeout). Because sparing is *random* among flagged episodes, the spared
set is an unbiased sample, and its precision / FP rate is an unbiased estimate for
the cut episodes too. `recall` and `FPR` count `cut` as caught-failures (the
standard detection-rate convention); `precision` is reported on the immune set only.

### `[probe] retrain done` — online retrain summary
```
[probe] retrain done ..s: tau=T (p95=..,max=..+0.02), loss=.., NS/NF (base=B+online=O)
```
| field | meaning |
|-------|---------|
| `..s` | retrain wall-clock (background thread) |
| `tau` | new threshold = `min(p95_succ + 0.02, 0.99)` |
| `loss` | final retrain loss |
| `NS / NF` | # success / # failure episodes in the merged training set |
| `base=B + online=O` | B bundled base episodes + O recent on-policy episodes |

> A second, **per-rollout-epoch** diagnostic line `[probe] epoch=E stage=N: ...`
> also exists, with `flagged / cut / FP / FN / cut@`. Its `dummy_skip=../..` field is
> **not** the probe-cut `saved` above — it is the v17 continuous-collect skip ratio
> (chunks skipped after the per-rank trajectory target is met). The names are kept
> distinct (`dummy_skip` vs `saved`) so the two are not conflated.
