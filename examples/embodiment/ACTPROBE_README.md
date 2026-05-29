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

## 1. Dependencies to download

| Item | Where | Used by |
|------|-------|---------|
| **RLinf-Gr00t-SFT-Long** (~7 GB) | https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Long | all modes (policy) |
| **Qwen3-Embedding-0.6B** (~1.2 GB) | https://huggingface.co/Qwen/Qwen3-Embedding-0.6B | actprobe (language conditioning) |
| probe_v19.pt (100 KB) | **bundled in this repo** (`checkpoints/probe_libero10_task0/`) | actprobe (offline-pretrained probe) |
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
| `checkpoint_path` | offline-pretrained probe weights (bundled) |
| `initial_tau` | starting cut threshold (recalibrated online) |
| `warmup_steps` | PPO steps before cutting starts (probe still collects) |
| `K`, `agg` | sliding-window size / aggregation for the cut decision |
| `online_update.*` | online retrain: interval, epochs, buffer, base data |

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
