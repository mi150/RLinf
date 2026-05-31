# ActProbe on RoboCasa + pi0.5

Port of the LIBERO/GR00T ActProbe (autoreset + probe-based early-cut, see
`ACTPROBE_LIBERO_README.md`) to **RoboCasa + pi0.5 (OpenPI)**. The failure-detection
probe predicts which rollout episodes will fail mid-rollout and force-cuts them,
saving environment interactions.

**Status.** The early-cut mechanism is validated on the long-horizon, multi-step
task **PnPSinkToCounter**: the online-retrained probe cuts ~98% of failures with
zero false positives and saves ~82% of failure-episode interactions. RL training
itself (PPO on pi0.5) is not yet tuned for RoboCasa — over a short run the task
success rate does not improve. Tuning RL to raise SR is future work, separate from
the cut mechanism.

---

## What's here

- **`rlinf/envs/robocasa/conf_probe.py`** — `ConfProbe`: a lang-conditioned LSTM
  failure probe with online retraining, supporting **2-feat** `[action_norm,
  chunk_mse]` and **8-feat** `[action_norm, chunk_mse, eef_xyz, eef_velocity]`.
  `n_feat` is read from the checkpoint; features and online retrain are
  n_feat-generic.
- **`rlinf/envs/robocasa/robocasa_env.py`** — probe hook + `force_cut_envs` +
  auto-reset obs fix + **rollout-data collection** + domain randomization.
- **Eval-mode rollout-data collection** (the OOM-free way to collect probe
  training data) — see below. Touches `env_worker.py` / `huggingface_worker.py`.
- **`rlinf/models/embodiment/openpi/`** — pi0.5 RoboCasa dataconfig + value head.
- Bundled (`checkpoints/probe_robocasa/`): `pnpsinktocounter_8feat_eef_seed1_absts.pt`
  (+ 2-feat variant) and `robocasa_base_8feat_pnpsinktocounter.pkl`.

---

## Pipeline

```
1. collect rollout data   -> robocasa_pnpsinktocounter_collect.yaml   (only_eval, no OOM)
2. train a probe offline  -> (your trainer) -> pnpsinktocounter_8feat_eef_seed1_absts.pt
3. run actprobe RL        -> robocasa_pnpsinktocounter_actprobe.yaml   (probe cuts + online retrain)
```

### 1. Collecting rollout data in eval (no OOM)

Collecting labeled rollouts used to require *train* mode (to expose the model's
predicted action chunk `chains[:, -1]`), but the actor's FSDP train step OOMs on
pi0.5. The fix: a single switch, **`env.eval.collect_rollout.enabled`**, makes
the eval rollout forward `chains[:, -1]` to the env so the collect buffer fills
during **pure-inference eval** (`runner.only_eval=True`) — no actor train, no OOM,
and `only_eval` uses the initial checkpoint, i.e. the step-0 policy. Off by
default → the eval path is unchanged for everyone else.

```bash
bash examples/embodiment/run_embodiment.sh robocasa_pnpsinktocounter_collect ROBOCASA
```
Writes per-rank `rollout_rank{N}.pkl`, each episode:
`{action_chunk (M,50,7), eef_pos (M,3), success, task_id, instruction, length}`.

**Important — sampling temperature.** The pi0.5 RoboCasa policy is miscalibrated:
at `temperature_eval=0.6` most tasks collapse to ~0% SR; collect at
**`temperature=1.0`** (the train-rollout sampling) to get a success/failure mix.

**Important — task choice.** The converted pi0.5 checkpoint only does a subset of
RoboCasa well, and very unevenly: pick-into-a-container tasks (`PnPCounterToMicrowave`,
`PnPMicrowaveToCounter`) stay near 0–6% even at temperature 1.0 (an init-state
distribution mismatch — the policy was trained on human-demo starts, RLinf resets
randomly), while open-target pick-and-place performs substantially better. `PnPSinkToCounter`
reaches ~24% SR at temperature 1.0, providing enough success samples and
long-horizon trajectories for a meaningful probe.

### 2. Generate the 8-feat base data for online retrain

`examples/embodiment/tools/gen_robocasa_8feat_base.py` turns collected rollouts
into the raw 8-feat base set (`feat (M,9) = [action_norm, chunk_mse, x,y,z,
vx,vy,vz, step]`) that online retrain mixes with the live buffer.

### 3. Run actprobe RL

```bash
bash examples/embodiment/run_embodiment.sh robocasa_pnpsinktocounter_actprobe ROBOCASA
```
`env.train.probe_cfg` loads the bundled 8-feat probe; `warmup_steps=2` (no cut for
2 steps), then the probe cuts flagged failures and retrains online each PPO step.
Logged metrics are the same as LIBERO (`[actprobe]` / `[actprobe-cm]`).

---

## Validation result (PnPSinkToCounter, 8-feat, steps 3–6)

| metric | value |
|--------|-------|
| recall (failures cut) | **~97–100%** |
| false positives (winners cut) | **0** |
| env-steps saved | **~81–86%** |
| mean cut position | **chunk 7.8 (≈157 env steps)** |

In this task, successes run ~10.3 chunks (≈206 steps) and failures time out at 32
chunks (640 steps). The probe cuts flagged failures at chunk ~7.8 — 25% of the
timeout horizon, and earlier than the average success completes — with zero false
positives. The probe predicts failure from the action-chunk and end-effector
features, saving ~75% of each failed episode's interactions.

---

## Dependencies

| Item | Where |
|------|-------|
| **pi0.5 RoboCasa policy** (converted to PyTorch) | set `actor.model.model_path` (`/path/to/model/RLinf-Pi05-RoboCasa`) |
| **Qwen3-Embedding-0.6B** | https://huggingface.co/Qwen/Qwen3-Embedding-0.6B (probe lang conditioning) |
| RoboCasa + OpenPI env | based on the RLinf RoboCasa/OpenPI integration |
| probe ckpt + base data | **bundled** in `checkpoints/probe_robocasa/` |
