# RoboCasa OpenPI Diagnostics Eval Design

## Goal

Add a standalone evaluation script for RoboCasa + OpenPI that records one JSONL row per episode. The output must include task success, the executed action trajectory, and per-step MuJoCo `mjData` diagnostics for early failure analysis. The script should not save observations.

## Non-Goals

- Do not add RoboCasa + GR00T support.
- Do not change the standard Ray-based embodied evaluation runner.
- Do not implement threshold-based early stopping or automatic failure classification in this change. The script records raw diagnostic signals so thresholds can be tuned later.
- Do not save image observations or full environment observation dictionaries.

## Entry Point

Add `examples/embodiment/eval_robocasa_openpi_diagnostics.py`.

The script will use Hydra and default to the existing `robocasa_closedrawer_ppo_openpi` configuration. It will not start Ray. It will instantiate `RobocasaEnv` directly and load the OpenPI policy through the existing OpenPI model factory.

The main loop is:

1. Build and validate the config with `runner.only_eval = True`.
2. Create a local `RobocasaEnv` for `cfg.env.eval`.
3. Load the OpenPI model from `cfg.actor.model`.
4. Reset the environment.
5. Predict action chunks with `predict_action_batch(..., mode="eval")`.
6. Step through the action chunk one action at a time.
7. After each step, fetch MuJoCo diagnostics from the RoboCasa subprocess env.
8. End the episode on success or truncation.
9. Write one JSON object to the output JSONL file.

Suggested Hydra overrides:

- `diagnostics.output_path`
- `diagnostics.num_episodes`
- `diagnostics.max_contacts`
- `diagnostics.include_model_names`
- `diagnostics.flush_every`

## MuJoCo Diagnostics Interface

Extend `rlinf/envs/robocasa/venv.py` with a read-only worker command named `get_mujoco_diagnostics`.

The command runs inside each RoboCasa subprocess and reads `env.sim.model` and `env.sim.data`. It returns JSON-serializable data for the current simulator state.

The returned diagnostic snapshot includes:

- `ncon`
- `contacts`
  - `dist`
  - `geom1`
  - `geom2`
  - `geom1_name`
  - `geom2_name`
  - `force`
- `qvel`
- `xpos`
- `xquat`
- `subtree_linvel`
- `energy`
- `kinetic_energy`
- `potential_energy`
- `body_names`
- `geom_names`

Contact force should be computed with `mujoco.mj_contactForce(model, data, contact_id, out)`. If the installed MuJoCo binding cannot provide the force, the snapshot should keep the rest of the contact data and set `force` to `null` with a short `force_error` string.

`max_contacts` limits the number of serialized contacts per step. Full kinematic, dynamic, and energy arrays are preserved.

Add a small pass-through method on `RobocasaSubprocEnv`, and if useful on `RobocasaEnv`, so the standalone script can call one method and receive a list of per-env diagnostic snapshots.

## JSONL Format

The file is newline-delimited JSON. Each row is one episode.

Required top-level fields:

- `episode_id`
- `env_id`
- `task_name`
- `seed`
- `success`
- `num_steps`
- `termination_reason`
- `task_description`
- `actions`
- `steps`

`actions` stores the full executed action trajectory as nested Python lists.

Each step record contains:

- `step`
- `reward`
- `success`
- `terminated`
- `truncated`
- `diagnostics`

The `diagnostics` field contains the MuJoCo snapshot described above.

The script should convert NumPy arrays, NumPy scalars, torch tensors, and booleans into standard JSON-compatible Python types before writing.

## Error Handling

The script should fail early with actionable messages when:

- the model path does not exist or cannot be loaded;
- RoboCasa, robosuite, MuJoCo, or OpenPI dependencies are missing;
- the selected config is not a RoboCasa + OpenPI config.

MuJoCo diagnostic collection should be best-effort for contact force only. A force extraction failure should not abort the episode.

## Testing

Add unit tests that do not require real RoboCasa assets or a real model checkpoint.

Coverage should include:

- diagnostic snapshot serialization from fake `model` and `data` objects;
- contact truncation with `max_contacts`;
- contact force failure fallback;
- JSON compatibility conversion for NumPy and torch values;
- episode record builder output shape and required fields.

Real end-to-end evaluation is intentionally not part of CI because it requires RoboCasa assets, MuJoCo/OpenGL, and an OpenPI checkpoint.

## Implementation Scope

Expected files:

- `examples/embodiment/eval_robocasa_openpi_diagnostics.py`
- `rlinf/envs/robocasa/venv.py`
- optionally `rlinf/envs/robocasa/robocasa_env.py`
- one or more focused unit test files under `tests/unit_tests/`

The implementation should preserve existing RoboCasa training and eval behavior unless the new diagnostics method is explicitly called.
