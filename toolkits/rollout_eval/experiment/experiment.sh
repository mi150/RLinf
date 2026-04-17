#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"
# Force Python to ignore stale root-owned .pyc caches
export PYTHONDONTWRITEBYTECODE=1

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH
echo "Using REPO_PATH=${REPO_PATH}"
echo "Using EMBODIED_PATH=${EMBODIED_PATH}"
echo "Using PYTHONPATH=${PYTHONPATH}"
# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Base path to Isaac Sim, only required when running the behavior experiment.
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

if [ -z "$1" ]; then
    CONFIG_NAME="maniskill_ppo_openvlaoft"
else
    CONFIG_NAME=$1
fi

# NOTE: Set the active robot platform (required for correct action dimension and normalization), supported platforms are LIBERO, ALOHA, BRIDGE, default is LIBERO
ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"LIBERO"}}

export ROBOT_PLATFORM

# Libero variant: standard, pro, plus
export LIBERO_TYPE=${LIBERO_TYPE:-"standard"}
if [ "$LIBERO_TYPE" == "pro" ]; then
    export LIBERO_PERTURBATION="all"  # all,swap,object,lan
    echo "Evaluation Mode: LIBERO-PRO | Perturbation: $LIBERO_PERTURBATION"
elif [ "$LIBERO_TYPE" == "plus" ]; then
    export LIBERO_SUFFIX="all"
    echo "Evaluation Mode: LIBERO-PLUS | Suffix: $LIBERO_SUFFIX"
else
    echo "Evaluation Mode: Standard LIBERO"
fi

# python -B -m toolkits.rollout_eval.experiment.run_experiment \
#     --config-path examples/embodiment/config \
#     --config-name libero_spatial_ppo_gr00t \
#     --phases cache_eval \
#     --record-video \
#     --seeds 42 \
#     --cache-mode cross_step_similarity \
#     --output-dir ./experiment_output

# ---------------------------------------------------------------------------
# Phase 3: Action replacement validation using pre-analyzed bottleneck pairs
#
# Pairs selected from /mnt/RLinf/results/bottleneck_success_test1000/
#   Pair 988: sid=170960 rank=1 env=19 ep=190  <->  sid=408399 rank=0 env=1  ep=176  K_B=6  post_cos=1.0
#   Pair 193: sid=27850  rank=0 env=31 ep=211  <->  sid=685703 rank=1 env=24 ep=182  K_B=5  post_cos=1.0
#
# Trajectory A (source for replacement actions):
#   step_0_sid_170960_rank_1_env_19_episode_190_success.pkl  (K_B=6)
#   step_0_sid_27850_rank_0_env_31_episode_211_success.pkl   (K_B=5)
#
# Trajectory B (live sim target - loaded via seed matching):
#   step_0_sid_408399_rank_0_env_1_episode_176_success.pkl
#   step_0_sid_685703_rank_1_env_24_episode_182_success.pkl
# ---------------------------------------------------------------------------

COLLECTED_DATA="/mnt/RLinf/logs/20260327-09:49:04-maniskill_async_ppo_openvlaoft/collected_data"
PHASE3_TRAJ_DIR="./experiment_output/phase3_source_trajs"
mkdir -p "${PHASE3_TRAJ_DIR}"

# Copy source trajectories (the ones whose bottleneck actions will be replayed)
cp "${COLLECTED_DATA}/step_0_sid_170960_rank_1_env_19_episode_190_success.pkl" "${PHASE3_TRAJ_DIR}/"
cp "${COLLECTED_DATA}/step_0_sid_27850_rank_0_env_31_episode_211_success.pkl"  "${PHASE3_TRAJ_DIR}/"

# Run Phase 3: open-loop replay of source trajectories in live sim
# --action-source external  : load actions from pkl files
# --external-trajectory-seeds: sid values parsed from filenames (170960, 27850)
# --bottleneck-k-b 5        : conservative K_B (min of the two pairs)
python -B -m toolkits.rollout_eval.experiment.run_experiment \
    --config-path examples/embodiment/config \
    --config-name maniskill_ppo_openvlaoft \
    --phases action_replace \
    --action-source external \
    --external-trajectory-dir "${PHASE3_TRAJ_DIR}" \
    --external-trajectory-seeds 2599,15641 \
    --bottleneck-k-b 1 \
    --record-video \
    --output-dir ./experiment_output
