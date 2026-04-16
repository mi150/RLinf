### ENV-ONLY: behavior（需要 .venv-behavior，内含 omnigibson）
export EMBODIED_PATH=/mnt/miliang/RLinf/examples/embodiment
export REPO_PATH=/mnt/miliang/RLinf
export OMNIGIBSON_DATA_PATH=/mnt/BEHAVIOR-1K-datasets/
export ISAAC_PATH=/mnt/isaac-sim/
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

/mnt/miliang/RLinf/.venv-behavior/bin/python -m toolkits.rollout_eval.run \
    --config-path examples/embodiment/config \
    --config-name behavior_ppo_openpi \
    --skip-validate-cfg \
    --env-only \
    --action-dim 23 \
    --output-dir /mnt/miliang/RLinf/rollout_eval_output/env_only_behavior \
    --override env.eval.total_num_envs=1 \
    --override env.eval.max_steps_per_rollout_epoch=10 \
    --override env.eval.video_cfg.save_video=False \
    --profile-batch-sizes 1

# export MUJOCO_GL=egl && \
# python -m toolkits.rollout_eval.run \
#     --config-path examples/embodiment/config \
#     --config-name maniskill_ppo_openvlaoft \
#     --skip-validate-cfg \
#     --env-only \
#     --action-dim 7 \
#     --output-dir /mnt/miliang/RLinf/rollout_eval_output/env_only_maniskill \
#     --override env.eval.total_num_envs=32 \
#     --override env.eval.max_steps_per_rollout_epoch=10 \
#     --override env.eval.video_cfg.save_video=False \
#     --override env.eval.init_params.id=PickCube-v1 \
#     --override env.eval.init_params.control_mode=pd_ee_delta_pose \
#     --override env.eval.init_params.obs_mode=rgb \
#     --override +env.eval.wrap_obs_mode=simple \
#     --override +env.eval.reward_mode=raw \
#     --override ~env.eval.init_params.obj_set \
#     --profile-batch-sizes 1,4,8,16,32

# export EMBODIED_PATH=/mnt/miliang/RLinf/examples/embodiment && \
# export REPO_PATH=/mnt/miliang/RLinf && \
# export MUJOCO_GL=egl && \
# python -m toolkits.rollout_eval.run \
# --config-path examples/embodiment/config \
# --config-name metaworld_50_ppo_openpi \
# --skip-validate-cfg \
# --output-dir /mnt/miliang/RLinf/rollout_eval_output/rollout_eval_openpi_metaworld \
# --override env.eval.total_num_envs=32 \
# --override env.eval.max_steps_per_rollout_epoch=2 \
# --override env.eval.video_cfg.save_video=False \
# --override rollout.model.model_path=/mnt/RLinf/models/RLinf-Pi0-MetaWorld-SFT \
# --override actor.model.model_path=/mnt/RLinf/models/RLinf-Pi0-MetaWorld-SFT \
# --override algorithm.sampling_params.do_sample=False \
# --profile-batch-sizes 1,4,8,16,32

### GR00T（LIBERO）

# export EMBODIED_PATH=/mnt/miliang/RLinf/examples/embodiment && \
# export REPO_PATH=/mnt/miliang/RLinf && \
# export MUJOCO_GL=egl && \
# python -m toolkits.rollout_eval.run \
# --config-path examples/embodiment/config \
# --config-name libero_spatial_ppo_gr00t \
# --output-dir /mnt/miliang/RLinf/rollout_eval_output/gr00t \
# --skip-validate-cfg \
# --override env.eval.total_num_envs=32 \
# --override env.eval.max_steps_per_rollout_epoch=2 \
# --override env.eval.video_cfg.save_video=False \
# --override rollout.model.model_path=/mnt/RLinf/models/RLinf-Gr00t-SFT-Spatial \
# --override actor.model.model_path=/mnt/RLinf/models/RLinf-Gr00t-SFT-Spatial \
# --profile-batch-sizes 1,4,8,16,32


### BENCHMARK: MPS/MIG 场景矩阵（注：MIG 需提前手动创建）

# export EMBODIED_PATH=/mnt/miliang/RLinf/examples/embodiment && \
# export REPO_PATH=/mnt/miliang/RLinf && \
# export MUJOCO_GL=egl && \
# python -m toolkits.rollout_eval.benchmark.run \
#   --config-path examples/embodiment/config \
#   --config-name maniskill_ppo_openvlaoft \
#   --scenario-set concurrent_mps,env_only_mps,model_only_mps \
#   --mps-sm 20,40,60 \
#   --env-model-preset maniskill_openvlaoft,behavior_openpi \
#   --warmup-steps 20 \
#   --measure-steps 200 \
#   --output-dir /mnt/miliang/RLinf/rollout_eval_output/benchmark_mps

# export EMBODIED_PATH=/mnt/miliang/RLinf/examples/embodiment && \
# export REPO_PATH=/mnt/miliang/RLinf && \
# export MUJOCO_GL=egl && \
# python -m toolkits.rollout_eval.benchmark.run \
#   --config-path examples/embodiment/config \
#   --config-name maniskill_ppo_openvlaoft \
#   --scenario-set concurrent_mig,env_only_mig,model_only_mig \
#   --mig-devices MIG-uuid-a,MIG-uuid-b \
#   --env-model-preset maniskill_openvlaoft,behavior_openpi \
#   --warmup-steps 20 \
#   --measure-steps 200 \
#   --output-dir /mnt/miliang/RLinf/rollout_eval_output/benchmark_mig

# export EMBODIED_PATH=/mnt/miliang/RLinf/examples/embodiment && \
# export REPO_PATH=/mnt/miliang/RLinf && \
# export MUJOCO_GL=egl && \
# python -m toolkits.rollout_eval.benchmark.run \
#   --config-path examples/embodiment/config \
#   --config-name libero_spatial_ppo_gr00t \
#   --scenario-set model_only_mps,concurrent_mps \
#   --mps-sm 20,40,60 \
#   --num-envs 32 \
#   --env-model-preset libero_gr00t \
#   --warmup-steps 20 \
#   --measure-steps 200 \
#   --pipeline-queue-timeout-s 8 \
#   --output-dir /mnt/miliang/RLinf/rollout_eval_output/benchmark_libero_gr00t
