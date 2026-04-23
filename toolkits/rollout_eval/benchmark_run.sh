### ENV-ONLY: behavior（需要 .venv-behavior，内含 omnigibson）
export EMBODIED_PATH=/data1/miliang/RLinf/examples/embodiment
export REPO_PATH=/data1/miliang/RLinf
export OMNIGIBSON_DATA_PATH=/data1/BEHAVIOR-1K-datasets/
export ISAAC_PATH=/data1/isaac-sim/
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

### BENCHMARK: MPS/MIG 场景矩阵（注：MIG 需提前手动创建）

export EMBODIED_PATH=/data1/miliang/RLinf/examples/embodiment && \
export REPO_PATH=/data1/miliang/RLinf && \
export MUJOCO_GL=egl

python -m toolkits.rollout_eval.benchmark.run \
    --config-path examples/embodiment/config \
    --scenario-set env_only_cpu_core \
    --cpu-bind-strategy default \
    --num-envs 32 \
    --env-model-preset robocasa_openpi \
    --config-name robocasa_closedrawer_ppo_openpi \
    --warmup-steps 1 \
    --measure-steps 10 \
    --output-dir /data1/miliang/RLinf/rollout_eval_output/benchmark_cpu_env_only_robocasa_64cores
# python -m toolkits.rollout_eval.benchmark.run \
#   --config-path examples/embodiment/config \
#   --scenario-set env_only_mps \
#   --mps-sm 20,40,60,80,100 \
#   --num-envs 32 \
#   --env-model-preset robocasa_openpi \
#   --config-name robocasa_closedrawer_ppo_openpi \
#   --warmup-steps 1 \
#   --measure-steps 5 \
#   --output-dir /data1/miliang/RLinf/rollout_eval_output/benchmark_mps_robocasa

### BENCHMARK: CPU core 场景矩阵

# python -m toolkits.rollout_eval.benchmark.run \
#   --config-path examples/embodiment/config \
#   --scenario-set env_only_cpu_core,concurrent_cpu_core \
#   --cpu-bind-cores 0-127 \
#   --cpu-bind-strategy even_split \
#   --num-envs 32 \
#   --env-model-preset robocasa_openpi \
#   --config-name robocasa_closedrawer_ppo_openpi \
#   --warmup-steps 1 \
#   --measure-steps 5 \
#   --output-dir /data1/miliang/RLinf/rollout_eval_output/benchmark_cpu_robocasa

# python -m toolkits.rollout_eval.benchmark.run \
#   --config-path examples/embodiment/config \
#   --scenario-set env_only_mps \
#   --mps-sm 20,40,60,80,100 \
#   --num-envs 32 \
#   --env-model-preset libero_gr00t \
#   --config-name libero_spatial_ppo_gr00t \
#   --warmup-steps 1 \
#   --measure-steps 5 \
#   --output-dir /data1/miliang/RLinf/rollout_eval_output/benchmark_mps_libero

# python -m toolkits.rollout_eval.benchmark.run \
#   --config-path examples/embodiment/config \
#   --scenario-set env_only_mps \
#   --mps-sm 20,40,60,80,100 \
#   --num-envs 32 \
#   --env-model-preset maniskill_openvlaoft \
#   --config-name maniskill_ppo_openvlaoft \
#   --warmup-steps 1 \
#   --measure-steps 5 \
#   --output-dir /data1/miliang/RLinf/rollout_eval_output/benchmark_mps_maniskill

# python -m toolkits.rollout_eval.benchmark.run \
#   --config-path examples/embodiment/config \
#   --scenario-set env_only_mps \
#   --mps-sm 20,40,60,80,100 \
#   --num-envs 32 \
#   --env-model-preset metaworld_openpi \
#   --config-name metaworld_50_ppo_openpi \
#   --warmup-steps 1 \
#   --measure-steps 5 \
#   --output-dir /data1/miliang/RLinf/rollout_eval_output/benchmark_mps_metaworld


# export EMBODIED_PATH=/data1/miliang/RLinf/examples/embodiment && \
# export REPO_PATH=/data1/miliang/RLinf && \
# export MUJOCO_GL=egl && \
# python -m toolkits.rollout_eval.benchmark.run \
#   --config-path examples/embodiment/config \
#   --config-name maniskill_ppo_openvlaoft \
#   --scenario-set concurrent_mig,env_only_mig,model_only_mig \
#   --mig-devices MIG-uuid-a,MIG-uuid-b \
#   --env-model-preset maniskill_openvlaoft,behavior_openpi,robocasa_openpi \
#   --warmup-steps 20 \
#   --measure-steps 200 \
#   --output-dir /data1/miliang/RLinf/rollout_eval_output/benchmark_mig
