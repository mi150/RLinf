export EMBODIED_PATH=/mnt/miliang/RLinf/examples/embodiment && \
export REPO_PATH=/mnt/miliang/RLinf && \
export MUJOCO_GL=egl && \
python -m toolkits.rollout_eval.run \
    --config-path examples/embodiment/config \
    --config-name maniskill_ppo_openvlaoft \
    --skip-validate-cfg \
    --output-dir /mnt/miliang/RLinf/rollout_eval_output/openvlaoft \
    --override env.eval.total_num_envs=32 \
    --override env.eval.max_steps_per_rollout_epoch=2 \
    --override env.eval.video_cfg.save_video=False \
    --override env.eval.init_params.id=PickCube-v1 \
    --override env.eval.init_params.control_mode=pd_ee_delta_pose \
    --override env.eval.init_params.obs_mode=rgb \
    --override +env.eval.wrap_obs_mode=simple \
    --override +env.eval.reward_mode=raw \
    --override ~env.eval.init_params.obj_set \
    --override rollout.model.model_path=/mnt/RLinf/models/RLinf-OpenVLAOFT-ManiSkill-Base-Main \
    --override actor.model.model_path=/mnt/RLinf/models/RLinf-OpenVLAOFT-ManiSkill-Base-Main \
    --override actor.model.lora_path=/mnt/RLinf/models/RLinf-OpenVLAOFT-ManiSkill-Base-Lora \
    --override algorithm.sampling_params.do_sample=False

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
# --override algorithm.sampling_params.do_sample=False

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
# --override actor.model.model_path=/mnt/RLinf/models/RLinf-Gr00t-SFT-Spatial
