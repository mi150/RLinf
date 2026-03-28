import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('EMBODIED_PATH','/mnt/miliang/RLinf/examples/embodiment')
os.environ.setdefault('REPO_PATH','/mnt/miliang/RLinf')

from hydra import compose, initialize_config_dir
from toolkits.rollout_eval.adapters.env_adapter import build_env_adapter

with initialize_config_dir(version_base='1.1', config_dir='/mnt/miliang/RLinf/examples/embodiment/config'):
    cfg = compose(config_name='maniskill_ppo_openvlaoft', overrides=['env.eval.total_num_envs=2'])

try:
    build_env_adapter(cfg, split='eval')
except Exception as e:
    print(type(e).__name__)
    print(str(e))