from omegaconf import OmegaConf

from toolkits.rollout_eval.config_bridge import build_eval_runtime_config


def test_build_eval_runtime_config_reads_num_envs_from_hydra() -> None:
    cfg = OmegaConf.create(
        {
            "env": {
                "eval": {
                    "env_type": "maniskill",
                    "total_num_envs": 16,
                    "group_size": 2,
                    "max_steps_per_rollout_epoch": 120,
                }
            },
            "actor": {
                "seed": 7,
                "model": {
                    "model_type": "openvla_oft",
                    "num_action_chunks": 8,
                    "model_path": "/tmp/model",
                },
            },
            "rollout": {
                "model": {
                    "model_path": "/tmp/model-rollout",
                    "precision": "bf16",
                }
            },
            "algorithm": {"sampling_params": {"temperature_eval": -1}},
        }
    )

    runtime = build_eval_runtime_config(cfg)

    assert runtime.env_type == "maniskill"
    assert runtime.num_envs == 16
    assert runtime.group_size == 2
    assert runtime.model_type == "openvla_oft"
    assert runtime.model_path == "/tmp/model-rollout"
    assert runtime.total_steps == 120
    assert runtime.warmup_steps > 0
