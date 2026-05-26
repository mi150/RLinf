import pytest
from omegaconf import OmegaConf

from rlinf.scheduler.resource_pool.config import ResourcePoolConfig


def test_resource_pool_config_disabled_when_missing() -> None:
    cfg = OmegaConf.create({"cluster": {"num_nodes": 1}})
    parsed = ResourcePoolConfig.from_cluster_cfg(cfg.cluster)
    assert parsed.enabled is False


def test_resource_pool_config_parses_cpu_and_mps_components() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "resource_pool": {
                    "enabled": True,
                    "allocation_mode": "default",
                    "cpu": {
                        "enabled": True,
                        "pools": {"env_cpu": {"node_group": "node", "cores": "0-7"}},
                        "components": {
                            "env": {"pool": "env_cpu", "granularity": "per_env"}
                        },
                    },
                    "gpu": {
                        "enabled": True,
                        "mode": "mps",
                        "pools": {
                            "gpu_pool": {"node_group": "cluster", "devices": "0-1"}
                        },
                        "components": {
                            "rollout": {"pool": "gpu_pool", "sm_percent": 40}
                        },
                    },
                }
            }
        }
    )

    parsed = ResourcePoolConfig.from_cluster_cfg(cfg.cluster)

    assert parsed.enabled is True
    assert parsed.cpu.components["env"].granularity == "per_env"
    assert parsed.gpu.components["rollout"].sm_percent == 40


def test_resource_pool_config_skips_disabled_cpu_config() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "resource_pool": {
                    "enabled": True,
                    "cpu": {
                        "enabled": False,
                        "components": {
                            "env": {"pool": "missing_pool", "granularity": "per_env"}
                        },
                    },
                }
            }
        }
    )

    parsed = ResourcePoolConfig.from_cluster_cfg(cfg.cluster)

    assert parsed.cpu.enabled is False
    assert parsed.cpu.pools == {}
    assert parsed.cpu.components == {}


def test_resource_pool_config_skips_disabled_gpu_config() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "resource_pool": {
                    "enabled": True,
                    "gpu": {
                        "enabled": False,
                        "mode": "invalid",
                        "components": {
                            "rollout": {"pool": "missing_pool", "sm_percent": 40}
                        },
                    },
                }
            }
        }
    )

    parsed = ResourcePoolConfig.from_cluster_cfg(cfg.cluster)

    assert parsed.gpu.enabled is False
    assert parsed.gpu.pools == {}
    assert parsed.gpu.components == {}


def test_resource_pool_config_rejects_invalid_sm_percent() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "resource_pool": {
                    "enabled": True,
                    "gpu": {
                        "enabled": True,
                        "mode": "mps",
                        "pools": {"gpu_pool": {"devices": "0"}},
                        "components": {
                            "rollout": {"pool": "gpu_pool", "sm_percent": 25}
                        },
                    },
                }
            }
        }
    )

    with pytest.raises(ValueError, match="sm_percent"):
        ResourcePoolConfig.from_cluster_cfg(cfg.cluster)


def test_plan_file_mode_requires_path() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "resource_pool": {"enabled": True, "allocation_mode": "plan_file"}
            }
        }
    )
    with pytest.raises(ValueError, match="allocation_plan_path"):
        ResourcePoolConfig.from_cluster_cfg(cfg.cluster)


def test_mig_device_config_requires_parent_gpu() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "resource_pool": {
                    "enabled": True,
                    "gpu": {
                        "enabled": True,
                        "mode": "mig",
                        "pools": {
                            "mig_pool": {
                                "mig_devices": [
                                    {
                                        "uuid": "MIG-A",
                                        "sm_percent": 20,
                                    }
                                ],
                            }
                        },
                        "components": {
                            "rollout": {"pool": "mig_pool", "sm_percent": 20}
                        },
                    },
                }
            }
        }
    )

    with pytest.raises(ValueError, match="parent_gpu"):
        ResourcePoolConfig.from_cluster_cfg(cfg.cluster)
