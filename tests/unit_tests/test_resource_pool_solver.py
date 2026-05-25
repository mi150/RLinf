import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from test_placement import create_fake_cluster

from rlinf.scheduler.resource_pool.config import ResourcePoolConfig
from rlinf.scheduler.resource_pool.pool import FineGrainedResourcePool
from rlinf.scheduler.resource_pool.solver import ResourcePoolSolver
from rlinf.utils.placement import HybridComponentPlacement


def test_default_solver_builds_env_per_env_cpu_groups() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {
                    "env": {"node_group": "node", "placement": "0:0-1"}
                },
                "resource_pool": {
                    "enabled": True,
                    "cpu": {
                        "enabled": True,
                        "pools": {"env_cpu": {"node_group": "node", "cores": "0-7"}},
                        "components": {
                            "env": {"pool": "env_cpu", "granularity": "per_env"}
                        },
                    },
                },
            },
            "env": {"train": {"total_num_envs": 4}, "eval": {"total_num_envs": 4}},
            "runner": {"only_eval": False, "val_check_interval": -1},
            "rollout": {"pipeline_stage_num": 1},
        }
    )
    cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=0)
    placement = HybridComponentPlacement(cfg, cluster)
    pool_cfg = ResourcePoolConfig.from_cluster_cfg(cfg.cluster)

    bindings = ResourcePoolSolver(pool_cfg, cfg, cluster, placement).solve()

    assert [binding.rank for binding in bindings["env"]] == [0, 1]
    assert bindings["env"][0].cpu.process_cpu_cores == (0, 1, 2, 3)
    assert bindings["env"][0].cpu.env_cpu_core_groups == ((0, 1), (2, 3))
    assert bindings["env"][1].cpu.process_cpu_cores == (4, 5, 6, 7)
    assert bindings["env"][1].cpu.env_cpu_core_groups == ((4, 5), (6, 7))


def test_default_solver_skips_gpu_binding_for_zero_quota() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {"env": "0"},
                "resource_pool": {
                    "enabled": True,
                    "gpu": {
                        "enabled": True,
                        "mode": "mps",
                        "pools": {
                            "gpu_pool": {"node_group": "cluster", "devices": "0"}
                        },
                        "components": {"env": {"pool": "gpu_pool", "sm_percent": 0}},
                    },
                },
            },
            "env": {"train": {"total_num_envs": 1}, "eval": {"total_num_envs": 1}},
            "runner": {"only_eval": False, "val_check_interval": -1},
            "rollout": {"pipeline_stage_num": 1},
        }
    )
    cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=1)
    placement = HybridComponentPlacement(cfg, cluster)
    pool_cfg = ResourcePoolConfig.from_cluster_cfg(cfg.cluster)

    bindings = ResourcePoolSolver(pool_cfg, cfg, cluster, placement).solve()

    assert bindings["env"][0].gpu is None


def test_default_solver_rejects_duplicate_mig_uuid() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {"rollout": "0-1"},
                "resource_pool": {
                    "enabled": True,
                    "gpu": {
                        "enabled": True,
                        "mode": "mig",
                        "pools": {
                            "mig_pool": {
                                "node_group": "cluster",
                                "mig_devices": [
                                    {
                                        "uuid": "MIG-A",
                                        "parent_gpu": 0,
                                        "sm_percent": 20,
                                    },
                                    {
                                        "uuid": "MIG-A",
                                        "parent_gpu": 1,
                                        "sm_percent": 20,
                                    },
                                ],
                            }
                        },
                        "components": {
                            "rollout": {"pool": "mig_pool", "sm_percent": 20}
                        },
                    },
                },
            },
            "env": {"train": {"total_num_envs": 1}, "eval": {"total_num_envs": 1}},
            "runner": {"only_eval": False, "val_check_interval": -1},
            "rollout": {"pipeline_stage_num": 1},
        }
    )
    cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=2)
    placement = HybridComponentPlacement(cfg, cluster)
    pool_cfg = ResourcePoolConfig.from_cluster_cfg(cfg.cluster)

    with pytest.raises(ValueError, match="Duplicate MIG UUID"):
        ResourcePoolSolver(pool_cfg, cfg, cluster, placement).solve()


def test_plan_file_mode_allows_explicit_cpu_sharing(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "bindings": [
                    {
                        "component": "env",
                        "rank": 0,
                        "cluster_node_rank": 0,
                        "node_group_label": "node",
                        "cpu": {
                            "process_cpu_cores": [0, 1],
                            "env_cpu_core_groups": [],
                        },
                        "gpu": None,
                    },
                    {
                        "component": "env",
                        "rank": 1,
                        "cluster_node_rank": 0,
                        "node_group_label": "node",
                        "cpu": {
                            "process_cpu_cores": [0, 1],
                            "env_cpu_core_groups": [],
                        },
                        "gpu": None,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {
                    "env": {"node_group": "node", "placement": "0:0-1"}
                },
                "resource_pool": {
                    "enabled": True,
                    "allocation_mode": "plan_file",
                    "allocation_plan_path": str(plan_path),
                },
            },
            "env": {"train": {"total_num_envs": 2}, "eval": {"total_num_envs": 2}},
            "runner": {"only_eval": False, "val_check_interval": -1},
            "rollout": {"pipeline_stage_num": 1},
        }
    )
    cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=0)
    placement = HybridComponentPlacement(cfg, cluster)

    pool = FineGrainedResourcePool.from_config(cfg, cluster, placement)

    assert pool.get_component_bindings("env")[0].cpu.process_cpu_cores == (0, 1)
    assert pool.summary["shared_cpu_cores"] == {
        "0": ["env:0", "env:1"],
        "1": ["env:0", "env:1"],
    }


def test_plan_file_mode_sorts_components_and_ranks(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "bindings": [
                    {
                        "component": "rollout",
                        "rank": 1,
                        "cluster_node_rank": 0,
                        "node_group_label": "cluster",
                        "cpu": None,
                        "gpu": None,
                    },
                    {
                        "component": "env",
                        "rank": 1,
                        "cluster_node_rank": 0,
                        "node_group_label": "node",
                        "cpu": None,
                        "gpu": None,
                    },
                    {
                        "component": "rollout",
                        "rank": 0,
                        "cluster_node_rank": 0,
                        "node_group_label": "cluster",
                        "cpu": None,
                        "gpu": None,
                    },
                    {
                        "component": "env",
                        "rank": 0,
                        "cluster_node_rank": 0,
                        "node_group_label": "node",
                        "cpu": None,
                        "gpu": None,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {
                    "env": {"node_group": "node", "placement": "0:0-1"},
                    "rollout": "0-1",
                },
                "resource_pool": {
                    "enabled": True,
                    "allocation_mode": "plan_file",
                    "allocation_plan_path": str(plan_path),
                },
            },
            "env": {"train": {"total_num_envs": 2}, "eval": {"total_num_envs": 2}},
            "runner": {"only_eval": False, "val_check_interval": -1},
            "rollout": {"pipeline_stage_num": 1},
        }
    )
    cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=2)
    placement = HybridComponentPlacement(cfg, cluster)

    pool = FineGrainedResourcePool.from_config(cfg, cluster, placement)

    assert list(pool.bindings) == ["env", "rollout"]
    assert [binding.rank for binding in pool.bindings["env"]] == [0, 1]
    assert [binding.rank for binding in pool.bindings["rollout"]] == [0, 1]
