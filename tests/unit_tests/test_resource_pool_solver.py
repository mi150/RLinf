import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from test_placement import create_fake_cluster

from rlinf.scheduler.resource_pool.bindings import WorkerResourceBinding
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


def test_default_solver_rejects_shared_cpu_pool_across_components() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {
                    "actor": {"node_group": "node", "placement": "0"},
                    "env": {"node_group": "node", "placement": "0"},
                },
                "resource_pool": {
                    "enabled": True,
                    "cpu": {
                        "enabled": True,
                        "pools": {"shared": {"node_group": "node", "cores": "0-3"}},
                        "components": {
                            "actor": {"pool": "shared"},
                            "env": {"pool": "shared"},
                        },
                    },
                },
            },
            "env": {"train": {"total_num_envs": 1}, "eval": {"total_num_envs": 1}},
            "runner": {"only_eval": False, "val_check_interval": -1},
            "rollout": {"pipeline_stage_num": 1},
        }
    )
    cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=0)
    placement = HybridComponentPlacement(cfg, cluster)
    pool_cfg = ResourcePoolConfig.from_cluster_cfg(cfg.cluster)

    with pytest.raises(ValueError, match="CPU pool|exclusive"):
        ResourcePoolSolver(pool_cfg, cfg, cluster, placement).solve()


def test_default_solver_rejects_overlapping_cpu_pools() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {
                    "actor": {"node_group": "node", "placement": "0"},
                    "env": {"node_group": "node", "placement": "0"},
                },
                "resource_pool": {
                    "enabled": True,
                    "cpu": {
                        "enabled": True,
                        "pools": {
                            "actor_cpu": {"node_group": "node", "cores": "0-3"},
                            "env_cpu": {"node_group": "node", "cores": "2-5"},
                        },
                        "components": {
                            "actor": {"pool": "actor_cpu"},
                            "env": {"pool": "env_cpu"},
                        },
                    },
                },
            },
            "env": {"train": {"total_num_envs": 1}, "eval": {"total_num_envs": 1}},
            "runner": {"only_eval": False, "val_check_interval": -1},
            "rollout": {"pipeline_stage_num": 1},
        }
    )
    cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=0)
    placement = HybridComponentPlacement(cfg, cluster)
    pool_cfg = ResourcePoolConfig.from_cluster_cfg(cfg.cluster)

    with pytest.raises(ValueError, match="CPU core|exclusive"):
        ResourcePoolSolver(pool_cfg, cfg, cluster, placement).solve()


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


def test_mps_binding_requires_all_visible_devices_in_pool() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {"rollout": "0-1:0"},
                "resource_pool": {
                    "enabled": True,
                    "gpu": {
                        "enabled": True,
                        "mode": "mps",
                        "pools": {
                            "gpu_pool": {"node_group": "cluster", "devices": "0"}
                        },
                        "components": {
                            "rollout": {"pool": "gpu_pool", "sm_percent": 20}
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

    with pytest.raises(ValueError, match="GPU pool|visible"):
        ResourcePoolSolver(pool_cfg, cfg, cluster, placement).solve()


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


def test_solver_solve_is_repeatable_for_mig_bindings() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {"rollout": "0"},
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
                                    }
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
    cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=1)
    placement = HybridComponentPlacement(cfg, cluster)
    pool_cfg = ResourcePoolConfig.from_cluster_cfg(cfg.cluster)
    solver = ResourcePoolSolver(pool_cfg, cfg, cluster, placement)

    first = solver.solve()
    second = solver.solve()

    assert first["rollout"][0].gpu.mig_device_uuid == "MIG-A"
    assert second["rollout"][0].gpu.mig_device_uuid == "MIG-A"


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
        "node0:cpu0": ["env:0", "env:1"],
        "node0:cpu1": ["env:0", "env:1"],
    }


def test_summary_keys_include_node_rank_for_local_ids(tmp_path: Path) -> None:
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
                            "process_cpu_cores": [0],
                            "env_cpu_core_groups": [],
                        },
                        "gpu": {
                            "mode": "mps",
                            "sm_percent": 20,
                            "visible_devices": ["0"],
                            "parent_gpu": 0,
                        },
                    },
                    {
                        "component": "env",
                        "rank": 1,
                        "cluster_node_rank": 1,
                        "node_group_label": "node",
                        "cpu": {
                            "process_cpu_cores": [0],
                            "env_cpu_core_groups": [],
                        },
                        "gpu": {
                            "mode": "mps",
                            "sm_percent": 20,
                            "visible_devices": ["0"],
                            "parent_gpu": 0,
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 2,
                "component_placement": {
                    "env": {"node_group": "node", "placement": "0-1"}
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
    cluster = create_fake_cluster(num_nodes=2, accelerators_per_node=1)
    placement = HybridComponentPlacement(cfg, cluster)

    pool = FineGrainedResourcePool.from_config(cfg, cluster, placement)

    assert pool.summary["shared_cpu_cores"] == {}
    assert pool.summary["mps_gpu_totals"] == {
        "node0:gpu0": 20,
        "node1:gpu0": 20,
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


def test_plan_file_mode_rejects_nonexistent_component_rank(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "bindings": [
                    {
                        "component": "env",
                        "rank": 2,
                        "cluster_node_rank": 0,
                        "node_group_label": "node",
                        "cpu": None,
                        "gpu": None,
                    }
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

    with pytest.raises(ValueError, match="plan file binding|rank 2"):
        FineGrainedResourcePool.from_config(cfg, cluster, placement)


def test_plan_file_mode_rejects_missing_component_rank(tmp_path: Path) -> None:
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
                        "cpu": None,
                        "gpu": None,
                    }
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

    with pytest.raises(ValueError, match="missing ranks"):
        FineGrainedResourcePool.from_config(cfg, cluster, placement)


def test_plan_file_mode_rejects_missing_configured_component(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"bindings": []}), encoding="utf-8")
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
                    "cpu": {
                        "enabled": True,
                        "pools": {"env_cpu": {"node_group": "node", "cores": "0-1"}},
                        "components": {"env": {"pool": "env_cpu"}},
                    },
                },
            },
            "env": {"train": {"total_num_envs": 2}, "eval": {"total_num_envs": 2}},
            "runner": {"only_eval": False, "val_check_interval": -1},
            "rollout": {"pipeline_stage_num": 1},
        }
    )
    cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=0)
    placement = HybridComponentPlacement(cfg, cluster)

    with pytest.raises(ValueError, match="missing components"):
        FineGrainedResourcePool.from_config(cfg, cluster, placement)


def test_plan_file_mode_rejects_unknown_component(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "bindings": [
                    {
                        "component": "missing",
                        "rank": 0,
                        "cluster_node_rank": 0,
                        "node_group_label": "cluster",
                        "cpu": None,
                        "gpu": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {"rollout": "0"},
                "resource_pool": {
                    "enabled": True,
                    "allocation_mode": "plan_file",
                    "allocation_plan_path": str(plan_path),
                },
            },
            "env": {"train": {"total_num_envs": 1}, "eval": {"total_num_envs": 1}},
            "runner": {"only_eval": False, "val_check_interval": -1},
            "rollout": {"pipeline_stage_num": 1},
        }
    )
    cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=1)
    placement = HybridComponentPlacement(cfg, cluster)

    with pytest.raises(ValueError, match="unknown component"):
        FineGrainedResourcePool.from_config(cfg, cluster, placement)


def test_plan_file_mode_rejects_invalid_sm_percent(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "bindings": [
                    {
                        "component": "rollout",
                        "rank": 0,
                        "cluster_node_rank": 0,
                        "node_group_label": "cluster",
                        "cpu": None,
                        "gpu": {
                            "mode": "mps",
                            "sm_percent": 25,
                            "visible_devices": ["0"],
                            "parent_gpu": 0,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {"rollout": "0"},
                "resource_pool": {
                    "enabled": True,
                    "allocation_mode": "plan_file",
                    "allocation_plan_path": str(plan_path),
                },
            },
            "env": {"train": {"total_num_envs": 1}, "eval": {"total_num_envs": 1}},
            "runner": {"only_eval": False, "val_check_interval": -1},
            "rollout": {"pipeline_stage_num": 1},
        }
    )
    cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=1)
    placement = HybridComponentPlacement(cfg, cluster)

    with pytest.raises(ValueError, match="sm_percent"):
        FineGrainedResourcePool.from_config(cfg, cluster, placement)


def test_plan_file_mode_requires_gpu_sm_percent(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "bindings": [
                    {
                        "component": "rollout",
                        "rank": 0,
                        "cluster_node_rank": 0,
                        "node_group_label": "cluster",
                        "cpu": None,
                        "gpu": {
                            "mode": "mps",
                            "visible_devices": ["0"],
                            "parent_gpu": 0,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {"rollout": "0"},
                "resource_pool": {
                    "enabled": True,
                    "allocation_mode": "plan_file",
                    "allocation_plan_path": str(plan_path),
                },
            },
            "env": {"train": {"total_num_envs": 1}, "eval": {"total_num_envs": 1}},
            "runner": {"only_eval": False, "val_check_interval": -1},
            "rollout": {"pipeline_stage_num": 1},
        }
    )
    cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=1)
    placement = HybridComponentPlacement(cfg, cluster)

    with pytest.raises(ValueError, match="sm_percent"):
        FineGrainedResourcePool.from_config(cfg, cluster, placement)


def test_plan_file_mode_rejects_null_gpu_for_configured_gpu_component(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "bindings": [
                    {
                        "component": "rollout",
                        "rank": 0,
                        "cluster_node_rank": 0,
                        "node_group_label": "cluster",
                        "cpu": None,
                        "gpu": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {"rollout": "0"},
                "resource_pool": {
                    "enabled": True,
                    "allocation_mode": "plan_file",
                    "allocation_plan_path": str(plan_path),
                    "gpu": {
                        "enabled": True,
                        "mode": "mps",
                        "pools": {
                            "gpu_pool": {"node_group": "cluster", "devices": "0"}
                        },
                        "components": {
                            "rollout": {"pool": "gpu_pool", "sm_percent": 20}
                        },
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

    with pytest.raises(ValueError, match="GPU binding"):
        FineGrainedResourcePool.from_config(cfg, cluster, placement)


def test_plan_file_mode_validates_mps_visible_devices(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "bindings": [
                    {
                        "component": "rollout",
                        "rank": 0,
                        "cluster_node_rank": 0,
                        "node_group_label": "cluster",
                        "cpu": None,
                        "gpu": {
                            "mode": "mps",
                            "sm_percent": 20,
                            "visible_devices": ["1"],
                            "parent_gpu": 1,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {"rollout": "0"},
                "resource_pool": {
                    "enabled": True,
                    "allocation_mode": "plan_file",
                    "allocation_plan_path": str(plan_path),
                },
            },
            "env": {"train": {"total_num_envs": 1}, "eval": {"total_num_envs": 1}},
            "runner": {"only_eval": False, "val_check_interval": -1},
            "rollout": {"pipeline_stage_num": 1},
        }
    )
    cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=2)
    placement = HybridComponentPlacement(cfg, cluster)

    with pytest.raises(ValueError, match="visible devices"):
        FineGrainedResourcePool.from_config(cfg, cluster, placement)


def test_plan_file_mode_validates_mig_metadata(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "bindings": [
                    {
                        "component": "rollout",
                        "rank": 0,
                        "cluster_node_rank": 0,
                        "node_group_label": "cluster",
                        "cpu": None,
                        "gpu": {
                            "mode": "mig",
                            "sm_percent": 40,
                            "mig_device_uuid": "MIG-A",
                            "parent_gpu": 0,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {"rollout": "0"},
                "resource_pool": {
                    "enabled": True,
                    "allocation_mode": "plan_file",
                    "allocation_plan_path": str(plan_path),
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
                                    }
                                ],
                            }
                        },
                        "components": {
                            "rollout": {"pool": "mig_pool", "sm_percent": 40}
                        },
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

    with pytest.raises(ValueError, match="MIG|sm_percent"):
        FineGrainedResourcePool.from_config(cfg, cluster, placement)


def test_plan_file_mode_requires_mig_parent_gpu(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "bindings": [
                    {
                        "component": "rollout",
                        "rank": 0,
                        "cluster_node_rank": 0,
                        "node_group_label": "cluster",
                        "cpu": None,
                        "gpu": {
                            "mode": "mig",
                            "sm_percent": 20,
                            "mig_device_uuid": "MIG-A",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {"rollout": "0"},
                "resource_pool": {
                    "enabled": True,
                    "allocation_mode": "plan_file",
                    "allocation_plan_path": str(plan_path),
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
                                    }
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
    cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=1)
    placement = HybridComponentPlacement(cfg, cluster)

    with pytest.raises(ValueError, match="parent GPU"):
        FineGrainedResourcePool.from_config(cfg, cluster, placement)


def test_plan_file_mode_validates_mig_parent_gpu_against_placement(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "bindings": [
                    {
                        "component": "rollout",
                        "rank": 0,
                        "cluster_node_rank": 0,
                        "node_group_label": "cluster",
                        "cpu": None,
                        "gpu": {
                            "mode": "mig",
                            "sm_percent": 20,
                            "mig_device_uuid": "MIG-B",
                            "parent_gpu": 1,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {"rollout": "0"},
                "resource_pool": {
                    "enabled": True,
                    "allocation_mode": "plan_file",
                    "allocation_plan_path": str(plan_path),
                    "gpu": {
                        "enabled": True,
                        "mode": "mig",
                        "pools": {
                            "mig_pool": {
                                "node_group": "cluster",
                                "mig_devices": [
                                    {
                                        "uuid": "MIG-B",
                                        "parent_gpu": 1,
                                        "sm_percent": 20,
                                    }
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

    with pytest.raises(ValueError, match="placement"):
        FineGrainedResourcePool.from_config(cfg, cluster, placement)


def test_write_plan_sorts_bindings_within_component(tmp_path: Path) -> None:
    pool = FineGrainedResourcePool(
        enabled=True,
        bindings={
            "env": [
                WorkerResourceBinding(
                    component="env",
                    rank=1,
                    cluster_node_rank=0,
                    node_group_label="node",
                ),
                WorkerResourceBinding(
                    component="env",
                    rank=0,
                    cluster_node_rank=0,
                    node_group_label="node",
                ),
            ]
        },
        summary={},
    )
    plan_path = tmp_path / "plan.json"

    pool.write_plan(plan_path)

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert [binding["rank"] for binding in payload["bindings"]] == [0, 1]
