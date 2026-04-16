from __future__ import annotations

from dataclasses import dataclass

from toolkits.rollout_eval.profiling.torch_profiler import aggregate_profile_metrics


@dataclass
class _Evt:
    key: str
    cpu_time_total: float = 0.0
    cuda_time_total: float = 0.0
    self_cpu_time_total: float = 0.0
    self_cuda_time_total: float = 0.0



def test_aggregate_profile_metrics_sums_env_model_and_split_stages() -> None:
    events = [
        _Evt("env.simulation", cpu_time_total=10.0, cuda_time_total=1.0),
        _Evt("model.inference", cpu_time_total=20.0, cuda_time_total=2.0),
        _Evt("model.backbone.openvla_oft.backbone", cpu_time_total=30.0, cuda_time_total=3.0),
        _Evt("model.action_head.openvla_oft.action_head", cpu_time_total=40.0, cuda_time_total=4.0),
    ]

    got = aggregate_profile_metrics(events)

    assert got["env_sim_cpu_us"] == 10.0
    assert got["model_infer_cpu_us"] == 20.0
    assert got["model_backbone_cpu_us"] == 30.0
    assert got["model_action_head_cpu_us"] == 40.0
    assert got["env_sim_cuda_us"] == 1.0
    assert got["model_infer_cuda_us"] == 2.0
    assert got["model_backbone_cuda_us"] == 3.0
    assert got["model_action_head_cuda_us"] == 4.0



def test_aggregate_profile_metrics_falls_back_to_self_times() -> None:
    events = [
        _Evt("env.simulation", self_cpu_time_total=7.0, self_cuda_time_total=0.5),
        _Evt("model.inference", self_cpu_time_total=11.0, self_cuda_time_total=1.5),
    ]

    got = aggregate_profile_metrics(events)

    assert got["env_sim_cpu_us"] == 7.0
    assert got["env_sim_cuda_us"] == 0.5
    assert got["model_infer_cpu_us"] == 11.0
    assert got["model_infer_cuda_us"] == 1.5
