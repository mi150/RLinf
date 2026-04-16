"""Adapter interfaces and factories for rollout eval."""

from toolkits.rollout_eval.adapters.env_adapter import (
    EnvAdapterProtocol,
    build_env_adapter,
)
from toolkits.rollout_eval.adapters.model_adapter import (
    ModelAdapterProtocol,
    build_model_adapter,
    build_null_model_adapter,
)

__all__ = [
    "EnvAdapterProtocol",
    "ModelAdapterProtocol",
    "build_env_adapter",
    "build_model_adapter",
    "build_null_model_adapter",
]
