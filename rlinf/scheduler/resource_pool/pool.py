from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .bindings import WorkerResourceBinding
from .config import ResourcePoolConfig
from .solver import ResourcePoolSolver


@dataclass
class FineGrainedResourcePool:
    """Resolved resource bindings and summary for fine-grained pools."""

    enabled: bool = False
    bindings: dict[str, list[WorkerResourceBinding]] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)

    @classmethod
    def disabled(cls) -> "FineGrainedResourcePool":
        """Create a disabled resource pool."""
        return cls(enabled=False)

    @classmethod
    def from_config(
        cls, cfg, cluster, component_placement
    ) -> "FineGrainedResourcePool":
        """Resolve a resource pool from scheduler config and placement."""
        pool_cfg = ResourcePoolConfig.from_cluster_cfg(cfg.cluster)
        if not pool_cfg.enabled:
            return cls.disabled()
        solver = ResourcePoolSolver(pool_cfg, cfg, cluster, component_placement)
        bindings = solver.solve()
        return cls(enabled=True, bindings=bindings, summary=solver.summary)

    def get_component_bindings(
        self, component: str
    ) -> list[WorkerResourceBinding] | None:
        """Return bindings for one component, or None when the pool is disabled."""
        if not self.enabled:
            return None
        return self.bindings.get(component)

    def write_plan(self, path: str | Path) -> None:
        """Write resolved bindings and summary as a JSON allocation plan."""
        payload = {
            "enabled": self.enabled,
            "bindings": [
                json.loads(binding.to_json())
                for component in sorted(self.bindings)
                for binding in self.bindings[component]
            ],
            "summary": self.summary,
        }
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
