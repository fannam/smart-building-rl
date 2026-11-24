"""Policy registry. A policy spec is either a starter-kit POLICY_MAP key ('pid_water', 'bang_bang', ...)
or 'package.module:ClassName' for our own controllers. Every policy is built as Class(meta, action_spec)."""
from __future__ import annotations

import importlib
from typing import Any, Callable


def resolve(spec: str) -> Callable[[dict, Any], Any]:
    if ":" in spec:
        module, name = spec.split(":", 1)
        cls = getattr(importlib.import_module(module), name)
    else:
        from smart_buildings import evaluate as ev

        if spec not in ev.POLICY_MAP:
            raise KeyError(f"unknown policy {spec!r}; kit policies: {sorted(ev.POLICY_MAP)}")
        cls = ev.POLICY_MAP[spec]

    def make(meta: dict, action_spec: Any):
        from smart_buildings import evaluate as ev

        return ev.initialize_policy(cls, meta, action_spec)

    return make


def label(spec: str) -> str:
    """Filesystem-safe name for a policy spec."""
    return spec.split(":")[-1] if ":" in spec else spec
