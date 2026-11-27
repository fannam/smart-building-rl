"""Policy registry. A policy spec is either a starter-kit POLICY_MAP key ('pid_water', 'bang_bang', ...)
or 'package.module:ClassName' for our own controllers. Every policy is built as Class(meta, action_spec).

'package.module:ClassName@params.json' builds Class(meta, action_spec, params=<json dict>) (e.g. TunablePolicy)."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Callable


def resolve(spec: str) -> Callable[[dict, Any], Any]:
    if "@" in spec:
        base, params_file = spec.split("@", 1)
        module, name = base.split(":", 1)
        cls = getattr(importlib.import_module(module), name)
        params = json.loads(Path(params_file).read_text())
        params = params.get("params", params)  # accept {"params": {...}, ...} records as well as bare dicts
        return lambda meta, action_spec: cls(meta, action_spec, params=params)
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
    if "@" in spec:
        base, params_file = spec.split("@", 1)
        return f"{base.split(':')[-1]}@{Path(params_file).stem}"
    return spec.split(":")[-1] if ":" in spec else spec
