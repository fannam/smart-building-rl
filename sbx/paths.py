"""Project paths. The starter kit location can be overridden with SBX_KIT."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = Path(os.environ.get("SBX_KIT", ROOT / "starter-kit")).resolve()
TRAINING_SCENARIOS = KIT / "smart_buildings" / "training_scenarios"
BASELINE_RETURNS = KIT / "baseline_returns.json"
RESULTS = ROOT / "results"

FULL_STEPS = 8640  # 30 days at 5 min; the local scorer and baseline_returns.json use this horizon


def scenario_key(scenario_dir: Path) -> str:
    """'building_x/weather_y' for a scenario directory."""
    return f"{scenario_dir.parent.name}/{scenario_dir.name}"


def discover(root: Path = TRAINING_SCENARIOS, patterns: list[str] | None = None) -> list[Path]:
    """Scenario directories (containing config.gin) under root, optionally filtered by glob patterns
    on the 'building/weather' key (e.g. 'building_train_1/*', '*/fairbanks_*')."""
    from fnmatch import fnmatch

    dirs = sorted(p.parent for p in Path(root).glob("*/*/config.gin"))
    if patterns:
        dirs = [d for d in dirs if any(fnmatch(scenario_key(d), pat) for pat in patterns)]
    return dirs
