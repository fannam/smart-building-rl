"""Validation suite: generated buildings x new weather strips, split into tune / val / test.

    python -m sbx.scenarios manifest     # buildings + weather YAMLs + manifests/suite.yaml (committed)
    python -m sbx.scenarios generate     # scenario dirs under data/scenarios/ from the manifest (gitignored)
    python -m sbx.scenarios summary

Design (see PLAN.md, M2):
- Buildings come from the kit's generator (geometry from the seed, materials stratified across the batch, the
  kit's sliver-zone check). Two size profiles: "shipped" fits the ranges of the six shipped buildings, which are
  smaller than the generator's current defaults; "kit_default" is those defaults (larger, 2-3x slower to simulate)
  and is used only in the test_large split, as a robustness check.
- Weather: the hidden set uses strips that are not among the 48 shipped. Shipped strips are 30-day windows of
  the 2023 station CSVs starting Sundays at 06:00, and together they use each Sunday from Jan 1 to Nov 26 exactly
  once. Per station, the 40 unused Sundays become new strips, split by date (3 of every 5 to tune, 1 to val,
  1 to test), so every split spans the year and no strip is shared between splits.
- Each building gets one strip per station, assigned by a seeded permutation, plus one extra deep-cold strip
  (7 scenarios per building).
- Every scenario is tagged with the strip's mean outdoor temperature and a climate class for weighting.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sbx import paths

MANIFESTS = paths.ROOT / "manifests"
SUITE = MANIFESTS / "suite.yaml"
SCENARIOS = paths.ROOT / "data" / "scenarios"
STATIONS = ("atlanta", "denver", "fairbanks", "minneapolis", "phoenix", "singapore")
EPISODE_DAYS = 30

PROFILES = {
    # Fitted to the six shipped buildings: width 318-430, height 188-301, floors 2-4, min room 25-35,
    # room span 11-16, shape ratio 0.29-0.38, coverage 0.65-0.80, occupants 1-2.
    "shipped": dict(width=(310, 440), height=(185, 305), num_floors=(2, 4), min_room_size=(25, 36),
                    room_size_span=(11, 16), shape_ratio=(0.28, 0.40), composite_coverage=(0.65, 0.85),
                    composite_style="architectural", num_ahus=(3, 7), occupants_per_zone=(1, 2)),
    "kit_default": None,  # smart_buildings.run_floor_generator.DEFAULT_RANGES
}
# (profile, split, number of buildings, first seed)
BUILDING_PLAN = [("shipped", None, 40, 1000), ("kit_default", "test_large", 8, 2000)]
SHIPPED_SPLITS = ("tune", "tune", "tune", "val", "test")  # building i -> SHIPPED_SPLITS[i % 5]: 24 / 8 / 8
STRIP_SPLITS = ("tune", "tune", "tune", "val", "test")    # same pattern over each station's unused Sundays


def climate_class(mean_c: float) -> str:
    return "deep_cold" if mean_c < -5 else "cold" if mean_c < 5 else "mild" if mean_c < 20 else "hot"


DEEP_COLD_SHARE = 0.25  # estimated share of deep-cold scenarios in the hidden set (PLAN.md, M2 step 4)


def weights(keys: list[str], info: dict[str, dict], deep_cold_share: float = DEEP_COLD_SHARE) -> np.ndarray:
    """Per-scenario weights summing to 1: deep-cold scenarios get `deep_cold_share` in total, the rest share
    the remainder equally. Falls back to equal weights when either group is empty."""
    cold = np.array([info.get(k, {}).get("climate") == "deep_cold" for k in keys])
    if cold.all() or not cold.any():
        return np.full(len(keys), 1.0 / len(keys))
    return np.where(cold, deep_cold_share / cold.sum(), (1.0 - deep_cold_share) / (~cold).sum())


def _ranges(profile: str) -> dict:
    if PROFILES[profile] is not None:
        return PROFILES[profile]
    from smart_buildings.run_floor_generator import DEFAULT_RANGES

    return DEFAULT_RANGES


def make_buildings(profile: str, n: int, start_seed: int) -> list[dict]:
    """n valid building configs: seeds from start_seed up, skipping seeds with a sliver zone (kit check)."""
    from smart_buildings.smart_control.utils.config_generator import generate_config
    from smart_buildings.smart_control.utils.materials import draw_materials
    from smart_buildings.smart_control.utils.scenario_generator import (check_zone_floor_areas,
                                                                        scenario_config_from_dict)

    configs, seed = [], start_seed
    while len(configs) < n:
        c = generate_config(name=f"building_{seed:04d}", seed=seed, building_shape="composite", **_ranges(profile))
        try:
            check_zone_floor_areas(scenario_config_from_dict(c))
            configs.append(c)
        except ValueError:
            pass
        seed += 1
    # Materials stratified across the kept batch, seeded from its seed list (as the kit's CLI does).
    digest = hashlib.sha256(repr([c["seed"] for c in configs]).encode()).digest()
    for c, m in zip(configs, draw_materials(random.Random(int.from_bytes(digest[:8], "big")), len(configs))):
        c["materials"] = m
    return configs


def new_strips() -> list[dict]:
    """Unused-Sunday strips per station with their split and mean outdoor temperature."""
    wdir = paths.KIT / "smart_buildings" / "weather"
    shipped = {s: [] for s in STATIONS}
    conv = {}
    for f in wdir.glob("*.yaml"):
        w = yaml.safe_load(f.read_text())
        st = f.stem.rsplit("_", 1)[0]
        if st in shipped and w.get("weather_csv"):
            shipped[st].append(pd.Timestamp(w["start_timestamp"]))
            conv[st] = w["convection_coefficient"]
    out = []
    for st in STATIONS:
        df = pd.read_csv(wdir / "csv" / f"{st}.csv")
        t = pd.to_datetime(df["Time"], format="%Y%m%d-%H%M")
        last = t.max() - pd.Timedelta(days=EPISODE_DAYS)
        sundays = [d for d in pd.date_range("2023-01-01 06:00", last, freq="7D") if d not in shipped[st]]
        for i, d in enumerate(sundays):
            win = (t >= d) & (t < d + pd.Timedelta(days=EPISODE_DAYS))
            mean_c = float(df.loc[win, "TempC"].mean())
            out.append({"name": f"{st}_{d:%m%d}", "station": st, "start": f"{d:%Y-%m-%d %H:%M}",
                        "convection_coefficient": conv[st], "split": STRIP_SPLITS[i % len(STRIP_SPLITS)],
                        "mean_temp_c": round(mean_c, 2), "climate": climate_class(mean_c)})
    return out


def build_manifest() -> dict:
    (MANIFESTS / "buildings").mkdir(parents=True, exist_ok=True)
    (MANIFESTS / "weather").mkdir(parents=True, exist_ok=True)

    buildings = []
    for profile, split, n, seed in BUILDING_PLAN:
        for i, c in enumerate(make_buildings(profile, n, seed)):
            (MANIFESTS / "buildings" / f"{c['name']}.yaml").write_text(
                yaml.dump(c, default_flow_style=False, sort_keys=False))
            buildings.append({"name": c["name"], "profile": profile,
                              "split": split or SHIPPED_SPLITS[i % len(SHIPPED_SPLITS)]})

    strips = new_strips()
    for s in strips:
        (MANIFESTS / "weather" / f"{s['name']}.yaml").write_text(yaml.dump(
            {"name": s["name"], "weather_csv": f"csv/{s['station']}.csv",
             "convection_coefficient": s["convection_coefficient"], "start_timestamp": s["start"]},
            sort_keys=False))

    # Pair: each building gets one strip per station from its split's strips (test_large uses test strips).
    rng = np.random.default_rng(2026)
    scenarios = []
    for split in ("tune", "val", "test", "test_large"):
        bs = [b for b in buildings if b["split"] == split]
        pool_split = "test" if split == "test_large" else split
        for st in STATIONS:
            pool = [s for s in strips if s["station"] == st and s["split"] == pool_split]
            order = np.concatenate([rng.permutation(len(pool)) for _ in range(-(-len(bs) // len(pool)))])
            for b, k in zip(bs, order):
                s = pool[k]
                scenarios.append({"building": b["name"], "weather": s["name"], "split": split,
                                  "mean_temp_c": s["mean_temp_c"], "climate": s["climate"]})
        # One extra deep-cold scenario per building: the hidden set leans to deep cold (PLAN.md, M1), and one
        # strip per station gives too few of them to tune and validate on.
        cold = [s for s in strips if s["split"] == pool_split and s["climate"] == "deep_cold"]
        for b in bs:
            used = {x["weather"] for x in scenarios if x["building"] == b["name"]}
            s = cold[int(rng.choice([i for i, c in enumerate(cold) if c["name"] not in used]))]
            scenarios.append({"building": b["name"], "weather": s["name"], "split": split,
                              "mean_temp_c": s["mean_temp_c"], "climate": s["climate"], "extra_cold": True})

    manifest = {"profiles": {k: {f: list(r) if isinstance(r, tuple) else r for f, r in _ranges(k).items()}
                             for k in PROFILES},
                "kit_sha": _kit_sha(), "buildings": buildings, "strips": strips, "scenarios": scenarios}
    SUITE.write_text(yaml.dump(manifest, sort_keys=False, width=120))
    return manifest


def _kit_sha() -> str | None:
    import subprocess

    try:
        return subprocess.run(["git", "-C", str(paths.KIT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def load_manifest() -> dict:
    return yaml.safe_load(SUITE.read_text())


def scenario_dir(s: dict) -> Path:
    return SCENARIOS / s["building"] / s["weather"]


def split_dirs(split: str) -> list[Path]:
    return [scenario_dir(s) for s in load_manifest()["scenarios"] if s["split"] == split]


def scenario_info() -> dict[str, dict]:
    """'building/weather' -> manifest entry, for the suite plus the 30 shipped training scenarios."""
    info = {f"{s['building']}/{s['weather']}": s for s in load_manifest()["scenarios"]} if SUITE.exists() else {}
    for d in paths.discover():
        w = yaml.safe_load((paths.KIT / "smart_buildings" / "weather" / f"{d.name}.yaml").read_text())
        df = pd.read_csv(paths.KIT / "smart_buildings" / "weather" / w["weather_csv"])
        t = pd.to_datetime(df["Time"], format="%Y%m%d-%H%M")
        start = pd.Timestamp(w["start_timestamp"])
        mean_c = float(df.loc[(t >= start) & (t < start + pd.Timedelta(days=EPISODE_DAYS)), "TempC"].mean())
        info[paths.scenario_key(d)] = {"building": d.parent.name, "weather": d.name, "split": "shipped",
                                       "mean_temp_c": round(mean_c, 2), "climate": climate_class(mean_c)}
    return info


def _generate_one(building: str, weather: str) -> str:
    from smart_buildings.smart_control.utils import scenario_generator as sg
    from smart_buildings.smart_control.utils.scenario_generator import (generate_scenario_from_config,
                                                                        load_scenario_from_parts)

    # The installed package has no weather data; resolve CSVs in the kit, so configs reference
    # 'smart_buildings/weather/csv/<station>.csv' relative to the kit root, like the shipped scenarios.
    sg.WEATHER_DIR = str(paths.KIT / "smart_buildings" / "weather")

    out = SCENARIOS / building / weather
    if (out / "config.gin").exists():
        return f"{building}/{weather} (exists)"
    cwd = os.getcwd()
    os.chdir(paths.KIT)  # the generator resolves weather CSVs relative to the kit
    try:
        cfg = load_scenario_from_parts(str(MANIFESTS / "buildings" / f"{building}.yaml"),
                                       str(MANIFESTS / "weather" / f"{weather}.yaml"),
                                       output_base_dir=str(SCENARIOS))
        generate_scenario_from_config(cfg)
    finally:
        os.chdir(cwd)
    return f"{building}/{weather}"


def generate(splits: list[str] | None, workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor

    todo = [s for s in load_manifest()["scenarios"] if splits is None or s["split"] in splits]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, r in enumerate(pool.map(_generate_one, [s["building"] for s in todo], [s["weather"] for s in todo]), 1):
            if i % 20 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] {r}", flush=True)


def summary() -> None:
    m = load_manifest()
    print(f"{len(m['buildings'])} buildings, {len(m['strips'])} new strips, {len(m['scenarios'])} scenarios")
    for split in ("tune", "val", "test", "test_large"):
        sc = [s for s in m["scenarios"] if s["split"] == split]
        cls = {c: sum(s["climate"] == c for s in sc) for c in ("deep_cold", "cold", "mild", "hot")}
        nb = len({s["building"] for s in sc})
        print(f"  {split:10s} {nb:2d} buildings, {len(sc):3d} scenarios, climate {cls}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("manifest", "generate", "summary"))
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    if a.command == "manifest":
        build_manifest()
        summary()
    elif a.command == "generate":
        generate(a.splits, a.workers)
    else:
        summary()


if __name__ == "__main__":
    main()
