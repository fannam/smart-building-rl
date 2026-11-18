"""Summarize a Phase 1 dataset: size, coverage, exploration impact, split.

    .venv/bin/python scripts/summarize_dataset.py --data data/phase1_sample \
        --out results/phase1/sample_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import assert_no_building_leakage, load_episode, load_manifest, read_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rows = load_manifest(a.data)
    split = read_split(a.data)
    assert_no_building_leakage(split)
    by = defaultdict(dict)
    ranges = defaultdict(lambda: [np.inf, -np.inf])
    for r in rows:
        by[r["scenario"]][r["profile"]] = r
        if r["profile"] != "baseline":
            ep = load_episode(a.data / r["path"], keys=["knobs", "tank_state", "zone_info", "action_discrete",
                                                        "baseline_discrete"])
            kn = {n: ep["knobs"][:, i] for i, n in enumerate(ep["names_knobs"])}
            K = r["num_ahus"]
            stats = {
                "forced_on": float(np.mean([np.mean(kn[f"ahu_force_{k}"] == 1) for k in range(K)])),
                "forced_off": float(np.mean([np.mean(kn[f"ahu_force_{k}"] == 0) for k in range(K)])),
                "guard_ahu": float(np.mean([np.mean(kn[f"guard_ahu_{k}"]) for k in range(K)])),
                "tank_T_min": float(ep["tank_state"][:, 0].min()), "tank_T_max": float(ep["tank_state"][:, 0].max()),
            }
            r["knob_stats"] = stats
            for k, v in (("tank_T", ep["tank_state"][:, 0]),
                         ("zone_T", ep["zone_info"][..., ep["names_zone_info"].index("zone_air_temperature")])):
                ranges[k] = [min(ranges[k][0], float(v.min())), max(ranges[k][1], float(v.max()))]
    table = []
    for scn, d in sorted(by.items()):
        b, m = d.get("baseline"), d.get("moderate")
        row = {"scenario": scn, "K": (b or m)["num_ahus"], "V": (b or m)["num_zones"], "steps": (b or m)["steps"]}
        for name, r in (("baseline", b), ("moderate", m)):
            if r:
                row[f"{name}_score"] = -r["mean_reward"]
                row[f"{name}_comfort"] = r["mean_comfort"]
                row[f"{name}_MB"] = r["bytes"] / 2**20
        if m:
            row["moderate_perturbed_frac"] = m["perturbed_step_fraction"]
            row.update({f"moderate_{k}": v for k, v in m.get("knob_stats", {}).items()})
        if b and m:
            row["score_delta_moderate_minus_baseline"] = row["moderate_score"] - row["baseline_score"]
        table.append(row)
    total_steps = sum(r["steps"] for r in rows)
    summary = {
        "episodes": len(rows), "transitions": total_steps,
        "bytes_total_MB": sum(r["bytes"] for r in rows) / 2**20,
        "MB_per_1k_transitions_by_building": {
            bld: float(np.mean([r["bytes"] / 2**20 / r["steps"] * 1000 for r in rows if r["building"] == bld]))
            for bld in sorted({r["building"] for r in rows})},
        "profiles": sorted({r["profile"] for r in rows}),
        "split": split,
        "split_transitions": {k: sum(r["steps"] for r in rows if r["building"] in set(v)) for k, v in split.items()},
        "mean_score": {p: float(np.mean([-r["mean_reward"] for r in rows if r["profile"] == p]))
                       for p in sorted({r["profile"] for r in rows})},
        "mean_comfort": {p: float(np.mean([r["mean_comfort"] for r in rows if r["profile"] == p]))
                         for p in sorted({r["profile"] for r in rows})},
        "collection_cost": {"policy_ms_mean": float(np.mean([r["policy_ms_mean"] for r in rows])),
                            "sim_ms_mean": float(np.mean([r["sim_ms_mean"] for r in rows])),
                            "wall_s_total": float(sum(r["wall_s"] for r in rows))},
        "state_ranges": dict(ranges),
        "per_scenario": table,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "per_scenario"}, indent=1))
    for row in table:
        print(f"{row['scenario']:30s} K={row['K']} V={row['V']:3d} base={row.get('baseline_score', np.nan):.5f} "
              f"mod={row.get('moderate_score', np.nan):.5f} pert={row.get('moderate_perturbed_frac', np.nan):.2f} "
              f"comfort b/m={row.get('baseline_comfort', np.nan):.2e}/{row.get('moderate_comfort', np.nan):.2e}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
