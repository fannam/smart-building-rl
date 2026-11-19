"""Phase 2C split manifest (branch-point level; sibling arms are never separated).

Unit of assignment: (scenario, day block = t_b // 288). Every branch point and all its arms inherit the
split of its scenario-day, so context/horizon segments of one source trajectory never straddle splits.
    Test-A  Phase 2B counterfactual set (configs/phase2c/phase2b_branch_points.json), untouched
    Test-B  every Phase 2C branch point of the held-out weather (default: denver_*) of each building
    Val     ~20 % of the remaining scenario-days (seeded hash)   -> lambda/checkpoint selection only
    Train   the rest
LOBO folds (configs/leave_one_building_out.json) are applied on top by the trainer: a fold trains on the
Train/Val points of its 5 training buildings only.

    python scripts/phase2c_splits.py --data <collection dir> --out configs/phase2c/split_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DAY = 288


def assign(rows, heldout_weather_prefix="denver", val_fraction=0.2, seed=0):
    out = {}
    for r in rows:
        if r["weather"].startswith(heldout_weather_prefix):
            s = "test_b"
        else:
            day = r["t_b"] // DAY
            u = zlib.crc32(f"{seed}|{r['scenario']}|{day}".encode()) / 2 ** 32
            s = "val" if u < val_fraction else "train"
        out[r["branch_id"]] = {"split": s, "scenario_day": f"{r['scenario']}#{r['t_b'] // DAY}", **{k: r[k] for k in ("scenario", "building", "weather", "t_b", "stratum")}}
    return out


def check(assignments, test_a, radius):
    """Leakage assertions; returns a list of problems."""
    errs = []
    sd = {}
    for bid, a in assignments.items():
        prev = sd.setdefault(a["scenario_day"], a["split"])
        if prev != a["split"]:
            errs.append(f"scenario-day {a['scenario_day']} in {prev} and {a['split']}")
    for bid, a in assignments.items():
        for u in test_a.get(a["scenario"], []):
            if abs(a["t_b"] - u) <= radius:
                errs.append(f"{bid} within {radius} steps of Test-A point {a['scenario']}@{u}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="collection dir with manifest.jsonl (from phase2c_collection_qa.py)")
    ap.add_argument("--out", default=str(ROOT / "configs/phase2c/split_manifest.json"))
    ap.add_argument("--test-a", default=str(ROOT / "configs/phase2c/phase2b_branch_points.json"))
    ap.add_argument("--heldout-weather", default="denver")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radius", type=int, default=48)
    a = ap.parse_args()
    rows = [json.loads(l) for l in (Path(a.data) / "manifest.jsonl").read_text().splitlines() if l.strip()]
    test_a = json.loads(Path(a.test_a).read_text())["branch_points"]
    asg = assign(rows, a.heldout_weather, a.val_fraction, a.seed)
    errs = check(asg, test_a, a.radius)
    if errs:
        raise SystemExit("SPLIT LEAKAGE:\n  " + "\n  ".join(errs[:20]))
    counts = Counter((v["building"], v["split"]) for v in asg.values())
    doc = {"version": "phase2c-splits-v1", "unit": "scenario-day (t_b // 288); all arms of a branch point share its split",
           "heldout_weather_prefix": a.heldout_weather, "val_fraction": a.val_fraction, "seed": a.seed,
           "test_a": {"source": a.test_a, "branch_points": test_a},
           "assignments": dict(sorted(asg.items())),
           "counts": {f"{b}|{s}": n for (b, s), n in sorted(counts.items())},
           "digest": hashlib.sha256(json.dumps(sorted(asg.items())).encode()).hexdigest()[:16]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1))
    tot = Counter(v["split"] for v in asg.values())
    print(f"splits: {dict(tot)} (+ Test-A {sum(len(v) for v in test_a.values())} Phase 2B points); digest {doc['digest']} -> {a.out}")


if __name__ == "__main__":
    main()
