"""Phase 2C post-collection QA + branch manifest.

Checks (exit code 1 if any HARD check fails):
  HARD  every branch file loads, is complete, has the configured arm count and schema
  HARD  no duplicate branch ids; every file belongs to the plan (plan.json)
  HARD  nominal replay bit-exact at every branch point
  HARD  no NaN/inf in non-aborted arms
  SOFT  planned vs collected per scenario (incomplete is reported, not an error: resume later)
Coverage report per action family: effective change, ACTIVE change (owning AHU running), no-ops,
saturated, clipped, no-qualifying, aborted, copied (not re-simulated), meaningful simulated response;
zone-level events for E1/E2 subset arms; strata / weather / E2-eligibility per building.

    python scripts/phase2c_collection_qa.py --out <collection dir>
writes <out>/qa_report.json and <out>/manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ZONE_THR, TANK_THR, GAS_THR = 0.02, 0.05, 500.0     # same "meaningful response" thresholds as Phase 2B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    plan = json.loads((out / "plan.json").read_text())
    A = int(plan["arms_per_branch"])
    planned = {(sc, p["t_b"]): p["stratum"] for sc, pts in plan["plan"].items() for p in pts}
    hard, manifest = [], []
    fam = defaultdict(Counter)
    zone_events = Counter()
    per_b = defaultdict(Counter)
    seen = set()
    files = sorted((out / "branches").glob("*.npz"))
    for f in files:
        try:
            z = np.load(f, allow_pickle=False)
            ok = int(z["complete"]) == 1 and z["arm_ac"].shape[1] == A
        except Exception as ex:  # noqa: BLE001
            hard.append(f"{f.name}: unreadable ({type(ex).__name__})"); continue
        if not ok:
            hard.append(f"{f.name}: incomplete or wrong arm count"); continue
        sc, tb, bid = str(z["scenario"]), int(z["t_b"][0]), str(z["branch_id"][0])
        if bid in seen:
            hard.append(f"duplicate branch id {bid}")
        seen.add(bid)
        if (sc, tb) not in planned:
            hard.append(f"{bid} not in plan")
        if not bool(z["nominal_bit_exact"][0]):
            hard.append(f"{bid}: nominal replay not bit-exact")
        ab = z["aborted"][0]
        for k in ("zone_temp", "tank", "energy"):
            x = z[k][0][~ab]
            if not np.isfinite(x).all():
                hard.append(f"{bid}: non-finite {k} in non-aborted arms")
        zt, tk, en = z["zone_temp"][0], z["tank"][0], z["energy"][0]
        b = sc.split("/")[0]
        per_b[b]["branch_points"] += 1
        per_b[b][f"stratum:{z['stratum'][0]}"] += 1
        per_b[b][f"oat:{z['oat_band'][0]}"] += 1
        per_b[b]["e2_eligible"] += int(z["e2_eligible"][0])
        per_b[b]["ahu_on_at_tb"] += int(z["ahu_on"][0])
        for i in range(1, A):
            key = f"{z['arm_group'][i]}:{z['arm_family'][i]}:{z['arm_kind'][i]}"
            c = fam[key]
            c["arms"] += 1
            c["effective_change"] += int(not z["effective_noop"][0, i])
            c["active_change"] += int(z["active_change"][0, i])
            for flag in ("effective_noop", "saturated", "clipped", "no_qualifying", "aborted", "extreme_excursion"):
                c[flag] += int(z[flag][0, i])
            c["copied_not_simulated"] += int(not z["simulated"][0, i])
            if not ab[i]:
                dz = np.abs(zt[i] - zt[0]).max(0)
                c["zone_response"] += int((dz > ZONE_THR).any())
                c["tank_response"] += int(np.abs(tk[i] - tk[0]).max() > TANK_THR)
                c["gas_response"] += int(np.abs(en[i, :, 3] - en[0, :, 3]).max() > GAS_THR)
            if z["arm_family"][i] in ("reheat_valve", "damper") and str(z["arm_kind"][i]).startswith("subset") and z["arm_group"][i] != "NOOP":
                zc = z["zone_eff_changed"][0, i, :, 0 if z["arm_family"][i] == "damper" else 1]
                zone_events[f"{z['arm_family'][i]}:zones_effectively_changed"] += int(zc.sum())
                if not ab[i]:
                    zone_events[f"{z['arm_family'][i]}:zones_with_response"] += int(((np.abs(zt[i] - zt[0]).max(0) > ZONE_THR) & zc).sum())
        manifest.append({"branch_id": bid, "scenario": sc, "building": b, "weather": sc.split("/")[1], "t_b": tb,
                         "stratum": str(z["stratum"][0]), "oat_band": str(z["oat_band"][0]), "oat_c": float(z["oat_c"][0]),
                         "e2_eligible": bool(z["e2_eligible"][0]), "ahu_on": bool(z["ahu_on"][0]), "heating": bool(z["heating"][0]),
                         "n_arms": A, "aborted_arms": int(ab.sum()), "file": str(f.relative_to(out))})
    per_sc = defaultdict(lambda: [0, 0])
    for sc, tb in planned:
        per_sc[sc][0] += 1
    for m in manifest:
        per_sc[m["scenario"]][1] += 1
    incomplete = {sc: v for sc, v in per_sc.items() if v[1] < v[0]}
    report = {"collection_dir": str(out), "planned_branch_points": len(planned), "complete_branch_points": len(manifest),
              "arms_per_branch": A, "all_planned_complete": not incomplete, "incomplete_scenarios": incomplete,
              "hard_failures": hard, "families": {k: dict(v) for k, v in sorted(fam.items())},
              "zone_subset_events": dict(zone_events), "per_building": {k: dict(v) for k, v in sorted(per_b.items())}}
    (out / "qa_report.json").write_text(json.dumps(report, indent=1))
    with (out / "manifest.jsonl").open("w") as fh:
        for m in sorted(manifest, key=lambda x: x["branch_id"]):
            fh.write(json.dumps(m) + "\n")
    print(f"branch points: {len(manifest)}/{len(planned)} complete; arms/branch {A}; hard failures: {len(hard)}")
    if incomplete:
        print(f"INCOMPLETE scenarios ({len(incomplete)}): resume in another session -> " +
              ", ".join(f"{s} {v[1]}/{v[0]}" for s, v in sorted(incomplete.items())[:8]) + (" ..." if len(incomplete) > 8 else ""))
    print(f"{'family':48s} arms  effchg active noop  sat clip noQ abort copied  zoneResp tankResp gasResp")
    for k, v in sorted(fam.items()):
        print(f"{k:48s} {v['arms']:4d} {v['effective_change']:6d} {v['active_change']:6d} {v['effective_noop']:4d} "
              f"{v['saturated']:4d} {v['clipped']:4d} {v['no_qualifying']:3d} {v['aborted']:5d} {v['copied_not_simulated']:6d} "
              f"{v['zone_response']:9d} {v['tank_response']:8d} {v['gas_response']:7d}")
    print("zone-subset events:", dict(zone_events))
    for h in hard[:20]:
        print("HARD FAIL:", h)
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
