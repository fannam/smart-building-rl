"""Summarize a run_eval.py results directory into leaderboard-style scores.

    .venv/bin/python src/summarize.py results/baseline_full
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
rows = defaultdict(dict)
for f in root.glob("*/*.json"):
    r = json.loads(f.read_text())
    if "error" in r:
        print("ERROR", f)
        continue
    rows[r["scenario"]][r["policy"]] = r

policies = sorted({p for d in rows.values() for p in d})
print(f"{'scenario':30s} " + " ".join(f"{p:>12s}" for p in policies) + "   comfort/energy/carbon (first policy)  K/V   wall ratio")
for s in sorted(rows):
    d = rows[s]
    first = d.get(policies[0])
    parts = (f"{-first['mean_comfort']:.4f}/{-first['mean_energy']:.4f}/{-first['mean_carbon']:.4f}"
             if first else "")
    ratio = ""
    if first and "bang_bang" in d and policies[0] != "bang_bang":
        ratio = f"{first['wall_seconds'] / d['bang_bang']['wall_seconds']:.2f}x"
    kv = f"{first['num_ahus']}/{first['num_zones']}" if first else ""
    print(f"{s:30s} " + " ".join(f"{-d[p]['mean_reward']:12.5f}" if p in d else f"{'-':>12s}" for p in policies)
          + f"   {parts:36s} {kv:6s} {ratio}")

print()
for p in policies:
    rs = [rows[s][p] for s in rows if p in rows[s]]
    if not rs:
        continue
    full = all(r["steps"] == 8640 for r in rs)
    score = sum(-r["mean_reward"] for r in rs) / len(rs)
    comfort = sum(-r["mean_comfort"] for r in rs) / len(rs)
    energy = sum(-r["mean_energy"] for r in rs) / len(rs)
    carbon = sum(-r["mean_carbon"] for r in rs) / len(rs)
    worst = max(rs, key=lambda r: -r["mean_reward"])
    print(f"{p:12s} score={score:.5f}  (comfort {comfort:.5f}, energy {energy:.5f}, carbon {carbon:.5f})  "
          f"n={len(rs)}{'' if full else ' NOT FULL HORIZON'}  worst={worst['scenario']} {-worst['mean_reward']:.5f}  "
          f"max wall={max(r['wall_seconds'] for r in rs):.0f}s")
