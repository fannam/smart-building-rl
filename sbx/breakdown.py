"""Loss breakdown of one policy in an evaluation run.

Every number is an additive share of the leaderboard score: for a scenario with n steps, the cell's share is
-sum(term over the cell's steps) / n, then averaged over scenarios with equal weight. Shares therefore add up
to the score. Terms are grouped into comfort, electricity (cost + carbon) and gas (cost + carbon).

    python -m sbx.breakdown m1_reference WideHotStorageReservePolicy
    python -m sbx.breakdown m1_reference WideHotStorageReservePolicy --json results/m1_reference/breakdown.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sbx import paths
from sbx.score import station

GROUPS = {"comfort": ("comfort",), "elec": ("elec_cost", "elec_carbon"), "gas": ("gas_cost", "gas_carbon")}


def load(run_dir: Path, policy: str) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for f in sorted((run_dir / policy).glob("*.npz")):
        d = np.load(f)
        if "occupancy" not in d.files:
            raise SystemExit(f"{f} has no per-step context; re-run it with the current evaluator")
        g = {k: -sum(d[t].astype(np.float64) for t in ts) for k, ts in GROUPS.items()}
        out[f.stem.replace("__", "/")] = {**g, "hour": d["local_hour"], "weekday": d["weekday"],
                                         "occupied": d["occupancy"] > 0}
    return out


def shares(data: dict, cell) -> dict[str, dict[str, float]]:
    """{cell_label: {group: share}} with `cell(scenario, arrays) -> {label: boolean mask}`."""
    acc: dict[str, dict[str, float]] = defaultdict(lambda: dict.fromkeys(GROUPS, 0.0))
    for s, a in data.items():
        n = len(a["hour"])
        for label, mask in cell(s, a).items():
            for g in GROUPS:
                acc[label][g] += float(a[g][mask].sum()) / n / len(data)
    return dict(acc)


def table(title: str, rows: dict[str, dict[str, float]], total: float) -> None:
    print(f"\n{title}")
    print(f"   {'':22s} {'comfort':>9s} {'elec':>9s} {'gas':>9s} {'total':>9s} {'share':>6s}")
    for k, v in rows.items():
        t = sum(v.values())
        print(f"   {k:22s} {v['comfort']:9.5f} {v['elec']:9.5f} {v['gas']:9.5f} {t:9.5f} {100 * t / total:5.1f}%")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("policy", help="policy label (directory name in the run)")
    ap.add_argument("--json", type=Path, default=None, help="also write all tables here")
    a = ap.parse_args(argv)
    run_dir = Path(a.run) if Path(a.run).exists() else paths.RESULTS / a.run
    data = load(run_dir, a.policy)
    if not data:
        raise SystemExit(f"no .npz results for {a.policy} in {run_dir}")

    def weekend(a):
        return a["weekday"] >= 5

    views = {
        "overall": shares(data, lambda s, a: {"all": np.ones_like(a["occupied"])}),
        "by station": dict(sorted(shares(data, lambda s, a: {station(s): np.ones_like(a["occupied"])}).items())),
        "by building": dict(sorted(shares(data, lambda s, a: {s.split("/")[0]: np.ones_like(a["occupied"])}).items())),
        "occupied vs empty": shares(data, lambda s, a: {"occupied": a["occupied"], "empty": ~a["occupied"]}),
        "weekday vs weekend": shares(data, lambda s, a: {"weekday": ~weekend(a), "weekend": weekend(a)}),
        "weekday by occupancy": shares(data, lambda s, a: {
            "weekday occupied": ~weekend(a) & a["occupied"], "weekday empty": ~weekend(a) & ~a["occupied"],
            "weekend occupied": weekend(a) & a["occupied"], "weekend empty": weekend(a) & ~a["occupied"]}),
        "by local hour": shares(data, lambda s, a: {f"{h:02d}:00": a["hour"].astype(int) == h for h in range(24)}),
    }
    total = sum(views["overall"]["all"].values())
    print(f"{a.policy} in {run_dir.name}: score {total:.5f} over {len(data)} scenarios")
    for title, rows in views.items():
        table(title, rows, total)

    # Per-scenario contribution to the mean score, worst first.
    per = sorted(((s, {g: float(v[g].mean()) / len(data) for g in GROUPS}) for s, v in data.items()),
                 key=lambda kv: -sum(kv[1].values()))
    table("top 10 scenarios by contribution", dict(per[:10]), total)

    # Concentration of comfort loss in the worst steps (runaway zones, quadratic penalty).
    print("\ncomfort concentration (share of each scenario's comfort loss in its worst 1% of steps), worst 5:")
    conc = []
    for s, v in data.items():
        c = np.sort(v["comfort"])[::-1]
        if c.sum() > 0:
            conc.append((s, float(c[: max(1, len(c) // 100)].sum() / c.sum()), float(c.mean())))
    for s, f, m in sorted(conc, key=lambda x: -x[1] * x[2])[:5]:
        print(f"   {s:32s} {100 * f:5.1f}% of comfort loss {m:.5f}")

    if a.json:
        a.json.write_text(json.dumps({"policy": a.policy, "score": total, **views,
                                      "per_scenario": dict(per)}, indent=2))
        print(f"-> {a.json}")


if __name__ == "__main__":
    main()
