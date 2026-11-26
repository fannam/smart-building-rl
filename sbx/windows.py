"""Do short windows rank controllers like full episodes? (PLAN.md, M2 step 3)

The score of an N-day window is the mean reward over the first N days of a full rollout, which is exactly what
an N-day rollout would give (the simulator is deterministic). So one full run per (policy, scenario) answers every
window length. Rankings are compared with Spearman's rho over policies, both unweighted and with the cold-weighted
objective, for all policies and for a "close" subset (controllers of similar quality, the case that matters).

    python -m sbx.windows --runs baselines m1_reference m2_variants \
        --close TransferPolicy CausalIdentifiedReservePolicy ... WideHotStorageReservePolicy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from sbx import paths
from sbx.scenarios import scenario_info, weights

STEPS_PER_DAY = 288


def load_rewards(runs: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """{policy: {scenario: per-step reward}} over all given runs (full 8640-step episodes only)."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for run in runs:
        for f in sorted((paths.RESULTS / run).glob("*/*.npz")):
            r = np.load(f)["reward"].astype(np.float64)
            if len(r) == paths.FULL_STEPS:
                out.setdefault(f.parent.name, {})[f.stem.replace("__", "/")] = r
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--close", nargs="*", default=[], help="policy labels forming the close-quality subset")
    ap.add_argument("--exclude", nargs="*", default=[], help="policy labels to leave out (e.g. incomplete runs)")
    ap.add_argument("--days", nargs="+", type=int, default=[3, 7, 14])
    ap.add_argument("--deep-cold-share", type=float, default=0.25)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)

    data = {p: v for p, v in load_rewards(a.runs).items() if p not in a.exclude}
    common = sorted(set.intersection(*(set(v) for v in data.values())))
    pols = sorted(data)
    print(f"{len(pols)} policies x {len(common)} common scenarios (full episodes)")
    w = weights(common, scenario_info(), a.deep_cold_share)

    def score(pol: str, days: int | None, weighted: bool) -> float:
        n = paths.FULL_STEPS if days is None else days * STEPS_PER_DAY
        s = np.array([-data[pol][k][:n].mean() for k in common])
        return float(np.dot(w, s) if weighted else s.mean())

    report: dict = {"policies": pols, "scenarios": common, "deep_cold_share": a.deep_cold_share, "rows": []}
    for weighted in (False, True):
        tag = f"cold-weighted ({a.deep_cold_share:.0%} deep cold)" if weighted else "unweighted"
        full = {p: score(p, None, weighted) for p in pols}
        print(f"\n== {tag}")
        print("   " + f"{'policy':32s} {'30 d':>8s}" + "".join(f"{f'{d} d':>9s}" for d in a.days))
        for p in sorted(pols, key=full.get):
            print(f"   {p:32s} {full[p]:8.5f}" + "".join(f"{score(p, d, weighted):9.5f}" for d in a.days))
        for subset_name, subset in (("all", pols), ("close", [p for p in pols if p in a.close])):
            if len(subset) < 3:
                continue
            f = [full[p] for p in subset]
            for d in a.days:
                s = [score(p, d, weighted) for p in subset]
                rho, tau = spearmanr(f, s)[0], kendalltau(f, s)[0]
                print(f"   {subset_name:5s} n={len(subset):2d}  {d:2d} d vs 30 d: spearman {rho:+.3f}  kendall {tau:+.3f}")
                report["rows"].append({"weighted": weighted, "subset": subset_name, "n": len(subset), "days": d,
                                       "spearman": rho, "kendall": tau})
        report["full_weighted" if weighted else "full"] = full

    if a.json:
        a.json.write_text(json.dumps(report, indent=2))
        print(f"-> {a.json}")


if __name__ == "__main__":
    main()
