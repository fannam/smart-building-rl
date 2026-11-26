"""Leaderboard-style scoring of an evaluation run.

Score = mean over scenarios of |mean per-step reward| (lower is better), equal weight per scenario.
Also prints the split into comfort / electricity / gas (cost + carbon), per-station means, and the cold-weighted
score (deep-cold scenarios weighted to DEEP_COLD_SHARE of the total, see sbx/scenarios.py).
(The kit's baseline_returns.json predates the 2026-11-16 simulator changes, so it is not used.)

    python -m sbx.score baselines
    python -m sbx.score baselines --per-scenario
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from sbx import paths
from sbx.scenarios import DEEP_COLD_SHARE, scenario_info, weights


def load_run(run_dir: Path) -> dict[str, dict[str, dict]]:
    """{policy_label: {scenario: result}} for every result JSON in the run."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for f in sorted(run_dir.glob("*/*.json")):
        r = json.loads(f.read_text())
        if "scenario" in r:
            out[f.parent.name][r["scenario"]] = r
    return out


def station(scenario: str) -> str:
    return re.sub(r"_\d+$", "", scenario.split("/")[1])


def split(r: dict) -> tuple[float, float, float]:
    """(comfort, electricity, gas) loss, each >= 0; they sum to the scenario score."""
    return (-r["mean_comfort"], -(r["mean_elec_cost"] + r["mean_elec_carbon"]),
            -(r["mean_gas_cost"] + r["mean_gas_carbon"]))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="results/<run> or a path")
    ap.add_argument("--per-scenario", action="store_true")
    a = ap.parse_args(argv)
    run_dir = Path(a.run) if Path(a.run).exists() else paths.RESULTS / a.run
    runs = load_run(run_dir)
    info = scenario_info()

    for pol, res in runs.items():
        ok = {k: r for k, r in res.items() if "error" not in r}
        bad = sorted(set(res) - set(ok))
        if not ok:
            print(f"{pol}: no successful results ({len(bad)} errors)")
            continue
        n = len(ok)
        score = sum(r["score"] for r in ok.values()) / n
        c, e, g = (sum(split(r)[i] for r in ok.values()) / n for i in range(3))
        steps = sorted({r["steps"] for r in ok.values()})
        print(f"\n== {pol}: score {score:.5f} over {n} scenarios  "
              f"(comfort {c:.5f} | elec {e:.5f} | gas {g:.5f})  steps={steps}"
              + (f"  ERRORS: {bad}" if bad else ""))
        keys = sorted(ok)
        w = weights(keys, info)
        n_cold = sum(info.get(k, {}).get("climate") == "deep_cold" for k in keys)
        print(f"   cold-weighted ({DEEP_COLD_SHARE:.0%} deep cold; {n_cold} deep-cold scenarios here): "
              f"{sum(wi * ok[k]['score'] for wi, k in zip(w, keys)):.5f}")

        by_station = defaultdict(list)
        for k, r in ok.items():
            by_station[station(k)].append(r["score"])
        print("   per station: " + "  ".join(f"{s} {sum(v) / len(v):.5f} (n={len(v)})"
                                             for s, v in sorted(by_station.items())))
        worst = max(ok.values(), key=lambda r: r["score"])
        print(f"   worst: {worst['scenario']} {worst['score']:.5f}   "
              f"sanitizer fixes: {sum(sum(r['sanitizer_fixes'].values()) for r in ok.values())}   "
              f"max wall {max(r['seconds']['wall'] for r in ok.values()):.0f}s")

        if a.per_scenario:
            for k, r in sorted(ok.items()):
                cc, ee, gg = split(r)
                print(f"   {k:32s} {r['score']:.5f}  c {cc:.5f} e {ee:.5f} g {gg:.5f}  "
                      f"K/V {r['num_ahus']}/{r['num_zones']}  wall {r['seconds']['wall']:.0f}s")


if __name__ == "__main__":
    main()
