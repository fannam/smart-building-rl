"""Leaderboard-style scoring of an evaluation run.

Score = mean over scenarios of |mean per-step reward| (lower is better), equal weight per scenario.
Also prints the split into comfort / electricity / gas (cost + carbon), per-station means, and, when the
run covers the shipped training scenarios at 8640 steps, a check against the kit's baseline_returns.json.

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
    ref = json.loads(paths.BASELINE_RETURNS.read_text()) if paths.BASELINE_RETURNS.exists() else {}

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

        by_station = defaultdict(list)
        for k, r in ok.items():
            by_station[station(k)].append(r["score"])
        print("   per station: " + "  ".join(f"{s} {sum(v) / len(v):.5f} (n={len(v)})"
                                             for s, v in sorted(by_station.items())))
        worst = max(ok.values(), key=lambda r: r["score"])
        print(f"   worst: {worst['scenario']} {worst['score']:.5f}   "
              f"sanitizer fixes: {sum(sum(r['sanitizer_fixes'].values()) for r in ok.values())}   "
              f"max wall {max(r['seconds']['wall'] for r in ok.values()):.0f}s")

        # Check against the kit's reference returns (same scenarios, 8640 steps only).
        matched = [(k, r) for k, r in ok.items() if pol in ref.get(k, {}) and r["steps"] == paths.FULL_STEPS]
        if matched:
            diffs = [abs(r["total_reward"] - ref[k][pol]) / abs(ref[k][pol]) for k, r in matched]
            ref_score = sum(abs(ref[k][pol]) / paths.FULL_STEPS for k, _ in matched) / len(matched)
            our = sum(r["score"] for _, r in matched) / len(matched)
            print(f"   vs baseline_returns.json on {len(matched)} scenarios: ours {our:.5f} ref {ref_score:.5f}  "
                  f"max rel diff per scenario {max(diffs):.2e}")

        if a.per_scenario:
            for k, r in sorted(ok.items()):
                cc, ee, gg = split(r)
                refv = ref.get(k, {}).get(pol)
                refs = (f"  ref {abs(refv) / paths.FULL_STEPS:.5f}"
                        if refv is not None and r["steps"] == paths.FULL_STEPS else "")
                print(f"   {k:32s} {r['score']:.5f}  c {cc:.5f} e {ee:.5f} g {gg:.5f}  "
                      f"K/V {r['num_ahus']}/{r['num_zones']}  wall {r['seconds']['wall']:.0f}s{refs}")


if __name__ == "__main__":
    main()
