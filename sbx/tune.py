"""CMA-ES tuning of TunablePolicy on windowed rollouts (PLAN.md, M4).

Objective per generation: the mean score of the first `--days` days over `--scenarios` scenarios drawn from the
tune split. Scenarios are drawn with the cold-weighted probabilities (sbx.scenarios.weights), so the plain mean over
the draw is an unbiased estimate of the cold-weighted objective. Every candidate of a generation sees the same
draw (common random numbers): CMA-ES only uses ranks within a generation, so the draw-to-draw shift cancels.
M2 showed 7-day windows rank controllers like full episodes (Spearman 0.964 on close controllers).

Search space: parameters normalized to [0, 1] between their bounds (log scale where marked), starting at the
defaults (or `--init`). State is saved every generation, so re-running the same command resumes.

    python -m sbx.tune --run r1 --generations 20 --popsize 12 --scenarios 8 --workers 12
    python -m sbx.tune --run r1 ... --max-hours 7.5   # stop cleanly between generations; same command resumes
    python -m sbx.tune --run r1 --report          # generation table + best/mean parameters -> results/tune/r1/
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sbx import paths
from sbx.controller.tunable import DEFAULTS, PARAMS

STEPS_PER_DAY = 288
OUT = paths.RESULTS / "tune"


# -- parameter encoding ----------------------------------------------------------------------------------------------
def encode(params: dict[str, float], names: list[str]) -> np.ndarray:
    x = []
    for k in names:
        s, v = PARAMS[k], params[k]
        x.append((np.log(v) - np.log(s.low)) / (np.log(s.high) - np.log(s.low)) if s.log
                 else (v - s.low) / (s.high - s.low))
    return np.clip(np.array(x), 0.0, 1.0)


def decode(x: np.ndarray, names: list[str], fixed: dict[str, float]) -> dict[str, float]:
    p = dict(fixed)
    for k, u in zip(names, np.clip(x, 0.0, 1.0)):
        s = PARAMS[k]
        p[k] = float(np.exp(np.log(s.low) + u * (np.log(s.high) - np.log(s.low))) if s.log
                     else s.low + u * (s.high - s.low))
    return p


# -- evaluation ------------------------------------------------------------------------------------------------------
def score_job(scenario_dir: str, params: dict, steps: int) -> float:
    from sbx.controller.tunable import TunablePolicy
    from sbx.env import rollout

    try:
        summary, _ = rollout(Path(scenario_dir), lambda m, s: TunablePolicy(m, s, params), paths.KIT, steps)
        return float(summary["score"]) if all(v == 0 for v in summary["sanitizer_fixes"].values()) else 1.0
    except Exception:
        return 1.0  # a crash is the worst possible candidate


def draw_scenarios(rng: np.random.Generator, split: str, k: int) -> list[str]:
    from sbx.scenarios import load_manifest, scenario_dir, weights

    entries = [s for s in load_manifest()["scenarios"] if s["split"] == split]
    keys = [f"{s['building']}/{s['weather']}" for s in entries]
    w = weights(keys, {k: s for k, s in zip(keys, entries)})
    pick = rng.choice(len(keys), size=k, replace=False, p=w)
    return [str(scenario_dir(entries[i])) for i in sorted(pick)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="tune")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--scenarios", type=int, default=8, help="scenarios drawn per generation")
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=0.15, help="initial step size in normalized units")
    ap.add_argument("--params", nargs="*", default=None, help="parameter names to tune (default: all)")
    ap.add_argument("--init", type=Path, default=None, help="JSON parameters to start from (default: defaults)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop after the last generation that fits in this wall time (resume later with the same command)")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)

    out = OUT / a.run
    out.mkdir(parents=True, exist_ok=True)
    log_path, state_path = out / "generations.jsonl", out / "state.pkl"
    if a.report:
        return report(out)

    import cma

    start = dict(DEFAULTS)
    if a.init:
        init = json.loads(a.init.read_text())
        start.update(init.get("params", init))
    names = a.params or list(PARAMS)
    if state_path.exists():
        es, rng, names, start = pickle.loads(state_path.read_bytes())
        print(f"resumed {a.run} at generation {es.countiter}")
    else:
        es = cma.CMAEvolutionStrategy(encode(start, names), a.sigma,
                                      {"bounds": [0.0, 1.0], "popsize": a.popsize, "seed": a.seed + 1,
                                       "verbose": -9})
        rng = np.random.default_rng(a.seed)
        (out / "config.json").write_text(json.dumps({**vars(a), "init": str(a.init), "names": names,
                                                     "start": start}, indent=2))
    steps = a.days * STEPS_PER_DAY

    started, last_gen = time.time(), 0.0
    with ProcessPoolExecutor(a.workers) as pool:
        while es.countiter < a.generations:
            if a.max_hours is not None and (time.time() - started + last_gen) / 3600 > a.max_hours:
                print(f"stopping: another generation (~{last_gen / 60:.0f} min) would exceed --max-hours "
                      f"{a.max_hours}; re-run the same command to continue", flush=True)
                break
            t0 = time.time()
            xs = es.ask()
            # Also score the current mean and the start point on the same draw: tracks progress across generations.
            cands = [decode(x, names, start) for x in xs] + [decode(es.mean, names, start), dict(start)]
            dirs = draw_scenarios(rng, a.split, a.scenarios)
            futs = [[pool.submit(score_job, d, c, steps) for d in dirs] for c in cands]
            scores = np.array([[f.result() for f in row] for row in futs])
            fit = scores.mean(axis=1)
            es.tell(xs, fit[: len(xs)].tolist())
            best = int(np.argmin(fit[: len(xs)]))
            rec = {"generation": es.countiter, "seconds": round(time.time() - t0), "scenarios": dirs,
                   "best": float(fit[best]), "median": float(np.median(fit[: len(xs)])),
                   "mean_candidate": float(fit[-2]), "start": float(fit[-1]),
                   "gain_vs_start": float(1 - fit[-2] / fit[-1]), "sigma": float(es.sigma),
                   "best_params": cands[best], "mean_params": cands[-2]}
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            state_path.write_bytes(pickle.dumps((es, rng, names, start)))
            last_gen = time.time() - t0
            print(f"gen {es.countiter:3d}  best {rec['best']:.5f}  median {rec['median']:.5f}  "
                  f"mean-candidate {rec['mean_candidate']:.5f}  start {rec['start']:.5f}  "
                  f"({rec['gain_vs_start']:+.1%})  sigma {es.sigma:.3f}  {rec['seconds']}s", flush=True)
    report(out)


def report(out: Path) -> None:
    recs = [json.loads(line) for line in (out / "generations.jsonl").read_text().splitlines()]
    for r in recs:
        print(f"gen {r['generation']:3d}  mean-candidate {r['mean_candidate']:.5f}  start {r['start']:.5f}  "
              f"({r['gain_vs_start']:+.1%})  sigma {r['sigma']:.3f}")
    last = recs[-1]["mean_params"]
    (out / "mean_params.json").write_text(json.dumps({"params": last}, indent=2))
    changed = {k: (DEFAULTS[k], v) for k, v in last.items() if not np.isclose(v, DEFAULTS[k])}
    print(f"\nlatest mean parameters -> {out / 'mean_params.json'}")
    for k, (d, v) in sorted(changed.items()):
        print(f"   {k:28s} {d:12.4f} -> {v:12.4f}")


if __name__ == "__main__":
    main()
