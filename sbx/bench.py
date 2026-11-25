"""Throughput benchmark: simulator seconds per step by building, at several worker counts.

For each (building, workers) it runs `workers` identical short rollouts at the same time and reports the
per-step simulator time (contended) and the aggregate steps/s. Run it on an otherwise idle machine.

    python -m sbx.bench                       # all buildings, 1 / 16 / 30 workers, 864 steps
    python -m sbx.bench --workers 1 8 --steps 288 --policy bang_bang
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from sbx import paths


def _job(scenario_dir: str, policy: str, steps: int, kit: str) -> dict:
    from sbx.env import rollout
    from sbx.policies import resolve

    summary, _ = rollout(Path(scenario_dir), resolve(policy), Path(kit), steps)
    return {"steps": summary["steps"], "K": summary["num_ahus"], "V": summary["num_zones"], **summary["seconds"]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 16, 30])
    ap.add_argument("--steps", type=int, default=864)
    ap.add_argument("--policy", default="pid_water")
    ap.add_argument("--buildings", nargs="*", default=None, help="building names (default: all)")
    ap.add_argument("--out", type=Path, default=paths.RESULTS / "bench" / "throughput.json")
    a = ap.parse_args(argv)

    # One scenario per building (the first weather strip alphabetically).
    first: dict[str, Path] = {}
    for d in paths.discover():
        first.setdefault(d.parent.name, d)
    buildings = a.buildings or sorted(first)

    rows = []
    for b in buildings:
        for w in a.workers:
            t0 = time.perf_counter()
            with ProcessPoolExecutor(max_workers=w) as pool:
                res = list(pool.map(_job, *zip(*[(str(first[b]), a.policy, a.steps, str(paths.KIT))] * w)))
            wall = time.perf_counter() - t0
            ms_step = [1000 * r["sim"] / r["steps"] for r in res]
            row = {
                "building": b, "scenario": paths.scenario_key(first[b]), "K": res[0]["K"], "V": res[0]["V"],
                "workers": w, "steps": a.steps,
                "sim_ms_per_step": statistics.mean(ms_step), "sim_ms_per_step_max": max(ms_step),
                "policy_ms_per_step": statistics.mean(1000 * r["policy"] / r["steps"] for r in res),
                "load_s": statistics.mean(r["load"] for r in res),
                "aggregate_steps_per_s": w * a.steps / wall, "wall_s": wall,
                # Projected wall time of one full 8640-step episode at this contention level.
                "full_episode_s": statistics.mean(r["wall"] / r["steps"] for r in res) * paths.FULL_STEPS,
            }
            rows.append(row)
            print(f"{b:18s} K/V {row['K']}/{row['V']:<3d} workers {w:2d}  sim {row['sim_ms_per_step']:6.1f} ms/step "
                  f"(max {row['sim_ms_per_step_max']:.1f})  aggregate {row['aggregate_steps_per_s']:6.1f} steps/s  "
                  f"full episode ≈ {row['full_episode_s'] / 60:.1f} min", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"cpu_count": os.cpu_count(), "policy": a.policy, "rows": rows}, indent=2))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
