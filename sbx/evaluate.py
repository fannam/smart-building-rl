"""Parallel evaluator: (policy x scenario) rollouts in separate processes.

Writes results/<run>/<policy>/<building>__<weather>.json (summary) and .npz (per-step reward + terms).
Existing results are skipped, so an interrupted run can be resumed by re-running the same command.

    python -m sbx.evaluate --policies pid_water bang_bang --run baselines
    python -m sbx.evaluate --policies pid_water --scenarios 'building_train_8/*' --steps 288 --run smoke
    python -m sbx.evaluate --policies bang_bang --scenarios building_train_8/atlanta_1 --until-done --run horizon
"""
from __future__ import annotations

import os

# One math thread per worker process: with per-process BLAS/numba pools, 6 workers gave only ~2.5x one
# worker's throughput on this machine; single-threaded, ~4x. Set before numpy is imported anywhere.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from sbx import paths


def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def run_job(scenario_dir: str, policy_spec: str, max_steps: int | None, out_json: str, kit: str) -> dict:
    import numpy as np

    from sbx.env import TERMS, rollout
    from sbx.policies import resolve

    key = paths.scenario_key(Path(scenario_dir))
    try:
        summary, arr = rollout(Path(scenario_dir), resolve(policy_spec), Path(kit), max_steps)
        result = {"scenario": key, "policy": policy_spec, "max_steps": max_steps, **summary}
        np.savez_compressed(Path(out_json).with_suffix(".npz"), reward=arr[0],
                            **{t: arr[i + 1] for i, t in enumerate(TERMS)})
    except Exception:
        result = {"scenario": key, "policy": policy_spec, "error": traceback.format_exc()}
    Path(out_json).write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> None:
    from sbx.policies import label

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--scenarios", nargs="*", default=None, help="glob patterns on 'building/weather' (default: all)")
    ap.add_argument("--scenario-root", type=Path, default=paths.TRAINING_SCENARIOS)
    ap.add_argument("--steps", type=int, default=paths.FULL_STEPS)
    ap.add_argument("--until-done", action="store_true", help="ignore --steps; run until the env terminates (gateway)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--run", required=True, help="results/<run>/")
    ap.add_argument("--force", action="store_true", help="re-run jobs that already have a result")
    a = ap.parse_args(argv)

    scenarios = paths.discover(a.scenario_root, a.scenarios)
    if not scenarios:
        sys.exit(f"no scenarios match {a.scenarios} under {a.scenario_root}")
    out = (paths.RESULTS / a.run).resolve()
    max_steps = None if a.until_done else a.steps

    jobs = []
    for spec in a.policies:
        (out / label(spec)).mkdir(parents=True, exist_ok=True)
        for d in scenarios:
            f = out / label(spec) / (paths.scenario_key(d).replace("/", "__") + ".json")
            if f.exists() and not a.force and "error" not in json.loads(f.read_text()):
                continue
            jobs.append((str(d), spec, max_steps, str(f), str(paths.KIT)))

    manifest = {
        "argv": sys.argv if argv is None else argv,
        "repo_sha": _git_sha(paths.ROOT),
        "kit_sha": _git_sha(paths.KIT),
        "scenario_root": str(a.scenario_root),
        "scenarios": [paths.scenario_key(d) for d in scenarios],
        "policies": a.policies,
        "max_steps": max_steps,
        "workers": a.workers,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / f"manifest_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(jobs)} jobs ({len(scenarios) * len(a.policies) - len(jobs)} already done), "
          f"{a.workers} workers, steps={max_steps or 'until done'} -> {out}", flush=True)

    t0 = time.perf_counter()
    failed = 0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(run_job, *j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if "error" in r:
                failed += 1
                print(f"[{i}/{len(jobs)}] FAIL {r['policy']} {r['scenario']}\n{r['error']}", flush=True)
            else:
                s = r["seconds"]
                print(f"[{i}/{len(jobs)}] {label(r['policy']):16s} {r['scenario']:30s} score={r['score']:.5f} "
                      f"steps={r['steps']} wall={s['wall']:.0f}s sim={s['sim']:.0f}s policy={s['policy']:.1f}s "
                      f"({time.perf_counter() - t0:.0f}s elapsed)", flush=True)
    if failed:
        sys.exit(f"{failed} job(s) failed")


if __name__ == "__main__":
    main()
