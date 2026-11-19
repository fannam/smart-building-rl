"""Phase 2C counterfactual collection driver (E9 paired branches with E1/E2 zone-subset excitation).

APPROVED design: configs/phase2c/collection.json. Intended to run on Kaggle CPU
(kaggle/phase2c_collect_counterfactual.ipynb); locally only for dry runs into scratch/.

Resumable: every branch point is one file <out>/branches/<scenario>__t<t_b>.npz written atomically; a
rerun (same or later session, same config) skips complete branch points and never duplicates them.
The main trajectory is replayed deterministically to reach pending branch points.

    python scripts/phase2c_collect.py --config configs/phase2c/collection.json --out <dir> --validate-only
    python scripts/phase2c_collect.py --config configs/phase2c/collection.json --out <dir> --parallel 4 --time-budget-hours 10.5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# measured Phase 2B seconds per arm under 12-way load (docs/phase2c_data_collection_proposal.md section 4)
SEC_PER_ARM = {"building_train_1": 12.5, "building_train_2": 22.2, "building_train_4": 17.7, "building_train_5": 17.7,
               "building_train_7": 7.8, "building_train_8": 6.7}


def worker(scenario, cfg, out, data_root, deadline, exclude):
    import os
    os.chdir(ROOT)
    from counterfactual.phase2c import collect_scenario
    try:
        return collect_scenario(scenario, cfg, out, data_root, deadline, exclude, log=lambda m: print(m, flush=True))
    except Exception:  # noqa: BLE001
        return {"scenario": scenario, "fatal": traceback.format_exc()}


def plan_all(cfg, scenarios, data_root, exclude_map):
    from counterfactual.interventions import step_conditions
    from counterfactual.phase2c import plan_branch_points
    from dataset import load_episode, load_scenario_meta
    plans = {}
    for sc in scenarios:
        ref = load_episode(Path(data_root) / "episodes" / f"{sc.replace('/', '__')}__baseline__s0__e0.npz",
                           keys=["obs", "reward", "calendar", "boiler_info", "ahu_info", "action_discrete"])
        cond = step_conditions(ref, load_scenario_meta(Path(data_root), sc))
        plans[sc] = plan_branch_points(cond, int(cfg["n_points_per_scenario"]), cfg["stratum_weights"], int(cfg["context"]),
                                       int(cfg["horizon"]), int(cfg["min_spacing_steps"]), exclude_map.get(sc, []),
                                       int(cfg["exclude_near_radius_steps"]), int(cfg["seed"]), sc,
                                       bool(cfg.get("prefer_ahu_on_within_stratum", False)))
    return plans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2c/collection.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=str(ROOT / "data/phase1_sample"))
    ap.add_argument("--scenarios", nargs="+", default=["all"])
    ap.add_argument("--n-points", type=int, default=None, help="override n_points_per_scenario (dry runs)")
    ap.add_argument("--exclude-json", default=str(ROOT / "configs/phase2c/phase2b_branch_points.json"))
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--time-budget-hours", type=float, default=10.5,
                    help="no new branch point is started after this; finished work is kept for the next session")
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args()
    # absolute paths: the simulator setup chdir()s into the starter kit
    a.out, a.data_root = str(Path(a.out).resolve()), str(Path(a.data_root).resolve())
    a.config, a.exclude_json = str(Path(a.config).resolve()), str(Path(a.exclude_json).resolve())
    from counterfactual.phase2c import SCHEMA_VERSION, all_scenarios, branch_complete, branch_file, expand_arms, validate_config
    cfg = json.loads(Path(a.config).read_text())
    if a.n_points is not None:
        cfg["n_points_per_scenario"] = a.n_points
    errs = validate_config(cfg)
    if errs:
        sys.exit("CONFIG INVALID:\n  " + "\n  ".join(errs))
    scenarios = all_scenarios(Path(a.data_root)) if a.scenarios == ["all"] else a.scenarios
    exclude_map = json.loads(Path(a.exclude_json).read_text())["branch_points"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    A = len(expand_arms(cfg)) + 1
    plans = plan_all(cfg, scenarios, a.data_root, exclude_map)
    done = {sc: sum(branch_complete(branch_file(out, sc, t), A) for t, _ in p) for sc, p in plans.items()}
    n_plan, n_done = sum(len(p) for p in plans.values()), sum(done.values())
    est = sum((len(p) - done[sc]) * A * SEC_PER_ARM[sc.split("/")[0]] for sc, p in plans.items())
    # Test-A leakage guard: no planned point within the exclusion radius of a Phase 2B point
    r = int(cfg["exclude_near_radius_steps"])
    leak = [(sc, t) for sc, p in plans.items() for t, _ in p for u in exclude_map.get(sc, []) if abs(t - u) <= r]
    if leak:
        sys.exit(f"Test-A leakage in plan: {leak[:5]}")
    plan_doc = {"schema_version": SCHEMA_VERSION, "config": cfg, "arms_per_branch": A, "scenarios": len(scenarios),
                "planned_branch_points": n_plan, "complete_branch_points": n_done,
                "plan": {sc: [{"t_b": t, "stratum": s} for t, s in p] for sc, p in plans.items()},
                "strata_counts": {s: sum(1 for p in plans.values() for _, x in p if x == s)
                                  for s in cfg["stratum_weights"]}}
    (out / "plan.json").write_text(json.dumps(plan_doc, indent=1))
    print(f"config OK: {len(scenarios)} scenarios, {n_plan} planned branch points x {A} arms = {n_plan * A} arms; "
          f"complete {n_done}, pending {n_plan - n_done}", flush=True)
    print(f"strata: {plan_doc['strata_counts']}", flush=True)
    print(f"rough pending cost: {est / 3600:.1f} core-hours (Phase 2B loaded rates; unloaded is faster) "
          f"-> ~{est / 3600 / max(a.parallel, 1):.1f} h wall at --parallel {a.parallel}", flush=True)
    if a.validate_only:
        return
    t0 = time.time()
    deadline = t0 + a.time_budget_hours * 3600
    todo = sorted([sc for sc in scenarios if done[sc] < len(plans[sc])],
                  key=lambda s: -(len(plans[s]) - done[s]) * SEC_PER_ARM[s.split("/")[0]])   # longest first
    results = []
    with ProcessPoolExecutor(max_workers=a.parallel) as pool:
        futs = {pool.submit(worker, sc, cfg, str(out), a.data_root, deadline, exclude_map.get(sc, [])): sc for sc in todo}
        for fu in as_completed(futs):
            r = fu.result()
            results.append(r)
            if "fatal" in r:
                print(f"FAIL {r['scenario']}\n{r['fatal']}", flush=True)
            else:
                print(f"== {r['scenario']}: +{r['collected']} (had {r['already_complete']}/{r['planned']})"
                      f"{' STOPPED BY TIME BUDGET' if r['stopped_by_deadline'] else ''}, replay max {r.get('replay_obs_max_abs', 0):.1e}, "
                      f"errors {len(r['errors'])}, {r['wall_s'] / 60:.1f} min  [{len(results)}/{len(todo)} scenarios, "
                      f"{(time.time() - t0) / 3600:.2f} h]", flush=True)
    log_path = out / "session_logs" / f"session_{int(t0)}.json"
    log_path.parent.mkdir(exist_ok=True)
    log_path.write_text(json.dumps({"argv": sys.argv, "started": t0, "wall_s": time.time() - t0, "results": results}, indent=1, default=str))
    n_done2 = sum(branch_complete(branch_file(out, sc, t), A) for sc, p in plans.items() for t, _ in p)
    print(f"SESSION END: complete {n_done2}/{n_plan} branch points "
          f"({'ALL DONE' if n_done2 == n_plan else 'resume in a new session'}); fatal {sum('fatal' in r for r in results)}", flush=True)


if __name__ == "__main__":
    main()
