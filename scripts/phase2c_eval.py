"""Phase 2C held-out counterfactual evaluation: model-predicted vs simulated paired responses.

For every branch point of the requested split(s), the models of the LOBO fold that HOLDS OUT that
point's building are evaluated, so every prediction is on a building the model never trained on.
`test_b` (default) is the `denver_*` scenario of each building: unseen building AND unseen weather
strip. `train`/`val` points of the held-out building are also unseen by that fold (LOBO trains on
the other 5 buildings only), so `--splits train val test_b` evaluates all 50 points per building.
Test-A (the Phase 2B set) is not supported here: its simulator archives are not part of this data.

Models (discovered under --runs; a checkpoint is used ONLY if its train_log_seed<s>.json exists, so a
partial checkpoint of an interrupted job is never evaluated):
    <run>_s<seed>   every finished checkpoint, e.g. graph_paired_lam1_s0, gru_arms_lam0_s0
    <run>_ens       ensemble mean, for runs with >= 2 finished seeds in that fold
    zero            predicts no response at all (the floor every model must beat)

Metrics per (branch point, arm, quantity, model) come from the Phase 2B evaluator
(`counterfactual.model_eval.response_metrics`): least-squares response gain (1 = exact), sign
agreement, latency, spurious response, path MAE. Added here: pooled gains at horizons 1/3/6/12/24/36
(response averaged over the first h steps, on the elements that respond within 3 h).

    .venv/bin/python scripts/phase2c_eval.py --splits test_b --parallel 6
    .venv/bin/python scripts/phase2c_eval.py --limit 2 --buildings building_train_8   # smoke test

Outputs <out>/{rows.csv.gz, summary.json, timing.json}; prints the pooled zone-temperature gain table.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

HORIZONS = (1, 3, 6, 12, 24, 36)


def discover_models(runs_dir: Path, fold: int) -> dict[str, list[tuple[str, Path]]]:
    """{run: [(kind, checkpoint), ...]} for one fold; finished checkpoints only."""
    out: dict[str, list[tuple[str, Path]]] = {}
    for ck in sorted(runs_dir.glob(f"*/fold_{fold}_*/model_seed*.pt")):
        seed = re.search(r"model_seed(\d+)\.pt$", ck.name).group(1)
        if not (ck.parent / f"train_log_seed{seed}.json").exists():
            print(f"  skip {ck} (no train log: interrupted job)", flush=True)
            continue
        run = ck.parent.parent.name
        out.setdefault(run, []).append((run.split("_")[0], ck))
    return out


def load_models(runs_dir: Path, fold: int, phase1_root: Path) -> dict:
    from counterfactual_eval import EnsembleMean
    from dynamics.train import load_gru
    from graph.train import GraphAuto, load_graph

    def load(kind, ck):
        return load_gru(ck) if kind == "gru" else GraphAuto(load_graph(ck), phase1_root)

    models = {}
    for run, cks in discover_models(runs_dir, fold).items():
        members = []
        for kind, ck in cks:
            m = load(kind, ck)
            models[f"{run}_s{re.search(r'seed(\d+)', ck.name).group(1)}"] = m
            members.append(m)
        if len(members) >= 2:
            models[f"{run}_ens"] = EnsembleMean(members)
    return models


def horizon_gains(ds: np.ndarray, dm: np.ndarray, thr: float) -> dict[str, float]:
    """Pooled-gain numerators/denominators of the response averaged over the first h steps."""
    ds2, dm2 = ds.reshape(ds.shape[0], -1), dm.reshape(dm.shape[0], -1)
    resp = np.abs(ds2).max(0) > thr
    out = {}
    if resp.any():
        for h in HORIZONS:
            s, m = ds2[:h, resp].mean(0), dm2[:h, resp].mean(0)
            out[f"dot_ms_h{h}"], out[f"dot_ss_h{h}"] = float(np.dot(m, s)), float(np.dot(s, s))
    return out


def eval_building(job: dict) -> dict:
    """All requested branch points of one building, with the models of the fold holding it out."""
    import os
    os.chdir(ROOT)
    import torch
    torch.set_num_threads(job["threads"])
    from counterfactual.model_eval import QUANTITIES, THRESH, build_items, model_quantities, response_metrics, sim_quantities
    from dataset import load_scenario_meta

    t0 = time.time()
    fold, cf, p1 = job["fold"], Path(job["cf_data"]), Path(job["phase1_root"])
    models = load_models(Path(job["runs"]), fold["fold"], p1)
    t_load = time.time() - t0
    rows, metas, per_point = [], {}, []
    for bid, split in job["points"]:
        tp = time.time()
        sim = dict(np.load(cf / "branches" / f"{bid}.npz"))
        sc = str(sim["scenario"])
        if sc not in metas:
            metas[sc] = load_scenario_meta(p1, sc)
        items = build_items(sim, metas[sc], 0)
        sq = sim_quantities(sim, 0)
        with torch.no_grad():
            preds = {mn: model_quantities(m, items) for mn, m in models.items()}
        names = [str(x) for x in sim["arm_names"]]
        ivs = {d["name"]: d for d in json.loads(str(sim["interventions_json"]))}
        ctx = {"branch_id": bid, "split": split, "building": job["building"], "fold": fold["fold"], "scenario": sc,
               "t_b": int(sim["t_b"][0]), "stratum": str(sim["stratum"][0]), "oat_c": float(sim["oat_c"][0]),
               "oat_band": str(sim["oat_band"][0]), "heating": bool(sim["heating"][0]), "ahu_on": bool(sim["ahu_on"][0]),
               "occupied": bool(sim["occupied"][0])}
        for mn, pq in preds.items():                       # nominal-arm absolute accuracy (sanity check)
            for q in QUANTITIES:
                rows.append({**ctx, "model": mn, "arm": "nominal", "quantity": q, "kind": "absolute",
                             "abs_err_mean": float(np.abs(pq[q][0] - sq[q][0]).mean())})
        for a in range(1, len(names)):
            if bool(sim["aborted"][0, a]):
                continue
            iv = ivs[names[a]]
            info = {**ctx, "arm": names[a], "group": iv["group"], "family": iv["family"], "magnitude": iv["magnitude"],
                    "duration": int(iv["duration"]), "value": float(sim["arm_value"][0, a]),
                    "effective_noop": bool(sim["effective_noop"][0, a]), "active_change": bool(sim["active_change"][0, a]),
                    "extreme_excursion": bool(sim["extreme_excursion"][0, a])}
            for q in QUANTITIES:
                ds = sq[q][a] - sq[q][0]
                for mn, pq in preds.items():
                    dm = pq[q][a] - pq[q][0]
                    rows.append({**info, "model": mn, "quantity": q, "kind": "response",
                                 **response_metrics(ds, dm, THRESH[q]), **horizon_gains(ds, dm, THRESH[q])})
                zero = np.zeros_like(ds)
                rows.append({**info, "model": "zero", "quantity": q, "kind": "response",
                             **response_metrics(ds, zero, THRESH[q]), **horizon_gains(ds, zero, THRESH[q])})
        per_point.append(time.time() - tp)
    return {"rows": rows, "building": job["building"], "fold": fold["fold"], "models": sorted(models) + ["zero"],
            "n_points": len(job["points"]), "load_s": t_load, "point_s": per_point, "wall_s": time.time() - t0}


def pooled(g):
    import pandas as pd
    d = {"n_cases": len(g), "gain_pooled": g.dot_ms.sum() / max(g.dot_ss.sum(), 1e-12),
         "sign_agree": g.sign_agree.mean(), "sign_wrong": g.sign_wrong.mean(),
         "norm_abs_err": g.sum_abs_err.sum() / max(g.sum_abs_s.sum(), 1e-12),
         "latency_sim": g.latency_sim.median(), "latency_model": g.latency_model.median(), "path_mae": g.path_mae.mean()}
    for h in HORIZONS:
        if f"dot_ms_h{h}" in g:
            d[f"gain_h{h}"] = g[f"dot_ms_h{h}"].sum() / max(g[f"dot_ss_h{h}"].sum(), 1e-12)
    return pd.Series(d)


def summarize(df) -> dict:
    r = df[df.kind == "response"]
    eff = r[~r.effective_noop.astype(bool)]
    resp = eff[eff.n_responsive > 0]
    z = resp[resp.quantity == "zone_temp"]

    def tab(frame, keys):
        return frame.groupby(keys).apply(pooled, include_groups=False).round(4).reset_index().to_dict("records")

    out = {"pooled_by_quantity_model": tab(resp, ["quantity", "model"]),
           "by_family_model": {q: tab(resp[resp.quantity == q], ["family", "model"]) for q in sorted(resp.quantity.unique())},
           "zone_by_family_magnitude_duration": tab(z, ["family", "magnitude", "duration", "model"])}
    for col in ("building", "stratum", "oat_band"):
        out[f"zone_by_{col}"] = tab(z, [col, "model"])
    noop = r[r.effective_noop.astype(bool) & (r.quantity == "zone_temp")]
    out["effective_noop_spurious_rate"] = noop.groupby("model").spurious_rate.mean().round(4).to_dict()
    a = df[df.kind == "absolute"]
    out["nominal_abs_err"] = a.groupby(["quantity", "model"]).abs_err_mean.mean().round(4).reset_index().to_dict("records")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test_b"], choices=["train", "val", "test_b"])
    ap.add_argument("--buildings", nargs="*", default=None, help="default: all six")
    ap.add_argument("--limit", type=int, default=None, help="max branch points per building (smoke tests)")
    ap.add_argument("--runs", default="results/phase2c/training")
    ap.add_argument("--split-manifest", default="results/phase2c/split_manifest.json")
    ap.add_argument("--cf-data", default="data/phase2c_counterfactual")
    ap.add_argument("--phase1-root", default="data/phase1_sample")
    ap.add_argument("--out", default=None, help="default: results/phase2c/eval/<splits>")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    a = ap.parse_args()
    import pandas as pd

    for p in (a.cf_data, a.phase1_root, a.split_manifest, a.runs):
        assert (ROOT / p).exists(), f"missing {p}: unpack data/phase2c_train.zip first (docs/phase2c_report.md §12)"
    man = json.loads((ROOT / a.split_manifest).read_text())
    folds = json.loads((ROOT / "configs/leave_one_building_out.json").read_text())["folds"]
    jobs = []
    for f in folds:
        b = f["test_building"]
        if a.buildings and b not in a.buildings:
            continue
        pts = sorted((k, v["split"]) for k, v in man["assignments"].items() if v["building"] == b and v["split"] in a.splits)
        pts = pts[: a.limit] if a.limit else pts
        if pts:
            jobs.append({"fold": f, "building": b, "points": pts, "runs": str(ROOT / a.runs), "cf_data": str(ROOT / a.cf_data),
                         "phase1_root": str(ROOT / a.phase1_root), "threads": a.threads})
    n = sum(len(j["points"]) for j in jobs)
    print(f"{n} branch points, {len(jobs)} buildings, splits {a.splits}, parallel {a.parallel}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(min(a.parallel, len(jobs))) as pool:
        res = []
        for r in pool.map(eval_building, jobs):
            pt = r["point_s"]
            print(f"  {r['building']} (fold {r['fold']}): {r['n_points']} points, {len(r['models'])} models, "
                  f"load {r['load_s']:.0f}s, {np.mean(pt):.1f}s/point, wall {r['wall_s']:.0f}s", flush=True)
            res.append(r)
    wall = time.time() - t0
    out = ROOT / (a.out or f"results/phase2c/eval/{'_'.join(a.splits)}" + (f"_limit{a.limit}" if a.limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row for r in res for row in r["rows"]])
    df.to_csv(out / "rows.csv.gz", index=False)
    summ = summarize(df)
    summ["run"] = {"splits": a.splits, "buildings": [j["building"] for j in jobs], "n_points": n, "limit": a.limit,
                   "models_per_building": {r["building"]: r["models"] for r in res}, "argv": sys.argv}
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    timing = {"wall_s": wall, "parallel": a.parallel, "threads": a.threads,
              "per_building": {r["building"]: {"points": r["n_points"], "models": len(r["models"]), "load_s": r["load_s"],
                                               "mean_point_s": float(np.mean(r["point_s"])), "wall_s": r["wall_s"]} for r in res}}
    (out / "timing.json").write_text(json.dumps(timing, indent=1))
    z = pd.DataFrame(summ["by_family_model"].get("zone_temp", []))
    if len(z):
        pd.set_option("display.width", 250)
        print("\nzone temperature, pooled response gain (1 = exact, 0 = no response):")
        print(z.pivot(index="family", columns="model", values="gain_pooled").round(2).to_string())
        print("\nzone temperature, sign agreement:")
        print(z.pivot(index="family", columns="model", values="sign_agree").round(2).to_string())
    print(f"\nwall {wall:.0f}s -> {out}")


if __name__ == "__main__":
    main()
