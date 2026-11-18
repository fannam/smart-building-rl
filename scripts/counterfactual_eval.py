"""Phase 2B Stage 1c: model-predicted vs simulated paired responses (true counterfactual validation).

For every simulated scenario (scripts/counterfactual_run.py) the models of the fold that HOLDS OUT
that building are evaluated (unseen building), plus a GRU from a fold that TRAINED on the building
('gru_indist', diagnostic: separates unseen-building generalization from action-response learning).

    .venv/bin/python scripts/counterfactual_eval.py --parallel 6
Outputs results/phase2b/counterfactual/model_eval/{rows.csv.gz, trajectories.npz, summary.json}
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


class EnsembleMean:
    def __init__(self, members):
        self.members = members

    def predict(self, batch, exo_mode="persistence"):
        import torch
        from dynamics.model import rollout
        outs = [rollout(m, batch, exo_mode) for m in self.members]
        return {k: torch.stack([o[k] for o in outs]).mean(0) for k in ("zone_temp", "hws", "energy_w")}


def eval_file(path: str, exo_modes=("persistence",)) -> dict:
    import os
    os.chdir(ROOT)
    import torch
    torch.set_num_threads(2)
    from counterfactual.model_eval import QUANTITIES, build_items, evaluate_scenario, model_quantities, sim_quantities
    from dataset import load_scenario_meta
    from dynamics.train import load_gru
    from evaluate_dynamics import fold_dir, load_models
    from evaluate_phase2b import members
    from graph.train import GraphAuto, load_graph
    sim = dict(np.load(path))
    scenario = Path(path).stem.replace("__", "/")
    sim["scenario"] = np.array(scenario)
    building = scenario.split("/")[0]
    folds = json.loads((ROOT / "configs/leave_one_building_out.json").read_text())["folds"]
    fold = next(f for f in folds if f["test_building"] == building)
    other = folds[(fold["fold"] + 1) % len(folds)]
    assert building in other["train_buildings"]
    root = ROOT / "data/phase1_sample"
    meta = load_scenario_meta(root, scenario)
    base = load_models(fold_dir(ROOT / "results/phase2a", fold))
    models = {"persistence": base["persistence"], "linear": base["linear"], "gru": base["gru"],
              "gru_indist": load_gru(fold_dir(ROOT / "results/phase2a", other) / "gru_seed0.pt")}
    gp = members(fold, "gru")
    if len(gp) > 1:
        models["gru_ens"] = EnsembleMean([load_gru(p) for p in gp])
    grp = members(fold, "graph")
    if grp:
        models["graph"] = GraphAuto(load_graph(grp[0]), root)
        if len(grp) > 1:
            models["graph_ens"] = EnsembleMean([GraphAuto(load_graph(p), root) for p in grp])
    rows = evaluate_scenario(models, sim, meta, exo_modes)
    for r in rows:
        r["scenario"], r["building"], r["fold"] = scenario, building, fold["fold"]
    # mean trajectories (zone-mean response) per arm for plots
    names = [str(x) for x in sim["arm_names"]]
    traj = {}
    for p in range(len(sim["t_b"])):
        items = build_items(sim, meta, p)
        sq = sim_quantities(sim, p)
        for q in ("zone_temp", "tank", "gas_w"):
            s = sq[q] - sq[q][:1]
            traj.setdefault(f"sim|{q}", []).append(s.mean(-1) if s.ndim == 3 else s)
        for mn in ("gru", "graph", "linear", "gru_indist"):
            if mn not in models:
                continue
            pq = model_quantities(models[mn], items)
            for q in ("zone_temp", "tank", "gas_w"):
                d = pq[q] - pq[q][:1]
                traj.setdefault(f"{mn}|{q}", []).append(d.mean(-1) if d.ndim == 3 else d)
    traj = {k: np.stack(v) for k, v in traj.items()}       # [P, A, H]
    return {"rows": rows, "traj": traj, "arm_names": names, "scenario": scenario, "models": list(models),
            "strata": [str(x) for x in sim["stratum"]], "oat_band": [str(x) for x in sim["oat_band"]],
            "checks": json.loads(str(sim["checks_json"]))}


def summarize(df):
    import pandas as pd
    out = {}
    r = df[(df.kind == "response")]
    eff = r[~r.effective_noop.astype(bool)]
    resp = eff[eff.n_responsive > 0]

    def pooled(g):
        return pd.Series({"n_cases": len(g), "sign_agree": g.sign_agree.mean(), "sign_wrong": g.sign_wrong.mean(),
                          "gain_pooled": g.dot_ms.sum() / max(g.dot_ss.sum(), 1e-12), "gain_median": g.gain.median(),
                          "rel_err_median": g.rel_err_cum.median(),
                          "norm_abs_err": g.sum_abs_err.sum() / max(g.sum_abs_s.sum(), 1e-12),
                          "latency_sim": g.latency_sim.median(), "latency_model": g.latency_model.median(),
                          "path_mae": g.path_mae.mean()})
    out["by_family_model"] = {q: resp[resp.quantity == q].groupby(["family", "model"]).apply(pooled).round(4).reset_index().to_dict("records")
                              for q in resp.quantity.unique()}
    out["by_family_model_magnitude"] = resp[resp.quantity == "zone_temp"].groupby(["family", "magnitude", "duration", "model"]).apply(pooled).round(4).reset_index().to_dict("records")
    for col in ("stratum", "oat_band", "building"):
        out[f"zone_by_{col}"] = resp[resp.quantity == "zone_temp"].groupby([col, "model"]).apply(pooled).round(4).reset_index().to_dict("records")
        out[f"zone_family_by_{col}"] = resp[resp.quantity == "zone_temp"].groupby(["family", col, "model"]).apply(pooled).round(4).reset_index().to_dict("records")
    noop = r[r.effective_noop.astype(bool)]
    out["effective_noop_arms"] = {"n": int(len(noop[noop.quantity == "zone_temp"]) // max(1, df.model.nunique())),
                                  "spurious_rate_by_model": noop[noop.quantity == "zone_temp"].groupby("model").spurious_rate.mean().round(4).to_dict()}
    nresp = eff[(eff.n_responsive == 0)]
    out["no_sim_response_spurious"] = nresp.groupby(["quantity", "model"]).spurious_rate.mean().round(4).reset_index().to_dict("records")
    a = df[df.kind == "absolute"]
    out["nominal_abs_err"] = a.groupby(["quantity", "model"]).abs_err_mean.mean().round(4).reset_index().to_dict("records")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", default="results/phase2b/counterfactual/sim")
    ap.add_argument("--out", default="results/phase2b/counterfactual/model_eval")
    ap.add_argument("--parallel", type=int, default=6)
    a = ap.parse_args()
    import pandas as pd
    files = sorted(str(p) for p in (ROOT / a.sim_dir).glob("building_train_*.npz"))
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.parallel) as pool:
        res = list(pool.map(eval_file, files))
    df = pd.DataFrame([r for x in res for r in x["rows"]])
    df.to_csv(out / "rows.csv.gz", index=False)
    np.savez_compressed(out / "trajectories.npz", **{f"{x['scenario']}::{k}": v for x in res for k, v in x["traj"].items()},
                        arm_names=np.array(res[0]["arm_names"]))
    summ = summarize(df)
    summ["scenarios"] = [{"scenario": x["scenario"], "strata": x["strata"], "oat_band": x["oat_band"], "models": x["models"],
                          "nominal_replay_bit_exact": all(c["nominal_replay_vs_original"]["bit_exact"] for c in x["checks"])}
                         for x in res]
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    z = pd.DataFrame(summ["by_family_model"]["zone_temp"])
    print(z.pivot(index="family", columns="model", values="gain_pooled").round(2).to_string())
    print(z.pivot(index="family", columns="model", values="sign_agree").round(2).to_string())


if __name__ == "__main__":
    main()
