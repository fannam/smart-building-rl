"""Phase 2B evaluation (Workstreams B + D) on every LOBO fold.

Per fold (held-out building; Phase 2A test windows: context 12, horizon 36, stride 6):
  1. persistence / linear / Phase 2A GRU / graph model through the UNCHANGED Phase 2A evaluator
     (dynamics.evaluate.evaluate_models + summarize): horizons 1, 6, 12, 24, 36, profiles,
     severe-cold scenarios, condition strata, 3 h reward-sum error. The GRU numbers are a
     reproducibility check against results/phase2a.
  2. ensembles (GRU seeds 0-2, graph seeds 0-2 where trained): ensemble-mean error, spread,
     Spearman, decile errors, calibration bins; scale fitted on TRAIN-building validation windows.
  3. OOD scores: action-feature Mahalanobis (Phase 2A), decision-state Mahalanobis, GRU latent
     Mahalanobis, ensemble spread. Thresholds (95th pct) and the 'unreliable' error level
     (90th pct of the GRU error) come from TRAIN-building validation windows only.

    .venv/bin/python scripts/evaluate_phase2b.py --parallel 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

ERR_TARGETS = (("zone", 11), ("zone", 35), ("tank", 11), ("elec", 11), ("gas", 11))


def members(fold, kind):
    """Checkpoint paths of ensemble members (Phase 2A artifacts are read-only inputs)."""
    from evaluate_dynamics import fold_dir
    a = fold_dir(ROOT / "results/phase2a", fold)
    b = fold_dir(ROOT / "results/phase2b/ensemble/gru", fold)
    g = fold_dir(ROOT / "results/phase2b/graph_dynamics", fold)
    if kind == "gru":
        out = []
        for s in (0, 1, 2):
            for d in (a, b):
                if (d / f"gru_seed{s}.pt").exists():
                    out.append(d / f"gru_seed{s}.pt"); break
        return out
    return [g / f"graph_seed{s}.pt" for s in (0, 1, 2) if (g / f"graph_seed{s}.pt").exists()]


def collect(ds, models_ens, gru0, ood_fit=None, batch_size=64):
    """Per-window arrays: ensemble errors/spreads per family, single-model errors, OOD features."""
    import torch
    from dynamics.train import loader
    from evaluate_dynamics import action_features
    from uncertainty.scores import decision_latent, ensemble_predict, state_features, window_errors, window_spread
    acc = {}
    zone_el = {}
    with torch.no_grad():
        for batch in loader(ds, batch_size, False, 0, 0):
            for fam, ms in models_ens.items():
                ens = ensemble_predict(ms, batch)
                e_ens = window_errors(ens["zone_mean"], ens["hws_mean"], ens["energy_mean"], batch)
                e_one = window_errors(ens["zone_m0"], ens["hws_m0"], ens["energy_m0"], batch)
                sp = window_spread(ens, batch)
                for q in e_ens:
                    acc.setdefault(f"{fam}|ens_err|{q}", []).append(e_ens[q])
                    acc.setdefault(f"{fam}|single_err|{q}", []).append(e_one[q])
                    acc.setdefault(f"{fam}|spread|{q}", []).append(sp[q])
                zm = batch["zone_mask"].numpy()
                tz = batch["targets"]["zone_temp"].numpy()
                for k in (11, 35):
                    el_err = np.abs(ens["zone_mean"][:, k] - tz[:, k])[zm]
                    el_sd = ens["zone_std"][:, k][zm]
                    zone_el.setdefault(f"{fam}|k{k}", [[], []])
                    zone_el[f"{fam}|k{k}"][0].append(el_err); zone_el[f"{fam}|k{k}"][1].append(el_sd)
            acc.setdefault("latent", []).append(decision_latent(gru0, batch))
            acc.setdefault("state", []).append(state_features(batch))
            acc.setdefault("oat_t0", []).append(batch["inputs"]["oat_obs"][:, -1].numpy())
            acc.setdefault("building", []).append(np.array(batch["info"]["building"]))
            acc.setdefault("scenario", []).append(np.array(batch["info"]["scenario"]))
            acc.setdefault("episode_id", []).append(np.array(batch["info"]["episode_id"]))
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out["action"] = action_features(ds)
    out["zone_elements"] = {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in zone_el.items()}
    return out


def eval_fold(fold_idx: int, out: str, cache: str) -> dict:
    import os
    os.chdir(ROOT)
    import torch
    torch.set_num_threads(4)
    from dynamics.evaluate import evaluate_models, summarize, window_cum_reward_error
    from dynamics.train import fold_data, inner_split, load_gru, make_windows
    from evaluate_dynamics import fold_dir, load_models
    from generalization import station_of
    from graph.train import GraphAuto, load_graph
    from uncertainty.scores import Mahalanobis, calibration_bins, score_vs_error
    cfg = json.loads((ROOT / "configs/phase2b.json").read_text())
    gcfg = json.loads((ROOT / "configs/graph_dynamics.json").read_text())
    fold = json.loads(Path(cfg["folds_config"]).read_text())["folds"][fold_idx]
    t_start = time.time()
    ec = cfg["evaluation"]
    C, H = ec["context"], ec["horizon"]
    root, train_rows, test_rows, _ = fold_data(gcfg, fold, Path(cache))
    assert not ({r["building"] for r in train_rows} & {r["building"] for r in test_rows})
    ds = make_windows(root, test_rows, C, H, Path(cache), stride=ec["test_stride"])
    _, va_f = inner_split(H)
    tr_f, _ = inner_split(H)
    ds_val = make_windows(root, train_rows, C, H, Path(cache), stride=12, t0_filter=va_f)
    ds_fit = make_windows(root, train_rows, C, H, Path(cache), stride=24, t0_filter=tr_f)
    d = Path(out) / f"fold_{fold['fold']}_{fold['test_building']}"
    d.mkdir(parents=True, exist_ok=True)
    base = load_models(fold_dir(ROOT / "results/phase2a", fold))
    gru_paths, graph_paths = members(fold, "gru"), members(fold, "graph")
    models = {"persistence": base["persistence"], "linear": base["linear"], "gru": base["gru"]}
    if graph_paths:
        models["graph"] = GraphAuto(load_graph(graph_paths[0]), root)
    res = evaluate_models(models, ds, ec["batch_size"], 0, ("persistence",))
    np.savez_compressed(d / "eval_results.npz", **{f"{k}::{kk}": vv for k, v in res.items() for kk, vv in v.items()})
    m = res["meta"]
    Hr = ec["report_horizons"]
    weather = np.array([s.split("/")[1] for s in m["scenario"]])
    profile = np.array([e.split("__")[2] for e in m["episode_id"]])
    cold_scen = np.array([w in ("fairbanks_1", "fairbanks_2", "fairbanks_8") for w in weather])
    metrics = {"fold": fold, "test_windows": int(len(m["t0"])), "models": {}, "members": {"gru": [str(p) for p in gru_paths],
                                                                                         "graph": [str(p) for p in graph_paths]}}
    for key in res:
        if key == "meta":
            continue
        metrics["models"][key] = {
            "overall": summarize(res, key, Hr),
            "by_weather": {w: summarize(res, key, Hr, weather == w) for w in sorted(set(weather))},
            "by_profile": {p: summarize(res, key, Hr, profile == p) for p in sorted(set(profile))},
            "severe_cold_scenarios": summarize(res, key, Hr, cold_scen) if cold_scen.any() else None,
            "conditions": {c: summarize(res, key, Hr, step_sel=m[c]) for c in
                           ("occ_transition", "comfort_violation", "heating_recovery", "recovery_window", "cold_step")},
            "cum_reward_error_H": window_cum_reward_error(res, key),
            "plausibility": {"zone_min_pred_k": float(np.min(res[key]["zone_min_pred"])),
                             "zone_max_pred_k": float(np.max(res[key]["zone_max_pred"])),
                             "energy_min_w": float(res[key]["e_pred"].min())}}
    # reproducibility vs Phase 2A
    p2a = json.loads((fold_dir(ROOT / "results/phase2a", fold) / "metrics.json").read_text())
    a12 = p2a["models"]["gru|persistence"]["overall"]["12"]["zone_temp_c"]["mae"]
    b12 = metrics["models"]["gru|persistence"]["overall"][12]["zone_temp_c"]["mae"]
    metrics["phase2a_reproduction"] = {"gru_zone_mae_h12_phase2a": a12, "gru_zone_mae_h12_now": b12, "abs_diff": abs(a12 - b12)}

    # ------------------------------------------------ ensembles, calibration, OOD
    ens_models = {"gru": [load_gru(p) for p in gru_paths]}
    if graph_paths:
        ens_models["graph"] = [GraphAuto(load_graph(p), root) for p in graph_paths]
    gru0 = ens_models["gru"][0]
    te = collect(ds, ens_models, gru0)
    va = collect(ds_val, ens_models, gru0)
    fi = collect(ds_fit, {"gru": ens_models["gru"][:1]}, gru0)
    md = {"action": Mahalanobis().fit(fi["action"]), "state": Mahalanobis().fit(fi["state"]),
          "latent": Mahalanobis().fit(fi["latent"])}
    cold = te["oat_t0"] < 263.15
    unc = {"members": {f: len(v) for f, v in ens_models.items()}, "test_windows": int(len(te["oat_t0"])),
           "cold_windows": int(cold.sum()), "ensemble": {}, "ood": {}, "calibration": {}}
    for fam in ens_models:
        if len(ens_models[fam]) < 2:
            continue
        unc["ensemble"][fam] = {}
        for q, k in ERR_TARGETS:
            sp_te, e_te = te[f"{fam}|spread|{q}"][:, k], te[f"{fam}|ens_err|{q}"][:, k]
            sp_va, e_va = va[f"{fam}|spread|{q}"][:, k], va[f"{fam}|ens_err|{q}"][:, k]
            thr = float(np.quantile(sp_va, 0.95))
            high = float(np.quantile(e_va, 0.90))
            r = {"ensemble_mean_mae": float(e_te.mean()), "single_seed0_mae": float(te[f"{fam}|single_err|{q}"][:, k].mean()),
                 "mean_spread": float(sp_te.mean()), "val_spread_p95": thr, "val_err_p90": high,
                 "all": score_vs_error(sp_te, e_te, thr, high)}
            if cold.any():
                r["cold"] = score_vs_error(sp_te[cold], e_te[cold], thr, high)
                r["not_cold"] = score_vs_error(sp_te[~cold], e_te[~cold], thr, high)
            unc["ensemble"][fam][f"{q}_h{k + 1}"] = r
        for k in (11, 35):
            ev, sv = va["zone_elements"][f"{fam}|k{k}"]
            et, st = te["zone_elements"][f"{fam}|k{k}"]
            scale = float(np.median(ev / np.maximum(sv, 1e-9)) / 0.6745)      # Gaussian-consistent, fitted on train-building validation
            unc["calibration"][f"{fam}|zone_h{k + 1}"] = {"raw": calibration_bins(st, et, 1.0),
                                                         "val_scaled": calibration_bins(st, et, scale),
                                                         "val_self_check": calibration_bins(sv, ev, scale)}
    # OOD scores vs GRU seed0 error (deployed single model) and graph seed0 error
    scores_te = {"action_md": md["action"](te["action"]), "state_md": md["state"](te["state"]), "latent_md": md["latent"](te["latent"])}
    scores_va = {"action_md": md["action"](va["action"]), "state_md": md["state"](va["state"]), "latent_md": md["latent"](va["latent"])}
    for fam in ens_models:
        if len(ens_models[fam]) >= 2:
            scores_te[f"{fam}_spread_zone_h12"] = te[f"{fam}|spread|zone"][:, 11]
            scores_va[f"{fam}_spread_zone_h12"] = va[f"{fam}|spread|zone"][:, 11]
    for fam in ens_models:
        unc["ood"][fam] = {}
        for q, k in ERR_TARGETS:
            e_te, e_va = te[f"{fam}|single_err|{q}"][:, k], va[f"{fam}|single_err|{q}"][:, k]
            high = float(np.quantile(e_va, 0.90))
            unc["ood"][fam][f"{q}_h{k + 1}"] = {}
            for sn in scores_te:
                thr = float(np.quantile(scores_va[sn], 0.95))
                r = {"all": score_vs_error(scores_te[sn], e_te, thr, high), "threshold_val_p95": thr, "val_err_p90": high}
                if cold.any():
                    r["cold"] = score_vs_error(scores_te[sn][cold], e_te[cold], thr, high)
                unc["ood"][fam][f"{q}_h{k + 1}"][sn] = r
    metrics["uncertainty"] = unc
    np.savez_compressed(d / "uncertainty_arrays.npz", **{f"test::{k}": v for k, v in te.items() if k != "zone_elements"},
                        **{f"val::{k}": v for k, v in va.items() if k != "zone_elements"},
                        **{f"score_test::{k}": v for k, v in scores_te.items()},
                        **{f"score_val::{k}": v for k, v in scores_va.items()})
    metrics["eval_s"] = time.time() - t_start
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    return {"fold": fold_idx, "building": fold["test_building"], "eval_s": metrics["eval_s"],
            "repro": metrics["phase2a_reproduction"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--out", default="results/phase2b/graph_dynamics/eval")
    ap.add_argument("--cache", default="scratch/phase1_5/cache")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.parallel) as pool:
        for r in pool.map(eval_fold, a.folds, [str(out)] * len(a.folds), [str(ROOT / a.cache)] * len(a.folds)):
            print(f"evaluated fold {r['fold']} ({r['building']}) in {r['eval_s']:.0f}s  repro {r['repro']}", flush=True)


if __name__ == "__main__":
    main()
