"""Phase 2A evaluation: persistence vs linear vs GRU on every LOBO fold (held-out building).

    .venv/bin/python scripts/evaluate_dynamics.py --config configs/dynamics_gru.json --parallel 6

Per fold (results/phase2a/fold_<i>_<building>/): eval_results.npz, metrics.json,
probes.json, action_distance.json, (fold 5) ensemble.json.
Across folds: results/phase2a/metrics.json, summary.md, plots/*.png
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

MODELS = ("persistence", "linear", "gru")


def fold_dir(out, fold):
    return out / f"fold_{fold['fold']}_{fold['test_building']}"


def load_models(d: Path):
    import torch
    from dynamics.model import LinearDynamics, Norm, PersistenceModel
    from dynamics.train import load_gru
    from generalization import FittedStatistics
    ms = {}
    for name, cls in (("persistence", PersistenceModel), ("linear", LinearDynamics)):
        ck = torch.load(d / f"{name}.pt", weights_only=False)
        m = cls(Norm.from_stats(FittedStatistics.from_json(ck["stats"])))
        m.load_state_dict(ck["state_dict"])
        ms[name] = m.eval()
    ms["gru"] = load_gru(d / "gru_seed0.pt")
    return ms


# ------------------------------------------------------------------ physics probes
def probes(model, ds, max_windows=400, batch_size=64):
    """Model response to controlled action changes on held-out windows where heating is active.
    Expectations are diagnostic, not enforced laws."""
    import torch
    from dynamics.model import rollout
    from dynamics.train import loader
    acc = {}
    seen = 0
    for batch in loader(ds, batch_size, False, 0, 0):
        inp = batch["inputs"]
        C = inp["zone_temp_obs"].shape[1]
        gas0 = batch["targets"]["energy"][:, 0, 3] > 0
        if not gas0.any():
            continue
        base = rollout(model, batch)
        run_zone = torch.gather(inp["ahu_act"][:, C, :, 0], 1, inp["zone_ahu"].long().clamp_min(0)) > 0.5   # AHU on at a_t0
        hot_water = (inp["hws_obs"][:, C - 1] - (inp["ahu_act"][:, C, :, 1].mean(1) * 10 + 295)) > 5
        zsel = (run_zone & batch["zone_mask"] & hot_water[:, None] & gas0[:, None])
        cases = {}
        b1 = {**batch, "inputs": dict(inp)}
        za = inp["zone_act"].clone(); za[:, C:, :, 1] = (za[:, C:, :, 1] + 0.3).clamp(0, 1)
        b1["inputs"]["zone_act"] = za
        cases["reheat_valve_+0.3"] = (b1, "zone")
        b2 = {**batch, "inputs": dict(inp)}
        ba = inp["bldg_act"].clone(); ba[:, C:, 0] = (ba[:, C:, 0] + 0.5).clamp(max=1.0)
        b2["inputs"]["bldg_act"] = ba
        cases["hws_setpoint_+10K"] = (b2, "plant")
        b3 = {**batch, "inputs": dict(inp)}
        aa = inp["ahu_act"].clone(); aa[:, C:, :, 1] = (aa[:, C:, :, 1] + 0.3).clamp(max=1.0)
        b3["inputs"]["ahu_act"] = aa
        cases["sat_+3K"] = (b3, "zone")
        for name, (bb, kind) in cases.items():
            p = rollout(model, bb)
            a = acc.setdefault(name, {})
            for k in (0, 2, 5, 11):
                dT = (p["zone_temp"][:, k] - base["zone_temp"][:, k])[zsel]
                a.setdefault(f"zone_dT_k{k + 1}", []).append(dT.numpy())
                dH = (p["hws"][:, k] - base["hws"][:, k])[gas0]
                a.setdefault(f"hws_dT_k{k + 1}", []).append(dH.numpy())
                dG = (p["energy_w"][:, k, 3] - base["energy_w"][:, k, 3])[gas0]
                a.setdefault(f"gas_dW_k{k + 1}", []).append(dG.numpy())
        seen += int(gas0.sum())
        if seen >= max_windows:
            break
    out = {"windows_with_heating": seen}
    for name, a in acc.items():
        out[name] = {}
        for k, lst in a.items():
            x = np.concatenate(lst)
            if x.size:
                out[name][k] = {"n": int(x.size), "mean": float(x.mean()), "frac_pos": float(np.mean(x > 1e-3)),
                                "frac_neg": float(np.mean(x < -1e-3)), "p5": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95))}
    return out


# ------------------------------------------------------------------ action distance
def action_features(ds, batch_size=128):
    from dynamics.train import loader
    feats, idx = [], []
    for batch in loader(ds, batch_size, False, 0, 0):
        inp = batch["inputs"]
        C = inp["zone_temp_obs"].shape[1]
        am = batch["ahu_mask"].unsqueeze(1).unsqueeze(-1).float()
        zm = batch["zone_mask"].unsqueeze(1).unsqueeze(-1).float()
        ahu = (inp["ahu_act"][:, C:] * am).sum(2) / am.sum(2).clamp_min(1)      # [B,H,4]
        zone = (inp["zone_act"][:, C:] * zm).sum(2) / zm.sum(2).clamp_min(1)    # [B,H,2]
        f = torch_cat_np([ahu.mean(1), ahu.std(1), inp["bldg_act"][:, C:].mean(1), zone.mean(1)])
        feats.append(f)
    return np.concatenate(feats, 0)


def torch_cat_np(lst):
    import torch
    return torch.cat(lst, -1).numpy().astype(np.float64)


def eval_fold(cfg_path: str, fold_idx: int, out: str, cache: str) -> dict:
    import os
    os.chdir(ROOT)
    import torch
    cfg = json.loads(Path(cfg_path).read_text())
    torch.set_num_threads(4)
    from dynamics.evaluate import evaluate_models, summarize, window_cum_reward_error
    from dynamics.train import fold_data, make_windows, load_gru
    from generalization import station_of
    fold = json.loads(Path(cfg["folds_config"]).read_text())["folds"][fold_idx]
    d = fold_dir(Path(out), fold)
    t_start = time.time()
    ec = cfg["evaluation"]
    root, train_rows, test_rows, _ = fold_data(cfg, fold, Path(cache))
    assert not ({r["building"] for r in train_rows} & {r["building"] for r in test_rows})
    ds = make_windows(root, test_rows, ec["context"], ec["horizon"], Path(cache), stride=ec["test_stride"])
    models = load_models(d)
    res = evaluate_models(models, ds, ec["batch_size"], 0, tuple(ec["exo_modes"]))
    np.savez_compressed(d / "eval_results.npz", **{f"{k}::{kk}": vv for k, v in res.items() for kk, vv in v.items()})
    m = res["meta"]
    H = ec["report_horizons"]
    weather = np.array([s.split("/")[1] for s in m["scenario"]])
    station = np.array([station_of(w) for w in weather])
    profile = np.array([e.split("__")[2] for e in m["episode_id"]])
    cold_scen = np.array([w in ("fairbanks_1", "fairbanks_2", "fairbanks_8") for w in weather])
    metrics = {"fold": fold, "test_windows": int(len(m["t0"])), "horizon": ec["horizon"], "context": ec["context"],
               "models": {}}
    for key in res:
        if key == "meta":
            continue
        mk = {"overall": summarize(res, key, H),
              "by_weather": {w: summarize(res, key, H, weather == w) for w in sorted(set(weather))},
              "by_profile": {p: summarize(res, key, H, profile == p) for p in sorted(set(profile))},
              "severe_cold_scenarios": summarize(res, key, H, cold_scen) if cold_scen.any() else None,
              "conditions": {c: summarize(res, key, H, step_sel=m[c]) for c in
                             ("occ_transition", "comfort_violation", "heating_recovery", "recovery_window", "cold_step")},
              "not_conditions": {c: summarize(res, key, H, step_sel=~m[c]) for c in ("occ_transition", "cold_step")},
              "cum_reward_error_H": window_cum_reward_error(res, key),
              "plausibility": {"zone_min_pred_k": float(np.min(res[key]["zone_min_pred"])),
                               "zone_max_pred_k": float(np.max(res[key]["zone_max_pred"])),
                               "energy_min_w": float(res[key]["e_pred"].min())}}
        # per-zone error at the reporting horizon
        ze = res[key]["zone_abs_err_h"]
        mk["per_zone_mae_c_h12"] = [float(x) for x in np.nanmean(ze, 0)]
        metrics["models"][key] = mk
    metrics["meta"] = {"by_station": {s: int((station == s).sum()) for s in sorted(set(station))},
                       "K": int(m["K"][0]), "V": int(m["V"][0]),
                       "condition_step_counts": {c: int(m[c].sum()) for c in
                                                 ("occ_transition", "comfort_violation", "heating_recovery", "recovery_window", "cold_step")},
                       "reward_sanity_true_exact_max_err": float(np.abs(m["r_true_exact"] - m["r_true"]).max()),
                       "zone_ids": json.loads((root / "scenarios" / test_rows[0]["scenario"].replace("/", "__") / "meta.json").read_text())["zones"]["ids"]}
    # probes + action distance (GRU and linear)
    with torch.no_grad():
        metrics["probes"] = {name: probes(models[name], ds) for name in ("gru", "linear")}
    tr_ds = make_windows(root, train_rows, ec["context"], ec["horizon"], Path(cache), stride=24)
    ftr, fte = action_features(tr_ds), action_features(ds)
    mu, cov = ftr.mean(0), np.cov(ftr.T) + 1e-6 * np.eye(ftr.shape[1])
    inv = np.linalg.inv(cov)
    dist = lambda f: np.sqrt(np.einsum("ni,ij,nj->n", f - mu, inv, f - mu))
    dtr, dte = dist(ftr), dist(fte)
    q = np.percentile(dtr, [50, 90, 99])
    bins = np.digitize(dte, q)
    ad = {"train_quantiles_50_90_99": q.tolist(), "test_frac_beyond_train_p99": float(np.mean(dte > q[2])), "by_bin": {}}
    for key in ("gru|persistence", "linear|persistence"):
        for b in range(4):
            s = bins == b
            if s.any():
                ad["by_bin"].setdefault(key, {})[["<p50", "p50-p90", "p90-p99", ">p99"][b]] = {
                    "windows": int(s.sum()), "zone_mae_h12": float(res[key]["zone_mae"][s, 11].mean()),
                    "zone_mae_h36": float(res[key]["zone_mae"][s, -1].mean()),
                    "gas_mae_kw_h12": float(np.abs(res[key]["e_pred"][s, 11, 3] - m["e_true"][s, 11, 3]).mean() / 1e3)}
    metrics["action_distance"] = ad
    np.save(d / "action_distance_test.npy", dte)
    # ensemble (only where extra seeds exist)
    seeds = sorted(int(p.stem.split("seed")[1]) for p in d.glob("gru_seed*.pt"))
    if len(seeds) > 1:
        from dynamics.model import rollout
        from dynamics.train import loader
        members = [load_gru(d / f"gru_seed{s}.pt") for s in seeds]
        spread, err_mean, err_single = [], [], []
        with torch.no_grad():
            for batch in loader(ds, ec["batch_size"], False, 0, 0):
                preds = torch.stack([rollout(mm, batch)["zone_temp"] for mm in members])     # [M,B,H,V]
                zm = batch["zone_mask"][:, None, :].float()
                tz = batch["targets"]["zone_temp"]
                nz = zm.sum(-1)
                spread.append(((preds.std(0) * zm).sum(-1) / nz).numpy())
                err_mean.append((((preds.mean(0) - tz).abs() * zm).sum(-1) / nz).numpy())
                err_single.append((((preds[0] - tz).abs() * zm).sum(-1) / nz).numpy())
        sp, em, es = (np.concatenate(x) for x in (spread, err_mean, err_single))
        from scipy.stats import spearmanr
        ens = {"seeds": seeds}
        for h in H:
            k = h - 1
            rho = spearmanr(sp[:, k], em[:, k]).statistic
            qs = np.quantile(sp[:, k], [0.5, 0.9])
            ens[h] = {"ensemble_mean_mae_c": float(em[:, k].mean()), "single_seed0_mae_c": float(es[:, k].mean()),
                      "mean_spread_c": float(sp[:, k].mean()), "spearman_spread_vs_error": float(rho),
                      "mae_low_spread_half": float(em[sp[:, k] <= qs[0], k].mean()),
                      "mae_top_spread_decile": float(em[sp[:, k] > qs[1], k].mean())}
        metrics["ensemble"] = ens
        np.savez_compressed(d / "ensemble_spread.npz", spread=sp, err_mean=em, err_single=es)
    metrics["eval_s"] = time.time() - t_start
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    return {"fold": fold_idx, "building": fold["test_building"], "eval_s": metrics["eval_s"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dynamics_gru.json")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--out", default="results/phase2a")
    ap.add_argument("--cache", default="scratch/phase1_5/cache")
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()
    out = ROOT / a.out
    if not a.aggregate_only:
        with ProcessPoolExecutor(max_workers=a.parallel) as pool:
            for r in pool.map(eval_fold, [str(ROOT / a.config)] * len(a.folds), a.folds, [str(out)] * len(a.folds),
                              [str(ROOT / a.cache)] * len(a.folds)):
                print(f"evaluated fold {r['fold']} ({r['building']}) in {r['eval_s']:.0f}s", flush=True)
    from dynamics.report import aggregate_and_plot
    aggregate_and_plot(out, json.loads((ROOT / a.config).read_text()))


if __name__ == "__main__":
    main()
