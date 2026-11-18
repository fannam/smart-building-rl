"""Phase 2B Workstream C: building-level occupancy forecasting + deployable reward estimation.

Per LOBO fold (identical folds/splits as Phase 2A; held-out building never used for fitting):
  forecasters: schedule-only, persistence, time-profile, learned (HistGradientBoosting)
  test windows: identical to the Phase 2A evaluation windows (context 12, horizon 36, stride 6)
  occupancy metrics: MAE/RMSE/bias (persons), occupied-indicator accuracy, arrival/departure
                     detection and timing, per period
  reward estimation (3 h window sums and per step), physics = TRUE or Phase 2A GRU prediction:
      true      simulator reward
      priv      privileged reconstruction (true per-zone occupancy and calendar)  [sanity == true]
      O:<name>  expected comfort = P_hat(N>0) x uniform-zone comfort, deployable calendar/tariff
      O:oracle_total  uniform zones gated by the TRUE future building total  [ceiling of building-level forecasting]

    .venv/bin/python scripts/occupancy_forecast.py --parallel 6 --out results/phase2b/occupancy
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

PERIODS = {"night": (0, 5), "morning": (5, 10), "midday": (10, 16), "evening": (16, 21), "late": (21, 24)}


def build_rows(ds, H, featurizers, with_targets=True):
    """Occupancy design matrix for every window of a DynamicsWindows dataset (runtime inputs only)."""
    from occupancy.forecast import episode_targets
    X, Y, meta = [], [], []
    tgt_cache = {}
    for i in ds.index:
        e, t0 = ds.aw.index[i]
        ep = ds.aw.episodes[e]
        f = featurizers[ep["scenario"]]
        obs = np.asarray(ep["arrays"]["obs"][:t0 + 1])          # rows o_0 .. o_t0 only
        X.append(f.runtime_features(obs, H))
        if with_targets:
            if e not in tgt_cache:
                tgt_cache[e] = episode_targets(ep["arrays"]["zone_info"])
            Y.append(tgt_cache[e][t0:t0 + H] / f.V)
        meta.append((ep["episode_id"], ep["scenario"], ep["building"], t0, f.V))
    return np.concatenate(X), (np.concatenate(Y) if with_targets else None), meta


def transitions(occ_true, occ_pred_p, kind):
    """Per window [H]: first step of a true flip and of the predicted flip of the occupied indicator.
    A predicted indicator already in the post-event state at k=0 counts as a (premature) event at 0."""
    s = occ_true.astype(bool)
    p = occ_pred_p >= 0.5
    want = (False, True) if kind == "arrival" else (True, False)

    def first(x):
        for k in range(1, len(x)):
            if x[k - 1] == want[0] and x[k] == want[1]:
                return k
        return None
    kt, kp = first(s), first(p)
    if kp is None and kt is not None and p[0] == want[1]:
        kp = 0
    return kt, kp


def run_fold(cfg_path, fold_idx, out, cache, seed):
    import os
    os.chdir(ROOT)
    import torch
    torch.set_num_threads(3)
    import reward_model as rm
    from dynamics.evaluate import _reward
    from dynamics.model import rollout
    from dynamics.train import fold_data, load_gru, loader, make_windows, inner_split
    from occupancy.forecast import FEATURES, LearnedHGB, OccupancyFeaturizer, Persistence, ScheduleOnly, TimeProfile
    cfg = json.loads(Path(cfg_path).read_text())
    fold = json.loads(Path(cfg["folds_config"]).read_text())["folds"][fold_idx]
    t_start = time.time()
    root, train_rows, test_rows, _ = fold_data(cfg, fold, Path(cache))
    assert not ({r["building"] for r in train_rows} & {r["building"] for r in test_rows})
    ec = cfg["evaluation"]
    C, H = ec["context"], ec["horizon"]
    tr_f, va_f = inner_split(H)
    ds_tr = make_windows(root, train_rows, C, H, Path(cache), stride=6, t0_filter=tr_f)
    ds_va = make_windows(root, train_rows, C, H, Path(cache), stride=6, t0_filter=va_f)
    ds_te = make_windows(root, test_rows, C, H, Path(cache), stride=ec["test_stride"])
    feats = {s: OccupancyFeaturizer(m) for s, m in {**ds_tr.aw.metas, **ds_te.aw.metas}.items()}
    Xtr, ytr, _ = build_rows(ds_tr, H, feats)
    Xva, yva, _ = build_rows(ds_va, H, feats)
    Xte, yte, mte = build_rows(ds_te, H, feats)
    models = {"schedule": ScheduleOnly().fit(Xtr, ytr, ytr > 0), "persistence": Persistence().fit(Xtr, ytr, ytr > 0),
              "time_profile": TimeProfile().fit(Xtr, ytr, ytr > 0),
              "hgb": LearnedHGB(seed).fit(Xtr, ytr, ytr > 0, Xva, yva, yva > 0)}
    Vrow = np.repeat([m[4] for m in mte], H).astype(np.float64)
    preds = {n: m.predict(Xte) for n, m in models.items()}
    W = len(mte)
    k = np.tile(np.arange(H), W)
    hour = np.mod(np.arctan2(Xte[:, 1], Xte[:, 2]), 2 * np.pi) * 24 / (2 * np.pi)
    occ_true = yte > 0
    res = {"fold": fold, "test_windows": W, "rows": int(len(yte)), "hgb_iters": models["hgb"].iters,
           "hgb_val_mae_per_zone": models["hgb"].val_mae, "train_rows": int(len(ytr)), "occupancy": {}}
    for n, (mu, p) in preds.items():
        err = (mu - yte) * Vrow
        d = {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())), "bias": float(err.mean()),
             "occ_acc": float(np.mean((p >= 0.5) == occ_true)),
             "occ_precision": float(np.mean(occ_true[p >= 0.5])) if (p >= 0.5).any() else float("nan"),
             "occ_recall": float(np.mean(p[occ_true] >= 0.5)) if occ_true.any() else float("nan"),
             "by_k": {int(kk): float(np.abs(err[k == kk]).mean()) for kk in (0, 5, 11, 23, 35)},
             "occupied_mae": float(np.abs(err[occ_true]).mean()), "unoccupied_mae": float(np.abs(err[~occ_true]).mean()),
             "periods": {pn: float(np.abs(err[(hour >= a) & (hour < b)]).mean()) for pn, (a, b) in PERIODS.items()
                         if ((hour >= a) & (hour < b)).any()}}
        for kind in ("arrival", "departure"):
            tim, n_true, n_det, n_fp = [], 0, 0, 0
            for w in range(W):
                sl = slice(w * H, (w + 1) * H)
                kt, kp = transitions(occ_true[sl], p[sl], kind)
                if kt is not None:
                    n_true += 1
                    if kp is not None and abs(kp - kt) <= 6:
                        n_det += 1; tim.append(abs(kp - kt))
                    elif kp is not None:
                        tim.append(abs(kp - kt))
                elif kp is not None:
                    n_fp += 1
            d[kind] = {"true_events": n_true, "detected_within_30min": n_det,
                       "recall": n_det / n_true if n_true else float("nan"), "false_events": n_fp,
                       "timing_err_steps_median": float(np.median(tim)) if tim else float("nan"),
                       "timing_err_steps_mean": float(np.mean(tim)) if tim else float("nan")}
        res["occupancy"][n] = d

    # ---------------- reward estimation (true physics and GRU physics)
    gru = load_gru(Path(cfg["phase2a_dir"]) / f"fold_{fold['fold']}_{fold['test_building']}" / "gru_seed0.pt")
    P = {n: p.reshape(W, H) for n, (_, p) in preds.items()}
    w0 = 0
    acc = {}
    per_window = {}
    with torch.no_grad():
        for batch in loader(ds_te, 64, False, 0, 0):
            t, inp = batch["targets"], batch["inputs"]
            B = t["zone_temp"].shape[0]
            Vm = t["zone_temp"].shape[2]
            zm = batch["zone_mask"].numpy()
            zmask = zm[:, None, :].repeat(H, 1)
            Kb = batch["info"]["K"].numpy()
            setp = inp["setpoint_next"][:, C - 1:].numpy().astype(np.float64)           # schedule setpoints (exact)
            heat, cool = np.broadcast_to(setp[..., :1], (B, H, Vm)), np.broadcast_to(setp[..., 1:], (B, H, Vm))
            hour_n, dow_n = inp["hour_next"][:, C - 1:].numpy(), inp["dow_next"][:, C - 1:].numpy()
            hour_d = np.floor(hour_n + 1e-6).astype(int) % 24
            wd_d = np.floor(dow_n + 1e-6) < 5
            cal = t["calendar"].numpy()
            r_true = t["reward"].numpy()
            occ_z = t["zone_occ"].numpy().astype(np.float64)
            phys = {"true": (t["zone_temp"].numpy().astype(np.float64), t["energy"].numpy().astype(np.float64))}
            pr = rollout(gru, batch)
            phys["gru"] = (pr["zone_temp"].numpy().astype(np.float64), pr["energy_w"].numpy().astype(np.float64))
            r_priv = _reward(np.where(zmask, phys["true"][0], 295.0), t["zone_setpoints"][..., 0].numpy(),
                             t["zone_setpoints"][..., 1].numpy(), occ_z * zmask, phys["true"][1][..., :3].sum(-1),
                             phys["true"][1][..., 3], cal[..., 0], cal[..., 5].astype(bool), Kb, cal[..., 3])
            est = {"true": r_true, "priv": r_priv}
            uni = zmask.astype(np.float64)
            gates = {f"O:{n}": P[n][w0:w0 + B] for n in P}                               # soft: E[comfort] = p * comfort
            gates.update({f"O:{n}_hard": (P[n][w0:w0 + B] >= 0.5).astype(np.float64) for n in ("time_profile", "hgb")})
            gates["O:oracle_total"] = (occ_z.sum(-1) > 0).astype(np.float64)
            for pn, (T, E) in phys.items():
                T = np.where(zmask, T, 295.0)
                elec, gas = E[..., :3].sum(-1), E[..., 3]
                r_uni = _reward(T, heat, cool, uni, elec, gas, hour_d, wd_d, Kb, None)
                r_noc = _reward(T, heat, cool, np.zeros_like(uni), elec, gas, hour_d, wd_d, Kb, None)   # energy only
                comfort_uni = r_uni - r_noc
                for gn, g in gates.items():
                    est[f"{pn}|{gn}"] = r_noc + g * comfort_uni
                est[f"{pn}|O:true_zone"] = _reward(T, heat, cool, occ_z * zmask, elec, gas, hour_d, wd_d, Kb, None)
            for name, r in est.items():
                per_window.setdefault(name, []).append(r)
            per_window.setdefault("_holiday", []).append(((cal[..., 5] == 0) & (cal[..., 2] < 5)).any(1))
            w0 += B
    R = {n: np.concatenate(v) for n, v in per_window.items()}
    hol = R.pop("_holiday")
    rt = R["true"]
    sum_t = rt.sum(1)
    res["reward"] = {}
    for name, r in R.items():
        if name == "true":
            continue
        e_step = np.abs(r - rt)
        e_win = np.abs(r.sum(1) - sum_t)
        res["reward"][name] = {"step_mae": float(e_step.mean()), "window_sum_mae": float(e_win.mean()),
                               "window_sum_bias": float((r.sum(1) - sum_t).mean()),
                               "window_sum_rel_median": float(np.median(e_win / np.maximum(np.abs(sum_t), 1e-3))),
                               "window_sum_err_p95": float(np.percentile(e_win, 95)),
                               "max_window_err_over_true_sum": float(np.max(e_win) / max(np.abs(sum_t).max(), 1e-9)),
                               "window_sum_mae_holiday": float(e_win[hol].mean()) if hol.any() else float("nan"),
                               "window_sum_mae_non_holiday": float(e_win[~hol].mean())}
    res["holiday_windows"] = int(hol.sum())
    res["true_window_sum_mean_abs"] = float(np.abs(sum_t).mean())
    d = Path(out) / f"fold_{fold['fold']}_{fold['test_building']}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "predictions.npz", X=Xte.astype(np.float32), y=yte.astype(np.float32), V=Vrow.astype(np.float32),
                        **{f"mu_{n}": p[0].astype(np.float32) for n, p in preds.items()},
                        **{f"p_{n}": p[1].astype(np.float32) for n, p in preds.items()},
                        **{f"reward_{n}": r.astype(np.float32) for n, r in R.items()}, holiday=hol,
                        episode_id=np.array([m[0] for m in mte]), t0=np.array([m[3] for m in mte]))
    import pickle
    with open(d / "hgb.pkl", "wb") as fh:
        pickle.dump(models["hgb"], fh)
    res["features"] = list(FEATURES)
    res["wall_s"] = time.time() - t_start
    (d / "metrics.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2b.json")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/phase2b/occupancy")
    ap.add_argument("--cache", default="scratch/phase1_5/cache")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.parallel) as pool:
        res = list(pool.map(run_fold, [str(ROOT / a.config)] * len(a.folds), a.folds, [str(out)] * len(a.folds),
                            [str(ROOT / a.cache)] * len(a.folds), [a.seed] * len(a.folds)))
    (out / "metrics_all_folds.json").write_text(json.dumps(res, indent=1))
    for r in res:
        print(r["fold"]["test_building"], {n: round(v["mae"], 2) for n, v in r["occupancy"].items()},
              {n: round(v["window_sum_mae"], 4) for n, v in r["reward"].items() if n.startswith("gru|") or n == "priv"})


if __name__ == "__main__":
    main()
