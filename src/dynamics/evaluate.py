"""Evaluation of dynamics models on a held-out building (one LOBO fold).

Per test window (context C, horizon H) and rollout step k (transition t0+k), stores:
    zone temperature error (MAE / signed / squared over zones), HWS error,
    predicted and true energy components [W], three reward variants, condition flags.
Teacher-forced = k = 0 (all inputs are true observations); autoregressive = k >= 1
(inputs are the model's own predictions; outside air per exo_mode).

Reward variants (reward_model.step_reward; true simulator reward is the reference):
    exact_phys   predicted zone T, E, G + TRUE per-zone occupancy, true hour/work-day/month
                 -> isolates dynamics error (NOT deployable: privileged occupancy/calendar)
    deployable   predicted zone T, E, G + schedule setpoints, uniform occupancy gated by the
                 schedule comfort mode, hour/dow from the clock, Mon-Fri, annual-mean gas price
    true_deploy  TRUE zone T, E, G with the deployable occupancy/calendar assumptions
                 -> the estimator-only error (no dynamics error)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

import reward_model as rm
from dynamics.features import ENERGY_NAMES
from dynamics.model import rollout
from dynamics.schedule import comfort_mode
from dynamics.train import loader, make_windows

COLD_STEP_K = 263.15     # outside air below -10 C at the step
TRANSITION_RADIUS = 3


def _reward(T, heat, cool, occ, elec, gas, hour, workday, K, month):
    """Flatten [N,H,(V)] -> per-step reward [N,H]."""
    N, H = elec.shape
    V = T.shape[-1]
    r = rm.step_reward(T.reshape(-1, V), heat.reshape(-1, V), cool.reshape(-1, V), occ.reshape(-1, V),
                       elec.reshape(-1), gas.reshape(-1), hour.reshape(-1), workday.reshape(-1),
                       np.repeat(K, H), None if month is None else month.reshape(-1))
    return r["reward"].reshape(N, H)


@torch.no_grad()
def evaluate_models(models: dict[str, Any], ds, batch_size=64, workers=0, exo_modes=("persistence",),
                    keep_zone_horizon=12) -> dict:
    """Returns {"meta": per-window arrays, "<model>|<exo>": per-window/per-k arrays}."""
    out: dict[str, dict[str, list]] = {}
    meta: dict[str, list] = {}
    for batch in loader(ds, batch_size, False, 0, workers):
        t, inp = batch["targets"], batch["inputs"]
        zm = batch["zone_mask"]
        B, H, Vm = t["zone_temp"].shape
        C = inp["zone_temp_obs"].shape[1]
        K = batch["info"]["K"].numpy()
        zmask = zm.numpy()[:, None, :].repeat(H, 1)
        heat, cool = t["zone_setpoints"][..., 0].numpy(), t["zone_setpoints"][..., 1].numpy()
        occ = t["zone_occ"].numpy()
        cal = t["calendar"].numpy()
        hour_true, wd_true, month_true = cal[..., 0], cal[..., 5].astype(bool), cal[..., 3]
        hour_n = inp["hour_next"][:, C - 1:].numpy()
        dow_n = inp["dow_next"][:, C - 1:].numpy()
        gate = np.zeros((B, H), bool)
        for b in range(B):
            sch = ds.feat[batch["info"]["scenario"][b]].schedule
            gate[b] = comfort_mode(hour_n[b], dow_n[b], sch)
        occ_deploy = zmask * gate[..., None]
        hour_deploy = np.floor(hour_n + 1e-6).astype(int) % 24
        wd_deploy = np.floor(dow_n + 1e-6) < 5
        e_true = t["energy"].numpy().astype(np.float64)
        T_true = t["zone_temp"].numpy().astype(np.float64)
        T_true_safe = np.where(zmask, T_true, 295.0)
        r_true = t["reward"].numpy()
        elec_t, gas_t = e_true[..., :3].sum(-1), e_true[..., 3]
        # meta + conditions (true values; used only to stratify)
        tot_occ = occ.sum(-1) > 0
        trans = np.zeros((B, H), bool)
        flips = np.zeros((B, H), bool)
        flips[:, 1:] = (tot_occ[:, 1:] != tot_occ[:, :-1]) | (gate[:, 1:] != gate[:, :-1])
        for k in range(H):
            lo, hi = max(0, k - TRANSITION_RADIUS), min(H, k + TRANSITION_RADIUS + 1)
            trans[:, k] = flips[:, lo:hi].any(1)
        dev = np.maximum(0, np.maximum(T_true - t["zone_setpoints"][..., 1].numpy(), t["zone_setpoints"][..., 0].numpy() - T_true))
        viol = ((dev > 0) & (occ > 0) & zmask).any(-1)
        recov_window = np.zeros((B, H), bool)
        for b in range(B):
            ms = ds.feat[batch["info"]["scenario"][b]].schedule["morning_start_hour"]
            recov_window[b] = (hour_n[b] >= ms - 1) & (hour_n[b] < ms + 2) & (dow_n[b] < 5)
        heating = gas_t > 0
        cold = t["oat_oracle"].numpy() < COLD_STEP_K
        for key, v in {"building": batch["info"]["building"], "scenario": batch["info"]["scenario"],
                       "episode_id": batch["info"]["episode_id"], "t0": batch["info"]["t0"], "K": K, "V": zm.sum(1).numpy(),
                       "occ_transition": trans, "comfort_violation": viol, "recovery_window": recov_window,
                       "heating_recovery": recov_window & heating, "heating_active": heating, "cold_step": cold,
                       "r_true": r_true, "e_true": e_true, "hws_true": t["hws"].numpy(),
                       "zone_mean_true": np.where(zmask, T_true, 0).sum(-1) / zmask.sum(-1), "oat_true": t["oat_oracle"].numpy(),
                       "zone_mean_obs_t0": (inp["zone_temp_obs"][:, C - 1].numpy() * zm.numpy()).sum(-1) / zm.numpy().sum(-1),
                       "r_true_deploy": _reward(T_true_safe, heat, cool, occ_deploy, elec_t, gas_t, hour_deploy, wd_deploy, K, None),
                       "r_true_exact": _reward(T_true_safe, heat, cool, occ, elec_t, gas_t, hour_true, wd_true, K, month_true),
                       }.items():
            meta.setdefault(key, []).append(np.asarray(v))
        for name, model in models.items():
            for exo in exo_modes:
                if exo == "oracle" and name != "gru":
                    continue
                p = rollout(model, batch, exo)
                Tp = p["zone_temp"].numpy().astype(np.float64)
                err = np.where(zmask, Tp - T_true, 0.0)
                nz = zmask.sum(-1)
                ep = p["energy_w"].numpy().astype(np.float64)
                Tp_safe = np.where(zmask, Tp, 295.0)
                elec_p, gas_p = ep[..., :3].sum(-1), ep[..., 3]
                rec = {
                    "zone_mae": np.abs(err).sum(-1) / nz, "zone_bias": err.sum(-1) / nz, "zone_mse": (err ** 2).sum(-1) / nz,
                    "zone_abs_err_h": np.abs(err[:, keep_zone_horizon - 1]),   # per-zone |err| at the reporting horizon
                    "zone_pred_h": Tp[:, keep_zone_horizon - 1], "zone_true_h": T_true[:, keep_zone_horizon - 1],
                    "hws_err": p["hws"].numpy() - t["hws"].numpy(),
                    "hws_pred": p["hws"].numpy(),
                    "e_pred": ep,
                    "r_exact_phys": _reward(Tp_safe, heat, cool, occ, elec_p, gas_p, hour_true, wd_true, K, month_true),
                    "r_deploy": _reward(Tp_safe, heat, cool, occ_deploy, elec_p, gas_p, hour_deploy, wd_deploy, K, None),
                    "zone_min_pred": np.where(zmask, Tp, np.inf).min(-1), "zone_max_pred": np.where(zmask, Tp, -np.inf).max(-1),
                    "zone_mean_pred": np.where(zmask, Tp, 0).sum(-1) / nz,
                }
                d = out.setdefault(f"{name}|{exo}", {})
                for k2, v in rec.items():
                    d.setdefault(k2, []).append(v)

    def cat(lst):
        if lst[0].ndim >= 2 and len({a.shape[-1] for a in lst}) > 1 and lst[0].ndim == 2:
            m = max(a.shape[-1] for a in lst)
            lst = [np.pad(a.astype(np.float64), ((0, 0), (0, m - a.shape[-1])), constant_values=np.nan) for a in lst]
        return np.concatenate(lst, 0)
    res = {"meta": {k: cat(v) for k, v in meta.items()}}
    for key, d in out.items():
        res[key] = {k: cat(v) for k, v in d.items()}
    return res


# ------------------------------------------------------------------ aggregation
def _stats_err(e):
    e = np.asarray(e, dtype=np.float64)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return None
    return {"n": int(e.size), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()),
            "p95_abs": float(np.percentile(np.abs(e), 95))}


def summarize(res: dict, model_key: str, horizons, sel: np.ndarray | None = None, step_sel: np.ndarray | None = None) -> dict:
    """Metrics by horizon for one model; sel: window mask [N]; step_sel: [N,H] mask."""
    r, m = res[model_key], res["meta"]
    N, H = r["zone_mae"].shape
    S = np.ones((N, H), bool) if step_sel is None else step_sel.copy()
    if sel is not None:
        S &= sel[:, None]
    out = {}
    for h in horizons:
        k = h - 1
        if k >= H:
            continue
        s = S[:, k]
        if not s.any():
            out[h] = None
            continue
        e_err = r["e_pred"][s, k] - m["e_true"][s, k]
        elec_err = e_err[:, :3].sum(-1)
        e_true = m["e_true"][s, k]
        o = {"windows": int(s.sum()),
             "zone_temp_c": {"mae": float(r["zone_mae"][s, k].mean()), "rmse": float(np.sqrt(r["zone_mse"][s, k].mean())),
                             "bias": float(r["zone_bias"][s, k].mean())},
             "hws_k": _stats_err(r["hws_err"][s, k]),
             "energy_kw": {name: {**_stats_err(e_err[:, c] / 1e3),
                                  "rel_mae": float(np.abs(e_err[:, c]).mean() / max(np.abs(e_true[:, c]).mean(), 1e-9)),
                                  "true_mean": float(e_true[:, c].mean() / 1e3)}
                           for c, name in enumerate(ENERGY_NAMES)},
             "electricity_total_kw": {**_stats_err(elec_err / 1e3),
                                      "rel_mae": float(np.abs(elec_err).mean() / max(np.abs(e_true[:, :3].sum(-1)).mean(), 1e-9))},
             "reward": {v: _stats_err(r[f"r_{v}"][s, k] - m["r_true"][s, k]) for v in ("exact_phys", "deploy")},
             "reward_true_mean": float(m["r_true"][s, k].mean()),
             }
        out[h] = o
    return out


def window_cum_reward_error(res, model_key, sel=None):
    """Error of the H-step SUM of reward along the rollout (what a planner compares)."""
    r, m = res[model_key], res["meta"]
    s = np.ones(len(m["t0"]), bool) if sel is None else sel
    out = {}
    for v in ("exact_phys", "deploy"):
        e = r[f"r_{v}"][s].sum(1) - m["r_true"][s].sum(1)
        out[v] = _stats_err(e)
    out["true_deploy"] = _stats_err(m["r_true_deploy"][s].sum(1) - m["r_true"][s].sum(1))
    return out
