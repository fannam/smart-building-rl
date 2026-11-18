"""Model-side counterfactual evaluation: predicted vs simulated paired responses.

For a branch point t_b and arm a, the model input is EXACTLY what a deployed controller
would have: o_{t_b-C+1..t_b}, the executed action history a_{t_b-C..t_b-1}, and the
candidate future action sequence of the arm (effective actions), plus schedule/clock
features computed from o_{t_b}. Outside air is held at o_{t_b} ('persistence', deployable).
No future observation, privileged state or occupancy enters the model.

    Delta_sim[a]   = y_sim(arm a)   - y_sim(nominal)       (paired, common random numbers)
    Delta_model[a] = y_model(arm a) - y_model(nominal)
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from dynamics.features import WindowFeaturizer, collate
from dynamics.model import rollout
from effective_action import ActionRepresentation

QUANTITIES = ("zone_temp", "tank", "fan_w", "cooling_w", "pump_w", "gas_w")
# Thresholds below which a simulated response is treated as "no response" (direction undefined).
THRESH = {"zone_temp": 0.02, "tank": 0.05, "fan_w": 50.0, "cooling_w": 200.0, "pump_w": 20.0, "gas_w": 500.0}


def build_items(sim: Mapping[str, np.ndarray], meta: Mapping[str, Any], p: int) -> list[dict]:
    """One featurized item per arm of branch point p (inputs + oracle-OAT target only)."""
    feat = WindowFeaturizer(meta)
    rep = ActionRepresentation(meta)
    H = int(sim["H"])
    ctx_ac, ctx_ad = sim["ctx_ac"][p], sim["ctx_ad"][p]
    runtime = {"obs": sim["ctx_obs"][p], "effective_action_history": rep.effective(ctx_ac, ctx_ad).effective_native,
               "action_history_discrete": ctx_ad, "action_history_mask": np.ones(len(ctx_ac), bool)}
    oat_oracle = feat._native(sim["oat_future_obs"][p], feat.oat_idx).astype(np.float32)
    items = []
    for a in range(sim["arm_ac"].shape[1]):
        e = rep.effective(sim["arm_ac"][p, a], sim["arm_ad"][p, a])
        acts = {"effective_native": e.effective_native, "discrete": sim["arm_ad"][p, a]}
        items.append({"inputs": feat.runtime_part(runtime, acts, H), "targets": {"oat_oracle": oat_oracle},
                      "info": {"K": feat.K, "V": feat.V, "arm": a, "scenario": str(sim["scenario"])}})
    return items


def sim_quantities(sim, p) -> dict[str, np.ndarray]:
    e = sim["energy"][p]
    return {"zone_temp": sim["zone_temp"][p].astype(np.float64), "tank": sim["tank"][p].astype(np.float64),
            "fan_w": e[..., 0].astype(np.float64), "cooling_w": e[..., 1].astype(np.float64),
            "pump_w": e[..., 2].astype(np.float64), "gas_w": e[..., 3].astype(np.float64)}


@torch.no_grad()
def model_quantities(model, items, exo_mode="persistence") -> dict[str, np.ndarray]:
    batch = collate(items)
    out = model.predict(batch, exo_mode) if hasattr(model, "predict") else rollout(model, batch, exo_mode)
    V = items[0]["info"]["V"]
    ew = out["energy_w"].double().numpy()
    return {"zone_temp": out["zone_temp"][..., :V].double().numpy(), "tank": out["hws"].double().numpy(),
            "fan_w": ew[..., 0], "cooling_w": ew[..., 1], "pump_w": ew[..., 2], "gas_w": ew[..., 3]}


def response_metrics(ds: np.ndarray, dm: np.ndarray, thr: float) -> dict[str, float]:
    """ds, dm: [H] or [H, V] paired responses (arm - nominal). Pools zones for zone temperature."""
    ds2 = ds.reshape(ds.shape[0], -1)
    dm2 = dm.reshape(dm.shape[0], -1)
    H = ds2.shape[0]
    final_s, final_m = ds2[-1], dm2[-1]
    cum_s, cum_m = ds2.mean(0), dm2.mean(0)                 # horizon-averaged response per element
    peak_s = np.abs(ds2).max(0)
    resp = peak_s > thr
    out = {"n_elem": int(ds2.shape[1]), "n_responsive": int(resp.sum()),
           "sim_mean_cum": float(cum_s.mean()), "model_mean_cum": float(cum_m.mean()),
           "sim_mean_final": float(final_s.mean()), "model_mean_final": float(final_m.mean()),
           "cum_abs_err": float(np.abs(cum_m - cum_s).mean()), "final_abs_err": float(np.abs(final_m - final_s).mean()),
           "path_mae": float(np.abs(dm2 - ds2).mean())}
    if resp.any():
        s, m = cum_s[resp], cum_m[resp]
        out["sign_agree"] = float(np.mean(np.sign(m) == np.sign(s)))
        out["sign_wrong"] = float(np.mean(np.sign(m) == -np.sign(s)))
        out["rel_err_cum"] = float(np.median(np.abs(m - s) / np.maximum(np.abs(s), thr)))
        out["gain"] = float(np.dot(m, s) / max(np.dot(s, s), 1e-12))      # least-squares response gain (1 = exact)
        out["dot_ms"], out["dot_ss"] = float(np.dot(m, s)), float(np.dot(s, s))   # for pooled gains
        out["sum_abs_s"], out["sum_abs_err"] = float(np.abs(s).sum()), float(np.abs(m - s).sum())
        # latency: first step reaching half of the element's peak |response|
        lat_s = np.argmax(np.abs(ds2[:, resp]) >= 0.5 * peak_s[resp], 0)
        pm = np.abs(dm2[:, resp]).max(0)
        lat_m = np.where(pm > thr, np.argmax(np.abs(dm2[:, resp]) >= 0.5 * np.maximum(pm, 1e-12), 0), H)
        out["latency_sim"] = float(np.median(lat_s)); out["latency_model"] = float(np.median(lat_m))
    nresp = ~resp
    if nresp.any():
        out["spurious_rate"] = float(np.mean(np.abs(dm2[:, nresp]).max(0) > thr))
    return out


def evaluate_scenario(models: Mapping[str, Any], sim, meta, exo_modes=("persistence",)) -> list[dict]:
    rows = []
    names = [str(x) for x in sim["arm_names"]]
    import json
    ivs = {d["name"]: d for d in json.loads(str(sim["interventions_json"]))}
    for p in range(len(sim["t_b"])):
        items = build_items(sim, meta, p)
        sq = sim_quantities(sim, p)
        preds = {(mn, ex): model_quantities(m, items, ex) for mn, m in models.items() for ex in exo_modes}
        ctx = {"t_b": int(sim["t_b"][p]), "stratum": str(sim["stratum"][p]), "oat_c": float(sim["oat_c"][p]),
               "oat_band": str(sim["oat_band"][p]), "heating": bool(sim["heating"][p]), "cooling": bool(sim["cooling"][p]),
               "ahu_on": bool(sim["ahu_on"][p]), "occupied": bool(sim["occupied"][p]), "hour": float(sim["hour"][p])}
        # nominal-arm absolute accuracy (observational sanity check)
        for (mn, ex), pq in preds.items():
            for q in QUANTITIES:
                rows.append({**ctx, "model": mn, "exo": ex, "arm": "nominal", "quantity": q, "kind": "absolute",
                             "abs_err_mean": float(np.abs(pq[q][0] - sq[q][0]).mean())})
        eff = sim["eff_delta"][p]
        cmd = sim["cmd_delta"][p]
        for a in range(1, len(names)):
            iv = ivs[names[a]]
            info = {**ctx, "arm": names[a], "family": iv["family"], "value": iv["value"], "duration": iv["duration"],
                    "magnitude": iv["magnitude"], "cmd_change": float(np.abs(cmd[a]).max()),
                    "eff_change": float(np.abs(eff[a]).max()), "effective_noop": bool(np.abs(eff[a]).max() < 1e-6)}
            for q in QUANTITIES:
                ds = sq[q][a] - sq[q][0]
                for (mn, ex), pq in preds.items():
                    dm = pq[q][a] - pq[q][0]
                    rows.append({**info, "model": mn, "exo": ex, "quantity": q, "kind": "response",
                                 **response_metrics(ds, dm, THRESH[q])})
    return rows
