"""Ensemble aggregation, uncertainty calibration and OOD scores (Phase 2B Workstream D).

All thresholds / calibration constants are fitted on TRAIN-building data only (inner-validation
windows, i.e. the last 20 % of every training episode), never on the held-out building.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from dynamics.model import rollout, step_features


@torch.no_grad()
def ensemble_predict(members: Sequence[Any], batch, exo_mode="persistence") -> dict[str, np.ndarray]:
    """Per-member rollouts -> mean and std. zone [B,H,V], hws [B,H], energy [B,H,4]."""
    outs = [rollout(m, batch, exo_mode) for m in members]
    z = torch.stack([o["zone_temp"] for o in outs]).double().numpy()        # [M,B,H,V]
    h = torch.stack([o["hws"] for o in outs]).double().numpy()              # [M,B,H]
    e = torch.stack([o["energy_w"] for o in outs]).double().numpy()         # [M,B,H,4]
    ddof = 1 if len(members) > 1 else 0
    return {"zone_mean": z.mean(0), "zone_std": z.std(0, ddof=ddof), "hws_mean": h.mean(0), "hws_std": h.std(0, ddof=ddof),
            "energy_mean": e.mean(0), "energy_std": e.std(0, ddof=ddof), "members": len(members),
            "zone_m0": z[0], "hws_m0": h[0], "energy_m0": e[0]}


def window_errors(pred_zone, pred_hws, pred_e, batch) -> dict[str, np.ndarray]:
    """Per window and step: zone MAE over zones [B,H], |tank err| [B,H], |elec err|, |gas err| [B,H] (W)."""
    t = batch["targets"]
    zm = batch["zone_mask"].numpy()[:, None, :]
    nz = zm.sum(-1)
    ze = (np.abs(pred_zone - t["zone_temp"].numpy()) * zm).sum(-1) / nz
    he = np.abs(pred_hws - t["hws"].numpy())
    et = t["energy"].numpy().astype(np.float64)
    ee = np.abs(pred_e[..., :3].sum(-1) - et[..., :3].sum(-1))
    ge = np.abs(pred_e[..., 3] - et[..., 3])
    return {"zone": ze, "tank": he, "elec": ee, "gas": ge}


def window_spread(ens: Mapping[str, np.ndarray], batch) -> dict[str, np.ndarray]:
    zm = batch["zone_mask"].numpy()[:, None, :]
    return {"zone": (ens["zone_std"] * zm).sum(-1) / zm.sum(-1), "tank": ens["hws_std"],
            "elec": np.sqrt((ens["energy_std"][..., :3] ** 2).sum(-1)), "gas": ens["energy_std"][..., 3]}


@torch.no_grad()
def decision_latent(model, batch) -> np.ndarray:
    """Phase 2A GRU building latent h_b (and mean zone latent) after the warm-up + decision step
    (true observations only, exactly as in rollout). [B, hb + hz]."""
    inp = batch["inputs"]
    C = inp["zone_temp_obs"].shape[1]
    state = model.init_state(batch)
    for j in range(C):
        state, _ = model.step(state, batch, j, inp["zone_temp_obs"][:, j], inp["hws_obs"][:, j], inp["oat_obs"][:, j])
    hz, hb = state
    zm = batch["zone_mask"].unsqueeze(-1).float()
    return torch.cat([hb, (hz * zm).sum(1) / zm.sum(1)], -1).double().numpy()


def state_features(batch) -> np.ndarray:
    """Runtime state/exogenous features at the decision step (o_t0 and clock): zone mean/min/max T,
    HWS, outside air, hod sin/cos, comfort mode."""
    inp = batch["inputs"]
    C = inp["zone_temp_obs"].shape[1]
    zm = batch["zone_mask"]
    T = inp["zone_temp_obs"][:, C - 1].double()
    mean = (T * zm).sum(1) / zm.sum(1)
    tmin = torch.where(zm, T, torch.full_like(T, 1e9)).min(1).values
    tmax = torch.where(zm, T, torch.full_like(T, -1e9)).max(1).values
    exo = inp["exo"][:, C - 1].double()
    return torch.stack([mean, tmin, tmax, inp["hws_obs"][:, C - 1].double(), inp["oat_obs"][:, C - 1].double(),
                        exo[:, 0], exo[:, 1], exo[:, 4]], -1).numpy()


class Mahalanobis:
    def fit(self, X: np.ndarray, shrink: float = 1e-3):
        self.mu = X.mean(0)
        cov = np.cov(X.T)
        cov = cov + shrink * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        self.inv = np.linalg.inv(cov)
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        d = X - self.mu
        return np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", d, self.inv, d), 0.0))


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    from scipy.stats import rankdata
    pos = positive.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def score_vs_error(score: np.ndarray, err: np.ndarray, threshold: float, high_err: float) -> dict[str, float]:
    """Usefulness of a (higher = less reliable) score for an error criterion.
    high_err: absolute error defining an 'unreliable' window (fixed from train-building validation)."""
    from scipy.stats import spearmanr
    ok = np.isfinite(score) & np.isfinite(err)
    s, e = score[ok], err[ok]
    bad = e > high_err
    flag = s > threshold
    dec = np.quantile(s, [0.1, 0.9])
    out = {"n": int(ok.sum()), "spearman": float(spearmanr(s, e).statistic) if len(s) > 2 else float("nan"),
           "auroc_unreliable": auroc(s, bad), "frac_unreliable": float(bad.mean()), "frac_flagged": float(flag.mean()),
           "mae_lowest_decile": float(e[s <= dec[0]].mean()), "mae_highest_decile": float(e[s >= dec[1]].mean()),
           "mae_all": float(e.mean())}
    tp = (flag & bad).sum()
    out["precision"] = float(tp / flag.sum()) if flag.any() else float("nan")
    out["recall"] = float(tp / bad.sum()) if bad.any() else float("nan")
    out["false_positive_rate"] = float((flag & ~bad).sum() / max((~bad).sum(), 1))
    out["false_negative_rate"] = float((~flag & bad).sum() / max(bad.sum(), 1))
    return out


def calibration_bins(spread: np.ndarray, err: np.ndarray, scale: float, n_bins: int = 10) -> dict[str, Any]:
    """Mean |error| vs mean (scaled) spread per spread-decile bin; Gaussian 1/2-sigma coverage of |err|."""
    q = np.quantile(spread, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(q, spread, side="right") - 1, 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            bins.append({"spread_mean": float(spread[m].mean() * scale), "abs_err_mean": float(err[m].mean()),
                         "rmse": float(np.sqrt((err[m] ** 2).mean())), "n": int(m.sum())})
    sig = np.maximum(spread * scale, 1e-12)
    return {"scale": float(scale), "bins": bins,
            "coverage_1sigma": float(np.mean(err <= sig)), "coverage_2sigma": float(np.mean(err <= 2 * sig)),
            "nominal_1sigma": 0.683, "nominal_2sigma": 0.954}
