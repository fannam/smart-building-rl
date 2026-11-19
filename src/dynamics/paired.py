"""Paired counterfactual response loss (Phase 2C).

For a branch group the nominal and every intervention arm are rolled out AUTOREGRESSIVELY from the same
runtime context with the same parameters (they are rows of one batch through the unchanged
`dynamics.model.rollout`; only the future action rows differ). Then, per arm r with nominal n(r):

    dhat_r = yhat_r - yhat_n(r)          d_r = y_r - y_n(r)     (simulator, common random numbers)
    L_cf   = sum_targets w_target * mean_{arms, all H steps} Huber((dhat_r - d_r) / sd_target)

over the whole 3 h trajectory (every horizon 1..H contributes), with
    zone temperature  per zone (padded zones masked), sd from TRAIN paired responses
    tank temperature
    energy            fan / cooling / pump / gas in per-AHU-normalized units, per-component sd
No-op arms (no effective action change) are kept as a consistency term with weight `noop_weight`
(their target is 0 up to simulator determinism), so they cannot dominate the loss.
"""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from dynamics.model import energy_norm_target


def paired_loss(pred: Mapping[str, torch.Tensor], batch: Mapping, dstats: Mapping, weights: Mapping[str, float],
                noop_weight: float = 0.1) -> tuple[torch.Tensor, dict]:
    pr = batch["pair"]
    dev = pred["zone_temp"].device
    nom = pr["nominal_row"].to(dev)
    arm = ~pr["is_nominal"].to(dev)
    if not bool(arm.any()):
        z = pred["zone_temp"].sum() * 0.0
        return z, {"cf_zone": 0.0, "cf_tank": 0.0, "cf_energy": 0.0, "cf_total": 0.0}
    w = torch.where(pr["noop"].to(dev), torch.tensor(noop_weight, device=dev), torch.tensor(1.0, device=dev))[arm]   # [R]
    t = batch["targets"]
    # zone [R,H,V]
    dz_hat = pred["zone_temp"][arm] - pred["zone_temp"][nom[arm]]
    dz = t["zone_temp"][arm] - t["zone_temp"][nom[arm]]
    zm = batch["zone_mask"][arm].unsqueeze(1).expand_as(dz).to(dz.dtype)
    ez = F.huber_loss((dz_hat - dz) / dstats["zone"], torch.zeros_like(dz), reduction="none") * zm
    lz = (ez.sum((1, 2)) / zm.sum((1, 2)).clamp_min(1) * w).sum() / w.sum()
    # tank [R,H]
    dh_hat = pred["hws"][arm] - pred["hws"][nom[arm]]
    dh = t["hws"][arm] - t["hws"][nom[arm]]
    lh = (F.huber_loss((dh_hat - dh) / dstats["tank"], torch.zeros_like(dh), reduction="none").mean(1) * w).sum() / w.sum()
    # energy [R,H,4] in per-AHU-normalized units
    en_t = energy_norm_target(batch)
    de_hat = pred["energy_norm"][arm] - pred["energy_norm"][nom[arm]]
    de = en_t[arm] - en_t[nom[arm]]
    sd = torch.as_tensor(dstats["energy"], dtype=de.dtype, device=dev)
    le_c = (F.huber_loss((de_hat - de) / sd, torch.zeros_like(de), reduction="none").mean(1) * w.unsqueeze(-1)).sum(0) / w.sum()
    le = weights["electricity"] * le_c[:3].sum() + weights["gas"] * le_c[3]
    total = weights["temperature"] * lz + weights["hws"] * lh + le
    return total, {"cf_zone": float(lz), "cf_tank": float(lh), "cf_energy": float(le), "cf_total": float(total)}
