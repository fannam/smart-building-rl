"""Losses, data setup and training for the dynamics models (one LOBO fold per call).

Loss: masked Huber (delta = 1 in normalized units) over every rollout step k = 0..H-1:
    L = wT * Huber((T_hat - T)/sd_dT) + wH * Huber((Hws_hat - Hws)/sd_dH) + wE * sum_c Huber((e_hat_c - e_c)/sd_c)
with energies in per-AHU-normalized units (W / (K * 80 kW), gas / (K * 205 kW)), so
building size and magnitude do not dominate. Huber: quadratic for typical errors,
linear for the rare large transients (morning recovery, burner switching) so they
do not dominate the gradient. All scales come from TRAIN buildings only.

Model selection: inner validation = the last 20 % of every TRAIN episode (time split
inside the training buildings). The held-out building is never used for training,
early stopping, normalization, or linear/persistence fitting.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from dataset import AlignedWindows, WindowSpec, load_manifest
from dynamics.features import PRIVILEGED_FOR_TRAINING, DynamicsWindows, collate
from dynamics.model import (GRUDynamics, LinearDynamics, Norm, PersistenceModel, energy_norm_target, fit_normalizer,
                            rollout)
from generalization import assert_provenance, split_rows

TASKS = ("temperature", "hws", "electricity", "gas")


def multitask_loss(pred, batch, norm: Norm, weights: dict[str, float]):
    t = batch["targets"]
    zm = batch["zone_mask"].unsqueeze(1).expand_as(t["zone_temp"]).to(torch.float32)
    eT = (pred["zone_temp"] - t["zone_temp"]) / norm.dTs
    lT = (F.huber_loss(eT, torch.zeros_like(eT), reduction="none") * zm).sum() / zm.sum().clamp_min(1)
    eH = (pred["hws"] - t["hws"]) / norm.dHs
    lH = F.huber_loss(eH, torch.zeros_like(eH))
    eE = (pred["energy_norm"] - energy_norm_target(batch)) / norm.es.to(pred["energy_norm"].device)
    per_c = F.huber_loss(eE, torch.zeros_like(eE), reduction="none").mean((0, 1))
    lE, lG = per_c[:3].sum(), per_c[3]
    total = weights["temperature"] * lT + weights["hws"] * lH + weights["electricity"] * lE + weights["gas"] * lG
    return total, {"temperature": float(lT), "hws": float(lH), "electricity": float(lE), "gas": float(lG),
                   "fan": float(per_c[0]), "cooling": float(per_c[1]), "pump": float(per_c[2]), "total": float(total)}


# ------------------------------------------------------------------ data
def fold_data(cfg: dict, fold: dict, cache: Path):
    root = Path(cfg["data_root"])
    rows = [r for r in load_manifest(root) if r["profile"] in set(cfg["profiles"])]
    train_rows, test_rows = split_rows(rows, fold)
    stats = fit_normalizer(root, train_rows)
    assert_provenance(stats, fold, train_rows)
    return root, train_rows, test_rows, stats


def make_windows(root, rows, C, H, cache, stride=1, t0_filter=None):
    spec = WindowSpec(horizon=H, context=C, privileged_targets=PRIVILEGED_FOR_TRAINING)
    aw = AlignedWindows.from_root(root, rows, spec, cache_dir=cache)
    keep = []
    for i, (e, t0) in enumerate(aw.index):
        T = aw.episodes[e]["T"]
        if t0 % stride == 0 and (t0_filter is None or t0_filter(t0, T, H)):
            keep.append(i)
    return DynamicsWindows(aw, keep)


def inner_split(H):
    tr = lambda t0, T, h: t0 + h + 2 <= int(0.8 * T)
    va = lambda t0, T, h: t0 >= int(0.8 * T)
    return tr, va


def loader(ds, batch_size, shuffle, seed, workers):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                                       num_workers=workers, generator=g, persistent_workers=workers > 0)


# ------------------------------------------------------------------ baselines (closed form / tables)
@torch.no_grad()
def fit_linear(model: LinearDynamics, ds, batch_size=256, ridge=1e-3, workers=0):
    C = ds.C
    XtX_z = XtY_z = XtX_b = XtY_h = XtY_e = None
    for batch in loader(ds, batch_size, False, 0, workers):
        inp, t = batch["inputs"], batch["targets"]
        zm = batch["zone_mask"]
        zt = inp["zone_temp_obs"]
        # Zone model, teacher forcing on HISTORY rows only (runtime-available past sensors):
        # step j: T_in = obs[s] (after s-2), T_prev = obs[s+1] (after s-1), target = obs[s+2] (after s)
        Xz_l, yz_l = [], []
        for j in range(C - 2):
            Xz_j, _ = model.design(batch, j, zt[:, j], inp["hws_obs"][:, j], inp["oat_obs"][:, j], T_prev=zt[:, j + 1])
            hist_ok = zm & inp["action_mask"][:, j:j + 1]
            Xz_l.append(Xz_j[hist_ok]); yz_l.append(((zt[:, j + 2] - zt[:, j]) / model.norm.dTs)[hist_ok])
        Xz = torch.cat(Xz_l).double()
        yz = torch.cat(yz_l).double().unsqueeze(-1)
        T_in, hws_in, oat = zt[:, C - 1], inp["hws_obs"][:, C - 1], inp["oat_obs"][:, C - 1]
        _, Xb = model.design(batch, C - 1, T_in, hws_in, oat)
        Xb = Xb.double()
        yh = ((t["hws"][:, 0] - hws_in) / model.norm.dHs).double().unsqueeze(-1)
        ye = energy_norm_target(batch)[:, 0].double()
        if XtX_z is None:
            XtX_z, XtY_z = Xz.T @ Xz, Xz.T @ yz
            XtX_b, XtY_h, XtY_e = Xb.T @ Xb, Xb.T @ yh, Xb.T @ ye
        else:
            XtX_z += Xz.T @ Xz; XtY_z += Xz.T @ yz
            XtX_b += Xb.T @ Xb; XtY_h += Xb.T @ yh; XtY_e += Xb.T @ ye
    def solve(A, b):
        lam = ridge * torch.trace(A) / A.shape[0]
        return torch.linalg.solve(A + lam * torch.eye(A.shape[0], dtype=A.dtype), b).float()
    model.Wz.copy_(solve(XtX_z, XtY_z)); model.Wh.copy_(solve(XtX_b, XtY_h)); model.We.copy_(solve(XtX_b, XtY_e))
    return model


@torch.no_grad()
def fit_persistence(model: PersistenceModel, ds, batch_size=256, workers=0):
    C = ds.C
    nb = model.table.shape[1]
    s = torch.zeros(24, nb, 4, dtype=torch.float64)
    n = torch.zeros(24, nb, 1, dtype=torch.float64)
    glob, gn = torch.zeros(4, dtype=torch.float64), 0
    for batch in loader(ds, batch_size, False, 0, workers):
        h, o = model.bins(batch, C - 1, batch["inputs"]["oat_obs"][:, C - 1])
        e = energy_norm_target(batch)[:, 0].double()
        s.index_put_((h, o), e, accumulate=True)
        n.index_put_((h, o), torch.ones(len(h), 1, dtype=torch.float64), accumulate=True)
        glob += e.sum(0); gn += len(h)
    table = torch.where(n > 0, s / n.clamp_min(1), glob / max(gn, 1))
    model.table.copy_(table.float())
    return model


# ------------------------------------------------------------------ device helpers
def resolve_device(tc: dict) -> torch.device:
    """training.device: 'cpu' (default, the Phase 2A protocol), 'cuda', or 'auto'."""
    want = tc.get("device", "cpu")
    if want == "auto":
        want = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(want)


def batch_to(batch, device):
    """Move every tensor of a collated batch to `device` (no-op on CPU)."""
    if device is None or device.type == "cpu":
        return batch
    def mv(x):
        if isinstance(x, torch.Tensor):
            return x.to(device, non_blocking=True)
        if isinstance(x, dict):
            return {k: mv(v) for k, v in x.items()}
        return x
    return mv(batch)


def cpu_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


# ------------------------------------------------------------------ GRU training
def evaluate_loss(model, dl, norm, weights, device=None):
    model.eval()
    agg, nb = {}, 0
    with torch.no_grad():
        for batch in dl:
            batch = batch_to(batch, device)
            pred = rollout(model, batch)
            _, parts = multitask_loss(pred, batch, norm, weights)
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
    return {k: v / max(nb, 1) for k, v in agg.items()}


def train_gru(cfg: dict, fold: dict, out_dir: Path, cache: Path, seed: int, log=print) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    tc, mc = cfg["training"], cfg["model"]
    root, train_rows, _, stats = fold_data(cfg, fold, cache)
    norm = Norm.from_stats(stats)
    C, H = tc["context"], tc["horizon_train"]
    tr_f, va_f = inner_split(H)
    ds_tr = make_windows(root, train_rows, C, H, cache, stride=tc["train_stride"], t0_filter=tr_f)
    ds_va = make_windows(root, train_rows, C, H, cache, stride=tc["val_stride"], t0_filter=va_f)
    if tc.get("max_train_windows"):
        rng = np.random.default_rng(seed)
        ds_tr.index = sorted(rng.choice(ds_tr.index, min(len(ds_tr.index), tc["max_train_windows"]), replace=False).tolist())
    device = resolve_device(tc)
    model = GRUDynamics(norm, mc["hidden_zone"], mc["hidden_bldg"], mc["head_hidden"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    dl_tr = loader(ds_tr, tc["batch_size"], True, seed, tc["workers"])
    dl_va = loader(ds_va, tc["batch_size"], False, seed, tc["workers"])
    w = cfg["loss_weights"]
    best, best_ep, patience, history = math.inf, -1, 0, []
    ckpt = out_dir / f"gru_seed{seed}.pt"
    t_start = time.time()
    total_steps = tc["epochs"] * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], total_steps=max(total_steps, 1), pct_start=0.1)
    for ep in range(tc["epochs"]):
        model.train()
        agg, nb, t_ep = {}, 0, time.time()
        for batch in dl_tr:
            batch = batch_to(batch, device)
            pred = rollout(model, batch)
            loss, parts = multitask_loss(pred, batch, norm, w)
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
        tr_loss = {k: v / nb for k, v in agg.items()}
        va_loss = evaluate_loss(model, dl_va, norm, w, device)
        rec = {"epoch": ep, "train": tr_loss, "val": va_loss, "epoch_s": time.time() - t_ep, "last_grad_norm": float(gn)}
        history.append(rec)
        log(f"[fold {fold['fold']} seed {seed}] ep {ep} train {tr_loss['total']:.4f} val {va_loss['total']:.4f} "
            f"(T {va_loss['temperature']:.3f} H {va_loss['hws']:.3f} E {va_loss['electricity']:.3f} G {va_loss['gas']:.3f}) "
            f"{rec['epoch_s']:.0f}s")
        if va_loss["total"] < best - 1e-4:
            best, best_ep, patience = va_loss["total"], ep, 0
            torch.save({"state_dict": cpu_state_dict(model), "device": str(device), "stats": stats.to_json(), "model_cfg": mc, "fold": fold,
                        "seed": seed, "epoch": ep}, ckpt)
        else:
            patience += 1
            # v1.1: never stop before min_epochs (one-cycle LR peaks early; stopping then is premature)
            if patience >= tc["patience"] and ep + 1 >= tc.get("min_epochs", 0):
                log(f"[fold {fold['fold']} seed {seed}] early stop at epoch {ep} (best {best_ep})")
                break
    info = {"seed": seed, "best_epoch": best_ep, "best_val_total": best, "epochs_run": len(history),
            "train_windows": len(ds_tr), "val_windows": len(ds_va), "train_s": time.time() - t_start,
            "params": sum(p.numel() for p in model.parameters()), "history": history, "checkpoint": str(ckpt),
            "device": str(device), "torch": torch.__version__}
    (out_dir / f"train_log_seed{seed}.json").write_text(json.dumps(info, indent=1))
    return info


def load_gru(ckpt_path: Path) -> GRUDynamics:
    from generalization import FittedStatistics
    ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    norm = Norm.from_stats(FittedStatistics.from_json(ck["stats"]))
    mc = ck["model_cfg"]
    m = GRUDynamics(norm, mc["hidden_zone"], mc["hidden_bldg"], mc["head_hidden"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m
