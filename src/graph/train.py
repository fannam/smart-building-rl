"""Graph-model data plumbing and training (same protocol as Phase 2A `dynamics.train.train_gru`).

Identical: LOBO fold, train-only normalizer, inner time split (last 20 % of every TRAIN episode),
windows (context 12, horizon 24, stride 4), multitask Huber loss and weights, AdamW + one-cycle,
gradient clipping, early stopping with patience 3 and min_epochs 6, 12 epochs max.
Only the model class and the batch's graph tensors differ.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from dynamics.features import collate
from dynamics.model import Norm, rollout
from dynamics.train import batch_to, cpu_state_dict, fold_data, inner_split, make_windows, multitask_loss, resolve_device
from graph.model import GraphDynamics
from graph.topology import build_topology, load_zone_map


class TopologyCache:
    def __init__(self, data_root: Path):
        self.root = Path(data_root)
        self._t = {}

    def __call__(self, scenario: str, meta) -> dict:
        b = scenario.split("/")[0]
        if b not in self._t:          # topology is public metadata, identical across a building's scenarios
            self._t[b] = build_topology(meta, load_zone_map(self.root, scenario))
        return self._t[b]


def graph_tensors(topos: list[dict], Vm: int) -> dict:
    src, dst, w = [], [], []
    for b, t in enumerate(topos):
        src.append(torch.as_tensor(t["zz_src"], dtype=torch.long) + b * Vm)
        dst.append(torch.as_tensor(t["zz_dst"], dtype=torch.long) + b * Vm)
        w.append(torch.as_tensor(t["zz_w"], dtype=torch.float32))
    return {"zz_src": torch.cat(src), "zz_dst": torch.cat(dst), "zz_w": torch.cat(w)}


class GraphWindows:
    """DynamicsWindows + per-item topology (kept outside 'inputs' so the base collate is unchanged)."""

    def __init__(self, dyn_windows, topo_cache: TopologyCache):
        self.dw = dyn_windows
        self.topo = topo_cache
        self.C, self.H = dyn_windows.C, dyn_windows.H

    def __len__(self):
        return len(self.dw)

    def __getitem__(self, i):
        it = self.dw[i]
        sc = it["info"]["scenario"]
        it["topo"] = self.topo(sc, self.dw.aw.metas[sc])
        return it


def graph_collate(items):
    batch = collate(items)
    batch["graph"] = graph_tensors([it["topo"] for it in items], batch["zone_mask"].shape[1])
    return batch


def gloader(ds, batch_size, shuffle, seed, workers):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=graph_collate,
                                       num_workers=workers, generator=g, persistent_workers=workers > 0)


def build_graph_model(norm: Norm, mc: dict) -> GraphDynamics:
    return GraphDynamics(norm, hidden=mc["hidden"], head_hidden=mc["head_hidden"],
                         use_zone_adjacency=mc.get("use_zone_adjacency", True))


def evaluate_loss(model, dl, norm, weights, device=None):
    model.eval()
    agg, nb = {}, 0
    with torch.no_grad():
        for batch in dl:
            batch = batch_to(batch, device)
            _, parts = multitask_loss(rollout(model, batch), batch, norm, weights)
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
    return {k: v / max(nb, 1) for k, v in agg.items()}


def train_graph(cfg: dict, fold: dict, out_dir: Path, cache: Path, seed: int, log=print) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    tc, mc = cfg["training"], cfg["model"]
    root, train_rows, _, stats = fold_data(cfg, fold, cache)
    norm = Norm.from_stats(stats)
    C, H = tc["context"], tc["horizon_train"]
    tr_f, va_f = inner_split(H)
    topo = TopologyCache(root)
    ds_tr = GraphWindows(make_windows(root, train_rows, C, H, cache, stride=tc["train_stride"], t0_filter=tr_f), topo)
    ds_va = GraphWindows(make_windows(root, train_rows, C, H, cache, stride=tc["val_stride"], t0_filter=va_f), topo)
    device = resolve_device(tc)
    if tc.get("max_train_windows"):
        rng = np.random.default_rng(seed)
        ds_tr.dw.index = sorted(rng.choice(ds_tr.dw.index, min(len(ds_tr.dw.index), tc["max_train_windows"]), replace=False).tolist())
    model = build_graph_model(norm, mc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    dl_tr = gloader(ds_tr, tc["batch_size"], True, seed, tc["workers"])
    dl_va = gloader(ds_va, tc["batch_size"], False, seed, tc["workers"])
    w = cfg["loss_weights"]
    best, best_ep, patience, history = math.inf, -1, 0, []
    ckpt = out_dir / f"graph_seed{seed}.pt"
    t_start = time.time()
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], total_steps=max(tc["epochs"] * len(dl_tr), 1), pct_start=0.1)
    for ep in range(tc["epochs"]):
        model.train()
        agg, nb, t_ep = {}, 0, time.time()
        for batch in dl_tr:
            batch = batch_to(batch, device)
            loss, parts = multitask_loss(rollout(model, batch), batch, norm, w)
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
        log(f"[graph fold {fold['fold']} seed {seed}] ep {ep} train {tr_loss['total']:.4f} val {va_loss['total']:.4f} "
            f"(T {va_loss['temperature']:.3f} H {va_loss['hws']:.3f} E {va_loss['electricity']:.3f} G {va_loss['gas']:.3f}) "
            f"{rec['epoch_s']:.0f}s")
        if va_loss["total"] < best - 1e-4:
            best, best_ep, patience = va_loss["total"], ep, 0
            torch.save({"state_dict": cpu_state_dict(model), "device": str(device), "stats": stats.to_json(), "model_cfg": mc, "fold": fold,
                        "seed": seed, "epoch": ep, "kind": "graph"}, ckpt)
        else:
            patience += 1
            if patience >= tc["patience"] and ep + 1 >= tc.get("min_epochs", 0):
                log(f"[graph fold {fold['fold']} seed {seed}] early stop at epoch {ep} (best {best_ep})")
                break
    info = {"seed": seed, "best_epoch": best_ep, "best_val_total": best, "epochs_run": len(history),
            "train_windows": len(ds_tr), "val_windows": len(ds_va), "train_s": time.time() - t_start,
            "params": sum(p.numel() for p in model.parameters()), "history": history, "checkpoint": str(ckpt),
            "device": str(device), "torch": torch.__version__}
    (out_dir / f"train_log_graph_seed{seed}.json").write_text(json.dumps(info, indent=1))
    return info


def load_graph(ckpt_path: Path) -> GraphDynamics:
    from generalization import FittedStatistics
    ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    m = build_graph_model(Norm.from_stats(FittedStatistics.from_json(ck["stats"])), ck["model_cfg"])
    m.load_state_dict(ck["state_dict"])
    return m.eval()


class GraphPredictor(torch.nn.Module):
    """Wraps a graph model for callers that build batches with the base `collate` (single building):
    attaches the topology, then runs the standard rollout."""

    def __init__(self, model: GraphDynamics, topology: dict):
        super().__init__()
        self.model, self.topology = model, topology

    def predict(self, batch, exo_mode="persistence"):
        B, Vm = batch["zone_mask"].shape
        dev = batch["zone_mask"].device
        batch = {**batch, "graph": {k: v.to(dev) for k, v in graph_tensors([self.topology] * B, Vm).items()}}
        return rollout(self.model, batch, exo_mode)


class GraphAuto(torch.nn.Module):
    """Adapter so code that builds batches with the base `collate` (the Phase 2A evaluator, the
    counterfactual evaluator) can drive a graph model through the standard `rollout`: on
    init_state it attaches the public topology of each item's building to the batch."""
    kind = "graph"

    def __init__(self, model: GraphDynamics, data_root: Path):
        super().__init__()
        self.model = model
        self.root = Path(data_root)
        self.topo = TopologyCache(self.root)
        self.norm = model.norm

    def _topology(self, scenario: str) -> dict:
        from dataset import load_scenario_meta
        return self.topo(scenario, load_scenario_meta(self.root, scenario))

    def init_state(self, batch):
        Vm = batch["zone_mask"].shape[1]
        dev = batch["zone_mask"].device
        batch["graph"] = {k: v.to(dev) for k, v in graph_tensors([self._topology(s) for s in batch["info"]["scenario"]], Vm).items()}
        return self.model.init_state(batch)

    def step(self, state, batch, j, T_in, hws_in, oat):
        return self.model.step(state, batch, j, T_in, hws_in, oat)
