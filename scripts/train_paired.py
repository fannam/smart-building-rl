"""Phase 2C training: existing GRU / graph models with (a) ordinary state supervision on the new branch
data ('arms', data-only control) or (b) the same plus the paired counterfactual loss ('paired').

    L = L_state(Phase 1 windows) + w_arm * L_state(all branch arms as ordinary trajectories) + lambda * L_cf(paired)
    'arms'   : lambda = 0      (isolates the effect of more, better-excited data)
    'paired' : lambda > 0      (isolates the effect of the paired objective; same data, same schedule)
The Phase 2A protocol is reused unchanged (LOBO fold, train-only normalizer, inner time split, Huber state
loss, AdamW + one-cycle, grad clip, patience/min_epochs early stopping). Branch points come from the split
manifest: TRAIN/VAL points of the fold's 5 training buildings only; Test-A/Test-B are never touched.

Job spec  arch:objective:fold:seed:lambda   e.g.  gru:paired:5:0:0.3   graph:arms:1:0:0
Outputs   <out>/<arch>_<objective>_lam<lambda>/fold_<i>_<building>/{model_seed<s>.pt, train_log_seed<s>.json, train_<s>.log}
A job whose train_log exists is skipped (resume across sessions).

    python scripts/train_paired.py --config configs/phase2c/paired_training.json --jobs gru:paired:5:0:0.3 --device auto
"""
from __future__ import annotations

import argparse
import json
import os
import math
import sys

# Jupyter kernels (e.g. Kaggle) export MPLBACKEND=module://matplotlib_inline.backend_inline; subprocesses using
# another environment cannot import that backend, and the simulator imports matplotlib -> force a headless backend.
if os.environ.get("MPLBACKEND", "").startswith("module://"):
    os.environ["MPLBACKEND"] = "Agg"
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def job_dir(out: Path, arch, objective, lam, fold) -> Path:
    return out / f"{arch}_{objective}_lam{lam:g}" / f"fold_{fold['fold']}_{fold['test_building']}"


def parse(spec: str):
    arch, objective, fold, seed, lam = spec.split(":")
    assert arch in ("gru", "graph") and objective in ("arms", "paired"), spec
    lam = float(lam)
    assert (objective == "arms") == (lam == 0.0), f"{spec}: 'arms' requires lambda 0, 'paired' lambda > 0"
    return arch, objective, int(fold), int(seed), lam


def run_job(spec: str, cfg_path: str, out: str, overrides: tuple, job_index: int) -> dict:
    import os
    os.chdir(ROOT)
    import numpy as np
    import torch
    from counterfactual.paired_data import BranchGroups, branch_ids_for, collate_groups, load_split_manifest
    from dynamics.model import GRUDynamics, Norm, rollout
    from dynamics.paired import paired_loss
    from dynamics.train import (batch_to, cpu_state_dict, evaluate_loss, fold_data, inner_split, loader, make_windows,
                                multitask_loss, resolve_device)
    from graph.train import GraphAuto, build_graph_model
    from train_dynamics import apply_overrides
    arch, objective, fold_idx, seed, lam = parse(spec)
    pc = json.loads(Path(cfg_path).read_text())
    base = json.loads((ROOT / pc["base_configs"][arch]).read_text())
    base["data_root"] = pc["phase1_root"]
    base["training"].update(pc.get("training_overrides", {}))
    base = apply_overrides(base, *overrides, job_index=job_index)
    tc, mc, w = base["training"], base["model"], base["loss_weights"]
    torch.set_num_threads(tc["torch_threads"])
    fold = json.loads((ROOT / base["folds_config"]).read_text())["folds"][fold_idx]
    d = job_dir(Path(out), arch, objective, lam, fold)
    d.mkdir(parents=True, exist_ok=True)
    if (d / f"train_log_seed{seed}.json").exists():
        return {"spec": spec, "skipped": True}
    logf = (d / f"train_{seed}.log").open("w")
    def log(m):
        logf.write(m + "\n"); logf.flush()
    try:
        torch.manual_seed(seed); np.random.seed(seed)
        t_start = time.time()
        cache = Path(pc["cache"])
        root, train_rows, _, stats = fold_data(base, fold, cache)
        norm = Norm.from_stats(stats)
        C, H = tc["context"], tc["horizon_train"]
        tr_f, va_f = inner_split(H)
        ds_tr = make_windows(root, train_rows, C, H, cache, stride=tc["train_stride"], t0_filter=tr_f)
        ds_va = make_windows(root, train_rows, C, H, cache, stride=tc["val_stride"], t0_filter=va_f)
        if tc.get("max_train_windows"):
            rng = np.random.default_rng(seed)
            ds_tr.index = sorted(rng.choice(ds_tr.index, min(len(ds_tr.index), tc["max_train_windows"]), replace=False).tolist())
        man = load_split_manifest(pc["split_manifest"])
        tr_ids = branch_ids_for(man, ["train"], fold["train_buildings"])
        va_ids = branch_ids_for(man, ["val"], fold["train_buildings"])
        assert not set(tr_ids) & set(va_ids)
        assert all(man["assignments"][b]["building"] != fold["test_building"] for b in tr_ids + va_ids)
        bg_tr = BranchGroups(pc["cf_data"], tr_ids, pc["phase1_root"])
        bg_va = BranchGroups(pc["cf_data"], va_ids, pc["phase1_root"])
        if not len(bg_tr):
            raise RuntimeError(f"no TRAIN branch points for fold {fold_idx}")
        dstats = bg_tr.delta_stats()
        device = resolve_device(tc)
        if arch == "gru":
            model = GRUDynamics(norm, mc["hidden_zone"], mc["hidden_bldg"], mc["head_hidden"]).to(device)
            inner = model
        else:
            inner = build_graph_model(norm, mc).to(device)
            model = GraphAuto(inner, root)                     # attaches public topology per batch
        opt = torch.optim.AdamW(inner.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
        dl_tr = loader(ds_tr, tc["batch_size"], True, seed, tc["workers"])
        dl_va = loader(ds_va, tc["batch_size"], False, seed, tc["workers"])
        g = torch.Generator(); g.manual_seed(seed)
        gb = pc["branch_groups_per_step"]
        bl_tr = torch.utils.data.DataLoader(bg_tr, batch_size=gb, shuffle=True, collate_fn=collate_groups, generator=g, drop_last=False)
        bl_va = torch.utils.data.DataLoader(bg_va, batch_size=gb, shuffle=False, collate_fn=collate_groups)
        every = int(pc["branch_every"])
        w_arm, noop_w = float(pc["w_arm"]), float(pc["noop_weight"])
        total_steps = tc["epochs"] * len(dl_tr)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], total_steps=max(total_steps, 1), pct_start=0.1)

        def branch_losses(batch):
            pred = rollout(model, batch)
            ls, _ = multitask_loss(pred, batch, norm, w)
            lc, parts = paired_loss(pred, batch, dstats, w, noop_w)
            return ls, lc, parts

        def eval_branch():
            model.eval()
            acc, n = {"arm_state": 0.0, "cf_total": 0.0, "cf_zone": 0.0, "cf_tank": 0.0, "cf_energy": 0.0}, 0
            with torch.no_grad():
                for b in bl_va:
                    ls, lc, parts = branch_losses(batch_to(b, device))
                    acc["arm_state"] += float(ls); n += 1
                    for k in ("cf_total", "cf_zone", "cf_tank", "cf_energy"):
                        acc[k] += parts[k]
            return {k: v / max(n, 1) for k, v in acc.items()}

        best, best_ep, patience, history = math.inf, -1, 0, []
        ckpt = d / f"model_seed{seed}.pt"
        it_br = iter(bl_tr)
        for ep in range(tc["epochs"]):
            model.train()
            agg, nb, t_ep = {}, 0, time.time()
            for step, batch in enumerate(dl_tr):
                batch = batch_to(batch, device)
                loss, parts = multitask_loss(rollout(model, batch), batch, norm, w)
                rec = {"state": float(loss)}
                if step % every == 0:
                    try:
                        bb = next(it_br)
                    except StopIteration:
                        it_br = iter(bl_tr); bb = next(it_br)
                    ls, lc, cp = branch_losses(batch_to(bb, device))
                    loss = loss + w_arm * ls + (lam * lc if lam > 0 else 0.0)
                    rec.update({"arm_state": float(ls), **cp})
                opt.zero_grad()
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(inner.parameters(), tc["grad_clip"])
                opt.step(); sched.step()
                for k, v in rec.items():
                    agg.setdefault(k, []).append(v)
                nb += 1
            tr = {k: float(np.mean(v)) for k, v in agg.items()}
            va_state = evaluate_loss(model, dl_va, norm, w, device)
            va_br = eval_branch()
            sel = va_state["total"] + w_arm * va_br["arm_state"] + lam * va_br["cf_total"]
            history.append({"epoch": ep, "train": tr, "val_state": va_state, "val_branch": va_br, "val_select": sel,
                            "epoch_s": time.time() - t_ep, "last_grad_norm": float(gn)})
            log(f"[{spec}] ep {ep} sel {sel:.4f} | val state {va_state['total']:.4f} arm {va_br['arm_state']:.4f} "
                f"cf {va_br['cf_total']:.4f} (zone {va_br['cf_zone']:.3f} tank {va_br['cf_tank']:.3f} E {va_br['cf_energy']:.3f}) "
                f"{time.time() - t_ep:.0f}s")
            if sel < best - 1e-4:
                best, best_ep, patience = sel, ep, 0
                torch.save({"state_dict": cpu_state_dict(inner), "stats": stats.to_json(), "model_cfg": mc, "fold": fold,
                            "seed": seed, "epoch": ep, "kind": arch, "objective": objective, "lambda": lam,
                            "delta_stats": dstats, "device": str(device), "spec": spec}, ckpt)
            else:
                patience += 1
                if patience >= tc["patience"] and ep + 1 >= tc.get("min_epochs", 0):
                    log(f"[{spec}] early stop at epoch {ep} (best {best_ep})")
                    break
        info = {"spec": spec, "arch": arch, "objective": objective, "lambda": lam, "seed": seed, "fold": fold,
                "best_epoch": best_ep, "best_val_select": best, "epochs_run": len(history),
                "best_val_state": history[best_ep]["val_state"]["total"], "best_val_cf": history[best_ep]["val_branch"]["cf_total"],
                "train_windows": len(ds_tr), "train_branch_points": len(bg_tr), "val_branch_points": len(bg_va),
                "dropped_aborted_arms": bg_tr.dropped_aborted + bg_va.dropped_aborted, "delta_stats": dstats,
                "branch_ids_train": tr_ids, "branch_ids_val": va_ids, "split_digest": man.get("digest"),
                "params": sum(p.numel() for p in inner.parameters()), "train_s": time.time() - t_start,
                "device": str(device), "torch": torch.__version__, "config": pc, "history": history, "checkpoint": str(ckpt)}
        (d / f"train_log_seed{seed}.json").write_text(json.dumps(info, indent=1))
        return {k: v for k, v in info.items() if k not in ("history", "branch_ids_train", "branch_ids_val", "config")}
    except Exception:
        log(traceback.format_exc())
        return {"spec": spec, "error": traceback.format_exc()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2c/paired_training.json")
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--out", default="results/phase2c/training")
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    for j in a.jobs:
        parse(j)
    out = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pc = json.loads(Path(a.config).read_text())
    from train_dynamics import warm_cache
    warm_cache(Path(pc["phase1_root"]), Path(pc["cache"]))
    (out / f"jobs_{int(time.time())}.json").write_text(json.dumps({"argv": sys.argv, "jobs": a.jobs, "config": pc}, indent=1))
    ov = (a.device, a.workers, a.threads)
    res = []
    with ProcessPoolExecutor(max_workers=a.parallel) as pool:
        futs = [pool.submit(run_job, j, str(Path(a.config).resolve()), str(out), ov, i) for i, j in enumerate(a.jobs)]
        for fu in as_completed(futs):
            r = fu.result(); res.append(r)
            if "error" in r:
                print(f"FAIL {r['spec']}\n{r['error']}", flush=True)
            elif r.get("skipped"):
                print(f"skip {r['spec']} (already trained)", flush=True)
            else:
                print(f"done {r['spec']}: best epoch {r['best_epoch']}/{r['epochs_run']} val state {r['best_val_state']:.4f} "
                      f"val cf {r['best_val_cf']:.4f} branches {r['train_branch_points']}/{r['val_branch_points']} "
                      f"{r['train_s'] / 60:.1f} min on {r['device']}", flush=True)
    (out / f"summary_{int(time.time())}.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
