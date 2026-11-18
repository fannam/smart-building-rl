"""Phase 2A training entry point: GRU + linear + persistence baselines per LOBO fold.

    .venv/bin/python scripts/train_dynamics.py --config configs/dynamics_gru.json \
        --jobs 0:0 1:0 2:0 3:0 4:0 5:0 5:1 5:2 --parallel 8

Each job "fold:seed" writes results/phase2a/fold_<i>_<building>/:
    gru_seed<seed>.pt, train_log_seed<seed>.json, linear.pt / persistence.pt (seed 0 only),
    normalizer.json (train-only statistics with provenance), fold.json, train_<seed>.log
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def apply_overrides(cfg: dict, device, workers, threads, job_index: int = 0) -> dict:
    """CLI overrides for remote (e.g. Kaggle GPU) runs; defaults leave the config untouched.
    With several GPUs, jobs are assigned round-robin (cuda:<job_index % n>)."""
    tc = cfg["training"]
    if device:
        tc["device"] = device
        if device in ("cuda", "auto"):
            import torch
            n = torch.cuda.device_count()
            if n > 1:
                tc["device"] = f"cuda:{job_index % n}"
    if workers is not None:
        tc["workers"] = workers
    if threads is not None:
        tc["torch_threads"] = threads
    return cfg


def warm_cache(data_root: Path, cache: Path) -> None:
    """Unpack every episode once, before parallel jobs start (avoids concurrent half-written .npy)."""
    from dataset import load_manifest, unpack_episode
    for r in load_manifest(data_root):
        unpack_episode(data_root / r["path"], cache)


def fold_dir(out: Path, fold: dict) -> Path:
    return out / f"fold_{fold['fold']}_{fold['test_building']}"


def job(cfg_path: str, fold_idx: int, seed: int, out: str, cache: str, baselines_only: bool = False,
        overrides: tuple = (None, None, None), job_index: int = 0) -> dict:
    import os
    os.chdir(ROOT)
    import torch
    cfg = apply_overrides(json.loads(Path(cfg_path).read_text()), *overrides, job_index=job_index)
    torch.set_num_threads(cfg["training"]["torch_threads"])
    from dynamics.model import LinearDynamics, Norm, PersistenceModel
    from dynamics.train import fit_linear, fit_persistence, fold_data, make_windows, train_gru
    fold = json.loads(Path(cfg["folds_config"]).read_text())["folds"][fold_idx]
    d = fold_dir(Path(out), fold)
    d.mkdir(parents=True, exist_ok=True)
    logf = (d / (f"baselines.log" if baselines_only else f"train_{seed}.log")).open("w")

    def log(msg):
        logf.write(msg + "\n"); logf.flush()

    try:
        t0 = time.time()
        res = {"fold": fold, "seed": seed}
        if seed == cfg["seed"]:
            root, train_rows, test_rows, stats = fold_data(cfg, fold, Path(cache))
            (d / "normalizer.json").write_text(json.dumps(stats.to_json(), indent=1))
            (d / "fold.json").write_text(json.dumps({**fold, "train_episodes": [r["episode_id"] for r in train_rows],
                                                     "test_episodes": [r["episode_id"] for r in test_rows]}, indent=1))
            norm = Norm.from_stats(stats)
            C = cfg["training"]["context"]
            ds_fit = make_windows(root, train_rows, C, 1, Path(cache), stride=cfg["baselines"]["fit_stride"])
            lin = fit_linear(LinearDynamics(norm), ds_fit, ridge=cfg["baselines"]["linear_ridge"])
            torch.save({"state_dict": lin.state_dict(), "stats": stats.to_json(), "fit_windows": len(ds_fit)}, d / "linear.pt")
            per = fit_persistence(PersistenceModel(norm), ds_fit)
            torch.save({"state_dict": per.state_dict(), "stats": stats.to_json(), "fit_windows": len(ds_fit)}, d / "persistence.pt")
            log(f"baselines fitted on {len(ds_fit)} train windows in {time.time() - t0:.0f}s")
        if baselines_only:
            return {**res, "baselines_only": True, "wall_s": time.time() - t0}
        info = train_gru(cfg, fold, d, Path(cache), seed, log=log)
        res.update({k: v for k, v in info.items() if k != "history"}, wall_s=time.time() - t0)
        return res
    except Exception:
        log(traceback.format_exc())
        return {"fold": fold, "seed": seed, "error": traceback.format_exc()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dynamics_gru.json")
    ap.add_argument("--jobs", nargs="+", default=["0:0", "1:0", "2:0", "3:0", "4:0", "5:0"])
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--out", default="results/phase2a")
    ap.add_argument("--cache", default="scratch/phase1_5/cache")
    ap.add_argument("--baselines-only", action="store_true", help="refit linear/persistence only (no GRU training)")
    ap.add_argument("--device", default=None, help="override training.device: cpu | cuda | auto (default: config, i.e. cpu)")
    ap.add_argument("--workers", type=int, default=None, help="override training.workers (DataLoader workers per job)")
    ap.add_argument("--threads", type=int, default=None, help="override training.torch_threads")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / a.config).read_text())
    if not a.baselines_only:
        (out / "config_used.json").write_text(json.dumps({"config": cfg, "jobs": a.jobs, "argv": sys.argv,
                                                          "overrides": {"device": a.device, "workers": a.workers, "threads": a.threads}}, indent=1))
    jobs = [tuple(map(int, j.split(":"))) for j in a.jobs]
    warm_cache(ROOT / cfg["data_root"], ROOT / a.cache)
    ov = (a.device, a.workers, a.threads)
    results = []
    with ProcessPoolExecutor(max_workers=a.parallel) as pool:
        futs = [pool.submit(job, str(ROOT / a.config), f, s, str(out), str(ROOT / a.cache), a.baselines_only, ov, i)
                for i, (f, s) in enumerate(jobs)]
        for fu in as_completed(futs):
            r = fu.result()
            results.append(r)
            if "error" in r:
                print(f"FAIL fold {r['fold']['fold']} seed {r['seed']}\n{r['error']}", flush=True)
            elif r.get("baselines_only"):
                print(f"baselines refit fold {r['fold']['fold']} in {r['wall_s']:.0f}s", flush=True)
            else:
                print(f"done fold {r['fold']['fold']} ({r['fold']['test_building']}) seed {r['seed']}: best epoch "
                      f"{r['best_epoch']} val {r['best_val_total']:.4f} epochs {r['epochs_run']} "
                      f"train_windows {r['train_windows']} {r['wall_s'] / 60:.1f} min", flush=True)
    name = "baselines_summary.json" if a.baselines_only else "training_summary.json"
    (out / name).write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
