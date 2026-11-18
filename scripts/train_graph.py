"""Phase 2B: train the graph-aware dynamics model on LOBO folds (same protocol as Phase 2A).

    .venv/bin/python scripts/train_graph.py --jobs 0:0 1:0 2:0 3:0 4:0 5:0 --parallel 6

Each job "fold:seed" writes results/phase2b/graph_dynamics/fold_<i>_<building>/graph_seed<seed>.pt,
train_log_graph_seed<seed>.json, train_graph_<seed>.log, topology.json
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
sys.path.insert(0, str(ROOT / "scripts"))


def job(cfg_path: str, fold_idx: int, seed: int, out: str, cache: str, overrides: tuple = (None, None, None),
        job_index: int = 0) -> dict:
    import os
    os.chdir(ROOT)
    import torch
    from train_dynamics import apply_overrides
    cfg = apply_overrides(json.loads(Path(cfg_path).read_text()), *overrides, job_index=job_index)
    torch.set_num_threads(cfg["training"]["torch_threads"])
    from graph.topology import build_topology, load_zone_map
    from graph.train import train_graph
    from dataset import load_scenario_meta
    fold = json.loads(Path(cfg["folds_config"]).read_text())["folds"][fold_idx]
    d = Path(out) / f"fold_{fold['fold']}_{fold['test_building']}"
    d.mkdir(parents=True, exist_ok=True)
    logf = (d / f"train_graph_{seed}.log").open("w")

    def log(msg):
        logf.write(msg + "\n"); logf.flush()
    try:
        t0 = time.time()
        topo = {}
        for b in fold["train_buildings"] + [fold["test_building"]]:
            sc = next(p.name for p in (Path(cfg["data_root"]) / "scenarios").glob(f"{b}__*")).replace("__", "/")
            t = build_topology(load_scenario_meta(Path(cfg["data_root"]), sc), load_zone_map(Path(cfg["data_root"]), sc))
            topo[b] = {k: t[k] for k in ("V", "K", "n_zz_edges", "adjacency_available", "adjacency_reason")}
        (d / "topology.json").write_text(json.dumps(topo, indent=1))
        info = train_graph(cfg, fold, d, Path(cache), seed, log=log)
        return {"fold": fold, "seed": seed, **{k: v for k, v in info.items() if k != "history"}, "wall_s": time.time() - t0}
    except Exception:
        log(traceback.format_exc())
        return {"fold": fold, "seed": seed, "error": traceback.format_exc()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/graph_dynamics.json")
    ap.add_argument("--jobs", nargs="+", default=["0:0", "1:0", "2:0", "3:0", "4:0", "5:0"])
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--out", default="results/phase2b/graph_dynamics")
    ap.add_argument("--cache", default="scratch/phase1_5/cache")
    ap.add_argument("--device", default=None, help="override training.device: cpu | cuda | auto")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / a.config).read_text())
    (out / f"config_used_{int(time.time())}.json").write_text(json.dumps({"config": cfg, "jobs": a.jobs, "argv": sys.argv,
                                                                            "overrides": {"device": a.device, "workers": a.workers, "threads": a.threads}}, indent=1))
    jobs = [tuple(map(int, j.split(":"))) for j in a.jobs]
    from train_dynamics import warm_cache
    warm_cache(ROOT / cfg["data_root"], ROOT / a.cache)
    ov = (a.device, a.workers, a.threads)
    results = []
    with ProcessPoolExecutor(max_workers=a.parallel) as pool:
        futs = [pool.submit(job, str(ROOT / a.config), f, s, str(out), str(ROOT / a.cache), ov, i)
                for i, (f, s) in enumerate(jobs)]
        for fu in as_completed(futs):
            r = fu.result()
            results.append(r)
            if "error" in r:
                print(f"FAIL fold {r['fold']['fold']} seed {r['seed']}\n{r['error']}", flush=True)
            else:
                print(f"done graph fold {r['fold']['fold']} ({r['fold']['test_building']}) seed {r['seed']}: best epoch "
                      f"{r['best_epoch']} val {r['best_val_total']:.4f} epochs {r['epochs_run']} {r['wall_s'] / 60:.1f} min", flush=True)
    name = f"training_summary_{'_'.join(a.jobs).replace(':', 's')}.json"
    (out / name).write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
