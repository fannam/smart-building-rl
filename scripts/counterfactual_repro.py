"""Phase 2B Stage 1a: simulator branch reproducibility (run BEFORE any counterfactual result is trusted).

For each scenario and branch point t_b (baseline policy, from reset):
  replay        fresh env + baseline policy vs the stored Phase 1 baseline episode (obs, reward)
  copy_restored copy at t_b + global-RNG restore, replay the original's actions for H steps -> vs original
  copy_copy     two independent restored copies -> vs each other
  copy_norestore copy at t_b WITHOUT RNG restore (the Phase 0 workaround) -> vs original
Also records copy wall time and RSS growth.

    .venv/bin/python scripts/counterfactual_repro.py --scenarios building_train_8/atlanta_1 building_train_2/fairbanks_2 \
        --branch-steps 300 900 --horizon 36 --out results/phase2b/counterfactual/reproducibility
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def rss_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return float("nan")


def run(scenario: str, branch_steps: list[int], H: int, data_root: str) -> dict:
    import numpy as np
    from counterfactual.branch import Branch, RNGState, env_copy
    from counterfactual.simrun import make_action, max_divergence, policy_action, record_step, setup, stack
    from dataset import load_episode
    env, smeta, policy, spec, obs = setup(scenario)
    rooms = smeta["zones"]["room_ids"]
    ep_id = f"{scenario.replace('/', '__')}__baseline__s0__e0"
    ref = load_episode(Path(data_root) / "episodes" / f"{ep_id}.npz", keys=["obs", "reward"])
    out = {"scenario": scenario, "H": H, "branches": []}
    recs, acts = [], []
    t = 0
    t_main = 0.0
    for tb in sorted(branch_steps):
        while t < tb:
            a = policy_action(policy, obs, spec)
            t0 = time.perf_counter()
            r = record_step(env, a, rooms)
            t_main += time.perf_counter() - t0
            recs.append(r); acts.append((a["continuous_action"].copy(), a["discrete_action"].copy()))
            obs = r["obs"]; t += 1
        rss0 = rss_mb()
        t0 = time.perf_counter()
        br = Branch.take(env)
        copy_s = time.perf_counter() - t0
        rss_copy = rss_mb() - rss0
        # original continues (closed loop) for H steps
        main = []
        for _ in range(H):
            a = policy_action(policy, obs, spec)
            r = record_step(env, a, rooms)
            main.append(r); recs.append(r); acts.append((a["continuous_action"].copy(), a["discrete_action"].copy()))
            obs = r["obs"]; t += 1
        rng_after = RNGState.capture()
        seq = acts[tb:tb + H]

        def replay(e):
            return stack([record_step(e, make_action(c, d, spec), rooms) for c, d in seq])
        m = stack(main)
        t0 = time.perf_counter()
        e1 = br.spawn(); r1 = replay(e1); del e1
        arm_s = time.perf_counter() - t0
        e2 = br.spawn(); r2 = replay(e2); del e2
        e3 = env_copy(br.env)                       # no RNG restore: global streams continue from "now"
        r3 = replay(e3); del e3
        rng_after.restore()                          # main trajectory continues with its own stream
        out["branches"].append({
            "t_b": tb, "copy_s": copy_s, "rss_per_copy_mb": rss_copy, "spawn_plus_H_steps_s": arm_s,
            "copy_restored_vs_original": max_divergence(r1, m),
            "copy_vs_copy": max_divergence(r1, r2),
            "copy_norestore_vs_original": max_divergence(r3, m),
            "zone_temp_norestore_mean_abs": float(np.mean(np.abs(r3["zone_info"][..., 2] - m["zone_info"][..., 2]))),
        })
        del br
    allr = stack(recs)
    n = min(len(allr["reward"]), len(ref["reward"]))
    out["replay_vs_phase1"] = {"steps": n, "obs_max_abs": float(np.max(np.abs(allr["obs"][:n] - ref["obs"][1:n + 1]))),
                               "reward_max_abs": float(np.max(np.abs(allr["reward"][:n] - ref["reward"][:n]))),
                               "note": "main trajectory interleaved with branch copies/arms; RNG restored after each"}
    out["sim_ms_per_step"] = 1e3 * t_main / max(1, t - H * len(branch_steps))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--branch-steps", nargs="+", type=int, default=[300, 900])
    ap.add_argument("--horizon", type=int, default=36)
    ap.add_argument("--data-root", default=str(ROOT / "data/phase1_sample"))
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--out", default="results/phase2b/counterfactual/reproducibility")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.parallel) as pool:
        res = list(pool.map(run, a.scenarios, [a.branch_steps] * len(a.scenarios), [a.horizon] * len(a.scenarios),
                            [a.data_root] * len(a.scenarios)))
    import platform
    import numba
    import numpy
    import scipy
    meta = {"argv": sys.argv, "python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "numba": numba.__version__, "platform": platform.platform()}
    (out / "reproducibility.json").write_text(json.dumps({"meta": meta, "results": res}, indent=1))
    for r in res:
        print(r["scenario"], json.dumps(r["replay_vs_phase1"]))
        for b in r["branches"]:
            print(f"  t_b={b['t_b']} copy {b['copy_s']:.2f}s rss {b['rss_per_copy_mb']:.0f}MB | restored: exact={b['copy_restored_vs_original']['bit_exact']} "
                  f"zT={b['copy_restored_vs_original']['zone_temp']:.2e} | copy-copy exact={b['copy_vs_copy']['bit_exact']} | "
                  f"norestore zT={b['copy_norestore_vs_original']['zone_temp']:.2e} obs={b['copy_norestore_vs_original']['obs']:.2e}")


if __name__ == "__main__":
    main()
