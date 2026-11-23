"""Runtime feasibility check: how much per-step time is left for a learned-model planner?

Measures, per scenario, on the machine it runs on (meant for a Kaggle CPU notebook,
kaggle/runtime_check.ipynb):

  A. Simulator cost: one rollout driven by the starter kit's `bang_bang` policy (cheapest policy), timing
     env load, reset, policy init, every env.step and get_action. `--steps 8640` is a full episode.
  B. Model cost: the Phase 2C graph `paired` ensemble (3 seeds, the fold that holds the scenario's
     building out) predicting N candidate action sequences of horizon H from the rollout's live context
     (last 12 observations and executed actions). Featurization (`build_items` + `collate`, the
     current per-candidate Python path) and the network rollout are timed separately; the median of
     --repeats calls is reported. Trained weights are not needed for timing, but the checkpoints
     must exist (any model_seed*.pt in the run directory is used, partial or not).
  C. Budget table: per-step headroom = (allowance per scenario - simulator rollout time) / steps, under
     three allowance hypotheses (phase0_findings.md section 2.4):
        48 scenarios in 9 h  -> 675 s/scenario
        48 scenarios in 12 h -> 900 s/scenario
        10x bang_bang        -> 10 * measured rollout time
     and the per-step cost of planner configurations (samples N, horizon H, CEM iterations, replan
     interval), each marked fits / does not fit.

    python scripts/runtime_check.py --kit <starter-kit root> --runs <.../results/phase2c/training> \
        --phase1-root data/phase1_sample --scenarios building_train_8/atlanta_1 building_train_2/atlanta_3 \
        --steps 8640 --out results/runtime_check
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

ALLOWANCES = {"48 in 9 h": 9 * 3600 / 48, "48 in 12 h": 12 * 3600 / 48}
PLANNERS = [  # (samples N, horizon H, CEM iterations, replan every k steps)
    (1, 12, 1, 1), (1, 12, 1, 6), (20, 12, 1, 6), (20, 12, 1, 1), (20, 36, 1, 1), (64, 12, 1, 1), (64, 12, 3, 1), (64, 36, 3, 3),
    (128, 12, 3, 3), (128, 36, 3, 6), (256, 36, 3, 12),
]


def stats_ms(xs) -> dict:
    a = np.asarray(xs, dtype=float) * 1e3
    return {"n": int(a.size), "mean_ms": float(a.mean()), "p50_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
            "max_ms": float(a.max()), "first_ms": float(a[0])} if a.size else {}


def simulate(kit: Path, scenario: str, steps: int) -> tuple[dict, dict]:
    """bang_bang rollout; returns timings and the last 12 observations / executed actions."""
    from smart_buildings import evaluate as ev
    t = time.perf_counter
    res: dict = {"scenario": scenario}
    cwd = os.getcwd()
    os.chdir(kit)                        # config.gin asset paths are relative to the simulator root
    try:
        d = kit / "smart_buildings" / "training_scenarios" / scenario
        t0 = t(); env = ev.load_hybrid_environment(d / "config.gin"); res["env_load_s"] = t() - t0
        env.metrics_path = None
        env._occupancy_normalization_constant = 125.0   # as the gateway and evaluate.py do (no-op)
        t0 = t(); meta = ev.create_building_meta_data(env, d); res["meta_s"] = t() - t0
    finally:
        os.chdir(cwd)
    spec = env.action_spec()
    t0 = t(); obs, _ = env.reset(); res["reset_s"] = t() - t0
    t0 = t(); pol = ev.initialize_policy(ev.POLICY_MAP["bang_bang"], meta, spec); res["policy_init_s"] = t() - t0
    res["K"], res["V"] = len(meta["ahu_zone_assignments"]), len(meta["zone_temp_indices"])
    hist_obs, hist_ac, hist_ad = [], [], []
    act_t, step_t = [], []
    t_roll = t()
    for _ in range(steps):
        t0 = t(); act = ev.validate_action(ev.get_policy_action(pol, obs), spec); t1 = t()
        hist_obs.append(np.asarray(obs, np.float32))
        hist_ac.append(np.asarray(act["continuous_action"], np.float32)); hist_ad.append(np.asarray(act["discrete_action"], np.int32))
        hist_obs, hist_ac, hist_ad = hist_obs[-12:], hist_ac[-12:], hist_ad[-12:]
        obs, _r, term, trunc, _ = env.step(act); t2 = t()
        act_t.append(t1 - t0); step_t.append(t2 - t1)
        if term or trunc:
            break
    res["steps"] = len(step_t)
    res["rollout_loop_s"] = t() - t_roll
    res["rollout_total_s"] = res["rollout_loop_s"] + res["env_load_s"] + res["meta_s"] + res["reset_s"] + res["policy_init_s"]
    res["env_step"], res["get_action"] = stats_ms(step_t), stats_ms(act_t)
    ctx = {"obs": np.stack(hist_obs), "ac": np.stack(hist_ac), "ad": np.stack(hist_ad), "last_obs": np.asarray(obs, np.float32)}
    return res, ctx


def model_timing(runs: Path, phase1_root: Path, scenario: str, ctx: dict, sizes, horizons, repeats: int, arch: str) -> dict:
    import torch
    from counterfactual.model_eval import build_items
    from counterfactual_eval import EnsembleMean
    from dataset import load_scenario_meta
    from dynamics.features import collate
    from dynamics.model import rollout
    from graph.train import GraphAuto, load_graph

    building = scenario.split("/")[0]
    folds = json.loads((ROOT / "configs/leave_one_building_out.json").read_text())["folds"]
    fold = next(f for f in folds if f["test_building"] == building)
    cks = sorted(runs.glob(f"{arch}/fold_{fold['fold']}_*/model_seed*.pt"))
    assert cks, f"no checkpoints under {runs}/{arch}/fold_{fold['fold']}_*"
    members = [GraphAuto(load_graph(c), phase1_root) for c in cks]
    ens = EnsembleMean(members)
    meta = load_scenario_meta(phase1_root, scenario)
    rng = np.random.default_rng(0)
    out = {"fold": fold["fold"], "checkpoints": [str(c.relative_to(runs)) for c in cks], "torch_threads": torch.get_num_threads(), "cases": []}
    for H in horizons:
        for N in sizes:
            base_ac = np.repeat(ctx["ac"][-1][None, None], N, 0).repeat(H, 1)                  # [N, H, n_c]
            arm_ac = np.clip(base_ac + rng.normal(0, 0.2, base_ac.shape), -1, 1).astype(np.float32)
            arm_ad = np.repeat(ctx["ad"][-1][None, None], N, 0).repeat(H, 1).astype(np.int32)
            sim = {"H": np.array(H), "ctx_obs": ctx["obs"][None], "ctx_ac": ctx["ac"][None], "ctx_ad": ctx["ad"][None],
                   "arm_ac": arm_ac[None], "arm_ad": arm_ad[None], "oat_future_obs": np.repeat(ctx["last_obs"][None], H, 0)[None],
                   "scenario": np.array(scenario)}
            t_feat, t_one, t_ens = [], [], []
            for r in range(repeats + 1):                                                       # first call = warm-up
                t0 = time.perf_counter()
                batch = collate(build_items(sim, meta, 0))
                t1 = time.perf_counter()
                with torch.no_grad():
                    rollout(members[0], dict(batch))
                    t2 = time.perf_counter()
                    ens.predict(dict(batch))
                    t3 = time.perf_counter()
                if r:
                    t_feat.append(t1 - t0); t_one.append(t2 - t1); t_ens.append(t3 - t2)
            case = {"N": N, "H": H, "featurize_ms": float(np.median(t_feat)) * 1e3, "model_1_ms": float(np.median(t_one)) * 1e3,
                    "ensemble_ms": float(np.median(t_ens)) * 1e3}
            case["call_ms"] = case["featurize_ms"] + case["ensemble_ms"]
            out["cases"].append(case)
            print(f"  [{scenario}] N={N:4d} H={H:2d}: featurize {case['featurize_ms']:8.1f} ms | 1 model {case['model_1_ms']:8.1f} ms | "
                  f"ensemble({len(members)}) {case['ensemble_ms']:8.1f} ms", flush=True)
    return out


def call_ms(cases: list[dict], N: int, H: int) -> float:
    """Cost of one ensemble call at (N, H): measured, or linearly interpolated/extrapolated in N."""
    pts = sorted((c["N"], c["call_ms"]) for c in cases if c["H"] == H)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if N in xs:
        return ys[xs.index(N)]
    if len(xs) == 1:
        return ys[0] * N / xs[0]
    slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) if N > xs[-1] else (ys[1] - ys[0]) / (xs[1] - xs[0])
    anchor = (xs[-1], ys[-1]) if N > xs[-1] else (xs[0], ys[0])
    return max(0.0, anchor[1] + slope * (N - anchor[0]))


def budget(sim: dict, model: dict) -> dict:
    T = sim["rollout_total_s"] * 8640 / sim["steps"]          # scaled to a full episode if --steps < 8640
    allow = {**ALLOWANCES, "10x bang_bang": 10 * T}
    head = {k: (v - T) / 8640 * 1e3 for k, v in allow.items()}  # ms per step left for the policy
    rows = []
    for N, H, it, k in PLANNERS:
        if not any(c["H"] == H for c in model["cases"]):
            continue
        per_step = call_ms(model["cases"], N, H) * it / k
        rows.append({"N": N, "H": H, "iters": it, "replan_every": k, "per_step_ms": per_step,
                     "fits": {a: bool(per_step <= h) for a, h in head.items()}})
    return {"rollout_full_episode_s": T, "allowance_s": allow, "headroom_ms_per_step": head, "planners": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True, help="simulator root (directory containing smart_buildings/)")
    ap.add_argument("--runs", default="results/phase2c/training")
    ap.add_argument("--arch", default="graph_paired_lam1")
    ap.add_argument("--phase1-root", default="data/phase1_sample")
    ap.add_argument("--scenarios", nargs="+", default=["building_train_8/atlanta_1", "building_train_2/atlanta_3"])
    ap.add_argument("--steps", type=int, default=8640)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 20, 64, 128])
    ap.add_argument("--horizons", type=int, nargs="+", default=[12, 36])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--threads", type=int, default=None, help="torch threads (default: torch's choice)")
    ap.add_argument("--out", default="results/runtime_check")
    a = ap.parse_args()
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    kit, runs, p1 = Path(a.kit).resolve(), (ROOT / a.runs).resolve(), (ROOT / a.phase1_root).resolve()
    out = (ROOT / a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"machine": {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
                          "torch": torch.__version__, "torch_threads": torch.get_num_threads()},
              "argv": sys.argv, "scenarios": {}}
    for sc in a.scenarios:
        print(f"== {sc}: bang_bang rollout, {a.steps} steps", flush=True)
        sim, ctx = simulate(kit, sc, a.steps)
        print(f"  rollout {sim['rollout_total_s']:.0f}s total, env.step mean {sim['env_step']['mean_ms']:.1f} ms "
              f"(K={sim['K']}, V={sim['V']})", flush=True)
        print(f"== {sc}: {a.arch} ensemble inference", flush=True)
        model = model_timing(runs, p1, sc, ctx, a.sizes, a.horizons, a.repeats, a.arch)
        b = budget(sim, model)
        report["scenarios"][sc] = {"simulator": sim, "model": model, "budget": b}
        (out / "runtime_check.json").write_text(json.dumps(report, indent=1))
        print(f"\n== {sc}: budget (full episode {b['rollout_full_episode_s']:.0f}s of simulator + bang_bang)")
        for k, h in b["headroom_ms_per_step"].items():
            print(f"  {k:14s}: allowance {b['allowance_s'][k]:7.0f} s/scenario -> headroom {h:8.1f} ms/step")
        names = list(b["headroom_ms_per_step"])
        print(f"  {'planner (N, H, iters, replan)':32s} {'ms/step':>9s}  " + "  ".join(f"{n:>13s}" for n in names))
        for r in b["planners"]:
            cfg = f"N={r['N']}, H={r['H']}, it={r['iters']}, every {r['replan_every']}"
            print(f"  {cfg:32s} {r['per_step_ms']:9.1f}  " + "  ".join(f"{'fits' if r['fits'][n] else '-':>13s}" for n in names))
    print(f"\n-> {out / 'runtime_check.json'}")


if __name__ == "__main__":
    main()
