"""Parallel evaluator for the Smart Buildings training scenarios.

Runs (scenario x policy) jobs in separate processes using the starter kit's own
environment loader and metadata builder, and records for each rollout:
  * total / mean reward (the leaderboard uses |mean reward| per scenario)
  * the weighted comfort / energy / carbon parts of that mean
  * wall-clock split into policy time and simulator time

Usage (from the project root):
    .venv/bin/python src/run_eval.py --policies baseline bang_bang --workers 12 --out results/baseline
    .venv/bin/python src/run_eval.py --policies baseline --scenarios building_train_1/atlanta_2 --steps 288
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "starter-kit"
SCENARIOS = KIT / "smart_buildings" / "training_scenarios"
FULL_STEPS = 8640


def _policy_class(key: str):
    sys.path.insert(0, str(ROOT / "src"))
    if key == "baseline":
        import baseline_policy
        return baseline_policy.WideHotStorageReservePolicy
    from smart_buildings import evaluate
    return evaluate.POLICY_MAP[key]


def run_job(scenario: str, policy_key: str, steps: int, out_dir: str) -> dict:
    os.chdir(KIT)  # gin configs use paths relative to the starter kit
    import numpy as np
    from smart_buildings import evaluate as ev

    out_file = Path(out_dir) / policy_key / (scenario.replace("/", "__") + ".json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        t_start = time.perf_counter()
        input_dir = SCENARIOS / scenario
        env = ev.load_hybrid_environment(input_dir / "config.gin")
        env.metrics_path = None
        env._occupancy_normalization_constant = 125.0  # same as evaluate.py
        meta = ev.create_building_meta_data(env, input_dir)
        spec = env.action_spec()
        obs, _ = env.reset()
        t_init_policy = time.perf_counter()
        policy = ev.initialize_policy(_policy_class(policy_key), meta, spec)
        policy_time = time.perf_counter() - t_init_policy
        sim_time = 0.0

        rewards, comfort, energy, carbon = [], [], [], []
        for _ in range(steps):
            t0 = time.perf_counter()
            action = ev.validate_action(ev.get_policy_action(policy, obs), spec)
            t1 = time.perf_counter()
            obs, reward, terminated, truncated, _ = env.step(action)
            t2 = time.perf_counter()
            policy_time += t1 - t0
            sim_time += t2 - t1
            _, resp = env.get_reward_info_and_response()
            w = resp.productivity_weight + resp.energy_cost_weight + resp.carbon_emission_weight
            rewards.append(float(reward))
            comfort.append(resp.productivity_weight * resp.normalized_productivity_regret / w)
            energy.append(-resp.energy_cost_weight * resp.normalized_energy_cost / w)
            carbon.append(-resp.carbon_emission_weight * resp.normalized_carbon_emission / w)
            if terminated or truncated:
                break

        n = len(rewards)
        result = {
            "scenario": scenario,
            "policy": policy_key,
            "steps": n,
            "total_reward": float(np.sum(rewards)),
            "mean_reward": float(np.mean(rewards)),
            "mean_comfort": float(np.mean(comfort)),
            "mean_energy": float(np.mean(energy)),
            "mean_carbon": float(np.mean(carbon)),
            "worst_step_reward": float(np.min(rewards)),
            "num_ahus": len(meta["ahu_zone_assignments"]),
            "num_zones": len(meta["zone_temp_indices"]),
            "policy_seconds": policy_time,
            "sim_seconds": sim_time,
            "wall_seconds": time.perf_counter() - t_start,
        }
        np.save(out_file.with_suffix(".rewards.npy"), np.asarray([rewards, comfort, energy, carbon], dtype=np.float32))
    except Exception:
        result = {"scenario": scenario, "policy": policy_key, "error": traceback.format_exc()}
    out_file.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["baseline"])
    ap.add_argument("--scenarios", nargs="*", default=None, help="building/weather keys (default: all 30)")
    ap.add_argument("--steps", type=int, default=FULL_STEPS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "run")
    a = ap.parse_args()

    scenarios = a.scenarios or sorted(
        f"{p.parent.name}/{p.name}" for p in SCENARIOS.glob("*/*") if (p / "config.gin").exists()
    )
    out = a.out.resolve()
    jobs = [(s, p) for p in a.policies for s in scenarios]
    print(f"{len(jobs)} jobs, {a.workers} workers, {a.steps} steps -> {out}", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(run_job, s, p, a.steps, str(out)) for s, p in jobs]
        for f in as_completed(futures):
            r = f.result()
            if "error" in r:
                print(f"FAIL {r['policy']:10s} {r['scenario']}\n{r['error']}", flush=True)
            else:
                print(f"done {r['policy']:10s} {r['scenario']:30s} score={-r['mean_reward']:.5f} "
                      f"wall={r['wall_seconds']:.0f}s (policy {r['policy_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
