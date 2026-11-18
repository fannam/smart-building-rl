"""Phase 0 benchmark (read-only w.r.t. the simulator and policies).

Single process, one scenario. Measures, for one policy:
  * env construction, metadata build, reset, policy __init__ times
  * env.step and policy.get_action latency distributions over N steps
  * copy.deepcopy(env) time, pickled size, RSS growth (optional)
  * determinism of a copy (copy and original stepped with the same action)
  * time alignment: obs hour/dow features vs. simulator timestamp vs. reward timestamp
  * HWS tank configuration (energy_balance flag, capacity)

    .venv/bin/python scratch/phase0/bench_env.py --scenario building_train_8/atlanta_1 \
        --policy baseline --steps 1000 --copies 100 --out results/phase0/x.json
"""
from __future__ import annotations

import argparse, copy, gc, json, math, os, pickle, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "starter-kit"
sys.path.insert(0, str(ROOT / "src"))


def rss_mb() -> float:
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return float("nan")


def env_copy(env):
    """deepcopy that shares the read-only SuperLU factorization(s) (unpicklable).

    FiniteDifferenceSolver caches scipy's splu() in `_factorization`; solving with
    it is read-only, so the copy may share the original object.
    """
    from smart_buildings.smart_control.simulator.finite_difference_solver import FiniteDifferenceSolver
    memo = {}
    for o in gc.get_objects():
        if isinstance(o, FiniteDifferenceSolver) and getattr(o, "_factorization", None) is not None:
            memo[id(o._factorization)] = o._factorization
    return copy.deepcopy(env, memo)


def stats(xs):
    import numpy as np
    a = np.asarray(xs) * 1e3
    if a.size == 0:
        return {}
    return {"n": int(a.size), "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)),
            "max_ms": float(a.max()), "first_ms": float(a[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--policy", default="baseline")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--copies", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out_path = a.out.resolve()
    os.chdir(KIT)
    t = time.perf_counter
    t0 = t()
    import numpy as np
    from smart_buildings import evaluate as ev
    res = {"scenario": a.scenario, "policy": a.policy, "import_s": t() - t0, "rss_start_mb": rss_mb()}

    d = KIT / "smart_buildings" / "training_scenarios" / a.scenario
    t0 = t(); env = ev.load_hybrid_environment(d / "config.gin"); res["env_load_s"] = t() - t0
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0  # as evaluate.py / run_eval.py
    t0 = t(); meta = ev.create_building_meta_data(env, d); res["meta_s"] = t() - t0
    spec = env.action_spec()
    t0 = t(); obs, _ = env.reset(); res["reset_s"] = t() - t0
    res["K"] = len(meta["ahu_zone_assignments"]); res["V"] = len(meta["zone_temp_indices"])
    res["obs_dim"] = int(np.asarray(obs).size)

    if a.policy == "baseline":
        import baseline_policy
        cls = baseline_policy.WideHotStorageReservePolicy
    else:
        cls = ev.POLICY_MAP[a.policy]
    t0 = t(); pol = ev.initialize_policy(cls, meta, spec); res["policy_init_s"] = t() - t0

    # HWS tank configuration
    hws = env.building.simulator._hvac.hot_water_system
    src = hws._heat_source
    res["hws"] = {"energy_balance": bool(hws.energy_balance), "heat_source": type(src).__name__,
                  "water_capacity_m3": getattr(src, "_water_capacity", None),
                  "heating_rate_K_per_min": getattr(src, "_heating_rate", None)}

    names = list(meta["observation_field_names"])
    fi = {n: i for i, n in enumerate(names)}
    align = []
    act_t, step_t, rewards = [], [], []
    for i in range(a.steps):
        ts = env.current_simulation_timestamp
        t0 = t(); act = ev.validate_action(ev.get_policy_action(pol, obs), spec); t1 = t()
        obs, r, term, trunc, _ = env.step(act); t2 = t()
        act_t.append(t1 - t0); step_t.append(t2 - t1); rewards.append(float(r))
        if i < 300 and i % 12 == 0 or i in (0, 1, 2):
            info, resp = env.get_reward_info_and_response()
            hod = math.atan2(float(obs[fi["hod_sin_000"]]), float(obs[fi["hod_cos_000"]])) % (2 * math.pi) * 12 / math.pi
            align.append({"step": i, "sim_ts_before_step": str(ts), "sim_ts_after_step": str(env.current_simulation_timestamp),
                          "reward_start": str(resp.start_timestamp.ToDatetime()), "obs_hour": round(hod, 3)})
        if term or trunc:
            break
    res["get_action"] = stats(act_t)
    res["env_step"] = stats(step_t)
    res["mean_reward_partial"] = float(np.mean(rewards))
    res["time_alignment"] = align

    if a.copies:
        gc.collect(); r0 = rss_mb()
        try:
            t0 = t(); blob = pickle.dumps(env); res["pickle_s"] = t() - t0; res["pickle_mb"] = len(blob) / 2**20
            del blob
        except Exception as e:  # noqa: BLE001
            res["pickle_error"] = repr(e)
        times, keep = [], []
        try:
            copy.deepcopy(env)
        except Exception as e:  # noqa: BLE001
            res["plain_deepcopy_error"] = repr(e)
        for _ in range(a.copies):
            t0 = t(); c = env_copy(env); times.append(t() - t0)
            if len(keep) < 5:
                keep.append(c)
            else:
                del c
        res["deepcopy"] = stats(times)
        gc.collect(); res["rss_with_5_copies_mb"] = rss_mb(); res["rss_before_copies_mb"] = r0
    if a.copies:
        # determinism: original and a fresh copy, same action sequence, 24 steps
        c = env_copy(env)
        p2 = copy.deepcopy(pol)
        o1, o2 = obs, obs
        diffs = []
        for _ in range(24):
            act = ev.validate_action(ev.get_policy_action(pol, o1), spec)
            act2 = ev.validate_action(ev.get_policy_action(p2, o2), spec)
            o1, r1, *_ = env.step(act); o2, r2, *_ = c.step(act2)
            diffs.append(abs(float(r1) - float(r2)) + float(np.max(np.abs(np.asarray(o1) - np.asarray(o2)))))
        res["copy_determinism_max_abs_diff"] = float(max(diffs))
    res["rss_end_mb"] = rss_mb()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "time_alignment"}, indent=1, default=str))


if __name__ == "__main__":
    main()
