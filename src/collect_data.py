"""Transition dataset collector for Latent-MPC dynamics training (Phase 1).

Runs (scenario x profile x episode) jobs in separate processes, building each
environment exactly as `run_eval.py` does, and drives it with the baseline
policy (optionally wrapped by the seeded explorer in `exploration.py`).

Per episode it writes one compressed NumPy archive with, per step t:
  obs[t], obs[t+1]                (raw, as returned by the env; obs has T+1 rows)
  executed hybrid action a_t      (exact float32/int32 arrays passed to env.step)
  counterfactual baseline action  (what the unperturbed baseline would have done
                                   from the same internal state and o_t)
  reward r_t (env.step), all RewardResponse fields, the RewardInfo physical
  quantities (per-zone temps/setpoints/flows/occupancy, per-device power),
  privileged hot-water-tank state, timestamps, calendar, knob values, timings.

Usage (from the project root):
    .venv/bin/python src/collect_data.py --out data/phase1 \
        --scenarios building_train_8/atlanta_1 --profiles baseline moderate \
        --episodes 1 --steps 2016 --seed 0 --workers 4
See docs/phase1_dataset.md for the schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_eval import FULL_STEPS, KIT, SCENARIOS  # noqa: E402  (reuse eval constants)

ZONE_INFO_FIELDS = ("heating_setpoint_temperature", "cooling_setpoint_temperature",
                    "zone_air_temperature", "air_flow_rate_setpoint", "air_flow_rate",
                    "average_occupancy")
AHU_INFO_FIELDS = ("blower_electrical_energy_rate", "air_conditioning_electrical_energy_rate")
BOILER_INFO_FIELDS = ("natural_gas_heating_energy_rate", "pump_electrical_energy_rate")
HEAT_PUMP_INFO_FIELDS = ("electricity_heating_energy_rate", "pump_electrical_energy_rate")
REWARD_RESPONSE_FIELDS = (
    "agent_reward_value", "productivity_reward", "electricity_energy_cost",
    "natural_gas_energy_cost", "carbon_emitted", "carbon_cost", "productivity_weight",
    "energy_cost_weight", "carbon_emission_weight", "person_productivity", "total_occupancy",
    "reward_scale", "reward_shift", "productivity_regret", "normalized_productivity_regret",
    "normalized_energy_cost", "normalized_carbon_emission")
TANK_FIELDS = ("tank_temperature_k", "burner_heat_rate_w", "tank_step_temperature_change_k",
               "return_water_temperature_k", "boiler_supply_water_setpoint_k", "hws_flow_rate")
CALENDAR_FIELDS = ("hour", "minute", "weekday", "month", "day", "is_workday")


def _git_rev(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_env(scenario: str):
    """Environment + metadata exactly as run_eval.run_job builds them."""
    from smart_buildings import evaluate as ev
    input_dir = SCENARIOS / scenario
    env = ev.load_hybrid_environment(input_dir / "config.gin")
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0  # same as evaluate.py / run_eval.py (no-op, see meta)
    meta = ev.create_building_meta_data(env, input_dir)
    return env, meta, input_dir


def _tank_state(env) -> list[float]:
    hws = env.building.simulator._hvac.hot_water_system
    b = hws._heat_source
    return [float(b._current_temperature), float(b._burner_heat_rate),
            float(b._step_tank_temperature_change), float(b._return_water_temperature_sensor),
            float(b._supply_water_temperature_setpoint), float(hws.flow_rate)]


def _epoch(ts) -> int:
    import pandas as pd
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp())


def write_scenario_meta(out_root: Path, scenario: str, smeta: dict, input_dir: Path, policy) -> Path:
    import numpy as np
    d = out_root / "scenarios" / scenario.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    smeta = dict(smeta)
    # Baseline's public geometry prior (zone floor area [m2], exterior UA proxy),
    # derived only from zone_map, in canonical zone order.
    smeta["zone_area_m2"] = [float(x) for x in policy.zone_area]
    smeta["zone_exterior_ua_proxy"] = [float(x) for x in policy.zone_ua]
    tmp = d / f"meta.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(smeta, indent=1, default=str))
    tmp.replace(d / "meta.json")
    for name in ("zone_map.npy", "floor_plan.npy"):
        if (input_dir / name).exists() and not (d / name).exists():
            np.save(d / f"{name}.{os.getpid()}.tmp.npy", np.load(input_dir / name))
            Path(d / f"{name}.{os.getpid()}.tmp.npy").replace(d / name)
    return d


def collect_episode(scenario: str, profile: str, episode: int, seed: int, steps: int,
                    out_dir: str, compress: bool = True) -> dict:
    """Run one episode and write `<out>/episodes/<id>.npz`. Returns a manifest row."""
    os.chdir(KIT)  # gin configs use paths relative to the starter kit
    import numpy as np
    from smart_buildings import evaluate as ev
    from smart_buildings.smart_control.utils import conversion_utils
    import baseline_policy
    import pandas as pd
    from building_meta import build_scenario_metadata
    from exploration import ExplorationBaselinePolicy, Explorer, episode_rng, resolve_profile

    out_root = Path(out_dir)
    building, weather = scenario.split("/")
    episode_id = f"{building}__{weather}__{profile}__s{seed}__e{episode}"
    out_file = out_root / "episodes" / f"{episode_id}.npz"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        t_start = time.perf_counter()
        env, meta, input_dir = build_env(scenario)
        spec = env.action_spec()
        smeta = build_scenario_metadata(env, meta, scenario)
        cfg = resolve_profile(profile)
        obs, _ = env.reset()
        t0 = time.perf_counter()
        if cfg is None:  # the unmodified baseline class, as run_eval uses it
            policy = ev.initialize_policy(baseline_policy.WideHotStorageReservePolicy, meta, spec)
            explorer = None
        else:
            policy = ev.initialize_policy(ExplorationBaselinePolicy, meta, spec)
            explorer = Explorer(cfg, smeta, policy, episode_rng(seed, scenario, episode))
        init_s = time.perf_counter() - t0
        write_scenario_meta(out_root, scenario, smeta, input_dir, policy)

        zone_rooms = smeta["zones"]["room_ids"]
        T = min(steps, int(env._num_timesteps_in_episode))
        D, nc, nd, V = smeta["obs_dim"], smeta["n_continuous_action"], smeta["n_discrete_action"], smeta["num_zones"]
        obs_buf = np.zeros((T + 1, D), np.float32)
        a_c = np.zeros((T, nc), np.float32)
        a_d = np.zeros((T, nd), np.int8)
        b_c = np.zeros((T, nc), np.float32)
        b_d = np.zeros((T, nd), np.int8)
        reward = np.zeros(T, np.float64)
        rr = np.zeros((T, len(REWARD_RESPONSE_FIELDS)), np.float64)
        zone = np.zeros((T, V, len(ZONE_INFO_FIELDS)), np.float32)
        sim_ts = np.zeros(T + 1, np.int64)
        r_ts = np.zeros((T, 2), np.int64)
        cal = np.zeros((T, len(CALENDAR_FIELDS)), np.int16)
        tank = np.zeros((T + 1, len(TANK_FIELDS)), np.float64)
        timing = np.zeros((T, 2), np.float32)
        done = np.zeros((T, 2), bool)
        knob_cols = explorer.columns if explorer else []
        knobs = np.zeros((T, len(knob_cols)), np.float32)
        ahu_keys = boiler_keys = hp_keys = None
        ahu_info = boiler_info = hp_info = None

        obs_buf[0] = obs
        sim_ts[0] = _epoch(env.current_simulation_timestamp)
        tank[0] = _tank_state(env)
        n = 0
        for t in range(T):
            tp0 = time.perf_counter()
            if explorer is None:
                raw = ev.get_policy_action(policy, obs)
                action = ev.validate_action(raw, spec)
                base = action
            else:
                internal_changed = explorer.begin_step(obs)
                base_raw = explorer.baseline_clone().get_action(obs) if internal_changed else None
                raw = ev.get_policy_action(policy, obs)
                if base_raw is None:
                    base_raw = raw
                executed, knobs[t] = explorer.apply_action_knobs(raw, obs)
                action = ev.validate_action(executed, spec)
                base = ev.validate_action(base_raw, spec)
            tp1 = time.perf_counter()
            obs, r, terminated, truncated, _ = env.step(action)
            tp2 = time.perf_counter()
            info, resp = env.get_reward_info_and_response()

            a_c[t], a_d[t] = action["continuous_action"], action["discrete_action"]
            b_c[t], b_d[t] = base["continuous_action"], base["discrete_action"]
            obs_buf[t + 1] = obs
            reward[t] = float(r)
            rr[t] = [float(getattr(resp, f)) for f in REWARD_RESPONSE_FIELDS]
            for j, room in enumerate(zone_rooms):
                zi = info.zone_reward_infos[room]
                zone[t, j] = [getattr(zi, f) for f in ZONE_INFO_FIELDS]
            if ahu_keys is None:
                ahu_keys = sorted(info.air_handler_reward_infos.keys())
                boiler_keys = sorted(info.boiler_reward_infos.keys())
                hp_keys = sorted(info.heat_pump_reward_infos.keys())
                ahu_info = np.zeros((T, len(ahu_keys), len(AHU_INFO_FIELDS)), np.float32)
                boiler_info = np.zeros((T, len(boiler_keys), len(BOILER_INFO_FIELDS)), np.float32)
                hp_info = np.zeros((T, len(hp_keys), len(HEAT_PUMP_INFO_FIELDS)), np.float32)
            for j, k in enumerate(ahu_keys):
                ahu_info[t, j] = [getattr(info.air_handler_reward_infos[k], f) for f in AHU_INFO_FIELDS]
            for j, k in enumerate(boiler_keys):
                boiler_info[t, j] = [getattr(info.boiler_reward_infos[k], f) for f in BOILER_INFO_FIELDS]
            for j, k in enumerate(hp_keys):
                hp_info[t, j] = [getattr(info.heat_pump_reward_infos[k], f) for f in HEAT_PUMP_INFO_FIELDS]
            start = conversion_utils.proto_to_pandas_timestamp(resp.start_timestamp)
            end = conversion_utils.proto_to_pandas_timestamp(resp.end_timestamp)
            r_ts[t] = [int(start.timestamp()), int(end.timestamp())]
            cal[t] = [start.hour, start.minute, start.weekday(), start.month, start.day,
                      int(conversion_utils.is_work_day(start))]
            sim_ts[t + 1] = _epoch(env.current_simulation_timestamp)
            tank[t + 1] = _tank_state(env)
            timing[t] = [(tp1 - tp0) * 1e3, (tp2 - tp1) * 1e3]
            done[t] = [terminated, truncated]
            n = t + 1
            if terminated or truncated:
                break

        sl = slice(0, n)
        mean_parts = {
            "mean_reward": float(np.mean(reward[sl])),
            "mean_comfort": float(np.mean(rr[sl, REWARD_RESPONSE_FIELDS.index("normalized_productivity_regret")] * 0.6)),
            "mean_energy": float(np.mean(-rr[sl, REWARD_RESPONSE_FIELDS.index("normalized_energy_cost")] * 0.2)),
            "mean_carbon": float(np.mean(-rr[sl, REWARD_RESPONSE_FIELDS.index("normalized_carbon_emission")] * 0.2)),
        }
        perturbed = np.any(a_c[sl] != b_c[sl], axis=1) | np.any(a_d[sl] != b_d[sl], axis=1)
        episode_meta = {
            "schema_version": smeta["schema_version"],
            "episode_id": episode_id, "scenario": scenario, "building": building, "weather": weather,
            "profile": profile, "exploration_config": cfg.to_dict() if cfg else None,
            "episode_index": episode, "seed": seed, "steps": n, "requested_steps": steps,
            "start_timestamp": int(sim_ts[0]), "end_timestamp": int(sim_ts[n]),
            "num_ahus": smeta["num_ahus"], "num_zones": V, "obs_dim": D,
            "n_continuous_action": nc, "n_discrete_action": nd,
            "policy_class": type(policy).__name__,
            "perturbed_step_fraction": float(perturbed.mean()),
            **mean_parts,
            "policy_init_s": init_s, "policy_ms_mean": float(timing[sl, 0].mean()),
            "sim_ms_mean": float(timing[sl, 1].mean()),
            "wall_s": time.perf_counter() - t_start,
            "code": {"repo_rev": _git_rev(ROOT), "starter_kit_rev": _git_rev(KIT),
                     "baseline_policy_sha": _file_sha(ROOT / "src" / "baseline_policy.py"),
                     "exploration_sha": _file_sha(ROOT / "src" / "exploration.py"),
                     "collect_data_sha": _file_sha(ROOT / "src" / "collect_data.py")},
        }
        arrays = {
            "obs": obs_buf[: n + 1], "action_continuous": a_c[sl], "action_discrete": a_d[sl],
            "baseline_continuous": b_c[sl], "baseline_discrete": b_d[sl], "perturbed": perturbed,
            "reward": reward[sl], "reward_response": rr[sl], "zone_info": zone[sl],
            "ahu_info": ahu_info[sl], "boiler_info": boiler_info[sl], "heat_pump_info": hp_info[sl],
            "sim_timestamp": sim_ts[: n + 1], "reward_timestamp": r_ts[sl], "calendar": cal[sl],
            "tank_state": tank[: n + 1], "timing_ms": timing[sl], "done": done[sl],
            "knobs": knobs[sl],
            "names_reward_response": np.array(REWARD_RESPONSE_FIELDS),
            "names_zone_info": np.array(ZONE_INFO_FIELDS),
            "names_ahu_info": np.array(AHU_INFO_FIELDS), "keys_ahu_info": np.array(ahu_keys),
            "names_boiler_info": np.array(BOILER_INFO_FIELDS), "keys_boiler_info": np.array(boiler_keys),
            "names_heat_pump_info": np.array(HEAT_PUMP_INFO_FIELDS),
            "keys_heat_pump_info": np.array(hp_keys, dtype=str),
            "names_tank_state": np.array(TANK_FIELDS), "names_calendar": np.array(CALENDAR_FIELDS),
            "names_knobs": np.array(knob_cols, dtype=str),
            "episode_json": np.array(json.dumps(episode_meta)),
        }
        tmp = out_file.with_name(out_file.name + f".{os.getpid()}.tmp.npz")
        (np.savez_compressed if compress else np.savez)(tmp, **arrays)
        tmp.replace(out_file)
        row = {**episode_meta, "path": str(out_file.relative_to(out_root)),
               "bytes": out_file.stat().st_size}
        row.pop("code", None)
        row.pop("exploration_config", None)
        return row
    except Exception:  # noqa: BLE001
        return {"episode_id": episode_id, "scenario": scenario, "profile": profile, "seed": seed,
                "episode_index": episode, "error": traceback.format_exc()}


def all_scenarios() -> list[str]:
    return sorted(f"{p.parent.name}/{p.name}" for p in SCENARIOS.glob("*/*") if (p / "config.gin").exists())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scenarios", nargs="*", default=None, help="building/weather keys (default: all 30)")
    ap.add_argument("--buildings", nargs="*", default=None, help="restrict to these buildings")
    ap.add_argument("--profiles", nargs="+", default=["baseline", "moderate"])
    ap.add_argument("--episodes", type=int, default=1, help="episodes per (scenario, profile)")
    ap.add_argument("--steps", type=int, default=FULL_STEPS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-compress", action="store_true")
    a = ap.parse_args()

    scenarios = a.scenarios or all_scenarios()
    if a.buildings:
        scenarios = [s for s in scenarios if s.split("/")[0] in set(a.buildings)]
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for p in a.profiles:
        # baseline/null are deterministic: one episode is enough.
        n_ep = 1 if p in ("baseline", "null") else a.episodes
        jobs += [(s, p, e) for s in scenarios for e in range(n_ep)]
    (out / f"collection_config.{int(time.time())}.json").write_text(json.dumps({
        "argv": sys.argv, "scenarios": scenarios, "profiles": a.profiles, "episodes": a.episodes,
        "steps": a.steps, "seed": a.seed, "jobs": len(jobs)}, indent=1))
    print(f"{len(jobs)} episodes, {a.workers} workers, {a.steps} steps -> {out}", flush=True)
    manifest = out / "manifest.jsonl"
    # Larger buildings first for better packing.
    jobs.sort(key=lambda j: j[0].split("/")[0] in ("building_train_8",))
    with ProcessPoolExecutor(max_workers=a.workers) as pool, manifest.open("a") as mf:
        futs = [pool.submit(collect_episode, s, p, e, a.seed, a.steps, str(out), not a.no_compress)
                for s, p, e in jobs]
        for f in as_completed(futs):
            r = f.result()
            if "error" in r:
                print(f"FAIL {r['episode_id']}\n{r['error']}", flush=True)
                continue
            mf.write(json.dumps(r) + "\n")
            mf.flush()
            print(f"done {r['episode_id']:55s} T={r['steps']} score={-r['mean_reward']:.5f} "
                  f"pert={r['perturbed_step_fraction']:.2f} {r['bytes']/2**20:.1f}MB "
                  f"wall={r['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
