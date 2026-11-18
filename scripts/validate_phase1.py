"""Phase 1 validation: collection pipeline correctness on a small building.

Collects short episodes on building_train_8 (K=3, V=30) and checks:
  1. short collection runs
  2. observation/action/timestamp alignment for consecutive steps
  3. reward reconstruction vs the simulator reward
  4. discrete/continuous action serialization (dtypes, ranges, exact replay)
  5. determinism under a fixed seed (and change under a different seed)
  6. validity of randomized actions (ranges, schedule rules, knob channels)
  7. loading through PyTorch
  8. baseline unchanged (sources untouched, null profile == baseline,
     full-horizon baseline profile == stored results/baseline_full rollout)

    .venv/bin/python scripts/validate_phase1.py [--steps 1152] [--full-baseline DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SCENARIO = "building_train_8/atlanta_1"
OUT = ROOT / "data" / "phase1_validation"
RESULTS = ROOT / "results" / "phase1"
CHECKS: dict[str, dict] = {}


def check(name: str, ok: bool, **info):
    CHECKS[name] = {"ok": bool(ok), **info}
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(info, default=float)[:300]}", flush=True)


def _collect(args):
    from collect_data import collect_episode
    sub, profile, episode, seed, steps = args
    return collect_episode(SCENARIO, profile, episode, seed, steps, str(OUT / sub))


def _ep_path(sub, profile, seed, episode=0):
    b, w = SCENARIO.split("/")
    return OUT / sub / "episodes" / f"{b}__{w}__{profile}__s{seed}__e{episode}.npz"


def replay(path: Path) -> dict:
    """Rebuild the env and step it with the stored actions; compare obs/reward bitwise."""
    os.chdir(ROOT / "starter-kit")
    from collect_data import build_env
    from dataset import load_episode
    ep = load_episode(path)
    env, meta, _ = build_env(SCENARIO)
    obs, _ = env.reset()
    obs_ok = np.array_equal(obs, ep["obs"][0])
    max_obs = 0.0
    rew_equal = True
    for t in range(ep["reward"].shape[0]):
        action = {"continuous_action": ep["action_continuous"][t].astype(np.float32),
                  "discrete_action": ep["action_discrete"][t].astype(np.int32)}
        obs, r, *_ = env.step(action)
        max_obs = max(max_obs, float(np.max(np.abs(obs - ep["obs"][t + 1]))))
        rew_equal &= float(r) == float(ep["reward"][t])
    return {"initial_obs_equal": bool(obs_ok), "max_abs_obs_diff": max_obs, "rewards_bit_equal": bool(rew_equal)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1152, help="4 days from Sun 06:00 (b8/atlanta_1)")
    ap.add_argument("--full-baseline", type=Path, default=OUT / "full_baseline")
    a = ap.parse_args()
    T = a.steps
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 1. collection
    jobs = [("run_a", "moderate", 0, 1, T), ("run_b", "moderate", 0, 1, T), ("run_a", "moderate", 0, 2, T),
            ("run_a", "null", 0, 1, T), ("run_a", "baseline", 0, 1, T), ("run_a", "mild", 0, 1, T)]
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        rows = list(pool.map(_collect, jobs))
    errors = [r for r in rows if "error" in r]
    check("1_short_collection", not errors, episodes=len(rows), steps=T,
          wall_s=[round(r.get("wall_s", -1), 1) for r in rows],
          policy_ms_mean=[round(r.get("policy_ms_mean", -1), 3) for r in rows],
          sim_ms_mean=[round(r.get("sim_ms_mean", -1), 2) for r in rows],
          mb=[round(r.get("bytes", 0) / 2**20, 2) for r in rows], errors=[e["error"][-500:] for e in errors])
    if errors:
        sys.exit(1)
    with (OUT / "run_a" / "manifest.jsonl").open("w") as mf:
        for job, r in zip(jobs, rows):
            if job[0] == "run_a":
                mf.write(json.dumps(r) + "\n")

    from building_meta import ActionCodec, ObservationCodec
    from dataset import load_episode, load_scenario_meta
    smeta = load_scenario_meta(OUT / "run_a", SCENARIO)
    oc, ac = ObservationCodec(smeta), ActionCodec(smeta)
    mod = load_episode(_ep_path("run_a", "moderate", 1))
    base = load_episode(_ep_path("run_a", "baseline", 1))

    # ---------------------------------------------------------------- 2. alignment
    st, rts = mod["sim_timestamp"], mod["reward_timestamp"]
    names = smeta["observation_field_names"]
    cont_names = smeta["continuous_action_names"]
    disc_names = smeta["discrete_action_names"]
    nxt_native = oc.native(mod["obs"][1:])
    prev_native = oc.native(mod["obs"][:-1])
    a_native = ac.to_native(mod["action_continuous"])
    echo = {}
    for j, n in enumerate(cont_names):
        if n in names and not n.endswith(("static_pressure_setpoint", "recirculation_percentage_command")):
            i = names.index(n)
            echo[n] = (float(np.max(np.abs(nxt_native[:, i] - a_native[:, j]))),
                       float(np.mean(np.abs(prev_native[1:, i] - a_native[1:, j]) > 1e-3)))
    for j, n in enumerate(disc_names):
        if n in names:
            i = names.index(n)
            on = nxt_native[:, i] > 0.0
            echo[n] = (float(np.max(np.abs(on.astype(int) - mod["action_discrete"][:, j]))),
                       float(np.mean((prev_native[1:, i] > 0) != mod["action_discrete"][1:, j].astype(bool))))
    worst = max(v[0] for v in echo.values())
    # discrimination: how often o_t (one step stale) disagrees with a_t on the same echoes
    stale = float(np.mean([v[1] for v in echo.values()]))
    hour_obs = oc.hour(mod["obs"][1:])
    hour_rew = mod["calendar"][:, 0] + mod["calendar"][:, 1] / 60.0
    hour_err = float(np.max(np.abs(((hour_obs - hour_rew) + 12) % 24 - 12)))
    # Sensor lag: RewardInfo of step t is what the sensors show in o_{t+2}.
    zi = mod["zone_info"].astype(np.float64)
    zn = mod["names_zone_info"]
    o2 = mod["obs"][2:]
    lag_err = {
        "zone_temp": float(np.max(np.abs(oc.zone_temperatures(o2) - zi[:-1, :, zn.index("zone_air_temperature")]))),
        "heating_sp": float(np.max(np.abs(oc.zone_bands(o2)[0] - zi[:-1, :, zn.index("heating_setpoint_temperature")]))),
        "cooling_sp": float(np.max(np.abs(oc.zone_bands(o2)[1] - zi[:-1, :, zn.index("cooling_setpoint_temperature")]))),
        "num_occupants": float(np.max(np.abs(oc.field(o2, "num_occupants") - zi[:-1, :, zn.index("average_occupancy")].sum(-1)))),
    }
    lag1_temp_err = float(np.max(np.abs(oc.zone_temperatures(mod["obs"][1:]) - zi[:, :, zn.index("zone_air_temperature")])))
    check("2_alignment",
          bool(np.all(np.diff(st) == 300) and np.all(rts[:, 0] == st[1:]) and np.all(rts[:, 1] - rts[:, 0] == 300)
               and worst < 1e-3 and hour_err < 1e-3 and max(lag_err.values()) < 1e-3),
          reward_info_t_equals_sensors_in_obs_t_plus_2=lag_err, zone_temp_err_if_using_obs_t_plus_1=lag1_temp_err,
          sim_dt_unique=np.unique(np.diff(st)).tolist(), reward_start_eq_next_sim_ts=bool(np.all(rts[:, 0] == st[1:])),
          echo_fields_checked=len(echo), echo_max_abs_err_vs_a_t=worst,
          stale_obs_mismatch_rate=stale, obs_hour_vs_reward_hour_max_err_h=hour_err,
          excluded_effective_echoes="static_pressure (0 when AHU off), recirculation (capped at 0.7)")

    # ---------------------------------------------------------------- 3. reward reconstruction
    from reward_reconstruction import compare, reconstruct
    import reward_model as rm
    from smart_buildings.smart_control.reward import electricity_energy_cost as ec, natural_gas_energy_cost as gc
    from smart_buildings.smart_control.utils import constants as kc
    tables_equal = (np.array_equal(rm.WEEKDAY_PRICE_CENTS_PER_KWH, ec.WEEKDAY_PRICE_BY_HOUR)
                    and np.array_equal(rm.WEEKEND_PRICE_CENTS_PER_KWH, ec.WEEKEND_PRICE_BY_HOUR)
                    and np.array_equal(rm.CARBON_G_PER_KWH_BY_HOUR, ec.CARBON_EMISSION_BY_HOUR)
                    and np.array_equal(rm.GAS_PRICE_BY_MONTH, gc.GAS_PRICE_BY_MONTH_SOURCE)
                    and rm.GAS_CO2 == kc.GAS_CO2 and rm.KWH_PER_KFT3_GAS == kc.KWH_PER_KFT3_GAS
                    and rm.JOULES_PER_KWH == kc.JOULES_PER_KWH)
    rc = smeta["reward_config"]
    consts_ok = (rc["max_electricity_rate_over_K"] == rm.DEFAULT_CONSTANTS.c_electricity_per_ahu_w
                 and rc["max_natural_gas_rate_over_K"] == rm.DEFAULT_CONSTANTS.c_natural_gas_per_ahu_w
                 and rc["productivity_deviation_ref"] == rm.DEFAULT_CONSTANTS.productivity_deviation_ref_k)
    rec = {p: compare(e, reconstruct(e, smeta)) for p, e in (("moderate", mod), ("baseline", base))}
    exact_err = max(rec[p]["exact"]["max_abs_err"] for p in rec)
    check("3_reward_reconstruction",
          tables_equal and consts_ok and exact_err < 1e-6 and all(rec[p]["step_reward_equals_response"] for p in rec),
          tables_equal_to_starter_kit=tables_equal, K_derived_normalizers_match_config=consts_ok,
          exact_max_abs_err=exact_err,
          variants={p: {v: {k: rec[p][v][k] for k in ("max_abs_err", "episode_mean_rel_err", "comfort_mean_err")}
                        for v in ("gas_mean", "occ_uniform", "occ_area", "obs_lag1", "obs_temps", "causal")} for p in rec},
          occupied_steps={p: rec[p]["occupied_steps"] for p in rec},
          num_occupants_obs_vs_true={p: rec[p]["num_occupants_obs_vs_total_avg_occ"] for p in rec},
          hour_dow_from_obs={p: (rec[p]["hour_from_obs_matches_reward_hour"], rec[p]["dow_from_obs_matches_reward_dow"]) for p in rec})

    # ---------------------------------------------------------------- 4. serialization
    ser = {}
    for p, e in (("moderate", mod), ("baseline", base)):
        ser[p] = {"cont_dtype": str(e["action_continuous"].dtype), "disc_dtype": str(e["action_discrete"].dtype),
                  "cont_in_range": bool(np.all(np.abs(e["action_continuous"]) <= 1.0)),
                  "disc_binary": bool(np.isin(e["action_discrete"], (0, 1)).all()),
                  "cont_shape": list(e["action_continuous"].shape), "disc_shape": list(e["action_discrete"].shape)}
    with ProcessPoolExecutor(max_workers=2) as pool:
        rp = dict(zip(("moderate", "baseline"), pool.map(replay, [_ep_path("run_a", "moderate", 1),
                                                                  _ep_path("run_a", "baseline", 1)])))
    check("4_action_serialization",
          all(s["cont_in_range"] and s["disc_binary"] for s in ser.values())
          and all(r["initial_obs_equal"] and r["max_abs_obs_diff"] == 0.0 and r["rewards_bit_equal"] for r in rp.values()),
          dtypes=ser, replay=rp,
          layout={"n_continuous": len(cont_names), "n_discrete": len(disc_names), "discrete_names": disc_names})

    # ---------------------------------------------------------------- 5. determinism
    b2 = load_episode(_ep_path("run_b", "moderate", 1))
    s2 = load_episode(_ep_path("run_a", "moderate", 2))
    skip = {"timing_ms", "meta"}
    diff_keys = [k for k in mod if k not in skip and not (
        np.array_equal(mod[k], b2[k]) if isinstance(mod[k], np.ndarray) else mod[k] == b2[k])]
    meta_equal = {k: v for k, v in mod["meta"].items() if k not in ("wall_s", "policy_init_s", "policy_ms_mean", "sim_ms_mean")} == \
                 {k: v for k, v in b2["meta"].items() if k not in ("wall_s", "policy_init_s", "policy_ms_mean", "sim_ms_mean")}
    seed_changes = not np.array_equal(mod["knobs"], s2["knobs"])
    check("5_determinism", not diff_keys and meta_equal and seed_changes,
          differing_arrays_same_seed=diff_keys, episode_meta_equal=meta_equal,
          different_seed_changes_knobs=seed_changes,
          seed2_action_diff_fraction=float(np.mean(np.any(mod["action_continuous"] != s2["action_continuous"], 1))))

    # ---------------------------------------------------------------- 6. validity of randomized actions
    from exploration import ExplorationBaselinePolicy
    PRE0 = float(ExplorationBaselinePolicy.DEFAULTS["precondition_hours"])
    val = {"default_precondition_hours": PRE0}
    for prof in ("moderate", "mild"):
        e = mod if prof == "moderate" else load_episode(_ep_path("run_a", "mild", 1))
        kn = {n: e["knobs"][:, i] for i, n in enumerate(e["names_knobs"])}
        K = smeta["num_ahus"]
        ah = smeta["ahus"]
        now = e["obs"][:-1, names.index("comfort_mode_now")] > .5
        soon = e["obs"][:-1, names.index("comfort_mode_soon")] > .5
        nat = ac.to_native(e["action_continuous"])
        sat = nat[:, ah["act_sat"]]
        pres = nat[:, ah["act_static_pressure"]]
        sw = nat[:, smeta["hws"]["act_supply_water_temp"]]
        forced_off = np.stack([kn[f"ahu_force_{k}"] == 0 for k in range(K)], 1)
        forced_off_while_occupied = int(np.sum(forced_off & (now | soon)[:, None]))
        run_exec = e["action_discrete"][:, ah["act_run"]]
        run_base = e["baseline_discrete"][:, ah["act_run"]]
        consistent_off = bool(np.all(run_exec[forced_off] == 0) and np.all(run_base[forced_off] == 1))
        # Steps where no policy-internal knob is active: only knob channels may differ.
        internal = (kn["setback_low"] != 0) | (kn["setback_high"] != 0) | (kn["precondition_h"] != PRE0)
        allowed = np.zeros(len(cont_names), bool)
        allowed[ah["act_sat"]] = allowed[ah["act_static_pressure"]] = True
        allowed[smeta["hws"]["act_supply_water_temp"]] = True
        dc = e["action_continuous"] != e["baseline_continuous"]
        outside_channels = int(np.sum(dc[~internal][:, ~allowed]))
        val[prof] = {
            "sat_native_range": [float(sat.min()), float(sat.max())],
            "pressure_native_range": [float(pres.min()), float(pres.max())],
            "hws_native_range": [float(sw.min()), float(sw.max())],
            "forced_off_steps": int(forced_off.sum()), "forced_on_steps": int(np.sum(np.stack(
                [kn[f"ahu_force_{k}"] == 1 for k in range(K)]))),
            "forced_off_while_occupied_or_prestart": forced_off_while_occupied,
            "forced_off_consistent": consistent_off,
            "nonknob_action_changes_without_internal_knobs": outside_channels,
            "active_fraction": {
                "sat": float(np.mean(np.stack([kn[f"sat_offset_{k}"] != 0 for k in range(K)]))),
                "pressure": float(np.mean(np.stack([kn[f"pressure_scale_{k}"] != 1 for k in range(K)]))),
                "hws": float(np.mean(kn["hws_offset"] != 0)),
                "setback": float(np.mean(kn["setback_low"] != 0)),
                "precondition_changed": float(np.mean(kn["precondition_h"] != PRE0)),
            },
            "guard_ahu_steps": int(sum(np.sum(kn[f"guard_ahu_{k}"]) for k in range(K))),
            "guard_hws_steps": int(np.sum(kn["guard_hws"])),
            "perturbed_step_fraction": float(e["perturbed"].mean()),
            "score": -float(e["reward"].mean()),
            "mean_comfort": float(0.6 * e["reward_response"][:, e["names_reward_response"].index("normalized_productivity_regret")].mean()),
        }
    val["baseline_same_window"] = {"score": -float(base["reward"].mean()),
                                   "mean_comfort": float(0.6 * base["reward_response"][:, base["names_reward_response"].index("normalized_productivity_regret")].mean())}
    ok6 = all(v["sat_native_range"][0] >= 285 - 1e-4 and v["sat_native_range"][1] <= 305 + 1e-4
              and v["pressure_native_range"][0] >= -1e-4 and v["pressure_native_range"][1] <= 2000 + 1e-2
              and v["hws_native_range"][0] >= 310 - 1e-4 and v["hws_native_range"][1] <= 350 + 1e-4
              and v["forced_off_while_occupied_or_prestart"] == 0 and v["forced_off_consistent"]
              and v["nonknob_action_changes_without_internal_knobs"] == 0
              for k, v in val.items() if k in ("moderate", "mild"))
    check("6_randomized_action_validity", ok6, **val)

    # ---------------------------------------------------------------- 7. torch loading
    import torch
    from dataset import TransitionWindows, load_manifest
    rows_a = [r for r in load_manifest(OUT / "run_a") if r["profile"] in ("moderate", "baseline")]
    ds = TransitionWindows(OUT / "run_a", rows_a, horizon=12, stride=4, cache_dir=OUT / "cache")
    tds = ds.as_torch()
    loader = torch.utils.data.DataLoader(tds, batch_sampler=ds.scenario_batches(32, seed=0), num_workers=2)
    batch = next(iter(loader))
    shapes = {k: (list(v.shape), str(v.dtype)) for k, v in batch.items() if hasattr(v, "shape")}
    ok7 = (shapes["obs"][0][1:] == [13, smeta["obs_dim"]] and shapes["action_continuous"][0][1:] == [12, len(cont_names)]
           and shapes["action_discrete"][0][1:] == [12, len(disc_names)] and shapes["obs"][1] == "torch.float32"
           and torch.equal(batch["obs"][0, 1:], torch.from_numpy(
               np.asarray(ds.episodes[int(batch["episode"][0])].arrays["obs"][int(batch["t0"][0]) + 1:int(batch["t0"][0]) + 13]))))
    check("7_torch_loading", ok7, torch=torch.__version__, windows=len(ds), batch_shapes=shapes)

    # ---------------------------------------------------------------- 8. baseline unchanged
    diff = subprocess.run(["git", "-C", str(ROOT), "diff", "--stat", "HEAD", "--", "src/run_eval.py",
                           "src/baseline_policy.py", "src/summarize.py"], capture_output=True, text=True).stdout.strip()
    null = load_episode(_ep_path("run_a", "null", 1))
    null_eq = all(np.array_equal(null[k], base[k]) for k in
                  ("obs", "action_continuous", "action_discrete", "reward", "zone_info", "tank_state"))
    info8 = {"tracked_sources_diff": diff or "none", "null_profile_equals_baseline": null_eq}
    full_ok = None
    full_dir = a.full_baseline
    full_eps = sorted((full_dir / "episodes").glob("*.npz")) if full_dir.exists() else []
    if full_eps:
        fe = load_episode(full_eps[0])
        ref_json = json.loads((ROOT / "results/baseline_full/baseline/building_train_8__atlanta_1.json").read_text())
        ref = np.load(ROOT / "results/baseline_full/baseline/building_train_8__atlanta_1.rewards.npy")
        full_ok = (fe["reward"].shape[0] == 8640 and np.array_equal(fe["reward"].astype(np.float32), ref[0])
                   and abs(fe["reward"].mean() - ref_json["mean_reward"]) < 1e-12)
        info8.update(full_steps=int(fe["reward"].shape[0]), full_mean_reward=float(fe["reward"].mean()),
                     stored_mean_reward=ref_json["mean_reward"],
                     per_step_rewards_equal_stored=bool(np.array_equal(fe["reward"].astype(np.float32), ref[0])),
                     full_policy_ms_mean=fe["meta"]["policy_ms_mean"], full_sim_ms_mean=fe["meta"]["sim_ms_mean"],
                     full_wall_s=fe["meta"]["wall_s"])
    else:
        info8["full_horizon"] = f"not found under {full_dir} (run collect_data.py --profiles baseline --steps 8640)"
    check("8_baseline_unchanged", not diff and null_eq and full_ok is not False, **info8)

    out = RESULTS / "validation.json"
    out.write_text(json.dumps(CHECKS, indent=1, default=float))
    print(f"\n{sum(c['ok'] for c in CHECKS.values())}/{len(CHECKS)} checks passed -> {out}")


if __name__ == "__main__":
    main()
