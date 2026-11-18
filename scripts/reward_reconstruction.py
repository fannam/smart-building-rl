"""Reward reconstruction and zone-occupancy-weighting study over a Phase 1 dataset.

For each episode, recompute the per-step reward with `src/reward_model.py`
under progressively more causal assumptions and compare with the simulator's
reward:

  exact        privileged: reward-info zone temps/setpoints, true per-zone
               occupancy, true month/work-day
  gas_mean     as exact, annual-mean gas price (month unobservable)
  occ_uniform  as exact, per-zone occupancy = uniform split of the total
  occ_area     as exact, per-zone occupancy = area-weighted split of the total
  obs_lag1     as exact, zone temps/setpoints decoded from o_{t+1} (WRONG alignment,
               kept to quantify it: sensors in o_{t+1} are one step stale)
  obs_temps    as exact, zone temps/setpoints decoded from o_{t+2}
  causal       everything observable: temps/setpoints/num_occupants from o_{t+2}
               (the sensor state that produced r_t), uniform occupancy split,
               hour/day-of-week from the time features of o_{t+1} (not stale),
               work-day = Mon-Fri (holidays unknown), annual-mean gas price.
               Energy E, G stay privileged (what the dynamics model must predict).

Alignment (verified in validate_phase1.py): RewardInfo zone temperatures,
setpoints and total occupancy of step t equal the sensor values in o_{t+2};
the time features in o_{t+1} equal the reward interval start. For the last
step o_{T+1} does not exist, so obs-based variants use o_T there.

    .venv/bin/python scripts/reward_reconstruction.py --data data/phase1_sample \
        --out results/phase1/reward_reconstruction.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import reward_model as rm  # noqa: E402
from building_meta import ObservationCodec  # noqa: E402
from dataset import load_episode, load_manifest, load_scenario_meta  # noqa: E402

VARIANTS = ("exact", "gas_mean", "occ_uniform", "occ_area", "obs_lag1", "obs_temps", "causal")


def physical_inputs(ep: dict) -> dict[str, np.ndarray]:
    zn = ep["names_zone_info"]
    z = ep["zone_info"].astype(np.float64)
    a = ep["ahu_info"].astype(np.float64)
    b = ep["boiler_info"].astype(np.float64)
    h = ep["heat_pump_info"].astype(np.float64)
    elec = (a[..., 0].sum(-1) + np.abs(a[..., 1]).sum(-1) + b[..., 1].sum(-1)
            + np.abs(h).sum((-1, -2)))
    gas = b[..., 0].sum(-1)
    cal = {n: ep["calendar"][:, i] for i, n in enumerate(ep["names_calendar"])}
    return {"heat": z[..., zn.index("heating_setpoint_temperature")],
            "cool": z[..., zn.index("cooling_setpoint_temperature")],
            "temp": z[..., zn.index("zone_air_temperature")],
            "occ": z[..., zn.index("average_occupancy")],
            "elec": elec, "gas": gas, **cal}


def reconstruct(ep: dict, smeta: dict) -> dict[str, dict[str, np.ndarray]]:
    p = physical_inputs(ep)
    K = smeta["num_ahus"]
    V = smeta["num_zones"]
    codec = ObservationCodec(smeta)
    nxt = ep["obs"][1:]
    sensed = np.concatenate([ep["obs"][2:], ep["obs"][-1:]])  # o_{t+2}; o_T for the last step
    total_occ = p["occ"].sum(-1)
    area = np.asarray(smeta["zone_area_m2"])
    lag1_temp = codec.zone_temperatures(nxt)
    lag1_heat, lag1_cool = codec.zone_bands(nxt)
    obs_temp = codec.zone_temperatures(sensed)
    obs_heat, obs_cool = codec.zone_bands(sensed)
    n_occ_obs = codec.field(sensed, "num_occupants")
    hour_obs = np.floor(codec.hour(nxt) + 1e-6).astype(int) % 24
    dow_ang = np.mod(np.arctan2(nxt[:, codec.index["dow_sin_000"]], nxt[:, codec.index["dow_cos_000"]]), 2 * np.pi)
    dow_obs = np.round(dow_ang * 7 / (2 * np.pi)).astype(int) % 7  # 0 = Monday (pandas dayofweek)
    wd_obs = dow_obs < 5

    def run(temp, heat, cool, occ, hour, wd, month):
        return rm.step_reward(temp, heat, cool, occ, p["elec"], p["gas"], hour, wd, K, month)

    month = p["month"]
    out = {
        "exact": run(p["temp"], p["heat"], p["cool"], p["occ"], p["hour"], p["is_workday"], month),
        "gas_mean": run(p["temp"], p["heat"], p["cool"], p["occ"], p["hour"], p["is_workday"], None),
        "occ_uniform": run(p["temp"], p["heat"], p["cool"],
                           rm.occupancy_weights("uniform", V, total_occ), p["hour"], p["is_workday"], month),
        "occ_area": run(p["temp"], p["heat"], p["cool"],
                        rm.occupancy_weights("area", V, total_occ, area), p["hour"], p["is_workday"], month),
        "obs_lag1": run(lag1_temp, lag1_heat, lag1_cool, p["occ"], p["hour"], p["is_workday"], month),
        "obs_temps": run(obs_temp, obs_heat, obs_cool, p["occ"], p["hour"], p["is_workday"], month),
        "causal": run(obs_temp, obs_heat, obs_cool, rm.occupancy_weights("uniform", V, n_occ_obs),
                      hour_obs, wd_obs, None),
    }
    out["_aux"] = {"total_occ": total_occ, "num_occupants_obs": n_occ_obs, "hour_obs": hour_obs,
                   "dow_obs": dow_obs, "hour_true": p["hour"], "dow_true": p["weekday"]}
    return out


def compare(ep: dict, rec: dict) -> dict:
    r = ep["reward"]
    rr = {n: ep["reward_response"][:, i] for i, n in enumerate(ep["names_reward_response"])}
    occupied = rec["_aux"]["total_occ"] > 0
    res = {"steps": int(r.size), "occupied_steps": int(occupied.sum()),
           "true_mean_reward": float(r.mean()),
           "true_mean_comfort": float(0.6 * rr["normalized_productivity_regret"].mean()),
           "step_reward_equals_response": bool(np.array_equal(r, rr["agent_reward_value"]))}
    for v in VARIANTS:
        e = rec[v]["reward"] - r
        comfort_err = rec[v]["normalized_productivity_regret"] - rr["normalized_productivity_regret"]
        res[v] = {
            "max_abs_err": float(np.max(np.abs(e))),
            "mean_abs_err": float(np.mean(np.abs(e))),
            "episode_mean_err": float(e.mean()),
            "episode_mean_rel_err": float(e.mean() / abs(r.mean())) if r.mean() else 0.0,
            "comfort_mae_occupied": float(np.mean(np.abs(comfort_err[occupied]))) if occupied.any() else 0.0,
            "comfort_mean_err": float(0.6 * comfort_err.mean()),
            "energy_max_abs_err": float(np.max(np.abs(rec[v]["normalized_energy_cost"] - rr["normalized_energy_cost"]))),
            "carbon_max_abs_err": float(np.max(np.abs(rec[v]["normalized_carbon_emission"] - rr["normalized_carbon_emission"]))),
        }
    aux = rec["_aux"]
    res["num_occupants_obs_vs_total_avg_occ"] = {
        "max_abs_diff": float(np.max(np.abs(aux["num_occupants_obs"] - aux["total_occ"]))),
        "gate_agreement": float(np.mean((aux["num_occupants_obs"] > 0) == occupied)),
    }
    res["hour_from_obs_matches_reward_hour"] = float(np.mean(aux["hour_obs"] == aux["hour_true"]))
    res["dow_from_obs_matches_reward_dow"] = float(np.mean(aux["dow_obs"] == aux["dow_true"]))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rows = load_manifest(a.data)
    results = []
    for r in rows:
        ep = load_episode(a.data / r["path"])
        smeta = load_scenario_meta(a.data, r["scenario"])
        res = compare(ep, reconstruct(ep, smeta))
        res.update(episode_id=r["episode_id"], scenario=r["scenario"], profile=r["profile"])
        results.append(res)
        print(f"{r['episode_id']:55s} exact={res['exact']['max_abs_err']:.1e} "
              f"unif={res['occ_uniform']['comfort_mean_err']:+.2e} area={res['occ_area']['comfort_mean_err']:+.2e} "
              f"causal_rel={res['causal']['episode_mean_rel_err']:+.3f}")
    agg = {}
    for v in VARIANTS:
        agg[v] = {
            "max_abs_err": max(x[v]["max_abs_err"] for x in results),
            "mean_abs_err": float(np.mean([x[v]["mean_abs_err"] for x in results])),
            "mean_abs_episode_rel_err": float(np.mean([abs(x[v]["episode_mean_rel_err"]) for x in results])),
            "max_abs_episode_rel_err": max(abs(x[v]["episode_mean_rel_err"]) for x in results),
            "mean_comfort_mae_occupied": float(np.mean([x[v]["comfort_mae_occupied"] for x in results])),
            "mean_abs_comfort_bias": float(np.mean([abs(x[v]["comfort_mean_err"]) for x in results])),
        }
    summary = {"episodes": len(results), "aggregate": agg, "per_episode": results}
    print(json.dumps(agg, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
