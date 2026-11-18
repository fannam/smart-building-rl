"""Sensor-group lag diagnostics on existing Phase 1 episodes (read-only).

Convention: a privileged quantity indexed by transition t (zone_info[t], action a_t,
boiler_info[t], tank_state[t+1]) is compared with obs[t + d] for d in OFFSETS.
"exact" = max |diff| below a tolerance tied to float32 storage.
Outside-air temperature is compared against the scenario's own weather controller
evaluated at sim_timestamp[i] + k*300 s (env constructed, never stepped beyond reset)."""
import json, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dataset import load_episode, load_manifest, load_scenario_meta
from building_meta import ObservationCodec, ActionCodec

DATA = ROOT / "data/phase1_sample"
OFFSETS = (-1, 0, 1, 2)

def cmp(obs_vals, priv, offsets=OFFSETS):
    """obs_vals: [T+1, n] decoded; priv: [T, n] indexed by transition t."""
    T = priv.shape[0]; out = {}
    for d in offsets:
        lo, hi = max(0, -d), min(T, T + 1 - d)
        a = obs_vals[lo + d: hi + d]; b = priv[lo:hi]
        diff = np.abs(a - b)
        scale = max(1e-9, float(np.abs(b).max()))
        c = np.corrcoef(a.ravel(), b.ravel())[0, 1] if a.std() > 0 and b.std() > 0 else float("nan")
        out[d] = {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()), "max_rel": float(diff.max() / scale),
                  "corr": float(c), "frac_exact_1e-4rel": float(np.mean(diff <= 1e-4 * scale + 1e-6))}
    return out

agg = {}
def add(group, res):
    g = agg.setdefault(group, {})
    for d, v in res.items():
        cur = g.setdefault(d, {"max_abs": 0, "max_rel": 0, "mean_abs": [], "corr": [], "frac_exact": []})
        cur["max_abs"] = max(cur["max_abs"], v["max_abs"]); cur["max_rel"] = max(cur["max_rel"], v["max_rel"])
        cur["mean_abs"].append(v["mean_abs"]); cur["corr"].append(v["corr"]); cur["frac_exact"].append(v["frac_exact_1e-4rel"])

rows = load_manifest(DATA)
for r in rows:
    ep = load_episode(DATA / r["path"]); sm = load_scenario_meta(DATA, r["scenario"])
    oc, ac = ObservationCodec(sm), ActionCodec(sm); N = oc.native(ep["obs"]); names = sm["observation_field_names"]
    ah, z, hw = sm["ahus"], sm["zones"], sm["hws"]; ids = ah["ids"]
    zi = ep["zone_info"].astype(np.float64); zn = ep["names_zone_info"]
    flow_z = zi[..., zn.index("air_flow_rate")]
    a_nat = ac.to_native(ep["action_continuous"]); run = ep["action_discrete"][:, ah["act_run"]].astype(float)
    flow_ahu = np.stack([flow_z[:, g].sum(1) for g in ah["zones"]], 1)
    f = lambda suffix: N[:, [names.index(f"ahs_{a}_{suffix}") for a in ids]]
    add("ahu_supply_air_flowrate_sensor vs sum zone_info.air_flow_rate[t]", cmp(f("supply_air_flowrate_sensor"), flow_ahu))
    add("ahu_supply_air_flowrate_setpoint vs sum zone_info.air_flow_rate[t]", cmp(f("supply_air_flowrate_setpoint"), flow_ahu))
    rec_eff = np.clip(a_nat[:, ah["act_recirculation"]], 0, 0.7)
    add("ahu_outside_air_flowrate_sensor vs (1-recirc_eff[t])*sum flow[t]", cmp(f("outside_air_flowrate_sensor"), (1 - rec_eff) * flow_ahu))
    add("ahu_supply_air_temperature_sensor vs commanded SAT a_t", cmp(f("supply_air_temperature_sensor"), a_nat[:, ah["act_sat"]]))
    p_eff = run * np.maximum(a_nat[:, ah["act_static_pressure"]], 200.0)
    add("ahu_supply_air_static_pressure_sensor vs run*max(cmd,200)[t]", cmp(f("supply_air_static_pressure_sensor"), p_eff))
    add("ahu_recirculation echo vs clip(cmd,0,0.7)[t]", cmp(f("recirculation_percentage_command"), rec_eff))
    add("ahu_supervisor_run_command echo vs run[t]", cmp(f("supervisor_run_command"), run))
    add("ahu_supply_fan_run_command vs run[t]", cmp(f("supply_fan_run_command"), run))
    vflow = N[:, z["obs_flow_sp"]]
    add("vav_supply_air_flowrate_setpoint vs zone_info.air_flow_rate_setpoint[t]", cmp(vflow, zi[..., zn.index("air_flow_rate_setpoint")]))
    add("vav_supply_air_flowrate_setpoint vs zone_info.air_flow_rate[t]", cmp(vflow, flow_z))
    damp_eff = np.maximum(a_nat[:, z["act_damper"]], 0.1)
    add("vav_damper echo vs max(cmd,0.1)[t]", cmp(N[:, [names.index(f"{v}_supply_air_damper_percentage_command") for v in z["ids"]]], damp_eff))
    add("vav_zone_air_temperature_sensor vs zone_info.zone_air_temperature[t]", cmp(N[:, z["obs_temp"]], zi[..., zn.index("zone_air_temperature")]))
    add("hws_supply_water_temperature_sensor vs tank_state[t+1].tank_T", cmp(N[:, [hw["obs_supply_water_temp"]]], ep["tank_state"][1:, :1]))
    add("hws_supply_water_temperature_setpoint echo vs cmd[t]", cmp(N[:, [names.index("hws_supply_water_temperature_setpoint")]], a_nat[:, [hw["act_supply_water_temp"]]]))
    add("hws_differential_pressure_sensor vs cmd dp[t]", cmp(N[:, [names.index("hws_differential_pressure_sensor")]], a_nat[:, [hw["act_differential_pressure"]]]))
    add("hws_differential_pressure_setpoint echo vs cmd dp[t]", cmp(N[:, [names.index("hws_differential_pressure_setpoint")]], a_nat[:, [hw["act_differential_pressure"]]]))
    occ = zi[..., zn.index("average_occupancy")].sum(-1, keepdims=True)
    add("num_occupants vs sum zone_info.average_occupancy[t]", cmp(oc.field(ep["obs"], "num_occupants")[:, None], occ))
    add("hws_run_status vs hws run cmd[t]", cmp(N[:, [names.index("hws_run_status")]], ep["action_discrete"][:, [hw["act_run"]]].astype(float)))

# Outside air temperature vs weather controller (6 scenarios, one per building, 2016 steps)
os.chdir(ROOT / "starter-kit")
import pandas as pd
from collect_data import build_env
oat = {}
for r in [x for x in rows if x["profile"] == "baseline"][::5]:
    ep = load_episode(DATA / r["path"], keys=["obs", "sim_timestamp"]); sm = load_scenario_meta(DATA, r["scenario"])
    env, meta, _ = build_env(r["scenario"]); wc = env.building.simulator._weather_controller
    names = sm["observation_field_names"]; ids = sm["ahus"]["ids"]
    o = np.stack([ep["obs"][:, names.index(f"ahs_{a}_outside_air_temperature_sensor")] for a in ids], 1).astype(np.float64)
    same_across_ahus = float(np.max(np.abs(o - o[:, :1])))
    ts = pd.to_datetime(ep["sim_timestamp"], unit="s")
    res = {}
    for k in (-2, -1, 0, 1, 2):
        w = np.array([wc.get_current_temp(t + pd.Timedelta(seconds=300 * k)) for t in ts])
        d = np.abs(o[:, 0] - w); res[k] = {"max_abs": float(d.max()), "mean_abs": float(d.mean())}
    oat[r["scenario"]] = {"max_spread_across_ahus": same_across_ahus, "vs_weather_at_sim_ts[i]+k*300s": res}
    print(r["scenario"], {k: round(v["max_abs"], 6) for k, v in res.items()}, flush=True)

summary = {g: {d: {"max_abs": v["max_abs"], "max_rel": v["max_rel"], "mean_abs": float(np.mean(v["mean_abs"])),
                   "corr_mean": float(np.nanmean(v["corr"])) if not np.all(np.isnan(v["corr"])) else None,
                   "frac_exact_min": float(np.min(v["frac_exact"]))} for d, v in res.items()} for g, res in agg.items()}
out = {"episodes": len(rows), "offset_convention": "obs[t+d] vs privileged[t]", "groups": summary, "outside_air_temperature": oat}
(ROOT / "scratch/phase1_5/sensor_lag.json").write_text(json.dumps(out, indent=1))
for g, res in summary.items():
    best = min(res, key=lambda d: res[d]["max_rel"])
    print(f"{g:70s} " + " ".join(f"d={d}:{res[d]['max_rel']:.1e}" for d in res) + f"  -> best d={best}")
