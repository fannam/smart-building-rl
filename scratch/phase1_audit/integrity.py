"""Read-only integrity / alignment scan of every collected episode."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dataset import load_episode, load_manifest, load_scenario_meta
from building_meta import ObservationCodec, ActionCodec

ROOTS = [ROOT / "data/phase1_sample", ROOT / "data/phase1_validation/run_a", ROOT / "data/phase1_validation/full_baseline"]

def scan(root, path, smeta):
    ep = load_episode(path); m = ep["meta"]; T = ep["reward"].shape[0]
    oc, ac = ObservationCodec(smeta), ActionCodec(smeta)
    names = smeta["observation_field_names"]; cn = smeta["continuous_action_names"]; dn = smeta["discrete_action_names"]
    issues = []
    D, nc, nd, V = smeta["obs_dim"], len(cn), len(dn), smeta["num_zones"]
    exp_shapes = {"obs": (T+1, D), "action_continuous": (T, nc), "action_discrete": (T, nd),
                  "baseline_continuous": (T, nc), "baseline_discrete": (T, nd), "perturbed": (T,),
                  "reward_response": (T, 17), "zone_info": (T, V, 6), "sim_timestamp": (T+1,),
                  "reward_timestamp": (T, 2), "calendar": (T, 6), "tank_state": (T+1, 6), "done": (T, 2)}
    for k, s in exp_shapes.items():
        if ep[k].shape != s: issues.append(f"shape {k} {ep[k].shape} != {s}")
    if m["steps"] != T: issues.append("meta steps mismatch")
    nonfinite = {k: int((~np.isfinite(v)).sum()) for k, v in ep.items()
                 if isinstance(v, np.ndarray) and v.dtype.kind == "f" and (~np.isfinite(v)).any()}
    if nonfinite: issues.append(f"nonfinite {nonfinite}")
    dtypes = {k: ep[k].dtype.str for k in ("obs", "action_continuous", "action_discrete", "reward")}
    rng = {"cont_abs_max": float(np.abs(ep["action_continuous"]).max()),
           "disc_values": sorted(np.unique(ep["action_discrete"]).tolist()),
           "base_cont_abs_max": float(np.abs(ep["baseline_continuous"]).max())}
    st, rts = ep["sim_timestamp"], ep["reward_timestamp"]
    time_ok = bool(np.all(np.diff(st) == 300) and np.all(rts[:, 0] == st[1:]) and np.all(rts[:, 1] - rts[:, 0] == 300))
    hour = oc.hour(ep["obs"][1:]); hc = ep["calendar"][:, 0] + ep["calendar"][:, 1] / 60
    hour_err = float(np.max(np.abs(((hour - hc) + 12) % 24 - 12)))
    rr0 = ep["reward_response"][:, ep["names_reward_response"].index("agent_reward_value")]
    reward_eq = bool(np.array_equal(ep["reward"], rr0))
    # echoes
    nxt = oc.native(ep["obs"][1:]); an = ac.to_native(ep["action_continuous"])
    def echo(sel):
        errs = [float(np.max(np.abs(nxt[:, names.index(cn[j])] - an[:, j]))) for j in sel if cn[j] in names]
        return max(errs) if errs else None
    ah, hw, z = smeta["ahus"], smeta["hws"], smeta["zones"]
    run = ep["action_discrete"][:, ah["act_run"]].astype(bool)
    sp_idx = [names.index(cn[j]) for j in ah["act_static_pressure"]]
    rc_idx = [names.index(cn[j]) for j in ah["act_recirculation"]]
    sp_echo, sp_cmd = nxt[:, sp_idx], an[:, ah["act_static_pressure"]]
    rc_echo, rc_cmd = nxt[:, rc_idx], an[:, ah["act_recirculation"]]
    echoes = {
        "sat": echo(ah["act_sat"]), "hws_supply_temp": echo([hw["act_supply_water_temp"]]),
        "hws_dp": echo([hw["act_differential_pressure"]]), "damper": echo(z["act_damper"]),
        "reheat_valve": echo(z["act_reheat_valve"]),
        "run_bits": float(np.max(np.abs((nxt[:, [names.index(n) for n in dn]] > 0).astype(int) - ep["action_discrete"]))),
        "static_pressure_when_run": float(np.max(np.abs(sp_echo - sp_cmd)[run])) if run.any() else None,
        "static_pressure_when_off_max_abs": float(np.max(np.abs(sp_echo[~run]))) if (~run).any() else None,
        "recirc_vs_min_cmd_0.7_when_run": float(np.max(np.abs(rc_echo - np.minimum(rc_cmd, 0.7))[run])) if run.any() else None,
        "recirc_when_off_vs_cmd": float(np.max(np.abs(rc_echo - rc_cmd)[~run])) if (~run).any() else None,
    }
    zi = ep["zone_info"].astype(np.float64); zn = ep["names_zone_info"]
    def lagerr(lag):
        o = ep["obs"][lag:]; n = min(len(o), T)
        return {"temp": float(np.max(np.abs(oc.zone_temperatures(o[:n]) - zi[:n, :, zn.index("zone_air_temperature")]))),
                "heat_sp": float(np.max(np.abs(oc.zone_bands(o[:n])[0] - zi[:n, :, 0]))),
                "cool_sp": float(np.max(np.abs(oc.zone_bands(o[:n])[1] - zi[:n, :, 1]))),
                "occupants": float(np.max(np.abs(oc.field(o[:n], "num_occupants") - zi[:n, :, 5].sum(-1))))}
    lag = {"obs_t+1": lagerr(1), "obs_t+2": lagerr(2)}
    done = ep["done"]
    term_rows = np.flatnonzero(done[:, 0]).tolist()
    trunc_rows = np.flatnonzero(done[:, 1]).tolist()
    gas = ep["boiler_info"][:, 0, 0].astype(np.float64)
    burner_next = ep["tank_state"][1:, 1]; burner_now = ep["tank_state"][:-1, 1]
    tank = {"gas_info_t_vs_tank_burner_t+1_max_rel": float(np.max(np.abs(gas - burner_next)) / max(1.0, gas.max())),
            "gas_info_t_vs_tank_burner_t_max_rel": float(np.max(np.abs(gas - burner_now)) / max(1.0, gas.max()))}
    pert_ok = bool(np.array_equal(ep["perturbed"], np.any(ep["action_continuous"] != ep["baseline_continuous"], 1)
                                  | np.any(ep["action_discrete"] != ep["baseline_discrete"], 1)))
    wd = ep["calendar"][:, 5].astype(bool); wk = ep["calendar"][:, 2] < 5
    out = {"episode_id": m["episode_id"], "root": root.name, "T": T, "profile": m["profile"], "issues": issues,
           "dtypes": dtypes, "ranges": rng, "time_ok": time_ok, "hour_err_h": hour_err, "reward_eq_response": reward_eq,
           "echo_max_err": echoes, "lag": lag, "terminated_rows": term_rows[:3], "n_terminated": len(term_rows),
           "n_truncated": len(trunc_rows), "tank": tank, "perturbed_flag_consistent": pert_ok,
           "holiday_weekday_steps": int(np.sum(wk & ~wd)),
           "baseline_profile_a_eq_b": bool(np.array_equal(ep["action_continuous"], ep["baseline_continuous"])) if m["profile"] == "baseline" else None}
    return out

res = []
obs_means = {}
for root in ROOTS:
    for r in load_manifest(root):
        if not (root / r["path"]).exists(): continue
        sm = load_scenario_meta(root, r["scenario"])
        obs_means.setdefault(r["building"], set()).add(tuple(np.round(sm["observation_mean"], 6)))
        res.append(scan(root, root / r["path"], sm))
agg = {
    "episodes": len(res), "episodes_with_issues": [x["episode_id"] for x in res if x["issues"]],
    "all_time_ok": all(x["time_ok"] for x in res), "max_hour_err_h": max(x["hour_err_h"] for x in res),
    "all_reward_eq_response": all(x["reward_eq_response"] for x in res),
    "all_perturbed_flag_consistent": all(x["perturbed_flag_consistent"] for x in res),
    "baseline_a_eq_b": all(x["baseline_profile_a_eq_b"] for x in res if x["baseline_profile_a_eq_b"] is not None),
    "max_cont_abs": max(x["ranges"]["cont_abs_max"] for x in res),
    "disc_values": sorted({v for x in res for v in x["ranges"]["disc_values"]}),
    "echo_max_err": {k: max((x["echo_max_err"][k] for x in res if x["echo_max_err"][k] is not None), default=None)
                     for k in res[0]["echo_max_err"]},
    "lag_obs_t+2_max": {k: max(x["lag"]["obs_t+2"][k] for x in res) for k in res[0]["lag"]["obs_t+2"]},
    "lag_obs_t+1_max": {k: max(x["lag"]["obs_t+1"][k] for x in res) for k in res[0]["lag"]["obs_t+1"]},
    "terminated": {x["episode_id"]: (x["T"], x["n_terminated"], x["terminated_rows"]) for x in res if x["n_terminated"]},
    "any_truncated": any(x["n_truncated"] for x in res),
    "tank_alignment_max_rel": {k: max(x["tank"][k] for x in res) for k in res[0]["tank"]},
    "holiday_weekday_steps_total": sum(x["holiday_weekday_steps"] for x in res),
    "distinct_obs_normalizations_per_building": {b: len(v) for b, v in obs_means.items()},
}
print(json.dumps(agg, indent=1))
(ROOT / "scratch/phase1_audit/out/integrity.json").write_text(json.dumps({"aggregate": agg, "episodes": res}, indent=1))
