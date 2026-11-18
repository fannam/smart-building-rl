"""Exploration audit: replay the requested knob schedule from the seed (draws are
observation-independent), compare with applied knobs and effective native action changes."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dataset import load_episode, load_manifest, load_scenario_meta
from building_meta import ActionCodec
from exploration import Explorer, ExplorationConfig, _Segment, episode_rng, PROFILES

ROOTS = [ROOT / "data/phase1_sample", ROOT / "scratch/phase1_audit/out/mild_sample",
         ROOT / "scratch/phase1_audit/out/validation_data/run_a"]

class RecSeg(_Segment):
    def __init__(self, prob, hold):
        super().__init__(prob, hold); self.log = []
    def advance(self, rng, sampler):
        if self.remaining <= 0:
            super().advance(rng, sampler); self.log.append((self.remaining + 1, self.active, self.payload))
        else:
            super().advance(rng, sampler)

def replay(cfg, scenario, seed, episode, K, V, T):
    ex = Explorer.__new__(Explorer); ex.cfg, ex.K, ex.V = cfg, K, V
    rng = episode_rng(seed, scenario, episode)
    segs = {"ahu": [RecSeg(cfg.ahu_prob, cfg.ahu_hold) for _ in range(K)],
            "sat": [RecSeg(cfg.sat_prob, cfg.sat_hold) for _ in range(K)],
            "pressure": [RecSeg(cfg.pressure_prob, cfg.pressure_hold) for _ in range(K)],
            "hws": [RecSeg(cfg.hws_prob, cfg.hws_hold)], "setback": [RecSeg(cfg.setback_prob, cfg.setback_hold)],
            "precondition": [RecSeg(cfg.precondition_prob, cfg.precondition_hold)]}
    samp = {"ahu": ex._s_ahu, "sat": ex._s_sat, "pressure": ex._s_p, "hws": ex._s_hws,
            "setback": ex._s_setback, "precondition": ex._s_pre}
    active = {k: np.zeros((T, len(v)), bool) for k, v in segs.items()}
    payload = {k: [[None] * T for _ in v] for k, v in segs.items()}
    for t in range(T):
        for k in ("ahu", "sat", "pressure", "hws", "setback", "precondition"):
            for i, s in enumerate(segs[k]):
                s.advance(rng, samp[k]); active[k][t, i] = s.active; payload[k][i][t] = s.payload
    return segs, active, payload

stats = defaultdict(lambda: defaultdict(list))
div = defaultdict(lambda: defaultdict(list))
consistency_errors = []
for root in ROOTS:
    if not (root / "manifest.jsonl").exists(): continue
    for r in load_manifest(root):
        ep = load_episode(root / r["path"]); sm = load_scenario_meta(root, r["scenario"])
        T, K, V = ep["reward"].shape[0], sm["num_ahus"], sm["num_zones"]
        ac = ActionCodec(sm); names = sm["observation_field_names"]; ah = sm["ahus"]; hw = sm["hws"]
        a_n, b_n = ac.to_native(ep["action_continuous"]), ac.to_native(ep["baseline_continuous"])
        run_e = ep["action_discrete"][:, ah["act_run"]].astype(bool); run_b = ep["baseline_discrete"][:, ah["act_run"]].astype(bool)
        sat = a_n[:, ah["act_sat"]]; pres = a_n[:, ah["act_static_pressure"]]; sw = a_n[:, hw["act_supply_water_temp"]]
        damp = a_n[:, sm["zones"]["act_damper"]]; valve = a_n[:, sm["zones"]["act_reheat_valve"]]
        prof = r["profile"]; key = f"{prof}|K{K}"
        div[key]["sat_std_running"].append(float(sat[run_e].std()) if run_e.any() else np.nan)
        div[key]["sat_2K_bins_visited"].append(int(len(np.unique(np.floor((sat[run_e] - 285) / 2)))) if run_e.any() else 0)
        div[key]["pressure_p5_p95_running"].append([float(np.percentile(pres[run_e], 5)), float(np.percentile(pres[run_e], 95))] if run_e.any() else None)
        div[key]["pressure_below_200_while_running_frac"].append(float(np.mean(pres[run_e] < 200 - 1e-3)) if run_e.any() else 0.0)
        div[key]["hws_values_distinct_0.5K"].append(int(len(np.unique(np.round(sw * 2)))))
        div[key]["hws_std"].append(float(sw.std()))
        div[key]["ahu_run_frac"].append(float(run_e.mean()))
        div[key]["damper_mean_std"].append(float(damp.std()))
        div[key]["valve_mean_std"].append(float(valve.std()))
        div[key]["damper_differs_from_b_frac"].append(float(np.mean(np.any(ep["action_continuous"][:, sm["zones"]["act_damper"]] != ep["baseline_continuous"][:, sm["zones"]["act_damper"]], 1))))
        div[key]["perturbed_frac"].append(float(ep["perturbed"].mean()))
        if prof == "baseline":
            continue
        cfg = ExplorationConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ep["meta"]["exploration_config"].items()})
        segs, act, pay = replay(cfg, r["scenario"], r["seed"], r["episode_index"], K, V, T)
        kn = {n: ep["knobs"][:, i] for i, n in enumerate(ep["names_knobs"])}
        now = ep["obs"][:-1, names.index("comfort_mode_now")] > .5; soon = ep["obs"][:-1, names.index("comfort_mode_soon")] > .5
        unocc = ~now & ~soon
        # consistency of applied knobs with requested schedule
        for k in range(K):
            ap = kn[f"sat_offset_{k}"] != 0
            if np.any(ap & ~act["sat"][:, k]): consistency_errors.append((r["episode_id"], "sat applied while inactive"))
            req = np.array([p if p is not None else 0.0 for p in pay["sat"][k]])
            if np.any(np.abs(kn[f"sat_offset_{k}"][ap] - req[ap]) > 1e-4): consistency_errors.append((r["episode_id"], "sat value"))
        P = f"{prof}"
        for name, a in act.items():
            stats[P][f"{name}_requested_active_frac"].append(float(a.mean()))
            segl = [s.log for s in segs[name]]
            stats[P][f"{name}_segments_active_frac"].append(float(np.mean([x[1] for l in segl for x in l])))
            stats[P][f"{name}_hold_active_mean"].extend([x[0] for l in segl for x in l if x[1]])
        # effective application
        sat_req = act["sat"]; p_req = act["pressure"]
        guard = np.stack([kn[f"guard_ahu_{k}"] for k in range(K)], 1) > 0
        stats[P]["sat_applied_frac"].append(float(np.mean(np.stack([kn[f"sat_offset_{k}"] != 0 for k in range(K)], 1))))
        stats[P]["sat_requested_but_ahu_off_frac"].append(float(np.mean(sat_req & ~run_e)))
        stats[P]["pressure_applied_frac"].append(float(np.mean(np.stack([kn[f"pressure_scale_{k}"] != 1 for k in range(K)], 1))))
        dsat = (sat - b_n[:, ah["act_sat"]])[run_e & run_b]
        stats[P]["sat_effective_delta_K"].extend(dsat[np.abs(dsat) > 1e-4].tolist())
        dp = (pres - b_n[:, ah["act_static_pressure"]])[run_e & run_b]
        stats[P]["pressure_effective_delta_Pa"].extend(dp[np.abs(dp) > 1e-2].tolist())
        dsw = sw - b_n[:, hw["act_supply_water_temp"]]
        stats[P]["hws_applied_frac"].append(float(np.mean(kn["hws_offset"] != 0)))
        stats[P]["hws_effective_frac"].append(float(np.mean(np.abs(dsw) > 1e-3)))
        stats[P]["hws_effective_delta_K"].extend(dsw[np.abs(dsw) > 1e-3].tolist())
        stats[P]["hws_offset_clipped_to_noop_frac"].append(float(np.mean((kn["hws_offset"] != 0) & (np.abs(dsw) <= 1e-3))))
        stats[P]["ahu_forced_on_frac"].append(float(np.mean(np.stack([kn[f"ahu_force_{k}"] == 1 for k in range(K)]))))
        stats[P]["ahu_forced_off_frac"].append(float(np.mean(np.stack([kn[f"ahu_force_{k}"] == 0 for k in range(K)]))))
        stats[P]["ahu_force_off_requested_frac"].append(float(np.mean([act["ahu"][t, k] and pay["ahu"][k][t][0] == 0 for t in range(T) for k in range(K)])))
        stats[P]["ahu_force_off_requested_while_baseline_running_unocc_frac"].append(float(np.mean([act["ahu"][t, k] and pay["ahu"][k][t][0] == 0 and run_b[t, k] and unocc[t] for t in range(T) for k in range(K)])))
        stats[P]["setback_requested_frac"].append(float(act["setback"].mean()))
        stats[P]["setback_effective_frac(unoccupied)"].append(float(np.mean(act["setback"][:, 0] & unocc)))
        stats[P]["setback_low_K"].extend(kn["setback_low"][kn["setback_low"] != 0].tolist())
        stats[P]["setback_high_K"].extend(kn["setback_high"][kn["setback_high"] != 0].tolist())
        prestart = soon & ~now
        pre_changed = kn["precondition_h"] != 1.0
        stats[P]["precondition_changed_frac"].append(float(pre_changed.mean()))
        stats[P]["precondition_changed_during_prestart_frac_of_steps"].append(float(np.mean(pre_changed & prestart)))
        stats[P]["precondition_values_h"].extend(np.unique(kn["precondition_h"][pre_changed]).tolist())
        anyreq = (sat_req | p_req | act["ahu"])
        stats[P]["guard_ahu_frac_all_steps"].append(float(guard.mean()))
        stats[P]["guard_ahu_frac_of_requested"].append(float(guard[anyreq].mean()) if anyreq.any() else 0.0)
        stats[P]["guard_hws_frac"].append(float(np.mean(kn["guard_hws"] > 0)))
        stats[P]["internal_knob_steps_frac"].append(float(np.mean((kn["setback_low"] != 0) | pre_changed)))

def summ(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v: return None
    if isinstance(v[0], list): return v
    a = np.asarray(v, float)
    return {"n": int(a.size), "mean": float(a.mean()), "p5": float(np.percentile(a, 5)), "p50": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "min": float(a.min()), "max": float(a.max())}
out = {"configured": {p: PROFILES[p].to_dict() for p in ("mild", "moderate")},
       "realized": {p: {k: summ(v) for k, v in d.items()} for p, d in stats.items()},
       "diversity": {p: {k: (summ(v) if not (v and isinstance(v[0], list)) else
                             [float(np.mean([x[0] for x in v if x])), float(np.mean([x[1] for x in v if x]))]) for k, v in d.items()}
                     for p, d in sorted(div.items())},
       "consistency_errors": consistency_errors}
(ROOT / "scratch/phase1_audit/out/exploration_audit.json").write_text(json.dumps(out, indent=1))
for p, d in out["realized"].items():
    print("==", p, "episodes:", d["sat_applied_frac"]["n"])
    for k, v in d.items():
        if v: print(f"  {k:58s} mean {v['mean']:+.4f} p5 {v['p5']:+.3f} p50 {v['p50']:+.3f} p95 {v['p95']:+.3f} [{v['min']:+.3f},{v['max']:+.3f}] n={v['n']}")
print("consistency errors:", consistency_errors[:5])
for p, d in out["diversity"].items():
    print(p, {k: (round(v["mean"], 3) if isinstance(v, dict) else [round(x, 1) for x in v]) for k, v in d.items()})
