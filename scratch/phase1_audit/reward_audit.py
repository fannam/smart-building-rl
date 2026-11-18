"""Per-step reward reconstruction metrics over the Phase 1 sample (read-only)."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import reward_model as rm
from reward_reconstruction import reconstruct, physical_inputs
from dataset import load_episode, load_manifest, load_scenario_meta

DATA = ROOT / "data/phase1_sample"
VARS = ["exact", "gas_mean", "workday_mon_fri", "occ_uniform", "occ_area", "obs_lag1", "obs_temps", "causal"]
err = defaultdict(list); comp = defaultdict(lambda: defaultdict(list))
by_b = defaultdict(lambda: defaultdict(list)); by_w = defaultdict(lambda: defaultdict(list))
trans = defaultdict(lambda: {"near": [], "far": []})
occ_comfort = defaultdict(list); ep_rel = defaultdict(list)
for r in load_manifest(DATA):
    ep = load_episode(DATA / r["path"]); sm = load_scenario_meta(DATA, r["scenario"])
    rec = reconstruct(ep, sm); p = physical_inputs(ep)
    rec["workday_mon_fri"] = rm.step_reward(p["temp"], p["heat"], p["cool"], p["occ"], p["elec"], p["gas"],
                                            p["hour"], p["weekday"] < 5, sm["num_ahus"], p["month"])
    true = ep["reward"]; rr = {n: ep["reward_response"][:, i] for i, n in enumerate(ep["names_reward_response"])}
    tparts = {"comfort": 0.6 * rr["normalized_productivity_regret"], "energy": -0.2 * rr["normalized_energy_cost"],
              "carbon": -0.2 * rr["normalized_carbon_emission"]}
    occ = p["occ"].sum(-1) > 0
    now = ep["obs"][1:, sm["observation_field_names"].index("comfort_mode_now")] > .5
    edge = np.zeros(len(true), bool)
    for sig in (occ, now):
        ch = np.flatnonzero(np.diff(sig.astype(int)) != 0)
        for c in ch:
            edge[max(0, c - 3):c + 5] = True
    b, w = r["building"], r["weather"].rstrip("0123456789").rstrip("_")
    for v in VARS:
        e = rec[v]["reward"] - true
        err[v].append(e); by_b[v][b].append(e); by_w[v][w].append(e)
        trans[v]["near"].append(e[edge]); trans[v]["far"].append(e[~edge])
        for c in tparts:
            comp[v][c].append(rec[v][c] - tparts[c])
        ep_rel[v].append(e.mean() / abs(true.mean()))
        occ_comfort[v].append((rec[v]["comfort"] - tparts["comfort"])[occ])

def stats(a):
    a = np.abs(np.concatenate(a)) if isinstance(a, list) else np.abs(a)
    return {"n": int(a.size), "mae": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}
out = {}
for v in VARS:
    out[v] = {"all": stats(err[v]),
              "by_component": {c: stats(comp[v][c]) for c in comp[v]},
              "comfort_on_occupied_steps": stats(occ_comfort[v]),
              "by_building": {b: stats(x) for b, x in sorted(by_b[v].items())},
              "by_weather": {w: stats(x) for w, x in sorted(by_w[v].items())},
              "near_transitions": stats(trans[v]["near"]), "away_from_transitions": stats(trans[v]["far"]),
              "episode_mean_rel_err": {"mean_abs": float(np.mean(np.abs(ep_rel[v]))), "max_abs": float(np.max(np.abs(ep_rel[v]))),
                                       "mean_signed": float(np.mean(ep_rel[v]))}}
(ROOT / "scratch/phase1_audit/out/reward_audit.json").write_text(json.dumps(out, indent=1))
for v in VARS:
    a = out[v]
    print(f"{v:16s} MAE {a['all']['mae']:.2e} med {a['all']['median']:.2e} p95 {a['all']['p95']:.2e} max {a['all']['max']:.2e} | "
          f"comp max c/e/k {a['by_component']['comfort']['max']:.1e}/{a['by_component']['energy']['max']:.1e}/{a['by_component']['carbon']['max']:.1e} | "
          f"near/far MAE {a['near_transitions']['mae']:.1e}/{a['away_from_transitions']['mae']:.1e} | ep rel mean {a['episode_mean_rel_err']['mean_abs']:.4f} max {a['episode_mean_rel_err']['max_abs']:.4f} signed {a['episode_mean_rel_err']['mean_signed']:+.4f}")
for v in ("exact", "occ_uniform", "occ_area", "causal"):
    print(v, "by building MAE:", {b: f"{s['mae']:.1e}" for b, s in out[v]["by_building"].items()})
    print(v, "by weather  MAE:", {w: f"{s['mae']:.1e}" for w, s in out[v]["by_weather"].items()})
