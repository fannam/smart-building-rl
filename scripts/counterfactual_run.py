"""Phase 2B Stage 1b: paired simulator interventions (true counterfactuals, offline).

Per scenario: run the unmodified baseline policy from reset (bit-identical to the stored
Phase 1 baseline episode); at stratified branch points t_b take a Branch (env copy +
global-RNG snapshot), let the original continue closed-loop for H steps (the NOMINAL
action sequence), then for every intervention spawn a branch and replay the nominal
actions with one action family perturbed (open loop). Arm 0 is the unperturbed replay
(must equal the original bit-for-bit; checked and stored).

    .venv/bin/python scripts/counterfactual_run.py --parallel 12 \
        --out results/phase2b/counterfactual/sim
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_SCENARIOS = [
    "building_train_1/fairbanks_1", "building_train_1/atlanta_2",
    "building_train_2/fairbanks_2", "building_train_2/atlanta_3",
    "building_train_4/fairbanks_4", "building_train_4/atlanta_5",
    "building_train_5/fairbanks_5", "building_train_5/atlanta_6",
    "building_train_7/fairbanks_7", "building_train_7/atlanta_8",
    "building_train_8/fairbanks_8", "building_train_8/atlanta_1",
]


def run_scenario(scenario: str, H: int, C: int, n_per_stratum: int, seed: int, pulse: int, data_root: str,
                 out_dir: str) -> dict:
    import numpy as np
    from counterfactual.branch import Branch, RNGState
    from counterfactual.interventions import apply_intervention, default_interventions, select_branch_points, step_conditions
    from counterfactual.simrun import make_action, max_divergence, policy_action, record_step, setup, stack
    from dataset import load_episode, load_scenario_meta
    from effective_action import ActionRepresentation
    t_start = time.time()
    root = Path(data_root)
    ep_id = f"{scenario.replace('/', '__')}__baseline__s0__e0"
    ref = load_episode(root / "episodes" / f"{ep_id}.npz")
    meta_ds = load_scenario_meta(root, scenario)
    cond = step_conditions(ref, meta_ds)
    points = select_branch_points(cond, n_per_stratum, C, H, seed)
    ivs = default_interventions(H, pulse)
    env, smeta, policy, spec, obs = setup(scenario)
    assert smeta["observation_field_names"] == meta_ds["observation_field_names"]
    rep = ActionRepresentation(meta_ds)
    rooms = smeta["zones"]["room_ids"]
    obs_hist, ac_hist, ad_hist = [obs.astype(np.float32)], [], []
    P, A = len(points), len(ivs) + 1
    V, D = smeta["num_zones"], smeta["obs_dim"]
    nc, nd = smeta["n_continuous_action"], smeta["n_discrete_action"]
    res = {k: [] for k in ("ctx_obs", "ctx_ac", "ctx_ad", "arm_ac", "arm_ad", "zone_temp", "zone_occ", "zone_sp", "tank",
                           "energy", "reward", "regret", "obs", "eff_delta", "cmd_delta")}
    checks, t = [], 0
    for tb, stratum in points:
        while t < tb:
            a = policy_action(policy, obs, spec)
            r = record_step(env, a, rooms)
            obs = r["obs"]; t += 1
            obs_hist.append(obs.astype(np.float32)); ac_hist.append(a["continuous_action"].copy()); ad_hist.append(a["discrete_action"].copy())
        br = Branch.take(env)
        main = []
        for _ in range(H):
            a = policy_action(policy, obs, spec)
            r = record_step(env, a, rooms)
            main.append(r); obs = r["obs"]; t += 1
            obs_hist.append(obs.astype(np.float32)); ac_hist.append(a["continuous_action"].copy()); ad_hist.append(a["discrete_action"].copy())
        rng_after = RNGState.capture()
        nom_c = np.asarray(ac_hist[tb:tb + H], np.float32)
        nom_d = np.asarray(ad_hist[tb:tb + H])
        eff_nom = rep.effective(nom_c, nom_d)
        arms_c, arms_d, outs = [nom_c], [nom_d], []
        for iv in ivs:
            c2, d2 = apply_intervention(iv, nom_c, nom_d, meta_ds)
            arms_c.append(c2); arms_d.append(d2)
        for i, (c2, d2) in enumerate(zip(arms_c, arms_d)):
            e = br.spawn()
            recs = [record_step(e, make_action(c2[k], d2[k], spec), rooms) for k in range(H)]
            outs.append(stack(recs))
            del e
        del br
        rng_after.restore()
        div = max_divergence(outs[0], stack(main))
        checks.append({"t_b": tb, "nominal_replay_vs_original": div})
        res["ctx_obs"].append(np.asarray(obs_hist[tb - C + 1:tb + 1], np.float32))
        res["ctx_ac"].append(np.asarray(ac_hist[tb - C:tb], np.float32))
        res["ctx_ad"].append(np.asarray(ad_hist[tb - C:tb]))
        res["arm_ac"].append(np.stack(arms_c)); res["arm_ad"].append(np.stack(arms_d))
        res["zone_temp"].append(np.stack([o["zone_info"][..., 2] for o in outs]).astype(np.float32))
        res["zone_occ"].append(np.stack([o["zone_info"][..., 5] for o in outs]).astype(np.float32))
        res["zone_sp"].append(np.stack([o["zone_info"][..., :2] for o in outs]).astype(np.float32))
        res["tank"].append(np.stack([o["tank"][:, 0] for o in outs]).astype(np.float32))
        res["energy"].append(np.stack([np.stack([o["ahu_info"][..., 0].sum(-1), np.abs(o["ahu_info"][..., 1]).sum(-1),
                                                 o["boiler_info"][..., 1].sum(-1), o["boiler_info"][..., 0].sum(-1)], -1)
                                       for o in outs]).astype(np.float32))
        res["reward"].append(np.stack([o["reward"] for o in outs]))
        res["regret"].append(np.stack([o["productivity_regret"] for o in outs]))
        res["obs"].append(np.stack([o["obs"] for o in outs]).astype(np.float32))
        effs = [rep.effective(c2, d2) for c2, d2 in zip(arms_c, arms_d)]
        res["eff_delta"].append(np.stack([np.concatenate([e.effective_native - eff_nom.effective_native,
                                                          (e.discrete - eff_nom.discrete).astype(np.float64)], -1) for e in effs]).astype(np.float32))
        res["cmd_delta"].append(np.stack([np.concatenate([e.commanded_native - eff_nom.commanded_native,
                                                          (e.discrete - eff_nom.discrete).astype(np.float64)], -1) for e in effs]).astype(np.float32))
    # the main trajectory (with branches interleaved) must still equal the stored Phase 1 episode
    n = len(obs_hist) - 1
    replay_obs = float(np.max(np.abs(np.asarray(obs_hist) - ref["obs"][:n + 1])))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{scenario.replace('/', '__')}.npz"
    arrays = {k: np.stack(v) for k, v in res.items()}
    tbs = np.array([p[0] for p in points])
    arrays.update({"t_b": tbs, "scenario": np.array(scenario), "stratum": np.array([p[1] for p in points]),
                   "oat_c": cond["oat_c"][tbs], "oat_band": cond["oat_band"][tbs].astype(str),
                   "heating": cond["heating"][tbs], "cooling": cond["cooling"][tbs], "ahu_on": cond["ahu_on"][tbs],
                   "occupied": cond["occupied"][tbs], "hour": cond["hour"][tbs],
                   "oat_future_obs": np.stack([ref["obs"][tb:tb + H] for tb in tbs]).astype(np.float32),
                   "arm_names": np.array(["nominal"] + [iv.name for iv in ivs]),
                   "interventions_json": np.array(json.dumps([iv.to_dict() for iv in ivs])),
                   "checks_json": np.array(json.dumps(checks)), "H": H, "C": C})
    np.savez_compressed(f, **arrays)
    return {"scenario": scenario, "points": points, "arms": A, "replay_obs_max_abs": replay_obs,
            "nominal_bit_exact_all": all(c["nominal_replay_vs_original"]["bit_exact"] for c in checks),
            "wall_s": time.time() - t_start, "file": str(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    ap.add_argument("--horizon", type=int, default=36)
    ap.add_argument("--context", type=int, default=12)
    ap.add_argument("--n-per-stratum", type=int, default=1)
    ap.add_argument("--pulse", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=12)
    ap.add_argument("--data-root", default=str(ROOT / "data/phase1_sample"))
    ap.add_argument("--out", default="results/phase2b/counterfactual/sim")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(vars(a), indent=1))
    log = []
    order = sorted(a.scenarios, key=lambda s: s.split("/")[0] not in ("building_train_2", "building_train_5"))
    with ProcessPoolExecutor(a.parallel) as pool:
        futs = {pool.submit(run_scenario, s, a.horizon, a.context, a.n_per_stratum, a.seed, a.pulse, a.data_root, str(out)): s
                for s in order}
        for fu in as_completed(futs):
            try:
                r = fu.result()
            except Exception as e:  # noqa: BLE001
                import traceback
                r = {"scenario": futs[fu], "error": traceback.format_exc()}
                print("FAIL", futs[fu], r["error"], flush=True)
            else:
                print(f"{r['scenario']}: {len(r['points'])} points x {r['arms']} arms, nominal exact={r['nominal_bit_exact_all']}, "
                      f"replay obs max {r['replay_obs_max_abs']:.1e}, {r['wall_s'] / 60:.1f} min", flush=True)
            log.append(r)
    (out / "run_log.json").write_text(json.dumps(log, indent=1, default=str))


if __name__ == "__main__":
    main()
