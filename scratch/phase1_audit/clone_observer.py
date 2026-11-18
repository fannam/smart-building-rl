"""Audit experiments on b8/atlanta_1 (1152 steps, seed 1). Read-only w.r.t. production code.

forced_clone : null config, but the counterfactual clone path is taken every step.
               b_t must equal a_t, and the rollout must equal the stored baseline episode.
observer_*   : record the baseline's internal one-step model error (policy.model_error,
               EWMA of |prediction residual|) under null / action-only / internal-only /
               moderate exploration, to quantify observer inconsistency.
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SCN, SEED, T = "building_train_8/atlanta_1", 1, 1152

def job(kind):
    os.chdir(ROOT / "starter-kit")
    from smart_buildings import evaluate as ev
    from collect_data import build_env
    from building_meta import build_scenario_metadata
    from exploration import ExplorationBaselinePolicy, Explorer, PROFILES, episode_rng
    env, meta, _ = build_env(SCN); spec = env.action_spec()
    smeta = build_scenario_metadata(env, meta, SCN)
    obs, _ = env.reset()
    pol = ev.initialize_policy(ExplorationBaselinePolicy, meta, spec)
    mod = PROFILES["moderate"]
    cfg = {"forced_clone": PROFILES["null"], "observer_null": PROFILES["null"], "observer_moderate": mod,
           "observer_action_only": replace(mod, name="action_only", setback_prob=0.0, precondition_prob=0.0),
           "observer_internal_only": replace(mod, name="internal_only", ahu_prob=0.0, sat_prob=0.0,
                                             pressure_prob=0.0, hws_prob=0.0)}[kind]
    ex = Explorer(cfg, smeta, pol, episode_rng(SEED, SCN, 0))
    a_c, b_c, a_d, b_d, rew, merr, occ = [], [], [], [], [], [], []
    names = smeta["observation_field_names"]; i_now = names.index("comfort_mode_now")
    for t in range(T):
        changed = ex.begin_step(obs)
        if kind == "forced_clone":
            changed = True
        base = ex.baseline_clone().get_action(obs) if changed else None
        raw = pol.get_action(obs)
        base = base if base is not None else raw
        exe, _ = ex.apply_action_knobs(raw, obs)
        exe = ev.validate_action(exe, spec); base = ev.validate_action(base, spec)
        occ.append(obs[i_now] > .5)
        obs, r, *_ = env.step(exe)
        a_c.append(exe["continuous_action"]); b_c.append(base["continuous_action"])
        a_d.append(exe["discrete_action"]); b_d.append(base["discrete_action"]); rew.append(float(r))
        merr.append(float(np.mean(pol.model_error)))
    a_c, b_c, a_d, b_d = map(np.array, (a_c, b_c, a_d, b_d)); merr = np.array(merr); occ = np.array(occ)
    out = {"kind": kind, "score": -float(np.mean(rew)),
           "model_error_mean": float(merr.mean()), "model_error_occupied_mean": float(merr[occ].mean()),
           "model_error_p95": float(np.percentile(merr, 95)),
           "b_equals_a_all_steps": bool(np.array_equal(a_c, b_c) and np.array_equal(a_d, b_d))}
    if kind == "forced_clone":
        sys.path.insert(0, str(ROOT / "src"))
        from dataset import load_episode
        import glob
        full = load_episode(glob.glob(str(ROOT / "data/phase1_validation/full_baseline/episodes/*.npz"))[0])
        out["rollout_equals_stored_baseline_first_T"] = bool(
            np.array_equal(a_c, full["action_continuous"][:T]) and np.array_equal(a_d.astype(np.int8), full["action_discrete"][:T])
            and np.array_equal(np.array(rew), full["reward"][:T]))
    return out

if __name__ == "__main__":
    kinds = ["forced_clone", "observer_null", "observer_moderate", "observer_action_only", "observer_internal_only"]
    with ProcessPoolExecutor(len(kinds)) as p:
        res = list(p.map(job, kinds))
    print(json.dumps(res, indent=1))
    (ROOT / "scratch/phase1_audit/out/clone_observer.json").write_text(json.dumps(res, indent=1))
