"""Thin simulator driver for counterfactual experiments (reuses the Phase 1 collector's env setup).

`record_step` returns the same physical quantities the Phase 1 dataset stores, so the
dynamics featurizer / targets can be applied unchanged to simulated branches.
"""
from __future__ import annotations

import numpy as np

from collect_data import AHU_INFO_FIELDS, BOILER_INFO_FIELDS, ZONE_INFO_FIELDS, _tank_state, build_env  # noqa: F401


def setup(scenario: str):
    """env, smeta, baseline policy (unmodified WideHotStorageReservePolicy), first obs."""
    import os
    from run_eval import KIT
    os.chdir(KIT)
    from smart_buildings import evaluate as ev
    import baseline_policy
    from building_meta import build_scenario_metadata
    env, meta, _ = build_env(scenario)
    spec = env.action_spec()
    smeta = build_scenario_metadata(env, meta, scenario)
    obs, _ = env.reset()
    policy = ev.initialize_policy(baseline_policy.WideHotStorageReservePolicy, meta, spec)
    return env, smeta, policy, spec, np.asarray(obs)


def policy_action(policy, obs, spec):
    from smart_buildings import evaluate as ev
    return ev.validate_action(ev.get_policy_action(policy, obs), spec)


def make_action(cont: np.ndarray, disc: np.ndarray, spec):
    from smart_buildings import evaluate as ev
    return ev.validate_action({"continuous_action": np.asarray(cont, np.float32),
                               "discrete_action": np.asarray(disc, np.int32)}, spec)


def record_step(env, action, zone_rooms) -> dict:
    obs, r, term, trunc, _ = env.step(action)
    info, resp = env.get_reward_info_and_response()
    zone = np.array([[getattr(info.zone_reward_infos[room], f) for f in ZONE_INFO_FIELDS] for room in zone_rooms], np.float64)
    ahu = np.array([[getattr(info.air_handler_reward_infos[k], f) for f in AHU_INFO_FIELDS]
                    for k in sorted(info.air_handler_reward_infos)], np.float64)
    boiler = np.array([[getattr(info.boiler_reward_infos[k], f) for f in BOILER_INFO_FIELDS]
                       for k in sorted(info.boiler_reward_infos)], np.float64)
    return {"obs": np.asarray(obs, np.float64), "reward": float(r), "zone_info": zone, "ahu_info": ahu,
            "boiler_info": boiler, "tank": np.asarray(_tank_state(env), np.float64),
            "productivity_regret": float(resp.normalized_productivity_regret),
            "energy_cost": float(resp.normalized_energy_cost), "carbon": float(resp.normalized_carbon_emission),
            "done": bool(term or trunc)}


def stack(records: list[dict]) -> dict:
    return {k: np.asarray([r[k] for r in records]) for k in records[0]}


def max_divergence(a: dict, b: dict) -> dict:
    """Max |a-b| per recorded quantity (aligned lists of record_step outputs, stacked)."""
    out = {}
    for k in ("obs", "reward", "zone_info", "tank", "ahu_info", "boiler_info"):
        x, y = np.asarray(a[k], np.float64), np.asarray(b[k], np.float64)
        n = min(len(x), len(y))
        out[k] = float(np.max(np.abs(x[:n] - y[:n]))) if n else float("nan")
    out["zone_temp"] = float(np.max(np.abs(np.asarray(a["zone_info"])[..., 2] - np.asarray(b["zone_info"])[..., 2])))
    out["bit_exact"] = all(out[k] == 0.0 for k in ("obs", "reward", "zone_info", "tank", "ahu_info", "boiler_info"))
    return out
