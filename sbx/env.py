"""Environment construction, action sanitizing and a single-scenario rollout.

Environment construction mirrors the Kaggle gateway (`kaggle_evaluation/gateway.py::_load_environment`)
and the starter kit's `evaluate.py`: same loader, `metrics_path = None`, the same (no-op)
`_occupancy_normalization_constant = 125.0`, and the same metadata builder.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

# Per-step reward decomposition stored for every rollout (reward-weighted, each <= 0; they sum to the reward).
TERMS = ("comfort", "elec_cost", "gas_cost", "elec_carbon", "gas_carbon")


def load_env(scenario_dir: Path, kit: Path):
    """Build (env, building_meta_data, action_spec) exactly as the gateway does."""
    from smart_buildings import evaluate as ev

    cwd = os.getcwd()
    os.chdir(kit)  # gin configs reference assets relative to the starter kit root
    try:
        env = ev.load_hybrid_environment(Path(scenario_dir).resolve() / "config.gin")
        env.metrics_path = None
        env._occupancy_normalization_constant = 125.0
        meta = ev.create_building_meta_data(env, Path(scenario_dir).resolve())
    finally:
        os.chdir(cwd)
    return env, meta, env.action_spec()


@dataclass
class Sanitizer:
    """Makes any policy output a valid action: finite, clipped to the spec, right dtype and shape.

    One invalid action fails the whole Kaggle submission, so this sits in front of every rollout.
    `fixes` counts how often it had to change something (should stay 0 for a correct policy).
    """

    spec: Any
    fixes: dict[str, int] = field(default_factory=lambda: {"nonfinite": 0, "clipped": 0, "shape": 0})

    def __call__(self, action: dict) -> dict:
        out = {}
        for key, dtype in (("continuous_action", np.float32), ("discrete_action", np.int32)):
            box = self.spec[key]
            a = np.asarray(action[key], dtype=np.float64).reshape(-1)
            if a.shape != tuple(box.shape):
                self.fixes["shape"] += 1
                a = np.resize(a, box.shape) if a.size else np.zeros(box.shape)
            bad = ~np.isfinite(a)
            if bad.any():
                self.fixes["nonfinite"] += 1
                a = np.where(bad, 0.0, a)
            if key == "discrete_action":
                a = np.rint(a)
            clipped = np.clip(a, box.low, box.high)
            if np.any(clipped != a):
                self.fixes["clipped"] += 1
            out[key] = clipped.astype(dtype)
        return out


def reward_terms(env, resp, info) -> tuple[float, ...]:
    """Split one step's reward into TERMS, using the reward function's own cost and carbon models."""
    from smart_buildings.smart_control.utils import conversion_utils

    w = resp.productivity_weight + resp.energy_cost_weight + resp.carbon_emission_weight
    comfort = resp.productivity_weight * resp.normalized_productivity_regret / w

    e_cost, g_cost = resp.electricity_energy_cost, resp.natural_gas_energy_cost
    energy = resp.energy_cost_weight * resp.normalized_energy_cost / w
    e_share = e_cost / (e_cost + g_cost) if e_cost + g_cost > 0 else 0.0

    fn = env.reward_function
    gas_rate = sum(b.natural_gas_heating_energy_rate for b in info.boiler_reward_infos.values())
    start = conversion_utils.proto_to_pandas_timestamp(info.start_timestamp)
    end = conversion_utils.proto_to_pandas_timestamp(info.end_timestamp)
    g_carbon = fn._natural_gas_energy_cost.carbon(start_time=start, end_time=end, energy_rate=gas_rate)
    total_carbon = resp.carbon_emitted
    carbon = resp.carbon_emission_weight * resp.normalized_carbon_emission / w
    g_share = g_carbon / total_carbon if total_carbon > 0 else 0.0

    return (
        comfort,
        -energy * e_share,
        -energy * (1.0 - e_share),
        -carbon * (1.0 - g_share),
        -carbon * g_share,
    )


def rollout(
    scenario_dir: Path,
    make_policy: Callable[[dict, Any], Any],
    kit: Path,
    max_steps: int | None,
) -> tuple[dict, np.ndarray]:
    """Run one scenario. max_steps=None runs until the environment terminates (gateway semantics).

    Returns (summary dict, per-step array of shape (1 + len(TERMS), n): reward then TERMS).
    """
    t0 = time.perf_counter()
    env, meta, spec = load_env(scenario_dir, kit)
    t_load = time.perf_counter() - t0

    obs, _ = env.reset()
    t1 = time.perf_counter()
    policy = make_policy(meta, spec)
    t_init = time.perf_counter() - t1

    sanitize = Sanitizer(spec)
    rows: list[tuple[float, ...]] = []
    t_policy = t_sim = t_terms = 0.0
    limit = max_steps if max_steps is not None else 10**9
    while len(rows) < limit:
        a = time.perf_counter()
        action = sanitize(policy.get_action(obs))
        b = time.perf_counter()
        obs, reward, terminated, truncated, _ = env.step(action)
        c = time.perf_counter()
        info, resp = env.get_reward_info_and_response()
        rows.append((float(reward), *reward_terms(env, resp, info)))
        t_terms += time.perf_counter() - c
        t_policy += b - a
        t_sim += c - b
        if terminated or truncated:
            break

    arr = np.asarray(rows, dtype=np.float64).T
    n = arr.shape[1]
    summary = {
        "steps": n,
        "terminated_naturally": max_steps is None or n < max_steps,
        "total_reward": float(arr[0].sum()),
        "mean_reward": float(arr[0].mean()),
        "score": float(abs(arr[0].mean())),
        **{f"mean_{t}": float(arr[i + 1].mean()) for i, t in enumerate(TERMS)},
        "terms_residual_max": float(np.max(np.abs(arr[0] - arr[1:].sum(axis=0)))),
        "worst_step_reward": float(arr[0].min()),
        "num_ahus": len(meta["temp_indices"]),
        "num_zones": len(meta["zone_temp_indices"]),
        "sanitizer_fixes": sanitize.fixes,
        "seconds": {
            "load": t_load,
            "policy_init": t_init,
            "policy": t_policy,
            "sim": t_sim,
            "reward_terms": t_terms,
            "wall": time.perf_counter() - t0,
        },
    }
    return summary, arr.astype(np.float32)
