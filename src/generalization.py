"""Leave-one-building-out (LOBO) evaluation framework (Phase 1.6).

Model-agnostic pieces reused by Phase 2:
  * make_lobo_folds / validate_fold    deterministic folds + leakage checks (raise LeakageError)
  * FittedStatistics                   preprocessing statistics fitted on TRAIN episodes only,
                                       with provenance that `assert_provenance` checks per fold
  * episode_diagnostics / aggregate    DESCRIPTIVE dataset statistics (no model involved)

Nothing here trains or evaluates a model; model metrics are listed as
`requires_model` in the reports and are never filled in.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from building_meta import ActionCodec, ObservationCodec
from dataset import load_episode, load_scenario_meta
from effective_action import ActionRepresentation, MIN_STATIC_PRESSURE_PA

FOLDS_VERSION = "lobo-v1"
SEVERE_COLD_MIN_OAT_C = -10.0      # scenario minimum outdoor temperature at or below this
CLIMATE_BANDS_C = ((-np.inf, 5.0, "cold"), (5.0, 20.0, "mild"), (20.0, np.inf, "hot"))  # by mean OAT
COMFORT_THRESHOLDS_K = (0.0, 0.5, 1.0, 2.0)
# float32 action normalization turns an exact 200 Pa command into 199.99998 Pa: not a binding floor
FLOOR_TOL_PA = 0.5
REQUIRES_MODEL = ["one-step / multi-step zone-temperature RMSE", "tank-temperature RMSE",
                  "energy (E) and gas (G) head error", "reward-reconstruction error from predicted states",
                  "closed-loop score"]


class LeakageError(AssertionError):
    pass


def station_of(weather: str) -> str:
    return weather.rsplit("_", 1)[0]


# ------------------------------------------------------------------ folds
def make_lobo_folds(buildings: Iterable[str]) -> list[dict[str, Any]]:
    b = sorted(set(buildings))
    if len(b) < 2:
        raise ValueError("LOBO needs at least two buildings")
    return [{"fold": i, "test_building": t, "train_buildings": [x for x in b if x != t]} for i, t in enumerate(b)]


def split_rows(rows: Sequence[dict[str, Any]], fold: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    tr = [r for r in rows if r["building"] in set(fold["train_buildings"])]
    te = [r for r in rows if r["building"] == fold["test_building"]]
    return tr, te


def validate_fold(fold: dict[str, Any], train_rows: Sequence[dict], test_rows: Sequence[dict]) -> dict[str, Any]:
    tb = fold["test_building"]
    if tb in fold["train_buildings"]:
        raise LeakageError(f"fold {fold['fold']}: test building {tb} listed in train")
    trb = {r["building"] for r in train_rows}
    teb = {r["building"] for r in test_rows}
    if tb in trb:
        raise LeakageError(f"fold {fold['fold']}: episodes of {tb} in train")
    if teb - {tb}:
        raise LeakageError(f"fold {fold['fold']}: test contains other buildings {sorted(teb - {tb})}")
    for key in ("episode_id", "scenario", "path"):
        ov = {r[key] for r in train_rows} & {r[key] for r in test_rows}
        if ov:
            raise LeakageError(f"fold {fold['fold']}: {key} overlap {sorted(ov)[:3]}")
    if not train_rows or not test_rows:
        raise LeakageError(f"fold {fold['fold']}: empty train or test")
    return {"fold": fold["fold"], "ok": True, "train_episodes": len(train_rows), "test_episodes": len(test_rows)}


def manifest_digest(rows: Sequence[dict]) -> str:
    return hashlib.sha256("\n".join(sorted(r["episode_id"] for r in rows)).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ fitted statistics
@dataclass
class FittedStatistics:
    """Group-level (building-size independent) mean/std of decoded runtime features.

    Fitted on train episodes only; `provenance` records exactly which.
    """
    stats: dict[str, dict[str, float]] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(cls, root: str | Path, train_rows: Sequence[dict]) -> "FittedStatistics":
        root = Path(root)
        acc: dict[str, list[np.ndarray]] = {}
        for r in train_rows:
            ep = load_episode(root / r["path"], keys=["obs", "action_continuous", "action_discrete"])
            sm = load_scenario_meta(root, r["scenario"])
            oc = ObservationCodec(sm)
            nat = oc.native(ep["obs"])
            ah = sm["ahus"]
            feats = {
                "zone_air_temperature_k": nat[:, sm["zones"]["obs_temp"]],
                "outside_air_temperature_k": nat[:, ah["obs_outside_air_temp"][:1]],
                "hws_supply_water_temperature_k": nat[:, [sm["hws"]["obs_supply_water_temp"]]],
                "num_occupants": nat[:, [oc.index["num_occupants"]]],
            }
            eff = ActionRepresentation(sm).effective(ep["action_continuous"], ep["action_discrete"]).effective_native
            feats["sat_setpoint_k"] = eff[:, ah["act_sat"]]
            feats["static_pressure_eff_pa"] = eff[:, ah["act_static_pressure"]]
            feats["hws_setpoint_k"] = eff[:, [sm["hws"]["act_supply_water_temp"]]]
            for k, v in feats.items():
                acc.setdefault(k, []).append(np.asarray(v, dtype=np.float64).ravel())
        stats = {}
        for k, parts in acc.items():
            x = np.concatenate(parts)
            stats[k] = {"mean": float(x.mean()), "std": float(x.std() + 1e-9), "n": int(x.size)}
        prov = {"buildings": sorted({r["building"] for r in train_rows}),
                "episode_ids": sorted(r["episode_id"] for r in train_rows),
                "manifest_digest": manifest_digest(train_rows)}
        return cls(stats, prov)

    def to_json(self) -> dict[str, Any]:
        return {"stats": self.stats, "provenance": self.provenance}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "FittedStatistics":
        return cls(d["stats"], d["provenance"])


def assert_provenance(stats: FittedStatistics, fold: dict[str, Any], train_rows: Sequence[dict]) -> None:
    used = set(stats.provenance["buildings"])
    if fold["test_building"] in used:
        raise LeakageError(f"fold {fold['fold']}: statistics fitted with held-out building {fold['test_building']}")
    if not used <= set(fold["train_buildings"]):
        raise LeakageError(f"fold {fold['fold']}: statistics used buildings outside train: {sorted(used)}")
    if stats.provenance["manifest_digest"] != manifest_digest(train_rows):
        raise LeakageError(f"fold {fold['fold']}: statistics provenance does not match the train manifest")


# ------------------------------------------------------------------ descriptive diagnostics
def _pct(x, q):
    x = np.asarray(x, dtype=np.float64).ravel()
    return float(np.percentile(x, q)) if x.size else None


def _p(x):
    return {"p5": _pct(x, 5), "p50": _pct(x, 50), "p95": _pct(x, 95)}


def episode_diagnostics(root: str | Path, row: dict[str, Any]) -> dict[str, Any]:
    """Descriptive statistics of one episode. Privileged arrays are read as *targets*
    for description only; nothing here is a model input."""
    root = Path(root)
    ep = load_episode(root / row["path"])
    sm = load_scenario_meta(root, row["scenario"])
    oc, ac = ObservationCodec(sm), ActionCodec(sm)
    K, V, T = sm["num_ahus"], sm["num_zones"], ep["reward"].shape[0]
    ah, hw = sm["ahus"], sm["hws"]
    nat = oc.native(ep["obs"])
    oat_c = nat[:, ah["obs_outside_air_temp"][0]] - 273.15
    zi = ep["zone_info"].astype(np.float64)
    zn = ep["names_zone_info"]
    tz = zi[..., zn.index("zone_air_temperature")]
    heat, cool = zi[..., zn.index("heating_setpoint_temperature")], zi[..., zn.index("cooling_setpoint_temperature")]
    occ_z = zi[..., zn.index("average_occupancy")]
    occ = occ_z.sum(-1)
    occupied = occ > 0
    dev = np.maximum(0.0, np.maximum(tz - cool, heat - tz))
    dev_occ = dev[occ_z > 0]  # zone-steps whose zone is occupied (these carry comfort weight)
    h = 300.0 / 3.6e6  # W -> kWh per step
    a_info, b_info = ep["ahu_info"].astype(np.float64), ep["boiler_info"].astype(np.float64)
    blower = a_info[..., 0].sum(-1)
    ac_el = np.abs(a_info[..., 1]).sum(-1)
    pump = b_info[..., 1].sum(-1)
    gas = b_info[..., 0].sum(-1)
    rep = ActionRepresentation(sm)
    e = rep.effective(ep["action_continuous"], ep["action_discrete"])
    run = ep["action_discrete"][:, ah["act_run"]].astype(bool)
    cmd = ac.to_native(ep["action_continuous"])
    sat = cmd[:, ah["act_sat"]][run]
    p_cmd = cmd[:, ah["act_static_pressure"]][run]
    p_eff = e.effective_native[:, ah["act_static_pressure"]][run]
    rc_cmd = cmd[:, ah["act_recirculation"]]
    rc_eff = e.effective_native[:, ah["act_recirculation"]]
    sw = cmd[:, hw["act_supply_water_temp"]]
    damp_eff = e.effective_native[:, sm["zones"]["act_damper"]]
    valve = cmd[:, sm["zones"]["act_reheat_valve"]]
    mean_oat = float(oat_c.mean())
    band = next(n for lo, hi, n in CLIMATE_BANDS_C if lo <= mean_oat < hi)
    d = {
        "episode_id": row["episode_id"], "building": row["building"], "weather": row["weather"],
        "station": station_of(row["weather"]), "profile": row["profile"], "K": K, "V": V, "transitions": int(T),
        "start_month_privileged": int(ep["calendar"][0, 3]),
        "oat_c": {"min": float(oat_c.min()), "mean": mean_oat, "max": float(oat_c.max())},
        "heating_degree_hours_18c": float(np.sum(np.maximum(0, 18 - oat_c[:-1])) / 12),
        "cooling_degree_hours_24c": float(np.sum(np.maximum(0, oat_c[:-1] - 24)) / 12),
        "climate_band": band, "severe_cold": bool(oat_c.min() <= SEVERE_COLD_MIN_OAT_C),
        "zone_temp_c": {"p1": _pct(tz - 273.15, 1), "p50": _pct(tz - 273.15, 50), "p99": _pct(tz - 273.15, 99)},
        "occupancy": {"occupied_frac": float(occupied.mean()),
                      "mean_total_when_occupied": float(occ[occupied].mean()) if occupied.any() else 0.0,
                      "max_total": float(occ.max())},
        "energy_kwh_per_ahu_per_day": {
            "electricity": float((blower + ac_el + pump).sum() * h / K / (T / 288)),
            "cooling_electric": float(ac_el.sum() * h / K / (T / 288)),
            "fan_electric": float(blower.sum() * h / K / (T / 288)),
            "pump_electric": float(pump.sum() * h / K / (T / 288)),
            "gas": float(gas.sum() * h / K / (T / 288))},
        "heating_active_frac": float(np.mean(gas > 0)), "cooling_active_frac": float(np.mean(ac_el > 0)),
        "comfort_occupied_zone_steps": int(dev_occ.size),
        "comfort_violation_frac": {f">{t}K": float(np.mean(dev_occ > t)) if dev_occ.size else 0.0
                                   for t in COMFORT_THRESHOLDS_K},
        "comfort_violation_mean_k_when_violating": float(dev_occ[dev_occ > 0].mean()) if np.any(dev_occ > 0) else 0.0,
        "comfort_violation_max_k": float(dev_occ.max()) if dev_occ.size else 0.0,
        "actions": {
            "ahu_run_frac": float(run.mean()),
            "sat_cmd_k_running": _p(sat),
            "static_pressure_cmd_pa_running": _p(p_cmd),
            "static_pressure_eff_pa_running": _p(p_eff),
            "pressure_floor_binding_frac_running": float(np.mean(p_cmd < MIN_STATIC_PRESSURE_PA - FLOOR_TOL_PA)) if p_cmd.size else 0.0,
            "recirc_cmd_mean": float(rc_cmd.mean()), "recirc_eff_mean": float(rc_eff.mean()),
            "recirc_cap_binding_frac": float(np.mean(rc_cmd > 0.7 + 1e-4)),
            "hws_setpoint_k": _p(sw),
            "damper_eff_mean": float(damp_eff.mean()), "damper_floor_binding_frac": float(np.mean(cmd[:, sm["zones"]["act_damper"]] < 0.1 - 1e-3)),
            "reheat_valve_mean": float(valve.mean()),
            "perturbed_frac": float(ep["perturbed"].mean()),
        },
        "reward_mean": float(ep["reward"].mean()),
    }
    return d


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, bool):
            out[key] = float(v)
    return out


def aggregate(diags: Sequence[dict[str, Any]], by: str | None = None) -> dict[str, Any]:
    """Group descriptive diagnostics; numeric fields -> transition-weighted mean, plus min/max."""
    groups: dict[str, list[dict]] = {}
    for d in diags:
        key = "all" if by is None else str(d[by])
        groups.setdefault(key, []).append(d)
    out = {}
    for key, ds in sorted(groups.items()):
        flat = [_flatten(d) for d in ds]
        w = np.array([d["transitions"] for d in ds], dtype=np.float64)
        keys = sorted(set().union(*flat))
        agg = {"episodes": len(ds), "transitions": int(w.sum()),
               "buildings": sorted({d["building"] for d in ds}), "stations": sorted({d["station"] for d in ds}),
               "weathers": sorted({d["weather"] for d in ds}), "K": sorted({d["K"] for d in ds}),
               "V": sorted({d["V"] for d in ds})}
        for k in keys:
            if k in ("K", "V", "transitions"):
                continue
            vals = np.array([f.get(k, np.nan) for f in flat])
            ok = ~np.isnan(vals)
            if not ok.any():
                continue
            agg[k] = {"mean": float(np.average(vals[ok], weights=w[ok])), "min": float(vals[ok].min()),
                      "max": float(vals[ok].max())}
        out[key] = agg
    return out


def coverage(train: Sequence[dict[str, Any]], test: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Does the train set span the held-out building's operating range? (descriptive)"""
    tr_oat = (min(d["oat_c"]["min"] for d in train), max(d["oat_c"]["max"] for d in train))
    te_oat = (min(d["oat_c"]["min"] for d in test), max(d["oat_c"]["max"] for d in test))
    return {
        "train_oat_c_range": tr_oat, "test_oat_c_range": te_oat,
        "test_oat_within_train_range": bool(te_oat[0] >= tr_oat[0] and te_oat[1] <= tr_oat[1]),
        "train_K": sorted({d["K"] for d in train}), "test_K": sorted({d["K"] for d in test}),
        "test_K_within_train_K_range": bool(min(d["K"] for d in test) >= min(d["K"] for d in train)
                                            and max(d["K"] for d in test) <= max(d["K"] for d in train)),
        "train_V_range": (min(d["V"] for d in train), max(d["V"] for d in train)),
        "test_V": sorted({d["V"] for d in test}),
        "test_severe_cold_scenarios": sorted({d["weather"] for d in test if d["severe_cold"]}),
        "train_severe_cold_scenarios": sorted({f'{d["building"]}/{d["weather"]}' for d in train if d["severe_cold"]}),
        "test_min_oat_c": te_oat[0],
    }

