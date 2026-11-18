"""Building metadata, action layout and observation decoding for the dataset.

Everything here is derived from the same objects the competition hands a
policy (`create_building_meta_data`) plus a few read-only simulator attributes
that are recorded as *privileged* (never available at run time):

  * per-field observation normalization, reproduced exactly from the env's
    StandardScoreObservationNormalizer (exact measurement-name match, otherwise
    identity), so normalized fields can be decoded to native units;
  * the hybrid action layout: ordered continuous/discrete action names and the
    native ranges of each continuous entry;
  * canonical zone and AHU orders (natural sort, same as the baseline policy)
    with index maps into observation and action vectors.

The dataset stores raw observations and actions exactly as exchanged with the
environment; this module only provides decoders.
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping

import numpy as np

SCHEMA_VERSION = "phase1-v1"
STEP_SECONDS = 300


def natural_key(value: str) -> tuple:
    return tuple(int(x) if x.isdigit() else x for x in re.split(r"(\d+)", str(value)))


def _is_discrete(action_name: str) -> bool:
    # environment.is_discrete_setpoint: "run_command" setpoints are discrete.
    return "run_command" in action_name


def observation_field_normalization(env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean, std, normalized_mask) per observation field, as the env applies it.

    The normalizer keys on the *measurement name* (field id without device
    prefix) and falls back to identity on a miss, which is why AHU-prefixed
    fields stay native. Time features, comfort flags and `num_occupants` are
    added after normalization (num_occupants has its own affine map, recorded
    separately in the metadata).
    """
    consts = env.observation_normalizer._normalization_constants
    field_to_measurement = {fid: key[1] for key, fid in env._id_map.items()}
    names = list(env.field_names)
    mean = np.zeros(len(names))
    std = np.ones(len(names))
    normalized = np.zeros(len(names), dtype=bool)
    for i, name in enumerate(names):
        measurement = field_to_measurement.get(name)
        if measurement is not None and measurement in consts:
            c = consts[measurement]
            if c.sample_variance > 0.0:
                mean[i] = c.sample_mean
                std[i] = math.sqrt(c.sample_variance)
                normalized[i] = True
    return mean, std, normalized


def build_scenario_metadata(env, meta: Mapping[str, Any], scenario: str) -> dict[str, Any]:
    """JSON-serializable static description of one scenario (building x weather)."""
    building, weather = scenario.split("/")
    names = [str(x) for x in meta["observation_field_names"]]
    fidx = {n: i for i, n in enumerate(names)}
    action_names = [str(x) for x in env.action_names]
    cont_names = [n for n in action_names if not _is_discrete(n)]
    disc_names = [n for n in action_names if _is_discrete(n)]

    # Native ranges per continuous entry, keyed by setpoint name (as the env does).
    setpoint_of = {fid: key[1] for key, fid in env._id_map.items()}
    norm = meta["action_normalization_metadata"]
    cont_low, cont_high = [], []
    for n in cont_names:
        spec = norm[setpoint_of[n]]
        if (spec["min_normalized_value"], spec["max_normalized_value"]) != (-1.0, 1.0):
            raise ValueError(f"unexpected normalized range for {n}: {spec}")
        cont_low.append(float(spec["min_native_value"]))
        cont_high.append(float(spec["max_native_value"]))

    zone_ids = sorted(meta["zone_action_indices"], key=natural_key)
    ahu_ids = sorted(meta["ahu_zone_assignments"], key=natural_key)
    zone_ahu = {}
    for k, ahu in enumerate(ahu_ids):
        for room in meta["ahu_zone_assignments"][ahu]:
            zone_ahu[str(room).replace("room_", "VAV_")] = k

    def obs_index(name):
        return fidx.get(name, -1)

    zones = {
        "ids": zone_ids,
        "room_ids": [z.replace("VAV_", "room_") for z in zone_ids],
        "ahu": [zone_ahu.get(z, -1) for z in zone_ids],
        "obs_temp": [obs_index(f"{z}_zone_air_temperature_sensor") for z in zone_ids],
        "obs_heating_sp": [obs_index(f"{z}_zone_air_heating_temperature_setpoint_sensor") for z in zone_ids],
        "obs_cooling_sp": [obs_index(f"{z}_zone_air_cooling_temperature_setpoint_sensor") for z in zone_ids],
        "obs_flow_sp": [obs_index(f"{z}_supply_air_flowrate_setpoint") for z in zone_ids],
        "act_damper": [cont_names.index(f"{z}_supply_air_damper_percentage_command") for z in zone_ids],
        "act_reheat_valve": [cont_names.index(f"{z}_reheat_valve_percentage_command") for z in zone_ids],
    }
    ahus = {
        "ids": ahu_ids,
        "act_run": [disc_names.index(f"ahs_{a}_supervisor_run_command") for a in ahu_ids],
        "act_sat": [cont_names.index(f"ahs_{a}_supply_air_temperature_setpoint") for a in ahu_ids],
        "act_static_pressure": [cont_names.index(f"ahs_{a}_supply_air_static_pressure_setpoint") for a in ahu_ids],
        "act_recirculation": [cont_names.index(f"ahs_{a}_recirculation_percentage_command") for a in ahu_ids],
        "obs_outside_air_temp": [obs_index(f"ahs_{a}_outside_air_temperature_sensor") for a in ahu_ids],
        "obs_supply_air_temp": [obs_index(f"ahs_{a}_supply_air_temperature_sensor") for a in ahu_ids],
        "zones": [[zone_ids.index(str(r).replace("room_", "VAV_")) for r in meta["ahu_zone_assignments"][a]
                   if str(r).replace("room_", "VAV_") in zone_ids] for a in ahu_ids],
    }
    hws = {
        "act_run": disc_names.index("hws_supervisor_run_command"),
        "act_supply_water_temp": cont_names.index("hws_supply_water_temperature_setpoint"),
        "act_differential_pressure": cont_names.index("hws_differential_pressure_setpoint"),
        "obs_supply_water_temp": obs_index("hws_supply_water_temperature_sensor"),
    }

    mean, std, normalized = observation_field_normalization(env)
    boiler = env.building.simulator._hvac.hot_water_system._heat_source
    rf = env.reward_function
    K = len(ahu_ids)
    reward_cfg = {k: float(v) for k, v in rf.json_metadata.items()}
    reward_cfg["num_ahus_K"] = K
    reward_cfg["max_electricity_rate_over_K"] = reward_cfg["max_electricity_rate"] / K
    reward_cfg["max_natural_gas_rate_over_K"] = reward_cfg["max_natural_gas_rate"] / K

    return {
        "schema_version": SCHEMA_VERSION,
        "scenario": scenario,
        "building": building,
        "weather": weather,
        "num_ahus": K,
        "num_zones": len(zone_ids),
        "obs_dim": len(names),
        "n_continuous_action": int(meta["n_continuous_action"]),
        "n_discrete_action": int(meta["n_discrete_action"]),
        "observation_field_names": names,
        "observation_mean": mean.tolist(),
        "observation_std": std.tolist(),
        "observation_normalized": normalized.tolist(),
        # num_occupants obs = (N - c) / (c + 1); c is the env's *public* attribute.
        # evaluate.py/run_eval.py set the private `_occupancy_normalization_constant`,
        # which the env never reads, so c stays at its gin value (0 -> raw count).
        "num_occupants_normalization_constant": float(env.occupancy_normalization_constant),
        "continuous_action_names": cont_names,
        "discrete_action_names": disc_names,
        "continuous_action_native_low": cont_low,
        "continuous_action_native_high": cont_high,
        "zones": zones,
        "ahus": ahus,
        "hws": hws,
        "ahu_zone_assignments": {k: list(v) for k, v in meta["ahu_zone_assignments"].items()},
        "zone_action_indices": meta["zone_action_indices"],
        "temp_indices": list(meta["temp_indices"]),
        "pressure_indices": list(meta["pressure_indices"]),
        "water_indices": list(meta["water_indices"]),
        "recirculation_indices": list(meta["recirculation_indices"]),
        "zone_temp_indices": list(meta["zone_temp_indices"]),
        "num_floors": int(meta["num_floors"]),
        "schedule": {
            "morning_start_hour": meta["morning_start_hour"],
            "evening_start_hour": meta["evening_start_hour"],
            "comfort_temp_window": list(map(float, meta["comfort_temp_window"])),
            "eco_temp_window": list(map(float, meta["eco_temp_window"])),
            "time_zone": str(meta["time_zone"]),
            "holidays": sorted(str(h) for h in (meta["holidays"] or [])),
        },
        "action_normalization_metadata": meta["action_normalization_metadata"],
        "observation_normalization_metadata": meta["observation_normalization_metadata"],
        "reward_config": reward_cfg,
        # Privileged: simulator internals, for physics priors only (not observable).
        "privileged": {
            "episode_start": str(env._start_timestamp),
            "episode_end": str(env._end_timestamp),
            "num_timesteps_in_episode": int(env._num_timesteps_in_episode),
            "boiler": {
                "energy_balance": bool(boiler.energy_balance),
                "has_tank": bool(boiler._has_tank),
                "water_capacity_m3": float(boiler._water_capacity),
                "heating_rate_k_per_min": float(boiler._heating_rate),
                "cooling_rate": float(boiler._cooling_rate),
                "tank_length_m": float(boiler._tank_length),
                "tank_radius_m": float(boiler._tank_radius),
                "insulation_conductivity": float(boiler._insulation_conductivity),
                "insulation_thickness_m": float(boiler._insulation_thickness),
                "convection_coefficient": float(boiler._convection_coefficient),
                "efficiency": 1.0,
            },
        },
    }


class ObservationCodec:
    """Decode raw observations (as returned by env.step) to native units."""

    def __init__(self, scenario_meta: Mapping[str, Any]):
        self.meta = scenario_meta
        self.names = list(scenario_meta["observation_field_names"])
        self.index = {n: i for i, n in enumerate(self.names)}
        self.mean = np.asarray(scenario_meta["observation_mean"], dtype=np.float64)
        self.std = np.asarray(scenario_meta["observation_std"], dtype=np.float64)
        self.occ_c = float(scenario_meta["num_occupants_normalization_constant"])
        z = scenario_meta["zones"]
        self.zone_temp_idx = np.asarray(z["obs_temp"])
        self.zone_heat_idx = np.asarray(z["obs_heating_sp"])
        self.zone_cool_idx = np.asarray(z["obs_cooling_sp"])

    def native(self, obs: np.ndarray) -> np.ndarray:
        """Native value of every field ([..., D]); occupancy decoded too."""
        x = np.asarray(obs, dtype=np.float64) * self.std + self.mean
        j = self.index.get("num_occupants")
        if j is not None:
            x[..., j] = np.asarray(obs, dtype=np.float64)[..., j] * (self.occ_c + 1.0) + self.occ_c
        return x

    def field(self, obs: np.ndarray, name: str) -> np.ndarray:
        j = self.index[name]
        if name == "num_occupants":
            return np.asarray(obs, dtype=np.float64)[..., j] * (self.occ_c + 1.0) + self.occ_c
        return np.asarray(obs, dtype=np.float64)[..., j] * self.std[j] + self.mean[j]

    def zone_temperatures(self, obs):
        return self.native(obs)[..., self.zone_temp_idx]

    def zone_bands(self, obs):
        n = self.native(obs)
        return n[..., self.zone_heat_idx], n[..., self.zone_cool_idx]

    def hour(self, obs):
        o = np.asarray(obs, dtype=np.float64)
        ang = np.arctan2(o[..., self.index["hod_sin_000"]], o[..., self.index["hod_cos_000"]])
        return np.mod(ang, 2 * np.pi) * 12.0 / np.pi


class ActionCodec:
    """Normalized [-1, 1] <-> native conversion for the continuous action vector."""

    def __init__(self, scenario_meta: Mapping[str, Any]):
        self.low = np.asarray(scenario_meta["continuous_action_native_low"], dtype=np.float64)
        self.high = np.asarray(scenario_meta["continuous_action_native_high"], dtype=np.float64)

    def to_native(self, a, idx=slice(None)):
        a = np.asarray(a, dtype=np.float64)
        return self.low[idx] + (a + 1.0) * 0.5 * (self.high[idx] - self.low[idx])

    def to_normalized(self, x, idx=slice(None)):
        x = np.asarray(x, dtype=np.float64)
        return -1.0 + 2.0 * (x - self.low[idx]) / (self.high[idx] - self.low[idx])
