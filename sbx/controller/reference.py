# Extracted verbatim from baseline/example-smart-building.ipynb (cell 2).
# NOT OUR CODE: written by other competitors (Daniel Wu, jazivxt, penguin069 and the notebook author).
# See baseline/README.md for full attribution.
# Public controller by Daniel Wu, reviewed 2026-11-03.
# Source: https://www.kaggle.com/code/danielwuu/cpu-b-conditional-hotwater-20261022
# Embedded policy.py SHA256: 185eccbe60fdfc3728b03ecf834de9a2e58943d241d25ce16a1012a043bb33f6
# Preserved unmodified below; reference-only benchmark, no learned weights.
"""Adaptive, physics-informed HVAC policy for the Smart Buildings challenge.

The only runtime dependency is NumPy.  All learned state belongs to the current
episode.  No building IDs, station IDs, hidden files, or weather traces are used.

The controller identifies a small thermal model for every zone, compensates for
the simulator's delayed zone-temperature observation, then searches feasible
air-handler operating points.  Airflow and reheat are allocated independently to
each zone; fan and hot-water pressure are reset to the smallest useful values.
"""

from __future__ import annotations

import math
import re
from collections import deque
from typing import Any, Mapping

import numpy as np


class TransferPolicy:
    """A transferable controller implementing the official two-method API."""

    name = "adaptive_physics"
    DEFAULTS = {
        "comfort_margin": 0.12,
        "precondition_hours": 1.0,
        "recovery_steps": 1.5,
        "forgetting_factor": 0.997,
        "model_residual_gain": 0.16,
        "comfort_cost": 65000.0,
        "gas_cost_ratio": 0.30,
        "hot_water_target": 345.0,
        "standby_minimum": 282.0,
        "standby_maximum": 307.0,
        "observation_delay": 1,
        "response_scale": 0.75,
        "disturbance_rate": 0.35,
    }
    AIR_CAPACITY = 1.2 * 1006.0  # volumetric heat capacity, J/(m3 K)
    WATER_VALVE_CAPACITY = 4180.0 * 1000.0 * 0.108 / 3600.0
    DESIGN_PRESSURE = 2000.0
    MIN_PRESSURE = 200.0
    MIN_DAMPER = 0.10
    MAX_RECIRCULATION = 0.70
    MAX_DISCHARGE_K = 308.15

    def __init__(
        self,
        building_meta_data: Mapping[str, Any],
        action_spec: Any = None,
        parameters: Mapping[str, float] | None = None,
    ) -> None:
        self.metadata = building_meta_data
        self.action_spec = action_spec
        self.parameters = dict(self.DEFAULTS)
        if parameters:
            unknown = set(parameters) - set(self.parameters)
            if unknown:
                raise ValueError(f"Unknown policy parameters: {sorted(unknown)}")
            self.parameters.update(parameters)
        self.n_continuous = int(building_meta_data["n_continuous_action"])
        self.n_discrete = int(building_meta_data["n_discrete_action"])
        self.field_names = [str(x) for x in building_meta_data["observation_field_names"]]
        self.field_index = {name: i for i, name in enumerate(self.field_names)}
        self.obs_norm = building_meta_data.get("observation_normalization_metadata", {})
        self.action_norm = building_meta_data.get("action_normalization_metadata", {})
        self.temp_indices = list(building_meta_data["temp_indices"])
        self.pressure_indices = list(building_meta_data["pressure_indices"])
        self.recirculation_indices = list(building_meta_data["recirculation_indices"])
        self.water_indices = list(building_meta_data["water_indices"])
        self.num_ahus = len(self.temp_indices)
        zone_actions = building_meta_data["zone_action_indices"]
        self.zone_ids = sorted(zone_actions, key=self._natural_key)
        self.n_zones = len(self.zone_ids)
        if not self.n_zones:
            raise ValueError("Building metadata contains no zone actuators")
        self.zone_number = {name: i for i, name in enumerate(self.zone_ids)}
        self.zone_obs = np.array([
            self.field_index.get(f"{z}_zone_air_temperature_sensor", -1)
            for z in self.zone_ids
        ], dtype=int)
        if np.any(self.zone_obs < 0):
            raise ValueError("A zone actuator has no matching temperature observation")
        self.zone_low_obs = np.array([
            self.field_index.get(f"{z}_zone_air_heating_temperature_setpoint_sensor", -1)
            for z in self.zone_ids
        ], dtype=int)
        self.zone_high_obs = np.array([
            self.field_index.get(f"{z}_zone_air_cooling_temperature_setpoint_sensor", -1)
            for z in self.zone_ids
        ], dtype=int)
        self.zone_flow_obs = np.array([
            self.field_index.get(f"{z}_supply_air_flowrate_setpoint", -1)
            for z in self.zone_ids
        ], dtype=int)
        self.damper_indices = np.array([
            zone_actions[z]["supply_air_damper_percentage_command"] for z in self.zone_ids
        ], dtype=int)
        self.valve_indices = np.array([
            zone_actions[z]["reheat_valve_percentage_command"] for z in self.zone_ids
        ], dtype=int)

        assignments = building_meta_data.get(
            "ahu_zone_assignments", building_meta_data.get("aha_zone_assignments", {})
        )
        self.ahu_ids = sorted(assignments, key=self._natural_key)
        if len(self.ahu_ids) != self.num_ahus:
            raise ValueError("AHU action indices and zone assignments disagree")
        self.groups = []
        self.zone_ahu = np.zeros(self.n_zones, dtype=int)
        self.outside_obs = []
        for i, ahu in enumerate(self.ahu_ids):
            group = np.array([
                self.zone_number[str(room).replace("room_", "VAV_")]
                for room in assignments[ahu]
                if str(room).replace("room_", "VAV_") in self.zone_number
            ], dtype=int)
            self.groups.append(group)
            self.zone_ahu[group] = i
            self.outside_obs.append(self._find_field(
                f"ahs_{ahu}_outside_air_temperature_sensor",
                f"{ahu}_outside_air_temperature_sensor",
                suffix="outside_air_temperature_sensor",
            ))
        self.water_sensor = self._find_field(suffix="supply_water_temperature_sensor")
        self.comfort_now_index = self.field_index.get("comfort_mode_now")
        self.comfort_soon_index = self.field_index.get("comfort_mode_soon")
        self.occupancy_index = self.field_index.get("num_occupants")
        self.comfort_band = tuple(map(float, building_meta_data["comfort_temp_window"]))
        self.eco_band = tuple(map(float, building_meta_data["eco_temp_window"]))
        self.start_hour = float(building_meta_data.get("morning_start_hour", 6))
        self.end_hour = float(building_meta_data.get("evening_start_hour", 19))

        self.zone_area, self.zone_ua = self._geometry_priors(building_meta_data.get("zone_map"))
        # The five-minute effective capacitance includes air, near-surface
        # furnishings, and the part of the fabric coupled over one time step.
        effective_capacity = 3800.0 * self.zone_area + 21000.0 * np.sqrt(self.zone_area)
        self.initial_gain = np.clip(300000.0 / effective_capacity, 0.08, 2.5)
        self.reset()

    @staticmethod
    def _natural_key(value: str) -> tuple:
        return tuple(int(x) if x.isdigit() else x for x in re.split(r"(\d+)", str(value)))

    def _find_field(self, *exact: str, suffix: str = "") -> int | None:
        for name in exact:
            if name in self.field_index:
                return self.field_index[name]
        if suffix:
            for name, index in self.field_index.items():
                if name.endswith(suffix):
                    return index
        return None

    def _geometry_priors(self, zone_map: Any) -> tuple[np.ndarray, np.ndarray]:
        """Read public floor geometry without importing a vision dependency.

        Official maps encode interior=0, wall=1, exterior=2; room numbering is
        the row-major order of four-connected interior components.
        """
        areas = np.full(self.n_zones, 55.0)
        ua = np.full(self.n_zones, 18.0)
        if zone_map is None:
            return areas, ua
        grid = np.asarray(zone_map)
        if grid.ndim != 2 or grid.size > 5000000:
            return areas, ua
        height, width = grid.shape
        inside = (grid == 0).ravel()
        visited = np.zeros(grid.size, dtype=bool)
        flat = grid.ravel()
        counts, external_edges = [], []
        for start in np.flatnonzero(inside):
            if visited[start]:
                continue
            stack = [int(start)]
            visited[start] = True
            count = 0
            edge = 0
            while stack:
                p = stack.pop()
                count += 1
                row, col = divmod(p, width)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = row + dr, col + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    q = nr * width + nc
                    if inside[q] and not visited[q]:
                        visited[q] = True
                        stack.append(q)
                    elif flat[q] == 2:
                        edge += 1
                    elif flat[q] == 1:
                        rr, cc = row + 2 * dr, col + 2 * dc
                        if 0 <= rr < height and 0 <= cc < width and grid[rr, cc] == 2:
                            edge += 1
            counts.append(count)
            external_edges.append(edge)
        if len(counts) != self.n_zones:
            return areas, ua
        for j, name in enumerate(self.zone_ids):
            match = re.search(r"(\d+)$", name)
            label = int(match.group(1)) - 1 if match else j
            if 0 <= label < len(counts):
                areas[j] = max(1.0, 0.04 * counts[label])
                ua[j] = max(0.5, external_edges[label] * 0.2 * 3.0 * 1.3)
        return areas, ua

    def reset(self) -> None:
        self.step_count = 0
        self.history: deque[dict[str, Any]] = deque(maxlen=3)
        self.previous_observed = None
        self.slow_temperature = None
        self.residual = np.zeros(self.n_zones)
        self.model_error = np.full(self.n_zones, 0.08)
        self.theta = np.zeros((self.n_zones, 6))
        self.theta[:, 0] = self.initial_gain * self.zone_area * 2.0 / 1000.0
        self.theta[:, 1] = self.initial_gain * self.zone_ua * 20.0 / 1000.0
        self.theta[:, 2] = 0.18
        self.theta[:, 3] = 0.08
        self.theta[:, 4] = self.initial_gain * self.zone_area * 8.0 / 1000.0
        self.theta[:, 5] = self.initial_gain * self.parameters["response_scale"]
        self.theta[:, :2] *= self.parameters["response_scale"]
        self.theta[:, 4] *= self.parameters["response_scale"]
        prior = np.array([0.12, 0.20, 0.15, 0.15, 0.16, 0.8])
        self.covariance = np.repeat(np.diag(prior)[None, :, :], self.n_zones, axis=0)
        self.last_heating_step = -1000
        self.last_action = None

    def _temperature(self, obs: np.ndarray, index: int | None, kind: str, fallback: float) -> float:
        if index is None:
            return fallback
        value = float(obs[index])
        # In the published normalizer, AHU-prefixed measurement names are
        # exact-match misses and remain native.  Physical-range detection also
        # supports a future gateway that normalizes those fields correctly.
        if 150.0 < value < 450.0:
            return value
        norm = self.obs_norm.get(kind, {})
        mean = float(norm.get("sample_mean", fallback))
        std = float(norm.get("sample_std", math.sqrt(norm.get("sample_variance", 1.0))))
        value = value * std + mean
        return value if np.isfinite(value) and 150.0 < value < 450.0 else fallback

    def _zone_temperatures(self, obs: np.ndarray) -> np.ndarray:
        norm = self.obs_norm.get("zone_air_temperature_sensor", {})
        mean = float(norm.get("sample_mean", 295.0))
        std = float(norm.get("sample_std", math.sqrt(norm.get("sample_variance", 25.0))))
        raw = obs[self.zone_obs].astype(float)
        value = np.where((raw > 150.0) & (raw < 450.0), raw, raw * std + mean)
        fallback = self.previous_observed if self.previous_observed is not None else mean
        return np.where(np.isfinite(value) & (value > 150.0) & (value < 450.0), value, fallback)

    def _hour(self, obs: np.ndarray) -> float:
        sine = self.field_index.get("hod_sin_000")
        cosine = self.field_index.get("hod_cos_000")
        if sine is None or cosine is None:
            return (self.start_hour + self.step_count / 12.0) % 24.0
        return (math.atan2(float(obs[sine]), float(obs[cosine])) % (2.0 * math.pi)) * 12.0 / math.pi

    def _band(self, obs: np.ndarray, temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        now = bool(obs[self.comfort_now_index] > 0.5) if self.comfort_now_index is not None else (
            self.start_hour <= self._hour(obs) < self.end_hour
        )
        soon = bool(obs[self.comfort_soon_index] > 0.5) if self.comfort_soon_index is not None else now
        default = self.comfort_band if now else self.eco_band
        low = np.full(self.n_zones, default[0])
        high = np.full(self.n_zones, default[1])
        valid = (self.zone_low_obs >= 0) & (self.zone_high_obs >= 0)
        if np.any(valid):
            sensor_low = obs[self.zone_low_obs[valid]].astype(float)
            sensor_high = obs[self.zone_high_obs[valid]].astype(float)
            good = (sensor_low > 150.0) & (sensor_high < 450.0) & (sensor_high > sensor_low)
            indices = np.flatnonzero(valid)[good]
            low[indices] = sensor_low[good]
            high[indices] = sensor_high[good]
        # Band sensors, like zone temperatures, are one step old at transitions.
        if now:
            low = np.maximum(low, self.comfort_band[0])
            high = np.minimum(high, self.comfort_band[1])
        elif soon:
            hour = self._hour(obs)
            remaining = max(0.0, (self.start_hour - hour) % 24.0)
            progress = np.clip(1.0 - remaining / max(0.1, self.parameters["precondition_hours"]), 0.0, 1.0)
            low = self.eco_band[0] + progress * (self.comfort_band[0] - self.eco_band[0]) + np.zeros(self.n_zones)
            high = self.eco_band[1] + progress * (self.comfort_band[1] - self.eco_band[1]) + np.zeros(self.n_zones)
        else:
            # With no occupants, comfort does not contribute to reward. Keep a
            # recoverable building but permit useful night/weekend setback.
            low[:] = self.parameters["standby_minimum"]
            high[:] = self.parameters["standby_maximum"]
        margin = self.parameters["comfort_margin"] if (now or soon) else 0.0
        low += margin
        high -= margin
        # Do not request the entire recovery in a single five-minute step.
        recovery = max(1.0, self.parameters["recovery_steps"])
        low = np.minimum(low, temperature + np.maximum((low - temperature) / recovery, 0.0))
        high = np.maximum(high, temperature - np.maximum((temperature - high) / recovery, 0.0))
        return low, high, now

    def _group_temperature(self, temperature: np.ndarray) -> np.ndarray:
        means = np.zeros(self.n_zones)
        for group in self.groups:
            if group.size:
                means[group] = np.average(temperature[group], weights=self.zone_area[group])
        return means

    def _features(self, temperature: np.ndarray, outside: np.ndarray, mode: bool, slow: np.ndarray) -> np.ndarray:
        return np.column_stack((
            np.ones(self.n_zones),
            np.clip((outside - temperature) / 20.0, -5.0, 5.0),
            np.clip((slow - temperature) / 5.0, -5.0, 5.0),
            np.clip((self._group_temperature(temperature) - temperature) / 5.0, -5.0, 5.0),
            np.full(self.n_zones, float(mode)),
        ))

    def _learn(self, observed: np.ndarray) -> None:
        delay = int(self.parameters["observation_delay"])
        if self.previous_observed is None or len(self.history) <= delay:
            return
        applied = list(self.history)[-(delay + 1)]
        old = self.previous_observed
        power = (applied["conductance"] * (applied["supply"] - observed) + applied["reheat"]) / 1000.0
        features = self._features(old, applied["outside"], applied["mode"], applied["slow"])
        x = np.column_stack((features, power))
        delta = np.clip(observed - old, -15.0, 15.0)
        prediction = np.einsum("ij,ij->i", self.theta, x)
        remaining = delta - prediction
        rate = float(self.parameters["disturbance_rate"])
        self.model_error = 0.94 * self.model_error + 0.06 * np.minimum(np.abs(remaining - self.residual), 2.0)
        self.residual = np.clip((1.0 - rate) * self.residual + rate * remaining, -5.0, 5.0)

    def _free_drift(self, temperature: np.ndarray, outside: np.ndarray, mode: bool) -> np.ndarray:
        features = self._features(temperature, outside, mode, self.slow_temperature)
        return np.clip(np.einsum("ij,ij->i", self.theta[:, :5], features) + self.residual, -5.0, 5.0)

    def _native_action(self, kind: str, value: Any, low: float, high: float) -> Any:
        spec = self.action_norm.get(kind)
        if spec is None:
            spec = next((v for k, v in self.action_norm.items() if str(k).endswith(kind)), None)
        if spec:
            low = float(spec.get("min_native_value", low))
            high = float(spec.get("max_native_value", high))
            out_low = float(spec.get("min_normalized_value", -1.0))
            out_high = float(spec.get("max_normalized_value", 1.0))
        else:
            out_low, out_high = -1.0, 1.0
        return np.clip(out_low + (np.asarray(value) - low) * (out_high - out_low) / max(high - low, 1e-12), out_low, out_high)

    def _choose_ahu(self, group: np.ndarray, temperature: np.ndarray, free: np.ndarray,
                    low: np.ndarray, high: np.ndarray, max_flow: np.ndarray,
                    outside: float, water: float) -> dict[str, Any]:
        """Joint supply-temperature, economizer, airflow, and reheat search."""
        t, d, lo, hi = temperature[group], free[group], low[group], high[group]
        response = self.theta[group, 5]
        capacity = max_flow[group]
        inside = float(np.average(t, weights=self.zone_area[group]))
        targets = np.array([285.0, 287.0, 289.0, 291.0, 293.0, 295.0, 297.0, 299.0, 302.0, 305.0])
        targets = np.concatenate((targets, [float(np.clip(np.min(hi) - 1.5, 285.0, 305.0))]))
        setpoint = np.repeat(targets, 2)
        recirc = np.full_like(setpoint, self.MAX_RECIRCULATION)
        if outside < inside - 0.5:
            recirc[1::2] = np.clip((targets - outside) / max(inside - outside, 1e-6), 0.0, self.MAX_RECIRCULATION)
        mixed = recirc * inside + (1.0 - recirc) * outside
        supply = np.minimum(mixed, setpoint)
        sa = supply[:, None]
        # Invert the implicit thermal model at the two comfort boundaries.
        cool_need = np.maximum(t + d - hi, 0.0) * 1000.0 / response
        heat_need = np.maximum(lo - t - d, 0.0) * 1000.0 / response
        cooling_flow = cool_need[None, :] / (self.AIR_CAPACITY * np.maximum(hi[None, :] - sa, 0.05))
        heating_flow = heat_need[None, :] / (self.AIR_CAPACITY * np.maximum(self.MAX_DISCHARGE_K - lo[None, :], 0.5))
        # Once the water coil saturates, more cold supply air reduces net
        # heating. The maximum useful heating flow occurs where the coil can
        # just lift the supply stream to the discharge-temperature limit.
        maximum_water_heat = self.WATER_VALVE_CAPACITY * math.sqrt(20.0) * np.maximum(water - sa, 0.0)
        maximum_heating_flow = maximum_water_heat / (
            self.AIR_CAPACITY * np.maximum(self.MAX_DISCHARGE_K - sa, 0.1)
        )
        heating_flow = np.minimum(heating_flow, maximum_heating_flow)
        needed_flow = np.maximum(cooling_flow, heating_flow)
        fraction = np.max(needed_flow / capacity[None, :], axis=1)
        pressure = np.clip(self.DESIGN_PRESSURE * fraction ** 2, self.MIN_PRESSURE, self.DESIGN_PRESSURE)
        pressure_ratio = np.sqrt(pressure / self.DESIGN_PRESSURE)
        dampers = np.clip(needed_flow / (capacity[None, :] * pressure_ratio[:, None]), self.MIN_DAMPER, 1.0)
        flow = dampers * capacity[None, :] * pressure_ratio[:, None]
        conductance = flow * self.AIR_CAPACITY
        water_gap = np.maximum(water - sa, 0.0)
        wanted_reheat = np.maximum((lo[None, :] - t[None, :] - d[None, :]) * 1000.0 / response[None, :]
                                    + conductance * (lo[None, :] - sa), 0.0)
        reheat_limit = np.minimum(
            conductance * np.maximum(self.MAX_DISCHARGE_K - sa, 0.0),
            self.WATER_VALVE_CAPACITY * math.sqrt(20.0) * water_gap,
        )
        reheat = np.minimum(wanted_reheat, reheat_limit)
        predicted = (t[None, :] + d[None, :] + response[None, :] * (conductance * sa + reheat) / 1000.0) / (
            1.0 + response[None, :] * conductance / 1000.0
        )
        violation = np.maximum(lo[None, :] - predicted, 0.0) + np.maximum(predicted - hi[None, :], 0.0)
        thermal_penalty = np.sum(violation + 2.0 * violation ** 2, axis=1)
        total_flow = flow.sum(axis=1)
        fan = total_flow * pressure * (2.0 - recirc) / 0.9
        chiller = total_flow * self.AIR_CAPACITY * np.maximum(mixed - supply, 0.0) / 3.5
        gas = reheat.sum(axis=1)
        objective = fan + chiller + self.parameters["gas_cost_ratio"] * gas
        objective += self.parameters["comfort_cost"] * thermal_penalty
        # The off candidate removes the enforced ventilation floor when the
        # building's own thermal inertia can keep this AHU's zones comfortable.
        off_prediction = t + d
        off_violation = np.maximum(lo - off_prediction, 0.0) + np.maximum(off_prediction - hi, 0.0)
        off_objective = self.parameters["comfort_cost"] * np.sum(off_violation + 2.0 * off_violation ** 2)
        best = int(np.argmin(objective))
        if off_objective <= objective[best]:
            return {"run": 0, "pressure": 0.0, "setpoint": 305.0, "recirc": 0.7,
                    "damper": np.full(group.size, self.MIN_DAMPER), "conductance": np.zeros(group.size),
                    "supply": np.full(group.size, mixed[best]), "reheat": np.zeros(group.size)}
        return {"run": 1, "pressure": float(pressure[best]), "setpoint": float(setpoint[best]),
                "recirc": float(recirc[best]), "damper": dampers[best],
                "conductance": conductance[best], "supply": np.full(group.size, supply[best]),
                "reheat": reheat[best]}

    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        obs = np.asarray(obs, dtype=float).reshape(-1)
        if obs.size != len(self.field_names):
            raise ValueError(f"Observation has {obs.size} fields; expected {len(self.field_names)}")
        observed = self._zone_temperatures(obs)
        if self.slow_temperature is None:
            self.slow_temperature = observed.copy()
        self._learn(observed)
        outside_ahu = np.array([
            self._temperature(obs, index, "outside_air_temperature_sensor", 290.0)
            for index in self.outside_obs
        ])
        outside = outside_ahu[self.zone_ahu]
        mode = bool(obs[self.comfort_now_index] > 0.5) if self.comfort_now_index is not None else True
        temperature = observed.copy()
        if self.history and int(self.parameters["observation_delay"]) == 1:
            last = self.history[-1]
            drift = self._free_drift(observed, last["outside"], last["mode"])
            gain = self.theta[:, 5] / 1000.0
            temperature = (observed + drift + gain * (last["conductance"] * last["supply"] + last["reheat"])) / (
                1.0 + gain * last["conductance"]
            )
            temperature = np.clip(temperature, observed - 8.0, observed + 8.0)
        low, high, mode = self._band(obs, temperature)
        free = self._free_drift(temperature, outside, mode)
        max_flow = np.full(self.n_zones, 2.0)
        valid = self.zone_flow_obs >= 0
        if np.any(valid):
            flow_reading = obs[self.zone_flow_obs[valid]]
            max_flow[valid] = np.where(np.isfinite(flow_reading) & (flow_reading > 0.0), flow_reading, 2.0)
        max_flow = np.clip(max_flow, 0.01, 100.0)
        water_observed = self._temperature(obs, self.water_sensor, "supply_water_temperature_sensor", 340.0)
        heat_demand = np.any(temperature + free < low)
        if heat_demand:
            self.last_heating_step = self.step_count
        recent_heat = self.step_count - self.last_heating_step < 24
        water_target = self.parameters["hot_water_target"] if recent_heat else 310.0
        water_available = water_observed
        continuous = np.full(self.n_continuous, -1.0, dtype=np.float32)
        discrete = np.zeros(self.n_discrete, dtype=np.int32)
        conductance = np.zeros(self.n_zones)
        supply = outside.copy()
        reheat = np.zeros(self.n_zones)
        for i, group in enumerate(self.groups):
            if not group.size:
                continue
            choice = self._choose_ahu(group, temperature, free, low, high, max_flow,
                                      float(outside_ahu[i]), float(water_available))
            discrete[i] = choice["run"]
            continuous[self.temp_indices[i]] = self._native_action(
                f"{self.ahu_ids[i]}_supply_air_temperature_setpoint", choice["setpoint"], 285.0, 305.0)
            continuous[self.pressure_indices[i]] = self._native_action(
                f"{self.ahu_ids[i]}_supply_air_static_pressure_setpoint", choice["pressure"], 0.0, 2000.0)
            continuous[self.recirculation_indices[i]] = self._native_action(
                f"{self.ahu_ids[i]}_recirculation_percentage_command", choice["recirc"], 0.0, 1.0)
            continuous[self.damper_indices[group]] = self._native_action(
                "supply_air_damper_percentage_command", choice["damper"], 0.0, 1.0)
            conductance[group] = choice["conductance"]
            supply[group] = choice["supply"]
            reheat[group] = choice["reheat"]
        if np.any(reheat > 10.0):
            # Reheat can also be needed to protect a satisfied zone from the
            # cold air supplied to its neighbour. Keep the boiler available in
            # that case even when there was no free-floating heating demand.
            self.last_heating_step = self.step_count
            water_target = self.parameters["hot_water_target"]
        # The valve law is Q = Kv * sqrt(dP) * rho_w * cp_w * (Tw - Ta).
        # Choose pressure from the most demanding valve, then allocate each
        # valve its own heat.  This avoids running the pump at 20 bar by habit.
        valve_need = reheat / (self.WATER_VALVE_CAPACITY * np.maximum(water_available - supply, 1.0))
        pressure_water = float(np.clip(np.max(valve_need) ** 2 * 1.05, 0.0, 20.0))
        if pressure_water > 1e-8:
            valves = np.clip(valve_need / math.sqrt(pressure_water), 0.0, 1.0)
            discrete[-1] = 1
        else:
            valves = np.zeros(self.n_zones)
        continuous[self.valve_indices] = self._native_action("reheat_valve_percentage_command", valves, 0.0, 1.0)
        if self.water_indices:
            continuous[self.water_indices[0]] = self._native_action("supply_water_temperature_setpoint", water_target, 310.0, 350.0)
        if len(self.pressure_indices) > self.num_ahus:
            continuous[self.pressure_indices[-1]] = self._native_action("differential_pressure_setpoint", pressure_water, 0.0, 20.0)
        # Record the physically delivered quantity, including actuator caps.
        actual_reheat = np.minimum(
            self.WATER_VALVE_CAPACITY * math.sqrt(pressure_water) * valves * np.maximum(water_available - supply, 0.0),
            conductance * np.maximum(self.MAX_DISCHARGE_K - supply, 0.0),
        )
        self.history.append({"conductance": conductance, "supply": supply,
                             "reheat": actual_reheat, "outside": outside.copy(),
                             "mode": mode, "slow": self.slow_temperature.copy()})
        self.slow_temperature += (temperature - self.slow_temperature) / 48.0
        self.previous_observed = observed.copy()
        self.step_count += 1
        action = {"continuous_action": np.clip(np.nan_to_num(continuous, nan=-1.0), -1.0, 1.0).astype(np.float32),
                  "discrete_action": discrete.astype(np.int32)}
        self.last_action = action
        return action



"""Deterministic thermal-control adaptations, developed against frozen V35.

Attribution: Daniel Wu's public analytical controller supplies the zone model,
observer, SAT/mixing search, fan reset and hydraulic allocation, unchanged in
public_daniel_thermal.py. No neural weights or RL are loaded here.
https://www.kaggle.com/code/danielwuu/cpu-b-conditional-hotwater-20261022
https://www.kaggle.com/code/danielwuu/cpu-a-alpha075-control-20261022
V35's telemetry principle is retained: reject nonphysical data and keep acting.
https://github.com/smart-buildings-2026/starter-kit

The public observer assumes T_next=(T+d+g*(G*T_supply+H))/(1+g*G).
Here G=rho*cp*q [W/K], H [W], g [K/W per decision], d [K/decision].
V41 tests bounded recursive least squares for the six observable heat-balance
coefficients, with exponential forgetting, instead of fixed geometry priors.
P <- (P-K*x.T*P)/lambda; K=P*x/(lambda+x.T*P*x).
Bounds and clipped innovations are engineering hypotheses, not calibrated
confidence intervals. Native five-minute reward, physics and limits decide gains.
V42 tests a modest reserve derived from past prediction error. That error is
not calibrated uncertainty and this reserve is not a comfort guarantee.
"""
import numpy as np


class GuardedJointThermalPolicy(TransferPolicy):
    name = 'facilities_v40_joint_thermal'

    def _zone_temperatures(self, obs):
        value = super()._zone_temperatures(obs)
        previous = self.previous_observed if self.previous_observed is not None else np.mean(self.comfort_band)
        return np.where(np.isfinite(value) & (value >= 150.) & (value <= 400.), value, previous)

    def get_action(self, obs):
        action = super().get_action(obs)
        self.last_diagnostics = dict(model_error_mean_K=float(np.mean(self.model_error)),
            model_residual_mean_K=float(np.mean(self.residual)),
            thermal_response_mean_K_per_kW=float(np.mean(self.theta[:, 5])))
        return action


def bounded_rls(theta, covariance, x, target, lower, upper, forgetting=.997):
    """Independent per-zone least-squares updates; caller supplies past data only."""
    px = np.einsum('nij,nj->ni', covariance, x)
    denominator = forgetting + np.einsum('ni,ni->n', x, px)
    gain = px / np.maximum(denominator[:, None], 1e-9)
    innovation = np.clip(target - np.einsum('ni,ni->n', theta, x), -2., 2.)
    fitted = np.clip(theta + gain * innovation[:, None], lower, upper)
    updated = (covariance - np.einsum('ni,nj->nij', gain, px)) / forgetting
    updated = .5 * (updated + updated.transpose(0, 2, 1))
    return fitted, updated


class IdentifiedJointThermalPolicy(GuardedJointThermalPolicy):
    name = 'facilities_v41_identified_thermal'

    def reset(self):
        super().reset()
        self._thermal_prior = self.theta[:, 5].copy()

    def _learn(self, observed):
        delay = int(self.parameters['observation_delay'])
        if self.previous_observed is None or len(self.history) <= delay:
            return
        applied = list(self.history)[-(delay + 1)]
        power = (applied['conductance'] * (applied['supply'] - observed) + applied['reheat']) / 1000.
        features = self._features(self.previous_observed, applied['outside'], applied['mode'], applied['slow'])
        x = np.column_stack((features, power))
        delta = observed - self.previous_observed
        valid = np.isfinite(x).all(axis=1) & np.isfinite(delta) & (np.abs(delta) <= 3.)
        lower = np.zeros_like(self.theta)
        upper = np.ones_like(self.theta) * np.array([2., 5., 2., 2., 2., 1.])
        lower[:, 0] = -2.
        lower[:, 5] = .5 * self._thermal_prior
        upper[:, 5] = 1.5 * self._thermal_prior
        if np.any(valid):
            fit, covariance = bounded_rls(self.theta[valid], self.covariance[valid], x[valid],
                delta[valid] - self.residual[valid], lower[valid], upper[valid],
                float(self.parameters['forgetting_factor']))
            self.theta[valid], self.covariance[valid] = fit, covariance
        remaining = np.clip(delta - np.einsum('ni,ni->n', self.theta, x), -5., 5.)
        self.model_error = .94 * self.model_error + .06 * np.minimum(np.abs(remaining-self.residual), 2.)
        rate = float(self.parameters['disturbance_rate'])
        self.residual = np.clip((1-rate)*self.residual + rate*remaining, -5., 5.)


class ErrorReserveJointThermalPolicy(GuardedJointThermalPolicy):
    name = 'facilities_v42_error_reserve_thermal'

    def _band(self, obs, temperature):
        low, high, mode = super()._band(obs, temperature)
        if mode:
            extra = np.minimum(np.clip(.5*(self.model_error-.08), 0., .25), .25*np.maximum(high-low, 0.))
            low, high = low+extra, high-extra
        return low, high, mode


NOV03_MODELS = {c.name: c for c in (GuardedJointThermalPolicy, IdentifiedJointThermalPolicy,
                                   ErrorReserveJointThermalPolicy)}


"""Four isolated operating-point variations of frozen V42, authorized Nov 3.

Analytical controller attribution and native model equations are preserved in
public_daniel_thermal.py and facilities_nov03_models.py.
https://www.kaggle.com/code/danielwuu/cpu-a-alpha075-control-20261022
These constants are control hypotheses evaluated with the native reward.
"""


class LeanReserveThermalPolicy(ErrorReserveJointThermalPolicy):
    name = 'facilities_v43_lean_reserve_thermal'
    DEFAULTS = dict(ErrorReserveJointThermalPolicy.DEFAULTS, comfort_margin=.08)


class WiderReserveThermalPolicy(ErrorReserveJointThermalPolicy):
    name = 'facilities_v44_wider_reserve_thermal'
    DEFAULTS = dict(ErrorReserveJointThermalPolicy.DEFAULTS, comfort_margin=.20)


class GradualRecoveryThermalPolicy(ErrorReserveJointThermalPolicy):
    name = 'facilities_v45_gradual_recovery_thermal'
    DEFAULTS = dict(ErrorReserveJointThermalPolicy.DEFAULTS, recovery_steps=2.)


class HeatingCostThermalPolicy(ErrorReserveJointThermalPolicy):
    name = 'facilities_v46_heating_cost_thermal'
    DEFAULTS = dict(ErrorReserveJointThermalPolicy.DEFAULTS, gas_cost_ratio=.45)


NOV03_RELEASE = {42: ErrorReserveJointThermalPolicy, 43: LeanReserveThermalPolicy,
                 44: WiderReserveThermalPolicy, 45: GradualRecoveryThermalPolicy,
                 46: HeatingCostThermalPolicy}

class IdentifiedReservePolicy(IdentifiedJointThermalPolicy, ErrorReserveJointThermalPolicy):
    name = "identified_reserve"

from collections import deque
from typing import Any, Mapping
import copy
import math
import numpy as np
import re
class CausalIdentifiedReservePolicy(IdentifiedReservePolicy):
    """Keep the supplied bounded fit; measure actual forecast error first.

    Computing reserve from the already-fitted equation residual can conceal
    prediction error. Use the old coefficients to predict the newly received
    temperature, then fit them and update the reserve from that prior forecast.
    All inputs are observations and actions already available in this episode.
    """

    name = "causal_identified"

    def _learn(self, observed):
        delay = int(self.parameters["observation_delay"])
        if self.previous_observed is None or len(self.history) <= delay:
            return
        applied = list(self.history)[-(delay + 1)]
        old = self.previous_observed
        features = self._features(old, applied["outside"], applied["mode"], applied["slow"])
        drift = np.einsum("ij,ij->i", self.theta[:, :5], features) + self.residual
        gain = self.theta[:, 5] / 1000.0
        prediction = (old + drift + gain * (applied["conductance"] * applied["supply"] + applied["reheat"])) / (
            1.0 + gain * applied["conductance"])
        error = np.where(np.isfinite(observed - prediction), np.minimum(np.abs(observed - prediction), 2.0), 2.0)
        previous_error = self.model_error.copy()
        super()._learn(observed)
        self.model_error = 0.94 * previous_error + 0.06 * error

class WarmStandbyPolicy(CausalIdentifiedReservePolicy):
    """Keep the existing hot setpoint for 36 hours after measured heat demand.

    The public controller drops the requested water temperature after two
    hours without demand. Reheating that reservoir the following morning can
    dominate short cold-period losses. A longer hold costs tank dissipation
    but avoids daily forced cooling and repeated recovery of stored heat.
    Demand timing is episode-local; no future weather or station ID is read.
    """

    name = "causal_warm_standby"
    HOLD_STEPS = 36 * 12

    def _native_action(self, kind, value, low, high):
        if (kind == "supply_water_temperature_setpoint"
                and self.step_count - self.last_heating_step < self.HOLD_STEPS):
            value = self.parameters["hot_water_target"]
        return super()._native_action(kind, value, low, high)

class WeekendWaterHoldPolicy(WarmStandbyPolicy):
    """Carry the warm reservoir through a longer pause after heat demand."""

    name = "warm_hold72"
    HOLD_STEPS = 72 * 12

class DeepFastRecoveryPolicy(WeekendWaterHoldPolicy):
    """Use a 274 K empty-period floor and one-step requested band recovery."""

    name = "hold72_deep_fast_recovery"
    DEFAULTS = dict(WeekendWaterHoldPolicy.DEFAULTS, standby_minimum=274.0, recovery_steps=1.0)

class EmptyDayConsistentPolicy(DeepFastRecoveryPolicy):
    """Use the empty-day band but keep the real schedule mode in the model."""

    name = "deep_empty_day_consistent_model"
    EMPTY_CONFIRMATION_HOURS = 2.0

    def reset(self):
        super().reset()
        self.empty_day_relaxed_steps = 0

    def _band(self, obs, temperature):
        low, high, schedule_mode = super()._band(obs, temperature)
        if not schedule_mode or self.occupancy_index is None:
            return low, high, schedule_mode
        occupants = float(obs[self.occupancy_index])
        elapsed = (self._hour(obs) - self.start_hour) % 24.0
        scheduled_duration = (self.end_hour - self.start_hour) % 24.0
        if (not np.isfinite(occupants) or occupants != 0.0
                or not self.EMPTY_CONFIRMATION_HOURS <= elapsed < scheduled_duration):
            return low, high, schedule_mode
        self.empty_day_relaxed_steps += 1
        # Empty low/high limits control the AHU allocation independently of
        # this bit. Schedule-driven internal heat is still physically active,
        # so keep its real mode for free drift and history['mode']. Historical
        # learning/observer calls then naturally use their own stored modes.
        return (np.full(self.n_zones, self.parameters["standby_minimum"]),
                np.full(self.n_zones, self.parameters["standby_maximum"]), schedule_mode)

    def experiment_diagnostics(self):
        return {"empty_day_relaxed_steps": self.empty_day_relaxed_steps}

class FineSupplySearchPolicy(WarmStandbyPolicy):
    """Enrich the supplied SAT search from coarse points to a 0.5 K grid."""

    name = "warm_fine_sat"

    def _choose_ahu(self, group, temperature, free, low, high, max_flow, outside, water):
        # The physical equations, limits and cost weights below match V42.
        # The 0.5 K grid includes every original fixed setpoint candidate.
        t, d, lo, hi = temperature[group], free[group], low[group], high[group]
        response = self.theta[group, 5]
        capacity = max_flow[group]
        inside = float(np.average(t, weights=self.zone_area[group]))
        targets = np.arange(285.0, 305.01, 0.5)
        targets = np.concatenate((targets, [float(np.clip(np.min(hi) - 1.5, 285.0, 305.0))]))
        setpoint = np.repeat(targets, 2)
        recirc = np.full_like(setpoint, self.MAX_RECIRCULATION)
        if outside < inside - 0.5:
            recirc[1::2] = np.clip((targets - outside) / max(inside - outside, 1e-6), 0.0, self.MAX_RECIRCULATION)
        mixed = recirc * inside + (1.0 - recirc) * outside
        supply = np.minimum(mixed, setpoint)
        sa = supply[:, None]
        cool_need = np.maximum(t + d - hi, 0.0) * 1000.0 / response
        heat_need = np.maximum(lo - t - d, 0.0) * 1000.0 / response
        cooling_flow = cool_need[None, :] / (self.AIR_CAPACITY * np.maximum(hi[None, :] - sa, 0.05))
        heating_flow = heat_need[None, :] / (self.AIR_CAPACITY * np.maximum(self.MAX_DISCHARGE_K - lo[None, :], 0.5))
        maximum_water_heat = self.WATER_VALVE_CAPACITY * math.sqrt(20.0) * np.maximum(water - sa, 0.0)
        maximum_heating_flow = maximum_water_heat / (self.AIR_CAPACITY * np.maximum(self.MAX_DISCHARGE_K - sa, 0.1))
        heating_flow = np.minimum(heating_flow, maximum_heating_flow)
        needed_flow = np.maximum(cooling_flow, heating_flow)
        fraction = np.max(needed_flow / capacity[None, :], axis=1)
        pressure = np.clip(self.DESIGN_PRESSURE * fraction ** 2, self.MIN_PRESSURE, self.DESIGN_PRESSURE)
        pressure_ratio = np.sqrt(pressure / self.DESIGN_PRESSURE)
        dampers = np.clip(needed_flow / (capacity[None, :] * pressure_ratio[:, None]), self.MIN_DAMPER, 1.0)
        flow = dampers * capacity[None, :] * pressure_ratio[:, None]
        conductance = flow * self.AIR_CAPACITY
        water_gap = np.maximum(water - sa, 0.0)
        wanted_reheat = np.maximum((lo[None, :] - t[None, :] - d[None, :]) * 1000.0 / response[None, :]
                                    + conductance * (lo[None, :] - sa), 0.0)
        reheat_limit = np.minimum(conductance * np.maximum(self.MAX_DISCHARGE_K - sa, 0.0),
                                  self.WATER_VALVE_CAPACITY * math.sqrt(20.0) * water_gap)
        reheat = np.minimum(wanted_reheat, reheat_limit)
        predicted = (t[None, :] + d[None, :] + response[None, :] * (conductance * sa + reheat) / 1000.0) / (
            1.0 + response[None, :] * conductance / 1000.0)
        violation = np.maximum(lo[None, :] - predicted, 0.0) + np.maximum(predicted - hi[None, :], 0.0)
        thermal_penalty = np.sum(violation + 2.0 * violation ** 2, axis=1)
        total_flow = flow.sum(axis=1)
        fan = total_flow * pressure * (2.0 - recirc) / 0.9
        chiller = total_flow * self.AIR_CAPACITY * np.maximum(mixed - supply, 0.0) / 3.5
        gas = reheat.sum(axis=1)
        objective = fan + chiller + self.parameters["gas_cost_ratio"] * gas
        objective += self.parameters["comfort_cost"] * thermal_penalty
        off_prediction = t + d
        off_violation = np.maximum(lo - off_prediction, 0.0) + np.maximum(off_prediction - hi, 0.0)
        off_objective = self.parameters["comfort_cost"] * np.sum(off_violation + 2.0 * off_violation ** 2)
        best = int(np.argmin(objective))
        if off_objective <= objective[best]:
            return {"run": 0, "pressure": 0.0, "setpoint": 305.0, "recirc": 0.7,
                    "damper": np.full(group.size, self.MIN_DAMPER), "conductance": np.zeros(group.size),
                    "supply": np.full(group.size, mixed[best]), "reheat": np.zeros(group.size)}
        return {"run": 1, "pressure": float(pressure[best]), "setpoint": float(setpoint[best]),
                "recirc": float(recirc[best]), "damper": dampers[best], "conductance": conductance[best],
                "supply": np.full(group.size, supply[best]), "reheat": reheat[best]}

class EmptyDayConsistentFineLeanPolicy(EmptyDayConsistentPolicy):
    """Combine consistent empty-day features with fine SAT and a lean margin."""

    name = "deep_empty_consistent_fine_margin08"
    DEFAULTS = dict(DeepFastRecoveryPolicy.DEFAULTS, comfort_margin=.08)

    def _choose_ahu(self, group, temperature, free, low, high, max_flow, outside, water):
        return FineSupplySearchPolicy._choose_ahu(
            self, group, temperature, free, low, high, max_flow, outside, water)

class StableMarginPolicy(EmptyDayConsistentFineLeanPolicy):
    """Halve fixed padding only for zones with small causal forecast error."""

    name = "deep_empty_stable_margin04"
    ERROR_THRESHOLD_K = .01
    FIXED_MARGIN_FRACTION = .5

    def reset(self):
        super().reset()
        self.adaptive_margin_band_calls = 0
        self.adaptive_margin_zone_decisions = 0

    def _eligible_margin_zones(self):
        error = self.model_error
        return np.isfinite(error) & (error >= 0.0) & (error <= self.ERROR_THRESHOLD_K)

    def _band(self, obs, temperature):
        previous_empty_count = self.empty_day_relaxed_steps
        low, high, physical_mode = super()._band(obs, temperature)
        # EmptyDayConsistentPolicy deliberately replaces both bands with
        # exact standby limits. A changed counter identifies this call, even
        # if earlier observations already triggered empty-day relaxation.
        if self.empty_day_relaxed_steps != previous_empty_count:
            return low, high, physical_mode
        soon = (bool(obs[self.comfort_soon_index] > .5)
                if self.comfort_soon_index is not None else bool(physical_mode))
        if not (physical_mode or soon):
            return low, high, physical_mode
        reduction = max(0.0, float(self.parameters["comfort_margin"])) * (1.0 - self.FIXED_MARGIN_FRACTION)
        eligible = self._eligible_margin_zones()
        if not reduction or not np.any(eligible):
            return low, high, physical_mode
        # Eligibility is below the inherited .08 K error-reserve threshold,
        # so the reserve is exactly zero for these zones. Keep all other
        # bands untouched. Undo/reapply only the inherited recovery transform
        # to remove fixed padding at its original point before recovery; this
        # also preserves prestart semantics for configured recovery >1.
        recovery = max(1.0, float(self.parameters["recovery_steps"]))
        t = temperature[eligible]
        old_low, old_high = low[eligible], high[eligible]
        target_low = np.where(old_low > t, t + recovery * (old_low - t), old_low) - reduction
        target_high = np.where(old_high < t, t + recovery * (old_high - t), old_high) + reduction
        low = low.copy()
        high = high.copy()
        low[eligible] = np.minimum(target_low, t + np.maximum((target_low - t) / recovery, 0.0))
        high[eligible] = np.maximum(target_high, t - np.maximum((t - target_high) / recovery, 0.0))
        self.adaptive_margin_band_calls += 1
        self.adaptive_margin_zone_decisions += int(np.count_nonzero(eligible))
        return low, high, physical_mode

    def experiment_diagnostics(self):
        return {**super().experiment_diagnostics(),
                "adaptive_margin_band_calls": self.adaptive_margin_band_calls,
                "adaptive_margin_zone_decisions": self.adaptive_margin_zone_decisions}

class VacancyBoundaryTolerancePolicy(EmptyDayConsistentFineLeanPolicy):
    """Correct only roundoff at a vacancy boundary, leaving prestart intact."""

    CLOCK_TOLERANCE_HOURS = 1e-6  # 3.6 ms; far below a five-minute decision.

    def _band(self, obs, temperature):
        before = self.empty_day_relaxed_steps
        low, high, mode = super()._band(obs, temperature)
        if (not mode or self.empty_day_relaxed_steps != before
                or self.occupancy_index is None):
            return low, high, mode
        occupants = float(obs[self.occupancy_index])
        if not np.isfinite(occupants) or occupants != 0.0:
            return low, high, mode
        raw_elapsed = self._hour(obs) - self.start_hour
        elapsed = 0.0 if abs(raw_elapsed) <= self.CLOCK_TOLERANCE_HOURS else raw_elapsed % 24.0
        duration = (self.end_hour - self.start_hour) % 24.0
        if (abs(elapsed - self.EMPTY_CONFIRMATION_HOURS) <= self.CLOCK_TOLERANCE_HOURS
                and elapsed < duration):
            self.empty_day_relaxed_steps += 1
            return (np.full(self.n_zones, self.parameters["standby_minimum"]),
                    np.full(self.n_zones, self.parameters["standby_maximum"]), mode)
        return low, high, mode

class StableHalfHourVacancyPolicy(StableMarginPolicy, VacancyBoundaryTolerancePolicy):
    """Compose stable half-padding with thirty-minute current-zero timing."""

    name = "deep_empty_confirm30m_stable_margin04"
    EMPTY_CONFIRMATION_HOURS = .5

class DemandSized340Policy(StableHalfHourVacancyPolicy):
    """Revisit the 340 K demand-sized reserve around the new best policy."""

    name = "best_water_demand340"
    MIN_HEATING_TARGET_K = 340.0
    COIL_CAPACITY_HEADROOM = .95
    TARGET_HEADROOM_K = 2.0

    def reset(self):
        super().reset()
        self.held_water_target = None
        self.observed_water_for_reserve = 340.0
        self.required_water_target = 310.0
        self.initial_water_temperature = None
        self.maximum_water_target = 310.0
        self.maximum_required_water_target = 310.0
        self.capacity_limited_target_requests = 0
        self.demand_sized_water_steps = 0

    def get_action(self, obs):
        self.observed_water_for_reserve = self._temperature(
            np.asarray(obs), self.water_sensor, "supply_water_temperature_sensor", 340.0)
        self.required_water_target = 310.0
        return super().get_action(obs)

    def _required_target(self, group, temperature, free, low, max_flow, choice):
        if not choice["run"]:
            return 310.0
        supply = choice["supply"]
        heating_need = np.maximum(low[group] - temperature[group] - free[group], 0.0) * 1000.0 / self.theta[group, 5]
        # Recover useful airflow before the inherited search clips it to the
        # current water capacity. Inspecting only selected reheat would never
        # request sufficient recovery from a cold, water-limited reservoir.
        desired_flow = heating_need / (self.AIR_CAPACITY * np.maximum(self.MAX_DISCHARGE_K - low[group], .5))
        desired_conductance = np.maximum(
            choice["conductance"], self.AIR_CAPACITY * np.minimum(desired_flow, max_flow[group]))
        wanted_reheat = np.maximum((low[group] - temperature[group] - free[group]) * 1000.0 / self.theta[group, 5] + desired_conductance * (low[group] - supply), 0.0)
        air_limit = desired_conductance * np.maximum(self.MAX_DISCHARGE_K - supply, 0.0)
        feasible_reheat = np.minimum(wanted_reheat, air_limit)
        active = feasible_reheat > 10.0
        if not np.any(active):
            return 310.0
        maximum_water_conductance = self.WATER_VALVE_CAPACITY * math.sqrt(20.0) * self.COIL_CAPACITY_HEADROOM
        targets = supply + feasible_reheat / maximum_water_conductance + self.TARGET_HEADROOM_K
        uncapped_target = float(np.max(targets[active]))
        self.maximum_required_water_target = max(self.maximum_required_water_target, uncapped_target)
        if uncapped_target > 350.0:
            self.capacity_limited_target_requests += 1
        return float(np.clip(uncapped_target, 310.0, 350.0))

    def _choose_ahu(self, group, temperature, free, low, high, max_flow, outside, water):
        choice = super()._choose_ahu(group, temperature, free, low, high, max_flow, outside, water)
        self.required_water_target = max(
            self.required_water_target,
            self._required_target(group, temperature, free, low, max_flow, choice))
        return choice

    def _native_action(self, kind, value, low, high):
        if kind != "supply_water_temperature_setpoint":
            return super()._native_action(kind, value, low, high)
        if self.step_count - self.last_heating_step < self.HOLD_STEPS:
            if self.held_water_target is None:
                self.initial_water_temperature = self.observed_water_for_reserve
                self.held_water_target = float(np.clip(
                    max(self.MIN_HEATING_TARGET_K, self.observed_water_for_reserve), 310.0, 350.0))
            self.held_water_target = min(350.0, max(self.held_water_target, self.required_water_target))
            value = self.held_water_target
            self.demand_sized_water_steps += 1
        else:
            value = 310.0
            self.held_water_target = None
        self.maximum_water_target = max(self.maximum_water_target, float(value))
        # Encode directly with the inherited normalizer law. Calling the
        # ancestor water-hold override here would replace the sized target
        # with 345 K. Non-water actuators still use the exact parent path.
        spec = self.action_norm.get(kind)
        if spec is None:
            spec = next((v for k, v in self.action_norm.items() if str(k).endswith(kind)), None)
        if spec:
            low = float(spec.get("min_native_value", low))
            high = float(spec.get("max_native_value", high))
            out_low = float(spec.get("min_normalized_value", -1.0))
            out_high = float(spec.get("max_normalized_value", 1.0))
        else:
            out_low, out_high = -1.0, 1.0
        return np.clip(out_low + (np.asarray(value) - low) * (out_high - out_low) / max(high - low, 1e-12), out_low, out_high)

    def experiment_diagnostics(self):
        return {**super().experiment_diagnostics(),
                "initial_reserve_water_k": self.initial_water_temperature,
                "maximum_water_target_k": self.maximum_water_target,
                "maximum_required_water_target_k": self.maximum_required_water_target,
                "capacity_limited_target_requests": self.capacity_limited_target_requests,
                "demand_sized_water_steps": self.demand_sized_water_steps,
                "held_water_target_k": self.held_water_target}

class SignedDemandSized320Policy(DemandSized340Policy):
    """Screen signed feasible duty with a 320 K qualified reservoir floor."""

    name = "best_water_signed320"
    MIN_HEATING_TARGET_K = 320.0

class NearMinimumPumpPressurePolicy(SignedDemandSized320Policy):
    """Retain 0.1 percent pressure reserve above the feasible valve minimum."""

    name = "best_water_pump_safety001"
    PRESSURE_SAFETY_FACTOR = 1.001
    VALVE_KV_M3_PER_HOUR = .108
    FLOW_PRODUCT_TOLERANCE = 8.0 * np.finfo(np.float32).eps

    def reset(self):
        super().reset()
        self.hydraulic_adjusted_steps = 0
        self.hydraulic_capacity_tie_steps = 0
        self.hydraulic_quantization_fallback_steps = 0
        self.hydraulic_power_wsteps_before = 0.0
        self.hydraulic_power_wsteps_after = 0.0
        self.hydraulic_maximum_flow_product_error = 0.0
        self.hydraulic_maximum_history_heat_correction_w = 0.0

    def _normalization_limits(self, kind, native_low, native_high):
        spec = self.action_norm.get(kind)
        if spec is None:
            spec = next((value for key, value in self.action_norm.items()
                         if str(key).endswith(kind)), None)
        if spec:
            return (float(spec.get("min_native_value", native_low)),
                    float(spec.get("max_native_value", native_high)),
                    float(spec.get("min_normalized_value", -1.0)),
                    float(spec.get("max_normalized_value", 1.0)))
        return float(native_low), float(native_high), -1.0, 1.0

    @staticmethod
    def _decode_action(value, limits):
        native_low, native_high, out_low, out_high = limits
        return native_low + (np.asarray(value, dtype=float) - out_low) * (
            native_high - native_low) / (out_high - out_low)

    def _encode_action_upward(self, kind, desired, limits, default_low, default_high):
        # Float32 action rounding must not silently remove the planned flow.
        encoded = np.asarray(self._native_action(kind, desired, default_low, default_high), dtype=np.float32)
        decoded = self._decode_action(encoded, limits)
        encoded = np.where(decoded < desired,
                           np.nextafter(encoded, np.float32(np.inf)), encoded).astype(np.float32)
        return np.clip(encoded, -1.0, 1.0).astype(np.float32)

    def get_action(self, obs):
        action = super().get_action(obs)
        if (len(self.pressure_indices) <= self.num_ahus
                or not action["discrete_action"][-1]):
            return action
        pressure_kind = "differential_pressure_setpoint"
        valve_kind = "reheat_valve_percentage_command"
        pressure_limits = self._normalization_limits(pressure_kind, 0.0, 20.0)
        valve_limits = self._normalization_limits(valve_kind, 0.0, 1.0)
        if (not np.isfinite(pressure_limits + valve_limits).all()
                or pressure_limits[1] <= pressure_limits[0]
                or pressure_limits[3] <= pressure_limits[2]
                or valve_limits[1] <= valve_limits[0]
                or valve_limits[3] <= valve_limits[2]):
            self.hydraulic_quantization_fallback_steps += 1
            return action
        continuous = action["continuous_action"]
        pressure_index = self.pressure_indices[-1]
        old_pressure = float(self._decode_action(continuous[pressure_index], pressure_limits))
        old_valves = np.clip(self._decode_action(continuous[self.valve_indices], valve_limits), 0.0, 1.0)
        if not np.isfinite(old_pressure) or old_pressure <= 0.0 or not np.isfinite(old_valves).all():
            return action
        # Respect the actual metadata valve cap, not an assumed full opening.
        valve_cap = min(1.0, valve_limits[1])
        if valve_cap <= 0.0:
            return action
        flow_product = old_valves * math.sqrt(old_pressure)
        hydraulic_scale = 100000.0 * self.VALVE_KV_M3_PER_HOUR / 3600.0
        old_power = hydraulic_scale * old_pressure * float(np.sum(flow_product))
        self.hydraulic_power_wsteps_before += old_power
        minimum_pressure = (float(np.max(flow_product)) / valve_cap) ** 2
        target_pressure = min(old_pressure, max(0.0, pressure_limits[0],
                                               minimum_pressure * self.PRESSURE_SAFETY_FACTOR))
        if target_pressure <= 0.0 or target_pressure >= old_pressure:
            self.hydraulic_capacity_tie_steps += 1
            self.hydraulic_power_wsteps_after += old_power
            return action
        encoded_pressure = self._encode_action_upward(
            pressure_kind, target_pressure, pressure_limits, 0.0, 20.0)
        new_pressure = float(self._decode_action(encoded_pressure, pressure_limits))
        if not (0.0 < new_pressure < old_pressure and new_pressure <= 20.0):
            self.hydraulic_quantization_fallback_steps += 1
            self.hydraulic_power_wsteps_after += old_power
            return action
        desired_valves = flow_product / math.sqrt(new_pressure)
        encoded_valves = self._encode_action_upward(
            valve_kind, desired_valves, valve_limits, 0.0, 1.0)
        new_valves = np.clip(self._decode_action(encoded_valves, valve_limits), 0.0, 1.0)
        new_product = new_valves * math.sqrt(new_pressure)
        difference = new_product - flow_product
        tolerance = self.FLOW_PRODUCT_TOLERANCE * np.maximum(1.0, flow_product)
        if (np.any(desired_valves > valve_cap)
                or np.any(new_valves > valve_cap + self.FLOW_PRODUCT_TOLERANCE)
                or np.any(np.abs(difference) > tolerance)
                or np.any(difference < -tolerance)):
            self.hydraulic_quantization_fallback_steps += 1
            self.hydraulic_power_wsteps_after += old_power
            return action
        new_power = hydraulic_scale * new_pressure * float(np.sum(new_product))
        if new_power >= old_power:
            self.hydraulic_quantization_fallback_steps += 1
            self.hydraulic_power_wsteps_after += old_power
            return action
        continuous[pressure_index] = encoded_pressure
        continuous[self.valve_indices] = encoded_valves
        # History must describe the new normalized/quantized commands. This
        # also accounts for metadata caps that clipped the parent's planned
        # pressure or valves. Every non-reheat history field stays unchanged.
        history = self.history[-1]
        delivered_reheat = np.minimum(
            self.WATER_VALVE_CAPACITY * new_product
            * np.maximum(self.observed_water_for_reserve - history["supply"], 0.0),
            history["conductance"] * np.maximum(self.MAX_DISCHARGE_K - history["supply"], 0.0))
        self.hydraulic_maximum_history_heat_correction_w = max(
            self.hydraulic_maximum_history_heat_correction_w,
            float(np.max(np.abs(delivered_reheat - history["reheat"]))))
        history["reheat"] = delivered_reheat
        self.last_action = action
        self.hydraulic_adjusted_steps += 1
        self.hydraulic_power_wsteps_after += new_power
        self.hydraulic_maximum_flow_product_error = max(
            self.hydraulic_maximum_flow_product_error, float(np.max(np.abs(difference))))
        return action

    def experiment_diagnostics(self):
        return {**super().experiment_diagnostics(),
                "hydraulic_adjusted_steps": self.hydraulic_adjusted_steps,
                "hydraulic_capacity_tie_steps": self.hydraulic_capacity_tie_steps,
                "hydraulic_quantization_fallback_steps": self.hydraulic_quantization_fallback_steps,
                "hydraulic_power_wsteps_before": self.hydraulic_power_wsteps_before,
                "hydraulic_power_wsteps_after": self.hydraulic_power_wsteps_after,
                "hydraulic_maximum_flow_product_error": self.hydraulic_maximum_flow_product_error,
                "hydraulic_maximum_history_heat_correction_w": self.hydraulic_maximum_history_heat_correction_w}

class DemandTrackingWaterPolicy(SignedDemandSized320Policy):
    """Stop maintaining requested capacity no longer supported by demand.

    During the inherited 72-hour qualified hold, command the greater of the
    present feasible duty, 320 K, and observed stored temperature. No earlier
    command is treated as measured stored heat. The inherited expiry still
    releases the reserve to 310 K; initial idle observations cannot qualify it.
    """

    name = "water_current_demand_nov10"
    DEMAND_MEMORY_STEPS = 0
    COAST_AFTER_EXPIRY = False

    def reset(self):
        super().reset()
        self.water_demand_window = deque()
        self.water_reserve_ever_qualified = False
        self.water_target_observed = 340.0
        self.last_valid_water_observation = None
        self.water_observation_valid = False
        self.invalid_water_observations = 0
        self.water_requested_peak_release_steps = 0
        self.water_requested_peak_release_k = 0.0
        self.water_coast_steps = 0
        self.water_peak_memory_steps = 0
        self.water_observed_above_native_cap_steps = 0

    def _tank_observation(self, obs):
        """Decode current public water using the frozen measurement convention.

        Bad measurements use the last finite observation, then the preceding
        command, then the best controller's 340 K fallback. This fallback is
        solely a reserve bound: idle or reset state does not become qualified.
        """
        fallback = self.last_valid_water_observation
        if fallback is None:
            fallback = self.held_water_target if self.held_water_target is not None else 340.0
        index = self.water_sensor
        if index is None or index >= obs.size:
            return float(fallback), False
        raw = float(obs[index])
        if np.isfinite(raw) and 150.0 < raw < 450.0:
            return raw, True
        norm = self.obs_norm.get("supply_water_temperature_sensor", {})
        mean = float(norm.get("sample_mean", 340.0))
        std = float(norm.get("sample_std", math.sqrt(norm.get("sample_variance", 1.0))))
        decoded = raw * std + mean
        if np.isfinite(decoded) and 150.0 < decoded < 450.0:
            return float(decoded), True
        return float(fallback), False

    def get_action(self, obs):
        flat_obs = np.asarray(obs, dtype=float).reshape(-1)
        self.water_target_observed, self.water_observation_valid = self._tank_observation(flat_obs)
        if self.water_observation_valid:
            self.last_valid_water_observation = self.water_target_observed
        else:
            self.invalid_water_observations += 1
        return super().get_action(flat_obs)

    def _remembered_demand(self, current):
        if self.DEMAND_MEMORY_STEPS <= 0:
            return current
        # This deque stores only requirements calculated from observations
        # available at that decision, never the future command or actual tank.
        while self.water_demand_window and (
                self.step_count - self.water_demand_window[0][0] >= self.DEMAND_MEMORY_STEPS):
            self.water_demand_window.popleft()
        if self.water_demand_window and self.water_demand_window[-1][0] == self.step_count:
            self.water_demand_window[-1] = (self.step_count, current)
        else:
            self.water_demand_window.append((self.step_count, current))
        remembered = max(demand for _, demand in self.water_demand_window)
        if remembered > current:
            self.water_peak_memory_steps += 1
        return remembered

    def _water_target(self):
        qualified = self.step_count - self.last_heating_step < self.HOLD_STEPS
        observed = float(np.clip(self.water_target_observed, 150.0, 450.0))
        if qualified:
            if self.held_water_target is None:
                self.initial_water_temperature = observed
            self.water_reserve_ever_qualified = True
            current = float(self.required_water_target)
            if not np.isfinite(current):
                current = 350.0  # Do not turn a malformed capacity request into low reserve.
            current = float(np.clip(current, 310.0, 350.0))
            remembered = self._remembered_demand(current)
            value = min(350.0, max(self.MIN_HEATING_TARGET_K, observed, remembered))
            previous = self.held_water_target
            if previous is not None and value < previous:
                self.water_requested_peak_release_steps += 1
                self.water_requested_peak_release_k += previous - value
            self.held_water_target = value
            self.demand_sized_water_steps += 1
        else:
            self.water_demand_window.clear()
            self.held_water_target = None
            if self.COAST_AFTER_EXPIRY and self.water_reserve_ever_qualified:
                # Preserve attained storage without heating to an old request.
                # This can add dissipation versus baseline expiry, and is an
                # independent experiment; the 96-hour hold did not transfer.
                value = float(np.clip(observed, 310.0, 350.0))
                self.water_coast_steps += 1
            else:
                value = 310.0
        self.maximum_water_target = max(self.maximum_water_target, value)
        return value

    def _native_action(self, kind, value, low, high):
        if kind != "supply_water_temperature_setpoint":
            return super()._native_action(kind, value, low, high)
        value = self._water_target()
        # Direct native encoding is necessary: ancestor water overrides would
        # replace the selected demand target with their fixed 345 K hold.
        spec = self.action_norm.get(kind)
        if spec is None:
            spec = next((v for k, v in self.action_norm.items() if str(k).endswith(kind)), None)
        if spec:
            low = float(spec.get("min_native_value", low))
            high = float(spec.get("max_native_value", high))
            out_low = float(spec.get("min_normalized_value", -1.0))
            out_high = float(spec.get("max_normalized_value", 1.0))
        else:
            out_low, out_high = -1.0, 1.0
        if self.water_target_observed > high:
            self.water_observed_above_native_cap_steps += 1
        return np.clip(out_low + (np.asarray(value) - low) * (out_high - out_low)
                       / max(high - low, 1e-12), out_low, out_high)

    def experiment_diagnostics(self):
        return {**super().experiment_diagnostics(),
                "water_reserve_ever_qualified": self.water_reserve_ever_qualified,
                "water_requested_peak_release_steps": self.water_requested_peak_release_steps,
                "water_requested_peak_release_k": self.water_requested_peak_release_k,
                "water_peak_memory_steps": self.water_peak_memory_steps,
                "water_coast_steps": self.water_coast_steps,
                "invalid_water_observations": self.invalid_water_observations,
                "water_observed_above_native_cap_steps": self.water_observed_above_native_cap_steps,
                "last_valid_water_observation_k": self.last_valid_water_observation}

class ObservedWaterCoastPolicy(DemandTrackingWaterPolicy):
    """Preserve attained reservoir heat after reserve expiry without old-peak heat."""

    name = "water_observed_coast_nov10"
    COAST_AFTER_EXPIRY = True

class StorageARXPolicy(ObservedWaterCoastPolicy):
    name="coast_storage_arx_nov12"
    MAX_RESPONSE_RATIO=4.
    INITIAL_STORAGE_RETENTION=.7

    def reset(self):
        super().reset()
        gain=self.theta[:,5]*1.1
        self.arx_theta=np.column_stack((self.theta[:,0],self.theta[:,1],
                                      np.full(self.n_zones,self.INITIAL_STORAGE_RETENTION),
                                      -self.INITIAL_STORAGE_RETENTION*gain,
                                      self.theta[:,4],gain))
        self.arx_prior=gain.copy()
        diagonal=np.asarray([.12,.2,.15,.3,.16,.8])
        self.arx_covariance=np.broadcast_to(np.diag(diagonal),(self.n_zones,6,6)).copy()
        self.arx_residual=np.zeros(self.n_zones)
        self.arx_previous_delta=np.zeros(self.n_zones)
        self.arx_observed=None
        self.arx_current_delta=np.zeros(self.n_zones)
        self.arx_observer_mode=True
        self.arx_updates=0
        self.arx_error_sum=0.
        self.arx_error_count=0
        self.theta[:,5]=gain/(1.+self.arx_theta[:,1]/20.)

    def _band(self,obs,temperature):
        self.arx_observer_mode=False
        return super()._band(obs,temperature)

    def _arx_features(self,temperature,outside,mode,delta,previous_power):
        return np.column_stack((np.ones(self.n_zones),
                                np.clip((outside-temperature)/20.,-5.,5.),
                                np.clip(delta,-8.,8.),np.clip(previous_power/1000.,-100.,100.),
                                np.full(self.n_zones,float(mode))))

    def _free_drift(self,temperature,outside,mode):
        if self.arx_observed is None:
            return super()._free_drift(temperature,outside,mode)
        if self.arx_observer_mode:
            delta=self.arx_current_delta
            previous=list(self.history)[-2] if len(self.history)>=2 else None
        else:
            delta=temperature-self.arx_observed
            previous=self.history[-1] if self.history else None
        power=(previous["conductance"]*(previous["supply"]-temperature)+previous["reheat"]
               if previous is not None else np.zeros(self.n_zones))
        features=self._arx_features(temperature,outside,mode,delta,power)
        multiplier=1.+self.arx_theta[:,1]/20.
        return np.clip((np.einsum("ij,ij->i",self.arx_theta[:,:5],features)+self.arx_residual)/multiplier,-5.,5.)

    def _learn(self,observed):
        self.arx_observer_mode=True
        old=self.previous_observed
        current_delta=np.zeros(self.n_zones) if old is None else observed-old
        if old is not None and len(self.history)>1:
            applied=self.history[-2]
            previous=self.history[-3] if len(self.history)>2 else None
            previous_power=(previous["conductance"]*(previous["supply"]-old)+previous["reheat"]
                            if previous is not None else np.zeros(self.n_zones))
            before=self._arx_features(old,applied["outside"],applied["mode"],self.arx_previous_delta,previous_power)
            raw=self.arx_theta
            multiplier=1.+raw[:,1]/20.
            gain=raw[:,5]/1000.
            drift=np.einsum("ij,ij->i",raw[:,:5],before)+self.arx_residual
            prediction=(multiplier*old+drift+gain*(applied["conductance"]*applied["supply"]+applied["reheat"]))/(
                multiplier+gain*applied["conductance"])
            error=np.where(np.isfinite(observed-prediction),np.minimum(np.abs(observed-prediction),2.),2.)
            self.model_error=.94*self.model_error+.06*error
            endpoint=self._arx_features(observed,applied["outside"],applied["mode"],self.arx_previous_delta,previous_power)
            power=(applied["conductance"]*(applied["supply"]-observed)+applied["reheat"])/1000.
            x=np.column_stack((endpoint,power))
            valid=np.isfinite(x).all(axis=1)&np.isfinite(current_delta)&(np.abs(current_delta)<=8.)
            if np.any(valid):
                covariance=self.arx_covariance[valid]
                magnitude=np.max(np.sum(np.abs(covariance),axis=2),axis=1)
                covariance*=np.minimum(1.,10./np.maximum(magnitude,1e-12))[:,None,None]
                px=np.einsum("nij,nj->ni",covariance,x[valid])
                forgetting=float(self.parameters["forgetting_factor"])
                den=forgetting+np.einsum("ni,ni->n",x[valid],px)
                kg=px/np.maximum(den[:,None],1e-9)
                innovation=np.clip(current_delta[valid]-self.arx_residual[valid]-np.einsum(
                    "ij,ij->i",raw[valid],x[valid]),-2.,2.)
                proposed=raw[valid]+kg*innovation[:,None]
                lower=np.zeros_like(proposed)
                upper=np.ones_like(proposed)*np.asarray([2.,10.,.99,0.,2.,1.])
                lower[:,0]=-2.
                lower[:,3]=-4.
                lower[:,5]=.25*self.arx_prior[valid]
                upper[:,5]=self.MAX_RESPONSE_RATIO*self.arx_prior[valid]
                proposed=np.clip(proposed,lower,upper)
                proposed[:,3]=np.maximum(proposed[:,3],-.99*proposed[:,5])
                self.arx_theta[valid]=proposed
                updated=(covariance-np.einsum("ni,nj->nij",kg,px))/forgetting
                self.arx_covariance[valid]=.5*(updated+updated.transpose(0,2,1))
                self.arx_updates+=1
                self.arx_error_sum+=float(np.sum(np.abs(observed[valid]-prediction[valid])))
                self.arx_error_count+=int(np.count_nonzero(valid))
            remaining=np.clip(current_delta-np.einsum("ij,ij->i",self.arx_theta,x),-5.,5.)
            remaining=np.where(np.isfinite(remaining),remaining,self.arx_residual)
            rate=float(self.parameters["disturbance_rate"])
            self.arx_residual=np.clip((1.-rate)*self.arx_residual+rate*remaining,-5.,5.)
        self.arx_observed=observed.copy()
        self.arx_previous_delta=current_delta.copy()
        self.arx_current_delta=current_delta.copy()
        multiplier=1.+self.arx_theta[:,1]/20.
        self.theta[:,5]=self.arx_theta[:,5]/multiplier
        self.residual=self.arx_residual/multiplier

    def experiment_diagnostics(self):
        return {**super().experiment_diagnostics(),"arx_updates":self.arx_updates,
                "arx_prefit_mean_absolute_error_k":self.arx_error_sum/max(1,self.arx_error_count),
                "arx_increment_coefficient_mean":float(np.mean(self.arx_theta[:,2])),
                "arx_previous_power_coefficient_mean":float(np.mean(self.arx_theta[:,3]))}

class GatedStorageARXPolicy(StorageARXPolicy):
    name="coast_gated_storage_arx_nov12"
    STORAGE_ERROR_RATIO=.9
    MIN_GATE_STEPS=24
    MAX_STORAGE_ZONE_FRACTION=1.
    RECOVERY_ONLY=False

    def reset(self):
        super().reset()
        self.reference_observer=ObservedWaterCoastPolicy(self.metadata,self.action_spec,self.parameters)
        self.storage_gate=np.zeros(self.n_zones,dtype=bool)
        self.reference_prefit_error=np.full(self.n_zones,.08)
        self.storage_prefit_error=np.full(self.n_zones,.08)
        self.gated_model_zone_steps=0
        self.gated_recovery_steps=0
        self.storage_reference_ratio_sum=0.
        self.storage_context_allowed=True

    def get_action(self,obs):
        flat=np.asarray(obs,dtype=float).reshape(-1)
        mode=bool(flat[self.comfort_now_index]>.5) if self.comfort_now_index is not None else True
        soon=bool(flat[self.comfort_soon_index]>.5) if self.comfort_soon_index is not None else mode
        hour=self._hour(flat)
        self.storage_context_allowed=(not self.RECOVERY_ONLY or soon or
                                      (mode and (hour-self.start_hour)%24.<2.))
        return super().get_action(flat)

    def _learn(self,observed):
        reference=self.reference_observer
        reference.previous_observed=None if self.previous_observed is None else self.previous_observed.copy()
        reference.history=deque((copy.deepcopy(entry) for entry in self.history),maxlen=3)
        reference.slow_temperature=self.slow_temperature.copy()
        reference._learn(observed)
        self.reference_prefit_error=reference.model_error.copy()
        # Restore the ARX forecast reserve before advancing its fit; preceding
        # action selection may have switched self.model_error to reference.
        self.model_error=self.storage_prefit_error.copy()
        super()._learn(observed)
        self.storage_prefit_error=self.model_error.copy()
        raw_response=self.arx_theta[:,5]/(1.+self.arx_theta[:,1]/20.)
        reference_response=reference.theta[:,5]
        useful=(self.step_count>=self.MIN_GATE_STEPS)&self.storage_context_allowed
        valid=(np.isfinite(raw_response)&(raw_response>.01)&np.isfinite(self.storage_prefit_error)
               &np.isfinite(self.reference_prefit_error)&(self.reference_prefit_error>.002))
        self.storage_gate=valid&useful&(self.storage_prefit_error<self.STORAGE_ERROR_RATIO*self.reference_prefit_error)
        ratio=raw_response/np.maximum(reference_response,.01)
        # Avoid selecting a gain direction not supported by safe bounded plant
        # behavior, even when a short quiet-fit residual happens to be tiny.
        self.storage_gate&=(ratio>=.35)&(ratio<=2.5)
        self.theta=reference.theta.copy()
        self.theta[self.storage_gate,5]=raw_response[self.storage_gate]
        self.residual=reference.residual.copy()
        self.model_error=np.where(self.storage_gate,self.storage_prefit_error,self.reference_prefit_error)
        count=int(np.count_nonzero(self.storage_gate))
        self.gated_model_zone_steps+=count
        if count:
            self.storage_reference_ratio_sum+=float(np.sum(self.storage_prefit_error[self.storage_gate]/
                                                           self.reference_prefit_error[self.storage_gate]))

    def _free_drift(self,temperature,outside,mode):
        reference=self.reference_observer
        reference.slow_temperature=self.slow_temperature
        baseline=reference._free_drift(temperature,outside,mode)
        if not np.any(self.storage_gate):
            return baseline
        storage=super()._free_drift(temperature,outside,mode)
        return np.where(self.storage_gate,storage,baseline)

    def experiment_diagnostics(self):
        return {**super().experiment_diagnostics(),"gated_storage_zone_steps":self.gated_model_zone_steps,
                "gated_storage_mean_error_ratio":self.storage_reference_ratio_sum/max(1,self.gated_model_zone_steps),
                "reference_prefit_error_k":float(np.mean(self.reference_prefit_error)),
                "storage_prefit_error_k":float(np.mean(self.storage_prefit_error)),"storage_recovery_only":self.RECOVERY_ONLY}

class StorageObserverOnlyPolicy(GatedStorageARXPolicy):
    """Use gated storage only to reconstruct the currently delayed temperature."""

    name = "split_storage_observer_nov12"
    OBSERVER_BLEND = 1.0
    MIN_GATE_STEPS = 24

    def _learn(self, observed):
        super()._learn(observed)
        self.split_reference_gain = self.reference_observer.theta[:, 5].copy()

    def _free_drift(self, temperature, outside, mode):
        baseline = self.reference_observer._free_drift(temperature, outside, mode)
        if not self.arx_observer_mode or not np.any(self.storage_gate):
            return baseline
        storage = super()._free_drift(temperature, outside, mode)
        return baseline + self.OBSERVER_BLEND * (storage - baseline)

    def _band(self, obs, temperature):
        # The parent just completed delay compensation. Restore the reference
        # inverse gain/reserve before computing targets and any action plan.
        self.theta = self.reference_observer.theta.copy()
        self.residual = self.reference_observer.residual.copy()
        self.model_error = self.reference_prefit_error.copy()
        return super()._band(obs, temperature)

class IntermediateRecirculationPolicy(SignedDemandSized320Policy):
    """Add two intermediate native ratios when they provide new mixed air."""

    name = "best_water_recirc_intermediate"
    RECIRCULATION_GRID = (.4, .55)
    ORIGINAL_POINT_COUNT = 84

    def reset(self):
        super().reset()
        self.recirculation_candidate_steps = 0
        self.recirculation_extra_choices = 0
        self.recirculation_objective_gain_sum = 0.0
        self.recirculation_lowest_selected_supply_k = 305.0

    def _operating_points(self, group, temperature, high, outside):
        inside = float(np.average(temperature[group], weights=self.zone_area[group]))
        # Exact frozen FineSupplySearchPolicy prefix, including dynamic SAT.
        targets = np.arange(285.0, 305.01, .5)
        targets = np.concatenate((targets, [float(np.clip(np.min(high[group]) - 1.5, 285.0, 305.0))]))
        setpoint = np.repeat(targets, 2)
        recirc = np.full_like(setpoint, self.MAX_RECIRCULATION)
        if outside < inside - .5:
            recirc[1::2] = np.clip((targets - outside) / max(inside - outside, 1e-6), 0.0, self.MAX_RECIRCULATION)
        if not (np.isfinite(inside) and np.isfinite(outside) and outside < inside - .5):
            return setpoint, recirc
        original_mixed = recirc * inside + (1.0 - recirc) * outside
        original_supply = np.minimum(original_mixed, setpoint)
        ratios = np.asarray(self.RECIRCULATION_GRID, dtype=float)
        mixed = ratios * inside + (1.0 - ratios) * outside
        # No new colder envelope, and no redundant maximum-recirculation or
        # coil-active intermediate points. The native 305 K SAT passes each
        # admitted mixture through unchanged under the physical cooling law.
        feasible = ((ratios >= 0.0) & (ratios < self.MAX_RECIRCULATION)
                    & (mixed >= float(np.min(original_supply)))
                    & (mixed <= float(np.max(original_supply))) & (mixed <= 305.0))
        ratios = ratios[feasible]
        if not ratios.size:
            return setpoint, recirc
        return (np.concatenate((setpoint, np.full(ratios.size, 305.0))),
                np.concatenate((recirc, ratios)))

    def _candidate_objectives(self, group, temperature, free, low, high, max_flow, outside, water):
        # All physical equations and objective weights are the frozen fine
        # search. Only its operating point arrays acquire appended mixtures.
        t, d, lo, hi = temperature[group], free[group], low[group], high[group]
        response = self.theta[group, 5]
        capacity = max_flow[group]
        inside = float(np.average(t, weights=self.zone_area[group]))
        setpoint, recirc = self._operating_points(group, temperature, high, outside)
        mixed = recirc * inside + (1.0 - recirc) * outside
        supply = np.minimum(mixed, setpoint)
        sa = supply[:, None]
        cool_need = np.maximum(t + d - hi, 0.0) * 1000.0 / response
        heat_need = np.maximum(lo - t - d, 0.0) * 1000.0 / response
        cooling_flow = cool_need[None, :] / (self.AIR_CAPACITY * np.maximum(hi[None, :] - sa, .05))
        heating_flow = heat_need[None, :] / (self.AIR_CAPACITY * np.maximum(self.MAX_DISCHARGE_K - lo[None, :], .5))
        maximum_water_heat = self.WATER_VALVE_CAPACITY * math.sqrt(20.0) * np.maximum(water - sa, 0.0)
        maximum_heating_flow = maximum_water_heat / (self.AIR_CAPACITY * np.maximum(self.MAX_DISCHARGE_K - sa, .1))
        heating_flow = np.minimum(heating_flow, maximum_heating_flow)
        needed_flow = np.maximum(cooling_flow, heating_flow)
        fraction = np.max(needed_flow / capacity[None, :], axis=1)
        pressure = np.clip(self.DESIGN_PRESSURE * fraction ** 2, self.MIN_PRESSURE, self.DESIGN_PRESSURE)
        pressure_ratio = np.sqrt(pressure / self.DESIGN_PRESSURE)
        dampers = np.clip(needed_flow / (capacity[None, :] * pressure_ratio[:, None]), self.MIN_DAMPER, 1.0)
        flow = dampers * capacity[None, :] * pressure_ratio[:, None]
        conductance = flow * self.AIR_CAPACITY
        water_gap = np.maximum(water - sa, 0.0)
        wanted_reheat = np.maximum((lo[None, :] - t[None, :] - d[None, :]) * 1000.0 / response[None, :]
                                    + conductance * (lo[None, :] - sa), 0.0)
        reheat_limit = np.minimum(conductance * np.maximum(self.MAX_DISCHARGE_K - sa, 0.0),
                                  self.WATER_VALVE_CAPACITY * math.sqrt(20.0) * water_gap)
        reheat = np.minimum(wanted_reheat, reheat_limit)
        predicted = (t[None, :] + d[None, :] + response[None, :] * (conductance * sa + reheat) / 1000.0) / (
            1.0 + response[None, :] * conductance / 1000.0)
        violation = np.maximum(lo[None, :] - predicted, 0.0) + np.maximum(predicted - hi[None, :], 0.0)
        thermal_penalty = np.sum(violation + 2.0 * violation ** 2, axis=1)
        total_flow = flow.sum(axis=1)
        fan = total_flow * pressure * (2.0 - recirc) / .9
        chiller = total_flow * self.AIR_CAPACITY * np.maximum(mixed - supply, 0.0) / 3.5
        gas = reheat.sum(axis=1)
        objective = fan + chiller + self.parameters["gas_cost_ratio"] * gas
        objective += self.parameters["comfort_cost"] * thermal_penalty
        off_prediction = t + d
        off_violation = np.maximum(lo - off_prediction, 0.0) + np.maximum(off_prediction - hi, 0.0)
        off_objective = self.parameters["comfort_cost"] * np.sum(off_violation + 2.0 * off_violation ** 2)
        return {"objective": objective, "off_objective": float(off_objective),
                "setpoint": setpoint, "recirc": recirc, "mixed": mixed, "supply": supply,
                "pressure": pressure, "dampers": dampers, "conductance": conductance,
                "reheat": reheat, "predicted": predicted}

    def _choose_ahu(self, group, temperature, free, low, high, max_flow, outside, water):
        data = self._candidate_objectives(group, temperature, free, low, high, max_flow, outside, water)
        best = int(np.argmin(data["objective"]))
        if data["objective"].size > self.ORIGINAL_POINT_COUNT:
            self.recirculation_candidate_steps += 1
        if data["off_objective"] <= data["objective"][best]:
            choice = {"run": 0, "pressure": 0.0, "setpoint": 305.0, "recirc": .7,
                      "damper": np.full(group.size, self.MIN_DAMPER), "conductance": np.zeros(group.size),
                      "supply": np.full(group.size, data["mixed"][best]), "reheat": np.zeros(group.size)}
        else:
            if best >= self.ORIGINAL_POINT_COUNT:
                self.recirculation_extra_choices += 1
                self.recirculation_objective_gain_sum += float(
                    np.min(data["objective"][:self.ORIGINAL_POINT_COUNT]) - data["objective"][best])
                self.recirculation_lowest_selected_supply_k = min(
                    self.recirculation_lowest_selected_supply_k, float(data["supply"][best]))
            choice = {"run": 1, "pressure": float(data["pressure"][best]), "setpoint": float(data["setpoint"][best]),
                      "recirc": float(data["recirc"][best]), "damper": data["dampers"][best],
                      "conductance": data["conductance"][best], "supply": np.full(group.size, data["supply"][best]),
                      "reheat": data["reheat"][best]}
        # The parent search wrapper sizes the current-best signed water
        # reserve from its selected air-side choice. Preserve that wrapper's
        # law here because this search deliberately replaces the parent call.
        self.required_water_target = max(
            self.required_water_target,
            self._required_target(group, temperature, free, low, max_flow, choice))
        return choice

    def experiment_diagnostics(self):
        return {**super().experiment_diagnostics(),
                "recirculation_candidate_steps": self.recirculation_candidate_steps,
                "recirculation_extra_choices": self.recirculation_extra_choices,
                "recirculation_objective_gain_sum": self.recirculation_objective_gain_sum,
                "recirculation_lowest_selected_supply_k": self.recirculation_lowest_selected_supply_k}

class FullRecirculationGridPolicy(IntermediateRecirculationPolicy):
    """Screen every tenth of native recirculation below the inherited cap."""

    name = "best_water_recirc_grid10"
    RECIRCULATION_GRID = (0.0, .1, .2, .3, .4, .5, .6)

class ObserverGridPumpPolicy(NearMinimumPumpPressurePolicy, StorageObserverOnlyPolicy, FullRecirculationGridPolicy):
    name = "observer_grid_pump_nov12"

def storage_temperature_step(temperature, previous_delta, previous_conductance,
                             previous_supply, previous_reheat, conductance,
                             supply, reheat, outside, mode, theta, residual):
    temperature = np.asarray(temperature, dtype=float)
    theta = np.asarray(theta, dtype=float)
    previous_power = previous_conductance * (previous_supply-temperature) + previous_reheat
    feature = np.stack((np.ones_like(temperature),
                        np.clip((outside-temperature)/20., -5., 5.),
                        np.clip(previous_delta, -8., 8.),
                        np.clip(previous_power/1000., -100., 100.),
                        np.broadcast_to(np.asarray(mode, dtype=float), temperature.shape)), axis=-1)
    multiplier = 1.+theta[..., 1]/20.
    drift = np.clip((np.sum(theta[..., :5]*feature, axis=-1)+residual)/multiplier, -5., 5.)
    gain = theta[..., 5]/(1000.*multiplier)
    return (temperature+drift+gain*(conductance*supply+reheat))/(1.+gain*conductance)

class ShortStorageHeatingPlanPolicy(ObserverGridPumpPolicy):
    name='storage_heat_plan3_nov13'
    PLAN_STEPS=3
    CHARGE_MARGINS_K=(.2,.45)
    MIN_PLAN_SAVING=.02
    MAX_PLAN_ERROR_K=.06
    MIN_QUALIFIED_FRACTION=.5
    ENABLE_PLAN=True

    def reset(self):
        super().reset()
        self.plan_mode=False
        self.plan_soon=False
        self.plan_hour=self.start_hour
        self.plan_parent_choices=0
        self.plan_eligible_groups=0
        self.plan_charge_choices=0
        self.plan_predicted_saving_wsteps=0.
        self.plan_maximum_charge_k=0.

    def _band(self,obs,temperature):
        low,high,mode=super()._band(obs,temperature)
        self.plan_mode=bool(mode)
        self.plan_soon=bool(obs[self.comfort_soon_index]>.5) if self.comfort_soon_index is not None else bool(mode)
        self.plan_hour=self._hour(obs)
        return low,high,mode

    def _plain_choice(self,group,temperature,drift,low,high,max_flow,outside,water):
        # The pure candidate producer has no reserve, history, or selected
        # action counters. Do not call the mutating parent selection wrapper.
        data=self._candidate_objectives(group,temperature,drift,low,high,max_flow,outside,water)
        best=int(np.argmin(data['objective']))
        if data['off_objective']<=data['objective'][best]:
            return {'run':0,'pressure':0.,'setpoint':305.,'recirc':.7,
                    'damper':np.full(group.size,self.MIN_DAMPER),'conductance':np.zeros(group.size),
                    'supply':np.full(group.size,data['mixed'][best]),'reheat':np.zeros(group.size)}
        return {'run':1,'pressure':float(data['pressure'][best]),'setpoint':float(data['setpoint'][best]),
                'recirc':float(data['recirc'][best]),'damper':data['dampers'][best].copy(),
                'conductance':data['conductance'][best].copy(),'supply':np.full(group.size,data['supply'][best]),
                'reheat':data['reheat'][best].copy()}

    def _plan_step(self,group,state,delta,previous,choice,outside,qualified,slow):
        graph_temperature=state.copy()
        reference=self.reference_observer
        feature=reference._features(graph_temperature,np.full(self.n_zones,outside),self.plan_mode,slow)
        drift=np.clip(np.sum(reference.theta[:,:5]*feature,axis=1)+reference.residual,-5.,5.)
        gain=reference.theta[group,5]/1000.
        fallback=(state[group]+drift[group]+gain*(choice['conductance']*choice['supply']+choice['reheat']))/(
            1.+gain*choice['conductance'])
        storage=storage_temperature_step(state[group],delta,previous['conductance'],previous['supply'],previous['reheat'],
                                        choice['conductance'],choice['supply'],choice['reheat'],outside,self.plan_mode,
                                        self.arx_theta[group],self.arx_residual[group])
        return np.where(qualified,storage,fallback)

    def _rollout_cost(self,group,first,temperature,low,high,max_flow,outside,water,qualified):
        state=temperature.copy()
        slow=self.slow_temperature.copy()
        delta=temperature[group]-self.arx_observed[group]
        last=self.history[-1]
        previous={key:last[key][group].copy() for key in ('conductance','supply','reheat')}
        choice=first
        total=0.
        for step in range(self.PLAN_STEPS):
            if step:
                reference=self.reference_observer
                feature=reference._features(state,np.full(self.n_zones,outside),self.plan_mode,slow)
                drift=np.clip(np.sum(reference.theta[:,:5]*feature,axis=1)+reference.residual,-5.,5.)
                choice=self._plain_choice(group,state,drift,low,high,max_flow,outside,water)
            predicted=self._plan_step(group,state,delta,previous,choice,outside,qualified,slow)
            # Use the original band reserve plus a bounded depth reserve.
            reserve=np.minimum(np.sqrt(step)*self.storage_prefit_error[group],.08)
            violation=np.maximum(low[group]+reserve-predicted,0.)+np.maximum(predicted-high[group]+reserve,0.)
            stage=self.parameters['comfort_cost']*float(np.sum(violation+2.*violation**2))
            if choice['run']:
                flow=float(choice['conductance'].sum())/self.AIR_CAPACITY
                inside=float(np.average(state[group],weights=self.zone_area[group]))
                mixed=choice['recirc']*inside+(1.-choice['recirc'])*outside
                stage+=flow*choice['pressure']*(2.-choice['recirc'])/.9
                stage+=flow*self.AIR_CAPACITY*max(0.,mixed-float(choice['supply'][0]))/3.5
                stage+=self.parameters['gas_cost_ratio']*float(choice['reheat'].sum())
            total+=stage
            slow[group]+=(state[group]-slow[group])/48.
            delta=predicted-state[group]
            state[group]=predicted
            previous={key:np.asarray(choice[key]).copy() for key in ('conductance','supply','reheat')}
        return float(total)

    def _choose_ahu(self,group,temperature,free,low,high,max_flow,outside,water):
        normal=super()._choose_ahu(group,temperature,free,low,high,max_flow,outside,water)
        self.plan_parent_choices+=1
        if (not self.ENABLE_PLAN or not normal['run'] or not self.plan_mode or not self.history
                or self.arx_observed is None or self.plan_hour<self.start_hour+2.
                or self.plan_hour>=self.end_hour-self.PLAN_STEPS/12.
                or not np.any(self.history[-1]['reheat'][group]>50.)
                or np.any(temperature[group]<low[group]-.15)
                or np.any(temperature[group]>high[group]+.15)):
            return normal
        qualified=(self.storage_gate[group] & np.isfinite(self.storage_prefit_error[group])
                   & (self.storage_prefit_error[group]<self.MAX_PLAN_ERROR_K)
                   & np.isfinite(self.arx_theta[group]).all(axis=1))
        if float(np.mean(qualified))<self.MIN_QUALIFIED_FRACTION:
            return normal
        heating=(free[group]<-.005)&(self.history[-1]['reheat'][group]>50.)&qualified
        if not heating.any():
            return normal
        self.plan_eligible_groups+=1
        baseline=self._rollout_cost(group,normal,temperature,low,high,max_flow,outside,water,qualified)
        best,best_cost,best_margin=normal,baseline,0.
        for margin in self.CHARGE_MARGINS_K:
            charge_low=low.copy()
            charge_low[group]=low[group]+np.where(heating,np.minimum(margin,.2*np.maximum(high[group]-low[group],0.)),0.)
            charge=self._plain_choice(group,temperature,free,charge_low,high,max_flow,outside,water)
            if not charge['run']:
                continue
            cost=self._rollout_cost(group,charge,temperature,low,high,max_flow,outside,water,qualified)
            if cost<best_cost:
                best,best_cost,best_margin=charge,cost,margin
        if best is normal or not np.isfinite(best_cost) or best_cost>=baseline*(1.-self.MIN_PLAN_SAVING):
            return normal
        self.plan_charge_choices+=1
        self.plan_predicted_saving_wsteps+=baseline-best_cost
        self.plan_maximum_charge_k=max(self.plan_maximum_charge_k,best_margin)
        self.required_water_target=max(self.required_water_target,self._required_target(group,temperature,free,low,max_flow,best))
        return best

    def experiment_diagnostics(self):
        return {**super().experiment_diagnostics(),'planning_horizon_steps':self.PLAN_STEPS,
                'plan_eligible_groups':self.plan_eligible_groups,'plan_charge_choices':self.plan_charge_choices,
                'plan_predicted_saving_wsteps':self.plan_predicted_saving_wsteps,
                'plan_maximum_charge_k':self.plan_maximum_charge_k}

class ForecastHeatGuardPolicy(ShortStorageHeatingPlanPolicy):
    name='forecast_heat_guard25_nov13'
    ENABLE_PLAN=False
    ENABLE_HEAT_GUARD=True
    GUARD_LIMIT_K=.25
    GUARD_GAIN=.5
    MIN_GUARD_DISAGREEMENT_K=.03
    MAX_GUARD_PRIOR_ERROR_K=.05
    GUARD_FIRST_SERVICE_HOURS=2.

    def reset(self):
        super().reset()
        self.heat_guard_choices=0
        self.heat_guard_zone_decisions=0
        self.heat_guard_added_target_k=0.
        self.heat_guard_predicted_shortfall_k=0.

    def _storage_choice_prediction(self,group,temperature,choice,outside):
        previous=self.history[-1]
        return storage_temperature_step(temperature[group],temperature[group]-self.arx_observed[group],
                                        previous['conductance'][group],previous['supply'][group],previous['reheat'][group],
                                        choice['conductance'],choice['supply'],choice['reheat'],outside,self.plan_mode,
                                        self.arx_theta[group],self.arx_residual[group])

    def _choose_ahu(self,group,temperature,free,low,high,max_flow,outside,water):
        normal=super()._choose_ahu(group,temperature,free,low,high,max_flow,outside,water)
        if (not self.ENABLE_HEAT_GUARD or not normal['run'] or not self.plan_mode or not self.history
                or self.arx_observed is None
                or not np.any(self.history[-1]['reheat'][group]>50.)):
            return normal
        elapsed=(self.plan_hour-self.start_hour)%24.
        if elapsed>=self.GUARD_FIRST_SERVICE_HOURS:
            return normal
        qualified=(self.storage_gate[group] & np.isfinite(self.storage_prefit_error[group])
                   & (self.storage_prefit_error[group]<self.MAX_GUARD_PRIOR_ERROR_K)
                   & (self.history[-1]['reheat'][group]>50.))
        prediction=self._storage_choice_prediction(group,temperature,normal,outside)
        shortfall=np.maximum(low[group]-prediction,0.)
        eligible=qualified & np.isfinite(prediction) & (shortfall>self.MIN_GUARD_DISAGREEMENT_K)
        if not eligible.any():
            return normal
        charge=low.copy()
        lift=np.where(eligible,np.minimum(self.GUARD_GAIN*shortfall,self.GUARD_LIMIT_K),0.)
        charge[group]=np.minimum(high[group]-.05,low[group]+lift)
        added=self._plain_choice(group,temperature,free,charge,high,max_flow,outside,water)
        if not added['run']:
            return normal
        revised=self._storage_choice_prediction(group,temperature,added,outside)
        # Only a strictly smaller qualified predicted shortfall may replace
        # the normal proposal. No post-hoc action mutation is performed.
        old_loss=float(np.sum(shortfall[eligible]))
        new_loss=float(np.sum(np.maximum(low[group][eligible]-revised[eligible],0.)))
        if not np.isfinite(new_loss) or new_loss>=old_loss-1e-6:
            return normal
        self.heat_guard_choices+=1
        self.heat_guard_zone_decisions+=int(eligible.sum())
        self.heat_guard_added_target_k+=float(lift.sum())
        self.heat_guard_predicted_shortfall_k+=old_loss-new_loss
        self.required_water_target=max(self.required_water_target,self._required_target(group,temperature,free,charge,max_flow,added))
        return added

    def experiment_diagnostics(self):
        return {**super().experiment_diagnostics(),'heat_guard_choices':self.heat_guard_choices,
                'heat_guard_zone_decisions':self.heat_guard_zone_decisions,
                'heat_guard_added_target_k':self.heat_guard_added_target_k,
                'heat_guard_predicted_shortfall_reduction_k':self.heat_guard_predicted_shortfall_k,
                'heat_guard_limit_k':self.GUARD_LIMIT_K,'heat_guard_first_service_hours':self.GUARD_FIRST_SERVICE_HOURS}

def heating_target_increment(shortfall,reference_response,forecast_response,conductance):
    reference_gain=np.maximum(np.asarray(reference_response,dtype=float),.01)/1000.
    forecast_gain=np.maximum(np.asarray(forecast_response,dtype=float),.01)/1000.
    sensitivity=(forecast_gain/reference_gain)*(1.+reference_gain*conductance)/(1.+forecast_gain*conductance)
    return np.asarray(shortfall,dtype=float)/np.maximum(sensitivity,.1)

class RefinedForecastHeatPolicy(ForecastHeatGuardPolicy):
    name='refined_heat_guard50_nov13'
    GUARD_LIMIT_K=.5
    REFINEMENT_PASSES=2

    def reset(self):
        super().reset()
        self.heat_refinement_passes=0

    def _choose_ahu(self,group,temperature,free,low,high,max_flow,outside,water):
        # Bypass only the parent's fixed-gain guard. The confirmed physical
        # proposal, grid/pump law and disabled multi-step planner remain.
        normal=ShortStorageHeatingPlanPolicy._choose_ahu(self,group,temperature,free,low,high,max_flow,outside,water)
        if (not self.ENABLE_HEAT_GUARD or not normal['run'] or not self.plan_mode or not self.history
                or self.arx_observed is None or (self.plan_hour-self.start_hour)%24.>=self.GUARD_FIRST_SERVICE_HOURS
                or not np.any(self.history[-1]['reheat'][group]>50.)):
            return normal
        qualified=(self.storage_gate[group] & np.isfinite(self.storage_prefit_error[group])
                   & (self.storage_prefit_error[group]<self.MAX_GUARD_PRIOR_ERROR_K)
                   & (self.history[-1]['reheat'][group]>50.))
        prediction=self._storage_choice_prediction(group,temperature,normal,outside)
        shortfall=np.maximum(low[group]-prediction,0.)
        eligible=qualified & np.isfinite(prediction) & (shortfall>self.MIN_GUARD_DISAGREEMENT_K)
        if not eligible.any():
            return normal
        original_loss=float(np.sum(shortfall[eligible]))
        best,best_loss=normal,original_loss
        cumulative=np.zeros(group.size)
        best_lift=cumulative.copy()
        for _ in range(self.REFINEMENT_PASSES):
            predicted=self._storage_choice_prediction(group,temperature,best,outside)
            gap=np.maximum(low[group]-predicted,0.)
            storage_response=self.arx_theta[group,5]/(1.+self.arx_theta[group,1]/20.)
            increment=heating_target_increment(gap,self.theta[group,5],storage_response,best['conductance'])
            cumulative=np.where(eligible,np.minimum(cumulative+np.clip(increment,0.,self.GUARD_LIMIT_K),self.GUARD_LIMIT_K),0.)
            requested=low.copy()
            requested[group]=np.minimum(high[group]-.05,low[group]+cumulative)
            candidate=self._plain_choice(group,temperature,free,requested,high,max_flow,outside,water)
            if not candidate['run']:
                break
            revised=self._storage_choice_prediction(group,temperature,candidate,outside)
            loss=float(np.sum(np.maximum(low[group][eligible]-revised[eligible],0.)))
            self.heat_refinement_passes+=1
            if not np.isfinite(loss) or loss>=best_loss-1e-6:
                break
            best,best_loss,best_lift=candidate,loss,cumulative.copy()
        if best is normal:
            return normal
        self.heat_guard_choices+=1
        self.heat_guard_zone_decisions+=int(eligible.sum())
        self.heat_guard_added_target_k+=float(best_lift.sum())
        self.heat_guard_predicted_shortfall_k+=original_loss-best_loss
        required_low=low.copy()
        required_low[group]+=best_lift
        self.required_water_target=max(self.required_water_target,self._required_target(group,temperature,free,required_low,max_flow,best))
        return best

    def experiment_diagnostics(self):
        return {**super().experiment_diagnostics(),'heat_refinement_passes':self.heat_refinement_passes}

class SmallRefinedForecastHeatPolicy(RefinedForecastHeatPolicy):
    name='refined_heat_guard25_nov13'
    GUARD_LIMIT_K=.25

class FrontloadedStartupPolicy(ObserverGridPumpPolicy):
    name='startup_frontload_nov13'
    STARTUP_EXPONENT=.5
    USE_SMOOTHSTEP=False
    ENABLE_STARTUP_SHAPE=True
    HEATING_ONLY=True

    def reset(self):
        super().reset()
        self.startup_shape_steps=0
        self.startup_shape_zone_steps=0
        self.startup_shape_target_added_k=0.

    def _band(self,obs,temperature):
        low,high,mode=super()._band(obs,temperature)
        soon=bool(obs[self.comfort_soon_index]>.5) if self.comfort_soon_index is not None else mode
        if not self.ENABLE_STARTUP_SHAPE or mode or not soon:
            return low,high,mode
        horizon=float(self.parameters['precondition_hours'])
        remaining=(self.start_hour-self._hour(obs))%24.
        if not np.isfinite(remaining) or horizon<=0. or not 1e-6<remaining<=horizon+1e-6:
            return low,high,mode
        progress=float(np.clip(1.-remaining/horizon,0.,1.))
        shaped=progress**self.STARTUP_EXPONENT
        if self.USE_SMOOTHSTEP:
            shaped=3.*progress**2-2.*progress**3
        margin=float(self.parameters['comfort_margin'])
        target=self.eco_band[0]+shaped*(self.comfort_band[0]-self.eco_band[0])+margin
        original=self.eco_band[0]+progress*(self.comfort_band[0]-self.eco_band[0])+margin
        shift=target-original
        # Preserve the inherited per-zone fixed-margin qualification and any
        # gradual recovery transform. Only the ramp's progress is reshaped.
        recovery=max(1.,float(self.parameters['recovery_steps']))
        absolute=np.where(low>temperature,temperature+recovery*(low-temperature),low)
        desired=absolute+shift
        changed=np.isfinite(temperature)&np.isfinite(self.model_error)&(self.model_error<.08)
        if self.HEATING_ONLY:
            changed&=(temperature<self.comfort_band[0])&(self.slow_temperature<self.comfort_band[0])
        requested=np.minimum(desired,temperature+np.maximum((desired-temperature)/recovery,0.))
        if not changed.any():
            return low,high,mode
        result=low.copy()
        result[changed]=np.minimum(high[changed]-.05,requested[changed])
        self.startup_shape_steps+=1
        self.startup_shape_zone_steps+=int(changed.sum())
        self.startup_shape_target_added_k+=float(np.sum(result[changed]-low[changed]))
        return result,high,mode

    def experiment_diagnostics(self):
        return {**super().experiment_diagnostics(),'startup_shape_steps':self.startup_shape_steps,
                'startup_shape_zone_steps':self.startup_shape_zone_steps,
                'startup_shape_target_added_k':self.startup_shape_target_added_k}

class SmoothStartupPolicy(FrontloadedStartupPolicy):
    name='startup_smoothstep_nov13'
    USE_SMOOTHSTEP=True

class SmoothRefinedRecoveryPolicy(SmallRefinedForecastHeatPolicy, SmoothStartupPolicy):
    name = "smooth_refined_recovery25_nov13"

class StorageDispatchPolicy(SmoothRefinedRecoveryPolicy):
    name = "storage_dispatch3_nov14"
    DISPATCH_HYSTERESIS_K = 3.0
    DISPATCH_CAPACITY_RESERVE_K = 1.5
    DISPATCH_START_AFTER_HOURS = 2.0
    DISPATCH_MIN_HISTORY = 24
    DISPATCH_RELEASE_COMMAND_K = 310.0

    def reset(self):
        super().reset()
        self.dispatch_required_k = 310.0
        self.dispatch_reheat_w = 0.0
        self.dispatch_charging = True
        self.dispatch_observation = None
        self.dispatch_mode = False
        self.dispatch_hour = 0.0
        self.dispatch_release_steps = 0
        self.dispatch_charge_steps = 0
        self.dispatch_capacity_restarts = 0
        self.dispatch_hold_steps = 0
        self.dispatch_water_min_k = 450.0
        self.dispatch_water_max_k = 150.0
        self.dispatch_required_sum_k = 0.0
        self.dispatch_last_command_k = 310.0

    def get_action(self, obs):
        flat = np.asarray(obs, dtype=float).reshape(-1)
        water, valid = self._tank_observation(flat)
        self.dispatch_observation = float(water) if valid else None
        self.dispatch_mode = (bool(flat[self.comfort_now_index] > .5)
                              if self.comfort_now_index is not None else True)
        self.dispatch_hour = self._hour(flat)
        self.dispatch_required_k = 310.0
        self.dispatch_reheat_w = 0.0
        if valid:
            self.dispatch_water_min_k = min(self.dispatch_water_min_k, water)
            self.dispatch_water_max_k = max(self.dispatch_water_max_k, water)
        return super().get_action(flat)

    def _choose_ahu(self, group, temperature, free, low, high, max_flow, outside, water):
        choice = super()._choose_ahu(group, temperature, free, low, high,
                                    max_flow, outside, water)
        if choice["run"]:
            heat = np.asarray(choice["reheat"], dtype=float)
            active = heat > 10.0
            self.dispatch_reheat_w += float(heat.sum())
            if active.any():
                # Use the heat the selected, physically capped coil can
                # actually deliver, rather than an infeasible recovery request.
                max_water_conductance = self.WATER_VALVE_CAPACITY * math.sqrt(20.0)
                need = np.asarray(choice["supply"]) + heat / max_water_conductance
                self.dispatch_required_k = max(self.dispatch_required_k,
                                              float(np.max(need[active])))
        return choice

    def _water_target(self):
        observed = self.dispatch_observation
        elapsed = (self.dispatch_hour - self.start_hour) % 24.0
        eligible = (observed is not None and self.dispatch_mode
                    and self.step_count >= self.DISPATCH_MIN_HISTORY
                    and elapsed >= self.DISPATCH_START_AFTER_HOURS
                    and self.dispatch_reheat_w > 10.0)
        if not eligible:
            self.dispatch_charging = True
            self.dispatch_hold_steps += 1
            target = super()._water_target()
        else:
            lower = min(350.0, max(310.0, self.dispatch_required_k
                                  + self.DISPATCH_CAPACITY_RESERVE_K))
            upper = min(350.0, lower + self.DISPATCH_HYSTERESIS_K)
            self.dispatch_required_sum_k += self.dispatch_required_k
            if observed < lower:
                if not self.dispatch_charging:
                    self.dispatch_capacity_restarts += 1
                self.dispatch_charging = True
            elif observed >= upper:
                self.dispatch_charging = False
            if self.dispatch_charging or lower >= 350.0:
                target = 350.0
                self.dispatch_charge_steps += 1
            else:
                target = self.DISPATCH_RELEASE_COMMAND_K
                self.dispatch_release_steps += 1
            self.held_water_target = float(target)
            self.water_reserve_ever_qualified = True
            self.maximum_water_target = max(self.maximum_water_target, float(target))
        self.dispatch_last_command_k = float(target)
        return float(target)

    def experiment_diagnostics(self):
        eligible_steps = self.dispatch_release_steps + self.dispatch_charge_steps
        return {**super().experiment_diagnostics(),
                "dispatch_hysteresis_k": self.DISPATCH_HYSTERESIS_K,
                "dispatch_capacity_reserve_k": self.DISPATCH_CAPACITY_RESERVE_K,
                "dispatch_release_steps": self.dispatch_release_steps,
                "dispatch_charge_steps": self.dispatch_charge_steps,
                "dispatch_capacity_restarts": self.dispatch_capacity_restarts,
                "dispatch_hold_steps": self.dispatch_hold_steps,
                "dispatch_observed_min_k": self.dispatch_water_min_k,
                "dispatch_observed_max_k": self.dispatch_water_max_k,
                "dispatch_mean_required_k": self.dispatch_required_sum_k / max(1, eligible_steps)}

class HotStorageReservePolicy(StorageDispatchPolicy):
    name = "hot_storage345_nov14"
    HOT_STORAGE_LOW_K = 345.0
    HOT_STORAGE_HIGH_K = 350.0

    def reset(self):
        super().reset()
        self.hot_storage_charging = True
        self.hot_storage_release_steps = 0
        self.hot_storage_charge_steps = 0
        self.hot_storage_required_capacity_sum_k = 0.0
        self.hot_storage_temperature_sum_k = 0.0

    def _water_target(self):
        observed = self.dispatch_observation
        if observed is None or self.dispatch_reheat_w <= 10.0:
            return super()._water_target()
        if observed < self.HOT_STORAGE_LOW_K:
            self.hot_storage_charging = True
        elif observed >= self.HOT_STORAGE_HIGH_K:
            self.hot_storage_charging = False
        if self.hot_storage_charging:
            target = self.HOT_STORAGE_HIGH_K
            self.hot_storage_charge_steps += 1
        else:
            target = self.DISPATCH_RELEASE_COMMAND_K
            self.hot_storage_release_steps += 1
        self.hot_storage_temperature_sum_k += observed
        self.hot_storage_required_capacity_sum_k += self.dispatch_required_k
        self.held_water_target = float(target)
        self.water_reserve_ever_qualified = True
        self.maximum_water_target = max(self.maximum_water_target, float(target))
        self.dispatch_last_command_k = float(target)
        return float(target)

    def experiment_diagnostics(self):
        count = self.hot_storage_release_steps + self.hot_storage_charge_steps
        return {**super().experiment_diagnostics(),
                "hot_storage_low_k": self.HOT_STORAGE_LOW_K,
                "hot_storage_high_k": self.HOT_STORAGE_HIGH_K,
                "hot_storage_release_steps": self.hot_storage_release_steps,
                "hot_storage_charge_steps": self.hot_storage_charge_steps,
                "hot_storage_mean_water_k": self.hot_storage_temperature_sum_k / max(1, count),
                "hot_storage_mean_required_capacity_k": self.hot_storage_required_capacity_sum_k / max(1, count)}

class WideHotStorageReservePolicy(HotStorageReservePolicy):
    name = "hot_storage340_nov14"
    HOT_STORAGE_LOW_K = 340.0
