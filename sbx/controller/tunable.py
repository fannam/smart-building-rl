"""Tunable wrapper around the reference controller (PLAN.md, M3).

`reference.py` stays verbatim (third-party code). This module exposes its constants as named parameters with search
bounds and adds a few new behaviours, all switched off by default so that `TunablePolicy()` with default
parameters reproduces `SmoothRefinedRecoveryPolicy` exactly (tests/test_tunable.py, and the M3 gate).

Three kinds of parameters:
  * "defaults": keys of the reference `DEFAULTS` dict, passed through its own `parameters` argument;
  * "constants": reference class constants, overridden per instance (they are always read as `self.X`). The
    reference's internal observer (`GatedStorageARXPolicy.reference_observer`) is a separate
    `ObservedWaterCoastPolicy` instance and keeps the reference constants;
  * "new": behaviours this module adds (see `_band`), neutral at their defaults.

Note on the clock: the simulator's `comfort_mode_soon` flag looks exactly 60 min ahead, so the reference can never
start preconditioning earlier than one hour before occupancy whatever `precondition_hours` says. `early_lead_*`
adds an earlier, outside-temperature-dependent ramp on workdays.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from sbx.controller.reference import SmoothRefinedRecoveryPolicy

BASE = SmoothRefinedRecoveryPolicy


@dataclass(frozen=True)
class Spec:
    default: float
    low: float
    high: float
    kind: str  # "defaults" | "constants" | "new"
    log: bool = False
    doc: str = ""


def _d(name: str, low: float, high: float, log: bool = False, doc: str = "") -> tuple[str, Spec]:
    return name, Spec(float(BASE.DEFAULTS[name]), low, high, "defaults", log, doc)


def _c(name: str, low: float, high: float, log: bool = False, doc: str = "") -> tuple[str, Spec]:
    return name, Spec(float(getattr(BASE, name)), low, high, "constants", log, doc)


PARAMS: dict[str, Spec] = dict([
    # -- reference DEFAULTS --------------------------------------------------------------------------------------
    _d("comfort_margin", 0.0, 0.4, doc="K inside the comfort band while occupied / preconditioning"),
    _d("precondition_hours", 0.25, 1.0, doc="ramp length inside the 1 h comfort-soon window"),
    _d("recovery_steps", 1.0, 4.0, doc="spread a band step over this many 5-min steps"),
    _d("forgetting_factor", 0.99, 0.9995, doc="RLS forgetting of the zone thermal model"),
    _d("model_residual_gain", 0.0, 0.5),
    _d("comfort_cost", 2e4, 2e5, log=True, doc="comfort weight in the AHU operating-point objective"),
    _d("gas_cost_ratio", 0.1, 0.8, doc="gas vs electricity weight in the AHU objective"),
    _d("hot_water_target", 320.0, 355.0, doc="K, boiler supply target while heating"),
    _d("standby_minimum", 268.0, 292.0, doc="K, setback heating limit when unoccupied"),
    _d("standby_maximum", 300.0, 318.0, doc="K, setback cooling limit when unoccupied"),
    _d("response_scale", 0.4, 1.5, log=True, doc="prior zone response gain"),
    _d("disturbance_rate", 0.1, 0.8),
    # -- reference class constants -------------------------------------------------------------------------------
    _c("EMPTY_CONFIRMATION_HOURS", 0.25, 2.0, doc="h of an empty occupied period before relaxing to standby"),
    _c("FIXED_MARGIN_FRACTION", 0.0, 1.0, doc="share of comfort_margin kept for well-modelled zones"),
    _c("MIN_HEATING_TARGET_K", 305.0, 335.0, doc="K, floor of the demand-sized hot-water target"),
    _c("TARGET_HEADROOM_K", 0.0, 6.0, doc="K added to the demand-sized hot-water target"),
    _c("COIL_CAPACITY_HEADROOM", 0.8, 1.0),
    _c("MIN_PRESSURE", 100.0, 600.0, doc="Pa, supply static-pressure floor"),
    _c("MIN_DAMPER", 0.03, 0.3, doc="minimum VAV damper fraction"),
    _c("MAX_RECIRCULATION", 0.5, 0.95, doc="recirculation cap"),
    _c("MAX_DISCHARGE_K", 300.0, 318.0, doc="K, maximum VAV discharge air temperature"),
    _c("GUARD_LIMIT_K", 0.0, 1.0, doc="K, forecast heat-guard lift cap"),
    _c("GUARD_GAIN", 0.0, 1.5),
    # -- new behaviours (neutral defaults) ------------------------------------------------------------------------
    ("early_lead_hours", Spec(0.0, 0.0, 3.0, "new", doc="h of workday preheat before the 1 h comfort-soon window")),
    ("early_lead_cold_slope", Spec(0.0, 0.0, 0.15, "new", doc="extra lead, h per K of outside air below cold_ref_k")),
    ("early_lead_max_hours", Spec(4.0, 1.0, 6.0, "new", doc="cap on the total extra lead")),
    ("cold_ref_k", Spec(283.15, 258.15, 293.15, "new", doc="K, outside temperature where cold adjustments start")),
    ("standby_cold_slope", Spec(0.0, 0.0, 0.5, "new", doc="raise standby_minimum by K per K below cold_ref_k")),
    ("weekend_standby_offset", Spec(0.0, -6.0, 0.0, "new", doc="K added to standby_minimum on weekend days")),
])

DEFAULTS = {k: s.default for k, s in PARAMS.items()}
WORKDAYS = frozenset(range(5))  # Monday..Friday (pandas weekday numbering, as the simulator's day-of-week feature)


def bounds() -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([s.low for s in PARAMS.values()])
    hi = np.array([s.high for s in PARAMS.values()])
    return lo, hi


class TunablePolicy(BASE):
    """`SmoothRefinedRecoveryPolicy` with every tuned constant exposed through `params`."""

    def __init__(self, building_meta_data: Mapping[str, Any], action_spec: Any = None,
                 params: Mapping[str, float] | None = None) -> None:
        p = dict(DEFAULTS)
        if params:
            unknown = set(params) - set(PARAMS)
            if unknown:
                raise ValueError(f"Unknown parameters: {sorted(unknown)}")
            p.update({k: float(v) for k, v in params.items()})
        self.params = p
        for k, s in PARAMS.items():
            if s.kind == "constants" and p[k] != s.default:
                setattr(self, k, p[k])
        reference = {k: p[k] for k, s in PARAMS.items() if s.kind == "defaults"}
        super().__init__(building_meta_data, action_spec, reference)
        self._dow = (self.field_index.get("dow_sin_000"), self.field_index.get("dow_cos_000"))
        self._new_active = any(p[k] != s.default for k, s in PARAMS.items() if s.kind == "new")

    # -- clock helpers ---------------------------------------------------------------------------------------------
    def _weekday(self, obs: np.ndarray) -> int | None:
        """Local day of week, 0 = Monday. The feature is 2*pi*weekday/7, so round (never floor) the angle."""
        s, c = self._dow
        if s is None or c is None:
            return None
        angle = math.atan2(float(obs[s]), float(obs[c])) % (2.0 * math.pi)
        return int(round(angle * 7.0 / (2.0 * math.pi))) % 7

    def _outside_now(self, obs: np.ndarray) -> float:
        values = [self._temperature(obs, i, "outside_air_temperature_sensor", float("nan")) for i in self.outside_obs]
        values = [v for v in values if np.isfinite(v)]
        return float(np.mean(values)) if values else float("nan")

    # -- new behaviours --------------------------------------------------------------------------------------------
    def _band(self, obs: np.ndarray, temperature: np.ndarray):
        low, high, mode = super()._band(obs, temperature)
        if not self._new_active:
            return low, high, mode
        p = self.params
        now = bool(obs[self.comfort_now_index] > 0.5) if self.comfort_now_index is not None else mode
        soon = bool(obs[self.comfort_soon_index] > 0.5) if self.comfort_soon_index is not None else now
        if now:
            return low, high, mode
        hour = self._hour(obs)
        weekday = self._weekday(obs)
        outside = self._outside_now(obs)
        cold = max(0.0, p["cold_ref_k"] - outside) if np.isfinite(outside) else 0.0
        recovery = max(1.0, float(self.parameters["recovery_steps"]))
        low = low.copy()

        if not soon:
            # Setback floor: weekend offset and a cold-weather raise (both only where the band is the standby band).
            standby = float(self.parameters["standby_minimum"])
            floor = standby + p["standby_cold_slope"] * cold
            if weekday is not None and weekday not in WORKDAYS:
                floor += p["weekend_standby_offset"]
            floor = min(floor, self.comfort_band[0] - 0.5)
            at_standby = np.isclose(low, np.minimum(standby, temperature + np.maximum((standby - temperature) / recovery, 0.0)))
            target = np.minimum(floor, temperature + np.maximum((floor - temperature) / recovery, 0.0))
            low = np.where(at_standby, target, low)

        # Early workday preheat: ramp from the setback floor to the comfort low limit over (1 h + lead).
        lead = min(p["early_lead_max_hours"], p["early_lead_hours"] + p["early_lead_cold_slope"] * cold)
        if lead > 0.0:
            remaining = (self.start_hour - hour) % 24.0
            start_day = weekday if hour < self.start_hour else (None if weekday is None else (weekday + 1) % 7)
            horizon = 1.0 + lead
            if 1e-6 < remaining <= horizon and (start_day is None or start_day in WORKDAYS):
                progress = float(np.clip(1.0 - remaining / horizon, 0.0, 1.0))
                base = float(self.parameters["standby_minimum"])
                ramp = base + progress * (self.comfort_band[0] - base)
                ramp_target = np.minimum(ramp, temperature + np.maximum((ramp - temperature) / recovery, 0.0))
                low = np.maximum(low, np.minimum(ramp_target, high - 0.05))
        return low, high, mode
