"""Canonical action representation: commanded vs effective (Phase 1.5).

The simulator does not always apply what is commanded. These rules come from
the starter-kit source and are verified against the action echoes in
obs[t+1] over the Phase 1 sample (configs/sensor_alignment.json):

  AHU static pressure   eff = run ? max(cmd, P_MIN) : 0        P_MIN = 200 Pa
  AHU recirculation     eff = clip(cmd, 0, R_MAX)               R_MAX = 0.7 (any run state)
  VAV damper position   eff = max(cmd, D_MIN)                   D_MIN = 0.1
  everything else       eff = cmd   (SAT setpoint, reheat valve, HWS setpoints, run bits)

All rules need only the commanded action and its own run bits, so they are
available at decision time and never read a future observation. `active`
marks entries that have a physical effect this step: AHU-level entries and the
dampers/valves of zones whose AHU is off are inactive (their value is applied
but no air moves).

Quantities that are *not* actions and cannot be reconstructed at decision time
are deliberately absent:
  * delivered supply-air temperature = min(mixed air, SAT setpoint); the mixed
    (return) air temperature is not observed. The AHU 'supply_air_temperature_sensor'
    is only the setpoint echo.
  * actual VAV/zone air flow (depends on pressure, damper and demand); the AHU
    total flow appears in obs[t+1] only, i.e. after the decision.

The limits are global constants in all 30 shipped configs (`min_static_pressure`,
`max_recirculation`, `min_damper_setting`). For unseen buildings use
`verify_against_echo()` on each new observation to detect a different limit.

Units: `*_native` arrays are in physical units (K, Pa, fraction, bar-equivalent);
`*_normalized` arrays use the env's [-1, 1] action normalization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from building_meta import ActionCodec, ObservationCodec

SCHEMA_VERSION = "effective-action-v1"
MIN_STATIC_PRESSURE_PA = 200.0
MAX_RECIRCULATION = 0.7
MIN_DAMPER = 0.1

RULE_IDENTITY = 0
RULE_PRESSURE = 1
RULE_RECIRC = 2
RULE_DAMPER = 3
RULE_NAMES = {RULE_IDENTITY: "identity", RULE_PRESSURE: "run*max(cmd,200Pa)",
              RULE_RECIRC: "clip(cmd,0,0.7)", RULE_DAMPER: "max(cmd,0.1)"}


@dataclass
class EffectiveAction:
    commanded_native: np.ndarray     # [..., n_c]
    effective_native: np.ndarray     # [..., n_c]
    effective_normalized: np.ndarray  # [..., n_c]
    discrete: np.ndarray             # [..., n_d] (unchanged; commanded == effective)
    active: np.ndarray               # [..., n_c] bool
    rule: np.ndarray                 # [n_c] int (RULE_*)


class ActionRepresentation:
    """Building-specific layout of the effective-action transform (any K, V)."""

    def __init__(self, scenario_meta: Mapping[str, Any], min_static_pressure: float = MIN_STATIC_PRESSURE_PA,
                 max_recirculation: float = MAX_RECIRCULATION, min_damper: float = MIN_DAMPER):
        self.meta = scenario_meta
        self.codec = ActionCodec(scenario_meta)
        self.nc = int(scenario_meta["n_continuous_action"])
        self.nd = int(scenario_meta["n_discrete_action"])
        self.p_min, self.r_max, self.d_min = float(min_static_pressure), float(max_recirculation), float(min_damper)
        ah, z = scenario_meta["ahus"], scenario_meta["zones"]
        self.rule = np.full(self.nc, RULE_IDENTITY, dtype=np.int8)
        self.rule[ah["act_static_pressure"]] = RULE_PRESSURE
        self.rule[ah["act_recirculation"]] = RULE_RECIRC
        self.rule[z["act_damper"]] = RULE_DAMPER
        # Which AHU's run bit governs each continuous entry (-1: none).
        self.owner_run = np.full(self.nc, -1, dtype=np.int64)
        for k, run_idx in enumerate(ah["act_run"]):
            for key in ("act_sat", "act_static_pressure", "act_recirculation"):
                self.owner_run[ah[key][k]] = run_idx
            for zi in ah["zones"][k]:
                self.owner_run[z["act_damper"][zi]] = run_idx
                self.owner_run[z["act_reheat_valve"][zi]] = run_idx
        self.continuous_names = list(scenario_meta["continuous_action_names"])

    def effective(self, action_continuous, action_discrete) -> EffectiveAction:
        """Vectorized over leading dims. Inputs: normalized continuous [-1,1], discrete {0,1}."""
        a = np.asarray(action_continuous, dtype=np.float64)
        d = np.asarray(action_discrete)
        if a.shape[-1] != self.nc or d.shape[-1] != self.nd:
            raise ValueError(f"expected [..., {self.nc}] / [..., {self.nd}], got {a.shape} / {d.shape}")
        cmd = self.codec.to_native(a)
        eff = cmd.copy()
        run_of = np.ones(a.shape, dtype=bool)
        has_owner = self.owner_run >= 0
        run_of[..., has_owner] = d[..., self.owner_run[has_owner]].astype(bool)
        p = self.rule == RULE_PRESSURE
        eff[..., p] = np.where(run_of[..., p], np.maximum(cmd[..., p], self.p_min), 0.0)
        r = self.rule == RULE_RECIRC
        eff[..., r] = np.clip(cmd[..., r], 0.0, self.r_max)
        dm = self.rule == RULE_DAMPER
        eff[..., dm] = np.maximum(cmd[..., dm], self.d_min)
        active = run_of.copy()
        return EffectiveAction(commanded_native=cmd, effective_native=eff,
                               effective_normalized=self.codec.to_normalized(eff), discrete=d.copy(),
                               active=active, rule=self.rule.copy())

    def echo_fields(self) -> dict[int, str]:
        """Continuous index -> obs field that echoes its effective value in obs[t+1]."""
        names = set(self.meta["observation_field_names"])
        return {j: n for j, n in enumerate(self.continuous_names) if n in names}

    def verify_against_echo(self, action_continuous, action_discrete, obs_next, atol: float = 1e-3) -> dict[str, Any]:
        """Offline / after-the-fact check: effective(a_t) vs the echoes in obs[t+1].

        At run time this can be applied one step late (obs[t+1] arrives after a_t)
        to detect a building whose actuator limits differ from the constants."""
        eff = self.effective(action_continuous, action_discrete).effective_native
        oc = ObservationCodec(self.meta)
        nat = oc.native(np.asarray(obs_next))
        idx = oc.index
        worst = {}
        for j, name in self.echo_fields().items():
            err = np.abs(nat[..., idx[name]] - eff[..., j])
            worst[name] = float(np.max(err))
        bad = {k: v for k, v in worst.items() if v > atol * max(1.0, abs(self.codec.high[self.continuous_names.index(k)]))}
        return {"max_abs_err": max(worst.values()) if worst else 0.0, "violations": bad, "checked": len(worst)}
