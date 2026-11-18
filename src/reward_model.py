"""Exact competition reward, reconstructed from physical quantities (pure NumPy).

This module has no dependency on the simulator. It reproduces
`SetpointEnergyCarbonRegretFunction.compute_reward` (starter kit, see
docs/phase0_findings.md §4) from:

  * per-zone air temperature, heating/cooling setpoints and occupancy weights,
  * total electricity rate E [W] (blowers + |air conditioning| + pumps),
  * total natural-gas rate G [W] (burner heat),
  * the hour, work-day flag and (optionally) month of the reward interval,
  * K = number of AHUs, read from building metadata (never hard-coded).

All organizer-tunable constants live in `RewardConstants` so a rule change is a
one-line update. `scripts/validate_phase1.py` asserts that the tables below are
identical to the starter-kit module constants.

Notes
-----
* The reward of interval [t, t+1] is labelled by the simulator with
  `start_timestamp` = the timestamp *after* the step (Phase 0 §3), which equals
  the hour decoded from the next observation o_{t+1}. Tariff lookups must use
  that hour.
* The month is not observable at run time. `month=None` uses the annual-mean
  gas price (causal fallback, ≤ 4.6 % error on the energy term, Phase 0 §3).
* Comfort needs per-zone occupancy, which is not observable (only the building
  total `num_occupants`). `occupancy_weights()` provides the uniform and
  area-weighted assumptions compared in Phase 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- public simulator tables (starter-kit module constants, copied verbatim) ---
WEEKDAY_PRICE_CENTS_PER_KWH = np.array(
    [16.0] * 6 + [18.0] * 6 + [20.0] * 7 + [16.0] * 5)
WEEKEND_PRICE_CENTS_PER_KWH = np.full(24, 16.0)
CARBON_G_PER_KWH_BY_HOUR = np.array([
    88.19666493, 87.79190866, 87.87607686, 87.83054163, 88.00279618, 88.19648183,
    89.70663283, 93.97947901, 98.85868291, 100.7853521, 101.3866866, 101.7795612,
    102.5919168, 103.4403736, 104.1380294, 104.7359292, 102.0714466, 97.04226176,
    93.57895651, 92.46355045, 91.72914657, 90.69209747, 89.76552213, 88.99950995,
])
GAS_PRICE_BY_MONTH = np.array(
    [9.02, 8.35, 7.77, 7.26, 6.69, 6.86, 6.77, 6.76, 6.99, 7.19, 7.96, 8.98])
GAS_CO2 = 53.12                 # kg CO2 / MMBtu (constants.GAS_CO2)
KWH_PER_KFT3_GAS = 293.07107    # constants.KWH_PER_KFT3_GAS
JOULES_PER_KWH = 3.6e6


@dataclass(frozen=True)
class RewardConstants:
    """Organizer-tunable reward constants (Phase 0 §4). Global, not per building."""
    c_electricity_per_ahu_w: float = 80_000.0
    c_natural_gas_per_ahu_w: float = 205_000.0
    productivity_deviation_ref_k: float = 5.0
    productivity_weight: float = 0.6
    energy_cost_weight: float = 0.2
    carbon_emission_weight: float = 0.2
    weekday_price: np.ndarray = field(default_factory=lambda: WEEKDAY_PRICE_CENTS_PER_KWH.copy())
    weekend_price: np.ndarray = field(default_factory=lambda: WEEKEND_PRICE_CENTS_PER_KWH.copy())
    carbon_by_hour: np.ndarray = field(default_factory=lambda: CARBON_G_PER_KWH_BY_HOUR.copy())
    gas_price_by_month: np.ndarray = field(default_factory=lambda: GAS_PRICE_BY_MONTH.copy())

    # Per-joule rates in the simulator's own units. Only ratios matter, since
    # every term is normalized by the same quantity evaluated at the max rate.
    def electricity_price(self, hour, workday):
        hour = np.asarray(hour, dtype=int)
        table = np.where(np.asarray(workday, dtype=bool),
                         self.weekday_price[hour], self.weekend_price[hour])
        return table / 100.0 / 1000.0 / 3600.0

    def electricity_carbon(self, hour):
        return self.carbon_by_hour[np.asarray(hour, dtype=int)] / 1.0e6 / 3600.0

    def gas_price(self, month=None):
        """$/J. `month` is 1-12; None -> annual mean (causal fallback)."""
        if month is None:
            prices = np.mean(self.gas_price_by_month)
        else:
            prices = self.gas_price_by_month[np.asarray(month, dtype=int) - 1]
        return prices / KWH_PER_KFT3_GAS / JOULES_PER_KWH

    @property
    def gas_carbon(self) -> float:
        return GAS_CO2 / KWH_PER_KFT3_GAS / JOULES_PER_KWH


DEFAULT_CONSTANTS = RewardConstants()


def productivity_shortfall(deviation, ref: float = 5.0):
    """s(d): d/ref for d <= ref, 1 + ((d-ref)/ref)^2 beyond (unbounded)."""
    d = np.maximum(np.asarray(deviation, dtype=float), 0.0)
    return np.where(d <= ref, d / ref, 1.0 + ((d - ref) / ref) ** 2)


def zone_deviation(zone_temp, heating_sp, cooling_sp):
    """Kelvin outside the [heating, cooling] deadband (0 inside)."""
    t = np.asarray(zone_temp, dtype=float)
    return np.maximum(0.0, np.maximum(t - np.asarray(cooling_sp), np.asarray(heating_sp) - t))


def comfort_term(zone_temp, heating_sp, cooling_sp, occupancy, ref: float = 5.0):
    """Normalized productivity regret in [-inf, 0]; 0 when nobody is present.

    Arrays may be [V] or [T, V]; occupancy weights broadcast against them.
    Equals (sum occ*(1-s))/sum occ - 1 = -sum(occ*s)/sum(occ).
    """
    s = productivity_shortfall(zone_deviation(zone_temp, heating_sp, cooling_sp), ref)
    occ = np.broadcast_to(np.asarray(occupancy, dtype=float), s.shape)
    total = occ.sum(axis=-1)
    weighted = (occ * s).sum(axis=-1)
    return np.where(total > 0.0, -weighted / np.where(total > 0.0, total, 1.0), 0.0)


def energy_terms(elec_w, gas_w, hour, workday, num_ahus, month=None,
                 c: RewardConstants = DEFAULT_CONSTANTS):
    """(normalized_energy_cost, normalized_carbon_emission) for each step."""
    elec_w = np.abs(np.asarray(elec_w, dtype=float))
    gas_w = np.maximum(np.asarray(gas_w, dtype=float), 0.0)
    max_e = c.c_electricity_per_ahu_w * num_ahus
    max_g = c.c_natural_gas_per_ahu_w * num_ahus
    pe, pg = c.electricity_price(hour, workday), c.gas_price(month)
    ke, kg = c.electricity_carbon(hour), c.gas_carbon
    energy = (pe * elec_w + pg * gas_w) / (pe * max_e + pg * max_g)
    carbon = (ke * elec_w + kg * gas_w) / (ke * max_e + kg * max_g)
    return energy, carbon


def step_reward(zone_temp, heating_sp, cooling_sp, occupancy, elec_w, gas_w,
                hour, workday, num_ahus, month=None,
                c: RewardConstants = DEFAULT_CONSTANTS) -> dict[str, np.ndarray]:
    """Full per-step reward and its weighted parts (comfort, -energy, -carbon)."""
    comfort = comfort_term(zone_temp, heating_sp, cooling_sp, occupancy,
                           c.productivity_deviation_ref_k)
    energy, carbon = energy_terms(elec_w, gas_w, hour, workday, num_ahus, month, c)
    w = c.productivity_weight + c.energy_cost_weight + c.carbon_emission_weight
    parts = {
        "comfort": c.productivity_weight * comfort / w,
        "energy": -c.energy_cost_weight * energy / w,
        "carbon": -c.carbon_emission_weight * carbon / w,
    }
    return {"reward": parts["comfort"] + parts["energy"] + parts["carbon"],
            "normalized_productivity_regret": comfort,
            "normalized_energy_cost": energy,
            "normalized_carbon_emission": carbon, **parts}


def occupancy_weights(kind: str, num_zones: int, total_occupancy=None, zone_area=None):
    """Per-zone occupancy assumption from the observable building total.

    kind: "uniform" (equal split) or "area" (proportional to floor area).
    `total_occupancy` (scalar or [T]) only gates comfort on/off; the comfort
    term is invariant to the scale of the weights.
    """
    if kind == "uniform":
        w = np.full(num_zones, 1.0 / num_zones)
    elif kind == "area":
        if zone_area is None:
            raise ValueError("area weighting needs zone_area")
        a = np.asarray(zone_area, dtype=float)
        w = a / a.sum()
    else:
        raise ValueError(kind)
    if total_occupancy is None:
        return w
    total = np.asarray(total_occupancy, dtype=float)
    return total[..., None] * w
