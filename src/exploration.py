"""Safe, seeded, high-level exploration around the baseline policy.

Knobs (all piecewise-constant over random hold lengths, so the data contains
low-frequency excitation that a multi-step dynamics model can identify):

  policy-internal (change what the baseline *aims for*; it stays closed-loop)
    * setback      : unoccupied zone band offsets (building-level + per-zone
                     jitter), applied only when comfort_mode_now == comfort_mode_soon == 0
    * precondition : `precondition_hours` (morning recovery lead), held per day

  action-level (applied to the baseline's action, then re-bounded)
    * ahu_schedule : force an AHU OFF (only while unoccupied and not pre-start)
                     or ON (with a drawn static pressure and a mild SAT)
    * sat          : AHU supply-air-temperature offset [K]
    * pressure     : AHU static-pressure multiplicative scale
    * hws          : hot-water supply setpoint offset [K]

Safety:
  * every continuous value is clipped to its native range, then to [-1, 1];
    discrete values stay in {0, 1};
  * a causal comfort guard (uses only o_t): during comfort/pre-start hours, an
    AHU whose zones are > guard_k outside their sensed band gets no
    action-level perturbation this step, and the HWS offset may not be negative
    while any zone is > guard_k below its heating setpoint.

Randomness comes only from a numpy Generator seeded by
SeedSequence([seed, crc32(scenario), episode]); draws never depend on the
observations, so a seed fixes the knob schedule exactly.
"""
from __future__ import annotations

import copy
import zlib
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from baseline_policy import WideHotStorageReservePolicy
from building_meta import ActionCodec, ObservationCodec


@dataclass(frozen=True)
class ExplorationConfig:
    name: str = "custom"
    # AHU schedule override
    ahu_prob: float = 0.0
    ahu_hold: tuple[int, int] = (6, 36)
    ahu_force_on_fraction: float = 0.5
    ahu_on_pressure_pa: tuple[float, float] = (300.0, 900.0)
    ahu_on_sat_k: float = 291.0
    # SAT offset
    sat_prob: float = 0.0
    sat_amp_k: float = 0.0
    sat_hold: tuple[int, int] = (12, 72)
    # static pressure scale (log-uniform in [1/s, s])
    pressure_prob: float = 0.0
    pressure_scale_max: float = 1.0
    pressure_hold: tuple[int, int] = (12, 72)
    # HWS setpoint offset
    hws_prob: float = 0.0
    hws_amp_k: float = 0.0
    hws_hold: tuple[int, int] = (24, 144)
    # unoccupied setback offsets (low raised by [0, a], high lowered by [0, b])
    setback_prob: float = 0.0
    setback_low_max_k: float = 0.0
    setback_high_max_k: float = 0.0
    setback_zone_jitter_k: float = 0.0
    setback_hold: tuple[int, int] = (72, 288)
    # morning recovery lead (precondition_hours), drawn per day
    precondition_prob: float = 0.0
    precondition_range_h: tuple[float, float] = (0.5, 3.0)
    precondition_hold: tuple[int, int] = (288, 288)
    # comfort guard
    guard_k: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PROFILES: dict[str, ExplorationConfig | None] = {
    # the unmodified baseline class, no explorer at all
    "baseline": None,
    # explorer + subclass with every probability 0: must equal "baseline" exactly
    "null": ExplorationConfig(name="null"),
    "mild": ExplorationConfig(
        name="mild", ahu_prob=0.10, sat_prob=0.30, sat_amp_k=2.0,
        pressure_prob=0.30, pressure_scale_max=1.3, hws_prob=0.30, hws_amp_k=5.0,
        setback_prob=0.40, setback_low_max_k=8.0, setback_high_max_k=4.0, setback_zone_jitter_k=0.5,
        precondition_prob=0.30, precondition_range_h=(0.5, 2.0)),
    "moderate": ExplorationConfig(
        name="moderate", ahu_prob=0.20, sat_prob=0.50, sat_amp_k=4.0,
        pressure_prob=0.50, pressure_scale_max=1.8, hws_prob=0.50, hws_amp_k=10.0,
        setback_prob=0.60, setback_low_max_k=14.0, setback_high_max_k=8.0, setback_zone_jitter_k=1.0,
        precondition_prob=0.50, precondition_range_h=(0.25, 3.0)),
}

SAT_RANGE = (285.0, 305.0)
PRESSURE_RANGE = (0.0, 2000.0)
HWS_RANGE = (310.0, 350.0)
MIN_SETBACK_DEADBAND_K = 1.0


def episode_rng(seed: int, scenario: str, episode: int) -> np.random.Generator:
    ss = np.random.SeedSequence([int(seed), zlib.crc32(scenario.encode()), int(episode)])
    return np.random.Generator(np.random.PCG64(ss))


class ExplorationBaselinePolicy(WideHotStorageReservePolicy):
    """The baseline with optional unoccupied band offsets.

    With `band_offset_low is None` it is the baseline: `_band` defers to the
    parent unchanged. Offsets are in the policy's zone order (natural sort of
    VAV ids), which equals the dataset's canonical zone order.
    """

    name = "baseline_explore"

    def reset(self):
        super().reset()
        self.band_offset_low = None
        self.band_offset_high = None

    def _band(self, obs, temperature):
        low, high, mode = super()._band(obs, temperature)
        if self.band_offset_low is None:
            return low, high, mode
        soon = bool(obs[self.comfort_soon_index] > .5) if self.comfort_soon_index is not None else mode
        if mode or soon:
            return low, high, mode
        low = low + self.band_offset_low
        high = high + self.band_offset_high
        low = np.minimum(low, high - MIN_SETBACK_DEADBAND_K)
        return low, high, mode


class _Segment:
    """Piecewise-constant channel: redraws (active, value) when its hold expires."""

    def __init__(self, prob: float, hold: tuple[int, int]):
        self.prob, self.hold = prob, hold
        self.remaining = 0
        self.active = False
        self.payload: Any = None

    def advance(self, rng: np.random.Generator, sampler) -> None:
        if self.remaining <= 0:
            # Fixed draw order and count, independent of prob, so profiles that
            # differ only in amplitude share their activation pattern.
            u = rng.random()
            payload = sampler(rng)
            self.remaining = int(rng.integers(self.hold[0], self.hold[1] + 1))
            self.active = bool(u < self.prob)
            self.payload = payload if self.active else None
        self.remaining -= 1


KNOB_COLUMNS_DOC = """
Per-step knob columns (applied values, after guards/validity), K = #AHUs:
  ahu_force_k        -1 none, 0 forced OFF, 1 forced ON          (K)
  sat_offset_k       K offset added to SAT setpoint              (K)
  pressure_scale_k   multiplicative static-pressure scale        (K)
  hws_offset         K offset added to HWS supply setpoint       (1)
  setback_low        building-level unoccupied low-band offset   (1)
  setback_high       building-level unoccupied high-band offset  (1)
  precondition_h     precondition_hours used by the policy       (1)
  guard_ahu_k        1 if the comfort guard blocked AHU k        (K)
  guard_hws          1 if the guard blocked a negative HWS offset(1)
"""


class Explorer:
    def __init__(self, cfg: ExplorationConfig, scenario_meta: Mapping[str, Any],
                 policy: ExplorationBaselinePolicy, rng: np.random.Generator):
        self.cfg, self.meta, self.policy, self.rng = cfg, scenario_meta, policy, rng
        self.obs_codec = ObservationCodec(scenario_meta)
        self.act_codec = ActionCodec(scenario_meta)
        a = scenario_meta["ahus"]
        self.K = len(a["ids"])
        self.V = scenario_meta["num_zones"]
        self.ahu_run = np.asarray(a["act_run"])
        self.ahu_sat = np.asarray(a["act_sat"])
        self.ahu_p = np.asarray(a["act_static_pressure"])
        self.ahu_zones = [np.asarray(z, dtype=int) for z in a["zones"]]
        self.hws_sw = scenario_meta["hws"]["act_supply_water_temp"]
        names = scenario_meta["observation_field_names"]
        self.i_now = names.index("comfort_mode_now")
        self.i_soon = names.index("comfort_mode_soon")
        self.default_precondition = float(policy.parameters["precondition_hours"])
        if list(policy.zone_ids) != list(scenario_meta["zones"]["ids"]):
            raise ValueError("policy zone order differs from dataset zone order")

        c = cfg
        self.seg_ahu = [_Segment(c.ahu_prob, c.ahu_hold) for _ in range(self.K)]
        self.seg_sat = [_Segment(c.sat_prob, c.sat_hold) for _ in range(self.K)]
        self.seg_p = [_Segment(c.pressure_prob, c.pressure_hold) for _ in range(self.K)]
        self.seg_hws = _Segment(c.hws_prob, c.hws_hold)
        self.seg_setback = _Segment(c.setback_prob, c.setback_hold)
        self.seg_pre = _Segment(c.precondition_prob, c.precondition_hold)
        self.columns = ([f"ahu_force_{k}" for k in range(self.K)] + [f"sat_offset_{k}" for k in range(self.K)]
                        + [f"pressure_scale_{k}" for k in range(self.K)]
                        + ["hws_offset", "setback_low", "setback_high", "precondition_h"]
                        + [f"guard_ahu_{k}" for k in range(self.K)] + ["guard_hws"])

    # ---- samplers (fixed number of draws each) ----
    def _s_ahu(self, rng):
        on = rng.random() < self.cfg.ahu_force_on_fraction
        p = rng.uniform(*self.cfg.ahu_on_pressure_pa)
        return (1 if on else 0, p)

    def _s_sat(self, rng):
        return rng.uniform(-1.0, 1.0) * self.cfg.sat_amp_k

    def _s_p(self, rng):
        s = max(self.cfg.pressure_scale_max, 1.0)
        return float(np.exp(rng.uniform(-np.log(s), np.log(s))))

    def _s_hws(self, rng):
        return rng.uniform(-1.0, 1.0) * self.cfg.hws_amp_k

    def _s_setback(self, rng):
        lo = rng.uniform(0.0, self.cfg.setback_low_max_k)
        hi = -rng.uniform(0.0, self.cfg.setback_high_max_k)
        jit = rng.normal(0.0, 1.0, size=self.V) * self.cfg.setback_zone_jitter_k
        return lo, hi, jit

    def _s_pre(self, rng):
        return rng.uniform(*self.cfg.precondition_range_h)

    # ---- per step ----
    def begin_step(self, obs: np.ndarray) -> bool:
        """Advance segments, configure policy-internal knobs. Returns True if the
        policy-internal knobs differ from the baseline this step (so the
        counterfactual baseline action needs a separate clone)."""
        rng = self.rng
        for s in self.seg_ahu:
            s.advance(rng, self._s_ahu)
        for s in self.seg_sat:
            s.advance(rng, self._s_sat)
        for s in self.seg_p:
            s.advance(rng, self._s_p)
        self.seg_hws.advance(rng, self._s_hws)
        self.seg_setback.advance(rng, self._s_setback)
        self.seg_pre.advance(rng, self._s_pre)

        pol = self.policy
        if self.seg_setback.active:
            lo, hi, jit = self.seg_setback.payload
            pol.band_offset_low = np.maximum(lo + jit, 0.0)
            pol.band_offset_high = np.minimum(hi + jit, 0.0)
        else:
            pol.band_offset_low = pol.band_offset_high = None
        pre = self.seg_pre.payload if self.seg_pre.active else self.default_precondition
        pol.parameters["precondition_hours"] = float(pre)
        return self.seg_setback.active or pre != self.default_precondition

    def baseline_clone(self) -> ExplorationBaselinePolicy:
        pol = self.policy
        shared = {id(pol.metadata): pol.metadata, id(pol.action_spec): pol.action_spec}
        clone = copy.deepcopy(pol, shared)
        clone.band_offset_low = clone.band_offset_high = None
        clone.parameters["precondition_hours"] = self.default_precondition
        return clone

    def apply_action_knobs(self, action: Mapping[str, np.ndarray], obs: np.ndarray):
        """Return (executed_action, knob_row). `action` is the policy output."""
        cont = np.asarray(action["continuous_action"], dtype=np.float64).copy()
        disc = np.asarray(action["discrete_action"], dtype=np.int64).copy()
        o = np.asarray(obs, dtype=np.float64)
        occupied = o[self.i_now] > 0.5 or o[self.i_soon] > 0.5
        temp = self.obs_codec.zone_temperatures(o)
        heat_sp, cool_sp = self.obs_codec.zone_bands(o)
        dev = np.maximum(heat_sp - temp, temp - cool_sp)
        cold = heat_sp - temp
        K = self.K
        force = np.full(K, -1.0)
        sat_off = np.zeros(K)
        p_scale = np.ones(K)
        guard = np.zeros(K)
        for k in range(K):
            blocked = occupied and self.ahu_zones[k].size and np.max(dev[self.ahu_zones[k]]) > self.cfg.guard_k
            if blocked and (self.seg_ahu[k].active or self.seg_sat[k].active or self.seg_p[k].active):
                guard[k] = 1.0
            if blocked:
                continue
            run = int(disc[self.ahu_run[k]])
            sat = float(self.act_codec.to_native(cont[self.ahu_sat[k]], self.ahu_sat[k]))
            pres = float(self.act_codec.to_native(cont[self.ahu_p[k]], self.ahu_p[k]))
            sat_changed = pres_changed = False
            if self.seg_ahu[k].active:
                want, on_pressure = self.seg_ahu[k].payload
                if want == 0 and not occupied and run == 1:
                    run, force[k] = 0, 0.0
                elif want == 1 and run == 0:
                    run, force[k] = 1, 1.0
                    pres, sat = on_pressure, self.cfg.ahu_on_sat_k
                    sat_changed = pres_changed = True
            if run == 1 and self.seg_sat[k].active:
                sat_off[k] = self.seg_sat[k].payload
                sat += sat_off[k]
                sat_changed = True
            if run == 1 and self.seg_p[k].active:
                p_scale[k] = self.seg_p[k].payload
                pres *= p_scale[k]
                pres_changed = True
            disc[self.ahu_run[k]] = run
            # Only re-encode values that were changed, so an inactive knob
            # leaves the baseline's float32 action bit-identical.
            if sat_changed:
                cont[self.ahu_sat[k]] = self.act_codec.to_normalized(np.clip(sat, *SAT_RANGE), self.ahu_sat[k])
            if pres_changed:
                cont[self.ahu_p[k]] = self.act_codec.to_normalized(np.clip(pres, *PRESSURE_RANGE), self.ahu_p[k])

        hws_off, guard_hws = 0.0, 0.0
        if self.seg_hws.active:
            hws_off = float(self.seg_hws.payload)
            if hws_off < 0.0 and occupied and np.max(cold) > self.cfg.guard_k:
                hws_off, guard_hws = 0.0, 1.0
            if hws_off != 0.0:
                sw = float(self.act_codec.to_native(cont[self.hws_sw], self.hws_sw)) + hws_off
                cont[self.hws_sw] = self.act_codec.to_normalized(np.clip(sw, *HWS_RANGE), self.hws_sw)

        pol = self.policy
        if pol.band_offset_low is not None:
            lo, hi, _ = self.seg_setback.payload
            setback = (max(lo, 0.0), min(hi, 0.0))
        else:
            setback = (0.0, 0.0)
        row = np.concatenate([force, sat_off, p_scale,
                              [hws_off, setback[0], setback[1], pol.parameters["precondition_hours"]],
                              guard, [guard_hws]])
        executed = {"continuous_action": np.clip(cont, -1.0, 1.0).astype(np.float32),
                    "discrete_action": np.clip(disc, 0, 1).astype(np.int32)}
        return executed, row.astype(np.float32)


def resolve_profile(name: str) -> ExplorationConfig | None:
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
    return PROFILES[name]


__all__ = ["ExplorationConfig", "PROFILES", "ExplorationBaselinePolicy", "Explorer",
           "episode_rng", "resolve_profile", "KNOB_COLUMNS_DOC"]
