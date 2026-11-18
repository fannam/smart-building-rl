"""Intervention definitions and stratified branch-point selection (Phase 2B Workstream A).

An intervention perturbs ONE action family of the nominal (closed-loop baseline) action
sequence for `duration` steps from the branch point t_b, then reverts to nominal:
    kind 'add': native command + value, clipped to the env's native bounds
    kind 'set': discrete run command forced to value
Families are applied to every member (all VAVs / all AHUs), so per-zone responses are
paired samples of the same building-wide intervention. Commanded and effective changes
are both recorded (actuator floors/caps can make a command change a no-op).
Unchanged entries keep their exact normalized float32 values (no re-encoding round trip).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

# family -> (metadata group, key, unit label)
FAMILIES = {
    "reheat_valve": ("zones", "act_reheat_valve", "fraction"),
    "damper": ("zones", "act_damper", "fraction"),
    "sat": ("ahus", "act_sat", "K"),
    "static_pressure": ("ahus", "act_static_pressure", "Pa"),
    "recirculation": ("ahus", "act_recirculation", "fraction"),
    "hws_setpoint": ("hws", "act_supply_water_temp", "K"),
    "hws_dp": ("hws", "act_differential_pressure", "psi-like native"),
    "ahu_run": ("ahus", "act_run", "binary"),
    "hws_run": ("hws", "act_run", "binary"),
}
MAGNITUDES = {  # (small, large) native deltas
    "reheat_valve": (0.25, 0.6), "damper": (0.2, 0.5), "sat": (2.0, 5.0), "static_pressure": (150.0, 500.0),
    "recirculation": (0.15, 0.3), "hws_setpoint": (5.0, 15.0), "hws_dp": (2.0, 6.0),
}


@dataclass(frozen=True)
class Intervention:
    name: str
    family: str
    kind: str          # 'add' | 'set'
    value: float
    duration: int      # steps from t_b (H = sustained)
    magnitude: str     # 'small' | 'large' | 'binary'

    def to_dict(self):
        return asdict(self)


def default_interventions(H: int, pulse: int = 6) -> list[Intervention]:
    out = []
    for fam, (s, l) in MAGNITUDES.items():
        for mag, v in (("small", s), ("large", l)):
            for sign in (+1, -1):
                out.append(Intervention(f"{fam}{'+' if sign > 0 else '-'}{v:g}_D{H}", fam, "add", sign * v, H, mag))
        for sign in (+1, -1):
            out.append(Intervention(f"{fam}{'+' if sign > 0 else '-'}{l:g}_D{pulse}", fam, "add", sign * l, pulse, "large"))
    for fam in ("ahu_run", "hws_run"):
        for v in (1, 0):
            out.append(Intervention(f"{fam}={v}_D{H}", fam, "set", float(v), H, "binary"))
    return out


def indices(meta: Mapping[str, Any], family: str) -> np.ndarray:
    grp, key, _ = FAMILIES[family]
    return np.atleast_1d(np.asarray(meta[grp][key], dtype=np.int64))


def apply_intervention(iv: Intervention, cont: np.ndarray, disc: np.ndarray, meta: Mapping[str, Any]):
    """cont [H, n_c] normalized float32, disc [H, n_d]. Returns perturbed copies."""
    from building_meta import ActionCodec
    c, d = np.array(cont, dtype=np.float32, copy=True), np.array(disc, copy=True)
    idx = indices(meta, iv.family)
    D = min(iv.duration, len(c))
    if iv.kind == "set":
        d[:D, idx] = int(iv.value)
        return c, d
    codec = ActionCodec(meta)
    lo = np.asarray(meta["continuous_action_native_low"], np.float64)[idx]
    hi = np.asarray(meta["continuous_action_native_high"], np.float64)[idx]
    nat = codec.to_native(c[:D, idx].astype(np.float64), idx)
    new = np.clip(nat + iv.value, lo, hi)
    changed = np.abs(new - nat) > 1e-9
    enc = codec.to_normalized(new, idx).astype(np.float32)
    block = c[:D, idx]
    block[changed] = enc[changed]
    c[:D, idx] = block
    return c, d


# ------------------------------------------------------------------ branch points
STRATA = ("morning_recovery", "occupied_heating", "occupied_cooling", "occupied_other", "unoccupied_night",
          "unoccupied_other")


def step_conditions(ep: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Per-step operating-condition labels for transition t (privileged allowed: offline stratification only)."""
    from building_meta import ObservationCodec
    oc = ObservationCodec(meta)
    nat = oc.native(np.asarray(ep["obs"]))
    T = len(ep["reward"])
    oat = nat[:T, meta["ahus"]["obs_outside_air_temp"][0]] - 273.15
    cal = np.asarray(ep["calendar"])
    hour, wd = cal[:, 0] + cal[:, 1] / 60.0, cal[:, 2]
    sch = meta["schedule"]
    workday = wd < 5
    occupied = workday & (hour >= sch["morning_start_hour"]) & (hour < sch["evening_start_hour"])
    recovery = workday & (hour >= sch["morning_start_hour"] - 3) & (hour < sch["morning_start_hour"])
    gas = np.asarray(ep["boiler_info"])[..., 0].sum(-1) > 0
    cool = np.abs(np.asarray(ep["ahu_info"])[..., 1]).sum(-1) > 0
    ahu_on = np.asarray(ep["action_discrete"])[:, meta["ahus"]["act_run"]].any(-1)
    night = (hour < 5) | (hour >= 21)
    stratum = np.full(T, "unoccupied_other", dtype=object)
    stratum[~occupied & night] = "unoccupied_night"
    stratum[occupied] = "occupied_other"
    stratum[occupied & cool & ~gas] = "occupied_cooling"
    stratum[occupied & gas] = "occupied_heating"
    stratum[recovery] = "morning_recovery"
    band = np.where(oat < -10, "cold", np.where(oat < 10, "cool", np.where(oat < 25, "mild", "warm")))
    return {"stratum": stratum, "oat_c": oat, "oat_band": band, "heating": gas, "cooling": cool, "ahu_on": ahu_on,
            "occupied": occupied, "hour": hour}


def select_branch_points(cond: Mapping[str, np.ndarray], n_per_stratum: int, C: int, H: int, seed: int,
                         spacing: int | None = None, max_t: int | None = None) -> list[tuple[int, str]]:
    """Stratified random t_b (deterministic for a seed); points are >= spacing apart (the main
    trajectory must advance H steps past each branch point)."""
    rng = np.random.default_rng(seed)
    spacing = spacing or H + 1
    T = len(cond["stratum"])
    hi = (max_t or T) - H - 1
    chosen: list[tuple[int, str]] = []
    for s in STRATA:
        cand = np.flatnonzero(cond["stratum"][:hi] == s)
        cand = cand[cand >= C + 1]
        rng.shuffle(cand)
        got = 0
        for t in cand:
            if all(abs(int(t) - u) >= spacing for u, _ in chosen):
                chosen.append((int(t), s)); got += 1
                if got >= n_per_stratum:
                    break
    return sorted(chosen)
