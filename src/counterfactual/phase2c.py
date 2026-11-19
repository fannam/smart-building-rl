"""Phase 2C counterfactual collection: E9 paired branches carrying E1/E2 zone-subset excitation.

Reuses the Phase 2B infrastructure unchanged:
    counterfactual.branch      bit-exact Branch.take/spawn (env copy + global RNG snapshot/restore)
    counterfactual.simrun      env setup, baseline policy, record_step
    counterfactual.interventions  building-wide `add` interventions (apply_intervention), step_conditions
    effective_action           commanded -> effective actions (floors, caps, AHU-off)

New here:
    * zone-SUBSET interventions (random 30 % of zones, optionally only zones whose nominal effective value
      qualifies), 'set' to a level for a hold of 3/6/12 steps; run-bit flips; random-magnitude adds
    * weighted, spaced branch-point selection that excludes the Phase 2B branch points (Test-A)
    * per-branch-point output files (atomic write) -> resumable across sessions without duplicates
    * per-arm metadata: subset mask, clipped, saturated, effective no-op, extreme excursion, aborted

Arm randomness uses its own numpy Generator seeded by crc32(seed|scenario|t_b|arm); it never touches the
global RNG streams that the simulator uses, so branching stays bit-exact.
"""
from __future__ import annotations

import json
import math
import os
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from counterfactual.interventions import FAMILIES, STRATA, Intervention, apply_intervention, indices

ZONE_FAMILIES = ("reheat_valve", "damper")
DISCRETE_FAMILIES = ("ahu_run", "hws_run")
SCHEMA_VERSION = "phase2c-cf-v1"


# ------------------------------------------------------------------ arm specs
@dataclass(frozen=True)
class ArmSpec:
    name: str
    group: str
    family: str
    kind: str
    duration: int
    level: float | None = None
    levels: tuple = ()
    value: float | None = None
    low: float | None = None
    high: float | None = None
    sign: int | None = None
    qualify_above: float | None = None
    qualify_below: float | None = None

    def to_dict(self):
        d = asdict(self)
        d["levels"] = list(self.levels)
        return d


def expand_arms(cfg: Mapping[str, Any]) -> list[ArmSpec]:
    """Config 'arms' -> concrete arm list (nominal excluded). A spec with `durations` expands to one
    arm per duration; a spec with `signs` expands to one arm per sign."""
    out: list[ArmSpec] = []
    for s in cfg["arms"]:
        base = {k: s[k] for k in ("group", "family", "kind")}
        durs = s.get("durations", [s.get("duration")])
        signs = s.get("signs", [None])
        for d in durs:
            for sg in signs:
                i = sum(1 for a in out if a.family == s["family"] and a.kind == s["kind"])
                name = f"{s['group']}:{s['family']}:{s['kind']}:{i}"
                out.append(ArmSpec(name=name, duration=int(d), sign=sg, level=s.get("level"), levels=tuple(s.get("levels", ())),
                                   value=s.get("value"), low=s.get("low"), high=s.get("high"),
                                   qualify_above=s.get("qualify_above"), qualify_below=s.get("qualify_below"), **base))
    return out


def arm_rng(seed: int, scenario: str, t_b: int, arm_index: int) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{seed}|{scenario}|{t_b}|{arm_index}".encode()))


def _member_native(meta, family, cont):
    from building_meta import ActionCodec
    idx = indices(meta, family)
    return ActionCodec(meta).to_native(np.asarray(cont, np.float64)[..., idx], idx), idx


def zones_with_ahu_on(meta: Mapping[str, Any], disc_row: np.ndarray) -> np.ndarray:
    """Bool [V]: the zone's AHU run bit is on in this discrete action row."""
    run = np.asarray(disc_row)[np.asarray(meta["ahus"]["act_run"])].astype(bool)
    return run[np.asarray(meta["zones"]["ahu"], dtype=np.int64)]


def apply_arm(spec: ArmSpec, cont: np.ndarray, disc: np.ndarray, meta: Mapping[str, Any], rep, rng: np.random.Generator,
              subset_fraction: float, owner_on_only: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (perturbed continuous [H,n_c] float32, perturbed discrete [H,n_d], info).
    info: subset_mask [V] (zone families), requested native values, clipped flag, no_qualifying flag."""
    from building_meta import ActionCodec
    c = np.array(cont, dtype=np.float32, copy=True)
    d = np.array(disc, copy=True)
    V = int(meta["num_zones"])
    D = min(spec.duration, len(c))
    info = {"subset_mask": np.zeros(V, bool), "clipped": False, "no_qualifying": False, "value": np.nan}
    fam = spec.family
    if spec.kind in ("subset_set", "subset_set_qualified", "subset_set_qualified_below"):
        assert fam in ZONE_FAMILIES, spec
        nat0, idx = _member_native(meta, fam, c[0])
        eff0 = rep.effective(c[:1], d[:1]).effective_native[0][idx]            # effective nominal value at t_b
        if spec.kind == "subset_set":
            cand = np.arange(V)
        elif spec.kind == "subset_set_qualified":
            cand = np.flatnonzero(eff0 > spec.qualify_above)
        else:
            cand = np.flatnonzero(eff0 < spec.qualify_below)
        if owner_on_only:                       # dampers/valves act only while their AHU moves air
            cand = cand[zones_with_ahu_on(meta, d[0])[cand]]
        if cand.size == 0:
            info["no_qualifying"] = True
            return c, d, info
        k = max(1, int(round(subset_fraction * cand.size)))
        members = np.sort(rng.choice(cand, size=k, replace=False))
        level = float(spec.level if spec.level is not None else rng.choice(np.asarray(spec.levels, float)))
        info["value"] = level
        lo = np.asarray(meta["continuous_action_native_low"], np.float64)[idx[members]]
        hi = np.asarray(meta["continuous_action_native_high"], np.float64)[idx[members]]
        target = np.clip(level, lo, hi)
        info["clipped"] = bool(np.any(target != level))
        codec = ActionCodec(meta)
        cols = idx[members]
        nat = codec.to_native(c[:D][:, cols].astype(np.float64), cols)
        new = np.broadcast_to(target, nat.shape)
        changed = np.abs(new - nat) > 1e-9
        if spec.kind == "subset_set_qualified_below":
            # no-op control: override only (step, zone) entries whose NOMINAL effective value is below the
            # threshold at that step (e.g. damper at its 0.1 floor), so the effective action never changes
            eff_hold = rep.effective(c[:D], d[:D]).effective_native[:, cols]
            changed &= eff_hold < spec.qualify_below
        enc = codec.to_normalized(np.array(new), cols).astype(np.float32)
        block = c[:D][:, cols]
        block[changed] = enc[changed]
        c[:D, cols] = block
        info["subset_mask"][members] = True
        return c, d, info
    if spec.kind == "flip":
        assert fam in DISCRETE_FAMILIES, spec
        idx = indices(meta, fam)
        d[:D, idx] = 1 - d[0, idx]
        info["value"] = float(np.mean(d[0, idx]))              # fraction of members ON after the flip
        return c, d, info
    # building-wide continuous adds (reuse the Phase 2B implementation)
    if spec.kind == "add":
        val = float(spec.value)
    elif spec.kind == "add_random_magnitude":
        val = float(spec.sign) * float(rng.uniform(spec.low, spec.high))
    elif spec.kind == "add_random_sign":
        val = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(spec.low, spec.high))
    else:
        raise ValueError(f"unknown arm kind {spec.kind}")
    info["value"] = val
    nat, idx = _member_native(meta, fam, c[:D])
    lo = np.asarray(meta["continuous_action_native_low"], np.float64)[idx]
    hi = np.asarray(meta["continuous_action_native_high"], np.float64)[idx]
    info["clipped"] = bool(np.any((nat + val < lo - 1e-9) | (nat + val > hi + 1e-9)))
    if fam in ZONE_FAMILIES:
        info["subset_mask"][:] = True
    c2, d2 = apply_intervention(Intervention(spec.name, fam, "add", val, D, "phase2c"), c, d, meta)
    return c2, d2, info


# ------------------------------------------------------------------ branch-point plan
def _crc(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode())


def plan_branch_points(cond: Mapping[str, np.ndarray], n: int, weights: Mapping[str, float], C: int, H: int,
                       spacing: int, exclude: list[int], exclude_radius: int, seed: int, scenario: str,
                       prefer_ahu_on: bool = False) -> list[tuple[int, str]]:
    """Weighted stratified selection of n spaced branch points (deterministic per seed+scenario).
    Counts per stratum ~ weight (largest remainder among strata that have candidates); shortfalls are
    filled from the remaining strata in weight order. Points within `exclude_radius` of any excluded
    t_b (Phase 2B Test-A) are never chosen."""
    rng = np.random.default_rng(_crc(seed, scenario, "plan"))
    T = len(cond["stratum"])
    hi = T - H - 1
    ok = np.zeros(T, bool)
    ok[C + 1:hi] = True
    for u in exclude:
        ok[max(0, u - exclude_radius):u + exclude_radius + 1] = False
    cands = {}
    for s in STRATA:
        c = np.flatnonzero(ok & (cond["stratum"] == s))
        rng.shuffle(c)
        if prefer_ahu_on:                        # AHU-on steps first (stable within the shuffled order)
            c = c[np.argsort(~np.asarray(cond["ahu_on"])[c], kind="stable")]
        if c.size:
            cands[s] = c
    present = [s for s in STRATA if s in cands and weights.get(s, 0) > 0]
    if not present:
        return []
    w = np.array([weights[s] for s in present], float)
    raw = n * w / w.sum()
    alloc = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - alloc))[: n - alloc.sum()]:
        alloc[i] += 1
    chosen: list[tuple[int, str]] = []

    def take(s, k):
        got = 0
        for t in cands[s]:
            if got >= k:
                break
            if all(abs(int(t) - u) >= spacing for u, _ in chosen):
                chosen.append((int(t), s)); got += 1
        return got
    short = 0
    for s, k in zip(present, alloc):
        short += k - take(s, int(k))
    for s in sorted(present, key=lambda x: -weights[x]):
        if short <= 0:
            break
        short -= take(s, short)
    return sorted(chosen)


def branch_id(scenario: str, t_b: int) -> str:
    return f"{scenario.replace('/', '__')}__t{t_b:05d}"


# ------------------------------------------------------------------ per-branch collection
def branch_file(out_dir: Path, scenario: str, t_b: int) -> Path:
    return Path(out_dir) / "branches" / f"{branch_id(scenario, t_b)}.npz"


def branch_complete(path: Path, n_arms: int) -> bool:
    try:
        with np.load(path, allow_pickle=False) as z:
            return int(z["complete"]) == 1 and z["arm_ac"].shape[1] == n_arms and str(z["schema_version"]) == SCHEMA_VERSION
    except Exception:  # noqa: BLE001  (missing / truncated / corrupt -> redo)
        return False


def _atomic_savez(path: Path, arrays: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def collect_scenario(scenario: str, cfg: Mapping[str, Any], out_dir: str, data_root: str, deadline: float,
                     exclude: list[int], log=print) -> dict:
    """Collect all planned branch points of one scenario that are not yet complete. Resumable."""
    from counterfactual.branch import Branch, RNGState
    from counterfactual.simrun import make_action, max_divergence, policy_action, record_step, setup, stack
    from dataset import load_episode, load_scenario_meta
    from effective_action import ActionRepresentation
    t_start = time.time()
    out = Path(out_dir).resolve()
    root = Path(data_root).resolve()
    C, H, seed = int(cfg["context"]), int(cfg["horizon"]), int(cfg["seed"])
    arms = expand_arms(cfg)
    A = len(arms) + 1
    ep_id = f"{scenario.replace('/', '__')}__baseline__s0__e0"
    ref = load_episode(root / "episodes" / f"{ep_id}.npz")
    meta = load_scenario_meta(root, scenario)
    from counterfactual.interventions import step_conditions
    cond = step_conditions(ref, meta)
    plan = plan_branch_points(cond, int(cfg["n_points_per_scenario"]), cfg["stratum_weights"], C, H,
                              int(cfg["min_spacing_steps"]), exclude, int(cfg["exclude_near_radius_steps"]), seed, scenario,
                              bool(cfg.get("prefer_ahu_on_within_stratum", False)))
    pending = [(t, s) for t, s in plan if not branch_complete(branch_file(out, scenario, t), A)]
    res = {"scenario": scenario, "planned": len(plan), "already_complete": len(plan) - len(pending), "collected": 0,
           "stopped_by_deadline": False, "errors": []}
    if not pending:
        res["wall_s"] = time.time() - t_start
        return res
    env, smeta, policy, spec, obs = setup(scenario)
    assert smeta["observation_field_names"] == meta["observation_field_names"]
    rep = ActionRepresentation(meta)
    rooms = smeta["zones"]["room_ids"]
    V = int(meta["num_zones"])
    tank_lo, tank_hi = cfg["quality_guards"]["tank_valid_k"]
    exc_k = float(cfg["quality_guards"]["extreme_excursion_k"])
    frac = float(cfg["zone_subset_fraction"])
    obs_hist, ac_hist, ad_hist = [obs.astype(np.float32)], [], []
    t = 0
    last = pending[-1][0]
    for tb, stratum in pending:
        if time.time() > deadline:
            res["stopped_by_deadline"] = True
            break
        while t < tb:
            a = policy_action(policy, obs, spec)
            r = record_step(env, a, rooms)
            obs = r["obs"]; t += 1
            obs_hist.append(obs.astype(np.float32)); ac_hist.append(a["continuous_action"].copy()); ad_hist.append(a["discrete_action"].copy())
        tp = time.time()
        br = Branch.take(env)
        main = []
        for _ in range(H):
            a = policy_action(policy, obs, spec)
            r = record_step(env, a, rooms)
            main.append(r); obs = r["obs"]; t += 1
            obs_hist.append(obs.astype(np.float32)); ac_hist.append(a["continuous_action"].copy()); ad_hist.append(a["discrete_action"].copy())
        rng_after = RNGState.capture()
        nom_c = np.asarray(ac_hist[tb:tb + H], np.float32)
        nom_d = np.asarray(ad_hist[tb:tb + H])
        arm_c, arm_d, infos = [nom_c], [nom_d], [{"subset_mask": np.zeros(V, bool), "clipped": False, "no_qualifying": False,
                                                  "value": np.nan}]
        for i, sp in enumerate(arms):
            c2, d2, inf = apply_arm(sp, nom_c, nom_d, meta, rep, arm_rng(seed, scenario, tb, i + 1), frac,
                                    bool(cfg.get("zone_arms_owner_on_only", False)))
            arm_c.append(c2); arm_d.append(d2); infos.append(inf)
        outs, aborted, simulated = [], np.zeros(A, bool), np.ones(A, bool)
        for i, (c2, d2) in enumerate(zip(arm_c, arm_d)):
            if i > 0 and np.array_equal(c2, nom_c) and np.array_equal(d2, nom_d):
                # identical commands + common random numbers => bit-identical to the nominal arm (verified by
                # tests/test_phase2c.py::test_identical_commands_reproduce_nominal); not re-simulated
                outs.append({k: np.array(v, copy=True) for k, v in outs[0].items()})
                simulated[i] = False
                continue
            e = br.spawn()
            try:
                recs = [record_step(e, make_action(c2[k], d2[k], spec), rooms) for k in range(H)]
                st = stack(recs)
                bad = (not np.isfinite(st["zone_info"]).all()) or (not np.isfinite(st["tank"]).all()) or \
                      st["tank"][:, 0].min() < tank_lo or st["tank"][:, 0].max() > tank_hi
            except Exception as ex:  # noqa: BLE001
                res["errors"].append(f"{branch_id(scenario, tb)} arm {i}: {type(ex).__name__}: {ex}")
                st, bad = None, True
            if bad:
                if i == 0:
                    raise RuntimeError(f"nominal arm invalid at {branch_id(scenario, tb)} (simulator state out of range)")
                aborted[i] = True                                   # kept, NaN-filled, flagged
                st = {k: np.full_like(np.asarray(v, np.float64), np.nan) for k, v in outs[0].items()}
            outs.append(st)
            del e
        del br
        rng_after.restore()
        div = max_divergence(outs[0], stack(main))
        effs = [rep.effective(c2, d2) for c2, d2 in zip(arm_c, arm_d)]
        e0 = effs[0]
        eff_delta = np.stack([np.concatenate([e.effective_native - e0.effective_native,
                                              (e.discrete - e0.discrete).astype(np.float64)], -1) for e in effs]).astype(np.float32)
        cmd_delta = np.stack([np.concatenate([e.commanded_native - e0.commanded_native,
                                              (e.discrete - e0.discrete).astype(np.float64)], -1) for e in effs]).astype(np.float32)
        cmd_changed = np.abs(cmd_delta) > 1e-6
        eff_changed = np.abs(eff_delta) > 1e-6
        saturated = (cmd_changed & ~eff_changed).any((1, 2))
        act = np.stack([np.concatenate([e.active, np.ones_like(e.discrete, bool)], -1) for e in effs])
        active_change = (eff_changed & act).any((1, 2))      # effective change while the owning AHU runs (or plant/run bits)
        noop = ~eff_changed.any((1, 2))
        zt = np.stack([o["zone_info"][..., 2] for o in outs])                          # [A,H,V]
        sp_ = np.stack([o["zone_info"][..., :2] for o in outs])
        occ = np.stack([o["zone_info"][..., 5] for o in outs])
        dev = np.maximum(0, np.maximum(zt - sp_[..., 1], sp_[..., 0] - zt))
        extreme = np.nan_to_num(np.where(occ > 0, dev, 0.0), nan=0.0).max((1, 2)) > exc_k
        zidx = np.asarray(meta["zones"]["act_damper"]), np.asarray(meta["zones"]["act_reheat_valve"])
        zone_eff_changed = np.stack([eff_changed[:, :, zidx[0]].any(1), eff_changed[:, :, zidx[1]].any(1)], -1)   # [A,V,2]
        e2 = bool(cond["heating"][tb]) and float(ref["tank_state"][tb, 0]) >= cfg["e2_eligibility"]["tank_min_k"] and bool(cond["ahu_on"][tb])
        arrays = {
            "schema_version": np.array(SCHEMA_VERSION), "complete": np.array(1), "scenario": np.array(scenario),
            "building": np.array(scenario.split("/")[0]), "episode_id": np.array(ep_id),
            "branch_id": np.array([branch_id(scenario, tb)]), "t_b": np.array([tb]),
            "sim_timestamp": np.array([int(ref["sim_timestamp"][tb])]), "calendar_tb": np.asarray(ref["calendar"][tb])[None],
            "stratum": np.array([stratum]), "oat_c": np.array([cond["oat_c"][tb]]), "oat_band": np.array([str(cond["oat_band"][tb])]),
            "heating": np.array([cond["heating"][tb]]), "cooling": np.array([cond["cooling"][tb]]), "ahu_on": np.array([cond["ahu_on"][tb]]),
            "occupied": np.array([cond["occupied"][tb]]), "hour": np.array([cond["hour"][tb]]), "e2_eligible": np.array([e2]),
            "ctx_obs": np.asarray(obs_hist[tb - C + 1:tb + 1], np.float32)[None],
            "ctx_ac": np.asarray(ac_hist[tb - C:tb], np.float32)[None], "ctx_ad": np.asarray(ad_hist[tb - C:tb])[None],
            "arm_ac": np.stack(arm_c)[None], "arm_ad": np.stack(arm_d)[None],
            "arm_eff": np.stack([e.effective_native for e in effs]).astype(np.float32)[None],
            "eff_delta": eff_delta[None], "cmd_delta": cmd_delta[None],
            "zone_temp": zt.astype(np.float32)[None], "zone_occ": occ.astype(np.float32)[None], "zone_sp": sp_.astype(np.float32)[None],
            "tank": np.stack([o["tank"][:, 0] for o in outs]).astype(np.float32)[None],
            "energy": np.stack([np.stack([o["ahu_info"][..., 0].sum(-1), np.abs(o["ahu_info"][..., 1]).sum(-1),
                                          o["boiler_info"][..., 1].sum(-1), o["boiler_info"][..., 0].sum(-1)], -1)
                                for o in outs]).astype(np.float32)[None],
            "reward": np.stack([o["reward"] for o in outs])[None], "regret": np.stack([o["productivity_regret"] for o in outs])[None],
            "obs": np.stack([o["obs"] for o in outs]).astype(np.float32)[None],
            "oat_future_obs": np.asarray(ref["obs"][tb:tb + H], np.float32)[None],
            "arm_names": np.array(["nominal"] + [a.name for a in arms]),
            "arm_group": np.array(["NOMINAL"] + [a.group for a in arms]),
            "arm_family": np.array(["nominal"] + [a.family for a in arms]),
            "arm_kind": np.array(["nominal"] + [a.kind for a in arms]),
            "arm_duration": np.array([H] + [a.duration for a in arms]),
            "arm_value": np.array([i["value"] for i in infos], np.float64)[None],
            "subset_mask": np.stack([i["subset_mask"] for i in infos])[None],
            "zone_eff_changed": zone_eff_changed[None],
            "clipped": np.array([i["clipped"] for i in infos])[None], "no_qualifying": np.array([i["no_qualifying"] for i in infos])[None],
            "saturated": saturated[None], "effective_noop": noop[None], "active_change": active_change[None],
            "simulated": simulated[None], "extreme_excursion": extreme[None], "aborted": aborted[None],
            "nominal_bit_exact": np.array([div["bit_exact"]]), "nominal_div_json": np.array(json.dumps(div)),
            "interventions_json": np.array(json.dumps([{**a.to_dict(), "magnitude": a.kind, "value": None} for a in arms])),
            "config_json": np.array(json.dumps(cfg)), "H": np.array(H), "C": np.array(C),
            "branch_wall_s": np.array(time.time() - tp),
        }
        _atomic_savez(branch_file(out, scenario, tb), arrays)
        res["collected"] += 1
        log(f"[{scenario}] t_b={tb} ({stratum}) {A} arms, nominal exact={div['bit_exact']}, noop={int(noop[1:].sum())}, "
            f"aborted={int(aborted.sum())}, {time.time() - tp:.0f}s  ({res['already_complete'] + res['collected']}/{len(plan)})")
        if tb == last:
            break
    n = len(obs_hist) - 1
    res["replay_obs_max_abs"] = float(np.max(np.abs(np.asarray(obs_hist) - np.asarray(ref["obs"][:n + 1], np.float32)))) if n else 0.0
    res["wall_s"] = time.time() - t_start
    return res


# ------------------------------------------------------------------ plan / validation helpers
def all_scenarios(data_root: Path) -> list[str]:
    from dataset import load_manifest
    return sorted({r["scenario"] for r in load_manifest(Path(data_root)) if r["profile"] == "baseline"})


def validate_config(cfg: Mapping[str, Any]) -> list[str]:
    """Returns a list of problems (empty = valid)."""
    errs = []
    for k in ("seed", "context", "horizon", "n_points_per_scenario", "min_spacing_steps", "exclude_near_radius_steps",
              "stratum_weights", "zone_subset_fraction", "arms", "quality_guards", "e2_eligibility"):
        if k not in cfg:
            errs.append(f"missing key {k}")
    if errs:
        return errs
    if cfg["min_spacing_steps"] < cfg["context"] + cfg["horizon"] + 1:
        errs.append("min_spacing_steps must be >= context + horizon + 1 (no overlapping sibling segments)")
    if not 0 < cfg["zone_subset_fraction"] <= 0.3 + 1e-9:
        errs.append("zone_subset_fraction must be in (0, 0.3]")
    if set(cfg["stratum_weights"]) - set(STRATA):
        errs.append(f"unknown strata {set(cfg['stratum_weights']) - set(STRATA)}")
    try:
        arms = expand_arms(cfg)
    except Exception as ex:  # noqa: BLE001
        return errs + [f"arm expansion failed: {ex}"]
    for a in arms:
        if a.family not in FAMILIES:
            errs.append(f"{a.name}: unknown family")
        if a.kind.startswith("subset") and a.family not in ZONE_FAMILIES:
            errs.append(f"{a.name}: subset kinds only for zone families")
        if a.kind == "flip" and a.family not in DISCRETE_FAMILIES:
            errs.append(f"{a.name}: flip only for run bits")
        if a.duration < 1 or a.duration > cfg["horizon"]:
            errs.append(f"{a.name}: duration {a.duration} outside [1, H]")
    if len(arms) + 1 != cfg.get("arms_per_branch", len(arms) + 1):
        errs.append(f"arms_per_branch {cfg.get('arms_per_branch')} != expanded {len(arms) + 1}")
    return errs
