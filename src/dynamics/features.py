"""Window featurization for the dynamics models (inputs and targets kept separate).

Step indexing for a window from `AlignedWindows(WindowSpec(horizon=H, context=C))`:

    step j = 0 .. S-1, S = C-1+H,  absolute step s = t0 - C + 1 + j
    warm-up steps j < C-1   (s < t0)        teacher-forced: true observation o_s
    decision step  j = C-1  (s = t0, k = 0) true observation o_{t0}
    rollout steps  j >= C   (s = t0 + k)    NO observation: inputs come from the model's own
                                            predictions + runtime-forecastable exogenous inputs
    action rows    A[i] = a[t0 - C + i], i = 0 .. C+H-1; step j uses a_{s-1} = A[j], a_s = A[j+1]
    targets        for rollout steps k = j-(C-1) = 0..H-1 -> transition t = t0 + k

What a step-s input means (configs/sensor_alignment.json):
    zone temperature     state after transition s-2   (obs[s] zone sensors; offset 2)
    HWS temperature      state after transition s-1   (obs[s] HWS sensor; offset 1)
    outside air          weather at sim_ts[s]          (current)
    clock / comfort mode at sim_ts[s] (and +60 min);  computed from o_{t0}'s clock (exact)
    setpoint window      at sim_ts[s+1] = the window behind reward[s] (schedule, exact)
Predictions of step s: zone temperature and HWS temperature after transition s, and the
energy of transition s (fan, cooling, pump electricity; gas burner heat), all in W.

`runtime_part()` reads only item["runtime_inputs"] and item["actions"];
`target_part()` reads only item["targets"]. Rollout steps never receive observations.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from dynamics.schedule import advance, clock_features, comfort_mode, decode_clock, temperature_window
from runtime_features import validate_runtime_features

ENERGY_NAMES = ("fan_electric_w", "cooling_electric_w", "pump_electric_w", "gas_burner_w")
# Physical, public scales for action features (action ranges of the env spec); not fitted.
SAT_C, SAT_S = 295.0, 10.0
P_S = 1000.0
HWS_C, HWS_S = 330.0, 20.0
DP_S = 10.0
SETPOINT_C, SETPOINT_S = 295.0, 5.0
# Per-AHU energy normalizers of the competition reward (public, global): W per AHU.
E_PER_AHU = 80_000.0
G_PER_AHU = 205_000.0


class WindowFeaturizer:
    """Per-scenario featurizer. Validates every runtime source it reads against the whitelist."""

    def __init__(self, meta: Mapping[str, Any]):
        self.meta = meta
        names = meta["observation_field_names"]
        self.fi = {n: i for i, n in enumerate(names)}
        z, ah, hw = meta["zones"], meta["ahus"], meta["hws"]
        self.K, self.V = int(meta["num_ahus"]), int(meta["num_zones"])
        self.zone_temp_idx = np.asarray(z["obs_temp"])
        self.hws_idx = int(hw["obs_supply_water_temp"])
        self.oat_idx = int(ah["obs_outside_air_temp"][0])
        self.mean = np.asarray(meta["observation_mean"], dtype=np.float64)
        self.std = np.asarray(meta["observation_std"], dtype=np.float64)
        self.zone_ahu = np.asarray(z["ahu"], dtype=np.int64)
        area = np.asarray(meta["zone_area_m2"], dtype=np.float64)
        ua = np.asarray(meta["zone_exterior_ua_proxy"], dtype=np.float64)
        self.zone_static = np.stack([(np.log(area) - 3.5) / 1.0, ua / 20.0], -1).astype(np.float32)
        self.act = {"damper": np.asarray(z["act_damper"]), "valve": np.asarray(z["act_reheat_valve"]),
                    "run": np.asarray(ah["act_run"]), "sat": np.asarray(ah["act_sat"]),
                    "p": np.asarray(ah["act_static_pressure"]), "rc": np.asarray(ah["act_recirculation"]),
                    "hws_run": int(hw["act_run"]), "hws_sp": int(hw["act_supply_water_temp"]),
                    "dp": int(hw["act_differential_pressure"])}
        self.schedule = meta["schedule"]
        self.features_used = (
            [f"obs:{names[i]}" for i in self.zone_temp_idx]
            + [f"obs:{names[self.hws_idx]}", f"obs:{names[self.oat_idx]}"]
            + [f"obs:{n}" for n in ("hod_sin_000", "hod_cos_000", "dow_sin_000", "dow_cos_000")]
            + ["effective_action", "action_discrete", "meta:zones", "meta:ahus", "meta:hws", "meta:schedule",
               "meta:zone_area_m2", "meta:zone_exterior_ua_proxy", "meta:observation_mean", "meta:observation_std",
               "meta:observation_field_names", "meta:num_ahus", "meta:num_zones"])
        validate_runtime_features(self.features_used, field_names=names)

    def _native(self, obs, idx):
        return np.asarray(obs, dtype=np.float64)[..., idx] * self.std[idx] + self.mean[idx]

    def _action_rows(self, eff_native: np.ndarray, disc: np.ndarray) -> dict[str, np.ndarray]:
        a = self.act
        zone = np.stack([eff_native[:, a["damper"]], eff_native[:, a["valve"]]], -1)            # [R,V,2]
        ahu = np.stack([disc[:, a["run"]].astype(np.float64), (eff_native[:, a["sat"]] - SAT_C) / SAT_S,
                        eff_native[:, a["p"]] / P_S, eff_native[:, a["rc"]]], -1)                 # [R,K,4]
        bldg = np.stack([(eff_native[:, a["hws_sp"]] - HWS_C) / HWS_S, eff_native[:, a["dp"]] / DP_S,
                         disc[:, a["hws_run"]].astype(np.float64)], -1)                           # [R,3]
        return {"zone_act": zone.astype(np.float32), "ahu_act": ahu.astype(np.float32), "bldg_act": bldg.astype(np.float32)}

    # ---------------------------------------------------------------- inputs (runtime only)
    def runtime_part(self, runtime_inputs: Mapping[str, np.ndarray], actions: Mapping[str, np.ndarray], H: int) -> dict:
        obs = np.asarray(runtime_inputs["obs"])                      # [C, D] = o_{t0-C+1 .. t0}
        C = obs.shape[0]
        eff = np.concatenate([runtime_inputs["effective_action_history"], actions["effective_native"]], 0)
        disc = np.concatenate([runtime_inputs["action_history_discrete"], actions["discrete"]], 0)
        mask = np.concatenate([runtime_inputs["action_history_mask"], np.ones(H, bool)])
        rows = self._action_rows(eff, disc)
        for k in rows:
            rows[k][~mask] = 0.0
        hour0, dow0 = decode_clock(obs[-1], self.fi)                 # clock of o_{t0}
        S = C - 1 + H
        rel = np.arange(S) - (C - 1)                                 # s - t0 for each step
        hour, dow = advance(hour0, dow0, rel)
        hour_n, dow_n = advance(hour0, dow0, rel + 1)                # sim_ts[s+1]
        hour_soon, dow_soon = advance(hour0, dow0, rel + 12)
        lo, hi = temperature_window(hour_n, dow_n, self.schedule)
        exo = np.concatenate([clock_features(hour, dow),
                              comfort_mode(hour, dow, self.schedule)[:, None].astype(np.float64),
                              comfort_mode(hour_soon, dow_soon, self.schedule)[:, None].astype(np.float64),
                              ((lo - SETPOINT_C) / SETPOINT_S)[:, None], ((hi - SETPOINT_C) / SETPOINT_S)[:, None]], -1)
        return {
            "zone_temp_obs": self._native(obs, self.zone_temp_idx).astype(np.float32),   # [C, V]  K
            "hws_obs": self._native(obs, self.hws_idx).astype(np.float32),               # [C]     K
            "oat_obs": self._native(obs, self.oat_idx).astype(np.float32),               # [C]     K
            "exo": exo.astype(np.float32),                                                # [S, 8]
            "hour_next": hour_n.astype(np.float32), "dow_next": dow_n.astype(np.float32),  # [S] clock of reward interval
            "setpoint_next": np.stack([lo, hi], -1).astype(np.float32),                   # [S, 2] K
            "action_mask": mask, **rows,                                                  # [C+H, ...]
            "zone_static": self.zone_static, "zone_ahu": self.zone_ahu,
        }

    # ---------------------------------------------------------------- targets (privileged allowed)
    def target_part(self, targets: Mapping[str, Any]) -> dict:
        p = targets["privileged"]
        zi = np.asarray(p["zone_info"], dtype=np.float64)          # [H, V, 6] zone_info[t]
        a, b = np.asarray(p["ahu_info"], np.float64), np.asarray(p["boiler_info"], np.float64)
        energy = np.stack([a[..., 0].sum(-1), np.abs(a[..., 1]).sum(-1), b[..., 1].sum(-1), b[..., 0].sum(-1)], -1)
        return {
            "zone_temp": zi[..., 2].astype(np.float32),              # after transition t
            "zone_occ": zi[..., 5].astype(np.float32),
            "zone_setpoints": zi[..., :2].astype(np.float32),
            "hws": np.asarray(p["tank_state"], np.float64)[:, 0].astype(np.float32),   # tank_state[t+1]
            "energy": energy.astype(np.float32),                      # [H, 4] W
            "reward": np.asarray(targets["reward"], np.float64),
            "calendar": np.asarray(p["calendar"], np.int64),
            # obs_window[k] = o_{t0+k}: ONLY for the labelled 'oracle exogenous' diagnostic (outside air)
            "oat_oracle": self._native(targets["obs_window"], self.oat_idx).astype(np.float32),
        }


PRIVILEGED_FOR_TRAINING = ("zone_info", "ahu_info", "boiler_info", "tank_state", "calendar")


class DynamicsWindows:
    """Wraps AlignedWindows (the canonical temporal contract) and featurizes items."""

    def __init__(self, aligned, keep_windows=None):
        self.aw = aligned
        self.H = aligned.spec.horizon
        self.C = aligned.spec.context
        if self.C < 2:
            raise ValueError("context must be >= 2 (rollout step k=1 needs the warm-up prediction of step t0-1)")
        self.feat = {s: WindowFeaturizer(m) for s, m in aligned.metas.items()}
        self.index = list(range(len(aligned))) if keep_windows is None else list(keep_windows)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        it = self.aw[self.index[i]]
        f = self.feat[it["index"]["scenario"]]
        out = {"inputs": f.runtime_part(it["runtime_inputs"], it["actions"], self.H),
               "targets": f.target_part(it["targets"]),
               "info": {**it["index"], "K": f.K, "V": f.V}}
        return out


def collate(items: list[dict]) -> dict:
    """Pad zones (V) and AHUs (K) across buildings; return torch tensors + masks."""
    import torch
    B = len(items)
    Vm = max(it["info"]["V"] for it in items)
    Km = max(it["info"]["K"] for it in items)

    def pad(arr, axis, n, value=0.0):
        width = [(0, 0)] * arr.ndim
        width[axis] = (0, n - arr.shape[axis])
        return np.pad(arr, width, constant_values=value)

    zone_axis = {"zone_temp_obs": 1, "zone_act": 1, "zone_static": 0, "zone_ahu": 0}
    ahu_axis = {"ahu_act": 1}
    t_zone = {"zone_temp": 1, "zone_occ": 1, "zone_setpoints": 1}
    out_in, out_t = {}, {}
    for k in items[0]["inputs"]:
        arrs = []
        for it in items:
            a = np.asarray(it["inputs"][k])
            if k in zone_axis:
                a = pad(a, zone_axis[k], Vm)
            if k in ahu_axis:
                a = pad(a, ahu_axis[k], Km)
            arrs.append(a)
        out_in[k] = torch.from_numpy(np.stack(arrs))
    for k in items[0]["targets"]:
        arrs = [pad(np.asarray(it["targets"][k]), t_zone[k], Vm) if k in t_zone else np.asarray(it["targets"][k]) for it in items]
        out_t[k] = torch.from_numpy(np.stack(arrs))
    zm = torch.zeros(B, Vm, dtype=torch.bool)
    am = torch.zeros(B, Km, dtype=torch.bool)
    for b, it in enumerate(items):
        zm[b, :it["info"]["V"]] = True
        am[b, :it["info"]["K"]] = True
    info = {k: [it["info"][k] for it in items] for k in items[0]["info"]}
    info["K"] = torch.tensor(info["K"], dtype=torch.float32)
    return {"inputs": out_in, "targets": out_t, "zone_mask": zm, "ahu_mask": am, "info": info}
