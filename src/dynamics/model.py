"""GRU multi-task dynamics model, baselines, and the shared rollout.

Size-agnostic design (no GNN): one GRU cell shared by every zone (zone latent h_z),
one building GRU cell (h_b), DeepSets mean-pooling between them, and per-AHU energy
heads summed over AHUs. Padding masks allow mixed-building batches.

All models implement   state = init_state(batch);  state, out = step(state, x)
and are driven by `rollout()`, which enforces the observation/feedback rules of
features.py: warm-up and decision steps receive true observations; rollout steps
receive only the model's own predictions and runtime-forecastable exogenous inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamics.features import E_PER_AHU, G_PER_AHU, SETPOINT_S
from generalization import FittedStatistics

ZONE_F, AHU_F, BLDG_F = 9, 8, 17
ENERGY_SCALE = torch.tensor([E_PER_AHU, E_PER_AHU, E_PER_AHU, G_PER_AHU])   # W per AHU, per component


# ------------------------------------------------------------------ normalizer (train-only fit)
def fit_normalizer(root, train_rows) -> FittedStatistics:
    """Group-level statistics from TRAIN episodes only (provenance recorded)."""
    from building_meta import ObservationCodec
    from dataset import load_episode, load_scenario_meta
    from generalization import manifest_digest
    acc = {k: [] for k in ("T", "dT", "hws", "dH", "oat", "e0", "e1", "e2", "e3")}
    for r in train_rows:
        ep = load_episode(root / r["path"], keys=["obs", "zone_info", "tank_state", "ahu_info", "boiler_info"])
        m = load_scenario_meta(root, r["scenario"])
        oc = ObservationCodec(m)
        nat = oc.native(ep["obs"])
        Tz = nat[:, m["zones"]["obs_temp"]]
        acc["T"].append(Tz.ravel())
        acc["dT"].append((ep["zone_info"][..., 2] - Tz[:-1]).ravel())          # after t  vs  after t-2
        hws = nat[:, m["hws"]["obs_supply_water_temp"]]
        acc["hws"].append(hws)
        acc["dH"].append(ep["tank_state"][1:, 0] - hws[:-1])                   # after t  vs  after t-1
        acc["oat"].append(nat[:, m["ahus"]["obs_outside_air_temp"][0]])
        K = m["num_ahus"]
        a, b = ep["ahu_info"].astype(np.float64), ep["boiler_info"].astype(np.float64)
        acc["e0"].append(a[..., 0].sum(-1) / (K * E_PER_AHU))
        acc["e1"].append(np.abs(a[..., 1]).sum(-1) / (K * E_PER_AHU))
        acc["e2"].append(b[..., 1].sum(-1) / (K * E_PER_AHU))
        acc["e3"].append(b[..., 0].sum(-1) / (K * G_PER_AHU))
    stats = {}
    for k, v in acc.items():
        x = np.concatenate(v)
        stats[k] = {"mean": float(x.mean()), "std": float(x.std() + 1e-6), "n": int(x.size)}
    prov = {"buildings": sorted({r["building"] for r in train_rows}), "episode_ids": sorted(r["episode_id"] for r in train_rows),
            "manifest_digest": manifest_digest(train_rows)}
    return FittedStatistics(stats, prov)


@dataclass
class Norm:
    Tm: float; Ts: float; dTs: float; Hm: float; Hs: float; dHs: float; Om: float; Os: float
    es: torch.Tensor   # [4] std of per-AHU-normalized energy
    em: torch.Tensor   # [4] mean of per-AHU-normalized energy (output scale of the energy heads)

    @classmethod
    def from_stats(cls, st: FittedStatistics) -> "Norm":
        s = st.stats
        return cls(s["T"]["mean"], s["T"]["std"], s["dT"]["std"], s["hws"]["mean"], s["hws"]["std"], s["dH"]["std"],
                   s["oat"]["mean"], s["oat"]["std"], torch.tensor([s[f"e{i}"]["std"] for i in range(4)], dtype=torch.float32),
                   torch.tensor([max(s[f"e{i}"]["mean"], 1e-5) for i in range(4)], dtype=torch.float32))


# ------------------------------------------------------------------ step features
def masked_mean(x, mask, dim):
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim) / m.sum(dim).clamp_min(1.0)


def step_features(batch, j, T_in, hws_in, oat, norm: Norm):
    """Network inputs of step j. T_in [B,V], hws_in [B], oat [B] in K (true or predicted)."""
    inp = batch["inputs"]
    za, aa, ba = inp["zone_act"], inp["ahu_act"], inp["bldg_act"]
    exo = inp["exo"][:, j]                                            # [B,8]
    lo = exo[:, 6] * SETPOINT_S + 295.0
    hi = exo[:, 7] * SETPOINT_S + 295.0
    zone_x = torch.cat([((T_in - norm.Tm) / norm.Ts).unsqueeze(-1), inp["zone_static"],
                        za[:, j], za[:, j + 1],                                       # damper, valve: a_{s-1}, a_s
                        ((lo.unsqueeze(1) - T_in) / 5.0).unsqueeze(-1), ((hi.unsqueeze(1) - T_in) / 5.0).unsqueeze(-1)], -1)
    ahu_x = torch.cat([aa[:, j], aa[:, j + 1]], -1)                                   # [B,K,8]
    zmean = masked_mean(T_in.unsqueeze(-1), batch["zone_mask"], 1).squeeze(-1)
    bldg_x = torch.cat([((hws_in - norm.Hm) / norm.Hs).unsqueeze(-1), ((oat - norm.Om) / norm.Os).unsqueeze(-1),
                        ((oat - zmean) / 10.0).unsqueeze(-1), exo, ba[:, j], ba[:, j + 1]], -1)   # [B,17]
    return zone_x, ahu_x, bldg_x


def gather_ahu(ahu_x, zone_ahu):
    idx = zone_ahu.long().clamp_min(0).unsqueeze(-1).expand(-1, -1, ahu_x.shape[-1])
    return torch.gather(ahu_x, 1, idx)


def ahu_pool(hz, zone_ahu, zone_mask, K):
    """Mean of zone latents per AHU -> [B,K,H]."""
    B, V, Hd = hz.shape
    onehot = F.one_hot(zone_ahu.long().clamp_min(0), K).to(hz.dtype) * zone_mask.unsqueeze(-1).to(hz.dtype)   # [B,V,K]
    s = torch.einsum("bvk,bvh->bkh", onehot, hz)
    return s / onehot.sum(1).clamp_min(1.0).unsqueeze(-1)


def mlp(i, h, o, layers=2):
    mods, d = [], i
    for _ in range(layers - 1):
        mods += [nn.Linear(d, h), nn.SiLU()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


# ------------------------------------------------------------------ models
class GRUDynamics(nn.Module):
    kind = "gru"

    def __init__(self, norm: Norm, hidden_zone=64, hidden_bldg=64, head_hidden=64):
        super().__init__()
        self.norm = norm
        self.hz, self.hb = hidden_zone, hidden_bldg
        self.zone_cell = nn.GRUCell(ZONE_F + AHU_F + BLDG_F + hidden_zone + hidden_bldg, hidden_zone)
        self.bldg_cell = nn.GRUCell(BLDG_F + hidden_zone + AHU_F, hidden_bldg)
        self.h0_zone = nn.Parameter(torch.zeros(hidden_zone))
        self.h0_bldg = nn.Parameter(torch.zeros(hidden_bldg))
        self.zone_head = mlp(hidden_zone + ZONE_F, head_hidden, 1)
        self.hws_head = mlp(hidden_bldg + BLDG_F, head_hidden, 1)
        self.ahu_energy_head = mlp(hidden_zone + AHU_F + hidden_bldg + BLDG_F, head_hidden, 2)
        self.plant_energy_head = mlp(hidden_bldg + BLDG_F, head_hidden, 2)

    def init_state(self, batch):
        B, V = batch["zone_mask"].shape
        return (self.h0_zone.expand(B, V, -1).contiguous(), self.h0_bldg.expand(B, -1).contiguous())

    def step(self, state, batch, j, T_in, hws_in, oat):
        hz, hb = state
        zm, am = batch["zone_mask"], batch["ahu_mask"]
        zone_ahu = batch["inputs"]["zone_ahu"]
        zone_x, ahu_x, bldg_x = step_features(batch, j, T_in, hws_in, oat, self.norm)
        B, V, _ = zone_x.shape
        K = ahu_x.shape[1]
        pooled = masked_mean(hz, zm, 1)
        zin = torch.cat([zone_x, gather_ahu(ahu_x, zone_ahu), bldg_x.unsqueeze(1).expand(-1, V, -1),
                         pooled.unsqueeze(1).expand(-1, V, -1), hb.unsqueeze(1).expand(-1, V, -1)], -1)
        hz = self.zone_cell(zin.reshape(B * V, -1), hz.reshape(B * V, -1)).reshape(B, V, -1)
        hz = hz * zm.unsqueeze(-1).to(hz.dtype)
        hb = self.bldg_cell(torch.cat([bldg_x, masked_mean(hz, zm, 1), masked_mean(ahu_x, am, 1)], -1), hb)
        dT = self.zone_head(torch.cat([hz, zone_x], -1)).squeeze(-1)
        T_out = T_in + self.norm.dTs * dT
        hws_out = hws_in + self.norm.dHs * self.hws_head(torch.cat([hb, bldg_x], -1)).squeeze(-1)
        ahu_emb = torch.cat([ahu_pool(hz, zone_ahu, zm, K), ahu_x, hb.unsqueeze(1).expand(-1, K, -1),
                             bldg_x.unsqueeze(1).expand(-1, K, -1)], -1)
        # softplus(0)/ln2 = 1, so at initialization each head outputs the training mean (scale only, no fit)
        em = self.norm.em.to(hz.device)
        e_ahu = F.softplus(self.ahu_energy_head(ahu_emb)) / math.log(2) * em[:2] * am.unsqueeze(-1).to(hz.dtype)
        e_fan_cool = e_ahu.sum(1) / am.sum(1, keepdim=True).clamp_min(1).to(hz.dtype)             # per-AHU normalized totals
        e_plant = F.softplus(self.plant_energy_head(torch.cat([hb, bldg_x], -1))) / math.log(2) * em[2:]
        energy = torch.cat([e_fan_cool, e_plant], -1)
        return (hz, hb), {"zone_temp": T_out, "hws": hws_out, "energy_norm": energy}


class LinearDynamics(nn.Module):
    """Linear autoregressive baseline on the SAME step features; fitted by ridge regression."""
    kind = "linear"

    def __init__(self, norm: Norm):
        super().__init__()
        self.norm = norm
        nz, nb = ZONE_F + AHU_F + BLDG_F + 2, BLDG_F + ZONE_F + AHU_F + 1
        self.register_buffer("Wz", torch.zeros(nz, 1))
        self.register_buffer("Wh", torch.zeros(nb, 1))
        self.register_buffer("We", torch.zeros(nb, 4))

    def init_state(self, batch):
        return None

    def design(self, batch, j, T_in, hws_in, oat, T_prev=None):
        """T_prev: zone state after s-1 (own previous prediction in rollouts; past sensor when fitting)."""
        zm, am = batch["zone_mask"], batch["ahu_mask"]
        zone_x, ahu_x, bldg_x = step_features(batch, j, T_in, hws_in, oat, self.norm)
        B, V, _ = zone_x.shape
        one_z = torch.ones(B, V, 1, device=zone_x.device)
        prev = torch.zeros(B, V, 1, device=zone_x.device) if T_prev is None else ((T_prev - T_in) / self.norm.dTs).unsqueeze(-1)
        Xz = torch.cat([zone_x, gather_ahu(ahu_x, batch["inputs"]["zone_ahu"]), bldg_x.unsqueeze(1).expand(-1, V, -1), prev, one_z], -1)
        Xb = torch.cat([bldg_x, masked_mean(zone_x, zm, 1), masked_mean(ahu_x, am, 1), torch.ones(B, 1, device=zone_x.device)], -1)
        return Xz, Xb

    def step(self, state, batch, j, T_in, hws_in, oat):
        # T_prev = zone state after s-1. It is OBSERVED (obs row j+1) for warm-up steps j <= C-2;
        # from the decision step on it is not, so the model's own previous prediction is used.
        C = batch["inputs"]["zone_temp_obs"].shape[1]
        T_prev = batch["inputs"]["zone_temp_obs"][:, j + 1] if j + 1 <= C - 1 else state
        Xz, Xb = self.design(batch, j, T_in, hws_in, oat, T_prev=T_prev)
        T_out = T_in + self.norm.dTs * (Xz @ self.Wz).squeeze(-1)
        hws_out = hws_in + self.norm.dHs * (Xb @ self.Wh).squeeze(-1)
        energy = (Xb @ self.We).clamp_min(0.0)          # clipped at 0 (documented)
        return T_out, {"zone_temp": T_out, "hws": hws_out, "energy_norm": energy}


class PersistenceModel(nn.Module):
    """Zone/HWS: latest available observation (persisted through the rollout).
    Energy: training-set mean per (hour of day, outside-air bin) -- no energy is observable."""
    kind = "persistence"
    OAT_EDGES = torch.tensor([-30.0, -15.0, -5.0, 5.0, 15.0, 25.0, 35.0]) + 273.15

    def __init__(self, norm: Norm):
        super().__init__()
        self.norm = norm
        self.register_buffer("table", torch.zeros(24, len(self.OAT_EDGES) + 1, 4))

    def init_state(self, batch):
        return None

    def bins(self, batch, j, oat):
        exo = batch["inputs"]["exo"][:, j]
        hour = torch.remainder(torch.atan2(exo[:, 0], exo[:, 1]), 2 * np.pi) * 24 / (2 * np.pi)
        return hour.floor().long().clamp(0, 23), torch.bucketize(oat.contiguous(), self.OAT_EDGES.to(oat.device))

    def step(self, state, batch, j, T_in, hws_in, oat):
        # Hold the most recent TRUE observation (o_s for s <= t0, o_t0 afterwards); ignores fed-back inputs.
        C = batch["inputs"]["zone_temp_obs"].shape[1]
        jj = min(j, C - 1)
        h, o = self.bins(batch, j, oat)
        return None, {"zone_temp": batch["inputs"]["zone_temp_obs"][:, jj].clone(),
                      "hws": batch["inputs"]["hws_obs"][:, jj].clone(), "energy_norm": self.table[h, o]}


# ------------------------------------------------------------------ rollout
def rollout(model, batch, exo_mode: str = "persistence"):
    """Returns predictions for rollout steps k = 0..H-1 (transition t0+k).

    exo_mode: 'persistence' (deployable: outside air held at o_{t0}) or 'oracle'
              (diagnostic only: true future outside air from obs_window)."""
    inp = batch["inputs"]
    C = inp["zone_temp_obs"].shape[1]
    S = inp["exo"].shape[1]
    H = S - (C - 1)
    state = model.init_state(batch)
    zpred, hpred, outs = [], [], []
    for j in range(S):
        if j <= C - 1:                                   # warm-up and decision step: true observations
            T_in, hws_in, oat = inp["zone_temp_obs"][:, j], inp["hws_obs"][:, j], inp["oat_obs"][:, j]
        else:                                            # rollout: own predictions only
            T_in = zpred[j - 2]                          # zone state after s-2 (offset 2)
            hws_in = hpred[j - 1]                        # tank after s-1 (offset 1)
            k = j - (C - 1)
            oat = inp["oat_obs"][:, C - 1] if exo_mode == "persistence" else batch["targets"]["oat_oracle"][:, k]
        state, out = model.step(state, batch, j, T_in, hws_in, oat)
        zpred.append(out["zone_temp"])
        hpred.append(out["hws"])
        if j >= C - 1:
            outs.append(out)
    res = {k: torch.stack([o[k] for o in outs], 1) for k in outs[0]}     # [B,H,...]
    K = batch["info"]["K"].unsqueeze(-1)
    res["energy_w"] = res["energy_norm"] * K.unsqueeze(-1) * ENERGY_SCALE.to(K.device)                      # [B,H,4] W
    assert res["zone_temp"].shape[1] == H
    return res


def energy_norm_target(batch):
    K = batch["info"]["K"].view(-1, 1, 1)
    return batch["targets"]["energy"] / (K * ENERGY_SCALE.to(K.device))
