"""Graph-aware recurrent dynamics model (Phase 2B Workstream B).

Heterogeneous graph per building (src/graph/topology.py): zone (V), AHU (K), plant/HWS (1),
global/building (1) nodes; typed edges zone<->AHU, zone<->plant, zone<->zone (wall adjacency
from the public zone_map, degree-normalized wall-length weights), and pooling to the global node.

Per step (same step contract, inputs and rollout as the Phase 2A GRU; `dynamics.model.rollout`
drives it unchanged):
  1. type-specific encoders of the step features (zone: T, static geometry, damper/valve
     a_{s-1}, a_s, setpoint gaps; AHU: run/SAT/pressure/recirculation a_{s-1}, a_s;
     plant: HWS temperature, HWS setpoint/dp/run a_{s-1}, a_s; global: outside air, clock,
     comfort mode, setpoint window)
  2. typed message passing from the previous recurrent node states:
        zone <- its AHU, zone <- neighbours (with the neighbour temperature difference, the
        physical driver of wall conduction), zone <- plant, zone <- global,
        AHU <- mean of its zones, plant <- mean of zones, global <- pooled zones/AHUs/plant
  3. one GRU cell per node type updates the latent states (messages are SUMMED, not
     concatenated, to keep the parameter count comparable to the Phase 2A GRU)
  4. heads: zone dT (residual, same scale as Phase 2A), tank dT from the plant node,
     fan/cooling energy per AHU node (summed over AHUs), pump/gas from the plant node.
Action inputs enter only the node they physically act on (zone actions -> zone nodes, AHU
actions -> AHU nodes, HWS actions -> plant node); their effect on other nodes must flow through
typed edges. This is the hypothesis being tested against the Phase 2A GRU (where every zone
receives the building action vector directly).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamics.features import SETPOINT_S
from dynamics.model import Norm, ahu_pool, gather_ahu, masked_mean, mlp

ZONE_IN, AHU_IN, PLANT_IN, GLOBAL_IN = 9, 8, 7, 10


def graph_step_features(batch, j, T_in, hws_in, oat, norm: Norm):
    inp = batch["inputs"]
    za, aa, ba = inp["zone_act"], inp["ahu_act"], inp["bldg_act"]
    exo = inp["exo"][:, j]
    lo = exo[:, 6] * SETPOINT_S + 295.0
    hi = exo[:, 7] * SETPOINT_S + 295.0
    zone_x = torch.cat([((T_in - norm.Tm) / norm.Ts).unsqueeze(-1), inp["zone_static"], za[:, j], za[:, j + 1],
                        ((lo.unsqueeze(1) - T_in) / 5.0).unsqueeze(-1), ((hi.unsqueeze(1) - T_in) / 5.0).unsqueeze(-1)], -1)
    ahu_x = torch.cat([aa[:, j], aa[:, j + 1]], -1)                                             # [B,K,8]
    plant_x = torch.cat([((hws_in - norm.Hm) / norm.Hs).unsqueeze(-1), ba[:, j], ba[:, j + 1]], -1)   # [B,7]
    zmean = masked_mean(T_in.unsqueeze(-1), batch["zone_mask"], 1).squeeze(-1)
    glob_x = torch.cat([((oat - norm.Om) / norm.Os).unsqueeze(-1), ((oat - zmean) / 10.0).unsqueeze(-1), exo], -1)  # [B,10]
    return zone_x, ahu_x, plant_x, glob_x


class GraphDynamics(nn.Module):
    kind = "graph"

    def __init__(self, norm: Norm, hidden=48, head_hidden=64, use_zone_adjacency=True):
        super().__init__()
        self.norm = norm
        d = hidden
        self.d = d
        self.use_zz = use_zone_adjacency
        self.enc_z, self.enc_a = mlp(ZONE_IN, d, d), mlp(AHU_IN, d, d)
        self.enc_p, self.enc_g = mlp(PLANT_IN, d, d), mlp(GLOBAL_IN, d, d)
        self.msg_za = nn.Linear(2 * d, d)        # AHU -> zone
        self.msg_zz = nn.Linear(d + 1, d)        # neighbour zone -> zone (+ temperature difference)
        self.msg_zp = nn.Linear(2 * d, d)        # plant -> zone
        self.msg_zg = nn.Linear(2 * d, d)        # global -> zone
        self.msg_az = nn.Linear(d, d)            # zones -> AHU (mean)
        self.msg_pz = nn.Linear(d, d)            # zones -> plant (mean)
        self.msg_g = nn.Linear(3 * d, d)         # pooled zones, AHUs, plant -> global
        self.cell_z, self.cell_a = nn.GRUCell(d, d), nn.GRUCell(d, d)
        self.cell_p, self.cell_g = nn.GRUCell(d, d), nn.GRUCell(d, d)
        self.h0 = nn.ParameterDict({k: nn.Parameter(torch.zeros(d)) for k in ("z", "a", "p", "g")})
        self.zone_head = mlp(d + ZONE_IN, head_hidden, 1)
        self.hws_head = mlp(d + PLANT_IN, head_hidden, 1)
        self.ahu_energy_head = mlp(d + AHU_IN + d, head_hidden, 2)
        self.plant_energy_head = mlp(d + PLANT_IN + d, head_hidden, 2)

    def init_state(self, batch):
        B, V = batch["zone_mask"].shape
        K = batch["ahu_mask"].shape[1]
        h = self.h0
        return (h["z"].expand(B, V, -1).contiguous(), h["a"].expand(B, K, -1).contiguous(),
                h["p"].expand(B, -1).contiguous(), h["g"].expand(B, -1).contiguous())

    def _zz(self, hz, T_in, batch):
        """Weighted sum over wall-adjacent zones of msg_zz([h_nb, (T_nb - T)/5])."""
        g = batch.get("graph")
        B, V, d = hz.shape
        if not self.use_zz or g is None or g["zz_src"].numel() == 0:
            return torch.zeros_like(hz)
        hf, Tf = hz.reshape(B * V, d), T_in.reshape(B * V)
        src, dst, w = g["zz_src"], g["zz_dst"], g["zz_w"]
        m = self.msg_zz(torch.cat([hf[src], ((Tf[src] - Tf[dst]) / 5.0).unsqueeze(-1)], -1)) * w.unsqueeze(-1)
        out = torch.zeros(B * V, d, dtype=hz.dtype, device=hz.device).index_add_(0, dst, m)
        return out.reshape(B, V, d)

    def step(self, state, batch, j, T_in, hws_in, oat):
        hz, ha, hp, hg = state
        zm, am = batch["zone_mask"], batch["ahu_mask"]
        zmf, amf = zm.unsqueeze(-1).to(hz.dtype), am.unsqueeze(-1).to(hz.dtype)
        zone_ahu = batch["inputs"]["zone_ahu"]
        zone_x, ahu_x, plant_x, glob_x = graph_step_features(batch, j, T_in, hws_in, oat, self.norm)
        B, V, _ = zone_x.shape
        K = ahu_x.shape[1]
        ez, ea, ep, eg = self.enc_z(zone_x), self.enc_a(ahu_x), self.enc_p(plant_x), self.enc_g(glob_x)
        # messages from previous states (+ current encodings of the sender)
        m_za = self.msg_za(gather_ahu(torch.cat([ha, ea], -1), zone_ahu))
        m_zz = self._zz(hz, T_in, batch)
        m_zp = self.msg_zp(torch.cat([hp, ep], -1)).unsqueeze(1)
        m_zg = self.msg_zg(torch.cat([hg, eg], -1)).unsqueeze(1)
        m_az = self.msg_az(ahu_pool(hz, zone_ahu, zm, K))
        m_pz = self.msg_pz(masked_mean(hz, zm, 1))
        # recurrent updates
        hz_new = self.cell_z(F.silu(ez + m_za + m_zz + m_zp + m_zg).reshape(B * V, -1), hz.reshape(B * V, -1)).reshape(B, V, -1) * zmf
        ha_new = self.cell_a(F.silu(ea + m_az).reshape(B * K, -1), ha.reshape(B * K, -1)).reshape(B, K, -1) * amf
        hp_new = self.cell_p(F.silu(ep + m_pz), hp)
        m_g = self.msg_g(torch.cat([masked_mean(hz_new, zm, 1), masked_mean(ha_new, am, 1), hp_new], -1))
        hg_new = self.cell_g(F.silu(eg + m_g), hg)
        # heads
        dT = self.zone_head(torch.cat([hz_new, zone_x], -1)).squeeze(-1)
        T_out = T_in + self.norm.dTs * dT
        hws_out = hws_in + self.norm.dHs * self.hws_head(torch.cat([hp_new, plant_x], -1)).squeeze(-1)
        e_ahu = F.softplus(self.ahu_energy_head(torch.cat([ha_new, ahu_x, hg_new.unsqueeze(1).expand(-1, K, -1)], -1)))
        em = self.norm.em.to(hz.device)
        e_ahu = e_ahu / math.log(2) * em[:2] * amf
        e_fan_cool = e_ahu.sum(1) / am.sum(1, keepdim=True).clamp_min(1).to(hz.dtype)
        e_plant = F.softplus(self.plant_energy_head(torch.cat([hp_new, plant_x, hg_new], -1))) / math.log(2) * em[2:]
        return (hz_new, ha_new, hp_new, hg_new), {"zone_temp": T_out, "hws": hws_out,
                                                  "energy_norm": torch.cat([e_fan_cool, e_plant], -1)}
