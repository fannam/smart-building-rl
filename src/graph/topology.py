"""Building HVAC topology from public metadata (Phase 2B Workstream B).

Node types: zone (V), ahu (K), plant/HWS (1), building (1).
Edges (all from public metadata; nothing invented):
  zone <-> ahu        meta["zones"]["ahu"]            (every zone has exactly one AHU)
  zone <-> plant      every VAV has a reheat valve fed by the hot-water system
                      (meta["zones"]["act_reheat_valve"] present for the zone)
  ahu  -> building, plant -> building, zone -> building (pooling)
  zone <-> zone       wall adjacency derived from `zone_map` (part of create_building_meta_data,
                      the same public geometry the baseline policy uses for zone areas). Rooms are
                      the 4-connected interior components (0=interior, 1=wall, 2=exterior) in
                      row-major order; two rooms are adjacent when a single wall cell separates
                      them. Weight = shared wall length (cells). If the component count does not
                      equal the number of zones, adjacency is marked UNAVAILABLE (no edges).
                      Inter-floor (vertical) adjacency is not represented in the 2D map and is
                      therefore not modelled.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def room_components(zone_map: np.ndarray) -> np.ndarray:
    """Label image: -1 for non-interior, component index (row-major discovery order) otherwise."""
    grid = np.asarray(zone_map)
    H, W = grid.shape
    lab = np.full(grid.shape, -1, dtype=np.int64)
    inside = grid == 0
    n = 0
    for start in np.flatnonzero(inside.ravel()):
        r0, c0 = divmod(int(start), W)
        if lab[r0, c0] >= 0:
            continue
        stack = [(r0, c0)]
        lab[r0, c0] = n
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and inside[rr, cc] and lab[rr, cc] < 0:
                    lab[rr, cc] = n
                    stack.append((rr, cc))
        n += 1
    return lab


def zone_adjacency(zone_map: np.ndarray | None, zone_ids: list[str]) -> dict[str, Any]:
    V = len(zone_ids)
    if zone_map is None:
        return {"available": False, "reason": "no zone_map", "edges": np.zeros((0, 2), np.int64), "weight": np.zeros(0)}
    lab = room_components(zone_map)
    n = int(lab.max()) + 1
    if n != V:
        return {"available": False, "reason": f"{n} interior components != {V} zones", "edges": np.zeros((0, 2), np.int64),
                "weight": np.zeros(0)}
    # component index -> zone index via the trailing number of the zone id (baseline convention)
    comp_to_zone = np.full(n, -1, dtype=np.int64)
    for j, name in enumerate(zone_ids):
        m = re.search(r"(\d+)$", name)
        lbl = int(m.group(1)) - 1 if m else j
        if 0 <= lbl < n:
            comp_to_zone[lbl] = j
    if (comp_to_zone < 0).any() or len(set(comp_to_zone.tolist())) != n:
        return {"available": False, "reason": "zone ids do not map 1:1 to components", "edges": np.zeros((0, 2), np.int64),
                "weight": np.zeros(0)}
    grid = np.asarray(zone_map)
    pairs: dict[tuple[int, int], int] = {}
    wall = grid == 1
    # horizontal: interior | wall | interior ; vertical likewise
    for a, b, w in ((lab[:, :-2], lab[:, 2:], wall[:, 1:-1]), (lab[:-2, :], lab[2:, :], wall[1:-1, :])):
        m = w & (a >= 0) & (b >= 0) & (a != b)
        for x, y in zip(a[m].tolist(), b[m].tolist()):
            key = (min(x, y), max(x, y))
            pairs[key] = pairs.get(key, 0) + 1
    if not pairs:
        return {"available": True, "reason": "no shared walls", "edges": np.zeros((0, 2), np.int64), "weight": np.zeros(0)}
    e = np.array([[comp_to_zone[x], comp_to_zone[y]] for x, y in pairs], dtype=np.int64)
    wgt = np.array(list(pairs.values()), dtype=np.float64)
    return {"available": True, "reason": "", "edges": e, "weight": wgt}


def build_topology(meta: Mapping[str, Any], zone_map: np.ndarray | None = None) -> dict[str, Any]:
    """Typed graph for one building (any K, V)."""
    V, K = int(meta["num_zones"]), int(meta["num_ahus"])
    zone_ahu = np.asarray(meta["zones"]["ahu"], dtype=np.int64)
    assert zone_ahu.shape == (V,) and zone_ahu.min() >= 0 and zone_ahu.max() < K
    has_reheat = np.asarray([i is not None for i in meta["zones"]["act_reheat_valve"]])
    adj = zone_adjacency(zone_map, list(meta["zones"]["ids"]))
    zz = adj["edges"]
    # symmetric, degree-normalized zone-zone weights (wall length normalized per receiving zone)
    if len(zz):
        src = np.concatenate([zz[:, 0], zz[:, 1]])
        dst = np.concatenate([zz[:, 1], zz[:, 0]])
        w = np.concatenate([adj["weight"], adj["weight"]])
        deg = np.bincount(dst, weights=w, minlength=V)
        w = w / deg[dst]
    else:
        src = dst = np.zeros(0, np.int64)
        w = np.zeros(0)
    return {"V": V, "K": K, "zone_ahu": zone_ahu, "zone_plant": has_reheat,
            "zz_src": src, "zz_dst": dst, "zz_w": w.astype(np.float32), "adjacency_available": bool(adj["available"]),
            "adjacency_reason": adj["reason"], "n_zz_edges": int(len(zz))}


def load_zone_map(data_root: Path, scenario: str) -> np.ndarray | None:
    p = Path(data_root) / "scenarios" / scenario.replace("/", "__") / "zone_map.npy"
    return np.load(p) if p.exists() else None
