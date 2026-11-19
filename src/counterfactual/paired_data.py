"""Branch-group dataset for paired counterfactual training (Phase 2C).

One item = ONE branch point = the nominal arm + all intervention arms. Arms of a branch point are never
split across batches or splits. Every arm is featurized with the Phase 2B builder
(`counterfactual.model_eval.build_items`): identical runtime context (o_{t_b-11..t_b}, executed action
history, schedule/clock features from o_{t_b}); only the future action sequence differs. Nothing from the
future (observations, occupancy, weather) is a model input; simulator trajectories are targets only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from counterfactual.model_eval import build_items
from dynamics.features import E_PER_AHU, G_PER_AHU, collate

ENERGY_PER_AHU = np.array([E_PER_AHU, E_PER_AHU, E_PER_AHU, G_PER_AHU])


def load_branch(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


class BranchGroups(torch.utils.data.Dataset):
    """Pre-featurized branch groups. Aborted arms are dropped (counted in .dropped_aborted)."""

    def __init__(self, data_dir: str | Path, branch_ids: Iterable[str], meta_root: str | Path):
        from dataset import load_scenario_meta
        self.data_dir, self.meta_root = Path(data_dir), Path(meta_root)
        self.groups, self.dropped_aborted = [], 0
        metas = {}
        for bid in sorted(branch_ids):
            sim = load_branch(self.data_dir / "branches" / f"{bid}.npz")
            sc = str(sim["scenario"])
            if sc not in metas:
                metas[sc] = load_scenario_meta(self.meta_root, sc)
            items = build_items(sim, metas[sc], 0)
            keep = ~sim["aborted"][0]
            self.dropped_aborted += int((~keep).sum())
            arms = []
            for a, it in enumerate(items):
                if not keep[a]:
                    continue
                it["targets"].update({"zone_temp": sim["zone_temp"][0, a].astype(np.float32),
                                      "hws": sim["tank"][0, a].astype(np.float32),
                                      "energy": sim["energy"][0, a].astype(np.float32)})
                it["info"].update({"branch_id": bid, "is_nominal": a == 0, "noop": bool(sim["effective_noop"][0, a]),
                                   "family": str(sim["arm_family"][a])})
                arms.append(it)
            assert arms and arms[0]["info"]["is_nominal"], f"{bid}: nominal arm missing"
            self.groups.append(arms)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        return self.groups[i]

    def delta_stats(self) -> dict:
        """Std of simulated paired responses over non-no-op arms (TRAIN groups only), per target, in the
        units the loss uses: zone K, tank K, energy per-AHU-normalized."""
        dz, dh, de = [], [], []
        for g in self.groups:
            nom = g[0]["targets"]
            K = g[0]["info"]["K"]
            for it in g[1:]:
                if it["info"]["noop"]:
                    continue
                t = it["targets"]
                dz.append((t["zone_temp"] - nom["zone_temp"]).ravel())
                dh.append(t["hws"] - nom["hws"])
                de.append((t["energy"] - nom["energy"]) / (K * ENERGY_PER_AHU))
        def sd(x):
            x = np.concatenate(x) if len(x) else np.zeros(1)
            return float(max(np.std(x), 1e-4))
        e = np.concatenate(de, 0) if de else np.zeros((1, 4))
        return {"zone": sd(dz), "tank": sd(dh), "energy": [float(max(np.std(e[:, c]), 1e-5)) for c in range(4)],
                "n_groups": len(self.groups), "n_arms": int(sum(len(g) - 1 for g in self.groups))}


def collate_groups(groups: list[list[dict]]) -> dict:
    """Flatten groups -> one padded batch (base collate) + per-row group bookkeeping:
    nominal_row[r] = row of r's nominal arm; arm_weight flags is_nominal / noop."""
    items, nom_row, is_nom, noop = [], [], [], []
    for g in groups:
        base = len(items)
        for it in g:
            items.append(it)
            nom_row.append(base)
            is_nom.append(it["info"]["is_nominal"])
            noop.append(it["info"]["noop"])
    batch = collate(items)
    batch["pair"] = {"nominal_row": torch.tensor(nom_row), "is_nominal": torch.tensor(is_nom),
                     "noop": torch.tensor(noop)}
    return batch


def branch_ids_for(split_manifest: Mapping[str, Any], splits: Iterable[str], buildings: Iterable[str]) -> list[str]:
    s, b = set(splits), set(buildings)
    return sorted(k for k, v in split_manifest["assignments"].items() if v["split"] in s and v["building"] in b)


def load_split_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
