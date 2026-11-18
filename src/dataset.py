"""Loading, building-level splits and a PyTorch view of the Phase 1 dataset.

Layout written by `collect_data.py`:

    <root>/manifest.jsonl                       one JSON row per episode (append-only;
                                                 later rows for the same episode_id win)
    <root>/scenarios/<bldg>__<weather>/meta.json  static scenario metadata (+ zone_map.npy,
                                                 floor_plan.npy)
    <root>/episodes/<episode_id>.npz            per-episode arrays (see docs/phase1_dataset.md)
    <root>/splits.json                          building-level split (write_split)

Splits are by *building*: every weather/episode of a building lands in the same
split, so validation measures transfer to unseen buildings (as in the hidden
evaluation, which uses generated buildings).

PyTorch is imported lazily so collection does not depend on it.

Two window APIs:
  * AlignedWindows (Phase 1.5, use this): explicit temporal contract, per-sensor-group
    target offsets from configs/sensor_alignment.json, runtime inputs separated from
    targets, privileged targets only on explicit request, effective actions.
  * TransitionWindows (Phase 1, legacy): raw slices, kept for backward compatibility.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ALL_BUILDINGS = ("building_train_1", "building_train_2", "building_train_4",
                 "building_train_5", "building_train_7", "building_train_8")
# Default hold-out: building_train_7 (mid-size) for validation. Override per study.
DEFAULT_VAL_BUILDINGS = ("building_train_7",)


# ------------------------------------------------------------------ manifest / io
def load_manifest(root: str | Path) -> list[dict[str, Any]]:
    rows: dict[str, dict] = {}
    path = Path(root) / "manifest.jsonl"
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["episode_id"]] = r
    return sorted(rows.values(), key=lambda r: r["episode_id"])


def load_scenario_meta(root: str | Path, scenario: str) -> dict[str, Any]:
    return json.loads((Path(root) / "scenarios" / scenario.replace("/", "__") / "meta.json").read_text())


def load_episode(path: str | Path, keys: Iterable[str] | None = None) -> dict[str, Any]:
    """All arrays of one episode (decompressed into memory) + parsed episode meta."""
    with np.load(path, allow_pickle=False) as z:
        names = z.files if keys is None else [
            k for k in z.files if k in set(keys) | {"episode_json"} or k.startswith(("names_", "keys_"))]
        out = {k: z[k] for k in names}
    out["meta"] = json.loads(str(out.pop("episode_json")))
    for k in list(out):
        if k.startswith(("names_", "keys_")):
            out[k] = [str(x) for x in out[k]]
    return out


def unpack_episode(npz_path: str | Path, cache_dir: str | Path) -> Path:
    """Write each array as an uncompressed .npy so it can be memory-mapped."""
    npz_path = Path(npz_path)
    d = Path(cache_dir) / npz_path.stem
    marker = d / ".complete"
    if marker.exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    with np.load(npz_path, allow_pickle=False) as z:
        for k in z.files:
            np.save(d / f"{k}.npy", z[k])
    marker.write_text("ok")
    return d


def open_unpacked(d: str | Path, keys: Iterable[str]) -> dict[str, np.ndarray]:
    return {k: np.load(Path(d) / f"{k}.npy", mmap_mode="r") for k in keys}


# ------------------------------------------------------------------ splits
def building_split(buildings: Sequence[str], val: Sequence[str] = DEFAULT_VAL_BUILDINGS,
                   test: Sequence[str] = ()) -> dict[str, list[str]]:
    val, test = list(val), list(test)
    overlap = set(val) & set(test)
    if overlap:
        raise ValueError(f"buildings in both val and test: {overlap}")
    train = [b for b in buildings if b not in set(val) | set(test)]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def leave_one_building_out(buildings: Sequence[str]) -> list[dict[str, list[str]]]:
    return [building_split(buildings, val=[b]) for b in sorted(buildings)]


def write_split(root: str | Path, split: dict[str, list[str]]) -> Path:
    p = Path(root) / "splits.json"
    p.write_text(json.dumps(split, indent=1))
    return p


def read_split(root: str | Path) -> dict[str, list[str]]:
    p = Path(root) / "splits.json"
    if p.exists():
        return json.loads(p.read_text())
    buildings = sorted({r["building"] for r in load_manifest(root)})
    return building_split(buildings, val=[b for b in DEFAULT_VAL_BUILDINGS if b in buildings])


def episodes_for(root: str | Path, split_name: str, split: dict[str, list[str]] | None = None,
                 profiles: Sequence[str] | None = None) -> list[dict[str, Any]]:
    split = split or read_split(root)
    keep = set(split[split_name])
    rows = [r for r in load_manifest(root) if r["building"] in keep]
    if profiles is not None:
        rows = [r for r in rows if r["profile"] in set(profiles)]
    return rows


def assert_no_building_leakage(split: dict[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    for name, blds in split.items():
        for b in blds:
            if b in seen:
                raise AssertionError(f"{b} in both {seen[b]} and {name}")
            seen[b] = name


# ------------------------------------------------------------------ torch view
WINDOW_KEYS = ("obs", "action_continuous", "action_discrete", "baseline_continuous",
               "baseline_discrete", "reward", "zone_info", "tank_state", "calendar")


@dataclass
class _Ep:
    episode_id: str
    scenario: str
    arrays: dict[str, np.ndarray]
    T: int


class TransitionWindows:
    """LEGACY (Phase 1) raw windows (o_t..o_{t+H}, a_t..a_{t+H-1}, r_t..r_{t+H-1}).

    Deprecated for training. It returns only H+1 observations, so the zone-sensor
    state behind the last reward (in o_{t+H+1}) is outside the window, and it
    mixes privileged arrays with observations. Use `AlignedWindows` instead.
    Kept unchanged for scripts/validate_phase1.py and raw inspection.
    """

    def __init__(self, root: str | Path, rows: Sequence[dict[str, Any]], horizon: int = 12,
                 stride: int = 1, cache_dir: str | Path | None = None,
                 keys: Sequence[str] = WINDOW_KEYS):
        import warnings
        warnings.warn("TransitionWindows is legacy (H+1 observations, privileged arrays mixed with "
                      "inputs); use AlignedWindows", DeprecationWarning, stacklevel=2)
        self.root, self.horizon, self.keys = Path(root), int(horizon), tuple(keys)
        self.episodes: list[_Ep] = []
        self.index: list[tuple[int, int]] = []
        for r in rows:
            path = self.root / r["path"]
            if cache_dir is not None:
                arrays = open_unpacked(unpack_episode(path, cache_dir), self.keys)
            else:
                ep = load_episode(path, self.keys)
                arrays = {k: ep[k] for k in self.keys}
            T = int(arrays["reward"].shape[0])
            e = len(self.episodes)
            self.episodes.append(_Ep(r["episode_id"], r["scenario"], arrays, T))
            self.index += [(e, t) for t in range(0, T - self.horizon + 1, stride)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, Any]:
        e, t = self.index[i]
        ep, H = self.episodes[e], self.horizon
        out = {}
        for k, a in ep.arrays.items():
            # Arrays with T+1 rows (obs, tank_state) give H+1 states.
            n = H + 1 if a.shape[0] == ep.T + 1 else H
            out[k] = np.array(a[t:t + n])  # copy: memory-mapped arrays are read-only
        out["episode"] = e
        out["t0"] = t
        return out

    def scenario_batches(self, batch_size: int, shuffle: bool = True, seed: int = 0):
        """Index batches that never mix scenarios (homogeneous tensor shapes)."""
        rng = np.random.default_rng(seed)
        by_scn: dict[str, list[int]] = {}
        for i, (e, _) in enumerate(self.index):
            by_scn.setdefault(self.episodes[e].scenario, []).append(i)
        batches = []
        for idx in by_scn.values():
            idx = np.array(idx)
            if shuffle:
                rng.shuffle(idx)
            batches += [idx[j:j + batch_size].tolist() for j in range(0, len(idx), batch_size)]
        if shuffle:
            rng.shuffle(batches)
        return batches

    def as_torch(self):
        import torch

        parent = self

        class _TorchView(torch.utils.data.Dataset):
            def __len__(self):
                return len(parent)

            def __getitem__(self, i):
                item = parent[i]
                return {k: (torch.from_numpy(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else v)
                        for k, v in item.items()}

        return _TorchView()


# ================================================================== Phase 1.5: aligned windows
DATASET_API_VERSION = "aligned-windows-v1"
SENSOR_ALIGNMENT_PATH = Path(__file__).resolve().parents[1] / "configs" / "sensor_alignment.json"
DEFAULT_SENSOR_TARGETS = ("zone_air_temperature", "zone_heating_setpoint", "zone_cooling_setpoint",
                          "num_occupants", "hws_supply_water_temperature", "ahu_supply_air_flow",
                          "ahu_outside_air_flow")
# Privileged arrays that may be requested as targets, with their row offset relative
# to transition t (tank_state has T+1 rows: tank_state[t+1] is the tank after transition t).
PRIVILEGED_TARGETS = {"zone_info": 0, "ahu_info": 0, "boiler_info": 0, "heat_pump_info": 0,
                      "reward_response": 0, "calendar": 0, "tank_state": 1}


class SensorAlignment:
    """Loads configs/sensor_alignment.json and maps observation fields to groups."""

    def __init__(self, spec: dict[str, Any]):
        import re
        self.spec = spec
        self.groups = {g["name"]: g for g in spec["groups"]}
        self._patterns = [(g["name"], re.compile(g["pattern"])) for g in spec["groups"]]

    @classmethod
    def load(cls, path: str | Path = SENSOR_ALIGNMENT_PATH) -> "SensorAlignment":
        return cls(json.loads(Path(path).read_text()))

    def group_of(self, field: str) -> str | None:
        hits = [n for n, p in self._patterns if p.match(field)]
        if len(hits) > 1:
            raise ValueError(f"field {field!r} matches several sensor groups: {hits}")
        return hits[0] if hits else None

    def field_groups(self, field_names: Sequence[str]) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for i, f in enumerate(field_names):
            g = self.group_of(f)
            out.setdefault(g if g is not None else "__unclassified__", []).append(i)
        return out

    def offset(self, group: str, allow_unverified: bool = False) -> int:
        g = self.groups[group]
        if g["obs_offset"] is None:
            raise ValueError(f"group {group!r} has no transition offset (kind {g['kind']}); "
                             "it is indexed by sim_timestamp, use the obs window instead")
        if not g["status"].startswith("verified") and not allow_unverified:
            raise ValueError(f"group {group!r} alignment is {g['status']!r}; pass allow_unverified=True to use it")
        return int(g["obs_offset"])


@dataclass(frozen=True)
class WindowSpec:
    """Temporal contract of an AlignedWindows item (all indices are absolute episode steps).

    For a window starting at decision step t0 with horizon H and context C:
      transitions          t = t0 + k, k = 0..H-1   (action a_t, reward r_t)
      runtime obs          obs[t0-C+1 .. t0]        (C rows; obs[t0] is the decision observation)
      action history       a[t0-C .. t0-1]          (C rows; rows before the episode start are
                                                     zero-filled and flagged in action_history_mask)
      sensor target g      obs[t + d_g][fields_g]   (d_g from configs/sensor_alignment.json)
      obs window (target)  obs[t0 .. t0+H+L-1]      (H+L rows, L = max d_g; row j = obs[t0+j])
      privileged target p  p[t + o_p]               (o_p = 1 for tank_state, else 0)
    A window is valid iff t0 >= C-1 and t0+H-1+L <= T (obs has T+1 rows), so every target
    of the last transition exists. Windows never cross episodes.
    """
    horizon: int = 12
    context: int = 1
    sensor_targets: tuple[str, ...] = DEFAULT_SENSOR_TARGETS
    privileged_targets: tuple[str, ...] = ()
    effective_actions: bool = True
    stride: int = 1
    allow_unverified_sensors: bool = False

    def __post_init__(self):
        if self.horizon < 1 or self.context < 1 or self.stride < 1:
            raise ValueError("horizon, context and stride must be >= 1")
        unknown = set(self.privileged_targets) - set(PRIVILEGED_TARGETS)
        if unknown:
            raise ValueError(f"unknown privileged targets {sorted(unknown)}; allowed {sorted(PRIVILEGED_TARGETS)}")


class AlignedWindows:
    """Map-style dataset with explicit runtime/target separation.

    item = {
      "runtime_inputs": {"obs": [C,D] f32, "action_history_continuous": [C,n_c], "action_history_discrete": [C,n_d],
                         "action_history_mask": [C] bool, ("effective_action_history": [C,n_c] native)},
      "actions":        {"continuous": [H,n_c], "discrete": [H,n_d],
                         ("effective_native": [H,n_c], "effective_active": [H,n_c] bool)},
      "targets":        {"obs_window": [H+L,D], "sensors": {group: [H,n_g]}, "reward": [H],
                         "privileged": {name: [H,...]}   # only names listed in spec.privileged_targets
                        },
      "index":          {"episode_id", "scenario", "building", "t0"}   # bookkeeping, never a feature
    }
    `runtime_item(i)` returns only runtime_inputs + actions + index.
    """

    RUNTIME_SOURCES = ("obs", "action_continuous", "action_discrete")

    def __init__(self, episodes: Sequence[dict[str, Any]], scenario_metas: dict[str, dict[str, Any]],
                 spec: WindowSpec = WindowSpec(), alignment: SensorAlignment | None = None, schema=None):
        from runtime_features import default_schema
        self.spec = spec
        self.alignment = alignment or SensorAlignment.load()
        self.schema = schema or default_schema()
        sources = list(self.RUNTIME_SOURCES) + (["effective_action"] if spec.effective_actions else [])
        self.schema.validate(sources)          # the runtime path itself must be whitelisted
        for p in spec.privileged_targets:       # and privileged requests must really be targets
            if self.schema.classify(p).allowed:
                raise ValueError(f"{p!r} is a runtime source, not a privileged target")
        self.offsets = {g: self.alignment.offset(g, spec.allow_unverified_sensors) for g in spec.sensor_targets}
        self.max_offset = max(self.offsets.values(), default=1)
        self.max_offset = max(self.max_offset, 1)   # reward[t] pairs with at least obs[t+1]
        self.metas = scenario_metas
        self._layout: dict[str, dict[str, Any]] = {}
        self.episodes: list[dict[str, Any]] = []
        self.index: list[tuple[int, int]] = []
        self.skipped_short: list[str] = []
        H, C, L = spec.horizon, spec.context, self.max_offset
        for ep in episodes:
            a = ep["arrays"]
            T = int(a["reward"].shape[0])
            if a["obs"].shape[0] != T + 1:
                raise ValueError(f"{ep['episode_id']}: obs has {a['obs'].shape[0]} rows, expected {T + 1}")
            self._layout_for(ep["scenario"])
            first, last = C - 1, T - H - L + 1
            e = len(self.episodes)
            self.episodes.append({**ep, "T": T})
            if last < first:
                self.skipped_short.append(ep["episode_id"])
                continue
            self.index += [(e, t0) for t0 in range(first, last + 1, spec.stride)]

    # ------------------------------------------------------------ construction from disk
    @classmethod
    def from_root(cls, root: str | Path, rows: Sequence[dict[str, Any]], spec: WindowSpec = WindowSpec(),
                  cache_dir: str | Path | None = None, **kw) -> "AlignedWindows":
        root = Path(root)
        keys = ["obs", "action_continuous", "action_discrete", "reward"] + list(spec.privileged_targets)
        episodes, metas = [], {}
        for r in rows:
            path = root / r["path"]
            if cache_dir is not None:
                arrays = open_unpacked(unpack_episode(path, cache_dir), keys)
            else:
                loaded = load_episode(path, keys)
                arrays = {k: loaded[k] for k in keys}
            episodes.append({"episode_id": r["episode_id"], "scenario": r["scenario"],
                             "building": r["building"], "arrays": arrays})
            if r["scenario"] not in metas:
                metas[r["scenario"]] = load_scenario_meta(root, r["scenario"])
        return cls(episodes, metas, spec, **kw)

    def _layout_for(self, scenario: str) -> dict[str, Any]:
        if scenario not in self._layout:
            meta = self.metas[scenario]
            fg = self.alignment.field_groups(meta["observation_field_names"])
            missing = [g for g in self.spec.sensor_targets if g not in fg]
            if missing:
                raise ValueError(f"{scenario}: no observation fields for sensor groups {missing}")
            rep = None
            if self.spec.effective_actions:
                from effective_action import ActionRepresentation
                rep = ActionRepresentation(meta)
            self._layout[scenario] = {"fields": {g: np.asarray(fg[g]) for g in self.spec.sensor_targets},
                                      "rep": rep}
        return self._layout[scenario]

    # ------------------------------------------------------------ access
    def __len__(self) -> int:
        return len(self.index)

    def _runtime(self, ep, t0):
        a, C = ep["arrays"], self.spec.context
        obs = np.array(a["obs"][t0 - C + 1:t0 + 1], dtype=np.float32)
        hist_idx = np.arange(t0 - C, t0)
        mask = hist_idx >= 0
        safe = np.clip(hist_idx, 0, None)
        ac = np.array(a["action_continuous"][safe], dtype=np.float32)
        ad = np.array(a["action_discrete"][safe])
        ac[~mask] = 0.0
        ad[~mask] = 0
        out = {"obs": obs, "action_history_continuous": ac, "action_history_discrete": ad,
               "action_history_mask": mask}
        rep = self._layout[ep["scenario"]]["rep"]
        if rep is not None:
            eff = rep.effective(ac, ad).effective_native
            eff[~mask] = 0.0
            out["effective_action_history"] = eff
        return out

    def _actions(self, ep, t0):
        a, H = ep["arrays"], self.spec.horizon
        out = {"continuous": np.array(a["action_continuous"][t0:t0 + H], dtype=np.float32),
               "discrete": np.array(a["action_discrete"][t0:t0 + H])}
        rep = self._layout[ep["scenario"]]["rep"]
        if rep is not None:
            e = rep.effective(out["continuous"], out["discrete"])
            out["effective_native"] = e.effective_native
            out["effective_active"] = e.active
        return out

    def _index(self, e, t0):
        ep = self.episodes[e]
        return {"episode_id": ep["episode_id"], "scenario": ep["scenario"], "building": ep["building"], "t0": t0}

    def runtime_item(self, i: int) -> dict[str, Any]:
        """Inference-safe view: no targets, no privileged arrays."""
        e, t0 = self.index[i]
        ep = self.episodes[e]
        return {"runtime_inputs": self._runtime(ep, t0), "actions": self._actions(ep, t0), "index": self._index(e, t0)}

    def __getitem__(self, i: int) -> dict[str, Any]:
        e, t0 = self.index[i]
        ep = self.episodes[e]
        a, H, L = ep["arrays"], self.spec.horizon, self.max_offset
        lay = self._layout[ep["scenario"]]
        k = np.arange(H)
        sensors = {}
        for g, d in self.offsets.items():
            rows = t0 + k + d
            assert rows.max() <= ep["T"], "window validity invariant violated"
            sensors[g] = np.array(a["obs"][rows][:, lay["fields"][g]], dtype=np.float32)
        targets = {"obs_window": np.array(a["obs"][t0:t0 + H + L], dtype=np.float32),
                   "sensors": sensors,
                   "reward": np.array(a["reward"][t0:t0 + H], dtype=np.float64)}
        if self.spec.privileged_targets:
            targets["privileged"] = {p: np.array(a[p][t0 + PRIVILEGED_TARGETS[p]:t0 + PRIVILEGED_TARGETS[p] + H])
                                     for p in self.spec.privileged_targets}
        return {"runtime_inputs": self._runtime(ep, t0), "actions": self._actions(ep, t0),
                "targets": targets, "index": self._index(e, t0)}

    def public_metadata(self, scenario: str) -> dict[str, Any]:
        return self.schema.public_metadata(self.metas[scenario])

    def sensor_fields(self, scenario: str) -> dict[str, list[str]]:
        """Column names of targets["sensors"][group] for a scenario.

        Columns follow the OBSERVATION field order (e.g. VAV_1, VAV_10, VAV_11, ...), which is
        not the canonical natural-sorted zone order of meta["zones"]["ids"]; map explicitly."""
        names = self.metas[scenario]["observation_field_names"]
        return {g: [names[j] for j in idx] for g, idx in self._layout_for(scenario)["fields"].items()}

    def scenario_batches(self, batch_size: int, shuffle: bool = True, seed: int = 0) -> list[list[int]]:
        rng = np.random.default_rng(seed)
        by: dict[str, list[int]] = {}
        for i, (e, _) in enumerate(self.index):
            by.setdefault(self.episodes[e]["scenario"], []).append(i)
        batches = []
        for idx in by.values():
            idx = np.array(idx)
            if shuffle:
                rng.shuffle(idx)
            batches += [idx[j:j + batch_size].tolist() for j in range(0, len(idx), batch_size)]
        if shuffle:
            rng.shuffle(batches)
        return batches

    def as_torch(self, runtime_only: bool = False):
        import torch

        parent = self

        def conv(x):
            if isinstance(x, dict):
                return {k: conv(v) for k, v in x.items()}
            if isinstance(x, np.ndarray):
                return torch.from_numpy(np.ascontiguousarray(x))
            return x

        class _View(torch.utils.data.Dataset):
            def __len__(self):
                return len(parent)

            def __getitem__(self, i):
                return conv(parent.runtime_item(i) if runtime_only else parent[i])

        return _View()
