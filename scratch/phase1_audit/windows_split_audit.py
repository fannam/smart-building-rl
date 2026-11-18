"""TransitionWindows ordering/boundaries, scenario batching, and building-split integrity (read-only;
the .npy cache is written under scratch/phase1_audit/out/cache)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import torch
from dataset import (TransitionWindows, building_split, leave_one_building_out, assert_no_building_leakage,
                     episodes_for, read_split, load_manifest, ALL_BUILDINGS)
DATA = ROOT / "data/phase1_sample"; CACHE = ROOT / "scratch/phase1_audit/out/cache"
res = {}
split = read_split(DATA)
assert_no_building_leakage(split)
tr, va = episodes_for(DATA, "train", split), episodes_for(DATA, "val", split)
res["split"] = split
res["splits_json_present"] = (DATA / "splits.json").exists()
res["train_val_episode_overlap"] = len({r["episode_id"] for r in tr} & {r["episode_id"] for r in va})
res["train_val_scenario_overlap"] = len({r["scenario"] for r in tr} & {r["scenario"] for r in va})
res["train_val_building_overlap"] = len({r["building"] for r in tr} & {r["building"] for r in va})
res["val_weathers"] = sorted({r["weather"] for r in va}); res["n_train_eps"], res["n_val_eps"] = len(tr), len(va)
folds = leave_one_building_out(sorted({r["building"] for r in load_manifest(DATA)}))
res["lobo_folds"] = len(folds)
res["lobo_each_building_val_once"] = sorted(b for f in folds for b in f["val"]) == sorted(ALL_BUILDINGS)
for f in folds: assert_no_building_leakage(f)
try:
    building_split(ALL_BUILDINGS, val=["building_train_7"], test=["building_train_7"]); res["overlap_rejected"] = False
except ValueError:
    res["overlap_rejected"] = True
keys = ("obs", "action_continuous", "action_discrete", "reward", "sim_timestamp", "zone_info", "tank_state", "calendar")
H = 12
ds = TransitionWindows(DATA, tr + va, horizon=H, stride=1, cache_dir=CACHE, keys=keys)
expected = sum(e.T - H + 1 for e in ds.episodes)
res["windows"] = len(ds); res["windows_expected"] = expected
rng = np.random.default_rng(0); idx = rng.choice(len(ds), 2000, replace=False)
bad_order = bad_bound = bad_content = 0
for i in idx:
    w = ds[i]; e = ds.episodes[w["episode"]]; t0 = w["t0"]
    if not np.all(np.diff(w["sim_timestamp"]) == 300): bad_order += 1
    if t0 + H > e.T or w["obs"].shape[0] != H + 1 or w["reward"].shape[0] != H: bad_bound += 1
    if not (np.array_equal(w["obs"], e.arrays["obs"][t0:t0 + H + 1]) and np.array_equal(w["reward"], e.arrays["reward"][t0:t0 + H])
            and np.array_equal(w["action_continuous"], e.arrays["action_continuous"][t0:t0 + H])): bad_content += 1
res["sampled_windows_checked"] = len(idx)
res["windows_bad_time_order"], res["windows_crossing_episode_end"], res["windows_content_mismatch"] = bad_order, bad_bound, bad_content
# sensor target for reward[t0+H-1] lives in obs[t0+H+1], which a horizon-H window does not include
res["obs_rows_per_window"] = H + 1
res["rewards_in_window_whose_sensor_state_is_outside_window"] = 1
batches = ds.scenario_batches(64, seed=0)
flat = [i for b in batches for i in b]
res["batches"] = len(batches)
res["batches_single_scenario"] = all(len({ds.episodes[ds.index[i][0]].scenario for i in b}) == 1 for b in batches)
res["batches_cover_each_index_once"] = sorted(flat) == list(range(len(ds)))
loader = torch.utils.data.DataLoader(ds.as_torch(), batch_sampler=batches[:3], num_workers=0)
shapes = [{k: list(v.shape) for k, v in b.items() if hasattr(v, "shape")} for b in loader]
res["torch_batch_shapes"] = shapes
examples = []
for i in idx[:3]:
    w = ds[i]; e = ds.episodes[w["episode"]]
    examples.append({"episode": e.episode_id, "t0": int(w["t0"]), "first_ts": int(w["sim_timestamp"][0]),
                     "hours": w["calendar"][:4, 0].tolist(), "reward[:3]": w["reward"][:3].round(6).tolist(),
                     "sat_cmd_norm[:3,0]": w["action_continuous"][:3, 0].round(4).tolist()})
res["example_windows"] = examples
print(json.dumps(res, indent=1, default=str))
(ROOT / "scratch/phase1_audit/out/windows_split_audit.json").write_text(json.dumps(res, indent=1, default=str))
