"""Reward / temperature / energy target distributions by building, weather and profile (read-only)."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dataset import load_episode, load_manifest, load_scenario_meta
DATA = ROOT / "data/phase1_sample"
rows = []
for r in load_manifest(DATA):
    ep = load_episode(DATA / r["path"]); zn = ep["names_zone_info"]
    z = ep["zone_info"]; occ = z[..., 5].sum(-1) > 0
    T = z[..., zn.index("zone_air_temperature")]
    dev = np.maximum(0, np.maximum(T - z[..., 1], z[..., 0] - T))
    elec = ep["ahu_info"][..., 0].sum(-1) + np.abs(ep["ahu_info"][..., 1]).sum(-1) + ep["boiler_info"][..., 1].sum(-1)
    gas = ep["boiler_info"][..., 0].sum(-1)
    K = r["num_ahus"]
    rows.append({"episode": r["episode_id"], "building": r["building"], "weather": r["weather"], "profile": r["profile"], "K": K,
                 "reward_mean": float(ep["reward"].mean()), "reward_p1": float(np.percentile(ep["reward"], 1)),
                 "reward_min": float(ep["reward"].min()), "reward_zero_frac": float(np.mean(ep["reward"] == 0)),
                 "zoneT_C_p1_p99": [float(np.percentile(T, 1) - 273.15), float(np.percentile(T, 99) - 273.15)],
                 "occupied_frac": float(occ.mean()),
                 "occ_dev_gt0_frac": float(np.mean(dev[occ] > 0)) if occ.any() else 0.0,
                 "occ_dev_gt1K_frac": float(np.mean(dev[occ] > 1)) if occ.any() else 0.0,
                 "occ_dev_max_K": float(dev[occ].max()) if occ.any() else 0.0,
                 "elec_per_K_kW_mean": float(elec.mean() / K / 1e3), "elec_per_K_kW_p99": float(np.percentile(elec, 99) / K / 1e3),
                 "gas_per_K_kW_mean": float(gas.mean() / K / 1e3), "gas_per_K_kW_p99": float(np.percentile(gas, 99) / K / 1e3),
                 "gas_zero_frac": float(np.mean(gas == 0)), "elec_zero_frac": float(np.mean(elec == 0)),
                 "outdoor_C_range": None,
                 "tank_T_range": [float(ep["tank_state"][:, 0].min()), float(ep["tank_state"][:, 0].max())],
                 "start": ep["meta"]["start_timestamp"]})
    sm = load_scenario_meta(DATA, r["scenario"])
    oa = ep["obs"][:, sm["ahus"]["obs_outside_air_temp"][0]]
    rows[-1]["outdoor_C_range"] = [float(oa.min() - 273.15), float(oa.max() - 273.15)]
import pandas as pd
df = pd.DataFrame(rows)
df["start_month"] = pd.to_datetime(df["start"], unit="s").dt.month
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
cols = ["reward_mean", "reward_min", "reward_zero_frac", "occupied_frac", "occ_dev_gt1K_frac", "occ_dev_max_K",
        "elec_per_K_kW_mean", "gas_per_K_kW_mean", "gas_zero_frac"]
print(df.groupby("profile")[cols].mean().T)
print(df[df.profile == "baseline"].groupby("building")[cols].mean().round(4))
df["station"] = df.weather.str.replace(r"_\d+$", "", regex=True)
print(df[df.profile == "baseline"].groupby("station")[cols].mean().round(4))
print(df[df.profile == "baseline"][["weather", "start_month", "outdoor_C_range"]].drop_duplicates("weather").to_string())
(ROOT / "scratch/phase1_audit/out/target_stats.json").write_text(json.dumps(rows, indent=1))
