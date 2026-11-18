"""Phase 1.6: six-fold leave-one-building-out split evaluation (no model is trained).

Writes
  configs/leave_one_building_out.json            fold definitions (deterministic; refuses to change silently)
  results/phase1_6/run_config.json               data root, profiles, seed, code versions
  results/phase1_6/diagnostics_per_episode.json  descriptive statistics per episode
  results/phase1_6/fold_<i>_<building>/          manifest_train.jsonl, manifest_test.jsonl,
                                                 train_statistics.json (fitted on train only),
                                                 diagnostics.json, smoke.json
  results/phase1_6/summary.json, summary.md

    .venv/bin/python scripts/evaluate_lobo.py --data data/phase1_sample
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import DATASET_API_VERSION, AlignedWindows, WindowSpec, load_manifest  # noqa: E402
from generalization import (FOLDS_VERSION, REQUIRES_MODEL, SEVERE_COLD_MIN_OAT_C, FittedStatistics, LeakageError,  # noqa: E402
                            aggregate, assert_provenance, coverage, episode_diagnostics, make_lobo_folds,
                            manifest_digest, split_rows, validate_fold)
from runtime_features import default_schema, validate_runtime_features  # noqa: E402


def write_folds(path: Path, folds, data_root: str, profiles, force: bool) -> dict:
    cfg = {"version": FOLDS_VERSION, "protocol": "leave-one-building-out",
           "assignment_rule": "buildings sorted lexicographically; fold i holds out the i-th building",
           "data_root": data_root, "profiles": sorted(profiles), "folds": folds}
    if path.exists():
        old = json.loads(path.read_text())
        if old != cfg and not force:
            raise SystemExit(f"{path} exists with different content; rerun with --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=1))
    return cfg


def smoke(data: Path, train_rows, test_rows, cache: Path, fold) -> dict:
    """Build AlignedWindows for both sides, pull one homogeneous torch batch each, check boundaries."""
    import torch
    spec = WindowSpec(horizon=12, context=4, privileged_targets=("zone_info", "boiler_info", "tank_state"))
    validate_runtime_features(["obs", "action_continuous", "action_discrete", "effective_action"])
    out = {}
    for side, rows in (("train", train_rows), ("test", test_rows)):
        ds = AlignedWindows.from_root(data, rows, spec, cache_dir=cache)
        allowed = set(fold["train_buildings"]) if side == "train" else {fold["test_building"]}
        bld = {ep["building"] for ep in ds.episodes}
        if not bld <= allowed:
            raise LeakageError(f"fold {fold['fold']} {side}: windows from {sorted(bld - allowed)}")
        batches = ds.scenario_batches(32, seed=0)
        loader = torch.utils.data.DataLoader(ds.as_torch(), batch_sampler=batches[:2], num_workers=0)
        shapes = None
        for b in loader:
            shapes = {"obs_window": list(b["targets"]["obs_window"].shape),
                      "runtime_obs": list(b["runtime_inputs"]["obs"].shape),
                      "actions": list(b["actions"]["continuous"].shape),
                      "effective": list(b["actions"]["effective_native"].shape),
                      "zone_temp_target": list(b["targets"]["sensors"]["zone_air_temperature"].shape),
                      "reward": list(b["targets"]["reward"].shape)}
            assert b["targets"]["obs_window"].shape[1] == spec.horizon + 2
            assert set(b["index"]["building"]) <= allowed
        rt = ds.runtime_item(0)
        assert "targets" not in rt and "privileged" not in json.dumps(list(rt))
        out[side] = {"episodes": len(ds.episodes), "windows": len(ds), "skipped_short": ds.skipped_short,
                     "buildings": sorted(bld), "first_batch_shapes": shapes}
    return out


def fmt(x, nd=2):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def markdown(summary) -> str:
    L = ["# Phase 1.6 — Leave-one-building-out split summary (generated)", "",
         f"Data: `{summary['data_root']}` · profiles: {', '.join(summary['profiles'])} · "
         f"{summary['episodes']} episodes / {summary['transitions']} transitions · generated {summary['generated']}", "",
         "All numbers are **descriptive dataset statistics**. No model was trained; model metrics "
         "(" + "; ".join(REQUIRES_MODEL) + ") are not reported.", "",
         "## Folds", "",
         "| fold | held-out building | K | V | train buildings | train eps / transitions | test eps / transitions | test min OAT °C | test OAT ⊂ train OAT | test K ⊂ train K range | severe-cold test scenarios | leakage checks |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in summary["folds"]:
        c = f["coverage"]
        L.append(f"| {f['fold']} | {f['test_building']} | {c['test_K']} | {c['test_V']} | {len(f['train_buildings'])} | "
                 f"{f['train_episodes']} / {f['train_transitions']} | {f['test_episodes']} / {f['test_transitions']} | "
                 f"{c['test_min_oat_c']:.1f} | {c['test_oat_within_train_range']} | {c['test_K_within_train_K_range']} | "
                 f"{', '.join(c['test_severe_cold_scenarios']) or 'none'} | {'pass' if f['leakage_ok'] else 'FAIL'} |")
    L += ["", "## Held-out building profile (test side of each fold)", "",
          "| building | K | V | min OAT °C | mean OAT °C | HDH (18 °C) per week | gas kWh/AHU/day | cooling kWh/AHU/day | occupied frac | comfort >0 K | comfort >1 K | max dev K | run frac | P_cmd<200 frac (running) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for b, a in summary["by_building"].items():
        g = lambda k, s="mean": a.get(k, {}).get(s)
        L.append(f"| {b} | {a['K']} | {a['V']} | {fmt(g('oat_c.min', 'min'), 1)} | {fmt(g('oat_c.mean'), 1)} | "
                 f"{fmt(g('heating_degree_hours_18c'), 0)} | {fmt(g('energy_kwh_per_ahu_per_day.gas'))} | "
                 f"{fmt(g('energy_kwh_per_ahu_per_day.cooling_electric'))} | {fmt(g('occupancy.occupied_frac'))} | "
                 f"{fmt(g('comfort_violation_frac.>0.0K'), 4)} | {fmt(g('comfort_violation_frac.>1.0K'), 4)} | "
                 f"{fmt(g('comfort_violation_max_k', 'max'))} | {fmt(g('actions.ahu_run_frac'))} | "
                 f"{fmt(g('actions.pressure_floor_binding_frac_running'), 3)} |")
    for title, key in (("Weather station", "by_station"), ("Climate band (mean OAT: cold <5 °C, mild 5–20, hot ≥20)", "by_climate_band"),
                       ("AHU count K", "by_K"), ("Zone count V", "by_V"), ("Profile", "by_profile")):
        L += ["", f"## By {title}", "",
              "| group | episodes | transitions | buildings | min OAT °C | mean OAT °C | gas kWh/AHU/day | cooling kWh/AHU/day | comfort >1 K | max dev K | SAT p50 K | P_eff p95 Pa | HWS p50 K |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for gname, a in summary[key].items():
            g = lambda k, s="mean": a.get(k, {}).get(s)
            L.append(f"| {gname} | {a['episodes']} | {a['transitions']} | {len(a['buildings'])} | {fmt(g('oat_c.min', 'min'), 1)} | "
                     f"{fmt(g('oat_c.mean'), 1)} | {fmt(g('energy_kwh_per_ahu_per_day.gas'))} | "
                     f"{fmt(g('energy_kwh_per_ahu_per_day.cooling_electric'))} | {fmt(g('comfort_violation_frac.>1.0K'), 4)} | "
                     f"{fmt(g('comfort_violation_max_k', 'max'))} | {fmt(g('actions.sat_cmd_k_running.p50'), 1)} | "
                     f"{fmt(g('actions.static_pressure_eff_pa_running.p95'), 0)} | {fmt(g('actions.hws_setpoint_k.p50'), 1)} |")
    L += ["", "## Coldest scenarios", "", "| scenario | building | min OAT °C | mean OAT °C | start month (privileged) | gas kWh/AHU/day (baseline) | comfort >1 K (baseline / moderate) |",
          "|---|---|---|---|---|---|---|"]
    for c in summary["coldest_scenarios"]:
        L.append(f"| {c['weather']} | {c['building']} | {c['min_oat_c']:.1f} | {c['mean_oat_c']:.1f} | {c['start_month']} | "
                 f"{fmt(c.get('gas_baseline'))} | {fmt(c.get('comfort1_baseline'), 4)} / {fmt(c.get('comfort1_moderate'), 4)} |")
    L += ["", f"Severe cold = scenario minimum OAT ≤ {SEVERE_COLD_MIN_OAT_C} °C.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/phase1_sample")
    ap.add_argument("--profiles", nargs="+", default=["baseline", "moderate"])
    ap.add_argument("--out", type=Path, default=ROOT / "results/phase1_6")
    ap.add_argument("--folds-config", type=Path, default=ROOT / "configs/leave_one_building_out.json")
    ap.add_argument("--cache", type=Path, default=ROOT / "scratch/phase1_5/cache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t_start = time.time()
    data = a.data.resolve()
    rows = [r for r in load_manifest(data) if r["profile"] in set(a.profiles)]
    buildings = sorted({r["building"] for r in rows})
    folds = make_lobo_folds(buildings)
    data_rel = str(data.relative_to(ROOT)) if data.is_relative_to(ROOT) else str(data)
    write_folds(a.folds_config, folds, data_rel, a.profiles, a.force)
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({
        "data_root": data_rel, "profiles": a.profiles, "seed": a.seed, "folds_config": str(a.folds_config.relative_to(ROOT)),
        "dataset_api": DATASET_API_VERSION, "folds_version": FOLDS_VERSION,
        "runtime_schema": default_schema().version, "argv": sys.argv}, indent=1))

    print(f"{len(rows)} episodes, {len(buildings)} buildings -> {len(folds)} folds", flush=True)
    diags = {r["episode_id"]: episode_diagnostics(data, r) for r in rows}
    (out / "diagnostics_per_episode.json").write_text(json.dumps(diags, indent=1))
    all_d = list(diags.values())

    fold_results = []
    for f in folds:
        tr, te = split_rows(rows, f)
        info = validate_fold(f, tr, te)
        stats = FittedStatistics.fit(data, tr)
        assert_provenance(stats, f, tr)
        fd = out / f"fold_{f['fold']}_{f['test_building']}"
        fd.mkdir(parents=True, exist_ok=True)
        for name, rs in (("manifest_train.jsonl", tr), ("manifest_test.jsonl", te)):
            (fd / name).write_text("".join(json.dumps(r) + "\n" for r in rs))
        (fd / "train_statistics.json").write_text(json.dumps(stats.to_json(), indent=1))
        dtr, dte = [diags[r["episode_id"]] for r in tr], [diags[r["episode_id"]] for r in te]
        cov = coverage(dtr, dte)
        diag = {"train": aggregate(dtr)["all"], "test": aggregate(dte)["all"],
                "test_by_weather": aggregate(dte, "weather"), "test_by_profile": aggregate(dte, "profile"),
                "coverage": cov}
        (fd / "diagnostics.json").write_text(json.dumps(diag, indent=1))
        sm = smoke(data, tr, te, a.cache, f)
        (fd / "smoke.json").write_text(json.dumps(sm, indent=1))
        fold_results.append({**f, "leakage_ok": info["ok"], "train_episodes": len(tr), "test_episodes": len(te),
                             "train_transitions": sum(r["steps"] for r in tr), "test_transitions": sum(r["steps"] for r in te),
                             "train_manifest_digest": manifest_digest(tr), "test_manifest_digest": manifest_digest(te),
                             "stats_provenance_buildings": stats.provenance["buildings"], "coverage": cov,
                             "smoke": {s: {k: v for k, v in sm[s].items() if k != "first_batch_shapes"} for s in sm},
                             "smoke_batch_shapes": sm["test"]["first_batch_shapes"]})
        print(f"fold {f['fold']} test={f['test_building']} train_eps={len(tr)} test_eps={len(te)} "
              f"test_minOAT={cov['test_min_oat_c']:.1f} cold_test={cov['test_severe_cold_scenarios']} smoke ok", flush=True)

    # coldest scenarios (per scenario, using baseline + moderate rows)
    by_scn = {}
    for d in all_d:
        s = by_scn.setdefault((d["building"], d["weather"]), {"building": d["building"], "weather": d["weather"],
                                                             "min_oat_c": d["oat_c"]["min"], "mean_oat_c": d["oat_c"]["mean"],
                                                             "start_month": d["start_month_privileged"]})
        s[f"gas_{d['profile']}"] = d["energy_kwh_per_ahu_per_day"]["gas"]
        s[f"comfort1_{d['profile']}"] = d["comfort_violation_frac"][">1.0K"]
    coldest = sorted(by_scn.values(), key=lambda s: s["min_oat_c"])[:8]

    summary = {"generated": time.strftime("%Y-%m-%d %H:%M"), "data_root": data_rel, "profiles": a.profiles,
               "episodes": len(rows), "transitions": sum(r["steps"] for r in rows),
               "folds": fold_results, "by_building": aggregate(all_d, "building"), "by_station": aggregate(all_d, "station"),
               "by_weather": aggregate(all_d, "weather"), "by_K": aggregate(all_d, "K"), "by_V": aggregate(all_d, "V"),
               "by_climate_band": aggregate(all_d, "climate_band"), "by_severe_cold": aggregate(all_d, "severe_cold"),
               "by_profile": aggregate(all_d, "profile"), "coldest_scenarios": coldest,
               "requires_model_not_computed": REQUIRES_MODEL, "wall_s": time.time() - t_start}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (out / "summary.md").write_text(markdown(summary))
    print(f"done in {summary['wall_s']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
