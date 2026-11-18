# Phase 1.6 — Leave-One-Building-Out Generalization Evaluation (split framework)

> **Date:** 2026-11-18 · **Data:** `data/phase1_sample` (profiles `baseline` + `moderate`; 60 episodes; 120,960 transitions; 1 week per scenario, seed 0)
> **Reproduce:** `.venv/bin/python scripts/evaluate_lobo.py --data data/phase1_sample` (about 7 s)
> **Machine-readable:**
> - `configs/leave_one_building_out.json` (folds)
> - `results/phase1_6/summary.json`
> - `results/phase1_6/diagnostics_per_episode.json`
> - `results/phase1_6/fold_<i>_<building>/` (manifests, train-only statistics, diagnostics, smoke test)
> - generated tables: `results/phase1_6/summary.md`
>
> **Scope:** split construction, leakage checks and **descriptive** dataset statistics. **No model was trained.** These metrics need a model and are *not* reported: one-step and multi-step zone-temperature RMSE, tank RMSE, E/G head error, reward error from predicted states, closed-loop score.

## 1. Protocol

- **Assignment.** Buildings are sorted lexicographically; fold *i* holds out the *i*-th building entirely (all 5 weather scenarios, both profiles). The rule is deterministic, and `scripts/evaluate_lobo.py` refuses to overwrite `configs/leave_one_building_out.json` with different content unless `--force` is given.
- **Leakage checks** (`generalization.validate_fold`, raise `LeakageError`). Each fold must have:
  - the held-out building absent from train;
  - only the held-out building in test;
  - no shared `episode_id`, `scenario` or file path;
  - both sides non-empty.

  They pass for all 6 folds. `tests/test_lobo.py` injects each kind of leak and checks that it is rejected.
- **Fitted statistics.** `FittedStatistics.fit` computes group-level mean/std over training episodes only: zone temperature, outdoor temperature, HWS temperature, occupants, SAT, effective static pressure, HWS setpoint. These are building-size independent. `provenance` records the buildings, episode ids and a manifest digest, and `assert_provenance` fails if the held-out building contributed. Verified for all 6 folds (saved in `train_statistics.json`; re-checked by `tests/test_lobo.py::TestSavedFolds`).
- **Windows.** `AlignedWindows` is built per side from the fold manifest. Windows never cross episodes, so they cannot cross folds. Per fold: 100,050 train and 20,010 test windows (H = 12, C = 4, zone targets at `obs[t+2]`).
- **Public metadata only** (`configs/runtime_feature_schema.json`). Public metadata is identical across a building's weather scenarios, so it carries no episode information.
- **Seeds and configuration:** `results/phase1_6/run_config.json` (data root, profiles, seed 0, dataset-API / runtime-schema / folds versions, argv). Per-fold manifests are saved as `manifest_train.jsonl` / `manifest_test.jsonl`.

## 2. Folds

| Fold | Held-out building | K | V | Train episodes / transitions | Test episodes / transitions | Test OAT range °C | Inside train OAT range? | Test K inside train K range? | Severe-cold test scenario | Leakage |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | building_train_1 | 4 | 39 | 50 / 100,800 | 10 / 20,160 | −39.4 … 46.1 | **no** (train min −25.6) | yes | fairbanks_1 | pass |
| 1 | building_train_2 | 9 | 85 | 50 / 100,800 | 10 / 20,160 | −25.6 … 42.2 | yes | **no** (train K 3–8, V 30–79) | fairbanks_2 | pass |
| 2 | building_train_4 | 6 | 54 | 50 / 100,800 | 10 / 20,160 | −5.6 … 33.7 | yes | yes | none | pass |
| 3 | building_train_5 | 8 | 79 | 50 / 100,800 | 10 / 20,160 | −2.8 … 33.9 | yes | yes | none | pass |
| 4 | building_train_7 | 4 | 36 | 50 / 100,800 | 10 / 20,160 | 1.6 … 36.7 | yes | yes | none | pass |
| 5 | building_train_8 | 3 | 30 | 50 / 100,800 | 10 / 20,160 | −16.7 … 42.2 | yes | **no** (train K 4–9, V 36–85) | fairbanks_8 | pass |

What the folds can and cannot test:

- **Size extrapolation:** folds 1 and 5 hold out the largest and smallest buildings, so they test extrapolation in K and V. Fold 5 is the design point for runtime (K = 3).
- **Cold extrapolation:** fold 0 tests extrapolation to colder weather than any training building has seen.
- **Within-range interpolation:** folds 2–4 test interpolation within the training ranges, but contain **no severe cold**.

## 3. Held-out building profiles

Descriptive; transition-weighted over the building's 10 episodes.

| Building | K | V | Min OAT °C | Mean OAT °C | HDH (18 °C) / week | Gas kWh/AHU/day | Cooling kWh/AHU/day | Occupied fraction | Occupied zone-steps > 0 K | > 1 K | Max deviation K | AHU run fraction | P_cmd < 200 Pa (running) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b1 | 4 | 39 | −39.4 | 13.1 | 1,839 | 195.8 | 39.1 | 0.37 | 0.0259 | 0.0015 | 4.53 | 0.46 | 0.106 |
| b2 | 9 | 85 | −25.6 | 14.0 | 1,626 | 79.8 | 27.6 | 0.37 | 0.0232 | 0.0011 | 7.89 | 0.46 | 0.111 |
| b4 | 6 | 54 | −5.6 | 18.4 | 664 | 16.8 | 25.3 | 0.37 | 0.0208 | 0.0000 | 0.94 | 0.43 | 0.117 |
| b5 | 8 | 79 | −2.8 | 19.8 | 506 | 17.8 | 21.8 | 0.37 | 0.0239 | 0.0000 | 1.52 | 0.42 | 0.108 |
| b7 | 4 | 36 | 1.6 | 19.0 | 576 | 42.5 | 21.9 | 0.35 | 0.0233 | 0.0000 | 0.78 | 0.41 | 0.121 |
| b8 | 3 | 30 | −16.7 | 18.2 | 1,056 | 73.0 | 35.6 | 0.39 | 0.0195 | 0.0002 | 4.60 | 0.47 | 0.112 |

- **Effective-action note.** About 11–12% of running AHU-steps command a static pressure below the 200 Pa floor. That is the `moderate` share averaged with the baseline, which never does. Effective pressure is therefore floored for those steps; the diagnostics report both commanded and effective pressure.
- **Comfort violations.** Violations above 1 K are confined to the buildings that have severe-cold scenarios (b1, b2, b8). The other three held-out buildings have essentially **no** comfort-violation examples to evaluate.

## 4. By weather station, climate band, K and V

From `results/phase1_6/summary.md`.

| Station | Episodes | Min OAT °C | Mean OAT °C | Gas kWh/AHU/day | Cooling kWh/AHU/day | Comfort > 1 K | Max deviation K | SAT p50 K | P_eff p95 Pa | HWS p50 K |
|---|---|---|---|---|---|---|---|---|---|---|
| atlanta | 12 | −1.7 | 18.3 | 25.2 | 19.6 | 0.0000 | 0.92 | 291.5 | 414 | 333.3 |
| denver | 12 | −5.6 | 13.1 | 41.8 | 10.1 | 0.0001 | 4.50 | 290.9 | 451 | 338.0 |
| fairbanks | 12 | −39.4 | −1.1 | 283.8 | 2.1 | 0.0022 | 7.89 | 289.0 | 406 | 344.2 |
| phoenix | 12 | 1.1 | 27.0 | 3.9 | 56.3 | 0.0000 | 1.51 | 288.2 | 415 | 321.2 |
| singapore | 12 | 23.6 | 28.3 | 0.0 | 54.6 | 0.0000 | 0.67 | 285.3 | 422 | 310.0 |

**Climate band** (mean OAT over the episode):

| Band | Episodes | Scenarios |
|---|---|---|
| cold (< 5 °C) | 8 | fairbanks_1, fairbanks_2, fairbanks_8, denver_8 |
| mild (5–20 °C) | 24 | 12 scenarios |
| hot (≥ 20 °C) | 28 | 14 scenarios |

The cold band carries 419.5 kWh/AHU/day of gas and the highest comfort-violation rate (0.0035 above 1 K).

**K and V:** each value of K except K = 4 (b1 + b7), and every value of V, comes from **one building**. Any per-K or per-V metric is therefore a single-building metric.

**Profile** (all buildings):

| Profile | Gas kWh/AHU/day | Comfort > 1 K | P_eff p95 Pa |
|---|---|---|---|
| baseline | 64.7 | 0.0003 | 212 |
| moderate | 77.2 | 0.0006 | 632 |

## 5. Cold-weather coverage

| Scenario | Building | Min OAT °C | Mean OAT °C | Start month (privileged) | Gas kWh/AHU/day (baseline) | Comfort > 1 K (baseline / moderate) |
|---|---|---|---|---|---|---|
| fairbanks_1 | b1 | −39.4 | −26.3 | Jan | 832.1 | 0.0054 / 0.0092 |
| fairbanks_2 | b2 | −25.6 | −13.4 | Feb | 281.2 | 0.0035 / 0.0061 |
| fairbanks_8 | b8 | −16.7 | −7.4 | Oct | 310.2 | 0.0000 / 0.0023 |
| denver_2 | b4 | −5.6 | 7.1 | Feb | 58.5 | 0.0000 / 0.0000 |
| denver_8 | b2 | −4.4 | 3.0 | Nov | 93.7 | 0.0000 / 0.0013 |
| denver_3 | b5 | −2.8 | 10.0 | Apr | 64.0 | 0.0001 / 0.0001 |
| atlanta_1 | b8 | −1.7 | 12.4 | Jan | 36.5 | 0.0000 / 0.0000 |
| phoenix_1 | b5 | 1.1 | 13.1 | Jan | 18.2 | 0.0000 / 0.0000 |

Severe cold means the scenario's minimum OAT is ≤ −10 °C. **The sample has only 3 such scenarios:** 6 episodes, 12,096 transitions, 10% of the data, in 3 buildings.

**Assessment: cold-weather coverage is insufficient for a complete cold-generalization evaluation.**

1. Only folds 0, 1 and 5 contain a severe-cold test scenario, and each contains exactly **one** (two episodes, both from the same week of weather). Any cold-weather test metric is an n = 1 scenario estimate.
2. In each of those folds, training sees only the other **two** severe-cold scenarios.
3. Fold 0's test minimum (−39.4 °C) is 14 K colder than anything in its training set, so it is an extrapolation test, not interpolation.
4. Folds 2–4 cannot say anything about cold-weather generalization.
5. Each scenario contributes only its first week. The coldest weeks of fairbanks_2 and fairbanks_8 (Feb and Oct starts) may lie outside the collected window. Full-horizon data would add cold hours without new buildings [inferred].

## 6. Reporting rules for Phase 2

Implemented in the diagnostics; to be followed by model metrics.

- Report every model metric per fold (held-out building). Never report only the 6-fold mean.
- Within each fold, break metrics down by station and by climate band, and list the severe-cold scenarios **individually**, flagged as n = 1.
- Label folds 1 and 5 as size extrapolation and fold 0 as temperature extrapolation.
- Fit every learned normalizer or preprocessing statistic with `FittedStatistics.fit(train_rows)` (or equivalent), and call `assert_provenance` before evaluation.
- Compare against the baseline's own one-step model on the same folds (audit §10).

## 7. What this evaluation does not establish

- Any prediction quality. No model exists.
- Generalization to *generated* buildings or unseen stations: all six buildings share the same five stations.
- A statistically meaningful per-K or per-V trend: one building per value, except K = 4.
