# Phase 1.5–1.6 Implementation Report — Data Pipeline Hardening and Generalization Evaluation

> **Date:** 2026-11-18 · **Starting point:** `docs/phase1_audit_report.md` (READY WITH CONDITIONS)
> **Scope:** dataset API, sensor alignment, runtime-feature boundary, effective actions, LOBO split framework.
>
> **Not done, by instruction:**
> - no Phase 2 work, no model, no planner;
> - no new collection, apart from *no* new episodes at all;
> - no change to exploration, the collector, the baseline or the starter kit.

## 1. Summary of changes

- **Explicit temporal window contract.** `dataset.AlignedWindows` + `WindowSpec` provide per-sensor-group target offsets loaded from a machine-readable spec. They return H + L observations (H + 2 for zone targets) and never cross episodes. Runtime inputs, decision actions, targets and bookkeeping are separated; privileged targets are returned only when explicitly requested.
- **Sensor lags measured.** Every observation group with a privileged counterpart now has an exact, measured alignment, recorded in `configs/sensor_alignment.json`. Two groups have no counterpart (fan speed, request counts) and are labelled `unknown`.
- **Runtime-feature whitelist.** `configs/runtime_feature_schema.json` plus `src/runtime_features.py` classify every dataset source and every `meta.json` key, and reject privileged, restricted or unknown inputs.
- **Effective action representation.** `src/effective_action.py` derives effective values from source-verified rules that need only the commanded action and run bits. Commanded values are never overwritten, and the rules are checked against simulator echoes.
- **LOBO framework.** `src/generalization.py` + `scripts/evaluate_lobo.py` provide six deterministic folds, leakage checks, train-only fitted statistics with provenance, and descriptive diagnostics. Outputs are in `results/phase1_6/`; the report is `docs/phase1_6_generalization_report.md`.
- **Tests.** 36 `unittest` tests (no new dependency: pytest is not installed, and `unittest` is stdlib).

## 2. Files

**Created**

| File | Purpose |
|---|---|
| `configs/sensor_alignment.json` | sensor-group alignment spec (`sensor-alignment-v1`) |
| `configs/runtime_feature_schema.json` | runtime-input whitelist (`runtime-features-v1`) |
| `configs/leave_one_building_out.json` | fold definitions (`lobo-v1`) |
| `src/runtime_features.py` | `RuntimeFeatureSchema`, `validate_runtime_features`, `public_metadata`, `PrivilegedFeatureError` |
| `src/effective_action.py` | `ActionRepresentation`, `EffectiveAction`, `verify_against_echo` (`effective-action-v1`) |
| `src/generalization.py` | `make_lobo_folds`, `validate_fold`, `FittedStatistics`, `assert_provenance`, `episode_diagnostics`, `aggregate`, `coverage` |
| `scripts/evaluate_lobo.py` | runs the 6-fold split evaluation |
| `tests/test_dataset_alignment.py`, `test_aligned_real_data.py`, `test_runtime_features.py`, `test_effective_action.py`, `test_lobo.py`, `_paths.py`, `__init__.py` | tests |
| `results/phase1_6/**` | fold manifests, train statistics, diagnostics, smoke tests, `summary.json`, `summary.md`, `run_config.json` |
| `docs/phase1_6_generalization_report.md`, `docs/phase1_5_implementation_report.md` | reports |
| `scratch/phase1_5/sensor_lag.py` (+ `.json`), `run_validation.py`, `validation/`, `validation.log`, `cache/` | diagnostics and the regression run (isolated) |

**Modified**

| File | Change |
|---|---|
| `src/dataset.py` | **Added** `SensorAlignment`, `WindowSpec`, `AlignedWindows`, `PRIVILEGED_TARGETS`, `DATASET_API_VERSION = "aligned-windows-v1"`. `TransitionWindows` is unchanged in behaviour but now emits a `DeprecationWarning` and is documented as legacy. Loading, split and manifest functions are unchanged. |
| `docs/phase1_dataset.md` | §3.1 gains a correction note (stale-sensor generalization, effective actions, SAT-sensor meaning, `done`, loader). The original text is kept for traceability. |

**Unchanged** (verified):
- `src/collect_data.py`, `src/exploration.py` and `src/baseline_policy.py` have SHA-256 prefixes identical to those recorded in every sample episode (`11ea6d20…`, `854cb056…`, `897eadfb…`).
- `src/run_eval.py`, `src/baseline_policy.py` and `src/summarize.py` show an empty `git diff`.
- The starter kit, all datasets and the earlier reports are untouched.
- The raw `.npz` schema stays `phase1-v1`; no stored field is reinterpreted.

## 3. Temporal alignment contract (`WindowSpec`)

For a window starting at decision step `t0`, with horizon H, context C and maximum target offset L = max d_g (2 with the default zone targets, at least 1):

| Item | Absolute indices | Shape |
|---|---|---|
| transitions | t = t0 + k, k = 0..H−1 | — |
| `runtime_inputs.obs` | obs[t0−C+1 .. t0] (obs[t0] is the decision observation) | [C, D] |
| `runtime_inputs.action_history_*` | a[t0−C .. t0−1]; indices < 0 are zero-filled and `action_history_mask = False` | [C, n] |
| `runtime_inputs.effective_action_history` | effective(a[t0−C .. t0−1]) | [C, n_c] |
| `actions.continuous / discrete / effective_native / effective_active` | a[t0 .. t0+H−1] | [H, n] |
| `targets.sensors[g]` | obs[t + d_g][fields_g] | [H, n_g] |
| `targets.obs_window` | obs[t0 .. t0+H+L−1]; row j = obs[t0+j] | [H+L, D] (H+2 by default) |
| `targets.reward` | reward[t0 .. t0+H−1] | [H] |
| `targets.privileged[p]` (only if requested) | p[t + o_p], o_p = 1 for `tank_state`, else 0 | [H, …] |
| `index` | episode_id, scenario, building, t0 (bookkeeping; whitelisted as non-feature) | — |

- **Validity:** `t0 ≥ C−1` and `t0 + H − 1 + L ≤ T`, so every target of the last transition exists. Shorter episodes yield no windows and are listed in `skipped_short`.
- **Unindexed groups:** time-indexed groups (outside air, clock) have no transition offset and cannot be requested as sensor targets; they live in `obs_window`.
- **Unverified groups:** these are rejected unless `allow_unverified_sensors=True`.
- **Column order:** `targets.sensors[g]` columns follow the observation field order (VAV_1, VAV_10, …), not the canonical zone order. `AlignedWindows.sensor_fields(scenario)` returns the column names.

## 4. Sensor lag findings

Measured by `scratch/phase1_5/sensor_lag.py` over all 60 sample episodes, offsets d ∈ {−1, 0, 1, 2}, and explained from simulator source. "Exact" means float32 level, and in each case every other offset is off by O(range).

| Group (fields) | Alignment | Max abs error at that alignment | Mechanism (source) |
|---|---|---|---|
| zone temperature | obs[t+2] = zone_info[t] | 9.5e-7 K | `Vav.update_settings` writes the sensor at the start of the next step |
| zone heating / cooling setpoints | obs[t+2] | 0 | same |
| `num_occupants` | obs[t+2] = Σ average_occupancy[t] | 0 | 5-min trailing average |
| HWS supply-water temperature | obs[t+1] = tank_state[t+1] | 1.5e-5 K | tank after `apply_heat_draw` |
| AHU supply-air flow (sensor and "setpoint") | obs[t+1] = Σ zone flow[t] per AHU | 1.0e-4 m³/s | both read `AirHandler.air_flow_rate`; 0 when off |
| AHU outside-air flow | obs[t+1] = (1 − recirc_eff)·flow | 3.1e-7 | `ambient_flow_rate` |
| outside-air temperature | obs[i] = weather(sim_ts[i]) (current) | 1.4e-5 K (other k: ≥ 0.37 K) | evaluated at the observation timestamp |
| clock features | obs[i] ↔ sim_ts[i] | 1.4e-7 h | — |
| AHU SAT "sensor" and setpoint | obs[t+1] = commanded SAT | 1.7e-5 K | **the sensor returns the setpoint**; delivered SAT is unobserved |
| static pressure sensor / setpoint | obs[t+1] = run·max(cmd, 200) | 1.2e-4 Pa | AHU property |
| recirculation | obs[t+1] = clip(cmd, 0, 0.7) | 3e-8 | setter clamp, **any** run state (corrects the audit's "when running") |
| run bits (AHU supervisor / supply fan, HWS run status) | obs[t+1] | 0 | exhaust-fan field verified by source only |
| damper position | obs[t+1] = max(cmd, 0.1) | 3e-8 | observable is `damper_setting` (position after the ventilation floor) |
| reheat valve, HWS setpoints, HWS dp sensor | obs[t+1] = cmd | ≤ 1.8e-5 | echo |
| VAV `supply_air_flowrate_setpoint` | static (box maximum) | 0 | not the delivered flow |
| comfort_mode_now / soon | sim_ts[i] (+60 min) | — | **source only**, no privileged counterpart |
| AHU fan speed fields | **unknown** (source suggests obs[t+1], = flow/max, not zeroed when off) | — | needs an instrumented rollout logging AirHandler internals |
| cooling / heating request counts | **unknown** | — | same |

At decision time t:
- zone sensors show transition t−2;
- flows, the tank and echoes show transition t−1;
- outdoor temperature and clock show time t.

## 5. Runtime feature whitelist

`configs/runtime_feature_schema.json` sorts every source into one category:

| Category | Members |
|---|---|
| `runtime_observation` | `obs` |
| `action_history` | `action_continuous`, `action_discrete`, `effective_action` |
| `public_metadata` | 31 listed `meta.json` keys |
| `restricted` (opt-in only) | `baseline_*` |
| `privileged_target` | `reward`, `reward_response`, `zone_info`, `ahu_info`, `boiler_info`, `heat_pump_info`, `tank_state` |
| `privileged_episode` | `calendar`, `sim_timestamp`, `reward_timestamp`, `episode_json`; `meta.privileged`, `meta.scenario`, `meta.weather` |
| identifier | `meta.building` |
| `bookkeeping` | `knobs`, `perturbed`, `timing_ms`, `done`, `index`, `meta.schema_version` |

Anything unlisted is rejected.

Metadata investigation [verified on all 30 scenarios]:
- Only `meta.privileged.episode_start/end` contain dates, so they reveal month and day.
- `meta.scenario` and `meta.weather` identify the climate trace; `meta.building` identifies the building.
- `schedule.holidays` is public but **empty** in all 30 scenarios, and `time_zone` is `US/Pacific` everywhere.
- Public metadata is **identical across a building's five weather scenarios**, so it carries no episode or date information. Tested.

## 6. Privileged data protection

- `validate_runtime_features()` raises `PrivilegedFeatureError` on any privileged, restricted (unless opted in), unknown or nonexistent feature, and names every offender.
- `public_metadata()` drops every non-whitelisted key, including new, unclassified ones.
- `AlignedWindows`:
  - validates its own runtime sources against the schema at construction;
  - loads privileged arrays only if they are named in `WindowSpec.privileged_targets`, which must be real targets (asking for `obs` or `sim_timestamp` raises);
  - returns them only under `targets.privileged`;
  - offers `runtime_item()` and `as_torch(runtime_only=True)` views that contain no targets at all.
- Raw datasets are unchanged, so supervision is not lost.

## 7. Effective action representation

`ActionRepresentation(meta).effective(a_cont, a_disc)` works on any K and V and any leading batch dimensions.

- **Rules** (constants in all 30 configs: `min_static_pressure`, `max_recirculation`, `min_damper_setting`):

  | Action | Effective value |
  |---|---|
  | AHU static pressure | `run ? max(cmd, 200 Pa) : 0` |
  | AHU recirculation | `clip(cmd, 0, 0.7)` |
  | VAV damper position | `max(cmd, 0.1)` |
  | SAT, valves, HWS setpoints, run bits | identity |

- **Returns:** `commanded_native`, `effective_native`, `effective_normalized`, `discrete` (unchanged), `active` (False for entries of an AHU that is off, including its VAVs) and `rule`.
- **Units:** `*_native` arrays are in physical units (K, Pa, fraction); `*_normalized` arrays use the env's [−1, 1] map.
- **Runtime availability:** every rule needs only the command and its own run bits, so effective actions are available at decision time.
- **Not reconstructable at decision time, deliberately not provided:** delivered supply-air temperature (needs the unobserved mixed-air temperature) and actual VAV flow. The AHU total flow only appears in obs[t+1], *after* the decision, so it is a target, not an input.
- **Verification:** `verify_against_echo()` compares effective(a_t) with the obs[t+1] echoes. All 15 checked episodes pass, and a negative control with the wrong pressure floor is detected. At run time the same check can run one step late to detect an unseen building with different limits.

## 8. Leave-one-building-out protocol

Six deterministic folds (sorted buildings; fold *i* holds out building *i*). Each fold has 50 train / 10 test episodes, 100,800 / 20,160 transitions and 100,050 / 20,010 windows (H = 12, C = 4).

All folds pass:
- the leakage checks;
- the provenance check on the train-only statistics;
- a PyTorch smoke test: homogeneous batches, `obs_window` = H + 2, held-out windows only from the held-out building.

Details and descriptive statistics by building, station, climate band, K, V, profile and the coldest scenarios are in `docs/phase1_6_generalization_report.md`.

**Key limitations:**
- only 3 severe-cold scenarios (10% of transitions), one per cold fold;
- fold 0 extrapolates 14 K colder than its training data;
- folds 1 and 5 extrapolate in K and V;
- every K and V value except K = 4 comes from a single building.

## 9. Tests executed

| # | Test / command | Result |
|---|---|---|
| 1 | Phase 1 validation: `scratch/phase1_5/run_validation.py` (runs `scripts/validate_phase1.py` unchanged, outputs redirected to `scratch/phase1_5/validation/`) | **8/8 PASS**, including full-horizon baseline equality (mean −0.004031689621114378) and null ≡ baseline |
| 2 | `python -m unittest tests.test_dataset_alignment` (synthetic, index-encoded arrays): first / middle / last-valid transition, short episode, H+2 shapes, action/reward/obs alignment, no cross-episode, mixed offsets (0/1/2), unverified/unindexed groups rejected, privileged only on request, torch collate | **12/12 OK** |
| 3 | `tests.test_aligned_real_data`: on real b8/b2 episodes, zone target (obs[t+2]) = zone_info[t], HWS target = tank_state[t+1], occupancy target = Σ occupancy[t], AHU-flow target = Σ zone flow[t]; runtime item has no privileged keys. Negative control (zone offset 1 instead of 2): 6.17 K error vs 0.0000 K | **2/2 OK** |
| 4 | Sensor alignment diagnostics `scratch/phase1_5/sensor_lag.py` (60 episodes + weather controllers of 6 scenarios) | all measurable groups exact at one offset (§4) |
| 5 | `tests.test_runtime_features`: allowed features; 28 privileged features rejected alone and mixed; unknown and nonexistent features; restricted opt-in; error message; metadata filter; all real meta keys classified; public metadata identical across weathers; no dates in public metadata | **9/9 OK** |
| 6 | `tests.test_effective_action`: rule layout, AHU-on pressure floor, AHU-off, recirculation cap (on and off), damper floor, vectorization / shape errors, echo equality on 15 real episodes, wrong-limit negative control | **8/8 OK** |
| 7 | `tests.test_lobo`: deterministic and complete folds, weathers kept together, 5 injected leakage cases, provenance (leaked and stale), saved config / manifests / statistics | **5/5 OK** |
| 8 | PyTorch loading: `AlignedWindows.as_torch()` in tests 2 and 3 and in the per-fold smoke tests | OK |
| 9 | `scripts/evaluate_lobo.py`: smoke test for every fold | 6/6 OK |
| 10 | Regression: collector, exploration and baseline SHA prefixes equal those recorded in the sample episodes; `git diff` on tracked baseline files empty; `scripts/reward_reconstruction.py` still runs against the modified `dataset.py` (exact error ≤ 6.9e-9) | OK |

Run all unit tests with `.venv/bin/python -W ignore -m unittest discover -s tests -t .` (36 tests, about 1 s). Nothing was skipped: the real-data tests skip only if `data/phase1_sample` is absent, and it is present.

## 10. Known limitations

1. **Unknown sensor groups.** AHU fan speed and request counts have no verified alignment; they are excluded from targets by default. Verifying them needs an instrumented rollout (read-only logging of `AirHandler` internals per step).
2. **`comfort_mode_*`** alignment is from source only.
3. **Effective-action rules are constants** taken from the 30 shipped configs. Hidden buildings could differ; `verify_against_echo` can detect it one step late, but the controller would need to handle that case.
4. **The legacy `TransitionWindows` remains** (with a warning) for `validate_phase1.py`. New code must use `AlignedWindows`.
5. **`FittedStatistics` is an example preprocessing set**, group-level and size-independent. Phase 2 models must route any learned normalizer through the same provenance check.
6. **The audit's exploration-coverage findings are unchanged by design:** pressure perturbations below the floor are no-ops, forced-OFF is rare, VAV-level variation is small, and comfort excursions are sparse. The effective representation removes the *ambiguity* of below-floor commands but not the coverage gap.

## 11. Remaining risks before Phase 2

| Risk | Severity | Note |
|---|---|---|
| Cold-weather data: 3 severe-cold scenarios, one per cold fold, fold 0 out of range | **High** | cold results are n = 1 per fold; a cold-generalization claim needs more cold data (full-horizon cold scenarios and/or generated buildings; requires approval) |
| Building count: 6 buildings, one per K/V value | **High** | LOBO measures transfer between these 6 only; not evidence for generated hidden buildings |
| Exploration coverage (audit R5, R7, R8) | **High** | model may be unreliable off the baseline's action manifold; Phase 2 must keep a trust region and uncertainty penalty |
| 1 week per scenario | Medium | full-horizon collection adds seasons and cold hours cheaply (not done: forbidden here) |
| Unknown sensor groups | Low | excluded by default |
| Deployable reward bias (+1.1%, audit §8) | Medium | unchanged; cost-model concern for Phase 2 |

## 12. Readiness decision

**READY WITH CONDITIONS.**

Every item of the Phase 1 audit's conditions 1–4 is implemented and tested:
- an alignment-safe loader;
- an enforced input/target boundary;
- the LOBO protocol;
- effective actions.

Controlled dynamics-model experiments can start on this pipeline.

This is **not** "READY FOR PHASE 2" without conditions, because the data does not yet cover the operating conditions that matter most. Conditions for interpreting Phase 2 results:

1. Report every model metric per LOBO fold. List the severe-cold scenarios individually and flag them as n = 1; label folds 0 (temperature) and 1 / 5 (size) as extrapolation.
2. Make no claim of cold-weather or unseen-building generalization from this dataset. Treat it as a pilot until full-horizon and/or generated-building data exist (collection requires your approval).
3. Use `AlignedWindows`, `validate_runtime_features` and `FittedStatistics`-style provenance for every experiment, and model inputs from effective actions.
4. Exclude sensor groups marked `unknown` from targets until they are verified.
