# Phase 1 Dataset — Schema and Collection Guide

> **Schema version:** `phase1-v1` · **Date:** 2026-11-18 · **Code:** `src/collect_data.py`, `src/exploration.py`, `src/building_meta.py`, `src/reward_model.py`, `src/dataset.py`
> **Validation:** `scripts/validate_phase1.py` → `results/phase1/validation.json`; reward study: `scripts/reward_reconstruction.py`.

## 1. What a sample is

One **episode** is one rollout of one scenario (building × weather) from `env.reset()`, for `steps` 5-minute steps (8,640 is the full 30 days). The collector builds the environment exactly as `src/run_eval.py` does, and drives it with the public baseline (`WideHotStorageReservePolicy`). The baseline may be wrapped in a seeded explorer.

For each step `t` the archive holds the transition `(o_t, a_t, r_t, o_{t+1})`:

- `o_t` / `o_{t+1}`: raw observation vectors, exactly as `env.reset()` / `env.step()` returned them (float32, simulator normalization untouched).
- `a_t`: the executed hybrid action, exactly the float32 / int32 arrays passed to `env.step()` (after `evaluate.validate_action`). The discrete part is stored as int8 (lossless for {0, 1}); cast it back to int32 before replaying.
- the **counterfactual baseline action** `b_t`: what the unperturbed baseline would have output from the same internal state and the same `o_t`. Compare it with `a_t` to tell baseline behaviour from randomized behaviour.
- `r_t`: the float reward returned by `env.step()`, plus every `RewardResponse` field and every `RewardInfo` physical quantity behind it.

## 2. Directory layout

```text
<root>/
  manifest.jsonl                         one JSON row per episode (append-only; last row per episode_id wins)
  collection_config.<unix>.json          argv / scenario list / seed of each collection run
  splits.json                            optional building-level split (dataset.write_split)
  scenarios/<building>__<weather>/
      meta.json                          static metadata (§4)
      zone_map.npy, floor_plan.npy       public geometry, copied from the scenario
  episodes/<building>__<weather>__<profile>__s<seed>__e<episode>.npz
```

Episode files are `np.savez_compressed` archives: about 0.47 MB per 1,000 transitions for the small building (V=30) and about 1.2 MB for V=85 (≈ 4–10 MB per full 30-day episode). The compression is dominated by the observation matrix. For training, `dataset.unpack_episode()` expands an archive once into uncompressed `.npy` files, which `TransitionWindows` then memory-maps.

## 3. Episode archive (`.npz`)

T = number of executed steps, D = obs dim, n_c / n_d = continuous / discrete action dims, V = zones, K = AHUs. **Zone order** is the canonical natural-sorted VAV order (`meta.zones.ids`), the same order the baseline uses. **Action order** is the simulator's own (`meta.continuous_action_names`, `meta.discrete_action_names`).

| Key | Shape / dtype | Content |
|---|---|---|
| `obs` | [T+1, D] f4 | o_0 … o_T, raw. `obs[t+1]` is the observation returned by `env.step(a_t)` |
| `action_continuous` | [T, n_c] f4 | executed continuous action in [-1, 1] |
| `action_discrete` | [T, n_d] i1 | executed discrete action in {0, 1}: AHU run bits in AHU order, then HWS run |
| `baseline_continuous`, `baseline_discrete` | as above | counterfactual baseline action (§5.3) |
| `perturbed` | [T] bool | `a_t != b_t` (any entry) |
| `reward` | [T] f8 | reward from `env.step()` |
| `reward_response` | [T, 17] f8 | all `RewardResponse` scalars; names in `names_reward_response` (includes `normalized_productivity_regret`, `normalized_energy_cost`, `normalized_carbon_emission`, `total_occupancy`) |
| `zone_info` | [T, V, 6] f4 | `RewardInfo` per zone: heating / cooling setpoint, **true** zone air temperature, flow setpoint, flow, **true per-zone average occupancy** (privileged) |
| `ahu_info` | [T, 1, 2] f4 | blower and air-conditioning electricity [W]. The simulator reports **one aggregate** for all AHUs (key `ahs`) |
| `boiler_info` | [T, 1, 2] f4 | natural-gas (burner heat) [W], pump electricity [W] (key `hws`) |
| `heat_pump_info` | [T, 0, 2] f4 | empty in all shipped scenarios |
| `sim_timestamp` | [T+1] i8 | simulator clock (epoch s, naive treated as UTC) before step t / after the last step |
| `reward_timestamp` | [T, 2] i8 | reward interval [start, end]. **start = sim_timestamp[t+1]** (Phase 0 §3) |
| `calendar` | [T, 6] i2 | hour, minute, weekday (Mon=0), month, day, is_workday at the reward start. Month, day and holidays are **privileged** |
| `tank_state` | [T+1, 6] f8 | privileged boiler/tank state: tank T [K], burner heat [W], step ΔT, return water T, boiler setpoint, HWS flow |
| `knobs` | [T, 4K+5] f4 | applied exploration knobs (§5), names in `names_knobs`; zero columns for `baseline` |
| `timing_ms` | [T, 2] f4 | policy time (including the counterfactual clone) and `env.step` time |
| `done` | [T, 2] bool | terminated, truncated |
| `names_*`, `keys_*` | str | column names and RewardInfo keys for the arrays above |
| `episode_json` | str | episode metadata: ids, seed, profile, exploration config, steps, start/end, score parts, perturbed fraction, timing, code hashes (git revs, file SHA of baseline / exploration / collector) |

### 3.1 Time alignment (verified by `validate_phase1.py`, check 2)

| Quantity | Equals |
|---|---|
| `sim_timestamp[t+1] − sim_timestamp[t]` | 300 s |
| `reward_timestamp[t, 0]` (the tariff hour of `r_t`) | `sim_timestamp[t+1]`, which is the hour / day-of-week decoded from `obs[t+1]` |
| action-echo fields in `obs[t+1]` (SAT, HWS setpoint and differential pressure, dampers, valves, run bits) | `a_t` (max error 1.6e-5 in native units) |
| zone temperature, zone heating / cooling setpoints and `num_occupants` in `obs[t+2]` | `zone_info[t]` / Σ occupancy behind `r_t` (**exact**) |

> **Correction (Phase 1.5, 2026-11-18).** Part of the paragraph below is wrong, and `configs/sensor_alignment.json` is now the authoritative per-group specification.
>
> - **Not every sensor is stale.** Only the **zone** sensors are: zone temperature, zone setpoints and `num_occupants` appear in `obs[t+2]`. AHU supply and outside-air flow, the HWS supply-water temperature (= true tank temperature) and all action echoes reflect transition t in `obs[t+1]`. Outside-air temperature and the clock features are current at `sim_timestamp[i]`.
> - **Effective actions.** Static pressure reads `run ? max(cmd, 200 Pa) : 0`. Recirculation is clamped to `[0, 0.7]` in **any** run state. Damper echoes show the position `max(cmd, 0.1)`. See `src/effective_action.py`.
> - **AHU `supply_air_temperature_sensor` is the setpoint echo,** not the delivered air temperature.
> - **`done`** is never set, even at full horizon. Episode ends are defined by the file.
> - **Training windows:** use `dataset.AlignedWindows`. The legacy `TransitionWindows` returns H+1 observations, so the zone target of its last reward is outside the window.
>
> The original text is kept below for traceability.

**Sensor values are one step stale; time features and action echoes are not.** The physical state that determines `r_t` (zone temperatures at the end of step t) first appears in `o_{t+2}`. Using `o_{t+1}` instead gives up to 6 K error at transitions. The sensor block of `o_{t+2}` depends only on actions up to `a_t`. At decision time t the policy sees `o_t`, whose sensors show the state at the end of step t−2 (equivalently, the start of step t−1). To score `r_t` the model rolls that state forward through `a_{t−1}` (already applied, known) and then `a_t`. This is the baseline's `observation_delay = 1` compensation. Two action echoes reflect *effective* rather than commanded values: static pressure reads 0 when the AHU is off, and recirculation is capped at 0.7.

Observable vs privileged: everything in `obs` and the actions is what a policy sees or does at run time. `zone_info`, `ahu_info`, `boiler_info`, `tank_state` and the month / day / holiday parts of `calendar` are **privileged training targets**. Use them as supervision (e.g. E, G and zone-temperature heads, tank-model fitting), never as run-time inputs.

## 4. Scenario metadata (`meta.json`)

- Everything `create_building_meta_data` provides (field names, index lists, AHU→zone assignments, schedule windows, holidays, normalization metadata).
- **Per-field observation normalization** reproduced from the env (`observation_mean`, `observation_std`, `observation_normalized`). The env normalizes by **exact measurement name** and otherwise passes the value through. Normalized fields are therefore only VAV zone temperature, damper / valve commands, supervisor run commands, HWS differential pressure and run status. AHU-prefixed fields, HWS supply-water temperature and flow setpoints stay **native**. `building_meta.ObservationCodec` decodes either kind.
- `num_occupants_normalization_constant`: the obs is `(N − c)/(c + 1)`, with `c` the env's *public* attribute. `run_eval.py` / `evaluate.py` set the private `_occupancy_normalization_constant = 125`, which the env never reads, so `c = 0` and the observation is the raw head count. The collector keeps that behaviour unchanged.
- Hybrid action layout: `continuous_action_names`, `discrete_action_names`, native low / high per continuous entry; `zones.*` and `ahus.*` give observation and action indices per zone and AHU in canonical order; `hws.*` likewise.
- `reward_config`: the env's reward constants, plus K and the per-AHU normalizers (always 80,000 and 205,000 W·K⁻¹ in the shipped set).
- `zone_area_m2`, `zone_exterior_ua_proxy`: the baseline's public geometry priors (from `zone_map`).
- `privileged.boiler`: tank parameters (energy balance, 1.5 m³, 0.5 K/min ramp, insulation, efficiency 1.0) and the episode start / end.

## 5. Exploration (`src/exploration.py`)

### 5.1 Profiles

| Profile | Policy class | Use |
|---|---|---|
| `baseline` | `WideHotStorageReservePolicy` (unmodified) | reference behaviour; deterministic, one episode per scenario |
| `null` | `ExplorationBaselinePolicy` + explorer with all probabilities 0 | regression check, must be bit-identical to `baseline` |
| `mild` | subclass + explorer | small excursions |
| `moderate` | subclass + explorer | default training exploration |

### 5.2 Knobs

All knobs are piecewise-constant. Each segment draws (active?, value, hold length); draw order and count are fixed and independent of observations.

| Knob | Where applied | moderate (mild) | Constraint / guard |
|---|---|---|---|
| AHU schedule | discrete run bit. Force OFF: only if the baseline runs it and `comfort_mode_now = comfort_mode_soon = 0`. Force ON: if the baseline has it off, with static pressure U[300, 900] Pa and SAT 291 K | p = 0.20 (0.10), hold 6–36 | never OFF during comfort or pre-start hours |
| SAT offset | `+U[−4, 4]` K (±2) on running AHUs | p = 0.5 (0.3), hold 12–72 | clip [285, 305] K |
| Static pressure | ×logU[1/1.8, 1.8] (1.3) on running AHUs | p = 0.5 (0.3), hold 12–72 | clip [0, 2000] Pa |
| HWS setpoint | `+U[−10, 10]` K (±5) | p = 0.5 (0.3), hold 24–144 | clip [310, 350] K; no negative offset while a zone is > 1 K below its heating setpoint during comfort / pre-start hours |
| Zone setback | inside the baseline's `_band`, only when unoccupied: low band +U[0, 14] K (8), high band −U[0, 8] K (4), plus per-zone N(0, 1 K) (0.5) jitter | p = 0.6 (0.4), hold 72–288 | keeps ≥ 1 K deadband; occupied and pre-start bands untouched |
| Recovery lead | baseline `precondition_hours` U[0.25, 3] h (0.5–2), held one day | p = 0.5 (0.3) | — |
| Comfort guard | per AHU: if any of its zones is > 1 K outside its sensed band during comfort / pre-start hours, its action-level knobs are suspended for the step (`guard_ahu_k = 1`) | — | causal (uses o_t only) |

The policy-internal knobs (setback, recovery lead) make the baseline aim for different targets while it stays closed-loop. The action-level knobs perturb its output. The baseline's internal observer therefore records what *it* commanded, not the perturbed SAT or pressure. That is intended for exploration, but it means the baseline's own model sees slightly inconsistent inputs in perturbed segments.

### 5.3 Counterfactual baseline action

When a policy-internal knob is active, the collector deep-copies the policy (about 0.3 ms; metadata shared), clears the knobs on the copy and records the copy's action from the same `o_t`. Otherwise the policy's own pre-perturbation output is the baseline action. Either way, `b_t` is "what the baseline would do now from the current (possibly perturbed) history", not the action of an independent unperturbed rollout.

### 5.4 Seeds

The knob schedule is drawn from `Generator(PCG64(SeedSequence([seed, crc32(scenario), episode])))`. The simulator itself is deterministic, so `(scenario, profile, seed, episode, steps)` fixes every array except `timing_ms`.

## 6. Splits

`dataset.building_split()` assigns **buildings**, never weathers or episodes, to train / val / test, so all 5 weathers and every episode of a building share a split. The default val building is `building_train_7` (K=4, V=36, close to the smallest-building design point). `leave_one_building_out()` gives 6 folds, and `assert_no_building_leakage()` guards custom splits. The plan (§7.6) still holds: once generated buildings exist, train on them and keep the 6 shipped buildings for validation.

## 7. Loading for PyTorch

```python
from dataset import TransitionWindows, episodes_for, read_split
import torch
rows = episodes_for("data/phase1_sample", "train", profiles=["baseline", "moderate"])
ds = TransitionWindows("data/phase1_sample", rows, horizon=12, stride=1, cache_dir="data/cache")
loader = torch.utils.data.DataLoader(ds.as_torch(), batch_sampler=ds.scenario_batches(64), num_workers=4)
batch = next(iter(loader))   # obs [B, 13, D] f32, action_continuous [B, 12, n_c] f32, reward [B, 12] f64, ...
```

Observation and action widths depend on the building, so `scenario_batches` never mixes scenarios in one batch. A size-agnostic per-zone or graph encoding is a Phase 2 modelling decision. `zones.*` / `ahus.*` in `meta.json` give all the indices it needs.

## 8. Reward reconstruction (`src/reward_model.py`)

`step_reward()` reproduces `SetpointEnergyCarbonRegretFunction` from (zone T, setpoints, per-zone occupancy, E, G, hour, work-day, month, K):

- E = Σ blower + |AC| + pump electricity; G = burner heat.
- K is read from metadata.
- The tariff, carbon and gas tables are verbatim copies; the validation asserts they equal the starter-kit constants.
- `month=None` uses the annual-mean gas price.

Use the **reward start hour**, which is the hour decoded from `o_{t+1}`, for tariff lookups.
