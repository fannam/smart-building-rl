# Phase 2B: Counterfactual Validation, Graph-Aware Dynamics, Occupancy Forecasting and Uncertainty

**Status:** complete, with documented limitations (§13).

**Decision: NOT READY** for a closed-loop MPC, and **NOT READY** for an offline MPC prototype that optimizes actions through the learned model (§14).

All numbers below come from saved artifacts under `results/phase2b/`. Nothing was estimated or extrapolated.

---

## 1. Executive summary

**1. The Phase 2A GRU does not learn action-conditioned dynamics.**
- Phase 2B ran true paired simulator counterfactuals: 12 scenarios across all 6 buildings, 68 branch points, 46 interventions, a 3 h horizon, and bit-exact branches.
- The GRU's predicted zone-temperature response is **0–16 % of the simulator's** for every action family. The pooled least-squares gain is **−0.02 to 0.14**, where 1.0 is a correct response.
- On zone temperature it is barely better than a model that predicts *no response at all* (error ratio 0.85–1.05).
- A GRU that was **trained on the same building** is equally bad (gain ≤ 0.16). The failure is in action-response learning, not unseen-building generalization.
- Its good observational MAE (0.21 K on the nominal branches) hides this, which is exactly the failure mode the Phase 2B spec warned about.

**2. Which effects are reliable:**
- *Least unreliable:* the HWS-setpoint → gas/tank response (gas gain 1.00, tank 0.55) and the SAT → zone response (gain 0.14; it still under-responds).
- *Least reliable:* zone-level actions (damper, reheat valve, gain ≈ 0) and HWS run/dp.
- *Over-responses a planner would exploit:* SAT → tank (2.5×) and SAT → gas (1.5×).
- Responses are weakest in the most control-relevant states: morning recovery, occupied heating and cold weather (zone gain 0.01–0.02).

**3. The graph-aware model** uses typed zone/AHU/plant/building nodes and wall adjacency from the public `zone_map`, which was verified against the simulator geometry for every zone checked.
- **Unseen-building accuracy improves in all 6 LOBO folds and at every horizon** (pooled zone MAE at h36: 0.246 vs 0.307 K).
- The largest gains are on the weakest cases:
  - b2 (h12 zone 0.168 vs 0.243 K; heating recovery 0.35 vs 1.23 K; 3 h reward-sum error 0.13 vs 0.34);
  - the severe-cold scenarios;
  - the tank (0.30 vs 0.49 K).
- It has a similar parameter count (100k vs 99k) and costs 1.3–1.6× the GRU's CPU latency.
- **It does not fix the action response:** zone gains are ≤ 0.25, essentially the same failure. Its improvement is observational.

**4. Occupancy forecasting helps materially where it matters, and the Phase 2A explanation was incomplete.**
- Building-level occupancy coincides exactly with the comfort schedule except on **US public holidays**, which are not observable from the clock.
- The large Phase 2A deployable-reward errors come from holidays.
- A learned forecaster (gradient-boosted trees on the observable `num_occupants` history) cuts the 3 h reward-sum error with GRU physics:

  | Building | Schedule gate | Learned forecaster | Ceiling (true building occupancy) |
  |---|---|---|---|
  | b1 | 0.277 | 0.085 | 0.074 |
  | b4 | 0.125 | 0.042 | 0.042 |
  | b7 | 0.359 | 0.033 | 0.028 |

- It **fails on b2's holiday** (3.26 vs 0.145).
- Zone-level occupancy is not forecastable: zones are statistically exchangeable in the simulator's occupancy model.

**5. Uncertainty.** A 3-seed GRU ensemble (all 6 folds):
- *Ranking:* its spread ranks observational errors well (zone h12 Spearman 0.50–0.68; AUROC 0.72–0.94 for the worst decile; gas Spearman 0.89–0.98).
- *Calibration:* it is **under-dispersed**. Raw 2σ coverage is 45–69 %, and still only 70–90 % after a scale fitted on training-building validation data.
- *Missed failures:* at a train-fitted threshold it catches only 0–53 % of unreliable zone windows in held-out buildings.
- *Blind to the action-response failure:* all members share it (ensemble gains ≈ single-model gains). Ensemble disagreement therefore **cannot** be relied on to flag the errors that matter for a planner.
- *Other scores:* the Phase 2A action-Mahalanobis distance is uninformative (Spearman ≈ 0, AUROC ≈ 0.5). The GRU-latent distance is the best non-ensemble score (AUROC 0.64–0.94).

**6. The main blocker is data, not architecture.**
- The baseline policy almost never moves zone actuators: the damper sits at its 0.1 floor and reheat valves at 0 in ≥ 75 % of steps.
- Both models therefore learn state persistence plus schedule effects.
- A data-expansion proposal (§12) prioritises zone-level excitation and paired branch supervision. Nothing was collected.

**Engineering:**
- **Branch reproducibility is solved.** The Phase 0 "near-deterministic copy" was caused by *global* RNG streams (numba and CPython) that `deepcopy` does not copy. Snapshotting and restoring them makes branches bit-exact, verified on all 68 branch points.
- **Reproduction:** the Phase 2A GRU metrics reproduce exactly (Δ = 0.0 in all six folds).
- **Tests:** 61 tests pass (43 Phase 2A + 18 new).

---

## 2. Repository and implementation audit (Stage 0)

### Reused unchanged
- the temporal contract and `AlignedWindows` (`src/dataset.py`);
- the runtime whitelist (`configs/runtime_feature_schema.json`, `src/runtime_features.py`);
- effective actions (`src/effective_action.py`);
- LOBO folds (`configs/leave_one_building_out.json`);
- the Phase 2A featurizer, rollout, loss, training loop and evaluator (`src/dynamics/*`);
- the reward model;
- the collector's env setup (`collect_data.build_env`).

### Changed in Phase 2A code (behaviour-preserving)
Device support was added for GPU training. Every change is a `.to(x.device)` call or a `device=` argument, a no-op on CPU:
- `src/dynamics/model.py`, `src/dynamics/train.py`, `scripts/train_dynamics.py`;
- `--device/--workers/--threads` overrides (default: config, i.e. CPU);
- one-time cache unpacking before parallel jobs;
- checkpoints saved as CPU tensors.

Verified on CPU:
- energy predictions identical to the stored Phase 2A evaluation;
- zone MAE identical to float32 rounding;
- Phase 2A metrics reproduced exactly in all 6 folds.

### New code
| Path | Purpose |
|---|---|
| `src/counterfactual/branch.py` | `RNGState` capture/restore (numba np/py, NumPy global, CPython `random`); `env_copy` sharing the SuperLU factorization; `Branch.take/spawn` |
| `src/counterfactual/simrun.py` | simulator driver recording the Phase 1 quantities |
| `src/counterfactual/interventions.py` | 46 interventions; stratified branch-point selection |
| `src/counterfactual/model_eval.py` | model-vs-simulator paired-response metrics |
| `src/graph/topology.py`, `model.py`, `train.py` | heterogeneous topology, `GraphDynamics`, training and evaluation adapters |
| `src/occupancy/forecast.py` | runtime-only occupancy features and 4 forecasters |
| `src/uncertainty/scores.py` | ensembles, calibration, Mahalanobis/latent OOD scores, AUROC |
| `scripts/counterfactual_repro.py`, `counterfactual_run.py`, `counterfactual_eval.py` | branch reproducibility; simulation; model evaluation |
| `scripts/occupancy_forecast.py`, `train_graph.py`, `evaluate_phase2b.py` | occupancy; graph training; graph, ensemble and OOD evaluation |
| `configs/phase2b.json`, `configs/graph_dynamics.json` | Phase 2B configurations (Phase 2A configs unchanged) |
| `kaggle/phase2b_train.ipynb` | GPU training of the graph models and GRU seeds 1–2 (same protocol) |
| `tests/test_phase2b.py` | 18 tests (§9.2 list) |

### Phase 2A artifacts
These are read-only inputs and were not modified: the `results/phase2a/` checkpoints and metrics.

Phase 2A's fold-5 seeds 1–2 (trained ≥ 6 epochs, so v1.1-valid) are reused as ensemble members.

### Deviation: training hardware
- The local CPU was oversubscribed by the simulations (graph epochs took 15–20 min).
- The graph models and GRU seeds 1–2 (folds 0–4) were therefore trained on Kaggle GPUs (2 GPUs, round-robin).
- The protocol is otherwise identical; the device is recorded in every `train_log*.json`.
- GRU seed 0 (Phase 2A) and fold-5 seeds 1–2 were trained on CPU, so the ensembles mix devices.
- An earlier, interrupted local run's partial checkpoints are quarantined in `results/phase2b/_stale_local_partial/` (with a README) and are not used anywhere.

---

## 3. Counterfactual methodology

### Branching (offline only)
- **Setup:** per scenario, the unmodified baseline policy is run from reset. This is bit-identical to the stored Phase 1 baseline episode, checked at the end of every run.
- **Branch points:** at each branch point t_b a `Branch` is taken (env copy + global RNG snapshot). The original continues closed-loop for H = 36 steps (3 h); its actions are the **nominal** sequence.
- **Arms:** every arm spawns a copy, restores the RNG snapshot and replays the nominal actions open-loop with **one action family perturbed**.
- **Common random numbers:** all arms see identical random numbers, so differences are causal effects of the intervention.
- **Model inputs:** these are exactly what a deployed controller has at t_b:
  - observations o_{t_b−11..t_b};
  - the executed action history;
  - the arm's candidate actions (effective);
  - schedule/clock features;
  - outside air held at o_{t_b}.

### Scenarios and branch points
- **Scenarios:** 12, i.e. fairbanks + atlanta for each of the 6 buildings.
- **Strata:** morning recovery, occupied heating, occupied cooling, occupied other, unoccupied night, unoccupied other.
- **Branch points:** one per stratum where available, deterministic for seed 0 and ≥ 37 steps apart. That gives **68 branch points**, with outside air from −37.8 °C to 33.0 °C.
- **Arms:** 47 per point (1 nominal + 46 interventions), 3,196 simulated arm rollouts in total.

### Interventions
Each intervention is applied to every member of its family (all VAVs, all AHUs, or the plant).

| Family | Interventions |
|---|---|
| Reheat valve | ±0.25 / ±0.6 |
| Damper | ±0.2 / ±0.5 |
| SAT | ±2 / ±5 K |
| Static pressure | ±150 / ±500 Pa |
| Recirculation | ±0.15 / ±0.3 |
| HWS setpoint | ±5 / ±15 K |
| HWS dp | ±2 / ±6 |
| AHU run, HWS run | forced 1 / 0 |

- **Durations:** sustained (36 steps); the large magnitudes also run as a 6-step pulse.
- **Bounds:** native commands are clipped to the env bounds, and unperturbed entries stay bit-identical.
- **Commanded vs effective:** both are recorded. Arms whose effective action does not change (floors, caps, AHU off) are labelled **effective no-ops**: 924 model-arm pairs.

### Metrics
These are per (branch point, arm, quantity), computed on Δ = arm − nominal, pooling zones for zone temperature.

| Metric | Definition |
|---|---|
| Direction agreement | Over elements whose simulated peak \|Δ\| exceeds a threshold (0.02 K zone; 0.05 K tank; 50 W fan; 200 W cooling; 20 W pump; 500 W gas) |
| Pooled response gain | Σ Δ_model·Δ_sim / Σ Δ_sim² on horizon-mean responses; 1 = exact, 0 = no response, < 0 = wrong sign |
| Normalized error | Σ\|Δ_model − Δ_sim\| / Σ\|Δ_sim\|; equals 1 for a no-response model |
| Latency | First step reaching half the peak response |
| Spurious response | Predicted response when the simulator shows none |

**Expected-sign diagnostics vs counterfactual validation.** Phase 2A's probes were expected-sign checks on observational windows. Phase 2B replaces them with simulator ground truth: no physical sign assumptions are needed, and several "obvious" signs do not hold in the simulator.

---

## 4. Branch reproducibility

**Root cause.** The stochastic convection kernel draws from numba's global `np.random` state (and CPython `random` on another path). `deepcopy(env)` copies the env graph but not these global streams. Arms stepped one after another therefore consumed *different* random numbers.

The shared SuperLU factorization is innocent. It is a read-only cache of a constant matrix, refactorized only if the wind convection coefficient changes, which no shipped weather model does.

`results/phase2b/counterfactual/reproducibility/reproducibility.json` covers 4 buildings × 2 branch points:

| Check | Result |
|---|---|
| Fresh replay vs stored Phase 1 episode (636 steps, branches interleaved) | **bit-exact** (obs and reward max \|Δ\| = 0) |
| Copy + RNG restore vs original (36 steps) | **bit-exact**, all 8 |
| Copy vs copy (both restored) | **bit-exact**, all 8 |
| Copy without RNG restore (the Phase 0 workaround) | zone temperature differs by 3e-5 to 1.5e-4 K (noise floor of unpaired copies) |
| Copy time / RSS | b8 0.9–1.3 s / 54–65 MB; b2 2.0–2.8 s / 142–168 MB (under 12-process load) |

In the full experiment, **all 68 nominal arms were bit-exact replays of the original**, and the main trajectory still matched the stored Phase 1 episode exactly in all 12 scenarios.

Unit test: `TestBranchReproducibility`.

The blocker named in the Phase 2B spec is resolved.

---

## 5. Action-response validation results

The fold model that **holds out** each building is evaluated on that building ("unseen building"). `gru_indist` is a GRU from a fold that **trained on** the building, a diagnostic. Source: `results/phase2b/counterfactual/model_eval/summary.json`.

### 5.1 Zone temperature, pooled response gain (1 = exact)

| Family | Persistence | Linear | GRU | GRU, trained on the building | GRU 3-seed ensemble | Graph |
|---|---|---|---|---|---|---|
| reheat_valve | 0 | 0.15 | 0.04 | 0.02 | 0.03 | 0.01 |
| damper | 0 | 0.00 | −0.02 | −0.01 | −0.02 | 0.04 |
| sat | 0 | 0.26 | 0.14 | 0.16 | 0.13 | 0.25 |
| static_pressure | 0 | 0.19 | 0.05 | 0.05 | 0.04 | 0.12 |
| recirculation | 0 | 0.04 | 0.01 | 0.01 | 0.01 | 0.01 |
| ahu_run | 0 | −0.01 | 0.03 | 0.01 | 0.02 | 0.08 |
| hws_setpoint | 0 | 0.02 | 0.05 | 0.02 | 0.04 | 0.00 |
| hws_dp | 0 | 0.17 | 0.02 | 0.02 | 0.03 | 0.02 |
| hws_run | 0 | 0.05 | 0.00 | −0.00 | −0.01 | −0.00 |

**Normalized error** (Σ|Δm − Δs| / Σ|Δs|; a no-response model scores 1.00):
- GRU: 0.85 (sat) to 1.05 (damper).
- Graph: 0.75 (sat) to 1.07 (reheat valve).

**Direction agreement** among responsive zones:
- GRU: 0.30 (hws_run) to 0.83 (recirculation).
- Graph: 0.29 to 0.82.

Direction is often right; magnitude is not.

### 5.2 Plant and energy responses (pooled gain)

| Family → quantity | Linear | GRU | Graph |
|---|---|---|---|
| hws_setpoint → gas | 0.08 | **1.00** | **0.96** |
| hws_setpoint → tank | 0.34 | 0.55 | 0.75 |
| sat → tank | 0.09 | **2.54** (over-response) | **1.78** |
| sat → gas | −0.38 | **1.48** | **1.71** |
| reheat_valve → gas | 0.48 | 0.22 | 0.36 |
| hws_run → tank | **−1.38** (wrong sign) | −0.26 | 0.26 |
| static_pressure → fan | 0.34 | 0.34 | 0.55 |
| ahu_run → fan | 0.36 | 0.71 | 0.89 |
| hws_run → cooling | −95 | −21 | −0.30 |

The last row is driven by tiny simulated responses, so the gain is unstable there; the normalized error is the more meaningful number.

### 5.3 By condition (zone temperature, pooled gain)

| Stratum | Linear | GRU | Graph |
|---|---|---|---|
| Morning recovery | 0.06 | 0.02 | 0.03 |
| Occupied heating | 0.07 | 0.01 | 0.01 |
| Occupied cooling | 0.11 | 0.03 | 0.10 |
| Unoccupied night | 0.06 | 0.19 | 0.29 |

| Outdoor-air band | Linear | GRU | Graph |
|---|---|---|---|
| Cold (< −10 °C) | 0.06 | 0.01 | 0.01 |
| Cool | 0.10 | 0.01 | 0.02 |
| Mild | 0.17 | 0.03 | 0.12 |
| Warm (≥ 25 °C) | 0.02 | 0.11 | 0.26 |

| Building | Linear | GRU | Graph |
|---|---|---|---|
| b1 | 0.06 | −0.00 | 0.01 |
| b2 | 0.07 | 0.01 | 0.02 |
| b4 | 0.08 | 0.05 | 0.20 |
| b5 | 0.03 | 0.12 | 0.26 |
| b7 | 0.18 | 0.01 | 0.01 |
| b8 | 0.08 | 0.01 | 0.02 |

**Magnitude and duration.** The gain does not grow with the intervention magnitude; e.g. GRU SAT: small 0.12, large 0.14, 6-step pulse 0.19. The models are not merely saturating on large steps.

**Latency.**
- The simulator responds in 1–6 steps (median; reheat 1, damper 2, SAT 3, HWS setpoint 6).
- The GRU's (weak) response reaches half its peak at 5–17 steps.

**Effective no-ops.**
- 0 % spurious responses for every model. This is structural, since effective-action inputs make the model inputs identical; it is not evidence of learning.

**Absolute accuracy on the nominal branches** (mean |error| over the 3 h path):

| Model | Zone temperature (K) | Tank (K) |
|---|---|---|
| Persistence | 0.52 | 2.06 |
| Linear | 0.44 | 2.90 |
| GRU | 0.21 | 1.23 |
| GRU, trained on the building | 0.15 | 0.71 |
| Graph | 0.18 | 0.82 |

### 5.4 Diagnosis: capacity, coverage, alignment, or action learning?

- **Not alignment.** The temporal contract is tested (unit tests), the GRU reproduces Phase 2A exactly, and nominal-branch errors match the Phase 2A error levels.
- **Not unseen-building generalization.** The in-distribution GRU fails identically.
- **Not capacity alone.** A different architecture with explicit action routing (graph) fails identically. The linear model, with its far weaker fit, responds more (gain up to 0.26).
- **Coverage, i.e. action excitation, is the leading explanation.**
  - In the baseline data the damper sits at its 0.1 floor in ≥ 75 % of steps, valves are at 0 in ≥ 75 %, HWS dp is 0 in ≥ 50 %, and AHUs are off 60 % of the time.
  - The "moderate" exploration perturbs AHU/plant knobs, not zone actuators.
  - The actions the baseline does take are a deterministic function of the state (closed-loop). A flexible model can therefore attribute their effects to state and schedule features.
  - The best-excited knob (SAT) is the one with the largest learned response.

---

## 6. Occupancy forecasting

### What the simulator provides
- **Building total.** `num_occupants` in `obs[i]` equals Σ_z `zone_info[i−2].average_occupancy` *exactly* (verified; test `test_observed_total_alignment`).
- **Per-zone occupancy** is privileged and never observable.
- **Occupancy model:** `RandomizedArrivalDepartureOccupancy` gives every zone the same scalar `zone_assignment` (1 or 2 occupants) with identical arrival/departure windows. Zones are exchangeable, so the expected per-zone share is uniform and the realized split is not forecastable.
- **Scope:** **only building-level forecasting is implemented**, by design.

### Key finding
Building total > 0 coincides with the schedule's comfort mode at **every** step, except:
1. **US public holidays.** Zero occupants all day while the schedule still says "comfort": Jan 2, Jun 19, Jul 4, Sep 4, Nov 10, Nov 23 in the shipped scenarios.
2. The first 5-minute step after the schedule opens.

Holidays are not observable from the clock features, and `schedule.holidays` is empty. This corrects the Phase 2A interpretation, which attributed the error to "zones below setpoint before anyone arrives".

### Forecasters
All inputs are runtime only; whitelisted and tested, with poisoning tests for rows after t0.

| Forecaster | Description |
|---|---|
| Schedule-only | Train-mean level during comfort hours |
| Persistence | Last observed total |
| Time profile | Train mean per (workday, 15-min bin) |
| Learned | `HistGradientBoosting`; iterations chosen on the training-building inner validation |

The learned forecaster's features:
- horizon and the target clock;
- comfort mode and hours since open / until close;
- last, mean-12 and slope of observed N/V;
- the same time yesterday (observed only);
- the 24 h max;
- today's max since open;
- an "absent while open ≥ 30 min" flag.

Evaluation uses the Phase 2A test windows (context 12, horizon 36, stride 6), with identical LOBO folds.

### Building-total accuracy (held-out building, persons)

| Building | Schedule | Persistence | Time profile | Learned |
|---|---|---|---|---|
| b1 | 8.17 | 6.74 | 4.19 | **1.58** |
| b2 | 16.97 | 14.38 | 9.12 | **3.70** |
| b4 | 8.87 | 4.57 | 8.75 | **1.74** |
| b5 | 16.12 | 13.23 | 8.06 | **2.54** |
| b7 | 5.91 | 2.94 | 5.78 | **1.10** |
| b8 | 6.08 | 5.31 | 2.59 | **1.67** |

**Occupied-indicator accuracy:**
- learned: 0.987–1.000;
- schedule: 0.967–1.000 (it misses only holidays);
- persistence: 0.89.

**Arrival/departure detection** (within 30 min; the timing error is the median):
- schedule, time profile and learned: recall 0.96–1.00, timing error 0 steps;
- persistence: recall 0, because it never predicts a transition.

All metrics, including period and occupied/unoccupied breakdowns, are in `results/phase2b/occupancy/fold_*/metrics.json`.

### Deployable reward estimation (3 h reward-sum MAE)

Comfort weight = P̂(occupied) × uniform-zone comfort; the energy terms use the deployable calendar.

| Building (holiday windows) | Physics | Schedule | Learned (hard gate) | True building occupancy (ceiling) | True per-zone occupancy |
|---|---|---|---|---|---|
| b1 (108) | true | 0.221 | 0.032 | 0.012 | 0.013 |
| | GRU | 0.277 | **0.085** | 0.074 | 0.074 |
| b2 (108) | true | 0.144 | 0.114 | 0.009 | 0.004 |
| | GRU | 0.459 | 0.429 | 0.327 | 0.338 |
| b4 (108) | true | 0.090 | 0.007 | 0.004 | 0.003 |
| | GRU | 0.125 | **0.042** | 0.042 | 0.043 |
| b5 (108) | true | 0.009 | 0.003 | 0.003 | 0.002 |
| | GRU | 0.032 | 0.022 | 0.022 | 0.021 |
| b7 (216) | true | 0.362 | 0.016 | 0.003 | 0.002 |
| | GRU | 0.359 | **0.033** | 0.028 | 0.028 |
| b8 (0) | true | 0.005 | 0.005 | 0.005 | 0.004 |
| | GRU | 0.024 | 0.024 | 0.024 | 0.024 |

**Reward definitions:**
- **True simulator reward:** the reference.
- **Privileged reconstruction:** true per-zone occupancy plus the true calendar; matches the simulator (error 0).
- **Predicted occupancy:** the rows above.

**Holiday windows only** (GRU physics):
- schedule gate: 0.32–6.18;
- learned forecaster: 0.005–0.36, **except b2 at 3.26**, where it misses the holiday;
- true building occupancy: 0.004–0.15.

**Other observations:**
- *Soft vs hard:* the soft expected-value gate (p̂ × comfort) is slightly worse than the hard gate (p̂ ≥ 0.5) in b8, because of calibration near transitions.
- *Remaining error is physics:* once building occupancy is right, the remaining error is dynamics. The true-building-occupancy row is the ceiling, and the GRU-physics error there (b2: 0.33) is physics.
- *b2 persistence anomaly:* persistence scores *better* than true occupancy under GRU physics in b2 (0.19 vs 0.33). That is compensating errors, not a better forecaster.

---

## 7. Graph-aware model architecture

### Topology (`src/graph/topology.py`, public metadata only)
**Node types:** zone (V), AHU (K), plant/HWS (1), global (1).

**Edges:**
- zone↔AHU from `meta.zones.ahu`;
- zone↔plant for every VAV with a reheat valve (all of them);
- pooling to the global node;
- **zone↔zone wall adjacency** from `zone_map`.

**Zone adjacency details:**
- `zone_map` is in `create_building_meta_data` and is the geometry the baseline uses.
- Rooms are 4-connected interior components; two rooms are adjacent across a single wall cell.
- Weights are degree-normalized wall lengths.
- **Verified against the simulator's own room geometry: 30/30, 39/39 and 85/85 zones matched** (b8, b1, b2).
- Edge counts: 58–202 per building; mean degree 3.8–4.8.
- Floors are tiled side by side in the 2D map, so inter-floor adjacency is not represented and **not invented**.
- If the component count ≠ V, adjacency is marked unavailable (tested).

### Model (`src/graph/model.py`, 100,398 parameters; GRU 99,142)
**Encoders:** type-specific, hidden = 44.

**Actions enter only their own nodes:**
- damper and valve → zone node;
- run, SAT, pressure, recirculation → AHU node;
- HWS setpoint, dp, run → plant node.

**Typed messages:**
- AHU→zone, neighbour→zone (with the neighbour temperature difference), plant→zone, global→zone;
- zones→AHU (mean), zones→plant (mean), pooled→global.

**Updates and heads:**
- one GRU cell per node type; messages are summed;
- heads: zone ΔT (Phase 2A residual scale), tank ΔT from the plant node, fan/cooling per AHU node, pump/gas from the plant node.

**Interface:** the same `init_state/step` interface, so the unchanged Phase 2A `rollout` and evaluator drive it (tested for padding invariance across mixed buildings, edge-block isolation, and target/future-observation leakage).

### Training
Identical to Phase 2A v1.1:
- context 12, horizon 24, stride 4;
- Huber multitask loss;
- AdamW + one-cycle, lr 2e-3;
- ≤ 12 epochs, patience 3, min 6 epochs.

Graph models are seed 0 only.

---

## 8. LOBO comparison against Phase 2A

Held-out building, the Phase 2A test windows (19,680 pooled), deployable mode (outside air held at o_t0). Source: `results/phase2b/graph_dynamics/eval/fold_*/metrics.json`.

### Zone temperature MAE (°C), pooled (sd across folds)

| Model | h1 | h6 | h12 | h24 | h36 |
|---|---|---|---|---|---|
| Persistence | 0.065 (0.019) | 0.143 | 0.254 (0.059) | 0.454 | 0.635 (0.159) |
| Linear | 0.065 (0.018) | 0.146 | 0.254 (0.059) | 0.440 | 0.601 (0.138) |
| GRU (Phase 2A) | 0.045 (0.018) | 0.093 | 0.144 (0.052) | 0.231 | 0.307 (0.108) |
| **Graph** | **0.033** (0.013) | **0.074** | **0.113** (0.033) | **0.183** | **0.246** (0.065) |

### Per fold, zone MAE h1 / h12 / h36

| Held-out building | GRU | Graph |
|---|---|---|
| b1 (V39 K4) | 0.056 / 0.168 / 0.327 | 0.042 / 0.131 / 0.259 |
| **b2 (V85 K9)** | 0.078 / 0.243 / 0.518 | **0.056 / 0.168 / 0.358** |
| b4 (V54 K6) | 0.024 / 0.077 / 0.172 | 0.016 / 0.060 / 0.145 |
| b5 (V79 K8) | 0.028 / 0.110 / 0.227 | 0.024 / 0.102 / 0.205 |
| b7 (V36 K4) | 0.038 / 0.128 / 0.297 | 0.026 / 0.110 / 0.273 |
| b8 (V30 K3) | 0.046 / 0.138 / 0.304 | 0.031 / 0.110 / 0.237 |

The graph model is better than the GRU in **all 6 folds at every horizon**, and its fold-to-fold spread is smaller.

### Other targets, pooled (h12 unless stated)

| Model | Tank h12 / h36 (K) | Electricity (kW) | Gas (kW) |
|---|---|---|---|
| Persistence | 0.83 / 1.19 | 5.06 | 16.7 |
| Linear | 2.37 / 3.48 | 2.50 | 12.2 |
| GRU | 0.49 / 0.75 | 1.00 | 8.45 |
| **Graph** | **0.30 / 0.45** | **0.81** | **6.55** |

### Weak spots

**Severe-cold scenarios** (zone MAE h12 in °C / gas MAE h12 in kW):

| Building (scenario) | GRU | Graph |
|---|---|---|
| b1 (fairbanks_1) | 0.497 / 28.1 | 0.365 / 24.6 |
| b2 (fairbanks_2) | 0.422 / 145.9 | 0.301 / 100.1 |
| b8 (fairbanks_8) | 0.277 / 5.8 | 0.200 / 5.2 |

**Heating recovery** (zone MAE h12, °C):
- b2: GRU 1.231 vs graph **0.345**.
- Elsewhere the two are equal or the graph is better (b5: 0.134 vs 0.137).

**Moderate-exploration trajectories** (zone MAE h36, °C):
- the graph is better in every building (b2: 0.535 → 0.356).

**3 h reward-sum error** (exact-physics variant):
- b2: 0.342 → **0.129**; b4: 0.045 → 0.015.
- b1 (0.066 vs 0.078) and b7 (0.028 vs 0.031) are slightly worse for the graph.

**Plausibility:** the graph's zone predictions stay in −0.8 to 33.9 °C. The GRU's range is similar.

### Verdict
- **The graph model meaningfully improves unseen-building generalization.** The gains are largest on the hardest case (b2, largest K/V, cold, recovery) rather than only in pooled MAE.
- It does **not** improve the action-response failure (§5). The improvement is in observational dynamics.
- **Caveat:** one graph seed vs one GRU seed. The GRU seed spread (inner-validation loss 0.35–0.42 on fold 5; ensemble-vs-single zone h12 differences ≤ 0.04 K) is much smaller than the graph–GRU gap in most folds, but a graph ensemble was not trained.

---

## 9. Ensemble and uncertainty results

**Setup:**
- GRU: 3 seeds in all 6 folds.
- Graph: 1 seed only, so there are no graph-ensemble uncertainty results.
- Thresholds (95th percentile of the score) and the "unreliable" error level (90th percentile of error) are fitted on **training-building inner-validation windows only**.

Source: `metrics.json → uncertainty`.

### Ensemble-mean vs single-model zone MAE h12 (°C)

| Building | Ensemble | Single |
|---|---|---|
| b1 | 0.162 | 0.168 |
| b2 | 0.206 | 0.243 |
| b4 | 0.063 | 0.077 |
| b5 | 0.097 | 0.110 |
| b7 | 0.114 | 0.128 |
| b8 | 0.126 | 0.138 |

Averaging alone helps in every fold.

### Spread vs error, held-out building

| Target | Spearman | AUROC (top-decile error) | Error in lowest / highest spread decile |
|---|---|---|---|
| Zone h12 | 0.50–0.68 | 0.72–0.94 | e.g. b1 0.047 / 0.716 K |
| Zone h36 | 0.50–0.66 | 0.78–0.98 | — |
| Tank h12 | 0.42–0.61 | 0.88–0.97 | — |
| Electricity h12 | 0.71–0.88 | 0.88–0.94 | — |
| Gas h12 | 0.89–0.98 | 0.94–1.00 | — |

### Calibration and detection
**Calibration (zone h12, per-zone elements):**
- raw 2σ coverage 0.45–0.69 (nominal 0.95);
- after a train-validation scale (×1.7–2.4): 0.70–0.90;
- on the validation data itself: 0.83–0.87.

The ensemble is under-dispersed, and more so on unseen buildings.

**Detection at the training-fitted threshold (zone h12):**
- precision 0.57–0.90 where anything is flagged;
- **recall 0.00–0.44**;
- false-positive rate ≤ 0.02.

Few false alarms, many misses. In b4 and b5 the spread never exceeds the training threshold even though the errors are larger.

### Cold weather (folds with cold windows)
| Building (cold windows) | Zone h12 Spearman | AUROC | Recall |
|---|---|---|---|
| b1 (512) | 0.77 | 0.89 | 0.73 |
| b2 (462) | 0.53 | 0.76 | 0.40 |
| b8 (206) | 0.71 | 0.78 | 0.39 |

Gas detection in cold is strong for b1 and b2 (AUROC 0.99–1.00), weaker for b8 (0.75).

### The decisive limitation
On the counterfactual arms, the ensemble mean's response gain equals the single model's (§5.1: 0.02–0.13 vs 0.03–0.14). All members share the same under-response, so **ensemble disagreement does not reveal action-response errors**. That is the failure a planner would exploit.

---

## 10. Cold-weather and OOD analysis

OOD scores vs GRU zone h12 error, held-out building; the threshold is the training-validation 95th percentile.

| Score | Spearman | AUROC (unreliable) | Recall | False-positive rate |
|---|---|---|---|---|
| Action Mahalanobis (Phase 2A) | −0.08 to 0.28 | 0.46–0.67 | 0.03–0.35 | 0.05–0.22 |
| Decision-state Mahalanobis | 0.24–0.56 | 0.54–0.88 | 0.00–0.37 | 0.00–0.08 |
| GRU latent Mahalanobis | 0.31–0.62 | 0.64–0.94 | 0.06–0.48 | 0.02–0.09 |
| Ensemble spread | 0.53–0.64 | 0.76–0.98 | 0.05–0.50 | 0.00–0.02 |

**Conclusions:**
- *Action distance:* the Phase 2A action-distance score is **not useful**. Its ranking is near random, and it produces false positives at a rate comparable to its true positives.
- *Latent distance and ensemble spread:* both are informative rankings, but with train-fitted thresholds they miss most unreliable windows on unseen buildings.
- *Cold weather:* severe cold remains the largest error source (§8). Scores rank cold errors comparably to warm ones, but the training thresholds are too permissive there.
- *No score covers counterfactual errors:* none of these scores was shown to detect counterfactual (action-response) errors; the scores were evaluated on observational windows.

---

## 11. Runtime and resource benchmarks

**Setup:** CPU, 1 thread, `torch 2.15.0+cpu`, building b2 (V = 85, K = 9), context 12 + horizon 36 (47 steps). Source: `results/phase2b/reports/runtime_benchmark.json`.

| Model | Params | Rollout, batch 1 | Per step, batch 1 | Rollout, batch 64 | Per step, batch 64 |
|---|---|---|---|---|---|
| GRU | 99,142 | 14.4 ms | 0.31 ms | 195 ms | 4.1 ms |
| Graph | 100,398 | 22.5 ms | 0.48 ms | 248 ms | 5.3 ms |

**Resources:**
- Peak RSS of the benchmark process (including data loading): 794 MB.
- **GPU inference latency: not measured.** The local torch is CPU-only.

**Training cost on Kaggle GPU** (2 GPUs, 3 concurrent jobs):

| Model | Time per epoch | Time per run |
|---|---|---|
| Graph | ~7 min | 48–83 min |
| GRU | ~4 min | 20–46 min |

Local CPU under load: 15–20 min per graph epoch.

**Simulation and evaluation cost (local CPU):**
- Counterfactual simulation: 12 scenarios, 12 processes, 32–105 min per scenario (b2 slowest).
- Evaluation: about 6 min per fold.

---

## 12. Data expansion proposal (not executed)

The full table with purpose, target, intervention, expected information gain and risk is in `results/phase2b/reports/data_expansion_proposal.md`. Summary in priority order:

1. **E2 – reheat valve excitation** (heating-active, hot tank, AHU on).
2. **E1 – zone damper excitation** (independent per zone, ≤ 30 % of zones, 15–60 min holds).
3. **E9 – paired counterfactual supervision** via the bit-exact branching tool. About 50 branch points per building; it allows a direct paired-response loss.
4. **E6/E8 – severe-cold coverage and heating-recovery scenarios** (all buildings on fairbanks weather; vary the recovery start).
5. **E5 – large generated buildings** (V 70–120).
6. **E4 – HWS setpoint/dp** perturbations; **E7 – OOD joint combinations**; **E3 – wider SAT**.

**Constraints:**
- uses the existing comfort guard;
- new, separately named profiles and a new data root;
- the Phase 1 collector and exploration are unchanged;
- 1–3 day windows, no 30-day episodes;
- about 100–150 core-hours in total.

**Acceptance criteria:**
- per-family effective excitation ≥ 20 % of steps;
- guard trigger rate < 5 %;
- exact reward reconstruction.

---

## 13. Failure cases and limitations

1. **Action response:** the GRU, the graph model and the ensemble all predict ≈ 0–25 % of the true zone response (§5). This is the decisive failure.
2. **Plant over-responses:** SAT → tank/gas (×1.5–2.5) is exploitable by a planner.
3. **b2:** the graph model improves it substantially but it is still the worst building (h36 0.36 K). The GRU seeds again select epoch 0–1 there.
4. **Occupancy:** the forecaster misses b2's holiday (3 h reward error 3.26 on holiday windows).
5. **Uncertainty:** the ensemble is under-dispersed and has low recall on unseen buildings. It is blind to shared action-response errors.
6. **Scope limits:**
   - one graph seed (no graph ensemble, no seed-variance estimate for the graph–GRU gap);
   - the counterfactual study has 68 branch points × 46 interventions, i.e. one branch point per stratum per scenario, so per-building/per-stratum cells are small (e.g. b5/atlanta has 4 points);
   - interventions are building-wide per family (no single-zone interventions);
   - horizons ≤ 3 h;
   - outside air held constant in deployable rollouts.
7. **Device mix:** the ensembles combine CPU-trained (Phase 2A seed 0, fold-5 seeds 1–2) and GPU-trained members.
8. **Confidence intervals** are not reported for the LOBO comparison. With 6 folds, fold-level consistency (6/6 at every horizon) is the evidence used instead.

---

## 14. MPC readiness assessment

**Decision: NOT READY.**

A model with good observational MAE but unreliable counterfactual action responses must not be declared MPC-ready, and that is the measured situation:
- **Missing response:** both learned models predict roughly no zone response to the actions an MPC would optimize (reheat, damper, SAT, pressure, AHU scheduling). An optimizer using them would see almost no comfort benefit from heating actions while still paying their predicted energy cost.
- **Exploitable over-response:** the over-predicted SAT → tank/gas responses are exploitable.
- **No safety net:** the available uncertainty signals do not flag these errors.

This applies to a limited **offline** MPC prototype too, if it optimizes actions through the model. A planner-in-the-loop experiment would only measure model exploitation.

**What is ready:**
- **Graph model as a state predictor:** usable for open-loop prediction under baseline-like actions.
- **Occupancy forecaster:** usable as a reward-estimation component.
- **Branching tool:** bit-exact, ready for counterfactual supervision and evaluation.

---

## 15. Answers to the research questions and next steps

**1. Does the Phase 2A GRU learn reliable action-conditioned dynamics?**
No. The zone-temperature response gain is −0.02 to 0.14 across all action families, on seen and unseen buildings alike.

**2. Which predicted effects are most and least reliable?**
- *Most reliable:* HWS setpoint → gas (gain 1.00) and → tank (0.55); AHU run → fan (0.71); SAT → zone (0.14, under).
- *Least reliable:*
  - damper, reheat valve, recirculation and HWS dp/run → zone (≈ 0);
  - HWS run → tank (wrong sign);
  - SAT → tank/gas (2.5× / 1.5× over).

**3. Does the graph model improve unseen-building generalization?**
Yes, for observational prediction: 6/6 folds at every horizon, with the largest gains on b2, cold and recovery. It does not fix action response.

**4. Does occupancy forecasting materially improve deployable reward estimation?**
Yes, where holidays occur. 3 h reward-sum error: b1 0.277 → 0.085, b4 0.125 → 0.042, b7 0.359 → 0.033. It does not help in b8 (no holidays), and it fails on b2's holiday.

**5. Can uncertainty estimation identify unreliable predictions?**
Partially, for observational errors:
- ensemble spread and latent distance rank errors (AUROC up to 0.94), but they are under-dispersed with low recall on unseen buildings;
- action Mahalanobis is not useful;
- nothing detects the shared action-response error.

**6. Which failure modes remain unresolved?**
- action-response magnitude (the dominant one);
- exploitable plant over-responses;
- b2 and cold extremes;
- holiday detection in b2;
- uncertainty calibration.

**7. Is the current model ready for a limited offline MPC prototype?**
No (§14).

**8. What evidence is needed before a closed-loop MPC experiment?**
- **(a) Response gains:** pooled counterfactual zone gains in roughly 0.7–1.3 with correct sign (≥ 0.9 direction agreement) for every action the planner may move, in heating, recovery and cold strata, on held-out buildings. Measured with this same branching harness.
- **(b) No exploitable over-responses:** gain ≤ 1.3 on energy/tank channels.
- **(c) Uncertainty:** a signal shown to rise on counterfactual arms where the model is wrong. Evaluate ensemble spread *on the counterfactual arms*, ideally with members trained on diverse excitation.
- **(d) Offline planner check:** an offline planner-vs-simulator check, where planned actions are re-simulated through the branching tool and predicted vs realized reward is compared, before any closed-loop run.

**Recommended next steps (each needs approval):**
1. Run E2/E1/E9 of the data-expansion proposal (zone excitation + paired branch supervision).
2. Retrain the graph model, plus a GRU control, with an auxiliary paired-response loss on branch data.
3. Repeat Workstream A.
4. Train 3 graph seeds per fold for ensemble uncertainty and evaluate uncertainty on counterfactual arms.
5. Add a holiday-absence feature check for b2-like cases to the occupancy forecaster.

---

### Reproduction
```
.venv/bin/python scripts/counterfactual_repro.py --scenarios building_train_8/atlanta_1 building_train_2/fairbanks_2 building_train_4/denver_2 building_train_1/fairbanks_1 --branch-steps 150 600
.venv/bin/python scripts/counterfactual_run.py --parallel 12            # results/phase2b/counterfactual/sim
.venv/bin/python scripts/occupancy_forecast.py --parallel 6             # results/phase2b/occupancy
kaggle/phase2b_train.ipynb  (or: scripts/train_graph.py; scripts/train_dynamics.py --out results/phase2b/ensemble/gru --jobs 0:1 ... 4:2)
.venv/bin/python scripts/evaluate_phase2b.py --parallel 6               # results/phase2b/graph_dynamics/eval
.venv/bin/python scripts/counterfactual_eval.py --parallel 12           # results/phase2b/counterfactual/model_eval
.venv/bin/python -m unittest discover -s tests -t .                     # 61 tests
```

**Environment:**
- Python 3.12, numpy 2.0.2, scipy and numba as pinned by the starter kit, scikit-learn 1.6.1;
- torch 2.15.0+cpu locally; CUDA torch on Kaggle;
- starter kit `1fb7560`; seeds recorded in the configs and training logs.
