# Phase 2A — GRU Multi-Task Dynamics Model: Report

> **Date:** 2026-11-18 · **Config:** `configs/dynamics_gru.json` (protocol `phase2a_gru_v1.1`)
> **Reproduce:**
> ```
> .venv/bin/python scripts/train_dynamics.py --jobs 0:0 1:0 2:0 3:0 4:0 5:0 5:1 5:2 --parallel 8
> .venv/bin/python scripts/evaluate_dynamics.py --parallel 6
> ```
> **Artifacts:** `results/phase2a/`
> - per fold, in `fold_<i>_<building>/`: checkpoints `gru_seed*.pt`, `linear.pt`, `persistence.pt`; `normalizer.json` (train-only statistics with provenance); `fold.json`; training logs; `eval_results.npz`; `metrics.json`
> - across folds: `metrics.json`, `summary.md` (all tables), `plots/*.png`, `training_summary_consolidated.json`
>
> **Scope:** dynamics model and baselines only. No MPC, no planner, no GNN, no new data collection.

## 0. Bottom line

- **Zone temperature.** The GRU beats persistence and the linear autoregressive (AR) baseline in all six held-out buildings, at every horizon.

  | Pooled zone MAE | h = 1 (5 min) | h = 12 (1 h) | h = 36 (3 h) |
  |---|---|---|---|
  | GRU | 0.045 °C | 0.144 °C | 0.307 °C |
  | linear AR | 0.065 °C | 0.254 °C | 0.601 °C |
  | persistence | 0.065 °C | 0.254 °C | 0.635 °C |

- **Tank temperature and electricity:** large, consistent gains. Tank MAE at h = 12 is 0.49 K vs 0.83 (persistence) and 2.37 (linear); electricity MAE is 1.0 kW vs 2.5 kW (linear).
- **Gas:** better in 5 of 6 folds. In the largest building (fold 1, b2), the GRU is **worse than both baselines** (38 kW vs 27–33 kW at h = 12; 146 kW on its severe-cold scenario fairbanks_2). In that fold, validation-based early stopping selected the epoch-0 model because inner-validation loss rose from the first epoch.
- **Not yet MPC-ready:**
  - reheat-valve sensitivity has an inconsistent sign;
  - error explodes for actions beyond the training distribution;
  - the deployable reward estimator, specifically the occupancy gate, dominates reward error.
- **Decision: READY WITH CONDITIONS** (§17).

## 1. Architecture (`src/dynamics/model.py`)

Size-agnostic and shared across buildings. It uses no graph message passing, only DeepSets-style mean pooling.

**Zone GRU cell** (hidden 64), shared by every zone. Input per zone per step:
- zone features (9): normalized zone temperature, log area and exterior-UA proxy, effective damper and valve for a_{s−1} and a_s, setpoint window at s+1 relative to the zone temperature;
- its own AHU's action features (8);
- building features (17);
- the masked mean of the previous zone latents;
- the building latent.

**Building GRU cell** (hidden 64). Input: building features, the masked mean of the new zone latents, and the mean AHU action features.

**Heads:**

| Head | Output |
|---|---|
| zone | ΔT, so that T̂(after s) = T_in(after s−2) + σ_ΔT · Δ |
| tank | ΔT_HWS, so that T̂_HWS(after s) = T_HWS,in(after s−1) + σ_ΔH · Δ |
| per-AHU energy (on the AHU-pooled zone latents, AHU actions, building latent) | fan and cooling electricity, softplus, **summed over AHUs** (the simulator reports only the aggregate) |
| plant | pump electricity and gas burner heat, softplus |

- **Energy scaling:** energy heads output per-AHU-normalized power (W / (K·80 kW), gas / (K·205 kW), i.e. the reward's own normalizers). They are scaled by the training-mean so that initialization is well conditioned.
- **Batching:** mixed-building batches with padding masks for zones and AHUs.
- **Size:** 99,142 parameters.

## 2. Runtime inputs (whitelisted)

`dynamics.features.WindowFeaturizer.features_used` is validated by `validate_runtime_features` at construction. It contains:

- `obs:` zone temperature sensors, the HWS supply-water sensor, AHU 1's outside-air sensor, and `hod/dow` sin/cos;
- `effective_action` and `action_discrete`;
- `meta:` zones, ahus, hws, schedule, zone_area_m2, zone_exterior_ua_proxy, normalization, field names, K, V.

It never uses `zone_info`, `tank_state`, `ahu_info`, `boiler_info`, `calendar`, timestamps or per-zone occupancy. `num_occupants` is deliberately **not** an input: its future value cannot be forecast at run time, and using it in teacher-forced steps only would make teacher-forced and autoregressive inputs inconsistent.

**Forecastable exogenous inputs:**
- Clock: exact arithmetic from o_t0's clock.
- Comfort mode (now / +60 min) and setpoint window: computed from metadata schedule hours and windows. Verified **exact** against the observed `comfort_mode_now/soon` (0 / 121,020 mismatches) and against `zone_info` setpoints at `sim_ts[t+1]` (0 / 13.0 M).
- Outside air: held at its o_t0 value in rollouts (deployable). An "oracle OAT" variant using true future outside air is reported as a labelled diagnostic only.

**Effective actions:** `effective_action.py` (pressure floor, recirculation cap, damper floor). Commanded values are not used directly.

## 3. Targets (privileged, training/evaluation only)

| Target | Source | Physical meaning |
|---|---|---|
| zone temperature after transition t | `zone_info[t].zone_air_temperature` (= obs[t+2] sensor) | K |
| tank temperature after t | `tank_state[t+1].tank_temperature_k` (= obs[t+1] HWS sensor) | K |
| fan electricity | Σ `ahu_info[t].blower_electrical_energy_rate` | W |
| cooling electricity | Σ \|`ahu_info[t].air_conditioning_electrical_energy_rate`\| | W |
| pump electricity | `boiler_info[t].pump_electrical_energy_rate` | W |
| gas | `boiler_info[t].natural_gas_heating_energy_rate` (burner heat, 100% efficiency) | W |
| reward (diagnostic) | `reward[t]`; reconstruction uses `zone_info` setpoints/occupancy and `calendar` | — |

E = fan + cooling + pump, exactly as `SetpointEnergyCarbonRegretFunction` sums it, so nothing is double-counted. G = burner heat. No auxiliary heads were added: occupancy forecasting would serve the reward estimator, and is recommended for Phase 2B (§16).

## 4. Temporal alignment (`src/dynamics/features.py`, tests in `tests/test_dynamics.py`)

Window from `AlignedWindows(WindowSpec(horizon=H, context=C))`. Steps j = 0..C−2+H; absolute s = t0−C+1+j; action rows A[i] = a[t0−C+i].

| Prediction target (step s) | Runtime information used at step s | Supervision target | Offset |
|---|---|---|---|
| zone T after s | zone T after s−2: obs[s] sensors if s ≤ t0, else the model's own prediction from step s−2; a_{s−1}, a_s; setpoint window at s+1; clock and comfort at s | `zone_info[s]` | sensor +2 |
| tank T after s | tank after s−1: obs[s] HWS sensor if s ≤ t0, else own prediction from step s−1; HWS actions a_{s−1}, a_s | `tank_state[s+1]` | sensor +1 |
| fan / cooling / pump electricity and gas of transition s | the latents above + a_s | `ahu_info[s]`, `boiler_info[s]` | 0 |
| reward of transition s (diagnostic) | predicted physics + schedule / clock at sim_ts[s+1] | `reward[s]` | tariff hour = sim_ts[s+1] |

- **Warm-up steps (s < t0) and the decision step (s = t0)** receive true observations.
- **Rollout steps (s > t0)** receive **no observation**: the batch for future steps contains only actions and forecastable exogenous inputs. `features.runtime_part()` reads only `runtime_inputs` and `actions`.

Tests:
- A recording dummy model shows that rollout steps get the prediction of step j−2 (zone) and j−1 (tank), and the outside air held at o_t0.
- NaN-poisoned targets never propagate in deployable mode; oracle mode changes only outside air.
- The featurizer matches absolute indices on real windows (zone target = obs[t+2], tank = obs[t+1], damper row C+k = effective a_{t0+k}, clock and setpoints).
- `runtime_part` is unchanged when every privileged target is set to NaN.
- Persistence holds o_t0.

All 43 repository tests pass.

**Rule for the linear baseline:** its "previous state" feature uses the observed sensor while it exists (warm-up) and its own prediction afterwards.

## 5. Training objective (`src/dynamics/train.py`)

**Multi-step rollout loss:** C = 12 warm-up steps, then H = 24 open-loop steps, with the loss summed over all 24.

L = 1.0·Huber(ΔT_zone / σ_ΔT) + 0.5·Huber(ΔT_tank / σ_ΔH) + 0.5·Σ_{fan,cool,pump} Huber(e / σ_e) + 0.5·Huber(g / σ_g)

- Huber with δ = 1 in normalized units: quadratic for typical errors, linear for rare large transients (recovery, burner switching), so those don't dominate.
- Zone loss is masked over padded zones.
- σ values are fitted on **training buildings only** (`fit_normalizer` → `FittedStatistics` with provenance; `assert_provenance` is checked per fold).
- Each task is logged separately; the weights are configurable.
- No target is zero-filled: every target exists for every valid window (AlignedWindows validity).
- Reward is not a training objective. It is a diagnostic (§13).

## 6. Data and splits

- **Data:** `data/phase1_sample`, profiles baseline + moderate: 60 episodes (1 week each), 6 buildings, 30 scenarios.
- **Folds:** six-fold LOBO from `configs/leave_one_building_out.json`. Each fold trains on 50 episodes from 5 buildings and tests on 10 episodes of the held-out building.
- **Inner validation (early stopping):** the last 20% of every **training** episode (time split). The held-out building is never used for training, early stopping, normalization, or linear / persistence fitting.

| Window set | Context C | Horizon H | Stride | Count |
|---|---|---|---|---|
| train | 12 | 24 | 4 | 19,700 per fold |
| inner validation | 12 | 24 | 8 | 2,350 |
| test | 12 | 36 | 6 | 3,280 per fold |

## 7. Training configuration and compute

- **Optimizer:** AdamW, peak lr 2e-3 on a one-cycle schedule, weight decay 1e-4, batch 32 windows, gradient clip 1.0.
- **Epochs:** at most 12, patience 3, `min_epochs` 6 (see below).
- **Hardware:** CPU only: 8 jobs in parallel × 3 torch threads on an i9-14900HX (32 threads).
- **Cost:** about 6 min per epoch; 25–78 min per fold job; about 1.2 GB RAM per job. Evaluation takes about 2 min per fold. Total wall time is about 1 h 20 min of training plus about 3 min of evaluation.
- **Determinism:** the fold-1 retrain reproduced the earlier epoch-0 to 3 validation losses exactly.

**Protocol v1 → v1.1.** v1 used patience 3 without a minimum. With one-cycle LR this can stop during the high-LR phase, before annealing. v1.1 adds `min_epochs = 6` globally. It is identical to v1 for every run that trained at least 6 epochs (all but fold 1), and fold 1 was retrained under v1.1. Its inner-validation loss rose monotonically (temperature term 0.86 → 1.17 while training loss fell 1.20 → 0.18), so v1.1 still selects **epoch 0**. Both runs are kept (`fold_1_*/v1_early_stop/`).

| Fold (held-out) | Best epoch | Epochs run | Inner-val loss | Train loss at best |
|---|---|---|---|---|
| 0 b1 | 8 | 12 | 0.391 | 0.110 |
| 1 b2 | **0** | 6 | 0.997 | 1.202 |
| 2 b4 | 2 | 6 | 0.469 | 0.316 |
| 3 b5 | 8 | 12 | 0.387 | 0.094 |
| 4 b7 | 3 | 7 | 0.462 | 0.219 |
| 5 b8 (seeds 0 / 1 / 2) | 8 / 7 / 11 | 12 / 11 / 12 | 0.422 / 0.401 / 0.349 | 0.101 / 0.119 / 0.083 |

**Generalization gap:** inner-validation loss is about 4× training loss even within the training buildings (later days of the same episodes). The model memorizes episode trajectories. With only 50 independent week-long episodes, this is the binding limitation.

## 8. Baselines

Same windows, same alignment, same rollout code (`rollout()`), same train-only fitting.

- **Persistence:** holds the latest true observation o_t0 for zone and tank. Energy is the training-set mean per (hour of day × outside-air bin). Energy is not observable, so no true persistence exists.
- **Linear AR:** ridge regression on the same step features, plus its own previous zone prediction (observed while available). Zone fitted by teacher forcing on history rows; tank and energy fitted at k = 0. Energy is clipped at 0.

Two baseline bugs were found and fixed *before* the final evaluation:
- persistence alternated between o_t0 and o_t0−1;
- the linear model lacked the state needed under a lag-2 sensor and zig-zagged.

## 9. Teacher-forced results (h = 1: every input is a true observation)

| Held-out | Zone MAE °C: pers / lin / GRU | Tank MAE K: pers / lin / GRU | Electricity MAE kW: pers / lin / GRU | Gas MAE kW: pers / lin / GRU |
|---|---|---|---|---|
| b1 (K 4, V 39) | 0.072 / 0.074 / **0.056** | 0.31 / 0.50 / **0.17** | 6.2 / 3.2 / **0.6** | 36.2 / 20.3 / **6.2** |
| b2 (K 9, V 85) | 0.090 / 0.089 / **0.078** | 0.56 / 0.61 / **0.45** | 7.6 / 4.1 / **3.7** | 34.7 / **30.3** / 45.0 |
| b4 (K 6, V 54) | 0.042 / 0.043 / **0.024** | 0.16 / 0.41 / **0.09** | 5.3 / 3.1 / **1.4** | 7.5 / 9.2 / **1.5** |
| b5 (K 8, V 79) | 0.043 / 0.042 / **0.028** | 0.21 / 0.39 / **0.08** | 5.8 / 2.6 / **0.7** | 9.0 / 7.9 / **2.4** |
| b7 (K 4, V 36) | 0.063 / 0.062 / **0.038** | 0.29 / 0.42 / **0.09** | 3.7 / 1.4 / **0.6** | 7.7 / 6.9 / **2.0** |
| b8 (K 3, V 30) | 0.083 / 0.079 / **0.046** | 0.29 / 0.42 / **0.15** | 3.6 / 1.6 / **0.3** | 10.6 / 7.0 / **1.3** |

Zone temperature "h = 1" is already two transitions beyond the newest zone sensor (lag 2).

## 10. Autoregressive results (own predictions fed back; outside air held at o_t0)

Zone MAE (°C) by horizon h, GRU / linear / persistence:

| Held-out | h = 3 | h = 6 | h = 12 | h = 36 |
|---|---|---|---|---|
| b1 | **0.074** / 0.109 / 0.099 | **0.114** / 0.164 / 0.150 | **0.168** / 0.284 / 0.264 | **0.327** / 0.659 / 0.661 |
| b2 | **0.093** / 0.120 / 0.118 | **0.151** / 0.186 / 0.184 | **0.243** / 0.323 / 0.329 | **0.518** / 0.758 / 0.823 |
| b4 | **0.034** / 0.066 / 0.064 | **0.051** / 0.102 / 0.102 | **0.077** / 0.175 / 0.179 | **0.172** / 0.419 / 0.438 |
| b5 | **0.042** / 0.065 / 0.065 | **0.071** / 0.101 / 0.101 | **0.110** / 0.171 / 0.175 | **0.227** / 0.404 / 0.418 |
| b7 | **0.049** / 0.097 / 0.094 | **0.081** / 0.154 / 0.149 | **0.128** / 0.271 / 0.265 | **0.297** / 0.660 / 0.660 |
| b8 | **0.057** / 0.108 / 0.111 | **0.089** / 0.168 / 0.173 | **0.138** / 0.297 / 0.313 | **0.304** / 0.706 / 0.807 |

Other quantities:
- **Tank MAE at h = 36:** GRU 0.32–0.86 K, except 1.48 K in b2. Linear degrades to 3.1–4.0 K: it is unstable when rolled out.
- **Electricity:** GRU error stays flat with horizon (≤ 1.4 kW in every fold except b2 at about 3 kW).
- **Bias:** GRU zone bias stays below 0.09 °C at h = 36, except b2 at −0.20 °C (cold drift).
- **Oracle vs held outside air:** the oracle helps by at most 0.03 °C at h = 36. Outside-air forecasting is not the bottleneck at 3 h.
- **Plots:**
  - `plots/error_vs_horizon.png`
  - `plots/error_by_building.png`
  - `plots/rollout_examples.png`
  - `plots/zone_pred_vs_actual_h12.png`
  - `plots/zone_residual_hist.png`
- **Per-zone MAE at h = 12** (GRU): median 0.07–0.22 °C by building; worst zones 0.14–0.51 °C. b2's VAV_1 is 0.51 °C. Per-zone vectors are in each `metrics.json`.

## 11. Six-fold LOBO summary

Pooled over 19,680 held-out windows:

| Model | Zone h1 / h12 / h36 °C | Tank h12 K | Electricity h12 kW | Gas h12 kW | Reward MAE h12 (deployable / exact-physics) |
|---|---|---|---|---|---|
| persistence | 0.065 / 0.254 / 0.635 | 0.83 | 5.1 | 16.7 | 0.0134 / — |
| linear AR | 0.065 / 0.254 / 0.601 | 2.37 | 2.5 | 12.2 | 0.0104 / — |
| **GRU** | **0.045 / 0.144 / 0.307** | **0.49** | **1.0** | **8.5** | **0.0059** / — |

Per-fold tables for every metric, horizon and model are in `results/phase2a/summary.md`.

| Station (pooled), zone MAE at h = 12 | GRU | linear | persistence |
|---|---|---|---|
| atlanta | 0.095 | 0.200 | 0.188 |
| denver | 0.141 | 0.239 | 0.255 |
| **fairbanks** | **0.256** | 0.432 | 0.413 |
| phoenix | 0.143 | 0.238 | 0.241 |
| singapore | 0.084 | 0.158 | 0.174 |

**By K and V:** each value comes from a single building (§6), so read the per-fold rows. The extrapolation folds behave differently:
- fold 5 (smallest, K = 3) generalizes well;
- fold 1 (largest, K = 9, V = 85) is the worst fold.

**By profile** (GRU, zone h = 12): error on `moderate` (perturbed) windows exceeds `baseline` windows in every fold (for example b1 0.21 vs 0.125; b8 0.148 vs 0.127).

## 12. Cold-weather results

Each severe-cold scenario is n = 1 (2 episodes, 656 windows):

| Scenario (held-out fold) | Zone h12 / h36 °C: GRU vs persistence | Tank h12 K: GRU vs persistence | Gas h12 kW: GRU / linear / persistence |
|---|---|---|---|
| fairbanks_1 (b1; −39 °C, outside training range) | 0.497 / 0.878 vs 0.660 / 1.644 | 1.52 vs 3.27 | **28.1** / 77.4 / 162.5 |
| fairbanks_2 (b2) | 0.422 / 0.894 vs 0.606 / 1.573 | 2.93 vs 3.31 | 145.9 / **95.4** / 119.2 — GRU worst |
| fairbanks_8 (b8) | 0.277 / 0.635 vs 0.596 / 1.574 | 2.00 vs 2.55 | **5.8** / 23.1 / 40.5 |
| not severe cold (all others) | 0.116 / 0.252 vs 0.214 / 0.528 | 0.30 vs 0.58 | **2.7** / 6.3 / 6.6 |

- Steps with OAT below −10 °C (pooled): GRU zone MAE 0.53 °C at h = 12, which is **4.6×** its warm-condition error.
- Heating recovery (pooled): GRU zone 0.41 °C vs persistence 1.33 and linear 1.02.
- **Gas during heating recovery:** the GRU is at or below linear in 5 folds (b1 35 vs 124 kW; b8 19 vs 40 kW) but **367 vs 201 kW in b2**.

**Conclusion:** cold is the dominant error regime, and the only fold where the GRU fails on gas is the large cold-capable building.

## 13. Reward reconstruction diagnostics

Variants (`src/dynamics/evaluate.py`):

| Variant | Inputs | Deployable? |
|---|---|---|
| `exact_phys` | predicted T, E, G + true occupancy and calendar | no; isolates dynamics error |
| `deploy` | predicted physics + schedule-gated uniform occupancy, clock hour, Mon–Fri, annual gas | yes |
| `true_deploy` | true physics + the deployable assumptions | estimator-only error |

Sanity check: true physics + true occupancy and calendar reproduces the simulator reward to ≤ 2.2e-8 per step in every fold.

**Per-step reward MAE at h = 12** (pooled; deployable): GRU 0.0059, linear 0.0104, persistence 0.0134. The mean |reward| per step is about 0.006.

**Sum over the 36-step rollout** (what a planner compares), MAE relative to the mean |true 36-step sum|:

| Held-out | exact_phys: GRU / linear | deploy: GRU | true_deploy (estimator only) |
|---|---|---|---|
| b1 | 0.13 / 0.61 | 0.55 | 0.44 |
| b2 | 1.28 / 1.70 | 1.72 | 0.54 |
| b4 | 0.32 / 1.63 | 0.88 | 0.64 |
| b5 | 0.17 / 1.26 | 0.25 | 0.07 |
| b7 | 0.17 / 1.47 | 2.16 | **2.18** |
| b8 | 0.08 / 1.35 | 0.09 | 0.02 |

**Finding:** apart from b2, the dynamics-driven reward error is moderate (8–32% of the 3 h reward sum for the GRU vs 61–163% for linear). The **deployable reward estimator itself** is the larger problem in several buildings: its error on the 3 h sum is up to 2.2× in b7.

The cause is the occupancy gate. Future occupancy is not observable, so the estimator gates comfort on the comfort-mode schedule. During recovery and early-morning comfort hours, zones are below setpoint but nobody has arrived yet, so the estimator charges comfort penalties the true reward does not.

This is much larger than the Phase 1 audit's 1.7% episode-mean figure. That figure used the *observed* `num_occupants` (lag 2) as the gate, and averaged over a whole episode.

**Reward error around occupancy transitions** (h = 12): GRU 0.023 vs persistence 0.111. Plot: `plots/reward_pred_vs_actual_h12.png`.

## 14. Physical plausibility

Model responses on held-out heating-active windows, per fold (`metrics.json → probes`). These are diagnostics, not enforced laws.

| Probe | Expectation | GRU | Linear |
|---|---|---|---|
| reheat valve +0.3 (zones on running AHUs, tank more than 5 K above SAT) | zone T does not fall | **inconsistent:** mean +0.011 to +0.061 K at k = 3, but 3–52% of zones predicted to *cool* (b1 52%) | always positive (+0.22–0.33 K) |
| HWS setpoint +10 K | tank T and gas rise | plausible: tank +0.19 to +4.4 K at k = 6 (≤ 3% negative); gas +0.5 to +21 kW at k = 1 | plausible |
| SAT +3 K | zone T rises (when air flows) | mostly positive; 4–33% negative | always positive |

- **Energy** is non-negative by construction (softplus) for the GRU and by clipping for linear.
- **Predicted zone temperatures** stay in −2.2 … 34.5 °C. True unheated weekend zones in fairbanks reach −2.5 °C (actual minimum 270.6 K), so no physically impossible values occur.

The **action sensitivity of the GRU is not reliable at zone level.** This matches the audit: damper and valve actions barely vary in the data (only 2–8% of steps differ from the baseline), so the model has learned the state trajectory far better than its causal response to VAV actions.

## 15. Uncertainty and model exploitation

**Action-distance analysis** (`metrics.json → action_distance`). Mahalanobis distance of each test window's planned actions (AHU / plant / VAV means and variability) from the training-window distribution.

| Held-out | Test windows beyond the training p99 | GRU zone MAE h12 (<p50 → >p99) | Gas MAE h12 (<p50 → >p99) |
|---|---|---|---|
| b1 | 6.5% | 0.15 → 0.74 °C | 0.1 → 65 kW |
| b2 | 4.3% | 0.19 → 1.08 °C | 0.6 → 543 kW |
| other folds | ≤ 1.7% | roughly flat | — |

These far-out windows coincide with extreme cold. Distance and cold are confounded, and this data cannot separate them.

**Ensemble** (fold 5, 3 seeds, b8):
- The ensemble mean improves zone MAE by 5–9% (h = 12: 0.126 vs 0.138 °C).
- Member spread correlates with error: Spearman 0.52 at h = 1, 0.52 at h = 12, 0.60 at h = 36.
- The top-decile-spread windows have 5–6× the error of the low-spread half (h = 12: 0.417 vs 0.078 °C).

**Use for MPC (not implemented):** penalize candidate trajectories by ensemble spread on zone temperature and gas, plus action distance from the data support. Reject or down-weight plans beyond about the training p90–p99 band, and keep the baseline plan as the fallback.

## 16. Failure cases

1. **Fold 1 (b2: largest building, K = 9, V = 85, outside the training K/V range).** Inner validation degrades from epoch 0. The selected epoch-0 model is undertrained:
   - gas is worse than both baselines (fairbanks_2: 146 kW; heating-recovery gas 367 kW);
   - per-step deployable reward at h = 1 is worse than linear;
   - the 3 h tank error is 1.48 K;
   - it shows a cold drift of −0.20 °C at h = 36.
2. **Deep-cold free fall (b8 coldest window, −17 °C).** Zones in setback cool about 1 K over 3 h, but both learned models predict warming of about 1.4 K (`plots/rollout_examples.png`, top right).
3. **Zone-level action response** has an inconsistent sign (§14).
4. **Out-of-distribution actions / extreme cold:** errors grow 5–10× (§15).
5. **Perturbed (`moderate`) windows** are consistently harder than baseline windows. This is the regime an MPC will create.
6. **Train vs inner-validation gap** of about 4× within the training buildings: trajectory memorization.

## 17. Limitations

- **Data:** 6 buildings, 1 week per scenario, 3 severe-cold scenarios (one per cold fold, n = 1). Every K / V value except K = 4 comes from one building.
- **Fixed hyperparameters, no search:** hidden 64; lr 2e-3; train stride 4. No search was performed, to avoid any risk of tuning on held-out buildings; inner-validation-based tuning is possible in Phase 2B.
- **Ensemble** for one fold only (compute).
- **Not modelled:** occupancy forecasting; outside air beyond holding o_t0 (the oracle shows it matters little at 3 h); per-AHU energy supervision (only the aggregate is logged).
- **Horizon:** evaluated to 3 h (36 steps). Longer horizons for tank and pre-heat planning are untested.
- **Probes** compare model responses with *expected* signs; the true simulator response to these interventions was not measured. That would need simulator counterfactual rollouts (affordable, but not done here).
- **Action-distance metric:** a simple Gaussian / Mahalanobis model over window-level action summaries.

## 18. Recommendations for Phase 2B

1. **Data before architecture.** The binding constraints are trajectory memorization, poor coverage of VAV-level actions, and sparse cold. Recommend (needs approval):
   - full-horizon collection;
   - an exploration profile that perturbs VAV dampers and valves;
   - generated buildings, especially large and cold ones.
2. **Graph-aware model vs this GRU,** on the same folds, metrics and code (`rollout`, `evaluate_models`). Target the failure modes:
   - per-zone action response (adjacency, AHU membership);
   - the largest building (size extrapolation).
3. **Regularize:** input noise or scheduled sampling, dropout, smaller latent, larger stride / more episodes. Select on inner validation only.
4. **Add an occupancy-forecast head** (total occupancy from clock and history; per-zone occupancy as privileged supervision) to replace the schedule gate in the deployable reward estimator. This is currently the largest reward-error source in several buildings.
5. **Ensemble or uncertainty heads** for all folds. Calibrate spread against held-out error, and use action-distance and spread penalties as the planner's trust region.
6. **Validate action sensitivities with simulator counterfactuals** (offline, using the Phase 0 env-copy method), not only expected signs.

## 19. Readiness decision

**READY WITH CONDITIONS.**

The GRU is a sound, reproducible reference model:
- leakage-safe inputs and tested alignment;
- train-only statistics;
- consistent, large gains over persistence and linear on zone temperature, tank temperature and electricity in all six held-out buildings;
- gas gains in five of six.

It is suitable as the **baseline for Phase 2B graph-aware model comparison**.

It is **not yet reliable enough for MPC**:
- Fold 1 fails on gas and reward.
- Zone-level action sensitivities are inconsistent in sign.
- Errors explode for out-of-distribution actions and deep cold.
- The deployable reward estimator, specifically the occupancy gate, can exceed the true 3 h reward sum.

These are the conditions to address before any planning experiment.
