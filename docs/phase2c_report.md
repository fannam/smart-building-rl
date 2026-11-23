# Phase 2C: Paired Counterfactual Data and Training — Report

> **Date:** 2026-11-22, held-out evaluation added 2026-11-23 · **Code:** `ca25cbb` (branch `phase2c` = `main`) + `scripts/phase2c_eval.py` · **Starter kit:** `1fb75606`
> **Companion:** [`phase2c_data_collection_proposal.md`](phase2c_data_collection_proposal.md) (design), [`phase2b_report.md`](phase2b_report.md) (the failure this phase addresses)
> **Artifacts:** `results/phase2c/` (split manifest, collection QA, training logs and checkpoints; held-out evaluation in `eval/train_val_test_b/`). The checkpoints (`*.pt`) and `rows.csv.gz` are gitignored.

**Status:** data collection, training and the held-out (unseen-building) evaluation are **complete**. Test-A (the Phase 2B branch points) is not evaluated: its simulator archives are not available locally.

**Decision: READY WITH CONDITIONS** for an offline MPC prototype (§10). On buildings the models never trained on, the zone-temperature response gain rose from Phase 2B's −0.02–0.14 to **0.82** (graph ensemble). Most of that improvement comes from the new data; the paired loss adds a smaller, consistent gain (§8.2). The conditions are the remaining weak spots: plant-level gas over-response, weakly modelled plant/recirculation effects, a one-step response lag, and cold and night conditions (§8.3).

**Two scopes of numbers:**
- §5–§7 are measured on **validation branch points of each fold's training buildings**.
- §8 is measured on **held-out buildings**: each branch point is scored by the models of the LOBO fold that never trained on its building.

All numbers come from saved artifacts: `results/phase2c/collection_qa/`, `results/phase2c/split_manifest.json`, `results/phase2c/training/*/*/train_log_seed*.json`, `results/phase2c/eval/train_val_test_b/` and the collection session logs in `data/phase2c_train.zip`. Nothing is estimated unless it is marked as an estimate.

---

## 1. Executive summary

1. **The collection ran as designed, with no failures.**
   - 300 / 300 branch points, 20 arms each (6,000 arms), across all 30 scenarios.
   - Every nominal replay was bit-exact (max |Δobs| = 0.0). There were 0 errors and 0 aborted arms.
   - It took 5.72 h on one Kaggle CPU session with 4 workers.
2. **Zone-level actions now get real excitation.**
   - Damper: 13,181 zone events changed effectively; 13,127 of them produced a measurable zone response.
   - Reheat valve: 12,628 zone events changed effectively; 5,161 produced a response.
   - For comparison, Phase 1 had 0.8–1.2 % effective deviation from the baseline for these families, and Phase 2B had only building-wide arms.
   - The totals are below the proposal's ≈ 19k-per-family estimate, mainly because many branch points have no zones that qualify for the "decrease" arms (§3).
3. **All 32 planned training jobs finished**, using 68.2 GPU-job hours over three Kaggle sessions (≈ 11.3 h, ≈ 11.2 h and ≈ 1.1 h wall time on 2 × T4):
   - λ sweep: 8 jobs;
   - `arms` controls: 4 jobs;
   - `paired` on all 6 folds: 12 jobs;
   - graph ensemble seeds 1–2: 12 jobs, with 4 overlapping the sweep.
   - All five jobs resumed across sessions finished. Every checkpoint's epoch and job name match its train log.
4. **The paired loss lowers the validation counterfactual (cf) loss without hurting state prediction.**
   - Graph `paired` (λ = 1, 3 seeds) vs graph `arms`: cf loss **−29 %** (fold 1) and **−30 %** (fold 5). The energy component falls by about half.
   - GRU `paired` (λ = 0.3, 1 seed) vs GRU `arms`: cf loss −11 % (fold 1) and −19 % (fold 5).
   - The state loss on Phase 1 validation windows changes by −5.6 % to +2.5 %. That is inside the seed-to-seed range (up to ±12 %, §7.4).
5. **The graph model is better than the GRU in every fold**: 25–33 % lower cf loss and 9–32 % lower state loss. Note that the two used different λ.
6. **Training is not converged.** In 29 of 32 jobs the best epoch is the last one or the one before it (epoch 10–11 of 12), and validation loss was still falling. The 12-epoch cap is inherited from the Phase 2A / 2B protocol.
7. **On unseen buildings, the models now respond to actions** (§8; all 300 branch points, each scored by the fold that held its building out):
   - zone-temperature pooled gain **0.82** for the graph ensemble and 0.74 for the GRU, against −0.02 to 0.14 in Phase 2B;
   - building-wide damper / reheat-valve arms, the design closest to Phase 2B's: **0.91 / 0.81**, against 0.04 / 0.04;
   - direction right in 86 % of cases; error 0.34 of a no-response model's on folds 1 and 5 (Phase 2B: 0.75–1.07);
   - unseen weather (Test-B, `denver_*`) costs little: 0.79 vs 0.83.
8. **Most of the improvement comes from the data, not the loss.** The `arms` controls (λ = 0) already reach 0.69 (graph) and 0.64 (GRU) on folds 1 and 5. The paired loss adds +0.04–0.08 overall, and +0.2 for the reheat valve; every paired seed beats its control (§8.2).
9. **Weak spots for a planner** (§8.3):
   - gas over-response to the hot-water pump pressure (≈ 3.3×) and static pressure (≈ 3.2×), which a planner could exploit;
   - poorly modelled zone effects of recirculation (0.11), hot-water setpoint (0.26) and pump pressure (0.25), and of damper *decreases* (0.26);
   - a one-step lag: gain 0.56 over the first 5 minutes vs 0.82 over 3 h;
   - weaker in cold weather (0.64) and unoccupied nights (0.47).
10. **Still open:** how well the ensemble's spread flags bad predictions, and Test-A.

---

## 2. What was implemented

| Component | Location | Notes |
|---|---|---|
| Phase 2C arm design (zone-subset interventions, seeded draws, AHU-on gating, no-op control) | `src/counterfactual/phase2c.py`, `src/counterfactual/interventions.py` | Reuses Phase 2B branching (`branch.py`) and the simulator driver (`simrun.py`) unchanged |
| Collector (resumable, time budget, nominal-replay check) | `scripts/phase2c_collect.py`, `configs/phase2c/collection.json` | The proposal named `counterfactual_run.py --design phase2c`; it was implemented as a separate script so the Phase 2B path stays untouched (proposal §6) |
| Collection QA and manifest | `scripts/phase2c_collection_qa.py` | Writes `qa_report.json` and `manifest.jsonl` |
| Split manifest and leakage checks | `scripts/phase2c_splits.py` | Version `phase2c-splits-v1`, digest `326251fd425e93f3` |
| Branch-group dataset (context + nominal + all arms of one point, never split apart) | `src/counterfactual/paired_data.py` | Δ normalization statistics come from TRAIN responses only |
| Paired loss | `src/dynamics/paired.py` | Autoregressive nominal and arms from the same context and parameters; Huber on normalized Δ̂ − Δ over all 36 steps; no-op arms weighted 0.1 |
| Training (`arms` / `paired`, resumable, session deadline) | `scripts/train_paired.py`, `configs/phase2c/paired_training.json` | Same protocol as Phase 2A / 2B (§5) |
| Kaggle notebooks | `kaggle/phase2c_collect_counterfactual.ipynb`, `kaggle/phase2c_train.ipynb` | Training is staged across 12 h sessions with automatic resume |
| Tests | `tests/test_phase2c.py` (17 tests) | Arm semantics, no-op guarantee, split integrity, paired-loss alignment |
| Held-out evaluation | `scripts/phase2c_eval.py` | Reuses the Phase 2B metrics (`counterfactual.model_eval`); scores each branch point with the models of the fold holding its building out; uses only checkpoints that have a train log; adds a zero-response model, ensembles and per-horizon gains (§8) |

**Not yet implemented (proposal Workstreams 3–4):**
- Test-A evaluation (needs the Phase 2B simulator archives);
- the regression check on nominal Phase 2A windows for the new checkpoints;
- uncertainty scoring of the ensemble's Δ spread.

---

## 3. Data collection

**Run:** `phase2c_collect.py --scenarios all --parallel 4 --time-budget-hours 10.5` on Kaggle CPU, wall 20,591 s (5.72 h). All 30 scenarios finished; none hit the deadline.

| Check | Result |
|---|---|
| Branch points complete | 300 / 300 (10 per scenario, 50 per building) |
| Arms per point | 20 (nominal + 19 interventions) |
| Nominal replay | bit-exact for 300 / 300 points (max abs obs difference 0.0) |
| Errors / aborted arms / QA hard failures | 0 / 0 / 0 |
| Arms copied rather than re-simulated (commands identical to nominal) | 722 |

**Branch-point coverage:**
- **Strata:** morning_recovery 110, occupied_heating 59, occupied_cooling 44, unoccupied_night 40, unoccupied_other 30, occupied_other 17.
- **Outdoor-temperature bands:** mild 117, warm 107, cool 55, cold 21.
- **State at the branch point:** at least one AHU on at 260 / 300, heating active at 124, E2-eligible at 110.

### 3.1 Per-family outcome (`qa_report.json`)

| Family | Arms | Effective change | Zone response | Tank response | Gas response | Extreme excursion | Main reason for no-ops |
|---|---|---|---|---|---|---|---|
| E1 damper, subset set | 900 | 780 | 780 | 121 | 182 | 16 | 120 points with no zone on an AHU that is on |
| E1 damper, subset decrease | 300 | 229 | 229 | 56 | 54 | 11 | 71 points with no zone above 0.2 |
| E1 damper, building-wide +0.3 | 300 | 300 | 263 | 76 | 81 | 9 | — (130 clipped) |
| E2 valve, subset set | 900 | 780 | 319 | 258 | 301 | 9 | 120 points with no zone on an AHU that is on |
| E2 valve, subset decrease | 300 | **101** | 99 | 91 | 83 | 25 | **199 points with no zone above 0.05** |
| E2 valve, building-wide +0.4 | 300 | 300 | 112 | 110 | 107 | 4 | — (113 clipped) |
| SAT ± U(2, 5) K | 600 | 511 | 367 | 67 | 73 | 4 | 89 no-ops (AHU off); 312 clipped |
| Static pressure +300 Pa | 300 | 260 | 260 | 44 | 64 | 2 | 114 saturated |
| Recirculation −0.3 | 300 | 291 | 133 | 86 | 88 | 5 | 67 clipped |
| AHU run flip | 300 | 300 | 300 | 102 | 87 | **38** | — |
| HWS setpoint ± 10 K | 300 | 237 | 45 | 120 | 120 | 2 | 63 no-ops; 184 clipped |
| HWS dp +4 | 300 | 299 | 110 | 104 | 106 | 2 | — |
| HWS run flip | 300 | 300 | 100 | 95 | 85 | **31** | — |
| NOOP damper control | 300 | 5 | 0 | 0 | 0 | 2 | by design (250 saturated at the floor) |

Notes:
- **Valve responses are still sparse.** 780 subset-set arms changed the valve, but only 319 produced a zone-temperature response. Of the 12,628 zone events where the valve changed effectively, 5,161 responded. The valve matters only when the coil has hot water and the zone has airflow. The "decrease" arm is usually a no-op because the nominal valve is almost always 0, which matches Phase 1, where 93 % of valve values are at 0.
- **The NOOP control behaves as designed.** There are 5 effective changes out of 300 and no response in any target. *Checked in the evaluation:* the simulator shows no response in any quantity for these 5, and the model's predicted change is ≈ 1e-7 K, so they are harmless.
- **Extreme excursions are kept and flagged, not dropped** (proposal §5). Most come from the run flips: AHU 38, HWS 31.

---

## 4. Splits

- **Split unit:** the scenario-day (`t_b // 288`). All arms of a branch point, and all points on the same scenario-day, share one split.
- **Test-A:** the 68 Phase 2B branch points (12 scenarios). They are not part of the 300 new points, and every new point is ≥ 48 steps away from them.

| Split | Branch points | Content | Used so far |
|---|---|---|---|
| Train | 184 | non-Denver scenarios, 80 % of day-blocks | training |
| Val | 56 | non-Denver scenarios, 20 % of day-blocks | λ selection, early stopping, all numbers in §7 |
| Test-B | 60 | every `denver_*` scenario (10 per building) | held-out evaluation only (§8) |
| Test-A | 68 (Phase 2B set) | building-wide arms, 3 h | **not used** |

Per fold, training uses only the train/val points of the fold's 5 training buildings: 149–157 train points and 43–51 val points. The held-out building's points are never loaded during training. So all 50 points of a building, whatever their split label, are unseen by the fold that holds it out, which is what §8 uses.

---

## 5. Training protocol

The architectures and training protocol are those of Phase 2A (GRU, 99,142 parameters) and Phase 2B (graph, 100,398 parameters), unchanged:
- LOBO fold, normalizer fit on training data only, inner time split;
- Huber state loss on Phase 1 windows (19,700 training windows per fold, context 12, horizon 24, stride 4);
- AdamW (lr 2e-3, weight decay 1e-4) with a one-cycle schedule, gradient clip 1.0;
- 12 epochs, patience 3, minimum 6 epochs.

Objectives:

    arms   : L = L_state(Phase 1) + 1.0 · L_state(branch arms as ordinary 3 h trajectories)
    paired : L = the same + λ · L_cf(paired Δ, all 36 steps)

- Two branch groups are added every 2nd step.
- No-op arms enter L_cf with weight 0.1.
- Checkpoints are selected by `val_state + arm_state + λ · cf` on the validation sets.
- **Training starts from scratch for each job.** The Phase 2A / 2B checkpoints are not fine-tuned.

Plan (`configs/phase2c/paired_training.json`):
1. λ sweep {0.1, 0.3, 1.0} on fold 5 (b8, the smallest building);
2. `arms` controls on folds 1 (b2, largest) and 5;
3. `paired` on all 6 folds at the selected λ;
4. graph ensemble seeds 1–2 on all folds.

### 5.1 Compute

| Session | Wall time | Jobs finished | Notes |
|---|---|---|---|
| 1 | ≈ 11.3 h | 18 | stage 1, most of stage 2; graph folds 3 and 4 interrupted at epochs 5 and 1 |
| 2 | ≈ 11.2 h | 11 | graph folds 3–4 resumed and finished; ensemble seed 1 (6 folds) and seed 2 (folds 0–2); seed 2 folds 3–5 interrupted at epoch 7 |
| 3 | ≈ 1.1 h | 3 | seed 2 folds 3–5 resumed and finished |

- **Cost per job:** GRU 71–78 min, graph 144–162 min on one T4.
- **Total:** 68.2 GPU-job hours. The proposal estimated ≈ 15–20 h of wall time; the actual wall time was ≈ 24 h with 3 jobs running in parallel.
- **Resume:** each resumed job continued from a per-epoch training state (`resumed_at_epochs` in its log). A resumed job is not guaranteed bit-identical to an uninterrupted run, because the branch-group iterator restarts.

---

## 6. λ selection (fold 5, seed 0)

**Rule:** take the lowest val cf loss among the λ whose val state loss is ≤ 1.05 × that of the λ = 0 `arms` control.

| Model | λ = 0 (`arms`) state / cf | λ = 0.1 | λ = 0.3 | λ = 1.0 | Selected |
|---|---|---|---|---|---|
| GRU | 0.3020 / 0.1822 | 0.2994 / 0.1591 | **0.2942 / 0.1468** | 0.3205 / 0.1313 (state 1.061×, excluded) | **0.3** |
| Graph | 0.2143 / 0.1403 | 0.2194 / 0.1294 | 0.2119 / 0.1248 | **0.2182 / 0.1028** (1.018×) | **1.0** |

Within the sweep, cf loss falls steadily as λ rises for both models.

The GRU was excluded from λ = 1 by a 6.1 % rise in state loss. That margin sits inside the seed-to-seed spread seen later (§7.4), so the GRU choice is not robust: with a different seed, λ = 1 might have been selected. The graph choice is clearer, because λ = 1 has the lowest cf loss and is comfortably within the state-loss tolerance.

---

## 7. Validation results

**What the numbers mean:**
- **"state"** is the total multitask Huber loss on Phase 1 inner-validation windows of the training buildings.
- **"cf"** is the paired Δ loss on the validation branch points of the training buildings, broken into zone / tank / energy.
- **Lower is better** for both.
- **Don't compare across folds.** Each fold has a different validation set: fold 0's low cf loss reflects its validation points, not a better model.

### 7.1 Effect of the paired loss (same data, same schedule)

| Fold | Model | `arms` state / cf | `paired` state / cf | Δ state | Δ cf | Δ zone / tank / energy |
|---|---|---|---|---|---|---|
| 1 (b2) | GRU (λ 0.3, 1 seed) | 0.4031 / 0.1873 | 0.3807 / 0.1668 | −5.6 % | −11 % | −4 % / −14 % / −17 % |
| 5 (b8) | GRU (λ 0.3, 1 seed) | 0.3020 / 0.1822 | 0.2942 / 0.1468 | −2.6 % | −19 % | −17 % / −10 % / −26 % |
| 1 (b2) | graph (λ 1, mean of 3 seeds) | 0.3654 / 0.1636 | 0.3562 / 0.1162 | −2.5 % | **−29 %** | −17 % / −17 % / −48 % |
| 5 (b8) | graph (λ 1, mean of 3 seeds) | 0.2143 / 0.1403 | 0.2197 / 0.0978 | +2.5 % | **−30 %** | −18 % / −19 % / −47 % |

- For the graph model, every `paired` seed has a lower cf loss than its `arms` control: fold 1 0.106–0.126 vs 0.164; fold 5 0.095–0.103 vs 0.140. The effect is larger than the seed noise.
- The energy component improves most. Zone and tank improve by 17–19 %.
- The state-loss changes are within ±6 %, about the size of the seed spread, so the paired objective has **no measurable cost to observational accuracy**.
- **Limitation:** the controls use one seed and cover only 2 folds.

### 7.2 Paired models on all six folds (seed 0)

| Fold (held-out building) | GRU λ 0.3 state / cf | Graph λ 1 state / cf | Graph vs GRU, cf |
|---|---|---|---|
| 0 (b1) | 0.2478 / 0.0643 | 0.1942 / 0.0431 | −33 % |
| 1 (b2) | 0.3807 / 0.1668 | 0.3457 / 0.1257 | −25 % |
| 2 (b4) | 0.3081 / 0.1167 | 0.2130 / 0.0852 | −27 % |
| 3 (b5) | 0.2669 / 0.1132 | 0.1811 / 0.0811 | −28 % |
| 4 (b7) | 0.2877 / 0.1320 | 0.1905 / 0.0943 | −29 % |
| 5 (b8) | 0.2942 / 0.1468 | 0.2182 / 0.1028 | −30 % |

The graph model has lower state and cf loss in every fold, which matches its Phase 2B observational advantage. Part of the cf gap could come from the larger λ (1 vs 0.3) rather than the architecture. Fold 1 (b2) has the highest losses for both models, as in Phase 2A / 2B.

### 7.3 Graph `paired` ensemble (λ 1, seeds 0 / 1 / 2)

| Fold | state (seeds 0 / 1 / 2) | mean | cf (seeds 0 / 1 / 2) | mean |
|---|---|---|---|---|
| 0 | 0.194 / 0.173 / 0.153 | 0.174 | 0.043 / 0.040 / 0.040 | 0.041 |
| 1 | 0.346 / 0.359 / 0.364 | 0.356 | 0.126 / 0.117 / 0.106 | 0.116 |
| 2 | 0.213 / 0.209 / 0.235 | 0.219 | 0.085 / 0.083 / 0.081 | 0.083 |
| 3 | 0.181 / 0.183 / 0.183 | 0.182 | 0.081 / 0.074 / 0.076 | 0.077 |
| 4 | 0.191 / 0.196 / 0.209 | 0.198 | 0.094 / 0.082 / 0.083 | 0.086 |
| 5 | 0.218 / 0.220 / 0.221 | 0.220 | 0.103 / 0.096 / 0.095 | 0.098 |

The metrics are of the individual members. Ensemble-mean predictions are evaluated in §8.

### 7.4 Seed variability

- **State loss:** the spread across seeds reaches 24 % of the mean (fold 0: 0.153–0.194). It is under 2 % in folds 3 and 5.
- **cf loss:** the spread is smaller, at 5–17 % of the mean.
- **Consequence:** single-seed state-loss differences under about 10 % (the λ-selection tolerance, the GRU controls) should not be treated as significant.

### 7.5 Convergence

| Best epoch (of 12) | Jobs |
|---|---|
| 11 (last) | 21 |
| 10 | 8 |
| 9 | 2 (GRU λ 1 fold 5; graph `arms` fold 5) |
| 8 | 1 (gru `arms` fold 5) |

- **No job stopped early.** The one-cycle schedule ends at epoch 12, and the last epochs still lower the validation loss, by about 0.5–1 % per epoch at the end.
- **Effect on the comparison:** `paired` jobs reach their best epoch later than the `arms` controls (fold 5: 10 vs 8–9). The cap may therefore favour the controls slightly, or slightly understate the paired effect.

---

## 8. Held-out evaluation (unseen buildings)

**Method** (`scripts/phase2c_eval.py --splits train val test_b`, results in `results/phase2c/eval/train_val_test_b/`):
- **Data:** all 300 branch points × 19 intervention arms.
- **Unseen buildings only:** each point is scored by the models of the LOBO fold that holds out its building. Its train/val/test labels only matter for the training buildings, so all 50 points of a building are unseen by that fold. Test-B (`denver_*`) adds an unseen weather strip.
- **Models:**
  - graph `paired` λ 1, seeds 0–2 and their mean (ensemble), all folds;
  - GRU `paired` λ 0.3, all folds;
  - the `arms` controls, folds 1 and 5;
  - the λ-sweep models, fold 5;
  - `zero`, which predicts no response.

  Only checkpoints with a train log are used.
- **Metrics** (same as Phase 2B, `counterfactual.model_eval.response_metrics`), on effective arms with a simulated response above threshold (zone 0.02 K):
  - **pooled gain** Σ Δ̂·Δ / Σ Δ²: the least-squares response gain, 1 = exact, 0 = no response;
  - **sign agreement**;
  - **normalized error** Σ|Δ̂ − Δ| / Σ|Δ|: the zero model scores 1.00;
  - **gain at horizon h:** the response averaged over the first h steps.
- **Runtime:** 166 s for all 300 points (6 workers, 2 torch threads each; 2.1–3.3 s per point).
- **Comparison with Phase 2B:** Phase 2B measured on its own 68 branch points (Test-A), which are not available here. The building-wide `add` arms reuse Phase 2B's design and are the closest like-for-like comparison.

### 8.1 Zone temperature, all six folds

| Family | n | Phase 2B best model | Graph ensemble (λ 1, 3 seeds) | Graph seeds 0 / 1 / 2 | GRU (λ 0.3) |
|---|---|---|---|---|---|
| damper | 1,272 | 0.04 | **0.88** | 0.87 / 0.89 / 0.89 | 0.85 |
| static_pressure | 260 | 0.12 | **0.85** | 0.81 / 0.90 / 0.84 | 0.73 |
| ahu_run | 300 | 0.08 | **0.81** | 0.78 / 0.82 / 0.82 | 0.74 |
| hws_run | 100 | 0.00 | 0.72 | 0.66 / 0.78 / 0.72 | 0.56 |
| reheat_valve | 530 | 0.04 | 0.69 | 0.67 / 0.66 / 0.75 | 0.44 |
| sat | 367 | 0.25 | 0.66 | 0.68 / 0.67 / 0.63 | 0.55 |
| hws_setpoint | 45 | 0.05 | 0.26 | 0.26 / 0.32 / 0.22 | 0.19 |
| hws_dp | 110 | 0.03 | 0.25 | 0.17 / 0.25 / 0.33 | 0.24 |
| recirculation | 133 | 0.01 | 0.11 | 0.11 / 0.09 / 0.14 | 0.02 |
| **all families** | 3,117 | −0.02 to 0.14 per family | **0.82** | 0.79 / 0.83 / 0.83 | 0.74 |

"Phase 2B best model" is the highest gain among the Phase 2B GRU, GRU ensemble and graph models (`phase2b_report.md` §5.1).

| Summary (zone temperature) | Graph ensemble | GRU |
|---|---|---|
| Building-wide damper / reheat-valve arms (Phase 2B design; Phase 2B: 0.04 / 0.04) | **0.91 / 0.81** | 0.88 / 0.53 |
| Median per-case gain | 0.75 | 0.59 |
| Sign agreement / sign wrong | 0.86 / 0.14 | 0.81 / 0.19 |
| Test-B (unseen weather) vs other points | 0.79 vs 0.83 | 0.66 vs 0.75 |
| Spurious response on effective no-op arms | 0 | 0 |
| Nominal-arm zone error (folds 1 + 5) | 0.141 K | 0.215 K |

**By held-out building** (graph ensemble / GRU): b1 0.77 / 0.65, b2 0.73 / 0.66, b4 0.86 / 0.81, b5 **1.12 / 1.09** (over-response), b7 0.85 / 0.73, b8 0.81 / 0.68.

**Other quantities** (pooled gain, graph ensemble / GRU): cooling 1.00 / 0.99, fan 0.97 / 0.95, gas 0.82 / 0.73, tank 0.63 / 0.60, pump 0.57 / 0.40.

**Seeds:** the gain varies across seeds by up to 0.12 within a fold (fold 0: 0.69 / 0.80 / 0.81). The ensemble is at or near the best seed in every fold, so the ensemble is the model to use.

### 8.2 Where the improvement comes from: data vs the paired loss (folds 1 and 5)

Same data and schedule; the `arms` controls differ from `paired` only by λ = 0.

| Zone gain | `zero` | GRU `arms` | GRU `paired` | graph `arms` | graph `paired` seed 0 | graph `paired` ensemble |
|---|---|---|---|---|---|---|
| all families | 0 | 0.64 | 0.67 | 0.69 | 0.73 | **0.75** |
| reheat_valve | 0 | 0.32 | 0.37 | 0.46 | 0.67 | 0.68 |
| hws_run | 0 | 0.32 | 0.45 | 0.52 | 0.66 | 0.69 |
| sat | 0 | 0.57 | 0.55 | 0.71 | 0.65 | 0.64 |
| damper | 0 | 0.77 | 0.77 | 0.77 | 0.78 | 0.79 |
| normalized error | 1.00 | 0.455 | 0.435 | 0.385 | 0.360 | **0.336** |

| Other quantities | GRU `arms` | GRU `paired` | graph `arms` | graph `paired` ensemble |
|---|---|---|---|---|
| tank | 0.50 | 0.59 | 0.71 | 0.72 |
| gas | 0.66 | 0.87 | 0.96 | 1.18 |
| pump | 0.30 | 0.38 | 0.39 | 0.57 |

- **The data does most of the work.** With λ = 0, the new zone-level arms alone take the zone gain from Phase 2B's ≤ 0.25 to 0.64–0.69.
- **The paired loss adds a real but smaller improvement** for the graph model. Every paired seed beats the control in both folds: fold 1 0.71–0.75 vs 0.675; fold 5 0.80–0.81 vs 0.73. The largest gain is the reheat valve: every paired seed is at 0.58–0.76 vs 0.45–0.47 for the control. SAT is slightly worse with the paired loss (0.71 → 0.65).
- For the GRU the paired effect is smaller (+0.02–0.06), with one seed only.

### 8.3 Weak spots

1. **Gas over-response on plant actions** (graph ensemble, all folds):

   | Family → gas | Cases | Simulated mean Δ | Predicted mean Δ | Pooled gain |
   |---|---|---|---|---|
   | hws_dp | 106 | 1.3 kW | 4.3 kW | **2.50** (per fold 1.2–3.4) |
   | static_pressure | 64 | 0.8 kW | 2.7 kW | **2.66** (per fold 0.07–10.6; sign agreement 0.66) |
   | reheat_valve | 491 | 2.3 kW | 4.5 kW | 1.08 (pooled; the mean is ≈ 1.9× too large) |
   | sat | 73 | 0.8 kW | 0.9 kW | 0.92 |
   | ahu_run | 87 | −36.0 kW | −35.6 kW | 0.83 |

   Because the predicted response is too large in both directions, a planner would see large gas savings from *reducing* the pump pressure or static pressure that the simulator would not deliver. Phase 2B's SAT → tank over-response (2.54) is gone (now 1.08); static pressure → tank is 1.19.
2. **Poorly modelled zone effects:**
   - recirculation 0.11, hot-water setpoint 0.26, hot-water pump pressure 0.25;
   - damper *decrease* arms 0.26 and valve decreases 0.49, against 0.82–0.91 for increases.
3. **Response lag.** Graph-ensemble zone gain at h = 1 / 3 / 6 / 12 / 24 / 36 is 0.56 / 0.63 / 0.67 / 0.70 / 0.79 / 0.82 (GRU: 0.21 / 0.35 / 0.44 / 0.51 / 0.67 / 0.74). The median half-response latency is 2.0 steps for the models vs 1.5 for the simulator.
4. **Conditions:**
   - by outdoor band: cold 0.64, cool 0.78, mild 0.86, warm 0.96;
   - by stratum: unoccupied night 0.47, unoccupied other 0.66, morning recovery 0.80, occupied heating 0.80, occupied cooling 0.95.

   Cold weather dominates the score (`latent_mpc_implementation_plan.md` §0), so this is the most costly weakness.
5. **Over-response in b5** (gain 1.12 graph, 1.09 GRU).

---

## 9. Limitations

1. **Test-A is not evaluated.** The Phase 2B simulator archives (`results/phase2b/counterfactual/sim/*.npz`) are gitignored and not in this checkout. The comparison with Phase 2B in §8 is therefore across different branch points; the building-wide arms share Phase 2B's design, but not its points.
2. **The Phase 2A / 2B checkpoints are not available locally**, so they could not be scored on the Phase 2C points. The `arms` controls (§8.2) are the controlled comparison instead.
3. **Validation sets are small.** There are 43–51 validation branch points per fold, drawn from 5 buildings. λ selection and checkpoint selection both use them.
4. **The controls are thin:** one seed and 2 folds for `arms`; one seed for GRU `paired`. Within-fold seed variation in held-out gain reaches 0.12 (§8.1).
5. **The λ sweep covered one fold (b8, the smallest building) with one seed.** The GRU choice rests on a margin smaller than the seed noise.
6. **The models are undertrained** under the inherited 12-epoch cap (§7.5).
7. **Valve excitation is still limited** by plant state: 41 % of effective valve-subset events produced a zone response (§3.1).
8. **The held-out points come from the same 30 scenarios as the training data** (other buildings, plus the Denver strips). The hidden competition set uses unseen buildings *and* unseen weather strips, possibly including the unused `minneapolis` station (`phase0_findings.md` §2.4). Gains could be lower there, particularly in the cold.
9. **Pooled gain is dominated by large responses.** The median per-case gain (0.75) is lower than the pooled gain (0.82).

---

## 10. Readiness

The Phase 2B gate for an offline MPC prototype requires reliable action responses on unseen buildings. Status:

| Criterion | Status |
|---|---|
| Zone-level actions excited in the data | **met** (§3) |
| Paired objective trains stably at scale on all folds | **met** (§5, §7) |
| Paired objective lowers validation Δ error without hurting state accuracy | **met on validation** (§7.1), within the limits in §9 |
| Response gain on held-out buildings close to 1 | **met for the airflow-side actions** (damper, static pressure, AHU run: 0.81–0.88; building-wide damper 0.91); **partly met** for reheat valve, SAT and HWS run (0.66–0.72); **not met** for recirculation, HWS setpoint and HWS dp (0.11–0.26). Test-A not measured |
| No over-response a planner could exploit | **not met for gas**: HWS dp and static pressure → gas ≈ 3× (§8.3). The Phase 2B SAT → tank / gas over-responses are fixed |
| Ensemble spread flags bad Δ predictions | **not measured** |
| Nominal accuracy no worse than Phase 2A / 2B on their test windows | **not measured** on Phase 2A windows; nominal-arm zone error on the branch points is 0.14 K |

**Decision: READY WITH CONDITIONS** for an offline MPC prototype:
- **Use** the graph `paired` λ 1 three-seed ensemble.
- **Plan only over** the actions it models well: per-zone airflow (damper), static pressure, AHU run/off, and with caution SAT and the reheat valve.
- **Don't take** gas and tank costs from the model for plant actions (HWS dp, HWS setpoint, static pressure → gas). Use an analytical plant model, or keep the planner from moving those actions.
- **Expect** weaker predictions in cold weather and at night, and a one-step lag in the first 15 minutes.

**Not decided here:** whether a learned-model planner fits the competition's runtime budget. Phase 0 §2.4 indicates the hidden evaluation runs about 48 scenarios sequentially, which may leave very little time per step. Whether any of the public baseline's code may ship in a submission is also a separate question.

---

## 11. Next steps

1. **Runtime feasibility first** (`phase0_findings.md` §8 item 7): time one full `bang_bang` rollout and one rollout of the evaluated graph ensemble's inference on a small building in a Kaggle notebook. If the per-step budget is near zero, the planner path stops here, and the models are used offline only (tuning or distilling a fast policy).
2. **Offline MPC / oracle study** on training scenarios, restricted to the well-modelled actions, with the simulator as ground truth. It measures the score gain multi-step planning could deliver before any runtime work.
3. **Fix the gas over-response:** use an analytical plant model for gas and tank, or add more plant-action arms at varied hot-water states and retrain.
4. **Ensemble uncertainty** (Workstream 4): compare the 3-seed graph Δ spread with Δ error on these arms, to see whether it flags the weak families and cold cases.
5. **A cold-weather held-out test:** generate scenarios with the unused `minneapolis` station for the shipped buildings, collect branch points there, and rerun `scripts/phase2c_eval.py`.
6. **Optional:**
   - a longer schedule (for example 24 epochs) as a protocol change (version bump, retrain the controls);
   - more control seeds;
   - Test-A, if the Phase 2B archives are regenerated.

---

## 12. Reproduction

```bash
# collection (Kaggle CPU, kaggle/phase2c_collect_counterfactual.ipynb)
python scripts/phase2c_collect.py --config configs/phase2c/collection.json --out <OUT> --scenarios all --parallel 4 --time-budget-hours 10.5
python scripts/phase2c_collection_qa.py --out <OUT>
python scripts/phase2c_splits.py --data <OUT> --out <split_manifest.json> --heldout-weather denver

# training (Kaggle 2 x T4, kaggle/phase2c_train.ipynb; rerun with the previous output attached until "ALL DONE")
python scripts/train_paired.py --config <run_config.json> --out results/phase2c/training --parallel 3 --device cuda \
    --deadline <unix> --jobs <arch>:<objective>:<fold>:<seed>:<lambda> ...

# held-out evaluation (local CPU, about 3 min for all 300 points; no simulator needed)
unzip -q data/phase2c_train.zip -d data/_p2c && mv data/_p2c/phase1_sample/phase1_sample data/phase1_sample \
  && mv data/_p2c/phase2c_counterfactual/phase2c_counterfactual data/phase2c_counterfactual && rm -r data/_p2c
.venv/bin/python scripts/phase2c_eval.py --splits train val test_b --parallel 6
```

**Inputs and outputs:**
- **Data input:** `data/phase2c_train.zip`, which holds `phase1_sample/` and `phase2c_counterfactual/`.
- **Training outputs:** `data/phase2c_training.zip` (session 1), `data/phase2c_training_2nd_run.zip`, `data/phase2c_training_3rd_run.zip` (final; installed as `results/phase2c/`).
- **Session-to-session resume:** the `data/phase2c_training_resume*.zip` files are the outputs repacked under a `phase2c_training/` folder, so the notebook's resume search finds them.
- **Evaluation outputs:** `results/phase2c/eval/train_val_test_b/` holds `summary.json` (tables by quantity, family, magnitude/duration, building, stratum and outdoor band), `timing.json` and `rows.csv.gz` (one row per point × arm × quantity × model).
