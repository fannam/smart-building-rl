# Phase 2C: Paired Counterfactual Data and Training — Report

> **Date:** 2026-11-22 · **Code:** `ca25cbb` (branch `phase2c` = `main`) · **Starter kit:** `1fb75606`
> **Companion:** [`phase2c_data_collection_proposal.md`](phase2c_data_collection_proposal.md) (design), [`phase2b_report.md`](phase2b_report.md) (the failure this phase addresses)
> **Artifacts:** `results/phase2c/` (split manifest, collection QA, training logs and checkpoints). The checkpoints (`*.pt`) are gitignored.

**Status:** data collection and training are **complete**. Held-out evaluation (Test-A / Test-B) is **not done yet**.

**Decision: PENDING.** Phase 2C has not yet shown that the action-response failure from Phase 2B is fixed. Every number in this report is measured on **validation branch points from each fold's training buildings**. None comes from a held-out building or from the Test-A / Test-B sets. The readiness gate (§9) needs the held-out response-gain evaluation, which has not been written yet.

All numbers come from saved artifacts: `results/phase2c/collection_qa/`, `results/phase2c/split_manifest.json`, `results/phase2c/training/*/*/train_log_seed*.json`, and the collection session logs in `data/phase2c_train.zip`. Nothing is estimated unless it is marked as an estimate.

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
7. **Still open:**
   - whether the response **gain** on unseen buildings has moved from Phase 2B's 0–0.16 toward 1;
   - zone-level accuracy split by action family;
   - how well the ensemble's uncertainty is calibrated.

   Lower validation cf loss is necessary for these, but it doesn't prove them.

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

**Not yet implemented (proposal Workstreams 3–4):**
- the held-out evaluation, i.e. a `counterfactual_eval.py` extension with a `--split` filter, per-horizon gains and the Phase 2C checkpoint layout;
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
- **The NOOP control behaves as designed.** There are 5 effective changes out of 300 and no response in any target. These 5 deserve a look when the evaluation is built (`effective_change` with no response).
- **Extreme excursions are kept and flagged, not dropped** (proposal §5). Most come from the run flips: AHU 38, HWS 31.

---

## 4. Splits

- **Split unit:** the scenario-day (`t_b // 288`). All arms of a branch point, and all points on the same scenario-day, share one split.
- **Test-A:** the 68 Phase 2B branch points (12 scenarios). They are not part of the 300 new points, and every new point is ≥ 48 steps away from them.

| Split | Branch points | Content | Used so far |
|---|---|---|---|
| Train | 184 | non-Denver scenarios, 80 % of day-blocks | training |
| Val | 56 | non-Denver scenarios, 20 % of day-blocks | λ selection, early stopping, all numbers in §7 |
| Test-B | 60 | every `denver_*` scenario (10 per building) | **not used** |
| Test-A | 68 (Phase 2B set) | building-wide arms, 3 h | **not used** |

Per fold, training uses only the train/val points of the fold's 5 training buildings: 149–157 train points and 43–51 val points. The held-out building's points are never loaded during training.

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

The metrics are of the individual members. Ensemble-mean predictions have not been evaluated.

### 7.4 Seed variability

- **State loss:** the spread across seeds reaches 24 % of the mean (fold 0: 0.153–0.194). It is under 2 % in folds 3 and 5.
- **cf loss:** the spread is smaller, at 5–17 % of the mean.
- **Consequence:** single-seed state-loss differences under about 10 % (the λ-selection tolerance, the GRU controls) should not be treated as significant.

### 7.5 Convergence

| Best epoch (of 12) | Jobs |
|---|---|
| 11 (last) | 21 |
| 10 | 8 |
| 9 | 2 (gru λ 1 fold 5; graph `arms` fold 5) |
| 8 | 1 (gru `arms` fold 5) |

- **No job stopped early.** The one-cycle schedule ends at epoch 12, and the last epochs still lower the validation loss, by about 0.5–1 % per epoch at the end.
- **Effect on the comparison:** `paired` jobs reach their best epoch later than the `arms` controls (fold 5: 10 vs 8–9). The cap may therefore favour the controls slightly, or slightly understate the paired effect.

---

## 8. Limitations

1. **No held-out evaluation.** The central Phase 2B metric, the pooled least-squares response gain on unseen buildings (Phase 2B: −0.02 to 0.14 for zone temperature), has not been measured for any Phase 2C model. The same applies to direction accuracy, latency and spurious response. Validation cf loss is a normalized Huber error on Δ for training buildings, so it does not directly measure gain.
2. **Validation sets are small.** There are 43–51 validation branch points per fold, drawn from 5 buildings. λ selection and checkpoint selection both use them.
3. **The controls are thin:** one seed and 2 folds for `arms`; one seed for GRU `paired`.
4. **The λ sweep covered one fold (b8, the smallest building) with one seed.** The GRU choice rests on a margin smaller than the seed noise.
5. **The models are undertrained** under the inherited 12-epoch cap (§7.5).
6. **Valve excitation is still limited** by plant state: 41 % of effective valve-subset events produced a zone response (§3.1).
7. **Test-A data is not present locally.** The Phase 2B simulator archives (`results/phase2b/counterfactual/sim/*.npz`) are gitignored, and only their logs are in this checkout. They must be regenerated or fetched before Test-A can be evaluated. The Phase 2C branch archives are in `data/phase2c_train.zip` (`phase2c_counterfactual/`).

---

## 9. Readiness

The Phase 2B gate for an offline MPC prototype requires reliable action responses on unseen buildings. Status:

| Criterion | Status |
|---|---|
| Zone-level actions excited in the data | **met** (§3) |
| Paired objective trains stably at scale on all folds | **met** (§5, §7) |
| Paired objective lowers validation Δ error without hurting state accuracy | **met on validation** (§7.1), within the limits in §8 |
| Response gain on held-out buildings close to 1 (Test-A, Test-B, by family and regime) | **not measured** |
| No over-response a planner could exploit (Phase 2B: SAT → tank 2.5×, SAT → gas 1.5×) | **not measured** |
| Ensemble spread flags bad Δ predictions | **not measured** |
| Nominal accuracy no worse than Phase 2A / 2B on their test windows | **not measured** |

**Decision: PENDING.** Training was a necessary step and succeeded, but the MPC-readiness question can only be answered by the held-out evaluation.

---

## 10. Next steps

1. **Held-out counterfactual evaluation** (proposal Workstream 3). Extend `scripts/counterfactual_eval.py` (or add `scripts/phase2c_eval.py`):
   - load the `results/phase2c/training/<arch>_<objective>_lam<λ>/fold_<i>_*/model_seed<s>.pt` layout, and **only checkpoints that have a `train_log_seed<s>.json`**;
   - evaluate every held-out building on its Test-B points (`denver_*`, never trained on by any fold) and its Test-A points;
   - report pooled gain, direction, latency, spurious response and nRMSE per action family, per horizon (1 / 3 / 6 / 12 / 24 / 36) and per regime (morning recovery, occupied heating, cold);
   - compare against the Phase 2A GRU and Phase 2B graph checkpoints, the `arms` controls, and a zero-response model;
   - check the 5 effective-change NOOP arms (§3.1).
2. **Nominal regression:** run `scripts/evaluate_phase2b.py` on the Phase 2C checkpoints.
3. **Ensemble uncertainty** (Workstream 4): compare the 3-seed graph Δ spread with Δ error on the test arms.
4. **Only if step 1 shows a clear gain improvement,** consider:
   - a longer schedule (for example 24 epochs) as a protocol change (version bump, retrain the controls with it);
   - more seeds for the controls;
   - a λ sweep on a second fold (b2).

---

## 11. Reproduction

```bash
# collection (Kaggle CPU, kaggle/phase2c_collect_counterfactual.ipynb)
python scripts/phase2c_collect.py --config configs/phase2c/collection.json --out <OUT> --scenarios all --parallel 4 --time-budget-hours 10.5
python scripts/phase2c_collection_qa.py --out <OUT>
python scripts/phase2c_splits.py --data <OUT> --out <split_manifest.json> --heldout-weather denver

# training (Kaggle 2 x T4, kaggle/phase2c_train.ipynb; rerun with the previous output attached until "ALL DONE")
python scripts/train_paired.py --config <run_config.json> --out results/phase2c/training --parallel 3 --device cuda \
    --deadline <unix> --jobs <arch>:<objective>:<fold>:<seed>:<lambda> ...
```

**Inputs and outputs:**
- **Data input:** `data/phase2c_train.zip`, which holds `phase1_sample/` and `phase2c_counterfactual/`.
- **Training outputs:** `data/phase2c_training.zip` (session 1), `data/phase2c_training_2nd_run.zip`, `data/phase2c_training_3rd_run.zip` (final; installed as `results/phase2c/`).
- **Session-to-session resume:** the `data/phase2c_training_resume*.zip` files are the outputs repacked under a `phase2c_training/` folder, so the notebook's resume search finds them.
