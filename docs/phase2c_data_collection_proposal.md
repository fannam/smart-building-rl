# Phase 2C: Data Collection Proposal and Implementation Plan

**Status:** proposal only. **No simulator data has been collected for Phase 2C. Collection needs explicit approval.**

Every number below is measured from existing artifacts (Phase 1 data, the Phase 2B counterfactual set and its run logs) unless it is marked as an estimate.

---

## 1. Reusable infrastructure (no rebuild needed)

| Component | Location | Phase 2C use |
|---|---|---|
| Bit-exact branching (env copy sharing SuperLU + global RNG snapshot/restore) | `src/counterfactual/branch.py` | E9 collection; verified on 68/68 branch points |
| Simulator driver recording the Phase 1 quantities | `src/counterfactual/simrun.py` | unchanged |
| Interventions, commanded vs effective deltas, no-op flags, stratified branch points | `src/counterfactual/interventions.py` | extend with zone-subset arms (§5) |
| Collector (main trajectory + arms, replay checks) | `scripts/counterfactual_run.py` | add a `--design phase2c` profile; Phase 2B defaults unchanged |
| Arm → model-input builder (runtime context, effective actions, schedule features) | `src/counterfactual/model_eval.build_items` | reused as the paired-training item builder |
| Paired-response metrics (gain, direction, latency, spurious, normalized error) | `src/counterfactual/model_eval.py`, `scripts/counterfactual_eval.py` | add RMSE/nRMSE, per-horizon gains and a split filter |
| GRU / graph models, rollout, multitask state loss, LOBO folds, train-only normalizer | `src/dynamics/*`, `src/graph/*` | unchanged; the paired loss is added on top |
| Device-aware training + Kaggle notebook | `scripts/train_*.py`, `kaggle/phase2b_train.ipynb` | template for the Phase 2C training notebook |
| Ensembles, calibration, latent/action OOD scores | `src/uncertainty/scores.py` | add counterfactual-Δ uncertainty |
| Occupancy forecaster | `src/occupancy/forecast.py` | not needed for Phase 2C training |

---

## 2. Coverage gaps in existing data (measured)

### 2.1 Phase 1 observational data
60 episodes × 2016 steps.

| Family | Effective value at floor/zero (or cap) | Steps where it changes from the previous step (baseline / moderate) | Moderate profile: effective deviation from the baseline action |
|---|---|---|---|
| Zone damper | 75 % at the 0.1 floor | 25.6 % / 25.3 % | **1.2 %** |
| Reheat valve | **93 % at 0** | 6.6 % / 7.4 % | **0.8 %** |
| Recirculation | **93 % at the 0.7 cap** | 2.9 % / 2.7 % | **0.2 %** |
| HWS dp | 84 % at 0 | 15.6 % / 17.2 % | 2.4 % |
| Static pressure | 60 % at 0 (AHU off) | 2.2 % / 4.0 % | 18.9 % |
| SAT | — | 6.6 % / 10.1 % | (perturbed by moderate) |
| HWS setpoint | — | 2.8 % / 10.7 % | (perturbed by moderate) |

**Operating context:**
- AHU on: 40 % (baseline) / 48 % (moderate) of steps.
- HWS on: 16 % / 18 %.
- Heating active with an AHU on: 14 % / 22 %. In those steps the valve is > 0.01 in only 28 % / 20 % of zone-steps.

**Interpretation:**
- Zone actuators are almost never moved *independently* of the baseline's closed-loop state feedback; the exogenous excitation is ≈ 1 %.
- When they do move, they move as a deterministic function of the state. Their effect is therefore confounded with the state features the model already sees.
- This is why both models learned ≈ zero response (Phase 2B §5.4).
- **Observational Phase 1 data is not counterfactual supervision and will not be presented as such.**

### 2.2 Phase 2B counterfactual set
`results/phase2b/counterfactual/sim`, 60 MB: 12 scenarios, 68 branch points, 46 interventions + nominal.

| Family | Arms | Effective no-ops | Arms with a meaningful zone response | Responsive zone elements |
|---|---|---|---|---|
| reheat_valve | 408 | 105 | 189 | 7,540 |
| damper | 408 | 80 | 253 | 11,832 |
| sat | 408 | 93 | 163 | 6,740 |
| static_pressure | 408 | 235 | 164 | 7,480 |
| recirculation | 408 | 155 | 160 | 7,046 |
| hws_setpoint | 408 | 43 | 76 | 2,394 |
| hws_dp | 408 | 107 | 195 | 5,427 |
| ahu_run | 136 | 50 | 86 | 4,145 |
| hws_run | 136 | 56 | 33 | 1,088 |
| **Total** | 3,128 | 924 | — | — |

**Is the existing data sufficient to test paired training?**
- **For a pipeline pilot, yes.** It is genuine paired counterfactual data with common random numbers.
- **For a reliable model, no.** Its limits:
  - **Size:** 68 branch points, about 11 per building, so roughly 57 training points per LOBO fold.
  - **Intervention design:** every intervention is *building-wide*, with all zones moving together. The model can't separate one zone's actuator from its neighbours'.
  - **Strata:** one branch point per stratum per scenario; heating-relevant states are thin (12 occupied-heating + morning-recovery points).
  - **Scenarios:** only 12 of 30 are covered.
  - **Its role:** it is the **untouched Phase 2B evaluation set**. Training final models on it would remove the only independent counterfactual test set.

**Proposal:**
- Keep the Phase 2B set **untouched as Test-A** for all final Phase 2C models.
- Allow a *clearly labelled* pilot to train on it only under a LOBO split (pilot models are never final models). This is used only to validate the implementation while approval is pending.

---

## 3. Proposed experiments: E9 paired branches carrying E1/E2 zone excitation

**Design choice.** The E1/E2 excitation runs **inside E9 branches**, not as standalone closed-loop exploration episodes:
- The main trajectory stays the unmodified baseline (bit-exact to Phase 1).
- Every perturbation lives in a 3 h arm that is discarded after recording.
- This gives paired supervision for every excitation event, with no comfort risk to any persistent trajectory.
- A standalone closed-loop E1/E2 profile (3-day windows) is listed as optional **E1b/E2b** and is not requested now.

### 3.1 Branch points
- **Scenarios:** all **30** (5 weather files × 6 buildings) instead of 12, which adds weather diversity.
- **Branch points:** **10 per scenario, i.e. 50 per building and 300 in total.** That is 4.4× Phase 2B.
- **Stratum weights** (heating-relevant states emphasized; weights are renormalized when a stratum is absent):

  | Stratum | Weight |
  |---|---|
  | morning_recovery | 3 |
  | occupied_heating | 3 |
  | occupied_cooling | 1 |
  | occupied_other | 1 |
  | unoccupied_night | 1 |
  | unoccupied_other | 1 |

- **Extra E2 eligibility tag** (recorded, not required): heating active, tank ≥ 330 K, at least one AHU on.
- **Spacing:** branch points are ≥ 49 steps apart (context 12 + horizon 36 + 1). No branch point lies within ±48 steps of a Phase 2B branch point in the same scenario, so Test-A stays leakage-free.
- **Horizon:** 36 steps (3 h); context 12. This matches Phase 2A/2B.

### 3.2 Arms per branch point: 20

| Group | Arm | Members | Change | Hold |
|---|---|---|---|---|
| — | nominal | — | baseline closed-loop actions | — |
| **E2 reheat valve** | 3 × subset increase | random 30 % of zones (≥ 1) | set to a level drawn from {0.3, 0.6, 1.0} | 3 / 6 / 12 steps (15 / 30 / 60 min) |
| | 1 × subset decrease | random 30 % of the zones with valve > 0.05 | set to 0 | 12 steps |
| | 1 × building-wide | all zones | +0.4 | 36 steps (comparable to Phase 2B) |
| **E1 zone damper** | 3 × subset increase | random 30 % of zones | set to a level drawn from {0.3, 0.6, 0.9} | 3 / 6 / 12 steps |
| | 1 × subset decrease | random 30 % of the zones with damper > 0.2 | set to 0.1 | 12 steps |
| | 1 × building-wide | all zones | +0.3 | 36 steps |
| **AHU** | SAT | all AHUs | ±U(2, 5) K, sign random | 12 steps |
| | Static pressure | all AHUs | +300 Pa | 12 steps |
| | Recirculation | all AHUs | −0.3 (effective: the 0.7 cap makes increases mostly no-ops) | 12 steps |
| | AHU run | all AHUs | flip the current run state | 12 steps |
| **Plant** | HWS setpoint | plant | ±10 K, sign random | 12 steps |
| | HWS dp | plant | +4 | 12 steps |
| | HWS run | plant | flip | 12 steps |
| **No-op control** | damper command | zones at the floor | 0.05 (effective stays 0.1) | 12 steps |

**Arm rules:**
- **Seeding:** every random draw (zone subset, level, sign) is seeded by `(seed, scenario, t_b, arm)`, so the design is deterministic and reproducible.
- **Precedence:** decreases and "flip" use the *nominal* value at the branch step. If no zone qualifies, the arm is recorded as an effective no-op and kept.
- **Bounds:** native commands are clipped to the env bounds. Unperturbed entries stay bit-identical (existing `apply_intervention` guarantee).

### 3.3 Schema
One `.npz` per scenario, extending the Phase 2B schema.

**Identity and timing:**
- building, scenario, episode_id of the source trajectory, `branch_id = <scenario>@<t_b>`, `t_b`;
- `sim_timestamp[t_b]` and calendar (offline metadata only).

**Inputs (runtime only):**
- `ctx_obs` [P, 12, D];
- `ctx_ac`, `ctx_ad` (executed action history).

**Arm actions:**
- `arm_ac`, `arm_ad` [P, A, H, ·] (commanded);
- `arm_eff` (effective native);
- `cmd_delta`, `eff_delta` vs nominal.

**Per-arm metadata:**
- family, group, kind, level/value, duration, `subset_mask` [V];
- `clipped` (the command hit an env bound);
- `saturated` (effective ≠ commanded because of the floor/cap/AHU-off);
- `effective_noop`.

**Targets** (nominal = arm 0; the training target is Δ = arm − nominal):
- `zone_temp`, `tank`, `energy` (fan, cooling, pump, gas);
- `reward` and its components;
- `zone_occ`, `zone_sp` (diagnostics only).

**Regimes:**
- stratum, OAT band, heating/cooling active, AHU on, occupied, E2-eligible.

**Checks:** nominal-replay bit-exactness per point; main-trajectory equality with the Phase 1 episode.

### 3.4 Splits (sibling arms never separated)
- **Split unit:** `(scenario, day-of-week block)`.
- **Grouping:** all arms and branch points within the same scenario-day stay together, so overlapping context/horizon segments from one source trajectory can't straddle splits.

| Split | Content | Use |
|---|---|---|
| **Test-A** | Phase 2B set (68 points, 12 scenarios, building-wide arms) | untouched final test |
| **Test-B (held-out scenarios)** | all new points of the `denver_*` scenario of every building (≈ 60 points) | untouched final test |
| **Train / Val** | the remaining 24 scenarios; day-blocks assigned 80 / 20 with a fixed seed | training / λ and checkpoint selection |
| **LOBO** | for each fold, train/val use only the 5 training buildings | held-out building tested on its Test-A + Test-B + all its other points |

The manifest is written to `configs/phase2c/split_manifest.json` with branch ids, splits, fold membership and a digest. A test asserts that no `branch_id` or scenario-day appears in two splits.

---

## 4. Volume, runtime and compute (estimates from Phase 2B measurements)

**Measured Phase 2B cost per arm** (12 concurrent workers on this 32-core machine, including the amortized main-trajectory replay):

| Building | b8 | b7 | b1 | b4 | b5 | b2 |
|---|---|---|---|---|---|---|
| Seconds per arm | 6.7 | 7.6–7.9 | 12.5 | 17.7 | 17.6–17.8 | 22.2 |

The mean simulator step is 31–120 ms per building; the env copy takes 0.4–2.8 s.

| Quantity | Estimate |
|---|---|
| Branch points × arms | 300 × 20 = **6,000 arms** (≈ 1.9× Phase 2B's 3,196) |
| Process time | 1,000 arms per building × the per-arm costs above ≈ **23–26 core-hours** |
| Wall time, local (12 workers) | **≈ 2–3 h** |
| Wall time, Kaggle CPU session (4 cores) | ≈ 6–8 h; can be split into 2 sessions by building |
| Disk | Phase 2B used ≈ 19 KB per arm, so **≈ 120 MB** in total (new root, gitignored) |
| Peak memory per worker | ≤ 2 env copies alive (b2 ≈ 170 MB each) + base ≈ 1–1.5 GB; 12 workers ≈ 15 GB |

**Expected coverage gain:**

| Family | Phase 2B | Proposed | Notes |
|---|---|---|---|
| Valve | 408 building-wide arms → 7.5k responsive zone elements | 300 × 4 subset arms × 30 % × V̄ ≈ 54 → **≈ 19k independently perturbed zone events** (+ 300 building-wide arms) | effective-change and responsive counts are reported after collection |
| Damper | as above (11.8k responsive elements) | **≈ 19k** independently perturbed zone events | same |
| AHU and plant families | 136–408 arms | **300 arms each**, spread over 30 scenarios | — |

Phase 1 observational data gives ≈ 0.8–1.2 % exogenous zone excitation by comparison.

**Phase 2C training compute** (Kaggle GPU; estimate, staged):

| Stage | Runs |
|---|---|
| λ_cf sweep | {0.1, 0.3, 1.0} × {GRU, graph} on the training/validation branch points of one fold: 6 short runs |
| Data-only controls (A+, C+: state loss with arm trajectories used as ordinary windows) | 2 folds (b2, b8) × 2 models = 4 runs |
| Paired models B, D | 6 folds × 2 = 12 runs |
| Graph ensemble E | 2 extra seeds × 6 folds = 12 runs |

Total ≈ 34 runs. Each run is expected at about 1.3–2× Phase 2B's 20–83 min (each training group adds 19 arm rollouts). **≈ 15–20 h of Kaggle GPU wall time over 2–3 sessions.** The sweep and ablations go first, so the full stage can be cancelled early if the pilot shows no effect.

---

## 5. Comfort guard

- **The persistent trajectory is never perturbed.** It is the baseline policy, verified bit-exact against Phase 1.
- **Arms are offline counterfactuals.** They last ≤ 3 h and are discarded after recording. Comfort degradation inside an arm affects nothing persistent; it is part of the response the model must learn.

**Data-quality guards:**
1. **Bounded holds:** zone holds ≤ 60 min (12 steps), except the building-wide 36-step arms that reproduce the Phase 2B design.
2. **Bounds:** all levels lie inside the env bounds; floors and caps are recorded as `saturated`, not hidden.
3. **Excursion logging:** arms whose comfort excursion is extreme (any occupied zone > 5 K outside the band, the reward's deviation reference) are kept, flagged `extreme_excursion`, and reported separately. They aren't dropped silently.
4. **Runaway check:** NaN/inf or a tank outside 280–370 K aborts the arm, logs it and continues.

**If standalone closed-loop E1b/E2b episodes are approved later:** they reuse the Phase 1 explorer's occupied-zone guard (`guard_k`) unchanged, under a new profile name.

---

## 6. Exact collection commands (as implemented)

The collection runs **only on Kaggle CPU** via `kaggle/phase2c_collect_counterfactual.ipynb`, which calls:

```bash
# validate config + plan (no simulation)
python scripts/phase2c_collect.py --config configs/phase2c/collection.json --out <OUT> --scenarios all --validate-only
# collect (resumable; no new branch point is started after the time budget)
python scripts/phase2c_collect.py --config configs/phase2c/collection.json --out <OUT> --scenarios all --parallel 4 --time-budget-hours 10.5
# QA + manifest, then the split manifest
python scripts/phase2c_collection_qa.py --out <OUT>
python scripts/phase2c_splits.py --data <OUT> --out configs/phase2c/split_manifest.json
```

The proposal referred to `scripts/counterfactual_run.py --design phase2c`. It was implemented instead as `scripts/phase2c_collect.py` + `src/counterfactual/phase2c.py`, which reuse `counterfactual.branch`, `simrun`, `interventions` and `effective_action`. This leaves the Phase 2B script untouched.

### Implementation notes (after the approved dry run, 2026-11-19)

1. **Arm count.** The arm table lists 18 interventions + nominal = 19 arms, while the text said 20. SAT is now two arms (+ and −, magnitude U(2, 5) K), giving 20 arms per branch point.
2. **AHU state gating.** With every AHU off, damper and valve changes have *exactly* zero effect (dry run), and only 141/300 planned points had an AHU on. So:
   - within each stratum, AHU-on steps are preferred; stratum counts are unchanged, and AHU-on points rise from 141 to 260/300;
   - E1/E2 subset arms draw only from zones whose AHU is on at t_b (otherwise they are flagged no-ops);
   - a per-arm `active_change` flag is recorded.
3. **NOOP control.** It overrides only the (step, zone) entries whose nominal damper is at the floor at that step. It is a guaranteed effective no-op (tested).
4. **Copied arms.** Arms whose commands are bit-identical to the nominal are copied from the nominal instead of re-simulated. Common random numbers make them identical (tested).
5. **Measured speed.** Unloaded on this machine: about 25–30 s per 20-arm branch point for b7/b8. The estimate for all 300 points is about 7 core-hours, i.e. about 2 h at 4 workers locally and about 3–4 h on a Kaggle CPU session (slower cores). This is below the 6–8 h in §4.

## 7. Minimum code changes and implementation plan

### Workstream 1: collection (code only until approval)
1. `src/counterfactual/interventions.py`: add `zone_subset` interventions (`set`/`add` with a seeded subset mask, qualifying-zone filters, flip), and `phase2c_arms()`. The Phase 2B `default_interventions` stays unchanged.
2. `scripts/counterfactual_run.py`: add `--design phase2c`, `--config`, `--n-points`, `--exclude-near`, scenario `all`, stratum weights, and the extra schema fields from §3.3. The Phase 2B path stays the default.
3. `scripts/phase2c_splits.py`: split manifest + leakage assertions.

### Workstream 2: paired training
4. `src/counterfactual/dataset.py`: branch-group dataset built on `model_eval.build_items` (one item = context + nominal + all arms of a branch point, never split). It provides Δ targets and train-only Δ normalization statistics, computed per target over *meaningful* responses.
5. `src/dynamics/paired.py`: the paired loss.
   - Both arms are rolled out autoregressively from the **same** context with the **same** parameters. The nominal and intervention batches are stacked, so they share one forward pass.
   - The loss is Huber on normalized Δ̂ − Δ over **all 36 steps**, and per-horizon gains are logged.
   - No-op arms get a separate, down-weighted consistency term (Δ̂ → 0, weight 0.1). Near-zero responses are masked from the gain statistics but kept in the loss.
   - Combined objective: L = L_state (unchanged, Phase 1 windows) + λ_cf · L_cf + λ_reg · L_reg, where L_reg is weight decay, as before.
6. `scripts/train_paired.py` (device-aware, `fold:seed` jobs like the existing scripts) and `kaggle/phase2c_train.ipynb`. The models are the existing classes, fine-tuned from scratch per the spec comparison table (A/C = the existing Phase 2A/2B checkpoints; A+/C+ = data-only controls).
7. Tests:
   - identical context and parameters for both arms;
   - no future observation or target access (NaN poisoning);
   - Δ alignment with the simulator arrays;
   - no-op Δ̂ = 0 for identical inputs;
   - split integrity;
   - subset-intervention semantics.

### Workstream 3: validation
8. Extend `scripts/counterfactual_eval.py`:
   - `--split` filter (Test-A, Test-B, LOBO held-out);
   - a zero-response model (already present as `persistence`, but to be named explicitly);
   - RMSE/nRMSE, per-horizon gains (1/3/6/12/24/36), latency mismatch;
   - magnitude/duration/regime tables;
   - the excluded near-zero counts.
9. Nominal regression: reuse `scripts/evaluate_phase2b.py` (Phase 2A windows) for every new checkpoint.

### Workstream 4: uncertainty
10. `src/uncertainty/scores.py`: ensemble Δ spread vs Δ error on test arms (Spearman, AUROC on the worst 10 %, threshold precision/recall chosen on Val, interval coverage). Compare it with latent distance, action distance and regime indicators, plus an explicit "all members wrong in the same direction" analysis.

### Order
- **Now, while approval is pending:** steps 1–5, 7 and 8.
- **After that**, a *labelled pilot*: LOBO paired fine-tuning on the Phase 2B set, used only to check that gains move off zero. It isn't a final result, and Test-A is then not used to evaluate pilot models.
- **After approval:** the dry run, then the full collection, then the splits, then the staged Kaggle training (§4), then evaluation and the report.

---

## 8. Approval requested

1. **Approve E9 + E1/E2-in-branch collection as specified:** 300 branch points × 20 arms, 30 scenarios, `data/phase2c_counterfactual/`, ≈ 120 MB.
2. **Where to run it:** locally (≈ 2–3 h, 12 workers) or on Kaggle CPU (≈ 6–8 h, 2 sessions).
3. **Pilot:** may the implementation pilot train on the Phase 2B set under LOBO (Test-A remains untouched for all final models)?
