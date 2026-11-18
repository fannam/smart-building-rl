# Latent MPC for the Smart Buildings Challenge — Implementation Plan

> **Status:** proposal for future work. Nothing in this document has been implemented.
> **Date:** 2026-11-18 · **Repo state audited:** `3fb47b5` (main) + untracked `results/kaggle_run/`
> **Scope:** design and roadmap only. All "phases" below are proposals, gated by go/no-go checks.
>
> **Phase 0 update (2026-11-18):** see [`phase0_findings.md`](phase0_findings.md). It supersedes the runtime assumption (R5, §6.6), the cost-model notes (§6.4) and the open questions (§12) below where they differ.
>
> **Phase 1 update (2026-11-18):** data-collection infrastructure is implemented. See §14 for the actual changes, and [`phase1_dataset.md`](phase1_dataset.md) for the dataset schema. The user-facing "Phase 1" covers the data logger and exploration from roadmap Phase 3, plus the zero-offset regression hook from roadmap Phase 1, at reduced scale (shipped buildings only, no generated buildings yet).

## 0. TL;DR

- **Score to beat:** our copy of the public analytical baseline (`WideHotStorageReservePolicy`, `src/baseline_policy.py`) scores **0.00695** on all 30 training scenarios at full horizon. The run is deterministic: it gave the same number locally and on Kaggle.
- **Where the loss is:** energy (0.00353) + carbon (0.00314) = **96%** of the score. Comfort is only 0.00027. Three scenarios (`building_train_1/fairbanks_1` 0.0465, `building_train_8/fairbanks_8` 0.0217, `building_train_1/phoenix_5` 0.0143) contribute **~39%** of the mean. Cold-climate heating (gas → carbon) is the largest single lever.
- **Why multi-step MPC can help:** the baseline is a **one-step** greedy search. It cannot plan pre-heating or pre-cooling against time-of-use tariffs, the timing of the hot-water tank charge (storage dispatch is currently hand-tuned hysteresis), night-setback depth versus morning recovery cost, or AHU on/off timing. All of these are intertemporal trade-offs.
- **Recommended architecture ("structured latent MPC"):**
  1. Per-zone and per-AHU latent states, anchored to physical quantities (the observed zone temperature is one latent coordinate). They are evolved by the baseline's physics update plus a **shared, size-agnostic learned residual** (a small message-passing net on the zone↔AHU graph), and conditioned on a **building context vector** inferred online.
  2. A **supervisory action space**: per-AHU comfort-band offsets, a run/off override, and a building-level water/storage target. The planner searches over these, not over raw `2V+3K+2` actuators.
  3. **Sampling MPC** (CEM/MPPI), blocked controls, horizon 1–3 h, replanning every 15 min. Inside the planner, a distilled surrogate stands in for the low-level allocator.
  4. The **baseline's analytical allocator** turns the chosen supervisory action into the full action vector at execution time. The baseline itself is the always-available fallback.
- *(Superseded by Phase 0: in `per_rollout` mode the measured allowance is ≈ 140 ms/step on the smallest building, design target ≈ 70 ms/step. The ≈ 25–30 ms figure applies only if the hidden rollouts turn out to run sequentially inside the notebook, which is unconfirmed.)* **Binding constraint is runtime, not modelling.** If the 30 hidden rollouts run sequentially inside the 9 h Kaggle notebook limit, the extra policy budget is only about **25–30 ms/step** above the baseline. The design must fit that, or plan at a lower rate.
- **First step (Phase 0) is a headroom study, not a model:** simulator-oracle lookahead on the training scenarios to measure how much multi-step planning *could* gain before investing in learned models.

---

## 1. Repository audit

### 1.1 What exists

| Path | What it is | Notes |
|---|---|---|
| `smart_buildings_competition_notes.md` | Consolidated rules, interface, scoring notes | Source for §2 |
| `baseline/example-smart-building.ipynb`, `baseline/README.md` | Public competitor notebook (not our work), with attribution | README: source URL still TODO |
| `src/baseline_policy.py` (2148 lines) | Cell 2 of the notebook extracted verbatim. About 40 classes in a multiple-inheritance chain ending in `WideHotStorageReservePolicy` | NumPy only; online RLS model ID; 1-step search |
| `src/run_eval.py` | Parallel evaluator: per-scenario mean reward, comfort/energy/carbon split, policy vs. simulator wall time, per-step `.rewards.npy` | Reuse as-is for every future evaluation |
| `src/summarize.py` | Leaderboard-style aggregation | Reuse |
| `scratch/marginal.py` | Marginal elec vs. gas reward penalty per watt by hour/day (uses public tariff classes) | Shows the TOU structure an MPC could exploit |
| `results/baseline_full/` | 30 baseline runs + 1 `bang_bang` run (full horizon) | §1.2 |
| `results/kaggle_run/smart-building-17nov.ipynb` (untracked) | Kaggle reproduction of the baseline eval | Same 0.00695 score |
| `starter-kit/` (commit `1fb75606`, 2026-11-16) | Official simulator, baselines, evaluator, generators | Simulator in `smart_buildings/smart_control/` |

### 1.2 Measured baseline (full 8640-step horizon, 30 training scenarios)

| Metric | Value |
|---|---:|
| Leaderboard-style score (lower is better) | **0.00695** |
| Comfort / energy / carbon parts | 0.00027 / 0.00353 / 0.00314 |
| Worst scenario | `building_train_1/fairbanks_1` = 0.04652 (comfort 0.00494, carbon 0.02970) |
| Policy time per rollout (local, 16 parallel workers) | 51–140 s (≈ 6–16 ms/step) |
| Simulator time per rollout (same conditions) | 630–2520 s (dominates wall time) |
| Kaggle gateway smoke test (README) | ≈ 805 s per rollout (building_train_1/atlanta_2) |
| `bang_bang` (1 scenario only) | 0.2028, wall 1271 s, policy 2.7 s |

Per-scenario breakdown (score; comfort/energy/carbon; K/V):

| Scenario | Score | c / e / k | K/V |
|---|---:|---|---|
| b1/atlanta_2 | 0.00274 | .00004/.00127/.00144 | 4/39 |
| b1/denver_7 | 0.00157 | .00006/.00064/.00087 | 4/39 |
| **b1/fairbanks_1** | **0.04652** | .00494/.01188/.02970 | 4/39 |
| **b1/phoenix_5** | **0.01426** | .00008/.01153/.00265 | 4/39 |
| b1/singapore_8 | 0.00620 | .00006/.00492/.00122 | 4/39 |
| b2/atlanta_3 | 0.00279 | .00005/.00218/.00056 | 9/85 |
| b2/denver_8 | 0.00359 | .00009/.00112/.00238 | 9/85 |
| b2/fairbanks_2 | 0.01080 | .00068/.00287/.00725 | 9/85 |
| b2/phoenix_6 | 0.00722 | .00006/.00582/.00134 | 9/85 |
| b2/singapore_1 | 0.00470 | .00006/.00373/.00090 | 9/85 |
| b4/atlanta_5 | 0.00632 | .00006/.00508/.00118 | 6/54 |
| b4/denver_2 | 0.00468 | .00007/.00133/.00328 | 6/54 |
| b4/fairbanks_4 | 0.00113 | .00004/.00052/.00056 | 6/54 |
| b4/phoenix_8 | 0.00124 | .00004/.00069/.00051 | 6/54 |
| b4/singapore_3 | 0.00662 | .00007/.00533/.00122 | 6/54 |
| b5/atlanta_6 | 0.00408 | .00008/.00325/.00076 | 8/79 |
| b5/denver_3 | 0.00337 | .00008/.00099/.00229 | 8/79 |
| b5/fairbanks_5 | 0.00137 | .00003/.00104/.00031 | 8/79 |
| b5/phoenix_1 | 0.00197 | .00007/.00073/.00116 | 8/79 |
| b5/singapore_4 | 0.00531 | .00008/.00425/.00098 | 8/79 |
| b7/atlanta_8 | 0.00586 | .00009/.00167/.00410 | 4/36 |
| b7/denver_5 | 0.00425 | .00006/.00334/.00085 | 4/36 |
| b7/fairbanks_7 | 0.00720 | .00006/.00186/.00528 | 4/36 |
| b7/phoenix_3 | 0.00544 | .00004/.00437/.00102 | 4/36 |
| b7/singapore_6 | 0.00547 | .00004/.00440/.00103 | 4/36 |
| b8/atlanta_1 | 0.00403 | .00005/.00122/.00277 | 3/30 |
| b8/denver_6 | 0.00309 | .00004/.00240/.00064 | 3/30 |
| **b8/fairbanks_8** | **0.02168** | .00077/.00563/.01529 | 3/30 |
| b8/phoenix_4 | 0.00963 | .00007/.00777/.00179 | 3/30 |
| b8/singapore_7 | 0.00528 | .00008/.00421/.00100 | 3/30 |

### 1.3 Audit findings and gaps

1. **`bang_bang` reference is incomplete.** Only 1 of 30 runs exists, and `results/bang_bang_rest.log` holds only its header. The runtime budget (10× `bang_bang` per rollout) cannot be quantified for every building until those runs finish.
2. **`bang_bang` return mismatch.** `starter-kit/baseline_returns.json` lists −1521.1 for b1/atlanta_2. Our run gives −1752.2. The reference file probably predates a simulator change. Use our own baseline runs as the reference, and record the starter-kit commit with every result.
3. **Runtime numbers were measured under contention** (16 workers). We need a single-process timing of sim vs. policy on a Kaggle-like CPU before fixing the planning budget.
4. **Baseline structure.** Behaviour is spread over overridden `_band`, `_choose_ahu`, `_water_target`, `get_action` across a deep MRO. The MPC will reuse three pieces: the observer/thermal model (`TransferPolicy._learn`, `_free_drift`, the delayed-observation roll-forward in `get_action`), the allocator (`_choose_ahu` variants), and the hydraulic allocation. To do that, the final MRO's effective versions of these methods must be identified and wrapped, not copied piecemeal.
5. **Storage-billing caveat** (baseline README). Tank charging to 340–350 K may exploit how the energy-balanced boiler tank is billed (`smart_control/simulator/hot_water_system.py`, `apply_heat_draw`, `energy_balance`). An MPC that optimizes the tank will exploit this more aggressively. Verify the accounting before designing around it, and watch for organizer patches.
6. **No tests, no config system, no data pipeline.** Everything for the learned components is greenfield.
7. **The policy sees no reward at evaluation time.** `get_action(obs)` only receives observations, so any cost model used by the planner has to be computed or learned offline. Tariffs are **not** in `building_meta_data` (the keys are action/observation indices, `floor_plan`, `zone_map`, `ahu_zone_assignments`, schedule, comfort/eco windows, time zone, holidays and normalization).

---

## 2. Competition requirements → design requirements

| # | Requirement (source) | Design consequence |
|---|---|---|
| R1 | One policy for unseen buildings: 2–4 floors, K = 3–9 AHUs, V ≈ 28–85+ zones, ≤ 10 zones per AHU (notes §4.3) | All learned parts must be **size-agnostic** (shared per-entity weights, set or graph aggregation); no fixed-size vectors |
| R2 | Unseen weather stations (notes §4.3) | Do not condition on station identity. Learn from outside-air temperature and time only, and validate leave-weather-out |
| R3 | No hard-coded building IDs or sensor signatures; code archives are scanned (notes §3.3) | Index everything via metadata field names; no per-building lookup tables |
| R4 | Per-rollout wall clock ≤ 10× `bang_bang` (paper) | Bounded per-step compute; deterministic worst-case cost |
| R5 | Kaggle notebook ≤ 9 h, internet off (notes §3.1). *Phase 0: whether hidden rollouts count against this is unresolved; treated as a separate budget, see phase0_findings §2* | If the 30 hidden rollouts run sequentially (≈ 805 s each for the baseline), the headroom is ≈ 9 min per rollout, i.e. **≈ 25–30 ms/step extra**. This is the tighter bound until proven otherwise |
| R6 | Every step is scored, no warm-up (notes §5.2) | Cold-start behaviour matters: fall back to baseline until online context or model estimates are trustworthy |
| R7 | Hybrid action: K+1 discrete, `3K+2+2V` continuous, interleaved indices, ventilation floors (notes §4.5) | Keep the baseline's verified index and normalization handling; the planner outputs supervisory knobs, not raw vectors |
| R8 | Delayed zone-temperature observation (baseline `observation_delay = 1`) | The latent filter must include a one-step roll-forward, as the baseline does |
| R9 | Offline artifacts only: weights packaged; NumPy works on Kaggle | Inference should be **NumPy-only** (export weights to `.npz`) to avoid torch import and startup cost and dependency risk |
| R10 | Score = mean over 30 scenarios of absolute mean reward; equal weight | Outlier scenarios (cold climates) dominate, so optimize for robustness and tail, not only the median |

---

## 3. Why the baseline leaves value on the table (hypotheses to verify in Phase 0)

| Lever | Baseline behaviour | Multi-step opportunity | Expected size |
|---|---|---|---|
| Pre-heating / pre-cooling vs. TOU tariff | Fixed `precondition_hours`, smoothstep warm-up | Choose start time and trajectory per day from weather and thermal mass | Medium; largest in fairbanks/phoenix |
| Hot-water tank dispatch | Hysteresis 340–350 K, standby hold 36–72 h | Charge when cheap and right before demand; coast otherwise | Medium (caveat §1.3-5) |
| Night/weekend setback depth | Constant 274 K floor, one-step recovery | Depth chosen against the next-morning recovery cost | Small–medium |
| AHU on/off | Greedy one-step "off if cheaper now" | Avoid on/off chattering and plan pulsed operation using thermal mass | Small–medium |
| Comfort spikes (worst step −0.895 in fairbanks_1) | Reactive | Anticipate cold snaps (outside air temperature trend) | Medium on outliers |
| Gas vs. electricity mix | Fixed `gas_cost_ratio = 0.30` | Time-varying ratio (`scratch/marginal.py` shows it varies by hour) | Small |

**Kill criterion:** if the Phase 0 oracle (perfect model, real simulator for lookahead) improves the training score by less than about 10% over 0.00695, a learned latent MPC is not worth its runtime and complexity. Invest in tuning and robustness of the baseline instead.

---

## 4. Literature (relevant prior work)

Citations are from memory. Verify the venue and year before citing in a paper.

**Latent world models + planning**
- PlaNet (Hafner et al., ICML 2019): RSSM latent dynamics, CEM planning in latent space.
- Dreamer / DreamerV3 (Hafner et al., 2020 / 2023): latent imagination with an actor-critic. Strong, but a policy-learning approach rather than MPC.
- TD-MPC (Hansen et al., ICML 2022) and TD-MPC2 (ICLR 2024): a decoder-free latent model trained with latent consistency, reward and value losses, and MPPI planning plus a terminal value. **This is the closest template for the learned part.**
- PETS (Chua et al., NeurIPS 2018): probabilistic ensembles + CEM. Uses ensemble disagreement as uncertainty, which is useful for the fallback gating.
- MPPI (Williams et al., 2017): information-theoretic sampling MPC. It parallelizes well in NumPy.

**Buildings-specific**
- Drgoňa et al., "All you need to know about model predictive control for buildings" (Annual Reviews in Control, 2020): RC grey-box models, economic MPC, TOU and pre-conditioning.
- Gnu-RL (Chen, Cai, Bergés, BuildSys 2019): a differentiable MPC policy for HVAC, initialized from expert data. Uses an MPC structure as a policy prior.
- Differentiable MPC (Amos et al., NeurIPS 2018); Differentiable Predictive Control / NeuroMANCER (Drgoňa et al., 2022): learn a policy offline by differentiating through the model. This is the amortized alternative if online planning is too slow.
- Koopman-based MPC (Korda & Mezić, Automatica 2018): a linear latent lifting gives fast QP-based MPC. An option for the dynamics core.
- The simulator lineage is Google's smart-buildings / `smart_control` RL environment (Goldfeder & Sipple et al.). Check the paper reference in the competition technical paper.

**Generalization across sizes and systems**
- Graph Network simulators (Sanchez-Gonzalez et al., ICML 2020); Set Transformer (Lee et al., ICML 2019): shared per-entity dynamics for variable topology.
- Rapid Motor Adaptation (Kumar et al., RSS 2021) and context-conditioned dynamics: infer a latent environment context online from history. Here that context is the building's thermal parameters.
- Residual policy learning (Silver et al., 2018; Johannink et al., 2019): learn corrections on top of a strong analytic controller. This is a fallback path if MPC headroom is small.
- Offline model-based RL with uncertainty penalties (MOPO, Yu et al., 2020; MOReL, Kidambi et al., 2020): penalize planning in regions the model has not seen.

**Takeaways applied here:** (a) decoder-free latent consistency training (TD-MPC2) avoids reconstructing a 1000+ dim observation; (b) building MPC succeeds with a *structured* low-order model plus economic cost; (c) variable topology calls for shared per-entity graph models; (d) when per-step compute is tight, planning should be amortized (distill the planner into a policy) or decimated.

---

## 5. Candidate architectures

### A. Grey-box latent MPC (physics-only)
Extend the baseline's per-zone one-step RC model with hidden states: slow thermal mass per zone and tank temperature. Identify them online with an EKF or RLS. Plan with MPPI over supervisory knobs.
- **Pros:** no training data needed, transfers by construction, cheap, interpretable.
- **Cons:** model error over 1–3 h horizons (the baseline's RLS is tuned for one step), a hand-built cost model, and limited capacity for coupling effects (inter-zone conduction, AHU mixing).

### B. Fully learned latent world model (TD-MPC2-style)
A graph encoder of the raw observation into per-entity latents, a learned latent dynamics model, reward heads, a value function, and MPPI over the raw continuous actions plus discrete bits.
- **Pros:** maximal flexibility; learns the true cost including tariffs.
- **Cons:** the action space is large (up to about 200 continuous dims) and hybrid, so planning is expensive and data-hungry. Simulation data is slow (≈ 5 steps/s per core), there is a large sim-to-unseen-building generalization risk, it is opaque, and it likely busts the R5 runtime.

### C. Structured latent MPC over supervisory actions (**recommended**)
A physics-anchored latent with a learned residual and an online context, a planner over a low-dimensional supervisory action, and the baseline allocator for execution (details in §6).
- **Pros:** reuses the validated low-level controller and index handling; the planning dimension is ≈ 2K+2 instead of 3K+2+2V+K+1; it degrades gracefully to the baseline; and the learned part is small, so it can run in NumPy.
- **Cons:** planner optimality is limited by the expressiveness of the supervisory space, and the allocator has to be wrapped cleanly.

### D. Amortized MPC (planner distilled into a policy)
Run C (or a simulator oracle) offline as a teacher and distill it into a graph policy network that outputs supervisory actions directly (DAgger-style).
- **Pros:** about 1 ms/step inference, so it satisfies R5 trivially.
- **Cons:** an extra imitation stage, and no online re-optimization.

### Comparison

| Criterion | A | B | C | D (on top of C) |
|---|---|---|---|---|
| Transfer to unseen buildings | High | Low–Med | High | Med–High |
| Expected gain over baseline | Low–Med | Unknown | Med | ≈ C minus distillation gap |
| Offline data/compute need | None | Very high | Medium | Medium–High |
| Per-step runtime | Low | Very high | Medium | Very low |
| Implementation risk | Low | High | Medium | Medium |
| Fallback to baseline | Natural | Hard | Natural | Natural |

**Decision:** C is the main path; A is its zero-data ablation (C with the residual disabled); D is the contingency if C cannot meet R5.

---

## 6. Recommended design (Architecture C) in detail

### 6.1 Entities and graph (built in `__init__` from metadata only)
- **Nodes:** zones (V), AHUs (K), one building node (outside conditions, time, occupancy, hot-water system).
- **Edges:** zone→AHU from `ahu_zone_assignments`; zone↔zone adjacency from `zone_map` (shared walls, reusing the connected-component logic in `TransferPolicy._geometry_priors`); AHU→building.
- **Static node features:** zone area, exterior-edge count / UA prior (from `_geometry_priors`), floor index if derivable, AHU group size. All computed from public metadata, so R3 is satisfied.

### 6.2 Latent state
Per zone `z_i = [T_i (observed, delay-compensated), m_i (hidden mass temperature), h_i (learned, d ≈ 4–8)]`; per AHU `a_k` (d ≈ 4); building `b = [T_tank (hidden, observed via the supply-water sensor), context c (d ≈ 8)]`.
- The first coordinates are **anchored to physics**, so the latent stays interpretable and the comfort cost can be computed exactly from predicted `T_i`.
- **Context `c`:** inferred online by a small recurrent or exponential-average encoder over (prediction residual, features) history. It captures building-specific envelope and material properties without IDs (RMA-style). It is initialized at the training-set mean, so the cold start equals the "average building".

### 6.3 Dynamics model
`z_{t+1} = f_phys(z_t, u_t, w_t) + g_θ(graph(z_t, u_t, w_t, c))`
- `f_phys`: the baseline's implicit one-step update `T' = (T + d + g(G·T_s + H)) / (1 + gG)`, extended with a two-node RC (air ↔ mass) and a tank energy balance.
- `g_θ`: a shared message-passing residual (2 rounds, hidden 32, about 10–20k params). It is applied with identical weights to every zone and AHU, so it is size-agnostic (R1).
- `u_t`: here, the **low-level** quantities produced by the allocator (airflow conductance, supply temperature, reheat power, AHU run bit). This keeps the model independent of the supervisory parameterization.
- `w_t`: exogenous inputs. Outside air temperature is extrapolated by persistence plus a learned diurnal trend; time features are known exactly; occupancy is a learned hour-of-week profile per building, updated online, combined with `comfort_mode_now/soon`.
- **Uncertainty:** a small ensemble (3–5 residual heads sharing a trunk). Disagreement drives the comfort reserve (generalizing the baseline's error-based reserve) and the fallback gate (§6.7).

### 6.4 Cost model (what the planner minimizes)
Per step: `J = 0.6·comfort + 0.2·energy_cost + 0.2·carbon`. This mirrors `SetpointEnergyCarbonRegretFunction` (`smart_control/reward/setpoint_energy_carbon_regret.py`).
- **Comfort:** computed analytically from predicted `T_i`, the comfort band and predicted occupancy, using the official shape (linear up to `w`, then quadratic). Read `productivity_deviation_ref` and the normalizers from the reward config. Verify which constants are global (shared by all scenarios) and which are building-specific. If they are building-specific and not in metadata, they have to be learned or bounded.
- **Energy / carbon:** fan, chiller and pump electricity plus boiler gas, predicted from low-level quantities. Use the baseline's physical proxies (`fan`, `chiller`, `gas` terms in `_choose_ahu`) refined by a learned correction head, multiplied by an hour-of-week **tariff and carbon profile**.
  - Tariffs are not in metadata. Option 1: use the public `ElectricityEnergyCost` / `NaturalGasEnergyCost` schedules (as `scratch/marginal.py` does). They are simulator-global, not building identifiers, but **confirm on the Rules/forum that this is allowed** (open question Q3). Option 2: learn a time-of-week cost head from training rollouts.
- **Terminal cost:** (phase 4+) a learned value `V(z_H, time)` trained by TD on rollouts of the MPC/baseline. It handles horizon truncation, for example tank state and mass temperature at the end of the horizon.

### 6.5 Supervisory action space (what the planner searches)
Per AHU k, per control block: `δlow_k, δhigh_k ∈ [−1.5, +1.5] K` offsets to the target band, and a run mode ∈ {auto, force-off}. Building-level: hot-water target ∈ {310 (coast), 320, 340, 350} K, or continuous in [310, 350].
Dimension ≈ 2K+1 continuous plus K discrete per block, i.e. ≤ 19 continuous for K = 9. These knobs feed straight into the baseline's `_band` / `_water_target` / `_choose_ahu` hooks, so the **allocator turns supervisory actions into the full action vector** with all index, normalization and ventilation-floor handling already validated.

### 6.6 Planner
- **Algorithm:** MPPI or CEM (2–3 iterations) with a warm start from the shifted previous plan plus the "zero-offset" plan, which is exactly the baseline's behaviour, so the planner can never select something it thinks is worse than the baseline.
- **Horizon and blocking:** H = 12–36 steps (1–3 h), in blocks of 3 steps (15 min), so 4–12 decision blocks. A longer horizon for tank and pre-heat decisions can come from a coarse second stage (hourly blocks to 6 h) or from the terminal value.
- **Replanning:** every 3 steps (15 min) and immediately on events (comfort-mode transition, a large prediction error). Between replans the allocator executes the committed plan with fresh observations.
- **Inner loop cost:** running the real allocator (`_choose_ahu`, 22+ candidates × zones) inside every rollout step is too slow. Instead, **distill a surrogate** `π_alloc(band, state) → (G, T_s, H, run, energy)` per AHU (a small MLP, shared across AHUs) trained on the baseline's own decisions. The planner uses the surrogate; execution uses the real allocator.
- **Budget sizing (target ≤ 25 ms/step averaged on Kaggle CPU):** N = 64 samples × 12 blocks × 3 steps = 2304 model steps per replan, vectorized over samples and zones (V ≤ 85, latent ≤ 12). Amortized over 3 steps, that has to fit in about 75 ms per replan. This is plausible in NumPy but **must be benchmarked in Phase 2**; if not, drop to N = 32, H = 12, or switch to D.

### 6.7 Safety, fallback, and cold start
- Steps 0 to ~288 (first day), or while ensemble disagreement or online residual error is above a threshold: use pure baseline actions and keep updating the context and model.
- Always evaluate the zero-offset (baseline) plan; choose the MPC plan only if its predicted cost improvement exceeds a margin scaled by uncertainty.
- Hard clamps: band offsets never widen the band during occupied comfort mode by more than X K. The allocator's ventilation floors stay in force.
- Watchdog: if the cumulative planner time exceeds the budget slope, switch to baseline-only for the rest of the episode (this protects R4/R5).

### 6.8 Online adaptation
- Keep the baseline's RLS on the physics coefficients; this is the A-level adaptation.
- Context `c` updated by the encoder from residual history.
- No gradient updates online (cost and rule uncertainty, see §2/notes §9).

---

## 7. Data and training pipeline (future)

1. **Building generation:** `python -m smart_buildings.run_floor_generator --start-seed 500 --count N` for about 50–200 buildings spanning K and V ranges; scenarios via `run_scenario_generator` across all 6 stations, including `minneapolis` (unused by the shipped set and a good held-out-weather proxy).
2. **Behaviour policies for data:** baseline plus **randomized supervisory perturbations** (random band offsets, tank targets, forced off periods). This covers states the MPC will visit. Add `pid_water` and `fixed_setpoint` for diversity. Avoid `bang_bang`-like extreme regimes beyond a small fraction.
3. **Logging:** extend `src/run_eval.py`'s loop pattern (it already captures reward components) to also store obs, allocator low-level outputs, supervisory knobs, the action and the reward breakdown per step, as compressed `.npz` per rollout.
4. **Throughput estimate:** ≈ 0.15–0.3 s/step/core (from §1.2 timings), so ≈ 16 cores give ≈ 60–100 steps/s ≈ 0.25–0.35 M steps/h. One million transitions (≈ 115 full rollouts) take ≈ 3–4 h locally. Shorter 7–10 day episodes increase scenario diversity per hour.
5. **Training (PyTorch, offline; export to `.npz` for NumPy inference):**
   - Multi-step latent rollout loss over 12–36 steps on anchored temperatures and tank temperature.
   - Energy and carbon head losses.
   - A latent consistency term for the free latents (TD-MPC2-style).
   - A context-encoder loss through the dynamics loss.
   - Allocator-surrogate imitation loss.
   - Later, a TD value loss.
6. **Splits:** leave-building-out (train on generated, validate on the 6 shipped buildings, never train on them), plus leave-station-out (hold out one station per fold).

---

## 8. Validation protocol and acceptance criteria

| Level | Test | Pass criterion |
|---|---|---|
| Model | Open-loop multi-step temperature RMSE on held-out buildings/stations, at 1/6/12/36 steps | 36-step RMSE better than the baseline RLS model rolled forward; calibrated ensemble spread |
| Model | Energy/carbon head error per step and per day | Daily cost error < 5% |
| Planner | Same scenario, planner using the **true simulator** vs. learned model | Learned-model plan recovers ≥ 50% of the oracle gain |
| Closed loop | `src/run_eval.py` on all 30 training scenarios, full horizon, then `src/summarize.py` | Mean score < 0.00695 and no scenario more than 5% worse than baseline |
| Generalization | 30+ generated unseen buildings × held-out stations | Mean improvement holds; worst-case not worse than baseline |
| Runtime | Single-process timing on a Kaggle CPU session, largest building | Policy ≤ baseline + 25 ms/step (R5) and total ≤ 10× `bang_bang` (R4) |
| Compliance | grep code and artifacts for building/station names; internet-off Kaggle run | Clean |

Always report the comfort/energy/carbon split and the worst step, not only the score.

---

## 9. Compute and runtime budget

- **Offline:** local 16-core CPU for simulation; GPU optional (models are tiny; CPU training is feasible).
- **Online:** derive from measurements, not assumptions. Phase 0 must establish (i) single-process sim time per step for each training building on Kaggle CPU, (ii) `bang_bang` wall time for all 30 scenarios (finishing finding §1.3-1), and (iii) whether hidden-scenario rollouts count against the 9 h notebook limit (Q1).

---

## 10. Roadmap (proposal, gated)

| Phase | Goal | Main work (future) | Exit / go-no-go |
|---|---|---|---|
| **0. Headroom & measurement** (≈ 3–4 days) | Know whether multi-step planning is worth it | Finish `bang_bang` × 30; single-process timing on Kaggle; **simulator-oracle lookahead**: deep-copy the env, evaluate a handful of supervisory plans per 15-min block on the 3 outlier + 3 typical scenarios; verify tank billing; confirm reward constants and tariff access | Oracle gain ≥ 10% → continue. Otherwise pivot to baseline robustness/tuning |
| **1. Refactor baseline into hooks** (≈ 2–3 days) | Make the allocator callable with supervisory knobs | Wrap the final-MRO `_band` / `_water_target` / `_choose_ahu` behind an interface; regression test: zero offsets → bit-identical actions and identical 0.00695 | Bit-identical replay on 30 scenarios |
| **2. Architecture A (grey-box MPC)** (≈ 1 week) | Working planner with zero learned params | Two-node RC + tank latent, EKF/RLS; MPPI over knobs; allocator surrogate (hand-derived first); runtime benchmark | Improves ≥ 3 of 6 Phase-0 scenarios without regressions; meets the runtime target |
| **3. Data pipeline** (≈ 1 week, overlaps 2) | Diverse transitions | Generator buildings, perturbation behaviour policy, logger, dataset cards | ≥ 1 M transitions, ≥ 50 buildings, all stations |
| **4. Learned residual + context + surrogate** (≈ 2 weeks) | Architecture C | Train, export `.npz`, NumPy inference, ensemble | Model-level criteria in §8 |
| **5. Closed-loop C + terminal value** (≈ 1–2 weeks) | Beat baseline robustly | Integrate, tune horizon/samples, fallback gating | All closed-loop and generalization criteria in §8 |
| **6. (Contingency) Distillation D** | Meet runtime if C is too slow | DAgger from C/oracle to a graph policy | Within 90% of C's gain at ≤ 2 ms/step |
| **7. Packaging** | Kaggle submission | Offline notebook following `baseline/` cell 3 wiring; artifacts; reproducibility bundle | Internet-off Kaggle run passes; score logged |

Calendar note: final deadline 2027-01-15, rules and team acceptance 2027-01-03. Phases 0–5 need about 7–9 weeks, which is tight. Phase 0's gate is what keeps the schedule realistic.

---

## 11. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Runtime budget (R5) tighter than the planner needs | Fatal | Decimated replanning, watchdog fallback, distillation D |
| Small true headroom over a heavily tuned baseline | Wasted effort | Phase 0 oracle gate |
| Gains rely on tank-billing artifact; organizers patch it | Score collapse | Measure gains with and without tank-dispatch freedom; prefer comfort/HVAC-side gains |
| Model exploitation (planner finds model errors) | Comfort spikes | Ensemble penalty, trust region around the baseline plan, zero-offset always evaluated |
| Overfitting to the 6 shipped buildings / 5 stations | Leaderboard drop | Train only on generated buildings; leave-station-out; `minneapolis` hold-out |
| Tariff use deemed non-compliant | Rework | Learned cost head as the alternative (Option 2, §6.4) |
| Baseline MRO hard to wrap | Delay | Phase 1 bit-identical regression test before any MPC work |
| Generated buildings larger or slower than shipped ones | Data cost | Cap V during data generation; use shorter episodes |

---

## 12. Open questions (need answers before or during Phase 0)

> Status after Phase 0: see [`phase0_findings.md`](phase0_findings.md) §1 and §7.

- **Q1.** Do the 30 hidden-scenario rollouts run inside the 9 h notebook budget, sequentially or in parallel? This determines whether R5 or R4 binds. Check the Kaggle submission template and forum.
- **Q2.** Is the 10× `bang_bang` limit measured per building on the evaluation machine? Is the time spent in `__init__` counted?
- **Q3.** May the policy embed the public tariff and carbon schedules from the starter kit (simulator-global constants), or must costs be learned?
- **Q4.** Are the reward normalizers (`max_*_rate`, `productivity_deviation_ref`, productivity per person-hour) identical across buildings, or building-specific?
- **Q5.** Is the energy-balanced tank accounting final for evaluation (§1.3-5)?
- **Q6.** Is deep-copying the environment for Phase 0 oracle lookahead fast enough (≈ 1 s per copy would make the oracle study take days)? Fallback: oracle on 1–3 day windows only.

---

## 13. Proposed future file layout (not created now)

```text
src/
  latent_mpc/
    graph.py          # metadata -> entity graph & static features
    model.py          # f_phys + residual (NumPy inference)
    cost.py           # official-reward-shaped cost model
    planner.py        # MPPI/CEM over supervisory actions
    policy.py         # TransferPolicy wrapper: filter -> plan -> allocator -> fallback
    allocator_hooks.py# wraps baseline final-MRO allocator with knobs
  data/collect.py     # perturbation rollouts logger (reuses run_eval loop)
  train/              # PyTorch training, export to .npz
artifacts/latent_mpc_v*.npz
tests/test_allocator_regression.py  # zero-offset == baseline, bit-identical
```

Reused as-is: `src/run_eval.py` (evaluation), `src/summarize.py` (scoring), `src/baseline_policy.py` (allocator, observer, fallback), `starter-kit` generators.

---

## 14. Phase 1 (executed 2026-11-18): data-collection infrastructure — actual changes

**Scope delivered.** This covers the logger and perturbation policy from roadmap Phase 3, and the zero-offset regression hook from roadmap Phase 1. It runs on the 6 shipped buildings only; building generation (§7.1) is not done yet. No controller, CEM or MPPI was written.

### 14.1 Files

| File | Role |
|---|---|
| `src/collect_data.py` | Parallel collector. Builds the env exactly as `run_eval.py` does and imports its constants. Writes one `.npz` per episode, a `manifest.jsonl` and per-scenario `meta.json` |
| `src/exploration.py` | `ExplorationBaselinePolicy` (baseline subclass whose only override is `_band`, for unoccupied setback offsets) and `Explorer` (seeded piecewise-constant knobs: AHU schedule, SAT, static pressure, HWS setpoint, setback, recovery lead), with clipping and a causal comfort guard. Profiles: `baseline`, `null`, `mild`, `moderate` |
| `src/building_meta.py` | Scenario metadata, per-field observation normalization (reproduced from the env), action layout, `ObservationCodec` / `ActionCodec` |
| `src/reward_model.py` | Pure-NumPy exact reward. K comes from metadata; constants are isolated in `RewardConstants`; annual-mean gas price fallback; uniform / area occupancy weights |
| `src/dataset.py` | Manifest / episode loading, building-level splits (default val = `building_train_7`, plus leave-one-building-out folds), `TransitionWindows` with a PyTorch view and scenario-homogeneous batches, `.npy` unpack for memory-mapping |
| `scripts/validate_phase1.py` | The 8 Phase 1 checks, written to `results/phase1/validation.json` |
| `scripts/reward_reconstruction.py` | Reward-variant and occupancy study, written to `results/phase1/reward_reconstruction.json` |
| `scripts/summarize_dataset.py` | Dataset summary, written to `results/phase1/sample_summary.json` |
| `docs/phase1_dataset.md` | Schema |
| `requirements.txt` | Adds `torch>=2.4` (training only; CPU wheel installed in `.venv`) |

`src/run_eval.py`, `src/baseline_policy.py`, `src/summarize.py` and `starter-kit/` are **unchanged**. `data/` is git-ignored.

### 14.2 Verified results (b8 = `building_train_8/atlanta_1`, K=3, V=30)

- **Baseline unchanged.**
  - Collecting with the `baseline` profile for the full 8,640 steps reproduces the stored `results/baseline_full` rollout bit-for-bit: mean reward −0.004031689621114378, all per-step rewards equal.
  - The `null` profile (subclass plus explorer with zero probabilities) is identical to `baseline`.
- **Replay.** Stepping a fresh env with the stored actions reproduces every observation and reward exactly. This confirms both serialization and simulator determinism.
- **Determinism.** The same seed gives identical arrays, except timing. A different seed changes the knob schedule.
- **Alignment.**
  - The action echoes in `o_{t+1}` equal `a_t` (69 fields).
  - The reward interval starts at `sim_ts[t+1]` = the hour in `o_{t+1}`.
  - **New:** the zone temperatures, setpoints and occupancy behind `r_t` appear in `o_{t+2}` exactly. Sensors are one step stale; time features are not. Using `o_{t+1}` gives up to 8.5 K error at transitions. The dynamics and cost alignment must follow this (see `phase1_dataset.md` §3.1).
- **Reward reconstruction (60 sample episodes, all 30 scenarios).**

  | Inputs | Max per-step error | Episode-mean error |
  |---|---|---|
  | privileged | 1.5e-8 | — |
  | observable comfort inputs (o_{t+2}) + true per-zone occupancy | 1.5e-8 | — |
  | annual-mean gas price | — | 1.4% mean, 2.9% max |
  | fully causal (also uniform occupancy, no holidays) | — | 1.7% mean, 5.0% max |

  The tariff tables and the K-derived normalizers are asserted equal to the starter kit.
- **Occupancy weighting gap (Phase 0 recommendation 5).**
  - Uniform and area-weighted splits are statistically tied: uniform has the lower occupied-step comfort MAE in 34/60 episodes and the lower bias in 27/60, and the sign of the bias varies.
  - Median bias is about 12% of the comfort term and 0.4% of the score; max per-step error is 0.16 (rare large excursions).
  - Cause: the generator gives every zone the same head-count (`zone_assignment`) with random arrivals, so the expected split is uniform and the residual is sampling noise.
  - **Decision:** use the uniform split (no geometry dependence). Do not invest in occupancy modelling beyond a comfort reserve.
- **Exploration safety** (`moderate`, 1-week sample on all 30 scenarios):
  - Every value stays within its native range.
  - No forced-OFF happens during comfort or pre-start hours.
  - Outside the knob channels, actions differ from the baseline only when a policy-internal knob is active.
  - Mean score is 0.00728 vs 0.00617 for the baseline (+18%). Mean comfort is −3.1e-4 vs −1.8e-4. The degradation is bounded and the profile explores 60–96% of steps.
- **Cost.**
  - Collector overhead is about 5 ms/step on b8 (191 s vs ≈ 148 s for a plain baseline rollout).
  - The counterfactual baseline action costs about 1.5 ms/step extra (policy clone).
  - Storage is ≈ 0.47 MB per 1k transitions for V=30 and ≈ 1.2 MB for V=85; the 60-episode sample (121k transitions) is 94 MB of episode archives.
- **PyTorch.** `TransitionWindows` + `DataLoader` deliver [B, H+1, D] float32 windows from memory-mapped arrays.

### 14.3 Decisions updated from Phase 0

- The cost model uses the exact formula with K from metadata and the public tables (subject to the rules question Q3). Its time inputs come from `o_{t+1}` and its comfort state from the model's prediction of the `o_{t+2}` sensor block.
- Gas price uses the annual mean (1.4% mean error on the sample). Month inference is low priority.
- Tariff arbitrage remains unimplemented. The 1-week sample spans weekday/weekend TOU, so its contribution can be tested on logged data before any effort goes into it.
- Tank: `tank_state` logs the true tank temperature and burner heat each step, and the boiler parameters are in `meta.privileged.boiler`, so the energy-balance / ramp / loss prior can be fitted and checked directly in Phase 2.
- AHU electricity is reported **only in aggregate** across AHUs, so per-AHU energy heads need an aggregate loss.

### 14.4 Known limitations / open items

1. Forced-OFF rarely triggers: the baseline already turns AHUs off when the building is unoccupied, and OFF is barred during occupancy for safety. If on/off scheduling during occupancy matters to the planner, it needs a separately gated profile.
2. Action-level knobs perturb the baseline's output after the fact, so its internal observer records commanded rather than executed SAT and pressure in those segments. This is intended for exploration, but it means `b_t` is "the baseline from its current perturbed state", not an independent rollout.
3. Only 6 shipped buildings exist. The building-level validation split therefore has 1 building, so leave-one-building-out folds are advisable until generated buildings (§7.1) are added. The collector already accepts any scenario directory layout via `SCENARIOS`, but it does not yet take generated-scenario paths on its CLI.
4. The default 1-week sample episodes start at the scenario start date. Full 30-day collection (`--steps 8640`) for the training set is about 60 episodes × 1.5–7 min single-process. Not run yet.
5. The rules question (Q3, public tariff tables) is still open with the organizers.
