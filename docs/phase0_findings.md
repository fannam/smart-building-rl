# Phase 0 Findings — Evaluation Flow, Reward Alignment, Tank Accounting, Simulator Benchmarks

> **Date:** 2026-11-18 · **Starter kit:** `1fb75606` (2026-11-16) · **Machine:** Intel i9-14900HX, 32 threads, 31 GB RAM
> **Scope:** verification and small benchmarks only. No controller code was written; simulator and policy sources are unmodified.
> **Companion:** [`latent_mpc_implementation_plan.md`](latent_mpc_implementation_plan.md) (its §2 R5, §6.6 and §12 are superseded by this document where they differ).

**Reproduce:**
```bash
# single-process, sequential (no contention); writes results/phase0/full_*.json
.venv/bin/python scratch/phase0/bench_env.py --scenario building_train_8/atlanta_1 --policy bang_bang --steps 8640 --out results/phase0/full_building_train_8__atlanta_1__bang_bang.json
.venv/bin/python scratch/phase0/bench_env.py --scenario building_train_8/atlanta_1 --policy baseline  --steps 8640 --copies 100 --out results/phase0/full_building_train_8__atlanta_1__baseline.json
# (same for building_train_4/atlanta_5 and building_train_2/atlanta_3)
cd starter-kit && ../.venv/bin/python ../scratch/phase0/tariff_sensitivity.py
```

---

## 1. Status summary

| Q | Topic | Status | One-line answer |
|---|---|---|---|
| Q1 | Where the 30 hidden rollouts execute | **Unresolved** (no public source) | The gateway owns the simulator. Whether its rollouts count against the 9 h notebook limit is not documented publicly. Two budget modes are kept configurable (§2.1) |
| Q2 | How the 10× `bang_bang` limit is measured | **Partially resolved** | The paper says "per-rollout wall-clock". No public enforcement code exists, and the boundary (`__init__`, IPC) is unknown. Measured locally: the per-step policy allowance is **≈ 140 ms (small building) to ≈ 410 ms (large)** |
| Q3 | Public tariff / carbon schedules | **Resolved (technically); rules permission unresolved** | They are public module constants, identical across all 30 configs. Hour and day type are observable; the **month is not** (needed for the gas price), but the annual mean gives ≤ 4.6% error |
| Q4 | Reward normalizers | **Resolved** | Energy and gas normalizers are 80 kW·K and 205 kW·K, where K (the number of AHUs) is known at runtime. Every other constant is global. The comfort term is independent of the $/person-hour constant |
| Q5 | Tank accounting | **Current implementation documented; finality unresolved** | The energy-balanced tank is active in all 30 scenarios. Gas is billed as burner heat at 100% efficiency. No free storage |
| Q6 | `deepcopy(env)` for oracle lookahead | **Resolved** | A plain deepcopy or pickle **fails** (SciPy `SuperLU`). A deepcopy that shares the factorization works: **0.40–0.99 s and ≈ 75–185 MB per copy**. Only near-deterministic. A full-horizon oracle is too expensive; use 1–3 day windows |

---

## 2. Q1 + Q2 — Execution flow and runtime budget

### 2.1 Q1: where the hidden rollouts execute

**Sources checked:**
- Technical paper (arXiv 2610.18511, §"Evaluation harness" and "Anti-cheating"): *"Scoring on the leaderboard is performed instead by an inference-server gateway, which owns the simulator, iterates the held-out scenarios and queries the submitted policy once per step"* and *"Per-rollout wall-clock is bounded at 10× the bang-bang baseline runtime"*. The paper does not say whether this happens inside the notebook session or what the notebook time limit covers.
- Baseline notebook cell 3 (`baseline/example-smart-building.ipynb`): the submission calls `SmartBuildingsInferenceServer(SubmissionPolicy).run(competition_data_folder=DATA_ROOT)`. The `kaggle_evaluation` package is part of the Kaggle competition data and **is not in the public starter kit**. This machine has no Kaggle credentials, so it could not be inspected.
- The Kaggle overview, data and rules pages render client-side, so their text could not be fetched. The challenge site (`smart-buildings-2026.github.io/competition/`, `competition.html`) has no runtime details. A web search found no discussion thread on this.

**Status: unresolved.** Next action: read `kaggle_evaluation/smart_buildings_gateway.py` (or similar) in the attached competition data inside a Kaggle notebook, and the Kaggle "Code Requirements" panel.

**Development assumption (configurable `budget_mode`):**
- `per_rollout` (**default**): only the paper's 10× `bang_bang` per-rollout bound applies (§2.3 numbers).
- `notebook_sequential` (**conservative fallback, not assumed**): if inspection shows all 30 rollouts run sequentially within 9 h, then 32,400 s / 30 ≈ 1,080 s per rollout. The Kaggle gateway smoke test took ≈ 805 s for the baseline on `building_train_1`, which leaves ≈ 30 ms/step for the policy. The controller must support a "lean" setting (decimated planning or the distilled policy) that fits this.

### 2.2 Q2: timing boundary

- The paper gives the 10× limit as **per rollout**.
- The local `starter-kit/smart_buildings/evaluate.py::run_policy` (lines 536–560) only times `tracker.step` (simulator) and enforces no limit. `score_leaderboard.py` does no timing. **No public code shows how the limit is enforced**, whether `__init__` is included, or how the `bang_bang` reference is obtained (same machine? per building?).
- **Working assumption (as instructed):** wall-clock of the whole rollout counts, including policy `__init__`, artifact/model loading, every `get_action`, simulator steps and gateway overhead. No credit for parallelism.

### 2.3 Measurements (single process, sequential, full 8,640 steps)

| Scenario | K/V | Env load | Policy `__init__` | `bang_bang` rollout | Baseline rollout | Sim step mean / p95 / max |
|---|---|---:|---:|---:|---:|---:|
| `building_train_8/atlanta_1` | 3/30 | 1.5 s | 0.13 s | **137 s** | 148 s (1.08×) | 15.8 / 16.9 / 367 ms |
| `building_train_4/atlanta_5` | 6/54 | 3.3–3.8 s | 0.33 s | **304 s** | 317 s (1.04×) | 35.0 / 37.0 / 924 ms |
| `building_train_2/atlanta_3` | 9/85 | 3.3 s | 0.38 s | **396 s** | 413 s (1.04×) | 45.7 / 47.5 / 1,429 ms |

Baseline (`WideHotStorageReservePolicy`) `get_action` latency:

| Building | mean | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| b8 (V=30) | 1.26 ms | 1.23 | 1.49 | 1.65 | 3.2 |
| b4 (V=54) | 1.67 ms | 1.67 | 1.83 | 2.19 | 5.7 |
| b2 (V=85) | 2.23 ms | 2.20 | 2.47 | 2.82 | 6.5 |

`bang_bang` `get_action`: 0.06–0.10 ms mean. Simulator max step (300–1,400 ms) is the first step, where the solver is factorized.

**Corrections to earlier numbers:**
- The earlier `results/baseline_full` timings (sim 630–2,520 s per rollout) were inflated **≈ 4–6×** by 16-worker contention. Single-process simulation costs **16–46 ms/step**.
- The baseline is only **1.04–1.08×** `bang_bang` per rollout, not close to the limit.
- Determinism holds: single-process scores (0.00403, 0.00632, 0.00279) equal the parallel-run scores exactly.

**Derived per-step policy allowance** = (10 · T_bb − T_sim − T_init − T_load) / 8640:

| Building | Allowance | Conservative design target (≤ 50% of allowance) |
|---|---:|---:|
| b8 (smallest, fastest sim, so binding) | **≈ 142 ms/step** | **≈ 70 ms/step** |
| b4 | ≈ 316 ms/step | ≈ 150 ms/step |
| b2 | ≈ 412 ms/step | ≈ 200 ms/step |

Notes:
- The smallest building binds, because the allowance scales with simulator cost while planner cost grows with V. Hidden buildings come from the same generator (K = 3–9), so a K = 3, V ≈ 28 building is the design point.
- Kaggle CPU is slower than this machine. The Kaggle smoke test ran b1 (4/39) in 805 s, against an estimated ≈ 180–200 s here. If the reference and the policy are timed on the same machine, the *ratio* carries over, but fixed costs (`__init__`, model load) do not scale the same way. **Re-measure on Kaggle before Phase 2 sign-off.**
- **Development rule:** total policy time per rollout ≤ 4 × T_bb(local) on the smallest building, plus watchdog fallback to the baseline. In `notebook_sequential` mode the target is ≤ 30 ms/step.

---

## 3. Q3 — Tariff and carbon schedules

**Source:** `starter-kit/smart_buildings/smart_control/reward/electricity_energy_cost.py`, `natural_gas_energy_cost.py`, `utils/conversion_utils.py::is_work_day`, `smart_control/utils/constants.py`.

| Item | Finding |
|---|---|
| Electricity price | `WEEKDAY_PRICE_BY_HOUR`: 16¢ for hours 0–5 and 19–23, 18¢ for 6–11, 20¢ for 12–18. `WEEKEND_PRICE_BY_HOUR`: flat 16¢. Indexed by `start_time.hour` and `is_work_day` (weekday and not a US holiday) |
| Electricity carbon | `CARBON_EMISSION_BY_HOUR`: 24 values, 87.8–104.7, indexed by hour (no day-type dependence) |
| Gas price | `GAS_PRICE_BY_MONTH_SOURCE`: 12 monthly values, $6.69–9.02 per kft³, indexed by `start_time.month` |
| Gas carbon | Constant (`constants.GAS_CO2`) |
| Building- or scenario-specific? | **No.** No config overrides `ElectricityEnergyCost` / `NaturalGasEnergyCost` (checked all 30 `config.gin`: they use the defaults via `@ElectricityEnergyCost()`). All 30 configs use `time_zone = "US/Pacific"` |
| Identical in hidden evaluation? | Very likely (same generator and code), but **not confirmed**. Ask the organizers |
| Observable alignment | The observation carries `hod_sin/cos_000` and `dow_sin/cos_000`. Empirically, the hour decoded from the observation returned by `step()` equals the simulator timestamp after the step, and the `start_timestamp` reported by `get_reward_info_and_response()` (b2 and b8 logs, including the midnight and 06:00 transitions). So hour and day-of-week are known exactly |
| Not observable | **Month and date.** Start dates vary by scenario (2023-01-01 … 2023-11-26) and are not in the metadata. Consequences: (i) gas price unknown, (ii) holidays unknown (`metadata["holidays"]` lists them, but without the date the policy can't match them) |

**Sensitivity** (`scratch/phase0/tariff_sensitivity.py`; per-watt normalized penalty including the normalization denominators):
- Electricity costs about **2×** the reward per watt of gas (gas/electricity ratio 0.43–0.55).
- Time-of-use swing of the electricity penalty is only ≈ 13–15% (e.g. January: 2.07 at 03:00 vs 2.25 at 15:00, ×1e-6 per W per AHU).
- Using the **annual-mean gas price** gives at most **4.6%** error on the electricity penalty and 3.3% on the gas penalty.

**Recommendation:**
- Embed the public schedules (hour and day-type tables, plus annual-mean gas price) in the planner's cost, **subject to a rules check**. They are generic simulator constants, not building or sensor signatures.
- **Causal fallbacks:**
  - Use the annual-mean gas price. Optionally estimate the season from the outside-air temperature history; low value, since the error is ≤ 5%.
  - Infer holidays causally: a weekday on which `comfort_mode_now` stays 0 through the morning start hour is treated as non-work (16¢ flat).
- **Implication for MPC:** price-driven load shifting is a **small** lever (≈ 15% swing). The main multi-step value has to come from thermal and tank dynamics: recovery timing, setback depth and on/off scheduling. This lowers the expected gain from TOU listed in plan §3.

---

## 4. Q4 — Reward normalizers

**Sources:** `smart_control/reward/setpoint_energy_carbon_regret.py::compute_reward` (lines 185–346), `reward/base_setpoint_energy_carbon_reward.py::_get_zone_productivity_reward`, `reward/energy_scale.py`, `check_scales.py`, all 30 `training_scenarios/*/*/config.gin`.

| Constant | Value(s) across the 30 configs | Global or building-specific | Available at runtime? |
|---|---|---|---|
| `max_electricity_rate` | 240k / 320k / 480k / 640k / 720k W | **= 80,000 · K** (`C_ELECTRICITY_PER_AHU`) | Yes: K = `len(meta["temp_indices"])` |
| `max_natural_gas_rate` | 615k / 820k / 1,230k / 1,640k / 1,845k W | **= 205,000 · K** (`C_NATURAL_GAS_PER_AHU`) | Yes (same K) |
| `productivity_deviation_ref` | 5.0 (all) | Global | Constant |
| `max_productivity_personhour_usd` / `min_…` | 300 / 0 (all) | Global; **cancels out** of the normalized comfort term | Not needed |
| Weights (p, c, e) | 0.6 / 0.2 / 0.2 (all) | Global | Constant |

`check_scales.py` asserts `max_* == c · K` against the constructed environment for every shipped scenario (and can be pointed at generated ones via `SCEN_DIR`). It wasn't re-run: the config grep shows the same equality for all 30.

**Exact per-step reward that the MPC objective must reproduce** (with ŵ = weights / Σ weights = weights here):
```
comfort  = −Σ_z occ_z · s(dev_z) / Σ_z occ_z      (0 if Σ occ = 0)
           dev_z = max(0, T_z − T_cool_z, T_heat_z − T_z)
           s(d) = d/5 for d ≤ 5,  1 + ((d−5)/5)² for d > 5
energy   = (p_e(h,wd)·E + p_g(month)·G) / (K·(p_e·80 000 + p_g·205 000))
carbon   = (k_e(h)·E + k_g·G)          / (K·(k_e·80 000 + k_g·205 000))
r        = 0.6·comfort − 0.2·energy − 0.2·carbon
```
- E = blower + cooling electricity (after COP) + pump.
- G = burner heat.
- `T_heat/T_cool` are the zone setpoints in the reward info (observable as `*_zone_air_heating/cooling_temperature_setpoint_sensor`).
- **Per-zone occupancy is not observable**, only `num_occupants` for the building. Comfort weighting across zones has to assume a split (uniform or area-weighted). That's a modelling gap to validate.

**Design rule:** read K from metadata and use the formulas above. **No hard-coded per-building normalizers.** Keep `C_ELECTRICITY_PER_AHU`, `C_NATURAL_GAS_PER_AHU` and `productivity_deviation_ref` in one config block so organizer changes are a one-line update.

---

## 5. Q5 — Hot-water tank accounting (current implementation)

**Sources:** `smart_control/simulator/boiler.py` (`__init__` lines 39–101, `apply_heat_draw` 138–180, `compute_thermal_energy_rate` 253–312), `simulator/hot_water_system.py::apply_heat_draw` (300–330), `simulator/simulator_flexible_floor_plan.py` (`apply_heat_draw` call ≈ line 506; `_get_hws_reward_infos` ≈ 806–840).

**Active configuration (verified at runtime in all three benchmarked scenarios; all 30 configs share the same boiler block):**
- `Boiler.energy_balance = True`. Defaults: `tank_radius = 0.5 m` (so `_has_tank`), `water_capacity = 1.5 m³`, `heating_rate = 0.5 K/min`, `cooling_rate = 0.1` (unused under energy balance).

**Accounting implemented today:**
1. Each step, after the return-water temperature is computed from the coil heat balance, `HotWaterSystem.apply_heat_draw(outside_temp, dt)` computes `coil_heat_w = c_w·ρ_w·flow·(T_supply − T_return)` and calls `Boiler.apply_heat_draw`.
2. `Boiler.apply_heat_draw` implements `C·dT/dt = Q_burner − Q_coils − Q_loss`, where `C = c_w·ρ_w·V_tank` and `Q_loss` comes from the insulation and convection model. The burner fires `Q_burner = clip(C·(T_set − T)/dt + Q_coils + Q_loss, 0, Q_coils + Q_loss + C·heating_rate/60)`. The tank never cools faster than draw plus loss.
3. **Billed gas:** `compute_thermal_energy_rate` returns `_burner_heat_rate` (W). `_get_hws_reward_infos` reports it as `natural_gas_heating_energy_rate`. There's no boiler efficiency factor; it is billed at 100% of burner heat.
4. Pump electricity is `compute_pump_power()`, separate from gas.

**Implications:**
- Storage is physically consistent. Every joule delivered is burned now or was burned earlier, and a hotter tank pays a higher standing loss. So "store heat" only pays when (a) the burner's ramp limit (0.5 K/min · C ≈ 52 kW) would otherwise starve the morning recovery, or (b) the *normalized* gas penalty is lower at charge time than at use time. (b) happens only through the electricity terms in the denominator (§3), which is a small effect. The gas price itself is flat within a month.
- The baseline's 340–350 K storage band therefore mainly buys **heating capacity headroom**, not cheaper energy. A multi-step planner should model the ramp limit and loss explicitly.
- **Isolation for future code:** keep tank parameters (`V_tank`, `heating_rate`, loss model, efficiency = 1.0) in one config object for the dynamics prior and the cost model, so an organizer change is a parameter update.

**Status:** the current implementation is documented. Whether it is final for evaluation is **unresolved**; ask the organizers.

---

## 6. Q6 — Environment copy for oracle lookahead

**Findings:**
- `pickle.dumps(env)` and `copy.deepcopy(env)` both **fail**: `TypeError: cannot pickle 'SuperLU' object`. The source is `FiniteDifferenceSolver._factorization` (`simulator/finite_difference_solver.py` ≈ lines 831, 994–1003), a lazily built `scipy.sparse.linalg.splu` cache. Consequence: **process-based parallel copying (multiprocessing pickling) does not work as-is.**
- **Workaround (benchmark-only, `scratch/phase0/bench_env.py::env_copy`):** pass `copy.deepcopy(env, memo)` with `memo[id(factorization)] = factorization`. Copies then *share* the read-only factorization. No simulator code is modified.

| Building | deepcopy mean / p95 / max | RSS growth per copy | Copy-vs-original divergence (24 steps, same actions) |
|---|---|---:|---:|
| b8 (3/30) | 397 / 458 / 469 ms | ≈ 73 MB | 2.7e-4 |
| b4 (6/54) | 921 / 1,008 / 1,025 ms | ≈ 172 MB | 3.3e-5 |
| b2 (9/85) | 990 / 1,079 / 1,103 ms | ≈ 183 MB | **1.3e-2** |

- Divergence is the max absolute difference over reward and normalized observation. Copies are **not bit-exact**. The likely cause is shared or mutated caches (e.g. `_sensed_response`) or RNG state. It is small, but the b2 value needs investigating before oracle results are trusted at the 1e-4 reward level. Note that full rollouts from a fresh environment *are* exactly reproducible (§2.3).
- 100 copies per run were timed, and 5 copies were kept alive for the memory measurement.

**Oracle cost estimate** (b2, the worst case; candidate = copy + H-step rollout at 46 ms/step):

| Study | Cost per scenario (1 core) |
|---|---:|
| Full 30 days, 8 candidates per 15-min block, H = 36 | 2,880 blocks × 8 × (1.0 + 1.66 s) ≈ **17 h**, so not worthwhile |
| **3-day window**, 8 candidates per block, H = 36 | 288 × 8 × 2.66 s ≈ **1.7 h** |
| 1-day window, 16 candidates, H = 24 | 96 × 16 × 2.1 s ≈ **0.9 h** |

- Memory: about 185 MB per live copy, so keep ≤ 2–3 alive per worker. Running scenarios in parallel across processes is fine (each process builds its own env from `config.gin`). Only copying *within* a process needs the memo workaround.
- A cheaper alternative to copying is rebuilding the env and replaying the action log to the branch point (deterministic). That is O(t), so it is only good for early branch points.

**Decision:** a limited oracle study is affordable. Use the 3 outlier scenarios plus 3 typical ones, with 3-day windows centred on cold or hot events, 8 candidates per block from the supervisory knob set, and the baseline as the reference. That's about 10 core-hours, or about 1 h of wall time on 16 workers. Run it in parallel with Phase 1 so it doesn't delay the MPC MVP.

---

## 7. Questions for the organizers

1. **Execution flow:** do the 30 hidden rollouts run inside the submitted notebook session, and do they count against the 9 h limit? Sequentially or in parallel?
2. **10× rule:** is the `bang_bang` reference measured per building, on the same hardware, in the same session? Are policy `__init__`, artifact loading and gateway IPC inside the timed window? What happens on violation: the rollout fails, or a fallback is used?
3. **Tariffs and carbon:** are the evaluation scenarios built with the default `ElectricityEnergyCost` / `NaturalGasEnergyCost` schedules and `US/Pacific`? May a policy embed these public tables?
4. **Normalizers:** are `C_ELECTRICITY_PER_AHU = 80,000`, `C_NATURAL_GAS_PER_AHU = 205,000` and `productivity_deviation_ref = 5` used unchanged for evaluation?
5. **Tank:** is the energy-balanced boiler (`energy_balance = True`, 1.5 m³, 0.5 K/min, 100% efficiency) final for evaluation?
6. **Calendar:** will evaluation scenarios also start at 06:00 on varied 2023 dates? Is the date or month intentionally hidden from the policy?

---

## 8. Updated recommendations for the next phase

1. **Runtime budget is looser than assumed in the plan, unless Q1 says otherwise.** In `per_rollout` mode the design target is about 70 ms/step mean on the smallest building, against the plan's 25 ms. This permits an MPPI planner with N ≈ 64–128 samples and H = 12–36 in NumPy. Keep the `notebook_sequential` lean mode (≤ 30 ms/step) as a configuration, and keep distillation (Architecture D) as the contingency.
2. **Cost model:** implement the exact formula in §4 with K from metadata and the public tariff tables, plus the annual-mean gas price and causal holiday detection. The learned components should only predict *physical* quantities: zone temperatures, E (fan, chiller, pump) and G (burner heat).
3. **Re-rank the multi-step levers:** time-of-use shifting is small (≈ 15% swing). Tank value comes from the ramp limit and standing loss. The main opportunities are **heating recovery timing and capacity** (cold outliers: fairbanks_1, fairbanks_8), setback depth, and AHU on/off scheduling. The oracle study should test exactly these knobs.
4. **Tank model in the dynamics prior:** a first-order tank with the burner ramp limit (≈ 52 kW cap above draw plus loss) and the insulation loss, with parameters isolated in config.
5. **Comfort weighting gap:** only building-level occupancy is observable. Validate the uniform vs. area-weighted zone split against the true reward on training rollouts in Phase 1.
6. **Phase 1 (refactor baseline into hooks)** can start now; it doesn't depend on any unresolved question. Run the Phase 0 oracle window study alongside it, first fixing or characterizing the b2 copy divergence.
7. **Measure on Kaggle** (one notebook session): `bang_bang` and baseline single-rollout time for one small building, plus an inspection of the `kaggle_evaluation` gateway source. This resolves Q1/Q2 or narrows them sharply.
