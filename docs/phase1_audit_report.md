# Phase 1 Audit — Dataset Validation, Reward Correctness, and Readiness for Phase 2

> **Date:** 2026-11-18 · **Scope:** audit only. No Phase 2 code, no model training, no changes to production collection code, the baseline or the starter kit.
> **Evidence tags:** **[S]** verified from source code · **[E]** verified experimentally (command in §3) · **[I]** inferred from available evidence · **[N]** not verified.
> **Audit artefacts:** scripts in `scratch/phase1_audit/`, outputs in `scratch/phase1_audit/out/`. Existing reports, results and datasets were not modified. The validation suite was re-run with its output directories redirected into scratch.

---

## 1. Executive summary

**Decision: READY WITH CONDITIONS.** No critical correctness or data-integrity defect was found.

What is verified:
- **Integrity [E].** 66 episodes (~132k transitions, every scenario) are free of NaN/Inf, shape errors, timestamp gaps and misalignment.
- **Exact reward reconstruction [E].** Rebuilding the reward from the logged privileged quantities matches the simulator to ≤ 1.5e-8 per step on every building and weather.
- **Baseline equivalence [E].** `baseline` and `null` are bit-identical on three buildings (b8 at 1,152 steps; b2 and b7 at 576 steps). The full-horizon b8 baseline reproduces the stored `results/baseline_full` rollout exactly.
- **Determinism and replay [E].** Same seed gives identical arrays, and stored actions replay bit-exactly.

The conditions concern:
- how Phase 2 consumes the data (temporal alignment and the input/target boundary);
- how generalization is evaluated;
- known coverage weaknesses of the exploration.

Key findings, most important first:

1. **Temporal alignment is correct in the data but easy to get wrong in training.**
   - Zone temperatures, zone setpoints and `num_occupants` behind `reward[t]` appear exactly in `obs[t+2]` [E].
   - **New:** the HWS supply-water sensor is *not* stale. `obs[t+1]` already equals the true tank temperature after step t [E]. The schema's blanket statement "sensor values are one step stale" is therefore wrong for the tank, and unverified for other sensor groups.
   - `TransitionWindows(horizon=H)` returns H+1 observations, so the sensor state behind its last reward lies outside the window [E]. This is an off-by-one trap for Phase 2.
2. **Exploration is much weaker than its configured probabilities suggest [E].**
   - Static pressure: the simulator floors it at 200 Pa while an AHU runs. The baseline always commands 200 Pa, so every downward pressure perturbation is a no-op: 20–23% of running moderate AHU-steps are commanded below the floor. The action logged is the commanded value, not the effective one.
   - AHU forced-OFF is effectively absent: 0.4% of AHU-steps (moderate) and 0.03% (mild).
   - Requested vs applied: SAT 50% → 23%; HWS 49% → 28% effective; recovery lead 53% → 1.6% of steps where it can matter.
   - VAV dampers and valves differ from the baseline on only 2–8% of steps.
   - Occupied comfort excursions above 1 K are rare: 0.05% of occupied zone-steps (baseline), 0.09% (moderate).
3. **The runtime (causal) reward estimate is biased [E].**
   - The episode-mean reward is +1.1% too high on average, 5.0% worst case (cost underestimated).
   - Most of that comes from the annual-mean gas price (+0.84% signed, up to 2.9% per episode).
   - The per-zone occupancy split adds per-step error concentrated around arrival/departure transitions: MAE 1.2e-3 near transitions vs 4.3e-5 elsewhere, max 0.16 per step.
   - Uniform and area-weighted splits are statistically indistinguishable.
4. **Generalization evidence is structurally limited [E/I].**
   - There are 6 buildings, 1 week per scenario, and each scenario is observed only from its start date.
   - The default validation building (`building_train_7`) never sees severe cold: its minimum outdoor temperature is 1.6 °C. Yet cold-climate heating is where most of the score is lost.
   - Single-building validation cannot demonstrate unseen-building generalization. Leave-one-building-out is required.
5. **Observer inconsistency is real but benign for labels [E/S].**
   - Action-level perturbations raise the baseline's internal model error by 37% (0.0155 → 0.0212), moderate as a whole by 52%. This shifts the *behaviour* distribution.
   - Labels stay correct, because executed actions and true outcomes are what is logged.
6. **No privileged leakage exists today, but the boundary isn't enforced [S].**
   - Nothing in the pipeline fits statistics or builds features.
   - However, `TransitionWindows` returns privileged arrays (`zone_info`, `tank_state`, `calendar`) in the same dict as `obs`.
   - `meta.json` carries `privileged.episode_start`, which reveals the month.

---

## 2. Files inspected

| File | Status | Notes |
|---|---|---|
| `docs/phase0_findings.md`, `docs/latent_mpc_implementation_plan.md` (incl. §14), `docs/phase1_dataset.md` | read | — |
| `src/collect_data.py`, `src/exploration.py`, `src/building_meta.py`, `src/reward_model.py`, `src/dataset.py` | read in full | Phase 1 production code |
| `src/run_eval.py` | read in full | unchanged vs HEAD [E: `git diff`] |
| `src/baseline_policy.py` | relevant classes read (`TransferPolicy`, `_band` chain, `StorageDispatchPolicy`, `HotStorageReservePolicy`, `WideHotStorageReservePolicy`) | unchanged vs HEAD [E] |
| `scripts/validate_phase1.py`, `scripts/reward_reconstruction.py`, `scripts/summarize_dataset.py` | read | — |
| `results/phase1/validation.json`, `reward_reconstruction.json`, `sample_summary.json`, `*.log` | read (not modified) | — |
| `data/phase1_sample/manifest.jsonl` + 60 episodes + 30 `meta.json` | scanned | — |
| `data/phase1_validation/run_a` (6 episodes), `full_baseline` (1 episode, 8,640 steps) | scanned | `run_a` was regenerated by the second Phase 1 validation run (same seeds, deterministic) |
| starter kit: `evaluate.py`, `environment.py` (step / observation), `hybrid_action_environment.py`, `setpoint_energy_carbon_regret.py`, `base_setpoint_energy_carbon_reward.py`, `electricity_energy_cost.py`, `natural_gas_energy_cost.py`, `constants.py`, `conversion_utils.py`, `observation_normalizer.py`, `randomized_arrival_departure_occupancy.py` | read (relevant parts) | not modified |

All files the audit brief lists exist. There is no separate "Phase 1 implementation plan" document; Phase 1 is §14 of `latent_mpc_implementation_plan.md`.

---

## 3. Experiments executed

All commands ran from the repository root, using `.venv/bin/python`.

| # | Command | Purpose | Result |
|---|---|---|---|
| E1 | `scratch/phase1_audit/run_validation.py` (runs `scripts/validate_phase1.py` unchanged; output and data redirected to `scratch/phase1_audit/out/validation_*`; full-baseline read-only from `data/phase1_validation/full_baseline`) | the 8 Phase 1 checks | **8/8 PASS**, numbers identical to the original run |
| E2 | `scratch/phase1_audit/integrity.py` | shapes, dtypes, NaN/Inf, ranges, timestamps, echoes, lags, done flags, tank alignment, on 66 episodes | no issues; findings in §4–5 |
| E3 | `scratch/phase1_audit/clone_observer.py` | forced counterfactual-clone path; baseline internal model error under null / action-only / internal-only / moderate (b8, 1,152 steps, seed 1) | §6, §7.4 |
| E4 | `src/collect_data.py --out scratch/phase1_audit/out/null_check --scenarios building_train_2/fairbanks_2 building_train_7/atlanta_8 --profiles null --steps 576` + array comparison with the sample `baseline` episodes | null ≡ baseline on the largest building and the val building | bit-identical, all arrays |
| E5 | `src/collect_data.py --out scratch/phase1_audit/out/mild_sample --scenarios <one per building> --profiles mild --steps 2016 --seed 0 --workers 6` | `mild` statistics (previously one episode only) | 6 episodes OK |
| E6 | `scratch/phase1_audit/exploration_audit.py` | replays the requested knob schedule from the seed; compares with applied knobs and effective native changes; action diversity | §7 |
| E7 | `scratch/phase1_audit/reward_audit.py` | per-step reward error distributions by variant, building, weather, component and transition proximity (60 sample episodes) | §8 |
| E8 | `scratch/phase1_audit/windows_split_audit.py` | `TransitionWindows` ordering and boundaries (2,000 random windows), scenario batching, split / LOBO integrity, PyTorch `DataLoader` | §10 |
| E9 | `scratch/phase1_audit/target_stats.py` | reward, temperature, energy and outdoor distributions by building, station and profile | §11 |
| E10 | inline checks (logged in this session): reward constants in all 30 `meta.json`; static-pressure echo investigation; HWS sensor lag vs `tank_state` | — | §4, §5 |

Nothing failed. **Not run:** full 30-day exploration rollouts (expensive, not needed); a Kaggle-side check (outside scope).

---

## 4. Data integrity findings

| Check | Result | Tag |
|---|---|---|
| `obs` shape `[T+1, D]`, D = `meta.obs_dim`; all per-step arrays have T rows, `tank_state` / `sim_timestamp` T+1 | all 66 episodes | [E] |
| Continuous action `[T, n_c]` float32 in [−1, 1] (max \|a\| = 1.0); discrete `[T, n_d]` int8 ∈ {0, 1}; baseline arrays likewise | all 66 | [E] |
| Action order = the simulator's `env.action_names` (continuous and discrete separately); discrete = AHU run bits in AHU order, then HWS run; index maps resolved by name, not position | — | [S] (`building_meta.build_scenario_metadata`), [E] (69 echo fields match) |
| `obs[t+1]` is the return of `env.step(action[t])` | `collect_data.py` assigns `obs_buf[t+1] = obs` immediately after `env.step(action)` | [S]; [E] exact replay of stored actions reproduces every observation and reward (b8 moderate and baseline, 1,152 steps) |
| `reward[t] == RewardResponse.agent_reward_value` read after the same step | all 66 | [E] |
| NaN / Inf in any float array | none | [E] |
| `perturbed` flag == (a ≠ b) | consistent in all 66 | [E] |
| Reward config identical in all 30 scenarios (80,000 / 205,000 W per AHU, deviation ref 5 K, weights 0.6/0.2/0.2, $300 / $0 per person-hour) | yes | [E] |
| Observation normalization constants identical across the weathers of a building | 1 distinct set per building | [E] |
| **Termination flags** | `done` is all-False in every episode, **including the 8,640-step full horizon**. The env checks `_step_count >= num_timesteps` before incrementing, so `terminated` would only fire on step 8,641. The collector caps T at `num_timesteps_in_episode`, so this is harmless and identical to `run_eval.py`. Episode boundaries are defined by the file, not by `done` | [E]/[S] |
| Truncation | never set by the env | [E] |
| Manifest | append-only, deduplicated by `episode_id` on load | [S] |

---

## 5. Temporal alignment findings

### 5.1 Verified relationships

All on 66 episodes unless stated.

| Relationship | Max error | Tag |
|---|---|---|
| `sim_timestamp[t+1] − sim_timestamp[t] = 300 s` | exact | [E] |
| `reward_timestamp[t] = [sim_ts[t+1], sim_ts[t+1] + 300]` | exact | [E] |
| hour decoded from `obs[t+1]` time features = reward-start hour (tariff and carbon index) | 1.4e-7 h | [E] |
| SAT, HWS supply-temperature setpoint, HWS differential pressure, damper and valve echoes in `obs[t+1]` = native(`a_t`) | 1.8e-5 (native units) | [E] |
| run-bit echoes in `obs[t+1]` = `a_d[t]` | exact | [E] |
| static-pressure echo when the AHU is **off** | always 0 | [E] |
| static-pressure echo when **running** = max(command, **200 Pa**). 36,464 of 300,103 running AHU-steps were commanded < 200 Pa and read back 200 | — | [E] (new; undocumented floor) |
| recirculation echo when running = min(command, 0.7); when off = command | 3e-8 | [E] (refines the doc, which says "capped at 0.7" without the running condition) |
| `zone_info[t]` (zone temperature, heating and cooling setpoint) = sensors in `obs[t+2]` | temperature 9.5e-7 K, setpoints exact | [E] |
| Σ `zone_info[t].average_occupancy` = `num_occupants` in `obs[t+2]` | exact | [E] |
| same against `obs[t+1]` | up to 15.8 K, 5 K setpoint, 35 occupants | [E] |
| `boiler_info[t]` gas = `tank_state[t+1]` burner heat | 5e-8 relative | [E] |
| **HWS supply-water sensor in `obs[i]` = `tank_state[i]` tank temperature (lag 0)**; with lag 1 or 2 the error is up to 40 K | 1.5e-5 K | [E] (new; contradicts the blanket "sensors stale" statement) |
| lag of AHU supply-air temperature, flow, static-pressure *sensor* fields and outside-air temperature | — | **[N]** |

### 5.2 Information available at decision time t

The policy acts after seeing `o_t`. It has:

- `o_0 … o_t` and `a_0 … a_{t−1}`, plus static metadata (field names, index maps, normalization, AHU↔zone map, zone_map / floor_plan, schedule windows, holidays list).
- In `o_t`:
  - time features and comfort flags describe the **current** time `sim_ts[t]`;
  - action echoes show `a_{t−1}`;
  - **zone temperatures, zone setpoints and `num_occupants`** show the state at the **end of step t−2**, i.e. the state behind `r_{t−2}`;
  - the **HWS supply-water temperature** shows the tank at the **end of step t−1**.

Targets associated with `reward[t]`:

- **Comfort state:** zone temperatures, setpoints and occupancy after step t. Equal to `zone_info[t]` and to the sensor block of `o_{t+2}`. This depends on `a_{≤t}` only.
- **Tariff and carbon hour:** `sim_ts[t+1]` = decision time + 5 min (known in advance from `o_t`).
- **Energy:** `ahu_info[t]` (aggregate blower + |AC|), `boiler_info[t]` (gas, pump). Privileged, no observable counterpart.
- **Tank:** `tank_state[t+1]` (privileged), equal to the HWS sensor in `o_{t+1}`.

### 5.3 Off-by-one risks for Phase 2

1. **(High) Window too short for the comfort target.** `TransitionWindows(horizon=H)` yields `obs[t0 : t0+H+1]` and `reward[t0 : t0+H]` [S, E]. The comfort state of `reward[t0+H−1]` is in `obs[t0+H+1]`, which the window does not include. A loss pairing `reward[k]` with `obs[k+1]` would be silently wrong by one step: up to 15.8 K temperature error at transitions, and an occupied-comfort error MAE 2.7× larger than the occupancy-split error [E]. Fix: return H+2 observations, or define an explicit `sensor_target(k) = obs[k+2]` with a unit test.
2. **(High) Heterogeneous lag.** Zone sensors lag by one step; the HWS sensor does not. A dynamics model that treats the observation vector as uniformly delayed will misalign the tank. Each sensor group needs a lag measured before it is used as a target. Lags not yet measured: AHU sensors, flows, outside air.
3. **(Medium) Effective vs commanded actuation.** Static pressure (200 Pa floor while running, 0 when off) and recirculation (cap 0.7 while running) differ from the logged commands. The model must either take the effective values (computable from the command and the run bit, both known at decision time) or learn the clip.
4. **(Low) Initial rows.** `o_0` and `o_1` sensors precede any logged action, so the first two comfort targets have no in-dataset cause. This is negligible with long episodes.

---

## 6. Baseline / null equivalence findings

| Claim | Evidence | Tag |
|---|---|---|
| `baseline` uses the unmodified `WideHotStorageReservePolicy` with no explorer | `collect_episode`: `cfg is None` → `ev.initialize_policy(baseline_policy.WideHotStorageReservePolicy, …)`, action = `validate_action(get_policy_action(policy, obs))`, the same calls as `run_eval.py` | [S] |
| `null` disables every perturbation | `PROFILES["null"] = ExplorationConfig(name="null")`; all `*_prob` default to 0.0. The explorer still runs (exercising the subclass and knob code) | [S] |
| null ≡ baseline, bit-identical (actions, baseline actions, obs, reward, reward response, zone / AHU / boiler info, tank state, timestamps, calendar) | b8/atlanta_1 1,152 steps (E1); b2/fairbanks_2 and b7/atlanta_8 576 steps (E4) | [E] |
| baseline profile through the collector ≡ stored `run_eval` rollout | b8/atlanta_1, 8,640 steps: all per-step rewards equal, mean −0.004031689621114378 both | [E] |
| Only `timing_ms` (and wall/time fields in the episode JSON) differ between identical-seed runs | E1 check 5 | [E] |
| `b_t` recorded correctly | Without active policy-internal knobs, `b_t` = the policy's own output before action-level knobs. With them, `b_t` comes from a deep copy taken **before** `policy.get_action`, with the knobs cleared. Forcing the clone path every step under null gives `b_t ≡ a_t` and reproduces the stored baseline rollout (E3) | [S], [E] |
| Semantics of `b_t` | "the baseline's action from the *current* (possibly perturbed) internal state and the same `o_t`", not an independent unperturbed rollout. Documented in `phase1_dataset.md` §5.3 and plan §14.4 | [S] |
| Baseline sources unchanged | `git diff HEAD -- src/run_eval.py src/baseline_policy.py src/summarize.py` empty | [E] |

Limitation: null ≡ baseline is verified for 576–1,152 steps on 3 of 6 buildings, not at full horizon on all 30 scenarios [N]. Given the code path, that is low risk [I].

---

## 7. Exploration quality and safety

The knob schedule is a pure function of (seed, scenario, episode) [S]. The audit replayed it exactly and found **zero mismatches** between the requested schedule and the applied knob values [E].

### 7.1 Per-knob statistics

Moderate: 32 episodes (30 sample + 2 validation). Mild: 7 episodes (6 audit + 1 validation). Fractions are over steps (per AHU where applicable).

| Knob | Configured p (mod / mild) | Requested active (mod / mild) | Actually applied / effective (mod / mild) | Magnitude (moderate, effective) | Hold (active segments) |
|---|---|---|---|---|---|
| AHU schedule | 0.20 / 0.10 | 20.2% / 10.8% | forced ON 5.5% / 2.7%; **forced OFF 0.39% / 0.03%** (OFF requested 10.6%, but the baseline runs the AHU while unoccupied in only 0.07% of those steps) | ON: SAT 291 K, 300–900 Pa | mean 20.8 steps (6–36) |
| SAT offset | 0.50 / 0.30 | 49.8% / 28.2% | 23.4% / 12.1% (requested while AHU off: 26.1%) | Δ p5/p50/p95 = −3.5 / +0.55 / +3.7 K; extremes ±16–20 K from forced-ON or internal knobs | 42.3 (12–72) |
| Static pressure | 0.50 / 0.30 | 51.0% / 29.0% | 24.2% / 12.1% applied. **20–23% (mod) / 9–15% (mild) of running steps commanded < 200 Pa → floored, no effect.** The baseline itself is at 200 Pa (p5 = p95 = 200) | Δ p5/p50/p95 = −84 / −1 / +143 Pa (commanded) | 42.4 (12–72) |
| HWS setpoint | 0.50 / 0.30 | 48.8% / 24.1% | offset applied 48.4% / 23.7%; **effective 28.3% / 14.3%** (21% / 9.5% clipped to a no-op at 310 / 350 K) | effective Δ p5/p95 = −9.8 / +9.4 K | 85.9 (24–144) |
| Zone setback | 0.60 / 0.40 | 60.0% / 33.8% | effective (unoccupied) 34.8% / 19.6% | low +7.3 K mean (0.2–14.0), high −3.8 K mean | 178 (73–288) |
| Recovery lead | 0.50 / 0.30 | 53.2% / 28.1% | changed 53%; **matters only pre-start: 1.6% / 1.0% of steps** | 0.30–2.98 h (default 1.0) | 288 (one day) |
| Comfort guard (AHU) | — | — | 1.4% of AHU-steps (1.7% of requested); p95 per episode 6.6% | — | — |
| Comfort guard (HWS) | — | — | 0.38% of steps | — | — |

Requested-active fractions match the configured p within sampling noise, as designed [E]. Applied and effective frequencies are **roughly half or less** for SAT, pressure and HWS, and far less for forced-OFF and recovery lead [E].

### 7.2 Constraints and validity

- All executed continuous actions are within [−1, 1].
- Native ranges: SAT [285, 305] K, pressure [0, 2000] Pa, HWS [310, 350] K.
- Discrete actions are in {0, 1}.
- No forced-OFF occurs during comfort or pre-start hours.
- Forced-OFF is consistent (executed 0, baseline 1).
- Outside the knob channels, actions differ from `b_t` only when a policy-internal knob is active.

All of the above is [E] (E1 check 6, E2, E6). The simulator accepted every action: `reward == agent_reward_value`, so there were no rejection rewards [E]. The guard uses `o_t` sensors, which show the state two steps old (§5.2), so it reacts with that lag [S/I]; it fired rarely.

### 7.3 Diversity and comfort cost

- **Supervisory channels widen.** Moderate vs baseline [E]:
  - SAT visits 8.4–10.6 of the ten 2 K bins vs 6–7;
  - HWS takes 13–18 distinct 0.5 K values vs 3.5–5;
  - AHU run fraction 45–51% vs 39–43%;
  - pressure spans p5–p95 ≈ 125–650 Pa commanded vs the baseline's 200–235.
- **Zone-level actions barely change.** Dampers differ from `b_t` on 2.3–8.4% of steps, and valves likewise. VAV-level actions remain essentially the baseline's.
- **Comfort cost of exploration (1-week sample):**
  - mean comfort term −3.1e-4 vs −1.8e-4;
  - occupied zone-steps > 1 K outside the band: 0.09% vs 0.05%;
  - max occupied deviation per episode 1.56 K vs 1.11 K (means);
  - score 0.00728 vs 0.00617 (+18%).
- The degradation is bounded and comfort excursions remain rare. That makes the exploration safe, but it also leaves the high-penalty comfort region (deviation > 1–5 K) nearly unobserved [E/I].

### 7.4 Observer inconsistency

When SAT or pressure is perturbed after the fact, the baseline's internal history records its own commanded values [S]. Effect on the baseline's internal one-step model error (`policy.model_error`, EWMA of |residual|), b8, 1,152 steps [E]:

| Run | Mean model error | Score |
|---|---|---|
| null (= forced-clone) | 0.0155 | 0.00336 |
| internal knobs only | 0.0178 (+15%) | 0.00339 |
| action knobs only | 0.0212 (+37%) | 0.00405 |
| moderate (both) | 0.0236 (+52%) | 0.00413 |

Assessment:

- **Labels stay correct.** The dataset logs the executed actions and the simulator's true responses, so a dynamics model trained on (o, a_exec) → o′ sees consistent physics. The inconsistency lives only inside the behaviour policy [S/I].
- **The behaviour distribution is distorted.** A miscalibrated baseline produces slightly worse, more erratic closed-loop behaviour in perturbed segments. This is part of why action-only exploration costs more score than internal-only.
- **One risk for Phase 2.** Anything that imitates `b_t` or uses the baseline's internal estimates as features inherits the bias [I]. Severity: Medium.

---

## 8. Reward reconstruction metrics

The implementation was checked against the starter-kit source [S]:

- Comfort is −Σ occ·s(dev)/Σ occ, with s linear up to 5 K and quadratic beyond (unbounded).
- E = Σ blower + Σ |AC| + Σ pump + Σ (|HP pump| + |HP heat|), with |E| priced.
- G is clipped at ≥ 0.
- Energy and carbon are normalized by the same rates evaluated at 80,000·K and 205,000·K, with K from metadata.
- Electricity uses the weekday/weekend hour tables with `is_work_day` (US holidays) at the UTC-labelled reward start; electricity carbon is hourly; gas price is monthly; gas carbon is constant.
- Weights are 0.6/0.2/0.2, divided by their sum.

The tables equal the starter-kit constants and the K-derived normalizers equal every scenario's config [E].

### 8.1 Per-step |reconstructed − simulator| reward

60 sample episodes, 120,960 steps [E].

| Variant | MAE | Median | p95 | Max | Episode-mean rel. error (mean \|·\| / max \|·\| / mean signed) |
|---|---|---|---|---|---|
| **exact** (privileged) | 1.4e-10 | 5.4e-12 | 6.9e-10 | **1.5e-8** | 0 / 0 / 0 |
| exact, but comfort state from `obs[t+2]` | 1.6e-10 | 5.5e-12 | 7.5e-10 | 1.5e-8 | 0 / 0 / 0 |
| comfort state from `obs[t+1]` (wrong alignment) | 2.4e-4 | 5.7e-12 | 5.1e-4 | 0.197 | 0.25% / 1.6% / −0.09% |
| annual-mean gas price | 1.3e-4 | 5.7e-6 | 4.4e-4 | 1.1e-2 | 1.35% / 2.9% / **+0.84%** |
| work-day = Mon–Fri (no holidays) | 4.2e-7 | 6e-12 | 8e-10 | 2.7e-3 | 0.01% / 0.06% / 0 |
| uniform occupancy split | 9.0e-5 | 5.7e-12 | 1.1e-4 | 0.164 | 0.61% / 3.6% / +0.25% |
| area-weighted occupancy split | 1.0e-4 | 5.7e-12 | 1.3e-4 | 0.143 | 0.61% / 2.8% / +0.22% |
| **causal** (all observable: `obs[t+2]` comfort state, uniform split, `obs[t+1]` hour/dow, Mon–Fri, annual gas; E and G still privileged) | 2.0e-4 | 5.8e-6 | 6.0e-4 | 0.167 | **1.68% / 5.05% / +1.09%** |

**Conclusion.** The reconstruction is **numerically equivalent** to the simulator reward (float32 rounding level) when privileged inputs are used [E]. The deployable estimate is an **approximation**, with a systematic positive bias (cost underestimated) of about 1% of the episode reward [E]. The E and G terms in "causal" are still privileged: a deployed controller adds its own E/G prediction error on top, which is not measured here [N].

### 8.2 By component

MAE / max per step, weighted parts [E]:

| Variant | comfort | energy | carbon |
|---|---|---|---|
| exact | 5e-12 / 8e-9 | 7e-11 / 6e-9 | 6e-11 / 1e-8 |
| gas_mean | — | 1.3e-4 / 1.1e-2 | unchanged (gas carbon is constant) |
| occ_uniform | 9.0e-5 / 0.164 | — | — |
| causal | 9.0e-5 / 0.164 | 1.3e-4 / 1.1e-2 | 6e-11 / 1e-8 |

### 8.3 By building and weather

- **exact:** max ≤ 1.5e-8 for every building and every station [E].
- **causal, MAE (max) by station:** atlanta 1.1e-4 (0.037), denver 1.1e-4 (0.042), **fairbanks 4.7e-4 (0.167)**, phoenix 1.7e-4 (0.023), singapore 1.3e-4 (0.014).
- **causal, MAE by building:** b1 4.2e-4, b2 2.6e-4, b4 1.4e-4, b5 1.0e-4, b7 1.1e-4, b8 1.6e-4.
- The error concentrates where heating and comfort excursions happen: fairbanks, and b1/b2, the cold outliers [E].

### 8.4 Around occupancy / schedule transitions

Window: −3 to +4 steps around a change of occupancy>0 or `comfort_mode_now`; 4,846 steps [E].

| Variant | MAE near | MAE elsewhere | p95 near |
|---|---|---|---|
| exact | 4.1e-10 | 1.3e-10 | 1.8e-9 |
| occ_uniform | **1.2e-3** | 4.3e-5 | 5.8e-3 |
| occ_area | 1.1e-3 | 5.7e-5 | 5.6e-3 |
| obs_lag1 | 2.9e-3 | 1.3e-4 | 1.7e-2 |
| causal | 1.4e-3 | 1.5e-4 | 6.2e-3 |

### 8.5 Occupancy weighting

- **Uniform vs area-weighted:** uniform has the lower occupied-step comfort MAE (2.4e-4 vs 2.7e-4 overall). Per episode, uniform is better in 34/60 by MAE and 27/60 by bias, and the bias sign varies [E]. Statistically indistinguishable.
- **Cause [S]:** every zone gets the same `zone_assignment` head-count with independent random arrival and departure times. The expected split is uniform; the residual is the randomness of who is present, largest during arrival and departure.
- **Runtime rule:** only `num_occupants` (building total, `obs[t+2]` for `r_t`) and `comfort_mode_*` are observable. Per-zone `average_occupancy` is privileged and must never be a runtime input [S]. No current code uses it as one [S].
- **Recommendation:** uniform split at run time, plus a comfort reserve sized to the transition-period error. Optionally learn a per-zone presence prior from the observable total, *not* from `zone_info`.

---

## 9. Observable versus privileged features

| Feature | Runtime? | Privileged? | Allowed as model input? | Recommended use |
|---|---|---|---|---|
| `obs` (all D fields, raw) | yes | no | **yes** | Primary input. Mind the per-group lags (§5) and normalized vs native fields (`ObservationCodec`) |
| Action history `a_{<t}` (executed) | yes | no | **yes** | Input, together with the candidate `a_t…` in planning. Effective-value transform for pressure / recirculation (§5.3) |
| `baseline_continuous/discrete` (`b_t`) | yes, *if* the deployed policy runs the baseline internally | no | yes, with care | Prior / residual target. Biased under perturbation (§7.4) |
| Static metadata (field names, index maps, AHU↔zone, zone_map, floor_plan, schedule windows, holidays list, obs and action normalization, `zone_area_m2`, `zone_exterior_ua_proxy`) | yes (`create_building_meta_data` + derived) | no | **yes** | Graph construction, static node features |
| `reward_config` / K | yes (K from metadata; constants global) | no | yes | Cost model |
| `meta.privileged.*` (episode start/end, boiler parameters) | **no** | **yes** | **no** | Episode start reveals the month (leak). Boiler parameters may be used for a physics prior only as *global constants* (same in all 30 configs [E]), not per-building inputs |
| `num_occupants_normalization_constant` | yes (decodes an observable field) | no | yes | Decoding |
| `zone_info` (true T, setpoints, flows, **per-zone occupancy**) | no (T, setpoints and total occupancy appear in `obs[t+2]`; per-zone occupancy never) | **yes** | **no** | Targets: comfort state, flow heads. Per-zone occupancy only for offline exact reward |
| `ahu_info` (aggregate blower, AC) | no | yes | no | Energy targets (aggregate loss: not per AHU) |
| `boiler_info` (gas, pump) | no | yes | no | Energy and gas targets |
| `tank_state` (tank T, burner, ΔT, return water, boiler setpoint, flow) | partly (tank T = HWS sensor, lag 0) | yes | **no** | Tank-model targets, physics-prior fitting |
| `calendar.hour`, `minute`, `weekday` | derivable from `obs` time features | no | yes (derive from obs, not from `calendar`) | Tariff and carbon lookup |
| `calendar.month`, `day`, `is_workday` (holidays) | **no** | **yes** | **no** | Offline exact reward only. Runtime: annual-mean gas, Mon–Fri |
| `reward`, `reward_response` | no (not given to the policy) | yes | no | Targets |
| `knobs`, `perturbed` | no | yes (behaviour-policy internals) | no | Analysis, stratified sampling, coverage weighting |
| `timing_ms`, `episode_json` code hashes | no | — | no | Provenance |

**Leakage audit [S]:**

- No normalization, feature scaling or statistics are fitted anywhere in `src/` or `scripts/`. Observation normalization comes from the env and is runtime metadata.
- Nothing in batching mixes splits.
- **Risks:** `TransitionWindows` returns privileged arrays in the same dict as `obs`, by default including `zone_info`, `tank_state` and `calendar`. And `meta.json` co-locates `privileged.*` with runtime metadata.
- Phase 2 needs an explicit input allowlist and a separate target dict.

---

## 10. Building split and generalization

| Check | Result | Tag |
|---|---|---|
| No building in two splits (`assert_no_building_leakage`; `building_split` rejects val/test overlap) | yes | [E] |
| All weathers and episodes of a building in the same split | splits filter by `building`; val = all 5 b7 scenarios, train = 50 episodes of 5 buildings; overlap of episodes / scenarios / buildings = 0 | [E] |
| Windows cannot leak across splits | windows are built per episode from split-filtered rows; 2,000 sampled windows never cross an episode end; time order preserved; content equals the source | [E] |
| Overlapping windows (stride 1) within an episode | stay inside one split (same building) | [S/E] |
| LOBO folds | 6 folds, each building validated exactly once | [E] |
| Statistics fitted on val/test | none exist | [S] |
| `splits.json` | absent in the sample; the default (`read_split`) is used | [E] |

**What the current split can show:** transfer of a model to one unseen building of mid size (K=4, V=36) under weathers whose *station types* are all seen in training (every building has the same 5 stations).

**What it cannot show:**

- generalization across building sizes. b7 is close to b1 and b8 in size, so there's no test at K=9 or V=85;
- generalization to unseen stations or climates;
- cold-climate behaviour on an unseen building. b7's coldest scenario (fairbanks_7, September) has min 1.6 °C, while the train buildings see −39 to −17 °C (fairbanks_1/2/8) [E];
- any statistically meaningful estimate. n = 1 building.

**Recommended protocol:**

1. **Leave-one-building-out (6 folds)** as the primary metric. Report per fold, per station and per K, with the cold scenarios (fairbanks_1, _2, _8) broken out.
2. **Leave-station-out** as a secondary axis. Note it is confounded here, since all buildings share the same 5 stations.
3. Before claiming unseen-building generalization: **generated buildings** (plan §7.1) for training, with the 6 shipped buildings held out entirely. Plus a `minneapolis` station hold-out.
4. Always compare against the baseline's own one-step model (RLS) rolled forward, on the same folds.

---

## 11. Dataset statistics and coverage

| Quantity | Value | Tag |
|---|---|---|
| Buildings | 6 (K = 3, 4, 4, 6, 8, 9; V = 30–85) | [E] |
| Weather scenarios | 30 (5 stations × 6 buildings; each scenario a different start date, Jan–Nov) | [E] |
| Sample episodes | 60 = 30 baseline + 30 moderate, 2,016 steps (1 week) each, seed 0 | [E] |
| Sample transitions | 120,960 (train split 100,800 / val 20,160) | [E] |
| Sample size | 94 MB of episode archives (0.47 MB per 1k transitions at V=30, 1.23 MB at V=85); 196 MB including copied zone maps | [E] |
| Other data | validation: 6 × 1,152 steps (b8) + 1 × 8,640 (b8 baseline); audit scratch: 6 mild × 2,016, 2 null × 576 | [E] |
| Perturbed-step fraction | moderate 61–96% per episode (mean ≈ 0.78); mild 0.45–0.67 | [E] |
| Reward | mean −0.0062 (baseline) / −0.0073 (moderate); min −0.089 / −0.096 (episode means of the minimum); exactly 0 on 33% / 20% of steps (plant off, building empty) | [E] |
| Zone temperature | 270.6–308.0 K overall | [E] |
| Occupied comfort | deviation > 0 rare; > 1 K on 0.05% / 0.09% of occupied zone-steps | [E] |
| Electricity per AHU | mean 1.3 / 1.5 kW | [E] |
| Gas per AHU | mean 2.7 / 3.2 kW; zero on 60–67% of steps; fairbanks 10.7 kW, singapore 0 | [E] |
| Tank | 310–350 K (the full setpoint range) | [E] |
| Outdoor | −39.4 °C (fairbanks_1) to +46.1 °C (phoenix_5) | [E] |
| Missing / invalid values | none | [E] |

**Distribution differences.** Building b1 has 2.2× the gas of b2 per AHU (7.1 vs 3.2 kW) and the worst rewards. Station dominates the energy mix: fairbanks is gas-heavy, phoenix and singapore are electricity-only. Each scenario covers only its **first week**, so a scenario's season is fixed by its start month [E].

**Sufficiency:**

- *Sufficient* for an initial dynamics-model experiment: one-step and multi-step zone-temperature and tank prediction; E/G heads; in-distribution and LOBO evaluation on 6 buildings [I].
- *Not sufficient* to claim unseen-building generalization: 6 buildings, 1 week per scenario, near-baseline behaviour, sparse comfort excursions, little VAV-level and forced-OFF variation [I].
- The transition count (121k) is not the binding constraint; the number of independent buildings and the operating-regime coverage are.

---

## 12. Risk register

| # | Risk | Severity | Evidence | Recommended action |
|---|---|---|---|---|
| R1 | Observation-delay misuse in training (window lacks `obs[t+H+1]`; heterogeneous lag: zone sensors lag 1, HWS sensor lag 0; other groups unmeasured) | **High** | §5.1, §5.3 [E] | Before Phase 2 training: make windows return H+2 observations or an explicit per-group target map; measure lags of AHU, flow and outside-air sensors; add unit tests asserting `zone_info[t] == decode(obs[t+2])` and `tank[t+1] == decode(obs[t+1])`; correct `phase1_dataset.md` §3.1 |
| R2 | Reward reconstruction wrong offline | **Low** | exact ≤ 1.5e-8 everywhere [E] | — |
| R2b | Deployable reward estimate biased (+1.1% mean, 5% max; gas month dominant; occupancy-split noise at transitions) | **Medium** | §8 [E] | Phase 2: optional causal season estimate from outdoor-temperature history (gas price); comfort reserve near transitions; measure the planner's sensitivity |
| R3 | Privileged leakage (per-zone occupancy, month, tank internals) into inputs | **Medium** (no current leak; boundary not enforced) | §9 [S] | Explicit input allowlist plus separate targets in the Phase 2 loader; move `meta.privileged` and `calendar.month/day/is_workday` out of any feature path; assertion test |
| R4 | Building leakage across splits | **Low** | §10 [E] | Keep `assert_no_building_leakage` in every training entry point; write `splits.json` per study |
| R4b | Validation design cannot show generalization (n = 1 building, no cold climate, no size range) | **High** | §10, §11 [E] | LOBO 6-fold as primary; per-station and per-K reporting; generated buildings before generalization claims |
| R5 | Exploration coverage: pressure perturbations below the 200 Pa floor are no-ops (and the commanded value is logged); forced-OFF ≈ 0; VAV-level actions ≈ baseline; recovery lead effective 1–2% of steps; comfort-edge region nearly unobserved | **High** | §7.1, §7.3, §5.1 [E] | Before scaling collection: pressure scale only upward from the floor (or log the effective pressure); a gated occupied forced-OFF profile; VAV damper / valve perturbations; occasional bounded comfort excursions; collect later-start windows and the full 30 days |
| R6 | Observer inconsistency in the behaviour policy | **Medium** | model error +37% action-only, +52% moderate [E] | Don't imitate `b_t` from perturbed segments without weighting; optionally feed the executed SAT/pressure back into the baseline history in a future collector (behaviour change, so needs approval) |
| R7 | Distribution shift: MPC actions outside the near-baseline data support | **High** | §7.3 diversity, §11 [E/I] | Plan the action space as supervisory knobs that map through the baseline allocator (plan §6.5), whose outputs are in support; add an uncertainty/ensemble OOD penalty; iterate data collection with MPC-in-the-loop (DAgger-style) |
| R8 | Model exploitation (planner finds optimistic model errors, especially comfort under deep setback / forced-OFF / low pressure) | **High** (for Phase 2 design; not a data bug) | sparse comfort excursions, forced-OFF ≈ 0 [E] | Trust region around the zero-offset plan; ensemble disagreement penalty; always evaluate the baseline plan; closed-loop validation on the simulator, never model-only |
| R9 | Effective vs commanded actuation (pressure floor 200 Pa, off → 0; recirculation cap 0.7 when running) | **Medium** | §5.1 [E] | Feature transform for effective values; document in the schema |
| R10 | `done` never set; episode end = file end | **Low** | §4 [E] | Document; never rely on `done` for boundaries |
| R11 | Schema doc inaccuracies (blanket "sensors stale"; recirculation cap condition; pressure floor missing) | **Low** (documentation) | §5.1 [E] | Update `phase1_dataset.md` |
| R12 | Each scenario observed only from its start date, 1 week | **Medium** | §11 [E] | Full-horizon collection for training; sub-sample windows |
| R13 | Comfort guard acts on 2-step-old zone state | **Low** | §5.2 [S/I] | Acceptable; guard rarely fires |
| R14 | null ≡ baseline not verified at full horizon on every scenario | **Low** | §6 [E for 3 buildings] | Optional overnight check |

No **Critical** items were found.

---

## 13. Recommendations before Phase 2

**Conditions** (must be done at the start of Phase 2, before any model is trained on the data):

1. **Alignment-safe loader (R1).** Window = H+2 observations (or an explicit `sensor_target` map). Unit tests: `zone_info[t] ≡ obs[t+2]` sensors; `tank_state[t+1] ≡` HWS sensor in `obs[t+1]`; the tariff hour comes from `obs[t+1]`. Measure the lag of the remaining sensor groups (AHU supply-air temperature, flows, static-pressure sensor, outside air).
2. **Input / target separation (R3).** An explicit runtime-feature allowlist; privileged arrays exposed only as targets; `meta.privileged` and calendar month / day / is_workday excluded from features; a test that fails if a privileged key reaches the model input.
3. **Evaluation protocol (R4b).** Leave-one-building-out (6 folds) with per-station, per-K and cold-scenario breakdowns. Compare against the baseline's own RLS model rolled forward.
4. **Effective actuation (R9).** Derive effective static pressure (`run · max(cmd, 200)`) and recirculation (`min(cmd, 0.7)` when running) as model inputs, or document that the model must learn them.

**Should do before or alongside the first training run** (High):

5. **Exploration revision for the next collection (R5, R7, R8).** Changes collection behaviour, so it needs explicit approval:
   - pressure scale ≥ 1 above the floor, or relative to 200 Pa;
   - an opt-in occupied forced-OFF profile with a short hold and a strict guard;
   - VAV damper / valve offsets;
   - rare, bounded comfort excursions to populate the penalty region;
   - start offsets and full 30-day episodes.
6. **Scale-out (R12).** Full-horizon `baseline` plus 2–3 seeds of exploration for the 30 scenarios (about 90 episodes, roughly 1–3 h on 14 workers at Phase 1 throughput [I]). Then generated buildings (plan §7.1).

**Documentation (Low):** fix `phase1_dataset.md` §3.1 (HWS lag 0, pressure floor, recirculation condition, `done` semantics).

---

## 14. Final readiness decision

**READY WITH CONDITIONS.**

Phase 1 produces a correct, reproducible and internally consistent dataset:

- exact alignment, verified across 66 episodes;
- exact offline reward;
- bit-exact baseline equivalence and replay;
- no building leakage.

Phase 2 may begin under conditions 1–4 of §13: an alignment-safe loader, an enforced input/target boundary, leave-one-building-out evaluation, and effective-actuation features.

The exploration-coverage and dataset-scale limitations (R5, R7, R8, R12, R4b) do not block an initial dynamics-model experiment. They do block any claim of unseen-building generalization or MPC robustness, and they should be addressed before the large-scale collection.

Phase 2 has **not** been started.
