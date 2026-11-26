# Plan — tuned analytical controller

> Written 2026-11-24. Deadlines: accept the rules by **2027-01-03**; final submission **2027-01-15**.
> Approach and verified simulator facts: [`README.md`](README.md). Rules and interface: [`smart_buildings_competition_notes.md`](smart_buildings_competition_notes.md).

**Goal:** finish in the top 8 on the private leaderboard (public #1 ≈ 0.008 on 2026-11-24) with one analytical
controller whose constants are tuned by CMA-ES against the simulator, and which generalizes to unseen buildings and
cold weather.

**Principles**
- The simulator is the only judge. Every number comes from full or windowed simulator rollouts, never from a model.
- Always score on buildings the tuning never saw. Weight cold weather, which is where the hidden set and the gas
  loss are.
- Use the public leaderboard sparingly (5 submissions a day, about 10 scenarios, ±0.0005 noise). It checks the
  local-to-leaderboard ratio; it is not an optimization target.
- Keep every step reproducible: a fixed scenario manifest, seeds, the parameter vector and the git SHA in every
  result file.

---

## Repository layout (to build)

```
sbx/
  env.py            # load scenario, build metadata, run a rollout (full or N steps), action sanitizer
  evaluate.py       # parallel runner -> one JSON per (policy, params, scenario): score + comfort/elec/gas parts + timing
  score.py          # aggregate: leaderboard score, per-station / per-building tables, cold-weighted score
  scenarios.py      # generate buildings + scenarios, write manifests (train / val / test splits)
  controller/
    base.py         # the analytical controller, all constants read from one Params object
    params.py       # parameter definitions: name, default, bounds, scale (log/linear), optional context terms
  tune/cmaes.py     # CMA-ES loop (resumable, parallel, logs every candidate)
  oracle/branch.py  # bit-exact env copy for the optional oracle study
submission/
  policy.py         # Kaggle gateway Policy wrapper + watchdog fallback
  build.py          # pack code + params JSON into the Kaggle notebook / dataset
manifests/          # scenario lists per split (committed)
results/            # JSON summaries only (arrays gitignored)
tests/
```

---

## Milestones

### M0 — Foundation (Nov 25 – Nov 28)

1. Clone the starter kit at `1fb75606` and download the **current** Kaggle competition data. Diff its
   `smart_buildings/` against the kit. If the simulator changed with the leaderboard reset, pin the bundle's version
   instead.
2. Write `sbx/env.py`, `evaluate.py` and `score.py`. Make the action sanitizer mandatory in every rollout.
3. Run the four kit baselines on the 30 training scenarios on the current simulator, as reference numbers only
   (`baseline_returns.json` predates the 2026-11-16 simulator changes and is not a target). Done 2026-11-25:
   `pid_water` 0.05787, `fixed_setpoint` 0.06708, `pid` 0.07068, `bang_bang` 0.24150; no failures, 0 sanitizer fixes.
4. Benchmark throughput on this machine (32 threads): seconds per rollout by building size, at 1 vs 16 vs 30
   workers. That sets the tuning budget.

**Deliverable:** a trusted evaluator, a throughput table.

**Throughput (done 2026-11-25, `python -m sbx.bench`, pid_water, 864 steps, `results/bench/throughput.json`):**

| Building | K/V | 1 worker ms/step | 1 worker full episode | 16 workers steps/s | 30 workers steps/s |
|---|---|---|---|---|---|
| train_8 | 3/30 | 16 | 2.8 min | 179 | 171 |
| train_7 | 4/36 | 18 | 3.1 min | 155 | 155 |
| train_1 | 4/39 | 26 | 4.5 min | 113 | 93 |
| train_4 | 6/54 | 37 | 6.3 min | 77 | 69 |
| train_5 | 8/79 | 38 | 6.5 min | 76 | 74 |
| train_2 | 9/85 | 53 | 9.0 min | 46 | 46 |

- Throughput saturates at about 16 workers (2.6–3.8x one worker); 30 workers add nothing. Likely memory-bandwidth
  or cache bound. Policy cost is < 1 ms/step; nearly all time is the simulator.
- Mean ≈ 88 steps/s at 16 workers over the six buildings. The M4 round as first planned (16 candidates x 24
  seven-day scenarios x 30 generations ≈ 23 M steps) would take **≈ 73 h**, not 4–6 h. See the M4 budget note.

### M1 — Starting controller (Nov 27 – Dec 2)

1. **Decision: which controller to start from.**
   - **Recommended:** the public analytical notebook controller `WideHotStorageReservePolicy` (0.00695 locally;
     recoverable from commit `b984845`, `src/baseline_policy.py`). Check the license of each
     source notebook and keep attribution.
   - Alternative: write our own (SADR-style: occupancy-gated modes, per-zone PI, SAT and static-pressure resets,
     economizer, hot-water reset). Cleaner to parametrize, but weeks of work to reach 0.007.
2. Run it on the 30 training scenarios. **Gate:** reproduces 0.00695.
3. **Submit it to the new public leaderboard.** This gives the real gap to #1 and the local-to-leaderboard ratio.
4. Loss breakdown per scenario: comfort / electricity / gas, by station, by hour of day, occupied vs unoccupied,
   weekday vs weekend. List the five biggest sources of loss.

**Deliverable:** baseline leaderboard score, loss breakdown, ranked list of levers.

**Results (2026-11-25, run `m1_reference`, `python -m sbx.breakdown m1_reference WideHotStorageReservePolicy`):**
- Ported unmodified to `sbx/controller/reference.py` from commit `b984845` (the `archive/learned-model-phase2c` tag
  does not exist; the pre-restart commit does). Attribution in `sbx/controller/ATTRIBUTION.md`. **Licenses of the
  four source notebooks are not yet verified** (Kaggle was unreachable from this machine).
- **Gate passed:** 0.00695 on the 30 training scenarios, the same as the archived number, so that number was
  already on the current simulator. Comfort 0.00027 · electricity 0.00329 · gas 0.00339. Policy ≈ 2–7 ms/step.
- Submission: `python -m submission.build` writes `submission/inference.py` (controller + sanitizer + fallback
  to the last valid action) and `submission/notebook.ipynb`. Bit-identical rewards to the raw controller locally.
  **Public leaderboard (2026-11-26): 0.01000, about rank 270** (#1 ≈ 0.008). Leaderboard / local = 1.44, the same
  ≈ 1.4× ratio as before the reset. To reach #1 we need ≈ 20% better on the board; the crowd around 0.0100 is
  mostly this public controller family.
  The Kaggle "save version" run took **14–15 min** for **one** gateway scenario: its `submission.parquet` has a
  single row, `building_train_1/atlanta_2`, mean reward −0.0027447, the same as our local rollout (−0.0027447),
  so the Kaggle run matches ours. That is ≈ 100 ms/step on Kaggle CPU for a
  K = 4 building, about 3× this machine's single-worker speed (≈ 31 ms/step all-in). The controller is ≈ 2–7
  ms/step here, so ≈ 6–20 ms/step on Kaggle: already 10–20% of the per-step budget.

**Loss sources, ranked** (additive shares of the 0.00695 score):

| # | Source | Share | Notes |
|---|---|---|---|
| 1 | Cold-weather gas | fairbanks 42.6% (gas 0.00258) | Two scenarios, `building_train_1/fairbanks_1` (0.0465 alone) and `building_train_8/fairbanks_8`, are 32.7% of the score. Matters most for the cold-leaning hidden set. |
| 2 | Morning recovery, 05:00–08:00 | 35.7% for 05–07 (gas 0.00154, elec 0.00073, comfort 0.00021) | 05:00 is before occupancy and alone is 10.8%. Nearly all comfort loss is at 06:00 (late recovery on cold mornings). Levers: optimum-start timing, setback depth, tank pre-charge. |
| 3 | Daytime electricity in warm climates | elec 0.00253 over 08–18; phoenix + singapore + atlanta elec 0.00284 | Flat ≈ 0.00023 per hour across the occupied day: fans and cooling. Levers: static-pressure and supply-air resets, economizer, minimum airflow. |
| 4 | Daytime gas while occupied | gas 0.00176 over 08–18 | Reheat and keeping the tank at 340–350 K. Levers: hot-water setpoint and band, reheat vs AHU heating split. |
| 5 | Comfort spikes | comfort only 3.9% overall, but the worst 1% of steps hold 63–97% of each scenario's comfort loss | Cold-morning runaways; the quadratic penalty makes this the main tail risk on unseen cold buildings. |

Nights (19:00–05:00) are 1.8% and weekends 1.6%: setback is already cheap, so little headroom there.

### M2 — Validation suite (Nov 28 – Dec 4, in parallel with M1)

1. Generate about 40 buildings with the procedural generator (2–4 floors, K 3–9). Check that they are not much more
   expensive to simulate than the shipped six, and cap the size if needed.
2. Build scenarios for all 6 stations, including the unused `minneapolis` strips, in three splits by building:
   - **tune:** about 24 buildings;
   - **val:** about 8 buildings (model selection, early stopping);
   - **test:** about 8 buildings plus the 30 shipped scenarios (touched only at milestones).
3. **Short windows for tuning.** Check whether a 7- or 14-day rollout ranks controllers the same as the full 30
   days. Compare 10 controller variants. **Gate:** Spearman ρ ≥ 0.9 between window and full score. Otherwise tune on
   full episodes with fewer scenarios.
4. **Cold-weighted objective.** Weight scenarios to match the hidden set. Classifying each scenario by its strip's
   mean outdoor temperature (deep cold < −5 °C) instead of by station: the 3 deep-cold shipped scenarios average
   0.0263 and the other 27 average 0.0048, so the 0.0100 public score fits a **≈ 24% deep-cold** mix. (An
   earlier station-level estimate of ≈ 50% was wrong: many Fairbanks strips are summer.) The objective gives
   deep-cold scenarios 25% of the weight (`sbx.scenarios.DEEP_COLD_SHARE`); the reference controller then scores
   0.01018 locally vs 0.01000 public. Still a single calibration point on ≈ 10 scenarios: re-check after each probe.

**Progress (2026-11-26):**
- The hidden buildings come "from the same generator and parameter ranges as the shipped six", but the shipped
  six (318–430 × 188–301 ft, rooms 25–35 ft) are smaller than the generator's current defaults (400–700 ×
  300–500 ft, rooms 30–60 ft). Regenerating shipped seeds 0 and 8 with the defaults gives 597×494 and 516×394 ft,
  so the shipped set came from smaller ranges. Default-range buildings have K 5–12, V 32–120 and are 2–3× slower
  to simulate (cost tracks footprint × floors, r = 0.87, not zone count). Decision: the main suite uses ranges
  fitted to the shipped six; the larger defaults are a separate robustness split.
- The hidden weather strips are not among the 48 shipped. Shipped strips are 30-day windows of the 2023 station
  CSVs starting on Sundays at 06:00 and use each Sunday Jan 1–Nov 26 exactly once across the six stations. Per
  station, the 40 unused Sundays become new strips (240), split by date 3:1:1 into tune / val / test.
- `python -m sbx.scenarios manifest | generate | summary` (`sbx/scenarios.py`): 48 buildings (seeds 1000+ shipped
  profile, 2000+ kit default), one strip per station plus one extra deep-cold strip per building:

  | Split | Buildings | Scenarios | Deep cold | Cold | Mild | Hot |
  |---|---|---|---|---|---|---|
  | tune | 24 | 168 | 36 | 17 | 47 | 68 |
  | val | 8 | 56 | 13 | 4 | 16 | 23 |
  | test | 8 | 56 | 11 | 5 | 18 | 22 |
  | test_large (kit-default sizes) | 8 | 56 | 11 | 5 | 18 | 22 |

  Building and weather YAMLs and `manifests/suite.yaml` are committed; scenario dirs (1.8 GB) are regenerated
  into `data/scenarios/`. `python -m sbx.evaluate --split val ...` runs a split.
- Window check (`python -m sbx.windows`): window scores are the first N days of full rollouts, so no extra
  simulation. Running 7 controllers from the reference's own class chain (close in quality) on the 30 shipped
  scenarios for the gate; with only the far-apart kit baselines, 3- and 7-day windows already swap `pid_water`
  and `fixed_setpoint`.

**Deliverable:** committed manifests, a validated tuning objective.

### M3 — Parametrize the controller (Dec 1 – Dec 6)

1. Refactor the controller so every constant comes from `Params`, with defaults equal to today's behaviour.
   **Gate:** the default parameters reproduce M1 exactly.
2. Expose about 20–30 parameters, prioritized by the M1 loss breakdown:
   - setback depth (weeknight and weekend) and a **frost guard**;
   - optimum-start lead time as a function of outside temperature and zone temperature;
   - AHU shutdown and restart rules;
   - supply-air temperature reset, static-pressure floor and reset;
   - recirculation limit in heating vs cooling;
   - hot-water setpoint by season, tank band, pump differential pressure;
   - reheat vs AHU heating split;
   - comfort margin inside the band.
3. Fix known pitfalls: day-of-week float rounding (Saturday treated as a workday), causal holiday detection, the
   runaway-zone guard for the quadratic comfort term.
4. **Context:** let a few key parameters depend on building features known at runtime (K, V, zones per AHU, floor
   area, envelope estimated online) and the outside-temperature regime, as a small linear map.

**Deliverable:** `controller/` with documented parameter bounds and unit tests (actions valid for every building in
the suite).

### M4 — Tuning (Dec 6 – Dec 22)

1. **Round 1:** CMA-ES on the global parameters (population ≈ 16, about 30 generations, the tune split, windowed
   objective). **Budget (from the M0 throughput table):** ≈ 88 steps/s locally, so the original sizing (≈ 23 M
   steps) is ≈ 73 h. Cut it to ≤ 12 h per round: 3-day windows if the M2 gate allows, 8–12 scenarios sampled per
   generation, generated buildings capped at K ≤ 6, and extra workers on the L4 VM / Kaggle. Also try pinning
   workers to P-cores to see whether scaling improves.
2. Every 5 generations, score the best candidate on **val** (full episodes). Stop or restart if the val score stops
   improving while the tune score keeps falling.
3. **Round 2:** add the context parameters, warm-started from round 1.
4. **Robustness check** of the best candidate on the test split, by station and building size. **Gate:** it beats
   the M1 controller on at least 80% of test scenarios and never loses by more than 10%. If it fails, add a penalty
   on worst-case scenarios to the objective.
5. Leaderboard probes: submit only at the end of a round (about 1 to 2 a week). Track the local-to-leaderboard ratio.

**Deliverable:** the tuned parameter set, a test-split report, leaderboard scores.

### M5 — Oracle study, optional (Dec 8 – Dec 15, in parallel with M4)

1. Port the bit-exact environment branching (env copy that shares the SuperLU factorization, plus global RNG
   snapshots) from commit `b984845`.
2. On about 6 cold windows of 3 days, search recovery start time and setback depth every 15 min with simulator
   lookahead, starting from the M4 controller.
3. **Gate:** if the oracle gains **≥ 5%** on these windows, add a cheap supervisory planner (per-zone RC + tank
   model, 2–3 knobs, replanning every 15–60 min, ≤ 5 ms per step) or distil the oracle's choices into rules. Below
   5%, stop.

### M6 — Hardening and submission (Dec 20 – Jan 12)

1. `submission/policy.py`: the gateway `Policy` wrapper. Sanitize every action (finite, clipped to `action_spec`,
   correct dtype and shape). Fall back to the plain controller on any exception, and per step if needed.
2. Fuzz test: run every building in the suite plus randomly generated ones for one day each, with assertions on the
   actions.
3. Run a full sweep on Kaggle hardware (48 scenarios if that is the true count) and check that it fits within 9 h
   with a margin. The policy should take ≤ 5 ms per step locally. Known point: the reference controller's
   submission (≈ 100 ms/step all-in on Kaggle for a K = 4 building) completed scoring, so any extra per-step cost
   must be measured against that, not assumed free.
4. **Accept the competition rules by Jan 3.**
5. Final picks (Jan 12–15):
   - **Conservative:** the best test-split score with the smallest worst case.
   - **Aggressive:** the best cold-weighted score.

---

## Compute

| Resource | Use |
|---|---|
| Local i9-14900HX, 32 threads | evaluator, tuning (main) |
| L4 VM | extra CMA-ES workers (CPU-bound: the GPU is not needed) |
| Kaggle CPU sessions | leaderboard submissions, runtime check, extra rollouts if needed |

## Risks

| Risk | Mitigation |
|---|---|
| The simulator changed with the leaderboard reset | M0 step 1: diff the fresh bundle before anything else |
| Overfitting to the tune buildings or public leaderboard | separate val/test buildings; few leaderboard probes; pick finals on the test split |
| Short windows don't rank like full episodes | M2 gate; fall back to full episodes |
| Black-box tuning exploits the reward (frozen empty buildings, runaway zones) | frost guard, comfort guard, inspect the loss breakdown of each new best |
| One invalid action fails the whole submission | mandatory sanitizer, fallback, fuzz tests |
| License of the public controller doesn't allow reuse | check in M1; fall back to writing our own |
| Timeline slip | M5 is optional; M6 starts no later than Dec 22 |
