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
3. Reproduce the four shipped baselines on the 30 training scenarios. **Gate:** scores match
   `baseline_returns.json` (`pid_water` 0.05414, `pid` 0.06736, `fixed_setpoint` 0.06592, `bang_bang` 0.2252).
4. Benchmark throughput on this machine (32 threads): seconds per rollout by building size, at 1 vs 16 vs 30
   workers. That sets the tuning budget.

**Deliverable:** a trusted evaluator, a throughput table.

### M1 — Starting controller (Nov 27 – Dec 2)

1. **Decision: which controller to start from.**
   - **Recommended:** the public analytical notebook controller `WideHotStorageReservePolicy` (0.00695 locally;
     recoverable from tag `archive/learned-model-phase2c`, `src/baseline_policy.py`). Check the license of each
     source notebook and keep attribution.
   - Alternative: write our own (SADR-style: occupancy-gated modes, per-zone PI, SAT and static-pressure resets,
     economizer, hot-water reset). Cleaner to parametrize, but weeks of work to reach 0.007.
2. Run it on the 30 training scenarios. **Gate:** reproduces 0.00695.
3. **Submit it to the new public leaderboard.** This gives the real gap to #1 and the local-to-leaderboard ratio.
4. Loss breakdown per scenario: comfort / electricity / gas, by station, by hour of day, occupied vs unoccupied,
   weekday vs weekend. List the five biggest sources of loss.

**Deliverable:** baseline leaderboard score, loss breakdown, ranked list of levers.

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
4. **Cold-weighted objective.** Weight stations to match the hidden set as estimated (≈ 1/3 deep cold). Re-estimate
   the weights once the M1 leaderboard ratio is known.

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
   objective). Estimated ≈ 100–150 core-hours, about 4–6 h on 30 workers.
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
   snapshots) from the archive tag.
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
   with a margin. The policy should take ≤ 5 ms per step.
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
