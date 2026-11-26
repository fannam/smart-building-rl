# Smart Buildings Challenge (NeurIPS 2026, Kaggle) — Approach

One HVAC control policy for unseen buildings and weather. Score = mean over scenarios of |mean per-step reward|,
lower is better. Rules, interface and scoring: [`smart_buildings_competition_notes.md`](smart_buildings_competition_notes.md).
Final submission deadline 2027-01-15; accept the rules by 2027-01-03.

Earlier work (learned dynamics model / latent MPC, phases 0–2C) is at commit `b984845` (the tag
`archive/learned-model-phase2c` was never pushed).

## Where we stand (2026-11-24)

- The public leaderboard was reset. #1 is now ≈ **0.008**. Earlier top scores of ≈ 0.0045 came from a gas-billing
  quirk that the current simulator (energy-balanced hot-water tank) no longer has.
- The public board scores ≈ 1/3 of the hidden set (≈ 10 scenarios), so differences below ≈ 0.0005 are noise.
- Before the reset, local scores on the 30 training scenarios were ≈ 1.4× better than leaderboard scores. The hidden
  set leans to deep cold weather. On the new board the ratio is still ≈ 1.44 (reference controller: 0.00695 local,
  0.01000 public on 2026-11-26, about rank 270).
- Kit baselines on the 30 training scenarios, current simulator (`1fb75606`, full 8,640 steps; reference only, run
  `baselines`, 2026-11-25): `pid_water` 0.05787, `fixed_setpoint` 0.06708, `pid` 0.07068, `bang_bang` 0.24150. The kit's
  `baseline_returns.json` predates the 2026-11-16 simulator changes (energy-balanced tank, airflow-weighted return
  air) and is not used. A public analytical notebook controller
  (`WideHotStorageReservePolicy`, derived from penguin069 / Daniel Wu / jazivxt notebooks) scores 0.00695 locally on
  the current simulator (re-measured 2026-11-25; ported as `sbx/controller/reference.py`). Loss breakdown and ranked
  levers: `PLAN.md`, M1 results.

## Approach: a tuned analytical controller

**Why not RL or online learned-model MPC**
- Runtime: the gateway runs every hidden scenario (probably 48) sequentially in one session. The simulator alone takes
  ≈ 50 ms/step on Kaggle CPU for the smallest building, and a neural ensemble call cost ≈ 180 ms. There is
  little time for online planning.
- The loss is mostly cold-weather gas (≈ 75% on Kaggle). Learned models were weakest exactly there.
- Evidence: the only published leaderboard-verified gain is CMA-ES tuning of a controller's constants on generated
  buildings (−16%). A public perfect-foresight bound puts multi-step planning headroom at ≈ 10%.
- RL is too sample-hungry at 16–50 ms per simulator step and 8,640 steps per episode.

**Plan**
1. **Baseline and real numbers.** Write a clean evaluator, reproduce the shipped baselines, and pick the strongest
   analytical controller as the starting point. Submit it to the new leaderboard to measure the real gap to #1.
2. **Validation set that looks like the hidden one.** Generate ≈ 24+ buildings with the procedural generator, times 6
   stations, including the unused `minneapolis` strips. Weight cold weather more. Use 7–14 day windows for tuning and
   full 30 days for final checks. Report gas, electricity and comfort separately. Always score on held-out buildings.
3. **Expose the controller's settings** (≈ 20–30 constants): setback depth and frost guard, optimum-start lead time
   vs outside temperature, AHU shutdown, supply-air temperature and static-pressure resets, recirculation in heating,
   hot-water setpoint and tank band, reheat vs AHU heating split. Make some depend on context (K, V, inferred
   envelope, outside-temperature regime).
4. **Tune with CMA-ES against the simulator**, in parallel (e.g. 16 candidates × 24 seven-day scenarios × 30
   generations ≈ 110 core-hours locally).
5. **Optional oracle study.** Use bit-exact environment copies on cold windows to measure what recovery timing and
   setback depth are worth. If more than ≈ 5%, add a cheap supervisory MPC (simple per-zone thermal + tank model,
   2–3 knobs, replanning every 15–60 min) or distil the result into rules.
6. **Make it safe, then submit.** Sanitize every action (finite, clipped to `action_spec`, right dtype): one invalid
   action fails the whole submission. Fall back to the baseline on any error. Time a full sweep on Kaggle hardware.
   Submit two final versions: one conservative, one more aggressive.

## Verified simulator facts (from the archived phase 0 study)

- **Gateway** (`kaggle_evaluation` in the competition data): scenarios run sequentially, one fresh policy per
  scenario, metadata only on the first turn. Timeouts: 15 min for the first turn, 60 s for every later turn. Any
  rejected or NaN action ends the whole submission.
- **Reward per step:** `r = 0.6·comfort − 0.2·energy − 0.2·carbon`.
  - Comfort: occupancy-weighted zone deviation `d` outside the band; penalty `d/5` up to 5 K, then
    `1 + ((d−5)/5)²`. Zero when the building is empty.
  - Normalizers scale with K (number of AHUs): electricity 80 kW·K, gas 205 kW·K.
- **Tariffs:** electricity 16/18/20 ¢ on weekdays by hour (flat 16 ¢ at weekends), carbon by hour, gas price by month.
  Hour and weekday are observable, the month is not (annual-mean gas price gives ≤ 5% error). Time-of-use swing is
  only ≈ 15%, so load shifting is a small lever. Per watt, electricity costs ≈ 2× as much reward as gas.
- **Hot-water tank:** energy-balanced, 1.5 m³, burner ramp limit 0.5 K/min (≈ 52 kW above draw + loss), gas billed
  at 100% of burner heat. Storing heat only helps for capacity during morning recovery.
- **Simulator** is deterministic from a fresh environment. `deepcopy(env)` fails on a SciPy `SuperLU` cache; share
  the factorization in the memo and snapshot the global RNG streams (numba, NumPy, `random`) for bit-exact branches.
- **Pitfalls others hit:** a float32 day-of-week rounding put Saturday on a workday (1.5–5% extra gas); black-box
  tuning drifted to freezing empty buildings (add a frost guard); one runaway zone can dominate a scenario through
  the quadratic comfort term.
