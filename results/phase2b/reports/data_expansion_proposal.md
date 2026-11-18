# Phase 2B: Data Expansion Proposal (NOT executed; needs explicit approval)

This proposal follows from the Phase 2A/2B findings. Excitation is concentrated in AHU-level knobs, while zone-level actions barely vary under the baseline:
- the damper sits at its 0.1 floor in ≥ 75% of steps;
- reheat valves are at 0 in ≥ 75% of steps;
- HWS differential pressure is 0 in ≥ 50% of steps.

Cold-weather and large-building coverage are thin as well (severe cold appears only in fairbanks_1/2/8; b2 is the only 85-zone building). The Phase 1 collector and exploration profiles stay unchanged. New profiles would be added as **new, separately named** profiles writing to a **new** data root (`data/phase2c_*`). Every experiment below:
- uses 1–3 day windows, not full 30-day episodes;
- uses the seeded explorer's existing comfort guard (no AHU/SAT/pressure perturbation while an occupied zone deviates by more than `guard_k`);
- records commanded **and** effective actions.

Branch-based counterfactual collection (the Phase 2B branching tool, bit-exact with RNG restore) is preferred where the goal is paired action-response supervision.

| # | Experiment | Purpose | Target building / state | Intervention (native units, respecting clamps) | Expected information gain | Risk / cost |
|---|---|---|---|---|---|---|
| E1 | Zone damper excitation | Learn zone airflow → temperature response; the damper is at its 0.1 floor in most data | All 6 buildings; AHU on, occupied and unoccupied | Per-zone damper random telegraph in [0.1, 0.9] (effective floor 0.1), hold 15–60 min, independent per zone, ≤ 30% of zones at once | High: currently near-zero zone-level excitation; per-zone independence identifies zone-specific gains | Comfort loss in perturbed zones (bounded by the guard); fan energy ↑. ~1 core-h per building-day |
| E2 | Reheat valve excitation | Learn the reheat → zone temperature response (Phase 2A probe sign inconsistent; Phase 2B counterfactual gain ≪ 1) | Heating-active states, HWS hot (tank ≥ 330 K), AHU on | Valve steps of ±0.25/0.6 around the baseline command, hold 15–60 min, ≤ 30% of zones | High: directly targets the weakest learned response | Gas ↑ and overheating in occupied zones (guard); reheat with the AHU off is a no-op (log it) |
| E3 | AHU SAT perturbations (wider) | Cover SAT 285–305 K at all run states | Cooling and heating seasons, AHU on | SAT steps ±2/±5 K, hold 30–120 min | Medium: SAT is already the best-excited knob | Cooling energy ↑; comfort guard |
| E4 | HWS setpoint / pump dp perturbations | Tank dynamics and burner ramp-limit response; dp is mostly 0 in the data | Heating-active, cold scenarios | HWS setpoint 310–350 K steps (±5/15 K); dp 0–10 | Medium–high for tank/gas heads (the gas error is largest in cold) | Recovery shortfall if the tank is too cold before morning; gas ↑ |
| E5 | Large-building coverage | b2 is the only V ≥ 80 building in folds where it is held out; size extrapolation is failure-prone | Generated buildings (starter-kit generator) with V = 70–120, K = 7–12 | Baseline + E1/E2 profiles on 2 new generated buildings | High for LOBO generalization in size | Generator settings must match the evaluation distribution; ~2× sim cost per step |
| E6 | Severe-cold coverage | Deep-cold free-fall and recovery are mispredicted (Phase 2A) | fairbanks_* weather on all 6 buildings (only 3 buildings currently have a severe-cold scenario) | Baseline + E2/E4 profiles, 3-day windows centred on the coldest days | High for cold extrapolation folds (0, 1, 5) | Long recoveries → comfort penalties; gas ↑ |
| E7 | OOD action combinations | Joint off-distribution action combinations, e.g. high SAT + high recirculation + low HWS | All buildings, mixed states | Latin-hypercube joint perturbations of 2–3 families, hold 30 min, 10% of steps | Medium: interaction terms; tests OOD scores | Hardest to keep safe; tight guard; short holds |
| E8 | Heating recovery scenarios | Morning recovery from deep setback (largest reward impact) | Cold scenarios, 03:00–09:00 on workdays | Vary recovery start (precondition hours 0–4), AHU start time, HWS pre-charge | High for MPC-relevant decisions | Occupied comfort penalties if recovery starts too late |
| E9 | Paired counterfactual supervision (branching) | Direct supervision for action responses, no confounding by the baseline's closed-loop policy | Stratified branch points (as in Phase 2B) on all buildings | Workstream A arms at more branch points (≈ 50 per building) | Highest per sim-step for action-response learning; supports a paired-response loss | Branch cost (2 s copy for b2, 0.4–1 s smaller); ≈ 6 core-h per building for 50 points × 47 arms × 36 steps |

**Budget estimate (all experiments, 1 seed):** about 100–150 core-hours (about 4–6 h of wall time on 30 cores). No full 30-day episodes.

**Order of priority:** E2, E1 and E9 first (action response), then E6/E8 (cold), then E5 (size), then E4, E7 and E3.

**Acceptance check for collected data:**
- effective-vs-commanded divergence logged;
- comfort-guard trigger rate below 5%;
- reward reconstruction stays exact (≤ 1e-6);
- per-family effective excitation rate (the fraction of steps where the effective action differs from the baseline) ≥ 20% for each targeted family.
