# Baseline: "Hot Storage340 Nov14" analytical controller

> **Attribution — this is not our work.**
> `example-smart-building.ipynb` is a public Kaggle notebook written by **another competitor** in the
> [NeurIPS 2026 Smart Buildings](https://www.kaggle.com/competitions/neurips-2026-smart-buildings/) competition.
> We copied it unmodified as our reference baseline. All credit belongs to its authors (listed below).
> Before reusing or adapting any of the code in a submission or paper, check the license on each source notebook
> and keep the attribution chain intact.
>
> The exact URL of this particular notebook was not recorded when it was downloaded. **TODO: add the source link.**

## Provenance

The notebook says it is derived from these public works:

| Source | Author | What it contributes |
|---|---|---|
| [cpu-b-conditional-hotwater-20261022](https://www.kaggle.com/code/danielwuu/cpu-b-conditional-hotwater-20261022) | Daniel Wu | Base `TransferPolicy` (`adaptive_physics`): the per-zone thermal model, delayed-observation compensation, supply-air temperature/economizer search, fan reset and hot-water valve allocation. Embedded unmodified. |
| [cpu-a-alpha075-control-20261022](https://www.kaggle.com/code/danielwuu/cpu-a-alpha075-control-20261022) | Daniel Wu | Related analytical controller, cited for operating-point variants. |
| [Smart Buildings Facilities V42](https://www.kaggle.com/code/jazivxt/smart-buildings-facilities-v42?scriptVersionId=365466747) | jazivxt | V40–V46 extensions: guarded joint thermal control, online model identification (RLS) and a reserve based on prediction error. |
| [smooth-refined-recovery25-nov13](https://www.kaggle.com/code/penguin069/smart-buildings-smooth-refined-recovery25-nov13) | penguin069 | Direct parent (`smooth_refined_recovery25_nov13`), submission entry 58302087. |

Notebook metadata: experiment date 2026-11-14, run on Kaggle CPU with internet off.
Reference controller SHA256 `a4c89286…f5f51`; the notebook reports the full inference SHA256 as `87bb7e6a…846d`.

## Scores

| Policy | Public LB score (lower is better) |
|---|---:|
| Parent `smooth_refined_recovery25_nov13` | **0.00982** (confirmed by the parent's author) |
| This notebook, `WideHotStorageReservePolicy` (`hot_storage340_nov14`) | **Not measured.** The notebook itself says so. |

The saved output shows one local gateway rollout only (`building_train_1/atlanta_2`: mean reward −0.0021 over 8,641 steps, **805 s** wall-clock). That is a smoke test, not a leaderboard score.

**Our first job is to measure this policy ourselves on all 30 training scenarios, using the full horizon.**

## How it works

The policy uses only NumPy. It has **no neural network and no RL**. It adapts online within each episode, using only the metadata it is given plus current and past observations. Building IDs, station IDs and future weather are never read.

### Base controller (Daniel Wu, `TransferPolicy`)
1. **Parse the metadata.** It reads action indices for each AHU and zone, the zone→AHU assignment and the observation field names. It also estimates zone area and UA from `zone_map` as starting priors for the thermal model.
2. **Zone thermal model.** Each zone is modelled one step ahead as
   `T_next = (T + d + g·(G·T_supply + H)) / (1 + g·G)`,
   where `d` is free drift (a linear function of outside temperature, a slow mean, the AHU-group mean and comfort mode), `G` is air conductance and `H` is reheat power. A residual term fitted online corrects it.
3. **Delayed-observation compensation.** Zone temperatures arrive one step late. The model rolls the reading forward using the action that was actually applied.
4. **Target band.** The comfort band applies when the building is occupied. The eco band ramps in during the pre-conditioning window. When the building is empty there is a deep setback (282–307 K). A small margin is added, and recovery toward the band is rate-limited.
5. **Per-AHU search.** It tries a grid of supply-air temperatures × recirculation settings. For each candidate it inverts the model to get the airflow each zone needs, plus reheat, static pressure (fan reset) and damper positions. It then scores fan + chiller + gas cost + `comfort_cost` × predicted band violation. If running the AHU isn't cheaper than letting the zones drift, the AHU is switched **off**.
6. **Hot-water plant.** Pump differential pressure is set by the most demanding reheat valve, and each valve gets exactly the heat its zone needs. The supply-water setpoint is raised only after recent heating demand.

### Layers added on top (inheritance chain in cell 2, about 40 classes)
- **Model identification (V41/V42):** recursive least squares with a forgetting factor, a causal forecast error, and a comfort reserve sized from that error.
- **Water standby:** keeps hot water warm for 36–72 h after heat demand instead of reheating the tank every morning.
- **Setback tuning:** 274 K empty floor, one-step recovery, empty-day detection from zero occupancy (confirmed for 30 min), and a smaller margin for zones the model predicts well.
- **Finer search:** supply-air temperature in 0.5 K steps and recirculation in 0.1 steps. Water demand is sized with a 320 K floor, and pump pressure is kept just above the feasible minimum.
- **Storage ARX observer:** a gated, higher-order zone model used only to reconstruct the delayed temperature.
- **Heat guard and startup:** a forecast-based heating guard (0.25 K), and a smoothstep-shaped warm-up before occupancy.
- **Storage dispatch (Nov 14):** a hysteresis rule that charges the tank toward 350 K, then coasts on a 310 K command.
- **Selected class, `WideHotStorageReservePolicy`:** while coils are heating, it alternates between charging to 350 K and using stored heat, within a 340–350 K band.

## Notebook layout

| Cell | Contents |
|---|---|
| 0 | Description and attribution (markdown) |
| 1 | Locates the competition data folder under `/kaggle/input/...` and adds it to `sys.path` |
| 2 | The full policy source (~2,100 lines): Daniel Wu's original, then the chain of subclasses |
| 3 | Wraps `WideHotStorageReservePolicy` as `SubmissionPolicy(..., smart_buildings_inference_server.Policy)` and runs `SmartBuildingsInferenceServer(...).run(competition_data_folder=DATA_ROOT)` |

Cell 3 is also a working example of the **real Kaggle submission wiring**, which our own notebook should follow.

## Caveats before building on it

- **Reservoir accounting.** The author warns that how the simulator bills stored hot-water heat is "a material limitation when interpreting score gains as physical savings". The 340–350 K storage trick may be taking advantage of billing behaviour rather than truly saving energy. Check this in the simulator code before relying on it, and watch for organizers changing it.
- **Runtime.** A single scenario took about 805 s. Evaluation allows 10× the `bang_bang` runtime per rollout, so measure `bang_bang` locally and confirm this policy stays well within the limit. On Kaggle, the 9 h notebook limit across 30 scenarios is also tight.
- **Overfitting to the leaderboard.** Many small constants (margins, hold times, temperature thresholds) were tuned in sequence against public scores and the training scenarios. Some gains may not carry over to the held-out buildings and weather. Validate on newly generated buildings.
- **Complexity.** A deep multiple-inheritance chain with overridden `_band`, `_choose_ahu` and `_water_target` makes behaviour hard to follow. Profile and simplify before extending it.

## How we plan to use it

1. Get it running locally against the starter kit and score it on all 30 training scenarios, recording the comfort/energy/carbon breakdown and runtime for each scenario.
2. Use it as the score to beat and as a safe fallback submission.
3. Candidate improvements: residual RL or learned corrections on top of its actions, better tuning of `comfort_cost` and margins on generated buildings, and a faster implementation.
