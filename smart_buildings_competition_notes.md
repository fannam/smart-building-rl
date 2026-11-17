# Smart Buildings Competition — Research & Submission Notes

> **Last checked:** 2026-11-16  
> **Competition:** [Smart Buildings Competition](https://www.kaggle.com/competitions/neurips-2026-smart-buildings/)  
> **Official starter kit:** https://github.com/smart-buildings-2026/starter-kit  
> **Official challenge site:** https://smart-buildings-2026.github.io/competition/  
> **Competition description / technical paper:** https://papers.creativemachineslab.com/2026%20-%20The%20Smart%20Buildings%20Challenge%20Learning%20Foundation%20Models%20for%20Real-World%20Optimization%20at%20Scale.pdf

This document consolidates the competition overview, technical task, rules and constraints, data, scoring, policy interface, submission process, and a practical development plan. It is a working summary, not a replacement for accepting and reading the live Kaggle Rules page.

---

## 1. Overview

The goal is to build **one HVAC control policy** that works across commercial buildings it has never seen before.

The policy controls building HVAC equipment to balance:

- Occupant comfort.
- Energy cost (electricity and gas).
- Carbon emissions.

The core challenge is **cross-building and cross-weather generalization**. The same policy must be deployed unchanged on unseen buildings with different floor plans, numbers of floors, zones, AHU configurations, construction materials, and climates. At runtime it receives the new building's metadata and live sensor observations.

This is a simulation-based control / optimization task. Reinforcement learning is one possible approach, but the competition does not require a particular algorithm. Rule-based control, PID, MPC, offline/online RL, learned dynamics, or hybrid approaches can be explored as long as the final policy follows the interface and competition constraints.

### HVAC components

- **AHU (Air Handler Unit):** conditions and distributes air through ducts.
- **VAV box:** one terminal unit per zone; controls that zone's damper and reheat valve.
- **HWS (Hot Water System):** central boiler and pump supplying hot water.
- **Zone:** room/area with its own temperature sensor and occupancy schedule.

An AHU can serve multiple zones. Changing one AHU's settings affects all zones it serves, so the control problem is coupled.

---

## 2. Competition timeline and prizes

Dates below are the dates currently published on the Kaggle competition page. Deadlines are stated as 11:59 PM UTC unless the organizer updates the schedule.

| Milestone | Date |
|---|---:|
| Competition start | 2026-10-13 |
| Round-one paper deadline (NeurIPS poster) | 2026-11-13 |
| NeurIPS workshop in Paris | 2026-12-13 |
| Entry / rules acceptance deadline | 2027-01-03 |
| Team merger deadline | 2027-01-03 |
| Final Kaggle submission deadline | 2027-01-15 |

**As of 2026-11-16:** the round-one paper deadline has passed; the entry/rules acceptance and final submission deadlines are still upcoming.

### Prize pool

| Track | Total |
|---|---:|
| Main Kaggle leaderboard | $25,000 |
| Paper awards | $15,000 |
| **Total** | **$40,000** |

Main leaderboard prizes published:
- 1st: $5,750
- 2nd: $3,750
- 3rd: $3,000
- 4th–8th: $2,500 each

Paper awards have separate OpenReview submission requirements and deadlines. See the official pages for the complete award breakdown.

---

## 3. Rules and constraints

### 3.1 Kaggle notebook submission requirements

The Kaggle competition page states that submissions must be made through **Kaggle Notebooks**. For the Submit button to be enabled after committing a notebook, the published requirements are:

| Requirement | Published constraint |
|---|---|
| CPU notebook runtime | ≤ 9 hours |
| GPU notebook runtime | ≤ 9 hours |
| Internet access | Disabled |
| External data | Freely and publicly available external data is allowed |
| Pretrained models | Allowed, subject to the public/free data and offline-runtime requirements |

The submitted notebook is therefore the packaging / execution entry point. It must be self-contained at evaluation time: do not rely on a remote inference endpoint, external API, or downloading dependencies/model files from the internet during the evaluation run.

### 3.2 What is actually evaluated

The technical paper describes the evaluation gateway as owning the simulator and querying the submitted policy at each control step. The submission is a policy implementation plus the artifacts needed to instantiate it; it is **not** a notebook that submits a precomputed list of actions or a hand-authored list of evaluation scenarios.

The policy API is initialized once per building with building metadata and an action specification. Then `get_action(obs)` is called repeatedly to produce the next action.

Practical interpretation:
- Train and experiment before final evaluation.
- Package the trained weights / artifacts with the policy.
- The evaluation path should load the artifacts and produce actions from observations.
- No network access or remote inference.
- The evaluation gateway supplies the held-out scenarios; do not package or depend on a scenario list for the final evaluation.

**Training location:** the public competition materials specify the notebook submission and runtime limits, but do not require that all model training must happen inside the final inference run. A practical workflow is to train in Kaggle notebooks or elsewhere, then package the resulting weights and code into the Kaggle submission notebook. If training is done in a Kaggle notebook that is itself submitted, it must still fit the applicable notebook runtime. Re-check the live Kaggle Rules and submission template for any updated artifact restrictions.

### 3.3 Generalization / anti-cheating requirements

The defining requirement is one transferable policy, not a collection of building-specific controllers.

The technical paper describes these safeguards:
1. Held-out evaluation scenarios are not released before the competition closes.
2. Policies run in an isolated container with no network access.
3. Per-rollout wall-clock is limited to 10× the `bang_bang` baseline runtime to prevent unbounded online planning.
4. Submitted code archives are scanned for hard-coded building IDs or sensor signatures; matches can lead to disqualification.

Therefore:
- Do not hardcode evaluation building IDs, hidden building properties, or sensor signatures.
- Do not rely on remote lookups or remote inference.
- Use the runtime-provided metadata and observation stream to adapt the same policy to each building.
- Keep inference computationally efficient.
- A stateful policy may be useful, but any adaptation/planning must respect the evaluation time budget and official rules.

### 3.4 Rules page caveat

The Kaggle Rules tab is the binding source for participation terms. Its full text was not available in the public page extraction used for this summary. Before competing, open the Rules tab while logged into Kaggle and accept the rules before **2027-01-03**. Verify eligibility, team, intellectual-property, external-data, and disqualification clauses directly there; do not treat this summary as a substitute.

The technical paper states a team may contain up to four members, but the live Kaggle Rules page should be treated as authoritative if anything differs.

---

## 4. Data and simulator

### 4.1 What is provided

The official starter kit contains:
- The building/HVAC simulator.
- Six public training building definitions.
- Weather data / weather strips.
- Thirty ready-to-run training scenarios.
- Procedural building and scenario generators.
- Reference baseline controllers.
- A local evaluator and leaderboard-score script.
- Baseline return values for the public training scenarios.

The task uses a Gym-compatible simulated environment. The buildings and scenarios are procedural/simulated; the technical paper says the competition data contains no personal information, real-building telemetry, or human-subject data.

### 4.2 Public training set

The public training set is:

- 6 training buildings.
- 5 weather regimes/stations used in the shipped scenarios.
- 30 building × weather scenarios total.
- Each scenario is a 30-day episode.
- Each control step is 5 minutes.
- 8,640 steps per scenario.

The starter kit names the five stations used in the shipped scenario set:
- `fairbanks`
- `denver`
- `atlanta`
- `phoenix`
- `singapore`

A sixth station, `minneapolis`, is included as generator input but is not used by the shipped training scenarios.

Weather strips replay different 30-day windows of recorded 2023 hourly station data. The starter kit README notes that the shipped set pairs buildings and weather strips so that each building sees one strip per station and no strip is reused twice.

### 4.3 Building variation

The procedural generator varies:
- Number of floors (2–4).
- Footprint dimensions and perimeter / floor-plan shape.
- Room layout and zone count.
- Occupancy.
- Six building-envelope material properties.
- Number of AHUs and the zones served by each AHU.

For the six shipped training buildings:
- AHUs (`K`): 3–9.
- Zones (`V`): 28–85.
- Each AHU serves at most 10 zones.
- One VAV box per zone.
- One central hot-water system per building.

The evaluation uses **6 unseen buildings × 5 unseen weather stations = 30 held-out scenarios**. The held-out buildings are generated from the same generator/ranges, while the held-out weather strips/stations are not the strips shipped for training. The held-out configurations are kept confidential before the competition closes.

### 4.4 Observation space

Observation length changes with the building. It includes:

**Per zone**
- Air temperature.
- Current heating and cooling setpoints.
- Damper position.
- Reheat valve position.
- Airflow setpoint.

**Per AHU**
- Supply-air temperature.
- Static pressure.
- Airflow (measured and setpoint).
- Recirculation.
- Cooling-request count.
- Fan run status.

**Building / plant / context**
- Hot-water supply temperature (measured and setpoint).
- Building-level occupancy count.
- Comfort-mode flags (now and imminently).
- Outside-air temperature.
- Time of day and day of week (cyclically encoded).

The policy also receives building metadata, including action indices, observation field names, zone-to-AHU assignment, floor plan, schedule, and normalization constants.

### 4.5 Action space

The action is hybrid and variable-dimensional.

For a building with:
- `K` = number of AHUs
- `V` = number of zones

The action dictionary contains:

```python
{
    "discrete_action":   np.int32[K + 1],
    "continuous_action": np.float32[3*K + 2 + 2*V],
}
```

Discrete actions:
- One on/off bit for each AHU.
- One on/off bit for the central HWS.
- Values are in `{0, 1}`.

Continuous actions:
- 3 setpoints per AHU: supply-air temperature, static pressure, recirculation.
- 2 setpoints for HWS: supply-water temperature and pump differential pressure.
- 2 controls per zone/VAV: damper position and reheat valve position.
- Normalized to `[-1, 1]`.

**Important implementation details from the starter kit:**
- Continuous action entries are interleaved; do not assume a simplistic fixed offset layout.
- Read the index mappings from `building_meta_data`, including `temp_indices`, `pressure_indices`, `recirculation_indices`, `water_indices`, and `zone_action_indices`.
- Do not assume the same number/order of AHUs or zones for every building.
- Zero action is the midpoint of normalized ranges; it is not equivalent to turning equipment off.
- The simulator enforces ventilation floors: minimum static pressure for a running AHU, maximum 70% recirculation, and minimum 10% damper opening.

---

## 5. Reward and leaderboard metric

### 5.1 Per-step reward

At each 5-minute step:

\[
r_t = 0.6p_t - 0.2c_t - 0.2e_t
\]

Where:
- `p_t`: occupant comfort / productivity term.
- `c_t`: energy cost (electricity and gas).
- `e_t`: carbon emissions.

All terms are penalties or zero:
- Comfort is zero when occupied zones are within their comfort band.
- Comfort penalty becomes more negative as temperatures leave the comfort band: linear up to 5 K outside the band, then quadratic and unbounded.
- Energy cost and carbon emissions are non-negative costs subtracted from reward.
- Comfort is weighted by occupancy, not floor area.

Thus `r_t <= 0`. Zero is the ideal but practically unattainable target: perfect comfort with no energy or carbon cost.

### 5.2 Episode and final score

Each scenario:
- 30 days.
- 5-minute timestep.
- 8,640 steps.
- Every step is scored, including the first; there is no warm-up period.

For scenario `s`:

\[
\bar r_s = \frac{1}{8640}\sum_{t=1}^{8640} r_{s,t}
\]

Leaderboard score:

\[
\text{Score} = \frac{1}{30}\sum_{s=1}^{30}|\bar r_s - 0|
             = -\frac{1}{30}\sum_{s=1}^{30}\bar r_s
\]

- Lower is better.
- All 30 held-out building/weather scenarios receive equal weight.
- The simulator is deterministic; one rollout per scenario is sufficient.
- The organizer re-runs top submissions end-to-end before final results.

**Do not compare scores from different episode lengths.** The local scorer expects the full 8,640-step horizon; shorter runs are only smoke tests and are not comparable.

### 5.3 Baselines

The starter kit includes:
- `bang_bang`
- `fixed_setpoint`
- `pid`
- `pid_water`
- `all_off` (reference only; not included among the four scored baseline comparisons)

Baseline returns on the public training scenarios are included for reference. They do not enter the leaderboard metric.

---

## 6. Policy API and submission

### 6.1 Local starter-kit interface

The starter kit README describes a plain Python class:

```python
class TransferPolicy:
    def __init__(self, building_meta_data, action_spec):
        ...

    def get_action(self, obs) -> dict:
        return {
            "continuous_action": ...,
            "discrete_action": ...,
        }
```

The class name `TransferPolicy` is already registered in the starter kit's `POLICY_MAP`. If changing the class name or adding another policy, register it in `POLICY_MAP`.

### 6.2 Competition inference-server interface

The technical paper gives the competition gateway interface in this form:

```python
from kaggle_evaluation import smart_buildings_inference_server

class MyPolicy(smart_buildings_inference_server.Policy):
    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        ...

    def get_action(self, obs) -> dict:
        ...

smart_buildings_inference_server.SmartBuildingsInferenceServer(
    MyPolicy
).run()
```

The constructor is called once per building. It receives the building metadata and action specification. `get_action(obs)` is then called for each control step and must return an action dictionary matching the provided action specification exactly.

Use the current Kaggle submission template as the final authority for import names and exact packaging details; the local starter-kit `TransferPolicy` interface and the Kaggle gateway wrapper are related but should not be assumed to be byte-for-byte identical.

### 6.3 What to package

The technical paper describes an entry containing:
1. Python module implementing the policy interface.
2. Model weights or other artifacts required to instantiate the policy.
3. Reproducibility bundle: code archive, seed list, dependency manifest.

A practical artifact layout could be:

```text
submission/
├── policy.py
├── model.pt
├── config.json
├── requirements.txt
└── README.md
```

The actual Kaggle notebook must follow the official submission template and load these artifacts without internet access.

### 6.4 Training versus inference

Recommended separation:

**Development / training**
- Can be performed in Kaggle notebooks or another permitted compute environment.
- Train against the public simulator and generated training buildings/scenarios.
- Save checkpoints and record seeds/configuration.
- Validate transfer to buildings not used for fitting.

**Final evaluation**
- Submit through a Kaggle Notebook.
- Load the policy code and model artifacts.
- The evaluation gateway supplies unseen building metadata and observations.
- The policy returns an action at each step.
- No remote inference or network dependency.
- Keep inference within the runtime constraints.

If choosing to train on Kaggle as well, use separate development/training notebooks and keep the final submission notebook focused on loading the selected checkpoint and running the inference interface. This is a recommended engineering organization, not a claim that Kaggle requires a particular notebook count.

---

## 7. Starter-kit repository structure

Repository: https://github.com/smart-buildings-2026/starter-kit

Important paths described in the README:

```text
smart_buildings/
  smart_control/          # simulator, environment, reward, models, utilities
  baselines/              # baseline policies
  buildings/              # six public training building definitions
  weather/                # weather presets
  training_scenarios/     # 30 building × weather scenarios
  run_floor_generator.py  # sample new building configurations
  run_scenario_generator.py
  run_environment.py      # inspect observation/action spaces and smoke-test
  evaluate.py             # evaluate a policy on scenarios
score_leaderboard.py      # calculate local leaderboard-style score
baseline_returns.json     # reference returns
check_scales.py           # validate reward scaling
```

### Quickstart commands

```bash
git clone https://github.com/smart-buildings-2026/starter-kit.git
cd starter-kit
pip install -e .

# Inspect one scenario / print spaces / short rollout
python -m smart_buildings.run_environment \
  --building building_train_1 \
  --weather fairbanks_1

# Evaluate starter policy on all 30 scenarios
python -m smart_buildings.evaluate
```

The shipped `transfer` starter policy initially returns zero actions; it is a scaffold to edit, not a strong learned baseline.

### Evaluate one scenario or all scenarios

```bash
# One complete 30-day scenario
python -m smart_buildings.evaluate \
  --policies transfer \
  --buildings building_train_1 \
  --weathers fairbanks_1

# All 30 training scenarios
python -m smart_buildings.evaluate --policies transfer

# One-day smoke test only (not score-comparable)
python -m smart_buildings.evaluate \
  --policies transfer \
  --buildings building_train_1 \
  --weathers fairbanks_1 \
  --steps 288
```

### Generate additional training buildings / scenarios

```bash
# Generate ten additional building configurations
python -m smart_buildings.run_floor_generator \
  --start-seed 500 \
  --count 10

# Generate scenarios from selected buildings and weather strips
python -m smart_buildings.run_scenario_generator \
  --buildings building_0500 building_0501 \
  --weathers fairbanks_1 phoenix_6

# Inspect a generated scenario
python -m smart_buildings.run_environment \
  --building building_0500 \
  --weather fairbanks_1
```

The generator can sample additional buildings during development. Generated buildings receive material properties sampled by the generator. Note that generated buildings can be larger and more expensive to simulate than the six shipped buildings; the number of AHUs may increase with the number of zones.

---

## 8. Suggested development strategy

Given the competition's variable-size action/observation spaces and cross-building generalization requirement, a sensible progression is:

1. **Understand the simulator and interface**
   - Inspect `transfer_policy.py`, `evaluate.py`, and the Kaggle inference template.
   - Log observation fields, metadata, action bounds, and reward components.
   - Verify action indexing for multiple buildings with different AHU/zone counts.

2. **Establish baselines**
   - Run `bang_bang`, `fixed_setpoint`, `pid`, and `pid_water`.
   - Record per-scenario scores and runtime.
   - Use the full 8,640-step horizon for comparisons.

3. **Build a strong non-learning controller**
   - Improve occupancy-aware schedules and temperature control.
   - Use PID / rule-based control as a reliable baseline.
   - Check comfort, energy, and carbon separately.

4. **Benchmark simulator throughput**
   - Measure steps per second for small and large buildings.
   - Estimate the cost of 8,640-step episodes × number of buildings × number of episodes.
   - Identify whether CPU simulation or neural-network updates are the bottleneck.

5. **Train for transfer**
   - Use generated buildings and varied weather scenarios.
   - Avoid fitting only the six shipped buildings.
   - Use leave-building-out / leave-weather-out validation.
   - Consider shared per-zone/per-AHU encoders, graph/set architectures, or permutation-aware representations because dimensions and mappings vary.

6. **Keep final inference lean**
   - Export a compact checkpoint.
   - Avoid training or expensive optimization in the final evaluation path unless explicitly permitted and within the time budget.
   - Load all dependencies and weights offline.
   - Test the final notebook with internet disabled.

### Kaggle-only versus hybrid workflow

**Option A — Full Kaggle**
- Generate scenarios, train, validate, and package on Kaggle.
- Advantage: one environment and easier reproducibility.
- Risk: notebook runtime, GPU availability/quota, CPU simulation throughput, and storage.

**Option B — Hybrid**
- Use Kaggle/local CPU for simulator and data generation.
- Train on a local or cloud GPU when useful.
- Export weights and dependencies.
- Run final inference through the Kaggle notebook.
- Advantage: more flexible compute.
- Risk: ensure the exported artifact and dependencies work in the offline Kaggle environment.

Recommendation: start with a simulator throughput benchmark and a full baseline evaluation. Then decide whether Kaggle-only training fits the compute budget. The simulation workload may be CPU-bound, so GPU availability alone does not determine feasibility.

---

## 9. Important distinctions / items to verify

- The Kaggle **Data** tab shows dataset metadata/license `CC BY-NC-SA 4.0`; the starter-kit repository itself states Apache License 2.0. These refer to different distributed assets; check the license attached to each asset before reusing it outside the competition.
- The Kaggle Rules tab is the binding source for legal/participation terms. This summary does not reproduce its full text.
- The official Kaggle submission template is authoritative for the exact inference-server imports and notebook packaging.
- The public competition page gives 9-hour CPU/GPU notebook limits and disabled internet. The technical paper additionally describes a per-rollout wall-clock limit of 10× the `bang_bang` baseline.
- Do not assume Kaggle's currently advertised GPU model, weekly quota, RAM, or disk size is fixed. Check the actual notebook session and current Kaggle documentation/account quota before planning training.
- Do not assume online learning is categorically forbidden based only on the notebook rules. The documented gateway repeatedly calls `get_action`; stateful behavior is technically possible, but it must comply with all official rules, the no-network sandbox, and the per-rollout runtime bound. Verify the current Rules page before relying on online adaptation.

---

## 10. Official references

1. **Kaggle competition overview, evaluation, timeline, prizes, and code requirements**  
   https://www.kaggle.com/competitions/neurips-2026-smart-buildings/overview

2. **Kaggle competition data tab and dataset license metadata**  
   https://www.kaggle.com/competitions/neurips-2026-smart-buildings/data

3. **Kaggle Rules tab**  
   https://www.kaggle.com/competitions/neurips-2026-smart-buildings/rules

4. **Official starter-kit repository and README**  
   https://github.com/smart-buildings-2026/starter-kit

5. **Official challenge website**  
   https://smart-buildings-2026.github.io/competition/

6. **Technical paper: The Smart Buildings Challenge — Learning Foundation Models for Real-World Optimization at Scale**  
   https://papers.creativemachineslab.com/2026%20-%20The%20Smart%20Buildings%20Challenge%20Learning%20Foundation%20Models%20for%20Real-World%20Optimization%20at%20Scale.pdf

7. **Kaggle competition submission documentation**  
   https://www.kaggle.com/docs/competitions

---

## 11. Short summary

- **Task:** one HVAC policy that generalizes to unseen buildings and weather.
- **Training data:** 6 public buildings × 5 weather regimes = 30 scenarios.
- **Evaluation:** 6 unseen buildings × 5 unseen weather stations = 30 scenarios.
- **Episode:** 30 days, 5-minute timestep, 8,640 scored steps.
- **Reward:** `0.6 × comfort − 0.2 × energy_cost − 0.2 × carbon`.
- **Metric:** mean absolute per-step reward against zero; lower is better.
- **Action:** hybrid, variable-sized discrete + continuous control.
- **Submission:** Kaggle Notebook, CPU/GPU runtime ≤ 9 hours, internet disabled.
- **Artifacts:** policy code, model weights/artifacts, reproducibility information.
- **Training:** can be developed separately; final evaluation runs the policy through the Kaggle inference interface. Verify the current Rules page for exact restrictions.
- **Final deadline:** 2027-01-15; accept rules by 2027-01-03.
