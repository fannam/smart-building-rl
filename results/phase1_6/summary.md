# Phase 1.6 — Leave-one-building-out split summary (generated)

Data: `data/phase1_sample` · profiles: baseline, moderate · 60 episodes / 120960 transitions · generated 2026-11-18 12:16

All numbers are **descriptive dataset statistics**. No model was trained; model metrics (one-step / multi-step zone-temperature RMSE; tank-temperature RMSE; energy (E) and gas (G) head error; reward-reconstruction error from predicted states; closed-loop score) are not reported.

## Folds

| fold | held-out building | K | V | train buildings | train eps / transitions | test eps / transitions | test min OAT °C | test OAT ⊂ train OAT | test K ⊂ train K range | severe-cold test scenarios | leakage checks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | building_train_1 | [4] | [39] | 5 | 50 / 100800 | 10 / 20160 | -39.4 | False | True | fairbanks_1 | pass |
| 1 | building_train_2 | [9] | [85] | 5 | 50 / 100800 | 10 / 20160 | -25.6 | True | False | fairbanks_2 | pass |
| 2 | building_train_4 | [6] | [54] | 5 | 50 / 100800 | 10 / 20160 | -5.6 | True | True | none | pass |
| 3 | building_train_5 | [8] | [79] | 5 | 50 / 100800 | 10 / 20160 | -2.8 | True | True | none | pass |
| 4 | building_train_7 | [4] | [36] | 5 | 50 / 100800 | 10 / 20160 | 1.6 | True | True | none | pass |
| 5 | building_train_8 | [3] | [30] | 5 | 50 / 100800 | 10 / 20160 | -16.7 | True | False | fairbanks_8 | pass |

## Held-out building profile (test side of each fold)

| building | K | V | min OAT °C | mean OAT °C | HDH (18 °C) per week | gas kWh/AHU/day | cooling kWh/AHU/day | occupied frac | comfort >0 K | comfort >1 K | max dev K | run frac | P_cmd<200 frac (running) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| building_train_1 | [4] | [39] | -39.4 | 13.1 | 1839 | 195.77 | 39.11 | 0.37 | 0.0259 | 0.0015 | 4.53 | 0.46 | 0.106 |
| building_train_2 | [9] | [85] | -25.6 | 14.0 | 1626 | 79.79 | 27.55 | 0.37 | 0.0232 | 0.0011 | 7.89 | 0.46 | 0.111 |
| building_train_4 | [6] | [54] | -5.6 | 18.4 | 664 | 16.82 | 25.30 | 0.37 | 0.0208 | 0.0000 | 0.94 | 0.43 | 0.117 |
| building_train_5 | [8] | [79] | -2.8 | 19.8 | 506 | 17.84 | 21.83 | 0.37 | 0.0239 | 0.0000 | 1.52 | 0.42 | 0.108 |
| building_train_7 | [4] | [36] | 1.6 | 19.0 | 576 | 42.47 | 21.85 | 0.35 | 0.0233 | 0.0000 | 0.78 | 0.41 | 0.121 |
| building_train_8 | [3] | [30] | -16.7 | 18.2 | 1056 | 72.95 | 35.64 | 0.39 | 0.0195 | 0.0002 | 4.60 | 0.47 | 0.109 |

## By Weather station

| group | episodes | transitions | buildings | min OAT °C | mean OAT °C | gas kWh/AHU/day | cooling kWh/AHU/day | comfort >1 K | max dev K | SAT p50 K | P_eff p95 Pa | HWS p50 K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| atlanta | 12 | 24192 | 6 | -1.7 | 18.3 | 25.23 | 19.56 | 0.0000 | 0.92 | 291.5 | 414 | 333.3 |
| denver | 12 | 24192 | 6 | -5.6 | 13.1 | 41.77 | 10.14 | 0.0001 | 4.50 | 290.9 | 451 | 338.0 |
| fairbanks | 12 | 24192 | 6 | -39.4 | -1.1 | 283.83 | 2.12 | 0.0022 | 7.89 | 289.0 | 406 | 344.2 |
| phoenix | 12 | 24192 | 6 | 1.1 | 27.0 | 3.85 | 56.27 | 0.0000 | 1.51 | 288.2 | 415 | 321.2 |
| singapore | 12 | 24192 | 6 | 23.6 | 28.3 | 0.00 | 54.64 | 0.0000 | 0.67 | 285.3 | 422 | 310.0 |

## By Climate band (mean OAT: cold <5 °C, mild 5–20, hot ≥20)

| group | episodes | transitions | buildings | min OAT °C | mean OAT °C | gas kWh/AHU/day | cooling kWh/AHU/day | comfort >1 K | max dev K | SAT p50 K | P_eff p95 Pa | HWS p50 K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold | 8 | 16128 | 3 | -39.4 | -11.0 | 419.50 | 0.36 | 0.0035 | 7.89 | 286.4 | 444 | 343.6 |
| hot | 28 | 56448 | 6 | 12.8 | 28.6 | 0.29 | 57.64 | 0.0000 | 1.03 | 285.8 | 420 | 312.8 |
| mild | 24 | 48384 | 6 | -5.6 | 13.0 | 37.17 | 3.99 | 0.0000 | 1.52 | 293.5 | 416 | 343.9 |

## By AHU count K

| group | episodes | transitions | buildings | min OAT °C | mean OAT °C | gas kWh/AHU/day | cooling kWh/AHU/day | comfort >1 K | max dev K | SAT p50 K | P_eff p95 Pa | HWS p50 K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 10 | 20160 | 1 | -16.7 | 18.2 | 72.95 | 35.64 | 0.0002 | 4.60 | 287.3 | 418 | 323.7 |
| 4 | 20 | 40320 | 2 | -39.4 | 16.1 | 119.12 | 30.48 | 0.0007 | 4.53 | 289.4 | 415 | 330.9 |
| 6 | 10 | 20160 | 1 | -5.6 | 18.4 | 16.82 | 25.30 | 0.0000 | 0.94 | 290.0 | 415 | 329.3 |
| 8 | 10 | 20160 | 1 | -2.8 | 19.8 | 17.84 | 21.83 | 0.0000 | 1.52 | 290.1 | 423 | 330.6 |
| 9 | 10 | 20160 | 1 | -25.6 | 14.0 | 79.79 | 27.55 | 0.0011 | 7.89 | 287.6 | 445 | 330.6 |

## By Zone count V

| group | episodes | transitions | buildings | min OAT °C | mean OAT °C | gas kWh/AHU/day | cooling kWh/AHU/day | comfort >1 K | max dev K | SAT p50 K | P_eff p95 Pa | HWS p50 K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 | 10 | 20160 | 1 | -16.7 | 18.2 | 72.95 | 35.64 | 0.0002 | 4.60 | 287.3 | 418 | 323.7 |
| 36 | 10 | 20160 | 1 | 1.6 | 19.0 | 42.47 | 21.85 | 0.0000 | 0.78 | 289.1 | 411 | 331.2 |
| 39 | 10 | 20160 | 1 | -39.4 | 13.1 | 195.77 | 39.11 | 0.0015 | 4.53 | 289.8 | 419 | 330.7 |
| 54 | 10 | 20160 | 1 | -5.6 | 18.4 | 16.82 | 25.30 | 0.0000 | 0.94 | 290.0 | 415 | 329.3 |
| 79 | 10 | 20160 | 1 | -2.8 | 19.8 | 17.84 | 21.83 | 0.0000 | 1.52 | 290.1 | 423 | 330.6 |
| 85 | 10 | 20160 | 1 | -25.6 | 14.0 | 79.79 | 27.55 | 0.0011 | 7.89 | 287.6 | 445 | 330.6 |

## By Profile

| group | episodes | transitions | buildings | min OAT °C | mean OAT °C | gas kWh/AHU/day | cooling kWh/AHU/day | comfort >1 K | max dev K | SAT p50 K | P_eff p95 Pa | HWS p50 K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 30 | 60480 | 6 | -39.4 | 17.1 | 64.71 | 27.49 | 0.0003 | 7.89 | 289.1 | 212 | 329.1 |
| moderate | 30 | 60480 | 6 | -39.4 | 17.1 | 77.17 | 29.60 | 0.0006 | 7.50 | 288.9 | 632 | 329.5 |

## Coldest scenarios

| scenario | building | min OAT °C | mean OAT °C | start month (privileged) | gas kWh/AHU/day (baseline) | comfort >1 K (baseline / moderate) |
|---|---|---|---|---|---|---|
| fairbanks_1 | building_train_1 | -39.4 | -26.3 | 1 | 832.09 | 0.0054 / 0.0092 |
| fairbanks_2 | building_train_2 | -25.6 | -13.4 | 2 | 281.22 | 0.0035 / 0.0061 |
| fairbanks_8 | building_train_8 | -16.7 | -7.4 | 10 | 310.19 | 0.0000 / 0.0023 |
| denver_2 | building_train_4 | -5.6 | 7.1 | 2 | 58.47 | 0.0000 / 0.0000 |
| denver_8 | building_train_2 | -4.4 | 3.0 | 11 | 93.70 | 0.0000 / 0.0013 |
| denver_3 | building_train_5 | -2.8 | 10.0 | 4 | 64.00 | 0.0001 / 0.0001 |
| atlanta_1 | building_train_8 | -1.7 | 12.4 | 1 | 36.46 | 0.0000 / 0.0000 |
| phoenix_1 | building_train_5 | 1.1 | 13.1 | 1 | 18.22 | 0.0000 / 0.0000 |

Severe cold = scenario minimum OAT ≤ -10.0 °C.
