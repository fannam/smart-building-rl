# Phase 2A — dynamics model results (generated)

Zone temperature in °C error (= K), HWS in K, energy in kW, reward in reward units. Deployable outside air (held at o_t0) unless marked oracle. h = rollout step (h = 1 is teacher-forced).

## Zone temperature MAE (°C) by horizon h

| fold / held-out | K | V | model | 1 | 3 | 6 | 12 | 36 |
|---|---|---|---|---|---|---|---|---|
| 0 building_train_1 | 4 | 39 | persistence | 0.072 | 0.099 | 0.150 | 0.264 | 0.661 |
| 0 building_train_1 | 4 | 39 | linear AR | 0.074 | 0.109 | 0.164 | 0.284 | 0.659 |
| 0 building_train_1 | 4 | 39 | GRU | 0.056 | 0.074 | 0.114 | 0.168 | 0.327 |
| 0 building_train_1 | 4 | 39 | GRU (oracle OAT) | 0.056 | 0.074 | 0.115 | 0.171 | 0.345 |
| 1 building_train_2 | 9 | 85 | persistence | 0.090 | 0.118 | 0.184 | 0.329 | 0.823 |
| 1 building_train_2 | 9 | 85 | linear AR | 0.089 | 0.120 | 0.186 | 0.323 | 0.758 |
| 1 building_train_2 | 9 | 85 | GRU | 0.078 | 0.093 | 0.151 | 0.243 | 0.518 |
| 1 building_train_2 | 9 | 85 | GRU (oracle OAT) | 0.078 | 0.093 | 0.151 | 0.243 | 0.519 |
| 2 building_train_4 | 6 | 54 | persistence | 0.042 | 0.064 | 0.102 | 0.179 | 0.438 |
| 2 building_train_4 | 6 | 54 | linear AR | 0.043 | 0.066 | 0.102 | 0.175 | 0.419 |
| 2 building_train_4 | 6 | 54 | GRU | 0.024 | 0.034 | 0.051 | 0.077 | 0.172 |
| 2 building_train_4 | 6 | 54 | GRU (oracle OAT) | 0.024 | 0.034 | 0.051 | 0.077 | 0.172 |
| 3 building_train_5 | 8 | 79 | persistence | 0.043 | 0.065 | 0.101 | 0.175 | 0.418 |
| 3 building_train_5 | 8 | 79 | linear AR | 0.042 | 0.065 | 0.101 | 0.171 | 0.404 |
| 3 building_train_5 | 8 | 79 | GRU | 0.028 | 0.042 | 0.071 | 0.110 | 0.227 |
| 3 building_train_5 | 8 | 79 | GRU (oracle OAT) | 0.028 | 0.043 | 0.072 | 0.114 | 0.252 |
| 4 building_train_7 | 4 | 36 | persistence | 0.063 | 0.094 | 0.149 | 0.265 | 0.660 |
| 4 building_train_7 | 4 | 36 | linear AR | 0.062 | 0.097 | 0.154 | 0.271 | 0.660 |
| 4 building_train_7 | 4 | 36 | GRU | 0.038 | 0.049 | 0.081 | 0.128 | 0.297 |
| 4 building_train_7 | 4 | 36 | GRU (oracle OAT) | 0.038 | 0.049 | 0.080 | 0.125 | 0.266 |
| 5 building_train_8 | 3 | 30 | persistence | 0.083 | 0.111 | 0.173 | 0.313 | 0.807 |
| 5 building_train_8 | 3 | 30 | linear AR | 0.079 | 0.108 | 0.168 | 0.297 | 0.706 |
| 5 building_train_8 | 3 | 30 | GRU | 0.046 | 0.057 | 0.089 | 0.138 | 0.304 |
| 5 building_train_8 | 3 | 30 | GRU (oracle OAT) | 0.046 | 0.057 | 0.090 | 0.142 | 0.325 |

## Zone temperature bias (°C) by horizon h

| fold / held-out | K | V | model | 1 | 3 | 6 | 12 | 36 |
|---|---|---|---|---|---|---|---|---|
| 0 building_train_1 | 4 | 39 | persistence | -0.000 | 0.002 | 0.001 | 0.003 | 0.009 |
| 0 building_train_1 | 4 | 39 | linear AR | 0.001 | 0.001 | 0.001 | 0.003 | 0.019 |
| 0 building_train_1 | 4 | 39 | GRU | 0.010 | 0.016 | 0.020 | 0.022 | 0.004 |
| 0 building_train_1 | 4 | 39 | GRU (oracle OAT) | 0.010 | 0.016 | 0.020 | 0.021 | -0.010 |
| 1 building_train_2 | 9 | 85 | persistence | -0.005 | 0.001 | -0.003 | -0.006 | -0.020 |
| 1 building_train_2 | 9 | 85 | linear AR | -0.001 | 0.002 | 0.001 | 0.002 | 0.009 |
| 1 building_train_2 | 9 | 85 | GRU | -0.017 | -0.022 | -0.046 | -0.080 | -0.195 |
| 1 building_train_2 | 9 | 85 | GRU (oracle OAT) | -0.017 | -0.022 | -0.046 | -0.080 | -0.197 |
| 2 building_train_4 | 6 | 54 | persistence | -0.003 | -0.003 | -0.007 | -0.013 | -0.038 |
| 2 building_train_4 | 6 | 54 | linear AR | -0.009 | -0.018 | -0.032 | -0.059 | -0.160 |
| 2 building_train_4 | 6 | 54 | GRU | -0.002 | -0.004 | -0.009 | -0.021 | -0.070 |
| 2 building_train_4 | 6 | 54 | GRU (oracle OAT) | -0.002 | -0.004 | -0.009 | -0.022 | -0.077 |
| 3 building_train_5 | 8 | 79 | persistence | -0.002 | -0.002 | -0.004 | -0.008 | -0.022 |
| 3 building_train_5 | 8 | 79 | linear AR | 0.004 | 0.007 | 0.012 | 0.021 | 0.055 |
| 3 building_train_5 | 8 | 79 | GRU | 0.001 | 0.003 | 0.006 | 0.010 | 0.024 |
| 3 building_train_5 | 8 | 79 | GRU (oracle OAT) | 0.001 | 0.003 | 0.006 | 0.008 | 0.008 |
| 4 building_train_7 | 4 | 36 | persistence | -0.002 | -0.000 | -0.003 | -0.006 | -0.011 |
| 4 building_train_7 | 4 | 36 | linear AR | 0.010 | 0.020 | 0.032 | 0.058 | 0.158 |
| 4 building_train_7 | 4 | 36 | GRU | 0.004 | 0.011 | 0.021 | 0.034 | 0.087 |
| 4 building_train_7 | 4 | 36 | GRU (oracle OAT) | 0.004 | 0.011 | 0.021 | 0.033 | 0.076 |
| 5 building_train_8 | 3 | 30 | persistence | 0.000 | 0.003 | 0.000 | 0.001 | 0.005 |
| 5 building_train_8 | 3 | 30 | linear AR | 0.002 | 0.001 | -0.003 | -0.003 | 0.004 |
| 5 building_train_8 | 3 | 30 | GRU | 0.005 | 0.010 | 0.013 | 0.020 | 0.048 |
| 5 building_train_8 | 3 | 30 | GRU (oracle OAT) | 0.005 | 0.010 | 0.013 | 0.020 | 0.034 |

## HWS / tank temperature MAE (K)

| fold / held-out | K | V | model | 1 | 3 | 6 | 12 | 36 |
|---|---|---|---|---|---|---|---|---|
| 0 building_train_1 | 4 | 39 | persistence | 0.31 | 0.81 | 0.90 | 0.98 | 1.45 |
| 0 building_train_1 | 4 | 39 | linear AR | 0.50 | 1.25 | 1.69 | 2.48 | 3.78 |
| 0 building_train_1 | 4 | 39 | GRU | 0.17 | 0.28 | 0.34 | 0.43 | 0.58 |
| 0 building_train_1 | 4 | 39 | GRU (oracle OAT) | 0.17 | 0.28 | 0.34 | 0.43 | 0.56 |
| 1 building_train_2 | 9 | 85 | persistence | 0.56 | 1.12 | 1.20 | 1.17 | 1.60 |
| 1 building_train_2 | 9 | 85 | linear AR | 0.61 | 1.36 | 1.98 | 2.75 | 4.01 |
| 1 building_train_2 | 9 | 85 | GRU | 0.45 | 0.84 | 0.94 | 1.00 | 1.48 |
| 1 building_train_2 | 9 | 85 | GRU (oracle OAT) | 0.45 | 0.84 | 0.94 | 1.00 | 1.47 |
| 2 building_train_4 | 6 | 54 | persistence | 0.16 | 0.38 | 0.50 | 0.62 | 1.01 |
| 2 building_train_4 | 6 | 54 | linear AR | 0.41 | 0.97 | 1.57 | 2.36 | 3.41 |
| 2 building_train_4 | 6 | 54 | GRU | 0.09 | 0.18 | 0.26 | 0.38 | 0.68 |
| 2 building_train_4 | 6 | 54 | GRU (oracle OAT) | 0.09 | 0.18 | 0.26 | 0.39 | 0.71 |
| 3 building_train_5 | 8 | 79 | persistence | 0.21 | 0.46 | 0.55 | 0.58 | 0.91 |
| 3 building_train_5 | 8 | 79 | linear AR | 0.39 | 0.91 | 1.44 | 2.14 | 3.12 |
| 3 building_train_5 | 8 | 79 | GRU | 0.08 | 0.13 | 0.19 | 0.23 | 0.32 |
| 3 building_train_5 | 8 | 79 | GRU (oracle OAT) | 0.08 | 0.13 | 0.19 | 0.23 | 0.35 |
| 4 building_train_7 | 4 | 36 | persistence | 0.29 | 0.64 | 0.77 | 0.85 | 1.11 |
| 4 building_train_7 | 4 | 36 | linear AR | 0.42 | 1.03 | 1.63 | 2.36 | 3.45 |
| 4 building_train_7 | 4 | 36 | GRU | 0.09 | 0.16 | 0.22 | 0.32 | 0.57 |
| 4 building_train_7 | 4 | 36 | GRU (oracle OAT) | 0.09 | 0.16 | 0.22 | 0.32 | 0.56 |
| 5 building_train_8 | 3 | 30 | persistence | 0.29 | 0.62 | 0.70 | 0.75 | 1.03 |
| 5 building_train_8 | 3 | 30 | linear AR | 0.42 | 0.98 | 1.46 | 2.14 | 3.11 |
| 5 building_train_8 | 3 | 30 | GRU | 0.15 | 0.28 | 0.42 | 0.57 | 0.86 |
| 5 building_train_8 | 3 | 30 | GRU (oracle OAT) | 0.15 | 0.28 | 0.42 | 0.57 | 0.86 |

## Total electricity MAE (kW) [relative MAE]

| fold / held-out | K | V | model | 1 | 3 | 6 | 12 | 36 |
|---|---|---|---|---|---|---|---|---|
| 0 building_train_1 | 4 | 39 | persistence | 6.2 [0.75] | 6.0 [0.76] | 6.0 [0.75] | 6.0 [0.75] | 6.0 [0.75] |
| 0 building_train_1 | 4 | 39 | linear AR | 3.2 [0.39] | 3.1 [0.38] | 3.1 [0.38] | 3.1 [0.39] | 3.2 [0.39] |
| 0 building_train_1 | 4 | 39 | GRU | 0.6 [0.07] | 0.6 [0.07] | 0.6 [0.08] | 0.6 [0.08] | 0.8 [0.10] |
| 0 building_train_1 | 4 | 39 | GRU (oracle OAT) | 0.6 [0.07] | 0.6 [0.07] | 0.6 [0.07] | 0.6 [0.07] | 0.5 [0.07] |
| 1 building_train_2 | 9 | 85 | persistence | 7.6 [0.57] | 6.9 [0.55] | 6.8 [0.54] | 6.8 [0.54] | 7.2 [0.57] |
| 1 building_train_2 | 9 | 85 | linear AR | 4.1 [0.31] | 3.6 [0.29] | 3.5 [0.28] | 3.5 [0.28] | 3.7 [0.29] |
| 1 building_train_2 | 9 | 85 | GRU | 3.7 [0.28] | 2.9 [0.23] | 2.7 [0.21] | 2.8 [0.22] | 3.2 [0.26] |
| 1 building_train_2 | 9 | 85 | GRU (oracle OAT) | 3.7 [0.28] | 2.9 [0.23] | 2.6 [0.21] | 2.7 [0.21] | 2.7 [0.22] |
| 2 building_train_4 | 6 | 54 | persistence | 5.3 [0.69] | 5.4 [0.70] | 5.3 [0.68] | 5.3 [0.68] | 5.3 [0.69] |
| 2 building_train_4 | 6 | 54 | linear AR | 3.1 [0.40] | 3.0 [0.40] | 3.1 [0.40] | 3.1 [0.40] | 3.1 [0.41] |
| 2 building_train_4 | 6 | 54 | GRU | 1.4 [0.18] | 1.3 [0.17] | 1.2 [0.15] | 1.1 [0.14] | 1.0 [0.12] |
| 2 building_train_4 | 6 | 54 | GRU (oracle OAT) | 1.4 [0.18] | 1.3 [0.17] | 1.2 [0.15] | 1.1 [0.14] | 1.0 [0.12] |
| 3 building_train_5 | 8 | 79 | persistence | 5.8 [0.65] | 5.7 [0.64] | 5.6 [0.63] | 5.5 [0.62] | 5.4 [0.60] |
| 3 building_train_5 | 8 | 79 | linear AR | 2.6 [0.29] | 2.6 [0.29] | 2.6 [0.29] | 2.6 [0.29] | 2.6 [0.29] |
| 3 building_train_5 | 8 | 79 | GRU | 0.7 [0.07] | 0.6 [0.07] | 0.7 [0.08] | 0.7 [0.08] | 0.9 [0.10] |
| 3 building_train_5 | 8 | 79 | GRU (oracle OAT) | 0.7 [0.07] | 0.6 [0.07] | 0.7 [0.08] | 0.7 [0.08] | 0.9 [0.10] |
| 4 building_train_7 | 4 | 36 | persistence | 3.7 [0.82] | 3.5 [0.81] | 3.5 [0.80] | 3.5 [0.80] | 3.6 [0.81] |
| 4 building_train_7 | 4 | 36 | linear AR | 1.4 [0.31] | 1.3 [0.29] | 1.3 [0.30] | 1.3 [0.30] | 1.3 [0.30] |
| 4 building_train_7 | 4 | 36 | GRU | 0.6 [0.13] | 0.5 [0.11] | 0.5 [0.11] | 0.5 [0.12] | 0.6 [0.14] |
| 4 building_train_7 | 4 | 36 | GRU (oracle OAT) | 0.6 [0.13] | 0.5 [0.12] | 0.5 [0.12] | 0.5 [0.12] | 0.5 [0.12] |
| 5 building_train_8 | 3 | 30 | persistence | 3.6 [0.66] | 3.3 [0.64] | 3.3 [0.64] | 3.3 [0.64] | 3.4 [0.65] |
| 5 building_train_8 | 3 | 30 | linear AR | 1.6 [0.29] | 1.4 [0.26] | 1.3 [0.26] | 1.4 [0.26] | 1.5 [0.28] |
| 5 building_train_8 | 3 | 30 | GRU | 0.3 [0.05] | 0.3 [0.05] | 0.3 [0.05] | 0.3 [0.06] | 0.4 [0.08] |
| 5 building_train_8 | 3 | 30 | GRU (oracle OAT) | 0.3 [0.05] | 0.3 [0.05] | 0.3 [0.05] | 0.3 [0.05] | 0.3 [0.06] |

## Gas burner MAE (kW) [relative MAE]

| fold / held-out | K | V | model | 1 | 3 | 6 | 12 | 36 |
|---|---|---|---|---|---|---|---|---|
| 0 building_train_1 | 4 | 39 | persistence | 36.2 [1.05] | 34.4 [1.07] | 35.2 [1.05] | 35.5 [1.06] | 36.2 [1.06] |
| 0 building_train_1 | 4 | 39 | linear AR | 20.3 [0.59] | 18.8 [0.58] | 19.1 [0.57] | 18.9 [0.56] | 18.9 [0.55] |
| 0 building_train_1 | 4 | 39 | GRU | 6.2 [0.18] | 5.2 [0.16] | 5.8 [0.17] | 5.8 [0.17] | 5.6 [0.16] |
| 0 building_train_1 | 4 | 39 | GRU (oracle OAT) | 6.2 [0.18] | 5.2 [0.16] | 5.7 [0.17] | 5.8 [0.17] | 5.5 [0.16] |
| 1 building_train_2 | 9 | 85 | persistence | 34.7 [1.04] | 31.2 [1.08] | 32.8 [1.16] | 32.7 [1.16] | 33.5 [1.18] |
| 1 building_train_2 | 9 | 85 | linear AR | 30.3 [0.91] | 27.5 [0.95] | 27.7 [0.98] | 27.1 [0.96] | 26.7 [0.94] |
| 1 building_train_2 | 9 | 85 | GRU | 45.0 [1.34] | 43.4 [1.50] | 40.3 [1.43] | 38.5 [1.36] | 38.4 [1.36] |
| 1 building_train_2 | 9 | 85 | GRU (oracle OAT) | 45.0 [1.34] | 43.3 [1.49] | 40.2 [1.42] | 38.2 [1.35] | 37.6 [1.33] |
| 2 building_train_4 | 6 | 54 | persistence | 7.5 [1.72] | 7.1 [1.71] | 7.1 [1.68] | 7.2 [1.69] | 7.7 [1.81] |
| 2 building_train_4 | 6 | 54 | linear AR | 9.2 [2.11] | 8.7 [2.09] | 8.2 [1.94] | 7.6 [1.80] | 7.0 [1.65] |
| 2 building_train_4 | 6 | 54 | GRU | 1.5 [0.34] | 1.5 [0.36] | 1.3 [0.31] | 1.3 [0.31] | 1.5 [0.36] |
| 2 building_train_4 | 6 | 54 | GRU (oracle OAT) | 1.5 [0.34] | 1.5 [0.35] | 1.2 [0.29] | 1.1 [0.27] | 1.1 [0.26] |
| 3 building_train_5 | 8 | 79 | persistence | 9.0 [1.47] | 8.4 [1.35] | 8.0 [1.38] | 8.1 [1.40] | 8.9 [1.52] |
| 3 building_train_5 | 8 | 79 | linear AR | 7.9 [1.30] | 6.8 [1.10] | 7.0 [1.19] | 6.4 [1.10] | 6.1 [1.05] |
| 3 building_train_5 | 8 | 79 | GRU | 2.4 [0.39] | 2.4 [0.38] | 1.9 [0.33] | 1.8 [0.32] | 1.8 [0.30] |
| 3 building_train_5 | 8 | 79 | GRU (oracle OAT) | 2.4 [0.39] | 2.4 [0.38] | 1.9 [0.32] | 1.8 [0.30] | 1.5 [0.27] |
| 4 building_train_7 | 4 | 36 | persistence | 7.7 [0.99] | 6.9 [1.01] | 7.3 [0.96] | 7.3 [0.96] | 7.4 [0.97] |
| 4 building_train_7 | 4 | 36 | linear AR | 6.9 [0.89] | 6.4 [0.94] | 6.7 [0.88] | 6.5 [0.86] | 6.7 [0.88] |
| 4 building_train_7 | 4 | 36 | GRU | 2.0 [0.26] | 1.8 [0.26] | 1.9 [0.25] | 1.9 [0.25] | 1.7 [0.23] |
| 4 building_train_7 | 4 | 36 | GRU (oracle OAT) | 2.0 [0.26] | 1.8 [0.27] | 2.0 [0.26] | 2.0 [0.26] | 1.9 [0.25] |
| 5 building_train_8 | 3 | 30 | persistence | 10.6 [1.12] | 9.8 [1.07] | 9.8 [1.15] | 9.6 [1.13] | 9.5 [1.11] |
| 5 building_train_8 | 3 | 30 | linear AR | 7.0 [0.74] | 6.4 [0.69] | 6.7 [0.78] | 6.7 [0.79] | 7.0 [0.82] |
| 5 building_train_8 | 3 | 30 | GRU | 1.3 [0.14] | 1.2 [0.13] | 1.2 [0.14] | 1.3 [0.16] | 1.5 [0.18] |
| 5 building_train_8 | 3 | 30 | GRU (oracle OAT) | 1.3 [0.14] | 1.2 [0.13] | 1.2 [0.14] | 1.3 [0.16] | 1.5 [0.17] |

## Reward MAE — deployable estimator

| fold / held-out | K | V | model | 1 | 3 | 6 | 12 | 36 |
|---|---|---|---|---|---|---|---|---|
| 0 building_train_1 | 4 | 39 | persistence | 0.01887 | 0.01828 | 0.01873 | 0.02200 | 0.03336 |
| 0 building_train_1 | 4 | 39 | linear AR | 0.01237 | 0.01190 | 0.01224 | 0.01486 | 0.02388 |
| 0 building_train_1 | 4 | 39 | GRU | 0.00811 | 0.00793 | 0.00789 | 0.00787 | 0.00785 |
| 0 building_train_1 | 4 | 39 | GRU (oracle OAT) | 0.00811 | 0.00793 | 0.00787 | 0.00781 | 0.00759 |
| 1 building_train_2 | 9 | 85 | persistence | 0.00966 | 0.00913 | 0.01029 | 0.01654 | 0.03943 |
| 1 building_train_2 | 9 | 85 | linear AR | 0.00823 | 0.00764 | 0.00838 | 0.01278 | 0.02780 |
| 1 building_train_2 | 9 | 85 | GRU | 0.01005 | 0.00957 | 0.00987 | 0.01223 | 0.01764 |
| 1 building_train_2 | 9 | 85 | GRU (oracle OAT) | 0.01005 | 0.00956 | 0.00984 | 0.01218 | 0.01748 |
| 2 building_train_4 | 6 | 54 | persistence | 0.00571 | 0.00569 | 0.00653 | 0.00812 | 0.01406 |
| 2 building_train_4 | 6 | 54 | linear AR | 0.00530 | 0.00526 | 0.00606 | 0.00741 | 0.01228 |
| 2 building_train_4 | 6 | 54 | GRU | 0.00318 | 0.00314 | 0.00335 | 0.00349 | 0.00410 |
| 2 building_train_4 | 6 | 54 | GRU (oracle OAT) | 0.00318 | 0.00314 | 0.00334 | 0.00348 | 0.00417 |
| 3 building_train_5 | 8 | 79 | persistence | 0.00315 | 0.00305 | 0.00391 | 0.00548 | 0.01117 |
| 3 building_train_5 | 8 | 79 | linear AR | 0.00213 | 0.00202 | 0.00284 | 0.00408 | 0.00791 |
| 3 building_train_5 | 8 | 79 | GRU | 0.00069 | 0.00069 | 0.00071 | 0.00081 | 0.00159 |
| 3 building_train_5 | 8 | 79 | GRU (oracle OAT) | 0.00069 | 0.00069 | 0.00070 | 0.00081 | 0.00179 |
| 4 building_train_7 | 4 | 36 | persistence | 0.01368 | 0.01346 | 0.01431 | 0.01671 | 0.02590 |
| 4 building_train_7 | 4 | 36 | linear AR | 0.01237 | 0.01223 | 0.01312 | 0.01488 | 0.02036 |
| 4 building_train_7 | 4 | 36 | GRU | 0.01065 | 0.01077 | 0.01044 | 0.01026 | 0.00944 |
| 4 building_train_7 | 4 | 36 | GRU (oracle OAT) | 0.01065 | 0.01078 | 0.01045 | 0.01027 | 0.00943 |
| 5 building_train_8 | 3 | 30 | persistence | 0.00629 | 0.00592 | 0.00705 | 0.01146 | 0.02840 |
| 5 building_train_8 | 3 | 30 | linear AR | 0.00345 | 0.00308 | 0.00431 | 0.00815 | 0.02103 |
| 5 building_train_8 | 3 | 30 | GRU | 0.00081 | 0.00075 | 0.00085 | 0.00089 | 0.00106 |
| 5 building_train_8 | 3 | 30 | GRU (oracle OAT) | 0.00081 | 0.00075 | 0.00084 | 0.00088 | 0.00124 |

## Reward MAE — exact-physics (privileged occupancy/calendar)

| fold / held-out | K | V | model | 1 | 3 | 6 | 12 | 36 |
|---|---|---|---|---|---|---|---|---|
| 0 building_train_1 | 4 | 39 | persistence | 0.01327 | 0.01263 | 0.01298 | 0.01513 | 0.02601 |
| 0 building_train_1 | 4 | 39 | linear AR | 0.00655 | 0.00598 | 0.00630 | 0.00815 | 0.01640 |
| 0 building_train_1 | 4 | 39 | GRU | 0.00224 | 0.00196 | 0.00213 | 0.00214 | 0.00212 |
| 0 building_train_1 | 4 | 39 | GRU (oracle OAT) | 0.00224 | 0.00196 | 0.00211 | 0.00209 | 0.00197 |
| 1 building_train_2 | 9 | 85 | persistence | 0.00605 | 0.00554 | 0.00728 | 0.01432 | 0.03755 |
| 1 building_train_2 | 9 | 85 | linear AR | 0.00476 | 0.00417 | 0.00533 | 0.01023 | 0.02545 |
| 1 building_train_2 | 9 | 85 | GRU | 0.00664 | 0.00611 | 0.00663 | 0.00941 | 0.01494 |
| 1 building_train_2 | 9 | 85 | GRU (oracle OAT) | 0.00664 | 0.00610 | 0.00660 | 0.00936 | 0.01487 |
| 2 building_train_4 | 6 | 54 | persistence | 0.00338 | 0.00336 | 0.00433 | 0.00585 | 0.01223 |
| 2 building_train_4 | 6 | 54 | linear AR | 0.00307 | 0.00300 | 0.00394 | 0.00529 | 0.01081 |
| 2 building_train_4 | 6 | 54 | GRU | 0.00092 | 0.00083 | 0.00114 | 0.00130 | 0.00208 |
| 2 building_train_4 | 6 | 54 | GRU (oracle OAT) | 0.00092 | 0.00083 | 0.00113 | 0.00128 | 0.00212 |
| 3 building_train_5 | 8 | 79 | persistence | 0.00295 | 0.00288 | 0.00343 | 0.00480 | 0.01057 |
| 3 building_train_5 | 8 | 79 | linear AR | 0.00197 | 0.00188 | 0.00251 | 0.00361 | 0.00754 |
| 3 building_train_5 | 8 | 79 | GRU | 0.00054 | 0.00053 | 0.00053 | 0.00060 | 0.00104 |
| 3 building_train_5 | 8 | 79 | GRU (oracle OAT) | 0.00054 | 0.00052 | 0.00052 | 0.00061 | 0.00121 |
| 4 building_train_7 | 4 | 36 | persistence | 0.00413 | 0.00381 | 0.00503 | 0.00804 | 0.02011 |
| 4 building_train_7 | 4 | 36 | linear AR | 0.00272 | 0.00247 | 0.00349 | 0.00563 | 0.01274 |
| 4 building_train_7 | 4 | 36 | GRU | 0.00090 | 0.00091 | 0.00086 | 0.00090 | 0.00112 |
| 4 building_train_7 | 4 | 36 | GRU (oracle OAT) | 0.00090 | 0.00092 | 0.00088 | 0.00092 | 0.00110 |
| 5 building_train_8 | 3 | 30 | persistence | 0.00633 | 0.00594 | 0.00700 | 0.01126 | 0.02797 |
| 5 building_train_8 | 3 | 30 | linear AR | 0.00348 | 0.00310 | 0.00431 | 0.00823 | 0.02102 |
| 5 building_train_8 | 3 | 30 | GRU | 0.00078 | 0.00072 | 0.00081 | 0.00083 | 0.00103 |
| 5 building_train_8 | 3 | 30 | GRU (oracle OAT) | 0.00078 | 0.00072 | 0.00080 | 0.00083 | 0.00122 |

## Pooled over all held-out buildings

| group | model | windows | zone MAE h1 | h12 | h36 | HWS MAE h12 | elec MAE kW h12 | gas MAE kW h12 | reward MAE (deploy) h12 |
|---|---|---|---|---|---|---|---|---|---|
| all | persistence | 19680 | 0.065 | 0.254 | 0.635 | 0.83 | 5.1 | 16.7 | 0.01338 |
| all | linear AR | 19680 | 0.065 | 0.254 | 0.601 | 2.37 | 2.5 | 12.2 | 0.01036 |
| all | GRU | 19680 | 0.045 | 0.144 | 0.307 | 0.49 | 1.0 | 8.5 | 0.00592 |
| all | GRU (oracle OAT) | 19680 | 0.045 | 0.145 | 0.313 | 0.49 | 1.0 | 8.4 | 0.00590 |

## By weather station (pooled)

| group | model | windows | zone MAE h1 | h12 | h36 | HWS MAE h12 | elec MAE kW h12 | gas MAE kW h12 | reward MAE (deploy) h12 |
|---|---|---|---|---|---|---|---|---|---|
| atlanta | persistence | 3936 | 0.045 | 0.188 | 0.466 | 0.62 | 4.3 | 6.2 | 0.00591 |
| atlanta | linear AR | 3936 | 0.047 | 0.200 | 0.478 | 2.20 | 1.8 | 5.2 | 0.00442 |
| atlanta | GRU | 3936 | 0.028 | 0.095 | 0.213 | 0.36 | 1.0 | 1.1 | 0.00100 |
| atlanta | GRU (oracle OAT) | 3936 | 0.028 | 0.096 | 0.214 | 0.35 | 1.0 | 1.2 | 0.00105 |
| denver | persistence | 3936 | 0.059 | 0.255 | 0.629 | 1.08 | 2.3 | 15.0 | 0.01225 |
| denver | linear AR | 3936 | 0.059 | 0.239 | 0.571 | 2.57 | 1.4 | 11.6 | 0.01088 |
| denver | GRU | 3936 | 0.040 | 0.141 | 0.304 | 0.53 | 0.5 | 9.2 | 0.00787 |
| denver | GRU (oracle OAT) | 3936 | 0.040 | 0.144 | 0.315 | 0.54 | 0.5 | 9.0 | 0.00781 |
| fairbanks | persistence | 3936 | 0.107 | 0.413 | 1.054 | 2.05 | 2.2 | 59.0 | 0.02741 |
| fairbanks | linear AR | 3936 | 0.109 | 0.432 | 1.013 | 3.27 | 1.6 | 36.8 | 0.01939 |
| fairbanks | GRU | 3936 | 0.083 | 0.256 | 0.528 | 1.28 | 0.7 | 31.1 | 0.01171 |
| fairbanks | GRU (oracle OAT) | 3936 | 0.083 | 0.258 | 0.538 | 1.28 | 0.7 | 31.0 | 0.01169 |
| phoenix | persistence | 3936 | 0.070 | 0.241 | 0.607 | 0.28 | 8.2 | 3.3 | 0.00983 |
| phoenix | linear AR | 3936 | 0.068 | 0.238 | 0.570 | 2.05 | 5.3 | 5.3 | 0.00851 |
| phoenix | GRU | 3936 | 0.047 | 0.143 | 0.314 | 0.17 | 1.3 | 0.7 | 0.00322 |
| phoenix | GRU (oracle OAT) | 3936 | 0.047 | 0.144 | 0.318 | 0.17 | 1.1 | 0.6 | 0.00317 |
| singapore | persistence | 3936 | 0.046 | 0.174 | 0.417 | 0.10 | 8.3 | 0.1 | 0.01153 |
| singapore | linear AR | 3936 | 0.041 | 0.158 | 0.373 | 1.77 | 2.4 | 2.1 | 0.00859 |
| singapore | GRU | 3936 | 0.027 | 0.084 | 0.179 | 0.10 | 1.6 | 0.1 | 0.00582 |
| singapore | GRU (oracle OAT) | 3936 | 0.027 | 0.085 | 0.181 | 0.10 | 1.5 | 0.1 | 0.00581 |

## Severe-cold scenarios (each n = 1 scenario, 2 episodes)

| group | model | windows | zone MAE h1 | h12 | h36 | HWS MAE h12 | elec MAE kW h12 | gas MAE kW h12 | reward MAE (deploy) h12 |
|---|---|---|---|---|---|---|---|---|---|
| fairbanks_1 | persistence | 656 | 0.184 | 0.660 | 1.644 | 3.27 | 3.1 | 162.5 | 0.08129 |
| fairbanks_1 | linear AR | 656 | 0.185 | 0.728 | 1.616 | 4.94 | 1.2 | 77.4 | 0.05523 |
| fairbanks_1 | GRU | 656 | 0.175 | 0.497 | 0.878 | 1.52 | 0.5 | 28.1 | 0.03683 |
| fairbanks_1 | GRU (oracle OAT) | 656 | 0.175 | 0.500 | 0.901 | 1.52 | 0.5 | 27.9 | 0.03671 |
| fairbanks_2 | persistence | 656 | 0.166 | 0.606 | 1.573 | 3.31 | 2.4 | 119.2 | 0.03544 |
| fairbanks_2 | linear AR | 656 | 0.176 | 0.658 | 1.540 | 4.31 | 1.5 | 95.4 | 0.02308 |
| fairbanks_2 | GRU | 656 | 0.143 | 0.422 | 0.894 | 2.93 | 2.4 | 145.9 | 0.02690 |
| fairbanks_2 | GRU (oracle OAT) | 656 | 0.143 | 0.423 | 0.895 | 2.93 | 2.4 | 145.3 | 0.02684 |
| fairbanks_8 | persistence | 656 | 0.153 | 0.596 | 1.574 | 2.55 | 0.7 | 40.5 | 0.02712 |
| fairbanks_8 | linear AR | 656 | 0.140 | 0.564 | 1.383 | 3.16 | 1.4 | 23.1 | 0.01938 |
| fairbanks_8 | GRU | 656 | 0.088 | 0.277 | 0.635 | 2.00 | 0.1 | 5.8 | 0.00205 |
| fairbanks_8 | GRU (oracle OAT) | 656 | 0.088 | 0.277 | 0.633 | 2.00 | 0.1 | 5.7 | 0.00209 |
| not severe cold | persistence | 17712 | 0.054 | 0.214 | 0.528 | 0.58 | 5.4 | 6.6 | 0.00954 |
| not severe cold | linear AR | 17712 | 0.054 | 0.210 | 0.500 | 2.18 | 2.6 | 6.3 | 0.00789 |
| not severe cold | GRU | 17712 | 0.035 | 0.116 | 0.252 | 0.30 | 1.0 | 2.7 | 0.00415 |
| not severe cold | GRU (oracle OAT) | 17712 | 0.035 | 0.117 | 0.258 | 0.30 | 1.0 | 2.7 | 0.00413 |

## Operating conditions (step-level selection)

| group | model | windows | zone MAE h1 | h12 | h36 | HWS MAE h12 | elec MAE kW h12 | gas MAE kW h12 | reward MAE (deploy) h12 |
|---|---|---|---|---|---|---|---|---|---|
| occ_transition | persistence | 600 | 0.406 | 1.017 | 0.974 | 2.05 | 6.2 | 50.0 | 0.11106 |
| occ_transition | linear AR | 600 | 0.328 | 0.858 | 0.930 | 2.90 | 4.7 | 36.3 | 0.08396 |
| occ_transition | GRU | 600 | 0.139 | 0.267 | 0.287 | 1.06 | 2.1 | 29.1 | 0.02327 |
| occ_transition | GRU (oracle OAT) | 600 | 0.139 | 0.268 | 0.281 | 1.06 | 2.1 | 29.1 | 0.02324 |
| comfort_violation | persistence | 1982 | 0.091 | 0.460 | 0.871 | 2.02 | 6.6 | 46.1 | 0.04523 |
| comfort_violation | linear AR | 1982 | 0.104 | 0.469 | 0.879 | 2.90 | 5.2 | 41.1 | 0.03357 |
| comfort_violation | GRU | 1982 | 0.069 | 0.164 | 0.279 | 1.23 | 2.6 | 40.8 | 0.01308 |
| comfort_violation | GRU (oracle OAT) | 1982 | 0.069 | 0.166 | 0.282 | 1.24 | 2.5 | 40.6 | 0.01303 |
| heating_recovery | persistence | 647 | 0.411 | 1.330 | 2.318 | 4.67 | 2.7 | 108.4 | 0.09897 |
| heating_recovery | linear AR | 647 | 0.362 | 1.024 | 1.581 | 4.06 | 2.9 | 75.7 | 0.06712 |
| heating_recovery | GRU | 647 | 0.259 | 0.409 | 0.602 | 2.16 | 1.5 | 92.9 | 0.03001 |
| heating_recovery | GRU (oracle OAT) | 647 | 0.259 | 0.408 | 0.595 | 2.16 | 1.6 | 92.5 | 0.02976 |
| cold_step | persistence | 1198 | 0.202 | 0.719 | 1.871 | 3.56 | 2.9 | 149.4 | 0.04956 |
| cold_step | linear AR | 1198 | 0.205 | 0.776 | 1.795 | 4.75 | 1.7 | 93.3 | 0.02794 |
| cold_step | GRU | 1198 | 0.181 | 0.531 | 1.052 | 2.65 | 1.4 | 89.2 | 0.01875 |
| cold_step | GRU (oracle OAT) | 1198 | 0.181 | 0.533 | 1.063 | 2.65 | 1.3 | 89.2 | 0.01872 |
