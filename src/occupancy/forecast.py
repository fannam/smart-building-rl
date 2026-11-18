"""Building-level occupancy forecasting (Phase 2B Workstream C).

Target: N(t) = sum_z zone_info[t].average_occupancy (privileged; training target only) for the
transitions t = t0 .. t0+H-1 of a decision window, and the indicator N(t) > 0.

Runtime inputs (all from observations up to o_{t0} and public metadata):
    num_occupants in obs[i] = N(i-2)  (offset 2, verified exact; configs/sensor_alignment.json)
    clock of o_{t0} -> exact clock of every future reward interval (sim_ts[t+1])
    public schedule (comfort hours), num_zones
History older than the context window is used (the deployed controller may keep its own past
observations), but never anything after o_{t0}. `runtime_features()` is the only function that
builds model inputs; it reads only `obs` rows <= t0.

Why building-level only: per-zone occupancy is not observable, and in every shipped scenario each
zone has the same number of occupants with the same arrival/departure distribution
(RandomizedArrivalDepartureOccupancy, scalar zone_assignment), so zones are exchangeable: the
expected per-zone share is uniform and the realized split is not forecastable from runtime data.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from dynamics.schedule import advance, comfort_mode, decode_clock
from runtime_features import validate_runtime_features

DAY = 288
FEATURES = ("k", "hod_sin", "hod_cos", "weekday", "comfort", "h_since_open", "h_until_close",
            "n_last", "n_mean12", "n_slope6", "n_yday_same", "n_max_24h", "n_max_today", "gate_on_h_now",
            "absent_while_open")


class OccupancyFeaturizer:
    def __init__(self, meta: Mapping[str, Any]):
        names = meta["observation_field_names"]
        self.fi = {n: i for i, n in enumerate(names)}
        self.i_occ = self.fi["num_occupants"]
        self.mean = float(np.asarray(meta["observation_mean"])[self.i_occ])
        self.std = float(np.asarray(meta["observation_std"])[self.i_occ])
        self.V = int(meta["num_zones"])
        self.schedule = meta["schedule"]
        self.features_used = ["obs:num_occupants", "obs:hod_sin_000", "obs:hod_cos_000", "obs:dow_sin_000",
                              "obs:dow_cos_000", "meta:schedule", "meta:num_zones", "meta:observation_mean",
                              "meta:observation_std", "meta:observation_field_names"]
        validate_runtime_features(self.features_used, field_names=names)

    def observed_total(self, obs: np.ndarray) -> np.ndarray:
        """N(i-2) for every observation row i (persons)."""
        return np.asarray(obs, np.float64)[..., self.i_occ] * self.std + self.mean

    def runtime_features(self, obs_upto_t0: np.ndarray, H: int) -> np.ndarray:
        """obs_upto_t0: observation rows o_0 .. o_{t0} (nothing later). Returns [H, len(FEATURES)]."""
        obs = np.asarray(obs_upto_t0)
        t0 = len(obs) - 1
        n = self.observed_total(obs) / self.V                     # per-zone occupants; n[i] = N(i-2)/V
        hour0, dow0 = decode_clock(obs[-1], self.fi)
        k = np.arange(H)
        hour, dow = advance(hour0, dow0, k + 1)                   # reward interval of transition t0+k
        sch = self.schedule
        comf = comfort_mode(hour, dow, sch)
        wd = (np.floor(dow + 1e-6) < 5).astype(np.float64)
        a = 2 * np.pi * hour / 24
        # history (all at or before row t0)
        last = n[-1]
        mean12 = n[-12:].mean()
        slope6 = (n[-1] - n[-7]) / 6 if len(n) >= 7 else 0.0
        # same time yesterday: N(t0+k-288) is in obs row t0+k-288+2 (must be <= t0)
        rows_y = t0 + k - DAY + 2
        yday = np.where((rows_y >= 0) & (rows_y <= t0), n[np.clip(rows_y, 0, t0)], np.nan)
        max24 = n[-DAY:].max() if len(n) >= DAY // 2 else np.nan
        # today: rows since the schedule opened today (hours since open at t0)
        h_now, d_now = decode_clock(obs[-1], self.fi)
        open_now = bool(comfort_mode(np.array([h_now]), np.array([d_now]), sch)[0])
        on_h = (h_now - sch["morning_start_hour"]) if open_now else -1.0
        steps_open = int(round(max(on_h, 0) * 12))
        max_today = n[-steps_open - 1:].max() if open_now else np.nan
        absent = float(open_now and on_h >= 0.5 and max_today == 0.0)
        X = np.stack([k / 36.0, np.sin(a), np.cos(a), wd, comf.astype(np.float64),
                      np.clip(hour - sch["morning_start_hour"], -12, 24), np.clip(sch["evening_start_hour"] - hour, -12, 24),
                      np.full(H, last), np.full(H, mean12), np.full(H, slope6), yday, np.full(H, max24),
                      np.full(H, max_today), np.full(H, on_h), np.full(H, absent)], -1)
        return X


def episode_targets(zone_info: np.ndarray) -> np.ndarray:
    """True N(t) for every transition (privileged)."""
    return np.asarray(zone_info, np.float64)[..., 5].sum(-1)


# ------------------------------------------------------------------ baselines
class ScheduleOnly:
    """N/V = train mean during comfort hours, 0 otherwise; P(occupied) = comfort indicator."""
    name = "schedule"

    def fit(self, X, y, occ):
        c = X[:, FEATURES.index("comfort")] > 0.5
        self.level = float(y[c].mean())
        return self

    def predict(self, X):
        c = X[:, FEATURES.index("comfort")] > 0.5
        return np.where(c, self.level, 0.0), c.astype(np.float64)


class Persistence:
    """Last observed total (o_{t0}) held over the horizon."""
    name = "persistence"

    def fit(self, X, y, occ):
        return self

    def predict(self, X):
        n = X[:, FEATURES.index("n_last")]
        return n, (n > 0).astype(np.float64)


class TimeProfile:
    """Train mean of N/V and P(N>0) per (weekday flag, 15-min bin of the target time)."""
    name = "time_profile"

    def _bin(self, X):
        h = np.mod(np.arctan2(X[:, 1], X[:, 2]), 2 * np.pi) * 24 / (2 * np.pi)
        return (np.floor(h * 4 + 1e-6).astype(int) % 96) + 96 * (X[:, 3] > 0.5)

    def fit(self, X, y, occ):
        b = self._bin(X)
        self.mu = np.zeros(192); self.p = np.zeros(192)
        for i in range(192):
            m = b == i
            self.mu[i] = y[m].mean() if m.any() else 0.0
            self.p[i] = occ[m].mean() if m.any() else 0.0
        return self

    def predict(self, X):
        b = self._bin(X)
        return self.mu[b], self.p[b]


class LearnedHGB:
    """Gradient-boosted trees (sklearn HistGradientBoosting): regressor on N/V, classifier on N>0.
    Iterations chosen on the inner (time-split) validation set of the TRAIN buildings."""
    name = "hgb"

    def __init__(self, seed=0, grid=(100, 300)):
        self.seed, self.grid = seed, grid

    def fit(self, X, y, occ, Xv=None, yv=None, occv=None):
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        best = None
        for it in self.grid:
            r = HistGradientBoostingRegressor(max_iter=it, learning_rate=0.05, max_leaf_nodes=31, random_state=self.seed,
                                              early_stopping=False).fit(X, y)
            c = HistGradientBoostingClassifier(max_iter=it, learning_rate=0.05, max_leaf_nodes=31, random_state=self.seed,
                                               early_stopping=False).fit(X, occ.astype(int))
            score = 0.0 if Xv is None else float(np.abs(np.clip(r.predict(Xv), 0, None) - yv).mean())
            if best is None or score < best[0]:
                best = (score, it, r, c)
        self.val_mae, self.iters, self.reg, self.clf = best
        return self

    def predict(self, X):
        return np.clip(self.reg.predict(X), 0.0, None), self.clf.predict_proba(X)[:, 1]
