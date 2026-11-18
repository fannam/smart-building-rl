"""Runtime-forecastable exogenous inputs: clock arithmetic and the comfort schedule.

Everything here uses only observation clock features (hod/dow sin/cos, current at
sim_timestamp[i]) and public metadata (schedule hours and windows), so future values
can be computed at decision time without reading future observations.

Simulator rule (setpoint_schedule.SetpointSchedule.is_comfort_mode, holidays empty in all
shipped scenarios): comfort  <=>  morning_start <= hour < evening_start  and  weekday.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

STEP_HOURS = 5.0 / 60.0


def decode_clock(obs_native_or_raw: np.ndarray, field_index: Mapping[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """(hour in [0,24), dow in [0,7) with 0 = Monday) from the base hod/dow features."""
    o = np.asarray(obs_native_or_raw, dtype=np.float64)
    hour = np.mod(np.arctan2(o[..., field_index["hod_sin_000"]], o[..., field_index["hod_cos_000"]]), 2 * np.pi) * 24 / (2 * np.pi)
    dow = np.mod(np.arctan2(o[..., field_index["dow_sin_000"]], o[..., field_index["dow_cos_000"]]), 2 * np.pi) * 7 / (2 * np.pi)
    return hour, dow


def advance(hour: np.ndarray, dow: np.ndarray, steps) -> tuple[np.ndarray, np.ndarray]:
    """Clock after `steps` 5-minute steps (exact arithmetic on decoded hours)."""
    total = np.asarray(hour, dtype=np.float64) + np.asarray(steps, dtype=np.float64) * STEP_HOURS
    days = np.floor((total + 1e-9) / 24.0)
    return np.mod(total, 24.0), np.mod(np.asarray(dow, dtype=np.float64) + days, 7.0)


def comfort_mode(hour, dow, schedule: Mapping[str, Any]) -> np.ndarray:
    h = np.floor(np.asarray(hour) + 1e-6)
    wd = np.floor(np.asarray(dow) + 1e-6) < 5
    return (h >= schedule["morning_start_hour"]) & (h < schedule["evening_start_hour"]) & wd


def temperature_window(hour, dow, schedule: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    c = comfort_mode(hour, dow, schedule)
    cw, ew = schedule["comfort_temp_window"], schedule["eco_temp_window"]
    return np.where(c, cw[0], ew[0]), np.where(c, cw[1], ew[1])


def clock_features(hour, dow) -> np.ndarray:
    """[..., 4]: hod sin/cos, dow sin/cos (same convention as the env's *_000 features)."""
    a, b = 2 * np.pi * np.asarray(hour) / 24.0, 2 * np.pi * np.asarray(dow) / 7.0
    return np.stack([np.sin(a), np.cos(a), np.sin(b), np.cos(b)], -1)
