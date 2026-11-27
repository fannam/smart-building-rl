import math
import os

import numpy as np
import pytest

from sbx import paths
from sbx.controller import reference
from sbx.controller.tunable import BASE, DEFAULTS, PARAMS, TunablePolicy, bounds


def test_defaults_mirror_the_reference():
    for k, s in PARAMS.items():
        assert s.low <= s.default <= s.high, k
        if s.kind == "defaults":
            assert s.default == BASE.DEFAULTS[k], k
        elif s.kind == "constants":
            assert s.default == getattr(BASE, k), k
    lo, hi = bounds()
    assert np.all(lo < hi) and len(lo) == len(DEFAULTS)


def test_weekday_rounds_the_day_of_week_angle():
    p = TunablePolicy.__new__(TunablePolicy)
    p._dow = (0, 1)
    for wd in range(7):
        angle = 2.0 * math.pi * wd / 7.0
        obs = np.array([math.sin(angle), math.cos(angle)], dtype=np.float32)  # float32, as the environment emits
        assert p._weekday(obs) == wd


def test_unknown_parameter_is_rejected():
    with pytest.raises(ValueError):
        TunablePolicy({}, None, {"no_such_parameter": 1.0})


def _scenario_dirs() -> list:
    shipped = sorted({d.parent for d in paths.TRAINING_SCENARIOS.glob("*/*/") if d.is_dir()})
    return [sorted(b.iterdir())[0] for b in shipped]


@pytest.mark.skipif(not os.environ.get("SBX_SLOW"), reason="set SBX_SLOW=1 (rolls out the simulator)")
def test_default_params_reproduce_reference_actions():
    from sbx.env import load_env

    env, meta, spec = load_env(_scenario_dirs()[0], paths.KIT)
    obs, _ = env.reset()
    a, b = BASE(meta, spec), TunablePolicy(meta, spec)
    for _ in range(288):
        x, y = a.get_action(obs), b.get_action(obs)
        assert all(np.array_equal(np.asarray(x[k]), np.asarray(y[k])) for k in x)
        obs, *_ = env.step(x)


@pytest.mark.skipif(not os.environ.get("SBX_SLOW"), reason="set SBX_SLOW=1 (rolls out the simulator)")
@pytest.mark.parametrize("corner", ["low", "high"])
def test_extreme_params_give_valid_actions_on_every_shipped_building(corner):
    from sbx.env import Sanitizer, load_env

    lo, hi = bounds()
    params = dict(zip(PARAMS, lo if corner == "low" else hi))
    for d in _scenario_dirs():
        env, meta, spec = load_env(d, paths.KIT)
        obs, _ = env.reset()
        policy, sanitize = TunablePolicy(meta, spec, params), Sanitizer(spec)
        for _ in range(36):
            obs, *_ = env.step(sanitize(policy.get_action(obs)))
        assert not any(sanitize.fixes.values()), f"{d}: controller emitted invalid actions"
