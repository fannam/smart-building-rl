import numpy as np
import pytest
from gymnasium.spaces import Box

from sbx import paths
from sbx.env import Sanitizer

SPEC = {
    "continuous_action": Box(-1.0, 1.0, (5,), np.float32),
    "discrete_action": Box(0, 1, (3,), np.int32),
}


def test_valid_action_passes_unchanged():
    s = Sanitizer(SPEC)
    a = {"continuous_action": np.array([-1, -0.5, 0, 0.5, 1], np.float32),
         "discrete_action": np.array([0, 1, 1], np.int32)}
    out = s(a)
    assert np.array_equal(out["continuous_action"], a["continuous_action"])
    assert np.array_equal(out["discrete_action"], a["discrete_action"])
    assert out["continuous_action"].dtype == np.float32 and out["discrete_action"].dtype == np.int32
    assert sum(s.fixes.values()) == 0


def test_nan_inf_out_of_range_and_float_discrete_are_fixed():
    s = Sanitizer(SPEC)
    out = s({"continuous_action": [np.nan, np.inf, -3.0, 0.2, 7.0], "discrete_action": [0.6, -2, 5]})
    assert np.all(np.isfinite(out["continuous_action"]))
    assert np.all(np.abs(out["continuous_action"]) <= 1.0)
    assert out["continuous_action"][0] == 0.0
    assert out["discrete_action"].tolist() == [1, 0, 1]
    assert s.fixes["nonfinite"] == 1 and s.fixes["clipped"] == 2


def test_wrong_shape_is_fixed():
    s = Sanitizer(SPEC)
    out = s({"continuous_action": np.zeros(4), "discrete_action": np.zeros(3)})
    assert out["continuous_action"].shape == (5,)
    assert s.fixes["shape"] == 1


@pytest.mark.skipif(not paths.TRAINING_SCENARIOS.exists(), reason="starter kit not cloned")
def test_discover_training_scenarios():
    all_ = paths.discover()
    assert len(all_) == 30
    cold = paths.discover(patterns=["*/fairbanks_*"])
    assert len(cold) == 6 and all("fairbanks" in paths.scenario_key(d) for d in cold)
