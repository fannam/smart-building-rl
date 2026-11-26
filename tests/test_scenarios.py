from collections import defaultdict

import numpy as np
import pytest
import yaml

from sbx import paths
from sbx.scenarios import SUITE, weights


def test_weights_give_deep_cold_its_share():
    info = {"a": {"climate": "deep_cold"}, "b": {"climate": "hot"}, "c": {"climate": "mild"}, "d": {"climate": "hot"}}
    w = weights(["a", "b", "c", "d"], info, 0.25)
    assert np.isclose(w.sum(), 1.0) and np.isclose(w[0], 0.25) and np.allclose(w[1:], 0.25)
    assert np.allclose(weights(["b", "c"], info, 0.25), 0.5)  # no deep cold -> equal weights


@pytest.mark.skipif(not SUITE.exists(), reason="suite manifest not built")
def test_suite_manifest_integrity():
    m = yaml.safe_load(SUITE.read_text())
    shipped = {f.stem for f in (paths.KIT / "smart_buildings" / "weather").glob("*.yaml")}
    strip_split = {s["name"]: s["split"] for s in m["strips"]}
    assert not shipped & set(strip_split), "a shipped weather strip is reused"

    per_building = defaultdict(list)
    for s in m["scenarios"]:
        per_building[s["building"]].append(s)
        expected = "test" if s["split"] == "test_large" else s["split"]
        assert strip_split[s["weather"]] == expected, f"strip {s['weather']} crosses splits"
    splits = {b["name"]: b["split"] for b in m["buildings"]}
    for b, sc in per_building.items():
        assert len(sc) == 7 and len({s["weather"] for s in sc}) == 7
        assert {s["split"] for s in sc} == {splits[b]}
        assert sum(s["climate"] == "deep_cold" for s in sc) >= 1
    assert {s: sum(v == s for v in splits.values()) for s in ("tune", "val", "test", "test_large")} == \
        {"tune": 24, "val": 8, "test": 8, "test_large": 8}
