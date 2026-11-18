"""Commanded vs effective actions: pressure floor, AHU off, recirculation cap, damper floor."""
import unittest

import numpy as np

from tests._paths import HAVE_SAMPLE, SAMPLE
from effective_action import ActionRepresentation, RULE_DAMPER, RULE_IDENTITY, RULE_PRESSURE, RULE_RECIRC

# 1 AHU serving 2 zones + HWS; continuous order mirrors the simulator's (AHU block, HWS block, VAVs)
CN = ["ahs_ahu_1_supply_air_temperature_setpoint", "ahs_ahu_1_supply_air_static_pressure_setpoint",
      "ahs_ahu_1_recirculation_percentage_command", "hws_supply_water_temperature_setpoint",
      "hws_differential_pressure_setpoint", "VAV_1_supply_air_damper_percentage_command",
      "VAV_1_reheat_valve_percentage_command", "VAV_2_supply_air_damper_percentage_command",
      "VAV_2_reheat_valve_percentage_command"]
LOW = [285, 0, 0, 310, 0, 0, 0, 0, 0]
HIGH = [305, 2000, 1, 350, 20, 1, 1, 1, 1]
META = {"n_continuous_action": 9, "n_discrete_action": 2, "continuous_action_names": CN,
        "continuous_action_native_low": LOW, "continuous_action_native_high": HIGH,
        "observation_field_names": [], "observation_mean": [], "observation_std": [],
        "num_occupants_normalization_constant": 0.0,
        "zones": {"ids": ["VAV_1", "VAV_2"], "obs_temp": [], "obs_heating_sp": [], "obs_cooling_sp": [],
                  "act_damper": [5, 7], "act_reheat_valve": [6, 8]},
        "ahus": {"ids": ["ahu_1"], "act_run": [0], "act_sat": [0], "act_static_pressure": [1],
                 "act_recirculation": [2], "zones": [[0, 1]]},
        "hws": {"act_run": 1, "act_supply_water_temp": 3, "act_differential_pressure": 4}}
REP = ActionRepresentation(META)


def norm(native):
    native = np.asarray(native, dtype=float)
    return -1 + 2 * (native - np.array(LOW)) / (np.array(HIGH) - np.array(LOW))


class TestEffective(unittest.TestCase):
    def run_case(self, native, run):
        e = REP.effective(norm(native)[None], np.array([[run, 1]]))
        return e

    def test_rules_layout(self):
        self.assertEqual(REP.rule[1], RULE_PRESSURE)
        self.assertEqual(REP.rule[2], RULE_RECIRC)
        self.assertEqual(list(REP.rule[[5, 7]]), [RULE_DAMPER, RULE_DAMPER])
        self.assertTrue(all(REP.rule[j] == RULE_IDENTITY for j in (0, 3, 4, 6, 8)))

    def test_ahu_on_pressure_floor(self):
        e = self.run_case([290, 100, 0.5, 340, 5, 0.5, 0.2, 0.5, 0.0], run=1)
        self.assertAlmostEqual(e.effective_native[0, 1], 200.0)
        self.assertAlmostEqual(e.commanded_native[0, 1], 100.0)             # commanded preserved
        e = self.run_case([290, 650, 0.5, 340, 5, 0.5, 0.2, 0.5, 0.0], run=1)
        self.assertAlmostEqual(e.effective_native[0, 1], 650.0)
        self.assertTrue(e.active[0, :3].all())

    def test_ahu_off(self):
        e = self.run_case([290, 650, 0.5, 340, 5, 0.5, 0.2, 0.5, 0.0], run=0)
        self.assertEqual(e.effective_native[0, 1], 0.0)                       # pressure 0 when off
        self.assertAlmostEqual(e.effective_native[0, 0], 290.0)               # SAT setpoint still applied
        self.assertFalse(e.active[0, [0, 1, 2, 5, 6, 7, 8]].any())            # AHU + its VAVs inactive
        self.assertTrue(e.active[0, [3, 4]].all())                            # HWS unaffected

    def test_recirculation_cap(self):
        for run in (0, 1):
            e = self.run_case([290, 500, 0.95, 340, 5, 0.5, 0.2, 0.5, 0.0], run=run)
            self.assertAlmostEqual(e.effective_native[0, 2], 0.7)
            self.assertAlmostEqual(e.commanded_native[0, 2], 0.95)
        e = self.run_case([290, 500, 0.3, 340, 5, 0.5, 0.2, 0.5, 0.0], run=1)
        self.assertAlmostEqual(e.effective_native[0, 2], 0.3)

    def test_damper_floor(self):
        e = self.run_case([290, 500, 0.3, 340, 5, 0.03, 0.2, 0.6, 0.0], run=1)
        self.assertAlmostEqual(e.effective_native[0, 5], 0.1)
        self.assertAlmostEqual(e.effective_native[0, 7], 0.6)
        self.assertAlmostEqual(e.effective_native[0, 6], 0.2)                 # valve unchanged

    def test_vectorized_and_discrete_preserved(self):
        a = np.random.default_rng(0).uniform(-1, 1, (5, 7, 9))
        d = np.random.default_rng(1).integers(0, 2, (5, 7, 2))
        e = REP.effective(a, d)
        self.assertEqual(e.effective_native.shape, (5, 7, 9))
        np.testing.assert_array_equal(e.discrete, d)
        np.testing.assert_allclose(REP.codec.to_native(e.effective_normalized), e.effective_native, atol=1e-9)
        with self.assertRaises(ValueError):
            REP.effective(a[..., :8], d)


@unittest.skipUnless(HAVE_SAMPLE, "data/phase1_sample not present")
class TestAgainstSimulatorEchoes(unittest.TestCase):
    def test_effective_equals_echo_in_next_obs(self):
        from dataset import load_episode, load_manifest, load_scenario_meta
        rows = load_manifest(SAMPLE)
        floor_hits = 0
        for r in rows[::4]:                     # 15 episodes, all 6 buildings, both profiles
            ep = load_episode(SAMPLE / r["path"], keys=["obs", "action_continuous", "action_discrete"])
            rep = ActionRepresentation(load_scenario_meta(SAMPLE, r["scenario"]))
            res = rep.verify_against_echo(ep["action_continuous"], ep["action_discrete"], ep["obs"][1:])
            self.assertEqual(res["violations"], {}, r["episode_id"])
            self.assertGreater(res["checked"], 0)
            e = rep.effective(ep["action_continuous"], ep["action_discrete"])
            floor_hits += int(np.sum(e.commanded_native != e.effective_native))
        self.assertGreater(floor_hits, 0)       # the real data does exercise the rules

    def test_wrong_limits_are_detected(self):
        """Negative control: a representation with the wrong floor/cap must fail the echo check."""
        from dataset import load_episode, load_manifest, load_scenario_meta
        r = [x for x in load_manifest(SAMPLE) if x["profile"] == "moderate"][0]
        ep = load_episode(SAMPLE / r["path"], keys=["obs", "action_continuous", "action_discrete"])
        meta = load_scenario_meta(SAMPLE, r["scenario"])
        wrong = ActionRepresentation(meta, min_static_pressure=0.0)
        res = wrong.verify_against_echo(ep["action_continuous"], ep["action_discrete"], ep["obs"][1:])
        self.assertTrue(any("static_pressure" in k for k in res["violations"]), res)


if __name__ == "__main__":
    unittest.main()
