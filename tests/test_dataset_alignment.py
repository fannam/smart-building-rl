"""AlignedWindows temporal contract on synthetic episodes with self-describing values.

obs[i, j] = 1000*i + j, reward[t] = t, action_continuous[t, :] = t, so every slice
can be checked for its exact absolute index."""
import unittest

import numpy as np

from tests import _paths  # noqa: F401
from dataset import AlignedWindows, SensorAlignment, WindowSpec

FIELDS = ["VAV_1_zone_air_temperature_sensor", "VAV_2_zone_air_temperature_sensor",
          "hws_supply_water_temperature_sensor", "ahs_ahu_1_outside_air_temperature_sensor",
          "num_occupants", "hod_sin_000"]
ALIGN = SensorAlignment({"schema_version": "test", "groups": [
    {"name": "zone_air_temperature", "pattern": r"^VAV_\d+_zone_air_temperature_sensor$", "kind": "physical_state", "obs_offset": 2, "status": "verified_exact"},
    {"name": "hws_supply_water_temperature", "pattern": r"^hws_supply_water_temperature_sensor$", "kind": "physical_state", "obs_offset": 1, "status": "verified_exact"},
    {"name": "same_step", "pattern": r"^num_occupants$", "kind": "physical_state", "obs_offset": 0, "status": "verified_exact"},
    {"name": "outside_air_temperature", "pattern": r"^ahs_ahu_\d+_outside_air_temperature_sensor$", "kind": "exogenous", "obs_offset": None, "status": "verified_exact"},
    {"name": "unverified", "pattern": r"^hod_sin_000$", "kind": "derived", "obs_offset": 1, "status": "unknown"},
]})
META = {"observation_field_names": FIELDS}


def episode(T, eid="ep0", scenario="b/w_1", building="b", nc=3, nd=2):
    D = len(FIELDS)
    obs = (1000.0 * np.arange(T + 1)[:, None] + np.arange(D)[None, :]).astype(np.float64)
    arrays = {"obs": obs, "reward": np.arange(T, dtype=np.float64),
              "action_continuous": np.repeat(np.arange(T, dtype=np.float32)[:, None], nc, 1),
              "action_discrete": (np.arange(T)[:, None] % 2 * np.ones((1, nd))).astype(np.int8),
              "tank_state": np.arange(T + 1, dtype=np.float64)[:, None] * np.ones((1, 6)),
              "zone_info": np.arange(T, dtype=np.float64)[:, None, None] * np.ones((1, 2, 6))}
    return {"episode_id": eid, "scenario": scenario, "building": building, "arrays": arrays}


def make(T=20, H=4, C=1, groups=("zone_air_temperature", "hws_supply_water_temperature"), priv=(), eps=None, **kw):
    spec = WindowSpec(horizon=H, context=C, sensor_targets=groups, privileged_targets=priv,
                      effective_actions=False, **kw)
    eps = eps or [episode(T)]
    return AlignedWindows(eps, {e["scenario"]: META for e in eps}, spec, alignment=ALIGN)


class TestAlignedWindows(unittest.TestCase):
    def test_valid_range_and_count(self):
        T, H = 20, 4
        ds = make(T, H)
        self.assertEqual(ds.max_offset, 2)
        t0s = [t0 for _, t0 in ds.index]
        self.assertEqual(t0s[0], 0)
        self.assertEqual(t0s[-1], T - H - 2 + 1)          # last transition t0+H-1 = T-2 needs obs[T]
        self.assertEqual(len(ds), T - H - 2 + 2)

    def test_first_transition(self):
        it = make()[0]
        self.assertEqual(it["index"]["t0"], 0)
        np.testing.assert_array_equal(it["runtime_inputs"]["obs"][:, 0], [0.0])        # obs[0]
        self.assertFalse(it["runtime_inputs"]["action_history_mask"][0])               # a[-1] does not exist
        np.testing.assert_array_equal(it["runtime_inputs"]["action_history_continuous"], 0.0)
        np.testing.assert_array_equal(it["targets"]["sensors"]["zone_air_temperature"][0], [2000.0, 2001.0])
        np.testing.assert_array_equal(it["targets"]["sensors"]["hws_supply_water_temperature"][0], [1002.0])

    def test_middle_transition(self):
        ds = make(T=30, H=5, C=3)
        i = [j for j, (_, t0) in enumerate(ds.index) if t0 == 10][0]
        it = ds[i]
        np.testing.assert_array_equal(it["runtime_inputs"]["obs"][:, 0], [8000, 9000, 10000])   # obs[8..10]
        np.testing.assert_array_equal(it["runtime_inputs"]["action_history_continuous"][:, 0], [7, 8, 9])
        self.assertTrue(it["runtime_inputs"]["action_history_mask"].all())
        np.testing.assert_array_equal(it["actions"]["continuous"][:, 0], [10, 11, 12, 13, 14])
        np.testing.assert_array_equal(it["targets"]["reward"], [10, 11, 12, 13, 14])
        # zone sensor of transition t is obs[t+2]; hws of transition t is obs[t+1]
        np.testing.assert_array_equal(it["targets"]["sensors"]["zone_air_temperature"][:, 0], [12000, 13000, 14000, 15000, 16000])
        np.testing.assert_array_equal(it["targets"]["sensors"]["hws_supply_water_temperature"][:, 0], [11002, 12002, 13002, 14002, 15002])

    def test_last_valid_transition(self):
        T, H = 20, 4
        ds = make(T, H)
        it = ds[len(ds) - 1]
        t0 = it["index"]["t0"]
        self.assertEqual(t0 + H - 1, T - 2)                       # last transition with an obs[t+2] target
        self.assertEqual(it["targets"]["sensors"]["zone_air_temperature"][-1, 0], 1000.0 * T)   # obs[T]
        self.assertEqual(it["targets"]["obs_window"][-1, 0], 1000.0 * T)
        self.assertEqual(it["targets"]["reward"][-1], T - 2)

    def test_short_episode(self):
        ds = make(eps=[episode(4, "short"), episode(20, "long")], H=4)
        self.assertEqual(ds.skipped_short, ["short"])               # T=4 < H + L - 1 = 5
        self.assertTrue(all(e == 1 for e, _ in ds.index))
        ds2 = make(eps=[episode(5, "fits")], H=4)                    # exactly one window
        self.assertEqual(len(ds2), 1)

    def test_shapes_h_plus_2(self):
        H = 6
        it = make(T=30, H=H)[3]
        D = len(FIELDS)
        self.assertEqual(it["targets"]["obs_window"].shape, (H + 2, D))
        self.assertEqual(it["actions"]["continuous"].shape, (H, 3))
        self.assertEqual(it["actions"]["discrete"].shape, (H, 2))
        self.assertEqual(it["targets"]["reward"].shape, (H,))
        t0 = it["index"]["t0"]
        np.testing.assert_array_equal(it["targets"]["obs_window"][:, 0], 1000.0 * np.arange(t0, t0 + H + 2))
        # with only offset-1 targets the window is H+1 (no unnecessary truncation)
        ds1 = make(T=30, H=H, groups=("hws_supply_water_temperature",))
        self.assertEqual(ds1[0]["targets"]["obs_window"].shape, (H + 1, D))
        self.assertEqual(len(ds1), 30 - H - 1 + 2)          # t0 in [0, T-H-L+1]

    def test_alignment_actions_rewards_obs(self):
        ds = make(T=40, H=7, C=2)
        for i in range(len(ds)):
            it = ds[i]
            t0 = it["index"]["t0"]
            t = t0 + np.arange(7)
            np.testing.assert_array_equal(it["actions"]["continuous"][:, 0], t)
            np.testing.assert_array_equal(it["targets"]["reward"], t)
            np.testing.assert_array_equal(it["targets"]["sensors"]["zone_air_temperature"][:, 1], 1000.0 * (t + 2) + 1)
            np.testing.assert_array_equal(it["targets"]["obs_window"][0], it["runtime_inputs"]["obs"][-1])

    def test_no_cross_episode(self):
        eps = [episode(15, "a", "s/x_1", "b1"), episode(12, "b", "s/x_2", "b1")]
        ds = make(eps=eps, H=3)
        for i in range(len(ds)):
            e, t0 = ds.index[i]
            T = ds.episodes[e]["T"]
            self.assertLessEqual(t0 + 3 - 1 + ds.max_offset, T)
            it = ds[i]
            self.assertEqual(it["index"]["episode_id"], eps[e]["episode_id"])
            self.assertLessEqual(it["targets"]["obs_window"][-1, 0], 1000.0 * T)

    def test_different_offsets(self):
        it = make(T=20, H=3, groups=("zone_air_temperature", "hws_supply_water_temperature", "same_step"))[4]
        t0 = it["index"]["t0"]
        t = t0 + np.arange(3)
        s = it["targets"]["sensors"]
        np.testing.assert_array_equal(s["zone_air_temperature"][:, 0], 1000.0 * (t + 2))
        np.testing.assert_array_equal(s["hws_supply_water_temperature"][:, 0], 1000.0 * (t + 1) + 2)
        np.testing.assert_array_equal(s["same_step"][:, 0], 1000.0 * t + 4)

    def test_unverified_and_unindexed_groups_rejected(self):
        with self.assertRaises(ValueError):
            make(groups=("unverified",))
        with self.assertRaises(ValueError):
            make(groups=("outside_air_temperature",))       # time-indexed, no transition offset
        ds = make(groups=("unverified",), allow_unverified_sensors=True)
        self.assertEqual(ds.offsets["unverified"], 1)

    def test_privileged_only_on_request(self):
        it = make()[0]
        self.assertNotIn("privileged", it["targets"])
        it = make(priv=("tank_state", "zone_info"))[2]
        t0 = it["index"]["t0"]
        np.testing.assert_array_equal(it["targets"]["privileged"]["tank_state"][:, 0], t0 + 1 + np.arange(4))   # tank_state[t+1]
        np.testing.assert_array_equal(it["targets"]["privileged"]["zone_info"][:, 0, 0], t0 + np.arange(4))    # zone_info[t]
        rt = make(priv=("zone_info",)).runtime_item(0)
        self.assertEqual(set(rt), {"runtime_inputs", "actions", "index"})
        with self.assertRaises(ValueError):
            WindowSpec(privileged_targets=("obs",))
        with self.assertRaises(ValueError):
            WindowSpec(privileged_targets=("sim_timestamp",))

    def test_torch_collate(self):
        import torch
        ds = make(T=30, H=4, C=2, priv=("zone_info",))
        loader = torch.utils.data.DataLoader(ds.as_torch(), batch_size=5)
        b = next(iter(loader))
        self.assertEqual(tuple(b["targets"]["obs_window"].shape), (5, 6, len(FIELDS)))
        self.assertEqual(tuple(b["runtime_inputs"]["obs"].shape), (5, 2, len(FIELDS)))
        self.assertEqual(tuple(b["targets"]["privileged"]["zone_info"].shape), (5, 4, 2, 6))
        rb = next(iter(torch.utils.data.DataLoader(ds.as_torch(runtime_only=True), batch_size=5)))
        self.assertNotIn("targets", rb)


if __name__ == "__main__":
    unittest.main()
