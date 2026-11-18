"""AlignedWindows on the real Phase 1 sample: targets equal the privileged physics they claim to align with."""
import unittest

import numpy as np

from tests._paths import HAVE_SAMPLE, SAMPLE


@unittest.skipUnless(HAVE_SAMPLE, "data/phase1_sample not present")
class TestRealAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dataset import AlignedWindows, WindowSpec, load_manifest
        rows = [r for r in load_manifest(SAMPLE) if r["building"] in ("building_train_8", "building_train_2")][:4]
        cls.spec = WindowSpec(horizon=12, context=3, privileged_targets=("zone_info", "tank_state", "ahu_info"))
        cls.ds = AlignedWindows.from_root(SAMPLE, rows, cls.spec)

    def test_sensor_targets_equal_privileged(self):
        from building_meta import ObservationCodec
        rng = np.random.default_rng(0)
        for i in rng.choice(len(self.ds), 300, replace=False):
            it = self.ds[int(i)]
            meta = self.ds.metas[it["index"]["scenario"]]
            oc = ObservationCodec(meta)
            fields = self.ds._layout[it["index"]["scenario"]]["fields"]
            zi = it["targets"]["privileged"]["zone_info"].astype(np.float64)
            # zone temperature target (obs[t+2]) == zone_info[t].zone_air_temperature
            zt = it["targets"]["sensors"]["zone_air_temperature"] * oc.std[fields["zone_air_temperature"]] + oc.mean[fields["zone_air_temperature"]]
            names = list(meta["observation_field_names"])
            order = [names[j].split("_zone")[0] for j in fields["zone_air_temperature"]]
            col = [meta["zones"]["ids"].index(v) for v in order]
            np.testing.assert_allclose(zt, zi[:, col, 2], atol=1e-3)
            # hws target (obs[t+1]) == tank_state[t+1]
            np.testing.assert_allclose(it["targets"]["sensors"]["hws_supply_water_temperature"][:, 0],
                                       it["targets"]["privileged"]["tank_state"][:, 0], atol=1e-3)
            # occupancy target (obs[t+2]) == sum zone_info[t].average_occupancy (c = 0)
            np.testing.assert_allclose(it["targets"]["sensors"]["num_occupants"][:, 0], zi[:, :, 5].sum(1), atol=1e-4)
            # AHU flow target (obs[t+1]) == sum of zone flows of transition t, per AHU
            fl = it["targets"]["sensors"]["ahu_supply_air_flow"]
            per_ahu = np.stack([zi[:, g, 4].sum(1) for g in meta["ahus"]["zones"]], 1)
            flow_names = [names[j] for j in self.ds._layout[it["index"]["scenario"]]["fields"]["ahu_supply_air_flow"]]
            for a, ahu in enumerate(meta["ahus"]["ids"]):
                j = flow_names.index(f"ahs_{ahu}_supply_air_flowrate_sensor")
                np.testing.assert_allclose(fl[:, j], per_ahu[:, a], atol=1e-3)

    def test_runtime_item_has_no_privileged(self):
        it = self.ds.runtime_item(0)
        self.assertEqual(set(it["runtime_inputs"]), {"obs", "action_history_continuous", "action_history_discrete",
                                                     "action_history_mask", "effective_action_history"})
        self.assertNotIn("targets", it)


if __name__ == "__main__":
    unittest.main()
