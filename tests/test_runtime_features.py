"""Runtime feature whitelist: allowed inputs pass, privileged ones are rejected."""
import json
import unittest
from collections import defaultdict

from tests._paths import HAVE_SAMPLE, SAMPLE
from runtime_features import PrivilegedFeatureError, RuntimeFeatureSchema, validate_runtime_features, public_metadata

S = RuntimeFeatureSchema.load()
FIELDS = ["VAV_1_zone_air_temperature_sensor", "num_occupants", "hod_sin_000"]


class TestWhitelist(unittest.TestCase):
    def test_allowed(self):
        ok = ["obs", "obs:VAV_1_zone_air_temperature_sensor", "obs:num_occupants", "action_continuous",
              "action_discrete", "effective_action", "meta:num_ahus", "meta:zones.obs_temp", "meta:schedule.holidays",
              "meta:zone_area_m2", "meta:reward_config"]
        decisions = validate_runtime_features(ok, field_names=FIELDS)
        self.assertTrue(all(d.allowed for d in decisions))

    def test_privileged_rejected(self):
        bad = ["zone_info", "zone_info:average_occupancy", "zone_info:zone_air_temperature", "ahu_info", "boiler_info",
               "heat_pump_info", "tank_state", "tank_state:burner_heat_rate_w", "reward", "reward_response",
               "calendar", "calendar:month", "calendar:day", "calendar:is_workday", "sim_timestamp", "reward_timestamp",
               "episode_json", "knobs", "perturbed", "timing_ms", "done", "index",
               "meta:privileged", "meta:privileged.episode_start", "meta:scenario", "meta:weather", "meta:building",
               "meta:schema_version"]
        for f in bad:
            with self.subTest(feature=f):
                with self.assertRaises(PrivilegedFeatureError):
                    validate_runtime_features([f])
                with self.assertRaises(PrivilegedFeatureError):           # also when mixed with allowed ones
                    validate_runtime_features(["obs", "action_continuous", f])

    def test_unknown_rejected(self):
        for f in ["", "foo", "meta:", "meta:not_a_key", "obs:VAV_9_not_a_field", "zone_occupancy"]:
            with self.subTest(feature=f):
                with self.assertRaises(PrivilegedFeatureError):
                    validate_runtime_features([f], field_names=FIELDS)

    def test_restricted_requires_opt_in(self):
        with self.assertRaises(PrivilegedFeatureError):
            validate_runtime_features(["baseline_continuous"])
        validate_runtime_features(["baseline_continuous", "baseline_discrete"], allow_restricted=True)

    def test_error_lists_every_offender(self):
        with self.assertRaises(PrivilegedFeatureError) as cm:
            validate_runtime_features(["obs", "zone_info", "calendar:month"])
        self.assertIn("zone_info", str(cm.exception))
        self.assertIn("calendar:month", str(cm.exception))

    def test_public_metadata_filter(self):
        meta = {"num_ahus": 3, "privileged": {"episode_start": "2023-01-22"}, "scenario": "b/w", "weather": "w",
                "building": "b", "schema_version": "x", "zones": {}, "surprise_new_key": 1}
        pub = public_metadata(meta)
        self.assertEqual(set(pub), {"num_ahus", "zones"})
        self.assertEqual(S.unclassified_meta_keys(meta), ["surprise_new_key"])


@unittest.skipUnless(HAVE_SAMPLE, "data/phase1_sample not present")
class TestRealMetadata(unittest.TestCase):
    def setUp(self):
        self.metas = {p.parent.name: json.loads(p.read_text()) for p in (SAMPLE / "scenarios").glob("*/meta.json")}

    def test_every_meta_key_classified(self):
        for name, m in self.metas.items():
            self.assertEqual(S.unclassified_meta_keys(m), [], name)

    def test_public_metadata_has_no_episode_information(self):
        """Public metadata must be identical across the weather scenarios of a building."""
        by_building = defaultdict(list)
        for name, m in self.metas.items():
            by_building[m["building"]].append(json.dumps(public_metadata(m), sort_keys=True))
        for b, blobs in by_building.items():
            self.assertGreater(len(blobs), 1)
            self.assertEqual(len(set(blobs)), 1, f"{b}: public metadata differs across weathers")

    def test_no_date_strings_in_public_metadata(self):
        import re
        date = re.compile(r"20\d\d-\d\d-\d\d")
        for name, m in self.metas.items():
            self.assertIsNone(date.search(json.dumps(public_metadata(m))), name)
            self.assertIsNotNone(date.search(json.dumps(m["privileged"])), name)   # the raw meta does contain dates


if __name__ == "__main__":
    unittest.main()
