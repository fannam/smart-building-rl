"""Leave-one-building-out folds: determinism, leakage detection, statistics provenance."""
import json
import unittest

from tests._paths import HAVE_SAMPLE, ROOT, SAMPLE
from generalization import (FittedStatistics, LeakageError, assert_provenance, make_lobo_folds, manifest_digest,
                            split_rows, validate_fold)

B = ["building_train_8", "building_train_1", "building_train_7", "building_train_2", "building_train_5", "building_train_4"]


def rows():
    return [{"episode_id": f"{b}__w{w}__p", "scenario": f"{b}/w_{w}", "building": b, "path": f"episodes/{b}_{w}.npz",
             "steps": 10} for b in B for w in range(5)]


class TestFolds(unittest.TestCase):
    def test_deterministic_and_complete(self):
        f1, f2 = make_lobo_folds(B), make_lobo_folds(list(reversed(B)))
        self.assertEqual(f1, f2)
        self.assertEqual(len(f1), 6)
        self.assertEqual(sorted(f["test_building"] for f in f1), sorted(B))
        for f in f1:
            self.assertNotIn(f["test_building"], f["train_buildings"])
            self.assertEqual(len(f["train_buildings"]), 5)

    def test_split_keeps_all_weathers_together(self):
        for f in make_lobo_folds(B):
            tr, te = split_rows(rows(), f)
            self.assertEqual(len(te), 5)
            self.assertEqual({r["building"] for r in te}, {f["test_building"]})
            validate_fold(f, tr, te)

    def test_leakage_detected(self):
        f = make_lobo_folds(B)[0]
        tr, te = split_rows(rows(), f)
        cases = {
            "test episode in train": (tr + [te[0]], te),
            "other building in test": (tr, te + [tr[0]]),
            "duplicated episode id": (tr + [dict(tr[0], building=f["train_buildings"][0])], te + [dict(tr[0], building=f["test_building"])]),
            "empty test": (tr, []),
        }
        for name, (a, b) in cases.items():
            with self.subTest(name):
                with self.assertRaises(LeakageError):
                    validate_fold(f, a, b)
        bad = dict(f, train_buildings=f["train_buildings"] + [f["test_building"]])
        with self.assertRaises(LeakageError):
            validate_fold(bad, tr, te)

    def test_provenance_detects_held_out_statistics(self):
        f = make_lobo_folds(B)[0]
        tr, te = split_rows(rows(), f)
        good = FittedStatistics({}, {"buildings": sorted({r["building"] for r in tr}), "episode_ids": [],
                                     "manifest_digest": manifest_digest(tr)})
        assert_provenance(good, f, tr)
        leaked = FittedStatistics({}, {"buildings": sorted({r["building"] for r in tr + te}), "episode_ids": [],
                                       "manifest_digest": manifest_digest(tr + te)})
        with self.assertRaises(LeakageError):
            assert_provenance(leaked, f, tr)
        stale = FittedStatistics({}, dict(good.provenance, manifest_digest="0" * 16))
        with self.assertRaises(LeakageError):
            assert_provenance(stale, f, tr)


@unittest.skipUnless(HAVE_SAMPLE and (ROOT / "results/phase1_6/summary.json").exists(), "LOBO outputs not generated")
class TestSavedFolds(unittest.TestCase):
    def test_saved_config_and_manifests(self):
        cfg = json.loads((ROOT / "configs/leave_one_building_out.json").read_text())
        from dataset import load_manifest
        all_rows = [r for r in load_manifest(SAMPLE) if r["profile"] in cfg["profiles"]]
        self.assertEqual(cfg["folds"], make_lobo_folds({r["building"] for r in all_rows}))
        for f in cfg["folds"]:
            d = ROOT / "results/phase1_6" / f"fold_{f['fold']}_{f['test_building']}"
            tr = [json.loads(x) for x in (d / "manifest_train.jsonl").read_text().splitlines()]
            te = [json.loads(x) for x in (d / "manifest_test.jsonl").read_text().splitlines()]
            validate_fold(f, tr, te)
            self.assertEqual(len(tr) + len(te), len(all_rows))
            stats = FittedStatistics.from_json(json.loads((d / "train_statistics.json").read_text()))
            assert_provenance(stats, f, tr)
            self.assertNotIn(f["test_building"], stats.provenance["buildings"])
            self.assertTrue(set(stats.provenance["episode_ids"]).isdisjoint({r["episode_id"] for r in te}))


if __name__ == "__main__":
    unittest.main()
