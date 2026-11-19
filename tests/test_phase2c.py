"""Phase 2C: zone-subset interventions, branch-point planning, resumable collection, split integrity,
and (simulator) reproducibility of the collector."""
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from tests._paths import HAVE_SAMPLE, ROOT, SAMPLE

CFG_PATH = ROOT / "configs/phase2c/collection.json"


def load_cfg():
    return json.loads(CFG_PATH.read_text())


class TestConfig(unittest.TestCase):
    def test_approved_config_expands_to_20_arms(self):
        from counterfactual.phase2c import expand_arms, validate_config
        cfg = load_cfg()
        self.assertEqual(validate_config(cfg), [])
        arms = expand_arms(cfg)
        self.assertEqual(len(arms) + 1, 20)
        self.assertEqual(len({a.name for a in arms}), len(arms))
        holds = sorted(a.duration for a in arms if a.kind == "subset_set")
        self.assertEqual(holds, [3, 3, 6, 6, 12, 12])                       # 15 / 30 / 60 min for E1 and E2

    def test_invalid_configs_rejected(self):
        from counterfactual.phase2c import validate_config
        cfg = load_cfg()
        bad = dict(cfg, zone_subset_fraction=0.5)
        self.assertTrue(validate_config(bad))
        bad = dict(cfg, min_spacing_steps=10)
        self.assertTrue(validate_config(bad))
        bad = dict(cfg, arms=[{"group": "X", "family": "sat", "kind": "subset_set", "levels": [1], "durations": [3]}])
        self.assertTrue(validate_config(bad))


@unittest.skipUnless(HAVE_SAMPLE, "data/phase1_sample not present")
class TestSubsetInterventions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dataset import load_episode, load_scenario_meta
        from effective_action import ActionRepresentation
        cls.meta = load_scenario_meta(SAMPLE, "building_train_2/atlanta_3")
        cls.rep = ActionRepresentation(cls.meta)
        ep = load_episode(SAMPLE / "episodes" / "building_train_2__atlanta_3__baseline__s0__e0.npz")
        run = ep["action_discrete"][:, cls.meta["ahus"]["act_run"]]
        t = int(np.flatnonzero(run.all(1))[50])                             # all AHUs on
        cls.c, cls.d = ep["action_continuous"][t:t + 36], ep["action_discrete"][t:t + 36]
        cls.V = cls.meta["num_zones"]

    def spec(self, **kw):
        from counterfactual.phase2c import ArmSpec
        base = dict(name="t", group="E2", family="reheat_valve", kind="subset_set", duration=6, levels=(0.3, 0.6, 1.0))
        base.update(kw)
        return ArmSpec(**base)

    def test_subset_size_members_and_hold(self):
        from counterfactual.phase2c import apply_arm
        from counterfactual.interventions import indices
        c2, d2, info = apply_arm(self.spec(), self.c, self.d, self.meta, self.rep, np.random.default_rng(1), 0.3)
        m = info["subset_mask"]
        self.assertEqual(m.sum(), round(0.3 * self.V))
        idx = indices(self.meta, "reheat_valve")
        changed_cols = np.flatnonzero((c2 != self.c).any(0))
        self.assertTrue(set(changed_cols) <= set(idx[m]))                    # only member valves change
        np.testing.assert_array_equal(c2[6:], self.c[6:])                     # hold respected (6 steps)
        np.testing.assert_array_equal(d2, self.d)
        nat = self.rep.codec.to_native(c2[:6].astype(np.float64))[:, idx[m]]
        np.testing.assert_allclose(nat, info["value"], atol=1e-5)             # members held at the level
        self.assertIn(info["value"], (0.3, 0.6, 1.0))

    def test_deterministic_per_seed_and_distinct_across_arms(self):
        from counterfactual.phase2c import apply_arm, arm_rng
        a = apply_arm(self.spec(), self.c, self.d, self.meta, self.rep, arm_rng(0, "s", 100, 1), 0.3)
        b = apply_arm(self.spec(), self.c, self.d, self.meta, self.rep, arm_rng(0, "s", 100, 1), 0.3)
        o = apply_arm(self.spec(), self.c, self.d, self.meta, self.rep, arm_rng(0, "s", 100, 2), 0.3)
        np.testing.assert_array_equal(a[0], b[0]); np.testing.assert_array_equal(a[2]["subset_mask"], b[2]["subset_mask"])
        self.assertFalse(np.array_equal(a[2]["subset_mask"], o[2]["subset_mask"]))

    def test_owner_on_only_excludes_zones_of_stopped_ahus(self):
        from counterfactual.phase2c import apply_arm, zones_with_ahu_on
        d = self.d.copy()
        run = np.asarray(self.meta["ahus"]["act_run"])
        d[:, run[0]] = 0                                                      # AHU 0 off
        on = zones_with_ahu_on(self.meta, d[0])
        self.assertFalse(on.all())
        for s in range(5):
            _, _, info = apply_arm(self.spec(), self.c, d, self.meta, self.rep, np.random.default_rng(s), 0.3, owner_on_only=True)
            self.assertFalse((info["subset_mask"] & ~on).any())
        d[:, run] = 0                                                         # all off -> flagged no-op
        c2, _, info = apply_arm(self.spec(), self.c, d, self.meta, self.rep, np.random.default_rng(0), 0.3, owner_on_only=True)
        self.assertTrue(info["no_qualifying"]); np.testing.assert_array_equal(c2, self.c)

    def test_effective_changes_for_damper_increase(self):
        from counterfactual.phase2c import apply_arm
        from counterfactual.interventions import indices
        sp = self.spec(family="damper", group="E1", levels=(0.9,))
        c2, d2, info = apply_arm(sp, self.c, self.d, self.meta, self.rep, np.random.default_rng(3), 0.3)
        idx = indices(self.meta, "damper")[info["subset_mask"]]
        e0 = self.rep.effective(self.c, self.d).effective_native[:6, idx]
        e1 = self.rep.effective(c2, d2).effective_native[:6, idx]
        self.assertTrue(((e1 - e0) > 1e-6).any())                             # commanded change is effective
        np.testing.assert_allclose(e1, 0.9, atol=1e-5)

    def test_noop_control_never_changes_effective_action(self):
        from counterfactual.phase2c import apply_arm
        sp = self.spec(family="damper", group="NOOP", kind="subset_set_qualified_below", level=0.05, levels=(),
                       qualify_below=0.1001, duration=12)
        for s in range(5):
            c2, d2, info = apply_arm(sp, self.c, self.d, self.meta, self.rep, np.random.default_rng(s), 0.3)
            e0, e1 = self.rep.effective(self.c, self.d), self.rep.effective(c2, d2)
            np.testing.assert_allclose(e1.effective_native, e0.effective_native, atol=1e-6)

    def test_flip_and_qualified_decrease(self):
        from counterfactual.phase2c import ArmSpec, apply_arm
        from counterfactual.interventions import indices
        sp = ArmSpec(name="f", group="AHU", family="ahu_run", kind="flip", duration=12)
        _, d2, _ = apply_arm(sp, self.c, self.d, self.meta, self.rep, np.random.default_rng(0), 0.3)
        r = indices(self.meta, "ahu_run")
        np.testing.assert_array_equal(d2[:12, r], 1 - self.d[0, r][None].repeat(12, 0))
        np.testing.assert_array_equal(d2[12:], self.d[12:])
        sp = self.spec(kind="subset_set_qualified", level=0.0, levels=(), qualify_above=0.05, duration=12)
        _, _, info = apply_arm(sp, self.c, self.d, self.meta, self.rep, np.random.default_rng(0), 0.3)
        eff0 = self.rep.effective(self.c[:1], self.d[:1]).effective_native[0][indices(self.meta, "reheat_valve")]
        if info["no_qualifying"]:
            self.assertFalse((eff0 > 0.05).any())
        else:
            self.assertTrue((eff0[info["subset_mask"]] > 0.05).all())


@unittest.skipUnless(HAVE_SAMPLE, "data/phase1_sample not present")
class TestPlanning(unittest.TestCase):
    def test_plan_deterministic_spaced_and_excludes_test_a(self):
        from counterfactual.interventions import step_conditions
        from counterfactual.phase2c import plan_branch_points
        from dataset import load_episode, load_scenario_meta
        cfg = load_cfg()
        sc = "building_train_8/atlanta_1"
        ep = load_episode(SAMPLE / "episodes" / "building_train_8__atlanta_1__baseline__s0__e0.npz")
        cond = step_conditions(ep, load_scenario_meta(SAMPLE, sc))
        excl = json.loads((ROOT / "configs/phase2c/phase2b_branch_points.json").read_text())["branch_points"][sc]
        args = (cond, 10, cfg["stratum_weights"], 12, 36, 49, excl, 48, 0, sc)
        p1, p2 = plan_branch_points(*args, prefer_ahu_on=True), plan_branch_points(*args, prefer_ahu_on=True)
        self.assertEqual(p1, p2)
        self.assertEqual(len(p1), 10)
        t = [x for x, _ in p1]
        self.assertTrue(all(abs(a - b) >= 49 for i, a in enumerate(t) for b in t[i + 1:]))
        self.assertTrue(all(abs(a - u) > 48 for a in t for u in excl))
        p0 = plan_branch_points(*args, prefer_ahu_on=False)
        self.assertGreaterEqual(sum(cond["ahu_on"][x] for x, _ in p1), sum(cond["ahu_on"][x] for x, _ in p0))

    def test_branch_complete_rejects_corrupt_files(self):
        from counterfactual.phase2c import branch_complete
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "x.npz")
            with open(p, "wb") as fh:
                fh.write(b"not a zip")
            self.assertFalse(branch_complete(p, 20))
            self.assertFalse(branch_complete(os.path.join(d, "missing.npz"), 20))
        finally:
            shutil.rmtree(d)


class TestSplits(unittest.TestCase):
    def test_scenario_days_never_straddle_and_test_a_respected(self):
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from phase2c_splits import assign, check
        rows = [{"branch_id": f"b{i}", "scenario": sc, "building": sc.split("/")[0], "weather": sc.split("/")[1], "t_b": t,
                 "stratum": "x"} for i, (sc, t) in enumerate([("b1/atlanta_1", 300), ("b1/atlanta_1", 400), ("b1/atlanta_1", 900),
                                                               ("b1/denver_2", 300), ("b2/phoenix_1", 1500)])]
        asg = assign(rows)
        self.assertEqual(check(asg, {}, 48), [])
        self.assertEqual(asg["b0"]["split"], asg["b1"]["split"])            # same scenario-day -> same split
        self.assertEqual(asg["b3"]["split"], "test_b")                       # held-out weather
        self.assertTrue(check(asg, {"b1/atlanta_1": [330]}, 48))            # near a Test-A point -> flagged


@unittest.skipUnless(HAVE_SAMPLE and os.environ.get("RUN_SIM_TESTS", "1") == "1", "simulator tests disabled")
class TestCollectorReproducibility(unittest.TestCase):
    """Small real collection: bit-exact nominal replay, identical results on rerun into a fresh dir,
    resume skips complete branch points, identical-command arms equal the nominal arm."""

    def collect(self, out, cfg):
        from counterfactual.phase2c import collect_scenario
        excl = json.loads((ROOT / "configs/phase2c/phase2b_branch_points.json").read_text())["branch_points"]
        cwd = os.getcwd()
        try:
            return collect_scenario("building_train_8/atlanta_1", cfg, out, str(SAMPLE), 1e18, excl["building_train_8/atlanta_1"],
                                    log=lambda m: None)
        finally:
            os.chdir(cwd)

    def test_collect_resume_and_reproduce(self):
        cfg = load_cfg()
        cfg["n_points_per_scenario"] = 1
        cfg["arms"] = [a for a in cfg["arms"] if a["family"] in ("reheat_valve",) and a["kind"] == "subset_set"] + \
                      [{"group": "COPY", "family": "damper", "kind": "subset_set_qualified", "level": 0.1, "qualify_above": 5.0,
                        "duration": 3}]                                       # never qualifies -> command-identical arm
        cfg.pop("arms_per_branch", None)
        d1, d2 = tempfile.mkdtemp(), tempfile.mkdtemp()
        try:
            r1 = self.collect(d1, cfg)
            self.assertEqual(r1["collected"], 1)
            self.assertEqual(r1["replay_obs_max_abs"], 0.0)
            self.assertEqual(self.collect(d1, cfg)["collected"], 0)                # resume: nothing duplicated
            self.collect(d2, cfg)
            f1 = sorted(os.listdir(os.path.join(d1, "branches")))
            f2 = sorted(os.listdir(os.path.join(d2, "branches")))
            self.assertEqual(f1, f2)
            z1, z2 = np.load(os.path.join(d1, "branches", f1[0])), np.load(os.path.join(d2, "branches", f2[0]))
            self.assertTrue(bool(z1["nominal_bit_exact"][0]))
            for k in z1.files:
                if k != "branch_wall_s":
                    np.testing.assert_array_equal(z1[k], z2[k], err_msg=k)    # bit-identical across independent runs
            # test_identical_commands_reproduce_nominal: the copied arm equals the nominal arm exactly
            last = z1["arm_ac"].shape[1] - 1
            self.assertFalse(bool(z1["simulated"][0, last]))
            np.testing.assert_array_equal(z1["zone_temp"][0, last], z1["zone_temp"][0, 0])
        finally:
            shutil.rmtree(d1); shutil.rmtree(d2)

    def test_identical_commands_reproduce_nominal(self):
        """Justifies copying command-identical arms: a re-simulated identical arm is bit-identical."""
        from counterfactual.branch import Branch
        from counterfactual.simrun import make_action, max_divergence, policy_action, record_step, setup, stack
        cwd = os.getcwd()
        try:
            env, smeta, pol, spec, obs = setup("building_train_8/atlanta_1")
            rooms = smeta["zones"]["room_ids"]
            for _ in range(20):
                obs = record_step(env, policy_action(pol, obs, spec), rooms)["obs"]
            acts = []
            br = Branch.take(env)
            e0 = br.spawn()
            o = obs
            nom = []
            for _ in range(6):
                a = policy_action(pol, o, spec)
                acts.append(a); r = record_step(e0, a, rooms); nom.append(r); o = r["obs"]
            e1 = br.spawn()
            rep = stack([record_step(e1, make_action(a["continuous_action"], a["discrete_action"], spec), rooms) for a in acts])
            self.assertTrue(max_divergence(rep, stack(nom))["bit_exact"])
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()


HAVE_TORCH = __import__("importlib").util.find_spec("torch") is not None


@unittest.skipUnless(HAVE_TORCH, "torch not installed (collection-only environment)")
@unittest.skipUnless(HAVE_SAMPLE and os.environ.get("RUN_SIM_TESTS", "1") == "1", "simulator tests disabled")
class TestPairedTraining(unittest.TestCase):
    """Paired objective on one REAL collected branch group (b8, heating, 3 arms + nominal)."""

    @classmethod
    def setUpClass(cls):
        import torch
        from counterfactual.paired_data import BranchGroups, collate_groups
        from counterfactual.phase2c import collect_scenario
        cfg = load_cfg()
        cfg["n_points_per_scenario"] = 1
        cfg["stratum_weights"] = {"occupied_heating": 1}
        cfg["arms"] = [{"group": "E2", "family": "reheat_valve", "kind": "add", "value": 0.6, "duration": 36},
                       {"group": "E1", "family": "damper", "kind": "add", "value": 0.5, "duration": 36},
                       {"group": "NOOP", "family": "damper", "kind": "subset_set_qualified_below", "level": 0.05,
                        "qualify_below": 0.1001, "duration": 12}]
        cfg.pop("arms_per_branch", None)
        cls.dir = tempfile.mkdtemp()
        excl = json.loads((ROOT / "configs/phase2c/phase2b_branch_points.json").read_text())["branch_points"]
        cwd = os.getcwd()
        try:
            collect_scenario("building_train_8/fairbanks_8", cfg, cls.dir, str(SAMPLE), 1e18, excl["building_train_8/fairbanks_8"],
                             log=lambda m: None)
        finally:
            os.chdir(cwd)
        bid = os.listdir(os.path.join(cls.dir, "branches"))[0][:-4]
        cls.bg = BranchGroups(cls.dir, [bid], SAMPLE)
        cls.batch = collate_groups([cls.bg[0]])
        torch.manual_seed(0)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir)

    def norm(self):
        import torch
        from dynamics.model import Norm
        return Norm(295, 3, 0.2, 340, 8, 1, 270, 15, torch.ones(4) * 0.05, torch.ones(4) * 0.05)

    def test_arms_share_context_and_differ_only_in_actions(self):
        import torch
        inp = self.batch["inputs"]
        C = inp["zone_temp_obs"].shape[1]
        for k, v in inp.items():
            if k in ("zone_act", "ahu_act", "bldg_act"):
                torch.testing.assert_close(v[:, :C], v[:1, :C].expand_as(v[:, :C]))   # identical executed history
                continue
            torch.testing.assert_close(v, v[:1].expand_as(v))                            # identical context/schedule
        self.assertTrue((inp["zone_act"][1, C:] != inp["zone_act"][0, C:]).any())        # valve arm differs in future actions

    def test_loss_zero_iff_predicted_deltas_match_and_noop_weighting(self):
        import torch
        from dynamics.model import energy_norm_target
        from dynamics.paired import paired_loss
        t = self.batch["targets"]
        w = {"temperature": 1.0, "hws": 0.5, "electricity": 0.5, "gas": 0.5}
        ds = self.bg.delta_stats()
        exact = {"zone_temp": t["zone_temp"] + 7.0, "hws": t["hws"] - 3.0, "energy_norm": energy_norm_target(self.batch) + 0.01}
        l0, _ = paired_loss(exact, self.batch, ds, w)                                   # offsets cancel in deltas
        self.assertLess(float(l0), 1e-8)
        flat = {"zone_temp": t["zone_temp"][:1].expand_as(t["zone_temp"]).clone(), "hws": t["hws"][:1].expand_as(t["hws"]).clone(),
                "energy_norm": energy_norm_target(self.batch)[:1].expand_as(energy_norm_target(self.batch)).clone()}
        l1, p1 = paired_loss(flat, self.batch, ds, w)                                   # zero-response prediction
        self.assertGreater(float(l1), 1e-3)
        self.assertTrue(bool(self.batch["pair"]["noop"][-1]))                          # NOOP arm is flagged

    def test_no_future_observation_or_target_leak(self):
        import copy
        import torch
        from dynamics.model import GRUDynamics, rollout
        m = GRUDynamics(self.norm(), 16, 16, 16).eval()
        b = copy.deepcopy(self.batch)
        with torch.no_grad():
            ref = rollout(m, b)
            for k in b["targets"]:
                b["targets"][k] = torch.full_like(b["targets"][k].double(), float("nan"))
            out = rollout(m, b)
        torch.testing.assert_close(out["zone_temp"], ref["zone_temp"])

    def test_paired_loss_can_learn_a_real_response(self):
        """Overfit one real branch group with L_cf only: the predicted valve/damper response must move from
        ~0 toward the simulator's (action information reaches the zone dynamics)."""
        import torch
        from dynamics.model import GRUDynamics, rollout
        from dynamics.paired import paired_loss
        torch.manual_seed(0)
        m = GRUDynamics(self.norm(), 32, 32, 32)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        ds = self.bg.delta_stats()
        w = {"temperature": 1.0, "hws": 0.5, "electricity": 0.5, "gas": 0.5}
        t = self.batch["targets"]["zone_temp"]
        dz = (t[1] - t[0]).double()
        def gain():
            with torch.no_grad():
                p = rollout(m, self.batch)["zone_temp"]
            dh = (p[1] - p[0]).double()
            return float((dh * dz).sum() / (dz * dz).sum())
        g0 = gain()
        losses = []
        for _ in range(150):
            l, _ = paired_loss(rollout(m, self.batch), self.batch, ds, w)
            opt.zero_grad(); l.backward(); opt.step(); losses.append(float(l))
        self.assertGreater(float((dz ** 2).mean()), 1e-4)                               # the simulator responded
        self.assertLess(losses[-1], 0.3 * losses[0])
        self.assertGreater(gain(), max(0.5, g0 + 0.3))
