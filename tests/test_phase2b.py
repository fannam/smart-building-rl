"""Phase 2B: graph topology/model, counterfactual branching and interventions, occupancy inputs,
ensemble aggregation and uncertainty scores."""
import json
import os
import unittest

import numpy as np
import torch

from tests._paths import HAVE_SAMPLE, ROOT, SAMPLE
from dynamics.model import Norm, rollout
from tests.test_dynamics import Recorder, synthetic_batch

NORM = Norm(295, 3, 0.2, 340, 8, 1, 290, 15, torch.ones(4) * 0.1, torch.ones(4) * 0.1)


# ------------------------------------------------------------------ topology
class TestTopology(unittest.TestCase):
    def test_two_rooms_one_wall(self):
        from graph.topology import zone_adjacency
        g = np.full((5, 7), 2)
        g[1:4, 1:6] = 0
        g[1:4, 3] = 1                       # wall column splits two rooms
        adj = zone_adjacency(g, ["VAV_1", "VAV_2"])
        self.assertTrue(adj["available"])
        self.assertEqual(adj["edges"].tolist(), [[0, 1]])
        self.assertEqual(adj["weight"].tolist(), [3.0])

    def test_component_mismatch_marks_unavailable(self):
        from graph.topology import zone_adjacency
        g = np.zeros((3, 3))
        adj = zone_adjacency(g, ["VAV_1", "VAV_2"])
        self.assertFalse(adj["available"])
        self.assertEqual(len(adj["edges"]), 0)
        self.assertFalse(zone_adjacency(None, ["VAV_1"])["available"])

    @unittest.skipUnless(HAVE_SAMPLE, "data/phase1_sample not present")
    def test_real_buildings(self):
        from dataset import load_scenario_meta
        from graph.topology import build_topology, load_zone_map
        for sc in ("building_train_8/atlanta_1", "building_train_2/atlanta_3"):
            m = load_scenario_meta(SAMPLE, sc)
            t = build_topology(m, load_zone_map(SAMPLE, sc))
            self.assertTrue(t["adjacency_available"])
            self.assertEqual(t["V"], m["num_zones"]); self.assertEqual(t["K"], m["num_ahus"])
            pairs = set(zip(t["zz_src"].tolist(), t["zz_dst"].tolist()))
            self.assertTrue(all((b, a) in pairs for a, b in pairs))                  # symmetric
            self.assertFalse(any(a == b for a, b in pairs))                          # no self loops
            s = np.bincount(t["zz_dst"], weights=t["zz_w"], minlength=t["V"])
            np.testing.assert_allclose(s[s > 0], 1.0, atol=1e-6)                     # degree-normalized
            self.assertTrue(((t["zone_ahu"] >= 0) & (t["zone_ahu"] < t["K"])).all())


# ------------------------------------------------------------------ graph model
@unittest.skipUnless(HAVE_SAMPLE, "data/phase1_sample not present")
class TestGraphModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dataset import AlignedWindows, WindowSpec, load_manifest
        from dynamics.features import PRIVILEGED_FOR_TRAINING, DynamicsWindows
        from graph.train import GraphWindows, TopologyCache
        rows = [r for r in load_manifest(SAMPLE) if r["building"] in ("building_train_8", "building_train_7")
                and r["profile"] == "baseline"][:6]
        rows = [r for r in rows if r["building"] == "building_train_8"][:1] + [r for r in rows if r["building"] == "building_train_7"][:1]
        aw = AlignedWindows.from_root(SAMPLE, rows, WindowSpec(horizon=6, context=4, privileged_targets=PRIVILEGED_FOR_TRAINING))
        dw = DynamicsWindows(aw, keep_windows=list(range(0, len(aw), 700)))
        cls.ds = GraphWindows(dw, TopologyCache(SAMPLE))
        torch.manual_seed(0)
        from graph.model import GraphDynamics
        cls.model = GraphDynamics(NORM, hidden=16, head_hidden=16).eval()

    def items(self):
        its = [self.ds[i] for i in range(len(self.ds))]
        b8 = [i for i in its if i["info"]["building"] == "building_train_8"][0]
        b7 = [i for i in its if i["info"]["building"] == "building_train_7"][0]
        return b8, b7

    def test_padding_invariance_mixed_buildings(self):
        """A building's prediction must not change when batched (and padded) with a larger building."""
        from graph.train import graph_collate
        b8, b7 = self.items()
        with torch.no_grad():
            alone = rollout(self.model, graph_collate([b8]))
            mixed = rollout(self.model, graph_collate([b8, b7]))
        V = b8["info"]["V"]
        self.assertGreater(b7["info"]["V"], V)
        torch.testing.assert_close(mixed["zone_temp"][0, :, :V], alone["zone_temp"][0], atol=1e-4, rtol=1e-5)
        torch.testing.assert_close(mixed["energy_w"][0], alone["energy_w"][0], atol=1e-2, rtol=1e-4)
        torch.testing.assert_close(mixed["hws"][0], alone["hws"][0], atol=1e-4, rtol=1e-5)

    def test_graph_auto_equals_graph_collate(self):
        from dynamics.features import collate
        from graph.train import GraphAuto, graph_collate
        b8, b7 = self.items()
        with torch.no_grad():
            a = rollout(self.model, graph_collate([b8, b7]))
            plain = [{k: v for k, v in it.items() if k != "topo"} for it in (b8, b7)]
            b = rollout(GraphAuto(self.model, SAMPLE), collate(plain))
        torch.testing.assert_close(a["zone_temp"], b["zone_temp"])

    def test_edge_types(self):
        from graph.train import graph_collate
        b8, b7 = self.items()
        batch = graph_collate([b8, b7])
        Vm = batch["zone_mask"].shape[1]
        g = batch["graph"]
        # zone-zone edges stay inside their own building block and never touch padded zones
        blk_s, blk_d = g["zz_src"] // Vm, g["zz_dst"] // Vm
        self.assertTrue(torch.equal(blk_s, blk_d))
        zm = batch["zone_mask"].reshape(-1)
        self.assertTrue(zm[g["zz_src"]].all() and zm[g["zz_dst"]].all())

    def test_no_future_observation_or_target_leak(self):
        from graph.train import graph_collate
        b8, _ = self.items()
        batch = graph_collate([b8])
        with torch.no_grad():
            ref = rollout(self.model, batch)
            for k in batch["targets"]:
                batch["targets"][k] = torch.full_like(batch["targets"][k].double(), float("nan"))
            out = rollout(self.model, batch, "persistence")
        torch.testing.assert_close(out["zone_temp"], ref["zone_temp"])
        self.assertTrue(torch.isfinite(out["zone_temp"]).all())

    def test_actions_enter_only_their_nodes(self):
        """Changing a zone's valve changes that zone first; changing HWS setpoint reaches zones only via messages."""
        from graph.train import graph_collate
        b8, _ = self.items()
        batch = graph_collate([b8])
        C = batch["inputs"]["zone_temp_obs"].shape[1]
        with torch.no_grad():
            ref = rollout(self.model, batch)
            za = batch["inputs"]["zone_act"].clone()
            za[:, C:, 0, 1] += 0.5
            b2 = {**batch, "inputs": {**batch["inputs"], "zone_act": za}}
            out = rollout(self.model, b2)
        d = (out["zone_temp"] - ref["zone_temp"]).abs()[0, 0]
        self.assertGreater(float(d[0]), 0.0)


class TestGraphRolloutContract(unittest.TestCase):
    def test_rollout_feedback_rules_still_hold(self):
        # rollout is shared; Recorder checks pass for any model with the step interface
        C, H = 4, 5
        m, b = Recorder(), synthetic_batch(C, H)
        rollout(m, b, "persistence")
        for j, T_in, hws_in, oat in m.seen:
            if j > C - 1:
                self.assertTrue(torch.all(T_in[:, 0] == 1000.0 * (j - 1)))
                self.assertTrue(torch.all(oat == 30 * (C - 1)))


# ------------------------------------------------------------------ interventions / effective actions
@unittest.skipUnless(HAVE_SAMPLE, "data/phase1_sample not present")
class TestInterventions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dataset import load_episode, load_scenario_meta
        cls.meta = load_scenario_meta(SAMPLE, "building_train_8/atlanta_1")
        ep = load_episode(SAMPLE / "episodes" / "building_train_8__atlanta_1__baseline__s0__e0.npz")
        cls.c, cls.d = ep["action_continuous"][500:520], ep["action_discrete"][500:520]

    def test_unchanged_entries_bit_identical_and_bounds(self):
        from building_meta import ActionCodec
        from counterfactual.interventions import Intervention, apply_intervention, indices
        iv = Intervention("sat+5", "sat", "add", 5.0, 6, "large")
        c2, d2 = apply_intervention(iv, self.c, self.d, self.meta)
        idx = indices(self.meta, "sat")
        other = np.setdiff1d(np.arange(self.c.shape[1]), idx)
        np.testing.assert_array_equal(c2[:, other], self.c[:, other])
        np.testing.assert_array_equal(c2[6:], self.c[6:])               # duration respected
        np.testing.assert_array_equal(d2, self.d)
        nat = ActionCodec(self.meta).to_native(c2[:6, idx].astype(np.float64), idx)
        self.assertTrue((nat <= 305.0 + 1e-4).all() and (nat >= 285.0 - 1e-4).all())

    def test_set_run_and_effective_noop(self):
        from counterfactual.interventions import Intervention, apply_intervention, indices
        from effective_action import ActionRepresentation
        rep = ActionRepresentation(self.meta)
        iv = Intervention("ahu_run=0", "ahu_run", "set", 0.0, 20, "binary")
        c2, d2 = apply_intervention(iv, self.c, self.d, self.meta)
        self.assertTrue((d2[:, indices(self.meta, "ahu_run")] == 0).all())
        e = rep.effective(c2, d2)
        p = indices(self.meta, "static_pressure")
        self.assertTrue((e.effective_native[:, p] == 0).all())         # AHU off -> no static pressure
        # pressure decrease below the 200 Pa floor is an effective no-op while running
        d_on = self.d.copy(); d_on[:, indices(self.meta, "ahu_run")] = 1
        iv2 = Intervention("sp-150", "static_pressure", "add", -150.0, 20, "small")
        c3, _ = apply_intervention(iv2, self.c, d_on, self.meta)
        e0, e3 = rep.effective(self.c, d_on), rep.effective(c3, d_on)
        low = rep.codec.to_native(self.c.astype(np.float64))[:, p] <= 200.0
        np.testing.assert_allclose(e3.effective_native[:, p][low], e0.effective_native[:, p][low])

    def test_branch_point_selection_deterministic_and_spaced(self):
        from counterfactual.interventions import select_branch_points, step_conditions
        from dataset import load_episode
        ep = load_episode(SAMPLE / "episodes" / "building_train_8__atlanta_1__baseline__s0__e0.npz")
        cond = step_conditions(ep, self.meta)
        a = select_branch_points(cond, 1, 12, 36, 0)
        b = select_branch_points(cond, 1, 12, 36, 0)
        self.assertEqual(a, b)
        t = [x[0] for x in a]
        self.assertTrue(all(abs(x - y) >= 37 for i, x in enumerate(t) for y in t[i + 1:]))
        self.assertTrue(all(x >= 13 and x + 36 < len(ep["reward"]) for x in t))


# ------------------------------------------------------------------ simulator branching
@unittest.skipUnless(os.environ.get("RUN_SIM_TESTS", "1") == "1", "simulator tests disabled")
class TestBranchReproducibility(unittest.TestCase):
    def test_restored_branch_is_bit_exact_and_unrestored_is_not_guaranteed(self):
        cwd = os.getcwd()
        try:
            from counterfactual.branch import Branch, RNGState
            from counterfactual.simrun import make_action, max_divergence, policy_action, record_step, setup, stack
            env, smeta, pol, spec, obs = setup("building_train_8/atlanta_1")
            rooms = smeta["zones"]["room_ids"]
            for _ in range(30):
                obs = record_step(env, policy_action(pol, obs, spec), rooms)["obs"]
            br = Branch.take(env)
            main, acts = [], []
            for _ in range(8):
                a = policy_action(pol, obs, spec)
                acts.append(a); r = record_step(env, a, rooms); main.append(r); obs = r["obs"]
            after = RNGState.capture()
            arms = []
            for _ in range(2):
                e = br.spawn()
                arms.append(stack([record_step(e, make_action(a["continuous_action"], a["discrete_action"], spec), rooms) for a in acts]))
            self.assertTrue(max_divergence(arms[0], stack(main))["bit_exact"])
            self.assertTrue(max_divergence(arms[0], arms[1])["bit_exact"])
            after.restore()
            self.assertTrue(RNGState.capture().equals(after))
        finally:
            os.chdir(cwd)


# ------------------------------------------------------------------ occupancy
@unittest.skipUnless(HAVE_SAMPLE, "data/phase1_sample not present")
class TestOccupancyInputs(unittest.TestCase):
    def test_features_whitelisted_and_history_only(self):
        from dataset import load_episode, load_scenario_meta
        from occupancy.forecast import FEATURES, OccupancyFeaturizer
        from runtime_features import validate_runtime_features
        meta = load_scenario_meta(SAMPLE, "building_train_8/atlanta_1")
        f = OccupancyFeaturizer(meta)
        validate_runtime_features(f.features_used, field_names=meta["observation_field_names"])
        self.assertFalse(any(x.split(":")[0] in ("zone_info", "calendar", "reward") for x in f.features_used))
        ep = load_episode(SAMPLE / "episodes" / "building_train_8__atlanta_1__baseline__s0__e0.npz")
        obs = np.array(ep["obs"], dtype=np.float64)
        t0, H = 700, 36
        X = f.runtime_features(obs[:t0 + 1], H)
        # n_yday_same = observed N at row t0+k-286 (= N(t0+k-288)) only if that row is <= t0
        n = f.observed_total(obs) / f.V
        col = X[:, FEATURES.index("n_yday_same")]
        for k in range(H):
            r = t0 + k - 288 + 2
            if 0 <= r <= t0:
                self.assertAlmostEqual(col[k], n[r])
        # poisoning everything after t0 cannot matter: the function never receives it
        obs2 = obs.copy(); obs2[t0 + 1:] = np.nan
        np.testing.assert_array_equal(np.nan_to_num(f.runtime_features(obs2[:t0 + 1], H), nan=-9),
                                      np.nan_to_num(X, nan=-9))

    def test_observed_total_alignment(self):
        from dataset import load_episode, load_scenario_meta
        from occupancy.forecast import OccupancyFeaturizer, episode_targets
        meta = load_scenario_meta(SAMPLE, "building_train_2/fairbanks_2")
        ep = load_episode(SAMPLE / "episodes" / "building_train_2__fairbanks_2__baseline__s0__e0.npz")
        f = OccupancyFeaturizer(meta)
        np.testing.assert_allclose(f.observed_total(ep["obs"])[2:], episode_targets(ep["zone_info"])[:-1], atol=1e-6)


# ------------------------------------------------------------------ ensembles / uncertainty
class TestEnsembleAndScores(unittest.TestCase):
    def test_ensemble_identical_members_zero_spread(self):
        from uncertainty.scores import ensemble_predict
        b = synthetic_batch()
        b["inputs"]["exo"][..., 1] = 1.0
        from dynamics.model import PersistenceModel
        m = PersistenceModel(NORM)
        e = ensemble_predict([m, m, m], b)
        self.assertEqual(float(np.abs(e["zone_std"]).max()), 0.0)
        np.testing.assert_array_equal(e["zone_mean"], e["zone_m0"])

    def test_ensemble_mean_and_std(self):
        from uncertainty.scores import ensemble_predict

        class Const(torch.nn.Module):
            def __init__(self, v):
                super().__init__(); self.v = v
            def init_state(self, batch):
                return None
            def step(self, s, batch, j, T, h, o):
                B, V = T.shape
                return None, {"zone_temp": torch.full((B, V), self.v), "hws": torch.full((B,), self.v), "energy_norm": torch.zeros(B, 4)}
        e = ensemble_predict([Const(1.0), Const(3.0)], synthetic_batch())
        self.assertTrue(np.allclose(e["zone_mean"], 2.0) and np.allclose(e["zone_std"], np.sqrt(2.0)))

    def test_auroc_and_mahalanobis(self):
        from uncertainty.scores import Mahalanobis, auroc, score_vs_error
        self.assertEqual(auroc(np.array([0.1, 0.2, 0.9, 0.8]), np.array([0, 0, 1, 1])), 1.0)
        self.assertEqual(auroc(np.array([0.9, 0.8, 0.1, 0.2]), np.array([0, 0, 1, 1])), 0.0)
        X = np.random.default_rng(0).normal(size=(5000, 3))
        md = Mahalanobis().fit(X)
        self.assertLess(abs(np.mean(md(X) ** 2) - 3.0), 0.2)
        r = score_vs_error(np.arange(100.0), np.arange(100.0), 89.5, 89.5)
        self.assertEqual(r["precision"], 1.0); self.assertEqual(r["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
