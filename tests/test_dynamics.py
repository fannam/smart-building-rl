"""Phase 2A: rollout feedback rules, featurizer alignment, and input/target separation."""
import copy
import unittest

import numpy as np
import torch

from tests._paths import HAVE_SAMPLE, SAMPLE
from dynamics.model import rollout


class Recorder(torch.nn.Module):
    """Dummy model: prediction of step j = 1000*(j+1) (+zone index); records its inputs."""
    def __init__(self):
        super().__init__()
        self.seen = []

    def init_state(self, batch):
        return None

    def step(self, state, batch, j, T_in, hws_in, oat):
        self.seen.append((j, T_in.clone(), hws_in.clone(), oat.clone()))
        B, V = T_in.shape
        z = 1000.0 * (j + 1) + torch.arange(V, dtype=torch.float32).expand(B, V)
        return None, {"zone_temp": z, "hws": torch.full((B,), 1000.0 * (j + 1) + 0.5),
                      "energy_norm": torch.zeros(B, 4)}


def synthetic_batch(C=4, H=5, V=3, B=2):
    S = C - 1 + H
    obs_t = torch.arange(C, dtype=torch.float32)
    return {"inputs": {"zone_temp_obs": (10 * obs_t)[None, :, None].expand(B, C, V).clone(),
                       "hws_obs": (20 * obs_t)[None].expand(B, C).clone(),
                       "oat_obs": (30 * obs_t)[None].expand(B, C).clone(),
                       "exo": torch.zeros(B, S, 8)},
            "targets": {"zone_temp": torch.full((B, H, V), float("nan")), "hws": torch.full((B, H), float("nan")),
                        "oat_oracle": (777.0 + torch.arange(H, dtype=torch.float32))[None].expand(B, H).clone()},
            "zone_mask": torch.ones(B, V, dtype=torch.bool), "ahu_mask": torch.ones(B, 1, dtype=torch.bool),
            "info": {"K": torch.ones(B)}}


class TestBaselineRollouts(unittest.TestCase):
    def test_persistence_holds_latest_true_observation(self):
        from dynamics.model import PersistenceModel, Norm
        norm = Norm(295, 3, 0.2, 340, 8, 1, 290, 15, torch.ones(4), torch.ones(4))
        b = synthetic_batch(C=4, H=6)
        b["inputs"]["exo"][..., 1] = 1.0
        out = rollout(PersistenceModel(norm), b)
        self.assertTrue(torch.all(out["zone_temp"] == 10 * 3))     # o_{t0} = obs row C-1 for every k
        self.assertTrue(torch.all(out["hws"] == 20 * 3))


class TestRolloutFeedback(unittest.TestCase):
    def test_warmup_uses_true_obs_and_rollout_uses_own_predictions(self):
        C, H = 4, 5
        m, b = Recorder(), synthetic_batch(C, H)
        out = rollout(m, b, "persistence")
        self.assertEqual(out["zone_temp"].shape[1], H)
        for j, T_in, hws_in, oat in m.seen:
            if j <= C - 1:          # warm-up + decision step: true observation j
                self.assertTrue(torch.all(T_in == 10 * j)); self.assertTrue(torch.all(hws_in == 20 * j))
                self.assertTrue(torch.all(oat == 30 * j))
            else:                   # zone input = own prediction of step j-2, HWS = own prediction of j-1
                self.assertTrue(torch.all(T_in[:, 0] == 1000.0 * (j - 1)), (j, T_in))
                self.assertTrue(torch.all(hws_in == 1000.0 * j + 0.5))
                self.assertTrue(torch.all(oat == 30 * (C - 1)))           # outside air held at o_{t0}

    def test_targets_never_read_in_deployable_mode(self):
        m, b = Recorder(), synthetic_batch()
        out = rollout(m, b, "persistence")          # targets are NaN: any leak would propagate
        self.assertTrue(torch.isfinite(out["zone_temp"]).all())
        for _, T_in, hws_in, oat in m.seen:
            self.assertTrue(torch.isfinite(T_in).all() and torch.isfinite(hws_in).all() and torch.isfinite(oat).all())

    def test_oracle_mode_only_changes_outside_air(self):
        C = 4
        m, b = Recorder(), synthetic_batch(C)
        rollout(m, b, "oracle")
        for j, _, _, oat in m.seen:
            if j > C - 1:
                self.assertTrue(torch.all(oat == 777.0 + (j - (C - 1))))


@unittest.skipUnless(HAVE_SAMPLE, "data/phase1_sample not present")
class TestFeaturizerOnRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dataset import AlignedWindows, WindowSpec, load_manifest
        from dynamics.features import PRIVILEGED_FOR_TRAINING, DynamicsWindows
        rows = [r for r in load_manifest(SAMPLE) if r["building"] == "building_train_8"][:2]
        cls.C, cls.H = 6, 8
        aw = AlignedWindows.from_root(SAMPLE, rows, WindowSpec(horizon=cls.H, context=cls.C, privileged_targets=PRIVILEGED_FOR_TRAINING))
        cls.ds = DynamicsWindows(aw)
        from dataset import load_episode
        cls.eps = {r["episode_id"]: load_episode(SAMPLE / r["path"]) for r in rows}

    def test_inputs_and_targets_match_absolute_indices(self):
        from building_meta import ActionCodec, ObservationCodec
        from effective_action import ActionRepresentation
        from dynamics.schedule import decode_clock
        C, H = self.C, self.H
        for i in list(range(0, len(self.ds), 997))[:6]:
            it = self.ds[i]
            ep = self.eps[it["info"]["episode_id"]]
            t0 = it["info"]["t0"]
            meta = self.ds.aw.metas[it["info"]["scenario"]]
            oc = ObservationCodec(meta)
            nat = oc.native(ep["obs"])
            inp, tg = it["inputs"], it["targets"]
            # runtime: o_{t0-C+1..t0}
            np.testing.assert_allclose(inp["zone_temp_obs"], nat[t0 - C + 1:t0 + 1][:, meta["zones"]["obs_temp"]], atol=1e-3)
            np.testing.assert_allclose(inp["hws_obs"], nat[t0 - C + 1:t0 + 1, meta["hws"]["obs_supply_water_temp"]], atol=1e-3)
            # targets: zone temp after t = sensors of obs[t+2]; HWS after t = obs[t+1]
            np.testing.assert_allclose(tg["zone_temp"], nat[t0 + 2:t0 + H + 2][:, meta["zones"]["obs_temp"]], atol=1e-3)
            np.testing.assert_allclose(tg["hws"], nat[t0 + 1:t0 + H + 1, meta["hws"]["obs_supply_water_temp"]], atol=1e-3)
            np.testing.assert_allclose(tg["energy"][:, 3], ep["boiler_info"][t0:t0 + H, 0, 0], rtol=1e-6)
            np.testing.assert_array_equal(tg["reward"], ep["reward"][t0:t0 + H])
            # actions: row i = a[t0-C+i]; damper row C+k = effective damper of a_{t0+k}
            eff = ActionRepresentation(meta).effective(ep["action_continuous"], ep["action_discrete"]).effective_native
            np.testing.assert_allclose(inp["zone_act"][C:, :, 0], eff[t0:t0 + H][:, meta["zones"]["act_damper"]], atol=1e-5)
            np.testing.assert_allclose(inp["zone_act"][1:C, :, 0], eff[t0 - C + 1:t0][:, meta["zones"]["act_damper"]], atol=1e-5)
            # clock of warm-up/decision steps equals the observed clock; setpoint window = schedule behind reward[t]
            h, d = decode_clock(ep["obs"][t0 - C + 1:t0 + 1], oc.index)
            np.testing.assert_allclose(inp["exo"][:C, 0], np.sin(2 * np.pi * h / 24), atol=1e-5)
            np.testing.assert_allclose(inp["setpoint_next"][C - 1:, 0], ep["zone_info"][t0:t0 + H, 0, 0], atol=1e-4)

    def test_runtime_part_independent_of_targets(self):
        it = self.ds.aw[123]
        f = self.ds.feat[it["index"]["scenario"]]
        a = f.runtime_part(it["runtime_inputs"], it["actions"], self.H)
        poisoned = copy.deepcopy(it)
        for k, v in poisoned["targets"]["privileged"].items():
            poisoned["targets"]["privileged"][k] = np.full_like(v, np.nan, dtype=np.float64)
        poisoned["targets"]["obs_window"] = np.full_like(poisoned["targets"]["obs_window"], np.nan)
        b = f.runtime_part(poisoned["runtime_inputs"], poisoned["actions"], self.H)
        for k in a:
            np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))

    def test_featurizer_inputs_whitelisted(self):
        from runtime_features import validate_runtime_features
        for f in self.ds.feat.values():
            validate_runtime_features(f.features_used, field_names=f.meta["observation_field_names"])
            self.assertFalse(any(x.split(":")[0] in ("zone_info", "tank_state", "calendar", "ahu_info", "boiler_info")
                                 for x in f.features_used))


if __name__ == "__main__":
    unittest.main()
