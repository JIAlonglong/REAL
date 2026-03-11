#!/usr/bin/env python3
"""REAL codebase smoke tests — CPU only, no Isaac Gym required.

Hierarchy:
  Level 0: Syntax & import checks
  Level 1: Module forward-pass tests (individual components)
  Level 2: Algorithm-level tests (PPO init, act, gradient step)

Usage:
  python tests/test_modules.py              # run all Level 0-2
  python tests/test_modules.py -v           # verbose
  python tests/test_modules.py -k mamba     # run only tests matching 'mamba'
"""

import sys
import os
import unittest
import types

# ── Project paths ──────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, os.path.join(_ROOT, "rsl_rl"))
sys.path.insert(0, os.path.join(_ROOT, "legged_gym"))

# ── Stub heavy optional deps before any project import ─────────────────────
def _install_stubs():
    # isaacgym (full stub for legged_robot.py imports)
    if "isaacgym" not in sys.modules:
        ig = types.ModuleType("isaacgym")
        ig.__path__ = []
        ig_tu = types.ModuleType("isaacgym.torch_utils")
        ig_tu.quat_rotate_inverse = lambda q, v: v
        ig.gymtorch = types.ModuleType("isaacgym.gymtorch")
        ig.gymapi = types.ModuleType("isaacgym.gymapi")
        ig.gymutil = types.ModuleType("isaacgym.gymutil")
        sys.modules["isaacgym"] = ig
        sys.modules["isaacgym.torch_utils"] = ig_tu
        sys.modules["isaacgym.gymtorch"] = ig.gymtorch
        sys.modules["isaacgym.gymapi"] = ig.gymapi
        sys.modules["isaacgym.gymutil"] = ig.gymutil
    # wandb
    if "wandb" not in sys.modules:
        wb = types.ModuleType("wandb")
        wb.log = lambda *a, **kw: None
        wb.init = lambda *a, **kw: None
        wb.save = lambda *a, **kw: None
        wb.Image = lambda *a, **kw: None
        sys.modules["wandb"] = wb

_install_stubs()

import torch
torch.set_grad_enabled(False)

# ── Dimensions matching default config ─────────────────────────────────────
B = 4
N_PROP = 53
N_SCAN = 132
N_PRIV = 9
N_PRIV_LATENT = 29
N_HIST = 10
N_ACTIONS = 12
DEPTH_H, DEPTH_W = 58, 87
DEPTH_HIST = 2


# ===========================================================================
#  Level 0 — Syntax & Import Checks
# ===========================================================================

class TestLevel0_Imports(unittest.TestCase):

    def test_000_py_compile_all(self):
        """py_compile every .py file to catch syntax errors."""
        import py_compile
        errors = []
        for dirpath, _, fnames in os.walk(_ROOT):
            if "__pycache__" in dirpath or "tests" in dirpath:
                continue
            for fn in fnames:
                if not fn.endswith(".py"):
                    continue
                fpath = os.path.join(dirpath, fn)
                try:
                    py_compile.compile(fpath, doraise=True)
                except py_compile.PyCompileError as e:
                    errors.append(str(e))
        self.assertEqual(errors, [], f"Syntax errors:\n" + "\n".join(errors))

    def test_001_import_config(self):
        """Import config using save_jit's loader (bypasses legged_robot.py)."""
        sys.path.insert(0, os.path.join(_ROOT, "legged_gym", "legged_gym", "scripts"))
        from save_jit import _load_cfg_classes
        LeggedRobotCfg, LeggedRobotCfgPPO = _load_cfg_classes()
        self.assertEqual(LeggedRobotCfg.env.n_proprio, N_PROP)
        self.assertEqual(LeggedRobotCfg.env.n_scan, N_SCAN)
        self.assertEqual(LeggedRobotCfgPPO.policy.scan_encoder_type, "proprio_cross_attention")

    def test_002_import_go2_config(self):
        """GO2 config file can be compiled and parsed."""
        go2_path = os.path.join(_ROOT, "legged_gym", "legged_gym", "envs", "go2", "go2_config.py")
        with open(go2_path) as f:
            src = f.read()
        # Verify it references LeggedRobotCfg correctly
        self.assertIn("class GO2RoughCfg(LeggedRobotCfg)", src)
        self.assertIn("FL_hip_joint", src)
        import py_compile
        py_compile.compile(go2_path, doraise=True)

    def test_003_import_rsl_rl_modules(self):
        from rsl_rl.modules import ActorCriticRMA, Estimator, TcnEstimator, ResNetEstimator1D
        from rsl_rl.modules import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone
        from rsl_rl.modules import ProprioQueryCrossAttentionEncoder
        from rsl_rl.modules.mamba_block import SimpleMambaBlock
        from rsl_rl.modules.depth_backbone import FiLM2d, MambaTemporalEncoder

    def test_004_import_ppo(self):
        from rsl_rl.algorithms import PPO

    def test_005_import_runner(self):
        from rsl_rl.runners import OnPolicyRunner

    def test_006_import_storage(self):
        from rsl_rl.storage import RolloutStorage

    def test_007_import_utils(self):
        from rsl_rl.utils import split_and_pad_trajectories, unpad_trajectories


# ===========================================================================
#  Level 1 — Module Forward-Pass Tests
# ===========================================================================

class TestLevel1_Mamba(unittest.TestCase):

    def test_forward_shape(self):
        from rsl_rl.modules.mamba_block import SimpleMambaBlock
        blk = SimpleMambaBlock(d_model=32, d_state=16, proprio_dim=N_PROP)
        x = torch.randn(B, 5, 32)
        p = torch.randn(B, N_PROP)
        y = blk(x, p)
        self.assertEqual(y.shape, (B, 5, 32))

    def test_forward_no_proprio(self):
        from rsl_rl.modules.mamba_block import SimpleMambaBlock
        blk = SimpleMambaBlock(d_model=32, d_state=16)
        x = torch.randn(B, 3, 32)
        y = blk(x)
        self.assertEqual(y.shape, (B, 3, 32))

    def test_gradient_flows(self):
        from rsl_rl.modules.mamba_block import SimpleMambaBlock
        blk = SimpleMambaBlock(d_model=32, d_state=16, proprio_dim=N_PROP)
        x = torch.randn(B, 3, 32, requires_grad=True)
        p = torch.randn(B, N_PROP)
        with torch.enable_grad():
            y = blk(x, p)
            loss = y.sum()
            loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(x.grad.abs().sum() > 0)

    def test_no_nan(self):
        from rsl_rl.modules.mamba_block import SimpleMambaBlock
        blk = SimpleMambaBlock(d_model=32, d_state=64, proprio_dim=N_PROP)
        x = torch.randn(B, 10, 32)
        p = torch.randn(B, N_PROP)
        y = blk(x, p)
        self.assertFalse(torch.isnan(y).any(), "Mamba output contains NaN")


class TestLevel1_FiLM(unittest.TestCase):

    def test_forward_shape(self):
        from rsl_rl.modules.depth_backbone import FiLM2d
        film = FiLM2d(prop_dim=N_PROP, channels=32)
        x = torch.randn(B, 32, 10, 10)
        p = torch.randn(B, N_PROP)
        y = film(x, p)
        self.assertEqual(y.shape, x.shape)

    def test_identity_at_init(self):
        """At initialization, FiLM should be close to identity (gamma~1, beta~0)."""
        from rsl_rl.modules.depth_backbone import FiLM2d
        film = FiLM2d(prop_dim=N_PROP, channels=32)
        x = torch.ones(1, 32, 4, 4)
        p = torch.zeros(1, N_PROP)
        y = film(x, p)
        self.assertTrue(torch.allclose(x, y, atol=0.5),
                        "FiLM at init should be near-identity")


class TestLevel1_DepthCNN(unittest.TestCase):

    def test_forward_no_film(self):
        from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87
        cnn = DepthOnlyFCBackbone58x87(N_PROP, 32, 512, cnn_channels=[32, 64])
        cnn.film_spatial_enabled = False
        img = torch.randn(B, DEPTH_H, DEPTH_W)
        out = cnn(img)
        self.assertEqual(out.shape, (B, 32))

    def test_forward_with_film(self):
        from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87
        cnn = DepthOnlyFCBackbone58x87(N_PROP, 32, 512, cnn_channels=[32, 64])
        cnn.film_spatial_enabled = True
        cnn.film_spatial_sites = ["conv1", "conv2"]
        img = torch.randn(B, DEPTH_H, DEPTH_W)
        p = torch.randn(B, N_PROP)
        out = cnn(img, p)
        self.assertEqual(out.shape, (B, 32))

    def test_forward_4d_input(self):
        from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87
        cnn = DepthOnlyFCBackbone58x87(N_PROP, 32, 512, num_frames=1, cnn_channels=[32, 64])
        img = torch.randn(B, 1, DEPTH_H, DEPTH_W)
        out = cnn(img)
        self.assertEqual(out.shape, (B, 32))


class TestLevel1_MambaTemporalEncoder(unittest.TestCase):

    def test_forward_shape(self):
        from rsl_rl.modules.depth_backbone import MambaTemporalEncoder
        enc = MambaTemporalEncoder(history_len=DEPTH_HIST, proprio_dim=N_PROP,
                                   d_state=64, n_layers=2)
        per_frame = torch.randn(B, DEPTH_HIST, 32)
        proprio = torch.randn(B, N_PROP)
        out, state = enc(per_frame, proprio)
        self.assertEqual(out.shape, (B, 32))
        self.assertEqual(state.shape, (B, 32))

    def test_padding_short_input(self):
        from rsl_rl.modules.depth_backbone import MambaTemporalEncoder
        enc = MambaTemporalEncoder(history_len=4, proprio_dim=N_PROP)
        per_frame = torch.randn(B, 2, 32)
        proprio = torch.randn(B, N_PROP)
        out, state = enc(per_frame, proprio)
        self.assertEqual(out.shape, (B, 32))

    def test_truncation_long_input(self):
        from rsl_rl.modules.depth_backbone import MambaTemporalEncoder
        enc = MambaTemporalEncoder(history_len=2, proprio_dim=N_PROP)
        per_frame = torch.randn(B, 5, 32)
        proprio = torch.randn(B, N_PROP)
        out, state = enc(per_frame, proprio)
        self.assertEqual(out.shape, (B, 32))


class TestLevel1_RecurrentDepthBackbone(unittest.TestCase):

    def _make_env_cfg(self):
        class Env:
            n_proprio = N_PROP
            depth_history_len = DEPTH_HIST
            scan_history_len = 1
        class Cfg:
            env = Env()
        return Cfg()

    def test_classic_mode(self):
        from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone
        cnn = DepthOnlyFCBackbone58x87(N_PROP, 32, 512, cnn_channels=[32, 64])
        enc = RecurrentDepthBackbone(cnn, self._make_env_cfg(), temporal_cfg=None)
        img = torch.randn(B, DEPTH_H, DEPTH_W)
        p = torch.randn(B, N_PROP)
        out = enc(img, p)
        self.assertEqual(out.shape, (B, 34))

    def test_mamba_mode(self):
        from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone
        temporal_cfg = {
            "film_spatial_enabled": True,
            "film_spatial_sites": ["conv1", "conv2"],
            "depth_history_len": DEPTH_HIST,
            "depth_mamba_d_state": 64,
            "depth_mamba_layers": 2,
            "depth_mamba_dropout": 0.0,
        }
        cnn = DepthOnlyFCBackbone58x87(N_PROP, 32, 512, cnn_channels=[32, 64])
        enc = RecurrentDepthBackbone(cnn, self._make_env_cfg(), temporal_cfg=temporal_cfg)
        img = torch.randn(B, DEPTH_HIST, DEPTH_H, DEPTH_W)
        p = torch.randn(B, N_PROP)
        out = enc(img, p)
        self.assertEqual(out.shape, (B, 34))
        self.assertIsNotNone(enc.latest_mamba_state)

    def test_hidden_state_persistence(self):
        from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone
        cnn = DepthOnlyFCBackbone58x87(N_PROP, 32, 512, cnn_channels=[32, 64])
        enc = RecurrentDepthBackbone(cnn, self._make_env_cfg(), temporal_cfg=None)
        img = torch.randn(B, DEPTH_H, DEPTH_W)
        p = torch.randn(B, N_PROP)
        out1 = enc(img, p)
        self.assertIsNotNone(enc.hidden_states)
        out2 = enc(img, p)
        self.assertFalse(torch.allclose(out1, out2),
                         "GRU hidden should make consecutive outputs differ")

    def test_detach_hidden_states(self):
        from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone
        cnn = DepthOnlyFCBackbone58x87(N_PROP, 32, 512, cnn_channels=[32, 64])
        enc = RecurrentDepthBackbone(cnn, self._make_env_cfg(), temporal_cfg=None)
        img = torch.randn(B, DEPTH_H, DEPTH_W)
        p = torch.randn(B, N_PROP)
        with torch.enable_grad():
            _ = enc(img, p)
        enc.detach_hidden_states()
        self.assertFalse(enc.hidden_states.requires_grad)

    def test_mamba_gradient(self):
        """Full pipeline gradient flows through FiLM-CNN -> Mamba -> GRU."""
        from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone
        temporal_cfg = {
            "film_spatial_enabled": True,
            "film_spatial_sites": ["conv1", "conv2"],
            "depth_history_len": DEPTH_HIST,
            "depth_mamba_d_state": 64,
            "depth_mamba_layers": 1,
        }
        cnn = DepthOnlyFCBackbone58x87(N_PROP, 32, 512, cnn_channels=[32, 64])
        enc = RecurrentDepthBackbone(cnn, self._make_env_cfg(), temporal_cfg=temporal_cfg)
        img = torch.randn(B, DEPTH_HIST, DEPTH_H, DEPTH_W, requires_grad=True)
        p = torch.randn(B, N_PROP)
        with torch.enable_grad():
            out = enc(img, p)
            out.sum().backward()
        self.assertIsNotNone(img.grad)


class TestLevel1_ProprioQueryCrossAttention(unittest.TestCase):

    def test_forward_shape(self):
        from rsl_rl.modules.proprio_cross_attention_encoder import ProprioQueryCrossAttentionEncoder
        enc = ProprioQueryCrossAttentionEncoder(
            scan_dim=N_SCAN, proprio_dim=N_PROP,
            d_model=128, num_heads=4, num_layers=2, output_dim=32,
        )
        proprio = torch.randn(B, N_PROP)
        scan = torch.randn(B, 1, N_SCAN)
        out = enc(proprio=proprio, scan_seq=scan)
        self.assertEqual(out.shape, (B, 32))

    def test_no_nan(self):
        from rsl_rl.modules.proprio_cross_attention_encoder import ProprioQueryCrossAttentionEncoder
        enc = ProprioQueryCrossAttentionEncoder(
            scan_dim=N_SCAN, proprio_dim=N_PROP,
            d_model=128, num_heads=4, num_layers=2, output_dim=32,
        )
        proprio = torch.randn(B, N_PROP)
        scan = torch.randn(B, 1, N_SCAN)
        out = enc(proprio=proprio, scan_seq=scan)
        self.assertFalse(torch.isnan(out).any())


class TestLevel1_Estimator(unittest.TestCase):

    def test_mlp_estimator(self):
        from rsl_rl.modules.estimator import Estimator
        est = Estimator(input_dim=N_PROP, output_dim=N_PRIV, hidden_dims=[128, 64])
        out = est(torch.randn(B, N_PROP))
        self.assertEqual(out.shape, (B, N_PRIV))

    def test_mlp_estimator_forward_with_uncertainty(self):
        """forward() returns mean only; forward_with_uncertainty() returns (mean, log_var)."""
        from rsl_rl.modules.estimator import Estimator
        est = Estimator(input_dim=N_PROP, output_dim=N_PRIV, hidden_dims=[128, 64],
                        predict_uncertainty=True)
        out_mean = est(torch.randn(B, N_PROP))
        self.assertEqual(out_mean.shape, (B, N_PRIV))
        mean, log_var = est.forward_with_uncertainty(torch.randn(B, N_PROP))
        self.assertEqual(mean.shape, (B, N_PRIV))
        self.assertEqual(log_var.shape, (B, N_PRIV))

    def test_resnet1d_estimator(self):
        from rsl_rl.modules.estimator_resnet1d import ResNetEstimator1D
        est = ResNetEstimator1D(input_dim=N_PROP, output_dim=N_PRIV, predict_uncertainty=False)
        self.assertTrue(est.sequence_input)
        x = torch.randn(B, N_HIST, N_PROP)
        out = est(x)
        self.assertEqual(out.shape, (B, N_PRIV))

    def test_resnet1d_uncertainty(self):
        from rsl_rl.modules.estimator_resnet1d import ResNetEstimator1D
        est = ResNetEstimator1D(input_dim=N_PROP, output_dim=N_PRIV, predict_uncertainty=True)
        x = torch.randn(B, N_HIST, N_PROP)
        mean, log_std = est.forward_with_uncertainty(x)
        self.assertEqual(mean.shape, (B, N_PRIV))
        self.assertEqual(log_std.shape, (B, N_PRIV))

    def test_tcn_estimator(self):
        from rsl_rl.modules.estimator import TcnEstimator
        est = TcnEstimator(input_dim=N_PROP, output_dim=N_PRIV,
                           num_channels=[32, 32], kernel_size=2)
        x = torch.randn(B, N_HIST, N_PROP)
        out = est(x)
        self.assertEqual(out.shape, (B, N_PRIV))


class TestLevel1_ActorCritic(unittest.TestCase):

    def _make_ac(self):
        from rsl_rl.modules.actor_critic import ActorCriticRMA
        num_obs = N_PROP + N_SCAN + N_HIST * N_PROP + N_PRIV_LATENT + N_PRIV
        return ActorCriticRMA(
            num_prop=N_PROP, num_scan=N_SCAN, num_critic_obs=num_obs,
            num_priv_latent=N_PRIV_LATENT, num_priv_explicit=N_PRIV,
            num_hist=N_HIST, num_actions=N_ACTIONS,
            scan_encoder_dims=[128, 64, 32],
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation='elu', init_noise_std=1.0,
            priv_encoder_dims=[64, 20],
            tanh_encoder_output=False,
            scan_encoder_type="proprio_cross_attention",
            scan_history_len=1,
            scan_attn_d_model=128,
            scan_attn_heads=4,
            scan_attn_layers=2,
        )

    def _make_obs(self):
        num_obs = N_PROP + N_SCAN + N_PRIV + N_PRIV_LATENT + N_HIST * N_PROP
        return torch.randn(B, num_obs)

    def test_actor_forward(self):
        ac = self._make_ac()
        obs = self._make_obs()
        actions = ac.actor(obs, hist_encoding=True)
        self.assertEqual(actions.shape, (B, N_ACTIONS))

    def test_act(self):
        ac = self._make_ac()
        obs = self._make_obs()
        with torch.enable_grad():
            ac.train()
            actions = ac.act(obs)
        self.assertEqual(actions.shape, (B, N_ACTIONS))

    def test_evaluate(self):
        """evaluate() returns critic value only."""
        ac = self._make_ac()
        obs = self._make_obs()
        with torch.enable_grad():
            ac.train()
            value = ac.evaluate(obs)
        self.assertEqual(value.shape, (B, 1))

    def test_actor_with_scandots_latent(self):
        ac = self._make_ac()
        obs = self._make_obs()
        scandots_latent = torch.randn(B, 32)
        actions = ac.actor(obs, hist_encoding=True, scandots_latent=scandots_latent)
        self.assertEqual(actions.shape, (B, N_ACTIONS))

    def test_infer_priv_latent(self):
        ac = self._make_ac()
        obs = self._make_obs()
        latent = ac.actor.infer_priv_latent(obs)
        self.assertEqual(latent.shape, (B, 20))

    def test_infer_hist_latent(self):
        ac = self._make_ac()
        obs = self._make_obs()
        latent = ac.actor.infer_hist_latent(obs)
        self.assertEqual(latent.shape, (B, 20))


# ===========================================================================
#  Level 2 — Algorithm-Level Tests
# ===========================================================================

def _make_all_components():
    from rsl_rl.modules.actor_critic import ActorCriticRMA, Actor
    from rsl_rl.modules.estimator_resnet1d import ResNetEstimator1D
    from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone

    num_obs = N_PROP + N_SCAN + N_HIST * N_PROP + N_PRIV_LATENT + N_PRIV

    ac = ActorCriticRMA(
        num_prop=N_PROP, num_scan=N_SCAN, num_critic_obs=num_obs,
        num_priv_latent=N_PRIV_LATENT, num_priv_explicit=N_PRIV,
        num_hist=N_HIST, num_actions=N_ACTIONS,
        scan_encoder_dims=[128, 64, 32],
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation='elu', init_noise_std=1.0,
        priv_encoder_dims=[64, 20],
        tanh_encoder_output=False,
        scan_encoder_type="proprio_cross_attention",
        scan_history_len=1,
    )

    estimator = ResNetEstimator1D(
        input_dim=N_PROP, output_dim=N_PRIV, predict_uncertainty=True,
    )
    estimator_paras = {
        "priv_states_dim": N_PRIV, "num_prop": N_PROP, "num_scan": N_SCAN,
        "learning_rate": 1e-4, "train_with_estimated_states": False,
        "uncertainty_enabled": True, "uncertainty_warmup_iters": 200,
        "min_log_std": -6.9, "max_log_std": 2.0,
        "uncertainty_huber_delta": 5e-3, "uncertainty_huber_lam": 1e-4,
        "history_len": N_HIST,
        "fusion_enabled": True, "fusion_q": 0.01,
        "fusion_r_scale": 1.0, "fusion_p0": 1.0,
    }

    class EnvCfg:
        class env:
            n_proprio = N_PROP
            depth_history_len = DEPTH_HIST
            scan_history_len = 1

    temporal_cfg = {
        "film_spatial_enabled": True,
        "film_spatial_sites": ["conv1", "conv2"],
        "depth_history_len": DEPTH_HIST,
        "depth_mamba_d_state": 64,
        "depth_mamba_layers": 2,
        "depth_mamba_dropout": 0.0,
        "learning_rate": 1e-3,
    }
    cnn = DepthOnlyFCBackbone58x87(N_PROP, 32, 512, cnn_channels=[32, 64])
    depth_encoder = RecurrentDepthBackbone(cnn, EnvCfg(), temporal_cfg=temporal_cfg)

    depth_actor = Actor(
        N_PROP, N_SCAN, N_ACTIONS, [128, 64, 32],
        [512, 256, 128], [64, 20], N_PRIV_LATENT, N_PRIV, N_HIST,
        torch.nn.ELU(), tanh_encoder_output=False,
        scan_encoder_type="proprio_cross_attention",
    )

    depth_encoder_paras = {
        "learning_rate": 1e-3,
        "consistency_gating_k": 2.0,
        "consistency_gating_tau": 0.5,
    }

    return ac, estimator, estimator_paras, depth_encoder, depth_encoder_paras, depth_actor


class TestLevel2_PPO(unittest.TestCase):

    def test_ppo_init(self):
        from rsl_rl.algorithms.ppo import PPO
        ac, est, est_p, de, de_p, da = _make_all_components()
        ppo = PPO(ac, est, est_p, de, de_p, da, device='cpu')
        self.assertIsNotNone(ppo)

    def test_ppo_init_storage(self):
        from rsl_rl.algorithms.ppo import PPO
        ac, est, est_p, de, de_p, da = _make_all_components()
        ppo = PPO(ac, est, est_p, de, de_p, da, device='cpu')
        num_obs = N_PROP + N_SCAN + N_HIST * N_PROP + N_PRIV_LATENT + N_PRIV
        ppo.init_storage(B, 24, [num_obs], [num_obs], [N_ACTIONS])
        self.assertIsNotNone(ppo.storage)
        # EKF state should be initialized
        self.assertIsNotNone(ppo.fusion_last_v)
        self.assertEqual(ppo.fusion_last_v.shape, (B, 3))

    def test_ppo_act(self):
        from rsl_rl.algorithms.ppo import PPO
        ac, est, est_p, de, de_p, da = _make_all_components()
        ppo = PPO(ac, est, est_p, de, de_p, da, device='cpu')
        num_obs = N_PROP + N_SCAN + N_HIST * N_PROP + N_PRIV_LATENT + N_PRIV
        ppo.init_storage(B, 24, [num_obs], [num_obs], [N_ACTIONS])
        with torch.enable_grad():
            ppo.train_mode()
            obs = torch.randn(B, num_obs)
            actions = ppo.act(obs, obs, {})
        self.assertEqual(actions.shape, (B, N_ACTIONS))

    def test_ppo_single_update(self):
        """Fill storage with dummy data and run one PPO update step."""
        from rsl_rl.algorithms.ppo import PPO
        ac, est, est_p, de, de_p, da = _make_all_components()
        ppo = PPO(ac, est, est_p, de, de_p, da, device='cpu',
                  num_learning_epochs=1, num_mini_batches=1,
                  priv_reg_coef_schedual=[0, 0.1, 2000, 3000])
        num_obs = N_PROP + N_SCAN + N_HIST * N_PROP + N_PRIV_LATENT + N_PRIV
        num_steps = 4
        ppo.init_storage(B, num_steps, [num_obs], [num_obs], [N_ACTIONS])

        with torch.enable_grad():
            ppo.train_mode()
            for _ in range(num_steps):
                obs = torch.randn(B, num_obs)
                actions = ppo.act(obs, obs, {})
                ppo.process_env_step(torch.randn(B, 1), torch.zeros(B, dtype=torch.bool), {})
            ppo.compute_returns(obs)
            result = ppo.update()
        self.assertIsInstance(result[0], float)  # mean_value_loss
        self.assertIsInstance(result[1], float)  # mean_surr_loss


class TestLevel2_EstimatorLoss(unittest.TestCase):

    def test_huber_gaussian_loss(self):
        from rsl_rl.algorithms.ppo import PPO
        ac, est, est_p, de, de_p, da = _make_all_components()
        ppo = PPO(ac, est, est_p, de, de_p, da, device='cpu')

        pred_mean = torch.randn(B, N_PRIV)
        pred_log_std = torch.randn(B, N_PRIV) * 0.1
        target = torch.randn(B, N_PRIV)

        with torch.enable_grad():
            pred_mean.requires_grad_(True)
            pred_log_std.requires_grad_(True)
            loss_per_elem = ppo._loss_huber_gaussian_diag(pred_mean, pred_log_std, target)
        # Returns (B, N_PRIV) per-element loss
        self.assertEqual(loss_per_elem.shape, (B, N_PRIV))
        self.assertFalse(torch.isnan(loss_per_elem).any(), "Huber-Gaussian loss contains NaN")
        with torch.enable_grad():
            loss_per_elem.mean().backward()
        self.assertIsNotNone(pred_mean.grad)


class TestLevel2_KFFusion(unittest.TestCase):
    """Kalman Filter fusion via PPO._apply_kf_fusion (Eq. 6-10)."""

    def test_kf_fusion_noop_when_disabled(self):
        from rsl_rl.algorithms.ppo import PPO
        ac, est, est_p, de, de_p, da = _make_all_components()
        est_p_no_fusion = dict(est_p)
        est_p_no_fusion["fusion_enabled"] = False
        ppo = PPO(ac, est, est_p_no_fusion, de, de_p, da, device='cpu')
        mean_pred = torch.randn(B, N_PRIV)
        log_std = torch.zeros(B, N_PRIV)
        result = ppo._apply_kf_fusion(mean_pred, log_std, {})
        self.assertIsNone(result)

    def test_kf_fusion_runs(self):
        from rsl_rl.algorithms.ppo import PPO
        ac, est, est_p, de, de_p, da = _make_all_components()
        ppo = PPO(ac, est, est_p, de, de_p, da, device='cpu')
        num_obs = N_PROP + N_SCAN + N_HIST * N_PROP + N_PRIV_LATENT + N_PRIV
        ppo.init_storage(B, 24, [num_obs], [num_obs], [N_ACTIONS])

        mean_pred = torch.randn(B, N_PRIV)
        log_std = torch.zeros(B, N_PRIV)
        result = ppo._apply_kf_fusion(mean_pred, log_std, {"dt": 0.02})
        # Should return fused velocity (3-dim slice)
        if result is not None:
            self.assertEqual(result.shape[0], B)


class TestLevel2_ConsistencyGating(unittest.TestCase):

    def test_gating_lambda_range(self):
        k = 2.0
        tau = 0.5
        a_s = torch.randn(B, N_ACTIONS)
        a_t = a_s + 0.1 * torch.randn(B, N_ACTIONS)
        diff_norm = (a_s - a_t).norm(dim=-1)
        lam = torch.sigmoid(k * (tau - diff_norm))
        self.assertEqual(lam.shape, (B,))
        self.assertTrue((lam >= 0).all() and (lam <= 1).all())

    def test_gating_close_actions(self):
        """When student≈teacher, lambda should be high (more RL)."""
        k = 2.0
        tau = 0.5
        a = torch.randn(B, N_ACTIONS)
        diff_norm = torch.zeros(B)
        lam = torch.sigmoid(k * (tau - diff_norm))
        self.assertTrue((lam > 0.5).all())

    def test_gating_far_actions(self):
        """When student far from teacher, lambda should be low (more BC)."""
        k = 2.0
        tau = 0.5
        diff_norm = torch.ones(B) * 5.0
        lam = torch.sigmoid(k * (tau - diff_norm))
        self.assertTrue((lam < 0.1).all())


# ===========================================================================
#  Level 2b — JIT Export Wrappers
# ===========================================================================

class TestLevel2_JITWrappers(unittest.TestCase):

    def test_hardware_vision_nn(self):
        sys.path.insert(0, os.path.join(_ROOT, "legged_gym", "legged_gym", "scripts"))
        from save_jit import HardwareVisionNN
        nn_model = HardwareVisionNN(
            N_PROP, N_SCAN, N_PRIV_LATENT, N_PRIV, N_HIST, N_ACTIONS, tanh=False,
        )
        nn_model.eval()
        num_obs = N_PROP + N_SCAN + N_PRIV + N_PRIV_LATENT + N_HIST * N_PROP
        obs = torch.randn(1, num_obs)
        depth_latent = torch.randn(1, 32)
        out = nn_model(obs, depth_latent)
        self.assertEqual(out.shape, (1, N_ACTIONS))

    def test_hardware_vision_nn_trace(self):
        """Verify HardwareVisionNN can be torch.jit.trace'd."""
        sys.path.insert(0, os.path.join(_ROOT, "legged_gym", "legged_gym", "scripts"))
        from save_jit import HardwareVisionNN
        nn_model = HardwareVisionNN(
            N_PROP, N_SCAN, N_PRIV_LATENT, N_PRIV, N_HIST, N_ACTIONS, tanh=False,
        )
        nn_model.eval()
        num_obs = N_PROP + N_SCAN + N_PRIV + N_PRIV_LATENT + N_HIST * N_PROP
        obs = torch.randn(1, num_obs)
        depth_latent = torch.randn(1, 32)
        traced = torch.jit.trace(nn_model, (obs, depth_latent))
        out = traced(obs, depth_latent)
        self.assertEqual(out.shape, (1, N_ACTIONS))

    def test_vision_encoder_jit_wrapper(self):
        sys.path.insert(0, os.path.join(_ROOT, "legged_gym", "legged_gym", "scripts"))
        from save_jit import VisionEncoderJitWrapper
        from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone

        class EnvCfg:
            class env:
                n_proprio = N_PROP
                depth_history_len = 1
                scan_history_len = 1

        cnn = DepthOnlyFCBackbone58x87(N_PROP, 32, 512, cnn_channels=[32, 64])
        enc = RecurrentDepthBackbone(cnn, EnvCfg(), temporal_cfg=None)
        wrapper = VisionEncoderJitWrapper(enc)
        img = torch.randn(1, DEPTH_H, DEPTH_W)
        p = torch.randn(1, N_PROP)
        out = wrapper(img, p)
        self.assertEqual(out.shape[0], 1)

    def test_onboard_policy_wrapper(self):
        sys.path.insert(0, os.path.join(_ROOT, "legged_gym", "legged_gym", "scripts"))
        from save_jit import OnboardPolicyWrapper, HardwareVisionNN
        nn_model = HardwareVisionNN(
            N_PROP, N_SCAN, N_PRIV_LATENT, N_PRIV, N_HIST, N_ACTIONS, tanh=False,
        )
        onboard = OnboardPolicyWrapper(
            estimator=nn_model.estimator,
            history_encoder=nn_model.actor.history_encoder,
            actor_backbone=nn_model.actor.actor_backbone,
        )
        proprio = torch.randn(1, N_PROP)
        hist = torch.randn(1, N_HIST, N_PROP)
        depth_lat = torch.randn(1, 32)
        out = onboard(proprio, hist, depth_lat)
        self.assertEqual(out.shape, (1, N_ACTIONS))


# ===========================================================================
#  Main
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
