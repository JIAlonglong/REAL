"""Export trained checkpoints to TorchScript for deployment.

Creates:
  - *-base_jit.pt: policy network (obs, depth_latent) -> actions
  - *-vision_weight.pt: depth encoder state dict
  - *-vision_jit.pt / *-vision_stateful_jit.pt: traced depth encoder
  - *-onboard_jit.pt: combined (estimator + history_encoder + actor_backbone)
"""

import os, sys
import importlib.util
import types

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS_DIR, "../../../rsl_rl")))
import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic import Actor, StateHistoryEncoder, get_activation, ActorCriticRMA
from rsl_rl.modules.estimator import Estimator, TcnEstimator
from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone
import argparse
import shutil

def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        val = getattr(obj, key)
        if isinstance(val, list):
            element = [class_to_dict(item) for item in val]
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def _load_cfg_classes():
    base_dir = os.path.abspath(os.path.join(_THIS_DIR, "..", "envs", "base"))
    pkg_legged = sys.modules.get("legged_gym") or types.ModuleType("legged_gym")
    pkg_envs = sys.modules.get("legged_gym.envs") or types.ModuleType("legged_gym.envs")
    pkg_base = sys.modules.get("legged_gym.envs.base") or types.ModuleType("legged_gym.envs.base")
    pkg_legged.__path__ = [os.path.abspath(os.path.join(_THIS_DIR, ".."))]
    pkg_envs.__path__ = [os.path.abspath(os.path.join(_THIS_DIR, "..", "envs"))]
    pkg_base.__path__ = [base_dir]
    sys.modules["legged_gym"] = pkg_legged
    sys.modules["legged_gym.envs"] = pkg_envs
    sys.modules["legged_gym.envs.base"] = pkg_base
    base_config_name = "legged_gym.envs.base.base_config"
    if base_config_name not in sys.modules:
        base_config_path = os.path.join(base_dir, "base_config.py")
        spec = importlib.util.spec_from_file_location(base_config_name, base_config_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[base_config_name] = mod
        spec.loader.exec_module(mod)
    cfg_name = "legged_gym.envs.base.legged_robot_config"
    if cfg_name in sys.modules:
        cfg_mod = sys.modules[cfg_name]
    else:
        cfg_path = os.path.join(base_dir, "legged_robot_config.py")
        spec = importlib.util.spec_from_file_location(cfg_name, cfg_path)
        cfg_mod = importlib.util.module_from_spec(spec)
        sys.modules[cfg_name] = cfg_mod
        spec.loader.exec_module(cfg_mod)
    return cfg_mod.LeggedRobotCfg, cfg_mod.LeggedRobotCfgPPO

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if not os.path.isdir(root):
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        model_names = os.listdir(model_parent)
        model_names = [name for name in model_names if os.path.isdir(os.path.join(model_parent, name))]
        for name in model_names:
            if len(name) >= 6:
                if name[:6] == model_name_cand:
                    root = os.path.join(model_parent, name)
    if checkpoint == -1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(root, model)
    return load_path, checkpoint


class HardwareVisionNN(nn.Module):
    def __init__(self, num_prop, num_scan, num_priv_latent, num_priv_explicit,
                 num_hist, num_actions, tanh,
                 actor_hidden_dims=[512, 256, 128],
                 scan_encoder_dims=[128, 64, 32],
                 depth_encoder_hidden_dim=512,
                 activation='elu',
                 priv_encoder_dims=[64, 20]):
        super(HardwareVisionNN, self).__init__()

        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.num_priv_explicit = num_priv_explicit
        num_obs = num_prop + num_scan + num_hist * num_prop + num_priv_latent + num_priv_explicit
        self.num_obs = num_obs
        activation = get_activation(activation)

        self.actor = Actor(
            num_prop, num_scan, num_actions, scan_encoder_dims,
            actor_hidden_dims, priv_encoder_dims, num_priv_latent,
            num_priv_explicit, num_hist, activation,
            tanh_encoder_output=tanh,
            scan_encoder_type="proprio_cross_attention",
            scan_history_len=1,
            scan_attn_d_model=128,
            scan_attn_heads=4,
            scan_attn_layers=2,
        )

        self.estimator = Estimator(input_dim=num_prop, output_dim=num_priv_explicit, hidden_dims=[128, 64])

    def _init_estimator(self, est_state_dict):
        """Auto-detect estimator type (MLP/TCN) from checkpoint keys."""
        is_tcn = any(k.startswith("tcn.") for k in est_state_dict.keys())
        if is_tcn:
            channels = []
            i = 0
            while f"tcn.network.{i}.conv1.weight_v" in est_state_dict:
                channels.append(est_state_dict[f"tcn.network.{i}.conv1.weight_v"].shape[0])
                i += 1
            ks = est_state_dict.get("tcn.network.0.conv1.weight_v")
            kernel_size = ks.shape[2] if ks is not None else 2
            has_unc = "logvar_head.weight" in est_state_dict
            self.estimator = TcnEstimator(
                input_dim=self.num_prop,
                output_dim=self.num_priv_explicit,
                num_channels=channels,
                kernel_size=kernel_size,
                predict_uncertainty=has_unc,
            )
        else:
            hidden = []
            i = 0
            while f"trunk.{2*i}.weight" in est_state_dict:
                hidden.append(est_state_dict[f"trunk.{2*i}.weight"].shape[0])
                i += 1
            if not hidden:
                hidden = [128, 64]
            has_unc = "logvar_head.weight" in est_state_dict
            self.estimator = Estimator(
                input_dim=self.num_prop,
                output_dim=self.num_priv_explicit,
                hidden_dims=hidden,
                predict_uncertainty=has_unc,
            )

    def forward(self, obs, depth_latent):
        obs[:, self.num_prop + self.num_scan:self.num_prop + self.num_scan + self.num_priv_explicit] = self.estimator(obs[:, :self.num_prop])
        return self.actor(obs, hist_encoding=True, eval=False, scandots_latent=depth_latent)


class OnboardPolicyWrapper(nn.Module):
    """Wrap estimator + history_encoder + actor_backbone for onboard deployment.

    Forward: (proprio, proprio_history, depth_latent) -> actions
    """
    def __init__(self, estimator, history_encoder, actor_backbone):
        super().__init__()
        self.estimator = estimator
        self.history_encoder = history_encoder
        self.actor_backbone = actor_backbone

    def forward(self, proprio: torch.Tensor, proprio_history: torch.Tensor, depth_latent: torch.Tensor) -> torch.Tensor:
        lin_vel_latent = self.estimator(proprio)
        priv_latent = self.history_encoder(proprio_history)
        actor_input = torch.cat([proprio, depth_latent, lin_vel_latent, priv_latent], dim=-1)
        return self.actor_backbone(actor_input)


class VisionEncoderJitWrapper(nn.Module):
    """Wrap depth encoder for TorchScript export (stateless)."""

    def __init__(self, depth_encoder: nn.Module, slice_to_dim: int = -1):
        super().__init__()
        self.depth_encoder = depth_encoder
        self.slice_to_dim = int(slice_to_dim)

    def forward(self, depth: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        out = self.depth_encoder(depth, proprio)
        if isinstance(out, tuple):
            out = out[0]
        if self.slice_to_dim > 0:
            return out[:, :self.slice_to_dim]
        return out


class VisionEncoderStatefulJitWrapper(nn.Module):
    """Wrap depth encoder with explicit GRU hidden state I/O for TorchScript.

    Forward: (depth, proprio, hidden) -> (latent, new_hidden)
    """

    def __init__(self, depth_encoder: nn.Module, slice_to_dim: int = -1):
        super().__init__()
        self.depth_encoder = depth_encoder
        self.slice_to_dim = int(slice_to_dim)

    def forward(self, depth: torch.Tensor, proprio: torch.Tensor, hidden: torch.Tensor):
        if hasattr(self.depth_encoder, "forward_with_hidden"):
            out, new_hidden = self.depth_encoder.forward_with_hidden(depth, proprio, hidden)
        else:
            try:
                self.depth_encoder.hidden_states = hidden
            except Exception:
                pass
            out = self.depth_encoder(depth, proprio)
            new_hidden = getattr(self.depth_encoder, "hidden_states", hidden)
        if self.slice_to_dim > 0:
            return out[:, :self.slice_to_dim], new_hidden
        return out, new_hidden


def play(args):
    if args.run_dir:
        load_run = args.run_dir
    else:
        load_run = "../logs/parkour_new/" + args.exptid
    checkpoint = args.checkpoint

    n_priv_explicit = 3 + 3 + 3
    n_priv_latent = 4 + 1 + 12 + 12
    num_scan = 132
    num_actions = 12
    depth_resized = (87, 58)
    n_proprio = 3 + 2 + 3 + 4 + 36 + 4 + 1
    history_len = 10

    device = torch.device('cpu')
    LeggedRobotCfg, LeggedRobotCfgPPO = _load_cfg_classes()
    env_cfg = LeggedRobotCfg()
    ppo_cfg = LeggedRobotCfgPPO()
    temporal_cfg = class_to_dict(ppo_cfg.depth_encoder)
    policy = HardwareVisionNN(n_proprio, num_scan, n_priv_latent, n_priv_explicit, history_len, num_actions, args.tanh).to(device)
    load_path, checkpoint = get_load_path(root=load_run, checkpoint=checkpoint)
    load_run = os.path.dirname(load_path)
    print(f"Loading model from: {load_path}")
    ac_state_dict = torch.load(load_path, map_location=device)

    # Auto-detect temporal encoder type from checkpoint keys
    depth_state = ac_state_dict.get("depth_encoder_state_dict", {})
    if isinstance(depth_state, dict) and depth_state:
        keys = list(depth_state.keys())
        has_temporal = any(k.startswith("temporal_encoder.") for k in keys)
        if has_temporal:
            if any("temporal_encoder.layers" in k for k in keys):
                temporal_cfg["depth_encoder_type"] = "mamba"
            elif any("temporal_encoder.rnn" in k for k in keys):
                temporal_cfg["depth_encoder_type"] = "gru"
        else:
            temporal_cfg = None

    policy.actor.load_state_dict(ac_state_dict['depth_actor_state_dict'], strict=True)
    policy._init_estimator(ac_state_dict['estimator_state_dict'])
    policy.estimator.load_state_dict(ac_state_dict['estimator_state_dict'])

    policy = policy.to(device)
    if not os.path.exists(os.path.join(load_run, "traced")):
        os.mkdir(os.path.join(load_run, "traced"))

    # Save depth encoder weights
    state_dict = {'depth_encoder_state_dict': ac_state_dict['depth_encoder_state_dict']}
    vision_weight_path = os.path.join(load_run, "traced", args.exptid + "-" + str(checkpoint) + "-vision_weight.pt")
    torch.save(state_dict, vision_weight_path)

    # Export depth encoder as TorchScript
    temporal_cfg_dict = temporal_cfg if temporal_cfg is not None else {}
    cnn_channels = temporal_cfg_dict.get("film_spatial_cnn_channels", [32, 64])
    if not isinstance(cnn_channels, (list, tuple)) or len(cnn_channels) != 2:
        cnn_channels = [32, 64]
    depth_backbone = DepthOnlyFCBackbone58x87(
        env_cfg.env.n_proprio, 32, 512, num_frames=1, cnn_channels=cnn_channels,
    ).to(device)
    depth_encoder = RecurrentDepthBackbone(depth_backbone, env_cfg, temporal_cfg=temporal_cfg).to(device)
    depth_encoder.load_state_dict(ac_state_dict['depth_encoder_state_dict'])
    depth_encoder.eval()
    depth_encoder.hidden_states = None

    with torch.no_grad():
        if temporal_cfg is None:
            depth_history_len = 1
        else:
            depth_history_len = int(temporal_cfg_dict.get("depth_history_len", getattr(env_cfg.env, "depth_history_len", 1)))
        if depth_history_len > 1:
            depth_example = torch.ones(1, depth_history_len, 58, 87, device=device)
        else:
            depth_example = torch.ones(1, 58, 87, device=device)
        proprio_example = torch.ones(1, n_proprio, device=device)
        _ = depth_encoder(depth_example, proprio_example)
        depth_encoder.hidden_states = None

        # Stateless vision encoder
        wrapped_depth_encoder = VisionEncoderJitWrapper(depth_encoder, slice_to_dim=-1).to(device)
        traced_depth_encoder = torch.jit.trace(wrapped_depth_encoder, (depth_example, proprio_example), check_trace=False)
        vision_jit_basename = args.exptid + "-" + str(checkpoint) + "-vision_jit.pt"
        vision_jit_path = os.path.join(load_run, "traced", vision_jit_basename)
        traced_depth_encoder.save(vision_jit_path)

        # Stateful vision encoder (explicit GRU hidden)
        stateful_wrapper = VisionEncoderStatefulJitWrapper(depth_encoder, slice_to_dim=-1).to(device)
        hidden_example = torch.zeros(1, 1, 512, device=device)
        traced_stateful = torch.jit.trace(stateful_wrapper, (depth_example, proprio_example, hidden_example), check_trace=False)
        stateful_basename = args.exptid + "-" + str(checkpoint) + "-vision_stateful_jit.pt"
        stateful_path = os.path.join(load_run, "traced", stateful_basename)
        traced_stateful.save(stateful_path)
        print("Saved stateful vision_jit at", os.path.abspath(stateful_path))

        # Baseline GRU-only encoder (no Mamba)
        baseline_backbone = DepthOnlyFCBackbone58x87(
            env_cfg.env.n_proprio, 32, 512, num_frames=1,
            cnn_channels=cnn_channels if "cnn_channels" in locals() else None,
        ).to(device)
        baseline_encoder = RecurrentDepthBackbone(baseline_backbone, env_cfg, temporal_cfg=None).to(device)
        baseline_encoder.load_state_dict(ac_state_dict['depth_encoder_state_dict'], strict=False)
        baseline_encoder.eval()
        baseline_encoder.hidden_states = None
        depth_example_baseline = torch.ones(1, 58, 87, device=device)
        _ = baseline_encoder(depth_example_baseline, proprio_example)
        baseline_encoder.hidden_states = None
        baseline_wrapped = VisionEncoderJitWrapper(baseline_encoder, slice_to_dim=-1).to(device)
        traced_baseline = torch.jit.trace(baseline_wrapped, (depth_example_baseline, proprio_example), check_trace=False)
        baseline_basename = args.exptid + "-" + str(checkpoint) + "-vision_jit_gru.pt"
        baseline_path = os.path.join(load_run, "traced", baseline_basename)
        traced_baseline.save(baseline_path)
        baseline_stateful = VisionEncoderStatefulJitWrapper(baseline_encoder, slice_to_dim=-1).to(device)
        traced_baseline_stateful = torch.jit.trace(baseline_stateful, (depth_example_baseline, proprio_example, hidden_example), check_trace=False)
        baseline_stateful_basename = args.exptid + "-" + str(checkpoint) + "-vision_stateful_jit_gru.pt"
        baseline_stateful_path = os.path.join(load_run, "traced", baseline_stateful_basename)
        traced_baseline_stateful.save(baseline_stateful_path)

    # Verify all exported models can be loaded
    for path in [vision_jit_path, stateful_path, baseline_path, baseline_stateful_path]:
        _ = torch.jit.load(path, map_location="cpu")
        print("Verified:", os.path.abspath(path))

    if args.export_dir:
        os.makedirs(args.export_dir, exist_ok=True)
        for basename, src_path in [
            (vision_jit_basename, vision_jit_path),
            (stateful_basename, stateful_path),
            (baseline_basename, baseline_path),
            (baseline_stateful_basename, baseline_stateful_path),
        ]:
            dst = os.path.join(args.export_dir, basename)
            shutil.copy2(src_path, dst)
            print("Copied to:", os.path.abspath(dst))

    # Trace full policy (obs, depth_latent) -> actions
    policy.eval()
    with torch.no_grad():
        num_envs = 1
        obs_input = torch.ones(num_envs, n_proprio + num_scan + n_priv_explicit + n_priv_latent + history_len * n_proprio, device=device)
        depth_latent = torch.ones(1, 32, device=device)
        _ = policy(obs_input, depth_latent)
        traced_policy = torch.jit.trace(policy, (obs_input, depth_latent))
        save_path = os.path.join(load_run, "traced", args.exptid + "-" + str(checkpoint) + "-base_jit.pt")
        traced_policy.save(save_path)
        print("Saved traced_actor at", os.path.abspath(save_path))

    # Export onboard model (estimator + history_encoder + actor_backbone)
    onboard = OnboardPolicyWrapper(
        estimator=policy.estimator,
        history_encoder=policy.actor.history_encoder,
        actor_backbone=policy.actor.actor_backbone,
    ).to(device)
    onboard.eval()
    with torch.no_grad():
        proprio_example = torch.ones(1, n_proprio, device=device)
        proprio_hist_example = torch.ones(1, history_len, n_proprio, device=device)
        depth_latent_example = torch.ones(1, 32, device=device)
        traced_onboard = torch.jit.trace(onboard, (proprio_example, proprio_hist_example, depth_latent_example))
        onboard_path = os.path.join(load_run, "traced", args.exptid + "-" + str(checkpoint) + "-onboard_jit.pt")
        traced_onboard.save(onboard_path)
        print("Saved onboard_jit at", os.path.abspath(onboard_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exptid', type=str)
    parser.add_argument('--run_dir', type=str, default='', help='Path to run directory with model_*.pt checkpoints')
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--tanh', action='store_true')
    parser.add_argument('--export_dir', type=str, default='')
    args = parser.parse_args()
    play(args)
