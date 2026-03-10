# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# flake8: noqa
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *  # noqa: F403
from legged_gym.utils import get_args, task_registry, load_estimator_state_dict
import wandb


def _short_scan_type(scan_encoder_type: str) -> str:
    m = {
        "mlp_concat": "mlp_concat",
        "mlp": "mlp_concat",
        "gru": "gru",
        "lstm": "lstm",
        "self_attention": "selfattn",
        "self_attn": "selfattn",
        "proprio_cross_attention": "pcrossattn",
        "proprio_cross_attn": "pcrossattn",
        "cross_attention_proprio": "pcrossattn",
    }
    if scan_encoder_type in m:
        return m[scan_encoder_type]
    # fallback: sanitize a bit
    s = str(scan_encoder_type).strip().lower()
    s = s.replace(" ", "_")
    return s[:24] if len(s) > 24 else s


def _auto_exptid(args) -> str:
    """
    ex-{task}-scan{type}-n{n}-seed{seed}-{MMDD-HHMM}
    """
    # pull defaults from registered cfgs (user edits config file -> reflected here)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    n = int(getattr(env_cfg.env, "scan_history_len", 1))
    scan_type = getattr(train_cfg.policy, "scan_encoder_type", "mlp_concat")
    scan_type_short = _short_scan_type(scan_type)

    seed = args.seed if getattr(args, "seed", None) is not None else getattr(train_cfg, "seed", 1)
    ts = datetime.now().strftime("%m%d-%H%M")
    return f"ex-{args.task}-scan{scan_type_short}-n{n}-seed{seed}-{ts}"


def train(args):
    # headless = True
    args.headless = True
    # Respect CLI headless flag. (Previously forced False, which is surprising and can waste resources.)

    # Auto-generate exptid when not manually specified (convenient for ablation comparison)
    if not getattr(args, "exptid", None):
        args.exptid = _auto_exptid(args)
        print(f"[train] auto exptid: {args.exptid}")

    log_root = "/data/parkour_logs"
    try:
        os.makedirs(log_root)
    except Exception:
        pass
    log_pth = os.path.join(log_root, args.proj_name, args.exptid)
    try:
        os.makedirs(log_pth)
    except Exception:
        pass
    if args.debug:
        mode = "disabled"
        args.rows = 10
        args.cols = 8
        args.num_envs = 64
    else:
        mode = "online"
    
    if args.no_wandb:
        mode = "disabled"
    # NOTE:
    # - mode="online": live upload to W&B (requires `wandb login` beforehand)
    # - mode="disabled": completely disable W&B (for debug / offline environments)
    wandb_init_kwargs = dict(
        project=args.proj_name,
        name=args.exptid,
        group=args.exptid[:3],
        mode=mode,
        dir="/data/parkour_logs",
    )
    # Don't hardcode entity; it easily causes 403 permission errors
    if getattr(args, "wandb_entity", None):
        wandb_init_kwargs["entity"] = args.wandb_entity
    wandb.init(**wandb_init_kwargs)
    wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", policy="now")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py", policy="now")

    # Load configs first so we can apply lightweight debug overrides before building the sim.
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # Optional: mask privileged explicit obs (deployment-like). When enabled, allocate privileged_obs_buf to keep unmasked copy.
    if getattr(args, "mask_priv_obs", False):
        env_cfg.play.mask_priv_obs = True
        env_cfg.env.num_privileged_obs = env_cfg.env.num_observations
    # If we are doing a short smoke-test (e.g. to validate AMP plumbing), use a light flat setup to avoid GPU OOM
    # from large trimesh terrains + huge num_envs.
    try:
        max_it = int(getattr(args, "max_iterations", 0) or 0)
    except Exception:
        max_it = 0
    if max_it > 0 and max_it <= 10:
        # flatten terrain
        env_cfg.terrain.mesh_type = "plane"
        env_cfg.terrain.measure_heights = False
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = 1
        env_cfg.terrain.height = [0.0, 0.0]
        env_cfg.terrain.terrain_dict = {"smooth flat": 1.0}
        env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected = False
        # keep it stable (AMP demo should dominate style, not randomization)
        env_cfg.noise.add_noise = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_base_com = False
        if hasattr(env_cfg.domain_rand, "randomize_motor"):
            env_cfg.domain_rand.randomize_motor = False
        # reduce default env count unless user explicitly set --num_envs
        if getattr(args, "num_envs", None) is None:
            env_cfg.env.num_envs = min(int(getattr(env_cfg.env, "num_envs", 256)), 256)

        # IMPORTANT:
        # When measure_heights=False, LeggedRobot will not append terrain scan to observations.
        # So we must also update the configured observation dimensions to avoid ActorCritic shape mismatch.
        try:
            env_cfg.env.n_scan = 0
            env_cfg.env.num_observations = (
                int(env_cfg.env.n_proprio)
                + int(env_cfg.env.n_scan)
                + int(env_cfg.env.history_len) * int(env_cfg.env.n_proprio)
                + int(env_cfg.env.n_priv_latent)
                + int(env_cfg.env.n_priv)
            )
            # If privileged obs is enabled and was tied to obs dim, keep it consistent.
            if getattr(env_cfg.env, "num_privileged_obs", None) is not None:
                env_cfg.env.num_privileged_obs = int(env_cfg.env.num_observations)
        except Exception as e:
            print(f"[train][debug] WARNING: failed to recompute obs dims after disabling heights: {e}")

    # Create env/runner with (possibly) overridden cfgs.
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root=log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg)

    # Warm Start with BC Checkpoint
    if getattr(args, "load_bc_checkpoint", None):
        print(f"[Warm Start] Loading BC checkpoint from {args.load_bc_checkpoint}...")
        try:
            import torch # Ensure torch is imported
            bc_checkpoint = torch.load(args.load_bc_checkpoint, map_location=ppo_runner.device)
            # Load only model_state_dict (ActorCritic)
            # strict=False to allow missing keys (e.g. if BC didn't save estimator/critic fully matching)
            # But BC script uses same ActorCritic structure, so it should match mostly.
            # Check if key is 'model_state_dict' or direct state_dict
            if 'model_state_dict' in bc_checkpoint:
                state_dict = bc_checkpoint['model_state_dict']
            else:
                state_dict = bc_checkpoint
                
            # Load only parameters with matching name + shape.
            # This allows reusing π_base across different scan_history_len / scan_encoder_type
            # without failing on size mismatch.
            target = ppo_runner.alg.actor_critic
            target_sd = target.state_dict()
            filtered = {}
            skipped = 0
            for k, v in state_dict.items():
                if k in target_sd and hasattr(v, "shape") and hasattr(target_sd[k], "shape"):
                    if tuple(v.shape) == tuple(target_sd[k].shape):
                        filtered[k] = v
                    else:
                        skipped += 1
            missing, unexpected = target.load_state_dict(filtered, strict=False)
            print(
                f"[Warm Start] Loaded {len(filtered)} tensors; "
                f"skipped_size_mismatch={skipped}, missing={len(missing)}, unexpected={len(unexpected)}"
            )
            print("[Warm Start] BC weights loaded successfully.")
        except Exception as e:
            print(f"[Warm Start] Failed to load BC checkpoint: {e}")
            raise e

    # Load standalone-trained estimator weights if provided
    if getattr(args, "load_estimator_checkpoint", None):
        path = args.load_estimator_checkpoint
        print(f"[Warm Start] Loading standalone-trained estimator from {path}...")
        try:
            state, _, info = load_estimator_state_dict(path, ppo_runner.device, ppo_runner.alg.estimator.state_dict())
            out = ppo_runner.alg.estimator.load_state_dict(state, strict=False)
            miss = getattr(out, "missing_keys", [])
            unexp = getattr(out, "unexpected_keys", [])
            print(
                f"[Warm Start] Estimator loaded. loaded={info.get('loaded')}, skipped_size_mismatch={info.get('skipped_size_mismatch')}, "
                f"missing_keys={len(miss)}, unexpected_keys={len(unexp)}"
            )
            if miss:
                print("  missing:", miss[:5], "..." if len(miss) > 5 else "")
            if unexp:
                print("  unexpected:", unexp[:5], "..." if len(unexp) > 5 else "")
        except Exception as e:
            print(f"[Warm Start] Failed to load estimator: {e}")
            raise e

    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    # Log configs immediately
    args = get_args()
    train(args)
