# flake8: noqa
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
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
import copy
import numpy as np
import random
# IMPORTANT: isaacgym must be imported before torch
from isaacgym import gymapi
from isaacgym import gymutil
import torch
import argparse
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if not os.path.isdir(root):  # use first 4 chars to mactch the run name
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        model_names = os.listdir(model_parent)
        model_names = [name for name in model_names if os.path.isdir(os.path.join(model_parent, name))]
        for name in model_names:
            if len(name) >= 6:
                if name[:6] == model_name_cand:
                    root = os.path.join(model_parent, name)
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(root, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        if args.use_camera:
            env_cfg.depth.use_camera = args.use_camera
        if getattr(args, "blind_depth", False):
            env_cfg.depth.blind = True
        depth_blind = getattr(env_cfg.depth, "blind", False)
        if depth_blind:
            env_cfg.commands.curriculum = False
            env_cfg.commands.ranges.heading = [0, 0]
            env_cfg.commands.max_ranges.heading = [0, 0]
            env_cfg.commands.ranges.ang_vel_yaw = [0, 0]
            env_cfg.commands.max_ranges.ang_vel_yaw = [0, 0]
        if env_cfg.depth.use_camera and args.headless and not depth_blind:
            env_cfg.env.num_envs = env_cfg.depth.camera_num_envs
            env_cfg.terrain.num_rows = env_cfg.depth.camera_terrain_num_rows
            env_cfg.terrain.num_cols = env_cfg.depth.camera_terrain_num_cols
            env_cfg.terrain.max_error = env_cfg.terrain.max_error_camera
            env_cfg.terrain.horizontal_scale = env_cfg.terrain.horizontal_scale_camera
            env_cfg.terrain.simplify_grid = True
            env_cfg.terrain.terrain_dict["parkour_hurdle"] = 0.2
            env_cfg.terrain.terrain_dict["parkour_flat"] = 0.05
            env_cfg.terrain.terrain_dict["parkour_gap"] = 0.2
            env_cfg.terrain.terrain_dict["parkour_step"] = 0.2
            env_cfg.terrain.terrain_dict["demo"] = 0.15
            env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
        if env_cfg.depth.use_camera and not depth_blind:
            env_cfg.terrain.y_range = [-0.1, 0.1]

        # Blind-zone depth masking experiment overrides (env side)
        if getattr(args, "blind_zone", False):
            env_cfg.depth.blind_zone_enabled = True
        if getattr(args, "blind_full", False):
            env_cfg.depth.blind_full_enabled = True
        if getattr(args, "blind_visible_dist", None) is not None:
            env_cfg.depth.blind_visible_dist = float(args.blind_visible_dist)
        if getattr(args, "blind_mask_dist", None) is not None:
            env_cfg.depth.blind_mask_dist = float(args.blind_mask_dist)
        if getattr(args, "blind_goal_idx", None) is not None:
            env_cfg.depth.blind_goal_idx = int(args.blind_goal_idx)

        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        if args.seed is not None:
            env_cfg.seed = args.seed
        if args.task_both:
            env_cfg.env.task_both = args.task_both
        if args.rows is not None:
            env_cfg.terrain.num_rows = args.rows
        if args.cols is not None:
            env_cfg.terrain.num_cols = args.cols
        if args.delay:
            env_cfg.domain_rand.action_delay = args.delay
        if not args.delay and not args.resume and not args.use_camera and args.headless: # if train from scratch
            env_cfg.domain_rand.action_delay = True
            env_cfg.domain_rand.action_curr_step = env_cfg.domain_rand.action_curr_step_scratch

        # scan history override: update scan_history_len, n_scan, and num_observations accordingly
        if getattr(args, "scan_history_len", None) is not None:
            env_cfg.env.scan_history_len = int(args.scan_history_len)
            # base scan dim from measured points (default 12*11=132)
            try:
                base_scan = int(len(env_cfg.terrain.measured_points_x) * len(env_cfg.terrain.measured_points_y))
            except Exception:
                base_scan = 132
            env_cfg.env.n_scan = int(base_scan * env_cfg.env.scan_history_len)
            env_cfg.env.num_observations = (
                env_cfg.env.n_proprio
                + env_cfg.env.n_scan
                + env_cfg.env.history_len * env_cfg.env.n_proprio
                + env_cfg.env.n_priv_latent
                + env_cfg.env.n_priv
            )

    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.use_camera:
            cfg_train.depth_encoder.if_depth = args.use_camera
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
            cfg_train.algorithm.priv_reg_coef_schedual = cfg_train.algorithm.priv_reg_coef_schedual_resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

        # depth encoder type override (student model temporal encoder)
        if getattr(args, "depth_encoder_type", None) is not None:
            cfg_train.depth_encoder.depth_encoder_type = args.depth_encoder_type

        # scan encoder ablation overrides
        if getattr(args, "scan_encoder_type", None) is not None:
            cfg_train.policy.scan_encoder_type = args.scan_encoder_type
        if getattr(args, "scan_history_len", None) is not None:
            cfg_train.policy.scan_history_len = int(args.scan_history_len)
            # keep estimator.num_scan consistent if exists
            if hasattr(cfg_train, "estimator") and hasattr(cfg_train.estimator, "num_scan"):
                cfg_train.estimator.num_scan = env_cfg.env.n_scan if env_cfg is not None else cfg_train.estimator.num_scan
        if getattr(args, "scan_rnn_hidden", None) is not None:
            cfg_train.policy.scan_rnn_hidden = int(args.scan_rnn_hidden)
        if getattr(args, "scan_attn_d_model", None) is not None:
            cfg_train.policy.scan_attn_d_model = int(args.scan_attn_d_model)
        if getattr(args, "scan_attn_heads", None) is not None:
            cfg_train.policy.scan_attn_heads = int(args.scan_attn_heads)
        if getattr(args, "scan_attn_layers", None) is not None:
            cfg_train.policy.scan_attn_layers = int(args.scan_attn_layers)

        # residual policy overrides
        if getattr(args, "residual_enabled", False):
            cfg_train.policy.residual_enabled = True
        if getattr(args, "residual_mode", None) is not None:
            cfg_train.policy.residual_mode = str(args.residual_mode)
        if getattr(args, "residual_base_checkpoint", None) is not None:
            cfg_train.policy.residual_base_checkpoint = str(args.residual_base_checkpoint)
        if getattr(args, "residual_joint_pos_scale", None) is not None:
            cfg_train.policy.residual_joint_pos_scale = float(args.residual_joint_pos_scale)
        if getattr(args, "residual_action_scale", None) is not None:
            cfg_train.policy.residual_action_scale = float(args.residual_action_scale)
        if getattr(args, "residual_init_zero", False):
            cfg_train.policy.residual_init_zero = True
        if getattr(args, "residual_no_base_history", False):
            cfg_train.policy.residual_base_use_history = False
        elif getattr(args, "residual_base_use_history", False):
            cfg_train.policy.residual_base_use_history = True

        # Ensure residual_action_scale matches the environment PD action_scale.
        # residual joint_pos mode uses: da = dq / action_scale
        # If these scales mismatch, residual magnitude will be wrong by a constant factor.
        #
        # NOTE: Some call sites pass env_cfg=None (e.g. make_alg_runner warm-start path),
        # so guard access to env_cfg.control.
        if getattr(cfg_train.policy, "residual_enabled", False) and env_cfg is not None:
            env_action_scale = float(
                getattr(env_cfg.control, "action_scale", getattr(cfg_train.policy, "residual_action_scale", 1.0))
            )
            cur = float(getattr(cfg_train.policy, "residual_action_scale", env_action_scale))
            if abs(cur - env_action_scale) > 1e-6:
                if getattr(args, "residual_action_scale", None) is None:
                    print(
                        f"[cfg] WARNING residual_action_scale={cur} != env.control.action_scale={env_action_scale}. "
                        f"Auto-fixing residual_action_scale -> {env_action_scale}."
                    )
                    cfg_train.policy.residual_action_scale = env_action_scale
                else:
                    print(
                        f"[cfg] WARNING residual_action_scale={cur} != env.control.action_scale={env_action_scale}. "
                        f"(Kept CLI override; make sure you intend this.)"
                    )

        # AMP overrides (algorithm side)
        if getattr(args, "amp_enabled", False):
            cfg_train.algorithm.amp_enabled = True
        if getattr(args, "amp_demo_path", None) is not None:
            cfg_train.algorithm.amp_demo_path = str(args.amp_demo_path)
        if getattr(args, "amp_reward_coef", None) is not None:
            cfg_train.algorithm.amp_reward_coef = float(args.amp_reward_coef)
        if getattr(args, "amp_warmup_iters", None) is not None:
            cfg_train.algorithm.amp_warmup_iters = int(args.amp_warmup_iters)
        if getattr(args, "amp_ramp_iters", None) is not None:
            cfg_train.algorithm.amp_ramp_iters = int(args.amp_ramp_iters)
        if getattr(args, "amp_disc_lr", None) is not None:
            cfg_train.algorithm.amp_disc_lr = float(args.amp_disc_lr)
        if getattr(args, "amp_disc_updates", None) is not None:
            cfg_train.algorithm.amp_disc_updates = int(args.amp_disc_updates)
        if getattr(args, "amp_disc_batch_size", None) is not None:
            cfg_train.algorithm.amp_disc_batch_size = int(args.amp_disc_batch_size)
        if getattr(args, "amp_disc_hidden_dims", None) is not None:
            # Accept "256,256" or "[256,256]"
            s = str(args.amp_disc_hidden_dims).strip()
            s = s.strip("[]()")
            dims = []
            if s:
                for tok in s.split(","):
                    tok = tok.strip()
                    if not tok:
                        continue
                    dims.append(int(tok))
            if dims:
                cfg_train.algorithm.amp_disc_hidden_dims = dims
        # depth distillation warm-start
        if getattr(args, "distill_base_checkpoint", None) is not None:
            cfg_train.depth_encoder.distill_base_checkpoint = str(args.distill_base_checkpoint)

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "go2", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int, "default": -1, "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--device", "type": str, "default": "cuda:0", "help": 'Device for sim, rl, and graphics'},

        {"name": "--rows", "type": int, "help": "num_rows."},
        {"name": "--cols", "type": int, "help": "num_cols"},
        {"name": "--debug", "action": "store_true", "default": False, "help": "Disable wandb logging"},
        {"name": "--proj_name", "type": str,  "default": "parkour_new", "help": "run folder name."},
        
        {"name": "--teacher", "type": str, "help": "Name of the teacher policy to use when distilling"},
        {"name": "--exptid", "type": str, "help": "exptid"},
        {"name": "--resumeid", "type": str, "help": "exptid"},
        {"name": "--daggerid", "type": str, "help": "name of dagger run"},
        {"name": "--use_camera", "action": "store_true", "default": False, "help": "render camera for distillation"},
        {"name": "--mask_obs", "action": "store_true", "default": False, "help": "Mask observation when playing"},
        {"name": "--use_jit", "action": "store_true", "default": False, "help": "Load jit script when playing"},
        {"name": "--use_latent", "action": "store_true", "default": False, "help": "Load depth latent when playing"},
        {"name": "--jit_prefix", "type": str, "default": "", "help": "Prefix of traced jit filenames (e.g., 'our_student' for 'our_student-9500-base_jit.pt'). Default: use exptid."},
        {"name": "--keep_terrain", "action": "store_true", "default": False, "help": "Keep original terrain settings (may be heavy). For play_jit quick playback, default is flat plane."},

        # depth encoder type override (student model temporal encoder)
        {"name": "--depth_encoder_type", "type": str, "default": None, "help": "Override depth_encoder.depth_encoder_type (mamba|transformer|self_attention|gru|lstm|mlp_concat)"},

        # scan encoder ablations (training-time overrides; do not require editing config file)
        {"name": "--scan_encoder_type", "type": str, "default": None, "help": "Override policy.scan_encoder_type (mlp_concat|gru|lstm|self_attention|proprio_cross_attention)"},
        {"name": "--scan_history_len", "type": int, "default": None, "help": "Override env.scan_history_len (n frames; scan dim becomes 132*n)"},
        {"name": "--scan_rnn_hidden", "type": int, "default": None, "help": "Override policy.scan_rnn_hidden (for gru/lstm)"},
        {"name": "--scan_attn_d_model", "type": int, "default": None, "help": "Override policy.scan_attn_d_model (for attention encoders)"},
        {"name": "--scan_attn_heads", "type": int, "default": None, "help": "Override policy.scan_attn_heads (for attention encoders)"},
        {"name": "--scan_attn_layers", "type": int, "default": None, "help": "Override policy.scan_attn_layers (for attention encoders)"},
        {"name": "--distill_base_checkpoint", "type": str, "default": None, "help": "Warm-start distillation from checkpoint path (no optimizer)"},

        # residual policy (frozen BC base + trainable residual)
        {"name": "--residual_enabled", "action": "store_true", "default": False, "help": "Enable residual policy (frozen BC base + trainable residual)"},
        {"name": "--residual_mode", "type": str, "default": None, "help": "Residual mode: joint_pos|action"},
        {"name": "--residual_base_checkpoint", "type": str, "default": None, "help": "Path to BC checkpoint (bc_policy_*.pt) used as frozen base"},
        {"name": "--residual_joint_pos_scale", "type": float, "default": None, "help": "Max |delta-q| (radians) for joint_pos mode (after tanh)"},
        {"name": "--residual_action_scale", "type": float, "default": None, "help": "Env action_scale used to convert delta-q(rad) -> delta-action"},
        {"name": "--residual_init_zero", "action": "store_true", "default": False, "help": "Init residual last layer to 0 so initial policy == base"},
        {"name": "--residual_base_use_history", "action": "store_true", "default": False, "help": "Base policy uses proprio history (if present)"},
        {"name": "--residual_no_base_history", "action": "store_true", "default": False, "help": "Base policy does NOT use history (override)"},

        {"name": "--num_trials", "type": int, "default": 100, "help": "Number of trials per terrain type"},
        {"name": "--num_robots", "type": int, "default": 5, "help": "Number of robots per terrain type"},
        {"name": "--eval_like", "action": "store_true", "default": False, "help": "Play with evaluation-like env override (multi-terrain columns, consistent episode length)."},
        {"name": "--mask_priv_obs", "action": "store_true", "default": False, "help": "Mask priv_explicit states in actor obs (deployment-like)."},
        {"name": "--exp_name", "type": str, "help": "Experiment name for report"},
        {"name": "--output_path", "type": str, "help": "Output path for the report"},
        {"name": "--terrains", "type": str, "default": None, "help": "Comma-separated terrain names to evaluate. If None: evaluate terrains with non-zero proportion."},
        {"name": "--include_zero_terrains", "action": "store_true", "default": False, "help": "If set: evaluate all terrains in terrain_dict (including zero proportion)."},
        {"name": "--single_terrain", "type": str, "default": None, "help": "Internal use: evaluate a single terrain name (worker mode)."},
        {"name": "--output_json", "type": str, "default": None, "help": "Internal use: write single-terrain metrics to this json file (worker mode)."},
        {"name": "--draw", "action": "store_true", "default": False, "help": "draw debug plot when playing"},
        {"name": "--save", "action": "store_true", "default": False, "help": "save data for evaluation"},

        {"name": "--record_video", "action": "store_true", "default": False, "help": "Record videos per terrain type during play"},
        {"name": "--videos_per_terrain", "type": int, "default": 2, "help": "Number of videos per terrain (only with --record_video)"},
        {"name": "--video_dir", "type": str, "default": None, "help": "Video save directory (default: logs/<proj>/<exptid>/play_videos)"},
        {"name": "--blind_depth", "action": "store_true", "default": False, "help": "Zero out depth input (blind mode)"},
        {"name": "--log_root", "type": str, "default": None, "help": "Log root directory (default: LEGGED_GYM_ROOT_DIR/logs)"},
        {"name": "--video_label", "type": str, "default": None, "help": "Label text on video overlay (default: exptid)"},
        {"name": "--difficulty_levels", "type": str, "default": "", "help": "Terrain difficulty levels for video, comma-separated e.g. 0,3,6,9"},
        {"name": "--difficulty_level", "type": int, "default": 3, "help": "Terrain difficulty for play_jit video (0=easiest)"},
        {"name": "--terrain", "type": str, "default": None, "help": "Single terrain name for video recording (e.g. parkour_hurdle)"},
        {"name": "--record_all_levels", "action": "store_true", "default": False, "help": "Record videos for all difficulty levels in sequence"},
        {"name": "--record_depth_overlay", "action": "store_true", "default": False, "help": "Overlay depth visualization in recorded videos"},
        {"name": "--record_split_blind", "action": "store_true", "default": False, "help": "Split video by blind/before-blind for blind-zone experiment"},
        {"name": "--video_depth_blind", "action": "store_true", "default": False, "help": "Blind only video depth overlay (does not affect env obs)"},
        {"name": "--video_depth_blind_dist", "type": float, "default": 1.0, "help": "Video depth blind distance threshold (m)"},
        {"name": "--video_depth_only", "action": "store_true", "default": False, "help": "Use depth only for video overlay, not for policy inference"},
        {"name": "--record_scandots", "action": "store_true", "default": False, "help": "Overlay red scandots (height samples) in recorded videos"},

        # Sensor degradation (affects play/eval only, not training)
        {"name": "--deg_enable", "action": "store_true", "default": False, "help": "Enable sensor degradation during evaluation/play"},
        {"name": "--deg_target", "type": str, "default": "both", "help": "Degradation target: depth|scandot|both"},
        {"name": "--deg_seed", "type": int, "default": 0, "help": "Random seed for degradation (reproducible)"},

        # Random dropout / Gaussian noise
        {"name": "--deg_p_drop", "type": float, "default": 0.0, "help": "Per-step probability to drop sensor input (set to zeros)"},
        {"name": "--deg_gauss_std", "type": float, "default": 0.0, "help": "Gaussian noise std added to sensor input"},

        # Block occlusion (depth: rect; scandot: consecutive dim segments)
        {"name": "--deg_occ_p", "type": float, "default": 0.0, "help": "Per-step probability to trigger block occlusion"},
        {"name": "--deg_occ_size", "type": float, "default": 0.3, "help": "Occlusion size ratio (0~1). depth: H/W ratio, scandot: last-dim ratio"},
        {"name": "--deg_occ_len", "type": int, "default": 1, "help": "Occlusion duration in steps"},

        # Outage: continuous L steps of zeros after trigger
        {"name": "--deg_outage_p", "type": float, "default": 0.0, "help": "Per-step probability to trigger outage"},
        {"name": "--deg_outage_len", "type": int, "default": 0, "help": "Outage duration in steps"},

        # Fixed delay: output sensor from t-k
        {"name": "--deg_delay_steps", "type": int, "default": 0, "help": "Fixed delay steps (output from t-k). Use small values for depth to save memory."},

        # Degradation visualization/export (debug)
        {"name": "--deg_vis_scandot", "action": "store_true", "default": False, "help": "Save scan_dot before/after corruption as images for debugging"},
        {"name": "--deg_vis_depth", "action": "store_true", "default": False, "help": "Save depth before/after corruption as images for debugging"},
        {"name": "--deg_vis_every", "type": int, "default": 50, "help": "Save visualization every N env steps"},
        {"name": "--deg_vis_env", "type": int, "default": 0, "help": "Which env id to visualize"},
        {"name": "--deg_vis_max_frames", "type": int, "default": 50, "help": "Max number of frames to save"},

        # Blind-zone experiment (for validation of memory modules)
        {"name": "--blind_zone", "action": "store_true", "default": False, "help": "Enable blind-zone depth masking experiment"},
        {"name": "--blind_full", "action": "store_true", "default": False, "help": "Enable full depth outage (all frames masked)"},
        {"name": "--blind_visible_dist", "type": float, "default": None, "help": "Visible window distance (m) for blind-zone experiment"},
        {"name": "--blind_mask_dist", "type": float, "default": None, "help": "Mask distance (m) for blind-zone experiment"},
        {"name": "--blind_goal_idx", "type": int, "default": None, "help": "Only apply blind-zone to this goal index (default: all goals)"},

        {"name": "--task_both", "action": "store_true", "default": False, "help": "Both climbing and hitting policies"},
        {"name": "--nodelay", "action": "store_true", "default": False, "help": "Add action delay"},
        {"name": "--delay", "action": "store_true", "default": False, "help": "Add action delay"},
        {"name": "--hitid", "type": str, "default": None, "help": "exptid fot hitting policy"},

        {"name": "--web", "action": "store_true", "default": False, "help": "if use web viewer"},
        {"name": "--no_wandb", "action": "store_true", "default": False, "help": "no wandb"},

        # wandb
        {"name": "--wandb_entity", "type": str, "default": None, "help": "Weights & Biases entity (username or team). If None: use current account default."},

        # BC Warm Start
        {"name": "--load_bc_checkpoint", "type": str, "default": None, "help": "Path to BC pretrained checkpoint for warm start"},
        # Estimator pretrain: load standalone estimator weights at parkour start
        {"name": "--load_estimator_checkpoint", "type": str, "default": None, "help": "Path to standalone-trained estimator state_dict (.pt). Load into alg.estimator before full parkour training."},
        {"name": "--load_estimator_run", "type": str, "default": None, "help": "Load estimator from another run (resumeid) under logs/<proj_name>/"},
        {"name": "--load_estimator_run_checkpoint", "type": int, "default": -1, "help": "Checkpoint index for load_estimator_run (-1 for latest)"},

        # Bayesian / HPO (train_bayesian_hpo.py)
        {"name": "--n_trials", "type": int, "default": 10, "help": "HPO: number of Optuna trials"},
        {"name": "--max_iterations_per_trial", "type": int, "default": 0, "help": "HPO: max training iterations per trial (0=use runner default)"},
        {"name": "--hpo_study_name", "type": str, "default": "estimator_robustness", "help": "HPO: Optuna study name"},
        {"name": "--hpo_mode", "type": str, "default": "estimator_robustness", "help": "HPO: estimator_robustness | estimator_fusion"},
        {"name": "--hpo_log_root", "type": str, "default": "/data/parkour_logs/hpo", "help": "HPO: log root for trial runs"},
        {"name": "--hpo_storage", "type": str, "default": None, "help": "HPO: Optuna storage URL (e.g. sqlite:///hpo.db) for persistence"},

        # Standalone train/test estimator (train_estimator_standalone.py)
        {"name": "--est_standalone_mode", "type": str, "default": "train", "help": "Estimator standalone: train | test | ablation(train+test) | ablation_test(test only, load existing A/B)"},
        {"name": "--est_standalone_steps", "type": int, "default": 50000, "help": "Estimator: number of samples to collect/evaluate (each = one (proprio, priv) per env; parallel envs yield num_envs per step)"},
        {"name": "--est_standalone_epochs", "type": int, "default": 100, "help": "Estimator train: epochs over collected buffer (default heuristic; usually sufficient for ~50k steps; adjust as needed)"},
        {"name": "--est_standalone_batch", "type": int, "default": 256, "help": "Estimator train: minibatch size"},
        {"name": "--est_standalone_checkpoint", "type": str, "default": "estimator_standalone.pt", "help": "Estimator: save (train) or load (test) path"},
        {"name": "--est_standalone_uncertainty", "action": "store_true", "default": False, "help": "Estimator train: use uncertainty head and NLL loss"},
        {"name": "--est_use_kf", "action": "store_true", "default": False, "help": "Estimator test: use Kalman Filter for fusion (requires uncertainty output)"},
        {"name": "--est_policy_log_root", "type": str, "default": None, "help": "Log root when using policy for estimator data collection (default: LEGGED_GYM_ROOT_DIR/logs). If training writes to /data/parkour_logs, pass --est_policy_log_root /data/parkour_logs"},
        {"name": "--est_ablation_dir", "type": str, "default": "estimator_ablation", "help": "Output directory for saving and reporting three models when est_standalone_mode=ablation"},
        {"name": "--est_standalone_priv_weights", "type": str, "default": None, "help": "Per-dimension loss weights for estimator training, comma-separated (e.g. 1,1,2 for first 3 dims; z gets higher weight). Missing dims use 1.0. Helps mitigate large base_lin_vel_z error"},
        {"name": "--est_model_type", "type": str, "default": "resnet", "help": "Estimator model type: mlp | resnet | tcn"},
    ]
    # parse arguments
    args = parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path, name):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, name+".pt")
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


def get_latest_model_path(run_dir, checkpoint=-1, model_prefix="model_"):
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    ckpt = int(checkpoint) if checkpoint is not None else -1
    if ckpt >= 0:
        path = os.path.join(run_dir, f"{model_prefix}{ckpt}.pt")
        if os.path.isfile(path):
            return path
        candidates = [
            f for f in os.listdir(run_dir)
            if f.startswith(model_prefix) and f.endswith(".pt") and f"{model_prefix}{ckpt}" in f
        ]
        if candidates:
            candidates.sort(key=lambda m: '{0:0>15}'.format(m))
            return os.path.join(run_dir, candidates[-1])
        raise FileNotFoundError(f"Checkpoint not found in {run_dir}: {ckpt}")
    models = [f for f in os.listdir(run_dir) if f.startswith(model_prefix) and f.endswith(".pt")]
    if not models:
        raise FileNotFoundError(f"No model checkpoints in {run_dir}")
    models.sort(key=lambda m: '{0:0>15}'.format(m))
    return os.path.join(run_dir, models[-1])


def load_estimator_state_dict(path, device, target_state_dict=None):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "estimator_state_dict" in ckpt:
        state = ckpt["estimator_state_dict"]
    else:
        state = ckpt
    info = {"loaded": 0, "skipped_size_mismatch": 0, "total": 0}
    if target_state_dict is not None and isinstance(state, dict):
        filtered = {}
        skipped = 0
        for k, v in state.items():
            if k in target_state_dict and hasattr(v, "shape") and hasattr(target_state_dict[k], "shape"):
                if tuple(v.shape) == tuple(target_state_dict[k].shape):
                    filtered[k] = v
                else:
                    skipped += 1
        info = {"loaded": len(filtered), "skipped_size_mismatch": skipped, "total": len(state)}
        state = filtered
    return state, path, info


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    
# overide gymutil
def parse_device_str(device_str):
    # defaults
    device = 'cpu'
    device_id = 0

    if device_str == 'cpu' or device_str == 'cuda':
        device = device_str
        device_id = 0
    else:
        device_args = device_str.split(':')
        assert len(device_args) == 2 and device_args[0] == 'cuda', f'Invalid device string "{device_str}"'
        device, device_id_s = device_args
        try:
            device_id = int(device_id_s)
        except ValueError:
            raise ValueError(f'Invalid device string "{device_str}". Cannot parse "{device_id}"" as a valid device id')
    return device, device_id

def parse_arguments(description="Isaac Gym Example", headless=False, no_graphics=False, custom_parameters=[]):
    parser = argparse.ArgumentParser(description=description)
    if headless:
        parser.add_argument('--headless', action='store_true', help='Run headless without creating a viewer window')
    if no_graphics:
        parser.add_argument('--nographics', action='store_true',
                            help='Disable graphics context creation, no viewer window is created, and no headless rendering is available')
    parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
    parser.add_argument('--pipeline', type=str, default="gpu", help='Tensor API pipeline (cpu/gpu)')
    parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')

    physics_group = parser.add_mutually_exclusive_group()
    physics_group.add_argument('--flex', action='store_true', help='Use FleX for physics')
    physics_group.add_argument('--physx', action='store_true', help='Use PhysX for physics')

    parser.add_argument('--num_threads', type=int, default=0, help='Number of cores used by PhysX')
    parser.add_argument('--subscenes', type=int, default=0, help='Number of PhysX subscenes to simulate in parallel')
    parser.add_argument('--slices', type=int, help='Number of client threads that process env slices')

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                else:
                    parser.add_argument(argument["name"], type=argument["type"], help=help_str)
            elif "action" in argument:
                # For action="store_true", argparse automatically sets default=False
                # Only pass default if it's explicitly provided and not False
                if "default" in argument and argument["default"] is not False:
                    parser.add_argument(argument["name"], action=argument["action"], default=argument["default"], help=help_str)
                else:
                    parser.add_argument(argument["name"], action=argument["action"], help=help_str)

        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()

    args = parser.parse_args()

    if args.device is not None:
        args.sim_device = args.device
        args.rl_device = args.device
    args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)
    pipeline = args.pipeline.lower()

    assert (pipeline == 'cpu' or pipeline in ('gpu', 'cuda')), f"Invalid pipeline '{args.pipeline}'. Should be either cpu or gpu."
    args.use_gpu_pipeline = (pipeline in ('gpu', 'cuda'))

    if args.sim_device_type != 'cuda' and args.flex:
        print("Can't use Flex with CPU. Changing sim device to 'cuda:0'")
        args.sim_device = 'cuda:0'
        args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)

    if (args.sim_device_type != 'cuda' and pipeline == 'gpu'):
        print("Can't use GPU pipeline with CPU Physics. Changing pipeline to 'CPU'.")
        args.pipeline = 'CPU'
        args.use_gpu_pipeline = False

    # Default to PhysX
    args.physics_engine = gymapi.SIM_PHYSX
    args.use_gpu = (args.sim_device_type == 'cuda')

    if args.flex:
        args.physics_engine = gymapi.SIM_FLEX

    # Using --nographics implies --headless
    if no_graphics and args.nographics:
        args.headless = True

    if args.slices is None:
        args.slices = args.subscenes

    return args
