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
import sys

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_LOCAL_RSL = os.path.join(_ROOT_DIR, "rsl_rl")
if os.path.isdir(_LOCAL_RSL) and _LOCAL_RSL not in sys.path:
    sys.path.insert(0, _LOCAL_RSL)

from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger, get_latest_model_path, load_estimator_state_dict
from legged_gym.utils.task_registry import _load_checkpoint_filtered
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
from legged_gym.utils import webviewer
from legged_gym.utils.math import quat_apply_yaw

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

def _detect_policy_class_name(checkpoint_path: str):
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if isinstance(state, dict):
            for k in state.keys():
                if k.startswith("actor.actor_backbone"):
                    return "ActorCriticRMA"
                if k.startswith("actor.0."):
                    return "ActorCritic"
    except Exception:
        return None
    return None

def _ensure_policy_class_available(policy_class_name: str):
    try:
        import rsl_rl.runners.on_policy_runner as runner_mod
        if hasattr(runner_mod, policy_class_name):
            return
        if policy_class_name == "ActorCriticRMA":
            from rsl_rl.modules.actor_critic import ActorCriticRMA
            setattr(runner_mod, "ActorCriticRMA", ActorCriticRMA)
    except Exception:
        return

def _parse_terrain_names(args, env_cfg):
    raw = str(getattr(args, "terrains", "") or "").strip()
    if raw:
        names = [s.strip() for s in raw.split(",") if s.strip()]
        return names
    base_keys = list(getattr(env_cfg.terrain, "terrain_dict", {}).keys())
    if getattr(args, "include_zero_terrains", False):
        return base_keys
    names = []
    for k, v in getattr(env_cfg.terrain, "terrain_dict", {}).items():
        if float(v) > 0:
            names.append(k)
    return names

def _override_env_cfg_for_single_terrain(env_cfg, terrain_name: str, num_envs: int, keep_rows: bool = False):
    env_cfg.env.num_envs = int(num_envs)
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = False
    env_cfg.terrain.num_cols = 1
    if not keep_rows:
        env_cfg.terrain.num_rows = 1
    base_keys = list(getattr(env_cfg.terrain, "terrain_dict", {}).keys())
    if not base_keys:
        raise ValueError("env_cfg.terrain.terrain_dict is empty; cannot build terrain proportions.")
    new_dict = {}
    for k in base_keys:
        new_dict[k] = 1.0 if k == terrain_name else 0.0
    env_cfg.terrain.terrain_dict = new_dict
    env_cfg.terrain.terrain_proportions = list(new_dict.values())

def _parse_difficulty_levels(args, env_cfg):
    max_level = int(getattr(env_cfg.terrain, "num_rows", 1)) - 1
    if max_level < 0:
        max_level = 0
    raw = str(getattr(args, "difficulty_levels", "") or "").strip()
    if raw:
        levels = []
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                lvl = int(tok)
            except Exception:
                continue
            lvl = max(0, min(lvl, max_level))
            if lvl not in levels:
                levels.append(lvl)
        if levels:
            return levels
    if bool(getattr(args, "record_all_levels", False)):
        return list(range(max_level + 1))
    if max_level <= 0:
        return [0]
    mid = max_level // 2
    levels = [0, mid, max_level]
    out = []
    for lvl in levels:
        if lvl not in out:
            out.append(lvl)
    return out

def _resolve_video_dir(args, exptid: str, proj_name: str):
    out_dir = str(getattr(args, "video_dir", "") or "")
    if out_dir:
        return out_dir
    return os.path.join(LEGGED_GYM_ROOT_DIR, "logs", proj_name, exptid, "play_videos")

def _should_use_depth_in_policy(args, env_cfg):
    return bool(getattr(args, "use_camera", False)) and bool(getattr(env_cfg.depth, "use_camera", False)) and (not bool(getattr(args, "video_depth_only", False)))

def _get_fps(env):
    dt = float(getattr(env, "dt", 0.02) or 0.02)
    if dt <= 0:
        dt = 0.02
    fps = int(round(1.0 / dt))
    return max(1, fps)

def _init_video_writer(path: str, width: int, height: int, fps: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, float(fps), (int(width), int(height)))

def _capture_frame(env, cam_handle, width: int, height: int):
    gym = env.gym
    sim = env.sim
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    image = gym.get_camera_image(sim, env.envs[0], cam_handle, gymapi.IMAGE_COLOR)
    frame = image.reshape(height, width, 4)[..., :3]
    return frame

def _project_points_to_pixels(pts_world, view_mat, proj_mat, W, H):
    N = pts_world.shape[0]
    homo = np.hstack([pts_world, np.ones((N, 1))])
    cam = (view_mat @ homo.T).T
    clip = (proj_mat @ cam.T).T
    behind = clip[:, 3] <= 0
    w = np.where(~behind, clip[:, 3], 1.0)
    ndc_x = clip[:, 0] / w
    ndc_y = clip[:, 1] / w
    u = ((ndc_x + 1.0) * 0.5 * W).astype(int)
    v = ((1.0 - (ndc_y + 1.0) * 0.5) * H).astype(int)
    in_bounds = (~behind) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return np.stack([u, v], axis=-1), in_bounds


def _get_scandot_world_positions(env, env_idx=0):
    base_pos = env.root_states[env_idx, :3].cpu().numpy()
    heights = env.measured_heights[env_idx].cpu().numpy()
    hp_local = env.height_points[env_idx]
    hp_world = quat_apply_yaw(
        env.base_quat[env_idx].repeat(hp_local.shape[0]),
        hp_local,
    ).cpu().numpy()
    pts = np.zeros((heights.shape[0], 3))
    pts[:, 0] = hp_world[:, 0] + base_pos[0]
    pts[:, 1] = hp_world[:, 1] + base_pos[1]
    pts[:, 2] = heights
    return pts


def _overlay_scandots(frame_bgr, env, cam_handle, width, height, env_idx=0,
                      color=(0, 0, 230), radius=5):
    if not hasattr(env, "measured_heights") or env.measured_heights is None:
        return frame_bgr
    scandot_pts = _get_scandot_world_positions(env, env_idx)
    view_raw = env.gym.get_camera_view_matrix(env.sim, env.envs[env_idx], cam_handle)
    proj_raw = env.gym.get_camera_proj_matrix(env.sim, env.envs[env_idx], cam_handle)
    view = np.array(view_raw).reshape(4, 4).T
    proj = np.array(proj_raw).reshape(4, 4).T
    uv, mask = _project_points_to_pixels(scandot_pts, view, proj, width, height)
    depth_img = env.gym.get_camera_image(env.sim, env.envs[env_idx], cam_handle, gymapi.IMAGE_DEPTH)
    depth_buf = np.array(depth_img).reshape(height, width)
    N = scandot_pts.shape[0]
    homo = np.hstack([scandot_pts, np.ones((N, 1))])
    cam_z = (view @ homo.T).T[:, 2]
    for j in range(N):
        if not mask[j]:
            continue
        pu, pv = int(uv[j, 0]), int(uv[j, 1])
        rendered_depth = depth_buf[pv, pu]
        point_depth = cam_z[j]
        if point_depth < rendered_depth - 0.05:
            continue
        cv2.circle(frame_bgr, (pu, pv), radius, color, -1)
    return frame_bgr


def _get_mask_flag(infos):
    if not isinstance(infos, dict):
        return False
    val = infos.get("depth_blind_mask", None)
    if val is None:
        return False
    if isinstance(val, torch.Tensor):
        return bool(val.reshape(-1)[0].item())
    if isinstance(val, np.ndarray):
        return bool(val.reshape(-1)[0])
    try:
        return bool(val)
    except Exception:
        return False

def _depth_to_colormap(depth_tensor, height: int, width: int):
    if depth_tensor is None:
        return None
    if isinstance(depth_tensor, torch.Tensor):
        depth = depth_tensor.detach().float().cpu().numpy()
    else:
        depth = np.array(depth_tensor, dtype=np.float32)
    if depth.ndim > 2:
        depth = depth[0]
    depth_norm = (depth + 0.5)
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    depth_img = (depth_norm * 255.0).astype(np.uint8)
    depth_img = cv2.resize(depth_img, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(depth_img, cv2.COLOR_GRAY2BGR)

def _record_videos(args):
    faulthandler.enable()
    exptid = str(getattr(args, "exptid", "") or "")
    if not exptid:
        raise ValueError("Please provide --exptid (run folder name).")
    proj_name = str(getattr(args, "proj_name", "parkour_new"))
    log_root = str(getattr(args, "log_root", "") or "")
    if log_root:
        log_pth = os.path.join(log_root, proj_name, exptid)
    else:
        log_pth = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", proj_name, exptid)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    play_cfg = getattr(env_cfg, "play", None)
    if play_cfg is not None:
        try:
            setattr(play_cfg, "debug_viz", False)
        except Exception:
            pass
    if getattr(args, "num_envs", None) is None or int(getattr(args, "num_envs", 1)) != 1:
        setattr(args, "num_envs", 1)
    terrain_names = _parse_terrain_names(args, env_cfg)
    if not terrain_names:
        raise ValueError("No terrain names found for recording.")
    difficulty_levels = _parse_difficulty_levels(args, env_cfg)
    video_dir = _resolve_video_dir(args, exptid, proj_name)
    os.makedirs(video_dir, exist_ok=True)
    videos_per_terrain = int(getattr(args, "videos_per_terrain", 2) or 1)
    width, height = 1280, 720
    train_cfg.runner.resume = True
    for terrain_name in terrain_names:
        safe_name = str(terrain_name).replace(" ", "_").replace("/", "_")
        for level in difficulty_levels:
            cfg = deepcopy(env_cfg)
            _override_env_cfg_for_single_terrain(
                cfg, terrain_name, int(getattr(args, "num_envs", 1)), keep_rows=True
            )
            cfg.terrain.fixed_terrain_level = int(level)
            cfg.depth.use_camera = True
            try:
                setattr(cfg.sim, "enable_camera_sensors", True)
            except Exception:
                pass
            env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=cfg)
            obs = env.get_observations()
            runner_cfg = deepcopy(train_cfg)
            if getattr(args, "scan_encoder_type", None) is not None:
                runner_cfg.policy.scan_encoder_type = str(args.scan_encoder_type)
            if getattr(args, "scan_history_len", None) is not None:
                runner_cfg.policy.scan_history_len = int(args.scan_history_len)
            runner_cfg.runner.resume = False
            model, _ = get_load_path(root=log_pth, checkpoint=runner_cfg.runner.checkpoint)
            resume_path = os.path.join(log_pth, model)
            policy_class_name = _detect_policy_class_name(resume_path)
            if policy_class_name:
                _ensure_policy_class_available(policy_class_name)
                runner_cfg.runner.policy_class_name = policy_class_name
            ppo_runner, _, _ = task_registry.make_alg_runner(
                log_root=log_pth,
                env=env,
                name=args.task,
                args=args,
                train_cfg=runner_cfg,
                return_log_dir=True,
            )
            _load_checkpoint_filtered(ppo_runner, resume_path)
            policy = ppo_runner.get_inference_policy(device=env.device)
            estimator = None
            if hasattr(ppo_runner, "get_estimator_inference_policy"):
                estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
            use_depth_in_policy = _should_use_depth_in_policy(args, env.cfg)
            if use_depth_in_policy and hasattr(ppo_runner, "get_depth_encoder_inference_policy"):
                depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
            cam_props = gymapi.CameraProperties()
            cam_props.width = width
            cam_props.height = height
            cam_handle = env.gym.create_camera_sensor(env.envs[0], cam_props)
            fps = _get_fps(env)
            max_steps = int(getattr(env, "max_episode_length", 1000))
            overlay_depth = bool(getattr(args, "record_depth_overlay", False))
            split_blind = bool(getattr(args, "record_split_blind", False))
            depth_width = int(width * 0.5) if overlay_depth and env.cfg.depth.use_camera else 0
            out_width = width + depth_width
            for vid_idx in range(videos_per_terrain):
                reset_out = env.reset()
                if isinstance(reset_out, tuple):
                    obs = reset_out[0]
                else:
                    obs = reset_out
                infos = {}
                if env.cfg.depth.use_camera:
                    infos["depth"] = env.depth_buffer.clone().to(env.device)[:, -1]
                pre_path = os.path.join(video_dir, f"play-{safe_name}-L{level}-{vid_idx + 1}-preblind.mp4")
                post_path = os.path.join(video_dir, f"play-{safe_name}-L{level}-{vid_idx + 1}-postblind.mp4")
                out_path = os.path.join(video_dir, f"play-{safe_name}-L{level}-{vid_idx + 1}.mp4")
                writer = None
                writer_pre = None
                writer_post = None
                masked_started = False
                if split_blind:
                    writer_pre = _init_video_writer(pre_path, out_width, height, fps)
                else:
                    writer = _init_video_writer(out_path, out_width, height, fps)
                for _ in range(max_steps):
                    root_pos = env.root_states[0, :3].detach().cpu().numpy()
                    cam_pos = root_pos + np.array([0.0, 2.0, 1.0])
                    env.gym.set_camera_location(
                        cam_handle,
                        env.envs[0],
                        gymapi.Vec3(*cam_pos),
                        gymapi.Vec3(*root_pos),
                    )
                    obs_for_policy = obs
                    depth_latent = None
                    if use_depth_in_policy:
                        yaw = torch.zeros((env.cfg.env.num_envs, 2), device=env.device)
                        if infos.get("depth", None) is not None:
                            obs_student = obs[:, : env.cfg.env.n_proprio].clone()
                            obs_student[:, 6:8] = 0
                            depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student, infos.get("applied_action", None))
                            depth_latent = depth_latent_and_yaw[:, :-2]
                            yaw = depth_latent_and_yaw[:, -2:]
                        obs_for_policy = obs.clone()
                        obs_for_policy[:, 6:8] = 1.5 * yaw
                    if getattr(args, "mask_priv_obs", False) and estimator is not None:
                        try:
                            est_in = _get_estimator_input(
                                obs_for_policy,
                                infos,
                                estimator,
                                int(env.cfg.env.n_proprio),
                                int(env.cfg.env.history_len),
                            )
                            if hasattr(estimator, "forward_with_uncertainty") and bool(getattr(ppo_runner.alg, "estimator_uncertainty_enabled", False)):
                                mean_pred, raw = estimator.forward_with_uncertainty(est_in)
                                if str(getattr(ppo_runner.alg, "estimator_uncertainty_param", "log_var")) == "log_std":
                                    log_std = raw
                                else:
                                    log_std = 0.5 * raw
                                log_std = torch.clamp(
                                    log_std,
                                    min=float(getattr(ppo_runner.alg, "estimator_min_log_std", -6.907755278982137)),
                                    max=float(getattr(ppo_runner.alg, "estimator_max_log_std", 2.0)),
                                )
                                mean_pred = _maybe_fuse_estimator_pred(
                                    ppo_runner,
                                    mean_pred,
                                    log_std,
                                    infos,
                                    int(cfg.env.num_envs),
                                    env.device,
                                )
                                priv_hat = mean_pred
                            else:
                                priv_hat = estimator(est_in)
                            priv_latent_mode = str(getattr(env.cfg.env, "priv_latent_mode", "env"))
                            total_priv = int(env.cfg.env.n_priv) + (int(env.cfg.env.n_priv_latent) if priv_latent_mode == "estimator" else 0)
                            s = env.cfg.env.n_proprio + env.cfg.env.n_scan
                            e = s + total_priv
                            obs_for_policy = obs_for_policy.clone()
                            obs_for_policy[:, s:e] = priv_hat
                        except Exception as e:
                            print(f"[play] WARNING: failed to fill priv_explicit via estimator: {e}")
                    if use_depth_in_policy and hasattr(ppo_runner.alg, "depth_actor"):
                        try:
                            actions = ppo_runner.alg.depth_actor(
                                obs_for_policy.detach(),
                                hist_encoding=True,
                                scandots_latent=depth_latent,
                            )
                        except TypeError:
                            actions = ppo_runner.alg.depth_actor(obs_for_policy.detach())
                    else:
                        try:
                            actions = policy(
                                obs_for_policy.detach(),
                                hist_encoding=True,
                                scandots_latent=depth_latent,
                            )
                        except TypeError:
                            actions = policy(obs_for_policy.detach())
                    obs, _, _, _, infos = env.step(actions.detach())
                    if env.cfg.depth.use_camera:
                        infos["depth"] = env.depth_buffer.clone().to(env.device)[:, -1]
                    video_mask_flag = False
                    if bool(getattr(args, "video_depth_blind", False)):
                        try:
                            d = getattr(env, "target_dist", None)
                            if d is not None:
                                if isinstance(d, torch.Tensor):
                                    video_mask_flag = bool((d.reshape(-1)[0] <= float(getattr(args, "video_depth_blind_dist", 1.0))).item())
                                else:
                                    video_mask_flag = bool(d <= float(getattr(args, "video_depth_blind_dist", 1.0)))
                        except Exception:
                            video_mask_flag = False
                    masked_flag = video_mask_flag if bool(getattr(args, "video_depth_blind", False)) else _get_mask_flag(infos)
                    if split_blind and (not masked_started) and masked_flag:
                        masked_started = True
                        if writer_pre is not None:
                            writer_pre.release()
                            writer_pre = None
                        if writer_post is None:
                            writer_post = _init_video_writer(post_path, out_width, height, fps)
                    frame = _capture_frame(env, cam_handle, width, height)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    if bool(getattr(args, "record_scandots", False)):
                        frame_bgr = _overlay_scandots(frame_bgr, env, cam_handle, width, height,
                                                      env_idx=0, color=(0, 0, 230), radius=5)
                    if overlay_depth and env.cfg.depth.use_camera:
                        depth_img = _depth_to_colormap(infos.get("depth", None), height, depth_width)
                        if depth_img is not None:
                            if bool(getattr(args, "video_depth_blind", False)) and video_mask_flag:
                                depth_img = np.zeros_like(depth_img)
                            frame_bgr = np.concatenate([frame_bgr, depth_img], axis=1)
                    phase = "postblind" if masked_started else "preblind"
                    mask_text = "masked" if masked_flag else "clear"
                    label_head = str(getattr(args, "video_label", "") or exptid)
                    label = f"{label_head} {safe_name} L{level} {phase} {mask_text}"
                    cv2.putText(frame_bgr, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    if split_blind:
                        if masked_started and writer_post is not None:
                            writer_post.write(frame_bgr)
                        elif writer_pre is not None:
                            writer_pre.write(frame_bgr)
                    else:
                        if writer is not None:
                            writer.write(frame_bgr)
                if writer is not None:
                    writer.release()
                if writer_pre is not None:
                    writer_pre.release()
                if writer_post is not None:
                    writer_post.release()
            try:
                env.close()
            except Exception:
                pass

def _get_estimator_input(obs, infos, estimator, n_prop, history_len):
    if getattr(estimator, "sequence_input", False):
        hist = None
        if isinstance(infos, dict):
            try:
                hist = infos.get("obs_history", None)
            except Exception:
                hist = None
        if hist is not None:
            return hist
        need = int(history_len * n_prop)
        if history_len > 1 and obs.shape[1] >= need:
            return obs[:, -need:].view(obs.shape[0], history_len, n_prop)
    return obs[:, :n_prop]


def _maybe_fuse_estimator_pred(ppo_runner, mean_pred, log_std, infos, num_envs, device):
    try:
        alg = ppo_runner.alg
    except Exception:
        return mean_pred
    if not bool(getattr(alg, "estimator_fusion_enabled", False)):
        return mean_pred
    if getattr(alg, "fusion_last_v", None) is None or getattr(alg, "fusion_last_P", None) is None:
        try:
            p0 = float(getattr(alg, "estimator_fusion_p0", 1.0))
        except Exception:
            p0 = 1.0
        alg.fusion_last_v = torch.zeros(num_envs, 3, device=device)
        alg.fusion_last_P = torch.eye(3, device=device).unsqueeze(0).repeat(num_envs, 1, 1) * p0
    try:
        fused = alg._apply_kf_fusion(mean_pred, log_std, infos)
        if fused is not None:
            return fused
    except Exception:
        return mean_pred
    return mean_pred


def play(args):
    if args.web:
        web_viewer = webviewer.WebViewer()
    faulthandler.enable()
    if getattr(args, "record_video", False):
        _record_videos(args)
        return
    exptid = args.exptid
    # Use absolute log path so play.py works regardless of current working directory.
    log_pth = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", str(args.proj_name), str(exptid))

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # Optionally mask privileged explicit obs (deployment-like). When enabled, allocate privileged_obs_buf to keep unmasked copy.
    if getattr(args, "mask_priv_obs", False):
        env_cfg.play.mask_priv_obs = True
        env_cfg.env.num_privileged_obs = env_cfg.env.num_observations
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 16 if not args.save else 64
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.curriculum = False

    depth_latent_buffer = []
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    use_depth_in_policy = _should_use_depth_in_policy(args, env_cfg)

    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    model, _ = get_load_path(root=log_pth, checkpoint=train_cfg.runner.checkpoint)
    resume_path = os.path.join(log_pth, model)
    policy_class_name = _detect_policy_class_name(resume_path)
    if policy_class_name:
        _ensure_policy_class_available(policy_class_name)
        train_cfg.runner.policy_class_name = policy_class_name
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)
    if getattr(args, "load_estimator_checkpoint", None) or getattr(args, "load_estimator_run", None):
        try:
            if getattr(args, "load_estimator_checkpoint", None):
                path = args.load_estimator_checkpoint
            else:
                run_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.proj_name, args.load_estimator_run)
                path = get_latest_model_path(run_dir, getattr(args, "load_estimator_run_checkpoint", -1))
            print(f"[Play] Loading estimator checkpoint from {path}...")
            state, _, info = load_estimator_state_dict(path, ppo_runner.device, ppo_runner.alg.estimator.state_dict())
            out = ppo_runner.alg.estimator.load_state_dict(state, strict=False)
            miss = getattr(out, "missing_keys", [])
            unexp = getattr(out, "unexpected_keys", [])
            print(
                f"[Play] Estimator loaded. loaded={info.get('loaded')}, skipped_size_mismatch={info.get('skipped_size_mismatch')}, "
                f"missing_keys={len(miss)}, unexpected_keys={len(unexp)}"
            )
        except Exception as e:
            print(f"[Play] Failed to load estimator: {e}")
            raise e
    
    if args.use_jit:
        path = os.path.join(log_pth, "traced")
        model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
        path = os.path.join(path, model)
        print("Loading jit for policy: ", path)
        policy_jit = torch.jit.load(path, map_location=env.device)
    else:
            policy = ppo_runner.get_inference_policy(device=env.device)
            estimator = None
            if hasattr(ppo_runner, "get_estimator_inference_policy"):
                estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
            if use_depth_in_policy and hasattr(ppo_runner, "get_depth_encoder_inference_policy"):
                depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
    # simulation loop
    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None

    for i in range(10*int(env.max_episode_length)):
        obs_for_policy = obs
        # For JIT path, we don't go through PPO.act(), so if priv_explicit is masked we must fill it manually here.
        if args.use_jit and getattr(args, "mask_priv_obs", False):
            try:
                est_in = _get_estimator_input(
                    obs,
                    infos,
                    estimator,
                    int(env.cfg.env.n_proprio),
                    int(env.cfg.env.history_len),
                )
                if hasattr(estimator, "forward_with_uncertainty") and bool(getattr(ppo_runner.alg, "estimator_uncertainty_enabled", False)):
                    mean_pred, raw = estimator.forward_with_uncertainty(est_in)
                    if str(getattr(ppo_runner.alg, "estimator_uncertainty_param", "log_var")) == "log_std":
                        log_std = raw
                    else:
                        log_std = 0.5 * raw
                    log_std = torch.clamp(
                        log_std,
                        min=float(getattr(ppo_runner.alg, "estimator_min_log_std", -6.907755278982137)),
                        max=float(getattr(ppo_runner.alg, "estimator_max_log_std", 2.0)),
                    )
                    mean_pred = _maybe_fuse_estimator_pred(
                        ppo_runner,
                        mean_pred,
                        log_std,
                        infos,
                        int(env_cfg.env.num_envs),
                        env.device,
                    )
                    priv_hat = mean_pred
                else:
                    priv_hat = estimator(est_in)
                obs_for_policy = obs.clone()
                priv_latent_mode = str(getattr(env.cfg.env, "priv_latent_mode", "env"))
                total_priv = int(env.cfg.env.n_priv) + (int(env.cfg.env.n_priv_latent) if priv_latent_mode == "estimator" else 0)
                s = env.cfg.env.n_proprio + env.cfg.env.n_scan
                e = s + total_priv
                obs_for_policy[:, s:e] = priv_hat
            except Exception as e:
                print(f"[play] WARNING: failed to fill priv_explicit via estimator (jit): {e}")
                obs_for_policy = obs
        if args.use_jit:
            if use_depth_in_policy:
                if infos["depth"] is not None:
                    depth_latent = torch.ones((env_cfg.env.num_envs, 32), device=env.device)
                    actions, depth_latent = policy_jit(obs_for_policy.detach(), True, infos["depth"], depth_latent)
                else:
                    depth_buffer = torch.ones((env_cfg.env.num_envs, 58, 87), device=env.device)
                    actions, depth_latent = policy_jit(obs_for_policy.detach(), False, depth_buffer, depth_latent)
            else:
                obs_jit = torch.cat((obs_for_policy.detach()[:, :env_cfg.env.n_proprio+env_cfg.env.n_priv], obs_for_policy.detach()[:, -env_cfg.env.history_len*env_cfg.env.n_proprio:]), dim=1)
                actions = policy(obs_jit)
        else:
            if use_depth_in_policy:
                # default yaw/depth_latent to avoid unbound vars if depth is missing
                yaw = torch.zeros((env_cfg.env.num_envs, 2), device=env.device)
                depth_latent = None
                if infos["depth"] is not None:
                    obs_student = obs[:, :env.cfg.env.n_proprio].clone()
                    obs_student[:, 6:8] = 0
                    depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student, infos.get("applied_action", None))
                    depth_latent = depth_latent_and_yaw[:, :-2]
                    yaw = depth_latent_and_yaw[:, -2:]
                obs_for_policy = obs.clone()
                obs_for_policy[:, 6:8] = 1.5 * yaw
            else:
                depth_latent = None

            # If priv_explicit is masked in env obs, fill it back using estimator prediction
            # so that play matches the "train_with_estimated_states" deployment-like behavior.
            if getattr(args, "mask_priv_obs", False):
                try:
                    est_in = _get_estimator_input(
                        obs_for_policy,
                        infos,
                        estimator,
                        int(env.cfg.env.n_proprio),
                        int(env.cfg.env.history_len),
                    )
                    if hasattr(estimator, "forward_with_uncertainty") and bool(getattr(ppo_runner.alg, "estimator_uncertainty_enabled", False)):
                        mean_pred, raw = estimator.forward_with_uncertainty(est_in)
                        if str(getattr(ppo_runner.alg, "estimator_uncertainty_param", "log_var")) == "log_std":
                            log_std = raw
                        else:
                            log_std = 0.5 * raw
                        log_std = torch.clamp(
                            log_std,
                            min=float(getattr(ppo_runner.alg, "estimator_min_log_std", -6.907755278982137)),
                            max=float(getattr(ppo_runner.alg, "estimator_max_log_std", 2.0)),
                        )
                        mean_pred = _maybe_fuse_estimator_pred(
                            ppo_runner,
                            mean_pred,
                            log_std,
                            infos,
                            int(env_cfg.env.num_envs),
                            env.device,
                        )
                        priv_hat = mean_pred
                    else:
                        priv_hat = estimator(est_in)
                    priv_latent_mode = str(getattr(env.cfg.env, "priv_latent_mode", "env"))
                    total_priv = int(env.cfg.env.n_priv) + (int(env.cfg.env.n_priv_latent) if priv_latent_mode == "estimator" else 0)
                    s = env.cfg.env.n_proprio + env.cfg.env.n_scan
                    e = s + total_priv
                    obs_for_policy[:, s:e] = priv_hat
                except Exception as e:
                    print(f"[play] WARNING: failed to fill priv_explicit via estimator: {e}")
            
            if use_depth_in_policy and hasattr(ppo_runner.alg, "depth_actor"):
                actions = ppo_runner.alg.depth_actor(obs_for_policy.detach(), hist_encoding=True, scandots_latent=depth_latent)
            else:
                actions = policy(obs_for_policy.detach(), hist_encoding=True, scandots_latent=depth_latent)
        obs, _, rews, dones, infos = env.step(actions.detach())
        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
        print("time:", env.episode_length_buf[env.lookat_id].item() / 50, 
              "cmd vx", env.commands[env.lookat_id, 0].item(),
              "actual vx", env.base_lin_vel[env.lookat_id, 0].item(), )
        
        id = env.lookat_id
        

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
