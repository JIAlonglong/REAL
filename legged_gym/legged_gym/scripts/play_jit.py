# flake8: noqa
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Play a trained policy using exported TorchScript files (base_jit + vision_jit).
#
# Typical usage:
#   source /home/bld/miniconda3/etc/profile.d/conda.sh
#   conda activate real
#   cd /home/bld/liujialong/training/REAL
#   python legged_gym/legged_gym/scripts/play_jit.py \
#       --task go2 --proj_name parkour_new --exptid learn_vision-01092252-posmamba2-film2 \
#       --use_camera --delay --checkpoint -1
#
# Recording with blind depth:
#   python legged_gym/legged_gym/scripts/play_jit.py \
#       --task go2 --proj_name parkour_new --exptid student-001-tcnesm-blind \
#       --checkpoint 4000 --use_camera --delay --record_video --blind_depth \
#       --videos_per_terrain 2 --video_dir /path/to/output

from __future__ import annotations

import os
from typing import Tuple, List

# IMPORTANT: isaacgym must be imported before torch
import isaacgym  # noqa: F401
from isaacgym import gymapi

import cv2
import numpy as np
import torch
import faulthandler

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym.utils import webviewer


def _find_latest_traced_checkpoint(traced_dir: str, jit_prefix: str) -> int:
    if not os.path.isdir(traced_dir):
        raise FileNotFoundError(traced_dir)
    cands: List[int] = []
    for fn in os.listdir(traced_dir):
        if not fn.startswith(jit_prefix + "-"):
            continue
        if not fn.endswith("-base_jit.pt"):
            continue
        parts = fn.split("-")
        if len(parts) < 3:
            continue
        try:
            cands.append(int(parts[-3]))
        except Exception:
            continue
    if not cands:
        raise FileNotFoundError(
            f"No '*-base_jit.pt' found in {traced_dir} for jit_prefix='{jit_prefix}'."
        )
    return int(max(cands))


def _resolve_traced_paths(
    run_dir: str,
    jit_prefix: str,
    checkpoint: int,
) -> Tuple[str, str]:
    traced_dir = os.path.join(run_dir, "traced")
    ckpt = int(checkpoint)
    if ckpt < 0:
        ckpt = _find_latest_traced_checkpoint(traced_dir, jit_prefix=jit_prefix)
    base_jit = os.path.join(traced_dir, f"{jit_prefix}-{ckpt}-base_jit.pt")
    vision_jit = os.path.join(traced_dir, f"{jit_prefix}-{ckpt}-vision_jit.pt")

    if not os.path.isfile(base_jit):
        suffix = f"-{ckpt}-base_jit.pt"
        cands = [fn for fn in os.listdir(traced_dir) if fn.endswith(suffix)]
        if len(cands) == 1:
            base_jit = os.path.join(traced_dir, cands[0])
            inferred_prefix = cands[0][: -len(suffix)]
            vision_jit = os.path.join(traced_dir, f"{inferred_prefix}-{ckpt}-vision_jit.pt")
        elif len(cands) > 1:
            raise FileNotFoundError(
                f"Multiple candidates for '*{suffix}' in {traced_dir}: {cands}. "
                f"Please set --jit_prefix explicitly."
            )

    if not os.path.isfile(base_jit):
        raise FileNotFoundError(base_jit)
    if not os.path.isfile(vision_jit):
        raise FileNotFoundError(vision_jit)
    return base_jit, vision_jit


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


def _run_vision(env, obs, infos, vision_jit_model, vision_device, device, blind_depth):
    """Compute depth_latent and update obs yaw from vision_jit."""
    if env.cfg.depth.use_camera:
        if blind_depth:
            depth_in = (
                torch.zeros_like(infos["depth"]).to(vision_device)
                if infos.get("depth") is not None
                else torch.zeros((env.num_envs, 58, 87), device=vision_device)
            )
        elif infos.get("depth", None) is not None:
            depth_in = infos["depth"].to(vision_device)
        else:
            depth_in = None

        if depth_in is not None:
            proprio = obs[:, : env.cfg.env.n_proprio].clone()
            proprio[:, 6:8] = 0.0
            proprio_in = proprio.to(vision_device)
            out = vision_jit_model(depth_in, proprio_in)
            if isinstance(out, tuple):
                out = out[0]
            out = out.to(device)
            depth_latent = out[:, :-2]
            yaw = out[:, -2:]
            obs[:, 6:8] = 1.5 * yaw
        else:
            depth_latent = torch.zeros((env.num_envs, 32), device=device)
    else:
        depth_latent = torch.zeros((env.num_envs, 32), device=device)
    return depth_latent


@torch.no_grad()
def play(args):
    if getattr(args, "web", False):
        web_viewer = webviewer.WebViewer()
    else:
        web_viewer = None
    faulthandler.enable()

    record_video = bool(getattr(args, "record_video", False))
    blind_depth = bool(getattr(args, "blind_depth", False))

    if getattr(args, "num_envs", None) is None:
        setattr(args, "num_envs", 1 if record_video else 16)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    if record_video:
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected = False
        fixed_level = int(getattr(args, "difficulty_level", 3))
        env_cfg.terrain.fixed_terrain_level = fixed_level
        env_cfg.terrain.num_rows = max(fixed_level + 1, 5)
        env_cfg.terrain.num_cols = 1
        terrain_keys = list(getattr(env_cfg.terrain, "terrain_dict", {}).keys())
        selected_terrain = str(getattr(args, "terrain", "") or "").strip()
        if terrain_keys:
            target = None
            if selected_terrain and selected_terrain in terrain_keys:
                target = selected_terrain
            else:
                for k, v in env_cfg.terrain.terrain_dict.items():
                    if float(v) > 0:
                        target = k
                        break
            if target is None:
                target = terrain_keys[0]
            for k in terrain_keys:
                env_cfg.terrain.terrain_dict[k] = 1.0 if k == target else 0.0
            env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
            print(f"[record] terrain type: {target}")
        args.headless = True
        play_cfg = getattr(env_cfg, "play", None)
        if play_cfg is not None:
            try:
                setattr(play_cfg, "debug_viz", False)
            except Exception:
                pass
    elif not bool(getattr(args, "keep_terrain", False)):
        try:
            env_cfg.terrain.mesh_type = "plane"
            env_cfg.terrain.curriculum = False
            env_cfg.terrain.selected = False
            env_cfg.terrain.num_rows = 1
            env_cfg.terrain.num_cols = 1
            env_cfg.terrain.height = [0.0, 0.0]
            env_cfg.terrain.terrain_dict = {"smooth flat": 1.0}
            env_cfg.terrain.terrain_proportions = [1.0]
            env_cfg.terrain.measure_heights = False
        except Exception:
            pass

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    if web_viewer is not None:
        web_viewer.setup(env)

    exptid = str(getattr(args, "exptid", "") or "")
    if not exptid:
        raise ValueError("Please provide --exptid (run folder name).")
    proj_name = str(getattr(args, "proj_name", "parkour_new"))
    run_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", proj_name, exptid)

    jit_prefix = str(getattr(args, "jit_prefix", "") or exptid)
    base_jit_path, vision_jit_path = _resolve_traced_paths(
        run_dir=run_dir,
        jit_prefix=jit_prefix,
        checkpoint=int(getattr(args, "checkpoint", -1)),
    )

    print(f"[play_jit] run_dir: {os.path.abspath(run_dir)}")
    print(f"[play_jit] base_jit: {os.path.abspath(base_jit_path)}")
    print(f"[play_jit] vision_jit: {os.path.abspath(vision_jit_path)}")
    print(f"[play_jit] use_camera={bool(env.cfg.depth.use_camera)}")
    print(f"[play_jit] blind_depth={blind_depth}, record_video={record_video}")

    device = env.device
    base_jit_model = torch.jit.load(base_jit_path, map_location=device)
    vision_device = torch.device("cpu")
    vision_jit_model = torch.jit.load(vision_jit_path, map_location=vision_device)
    base_jit_model.eval()
    vision_jit_model.eval()

    infos = {"depth": env.depth_buffer.clone().to(device)[:, -1] if env.cfg.depth.use_camera else None}
    actions = torch.zeros(env.num_envs, env.num_actions, device=device)

    # --- Recording state ---
    rec_cam = None
    rec_writer = None
    rec_video_dir = None
    rec_vid_idx = 0
    rec_frame_count = 0
    rec_max_videos = int(getattr(args, "videos_per_terrain", 2) or 2)
    rec_width, rec_height = 960, 540

    if record_video:
        rec_video_dir = str(getattr(args, "video_dir", "") or "")
        if not rec_video_dir:
            rec_video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", proj_name, exptid, "play_videos")
        os.makedirs(rec_video_dir, exist_ok=True)
        cam_props = gymapi.CameraProperties()
        cam_props.width = rec_width
        cam_props.height = rec_height
        rec_cam = env.gym.create_camera_sensor(env.envs[0], cam_props)
        fps = _get_fps(env)
        rec_terrain_tag = str(getattr(args, "terrain", "") or "").strip()
        if not rec_terrain_tag:
            rec_terrain_tag = "mixed"
        blind_tag = "-blind" if blind_depth else ""
        level_tag = f"-L{fixed_level}"
        vid_name = f"{exptid}{blind_tag}-{rec_terrain_tag}{level_tag}-{rec_vid_idx + 1}.mp4"
        out_path = os.path.join(rec_video_dir, vid_name)
        rec_writer = _init_video_writer(out_path, rec_width, rec_height, fps)
        print(f"[record] recording video {rec_vid_idx + 1}/{rec_max_videos}: {out_path}")

    max_ep = int(getattr(env, "max_episode_length", 1000))
    total_steps = max_ep * rec_max_videos if record_video else 10 * max_ep

    for step_i in range(total_steps):
        depth_latent = _run_vision(
            env, obs, infos, vision_jit_model, vision_device, device, blind_depth
        )
        actions = base_jit_model(obs, depth_latent)
        obs, _, _, _, infos = env.step(actions.detach())

        # --- Record frame ---
        if record_video and rec_cam is not None and rec_writer is not None:
            root_pos = env.root_states[0, :3].detach().cpu().numpy()
            cam_pos = root_pos + np.array([0.0, 2.0, 1.0])
            env.gym.set_camera_location(
                rec_cam, env.envs[0],
                gymapi.Vec3(*cam_pos), gymapi.Vec3(*root_pos),
            )
            frame = _capture_frame(env, rec_cam, rec_width, rec_height)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if env.cfg.depth.use_camera:
                if blind_depth:
                    depth_color = np.zeros((58, 87, 3), dtype=np.uint8)
                elif infos.get("depth") is not None:
                    d = infos["depth"][0].cpu().numpy()
                    depth_vis = np.clip((d + 0.5) * 255, 0, 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                else:
                    depth_color = np.zeros((58, 87, 3), dtype=np.uint8)
                oh, ow = int(rec_height * 0.25), int(rec_width * 0.25)
                depth_small = cv2.resize(depth_color, (ow, oh))
                frame_bgr[2:2+oh, 2:2+ow] = depth_small
                cv2.rectangle(frame_bgr, (2, 2), (2+ow, 2+oh), (255, 255, 255), 1)

            rec_writer.write(frame_bgr)
            rec_frame_count += 1

            ep_done = False
            if hasattr(env, "reset_buf"):
                ep_done = bool(env.reset_buf[0].item())
            if ep_done or rec_frame_count >= max_ep:
                rec_writer.release()
                rec_vid_idx += 1
                rec_frame_count = 0
                print(f"[record] video {rec_vid_idx}/{rec_max_videos} saved.")
                if rec_vid_idx >= rec_max_videos:
                    print(f"\n[record] ALL {rec_max_videos} videos saved to: {rec_video_dir}")
                    break
                blind_tag = "-blind" if blind_depth else ""
                level_tag = f"-L{fixed_level}" if record_video else ""
                vid_name = f"{exptid}{blind_tag}-{rec_terrain_tag}{level_tag}-{rec_vid_idx + 1}.mp4"
                out_path = os.path.join(rec_video_dir, vid_name)
                fps = _get_fps(env)
                rec_writer = _init_video_writer(out_path, rec_width, rec_height, fps)
                print(f"[record] recording video {rec_vid_idx + 1}/{rec_max_videos}: {out_path}")

        if web_viewer is not None:
            web_viewer.render(
                fetch_results=True,
                step_graphics=True,
                render_all_camera_sensors=True,
                wait_for_page_load=True,
            )

    if rec_writer is not None:
        try:
            rec_writer.release()
        except Exception:
            pass
        if rec_vid_idx < rec_max_videos:
            rec_vid_idx += 1
            print(f"[record] video {rec_vid_idx}/{rec_max_videos} saved.")
        print(f"\n[record] ALL videos saved to: {rec_video_dir}")


if __name__ == "__main__":
    args = get_args()
    play(args)
