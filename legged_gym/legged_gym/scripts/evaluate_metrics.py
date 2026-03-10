#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# flake8: noqa

"""
批量评测脚本：
- 按“地形类型”逐个评测（每次只生成一种地形：强制 terrain.num_cols=1）
- 每个地形并行 num_robots 个环境，累计跑满 num_trials 个 episode
- 自动统计并输出 Markdown 报告（包含实验名）

指标（与用户表格口径对齐）：
- MXD: episode 终止时 X 方向位移 / terrain_length，裁剪到 [0,1] 后取均值/方差
- MEV: episode 内脚部在 edge 上的接触次数（每 step feet_at_edge.sum 累加），取均值/方差
- 地形通过成功率: 成功 episode 数 / 总 episode 数（以 cur_goal_idx >= num_goals 判定）
- 碰撞率: 碰撞 step 数 / 总 step 数（以 _reward_collision()>0 判定）
"""

from __future__ import annotations

import os
from datetime import datetime
from copy import deepcopy
from typing import Dict, List, Any

import numpy as np

# IMPORTANT: isaacgym must be imported before torch
import isaacgym  # noqa: F401

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry, get_latest_model_path, load_estimator_state_dict
from legged_gym.scripts.sensor_corruption import cfg_from_args, CorruptionState, corrupt_depth, corrupt_scandot


def _maybe_save_scandot_vis(out_dir: str, step_i: int, env_id: int, scan_before: torch.Tensor, scan_after: torch.Tensor):
    """
    保存 scan_dot（132维）退化前/后对比图。使用 matplotlib 的 Agg 后端（无显示环境可用）。
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[deg_vis] matplotlib not available: {e}")
        return

    b = scan_before.detach().float().cpu().numpy()
    a = scan_after.detach().float().cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    axes[0].plot(b, lw=1)
    axes[0].set_title(f"scan_dot BEFORE (env={env_id}, step={step_i})")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(a, lw=1, color="tab:orange")
    axes[1].set_title("scan_dot AFTER")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("scan index")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"scandot_env{env_id:03d}_step{step_i:06d}.png"), dpi=150)
    plt.close(fig)


def _maybe_save_depth_vis(
    out_dir: str,
    step_i: int,
    env_id: int,
    depth_before: torch.Tensor,
    depth_after: torch.Tensor,
    *,
    masked: bool = False,
):
    """
    保存 depth（H,W）退化前/后对比图。depth 预期为归一化到 [-0.5, 0.5] 的张量。
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[deg_vis] matplotlib not available: {e}")
        return

    b = depth_before.detach().float().cpu().numpy()
    a = depth_after.detach().float().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(b, cmap="gray", vmin=-0.5, vmax=0.5)
    axes[0].set_title(f"depth BEFORE (env={env_id}, step={step_i})")
    axes[0].axis("off")
    axes[1].imshow(a, cmap="gray", vmin=-0.5, vmax=0.5)
    axes[1].set_title(f"depth AFTER (masked={masked})")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"depth_env{env_id:03d}_step{step_i:06d}.png"), dpi=150)
    plt.close(fig)


def _parse_terrain_names(env_cfg, args) -> List[str]:
    if getattr(args, "terrains", None):
        names = [x.strip() for x in args.terrains.split(",") if x.strip()]
        return names

    terrain_dict = getattr(env_cfg.terrain, "terrain_dict", {})
    if getattr(args, "include_zero_terrains", False):
        return list(terrain_dict.keys())

    # 默认：只评测比例>0 的地形
    names = []
    for k, v in terrain_dict.items():
        try:
            if float(v) > 0:
                names.append(k)
        except Exception:
            continue
    return names


def _override_env_cfg_for_single_terrain(env_cfg, terrain_name: str, num_envs: int):
    # 并行机器人数量（= 并行环境数）
    env_cfg.env.num_envs = int(num_envs)

    # 强制单一地形类型：否则 terrain_types 会把并行 env 强行均分到多个“列”
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = False
    env_cfg.terrain.num_cols = 1

    # 只保留该地形
    env_cfg.terrain.terrain_dict = {terrain_name: 1.0}
    env_cfg.terrain.terrain_proportions = [1.0]


def _override_env_cfg_for_multi_terrain(env_cfg, terrain_names_in_cfg_order: List[str], num_robots: int):
    """
    单次创建环境同时评测多个地形（避免同进程多次 create_sim 导致 PhysX Foundation 崩溃）：
    - num_cols = 地形数
    - num_envs = num_cols * num_robots（保证每一列恰好 num_robots 个并行环境）
    - 通过 terrain_proportions 将每列 deterministically 映射到一个地形（Terrain.curiculum(random=True) 用 choice=j/num_cols+0.001）
    """
    num_cols = int(len(terrain_names_in_cfg_order))
    env_cfg.env.num_envs = int(num_cols * num_robots)

    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = False
    env_cfg.terrain.num_cols = int(num_cols)

    # 保持 dict 的原始 key 顺序（对应 Terrain.make_terrain 的比例槽位顺序），只给选中的地形分配均匀概率，其余置 0
    base_keys = list(getattr(env_cfg.terrain, "terrain_dict", {}).keys())
    if not base_keys:
        raise ValueError("env_cfg.terrain.terrain_dict is empty; cannot build terrain proportions.")

    per = 1.0 / float(num_cols)
    new_dict = {}
    for k in base_keys:
        new_dict[k] = per if k in terrain_names_in_cfg_order else 0.0
    env_cfg.terrain.terrain_dict = new_dict
    env_cfg.terrain.terrain_proportions = list(new_dict.values())


@torch.no_grad()
def _infer_actions(
    ppo_runner,
    env,
    obs,
    infos,
    depth_latent_state: Dict[str, Any],
    deg_cfg=None,
    deg_state=None,
    mask_priv_obs: bool = False,
):
    """
    兼容 depth camera / depth_actor 的推理分支。
    返回 actions, updated_depth_latent_state
    """
    # 退化：scan_dot（本项目里 scan_dot 对应 n_scan 高度扫描段，位于 obs[n_proprio : n_proprio+n_scan]）
    if (
        deg_cfg is not None
        and deg_state is not None
        and bool(getattr(deg_cfg, "enable", False))
        and ("scandot" in deg_state)
        and getattr(env.cfg.terrain, "measure_heights", False)
    ):
        n_prop = int(env.cfg.env.n_proprio)
        n_scan = int(env.cfg.env.n_scan)
        if obs.shape[1] >= n_prop + n_scan and n_scan > 0:
            scan_before = obs[:, n_prop:n_prop+n_scan].clone()
            scan_after = corrupt_scandot(scan_before, deg_cfg, deg_state["scandot"])
            obs[:, n_prop:n_prop+n_scan] = scan_after

            # optional visualization
            vis = deg_state.get("vis", None)
            if vis is not None and vis["enabled"]:
                vis["step_i"] += 1
                if (vis["step_i"] % vis["every"] == 0) and (vis["saved"] < vis["max_frames"]):
                    eid = int(vis["env_id"])
                    if 0 <= eid < env.num_envs:
                        _maybe_save_scandot_vis(vis["out_dir"], vis["step_i"], eid, scan_before[eid], scan_after[eid])
                        vis["saved"] += 1

    # depth: 维持与 play/evaluate 的用法一致
    if env.cfg.depth.use_camera:
        if depth_latent_state.get("depth_encoder") is None:
            depth_latent_state["depth_encoder"] = ppo_runner.get_depth_encoder_inference_policy(device=env.device)

        if infos.get("depth", None) is not None:
            # save before/after for visualization
            try:
                depth_before = infos["depth"].clone()
            except Exception:
                depth_before = infos["depth"]

            if deg_cfg is not None and deg_state is not None and "depth" in deg_state:
                infos["depth"] = corrupt_depth(infos["depth"], deg_cfg, deg_state["depth"])
            depth_after = infos["depth"]

            # optional depth visualization (also useful to visualize env-side blind-zone masking)
            try:
                vis_d = deg_state.get("vis_depth", None) if deg_state is not None else None
                if vis_d is not None and vis_d["enabled"]:
                    vis_d["step_i"] += 1
                    if (vis_d["step_i"] % vis_d["every"] == 0) and (vis_d["saved"] < vis_d["max_frames"]):
                        eid = int(vis_d["env_id"])
                        if 0 <= eid < env.num_envs:
                            masked = False
                            try:
                                m = infos.get("depth_blind_mask", None)
                                if isinstance(m, torch.Tensor):
                                    masked = bool(m[eid].item())
                            except Exception:
                                masked = False
                            _maybe_save_depth_vis(
                                vis_d["out_dir"],
                                vis_d["step_i"],
                                eid,
                                depth_before[eid],
                                depth_after[eid],
                                masked=masked,
                            )
                            vis_d["saved"] += 1
            except Exception:
                pass
            obs_student = obs[:, :env.cfg.env.n_proprio].clone()
            obs_student[:, 6:8] = 0
            depth_latent_and_yaw = depth_latent_state["depth_encoder"](infos["depth"], obs_student)
            depth_latent = depth_latent_and_yaw[:, :-2]
            yaw = depth_latent_and_yaw[:, -2:]
            obs[:, 6:8] = 1.5 * yaw
        else:
            depth_latent = None
    else:
        depth_latent = None

    # If priv_explicit is masked in env obs, fill it back using estimator prediction
    # so that evaluation matches the "train_with_estimated_states" deployment-like behavior.
    obs_for_policy = obs
    if mask_priv_obs:
        try:
            if depth_latent_state.get("estimator") is None:
                depth_latent_state["estimator"] = ppo_runner.get_estimator_inference_policy(device=env.device)
            est = depth_latent_state["estimator"]
            n_prop = int(env.cfg.env.n_proprio)
            n_scan = int(env.cfg.env.n_scan)
            priv_latent_mode = str(getattr(env.cfg.env, "priv_latent_mode", "env"))
            n_priv = int(env.cfg.env.n_priv) + (int(env.cfg.env.n_priv_latent) if priv_latent_mode == "estimator" else 0)
            priv_hat = est(obs[:, :n_prop])
            obs_for_policy = obs.clone()
            s = n_prop + n_scan
            e = s + n_priv
            if obs_for_policy.shape[1] >= e:
                obs_for_policy[:, s:e] = priv_hat
        except Exception as e:
            print(f"[evaluate_metrics] WARNING: failed to fill priv_explicit via estimator: {e}")
            obs_for_policy = obs

    if hasattr(ppo_runner.alg, "depth_actor"):
        actions = ppo_runner.alg.depth_actor(
            obs_for_policy.detach(),
            hist_encoding=True,
            scandots_latent=depth_latent,
            depth_latent=depth_latent,
        )
    else:
        policy = depth_latent_state["policy"]
        actions = policy(
            obs_for_policy.detach(),
            hist_encoding=True,
            scandots_latent=depth_latent,
            depth_latent=depth_latent,
        )

    return actions, depth_latent_state


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _write_md_report(path: str, exp_name: str, meta: Dict[str, Any], rows: List[Dict[str, Any]]):
    lines = []
    lines.append(f"# 评测报告：{exp_name}")
    lines.append("")
    lines.append(f"- 时间：{meta['time']}")
    lines.append(f"- task：`{meta['task']}`")
    lines.append(f"- proj_name：`{meta['proj_name']}`")
    lines.append(f"- resumeid：`{meta['resumeid']}`")
    lines.append(f"- checkpoint：`{meta['checkpoint']}`")
    lines.append(f"- num_trials（每地形 episode 数）：`{meta['num_trials']}`")
    lines.append(f"- num_robots（并行 env 数）：`{meta['num_robots']}`")
    if meta.get("deg_enable", False):
        lines.append("")
        lines.append("## 感知退化配置")
        lines.append("")
        for k in [
            "deg_target", "deg_seed",
            "deg_p_drop", "deg_gauss_std",
            "deg_occ_p", "deg_occ_size", "deg_occ_len",
            "deg_outage_p", "deg_outage_len",
            "deg_delay_steps",
        ]:
            if k in meta:
                lines.append(f"- {k}：`{meta[k]}`")
    lines.append("")

    # 分组统计：Seen（训练见过） vs Unseen（未见过）
    # 假设：非 "parkour" 开头的地形为 Seen，"parkour" 开头的为 Unseen (或根据实际 terrain_dict 调整)
    seen_metrics = {"mxd": [], "mev": [], "success": [], "timeout": [], "collision": []}
    unseen_metrics = {"mxd": [], "mev": [], "success": [], "timeout": [], "collision": []}

    lines.append("")
    lines.append("## 分地形结果")
    lines.append("")
    has_drop = any(("success_drop" in r) for r in rows)
    if has_drop:
        lines.append("| 地形 | 类型 | MXD(均值±std) | MEV(均值±std) | 通过成功率(到达goal) | 超时失败率(time_out且未到goal) | 碰撞率(step) | ΔSuccess(退化-干净) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    else:
        lines.append("| 地形 | 类型 | MXD(均值±std) | MEV(均值±std) | 通过成功率(到达goal) | 超时失败率(time_out且未到goal) | 碰撞率(step) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
    
    for r in rows:
        # 简单规则区分：名字里带 'parkour' 算 Unseen (Demo)，否则算 Seen (Training)
        # 根据实际情况，这里可以改得更灵活，比如 args 传入 unseen_terrains 列表
        is_unseen = "parkour" in r['terrain'] or "demo" in r['terrain']
        terrain_type = "Unseen" if is_unseen else "Seen"
        
        target_dict = unseen_metrics if is_unseen else seen_metrics
        target_dict["mxd"].append(r['mxd_mean'])
        target_dict["mev"].append(r['mev_mean'])
        target_dict["success"].append(r['success_rate'])
        target_dict["timeout"].append(r['timeout_fail_rate'])
        target_dict["collision"].append(r['collision_rate'])

        if has_drop:
            d = float(r.get("success_drop", 0.0))
            lines.append(
                f"| `{r['terrain']}` | {terrain_type} | {r['mxd_mean']:.4f} ± {r['mxd_std']:.4f} | "
                f"{r['mev_mean']:.4f} ± {r['mev_std']:.4f} | {r['success_rate']:.4f} | {r['timeout_fail_rate']:.4f} | {r['collision_rate']:.4f} | {d:.4f} |"
            )
        else:
            lines.append(
                f"| `{r['terrain']}` | {terrain_type} | {r['mxd_mean']:.4f} ± {r['mxd_std']:.4f} | "
                f"{r['mev_mean']:.4f} ± {r['mev_std']:.4f} | {r['success_rate']:.4f} | {r['timeout_fail_rate']:.4f} | {r['collision_rate']:.4f} |"
            )

    # overall（按 episode 等权；每个地形同 num_trials）
    if rows:
        lines.append("")
        lines.append("## 总体汇总")
        
        def _agg_stats(metrics_dict, label):
            if not metrics_dict["mxd"]:
                return
            mxd = np.mean(metrics_dict["mxd"])
            mev = np.mean(metrics_dict["mev"])
            suc = np.mean(metrics_dict["success"])
            to = np.mean(metrics_dict["timeout"])
            col = np.mean(metrics_dict["collision"])
            drop = None
            if has_drop and ("success_drop" in metrics_dict):
                try:
                    drop = float(np.mean(metrics_dict["success_drop"]))
                except Exception:
                    drop = None
            
            lines.append(f"### {label} Terrains (Avg)")
            lines.append(f"- MXD：{mxd:.4f}")
            lines.append(f"- MEV：{mev:.4f}")
            lines.append(f"- 地形通过成功率：{suc:.4f}")
            lines.append(f"- 超时失败率：{to:.4f}")
            lines.append(f"- 碰撞率：{col:.4f}")
            if drop is not None:
                lines.append(f"- ΔSuccess(退化-干净)：{drop:.4f}")
            lines.append("")

        _agg_stats(seen_metrics, "Seen (Training)")
        _agg_stats(unseen_metrics, "Unseen (Generalization)")

        # Total
        mxd_mean = float(np.mean([r["mxd_mean"] for r in rows]))
        mev_mean = float(np.mean([r["mev_mean"] for r in rows]))
        success_mean = float(np.mean([r["success_rate"] for r in rows]))
        timeout_fail_mean = float(np.mean([r["timeout_fail_rate"] for r in rows]))
        collision_mean = float(np.mean([r["collision_rate"] for r in rows]))
        success_drop_mean = None
        if has_drop:
            try:
                success_drop_mean = float(np.mean([float(r.get("success_drop", 0.0)) for r in rows]))
            except Exception:
                success_drop_mean = None
        lines.append("")
        lines.append("## 总体汇总（各地形等权平均）")
        lines.append("")
        lines.append(f"- MXD：{mxd_mean:.4f}")
        lines.append(f"- MEV：{mev_mean:.4f}")
        lines.append(f"- 地形通过成功率（到达goal）：{success_mean:.4f}")
        lines.append(f"- 超时失败率（time_out且未到goal）：{timeout_fail_mean:.4f}")
        lines.append(f"- 碰撞率(step)：{collision_mean:.4f}")
        if success_drop_mean is not None:
            lines.append(f"- ΔSuccess=Success_deg−Success_clean：{success_drop_mean:.4f}")

    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def evaluate(args):
    # 实验名与输出路径
    exp_name = args.exp_name or args.exptid or "eval"
    if getattr(args, "deg_enable", False):
        suffix_parts = []
        suffix_parts.append(getattr(args, "deg_target", "both"))
        delay = int(getattr(args, "deg_delay_steps", 0))
        if delay > 0:
            suffix_parts.append(f"delay{delay}")
        
        gauss = float(getattr(args, "deg_gauss_std", 0.0))
        if gauss > 0:
            suffix_parts.append(f"noise{gauss:.2f}".replace('.', ''))
            
        p_drop = float(getattr(args, "deg_p_drop", 0.0))
        if p_drop > 0:
            suffix_parts.append(f"drop{p_drop:.2f}".replace('.', ''))
            
        occ_p = float(getattr(args, "deg_occ_p", 0.0))
        if occ_p > 0:
            suffix_parts.append(f"occ{occ_p:.2f}".replace('.', ''))

        outage_p = float(getattr(args, "deg_outage_p", 0.0))
        if outage_p > 0:
            suffix_parts.append(f"outage{outage_p:.2f}".replace('.', ''))
            
        deg_suffix = "_" + "_".join(suffix_parts)
    else:
        deg_suffix = ""
    if args.output_path:
        out_path = args.output_path
        if os.path.isdir(out_path) or out_path.endswith(os.sep):
            out_path = os.path.join(out_path, f"{exp_name}{deg_suffix}.md")
    else:
        # 自动生成包含关键退化参数的文件名，避免手动重命名
        out_path = os.path.join(LEGGED_GYM_ROOT_DIR, "eval_reports", f"{exp_name}{deg_suffix}.md")

    # 兼容：没填 resumeid 就用 exptid
    if not getattr(args, "resumeid", None):
        args.resumeid = args.exptid

    if not args.resumeid:
        raise ValueError("请提供 --resumeid 或 --exptid 用于定位日志目录(模型加载)。")

    if not getattr(args, "use_camera", False):
        args.use_camera = True

    # base cfg：读取训练时的 env / train 配置
    base_env_cfg, base_train_cfg = task_registry.get_cfgs(name=args.task)

    terrain_names = _parse_terrain_names(base_env_cfg, args)
    if not terrain_names:
        raise ValueError("未解析到要评测的地形列表；请设置 --terrains 或检查 terrain_dict/terrain_proportions。")

    # IMPORTANT: 避免同一进程多次创建 IsaacGym sim（会触发 PhysX Foundation 崩溃）
    # 这里一次性创建包含多列的 Terrain，每列对应一个地形；每列并行 num_robots 个 env。
    cfg_key_order = list(getattr(base_env_cfg.terrain, "terrain_dict", {}).keys())
    terrain_names_in_cfg_order = [k for k in cfg_key_order if k in set(terrain_names)]
    if len(terrain_names_in_cfg_order) != len(terrain_names):
        missing = sorted(set(terrain_names) - set(terrain_names_in_cfg_order))
        raise ValueError(f"以下 terrains 不在 terrain_dict 里：{missing}")

    env_cfg = deepcopy(base_env_cfg)
    train_cfg = deepcopy(base_train_cfg)
    # Optional: mask privileged explicit obs (deployment-like).
    # When enabled, allocate privileged_obs_buf to keep an unmasked copy available if needed.
    if getattr(args, "mask_priv_obs", False):
        env_cfg.play.mask_priv_obs = True
        env_cfg.env.num_privileged_obs = env_cfg.env.num_observations
    # 可选覆盖评测时的 episode 长度（例如希望比训练更长）：环境变量 EVAL_EPISODE_LENGTH_S
    try:
        ep_len_env = os.getenv("EVAL_EPISODE_LENGTH_S", None)
        if ep_len_env:
            env_cfg.env.episode_length_s = float(ep_len_env)
    except Exception:
        pass
    # 一次性创建多列地形，避免多次 create_sim 触发 PhysX 崩溃
    _override_env_cfg_for_multi_terrain(env_cfg, terrain_names_in_cfg_order, int(args.num_robots))

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # 感知退化状态（在编码器之前对 depth/scan_dot 注入噪声/丢失/遮挡/延时）
    deg_cfg = cfg_from_args(args)
    deg_state = {}
    if deg_cfg.enable:
        deg_state = {
            "depth": CorruptionState(env.num_envs, env.device, delay_steps=deg_cfg.delay_steps, seed=deg_cfg.seed + 101),
            "scandot": CorruptionState(env.num_envs, env.device, delay_steps=deg_cfg.delay_steps, seed=deg_cfg.seed + 202),
        }

    # 可视化（可在 deg_enable=False 时也开启，用于观察 blind-zone 是否生效）
    exp_name = args.exp_name or args.exptid or "eval"
    if getattr(args, "deg_vis_scandot", False):
        vis_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "eval_reports", exp_name, "scandot_vis")
        deg_state["vis"] = {
            "enabled": True,
            "out_dir": vis_dir,
            "every": max(1, int(getattr(args, "deg_vis_every", 50))),
            "env_id": int(getattr(args, "deg_vis_env", 0)),
            "max_frames": int(getattr(args, "deg_vis_max_frames", 50)),
            "saved": 0,
            "step_i": 0,
        }
    if getattr(args, "deg_vis_depth", False):
        vis_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "eval_reports", exp_name, "depth_vis")
        deg_state["vis_depth"] = {
            "enabled": True,
            "out_dir": vis_dir,
            "every": max(1, int(getattr(args, "deg_vis_every", 50))),
            "env_id": int(getattr(args, "deg_vis_env", 0)),
            "max_frames": int(getattr(args, "deg_vis_max_frames", 50)),
            "saved": 0,
            "step_i": 0,
        }

    train_cfg.runner.resume = True
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.proj_name, args.resumeid)
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(
        log_root=log_root, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True
    )
    if getattr(args, "load_estimator_checkpoint", None) or getattr(args, "load_estimator_run", None):
        try:
            if getattr(args, "load_estimator_checkpoint", None):
                path = args.load_estimator_checkpoint
            else:
                run_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.proj_name, args.load_estimator_run)
                path = get_latest_model_path(run_dir, getattr(args, "load_estimator_run_checkpoint", -1))
            print(f"[Eval] Loading estimator checkpoint from {path}...")
            state, _, info = load_estimator_state_dict(path, ppo_runner.device, ppo_runner.alg.estimator.state_dict())
            out = ppo_runner.alg.estimator.load_state_dict(state, strict=False)
            miss = getattr(out, "missing_keys", [])
            unexp = getattr(out, "unexpected_keys", [])
            print(
                f"[Eval] Estimator loaded. loaded={info.get('loaded')}, skipped_size_mismatch={info.get('skipped_size_mismatch')}, "
                f"missing_keys={len(miss)}, unexpected_keys={len(unexp)}"
            )
        except Exception as e:
            print(f"[Eval] Failed to load estimator: {e}")
            raise e
    if hasattr(ppo_runner, "alg"):
        try:
            ppo_runner.alg.estimator_uncertainty_enabled = False
            ppo_runner.alg.estimator_privilege_robustness = "none"
        except Exception:
            pass
    policy = ppo_runner.get_inference_policy(device=env.device)
    def _rollout_once(_deg_cfg, _deg_state):
        # reset env state between clean and deg runs to reduce coupling
        try:
            env.reset()
        except Exception:
            pass
        obs = env.get_observations()
        infos: Dict[str, Any] = {
            "depth": env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if getattr(ppo_runner, "if_depth", False) else None
        }
        depth_latent_state: Dict[str, Any] = {"policy": policy, "depth_encoder": None}

        start_pos = env.root_states[:, :3].clone()
        edge_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)

        num_cols = len(terrain_names_in_cfg_order)
        if hasattr(env, "terrain_types"):
            num_cols = int(env.terrain_types.max().item()) + 1
        per_col_mxd: List[List[float]] = [[] for _ in range(num_cols)]
        per_col_mev: List[List[float]] = [[] for _ in range(num_cols)]
        per_col_success: List[List[bool]] = [[] for _ in range(num_cols)]
        per_col_timeout_fail: List[List[bool]] = [[] for _ in range(num_cols)]

        collision_steps = [0.0 for _ in range(num_cols)]
        total_steps = [0.0 for _ in range(num_cols)]

        done_cols = [False for _ in range(num_cols)]
        target = int(args.num_trials)

        while not all(done_cols):
            actions, depth_latent_state = _infer_actions(
                ppo_runner,
                env,
                obs,
                infos,
                depth_latent_state,
                deg_cfg=_deg_cfg,
                deg_state=_deg_state,
                mask_priv_obs=bool(getattr(args, "mask_priv_obs", False)),
            )
            obs, _, _, _, infos = env.step(actions.detach())

            if hasattr(env, "_reward_feet_edge"):
                env._reward_feet_edge()
                edge_counter += env.feet_at_edge.sum(dim=1).to(torch.float)

            if hasattr(env, "_reward_collision"):
                col_mask = env._reward_collision() > 0
            else:
                col_mask = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)

            col_ids = env.terrain_types.to(torch.long) if hasattr(env, "terrain_types") else torch.zeros((env.num_envs,), device=env.device, dtype=torch.long)
            for c in range(num_cols):
                env_mask = (col_ids == c)
                total_steps[c] += float(env_mask.sum().item())
                collision_steps[c] += float((col_mask & env_mask).sum().item())

            terminal = infos.get("terminal", None)
            if terminal is None:
                continue

            env_ids = terminal["env_ids"]
            term_pos = terminal["root_pos"]
            term_goal_idx = terminal["cur_goal_idx"]
            term_time_out = terminal.get("time_out", None)

            for k in range(len(env_ids)):
                eid = int(env_ids[k].item())
                c = int(col_ids[eid].item())
                if c >= len(done_cols) or done_cols[c]:
                    start_pos[eid] = env.root_states[eid, :3].clone()
                    edge_counter[eid] = 0.0
                    continue

                dx = float((term_pos[k, 0] - start_pos[eid, 0]).item())
                denom = float(getattr(env.cfg.terrain, "terrain_length", 1.0))
                mxd = 0.0 if denom <= 0 else float(np.clip(dx / denom, 0.0, 1.0))
                per_col_mxd[c].append(mxd)
                per_col_mev[c].append(float(edge_counter[eid].item()))

                success = bool((term_goal_idx[k] >= env.cfg.terrain.num_goals).item())
                per_col_success[c].append(success)
                if term_time_out is not None:
                    is_to = bool(term_time_out[k].item())
                    per_col_timeout_fail[c].append(bool(is_to and (not success)))
                else:
                    per_col_timeout_fail[c].append(False)

                start_pos[eid] = env.root_states[eid, :3].clone()
                edge_counter[eid] = 0.0

                if len(per_col_mxd[c]) >= target:
                    done_cols[c] = True

        _results = []
        for c, terrain_name in enumerate(terrain_names_in_cfg_order):
            mxd_arr = np.array(per_col_mxd[c][:target], dtype=np.float32)
            mev_arr = np.array(per_col_mev[c][:target], dtype=np.float32)
            success_rate = float(np.mean(np.array(per_col_success[c][:target], dtype=np.float32)))
            timeout_fail_rate = float(np.mean(np.array(per_col_timeout_fail[c][:target], dtype=np.float32)))
            collision_rate = float(collision_steps[c] / max(total_steps[c], 1.0))
            _results.append(
                dict(
                    terrain=terrain_name,
                    mxd_mean=float(np.mean(mxd_arr)),
                    mxd_std=float(np.std(mxd_arr)),
                    mev_mean=float(np.mean(mev_arr)),
                    mev_std=float(np.std(mev_arr)),
                    success_rate=success_rate,
                    timeout_fail_rate=timeout_fail_rate,
                    collision_rate=collision_rate,
                )
            )
        return _results

    # If degradation is enabled, also run a clean baseline in the same script run and report ΔSuccess.
    clean_results = None
    if bool(getattr(deg_cfg, "enable", False)):
        clean_cfg = deepcopy(deg_cfg)
        clean_cfg.enable = False
        clean_results = _rollout_once(clean_cfg, {})
    results = _rollout_once(deg_cfg, deg_state)
    if clean_results is not None:
        clean_map = {r["terrain"]: float(r.get("success_rate", 0.0)) for r in clean_results}
        for r in results:
            t = r.get("terrain")
            if t in clean_map:
                r["success_drop"] = float(r.get("success_rate", 0.0)) - float(clean_map[t])

    meta = dict(
        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        task=args.task,
        proj_name=args.proj_name,
        resumeid=args.resumeid,
        checkpoint=getattr(args, "checkpoint", -1),
        num_trials=int(args.num_trials),
        num_robots=int(args.num_robots),
        deg_enable=bool(getattr(args, "deg_enable", False)),
        deg_target=getattr(args, "deg_target", "both"),
        deg_seed=int(getattr(args, "deg_seed", 0)),
        deg_p_drop=float(getattr(args, "deg_p_drop", 0.0)),
        deg_gauss_std=float(getattr(args, "deg_gauss_std", 0.0)),
        deg_occ_p=float(getattr(args, "deg_occ_p", 0.0)),
        deg_occ_size=float(getattr(args, "deg_occ_size", 0.3)),
        deg_occ_len=int(getattr(args, "deg_occ_len", 1)),
        deg_outage_p=float(getattr(args, "deg_outage_p", 0.0)),
        deg_outage_len=int(getattr(args, "deg_outage_len", 0)),
        deg_delay_steps=int(getattr(args, "deg_delay_steps", 0)),
    )
    _write_md_report(out_path, exp_name, meta, results)
    print(f"[evaluate_metrics] report saved to: {out_path}")

    # 显式清理 IsaacGym 资源，避免退出时偶发的 segfault
    try:
        if env is not None and hasattr(env, "gym") and hasattr(env, "sim"):
            env.gym.destroy_sim(env.sim)
    except Exception:
        pass


if __name__ == "__main__":
    args = get_args()
    evaluate(args)
