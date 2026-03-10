#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
模块用途（中文说明）：
- 在评测/演示阶段，为传感器数据（scan_dot / depth）注入“感知退化”（丢包、噪声、遮挡、固定延时等），
  用于测试策略在感知异常情况下的鲁棒性。
- 该退化是“在编码器之前”进行的，不影响训练过程；只在 evaluate/play 推理时生效。
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class DegradationConfig:
    enable: bool = False
    target: str = "both"  # depth|scandot|both
    seed: int = 0

    # Single-step drop / Gaussian noise
    p_drop: float = 0.0
    gauss_std: float = 0.0

    # “块状遮挡”参数（scandot 为连续维度区间；depth 为图像矩形区域）
    occ_p: float = 0.0
    occ_size: float = 0.3
    occ_len: int = 1

    # Prolonged outage (output zeros for L consecutive steps once triggered)
    outage_p: float = 0.0
    outage_len: int = 0

    # Fixed delay (output data from step t-k)
    delay_steps: int = 0


class CorruptionState:
    """Per-env persistent state for occlusion duration, outage, and delay buffer."""
    def __init__(self, num_envs: int, device: str | torch.device, delay_steps: int = 0, seed: int = 0):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(seed))

        # Remaining duration counters for outage / occlusion
        self.outage_remaining = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.occ_remaining = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)

        # Per-env occlusion parameters (start, width, etc.)
        self.occ_params: Optional[Tuple[torch.Tensor, ...]] = None

        # Ring buffer for fixed delay (lazily allocated)
        self.delay_steps = int(delay_steps)
        self._delay_buf: Optional[torch.Tensor] = None  # shape: (K, num_envs, ...)
        self._delay_idx = 0

    def reset_delay(self):
        self._delay_buf = None
        self._delay_idx = 0


def cfg_from_args(args) -> DegradationConfig:
    # Parse degradation config from CLI args (used by evaluate_metrics.py)
    return DegradationConfig(
        enable=bool(getattr(args, "deg_enable", False)),
        target=str(getattr(args, "deg_target", "both")),
        seed=int(getattr(args, "deg_seed", 0)),
        p_drop=float(getattr(args, "deg_p_drop", 0.0)),
        gauss_std=float(getattr(args, "deg_gauss_std", 0.0)),
        occ_p=float(getattr(args, "deg_occ_p", 0.0)),
        occ_size=float(getattr(args, "deg_occ_size", 0.3)),
        occ_len=int(getattr(args, "deg_occ_len", 1)),
        outage_p=float(getattr(args, "deg_outage_p", 0.0)),
        outage_len=int(getattr(args, "deg_outage_len", 0)),
        delay_steps=int(getattr(args, "deg_delay_steps", 0)),
    )


def _apply_delay(x: torch.Tensor, state: CorruptionState) -> torch.Tensor:
    # Fixed delay: return the value written k steps ago via ring buffer
    k = int(state.delay_steps)
    if k <= 0:
        return x
    if state._delay_buf is None:
        # allocate (K+1) so we can always read k steps behind after write
        buf_shape = (k + 1, ) + tuple(x.shape)
        state._delay_buf = torch.zeros(buf_shape, device=x.device, dtype=x.dtype)
        state._delay_idx = 0

    buf = state._delay_buf
    idx = state._delay_idx
    buf[idx].copy_(x)
    out_idx = (idx - k) % buf.shape[0]
    state._delay_idx = (idx + 1) % buf.shape[0]
    return buf[out_idx]


def _maybe_trigger_counter(p: float, state: CorruptionState) -> torch.Tensor:
    # Trigger an event with probability p, returning a boolean mask per env
    if p <= 0:
        return torch.zeros(state.num_envs, device=state.device, dtype=torch.bool)
    return torch.rand((state.num_envs,), device=state.device, generator=state.gen) < p


def _apply_gauss(x: torch.Tensor, std: float, state: CorruptionState) -> torch.Tensor:
    if std <= 0:
        return x
    noise = torch.randn_like(x) * std
    return x + noise


def _apply_drop_outage(x: torch.Tensor, cfg: DegradationConfig, state: CorruptionState) -> torch.Tensor:
    # outage (multi-step)
    if cfg.outage_len > 0 and cfg.outage_p > 0:
        trigger = _maybe_trigger_counter(cfg.outage_p, state)
        state.outage_remaining = torch.maximum(
            state.outage_remaining,
            trigger.to(torch.int32) * int(cfg.outage_len)
        )
    active_outage = state.outage_remaining > 0
    state.outage_remaining = torch.clamp(state.outage_remaining - 1, min=0)

    # drop (single-step)
    drop = _maybe_trigger_counter(cfg.p_drop, state) if cfg.p_drop > 0 else torch.zeros_like(active_outage)

    mask = active_outage | drop
    if mask.any():
        x = x.clone()
        x[mask] = 0
    return x


def _apply_block_occlusion_scandot(x: torch.Tensor, cfg: DegradationConfig, state: CorruptionState) -> torch.Tensor:
    """Block-occlude a contiguous segment along the last dim of scandot (num_envs, D)."""
    if cfg.occ_p <= 0 or cfg.occ_len <= 0 or cfg.occ_size <= 0:
        return x

    # update remaining + sample params if needed
    trigger = _maybe_trigger_counter(cfg.occ_p, state)
    new_occ = trigger & (state.occ_remaining <= 0)
    if new_occ.any():
        D = x.shape[-1]
        width = max(1, int(round(float(cfg.occ_size) * D)))
        start = torch.randint(0, max(1, D - width + 1), (state.num_envs,), device=state.device, generator=state.gen)
        state.occ_params = (start, torch.full_like(start, width))
        state.occ_remaining = torch.maximum(state.occ_remaining, new_occ.to(torch.int32) * int(cfg.occ_len))

    active = state.occ_remaining > 0
    state.occ_remaining = torch.clamp(state.occ_remaining - 1, min=0)

    if not active.any() or state.occ_params is None:
        return x

    start, width = state.occ_params
    x = x.clone()
    # broadcast mask along other dims except env and last dim
    for eid in torch.nonzero(active, as_tuple=False).flatten().tolist():
        s = int(start[eid].item())
        w = int(width[eid].item())
        x[eid, ..., s:s+w] = 0
    return x


def _apply_block_occlusion_depth(x: torch.Tensor, cfg: DegradationConfig, state: CorruptionState) -> torch.Tensor:
    """Rectangular occlusion on depth images (num_envs, H, W) or (num_envs, C, H, W)."""
    if cfg.occ_p <= 0 or cfg.occ_len <= 0 or cfg.occ_size <= 0:
        return x

    # find H/W
    if x.dim() == 3:
        # (N,H,W)
        _, H, W = x.shape
        ch_slice = (slice(None),)
    elif x.dim() == 4:
        # (N,C,H,W)
        _, _, H, W = x.shape
        ch_slice = (slice(None), slice(None))
    else:
        # unknown, fallback to scandot-style
        return _apply_block_occlusion_scandot(x, cfg, state)

    trigger = _maybe_trigger_counter(cfg.occ_p, state)
    new_occ = trigger & (state.occ_remaining <= 0)
    if new_occ.any():
        h = max(1, int(round(float(cfg.occ_size) * H)))
        w = max(1, int(round(float(cfg.occ_size) * W)))
        y0 = torch.randint(0, max(1, H - h + 1), (state.num_envs,), device=state.device, generator=state.gen)
        x0 = torch.randint(0, max(1, W - w + 1), (state.num_envs,), device=state.device, generator=state.gen)
        state.occ_params = (y0, x0, torch.full_like(y0, h), torch.full_like(x0, w))
        state.occ_remaining = torch.maximum(state.occ_remaining, new_occ.to(torch.int32) * int(cfg.occ_len))

    active = state.occ_remaining > 0
    state.occ_remaining = torch.clamp(state.occ_remaining - 1, min=0)
    if not active.any() or state.occ_params is None:
        return x

    y0, x0, hh, ww = state.occ_params
    x = x.clone()
    for eid in torch.nonzero(active, as_tuple=False).flatten().tolist():
        yy = int(y0[eid].item())
        xx = int(x0[eid].item())
        h = int(hh[eid].item())
        w = int(ww[eid].item())
        if x.dim() == 3:
            x[eid, yy:yy+h, xx:xx+w] = 0
        else:
            x[(eid,) + ch_slice + (slice(yy, yy+h), slice(xx, xx+w))] = 0
    return x


def corrupt_scandot(x: torch.Tensor, cfg: DegradationConfig, state: CorruptionState) -> torch.Tensor:
    # Pipeline: delay -> drop/outage -> gaussian noise -> block occlusion
    if (not cfg.enable) or (cfg.target not in ("scandot", "both")):
        return x
    x = _apply_delay(x, state)
    x = _apply_drop_outage(x, cfg, state)
    x = _apply_gauss(x, cfg.gauss_std, state)
    x = _apply_block_occlusion_scandot(x, cfg, state)
    return x


def corrupt_depth(x: torch.Tensor, cfg: DegradationConfig, state: CorruptionState) -> torch.Tensor:
    # 对 depth 传感器进行退化：延时 → 丢包/长失效 → 高斯噪声 → 矩形遮挡
    if (not cfg.enable) or (cfg.target not in ("depth", "both")):
        return x
    x = _apply_delay(x, state)
    x = _apply_drop_outage(x, cfg, state)
    x = _apply_gauss(x, cfg.gauss_std, state)
    x = _apply_block_occlusion_depth(x, cfg, state)
    return x

