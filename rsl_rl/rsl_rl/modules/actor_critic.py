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

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU
from .cross_attention_encoder import CrossAttentionEncoder
from .proprio_cross_attention_encoder import ProprioQueryCrossAttentionEncoder


class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        # self.device = device
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10
        # last_activation = nn.ELU()

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )

    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        # print("obs device", obs.device)
        # print("encoder device", next(self.encoder.parameters()).device)
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output

class Actor(nn.Module):
    def __init__(self, num_prop, 
                 num_scan, 
                 num_actions, 
                 scan_encoder_dims,
                 actor_hidden_dims, 
                 priv_encoder_dims, 
                 num_priv_latent, 
                 num_priv_explicit, 
                 num_hist, activation, 
                 tanh_encoder_output=False,
                 use_cross_attention=False,
                 cross_attention_cfg=None,
                 # scan temporal encoder ablations
                 scan_encoder_type: str = "mlp_concat",
                 scan_history_len: int = 1,
                 scan_rnn_hidden: int = 128,
                 scan_attn_d_model: int = 128,
                 scan_attn_heads: int = 4,
                 scan_attn_layers: int = 2,
                 # Scheme A: Conv1D feature extractor over per-frame scan beams before proprio cross-attention
                 scan_conv1d_enabled: bool = False,
                 scan_conv_channels: int = 64,
                 scan_conv_kernel_size: int = 5,
                 scan_conv_layers: int = 2,
                 scan_conv_pool: str = "mean",
                 ) -> None:
        super().__init__()
        # prop -> scan -> priv_explicit -> priv_latent -> hist
        # actor input: prop -> scan -> priv_explicit -> latent
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.num_priv_explicit = num_priv_explicit
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0
        self.use_cross_attention = use_cross_attention
        self.scan_encoder_type = scan_encoder_type
        self.scan_history_len = int(max(1, scan_history_len))
        self.scan_rnn_hidden = int(scan_rnn_hidden)
        self.scan_attn_d_model = int(scan_attn_d_model)
        self.scan_attn_heads = int(scan_attn_heads)
        self.scan_attn_layers = int(scan_attn_layers)
        self.scan_conv1d_enabled = bool(scan_conv1d_enabled)
        self.scan_conv_channels = int(scan_conv_channels)
        self.scan_conv_kernel_size = int(scan_conv_kernel_size)
        self.scan_conv_layers = int(scan_conv_layers)
        self.scan_conv_pool = str(scan_conv_pool)

        # scan is provided as concatenated history: (scan_per_frame_dim * scan_history_len)
        # environment already maintains scan_history_buf and flattens it into obs.
        self.scan_per_frame_dim = int(self.num_scan // self.scan_history_len) if self.scan_history_len > 0 else int(self.num_scan)

        if len(priv_encoder_dims) > 0:
                    priv_encoder_layers = []
                    priv_encoder_layers.append(nn.Linear(num_priv_latent, priv_encoder_dims[0]))
                    priv_encoder_layers.append(activation)
                    for l in range(len(priv_encoder_dims) - 1):
                        priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                        priv_encoder_layers.append(activation)
                    self.priv_encoder = nn.Sequential(*priv_encoder_layers)
                    priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, priv_encoder_output_dim)

        # 32-dim scan latent by default (consistent with scan_encoder_dims[-1])
        self.scan_encoder_output_dim = scan_encoder_dims[-1] if scan_encoder_dims else 256

        # legacy cross-attention (scan_dot OR depth_latent) path
        if self.use_cross_attention and cross_attention_cfg is not None:
            # Use cross-attention encoder for scan_dot OR depth_latent (mutually exclusive)
            self.cross_attention_encoder = CrossAttentionEncoder(
                scan_input_dim=num_scan,
                depth_latent_dim=cross_attention_cfg.get('depth_latent_dim', 32),
                d_model=cross_attention_cfg.get('d_model', 256),
                num_heads=cross_attention_cfg.get('num_heads', 8),
                num_layers=cross_attention_cfg.get('num_layers', 2),
                scan_proj_dims=cross_attention_cfg.get('scan_proj_dims', [256, 256]),
                depth_proj_dim=cross_attention_cfg.get('depth_proj_dim', 256),
                output_dim=self.scan_encoder_output_dim,
                activation=activation,
                dropout=cross_attention_cfg.get('dropout', 0.1),
            )
            self.scan_encoder = None  # Will use cross_attention_encoder instead
        elif self.if_scan_encode:
            # scan encoder ablations (all output to scan_encoder_output_dim)
            if self.scan_encoder_type in ("mlp_concat", "mlp"):
                scan_encoder = []
                scan_encoder.append(nn.Linear(num_scan, scan_encoder_dims[0]))
                scan_encoder.append(activation)
                for l in range(len(scan_encoder_dims) - 1):
                    if l == len(scan_encoder_dims) - 2:
                        scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l+1]))
                        scan_encoder.append(nn.Tanh())
                    else:
                        scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l + 1]))
                        scan_encoder.append(activation)
                self.scan_encoder = nn.Sequential(*scan_encoder)
                self.scan_rnn = None
                self.scan_rnn_out = None
                self.scan_attn_encoder = None
                self.proprio_cross_attn_encoder = None
            elif self.scan_encoder_type in ("gru", "lstm"):
                rnn_cls = nn.GRU if self.scan_encoder_type == "gru" else nn.LSTM
                self.scan_rnn = rnn_cls(
                    input_size=self.scan_per_frame_dim,
                    hidden_size=self.scan_rnn_hidden,
                    num_layers=1,
                    batch_first=True,
                )
                self.scan_rnn_out = nn.Sequential(
                    nn.Linear(self.scan_rnn_hidden, self.scan_encoder_output_dim),
                    nn.Tanh(),
                )
                self.scan_encoder = None
                self.scan_attn_encoder = None
                self.proprio_cross_attn_encoder = None
            elif self.scan_encoder_type in ("self_attention", "self_attn"):
                # reuse CrossAttentionEncoder as a self-attention encoder over scan history frames
                self.scan_attn_encoder = CrossAttentionEncoder(
                    scan_input_dim=self.scan_per_frame_dim,
                    depth_latent_dim=32,
                    d_model=self.scan_attn_d_model,
                    num_heads=self.scan_attn_heads,
                    num_layers=self.scan_attn_layers,
                    scan_proj_dims=[self.scan_attn_d_model, self.scan_attn_d_model],
                    depth_proj_dim=self.scan_attn_d_model,
                    output_dim=self.scan_encoder_output_dim,
                    activation=activation,
                    dropout=0.1,
                    frame_attention=True,
                    num_history_frames=self.scan_history_len,
                    use_positional_encoding=True,
                    use_layer_norm=True,
                )
                self.scan_encoder = None
                self.scan_rnn = None
                self.scan_rnn_out = None
                self.proprio_cross_attn_encoder = None
            elif self.scan_encoder_type in ("proprio_cross_attention", "proprio_cross_attn", "cross_attention_proprio"):
                self.proprio_cross_attn_encoder = ProprioQueryCrossAttentionEncoder(
                    scan_dim=self.scan_per_frame_dim,
                    proprio_dim=self.num_prop,
                    d_model=self.scan_attn_d_model,
                    num_heads=self.scan_attn_heads,
                    num_layers=self.scan_attn_layers,
                    output_dim=self.scan_encoder_output_dim,
                    dropout=0.1,
                    activation=activation,
                    scan_conv1d_enabled=self.scan_conv1d_enabled,
                    scan_conv_channels=self.scan_conv_channels,
                    scan_conv_kernel_size=self.scan_conv_kernel_size,
                    scan_conv_layers=self.scan_conv_layers,
                    scan_conv_pool=self.scan_conv_pool,
                )
                self.scan_encoder = None
                self.scan_rnn = None
                self.scan_rnn_out = None
                self.scan_attn_encoder = None
            else:
                raise ValueError(f"Unknown scan_encoder_type: {self.scan_encoder_type}")

            self.cross_attention_encoder = None
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan
            self.cross_attention_encoder = None
            self.scan_rnn = None
            self.scan_rnn_out = None
            self.scan_attn_encoder = None
            self.proprio_cross_attn_encoder = None
        
        actor_layers = []
        actor_layers.append(nn.Linear(num_prop+
                                      self.scan_encoder_output_dim+
                                      num_priv_explicit+
                                      priv_encoder_output_dim, 
                                      actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        if tanh_encoder_output:
            actor_layers.append(nn.Tanh())
        self.actor_backbone = nn.Sequential(*actor_layers)

    def _encode_scan(self, obs: torch.Tensor, scandots_latent=None, depth_latent=None) -> torch.Tensor:
        """
        Encode scan from obs.
        obs_scan is provided as concatenated history: (B, num_scan) = (B, scan_history_len * scan_per_frame_dim).
        """
        # legacy override (teacher/student distillation backward compatibility)
        if scandots_latent is not None:
            return scandots_latent

        obs_scan_flat = obs[:, self.num_prop:self.num_prop + self.num_scan]

        # legacy cross-attention (scan_dot OR depth_latent)
        if self.use_cross_attention and self.cross_attention_encoder is not None:
            if depth_latent is not None:
                return self.cross_attention_encoder(depth_latent=depth_latent)
            return self.cross_attention_encoder(scan_dot=obs_scan_flat)

        # reshape to sequence if needed
        if self.scan_history_len > 1 and (self.num_scan % self.scan_history_len == 0):
            scan_seq = obs_scan_flat.view(-1, self.scan_history_len, self.scan_per_frame_dim)
        else:
            scan_seq = None

        if self.scan_encoder_type in ("mlp_concat", "mlp"):
            return self.scan_encoder(obs_scan_flat)

        if self.scan_encoder_type in ("gru", "lstm"):
            if scan_seq is None:
                # fallback: treat as single frame
                scan_seq = obs_scan_flat.view(-1, 1, self.num_scan)
            out, h = self.scan_rnn(scan_seq)
            if isinstance(h, tuple):  # LSTM returns (h, c)
                h = h[0]
            last_h = h[-1]  # (B, hidden)
            return self.scan_rnn_out(last_h)

        if self.scan_encoder_type in ("self_attention", "self_attn"):
            if scan_seq is None:
                scan_seq = obs_scan_flat.view(-1, 1, self.num_scan)
            return self.scan_attn_encoder(scan_dot=scan_seq)

        if self.scan_encoder_type in ("proprio_cross_attention", "proprio_cross_attn", "cross_attention_proprio"):
            if scan_seq is None:
                scan_seq = obs_scan_flat.view(-1, 1, self.num_scan)
            proprio = obs[:, :self.num_prop]
            return self.proprio_cross_attn_encoder(proprio=proprio, scan_seq=scan_seq)

        raise ValueError(f"Unknown scan_encoder_type: {self.scan_encoder_type}")

    def forward(self, obs, hist_encoding: bool, eval=False, scandots_latent=None, depth_latent=None):
        """
        Args:
            obs: observations
            hist_encoding: whether to use history encoding
            eval: evaluation mode
            scandots_latent: pre-computed scan/depth latent (for backward compatibility)
            depth_latent: depth encoder output (for cross-attention)
        """
        if not eval:
            if self.if_scan_encode:
                scan_latent = self._encode_scan(obs, scandots_latent=scandots_latent, depth_latent=depth_latent)
                obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1)
            else:
                obs_prop_scan = obs[:, :self.num_prop + self.num_scan]
            obs_priv_explicit = obs[:, self.num_prop + self.num_scan:self.num_prop + self.num_scan + self.num_priv_explicit]
            if hist_encoding:
                latent = self.infer_hist_latent(obs)
            else:
                latent = self.infer_priv_latent(obs)
            backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)
            backbone_output = self.actor_backbone(backbone_input)
            return backbone_output
        else:
            if self.if_scan_encode:
                scan_latent = self._encode_scan(obs, scandots_latent=scandots_latent, depth_latent=depth_latent)
                obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1)
            else:
                obs_prop_scan = obs[:, :self.num_prop + self.num_scan]
            obs_priv_explicit = obs[:, self.num_prop + self.num_scan:self.num_prop + self.num_scan + self.num_priv_explicit]
            if hist_encoding:
                latent = self.infer_hist_latent(obs)
            else:
                latent = self.infer_priv_latent(obs)
            backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)
            backbone_output = self.actor_backbone(backbone_input)
            return backbone_output
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan + self.num_priv_explicit: self.num_prop + self.num_scan + self.num_priv_explicit + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    def infer_scandots_latent(self, obs):
        # Keep backward compatibility for teacher/student pipelines
        return self._encode_scan(obs, scandots_latent=None, depth_latent=None)

class ActorCriticRMA(nn.Module):
    is_recurrent = False
    def __init__(self,  num_prop,
                        num_scan,
                        num_critic_obs,
                        num_priv_latent, 
                        num_priv_explicit,
                        num_hist,
                        num_actions,
                        scan_encoder_dims=[256, 256, 256],
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        # Be careful: we historically passed many config fields via **kwargs.
        # Only warn for *truly* unknown keys; otherwise this message is misleading.
        if kwargs:
            known = {
                # core wiring
                "priv_encoder_dims",
                "tanh_encoder_output",
                # scan encoders
                "use_cross_attention",
                "cross_attention_cfg",
                "scan_encoder_type",
                "scan_history_len",
                "scan_rnn_hidden",
                "scan_attn_d_model",
                "scan_attn_heads",
                "scan_attn_layers",
                # Scheme A (scan conv1d before proprio cross-attn)
                "scan_conv1d_enabled",
                "scan_conv_channels",
                "scan_conv_kernel_size",
                "scan_conv_layers",
                "scan_conv_pool",
                # residual policy
                "residual_enabled",
                "residual_mode",
                "residual_scale",
                "residual_joint_pos_scale",
                "residual_action_scale",
                "residual_init_zero",
                "residual_base_checkpoint",
                "residual_base_use_history",
                # terrain gate
                "terrain_gate_enabled",
                "terrain_gate_mode",
                "terrain_gate_metric",
                "terrain_gate_use_last_scan_frame",
                "terrain_gate_c0",
                "terrain_gate_c1",
                "terrain_gate_k",
                # misc
                "continue_from_last_std",
                "rnn_hidden_size",
                "rnn_num_layers",
                "rnn_type",
            }
            unexpected = [k for k in kwargs.keys() if k not in known]
            if unexpected:
                print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(unexpected))
        super(ActorCriticRMA, self).__init__()

        self.kwargs = kwargs
        priv_encoder_dims= kwargs['priv_encoder_dims']
        activation = get_activation(activation)
        
        # Check if cross-attention is enabled
        use_cross_attention = kwargs.get('use_cross_attention', False)
        cross_attention_cfg = kwargs.get('cross_attention_cfg', None)
        scan_encoder_type = kwargs.get("scan_encoder_type", "mlp_concat")
        scan_history_len = kwargs.get("scan_history_len", 1)
        scan_rnn_hidden = kwargs.get("scan_rnn_hidden", 128)
        scan_attn_d_model = kwargs.get("scan_attn_d_model", 128)
        scan_attn_heads = kwargs.get("scan_attn_heads", 4)
        scan_attn_layers = kwargs.get("scan_attn_layers", 2)
        scan_conv1d_enabled = kwargs.get("scan_conv1d_enabled", False)
        scan_conv_channels = kwargs.get("scan_conv_channels", 64)
        scan_conv_kernel_size = kwargs.get("scan_conv_kernel_size", 5)
        scan_conv_layers = kwargs.get("scan_conv_layers", 2)
        scan_conv_pool = kwargs.get("scan_conv_pool", "mean")
        
        self.actor = Actor(num_prop, num_scan, num_actions, scan_encoder_dims, actor_hidden_dims, priv_encoder_dims, num_priv_latent, num_priv_explicit, num_hist, activation, 
                          tanh_encoder_output=kwargs['tanh_encoder_output'],
                          use_cross_attention=use_cross_attention,
                          cross_attention_cfg=cross_attention_cfg,
                          scan_encoder_type=scan_encoder_type,
                          scan_history_len=scan_history_len,
                          scan_rnn_hidden=scan_rnn_hidden,
                          scan_attn_d_model=scan_attn_d_model,
                          scan_attn_heads=scan_attn_heads,
                          scan_attn_layers=scan_attn_layers,
                          scan_conv1d_enabled=scan_conv1d_enabled,
                          scan_conv_channels=scan_conv_channels,
                          scan_conv_kernel_size=scan_conv_kernel_size,
                          scan_conv_layers=scan_conv_layers,
                          scan_conv_pool=scan_conv_pool)

        # Residual policy mode (frozen BC base + trainable residual).
        # Two modes:
        # - action:      a = a_base + residual_scale * a_res
        # - joint_pos:   a = a_base + (residual_joint_pos_scale / action_scale) * tanh(a_res_sim)
        self.residual_enabled = bool(kwargs.get("residual_enabled", False))
        self.residual_mode = str(kwargs.get("residual_mode", "action")).lower()
        self.residual_scale = float(kwargs.get("residual_scale", 1.0))
        self.residual_joint_pos_scale = float(
            kwargs.get("residual_joint_pos_scale", kwargs.get("residual_scale", 0.2))
        )
        self.residual_action_scale = float(kwargs.get("residual_action_scale", 0.25))
        # will be populated each step in residual mode (for reward penalty)
        self.last_delta_q = None           # (B, 12) in radians (sim order)
        self.last_delta_q_norm2 = None     # (B,) sum(delta_q^2)
        self.residual_init_zero = bool(kwargs.get("residual_init_zero", True))
        self.residual_base_checkpoint = kwargs.get("residual_base_checkpoint", None)
        self.residual_base_use_history = bool(kwargs.get("residual_base_use_history", True))
        self._reindex_idx = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=torch.long)
        # base checkpoint conventions (populated when loading base)
        self.base_action_order = "sim"          # "sim" | "policy"
        self.base_collector_format = None       # e.g. "mpc_proprio_v1"

        # 地形复杂度门控（最小侵入版）
        # 目标：根据前方 scan（局部高度图）的“复杂度”自适应缩放 residual 的介入比例
        # 形式：a_sim = a_base_sim + g(scan) * Δa_sim，其中 g ∈ [0, 1]
        self.terrain_gate_enabled = bool(kwargs.get("terrain_gate_enabled", False))
        self.terrain_gate_mode = str(kwargs.get("terrain_gate_mode", "linear")).lower()  # "linear" | "sigmoid"
        self.terrain_gate_metric = str(kwargs.get("terrain_gate_metric", "std")).lower()  # "std"（可扩展）
        self.terrain_gate_use_last_scan_frame = bool(kwargs.get("terrain_gate_use_last_scan_frame", True))
        # linear: g = clip((c-c0)/(c1-c0), 0, 1)
        self.terrain_gate_c0 = float(kwargs.get("terrain_gate_c0", 0.05))
        self.terrain_gate_c1 = float(kwargs.get("terrain_gate_c1", 0.25))
        # sigmoid: g = sigmoid(k*(c-c0))
        self.terrain_gate_k = float(kwargs.get("terrain_gate_k", 20.0))

        self.base_actor = None
        self.base_num_prop = int(num_prop)
        self.base_num_hist = int(num_hist)
        self.base_obs_dim = int(num_critic_obs)

        if self.residual_enabled and self.residual_base_checkpoint:
            self._init_base_actor_from_checkpoint(
                checkpoint_path=self.residual_base_checkpoint,
                num_actions=num_actions,
                scan_encoder_dims=scan_encoder_dims,
                actor_hidden_dims=actor_hidden_dims,
                activation=activation,
                priv_encoder_dims=priv_encoder_dims,
                num_priv_latent=num_priv_latent,
                num_priv_explicit=num_priv_explicit,
                tanh_encoder_output=kwargs.get("tanh_encoder_output", False),
                use_cross_attention=use_cross_attention,
                cross_attention_cfg=cross_attention_cfg,
                scan_encoder_type=scan_encoder_type,
                scan_history_len=scan_history_len,
                scan_rnn_hidden=scan_rnn_hidden,
                scan_attn_d_model=scan_attn_d_model,
                scan_attn_heads=scan_attn_heads,
                scan_attn_layers=scan_attn_layers,
                scan_conv1d_enabled=scan_conv1d_enabled,
                scan_conv_channels=scan_conv_channels,
                scan_conv_kernel_size=scan_conv_kernel_size,
                scan_conv_layers=scan_conv_layers,
                scan_conv_pool=scan_conv_pool,
            )

        # If residual is enabled, it's usually best to start with zero residual so that
        # initial policy == base policy.
        if self.residual_enabled and self.residual_init_zero:
            try:
                last = None
                for m in reversed(list(self.actor.actor_backbone.modules())):
                    if isinstance(m, nn.Linear):
                        last = m
                        break
                if last is not None:
                    nn.init.zeros_(last.weight)
                    if last.bias is not None:
                        nn.init.zeros_(last.bias)
            except Exception:
                pass
        

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, hist_encoding, scandots_latent=None, depth_latent=None):
        if self.residual_enabled and self.base_actor is not None:
            mean = self._residual_mean(observations, hist_encoding=hist_encoding)
        else:
            mean = self.actor(
                observations,
                hist_encoding,
                eval=False,
                scandots_latent=scandots_latent,
                depth_latent=depth_latent,
            )
        mean = torch.nan_to_num(mean)
        std = torch.nan_to_num(self.std).clamp_min(1e-5)
        self.distribution = Normal(mean, mean*0. + std)

    def act(self, observations, hist_encoding=False, **kwargs):
        self.update_distribution(
            observations,
            hist_encoding,
            scandots_latent=kwargs.get("scandots_latent", None),
            depth_latent=kwargs.get("depth_latent", None),
        )
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, hist_encoding=False, eval=False, scandots_latent=None, depth_latent=None, **kwargs):
        if not eval:
            if self.residual_enabled and self.base_actor is not None:
                actions_mean = self._residual_mean(observations, hist_encoding=hist_encoding)
            else:
                actions_mean = self.actor(observations, hist_encoding, eval, scandots_latent, depth_latent)
            return actions_mean
        else:
            actions_mean, latent_hist, latent_priv = self.actor(observations, hist_encoding, eval=True)
            return actions_mean, latent_hist, latent_priv

    def _init_base_actor_from_checkpoint(
        self,
        checkpoint_path: str,
        num_actions: int,
        scan_encoder_dims,
        actor_hidden_dims,
        activation,
        priv_encoder_dims,
        num_priv_latent: int,
        num_priv_explicit: int,
        tanh_encoder_output: bool,
        use_cross_attention: bool,
        cross_attention_cfg,
        scan_encoder_type: str,
        scan_history_len: int,
        scan_rnn_hidden: int,
        scan_attn_d_model: int,
        scan_attn_heads: int,
        scan_attn_layers: int,
    ):
        # Load checkpoint (CPU) and build a frozen base Actor.
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
        # record conventions for residual mixing
        try:
            self.base_action_order = str(meta.get("action_order", "sim")).strip().lower()
        except Exception:
            self.base_action_order = "sim"
        self.base_collector_format = meta.get("collector_format", None) if isinstance(meta, dict) else None
        # base dims (prefer meta; fallback to current dims)
        self.base_num_prop = int(meta.get("n_proprio", self.base_num_prop))
        base_num_scan = int(meta.get("n_scan", 0))
        self.base_num_hist = int(meta.get("num_hist", self.base_num_hist))
        self.base_obs_dim = int(meta.get("num_observations", self.base_obs_dim))

        # Build base actor (same module type) so we can forward it.
        self.base_actor = Actor(
            num_prop=self.base_num_prop,
            num_scan=base_num_scan,
            num_actions=num_actions,
            scan_encoder_dims=scan_encoder_dims,
            actor_hidden_dims=actor_hidden_dims,
            priv_encoder_dims=priv_encoder_dims,
            num_priv_latent=num_priv_latent,
            num_priv_explicit=num_priv_explicit,
            num_hist=self.base_num_hist,
            activation=activation,
            tanh_encoder_output=tanh_encoder_output,
            use_cross_attention=use_cross_attention,
            cross_attention_cfg=cross_attention_cfg,
            scan_encoder_type=meta.get("scan_encoder_type", scan_encoder_type),
            scan_history_len=int(meta.get("scan_history_len", scan_history_len)),
            scan_rnn_hidden=scan_rnn_hidden,
            scan_attn_d_model=scan_attn_d_model,
            scan_attn_heads=scan_attn_heads,
            scan_attn_layers=scan_attn_layers,
        )

        # Load weights (prefer actor_state_dict if present)
        sd = None
        if isinstance(ckpt, dict):
            if "actor_state_dict" in ckpt:
                sd = ckpt["actor_state_dict"]
            elif "model_state_dict" in ckpt:
                # actor keys are prefixed with 'actor.' in full model_state_dict
                raw = ckpt["model_state_dict"]
                sd = {k[len("actor."):]: v for k, v in raw.items() if k.startswith("actor.")}
        if sd is None:
            sd = ckpt

        # Filter by name+shape
        base_sd = self.base_actor.state_dict()
        filtered = {}
        for k, v in sd.items():
            if k in base_sd and hasattr(v, "shape") and tuple(v.shape) == tuple(base_sd[k].shape):
                filtered[k] = v
        self.base_actor.load_state_dict(filtered, strict=False)
        self.base_actor.eval()
        for p in self.base_actor.parameters():
            p.requires_grad_(False)

    def _residual_mean(self, observations: torch.Tensor, hist_encoding: bool) -> torch.Tensor:
        """
        Compute policy-order mean action for residual mode.

        Notes about action orders:
        - legged_gym env.step() internally does: actions_sim = reindex(actions_policy)
        - BC dataset actions are in sim DOF order, so base_actor outputs sim-order actions.
        - The permutation used by env.reindex is self-inverse, so we can convert between
          sim <-> policy with the same index.
        """
        B = observations.shape[0]
        device = observations.device
        # Base obs: pad/truncate to base_obs_dim; fill proprio (+ optionally history) only.
        base_obs = torch.zeros((B, self.base_obs_dim), device=device, dtype=observations.dtype)
        prop_dim = min(self.base_num_prop, observations.shape[1])
        base_obs[:, :prop_dim] = observations[:, :prop_dim]
        if self.residual_base_use_history:
            hist_len = self.base_num_hist * self.base_num_prop
            if hist_len > 0 and observations.shape[1] >= hist_len:
                base_obs[:, -hist_len:] = observations[:, -hist_len:]

        # Optional: match the MPC collector's proprio convention (yaw deltas masked, env flags fixed).
        # Collector layout for first 13 dims:
        # 0:3 ang_vel, 3:5 imu, 5:8 yaw-related (masked to 0), 10 cmd_x, 11/12 env_class flags.
        if self.base_collector_format == "mpc_proprio_v1" and self.base_num_prop >= 13:
            base_obs[:, 5:8] = 0.0
            base_obs[:, 11] = 1.0
            base_obs[:, 12] = 0.0

        with torch.no_grad():
            base_out = self.base_actor(base_obs, hist_encoding=True)
            base_out = torch.clamp(base_out, -1.0, 1.0)

        # residual network output is policy-order by design (same as env expects)
        res_policy = self.actor(observations, hist_encoding=hist_encoding)
        idx = self._reindex_idx.to(device)
        # Backward compatibility:
        # - Old checkpoints may have been trained with sim-order action labels.
        # - New checkpoints from train_bc.py save meta.action_order="policy".
        if self.base_action_order == "policy":
            # Keep everything in policy order
            base_policy = base_out
            res_for_add = res_policy
        else:
            # Treat base output as sim-order, convert residual to sim-order for addition,
            # and later convert total back to policy order.
            base_policy = None
            base_sim = base_out
            res_for_add = res_policy[:, idx]

        # 计算地形复杂度门控系数 g（标量，范围 [0,1]），用于缩放 residual 的影响
        # 注意：这里做的是启发式门控，不引入额外可学习参数，且不会引入“有状态”的滤波，避免训练/回放不一致。
        gate = None
        if self.terrain_gate_enabled:
            try:
                num_prop = int(getattr(self.actor, "num_prop", 0))
                num_scan = int(getattr(self.actor, "num_scan", 0))
                if num_scan > 0 and observations.shape[1] >= (num_prop + num_scan):
                    scan_flat = observations[:, num_prop:num_prop + num_scan]
                    # 若 scan 是多帧拼接（scan_history_len>1），默认只取“最近一帧”做复杂度估计
                    if self.terrain_gate_use_last_scan_frame:
                        scan_history_len = int(getattr(self.actor, "scan_history_len", 1))
                        scan_per_frame_dim = int(getattr(self.actor, "scan_per_frame_dim", num_scan))
                        if scan_history_len > 1 and scan_per_frame_dim > 0 and scan_flat.shape[1] >= scan_per_frame_dim:
                            scan_used = scan_flat[:, -scan_per_frame_dim:]
                        else:
                            scan_used = scan_flat
                    else:
                        scan_used = scan_flat

                    # 复杂度标量 c：默认用 scan 的标准差（平地→小，崎岖/障碍→大）
                    if self.terrain_gate_metric == "std":
                        c = scan_used.std(dim=1)
                    else:
                        c = scan_used.std(dim=1)

                    if self.terrain_gate_mode == "sigmoid":
                        gate = torch.sigmoid(self.terrain_gate_k * (c - self.terrain_gate_c0))
                    else:
                        denom = (self.terrain_gate_c1 - self.terrain_gate_c0)
                        if abs(denom) < 1e-12:
                            gate = torch.ones_like(c)
                        else:
                            gate = torch.clamp((c - self.terrain_gate_c0) / denom, 0.0, 1.0)
                    gate = gate.unsqueeze(1)  # (B,1) 便于广播到 12 维动作
            except Exception:
                gate = None
        if gate is None:
            gate = torch.ones((B, 1), device=device, dtype=observations.dtype)

        if self.residual_mode in ("joint_pos", "joint_position", "q"):
            # Interpret residual output as delta joint position in radians (bounded by tanh).
            # Convert to action units via division by action_scale.
            dq = torch.tanh(res_for_add) * float(self.residual_joint_pos_scale)
            da_sim = dq / float(self.residual_action_scale)
            # 用门控系数缩放 residual：地形越复杂 gate 越接近 1，平地/简单地形 gate 越接近 0
            da_sim = da_sim * gate
            if self.base_action_order == "policy":
                total_policy = base_policy + da_sim
            else:
                total_sim = base_sim + da_sim
            # expose for reward shaping: -||dq||^2
            dq_eff = dq.detach() * gate.detach()
            self.last_delta_q = dq_eff
            self.last_delta_q_norm2 = torch.sum(dq_eff ** 2, dim=1)
        else:
            # default: action residual
            if self.base_action_order == "policy":
                total_policy = base_policy + (float(self.residual_scale) * res_for_add * gate)
            else:
                total_sim = base_sim + (float(self.residual_scale) * res_for_add * gate)
            self.last_delta_q = None
            self.last_delta_q_norm2 = None
        if self.base_action_order == "policy":
            return torch.clamp(total_policy, -1.0, 1.0)
        total_policy = total_sim[:, idx]
        return torch.clamp(total_policy, -1.0, 1.0)

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
    def reset_std(self, std, num_actions, device):
        new_std = std * torch.ones(num_actions, device=device)
        self.std.data = new_std.data

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
