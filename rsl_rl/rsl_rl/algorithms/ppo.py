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

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import pickle
import numpy as np
import glob

from rsl_rl.modules import ActorCriticRMA
from rsl_rl.storage import RolloutStorage
import wandb
from rsl_rl.utils import unpad_trajectories
from isaacgym.torch_utils import quat_rotate_inverse


class RMS(object):
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape, device=device)
        self.S = torch.ones(shape, device=device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs + (delta**2) * self.n * bs / (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S

class PPO:
    actor_critic: ActorCriticRMA
    def __init__(self,
                 actor_critic,
                 estimator,
                 estimator_paras,
                 depth_encoder,
                 depth_encoder_paras,
                 depth_actor,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 dagger_update_freq=20,
                 priv_reg_coef_schedual = [0, 0, 0],
                 **kwargs
                 ):

        
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        # Only optimize trainable parameters (allows freezing submodules like residual base actor)
        trainable_params = [p for p in self.actor_critic.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable_params, lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Keep PPO-side reward postprocessing consistent with env-side `only_positive_rewards`.
        # We only clip on non-terminal steps to avoid erasing termination penalties.
        self.only_positive_rewards = bool(kwargs.get("only_positive_rewards", False))
        # Warmup/anneal coefficient for only-positive reward postprocessing (scheduled by runner):
        # alpha=1 -> hard clip at 0 (legacy behavior)
        # alpha=0 -> no-op (numerically equivalent to only_positive_rewards=False for non-terminal rewards)
        self.only_positive_alpha = 1.0

        # Adaptation
        self.hist_encoder_optimizer = optim.Adam(self.actor_critic.actor.history_encoder.parameters(), lr=learning_rate)
        self.priv_reg_coef_schedual = priv_reg_coef_schedual
        self.dagger_update_freq = dagger_update_freq
        self.counter = 0

        # Estimator
        self.estimator = estimator
        self.priv_states_dim = estimator_paras["priv_states_dim"]
        self.num_prop = estimator_paras["num_prop"]
        self.num_scan = estimator_paras["num_scan"]
        self.estimator_optimizer = optim.Adam(self.estimator.parameters(), lr=estimator_paras["learning_rate"])
        self.train_with_estimated_states = estimator_paras["train_with_estimated_states"]
        # Uncertainty-aware estimation (Eq. 4-5)
        self.estimator_uncertainty_enabled = bool(estimator_paras.get("uncertainty_enabled", False))
        self.estimator_uncertainty_warmup_iters = int(estimator_paras.get("uncertainty_warmup_iters", 0))
        self.estimator_min_log_std = float(estimator_paras.get("min_log_std", -6.9))
        self.estimator_max_log_std = float(estimator_paras.get("max_log_std", 2.0))
        self.estimator_uncertainty_huber_delta = float(estimator_paras.get("uncertainty_huber_delta", 5e-3))
        self.estimator_uncertainty_huber_lam = float(estimator_paras.get("uncertainty_huber_lam", 1e-4))
        self.estimator_history_len = int(estimator_paras.get("history_len", 1))
        self.estimator_sequence_input = bool(getattr(self.estimator, "sequence_input", False))
        self.estimator_fusion_enabled = bool(estimator_paras.get("fusion_enabled", False))
        self.estimator_fusion_q = 0.01
        self.estimator_fusion_r_scale = 1.0
        self.estimator_fusion_p0 = 1.0
        self.estimator_fusion_dt = float(estimator_paras.get("fusion_dt", 0.0))
        self.fusion_last_v = None
        self.fusion_last_P = None
        self.estimator_usage_steps = 0
        self.estimator_total_steps = 0
        self.estimator_last_source = "env"

        # Depth encoder
        self.if_depth = depth_encoder != None
        if self.if_depth:
            self.depth_encoder = depth_encoder
            self.depth_encoder_optimizer = optim.Adam(self.depth_encoder.parameters(), lr=depth_encoder_paras["learning_rate"])
            self.depth_encoder_paras = depth_encoder_paras
            self.depth_actor = depth_actor
            # Match original project: depth_actor optimizer includes BOTH depth_actor and depth_encoder parameters.
            self.depth_actor_optimizer = optim.Adam(
                [*self.depth_actor.parameters(), *self.depth_encoder.parameters()],
                lr=depth_encoder_paras["learning_rate"],
            )
            self.aux_optimizer = None
            self.depth_batch_trimmed_last = 0
            self.depth_batch_trimmed_count = 0
            self.depth_batch_trimmed_sizes = None

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, scandots_latent_shape=None):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
            scandots_latent_shape=scandots_latent_shape,
        )
        if self.estimator_fusion_enabled:
            self.fusion_last_v = torch.zeros(num_envs, 3, device=self.device)
            self.fusion_last_P = torch.eye(3, device=self.device).unsqueeze(0).repeat(num_envs, 1, 1) * float(self.estimator_fusion_p0)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def _extract_history_from_obs(self, obs):
        hlen = int(self.estimator_history_len)
        if hlen <= 1:
            return None
        need = int(hlen * self.num_prop)
        if obs is None or obs.shape[1] < need:
            return None
        return obs[:, -need:].view(obs.shape[0], hlen, self.num_prop)

    def _get_estimator_input(self, obs, info=None):
        if not self.estimator_sequence_input:
            return obs[:, : self.num_prop]
        if isinstance(info, dict):
            hist = info.get("obs_history", None)
        else:
            hist = None
        if hist is not None:
            return hist
        hist = self._extract_history_from_obs(obs)
        if hist is not None:
            return hist
        return obs[:, : self.num_prop]

    def _apply_kf_fusion(self, mean_pred, log_std, info):
        if not self.estimator_fusion_enabled:
            return None
        if self.fusion_last_v is None or self.fusion_last_P is None:
            return None
        if not isinstance(info, dict):
            return None
        acc_world = info.get("base_lin_acc", None)
        ang_vel = info.get("base_ang_vel", None)
        quat = info.get("base_quat", None)
        if acc_world is None or ang_vel is None or quat is None:
            return None
        dt = float(info.get("dt", self.estimator_fusion_dt))
        if dt <= 0.0:
            return None
        if mean_pred.shape[1] < 3:
            return None
        device = mean_pred.device
        dtype = mean_pred.dtype
        acc_world = acc_world.to(device=device, dtype=dtype)
        ang_vel = ang_vel.to(device=device, dtype=dtype)
        quat = quat.to(device=device, dtype=dtype)
        gravity = torch.tensor([0.0, 0.0, -9.81], device=device, dtype=dtype).unsqueeze(0)
        acc_pure = acc_world - gravity
        acc_body = quat_rotate_inverse(quat, acc_pure)
        v = self.fusion_last_v.to(device=device, dtype=dtype)
        P = self.fusion_last_P.to(device=device, dtype=dtype)
        wxv = torch.cross(ang_vel, v, dim=1)
        v_prop = v + (acc_body - wxv) * dt
        w = ang_vel
        wx = torch.zeros((w.shape[0], 3, 3), device=device, dtype=dtype)
        wx[:, 0, 1] = -w[:, 2]
        wx[:, 0, 2] = w[:, 1]
        wx[:, 1, 0] = w[:, 2]
        wx[:, 1, 2] = -w[:, 0]
        wx[:, 2, 0] = -w[:, 1]
        wx[:, 2, 1] = w[:, 0]
        I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        F = I - wx * dt
        Q = I * 0.01
        P_prop = F @ P @ F.transpose(1, 2) + Q
        sigma2 = torch.exp(2.0 * log_std[:, :3])
        R = torch.diag_embed(sigma2)
        S = P_prop + R
        K = P_prop @ torch.linalg.inv(S)
        residual = (mean_pred[:, :3] - v_prop).unsqueeze(-1)
        v_upd = v_prop + (K @ residual).squeeze(-1)
        P_new = (I - K) @ P_prop
        self.fusion_last_v = v_upd.detach()
        self.fusion_last_P = P_new.detach()
        fused = mean_pred.clone()
        fused[:, :3] = v_upd
        return fused

    @staticmethod
    def _raw_to_log_std(raw, estimator):
        """Convert raw uncertainty output to log σ. Handles both log_var and log_std heads."""
        if getattr(estimator, "logvar_head", None) is not None:
            return 0.5 * raw
        return raw

    def _loss_huber_gaussian_diag(self, pred, pred_logstd, targ):
        delta = float(self.estimator_uncertainty_huber_delta)
        lam = float(self.estimator_uncertainty_huber_lam)
        diff = torch.abs(pred - targ)
        quad = 0.5 * (diff ** 2)
        lin = delta * (diff - 0.5 * delta)
        huber = torch.where(diff < delta, quad, lin)
        min_log_std = float(self.estimator_min_log_std)
        pred_logstd = torch.maximum(pred_logstd, min_log_std * torch.ones_like(pred_logstd))
        gauss = ((pred - targ).pow(2)) / (2 * torch.exp(2 * pred_logstd)) + pred_logstd
        return huber + lam * gauss

    def act(self, obs, critic_obs, info, hist_encoding=False):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values, use proprio to compute estimated priv_states then actions, but store true priv_states
        scandots_latent = None
        obs_policy = obs
        # If depth is available and a depth_encoder exists, compute depth_latent and (optionally) yaw injection.
        # This enables PPO to optimize `depth_actor` directly when actor_critic.actor is swapped to depth_actor.
        try:
            if self.if_depth and isinstance(info, dict) and ("depth" in info) and (info["depth"] is not None):
                d = info["depth"]
                obs_prop_depth = obs[:, :self.num_prop].clone()
                obs_prop_depth[:, 6:8] = 0
                aa = info.get("applied_action", None)
                depth_latent_and_yaw = self.depth_encoder(d.clone(), obs_prop_depth, aa)
                depth_latent = depth_latent_and_yaw[:, :-2]
                yaw = 1.5 * depth_latent_and_yaw[:, -2:]
                scandots_latent = depth_latent
                # inject yaw prediction into policy obs when delta_yaw_ok mask is provided
                if "delta_yaw_ok" in info and info["delta_yaw_ok"] is not None:
                    try:
                        obs_policy = obs.clone()
                        ok = info["delta_yaw_ok"].to(obs_policy.device).bool()
                        obs_policy[ok, 6:8] = yaw.detach()[ok]
                    except Exception:
                        obs_policy = obs
        except Exception:
            scandots_latent = None
            obs_policy = obs

        batch_size = int(obs.shape[0]) if hasattr(obs, "shape") else 0
        if batch_size > 0:
            self.estimator_total_steps += batch_size
        if self.train_with_estimated_states:
            if batch_size > 0:
                self.estimator_usage_steps += batch_size
            self.estimator_last_source = "estimator"
            obs_est = obs_policy.clone()
            est_in = self._get_estimator_input(obs_est, info)
            has_uncertainty = (
                self.estimator_uncertainty_enabled
                and hasattr(self.estimator, "forward_with_uncertainty")
                and getattr(self.estimator, "predict_uncertainty", False)
            )
            log_std = None
            if has_uncertainty:
                mean_pred, raw = self.estimator.forward_with_uncertainty(est_in)
                log_std = self._raw_to_log_std(raw, self.estimator)
                log_std = torch.clamp(
                    log_std,
                    min=float(self.estimator_min_log_std),
                    max=float(self.estimator_max_log_std),
                )
            else:
                mean_pred = self.estimator(est_in)
            mean_for_policy = mean_pred
            if log_std is not None:
                fused = self._apply_kf_fusion(mean_pred, log_std, info)
                if fused is not None:
                    mean_for_policy = fused
            priv_states_estimated = mean_for_policy
            obs_est[:, self.num_prop+self.num_scan:self.num_prop+self.num_scan+self.priv_states_dim] = priv_states_estimated
            self.transition.actions = self.actor_critic.act(obs_est, hist_encoding, scandots_latent=scandots_latent).detach()
            # Store the *actual* actor input (with estimated priv states) to keep PPO update consistent.
            self.transition.observations = obs_est
        else:
            self.estimator_last_source = "env"
            self.transition.actions = self.actor_critic.act(obs_policy, hist_encoding, scandots_latent=scandots_latent).detach()
            self.transition.observations = obs_policy
        # Store optional scan/depth latent for PPO update (if storage supports it).
        self.transition.scandots_latent = scandots_latent

        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.critic_observations = critic_obs

        return self.transition.actions

    def reset_estimator_usage(self):
        self.estimator_usage_steps = 0
        self.estimator_total_steps = 0

    def get_estimator_usage_ratio(self):
        total = float(self.estimator_total_steps)
        if total <= 0.0:
            return 0.0
        return float(self.estimator_usage_steps) / total

    def get_estimator_input_source(self):
        if self.train_with_estimated_states:
            return "estimator"
        return self.estimator_last_source
    
    def process_env_step(self, rewards, dones, infos, next_obs=None):
        rewards_total = rewards.clone()
        self.transition.rewards = rewards_total.clone()
        self.transition.dones = dones
        if self.estimator_fusion_enabled and self.fusion_last_v is not None and self.fusion_last_P is not None:
            done_ids = dones.nonzero(as_tuple=False).flatten()
            if done_ids.numel() > 0:
                self.fusion_last_v[done_ids] = 0.0
                self.fusion_last_P[done_ids] = torch.eye(3, device=self.fusion_last_P.device).unsqueeze(0) * float(self.estimator_fusion_p0)
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
            rewards_total = self.transition.rewards.clone()

        # Keep reward postprocessing consistent with env-side `only_positive_rewards`.
        # Note: env clips BEFORE adding termination reward; PPO may further modify rewards (residual penalty, timeout bootstrap).
        # We postprocess again here *after* all PPO-side adjustments.
        if self.only_positive_rewards:
            # time_outs are truncated episodes (should be treated as non-terminal in return computation),
            # so we also postprocess those even though dones==True for them.
            if 'time_outs' in infos:
                non_terminal = (~dones) | infos['time_outs'].to(dones.device).bool()
            else:
                non_terminal = ~dones

            alpha = float(getattr(self, "only_positive_alpha", 1.0))
            if alpha <= 0.0:
                # no-op: numerically equivalent to only_positive_rewards=False for non-terminal rewards
                pass
            elif alpha >= 1.0:
                clipped = torch.clip(rewards_total, min=0.0)
                rewards_total = torch.where(non_terminal, clipped, rewards_total)
                self.transition.rewards = rewards_total.clone()
            else:
                # r' = α*clip(r,0) + (1-α)*r, only applied to non-terminal transitions
                clipped = torch.clip(rewards_total, min=0.0)
                mixed = alpha * clipped + (1.0 - alpha) * rewards_total
                rewards_total = torch.where(non_terminal, mixed, rewards_total)
                self.transition.rewards = rewards_total.clone()

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

        return rewards_total

    def set_only_positive_alpha(self, alpha: float):
        self.only_positive_alpha = max(0.0, min(1.0, float(alpha)))

    def compute_returns(self, last_critic_obs):
        # In learn_vision we may carry `critic_obs` tensors created under `torch.inference_mode()`.
        # Such "inference tensors" cannot be used in autograd-tracked ops (PyTorch will error even if we detach later).
        # We only need values here (no gradients), so:
        # 1) clone to get a normal tensor
        # 2) run under no_grad/inference_mode
        last_critic_obs = last_critic_obs.clone()
        with torch.no_grad():
            last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
    

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimator_loss = 0
        mean_estimator_rmse = 0
        mean_priv_reg_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            scandots_latent_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:

                self.actor_critic.act(
                    obs_batch,
                    masks=masks_batch,
                    hidden_states=hid_states_batch[0],
                    scandots_latent=scandots_latent_batch,
                ) # match distribution dimension

                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy
                
                # Adaptation module update
                priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
                with torch.inference_mode():
                    hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
                priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
                priv_reg_stage = min(max((self.counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3], 1)
                priv_reg_coef = priv_reg_stage * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0]) + self.priv_reg_coef_schedual[0]

                # Estimator (supervised): default MSE; optional two-stage MSE->Gaussian NLL
                # Supervision target for Estimator: prefer critic_obs (can carry unmasked privileged obs)
                target_src = critic_obs_batch if critic_obs_batch is not None else obs_batch
                target_priv = target_src[:, self.num_prop + self.num_scan : self.num_prop + self.num_scan + self.priv_states_dim]
                est_in = self._get_estimator_input(obs_batch, None)
                if self.estimator_uncertainty_enabled and hasattr(self.estimator, "forward_with_uncertainty"):
                    mean_pred, raw = self.estimator.forward_with_uncertainty(est_in)
                    err = mean_pred - target_priv
                    pred_log_std = self._raw_to_log_std(raw, self.estimator)
                    pred_log_std = torch.clamp(pred_log_std, min=self.estimator_min_log_std, max=self.estimator_max_log_std)

                    rmse_val = torch.sqrt((err ** 2 + torch.exp(2.0 * pred_log_std)).mean())
                    use_nll = (self.counter >= max(int(self.estimator_uncertainty_warmup_iters), 0))
                    if not use_nll:
                        pred_log_std = pred_log_std.detach()
                    estimator_loss = self._loss_huber_gaussian_diag(mean_pred, pred_log_std, target_priv).mean()
                else:
                    priv_states_predicted = self.estimator(est_in)
                    err = priv_states_predicted - target_priv
                    rmse_val = torch.sqrt((err ** 2).mean())
                    estimator_loss = (priv_states_predicted - target_priv).pow(2).mean()
                self.estimator_optimizer.zero_grad()
                estimator_loss.backward()
                nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
                self.estimator_optimizer.step()
                
                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + \
                       self.value_loss_coef * value_loss - \
                       self.entropy_coef * entropy_batch.mean() + \
                       priv_reg_coef * priv_reg_loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_estimator_loss += estimator_loss.item()
                mean_estimator_rmse += float(rmse_val.item())
                mean_priv_reg_loss += priv_reg_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimator_loss /= num_updates
        mean_estimator_rmse /= num_updates
        mean_priv_reg_loss /= num_updates
        self.storage.clear()
        self.update_counter()
        return mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_estimator_rmse, mean_priv_reg_loss, priv_reg_coef

    def update_dagger(self):
        mean_hist_latent_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            _scandots_latent_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
                with torch.inference_mode():
                    self.actor_critic.act(obs_batch, hist_encoding=True, masks=masks_batch, hidden_states=hid_states_batch[0])

                # Adaptation module update
                with torch.inference_mode():
                    priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
                hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
                hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
                self.hist_encoder_optimizer.zero_grad()
                hist_latent_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.actor.history_encoder.parameters(), self.max_grad_norm)
                self.hist_encoder_optimizer.step()
                
                mean_hist_latent_loss += hist_latent_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates
        self.storage.clear()
        self.update_counter()
        return mean_hist_latent_loss

    def update_depth_encoder(self, depth_latent_batch, scandots_latent_batch):
        # Match original project: update depth_encoder using latent alignment only.
        if not self.if_depth:
            return 0.0
        if not self.depth_encoder_paras.get("enable_latent_loss", True):
            return 0.0

        latent_loss_weight = float(self.depth_encoder_paras.get("latent_loss_weight", 1.0))
        depth_encoder_loss = (
            (scandots_latent_batch.detach() - depth_latent_batch).norm(p=2, dim=1).mean()
        )
        depth_encoder_loss = latent_loss_weight * depth_encoder_loss

        self.depth_encoder_optimizer.zero_grad()
        # NOTE: to allow subsequent actor+yaw update to backprop through the same graphs (as in original split updates),
        # we retain the graph here.
        depth_encoder_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.depth_encoder.parameters(), self.max_grad_norm)
        self.depth_encoder_optimizer.step()
        return float(depth_encoder_loss.item())
    
    def update_depth_actor(self, actions_student_batch, actions_teacher_batch,
                           yaw_student_batch, yaw_teacher_batch,
                           gating_k=0.0, gating_tau=0.0):
        """Update depth_actor via action + yaw imitation with optional Consistency-Aware Loss Gating.

        When gating_k > 0, applies λ = σ(k·(τ − ‖aS − aT‖₂)) per sample:
          L_BC is weighted by (1 − λ), reducing BC pressure when student
          already matches the teacher well.
        """
        if not self.if_depth:
            return 0.0, 0.0

        bc_per_sample = (actions_teacher_batch.detach() - actions_student_batch).norm(p=2, dim=1)

        if gating_k > 0:
            with torch.no_grad():
                lam = torch.sigmoid(gating_k * (gating_tau - bc_per_sample.detach()))
            depth_actor_loss = ((1.0 - lam) * bc_per_sample).mean()
        else:
            depth_actor_loss = bc_per_sample.mean()

        yaw_loss = (yaw_teacher_batch.detach() - yaw_student_batch).norm(p=2, dim=1).mean()

        loss = depth_actor_loss + yaw_loss
        self.depth_actor_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_([*self.depth_actor.parameters(), *self.depth_encoder.parameters()], self.max_grad_norm)
        self.depth_actor_optimizer.step()
        return float(depth_actor_loss.item()), float(yaw_loss.item())

    @staticmethod
    def _gaussian_log_prob(action, mean, std):
        """Log probability of *action* under diagonal Gaussian N(mean, diag(std^2))."""
        var = std.pow(2)
        return (-0.5 * ((action - mean).pow(2) / var + var.log() + math.log(2 * math.pi))).sum(dim=-1)

    def update_depth_actor_gated(
        self,
        obs_student_batch,
        actions_executed_batch,
        old_log_probs_batch,
        advantages_batch,
        actions_teacher_batch,
        yaw_student_batch,
        yaw_teacher_batch,
        depth_latent_batch,
        gating_k,
        gating_tau,
    ):
        """RL + BC gated update for student policy (Eq. 11-12).

        L_total = λ · L_RL + (1 − λ) · L_BC
        λ = σ(k · (τ − ‖aS − aT‖₂))
        """
        if not self.if_depth:
            return 0.0, 0.0, 0.0

        action_std = self.actor_critic.std.detach()

        new_action_mean = self.depth_actor(
            obs_student_batch, hist_encoding=True, scandots_latent=depth_latent_batch,
        )

        new_log_prob = self._gaussian_log_prob(actions_executed_batch, new_action_mean, action_std)
        ratio = torch.exp(new_log_prob - old_log_probs_batch)
        surr1 = -advantages_batch * ratio
        surr2 = -advantages_batch * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        l_rl = torch.max(surr1, surr2)

        l_bc = (actions_teacher_batch.detach() - new_action_mean).norm(p=2, dim=1)

        with torch.no_grad():
            lam = torch.sigmoid(gating_k * (gating_tau - l_bc.detach()))

        loss_gated = (lam * l_rl + (1.0 - lam) * l_bc).mean()

        yaw_loss = torch.tensor(0.0, device=loss_gated.device)
        if yaw_student_batch is not None and yaw_teacher_batch is not None:
            yaw_loss = (yaw_teacher_batch.detach() - yaw_student_batch).norm(p=2, dim=1).mean()

        total_loss = loss_gated + yaw_loss
        self.depth_actor_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            [*self.depth_actor.parameters(), *self.depth_encoder.parameters()],
            self.max_grad_norm,
        )
        self.depth_actor_optimizer.step()
        return float(loss_gated.item()), float(yaw_loss.item()), float(lam.mean().item())

    def update_counter(self):
        self.counter += 1
