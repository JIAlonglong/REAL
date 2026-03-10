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

import time
import os
from collections import deque
import statistics

# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
try:
    import wandb
except Exception:
    wandb = None
# import ml_runlog
import datetime

from rsl_rl.algorithms import PPO
from rsl_rl.modules import *
from rsl_rl.env import VecEnv
import sys
from copy import copy, deepcopy
import warnings

class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 init_wandb=True,
                 device='cpu', **kwargs):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.estimator_cfg = train_cfg["estimator"]
        self.depth_encoder_cfg = train_cfg["depth_encoder"]
        self.device = device
        self.env = env

        # Make PPO-side reward postprocessing consistent with env-side reward clipping.
        # Env does: clip(total_reward, min=0) when cfg.rewards.only_positive_rewards is True (before termination reward).
        # PPO may further modify reward (e.g., residual dq penalty), so we pass the flag into PPO for safe postprocessing.
        try:
            only_pos = bool(getattr(getattr(self.env.cfg, "rewards", None), "only_positive_rewards", False))
        except Exception:
            only_pos = False
        if isinstance(self.alg_cfg, dict):
            self.alg_cfg = dict(self.alg_cfg)
            self.alg_cfg.setdefault("only_positive_rewards", only_pos)

        # NOTE: this is legacy wording from the upstream repo; keep message accurate for debugging.
        try:
            scan_type = str(self.policy_cfg.get("scan_encoder_type", "mlp_concat"))
        except Exception:
            scan_type = "unknown"
        print(f"Using ActorCriticRMA (scan_encoder_type={scan_type})")
        # Extract cross-attention config if enabled
        policy_cfg_dict = dict(self.policy_cfg)
        if policy_cfg_dict.get('use_cross_attention', False):
            policy_cfg_dict['cross_attention_cfg'] = policy_cfg_dict.get('cross_attention_cfg', {})
            print(f"Cross-attention enabled: {policy_cfg_dict['cross_attention_cfg']}")
        
        actor_critic: ActorCriticRMA = ActorCriticRMA(self.env.cfg.env.n_proprio,
                                                      self.env.cfg.env.n_scan,
                                                      self.env.num_obs,
                                                      self.env.cfg.env.n_priv_latent,
                                                      self.env.cfg.env.n_priv,
                                                      self.env.cfg.env.history_len,
                                                      self.env.num_actions,
                                                      **policy_cfg_dict).to(self.device)
        priv_latent_mode = str(getattr(env.cfg.env, "priv_latent_mode", "env"))
        extra_priv = int(env.cfg.env.n_priv_latent) if priv_latent_mode == "estimator" else 0
        priv_states_dim = int(env.cfg.env.n_priv) + extra_priv
        if isinstance(self.estimator_cfg, dict):
            self.estimator_cfg = dict(self.estimator_cfg)
            self.estimator_cfg["priv_states_dim"] = priv_states_dim
        else:
            try:
                setattr(self.estimator_cfg, "priv_states_dim", priv_states_dim)
            except Exception:
                pass
        est_model_type = str(self.estimator_cfg.get("model_type", "mlp")).lower()
        if est_model_type == "tcn":
            estimator = TcnEstimator(
                input_dim=env.cfg.env.n_proprio,
                output_dim=priv_states_dim,
                num_channels=self.estimator_cfg.get("tcn_channels", [64, 64, 64, 64, 128, 128, 128]),
                kernel_size=self.estimator_cfg.get("tcn_kernel_size", 2),
                dropout=self.estimator_cfg.get("tcn_dropout", 0.2),
                activation=self.estimator_cfg.get("tcn_activation", "ReLU"),
                predict_uncertainty=bool(self.estimator_cfg.get("uncertainty_enabled", False)),
            ).to(self.device)
        elif est_model_type in ("resnet", "resnet1d"):
            estimator = ResNetEstimator1D(
                input_dim=env.cfg.env.n_proprio,
                output_dim=priv_states_dim,
                input_kernel_size=self.estimator_cfg.get("resnet_kernel_size", 7),
                predict_uncertainty=bool(self.estimator_cfg.get("uncertainty_enabled", False)),
            ).to(self.device)
        else:
            estimator = Estimator(
                input_dim=env.cfg.env.n_proprio,
                output_dim=priv_states_dim,
                hidden_dims=self.estimator_cfg["hidden_dims"],
                predict_uncertainty=bool(self.estimator_cfg.get("uncertainty_enabled", False)),
            ).to(self.device)
        # Depth encoder
        self.if_depth = self.depth_encoder_cfg["if_depth"]
        if self.if_depth:
            cnn_channels = self.depth_encoder_cfg.get("film_spatial_cnn_channels", [32, 64])
            if not isinstance(cnn_channels, (list, tuple)) or len(cnn_channels) != 2:
                cnn_channels = [32, 64]
            depth_backbone = DepthOnlyFCBackbone58x87(
                env.cfg.env.n_proprio,
                self.policy_cfg["scan_encoder_dims"][-1],
                self.depth_encoder_cfg["hidden_dims"],
                num_frames=1,
                cnn_channels=cnn_channels,
            )
            depth_encoder = RecurrentDepthBackbone(depth_backbone, env.cfg, temporal_cfg=self.depth_encoder_cfg).to(self.device)
            depth_actor = deepcopy(actor_critic.actor)
        else:
            depth_encoder = None
            depth_actor = None
        # self.depth_encoder = depth_encoder
        # self.depth_encoder_optimizer = optim.Adam(self.depth_encoder.parameters(), lr=self.depth_encoder_cfg["learning_rate"])
        # self.depth_encoder_paras = self.depth_encoder_cfg
        # self.depth_encoder_criterion = nn.MSELoss()
        # Create algorithm
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, 
                                  estimator, self.estimator_cfg, 
                                  depth_encoder, self.depth_encoder_cfg, depth_actor,
                                  device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]

        self.alg.init_storage(
            self.env.num_envs, 
            self.num_steps_per_env, 
            [self.env.num_obs], 
            [self.env.num_privileged_obs], 
            [self.env.num_actions],
            scandots_latent_shape=None,
        )

        self.learn = self.learn_RL if not self.if_depth else self.learn_vision
        # Warm-start from distillation checkpoint if provided (weights only; no optimizer)
        try:
            ckpt = str(self.depth_encoder_cfg.get("distill_base_checkpoint", "")).strip()
            resuming = bool(self.cfg.get("resume", False))
            if self.if_depth and (not resuming) and ckpt:
                print(f"[Runner] Warm-start from distillation checkpoint: {ckpt}")
                self.load(ckpt, load_optimizer=False)
        except Exception as e:
            print(f"[Runner] WARNING: failed to warm-start from checkpoint: {e}")
            
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        

    def learn_RL(self, num_learning_iterations, init_at_random_ep_len=False):
        mean_value_loss = 0.
        mean_surrogate_loss = 0.
        mean_estimator_loss = 0.
        mean_estimator_rmse = 0.
        mean_hist_latent_loss = 0.
        mean_priv_reg_loss = 0. 
        priv_reg_coef = 0.
        entropy_coef = 0.
        # initialize writer
        # if self.log_dir is not None and self.writer is None:
        #     self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        infos = {}
        # Use only the latest depth frame for stability
        infos["depth"] = self.env.depth_buffer.clone().to(self.device)[:, -1] if self.if_depth else None
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        rew_explr_buffer = deque(maxlen=100)
        rew_entropy_buffer = deque(maxlen=100)
        # AMP style reward is computed inside PPO.process_env_step() and stored in infos["amp_style_reward_mean"]
        # as a scalar (mean over envs) for the current step. We keep a short buffer for iteration-level logging.
        amp_style_buffer = deque(maxlen=200)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_explr_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_entropy_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            hist_encoding = it % self.dagger_update_freq == 0
            try:
                self.alg.reset_estimator_usage()
            except Exception:
                pass

            # --- only_positive_rewards warm-up + anneal schedule (Scheme 1) ---
            # alpha=1 -> hard clip at 0 (legacy only_positive_rewards=True)
            # alpha=0 -> no clip (equivalent to only_positive_rewards=False for non-terminal reward)
            only_positive_alpha = None
            try:
                rew_cfg = getattr(self.env.cfg, "rewards", None)
                only_pos_enabled = bool(getattr(rew_cfg, "only_positive_rewards", False)) if rew_cfg is not None else False
                if only_pos_enabled and hasattr(self.env, "set_only_positive_alpha"):
                    warm = int(getattr(rew_cfg, "only_positive_warmup_iters", 0) or 0)
                    anneal = int(getattr(rew_cfg, "only_positive_anneal_iters", 0) or 0)
                    if anneal <= 0:
                        alpha = 1.0
                    else:
                        if it < warm:
                            alpha = 1.0
                        else:
                            alpha = 1.0 - float(it - warm) / float(anneal)
                            if alpha < 0.0:
                                alpha = 0.0
                            elif alpha > 1.0:
                                alpha = 1.0
                    only_positive_alpha = float(alpha)
                    self.env.set_only_positive_alpha(only_positive_alpha)
                    # PPO also does reward postprocessing after residual/timeout bootstrap; keep it consistent.
                    if hasattr(self.alg, "set_only_positive_alpha"):
                        self.alg.set_only_positive_alpha(only_positive_alpha)
            except Exception:
                only_positive_alpha = None

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, infos, hist_encoding)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)  # obs has changed to next_obs !! if done obs has been reset
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    total_rew = self.alg.process_env_step(rewards, dones, infos, next_obs=obs)
                    
                    if self.log_dir is not None:
                        # AMP step-wise style reward (mean across envs)
                        if isinstance(infos, dict) and "amp_style_reward_mean" in infos:
                            try:
                                amp_style_buffer.append(float(infos["amp_style_reward_mean"]))
                            except Exception:
                                pass
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += total_rew
                        cur_reward_explr_sum += 0
                        cur_reward_entropy_sum += 0
                        cur_episode_length += 1
                        
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_explr_buffer.extend(cur_reward_explr_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_entropy_buffer.extend(cur_reward_entropy_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        
                        cur_reward_sum[new_ids] = 0
                        cur_reward_explr_sum[new_ids] = 0
                        cur_reward_entropy_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_estimator_rmse, mean_priv_reg_loss, priv_reg_coef = self.alg.update()
            if hist_encoding:
                print("Updating dagger...")
                mean_hist_latent_loss = self.alg.update_dagger()
            
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it < 2500:
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            elif it < 5000:
                if it % (2*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            else:
                if it % (5*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        # Expose last mean episode return for HPO / tuning (e.g. Bayesian optimization).
        self.last_mean_episode_return = float(statistics.mean(rewbuffer)) if rewbuffer else 0.0
        # self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def _apply_estimator_gate(self, obs_student, infos, gate_threshold):
        """Replace privileged states in obs with estimator predictions when error is below threshold."""
        n_prop = int(self.env.cfg.env.n_proprio)
        n_scan = int(self.env.cfg.env.n_scan)
        n_priv = int(self.env.cfg.env.n_priv)
        s, e = n_prop + n_scan, n_prop + n_scan + n_priv
        if n_priv <= 0 or obs_student.shape[1] < e:
            return obs_student

        with torch.no_grad():
            est_in = obs_student[:, :n_prop]
            if getattr(self.alg.estimator, "sequence_input", False):
                hist = infos.get("obs_history", None)
                if hist is not None:
                    est_in = hist
                else:
                    hlen = int(self.estimator_cfg.get("history_len", self.env.cfg.env.history_len))
                    need = hlen * n_prop
                    if hlen > 1 and obs_student.shape[1] >= need:
                        est_in = obs_student[:, -need:].view(obs_student.shape[0], hlen, n_prop)
            priv_pred = self.alg.estimator(est_in)

        self.alg.estimator_usage_steps += int(obs_student.shape[0])
        self.alg.estimator_last_source = "estimator"

        priv_mode = str(getattr(self.env.cfg.env, "priv_explicit_mode", "vel_only"))
        if priv_mode == "vel_only" and n_priv >= 3:
            mask = torch.zeros(n_priv, device=obs_student.device, dtype=torch.bool)
            mask[:3] = True
        else:
            mask = torch.ones(n_priv, device=obs_student.device, dtype=torch.bool)

        if mask.any():
            diff = (priv_pred - obs_student[:, s:e])[:, mask]
            err = diff.norm(p=2, dim=1) / float(mask.sum().item()) ** 0.5
            ok = err <= gate_threshold
            if ok.any():
                obs_student = obs_student.clone()
                filled = obs_student[:, s:e].clone()
                filled[:, mask] = priv_pred[:, mask]
                obs_student[:, s:e] = torch.where(ok[:, None], filled, obs_student[:, s:e])
        return obs_student

    def learn_vision(self, num_learning_iterations, init_at_random_ep_len=False):
        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        infos = {}
        infos["depth"] = self.env.depth_buffer.clone().to(self.device)[:, -1] if self.if_depth else None
        infos["delta_yaw_ok"] = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        infos["applied_action"] = torch.zeros(self.env.num_envs, self.env.num_actions, dtype=torch.float, device=self.device)

        # Always keep modules in train mode; if RL is enabled we also train actor_critic.
        self.alg.actor_critic.train()
        self.alg.depth_encoder.train()
        self.alg.depth_actor.train()
        # Teacher policy for distillation signals.
        # If a teacher is attached externally as `alg.teacher_actor_critic`, use it; otherwise fall back to `alg.actor_critic`.
        teacher_ac = getattr(self.alg, "teacher_actor_critic", None)
        if teacher_ac is None:
            teacher_ac = self.alg.actor_critic
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            hist_encoding = it % self.dagger_update_freq == 0
            self.alg.reset_estimator_usage()

            # --- only_positive_rewards warm-up + anneal schedule (Scheme 1) ---
            only_positive_alpha = None
            try:
                rew_cfg = getattr(self.env.cfg, "rewards", None)
                only_pos_enabled = bool(getattr(rew_cfg, "only_positive_rewards", False)) if rew_cfg is not None else False
                if only_pos_enabled and hasattr(self.env, "set_only_positive_alpha"):
                    warm = int(getattr(rew_cfg, "only_positive_warmup_iters", 0) or 0)
                    anneal = int(getattr(rew_cfg, "only_positive_anneal_iters", 0) or 0)
                    if anneal <= 0:
                        alpha = 1.0
                    else:
                        if it < warm:
                            alpha = 1.0
                        else:
                            alpha = 1.0 - float(it - warm) / float(anneal)
                            if alpha < 0.0:
                                alpha = 0.0
                            elif alpha > 1.0:
                                alpha = 1.0
                    only_positive_alpha = float(alpha)
                    self.env.set_only_positive_alpha(only_positive_alpha)
                    if hasattr(self.alg, "set_only_positive_alpha"):
                        self.alg.set_only_positive_alpha(only_positive_alpha)
            except Exception:
                only_positive_alpha = None
            do_rl = False
            do_distill = True
            depth_latent_buffer = []
            scandots_latent_buffer = []
            actions_teacher_buffer = []
            actions_student_buffer = []
            yaw_buffer_student = []
            yaw_buffer_teacher = []
            delta_yaw_ok_buffer = []
            gating_k = float(self.depth_encoder_cfg.get("consistency_gating_k", 0.0))
            gating_tau = float(self.depth_encoder_cfg.get("consistency_gating_tau", 0.0))
            use_gating_rl = gating_k > 0
            rl_obs_student_buf = []
            rl_actions_exec_buf = []
            rl_old_lp_buf = []
            rl_value_buf = []
            rl_reward_buf = []
            rl_done_buf = []
            rl_depth_latent_buf = []
            depth_latent = None
            yaw = None
            ok_mask = None
            step_count = int(self.depth_encoder_cfg["num_steps_per_env"])
            for i in range(step_count):
                collect_depth_step = infos.get("depth", None) is not None

                # --- Teacher action (no grad; always available) ---
                with torch.no_grad():
                    actions_teacher = teacher_ac.act_inference(
                        obs, hist_encoding=True, scandots_latent=None
                    )
                actions_teacher_buffer.append(actions_teacher)

                # --- Teacher scan latent + student depth latent/yaw (only when a depth frame arrives) ---
                if collect_depth_step:
                    with torch.no_grad():
                        scandots_latent = teacher_ac.actor.infer_scandots_latent(obs)
                    scandots_latent_buffer.append(scandots_latent)

                    obs_prop_depth = obs[:, :self.env.cfg.env.n_proprio].clone()
                    obs_prop_depth[:, 6:8] = 0
                    aa = infos.get("applied_action", None)
                    depth_latent_and_yaw = self.alg.depth_encoder(infos["depth"].clone(), obs_prop_depth, aa)
                    depth_latent = depth_latent_and_yaw[:, :-2]
                    yaw = 1.5 * depth_latent_and_yaw[:, -2:]
                    depth_latent_buffer.append(depth_latent)
                # else: reuse last depth_latent/yaw if present

                if yaw is None or depth_latent is None:
                    actions_student = actions_teacher
                else:
                    obs_student = obs.clone()
                    ok_mask = infos.get("delta_yaw_ok", None)
                    if ok_mask is not None:
                        obs_student[ok_mask, 6:8] = yaw.detach()[ok_mask]
                    else:
                        obs_student[:, 6:8] = yaw.detach()

                    self.alg.estimator_total_steps += int(obs_student.shape[0])
                    self.alg.estimator_last_source = "env"

                    gate_enabled = bool(self.estimator_cfg.get("priv_explicit_gate_enabled", False))
                    gate_threshold = float(self.estimator_cfg.get("priv_explicit_gate_threshold", 0.0))
                    if gate_enabled and gate_threshold > 0.0:
                        obs_student = self._apply_estimator_gate(obs_student, infos, gate_threshold)

                    actions_student = self.alg.depth_actor(
                        obs_student, hist_encoding=True, scandots_latent=depth_latent,
                    )
                actions_student_buffer.append(actions_student)
                if collect_depth_step:
                    yaw_buffer_student.append(yaw)
                    yaw_buffer_teacher.append(obs[:, 6:8])
                if ok_mask is not None:
                    delta_yaw_ok_buffer.append(
                        float(ok_mask.sum().item()) / float(ok_mask.numel())
                    )

                # Step env (detach actions to avoid backprop through physics)
                is_student_step = (depth_latent is not None) and (actions_student is not actions_teacher)
                if use_gating_rl and is_student_step:
                    action_std = self.alg.actor_critic.std.detach()
                    noise = torch.randn_like(actions_student)
                    actions_exec = (actions_student.detach() + action_std * noise)
                    old_lp = self.alg._gaussian_log_prob(actions_exec, actions_student.detach(), action_std)
                    with torch.no_grad():
                        val = self.alg.actor_critic.evaluate(critic_obs).squeeze(-1)
                    rl_obs_student_buf.append(obs_student.clone().detach())
                    rl_actions_exec_buf.append(actions_exec.detach())
                    rl_old_lp_buf.append(old_lp.detach())
                    rl_value_buf.append(val)
                    rl_depth_latent_buf.append(depth_latent.clone().detach())
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions_exec.detach())
                else:
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions_student.detach())
                critic_obs = privileged_obs if privileged_obs is not None else obs
                obs, critic_obs, rewards, dones = (
                    obs.to(self.device),
                    critic_obs.to(self.device),
                    rewards.to(self.device),
                    dones.to(self.device),
                )
                if use_gating_rl and is_student_step:
                    rl_reward_buf.append(rewards)
                    rl_done_buf.append(dones)

                if self.log_dir is not None:
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0
                
            stop = time.time()
            collection_time = stop - start
            start = stop

            # PPO learning step (optional)
            mean_value_loss = 0.0
            mean_surrogate_loss = 0.0
            mean_estimator_loss = 0.0
            mean_estimator_rmse = 0.0
            mean_priv_reg_loss = 0.0
            priv_reg_coef = 0.0
            mean_hist_latent_loss = 0.0
            if do_rl:
                self.alg.compute_returns(critic_obs)
                (
                    mean_value_loss,
                    mean_surrogate_loss,
                    mean_estimator_loss,
                    mean_estimator_rmse,
                    mean_priv_reg_loss,
                    priv_reg_coef,
                ) = self.alg.update()
                if hist_encoding:
                    # keep consistent with learn_RL: occasional dagger update for history encoder
                    mean_hist_latent_loss = self.alg.update_dagger()

            # In alternate training, some iterations may skip distillation buffers entirely.
            # Avoid division-by-zero in logging.
            if len(delta_yaw_ok_buffer) > 0:
                delta_yaw_ok_percentage = sum(delta_yaw_ok_buffer) / len(delta_yaw_ok_buffer)
            else:
                delta_yaw_ok_percentage = 1.0
            # When rollout is under inference_mode, actions_student/depth_latent have no grad.
            # Prefer recompute-with-grad path if we have raw obs/depth buffers.
            if do_distill and (len(actions_student_buffer) > 0):
                # Original-style split updates:
                # 1) update encoder with latent alignment
                # 2) update actor with action+yaw imitation (optimizer may include encoder too, as in original)
                scandots_latent_batch = torch.cat(scandots_latent_buffer, dim=0)
                actions_teacher_batch = torch.cat(actions_teacher_buffer, dim=0)
                actions_student_batch = torch.cat(actions_student_buffer, dim=0)
                # depth_latent/yaw may be sparse if depth frames are sparse; guard by using only collected entries
                depth_encoder_loss = 0.0
                yaw_loss = 0.0
                if len(depth_latent_buffer) > 0 and len(scandots_latent_buffer) > 0:
                    depth_latent_batch = torch.cat(depth_latent_buffer, dim=0)
                    # Match original project: encoder update uses latent alignment only.
                    n = min(int(depth_latent_batch.shape[0]), int(scandots_latent_batch.shape[0]))
                    depth_encoder_loss = self.alg.update_depth_encoder(
                        depth_latent_batch[:n],
                        scandots_latent_batch[:n],
                    )
                    # yaw loss is computed in actor update
                    yaw_loss = 0.0

                m = min(int(actions_student_batch.shape[0]), int(actions_teacher_batch.shape[0]))
                yaw_student_batch = torch.cat(yaw_buffer_student, dim=0) if len(yaw_buffer_student) > 0 else None
                yaw_teacher_batch = torch.cat(yaw_buffer_teacher, dim=0) if len(yaw_buffer_teacher) > 0 else None

                if use_gating_rl and len(rl_obs_student_buf) > 0:
                    # Compute GAE advantages for RL term
                    n_rl = len(rl_reward_buf)
                    with torch.no_grad():
                        last_val = self.alg.actor_critic.evaluate(critic_obs).squeeze(-1)
                    gamma = float(getattr(self.alg, "gamma", 0.99))
                    gae_lam = float(getattr(self.alg, "lam", 0.95))
                    adv_list = [None] * n_rl
                    gae = torch.zeros_like(rl_reward_buf[0])
                    for t in reversed(range(n_rl)):
                        nv = last_val if t == n_rl - 1 else rl_value_buf[t + 1]
                        nt = 1.0 - rl_done_buf[t].float()
                        delta = rl_reward_buf[t] + gamma * nv * nt - rl_value_buf[t]
                        gae = delta + gamma * gae_lam * nt * gae
                        adv_list[t] = gae
                    adv_batch = torch.cat(adv_list, dim=0)
                    adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

                    obs_s = torch.cat(rl_obs_student_buf, dim=0)
                    act_e = torch.cat(rl_actions_exec_buf, dim=0)
                    olp = torch.cat(rl_old_lp_buf, dim=0)
                    dl = torch.cat(rl_depth_latent_buf, dim=0)
                    at = torch.cat(actions_teacher_buffer, dim=0)[:obs_s.shape[0]]

                    yw_s = yaw_student_batch[:obs_s.shape[0]] if yaw_student_batch is not None else None
                    yw_t = yaw_teacher_batch[:obs_s.shape[0]] if yaw_teacher_batch is not None else None

                    depth_actor_loss, yaw_loss, _mean_lam = self.alg.update_depth_actor_gated(
                        obs_s, act_e, olp, adv_batch, at,
                        yw_s, yw_t, dl,
                        gating_k, gating_tau,
                    )
                elif yaw_student_batch is not None and yaw_teacher_batch is not None:
                    k = min(int(yaw_student_batch.shape[0]), int(yaw_teacher_batch.shape[0]), int(m))
                    depth_actor_loss, yaw_loss = self.alg.update_depth_actor(
                        actions_student_batch[:k],
                        actions_teacher_batch[:k],
                        yaw_student_batch[:k],
                        yaw_teacher_batch[:k],
                        gating_k=gating_k,
                        gating_tau=gating_tau,
                    )
                else:
                    depth_actor_loss, yaw_loss = 0.0, 0.0
                amp_style_buffer = []
                depth_batch_trimmed_last = 0
                depth_batch_trimmed_count = int(getattr(self.alg, "depth_batch_trimmed_count", 0))
            else:
                depth_encoder_loss, depth_actor_loss, yaw_loss = 0.0, 0.0, 0.0
                amp_style_buffer = []
                depth_batch_trimmed_last = 0
                depth_batch_trimmed_count = int(getattr(self.alg, "depth_batch_trimmed_count", 0))
            self.alg.update_counter()

            stop = time.time()
            learn_time = stop - start

            self.alg.depth_encoder.detach_hidden_states()

            if self.log_dir is not None:
                self.log_vision(locals())
            if (it-self.start_learning_iteration < 2500 and it % self.save_interval == 0) or \
               (it-self.start_learning_iteration < 5000 and it % (2*self.save_interval) == 0) or \
               (it-self.start_learning_iteration >= 5000 and it % (5*self.save_interval) == 0):
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
    
    def log_vision(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                wandb_dict['Episode_rew/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        wandb_dict['Loss_depth/delta_yaw_ok_percent'] = locs['delta_yaw_ok_percentage']
        wandb_dict['Loss_depth/depth_encoder'] = locs['depth_encoder_loss']
        wandb_dict['Loss_depth/depth_actor'] = locs['depth_actor_loss']
        wandb_dict['Loss_depth/yaw'] = locs['yaw_loss']
        try:
            wandb_dict['Estimator/usage_ratio'] = float(self.alg.get_estimator_usage_ratio())
            wandb_dict['Estimator/use_estimator'] = 1.0 if self.alg.get_estimator_input_source() == "estimator" else 0.0
        except Exception:
            pass
        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
        if 'amp_style_buffer' in locs and len(locs['amp_style_buffer']) > 0:
            wandb_dict['Train/amp_style_reward_mean'] = statistics.mean(locs['amp_style_buffer'])
        if 'depth_batch_trimmed_last' in locs:
            wandb_dict['Loss_depth/batch_trimmed_last'] = float(locs['depth_batch_trimmed_last'])
        if 'depth_batch_trimmed_count' in locs:
            wandb_dict['Loss_depth/batch_trimmed_count'] = float(locs['depth_batch_trimmed_count'])
        
        
        if wandb is not None:
            wandb.log(wandb_dict, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            amp_style_line = ""
            if 'amp_style_buffer' in locs and len(locs['amp_style_buffer']) > 0:
                amp_style_line = f"""{'AMP style reward (mean):':>{pad}} {statistics.mean(locs['amp_style_buffer']):.4f}\n"""
            trim_line = ""
            if 'depth_batch_trimmed_last' in locs and 'depth_batch_trimmed_count' in locs:
                trim_line = f"""{'Batch trimmed (last/count):':>{pad}} {int(locs['depth_batch_trimmed_last'])}/{int(locs['depth_batch_trimmed_count'])}\n"""
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Estimator input:':>{pad}} {self.alg.get_estimator_input_source()}\n"""
                          f"""{'Estimator usage ratio:':>{pad}} {self.alg.get_estimator_usage_ratio():.4f}\n"""
                          f"""{'Estimator RMSE:':>{pad}} {locs['mean_estimator_rmse']:.6f}\n"""
                          f"""{amp_style_line}"""
                          f"""{trim_line}"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Depth encoder loss:':>{pad}} {locs['depth_encoder_loss']:.4f}\n"""
                          f"""{'Depth actor loss:':>{pad}} {locs['depth_actor_loss']:.4f}\n"""
                          f"""{'Yaw loss:':>{pad}} {locs['yaw_loss']:.4f}\n"""
                          f"""{'Delta yaw ok percentage:':>{pad}} {locs['delta_yaw_ok_percentage']:.4f}\n""")
        else:
            log_string = (f"""{'#' * width}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                wandb_dict['Episode_rew/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        wandb_dict['Loss/value_function'] = ['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_dict['Loss/estimator'] = locs['mean_estimator_loss']
        wandb_dict['Estimator/rmse'] = locs['mean_estimator_rmse']
        wandb_dict['Loss/hist_latent_loss'] = locs['mean_hist_latent_loss']
        wandb_dict['Loss/priv_reg_loss'] = locs['mean_priv_reg_loss']
        wandb_dict['Loss/priv_ref_lambda'] = locs['priv_reg_coef']
        wandb_dict['Loss/entropy_coef'] = locs['entropy_coef']
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
        try:
            wandb_dict['Estimator/usage_ratio'] = float(self.alg.get_estimator_usage_ratio())
            wandb_dict['Estimator/use_estimator'] = 1.0 if self.alg.get_estimator_input_source() == "estimator" else 0.0
        except Exception:
            pass

        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        # Reward clipping schedule (only_positive_rewards alpha)
        if "only_positive_alpha" in locs and locs["only_positive_alpha"] is not None:
            try:
                wandb_dict["Train/only_positive_alpha"] = float(locs["only_positive_alpha"])
            except Exception:
                pass
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_reward_explr'] = statistics.mean(locs['rew_explr_buffer'])
            wandb_dict['Train/mean_reward_task'] = wandb_dict['Train/mean_reward'] - wandb_dict['Train/mean_reward_explr']
            wandb_dict['Train/mean_reward_entropy'] = statistics.mean(locs['rew_entropy_buffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
            # wandb_dict['Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            # wandb_dict['Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)
        if 'amp_style_buffer' in locs and len(locs['amp_style_buffer']) > 0:
            wandb_dict['Train/amp_style_reward_mean'] = statistics.mean(locs['amp_style_buffer'])
        
        
        if wandb is not None:
            wandb.log(wandb_dict, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            amp_style_line = ""
            if 'amp_style_buffer' in locs and len(locs['amp_style_buffer']) > 0:
                amp_style_line = f"""{'AMP style reward (mean):':>{pad}} {statistics.mean(locs['amp_style_buffer']):.4f}\n"""
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimator RMSE:':>{pad}} {locs['mean_estimator_rmse']:.6f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{amp_style_line}"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean reward (task):':>{pad}} {statistics.mean(locs['rewbuffer']) - statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                          f"""{'Mean reward (exploration):':>{pad}} {statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                          f"""{'Mean reward (entropy):':>{pad}} {statistics.mean(locs['rew_entropy_buffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimator loss:':>{pad}} {locs['mean_estimator_loss']:.4f}\n"""
                          f"""{'Estimator RMSE:':>{pad}} {locs['mean_estimator_rmse']:.6f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Estimator input:':>{pad}} {self.alg.get_estimator_input_source()}\n"""
                          f"""{'Estimator usage ratio:':>{pad}} {self.alg.get_estimator_usage_ratio():.4f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)

    def save(self, path, infos=None):
        state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'estimator_state_dict': self.alg.estimator.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }
        if self.if_depth:
            state_dict['depth_encoder_state_dict'] = self.alg.depth_encoder.state_dict()
            state_dict['depth_actor_state_dict'] = self.alg.depth_actor.state_dict()
        torch.save(state_dict, path)

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print("Loading model from {}...".format(path))
        loaded_dict = torch.load(path, map_location=self.device)
        # Backward compatibility: actor_critic may gain optional modules over time (e.g., scan conv encoders).
        # Try strict load first; if it fails due to missing/unexpected keys, fall back to strict=False.
        try:
            self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        except RuntimeError as e:
            print("[WARN] actor_critic strict load failed, retrying with strict=False.")
            print(f"[WARN] strict load error: {e}")
            try:
                out = self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'], strict=False)
                # torch returns an _IncompatibleKeys object (missing_keys, unexpected_keys)
                if hasattr(out, "missing_keys") and hasattr(out, "unexpected_keys"):
                    print(f"[WARN] actor_critic missing keys: {len(out.missing_keys)}; unexpected keys: {len(out.unexpected_keys)}")
            except TypeError:
                # Older torch versions may not accept strict kw here (rare); keep previous behavior.
                self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        try:
            strict = True
            if bool(getattr(self.alg.estimator, "predict_uncertainty", False)):
                strict = False
            try:
                self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'], strict=strict)
            except RuntimeError as e:
                print("[WARN] estimator strict load failed, retrying with strict=False.")
                print(f"[WARN] strict load error: {e}")
                out = self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'], strict=False)
                if hasattr(out, "missing_keys") and hasattr(out, "unexpected_keys"):
                    print(f"[WARN] estimator missing keys: {len(out.missing_keys)}; unexpected keys: {len(out.unexpected_keys)}")
        except TypeError:
            self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'])
        if self.if_depth:
            if 'depth_encoder_state_dict' not in loaded_dict:
                warnings.warn("'depth_encoder_state_dict' key does not exist, not loading depth encoder...")
            else:
                print("Saved depth encoder detected, loading...")
                # Backward compatibility: depth encoder may gain optional modules (e.g., delay-aware GRU).
                self.alg.depth_encoder.load_state_dict(loaded_dict['depth_encoder_state_dict'], strict=False)
            if 'depth_actor_state_dict' in loaded_dict:
                print("Saved depth actor detected, loading...")
                self.alg.depth_actor.load_state_dict(loaded_dict['depth_actor_state_dict'])
            else:
                print("No saved depth actor, Copying actor critic actor to depth actor...")
                self.alg.depth_actor.load_state_dict(self.alg.actor_critic.actor.state_dict())
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        # self.current_learning_iteration = loaded_dict['iter']
        print("*" * 80)
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    
    def get_depth_actor_inference_policy(self, device=None):
        self.alg.depth_actor.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.depth_actor.to(device)
        return self.alg.depth_actor
    
    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic
    
    def get_estimator_inference_policy(self, device=None):
        self.alg.estimator.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.estimator.to(device)
        return self.alg.estimator.inference

    def get_depth_encoder_inference_policy(self, device=None):
        self.alg.depth_encoder.eval()
        if device is not None:
            self.alg.depth_encoder.to(device)
        return self.alg.depth_encoder
    
