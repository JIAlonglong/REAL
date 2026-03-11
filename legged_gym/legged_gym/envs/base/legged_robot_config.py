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

from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class play:
        load_student_config = False
        mask_priv_obs = False
        debug_viz = True

    class env:
        num_envs = 4096
        scan_history_len = 1
        n_scan = 132 * scan_history_len
        n_priv = 3 + 3 + 3
        n_priv_latent = 4 + 1 + 12 + 12
        n_proprio = 3 + 2 + 3 + 4 + 36 + 5  # 53
        history_len = 10
        depth_history_len = 2

        num_observations = n_proprio + n_scan + history_len * n_proprio + n_priv_latent + n_priv
        num_privileged_obs = None
        num_actions = 12
        env_spacing = 3.
        send_timeouts = True
        episode_length_s = 20
        obs_type = "og"

        # priv_explicit content: "vel_only" or "vel_dyn_params"
        priv_explicit_mode = "vel_dyn_params"
        priv_latent_mode = "env"

        history_encoding = True
        reorder_dofs = True

        include_foot_contacts = True

        randomize_start_pos = False
        randomize_start_vel = False
        randomize_start_yaw = False
        rand_yaw_range = 1.2
        randomize_start_y = False
        rand_y_range = 0.5
        randomize_start_pitch = False
        rand_pitch_range = 1.6

        contact_buf_len = 100
        next_goal_threshold = 0.2
        reach_goal_delay = 0.1
        num_future_goal_obs = 2

    class termination:
        pass

    class depth:
        use_camera = False
        blind = False
        camera_num_envs = 200
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.32, 0, 0.12]
        angle = [-5, 5]  # positive pitch down
        update_interval = 5

        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 3

        near_clip = 0
        far_clip = 2
        dis_noise = 0.0
        scale = 1
        invert = True

        # Blind-zone masking (memory study)
        blind_zone_enabled = False
        blind_full_enabled = False
        blind_visible_dist = 2.0
        blind_mask_dist = 1.0
        blind_goal_idx = -1
        blind_mask_value = 0.0

    class depth_rand:
        enabled = False
        curriculum_enabled = True
        curriculum_warmup_steps = 24 * 10000
        curriculum_start = 0.0
        curriculum_end = 1.0
        cam_pos_jitter = [0.05, 0.05, 0.05]
        cam_angle_jitter = [5.0, 5.0, 0.0]
        fov_jitter = 5.0
        scale_jitter = [0.95, 1.05]
        depth_noise_std = [0.0, 0.02]
        drop_rate = [0.0, 0.05]
        occlusion_prob = 0.3
        occlusion_blocks = [1, 2]
        occlusion_size = [[0.1, 0.3], [0.1, 0.3]]
        update_interval_range = [4, 7]
        near_clip_range = [0.0, 0.05]
        far_clip_range = [1.8, 2.2]

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 1.2

    class noise:
        add_noise = True
        noise_level = 1.0
        quantize_height = True

        class noise_scales:
            rotation = 0.0
            dof_pos = 0.01
            dof_vel = 0.05
            lin_vel = 0.05
            ang_vel = 0.05
            gravity = 0.02
            height_measurements = 0.02

    class terrain:
        mesh_type = 'trimesh'
        hf2mesh_method = "grid"
        max_error = 0.1
        max_error_camera = 2

        y_range = [-0.4, 0.4]
        edge_width_thresh = 0.05
        horizontal_scale = 0.05
        horizontal_scale_camera = 0.1
        vertical_scale = 0.005
        border_size = 5
        height = [0.02, 0.06]
        simplify_grid = False
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = True

        all_vertical = False
        no_flat = True
        flat_wall = False

        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2]
        measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]
        measure_horizontal_noise = 0.0

        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        fixed_terrain_level = -1
        terrain_length = 18.
        terrain_width = 4
        num_rows = 10
        num_cols = 40
        terrain_dict = {"smooth slope": 0.,
                        "rough slope up": 0.0,
                        "rough slope down": 0.0,
                        "rough stairs up": 0.,
                        "rough stairs down": 0.,
                        "discrete": 0.,
                        "stepping stones": 0.0,
                        "gaps": 0.,
                        "smooth flat": 0,
                        "pit": 0.0,
                        "wall": 0.0,
                        "platform": 0.,
                        "large stairs up": 0.,
                        "large stairs down": 0.,
                        "parkour": 0.2,
                        "parkour_hurdle": 0.2,
                        "parkour_flat": 0.2,
                        "parkour_step": 0.2,
                        "parkour_gap": 0.2,
                        "demo": 0.0,
                        "elastic_flat": 0.0,}
        terrain_proportions = list(terrain_dict.values())

        slope_treshold = 1.5
        origin_zero_z = True
        num_goals = 8

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 6.
        heading_command = True

        lin_vel_clip = 0.2
        ang_vel_clip = 0.4

        class ranges:
            lin_vel_x = [0., 1.5]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0, 0]
            heading = [0, 0]

        class max_ranges:
            lin_vel_x = [0.3, 0.8]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0, 0]
            heading = [-1.6, 1.6]

        class crclm_incremnt:
            lin_vel_x = 0.1
            lin_vel_y = 0.1
            ang_vel_yaw = 0.1
            heading = 0.5

        waypoint_delta = 0.7

    class init_state:
        pos = [0.0, 0.0, 1.]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {"joint_a": 0., "joint_b": 0.}

    class control:
        control_type = 'P'
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class asset:
        file = ""
        foot_name = "None"
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = False
        default_dof_drive_mode = 3  # 0=none, 1=pos, 2=vel, 3=effort
        self_collisions = 0
        replace_cylinder_with_capsule = True
        flip_visual_attachments = True
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0., 2.]
        randomize_base_mass = True
        added_mass_range = [0., 3.]
        randomize_base_com = True
        added_com_range = [-0.2, 0.2]
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.5
        randomize_motor = True
        motor_strength_range = [0.8, 1.2]

        delay_update_global_steps = 24 * 5000
        action_delay = False
        action_curr_step = [0, 0, 1, 1]
        action_curr_step_scratch = [0, 1]
        action_delay_view = 1
        action_buf_len = 8

    class rewards:
        class scales:
            tracking_goal_vel = 1.5
            tracking_yaw = 0.5
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.
            dof_acc = -2.5e-7
            collision = -10.
            action_rate = -0.1
            delta_torques = -1.0e-7
            torques = -0.00001
            hip_pos = -0.5
            dof_error = -0.04
            feet_stumble = -1
            feet_edge = -1

        only_positive_rewards = True
        tracking_sigma = 0.2
        soft_dof_pos_limit = 1.
        soft_dof_vel_limit = 1
        soft_torque_limit = 0.4
        base_height_target = 1.
        max_contact_force = 40.

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11., 5, 3.]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        up_axis = 1  # 0=y, 1=z

        class physx:
            num_threads = 10
            solver_type = 1  # 0=pgs, 1=tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        continue_from_last_std = True
        scan_encoder_dims = [128, 64, 32]
        # Cross-modal attention encoder (Eq. 1)
        scan_encoder_type = "proprio_cross_attention"
        scan_history_len = LeggedRobotCfg.env.scan_history_len
        scan_attn_d_model = 128
        scan_attn_heads = 4
        scan_attn_layers = 2
        scan_conv1d_enabled = False
        scan_conv_channels = 64
        scan_conv_kernel_size = 5
        scan_conv_layers = 1
        scan_conv_pool = "mean"
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        priv_encoder_dims = [64, 20]
        activation = 'elu'

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 3
        num_mini_batches = 4
        learning_rate = 1.5e-4
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 2.
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 2000, 3000]
        priv_reg_coef_schedual_resume = [0, 0.1, 0, 1]

    class depth_encoder:
        if_depth = LeggedRobotCfg.depth.use_camera
        depth_encoder_type = "mamba"
        depth_history_len = LeggedRobotCfg.env.depth_history_len
        depth_rnn_hidden = 512
        depth_mamba_d_state = 128
        depth_mamba_layers = 2
        depth_mamba_dropout = 0.0
        depth_mamba_state_loss_coef = 0.0
        depth_mamba_state_proj_dims = [64, 32]
        depth_mamba_pos_enc = True
        depth_shape = LeggedRobotCfg.depth.resized
        buffer_len = LeggedRobotCfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.e-3
        num_steps_per_env = LeggedRobotCfg.depth.update_interval * 24
        enable_latent_loss = False
        latent_loss_weight = 0.0
        # FiLM spatial modulation (Eq. 2)
        film_spatial_enabled = True
        film_spatial_sites = ["conv1", "conv2"]
        film_spatial_cnn_channels = [32, 64]
        film_gamma_scale = 0.3
        film_beta_scale = 0.3
        # Consistency-aware loss gating (Eq. 11-12)
        # k=0 disables gating (pure BC)
        consistency_gating_k = 2.0
        consistency_gating_tau = 0.5

    class estimator:
        model_type = "resnet1d"
        history_len = 10
        resnet_kernel_size = 3
        train_with_estimated_states = False
        priv_explicit_gate_enabled = True
        priv_explicit_gate_threshold = 0.2
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = LeggedRobotCfg.env.n_priv
        num_prop = LeggedRobotCfg.env.n_proprio
        num_scan = LeggedRobotCfg.env.n_scan
        # Huber-Gaussian uncertainty loss (Eq. 4-5)
        uncertainty_enabled = True
        uncertainty_warmup_iters = 200
        uncertainty_huber_delta = 5e-3
        min_log_std = -6.9
        max_log_std = 2.0
        # EKF fusion (Eq. 6-10)
        fusion_enabled = True
        fusion_q = 0.01
        fusion_r_scale = 1.0
        fusion_p0 = 1.0

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 40000
        save_interval = 100
        experiment_name = 'real_go2'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
