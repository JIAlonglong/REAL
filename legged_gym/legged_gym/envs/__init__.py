# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .base.legged_robot import LeggedRobot
from .go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO

from legged_gym.utils.task_registry import task_registry

task_registry.register("go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())
