# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str | None = None,
    near_goal_threshold: float = 0.1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # good idea
    """Reward the agent for lifting the object above the minimal height, but stop rewarding when near goal."""
    object: RigidObject = env.scene[object_cfg.name]
    is_lifted = object.data.root_pos_w[:, 2] > minimal_height

    # If command_name is provided, check if near goal and reduce lifting reward
    if command_name is not None:
        robot: RigidObject = env.scene[robot_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_pos_b = command[:, :3]
        des_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
        )
        distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
        near_goal = distance < near_goal_threshold
        # Don't reward lifting when near goal
        return torch.where(is_lifted & ~near_goal, 1.0, 0.0)

    return torch.where(is_lifted, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (
        1 - torch.tanh(distance / std)
    )


def object_goal_distance_on_table(
    env: ManagerBasedRLEnv,
    std: float,
    table_height: float,
    height_tolerance: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for placing the object at the goal position on the table."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )
    # distance of the object to the goal position: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # check if object is near table height
    height_diff = torch.abs(object.data.root_pos_w[:, 2] - table_height)
    on_table = height_diff < height_tolerance

    # Get gripper action from the action manager (last action is gripper binary action)
    # For BinaryJointAction: positive = open, negative = close
    gripper_action = env.action_manager.get_term("gripper_action").raw_actions.squeeze(-1)

    # Only reward placing when near goal with gripper open (positive action)
    near_goal = distance < 0.08
    gripper_open = gripper_action > 0.0

    # Strong reward for object on table, at goal, with gripper open
    success_reward = torch.where(
        on_table & near_goal & gripper_open,
        (1 - torch.tanh(distance / std)) * 2.0,  # Double reward when gripper is open
        torch.zeros_like(distance),
    )

    # Base reward for just being close to table position (encourages approach)
    approach_reward = on_table * (1 - torch.tanh(distance / std)) * 0.3

    return success_reward + approach_reward
