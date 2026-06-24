# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-only termination terms for repro MDPs.

Ports the dexsuite lift terminations (``time_out``, ``abnormal_robot_state``, object
out-of-bound) to operate on plain tensors gathered from a :class:`newton.State`.
"""

from __future__ import annotations

import torch


def time_out(episode_length: torch.Tensor, max_episode_length: int) -> torch.Tensor:
    """``True`` where the per-env step count reached the horizon (truncation), shape ``(num_envs,)``."""
    return episode_length >= int(max_episode_length)


def abnormal_robot_state(joint_vel: torch.Tensor, joint_vel_limits: torch.Tensor) -> torch.Tensor:
    """``True`` where any actuated joint speed exceeds twice its limit (unstable physics).

    Mirrors ``dexsuite.mdp.abnormal_robot_state``.

    Args:
        joint_vel: Actuated joint velocities ``(num_envs, num_dofs)`` [rad/s].
        joint_vel_limits: Per-joint speed limits ``(num_envs, num_dofs)`` or ``(num_dofs,)`` [rad/s].
    """
    return (joint_vel.abs() > (joint_vel_limits * 2.0)).any(dim=1)


def object_out_of_bound(
    object_pos_w: torch.Tensor,
    env_origins: torch.Tensor,
    bounds_x: tuple[float, float] = (-1.5, 0.5),
    bounds_y: tuple[float, float] = (-2.0, 2.0),
    bounds_z: tuple[float, float] = (0.0, 2.0),
) -> torch.Tensor:
    """``True`` where the object left the per-env workspace box (object dropped / launched).

    Approximates ``dexsuite.mdp.out_of_bound`` (which uses the object AABB) with the object root
    position relative to the environment origin.

    Args:
        object_pos_w: Object world position ``(num_envs, 3)`` [m].
        env_origins: Per-env origin ``(num_envs, 3)`` [m].
        bounds_x: Allowed x range relative to the env origin [m].
        bounds_y: Allowed y range relative to the env origin [m].
        bounds_z: Allowed z range relative to the env origin [m].
    """
    rel = object_pos_w - env_origins
    below = (rel < torch.tensor([bounds_x[0], bounds_y[0], bounds_z[0]], device=rel.device)).any(dim=1)
    above = (rel > torch.tensor([bounds_x[1], bounds_y[1], bounds_z[1]], device=rel.device)).any(dim=1)
    return below | above
