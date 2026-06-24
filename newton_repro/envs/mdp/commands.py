# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-only port of the dexsuite object pose command.

Mirrors ``isaaclab_tasks.core.dexsuite.mdp.commands.ObjectUniformPoseCommand``: each environment
holds a target object pose in the robot **root frame** and resamples it on a per-env timer. The
command buffer is ``(num_envs, 7)`` = ``[x, y, z, qx, qy, qz, qw]`` (quaternion **xyzw**), which is
exactly the ``target_object_pose_b`` policy observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from envs.math.torch import quat_from_euler_xyz, quat_unique


@dataclass
class PoseCommandRanges:
    """Uniform sampling ranges for the target pose (robot root frame)."""

    pos_x: tuple[float, float]
    pos_y: tuple[float, float]
    pos_z: tuple[float, float]
    roll: tuple[float, float] = (0.0, 0.0)
    pitch: tuple[float, float] = (0.0, 0.0)
    yaw: tuple[float, float] = (0.0, 0.0)


class ObjectUniformPoseCommand:
    """Per-env uniform object pose command with timed resampling.

    Args:
        num_envs: Number of environments.
        ranges: Sampling ranges in the robot root frame.
        resampling_time_range: ``(min, max)`` resample interval [s].
        step_dt: Control step duration [s] (``decimation * physics_dt``).
        make_quat_unique: Standardize quaternions to a non-negative real part (matches the cfg).
        device: Torch device.
    """

    def __init__(
        self,
        num_envs: int,
        ranges: PoseCommandRanges,
        resampling_time_range: tuple[float, float],
        step_dt: float,
        make_quat_unique: bool = True,
        device: str = "cuda:0",
    ) -> None:
        self.num_envs = int(num_envs)
        self.ranges = ranges
        self.resampling_time_range = resampling_time_range
        self.step_dt = float(step_dt)
        self.make_quat_unique = bool(make_quat_unique)
        self.device = device

        self.pose_command_b = torch.zeros(self.num_envs, 7, device=device)
        self.pose_command_b[:, 3] = 1.0  # identity quaternion (xyzw) until first resample
        self._time_left = torch.zeros(self.num_envs, device=device)

    @property
    def command(self) -> torch.Tensor:
        """Target pose in the robot root frame, ``(num_envs, 7)`` = ``[x, y, z, qx, qy, qz, qw]``."""
        return self.pose_command_b

    def _resample(self, env_ids: torch.Tensor) -> None:
        n = int(env_ids.numel())
        if n == 0:
            return
        r = torch.empty(n, device=self.device)
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.ranges.pos_x)
        self.pose_command_b[env_ids, 1] = r.uniform_(*self.ranges.pos_y)
        self.pose_command_b[env_ids, 2] = r.uniform_(*self.ranges.pos_z)
        euler = torch.zeros(n, 3, device=self.device)
        euler[:, 0].uniform_(*self.ranges.roll)
        euler[:, 1].uniform_(*self.ranges.pitch)
        euler[:, 2].uniform_(*self.ranges.yaw)
        quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.make_quat_unique else quat
        self._time_left[env_ids] = torch.empty(n, device=self.device).uniform_(*self.resampling_time_range)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Force a fresh resample for *env_ids* (called on episode reset)."""
        self._resample(env_ids)

    def step(self) -> None:
        """Advance per-env timers and resample expired environments (call once per control step)."""
        self._time_left -= self.step_dt
        expired = (self._time_left <= 0.0).nonzero(as_tuple=False).flatten()
        self._resample(expired)
