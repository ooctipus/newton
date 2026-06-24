# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-only ports of the dexsuite manipulation observation terms.

These mirror ``isaaclab_tasks.core.dexsuite.mdp.observations`` (``object_quat_b``,
``body_state_b``) but read directly from a :class:`newton.State` instead of articulation
views. They reuse :mod:`envs.math.torch` -- a verbatim copy of ``isaaclab.utils.math`` -- so the
arithmetic matches the live terms exactly.

Newton conventions handled here:

* ``state.body_q`` is ``(num_bodies, 7)`` = ``[px, py, pz, qx, qy, qz, qw]`` (quaternion **xyzw**).
  Isaac Lab 3.0 / Newton and :mod:`envs.math.torch` all use the **xyzw** convention, so quaternions
  are used as-is (no reordering).
* ``state.body_qd`` is ``(num_bodies, 6)`` spatial velocity ``[vx, vy, vz, wx, wy, wz]`` in the
  world frame (linear first, then angular; verified against a finite difference of ``body_q``).
"""

from __future__ import annotations

import torch
from envs.math.torch import quat_apply_inverse, quat_inv, quat_mul, subtract_frame_transforms


def body_pose_w(body_q: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world position ``(N, 3)`` and quaternion ``(N, 4)`` xyzw for body indices ``idx``.

    Newton ``body_q`` quaternions are xyzw, matching Isaac Lab 3.0 and :mod:`envs.math.torch`,
    so they are returned without reordering.
    """
    sel = body_q[idx]
    pos = sel[..., 0:3]
    quat_xyzw = sel[..., 3:7]
    return pos, quat_xyzw


def body_vel_w(body_qd: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world linear ``(N, 3)`` and angular ``(N, 3)`` velocity for body indices ``idx``.

    Newton ``body_qd`` packs linear velocity first (``[0:3]``), then angular (``[3:6]``).
    """
    sel = body_qd[idx]
    lin_vel = sel[..., 0:3]
    ang_vel = sel[..., 3:6]
    return lin_vel, ang_vel


def object_quat_b(robot_root_quat_w: torch.Tensor, object_quat_w: torch.Tensor) -> torch.Tensor:
    """Object orientation in the robot root frame, ``(N, 4)`` xyzw.

    Mirrors ``dexsuite.mdp.object_quat_b`` = ``quat_mul(quat_inv(robot_quat), object_quat)``.
    """
    return quat_mul(quat_inv(robot_root_quat_w), object_quat_w)


def body_state_b(
    body_q: torch.Tensor,
    body_qd: torch.Tensor,
    body_idx: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    zero_velocity: bool = False,
) -> torch.Tensor:
    """Per-body state ``[pos(3), quat(4) xyzw, linvel(3), angvel(3)]`` in the robot root frame.

    Mirrors ``dexsuite.mdp.body_state_b``. ``body_idx`` is ``(num_envs, num_bodies)``; the output
    is ``(num_envs, num_bodies * 13)`` concatenated per body.

    Args:
        body_q: Flat world body transforms ``(total_bodies, 7)``.
        body_qd: Flat world body spatial velocities ``(total_bodies, 6)``.
        body_idx: Per-env body indices to observe, shape ``(num_envs, num_bodies)``.
        root_pos_w: Robot root world position ``(num_envs, 3)``.
        root_quat_w: Robot root world quaternion ``(num_envs, 4)`` xyzw.
        zero_velocity: If ``True``, force the linear/angular velocity components to zero. Isaac
            Lab's Newton backend does not populate per-body velocities for these fixed/welded hand
            bodies (they read back as ``0``), so the trained policy always saw zeros there.
    """
    num_envs, num_bodies = body_idx.shape
    flat_idx = body_idx.reshape(-1)
    body_pos_w, body_quat_w = body_pose_w(body_q, flat_idx)

    root_pos = root_pos_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 3)
    root_quat = root_quat_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 4)

    body_pos_b, body_quat_b = subtract_frame_transforms(root_pos, root_quat, body_pos_w, body_quat_w)
    if zero_velocity:
        body_lin_vel_b = torch.zeros_like(body_pos_b)
        body_ang_vel_b = torch.zeros_like(body_pos_b)
    else:
        body_lin_vel_w, body_ang_vel_w = body_vel_w(body_qd, flat_idx)
        body_lin_vel_b = quat_apply_inverse(root_quat, body_lin_vel_w)
        body_ang_vel_b = quat_apply_inverse(root_quat, body_ang_vel_w)

    out = torch.cat((body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b), dim=1)
    return out.view(num_envs, -1)
