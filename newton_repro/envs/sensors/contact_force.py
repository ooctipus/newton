# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone fingertip contact-force observation via Newton's contact sensor.

Reuses :class:`newton.sensors.SensorContact` (the same sensor the live
``isaaclab_newton`` contact sensor wraps) to read the per-counterpart contact force between a
set of sensing bodies (the fingertips) and a counterpart (the object), then expresses it in the
robot root frame -- mirroring ``dexsuite.mdp.fingers_contact_force_b``.

One sensor measures all sensing bodies across all environments; rows are sorted by body index,
so they group as ``[env0 fingers..., env1 fingers..., ...]`` and reshape to
``(num_envs, num_fingers, 3)``. Cross-environment contacts are physically impossible (envs are
spatially separated), so summing each row over its counterparts yields the per-finger force.
"""

from __future__ import annotations

import torch
import warp as wp
from envs.math.torch import quat_apply_inverse
from newton.sensors import SensorContact


class FingerContactForce:
    """Per-fingertip object-contact force in the robot root frame.

    Args:
        sim: The standalone ``NewtonSim`` (owns ``model``, ``state``, ``contacts``).
        num_envs: Number of environments.
        num_fingers: Number of fingertip sensing bodies per env (e.g. 4).
        sensing_body_expr: Regex(es) selecting the fingertip bodies (any-match).
        counterpart_body_expr: Regex(es) selecting the counterpart object body.
        device: Torch device.
    """

    def __init__(
        self,
        sim,
        *,
        num_envs: int,
        num_fingers: int,
        sensing_body_expr: list[str] | str,
        counterpart_body_expr: list[str] | str,
        device: str = "cuda:0",
    ) -> None:
        self.sim = sim
        self.num_envs = int(num_envs)
        self.num_fingers = int(num_fingers)
        self.device = device
        self.sensor = SensorContact(
            sim.model,
            sensing_bodies=sensing_body_expr,
            counterpart_bodies=counterpart_body_expr,
        )
        n_rows = self.num_envs * self.num_fingers
        if self.sensor.force_matrix is None or self.sensor.force_matrix.shape[0] != n_rows:
            shape = None if self.sensor.force_matrix is None else tuple(self.sensor.force_matrix.shape)
            raise RuntimeError(
                f"SensorContact resolved force_matrix {shape}; expected first dim {n_rows} "
                f"(num_envs={self.num_envs} x num_fingers={self.num_fingers})."
            )

    def forces_b(self, root_quat_w: torch.Tensor) -> torch.Tensor:
        """Return per-finger contact forces in the robot root frame, ``(num_envs, num_fingers * 3)``.

        Args:
            root_quat_w: Robot root world quaternion ``(num_envs, 4)`` xyzw.
        """
        self.sim.update_contacts()
        self.sensor.update(self.sim.state, self.sim.contacts)
        # (num_envs * num_fingers, max_counterparts, 3) -> sum over counterparts.
        force_w = wp.to_torch(self.sensor.force_matrix).sum(dim=1)
        force_w = force_w.view(self.num_envs, self.num_fingers, 3)
        root_quat = root_quat_w.unsqueeze(1).repeat_interleave(self.num_fingers, dim=1)
        forces_b = quat_apply_inverse(root_quat, force_w)
        return forces_b.view(self.num_envs, -1)
