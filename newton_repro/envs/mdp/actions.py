# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-only joint actuator helpers for repro MDPs.

Provides :class:`ImplicitJointActuator`, a standalone analogue of Isaac Lab's
``ImplicitActuator`` + ``RelativeJointPositionAction`` on the Newton MuJoCo backend:

* Implicit PD is realized by writing per-DOF ``model.joint_target_ke`` / ``joint_target_kd``
  (and ``joint_armature``) and letting the solver track ``control.joint_target_pos``. The
  freshly rebuilt model carries USD-default drive gains, so the captured live gains (which also
  include startup-randomization scaling) must be applied here.
* Relative position action: ``target = current_joint_pos + scale * raw_action`` (matching
  :class:`isaaclab.envs.mdp.actions.RelativeJointPositionAction` with ``use_zero_offset=True``),
  computed once per control step and held across the decimation substeps.
"""

from __future__ import annotations

import torch
import warp as wp
from newton.solvers import SolverNotifyFlags


class ImplicitJointActuator:
    """Implicit PD position actuator over a robot's actuated DOFs.

    Args:
        sim: The standalone ``NewtonSim`` (owns ``model``, ``state``, ``control``).
        entities: :class:`~envs.newton_entities.RobotEntities` index tables.
        target_ke: Per-DOF position gains for env 0, shape ``(num_dofs,)`` [N·m/rad].
        target_kd: Per-DOF damping gains for env 0, shape ``(num_dofs,)`` [N·m·s/rad].
        armature: Optional per-DOF armature for env 0, shape ``(num_dofs,)`` [kg·m^2].
        scale: Relative-action scale (Isaac Lab ``RelativeJointPositionActionCfg.scale``).
        device: Torch device.
    """

    def __init__(
        self,
        sim,
        entities,
        *,
        target_ke: torch.Tensor,
        target_kd: torch.Tensor,
        armature: torch.Tensor | None = None,
        effort_limit: torch.Tensor | None = None,
        scale: float = 0.1,
        device: str = "cuda:0",
    ) -> None:
        self.sim = sim
        self.entities = entities
        self.scale = float(scale)
        self.device = device
        self.num_envs = entities.num_envs
        self.num_dofs = entities.num_dofs
        # For 1-DOF actuated joints the coord and DOF indices coincide.
        self._dof_flat = entities.joint_dof_idx.reshape(-1)
        self._coord_flat = entities.joint_coord_idx.reshape(-1)

        self._apply_gains(target_ke, target_kd, armature, effort_limit)

        self._joint_q = wp.to_torch(sim.state.joint_q)
        self._target_pos = wp.to_torch(sim.control.joint_target_pos)
        self._targets = torch.zeros(self.num_envs, self.num_dofs, device=device)

    def _apply_gains(self, target_ke, target_kd, armature, effort_limit) -> None:
        for name, vals in (
            ("joint_target_ke", target_ke),
            ("joint_target_kd", target_kd),
            ("joint_armature", armature),
            ("joint_effort_limit", effort_limit),
        ):
            if vals is None:
                continue
            arr = getattr(self.sim.model, name, None)
            if arr is None:
                continue
            tensor = wp.to_torch(arr)
            per_env = torch.as_tensor(vals, dtype=tensor.dtype, device=tensor.device).reshape(self.num_dofs)
            tensor[self._dof_flat] = per_env.repeat(self.num_envs)
        self.sim.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    def current_joint_pos(self) -> torch.Tensor:
        """Current actuated joint positions ``(num_envs, num_dofs)`` [rad]."""
        return self._joint_q[self._coord_flat].view(self.num_envs, self.num_dofs)

    def set_action(self, raw_action: torch.Tensor) -> None:
        """Snapshot the position target from the raw action at the start of a control step."""
        self._targets = self.current_joint_pos() + self.scale * raw_action.to(self.device)

    def apply(self) -> None:
        """Write the held position target into ``control.joint_target_pos`` (idempotent per substep)."""
        self._target_pos[self._dof_flat] = self._targets.reshape(-1).to(self._target_pos.dtype)
