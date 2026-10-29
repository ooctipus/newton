# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The joint and free-body tables that rows and drives read, declared once as fields (:mod:`.field_group`).

:data:`JointTable` is the Warp struct of the model's joint topology, drive gains and limits, and the solver's
articulation maps; :data:`FreeBodies` that of the free rigid bodies, their articulations and speed limits. The
step's coordinates and drive targets stay separate arguments, since launches read different ones (the input
state, the predicted velocity, the prescaled velocity).
"""

import warp as wp

from . import field_group
from .field_group import Field, FieldGroup
from .tree_passes import DYNAMICS

JOINTS = (
    *FieldGroup(*DYNAMICS).select(
        "articulation_start",
        "joint_type",
        "joint_child",
        "joint_q_start",
        "joint_qd_start",
        "joint_dof_dim",
        "body_flags",
        "articulation_dof_start",
    ),
    Field("joint_target_q_start", wp.int32, ("joint_bound",), owner="model"),
    Field("joint_target_ke", wp.float32, ("dof",), attr="_joint_target_ke"),
    Field("joint_target_kd", wp.float32, ("dof",), attr="_joint_target_kd"),
    Field("joint_effort_limit", wp.float32, ("dof",), owner="model"),
    Field("joint_velocity_limit", wp.float32, ("dof",), attr="_joint_velocity_limit"),
    Field("joint_limit_lower", wp.float32, ("dof",), attr="_joint_limit_lower"),
    Field("joint_limit_upper", wp.float32, ("dof",), attr="_joint_limit_upper"),
    Field("articulation_H_rows", wp.int32, ("articulation",)),
    Field("art_to_world", wp.int32, ("articulation",)),
)
"""The model's joint topology and effort limits, the drive gains and velocity and position limits the solver's rows
read (its kinematic state's effective arrays: none on a prescribed coordinate), and the solver's articulation
coordinate starts, coordinate counts and worlds."""

JointTable = field_group.struct("JointTable", JOINTS, __name__)


def bind(solver, data=None):
    """Bind the model's and the solver's joint table into a :data:`JointTable` (``data``: an instance to refill)."""
    return field_group.bind(JointTable, data, model=solver.model, solver=solver)


FREE_BODIES = (
    Field("free_rigid_body_indices", wp.int32, ("free_body",)),
    *FieldGroup(*DYNAMICS).select("body_to_articulation", "is_free_rigid"),
    FieldGroup(*JOINTS)["art_to_world"],
    Field("max_linear_velocity", wp.float32, ("body",), attr="rigid_body_max_linear_velocity"),
    Field("max_angular_velocity", wp.float32, ("body",), attr="rigid_body_max_angular_velocity"),
)
"""Each free rigid body, every body's articulation, whether an articulation is a free rigid body, each
articulation's world, and every body's linear and angular speed limit."""

FreeBodies = field_group.struct("FreeBodies", FREE_BODIES, __name__)


def bind_free_bodies(solver):
    """Bind the solver's free-body table into a :data:`FreeBodies`."""
    return field_group.bind(FreeBodies, solver=solver)
