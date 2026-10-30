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
    Field("joint_target_ke", wp.float32, ("dof",), owner="model"),
    Field("joint_target_kd", wp.float32, ("dof",), owner="model"),
    Field("joint_effort_limit", wp.float32, ("dof",), owner="model"),
    Field("joint_velocity_limit", wp.float32, ("dof",), owner="model"),
    Field("joint_limit_lower", wp.float32, ("dof",), owner="model"),
    Field("joint_limit_upper", wp.float32, ("dof",), owner="model"),
    Field("prescribed", wp.int32, ("dof",), owner="factor", attr="kinematic_dof_mask"),
    Field("articulation_H_rows", wp.int32, ("articulation",), owner="forest", attr="articulation_dof_count"),
    Field("art_to_world", wp.int32, ("articulation",), owner="forest"),
)
"""The model's joint topology, drive gains, effort, velocity and position limits (its live arrays, so a write
reaches the rows without a notification), the tree factor's prescribed coordinates, and the forest plan's
articulation coordinate starts, coordinate counts and worlds.

A prescribed coordinate (``prescribed``, the body flags' kinematic coordinate mask) takes no row on it alone: no
position limit, velocity limit or drive. Its 1e10 armature leaves such a row a response of about 1e-10 against its
cfm and no fixed point, as its impulse grows with every sweep. Every decision that reads a limit or a drive gain
skips it; rows that couple it with dynamic coordinates (contacts, mimic, connect) stay."""

JointTable = field_group.struct("JointTable", JOINTS, __name__)


def bind(solver, data=None):
    """Bind the model's and the solver's joint table into a :data:`JointTable` (``data``: an instance to refill)."""
    return field_group.bind(JointTable, data, model=solver.model, factor=solver.factor, forest=solver.forest)


FREE_BODIES = (
    Field("free_rigid_body_indices", wp.int32, ("free_body",), owner="forest"),
    *FieldGroup(*DYNAMICS).select("body_to_articulation", "is_free_rigid"),
    FieldGroup(*JOINTS)["art_to_world"],
    Field("max_linear_velocity", wp.float32, ("body",), owner="laws", attr="rigid_body_max_linear_velocity"),
    Field("max_angular_velocity", wp.float32, ("body",), owner="laws", attr="rigid_body_max_angular_velocity"),
)
"""Each free rigid body, every body's articulation, whether an articulation is a free rigid body, each
articulation's world, and every body's linear and angular speed limit."""

FreeBodies = field_group.struct("FreeBodies", FREE_BODIES, __name__)


def bind_free_bodies(solver):
    """Bind the solver's free-body table into a :data:`FreeBodies`."""
    return field_group.bind(FreeBodies, solver=solver, forest=solver.forest, laws=solver.laws)
