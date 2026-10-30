# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The joint and free-body tables that rows and drives read, declared once as fields (:mod:`.field_group`).

:data:`JointTable` is the Warp struct of the model's joint topology, drive gains and limits, and the solver's
articulation maps; :data:`FreeBodies` that of the free rigid bodies, their articulations and speed limits. The
step's coordinates and drive targets stay separate arguments, since launches read different ones (the input
state, the predicted velocity, the prescaled velocity).
"""

import numpy as np
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
    Field("joint_target_ke", wp.float32, ("dof",), owner="laws", attr="_joint_target_ke"),
    Field("joint_target_kd", wp.float32, ("dof",), owner="laws", attr="_joint_target_kd"),
    Field("joint_effort_limit", wp.float32, ("dof",), owner="model"),
    Field("joint_velocity_limit", wp.float32, ("dof",), owner="laws", attr="_joint_velocity_limit"),
    Field("joint_limit_lower", wp.float32, ("dof",), owner="laws", attr="_joint_limit_lower"),
    Field("joint_limit_upper", wp.float32, ("dof",), owner="laws", attr="_joint_limit_upper"),
    Field("articulation_H_rows", wp.int32, ("articulation",), owner="forest", attr="articulation_dof_count"),
    Field("art_to_world", wp.int32, ("articulation",), owner="forest"),
)
"""The model's joint topology and effort limits, the drive gains and velocity and position limits the solver's rows
read (the effective row data it publishes every step, :func:`publish_effective`: none on a prescribed coordinate), and
the solver's articulation coordinate starts, coordinate counts and worlds."""

JointTable = field_group.struct("JointTable", JOINTS, __name__)

EFFECTIVE = ("joint_limit_lower", "joint_limit_upper", "joint_velocity_limit", "joint_target_ke", "joint_target_kd")
"""The per-coordinate row data the rows read: the model's arrays, but none on a prescribed (kinematic) coordinate
(:func:`publish_effective`). The solver holds them as private arrays (``_name``)."""


@wp.kernel
def _publish_effective_kernel(
    prescribed: wp.array[wp.int32],
    lower: wp.array[wp.float32],
    upper: wp.array[wp.float32],
    velocity_limit: wp.array[wp.float32],
    ke: wp.array[wp.float32],
    kd: wp.array[wp.float32],
    effective_lower: wp.array[wp.float32],
    effective_upper: wp.array[wp.float32],
    effective_velocity_limit: wp.array[wp.float32],
    effective_ke: wp.array[wp.float32],
    effective_kd: wp.array[wp.float32],
):
    dof = wp.tid()
    if prescribed[dof] != 0:
        # Infinite position limits, no velocity limit (a non-positive limit is none,
        # row_laws.velocity_limit_active) and no drive: no row on this coordinate alone.
        effective_lower[dof] = float(-wp.inf)
        effective_upper[dof] = float(wp.inf)
        effective_velocity_limit[dof] = 0.0
        effective_ke[dof] = 0.0
        effective_kd[dof] = 0.0
    else:
        effective_lower[dof] = lower[dof]
        effective_upper[dof] = upper[dof]
        effective_velocity_limit[dof] = velocity_limit[dof]
        effective_ke[dof] = ke[dof]
        effective_kd[dof] = kd[dof]


def publish_effective(model, prescribed: wp.array, effective: dict[str, wp.array]) -> None:
    """Write the effective row data of :data:`EFFECTIVE` into ``effective`` (by name) from the model's live arrays.

    A prescribed (kinematic) coordinate is outside the solver's unknowns: a row supported on it alone (a position or
    velocity limit, a PhysX drive) has a response of the 1e10 armature's size against its cfm and no fixed point, as
    its impulse grows with every sweep. Where ``prescribed`` (the kinematic coordinate mask) is set, the position
    limits are therefore infinite and the velocity limit and drive gains zero (unlimited, undriven); rows that couple
    it with dynamic coordinates (contacts, mimic, connect) stay. Elsewhere the model's values pass through, so the
    solver, which publishes every step, reads a write to the model without a notification.
    """
    if not model.joint_dof_count:
        return
    wp.launch(
        _publish_effective_kernel,
        dim=model.joint_dof_count,
        inputs=[prescribed, *(getattr(model, name) for name in EFFECTIVE)],
        outputs=[effective[name] for name in EFFECTIVE],
        device=model.device,
    )


def effective(model, prescribed=None) -> dict[str, wp.array]:
    """New arrays of the effective row data of :data:`EFFECTIVE`, published once from the model (with ``prescribed``,
    a coordinate mask, else no prescribed coordinate): what a joint table bound from a model rather than from a solver
    takes."""
    if prescribed is None:
        prescribed = wp.zeros(model.joint_dof_count, dtype=wp.int32, device=model.device)
    elif not isinstance(prescribed, wp.array):
        prescribed = wp.array(np.asarray(prescribed, dtype=np.int32), dtype=wp.int32, device=model.device)
    data = {name: wp.empty_like(getattr(model, name)) for name in EFFECTIVE}
    publish_effective(model, prescribed, data)
    return data


def bind(solver, data=None):
    """Bind the model's and the solver's joint table into a :data:`JointTable` (``data``: an instance to refill)."""
    return field_group.bind(JointTable, data, model=solver.model, solver=solver, forest=solver.forest, laws=solver.laws)


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
