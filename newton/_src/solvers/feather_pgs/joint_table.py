# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The joint table that joint rows and joint drives read, declared once as fields (:mod:`.field_group`).

:data:`JointTable` is its Warp struct: the model's joint topology, drive gains and limits, and the solver's
articulation maps. The step's coordinates and drive targets stay separate arguments, since launches read
different ones (the input state, the predicted velocity, the prescaled velocity).
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
    Field("articulation_H_rows", wp.int32, ("articulation",)),
    Field("art_to_world", wp.int32, ("articulation",)),
)
"""The model's joint topology, drive gains, effort, velocity and position limits, and the solver's
articulation coordinate starts, coordinate counts and worlds."""

JointTable = field_group.struct("JointTable", JOINTS, __name__)


def bind(solver, data=None):
    """Bind the model's and the solver's joint table into a :data:`JointTable` (``data``: an instance to refill)."""
    return field_group.bind(JointTable, data, model=solver.model, solver=solver)
