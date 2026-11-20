# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The joint table that rows and drives read, declared once as fields (:mod:`.field_group`).

:data:`JointTable` is the Warp struct of the model's joint topology, drive gains and limits, and the solver's
articulation maps. The step's coordinates and drive targets stay separate arguments, since launches read different
ones (the input state, the predicted velocity).
"""

import warp as wp

from . import field_group
from .field_group import Field

JOINTS = (
    Field("articulation_start", wp.int32, ("articulation_bound",), owner="model"),
    Field("joint_type", wp.int32, ("joint",), owner="model"),
    Field("joint_child", wp.int32, ("joint",), owner="model"),
    Field("joint_q_start", wp.int32, ("joint_bound",), owner="model"),
    Field("joint_qd_start", wp.int32, ("joint_bound",), owner="model"),
    Field("joint_dof_dim", wp.int32, ("joint", 2), owner="model"),
    Field("body_flags", wp.int32, ("body",), owner="model"),
    Field("articulation_dof_start", wp.int32, ("articulation",), owner="forest"),
    Field("joint_target_q_start", wp.int32, ("joint_bound",), owner="model"),
    Field("joint_target_ke", wp.float32, ("dof",), owner="model"),
    Field("joint_target_kd", wp.float32, ("dof",), owner="model"),
    Field("joint_effort_limit", wp.float32, ("dof",), owner="model"),
    Field("joint_limit_lower", wp.float32, ("dof",), owner="model"),
    Field("joint_limit_upper", wp.float32, ("dof",), owner="model"),
    Field("prescribed", wp.int32, ("dof",), owner="factor", attr="kinematic_dof_mask"),
    Field("articulation_H_rows", wp.int32, ("articulation",), owner="forest", attr="articulation_dof_count"),
    Field("art_to_world", wp.int32, ("articulation",), owner="forest"),
)
"""The model's joint topology, drive gains, effort and position limits (its live arrays, so a write
reaches the rows without a notification), the tree factor's prescribed coordinates, and the forest plan's
articulation coordinate starts, coordinate counts and worlds.

A prescribed coordinate (``prescribed``, the body flags' kinematic coordinate mask) takes no row on it alone: no
position limit or drive. Its 1e10 armature leaves such a row a response of about 1e-10 against its
cfm and no fixed point, as its impulse grows with every sweep. Every decision that reads a limit or a drive gain
skips it; rows that couple it with dynamic coordinates (contacts, mimic, links) stay."""

JointTable = field_group.struct("JointTable", JOINTS, __name__)


def bind(solver, data=None):
    """Bind the model's and the solver's joint table into a :data:`JointTable` (``data``: an instance to refill)."""
    return field_group.bind(JointTable, data, model=solver.model, factor=solver.factor, forest=solver.forest)
