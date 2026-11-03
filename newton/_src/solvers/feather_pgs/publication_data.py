# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The model, solver, state and cache arrays a cached-dynamics owner publishes through."""

import warp as wp

from . import field_group
from .field_group import Field
from .spatial_laws import inertia_record

PUBLICATION = (
    Field("joint_type", wp.int32, ("joint",), owner="model"),
    Field("joint_parent", wp.int32, ("joint",), owner="model"),
    Field("joint_child", wp.int32, ("joint",), owner="model"),
    Field("joint_q_start", wp.int32, ("joint",), owner="model"),
    Field("joint_qd_start", wp.int32, ("joint",), owner="model"),
    Field("joint_dof_dim", wp.int32, ("joint", 2), owner="model"),
    Field("joint_axis", wp.vec3, ("dof",), owner="model"),
    Field("joint_X_p", wp.transform, ("joint",), owner="model"),
    Field("joint_X_c", wp.transform, ("joint",), owner="model"),
    Field("kinematic_dof_mask", wp.int32, ("dof",), owner="factor"),
    Field("kinematic_joint_mask", wp.int32, ("joint",), owner="factor"),
    Field("free_root_joint_indices", wp.int32, ("free_root",), owner="forest"),
    Field("body_X_com", wp.transform, ("body",), owner="factor"),
    Field("body_com", wp.vec3, ("body",), owner="model"),
    Field("body_mass", wp.float32, ("body",), owner="model"),
    Field("body_inertia", wp.mat33, ("body",), owner="model"),
    Field("body_to_articulation", wp.int32, ("body",), owner="forest"),
    Field("is_free_rigid", wp.int32, ("articulation",), owner="forest"),
    Field("gravity", wp.vec3, ("world",), owner="model"),
    Field("joint_q", wp.float32, ("coord",), owner="state_in"),
    Field("joint_qd", wp.float32, ("dof",), owner="state_in"),
    Field("v_out", wp.float32, ("dof",)),
    Field("joint_qdd", wp.float32, ("dof",), owner="state_aug"),
    Field("joint_q_new", wp.float32, ("coord",), owner="state_out", attr="joint_q"),
    Field("joint_qd_new", wp.float32, ("dof",), owner="state_out", attr="joint_qd"),
    Field("body_q", wp.transform, ("body",), owner="state_out"),
    Field("body_qd", wp.spatial_vector, ("body",), owner="state_out"),
    Field("body_q_com", wp.transform, ("body",), owner="state_aug"),
    Field("joint_S_s", wp.spatial_vector, ("dof",), owner="state_aug"),
    Field("articulation_origin", wp.vec3, ("articulation",), owner="factor"),
    # The walk's step geometry (tree_factor.WalkTrees.now, .now_link; empty where nothing walks): each coordinate's
    # column at its joint's frame where a column has two halves (else the walk reads joint_S_s), each joint's frame
    # point relative to its articulation's origin (tree_factor.WALK_LINK floats).
    Field("walk_now", wp.spatial_vector, ("dof",), owner="factor"),
    Field("walk_now_link", wp.float32, ("joint", "link"), owner="factor"),
    # The drives' K-free mass, ``H_ii + R_ii`` (tree_kinetic_state.SATURATE_DRIVES), which the collection writes where it
    # forms ``H_ii``: each coordinate's drive row (-1: none), the rows' masses and each coordinate's armature ``R``.
    Field("drive_row", wp.int32, ("dof",), owner="factor", attr="drive_row_by_dof"),
    Field("row_H", wp.float32, ("drive_row",), owner="factor", attr="aug_row_H"),
    Field("armature", wp.float32, ("dof",), owner="factor", attr="dof_armature"),
    Field("body_v_s", wp.spatial_vector, ("body",), owner="state_aug"),
    Field("body_a_s", wp.spatial_vector, ("body",), owner="state_aug"),
    Field("body_f_s", wp.spatial_vector, ("body",), owner="state_aug"),
    Field("body_I_s", wp.spatial_matrix, ("body",), owner="state_aug"),
    Field("body_inertia_terms", inertia_record, ("body",), owner="factor"),
    Field("fk_id_cache_valid", wp.int32, ("articulation",), owner="factor"),
    Field("dt", wp.float32, owner="step"),
    # Per-group stage cycle counts when FEATHER_PGS_STAGE_CLOCKS=1 (a 1x1 placeholder otherwise).
    Field("stage_clocks", wp.int64, ("group", "stage"), owner="step"),
)
"""The model, solver, state and cache arrays a cached-dynamics owner publishes through: the model's by name, the
solver's through its attributes, the step's states (``state_in``, ``state_aug``, ``state_out``) and the step's
values."""

PublicationData = field_group.struct("PublicationData", PUBLICATION, __name__)


# Body flags are live inputs, not plan fields: the kernels read the kinematic
# masks and effective armature derived from them, which the solver re-derives
# in place on a body-property notification. Their structural consequence, the
# prescribed-response articulations, is fixed at construction, and the solver
# rejects flags that contradict it.
PLAN_FIELDS = (
    "articulation_start",
    "joint_type",
    "joint_parent",
    "joint_child",
    "joint_q_start",
    "joint_qd_start",
    "joint_dof_dim",
)
