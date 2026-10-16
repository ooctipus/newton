# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The model, solver, state and cache arrays a cached-dynamics owner publishes through."""

import warp as wp


@wp.struct
class PublicationData:
    joint_type: wp.array[int]
    joint_parent: wp.array[int]
    joint_child: wp.array[int]
    joint_q_start: wp.array[int]
    joint_qd_start: wp.array[int]
    joint_dof_dim: wp.array2d[int]
    joint_axis: wp.array[wp.vec3]
    joint_X_p: wp.array[wp.transform]
    joint_X_c: wp.array[wp.transform]
    kinematic_dof_mask: wp.array[int]
    kinematic_joint_mask: wp.array[int]
    free_root_joint_indices: wp.array[int]
    body_X_com: wp.array[wp.transform]
    body_com: wp.array[wp.vec3]
    body_mass: wp.array[float]
    body_inertia: wp.array[wp.mat33]
    body_to_articulation: wp.array[int]
    is_free_rigid: wp.array[int]
    gravity: wp.array[wp.vec3]
    joint_q: wp.array[float]
    joint_qd: wp.array[float]
    v_out: wp.array[float]
    joint_qdd: wp.array[float]
    joint_q_new: wp.array[float]
    joint_qd_new: wp.array[float]
    body_q: wp.array[wp.transform]
    body_qd: wp.array[wp.spatial_vector]
    body_q_com: wp.array[wp.transform]
    joint_S_s: wp.array[wp.spatial_vector]
    articulation_origin: wp.array[wp.vec3]
    body_v_s: wp.array[wp.spatial_vector]
    body_a_s: wp.array[wp.spatial_vector]
    body_f_s: wp.array[wp.spatial_vector]
    body_I_s: wp.array[wp.spatial_matrix]
    body_inertia_terms: wp.array2d[float]
    fk_id_cache_valid: wp.array[int]
    dt: float
    angular_damping: float
    materialize_all_body_inertia: int
    materialize_body_inertia_terms: int


# Model arrays bound by name, solver arrays bound through their attribute,
# state-cache arrays taken from the augmented state, and the model plan
# fields an owner freezes into its kernels.
MODEL_FIELDS = (
    "joint_type",
    "joint_parent",
    "joint_child",
    "joint_q_start",
    "joint_qd_start",
    "joint_dof_dim",
    "joint_axis",
    "joint_X_p",
    "joint_X_c",
    "body_com",
    "body_mass",
    "body_inertia",
    "gravity",
)
SOLVER_FIELDS = {
    "kinematic_dof_mask": "_kinematic_dof_mask",
    "kinematic_joint_mask": "_kinematic_joint_mask",
    "free_root_joint_indices": "_free_root_joint_indices",
    "body_X_com": "body_X_com",
    "body_to_articulation": "body_to_articulation",
    "is_free_rigid": "is_free_rigid",
    "v_out": "v_out",
    "fk_id_cache_valid": "_fk_id_cache_valid",
}
CACHE_FIELDS = (
    "body_q_com",
    "joint_S_s",
    "articulation_origin",
    "body_v_s",
    "body_a_s",
    "body_f_s",
    "body_I_s",
    "body_inertia_terms",
)
PLAN_FIELDS = (
    "articulation_start",
    "joint_type",
    "joint_parent",
    "joint_child",
    "joint_q_start",
    "joint_qd_start",
    "joint_dof_dim",
    "body_flags",
)
