# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Opt-in current linear state for fixed-root, one-body prismatic branches.

Only the direct-diagonal response owner admits this representation. Ordinary
compact inertia is unused there; exceptional masked inertia refresh remains
the original complete producer. All frames and numerical inputs stay current.
"""

from functools import cache

import numpy as np
import warp as wp

from .kernels import finalize_body_dynamics_body


def configure_linear_state(solver) -> None:
    """Require the existing direct response and complete cache ownership."""
    plan = solver._prismatic_publication
    if not (
        solver._fk_id_cache_enabled
        and solver.pgs_mode == "matrix_free"
        and solver._direct_compact_diagonal_inertia
        and not solver.grouped_dynamics
        and not solver._fused_k1
        and plan is not None
    ):
        raise ValueError("prismatic linear state requires cached matrix-free direct-diagonal prismatic response")
    starts = solver.model.articulation_start.numpy()
    ends = plan.joint_end.numpy()
    admitted = ends != solver.articulation_joint_end.numpy()
    direct = solver._model_plan.response_dof_count == solver._compact_diagonal_mass_size
    if not np.array_equal(admitted, direct) or np.any(ends[admitted] != starts[admitted] + 1):
        raise ValueError("prismatic linear state requires every direct branch to belong to the prismatic plan")
    solver._prismatic_linear_valid = wp.empty_like(solver._fk_id_cache_valid)
    solver._direct_compact_diagonal_inertia_kernel = get_linear_inverse_mass_kernel(solver._compact_diagonal_mass_size)
    solver._prismatic_linear_state = True


@wp.func
def _write_linear_body(
    body: int,
    joint: int,
    joint_parent: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_X_com: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    body_to_articulation: wp.array[int],
    body_mass: wp.array[float],
    articulation_origin: wp.array[wp.vec3],
    gravity: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    joint_S_s: wp.array[wp.spatial_vector],
    body_v_s: wp.array[wp.spatial_vector],
    body_a_s: wp.array[wp.spatial_vector],
    body_f_s: wp.array[wp.spatial_vector],
):
    dof = joint_qd_start[joint]
    X_wpj = body_q[joint_parent[joint]] * joint_X_p[joint]
    axis = joint_axis[dof]
    X_j = wp.transform(axis * joint_q[joint_q_start[joint]], wp.quat_identity())
    X_wc = (X_wpj * X_j) * wp.transform_inverse(joint_X_c[joint])
    X_com = X_wc * body_X_com[body]
    body_q[body] = X_wc
    body_q_com[body] = X_com
    motion = wp.spatial_vector(wp.quat_rotate(wp.transform_get_rotation(X_wpj), axis), wp.vec3())
    joint_S_s[dof] = motion
    body_v_s[body] = motion * joint_qd[dof]
    body_a_s[body] = wp.spatial_vector()
    com = wp.transform_get_translation(X_com) - articulation_origin[body_to_articulation[body]]
    force = body_mass[body] * gravity[0]
    body_f_s[body] = -wp.spatial_vector(force, wp.cross(com, force))


@wp.kernel(enable_backward=False)
def repair_prismatic_linear_state(
    body_joint: wp.array[int],
    previous_valid: wp.array[int],
    joint_parent: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_X_com: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    body_to_articulation: wp.array[int],
    body_mass: wp.array[float],
    articulation_origin: wp.array[wp.vec3],
    gravity: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    joint_S_s: wp.array[wp.spatial_vector],
    body_v_s: wp.array[wp.spatial_vector],
    body_a_s: wp.array[wp.spatial_vector],
    body_f_s: wp.array[wp.spatial_vector],
):
    """Repair cold leaves after their roots using pre-launch cache validity."""
    body = wp.tid()
    joint = body_joint[body]
    if joint < 0:
        return
    if previous_valid[body_to_articulation[body]] != 0:
        return
    _write_linear_body(
        body,
        joint,
        joint_parent,
        joint_q_start,
        joint_qd_start,
        joint_q,
        joint_qd,
        joint_X_p,
        joint_X_c,
        body_X_com,
        joint_axis,
        body_to_articulation,
        body_mass,
        articulation_origin,
        gravity,
        body_q,
        body_q_com,
        joint_S_s,
        body_v_s,
        body_a_s,
        body_f_s,
    )


@wp.kernel(enable_backward=False)
def finalize_prismatic_linear_state(
    body_joint: wp.array[int],
    joint_parent: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_X_com: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_S_s: wp.array[wp.spatial_vector],
    body_to_articulation: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    is_free_rigid: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    materialize_all_body_inertia: int,
    materialize_body_inertia_terms: int,
    gravity: wp.array[wp.vec3],
    body_v_s: wp.array[wp.spatial_vector],
    body_a_s: wp.array[wp.spatial_vector],
    body_I_s: wp.array[wp.spatial_matrix],
    body_inertia_terms: wp.array2d[float],
    body_f_s: wp.array[wp.spatial_vector],
    body_qd: wp.array[wp.spatial_vector],
):
    """Publish admitted linear bodies and retain full dynamics for other bodies."""
    body = wp.tid()
    joint = body_joint[body]
    if joint >= 0:
        _write_linear_body(
            body,
            joint,
            joint_parent,
            joint_q_start,
            joint_qd_start,
            joint_q,
            joint_qd,
            joint_X_p,
            joint_X_c,
            body_X_com,
            joint_axis,
            body_to_articulation,
            body_mass,
            articulation_origin,
            gravity,
            body_q,
            body_q_com,
            joint_S_s,
            body_v_s,
            body_a_s,
            body_f_s,
        )
        body_qd[body] = body_v_s[body]
    else:
        finalize_body_dynamics_body(
            body,
            body_to_articulation,
            body_q,
            body_q_com,
            body_com,
            body_mass,
            body_inertia,
            is_free_rigid,
            articulation_origin,
            materialize_all_body_inertia,
            materialize_body_inertia_terms,
            gravity,
            body_v_s,
            body_a_s,
            body_I_s,
            body_inertia_terms,
            body_f_s,
            body_qd,
        )


@cache
def get_linear_inverse_mass_kernel(n_dofs: int) -> wp.Kernel:
    """Retain the direct-mass ABI while removing its unused inertia projection."""
    N = int(n_dofs)
    if N <= 0:
        raise ValueError("linear direct inertia requires a positive DOF count")

    def compute_linear_inverse_mass(
        articulation_start: wp.array[int],
        articulation_dof_start: wp.array[int],
        mass_update_mask: wp.array[int],
        joint_child: wp.array[int],
        joint_S_s: wp.array[wp.spatial_vector],
        body_mass: wp.array[float],
        body_inertia_terms: wp.array2d[float],
        group_to_art: wp.array[int],
        dof_joint_offset: wp.array[int],
        armature: wp.array2d[float],
        drive_row_by_dof: wp.array[int],
        drive_row_K: wp.array[float],
        diagonal_inverse_mass: wp.array[float],
    ):
        element = wp.tid()
        group = element // N
        dof = element - group * N
        art = group_to_art[group]
        if mass_update_mask[art] == 0:
            return
        global_dof = articulation_dof_start[art] + dof
        joint = articulation_start[art] + dof_joint_offset[dof]
        body = joint_child[joint]
        linear = wp.spatial_top(joint_S_s[global_dof])
        mass = body_mass[body] * wp.dot(linear, linear) + armature[group, dof]
        drive_row = drive_row_by_dof[global_dof]
        if drive_row >= 0:
            stiffness = drive_row_K[drive_row]
            if stiffness > 0.0:
                mass += stiffness
        diagonal_inverse_mass[global_dof] = 1.0 / mass

    name = f"compute_prismatic_linear_inverse_mass_{N}"
    compute_linear_inverse_mass.__name__ = name
    compute_linear_inverse_mass.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(compute_linear_inverse_mass)
