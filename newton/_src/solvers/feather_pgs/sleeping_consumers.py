# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental awake-only consumers using the ordinary FPGS physical laws.

The sleep controller owns admission and current-contact wake-up. These kernels
only retire work for admitted sleeping components; unsupported bodies remain
awake. Original default-off kernels and their launch ABI are unchanged.
"""

from functools import cache

import warp as wp

from ...sim import BodyFlags
from .kernels import finalize_body_dynamics_body, jcalc_integrate, jcalc_tau
from .prismatic_linear_state import _write_linear_body


@wp.kernel(enable_backward=False)
def integrate_awake_joints(
    joint_awake: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    kinematic_joint_mask: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    body_com: wp.array[wp.vec3],
    joint_X_c: wp.array[wp.transform],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_qdd: wp.array[float],
    dt: float,
    angular_damping: float,
    joint_q_new: wp.array[float],
    joint_qd_new: wp.array[float],
):
    """Copy sleeping coordinates across buffers and integrate every awake joint."""
    joint = wp.tid()
    coord_start = joint_q_start[joint]
    dof_start = joint_qd_start[joint]
    if joint_awake[joint] == 0 or kinematic_joint_mask[joint] != 0:
        for coord in range(coord_start, joint_q_start[joint + 1]):
            joint_q_new[coord] = joint_q[coord]
        for dof in range(dof_start, joint_qd_start[joint + 1]):
            joint_qd_new[dof] = joint_qd[dof]
        return
    jcalc_integrate(
        joint_type[joint],
        joint_child[joint],
        body_com,
        joint_X_c[joint],
        joint_q,
        joint_qd,
        joint_qdd,
        coord_start,
        dof_start,
        joint_dof_dim[joint, 0],
        joint_dof_dim[joint, 1],
        dt,
        angular_damping,
        joint_parent[joint],
        joint_q_new,
        joint_qd_new,
    )


@wp.kernel(enable_backward=False)
def finalize_awake_linear_state(
    body_awake: wp.array[int],
    body_q_previous: wp.array[wp.transform],
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
    """Preserve frozen public state without rebuilding sleeping linear dynamics."""
    body = wp.tid()
    if body_awake[body] == 0:
        # The output may be a fresh or older ping-pong buffer. The controller
        # zeroed velocity before the final awake publication at sleep entry.
        body_q[body] = body_q_previous[body]
        body_qd[body] = wp.spatial_vector()
        return
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
def get_awake_direct_tau_kernel(n_dofs: int) -> wp.Kernel:
    """Preserve direct-branch force evaluation for awake components."""
    N = int(n_dofs)
    if N <= 0:
        raise ValueError("direct force group must have a positive DOF count")

    def awake_direct_tau(
        body_awake: wp.array[int],
        group_to_art: wp.array[int],
        dof_joint_offset: wp.array[int],
        articulation_start: wp.array[int],
        joint_type: wp.array[int],
        joint_child: wp.array[int],
        joint_qd_start: wp.array[int],
        joint_q_start: wp.array[int],
        joint_dof_dim: wp.array2d[int],
        joint_f: wp.array[float],
        joint_q: wp.array[float],
        joint_qd: wp.array[float],
        joint_spring_stiffness: wp.array[float],
        joint_spring_ref: wp.array[float],
        joint_damping: wp.array[float],
        joint_S_s: wp.array[wp.spatial_vector],
        body_fb_s: wp.array[wp.spatial_vector],
        body_f_ext: wp.array[wp.spatial_vector],
        body_flags: wp.array[int],
        body_q: wp.array[wp.transform],
        body_com: wp.array[wp.vec3],
        articulation_origin: wp.array[wp.vec3],
        add_existing_tau: int,
        tau: wp.array[float],
    ):
        element = wp.tid()
        group = element // N
        local_dof = element - group * N
        articulation = group_to_art[group]
        joint = articulation_start[articulation] + dof_joint_offset[local_dof]
        child = joint_child[joint]
        if body_awake[child] == 0:
            # Driven components and externally forced components cannot sleep.
            # The frozen equilibrium also replaces its passive/limit response.
            tau[joint_qd_start[joint]] = 0.0
            return
        external_com = wp.spatial_vector()
        if (body_flags[child] & BodyFlags.KINEMATIC) == 0:
            external_com = body_f_ext[child]
        external_force = wp.spatial_bottom(external_com)
        external_torque = wp.spatial_top(external_com)
        com_world = wp.transform_point(body_q[child], body_com[child])
        com_relative = com_world - articulation_origin[articulation]
        external_origin = wp.spatial_vector(
            external_torque,
            external_force + wp.cross(com_relative, external_torque),
        )
        jcalc_tau(
            joint_type[joint],
            joint_S_s,
            joint_f,
            joint_q,
            joint_qd,
            joint_spring_stiffness,
            joint_spring_ref,
            joint_damping,
            joint_q_start[joint],
            joint_qd_start[joint],
            joint_dof_dim[joint, 0],
            joint_dof_dim[joint, 1],
            body_fb_s[child] - external_origin,
            add_existing_tau,
            tau,
        )

    name = f"awake_direct_branch_tau_{N}"
    awake_direct_tau.__name__ = name
    awake_direct_tau.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(awake_direct_tau)
