# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The general route's dynamics passes as tree-scan configurations: FK, ID, tau, composite inertia and CRBA.

One thread per articulation, in Warp (differentiable), as the serial route
runs today. Each pass is three steps over the articulation's tree joints: a
map that forms every node's own value, one :func:`.tree_scan.serial_scan`, and a
map that consumes the results. Values live in the body-indexed arrays the
solver already holds; a joint's slot is its child body and its parent slot its
parent body (``joint_parent``).

* :data:`fk_kinematics`: relative poses ``X_p X_j X_c^-1``, the TRANSFORM scan,
  COM poses and the articulation origin; then each joint's relative twist about
  the origin, the TWIST scan (zero own acceleration: the velocity-product pass).
* :data:`fk_id`: the same, then each body's inverse-dynamics wrench.
* :data:`inverse_dynamics_tau`: each body's net wrench less its external one,
  the ADD scan up, then the joint forces.
* :data:`composite_inertia`: the ADD scan up of the 36 inertia channels.
* :data:`crba_fill`: a PATH from each column's joint, its joint read from the
  plan's coordinate-to-node table instead of a search.

The per-node laws are the solver's own (``kernels``); only the traversal is the
scan's.
"""

import functools

import warp as wp

from .device_preludes import cuda_native, lane_group
from .kernels import (
    _compute_body_net_wrench,
    assemble_com_spatial_inertia,
    jcalc_motion,
    jcalc_tau,
    mul_com_spatial_inertia,
    spatial_cross_dual,
    transform_com_inertia_terms,
)
from .spatial_laws import joint_transform
from .tree_plan import TreePlan
from .tree_scan import TRANSFORM, TWIST, Direction, Group, PlanAccess, Schedule, Storage, add, serial_scan, tree_scan

# The serial scans of a joint range need no tree literal: a parent-first plan stands for every range.
_SERIAL_PLAN = TreePlan.build([-1])
_POSES = serial_scan(TRANSFORM, Direction.DOWN)
_TWISTS = serial_scan(TWIST, Direction.DOWN)
_SUBTREE_SUM = serial_scan(add(Group("value", 1)), Direction.UP)


@wp.func
def _relative_poses(
    start: int,
    end: int,
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q: wp.array[float],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_X_com: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
):
    """World poses of the tree's bodies: each joint's pose in its parent, then the pose scan."""
    for i in range(start, end):
        X_j = joint_transform(
            joint_type[i],
            joint_axis,
            joint_qd_start[i],
            joint_dof_dim[i, 0],
            joint_dof_dim[i, 1],
            joint_q,
            joint_q_start[i],
        )
        body_q[joint_child[i]] = joint_X_p[i] * X_j * wp.transform_inverse(joint_X_c[i])
    _POSES(start, end, joint_parent, joint_child, body_q)
    for i in range(start, end):
        child = joint_child[i]
        body_q_com[child] = body_q[child] * body_X_com[child]


@wp.func
def _twists(
    start: int,
    end: int,
    origin: wp.vec3,
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_qd: wp.array[float],
    joint_X_p: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_q: wp.array[wp.transform],
    joint_S_s: wp.array[wp.spatial_vector],
    body_v_s: wp.array[wp.spatial_vector],
    body_a_s: wp.array[wp.spatial_vector],
):
    """Body twists and velocity-product accelerations about the origin: relative twists, then the twist scan."""
    for i in range(start, end):
        parent = joint_parent[i]
        anchor = joint_X_p[i]
        if parent >= 0:
            anchor = body_q[parent] * anchor
        local = wp.transform(wp.transform_get_translation(anchor) - origin, wp.transform_get_rotation(anchor))
        child = joint_child[i]
        body_v_s[child] = jcalc_motion(
            joint_type[i],
            joint_axis,
            joint_dof_dim[i, 0],
            joint_dof_dim[i, 1],
            local,
            joint_qd,
            joint_qd_start[i],
            joint_S_s,
        )
        body_a_s[child] = wp.spatial_vector()
    _TWISTS(start, end, joint_parent, joint_child, body_v_s, body_a_s)


@wp.func
def _origin(
    start: int,
    articulation_end: int,
    joint_child: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
):
    """The articulation's origin: its first root's center of mass."""
    origin = wp.vec3()
    if start < articulation_end:
        root_body = joint_child[start]
        if root_body >= 0:
            origin = wp.transform_point(body_q[root_body], body_com[root_body])
    return origin


@wp.kernel(enable_backward=False)
def fk_kinematics(
    articulation_start: wp.array[int],
    articulation_joint_end: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_X_com: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_com: wp.array[wp.vec3],
    # outputs
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    articulation_origin: wp.array[wp.vec3],
    joint_S_s: wp.array[wp.spatial_vector],
    body_v_s: wp.array[wp.spatial_vector],
    body_a_s: wp.array[wp.spatial_vector],
    fk_id_cache_valid: wp.array[int],
):
    """``kernels.eval_rigid_fk_kinematics`` as two scans: poses, then twists about the origin."""
    index = wp.tid()
    start = articulation_start[index]
    end = articulation_joint_end[index]
    _relative_poses(
        start,
        end,
        joint_type,
        joint_parent,
        joint_child,
        joint_q_start,
        joint_qd_start,
        joint_q,
        joint_X_p,
        joint_X_c,
        body_X_com,
        joint_axis,
        joint_dof_dim,
        body_q,
        body_q_com,
    )
    origin = _origin(start, articulation_start[index + 1], joint_child, body_q, body_com)
    articulation_origin[index] = origin
    _twists(
        start,
        end,
        origin,
        joint_type,
        joint_parent,
        joint_child,
        joint_qd_start,
        joint_qd,
        joint_X_p,
        joint_axis,
        joint_dof_dim,
        body_q,
        joint_S_s,
        body_v_s,
        body_a_s,
    )
    fk_id_cache_valid[index] = 1


@wp.kernel(enable_backward=False)
def fk_id(
    articulation_start: wp.array[int],
    articulation_joint_end: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_X_com: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    is_free_rigid: wp.array[int],
    materialize_all_body_inertia: int,
    materialize_body_inertia_terms: int,
    reuse_cached: int,
    gravity: wp.array[wp.vec3],
    # outputs
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    articulation_origin: wp.array[wp.vec3],
    joint_S_s: wp.array[wp.spatial_vector],
    body_I_s: wp.array[wp.spatial_matrix],
    body_inertia_terms: wp.array2d[float],
    body_v_s: wp.array[wp.spatial_vector],
    body_f_s: wp.array[wp.spatial_vector],
    body_a_s: wp.array[wp.spatial_vector],
    fk_id_cache_valid: wp.array[int],
):
    """``kernels.eval_rigid_fk_id`` as two scans and a body map: the inverse-dynamics wrench of every body."""
    index = wp.tid()
    if reuse_cached != 0 and fk_id_cache_valid[index] != 0:
        return
    start = articulation_start[index]
    end = articulation_joint_end[index]
    _relative_poses(
        start,
        end,
        joint_type,
        joint_parent,
        joint_child,
        joint_q_start,
        joint_qd_start,
        joint_q,
        joint_X_p,
        joint_X_c,
        body_X_com,
        joint_axis,
        joint_dof_dim,
        body_q,
        body_q_com,
    )
    origin = _origin(start, articulation_start[index + 1], joint_child, body_q, body_com)
    articulation_origin[index] = origin
    _twists(
        start,
        end,
        origin,
        joint_type,
        joint_parent,
        joint_child,
        joint_qd_start,
        joint_qd,
        joint_X_p,
        joint_axis,
        joint_dof_dim,
        body_q,
        joint_S_s,
        body_v_s,
        body_a_s,
    )
    write_body_inertia = materialize_all_body_inertia
    if is_free_rigid[index] != 0:
        write_body_inertia = 1
    gravity_s = gravity[0]
    for i in range(start, end):
        child = joint_child[i]
        X_sm = body_q_com[child]
        X_sm_local = wp.transform(wp.transform_get_translation(X_sm) - origin, wp.transform_get_rotation(X_sm))
        mass = body_mass[child]
        f_g = mass * gravity_s
        com, inertia_origin = transform_com_inertia_terms(X_sm_local, mass, body_inertia[child])
        f_g_s = wp.spatial_vector(f_g, wp.cross(com, f_g))
        if write_body_inertia != 0:
            body_I_s[child] = assemble_com_spatial_inertia(mass, com, inertia_origin)
        if materialize_body_inertia_terms != 0:
            body_inertia_terms[child, 0] = com[0]
            body_inertia_terms[child, 1] = com[1]
            body_inertia_terms[child, 2] = com[2]
            for row in range(3):
                for col in range(3):
                    body_inertia_terms[child, 3 + 3 * row + col] = inertia_origin[row, col]
        v_s = body_v_s[child]
        coriolis = spatial_cross_dual(v_s, mul_com_spatial_inertia(mass, com, inertia_origin, v_s))
        f_b_s = mul_com_spatial_inertia(mass, com, inertia_origin, body_a_s[child]) + coriolis
        body_f_s[child] = f_b_s - f_g_s
    fk_id_cache_valid[index] = 1


@wp.kernel(enable_backward=False)
def inverse_dynamics_tau(
    articulation_start: wp.array[int],
    articulation_joint_end: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_articulation: wp.array[int],
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
    body_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    articulation_origin: wp.array[wp.vec3],
    add_existing_tau: int,
    # outputs
    body_ft_s: wp.array[wp.spatial_vector],
    tau: wp.array[float],
):
    """``kernels.eval_rigid_tau`` as one scan up: each body's subtree wrench, then the joint forces.

    ``body_ft_s`` holds each body's subtree wrench (the existing kernel leaves its children's sum there).
    """
    index = wp.tid()
    start = articulation_start[index]
    end = articulation_joint_end[index]
    for i in range(start, end):
        child = joint_child[i]
        articulation = joint_articulation[i]
        origin = wp.vec3()
        if articulation >= 0:
            origin = articulation_origin[articulation]
        body_ft_s[child] = _compute_body_net_wrench(
            child, wp.spatial_vector(), origin, body_fb_s, body_f_ext, body_flags, body_q, body_com
        )
    _SUBTREE_SUM(start, end, joint_parent, joint_child, body_ft_s)
    for i in range(start, end):
        jcalc_tau(
            joint_type[i],
            joint_S_s,
            joint_f,
            joint_q,
            joint_qd,
            joint_spring_stiffness,
            joint_spring_ref,
            joint_damping,
            joint_q_start[i],
            joint_qd_start[i],
            joint_dof_dim[i, 0],
            joint_dof_dim[i, 1],
            body_ft_s[joint_child[i]],
            add_existing_tau,
            tau,
        )


@wp.kernel(enable_backward=False)
def composite_inertia(
    articulation_start: wp.array[int],
    articulation_joint_end: wp.array[int],
    mass_update_mask: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    body_I_s: wp.array[wp.spatial_matrix],
    # outputs
    body_I_c: wp.array[wp.spatial_matrix],
):
    """``kernels.compute_composite_inertia`` as one scan up of the 36 inertia channels."""
    index = wp.tid()
    if mass_update_mask[index] == 0:
        return
    start = articulation_start[index]
    end = articulation_joint_end[index]
    for i in range(start, end):
        body_I_c[joint_child[i]] = body_I_s[joint_child[i]]
    _SUBTREE_SUM(start, end, joint_parent, joint_child, body_I_c)


@wp.kernel(enable_backward=False)
def crba_fill(
    articulation_dof_start: wp.array[int],
    mass_update_mask: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    articulation_start: wp.array[int],
    dof_joint: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    body_I_c: wp.array[wp.spatial_matrix],
    group_to_art: wp.array[int],
    n_dofs: int,
    fused_augmented_drive: int,
    drive_row_by_dof: wp.array[int],
    row_K: wp.array[float],
    # outputs
    H_group: wp.array3d[float],
):
    """``kernels.crba_fill_par_dof`` as a path from each column's joint, read from the plan's coordinate table."""
    tid = wp.tid()
    group = tid // n_dofs
    col = tid % n_dofs
    art = group_to_art[group]
    if mass_update_mask[art] == 0:
        return
    first_dof = articulation_dof_start[art]
    column_dof = first_dof + col
    joint_start = articulation_start[art]
    pivot = dof_joint[column_dof]
    force = body_I_c[joint_child[pivot]] * joint_S_s[column_dof]
    ancestor = pivot
    while ancestor != -1:
        if ancestor < joint_start:
            break
        q_start = joint_qd_start[ancestor]
        q_dim = joint_dof_dim[ancestor]
        local = q_start - first_dof
        for k in range(q_dim[0] + q_dim[1]):
            row = local + k
            value = wp.dot(joint_S_s[q_start + k], force)
            if fused_augmented_drive != 0 and row == col:
                drive_row = drive_row_by_dof[column_dof]
                if drive_row >= 0:
                    K = row_K[drive_row]
                    if K > 0.0:
                        value += K
            H_group[group, row, col] = value
            H_group[group, col, row] = value
        ancestor = joint_ancestor[ancestor]


# ---------------------------------------------------------------------------------------------------------------------
# Composite inertia, one warp per articulation (the solver's CUDA form)
# ---------------------------------------------------------------------------------------------------------------------

_DENSE_INERTIA = add(Group("inertia", 36))
_ARTICULATION_JOINTS = PlanAccess(
    parent="(joint_ancestor.data[start+({node})]>=start?joint_ancestor.data[start+({node})]-start:-1)"
)
_BODY_INERTIA = Storage({"inertia": "reinterpret_cast<float*>(&body_I_c.data[joint_child.data[start+({node})]])[{k}]"})

_COMPACT_INITIALIZE = """
        const float mass = body_mass.data[body];
        const float* terms = &body_inertia_terms.data[body * 12];
        float* dst = reinterpret_cast<float*>(&body_I_c.data[body]);
        for (int element = LANE; element < 36; element += STRIDE) {
            const int row = element / 6;
            const int col = element - row * 6;
            float value = 0.0f;
            if (row < 3 && col < 3) {
                if (row == col) value = mass;
            } else if (row >= 3 && col >= 3) {
                value = terms[3 + (row - 3) * 3 + col - 3];
            } else {
                const int cross_row = row < 3 ? row : row - 3;
                const int cross_col = col < 3 ? col : col - 3;
                float cross = 0.0f;
                if (cross_row == 0 && cross_col == 1) cross = -terms[2];
                if (cross_row == 0 && cross_col == 2) cross = terms[1];
                if (cross_row == 1 && cross_col == 0) cross = terms[2];
                if (cross_row == 1 && cross_col == 2) cross = -terms[0];
                if (cross_row == 2 && cross_col == 0) cross = -terms[1];
                if (cross_row == 2 && cross_col == 1) cross = terms[0];
                value = (row < 3 ? -mass : mass) * cross;
            }
            dst[element] = value;
        }
"""
_DENSE_INITIALIZE = """
        const float* src = reinterpret_cast<const float*>(&body_I_s.data[body]);
        float* dst = reinterpret_cast<float*>(&body_I_c.data[body]);
        for (int element = LANE; element < 36; element += STRIDE) dst[element] = src[element];
"""


@functools.cache
def composite_inertia_lanes(warps_per_block: int, compact_terms: bool = False) -> "wp.Kernel":
    """The solver's warp-per-articulation composite inertia as a scan configuration: SERIAL up, lanes over
    the 36 channels (``solver_feather_pgs._get_composite_inertia_warp_kernel``, same signature and launch)."""
    scan = tree_scan(
        _SERIAL_PLAN,
        _DENSE_INERTIA,
        Direction.UP,
        Schedule.SERIAL,
        storage=_BODY_INERTIA,
        access=_ARTICULATION_JOINTS,
        sizes={"nodes": "(end-start)"},
        channel_lanes=True,
    )
    snippet = f"""    const int candidate = block * {warps_per_block} + SUB;
    if (candidate >= composite_articulation_count) return;
    const int articulation = composite_articulations.data[candidate];
    const int start = articulation_start.data[articulation];
    const int end = articulation_joint_end.data[articulation];
    for (int joint = start; joint < end; ++joint) {{
        const int body = joint_child.data[joint];
{_COMPACT_INITIALIZE if compact_terms else _DENSE_INITIALIZE}
    }}
    SYNC();
{scan.cuda}#endif
"""
    snippet = cuda_native(snippet, preludes=(lane_group(32),))
    if compact_terms:

        @wp.func_native(snippet)
        def native(
            block: int,
            composite_articulation_count: int,
            composite_articulations: wp.array[int],
            articulation_start: wp.array[int],
            articulation_joint_end: wp.array[int],
            joint_ancestor: wp.array[int],
            joint_child: wp.array[int],
            body_mass: wp.array[float],
            body_inertia_terms: wp.array2d[float],
            body_I_c: wp.array[wp.spatial_matrix],
        ): ...

        def kernel(
            composite_articulation_count: int,
            composite_articulations: wp.array[int],
            articulation_start: wp.array[int],
            articulation_joint_end: wp.array[int],
            joint_ancestor: wp.array[int],
            joint_child: wp.array[int],
            body_mass: wp.array[float],
            body_inertia_terms: wp.array2d[float],
            body_I_c: wp.array[wp.spatial_matrix],
        ):
            block, _lane = wp.tid()
            native(
                block,
                composite_articulation_count,
                composite_articulations,
                articulation_start,
                articulation_joint_end,
                joint_ancestor,
                joint_child,
                body_mass,
                body_inertia_terms,
                body_I_c,
            )

    else:

        @wp.func_native(snippet)
        def native(
            block: int,
            composite_articulation_count: int,
            composite_articulations: wp.array[int],
            articulation_start: wp.array[int],
            articulation_joint_end: wp.array[int],
            joint_ancestor: wp.array[int],
            joint_child: wp.array[int],
            body_I_s: wp.array[wp.spatial_matrix],
            body_I_c: wp.array[wp.spatial_matrix],
        ): ...

        def kernel(
            composite_articulation_count: int,
            composite_articulations: wp.array[int],
            articulation_start: wp.array[int],
            articulation_joint_end: wp.array[int],
            joint_ancestor: wp.array[int],
            joint_child: wp.array[int],
            body_I_s: wp.array[wp.spatial_matrix],
            body_I_c: wp.array[wp.spatial_matrix],
        ):
            block, _lane = wp.tid()
            native(
                block,
                composite_articulation_count,
                composite_articulations,
                articulation_start,
                articulation_joint_end,
                joint_ancestor,
                joint_child,
                body_I_s,
                body_I_c,
            )

    name = f"tree_scan_composite_inertia_warp{warps_per_block}" + ("_compact" if compact_terms else "")
    kernel.__name__ = kernel.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(kernel)
