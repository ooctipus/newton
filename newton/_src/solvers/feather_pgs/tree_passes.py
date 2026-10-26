# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The general route's dynamics passes as tree-scan configurations: FK, ID, tau, composite inertia and CRBA.

One thread per articulation, in Warp, as the serial route runs today, and
differentiable: Warp's tape records them when a model requires gradients (the
CUDA composite form, native, is forward only). Each pass is the serial
schedule of :func:`.tree_scan.serial_scan` (or :func:`.tree_scan.serial_path`)
over the articulation's tree joints with the pass's per-node laws as hooks:
the node's own value and the consumer of its result, in the same visit, so
the scan reads and writes the tree once. Values live in the body-indexed
arrays the solver already holds; a joint's slot is its child body and its
parent slot its parent body (``joint_parent``). The arrays the hooks read are
bound into one struct per pass.

* :data:`fk_kinematics`: TRANSFORM down from each joint's pose in its parent,
  ``X_p X_j X_c^-1``, publishing COM poses; then TWIST down from each joint's
  relative twist about the origin (zero own acceleration: the velocity
  product).
* :data:`fk_id`: the same, each body's inverse-dynamics wrench formed as its
  twist completes.
* :data:`inverse_dynamics_tau`: ADD up, each body's net wrench from its
  children's sum, then its joint forces.
* :data:`composite_inertia`: ADD up of the 36 inertia channels, own values
  first; :func:`composite_inertia_lanes` the solver's warp-per-articulation
  form, lanes over the channels.
* :data:`crba_fill`: a PATH (:func:`.tree_scan.serial_path`) from each column's
  joint to the root.

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
from .tree_scan import (
    TRANSFORM,
    TWIST,
    Direction,
    Group,
    PlanAccess,
    Schedule,
    Storage,
    add,
    serial_path,
    serial_scan,
    tree_scan,
)

# The serial scans of a joint range need no tree literal: a parent-first plan stands for every range.
_SERIAL_PLAN = TreePlan.build([-1])
_SUM = add(Group("value", 1))


# ---------------------------------------------------------------------------------------------------------------------
# Kinematics and inverse dynamics: the hooks of the down scans
# ---------------------------------------------------------------------------------------------------------------------


@wp.struct
class _Poses:
    joint_type: wp.array[int]
    joint_child: wp.array[int]
    joint_q_start: wp.array[int]
    joint_qd_start: wp.array[int]
    joint_q: wp.array[float]
    joint_X_p: wp.array[wp.transform]
    joint_X_c: wp.array[wp.transform]
    joint_axis: wp.array[wp.vec3]
    joint_dof_dim: wp.array2d[int]
    body_X_com: wp.array[wp.transform]
    body_q_com: wp.array[wp.transform]


@wp.func
def _relative_pose(node: int, d: _Poses) -> wp.transform:
    """A joint's child pose in its parent body's frame: ``X_p X_j X_c^-1``."""
    X_j = joint_transform(
        d.joint_type[node],
        d.joint_axis,
        d.joint_qd_start[node],
        d.joint_dof_dim[node, 0],
        d.joint_dof_dim[node, 1],
        d.joint_q,
        d.joint_q_start[node],
    )
    return d.joint_X_p[node] * X_j * wp.transform_inverse(d.joint_X_c[node])


@wp.func
def _com_pose(node: int, pose: wp.transform, d: _Poses):
    child = d.joint_child[node]
    d.body_q_com[child] = pose * d.body_X_com[child]


@wp.struct
class _Twists:
    origin: wp.vec3
    gravity: wp.vec3
    write_body_inertia: int
    write_body_inertia_terms: int
    joint_type: wp.array[int]
    joint_parent: wp.array[int]
    joint_child: wp.array[int]
    joint_qd_start: wp.array[int]
    joint_qd: wp.array[float]
    joint_X_p: wp.array[wp.transform]
    joint_axis: wp.array[wp.vec3]
    joint_dof_dim: wp.array2d[int]
    body_q: wp.array[wp.transform]
    body_q_com: wp.array[wp.transform]
    body_mass: wp.array[float]
    body_inertia: wp.array[wp.mat33]
    joint_S_s: wp.array[wp.spatial_vector]
    body_I_s: wp.array[wp.spatial_matrix]
    body_inertia_terms: wp.array2d[float]
    body_f_s: wp.array[wp.spatial_vector]


@wp.func
def _relative_twist(node: int, d: _Twists):
    """A joint's relative twist about the origin, its motion columns stored; no own acceleration."""
    parent = d.joint_parent[node]
    anchor = d.joint_X_p[node]
    if parent >= 0:
        anchor = d.body_q[parent] * anchor
    local = wp.transform(wp.transform_get_translation(anchor) - d.origin, wp.transform_get_rotation(anchor))
    velocity = jcalc_motion(
        d.joint_type[node],
        d.joint_axis,
        d.joint_dof_dim[node, 0],
        d.joint_dof_dim[node, 1],
        local,
        d.joint_qd,
        d.joint_qd_start[node],
        d.joint_S_s,
    )
    return velocity, wp.spatial_vector()


@wp.func
def _body_wrench(node: int, v_s: wp.spatial_vector, a_s: wp.spatial_vector, d: _Twists):
    """A body's inverse-dynamics wrench at the origin, as its twist completes (``kernels.compute_link_velocity``)."""
    child = d.joint_child[node]
    X_sm = d.body_q_com[child]
    X_sm_local = wp.transform(wp.transform_get_translation(X_sm) - d.origin, wp.transform_get_rotation(X_sm))
    mass = d.body_mass[child]
    f_g = mass * d.gravity
    com, inertia_origin = transform_com_inertia_terms(X_sm_local, mass, d.body_inertia[child])
    f_g_s = wp.spatial_vector(f_g, wp.cross(com, f_g))
    if d.write_body_inertia != 0:
        d.body_I_s[child] = assemble_com_spatial_inertia(mass, com, inertia_origin)
    if d.write_body_inertia_terms != 0:
        d.body_inertia_terms[child, 0] = com[0]
        d.body_inertia_terms[child, 1] = com[1]
        d.body_inertia_terms[child, 2] = com[2]
        for row in range(3):
            for col in range(3):
                d.body_inertia_terms[child, 3 + 3 * row + col] = inertia_origin[row, col]
    coriolis = spatial_cross_dual(v_s, mul_com_spatial_inertia(mass, com, inertia_origin, v_s))
    f_b_s = mul_com_spatial_inertia(mass, com, inertia_origin, a_s) + coriolis
    d.body_f_s[child] = f_b_s - f_g_s


_POSES = serial_scan(TRANSFORM, Direction.DOWN, _relative_pose, _com_pose)
_TWISTS = serial_scan(TWIST, Direction.DOWN, _relative_twist)
_TWISTS_AND_WRENCHES = serial_scan(TWIST, Direction.DOWN, _relative_twist, _body_wrench)


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


@wp.func
def _bind_poses(
    joint_type: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q: wp.array[float],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_X_com: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
) -> _Poses:
    d = _Poses()
    d.joint_type = joint_type
    d.joint_child = joint_child
    d.joint_q_start = joint_q_start
    d.joint_qd_start = joint_qd_start
    d.joint_q = joint_q
    d.joint_X_p = joint_X_p
    d.joint_X_c = joint_X_c
    d.joint_axis = joint_axis
    d.joint_dof_dim = joint_dof_dim
    d.body_X_com = body_X_com
    d.body_q_com = body_q_com
    return d


@wp.func
def _bind_twists(
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
) -> _Twists:
    d = _Twists()
    d.origin = origin
    d.joint_type = joint_type
    d.joint_parent = joint_parent
    d.joint_child = joint_child
    d.joint_qd_start = joint_qd_start
    d.joint_qd = joint_qd
    d.joint_X_p = joint_X_p
    d.joint_axis = joint_axis
    d.joint_dof_dim = joint_dof_dim
    d.body_q = body_q
    d.joint_S_s = joint_S_s
    return d


@wp.kernel
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
    """``kernels.eval_rigid_fk_kinematics`` as two serial scans: poses, then twists about the origin."""
    index = wp.tid()
    start = articulation_start[index]
    end = articulation_joint_end[index]
    poses = _bind_poses(
        joint_type,
        joint_child,
        joint_q_start,
        joint_qd_start,
        joint_q,
        joint_X_p,
        joint_X_c,
        joint_axis,
        joint_dof_dim,
        body_X_com,
        body_q_com,
    )
    _POSES(start, end, joint_parent, joint_child, body_q, poses)
    origin = _origin(start, articulation_start[index + 1], joint_child, body_q, body_com)
    articulation_origin[index] = origin
    twists = _bind_twists(
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
    )
    _TWISTS(start, end, joint_parent, joint_child, body_v_s, body_a_s, twists)
    fk_id_cache_valid[index] = 1


@wp.kernel
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
    """``kernels.eval_rigid_fk_id`` as two serial scans, each body's wrench formed as its twist completes."""
    index = wp.tid()
    if reuse_cached != 0 and fk_id_cache_valid[index] != 0:
        return
    start = articulation_start[index]
    end = articulation_joint_end[index]
    poses = _bind_poses(
        joint_type,
        joint_child,
        joint_q_start,
        joint_qd_start,
        joint_q,
        joint_X_p,
        joint_X_c,
        joint_axis,
        joint_dof_dim,
        body_X_com,
        body_q_com,
    )
    _POSES(start, end, joint_parent, joint_child, body_q, poses)
    origin = _origin(start, articulation_start[index + 1], joint_child, body_q, body_com)
    articulation_origin[index] = origin
    twists = _bind_twists(
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
    )
    twists.gravity = gravity[0]
    twists.write_body_inertia = materialize_all_body_inertia
    if is_free_rigid[index] != 0:
        twists.write_body_inertia = 1
    twists.write_body_inertia_terms = materialize_body_inertia_terms
    twists.body_q_com = body_q_com
    twists.body_mass = body_mass
    twists.body_inertia = body_inertia
    twists.body_I_s = body_I_s
    twists.body_inertia_terms = body_inertia_terms
    twists.body_f_s = body_f_s
    _TWISTS_AND_WRENCHES(start, end, joint_parent, joint_child, body_v_s, body_a_s, twists)
    fk_id_cache_valid[index] = 1


# ---------------------------------------------------------------------------------------------------------------------
# Joint forces and composite inertia: the up scans
# ---------------------------------------------------------------------------------------------------------------------


@wp.struct
class _Forces:
    add_existing_tau: int
    joint_type: wp.array[int]
    joint_child: wp.array[int]
    joint_articulation: wp.array[int]
    joint_qd_start: wp.array[int]
    joint_q_start: wp.array[int]
    joint_dof_dim: wp.array2d[int]
    joint_f: wp.array[float]
    joint_q: wp.array[float]
    joint_qd: wp.array[float]
    joint_spring_stiffness: wp.array[float]
    joint_spring_ref: wp.array[float]
    joint_damping: wp.array[float]
    joint_S_s: wp.array[wp.spatial_vector]
    body_fb_s: wp.array[wp.spatial_vector]
    body_f_ext: wp.array[wp.spatial_vector]
    body_flags: wp.array[wp.int32]
    body_q: wp.array[wp.transform]
    body_com: wp.array[wp.vec3]
    articulation_origin: wp.array[wp.vec3]
    tau: wp.array[float]


@wp.func
def _net_wrench(node: int, children: wp.spatial_vector, d: _Forces) -> wp.spatial_vector:
    """A body's wrench on its joint: its own, its subtree's, less its external wrench at the origin."""
    articulation = d.joint_articulation[node]
    origin = wp.vec3()
    if articulation >= 0:
        origin = d.articulation_origin[articulation]
    return _compute_body_net_wrench(
        d.joint_child[node], children, origin, d.body_fb_s, d.body_f_ext, d.body_flags, d.body_q, d.body_com
    )


@wp.func
def _joint_force(node: int, wrench: wp.spatial_vector, d: _Forces):
    jcalc_tau(
        d.joint_type[node],
        d.joint_S_s,
        d.joint_f,
        d.joint_q,
        d.joint_qd,
        d.joint_spring_stiffness,
        d.joint_spring_ref,
        d.joint_damping,
        d.joint_q_start[node],
        d.joint_qd_start[node],
        d.joint_dof_dim[node, 0],
        d.joint_dof_dim[node, 1],
        wrench,
        d.add_existing_tau,
        d.tau,
    )


_JOINT_FORCES = serial_scan(_SUM, Direction.UP, _net_wrench, _joint_force)


@wp.kernel
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
    """``kernels.eval_rigid_tau`` (and ``_add``) as one serial scan up: ``body_ft_s`` holds the children's sums."""
    index = wp.tid()
    d = _Forces()
    d.add_existing_tau = add_existing_tau
    d.joint_type = joint_type
    d.joint_child = joint_child
    d.joint_articulation = joint_articulation
    d.joint_qd_start = joint_qd_start
    d.joint_q_start = joint_q_start
    d.joint_dof_dim = joint_dof_dim
    d.joint_f = joint_f
    d.joint_q = joint_q
    d.joint_qd = joint_qd
    d.joint_spring_stiffness = joint_spring_stiffness
    d.joint_spring_ref = joint_spring_ref
    d.joint_damping = joint_damping
    d.joint_S_s = joint_S_s
    d.body_fb_s = body_fb_s
    d.body_f_ext = body_f_ext
    d.body_flags = body_flags
    d.body_q = body_q
    d.body_com = body_com
    d.articulation_origin = articulation_origin
    d.tau = tau
    _JOINT_FORCES(articulation_start[index], articulation_joint_end[index], joint_parent, joint_child, body_ft_s, d)


@wp.struct
class _Inertias:
    joint_child: wp.array[int]
    body_I_s: wp.array[wp.spatial_matrix]


@wp.func
def _body_inertia(node: int, d: _Inertias) -> wp.spatial_matrix:
    return d.body_I_s[d.joint_child[node]]


_COMPOSITE = serial_scan(_SUM, Direction.UP, _body_inertia, own_first=True)


@wp.kernel
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
    """``kernels.compute_composite_inertia`` as one serial scan up of the 36 inertia channels, own values first."""
    index = wp.tid()
    if mass_update_mask[index] == 0:
        return
    d = _Inertias()
    d.joint_child = joint_child
    d.body_I_s = body_I_s
    _COMPOSITE(articulation_start[index], articulation_joint_end[index], joint_parent, joint_child, body_I_c, d)


@wp.struct
class _Column:
    group: int
    col: int
    first_dof: int
    column_dof: int
    fused_augmented_drive: int
    joint_qd_start: wp.array[int]
    joint_dof_dim: wp.array2d[int]
    joint_S_s: wp.array[wp.spatial_vector]
    drive_row_by_dof: wp.array[int]
    row_K: wp.array[float]
    H_group: wp.array3d[float]


@wp.func
def _mass_entries(ancestor: int, pivot: int, force: wp.spatial_vector, d: _Column):
    """Column ``d.col``'s entries in the rows of ``ancestor``'s coordinates, ``S_row . force``, and their mirrors.

    Each entry is written once: in the column's own joint only the rows up to the column, so of two coordinates
    ``i < j`` of one joint, column ``j`` writes the shared entries with ``S_i . (I S_j)``.
    """
    q_start = d.joint_qd_start[ancestor]
    count = d.joint_dof_dim[ancestor, 0] + d.joint_dof_dim[ancestor, 1]
    local = q_start - d.first_dof
    if ancestor == pivot:
        count = d.col - local + 1
    for k in range(count):
        row = local + k
        value = wp.dot(d.joint_S_s[q_start + k], force)
        if d.fused_augmented_drive != 0 and row == d.col:
            drive_row = d.drive_row_by_dof[d.column_dof]
            if drive_row >= 0:
                K = d.row_K[drive_row]
                if K > 0.0:
                    value += K
        d.H_group[d.group, row, d.col] = value
        d.H_group[d.group, d.col, row] = value


_MASS_COLUMN = serial_path(_mass_entries)


@wp.func
def _coordinate_joint(coordinate: int, first: int, last: int, joint_qd_start: wp.array[int]) -> int:
    """The joint among ``first`` to ``last - 1`` whose coordinate range holds ``coordinate``, or ``-1``.

    A function without an early exit: Warp's tape replays a function's loop, but not a loop in a kernel's own
    body or one left by ``break``, whose result the gradient pass would not see.
    """
    joint = int(-1)
    for j in range(first, last):
        if coordinate >= joint_qd_start[j] and coordinate < joint_qd_start[j + 1]:
            joint = j
    return joint


@wp.kernel
def crba_fill(
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    mass_update_mask: wp.array[int],
    dfs_order: wp.array[int],
    dfs_position: wp.array[int],
    dfs_end: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
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
    """``kernels.crba_fill_par_dof``: a PATH from each column's joint, one thread per (articulation, column).

    Differentiable, where the existing kernel's tape gradient is zero: the gradient pass does not see the
    joint its in-kernel search finds (a loop in the kernel's body, left by ``break``) and replays its pointer
    walk with the last pointer. Here a function searches the articulation's coordinate ranges for the
    column's joint (:func:`_coordinate_joint`), and the walk tests the joint forest's depth-first intervals
    (``dfs_order``, ``dfs_position``, ``dfs_end`` of :class:`.tree_plan.TreePlan`) in place of following
    ``joint_ancestor``: independent loads where the pointer walk waits on each, measured 0.68-0.98 of the
    existing kernel's time (RTX PRO 6000; the full search beat both the early exit and the plan's coordinate
    table). Each entry is written once (:func:`_mass_entries`), where the existing kernel lets both columns of
    a multi-coordinate joint write their shared entries and the schedule decide.
    """
    tid = wp.tid()
    group = tid // n_dofs
    col = tid % n_dofs
    art = group_to_art[group]
    if mass_update_mask[art] == 0:
        return
    first_dof = articulation_dof_start[art]
    column_dof = first_dof + col
    joint_start = articulation_start[art]
    pivot = _coordinate_joint(column_dof, joint_start, articulation_start[art + 1], joint_qd_start)
    if pivot == -1:
        return
    force = body_I_c[joint_child[pivot]] * joint_S_s[column_dof]
    d = _Column()
    d.group = group
    d.col = col
    d.first_dof = first_dof
    d.column_dof = column_dof
    d.fused_augmented_drive = fused_augmented_drive
    d.joint_qd_start = joint_qd_start
    d.joint_dof_dim = joint_dof_dim
    d.joint_S_s = joint_S_s
    d.drive_row_by_dof = drive_row_by_dof
    d.row_K = row_K
    d.H_group = H_group
    _MASS_COLUMN(joint_start, pivot, force, dfs_order, dfs_position, dfs_end, d)


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
