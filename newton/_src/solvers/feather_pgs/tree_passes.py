# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The general route's dynamics passes: FK, inverse dynamics, joint forces, composite inertia and CRBA.

One thread per articulation (CRBA: per articulation and column) in Warp, so the
passes run on the CPU and record on Warp's tape when a model requires gradients.
Each pass is a :mod:`.tree_scan` schedule over the articulation's tree joints
with the pass's per-node laws, all from :mod:`.spatial_laws`, as hooks:

* :data:`fk_id`: TRANSFORM down from each joint's pose in its parent,
  ``X_p X_j X_c^-1``, publishing COM poses; then TWIST down from each joint's
  relative twist about the origin, each body's inertia record and
  inverse-dynamics wrench formed as its twist completes (:func:`body_dynamics`);
* :data:`inverse_dynamics_tau`: ADD up, each body's net wrench from its
  children's sum, then its joint's generalized forces;
* :data:`composite_inertia`: ADD up of the bodies' inertia records
  (:data:`.spatial_laws.inertia_record`, 13 channels);
* :data:`crba_fill`: a PATH from each column's joint to the root.

Every pass reads one struct, :data:`DynamicsData`, generated from the field
declarations :data:`DYNAMICS` and bound on the host from the model, the solver,
the states and the control (:func:`.field_group.bind`). Values live in the
body-indexed arrays the solver holds: a joint's slot is its child body and its
parent slot its parent body (``joint_parent``).
"""

import warp as wp

from ...sim import BodyFlags
from . import field_group
from .field_group import Field
from .spatial_laws import (
    body_wrench,
    inertia_apply,
    inertia_at,
    inertia_matrix,
    inertia_record,
    joint_force,
    joint_frame_pose,
    joint_motion,
    joint_transform,
    pack_inertia,
    point_velocity,
    unpack_inertia,
    wrench_shift,
)
from .tree_scan import TRANSFORM, TWIST, Direction, Group, add, serial_path, serial_scan

__all__ = [
    "DYNAMICS",
    "SERIAL_BLOCK_DIM",
    "DynamicsData",
    "body_dynamics",
    "composite_inertia",
    "crba_fill",
    "fk_id",
    "inverse_dynamics_tau",
    "refresh_body_inertia",
]

# ---------------------------------------------------------------------------------------------------------------------
# The passes' data
# ---------------------------------------------------------------------------------------------------------------------

DYNAMICS = (
    # The model's topology, frames and inertia.
    Field("articulation_start", wp.int32, ("articulation_bound",), owner="model"),
    Field("joint_type", wp.int32, ("joint",), owner="model"),
    Field("joint_parent", wp.int32, ("joint",), owner="model"),
    Field("joint_child", wp.int32, ("joint",), owner="model"),
    Field("joint_articulation", wp.int32, ("joint",), owner="model"),
    Field("joint_q_start", wp.int32, ("joint_bound",), owner="model"),
    Field("joint_qd_start", wp.int32, ("joint_bound",), owner="model"),
    Field("joint_dof_dim", wp.int32, ("joint", 2), owner="model"),
    Field("joint_axis", wp.vec3, ("dof",), owner="model"),
    Field("joint_X_p", wp.transform, ("joint",), owner="model"),
    Field("joint_X_c", wp.transform, ("joint",), owner="model"),
    Field("body_com", wp.vec3, ("body",), owner="model"),
    Field("body_mass", wp.float32, ("body",), owner="model"),
    Field("body_inertia", wp.mat33, ("body",), owner="model"),
    Field("body_flags", wp.int32, ("body",), owner="model"),
    Field("gravity", wp.vec3, ("world",), owner="model"),
    # The solver's tables and derived state.
    Field("articulation_joint_end", wp.int32, ("articulation",), owner="forest"),
    Field("articulation_dof_start", wp.int32, ("articulation",), owner="forest"),
    Field("body_to_articulation", wp.int32, ("body",), owner="forest"),
    Field("body_X_com", wp.transform, ("body",)),
    Field("is_free_rigid", wp.int32, ("articulation",), owner="forest"),
    Field("articulation_origin", wp.vec3, ("articulation",)),
    Field("body_inertia_terms", inertia_record, ("body",), attr="_body_inertia_terms"),
    Field("body_I_c", inertia_record, ("body",)),
    Field("mass_update_mask", wp.int32, ("articulation",)),
    Field("spring_stiffness", wp.float32, ("dof",), attr="_passive_spring_stiffness"),
    Field("spring_reference", wp.float32, ("dof",), attr="_passive_spring_ref"),
    Field("damping", wp.float32, ("dof",), attr="_passive_joint_damping"),
    Field("row_K", wp.float32, ("drive_row",), attr="aug_row_K"),
    Field("dfs_order", wp.int32, ("joint",), attr="_joint_dfs_order"),
    Field("dfs_position", wp.int32, ("joint",), attr="_joint_dfs_position"),
    Field("dfs_end", wp.int32, ("joint",), attr="_joint_dfs_end"),
    Field("up_flags", wp.int32, ("joint",), attr="_joint_up_flags"),
    # The state the passes read coordinates from and publish poses and COM twists to.
    Field("joint_q", wp.float32, ("coord",), owner="state"),
    Field("joint_qd", wp.float32, ("dof",), owner="state"),
    Field("body_q", wp.transform, ("body",), owner="state"),
    Field("body_qd", wp.spatial_vector, ("body",), owner="state"),
    Field("body_f", wp.spatial_vector, ("body",), owner="state"),
    # The augmented state the passes derive.
    Field("body_q_com", wp.transform, ("body",), owner="aug"),
    Field("joint_S_s", wp.spatial_vector, ("dof",), owner="aug"),
    Field("body_I_s", wp.spatial_matrix, ("body",), owner="aug"),
    Field("body_v_s", wp.spatial_vector, ("body",), owner="aug"),
    Field("body_a_s", wp.spatial_vector, ("body",), owner="aug"),
    Field("body_f_s", wp.spatial_vector, ("body",), owner="aug"),
    Field("body_ft_s", wp.spatial_vector, ("body",), owner="aug"),
    Field("tau", wp.float32, ("dof",), owner="aug", attr="joint_tau"),
    Field("joint_f", wp.float32, ("dof",), owner="control"),
    # The launch's switches.
    Field("write_inertia_records", wp.int32, owner="step"),
)
"""Everything the dynamics passes read and write, by owner role: ``model``; the solver (the default);
``state``, whose coordinates the passes read and to which they publish poses and COM twists; ``aug``, the
augmented state; ``control``; ``step``, the launch's switches."""

DynamicsData = field_group.struct("DynamicsData", DYNAMICS, __name__)
"""The dynamics passes' one argument (:func:`.field_group.bind` fills it)."""

# Kernels launched with one block size share a Warp module (a module builds once per block size): the walks of
# one articulation per thread at SERIAL_BLOCK_DIM, the per-body and per-column mass passes at 128; the one-body
# branches' forces (256) keep a module of their own.
_SERIAL = wp.Module(f"{__name__}.serial")
_MASS_DYNAMICS = wp.Module(f"{__name__}.mass_dynamics")

SERIAL_BLOCK_DIM = 4
"""Threads per block of the passes that walk one articulation per thread: FK, FK/ID, the joint forces, the composite
inertia and the drives.

A thread's walk is serial and latency-bound, so these passes run fastest with few threads per warp: on the RTX PRO
6000 (``tools/fpgs_bench/tree_scan_bench.py serial``; 2- to 60-body trees, 4096 to 65536 articulations) four lanes
ran within 5% of the fastest block size of every pass but the 2-body tree's at 65536 articulations, where FK ran at
a fixed 16 (1.1-1.7 times four's time) and the joint forces at 256 (1.2-3.4 times). One or two lanes waste issue slots;
eight lost up to 1.6 times on 13-body trees at 65536 articulations."""


_SUM = add(Group("value", 1))


# ---------------------------------------------------------------------------------------------------------------------
# One body's dynamics: inertia about the origin, inverse-dynamics wrench, COM twist
# ---------------------------------------------------------------------------------------------------------------------


@wp.func
def body_dynamics(
    body: int,
    articulation: int,
    pose: wp.transform,
    pose_com: wp.transform,
    origin: wp.vec3,
    velocity: wp.spatial_vector,
    acceleration: wp.spatial_vector,
    mass: float,
    inertia: wp.mat33,
    com: wp.vec3,
    gravity: wp.vec3,
    write_matrix: int,
    write_record: int,
    publish_twist: int,
    body_I_s: wp.array[wp.spatial_matrix],
    body_inertia_terms: wp.array[inertia_record],
    body_f_s: wp.array[wp.spatial_vector],
    body_qd: wp.array[wp.spatial_vector],
):
    """Store a body's inverse-dynamics wrench about the origin (:func:`.spatial_laws.body_wrench`); optionally
    its inertia there and its COM twist.

    ``write_matrix`` stores the dense spatial inertia, ``write_record`` the inertia record. A body outside an
    articulation (``articulation < 0``) only publishes its COM twist.
    """
    if articulation < 0:
        if publish_twist != 0:
            linear = point_velocity(velocity, wp.transform_point(pose, com))
            body_qd[body] = wp.spatial_vector(linear, wp.spatial_bottom(velocity))
        return
    offset = wp.transform_get_translation(pose_com) - origin
    rotation = wp.transform_get_rotation(pose_com)
    if write_matrix != 0 or write_record != 0:
        body_mass, moment, rotational = inertia_at(mass, inertia, rotation, offset)
        if write_matrix != 0:
            body_I_s[body] = inertia_matrix(body_mass, moment, rotational)
        if write_record != 0:
            body_inertia_terms[body] = pack_inertia(body_mass, moment, rotational)
    body_f_s[body] = body_wrench(mass, rotation, inertia, offset, velocity, acceleration, gravity)
    if publish_twist != 0:
        linear = point_velocity(velocity, wp.transform_point(pose, com) - origin)
        body_qd[body] = wp.spatial_vector(linear, wp.spatial_bottom(velocity))


# ---------------------------------------------------------------------------------------------------------------------
# Kinematics and inverse dynamics: the down scans
# ---------------------------------------------------------------------------------------------------------------------


@wp.func
def relative_pose(node: int, d: DynamicsData) -> wp.transform:
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
    return joint_frame_pose(d.joint_X_p[node], X_j, d.joint_X_c[node])


@wp.func
def com_pose(node: int, pose: wp.transform, d: DynamicsData):
    """Publish the child body's center-of-mass frame with its pose."""
    child = d.joint_child[node]
    d.body_q_com[child] = pose * d.body_X_com[child]


@wp.func
def joint_origin(node: int, d: DynamicsData) -> wp.vec3:
    """The origin of a joint's articulation, its first root's center of mass (published by the pose scan)."""
    articulation = d.joint_articulation[node]
    if articulation < 0:
        return wp.vec3()
    return d.articulation_origin[articulation]


@wp.func
def relative_twist(node: int, d: DynamicsData):
    """A joint's twist about the origin, its motion columns stored; no own acceleration."""
    parent = d.joint_parent[node]
    anchor = d.joint_X_p[node]
    if parent >= 0:
        anchor = d.body_q[parent] * anchor
    local = wp.transform(
        wp.transform_get_translation(anchor) - joint_origin(node, d), wp.transform_get_rotation(anchor)
    )
    twist = joint_motion(
        d.joint_type[node],
        d.joint_axis,
        d.joint_dof_dim[node, 0],
        d.joint_dof_dim[node, 1],
        local,
        d.joint_qd,
        d.joint_qd_start[node],
        d.joint_S_s,
    )
    return twist, wp.spatial_vector()


@wp.func
def joint_body_dynamics(node: int, velocity: wp.spatial_vector, acceleration: wp.spatial_vector, d: DynamicsData):
    """A joint's child body's dynamics as its twist completes (the result hook of :data:`fk_id`)."""
    child = d.joint_child[node]
    articulation = d.joint_articulation[node]
    body_dynamics(
        child,
        articulation,
        d.body_q[child],
        d.body_q_com[child],
        d.articulation_origin[articulation],
        velocity,
        acceleration,
        d.body_mass[child],
        d.body_inertia[child],
        d.body_com[child],
        d.gravity[0],
        d.is_free_rigid[articulation],
        d.write_inertia_records,
        0,
        d.body_I_s,
        d.body_inertia_terms,
        d.body_f_s,
        d.body_qd,
    )


_POSES = serial_scan(TRANSFORM, Direction.DOWN, relative_pose, com_pose)
_TWISTS_AND_DYNAMICS = serial_scan(TWIST, Direction.DOWN, relative_twist, joint_body_dynamics)


@wp.func
def articulation_poses(index: int, d: DynamicsData):
    """Every body pose of articulation ``index`` and its origin, its first root's center of mass."""
    start = d.articulation_start[index]
    _POSES(start, d.articulation_joint_end[index], d.joint_parent, d.joint_child, d.body_q, d)
    origin = wp.vec3()
    if start < d.articulation_start[index + 1]:
        root_body = d.joint_child[start]
        if root_body >= 0:
            origin = wp.transform_point(d.body_q[root_body], d.body_com[root_body])
    d.articulation_origin[index] = origin


@wp.kernel(module=_SERIAL)
def fk_id(d: DynamicsData):
    """Body poses, COM frames, origins, motion columns, twists and bias accelerations, with each body's inertia and
    inverse-dynamics wrench as its twist completes; one thread per articulation."""
    index = wp.tid()
    articulation_poses(index, d)
    start, end = d.articulation_start[index], d.articulation_joint_end[index]
    _TWISTS_AND_DYNAMICS(start, end, d.joint_parent, d.joint_child, d.body_v_s, d.body_a_s, d)


@wp.kernel(module=_MASS_DYNAMICS)
def refresh_body_inertia(d: DynamicsData):
    """Place the bodies of the articulations a reuse step's mass mask selects at their current poses; one thread
    per joint."""
    joint = wp.tid()
    articulation = d.joint_articulation[joint]
    if articulation < 0 or joint >= d.articulation_joint_end[articulation]:
        return
    if d.mass_update_mask[articulation] == 0:
        return
    child = d.joint_child[joint]
    pose_com = d.body_q_com[child]
    offset = wp.transform_get_translation(pose_com) - d.articulation_origin[articulation]
    mass, moment, rotational = inertia_at(
        d.body_mass[child], d.body_inertia[child], wp.transform_get_rotation(pose_com), offset
    )
    if d.is_free_rigid[articulation] != 0:
        d.body_I_s[child] = inertia_matrix(mass, moment, rotational)
    d.body_inertia_terms[child] = pack_inertia(mass, moment, rotational)


# ---------------------------------------------------------------------------------------------------------------------
# Joint forces and composite inertia: the up scans
# ---------------------------------------------------------------------------------------------------------------------


@wp.func
def net_wrench(node: int, children: wp.spatial_vector, d: DynamicsData) -> wp.spatial_vector:
    """A body's wrench on its joint: its own, its subtree's, less its external wrench moved to the origin.

    ``body_f`` holds each body's external wrench at its center of mass; a kinematic body takes none.
    """
    child = d.joint_child[node]
    external = wp.spatial_vector()
    if (d.body_flags[child] & BodyFlags.KINEMATIC) == 0:
        external = d.body_f[child]
    offset = wp.transform_point(d.body_q[child], d.body_com[child]) - joint_origin(node, d)
    return d.body_f_s[child] + children - wrench_shift(external, offset)


@wp.func
def generalized_force(node: int, wrench: wp.spatial_vector, d: DynamicsData):
    """A joint's generalized forces from its subtree wrench, into ``tau``."""
    joint_force(
        d.joint_type[node],
        d.joint_S_s,
        d.joint_f,
        d.joint_q,
        d.joint_qd,
        d.spring_stiffness,
        d.spring_reference,
        d.damping,
        d.joint_q_start[node],
        d.joint_qd_start[node],
        d.joint_dof_dim[node, 0],
        d.joint_dof_dim[node, 1],
        wrench,
        d.tau,
    )


_JOINT_FORCES = serial_scan(_SUM, Direction.UP, net_wrench, generalized_force)


@wp.func
def articulation_tau(index: int, d: DynamicsData):
    """The joint forces of articulation ``index``; ``body_ft_s`` receives each body's children's sum."""
    start, end = d.articulation_start[index], d.articulation_joint_end[index]
    _JOINT_FORCES(start, end, d.joint_parent, d.joint_child, d.up_flags, d.body_ft_s, d)


@wp.kernel(module=_SERIAL)
def inverse_dynamics_tau(d: DynamicsData):
    """Every articulation's joint forces; one thread per articulation."""
    articulation_tau(wp.tid(), d)


@wp.func
def subtree_record(node: int, children: inertia_record, d: DynamicsData) -> inertia_record:
    """A joint's subtree's inertia record: its child body's and its children's."""
    return d.body_inertia_terms[d.joint_child[node]] + children


_COMPOSITE = serial_scan(_SUM, Direction.UP, subtree_record, store_total=True)


@wp.kernel(module=_SERIAL)
def composite_inertia(d: DynamicsData):
    """Every masked articulation's composite inertia records, each body's record plus its subtree's; one thread
    per articulation."""
    index = wp.tid()
    if d.mass_update_mask[index] == 0:
        return
    start, end = d.articulation_start[index], d.articulation_joint_end[index]
    _COMPOSITE(start, end, d.joint_parent, d.joint_child, d.up_flags, d.body_I_c, d)


# ---------------------------------------------------------------------------------------------------------------------
# The joint-space inertia: CRBA
# ---------------------------------------------------------------------------------------------------------------------


@wp.struct
class _Column:
    """One CRBA column: its size group's matrix, its articulation's group row and its coordinates."""

    group: int
    col: int
    first_dof: int
    H_group: wp.array3d[float]


@wp.func
def _mass_entries(ancestor: int, pivot: int, force: wp.spatial_vector, d: DynamicsData, column: _Column):
    """Column ``col``'s entries in the rows of ``ancestor``'s coordinates, ``S_row . force``, and their mirrors.

    Each entry is written once: in the column's own joint only the rows from the column on, so of two coordinates
    ``i < j`` of one joint, the lower column ``i`` writes the shared entries with ``S_j . (I S_i)``.
    """
    q_start = d.joint_qd_start[ancestor]
    count = d.joint_dof_dim[ancestor, 0] + d.joint_dof_dim[ancestor, 1]
    local = q_start - column.first_dof
    first = int(0)
    if ancestor == pivot:
        first = column.col - local
    for k in range(first, count):
        row = local + k
        value = wp.dot(d.joint_S_s[q_start + k], force)
        column.H_group[column.group, row, column.col] = value
        column.H_group[column.group, column.col, row] = value


_MASS_COLUMN = serial_path(_mass_entries)


@wp.func
def coordinate_joint(coordinate: int, first: int, last: int, joint_qd_start: wp.array[int]) -> int:
    """The joint among ``first`` to ``last - 1`` whose coordinate range holds ``coordinate``, or ``-1``.

    A function without an early exit: Warp's tape replays a function's loop, but not a loop in a kernel's own
    body or one left by ``break``, whose result the gradient pass would not see.
    """
    joint = int(-1)
    for j in range(first, last):
        if coordinate >= joint_qd_start[j] and coordinate < joint_qd_start[j + 1]:
            joint = j
    return joint


@wp.func
def column_force(joint: int, coordinate: int, records: wp.array[inertia_record], d: DynamicsData) -> wp.spatial_vector:
    """The momentum wrench of a coordinate's motion column under its joint's child record: ``I S``."""
    mass, moment, rotational = unpack_inertia(records[d.joint_child[joint]])
    return inertia_apply(mass, moment, rotational, d.joint_S_s[coordinate])


@wp.kernel(module=_MASS_DYNAMICS)
def crba_fill(
    d: DynamicsData,
    records: wp.array[inertia_record],
    group_to_art: wp.array[int],
    n_dofs: int,
    H_group: wp.array3d[float],
):
    """A size group's joint-space inertias, one thread per (articulation, column): a PATH from the column's joint.

    ``records`` holds the composite records (or the bodies' own, for one-body articulations). The walk tests
    the joint forest's depth-first intervals in place of following ``joint_ancestor``
    (:func:`.tree_scan.serial_path`) and the column's joint comes from a search without an early exit, so the
    tape differentiates the pass; each entry is written once (:func:`_mass_entries`).
    """
    tid = wp.tid()
    group = tid // n_dofs
    col = tid % n_dofs
    art = group_to_art[group]
    if d.mass_update_mask[art] == 0:
        return
    first_dof = d.articulation_dof_start[art]
    column_dof = first_dof + col
    joint_start = d.articulation_start[art]
    pivot = coordinate_joint(column_dof, joint_start, d.articulation_start[art + 1], d.joint_qd_start)
    if pivot == -1:
        return
    column = _Column()
    column.group = group
    column.col = col
    column.first_dof = first_dof
    column.H_group = H_group
    force = column_force(pivot, column_dof, records, d)
    _MASS_COLUMN(joint_start, pivot, force, d.dfs_order, d.dfs_position, d.dfs_end, d, column)
