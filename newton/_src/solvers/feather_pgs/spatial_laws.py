# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The spatial kit: one twist convention and one inertia representation, for Warp and native CUDA alike.

Every law is written once in the subset of Warp's language that
:mod:`.law_source` renders, so Warp kernels call the Warp function and native
tree scans call ``Spatial::law(...)``.

**Twists and wrenches.** A spatial vector keeps Newton's layout, linear part on
top: a twist ``(v, w)`` holds the velocity ``v`` of the material point at the
frame's reference point and the angular velocity ``w``; a wrench ``(f, t)``
holds the force and the moment about that point. Tree quantities share one
reference point per articulation, its origin (the root's center of mass), and
world-aligned axes:

* :func:`point_velocity` is the velocity of a point at ``offset`` from the
  reference point, ``v + w x offset`` (a body's public COM twist, a contact
  point's velocity);
* :func:`wrench_shift` moves a wrench's reference point by ``offset``:
  ``(f, t + offset x f)`` (a COM wrench to the origin);
* :func:`twist_transform` changes a twist's frame, as Newton's
  ``transform_twist``;
* :func:`twist_compose` composes a parent's twist and its bias acceleration
  with a joint's relative twist and acceleration, the combine of the motion
  scan; :func:`transform_compose` is the pose scan's;
* :func:`free_root_transport` is the transport acceleration ``w x v`` that a
  free root's linear coordinate (a velocity of a point fixed in the moving
  body) carries.

**Inertia.** A rigid body's inertia about the origin is ``(mass, moment,
rotational)``: the mass ``m``, the first moment ``h = m c`` of its center of
mass ``c`` relative to the origin, and the rotational inertia about the
origin, ``R I R^T + m (|c|^2 1 - c c^T)``. The three are linear in the body,
so a subtree's composite inertia is the channel-wise sum of its bodies' (13
channels, the tree scans' add combine). :func:`inertia_at` places a body's
inertia, :func:`inertia_apply` maps a twist to the momentum wrench,
:func:`inertia_matrix` packs the dense 6x6 matrix for consumers that read one
and :func:`body_wrench` is a body's inverse-dynamics wrench, Newton-Euler at
its center of mass moved to the origin.

**Joints.** :func:`joint_transform` is a joint's transform for every joint
type, :func:`axis_transform` its one-axis case and :func:`motion_axis` one
column of the motion subspace about the origin.

The arithmetic of each law is the form of the held-tree kernels, the solver's
hot path; ``tests/test_feather_pgs_spatial_laws.py`` records how far each other
site's spelling sits from it.
"""

import warp as wp

from ...sim import JointType
from ...sim.articulation import (
    compute_2d_rotational_dofs,
    compute_3d_rotational_dofs,
    transform_2d_rotational_axes,
    transform_3d_rotational_axes,
)
from .device_preludes import DevicePrelude
from .law_source import LawFamily

SPATIAL = LawFamily("Spatial")
"""The spatial kit's family; native code calls ``Spatial::point_velocity(...)``."""


# ---------------------------------------------------------------------------------------------------------------------
# Composition (the tree scans' combines)
# ---------------------------------------------------------------------------------------------------------------------


@SPATIAL.law
def transform_compose(parent: wp.transform, child: wp.transform) -> wp.transform:
    """Return the pose ``parent * child``: a child's pose in its parent's frame, carried into the parent's frame."""
    return wp.transform_multiply(parent, child)


@SPATIAL.law
def joint_frame_pose(parent_anchor: wp.transform, motion: wp.transform, child_anchor: wp.transform) -> wp.transform:
    """Return a joint's child pose in its parent body's frame: ``parent_anchor * motion * child_anchor^-1``.

    ``parent_anchor`` is the joint frame in the parent body, ``motion`` the joint's transform and
    ``child_anchor`` the joint frame in the child body; the pose scan composes the result with the parent's
    pose (:func:`transform_compose`).
    """
    return wp.transform_multiply(wp.transform_multiply(parent_anchor, motion), wp.transform_inverse(child_anchor))


@SPATIAL.law
def twist_compose(
    parent_velocity: wp.spatial_vector,
    parent_acceleration: wp.spatial_vector,
    velocity: wp.spatial_vector,
    acceleration: wp.spatial_vector,
) -> tuple[wp.spatial_vector, wp.spatial_vector]:
    """Return the twist and bias acceleration of a child from its parent's and its own relative ones.

    The twist adds; the acceleration adds plus the velocity product of the
    parent twist with the relative twist (the motion cross product
    ``parent_velocity x velocity``). The composition is associative, so a scan
    may compose path segments in any grouping.
    """
    parent_linear = wp.spatial_top(parent_velocity)
    parent_angular = wp.spatial_bottom(parent_velocity)
    linear = wp.spatial_top(velocity)
    angular = wp.spatial_bottom(velocity)
    product_linear = wp.cross(parent_angular, linear) + wp.cross(parent_linear, angular)
    product_angular = wp.cross(parent_angular, angular)
    bias_linear = wp.spatial_top(parent_acceleration) + wp.spatial_top(acceleration) + product_linear
    bias_angular = wp.spatial_bottom(parent_acceleration) + wp.spatial_bottom(acceleration) + product_angular
    return parent_velocity + velocity, wp.spatial_vector(bias_linear, bias_angular)


# ---------------------------------------------------------------------------------------------------------------------
# Twists and wrenches
# ---------------------------------------------------------------------------------------------------------------------


@SPATIAL.law
def twist_transform(frame: wp.transform, twist: wp.spatial_vector) -> wp.spatial_vector:
    """Return ``twist`` carried by the rigid ``frame``: ``w' = R w``, ``v' = R v + p x w'`` (Newton's layout)."""
    rotation = wp.transform_get_rotation(frame)
    angular = wp.quat_rotate(rotation, wp.spatial_bottom(twist))
    linear = wp.quat_rotate(rotation, wp.spatial_top(twist)) + wp.cross(wp.transform_get_translation(frame), angular)
    return wp.spatial_vector(linear, angular)


@SPATIAL.law
def point_velocity(twist: wp.spatial_vector, offset: wp.vec3) -> wp.vec3:
    """Return the velocity of the point at ``offset`` from the twist's reference point: ``v + w x offset``."""
    return wp.spatial_top(twist) + wp.cross(wp.spatial_bottom(twist), offset)


@SPATIAL.law
def wrench_shift(wrench: wp.spatial_vector, offset: wp.vec3) -> wp.spatial_vector:
    """Return ``wrench`` about the point at ``-offset``: the force keeps, the moment gains ``offset x force``.

    ``offset`` runs from the new reference point to the old one, as a center of
    mass's offset from the origin moves a COM wrench to the origin.
    """
    force = wp.spatial_top(wrench)
    return wp.spatial_vector(force, wp.spatial_bottom(wrench) + wp.cross(offset, force))


@SPATIAL.law
def free_root_transport(linear: wp.vec3, angular: wp.vec3) -> wp.vec3:
    """Return the transport acceleration ``angular x linear`` of a free root's linear coordinate.

    A free root's linear coordinate is the velocity of a point fixed in the
    rotating body, so it changes by this term at constant body momentum. The
    integrator realizes it, the predictor adds it and the velocity-to-acceleration
    conversion removes it, all from this law.
    """
    return wp.cross(angular, linear)


# ---------------------------------------------------------------------------------------------------------------------
# Inertia: (mass, first moment, rotational inertia) about the origin
# ---------------------------------------------------------------------------------------------------------------------


@SPATIAL.law
def inertia_shift(mass: float, rotational: wp.mat33, offset: wp.vec3) -> tuple[float, wp.vec3, wp.mat33]:
    """Return the inertia about the origin of a body whose center of mass lies at ``offset`` from it.

    ``rotational`` is the body's rotational inertia about its center of mass
    in the origin's axes. The result is ``(mass, mass * offset, rotational +
    mass (|offset|^2 1 - offset offset^T))``, the parallel-axis law entry by
    entry.
    """
    square = wp.dot(offset, offset)
    x = offset[0]
    y = offset[1]
    z = offset[2]
    shifted = wp.mat33(
        rotational[0, 0] + mass * (square - x * x),
        rotational[0, 1] + mass * (0.0 - x * y),
        rotational[0, 2] + mass * (0.0 - x * z),
        rotational[1, 0] + mass * (0.0 - y * x),
        rotational[1, 1] + mass * (square - y * y),
        rotational[1, 2] + mass * (0.0 - y * z),
        rotational[2, 0] + mass * (0.0 - z * x),
        rotational[2, 1] + mass * (0.0 - z * y),
        rotational[2, 2] + mass * (square - z * z),
    )
    return mass, mass * offset, shifted


@SPATIAL.law
def inertia_at(mass: float, inertia: wp.mat33, rotation: wp.quat, offset: wp.vec3) -> tuple[float, wp.vec3, wp.mat33]:
    """Return a body's inertia about the origin: ``(mass, mass * offset, rotational)``.

    ``inertia`` is the body-frame inertia about the center of mass, ``rotation``
    the body's orientation and ``offset`` its center of mass relative to the
    origin: the rotated inertia ``R I R^T`` shifted to the origin
    (:func:`inertia_shift`).
    """
    basis = wp.quat_to_matrix(rotation)
    total, moment, rotational = inertia_shift(mass, basis * inertia * wp.transpose(basis), offset)
    return total, moment, rotational


@SPATIAL.law
def inertia_apply(mass: float, moment: wp.vec3, rotational: wp.mat33, twist: wp.spatial_vector) -> wp.spatial_vector:
    """Return the momentum wrench of ``twist`` under the inertia ``(mass, moment, rotational)``.

    ``(m v + w x h, h x v + I w)``: the force is the mass times the velocity of
    the center of mass, the moment is the angular momentum about the origin.
    """
    linear = wp.spatial_top(twist)
    angular = wp.spatial_bottom(twist)
    force = mass * linear + wp.cross(angular, moment)
    lever = wp.cross(moment, linear)
    torque = wp.vec3(
        lever[0] + rotational[0, 0] * angular[0] + rotational[0, 1] * angular[1] + rotational[0, 2] * angular[2],
        lever[1] + rotational[1, 0] * angular[0] + rotational[1, 1] * angular[1] + rotational[1, 2] * angular[2],
        lever[2] + rotational[2, 0] * angular[0] + rotational[2, 1] * angular[1] + rotational[2, 2] * angular[2],
    )
    return wp.spatial_vector(force, torque)


@SPATIAL.law(cuda=False)
def inertia_matrix(mass: float, moment: wp.vec3, rotational: wp.mat33) -> wp.spatial_matrix:
    """Return the dense 6x6 spatial inertia of ``(mass, moment, rotational)`` in Newton's layout.

    ``[[m 1, -[h]x], [[h]x, I]]``, for consumers that read a matrix. No native
    consumer packs one, so the law has no CUDA form.
    """
    lever = wp.skew(moment)
    # fmt: off
    return wp.spatial_matrix(
        mass, 0.0, 0.0, -lever[0, 0], -lever[0, 1], -lever[0, 2],
        0.0, mass, 0.0, -lever[1, 0], -lever[1, 1], -lever[1, 2],
        0.0, 0.0, mass, -lever[2, 0], -lever[2, 1], -lever[2, 2],
        lever[0, 0], lever[0, 1], lever[0, 2], rotational[0, 0], rotational[0, 1], rotational[0, 2],
        lever[1, 0], lever[1, 1], lever[1, 2], rotational[1, 0], rotational[1, 1], rotational[1, 2],
        lever[2, 0], lever[2, 1], lever[2, 2], rotational[2, 0], rotational[2, 1], rotational[2, 2],
    )
    # fmt: on


@SPATIAL.law
def inertia_action(rotation: wp.quat, inertia: wp.mat33, vector: wp.vec3) -> wp.vec3:
    """Return a body-frame ``inertia`` applied to a world-frame ``vector``: ``R (I (R^T vector))``."""
    return wp.quat_rotate(rotation, inertia * wp.quat_rotate_inv(rotation, vector))


@SPATIAL.law
def body_wrench(
    mass: float,
    rotation: wp.quat,
    inertia: wp.mat33,
    offset: wp.vec3,
    velocity: wp.spatial_vector,
    acceleration: wp.spatial_vector,
    gravity: wp.vec3,
) -> wp.spatial_vector:
    """Return a body's inverse-dynamics wrench about the origin: its momentum rate at its twist ``velocity``
    and bias ``acceleration``, less its weight.

    ``inertia`` is the body-frame inertia about the center of mass, ``rotation``
    the body's orientation and ``offset`` its center of mass relative to the
    origin. Newton-Euler at the center of mass, ``f = m (a_c + w x v_c - g)``
    and ``t_c = I alpha + w x (I w)``, with ``v_c`` and ``a_c`` the velocity and
    acceleration of the center of mass (:func:`point_velocity`), moved to the
    origin (:func:`wrench_shift`).
    """
    angular = wp.spatial_bottom(velocity)
    velocity_com = point_velocity(velocity, offset)
    force = mass * (point_velocity(acceleration, offset) + wp.cross(angular, velocity_com) - gravity)
    torque = inertia_action(rotation, inertia, wp.spatial_bottom(acceleration)) + wp.cross(
        angular, inertia_action(rotation, inertia, angular)
    )
    return wrench_shift(wp.spatial_vector(force, torque), offset)


INERTIA_CHANNELS = 13
"""Channels of an inertia record: mass, first moment (3), rotational inertia about the origin (9, row-major)."""

inertia_record = wp.types.vector(length=INERTIA_CHANNELS, dtype=wp.float32)
"""A body's or a subtree's inertia about the origin as one value: channel-wise sums add inertias."""


@wp.func
def unpack_inertia(record: inertia_record):
    """Return the ``(mass, moment, rotational)`` of an inertia record."""
    moment = wp.vec3(record[1], record[2], record[3])
    rotational = wp.mat33(
        record[4], record[5], record[6], record[7], record[8], record[9], record[10], record[11], record[12]
    )
    return record[0], moment, rotational


@wp.func
def pack_inertia(mass: float, moment: wp.vec3, rotational: wp.mat33) -> inertia_record:
    """Return the inertia record of ``(mass, moment, rotational)``."""
    record = inertia_record()
    record[0] = mass
    for k in range(3):
        record[1 + k] = moment[k]
    for i in range(3):
        for j in range(3):
            record[4 + 3 * i + j] = rotational[i, j]
    return record


_INERTIA_RECORDS = """    struct InertiaRecord {
        // An inertia record: mass, first moment, rotational inertia about the origin (row-major), 13 floats.
        static CUDA_CALLABLE inline wp::spatial_vector apply(const float* r, const wp::spatial_vector& twist) {
            return Spatial::inertia_apply(r[0], wp::vec3(r[1], r[2], r[3]),
                wp::mat33(r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12]), twist);
        }
    };"""

INERTIA_RECORDS = DevicePrelude("inertia_records", _INERTIA_RECORDS, cpu=_INERTIA_RECORDS)
"""Native reader of an inertia record, ``InertiaRecord::apply(record, twist)``, after the kit's prelude."""


# ---------------------------------------------------------------------------------------------------------------------
# Joints
# ---------------------------------------------------------------------------------------------------------------------


@SPATIAL.law
def axis_transform(angular: bool, axis: wp.vec3, q: float) -> wp.transform:
    """Return a one-axis joint's transform: a rotation by ``q`` about ``axis``, or a slide of ``q`` along it."""
    if angular:
        return wp.transform(wp.vec3(0.0), wp.quat_from_axis_angle(axis, q))
    return wp.transform(axis * q, wp.quat(0.0, 0.0, 0.0, 1.0))


@SPATIAL.law
def motion_axis(anchor: wp.transform, axis: wp.vec3, angular: bool) -> wp.spatial_vector:
    """Return one motion-subspace column: the unit twist of ``axis`` in the joint ``anchor`` frame, about the origin.

    ``anchor`` is the joint's parent anchor relative to the origin; an angular
    axis rotates about the anchor, a linear one slides along it.
    """
    if angular:
        return twist_transform(anchor, wp.spatial_vector(wp.vec3(0.0), axis))
    return twist_transform(anchor, wp.spatial_vector(axis, wp.vec3(0.0)))


@SPATIAL.law(cuda=False)
def joint_transform(
    joint_type: int,
    joint_axis: wp.array[wp.vec3],
    axis_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    joint_q: wp.array[float],
    q_start: int,
) -> wp.transform:
    """Return a joint's transform ``X_j(q)`` for every joint type.

    The rotations of two- and three-axis D6 joints come from Newton's
    articulation helpers, which have no CUDA form, so neither has this law;
    :func:`axis_transform` is its one-axis case for native code.
    """
    if joint_type == JointType.PRISMATIC:
        return axis_transform(False, joint_axis[axis_start], joint_q[q_start])
    if joint_type == JointType.REVOLUTE:
        return axis_transform(True, joint_axis[axis_start], joint_q[q_start])
    if joint_type == JointType.BALL:
        rotation = wp.quat(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2], joint_q[q_start + 3])
        return wp.transform(wp.vec3(0.0), rotation)
    if joint_type == JointType.FIXED:
        return wp.transform_identity()
    if joint_type == JointType.FREE or joint_type == JointType.DISTANCE:
        position = wp.vec3(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2])
        rotation = wp.quat(joint_q[q_start + 3], joint_q[q_start + 4], joint_q[q_start + 5], joint_q[q_start + 6])
        return wp.transform(position, rotation)
    if joint_type == JointType.D6:
        position = wp.vec3(0.0)
        rotation = wp.quat_identity()
        if lin_axis_count > 0:
            position += joint_axis[axis_start + 0] * joint_q[q_start + 0]
        if lin_axis_count > 1:
            position += joint_axis[axis_start + 1] * joint_q[q_start + 1]
        if lin_axis_count > 2:
            position += joint_axis[axis_start + 2] * joint_q[q_start + 2]
        ia = axis_start + lin_axis_count
        iq = q_start + lin_axis_count
        if ang_axis_count == 1:
            rotation = wp.quat_from_axis_angle(joint_axis[ia], joint_q[iq])
        if ang_axis_count == 2:
            rotation, _ = compute_2d_rotational_dofs(
                joint_axis[ia + 0], joint_axis[ia + 1], joint_q[iq + 0], joint_q[iq + 1], 0.0, 0.0
            )
        if ang_axis_count == 3:
            rotation, _ = compute_3d_rotational_dofs(
                joint_axis[ia + 0],
                joint_axis[ia + 1],
                joint_axis[ia + 2],
                joint_q[iq + 0],
                joint_q[iq + 1],
                joint_q[iq + 2],
                0.0,
                0.0,
                0.0,
            )
        return wp.transform(position, rotation)
    return wp.transform_identity()


@wp.func
def joint_columns(
    joint_type: int,
    joint_axis: wp.array[wp.vec3],
    lin_axis_count: int,
    ang_axis_count: int,
    anchor: wp.transform,
    joint_q: wp.array[float],
    q_start: int,
    qd_start: int,
    columns: wp.array[wp.spatial_vector],
):
    """Store a joint's motion-subspace columns about the point ``anchor`` is relative to in ``columns`` and return them:
    its sliding columns (up to three, a prismatic joint's and a D6 joint's linear axes, a free joint's translations)
    then its turning columns (up to three), zero past the joint's.

    ``anchor`` is the joint's parent anchor relative to that point (the origin for ``S``; the joint's own frame, its
    rotation alone, for the walk's step columns); every column is :func:`motion_axis` of an axis (the kind a constant
    in each branch) but a free joint's, whose coordinates are the root's velocity and angular velocity in its rotated
    frame. A D6 joint is the chain of its axes that :func:`joint_transform` composes: its linear axes slide along the
    anchor's, then each angular axis, carried through the rotations before it, turns about the joint origin the linear
    coordinates moved.
    """
    l0 = wp.spatial_vector()
    l1 = wp.spatial_vector()
    l2 = wp.spatial_vector()
    a0 = wp.spatial_vector()
    a1 = wp.spatial_vector()
    a2 = wp.spatial_vector()
    if joint_type == JointType.PRISMATIC:
        l0 = motion_axis(anchor, joint_axis[qd_start], False)
        columns[qd_start] = l0
    elif joint_type == JointType.REVOLUTE:
        a0 = motion_axis(anchor, joint_axis[qd_start], True)
        columns[qd_start] = a0
    elif joint_type == JointType.D6:
        position = wp.vec3(0.0)
        # Unrolled, each axis kind a constant, as the one-axis branches.
        if lin_axis_count > 0:
            l0 = motion_axis(anchor, joint_axis[qd_start + 0], False)
            columns[qd_start + 0] = l0
            position += joint_axis[qd_start + 0] * joint_q[q_start + 0]
        if lin_axis_count > 1:
            l1 = motion_axis(anchor, joint_axis[qd_start + 1], False)
            columns[qd_start + 1] = l1
            position += joint_axis[qd_start + 1] * joint_q[q_start + 1]
        if lin_axis_count > 2:
            l2 = motion_axis(anchor, joint_axis[qd_start + 2], False)
            columns[qd_start + 2] = l2
            position += joint_axis[qd_start + 2] * joint_q[q_start + 2]
        first = qd_start + lin_axis_count
        angle = q_start + lin_axis_count
        pivot = wp.transform(wp.transform_point(anchor, position), wp.transform_get_rotation(anchor))
        axis_0 = wp.vec3(0.0)
        axis_1 = wp.vec3(0.0)
        axis_2 = wp.vec3(0.0)
        if ang_axis_count == 1:
            axis_0 = joint_axis[first + 0]
        if ang_axis_count == 2:
            axis_0, axis_1 = transform_2d_rotational_axes(joint_axis[first + 0], joint_axis[first + 1], joint_q[angle])
        if ang_axis_count == 3:
            axis_0, axis_1, axis_2 = transform_3d_rotational_axes(
                joint_axis[first + 0],
                joint_axis[first + 1],
                joint_axis[first + 2],
                joint_q[angle + 0],
                joint_q[angle + 1],
            )
        if ang_axis_count > 0:
            a0 = motion_axis(pivot, axis_0, True)
            columns[first + 0] = a0
        if ang_axis_count > 1:
            a1 = motion_axis(pivot, axis_1, True)
            columns[first + 1] = a1
        if ang_axis_count > 2:
            a2 = motion_axis(pivot, axis_2, True)
            columns[first + 2] = a2
    elif joint_type == JointType.BALL:
        a0 = motion_axis(anchor, wp.vec3(1.0, 0.0, 0.0), True)
        a1 = motion_axis(anchor, wp.vec3(0.0, 1.0, 0.0), True)
        a2 = motion_axis(anchor, wp.vec3(0.0, 0.0, 1.0), True)
        columns[qd_start + 0] = a0
        columns[qd_start + 1] = a1
        columns[qd_start + 2] = a2
    elif joint_type == JointType.FREE or joint_type == JointType.DISTANCE:
        rotation = wp.transform_get_rotation(anchor)
        ex = wp.quat_rotate(rotation, wp.vec3(1.0, 0.0, 0.0))
        ey = wp.quat_rotate(rotation, wp.vec3(0.0, 1.0, 0.0))
        ez = wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0))
        l0 = wp.spatial_vector(ex, wp.vec3())
        l1 = wp.spatial_vector(ey, wp.vec3())
        l2 = wp.spatial_vector(ez, wp.vec3())
        a0 = wp.spatial_vector(wp.vec3(), ex)
        a1 = wp.spatial_vector(wp.vec3(), ey)
        a2 = wp.spatial_vector(wp.vec3(), ez)
        columns[qd_start + 0] = l0
        columns[qd_start + 1] = l1
        columns[qd_start + 2] = l2
        columns[qd_start + 3] = a0
        columns[qd_start + 4] = a1
        columns[qd_start + 5] = a2
    return l0, l1, l2, a0, a1, a2


@SPATIAL.law(cuda=False)
def joint_motion(
    joint_type: int,
    joint_axis: wp.array[wp.vec3],
    lin_axis_count: int,
    ang_axis_count: int,
    anchor: wp.transform,
    joint_q: wp.array[float],
    q_start: int,
    joint_qd: wp.array[float],
    qd_start: int,
    joint_S_s: wp.array[wp.spatial_vector],
) -> tuple[wp.spatial_vector, wp.spatial_vector]:
    """Store a joint's motion-subspace columns about the origin in ``joint_S_s`` (:func:`joint_columns`); return its
    twist and its own bias acceleration, formed from the columns as they are stored.

    ``anchor`` is the joint's parent anchor relative to the origin. A free joint's twist is its root's velocity and
    angular velocity in its rotated frame. A D6 joint's twist and own bias acceleration are its chain's, composed by
    :func:`twist_compose`; every other joint's own bias is zero.
    """
    l0, l1, l2, a0, a1, a2 = joint_columns(
        joint_type, joint_axis, lin_axis_count, ang_axis_count, anchor, joint_q, q_start, qd_start, joint_S_s
    )
    bias = wp.spatial_vector()
    if joint_type == JointType.PRISMATIC:
        return l0 * joint_qd[qd_start], bias
    if joint_type == JointType.REVOLUTE:
        return a0 * joint_qd[qd_start], bias
    if joint_type == JointType.D6:
        twist = wp.spatial_vector()
        if lin_axis_count > 0:
            twist += l0 * joint_qd[qd_start + 0]
        if lin_axis_count > 1:
            twist += l1 * joint_qd[qd_start + 1]
        if lin_axis_count > 2:
            twist += l2 * joint_qd[qd_start + 2]
        first = qd_start + lin_axis_count
        if ang_axis_count > 0:
            twist, bias = twist_compose(twist, bias, a0 * joint_qd[first + 0], wp.spatial_vector())
        if ang_axis_count > 1:
            twist, bias = twist_compose(twist, bias, a1 * joint_qd[first + 1], wp.spatial_vector())
        if ang_axis_count > 2:
            twist, bias = twist_compose(twist, bias, a2 * joint_qd[first + 2], wp.spatial_vector())
        return twist, bias
    if joint_type == JointType.BALL:
        return a0 * joint_qd[qd_start + 0] + a1 * joint_qd[qd_start + 1] + a2 * joint_qd[qd_start + 2], bias
    if joint_type == JointType.FREE or joint_type == JointType.DISTANCE:
        rotation = wp.transform_get_rotation(anchor)
        linear = wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2])
        angular = wp.vec3(joint_qd[qd_start + 3], joint_qd[qd_start + 4], joint_qd[qd_start + 5])
        return wp.spatial_vector(wp.quat_rotate(rotation, linear), wp.quat_rotate(rotation, angular)), bias
    return wp.spatial_vector(), bias


@SPATIAL.law
def coordinate_force(
    motion: wp.spatial_vector,
    wrench: wp.spatial_vector,
    force: float,
    stiffness: float,
    reference: float,
    damping: float,
    q: float,
    qd: float,
) -> float:
    """Return one coordinate's generalized force: ``force - S . wrench`` with its passive spring and damper,
    ``stiffness (reference - q) - damping qd``, added last: :func:`spring_damper_force` at ``theta = 0``, whose
    ``K = damping dt`` is in the mass (:func:`drive_stiffness`)."""
    passive = stiffness * (reference - q)
    passive -= damping * qd
    return -wp.dot(motion, wrench) + force + passive


GEOMETRY_REQUEST = wp.constant(1)
"""A ``mass_update_requested`` bit: the articulation's mass and factor are stale."""
FACTOR_REQUEST = wp.constant(2)
"""A ``mass_update_requested`` bit: only its ``K`` changed, so its factor is stale and its geometry is not. The drives set
it while the held tree's repair reads the other bit on another stream, so a repair never races a saturation change."""


@SPATIAL.law
def spring_damper_stiffness(stiffness: float, damping: float, theta: float, dt: float) -> float:
    """Return the ``K = theta k dt^2 + c dt`` a spring and damper add to their coordinate's mass.

    One law holds every joint element: ``(H + K) dv = dt (tau + u0)`` with :func:`spring_damper_force` in ``tau``. The
    damper is always implicit. ``theta`` is physics: at 1 (a drive) the spring is backward Euler, stable at any gain and
    damping an undamped spring at ``omega^2 dt / 2`` per second; at 0 (a passive spring) it is symplectic Euler, which
    keeps the oscillator's energy and is stable to ``omega dt < 2``.
    """
    return theta * stiffness * dt * dt + damping * dt


@wp.func
def spring_damper_force(
    stiffness: float, damping: float, q: float, qd: float, q_target: float, qd_target: float, theta: float, dt: float
) -> float:
    """Return a spring and damper's explicit force ``u0 = -(k (q - q_target + theta dt qd) + c (qd - qd_target))``, the
    force of :func:`spring_damper_stiffness`'s step."""
    return -(stiffness * (q - q_target + theta * dt * qd) + damping * (qd - qd_target))


@wp.func
def drive_stiffness(ke: float, kd: float, damping: float, dt: float) -> float:
    """Return the ``K`` of a coordinate: its augmented drive's (``theta = 1``) and its passive damper's (``theta = 0``,
    :func:`coordinate_force` holds its force)."""
    return spring_damper_stiffness(ke, kd, 1.0, dt) + spring_damper_stiffness(0.0, damping, 0.0, dt)


@wp.func
def drive_force(ke: float, kd: float, q: float, qd: float, q_target: float, qd_target: float, dt: float) -> float:
    """Return an augmented drive's explicit force (:func:`spring_damper_force` at ``theta = 1``), not limited:
    :func:`drive_step_force` reads the step."""
    return spring_damper_force(ke, kd, q, qd, q_target, qd_target, 1.0, dt)


@SPATIAL.law
def limited_drive_force(u0: float, effort_limit: float) -> float:
    """Return ``u0`` clamped to a positive effort limit (the actuator's force range), the force the trial solve holds."""
    if effort_limit > 0.0:
        return wp.clamp(u0, -effort_limit, effort_limit)
    return u0


@SPATIAL.law
def drive_step_force(u0: float, limited: float, K: float, qdd: float, inertia: float) -> float:
    """Return a drive's force on its step, read from a trial solve with every ``K`` in the mass and every force limited.

    An effort limit makes the drive's energy Huber, not quadratic: ``K`` belongs in the mass only while the drive is
    below its limit, and a saturated drive is a constant force, so the joint accelerates at ``limit / I``, not
    ``limit / (I + K)``. Past the limit the drive holds no ``K`` and pushes ``+-limit``; the decision is one step.

    The trial is the coupled step: its ``limited - K qdd`` holds the other drives, ``C``, gravity and ``joint_f``, with no
    drive pushing past what it can. What a drive feels of them is the tree's, not ``H_ii``'s: ``1 / (A^-1)_ii`` falls far
    below ``H_ii`` where two axes align. Its own excess ``u0 - limited`` adds ``I / (I + K)`` of itself, ``inertia`` being
    ``I``, which is what the excess alone would add on one coordinate. So on one coordinate, under any other force,
    this is the exact step's force, and it decides what the limit decides: a stiff drive on a light joint keeps its ``K``
    at every ``u0`` it meets. Solving with every ``u0`` unlimited instead would charge each drive with its neighbours'
    forces past their limits, and saturate the free joint beside a saturated one. ``inertia`` is read only with an excess.
    """
    force = limited - K * qdd
    if u0 != limited:
        force += (u0 - limited) * inertia / (inertia + K)
    return force


@SPATIAL.law
def free_drive_force(
    u0: float, limited: float, K: float, qdd: float, rows_qdd: float, inertia: float, effort_limit: float
) -> float:
    """Return the force of a drive that :func:`drive_step_force` left below its limit, for the solve that follows.

    The excess ``u0 - limited`` is what the exact step asks of a free joint, but the trial solves no rows, and a joint
    blocked by a contact or a limit moves nowhere: its force is its whole ``u0``. So the excess comes back only as far as
    it stays within the limit on the step as last step's rows left it, ``qdd + rows_qdd``. With no rows that is all of it,
    the exact step. Blocked, none: the trial's limited force and ``K`` hold the joint at its limit. ``K`` stays in the
    mass either way, because taking it out on a lagged reading gives a light joint ``limit / I`` beside a contact that
    has left, while a force within ``limited`` and ``u0`` costs at most the limited step. A row that arrives this step
    is read on the next: the drive pushes ``u0`` against it for that one step.
    """
    excess = u0 - limited
    side = wp.sign(excess)
    room = wp.max(effort_limit - side * (limited - K * (qdd + rows_qdd)), 0.0)
    return limited + side * wp.min(wp.abs(excess), room * (inertia + K) / inertia)
