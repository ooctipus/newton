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
from ...sim.articulation import compute_2d_rotational_dofs, compute_3d_rotational_dofs
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
        # Unrolled so the joint transform stays differentiable.
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


@SPATIAL.law(cuda=False)
def joint_motion(
    joint_type: int,
    joint_axis: wp.array[wp.vec3],
    lin_axis_count: int,
    ang_axis_count: int,
    anchor: wp.transform,
    joint_qd: wp.array[float],
    qd_start: int,
    joint_S_s: wp.array[wp.spatial_vector],
) -> wp.spatial_vector:
    """Store a joint's motion-subspace columns about the origin in ``joint_S_s`` and return its twist.

    ``anchor`` is the joint's parent anchor relative to the origin; every column is :func:`motion_axis`
    of an axis (the kind a constant in each branch) but a free joint's, whose coordinates are the root's
    velocity and angular velocity in its rotated frame.
    """
    if joint_type == JointType.PRISMATIC:
        column = motion_axis(anchor, joint_axis[qd_start], False)
        joint_S_s[qd_start] = column
        return column * joint_qd[qd_start]
    if joint_type == JointType.REVOLUTE:
        column = motion_axis(anchor, joint_axis[qd_start], True)
        joint_S_s[qd_start] = column
        return column * joint_qd[qd_start]
    if joint_type == JointType.D6:
        twist = wp.spatial_vector()
        # Unrolled, each axis kind a constant, as the one-axis branches.
        if lin_axis_count > 0:
            column = motion_axis(anchor, joint_axis[qd_start + 0], False)
            twist += column * joint_qd[qd_start + 0]
            joint_S_s[qd_start + 0] = column
        if lin_axis_count > 1:
            column = motion_axis(anchor, joint_axis[qd_start + 1], False)
            twist += column * joint_qd[qd_start + 1]
            joint_S_s[qd_start + 1] = column
        if lin_axis_count > 2:
            column = motion_axis(anchor, joint_axis[qd_start + 2], False)
            twist += column * joint_qd[qd_start + 2]
            joint_S_s[qd_start + 2] = column
        first = qd_start + lin_axis_count
        if ang_axis_count > 0:
            column = motion_axis(anchor, joint_axis[first + 0], True)
            twist += column * joint_qd[first + 0]
            joint_S_s[first + 0] = column
        if ang_axis_count > 1:
            column = motion_axis(anchor, joint_axis[first + 1], True)
            twist += column * joint_qd[first + 1]
            joint_S_s[first + 1] = column
        if ang_axis_count > 2:
            column = motion_axis(anchor, joint_axis[first + 2], True)
            twist += column * joint_qd[first + 2]
            joint_S_s[first + 2] = column
        return twist
    if joint_type == JointType.BALL:
        x = motion_axis(anchor, wp.vec3(1.0, 0.0, 0.0), True)
        y = motion_axis(anchor, wp.vec3(0.0, 1.0, 0.0), True)
        z = motion_axis(anchor, wp.vec3(0.0, 0.0, 1.0), True)
        joint_S_s[qd_start + 0] = x
        joint_S_s[qd_start + 1] = y
        joint_S_s[qd_start + 2] = z
        return x * joint_qd[qd_start + 0] + y * joint_qd[qd_start + 1] + z * joint_qd[qd_start + 2]
    if joint_type == JointType.FREE or joint_type == JointType.DISTANCE:
        rotation = wp.transform_get_rotation(anchor)
        linear = wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2])
        angular = wp.vec3(joint_qd[qd_start + 3], joint_qd[qd_start + 4], joint_qd[qd_start + 5])
        ex = wp.quat_rotate(rotation, wp.vec3(1.0, 0.0, 0.0))
        ey = wp.quat_rotate(rotation, wp.vec3(0.0, 1.0, 0.0))
        ez = wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0))
        joint_S_s[qd_start + 0] = wp.spatial_vector(ex, wp.vec3())
        joint_S_s[qd_start + 1] = wp.spatial_vector(ey, wp.vec3())
        joint_S_s[qd_start + 2] = wp.spatial_vector(ez, wp.vec3())
        joint_S_s[qd_start + 3] = wp.spatial_vector(wp.vec3(), ex)
        joint_S_s[qd_start + 4] = wp.spatial_vector(wp.vec3(), ey)
        joint_S_s[qd_start + 5] = wp.spatial_vector(wp.vec3(), ez)
        return wp.spatial_vector(wp.quat_rotate(rotation, linear), wp.quat_rotate(rotation, angular))
    return wp.spatial_vector()


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
    ``stiffness (reference - q) - damping qd``, added last."""
    passive = stiffness * (reference - q)
    passive -= damping * qd
    return -wp.dot(motion, wrench) + force + passive


@wp.func
def joint_force(
    joint_type: int,
    joint_S_s: wp.array[wp.spatial_vector],
    joint_f: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    spring_stiffness: wp.array[float],
    spring_reference: wp.array[float],
    damping: wp.array[float],
    q_start: int,
    qd_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    wrench: wp.spatial_vector,
    tau: wp.array[float],
):
    """Write a joint's generalized forces: ``joint_f - S^T wrench``, with the passive spring and damper of
    one-coordinate axes (prismatic, revolute and D6 joints), into ``tau``.

    ``wrench`` is the joint's subtree wrench about the origin. Drive gains stay on the implicit augmented
    path; a ball joint's quaternion is never a spring coordinate.
    """
    count = int(0)
    passive = int(0)
    if joint_type == JointType.BALL:
        count = 3
    elif joint_type == JointType.FREE or joint_type == JointType.DISTANCE:
        count = 6
    elif joint_type == JointType.PRISMATIC or joint_type == JointType.REVOLUTE or joint_type == JointType.D6:
        count = lin_axis_count + ang_axis_count
        passive = 1
    for i in range(count):
        dof = qd_start + i
        value = float(0.0)
        if passive != 0:
            value = coordinate_force(
                joint_S_s[dof],
                wrench,
                joint_f[dof],
                spring_stiffness[dof],
                spring_reference[dof],
                damping[dof],
                joint_q[q_start + i],
                joint_qd[dof],
            )
        else:
            value = -wp.dot(joint_S_s[dof], wrench) + joint_f[dof]
        tau[dof] = value
