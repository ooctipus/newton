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
inertia, :func:`inertia_apply` maps a twist to the momentum wrench and
:func:`inertia_matrix` packs the dense 6x6 matrix for consumers that read one.

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
def inertia_at(mass: float, inertia: wp.mat33, rotation: wp.quat, offset: wp.vec3) -> tuple[float, wp.vec3, wp.mat33]:
    """Return a body's inertia about the origin: ``(mass, mass * offset, rotational)``.

    ``inertia`` is the body-frame inertia about the center of mass, ``rotation``
    the body's orientation and ``offset`` its center of mass relative to the
    origin. The rotational inertia is ``R I R^T + mass (|offset|^2 1 - offset
    offset^T)``, the parallel-axis law entry by entry.
    """
    basis = wp.quat_to_matrix(rotation)
    rotated = basis * inertia * wp.transpose(basis)
    square = wp.dot(offset, offset)
    x = offset[0]
    y = offset[1]
    z = offset[2]
    rotational = wp.mat33(
        rotated[0, 0] + mass * (square - x * x),
        rotated[0, 1] + mass * (0.0 - x * y),
        rotated[0, 2] + mass * (0.0 - x * z),
        rotated[1, 0] + mass * (0.0 - y * x),
        rotated[1, 1] + mass * (square - y * y),
        rotated[1, 2] + mass * (0.0 - y * z),
        rotated[2, 0] + mass * (0.0 - z * x),
        rotated[2, 1] + mass * (0.0 - z * y),
        rotated[2, 2] + mass * (square - z * z),
    )
    return mass, mass * offset, rotational


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
