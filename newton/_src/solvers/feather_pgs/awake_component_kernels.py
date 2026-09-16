# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scalar dynamics and publication for independent prismatic components.

The owner refreshes these model-derived parameters at notified boundaries.
The world axis retains its authored length; it is not a unit-axis assumption.
"""

import warp as wp


@wp.struct
class ScalarParameters:
    """Store the fixed-frame scalar model in SI units.

    ``rest_body`` is the body pose at coordinate zero, and ``com_offset`` is
    its world-frame center-of-mass offset [m]. ``axis`` maps generalized
    position and velocity to world translation [m] and linear velocity [m/s].
    """

    axis: wp.vec3
    rest_body: wp.transform
    com_offset: wp.vec3
    gravity: wp.vec3
    mass: float
    armature: float
    ke: float
    kd: float
    effort: float
    passive_k: float
    passive_ref: float
    damping: float
    allow_sleep: int


@wp.func
def scalar_force(
    p: ScalarParameters,
    q: float,
    v: float,
    target: float,
    target_v: float,
    joint_force: float,
    external: wp.vec3,
    dt: float,
):
    """Return total generalized force and positive augmented inertia.

    ``external`` is the world-frame linear force at the body COM [N]. Only
    actuator force is effort-clamped; gravity, passive and authored forces
    retain the original uncapped scalar law.
    """
    K = p.ke * dt * dt + p.kd * dt
    drive = float(0.0)
    positive_K = float(0.0)
    if K <= 0.0:
        drive = 0.0
    else:
        drive = -(p.ke * (q - target + dt * v) + p.kd * (v - target_v))
        if p.effort > 0.0:
            drive = wp.clamp(drive, -p.effort, p.effort)
        if K > 0.0:
            positive_K = K
    passive = p.passive_k * (p.passive_ref - q) - p.damping * v
    tau = wp.dot(p.axis, p.mass * p.gravity + external) + joint_force + passive
    tau += drive
    return tau, positive_K


@wp.func
def scalar_publication(p: ScalarParameters, q: float, v: float, origin: wp.vec3):
    """Return current body/COM poses, motion, bias acceleration and wrench.

    Poses are in world space [m], velocity is linear/angular [m/s, rad/s],
    and the gravity bias wrench is about the articulation origin [N, N m].
    Bias acceleration is zero; it is not the solved generalized acceleration.
    """
    rotation = wp.transform_get_rotation(p.rest_body)
    position = wp.transform_get_translation(p.rest_body) + p.axis * q
    com_position = position + p.com_offset
    body_q = wp.transform(position, rotation)
    body_q_com = wp.transform(com_position, rotation)
    motion = wp.spatial_vector(p.axis, wp.vec3())
    body_v = motion * v
    body_a = wp.spatial_vector()
    gravity_force = p.mass * p.gravity
    body_f = -wp.spatial_vector(gravity_force, wp.cross(com_position - origin, gravity_force))
    return body_q, body_q_com, motion, body_v, body_a, body_f
