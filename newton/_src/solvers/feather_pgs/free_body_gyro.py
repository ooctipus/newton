# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implicit gyroscopic correction for structurally independent free bodies.

This changes only the unconstrained predictor, before contact construction.
The existing effective inertia, external forces and armature remain intact.
Three Newton iterations approximate the implicit midpoint gyroscopic torque;
they do not guarantee convergence for arbitrarily large angular steps. Neither
an angular-speed clamp nor extra damping is applied. The existing orientation
update is not a midpoint update, so angular-momentum direction is not exact.
"""

import warp as wp


@wp.func
def _implicit_gyro_velocity(
    inertia: wp.mat33,
    effective: wp.mat33,
    old: wp.vec3,
    predicted: wp.vec3,
    dt: float,
) -> wp.vec3:
    """Replace explicit gyro by an implicit torque without replacing other forces.

    Both inertias and angular velocities use the same fixed joint-parent axes.
    ``effective`` must be positive definite. Normalization avoids the absolute
    determinant cutoff of a float32 inverse for small physical inertias.

    Without forces or effective-mass additions, dotting the converged residual
    with midpoint velocity gives zero kinetic-energy change: the gyroscopic
    torque is perpendicular to that velocity. The subsequent normalized pose
    update rotates about the new angular velocity, preserving its energy across
    the corresponding inertia rotation. This is not a momentum-direction proof.
    """
    scale = float(0.0)
    for row in range(3):
        for column in range(3):
            scale = wp.max(scale, wp.abs(effective[row, column]))
    physical = inertia / scale
    mass = effective / scale
    old_gyro = wp.cross(old, physical * old)
    omega = old
    for _ in range(3):
        midpoint = 0.5 * (old + omega)
        momentum = physical * midpoint
        residual = mass * (omega - predicted) + dt * (wp.cross(midpoint, momentum) - old_gyro)
        derivative = mass + 0.5 * dt * (wp.skew(midpoint) * physical - wp.skew(momentum))
        omega = omega - wp.inverse(derivative) * residual
    return omega


@wp.kernel
def correct_free_body_gyroscopic_predictor(
    group_to_art: wp.array[int],
    is_free_rigid: wp.array[int],
    articulation_start: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    kinematic_joint_mask: wp.array[int],
    body_q_com: wp.array[wp.transform],
    joint_X_p: wp.array[wp.transform],
    body_inertia: wp.array[wp.mat33],
    factors: wp.array3d[float],
    joint_qd: wp.array[float],
    dt: float,
    v_hat: wp.array[float],
):
    """Correct eligible six-DOF predictors; leave linear and other DOFs untouched.

    ``is_free_rigid`` identifies one FREE root and no descendant joints. Its
    solve origin is the body COM, so linear/angular effective inertia blocks
    decouple. Root angular DOFs use the parent-anchor axes, not body axes.
    The grouped leaf-first Cholesky factor (``P H P^T = L L^T``, ``P`` the
    coordinates reversed) includes effective mass additions; the physical COM
    inertia alone supplies the gyroscopic torque.
    """
    group = wp.tid()
    articulation = group_to_art[group]
    if is_free_rigid[articulation] == 0:
        return
    joint = articulation_start[articulation]
    if kinematic_joint_mask[joint] != 0:
        return
    body = joint_child[joint]
    dof = joint_qd_start[joint]

    parent_rotation = wp.transform_get_rotation(joint_X_p[joint])
    body_rotation = wp.transform_get_rotation(body_q_com[body])
    relative_rotation = wp.quat_to_matrix(wp.quat_inverse(parent_rotation) * body_rotation)
    physical = relative_rotation * body_inertia[body] * wp.transpose(relative_rotation)

    # The angular block of H = P^T L L^T P: physical coordinate 3 + row is factor row 2 - row (leaf-first).
    effective = wp.mat33()
    for row in range(3):
        for column in range(3):
            coefficient = float(0.0)
            for k in range(6):
                if k <= 2 - row and k <= 2 - column:
                    coefficient += factors[group, 2 - row, k] * factors[group, 2 - column, k]
            effective[row, column] = coefficient

    old = wp.vec3(joint_qd[dof + 3], joint_qd[dof + 4], joint_qd[dof + 5])
    predicted = wp.vec3(v_hat[dof + 3], v_hat[dof + 4], v_hat[dof + 5])
    corrected = _implicit_gyro_velocity(physical, effective, old, predicted, dt)
    v_hat[dof + 3] = corrected[0]
    v_hat[dof + 4] = corrected[1]
    v_hat[dof + 5] = corrected[2]
