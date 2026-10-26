# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import cache

import warp as wp

from ...math.spatial import transform_twist
from ...sim import BodyFlags, JointType
from ...sim.articulation import (
    compute_2d_rotational_dofs,
    compute_3d_rotational_dofs,
    eval_single_articulation_fk,
)
from .contact_filters import contact_friction_eligible, contact_normal_gap_limit
from .contact_laws import (
    contact_endpoints,
    contact_friction_points,
    contact_geometry,
    contact_materials,
    contact_points,
    contact_responds,
    contact_row_frame,
    contact_tangent_basis,
    free_body_jacobian,
    jacobian_entry,
    prescribed_target,
)
from .device_preludes import (
    dual_native,
    lane_group,
    lane_minimum,
    reserve,
    reserved_count,
    warp_scan,
    warp_sum,
    warp_sync,
)
from .friction_patches import CONTACT_PATH_RESOLVED, FrictionPatches, patch_friction_location, patch_normal_load
from .projection_laws import (
    coulomb_bisection,
    coulomb_newton,
    de_saxce_target,
    disk,
    friction_gate,
    friction_pair,
    regularized_delta,
    relaxed_step,
    speed_row,
    unilateral,
)
from .row_laws import (
    contact_depenetration_cap,
    contact_restitution_fires,
    contact_restitution_rhs,
    contact_rhs,
    joint_limit_side,
    limit_rhs,
    mimic_phi,
    row_rhs,
    velocity_limit_active,
)

PGS_CONSTRAINT_TYPE_CONTACT = 0
PGS_CONSTRAINT_TYPE_JOINT_TARGET = 1
PGS_CONSTRAINT_TYPE_FRICTION = 2
PGS_CONSTRAINT_TYPE_JOINT_LIMIT = 3
# Joint velocity-limit row. Mirrors the PhysX per-DOF velocity clamp (see
# ``notes/investigations/velocity-spike/physx-deep-dive.md`` §4, math
# appendix). Finite limits create two unilateral rows every step: one lower
# bound row and one upper bound row. No Baumgarte / ERP bias.
PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT = 4
# Mimic (joint coupling) row: the bilateral equality
# ``q_follower - coef1 * q_leader - coef0 = 0`` between two 1-DoF joints of the
# same articulation, from ``Model.constraint_mimic_*``. Jacobian has two
# entries (+1 on the follower DOF, -coef1 on the leader DOF); lambda is
# unbounded; drift is removed with the standard Baumgarte ``pgs_beta`` bias.
PGS_CONSTRAINT_TYPE_MIMIC = 5
# Connect (loop-closure) row: one axis of the bilateral 3-DoF point-coincidence equality
# ``p_anchor(parent) - p_anchor(child) = 0`` closing a kinematic loop between two links of
# the same articulation (imported MJCF ``connect`` equalities become trailing BALL loop
# joints; see ``_FeatherPGSModelPlan.build``). Three rows per closure, world-axis
# directions, unbounded lambda, Baumgarte drift correction.
PGS_CONSTRAINT_TYPE_CONNECT = 6
PGS_CONSTRAINT_TYPE_TORSION = 7
PGS_CONSTRAINT_TYPE_COUNT = 8

# Keep launch-geometry-specific dynamics kernels out of the large general
# kernel module. Warp compiles one whole module variant per block dimension.
_KINEMATICS_KERNEL_MODULE = wp.Module(f"{__name__}.kinematics")
_INVERSE_DYNAMICS_KERNEL_MODULE = wp.Module(f"{__name__}.inverse_dynamics")
_MASS_DYNAMICS_KERNEL_MODULE = wp.Module(f"{__name__}.mass_dynamics")

PGS_LOCAL_SOLVE_OWNER_GENERAL = 0
PGS_LOCAL_SOLVE_OWNER_SINGLE = 1
PGS_LOCAL_SOLVE_OWNER_PAIR = 2
PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL = 3


@wp.kernel
def local_solve_launch_gate():
    """Create a minimal graph dependency ahead of the bulk local solves."""

    pass


# Numeric IDs for the ``friction_mode`` argument passed to the matrix-free
# PGS solver kernels.  Mirrors the Python-side string enum on
# :class:`~newton.solvers.SolverFeatherPGS` (``"current"`` /
# ``"bisection"`` / ``"bisection_desaxce"`` / ``"coulomb_newton"``).
# The matrix-free kernel branches on the numeric id at each PGS row to
# avoid duplicating the kernel body per friction strategy (see the
# ``[FPGS Friction Modes]`` issue series).
FRICTION_MODE_CURRENT = 0
FRICTION_MODE_BISECTION = 1
FRICTION_MODE_BISECTION_DESAXCE = 2
# Gilles Daviet's scalar bracketed Newton on the tangential-force ratio
# alpha (FPGS Friction Modes 7/13): projection_laws.coulomb_newton.
FRICTION_MODE_COULOMB_NEWTON = 3


@wp.kernel
def compute_spatial_inertia(
    body_inertia: wp.array[wp.mat33],
    body_mass: wp.array[float],
    # outputs
    body_I_m: wp.array[wp.spatial_matrix],
):
    tid = wp.tid()
    I = body_inertia[tid]
    m = body_mass[tid]
    # fmt: off
    body_I_m[tid] = wp.spatial_matrix(
        m,   0.0, 0.0, 0.0,     0.0,     0.0,
        0.0, m,   0.0, 0.0,     0.0,     0.0,
        0.0, 0.0, m,   0.0,     0.0,     0.0,
        0.0, 0.0, 0.0, I[0, 0], I[0, 1], I[0, 2],
        0.0, 0.0, 0.0, I[1, 0], I[1, 1], I[1, 2],
        0.0, 0.0, 0.0, I[2, 0], I[2, 1], I[2, 2],
    )
    # fmt: on


@wp.kernel
def compute_com_transforms(
    body_com: wp.array[wp.vec3],
    # outputs
    body_X_com: wp.array[wp.transform],
):
    tid = wp.tid()
    com = body_com[tid]
    body_X_com[tid] = wp.transform(com, wp.quat_identity())


@wp.kernel
def clamp_free_root_velocity_limits(
    articulation_start: wp.array[int],
    joint_child: wp.array[int],
    body_flags: wp.array[wp.int32],
    articulation_root_is_free: wp.array[int],
    articulation_root_dof_start: wp.array[int],
    rigid_body_max_linear_velocity: wp.array[float],
    rigid_body_max_angular_velocity: wp.array[float],
    # outputs
    qd: wp.array[float],
):
    """Clamp free-root linear/angular velocity magnitudes to PhysX rigid-body limits."""
    art = wp.tid()
    if articulation_root_is_free[art] == 0:
        return

    root_joint = articulation_start[art]
    root_body = joint_child[root_joint]
    if root_body < 0:
        return
    if (body_flags[root_body] & BodyFlags.KINEMATIC) != 0:
        return

    ds = articulation_root_dof_start[art]

    max_lin = rigid_body_max_linear_velocity[root_body]
    if max_lin > 0.0 and wp.isfinite(max_lin):
        lin = wp.vec3(qd[ds + 0], qd[ds + 1], qd[ds + 2])
        lin_speed = wp.length(lin)
        if lin_speed > max_lin:
            scale = max_lin / lin_speed
            qd[ds + 0] = lin[0] * scale
            qd[ds + 1] = lin[1] * scale
            qd[ds + 2] = lin[2] * scale

    max_ang = rigid_body_max_angular_velocity[root_body]
    if max_ang > 0.0 and wp.isfinite(max_ang):
        ang = wp.vec3(qd[ds + 3], qd[ds + 4], qd[ds + 5])
        ang_speed = wp.length(ang)
        if ang_speed > max_ang:
            scale = max_ang / ang_speed
            qd[ds + 3] = ang[0] * scale
            qd[ds + 4] = ang[1] * scale
            qd[ds + 5] = ang[2] * scale


@wp.func
def _prescaled_axis_count(
    joint: int,
    joint_type: wp.array[int],
    joint_child: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    body_flags: wp.array[wp.int32],
) -> int:
    """Return how many leading DOFs of ``joint`` the PhysX velocity prescale reads and scales.

    Joint types without a PhysX joint velocity limit and kinematic children
    have none.
    """
    jtype = joint_type[joint]
    if jtype != JointType.PRISMATIC and jtype != JointType.REVOLUTE and jtype != JointType.D6:
        return 0
    if (body_flags[joint_child[joint]] & BodyFlags.KINEMATIC) != 0:
        return 0
    return joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]


@wp.kernel(enable_backward=False)
def prescale_joint_velocity_limits(
    articulation_count: int,
    lanes: int,
    articulation_start: wp.array[int],
    joint_type: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_velocity_limit: wp.array[float],
    body_flags: wp.array[wp.int32],
    drive_slot: wp.array[int],
    skip_driven: int,
    joint_qd: wp.array[float],
    # out
    prescaled_joint_qd: wp.array[float],
):
    """PhysX-style pre-solve joint velocity scaling.

    PhysX computes a single ratio per articulation from maxJointVelocity and
    applies that ratio to all articulation DOFs before building link velocities.
    This is separate from the velocity-limit constraint rows solved later.

    ``skip_driven != 0`` (the ``fuse_joint_velocity_limits`` path) excludes
    DOFs with a PhysX drive row (``drive_slot[dof] >= 0``) from both the
    ratio computation and the scaling: those DOFs are clamped in-solve by the
    fused end-of-iteration velocity clamp instead. ``drive_slot`` holds the previous
    step's allocation at this point in the step (it is initialized to -1, so
    the very first step prescales exactly as before); the driven-DOF set is
    derived from ``joint_target_ke/kd > 0`` and is stable across steps.
    When ``skip_driven == 0``, ``drive_slot`` is never read (a 1-element
    dummy is safe).

    An aligned group of ``lanes`` threads owns each articulation, the joints
    from ``articulation_start[art]`` up to the next articulation's start. Lane
    ``k`` takes joints ``k, k + lanes, ...`` and the group combines the lanes'
    ratios with a minimum. Every candidate ratio is 1 or a quotient of two
    positive numbers, never NaN or -0, so the minimum selects one of them
    and its bits do not depend on the order, as in the serial loop. Each DOF
    of ``prescaled_joint_qd`` is written once: scaled DOFs get
    ``joint_qd * ratio`` and every other DOF, including those of joints ahead
    of the first articulation, a copy of ``joint_qd``. CUDA launches cover
    complete warps with a power-of-two ``lanes`` of at most 32; CPU launches
    use one lane.
    """
    tid = wp.tid()
    art = tid // lanes
    lane = tid - art * lanes
    joint_start = int(0)
    joint_end = int(0)
    if art < articulation_count:
        joint_start = articulation_start[art]
        joint_end = articulation_start[art + 1]

    ratio = float(1.0)
    for j in range(joint_start + lane, joint_end, lanes):
        qd_start = joint_qd_start[j]
        for axis in range(_prescaled_axis_count(j, joint_type, joint_child, joint_dof_dim, body_flags)):
            dof = qd_start + axis
            if skip_driven != 0:
                if drive_slot[dof] >= 0:
                    continue
            limit = joint_velocity_limit[dof]
            qd_abs = wp.abs(joint_qd[dof])
            if limit > 0.0 and wp.isfinite(limit) and qd_abs > 0.0:
                scale = limit / qd_abs
                if scale < ratio:
                    ratio = scale
    # Padding lanes past the last articulation take part with ratio 1.
    ratio = lane_minimum(ratio, lanes)

    if art == 0:
        # Joints ahead of the first articulation keep their velocity.
        for dof in range(lane, joint_qd_start[joint_start], lanes):
            prescaled_joint_qd[dof] = joint_qd[dof]
    if ratio < 1.0:
        for j in range(joint_start + lane, joint_end, lanes):
            qd_start = joint_qd_start[j]
            scaled_end = qd_start + _prescaled_axis_count(j, joint_type, joint_child, joint_dof_dim, body_flags)
            for dof in range(qd_start, joint_qd_start[j + 1]):
                qd = joint_qd[dof]
                if dof < scaled_end:
                    if skip_driven == 0:
                        qd = qd * ratio
                    elif drive_slot[dof] < 0:
                        qd = qd * ratio
                prescaled_joint_qd[dof] = qd
    else:
        for dof in range(joint_qd_start[joint_start] + lane, joint_qd_start[joint_end], lanes):
            prescaled_joint_qd[dof] = joint_qd[dof]


@wp.func
def transform_spatial_inertia(t: wp.transform, I: wp.spatial_matrix):
    """
    Transform a spatial inertia tensor to a new coordinate frame.

    This computes the change of coordinates for a spatial inertia tensor under a rigid-body
    transformation `t`. The result is mathematically equivalent to:

        adj_t^-T * I * adj_t^-1

    where `adj_t` is the adjoint transformation matrix of `t`, and `I` is the spatial inertia
    tensor in the original frame. This operation is described in Frank & Park, "Modern Robotics",
    Section 8.2.3 (pg. 290).

    Args:
        t (wp.transform): The rigid-body transform (destination ← source).
        I (wp.spatial_matrix): The spatial inertia tensor in the source frame.

    Returns:
        wp.spatial_matrix: The spatial inertia tensor expressed in the destination frame.
    """
    t_inv = wp.transform_inverse(t)

    q = wp.transform_get_rotation(t_inv)
    p = wp.transform_get_translation(t_inv)

    r1 = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    r2 = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    r3 = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))

    R = wp.matrix_from_cols(r1, r2, r3)
    S = wp.skew(p) @ R

    T = wp.spatial_matrix(
        R[0, 0],
        R[0, 1],
        R[0, 2],
        S[0, 0],
        S[0, 1],
        S[0, 2],
        R[1, 0],
        R[1, 1],
        R[1, 2],
        S[1, 0],
        S[1, 1],
        S[1, 2],
        R[2, 0],
        R[2, 1],
        R[2, 2],
        S[2, 0],
        S[2, 1],
        S[2, 2],
        0.0,
        0.0,
        0.0,
        R[0, 0],
        R[0, 1],
        R[0, 2],
        0.0,
        0.0,
        0.0,
        R[1, 0],
        R[1, 1],
        R[1, 2],
        0.0,
        0.0,
        0.0,
        R[2, 0],
        R[2, 1],
        R[2, 2],
    )

    return wp.mul(wp.mul(wp.transpose(T), I), T)


@wp.func
def transform_com_inertia_terms(t: wp.transform, mass: float, inertia_com: wp.mat33):
    """Rotate COM inertia and shift its angular block to the solve origin."""
    rotation = wp.quat_to_matrix(wp.transform_get_rotation(t))
    com = wp.transform_get_translation(t)
    com_cross = wp.skew(com)
    inertia_origin = rotation * inertia_com * wp.transpose(rotation) - mass * com_cross * com_cross
    return com, inertia_origin


@wp.func
def assemble_com_spatial_inertia(mass: float, com: wp.vec3, inertia_origin: wp.mat33):
    """Assemble a solve-frame spatial inertia from compact COM terms."""
    mass_com_cross = mass * wp.skew(com)
    # fmt: off
    return wp.spatial_matrix(
        mass, 0.0,  0.0,  -mass_com_cross[0, 0], -mass_com_cross[0, 1], -mass_com_cross[0, 2],
        0.0,  mass, 0.0,  -mass_com_cross[1, 0], -mass_com_cross[1, 1], -mass_com_cross[1, 2],
        0.0,  0.0,  mass, -mass_com_cross[2, 0], -mass_com_cross[2, 1], -mass_com_cross[2, 2],
        mass_com_cross[0, 0], mass_com_cross[0, 1], mass_com_cross[0, 2],
        inertia_origin[0, 0], inertia_origin[0, 1], inertia_origin[0, 2],
        mass_com_cross[1, 0], mass_com_cross[1, 1], mass_com_cross[1, 2],
        inertia_origin[1, 0], inertia_origin[1, 1], inertia_origin[1, 2],
        mass_com_cross[2, 0], mass_com_cross[2, 1], mass_com_cross[2, 2],
        inertia_origin[2, 0], inertia_origin[2, 1], inertia_origin[2, 2],
    )
    # fmt: on


@wp.func
def mul_com_spatial_inertia(mass: float, com: wp.vec3, inertia_origin: wp.mat33, velocity: wp.spatial_vector):
    """Multiply a solve-frame twist by a COM-centered rigid-body inertia."""
    linear = wp.spatial_top(velocity)
    angular = wp.spatial_bottom(velocity)
    return wp.spatial_vector(
        mass * (linear - wp.cross(com, angular)),
        mass * wp.cross(com, linear) + inertia_origin * angular,
    )


# compute transform across a joint
@wp.func
def jcalc_transform(
    type: int,
    joint_axis: wp.array[wp.vec3],
    axis_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    joint_q: wp.array[float],
    q_start: int,
):
    if type == JointType.PRISMATIC:
        q = joint_q[q_start]
        axis = joint_axis[axis_start]
        X_jc = wp.transform(axis * q, wp.quat_identity())
        return X_jc

    if type == JointType.REVOLUTE:
        q = joint_q[q_start]
        axis = joint_axis[axis_start]
        X_jc = wp.transform(wp.vec3(), wp.quat_from_axis_angle(axis, q))
        return X_jc

    if type == JointType.BALL:
        qx = joint_q[q_start + 0]
        qy = joint_q[q_start + 1]
        qz = joint_q[q_start + 2]
        qw = joint_q[q_start + 3]

        X_jc = wp.transform(wp.vec3(), wp.quat(qx, qy, qz, qw))
        return X_jc

    if type == JointType.FIXED:
        X_jc = wp.transform_identity()
        return X_jc

    if type == JointType.FREE or type == JointType.DISTANCE:
        px = joint_q[q_start + 0]
        py = joint_q[q_start + 1]
        pz = joint_q[q_start + 2]

        qx = joint_q[q_start + 3]
        qy = joint_q[q_start + 4]
        qz = joint_q[q_start + 5]
        qw = joint_q[q_start + 6]

        X_jc = wp.transform(wp.vec3(px, py, pz), wp.quat(qx, qy, qz, qw))
        return X_jc

    if type == JointType.D6:
        pos = wp.vec3(0.0)
        rot = wp.quat_identity()

        # unroll for loop to ensure joint actions remain differentiable
        # (since differentiating through a for loop that updates a local variable is not supported)

        if lin_axis_count > 0:
            axis = joint_axis[axis_start + 0]
            pos += axis * joint_q[q_start + 0]
        if lin_axis_count > 1:
            axis = joint_axis[axis_start + 1]
            pos += axis * joint_q[q_start + 1]
        if lin_axis_count > 2:
            axis = joint_axis[axis_start + 2]
            pos += axis * joint_q[q_start + 2]

        ia = axis_start + lin_axis_count
        iq = q_start + lin_axis_count
        if ang_axis_count == 1:
            axis = joint_axis[ia]
            rot = wp.quat_from_axis_angle(axis, joint_q[iq])
        if ang_axis_count == 2:
            rot, _ = compute_2d_rotational_dofs(
                joint_axis[ia + 0],
                joint_axis[ia + 1],
                joint_q[iq + 0],
                joint_q[iq + 1],
                0.0,
                0.0,
            )
        if ang_axis_count == 3:
            rot, _ = compute_3d_rotational_dofs(
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

        X_jc = wp.transform(pos, rot)
        return X_jc

    # default case
    return wp.transform_identity()


# compute motion subspace and velocity for a joint
@wp.func
def jcalc_motion(
    type: int,
    joint_axis: wp.array[wp.vec3],
    lin_axis_count: int,
    ang_axis_count: int,
    X_sc: wp.transform,
    joint_qd: wp.array[float],
    qd_start: int,
    # outputs
    joint_S_s: wp.array[wp.spatial_vector],
):
    if type == JointType.PRISMATIC:
        axis = joint_axis[qd_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
        v_j_s = S_s * joint_qd[qd_start]
        joint_S_s[qd_start] = S_s
        return v_j_s

    if type == JointType.REVOLUTE:
        axis = joint_axis[qd_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
        v_j_s = S_s * joint_qd[qd_start]
        joint_S_s[qd_start] = S_s
        return v_j_s

    if type == JointType.D6:
        v_j_s = wp.spatial_vector()
        if lin_axis_count > 0:
            axis = joint_axis[qd_start + 0]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + 0]
            joint_S_s[qd_start + 0] = S_s
        if lin_axis_count > 1:
            axis = joint_axis[qd_start + 1]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + 1]
            joint_S_s[qd_start + 1] = S_s
        if lin_axis_count > 2:
            axis = joint_axis[qd_start + 2]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + 2]
            joint_S_s[qd_start + 2] = S_s
        if ang_axis_count > 0:
            axis = joint_axis[qd_start + lin_axis_count + 0]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 0]
            joint_S_s[qd_start + lin_axis_count + 0] = S_s
        if ang_axis_count > 1:
            axis = joint_axis[qd_start + lin_axis_count + 1]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 1]
            joint_S_s[qd_start + lin_axis_count + 1] = S_s
        if ang_axis_count > 2:
            axis = joint_axis[qd_start + lin_axis_count + 2]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 2]
            joint_S_s[qd_start + lin_axis_count + 2] = S_s

        return v_j_s

    if type == JointType.BALL:
        S_0 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        S_1 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        S_2 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

        joint_S_s[qd_start + 0] = S_0
        joint_S_s[qd_start + 1] = S_1
        joint_S_s[qd_start + 2] = S_2

        return S_0 * joint_qd[qd_start + 0] + S_1 * joint_qd[qd_start + 1] + S_2 * joint_qd[qd_start + 2]

    if type == JointType.FIXED:
        return wp.spatial_vector()

    if type == JointType.FREE or type == JointType.DISTANCE:
        # For FREE/DISTANCE joints we treat linear/angular velocity components as
        # referenced at the root COM world point to avoid world-origin conditioning.
        q_sc = wp.transform_get_rotation(X_sc)

        v_local = wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2])
        w_local = wp.vec3(joint_qd[qd_start + 3], joint_qd[qd_start + 4], joint_qd[qd_start + 5])
        v_j_s = wp.spatial_vector(wp.quat_rotate(q_sc, v_local), wp.quat_rotate(q_sc, w_local))

        ex = wp.quat_rotate(q_sc, wp.vec3(1.0, 0.0, 0.0))
        ey = wp.quat_rotate(q_sc, wp.vec3(0.0, 1.0, 0.0))
        ez = wp.quat_rotate(q_sc, wp.vec3(0.0, 0.0, 1.0))

        joint_S_s[qd_start + 0] = wp.spatial_vector(ex, wp.vec3())
        joint_S_s[qd_start + 1] = wp.spatial_vector(ey, wp.vec3())
        joint_S_s[qd_start + 2] = wp.spatial_vector(ez, wp.vec3())
        joint_S_s[qd_start + 3] = wp.spatial_vector(wp.vec3(), ex)
        joint_S_s[qd_start + 4] = wp.spatial_vector(wp.vec3(), ey)
        joint_S_s[qd_start + 5] = wp.spatial_vector(wp.vec3(), ez)

        return v_j_s

    wp.printf("jcalc_motion not implemented for joint type %d\n", type)

    # default case
    return wp.spatial_vector()


# computes joint space forces/torques in tau
@wp.func
def jcalc_tau(
    type: int,
    joint_S_s: wp.array[wp.spatial_vector],
    joint_f: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_spring_stiffness: wp.array[float],
    joint_spring_ref: wp.array[float],
    joint_damping: wp.array[float],
    coord_start: int,
    dof_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    body_f_s: wp.spatial_vector,
    add_existing_tau: int,
    # outputs
    tau: wp.array[float],
):
    if type == JointType.BALL:
        # target_ke = joint_target_ke[dof_start]
        # target_kd = joint_target_kd[dof_start]

        for i in range(3):
            S_s = joint_S_s[dof_start + i]

            # w = joint_qd[dof_start + i]
            # r = joint_q[coord_start + i]

            value = -wp.dot(S_s, body_f_s) + joint_f[dof_start + i]
            if add_existing_tau != 0:
                value += tau[dof_start + i]
            tau[dof_start + i] = value
            # tau -= w * target_kd - r * target_ke

        return

    if type == JointType.FREE or type == JointType.DISTANCE:
        for i in range(6):
            S_s = joint_S_s[dof_start + i]
            value = -wp.dot(S_s, body_f_s) + joint_f[dof_start + i]
            if add_existing_tau != 0:
                value += tau[dof_start + i]
            tau[dof_start + i] = value

        return

    if type == JointType.PRISMATIC or type == JointType.REVOLUTE or type == JointType.D6:
        axis_count = lin_axis_count + ang_axis_count

        for i in range(axis_count):
            j = dof_start + i
            S_s = joint_S_s[j]
            # Passive spring/damping applied explicitly; the drive gains stay on the
            # implicit augmented path. These joint types have one coordinate per DOF,
            # so coord_start + i addresses the axis position.
            passive_f = joint_spring_stiffness[j] * (joint_spring_ref[j] - joint_q[coord_start + i])
            passive_f -= joint_damping[j] * joint_qd[j]
            # total torque / force on the joint (drive forces handled via augmented mass)
            value = -wp.dot(S_s, body_f_s) + joint_f[j] + passive_f
            if add_existing_tau != 0:
                value += tau[j]
            tau[j] = value

        return


@wp.func
def jcalc_integrate(
    type: int,
    child: int,
    body_com: wp.array[wp.vec3],
    X_cj: wp.transform,
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_qdd: wp.array[float],
    coord_start: int,
    dof_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    dt: float,
    angular_damping: float,
    parent: int,
    # outputs
    joint_q_new: wp.array[float],
    joint_qd_new: wp.array[float],
):
    if type == JointType.FIXED:
        return

    # prismatic / revolute
    if type == JointType.PRISMATIC or type == JointType.REVOLUTE:
        qdd = joint_qdd[dof_start]
        qd = joint_qd[dof_start]
        q = joint_q[coord_start]

        qd_new = qd + qdd * dt
        q_new = q + qd_new * dt

        joint_qd_new[dof_start] = qd_new
        joint_q_new[coord_start] = q_new

        return

    # ball
    if type == JointType.BALL:
        m_j = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])
        w_j = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])

        r_j = wp.quat(
            joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2], joint_q[coord_start + 3]
        )

        # symplectic Euler
        w_j_new = w_j + m_j * dt

        drdt_j = wp.quat(w_j_new, 0.0) * r_j * 0.5

        # new orientation (normalized)
        r_j_new = wp.normalize(r_j + drdt_j * dt)

        # update joint coords
        joint_q_new[coord_start + 0] = r_j_new[0]
        joint_q_new[coord_start + 1] = r_j_new[1]
        joint_q_new[coord_start + 2] = r_j_new[2]
        joint_q_new[coord_start + 3] = r_j_new[3]

        # update joint vel
        joint_qd_new[dof_start + 0] = w_j_new[0]
        joint_qd_new[dof_start + 1] = w_j_new[1]
        joint_qd_new[dof_start + 2] = w_j_new[2]

        return

    if type == JointType.FREE or type == JointType.DISTANCE:
        a_s = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])
        m_s = wp.vec3(joint_qdd[dof_start + 3], joint_qdd[dof_start + 4], joint_qdd[dof_start + 5])

        v_com = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])
        w_s = wp.vec3(joint_qd[dof_start + 3], joint_qd[dof_start + 4], joint_qd[dof_start + 5])

        # symplectic Euler. joint_qdd's linear rows give the acceleration of the articulation-frame
        # origin, a point fixed in the root body, so its velocity also changes by transport as the
        # body rotates: that is the omega x v term. SolverFeatherstone performs the same conversion
        # explicitly (a_com = a + alpha x x_com + omega x v_com); omitting it leaves the free base
        # short by a term proportional to the spin. The same term must appear in the velocity
        # predictor and be removed by the velocity-to-acceleration conversion (see
        # apply_free_root_transport_to_predictor / remove_free_root_transport_from_qdd), or
        # constraint rows are built against a velocity this integration never realizes.
        w_prev = w_s
        w_s = w_s + m_s * dt
        if parent < 0:
            v_com = v_com + (a_s + wp.cross(w_prev, v_com)) * dt
        else:
            # A descendant free joint's coordinate is a RELATIVE twist in the parent anchor
            # frame; the root transport rule above is not derived for it, so integrate
            # component-wise until the parent-frame transport is.
            v_com = v_com + a_s * dt
        w_s_integrate = w_s

        p_s = wp.vec3(joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2])

        r_s = wp.quat(
            joint_q[coord_start + 3], joint_q[coord_start + 4], joint_q[coord_start + 5], joint_q[coord_start + 6]
        )
        # (p_s, r_s) track the child ANCHOR frame, so the lever to the COM must go through the
        # child anchor transform: with a non-identity X_cj the COM does not sit at
        # body_com[child] in anchor coordinates.
        r_ac = wp.transform_point(wp.transform_inverse(X_cj), body_com[child])

        drdt_s = wp.quat(w_s_integrate, 0.0) * r_s * 0.5
        r_s_new = wp.normalize(r_s + drdt_s * dt)

        if parent < 0:
            # Reconstruct the root anchor from the integrated COM instead of
            # advancing it with the linearized lever velocity: the COM is the
            # point whose velocity the coordinate stores, so integrate it
            # directly and place the anchor at x_com - R_new * r_ac.  A
            # force-free COM then stays stationary to roundoff, where the
            # linearized form (p += (v - w x R_old*r_ac) * dt) has
            # O(omega^2 * |r_ac| * dt^2) local error, which accumulates into
            # first-order global drift.
            # SolverFeatherstone integrates body poses around the COM the
            # same way.
            x_com_new = p_s + wp.quat_rotate(r_s, r_ac) + v_com * dt
            p_s_new = x_com_new - wp.quat_rotate(r_s_new, r_ac)
        else:
            # Descendant free joints keep the linearized relative update until
            # the moving-parent-frame transport is derived.
            dpdt_s = v_com - wp.cross(w_s_integrate, wp.quat_rotate(r_s, r_ac))
            p_s_new = p_s + dpdt_s * dt

        if parent < 0:
            w_s = w_s * (1.0 - angular_damping * dt)

        # update transform
        joint_q_new[coord_start + 0] = p_s_new[0]
        joint_q_new[coord_start + 1] = p_s_new[1]
        joint_q_new[coord_start + 2] = p_s_new[2]

        joint_q_new[coord_start + 3] = r_s_new[0]
        joint_q_new[coord_start + 4] = r_s_new[1]
        joint_q_new[coord_start + 5] = r_s_new[2]
        joint_q_new[coord_start + 6] = r_s_new[3]

        joint_qd_new[dof_start + 0] = v_com[0]
        joint_qd_new[dof_start + 1] = v_com[1]
        joint_qd_new[dof_start + 2] = v_com[2]
        joint_qd_new[dof_start + 3] = w_s[0]
        joint_qd_new[dof_start + 4] = w_s[1]
        joint_qd_new[dof_start + 5] = w_s[2]

        return

    # other joint types (compound, universal, D6)
    if type == JointType.D6:
        axis_count = lin_axis_count + ang_axis_count

        for i in range(axis_count):
            qdd = joint_qdd[dof_start + i]
            qd = joint_qd[dof_start + i]
            q = joint_q[coord_start + i]

            qd_new = qd + qdd * dt
            q_new = q + qd_new * dt

            joint_qd_new[dof_start + i] = qd_new
            joint_q_new[coord_start + i] = q_new

        return


@wp.func
def compute_link_transform(
    i: int,
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
    # outputs
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
):
    # parent transform
    parent = joint_parent[i]
    child = joint_child[i]

    # parent transform in spatial coordinates
    X_pj = joint_X_p[i]
    X_cj = joint_X_c[i]
    # parent anchor frame in world space
    X_wpj = X_pj
    if parent >= 0:
        X_wp = body_q[parent]
        X_wpj = X_wp * X_wpj

    type = joint_type[i]
    qd_start = joint_qd_start[i]
    lin_axis_count = joint_dof_dim[i, 0]
    ang_axis_count = joint_dof_dim[i, 1]
    coord_start = joint_q_start[i]

    # compute transform across joint
    X_j = jcalc_transform(type, joint_axis, qd_start, lin_axis_count, ang_axis_count, joint_q, coord_start)

    # transform from world to joint anchor frame at child body
    X_wcj = X_wpj * X_j
    # transform from world to child body frame
    X_wc = X_wcj * wp.transform_inverse(X_cj)

    # compute transform of center of mass
    X_cm = body_X_com[child]
    X_sm = X_wc * X_cm

    # store geometry transforms
    body_q[child] = X_wc
    body_q_com[child] = X_sm


@wp.func
def spatial_cross(a: wp.spatial_vector, b: wp.spatial_vector):
    w_a = wp.spatial_bottom(a)
    v_a = wp.spatial_top(a)

    w_b = wp.spatial_bottom(b)
    v_b = wp.spatial_top(b)

    w = wp.cross(w_a, w_b)
    v = wp.cross(w_a, v_b) + wp.cross(v_a, w_b)

    return wp.spatial_vector(v, w)


@wp.func
def spatial_cross_dual(a: wp.spatial_vector, b: wp.spatial_vector):
    w_a = wp.spatial_bottom(a)
    v_a = wp.spatial_top(a)

    w_b = wp.spatial_bottom(b)
    v_b = wp.spatial_top(b)

    w = wp.cross(w_a, w_b) + wp.cross(v_a, v_b)
    v = wp.cross(w_a, v_b)

    return wp.spatial_vector(v, w)


@wp.func
def dense_index(stride: int, i: int, j: int):
    return i * stride + j


@wp.func
def compute_link_kinematics(
    i: int,
    parent: int,
    child: int,
    parent_v_s: wp.spatial_vector,
    parent_a_s: wp.spatial_vector,
    origin: wp.vec3,
    joint_type: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_qd: wp.array[float],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_q: wp.array[wp.transform],
    joint_X_p: wp.array[wp.transform],
    # outputs
    joint_S_s: wp.array[wp.spatial_vector],
    body_v_s: wp.array[wp.spatial_vector],
    body_a_s: wp.array[wp.spatial_vector],
):
    type = joint_type[i]
    qd_start = joint_qd_start[i]

    X_pj = joint_X_p[i]
    # X_cj = joint_X_c[i]

    # parent anchor frame in world space
    X_wpj = X_pj
    if parent >= 0:
        X_wp = body_q[parent]
        X_wpj = X_wp * X_wpj
    X_wpj_local = wp.transform(
        wp.transform_get_translation(X_wpj) - origin,
        wp.transform_get_rotation(X_wpj),
    )

    # compute motion subspace and velocity across the joint (also stores S_s to global memory)
    lin_axis_count = joint_dof_dim[i, 0]
    ang_axis_count = joint_dof_dim[i, 1]
    v_j_s = jcalc_motion(
        type,
        joint_axis,
        lin_axis_count,
        ang_axis_count,
        X_wpj_local,
        joint_qd,
        qd_start,
        joint_S_s,
    )

    # body velocity, acceleration
    v_s = parent_v_s + v_j_s
    a_s = parent_a_s + spatial_cross(v_s, v_j_s)

    body_v_s[child] = v_s
    body_a_s[child] = a_s
    return v_s, a_s


@wp.func
def compute_link_velocity(
    i: int,
    parent: int,
    child: int,
    parent_v_s: wp.spatial_vector,
    parent_a_s: wp.spatial_vector,
    origin: wp.vec3,
    gravity: wp.vec3,
    joint_type: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_qd: wp.array[float],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    write_body_inertia: int,
    write_body_inertia_terms: int,
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    joint_X_p: wp.array[wp.transform],
    # outputs
    joint_S_s: wp.array[wp.spatial_vector],
    body_I_s: wp.array[wp.spatial_matrix],
    body_inertia_terms: wp.array2d[float],
    body_v_s: wp.array[wp.spatial_vector],
    body_f_s: wp.array[wp.spatial_vector],
    body_a_s: wp.array[wp.spatial_vector],
):
    v_s, a_s = compute_link_kinematics(
        i,
        parent,
        child,
        parent_v_s,
        parent_a_s,
        origin,
        joint_type,
        joint_qd_start,
        joint_qd,
        joint_axis,
        joint_dof_dim,
        body_q,
        joint_X_p,
        joint_S_s,
        body_v_s,
        body_a_s,
    )

    # compute body forces
    X_sm = body_q_com[child]
    X_sm_local = wp.transform(
        wp.transform_get_translation(X_sm) - origin,
        wp.transform_get_rotation(X_sm),
    )
    mass = body_mass[child]

    # gravity and external forces (expressed in frame aligned with s but centered at body mass)
    f_g = mass * gravity
    com, inertia_origin = transform_com_inertia_terms(X_sm_local, mass, body_inertia[child])
    f_g_s = wp.spatial_vector(f_g, wp.cross(com, f_g))

    # body forces
    if write_body_inertia != 0:
        body_I_s[child] = assemble_com_spatial_inertia(mass, com, inertia_origin)
    if write_body_inertia_terms != 0:
        body_inertia_terms[child, 0] = com[0]
        body_inertia_terms[child, 1] = com[1]
        body_inertia_terms[child, 2] = com[2]
        for row in range(3):
            for col in range(3):
                body_inertia_terms[child, 3 + 3 * row + col] = inertia_origin[row, col]

    # The root's linear inertial wrench is NOT spurious: the solve frame is centred on a material
    # point of the root body, so that point accelerates as the body rotates and this term is what
    # carries it. SolverFeatherstone keeps it and conserves momentum; zeroing it here leaked
    # momentum on every rotating multi-link articulation.
    coriolis = spatial_cross_dual(v_s, mul_com_spatial_inertia(mass, com, inertia_origin, v_s))

    f_b_s = mul_com_spatial_inertia(mass, com, inertia_origin, a_s) + coriolis

    body_f_s[child] = f_b_s - f_g_s
    return v_s, a_s


@wp.kernel(module=_KINEMATICS_KERNEL_MODULE)
def eval_rigid_fk_kinematics(
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
    """Propagate articulation poses, motion subspaces, velocities, and accelerations."""
    index = wp.tid()
    start = articulation_start[index]
    end = articulation_joint_end[index]

    for i in range(start, end):
        compute_link_transform(
            i,
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

    origin = wp.vec3()
    if start < articulation_start[index + 1]:
        root_body = joint_child[start]
        if root_body >= 0:
            origin = wp.transform_point(body_q[root_body], body_com[root_body])
    articulation_origin[index] = origin

    cached_child = int(-1)
    cached_v_s = wp.spatial_vector()
    cached_a_s = wp.spatial_vector()
    for i in range(start, end):
        parent = joint_parent[i]
        child = joint_child[i]
        parent_v_s = wp.spatial_vector()
        parent_a_s = wp.spatial_vector()
        if parent >= 0:
            if parent == cached_child:
                parent_v_s = cached_v_s
                parent_a_s = cached_a_s
            else:
                parent_v_s = body_v_s[parent]
                parent_a_s = body_a_s[parent]
        cached_v_s, cached_a_s = compute_link_kinematics(
            i,
            parent,
            child,
            parent_v_s,
            parent_a_s,
            origin,
            joint_type,
            joint_qd_start,
            joint_qd,
            joint_axis,
            joint_dof_dim,
            body_q,
            joint_X_p,
            joint_S_s,
            body_v_s,
            body_a_s,
        )
        cached_child = child
    fk_id_cache_valid[index] = 1


@wp.func
def finalize_body_dynamics_values(
    body: int,
    articulation: int,
    pose: wp.transform,
    pose_com: wp.transform,
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    is_free_rigid: wp.array[int],
    origin: wp.vec3,
    materialize_all_body_inertia: int,
    materialize_body_inertia_terms: int,
    gravity: wp.array[wp.vec3],
    v_s: wp.spatial_vector,
    a_s: wp.spatial_vector,
    # outputs
    body_I_s: wp.array[wp.spatial_matrix],
    body_inertia_terms: wp.array2d[float],
    body_f_s: wp.array[wp.spatial_vector],
    body_qd: wp.array[wp.spatial_vector],
):
    """Build body dynamics and COM velocity from explicit current values."""
    if articulation < 0:
        com_world = wp.transform_point(pose, body_com[body])
        v_com = wp.spatial_top(v_s) + wp.cross(wp.spatial_bottom(v_s), com_world)
        body_qd[body] = wp.spatial_vector(v_com, wp.spatial_bottom(v_s))
        return

    X_sm = pose_com
    X_sm_local = wp.transform(
        wp.transform_get_translation(X_sm) - origin,
        wp.transform_get_rotation(X_sm),
    )
    mass = body_mass[body]
    com, inertia_origin = transform_com_inertia_terms(X_sm_local, mass, body_inertia[body])

    write_body_inertia = materialize_all_body_inertia
    if is_free_rigid[articulation] != 0:
        write_body_inertia = 1
    if write_body_inertia != 0:
        body_I_s[body] = assemble_com_spatial_inertia(mass, com, inertia_origin)
    if materialize_body_inertia_terms != 0:
        body_inertia_terms[body, 0] = com[0]
        body_inertia_terms[body, 1] = com[1]
        body_inertia_terms[body, 2] = com[2]
        for row in range(3):
            for col in range(3):
                body_inertia_terms[body, 3 + 3 * row + col] = inertia_origin[row, col]

    coriolis = spatial_cross_dual(v_s, mul_com_spatial_inertia(mass, com, inertia_origin, v_s))
    f_b_s = mul_com_spatial_inertia(mass, com, inertia_origin, a_s) + coriolis
    f_g = mass * gravity[0]
    body_f_s[body] = f_b_s - wp.spatial_vector(f_g, wp.cross(com, f_g))

    com_world = wp.transform_point(pose, body_com[body])
    com_rel = com_world - origin
    v_com = wp.spatial_top(v_s) + wp.cross(wp.spatial_bottom(v_s), com_rel)
    body_qd[body] = wp.spatial_vector(v_com, wp.spatial_bottom(v_s))


@wp.func
def finalize_body_dynamics_body(
    body: int,
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
    # outputs
    body_I_s: wp.array[wp.spatial_matrix],
    body_inertia_terms: wp.array2d[float],
    body_f_s: wp.array[wp.spatial_vector],
    body_qd: wp.array[wp.spatial_vector],
):
    """Load current canonical values and publish one body's dynamics."""
    articulation = body_to_articulation[body]
    origin = wp.vec3(0.0)
    pose_com = wp.transform_identity()
    acceleration = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
    if articulation >= 0:
        origin = articulation_origin[articulation]
        pose_com = body_q_com[body]
        acceleration = body_a_s[body]
    finalize_body_dynamics_values(
        body,
        articulation,
        body_q[body],
        pose_com,
        body_com,
        body_mass,
        body_inertia,
        is_free_rigid,
        origin,
        materialize_all_body_inertia,
        materialize_body_inertia_terms,
        gravity,
        body_v_s[body],
        acceleration,
        body_I_s,
        body_inertia_terms,
        body_f_s,
        body_qd,
    )


@wp.kernel(module=_MASS_DYNAMICS_KERNEL_MODULE)
def finalize_body_dynamics(
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
    """Build independent body dynamics and publish COM velocity in parallel."""
    finalize_body_dynamics_body(
        wp.tid(),
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


@wp.kernel(module=_KINEMATICS_KERNEL_MODULE)
def eval_rigid_fk_id(
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
    """Evaluate and cache articulation poses and inverse dynamics."""
    index = wp.tid()
    if reuse_cached != 0 and fk_id_cache_valid[index] != 0:
        return
    start = articulation_start[index]
    end = articulation_joint_end[index]

    for i in range(start, end):
        compute_link_transform(
            i,
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

    origin = wp.vec3()
    if start < articulation_start[index + 1]:
        root_body = joint_child[start]
        if root_body >= 0:
            origin = wp.transform_point(body_q[root_body], body_com[root_body])
    articulation_origin[index] = origin

    gravity_s = gravity[0]
    write_body_inertia = materialize_all_body_inertia
    if is_free_rigid[index] != 0:
        write_body_inertia = 1
    cached_child = int(-1)
    cached_v_s = wp.spatial_vector()
    cached_a_s = wp.spatial_vector()
    for i in range(start, end):
        parent = joint_parent[i]
        child = joint_child[i]
        parent_v_s = wp.spatial_vector()
        parent_a_s = wp.spatial_vector()
        if parent >= 0:
            if parent == cached_child:
                parent_v_s = cached_v_s
                parent_a_s = cached_a_s
            else:
                parent_v_s = body_v_s[parent]
                parent_a_s = body_a_s[parent]
        cached_v_s, cached_a_s = compute_link_velocity(
            i,
            parent,
            child,
            parent_v_s,
            parent_a_s,
            origin,
            gravity_s,
            joint_type,
            joint_qd_start,
            joint_qd,
            joint_axis,
            joint_dof_dim,
            body_mass,
            body_inertia,
            write_body_inertia,
            materialize_body_inertia_terms,
            body_q,
            body_q_com,
            joint_X_p,
            joint_S_s,
            body_I_s,
            body_inertia_terms,
            body_v_s,
            body_f_s,
            body_a_s,
        )
        cached_child = child
    fk_id_cache_valid[index] = 1


@cache
def _get_tree_fk_kernel(lanes: int, mode: str):
    """Build a cooperative tree traversal without replacing its per-joint mathematics.

    ``id`` and ``kinematics`` share the ordinary FK/ID signature after the six
    schedule arguments. ``public`` instead takes the public FK helper's arrays.
    Unary chains use joint-sized segments when that shortens the topology span.
    Launch complete 32-thread warps;
    CPU and differentiable execution stay serial.
    """
    if lanes not in (1, 2, 4, 8, 16, 32):
        raise ValueError("Tree traversal lanes must be a power of two between 1 and 32")
    if mode not in ("id", "kinematics", "public"):
        raise ValueError(f"Unknown tree traversal mode: {mode}")
    module = wp.Module(f"{__name__}.tree_fk_{mode}_{lanes}")

    if mode == "public":

        @wp.kernel(module=module, enable_backward=False)
        def tree_fk_public(
            group_count: int,
            max_levels: int,
            art_indices: wp.array[int],
            level_offsets: wp.array2d[int],
            segment_offsets: wp.array[int],
            segment_joints: wp.array[int],
            joint_articulation: wp.array[int],
            joint_q: wp.array[float],
            joint_qd: wp.array[float],
            joint_q_start: wp.array[int],
            joint_qd_start: wp.array[int],
            joint_type: wp.array[int],
            joint_parent: wp.array[int],
            joint_child: wp.array[int],
            joint_X_p: wp.array[wp.transform],
            joint_X_c: wp.array[wp.transform],
            joint_axis: wp.array[wp.vec3],
            joint_dof_dim: wp.array2d[int],
            body_com: wp.array[wp.vec3],
            body_flags: wp.array[wp.int32],
            body_flag_filter: int,
            body_q: wp.array[wp.transform],
            body_qd: wp.array[wp.spatial_vector],
        ):
            block, lane = wp.tid()
            group = block * (32 // lanes) + lane // lanes
            local = lane % lanes
            for level in range(max_levels):
                if group < group_count:
                    segment = level_offsets[group, level] + local
                    end = level_offsets[group, level + 1]
                    while segment < end:
                        for entry in range(segment_offsets[segment], segment_offsets[segment + 1]):
                            joint = segment_joints[entry]
                            # Retain the public FREE/D6 COM-velocity conventions.
                            eval_single_articulation_fk(
                                joint,
                                joint + 1,
                                joint_articulation,
                                joint_q,
                                joint_qd,
                                joint_q_start,
                                joint_qd_start,
                                joint_type,
                                joint_parent,
                                joint_child,
                                joint_X_p,
                                joint_X_c,
                                joint_axis,
                                joint_dof_dim,
                                body_com,
                                body_flags,
                                body_flag_filter,
                                body_q,
                                body_qd,
                            )
                        segment += lanes
                if wp.static(lanes > 1):
                    warp_sync()

        return tree_fk_public

    compute_dynamics = mode == "id"

    @wp.kernel(module=module, enable_backward=False)
    def tree_fk(
        group_count: int,
        max_levels: int,
        art_indices: wp.array[int],
        level_offsets: wp.array2d[int],
        segment_offsets: wp.array[int],
        segment_joints: wp.array[int],
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
        block, lane = wp.tid()
        group = block * (32 // lanes) + lane // lanes
        local = lane % lanes
        active = group < group_count
        articulation = int(0)
        if active:
            articulation = art_indices[group]
            if wp.static(compute_dynamics):
                if reuse_cached != 0 and fk_id_cache_valid[articulation] != 0:
                    active = False

        # The authored first root defines the common frame, including forests.
        seed_joint = int(-1)
        if active:
            start = articulation_start[articulation]
            if start < articulation_joint_end[articulation]:
                seed_joint = start
            if local == 0:
                origin = wp.vec3()
                if seed_joint >= 0:
                    compute_link_transform(
                        seed_joint,
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
                    root_body = joint_child[seed_joint]
                    if root_body >= 0:
                        origin = wp.transform_point(body_q[root_body], body_com[root_body])
                articulation_origin[articulation] = origin
        if wp.static(lanes > 1):
            warp_sync()

        origin = wp.vec3()
        write_body_inertia = materialize_all_body_inertia
        if active:
            origin = articulation_origin[articulation]
            if wp.static(compute_dynamics):
                if is_free_rigid[articulation] != 0:
                    write_body_inertia = 1

        # A joint needs only its current parent and the root frame, not descendant
        # poses. Fuse both passes and retain parent motion along each unary chain.
        for level in range(max_levels):
            if active:
                segment = level_offsets[group, level] + local
                end = level_offsets[group, level + 1]
                while segment < end:
                    begin = segment_offsets[segment]
                    finish = segment_offsets[segment + 1]
                    parent = joint_parent[segment_joints[begin]]
                    parent_v_s = wp.spatial_vector()
                    parent_a_s = wp.spatial_vector()
                    if parent >= 0:
                        parent_v_s = body_v_s[parent]
                        parent_a_s = body_a_s[parent]
                    for entry in range(begin, finish):
                        joint = segment_joints[entry]
                        parent = joint_parent[joint]
                        child = joint_child[joint]
                        if joint != seed_joint:
                            compute_link_transform(
                                joint,
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
                        if wp.static(compute_dynamics):
                            parent_v_s, parent_a_s = compute_link_velocity(
                                joint,
                                parent,
                                child,
                                parent_v_s,
                                parent_a_s,
                                origin,
                                gravity[0],
                                joint_type,
                                joint_qd_start,
                                joint_qd,
                                joint_axis,
                                joint_dof_dim,
                                body_mass,
                                body_inertia,
                                write_body_inertia,
                                materialize_body_inertia_terms,
                                body_q,
                                body_q_com,
                                joint_X_p,
                                joint_S_s,
                                body_I_s,
                                body_inertia_terms,
                                body_v_s,
                                body_f_s,
                                body_a_s,
                            )
                        else:
                            parent_v_s, parent_a_s = compute_link_kinematics(
                                joint,
                                parent,
                                child,
                                parent_v_s,
                                parent_a_s,
                                origin,
                                joint_type,
                                joint_qd_start,
                                joint_qd,
                                joint_axis,
                                joint_dof_dim,
                                body_q,
                                joint_X_p,
                                joint_S_s,
                                body_v_s,
                                body_a_s,
                            )
                    segment += lanes
            # Inactive groups still participate in every full-warp barrier.
            if wp.static(lanes > 1):
                warp_sync()
        if active and local == 0:
            fk_id_cache_valid[articulation] = 1

    return tree_fk


@wp.kernel
def refresh_masked_body_inertia(
    articulation_joint_end: wp.array[int],
    joint_articulation: wp.array[int],
    joint_child: wp.array[int],
    mass_update_mask: wp.array[int],
    body_q_com: wp.array[wp.transform],
    articulation_origin: wp.array[wp.vec3],
    body_I_m: wp.array[wp.spatial_matrix],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    write_body_inertia_terms: int,
    # outputs
    body_I_s: wp.array[wp.spatial_matrix],
    body_inertia_terms: wp.array2d[float],
):
    """Materialize current link inertias selected by a reuse-step mass update mask.

    The compact COM terms feed the direct diagonal inertia owner, which bypasses ``body_I_s``; refresh them
    under the same mask so a masked inertial update reaches every mass consumer.
    """
    joint = wp.tid()
    articulation = joint_articulation[joint]
    if articulation < 0 or joint >= articulation_joint_end[articulation] or mass_update_mask[articulation] == 0:
        return
    child = joint_child[joint]
    X_sm = body_q_com[child]
    X_sm_local = wp.transform(
        wp.transform_get_translation(X_sm) - articulation_origin[articulation],
        wp.transform_get_rotation(X_sm),
    )
    body_I_s[child] = transform_spatial_inertia(X_sm_local, body_I_m[child])
    if write_body_inertia_terms != 0:
        com, inertia_origin = transform_com_inertia_terms(X_sm_local, body_mass[child], body_inertia[child])
        body_inertia_terms[child, 0] = com[0]
        body_inertia_terms[child, 1] = com[1]
        body_inertia_terms[child, 2] = com[2]
        for row in range(3):
            for col in range(3):
                body_inertia_terms[child, 3 + 3 * row + col] = inertia_origin[row, col]


@wp.func
def _compute_body_net_wrench(
    child: int,
    f_t_s: wp.spatial_vector,
    origin: wp.vec3,
    body_fb_s: wp.array[wp.spatial_vector],
    body_f_ext: wp.array[wp.spatial_vector],
    body_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
):
    """Subtract the external COM wrench in the articulation's origin frame."""
    f_ext_com = wp.spatial_vector()
    if (body_flags[child] & BodyFlags.KINEMATIC) == 0:
        f_ext_com = body_f_ext[child]
    f_ext_f = wp.spatial_bottom(f_ext_com)
    f_ext_t = wp.spatial_top(f_ext_com)
    com_world = wp.transform_point(body_q[child], body_com[child])
    com_rel = com_world - origin
    tau_origin = f_ext_f + wp.cross(com_rel, f_ext_t)
    f_ext_origin = wp.spatial_vector(f_ext_t, tau_origin)
    return body_fb_s[child] + f_t_s - f_ext_origin


@wp.func
def accumulate_articulation_tau(
    index: int,
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
    start = articulation_start[index]
    # Tree prefix only: trailing loop-closing joints are handled as constraint rows.
    end = articulation_joint_end[index]
    count = end - start

    # compute joint forces
    for offset in range(count):
        # for backwards traversal
        i = end - offset - 1

        type = joint_type[i]
        parent = joint_parent[i]
        child = joint_child[i]
        articulation = joint_articulation[i]
        dof_start = joint_qd_start[i]
        lin_axis_count = joint_dof_dim[i, 0]
        ang_axis_count = joint_dof_dim[i, 1]
        origin = wp.vec3()
        if articulation >= 0:
            origin = articulation_origin[articulation]

        f_s = _compute_body_net_wrench(
            child, body_ft_s[child], origin, body_fb_s, body_f_ext, body_flags, body_q, body_com
        )

        # compute joint-space forces, writes out tau
        jcalc_tau(
            type,
            joint_S_s,
            joint_f,
            joint_q,
            joint_qd,
            joint_spring_stiffness,
            joint_spring_ref,
            joint_damping,
            joint_q_start[i],
            dof_start,
            lin_axis_count,
            ang_axis_count,
            f_s,
            add_existing_tau,
            tau,
        )

        if parent >= 0:
            # One thread owns the complete articulation and visits children
            # before parents, so no other thread can update this accumulator.
            body_ft_s[parent] = body_ft_s[parent] + f_s


@cache
def _get_tree_tau_kernel(lanes: int):
    """Reduce chain wrenches without atomics, preserving descending child order."""
    if lanes not in (1, 2, 4, 8, 16, 32):
        raise ValueError("Tree traversal lanes must be a power of two between 1 and 32")
    module = wp.Module(f"{__name__}.tree_tau_{lanes}")

    @wp.kernel(module=module, enable_backward=False)
    def tree_tau(
        group_count: int,
        max_levels: int,
        art_indices: wp.array[int],
        level_offsets: wp.array2d[int],
        segment_offsets: wp.array[int],
        segment_joints: wp.array[int],
        child_offsets: wp.array[int],
        child_segments: wp.array[int],
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
        body_ft_s: wp.array[wp.spatial_vector],
        tau: wp.array[float],
        segment_net: wp.array[wp.spatial_vector],
    ):
        block, lane = wp.tid()
        group = block * (32 // lanes) + lane // lanes
        local = lane % lanes
        for reverse_level in range(max_levels):
            level = max_levels - reverse_level - 1
            if group < group_count:
                segment = level_offsets[group, level] + local
                end = level_offsets[group, level + 1]
                while segment < end:
                    begin = segment_offsets[segment]
                    finish = segment_offsets[segment + 1]
                    tail_body = joint_child[segment_joints[finish - 1]]
                    f_t_s = body_ft_s[tail_body]
                    # The host orders immediate children by descending joint index,
                    # exactly matching the serial articulation's accumulation order.
                    for entry in range(child_offsets[segment], child_offsets[segment + 1]):
                        f_t_s = f_t_s + segment_net[child_segments[entry]]
                    f_s = wp.spatial_vector()
                    for offset in range(finish - begin):
                        joint = segment_joints[finish - offset - 1]
                        child = joint_child[joint]
                        if offset > 0:
                            f_t_s = body_ft_s[child] + f_s
                        body_ft_s[child] = f_t_s
                        articulation = joint_articulation[joint]
                        origin = wp.vec3()
                        if articulation >= 0:
                            origin = articulation_origin[articulation]
                        f_s = _compute_body_net_wrench(
                            child, f_t_s, origin, body_fb_s, body_f_ext, body_flags, body_q, body_com
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
                            f_s,
                            add_existing_tau,
                            tau,
                        )
                    segment_net[segment] = f_s
                    segment += lanes
            # Tau is never skipped on FK cache hits: controls and wrenches can change.
            if wp.static(lanes > 1):
                warp_sync()

    return tree_tau


@wp.kernel(module=_INVERSE_DYNAMICS_KERNEL_MODULE)
def eval_rigid_tau(
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
    # outputs
    body_ft_s: wp.array[wp.spatial_vector],
    tau: wp.array[float],
):
    # one thread per articulation
    accumulate_articulation_tau(
        wp.tid(),
        articulation_start,
        articulation_joint_end,
        joint_type,
        joint_parent,
        joint_child,
        joint_articulation,
        joint_qd_start,
        joint_q_start,
        joint_dof_dim,
        joint_f,
        joint_q,
        joint_qd,
        joint_spring_stiffness,
        joint_spring_ref,
        joint_damping,
        joint_S_s,
        body_fb_s,
        body_f_ext,
        body_flags,
        body_q,
        body_com,
        articulation_origin,
        0,
        body_ft_s,
        tau,
    )


@wp.kernel(module=_INVERSE_DYNAMICS_KERNEL_MODULE)
def eval_rigid_tau_add(
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
    # outputs
    body_ft_s: wp.array[wp.spatial_vector],
    tau: wp.array[float],
):
    """Accumulate inverse dynamics onto a previously prepared drive-force bucket."""
    accumulate_articulation_tau(
        wp.tid(),
        articulation_start,
        articulation_joint_end,
        joint_type,
        joint_parent,
        joint_child,
        joint_articulation,
        joint_qd_start,
        joint_q_start,
        joint_dof_dim,
        joint_f,
        joint_q,
        joint_qd,
        joint_spring_stiffness,
        joint_spring_ref,
        joint_damping,
        joint_S_s,
        body_fb_s,
        body_f_ext,
        body_flags,
        body_q,
        body_com,
        articulation_origin,
        1,
        body_ft_s,
        tau,
    )


@wp.kernel(module=_MASS_DYNAMICS_KERNEL_MODULE)
def compute_composite_inertia(
    articulation_start: wp.array[int],
    articulation_joint_end: wp.array[int],
    mass_update_mask: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_child: wp.array[int],
    body_I_s: wp.array[wp.spatial_matrix],
    # outputs
    body_I_c: wp.array[wp.spatial_matrix],
):
    art_idx = wp.tid()

    if mass_update_mask[art_idx] == 0:
        return

    start = articulation_start[art_idx]
    # Tree prefix only: trailing loop-closing joints carry no link inertia.
    end = articulation_joint_end[art_idx]
    count = end - start

    # body_I_s/body_I_c are BODY-indexed (see compute_link_velocity); index them through
    # joint_child. Joint index and child body index only coincide in loop-free models —
    # a loop joint shifts every later joint index off its body row.
    for i in range(count):
        body_I_c[joint_child[start + i]] = body_I_s[joint_child[start + i]]

    for i in range(count - 1, -1, -1):
        joint_i = start + i
        parent_joint = joint_ancestor[joint_i]

        if parent_joint >= start:
            body_I_c[joint_child[parent_joint]] += body_I_c[joint_child[joint_i]]


@wp.func
def dense_cholesky(
    n: int,
    A: wp.array[float],
    R: wp.array[float],
    A_start: int,
    R_start: int,
    # outputs
    L: wp.array[float],
):
    # compute the Cholesky factorization of A = L L^T with diagonal regularization R
    for j in range(n):
        s = A[A_start + dense_index(n, j, j)] + R[R_start + j]

        for k in range(j):
            r = L[A_start + dense_index(n, j, k)]
            s -= r * r

        s = wp.sqrt(s)
        invS = 1.0 / s

        L[A_start + dense_index(n, j, j)] = s

        for i in range(j + 1, n):
            s = A[A_start + dense_index(n, i, j)]

            for k in range(j):
                s -= L[A_start + dense_index(n, i, k)] * L[A_start + dense_index(n, j, k)]

            L[A_start + dense_index(n, i, j)] = s * invS


@wp.kernel
def cholesky_loop(
    H_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
    R_group: wp.array2d[float],  # [n_arts, n_dofs]
    group_to_art: wp.array[int],
    mass_update_mask: wp.array[int],
    n_dofs: int,
    # output
    L_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
):
    """Non-tiled Cholesky for grouped articulation storage.

    One thread per articulation, loop-based Cholesky decomposition.
    Efficient for small articulations where tile overhead dominates.
    """
    group_idx = wp.tid()
    art_idx = group_to_art[group_idx]

    if mass_update_mask[art_idx] == 0:
        return

    # Cholesky decomposition with regularization: L L^T = H + diag(R)
    for j in range(n_dofs):
        # Compute diagonal element L[j,j]
        s = H_group[group_idx, j, j] + R_group[group_idx, j]

        for k in range(j):
            r = L_group[group_idx, j, k]
            s -= r * r

        s = wp.sqrt(s)
        inv_s = 1.0 / s
        L_group[group_idx, j, j] = s

        # Compute off-diagonal elements L[i,j] for i > j
        for i in range(j + 1, n_dofs):
            s = H_group[group_idx, i, j]

            for k in range(j):
                s -= L_group[group_idx, i, k] * L_group[group_idx, j, k]

            L_group[group_idx, i, j] = s * inv_s


@wp.kernel(module=_MASS_DYNAMICS_KERNEL_MODULE)
def factor_diagonal_mass(
    H_group: wp.array3d[float],
    R_group: wp.array2d[float],
    group_to_art: wp.array[int],
    mass_update_mask: wp.array[int],
    n_dofs: int,
    # output
    L_group: wp.array3d[float],
):
    """Factor structurally diagonal generalized mass matrices."""
    element = wp.tid()
    group = element // n_dofs
    dof = element - group * n_dofs
    art = group_to_art[group]
    if mass_update_mask[art] != 0:
        mass = H_group[group, dof, dof] + R_group[group, dof]
        L_group[group, dof, dof] = wp.sqrt(mass)


@wp.kernel(module=_MASS_DYNAMICS_KERNEL_MODULE)
def compute_compact_diagonal_inverse_mass(
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    mass_update_mask: wp.array[int],
    joint_child: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    body_I_c: wp.array[wp.spatial_matrix],
    group_to_art: wp.array[int],
    dof_joint_offset: wp.array[int],
    armature: wp.array2d[float],
    drive_row_by_dof: wp.array[int],
    drive_row_K: wp.array[float],
    n_dofs: int,
    # output
    diagonal_inverse_mass: wp.array[float],
):
    """Build an inverse diagonal directly from independent articulation branches."""
    element = wp.tid()
    group = element // n_dofs
    dof = element - group * n_dofs
    art = group_to_art[group]
    if mass_update_mask[art] == 0:
        return

    global_dof = articulation_dof_start[art] + dof
    joint = articulation_start[art] + dof_joint_offset[dof]
    motion = joint_S_s[global_dof]
    mass = wp.dot(motion, body_I_c[joint_child[joint]] * motion) + armature[group, dof]
    drive_row = drive_row_by_dof[global_dof]
    if drive_row >= 0:
        stiffness = drive_row_K[drive_row]
        if stiffness > 0.0:
            mass += stiffness
    diagonal_inverse_mass[global_dof] = 1.0 / mass


@wp.kernel(module=_MASS_DYNAMICS_KERNEL_MODULE)
def prepare_fused_diagonal_joint_limits(
    world_dof_indices: wp.array2d[int],
    max_world_dofs: int,
    fused_limit_dof_mask: wp.array[int],
    limit_q_index: wp.array[int],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_q: wp.array[float],
    activation_gap: float,
    pgs_beta: float,
    dt: float,
    # outputs
    active_sides: wp.array2d[int],
    lower_rhs: wp.array2d[float],
    upper_rhs: wp.array2d[float],
):
    """Prepare independent diagonal-articulation limit projections per world coordinate."""
    element = wp.tid()
    world = element // max_world_dofs
    local_dof = element - world * max_world_dofs
    global_dof = world_dof_indices[world, local_dof]

    active = int(0)
    rhs_lower = float(0.0)
    rhs_upper = float(0.0)
    if global_dof >= 0 and fused_limit_dof_mask[global_dof] != 0:
        q_index = limit_q_index[global_dof]
        if q_index >= 0:
            q = joint_q[q_index]
            lower = joint_limit_lower[global_dof]
            upper = joint_limit_upper[global_dof]
            inv_dt = 1.0 / dt
            lower_active, lower_phi, _lower_sign = joint_limit_side(q, lower, upper, 0, activation_gap)
            if lower_active:
                rhs_lower = limit_rhs(lower_phi, pgs_beta, 0.0, inv_dt, 1.0, 1.0)
                active |= 1
            upper_active, upper_phi, _upper_sign = joint_limit_side(q, lower, upper, 1, activation_gap)
            if upper_active:
                rhs_upper = limit_rhs(upper_phi, pgs_beta, 0.0, inv_dt, 1.0, 1.0)
                active |= 2

    active_sides[world, local_dof] = active
    lower_rhs[world, local_dof] = rhs_lower
    upper_rhs[world, local_dof] = rhs_upper


@wp.kernel(module=_MASS_DYNAMICS_KERNEL_MODULE)
def solve_diagonal_mass(
    L_group: wp.array3d[float],
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    n_dofs: int,
    joint_tau: wp.array[float],
    # output
    joint_qdd: wp.array[float],
):
    """Solve a structurally diagonal generalized mass system."""
    element = wp.tid()
    group = element // n_dofs
    dof = element - group * n_dofs
    art = group_to_art[group]
    global_dof = articulation_dof_start[art] + dof
    diagonal = L_group[group, dof, dof]
    value = joint_tau[global_dof] / diagonal
    joint_qdd[global_dof] = value / diagonal


@wp.kernel(module=_MASS_DYNAMICS_KERNEL_MODULE)
def solve_compact_diagonal_mass(
    diagonal_inverse_mass: wp.array[float],
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    n_dofs: int,
    joint_tau: wp.array[float],
    # output
    joint_qdd: wp.array[float],
):
    """Apply a directly stored inverse diagonal to generalized forces."""
    element = wp.tid()
    group = element // n_dofs
    dof = element - group * n_dofs
    art = group_to_art[group]
    global_dof = articulation_dof_start[art] + dof
    joint_qdd[global_dof] = joint_tau[global_dof] * diagonal_inverse_mass[global_dof]


@wp.func
def _active_free_root_dof_start(
    free_root_joint_indices: wp.array[int],
    joint_qd_start: wp.array[int],
    kinematic_joint_mask: wp.array[int],
    root_index: int,
):
    """DOF start of an active free/distance root, or ``-1`` when kinematic.

    Structural eligibility is precomputed once; the device-side kinematic
    check remains live so model-property notifications and graph replay keep
    the same launch topology. Eligibility never depends on the numeric value
    of ``omega x v`` because its forward value can be zero while its derivative
    is not. Each root owns all six of its DOFs, so writing its three linear
    entries unconditionally races with nobody.
    """
    joint = free_root_joint_indices[root_index]
    if kinematic_joint_mask[joint] != 0:
        return -1
    return joint_qd_start[joint]


@wp.kernel
def apply_free_root_transport_to_predictor(
    free_root_joint_indices: wp.array[int],
    joint_qd_start: wp.array[int],
    kinematic_joint_mask: wp.array[int],
    joint_qd: wp.array[float],
    dt: float,
    # in/out
    v_hat: wp.array[float],
):
    """Lift the free root's velocity predictor onto the integrator's convention.

    ``jcalc_integrate`` realizes ``qd + (qdd + omega x v) * dt`` for the root's
    linear coordinate. Constraint rows are built against ``v_hat``, so without
    the same term here every contact, friction, and velocity-limit row sees a
    COM velocity the integrator never produces, off by ``dt * (omega x v)``.
    """
    root_index = wp.tid()
    d = _active_free_root_dof_start(free_root_joint_indices, joint_qd_start, kinematic_joint_mask, root_index)
    if d < 0:
        return
    v = wp.vec3(joint_qd[d + 0], joint_qd[d + 1], joint_qd[d + 2])
    w = wp.vec3(joint_qd[d + 3], joint_qd[d + 4], joint_qd[d + 5])
    c = wp.cross(w, v)
    v_hat[d + 0] = v_hat[d + 0] + c[0] * dt
    v_hat[d + 1] = v_hat[d + 1] + c[1] * dt
    v_hat[d + 2] = v_hat[d + 2] + c[2] * dt


@wp.func
def _gyro_skew(v: wp.vec3):
    return wp.mat33(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0)


@wp.func
def _gyroscopic_velocity(inertia: wp.mat33, effective_inertia: wp.mat33, omega: wp.vec3, predicted: wp.vec3, dt: float):
    """Replace the explicit gyroscopic kick with energy-preserving Cayley updates.

    Every solve has the form ``(A-S) w = (A+S) u``, with symmetric positive
    definite ``A`` and skew ``S``. Thus ``w.T A w == u.T A u`` independently
    of fixed-point convergence. Updating S from the midpoint approximates
    implicit midpoint without an unconverged Newton step injecting energy.
    External torque remains in u; only the gyroscopic kick is replaced.
    """
    scale = wp.max(effective_inertia[0, 0], wp.max(effective_inertia[1, 1], effective_inertia[2, 2]))
    a = effective_inertia / scale
    physical = inertia / scale
    inverse = wp.inverse(a)
    u = predicted + dt * (inverse * wp.cross(omega, physical * omega))
    # An energy bound on angular speed chooses inexpensive local gyro
    # microsteps. Geometry and the constraint solver still run once per step.
    # The fixed cap bounds work; energy preservation does not depend on it.
    speed_bound = wp.sqrt(wp.max(wp.dot(u, a * u) * wp.trace(inverse), 0.0))
    microsteps = wp.int32(wp.clamp(wp.ceil(2.0 * wp.abs(dt) * speed_bound), 1.0, 32.0))
    h = dt / float(microsteps)
    w = u
    for _ in range(microsteps):
        u = w
        energy = wp.dot(u, a * u)
        for _iteration in range(8):
            s = (0.5 * h) * _gyro_skew(physical * (0.5 * (u + w)))
            candidate = wp.inverse(a - s) * ((a + s) * u)
            delta = candidate - w
            w = candidate
            if wp.dot(delta, a * delta) <= 1.0e-12 * energy:
                break
    return w


@wp.func
def _light_apply_free_root_velocity_corrections(
    root_index: int,
    free_root_joint_indices: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_child: wp.array[int],
    body_to_articulation: wp.array[int],
    is_free_rigid: wp.array[int],
    art_group_index: wp.array[int],
    kinematic_joint_mask: wp.array[int],
    body_q: wp.array[wp.transform],
    body_inertia: wp.array[wp.mat33],
    cholesky: wp.array3d[float],
    joint_qd: wp.array[float],
    dt: float,
    requires_grad: bool,
    apply_transport: bool,
    v_hat: wp.array[float],
):
    """Fuse free-root transport with the isolated rigid-body gyroscopic update."""
    d = _active_free_root_dof_start(free_root_joint_indices, joint_qd_start, kinematic_joint_mask, root_index)
    if d < 0:
        return
    w = wp.vec3(joint_qd[d + 3], joint_qd[d + 4], joint_qd[d + 5])
    if apply_transport:
        v = wp.vec3(joint_qd[d + 0], joint_qd[d + 1], joint_qd[d + 2])
        c = wp.cross(w, v)
        v_hat[d + 0] = v_hat[d + 0] + c[0] * dt
        v_hat[d + 1] = v_hat[d + 1] + c[1] * dt
        v_hat[d + 2] = v_hat[d + 2] + c[2] * dt
    body = joint_child[free_root_joint_indices[root_index]]
    art = body_to_articulation[body]
    if is_free_rigid[art] == 0:
        return
    predicted_world = wp.vec3(v_hat[d + 3], v_hat[d + 4], v_hat[d + 5])
    if not requires_grad:
        if (
            w[0] == 0.0
            and w[1] == 0.0
            and (w[2] == 0.0)
            and (predicted_world[0] == 0.0)
            and (predicted_world[1] == 0.0)
            and (predicted_world[2] == 0.0)
        ):
            return
    inertia = body_inertia[body]
    if not requires_grad:
        if (
            inertia[0, 0] == inertia[1, 1]
            and inertia[1, 1] == inertia[2, 2]
            and (inertia[0, 1] == 0.0)
            and (inertia[0, 2] == 0.0)
            and (inertia[1, 0] == 0.0)
            and (inertia[1, 2] == 0.0)
            and (inertia[2, 0] == 0.0)
            and (inertia[2, 1] == 0.0)
        ):
            return
    group = art_group_index[art]
    lower = wp.mat33(0.0)
    for r in range(3):
        for c in range(r + 1):
            lower[r, c] = cholesky[group, r + 3, c + 3]
    rotation = wp.transform_get_rotation(body_q[body])
    basis = wp.quat_to_matrix(rotation)
    effective = wp.transpose(basis) * (lower * wp.transpose(lower)) * basis
    omega = wp.quat_rotate_inv(rotation, w)
    predicted = wp.quat_rotate_inv(rotation, predicted_world)
    corrected = wp.quat_rotate(rotation, _gyroscopic_velocity(inertia, effective, omega, predicted, dt))
    for k in range(3):
        v_hat[d + 3 + k] = corrected[k]


@wp.kernel
def apply_free_root_velocity_corrections(
    free_root_joint_indices: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_child: wp.array[int],
    body_to_articulation: wp.array[int],
    is_free_rigid: wp.array[int],
    art_group_index: wp.array[int],
    kinematic_joint_mask: wp.array[int],
    body_q: wp.array[wp.transform],
    body_inertia: wp.array[wp.mat33],
    cholesky: wp.array3d[float],
    joint_qd: wp.array[float],
    dt: float,
    requires_grad: bool,
    apply_transport: bool,
    v_hat: wp.array[float],
):
    _light_apply_free_root_velocity_corrections(
        wp.tid(),
        free_root_joint_indices,
        joint_qd_start,
        joint_child,
        body_to_articulation,
        is_free_rigid,
        art_group_index,
        kinematic_joint_mask,
        body_q,
        body_inertia,
        cholesky,
        joint_qd,
        dt,
        requires_grad,
        apply_transport,
        v_hat,
    )


@wp.func
def _light_remove_free_root_transport_from_qdd(
    root_index: int,
    free_root_joint_indices: wp.array[int],
    joint_qd_start: wp.array[int],
    kinematic_joint_mask: wp.array[int],
    joint_qd: wp.array[float],
    joint_qdd: wp.array[float],
):
    """Make ``jcalc_integrate`` reproduce the solved velocity exactly.

    The solver commits to ``v_out``; ``qdd = (v_out - qd) / dt`` alone would let
    the integrator's transport term push the realized root velocity to
    ``v_out + dt * (omega x v)``. Subtracting the term here closes the loop, and
    in the contact-free case recovers the dynamics' own ``qdd`` bit for bit.
    """
    d = _active_free_root_dof_start(free_root_joint_indices, joint_qd_start, kinematic_joint_mask, root_index)
    if d < 0:
        return
    v = wp.vec3(joint_qd[d + 0], joint_qd[d + 1], joint_qd[d + 2])
    w = wp.vec3(joint_qd[d + 3], joint_qd[d + 4], joint_qd[d + 5])
    c = wp.cross(w, v)
    joint_qdd[d + 0] = joint_qdd[d + 0] - c[0]
    joint_qdd[d + 1] = joint_qdd[d + 1] - c[1]
    joint_qdd[d + 2] = joint_qdd[d + 2] - c[2]


@wp.kernel
def remove_free_root_transport_from_qdd(
    free_root_joint_indices: wp.array[int],
    joint_qd_start: wp.array[int],
    kinematic_joint_mask: wp.array[int],
    joint_qd: wp.array[float],
    joint_qdd: wp.array[float],
):
    _light_remove_free_root_transport_from_qdd(
        wp.tid(), free_root_joint_indices, joint_qd_start, kinematic_joint_mask, joint_qd, joint_qdd
    )


@wp.func
def _light_integrate_generalized_joints(
    index: int,
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
    type = joint_type[index]
    parent = joint_parent[index]
    child = joint_child[index]
    coord_start = joint_q_start[index]
    dof_start = joint_qd_start[index]
    if kinematic_joint_mask[index] != 0:
        for coord in range(coord_start, joint_q_start[index + 1]):
            joint_q_new[coord] = joint_q[coord]
        for dof in range(dof_start, joint_qd_start[index + 1]):
            joint_qd_new[dof] = joint_qd[dof]
        return
    lin_axis_count = joint_dof_dim[index, 0]
    ang_axis_count = joint_dof_dim[index, 1]
    jcalc_integrate(
        type,
        child,
        body_com,
        joint_X_c[index],
        joint_q,
        joint_qd,
        joint_qdd,
        coord_start,
        dof_start,
        lin_axis_count,
        ang_axis_count,
        dt,
        angular_damping,
        parent,
        joint_q_new,
        joint_qd_new,
    )


@wp.kernel
def integrate_generalized_joints(
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
    _light_integrate_generalized_joints(
        wp.tid(),
        joint_type,
        joint_parent,
        joint_child,
        joint_q_start,
        joint_qd_start,
        kinematic_joint_mask,
        joint_dof_dim,
        body_com,
        joint_X_c,
        joint_q,
        joint_qd,
        joint_qdd,
        dt,
        angular_damping,
        joint_q_new,
        joint_qd_new,
    )


@wp.kernel
def compute_velocity_predictor(
    joint_qd: wp.array[float],
    kinematic_dof_mask: wp.array[int],
    dt: float,
    # in/out
    joint_qdd: wp.array[float],
    # outputs
    v_hat: wp.array[float],
):
    tid = wp.tid()
    if kinematic_dof_mask[tid] != 0:
        joint_qdd[tid] = 0.0
    v_hat[tid] = joint_qd[tid] + joint_qdd[tid] * dt


@wp.func
def _light_update_qdd_from_velocity(
    tid: int,
    joint_qd: wp.array[float],
    kinematic_dof_mask: wp.array[int],
    inv_dt: float,
    v_new: wp.array[float],
    joint_qdd: wp.array[float],
):
    if kinematic_dof_mask[tid] != 0:
        v_new[tid] = joint_qd[tid]
        joint_qdd[tid] = 0.0
    else:
        joint_qdd[tid] = (v_new[tid] - joint_qd[tid]) * inv_dt


@wp.kernel
def update_qdd_from_velocity(
    joint_qd: wp.array[float],
    kinematic_dof_mask: wp.array[int],
    inv_dt: float,
    v_new: wp.array[float],
    joint_qdd: wp.array[float],
):
    _light_update_qdd_from_velocity(wp.tid(), joint_qd, kinematic_dof_mask, inv_dt, v_new, joint_qdd)


@wp.func
def _contact_linear_force(
    c: int,
    contact_count: wp.array[wp.int32],
    contact_normal: wp.array[wp.vec3],
    contact_world: wp.array[wp.int32],
    contact_slot: wp.array[wp.int32],
    contact_path: wp.array[wp.int32],
    world_impulses: wp.array2d[wp.float32],
    mf_impulses: wp.array2d[wp.float32],
    world_constraint_count: wp.array[wp.int32],
    mf_constraint_count: wp.array[wp.int32],
    world_row_type: wp.array2d[wp.int32],
    world_row_parent: wp.array2d[wp.int32],
    mf_row_type: wp.array2d[wp.int32],
    mf_row_parent: wp.array2d[wp.int32],
    enable_friction: int,
    inv_dt: float,
    # outputs
    rigid_contact_force: wp.array[wp.vec3],
):
    """Convert solved FeatherPGS contact impulses into world-frame forces."""
    total_contacts = contact_count[0]
    if c >= total_contacts:
        return

    force = wp.vec3(0.0)
    slot = contact_slot[c]
    path = contact_path[c]

    if slot >= 0 and path >= 0 and inv_dt > 0.0:
        world = contact_world[c]
        # Contacts store normals from shape 0 toward shape 1 (A-to-B). FeatherPGS
        # solves along the opposite direction internally, which corresponds to the
        # force on shape/body 0 from shape/body 1.
        normal = -contact_normal[c]

        lam_n = 0.0
        lam_t0 = 0.0
        lam_t1 = 0.0
        if path == 0:
            lam_n = world_impulses[world, slot]
            count = world_constraint_count[world]
            if (
                enable_friction != 0
                and slot + 2 < count
                and world_row_type[world, slot + 1] == PGS_CONSTRAINT_TYPE_FRICTION
                and world_row_parent[world, slot + 1] == slot
                and world_row_type[world, slot + 2] == PGS_CONSTRAINT_TYPE_FRICTION
                and world_row_parent[world, slot + 2] == slot
            ):
                lam_t0 = world_impulses[world, slot + 1]
                lam_t1 = world_impulses[world, slot + 2]
        elif path == 1:
            lam_n = mf_impulses[world, slot]
            count = mf_constraint_count[world]
            if (
                enable_friction != 0
                and slot + 2 < count
                and mf_row_type[world, slot + 1] == PGS_CONSTRAINT_TYPE_FRICTION
                and mf_row_parent[world, slot + 1] == slot
                and mf_row_type[world, slot + 2] == PGS_CONSTRAINT_TYPE_FRICTION
                and mf_row_parent[world, slot + 2] == slot
            ):
                lam_t0 = mf_impulses[world, slot + 1]
                lam_t1 = mf_impulses[world, slot + 2]

        force = lam_n * normal
        if enable_friction != 0:
            tangent0, tangent1 = contact_tangent_basis(normal)
            force += lam_t0 * tangent0 + lam_t1 * tangent1
        force *= inv_dt

    rigid_contact_force[c] = force


@wp.kernel
def compute_contact_linear_force_from_impulses(
    contact_count: wp.array[wp.int32],
    contact_normal: wp.array[wp.vec3],
    contact_world: wp.array[wp.int32],
    contact_slot: wp.array[wp.int32],
    contact_path: wp.array[wp.int32],
    world_impulses: wp.array2d[wp.float32],
    mf_impulses: wp.array2d[wp.float32],
    world_constraint_count: wp.array[wp.int32],
    mf_constraint_count: wp.array[wp.int32],
    world_row_type: wp.array2d[wp.int32],
    world_row_parent: wp.array2d[wp.int32],
    mf_row_type: wp.array2d[wp.int32],
    mf_row_parent: wp.array2d[wp.int32],
    enable_friction: int,
    inv_dt: float,
    # outputs
    rigid_contact_force: wp.array[wp.vec3],
):
    "Convert solved FeatherPGS contact impulses into world-frame forces."
    _contact_linear_force(
        wp.tid(),
        contact_count,
        contact_normal,
        contact_world,
        contact_slot,
        contact_path,
        world_impulses,
        mf_impulses,
        world_constraint_count,
        mf_constraint_count,
        world_row_type,
        world_row_parent,
        mf_row_type,
        mf_row_parent,
        enable_friction,
        inv_dt,
        rigid_contact_force,
    )


@wp.func
def _pack_contact_linear_force(
    c: int,
    contact_count: wp.array[wp.int32],
    rigid_contact_force: wp.array[wp.vec3],
    # outputs
    contact_force: wp.array[wp.spatial_vector],
):
    """Pack linear contact forces into Newton's spatial-force contact buffer."""
    total_contacts = contact_count[0]
    if c >= total_contacts:
        return

    contact_force[c] = wp.spatial_vector(rigid_contact_force[c], wp.vec3(0.0))


@wp.kernel
def pack_contact_linear_force_as_spatial(
    contact_count: wp.array[wp.int32],
    rigid_contact_force: wp.array[wp.vec3],
    # outputs
    contact_force: wp.array[wp.spatial_vector],
):
    "Pack linear contact forces into Newton's spatial-force contact buffer."
    _pack_contact_linear_force(
        wp.tid(),
        contact_count,
        rigid_contact_force,
        contact_force,
    )


@wp.func
def prepare_articulation_augmented_drives(
    articulation: int,
    articulation_start: wp.array[int],
    articulation_H_rows: wp.array[int],
    joint_type: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q_start: wp.array[int],
    joint_target_q_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_pos: wp.array[float],
    joint_target_vel: wp.array[float],
    joint_effort_limit: wp.array[float],
    max_dofs: int,
    dt: float,
    reset_tau: int,
    # outputs
    row_counts: wp.array[int],
    row_dof_index: wp.array[int],
    row_K: wp.array[float],
    tau: wp.array[float],
):
    """Prepare one articulation's implicit drive rows and explicit force."""
    joint_start = articulation_start[articulation]
    joint_end = articulation_start[articulation + 1]
    if reset_tau != 0:
        dof_start = joint_qd_start[joint_start]
        dof_end = dof_start + articulation_H_rows[articulation]
        for dof_index in range(dof_start, dof_end):
            tau[dof_index] = 0.0

    if articulation_H_rows[articulation] == 0:
        row_counts[articulation] = 0
        return

    slot = int(0)
    for joint_index in range(joint_start, joint_end):
        type = joint_type[joint_index]
        if type != JointType.PRISMATIC and type != JointType.REVOLUTE and type != JointType.D6:
            continue

        axis_count = joint_dof_dim[joint_index, 0] + joint_dof_dim[joint_index, 1]
        qd_start = joint_qd_start[joint_index]
        coord_start = joint_q_start[joint_index]
        target_start = joint_target_q_start[joint_index]
        for axis in range(axis_count):
            if slot >= max_dofs:
                break
            dof_index = qd_start + axis
            ke = joint_target_ke[dof_index]
            kd = joint_target_kd[dof_index]
            if ke <= 0.0 and kd <= 0.0:
                continue

            K = ke * dt * dt + kd * dt
            if K <= 0.0:
                continue

            q = joint_q[coord_start + axis]
            qd = joint_qd[dof_index]
            u0 = -(ke * (q - joint_target_pos[target_start + axis] + dt * qd) + kd * (qd - joint_target_vel[dof_index]))
            effort_limit = joint_effort_limit[dof_index]
            if effort_limit > 0.0:
                u0 = wp.clamp(u0, -effort_limit, effort_limit)

            row_index = articulation * max_dofs + slot
            row_dof_index[row_index] = dof_index
            row_K[row_index] = K
            if reset_tau != 0:
                tau[dof_index] = u0
            else:
                tau[dof_index] = tau[dof_index] + u0
            slot += 1
            if slot >= max_dofs:
                break

    row_counts[articulation] = slot


@wp.kernel(module=_INVERSE_DYNAMICS_KERNEL_MODULE)
def eval_rigid_tau_and_augmented_drives(
    articulation_start: wp.array[int],
    articulation_joint_end: wp.array[int],
    articulation_H_rows: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_articulation: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q_start: wp.array[int],
    joint_target_q_start: wp.array[int],
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
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_pos: wp.array[float],
    joint_target_vel: wp.array[float],
    joint_effort_limit: wp.array[float],
    max_dofs: int,
    dt: float,
    # outputs
    body_ft_s: wp.array[wp.spatial_vector],
    row_counts: wp.array[int],
    row_dof_index: wp.array[int],
    row_K: wp.array[float],
    tau: wp.array[float],
):
    """Accumulate articulation forces and augmented drives in one launch."""
    articulation = wp.tid()
    accumulate_articulation_tau(
        articulation,
        articulation_start,
        articulation_joint_end,
        joint_type,
        joint_parent,
        joint_child,
        joint_articulation,
        joint_qd_start,
        joint_q_start,
        joint_dof_dim,
        joint_f,
        joint_q,
        joint_qd,
        joint_spring_stiffness,
        joint_spring_ref,
        joint_damping,
        joint_S_s,
        body_fb_s,
        body_f_ext,
        body_flags,
        body_q,
        body_com,
        articulation_origin,
        0,
        body_ft_s,
        tau,
    )

    prepare_articulation_augmented_drives(
        articulation,
        articulation_start,
        articulation_H_rows,
        joint_type,
        joint_qd_start,
        joint_q_start,
        joint_target_q_start,
        joint_dof_dim,
        joint_q,
        joint_qd,
        joint_target_ke,
        joint_target_kd,
        joint_target_pos,
        joint_target_vel,
        joint_effort_limit,
        max_dofs,
        dt,
        0,
        row_counts,
        row_dof_index,
        row_K,
        tau,
    )


@wp.kernel(module=_INVERSE_DYNAMICS_KERNEL_MODULE)
def prepare_augmented_joint_drives(
    articulation_start: wp.array[int],
    articulation_H_rows: wp.array[int],
    joint_type: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q_start: wp.array[int],
    joint_target_q_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_pos: wp.array[float],
    joint_target_vel: wp.array[float],
    joint_effort_limit: wp.array[float],
    max_dofs: int,
    dt: float,
    # outputs
    row_counts: wp.array[int],
    row_dof_index: wp.array[int],
    row_K: wp.array[float],
    drive_tau: wp.array[float],
):
    """Prepare implicit drive rows and their explicit force independently of inverse dynamics."""
    prepare_articulation_augmented_drives(
        wp.tid(),
        articulation_start,
        articulation_H_rows,
        joint_type,
        joint_qd_start,
        joint_q_start,
        joint_target_q_start,
        joint_dof_dim,
        joint_q,
        joint_qd,
        joint_target_ke,
        joint_target_kd,
        joint_target_pos,
        joint_target_vel,
        joint_effort_limit,
        max_dofs,
        dt,
        1,
        row_counts,
        row_dof_index,
        row_K,
        drive_tau,
    )


@wp.kernel
def allocate_physx_drive_slots(
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    articulation_H_rows: wp.array[int],
    joint_type: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    art_to_world: wp.array[int],
    max_constraints: int,
    # outputs
    drive_slot: wp.array[int],
    world_slot_counter: wp.array[int],
    first_rejected_slot: wp.array[int],
):
    """Allocate one dense PhysX-style drive row for each driven scalar DOF."""
    art = wp.tid()
    world = art_to_world[art]

    dof_base = articulation_dof_start[art]
    dof_count = articulation_H_rows[art]
    for d in range(dof_count):
        drive_slot[dof_base + d] = -1

    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]

    for j in range(joint_start, joint_end):
        jtype = joint_type[j]
        if jtype != JointType.PRISMATIC and jtype != JointType.REVOLUTE and jtype != JointType.D6:
            continue

        lin_count = joint_dof_dim[j, 0]
        ang_count = joint_dof_dim[j, 1]
        axis_count = lin_count + ang_count
        qd_start = joint_qd_start[j]

        for axis in range(axis_count):
            dof = qd_start + axis
            stiffness = joint_target_ke[dof]
            damping = joint_target_kd[dof]
            if stiffness <= 0.0 and damping <= 0.0:
                continue

            drive_slot[dof] = reserve(world_slot_counter, world, 1, max_constraints, first_rejected_slot)


@wp.kernel
def clear_grouped_jacobian_active_rows(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    world_constraint_count: wp.array[int],
    dof_count: int,
    max_constraints: int,
    # outputs
    J_group: wp.array3d[float],
):
    """Clear only Jacobian rows that were active in the completed solve.

    The launch assigns one warp to each articulation. Packing those warps in
    ordinary CUDA blocks avoids the one-block-per-world scheduling overhead,
    while lanes clear the compact active prefix cooperatively.
    """
    tid = wp.tid()
    group_idx = tid // 32
    lane = tid % 32
    if group_idx >= group_to_art.shape[0]:
        return

    art = group_to_art[group_idx]
    world = art_to_world[art]
    row_count = wp.min(world_constraint_count[world], max_constraints)
    element_count = row_count * dof_count
    for element in range(lane, element_count, 32):
        row = element // dof_count
        dof = element - row * dof_count
        J_group[group_idx, row, dof] = 0.0


@wp.kernel
def populate_physx_drive_J_for_size(
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    joint_type: wp.array[int],
    joint_q_start: wp.array[int],
    joint_target_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_effort_limit: wp.array[float],
    joint_q: wp.array[float],
    joint_target_pos: wp.array[float],
    joint_target_vel: wp.array[float],
    joint_velocity_limit: wp.array[float],
    fuse_vel_limits: int,
    art_to_world: wp.array[int],
    drive_slot: wp.array[int],
    group_to_art: wp.array[int],
    # outputs
    J_group: wp.array3d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_cfm: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
    world_drive_stiffness: wp.array2d[float],
    world_drive_damping: wp.array2d[float],
    world_drive_geom_error: wp.array2d[float],
    world_drive_max_force: wp.array2d[float],
    world_drive_vel_limit: wp.array2d[float],
):
    """Populate PhysX-style drive rows and row data for one articulation size group.

    ``fuse_vel_limits != 0`` (the ``fuse_joint_velocity_limits`` path)
    additionally records the DOF's joint velocity limit per drive row in
    ``world_drive_vel_limit`` so the solve kernels can apply the PhysX-style
    stateless velocity clamp at the end of each iteration in place of
    dedicated velocity-limit rows. Non-positive / non-finite limits are stored as
    ``+inf`` so the fused clamp is a no-op for unlimited DOFs. When
    ``fuse_vel_limits == 0`` neither ``joint_velocity_limit`` nor
    ``world_drive_vel_limit`` is touched (1-element dummies are safe).
    """
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    world = art_to_world[art]
    dof_start = articulation_dof_start[art]

    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]

    for j in range(joint_start, joint_end):
        jtype = joint_type[j]
        if jtype != JointType.PRISMATIC and jtype != JointType.REVOLUTE and jtype != JointType.D6:
            continue

        lin_count = joint_dof_dim[j, 0]
        ang_count = joint_dof_dim[j, 1]
        axis_count = lin_count + ang_count
        qd_start = joint_qd_start[j]
        q_start = joint_q_start[j]

        for axis in range(axis_count):
            dof = qd_start + axis
            slot = drive_slot[dof]
            if slot < 0:
                continue

            local_dof = dof - dof_start
            J_group[group_idx, slot, local_dof] = 1.0

            stiffness = joint_target_ke[dof]
            damping = joint_target_kd[dof]
            target_pos = joint_target_pos[joint_target_q_start[j] + axis]
            target_vel = joint_target_vel[dof]
            q = joint_q[q_start + axis]

            world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_JOINT_TARGET
            world_row_parent[world, slot] = -1
            world_row_mu[world, slot] = 0.0
            world_row_beta[world, slot] = 0.0
            world_row_cfm[world, slot] = 0.0
            world_phi[world, slot] = 0.0
            world_target_velocity[world, slot] = target_vel
            world_drive_stiffness[world, slot] = stiffness
            world_drive_damping[world, slot] = damping
            world_drive_geom_error[world, slot] = target_pos - q
            world_drive_max_force[world, slot] = joint_effort_limit[dof]
            if fuse_vel_limits != 0:
                qdot_max = joint_velocity_limit[dof]
                if qdot_max <= 0.0 or not wp.isfinite(qdot_max):
                    qdot_max = float(wp.inf)
                world_drive_vel_limit[world, slot] = qdot_max


@wp.kernel
def compute_physx_pgs_drive_desc(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    world_dof_indices: wp.array2d[int],
    max_world_dofs: int,
    world_row_type: wp.array2d[int],
    world_diag: wp.array2d[float],
    world_J: wp.array3d[float],
    position_delta_velocity: wp.array[float],
    world_target_velocity: wp.array2d[float],
    world_drive_stiffness: wp.array2d[float],
    world_drive_damping: wp.array2d[float],
    world_drive_geom_error: wp.array2d[float],
    world_drive_max_force: wp.array2d[float],
    dt: float,
    position_bias_scale: float,
    position_delta_scale: float,
    # outputs
    world_drive_target_vel_bias: wp.array2d[float],
    world_drive_vel_multiplier: wp.array2d[float],
    world_drive_impulse_multiplier: wp.array2d[float],
    world_drive_max_impulse: wp.array2d[float],
):
    """Build PhysX PGS force-drive descriptor fields for dense joint target rows."""
    world = wp.tid()
    m = world_constraint_count[world]

    for i in range(m):
        if world_row_type[world, i] != PGS_CONSTRAINT_TYPE_JOINT_TARGET:
            world_drive_target_vel_bias[world, i] = 0.0
            world_drive_vel_multiplier[world, i] = 0.0
            world_drive_impulse_multiplier[world, i] = 0.0
            world_drive_max_impulse[world, i] = 0.0
            continue

        stiffness = world_drive_stiffness[world, i]
        damping = world_drive_damping[world, i]
        target_vel = world_target_velocity[world, i]
        geom_error = world_drive_geom_error[world, i]
        unit_response = world_diag[world, i]

        a = dt * (dt * stiffness + damping)
        b = dt * (damping * target_vel)
        x = float(0.0)
        if unit_response > 0.0:
            x = 1.0 / (1.0 + a * unit_response)

        drive_bias_coeff = stiffness * x * dt
        position_delta = float(0.0)
        if position_delta_scale != 0.0:
            for d in range(max_world_dofs):
                global_dof = world_dof_indices[world, d]
                if global_dof >= 0:
                    position_delta += world_J[world, i, d] * position_delta_velocity[global_dof]
            position_delta = position_delta_scale * dt * position_delta

        world_drive_target_vel_bias[world, i] = x * b + drive_bias_coeff * (
            position_bias_scale * geom_error - position_delta
        )
        world_drive_vel_multiplier[world, i] = -x * a
        # PGS path: PhysX uses 1 - x. TGS would use 1 and additional position bias.
        world_drive_impulse_multiplier[world, i] = 1.0 - x

        max_force = world_drive_max_force[world, i]
        max_impulse = float(1.0e20)
        if max_force > 0.0 and wp.isfinite(max_force):
            max_impulse = max_force * dt
        world_drive_max_impulse[world, i] = max_impulse


@wp.kernel
def build_mass_update_mask(
    global_flag: int,
    mass_update_requested: wp.array[int],
    # outputs
    mass_update_mask: wp.array[int],
):
    tid = wp.tid()
    flag = 1 if global_flag != 0 else 0
    if mass_update_requested[tid] != 0:
        flag = 1
    mass_update_mask[tid] = flag


# =============================================================================
# Joint Limit Constraint Kernels
# =============================================================================


@wp.kernel
def build_joint_limit_rows_for_size(
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    joint_type: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_q: wp.array[float],
    joint_limit_activation_gap: float,
    art_to_world: wp.array[int],
    group_to_art: wp.array[int],
    max_constraints: int,
    pgs_beta: float,
    pgs_cfm: float,
    resolved_worlds: wp.array[int],
    # outputs
    world_slot_counter: wp.array[int],
    first_rejected_slot: wp.array[int],
    J_group: wp.array3d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_cfm: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
):
    """Allocate and populate active joint-limit rows in one articulation pass."""
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    world = art_to_world[art]
    if world >= 0 and world < resolved_worlds.shape[0] and resolved_worlds[world] != 0:
        return
    dof_start = articulation_dof_start[art]

    for j in range(articulation_start[art], articulation_start[art + 1]):
        jtype = joint_type[j]
        if jtype != JointType.PRISMATIC and jtype != JointType.REVOLUTE and jtype != JointType.D6:
            continue

        axis_count = joint_dof_dim[j, 0] + joint_dof_dim[j, 1]
        qd_start = joint_qd_start[j]
        q_start = joint_q_start[j]
        for axis in range(axis_count):
            dof = qd_start + axis
            q_val = joint_q[q_start + axis]
            lower = joint_limit_lower[dof]
            upper = joint_limit_upper[dof]

            local_dof = dof - dof_start

            for side in range(2):
                active, phi, sign = joint_limit_side(q_val, lower, upper, side, joint_limit_activation_gap)
                if not active:
                    continue

                slot = reserve(world_slot_counter, world, 1, max_constraints, first_rejected_slot)
                if slot < 0:
                    continue

                J_group[group_idx, slot, local_dof] = sign
                world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_JOINT_LIMIT
                world_row_parent[world, slot] = -1
                world_row_mu[world, slot] = 0.0
                world_row_beta[world, slot] = pgs_beta
                world_row_cfm[world, slot] = pgs_cfm
                world_phi[world, slot] = phi
                world_target_velocity[world, slot] = 0.0


# =============================================================================
# Mimic (Joint Coupling) Constraint Kernels
# =============================================================================
# Bilateral equality rows enforcing ``q_follower = coef0 + coef1 * q_leader``
# between two 1-DoF joints of the same articulation, sourced from
# ``Model.constraint_mimic_*``. This is the FeatherPGS analogue of PhysX's
# ``PxArticulationMimicJoint`` (``qA + gearRatio*qB + offset = 0`` with
# ``gearRatio = -coef1``, ``offset = -coef0``): a joint-space row with two
# Jacobian entries, an unbounded multiplier, and standard Baumgarte drift
# correction. See ``reports/vishal/fpgs_mimic_design.md`` in the parent repo.
#
# ``coef0``/``coef1``/``enabled`` are read from the model arrays at row-build
# time, so runtime mutation (NotifyFlags.CONSTRAINT_PROPERTIES) takes effect
# without solver re-initialization and stays CUDA-graph-safe. The static
# validity mask (same articulation, 1-DoF joint types, distinct DOFs) is
# precomputed on the host at solver init.


@wp.kernel
def allocate_mimic_slots(
    mimic_valid: wp.array[int],
    mimic_enabled: wp.array[wp.bool],
    mimic_world: wp.array[int],
    max_constraints: int,
    # outputs
    mimic_slot: wp.array[int],
    world_slot_counter: wp.array[int],
    first_rejected_slot: wp.array[int],
):
    """Allocate one dense constraint slot per enabled, valid mimic constraint.

    Launched with ``dim = constraint_mimic_count``. Disabled or invalid mimics
    get ``mimic_slot = -1`` and consume no slot.
    """
    k = wp.tid()
    mimic_slot[k] = -1
    if mimic_valid[k] == 0:
        return
    if not mimic_enabled[k]:
        return
    mimic_slot[k] = reserve(world_slot_counter, mimic_world[k], 1, max_constraints, first_rejected_slot)


@wp.kernel
def populate_mimic_J_for_size(
    articulation_dof_start: wp.array[int],
    art_to_world: wp.array[int],
    group_to_art: wp.array[int],
    mimic_slot: wp.array[int],
    mimic_art_start: wp.array[int],
    mimic_art_list: wp.array[int],
    mimic_dof0: wp.array[int],
    mimic_dof1: wp.array[int],
    mimic_q0: wp.array[int],
    mimic_q1: wp.array[int],
    mimic_coef0: wp.array[float],
    mimic_coef1: wp.array[float],
    joint_q: wp.array[float],
    pgs_beta: float,
    pgs_cfm: float,
    # outputs
    J_group: wp.array3d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_cfm: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
):
    """Populate Jacobian and metadata for mimic constraint rows.

    Launched once per size group with ``dim = n_arts_of_size``, matching the
    joint-limit populate kernel. Each thread visits only its articulation's
    range in the precomputed mimic table:

    * Jacobian ``J = e_follower - coef1 * e_leader`` — two entries.
    * ``phi = q_follower - coef1 * q_leader - coef0`` — the signed violation.
    """
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    world = art_to_world[art]
    dof_start = articulation_dof_start[art]

    for m in range(mimic_art_start[art], mimic_art_start[art + 1]):
        k = mimic_art_list[m]
        slot = mimic_slot[k]
        if slot < 0:
            continue

        c0 = mimic_coef0[k]
        c1 = mimic_coef1[k]

        # Jacobian: +1 on the follower DOF, -coef1 on the leader DOF. The two
        # DOFs are guaranteed distinct by the host-side validity mask.
        J_group[group_idx, slot, mimic_dof0[k] - dof_start] = 1.0
        J_group[group_idx, slot, mimic_dof1[k] - dof_start] = -c1

        world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_MIMIC
        world_row_parent[world, slot] = -1
        world_row_mu[world, slot] = 0.0
        world_row_beta[world, slot] = pgs_beta
        world_row_cfm[world, slot] = pgs_cfm
        world_phi[world, slot] = mimic_phi(joint_q[mimic_q0[k]], joint_q[mimic_q1[k]], c0, c1)
        world_target_velocity[world, slot] = 0.0


# =============================================================================
# Connect (Loop-Closure) Constraint Kernels
# =============================================================================
# Three bilateral rows per loop closure enforcing point coincidence of the loop
# joint's parent/child anchors — the FeatherPGS realization of MJCF ``connect``
# equalities (design: ``reports/vishal/fpgs_connect_design.md`` in the parent
# repo). The Jacobian per axis is the anchor-point Jacobian difference of the
# two bodies, built with the same ancestor walk contact rows use.


@wp.kernel
def invalidate_fk_id_cache_for_articulations(
    articulations: wp.array[int],
    # outputs
    fk_id_cache_valid: wp.array[int],
):
    """Clear the FK/ID cache flag of the listed articulations only."""
    k = wp.tid()
    art = articulations[k]
    if art >= 0 and art < fk_id_cache_valid.shape[0]:
        fk_id_cache_valid[art] = 0


@wp.kernel
def allocate_connect_slots(
    connect_valid: wp.array[int],
    connect_enabled: wp.array[int],
    connect_world: wp.array[int],
    max_constraints: int,
    # outputs
    connect_slot: wp.array[int],
    world_slot_counter: wp.array[int],
    first_rejected_slot: wp.array[int],
):
    """Allocate three consecutive dense slots per enabled, valid loop closure.

    Launched with ``dim = n_connect``. If the buffer cannot hold all three rows the
    closure is dropped whole (``connect_slot = -1``) rather than partially enforced.
    """
    k = wp.tid()
    connect_slot[k] = -1
    if connect_valid[k] == 0 or connect_enabled[k] == 0:
        return
    connect_slot[k] = reserve(world_slot_counter, connect_world[k], 3, max_constraints, first_rejected_slot)


@wp.kernel
def populate_connect_J_for_size(
    articulation_dof_start: wp.array[int],
    art_to_world: wp.array[int],
    group_to_art: wp.array[int],
    n_dofs: int,
    connect_slot: wp.array[int],
    connect_art: wp.array[int],
    connect_body_p: wp.array[int],
    connect_body_c: wp.array[int],
    connect_anchor_p: wp.array[wp.vec3],
    connect_anchor_c: wp.array[wp.vec3],
    connect_parent_prescribed: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_to_joint: wp.array[int],
    body_to_articulation: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    articulation_origin: wp.array[wp.vec3],
    pgs_beta: float,
    pgs_cfm: float,
    # outputs
    J_group: wp.array3d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_cfm: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
):
    """Populate Jacobian and metadata for connect (loop-closure) rows.

    Launched once per size group with ``dim = n_arts_of_size``. Each thread scans the
    (small) closure table and, for closures belonging to its articulation, writes three
    rows: for world axis ``e_k``, ``phi = (p_A - p_B) . e_k`` and
    ``J = e_k . (Jpoint(parent, p_A) - Jpoint(child, p_B))`` via the contact ancestor
    walk. One near-redundant axis on planar linkages is expected; its vanishing Delassus
    diagonal makes the sweep skip it.

    A closure whose parent is prescribed (``connect_parent_prescribed[k] != 0``: a
    kinematic body outside the child's articulation, or the world when
    ``connect_body_p[k] < 0``) contributes no parent DOFs. Its anchor velocity is
    instead moved to the row target, ``J v = -(e_k . v_A)``, so the child anchor tracks
    the moving parent anchor exactly as a same-articulation closure would.
    """
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    world = art_to_world[art]
    dof_start = articulation_dof_start[art]

    n_connect = connect_art.shape[0]
    for k in range(n_connect):
        if connect_art[k] != art:
            continue
        base_slot = connect_slot[k]
        if base_slot < 0:
            continue

        body_p = connect_body_p[k]
        body_c = connect_body_c[k]
        prescribed = connect_parent_prescribed[k] != 0
        p_a = connect_anchor_p[k]
        v_a = wp.vec3(0.0, 0.0, 0.0)
        if body_p >= 0:
            p_a = wp.transform_point(body_q[body_p], connect_anchor_p[k])
            if prescribed:
                # Prescribed parent: the anchor velocity comes from the body twist
                # (v_com_world, omega_world) rather than from tree DOFs.
                twist = body_qd[body_p]
                v_com = wp.spatial_top(twist)
                omega = wp.spatial_bottom(twist)
                x_com = wp.transform_point(body_q[body_p], body_com[body_p])
                v_a = v_com + wp.cross(omega, p_a - x_com)
        p_b = wp.transform_point(body_q[body_c], connect_anchor_c[k])
        origin = articulation_origin[art]
        delta = p_a - p_b

        for axis in range(3):
            slot = base_slot + axis
            # The two ancestor walks below ACCUMULATE (shared ancestors partially cancel),
            # so the row must start from zero regardless of buffer reuse policy.
            for d in range(n_dofs):
                J_group[group_idx, slot, d] = 0.0
            e = wp.vec3(0.0, 0.0, 0.0)
            e[axis] = 1.0
            target = float(0.0)

            # J = e . (Jpoint(parent, p_a) - Jpoint(child, p_b)): ancestor walks with
            # opposite signs; shared ancestors partially cancel automatically.
            if prescribed:
                target = -wp.dot(e, v_a)
            else:
                curr = body_to_joint[body_p]
                while curr != -1:
                    d0 = joint_qd_start[curr]
                    d1 = joint_qd_start[curr + 1]
                    for d in range(d0, d1):
                        J_group[group_idx, slot, d - dof_start] += jacobian_entry(joint_S_s[d], e, p_a, origin)
                    curr = joint_ancestor[curr]
            curr = body_to_joint[body_c]
            while curr != -1:
                d0 = joint_qd_start[curr]
                d1 = joint_qd_start[curr + 1]
                for d in range(d0, d1):
                    J_group[group_idx, slot, d - dof_start] -= jacobian_entry(joint_S_s[d], e, p_b, origin)
                curr = joint_ancestor[curr]

            # Normalize the row: on planar linkages one axis is near-redundant with the
            # tree, leaving ||J|| ~ 1e-4 — its Delassus diagonal collapses to pure CFM
            # while H^-1 J^T stays large against small link inertias, and PGS then relaxes
            # a real velocity effect against a near-zero denominator (measured to NaN in
            # one sweep). Normalizing keeps the identical constraint with a well-scaled
            # diagonal; truly degenerate rows are left inert (zero J, zero phi).
            norm_sq = float(0.0)
            for d in range(n_dofs):
                norm_sq += J_group[group_idx, slot, d] * J_group[group_idx, slot, d]
            phi_axis = delta[axis]
            if norm_sq > 1.0e-8:
                inv_norm = 1.0 / wp.sqrt(norm_sq)
                for d in range(n_dofs):
                    J_group[group_idx, slot, d] *= inv_norm
                phi_axis *= inv_norm
                target *= inv_norm
            else:
                for d in range(n_dofs):
                    J_group[group_idx, slot, d] = 0.0
                phi_axis = 0.0
                target = 0.0

            world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_CONNECT
            world_row_parent[world, slot] = -1
            world_row_mu[world, slot] = 0.0
            world_row_beta[world, slot] = pgs_beta
            world_row_cfm[world, slot] = pgs_cfm
            world_phi[world, slot] = phi_axis
            world_target_velocity[world, slot] = target


# =============================================================================
# Bilateral Pre-elimination Kernels (mimic + connect Schur complement)
# =============================================================================
# Fold the selected bilateral internal equality rows (MIMIC and/or CONNECT) into the
# response operator so every other row sees the closed-loop effective mass:
# with B the bilateral block of one articulation, Y_B = H^-1 J_B^T and
# S = J_B Y_B (+ regularization), every other row's response is corrected to
# Y'_i = Y_i - Y_B S^-1 (J_B Y_i), which makes J_B Y'_i ~ 0 — sweep impulses
# then preserve the closures exactly, and the corrected row diagonals fall out
# of the unchanged diag_from_JY pass. The predictor velocity is projected once
# (J_B v + b_B -> 0) to replace the eliminated rows' Baumgarte work. See
# reports/vishal/robotiq/fpgs_preelimination_design.md (skild-IL-solver).

PREELIM_MAX_ROWS = 8
"""Per-articulation capacity of the pre-eliminated bilateral block (the
Robotiq 2F-85 needs 7: one mimic + two 3-row connect closures)."""

_preelim_vec = wp.types.vector(length=PREELIM_MAX_ROWS, dtype=wp.float32)


@wp.func
def preelim_solve(
    n: int,
    L: wp.array[float],
    L_start: int,
    b: _preelim_vec,
) -> _preelim_vec:
    """Solve (L L^T) x = b for the packed per-articulation Cholesky factor.

    Mirrors :func:`dense_subs` but keeps the right-hand side in registers so
    per-row callers need no global scratch.
    """
    x = _preelim_vec()
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= L[L_start + dense_index(n, i, j)] * x[j]
        x[i] = s / L[L_start + dense_index(n, i, i)]
    for ii in range(n):
        i = n - 1 - ii
        s = x[i]
        for j in range(i + 1, n):
            s -= L[L_start + dense_index(n, j, i)] * x[j]
        x[i] = s / L[L_start + dense_index(n, i, i)]
    return x


@wp.kernel
def preelim_setup_for_size(
    group_to_art: wp.array[int],
    art_to_preelim: wp.array[int],
    mimic_slot: wp.array[int],
    mimic_art_start: wp.array[int],
    mimic_art_list: wp.array[int],
    n_mimic: int,
    connect_slot: wp.array[int],
    connect_art: wp.array[int],
    n_connect: int,
    J_group: wp.array3d[float],
    Y_group: wp.array3d[float],
    n_dofs: int,
    reg_rel: float,
    reg_floor: float,
    # outputs
    preelim_slots: wp.array[int],
    preelim_nB: wp.array[int],
    S_scratch: wp.array[float],
    reg: wp.array[float],
    LS: wp.array[float],
):
    """Gather the selected bilateral block, form S = J_B Y_B, and factor it.

    Launched once per size group with ``dim = n_arts_of_size`` after the
    ``Y = H^-1 J^T`` stage; one thread per articulation (the block is tiny).
    Ownership comes from the per-constraint ``*_art``/``*_slot`` tables — NOT
    ``row_type``, which is per-world and would admit other articulations'
    zero-J rows and make S singular. Zero-norm rows (the degenerate connect
    axis the populate kernel already zeroes) are dropped here for the same
    reason.
    """
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    pe = art_to_preelim[art]
    if pe < 0:
        return
    base = pe * PREELIM_MAX_ROWS

    # Gather candidate slots owned by this articulation.
    n = int(0)
    if n_mimic > 0:
        for m in range(mimic_art_start[art], mimic_art_start[art + 1]):
            k = mimic_art_list[m]
            if mimic_slot[k] >= 0 and n < PREELIM_MAX_ROWS:
                preelim_slots[base + n] = mimic_slot[k]
                n += 1
    for k in range(n_connect):
        if connect_art[k] == art and connect_slot[k] >= 0:
            for a in range(3):
                if n < PREELIM_MAX_ROWS:
                    preelim_slots[base + n] = connect_slot[k] + a
                    n += 1

    # Drop zero-norm rows (inert degenerate axes) by compaction.
    m = int(0)
    for p in range(n):
        s = preelim_slots[base + p]
        nrm = float(0.0)
        for d in range(n_dofs):
            nrm += J_group[group_idx, s, d] * J_group[group_idx, s, d]
        if nrm > 1.0e-10:
            preelim_slots[base + m] = s
            m += 1
    for p in range(m, PREELIM_MAX_ROWS):
        preelim_slots[base + p] = -1
    preelim_nB[pe] = m
    if m == 0:
        return

    # S[p, q] = J_B[p] . Y_B[q]  (n x n, row-major with stride m).
    s_base = pe * PREELIM_MAX_ROWS * PREELIM_MAX_ROWS
    for p in range(m):
        sp = preelim_slots[base + p]
        for q in range(m):
            sq = preelim_slots[base + q]
            acc = float(0.0)
            for d in range(n_dofs):
                acc += J_group[group_idx, sp, d] * Y_group[group_idx, sq, d]
            S_scratch[s_base + dense_index(m, p, q)] = acc

    # RELATIVE diagonal regularization: a planar four-bar's closure has one
    # near-dependent axis, so after eliminating the well-conditioned rows the
    # last pivot is pure float32 cancellation noise around zero — an absolute
    # epsilon under the Delassus scale (which is huge for gram-scale links)
    # goes negative in the sqrt. Scaling by the row's own diagonal keeps the
    # factor SPD at any mass scale; the redundant axis just turns ~reg_rel
    # soft, which is benign (the tree already enforces that direction).
    for p in range(m):
        reg[base + p] = reg_rel * S_scratch[s_base + dense_index(m, p, p)] + reg_floor

    dense_cholesky(m, S_scratch, reg, s_base, base, LS)


@wp.kernel
def preelim_correct_Y_for_size(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    art_to_preelim: wp.array[int],
    constraint_count: wp.array[int],
    preelim_slots: wp.array[int],
    preelim_nB: wp.array[int],
    LS: wp.array[float],
    J_group: wp.array3d[float],
    n_dofs: int,
    max_constraints: int,
    n_arts: int,
    # outputs
    Y_group: wp.array3d[float],
):
    """Correct every non-B row's response: Y_i -= Y_B S^-1 (J_B Y_i).

    Launched per size group with ``dim = n_arts_of_size * max_constraints``.
    B rows themselves are left intact (the projection kernel needs Y_B; their
    sweep visits self-neutralize because the projected velocity already
    satisfies J_B v = -b_B, so their residual is ~0).
    """
    idx = wp.tid()
    group_idx = idx // max_constraints
    i = idx % max_constraints
    if group_idx >= n_arts:
        return
    art = group_to_art[group_idx]
    pe = art_to_preelim[art]
    if pe < 0:
        return
    world = art_to_world[art]
    if i >= constraint_count[world]:
        return
    base = pe * PREELIM_MAX_ROWS
    m = preelim_nB[pe]
    if m == 0:
        return
    for p in range(m):
        if preelim_slots[base + p] == i:
            return  # B row: keep its response for the projection

    # w = J_B . Y_i
    w = _preelim_vec()
    nonzero = int(0)
    for p in range(m):
        sp = preelim_slots[base + p]
        acc = float(0.0)
        for d in range(n_dofs):
            acc += J_group[group_idx, sp, d] * Y_group[group_idx, i, d]
        w[p] = acc
        if acc != 0.0:
            nonzero = 1
    if nonzero == 0:
        return  # row does not touch this articulation (e.g. other art's row)

    z = preelim_solve(m, LS, pe * PREELIM_MAX_ROWS * PREELIM_MAX_ROWS, w)

    for d in range(n_dofs):
        acc = float(0.0)
        for p in range(m):
            acc += Y_group[group_idx, preelim_slots[base + p], d] * z[p]
        Y_group[group_idx, i, d] -= acc


@wp.kernel
def preelim_project_velocity_for_size(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    art_to_preelim: wp.array[int],
    articulation_dof_start: wp.array[int],
    preelim_slots: wp.array[int],
    preelim_nB: wp.array[int],
    LS: wp.array[float],
    J_group: wp.array3d[float],
    Y_group: wp.array3d[float],
    world_rhs: wp.array2d[float],
    n_dofs: int,
    # outputs
    v_out: wp.array[float],
):
    """One-shot bilateral projection: v_out -= Y_B S^-1 (J_B v_out + b_B).

    Launched per size group with ``dim = n_arts_of_size`` after the sweep's
    velocity buffer is seeded with the predictor (v_out = v_hat). Afterwards
    J_B v_out = -b_B holds, and it stays held through the sweep because every
    corrected column satisfies J_B Y'_i ~ 0. Each thread owns one
    articulation's contiguous DOF window, so there are no write races.
    """
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    pe = art_to_preelim[art]
    if pe < 0:
        return
    m = preelim_nB[pe]
    if m == 0:
        return
    base = pe * PREELIM_MAX_ROWS
    world = art_to_world[art]
    dof_start = articulation_dof_start[art]

    r = _preelim_vec()
    for p in range(m):
        sp = preelim_slots[base + p]
        acc = world_rhs[world, sp]
        for d in range(n_dofs):
            acc += J_group[group_idx, sp, d] * v_out[dof_start + d]
        r[p] = acc

    z = preelim_solve(m, LS, pe * PREELIM_MAX_ROWS * PREELIM_MAX_ROWS, r)

    for d in range(n_dofs):
        acc = float(0.0)
        for p in range(m):
            acc += Y_group[group_idx, preelim_slots[base + p], d] * z[p]
        if acc != 0.0:
            v_out[dof_start + d] -= acc


# =============================================================================
# Joint Velocity-Limit Constraint Kernels
# =============================================================================
# These kernels mirror the PhysX per-DOF velocity-limit formulation documented
# in ``notes/investigations/velocity-spike/physx-deep-dive.md`` §4 and the math
# appendix. They reuse the same allocation / populate shape as the
# joint-position-limit kernels above, but finite velocity limits allocate both
# sides of the bilateral velocity box rather than waiting for a violation.


@wp.kernel
def allocate_joint_velocity_limit_slots(
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    articulation_H_rows: wp.array[int],
    joint_type: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_velocity_limit: wp.array[float],
    joint_qd: wp.array[float],
    velocity_limit_activation_fraction: float,
    drive_slot: wp.array[int],
    skip_driven: int,
    art_to_world: wp.array[int],
    max_constraints: int,
    # outputs
    velocity_limit_slot: wp.array[int],
    velocity_limit_sign: wp.array[float],
    world_slot_counter: wp.array[int],
    first_rejected_slot: wp.array[int],
):
    """Allocate lower/upper velocity-limit rows for every finitely limited DOF.

    For each non-locked DOF of a PRISMATIC / REVOLUTE / D6 joint
    with ``joint_velocity_limit[i] > 0``, two slots are atomically reserved in
    the per-world counter. The sign encodes which side of the bilateral box
    ``[-qdot_max, +qdot_max]`` each row enforces:

    * ``sign = +1`` — lower-limit violation (``qdot_i < -qdot_max``). The row
      pushes velocity back up (``J = +e_i``, ``target_vel = -qdot_max``).
    * ``sign = -1`` — upper-limit violation (``qdot_i > +qdot_max``). The row
      pushes velocity back down (``J = -e_i``, ``target_vel = -qdot_max``).

    The matrix-free solver treats these rows as stateless PhysX-style clamp
    rows: satisfied sides apply no impulse, and a violated side applies only
    the current velocity overshoot correction.

    ``velocity_limit_activation_fraction`` proximity-gates the allocation:
    with a positive fraction the lower/upper pair is reserved only when
    ``|joint_qd[dof]| >= fraction * qdot_max``, i.e. the DOF is close enough
    to the velocity box edge that the rows could act
    (:func:`.row_laws.velocity_limit_active`). A fraction of ``0.0`` disables
    the gate: every limited DOF takes its pair. Because the gate samples the pre-solve
    velocity, a DOF that crosses the threshold during a step is clamped one
    step late.

    ``skip_driven != 0`` (the ``fuse_joint_velocity_limits`` path) skips DOFs
    with a PhysX drive row (``drive_slot[dof] >= 0``): their velocity limit is
    enforced as a stateless clamp at the end of each solver iteration
    instead of a dedicated row pair (PhysX ``PxClamp`` math, run after the
    contact phases like the rows it replaces). DOFs with a velocity limit but no drive row keep their
    dedicated rows. When ``skip_driven == 0``, ``drive_slot`` is never read
    (a 1-element dummy is safe).

    Outputs two entries per DOF in ``velocity_limit_slot`` (world-constraint
    row, or -1) and ``velocity_limit_sign`` (+1 / -1).
    """
    art = wp.tid()
    world = art_to_world[art]

    # Initialize all DOFs of this articulation to "no limit active"
    dof_base = articulation_dof_start[art]
    dof_count = articulation_H_rows[art]
    for d in range(dof_count):
        lower_idx = 2 * (dof_base + d)
        upper_idx = lower_idx + 1
        velocity_limit_slot[lower_idx] = -1
        velocity_limit_slot[upper_idx] = -1
        velocity_limit_sign[lower_idx] = 0.0
        velocity_limit_sign[upper_idx] = 0.0

    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]
    n_eligible = int(0)

    for j in range(joint_start, joint_end):
        jtype = joint_type[j]
        if jtype != JointType.PRISMATIC and jtype != JointType.REVOLUTE and jtype != JointType.D6:
            continue

        lin_count = joint_dof_dim[j, 0]
        ang_count = joint_dof_dim[j, 1]
        axis_count = lin_count + ang_count
        qd_start = joint_qd_start[j]

        for axis in range(axis_count):
            dof = qd_start + axis
            # A non-positive limit is unlimited (PhysX pins ``recipResponse`` off for ``unitResponse <= 0``);
            # a positive fraction admits only a DOF whose speed reaches that fraction of the limit.
            if not velocity_limit_active(joint_velocity_limit[dof], joint_qd[dof], velocity_limit_activation_fraction):
                continue

            # Fused path: driven DOFs are clamped at the end of each iteration.
            if skip_driven != 0:
                if drive_slot[dof] >= 0:
                    continue

            # Mark eligible; slots are reserved below with a single atomic per articulation (the
            # per-DOF pair of atomics was a 2*dofs-long serial chain per thread).
            velocity_limit_sign[2 * dof] = 1.0
            n_eligible += 1

    if n_eligible == 0:
        return
    # One reservation per articulation: its pairs fit together or none of them does.
    slot = reserve(world_slot_counter, world, 2 * n_eligible, max_constraints, first_rejected_slot)
    for d in range(dof_count):
        dof = dof_base + d
        lower_idx = 2 * dof
        if velocity_limit_sign[lower_idx] != 1.0:
            continue
        if slot < 0:
            velocity_limit_sign[lower_idx] = 0.0
            continue
        velocity_limit_slot[lower_idx] = slot
        velocity_limit_slot[lower_idx + 1] = slot + 1
        velocity_limit_sign[lower_idx + 1] = -1.0
        slot += 2


@wp.kernel
def populate_joint_velocity_limit_J_for_size(
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    joint_type: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_velocity_limit: wp.array[float],
    art_to_world: wp.array[int],
    velocity_limit_slot: wp.array[int],
    velocity_limit_sign: wp.array[float],
    group_to_art: wp.array[int],
    pgs_cfm: float,
    articulation_dof_count_of_size: int,
    # outputs
    J_group: wp.array3d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_cfm: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
):
    """Populate Jacobian and metadata for joint velocity-limit rows.

    Launched once per size group with ``dim = n_arts_of_size * size * 2`` (one thread per DOF side;
    eligibility is read from ``velocity_limit_slot``, so no joint-type walk is needed). For every DOF
    whose two ``velocity_limit_slot`` entries are non-negative, writes signed
    ±1 entries into the local DOF column of the grouped Jacobian and sets the
    constraint metadata. The rows have **no Baumgarte bias** (``beta = 0``,
    ``phi = 0``) — PhysX's velocity-limit row has no ``data.erp`` either.
    The target velocity is ``-qdot_max`` for both sides of the box: combined
    with the sign flip on ``J``, this encodes the bilateral projection as
    two unilateral ``J*v >= target_vel`` rows with ``lambda >= 0``.

    DOFs skipped by :func:`allocate_joint_velocity_limit_slots` (activation
    gate, or driven DOFs under ``fuse_joint_velocity_limits``) carry
    ``velocity_limit_slot == -1`` and are skipped here through the same
    per-row slot check; this kernel needs no separate skip flag.
    """
    tid = wp.tid()
    n_dofs = articulation_dof_count_of_size
    side = tid % 2
    local_dof = (tid // 2) % n_dofs
    group_idx = tid // (2 * n_dofs)
    art = group_to_art[group_idx]
    world = art_to_world[art]
    dof = articulation_dof_start[art] + local_dof
    row_idx = 2 * dof + side
    slot = velocity_limit_slot[row_idx]
    if slot < 0:
        return
    qdot_max = joint_velocity_limit[dof]
    sign = velocity_limit_sign[row_idx]

    J_group[group_idx, slot, local_dof] = sign

    world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT
    world_row_parent[world, slot] = -1
    world_row_mu[world, slot] = 0.0
    world_row_beta[world, slot] = 0.0
    world_row_cfm[world, slot] = pgs_cfm
    world_phi[world, slot] = 0.0
    world_target_velocity[world, slot] = -qdot_max


# =============================================================================
# Multi-Articulation Contact Building Kernels
# =============================================================================
# These kernels enable contacts between multiple articulations within the same
# world. The constraint system becomes world-level instead of per-articulation.


@wp.func
def _admit_contact(
    c: int,
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    body_flags: wp.array[wp.int32],
    body_has_response_dofs: wp.array[int],
    is_free_rigid: wp.array[int],
    contact_gap_gate: float,
    same_articulation_contact_gap_gate: float,
    articulation_pair_contact_gap_gate: float,
):
    """Return whether a contact gets rows, and its articulations, world and gap: ``(admitted, art_a, art_b, world, phi)``.

    A contact needs an endpoint that responds to its impulses, one world (not
    two, nor none without an articulation) and a gap within the normal gap limit.
    """
    body_a, body_b, art_a, art_b, world = contact_endpoints(
        contact_shape0[c], contact_shape1[c], shape_body, body_to_articulation, art_to_world
    )
    responds = contact_responds(art_a, body_a, articulation_response_dof_count, body_has_response_dofs, body_flags)
    if not responds:
        responds = contact_responds(art_b, body_b, articulation_response_dof_count, body_has_response_dofs, body_flags)
    phi = float(0.0)
    admitted = responds and world >= 0
    if admitted:
        _normal, _point_a, _point_b, phi = contact_points(
            contact_normal[c],
            contact_point0[c],
            contact_point1[c],
            contact_thickness0[c],
            contact_thickness1[c],
            body_a,
            body_b,
            body_q,
        )
        a_non_free = art_a >= 0 and is_free_rigid[art_a] == 0
        b_non_free = art_b >= 0 and is_free_rigid[art_b] == 0
        normal_gap_limit = contact_normal_gap_limit(
            a_non_free,
            b_non_free,
            art_a == art_b,
            contact_gap_gate,
            articulation_pair_contact_gap_gate,
            same_articulation_contact_gap_gate,
        )
        admitted = not (phi > normal_gap_limit)
    return admitted, art_a, art_b, world, phi


@wp.func
def _allocate_world_contact_slot(
    c: int,
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    body_flags: wp.array[wp.int32],
    body_has_response_dofs: wp.array[int],
    is_free_rigid: wp.array[int],
    has_free_rigid: int,
    contact_gap_gate: float,
    same_articulation_contact_gap_gate: float,
    articulation_pair_contact_gap_gate: float,
    max_constraints: int,
    mf_max_constraints: int,
    enable_friction: int,
    contact_friction_gap_threshold: float,
    contact_friction_articulation_pairs_only: int,
    row_capacity_telemetry: int,
    friction_patches: FrictionPatches,
    # outputs
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    world_slot_counter: wp.array[int],
    contact_path: wp.array[int],
    mf_slot_counter: wp.array[int],
    contact_slots_needed: wp.array[int],
    dense_dropped_contact_rows: wp.array[int],
    mf_dropped_contact_rows: wp.array[int],
    dense_first_rejected_slot: wp.array[int],
    mf_first_rejected_slot: wp.array[int],
):
    """Classify and allocate rows for one active contact.

    Allocates world-level constraint slots for each contact and records
    which articulations are involved. Contacts where both sides are free
    rigid bodies (or ground) are routed to the matrix-free path.

    Each contact reserves 1 slot for the normal row and, when enabled below the
    friction gap threshold, 2 adjacent slots for Coulomb friction rows.
    A positive contact gap gate excludes wider speculative contacts before any
    route reserves slots; zero disables the gate.

    """
    admitted, art_a, art_b, world, phi = _admit_contact(
        c,
        contact_shape0,
        contact_shape1,
        contact_point0,
        contact_point1,
        contact_normal,
        contact_thickness0,
        contact_thickness1,
        body_q,
        shape_body,
        body_to_articulation,
        art_to_world,
        articulation_response_dof_count,
        body_flags,
        body_has_response_dofs,
        is_free_rigid,
        contact_gap_gate,
        same_articulation_contact_gap_gate,
        articulation_pair_contact_gap_gate,
    )
    if not admitted:
        contact_slot[c] = -1
        contact_path[c] = -1
        return
    a_has_dofs = art_a >= 0 and articulation_response_dof_count[art_a] > 0
    b_has_dofs = art_b >= 0 and articulation_response_dof_count[art_b] > 0
    a_non_free = art_a >= 0 and is_free_rigid[art_a] == 0
    b_non_free = art_b >= 0 and is_free_rigid[art_b] == 0

    # Classify: MF path if both sides are free rigid or ground.
    is_mf = 0
    if has_free_rigid != 0:
        a_is_mf_compatible = not a_has_dofs or is_free_rigid[art_a] != 0
        b_is_mf_compatible = not b_has_dofs or is_free_rigid[art_b] != 0
        if a_is_mf_compatible and b_is_mf_compatible:
            is_mf = 1

    # Every normal survives; only the selected patch anchors receive tangent rows.
    slots_needed = 1
    add_friction = enable_friction != 0 and contact_friction_eligible(
        phi, a_non_free, b_non_free, contact_friction_articulation_pairs_only, contact_friction_gap_threshold
    )
    if friction_patches.enabled != 0:
        add_friction = add_friction and friction_patches.weight[c] > 0.0
    if add_friction:
        slots_needed = 3
    contact_slots_needed[c] = slots_needed

    if is_mf != 0:
        # Matrix-free path
        slot = reserve(mf_slot_counter, world, slots_needed, mf_max_constraints, mf_first_rejected_slot)
        if slot < 0:
            if row_capacity_telemetry != 0:
                wp.atomic_add(mf_dropped_contact_rows, world, slots_needed)
            contact_slot[c] = -1
            contact_path[c] = -1
            return
        contact_world[c] = world
        contact_slot[c] = slot
        contact_art_a[c] = art_a
        contact_art_b[c] = art_b
        contact_path[c] = 1
    else:
        # Dense path
        slot = reserve(world_slot_counter, world, slots_needed, max_constraints, dense_first_rejected_slot)
        if slot < 0:
            if row_capacity_telemetry != 0:
                wp.atomic_add(dense_dropped_contact_rows, world, slots_needed)
            contact_slot[c] = -1
            contact_path[c] = -1
            return
        contact_world[c] = world
        contact_slot[c] = slot
        contact_art_a[c] = art_a
        contact_art_b[c] = art_b
        contact_path[c] = 0


@wp.func
def _mark_resolved_patch_contact(
    c: int,
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    body_flags: wp.array[wp.int32],
    body_has_response_dofs: wp.array[int],
    is_free_rigid: wp.array[int],
    contact_gap_gate: float,
    same_articulation_contact_gap_gate: float,
    articulation_pair_contact_gap_gate: float,
    # outputs
    contact_slot: wp.array[int],
    contact_path: wp.array[int],
):
    """Mark a contact of a resolved world that the row allocator would reserve rows for.

    Repeats the allocator's checks before any reservation: a responding
    endpoint, one world and the normal gap limit. Such a contact gets
    CONTACT_PATH_RESOLVED and no slot; persistent patch friction releases its
    anchor at the finish, as the complete path releases an anchor whose rows
    solve zero load. Every other contact keeps path -1, and its anchor history.
    """
    admitted, _art_a, _art_b, _world, _phi = _admit_contact(
        c,
        contact_shape0,
        contact_shape1,
        contact_point0,
        contact_point1,
        contact_normal,
        contact_thickness0,
        contact_thickness1,
        body_q,
        shape_body,
        body_to_articulation,
        art_to_world,
        articulation_response_dof_count,
        body_flags,
        body_has_response_dofs,
        is_free_rigid,
        contact_gap_gate,
        same_articulation_contact_gap_gate,
        articulation_pair_contact_gap_gate,
    )
    contact_slot[c] = -1
    contact_path[c] = -1
    if admitted:
        contact_path[c] = CONTACT_PATH_RESOLVED


@wp.kernel
def allocate_world_contact_slots(
    contact_count: wp.array[int],
    total_num_threads: int,
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    body_flags: wp.array[wp.int32],
    body_has_response_dofs: wp.array[int],
    is_free_rigid: wp.array[int],
    has_free_rigid: int,
    contact_gap_gate: float,
    same_articulation_contact_gap_gate: float,
    articulation_pair_contact_gap_gate: float,
    max_constraints: int,
    mf_max_constraints: int,
    enable_friction: int,
    contact_friction_gap_threshold: float,
    contact_friction_articulation_pairs_only: int,
    row_capacity_telemetry: int,
    friction_patches: FrictionPatches,
    resolved_worlds: wp.array[int],
    # outputs
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    world_slot_counter: wp.array[int],
    contact_path: wp.array[int],
    mf_slot_counter: wp.array[int],
    contact_slots_needed: wp.array[int],
    dense_dropped_contact_rows: wp.array[int],
    mf_dropped_contact_rows: wp.array[int],
    dense_first_rejected_slot: wp.array[int],
    mf_first_rejected_slot: wp.array[int],
):
    """Allocate active contacts with work proportional to the materialized prefix.

    A collision pass clamps :paramref:`contact_count` to the rows its writers
    filled, every one of them, and flags an overflow in its own capacity
    status, so the count always describes a materialized prefix.

    Slot counters only grow. A contact whose reservation does not fit records its
    slot in the family's ``*_first_rejected_slot`` (per world, preset to a large
    value) instead of rolling the counter back: every later reservation starts at
    or past that slot, so the finalize kernels truncate the row count there and
    no accepted contact can hold a slot at or beyond the count. Rolling back
    raced with concurrent reservations and left accepted rows past the count,
    which the friction patch links then chained into.
    """
    thread = wp.tid()
    # A collision pass clamps the count to the rows it wrote, all of them, and flags the overflow itself.
    total_contacts = contact_count[0]

    for c in range(thread, total_contacts, total_num_threads):
        if resolved_worlds.shape[0] > 0:
            _body_a, _body_b, art_a, art_b, world = contact_endpoints(
                contact_shape0[c], contact_shape1[c], shape_body, body_to_articulation, art_to_world
            )
            if world >= 0 and world < resolved_worlds.shape[0] and resolved_worlds[world] != 0:
                # Refresh every active mapping each solve. The original force
                # converter publishes zero before reading any row for this path.
                contact_world[c] = world
                contact_art_a[c] = art_a
                contact_art_b[c] = art_b
                contact_slot[c] = -1
                contact_path[c] = -1
                contact_slots_needed[c] = 0
                continue
        _allocate_world_contact_slot(
            c,
            contact_shape0,
            contact_shape1,
            contact_point0,
            contact_point1,
            contact_normal,
            contact_thickness0,
            contact_thickness1,
            body_q,
            shape_body,
            body_to_articulation,
            art_to_world,
            articulation_response_dof_count,
            body_flags,
            body_has_response_dofs,
            is_free_rigid,
            has_free_rigid,
            contact_gap_gate,
            same_articulation_contact_gap_gate,
            articulation_pair_contact_gap_gate,
            max_constraints,
            mf_max_constraints,
            enable_friction,
            contact_friction_gap_threshold,
            contact_friction_articulation_pairs_only,
            row_capacity_telemetry,
            friction_patches,
            contact_world,
            contact_slot,
            contact_art_a,
            contact_art_b,
            world_slot_counter,
            contact_path,
            mf_slot_counter,
            contact_slots_needed,
            dense_dropped_contact_rows,
            mf_dropped_contact_rows,
            dense_first_rejected_slot,
            mf_first_rejected_slot,
        )


# Persistent patch friction with the resolved-world classifier: in a world the classifier skips,
# the twin of the kernel above marks the contacts that kernel would reserve rows for with
# CONTACT_PATH_RESOLVED, so the patch finish releases their anchors as the complete path does.
# It differs from that kernel only by the _mark_resolved_patch_contact call.
@wp.kernel
def allocate_world_contact_slots_patch_resolved(
    contact_count: wp.array[int],
    total_num_threads: int,
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    body_flags: wp.array[wp.int32],
    body_has_response_dofs: wp.array[int],
    is_free_rigid: wp.array[int],
    has_free_rigid: int,
    contact_gap_gate: float,
    same_articulation_contact_gap_gate: float,
    articulation_pair_contact_gap_gate: float,
    max_constraints: int,
    mf_max_constraints: int,
    enable_friction: int,
    contact_friction_gap_threshold: float,
    contact_friction_articulation_pairs_only: int,
    row_capacity_telemetry: int,
    friction_patches: FrictionPatches,
    resolved_worlds: wp.array[int],
    # outputs
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    world_slot_counter: wp.array[int],
    contact_path: wp.array[int],
    mf_slot_counter: wp.array[int],
    contact_slots_needed: wp.array[int],
    dense_dropped_contact_rows: wp.array[int],
    mf_dropped_contact_rows: wp.array[int],
    dense_first_rejected_slot: wp.array[int],
    mf_first_rejected_slot: wp.array[int],
):
    """Allocate active contacts with work proportional to the materialized prefix.

    A collision pass clamps :paramref:`contact_count` to the rows its writers
    filled, every one of them, and flags an overflow in its own capacity
    status, so the count always describes a materialized prefix.

    Slot counters only grow. A contact whose reservation does not fit records its
    slot in the family's ``*_first_rejected_slot`` (per world, preset to a large
    value) instead of rolling the counter back: every later reservation starts at
    or past that slot, so the finalize kernels truncate the row count there and
    no accepted contact can hold a slot at or beyond the count. Rolling back
    raced with concurrent reservations and left accepted rows past the count,
    which the friction patch links then chained into.
    """
    thread = wp.tid()
    # A collision pass clamps the count to the rows it wrote, all of them, and flags the overflow itself.
    total_contacts = contact_count[0]

    for c in range(thread, total_contacts, total_num_threads):
        if resolved_worlds.shape[0] > 0:
            _body_a, _body_b, art_a, art_b, world = contact_endpoints(
                contact_shape0[c], contact_shape1[c], shape_body, body_to_articulation, art_to_world
            )
            if world >= 0 and world < resolved_worlds.shape[0] and resolved_worlds[world] != 0:
                # Refresh every active mapping each solve. The original force
                # converter publishes zero before reading any row for this path.
                contact_world[c] = world
                contact_art_a[c] = art_a
                contact_art_b[c] = art_b
                contact_slot[c] = -1
                contact_path[c] = -1
                contact_slots_needed[c] = 0
                _mark_resolved_patch_contact(
                    c,
                    contact_shape0,
                    contact_shape1,
                    contact_point0,
                    contact_point1,
                    contact_normal,
                    contact_thickness0,
                    contact_thickness1,
                    body_q,
                    shape_body,
                    body_to_articulation,
                    art_to_world,
                    articulation_response_dof_count,
                    body_flags,
                    body_has_response_dofs,
                    is_free_rigid,
                    contact_gap_gate,
                    same_articulation_contact_gap_gate,
                    articulation_pair_contact_gap_gate,
                    contact_slot,
                    contact_path,
                )
                continue
        _allocate_world_contact_slot(
            c,
            contact_shape0,
            contact_shape1,
            contact_point0,
            contact_point1,
            contact_normal,
            contact_thickness0,
            contact_thickness1,
            body_q,
            shape_body,
            body_to_articulation,
            art_to_world,
            articulation_response_dof_count,
            body_flags,
            body_has_response_dofs,
            is_free_rigid,
            has_free_rigid,
            contact_gap_gate,
            same_articulation_contact_gap_gate,
            articulation_pair_contact_gap_gate,
            max_constraints,
            mf_max_constraints,
            enable_friction,
            contact_friction_gap_threshold,
            contact_friction_articulation_pairs_only,
            row_capacity_telemetry,
            friction_patches,
            contact_world,
            contact_slot,
            contact_art_a,
            contact_art_b,
            world_slot_counter,
            contact_path,
            mf_slot_counter,
            contact_slots_needed,
            dense_dropped_contact_rows,
            mf_dropped_contact_rows,
            dense_first_rejected_slot,
            mf_first_rejected_slot,
        )


# The allocators above reserve each contact's rows with an atomic on its world's counter, so on CUDA a
# world's rows follow the order in which warps reach that counter, which varies between runs, and an
# unconverged Gauss-Seidel solve changes with it. Under a deterministic collision pipeline
# (Contacts.rigid_contacts_pair_sorted) the contacts arrive sorted by (shape0, shape1), and the two kernels
# below give every world's rows that order instead. Shapes are numbered world by world, with the global
# shapes before and after every world's (Model.shape_world_start), so a world's contacts form one contiguous
# run per block: the run its own shapes head, and one per global shape that heads its contacts with the
# world. record_world_contact_runs notes each run's first and last contact and rows, and
# place_world_contact_runs, one warp per world, places the runs block after block from the world's first
# contact row. A family whose reservations overflowed keeps the allocator's layout.

# The first contact of a run without contacts.
_CONTACT_RUN_EMPTY = wp.constant(2**31 - 1)


@wp.func
def _contact_run_block(shape: int, global_lead: int, global_trail_start: int) -> int:
    """Block of a contact whose first shape is ``shape``: its own if the shape is global, else the worlds' block."""
    if shape < global_lead:
        return wp.max(shape, 0)
    if shape < global_trail_start:
        return global_lead
    return global_lead + 1 + shape - global_trail_start


@wp.func_native(
    dual_native(
        """
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    // The lanes of one warp iteration hold ascending contacts; the lanes of one run form a group,
    // whose lowest lane records the group's first and last contact and rows.
    const unsigned group = __match_any_sync(0xffffffffu, key);
    const int leader = FIRST(group);
    const int first = READ(contact, leader);
    const int last = READ(contact, 31 - __clz(group));
#if __CUDA_ARCH__ >= 800
    const int dense_rows = __reduce_add_sync(group, dense);
    const int mf_rows = __reduce_add_sync(group, mf);
#else
    int dense_rows = 0;
    int mf_rows = 0;
    for (int source = 0; source < 32; ++source) {
        const int source_dense = READ(dense, source);
        const int source_mf = READ(mf, source);
        if (group & (1u << source)) {
            dense_rows += source_dense;
            mf_rows += source_mf;
        }
    }
#endif
    if (key >= 0 && LANE == leader) {
        atomicMin(&run_first.data[key], first);
        atomicMax(&run_last.data[key], last);
        if (dense_rows != 0)
            atomicAdd(&run_rows.data[key].c[0], dense_rows);
        if (mf_rows != 0)
            atomicAdd(&run_rows.data[key].c[1], mf_rows);
    }
#elif defined(__CUDA_ARCH__)
    if (key >= 0) {
        atomicMin(&run_first.data[key], contact);
        atomicMax(&run_last.data[key], contact);
        atomicAdd(&run_rows.data[key].c[0], dense);
        atomicAdd(&run_rows.data[key].c[1], mf);
    }
#else
    if (key >= 0) {
        if (contact < run_first.data[key])
            run_first.data[key] = contact;
        if (contact > run_last.data[key])
            run_last.data[key] = contact;
        run_rows.data[key].c[0] += dense;
        run_rows.data[key].c[1] += mf;
    }
#endif
""",
        preludes=(lane_group(32),),
    )
)
def _record_contact_run(
    key: int,
    contact: int,
    dense: int,
    mf: int,
    run_first: wp.array2d[int],
    run_last: wp.array2d[int],
    run_rows: wp.array2d[wp.vec2i],
):
    """Merge a contact into the run ``key`` (flat world and block index; -1 for none), once per run and warp on CUDA.

    Every lane of the complete warp must call it, with ascending contacts.
    """


@wp.kernel(enable_backward=False)
def record_world_contact_runs(
    contact_count: wp.array[int],
    total_num_threads: int,
    lanes: int,
    contact_shape0: wp.array[int],
    contact_world: wp.array[int],
    contact_path: wp.array[int],
    contact_slots_needed: wp.array[int],
    order_dense: int,
    order_mf: int,
    global_lead: int,
    global_trail_start: int,
    # outputs
    run_first: wp.array2d[int],
    run_last: wp.array2d[int],
    run_rows: wp.array2d[wp.vec2i],
):
    """Record each (world, block) run's first and last routed contact and its rows (dense, matrix-free).

    ``total_num_threads`` is a multiple of ``lanes`` (32 on CUDA, 1 on CPU), so each warp
    iteration covers 32 consecutive contacts.
    """
    thread = wp.tid()
    lane = thread % lanes
    total_contacts = contact_count[0]
    blocks = run_rows.shape[1]
    for base in range(thread - lane, total_contacts, total_num_threads):
        c = base + lane
        key = int(-1)
        dense = int(0)
        mf = int(0)
        if c < total_contacts:
            path = contact_path[c]
            world = contact_world[c]
            shape = contact_shape0[c]
            needed = contact_slots_needed[c]
            if path == 0 and order_dense != 0:
                dense = needed
            if path == 1 and order_mf != 0:
                mf = needed
            if dense > 0 or mf > 0:
                key = world * blocks + _contact_run_block(shape, global_lead, global_trail_start)
        _record_contact_run(key, c, dense, mf, run_first, run_last, run_rows)


@wp.kernel(enable_backward=False)
def place_world_contact_runs(
    contact_shape0: wp.array[int],
    contact_world: wp.array[int],
    contact_path: wp.array[int],
    contact_slots_needed: wp.array[int],
    dense_slot_counter: wp.array[int],
    dense_first_rejected_slot: wp.array[int],
    mf_slot_counter: wp.array[int],
    mf_first_rejected_slot: wp.array[int],
    global_lead: int,
    global_trail_start: int,
    lanes: int,
    # outputs
    run_first: wp.array2d[int],
    run_last: wp.array2d[int],
    run_rows: wp.array2d[wp.vec2i],
    contact_slot: wp.array[int],
):
    """Place each world's recorded contact runs in block order and reset the records for the next step.

    Launched over ``(world_count, lanes)``: one complete warp per world on CUDA (``lanes`` 32),
    one thread on CPU (``lanes`` 1). Each family's runs start at the slot where the allocator's
    counter began, which the counter less the family's rows recovers; a family whose
    reservations overflowed (a first rejected slot below its counter) keeps the allocator's
    slots, all of which lie below that slot.
    """
    world, lane = wp.tid()
    blocks = run_rows.shape[1]
    total = wp.vec2i(0, 0)
    for block in range(blocks):
        total += run_rows[world, block]
    place_dense = total[0] > 0 and dense_first_rejected_slot[world] >= dense_slot_counter[world]
    place_mf = total[1] > 0 and mf_first_rejected_slot[world] >= mf_slot_counter[world]
    next_dense = dense_slot_counter[world] - total[0]
    next_mf = mf_slot_counter[world] - total[1]
    for block in range(blocks):
        last = run_last[world, block]
        if not (place_dense or place_mf):
            last = -1
        for base in range(run_first[world, block], last + 1, lanes):
            c = base + lane
            dense = int(0)
            mf = int(0)
            if c <= last:
                path = contact_path[c]
                owner = contact_world[c]
                shape = contact_shape0[c]
                needed = contact_slots_needed[c]
                if owner == world and _contact_run_block(shape, global_lead, global_trail_start) == block:
                    if path == 0 and place_dense:
                        dense = needed
                    if path == 1 and place_mf:
                        mf = needed
            dense_below = warp_scan(dense)
            mf_below = warp_scan(mf)
            if dense > 0:
                contact_slot[c] = next_dense + dense_below
            if mf > 0:
                contact_slot[c] = next_mf + mf_below
            next_dense += warp_sum(dense)
            next_mf += warp_sum(mf)
    # Every lane has read the records before any is reset.
    warp_sync()
    for block in range(lane, blocks, lanes):
        run_first[world, block] = _CONTACT_RUN_EMPTY
        run_last[world, block] = -1
        run_rows[world, block] = wp.vec2i(0, 0)


@wp.func
def accumulate_jacobian_row_world(
    body_index: int,
    sign: float,
    point_world: wp.vec3,
    origin: wp.vec3,
    direction: wp.vec3,
    body_to_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    art_dof_start: int,
    n_dofs: int,
    group_idx: int,
    row: int,
    J_group: wp.array3d[float],
):
    """Accumulate Jacobian contributions by walking up the kinematic tree."""
    if body_index < 0:
        return

    curr_joint = body_to_joint[body_index]
    while curr_joint >= 0:
        for global_dof in range(joint_qd_start[curr_joint], joint_qd_start[curr_joint + 1]):
            local_dof = global_dof - art_dof_start
            if local_dof >= 0 and local_dof < n_dofs:
                entry = jacobian_entry(joint_S_s[global_dof], direction, point_world, origin)
                J_group[group_idx, row, local_dof] += sign * entry
        curr_joint = joint_ancestor[curr_joint]


@wp.func
def _write_dense_contact_row(
    c: int,
    world: int,
    slot: int,
    row: int,
    mu: float,
    phi: float,
    restitution: float,
    target: float,
    contact_friction_scale: float,
    pgs_beta: float,
    pgs_cfm: float,
    friction_patches: FrictionPatches,
    friction_anchor_beta: float,
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_cfm: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
    world_row_restitution: wp.array2d[float],
):
    """Write the metadata of contact ``c``'s row ``row`` (0 the normal, 1 and 2 the tangents) at ``slot + row``.

    A tangent row carries the patch anchor's tangential displacement as its gap, with
    ``friction_anchor_beta`` and the scaled friction coefficient.
    """
    r = slot + row
    if row == 0:
        world_row_type[world, r] = PGS_CONSTRAINT_TYPE_CONTACT
        world_row_parent[world, r] = -1
        world_row_mu[world, r] = mu
        world_row_beta[world, r] = pgs_beta
        world_phi[world, r] = phi
        world_row_restitution[world, r] = restitution
    else:
        world_row_type[world, r] = PGS_CONSTRAINT_TYPE_FRICTION
        world_row_parent[world, r] = slot
        world_row_mu[world, r] = mu * contact_friction_scale
        world_row_beta[world, r] = friction_anchor_beta
        world_phi[world, r] = friction_patches.phi[c][row - 1]
        world_row_restitution[world, r] = 0.0
    world_row_cfm[world, r] = pgs_cfm
    world_target_velocity[world, r] = target


@wp.kernel
def prepare_world_contact_rows(
    contact_count: wp.array[int],
    total_num_threads: int,
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    contact_path: wp.array[int],
    contact_slots_needed: wp.array[int],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_v_s: wp.array[wp.spatial_vector],
    prescribed_articulation: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    shape_material_mu: wp.array[float],
    shape_material_restitution: wp.array[float],
    contact_friction_shared_anchor: int,
    contact_friction_scale: float,
    contact_shared_anchor: int,
    pgs_beta: float,
    pgs_cfm: float,
    friction_patches: FrictionPatches,
    friction_anchor_beta: float,
    # outputs
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_cfm: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
    world_row_restitution: wp.array2d[float],
):
    """Build current metadata inside each allocator-recorded contact row reservation."""
    total_contacts = contact_count[0]
    for c in range(wp.tid(), total_contacts, total_num_threads):
        slot = contact_slot[c]
        if contact_path[c] != 0 or slot < 0:
            continue
        art_a = contact_art_a[c]
        art_b = contact_art_b[c]
        body_a, body_b, normal, point_a, point_b, phi = contact_geometry(
            c,
            contact_shape0,
            contact_shape1,
            contact_point0,
            contact_point1,
            contact_normal,
            contact_thickness0,
            contact_thickness1,
            shape_body,
            body_q,
        )
        mu, restitution = contact_materials(
            contact_shape0[c], contact_shape1[c], shape_material_mu, shape_material_restitution
        )
        tangent0, tangent1 = contact_tangent_basis(normal)
        friction_a, friction_b = contact_friction_points(
            point_a, point_b, contact_shared_anchor, contact_friction_shared_anchor
        )
        if friction_patches.enabled != 0:
            friction_a, friction_b = patch_friction_location(friction_patches, c)
        # Allocation owns the row extent; recomputing eligibility can cross a
        # floating-point threshold and overwrite the following contact.
        rows = 1
        if contact_slots_needed[c] >= 3:
            rows = 3
        for row in range(rows):
            direction, p0, p1 = contact_row_frame(
                row, contact_shared_anchor, normal, tangent0, tangent1, point_a, point_b, friction_a, friction_b
            )
            target = prescribed_target(
                body_a, art_a, body_b, art_b, p0, p1, direction, prescribed_articulation, articulation_origin, body_v_s
            )
            _write_dense_contact_row(
                c,
                contact_world[c],
                slot,
                row,
                mu,
                phi,
                restitution,
                target,
                contact_friction_scale,
                pgs_beta,
                pgs_cfm,
                friction_patches,
                friction_anchor_beta,
                world_row_type,
                world_row_parent,
                world_row_mu,
                world_row_beta,
                world_row_cfm,
                world_phi,
                world_target_velocity,
                world_row_restitution,
            )


@wp.kernel
def populate_world_J_for_compact_size(
    contact_count: wp.array[int],
    total_num_workers: int,
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    contact_path: wp.array[int],
    contact_slots_needed: wp.array[int],
    target_size: int,
    articulation_response_dof_count: wp.array[int],
    art_group_idx: wp.array[int],
    art_dof_start: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_response_dof_mask: wp.array[wp.uint32],
    joint_S_s: wp.array[wp.spatial_vector],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    contact_friction_shared_anchor: int,
    friction_patches: FrictionPatches,
    contact_shared_anchor: int,
    # output
    J_group: wp.array3d[float],
):
    """Project one dense contact row/DOF per lane for compact articulations."""
    worker, lane = wp.tid()
    total_contacts = contact_count[0]
    row = lane // target_size
    local_dof = lane - row * target_size
    if row >= 3 or local_dof >= target_size:
        return

    for c in range(worker, total_contacts, total_num_workers):
        if contact_path[c] != 0 or contact_slot[c] < 0 or row >= contact_slots_needed[c]:
            continue
        body_a, body_b, normal, point_a, point_b, _phi = contact_geometry(
            c,
            contact_shape0,
            contact_shape1,
            contact_point0,
            contact_point1,
            contact_normal,
            contact_thickness0,
            contact_thickness1,
            shape_body,
            body_q,
        )
        tangent0 = wp.vec3(0.0)
        tangent1 = wp.vec3(0.0)
        if row > 0:
            tangent0, tangent1 = contact_tangent_basis(normal)
        friction_a, friction_b = contact_friction_points(
            point_a, point_b, contact_shared_anchor, contact_friction_shared_anchor
        )
        if friction_patches.enabled != 0:
            friction_a, friction_b = patch_friction_location(friction_patches, c)
        direction, p0, p1 = contact_row_frame(
            row, contact_shared_anchor, normal, tangent0, tangent1, point_a, point_b, friction_a, friction_b
        )

        art_a = contact_art_a[c]
        art_b = contact_art_b[c]
        group_a = -1
        group_b = -1
        value_a = float(0.0)
        value_b = float(0.0)
        bit = wp.uint32(1) << wp.uint32(local_dof)
        if art_a >= 0 and articulation_response_dof_count[art_a] == target_size:
            group_a = art_group_idx[art_a]
            if body_a >= 0 and (body_response_dof_mask[body_a] & bit) != wp.uint32(0):
                motion_a = joint_S_s[art_dof_start[art_a] + local_dof]
                value_a = jacobian_entry(motion_a, direction, p0, articulation_origin[art_a])
        if art_b >= 0 and articulation_response_dof_count[art_b] == target_size:
            group_b = art_group_idx[art_b]
            if body_b >= 0 and (body_response_dof_mask[body_b] & bit) != wp.uint32(0):
                motion_b = joint_S_s[art_dof_start[art_b] + local_dof]
                value_b = -jacobian_entry(motion_b, direction, p1, articulation_origin[art_b])

        slot = contact_slot[c] + row
        if group_a >= 0:
            if group_b == group_a:
                J_group[group_a, slot, local_dof] = value_a + value_b
            else:
                J_group[group_a, slot, local_dof] = value_a
        if group_b >= 0 and group_b != group_a:
            J_group[group_b, slot, local_dof] = value_b


@wp.kernel
def clear_sparse_diagonal_response_prefix(
    dense_phase_bounds: wp.array2d[int],
    # in/out
    sparse_row_dof: wp.array3d[int],
):
    """Remove stale sparse entries from the non-contact row prefix."""
    world = wp.tid()
    for row in range(dense_phase_bounds[world, 1]):
        sparse_row_dof[world, row, 0] = -1
        sparse_row_dof[world, row, 1] = -1


@wp.kernel
def populate_sparse_diagonal_contact_response(
    contact_count: wp.array[int],
    total_num_workers: int,
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    contact_path: wp.array[int],
    contact_slots_needed: wp.array[int],
    target_size: int,
    articulation_response_dof_count: wp.array[int],
    articulation_dof_start: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_single_response_dof: wp.array[int],
    diagonal_inverse_mass: wp.array[float],
    joint_S_s: wp.array[wp.spatial_vector],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    contact_friction_shared_anchor: int,
    friction_patches: FrictionPatches,
    contact_shared_anchor: int,
    # outputs
    sparse_row_dof: wp.array3d[int],
    sparse_row_jy: wp.array3d[float],
):
    """Build exact ``dense + two sparse`` contact responses for an independent articulation.

    Tangent rows use the persistent patch anchors when patch friction is enabled, exactly as the dense
    contact Jacobian builders do, so the sparse coordinates see the same row geometry.
    """
    worker = wp.tid()
    total_contacts = contact_count[0]
    for c in range(worker, total_contacts, total_num_workers):
        slot = contact_slot[c]
        if contact_path[c] != 0 or slot < 0:
            continue
        body_a, body_b, normal, point_a, point_b, _phi = contact_geometry(
            c,
            contact_shape0,
            contact_shape1,
            contact_point0,
            contact_point1,
            contact_normal,
            contact_thickness0,
            contact_thickness1,
            shape_body,
            body_q,
        )
        tangent0, tangent1 = contact_tangent_basis(normal)
        friction_a, friction_b = contact_friction_points(
            point_a, point_b, contact_shared_anchor, contact_friction_shared_anchor
        )
        if friction_patches.enabled != 0:
            friction_a, friction_b = patch_friction_location(friction_patches, c)
        world = contact_world[c]
        art_a = contact_art_a[c]
        art_b = contact_art_b[c]
        rows = contact_slots_needed[c]

        for row in range(3):
            if row >= rows:
                continue
            direction, p0, p1 = contact_row_frame(
                row, contact_shared_anchor, normal, tangent0, tangent1, point_a, point_b, friction_a, friction_b
            )
            dof_a = -1
            dof_b = -1
            coord_a = -1
            coord_b = -1
            value_a = float(0.0)
            value_b = float(0.0)
            if art_a >= 0 and body_a >= 0 and articulation_response_dof_count[art_a] == target_size:
                dof_a = body_single_response_dof[body_a]
                if dof_a >= 0:
                    local_dof_a = dof_a - articulation_dof_start[art_a]
                    if local_dof_a >= 0 and local_dof_a < target_size:
                        value_a = jacobian_entry(joint_S_s[dof_a], direction, p0, articulation_origin[art_a])
                        coord_a = articulation_world_dof_offset[art_a] + local_dof_a

            if art_b >= 0 and body_b >= 0 and articulation_response_dof_count[art_b] == target_size:
                dof_b = body_single_response_dof[body_b]
                if dof_b >= 0:
                    local_dof_b = dof_b - articulation_dof_start[art_b]
                    if local_dof_b >= 0 and local_dof_b < target_size:
                        value_b = -jacobian_entry(joint_S_s[dof_b], direction, p1, articulation_origin[art_b])
                        coord_b = articulation_world_dof_offset[art_b] + local_dof_b

            response_a = float(0.0)
            response_b = float(0.0)
            if dof_a >= 0:
                response_a = value_a * diagonal_inverse_mass[dof_a]
            if dof_b >= 0:
                response_b = value_b * diagonal_inverse_mass[dof_b]
            if coord_a >= 0 and coord_a == coord_b:
                value_a += value_b
                response_a += response_b
                coord_b = -1
                value_b = 0.0
                response_b = 0.0

            output_row = slot + row
            sparse_row_dof[world, output_row, 0] = coord_a
            sparse_row_dof[world, output_row, 1] = coord_b
            sparse_row_jy[world, output_row, 0] = value_a
            sparse_row_jy[world, output_row, 1] = response_a
            sparse_row_jy[world, output_row, 2] = value_b
            sparse_row_jy[world, output_row, 3] = response_b


@wp.kernel
def accumulate_sparse_diagonal_response_diag(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    sparse_row_dof: wp.array3d[int],
    sparse_row_jy: wp.array3d[float],
    # in/out
    world_diag: wp.array2d[float],
):
    """Accumulate the two sparse response entries into each active row diagonal."""
    tid = wp.tid()
    world = tid // max_constraints
    row = tid - world * max_constraints
    if row >= world_constraint_count[world]:
        return
    value = float(0.0)
    if sparse_row_dof[world, row, 0] >= 0:
        value += sparse_row_jy[world, row, 0] * sparse_row_jy[world, row, 1]
    if sparse_row_dof[world, row, 1] >= 0:
        value += sparse_row_jy[world, row, 2] * sparse_row_jy[world, row, 3]
    world_diag[world, row] += value


@wp.kernel
def apply_sparse_diagonal_contact_restitution_matrix_free(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    world_phi: wp.array2d[float],
    world_row_type: wp.array2d[int],
    world_target_velocity: wp.array2d[float],
    world_row_restitution: wp.array2d[float],
    world_incident_velocity: wp.array[float],
    world_dof_indices: wp.array2d[int],
    dense_offsets: wp.array[int],
    dense_groups: wp.array[int],
    dense_dofs: int,
    dense_J: wp.array3d[float],
    sparse_row_dof: wp.array3d[int],
    sparse_row_jy: wp.array3d[float],
    dt: float,
    restitution_velocity_threshold: float,
    # in/out
    world_rhs: wp.array2d[float],
):
    """Apply restitution from a compact dense component plus two sparse response coordinates."""
    tid = wp.tid()
    world = tid // max_constraints
    row = tid - world * max_constraints
    if row >= world_constraint_count[world] or world_row_type[world, row] != PGS_CONSTRAINT_TYPE_CONTACT:
        return

    restitution = world_row_restitution[world, row]
    if restitution <= 0.0:
        return

    relative_incident = float(0.0)
    dense_offset = dense_offsets[world]
    dense_group = dense_groups[world]
    for local_dof in range(dense_dofs):
        world_dof = dense_offset + local_dof
        global_dof = world_dof_indices[world, world_dof]
        if global_dof >= 0:
            relative_incident += dense_J[dense_group, row, local_dof] * world_incident_velocity[global_dof]
    for sparse_slot in range(2):
        world_dof = sparse_row_dof[world, row, sparse_slot]
        if world_dof >= 0:
            global_dof = world_dof_indices[world, world_dof]
            if global_dof >= 0:
                relative_incident += sparse_row_jy[world, row, sparse_slot * 2] * world_incident_velocity[global_dof]

    target_vel = world_target_velocity[world, row]
    relative_incident -= target_vel
    fires, bounce = contact_restitution_rhs(
        world_phi[world, row], relative_incident, target_vel, restitution, dt, restitution_velocity_threshold
    )
    if fires:
        world_rhs[world, row] = bounce


@wp.func
def world_contact_row_dot(
    world_dof_count: wp.array[int],
    world_dof_indices: wp.array2d[int],
    world_J: wp.array3d[float],
    velocity: wp.array[float],
    world: int,
    i: int,
):
    """Dot one dense world row's Jacobian with a global velocity state."""
    out = float(0.0)
    for d in range(world_dof_count[world]):
        global_dof = world_dof_indices[world, d]
        if global_dof >= 0:
            out += world_J[world, i, d] * velocity[global_dof]
    return out


@wp.func
def mf_contact_row_dot(
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    dof_a: int,
    dof_b: int,
    world_dof_indices: wp.array2d[int],
    velocity: wp.array[float],
    world: int,
    i: int,
):
    """Dot one matrix-free row's two body Jacobians with a global velocity."""
    out = float(0.0)
    if dof_a >= 0:
        for k in range(6):
            global_dof = world_dof_indices[world, dof_a + k]
            if global_dof >= 0:
                out += mf_J_a[world, i, k] * velocity[global_dof]
    if dof_b >= 0:
        for k in range(6):
            global_dof = world_dof_indices[world, dof_b + k]
            if global_dof >= 0:
                out += mf_J_b[world, i, k] * velocity[global_dof]
    return out


@wp.func
def _populate_world_J_for_size_contact(
    c: int,
    total_contacts: int,
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    contact_path: wp.array[int],
    contact_slots_needed: wp.array[int],
    target_size: int,
    articulation_response_dof_count: wp.array[int],
    art_group_idx: wp.array[int],
    art_dof_start: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_to_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_v_s: wp.array[wp.spatial_vector],
    prescribed_articulation: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_material_mu: wp.array[float],
    shape_material_restitution: wp.array[float],
    contact_friction_shared_anchor: int,
    contact_friction_scale: float,
    contact_shared_anchor: int,
    pgs_beta: float,
    pgs_cfm: float,
    friction_patches: FrictionPatches,
    friction_anchor_beta: float,
    # outputs
    J_group: wp.array3d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_cfm: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
    world_row_restitution: wp.array2d[float],
):
    """Populate one contact for a specific articulation-size group.

    Each contact may contribute to multiple articulations' Jacobian matrices.
    Contacts routed to the matrix-free path (contact_path==1) are skipped.
    """
    slot = contact_slot[c]
    if contact_path[c] != 0 or slot < 0:
        return
    art_a = contact_art_a[c]
    art_b = contact_art_b[c]
    body_a, body_b, normal, point_a, point_b, phi = contact_geometry(
        c,
        contact_shape0,
        contact_shape1,
        contact_point0,
        contact_point1,
        contact_normal,
        contact_thickness0,
        contact_thickness1,
        shape_body,
        body_q,
    )
    mu, restitution = contact_materials(
        contact_shape0[c], contact_shape1[c], shape_material_mu, shape_material_restitution
    )
    tangent0, tangent1 = contact_tangent_basis(normal)
    friction_a, friction_b = contact_friction_points(
        point_a, point_b, contact_shared_anchor, contact_friction_shared_anchor
    )
    if friction_patches.enabled != 0:
        friction_a, friction_b = patch_friction_location(friction_patches, c)
    a_matches = art_a >= 0 and articulation_response_dof_count[art_a] == target_size
    b_matches = art_b >= 0 and articulation_response_dof_count[art_b] == target_size
    # Allocation owns the row extent; recomputing eligibility can cross a
    # floating-point threshold and overwrite the following contact.
    rows = 1
    if contact_slots_needed[c] >= 3:
        rows = 3
    for row in range(rows):
        direction, p0, p1 = contact_row_frame(
            row, contact_shared_anchor, normal, tangent0, tangent1, point_a, point_b, friction_a, friction_b
        )
        if a_matches:
            accumulate_jacobian_row_world(
                body_a,
                1.0,
                p0,
                articulation_origin[art_a],
                direction,
                body_to_joint,
                joint_ancestor,
                joint_qd_start,
                joint_S_s,
                art_dof_start[art_a],
                target_size,
                art_group_idx[art_a],
                slot + row,
                J_group,
            )
        if b_matches:
            accumulate_jacobian_row_world(
                body_b,
                -1.0,
                p1,
                articulation_origin[art_b],
                direction,
                body_to_joint,
                joint_ancestor,
                joint_qd_start,
                joint_S_s,
                art_dof_start[art_b],
                target_size,
                art_group_idx[art_b],
                slot + row,
                J_group,
            )
        # The row metadata, from the size group of either endpoint.
        if a_matches or b_matches:
            target = prescribed_target(
                body_a, art_a, body_b, art_b, p0, p1, direction, prescribed_articulation, articulation_origin, body_v_s
            )
            _write_dense_contact_row(
                c,
                contact_world[c],
                slot,
                row,
                mu,
                phi,
                restitution,
                target,
                contact_friction_scale,
                pgs_beta,
                pgs_cfm,
                friction_patches,
                friction_anchor_beta,
                world_row_type,
                world_row_parent,
                world_row_mu,
                world_row_beta,
                world_row_cfm,
                world_phi,
                world_target_velocity,
                world_row_restitution,
            )


@wp.kernel
def populate_world_J_for_size(
    contact_count: wp.array[int],
    total_num_threads: int,
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    contact_path: wp.array[int],
    contact_slots_needed: wp.array[int],
    target_size: int,
    articulation_response_dof_count: wp.array[int],
    art_group_idx: wp.array[int],
    art_dof_start: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_to_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_v_s: wp.array[wp.spatial_vector],
    prescribed_articulation: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_material_mu: wp.array[float],
    shape_material_restitution: wp.array[float],
    contact_friction_shared_anchor: int,
    contact_friction_scale: float,
    contact_shared_anchor: int,
    pgs_beta: float,
    pgs_cfm: float,
    friction_patches: FrictionPatches,
    friction_anchor_beta: float,
    # outputs
    J_group: wp.array3d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_cfm: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
    world_row_restitution: wp.array2d[float],
):
    """Populate active dense contacts with a capacity-independent launch."""
    total_contacts = contact_count[0]
    for c in range(wp.tid(), total_contacts, total_num_threads):
        _populate_world_J_for_size_contact(
            c,
            total_contacts,
            contact_point0,
            contact_point1,
            contact_normal,
            contact_shape0,
            contact_shape1,
            contact_thickness0,
            contact_thickness1,
            contact_world,
            contact_slot,
            contact_art_a,
            contact_art_b,
            contact_path,
            contact_slots_needed,
            target_size,
            articulation_response_dof_count,
            art_group_idx,
            art_dof_start,
            articulation_origin,
            body_to_joint,
            joint_ancestor,
            joint_qd_start,
            joint_S_s,
            shape_body,
            body_q,
            body_v_s,
            prescribed_articulation,
            shape_transform,
            shape_material_mu,
            shape_material_restitution,
            contact_friction_shared_anchor,
            contact_friction_scale,
            contact_shared_anchor,
            pgs_beta,
            pgs_cfm,
            friction_patches,
            friction_anchor_beta,
            J_group,
            world_row_type,
            world_row_parent,
            world_row_mu,
            world_row_beta,
            world_row_cfm,
            world_phi,
            world_target_velocity,
            world_row_restitution,
        )


@wp.kernel
def finalize_constraint_counts_with_status(
    slot_counter: wp.array[int],
    capacity: int,
    family: int,
    first_rejected_slot: wp.array[int],
    constraint_count: wp.array[int],
    capacity_status: wp.array[int],
):
    """Clamp the accepted prefix while retaining sticky overflow from every row producer."""
    world = wp.tid()
    constraint_count[world] = reserved_count(
        slot_counter, world, capacity, first_rejected_slot, capacity_status, family
    )


@wp.kernel
def snapshot_dense_phase_bound(
    world_slot_counter: wp.array[int],
    bound_index: int,
    # outputs
    dense_phase_bounds: wp.array2d[int],
):
    """Record the current dense slot watermark for one row-family boundary.

    The dense per-world slot layout is [drive][joint-limit][joint-vel-limit]
    [contact/friction]. Launched right after the joint-limit allocation
    (``bound_index`` 0) and the joint-velocity-limit allocation
    (``bound_index`` 1) so the PhysX-grasp GS phases can loop only their own
    contiguous row range instead of scanning every dense row. The raw counter
    is stored; consumers clamp against the per-world constraint count.
    """
    world = wp.tid()
    dense_phase_bounds[world, bound_index] = world_slot_counter[world]


@wp.kernel
def apply_augmented_mass_diagonal_grouped(
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    n_dofs: int,
    max_dofs: int,
    mass_update_mask: wp.array[int],
    row_counts: wp.array[int],
    row_dof_index: wp.array[int],
    row_K: wp.array[float],
    # outputs
    H_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
):
    """Apply augmented mass diagonal for grouped H storage."""
    idx = wp.tid()
    articulation = group_to_art[idx]

    if mass_update_mask[articulation] == 0:
        return

    count = row_counts[articulation]
    if count == 0:
        return

    dof_start = articulation_dof_start[articulation]

    for i in range(count):
        row_index = articulation * max_dofs + i
        dof = row_dof_index[row_index]
        local = dof - dof_start
        if local < 0 or local >= n_dofs:
            continue

        K = row_K[row_index]
        if K <= 0.0:
            continue

        H_group[idx, local, local] += K


# --- Tile configuration for contact system build ---
# Kernel naming: {op}_{parallelism}
# parallelism: tiled | loop | par_row | par_row_col | par_dof

# Max generalized dofs per articulation we support in the tiled path.
# joint_dof_count per articulation must be <= TILE_DOF or we use fall back
TILE_DOF = wp.constant(49)

# Max constraints per articulation we support in the tiled path.
# dense_max_constraints must be <= TILE_CONSTRAINTS or we use fall back
TILE_CONSTRAINTS = wp.constant(128)

# Threads per tile/block for tile kernels
TILE_THREADS = 64


@wp.kernel
def update_body_qd_from_featherstone(
    body_v_s: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_to_articulation: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_qd_out: wp.array[wp.spatial_vector],
):
    tid = wp.tid()

    twist = body_v_s[tid]  # spatial twist about origin
    v0 = wp.spatial_top(twist)
    w = wp.spatial_bottom(twist)

    X_wb = body_q[tid]
    com_local = body_com[tid]
    com_world = wp.transform_point(X_wb, com_local)
    art = body_to_articulation[tid]
    origin = wp.vec3()
    if art >= 0:
        origin = articulation_origin[art]
    com_rel = com_world - origin

    v_com = v0 + wp.cross(w, com_rel)

    body_qd_out[tid] = wp.spatial_vector(v_com, w)


# =============================================================================
# World-Level PGS and Velocity Kernels for Multi-Articulation
# =============================================================================


@wp.kernel
def compute_world_contact_bias(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    world_phi: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_type: wp.array2d[int],
    world_target_velocity: wp.array2d[float],
    dt: float,
    bias_scale: float,
    contact_speculative_scale: float,
    joint_limit_speculative_scale: float,
    contact_w: float,
    # outputs
    world_rhs: wp.array2d[float],
    world_row_w: wp.array2d[float],
):
    """Compute the RHS bias term for world-level PGS solve: each row's position law (:func:`.row_laws.row_rhs`).

    The RHS follows the convention: rhs = J*v + stabilization; a penetrating contact's
    Baumgarte term ``bias_scale * beta * phi / dt`` is negative and leads to positive
    impulses when resolved by PGS. ``J*v`` is added later.

    One thread per (world, row): ``dim = world_count * max_constraints``. The previous one-thread-per-world loop
    left the GPU idle (64-128 blocks) for 30-60 us per launch.
    """
    tid = wp.tid()
    world = tid // max_constraints
    i = tid - world * max_constraints
    m = world_constraint_count[world]

    inv_dt = 1.0 / dt

    if i < m:
        rhs, row_w = row_rhs(
            world_row_type[world, i],
            world_phi[world, i],
            world_row_beta[world, i],
            world_target_velocity[world, i],
            inv_dt,
            bias_scale,
            contact_speculative_scale,
            joint_limit_speculative_scale,
            contact_w,
        )
        world_rhs[world, i] = rhs
        if contact_w < 1.0:
            world_row_w[world, i] = row_w


@wp.kernel
def apply_world_contact_restitution_matrix_free(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    world_dof_count: wp.array[int],
    world_phi: wp.array2d[float],
    world_row_type: wp.array2d[int],
    world_target_velocity: wp.array2d[float],
    world_row_restitution: wp.array2d[float],
    world_incident_velocity: wp.array[float],
    world_dof_indices: wp.array2d[int],
    world_J: wp.array3d[float],
    dt: float,
    restitution_velocity_threshold: float,
    write_row_w: int,
    # in/out
    world_rhs: wp.array2d[float],
    world_row_w: wp.array2d[float],
):
    """Replace a matrix-free contact bias with a one-shot restitution target."""
    tid = wp.tid()
    world = tid // max_constraints
    i = tid - world * max_constraints
    m = world_constraint_count[world]
    if i >= m:
        return
    if world_row_type[world, i] != PGS_CONSTRAINT_TYPE_CONTACT:
        return

    restitution = world_row_restitution[world, i]
    if restitution <= 0.0:
        return

    phi = world_phi[world, i]
    target_vel = world_target_velocity[world, i]
    relative_incident = (
        world_contact_row_dot(world_dof_count, world_dof_indices, world_J, world_incident_velocity, world, i)
        - target_vel
    )

    fires, bounce = contact_restitution_rhs(
        phi, relative_incident, target_vel, restitution, dt, restitution_velocity_threshold
    )
    if fires:
        # Matrix-free GS adds live J*v itself, so store only
        # -target + e*u_incident as the row bias.
        world_rhs[world, i] = bounce
        if write_row_w != 0:
            world_row_w[world, i] = 1.0


@wp.kernel
def apply_world_contact_restitution_accumulated(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    world_phi: wp.array2d[float],
    world_row_beta: wp.array2d[float],
    world_row_type: wp.array2d[int],
    world_row_restitution: wp.array2d[float],
    dt: float,
    contact_speculative_scale: float,
    restitution_velocity_threshold: float,
    write_row_w: int,
    # in/out
    world_rhs: wp.array2d[float],
    world_row_w: wp.array2d[float],
):
    """Replace an accumulated impulse-space contact RHS with restitution."""
    tid = wp.tid()
    world = tid // max_constraints
    i = tid - world * max_constraints
    if i >= world_constraint_count[world]:
        return
    if world_row_type[world, i] != PGS_CONSTRAINT_TYPE_CONTACT:
        return

    restitution = world_row_restitution[world, i]
    if restitution <= 0.0:
        return

    phi = world_phi[world, i]
    # The geometric bias the position pass added (bias scale 1): the contact law without a target.
    geometric_bias, _row_w = contact_rhs(
        phi, world_row_beta[world, i], 0.0, 1.0 / dt, 1.0, contact_speculative_scale, 1.0, 0.0, 0.0
    )
    relative_incident = world_rhs[world, i] - geometric_bias
    if contact_restitution_fires(phi, relative_incident, dt, restitution_velocity_threshold):
        # Impulse-space RHS contains u_incident already.  Replacing geometric
        # bias with the Newton target yields (1+e)*u_incident.
        world_rhs[world, i] = (1.0 + restitution) * relative_incident
        if write_row_w != 0:
            world_row_w[world, i] = 1.0


@wp.kernel
def compute_world_contact_velocity_bias(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    world_dof_count: wp.array[int],
    world_phi: wp.array2d[float],
    world_row_type: wp.array2d[int],
    world_target_velocity: wp.array2d[float],
    world_row_restitution: wp.array2d[float],
    world_position_velocity: wp.array[float],
    world_incident_velocity: wp.array[float],
    world_dof_indices: wp.array2d[int],
    world_J: wp.array3d[float],
    dt: float,
    apply_restitution: int,
    restitution_velocity_threshold: float,
    # outputs
    world_rhs: wp.array2d[float],
):
    """Build the velocity-pass RHS from the position solution's end gap."""
    tid = wp.tid()
    world = tid // max_constraints
    i = tid - world * max_constraints
    if i >= world_constraint_count[world]:
        return
    inv_dt = 1.0 / dt

    phi = world_phi[world, i]
    row_type = world_row_type[world, i]
    target_vel = world_target_velocity[world, i]
    rhs = -target_vel

    # The velocity pass keeps no Baumgarte term: a closing contact rebounds by restitution, a separated one keeps
    # its speculative bias while the position solution's end gap stays open, a limit allows motion up to its bound.
    if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
        fires = bool(False)
        bounce = float(0.0)
        restitution = float(0.0)
        if apply_restitution != 0:
            restitution = world_row_restitution[world, i]
        if restitution > 0.0:
            relative_incident = (
                world_contact_row_dot(world_dof_count, world_dof_indices, world_J, world_incident_velocity, world, i)
                - target_vel
            )
            fires, bounce = contact_restitution_rhs(
                phi, relative_incident, target_vel, restitution, dt, restitution_velocity_threshold
            )
        if fires:
            rhs = bounce
        else:
            end_gap = float(0.0)
            if phi > 0.0:
                jv_position = world_contact_row_dot(
                    world_dof_count, world_dof_indices, world_J, world_position_velocity, world, i
                )
                end_gap = phi + dt * (jv_position - target_vel)
            rhs, _row_w = contact_rhs(phi, 0.0, target_vel, inv_dt, 0.0, 0.0, 1.0, 0.0, end_gap)
    elif row_type == PGS_CONSTRAINT_TYPE_JOINT_LIMIT:
        rhs = limit_rhs(phi, 0.0, target_vel, inv_dt, 0.0, 1.0)
    elif row_type == PGS_CONSTRAINT_TYPE_JOINT_TARGET:
        rhs = 0.0

    world_rhs[world, i] = rhs


@wp.kernel
def rhs_accum_world_par_art(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    art_to_world: wp.array[int],
    art_dof_start: wp.array[int],
    v_hat: wp.array[float],
    group_to_art: wp.array[int],
    J_group: wp.array3d[float],
    n_dofs: int,
    # outputs
    world_rhs: wp.array2d[float],
):
    """
    Accumulate J*v_hat into world RHS for a single size group.

    RHS = J*v + stabilization (already includes stabilization from compute_world_contact_bias)
    This kernel is launched once per size group to accumulate velocity contributions.
    """
    idx = wp.tid()
    art = group_to_art[idx]
    world = art_to_world[art]
    n_constraints = world_constraint_count[world]

    if n_constraints == 0:
        return

    dof_start = art_dof_start[art]

    for c in range(n_constraints):
        jv = float(0.0)
        for d in range(n_dofs):
            jv += J_group[idx, c, d] * v_hat[dof_start + d]
        wp.atomic_add(world_rhs, world, c, jv)  # Add J*v (positive)


@wp.kernel(enable_backward=False, module="unique")
def rhs_accum_world_par_row(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    art_to_world: wp.array[int],
    art_dof_start: wp.array[int],
    v_hat: wp.array[float],
    group_to_art: wp.array[int],
    J_group: wp.array3d[float],
    n_dofs: int,
    world_rhs: wp.array2d[float],
):
    """Accumulate independent rows without changing their serial DOF reduction.

    Use only when each world has one solve articulation. Multiple articulation
    writers retain the original mapping and its atomic accumulation order.
    """
    tid = wp.tid()
    idx = tid // max_constraints
    c = tid - idx * max_constraints
    art = group_to_art[idx]
    world = art_to_world[art]
    n_constraints = world_constraint_count[world]
    if c >= n_constraints:
        return
    dof_start = art_dof_start[art]
    jv = float(0.0)
    for d in range(n_dofs):
        jv += J_group[idx, c, d] * v_hat[dof_start + d]
    # Retain CUDA atomic floating-point semantics, including denormal handling.
    wp.atomic_add(world_rhs, world, c, jv)


@wp.kernel
def prepare_world_impulses(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    warmstart: int,
    # in/out
    world_impulses: wp.array2d[float],
):
    """Cold-initialize dense rows before identity-gathering contact impulses.

    No current non-contact row family has a cross-step identity contract:
    limits are activation-allocated, while mimic/connect rows can be enabled
    dynamically and allocated atomically. Warm start therefore clears the
    full capacity before the contact/friction identity gather; cold solves only
    need to clear the current active prefix.
    """
    world = wp.tid()
    m = world_constraint_count[world]

    # Cold-started solves consume only the current active prefix. Rows outside
    # that prefix are ignored, and a row that becomes active on a later step is
    # cleared then as part of that step's prefix. Warm-started solves clear full
    # capacity so no inactive cache can survive a later layout growth.
    clear_count = max_constraints
    if warmstart == 0:
        clear_count = m

    for i in range(clear_count):
        world_impulses[world, i] = 0.0


@wp.kernel
def reset_world_warmstart_buffers(
    world_mask: wp.array[wp.bool],
    dense_impulses: wp.array2d[float],
    prev_mf_impulses: wp.array2d[float],
    prev_mf_row_type: wp.array2d[int],
    prev_mf_row_parent: wp.array2d[int],
):
    """Clear persistent warm-start state for selected worlds."""
    world = wp.tid()
    if world_mask and not world_mask[world]:
        return

    if world < dense_impulses.shape[0]:
        for row in range(dense_impulses.shape[1]):
            dense_impulses[world, row] = 0.0
    if world < prev_mf_impulses.shape[0]:
        for row in range(prev_mf_impulses.shape[1]):
            prev_mf_impulses[world, row] = 0.0
    if world < prev_mf_row_type.shape[0]:
        for row in range(prev_mf_row_type.shape[1]):
            prev_mf_row_type[world, row] = -1
    if world < prev_mf_row_parent.shape[0]:
        for row in range(prev_mf_row_parent.shape[1]):
            prev_mf_row_parent[world, row] = -1


@wp.kernel
def invalidate_articulation_fk_id_cache(
    world_mask: wp.array[wp.bool],
    art_to_world: wp.array[int],
    fk_id_cache_valid: wp.array[int],
):
    """Invalidate cached articulation dynamics for selected worlds."""
    articulation = wp.tid()
    world = art_to_world[articulation]
    if world_mask and not world_mask[world]:
        return
    fk_id_cache_valid[articulation] = 0


# =============================================================================
# Fully Matrix-Free PGS Kernels (velocity-space Jacobi)
# =============================================================================


@wp.kernel
def diag_from_JY_par_art(
    J_group: wp.array3d[float],  # [n_arts_of_size, max_constraints, n_dofs]
    Y_group: wp.array3d[float],  # [n_arts_of_size, max_constraints, n_dofs]
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    world_constraint_count: wp.array[int],
    n_dofs: int,
    max_constraints: int,
    n_arts: int,
    mf_constraint_count: wp.array[int],
    skip_rows_le: int,
    # output
    world_diag: wp.array2d[float],
):
    """Compute diagonal of Delassus from J and Y without assembling the full matrix.

    diag[w,c] += sum_k J[idx,c,k] * Y[idx,c,k]. Thread dim: n_arts * max_constraints.
    Worlds with at most ``skip_rows_le`` dense rows and no matrix-free rows are skipped when
    ``skip_rows_le > 0`` (their sweep kernel forms the diagonal itself).
    """
    tid = wp.tid()
    c = tid % max_constraints
    idx = tid // max_constraints
    if idx >= n_arts:
        return
    art = group_to_art[idx]
    world = art_to_world[art]
    m = world_constraint_count[world]
    if c >= m:
        return
    if skip_rows_le > 0 and m <= skip_rows_le and mf_constraint_count[world] == 0:
        return
    val = float(0.0)
    for k in range(n_dofs):
        val += J_group[idx, c, k] * Y_group[idx, c, k]
    if val != 0.0:
        wp.atomic_add(world_diag, world, c, val)


@wp.kernel
def accumulate_group_diag_worlds(
    group_diag: wp.array2d[float],
    world_group_art_start: wp.array[int],
    world_group_to_art: wp.array[int],
    art_group_idx: wp.array[int],
    world_constraint_count: wp.array[int],
    max_constraints: int,
    mf_constraint_count: wp.array[int],
    skip_rows_le: int,
    # output
    world_diag: wp.array2d[float],
):
    """Accumulate one size group's response diagonal in deterministic world order."""
    tid = wp.tid()
    c = tid % max_constraints
    world = tid // max_constraints
    m = world_constraint_count[world]
    if c >= m:
        return
    if skip_rows_le > 0 and m <= skip_rows_le and mf_constraint_count[world] == 0:
        return
    value = float(0.0)
    for offset in range(world_group_art_start[world], world_group_art_start[world + 1]):
        art = world_group_to_art[offset]
        value += group_diag[art_group_idx[art], c]
    if value != 0.0:
        world_diag[world, c] += value


@wp.kernel
def gather_JY_to_world(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    world_constraint_count: wp.array[int],
    J_group: wp.array3d[float],
    Y_group: wp.array3d[float],
    n_dofs: int,
    max_constraints: int,
    n_arts: int,
    # outputs
    J_world: wp.array3d[float],
    Y_world: wp.array3d[float],
):
    """Gather per-size-group J/Y into world-indexed arrays.

    Thread dim: n_arts * max_constraints * n_dofs.
    """
    tid = wp.tid()
    d = tid % n_dofs
    remainder = tid // n_dofs
    c = remainder % max_constraints
    idx = remainder // max_constraints
    if idx >= n_arts:
        return
    art = group_to_art[idx]
    world = art_to_world[art]
    if c >= world_constraint_count[world]:
        return
    local_d = articulation_world_dof_offset[art] + d
    # Write unconditionally (including zeros) so J_world/Y_world don't need pre-zeroing
    J_world[world, c, local_d] = J_group[idx, c, d]
    Y_world[world, c, local_d] = Y_group[idx, c, d]


@wp.kernel
def diag_from_JY_world(
    world_constraint_count: wp.array[int],
    local_solve_owner: wp.array[int],
    world_dof_count: wp.array[int],
    J_world: wp.array3d[float],
    Y_world: wp.array3d[float],
    max_constraints: int,
    # output
    world_diag: wp.array2d[float],
):
    """Compute the matrix-free Delassus diagonal from world-indexed J/Y buffers."""
    tid = wp.tid()
    row = tid % max_constraints
    world = tid // max_constraints
    if row >= world_constraint_count[world]:
        return
    if local_solve_owner[world] != PGS_LOCAL_SOLVE_OWNER_GENERAL:
        return

    value = float(0.0)
    for dof in range(world_dof_count[world]):
        value += J_world[world, row, dof] * Y_world[world, row, dof]
    world_diag[world, row] = value


# =============================================================================
# Matrix-Free PGS Kernels for Free Rigid Bodies
# =============================================================================


@wp.func
def _build_mf_contact_row(
    c: int,
    total_contacts: int,
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_path: wp.array[int],
    contact_slots_needed: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_v_s: wp.array[wp.spatial_vector],
    prescribed_articulation: wp.array[int],
    has_target_velocity: int,
    shape_material_mu: wp.array[float],
    shape_material_restitution: wp.array[float],
    contact_friction_shared_anchor: int,
    contact_friction_scale: float,
    contact_shared_anchor: int,
    friction_patches: FrictionPatches,
    friction_anchor_beta: float,
    # outputs
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_row_type: wp.array2d[int],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    mf_phi: wp.array2d[float],
    mf_target_velocity: wp.array2d[float],
    mf_row_restitution: wp.array2d[float],
):
    """Build MF constraint rows for contacts between free rigid bodies / ground.

    For root free joints, the internal qd used here stores the COM-point linear
    term and angular velocity, i.e. `qd = [v_com_point, omega]`, where the COM
    point is the root body's world-space COM position. The MF contact Jacobian
    uses contact position relative to that point:
        J = [d, r x d]   (r = p_contact - p_com_world)
    """
    slot = contact_slot[c]
    if contact_path[c] != 1 or slot < 0:
        return
    world = contact_world[c]
    art_a = contact_art_a[c]
    art_b = contact_art_b[c]
    body_a, body_b, normal, point_a, point_b, phi = contact_geometry(
        c,
        contact_shape0,
        contact_shape1,
        contact_point0,
        contact_point1,
        contact_normal,
        contact_thickness0,
        contact_thickness1,
        shape_body,
        body_q,
    )
    # Zero-DOF articulations contribute collision geometry but own no
    # generalized response: their rows treat them like ground.
    response_body_a = -1
    response_body_b = -1
    if body_a >= 0 and art_a >= 0 and articulation_response_dof_count[art_a] > 0:
        response_body_a = body_a
    if body_b >= 0 and art_b >= 0 and articulation_response_dof_count[art_b] > 0:
        response_body_b = body_b
    mu, restitution = contact_materials(
        contact_shape0[c], contact_shape1[c], shape_material_mu, shape_material_restitution
    )
    tangent0, tangent1 = contact_tangent_basis(normal)
    friction_a, friction_b = contact_friction_points(
        point_a, point_b, contact_shared_anchor, contact_friction_shared_anchor
    )
    if friction_patches.enabled != 0:
        friction_a, friction_b = patch_friction_location(friction_patches, c)
    # Allocation owns the row extent; recomputing eligibility can cross a
    # floating-point threshold and overwrite the following contact.
    rows = 1
    if contact_slots_needed[c] >= 3:
        rows = 3
    for row in range(rows):
        row_idx = slot + row
        direction, p0, p1 = contact_row_frame(
            row, contact_shared_anchor, normal, tangent0, tangent1, point_a, point_b, friction_a, friction_b
        )
        # A free body's row about its articulation origin: J = [d, r x d], opposite on body B.
        if response_body_a >= 0:
            J_a = free_body_jacobian(direction, p0, articulation_origin[art_a])
            for k in range(6):
                mf_J_a[world, row_idx, k] = J_a[k]
        if response_body_b >= 0:
            J_b = free_body_jacobian(direction, p1, articulation_origin[art_b])
            for k in range(6):
                mf_J_b[world, row_idx, k] = -J_b[k]
        mf_body_a[world, row_idx] = response_body_a
        mf_body_b[world, row_idx] = response_body_b
        if row == 0:
            mf_row_type[world, row_idx] = PGS_CONSTRAINT_TYPE_CONTACT
            mf_row_parent[world, row_idx] = -1
            mf_row_mu[world, row_idx] = mu
            mf_phi[world, row_idx] = phi
            mf_row_restitution[world, row_idx] = restitution
        else:
            mf_row_type[world, row_idx] = PGS_CONSTRAINT_TYPE_FRICTION
            mf_row_parent[world, row_idx] = slot
            mf_row_mu[world, row_idx] = mu * contact_friction_scale
            mf_phi[world, row_idx] = friction_anchor_beta * friction_patches.phi[c][row - 1]
            mf_row_restitution[world, row_idx] = 0.0
        if has_target_velocity != 0:
            mf_target_velocity[world, row_idx] = prescribed_target(
                body_a, art_a, body_b, art_b, p0, p1, direction, prescribed_articulation, articulation_origin, body_v_s
            )


@wp.kernel
def build_mf_contact_rows(
    contact_count: wp.array[int],
    total_num_threads: int,
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_path: wp.array[int],
    contact_slots_needed: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    articulation_response_dof_count: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_v_s: wp.array[wp.spatial_vector],
    prescribed_articulation: wp.array[int],
    has_target_velocity: int,
    shape_material_mu: wp.array[float],
    shape_material_restitution: wp.array[float],
    contact_friction_shared_anchor: int,
    contact_friction_scale: float,
    contact_shared_anchor: int,
    friction_patches: FrictionPatches,
    friction_anchor_beta: float,
    # outputs
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_row_type: wp.array2d[int],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    mf_phi: wp.array2d[float],
    mf_target_velocity: wp.array2d[float],
    mf_row_restitution: wp.array2d[float],
):
    """Build active matrix-free contacts with a capacity-independent launch."""
    total_contacts = contact_count[0]
    for c in range(wp.tid(), total_contacts, total_num_threads):
        _build_mf_contact_row(
            c,
            total_contacts,
            contact_point0,
            contact_point1,
            contact_normal,
            contact_shape0,
            contact_shape1,
            contact_thickness0,
            contact_thickness1,
            contact_world,
            contact_slot,
            contact_path,
            contact_slots_needed,
            contact_art_a,
            contact_art_b,
            articulation_response_dof_count,
            articulation_origin,
            shape_body,
            body_q,
            body_v_s,
            prescribed_articulation,
            has_target_velocity,
            shape_material_mu,
            shape_material_restitution,
            contact_friction_shared_anchor,
            contact_friction_scale,
            contact_shared_anchor,
            friction_patches,
            friction_anchor_beta,
            mf_body_a,
            mf_body_b,
            mf_J_a,
            mf_J_b,
            mf_row_type,
            mf_row_parent,
            mf_row_mu,
            mf_phi,
            mf_target_velocity,
            mf_row_restitution,
        )


@wp.func
def _transport_warmstart_contact(
    normal: wp.vec3,
    previous_normal: wp.vec3,
    world: int,
    slot: int,
    count: int,
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    row_mu: wp.array2d[float],
    impulses: wp.array2d[float],
):
    """Transport the cached tangent frame and enforce the current contact cone."""
    normal_impulse = impulses[world, slot]
    if not wp.isfinite(normal_impulse) or wp.dot(normal, previous_normal) <= 0.0:
        normal_impulse = 0.0
    normal_impulse = wp.max(normal_impulse, 0.0)
    impulses[world, slot] = normal_impulse
    if slot + 2 >= count:
        return
    if (
        row_type[world, slot + 1] != PGS_CONSTRAINT_TYPE_FRICTION
        or row_type[world, slot + 2] != PGS_CONSTRAINT_TYPE_FRICTION
        or row_parent[world, slot + 1] != slot
        or row_parent[world, slot + 2] != slot
    ):
        return

    # Collision normals are A-to-B; every contact Jacobian uses B-to-A.
    old_t0, old_t1 = contact_tangent_basis(-previous_normal)
    new_t0, new_t1 = contact_tangent_basis(-normal)
    tangent_world = impulses[world, slot + 1] * old_t0 + impulses[world, slot + 2] * old_t1
    tangent = wp.vec2(wp.dot(tangent_world, new_t0), wp.dot(tangent_world, new_t1))
    magnitude = wp.length(tangent)
    radius = wp.max(row_mu[world, slot + 1] * normal_impulse, 0.0)
    if not wp.isfinite(magnitude) or radius <= 0.0:
        tangent = wp.vec2(0.0)
    else:
        tangent = disk(tangent, radius)
    impulses[world, slot + 1] = tangent[0]
    impulses[world, slot + 2] = tangent[1]


@wp.kernel
def gather_mf_warmstart(
    contact_count: wp.array[int],
    contact_path: wp.array[int],
    contact_slot: wp.array[int],
    contact_world: wp.array[int],
    match_index: wp.array[int],  # rigid_contact_match_index (sorted-current -> prev-sorted idx)
    prev_slot_sorted: wp.array[int],  # prev-sorted contact idx -> prev base MF slot (or -1)
    prev_mf_impulses: wp.array2d[float],
    prev_mf_row_type: wp.array2d[int],
    prev_mf_row_parent: wp.array2d[int],
    mf_constraint_count: wp.array[int],
    mf_row_type: wp.array2d[int],  # THIS step's row types (already built)
    mf_row_parent: wp.array2d[int],
    contact_normal: wp.array[wp.vec3],
    previous_normal: wp.array[wp.vec3],
    mf_row_mu: wp.array2d[float],
    decay: float,
    dt_scale: float,
    mf_max_c: int,
    # in-out
    mf_impulses: wp.array2d[float],
):
    """Seed this step's MF impulse buffer from the previous step's converged
    impulses, matched by contact identity.

    Runs after ``allocate_world_contact_slots`` + ``build_mf_contact_rows`` (so
    ``contact_slot`` and ``mf_row_type`` are populated for this step) and after
    the warm-start branch has zeroed ``mf_impulses``. One thread per contact;
    each writes only its own disjoint slot range, so unmatched / cold contacts
    keep the zero left by the memset and no stale slot survives.

    ``dt_scale`` = dt_now / dt_prev rescales carried impulses across step-size
    changes (impulse is proportional to dt for quasi-static loads); 1.0 at fixed dt.

    ``match_index[c] >= 0`` is the previous frame's *sorted* contact index;
    ``prev_slot_sorted`` is keyed the same way (see solver writeback), so
    ``prev_slot = prev_slot_sorted[match_index[c]]`` is the base slot that
    contact occupied last step. Friction rows are only seeded when BOTH this
    step and the previous step allocated a friction row at the corresponding
    offset (guards the variable 1-vs-3 stride).
    """
    c = wp.tid()
    if c >= contact_count[0]:
        return
    if contact_path[c] != 1:  # MF path only
        return
    new_slot = contact_slot[c]
    if new_slot < 0:
        return

    world = contact_world[c]
    count = mf_constraint_count[world]
    if new_slot >= count:
        return

    mi = match_index[c]
    matched = mi >= 0
    prev_slot = int(-1)
    if matched:
        prev_slot = prev_slot_sorted[mi]

    # Normal row (offset 0): always present for an MF contact.
    if (
        matched
        and prev_slot >= 0
        and prev_slot < mf_max_c
        and mf_row_type[world, new_slot] == PGS_CONSTRAINT_TYPE_CONTACT
        and prev_mf_row_type[world, prev_slot] == PGS_CONSTRAINT_TYPE_CONTACT
    ):
        mf_impulses[world, new_slot] = decay * dt_scale * prev_mf_impulses[world, prev_slot]
    # else: leave 0 (already memset)

    # Friction rows (offsets 1..2): only if THIS step allocated them here and
    # they belong to this contact's block (parent == base slot).
    for r in range(1, 3):
        new_r = new_slot + r
        if new_r < count and new_r < mf_max_c:
            if mf_row_type[world, new_r] == PGS_CONSTRAINT_TYPE_FRICTION and mf_row_parent[world, new_r] == new_slot:
                prev_r = prev_slot + r
                if (
                    matched
                    and prev_slot >= 0
                    and prev_r < mf_max_c
                    and prev_mf_row_type[world, prev_r] == PGS_CONSTRAINT_TYPE_FRICTION
                    and prev_mf_row_parent[world, prev_r] == prev_slot
                ):
                    mf_impulses[world, new_r] = decay * dt_scale * prev_mf_impulses[world, prev_r]
                # else: leave 0

    if mi >= 0 and prev_slot >= 0 and prev_slot < mf_max_c:
        _transport_warmstart_contact(
            contact_normal[c],
            previous_normal[mi],
            world,
            new_slot,
            count,
            mf_row_type,
            mf_row_parent,
            mf_row_mu,
            mf_impulses,
        )


@wp.kernel
def snapshot_contact_warmstart(
    contact_count: wp.array[int],
    contact_path: wp.array[int],
    contact_slot: wp.array[int],
    contact_normal: wp.array[wp.vec3],
    previous_dense_slot: wp.array[int],
    previous_mf_slot: wp.array[int],
    previous_normal: wp.array[wp.vec3],
):
    """Save all contact routes and live normals in one launch."""
    contact = wp.tid()
    active = contact < contact_count[0]
    path = int(-1)
    slot = int(-1)
    if active:
        path = contact_path[contact]
        slot = contact_slot[contact]
        previous_normal[contact] = contact_normal[contact]
    # One thread per ready row; an empty table is a path the solver does not take.
    if previous_dense_slot.shape[0] > 0:
        previous_dense_slot[contact] = wp.where(path == 0, slot, -1)
    if previous_mf_slot.shape[0] > 0:
        previous_mf_slot[contact] = wp.where(path == 1, slot, -1)


@wp.kernel
def gather_dense_warmstart(
    contact_count: wp.array[int],
    contact_path: wp.array[int],
    contact_slot: wp.array[int],
    contact_world: wp.array[int],
    match_index: wp.array[int],  # rigid_contact_match_index (sorted-current -> prev-sorted idx)
    prev_slot_sorted: wp.array[int],  # prev-sorted contact idx -> prev dense normal row (or -1)
    prev_dense_impulses: wp.array2d[float],
    prev_dense_row_type: wp.array2d[int],
    prev_dense_row_parent: wp.array2d[int],
    world_constraint_count: wp.array[int],
    world_row_type: wp.array2d[int],  # THIS step's dense row types (already built)
    world_row_parent: wp.array2d[int],
    contact_normal: wp.array[wp.vec3],
    previous_normal: wp.array[wp.vec3],
    world_row_mu: wp.array2d[float],
    decay: float,
    dt_scale: float,
    max_constraints: int,
    # in-out
    world_impulses: wp.array2d[float],
):
    """Seed this step's DENSE impulse buffer from the previous step's converged
    impulses, matched by contact identity — the dense mirror of
    :func:`gather_mf_warmstart`.

    Runs after ``prepare_world_impulses`` has cold-zeroed every dense row, so
    unmatched or fresh contacts stay cold and no stale impulse survives a
    slot-layout shift. Friction rows
    (offsets 1, 2 after the normal row) are seeded only when BOTH this step and
    the previous step allocated a friction row at the corresponding offset.
    ``decay`` scales every seeded impulse: at persistent quasi-static contacts
    it damps the accumulate-on-impact overshoot while keeping the sweep's
    starting point near the converged solution.
    """
    c = wp.tid()
    if c >= contact_count[0]:
        return
    if contact_path[c] != 0:  # dense path only
        return
    new_slot = contact_slot[c]
    if new_slot < 0:
        return

    world = contact_world[c]
    count = world_constraint_count[world]
    if new_slot >= count:
        return

    mi = match_index[c]
    matched = mi >= 0
    prev_slot = int(-1)
    if matched:
        prev_slot = prev_slot_sorted[mi]

    # Normal row (offset 0).
    if (
        matched
        and prev_slot >= 0
        and prev_slot < max_constraints
        and prev_dense_row_type[world, prev_slot] == PGS_CONSTRAINT_TYPE_CONTACT
    ):
        if world_row_type[world, new_slot] == PGS_CONSTRAINT_TYPE_CONTACT:
            world_impulses[world, new_slot] = decay * dt_scale * prev_dense_impulses[world, prev_slot]

    # Friction rows (offsets 1, 2): only if THIS step allocated them here.
    for r in range(1, 3):
        new_r = new_slot + r
        if new_r < count and new_r < max_constraints:
            if (
                world_row_type[world, new_r] == PGS_CONSTRAINT_TYPE_FRICTION
                and world_row_parent[world, new_r] == new_slot
            ):
                prev_r = prev_slot + r
                if (
                    matched
                    and prev_slot >= 0
                    and prev_r < max_constraints
                    and prev_dense_row_type[world, prev_r] == PGS_CONSTRAINT_TYPE_FRICTION
                    and prev_dense_row_parent[world, prev_r] == prev_slot
                ):
                    world_impulses[world, new_r] = decay * dt_scale * prev_dense_impulses[world, prev_r]

    if mi >= 0 and prev_slot >= 0 and prev_slot < max_constraints:
        _transport_warmstart_contact(
            contact_normal[c],
            previous_normal[mi],
            world,
            new_slot,
            count,
            world_row_type,
            world_row_parent,
            world_row_mu,
            world_impulses,
        )


RIGID_SPEED_ROWS = wp.constant(2)
"""Speed rows of a limited free rigid body: the first bounds its linear speed, the second its angular speed."""


@wp.kernel
def allocate_rigid_velocity_limit_slots(
    free_rigid_body_indices: wp.array[int],
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
    is_free_rigid: wp.array[int],
    body_flags: wp.array[wp.int32],
    rigid_body_max_linear_velocity: wp.array[float],
    rigid_body_max_angular_velocity: wp.array[float],
    articulation_root_dof_start: wp.array[int],
    joint_qd: wp.array[float],
    velocity_limit_activation_fraction: float,
    mf_max_constraints: int,
    resolved_worlds: wp.array[int],
    # outputs
    rigid_velocity_limit_slot: wp.array[int],
    mf_slot_counter: wp.array[int],
    mf_first_rejected_slot: wp.array[int],
):
    """Allocate one MF speed row per finitely limited speed of a free rigid body.

    Free-rigid generalized velocity is stored as six root DOFs, the center of
    mass velocity and the angular velocity: ``[v, omega]``. PhysX's
    ``maxLinearVelocity`` and ``maxAngularVelocity`` bound their magnitudes,
    ``|v| <= limit`` and ``|omega| <= limit``, so each finite limit owns one
    speed row (:func:`populate_rigid_velocity_limit_rows`) that the sweeps
    project onto that ball, along the current velocity.

    ``velocity_limit_activation_fraction`` proximity-gates the allocation,
    mirroring :func:`allocate_joint_velocity_limit_slots`: a positive fraction
    reserves a row only when the speed read from ``joint_qd`` at the free
    root's DOFs satisfies ``|v| >= fraction * limit``. A fraction of ``0.0``
    reserves every finitely limited speed's row without reading ``joint_qd``.
    """
    candidate = wp.tid()
    body = free_rigid_body_indices[candidate]
    base = RIGID_SPEED_ROWS * candidate

    for row in range(RIGID_SPEED_ROWS):
        rigid_velocity_limit_slot[base + row] = -1

    art = body_to_articulation[body]
    if art < 0:
        return
    if is_free_rigid[art] == 0:
        return
    if (body_flags[body] & BodyFlags.KINEMATIC) != 0:
        return

    world = art_to_world[art]
    if world < 0:
        return
    if world < resolved_worlds.shape[0] and resolved_worlds[world] != 0:
        return

    for speed_axis in range(RIGID_SPEED_ROWS):
        limit = rigid_body_max_linear_velocity[body]
        if speed_axis == 1:
            limit = rigid_body_max_angular_velocity[body]
        # An infinite speed limit is no limit; otherwise the velocity-limit activation law, whose proximity
        # gate reserves the row only when the speed reaches ``fraction * limit``.
        if not wp.isfinite(limit):
            continue
        root_dof = articulation_root_dof_start[art] + 3 * speed_axis
        speed = wp.length(wp.vec3(joint_qd[root_dof], joint_qd[root_dof + 1], joint_qd[root_dof + 2]))
        if not velocity_limit_active(limit, speed, velocity_limit_activation_fraction):
            continue

        rigid_velocity_limit_slot[base + speed_axis] = reserve(
            mf_slot_counter, world, 1, mf_max_constraints, mf_first_rejected_slot
        )


@wp.kernel
def populate_rigid_velocity_limit_rows(
    free_rigid_body_indices: wp.array[int],
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
    is_free_rigid: wp.array[int],
    rigid_body_max_linear_velocity: wp.array[float],
    rigid_body_max_angular_velocity: wp.array[float],
    rigid_velocity_limit_slot: wp.array[int],
    # outputs
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_row_type: wp.array2d[int],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    mf_phi: wp.array2d[float],
):
    """Populate the MF speed rows of free rigid bodies' linear and angular velocity limits.

    A speed row's Jacobian is the unit vector of the first DOF of the
    velocity it bounds (``e_0`` for the linear velocity, ``e_3`` for the
    angular one); the sweeps read the row's three DOFs from there, and its
    effective mass converts the speed correction into the impulse the row
    stores. ``mf_phi`` holds the limit.
    """
    candidate = wp.tid()
    body = free_rigid_body_indices[candidate]
    art = body_to_articulation[body]
    if art < 0:
        return
    if is_free_rigid[art] == 0:
        return

    world = art_to_world[art]
    if world < 0:
        return

    base = RIGID_SPEED_ROWS * candidate
    for speed_axis in range(RIGID_SPEED_ROWS):
        slot = rigid_velocity_limit_slot[base + speed_axis]
        if slot < 0:
            continue
        limit = rigid_body_max_linear_velocity[body]
        if speed_axis == 1:
            limit = rigid_body_max_angular_velocity[body]

        for k in range(6):
            mf_J_a[world, slot, k] = 0.0
            mf_J_b[world, slot, k] = 0.0
        mf_J_a[world, slot, 3 * speed_axis] = 1.0

        mf_body_a[world, slot] = body
        mf_body_b[world, slot] = -1
        mf_row_type[world, slot] = PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT
        mf_row_parent[world, slot] = -1
        mf_row_mu[world, slot] = 0.0
        # For velocity-limit rows mf_phi stores the speed limit; contact rows
        # use it as geometric gap. Row type disambiguates the meaning.
        mf_phi[world, slot] = limit


@wp.func
def project_free_body_speed(
    velocity: wp.array[float],
    dof: int,
    speed_limit: float,
    effective_mass: float,
    old_impulse: float,
    accumulated: int,
) -> wp.vec2:
    """Project the free-body velocity ``velocity[dof:dof + 3]`` of a speed row onto ``|v| <= speed_limit``.

    The row's law is :func:`~.projection_laws.speed_row` (the stateless clamp
    or the accumulated total of :func:`~.projection_laws.velocity_limit_update`),
    and its correction acts along the current velocity, so a body over its limit
    keeps its direction and ends on the limit. Returns the row's new impulse
    and the impulse change applied, both in units of ``effective_mass`` times
    speed.
    """
    v = wp.vec3(velocity[dof], velocity[dof + 1], velocity[dof + 2])
    row = speed_row(wp.length(v), speed_limit, effective_mass, old_impulse, accumulated)
    if row[2] != 0.0:
        change = v * row[2]
        velocity[dof] = velocity[dof] + change[0]
        velocity[dof + 1] = velocity[dof + 1] + change[1]
        velocity[dof + 2] = velocity[dof + 2] + change[2]
    return wp.vec2(row[0], row[1])


@wp.func
def free_body_speed_dof(mf_J_a: wp.array3d[float], world: int, row: int) -> int:
    """Offset of the velocity a free-body speed row bounds: 0 for the linear velocity, 3 for the angular."""
    if mf_J_a[world, row, 3] != 0.0:
        return 3
    return 0


@wp.func
def spatial_matrix_block_inverse(M: wp.spatial_matrix):
    """Invert a 6x6 spatial matrix using 3x3 block inversion.

    Partition M = [A B; C D] into 3x3 blocks, then:
        S = D - C * A^-1 * B   (Schur complement)
        M^-1 = [A^-1 + A^-1*B*S^-1*C*A^-1,  -A^-1*B*S^-1]
               [-S^-1*C*A^-1,                 S^-1]
    """
    A = wp.mat33(
        M[0, 0],
        M[0, 1],
        M[0, 2],
        M[1, 0],
        M[1, 1],
        M[1, 2],
        M[2, 0],
        M[2, 1],
        M[2, 2],
    )
    B = wp.mat33(
        M[0, 3],
        M[0, 4],
        M[0, 5],
        M[1, 3],
        M[1, 4],
        M[1, 5],
        M[2, 3],
        M[2, 4],
        M[2, 5],
    )
    C = wp.mat33(
        M[3, 0],
        M[3, 1],
        M[3, 2],
        M[4, 0],
        M[4, 1],
        M[4, 2],
        M[5, 0],
        M[5, 1],
        M[5, 2],
    )
    D = wp.mat33(
        M[3, 3],
        M[3, 4],
        M[3, 5],
        M[4, 3],
        M[4, 4],
        M[4, 5],
        M[5, 3],
        M[5, 4],
        M[5, 5],
    )

    Ainv = wp.inverse(A)
    AinvB = Ainv * B
    S = D - C * AinvB
    Sinv = wp.inverse(S)
    SinvCAinv = Sinv * C * Ainv

    # Top-left: Ainv + AinvB * SinvCAinv
    TL = Ainv + AinvB * SinvCAinv
    # Top-right: -AinvB * Sinv
    TR = -AinvB * Sinv
    # Bottom-left: -SinvCAinv
    BL = -SinvCAinv
    # Bottom-right: Sinv
    BR = Sinv

    return wp.spatial_matrix(
        TL[0, 0],
        TL[0, 1],
        TL[0, 2],
        TR[0, 0],
        TR[0, 1],
        TR[0, 2],
        TL[1, 0],
        TL[1, 1],
        TL[1, 2],
        TR[1, 0],
        TR[1, 1],
        TR[1, 2],
        TL[2, 0],
        TL[2, 1],
        TL[2, 2],
        TR[2, 0],
        TR[2, 1],
        TR[2, 2],
        BL[0, 0],
        BL[0, 1],
        BL[0, 2],
        BR[0, 0],
        BR[0, 1],
        BR[0, 2],
        BL[1, 0],
        BL[1, 1],
        BL[1, 2],
        BR[1, 0],
        BR[1, 1],
        BR[1, 2],
        BL[2, 0],
        BL[2, 1],
        BL[2, 2],
        BR[2, 0],
        BR[2, 1],
        BR[2, 2],
    )


@wp.kernel
def compute_mf_body_Hinv(
    free_rigid_body_indices: wp.array[int],
    body_I_s: wp.array[wp.spatial_matrix],
    is_free_rigid: wp.array[int],
    body_to_articulation: wp.array[int],
    body_flags: wp.array[wp.int32],
    # outputs
    mf_body_Hinv: wp.array[wp.spatial_matrix],
):
    """Compute H^-1 = inverse(body_I_s) for free rigid bodies.

    For root free joints, H = body_I_s in articulation-local coordinates.
    This remains a full 6x6 matrix for bodies with non-zero CoM offsets.
    """
    b = free_rigid_body_indices[wp.tid()]
    art = body_to_articulation[b]
    if art < 0:
        return
    if is_free_rigid[art] == 0:
        return
    if (body_flags[b] & BodyFlags.KINEMATIC) != 0:
        mf_body_Hinv[b] = wp.spatial_matrix(0.0)
        return

    mf_body_Hinv[b] = spatial_matrix_block_inverse(body_I_s[b])


@wp.kernel
def compute_mf_effective_mass_and_rhs(
    mf_constraint_count: wp.array[int],
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_body_Hinv: wp.array[wp.spatial_matrix],
    mf_phi: wp.array2d[float],
    mf_row_type: wp.array2d[int],
    mf_target_velocity: wp.array2d[float],
    mf_row_restitution: wp.array2d[float],
    has_target_velocity: int,
    body_to_articulation: wp.array[int],
    articulation_dof_start: wp.array[int],
    incident_velocity: wp.array[float],
    rigid_body_max_depenetration_velocity: wp.array[float],
    pgs_cfm: float,
    pgs_beta: float,
    contact_w: float,
    dt: float,
    contact_speculative_scale: float,
    restitution_velocity_threshold: float,
    mf_max_constraints: int,
    # outputs
    mf_eff_mass_inv: wp.array2d[float],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    mf_rhs: wp.array2d[float],
    mf_row_w: wp.array2d[float],
):
    """Compute effective mass diagonal, H^-1*J^T, and RHS bias for MF constraints.

    The effective mass for constraint i is:
        d_ii = J_a^T * H_a_inv * J_a + J_b^T * H_b_inv * J_b + cfm

    H_inv is the full 6x6 inverse of the spatial inertia in articulation-local
    coordinates for each free rigid articulation.

    RHS stores only the stabilization bias (not J*v), since the MF PGS
    recomputes J*v each iteration from the live velocity array.
    """
    tid = wp.tid()
    world = tid // mf_max_constraints
    i = tid % mf_max_constraints
    if i >= mf_constraint_count[world]:
        return

    ba = mf_body_a[world, i]
    bb = mf_body_b[world, i]

    # Load Jacobian as spatial_vector
    Ja = wp.spatial_vector(
        mf_J_a[world, i, 0],
        mf_J_a[world, i, 1],
        mf_J_a[world, i, 2],
        mf_J_a[world, i, 3],
        mf_J_a[world, i, 4],
        mf_J_a[world, i, 5],
    )
    Jb = wp.spatial_vector(
        mf_J_b[world, i, 0],
        mf_J_b[world, i, 1],
        mf_J_b[world, i, 2],
        mf_J_b[world, i, 3],
        mf_J_b[world, i, 4],
        mf_J_b[world, i, 5],
    )

    d = pgs_cfm

    # Side A: MiJt_a = H_a_inv * J_a, d += J_a^T * MiJt_a
    if ba >= 0:
        Hinv_a = mf_body_Hinv[ba]
        MiJt_a = Hinv_a * Ja
        d += wp.dot(Ja, MiJt_a)
        mf_MiJt_a[world, i, 0] = MiJt_a[0]
        mf_MiJt_a[world, i, 1] = MiJt_a[1]
        mf_MiJt_a[world, i, 2] = MiJt_a[2]
        mf_MiJt_a[world, i, 3] = MiJt_a[3]
        mf_MiJt_a[world, i, 4] = MiJt_a[4]
        mf_MiJt_a[world, i, 5] = MiJt_a[5]

    # Side B
    if bb >= 0:
        Hinv_b = mf_body_Hinv[bb]
        MiJt_b = Hinv_b * Jb
        d += wp.dot(Jb, MiJt_b)
        mf_MiJt_b[world, i, 0] = MiJt_b[0]
        mf_MiJt_b[world, i, 1] = MiJt_b[1]
        mf_MiJt_b[world, i, 2] = MiJt_b[2]
        mf_MiJt_b[world, i, 3] = MiJt_b[3]
        mf_MiJt_b[world, i, 4] = MiJt_b[4]
        mf_MiJt_b[world, i, 5] = MiJt_b[5]

    if d > 0.0:
        mf_eff_mass_inv[world, i] = 1.0 / d
    else:
        mf_eff_mass_inv[world, i] = 0.0

    # Contact bias only (not J*v -- recomputed each PGS iter): the contact law, its Baumgarte term capped by the
    # bodies' depenetration speed, and a closing contact's rebound target.
    target = float(0.0)
    if has_target_velocity != 0:
        target = mf_target_velocity[world, i]
    inv_dt = 1.0 / dt
    rhs = -target
    row_w = float(1.0)
    rtype = mf_row_type[world, i]
    if rtype == PGS_CONSTRAINT_TYPE_CONTACT:
        phi_val = mf_phi[world, i]
        cap = contact_depenetration_cap(ba, bb, rigid_body_max_depenetration_velocity)
        rhs, row_w = contact_rhs(phi_val, pgs_beta, target, inv_dt, 1.0, contact_speculative_scale, contact_w, cap, 0.0)
        restitution = mf_row_restitution[world, i]
        if restitution > 0.0:
            relative_incident = float(0.0)
            if ba >= 0:
                dof_start = articulation_dof_start[body_to_articulation[ba]]
                for k in range(6):
                    relative_incident += mf_J_a[world, i, k] * incident_velocity[dof_start + k]
            if bb >= 0:
                dof_start = articulation_dof_start[body_to_articulation[bb]]
                for k in range(6):
                    relative_incident += mf_J_b[world, i, k] * incident_velocity[dof_start + k]
            relative_incident -= target
            fires, bounce = contact_restitution_rhs(
                phi_val, relative_incident, target, restitution, dt, restitution_velocity_threshold
            )
            if fires:
                rhs = bounce
                # An impact is impulsive, not a spring: keep the rebound exact.
                row_w = 1.0
    elif rtype == PGS_CONSTRAINT_TYPE_FRICTION:
        # The row builder pre-multiplies the anchor gain into phi.
        rhs, _w = row_rhs(rtype, mf_phi[world, i], 1.0, target, inv_dt, 1.0, 0.0, 0.0, 1.0)
    elif rtype == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
        # A free body's speed row carries its limit in phi.
        rhs = mf_phi[world, i] - target
    mf_rhs[world, i] = rhs
    if contact_w < 1.0:
        mf_row_w[world, i] = row_w


@wp.kernel
def compute_mf_rhs_bias(
    mf_constraint_count: wp.array[int],
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_dof_a: wp.array2d[int],
    mf_dof_b: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    world_dof_indices: wp.array2d[int],
    mf_phi: wp.array2d[float],
    mf_row_type: wp.array2d[int],
    mf_target_velocity: wp.array2d[float],
    mf_row_restitution: wp.array2d[float],
    has_target_velocity: int,
    rigid_body_max_depenetration_velocity: wp.array[float],
    pgs_beta: float,
    dt: float,
    bias_scale: float,
    speculative_scale: float,
    position_velocity: wp.array[float],
    incident_velocity: wp.array[float],
    preserve_unreached_speculative: int,
    apply_restitution: int,
    restitution_velocity_threshold: float,
    mf_max_constraints: int,
    # outputs
    mf_rhs: wp.array2d[float],
):
    """Compute only the MF contact RHS bias for a previously-built contact set."""
    tid = wp.tid()
    world = tid // mf_max_constraints
    i = tid % mf_max_constraints
    if i >= mf_constraint_count[world]:
        return

    target = float(0.0)
    if has_target_velocity != 0:
        target = mf_target_velocity[world, i]
    inv_dt = 1.0 / dt
    rhs = -target
    row_type = mf_row_type[world, i]
    if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
        phi_val = mf_phi[world, i]
        fires = bool(False)
        bounce = float(0.0)
        restitution = float(0.0)
        if apply_restitution != 0:
            restitution = mf_row_restitution[world, i]
        if restitution > 0.0:
            relative_incident = (
                mf_contact_row_dot(
                    mf_J_a,
                    mf_J_b,
                    mf_dof_a[world, i],
                    mf_dof_b[world, i],
                    world_dof_indices,
                    incident_velocity,
                    world,
                    i,
                )
                - target
            )
            fires, bounce = contact_restitution_rhs(
                phi_val, relative_incident, target, restitution, dt, restitution_velocity_threshold
            )
        if fires:
            rhs = bounce
        else:
            # The velocity pass keeps a separated row's bias while the position solve's linearized end gap,
            # evaluated with its realized velocity, stays open.
            end_gap = float(0.0)
            if preserve_unreached_speculative != 0 and phi_val > 0.0:
                jv_position = mf_contact_row_dot(
                    mf_J_a,
                    mf_J_b,
                    mf_dof_a[world, i],
                    mf_dof_b[world, i],
                    world_dof_indices,
                    position_velocity,
                    world,
                    i,
                )
                end_gap = phi_val + dt * (jv_position - target)
            cap = contact_depenetration_cap(
                mf_body_a[world, i], mf_body_b[world, i], rigid_body_max_depenetration_velocity
            )
            rhs, _row_w = contact_rhs(
                phi_val, pgs_beta, target, inv_dt, bias_scale, speculative_scale, 1.0, cap, end_gap
            )
    elif row_type == PGS_CONSTRAINT_TYPE_FRICTION:
        # The row builder pre-multiplies the anchor gain into phi.
        rhs, _w = row_rhs(row_type, mf_phi[world, i], 1.0, target, inv_dt, bias_scale, 0.0, 0.0, 1.0)
    elif row_type == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
        rhs = mf_phi[world, i] - target
    mf_rhs[world, i] = rhs


@wp.func
def friction_step_current(
    world: int,
    i: int,
    omega: float,
    mf_eff_mass_inv: wp.array2d[float],
    mf_rhs: wp.array2d[float],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    body_to_articulation: wp.array[int],
    art_dof_start: wp.array[int],
    mf_impulses: wp.array2d[float],
    v_out: wp.array[float],
):
    """Solve the tangent pair at the current patch or point normal load.

    The caller applies the first tangent's velocity change; this function
    applies the second and makes its subsequent row visit a no-op.
    """
    parent_idx = mf_row_parent[world, i]
    if i != parent_idx + 1:
        return mf_impulses[world, i]
    sib = parent_idx + 2
    lambda_n = patch_normal_load(mf_row_parent, mf_impulses, world, parent_idx)
    radius = wp.max(mf_row_mu[world, i] * lambda_n, 0.0)
    residual = wp.vec2(mf_rhs[world, i], mf_rhs[world, sib])
    ba = mf_body_a[world, i]
    bb = mf_body_b[world, i]
    cross = float(0.0)
    for k in range(6):
        if ba >= 0:
            ds = art_dof_start[body_to_articulation[ba]]
            residual[0] += mf_J_a[world, i, k] * v_out[ds + k]
            residual[1] += mf_J_a[world, sib, k] * v_out[ds + k]
            cross += mf_J_a[world, i, k] * mf_MiJt_a[world, sib, k]
        if bb >= 0:
            ds = art_dof_start[body_to_articulation[bb]]
            residual[0] += mf_J_b[world, i, k] * v_out[ds + k]
            residual[1] += mf_J_b[world, sib, k] * v_out[ds + k]
            cross += mf_J_b[world, i, k] * mf_MiJt_b[world, sib, k]
    inv0 = mf_eff_mass_inv[world, i]
    inv1 = mf_eff_mass_inv[world, sib]
    sibling_diag = float(0.0)
    if inv1 > 0.0:
        sibling_diag = 1.0 / inv1
    first_diag = float(0.0)
    if inv0 > 0.0:
        first_diag = 1.0 / inv0
    trial = disk(
        friction_pair(
            first_diag,
            cross,
            sibling_diag,
            residual,
            wp.vec2(mf_impulses[world, i], mf_impulses[world, sib]),
            radius,
            omega,
        ),
        radius,
    )
    sibling_delta = trial[1] - mf_impulses[world, sib]
    mf_impulses[world, sib] = trial[1]
    for k in range(6):
        if ba >= 0:
            ds = art_dof_start[body_to_articulation[ba]]
            v_out[ds + k] += mf_MiJt_a[world, sib, k] * sibling_delta
        if bb >= 0:
            ds = art_dof_start[body_to_articulation[bb]]
            v_out[ds + k] += mf_MiJt_b[world, sib, k] * sibling_delta
    return trial[0]


@wp.func
def friction_step_coulomb(
    world: int,
    i: int,
    friction_mode: int,
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    mf_rhs: wp.array2d[float],
    body_to_articulation: wp.array[int],
    art_dof_start: wp.array[int],
    mf_impulses: wp.array2d[float],
    v_out: wp.array[float],
):
    """Solve a matrix-free contact's Coulomb problem with a research friction mode's law.

    ``friction_mode`` picks :func:`~.projection_laws.coulomb_bisection` (RAISim's
    bisection on the normal impulse; ``FRICTION_MODE_BISECTION_DESAXCE`` first raises
    the normal target with :func:`~.projection_laws.de_saxce_target`) or
    :func:`~.projection_laws.coulomb_newton` (Daviet's scalar Newton).

    A matrix-free contact is its normal row and two friction rows. The solve runs at
    the first friction row (``i == parent + 1``) on the contact's rows' velocities
    and its Delassus block ``G = J H^-1 J^T``, both gathered from ``v_out`` and the
    rows' ``mf_J_*`` / ``mf_MiJt_*``. It writes the normal and second friction
    impulses with their velocity changes and returns the first friction row's new
    impulse for the caller to apply; the second friction row's visit returns its
    stored impulse, a zero change. The normal row's ``mf_rhs`` is ``beta phi / dt``,
    negative on penetration, so the laws' normal target velocity is ``-mf_rhs``.

    Returns:
        The first friction row's new impulse [N·s], or the second friction row's
        stored one.
    """
    parent_idx = mf_row_parent[world, i]
    i_n = parent_idx
    i_t1 = parent_idx + 1
    i_t2 = parent_idx + 2
    if i != i_t1:
        return mf_impulses[world, i]

    mu_val = mf_row_mu[world, i]
    ba = mf_body_a[world, i_n]
    bb = mf_body_b[world, i_n]

    # The rows' velocities and the block G, from the current v_out.
    u_n = float(0.0)
    u_t1 = float(0.0)
    u_t2 = float(0.0)
    G_nn = float(0.0)
    G_nt1 = float(0.0)
    G_nt2 = float(0.0)
    G_t1t1 = float(0.0)
    G_t1t2 = float(0.0)
    G_t2t2 = float(0.0)
    ds_a = int(0)
    ds_b = int(0)
    if ba >= 0:
        ds_a = art_dof_start[body_to_articulation[ba]]
        for k in range(6):
            va_k = v_out[ds_a + k]
            u_n = u_n + mf_J_a[world, i_n, k] * va_k
            u_t1 = u_t1 + mf_J_a[world, i_t1, k] * va_k
            u_t2 = u_t2 + mf_J_a[world, i_t2, k] * va_k
    if bb >= 0:
        ds_b = art_dof_start[body_to_articulation[bb]]
        for k in range(6):
            vb_k = v_out[ds_b + k]
            u_n = u_n + mf_J_b[world, i_n, k] * vb_k
            u_t1 = u_t1 + mf_J_b[world, i_t1, k] * vb_k
            u_t2 = u_t2 + mf_J_b[world, i_t2, k] * vb_k
    if ba >= 0:
        for k in range(6):
            Jna = mf_J_a[world, i_n, k]
            Jt1a = mf_J_a[world, i_t1, k]
            Jt2a = mf_J_a[world, i_t2, k]
            Mna = mf_MiJt_a[world, i_n, k]
            Mt1a = mf_MiJt_a[world, i_t1, k]
            Mt2a = mf_MiJt_a[world, i_t2, k]
            G_nn = G_nn + Jna * Mna
            G_nt1 = G_nt1 + Jna * Mt1a
            G_nt2 = G_nt2 + Jna * Mt2a
            G_t1t1 = G_t1t1 + Jt1a * Mt1a
            G_t1t2 = G_t1t2 + Jt1a * Mt2a
            G_t2t2 = G_t2t2 + Jt2a * Mt2a
    if bb >= 0:
        for k in range(6):
            Jnb = mf_J_b[world, i_n, k]
            Jt1b = mf_J_b[world, i_t1, k]
            Jt2b = mf_J_b[world, i_t2, k]
            Mnb = mf_MiJt_b[world, i_n, k]
            Mt1b = mf_MiJt_b[world, i_t1, k]
            Mt2b = mf_MiJt_b[world, i_t2, k]
            G_nn = G_nn + Jnb * Mnb
            G_nt1 = G_nt1 + Jnb * Mt1b
            G_nt2 = G_nt2 + Jnb * Mt2b
            G_t1t1 = G_t1t1 + Jt1b * Mt1b
            G_t1t2 = G_t1t2 + Jt1b * Mt2b
            G_t2t2 = G_t2t2 + Jt2b * Mt2b
    W = wp.mat33(G_nn, G_nt1, G_nt2, G_nt1, G_t1t1, G_t1t2, G_nt2, G_t1t2, G_t2t2)
    u = wp.vec3(u_n, u_t1, u_t2)
    old = wp.vec3(mf_impulses[world, i_n], mf_impulses[world, i_t1], mf_impulses[world, i_t2])
    target_vel_n = -mf_rhs[world, i_n]
    new = old
    if friction_mode == FRICTION_MODE_COULOMB_NEWTON:
        new = coulomb_newton(W, u, old, mu_val, target_vel_n)
    else:
        if friction_mode == FRICTION_MODE_BISECTION_DESAXCE:
            target_vel_n = de_saxce_target(target_vel_n, u, mu_val)
        new = coulomb_bisection(W, u, old, mu_val, target_vel_n)

    # The normal and second friction rows' velocity changes; the caller applies the first friction row's.
    d_n_total = new[0] - old[0]
    d_t2_total = new[2] - old[2]
    if ba >= 0:
        for k in range(6):
            v_out[ds_a + k] = (
                v_out[ds_a + k] + mf_MiJt_a[world, i_n, k] * d_n_total + mf_MiJt_a[world, i_t2, k] * d_t2_total
            )
    if bb >= 0:
        for k in range(6):
            v_out[ds_b + k] = (
                v_out[ds_b + k] + mf_MiJt_b[world, i_n, k] * d_n_total + mf_MiJt_b[world, i_t2, k] * d_t2_total
            )
    mf_impulses[world, i_n] = new[0]
    mf_impulses[world, i_t2] = new[2]
    return new[1]


@wp.kernel
def pgs_solve_mf_loop(
    mf_constraint_count: wp.array[int],
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_eff_mass_inv: wp.array2d[float],
    mf_rhs: wp.array2d[float],
    mf_row_type: wp.array2d[int],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    mf_row_w: wp.array2d[float],
    body_to_articulation: wp.array[int],
    art_dof_start: wp.array[int],
    iterations: int,
    omega: float,
    regularize: int,
    friction_mode: int,
    friction_start_iteration: int,
    iteration_offset: int,
    # in/out
    mf_impulses: wp.array2d[float],
    v_out: wp.array[float],
):
    """Matrix-free PGS solver for free rigid body contacts.

    Operates directly on body velocities stored in v_out (generalized coordinates).
    Each iteration recomputes J*v from v_out and applies velocity corrections
    immediately (Gauss-Seidel style).

    The per-row Coulomb friction projection is delegated to
    :func:`friction_step_current` (``friction_mode == FRICTION_MODE_CURRENT``)
    or :func:`friction_step_coulomb` (the research modes: RAISim's
    bisection, with or without the de Saxce bias, and Gilles Daviet's
    Newton), so alternate strategies can be plugged in without rewriting
    this kernel body.
    """
    world = wp.tid()
    m_count = mf_constraint_count[world]
    if m_count == 0:
        return

    for it in range(iterations):
        for i in range(m_count):
            row_type = mf_row_type[world, i]
            if friction_gate(row_type, iteration_offset + it, friction_start_iteration):
                mf_impulses[world, i] = 0.0
                continue

            eff_inv = mf_eff_mass_inv[world, i]
            if eff_inv <= 0.0 and not (
                row_type == PGS_CONSTRAINT_TYPE_FRICTION and friction_mode == FRICTION_MODE_CURRENT
            ):
                continue

            ba = mf_body_a[world, i]
            bb = mf_body_b[world, i]

            if row_type == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
                # A free body's speed row, behind every other row: the stateless clamp of its speed.
                if ba >= 0:
                    dof = art_dof_start[body_to_articulation[ba]] + free_body_speed_dof(mf_J_a, world, i)
                    projected = project_free_body_speed(v_out, dof, mf_rhs[world, i], eff_inv, 0.0, 0)
                    mf_impulses[world, i] = projected[0]
                continue

            # Compute current J * v
            jv = float(0.0)
            if ba >= 0:
                art_a = body_to_articulation[ba]
                ds_a = art_dof_start[art_a]
                for k in range(6):
                    jv += mf_J_a[world, i, k] * v_out[ds_a + k]
            if bb >= 0:
                art_b = body_to_articulation[bb]
                ds_b = art_dof_start[art_b]
                for k in range(6):
                    jv += mf_J_b[world, i, k] * v_out[ds_b + k]

            # The relaxed step of the residual J v + b, regularized on contact rows.
            residual = jv + mf_rhs[world, i]
            old_impulse = mf_impulses[world, i]
            delta = -residual * eff_inv
            if row_type == PGS_CONSTRAINT_TYPE_CONTACT and regularize != 0:
                delta = regularized_delta(delta, mf_row_w[world, i], old_impulse)
            new_impulse = relaxed_step(old_impulse, delta, omega)

            # Project
            if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
                new_impulse = unilateral(new_impulse)
            elif row_type == PGS_CONSTRAINT_TYPE_FRICTION:
                if friction_mode != FRICTION_MODE_CURRENT:
                    new_impulse = friction_step_coulomb(
                        world,
                        i,
                        friction_mode,
                        mf_body_a,
                        mf_body_b,
                        mf_J_a,
                        mf_J_b,
                        mf_MiJt_a,
                        mf_MiJt_b,
                        mf_row_parent,
                        mf_row_mu,
                        mf_rhs,
                        body_to_articulation,
                        art_dof_start,
                        mf_impulses,
                        v_out,
                    )
                else:
                    new_impulse = friction_step_current(
                        world,
                        i,
                        omega,
                        mf_eff_mass_inv,
                        mf_rhs,
                        mf_J_a,
                        mf_J_b,
                        mf_body_a,
                        mf_body_b,
                        mf_MiJt_a,
                        mf_MiJt_b,
                        mf_row_parent,
                        mf_row_mu,
                        body_to_articulation,
                        art_dof_start,
                        mf_impulses,
                        v_out,
                    )

            delta_impulse = new_impulse - old_impulse
            mf_impulses[world, i] = new_impulse

            # Apply velocity correction: v += M_inv * J^T * delta_impulse
            if ba >= 0:
                art_a2 = body_to_articulation[ba]
                ds_a2 = art_dof_start[art_a2]
                for k in range(6):
                    v_out[ds_a2 + k] = v_out[ds_a2 + k] + mf_MiJt_a[world, i, k] * delta_impulse
            if bb >= 0:
                art_b2 = body_to_articulation[bb]
                ds_b2 = art_dof_start[art_b2]
                for k in range(6):
                    v_out[ds_b2 + k] = v_out[ds_b2 + k] + mf_MiJt_b[world, i, k] * delta_impulse


@wp.kernel
def build_mf_body_map(
    mf_constraint_count: wp.array[int],
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    body_to_articulation: wp.array[int],
    art_dof_start: wp.array[int],
    max_mf_bodies: int,
    # outputs
    mf_body_list: wp.array2d[int],
    mf_body_dof_start: wp.array2d[int],
    mf_body_count: wp.array[int],
    mf_local_body_a: wp.array2d[int],
    mf_local_body_b: wp.array2d[int],
):
    """Build per-world local body table and local body index mapping.

    Scans all MF constraint body indices, builds a unique body list per world,
    and maps each constraint's body indices to local indices.
    """
    world = wp.tid()
    m = mf_constraint_count[world]
    if m == 0:
        mf_body_count[world] = 0
        return

    n_bodies = int(0)

    for i in range(m):
        # Process body A
        ba = mf_body_a[world, i]
        if ba >= 0:
            # Search for ba in body_list
            found_a = int(-1)
            for b in range(n_bodies):
                if mf_body_list[world, b] == ba:
                    found_a = b
                    break
            if found_a < 0 and n_bodies < max_mf_bodies:
                found_a = n_bodies
                mf_body_list[world, n_bodies] = ba
                art_a = body_to_articulation[ba]
                mf_body_dof_start[world, n_bodies] = art_dof_start[art_a]
                n_bodies += 1
            mf_local_body_a[world, i] = found_a
        else:
            mf_local_body_a[world, i] = -1

        # Process body B
        bb = mf_body_b[world, i]
        if bb >= 0:
            found_b = int(-1)
            for b in range(n_bodies):
                if mf_body_list[world, b] == bb:
                    found_b = b
                    break
            if found_b < 0 and n_bodies < max_mf_bodies:
                found_b = n_bodies
                mf_body_list[world, n_bodies] = bb
                mf_body_dof_start[world, n_bodies] = art_dof_start[body_to_articulation[bb]]
                n_bodies += 1
            mf_local_body_b[world, i] = found_b
        else:
            mf_local_body_b[world, i] = -1

    mf_body_count[world] = n_bodies


@wp.kernel
def compute_mf_world_dof_offsets(
    mf_constraint_count: wp.array[int],
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    body_to_articulation: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    mf_max_constraints: int,
    # outputs
    mf_dof_a: wp.array2d[int],
    mf_dof_b: wp.array2d[int],
):
    """Compute world-relative DOF offsets for each MF contact body.

    For each MF constraint, stores the articulation's compact response
    offset. The two-phase GS kernel uses these offsets to index its shared
    velocity vector.
    """
    tid = wp.tid()
    world = tid // mf_max_constraints
    c = tid % mf_max_constraints
    if c >= mf_constraint_count[world]:
        return
    ba = mf_body_a[world, c]
    bb = mf_body_b[world, c]
    if ba >= 0:
        mf_dof_a[world, c] = articulation_world_dof_offset[body_to_articulation[ba]]
    else:
        mf_dof_a[world, c] = -1
    if bb >= 0:
        mf_dof_b[world, c] = articulation_world_dof_offset[body_to_articulation[bb]]
    else:
        mf_dof_b[world, c] = -1


@wp.kernel
def pgs_solve_loop(
    world_constraint_count: wp.array[int],
    max_constraints: int,
    world_diag: wp.array2d[float],
    world_C: wp.array3d[float],
    world_rhs: wp.array2d[float],
    world_impulses: wp.array2d[float],
    iterations: int,
    omega: float,
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_row_w: wp.array2d[float],
    regularize: int,
    friction_start_iteration: int,
    iteration_offset: int,
):
    """World-level projected Gauss-Seidel over the assembled Delassus rows.

    ``world_C`` and ``world_diag`` hold the unregularized system; with ``regularize``
    a contact row's step takes its weight ``world_row_w`` (:func:`.projection_laws.regularized_delta`).
    """
    world = wp.tid()
    m = world_constraint_count[world]

    if m == 0:
        return

    for it in range(iterations):
        for i in range(m):
            row_type = world_row_type[world, i]
            if friction_gate(row_type, iteration_offset + it, friction_start_iteration):
                world_impulses[world, i] = 0.0
                continue
            if row_type == PGS_CONSTRAINT_TYPE_FRICTION and i != world_row_parent[world, i] + 1:
                continue

            # The residual r = rhs_i + sum_j C_ij * lambda_j.
            residual = world_rhs[world, i]
            for j in range(m):
                residual += world_C[world, i, j] * world_impulses[world, j]

            denom = world_diag[world, i]
            if denom <= 0.0 and row_type != PGS_CONSTRAINT_TYPE_FRICTION:
                continue

            # --- Normal contact, joint limit, or joint velocity limit:
            #     lambda_n >= 0. The velocity-limit row uses a signed Jacobian
            #     so the unilateral projector handles both sides of the
            #     bilateral ``[-qdot_max, +qdot_max]`` box.
            if (
                row_type == PGS_CONSTRAINT_TYPE_CONTACT
                or row_type == PGS_CONSTRAINT_TYPE_JOINT_LIMIT
                or row_type == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT
            ):
                delta = -residual / denom
                if row_type == PGS_CONSTRAINT_TYPE_CONTACT and regularize != 0:
                    delta = regularized_delta(delta, world_row_w[world, i], world_impulses[world, i])
                world_impulses[world, i] = unilateral(relaxed_step(world_impulses[world, i], delta, omega))

            # --- Friction: isotropic Coulomb ---
            elif row_type == PGS_CONSTRAINT_TYPE_FRICTION:
                parent_idx = world_row_parent[world, i]
                lambda_n = patch_normal_load(world_row_parent, world_impulses, world, parent_idx)
                mu = world_row_mu[world, i]
                radius = wp.max(mu * lambda_n, 0.0)

                if radius <= 0.0:
                    world_impulses[world, i] = 0.0
                    world_impulses[world, i + 1] = 0.0
                    continue

                if i != parent_idx + 1:
                    continue
                sib = parent_idx + 2
                sibling_residual = world_rhs[world, sib]
                for j in range(m):
                    sibling_residual += world_C[world, sib, j] * world_impulses[world, j]

                trial = disk(
                    friction_pair(
                        denom,
                        world_C[world, i, sib],
                        world_diag[world, sib],
                        wp.vec2(residual, sibling_residual),
                        wp.vec2(world_impulses[world, i], world_impulses[world, sib]),
                        radius,
                        omega,
                    ),
                    radius,
                )
                world_impulses[world, i] = trial[0]
                world_impulses[world, sib] = trial[1]

            else:
                world_impulses[world, i] = relaxed_step(world_impulses[world, i], -residual / denom, omega)


@wp.kernel
def apply_impulses_world_par_dof(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    art_dof_start: wp.array[int],
    n_dofs: int,
    n_arts: int,
    world_constraint_count: wp.array[int],
    max_constraints: int,
    Y_group: wp.array3d[float],
    world_impulses: wp.array2d[float],
    v_hat: wp.array[float],
    # outputs
    v_out: wp.array[float],
):
    """
    Accumulate velocity changes from world impulses for a single size group.
    DOF-parallelized: each thread handles one (articulation, DOF) pair.

    v_out = v_hat + Y * impulses
    """
    tid = wp.tid()

    # Decode thread index
    local_dof = tid % n_dofs
    idx = tid // n_dofs  # group index

    if idx >= n_arts:
        return

    art = group_to_art[idx]
    world = art_to_world[art]
    n_constraints = world_constraint_count[world]
    dof_start = art_dof_start[art]

    # Inner loop only over constraints
    delta_v = float(0.0)
    for c in range(n_constraints):
        delta_v += Y_group[idx, c, local_dof] * world_impulses[world, c]

    global_dof = dof_start + local_dof
    v_out[global_dof] = v_hat[global_dof] + delta_v


@wp.kernel
def apply_mf_warmstart_impulses(
    mf_constraint_count: wp.array[int],
    mf_body_count: wp.array[int],
    mf_body_dof_start: wp.array2d[int],
    mf_local_body_a: wp.array2d[int],
    mf_local_body_b: wp.array2d[int],
    mf_MiJt_a: wp.array3d[float],
    mf_MiJt_b: wp.array3d[float],
    mf_impulses: wp.array2d[float],
    max_mf_bodies: int,
    set_output_to_delta: int,
    # in/out
    velocity: wp.array[float],
):
    """Apply cached MF impulses so live velocity and lambda share one state.

    PGS updates velocity by ``M^-1 J^T (lambda_new-lambda_old)``.  A nonzero
    warm-start lambda must therefore have its existing velocity contribution
    installed before the first iteration; otherwise retracting a stale normal
    impulse creates an attractive impulse.
    """
    tid = wp.tid()
    component = tid % 6
    local_body = (tid // 6) % max_mf_bodies
    world = tid // (6 * max_mf_bodies)
    if local_body >= mf_body_count[world]:
        return

    global_dof = mf_body_dof_start[world, local_body] + component
    delta_velocity = float(0.0)
    for i in range(mf_constraint_count[world]):
        impulse = mf_impulses[world, i]
        if impulse == 0.0:
            continue
        if mf_local_body_a[world, i] == local_body:
            delta_velocity += mf_MiJt_a[world, i, component] * impulse
        if mf_local_body_b[world, i] == local_body:
            delta_velocity += mf_MiJt_b[world, i, component] * impulse

    if set_output_to_delta != 0:
        velocity[global_dof] = delta_velocity
    else:
        velocity[global_dof] += delta_velocity


@wp.kernel
def classify_parallel_sweep_tiers(
    world_constraint_count: wp.array[int],
    mf_constraint_count: wp.array[int],
    tier_lo: wp.array[int],
    tier_hi: wp.array[int],
    n_tiers: int,
    # outputs
    tier_worlds: wp.array2d[int],
    tier_counts: wp.array[int],
):
    """Append each dense-only world to the list of the parallel sweep tier that owns it.

    Tier ``t`` owns worlds with ``tier_lo[t] < rows <= tier_hi[t]`` and no matrix-free rows. Each tier kernel then
    grid-strides over its own list instead of launching every world and exiting (an exiting block still paid four
    dependent global loads: ~55 us per launch at 16k worlds).
    """
    world = wp.tid()
    m = world_constraint_count[world]
    if m <= 0 or mf_constraint_count[world] != 0:
        return
    for t in range(n_tiers):
        if m > tier_lo[t] and m <= tier_hi[t]:
            idx = wp.atomic_add(tier_counts, t, 1)
            tier_worlds[t, idx] = world
            return


@wp.kernel
def finalize_world_diag_cfm(
    world_constraint_count: wp.array[int],
    world_row_cfm: wp.array2d[float],
    mf_constraint_count: wp.array[int],
    skip_rows_le: int,
    # in/out
    world_diag: wp.array2d[float],
):
    """Add CFM to world diagonal after Delassus accumulation."""
    world = wp.tid()
    m = world_constraint_count[world]
    if skip_rows_le > 0 and m <= skip_rows_le and mf_constraint_count[world] == 0:
        return

    for i in range(m):
        world_diag[world, i] += world_row_cfm[world, i]


# =============================================================================
# Parallelized Non-Tiled Kernels for Heterogeneous Multi-Articulation
# =============================================================================
# These kernels parallelize across constraints (and constraint pairs) to achieve
# much better GPU utilization than the single-thread-per-articulation versions.


@wp.kernel
def hinv_jt_par_row(
    # Grouped Cholesky factor storage [n_arts, n_dofs, n_dofs]
    L_group: wp.array3d[float],
    # Size-grouped Jacobian [n_arts_of_size, max_constraints, n_dofs]
    J_group: wp.array3d[float],
    # Indirection arrays
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    world_constraint_count: wp.array[int],
    # Size parameters
    n_dofs: int,
    max_constraints: int,
    n_arts: int,
    write_world: int,
    # Output: Y = H^-1 * J^T [n_arts_of_size, max_constraints, n_dofs]
    Y_group: wp.array3d[float],
    J_world: wp.array3d[float],
    Y_world: wp.array3d[float],
):
    """
    Compute Y = H^-1 * J^T for one size group using forward/backward substitution.

    Uses L_group (3D array) grouped by DOF size.
    Efficient for small articulations where tile overhead dominates.

    Each thread handles one (articulation, constraint) pair.

    For each articulation in the group, solves:
        L * L^T * Y = J^T
    Using:
        1. Forward substitution: L * Z = J^T
        2. Backward substitution: L^T * Y = Z

    Thread dimension: n_arts_of_size * max_constraints
    """
    tid = wp.tid()

    # Decode thread index
    c = tid % max_constraints  # constraint index
    idx = tid // max_constraints  # group index (articulation within size group)

    # Bounds check for articulation
    if idx >= n_arts:
        return

    art = group_to_art[idx]
    world = art_to_world[art]
    n_constraints = world_constraint_count[world]

    # Early exit if this constraint is beyond the actual count
    if c >= n_constraints:
        return

    # ----------------------------------------------------------------
    # Forward substitution: L * z = j
    # L is lower triangular, so solve from top to bottom
    # ----------------------------------------------------------------
    for i in range(n_dofs):
        # z[i] = (j[i] - sum_{k<i} L[i,k] * z[k]) / L[i,i]
        val = J_group[idx, c, i]

        for k in range(i):
            # z[k] is stored in Y_group temporarily
            val -= L_group[idx, i, k] * Y_group[idx, c, k]

        L_ii = L_group[idx, i, i]
        if L_ii != 0.0:
            Y_group[idx, c, i] = val / L_ii
        else:
            Y_group[idx, c, i] = 0.0

    # ----------------------------------------------------------------
    # Backward substitution: L^T * y = z
    # L^T is upper triangular, so solve from bottom to top
    # z is currently stored in Y_group, we overwrite with y
    # ----------------------------------------------------------------
    for i_rev in range(n_dofs):
        i = n_dofs - 1 - i_rev

        # y[i] = (z[i] - sum_{k>i} L[k,i] * y[k]) / L[i,i]
        # Note: L^T[i,k] = L[k,i], so we read L[k,i] for k > i
        val = Y_group[idx, c, i]  # This is z[i] from forward pass

        for k in range(i + 1, n_dofs):
            val -= L_group[idx, k, i] * Y_group[idx, c, k]

        L_ii = L_group[idx, i, i]
        if L_ii != 0.0:
            Y_group[idx, c, i] = val / L_ii
        else:
            Y_group[idx, c, i] = 0.0

    if write_world != 0:
        dof_offset = articulation_world_dof_offset[art]
        for i in range(n_dofs):
            J_world[world, c, dof_offset + i] = J_group[idx, c, i]
            Y_world[world, c, dof_offset + i] = Y_group[idx, c, i]


@wp.kernel(module=_MASS_DYNAMICS_KERNEL_MODULE)
def hinv_jt_diagonal(
    L_group: wp.array3d[float],
    J_group: wp.array3d[float],
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    world_constraint_count: wp.array[int],
    n_dofs: int,
    max_constraints: int,
    n_arts: int,
    write_group: int,
    write_world: int,
    compute_diag: int,
    # outputs
    Y_group: wp.array3d[float],
    J_world: wp.array3d[float],
    Y_world: wp.array3d[float],
    diag_group: wp.array2d[float],
):
    """Apply a structurally diagonal inverse to one Jacobian row."""
    element = wp.tid()
    constraint = element % max_constraints
    group = element // max_constraints
    if group >= n_arts:
        return

    art = group_to_art[group]
    world = art_to_world[art]
    if constraint >= world_constraint_count[world]:
        return

    dof_offset = int(0)
    if write_world != 0:
        dof_offset = articulation_world_dof_offset[art]
    diagonal_sum = float(0.0)
    for dof in range(n_dofs):
        jacobian = J_group[group, constraint, dof]
        factor = L_group[group, dof, dof]
        response = jacobian / factor
        response = response / factor
        if write_group != 0:
            Y_group[group, constraint, dof] = response
        if write_world != 0:
            J_world[world, constraint, dof_offset + dof] = jacobian
            Y_world[world, constraint, dof_offset + dof] = response
        if compute_diag != 0:
            diagonal_sum += jacobian * response

    if compute_diag != 0:
        diag_group[group, constraint] = diagonal_sum


@wp.kernel
def hinv_jt_par_row_contact_fallback(
    L_group: wp.array3d[float],
    J_group: wp.array3d[float],
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    world_constraint_count: wp.array[int],
    local_solve_owner: wp.array[int],
    world_row_restitution: wp.array2d[float],
    n_dofs: int,
    max_constraints: int,
    n_arts: int,
    write_world: int,
    Y_group: wp.array3d[float],
    J_world: wp.array3d[float],
    Y_world: wp.array3d[float],
):
    """Compute response only for worlds that need the general contact solver."""
    tid = wp.tid()
    group_index = tid // 32
    lane = tid % 32
    if group_index >= n_arts:
        return

    art = group_to_art[group_index]
    world = art_to_world[art]
    constraint_count = world_constraint_count[world]
    if local_solve_owner[world] != PGS_LOCAL_SOLVE_OWNER_GENERAL:
        # Local owners construct response in their fused solver.  Only impact
        # rows need a world Jacobian for the separate restitution target pass;
        # keep every other local row in articulation-local storage.
        if write_world != 0:
            dof_offset = articulation_world_dof_offset[art]
            constraint = lane
            while constraint < constraint_count:
                if world_row_restitution[world, constraint] > 0.0:
                    for i in range(n_dofs):
                        J_world[world, constraint, dof_offset + i] = J_group[group_index, constraint, i]
                constraint += 32
        return

    constraint = lane
    while constraint < constraint_count:
        for i in range(n_dofs):
            value = J_group[group_index, constraint, i]
            for k in range(i):
                value -= L_group[group_index, i, k] * Y_group[group_index, constraint, k]

            diagonal = L_group[group_index, i, i]
            if diagonal != 0.0:
                Y_group[group_index, constraint, i] = value / diagonal
            else:
                Y_group[group_index, constraint, i] = 0.0

        for reverse in range(n_dofs):
            i = n_dofs - 1 - reverse
            value = Y_group[group_index, constraint, i]
            for k in range(i + 1, n_dofs):
                value -= L_group[group_index, k, i] * Y_group[group_index, constraint, k]

            diagonal = L_group[group_index, i, i]
            if diagonal != 0.0:
                Y_group[group_index, constraint, i] = value / diagonal
            else:
                Y_group[group_index, constraint, i] = 0.0

        if write_world != 0:
            dof_offset = articulation_world_dof_offset[art]
            for i in range(n_dofs):
                J_world[world, constraint, dof_offset + i] = J_group[group_index, constraint, i]
                Y_world[world, constraint, dof_offset + i] = Y_group[group_index, constraint, i]
        constraint += 32


@wp.kernel
def classify_local_solve_worlds(
    world_constraint_count: wp.array[int],
    dense_phase_bounds: wp.array2d[int],
    mf_constraint_count: wp.array[int],
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    mf_row_type: wp.array2d[int],
    body_to_articulation: wp.array[int],
    articulation_dof_count: wp.array[int],
    local_primary_articulation: wp.array[int],
    local_pair_articulation: wp.array[int],
    local_residual_pair_articulation: wp.array[int],
    local_max_constraints: int,
    local_residual_max_constraints: int,
    local_residual_mf_max_constraints: int,
    # outputs
    local_solve_owner: wp.array[int],
    general_world_count: wp.array[int],
    general_worlds: wp.array[int],
):
    """Assign exact solver ownership and compact active general worlds."""
    world = wp.tid()
    row_count = world_constraint_count[world]
    mf_count = mf_constraint_count[world]
    primary_articulation = local_primary_articulation[world]
    pair_articulation = local_pair_articulation[world]
    residual_pair_articulation = local_residual_pair_articulation[world]
    single_phase = dense_phase_bounds[world, 1] == row_count

    local_mf = mf_count > 0 and mf_count <= local_residual_mf_max_constraints and residual_pair_articulation >= 0
    mf_row = int(0)
    while mf_row < mf_count and local_mf:
        body_a = mf_body_a[world, mf_row]
        body_b = mf_body_b[world, mf_row]
        if body_a >= 0 and body_to_articulation[body_a] != residual_pair_articulation:
            local_mf = False
        if body_b >= 0 and body_to_articulation[body_b] != residual_pair_articulation:
            local_mf = False
        # Free-body velocity-limit rows follow the general owner's stateless unilateral law, which the
        # residual local loop does not implement: keep such worlds on the general owner.
        if mf_row_type[world, mf_row] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
            local_mf = False
        mf_row += 1

    owner = PGS_LOCAL_SOLVE_OWNER_GENERAL
    single_row_capacity = int(0)
    if primary_articulation >= 0:
        single_row_capacity = wp.min(local_max_constraints, articulation_dof_count[primary_articulation])
    if row_count > 0 and mf_count == 0:
        if single_phase and row_count <= single_row_capacity:
            owner = PGS_LOCAL_SOLVE_OWNER_SINGLE
        elif not single_phase and row_count <= local_max_constraints and pair_articulation >= 0:
            owner = PGS_LOCAL_SOLVE_OWNER_PAIR
    if owner == PGS_LOCAL_SOLVE_OWNER_GENERAL and (
        row_count > 0
        and row_count <= local_residual_max_constraints
        and residual_pair_articulation >= 0
        and ((mf_count == 0 and not single_phase) or local_mf)
    ):
        owner = PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL
    local_solve_owner[world] = owner
    if owner == PGS_LOCAL_SOLVE_OWNER_GENERAL and (row_count > 0 or mf_count > 0):
        general_index = wp.atomic_add(general_world_count, 0, 1)
        general_worlds[general_index] = world


@wp.kernel
def compact_local_pair_candidates(
    candidate_articulations: wp.array[int],
    candidate_secondary_articulations: wp.array[int],
    articulation_world: wp.array[int],
    local_solve_owner: wp.array[int],
    expected_owner: int,
    # outputs
    active_count: wp.array[int],
    active_articulations: wp.array[int],
    active_secondary_articulations: wp.array[int],
):
    """Compact topology candidates that selected paired local ownership."""
    candidate = wp.tid()
    articulation = candidate_articulations[candidate]
    world = articulation_world[articulation]
    if local_solve_owner[world] == expected_owner:
        active_index = wp.atomic_add(active_count, 0, 1)
        active_articulations[active_index] = articulation
        active_secondary_articulations[active_index] = candidate_secondary_articulations[candidate]


@wp.kernel
def clear_local_solve_diag(
    world_constraint_count: wp.array[int],
    local_solve_owner: wp.array[int],
    max_constraints: int,
    # output
    world_diag: wp.array2d[float],
):
    """Discard stale response diagonals for locally owned worlds."""
    tid = wp.tid()
    row = tid % max_constraints
    world = tid // max_constraints
    if local_solve_owner[world] != PGS_LOCAL_SOLVE_OWNER_GENERAL and row < world_constraint_count[world]:
        world_diag[world, row] = 0.0


@wp.kernel
def delassus_par_row_col(
    # Size-grouped arrays
    J_group: wp.array3d[float],  # [n_arts_of_size, max_constraints, n_dofs]
    Y_group: wp.array3d[float],  # [n_arts_of_size, max_constraints, n_dofs]
    # Indirection arrays
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    world_constraint_count: wp.array[int],
    # Size parameters
    n_dofs: int,
    max_constraints: int,
    n_arts: int,
    # Output: Delassus matrix C and diagonal (accumulated via atomics)
    world_C: wp.array3d[float],  # [world_count, max_constraints, max_constraints]
    world_diag: wp.array2d[float],  # [world_count, max_constraints]
):
    """
    Accumulate Delassus matrix contribution C += J * Y^T from one size group.

    PARALLELIZED VERSION: Each thread handles one (articulation, i, j) triplet.

    The Delassus matrix is: C = sum_art J_art * H_art^-1 * J_art^T = sum_art J_art * Y_art^T

    Since Y is stored as [constraint, dof], we compute:
        C[i,j] = sum_k J[i,k] * Y[j,k]

    Thread dimension: n_arts_of_size * max_constraints * max_constraints
    """
    tid = wp.tid()

    # Decode thread index
    j = tid % max_constraints
    i = (tid // max_constraints) % max_constraints
    idx = tid // (max_constraints * max_constraints)

    # Bounds check for articulation
    if idx >= n_arts:
        return

    art = group_to_art[idx]
    world = art_to_world[art]
    n_constraints = world_constraint_count[world]

    # Early exit if this (i, j) is beyond the actual constraint count
    if i >= n_constraints or j >= n_constraints:
        return

    # Compute C[i,j] = sum_k J[i,k] * Y[j,k]
    val = float(0.0)
    for k in range(n_dofs):
        val += J_group[idx, i, k] * Y_group[idx, j, k]

    if val != 0.0:
        wp.atomic_add(world_C, world, i, j, val)

    # Also accumulate diagonal separately (only when i == j)
    if i == j and val != 0.0:
        wp.atomic_add(world_diag, world, i, val)


# =============================================================================
# Tiled kernels for homogenous multi-articulation support
# =============================================================================


@wp.kernel(module=_MASS_DYNAMICS_KERNEL_MODULE)
def crba_fill_par_dof(
    articulation_start: wp.array[int],
    articulation_dof_start: wp.array[int],
    mass_update_mask: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_S_s: wp.array[wp.spatial_vector],
    body_I_c: wp.array[wp.spatial_matrix],
    # Size-group parameters
    group_to_art: wp.array[int],
    n_dofs: int,  # = TILE_DOF for tiled path
    fused_augmented_drive: int,
    drive_row_by_dof: wp.array[int],
    row_K: wp.array[float],
    # outputs
    H_group: wp.array3d[float],  # [n_arts_of_size, n_dofs, n_dofs]
):
    """
    CRBA fill kernel that writes directly to size-grouped H storage.

    Thread dimension: n_arts_of_size * n_dofs (one thread per articulation-column pair)

    This version is for homogenous multi-articulation where all articulations have
    the same DOF count equal to TILE_DOF.
    """
    tid = wp.tid()

    group_idx = tid // n_dofs
    col_idx = tid % n_dofs

    art_idx = group_to_art[group_idx]

    if mass_update_mask[art_idx] == 0:
        return

    # All articulations in this group have exactly n_dofs DOFs
    if col_idx >= n_dofs:
        return

    global_dof_start = articulation_dof_start[art_idx]
    target_dof_global = global_dof_start + col_idx

    joint_start = articulation_start[art_idx]
    joint_end = articulation_start[art_idx + 1]

    # Find the joint that owns this DOF
    pivot_joint = int(-1)
    for j in range(joint_start, joint_end):
        q_start = joint_qd_start[j]
        q_end = joint_qd_start[j + 1]
        if target_dof_global >= q_start and target_dof_global < q_end:
            pivot_joint = j
            break

    if pivot_joint == -1:
        return

    # Compute Force F = I_c[pivot] * S[column]
    S_col = joint_S_s[target_dof_global]
    # body_I_c is BODY-indexed; joint index only coincides with the child body
    # index in loop-free models.
    I_comp = body_I_c[joint_child[pivot_joint]]
    F = I_comp * S_col

    # Walk up the tree and project F onto ancestors
    # H[row, col] = S[row] * F
    curr = pivot_joint

    while curr != -1:
        if curr < joint_start:
            break

        q_start = joint_qd_start[curr]
        q_dim = joint_dof_dim[curr]
        count = q_dim[0] + q_dim[1]

        dof_offset_local = q_start - global_dof_start

        for k in range(count):
            row_idx = dof_offset_local + k

            S_row = joint_S_s[q_start + k]
            val = wp.dot(S_row, F)

            # The column thread uniquely owns its diagonal. Fold the
            # augmented-drive inertia into that write so the immutable drive
            # topology does not require a second read/modify/write pass over H.
            if fused_augmented_drive != 0 and row_idx == col_idx:
                drive_row = drive_row_by_dof[target_dof_global]
                if drive_row >= 0:
                    K = row_K[drive_row]
                    if K > 0.0:
                        val += K

            # Write to grouped 3D array
            H_group[group_idx, row_idx, col_idx] = val
            H_group[group_idx, col_idx, row_idx] = val

        curr = joint_ancestor[curr]


@wp.kernel
def trisolve_loop(
    L_group: wp.array3d[float],  # [n_arts_of_size, n_dofs, n_dofs]
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    n_dofs: int,
    joint_tau: wp.array[float],  # [total_dofs]
    # output
    joint_qdd: wp.array[float],  # [total_dofs]
):
    """
    Solve L * L^T * qdd = tau for grouped articulations using forward/backward substitution.

    Thread dimension: n_arts_of_size (one thread per articulation in this size group)
    """
    idx = wp.tid()
    art = group_to_art[idx]
    dof_start = articulation_dof_start[art]

    # Forward substitution: L * z = tau
    # z is stored temporarily in joint_qdd
    for i in range(n_dofs):
        val = joint_tau[dof_start + i]
        for k in range(i):
            L_ik = L_group[idx, i, k]
            val -= L_ik * joint_qdd[dof_start + k]

        L_ii = L_group[idx, i, i]
        if L_ii != 0.0:
            joint_qdd[dof_start + i] = val / L_ii
        else:
            joint_qdd[dof_start + i] = 0.0

    # Backward substitution: L^T * qdd = z
    for i_rev in range(n_dofs):
        i = n_dofs - 1 - i_rev

        val = joint_qdd[dof_start + i]
        for k in range(i + 1, n_dofs):
            L_ki = L_group[idx, k, i]
            val -= L_ki * joint_qdd[dof_start + k]

        L_ii = L_group[idx, i, i]
        if L_ii != 0.0:
            joint_qdd[dof_start + i] = val / L_ii
        else:
            joint_qdd[dof_start + i] = 0.0


@wp.kernel
def gather_tau_to_groups(
    joint_tau: wp.array[float],  # [total_dofs]
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    n_dofs: int,
    tau_group: wp.array3d[float],  # [n_arts, n_dofs, 1]
):
    """Gather joint_tau from 1D array into grouped 3D buffer for tiled solve.

    Thread dimension: n_arts_of_size (one thread per articulation in this size group)
    """
    idx = wp.tid()
    art = group_to_art[idx]
    dof_start = articulation_dof_start[art]
    for i in range(n_dofs):
        tau_group[idx, i, 0] = joint_tau[dof_start + i]


@wp.kernel
def scatter_qdd_from_groups(
    qdd_group: wp.array3d[float],  # [n_arts, n_dofs, 1]
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    n_dofs: int,
    joint_qdd: wp.array[float],  # [total_dofs]
):
    """Scatter qdd from grouped 3D buffer back to 1D array after tiled solve.

    Thread dimension: n_arts_of_size (one thread per articulation in this size group)
    """
    idx = wp.tid()
    art = group_to_art[idx]
    dof_start = articulation_dof_start[art]
    for i in range(n_dofs):
        joint_qdd[dof_start + i] = qdd_group[idx, i, 0]


@wp.kernel
def vector_add_inplace(a: wp.array[float], b: wp.array[float]):
    """a[i] += b[i]"""
    i = wp.tid()
    a[i] = a[i] + b[i]


@wp.kernel
def compute_delta_and_accumulate(
    v_out: wp.array[float],
    v_snap: wp.array[float],
    v_accum: wp.array[float],
):
    """delta = v_out - v_snap; v_accum += delta; v_snap = delta (reuse buffer for rhs_accum input)"""
    i = wp.tid()
    delta = v_out[i] - v_snap[i]
    v_accum[i] = v_accum[i] + delta
    v_snap[i] = delta


@wp.func
def world_response_velocity(
    v_out: wp.array[float],
    world_dof_indices: wp.array2d[int],
    world: int,
    local_dof: int,
) -> float:
    """Read a compact response velocity, returning zero for padding."""
    global_dof = world_dof_indices[world, local_dof]
    if global_dof >= 0:
        return v_out[global_dof]
    return 0.0


# =============================================================================
# PGS Convergence Diagnostic Kernel (velocity-space mode)
# =============================================================================


@wp.kernel
def pgs_convergence_diagnostic_velocity(
    # Dense constraints
    constraint_count: wp.array[int],
    world_dof_indices: wp.array2d[int],
    rhs: wp.array2d[float],
    impulses: wp.array2d[float],
    prev_impulses: wp.array2d[float],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    row_mu: wp.array2d[float],
    J_world: wp.array3d[float],
    max_constraints: int,
    max_world_dofs: int,
    # MF constraints
    mf_constraint_count: wp.array[int],
    mf_rhs: wp.array2d[float],
    mf_impulses: wp.array2d[float],
    prev_mf_impulses: wp.array2d[float],
    mf_row_type: wp.array2d[int],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_dof_a: wp.array2d[int],
    mf_dof_b: wp.array2d[int],
    mf_max_constraints: int,
    # Velocity
    v_out: wp.array[float],
    # Output: [worlds, 4]
    metrics: wp.array2d[float],
):
    """Compute per-world PGS convergence metrics for velocity-space mode.

    Metrics:
        [0] max|delta_lambda| across all constraint rows
        [1] sum(lambda_n * residual_n) for normal contacts (complementarity gap)
        [2] sum(residual_t^2) for sticking friction contacts (tangent residual energy)
        [3] sum(FB(lambda_n, residual_n)^2) for normal contacts (Fischer-Burmeister)
    """
    world = wp.tid()

    m_dense = constraint_count[world]
    m_mf = mf_constraint_count[world]

    max_dl = float(0.0)
    comp_gap = float(0.0)
    tang_res = float(0.0)
    fb_merit = float(0.0)

    # --- Dense constraints ---
    for i in range(m_dense):
        lam = impulses[world, i]
        prev_lam = prev_impulses[world, i]
        dl = wp.abs(lam - prev_lam)
        if dl > max_dl:
            max_dl = dl

        # Compute residual: r_i = J_i * v + bias_i
        jv = float(0.0)
        for d in range(max_world_dofs):
            jv += J_world[world, i, d] * world_response_velocity(v_out, world_dof_indices, world, d)
        residual = jv + rhs[world, i]

        rt = row_type[world, i]
        if rt == PGS_CONSTRAINT_TYPE_CONTACT:
            # Normal: complementarity gap and FB
            comp_gap += lam * residual
            fb_val = wp.sqrt(lam * lam + residual * residual) - lam - residual
            fb_merit += fb_val * fb_val
        elif rt == PGS_CONSTRAINT_TYPE_FRICTION:
            # Friction: tangent residual for sticking contacts
            parent_idx = row_parent[world, i]
            lambda_n = patch_normal_load(row_parent, impulses, world, parent_idx)
            mu = row_mu[world, i]
            radius = mu * lambda_n
            if radius > 0.0:
                # Check if sticking: |lambda_t| < mu * lambda_n
                # Get sibling
                if i == parent_idx + 1:
                    sib = parent_idx + 2
                else:
                    sib = parent_idx + 1
                lam_t1 = impulses[world, i]
                lam_t2 = impulses[world, sib]
                t_mag = wp.sqrt(lam_t1 * lam_t1 + lam_t2 * lam_t2)
                if t_mag < radius * 0.999:  # sticking (with small tolerance)
                    tang_res += residual * residual

    # --- MF constraints ---
    for i in range(m_mf):
        lam = mf_impulses[world, i]
        prev_lam = prev_mf_impulses[world, i]
        dl = wp.abs(lam - prev_lam)
        if dl > max_dl:
            max_dl = dl

        # Compute residual: r = J_a * v_a + J_b * v_b + bias
        dof_a = mf_dof_a[world, i]
        dof_b = mf_dof_b[world, i]
        jv = float(0.0)
        if dof_a >= 0:
            for k in range(6):
                jv += mf_J_a[world, i, k] * world_response_velocity(v_out, world_dof_indices, world, dof_a + k)
        if dof_b >= 0:
            for k in range(6):
                jv += mf_J_b[world, i, k] * world_response_velocity(v_out, world_dof_indices, world, dof_b + k)
        residual = jv + mf_rhs[world, i]

        rt = mf_row_type[world, i]
        if rt == PGS_CONSTRAINT_TYPE_CONTACT:
            comp_gap += lam * residual
            fb_val = wp.sqrt(lam * lam + residual * residual) - lam - residual
            fb_merit += fb_val * fb_val
        elif rt == PGS_CONSTRAINT_TYPE_FRICTION:
            parent_idx = mf_row_parent[world, i]
            lambda_n = patch_normal_load(mf_row_parent, mf_impulses, world, parent_idx)
            mu = mf_row_mu[world, i]
            radius = mu * lambda_n
            if radius > 0.0:
                if i == parent_idx + 1:
                    sib = parent_idx + 2
                else:
                    sib = parent_idx + 1
                lam_t1 = mf_impulses[world, i]
                lam_t2 = mf_impulses[world, sib]
                t_mag = wp.sqrt(lam_t1 * lam_t1 + lam_t2 * lam_t2)
                if t_mag < radius * 0.999:
                    tang_res += residual * residual

    metrics[world, 0] = max_dl
    metrics[world, 1] = comp_gap
    metrics[world, 2] = tang_res
    metrics[world, 3] = fb_merit


# =============================================================================
# PGS NCP / MDP Residual Diagnostic Kernel (velocity-space mode)
# =============================================================================


@wp.func
def contact_friction_residuals(
    normal_impulse: float,
    friction_load: float,
    mu: float,
    normal_velocity: float,
    tangent_velocity: wp.vec2,
    tangent_impulse: wp.vec2,
):
    """Physical cone/MDP residuals with a separate normal budget for patch anchors."""
    speed = wp.length(tangent_velocity)
    radius = mu * friction_load
    cone = wp.max(wp.length(tangent_impulse) - radius, 0.0)
    # Normal complementarity stays local; tangent dissipation uses patch support.
    complementarity = wp.abs(
        normal_impulse * normal_velocity + radius * speed + wp.dot(tangent_impulse, tangent_velocity)
    )
    dual = wp.max(-normal_velocity, 0.0)
    direction = float(0.0)
    if speed > 1.0e-8 and friction_load > 1.0e-8:
        direction = wp.length(tangent_impulse + radius * tangent_velocity / speed)
        if radius > 1.0e-8:
            direction /= radius
    return wp.vec4(cone, complementarity, dual, direction)


@wp.kernel
def pgs_ncp_residuals_diagnostic_velocity(
    # Dense constraints
    constraint_count: wp.array[int],
    world_dof_indices: wp.array2d[int],
    rhs: wp.array2d[float],
    impulses: wp.array2d[float],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    row_mu: wp.array2d[float],
    row_phi: wp.array2d[float],
    J_world: wp.array3d[float],
    max_constraints: int,
    max_world_dofs: int,
    # MF constraints
    mf_constraint_count: wp.array[int],
    mf_rhs: wp.array2d[float],
    mf_impulses: wp.array2d[float],
    mf_row_type: wp.array2d[int],
    mf_row_parent: wp.array2d[int],
    mf_row_mu: wp.array2d[float],
    mf_row_phi: wp.array2d[float],
    mf_J_a: wp.array3d[float],
    mf_J_b: wp.array3d[float],
    mf_dof_a: wp.array2d[int],
    mf_dof_b: wp.array2d[int],
    mf_max_constraints: int,
    # Velocity
    v_out: wp.array[float],
    # Output: [worlds, 6]
    metrics: wp.array2d[float],
):
    """Compute per-world NCP / MDP residuals on the matrix_free PGS path.

    The six residuals per world are reduced with ``max`` across contact
    groups and follow the formulation in ``SolverRaisim.residuals`` (see
    ``artifacts/2026-04-16-slack-raisim/repos/mmacklin-newton-solver-raisim/
    newton/_src/solvers/raisim/residuals.py``):

    ``[0] r_compl``
        ``max_i |min(lambda_n_i, u_n_i + b_n_i)|`` — standard NCP
        complementarity residual.
    ``[1] r_cone``
        ``max_i max(||lambda_t_i|| - mu_i * lambda_n_i, 0)`` — friction
        cone violation (Coulomb).
    ``[2] r_gap``
        ``max_i max(-phi_i, 0)`` — signed-distance penetration.
    ``[3] r_ds_compl``
        ``max_i |<lambda, c + Gamma(c, mu)>|`` — de Saxcé / MDP
        complementarity with Gamma = (0, 0, mu*||c_T||).
    ``[4] r_ds_dual``
        ``max_i max(-u_n_i, 0)`` — dual-cone feasibility for the
        augmented velocity ``c + Gamma``, which simplifies to
        ``max(-u_n, 0)``.
    ``[5] r_mdp_dir``
        ``max_i ||lambda_t - (-mu*lambda_n)*(c_T / ||c_T||)|| / (mu*lambda_n)``
        — MDP direction error for actively sliding contacts
        (``||c_T|| > 1e-8`` and ``lambda_n > 1e-8``).

    For patch friction, tangent cone and MDP terms use the complete region's
    normal impulse and the anchor's share of its friction coefficient. Normal
    complementarity and gap remain per contact.

    Residual computation distinguishes row kinds:
    * Only rows of type ``PGS_CONSTRAINT_TYPE_CONTACT`` drive a contact
      iteration (both in the dense articulated buffer and the
      matrix-free free-rigid buffer).
    * Joint-limit rows (``PGS_CONSTRAINT_TYPE_JOINT_LIMIT``) and
      joint-target rows (``PGS_CONSTRAINT_TYPE_JOINT_TARGET``) are
      **skipped** — friction, cone, and MDP residuals do not apply to
      them. ``r_gap`` is also skipped for these rows (the gap concept is
      specific to contact normals).
    * Friction rows (``PGS_CONSTRAINT_TYPE_FRICTION``) are **not**
      iterated directly — they are read via the parent CONTACT row at
      ``parent_idx + 1`` and ``parent_idx + 2``, which avoids
      double-counting and correctly pairs tangent basis components.
    * ``u_n``, ``u_t1`` and ``u_t2`` are computed as the bias-free
      ``J_row * v_out`` (matching the raisim reference which reads
      joint-velocity directly). ``r_compl`` still uses ``u_n + b_n`` via
      ``rhs[row]`` to match the NCP statement.
    """
    world = wp.tid()

    m_dense = constraint_count[world]
    m_mf = mf_constraint_count[world]

    r_compl = float(0.0)
    r_cone = float(0.0)
    r_gap = float(0.0)
    r_ds_compl = float(0.0)
    r_ds_dual = float(0.0)
    r_mdp_dir = float(0.0)

    # ---- Dense constraints (articulated contacts + joint limits) ----
    for i in range(m_dense):
        rt = row_type[world, i]
        if rt != PGS_CONSTRAINT_TYPE_CONTACT:
            # skip friction rows (handled via parent), joint-limit rows,
            # and joint-target rows — they do not contribute NCP/MDP
            # contact residuals.
            continue

        # Normal row velocity (bias-free): u_n = J_n * v
        u_n = float(0.0)
        for d in range(max_world_dofs):
            u_n += J_world[world, i, d] * world_response_velocity(v_out, world_dof_indices, world, d)
        b_n = rhs[world, i]
        ln = impulses[world, i]

        # r_compl: |min(ln, u_n + b_n)|
        ubn = u_n + b_n
        if ln < ubn:
            compl = wp.abs(ln)
        else:
            compl = wp.abs(ubn)
        if compl > r_compl:
            r_compl = compl

        # r_gap: max(-phi, 0)
        neg_phi = -row_phi[world, i]
        if neg_phi > r_gap:
            r_gap = neg_phi

        # Friction rows at i+1, i+2 (if present and parented to i)
        lt1 = float(0.0)
        lt2 = float(0.0)
        u_t1 = float(0.0)
        u_t2 = float(0.0)
        mu = float(0.0)

        i1 = i + 1
        if i1 < max_constraints and i1 < m_dense:
            if row_type[world, i1] == PGS_CONSTRAINT_TYPE_FRICTION and row_parent[world, i1] == i:
                lt1 = impulses[world, i1]
                mu = row_mu[world, i1]
                for d in range(max_world_dofs):
                    u_t1 += J_world[world, i1, d] * world_response_velocity(v_out, world_dof_indices, world, d)

        i2 = i + 2
        if i2 < max_constraints and i2 < m_dense:
            if row_type[world, i2] == PGS_CONSTRAINT_TYPE_FRICTION and row_parent[world, i2] == i:
                lt2 = impulses[world, i2]
                if mu == 0.0:
                    mu = row_mu[world, i2]
                for d in range(max_world_dofs):
                    u_t2 += J_world[world, i2, d] * world_response_velocity(v_out, world_dof_indices, world, d)

        # r_cone
        friction_load = ln
        if mu > 0.0:
            friction_load = patch_normal_load(row_parent, impulses, world, i)
        friction_residual = contact_friction_residuals(
            ln, friction_load, mu, u_n, wp.vec2(u_t1, u_t2), wp.vec2(lt1, lt2)
        )
        r_cone = wp.max(r_cone, friction_residual[0])
        r_ds_compl = wp.max(r_ds_compl, friction_residual[1])
        r_ds_dual = wp.max(r_ds_dual, friction_residual[2])
        r_mdp_dir = wp.max(r_mdp_dir, friction_residual[3])

    # ---- Matrix-free constraints (free-rigid contacts) ----
    for i in range(m_mf):
        rt = mf_row_type[world, i]
        if rt != PGS_CONSTRAINT_TYPE_CONTACT:
            continue

        dof_a = mf_dof_a[world, i]
        dof_b = mf_dof_b[world, i]
        u_n = float(0.0)
        if dof_a >= 0:
            for k in range(6):
                u_n += mf_J_a[world, i, k] * world_response_velocity(v_out, world_dof_indices, world, dof_a + k)
        if dof_b >= 0:
            for k in range(6):
                u_n += mf_J_b[world, i, k] * world_response_velocity(v_out, world_dof_indices, world, dof_b + k)
        b_n = mf_rhs[world, i]
        ln = mf_impulses[world, i]

        ubn = u_n + b_n
        if ln < ubn:
            compl = wp.abs(ln)
        else:
            compl = wp.abs(ubn)
        if compl > r_compl:
            r_compl = compl

        neg_phi = -mf_row_phi[world, i]
        if neg_phi > r_gap:
            r_gap = neg_phi

        lt1 = float(0.0)
        lt2 = float(0.0)
        u_t1 = float(0.0)
        u_t2 = float(0.0)
        mu = float(0.0)

        i1 = i + 1
        if i1 < mf_max_constraints and i1 < m_mf:
            if mf_row_type[world, i1] == PGS_CONSTRAINT_TYPE_FRICTION and mf_row_parent[world, i1] == i:
                lt1 = mf_impulses[world, i1]
                mu = mf_row_mu[world, i1]
                dof_a1 = mf_dof_a[world, i1]
                dof_b1 = mf_dof_b[world, i1]
                if dof_a1 >= 0:
                    for k in range(6):
                        u_t1 += mf_J_a[world, i1, k] * world_response_velocity(
                            v_out, world_dof_indices, world, dof_a1 + k
                        )
                if dof_b1 >= 0:
                    for k in range(6):
                        u_t1 += mf_J_b[world, i1, k] * world_response_velocity(
                            v_out, world_dof_indices, world, dof_b1 + k
                        )

        i2 = i + 2
        if i2 < mf_max_constraints and i2 < m_mf:
            if mf_row_type[world, i2] == PGS_CONSTRAINT_TYPE_FRICTION and mf_row_parent[world, i2] == i:
                lt2 = mf_impulses[world, i2]
                if mu == 0.0:
                    mu = mf_row_mu[world, i2]
                dof_a2 = mf_dof_a[world, i2]
                dof_b2 = mf_dof_b[world, i2]
                if dof_a2 >= 0:
                    for k in range(6):
                        u_t2 += mf_J_a[world, i2, k] * world_response_velocity(
                            v_out, world_dof_indices, world, dof_a2 + k
                        )
                if dof_b2 >= 0:
                    for k in range(6):
                        u_t2 += mf_J_b[world, i2, k] * world_response_velocity(
                            v_out, world_dof_indices, world, dof_b2 + k
                        )

        friction_load = ln
        if mu > 0.0:
            friction_load = patch_normal_load(mf_row_parent, mf_impulses, world, i)
        friction_residual = contact_friction_residuals(
            ln, friction_load, mu, u_n, wp.vec2(u_t1, u_t2), wp.vec2(lt1, lt2)
        )
        r_cone = wp.max(r_cone, friction_residual[0])
        r_ds_compl = wp.max(r_ds_compl, friction_residual[1])
        r_ds_dual = wp.max(r_ds_dual, friction_residual[2])
        r_mdp_dir = wp.max(r_mdp_dir, friction_residual[3])

    metrics[world, 0] = r_compl
    metrics[world, 1] = r_cone
    metrics[world, 2] = r_gap
    metrics[world, 3] = r_ds_compl
    metrics[world, 4] = r_ds_dual
    metrics[world, 5] = r_mdp_dir
