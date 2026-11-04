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


import warp as wp

from ...sim import BodyFlags, JointType
from .contact_filters import contact_normal_gap_limit
from .contact_input import ContactInput, ContactRoutes
from .contact_laws import (
    CONTACT_SPIN_ROW,
    contact_endpoints,
    contact_points,
    contact_responds,
    contact_tangent_basis,
    jacobian_entry,
    spin_entry,
)
from .dense_rows import (
    UNIT_CONTACT,
    UNIT_LINK,
    DenseRows,
    MfRows,
    RowSupport,
    clear_side,
    jacobian_row,
    row_sides,
    unit_source,
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
from .friction_patches import (
    CONTACT_PATH_RESOLVED,
    FrictionPatches,
)
from .joint_table import FreeBodies, JointTable
from .projection_laws import (
    ball,
    disk,
)
from .row_laws import (
    LINK_ROWS,
    contact_restitution_rhs,
    joint_limit_side,
    limit_rhs,
    row_rhs,
    velocity_limit_active,
)
from .solver_fields import Links, LocalOwners
from .spatial_laws import free_root_transport

PGS_CONSTRAINT_TYPE_CONTACT = 0
# A free rigid body's speed row: it bounds the magnitude of the body's linear or angular velocity (type 1 was the
# PhysX drive row; drives fold into the mass matrix).
PGS_CONSTRAINT_TYPE_SPEED = 1
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
# Link row: one of the six bilateral rows that tie a body's shadow to its primary piece (:mod:`.body_cut`), unbounded
# lambda, Baumgarte drift correction.
PGS_CONSTRAINT_TYPE_LINK = 6
PGS_CONSTRAINT_TYPE_TORSION = 7
PGS_CONSTRAINT_TYPE_COUNT = 8

# Keep launch-geometry-specific kernels out of the large general kernel module: Warp compiles one whole module variant
# per block dimension.
_MASS_DYNAMICS_KERNEL_MODULE = wp.Module(f"{__name__}.mass_dynamics")

PGS_LOCAL_SOLVE_OWNER_GENERAL = 0
PGS_LOCAL_SOLVE_OWNER_SINGLE = 1
PGS_LOCAL_SOLVE_OWNER_PAIR = 2
PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL = 3


@wp.kernel
def local_solve_launch_gate():
    """Create a minimal graph dependency ahead of the bulk local solves."""

    pass


@wp.kernel
def compute_com_transforms(
    body_com: wp.array[wp.vec3],
    # outputs
    body_X_com: wp.array[wp.transform],
):
    tid = wp.tid()
    com = body_com[tid]
    body_X_com[tid] = wp.transform(com, wp.quat_identity())


@wp.func
def _prescaled_axis_count(joint: int, joints: JointTable) -> int:
    """Return how many leading DOFs of ``joint`` the PhysX velocity prescale reads and scales.

    Joint types without a PhysX joint velocity limit and kinematic children
    have none.
    """
    jtype = joints.joint_type[joint]
    if jtype != JointType.PRISMATIC and jtype != JointType.REVOLUTE and jtype != JointType.D6:
        return 0
    if (joints.body_flags[joints.joint_child[joint]] & BodyFlags.KINEMATIC) != 0:
        return 0
    return joints.joint_dof_dim[joint, 0] + joints.joint_dof_dim[joint, 1]


@wp.kernel(enable_backward=False)
def prescale_joint_velocity_limits(
    joints: JointTable,
    articulation_count: int,
    lanes: int,
    joint_qd: wp.array[float],
    # out
    prescaled_joint_qd: wp.array[float],
):
    """PhysX-style pre-solve joint velocity scaling.

    PhysX computes a single ratio per articulation from maxJointVelocity and
    applies that ratio to all articulation DOFs before building link velocities.
    This is separate from the velocity-limit constraint rows solved later.

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
        joint_start = joints.articulation_start[art]
        joint_end = joints.articulation_start[art + 1]

    ratio = float(1.0)
    for j in range(joint_start + lane, joint_end, lanes):
        qd_start = joints.joint_qd_start[j]
        for axis in range(_prescaled_axis_count(j, joints)):
            dof = qd_start + axis
            limit = joints.joint_velocity_limit[dof]
            qd_abs = wp.abs(joint_qd[dof])
            if limit > 0.0 and wp.isfinite(limit) and qd_abs > 0.0:
                scale = limit / qd_abs
                if scale < ratio:
                    ratio = scale
    # Padding lanes past the last articulation take part with ratio 1.
    ratio = lane_minimum(ratio, lanes)

    if art == 0:
        # Joints ahead of the first articulation keep their velocity.
        for dof in range(lane, joints.joint_qd_start[joint_start], lanes):
            prescaled_joint_qd[dof] = joint_qd[dof]
    if ratio < 1.0:
        for j in range(joint_start + lane, joint_end, lanes):
            qd_start = joints.joint_qd_start[j]
            scaled_end = qd_start + _prescaled_axis_count(j, joints)
            for dof in range(qd_start, joints.joint_qd_start[j + 1]):
                qd = joint_qd[dof]
                if dof < scaled_end:
                    qd = qd * ratio
                prescaled_joint_qd[dof] = qd
    else:
        for dof in range(joints.joint_qd_start[joint_start] + lane, joints.joint_qd_start[joint_end], lanes):
            prescaled_joint_qd[dof] = joint_qd[dof]


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
            v_com = v_com + (a_s + free_root_transport(v_com, w_prev)) * dt
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


@wp.kernel(module=_MASS_DYNAMICS_KERNEL_MODULE)
def prepare_fused_diagonal_joint_limits(
    joints: JointTable,
    world_dof_indices: wp.array2d[int],
    max_world_dofs: int,
    fused_limit_dof_mask: wp.array[int],
    limit_q_index: wp.array[int],
    joint_q: wp.array[float],
    activation_gap: float,
    pgs_beta: float,
    dt: float,
    v_hat: wp.array[float],
    # outputs
    active_sides: wp.array2d[int],
    lower_rhs: wp.array2d[float],
    upper_rhs: wp.array2d[float],
):
    """Prepare independent diagonal-articulation limit projections per world coordinate: each active side's
    right-hand side and its incident velocity, ``+v_hat`` for the lower side's row and ``-v_hat`` for the upper's."""
    element = wp.tid()
    world = element // max_world_dofs
    local_dof = element - world * max_world_dofs
    global_dof = world_dof_indices[world, local_dof]

    active = int(0)
    rhs_lower = float(0.0)
    rhs_upper = float(0.0)
    if global_dof >= 0 and fused_limit_dof_mask[global_dof] != 0 and joints.prescribed[global_dof] == 0:
        q_index = limit_q_index[global_dof]
        if q_index >= 0:
            q = joint_q[q_index]
            lower = joints.joint_limit_lower[global_dof]
            upper = joints.joint_limit_upper[global_dof]
            inv_dt = 1.0 / dt
            lower_active, lower_phi, _lower_sign = joint_limit_side(q, lower, upper, 0, activation_gap)
            if lower_active:
                rhs_lower = limit_rhs(lower_phi, pgs_beta, 0.0, inv_dt, 1.0, 1.0) + v_hat[global_dof]
                active |= 1
            upper_active, upper_phi, _upper_sign = joint_limit_side(q, lower, upper, 1, activation_gap)
            if upper_active:
                rhs_upper = limit_rhs(upper_phi, pgs_beta, 0.0, inv_dt, 1.0, 1.0) - v_hat[global_dof]
                active |= 2

    active_sides[world, local_dof] = active
    lower_rhs[world, local_dof] = rhs_lower
    upper_rhs[world, local_dof] = rhs_upper


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
    kinematic_joint_mask: wp.array[int],
    body_q: wp.array[wp.transform],
    body_inertia: wp.array[wp.mat33],
    walk: wp.array2d[float],
    joint_qd: wp.array[float],
    dt: float,
    v_hat: wp.array[float],
):
    """The isolated rigid-body gyroscopic update of a free root's predicted velocity. Its effective angular inertia is
    the angular block of the body's walk, ``L_aa L_aa^T`` (armature included), with ``L_aa`` from the walk's records
    of the body's coordinates (``walk``, :data:`.tree_factor.WALK_COORDINATE` floats a global coordinate, each column by
    its non-zero half at 0..2, ``ut`` at 4..9, ``1 / ell`` signed at 3). A free root's turns are its class's angular
    half (:data:`.tree_scan.WALK_ANGULAR`: no side, the record's other half zero), so ``s_i . ut_k`` reads the stored
    half against ``ut_k``'s angular part. The walk eliminates the angular coordinates first, last first, so
    ``L(i, i) = ell_i`` and ``L(i, k) = s_i . ut_k`` for the coordinate ``k > i`` eliminated before ``i``."""
    d = _active_free_root_dof_start(free_root_joint_indices, joint_qd_start, kinematic_joint_mask, root_index)
    if d < 0:
        return
    w = wp.vec3(joint_qd[d + 3], joint_qd[d + 4], joint_qd[d + 5])
    body = joint_child[free_root_joint_indices[root_index]]
    art = body_to_articulation[body]
    if is_free_rigid[art] == 0:
        return
    predicted_world = wp.vec3(v_hat[d + 3], v_hat[d + 4], v_hat[d + 5])
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
    # The angular block of the walk's leaf-first factor P H P^T = L L^T: row r is physical coordinate 3 + r, column c
    # the coordinate 5 - c eliminated c-th, so the block is ``lower lower^T``.
    lower = wp.mat33(0.0)
    for r in range(3):
        i = d + 3 + r
        for c in range(3 - r):
            k = d + 5 - c
            if k == i:
                lower[r, c] = 1.0 / wp.abs(walk[i, 3])
            else:
                value = float(0.0)
                for m in range(3):
                    value += walk[i, m] * walk[k, 7 + m]
                lower[r, c] = value
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
    kinematic_joint_mask: wp.array[int],
    body_q: wp.array[wp.transform],
    body_inertia: wp.array[wp.mat33],
    walk: wp.array2d[float],
    joint_qd: wp.array[float],
    dt: float,
    v_hat: wp.array[float],
):
    _light_apply_free_root_velocity_corrections(
        wp.tid(),
        free_root_joint_indices,
        joint_qd_start,
        joint_child,
        body_to_articulation,
        is_free_rigid,
        kinematic_joint_mask,
        body_q,
        body_inertia,
        walk,
        joint_qd,
        dt,
        v_hat,
    )


@wp.kernel
def damp_free_root_spin(
    free_root_joint_indices: wp.array[int],
    joint_qd_start: wp.array[int],
    kinematic_joint_mask: wp.array[int],
    spin_scale: float,
    v_hat: wp.array[float],
):
    """Scale each active free root's predicted spin by ``1 - angular_damping dt``, before the rows see it: the solved
    velocity then satisfies its rows at the step's end."""
    d = _active_free_root_dof_start(free_root_joint_indices, joint_qd_start, kinematic_joint_mask, wp.tid())
    if d < 0:
        return
    v_hat[d + 3] = v_hat[d + 3] * spin_scale
    v_hat[d + 4] = v_hat[d + 4] * spin_scale
    v_hat[d + 5] = v_hat[d + 5] * spin_scale


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
        parent,
        joint_q_new,
        joint_qd_new,
    )


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


@wp.func
def _contact_linear_force(
    dense_first: wp.array[int],
    body_base: int,
    routes: ContactRoutes,
    c: int,
    world_impulses: wp.array2d[wp.float32],
    world_constraint_count: wp.array[wp.int32],
    mf_constraint_count: wp.array[wp.int32],
    world_row_type: wp.array2d[wp.int32],
    world_row_parent: wp.array2d[wp.int32],
    enable_friction: int,
    inv_dt: float,
    # outputs
    rigid_contact_force: wp.array[wp.vec3],
):
    """Convert solved FeatherPGS contact impulses into world-frame forces: a contact's rows sit at its slot of its
    world's line, a free body's contact's (``path`` 1) among the free bodies' rows from ``body_base``."""
    total_contacts = routes.count[0]
    if c >= total_contacts:
        return

    force = wp.vec3(0.0)
    slot = routes.slot[c]
    path = routes.path[c]

    if slot >= 0 and path >= 0 and inv_dt > 0.0:
        world = routes.world[c]
        line = dense_first[world]
        # Contacts store normals from shape 0 toward shape 1 (A-to-B). FeatherPGS
        # solves along the opposite direction internally, which corresponds to the
        # force on shape/body 0 from shape/body 1.
        normal = -routes.normal[c]

        lam_n = world_impulses[line, slot]
        lam_t0 = 0.0
        lam_t1 = 0.0
        # The end of the contact's class in the line.
        end = world_constraint_count[world]
        if path == 1:
            end = body_base + mf_constraint_count[world]
        if (
            enable_friction != 0
            and slot + 2 < end
            and world_row_type[line, slot + 1] == PGS_CONSTRAINT_TYPE_FRICTION
            and world_row_parent[line, slot + 1] == slot
            and world_row_type[line, slot + 2] == PGS_CONSTRAINT_TYPE_FRICTION
            and world_row_parent[line, slot + 2] == slot
        ):
            lam_t0 = world_impulses[line, slot + 1]
            lam_t1 = world_impulses[line, slot + 2]

        force = lam_n * normal
        if enable_friction != 0:
            tangent0, tangent1 = contact_tangent_basis(normal)
            force += lam_t0 * tangent0 + lam_t1 * tangent1
        force *= inv_dt

    rigid_contact_force[c] = force


@wp.kernel
def compute_contact_linear_force_from_impulses(
    dense_first: wp.array[int],
    body_base: int,
    routes: ContactRoutes,
    world_impulses: wp.array2d[wp.float32],
    world_constraint_count: wp.array[wp.int32],
    mf_constraint_count: wp.array[wp.int32],
    world_row_type: wp.array2d[wp.int32],
    world_row_parent: wp.array2d[wp.int32],
    enable_friction: int,
    inv_dt: float,
    # outputs
    rigid_contact_force: wp.array[wp.vec3],
):
    "Convert solved FeatherPGS contact impulses into world-frame forces."
    _contact_linear_force(
        dense_first,
        body_base,
        routes,
        wp.tid(),
        world_impulses,
        world_constraint_count,
        mf_constraint_count,
        world_row_type,
        world_row_parent,
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


@wp.kernel
def build_mass_update_mask(
    global_flag: int,
    # in/out
    mass_update_requested: wp.array[int],
    # outputs
    mass_update_mask: wp.array[int],
):
    """Consume the requests into a step's mask: a request left set changes the factor cadence on subsequent captured
    steps."""
    tid = wp.tid()
    flag = 1 if global_flag != 0 else 0
    if mass_update_requested[tid] != 0:
        flag = 1
    mass_update_mask[tid] = flag
    mass_update_requested[tid] = 0


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


# =============================================================================
# Link rows
# =============================================================================
# A link ties a body's primary piece to a shadow (:mod:`.body_cut`): six bilateral rows, one law for every joint the
# body was cut at. The pieces share their center of mass, so the rows read ``twist(primary) - twist(shadow)`` there,
# three linear and three angular, with their pose error ``phi``. No row is scaled: the block's pivot decides rank.


@wp.func
def rotation_log(primary: wp.quat, shadow: wp.quat) -> wp.vec3:
    """``log(R_primary R_shadow^T)`` [rad], the rotation that takes the shadow to the primary, the short way round."""
    error = primary * wp.quat_inverse(shadow)
    v = wp.vec3(error[0], error[1], error[2])
    if error[3] < 0.0:
        return -2.0 * wp.atan2(wp.length(v), -error[3]) * wp.normalize(v)
    return 2.0 * wp.atan2(wp.length(v), error[3]) * wp.normalize(v)


@wp.func
def link_side(
    joints: JointTable,
    support: RowSupport,
    body: int,
    art: int,
    sign: float,
    point: wp.vec3,
    direction: wp.vec3,
    angular: bool,
    sides: wp.vec2i,
    body_to_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    articulation_origin: wp.array[wp.vec3],
    row: int,
    J: wp.array3d[float],
):
    """Accumulate one piece's twist into link row ``row``; a piece on no response coordinates adds nothing."""
    if support.dofs[art] > 0:
        accumulate_jacobian_row_world(
            body,
            sign,
            point,
            articulation_origin[art],
            direction,
            angular,
            body_to_joint,
            joint_ancestor,
            joints.joint_qd_start,
            joint_S_s,
            joints.articulation_dof_start[art],
            support.dofs[art],
            jacobian_row(support, 0, art, sides),
            row,
            J,
        )


@wp.kernel
def allocate_link_slots(
    links: Links,
    max_constraints: int,
    # outputs
    world_slot_counter: wp.array[int],
    first_rejected_slot: wp.array[int],
):
    """Reserve each enabled link's rows, whole or not at all."""
    k = wp.tid()
    links.slot[k] = -1
    if links.enabled[k] != 0:
        links.slot[k] = reserve(world_slot_counter, links.world[k], LINK_ROWS, max_constraints, first_rejected_slot)


@wp.kernel
def write_link_rows(
    joints: JointTable,
    links: Links,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_to_joint: wp.array[int],
    body_to_articulation: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    articulation_origin: wp.array[wp.vec3],
    pgs_beta: float,
    support: RowSupport,
    # outputs
    J: wp.array3d[float],
    rows: DenseRows,
):
    """Write each link's rows, on the sides of its two pieces' trees (:func:`.dense_rows.row_sides`), which may be one
    tree. The rows cover the sides whole, zeros included: the walks accumulate, so shared ancestors cancel."""
    k = wp.tid()
    base = links.slot[k]
    if base < 0:
        return
    primary, shadow = links.primary[k], links.shadow[k]
    point_p = wp.transform_point(body_q[primary], body_com[primary])
    point_s = wp.transform_point(body_q[shadow], body_com[shadow])
    gap = point_p - point_s
    turn = rotation_log(wp.transform_get_rotation(body_q[primary]), wp.transform_get_rotation(body_q[shadow]))
    art_p, art_s = body_to_articulation[primary], body_to_articulation[shadow]
    sides = row_sides(art_p, art_s, support.dofs)
    world = links.world[k]
    line = rows.first[world]
    for row in range(LINK_ROWS):
        slot = base + row
        axis = row % 3
        angular = row >= 3
        if support.walks != 0:
            support.units[line, slot] = unit_source(UNIT_LINK, LINK_ROWS * k + row)
        else:
            for side in range(2):
                if sides[side] >= 0:
                    clear_side(support, J, jacobian_row(support, 0, sides[side], sides), slot, sides[side])
            direction = wp.vec3(0.0, 0.0, 0.0)
            direction[axis] = 1.0
            link_side(
                joints,
                support,
                primary,
                art_p,
                1.0,
                point_p,
                direction,
                angular,
                sides,
                body_to_joint,
                joint_ancestor,
                joint_S_s,
                articulation_origin,
                slot,
                J,
            )
            link_side(
                joints,
                support,
                shadow,
                art_s,
                -1.0,
                point_s,
                direction,
                angular,
                sides,
                body_to_joint,
                joint_ancestor,
                joint_S_s,
                articulation_origin,
                slot,
                J,
            )
        rows.row_type[line, slot] = PGS_CONSTRAINT_TYPE_LINK
        rows.sides[line, slot] = sides
        rows.row_parent[line, slot] = -1
        rows.row_mu[line, slot] = 0.0
        rows.row_beta[line, slot] = pgs_beta
        rows.phi[line, slot] = turn[axis] if angular else gap[axis]
        rows.target[line, slot] = 0.0


# =============================================================================
# Multi-Articulation Contact Building Kernels
# =============================================================================
# These kernels enable contacts between multiple articulations within the same
# world. The constraint system becomes world-level instead of per-articulation.


@wp.func
def _admit_contact(
    c: int,
    data: ContactInput,
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
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
        data.shape0[c], data.shape1[c], data.shape_body, body_to_articulation, art_to_world
    )
    responds = contact_responds(art_a, body_a, data.response_dofs, body_has_response_dofs, body_flags)
    if not responds:
        responds = contact_responds(art_b, body_b, data.response_dofs, body_has_response_dofs, body_flags)
    phi = float(0.0)
    admitted = responds and world >= 0
    if admitted:
        _normal, _translation_a, _offset_a, _translation_b, _offset_b, _separation, phi = contact_points(
            data.normal[c],
            data.point0[c],
            data.point1[c],
            data.margin0[c],
            data.margin1[c],
            body_a,
            body_b,
            data.body_q,
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


@wp.kernel
def allocate_world_contact_slots(
    data: ContactInput,
    total_num_threads: int,
    body_to_articulation: wp.array[int],
    art_to_world: wp.array[int],
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
    row_capacity_telemetry: int,
    friction_patches: FrictionPatches,
    resolved_worlds: wp.array[int],
    # outputs
    world_slot_counter: wp.array[int],
    mf_slot_counter: wp.array[int],
    dense_dropped_contact_rows: wp.array[int],
    mf_dropped_contact_rows: wp.array[int],
    dense_first_rejected_slot: wp.array[int],
    mf_first_rejected_slot: wp.array[int],
    units: wp.array2d[int],
):
    """Classify and allocate the rows of every active contact, with work proportional to the materialized prefix.

    Where the one writer takes the rows (``units`` holds the placement's unit sources; empty otherwise), each dense
    row's unit source is written with its slot: the contact's rows are the writer's to fill.

    A contact admitted by :func:`_admit_contact` reserves 1 slot for the normal row and, when enabled below the
    friction gap threshold (and, under persistent patches, for a loaded anchor), 2 adjacent slots for the friction
    rows, and with contact torsion (``data.torsion``) a third for its spin row. Contacts where both sides are free rigid bodies (or ground) take matrix-free rows, the others world-level
    dense rows. A contact's slot is its first row in its world's line, the free bodies' rows from the dense capacity
    ``max_constraints``; the counters count each class's own rows. A positive contact gap gate excludes wider speculative contacts before any route reserves slots;
    zero disables the gate.

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

    In a world the resolved-world classifier skips, no contact gets rows. Under persistent patch friction, a
    contact the allocator would give rows there takes CONTACT_PATH_RESOLVED and no slot, so the patch finish
    releases its anchor, as the complete path releases an anchor whose rows solve zero load; every other contact
    keeps path -1, and its anchor history.
    """
    thread = wp.tid()
    # A collision pass clamps the count to the rows it wrote, all of them, and flags the overflow itself.
    total_contacts = data.count[0]

    for c in range(thread, total_contacts, total_num_threads):
        resolved = False
        if resolved_worlds.shape[0] > 0:
            _body_a, _body_b, art_a, art_b, world = contact_endpoints(
                data.shape0[c], data.shape1[c], data.shape_body, body_to_articulation, art_to_world
            )
            if world >= 0 and world < resolved_worlds.shape[0] and resolved_worlds[world] != 0:
                # Refresh every active mapping each solve. The original force
                # converter publishes zero before reading any row for this path.
                data.world[c] = world
                data.art0[c] = art_a
                data.art1[c] = art_b
                data.slot[c] = -1
                data.path[c] = -1
                data.slots_needed[c] = 0
                if friction_patches.enabled == 0:
                    continue
                resolved = True
        admitted, art_a, art_b, world, phi = _admit_contact(
            c,
            data,
            body_to_articulation,
            art_to_world,
            body_flags,
            body_has_response_dofs,
            is_free_rigid,
            contact_gap_gate,
            same_articulation_contact_gap_gate,
            articulation_pair_contact_gap_gate,
        )
        if resolved:
            if admitted:
                data.path[c] = CONTACT_PATH_RESOLVED
            continue
        if not admitted:
            data.slot[c] = -1
            data.path[c] = -1
            continue
        a_has_dofs = art_a >= 0 and data.response_dofs[art_a] > 0
        b_has_dofs = art_b >= 0 and data.response_dofs[art_b] > 0
        # Classify: MF path if both sides are free rigid or ground.
        is_mf = 0
        if has_free_rigid != 0:
            a_is_mf_compatible = not a_has_dofs or is_free_rigid[art_a] != 0
            b_is_mf_compatible = not b_has_dofs or is_free_rigid[art_b] != 0
            if a_is_mf_compatible and b_is_mf_compatible:
                is_mf = 1

        # Every normal survives; only the selected patch anchors receive tangent rows.
        slots_needed = 1
        add_friction = enable_friction != 0 and phi <= contact_friction_gap_threshold
        if friction_patches.enabled != 0:
            add_friction = add_friction and friction_patches.weight[c] > 0.0
        if add_friction:
            slots_needed = 3 + data.torsion
        data.slots_needed[c] = slots_needed

        # The matrix-free or the dense family's counter.
        slot = int(-1)
        if is_mf != 0:
            slot = reserve(mf_slot_counter, world, slots_needed, mf_max_constraints, mf_first_rejected_slot)
            if slot < 0 and row_capacity_telemetry != 0:
                wp.atomic_add(mf_dropped_contact_rows, world, slots_needed)
        else:
            slot = reserve(world_slot_counter, world, slots_needed, max_constraints, dense_first_rejected_slot)
            if slot < 0 and row_capacity_telemetry != 0:
                wp.atomic_add(dense_dropped_contact_rows, world, slots_needed)
        if slot < 0:
            data.slot[c] = -1
            data.path[c] = -1
            continue
        if is_mf != 0:
            # The free bodies' rows follow the dense class's in the world's line: one row index for both classes.
            slot += max_constraints
        data.world[c] = world
        data.slot[c] = slot
        data.art0[c] = art_a
        data.art1[c] = art_b
        data.path[c] = is_mf
        if is_mf == 0 and units.shape[0] > 0:
            for r in range(slots_needed):
                units[data.first[world], slot + r] = unit_source(UNIT_CONTACT, 4 * c + r)


# The allocator above reserves each contact's rows with an atomic on its world's counter, so on CUDA a
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
    body_base: int,
    global_lead: int,
    global_trail_start: int,
    lanes: int,
    dense_first: wp.array[int],
    # outputs
    run_first: wp.array2d[int],
    run_last: wp.array2d[int],
    run_rows: wp.array2d[wp.vec2i],
    contact_slot: wp.array[int],
    units: wp.array2d[int],
):
    """Place each world's recorded contact runs in block order and reset the records for the next step, with each
    placed dense row's unit source where ``units`` holds the one writer's (:func:`allocate_world_contact_slots`).

    Launched over ``(world_count, lanes)``: one complete warp per world on CUDA (``lanes`` 32),
    one thread on CPU (``lanes`` 1). Each family's runs start at the slot where the allocator's
    counter began, which the counter less the family's rows recovers (the free bodies' from ``body_base``); a family whose
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
    next_mf = body_base + mf_slot_counter[world] - total[1]
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
                if units.shape[0] > 0:
                    for r in range(dense):
                        units[dense_first[world], next_dense + dense_below + r] = unit_source(UNIT_CONTACT, 4 * c + r)
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
    spin: bool,
    body_to_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    art_dof_start: int,
    n_dofs: int,
    at: wp.vec2i,
    row: int,
    J_group: wp.array3d[float],
):
    """Accumulate Jacobian contributions by walking up the kinematic tree, into the side at ``at``
    (:func:`.dense_rows.jacobian_row`) of row ``row``: :func:`.contact_laws.jacobian_entry` along ``direction`` at
    ``point_world``, or a contact's spin row (``spin``) about ``direction`` (:func:`.contact_laws.spin_entry`)."""
    if body_index < 0:
        return

    curr_joint = body_to_joint[body_index]
    while curr_joint >= 0:
        for global_dof in range(joint_qd_start[curr_joint], joint_qd_start[curr_joint + 1]):
            local_dof = global_dof - art_dof_start
            if local_dof >= 0 and local_dof < n_dofs:
                entry = float(0.0)
                if spin:
                    entry = spin_entry(joint_S_s[global_dof], direction)
                else:
                    entry = jacobian_entry(joint_S_s[global_dof], direction, point_world, origin)
                J_group[at[0], row, at[1] + local_dof] += sign * entry
        curr_joint = joint_ancestor[curr_joint]


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


# =============================================================================
# World-Level PGS and Velocity Kernels for Multi-Articulation
# =============================================================================


@wp.struct
class RowBias:
    """The rows' law's step parameters (:func:`row_bias`): the step ``dt``, a positive gap's speculative scale, the
    contacts' weight ``contact_w``, the restitution threshold, whether rows store their regularization weight
    (``weighted``), and whether contacts carry a material (``material``: their writer stored their rhs and compliance).
    """

    dt: float
    speculative_scale: float
    contact_w: float
    restitution_threshold: float
    weighted: int
    material: int


@wp.func
def row_bias(rows: DenseRows, bias: RowBias, line: int, i: int, incident: float):
    """Row ``i``'s rhs by its row law (:func:`.row_laws.row_rhs`, its parameters ``bias``) from its records, and a
    bouncing contact's rebound from its incident velocity ``J v_hat`` (:func:`.row_laws.contact_restitution_rhs`,
    weight 1); a material contact (its writer stored its rhs and compliance) keeps its rhs. ``weighted`` rows store
    their regularization weight."""
    kind = rows.row_type[line, i]
    if bias.material != 0 and kind == PGS_CONSTRAINT_TYPE_CONTACT:
        if rows.row_eps[line, i] > 0.0:
            return
    inv_dt = 1.0 / bias.dt
    phi = rows.phi[line, i]
    target = rows.target[line, i]
    rhs, row_w = row_rhs(
        kind, phi, rows.row_beta[line, i], target, inv_dt, 1.0, bias.speculative_scale, 1.0, bias.contact_w
    )
    if kind == PGS_CONSTRAINT_TYPE_CONTACT:
        restitution = rows.restitution[line, i]
        if restitution > 0.0:
            fires, bounce = contact_restitution_rhs(
                phi, incident - target, target, restitution, bias.dt, bias.restitution_threshold
            )
            if fires:
                rhs = bounce
                row_w = 1.0
    rows.rhs[line, i] = rhs
    if bias.weighted != 0:
        rows.row_w[line, i] = row_w


@wp.kernel
def compute_world_contact_bias(
    rows: DenseRows,
    row_incident: wp.array2d[float],
    world_constraint_count: wp.array[int],
    max_constraints: int,
    base: int,
    lanes: int,
    bias: RowBias,
):
    """Compute each world row's RHS bias by its row law (:func:`.row_laws.row_rhs`); ``J*v`` is added later.

    The RHS follows the convention: rhs = J*v + stabilization; a penetrating contact's Baumgarte term
    ``beta * phi / dt`` is negative and leads to positive impulses when resolved by PGS. A restitutive contact that
    fires takes the rebound of :func:`.row_laws.contact_restitution_rhs` from its incident velocity instead (weight 1).

    ``weighted`` rows store their relative regularization weight. A material contact (``material``: its writer stored
    its rhs and compliance ``eps``, from which the sweeps form its weight, :func:`.sweep.row_weight`) keeps
    its rhs.

    A world's ``lanes`` threads stride over its live rows (``dim = (world_count, lanes)``): the grid follows the
    worlds, and a world's slots past its rows cost no thread beyond its lanes. The previous one-thread-per-world loop
    left the GPU idle (64-128 blocks) for 30-60 us per launch. A class's rows start at ``base`` of the store's line: 0
    the dense rows, ``body_base`` the free bodies' (their writers stored their diagonals and incident velocities).
    """
    world, lane = wp.tid()
    count = wp.min(world_constraint_count[world], max_constraints)
    if lane >= count:
        return
    line = rows.first[world]
    for i in range(base + lane, base + count, lanes):
        row_bias(rows, bias, line, i, row_incident[line, i])


@wp.kernel
def prepare_world_impulses(
    dense_first: wp.array[int],
    world_constraint_count: wp.array[int],
    max_constraints: int,
    warmstart: int,
    # in/out
    world_impulses: wp.array2d[float],
):
    """Cold-initialize dense rows before identity-gathering contact impulses.

    No current non-contact row family has a cross-step identity contract:
    limits are activation-allocated, while mimic and link rows can be enabled
    dynamically and allocated atomically. Warm start therefore clears the
    full capacity before the contact/friction identity gather; cold solves only
    need to clear the current active prefix.
    """
    world = wp.tid()
    line = dense_first[world]
    m = world_constraint_count[world]

    # Cold-started solves consume only the current active prefix. Rows outside
    # that prefix are ignored, and a row that becomes active on a later step is
    # cleared then as part of that step's prefix. Warm-started solves clear full
    # capacity so no inactive cache can survive a later layout growth.
    clear_count = max_constraints
    if warmstart == 0:
        clear_count = m

    for i in range(clear_count):
        world_impulses[line, i] = 0.0


@wp.kernel
def reset_world_warmstart_buffers(
    first: wp.array[int],
    world_mask: wp.array[wp.bool],
    impulses: wp.array2d[float],
    prev_impulses: wp.array2d[float],
    prev_row_type: wp.array2d[int],
    prev_row_parent: wp.array2d[int],
):
    """Clear persistent warm-start state for selected worlds: their lines of the row store, both classes' rows."""
    world = wp.tid()
    if world_mask and not world_mask[world]:
        return
    line = first[world]
    for row in range(impulses.shape[1]):
        impulses[line, row] = 0.0
    for row in range(prev_impulses.shape[1]):
        prev_impulses[line, row] = 0.0
    for row in range(prev_row_type.shape[1]):
        prev_row_type[line, row] = -1
    for row in range(prev_row_parent.shape[1]):
        prev_row_parent[line, row] = -1


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


# =============================================================================
# Matrix-Free PGS Kernels for Free Rigid Bodies
# =============================================================================
#
# A free body is a one-node tree: its six velocity coordinates (its center-of-mass velocity, then its angular
# velocity) have the 6x6 mass matrix ``H``, armature included, which the walk takes leaf-first (``P`` reverses the
# coordinates). The walk whitens a free-body row, ``Z`` per body, and the sweeps hold the change of the body's
# velocity in the walk's coordinates ``u``, the walk in of ``dv`` (``WALK_ENTER``; ``u . u = dv^T H dv``): a row's
# velocity term is ``Z^T u`` (``J dv``), its update ``u += Z dlambda``, and the walk down leaves ``dv``. Walk
# coordinate ``i`` of ``Z`` and ``u`` sits at position ``5 - i``, the physical coordinate it reverses
# (:func:`.response_operator.leaf_first`). The walk whitens the rows and leaves them (:mod:`.walk_rows`,
# :func:`.tree_factor.walk_leave_kernel`).


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
    """Transport the cached tangent frame and enforce the current contact cone: the disk, or with the contact's spin
    row (contact torsion) the elliptic cone of the pair and the spin (:func:`.projection_laws.ball`)."""
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
    spin = slot + CONTACT_SPIN_ROW
    if spin < count and row_type[world, spin] == PGS_CONSTRAINT_TYPE_TORSION and row_parent[world, spin] == slot:
        # The spin about the normal needs no frame transport: the pair and the spin project onto their cone.
        triple = wp.vec3(tangent[0], tangent[1], impulses[world, spin])
        if not wp.isfinite(magnitude) or not wp.isfinite(triple[2]) or normal_impulse <= 0.0:
            triple = wp.vec3(0.0)
        else:
            triple = ball(triple, row_mu[world, slot + 1], row_mu[world, spin], normal_impulse)
        impulses[world, slot + 1] = triple[0]
        impulses[world, slot + 2] = triple[1]
        impulses[world, spin] = triple[2]
        return
    if not wp.isfinite(magnitude) or radius <= 0.0:
        tangent = wp.vec2(0.0)
    else:
        tangent = disk(tangent, radius)
    impulses[world, slot + 1] = tangent[0]
    impulses[world, slot + 2] = tangent[1]


@wp.kernel
def gather_mf_warmstart(
    routes: ContactRoutes,
    mf: MfRows,
    match_index: wp.array[int],  # rigid_contact_match_index (sorted-current -> prev-sorted idx)
    prev_slot_sorted: wp.array[int],  # prev-sorted contact idx -> prev base MF slot (or -1)
    prev_mf_impulses: wp.array2d[float],
    prev_mf_row_type: wp.array2d[int],
    prev_mf_row_parent: wp.array2d[int],
    mf_constraint_count: wp.array[int],
    previous_normal: wp.array[wp.vec3],
    decay: float,
    dt_scale: float,
    mf_max_c: int,
    # in-out
    mf_impulses: wp.array2d[float],
):
    """Seed this step's MF impulse buffer from the previous step's converged
    impulses, matched by contact identity.

    Runs after ``allocate_world_contact_slots`` + ``walk_rows.write_free_body_rows`` (so
    ``contact_slot`` and the rows' types are populated for this step) and after
    the warm-start branch has zeroed their impulses (the row store's free bodies' rows). One thread per contact;
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
    if c >= routes.count[0]:
        return
    if routes.path[c] != 1:  # MF path only
        return
    new_slot = routes.slot[c]
    if new_slot < 0:
        return

    world = routes.world[c]
    # The free bodies' rows in the world's line: from ``mf.base``, the slots' and parents' row index.
    count = mf.base + mf_constraint_count[world]
    capacity = mf.base + mf_max_c
    if new_slot >= count:
        return
    mf_line = mf.first[world]

    mi = match_index[c]
    matched = mi >= 0
    prev_slot = int(-1)
    if matched:
        prev_slot = prev_slot_sorted[mi]

    # Normal row (offset 0): always present for an MF contact.
    if (
        matched
        and prev_slot >= 0
        and prev_slot < capacity
        and mf.row_type[mf_line, new_slot] == PGS_CONSTRAINT_TYPE_CONTACT
        and prev_mf_row_type[mf_line, prev_slot] == PGS_CONSTRAINT_TYPE_CONTACT
    ):
        mf_impulses[mf_line, new_slot] = decay * dt_scale * prev_mf_impulses[mf_line, prev_slot]
    # else: leave 0 (already memset)

    # Friction rows (offsets 1..2): only if THIS step allocated them here and
    # they belong to this contact's block (parent == base slot).
    for r in range(1, 3):
        new_r = new_slot + r
        if new_r < count and new_r < capacity:
            if (
                mf.row_type[mf_line, new_r] == PGS_CONSTRAINT_TYPE_FRICTION
                and mf.row_parent[mf_line, new_r] == new_slot
            ):
                prev_r = prev_slot + r
                if (
                    matched
                    and prev_slot >= 0
                    and prev_r < capacity
                    and prev_mf_row_type[mf_line, prev_r] == PGS_CONSTRAINT_TYPE_FRICTION
                    and prev_mf_row_parent[mf_line, prev_r] == prev_slot
                ):
                    mf_impulses[mf_line, new_r] = decay * dt_scale * prev_mf_impulses[mf_line, prev_r]
                # else: leave 0
    # The spin row (contact torsion): carried as the friction rows are.
    new_r = new_slot + CONTACT_SPIN_ROW
    prev_r = prev_slot + CONTACT_SPIN_ROW
    if (
        matched
        and prev_slot >= 0
        and new_r < count
        and new_r < capacity
        and prev_r < capacity
        and mf.row_type[mf_line, new_r] == PGS_CONSTRAINT_TYPE_TORSION
        and mf.row_parent[mf_line, new_r] == new_slot
        and prev_mf_row_type[mf_line, prev_r] == PGS_CONSTRAINT_TYPE_TORSION
        and prev_mf_row_parent[mf_line, prev_r] == prev_slot
    ):
        mf_impulses[mf_line, new_r] = decay * dt_scale * prev_mf_impulses[mf_line, prev_r]

    if mi >= 0 and prev_slot >= 0 and prev_slot < capacity:
        _transport_warmstart_contact(
            routes.normal[c],
            previous_normal[mi],
            mf_line,
            new_slot,
            count,
            mf.row_type,
            mf.row_parent,
            mf.row_mu,
            mf_impulses,
        )


@wp.kernel
def snapshot_contact_warmstart(
    routes: ContactRoutes,
    previous_dense_slot: wp.array[int],
    previous_mf_slot: wp.array[int],
    previous_normal: wp.array[wp.vec3],
):
    """Save all contact routes and live normals in one launch."""
    contact = wp.tid()
    active = contact < routes.count[0]
    path = int(-1)
    slot = int(-1)
    if active:
        path = routes.path[contact]
        slot = routes.slot[contact]
        previous_normal[contact] = routes.normal[contact]
    # One thread per ready row; an empty table is a path the solver does not take.
    if previous_dense_slot.shape[0] > 0:
        previous_dense_slot[contact] = wp.where(path == 0, slot, -1)
    if previous_mf_slot.shape[0] > 0:
        previous_mf_slot[contact] = wp.where(path == 1, slot, -1)


@wp.kernel
def gather_dense_warmstart(
    routes: ContactRoutes,
    rows: DenseRows,
    match_index: wp.array[int],  # rigid_contact_match_index (sorted-current -> prev-sorted idx)
    prev_slot_sorted: wp.array[int],  # prev-sorted contact idx -> prev dense normal row (or -1)
    prev_dense_impulses: wp.array2d[float],
    prev_dense_row_type: wp.array2d[int],
    prev_dense_row_parent: wp.array2d[int],
    world_constraint_count: wp.array[int],
    previous_normal: wp.array[wp.vec3],
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
    if c >= routes.count[0]:
        return
    if routes.path[c] != 0:  # dense path only
        return
    new_slot = routes.slot[c]
    if new_slot < 0:
        return

    world = routes.world[c]
    count = world_constraint_count[world]
    if new_slot >= count:
        return
    line = rows.first[world]

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
        and prev_dense_row_type[line, prev_slot] == PGS_CONSTRAINT_TYPE_CONTACT
    ):
        if rows.row_type[line, new_slot] == PGS_CONSTRAINT_TYPE_CONTACT:
            world_impulses[line, new_slot] = decay * dt_scale * prev_dense_impulses[line, prev_slot]

    # Friction rows (offsets 1, 2): only if THIS step allocated them here.
    for r in range(1, 3):
        new_r = new_slot + r
        if new_r < count and new_r < max_constraints:
            if rows.row_type[line, new_r] == PGS_CONSTRAINT_TYPE_FRICTION and rows.row_parent[line, new_r] == new_slot:
                prev_r = prev_slot + r
                if (
                    matched
                    and prev_slot >= 0
                    and prev_r < max_constraints
                    and prev_dense_row_type[line, prev_r] == PGS_CONSTRAINT_TYPE_FRICTION
                    and prev_dense_row_parent[line, prev_r] == prev_slot
                ):
                    world_impulses[line, new_r] = decay * dt_scale * prev_dense_impulses[line, prev_r]
    # The spin row (contact torsion): carried as the friction rows are.
    new_r = new_slot + CONTACT_SPIN_ROW
    prev_r = prev_slot + CONTACT_SPIN_ROW
    if (
        matched
        and prev_slot >= 0
        and new_r < count
        and new_r < max_constraints
        and prev_r < max_constraints
        and rows.row_type[line, new_r] == PGS_CONSTRAINT_TYPE_TORSION
        and rows.row_parent[line, new_r] == new_slot
        and prev_dense_row_type[line, prev_r] == PGS_CONSTRAINT_TYPE_TORSION
        and prev_dense_row_parent[line, prev_r] == prev_slot
    ):
        world_impulses[line, new_r] = decay * dt_scale * prev_dense_impulses[line, prev_r]

    if mi >= 0 and prev_slot >= 0 and prev_slot < max_constraints:
        _transport_warmstart_contact(
            routes.normal[c],
            previous_normal[mi],
            line,
            new_slot,
            count,
            rows.row_type,
            rows.row_parent,
            rows.row_mu,
            world_impulses,
        )


RIGID_SPEED_ROWS = wp.constant(2)
"""Speed rows of a limited free rigid body: the first bounds its linear speed, the second its angular speed."""


@wp.kernel
def allocate_rigid_velocity_limit_slots(
    bodies: FreeBodies,
    body_flags: wp.array[wp.int32],
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
    speed row (:func:`.walk_rows.populate_rigid_velocity_limit_rows`) that the sweeps
    project onto that ball, along the current velocity.

    ``velocity_limit_activation_fraction`` proximity-gates the allocation,
    as the joint velocity limits (:func:`.prefix_rows.velocity_limits`): a positive fraction
    reserves a row only when the speed read from ``joint_qd`` at the free
    root's DOFs satisfies ``|v| >= fraction * limit``. A fraction of ``0.0``
    reserves every finitely limited speed's row without reading ``joint_qd``.
    """
    candidate = wp.tid()
    body = bodies.free_rigid_body_indices[candidate]
    base = RIGID_SPEED_ROWS * candidate

    for row in range(RIGID_SPEED_ROWS):
        rigid_velocity_limit_slot[base + row] = -1

    art = bodies.body_to_articulation[body]
    if art < 0:
        return
    if bodies.is_free_rigid[art] == 0:
        return
    if (body_flags[body] & BodyFlags.KINEMATIC) != 0:
        return

    world = bodies.art_to_world[art]
    if world < 0:
        return
    if world < resolved_worlds.shape[0] and resolved_worlds[world] != 0:
        return

    for speed_axis in range(RIGID_SPEED_ROWS):
        limit = bodies.max_linear_velocity[body]
        if speed_axis == 1:
            limit = bodies.max_angular_velocity[body]
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


# =============================================================================
# Parallelized Non-Tiled Kernels for Heterogeneous Multi-Articulation
# =============================================================================
# These kernels parallelize across constraints (and constraint pairs) to achieve
# much better GPU utilization than the single-thread-per-articulation versions.


@wp.kernel
def classify_local_solve_worlds(
    mf: MfRows,
    owners: LocalOwners,
    world_constraint_count: wp.array[int],
    dense_phase_bounds: wp.array2d[int],
    mf_constraint_count: wp.array[int],
    articulation_dof_count: wp.array[int],
):
    """Assign exact solver ownership and compact active general worlds."""
    world = wp.tid()
    mf_line = mf.first[world]
    row_count = world_constraint_count[world]
    mf_count = mf_constraint_count[world]
    primary_articulation = owners.primary_articulation[world]
    pair_articulation = owners.pair_articulation[world]
    residual_pair_articulation = owners.residual_pair_articulation[world]
    single_phase = dense_phase_bounds[world, 1] == row_count

    local_mf = mf_count > 0 and mf_count <= owners.residual_mf_max_rows and residual_pair_articulation >= 0
    mf_row = int(0)
    while mf_row < mf_count and local_mf:
        sides = mf.sides[mf_line, mf.base + mf_row]
        if sides[0] >= 0 and sides[0] != residual_pair_articulation:
            local_mf = False
        if sides[1] >= 0 and sides[1] != residual_pair_articulation:
            local_mf = False
        # The residual local loop does not implement the speed law (projection_laws.speed_row): keep worlds with speed
        # rows on the general owner.
        if mf.row_type[mf_line, mf.base + mf_row] == PGS_CONSTRAINT_TYPE_SPEED:
            local_mf = False
        mf_row += 1

    owner = PGS_LOCAL_SOLVE_OWNER_GENERAL
    single_row_capacity = int(0)
    if primary_articulation >= 0:
        single_row_capacity = wp.min(owners.max_rows, articulation_dof_count[primary_articulation])
    if row_count > 0 and mf_count == 0:
        if single_phase and row_count <= single_row_capacity:
            owner = PGS_LOCAL_SOLVE_OWNER_SINGLE
        elif not single_phase and row_count <= owners.max_rows and pair_articulation >= 0:
            owner = PGS_LOCAL_SOLVE_OWNER_PAIR
    if owner == PGS_LOCAL_SOLVE_OWNER_GENERAL and (
        row_count > 0
        and row_count <= owners.residual_max_rows
        and residual_pair_articulation >= 0
        and ((mf_count == 0 and not single_phase) or local_mf)
    ):
        owner = PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL
    owners.owner[world] = owner
    if owner == PGS_LOCAL_SOLVE_OWNER_GENERAL and (row_count > 0 or mf_count > 0):
        general_index = wp.atomic_add(owners.general_world_count, 0, 1)
        owners.general_worlds[general_index] = world


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


# =============================================================================
# Tiled kernels for homogenous multi-articulation support
# =============================================================================
