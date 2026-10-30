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


import functools

import warp as wp

from ...sim import BodyFlags, JointType
from .contact_filters import contact_friction_eligible, contact_normal_gap_limit
from .contact_input import ContactInput, ContactRoutes, contact_frame
from .contact_laws import (
    CONTACT_SPIN_ROW,
    contact_endpoints,
    contact_friction_points,
    contact_geometry,
    contact_materials,
    contact_points,
    contact_responds,
    contact_row_frame,
    contact_tangent_basis,
    free_body_jacobian,
    free_body_spin_jacobian,
    jacobian_entry,
    prescribed_spin_target,
    prescribed_target,
    spin_entry,
)
from .dense_rows import DenseRows, MfRows, RowSupport, clear_side, jacobian_row, row_sides
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
    PatchRowLaw,
    patch_friction_location,
    patch_normal_load,
    patch_tangent_row,
)
from .joint_table import FreeBodies, JointTable
from .projection_laws import (
    ball,
    disk,
)
from .row_build import contact_torsion_mu, write_contact_row
from .row_laws import (
    contact_material_rhs,
    contact_restitution_rhs,
    joint_limit_side,
    limit_rhs,
    mimic_phi,
    row_rhs,
    velocity_limit_active,
)
from .solver_fields import Connects, LocalOwners, Mimics
from .spatial_laws import drive_force, drive_stiffness, free_root_transport, point_velocity
from .tree_passes import (
    DynamicsData,
    articulation_tau,
)

PGS_CONSTRAINT_TYPE_CONTACT = 0
# Type 1 was the PhysX drive row; drives fold into the mass matrix.
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
# joints; see ``forest_plan.ForestPlan.build``). Three rows per closure, world-axis
# directions, unbounded lambda, Baumgarte drift correction.
PGS_CONSTRAINT_TYPE_CONNECT = 6
PGS_CONSTRAINT_TYPE_TORSION = 7
PGS_CONSTRAINT_TYPE_COUNT = 8

# Keep launch-geometry-specific dynamics kernels out of the large general
# kernel module. Warp compiles one whole module variant per block dimension:
# the inverse-dynamics drives launch at tree_passes.SERIAL_BLOCK_DIM.
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
def dense_index(stride: int, i: int, j: int):
    return i * stride + j


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
    if global_dof >= 0 and fused_limit_dof_mask[global_dof] != 0 and joints.prescribed[global_dof] == 0:
        q_index = limit_q_index[global_dof]
        if q_index >= 0:
            q = joint_q[q_index]
            lower = joints.joint_limit_lower[global_dof]
            upper = joints.joint_limit_upper[global_dof]
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
    """Solve a structurally diagonal generalized mass system through its leaf-first pivots."""
    element = wp.tid()
    group = element // n_dofs
    dof = element - group * n_dofs
    art = group_to_art[group]
    global_dof = articulation_dof_start[art] + dof
    pivot = n_dofs - 1 - dof
    diagonal = L_group[group, pivot, pivot]
    value = joint_tau[global_dof] / diagonal
    joint_qdd[global_dof] = value / diagonal


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
    group = art_group_index[art]
    # The angular block of the leaf-first factor P H P^T = L L^T: physical coordinate 3 + r is factor row 2 - r, so
    # with ``lower[r] = L[2 - r]`` the block is ``lower lower^T``.
    lower = wp.mat33(0.0)
    for r in range(3):
        for c in range(3 - r):
            lower[r, c] = cholesky[group, 2 - r, c]
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
    routes: ContactRoutes,
    c: int,
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
    total_contacts = routes.count[0]
    if c >= total_contacts:
        return

    force = wp.vec3(0.0)
    slot = routes.slot[c]
    path = routes.path[c]

    if slot >= 0 and path >= 0 and inv_dt > 0.0:
        world = routes.world[c]
        # Contacts store normals from shape 0 toward shape 1 (A-to-B). FeatherPGS
        # solves along the opposite direction internally, which corresponds to the
        # force on shape/body 0 from shape/body 1.
        normal = -routes.normal[c]

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
    routes: ContactRoutes,
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
        routes,
        wp.tid(),
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
    # outputs
    row_counts: wp.array[int],
    row_dof_index: wp.array[int],
    row_K: wp.array[float],
    tau: wp.array[float],
):
    """Prepare one articulation's implicit drive rows and add their explicit force to ``tau``."""
    joint_start = articulation_start[articulation]
    joint_end = articulation_start[articulation + 1]
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

            K = drive_stiffness(ke, kd, dt)
            if K <= 0.0:
                continue

            u0 = drive_force(
                ke,
                kd,
                joint_q[coord_start + axis],
                joint_qd[dof_index],
                joint_target_pos[target_start + axis],
                joint_target_vel[dof_index],
                joint_effort_limit[dof_index],
                dt,
            )

            row_index = articulation * max_dofs + slot
            row_dof_index[row_index] = dof_index
            row_K[row_index] = K
            tau[dof_index] = tau[dof_index] + u0
            slot += 1
            if slot >= max_dofs:
                break

    row_counts[articulation] = slot


@wp.kernel(module=_INVERSE_DYNAMICS_KERNEL_MODULE)
def eval_rigid_tau_and_augmented_drives(
    d: DynamicsData,
    articulation_H_rows: wp.array[int],
    joint_target_q_start: wp.array[int],
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
):
    """An articulation's joint forces (:func:`.tree_passes.articulation_tau`) and its augmented drives in one
    launch; one thread per articulation."""
    articulation = wp.tid()
    articulation_tau(articulation, d)
    prepare_articulation_augmented_drives(
        articulation,
        d.articulation_start,
        articulation_H_rows,
        d.joint_type,
        d.joint_qd_start,
        d.joint_q_start,
        joint_target_q_start,
        d.joint_dof_dim,
        d.joint_q,
        d.joint_qd,
        joint_target_ke,
        joint_target_kd,
        joint_target_pos,
        joint_target_vel,
        joint_effort_limit,
        max_dofs,
        dt,
        row_counts,
        row_dof_index,
        d.row_K,
        d.tau,
    )


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
    joints: JointTable,
    joint_q: wp.array[float],
    joint_limit_activation_gap: float,
    group_to_art: wp.array[int],
    max_constraints: int,
    pgs_beta: float,
    pgs_cfm: float,
    resolved_worlds: wp.array[int],
    support: RowSupport,
    # outputs
    world_slot_counter: wp.array[int],
    first_rejected_slot: wp.array[int],
    J_group: wp.array3d[float],
    rows: DenseRows,
):
    """Allocate and populate active joint-limit rows in one articulation pass."""
    group_idx = wp.tid()
    art = group_to_art[group_idx]
    world = joints.art_to_world[art]
    if world >= 0 and world < resolved_worlds.shape[0] and resolved_worlds[world] != 0:
        return
    dof_start = joints.articulation_dof_start[art]

    for j in range(joints.articulation_start[art], joints.articulation_start[art + 1]):
        jtype = joints.joint_type[j]
        if jtype != JointType.PRISMATIC and jtype != JointType.REVOLUTE and jtype != JointType.D6:
            continue

        axis_count = joints.joint_dof_dim[j, 0] + joints.joint_dof_dim[j, 1]
        qd_start = joints.joint_qd_start[j]
        q_start = joints.joint_q_start[j]
        for axis in range(axis_count):
            dof = qd_start + axis
            if joints.prescribed[dof] != 0:
                continue
            q_val = joint_q[q_start + axis]
            lower = joints.joint_limit_lower[dof]
            upper = joints.joint_limit_upper[dof]

            local_dof = dof - dof_start

            for side in range(2):
                active, phi, sign = joint_limit_side(q_val, lower, upper, side, joint_limit_activation_gap)
                if not active:
                    continue

                slot = reserve(world_slot_counter, world, 1, max_constraints, first_rejected_slot)
                if slot < 0:
                    continue

                at = jacobian_row(support, group_idx, art, wp.vec2i(art, -1))
                clear_side(support, J_group, at, slot, art)
                J_group[at[0], slot, at[1] + local_dof] = sign
                rows.row_type[world, slot] = PGS_CONSTRAINT_TYPE_JOINT_LIMIT
                rows.sides[world, slot] = wp.vec2i(art, -1)
                rows.row_parent[world, slot] = -1
                rows.row_mu[world, slot] = 0.0
                rows.row_beta[world, slot] = pgs_beta
                rows.row_cfm[world, slot] = pgs_cfm
                rows.phi[world, slot] = phi
                rows.target[world, slot] = 0.0


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
    mimics: Mimics,
    max_constraints: int,
    # outputs
    world_slot_counter: wp.array[int],
    first_rejected_slot: wp.array[int],
):
    """Allocate one dense constraint slot per enabled, valid mimic constraint.

    Launched with ``dim = constraint_mimic_count``. Disabled or invalid mimics
    get ``mimic_slot = -1`` and consume no slot.
    """
    k = wp.tid()
    mimics.slot[k] = -1
    if mimics.valid[k] == 0:
        return
    if not mimics.enabled[k]:
        return
    mimics.slot[k] = reserve(world_slot_counter, mimics.world[k], 1, max_constraints, first_rejected_slot)


@wp.kernel
def populate_mimic_J_for_size(
    mimics: Mimics,
    joints: JointTable,
    group_to_art: wp.array[int],
    joint_q: wp.array[float],
    pgs_beta: float,
    pgs_cfm: float,
    support: RowSupport,
    # outputs
    J_group: wp.array3d[float],
    rows: DenseRows,
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
    world = joints.art_to_world[art]
    dof_start = joints.articulation_dof_start[art]

    for m in range(mimics.art_start[art], mimics.art_start[art + 1]):
        k = mimics.art_list[m]
        slot = mimics.slot[k]
        if slot < 0:
            continue

        c0 = mimics.coef0[k]
        c1 = mimics.coef1[k]

        # Jacobian: +1 on the follower DOF, -coef1 on the leader DOF. The two
        # DOFs are guaranteed distinct by the host-side validity mask.
        at = jacobian_row(support, group_idx, art, wp.vec2i(art, -1))
        clear_side(support, J_group, at, slot, art)
        J_group[at[0], slot, at[1] + mimics.dof0[k] - dof_start] = 1.0
        J_group[at[0], slot, at[1] + mimics.dof1[k] - dof_start] = -c1

        rows.row_type[world, slot] = PGS_CONSTRAINT_TYPE_MIMIC
        rows.sides[world, slot] = wp.vec2i(art, -1)
        rows.row_parent[world, slot] = -1
        rows.row_mu[world, slot] = 0.0
        rows.row_beta[world, slot] = pgs_beta
        rows.row_cfm[world, slot] = pgs_cfm
        rows.phi[world, slot] = mimic_phi(joint_q[mimics.q0[k]], joint_q[mimics.q1[k]], c0, c1)
        rows.target[world, slot] = 0.0


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
    connects: Connects,
    max_constraints: int,
    # outputs
    world_slot_counter: wp.array[int],
    first_rejected_slot: wp.array[int],
):
    """Allocate three consecutive dense slots per enabled, valid loop closure.

    Launched with ``dim = n_connect``. If the buffer cannot hold all three rows the
    closure is dropped whole (``connect_slot = -1``) rather than partially enforced.
    """
    k = wp.tid()
    connects.slot[k] = -1
    if connects.valid[k] == 0 or connects.enabled[k] == 0:
        return
    connects.slot[k] = reserve(world_slot_counter, connects.world[k], 3, max_constraints, first_rejected_slot)


@wp.kernel
def populate_connect_J_for_size(
    joints: JointTable,
    connects: Connects,
    group_to_art: wp.array[int],
    n_dofs: int,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_to_joint: wp.array[int],
    body_to_articulation: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    articulation_origin: wp.array[wp.vec3],
    pgs_beta: float,
    pgs_cfm: float,
    support: RowSupport,
    # outputs
    J_group: wp.array3d[float],
    rows: DenseRows,
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
    world = joints.art_to_world[art]
    dof_start = joints.articulation_dof_start[art]

    n_connect = connects.art.shape[0]
    for k in range(n_connect):
        if connects.art[k] != art:
            continue
        base_slot = connects.slot[k]
        if base_slot < 0:
            continue

        body_p = connects.body_p[k]
        body_c = connects.body_c[k]
        prescribed = connects.parent_prescribed[k] != 0
        p_a = connects.anchor_p[k]
        v_a = wp.vec3(0.0, 0.0, 0.0)
        if body_p >= 0:
            p_a = wp.transform_point(body_q[body_p], connects.anchor_p[k])
            if prescribed:
                # Prescribed parent: the anchor velocity comes from the body twist
                # (v_com_world, omega_world) rather than from tree DOFs.
                x_com = wp.transform_point(body_q[body_p], body_com[body_p])
                v_a = point_velocity(body_qd[body_p], p_a - x_com)
        p_b = wp.transform_point(body_q[body_c], connects.anchor_c[k])
        origin = articulation_origin[art]
        delta = p_a - p_b
        at = jacobian_row(support, group_idx, art, wp.vec2i(art, -1))

        for axis in range(3):
            slot = base_slot + axis
            # The two ancestor walks below ACCUMULATE (shared ancestors partially cancel),
            # so the row must start from zero regardless of buffer reuse policy.
            for d in range(n_dofs):
                J_group[at[0], slot, at[1] + d] = 0.0
            e = wp.vec3(0.0, 0.0, 0.0)
            e[axis] = 1.0
            target = float(0.0)

            # J = e . (Jpoint(parent, p_a) - Jpoint(child, p_b)): ancestor walks with
            # opposite signs; shared ancestors partially cancel automatically.
            if prescribed:
                target = -wp.dot(e, v_a)
            else:
                accumulate_jacobian_row_world(
                    body_p,
                    1.0,
                    p_a,
                    origin,
                    e,
                    False,
                    body_to_joint,
                    joint_ancestor,
                    joints.joint_qd_start,
                    joint_S_s,
                    dof_start,
                    n_dofs,
                    at,
                    slot,
                    J_group,
                )
            accumulate_jacobian_row_world(
                body_c,
                -1.0,
                p_b,
                origin,
                e,
                False,
                body_to_joint,
                joint_ancestor,
                joints.joint_qd_start,
                joint_S_s,
                dof_start,
                n_dofs,
                at,
                slot,
                J_group,
            )

            # Normalize the row: on planar linkages one axis is near-redundant with the
            # tree, leaving ||J|| ~ 1e-4 — its Delassus diagonal collapses to pure CFM
            # while H^-1 J^T stays large against small link inertias, and PGS then relaxes
            # a real velocity effect against a near-zero denominator (measured to NaN in
            # one sweep). Normalizing keeps the identical constraint with a well-scaled
            # diagonal; truly degenerate rows are left inert (zero J, zero phi).
            norm_sq = float(0.0)
            for d in range(n_dofs):
                norm_sq += J_group[at[0], slot, at[1] + d] * J_group[at[0], slot, at[1] + d]
            phi_axis = delta[axis]
            if norm_sq > 1.0e-8:
                inv_norm = 1.0 / wp.sqrt(norm_sq)
                for d in range(n_dofs):
                    J_group[at[0], slot, at[1] + d] *= inv_norm
                phi_axis *= inv_norm
                target *= inv_norm
            else:
                for d in range(n_dofs):
                    J_group[at[0], slot, at[1] + d] = 0.0
                phi_axis = 0.0
                target = 0.0

            rows.row_type[world, slot] = PGS_CONSTRAINT_TYPE_CONNECT
            rows.sides[world, slot] = wp.vec2i(art, -1)
            rows.row_parent[world, slot] = -1
            rows.row_mu[world, slot] = 0.0
            rows.row_beta[world, slot] = pgs_beta
            rows.row_cfm[world, slot] = pgs_cfm
            rows.phi[world, slot] = phi_axis
            rows.target[world, slot] = target


# =============================================================================
# Bilateral elimination: the block kit
# =============================================================================
# The factor and register types of response_builds_rows.elimination, which
# eliminates each articulation's bilateral rows (MIMIC and CONNECT) from the
# response operator, sized by the model's largest block.


@wp.func
def pivoted_cholesky(n: int, A: wp.array[float], A_start: int, floor: float, L: wp.array[float]) -> int:
    """Factor the ``n x n`` block ``A`` (row-major at ``A_start``) in row order, dropping each row whose pivot falls to
    ``floor`` or below: such a row depends on the rows kept before it. ``L`` (the same layout) holds the factor over
    the kept rows; a dropped row's column, its diagonal included, is zero. Returns the kept count."""
    kept = int(0)
    for j in range(n):
        s = A[A_start + dense_index(n, j, j)]
        for k in range(j):
            r = L[A_start + dense_index(n, j, k)]
            s -= r * r
        if s > floor:
            d = wp.sqrt(s)
            L[A_start + dense_index(n, j, j)] = d
            for i in range(j + 1, n):
                acc = A[A_start + dense_index(n, i, j)]
                for k in range(j):
                    acc -= L[A_start + dense_index(n, i, k)] * L[A_start + dense_index(n, j, k)]
                L[A_start + dense_index(n, i, j)] = acc / d
            kept += 1
        else:
            for i in range(j, n):
                L[A_start + dense_index(n, i, j)] = 0.0
    return kept


@functools.cache
def block_kit(rows: int):
    """The register types of a bilateral block of up to ``rows`` rows, its values and its row indices, and the solve
    ``(L L^T) x = b`` of its kept factor (row-major, stride ``n``) with the right-hand side in registers."""
    vec = wp.types.vector(length=rows, dtype=wp.float32)
    ivec = wp.types.vector(length=rows, dtype=wp.int32)

    def solve(n: int, L: wp.array[float], L_start: int, b: vec) -> vec:
        x = vec()
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

    solve.__name__ = solve.__qualname__ = f"block_solve_{rows}"
    return vec, ivec, wp.func(solve)


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
    joints: JointTable,
    joint_qd: wp.array[float],
    velocity_limit_activation_fraction: float,
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

    The sweeps project these rows as unilateral rows with a non-negative
    total (:func:`.projection_laws.velocity_limit_update`): a violated side
    accumulates the impulse that holds it on its limit, and a satisfied side
    releases its impulse.

    ``velocity_limit_activation_fraction`` proximity-gates the allocation:
    with a positive fraction the lower/upper pair is reserved only when
    ``|joint_qd[dof]| >= fraction * qdot_max``, i.e. the DOF is close enough
    to the velocity box edge that the rows could act
    (:func:`.row_laws.velocity_limit_active`). A fraction of ``0.0`` disables
    the gate: every limited DOF takes its pair. Because the gate samples the pre-solve
    velocity, a DOF that crosses the threshold during a step is clamped one
    step late.

    Outputs two entries per DOF in ``velocity_limit_slot`` (world-constraint
    row, or -1) and ``velocity_limit_sign`` (+1 / -1).
    """
    art = wp.tid()
    world = joints.art_to_world[art]

    # Initialize all DOFs of this articulation to "no limit active"
    dof_base = joints.articulation_dof_start[art]
    dof_count = joints.articulation_H_rows[art]
    for d in range(dof_count):
        lower_idx = 2 * (dof_base + d)
        upper_idx = lower_idx + 1
        velocity_limit_slot[lower_idx] = -1
        velocity_limit_slot[upper_idx] = -1
        velocity_limit_sign[lower_idx] = 0.0
        velocity_limit_sign[upper_idx] = 0.0

    joint_start = joints.articulation_start[art]
    joint_end = joints.articulation_start[art + 1]
    n_eligible = int(0)

    for j in range(joint_start, joint_end):
        jtype = joints.joint_type[j]
        if jtype != JointType.PRISMATIC and jtype != JointType.REVOLUTE and jtype != JointType.D6:
            continue

        lin_count = joints.joint_dof_dim[j, 0]
        ang_count = joints.joint_dof_dim[j, 1]
        axis_count = lin_count + ang_count
        qd_start = joints.joint_qd_start[j]

        for axis in range(axis_count):
            dof = qd_start + axis
            # A non-positive limit is unlimited (PhysX pins ``recipResponse`` off for ``unitResponse <= 0``);
            # a positive fraction admits only a DOF whose speed reaches that fraction of the limit.
            if joints.prescribed[dof] != 0 or not velocity_limit_active(
                joints.joint_velocity_limit[dof], joint_qd[dof], velocity_limit_activation_fraction
            ):
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
    joints: JointTable,
    velocity_limit_slot: wp.array[int],
    velocity_limit_sign: wp.array[float],
    group_to_art: wp.array[int],
    pgs_cfm: float,
    articulation_dof_count_of_size: int,
    support: RowSupport,
    # outputs
    J_group: wp.array3d[float],
    rows: DenseRows,
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

    DOFs skipped by :func:`allocate_joint_velocity_limit_slots` (the activation
    gate, a prescribed coordinate) carry ``velocity_limit_slot == -1`` and are skipped here through the same
    per-row slot check; this kernel needs no separate skip flag.
    """
    tid = wp.tid()
    n_dofs = articulation_dof_count_of_size
    side = tid % 2
    local_dof = (tid // 2) % n_dofs
    group_idx = tid // (2 * n_dofs)
    art = group_to_art[group_idx]
    world = joints.art_to_world[art]
    dof = joints.articulation_dof_start[art] + local_dof
    row_idx = 2 * dof + side
    slot = velocity_limit_slot[row_idx]
    if slot < 0:
        return
    qdot_max = joints.joint_velocity_limit[dof]
    sign = velocity_limit_sign[row_idx]

    at = jacobian_row(support, group_idx, art, wp.vec2i(art, -1))
    clear_side(support, J_group, at, slot, art)
    J_group[at[0], slot, at[1] + local_dof] = sign

    rows.row_type[world, slot] = PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT
    rows.sides[world, slot] = wp.vec2i(art, -1)
    rows.row_parent[world, slot] = -1
    rows.row_mu[world, slot] = 0.0
    rows.row_beta[world, slot] = 0.0
    rows.row_cfm[world, slot] = pgs_cfm
    rows.phi[world, slot] = 0.0
    rows.target[world, slot] = -qdot_max


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
        _normal, _point_a, _point_b, phi = contact_points(
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
    contact_friction_articulation_pairs_only: int,
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
):
    """Classify and allocate the rows of every active contact, with work proportional to the materialized prefix.

    A contact admitted by :func:`_admit_contact` reserves 1 slot for the normal row and, when enabled below the
    friction gap threshold (and, under persistent patches, for a loaded anchor), 2 adjacent slots for the friction
    rows, and with contact torsion (``data.torsion``) a third for its spin row. Contacts where both sides are free rigid bodies (or ground) take matrix-free rows, the others world-level
    dense rows. A positive contact gap gate excludes wider speculative contacts before any route reserves slots;
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
        data.world[c] = world
        data.slot[c] = slot
        data.art0[c] = art_a
        data.art1[c] = art_b
        data.path[c] = is_mf


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
def prepare_world_contact_rows(data: ContactInput, total_num_threads: int, law: PatchRowLaw):
    """Build current metadata inside each allocator-recorded contact row reservation (:func:`.row_build.write_contact_row`)."""
    total_contacts = data.count[0]
    for c in range(wp.tid(), total_contacts, total_num_threads):
        if data.path[c] != 0 or data.slot[c] < 0:
            continue
        # Allocation owns the row extent; recomputing eligibility can cross a
        # floating-point threshold and overwrite the following contact.
        rows = 1
        if data.slots_needed[c] >= 3:
            rows = data.slots_needed[c]
        for row in range(rows):
            write_contact_row(data, law, c, row, False)


@wp.kernel
def populate_world_J_for_compact_size(
    data: ContactInput,
    total_num_workers: int,
    target_size: int,
    art_group_idx: wp.array[int],
    law: PatchRowLaw,
    support: RowSupport,
    # output
    J_group: wp.array3d[float],
):
    """Project one dense contact row/DOF per lane for compact articulations: every coordinate of each side of the
    size, zeros included. A normal's lanes also write its spin row (contact torsion), which shares its direction."""
    worker, lane = wp.tid()
    total_contacts = data.count[0]
    row = lane // target_size
    local_dof = lane - row * target_size
    if row >= 3 or local_dof >= target_size:
        return

    for c in range(worker, total_contacts, total_num_workers):
        if data.path[c] != 0 or data.slot[c] < 0 or row >= data.slots_needed[c]:
            continue
        body_a, body_b, normal, point_a, point_b, _phi = contact_geometry(
            c,
            data.shape0,
            data.shape1,
            data.point0,
            data.point1,
            data.normal,
            data.margin0,
            data.margin1,
            data.shape_body,
            data.body_q,
        )
        # A normal-row lane needs no tangent basis (contact_frame computes it for every row).
        tangent0 = wp.vec3(0.0)
        tangent1 = wp.vec3(0.0)
        if row > 0:
            tangent0, tangent1 = contact_tangent_basis(normal)
        friction_a, friction_b = contact_friction_points(
            point_a, point_b, data.shared_anchor, data.friction_shared_anchor
        )
        if law.patches.enabled != 0:
            friction_a, friction_b = patch_friction_location(law.patches, c)
        direction, p0, p1 = contact_row_frame(
            row, data.shared_anchor, normal, tangent0, tangent1, point_a, point_b, friction_a, friction_b
        )

        art_a = data.art0[c]
        art_b = data.art1[c]
        group_a = -1
        group_b = -1
        value_a = float(0.0)
        value_b = float(0.0)
        spin_a = float(0.0)
        spin_b = float(0.0)
        spin = row == 0 and data.slots_needed[c] > CONTACT_SPIN_ROW
        bit = wp.uint32(1) << wp.uint32(local_dof)
        if art_a >= 0 and data.response_dofs[art_a] == target_size:
            group_a = art_group_idx[art_a]
            if body_a >= 0 and (data.body_mask[body_a] & bit) != wp.uint32(0):
                motion_a = data.motion[data.dof_start[art_a] + local_dof]
                value_a = jacobian_entry(motion_a, direction, p0, data.origin[art_a])
                if spin:
                    spin_a = spin_entry(motion_a, normal)
        if art_b >= 0 and data.response_dofs[art_b] == target_size:
            group_b = art_group_idx[art_b]
            if body_b >= 0 and (data.body_mask[body_b] & bit) != wp.uint32(0):
                motion_b = data.motion[data.dof_start[art_b] + local_dof]
                value_b = -jacobian_entry(motion_b, direction, p1, data.origin[art_b])
                if spin:
                    spin_b = -spin_entry(motion_b, normal)

        slot = data.slot[c] + row
        sides = row_sides(art_a, art_b, data.response_dofs)
        if group_a >= 0:
            at = jacobian_row(support, group_a, art_a, sides)
            if group_b == group_a:
                J_group[at[0], slot, at[1] + local_dof] = value_a + value_b
                if spin:
                    J_group[at[0], slot + CONTACT_SPIN_ROW, at[1] + local_dof] = spin_a + spin_b
            else:
                J_group[at[0], slot, at[1] + local_dof] = value_a
                if spin:
                    J_group[at[0], slot + CONTACT_SPIN_ROW, at[1] + local_dof] = spin_a
        if group_b >= 0 and group_b != group_a:
            at = jacobian_row(support, group_b, art_b, sides)
            J_group[at[0], slot, at[1] + local_dof] = value_b
            if spin:
                J_group[at[0], slot + CONTACT_SPIN_ROW, at[1] + local_dof] = spin_b


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


@wp.func
def sparse_diagonal_row_dot(
    world: int,
    row: int,
    velocity: wp.array[float],
    world_dof_indices: wp.array2d[int],
    dense_offsets: wp.array[int],
    dense_groups: wp.array[int],
    dense_dofs: int,
    dense_L: wp.array3d[float],
    dense_Z: wp.array3d[float],
    sparse_row_dof: wp.array3d[int],
    sparse_row_z: wp.array3d[float],
    diagonal_inverse_mass: wp.array[float],
):
    """``J v`` of one sparse-diagonal row with a global velocity state, from its whitened entries: ``Z^T u``, the
    velocity entered on the fly, ``u = L^T P v`` on the compact dense part (factor coordinate ``k`` at position
    ``dense_dofs - 1 - k``, each coordinate's terms from zero, ascending) and ``u = v / s`` on a slot's coordinate
    (``s = m^-1/2``)."""
    out = float(0.0)
    dense_offset = dense_offsets[world]
    group = dense_groups[world]
    for k in range(dense_dofs):
        u = float(0.0)
        for j in range(k, dense_dofs):
            global_dof = world_dof_indices[world, dense_offset + dense_dofs - 1 - j]
            if global_dof >= 0:
                u += dense_L[group, j, k] * velocity[global_dof]
        out += dense_Z[group, row, dense_dofs - 1 - k] * u
    for sparse_slot in range(2):
        world_dof = sparse_row_dof[world, row, sparse_slot]
        if world_dof >= 0:
            global_dof = world_dof_indices[world, world_dof]
            if global_dof >= 0:
                scale = wp.sqrt(diagonal_inverse_mass[global_dof])
                if scale > 0.0:
                    out += sparse_row_z[world, row, sparse_slot] * (velocity[global_dof] / scale)
    return out


@wp.func
def world_contact_row_dot(
    world_dof_indices: wp.array2d[int],
    support: RowSupport,
    packed_J: wp.array3d[float],
    velocity: wp.array[float],
    world: int,
    i: int,
):
    """``J_i v`` of one dense row with a global velocity state: its packed Jacobian entries (:mod:`.packed_rows`)
    times the velocity at their world coordinates, side by side in entry order from zero (a world's coordinates
    follow its articulations' order, so this is the rows' entry order)."""
    out = float(0.0)
    sides = support.sides[world, i]
    first = int(0)
    for side in range(2):
        art = sides[side]
        if art >= 0:
            n = support.dofs[art]
            offset = support.offset[art]
            at = first
            if support.dense != 0:
                at = offset
            for e in range(n):
                out += packed_J[world, i, at + e] * velocity[world_dof_indices[world, offset + e]]
            first += n
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Free bodies in factor coordinates
# ---------------------------------------------------------------------------------------------------------------------
#
# A free body is a one-node tree: its six velocity coordinates (its center-of-mass velocity, then its angular
# velocity) have the 6x6 mass matrix ``H`` the dynamics factor leaf-first, ``P H P^T = L L^T`` with ``P`` the
# coordinates reversed, armature included (``L_by_size[6]`` at its articulation's group index, lower, row-major). A
# free-body row is whitened, ``Z = L^-1 P J^T`` per body, and the sweeps hold the body's velocity in factor
# coordinates ``u = L^T P v``: a row's velocity term is ``Z^T u``, its update ``u += Z dlambda``, and ``v = P^T L^-T
# u`` leaves them. Factor coordinate ``i`` of ``Z`` and ``u`` sits at position ``5 - i``, the physical coordinate it
# reverses (:func:`.response_operator.leaf_first`). The factor's three actions below take
# :class:`.response_operator.CholeskySolve`'s orders and its zero-diagonal convention (a zero pivot gives zero),
# which the native sweeps emit.


@wp.struct
class BodyFactors:
    """Where each free body's held factor and velocity coordinates live: its articulation, the articulation's index
    in ``L_by_size[6]`` (``factor_index``, the solver's ``art_group_idx``) and first coordinate (``dof_start``), the
    factor index of the free body whose coordinates start at each global coordinate (``dof_factor``, -1 elsewhere:
    the native sweeps' lookup), and the size-6 factors ``L``.

    Every articulation's factor too, for the sweeps that hold a world's articulations in factor coordinates
    (:class:`.sweep_routes.FactorCoordinates`): the articulation whose coordinates start at each global coordinate
    (``first_art``, -1 elsewhere), and each articulation's coordinates (``size``), first world coordinate
    (``world_offset``) and the address of its row-major leaf-first factor (``address``: ``L_by_size[size]`` at its
    group index, 0 without a dense factor)."""

    body_to_articulation: wp.array[int]
    factor_index: wp.array[int]
    dof_start: wp.array[int]
    dof_factor: wp.array[int]
    L: wp.array3d[float]
    first_art: wp.array[int]
    size: wp.array[int]
    world_offset: wp.array[int]
    address: wp.array[wp.uint64]


@wp.func
def free_body_factor(body: int, factors: BodyFactors) -> int:
    """The index of ``body``'s held factor in ``factors.L``, -1 for no body."""
    if body < 0:
        return -1
    return factors.factor_index[factors.body_to_articulation[body]]


@wp.func
def factor_whiten(L: wp.array3d[float], factor: int, x: wp.spatial_vector) -> wp.spatial_vector:
    """``z = L^-1 P x``: rows ascending, each ``x`` less the earlier rows' terms in order, over its pivot."""
    z = wp.spatial_vector()
    for row in range(6):
        value = x[5 - row]
        for k in range(row):
            value -= L[factor, row, k] * z[5 - k]
        pivot = L[factor, row, row]
        z[5 - row] = wp.where(pivot != 0.0, value / pivot, 0.0)
    return z


@wp.func
def factor_enter(L: wp.array3d[float], factor: int, v: wp.spatial_vector) -> wp.spatial_vector:
    """``u = L^T P v``: each coordinate's terms from zero, ascending."""
    u = wp.spatial_vector()
    for row in range(6):
        value = float(0.0)
        for k in range(row, 6):
            value += L[factor, k, row] * v[5 - k]
        u[5 - row] = value
    return u


@wp.func
def factor_leave(L: wp.array3d[float], factor: int, u: wp.spatial_vector) -> wp.spatial_vector:
    """``v = P^T L^-T u``: rows descending, each ``u`` less the later rows' terms in order, over its pivot."""
    v = wp.spatial_vector()
    for step in range(6):
        row = 5 - step
        value = u[5 - row]
        for k in range(row + 1, 6):
            value -= L[factor, k, row] * v[5 - k]
        pivot = L[factor, row, row]
        v[5 - row] = wp.where(pivot != 0.0, value / pivot, 0.0)
    return v


@wp.func
def free_body_velocity(body: int, factors: BodyFactors, velocity: wp.array[float]) -> wp.spatial_vector:
    """``body``'s six coordinates of a velocity in global coordinates."""
    start = factors.dof_start[factors.body_to_articulation[body]]
    return wp.spatial_vector(
        velocity[start],
        velocity[start + 1],
        velocity[start + 2],
        velocity[start + 3],
        velocity[start + 4],
        velocity[start + 5],
    )


@wp.func
def free_body_side_velocity(
    Z: wp.array3d[float], world: int, row: int, body: int, factors: BodyFactors, velocity: wp.array[float]
) -> float:
    """``Z_side^T (L^T v)``: one side of a whitened row over a physical velocity (zero for no body)."""
    if body < 0:
        return 0.0
    u = factor_enter(factors.L, free_body_factor(body, factors), free_body_velocity(body, factors, velocity))
    out = float(0.0)
    for k in range(6):
        out += Z[world, row, k] * u[k]
    return out


@wp.func
def free_body_row_velocity(
    Z_a: wp.array3d[float],
    Z_b: wp.array3d[float],
    world: int,
    row: int,
    body_a: int,
    body_b: int,
    factors: BodyFactors,
    velocity: wp.array[float],
) -> float:
    """``J_row v = Z^T u`` of a whitened row over a physical velocity: body ``a``'s side, then ``b``'s."""
    out = free_body_side_velocity(Z_a, world, row, body_a, factors, velocity)
    out += free_body_side_velocity(Z_b, world, row, body_b, factors, velocity)
    return out


@wp.kernel
def populate_world_J_for_size(
    data: ContactInput,
    total_num_threads: int,
    target_size: int,
    art_group_idx: wp.array[int],
    body_to_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    law: PatchRowLaw,
    support: RowSupport,
    # outputs
    J_group: wp.array3d[float],
):
    """Populate active dense contacts for one articulation-size group with a capacity-independent launch.

    Each contact may contribute to multiple articulations' Jacobian matrices: each endpoint of the size walks its
    kinematic chain into its side of the row, cleared first (:func:`.dense_rows.clear_side`; a size group's rows are
    pre-cleared). Contacts routed to the matrix-free path (contact_path==1) are skipped.
    """
    total_contacts = data.count[0]
    for c in range(wp.tid(), total_contacts, total_num_threads):
        slot = data.slot[c]
        if data.path[c] != 0 or slot < 0:
            continue
        art_a = data.art0[c]
        art_b = data.art1[c]
        a_matches = art_a >= 0 and data.response_dofs[art_a] == target_size
        b_matches = art_b >= 0 and data.response_dofs[art_b] == target_size
        if not a_matches and not b_matches:
            continue
        # Allocation owns the row extent; recomputing eligibility can cross a
        # floating-point threshold and overwrite the following contact.
        rows = 1
        if data.slots_needed[c] >= 3:
            rows = data.slots_needed[c]
        sides = row_sides(art_a, art_b, data.response_dofs)
        at_a = wp.vec2i(0, 0)
        at_b = wp.vec2i(0, 0)
        if a_matches:
            at_a = jacobian_row(support, art_group_idx[art_a], art_a, sides)
        if b_matches:
            at_b = jacobian_row(support, art_group_idx[art_b], art_b, sides)
        for row in range(rows):
            # The row metadata, from the size group of either endpoint, and the row's frame.
            body_a, body_b, direction, p0, p1 = write_contact_row(data, law, c, row, False)
            if a_matches:
                clear_side(support, J_group, at_a, slot + row, art_a)
            if b_matches and not (a_matches and art_b == art_a):
                clear_side(support, J_group, at_b, slot + row, art_b)
            if a_matches:
                accumulate_jacobian_row_world(
                    body_a,
                    1.0,
                    p0,
                    data.origin[art_a],
                    direction,
                    row == CONTACT_SPIN_ROW,
                    body_to_joint,
                    joint_ancestor,
                    joint_qd_start,
                    data.motion,
                    data.dof_start[art_a],
                    target_size,
                    at_a,
                    slot + row,
                    J_group,
                )
            if b_matches:
                accumulate_jacobian_row_world(
                    body_b,
                    -1.0,
                    p1,
                    data.origin[art_b],
                    direction,
                    row == CONTACT_SPIN_ROW,
                    body_to_joint,
                    joint_ancestor,
                    joint_qd_start,
                    data.motion,
                    data.dof_start[art_b],
                    target_size,
                    at_b,
                    slot + row,
                    J_group,
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
    com_world = wp.transform_point(body_q[tid], body_com[tid])
    art = body_to_articulation[tid]
    origin = wp.vec3()
    if art >= 0:
        origin = articulation_origin[art]
    body_qd_out[tid] = wp.spatial_vector(point_velocity(twist, com_world - origin), wp.spatial_bottom(twist))


# =============================================================================
# World-Level PGS and Velocity Kernels for Multi-Articulation
# =============================================================================


@wp.kernel
def compute_world_contact_bias(
    rows: DenseRows,
    world_constraint_count: wp.array[int],
    max_constraints: int,
    dt: float,
    contact_speculative_scale: float,
    contact_w: float,
    weighted: int,
    material: int,
):
    """Compute each world row's RHS bias by its row law (:func:`.row_laws.row_rhs`); ``J*v`` is added later.

    The RHS follows the convention: rhs = J*v + stabilization; a penetrating contact's Baumgarte term
    ``beta * phi / dt`` is negative and leads to positive impulses when resolved by PGS.

    ``weighted`` rows store their relative regularization weight. A material contact (``material``: its writer stored
    its rhs and compliance ``eps``, from which the sweeps form its weight, :func:`.sweep.row_weight`) keeps
    its rhs.

    One thread per (world, row): ``dim = world_count * max_constraints``. The previous one-thread-per-world loop
    left the GPU idle (64-128 blocks) for 30-60 us per launch.
    """
    tid = wp.tid()
    world = tid // max_constraints
    i = tid - world * max_constraints
    if i >= world_constraint_count[world]:
        return
    kind = rows.row_type[world, i]
    if material != 0 and kind == PGS_CONSTRAINT_TYPE_CONTACT:
        if rows.row_eps[world, i] > 0.0:
            return
    inv_dt = 1.0 / dt
    phi = rows.phi[world, i]
    target = rows.target[world, i]
    rhs, row_w = row_rhs(
        kind, phi, rows.row_beta[world, i], target, inv_dt, 1.0, contact_speculative_scale, 1.0, contact_w
    )
    rows.rhs[world, i] = rhs
    if weighted != 0:
        rows.row_w[world, i] = row_w


@wp.kernel
def apply_world_contact_restitution_matrix_free(
    rows: DenseRows,
    world_constraint_count: wp.array[int],
    max_constraints: int,
    world_incident_velocity: wp.array[float],
    world_dof_indices: wp.array2d[int],
    row_support: RowSupport,
    packed_J: wp.array3d[float],
    sparse_diagonal: int,
    dense_offsets: wp.array[int],
    dense_groups: wp.array[int],
    dense_dofs: int,
    dense_L: wp.array3d[float],
    dense_Z: wp.array3d[float],
    sparse_row_dof: wp.array3d[int],
    sparse_row_z: wp.array3d[float],
    diagonal_inverse_mass: wp.array[float],
    dt: float,
    restitution_velocity_threshold: float,
    write_row_w: int,
):
    """Replace a matrix-free contact bias with a one-shot restitution target.

    The row's incident velocity is its packed Jacobian row's (:func:`world_contact_row_dot`), or under the
    ``sparse_diagonal`` owner its whitened row's against the entered velocity (:func:`sparse_diagonal_row_dot`). Matrix-free GS adds live J*v itself, so the row stores only
    ``-target + e*u_incident``.
    """
    tid = wp.tid()
    world = tid // max_constraints
    i = tid - world * max_constraints
    if i >= world_constraint_count[world] or rows.row_type[world, i] != PGS_CONSTRAINT_TYPE_CONTACT:
        return
    restitution = rows.restitution[world, i]
    if restitution <= 0.0:
        return
    relative_incident = float(0.0)
    if sparse_diagonal != 0:
        relative_incident = sparse_diagonal_row_dot(
            world,
            i,
            world_incident_velocity,
            world_dof_indices,
            dense_offsets,
            dense_groups,
            dense_dofs,
            dense_L,
            dense_Z,
            sparse_row_dof,
            sparse_row_z,
            diagonal_inverse_mass,
        )
    else:
        relative_incident = world_contact_row_dot(
            world_dof_indices, row_support, packed_J, world_incident_velocity, world, i
        )
    target_vel = rows.target[world, i]
    relative_incident -= target_vel
    fires, bounce = contact_restitution_rhs(
        rows.phi[world, i], relative_incident, target_vel, restitution, dt, restitution_velocity_threshold
    )
    if fires:
        rows.rhs[world, i] = bounce
        if write_row_w != 0:
            rows.row_w[world, i] = 1.0


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


# =============================================================================
# Matrix-Free PGS Kernels for Free Rigid Bodies
# =============================================================================


@wp.kernel
def build_mf_contact_rows(
    mf: MfRows,
    data: ContactInput,
    total_num_threads: int,
    has_target_velocity: int,
    law: PatchRowLaw,
):
    """Build matrix-free rows for contacts between free rigid bodies / ground, with a capacity-independent launch.

    For root free joints, the internal qd used here stores the COM-point linear
    term and angular velocity, i.e. `qd = [v_com_point, omega]`, where the COM
    point is the root body's world-space COM position. The MF contact Jacobian
    uses contact position relative to that point:
        J = [d, r x d]   (r = p_contact - p_com_world)
    A tangent row folds the row law's beta into its gap (matrix-free rows keep no beta). A material contact
    (``contact_compliance``, positive stiffness) writes its normal's rhs and compliance
    (:func:`.row_laws.contact_material_rhs`) and takes no restitution, its tangents the contact's friction scale, as
    :func:`.row_build.write_contact_row` does.
    """
    total_contacts = data.count[0]
    for c in range(wp.tid(), total_contacts, total_num_threads):
        slot = data.slot[c]
        if data.path[c] != 1 or slot < 0:
            continue
        world = data.world[c]
        art_a = data.art0[c]
        art_b = data.art1[c]
        body_a, body_b, normal, point_a, point_b, phi, tangent0, tangent1, friction_a, friction_b = contact_frame(
            data, law, c
        )
        # Zero-DOF articulations contribute collision geometry but own no
        # generalized response: their rows treat them like ground.
        response_body_a = -1
        response_body_b = -1
        if body_a >= 0 and art_a >= 0 and data.response_dofs[art_a] > 0:
            response_body_a = body_a
        if body_b >= 0 and art_b >= 0 and data.response_dofs[art_b] > 0:
            response_body_b = body_b
        mu, restitution = contact_materials(data.shape0[c], data.shape1[c], data.material_mu, data.material_restitution)
        stiffness = float(0.0)
        if data.material != 0:
            stiffness = data.stiffness[c]
        if stiffness > 0.0:
            restitution = 0.0
        # Allocation owns the row extent; recomputing eligibility can cross a
        # floating-point threshold and overwrite the following contact.
        rows = 1
        if data.slots_needed[c] >= 3:
            rows = data.slots_needed[c]
        for row in range(rows):
            row_idx = slot + row
            direction, p0, p1 = contact_row_frame(
                row, data.shared_anchor, normal, tangent0, tangent1, point_a, point_b, friction_a, friction_b
            )
            spin = row == CONTACT_SPIN_ROW
            # A free body's row about its articulation origin: J = [d, r x d], opposite on body B; a spin row's
            # [0, n]. The rows' build whitens it in place (Z = L^-1 J^T).
            if response_body_a >= 0:
                J_a = free_body_jacobian(direction, p0, data.origin[art_a])
                if spin:
                    J_a = free_body_spin_jacobian(normal)
                for k in range(6):
                    mf.Z_a[world, row_idx, k] = J_a[k]
            if response_body_b >= 0:
                J_b = free_body_jacobian(direction, p1, data.origin[art_b])
                if spin:
                    J_b = free_body_spin_jacobian(normal)
                for k in range(6):
                    mf.Z_b[world, row_idx, k] = -J_b[k]
            mf.body_a[world, row_idx] = response_body_a
            mf.body_b[world, row_idx] = response_body_b
            if row == 0:
                mf.row_type[world, row_idx] = PGS_CONSTRAINT_TYPE_CONTACT
                mf.row_parent[world, row_idx] = -1
                mf.row_mu[world, row_idx] = mu
                mf.phi[world, row_idx] = phi
                mf.restitution[world, row_idx] = restitution
            elif spin:
                mf.row_type[world, row_idx] = PGS_CONSTRAINT_TYPE_TORSION
                mf.row_parent[world, row_idx] = slot
                mf.row_mu[world, row_idx] = contact_torsion_mu(data, c)
                mf.phi[world, row_idx] = 0.0
                mf.restitution[world, row_idx] = 0.0
            else:
                tangent_phi, beta, friction_mu = patch_tangent_row(law, c, row, mu, data.friction_scale)
                if stiffness > 0.0 and data.friction_weight[c] > 0.0:
                    friction_mu *= data.friction_weight[c]
                mf.row_type[world, row_idx] = PGS_CONSTRAINT_TYPE_FRICTION
                mf.row_parent[world, row_idx] = slot
                mf.row_mu[world, row_idx] = friction_mu
                mf.phi[world, row_idx] = beta * tangent_phi
                mf.restitution[world, row_idx] = 0.0
            target = float(0.0)
            if has_target_velocity != 0:
                target = prescribed_target(
                    body_a, art_a, body_b, art_b, p0, p1, direction, data.prescribed, data.origin, data.body_v
                )
                if spin:
                    target = prescribed_spin_target(body_a, art_a, body_b, art_b, normal, data.prescribed, data.body_v)
                mf.target[world, row_idx] = target
            if data.material != 0:
                eps = float(0.0)
                if row == 0 and stiffness > 0.0:
                    material_rhs, eps = contact_material_rhs(phi, target, stiffness, data.damping[c], data.dt)
                    mf.rhs[world, row_idx] = material_rhs
                mf.eps[world, row_idx] = eps


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
    if c >= routes.count[0]:
        return
    if routes.path[c] != 1:  # MF path only
        return
    new_slot = routes.slot[c]
    if new_slot < 0:
        return

    world = routes.world[c]
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
        and mf.row_type[world, new_slot] == PGS_CONSTRAINT_TYPE_CONTACT
        and prev_mf_row_type[world, prev_slot] == PGS_CONSTRAINT_TYPE_CONTACT
    ):
        mf_impulses[world, new_slot] = decay * dt_scale * prev_mf_impulses[world, prev_slot]
    # else: leave 0 (already memset)

    # Friction rows (offsets 1..2): only if THIS step allocated them here and
    # they belong to this contact's block (parent == base slot).
    for r in range(1, 3):
        new_r = new_slot + r
        if new_r < count and new_r < mf_max_c:
            if mf.row_type[world, new_r] == PGS_CONSTRAINT_TYPE_FRICTION and mf.row_parent[world, new_r] == new_slot:
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
    # The spin row (contact torsion): carried as the friction rows are.
    new_r = new_slot + CONTACT_SPIN_ROW
    prev_r = prev_slot + CONTACT_SPIN_ROW
    if (
        matched
        and prev_slot >= 0
        and new_r < count
        and new_r < mf_max_c
        and prev_r < mf_max_c
        and mf.row_type[world, new_r] == PGS_CONSTRAINT_TYPE_TORSION
        and mf.row_parent[world, new_r] == new_slot
        and prev_mf_row_type[world, prev_r] == PGS_CONSTRAINT_TYPE_TORSION
        and prev_mf_row_parent[world, prev_r] == prev_slot
    ):
        mf_impulses[world, new_r] = decay * dt_scale * prev_mf_impulses[world, prev_r]

    if mi >= 0 and prev_slot >= 0 and prev_slot < mf_max_c:
        _transport_warmstart_contact(
            routes.normal[c],
            previous_normal[mi],
            world,
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
        if rows.row_type[world, new_slot] == PGS_CONSTRAINT_TYPE_CONTACT:
            world_impulses[world, new_slot] = decay * dt_scale * prev_dense_impulses[world, prev_slot]

    # Friction rows (offsets 1, 2): only if THIS step allocated them here.
    for r in range(1, 3):
        new_r = new_slot + r
        if new_r < count and new_r < max_constraints:
            if (
                rows.row_type[world, new_r] == PGS_CONSTRAINT_TYPE_FRICTION
                and rows.row_parent[world, new_r] == new_slot
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
    # The spin row (contact torsion): carried as the friction rows are.
    new_r = new_slot + CONTACT_SPIN_ROW
    prev_r = prev_slot + CONTACT_SPIN_ROW
    if (
        matched
        and prev_slot >= 0
        and new_r < count
        and new_r < max_constraints
        and prev_r < max_constraints
        and rows.row_type[world, new_r] == PGS_CONSTRAINT_TYPE_TORSION
        and rows.row_parent[world, new_r] == new_slot
        and prev_dense_row_type[world, prev_r] == PGS_CONSTRAINT_TYPE_TORSION
        and prev_dense_row_parent[world, prev_r] == prev_slot
    ):
        world_impulses[world, new_r] = decay * dt_scale * prev_dense_impulses[world, prev_r]

    if mi >= 0 and prev_slot >= 0 and prev_slot < max_constraints:
        _transport_warmstart_contact(
            routes.normal[c],
            previous_normal[mi],
            world,
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
    speed row (:func:`populate_rigid_velocity_limit_rows`) that the sweeps
    project onto that ball, along the current velocity.

    ``velocity_limit_activation_fraction`` proximity-gates the allocation,
    mirroring :func:`allocate_joint_velocity_limit_slots`: a positive fraction
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
def populate_rigid_velocity_limit_rows(
    bodies: FreeBodies,
    mf: MfRows,
    rigid_velocity_limit_slot: wp.array[int],
):
    """Populate the MF speed rows of free rigid bodies' linear and angular velocity limits.

    A speed row's Jacobian is the unit vector of the first DOF of the
    velocity it bounds (``e_0`` for the linear velocity, ``e_3`` for the
    angular one); the sweeps read the row's three DOFs from there, and its
    effective mass converts the speed correction into the impulse the row
    stores. ``mf_phi`` holds the limit.
    """
    candidate = wp.tid()
    body = bodies.free_rigid_body_indices[candidate]
    art = bodies.body_to_articulation[body]
    if art < 0:
        return
    if bodies.is_free_rigid[art] == 0:
        return

    world = bodies.art_to_world[art]
    if world < 0:
        return

    base = RIGID_SPEED_ROWS * candidate
    for speed_axis in range(RIGID_SPEED_ROWS):
        slot = rigid_velocity_limit_slot[base + speed_axis]
        if slot < 0:
            continue
        limit = bodies.max_linear_velocity[body]
        if speed_axis == 1:
            limit = bodies.max_angular_velocity[body]

        # The axis stays unwhitened: the speed law reads the body's velocity through its factor, not a row.
        for k in range(6):
            mf.Z_a[world, slot, k] = 0.0
            mf.Z_b[world, slot, k] = 0.0
        mf.Z_a[world, slot, 3 * speed_axis] = 1.0

        mf.body_a[world, slot] = body
        mf.body_b[world, slot] = -1
        mf.row_type[world, slot] = PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT
        mf.row_parent[world, slot] = -1
        mf.row_mu[world, slot] = 0.0
        # For velocity-limit rows mf_phi stores the speed limit; contact rows
        # use it as geometric gap. Row type disambiguates the meaning.
        mf.phi[world, slot] = limit


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
    body_to_articulation: wp.array[int],
    articulation_dof_count: wp.array[int],
):
    """Assign exact solver ownership and compact active general worlds."""
    world = wp.tid()
    row_count = world_constraint_count[world]
    mf_count = mf_constraint_count[world]
    primary_articulation = owners.primary_articulation[world]
    pair_articulation = owners.pair_articulation[world]
    residual_pair_articulation = owners.residual_pair_articulation[world]
    single_phase = dense_phase_bounds[world, 1] == row_count

    local_mf = mf_count > 0 and mf_count <= owners.residual_mf_max_rows and residual_pair_articulation >= 0
    mf_row = int(0)
    while mf_row < mf_count and local_mf:
        body_a = mf.body_a[world, mf_row]
        body_b = mf.body_b[world, mf_row]
        if body_a >= 0 and body_to_articulation[body_a] != residual_pair_articulation:
            local_mf = False
        if body_b >= 0 and body_to_articulation[body_b] != residual_pair_articulation:
            local_mf = False
        # Free-body velocity-limit rows are speed rows (projection_laws.speed_row), which the residual local loop
        # does not implement: keep such worlds on the general owner.
        if mf.row_type[world, mf_row] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
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


# =============================================================================
# Tiled kernels for homogenous multi-articulation support
# =============================================================================


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
    Solve H qdd = tau for grouped articulations by forward/backward substitution with the leaf-first factor
    ``P H P^T = L L^T`` (factor coordinate ``i`` at physical coordinate ``n_dofs - 1 - i``).

    Thread dimension: n_arts_of_size (one thread per articulation in this size group)
    """
    idx = wp.tid()
    art = group_to_art[idx]
    last = articulation_dof_start[art] + n_dofs - 1

    # Forward substitution: L z = P tau
    # z is stored temporarily in joint_qdd
    for i in range(n_dofs):
        val = joint_tau[last - i]
        for k in range(i):
            L_ik = L_group[idx, i, k]
            val -= L_ik * joint_qdd[last - k]

        L_ii = L_group[idx, i, i]
        if L_ii != 0.0:
            joint_qdd[last - i] = val / L_ii
        else:
            joint_qdd[last - i] = 0.0

    # Backward substitution: L^T (P qdd) = z
    for i_rev in range(n_dofs):
        i = n_dofs - 1 - i_rev

        val = joint_qdd[last - i]
        for k in range(i + 1, n_dofs):
            L_ki = L_group[idx, k, i]
            val -= L_ki * joint_qdd[last - k]

        L_ii = L_group[idx, i, i]
        if L_ii != 0.0:
            joint_qdd[last - i] = val / L_ii
        else:
            joint_qdd[last - i] = 0.0


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


# =============================================================================
# PGS Convergence Diagnostic Kernel (velocity-space mode)
# =============================================================================


@wp.kernel
def pgs_convergence_diagnostic_velocity(
    rows: DenseRows,
    mf: MfRows,
    # Dense constraints
    constraint_count: wp.array[int],
    world_dof_indices: wp.array2d[int],
    impulses: wp.array2d[float],
    prev_impulses: wp.array2d[float],
    row_support: RowSupport,
    packed_J: wp.array3d[float],
    max_constraints: int,
    # MF constraints
    mf_constraint_count: wp.array[int],
    mf_impulses: wp.array2d[float],
    prev_mf_impulses: wp.array2d[float],
    mf_max_constraints: int,
    factors: BodyFactors,
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
        jv = world_contact_row_dot(world_dof_indices, row_support, packed_J, v_out, world, i)
        residual = jv + rows.rhs[world, i]

        rt = rows.row_type[world, i]
        if rt == PGS_CONSTRAINT_TYPE_CONTACT:
            # Normal: complementarity gap and FB
            comp_gap += lam * residual
            fb_val = wp.sqrt(lam * lam + residual * residual) - lam - residual
            fb_merit += fb_val * fb_val
        elif rt == PGS_CONSTRAINT_TYPE_FRICTION:
            # Friction: tangent residual for sticking contacts
            parent_idx = rows.row_parent[world, i]
            lambda_n = patch_normal_load(rows.row_parent, impulses, world, parent_idx)
            mu = rows.row_mu[world, i]
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

        # Compute residual: r = Z^T (L^T v) + bias
        jv = free_body_row_velocity(mf.Z_a, mf.Z_b, world, i, mf.body_a[world, i], mf.body_b[world, i], factors, v_out)
        residual = jv + mf.rhs[world, i]

        rt = mf.row_type[world, i]
        if rt == PGS_CONSTRAINT_TYPE_CONTACT:
            comp_gap += lam * residual
            fb_val = wp.sqrt(lam * lam + residual * residual) - lam - residual
            fb_merit += fb_val * fb_val
        elif rt == PGS_CONSTRAINT_TYPE_FRICTION:
            parent_idx = mf.row_parent[world, i]
            lambda_n = patch_normal_load(mf.row_parent, mf_impulses, world, parent_idx)
            mu = mf.row_mu[world, i]
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
    rows: DenseRows,
    mf: MfRows,
    # Dense constraints
    constraint_count: wp.array[int],
    world_dof_indices: wp.array2d[int],
    impulses: wp.array2d[float],
    row_support: RowSupport,
    packed_J: wp.array3d[float],
    max_constraints: int,
    # MF constraints
    mf_constraint_count: wp.array[int],
    mf_impulses: wp.array2d[float],
    mf_max_constraints: int,
    factors: BodyFactors,
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
    * Joint-limit rows (``PGS_CONSTRAINT_TYPE_JOINT_LIMIT``) are
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
        rt = rows.row_type[world, i]
        if rt != PGS_CONSTRAINT_TYPE_CONTACT:
            # skip friction rows (handled via parent), joint-limit rows,
            # and joint-target rows — they do not contribute NCP/MDP
            # contact residuals.
            continue

        # Normal row velocity (bias-free): u_n = J_n * v
        u_n = world_contact_row_dot(world_dof_indices, row_support, packed_J, v_out, world, i)
        b_n = rows.rhs[world, i]
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
        neg_phi = -rows.phi[world, i]
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
            if rows.row_type[world, i1] == PGS_CONSTRAINT_TYPE_FRICTION and rows.row_parent[world, i1] == i:
                lt1 = impulses[world, i1]
                mu = rows.row_mu[world, i1]
                u_t1 = world_contact_row_dot(world_dof_indices, row_support, packed_J, v_out, world, i1)

        i2 = i + 2
        if i2 < max_constraints and i2 < m_dense:
            if rows.row_type[world, i2] == PGS_CONSTRAINT_TYPE_FRICTION and rows.row_parent[world, i2] == i:
                lt2 = impulses[world, i2]
                if mu == 0.0:
                    mu = rows.row_mu[world, i2]
                u_t2 = world_contact_row_dot(world_dof_indices, row_support, packed_J, v_out, world, i2)

        # r_cone
        friction_load = ln
        if mu > 0.0:
            friction_load = patch_normal_load(rows.row_parent, impulses, world, i)
        friction_residual = contact_friction_residuals(
            ln, friction_load, mu, u_n, wp.vec2(u_t1, u_t2), wp.vec2(lt1, lt2)
        )
        r_cone = wp.max(r_cone, friction_residual[0])
        r_ds_compl = wp.max(r_ds_compl, friction_residual[1])
        r_ds_dual = wp.max(r_ds_dual, friction_residual[2])
        r_mdp_dir = wp.max(r_mdp_dir, friction_residual[3])

    # ---- Matrix-free constraints (free-rigid contacts) ----
    for i in range(m_mf):
        rt = mf.row_type[world, i]
        if rt != PGS_CONSTRAINT_TYPE_CONTACT:
            continue

        u_n = free_body_row_velocity(mf.Z_a, mf.Z_b, world, i, mf.body_a[world, i], mf.body_b[world, i], factors, v_out)
        b_n = mf.rhs[world, i]
        ln = mf_impulses[world, i]

        ubn = u_n + b_n
        if ln < ubn:
            compl = wp.abs(ln)
        else:
            compl = wp.abs(ubn)
        if compl > r_compl:
            r_compl = compl

        neg_phi = -mf.phi[world, i]
        if neg_phi > r_gap:
            r_gap = neg_phi

        lt1 = float(0.0)
        lt2 = float(0.0)
        u_t1 = float(0.0)
        u_t2 = float(0.0)
        mu = float(0.0)

        i1 = i + 1
        if i1 < mf_max_constraints and i1 < m_mf:
            if mf.row_type[world, i1] == PGS_CONSTRAINT_TYPE_FRICTION and mf.row_parent[world, i1] == i:
                lt1 = mf_impulses[world, i1]
                mu = mf.row_mu[world, i1]
                u_t1 = free_body_row_velocity(
                    mf.Z_a, mf.Z_b, world, i1, mf.body_a[world, i1], mf.body_b[world, i1], factors, v_out
                )

        i2 = i + 2
        if i2 < mf_max_constraints and i2 < m_mf:
            if mf.row_type[world, i2] == PGS_CONSTRAINT_TYPE_FRICTION and mf.row_parent[world, i2] == i:
                lt2 = mf_impulses[world, i2]
                if mu == 0.0:
                    mu = mf.row_mu[world, i2]
                u_t2 = free_body_row_velocity(
                    mf.Z_a, mf.Z_b, world, i2, mf.body_a[world, i2], mf.body_b[world, i2], factors, v_out
                )

        friction_load = ln
        if mu > 0.0:
            friction_load = patch_normal_load(mf.row_parent, mf_impulses, world, i)
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
