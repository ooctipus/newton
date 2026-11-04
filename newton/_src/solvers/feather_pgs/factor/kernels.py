# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The factor's kernels outside the walk.

The velocity prescale ahead of the prediction, the free roots' prediction corrections (gyroscopic velocity, spin
damping, the transport the integrator adds), the joint integrator, the bodies' center-of-mass transforms, and the
flags of the mass matrices' and the FK/ID cache's refresh.
"""

import warp as wp

from ....sim import BodyFlags, JointType
from ..device_preludes import lane_minimum
from ..joint_table import JointTable
from ..spatial_laws import free_root_transport


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


@wp.struct
class Clock:
    """The step's clock on the device, so a captured graph's replays keep their own cadence: the step's index, the
    duration of the step it last advanced and of the one before (0 before any), the mass refresh's interval, and
    whether the indexed step refreshes every mass matrix on the cadence.

    The walk's refresh advances it (:func:`clock_advance`), after the step's repair and mass-update mask and before
    its finish and warm start, in a kernel that reads it nowhere else: the readers before the refresh see the step's
    own index with the last step's duration (:func:`clock_refresh`, :func:`clock_changed`, with the step's duration
    they hold); the readers after it see the next step's index with this step's duration (the finish's next-step
    geometry: ``cadence``; the warm start's ratio: :func:`clock_dt_ratio`)."""

    step: int
    dt: float
    prev_dt: float
    interval: int
    cadence: int


@wp.func
def clock_advance(clock: wp.array[Clock], dt: float):
    """Advance the clock past a step of duration ``dt``: the next index and its cadence, this step's duration kept."""
    c = clock[0]
    c.step = c.step + 1
    c.prev_dt = c.dt
    c.dt = dt
    c.cadence = int(c.step % c.interval == 0)
    clock[0] = c


@wp.kernel
def advance_clock(clock: wp.array[Clock], dt: float):
    """The clock's own launch, where no tree walks (a model of stars alone, :mod:`.scalar_branches`): the walk's
    refresh advances it everywhere else."""
    clock_advance(clock, dt)


@wp.func
def clock_changed(c: Clock, dt: float) -> int:
    """Whether a step of duration ``dt`` differs from the last step's (read before the advance): the mass refreshes and
    every cache repairs."""
    return int(c.dt > 0.0 and wp.abs(dt - c.dt) > 1.0e-8)


@wp.func
def clock_refresh(c: Clock, dt: float) -> int:
    """Whether a step of duration ``dt`` refreshes every mass matrix (read before the advance): its cadence, or its
    changed duration."""
    return int(c.cadence != 0 or clock_changed(c, dt) != 0)


@wp.func
def clock_dt_ratio(c: Clock) -> float:
    """The step's duration over the last step's (1 on the first step), read after the advance: a carried impulse,
    proportional to the step for a quasi-static load, is rescaled by it."""
    ratio = 1.0
    if c.prev_dt > 0.0:
        ratio = c.dt / c.prev_dt
    return ratio


@wp.kernel
def build_mass_update_mask(
    clock: wp.array[Clock],
    dt: float,
    # in/out
    mass_update_requested: wp.array[int],
    # outputs
    mass_update_mask: wp.array[int],
):
    """Consume the requests into a step's mask, with the clock's refresh."""
    tid = wp.tid()
    flag = clock_refresh(clock[0], dt)
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
