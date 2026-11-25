# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The factor's kernels outside the walk.

The turns' gyroscopic moments and the coordinate forces they make, the free roots' prediction corrections (spin
damping, the transport the integrator adds), the joint integrator, and the flags of the mass matrices' and the FK/ID
cache's refresh.
"""

import warp as wp

from ....sim import JointType
from ..spatial_laws import free_root_transport, turn_moment

# FeatherPGS takes no gradients: its kernels compile no adjoints.
wp.set_module_options({"enable_backward": False})


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


TURN_BLOCK = 64
"""Threads a block of :func:`turn_forces`: a thread's law is a long serial chain, so small blocks spread the trees over
every multiprocessor."""


@wp.func
def _turn_moment(
    joint: int,
    body: int,
    start: int,
    turns: int,
    linear: int,
    joint_X_c: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    joint_S_s: wp.array[wp.spatial_vector],
    joint_qd: wp.array[float],
    armature: wp.array[float],
    dt: float,
):
    """A joint of two or more turns' gyroscopic moment on its body (world; the negative of
    :func:`.spatial_laws.turn_moment`, the body's external moment), from the step's input state: the turns' columns and
    velocities (coordinates ``start`` on), the joint's angular armature and the point it pins (its anchor, or the
    body's centre of mass where the joint has a linear axis)."""
    rotation = wp.transform_get_rotation(body_q[body])
    lever = wp.vec3(0.0)
    if linear == 0:
        lever = wp.quat_rotate(rotation, body_com[body] - wp.transform_get_translation(joint_X_c[joint]))
    rotors = wp.mat33(0.0)
    axes = wp.mat33(0.0)
    coordinates = wp.vec3(0.0)
    for k in range(turns):
        axis = wp.quat_rotate_inv(rotation, wp.spatial_bottom(joint_S_s[start + k]))
        rotors += armature[start + k] * wp.outer(axis, axis)
        for i in range(3):
            axes[i, k] = axis[i]
        coordinates[k] = joint_qd[start + k]
    moment = turn_moment(
        body_mass[body],
        rotation,
        body_inertia[body],
        wp.spatial_bottom(body_qd[body]),
        lever,
        rotors,
        axes,
        coordinates,
        dt,
    )
    return -moment


@wp.kernel
def turn_moments(
    turning_joints: wp.array[int],
    turn_body: wp.array[int],
    turn_dofs: wp.array[wp.vec3i],
    joint_X_c: wp.array[wp.transform],
    kinematic_joint_mask: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    joint_S_s: wp.array[wp.spatial_vector],
    joint_qd: wp.array[float],
    armature: wp.array[float],
    dt: float,
    moments: wp.array[wp.vec3],
):
    """Each turning joint's gyroscopic moment (:func:`_turn_moment`), a thread a joint (its body and its turns' first
    coordinate, count and linear axes in ``turn_body`` / ``turn_dofs``); a kinematic joint's body takes none."""
    k = wp.tid()
    joint = turning_joints[k]
    moment = wp.vec3(0.0)
    if kinematic_joint_mask[joint] == 0:
        dofs = turn_dofs[k]
        moment = _turn_moment(
            joint,
            turn_body[k],
            dofs[0],
            dofs[1],
            dofs[2],
            joint_X_c,
            body_q,
            body_qd,
            body_com,
            body_mass,
            body_inertia,
            joint_S_s,
            joint_qd,
            armature,
            dt,
        )
    moments[k] = moment


@wp.kernel
def turn_forces(
    tree_start: wp.array[int],
    turning_joints: wp.array[int],
    parent_joint: wp.array[int],
    kinematic_joint_mask: wp.array[int],
    turn_coordinates: wp.array[wp.vec3i],
    joint_S_s: wp.array[wp.spatial_vector],
    moments: wp.array[wp.vec3],
    turn_tau: wp.array[float],
):
    """The turning joints' moments (:func:`turn_moments`) as coordinate forces, ``J^T`` of each pure moment, a thread a
    coordinate that takes one (``turn_coordinates``: the coordinate, its joint and its tree): the turn coordinates of
    a joint and of its ancestors (``parent_joint``) take their axis against it, summed over the tree's turning joints in
    joint order. A moment has no lever, so no wrench is carried. A kinematic joint's body takes none."""
    entry = turn_coordinates[wp.tid()]
    dof, owner, tree = entry[0], entry[1], entry[2]
    axis = wp.spatial_bottom(joint_S_s[dof])
    tau = float(0.0)
    for k in range(tree_start[tree], tree_start[tree + 1]):
        joint = turning_joints[k]
        if kinematic_joint_mask[joint] != 0:
            continue
        # The coordinate takes the moment where its joint is this one or one of its ancestors.
        up = joint
        while up >= 0 and up != owner:
            up = parent_joint[up]
        if up >= 0:
            tau += wp.dot(axis, moments[k])
    turn_tau[dof] = tau


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
    """The clock's own launch, where no tree walks (a model whose trees respond by their compact inverse alone): the
    walk's refresh advances it everywhere else."""
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
def take_gravity(
    gravity: wp.array[wp.vec3],
    # in/out
    held_gravity: wp.array[wp.vec3],
    # outputs
    changed: wp.array[int],
):
    """Mark each world whose gravity differs from the one the cached dynamics hold (``held_gravity``), and hold the
    model's. A NaN component never compares equal: its world is marked at every notification."""
    world = wp.tid()
    held = held_gravity[world]
    now = gravity[world]
    changed[world] = int(now[0] != held[0] or now[1] != held[1] or now[2] != held[2])
    held_gravity[world] = now


@wp.kernel
def invalidate_fk_id_cache_for_gravity(
    changed: wp.array[int],
    body_world: wp.array[int],
    body_to_articulation: wp.array[int],
    # outputs
    fk_id_cache_valid: wp.array[int],
):
    """Clear the FK/ID cache flag of each articulation that holds a body of a world whose gravity changed
    (:func:`take_gravity`; a body of the global world, ``-1``, reads the last entry, as its gravity)."""
    body = wp.tid()
    if changed[body_world[body]] != 0:
        articulation = body_to_articulation[body]
        if articulation >= 0:
            fk_id_cache_valid[articulation] = 0


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


@wp.func
def _same_pose(a: wp.transform, b: wp.transform) -> bool:
    for i in range(7):
        if a[i] != b[i]:
            return False
    return True


@wp.func
def _same_twist(a: wp.spatial_vector, b: wp.spatial_vector) -> bool:
    for i in range(6):
        if a[i] != b[i]:
            return False
    return True


@wp.kernel
def follow_prescribed_bodies(
    roots: wp.array[int],
    root_articulation: wp.array[int],
    joint_child: wp.array[int],
    kinematic_joint_mask: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    published_q: wp.array[wp.transform],
    published_qd: wp.array[wp.spatial_vector],
    published_valid: wp.array[int],
    # outputs
    written: wp.array[wp.bool],
    fk_id_cache_valid: wp.array[int],
):
    """Mark each kinematic root whose body state was written since the solver published it (:func:`publish_prescribed_bodies`):
    its articulation's coordinates then follow the body (the step's inverse kinematics, which ``written`` selects) and
    its cached kinematics are stale. A root whose publication was forgotten (a reset, a joint-state write) keeps its
    joint state."""
    k = wp.tid()
    joint = roots[k]
    articulation = root_articulation[k]
    written[articulation] = False
    if kinematic_joint_mask[joint] == 0 or published_valid[k] == 0:
        return
    body = joint_child[joint]
    if _same_pose(body_q[body], published_q[k]) and _same_twist(body_qd[body], published_qd[k]):
        return
    written[articulation] = True
    fk_id_cache_valid[articulation] = 0


@wp.kernel
def publish_prescribed_bodies(
    roots: wp.array[int],
    joint_child: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    # outputs
    published_q: wp.array[wp.transform],
    published_qd: wp.array[wp.spatial_vector],
    published_valid: wp.array[int],
):
    """Record each root body's state as the step published it, which a later write is told from."""
    k = wp.tid()
    body = joint_child[roots[k]]
    published_q[k] = body_q[body]
    published_qd[k] = body_qd[body]
    published_valid[k] = 1


@wp.kernel
def forget_prescribed_bodies(
    world_mask: wp.array[wp.bool],
    articulations: wp.array[int],
    root_of_articulation: wp.array[int],
    root_articulation: wp.array[int],
    art_to_world: wp.array[int],
    # outputs
    published_valid: wp.array[int],
):
    """Forget the publication of the roots of the listed ``articulations`` (a joint-state write), or else of the
    worlds ``world_mask`` selects (a reset; none: every root): their joint state stands at the next step."""
    k = wp.tid()
    if articulations:
        root = root_of_articulation[articulations[k]]
        if root >= 0:
            published_valid[root] = 0
        return
    if world_mask:
        if not world_mask[art_to_world[root_articulation[k]]]:
            return
    published_valid[k] = 0
