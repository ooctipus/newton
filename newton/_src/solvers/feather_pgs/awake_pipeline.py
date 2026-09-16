# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental complete scalar-component ownership around the existing solve.

The immutable topology partitions independent prismatic components from the
ordinary articulated fallback. Owned components do not also execute the old
drive, force, predictor, integration and publication producers.
"""

import math

import numpy as np
import warp as wp

from ...sim import BodyFlags, ModelFlags
from .awake_component_kernels import ScalarParameters, scalar_force, scalar_publication
from .kernels import finalize_body_dynamics_body, jcalc_integrate
from .sleeping import _REST_HORIZON, _mark_contacts


@wp.struct
class ComponentData:
    owned: wp.array[int]
    body: wp.array[int]
    joint: wp.array[int]
    coordinate: wp.array[int]
    dof: wp.array[int]
    world: wp.array[int]
    group: wp.array[int]
    local_dof: wp.array[int]
    parameters: wp.array[ScalarParameters]
    dirty_world: wp.array[int]
    contact: wp.array[int]
    invalid_contact: wp.array[int]
    sleeping: wp.array[int]
    counters: wp.array[int]
    expected_valid: wp.array[int]
    expected_q: wp.array[float]
    expected_qd: wp.array[float]
    expected_target_q: wp.array[float]
    expected_target_qd: wp.array[float]
    expected_dt: wp.array[float]
    begin_q: wp.array[float]
    begin_target_q: wp.array[float]
    begin_target_qd: wp.array[float]
    can_sleep: wp.array[int]
    body_awake: wp.array[int]
    joint_awake: wp.array[int]


@wp.struct
class ParameterSources:
    body_joint: wp.array[int]
    parent: wp.array[int]
    articulation: wp.array[int]
    parent_frame: wp.array[wp.transform]
    child_frame: wp.array[wp.transform]
    axis: wp.array[wp.vec3]
    body_com: wp.array[wp.vec3]
    body_flags: wp.array[int]
    mass: wp.array[float]
    armature: wp.array2d[float]
    ke: wp.array[float]
    kd: wp.array[float]
    effort: wp.array[float]
    passive_k: wp.array[float]
    passive_ref: wp.array[float]
    damping: wp.array[float]
    gravity: wp.array[wp.vec3]
    drive_row: wp.array[int]


@wp.struct
class ComponentState:
    body_q: wp.array[wp.transform]
    body_q_com: wp.array[wp.transform]
    joint_S: wp.array[wp.spatial_vector]
    body_v: wp.array[wp.spatial_vector]
    body_a: wp.array[wp.spatial_vector]
    body_f: wp.array[wp.spatial_vector]
    origin: wp.array[wp.vec3]
    tau: wp.array[float]
    qdd: wp.array[float]
    v_hat: wp.array[float]
    inverse_mass: wp.array[float]
    drive_K: wp.array[float]


@wp.func
def _parameters(data: ComponentData, source: ParameterSources, component: int):
    joint = data.joint[component]
    body = data.body[component]
    dof = data.dof[component]
    frame = source.parent_frame[joint]
    parent = source.parent[joint]
    safe = source.body_flags[body] == BodyFlags.DYNAMIC
    while parent >= 0:
        ancestor = source.body_joint[parent]
        safe = safe and source.body_flags[parent] == BodyFlags.DYNAMIC
        frame = (source.parent_frame[ancestor] * wp.transform_inverse(source.child_frame[ancestor])) * frame
        parent = source.parent[ancestor]
    p = ScalarParameters()
    p.axis = wp.quat_rotate(wp.transform_get_rotation(frame), source.axis[dof])
    p.rest_body = frame * wp.transform_inverse(source.child_frame[joint])
    p.com_offset = wp.quat_rotate(wp.transform_get_rotation(p.rest_body), source.body_com[body])
    p.gravity = source.gravity[0]
    p.mass = source.mass[body]
    p.armature = source.armature[data.group[component], data.local_dof[component]]
    p.ke = source.ke[dof]
    p.kd = source.kd[dof]
    if source.drive_row[dof] < 0:
        p.ke = 0.0
        p.kd = 0.0
    p.effort = source.effort[dof]
    p.passive_k = source.passive_k[dof]
    p.passive_ref = source.passive_ref[dof]
    p.damping = source.damping[dof]
    safe = safe and wp.isfinite(p.mass) and p.mass > 0.0
    safe = safe and wp.isfinite(p.ke) and p.ke >= 0.0 and wp.isfinite(p.kd) and p.kd >= 0.0
    safe = safe and wp.isfinite(p.passive_k) and wp.isfinite(p.passive_ref) and wp.isfinite(p.damping)
    safe = safe and wp.isfinite(p.armature)
    for element in range(3):
        safe = safe and wp.isfinite(p.axis[element]) and wp.isfinite(p.gravity[element])
        safe = safe and wp.isfinite(p.com_offset[element])
    for element in range(7):
        safe = safe and wp.isfinite(p.rest_body[element])
    scale = wp.length(p.axis)
    safe = safe and wp.isfinite(scale) and scale > 0.0
    p.allow_sleep = int(safe)
    return p


@wp.kernel(enable_backward=False)
def prepare_components(
    data: ComponentData,
    source: ParameterSources,
    state: ComponentState,
    q: wp.array[float],
    qd: wp.array[float],
    target_q: wp.array[float],
    target_qd: wp.array[float],
    joint_force: wp.array[float],
    body_force: wp.array[wp.spatial_vector],
    kinematic: wp.array[int],
    dt: float,
    source_changed: int,
):
    component = data.owned[wp.tid()]
    body = data.body[component]
    joint = data.joint[component]
    dof = data.dof[component]
    coordinate = data.coordinate[component]
    world = data.world[component]
    dirty = data.dirty_world[world] != 0 or data.dirty_world[data.dirty_world.shape[0] - 1] != 0
    if dirty:
        p = _parameters(data, source, component)
        data.parameters[component] = p
    position = q[coordinate]
    velocity = qd[dof]
    target = target_q[dof]
    target_velocity = target_qd[dof]
    force = body_force[body]
    authored = data.expected_valid[component] == 0
    authored = authored or position != data.expected_q[component] or velocity != data.expected_qd[component]
    target_changed = (
        target != data.expected_target_q[component] or target_velocity != data.expected_target_qd[component]
    )
    allowed = data.parameters[component].allow_sleep != 0
    allowed = allowed and data.invalid_contact[0] == 0 and data.contact[component] == 0
    allowed = allowed and wp.isfinite(position) and wp.isfinite(velocity) and wp.isfinite(target)
    allowed = allowed and target_velocity == 0.0 and joint_force[dof] == 0.0 and kinematic[dof] == 0
    for element in range(6):
        allowed = allowed and force[element] == 0.0
    if dirty or authored or target_changed or data.expected_dt[component] != dt or not allowed:
        data.sleeping[component] = 0
        data.counters[component] = 0
    # Authored scalar coordinates change neither their fixed root nor an
    # unrelated articulation's mass. Their complete owner repairs geometry
    # below; do not invalidate those independent producers or request a global
    # factor refresh. Explicit reset/model notifications retain their owners.
    asleep = data.sleeping[component] != 0
    data.body_awake[body] = int(not asleep)
    data.joint_awake[joint] = int(not asleep)
    if asleep:
        # Grant established zero tau/qdd/predictor and fixed drive response.
        # No fallback producer writes these entries; unchanged input/contact
        # checks above are sufficient to retain the complete scalar lease.
        # A different state allocation still needs its public geometry filled,
        # but not a second force, inverse-mass, or predictor evaluation.
        if source_changed != 0:
            p = data.parameters[component]
            articulation = source.articulation[joint]
            pose, com, motion, body_v, body_a, body_f = scalar_publication(
                p, position, velocity, state.origin[articulation]
            )
            state.body_q[body] = pose
            state.body_q_com[body] = com
            state.joint_S[dof] = motion
            state.body_v[body] = body_v
            state.body_a[body] = body_a
            state.body_f[body] = body_f
        return
    p = data.parameters[component]
    articulation = source.articulation[joint]
    data.can_sleep[component] = int(allowed)
    data.begin_q[component] = position
    data.begin_target_q[component] = target
    data.begin_target_qd[component] = target_velocity
    # Keep coefficient lifetime on the device: a cold CUDA graph capture must
    # not bake a Python "timestep changed" decision into every later replay.
    data.expected_dt[component] = dt
    external = wp.spatial_top(force)
    if kinematic[dof] != 0:
        external = wp.vec3()
    tau, K = scalar_force(p, position, velocity, target, target_velocity, joint_force[dof], external, dt)
    row = source.drive_row[dof]
    if row >= 0:
        state.drive_K[row] = K
    # This mass is independent of coordinates. Cached model properties change
    # only at notified boundaries; evaluating it does not shorten held-factor
    # cadence for the articulated fallback.
    inverse = 1.0 / (p.mass * wp.dot(p.axis, p.axis) + p.armature + K)
    state.inverse_mass[dof] = inverse
    acceleration = tau * inverse
    if kinematic[dof] != 0:
        acceleration = 0.0
    state.tau[dof] = tau
    state.qdd[dof] = acceleration
    state.v_hat[dof] = velocity + dt * acceleration
    if dirty or authored or source_changed != 0:
        pose, com, motion, body_v, body_a, body_f = scalar_publication(
            p, position, velocity, state.origin[articulation]
        )
        state.body_q[body] = pose
        state.body_q_com[body] = com
        state.joint_S[dof] = motion
        state.body_v[body] = body_v
        state.body_a[body] = body_a
        state.body_f[body] = body_f


@wp.kernel(enable_backward=False)
def finish_components(
    data: ComponentData,
    source: ParameterSources,
    state: ComponentState,
    q: wp.array[float],
    qd: wp.array[float],
    body_q_previous: wp.array[wp.transform],
    kinematic: wp.array[int],
    v_out: wp.array[float],
    dt: float,
    tolerance: float,
    quiet_steps: int,
    q_new: wp.array[float],
    qd_new: wp.array[float],
    body_q_new: wp.array[wp.transform],
    body_qd_new: wp.array[wp.spatial_vector],
):
    component = data.owned[wp.tid()]
    body = data.body[component]
    joint = data.joint[component]
    dof = data.dof[component]
    coordinate = data.coordinate[component]
    was_asleep = data.sleeping[component] != 0
    if was_asleep and v_out[dof] == 0.0:
        # Public output may be a fresh allocation, so publication cannot be
        # skipped even though the internal stationary state is already valid.
        q_new[coordinate] = q[coordinate]
        qd_new[dof] = 0.0
        body_q_new[body] = body_q_previous[body]
        body_qd_new[body] = wp.spatial_vector()
        v_out[dof] = 0.0
        return
    p = data.parameters[component]
    position = q[coordinate]
    velocity = qd[dof]
    acceleration = (v_out[dof] - velocity) / dt
    if kinematic[dof] != 0:
        v_out[dof] = velocity
        acceleration = 0.0
    if not was_asleep and kinematic[dof] == 0:
        velocity += acceleration * dt
        position += velocity * dt
    scale = wp.length(p.axis)
    speed = wp.abs(velocity) * scale
    displacement = wp.abs(position - data.begin_q[component]) * scale
    quiet = data.can_sleep[component] != 0
    quiet = quiet and wp.isfinite(position) and wp.isfinite(velocity)
    quiet = quiet and wp.isfinite(speed) and wp.isfinite(displacement)
    quiet = quiet and speed <= tolerance and displacement <= tolerance * dt
    quiet = quiet and wp.isfinite(acceleration) and wp.abs(acceleration) * scale * _REST_HORIZON <= tolerance
    if quiet:
        data.counters[component] = wp.min(data.counters[component] + 1, quiet_steps)
        if data.counters[component] >= quiet_steps:
            data.sleeping[component] = 1
            velocity = 0.0
            acceleration = 0.0
            v_out[dof] = 0.0
            state.tau[dof] = 0.0
            state.v_hat[dof] = 0.0
    else:
        data.counters[component] = 0
        data.sleeping[component] = 0
    q_new[coordinate] = position
    qd_new[dof] = velocity
    state.qdd[dof] = acceleration
    data.expected_q[component] = position
    data.expected_qd[component] = velocity
    data.expected_target_q[component] = data.begin_target_q[component]
    data.expected_target_qd[component] = data.begin_target_qd[component]
    data.expected_valid[component] = 1
    if was_asleep:
        body_q_new[body] = body_q_previous[body]
        body_qd_new[body] = wp.spatial_vector()
    else:
        articulation = source.articulation[joint]
        pose, com, motion, body_v, body_a, body_f = scalar_publication(
            p, position, velocity, state.origin[articulation]
        )
        body_q_new[body] = pose
        body_qd_new[body] = body_v
        state.body_q_com[body] = com
        state.joint_S[dof] = motion
        state.body_v[body] = body_v
        state.body_a[body] = body_a
        state.body_f[body] = body_f


@wp.kernel(enable_backward=False)
def mark_dirty_worlds(changed: wp.array[int], dirty: wp.array[int]):
    world = wp.tid()
    if changed[world] != 0:
        dirty[world] = 1


@wp.kernel(enable_backward=False)
def fallback_mass_mask(owned_art: wp.array[int], global_flag: int, requested: wp.array[int], mask: wp.array[int]):
    art = wp.tid()
    mask[art] = int(owned_art[art] == 0 and (global_flag != 0 or requested[0] != 0))


@wp.kernel(enable_backward=False)
def zero_fallback_qdd(indices: wp.array[int], qdd: wp.array[float]):
    qdd[indices[wp.tid()]] = 0.0


@wp.kernel(enable_backward=False)
def zero_fallback_body_force(indices: wp.array[int], force: wp.array[wp.spatial_vector]):
    force[indices[wp.tid()]] = wp.spatial_vector()


@wp.kernel(enable_backward=False)
def prepare_fallback_drives(
    indices: wp.array[int],
    row_by_dof: wp.array[int],
    q_by_dof: wp.array[int],
    q: wp.array[float],
    qd: wp.array[float],
    ke: wp.array[float],
    kd: wp.array[float],
    target: wp.array[float],
    target_velocity: wp.array[float],
    effort: wp.array[float],
    dt: float,
    row_K: wp.array[float],
    tau: wp.array[float],
):
    dof = indices[wp.tid()]
    tau[dof] = 0.0
    row = row_by_dof[dof]
    if row < 0:
        return
    K = ke[dof] * dt * dt + kd[dof] * dt
    if K <= 0.0:
        row_K[row] = 0.0
        return
    u0 = -(ke[dof] * (q[q_by_dof[dof]] - target[dof] + dt * qd[dof]) + kd[dof] * (qd[dof] - target_velocity[dof]))
    if effort[dof] > 0.0:
        u0 = wp.clamp(u0, -effort[dof], effort[dof])
    row_K[row] = K
    tau[dof] = u0


@wp.kernel(enable_backward=False)
def predict_fallback(
    indices: wp.array[int],
    qd: wp.array[float],
    kinematic: wp.array[int],
    dt: float,
    qdd: wp.array[float],
    v_hat: wp.array[float],
):
    dof = indices[wp.tid()]
    if kinematic[dof] != 0:
        qdd[dof] = 0.0
    v_hat[dof] = qd[dof] + qdd[dof] * dt


@wp.kernel(enable_backward=False)
def update_fallback_qdd(
    indices: wp.array[int],
    qd: wp.array[float],
    kinematic: wp.array[int],
    inv_dt: float,
    v_out: wp.array[float],
    qdd: wp.array[float],
):
    dof = indices[wp.tid()]
    if kinematic[dof] != 0:
        v_out[dof] = qd[dof]
        qdd[dof] = 0.0
    else:
        qdd[dof] = (v_out[dof] - qd[dof]) * inv_dt


@wp.kernel(enable_backward=False)
def integrate_fallback(
    indices: wp.array[int],
    joint_type: wp.array[int],
    parent: wp.array[int],
    child: wp.array[int],
    q_start: wp.array[int],
    qd_start: wp.array[int],
    kinematic: wp.array[int],
    dims: wp.array2d[int],
    body_com: wp.array[wp.vec3],
    child_frame: wp.array[wp.transform],
    q: wp.array[float],
    qd: wp.array[float],
    qdd: wp.array[float],
    dt: float,
    angular_damping: float,
    q_new: wp.array[float],
    qd_new: wp.array[float],
):
    joint = indices[wp.tid()]
    if kinematic[joint] != 0:
        for coordinate in range(q_start[joint], q_start[joint + 1]):
            q_new[coordinate] = q[coordinate]
        for dof in range(qd_start[joint], qd_start[joint + 1]):
            qd_new[dof] = qd[dof]
        return
    jcalc_integrate(
        joint_type[joint],
        child[joint],
        body_com,
        child_frame[joint],
        q,
        qd,
        qdd,
        q_start[joint],
        qd_start[joint],
        dims[joint, 0],
        dims[joint, 1],
        dt,
        angular_damping,
        parent[joint],
        q_new,
        qd_new,
    )


@wp.kernel(enable_backward=False)
def finalize_fallback(
    indices: wp.array[int],
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
    body = indices[wp.tid()]
    finalize_body_dynamics_body(
        body,
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


class AwakePipeline:
    """Partition complete scalar components from ordinary fallback producers."""

    @classmethod
    def create(cls, solver):
        """Return an owner only when every scalar producer has disjoint coverage."""
        sleep = getattr(solver, "_sleeping", None)
        if sleep is None or not sleep.enabled:
            return None
        model = solver.model
        if (
            solver.pgs_mode != "matrix_free"
            or solver.drive_mode != "augmented"
            or not solver._prismatic_linear_state
            or solver._prismatic_publication is None
            or not solver._fk_id_cache_enabled
            or solver._fk_id_cache_uses_snapshot
            or not solver._direct_compact_diagonal_inertia
            or not solver._sparse_diagonal_contact_solve
            or not solver._async_augmented_drives
            or not solver._parallel_augmented_drive_topology
            or solver.enable_joint_velocity_limits
            or solver.pgs_velocity_iterations > 0
            or solver.pgs_warmstart
            or model.requires_grad
            or model.particle_count
            or solver.grouped_dynamics
            or solver._grouped_topology is not None
            or solver._grouped_tau_mass
            or solver._grouped_mass
            or solver._fused_k1
        ):
            return None
        if any(
            getattr(solver, name, None) is not None
            for name in (
                "_kinetic_world",
                "_joint_world",
                "_world_scan_publication",
                "_g1_kinetic_state",
                "_franka_kinetic_state",
                "_sparse_factor",
                "_row_packets",
                "_allegro_kinetic_rows",
                "_single_factor",
            )
        ):
            return None
        direct = solver._compact_diagonal_mass_size
        if direct is None or direct <= 0 or direct not in solver.group_to_art:
            return None
        eligible = sleep.component_eligible.numpy() != 0
        owned = np.flatnonzero(eligible).astype(np.int32)
        if not len(owned):
            return None
        bodies = sleep.component_body.numpy()[owned]
        joints = sleep.component_joint.numpy()[owned]
        dofs = sleep.component_dof.numpy()[owned]
        art = model.joint_articulation.numpy()
        group_to_art = solver.group_to_art[direct].numpy()
        owned_art = np.zeros(model.articulation_count, dtype=np.int32)
        owned_art[art[joints]] = 1
        expected_dofs = int(len(group_to_art) * direct)
        if len(owned) != expected_dofs or not np.array_equal(np.flatnonzero(owned_art), np.sort(group_to_art)):
            return None
        return cls(solver, _coverage=(owned, bodies, joints, dofs, art, group_to_art, owned_art))

    def __init__(self, solver, *, _coverage):
        self.solver = solver
        self.model = model = solver.model
        self.sleep = sleep = solver._sleeping
        owned, bodies, joints, dofs, art, group_to_art, owned_art = _coverage
        direct = solver._compact_diagonal_mass_size
        device = model.device

        def remaining(count, selected):
            mask = np.ones(count, dtype=bool)
            mask[selected] = False
            return wp.array(np.flatnonzero(mask).astype(np.int32), dtype=wp.int32, device=device)

        self.owned_art = wp.array(owned_art, dtype=wp.int32, device=device)
        self.fallback_dofs = remaining(model.joint_dof_count, dofs)
        self.fallback_joints = remaining(model.joint_count, joints)
        self.fallback_bodies = remaining(model.body_count, bodies)
        data = ComponentData()
        data.owned = wp.array(owned, dtype=wp.int32, device=device)
        data.body, data.joint = sleep.component_body, sleep.component_joint
        data.coordinate, data.dof = sleep.component_q, sleep.component_dof
        data.world = sleep.plan.component_world
        group = np.full(sleep.plan.component_count, -1, dtype=np.int32)
        local_dof = group.copy()
        art_to_group = np.full(model.articulation_count, -1, dtype=np.int32)
        art_to_group[group_to_art] = np.arange(len(group_to_art), dtype=np.int32)
        group[owned] = art_to_group[art[joints]]
        local_dof[owned] = dofs - solver._model_plan.articulation_dof_start[art[joints]]
        data.group = wp.array(group, dtype=wp.int32, device=device)
        data.local_dof = wp.array(local_dof, dtype=wp.int32, device=device)
        data.parameters = wp.zeros(sleep.plan.component_count, dtype=ScalarParameters, device=device)
        data.expected_dt = wp.zeros(sleep.plan.component_count, dtype=wp.float32, device=device)
        data.dirty_world = wp.ones(solver.world_count + 1, dtype=wp.int32, device=device)
        data.contact, data.invalid_contact = sleep.contact_component, sleep.invalid_contacts
        for name in (
            "sleeping",
            "counters",
            "expected_valid",
            "expected_q",
            "expected_qd",
            "expected_target_q",
            "expected_target_qd",
            "begin_q",
            "begin_target_q",
            "begin_target_qd",
            "can_sleep",
            "body_awake",
            "joint_awake",
        ):
            setattr(data, name, getattr(sleep, name))
        self.data = data
        source = ParameterSources()
        body_joint = np.full(model.body_count, -1, dtype=np.int32)
        children = model.joint_child.numpy()
        for link, body in enumerate(children):
            if body >= 0 and body_joint[body] < 0:
                body_joint[body] = link
        source.body_joint = wp.array(body_joint, dtype=wp.int32, device=device)
        for name, attribute in (
            ("parent", "joint_parent"),
            ("articulation", "joint_articulation"),
            ("parent_frame", "joint_X_p"),
            ("child_frame", "joint_X_c"),
            ("axis", "joint_axis"),
            ("body_com", "body_com"),
            ("body_flags", "body_flags"),
            ("mass", "body_mass"),
            ("ke", "joint_target_ke"),
            ("kd", "joint_target_kd"),
            ("effort", "joint_effort_limit"),
            ("gravity", "gravity"),
        ):
            setattr(source, name, getattr(model, attribute))
        source.armature = solver.R_by_size[direct]
        source.passive_k = solver._passive_spring_stiffness
        source.passive_ref = solver._passive_spring_ref
        source.damping = solver._passive_joint_damping
        source.drive_row = solver._augmented_drive_row_by_dof
        self.source = source

    def _state(self, state_in, state_aug):
        state = ComponentState()
        state.body_q = state_in.body_q
        for name, attribute in (
            ("body_q_com", "body_q_com"),
            ("joint_S", "joint_S_s"),
            ("body_v", "body_v_s"),
            ("body_a", "body_a_s"),
            ("body_f", "body_f_s"),
            ("tau", "joint_tau"),
            ("qdd", "joint_qdd"),
        ):
            setattr(state, name, getattr(state_aug, attribute))
        for name, attribute in (
            ("origin", "articulation_origin"),
            ("v_hat", "v_hat"),
            ("inverse_mass", "_diagonal_inverse_mass"),
            ("drive_K", "aug_row_K"),
        ):
            setattr(state, name, getattr(self.solver, attribute))
        return state

    def begin(self, state_in, state_aug, control, contacts, dt):
        """Produce current owned scalar state and predictor before fallback work."""
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("sleeping requires a positive finite timestep")
        sleep = self.sleep
        sleep.contact_component.zero_()
        sleep.invalid_contacts.zero_()
        if contacts is not None:
            wp.launch(
                _mark_contacts,
                dim=max(1, contacts.rigid_contact_max),
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    self.model.shape_body,
                    sleep.plan.body_component,
                ],
                outputs=[sleep.contact_component, sleep.invalid_contacts],
                device=self.model.device,
            )
        wp.launch(
            prepare_components,
            dim=len(self.data.owned),
            inputs=[
                self.data,
                self.source,
                self._state(state_in, state_aug),
                state_in.joint_q,
                state_in.joint_qd,
                control.joint_target_q,
                control.joint_target_qd,
                control.joint_f,
                state_in.body_f,
                self.solver._kinematic_dof_mask,
                dt,
                int(self.solver._fk_id_cache_source_state is not state_in),
            ],
            device=self.model.device,
        )
        self.data.dirty_world.zero_()

    def finish(self, state_in, state_aug, state_out, dt):
        """Integrate and publish owned state with the current solved response."""
        wp.launch(
            finish_components,
            dim=len(self.data.owned),
            inputs=[
                self.data,
                self.source,
                self._state(state_in, state_aug),
                state_in.joint_q,
                state_in.joint_qd,
                state_in.body_q,
                self.solver._kinematic_dof_mask,
                self.solver.v_out,
                dt,
                self.sleep.velocity_tolerance,
                self.sleep.quiet_steps,
            ],
            outputs=[state_out.joint_q, state_out.joint_qd, state_out.body_q, state_out.body_qd],
            device=self.model.device,
        )

    def notify_model_changed(self, flags):
        """Invalidate invariant descriptors only where a pose property changed."""
        if int(flags) == int(ModelFlags.JOINT_PROPERTIES):
            wp.launch(
                mark_dirty_worlds,
                dim=len(self.data.dirty_world),
                inputs=[self.sleep._changed_joint_world],
                outputs=[self.data.dirty_world],
                device=self.model.device,
            )
        else:
            self.data.dirty_world.fill_(1)
