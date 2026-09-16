# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Conservative contact-free component sleeping for supported state consumers.

Topology is task-independent. The first consumer capability is deliberately
limited to complete one-body prismatic components. Contacted, driven and
unsupported components always retain the original solver path. Constant PD
position targets may settle; imposed target motion always remains awake.
"""

import math

import numpy as np
import warp as wp

from ...sim import BodyFlags, ModelFlags
from .sleep_topology import build_sleep_topology

# A quiet velocity must also remain quiet under the current solved acceleration
# over this physical horizon [s], independently of a small simulation timestep.
_REST_HORIZON = wp.constant(0.05)


@wp.kernel(enable_backward=False)
def _mark_contacts(
    count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    shape_body: wp.array[int],
    body_component: wp.array[int],
    contact_component: wp.array[int],
    invalid_contacts: wp.array[int],
):
    index = wp.tid()
    total = count[0]
    if total < 0 or total > shape0.shape[0] or total > shape1.shape[0]:
        wp.atomic_max(invalid_contacts, 0, 1)
        return
    if index >= total:
        return
    for side in range(2):
        shape = shape0[index]
        if side == 1:
            shape = shape1[index]
        if shape < 0 or shape >= shape_body.shape[0]:
            wp.atomic_max(invalid_contacts, 0, 1)
        else:
            body = shape_body[shape]
            if body >= body_component.shape[0] or body < -1:
                wp.atomic_max(invalid_contacts, 0, 1)
            elif body >= 0:
                component = body_component[body]
                if component >= 0:
                    wp.atomic_max(contact_component, component, 1)


@wp.kernel(enable_backward=False)
def _begin_components(
    eligible: wp.array[int],
    component_body: wp.array[int],
    component_joint: wp.array[int],
    component_q: wp.array[int],
    component_dof: wp.array[int],
    contact_component: wp.array[int],
    invalid_contacts: wp.array[int],
    q: wp.array[float],
    qd: wp.array[float],
    joint_f: wp.array[float],
    body_f: wp.array[wp.spatial_vector],
    target_ke: wp.array[float],
    target_kd: wp.array[float],
    target_q: wp.array[float],
    target_qd: wp.array[float],
    target_q_start: wp.array[int],
    body_flags: wp.array[int],
    joint_parent: wp.array[int],
    joint_articulation: wp.array[int],
    axis: wp.array[wp.vec3],
    mass: wp.array[float],
    gravity: wp.array[wp.vec3],
    spring: wp.array[float],
    spring_ref: wp.array[float],
    damping: wp.array[float],
    expected_valid: wp.array[int],
    expected_q: wp.array[float],
    expected_qd: wp.array[float],
    expected_target_q: wp.array[float],
    expected_target_qd: wp.array[float],
    sleeping: wp.array[int],
    counters: wp.array[int],
    can_sleep: wp.array[int],
    begin_q: wp.array[float],
    begin_target_q: wp.array[float],
    begin_target_qd: wp.array[float],
    fk_valid: wp.array[int],
    mass_requested: wp.array[int],
):
    component = wp.tid()
    can_sleep[component] = 0
    if eligible[component] == 0:
        sleeping[component] = 0
        counters[component] = 0
        return
    body = component_body[component]
    dof = component_dof[component]
    position = q[component_q[component]]
    velocity = qd[dof]
    link = component_joint[component]
    position_target = target_q[target_q_start[link]]
    velocity_target = target_qd[dof]
    begin_q[component] = position
    begin_target_q[component] = position_target
    begin_target_qd[component] = velocity_target
    unsupported = invalid_contacts[0] != 0 or contact_component[component] != 0
    unsupported = unsupported or not wp.isfinite(position) or not wp.isfinite(velocity)
    # USD passive springs may arrive as constant PD drives. Preserve their
    # complete force law while awake, then lease only an unchanged rest target.
    unsupported = unsupported or not wp.isfinite(target_ke[dof]) or target_ke[dof] < 0.0
    unsupported = unsupported or not wp.isfinite(target_kd[dof]) or target_kd[dof] < 0.0
    unsupported = unsupported or not wp.isfinite(position_target) or velocity_target != 0.0
    unsupported = unsupported or joint_f[dof] != 0.0
    unsupported = unsupported or body_flags[body] != BodyFlags.DYNAMIC
    parent = joint_parent[link]
    if parent >= 0:
        unsupported = unsupported or body_flags[parent] != BodyFlags.DYNAMIC
    unsupported = unsupported or not wp.isfinite(mass[body]) or mass[body] <= 0.0
    unsupported = unsupported or not wp.isfinite(spring[dof]) or not wp.isfinite(spring_ref[dof])
    unsupported = unsupported or not wp.isfinite(damping[dof])
    direction = axis[dof]
    for element in range(3):
        unsupported = unsupported or not wp.isfinite(direction[element])
        unsupported = unsupported or not wp.isfinite(gravity[0][element])
    axis_length = wp.length(direction)
    unsupported = unsupported or not wp.isfinite(axis_length) or axis_length <= 0.0
    force = body_f[body]
    for element in range(6):
        unsupported = unsupported or force[element] != 0.0
    # NaN gains/forces also compare unequal to zero, so cannot be hidden by a
    # sleeping return. Compare values, not State identity: A/B and in-place
    # stepping both share the same previous-output contract.
    authored = expected_valid[component] == 0
    target_changed = authored
    if not authored:
        authored = position != expected_q[component] or velocity != expected_qd[component]
        target_changed = (
            position_target != expected_target_q[component] or velocity_target != expected_target_qd[component]
        )
    if unsupported or authored or target_changed:
        sleeping[component] = 0
        counters[component] = 0
    if authored:
        # State writes need fresh geometry as well as an awake component.
        articulation = joint_articulation[link]
        if articulation >= 0:
            wp.atomic_min(fk_valid, articulation, 0)
        # Mass refresh is a solver-wide scalar request, not a per-articulation array.
        wp.atomic_max(mass_requested, 0, 1)
    if not unsupported:
        can_sleep[component] = 1


@wp.kernel(enable_backward=False)
def _publish_awake_masks(
    body_component: wp.array[int],
    joint_component: wp.array[int],
    sleeping: wp.array[int],
    body_awake: wp.array[int],
    joint_awake: wp.array[int],
):
    index = wp.tid()
    if index < body_component.shape[0]:
        component = body_component[index]
        awake = int(1)
        if component >= 0:
            awake = 1 - sleeping[component]
        body_awake[index] = awake
    if index < joint_component.shape[0]:
        component = joint_component[index]
        awake = int(1)
        if component >= 0:
            awake = 1 - sleeping[component]
        joint_awake[index] = awake


@wp.kernel(enable_backward=False)
def _finish_components(
    eligible: wp.array[int],
    component_q: wp.array[int],
    component_dof: wp.array[int],
    axis: wp.array[wp.vec3],
    can_sleep: wp.array[int],
    begin_q: wp.array[float],
    begin_target_q: wp.array[float],
    begin_target_qd: wp.array[float],
    q: wp.array[float],
    qd: wp.array[float],
    dt: float,
    velocity_tolerance: float,
    quiet_steps: int,
    sleeping: wp.array[int],
    counters: wp.array[int],
    expected_valid: wp.array[int],
    expected_q: wp.array[float],
    expected_qd: wp.array[float],
    expected_target_q: wp.array[float],
    expected_target_qd: wp.array[float],
    qdd: wp.array[float],
    velocity_out: wp.array[float],
):
    component = wp.tid()
    if eligible[component] == 0:
        return
    coordinate = component_q[component]
    dof = component_dof[component]
    position = q[coordinate]
    velocity = qd[dof]
    scale = wp.length(axis[dof])
    speed = wp.abs(velocity) * scale
    acceleration = wp.abs(qdd[dof]) * scale
    displacement = wp.abs(position - begin_q[component]) * scale
    quiet = can_sleep[component] != 0
    quiet = quiet and wp.isfinite(position) and wp.isfinite(velocity)
    quiet = quiet and wp.isfinite(speed) and wp.isfinite(displacement)
    quiet = quiet and speed <= velocity_tolerance and displacement <= velocity_tolerance * dt
    quiet = quiet and wp.isfinite(acceleration) and acceleration * _REST_HORIZON <= velocity_tolerance
    if quiet:
        counters[component] = wp.min(counters[component] + 1, quiet_steps)
        if counters[component] >= quiet_steps:
            sleeping[component] = 1
            qd[dof] = 0.0
            qdd[dof] = 0.0
            velocity_out[dof] = 0.0
            velocity = 0.0
    else:
        counters[component] = 0
        sleeping[component] = 0
    expected_q[component] = position
    expected_qd[component] = velocity
    expected_target_q[component] = begin_target_q[component]
    expected_target_qd[component] = begin_target_qd[component]
    expected_valid[component] = 1
    # Deliberately do not publish the next masks here: a newly sleeping body
    # must publish its final pose and its now-zero velocity in this Stage7.


@wp.kernel(enable_backward=False)
def _mark_changed_joint_worlds(
    joint_world: wp.array[int],
    q_start: wp.array[int],
    q: wp.array[float],
    parent_frame: wp.array[wp.transform],
    child_frame: wp.array[wp.transform],
    saved_q: wp.array[float],
    saved_parent_frame: wp.array[wp.transform],
    saved_child_frame: wp.array[wp.transform],
    changed_world: wp.array[int],
):
    joint = wp.tid()
    parent, child = parent_frame[joint], child_frame[joint]
    old_parent, old_child = saved_parent_frame[joint], saved_child_frame[joint]
    changed = bool(False)
    for element in range(7):
        changed = changed or parent[element] != old_parent[element] or child[element] != old_child[element]
        changed = changed or not wp.isfinite(parent[element]) or not wp.isfinite(child[element])
    for coordinate in range(q_start[joint], q_start[joint + 1]):
        value = q[coordinate]
        changed = changed or value != saved_q[coordinate] or not wp.isfinite(value)
        saved_q[coordinate] = value
    saved_parent_frame[joint] = parent
    saved_child_frame[joint] = child
    if changed:
        world = joint_world[joint]
        global_slot = changed_world.shape[0] - 1
        if world < 0 or world >= global_slot:
            world = global_slot
        wp.atomic_max(changed_world, world, 1)


@wp.kernel(enable_backward=False)
def _invalidate_changed_joint_worlds(
    component_world: wp.array[int],
    changed_world: wp.array[int],
    sleeping: wp.array[int],
    counters: wp.array[int],
    expected_valid: wp.array[int],
):
    component = wp.tid()
    world = component_world[component]
    global_slot = changed_world.shape[0] - 1
    changed = changed_world[global_slot] != 0
    if world >= 0 and world < global_slot:
        changed = changed or changed_world[world] != 0
    else:
        for index in range(global_slot):
            changed = changed or changed_world[index] != 0
    if changed:
        sleeping[component] = 0
        counters[component] = 0
        expected_valid[component] = 0


@wp.kernel(enable_backward=False)
def _invalidate_components(
    component_world: wp.array[int],
    world_mask: wp.array[wp.bool],
    masked: int,
    sleeping: wp.array[int],
    counters: wp.array[int],
    expected_valid: wp.array[int],
):
    component = wp.tid()
    invalidate = masked == 0
    if masked != 0:
        world = component_world[component]
        if world >= 0 and world < world_mask.shape[0]:
            invalidate = world_mask[world] != 0
        else:
            # Global components conservatively belong to any selected world.
            for index in range(world_mask.shape[0]):
                invalidate = invalidate or world_mask[index] != 0
    if invalidate:
        sleeping[component] = 0
        counters[component] = 0
        expected_valid[component] = 0


class SleepController:
    """Own device sleep leases for complete supported dynamic components."""

    def __init__(self, solver, *, velocity_tolerance: float = 1.0e-5, quiet_steps: int = 16):
        if not math.isfinite(velocity_tolerance) or velocity_tolerance < 0.0:
            raise ValueError("sleep velocity tolerance must be finite and nonnegative")
        if isinstance(quiet_steps, bool) or int(quiet_steps) != quiet_steps or quiet_steps < 1:
            raise ValueError("sleep quiet_steps must be a positive integer")
        self.solver = solver
        self.model = model = solver.model
        self.velocity_tolerance = float(velocity_tolerance)
        self.quiet_steps = int(quiet_steps)
        self.enabled = False
        self.plan = None
        self.body_awake = None
        self.joint_awake = None
        replacing_owner = any(
            getattr(solver, name, None) is not None
            for name in (
                "_kinetic_world",
                "_joint_world",
                "_world_scan_publication",
                "_g1_kinetic_state",
                "_franka_kinetic_state",
            )
        )
        if (
            not getattr(solver, "_prismatic_linear_state", False)
            or not solver._async_augmented_drives
            or solver.drive_mode != "augmented"
            or solver.pgs_velocity_iterations > 0
            or solver.pgs_warmstart
            or model.particle_count
            or model.requires_grad
            or replacing_owner
        ):
            return
        self.plan = plan = build_sleep_topology(solver)
        count = plan.component_count
        eligible = np.array(plan.component_eligible_host, dtype=np.int32, copy=True)
        body = np.full(count, -1, dtype=np.int32)
        joint = np.full(count, -1, dtype=np.int32)
        coordinate = np.full(count, -1, dtype=np.int32)
        dof = np.full(count, -1, dtype=np.int32)
        publication = getattr(solver, "_prismatic_publication", None)
        if not getattr(solver, "_prismatic_linear_state", False) or publication is None:
            eligible.fill(0)
        else:
            body_joint = publication.body_joint.numpy()
            q_start = model.joint_q_start.numpy()
            qd_start = model.joint_qd_start.numpy()
            target_start = model.joint_target_q_start.numpy()
            body_count = np.bincount(plan.body_component_host[plan.body_component_host >= 0], minlength=count)
            joint_count = np.bincount(plan.joint_component_host[plan.joint_component_host >= 0], minlength=count)
            for index, component in enumerate(plan.body_component_host):
                if component >= 0:
                    body[component] = index
            for index, component in enumerate(plan.joint_component_host):
                if component >= 0:
                    joint[component] = index
            for component in range(count):
                child = body[component]
                link = joint[component]
                supported = body_count[component] == 1 and joint_count[component] == 1
                if supported:
                    supported = body_joint[child] == link
                    supported = supported and q_start[link + 1] - q_start[link] == 1
                    supported = supported and qd_start[link + 1] - qd_start[link] == 1
                    # The retained asynchronous drive consumer indexes its
                    # position target by DOF. Do not hide its layout mismatch.
                    supported = supported and target_start[link] == qd_start[link]
                if not supported:
                    eligible[component] = 0
                else:
                    coordinate[component] = q_start[link]
                    dof[component] = qd_start[link]
        device = model.device
        self.enabled = bool(np.any(eligible))
        self.component_eligible = wp.array(eligible, dtype=wp.int32, device=device)
        self.component_body = wp.array(body, dtype=wp.int32, device=device)
        self.component_joint = wp.array(joint, dtype=wp.int32, device=device)
        self.component_q = wp.array(coordinate, dtype=wp.int32, device=device)
        self.component_dof = wp.array(dof, dtype=wp.int32, device=device)
        self.sleeping = wp.zeros(count, dtype=wp.int32, device=device)
        self.counters = wp.zeros(count, dtype=wp.int32, device=device)
        self.expected_valid = wp.zeros(count, dtype=wp.int32, device=device)
        self.expected_q = wp.zeros(count, dtype=wp.float32, device=device)
        self.expected_qd = wp.zeros(count, dtype=wp.float32, device=device)
        self.expected_target_q = wp.zeros(count, dtype=wp.float32, device=device)
        self.expected_target_qd = wp.zeros(count, dtype=wp.float32, device=device)
        self.begin_q = wp.zeros(count, dtype=wp.float32, device=device)
        self.begin_target_q = wp.zeros(count, dtype=wp.float32, device=device)
        self.begin_target_qd = wp.zeros(count, dtype=wp.float32, device=device)
        self.can_sleep = wp.zeros(count, dtype=wp.int32, device=device)
        self.contact_component = wp.zeros(count, dtype=wp.int32, device=device)
        self.invalid_contacts = wp.zeros(1, dtype=wp.int32, device=device)
        self.body_awake = wp.ones(model.body_count, dtype=wp.int32, device=device)
        self.joint_awake = wp.ones(model.joint_count, dtype=wp.int32, device=device)
        self._empty_mask = wp.empty(0, dtype=wp.bool, device=device)
        # JOINT_PROPERTIES has no world mask in the public API. A partial
        # fixed-root reset must not erase every other world's quiet history.
        # Include the immobile root joints, which have no dynamic component.
        joint_articulation = model.joint_articulation.numpy()
        joint_world = np.full(model.joint_count, -1, dtype=np.int32)
        valid = (joint_articulation >= 0) & (joint_articulation < model.articulation_count)
        art_world = (
            model.articulation_world.numpy()
            if model.articulation_world is not None
            else solver._model_plan.articulation_world
        )
        joint_world[valid] = art_world[joint_articulation[valid]]
        self._joint_world = wp.array(joint_world, dtype=wp.int32, device=device)
        self._saved_model_q = wp.clone(model.joint_q)
        self._saved_parent_frame = wp.clone(model.joint_X_p)
        self._saved_child_frame = wp.clone(model.joint_X_c)
        self._changed_joint_world = wp.zeros(solver.world_count + 1, dtype=wp.int32, device=device)

    def _publish_masks(self) -> None:
        wp.launch(
            _publish_awake_masks,
            dim=max(self.model.body_count, self.model.joint_count),
            inputs=[self.plan.body_component, self.plan.joint_component, self.sleeping],
            outputs=[self.body_awake, self.joint_awake],
            device=self.model.device,
        )

    def begin(self, state_in, control, contacts, dt: float) -> None:
        """Wake current input/contact dependencies before skipped producers."""
        if not self.enabled:
            return
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("sleeping requires a positive finite timestep")
        model = self.model
        self.contact_component.zero_()
        self.invalid_contacts.zero_()
        if contacts is not None:
            wp.launch(
                _mark_contacts,
                dim=max(1, contacts.rigid_contact_max),
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    model.shape_body,
                    self.plan.body_component,
                ],
                outputs=[self.contact_component, self.invalid_contacts],
                device=model.device,
            )
        wp.launch(
            _begin_components,
            dim=self.plan.component_count,
            inputs=[
                self.component_eligible,
                self.component_body,
                self.component_joint,
                self.component_q,
                self.component_dof,
                self.contact_component,
                self.invalid_contacts,
                state_in.joint_q,
                state_in.joint_qd,
                control.joint_f,
                state_in.body_f,
                model.joint_target_ke,
                model.joint_target_kd,
                control.joint_target_q,
                control.joint_target_qd,
                model.joint_target_q_start,
                model.body_flags,
                model.joint_parent,
                model.joint_articulation,
                model.joint_axis,
                model.body_mass,
                model.gravity,
                self.solver._passive_spring_stiffness,
                self.solver._passive_spring_ref,
                self.solver._passive_joint_damping,
                self.expected_valid,
                self.expected_q,
                self.expected_qd,
                self.expected_target_q,
                self.expected_target_qd,
            ],
            outputs=[
                self.sleeping,
                self.counters,
                self.can_sleep,
                self.begin_q,
                self.begin_target_q,
                self.begin_target_qd,
                self.solver._fk_id_cache_valid,
                self.solver._mass_update_requested,
            ],
            device=model.device,
        )
        self._publish_masks()

    def finish(self, state_in, state_out, dt: float) -> None:
        """Record solved quiet motion and grant leases before final publication."""
        del state_in  # begin_q survives in-place integration of this same State.
        if not self.enabled:
            return
        wp.launch(
            _finish_components,
            dim=self.plan.component_count,
            inputs=[
                self.component_eligible,
                self.component_q,
                self.component_dof,
                self.model.joint_axis,
                self.can_sleep,
                self.begin_q,
                self.begin_target_q,
                self.begin_target_qd,
                state_out.joint_q,
                state_out.joint_qd,
                dt,
                self.velocity_tolerance,
                self.quiet_steps,
            ],
            outputs=[
                self.sleeping,
                self.counters,
                self.expected_valid,
                self.expected_q,
                self.expected_qd,
                self.expected_target_q,
                self.expected_target_qd,
                getattr(self.solver, "_last_debug_state_aug", self.solver).joint_qdd,
                self.solver.v_out,
            ],
            device=self.model.device,
        )

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Wake actual changed worlds for pose-only model notifications.

        Other properties remain a conservative global wake. These snapshots
        cover the complete documented JOINT_PROPERTIES set, not task-specific
        reset assumptions. State coordinate writes are also checked by begin.
        """
        if not self.enabled:
            return
        if flags & ModelFlags.JOINT_PROPERTIES:
            self._changed_joint_world.zero_()
            wp.launch(
                _mark_changed_joint_worlds,
                dim=self.model.joint_count,
                inputs=[
                    self._joint_world,
                    self.model.joint_q_start,
                    self.model.joint_q,
                    self.model.joint_X_p,
                    self.model.joint_X_c,
                ],
                outputs=[
                    self._saved_model_q,
                    self._saved_parent_frame,
                    self._saved_child_frame,
                    self._changed_joint_world,
                ],
                device=self.model.device,
            )
        if int(flags) != int(ModelFlags.JOINT_PROPERTIES):
            self.invalidate()
            return
        wp.launch(
            _invalidate_changed_joint_worlds,
            dim=self.plan.component_count,
            inputs=[self.plan.component_world, self._changed_joint_world],
            outputs=[self.sleeping, self.counters, self.expected_valid],
            device=self.model.device,
        )
        self._publish_masks()

    def invalidate(self, world_mask=None) -> None:
        """Wake reset/changed-model components without changing authored state."""
        if not self.enabled:
            return
        if world_mask is not None and world_mask.shape != (self.solver.world_count,):
            raise ValueError("sleep reset mask must have one entry per solver world")
        wp.launch(
            _invalidate_components,
            dim=self.plan.component_count,
            inputs=[
                self.plan.component_world,
                self._empty_mask if world_mask is None else world_mask,
                int(world_mask is not None),
            ],
            outputs=[self.sleeping, self.counters, self.expected_valid],
            device=self.model.device,
        )
        self._publish_masks()
