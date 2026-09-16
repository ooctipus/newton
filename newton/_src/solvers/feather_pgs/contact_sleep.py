# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental complete scalar consumer of topology-general contact islands.

One controller owns one canonical Contacts allocation and one collision
pipeline. The caller must order collision, solve and graph replay, including
synchronizing any stream migration; concurrent use is unsupported.
"""

import numpy as np
import warp as wp

from ...sim import BodyFlags, ModelFlags
from .awake_pipeline import ComponentData, apply_sleep_grants, assess_sleep_ready
from .sleep_contact_cache import SleepContactCache, pair_is_dormant
from .sleep_contact_rows import capture_dormant_contact_forces, publish_dormant_contact_forces
from .sleep_islands import SleepIslands


@wp.kernel(enable_backward=False)
def _static_changes(
    body_component: wp.array[int],
    body_world: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    saved: wp.array[wp.transform],
    valid: wp.array[int],
    changed_world: wp.array[int],
):
    body = wp.tid()
    if body_component[body] != -1:
        return
    changed = valid[body] == 0
    for i in range(7):
        changed = changed or body_q[body][i] != saved[body][i]
    for i in range(6):
        changed = changed or body_qd[body][i] != 0.0
    if changed:
        world = body_world[body]
        if world < 0 or world >= changed_world.shape[0] - 1:
            world = changed_world.shape[0] - 1
        wp.atomic_max(changed_world, world, 1)


@wp.kernel(enable_backward=False)
def _input_wake(
    data: ComponentData,
    eligible: wp.array[int],
    q: wp.array[float],
    qd: wp.array[float],
    target: wp.array[float],
    target_velocity: wp.array[float],
    joint_force: wp.array[float],
    body_force: wp.array[wp.spatial_vector],
    kinematic: wp.array[int],
    dt: float,
    collision_q: wp.array[float],
    collision_valid: wp.array[int],
    collision_generation: wp.array[int],
    generation: wp.array[int],
    static_changed: wp.array[int],
    require_seal: int,
    wake: wp.array[int],
    changed: wp.array[int],
):
    component = wp.tid()
    if eligible[component] == 0:
        wake[component] = 1
        changed[component] = 1
        return
    dof = data.dof[component]
    body = data.body[component]
    position = q[data.coordinate[component]]
    velocity = qd[dof]
    world = data.world[component]
    global_world = data.dirty_world.shape[0] - 1
    dirty = data.dirty_world[world] != 0 or data.dirty_world[global_world] != 0
    dirty = dirty or static_changed[world] != 0 or static_changed[global_world] != 0
    modified = dirty or data.expected_valid[component] == 0
    modified = modified or position != data.expected_q[component] or velocity != data.expected_qd[component]
    allowed = data.parameters[component].allow_sleep != 0
    allowed = allowed and wp.isfinite(position) and wp.isfinite(velocity) and kinematic[dof] == 0
    # Collision may precede the first solve, including in a cold captured
    # graph. Only the solve boundary has authoritative current control/dt.
    # A later control wake closes the old island before rebuilding its rows
    # from the still-current canonical geometry.
    if require_seal == 0:
        modified = modified or target[dof] != data.expected_target_q[component]
        modified = modified or target_velocity[dof] != data.expected_target_qd[component]
        modified = modified or data.expected_dt[component] != dt
        allowed = allowed and wp.isfinite(target[dof])
        allowed = allowed and target_velocity[dof] == 0.0 and joint_force[dof] == 0.0
    force = body_force[body]
    for i in range(6):
        allowed = allowed and force[i] == 0.0
    generation_changed = collision_generation[component] != generation[0]
    if dirty or generation_changed:
        collision_valid[component] = 0
    # Ordinary hooked collision seals its new generation before begin. A new
    # generation without that seal (for example external Contacts.clear())
    # revokes an old lease even when this call does not require a pose seal.
    if generation_changed and data.sleeping[component] != 0:
        modified = True
    if require_seal != 0 and data.sleeping[component] != 0:
        modified = modified or collision_valid[component] == 0 or position != collision_q[component]
    modified = modified or not allowed
    changed[component] = int(modified)
    wake[component] = int(modified or data.sleeping[component] == 0)


@wp.kernel(enable_backward=False)
def _apply_wake(data: ComponentData, awake: wp.array[int], changed: wp.array[int]):
    component = wp.tid()
    if awake[component] != 0:
        if data.sleeping[component] != 0 or changed[component] != 0:
            data.counters[component] = 0
        data.sleeping[component] = 0


@wp.kernel(enable_backward=False)
def _contact_mask(
    count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    shape_body: wp.array[int],
    body_component: wp.array[int],
    asleep: wp.array[int],
    clear_awake: int,
    held_force: wp.array[wp.vec3],
    held_valid: wp.array[int],
    mask: wp.array[int],
):
    slot = wp.tid()
    total = count[0]
    dormant = False
    if total >= 0 and total <= mask.shape[0] and slot < total:
        dormant = pair_is_dormant(shape0[slot], shape1[slot], shape_body, body_component, asleep)
    mask[slot] = int(dormant)
    if clear_awake != 0 and not dormant:
        held_force[slot] = wp.vec3()
        held_valid[slot] = 0


@wp.kernel(enable_backward=False)
def _filter_pairs(
    count: wp.array[int],
    pairs: wp.array[wp.vec2i],
    shape_body: wp.array[int],
    body_component: wp.array[int],
    asleep: wp.array[int],
    status: wp.array[int],
):
    index = wp.tid()
    total = count[0]
    if status[0] != 0 or total < 0 or total > pairs.shape[0] or index >= total:
        return
    pair = pairs[index]
    if pair_is_dormant(pair[0], pair[1], shape_body, body_component, asleep):
        pairs[index] = wp.vec2i(-1, -1)


@wp.kernel(enable_backward=False)
def _seal_geometry(
    data: ComponentData,
    q: wp.array[float],
    generation: wp.array[int],
    status: wp.array[int],
    collision_q: wp.array[float],
    collision_valid: wp.array[int],
    collision_generation: wp.array[int],
):
    component = data.owned[wp.tid()]
    collision_q[component] = q[data.coordinate[component]]
    collision_valid[component] = int(status[0] == 0)
    collision_generation[component] = generation[0]


@wp.kernel(enable_backward=False)
def _save_static(
    body_component: wp.array[int], q: wp.array[wp.transform], saved: wp.array[wp.transform], valid: wp.array[int]
):
    body = wp.tid()
    if body_component[body] == -1:
        saved[body] = q[body]
        valid[body] = 1


@wp.kernel(enable_backward=False)
def _not_ready(ready: wp.array[int], wake: wp.array[int]):
    component = wp.tid()
    wake[component] = int(ready[component] == 0)


@wp.kernel(enable_backward=False)
def _approve(awake: wp.array[int], approved: wp.array[int]):
    component = wp.tid()
    approved[component] = int(awake[component] == 0)


@wp.kernel(enable_backward=False)
def _merge_status(first: wp.array[int], second: wp.array[int], output: wp.array[int]):
    output[0] = int(first[0] != 0 or second[0] != 0)


@wp.kernel(enable_backward=False)
def _record_status(current: wp.array[int], sticky: wp.array[int]):
    wp.atomic_or(sticky, 0, current[0])


@wp.kernel(enable_backward=False)
def _save_lease(
    roots: wp.array[int],
    approved: wp.array[int],
    status: wp.array[int],
    old_roots: wp.array[int],
    leased: wp.array[int],
):
    component = wp.tid()
    old_roots[component] = roots[component]
    leased[component] = int(status[0] == 0 and approved[component] != 0)


@wp.kernel(enable_backward=False)
def _invalidate(
    data: ComponentData,
    collision_valid: wp.array[int],
    mask: wp.array[wp.bool],
    dirty: wp.array[int],
    use_dirty: int,
):
    component = wp.tid()
    world = data.world[component]
    selected = mask.shape[0] == 0 or world < 0
    if world >= 0 and world < mask.shape[0]:
        selected = mask[world]
    if use_dirty != 0:
        selected = dirty[dirty.shape[0] - 1] != 0
        if world >= 0 and world < dirty.shape[0] - 1:
            selected = selected or dirty[world] != 0
    if selected:
        collision_valid[component] = 0
        data.sleeping[component] = 0
        data.counters[component] = 0


class ContactSleep:
    """Join collision cache, full island wake closure and scalar sleep grants."""

    @classmethod
    def create(cls, awake_pipeline):
        model = awake_pipeline.model
        if (
            model.requires_grad
            or model.particle_count
            or np.any(model.body_flags.numpy() != int(BodyFlags.DYNAMIC))
            or not awake_pipeline.sleep.enabled
        ):
            return None
        return cls(awake_pipeline)

    def __init__(self, awake_pipeline):
        self.awake = awake_pipeline
        self.solver = awake_pipeline.solver
        self.model = awake_pipeline.model
        self.sleep = awake_pipeline.sleep
        self.data = awake_pipeline.data
        self.plan = self.sleep.plan
        self.device = self.model.device
        self.islands = SleepIslands(self.plan)
        count = self.plan.component_count
        self.wake = wp.zeros(count, dtype=wp.int32, device=self.device)
        self.changed = wp.zeros_like(self.wake)
        self.ready = wp.zeros_like(self.wake)
        self.approved = wp.zeros_like(self.wake)
        self.leased = wp.zeros_like(self.wake)
        self.leased_roots = wp.array(np.arange(count, dtype=np.int32), dtype=wp.int32, device=self.device)
        self.collision_q = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.collision_valid = wp.zeros_like(self.wake)
        self.collision_generation = wp.zeros_like(self.wake)
        self.static_q = wp.empty(self.model.body_count, dtype=wp.transform, device=self.device)
        self.static_valid = wp.zeros(self.model.body_count, dtype=wp.int32, device=self.device)
        self.static_changed = wp.zeros(self.solver.world_count + 1, dtype=wp.int32, device=self.device)
        self.status = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.sticky_status = wp.zeros_like(self.status)
        self.empty_mask = wp.empty(0, dtype=wp.bool, device=self.device)
        self.contacts = None
        self.pipeline = None
        self.cache = None
        self.contact_asleep = None

    def _bind(self, contacts):
        if contacts is None:
            raise ValueError("Contact sleeping requires an explicit canonical Contacts buffer")
        if self.contacts is not None:
            if contacts is not self.contacts:
                raise ValueError("Contact sleeping is bound to one Contacts allocation")
            return
        if self.device.is_cuda and self.device.is_capturing:
            raise RuntimeError(
                "Call solver.prepare_contact_sleep(contacts) before the first collision and CUDA capture"
            )
        if contacts.device != self.device or contacts._rigid_sleep_owner is not None:
            raise ValueError("Contact sleeping requires an unowned same-device Contacts buffer")
        self.cache = SleepContactCache(contacts)
        self.contact_asleep = wp.zeros(contacts.rigid_contact_max, dtype=wp.int32, device=self.device)
        self.contacts = contacts
        contacts._rigid_sleep_owner = self

    def _classify(self, state, control, dt, *, require_seal):
        self.static_changed.zero_()
        wp.launch(
            _static_changes,
            dim=self.model.body_count,
            inputs=[
                self.plan.body_component,
                self.model.body_world,
                state.body_q,
                state.body_qd,
                self.static_q,
                self.static_valid,
                self.static_changed,
            ],
            device=self.device,
        )
        wp.launch(
            _input_wake,
            dim=self.plan.component_count,
            inputs=[
                self.data,
                self.sleep.component_eligible,
                state.joint_q,
                state.joint_qd,
                control.joint_target_q if control is not None else state.joint_qd,
                control.joint_target_qd if control is not None else state.joint_qd,
                control.joint_f if control is not None else state.joint_qd,
                state.body_f,
                self.solver._kinematic_dof_mask,
                dt,
                self.collision_q,
                self.collision_valid,
                self.collision_generation,
                self.contacts.contact_generation,
                self.static_changed,
                int(require_seal),
                self.wake,
                self.changed,
            ],
            device=self.device,
        )

    def _close(self, *, pairs=None):
        self.islands.reset()
        self.islands.union_previous(self.leased_roots, self.leased)
        if pairs is None:
            contacts = self.contacts
            self.islands.union_contacts(
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                self.model.shape_body,
                self.model.shape_world,
            )
        else:
            self.islands.union_pairs(
                pairs.broad_phase_pair_count,
                pairs.broad_phase_shape_pairs,
                self.model.shape_body,
                self.model.shape_world,
            )
        self.islands.finalize(self.wake, self.sleep.component_eligible)
        wp.launch(_record_status, dim=1, inputs=[self.islands.status, self.sticky_status], device=self.device)

    def _wake_closed(self):
        wp.launch(
            _apply_wake,
            dim=self.plan.component_count,
            inputs=[self.data, self.islands.component_awake, self.changed],
            device=self.device,
        )

    def _mask_contacts(self, asleep, *, clear_awake):
        contacts = self.contacts
        if contacts.rigid_contact_max:
            wp.launch(
                _contact_mask,
                dim=contacts.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    self.model.shape_body,
                    self.plan.body_component,
                    asleep,
                    int(clear_awake),
                    self.cache.held_force,
                    self.cache.held_valid,
                    self.contact_asleep,
                ],
                device=self.device,
            )

    def begin(self, state_in, state_aug, control, contacts, dt):
        """Close current input wake over all raw contacts and old leased membership."""
        self._bind(contacts)
        self._classify(state_in, control, dt, require_seal=False)
        self._close()
        self._wake_closed()
        self._mask_contacts(self.data.sleeping, clear_awake=True)

    def prepare_finish(self, state_in, state_aug, state_out, dt):
        """Assess complete islands, capture their solved forces, then grant frozen poses."""
        self.ready.zero_()
        wp.launch(
            assess_sleep_ready,
            dim=len(self.data.owned),
            inputs=[
                self.data,
                state_in.joint_q,
                state_in.joint_qd,
                self.solver._kinematic_dof_mask,
                self.solver.v_out,
                dt,
                self.sleep.velocity_tolerance,
                self.sleep.quiet_steps,
                self.collision_q,
                self.collision_valid,
                self.ready,
            ],
            device=self.device,
        )
        wp.launch(_not_ready, dim=self.plan.component_count, inputs=[self.ready, self.wake], device=self.device)
        self._close()
        wp.launch(
            _approve,
            dim=self.plan.component_count,
            inputs=[self.islands.component_awake, self.approved],
            device=self.device,
        )
        self._mask_contacts(self.approved, clear_awake=False)
        wp.launch(
            _merge_status, dim=1, inputs=[self.islands.status, self.cache.status, self.status], device=self.device
        )
        contacts = self.contacts
        if contacts.rigid_contact_max:
            wp.launch(
                capture_dormant_contact_forces,
                dim=contacts.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    self.contact_asleep,
                    contacts.rigid_contact_normal,
                    self.solver.contact_world,
                    self.solver.contact_slot,
                    self.solver.contact_path,
                    self.solver.impulses,
                    self.solver.constraint_count,
                    self.solver.row_type,
                    self.solver.row_parent,
                    int(self.solver.enable_contact_friction),
                    1.0 / dt,
                    self.cache.held_force,
                    self.cache.held_valid,
                    self.status,
                ],
                device=self.device,
            )
        wp.launch(
            apply_sleep_grants,
            dim=len(self.data.owned),
            inputs=[
                self.data,
                self.awake.source,
                self.awake._state(state_in, state_aug),
                state_in.joint_q,
                self.approved,
                self.status,
                self.solver.v_out,
            ],
            device=self.device,
        )
        wp.launch(
            _save_lease,
            dim=self.plan.component_count,
            inputs=[self.islands.roots, self.approved, self.status, self.leased_roots, self.leased],
            device=self.device,
        )
        wp.launch(_record_status, dim=1, inputs=[self.status, self.sticky_status], device=self.device)
        self._mask_contacts(self.data.sleeping, clear_awake=False)

    def before_collision(self, pipeline, state, contacts):
        """Validate the complete binding before any snapshot or collision mutation."""
        self._bind(contacts)
        if self.pipeline is not None and self.pipeline is not pipeline:
            raise ValueError("Contact sleeping is bound to one CollisionPipeline")
        if (
            pipeline.model is not self.model
            or pipeline.device != self.device
            or pipeline._rigid_contact_max != contacts.rigid_contact_max
            or pipeline.requires_grad
            or pipeline.deterministic
            or pipeline._matching_enabled
            or pipeline._body_pair_reducer is not None
            or pipeline._speculative_enabled
            or pipeline.narrow_phase.hydroelastic_sdf is not None
            or pipeline.enable_rigid_soft_full_surface_contact
            or self.model.particle_count
        ):
            raise ValueError("Unsupported contact-sleep collision capability")
        self.cache._check_binding(contacts)
        self.pipeline = pipeline
        self._classify(state, None, 0.0, require_seal=True)
        self.cache.snapshot(contacts)

    def after_broad_phase(self, pipeline, state, contacts):
        """Wake old islands and current broad neighbors before any narrow exclusion."""
        self._close(pairs=pipeline)
        self._wake_closed()
        self.cache.emit(contacts, self.model.shape_body, self.plan.body_component, self.data.sleeping)
        wp.launch(_record_status, dim=1, inputs=[self.cache.status, self.sticky_status], device=self.device)
        wp.launch(
            _filter_pairs,
            dim=pipeline.shape_pairs_max,
            inputs=[
                pipeline.broad_phase_pair_count,
                pipeline.broad_phase_shape_pairs,
                self.model.shape_body,
                self.plan.body_component,
                self.data.sleeping,
                self.cache.status,
            ],
            device=self.device,
        )

    def after_collision(self, pipeline, state, contacts):
        """Seal current canonical geometry only after the full rigid producer boundary."""
        wp.launch(
            _merge_status, dim=1, inputs=[self.islands.status, self.cache.status, self.status], device=self.device
        )
        wp.launch(
            _seal_geometry,
            dim=len(self.data.owned),
            inputs=[
                self.data,
                state.joint_q,
                contacts.contact_generation,
                self.status,
                self.collision_q,
                self.collision_valid,
                self.collision_generation,
            ],
            device=self.device,
        )
        wp.launch(
            _save_static,
            dim=self.model.body_count,
            inputs=[self.plan.body_component, state.body_q, self.static_q, self.static_valid],
            device=self.device,
        )

    def notify_model_changed(self, flags):
        """Invalidate actual changed pose worlds, or all worlds for other model changes."""
        wp.launch(
            _invalidate,
            dim=self.plan.component_count,
            inputs=[
                self.data,
                self.collision_valid,
                self.empty_mask,
                self.data.dirty_world,
                int(int(flags) == int(ModelFlags.JOINT_PROPERTIES)),
            ],
            device=self.device,
        )

    def invalidate(self, world_mask=None):
        """Revoke reset-world geometry while retaining old membership for wake closure."""
        if world_mask is not None and world_mask.shape != (self.solver.world_count,):
            raise ValueError("Contact-sleep reset mask must have one entry per world")
        wp.launch(
            _invalidate,
            dim=self.plan.component_count,
            inputs=[
                self.data,
                self.collision_valid,
                self.empty_mask if world_mask is None else world_mask,
                self.data.dirty_world,
                0,
            ],
            device=self.device,
        )

    def publish_contacts(self, contacts):
        """Overwrite held dormant linear forces after ordinary export, before spatial packing."""
        if contacts is not self.contacts:
            return False
        if contacts.rigid_contact_max:
            wp.launch(
                publish_dormant_contact_forces,
                dim=contacts.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    self.contact_asleep,
                    self.cache.held_force,
                    self.cache.held_valid,
                    contacts.rigid_contact_force,
                ],
                device=self.device,
            )
        return True

    def check_status(self):
        """Reject any recorded primitive failure at an explicit untimed boundary."""
        if self.device.is_cuda and self.device.is_capturing:
            raise RuntimeError("Contact-sleep status checks must run outside CUDA capture")
        if self.cache is not None:
            self.cache.check_capacity(self.contacts)
        status = int(self.sticky_status.numpy()[0])
        if status != 0:
            raise RuntimeError(f"Contact sleeping recorded invalid island/cache/force state: status={status}")
