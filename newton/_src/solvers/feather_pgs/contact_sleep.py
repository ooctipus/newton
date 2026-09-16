# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Singleton/static-contact sleeping around complete scalar owners.

Every dynamic-dynamic broad candidate vetoes both endpoints for the complete
collision generation. Coupled components never acquire leases: no contact
graph or previous-island closure is required. The caller orders collision,
solve and graph replay, including synchronized stream migration.
"""

import numpy as np
import warp as wp

from ...sim import BodyFlags, ModelFlags
from .awake_pipeline import ComponentData
from .sleep_contact_cache import SleepContactCache, pair_is_dormant
from .sleep_contact_rows import capture_dormant_contact_forces, publish_dormant_contact_forces
from .sleep_islands import _shape_component


@wp.kernel(enable_backward=False)
def _static_changes(
    indices: wp.array[int],
    body_world: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    saved: wp.array[wp.transform],
    valid: wp.array[int],
    changed_world: wp.array[int],
):
    index = wp.tid()
    body = indices[index]
    changed = valid[index] == 0
    for i in range(7):
        changed = changed or body_q[body][i] != saved[index][i]
    for i in range(6):
        changed = changed or body_qd[body][i] != 0.0
    if changed:
        world = body_world[body]
        if world < 0 or world >= changed_world.shape[0] - 1:
            world = changed_world.shape[0] - 1
        wp.atomic_max(changed_world, world, 1)


@wp.kernel(enable_backward=False)
def _collision_state(
    data: ComponentData,
    eligible: wp.array[int],
    q: wp.array[float],
    qd: wp.array[float],
    body_force: wp.array[wp.spatial_vector],
    kinematic: wp.array[int],
):
    """Start one generation, checking geometry before any cached exclusion."""
    component = wp.tid()
    if component == 0:
        # snapshot resets its per-collision status after this kernel. Preserve
        # any earlier cache failure for the untimed sticky boundary check.
        wp.atomic_or(data.contact_status, 0, data.cache_status[0])
    data.contact[component] = 0
    data.contact_blocked[component] = int(eligible[component] == 0)
    if eligible[component] == 0:
        data.sleeping[component] = 0
        return
    dof = data.dof[component]
    position = q[data.coordinate[component]]
    velocity = qd[dof]
    world = data.world[component]
    global_world = data.dirty_world.shape[0] - 1
    dirty = data.dirty_world[world] != 0 or data.dirty_world[global_world] != 0
    dirty = dirty or data.static_changed[world] != 0 or data.static_changed[global_world] != 0
    generation_changed = data.collision_generation[component] != data.contact_generation[0]
    if dirty or generation_changed:
        data.collision_valid[component] = 0
    changed = dirty or data.expected_valid[component] == 0
    changed = changed or position != data.expected_q[component] or velocity != data.expected_qd[component]
    allowed = data.parameters[component].allow_sleep != 0 and kinematic[dof] == 0
    allowed = allowed and wp.isfinite(position) and wp.isfinite(velocity)
    force = body_force[data.body[component]]
    for i in range(6):
        allowed = allowed and force[i] == 0.0
    if data.sleeping[component] != 0:
        changed = changed or data.collision_valid[component] == 0 or position != data.collision_q[component]
    if changed or not allowed or data.contact_status[0] != 0 or data.cache_status[0] != 0:
        data.sleeping[component] = 0
        data.counters[component] = 0


@wp.func
def _pair_components(
    first: int,
    second: int,
    shape_body: wp.array[int],
    shape_world: wp.array[int],
    body_component: wp.array[int],
    data: ComponentData,
):
    a = _shape_component(first, shape_body, body_component, data.sleeping.shape[0])
    b = _shape_component(second, shape_body, body_component, data.sleeping.shape[0])
    valid = a != -2 and b != -2
    if valid:
        wa = shape_world[first]
        wb = shape_world[second]
        valid = wa >= -1 and wb >= -1 and (wa < 0 or wb < 0 or wa == wb)
        if a >= 0:
            valid = valid and data.world[a] >= 0 and wa == data.world[a]
        if b >= 0:
            valid = valid and data.world[b] >= 0 and wb == data.world[b]
    if not valid:
        wp.atomic_or(data.contact_status, 0, 2)
        a = -2
        b = -2
    return a, b


@wp.kernel(enable_backward=False)
def _block_dynamic_pairs(
    count: wp.array[int],
    pairs: wp.array[wp.vec2i],
    shape_body: wp.array[int],
    shape_world: wp.array[int],
    body_component: wp.array[int],
    data: ComponentData,
):
    index = wp.tid()
    total = count[0]
    if total < 0 or total > pairs.shape[0]:
        if index == 0:
            wp.atomic_or(data.contact_status, 0, 1)
        return
    if index < total:
        pair = pairs[index]
        a, b = _pair_components(pair[0], pair[1], shape_body, shape_world, body_component, data)
        if a >= 0 and b >= 0:
            wp.atomic_max(data.contact_blocked, a, 1)
            wp.atomic_max(data.contact_blocked, b, 1)


@wp.kernel(enable_backward=False)
def _apply_collision_veto(data: ComponentData):
    component = wp.tid()
    if data.contact_blocked[component] != 0 or data.contact_status[0] != 0 or data.cache_status[0] != 0:
        data.sleeping[component] = 0
        data.counters[component] = 0


@wp.kernel(enable_backward=False)
def _filter_pairs(
    count: wp.array[int],
    pairs: wp.array[wp.vec2i],
    shape_body: wp.array[int],
    body_component: wp.array[int],
    data: ComponentData,
):
    index = wp.tid()
    total = count[0]
    if data.contact_status[0] != 0 or data.cache_status[0] != 0:
        return
    if total < 0 or total > pairs.shape[0] or index >= total:
        return
    pair = pairs[index]
    if pair_is_dormant(pair[0], pair[1], shape_body, body_component, data.sleeping):
        pairs[index] = wp.vec2i(-1, -1)


@wp.kernel(enable_backward=False)
def _contact_incidence(
    count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    shape_body: wp.array[int],
    shape_world: wp.array[int],
    body_component: wp.array[int],
    data: ComponentData,
):
    """Build incidence once after cached and fresh geometry are complete."""
    index = wp.tid()
    total = count[0]
    if total < 0 or total > shape0.shape[0] or total > shape1.shape[0]:
        if index == 0:
            wp.atomic_or(data.contact_status, 0, 1)
            data.invalid_contact[0] = 1
        return
    if index < total:
        a, b = _pair_components(shape0[index], shape1[index], shape_body, shape_world, body_component, data)
        if a >= 0:
            wp.atomic_max(data.contact, a, 1)
        if b >= 0:
            wp.atomic_max(data.contact, b, 1)
        if a >= 0 and b >= 0:
            wp.atomic_max(data.contact_blocked, a, 1)
            wp.atomic_max(data.contact_blocked, b, 1)


@wp.kernel(enable_backward=False)
def _seal_geometry(data: ComponentData, q: wp.array[float]):
    component = data.owned[wp.tid()]
    valid = data.contact_status[0] == 0 and data.cache_status[0] == 0
    data.collision_q[component] = q[data.coordinate[component]]
    data.collision_valid[component] = int(valid)
    data.collision_generation[component] = data.contact_generation[0]
    if not valid or data.contact_blocked[component] != 0:
        data.sleeping[component] = 0
        data.counters[component] = 0


@wp.kernel(enable_backward=False)
def _save_static(
    indices: wp.array[int], q: wp.array[wp.transform], saved: wp.array[wp.transform], valid: wp.array[int]
):
    index = wp.tid()
    saved[index] = q[indices[index]]
    valid[index] = 1


@wp.kernel(enable_backward=False)
def _invalidate(data: ComponentData, mask: wp.array[wp.bool], use_dirty: int):
    component = wp.tid()
    world = data.world[component]
    selected = mask.shape[0] == 0 or world < 0
    if world >= 0 and world < mask.shape[0]:
        selected = mask[world]
    if use_dirty != 0:
        selected = data.dirty_world[data.dirty_world.shape[0] - 1] != 0
        if world >= 0 and world < data.dirty_world.shape[0] - 1:
            selected = selected or data.dirty_world[world] != 0
    if selected:
        # SleepController owns reset of lease/counters/expected state; prepare
        # and collision consume the existing dirty-world notification owner.
        data.collision_valid[component] = 0


class ContactSleep:
    """Own singleton/static contact generations, not coupled sleeping islands."""

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
        # Consumer eligibility already requires one body and one prismatic
        # joint per mechanical component, with complete scalar owner coverage.
        return cls(awake_pipeline)

    def __init__(self, awake_pipeline):
        self.awake = awake_pipeline
        self.solver = awake_pipeline.solver
        self.model = awake_pipeline.model
        self.sleep = awake_pipeline.sleep
        self.data = awake_pipeline.data
        self.plan = self.sleep.plan
        self.device = self.model.device
        count = self.plan.component_count
        self.collision_q = wp.zeros(count, dtype=wp.float32, device=self.device)
        self.collision_valid = wp.zeros(count, dtype=wp.int32, device=self.device)
        self.collision_generation = wp.zeros_like(self.collision_valid)
        self.collision_blocked = wp.zeros_like(self.collision_valid)
        static = np.flatnonzero(self.plan.body_component_host == -1).astype(np.int32)
        self.static_indices = wp.array(static, dtype=wp.int32, device=self.device)
        self.static_q = wp.empty(len(static), dtype=wp.transform, device=self.device)
        self.static_valid = wp.zeros(len(static), dtype=wp.int32, device=self.device)
        self.static_changed = wp.zeros(self.solver.world_count + 1, dtype=wp.int32, device=self.device)
        self.status = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.sticky_status = self.status
        self.empty_mask = wp.empty(0, dtype=wp.bool, device=self.device)
        self.data.collision_q = self.collision_q
        self.data.collision_valid = self.collision_valid
        self.data.collision_generation = self.collision_generation
        self.data.contact_blocked = self.collision_blocked
        self.data.contact_status = self.status
        self.data.static_changed = self.static_changed
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
        self.data.contact_generation = contacts.contact_generation
        self.data.cache_status = self.cache.status
        contacts._rigid_sleep_owner = self

    def _check_static(self, state):
        if len(self.static_indices):
            self.static_changed.zero_()
            wp.launch(
                _static_changes,
                dim=len(self.static_indices),
                inputs=[
                    self.static_indices,
                    self.model.body_world,
                    state.body_q,
                    state.body_qd,
                    self.static_q,
                    self.static_valid,
                    self.static_changed,
                ],
                device=self.device,
            )

    def begin(self, state_in, state_aug, control, contacts, dt):
        """Check static skeletons; existing prepare owns all scalar input wakes."""
        self._bind(contacts)
        self._check_static(state_in)

    def prepare_finish(self, state_in, state_aug, state_out, dt):
        """Capture current scalar/static forces before the fused finish grants."""
        contacts = self.contacts
        if contacts.rigid_contact_max:
            wp.launch(
                capture_dormant_contact_forces,
                dim=contacts.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    self.plan.body_component,
                    self.sleep.component_eligible,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    self.model.shape_body,
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

    def before_collision(self, pipeline, state, contacts):
        """Validate and snapshot before clearing the canonical contact counter."""
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
        self._check_static(state)
        wp.launch(
            _collision_state,
            dim=self.plan.component_count,
            inputs=[
                self.data,
                self.sleep.component_eligible,
                state.joint_q,
                state.joint_qd,
                state.body_f,
                self.solver._kinematic_dof_mask,
            ],
            device=self.device,
        )
        self.cache.snapshot(contacts)

    def after_broad_phase(self, pipeline, state, contacts):
        """Veto both dynamic endpoints before cache emission or pair exclusion."""
        wp.launch(
            _block_dynamic_pairs,
            dim=max(1, pipeline.shape_pairs_max),
            inputs=[
                pipeline.broad_phase_pair_count,
                pipeline.broad_phase_shape_pairs,
                self.model.shape_body,
                self.model.shape_world,
                self.plan.body_component,
                self.data,
            ],
            device=self.device,
        )
        wp.launch(_apply_collision_veto, dim=self.plan.component_count, inputs=[self.data], device=self.device)
        self.cache.emit(contacts, self.model.shape_body, self.plan.body_component, self.data.sleeping)
        wp.launch(
            _filter_pairs,
            dim=max(1, pipeline.shape_pairs_max),
            inputs=[
                pipeline.broad_phase_pair_count,
                pipeline.broad_phase_shape_pairs,
                self.model.shape_body,
                self.plan.body_component,
                self.data,
            ],
            device=self.device,
        )

    def after_collision(self, pipeline, state, contacts):
        """Build incidence and seal geometry once for this collision generation."""
        wp.launch(
            _contact_incidence,
            dim=max(1, contacts.rigid_contact_max),
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                self.model.shape_body,
                self.model.shape_world,
                self.plan.body_component,
                self.data,
            ],
            device=self.device,
        )
        wp.launch(_seal_geometry, dim=len(self.data.owned), inputs=[self.data, state.joint_q], device=self.device)
        if len(self.static_indices):
            wp.launch(
                _save_static,
                dim=len(self.static_indices),
                inputs=[self.static_indices, state.body_q, self.static_q, self.static_valid],
                device=self.device,
            )

    def notify_model_changed(self, flags):
        """Invalidate geometry only where the existing descriptor owner is dirty."""
        wp.launch(
            _invalidate,
            dim=self.plan.component_count,
            inputs=[self.data, self.empty_mask, int(int(flags) == int(ModelFlags.JOINT_PROPERTIES))],
            device=self.device,
        )

    def invalidate(self, world_mask):
        """Invalidate geometry after SleepController clears reset-world leases."""
        if world_mask is not None and world_mask.shape != (self.solver.world_count,):
            raise ValueError("Contact sleeping reset requires one mask entry per solver world")
        wp.launch(
            _invalidate,
            dim=self.plan.component_count,
            inputs=[self.data, self.empty_mask if world_mask is None else world_mask, 0],
            device=self.device,
        )

    def publish_contacts(self, contacts):
        """Publish held forces only for actual post-finish singleton leases."""
        if contacts is not self.contacts:
            return False
        if contacts.rigid_contact_max:
            wp.launch(
                publish_dormant_contact_forces,
                dim=contacts.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    self.plan.body_component,
                    self.data.sleeping,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    self.model.shape_body,
                    self.contact_asleep,
                    self.cache.held_force,
                    self.cache.held_valid,
                    contacts.rigid_contact_force,
                ],
                device=self.device,
            )
        return True

    def check_status(self):
        """Reject recorded cache, broad-input or force failures outside capture."""
        if self.device.is_cuda and self.device.is_capturing:
            raise RuntimeError("Contact-sleep status checks must run outside CUDA capture")
        if self.cache is not None:
            self.cache.check_capacity(self.contacts)
        status = int(self.status.numpy()[0])
        if status != 0:
            raise RuntimeError(f"Contact sleeping recorded invalid broad/contact/force state: status={status}")
