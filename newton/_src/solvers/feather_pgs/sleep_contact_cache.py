# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Capacity-preserving canonical contact history for experimental sleeping.

The caller owns geometry validity, contact-island wake-up and force capture.
This primitive neither establishes a sleep lease nor changes collision dispatch.
"""

import warp as wp

from ...sim import Contacts


@wp.struct
class ContactGeometry:
    point_id: wp.array[int]
    shape0: wp.array[int]
    shape1: wp.array[int]
    point0: wp.array[wp.vec3]
    point1: wp.array[wp.vec3]
    offset0: wp.array[wp.vec3]
    offset1: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    margin0: wp.array[float]
    margin1: wp.array[float]
    tids: wp.array[int]
    force: wp.array[wp.vec3]


_GEOMETRY_FIELDS = (
    "point_id",
    "shape0",
    "shape1",
    "point0",
    "point1",
    "offset0",
    "offset1",
    "normal",
    "margin0",
    "margin1",
    "tids",
    "force",
)


@wp.func
def _endpoint_state(shape: int, shape_body: wp.array[int], body_component: wp.array[int], asleep: wp.array[int]):
    # -1: awake/unsupported/invalid; 0: static; 1: sleeping dynamic component.
    if shape < 0 or shape >= shape_body.shape[0]:
        return -1
    body = shape_body[shape]
    if body == -1:
        return 0
    if body < 0 or body >= body_component.shape[0]:
        return -1
    component = body_component[body]
    if component == -1:
        return 0
    if component < 0 or component >= asleep.shape[0]:
        return -1
    if asleep[component] != 0:
        return 1
    return -1


@wp.func
def pair_is_dormant(
    shape0: int,
    shape1: int,
    shape_body: wp.array[int],
    body_component: wp.array[int],
    component_asleep: wp.array[int],
):
    """Select sleeping/sleeping or sleeping/static, never static/static.

    The topology's -1 body component denotes an immobile skeleton. Unsupported
    dynamic components must have their component_asleep entry cleared by the
    lease owner; arbitrary unmapped dynamic bodies are not static sentinels.
    """
    first = _endpoint_state(shape0, shape_body, body_component, component_asleep)
    second = _endpoint_state(shape1, shape_body, body_component, component_asleep)
    return first >= 0 and second >= 0 and first + second > 0


@wp.func
def _copy_geometry(source: ContactGeometry, destination: ContactGeometry, source_slot: int, destination_slot: int):
    destination.point_id[destination_slot] = source.point_id[source_slot]
    destination.shape0[destination_slot] = source.shape0[source_slot]
    destination.shape1[destination_slot] = source.shape1[source_slot]
    destination.point0[destination_slot] = source.point0[source_slot]
    destination.point1[destination_slot] = source.point1[source_slot]
    destination.offset0[destination_slot] = source.offset0[source_slot]
    destination.offset1[destination_slot] = source.offset1[source_slot]
    destination.normal[destination_slot] = source.normal[source_slot]
    destination.margin0[destination_slot] = source.margin0[source_slot]
    destination.margin1[destination_slot] = source.margin1[source_slot]
    destination.tids[destination_slot] = source.tids[source_slot]
    destination.force[destination_slot] = source.force[source_slot]


@wp.kernel(enable_backward=False)
def _snapshot(
    current: ContactGeometry,
    count: wp.array[int],
    held_force: wp.array[wp.vec3],
    held_valid: wp.array[int],
    source_index: wp.array[int],
    history: ContactGeometry,
    history_count: wp.array[int],
    history_force: wp.array[wp.vec3],
    history_valid: wp.array[int],
    status: wp.array[int],
):
    slot = wp.tid()
    capacity = held_valid.shape[0]
    raw_count = count[0]
    valid_count = raw_count >= 0 and raw_count <= capacity
    if slot == 0:
        history_count[0] = raw_count
        status[0] = int(not valid_count)
    if slot >= capacity:
        return
    if valid_count and slot < raw_count:
        _copy_geometry(current, history, slot, slot)
        history_force[slot] = held_force[slot]
        history_valid[slot] = held_valid[slot]
    # Current metadata is private to the cache. Clearing here avoids a third
    # full-capacity launch and cannot race another slot's history read.
    held_force[slot] = wp.vec3()
    held_valid[slot] = 0
    source_index[slot] = -1


@wp.kernel(enable_backward=False)
def _emit(
    history: ContactGeometry,
    history_count: wp.array[int],
    history_force: wp.array[wp.vec3],
    history_valid: wp.array[int],
    current: ContactGeometry,
    count: wp.array[int],
    held_force: wp.array[wp.vec3],
    held_valid: wp.array[int],
    source_index: wp.array[int],
    shape_body: wp.array[int],
    body_component: wp.array[int],
    component_asleep: wp.array[int],
    status: wp.array[int],
):
    slot = wp.tid()
    capacity = held_valid.shape[0]
    raw_count = history_count[0]
    if slot == 0 and (count[0] < 0 or count[0] > capacity):
        wp.atomic_max(status, 0, 2)
    # An invalid history is never treated as a usable clamped prefix.
    if raw_count < 0 or raw_count > capacity or slot >= raw_count:
        return
    if not pair_is_dormant(history.shape0[slot], history.shape1[slot], shape_body, body_component, component_asleep):
        return
    destination = wp.atomic_add(count, 0, 1)
    if destination < 0 or destination >= capacity:
        wp.atomic_max(status, 0, 2)
        return
    _copy_geometry(history, current, slot, destination)
    held_force[destination] = history_force[slot]
    held_valid[destination] = history_valid[slot]
    source_index[destination] = slot


class SleepContactCache:
    """Own one exact-capacity history bank and current-slot held-force metadata.

    Call snapshot before Contacts.clear, then emit after clear and before fresh
    narrowphase appends. Caller-owned held_force stores world linear force [N];
    held_valid distinguishes a captured zero force from an uncaptured force.
    source_index is -1 for fresh slots and the prior slot for cached records.
    Extended Contacts.force is untouched, just as by ordinary geometry writers.

    status is 0 for success, 1 for an invalid historical count, and 2 for an
    invalid emission reservation. The original raw counter remains authoritative
    for total cached-plus-fresh capacity; overflow is an error, not truncation.
    """

    @staticmethod
    def _validate(contacts: Contacts):
        if (
            contacts.requires_grad
            or contacts.per_contact_shape_properties
            or contacts.contact_matching
            or contacts.contact_report
            or contacts.rigid_contacts_pair_sorted
            or contacts.rigid_contacts_body_pair_reduced
            or contacts.rigid_contacts_body_pair_reduced_capture
        ):
            raise ValueError(
                "Sleeping contact cache requires unsorted, unreduced rigid geometry without optional modes"
            )

    def __init__(self, contacts: Contacts):
        self._validate(contacts)
        self._contacts = contacts
        self.capacity = int(contacts.rigid_contact_max)
        self.current = ContactGeometry()
        self.history = ContactGeometry()
        for name in _GEOMETRY_FIELDS:
            array = getattr(contacts, "rigid_contact_" + name)
            if array.shape != (self.capacity,) or array.device != contacts.device:
                raise ValueError("Canonical rigid contact arrays must retain their exact capacity and device")
            setattr(self.current, name, array)
            setattr(self.history, name, wp.empty_like(array))
        self.held_force = wp.zeros(self.capacity, dtype=wp.vec3, device=contacts.device)
        self.held_valid = wp.zeros(self.capacity, dtype=wp.int32, device=contacts.device)
        self.source_index = wp.full(self.capacity, -1, dtype=wp.int32, device=contacts.device)
        self.history_force = wp.zeros_like(self.held_force)
        self.history_valid = wp.zeros_like(self.held_valid)
        self.history_count = wp.zeros(1, dtype=wp.int32, device=contacts.device)
        self.status = wp.zeros(1, dtype=wp.int32, device=contacts.device)

    def _check_binding(self, contacts: Contacts):
        self._validate(contacts)
        if contacts is not self._contacts or contacts.rigid_contact_max != self.capacity:
            raise ValueError("Sleeping contact cache is bound to one fixed-capacity Contacts allocation")
        if any(
            getattr(contacts, "rigid_contact_" + name) is not getattr(self.current, name) for name in _GEOMETRY_FIELDS
        ):
            raise ValueError("Canonical rigid contact arrays changed after cache construction")

    def snapshot(self, contacts: Contacts):
        """Save all active geometry and held forces, then clear current cache metadata."""
        self._check_binding(contacts)
        wp.launch(
            _snapshot,
            dim=max(self.capacity, 1),
            inputs=[
                self.current,
                contacts.rigid_contact_count,
                self.held_force,
                self.held_valid,
                self.source_index,
                self.history,
                self.history_count,
                self.history_force,
                self.history_valid,
                self.status,
            ],
            device=contacts.device,
        )

    def emit(self, contacts: Contacts, shape_body: wp.array, body_component: wp.array, component_asleep: wp.array):
        """Append every dormant record, including zero-force and uncaptured-force geometry."""
        self._check_binding(contacts)
        wp.launch(
            _emit,
            dim=max(self.capacity, 1),
            inputs=[
                self.history,
                self.history_count,
                self.history_force,
                self.history_valid,
                self.current,
                contacts.rigid_contact_count,
                self.held_force,
                self.held_valid,
                self.source_index,
                shape_body,
                body_component,
                component_asleep,
                self.status,
            ],
            device=contacts.device,
        )

    def check_capacity(self, contacts: Contacts):
        """Check the completed cached-plus-fresh raw count at an untimed boundary."""
        self._check_binding(contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        status = int(self.status.numpy()[0])
        if status != 0 or count < 0 or count > self.capacity:
            raise RuntimeError(
                f"Sleeping contact cache overflow/invalid count: count={count}, max={self.capacity}, status={status}"
            )
