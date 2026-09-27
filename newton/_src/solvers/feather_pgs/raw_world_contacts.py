# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Complete raw-contact ID buckets for private joint-world traversal.

No geometry is copied or removed. IDs refer to the original raw prefix, so
anchor-neighbor queries must still use raw IDs, never neighboring bucket slots.
An invalid prefix or ambiguous world disables the whole optimized frame; these
private flags do not clear or replace the existing collision-capacity flags.
"""

import warp as wp


@wp.struct
class RawWorldContacts:
    offsets: wp.array[int]
    ids: wp.array[int]
    invalid: wp.array[int]


@wp.func
def _endpoint_world(
    shape: int, shape_body: wp.array[int], body_art: wp.array[int], art_world: wp.array[int], worlds: int
):
    world = int(-1)
    if shape < -1 or shape >= shape_body.shape[0]:
        return int(-2)
    if shape >= 0:
        body = shape_body[shape]
        if body < -1 or body >= body_art.shape[0]:
            return int(-2)
        if body >= 0:
            art = body_art[body]
            if art < -1 or art >= art_world.shape[0]:
                return int(-2)
            if art >= 0:
                world = art_world[art]
                if world < 0 or world >= worlds:
                    return int(-2)
    return world


@wp.func
def _contact_world(
    c: int,
    shape0: wp.array[int],
    shape1: wp.array[int],
    shape_body: wp.array[int],
    body_art: wp.array[int],
    art_world: wp.array[int],
    worlds: int,
):
    a = _endpoint_world(shape0[c], shape_body, body_art, art_world, worlds)
    b = _endpoint_world(shape1[c], shape_body, body_art, art_world, worlds)
    if a < -1 or b < -1 or (a >= 0 and b >= 0 and a != b):
        return int(-1)
    return wp.max(a, b)


@wp.kernel(enable_backward=False)
def count_raw_world_contacts(
    threads: int,
    raw_count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    shape_body: wp.array[int],
    body_art: wp.array[int],
    art_world: wp.array[int],
    counts: wp.array[int],
    data: RawWorldContacts,
):
    tid = wp.tid()
    count = raw_count[0]
    capacity = data.ids.shape[0]
    if count < 0 or count > capacity:
        if tid == 0:
            wp.atomic_max(data.invalid, 0, 1)
        return
    for c in range(tid, count, threads):
        world = _contact_world(c, shape0, shape1, shape_body, body_art, art_world, counts.shape[0] - 1)
        if world < 0:
            wp.atomic_max(data.invalid, 0, 1)
        else:
            wp.atomic_add(counts, world, 1)


@wp.kernel(enable_backward=False)
def scatter_raw_world_contacts(
    threads: int,
    raw_count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    shape_body: wp.array[int],
    body_art: wp.array[int],
    art_world: wp.array[int],
    counts: wp.array[int],
    data: RawWorldContacts,
):
    tid = wp.tid()
    count = raw_count[0]
    # This decision is complete before this launch. No thread returns based on
    # a flag that another scatter thread could set during the same launch.
    if count < 0 or count > data.ids.shape[0]:
        return
    for c in range(tid, count, threads):
        world = _contact_world(c, shape0, shape1, shape_body, body_art, art_world, counts.shape[0] - 1)
        if world >= 0:
            slot = data.offsets[world] + wp.atomic_add(counts, world, 1)
            if slot >= data.offsets[world] and slot < data.offsets[world + 1] and slot < data.ids.shape[0]:
                data.ids[slot] = c
            else:
                wp.atomic_max(data.invalid, 0, 1)


@wp.kernel(enable_backward=False)
def validate_raw_world_contacts(raw_count: wp.array[int], counts: wp.array[int], data: RawWorldContacts):
    world = wp.tid()
    begin, end = data.offsets[world], data.offsets[world + 1]
    if begin < 0 or end < begin or end > data.ids.shape[0] or end - begin != counts[world]:
        wp.atomic_max(data.invalid, 0, 1)
    if world == 0:
        total = data.offsets[data.offsets.shape[0] - 1]
        if begin != 0 or total != raw_count[0]:
            wp.atomic_max(data.invalid, 0, 1)


class RawWorldContactBuckets:
    """Own exact-capacity integer IDs and O(world_count) scan scratch."""

    def __init__(self, world_count: int, contact_capacity: int, *, device):
        if world_count <= 0 or contact_capacity < 0 or contact_capacity > 2**31 - 1:
            raise ValueError("Raw buckets require positive worlds and an int32 contact capacity")
        self.world_count = int(world_count)
        self.contact_capacity = int(contact_capacity)
        self.device = wp.get_device(device)
        self.data = RawWorldContacts()
        self.data.offsets = wp.empty(world_count + 1, dtype=int, device=self.device)
        self.data.ids = wp.empty(contact_capacity, dtype=int, device=self.device)
        self.data.invalid = wp.zeros(1, dtype=int, device=self.device)
        self.counts = wp.zeros(world_count + 1, dtype=int, device=self.device)
        self.storage_bytes = 4 * (contact_capacity + 2 * (world_count + 1) + 1)

    def build(self, count, shape0, shape1, shape_body, body_to_articulation, art_to_world):
        """Build on the caller's stream after the raw contact producer completes."""
        if count.shape != (1,) or shape0.shape != (self.contact_capacity,) or shape1.shape != shape0.shape:
            raise ValueError("Raw bucket descriptors must match the exact allocated contact capacity")
        arrays = (count, shape0, shape1, shape_body, body_to_articulation, art_to_world)
        if any(array.dtype != wp.int32 or array.ndim != 1 or array.device != self.device for array in arrays):
            raise ValueError("Raw bucket inputs must be one-dimensional int32 arrays on the owner device")
        self.counts.zero_()
        self.data.invalid.zero_()
        threads = max(1, min(self.contact_capacity, 65536))
        inputs = [threads, *arrays, self.counts, self.data]
        wp.launch(count_raw_world_contacts, dim=threads, inputs=inputs, device=self.device)
        wp.utils.array_scan(self.counts, self.data.offsets, inclusive=False)
        self.counts.zero_()
        wp.launch(scatter_raw_world_contacts, dim=threads, inputs=inputs, device=self.device)
        wp.launch(
            validate_raw_world_contacts,
            dim=self.world_count,
            inputs=[count, self.counts, self.data],
            device=self.device,
        )
