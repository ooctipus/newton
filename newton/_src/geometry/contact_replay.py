# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exact replay of unchanged reduced SDF contacts."""

from __future__ import annotations

import warp as wp

from ..core.reset import reset_world_selected
from .contact_data import SHAPE_PAIR_INDEX_MASK
from .flags import ShapeFlags
from .types import GeoType


@wp.struct
class ContactRows:
    """Device arrays holding final rigid-contact rows."""

    capacity: int
    count: wp.array[wp.int32]
    shape0: wp.array[wp.int32]
    shape1: wp.array[wp.int32]
    point0: wp.array[wp.vec3]
    point1: wp.array[wp.vec3]
    offset0: wp.array[wp.vec3]
    offset1: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    margin0: wp.array[wp.float32]
    margin1: wp.array[wp.float32]
    sort_key: wp.array[wp.int64]


@wp.struct
class ShapeReplaySignature:
    """Collision inputs required to prove that a shape is unchanged."""

    shape_transform: wp.transform
    body_transform: wp.transform
    shape_data: wp.vec4
    linear_velocity: wp.vec3
    angular_velocity: wp.vec3
    collision_aabb_lower: wp.vec3
    collision_aabb_upper: wp.vec3
    source: wp.uint64
    shape_type: wp.int32
    mesh_properties: wp.int32
    sdf_index: wp.int32
    edge_range: wp.vec2i
    voxel_resolution: wp.vec3i
    shape_body: wp.int32
    shape_sleep_index: wp.vec2i
    body_flags: wp.int32
    shape_flags: wp.int32
    shape_world: wp.int32
    collision_group: wp.int32
    gap: wp.float32
    base_gap: wp.float32
    collision_radius: wp.float32
    collision_update_dt: wp.float32
    max_speculative_extension: wp.float32


@wp.func
def _vec2i_equal(a: wp.vec2i, b: wp.vec2i) -> bool:
    return a[0] == b[0] and a[1] == b[1]


@wp.func
def _vec3i_equal(a: wp.vec3i, b: wp.vec3i) -> bool:
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


@wp.func
def _vec3_equal(a: wp.vec3, b: wp.vec3) -> bool:
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


@wp.func
def _vec4_equal(a: wp.vec4, b: wp.vec4) -> bool:
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and a[3] == b[3]


@wp.func
def _quat_equal(a: wp.quat, b: wp.quat) -> bool:
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and a[3] == b[3]


@wp.func
def _transform_equal(a: wp.transform, b: wp.transform) -> bool:
    return _vec3_equal(wp.transform_get_translation(a), wp.transform_get_translation(b)) and _quat_equal(
        wp.transform_get_rotation(a), wp.transform_get_rotation(b)
    )


@wp.kernel(enable_backward=False)
def classify_unchanged_shapes(
    shape_data: wp.array[wp.vec4],
    shape_transform: wp.array[wp.transform],
    shape_source: wp.array[wp.uint64],
    shape_type: wp.array[wp.int32],
    shape_mesh_properties: wp.array[wp.int32],
    shape_sdf_index: wp.array[wp.int32],
    shape_edge_range: wp.array[wp.vec2i],
    shape_voxel_resolution: wp.array[wp.vec3i],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_sleep_index: wp.array[wp.vec2i],
    body_q: wp.array[wp.transform],
    body_flags: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    shape_collision_group: wp.array[wp.int32],
    shape_gap: wp.array[wp.float32],
    shape_base_gap: wp.array[wp.float32],
    shape_collision_radius: wp.array[wp.float32],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    collision_update_dt: float,
    max_speculative_extension: float,
    signatures: wp.array[ShapeReplaySignature],
    signature_valid: wp.array[wp.int32],
    shape_unchanged: wp.array[wp.int32],
):
    """Classify exact shape identity and update the next-pass signature."""
    shape = wp.tid()
    body = shape_body[shape]
    body_transform = wp.transform_identity()
    current_body_flags = int(0)
    if body >= 0:
        body_transform = body_q[body]
        current_body_flags = body_flags[body]

    current = ShapeReplaySignature()
    current.shape_transform = shape_transform[shape]
    current.body_transform = body_transform
    current.shape_data = shape_data[shape]
    current.linear_velocity = shape_linear_velocity[shape]
    current.angular_velocity = shape_angular_velocity[shape]
    current.collision_aabb_lower = shape_collision_aabb_lower[shape]
    current.collision_aabb_upper = shape_collision_aabb_upper[shape]
    current.source = shape_source[shape]
    current.shape_type = shape_type[shape]
    current.mesh_properties = shape_mesh_properties[shape]
    current.sdf_index = shape_sdf_index[shape]
    current.edge_range = shape_edge_range[shape]
    current.voxel_resolution = shape_voxel_resolution[shape]
    current.shape_body = body
    current.shape_sleep_index = shape_sleep_index[shape]
    current.body_flags = current_body_flags
    current.shape_flags = shape_flags[shape]
    current.shape_world = shape_world[shape]
    current.collision_group = shape_collision_group[shape]
    current.gap = shape_gap[shape]
    current.base_gap = shape_base_gap[shape]
    current.collision_radius = shape_collision_radius[shape]
    current.collision_update_dt = collision_update_dt
    current.max_speculative_extension = max_speculative_extension

    previous = signatures[shape]
    unchanged = signature_valid[shape] != 0
    unchanged = unchanged and _transform_equal(previous.shape_transform, current.shape_transform)
    unchanged = unchanged and _transform_equal(previous.body_transform, current.body_transform)
    unchanged = unchanged and _vec4_equal(previous.shape_data, current.shape_data)
    unchanged = unchanged and _vec3_equal(previous.linear_velocity, current.linear_velocity)
    unchanged = unchanged and _vec3_equal(previous.angular_velocity, current.angular_velocity)
    unchanged = unchanged and _vec3_equal(previous.collision_aabb_lower, current.collision_aabb_lower)
    unchanged = unchanged and _vec3_equal(previous.collision_aabb_upper, current.collision_aabb_upper)
    unchanged = unchanged and previous.source == current.source
    unchanged = unchanged and previous.shape_type == current.shape_type
    unchanged = unchanged and previous.mesh_properties == current.mesh_properties
    unchanged = unchanged and previous.sdf_index == current.sdf_index
    unchanged = unchanged and _vec2i_equal(previous.edge_range, current.edge_range)
    unchanged = unchanged and _vec3i_equal(previous.voxel_resolution, current.voxel_resolution)
    unchanged = unchanged and previous.shape_body == current.shape_body
    unchanged = unchanged and _vec2i_equal(previous.shape_sleep_index, current.shape_sleep_index)
    unchanged = unchanged and previous.body_flags == current.body_flags
    unchanged = unchanged and previous.shape_flags == current.shape_flags
    unchanged = unchanged and previous.shape_world == current.shape_world
    unchanged = unchanged and previous.collision_group == current.collision_group
    unchanged = unchanged and previous.gap == current.gap
    unchanged = unchanged and previous.base_gap == current.base_gap
    unchanged = unchanged and previous.collision_radius == current.collision_radius
    unchanged = unchanged and previous.collision_update_dt == current.collision_update_dt
    unchanged = unchanged and previous.max_speculative_extension == current.max_speculative_extension

    shape_unchanged[shape] = int(unchanged)
    signatures[shape] = current
    signature_valid[shape] = 1


@wp.func
def _is_replay_pair(
    shape_a: int,
    shape_b: int,
    shape_type: wp.array[wp.int32],
    shape_sdf_index: wp.array[wp.int32],
    shape_edge_range: wp.array[wp.vec2i],
    shape_flags: wp.array[wp.int32],
    shape_sleep_index: wp.array[wp.vec2i],
) -> bool:
    """Return whether a pair belongs to the reduced dynamic-kinematic SDF path."""
    if shape_a < 0 or shape_b < 0:
        return False
    type_a = shape_type[shape_a]
    type_b = shape_type[shape_b]
    if type_a == GeoType.HFIELD or type_b == GeoType.HFIELD:
        return False
    if type_a == GeoType.BOX and type_b == GeoType.BOX:
        return False
    if shape_sdf_index[shape_a] < 0 or shape_sdf_index[shape_b] < 0:
        return False
    if shape_edge_range[shape_a][1] <= 0 or shape_edge_range[shape_b][1] <= 0:
        return False
    if (shape_flags[shape_a] & ShapeFlags.HYDROELASTIC) != 0:
        return False
    if (shape_flags[shape_b] & ShapeFlags.HYDROELASTIC) != 0:
        return False

    sleep_a = shape_sleep_index[shape_a]
    sleep_b = shape_sleep_index[shape_b]
    dynamic_a = sleep_a[1] >= 0
    dynamic_b = sleep_b[1] >= 0
    kinematic_a = sleep_a[1] == -2
    kinematic_b = sleep_b[1] == -2
    return (dynamic_a and kinematic_b) or (dynamic_b and kinematic_a)


@wp.func
def _is_replay_eligible(
    shape_a: int,
    shape_b: int,
    shape_type: wp.array[wp.int32],
    shape_sdf_index: wp.array[wp.int32],
    shape_edge_range: wp.array[wp.vec2i],
    shape_flags: wp.array[wp.int32],
    shape_sleep_index: wp.array[wp.vec2i],
    tree_asleep: wp.array2d[wp.int32],
    shape_unchanged: wp.array[wp.int32],
) -> bool:
    """Return whether a cached pair is exact and currently asleep."""
    if not _is_replay_pair(
        shape_a,
        shape_b,
        shape_type,
        shape_sdf_index,
        shape_edge_range,
        shape_flags,
        shape_sleep_index,
    ):
        return False
    if shape_unchanged[shape_a] == 0 or shape_unchanged[shape_b] == 0:
        return False

    sleep_a = shape_sleep_index[shape_a]
    sleep_b = shape_sleep_index[shape_b]
    asleep_a = sleep_a[1] >= 0 and tree_asleep[sleep_a[0], sleep_a[1]] >= 0
    asleep_b = sleep_b[1] >= 0 and tree_asleep[sleep_b[0], sleep_b[1]] >= 0
    return (asleep_a and sleep_b[1] == -2) or (asleep_b and sleep_a[1] == -2)


@wp.func
def _copy_contact_row(source: ContactRows, source_index: int, target: ContactRows, target_index: int):
    target.shape0[target_index] = source.shape0[source_index]
    target.shape1[target_index] = source.shape1[source_index]
    target.point0[target_index] = source.point0[source_index]
    target.point1[target_index] = source.point1[source_index]
    target.offset0[target_index] = source.offset0[source_index]
    target.offset1[target_index] = source.offset1[source_index]
    target.normal[target_index] = source.normal[source_index]
    target.margin0[target_index] = source.margin0[source_index]
    target.margin1[target_index] = source.margin1[source_index]
    if target.sort_key.shape[0] > 0:
        target.sort_key[target_index] = wp.int64(0)
        if source.sort_key.shape[0] > 0:
            target.sort_key[target_index] = source.sort_key[source_index]


@wp.kernel(enable_backward=False)
def replay_contacts_and_mask_pairs(
    cached: ContactRows,
    cache_complete: wp.array[wp.int32],
    output: ContactRows,
    output_tids: wp.array[wp.int32],
    candidate_pairs: wp.array[wp.vec2i],
    candidate_pair_count: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_sdf_index: wp.array[wp.int32],
    shape_edge_range: wp.array[wp.vec2i],
    shape_flags: wp.array[wp.int32],
    shape_sleep_index: wp.array[wp.vec2i],
    tree_asleep: wp.array2d[wp.int32],
    shape_unchanged: wp.array[wp.int32],
    total_num_threads: int,
):
    """Append exact cached rows and remove their pairs from the current scan."""
    if cache_complete[0] == 0 or cached.count[0] > cached.capacity or cached.count[0] > output.capacity:
        return

    tid = wp.tid()
    cached_count = wp.min(cached.count[0], cached.capacity)
    for index in range(tid, cached_count, total_num_threads):
        shape_a = cached.shape0[index]
        shape_b = cached.shape1[index]
        if _is_replay_eligible(
            shape_a,
            shape_b,
            shape_type,
            shape_sdf_index,
            shape_edge_range,
            shape_flags,
            shape_sleep_index,
            tree_asleep,
            shape_unchanged,
        ):
            output_index = wp.atomic_add(output.count, 0, 1)
            if output_index < output.capacity:
                _copy_contact_row(cached, index, output, output_index)
                output_tids[output_index] = 0

    pair_count = wp.min(candidate_pair_count[0], candidate_pairs.shape[0])
    for index in range(tid, pair_count, total_num_threads):
        pair = candidate_pairs[index]
        if pair[0] < 0 or pair[1] < 0:
            continue
        shape_a = pair[0] & SHAPE_PAIR_INDEX_MASK
        shape_b = pair[1] & SHAPE_PAIR_INDEX_MASK
        if _is_replay_eligible(
            shape_a,
            shape_b,
            shape_type,
            shape_sdf_index,
            shape_edge_range,
            shape_flags,
            shape_sleep_index,
            tree_asleep,
            shape_unchanged,
        ):
            candidate_pairs[index] = wp.vec2i(-1, -1)


@wp.kernel(enable_backward=False)
def begin_contact_cache_save(
    output_count: wp.array[wp.int32],
    output_capacity: int,
    broad_phase_pair_count: wp.array[wp.int32],
    broad_phase_pair_capacity: int,
    sdf_pair_count: wp.array[wp.int32],
    sdf_pair_capacity: int,
    reducer_contact_count: wp.array[wp.int32],
    reducer_contact_capacity: int,
    reducer_insert_failures: wp.array[wp.int32],
    cached_count: wp.array[wp.int32],
    cache_complete: wp.array[wp.int32],
):
    """Reset the compact cache and reject incomplete collision output."""
    cached_count[0] = 0
    cache_complete[0] = int(
        output_count[0] <= output_capacity
        and broad_phase_pair_count[0] <= broad_phase_pair_capacity
        and sdf_pair_count[0] <= sdf_pair_capacity
        and reducer_contact_count[0] < reducer_contact_capacity
        and reducer_insert_failures[0] == 0
    )


@wp.kernel(enable_backward=False)
def save_replay_contacts(
    output: ContactRows,
    cached: ContactRows,
    cache_complete: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_sdf_index: wp.array[wp.int32],
    shape_edge_range: wp.array[wp.vec2i],
    shape_flags: wp.array[wp.int32],
    shape_sleep_index: wp.array[wp.vec2i],
    total_num_threads: int,
):
    """Compact replay-capable final contact rows for the next pass."""
    tid = wp.tid()
    output_count = wp.min(output.count[0], output.capacity)
    for index in range(tid, output_count, total_num_threads):
        shape_a = output.shape0[index]
        shape_b = output.shape1[index]
        if _is_replay_pair(
            shape_a,
            shape_b,
            shape_type,
            shape_sdf_index,
            shape_edge_range,
            shape_flags,
            shape_sleep_index,
        ):
            cached_index = wp.atomic_add(cached.count, 0, 1)
            if cached_index < cached.capacity:
                _copy_contact_row(output, index, cached, cached_index)
            else:
                cache_complete[0] = 0


@wp.kernel(enable_backward=False)
def invalidate_shape_signatures(
    shape_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    world_count: int,
    signature_valid: wp.array[wp.int32],
):
    """Invalidate replay signatures for reset-selected worlds."""
    shape = wp.tid()
    if reset_world_selected(shape_world[shape], world_mask, world_count):
        signature_valid[shape] = 0


class SDFContactReplay:
    """Own the bounded cache for exact unchanged SDF contact replay."""

    def __init__(self, shape_count: int, contact_capacity: int, device: wp.Device):
        self.device = device
        self.contact_capacity = contact_capacity
        with wp.ScopedDevice(device):
            self.signatures = wp.zeros(shape_count, dtype=ShapeReplaySignature, device=device)
            self.signature_valid = wp.zeros(shape_count, dtype=wp.int32, device=device)
            self.shape_unchanged = wp.zeros(shape_count, dtype=wp.int32, device=device)
            self.contact_count = wp.zeros(1, dtype=wp.int32, device=device)
            self.complete = wp.zeros(1, dtype=wp.int32, device=device)
            self._empty_sort_key = wp.zeros(0, dtype=wp.int64, device=device)
            self.rows = ContactRows()
            self.rows.capacity = contact_capacity
            self.rows.count = self.contact_count
            self.rows.shape0 = wp.empty(contact_capacity, dtype=wp.int32, device=device)
            self.rows.shape1 = wp.empty(contact_capacity, dtype=wp.int32, device=device)
            self.rows.point0 = wp.empty(contact_capacity, dtype=wp.vec3, device=device)
            self.rows.point1 = wp.empty(contact_capacity, dtype=wp.vec3, device=device)
            self.rows.offset0 = wp.empty(contact_capacity, dtype=wp.vec3, device=device)
            self.rows.offset1 = wp.empty(contact_capacity, dtype=wp.vec3, device=device)
            self.rows.normal = wp.empty(contact_capacity, dtype=wp.vec3, device=device)
            self.rows.margin0 = wp.empty(contact_capacity, dtype=wp.float32, device=device)
            self.rows.margin1 = wp.empty(contact_capacity, dtype=wp.float32, device=device)
            self.rows.sort_key = wp.empty(contact_capacity, dtype=wp.int64, device=device)

    def make_rows(self, contacts, sort_key: wp.array[wp.int64]) -> ContactRows:
        """Build a row view over a live :class:`newton.Contacts` buffer."""
        rows = ContactRows()
        rows.capacity = contacts.rigid_contact_max
        rows.count = contacts.rigid_contact_count
        rows.shape0 = contacts.rigid_contact_shape0
        rows.shape1 = contacts.rigid_contact_shape1
        rows.point0 = contacts.rigid_contact_point0
        rows.point1 = contacts.rigid_contact_point1
        rows.offset0 = contacts.rigid_contact_offset0
        rows.offset1 = contacts.rigid_contact_offset1
        rows.normal = contacts.rigid_contact_normal
        rows.margin0 = contacts.rigid_contact_margin0
        rows.margin1 = contacts.rigid_contact_margin1
        rows.sort_key = sort_key
        return rows

    def reset(self, shape_world: wp.array[wp.int32], world_count: int, world_mask: wp.array[wp.bool] | None):
        """Invalidate all or reset-selected shape signatures."""
        if world_mask is None:
            self.signature_valid.zero_()
            self.contact_count.zero_()
            self.complete.zero_()
            return
        wp.launch(
            invalidate_shape_signatures,
            dim=shape_world.shape[0],
            inputs=[shape_world, world_mask, world_count, self.signature_valid],
            device=self.device,
            record_tape=False,
        )

    def classify(
        self,
        *,
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_type: wp.array[wp.int32],
        shape_mesh_properties: wp.array[wp.int32],
        shape_sdf_index: wp.array[wp.int32],
        shape_edge_range: wp.array[wp.vec2i],
        shape_voxel_resolution: wp.array[wp.vec3i],
        shape_collision_aabb_lower: wp.array[wp.vec3],
        shape_collision_aabb_upper: wp.array[wp.vec3],
        shape_body: wp.array[wp.int32],
        shape_sleep_index: wp.array[wp.vec2i],
        body_q: wp.array[wp.transform],
        body_flags: wp.array[wp.int32],
        shape_flags: wp.array[wp.int32],
        shape_world: wp.array[wp.int32],
        shape_collision_group: wp.array[wp.int32],
        shape_gap: wp.array[wp.float32],
        shape_base_gap: wp.array[wp.float32],
        shape_collision_radius: wp.array[wp.float32],
        shape_linear_velocity: wp.array[wp.vec3],
        shape_angular_velocity: wp.array[wp.vec3],
        collision_update_dt: float,
        max_speculative_extension: float,
    ):
        """Update exact per-shape replay certificates."""
        wp.launch(
            classify_unchanged_shapes,
            dim=shape_data.shape[0],
            inputs=[
                shape_data,
                shape_transform,
                shape_source,
                shape_type,
                shape_mesh_properties,
                shape_sdf_index,
                shape_edge_range,
                shape_voxel_resolution,
                shape_collision_aabb_lower,
                shape_collision_aabb_upper,
                shape_body,
                shape_sleep_index,
                body_q,
                body_flags,
                shape_flags,
                shape_world,
                shape_collision_group,
                shape_gap,
                shape_base_gap,
                shape_collision_radius,
                shape_linear_velocity,
                shape_angular_velocity,
                collision_update_dt,
                max_speculative_extension,
            ],
            outputs=[self.signatures, self.signature_valid, self.shape_unchanged],
            device=self.device,
            record_tape=False,
        )

    def replay_and_mask(
        self,
        *,
        output: ContactRows,
        output_tids: wp.array[wp.int32],
        candidate_pairs: wp.array[wp.vec2i],
        candidate_pair_count: wp.array[wp.int32],
        shape_type: wp.array[wp.int32],
        shape_sdf_index: wp.array[wp.int32],
        shape_edge_range: wp.array[wp.vec2i],
        shape_flags: wp.array[wp.int32],
        shape_sleep_index: wp.array[wp.vec2i],
        tree_asleep: wp.array2d[wp.int32],
    ):
        """Replay cached rows and mask the corresponding current SDF pairs."""
        thread_count = min(max(self.contact_capacity, candidate_pairs.shape[0], 1), 65_536)
        wp.launch(
            replay_contacts_and_mask_pairs,
            dim=thread_count,
            inputs=[
                self.rows,
                self.complete,
                output,
                output_tids,
                candidate_pairs,
                candidate_pair_count,
                shape_type,
                shape_sdf_index,
                shape_edge_range,
                shape_flags,
                shape_sleep_index,
                tree_asleep,
                self.shape_unchanged,
                thread_count,
            ],
            device=self.device,
            record_tape=False,
        )

    def save(
        self,
        *,
        output: ContactRows,
        broad_phase_pair_count: wp.array[wp.int32],
        broad_phase_pair_capacity: int,
        sdf_pair_count: wp.array[wp.int32],
        sdf_pair_capacity: int,
        reducer_contact_count: wp.array[wp.int32],
        reducer_contact_capacity: int,
        reducer_insert_failures: wp.array[wp.int32],
        shape_type: wp.array[wp.int32],
        shape_sdf_index: wp.array[wp.int32],
        shape_edge_range: wp.array[wp.vec2i],
        shape_flags: wp.array[wp.int32],
        shape_sleep_index: wp.array[wp.vec2i],
    ):
        """Replace the compact cache with replay-capable final rows."""
        wp.launch(
            begin_contact_cache_save,
            dim=1,
            inputs=[
                output.count,
                output.capacity,
                broad_phase_pair_count,
                broad_phase_pair_capacity,
                sdf_pair_count,
                sdf_pair_capacity,
                reducer_contact_count,
                reducer_contact_capacity,
                reducer_insert_failures,
            ],
            outputs=[self.contact_count, self.complete],
            device=self.device,
            record_tape=False,
        )
        thread_count = min(max(output.capacity, 1), 65_536)
        wp.launch(
            save_replay_contacts,
            dim=thread_count,
            inputs=[
                output,
                self.rows,
                self.complete,
                shape_type,
                shape_sdf_index,
                shape_edge_range,
                shape_flags,
                shape_sleep_index,
                thread_count,
            ],
            device=self.device,
            record_tape=False,
        )
