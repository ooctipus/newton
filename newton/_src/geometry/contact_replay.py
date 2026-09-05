# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dormant contact store for sleeping dynamic shapes resting on immovable shapes.

The collision pipeline emits only *live* rows (at least one awake dynamic tree) into
the :class:`~newton.Contacts` buffer. Rows between an asleep dynamic shape and an
immovable (static or kinematic) partner are parked in a per-dynamic-shape slab
instead. Exact per-shape signatures certify that a slab still describes the current
contact configuration so the corresponding SDF pairs can be skipped, and the MuJoCo
adapter injects slab rows when a tree wakes inside a step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..core.reset import reset_world_selected
from .contact_data import SHAPE_PAIR_INDEX_MASK
from .flags import ShapeFlags
from .types import GeoType

if TYPE_CHECKING:
    from ..sim.model import Model

# Grid-stride budget for the candidate-pair mask launch.
_MASK_MAX_THREADS = 65_536


@wp.struct
class DormantContactSlabs:
    """Per-dynamic-shape slabs of contact rows that are absent from the live contact buffer.

    Rows are stored flattened as ``slab * capacity + row`` in the pipeline's
    body-local contact format (the same nine fields as the live buffer).
    """

    capacity: int
    """Rows per slab."""
    slab_of_shape: wp.array[wp.int32]
    """Slab index per collision shape, ``-1`` for shapes without a slab, shape ``[shape_count]``."""
    slab_shape: wp.array[wp.int32]
    """Dynamic shape owning each slab, shape ``[slab_count]``."""
    slab_count: wp.array[wp.int32]
    """Rows currently held per slab (may exceed ``capacity`` after an overflow), shape ``[slab_count]``."""
    slab_overflow: wp.array[wp.int32]
    """Nonzero when a slab dropped rows to the live buffer this pass, shape ``[slab_count]``."""
    slab_live_gen: wp.array[wp.int32]
    """Contact generation whose live buffer already holds this slab's rows, shape ``[slab_count]``."""
    shape0: wp.array[wp.int32]
    shape1: wp.array[wp.int32]
    point0: wp.array[wp.vec3]
    """Body-frame contact point on shape 0 [m]."""
    point1: wp.array[wp.vec3]
    """Body-frame contact point on shape 1 [m]."""
    offset0: wp.array[wp.vec3]
    """Body-frame friction anchor offset for shape 0 [m]."""
    offset1: wp.array[wp.vec3]
    """Body-frame friction anchor offset for shape 1 [m]."""
    normal: wp.array[wp.vec3]
    """Contact normal from shape 0 toward shape 1."""
    margin0: wp.array[wp.float32]
    """Surface thickness for shape 0 [m]."""
    margin1: wp.array[wp.float32]
    """Surface thickness for shape 1 [m]."""


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
def _move_slab_row(slabs: DormantContactSlabs, source: int, target: int):
    slabs.shape0[target] = slabs.shape0[source]
    slabs.shape1[target] = slabs.shape1[source]
    slabs.point0[target] = slabs.point0[source]
    slabs.point1[target] = slabs.point1[source]
    slabs.offset0[target] = slabs.offset0[source]
    slabs.offset1[target] = slabs.offset1[source]
    slabs.normal[target] = slabs.normal[source]
    slabs.margin0[target] = slabs.margin0[source]
    slabs.margin1[target] = slabs.margin1[source]


@wp.kernel(enable_backward=False)
def retain_dormant_slabs(
    slabs: DormantContactSlabs,
    shape_type: wp.array[wp.int32],
    shape_sdf_index: wp.array[wp.int32],
    shape_edge_range: wp.array[wp.vec2i],
    shape_flags: wp.array[wp.int32],
    shape_sleep_index: wp.array[wp.vec2i],
    tree_asleep: wp.array2d[wp.int32],
    shape_unchanged: wp.array[wp.int32],
    contact_generation: wp.array[wp.int32],
    slab_masking: wp.array[wp.int32],
):
    """Keep slab rows whose pair is replayed this pass and drop everything else.

    One thread per slab. A slab survives only when its dynamic shape is asleep,
    exactly unchanged, and did not overflow last pass; surviving rows are then
    compacted to those whose partner is also unchanged, i.e. exactly the pairs
    :func:`mask_dormant_pairs` removes from the current candidate list.
    """
    slab = wp.tid()
    shape = slabs.slab_shape[slab]
    sleep = shape_sleep_index[shape]
    asleep = tree_asleep[sleep[0], sleep[1]] >= 0
    if not asleep:
        # The writer exports this shape's rows live this pass; stamp the slab so a
        # wake later in the tick does not inject them a second time.
        slabs.slab_live_gen[slab] = contact_generation[0]
    if not asleep or shape_unchanged[shape] == 0 or slabs.slab_overflow[slab] != 0:
        slabs.slab_count[slab] = 0
        slabs.slab_overflow[slab] = 0
        slab_masking[slab] = 0
        return

    capacity = slabs.capacity
    base = slab * capacity
    count = wp.min(slabs.slab_count[slab], capacity)
    kept = int(0)
    for row in range(count):
        shape_a = slabs.shape0[base + row]
        shape_b = slabs.shape1[base + row]
        partner = shape_b
        if shape_b == shape:
            partner = shape_a
        keep = shape_unchanged[partner] != 0 and _is_replay_pair(
            shape_a,
            shape_b,
            shape_type,
            shape_sdf_index,
            shape_edge_range,
            shape_flags,
            shape_sleep_index,
        )
        if keep:
            if kept != row:
                _move_slab_row(slabs, base + row, base + kept)
            kept += 1
    slabs.slab_count[slab] = kept
    slab_masking[slab] = 1


@wp.kernel(enable_backward=False)
def mask_dormant_pairs(
    candidate_pairs: wp.array[wp.vec2i],
    candidate_pair_count: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_sdf_index: wp.array[wp.int32],
    shape_edge_range: wp.array[wp.vec2i],
    shape_flags: wp.array[wp.int32],
    shape_sleep_index: wp.array[wp.vec2i],
    tree_asleep: wp.array2d[wp.int32],
    shape_unchanged: wp.array[wp.int32],
    slab_of_shape: wp.array[wp.int32],
    slab_masking: wp.array[wp.int32],
    total_num_threads: int,
):
    """Remove candidate pairs whose rows are retained in a certified slab."""
    tid = wp.tid()
    pair_count = wp.min(candidate_pair_count[0], candidate_pairs.shape[0])
    for index in range(tid, pair_count, total_num_threads):
        pair = candidate_pairs[index]
        if pair[0] < 0 or pair[1] < 0:
            continue
        shape_a = pair[0] & SHAPE_PAIR_INDEX_MASK
        shape_b = pair[1] & SHAPE_PAIR_INDEX_MASK
        if not _is_replay_eligible(
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
            continue
        dynamic = shape_a
        if shape_sleep_index[shape_a][1] < 0:
            dynamic = shape_b
        slab = slab_of_shape[dynamic]
        if slab >= 0 and slab_masking[slab] != 0:
            candidate_pairs[index] = wp.vec2i(-1, -1)


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


@wp.kernel(enable_backward=False)
def invalidate_dormant_slabs(
    slab_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    world_count: int,
    slab_count: wp.array[wp.int32],
    slab_overflow: wp.array[wp.int32],
):
    """Drop slab rows of reset-selected worlds."""
    slab = wp.tid()
    if reset_world_selected(slab_world[slab], world_mask, world_count):
        slab_count[slab] = 0
        slab_overflow[slab] = 0


class DormantContactStore:
    """Own the per-dynamic-shape dormant slabs and the exact per-shape replay certificates.

    One slab is allocated for every collision-enabled shape attached to a dynamic
    MuJoCo tree (``shape_sleep_index[shape][1] >= 0``).
    """

    def __init__(
        self,
        model: Model,
        shape_sleep_index: wp.array[wp.vec2i],
        rows_per_shape: int,
        device: wp.Device,
    ):
        if rows_per_shape <= 0:
            raise ValueError(f"rows_per_shape must be positive, got {rows_per_shape}")
        shape_count = int(model.shape_count)
        if shape_sleep_index.shape != (shape_count,):
            raise ValueError("shape_sleep_index must have one entry per collision shape")
        self.device = device
        self.shape_sleep_index = shape_sleep_index
        self.rows_per_shape = int(rows_per_shape)

        sleep_index = shape_sleep_index.numpy()
        shape_flags = model.shape_flags.numpy()
        has_slab = (sleep_index[:, 1] >= 0) & ((shape_flags & int(ShapeFlags.COLLIDE_SHAPES)) != 0)
        slab_shapes = np.nonzero(has_slab)[0].astype(np.int32)
        slab_of_shape = np.full(shape_count, -1, dtype=np.int32)
        slab_of_shape[slab_shapes] = np.arange(slab_shapes.shape[0], dtype=np.int32)
        self.slab_count_total = int(slab_shapes.shape[0])
        """Number of slabs (dynamic collision shapes)."""
        row_total = self.slab_count_total * self.rows_per_shape

        with wp.ScopedDevice(device):
            self.signatures = wp.zeros(shape_count, dtype=ShapeReplaySignature, device=device)
            self.signature_valid = wp.zeros(shape_count, dtype=wp.int32, device=device)
            self.shape_unchanged = wp.zeros(shape_count, dtype=wp.int32, device=device)
            self.slab_world_tree = wp.array(sleep_index[slab_shapes], dtype=wp.vec2i, device=device)
            """MuJoCo ``(world, tree)`` per slab, shape ``[slab_count_total]``."""
            self.slab_world = wp.array(model.shape_world.numpy()[slab_shapes], dtype=wp.int32, device=device)
            self.slab_masking = wp.zeros(self.slab_count_total, dtype=wp.int32, device=device)
            """Nonzero for slabs whose pairs are masked in the current pass, shape ``[slab_count_total]``."""
            # Scratch for the solver-side wake injection (slabs whose tree woke this substep).
            self.inject_slabs = wp.zeros(self.slab_count_total, dtype=wp.int32, device=device)
            self.inject_count = wp.zeros(1, dtype=wp.int32, device=device)
            slabs = DormantContactSlabs()
            slabs.capacity = self.rows_per_shape
            slabs.slab_of_shape = wp.array(slab_of_shape, dtype=wp.int32, device=device)
            slabs.slab_shape = wp.array(slab_shapes, dtype=wp.int32, device=device)
            slabs.slab_count = wp.zeros(self.slab_count_total, dtype=wp.int32, device=device)
            slabs.slab_overflow = wp.zeros(self.slab_count_total, dtype=wp.int32, device=device)
            slabs.slab_live_gen = wp.full(self.slab_count_total, -1, dtype=wp.int32, device=device)
            slabs.shape0 = wp.empty(row_total, dtype=wp.int32, device=device)
            slabs.shape1 = wp.empty(row_total, dtype=wp.int32, device=device)
            slabs.point0 = wp.empty(row_total, dtype=wp.vec3, device=device)
            slabs.point1 = wp.empty(row_total, dtype=wp.vec3, device=device)
            slabs.offset0 = wp.empty(row_total, dtype=wp.vec3, device=device)
            slabs.offset1 = wp.empty(row_total, dtype=wp.vec3, device=device)
            slabs.normal = wp.empty(row_total, dtype=wp.vec3, device=device)
            slabs.margin0 = wp.empty(row_total, dtype=wp.float32, device=device)
            slabs.margin1 = wp.empty(row_total, dtype=wp.float32, device=device)
            self.slabs = slabs
            """Device view of the slab arrays shared with the contact writer and the solver."""

    @property
    def slab_count(self) -> wp.array[wp.int32]:
        """Rows held per slab, shape ``[slab_count_total]``."""
        return self.slabs.slab_count

    @property
    def slab_overflow(self) -> wp.array[wp.int32]:
        """Overflow flag per slab, shape ``[slab_count_total]``."""
        return self.slabs.slab_overflow

    @property
    def slab_live_gen(self) -> wp.array[wp.int32]:
        """Contact generation whose live buffer holds each slab's rows, shape ``[slab_count_total]``."""
        return self.slabs.slab_live_gen

    @property
    def slab_of_shape(self) -> wp.array[wp.int32]:
        """Slab index per shape (``-1`` without a slab), shape ``[shape_count]``."""
        return self.slabs.slab_of_shape

    def reset(self, shape_world: wp.array[wp.int32], world_count: int, world_mask: wp.array[wp.bool] | None):
        """Invalidate signatures and drop slab rows for all or reset-selected worlds."""
        if world_mask is None:
            self.signature_valid.zero_()
            self.slabs.slab_count.zero_()
            self.slabs.slab_overflow.zero_()
            return
        wp.launch(
            invalidate_shape_signatures,
            dim=shape_world.shape[0],
            inputs=[shape_world, world_mask, world_count, self.signature_valid],
            device=self.device,
            record_tape=False,
        )
        if self.slab_count_total > 0:
            wp.launch(
                invalidate_dormant_slabs,
                dim=self.slab_count_total,
                inputs=[self.slab_world, world_mask, world_count, self.slabs.slab_count, self.slabs.slab_overflow],
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

    def retain_and_mask(
        self,
        *,
        candidate_pairs: wp.array[wp.vec2i],
        candidate_pair_count: wp.array[wp.int32],
        contact_generation: wp.array[wp.int32],
        shape_type: wp.array[wp.int32],
        shape_sdf_index: wp.array[wp.int32],
        shape_edge_range: wp.array[wp.vec2i],
        shape_flags: wp.array[wp.int32],
        shape_sleep_index: wp.array[wp.vec2i],
        tree_asleep: wp.array2d[wp.int32],
    ):
        """Retain certified slab rows, then mask their candidate pairs.

        Two launches: the mask pass reads the per-slab masking flags the retention
        pass writes, so they must not share a launch.
        """
        if self.slab_count_total == 0:
            return
        wp.launch(
            retain_dormant_slabs,
            dim=self.slab_count_total,
            inputs=[
                self.slabs,
                shape_type,
                shape_sdf_index,
                shape_edge_range,
                shape_flags,
                shape_sleep_index,
                tree_asleep,
                self.shape_unchanged,
                contact_generation,
            ],
            outputs=[self.slab_masking],
            device=self.device,
            record_tape=False,
        )
        thread_count = min(max(candidate_pairs.shape[0], 1), _MASK_MAX_THREADS)
        wp.launch(
            mask_dormant_pairs,
            dim=thread_count,
            inputs=[
                candidate_pairs,
                candidate_pair_count,
                shape_type,
                shape_sdf_index,
                shape_edge_range,
                shape_flags,
                shape_sleep_index,
                tree_asleep,
                self.shape_unchanged,
                self.slabs.slab_of_shape,
                self.slab_masking,
                thread_count,
            ],
            device=self.device,
            record_tape=False,
        )
