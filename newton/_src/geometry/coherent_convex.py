# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private rejection-only current-geometry axis hints and serialized cache ownership."""

import numpy as np
import warp as wp

from .support_function import GenericShapeData, unpack_mesh_ptr
from .types import GeoType

_ROUND_GUARD = wp.constant(128.0 * 1.1920928955078125e-7)
_ROUND_FLOOR = wp.constant(1.0e-30)


@wp.func
def _finite_vector(v: wp.vec3) -> bool:
    return wp.isfinite(v[0]) and wp.isfinite(v[1]) and wp.isfinite(v[2])


@wp.func
def _norm_one(v: wp.vec3) -> float:
    return wp.abs(v[0]) + wp.abs(v[1]) + wp.abs(v[2])


@wp.struct
class _SupportFeature:
    point: wp.vec3
    index: int
    value: float
    magnitude: float
    valid: int


@wp.func
def _support_feature(geom: GenericShapeData, direction: wp.vec3) -> _SupportFeature:
    """Scan every current vertex and retain a feature ID plus an error scale."""
    result = _SupportFeature()
    result.index = -1
    if not _finite_vector(direction) or not _finite_vector(geom.scale):
        return result
    if geom.shape_type == int(GeoType.BOX):
        if geom.scale[0] < 0.0 or geom.scale[1] < 0.0 or geom.scale[2] < 0.0:
            return result
        # Unlike the MPR centered/deadband policy, a certificate must use an
        # actual extremum. Equal-score corners may use either valid sign.
        code = int(0)
        point = wp.vec3()
        for axis in range(3):
            if direction[axis] >= 0.0:
                code = code | (1 << axis)
                point[axis] = geom.scale[axis]
            else:
                point[axis] = -geom.scale[axis]
        result.point = point
        result.index = code
        result.value = wp.dot(point, direction)
        result.magnitude = _norm_one(geom.scale)
        result.valid = int(wp.isfinite(result.value) and wp.isfinite(result.magnitude))
    elif geom.shape_type == int(GeoType.CONVEX_MESH):
        mesh = wp.mesh_get(unpack_mesh_ptr(geom.auxiliary))
        count = mesh.points.shape[0]
        if count == 0:
            return result
        scaled_direction = wp.cw_mul(geom.scale, direction)
        if not _finite_vector(scaled_direction):
            return result
        largest_scale = wp.max(wp.abs(geom.scale[0]), wp.max(wp.abs(geom.scale[1]), wp.abs(geom.scale[2])))
        maximum = float(0.0)
        magnitude = float(0.0)
        index = int(0)
        valid = int(1)
        for i in range(count):
            point = mesh.points[i]
            value = wp.dot(point, scaled_direction)
            local_magnitude = _norm_one(point)
            if not _finite_vector(point) or not wp.isfinite(value) or not wp.isfinite(local_magnitude):
                valid = 0
            if i == 0 or value > maximum:
                maximum = value
                index = i
            magnitude = wp.max(magnitude, local_magnitude)
        result.point = wp.cw_mul(mesh.points[index], geom.scale)
        result.index = index
        result.value = maximum
        result.magnitude = magnitude * largest_scale
        result.valid = int(valid != 0 and _finite_vector(result.point) and wp.isfinite(result.magnitude))
    return result


@wp.struct
class _PlaneBound:
    lower: float
    round_guard: float
    magnitude: float
    valid: int


@wp.func
def _plane_lower_bound(
    geom_a: GenericShapeData,
    geom_b: GenericShapeData,
    orientation_b: wp.quat,
    position_b: wp.vec3,
    direction: wp.vec3,
) -> _PlaneBound:
    """Lower-bound current complete-shape separation along a finite axis.

    The 128-epsilon envelope covers scale/dot products, quaternion direction
    transformation, translation subtraction, and norm rounding with an L1
    input-magnitude bound. It is intentionally loose. Overflow, subnormal
    axes, unsupported geometry and nonfinite inputs cannot certify a result.
    The independent float64 full-vertex oracle is the implementation gate.
    """
    result = _PlaneBound()
    if not _finite_vector(position_b) or not _finite_vector(direction):
        return result
    quaternion_norm_one = float(0.0)
    for axis in range(4):
        if not wp.isfinite(orientation_b[axis]):
            return result
        quaternion_norm_one += wp.abs(orientation_b[axis])
    direction_length = wp.length(direction)
    if not wp.isfinite(direction_length) or direction_length <= 1.0e-12:
        return result
    local_direction_b = wp.quat_rotate_inv(orientation_b, -direction)
    a = _support_feature(geom_a, direction)
    b = _support_feature(geom_b, local_direction_b)
    if a.valid == 0 or b.valid == 0:
        return result
    # Warp's rotation polynomial includes (2*w*w-1), even for nonunit q.
    # Its coefficient sum is bounded by 1+3*|q|_1^2.
    # The factor below includes that bound, all short FP32 expression chains,
    # and the computed magnitude/norm error. No unit-quaternion assumption.
    magnitude = (
        a.magnitude + (1.0 + 3.0 * quaternion_norm_one * quaternion_norm_one) * b.magnitude + _norm_one(position_b)
    )
    error = _ROUND_GUARD * wp.max(magnitude, 1.0e-12) * _norm_one(direction) + _ROUND_FLOOR
    gap = wp.dot(position_b, direction) - a.value - b.value
    norm_upper = direction_length * (1.0 + _ROUND_GUARD) + _ROUND_FLOOR
    numerator = gap - error
    denominator = norm_upper
    if numerator < 0.0:
        denominator = direction_length * (1.0 - _ROUND_GUARD) - _ROUND_FLOOR
    lower = numerator / denominator
    if not wp.isfinite(lower) or not wp.isfinite(error) or not wp.isfinite(magnitude):
        return result
    result.lower = lower
    result.round_guard = error / norm_upper
    result.magnitude = magnitude
    result.valid = 1
    return result


@wp.struct
class _CacheData:
    keys: wp.array[wp.uint64]
    world: wp.array[int]
    world_epoch: wp.array[wp.uint64]
    entry_epoch: wp.array[wp.uint64]
    generation: wp.array[wp.uint64]
    entry_generation: wp.array[wp.uint64]
    source_a: wp.array[wp.uint64]
    source_b: wp.array[wp.uint64]
    types: wp.array[wp.vec2i]
    mode: wp.array[int]
    axis: wp.array[wp.vec3]
    unresolved_work_items: wp.array[int]
    unresolved_work_count: wp.array[int]
    query_slot: wp.array[int]
    query_done: wp.array[int]
    stats: wp.array[int]


@wp.func
def _pair_key(pair: wp.vec2i) -> wp.uint64:
    return (wp.uint64(pair[0]) << wp.uint64(32)) | wp.uint64(pair[1])


@wp.func
def _find_slot(cache: _CacheData, pair: wp.vec2i) -> int:
    if pair[0] < 0 or pair[1] < 0:
        return -1
    key = _pair_key(pair)
    low = int(0)
    high = cache.keys.shape[0]
    while low < high:
        middle = (low + high) // 2
        if cache.keys[middle] < key:
            low = middle + 1
        else:
            high = middle
    if low < cache.keys.shape[0] and cache.keys[low] == key:
        return low
    return -1


@wp.kernel(enable_backward=False)
def _advance_generation(cache: _CacheData):
    cache.generation[0] = cache.generation[0] + wp.uint64(1)
    cache.unresolved_work_count[0] = 0
    for i in range(cache.stats.shape[0]):
        cache.stats[i] = 0


@wp.kernel(enable_backward=False)
def _reset_world_epochs(cache: _CacheData, world_mask: wp.array[wp.bool]):
    world = wp.tid()
    # Global/local pairs use the local bucket, so a global reset invalidates all buckets.
    if world_mask[world] or world_mask[world_mask.shape[0] - 1]:
        cache.world_epoch[world] = cache.world_epoch[world] + wp.uint64(1)


class _ConvexQueryCache:
    """Own fixed pair slots and separating-axis history for one serialized pipeline.

    Like the pipeline's existing query/contact scratch, invocations and resets
    must be ordered by the caller, including eager/capture stream handoffs.
    A graph lease retains these arrays and refuses a second live capture;
    it cannot observe or serialize arbitrary concurrent graph replays.
    """

    class _GraphLease:
        def __init__(self, owner):
            self.owner = owner
            self.token = object()
            owner._graph_token = self.token

        def release(self):
            owner = self.owner
            self.owner = None
            if owner is not None and owner._graph_token is self.token:
                owner._graph_token = None

        def __del__(self):
            self.release()

    def __init__(self, *, pairs, shape_types, shape_world, world_count, query_capacity, device):
        pairs = np.asarray(pairs)
        shape_types = np.asarray(shape_types)
        shape_world = np.asarray(shape_world)
        if (
            shape_types.ndim != 1
            or shape_world.shape != shape_types.shape
            or not np.issubdtype(shape_types.dtype, np.integer)
            or not np.issubdtype(shape_world.dtype, np.integer)
            or not isinstance(world_count, (int, np.integer))
            or world_count < 0
            or not isinstance(query_capacity, (int, np.integer))
            or query_capacity < 0
            or np.any(shape_world < -1)
            or np.any(shape_world >= world_count)
        ):
            raise ValueError("coherent convex cache requires consistent integer shape/world metadata and capacities")
        if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
            raise ValueError("coherent convex cache requires an integer pair array of shape (n, 2)")
        if np.any(pairs < 0) or np.any(pairs >= len(shape_types)) or np.any(pairs[:, 0] >= pairs[:, 1]):
            raise ValueError("coherent convex cache requires valid canonical shape IDs (a < b)")
        keys = (pairs[:, 0].astype(np.uint64) << np.uint64(32)) | pairs[:, 1].astype(np.uint64)
        if len(np.unique(keys)) != len(keys):
            raise ValueError("coherent convex cache requires unique ordered pairs")
        supported = np.isin(shape_types, [int(GeoType.BOX), int(GeoType.CONVEX_MESH)])
        selected = supported[pairs[:, 0]] & supported[pairs[:, 1]]
        pairs, keys = pairs[selected], keys[selected]
        order = np.argsort(keys)
        pairs, keys = pairs[order], keys[order]
        worlds_a, worlds_b = shape_world[pairs[:, 0]], shape_world[pairs[:, 1]]
        if np.any((worlds_a >= 0) & (worlds_b >= 0) & (worlds_a != worlds_b)):
            raise ValueError("coherent convex cache refuses cross-world pairs")
        worlds = np.maximum(worlds_a, worlds_b)
        worlds = np.where(worlds < 0, world_count, worlds).astype(np.int32)
        if np.any(worlds > world_count):
            raise ValueError("coherent convex cache received an invalid shape world")
        self.device = wp.get_device(device)
        self.data = _CacheData()
        self.data.keys = wp.array(keys, dtype=wp.uint64, device=device)
        self.data.world = wp.array(worlds, dtype=int, device=device)
        self.data.world_epoch = wp.zeros(world_count + 1, dtype=wp.uint64, device=device)
        self.data.generation = wp.zeros(1, dtype=wp.uint64, device=device)
        n = len(keys)
        for name in ("entry_epoch", "entry_generation", "source_a", "source_b"):
            setattr(self.data, name, wp.zeros(n, dtype=wp.uint64, device=device))
        self.data.types = wp.zeros(n, dtype=wp.vec2i, device=device)
        self.data.mode = wp.zeros(n, dtype=int, device=device)
        self.data.axis = wp.zeros(n, dtype=wp.vec3, device=device)
        self.data.unresolved_work_items = wp.zeros(query_capacity, dtype=int, device=device)
        self.data.unresolved_work_count = wp.zeros(1, dtype=int, device=device)
        self.data.query_slot = wp.zeros(query_capacity, dtype=int, device=device)
        self.data.query_done = wp.zeros(query_capacity, dtype=int, device=device)
        self.data.stats = wp.zeros(8, dtype=int, device=device)
        self._graph_token = None

    def _acquire_graph(self, graph):
        if graph is None:
            return
        leases = getattr(graph, "_newton_coherent_convex_leases", None)
        if leases is None:
            leases = []
            graph._newton_coherent_convex_leases = leases
        if any(lease.owner is self for lease in leases):
            return
        if self._graph_token is not None:
            raise RuntimeError("experimental coherent convex cache cannot belong to two live CUDA graphs")
        leases.append(self._GraphLease(self))

    def _capture_owner(self):
        if self.device.is_cuda:
            stream = wp.get_stream(self.device)
            self._acquire_graph(getattr(self.device, "captures", {}).get(stream))

    def begin(self):
        self._capture_owner()
        wp.launch(_advance_generation, dim=1, inputs=[self.data], device=self.device)

    def reset(self, world_mask=None):
        self._capture_owner()
        if world_mask is None:
            self.data.mode.zero_()
        else:
            if (
                world_mask.dtype is not wp.bool
                or world_mask.ndim != 1
                or world_mask.shape[0] != self.data.world_epoch.shape[0]
                or world_mask.device != self.device
            ):
                raise ValueError(
                    "coherent convex reset requires a bool mask of shape (world_count + 1,) on the cache device"
                )
            wp.launch(
                _reset_world_epochs,
                dim=world_mask.shape[0],
                inputs=[self.data, world_mask],
                device=self.device,
            )
