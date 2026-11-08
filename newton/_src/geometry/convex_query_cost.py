# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent query primitives and sampled work for experimental method choice."""

import time
from dataclasses import dataclass
from statistics import median

import numpy as np
import warp as wp

from .coherent_convex import _plane_lower_bound
from .coherent_convex_rejection import _PairInputs
from .mpr import create_solve_mpr, create_support_map_function
from .simplex_solver import create_solve_closest_distance
from .support_function import GenericShapeData, SupportMapDataProvider, support_map_lean, unpack_mesh_ptr
from .types import GeoType


@wp.struct
class _WorkCounter:
    values: wp.array2d[float]
    row: int
    # A support provider's hull data (SupportMapDataProvider), left empty: the counted queries use no hull topology.
    shape_support_data: wp.array[wp.vec4i]
    support_neighbors: wp.array[int]


@wp.func
def _vertex_count(geom: GenericShapeData) -> int:
    if geom.shape_type == int(GeoType.CONVEX_MESH):
        return wp.mesh_get(unpack_mesh_ptr(geom.auxiliary)).points.shape[0]
    return 0


@wp.func
def _counted_support(geom: GenericShapeData, direction: wp.vec3, counter: _WorkCounter) -> wp.vec3:
    counter.values[counter.row, 0] += 1.0
    counter.values[counter.row, 1] += float(_vertex_count(geom))
    return support_map_lean(geom, direction, SupportMapDataProvider())


def _query_functions(*, count_work):
    support = _counted_support if count_work else support_map_lean
    maps = create_support_map_function(support, use_precomputed_center=True)
    return create_solve_mpr(support, _support_funcs=maps).core, create_solve_closest_distance(
        support, _support_funcs=maps
    ).core


def _primitive_query(*, count_work=False):
    mpr, gjk = _query_functions(count_work=count_work)

    @wp.kernel(module="unique", enable_backward=False)
    def query(geom: GenericShapeData, distance: float, metrics: wp.array2d[float], output: wp.array[wp.vec4]):
        row = wp.tid()
        orientation = wp.quat_rpy(0.2, -0.3, 0.4)
        position = wp.vec3(distance, 0.07 * float(row % 4), 0.03)
        if wp.static(count_work):
            metrics[row, 0] = 0.0
            metrics[row, 1] = 0.0
            provider = _WorkCounter()
            provider.values = metrics
            provider.row = row
        else:
            provider = SupportMapDataProvider()
        collision, _a, _b, normal, depth = mpr(geom, geom, orientation, position, 0.0, provider)
        gap = -depth
        if not collision:
            _separated, _a, _b, normal, gap = gjk(geom, geom, orientation, position, 0.0, provider)
        output[row] = wp.vec4(normal[0], normal[1], normal[2], gap)

    return query


@wp.kernel(enable_backward=False)
def _primitive_screen(geom: GenericShapeData, output: wp.array[wp.vec4]):
    row = wp.tid()
    bound = _plane_lower_bound(
        geom, geom, wp.quat_rpy(0.2, -0.3, 0.4), wp.vec3(1.2, 0.07 * float(row % 4), 0.03), wp.vec3(1.0, 0.1, 0.0)
    )
    output[row] = wp.vec4(bound.lower, bound.round_guard, bound.magnitude, float(bound.valid))


@wp.kernel(enable_backward=False)
def _primitive_lookup(keys: wp.array[wp.uint64], output: wp.array[wp.vec4]):
    row = wp.tid()
    key = keys[(row * 101) % keys.shape[0]]
    low = int(0)
    high = keys.shape[0]
    while low < high:
        middle = (low + high) // 2
        if keys[middle] < key:
            low = middle + 1
        else:
            high = middle
    output[row] = wp.vec4(float(low), float(key & wp.uint64(65535)), 0.0, 0.0)


@wp.kernel(enable_backward=False)
def _primitive_empty(output: wp.array[wp.vec4]):
    row = wp.tid()
    output[row] = wp.vec4(float(row), 0.0, 0.0, 0.0)


def _classification_primitive(device, count, launch_ns):
    from .coherent_convex import _ConvexQueryCache  # noqa: PLC0415
    from .coherent_convex_rejection import _create_rejection_query_kernels  # noqa: PLC0415
    from .collision_convex import ConvexQueryResult  # noqa: PLC0415

    poses = np.zeros((2 * count, 7), dtype=np.float32)
    poses[:, 6] = 1
    poses[1::2, 0] = 1.2
    inputs = _PairInputs()
    inputs.shape_types = wp.full(2 * count, int(GeoType.CONVEX_MESH), dtype=int, device=device)
    inputs.shape_data = wp.full(2 * count, wp.vec4(0.5, 0.4, 0.3, 0.0), dtype=wp.vec4, device=device)
    inputs.shape_transform = wp.array(poses, dtype=wp.transform, device=device)
    # Preparation packs the source pointer without dereferencing it.
    inputs.shape_source = wp.zeros(2 * count, dtype=wp.uint64, device=device)
    inputs.shape_support_data = wp.zeros(0, dtype=wp.vec4i, device=device)
    inputs.support_lut = wp.zeros(0, dtype=int, device=device)
    inputs.support_vertex_offsets = wp.zeros(0, dtype=int, device=device)
    inputs.support_neighbors = wp.zeros(0, dtype=int, device=device)
    inputs.shape_gap = wp.full(2 * count, 0.01, dtype=float, device=device)
    inputs.shape_collision_radius = wp.full(2 * count, 2.0, dtype=float, device=device)
    inputs.shape_aabb_lower = wp.full(2 * count, wp.vec3(-3.0), dtype=wp.vec3, device=device)
    inputs.shape_aabb_upper = wp.full(2 * count, wp.vec3(3.0), dtype=wp.vec3, device=device)
    inputs.shape_collision_aabb_lower = wp.full(2 * count, wp.vec3(-0.5), dtype=wp.vec3, device=device)
    inputs.shape_collision_aabb_upper = wp.full(2 * count, wp.vec3(0.5), dtype=wp.vec3, device=device)
    pairs = np.arange(2 * count, dtype=np.int32).reshape(count, 2)
    pair_array = wp.array(pairs, dtype=wp.vec2i, device=device)
    pair_count = wp.array([count], dtype=int, device=device)
    cache = _ConvexQueryCache(
        pairs=pairs,
        shape_types=inputs.shape_types.numpy(),
        shape_world=np.zeros(2 * count, dtype=np.int32),
        world_count=1,
        query_capacity=count,
        device=device,
    )
    buffers = [
        wp.empty(count, dtype=ConvexQueryResult, device=device),
        wp.empty(count, dtype=int, device=device),
        wp.zeros(1, dtype=int, device=device),
        wp.empty(count, dtype=int, device=device),
        wp.zeros(1, dtype=int, device=device),
    ]
    classify = _create_rejection_query_kernels()[0]

    def action():
        cache.begin()
        wp.launch(
            classify, count, [pair_array, pair_count, inputs, cache.data, count, *buffers], device=device, block_dim=128
        )

    # No cached axes: measure actual preparation, lookup and queue traffic.
    # Retain preparation in registers exactly as the production classifier does.
    elapsed = _measure(action, device)
    return max(0.0, elapsed - 2 * launch_ns) / count


def _nonnegative_fit(features, durations):
    """Solve the tiny nonnegative fit by checking every possible active set."""
    matrix, target = np.asarray(features), np.asarray(durations)
    best, error = np.zeros(matrix.shape[1]), float(np.dot(target, target))
    for mask in range(1, 1 << matrix.shape[1]):
        active = [column for column in range(matrix.shape[1]) if mask & (1 << column)]
        values = np.linalg.lstsq(matrix[:, active], target, rcond=None)[0]
        if np.any(values < 0):
            continue
        candidate = np.zeros(matrix.shape[1])
        candidate[active] = values
        residual = matrix @ candidate - target
        candidate_error = float(np.dot(residual, residual))
        if candidate_error < error:
            best, error = candidate, candidate_error
    return best


@dataclass(frozen=True)
class _QueryCostModel:
    """Nanoseconds per pair, support call and visited vertex; independent primitives only."""

    query_base_ns: float
    support_ns: float
    vertex_ns: float
    screen_base_ns: float
    screen_vertex_ns: float
    lookup_ns: float
    launch_ns: float
    calibration_seconds: float
    measurements: tuple
    classify_ns: float = 0.0

    def coefficients(self, slots):
        # The classifier primitive includes preparation, base lookup and queue
        # traffic. Only additional lookup depth is charged separately.
        lookup = self.lookup_ns * max(1.0, np.log2(max(2, slots)) / 13.0)
        return np.array(
            [
                self.query_base_ns,
                self.support_ns,
                self.vertex_ns,
                self.screen_base_ns,
                self.screen_vertex_ns,
                self.classify_ns + lookup - self.lookup_ns,
                self.launch_ns,
            ],
            dtype=np.float32,
        )


_DEVICE_MODELS = {}


def _measure(action, device):
    action()
    with wp.ScopedCapture(device=device) as capture:
        for _ in range(8):
            action()
    for _ in range(3):
        wp.capture_launch(capture.graph)
    values = []
    for _ in range(7):
        begin, end = wp.Event(device=device, enable_timing=True), wp.Event(device=device, enable_timing=True)
        wp.record_event(begin)
        wp.capture_launch(capture.graph)
        wp.record_event(end)
        values.append(wp.get_event_elapsed_time(begin, end) * 1.0e6 / 8)
    return median(values)


def _calibrate_query_cost(device):
    """Calibrate synthetic convex queries without a task, robot or task timing."""
    device = wp.get_device(device)
    key = (device.alias, device.uuid)
    if key in _DEVICE_MODELS:
        return _DEVICE_MODELS[key]
    if device.is_capturing:
        raise RuntimeError("Collision cost calibration must precede graph capture")
    start = time.monotonic()
    count = 8192
    records = []
    with wp.ScopedDevice(device):
        output = wp.zeros(count, dtype=wp.vec4, device=device)
        metrics = wp.zeros((count, 2), dtype=float, device=device)
        plain, counted = _primitive_query(), _primitive_query(count_work=True)
        launch_ns = _measure(lambda: wp.launch(_primitive_empty, 1, [output], device=device), device)
        keys = wp.array(np.arange(count, dtype=np.uint64), dtype=wp.uint64, device=device)
        lookup_ns = (
            max(
                0.0,
                _measure(lambda: wp.launch(_primitive_lookup, count, [keys, output], device=device), device)
                - launch_ns,
            )
            / count
        )
        features, durations, screen_features, screen_durations = [], [], [], []
        for vertices in (8, 32, 128, 256):
            if vertices == 8:
                points = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float32)
            else:
                index = np.arange(vertices)
                z = 1 - 2 * (index + 0.5) / vertices
                theta = index * (np.pi * (3 - np.sqrt(5)))
                radius = np.sqrt(1 - z * z)
                points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z)).astype(np.float32)
            mesh = wp.Mesh(
                points=wp.array(points, dtype=wp.vec3, device=device),
                indices=wp.array([0, 1, 2], dtype=int, device=device),
            )
            geom = GenericShapeData()
            geom.shape_type = int(GeoType.CONVEX_MESH)
            geom.scale = wp.vec3(0.5, 0.4, 0.3)
            geom.auxiliary = wp.vec3(mesh.id & 0x3FFFFF, (mesh.id >> 22) & 0x3FFFFF, (mesh.id >> 44) & 0xFFFFF)
            geom.center = wp.vec3()
            for distance in (0.2, 0.85, 1.2):
                elapsed = _measure(
                    lambda geom=geom, distance=distance: wp.launch(
                        plain, count, [geom, distance, metrics, output], device=device
                    ),
                    device,
                )
                wp.launch(counted, count, [geom, distance, metrics, output], device=device)
                calls, visits = metrics.numpy().mean(axis=0).tolist()
                features.append([1, calls, visits])
                durations.append(max(0.0, elapsed - launch_ns) / count)
                records.append(
                    {"vertices": vertices, "distance": distance, "ns": elapsed, "calls": calls, "visits": visits}
                )
            elapsed = _measure(
                lambda geom=geom: wp.launch(_primitive_screen, count, [geom, output], device=device), device
            )
            screen_features.append([1, 2 * vertices])
            screen_durations.append(max(0.0, elapsed - launch_ns) / count)
            records.append({"vertices": vertices, "screen_ns": elapsed})
        query = _nonnegative_fit(features, durations)
        screen = _nonnegative_fit(screen_features, screen_durations)
        classify_ns = _classification_primitive(device, count, launch_ns)
        records.append({"classify_ns_per_pair": classify_ns})
    model = _QueryCostModel(
        *query, *screen, lookup_ns, launch_ns, time.monotonic() - start, tuple(records), classify_ns=classify_ns
    )
    if not np.isfinite(model.coefficients(count)).all() or model.support_ns + model.vertex_ns <= 0:
        raise RuntimeError("Independent collision primitives produced an invalid cost model")
    _DEVICE_MODELS[key] = model
    return model


def _create_work_sampler():
    from .narrow_phase import create_prepare_convex_pair  # noqa: PLC0415

    prepare = create_prepare_convex_pair(True, False)
    mpr, gjk = _query_functions(count_work=True)

    @wp.kernel(module="unique", enable_backward=False)
    def sample(
        pairs: wp.array[wp.vec2i],
        count_array: wp.array[int],
        inputs: _PairInputs,
        epoch: wp.array[int],
        period: int,
        work: wp.array2d[float],
    ):
        if epoch[0] % period != 0:
            return
        row = wp.tid()
        for column in range(work.shape[1]):
            work[row, column] = 0.0
        count = wp.min(pairs.shape[0], count_array[0])
        if row >= count:
            return
        index = row
        if count > work.shape[0]:
            seed = wp.rand_init(epoch[0], row)
            index = wp.randi(seed, 0, count)
        pair = pairs[index]
        valid, query = prepare(
            pair,
            inputs.shape_types,
            inputs.shape_data,
            inputs.shape_transform,
            inputs.shape_source,
            inputs.shape_gap,
            inputs.shape_collision_radius,
            inputs.shape_aabb_lower,
            inputs.shape_aabb_upper,
            inputs.shape_collision_aabb_lower,
            inputs.shape_collision_aabb_upper,
        )
        if not valid:
            return
        work[row, 4] = 1.0
        provider = _WorkCounter()
        provider.values = work
        provider.row = row
        collision, point_a, point_b, normal, penetration = mpr(
            query.geom_a, query.geom_b, query.relative_orientation_b, query.relative_position_b, query.enlarge, provider
        )
        gap = -penetration + query.enlarge
        point_a -= normal * (0.5 * query.enlarge)
        point_b += normal * (0.5 * query.enlarge)
        if not collision:
            _separated, point_a, point_b, normal, gap = gjk(
                query.geom_a, query.geom_b, query.relative_orientation_b, query.relative_position_b, 0.0, provider
            )
        eligible = (query.type_a == int(GeoType.BOX) or query.type_a == int(GeoType.CONVEX_MESH)) and (
            query.type_b == int(GeoType.BOX) or query.type_b == int(GeoType.CONVEX_MESH)
        )
        if eligible:
            work[row, 5] = 1.0
            if gap > query.contact_threshold:
                # Estimate steady rejection opportunity from a fresh cold axis.
                # Only the actual classifier can reject a published query.
                work[row, 3] = float(_vertex_count(query.geom_a) + _vertex_count(query.geom_b))
                bound = _plane_lower_bound(
                    query.geom_a, query.geom_b, query.relative_orientation_b, query.relative_position_b, normal
                )
                if bound.valid != 0 and bound.lower > query.contact_threshold:
                    work[row, 2] = 1.0

    return sample


@wp.kernel(enable_backward=False)
def _choose_method(
    work: wp.array2d[float],
    count_array: wp.array[int],
    coefficients: wp.array[float],
    choice: wp.array[int],
    estimates: wp.array[float],
    period: int,
    epoch: wp.array[int],
    ordinary_count: wp.array[int],
    rejection_count: wp.array[int],
    rejection_gjk_count: wp.array[int],
):
    count = wp.max(0, count_array[0])
    if epoch[0] % period == 0:
        samples = wp.min(count, work.shape[0])
        total = float(0.0)
        saving_sum = float(0.0)
        saving_square = float(0.0)
        for row in range(samples):
            query = coefficients[0] * work[row, 4] + coefficients[1] * work[row, 0] + coefficients[2] * work[row, 1]
            # Prepare/lookup traffic applies to all candidates. Certificate scans
            # apply to eligible separated shapes; failed hints are budgeted too.
            screen = coefficients[5] + work[row, 5] * coefficients[3] + work[row, 3] * coefficients[4]
            saving = work[row, 2] * query - screen
            total += query
            saving_sum += saving
            saving_square += saving * saving
        denominator = float(wp.max(1, samples))
        mean = saving_sum / denominator
        variance = wp.max(0.0, saving_square / denominator - mean * mean)
        uncertainty = 2.0 * wp.sqrt(variance / denominator)
        ordinary_ns = total / denominator * float(count) + 2.0 * coefficients[6]
        # Extra rejection/empty-queue launches and amortized probes.
        overhead_ns = 6.0 * coefficients[6] + (total + 3.0 * coefficients[6]) / float(period)
        saving_ns = (mean - uncertainty) * float(count) - overhead_ns
        estimates[0] = ordinary_ns * 1.0e-6
        estimates[1] = (ordinary_ns - saving_ns) * 1.0e-6
        estimates[2] = uncertainty * float(count) * 1.0e-6
        estimates[3] = float(count)
        choice[0] = int(samples >= 32 and wp.isfinite(saving_ns) and saving_ns > 0.15 * ordinary_ns)
    epoch[0] += 1
    ordinary_count[0] = wp.where(choice[0] == 0, count, 0)
    rejection_count[0] = wp.where(choice[0] == 1, count, 0)
    rejection_gjk_count[0] = 0


class _ConvexMethodChoice:
    """Periodically choose on device; graph replay never uses host task labels."""

    def __init__(self, *, device, slots):
        self.device = wp.get_device(device)
        self.model = _calibrate_query_cost(self.device)
        self.period = 64
        self.epoch = wp.zeros(1, dtype=int, device=device)
        self.ordinary_count = wp.zeros(1, dtype=int, device=device)
        self.rejection_count = wp.zeros(1, dtype=int, device=device)
        self.rejection_gjk_count = wp.zeros(1, dtype=int, device=device)
        self.choice = wp.zeros(1, dtype=int, device=device)
        self.estimates = wp.zeros(4, dtype=float, device=device)
        self.work = wp.zeros((512, 6), dtype=float, device=device)
        self.coefficients = wp.array(self.model.coefficients(slots), dtype=float, device=device)
        self.sample = _create_work_sampler()

    def launch(self, pairs, count, inputs, ordinary, rejection):
        # Empty work queues keep the parent capture and its cross-stream events
        # intact. Native conditional graph capture pauses invalidate older events.
        wp.launch(
            self.sample,
            self.work.shape[0],
            [pairs, count, inputs, self.epoch, self.period, self.work],
            device=self.device,
        )
        wp.launch(
            _choose_method,
            1,
            [
                self.work,
                count,
                self.coefficients,
                self.choice,
                self.estimates,
                self.period,
                self.epoch,
                self.ordinary_count,
                self.rejection_count,
                self.rejection_gjk_count,
            ],
            device=self.device,
        )
        ordinary(self.ordinary_count)
        rejection(self.rejection_count, self.rejection_gjk_count)

    def describe(self):
        """Read the latest estimate outside capture; sampling is not a timing fit."""
        values = self.estimates.numpy().tolist()
        return {
            "method": ("ordinary", "reject_only")[self.choice.numpy()[0]],
            "query_ms": values[0],
            "rejection_ms": values[1],
            "uncertainty_ms": values[2],
            "candidates": int(values[3]),
            "period": self.period,
            "calibration_seconds": self.model.calibration_seconds,
            "source": "independent synthetic primitives and sampled support work",
        }
