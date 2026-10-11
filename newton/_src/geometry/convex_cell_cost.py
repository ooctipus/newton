# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent support primitives for choosing immutable direction cells."""

import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import warp as wp

from .coherent_convex import _support_feature
from .convex_cells import _bind_shape, _cell_support_feature, _CellData, _CellOwner, _create_support, _direction_cell
from .support_function import GenericShapeData, SupportMapDataProvider, support_map_lean, unpack_mesh_ptr
from .types import GeoType


@wp.func
def _candidate_count(shape: GenericShapeData, direction: wp.vec3, cells: _CellData) -> int:
    bound = _bind_shape(shape, cells)
    if bound.descriptor >= 0:
        cell = _direction_cell(wp.cw_mul(direction, shape.scale))
        if cell >= 0:
            mask = cells.masks[bound.descriptor * 1536 + cell]
            count = int(0)
            while mask != wp.uint64(0):
                count += 1
                mask = mask & (mask - wp.uint64(1))
            return count
    return -1


@wp.func
def _support_saving(
    shape: GenericShapeData, direction: wp.vec3, cells: _CellData, costs: wp.array[float], feature: bool = False
) -> float:
    offset = wp.where(feature, 5, 0)
    # Failed descriptors retain the original support and pay the cell check.
    if shape.shape_type != int(GeoType.CONVEX_MESH):
        return -costs[offset + 2]
    candidates = _candidate_count(shape, direction, cells)
    if candidates < 0:
        return -costs[offset + 2]
    vertices = wp.mesh_get(unpack_mesh_ptr(shape.auxiliary)).points.shape[0]
    return (
        costs[offset] + costs[offset + 1] * float(vertices) - costs[offset + 2] - costs[offset + 3] * float(candidates)
    )


def _support_primitive(*, cells=False, counted=False, feature=False):
    support = _create_support(True)

    @wp.kernel(module="unique", enable_backward=False)
    def primitive(
        shape: GenericShapeData,
        data: _CellData,
        rays: wp.array[wp.vec3],
        calls: int,
        output: wp.array[wp.vec3],
        visits: wp.array[float],
    ):
        row = wp.tid()
        result = wp.vec3(0.0)
        candidates = float(0.0)
        if wp.static(cells):
            bound = _bind_shape(shape, data)
        for i in range(calls):
            direction = rays[(row + i * 17) % rays.shape[0]]
            if wp.static(feature):
                if wp.static(cells):
                    found = _cell_support_feature(bound, direction, data)
                else:
                    found = _support_feature(shape, direction)
                result += found.point + wp.vec3(found.value, found.magnitude, float(found.valid + found.index))
            elif wp.static(cells):
                result += support(bound, direction, data)
            else:
                result += support_map_lean(shape, direction, SupportMapDataProvider())
            if wp.static(counted):
                candidates += float(_candidate_count(shape, direction, data))
        output[row] = result
        if wp.static(counted):
            visits[row] = candidates

    return primitive


@dataclass(frozen=True)
class _CellCostModel:
    """Support and descriptor nanoseconds, measured without task timings."""

    scan_call_ns: float
    scan_vertex_ns: float
    cell_call_ns: float
    cell_vertex_ns: float
    bind_ns: float
    calibration_seconds: float
    measurements: tuple
    scan_base_ns: float = 0.0
    cell_base_ns: float = 0.0
    feature_costs: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)

    def coefficients(self):
        return np.array(
            [
                self.scan_call_ns,
                self.scan_vertex_ns,
                self.cell_call_ns,
                self.cell_vertex_ns,
                self.bind_ns,
                *self.feature_costs,
            ],
            dtype=np.float32,
        )


_DEVICE_MODELS = {}


def _fit_support_cost(records, *, feature=False):
    from .convex_query_cost import _nonnegative_fit  # noqa: PLC0415

    prefix = "feature_" if feature else ""
    scan = _nonnegative_fit(
        [[1, r["calls"], r["calls"] * r["vertices"]] for r in records],
        [r[prefix + "scan_ns"] for r in records],
    )
    cell = _nonnegative_fit(
        [[1, r["calls"], r["candidates"]] for r in records], [r[prefix + "cell_ns"] for r in records]
    )
    # Both primitives write one result per row. Charge only the intercept
    # difference to binding, not output traffic shared with full scans.
    return (scan[1], scan[2], cell[1], cell[2], max(0.0, cell[0] - scan[0])), (scan[0], cell[0])


def _calibrate_cell_cost(device):
    """Measure full scans and candidate scans on synthetic closed convex hulls."""
    from .convex_query_cost import _measure, _primitive_empty  # noqa: PLC0415

    device = wp.get_device(device)
    key = (device.alias, device.uuid)
    if key in _DEVICE_MODELS:
        return _DEVICE_MODELS[key]
    if device.is_capturing:
        raise RuntimeError("Cell cost calibration must precede graph capture")
    start = time.monotonic()
    count = 8192
    rays = np.random.default_rng(719).normal(size=(count, 3)).astype(np.float32)
    directions = wp.array(rays, dtype=wp.vec3, device=device)
    output = wp.empty(count, dtype=wp.vec3, device=device)
    visits = wp.empty(count, dtype=float, device=device)
    plain, cells, counted = _support_primitive(), _support_primitive(cells=True), _support_primitive(counted=True)
    feature_plain, feature_cells = _support_primitive(feature=True), _support_primitive(feature=True, cells=True)
    empty_output = wp.empty(1, dtype=wp.vec4, device=device)
    launch = _measure(lambda: wp.launch(_primitive_empty, 1, [empty_output], device=device), device)
    records = []
    for vertices in (8, 16, 32, 64):
        # Bipyramids need no hull library and exercise different candidate counts.
        angle = np.arange(vertices - 2) * (2 * np.pi / (vertices - 2))
        points = np.concatenate(
            [[[0, 0, 0.25], [0, 0, -0.25]], np.column_stack((0.25 * np.cos(angle), 0.25 * np.sin(angle), angle * 0))]
        ).astype(np.float32)
        faces = []
        for i in range(vertices - 2):
            a, b = i + 2, (i + 1) % (vertices - 2) + 2
            faces.extend([[0, a, b], [1, b, a]])
        mesh = wp.Mesh(
            wp.array(points, dtype=wp.vec3, device=device), wp.array(np.ravel(faces), dtype=int, device=device)
        )
        model = SimpleNamespace(
            _mesh_keep_alive=[mesh],
            device=device,
            shape_source_ptr=wp.array([mesh.id], dtype=wp.uint64, device=device),
            shape_type=wp.array([int(GeoType.CONVEX_MESH)], dtype=int, device=device),
        )
        owner = _CellOwner(model)
        if not owner.metadata["admitted_meshes"]:
            raise RuntimeError("Synthetic cell calibration hull was not admitted")
        shape = GenericShapeData()
        shape.shape_type = int(GeoType.CONVEX_MESH)
        shape.scale = wp.vec3(1.0)
        shape.auxiliary = wp.vec3(mesh.id & 0x3FFFFF, (mesh.id >> 22) & 0x3FFFFF, (mesh.id >> 44) & 0xFFFFF)
        for calls in (1, 8, 32):
            args = [shape, owner.data, directions, calls, output, visits]
            wp.launch(counted, count, args, device=device)
            candidates = float(visits.numpy().mean())
            scan_ns = (
                max(0.0, _measure(lambda args=args: wp.launch(plain, count, args, device=device), device) - launch)
                / count
            )
            cell_ns = (
                max(0.0, _measure(lambda args=args: wp.launch(cells, count, args, device=device), device) - launch)
                / count
            )
            feature_scan_ns = (
                max(
                    0.0,
                    _measure(lambda args=args: wp.launch(feature_plain, count, args, device=device), device) - launch,
                )
                / count
            )
            feature_cell_ns = (
                max(
                    0.0,
                    _measure(lambda args=args: wp.launch(feature_cells, count, args, device=device), device) - launch,
                )
                / count
            )
            records.append(
                {
                    "vertices": vertices,
                    "calls": calls,
                    "candidates": candidates,
                    "scan_ns": scan_ns,
                    "cell_ns": cell_ns,
                    "feature_scan_ns": feature_scan_ns,
                    "feature_cell_ns": feature_cell_ns,
                }
            )
    support, intercepts = _fit_support_cost(records)
    feature, _ = _fit_support_cost(records, feature=True)
    result = _CellCostModel(
        *support,
        time.monotonic() - start,
        tuple(records),
        scan_base_ns=intercepts[0],
        cell_base_ns=intercepts[1],
        feature_costs=feature,
    )
    if not np.isfinite(result.coefficients()).all():
        raise RuntimeError("Independent support primitives produced an invalid cell cost model")
    _DEVICE_MODELS[key] = result
    return result
