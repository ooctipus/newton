# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Choose terrain support from sampled work and independent hardware primitives."""

import warp as wp

from ..utils.heightfield import HeightfieldData
from .contact_data import ContactData
from .convex_cell_cost import _calibrate_cell_cost, _support_saving
from .convex_cells import _CellData
from .heightfield_convex_cells import TerrainSupportData, get_query
from .support_function import GenericShapeData, SupportMapDataProvider, support_map, unpack_mesh_ptr
from .types import GeoType


@wp.struct
class _Counter:
    cells: _CellData
    costs: wp.array[float]
    work: wp.array2d[float]
    row: int


@wp.func
def _counted_support(geom: GenericShapeData, direction: wp.vec3, data: _Counter) -> wp.vec3:
    if geom.shape_type == GeoType.CONVEX_MESH:
        vertices = wp.mesh_get(unpack_mesh_ptr(geom.auxiliary)).points.shape[0]
        data.work[data.row, 0] += data.costs[0] + data.costs[1] * float(vertices)
        # This owner binds the descriptor for every convex support request.
        # Prisms retain the original provider without checking the choice.
        data.work[data.row, 1] += _support_saving(geom, direction, data.cells, data.costs) - data.costs[4]
    return support_map(geom, direction, SupportMapDataProvider())


@wp.func
def _discard_contact(contact: ContactData, data: _Counter, output_index: int):
    pass


def _sample_kernel():
    query = get_query(_counted_support, _discard_contact)

    @wp.kernel(module="unique", enable_backward=False)
    def sample_terrain_support(
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_gap: wp.array[float],
        shape_heightfield_index: wp.array[int],
        heightfield_data: wp.array[HeightfieldData],
        heightfield_elevations: wp.array[float],
        pairs: wp.array[wp.vec3i],
        pair_count: wp.array[int],
        cells: _CellData,
        costs: wp.array[float],
        work: wp.array2d[float],
        epoch: wp.array[int],
        period: int,
    ):
        if epoch[0] % period != 0:
            return
        row = wp.tid()
        work[row, 0] = 0.0
        work[row, 1] = 0.0
        work[row, 2] = 0.0
        count = wp.min(wp.max(0, pair_count[0]), pairs.shape[0])
        samples = wp.min(count, work.shape[0])
        if row >= samples:
            return
        index = int(wp.int64(row) * wp.int64(count) // wp.int64(samples))
        pair = pairs[index]
        if shape_types[pair[0]] != GeoType.HFIELD or shape_types[pair[1]] != GeoType.CONVEX_MESH:
            return
        data = _Counter()
        data.cells = cells
        data.costs = costs
        data.work = work
        data.row = row
        work[row, 2] = 1.0
        query(
            shape_types,
            shape_data,
            shape_transform,
            shape_source,
            shape_gap,
            shape_heightfield_index,
            heightfield_data,
            heightfield_elevations,
            pair[0],
            pair[1],
            pair[2],
            data,
            data,
        )

    return sample_terrain_support


@wp.kernel(enable_backward=False)
def _choose(
    work: wp.array2d[float],
    count_array: wp.array[int],
    capacity: int,
    epoch: wp.array[int],
    period: int,
    launch_ns: float,
    valid: wp.array[int],
    choice: wp.array[int],
    estimates: wp.array[float],
):
    if epoch[0] % period == 0:
        count = wp.min(wp.max(0, count_array[0]), capacity)
        samples = wp.min(count, work.shape[0])
        total = float(0.0)
        saving = float(0.0)
        square = float(0.0)
        eligible = float(0.0)
        for row in range(samples):
            total += work[row, 0]
            saving += work[row, 1]
            square += work[row, 1] * work[row, 1]
            eligible += work[row, 2]
        denominator = float(wp.max(1, samples))
        mean = saving / denominator
        uncertainty = 2.0 * wp.sqrt(wp.max(0.0, square / denominator - mean * mean) / denominator)
        ordinary_ns = total / denominator * float(count)
        # Price both selection launches, the separate terrain owner and the
        # periodically repeated original support work. Timings of task kernels
        # never enter these coefficients or the choice.
        overhead = 3.0 * launch_ns + total / float(period)
        saving_ns = (mean - uncertainty) * float(count) - overhead
        choice[0] = int(eligible >= 32.0 and wp.isfinite(saving_ns) and saving_ns > 0.15 * ordinary_ns)
        estimates[0] = ordinary_ns * 1.0e-6
        estimates[1] = (ordinary_ns - saving_ns) * 1.0e-6
        estimates[2] = uncertainty * float(count) * 1.0e-6
        estimates[3] = float(count)
        estimates[4] = eligible
    if valid[0] == 0:
        choice[0] = 0
    epoch[0] += 1


class TerrainMethodChoice:
    """Keep both support implementations valid throughout graph replay."""

    def __init__(self, cells):
        from .convex_query_cost import _measure, _primitive_empty  # noqa: PLC0415

        self.device = cells.model.device
        self.cells = cells
        self.model = _calibrate_cell_cost(self.device)
        scratch = wp.empty(1, dtype=wp.vec4, device=self.device)
        self.launch_ns = _measure(lambda: wp.launch(_primitive_empty, 1, [scratch], device=self.device), self.device)
        self.period = 64
        self.work = wp.zeros((512, 3), dtype=float, device=self.device)
        self.epoch = wp.zeros(1, dtype=int, device=self.device)
        self.choice = wp.zeros(1, dtype=int, device=self.device)
        self.estimates = wp.zeros(5, dtype=float, device=self.device)
        self.costs = wp.array(self.model.coefficients(), dtype=float, device=self.device)
        self.data = TerrainSupportData()
        self.data.cells, self.data.choice = cells.data, self.choice
        self.sample = _sample_kernel()

    def launch(self, triangle_inputs):
        wp.launch(
            self.sample,
            self.work.shape[0],
            [*triangle_inputs[:10], self.cells.data, self.costs, self.work, self.epoch, self.period],
            device=self.device,
        )
        wp.launch(
            _choose,
            1,
            [
                self.work,
                triangle_inputs[9],
                triangle_inputs[8].shape[0],
                self.epoch,
                self.period,
                self.launch_ns,
                self.cells.data.valid,
                self.choice,
                self.estimates,
            ],
            device=self.device,
        )

    def describe(self):
        values = self.estimates.numpy().tolist()
        return {
            "method": "convex_cells"
            if int(self.choice.numpy()[0]) and not self.cells.metadata["invalidated"]
            else "ordinary",
            "reason": "sampled_support_cost",
            "period": self.period,
            "support_estimate_ms": {"ordinary": values[0], "cells": values[1]},
            "uncertainty_ms": values[2],
            "triangle_queries": int(values[3]),
            "eligible_samples": int(values[4]),
        }
