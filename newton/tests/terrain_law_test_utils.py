# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures of the terrain law tests: terrains and the float64 oracle's view of a model."""

from __future__ import annotations

import numpy as np

import newton
from newton.tests import terrain_law_reference as ref

# ----------------------------------------------------------------------------------------------------------------
# terrains
# ----------------------------------------------------------------------------------------------------------------


def heightfield_model(heights, hx, hy, *, device, min_z=None, max_z=None, extra=()):
    """A model holding heightfields (each ``(heights, hx, hy[, xform])``) and nothing else."""
    builder = newton.ModelBuilder()
    for item in ((heights, hx, hy), *extra):
        data, sx, sy = item[:3]
        data = np.asarray(data, np.float32)
        terrain = newton.Heightfield(
            data=data, nrow=data.shape[0], ncol=data.shape[1], hx=sx, hy=sy, min_z=min_z, max_z=max_z
        )
        builder.add_shape_heightfield(heightfield=terrain, xform=item[3] if len(item) > 3 else None)
    return builder.finalize(device=device)


def reference_surface(model, heightfield: int = 0) -> ref.ReferenceHeightfield:
    """The float64 oracle's surface of heightfield index ``heightfield`` of a model (narrow-phase decode)."""
    row = model.heightfield_data.numpy()[heightfield]
    nrow, ncol, offset = int(row["nrow"]), int(row["ncol"]), int(row["data_offset"])
    e = model.heightfield_elevations.numpy()[offset : offset + nrow * ncol]
    Z = ref.decode_heights(e, nrow, ncol, float(row["min_z"]), float(row["max_z"]))
    return ref.ReferenceHeightfield(Z, float(row["hx"]), float(row["hy"]))


def suite_like_heights(rng, rows: int = 64, cols: int = 64, *, tile: int = 32):
    """A mosaic of the suite's sub-terrains on 0.1 m cells: noisy flat, quantized rough, stairs, slope, boxes.

    Returns float64 heights ``(rows + 1, cols + 1)`` [m]. The flat tile carries +-1.4 um of noise, as the recorded
    suite terrain's flat border does; the rough tile is uniform noise in 5 mm steps.
    """
    nrow, ncol = rows + 1, cols + 1
    r, c = np.meshgrid(np.arange(nrow), np.arange(ncol), indexing="ij")
    Z = np.zeros((nrow, ncol))
    kind = ((r // tile) * 7 + (c // tile) * 3) % 5
    noise = rng.integers(-6, 7, size=(nrow, ncol)) * 2.4e-7
    rough = np.round(rng.uniform(0.0, 0.08, size=(nrow, ncol)) / 0.005) * 0.005
    stairs = np.floor((c % tile) / 4) * 0.05
    slope = np.tan(np.radians(12.0)) * (0.06 * (c % tile) - 0.08 * (r % tile)) + 0.3
    boxes = np.where(((r // 6) + (c // 6)) % 3 == 0, 0.12, 0.0)
    Z = np.where(kind == 0, noise, Z)
    Z = np.where(kind == 1, rough, Z)
    Z = np.where(kind == 2, stairs, Z)
    Z = np.where(kind == 3, slope, Z)
    Z = np.where(kind == 4, boxes + noise, Z)
    return Z
