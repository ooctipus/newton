# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental immutable direction-cell candidates for original support queries."""

from functools import cache
from typing import Any

import numpy as np
import warp as wp

from .coherent_convex import _finite_vector, _support_feature, _SupportFeature
from .contact_data import ContactData
from .support_function import (
    GenericShapeData,
    SupportMapDataProvider,
    _support_map_box,
    support_map,
    support_map_lean,
    unpack_mesh_ptr,
)
from .types import GeoType


@wp.func_native("""
float absolute[3] = {::fabsf(direction[0]), ::fabsf(direction[1]), ::fabsf(direction[2])};
if (!::isfinite(absolute[0]) || !::isfinite(absolute[1]) || !::isfinite(absolute[2])) return -1;
int axis = absolute[1] > absolute[0] ? 1 : 0;
if (absolute[2] > absolute[axis]) axis = 2;
float magnitude = absolute[axis];
if (!(magnitude >= 0x1p-60f && magnitude <= 0x1p60f)) return -1;
int first = axis == 0 ? 1 : 0;
int second = axis == 2 ? 1 : 2;
#if defined(__CUDA_ARCH__)
float x = __fmul_rn(__fadd_rn(__fdiv_rn(direction[first], magnitude), 1.0f), 8.0f);
float y = __fmul_rn(__fadd_rn(__fdiv_rn(direction[second], magnitude), 1.0f), 8.0f);
#else
volatile float qx = direction[first] / magnitude, qy = direction[second] / magnitude;
volatile float sx = qx + 1.0f, sy = qy + 1.0f;
float x = sx * 8.0f, y = sy * 8.0f;
#endif
int ix = static_cast<int>(x), iy = static_cast<int>(y);
ix = ix < 0 ? 0 : (ix > 15 ? 15 : ix);
iy = iy < 0 ? 0 : (iy > 15 ? 15 : iy);
int face = 2*axis + (direction[axis] < 0.0f ? 1 : 0);
return face*256 + ix*16 + iy;
""")
def _direction_cell(direction: wp.vec3) -> int:
    """Use explicitly rounded normalization within the offline coverage range."""
    ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return __ffsll(mask)-1;
#else
return __builtin_ctzll(mask);
#endif
""")
def _first_bit(mask: wp.uint64) -> int:
    """Enumerate a nonempty candidate mask in original vertex order."""
    ...


@wp.struct
class _CellData:
    mesh_ids: wp.array[wp.uint64]
    masks: wp.array[wp.uint64]
    norm_one: wp.array[float]
    valid: wp.array[int]


@wp.struct
class _CellShape:
    shape_type: int
    scale: wp.vec3
    auxiliary: wp.vec3
    center: wp.vec3
    descriptor: int
    points: wp.array[wp.vec3]


@wp.func
def _original_shape(geom: _CellShape) -> GenericShapeData:
    result = GenericShapeData()
    result.shape_type = geom.shape_type
    result.scale = geom.scale
    result.auxiliary = geom.auxiliary
    result.center = geom.center
    return result


@wp.func
def _cell_box(geom: _CellShape, direction: wp.vec3) -> wp.vec3:
    return _support_map_box(_original_shape(geom), direction)


@wp.func
def _bind_shape(geom: GenericShapeData, data: _CellData) -> _CellShape:
    result = _CellShape()
    result.shape_type = geom.shape_type
    result.scale = geom.scale
    result.auxiliary = geom.auxiliary
    result.center = geom.center
    result.descriptor = -1
    if geom.shape_type == GeoType.CONVEX_MESH and data.valid[0] != 0:
        pointer = unpack_mesh_ptr(geom.auxiliary)
        lo = int(0)
        hi = data.mesh_ids.shape[0]
        while lo < hi:
            middle = (lo + hi) // 2
            if data.mesh_ids[middle] < pointer:
                lo = middle + 1
            else:
                hi = middle
        if lo < data.mesh_ids.shape[0] and data.mesh_ids[lo] == pointer:
            result.descriptor = lo
            result.points = wp.mesh_get(pointer).points
    return result


@wp.func
def _query(geom: _CellShape, direction: wp.vec3, data: _CellData):
    winner = int(-1)
    score = float(0.0)
    if geom.descriptor >= 0:
        cell = _direction_cell(direction)
        if cell >= 0:
            mask = data.masks[geom.descriptor * 1536 + cell]
            while mask != wp.uint64(0):
                index = _first_bit(mask)
                value = wp.dot(geom.points[index], direction)
                if winner < 0 or value > score:
                    winner = index
                    score = value
                mask = mask & (mask - wp.uint64(1))
    return winner, score


@cache
def _create_support(lean: bool):
    """Preserve original primitive policies and sentinel semantics on fallback."""
    original = support_map_lean if lean else support_map

    @wp.func
    def cell_support(geom: Any, direction: wp.vec3, data: _CellData) -> wp.vec3:
        winner, score = _query(geom, wp.cw_mul(direction, geom.scale), data)
        result = wp.vec3(0.0)
        if winner >= 0 and score > -1.0e10:
            result = wp.cw_mul(geom.points[winner], geom.scale)
        else:
            result = original(_original_shape(geom), direction, SupportMapDataProvider())
        return result

    cell_support._preserve_builtin_box_support = True
    cell_support._builtin_box_support = _cell_box
    return cell_support


@wp.func
def _cell_support_feature(geom: _CellShape, direction: wp.vec3, data: _CellData) -> _SupportFeature:
    result = _SupportFeature()
    result.index = -1
    if geom.descriptor >= 0 and _finite_vector(direction) and _finite_vector(geom.scale):
        winner, value = _query(geom, wp.cw_mul(geom.scale, direction), data)
        if winner >= 0:
            result.index = winner
            result.value = value
            result.point = wp.cw_mul(geom.points[winner], geom.scale)
            scale = wp.max(wp.abs(geom.scale[0]), wp.max(wp.abs(geom.scale[1]), wp.abs(geom.scale[2])))
            result.magnitude = data.norm_one[geom.descriptor] * scale
            result.valid = int(_finite_vector(result.point) and wp.isfinite(result.magnitude))
            return result
    return _support_feature(_original_shape(geom), direction)


@cache
def _create_postprocess(lean: bool):
    """Forward the unchanged contact law with original typed shape descriptors."""
    from .collision_core import post_process_axial_on_discrete_contact, post_process_minkowski_only  # noqa: PLC0415

    original = post_process_minkowski_only if lean else post_process_axial_on_discrete_contact

    @wp.func
    def cell_postprocess(
        contact: ContactData,
        geom_a: _CellShape,
        position_a: wp.vec3,
        orientation_a: wp.quat,
        geom_b: _CellShape,
        position_b: wp.vec3,
        orientation_b: wp.quat,
    ) -> ContactData:
        return original(
            contact,
            _original_shape(geom_a),
            position_a,
            orientation_a,
            _original_shape(geom_b),
            position_b,
            orientation_b,
        )

    return cell_postprocess


class _CellOwner:
    """Own immutable tables; geometry invalidation withdraws them for existing graphs."""

    def __init__(self, model):
        from .convex_cells_build import build_masks  # noqa: PLC0415

        meshes = {mesh.id: mesh for mesh in model._mesh_keep_alive}
        pointers = np.unique(model.shape_source_ptr.numpy()[model.shape_type.numpy() == GeoType.CONVEX_MESH])
        ids, tables, norms = [], [], []
        self.metadata = {"admitted_meshes": [], "unsupported_meshes": [], "invalidated": False, "resolution": 16}
        for pointer in sorted(int(p) for p in pointers):
            try:
                mesh = meshes.get(pointer)
                if mesh is None:
                    raise ValueError("Cooked mesh lifetime is not model-owned")
                vertices = mesh.points.numpy()
                if (
                    vertices.ndim != 2
                    or vertices.shape[1] != 3
                    or not 4 <= len(vertices) <= 64
                    or not np.isfinite(vertices).all()
                    or np.any((vertices != 0) & (abs(vertices) < np.finfo(np.float32).tiny))
                ):
                    raise ValueError("Cell support requires 4--64 finite normal-coordinate vertices")
                masks, details = build_masks(vertices, mesh.indices.numpy().reshape(-1, 3))
            except (ValueError, OverflowError) as error:
                self.metadata["unsupported_meshes"].append({"mesh_id": pointer, "reason": str(error)})
                continue
            ids.append(pointer)
            tables.append(masks.reshape(-1))
            norm = np.max(np.sum(abs(vertices).astype(float), axis=1)) * (1 + 4 * np.finfo(np.float32).eps)
            norms.append(np.nextafter(np.float32(norm), np.float32(np.inf)))
            self.metadata["admitted_meshes"].append({"mesh_id": pointer, "vertices": len(vertices), **details})
        data = _CellData()
        device = model.device
        data.mesh_ids = wp.array(ids, dtype=wp.uint64, device=device)
        data.masks = wp.array(np.concatenate(tables) if tables else [], dtype=wp.uint64, device=device)
        data.norm_one = wp.array(norms, dtype=float, device=device)
        data.valid = wp.full(1, 1, dtype=int, device=device)
        self.data, self.model = data, model
        self.metadata["mask_bytes"] = data.masks.capacity
        self.metadata["pair_hint_bytes"] = 0
        self.metadata["descriptor_bytes"] = sum(
            getattr(data, name).capacity for name in ("mesh_ids", "masks", "norm_one", "valid")
        )

    def invalidate(self):
        """Disable descriptors before geometry edits while retaining graph pointers."""
        self.data.valid.zero_()
        self.metadata["invalidated"] = True
