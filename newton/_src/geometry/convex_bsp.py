# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental immutable-hull BSP support for the complete split-convex family."""

from functools import cache
from typing import Any

import numpy as np
import warp as wp

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
const double d[3] = {double(direction[0]), double(direction[1]), double(direction[2])};
if (!::isfinite(d[0]) || !::isfinite(d[1]) || !::isfinite(d[2])) return 2;
auto add = [](double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dadd_rn(a, b);
#else
    volatile double c = a + b; return double(c);
#endif
};
auto sub = [](double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dsub_rn(a, b);
#else
    volatile double c = a - b; return double(c);
#endif
};
auto mul = [](double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dmul_rn(a, b);
#else
    volatile double c = a * b; return double(c);
#endif
};
auto addf = [](float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fadd_rn(a, b);
#else
    volatile float c = a + b; return (float)(c);
#endif
};
auto mulf = [](float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    volatile float c = a * b; return (float)(c);
#endif
};
const float p0 = (float)(high[0]), p1 = (float)(high[1]), p2 = (float)(high[2]);
const float a0 = mulf(p0, direction[0]), a1 = mulf(p1, direction[1]), a2 = mulf(p2, direction[2]);
const float value32 = addf(addf(a0, a1), a2);
const float magnitude32 = addf(addf(::fabsf(a0), ::fabsf(a1)), ::fabsf(a2));
// 16*eps covers coefficient conversion and five rounded operations. The
// absolute term also covers FTZ of all three products. Nonzero stored plane
// components are at least 2^-106, hence normal in FP32 before multiplication.
const float error32 = addf(mulf(0x1p-19f, magnitude32), 0x1p-120f);
if (::isfinite(value32) && ::fabsf(value32) > error32) return value32 > 0.0f ? 1 : -1;
const double b0 = mul(high[0], d[0]), b1 = mul(high[1], d[1]), b2 = mul(high[2], d[2]);
const double value64 = add(add(b0, b1), b2);
const double magnitude64 = add(add(::fabs(b0), ::fabs(b1)), ::fabs(b2));
const double error64 = mul(0x1p-48, magnitude64);
if (::fabs(value64) > error64) return value64 > 0.0 ? 1 : -1;
// Six exact products (two FP64 coefficient limbs times three FP32 inputs).
// All exponents lie strictly inside normal FP64 range. Grow-expansion with
// explicit RN TwoSum and FMA TwoProduct preserves their exact sum and zero.
double expansion[12];
int length = 0;
for (int term = 0; term < 6; ++term) {
    const double coefficient = term < 3 ? high[term] : low[term - 3];
    const double operand = d[term % 3];
    const double product = mul(coefficient, operand);
    const double residual = ::fma(coefficient, operand, -product);
    for (int part = 0; part < 2; ++part) {
        double q = part == 0 ? residual : product;
        int out = 0;
        const int old_length = length;
        for (int j = 0; j < old_length; ++j) {
            const double item = expansion[j];
            const double total = add(q, item);
            const double bv = sub(total, q);
            const double av = sub(total, bv);
            const double br = sub(item, bv);
            const double ar = sub(q, av);
            const double error = add(ar, br);
            if (error != 0.0) expansion[out++] = error;
            q = total;
        }
        if (q != 0.0 || out == 0) expansion[out++] = q;
        length = out;
    }
}
for (int j = length - 1; j >= 0; --j) {
    if (expansion[j] > 0.0) return 1;
    if (expansion[j] < 0.0) return -1;
}
return 0;
""")
def _adaptive_sign(high: wp.vec3d, low: wp.vec3d, direction: wp.vec3) -> int:
    """Return an exact plane sign for admitted coefficients, or two for nonfinite input."""
    ...


@wp.struct
class _BspData:
    mesh_ids: wp.array[wp.uint64]
    roots: wp.array[int]
    plane_high: wp.array[wp.vec3d]
    plane_low: wp.array[wp.vec3d]
    node_plane: wp.array[int]
    children: wp.array[wp.vec2i]
    valid: wp.array[int]


@wp.struct
class _BspShape:
    shape_type: int
    scale: wp.vec3
    auxiliary: wp.vec3
    center: wp.vec3
    root: int


@wp.func
def _original_shape(geom: _BspShape) -> GenericShapeData:
    result = GenericShapeData()
    result.shape_type = geom.shape_type
    result.scale = geom.scale
    result.auxiliary = geom.auxiliary
    result.center = geom.center
    return result


@wp.func
def _bsp_box(geom: _BspShape, direction: wp.vec3) -> wp.vec3:
    return _support_map_box(_original_shape(geom), direction)


@wp.func
def _bind_shape(geom: GenericShapeData, data: _BspData) -> _BspShape:
    result = _BspShape()
    result.shape_type = geom.shape_type
    result.scale = geom.scale
    result.auxiliary = geom.auxiliary
    result.center = geom.center
    result.root = -1
    if geom.shape_type == GeoType.CONVEX_MESH and data.valid[0] != 0:
        pointer = unpack_mesh_ptr(geom.auxiliary)
        lo = int(0)
        hi = data.mesh_ids.shape[0]
        while lo < hi:
            middle = (lo + hi) // 2
            value = data.mesh_ids[middle]
            if value < pointer:
                lo = middle + 1
            else:
                hi = middle
        if lo < data.mesh_ids.shape[0] and data.mesh_ids[lo] == pointer:
            result.root = data.roots[lo]
    return result


@wp.func
def _query(data: _BspData, root: int, direction: wp.vec3) -> int:
    index = root
    while index >= 0:
        plane = data.node_plane[index]
        sign = _adaptive_sign(data.plane_high[plane], data.plane_low[plane], direction)
        if sign == 2:
            return -1
        pair = data.children[index]
        index = pair[0] if sign >= 0 else pair[1]
    return -index - 1


def _create_support(lean: bool):
    """Preserve the exact primitive callback while replacing admitted convex support."""
    original = support_map_lean if lean else support_map

    @wp.func
    def bsp_support(geom: Any, direction: wp.vec3, data: _BspData) -> wp.vec3:
        result = wp.vec3(0.0)
        winner = int(-1)
        if geom.shape_type == GeoType.CONVEX_MESH and geom.root >= 0:
            winner = _query(data, geom.root, wp.cw_mul(direction, geom.scale))
        if winner >= 0:
            mesh = wp.mesh_get(unpack_mesh_ptr(geom.auxiliary))
            result = wp.cw_mul(mesh.points[winner], geom.scale)
        else:
            result = original(_original_shape(geom), direction, SupportMapDataProvider())
        return result

    bsp_support._preserve_builtin_box_support = True
    bsp_support._builtin_box_support = _bsp_box
    bsp_support.__name__ = "bsp_support_lean" if lean else "bsp_support_full"
    return bsp_support


@cache
def _create_postprocess(lean: bool):
    """Forward unchanged contact postprocessing with the original shape descriptor type."""
    from .collision_core import post_process_axial_on_discrete_contact, post_process_minkowski_only  # noqa: PLC0415

    original = post_process_minkowski_only if lean else post_process_axial_on_discrete_contact

    @wp.func
    def bsp_postprocess(
        contact: ContactData,
        geom_a: _BspShape,
        position_a: wp.vec3,
        orientation_a: wp.quat,
        geom_b: _BspShape,
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

    return bsp_postprocess


class _BspOwner:
    """Own immutable exact-size descriptors until all referencing graphs are destroyed.

    Call ``invalidate`` before editing/refitting mesh points. Re-enable by
    constructing a new pipeline and recapturing its graphs. This owner never
    reactivates or replaces arrays, so old graphs safely retain original support.
    Unnotified writes to mesh memory violate this explicit opt-in contract.
    """

    def __init__(self, model):
        from .convex_bsp_build import build_tree  # noqa: PLC0415

        meshes = {mesh.id: mesh for mesh in model._mesh_keep_alive}
        pointers = np.unique(model.shape_source_ptr.numpy()[model.shape_type.numpy() == GeoType.CONVEX_MESH])
        ids, roots, highs, lows, planes, children = [], [], [], [], [], []
        self.metadata = {"admitted_meshes": [], "unsupported_meshes": [], "invalidated": False}
        for pointer in sorted(int(p) for p in pointers if int(p) in meshes):
            mesh = meshes[pointer]
            try:
                vertices = mesh.points.numpy()
                faces = mesh.indices.numpy().reshape(-1, 3)
                if not 4 <= len(vertices) <= 256 or not np.isfinite(vertices).all():
                    raise ValueError("BSP requires four to 256 finite stored vertices")
                tree = build_tree(vertices, faces)
            except (ValueError, AssertionError, OverflowError) as error:
                self.metadata["unsupported_meshes"].append({"mesh_id": pointer, "reason": str(error)})
                continue
            plane_map = {}
            start = len(children)
            for plane, positive, negative in tree["nodes"]:
                if plane not in plane_map:
                    plane_map[plane] = len(highs)
                    highs.append(tree["rounded64"][plane])
                    lows.append(tree["plane_low"][plane])
                planes.append(plane_map[plane])
                children.append(
                    (positive + start if positive >= 0 else positive, negative + start if negative >= 0 else negative)
                )
            ids.append(pointer)
            roots.append(start + tree["root"])
            self.metadata["admitted_meshes"].append(
                {
                    "mesh_id": pointer,
                    "vertices": len(vertices),
                    "nodes": len(tree["nodes"]),
                    "max_depth": max(d for _, d in tree["leaves"]),
                }
            )
        self.data = _BspData()
        device = model.device
        self.data.mesh_ids = wp.array(ids, dtype=wp.uint64, device=device)
        self.data.roots = wp.array(roots, dtype=int, device=device)
        self.data.plane_high = wp.array(np.asarray(highs).reshape(-1, 3), dtype=wp.vec3d, device=device)
        self.data.plane_low = wp.array(np.asarray(lows).reshape(-1, 3), dtype=wp.vec3d, device=device)
        self.data.node_plane = wp.array(planes, dtype=int, device=device)
        self.data.children = wp.array(np.asarray(children).reshape(-1, 2), dtype=wp.vec2i, device=device)
        self.data.valid = wp.full(1, 1, dtype=int, device=device)
        self.metadata["descriptor_bytes"] = sum(
            a.capacity
            for a in (
                self.data.mesh_ids,
                self.data.roots,
                self.data.plane_high,
                self.data.plane_low,
                self.data.node_plane,
                self.data.children,
                self.data.valid,
            )
        )
        self.model = model

    def invalidate(self):
        """Disable all old graph-visible trees before a geometry edit, without replacing arrays."""
        self.data.valid.zero_()
        self.metadata["invalidated"] = True
