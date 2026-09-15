# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Opt-in complete pair-local heightfield query and contact reduction.

Two sixteen-lane pairs share a CUDA block, but never scratch or barriers.
Overflow pairs are replayed by a separate serial, bounded-winner owner before
any of their contacts are published. Public and legacy capacity counters remain
live; the old triangle/reducer allocations are reserved, not used as a cache.
"""

import functools
import inspect
import linecache
import textwrap
from types import SimpleNamespace
from typing import Any

import numpy as np
import warp as wp

from ..utils.heightfield import HeightfieldData, get_triangle_shape_from_heightfield
from .collision_core import create_compute_gjk_mpr_contacts
from .contact_data import ContactData
from .contact_reduction import compute_voxel_index, get_slot, get_spatial_direction_2d, project_point_to_plane
from .contact_reduction_global import (
    BETA_THRESHOLD,
    _floats_are_near_ulps,
    compute_effective_radius,
    decode_oct,
    encode_oct,
    make_contact_value,
)
from .heightfield_features import WELD_FLAT_SEAMS, HeightfieldFeatureContext, flat_seam_query_allowed
from .heightfield_finite import query_contacts
from .heightfield_geometric import _geometric_midphase, _prepare_pair
from .support_function import create_triangle_prism_penetration_refiner, extract_shape_data, support_map
from .types import GeoType


@wp.struct
class Scene:
    types: wp.array[int]
    data: wp.array[wp.vec4]
    transforms: wp.array[wp.transform]
    sources: wp.array[wp.uint64]
    gaps: wp.array[float]
    heightfield_index: wp.array[int]
    heightfield: wp.array[HeightfieldData]
    elevations: wp.array[float]
    lower: wp.array[wp.vec3]
    upper: wp.array[wp.vec3]
    voxels: wp.array[wp.vec3i]
    bounds: wp.array2d[wp.vec3]
    bound_source: wp.array[wp.uint64]
    pairs: wp.array[wp.vec2i]
    pair_count: wp.array[int]
    triangle_count: wp.array[int]
    callback_count: wp.array[int]
    overflow: wp.array[int]
    status: wp.array[int]
    triangle_capacity: int
    callback_capacity: int


@wp.struct
class Scratch:
    ptr: wp.uint64
    capacity: int
    streaming: int
    a: int
    b: int
    inverse_a: wp.transform
    lower: wp.vec3
    upper: wp.vec3
    voxels: wp.vec3i
    beta: float


# Arena: 245 uint64 scores, 64 triangle IDs, packed 7-word records, and four
# header words (triangle count, callback count, two exported-ID bitmap words).
# Exceptional arenas store one record per score slot, and do not store triangles.
@functools.cache
def _arena(capacity, streaming):
    words = 490 + (0 if streaming else 64) + 7 * (245 if streaming else capacity) + 4
    words = (words + 1) & ~1
    copies = 1 if streaming else 2
    snippet = f"""
#if defined(__CUDA_ARCH__)
__shared__ __align__(8) unsigned int arena[{copies * words}];
const int group = {"0" if streaming else "(threadIdx.x & 31) >> 4"};
return reinterpret_cast<uint64_t>(arena + group * {words});
#else
return reinterpret_cast<uint64_t>(malloc({words} * sizeof(unsigned int)));
#endif
"""

    @wp.func_native(snippet)
    def arena() -> wp.uint64: ...

    return arena


@wp.func_native("""
#if defined(__CUDA_ARCH__)
if (streaming == 0) __syncwarp(0xffffu << (16 * ((threadIdx.x & 31) >> 4)));
#endif
""")
def _sync(streaming: int): ...


@wp.func_native("""
#if !defined(__CUDA_ARCH__)
free(reinterpret_cast<void*>(ptr));
#endif
""")
def _release(ptr: wp.uint64): ...


@wp.func_native("""
auto p = reinterpret_cast<unsigned int*>(s.ptr);
const int records = s.streaming ? 245 : s.capacity;
const int head = 490 + (s.streaming ? 0 : 64) + 7 * records;
if (lane == 0) for (int k=0;k<4;++k) p[head+k]=0;
auto scores = reinterpret_cast<uint64_t*>(p);
for (int k=lane;k<245;k+=lanes) scores[k]=0;
""")
def _clear(s: Scratch, lane: int, lanes: int): ...


@wp.func_native("""
auto p = reinterpret_cast<unsigned int*>(s.ptr);
const int head = 490 + (s.streaming ? 0 : 64) + 7*(s.streaming ? 245 : s.capacity);
return static_cast<int>(p[head+field]);
""")
def _count(s: Scratch, field: int) -> int: ...


@wp.func_native("""
auto p = reinterpret_cast<unsigned int*>(s.ptr);
const int head=490+64+7*s.capacity;
const int i=static_cast<int>(p[head]++);
if (i<64) p[490+i]=static_cast<unsigned int>(tri);
""")
def _append_triangle(s: Scratch, tri: int): ...


@wp.func_native("return static_cast<int>(reinterpret_cast<unsigned int*>(s.ptr)[490+i]);")
def _triangle(s: Scratch, i: int) -> int: ...


@wp.func_native("""
auto p = reinterpret_cast<unsigned int*>(s.ptr);
const int head=490+(s.streaming ? 0 : 64)+7*(s.streaming ? 245 : s.capacity);
#if defined(__CUDA_ARCH__)
return static_cast<int>(atomicAdd(p+head+1,1u));
#else
return static_cast<int>(p[head+1]++);
#endif
""")
def _reserve(s: Scratch) -> int: ...


@wp.func_native("""
auto p=reinterpret_cast<unsigned int*>(s.ptr)+490+(s.streaming ? 0 : 64)+7*i;
auto f=reinterpret_cast<float*>(p);
for(int k=0;k<4;++k) f[k]=pd[k];
f[4]=normal[0]; f[5]=normal[1]; p[6]=static_cast<unsigned int>(fingerprint);
""")
def _store(s: Scratch, i: int, pd: wp.vec4, normal: wp.vec2, fingerprint: int): ...


@wp.func_native("""
auto f=reinterpret_cast<float*>(s.ptr)+490+(s.streaming ? 0 : 64)+7*i;
return wp::vec4(f[0],f[1],f[2],f[3]);
""")
def _pd(s: Scratch, i: int) -> wp.vec4: ...


@wp.func_native("""
auto f=reinterpret_cast<float*>(s.ptr)+490+(s.streaming ? 0 : 64)+7*i;
return wp::vec2(f[4],f[5]);
""")
def _normal(s: Scratch, i: int) -> wp.vec2: ...


@wp.func_native("""
auto p=reinterpret_cast<unsigned int*>(s.ptr)+490+(s.streaming ? 0 : 64)+7*i;
return static_cast<int>(p[6]);
""")
def _fingerprint(s: Scratch, i: int) -> int: ...


@wp.func_native("return reinterpret_cast<uint64_t*>(s.ptr)[slot];")
def _value(s: Scratch, slot: int) -> wp.uint64: ...


@wp.func_native("reinterpret_cast<uint64_t*>(s.ptr)[slot]=0;")
def _erase(s: Scratch, slot: int): ...


@wp.func_native("""
auto p=reinterpret_cast<unsigned long long*>(s.ptr)+slot;
if(s.streaming) { if(value>*p) {*p=value;return true;} return false; }
#if defined(__CUDA_ARCH__)
atomicMax(p,static_cast<unsigned long long>(value));
#else
if(value>*p) *p=value;
#endif
return false;
""")
def _update(s: Scratch, slot: int, value: wp.uint64) -> bool: ...


@wp.func_native("""
const unsigned int id=static_cast<unsigned int>(value);
if(s.streaming) {
    auto p=reinterpret_cast<uint64_t*>(s.ptr);
    for(int k=0;k<slot;++k) if(p[k] && static_cast<unsigned int>(p[k])==id) return false;
    return true;
}
auto p=reinterpret_cast<unsigned int*>(s.ptr);
const int head=490+64+7*s.capacity;
const unsigned int mask=1u<<((id-1)&31);
#if defined(__CUDA_ARCH__)
return !(atomicOr(p+head+2+((id-1)>>5),mask)&mask);
#else
auto at=p+head+2+((id-1)>>5);const auto old=*at;*at|=mask;return !(old&mask);
#endif
""")
def _first_export(s: Scratch, slot: int, value: wp.uint64) -> bool: ...


@wp.func
def _choose(s: Scratch, slot: int, score: float, i: int, pd: wp.vec4, normal: wp.vec2, fingerprint: int):
    value = make_contact_value(score, fingerprint, i + 1, 0)
    if _update(s, slot, value):
        # Only the exceptional single-lane callback writes winner payloads.
        _store(s, slot, pd, normal, fingerprint)


@wp.func
def _write_local(contact: ContactData, s: Scratch, ignored: int):
    i = _reserve(s)
    if s.streaming == 0 and i >= s.capacity:
        return
    p = contact.contact_point_center
    pd = wp.vec4(p[0], p[1], p[2], contact.contact_distance)
    encoded = encode_oct(contact.contact_normal_a_to_b)
    normal = decode_oct(encoded)
    fingerprint = contact.sort_sub_key
    if s.streaming == 0:
        _store(s, i, pd, encoded, fingerprint)
    bin_id = get_slot(normal)
    pos_2d = project_point_to_plane(bin_id, p)
    if pd[3] < s.beta:
        for k in range(6):
            _choose(s, 7 * bin_id + k, wp.dot(pos_2d, get_spatial_direction_2d(k)), i, pd, encoded, fingerprint)
    _choose(s, 7 * bin_id + 6, -pd[3], i, pd, encoded, fingerprint)
    local = wp.transform_point(s.inverse_a, p)
    voxel = wp.clamp(compute_voxel_index(local, s.lower, s.upper, s.voxels), 0, 99)
    _choose(s, 140 + voxel, -pd[3], i, pd, encoded, fingerprint)


@wp.func
def _equivalent(s: Scratch, a: int, b: int) -> bool:
    pa, pb = _pd(s, a), _pd(s, b)
    na, nb = _normal(s, a), _normal(s, b)
    for k in range(4):
        if not _floats_are_near_ulps(pa[k], pb[k]):
            return False
    return _floats_are_near_ulps(na[0], nb[0]) and _floats_are_near_ulps(na[1], nb[1])


@wp.func
def _suppress_bin(s: Scratch, bin_id: int):
    # Match the original simultaneous 21 comparisons, then suppress slots.
    bits = int(0)
    for a in range(7):
        sa = 7 * bin_id + a
        va = _value(s, sa)
        if va == wp.uint64(0):
            continue
        for b in range(a + 1, 7):
            sb = 7 * bin_id + b
            vb = _value(s, sb)
            if vb == wp.uint64(0) or wp.uint32(va) == wp.uint32(vb):
                continue
            ra, rb = int(wp.uint32(va)) - 1, int(wp.uint32(vb)) - 1
            if s.streaming != 0:
                ra, rb = sa, sb
            if _equivalent(s, ra, rb):
                if _fingerprint(s, rb) < _fingerprint(s, ra):
                    bits = bits | (1 << a)
                else:
                    bits = bits | (1 << b)
    for k in range(7):
        if bits & (1 << k) != 0:
            _erase(s, 7 * bin_id + k)


@wp.func
def _finite_triangle(tri: int, scene: Scene, s: Scratch) -> bool:
    a, b = s.a, s.b
    if scene.types[b] != GeoType.BOX and scene.types[b] != GeoType.CONVEX_MESH:
        return False
    if scene.bounds[b, 1][0] <= 0.0 or scene.bound_source[b] != scene.sources[b]:
        return False
    margin_a, margin_b = scene.data[a][3], scene.data[b][3]
    if margin_a + margin_b < 1.0e-4:
        return False
    data = scene.data[b]
    scale = wp.vec3(data[0], data[1], data[2])
    if scale[0] <= 0.0 or scale[1] <= 0.0 or scale[2] <= 0.0:
        return False
    xa, xb = scene.transforms[a], scene.transforms[b]
    qa = wp.normalize(wp.transform_get_rotation(xa))
    qb = wp.normalize(wp.transform_get_rotation(xb))
    geom, origin = get_triangle_shape_from_heightfield(
        scene.heightfield[scene.heightfield_index[a]], scene.elevations, xa, tri
    )
    center_world = wp.transform_get_translation(xb) + wp.quat_rotate(qb, wp.cw_mul(scene.bounds[b, 0], scale))
    center = wp.quat_rotate_inv(qa, center_world - origin)
    gap = scene.gaps[a] + scene.gaps[b]
    value = query_contacts(
        geom.scale,
        geom.auxiliary,
        center,
        wp.quat_inverse(qa) * qb,
        wp.cw_mul(scene.bounds[b, 1], scale),
        gap + margin_a + margin_b,
        margin_a + margin_b,
    )
    count = int(value[0])
    if count < 0:
        return False
    local_normal = wp.vec3(value[1], value[2], value[3])
    feature = HeightfieldFeatureContext()
    if wp.static(WELD_FLAT_SEAMS):
        feature.heightfield = scene.heightfield[scene.heightfield_index[a]]
        feature.elevations = scene.elevations
        feature.triangle = tri
    normal = wp.quat_rotate(qa, local_normal)
    for k in range(count):
        if wp.static(WELD_FLAT_SEAMS):
            terrain_point = wp.vec3(value[4 + 4 * k], value[5 + 4 * k], value[6 + 4 * k])
            terrain_point -= 0.5 * value[7 + 4 * k] * local_normal
            if value[7 + 4 * k] >= 0.0 and not flat_seam_query_allowed(geom, terrain_point, local_normal, feature):
                continue
        contact = ContactData()
        contact.shape_a = a
        contact.shape_b = b
        contact.margin_a = margin_a
        contact.margin_b = margin_b
        contact.gap_sum = gap
        contact.contact_point_center = origin + wp.quat_rotate(
            qa, wp.vec3(value[4 + 4 * k], value[5 + 4 * k], value[6 + 4 * k])
        )
        contact.contact_normal_a_to_b = normal
        contact.contact_distance = value[7 + 4 * k]
        contact.sort_sub_key = (((tri << 1) | 1) << 3) | k
        _write_local(contact, s, -1)
    return True


@wp.func
def _query_triangle(tri: int, scene: Scene, s: Scratch):
    if _finite_triangle(tri, scene, s):
        return
    a, b = s.a, s.b
    hfd = scene.heightfield[scene.heightfield_index[a]]
    geom, origin = get_triangle_shape_from_heightfield(hfd, scene.elevations, scene.transforms[a], tri)
    pos_b, quat_b, geom_b, _scale_b, margin_b = extract_shape_data(
        b, scene.transforms, scene.types, scene.data, scene.sources
    )
    feature = HeightfieldFeatureContext()
    if wp.static(WELD_FLAT_SEAMS):
        feature.elevations = scene.elevations
        feature.triangle = tri
        feature.heightfield = hfd
    wp.static(
        create_compute_gjk_mpr_contacts(
            _write_local,
            penetration_refiner=create_triangle_prism_penetration_refiner(support_map),
            query_filter=flat_seam_query_allowed if WELD_FLAT_SEAMS else None,
        )
    )(
        geom,
        geom_b,
        wp.transform_get_rotation(scene.transforms[a]),
        quat_b,
        origin,
        pos_b,
        scene.gaps[a] + scene.gaps[b],
        a,
        b,
        scene.data[a][3],
        margin_b,
        s,
        (tri << 1) | 1,
        feature,
    )


@wp.func
def _collect_triangle(tri: int, scene: Scene, s: Scratch):
    _append_triangle(s, tri)


@functools.cache
def _traversal(visitor):
    """Reuse the existing range/cell/geometric cull body, changing only append."""
    source = textwrap.dedent(inspect.getsource(_geometric_midphase.func))
    source = source[source.index("def ") :]
    source = source.replace("def _geometric_midphase(", "def visit_pair(", 1)
    seam = "    triples: wp.array[wp.vec3i],\n    count: wp.array[int],"
    if source.count(seam) != 1:
        raise RuntimeError("Heightfield traversal signature changed")
    source = source.replace(seam, "    scene: Scene,\n    s: Scratch,", 1)
    source = source.replace(
        "if _triangle_is_separated(pair, v0, v1, v2):",
        "if pair.enabled and _triangle_is_separated(pair, v0, v1, v2):",
        1,
    )
    seam = "                out = wp.atomic_add(count, 0, 1)\n                if out < triples.shape[0]:\n                    triples[out] = wp.vec3i(a, b, tri)"
    if source.count(seam) != 1:
        raise RuntimeError("Heightfield append changed")
    source = source.replace(seam, "                wp.static(visitor)(tri, scene, s)", 1)
    name = f"<pair_terrain_traversal_{visitor.key}>"
    linecache.cache[name] = (len(source), None, source.splitlines(True), name)
    namespace = dict(_geometric_midphase.func.__globals__)
    namespace.update(Scene=Scene, Scratch=Scratch, visitor=visitor)
    exec(compile(source, name, "exec"), namespace)
    return wp.func(namespace["visit_pair"])


@functools.cache
def create_pair_kernels(writer_func, local_capacity=64):
    """Return complete fast/exceptional owners; a smaller capacity forces replay."""
    if not isinstance(local_capacity, int) or not 1 <= local_capacity <= 64:
        raise ValueError("Pair local capacity must be in [1,64]")

    @wp.func
    def publish(s: Scratch, scene: Scene, slot: int, writer_data: Any):
        value = _value(s, slot)
        if value == wp.uint64(0) or not _first_export(s, slot, value):
            return
        record = int(wp.uint32(value)) - 1
        if s.streaming != 0:
            record = slot
        pd = _pd(s, record)
        contact = ContactData()
        contact.contact_point_center = wp.vec3(pd[0], pd[1], pd[2])
        contact.contact_distance = pd[3]
        contact.contact_normal_a_to_b = decode_oct(_normal(s, record))
        contact.shape_a = s.a
        contact.shape_b = s.b
        contact.margin_a = scene.data[s.a][3]
        contact.margin_b = scene.data[s.b][3]
        contact.radius_eff_a = compute_effective_radius(scene.types[s.a], scene.data[s.a])
        contact.radius_eff_b = compute_effective_radius(scene.types[s.b], scene.data[s.b])
        contact.gap_sum = scene.gaps[s.a] + scene.gaps[s.b]
        contact.sort_sub_key = _fingerprint(s, record)
        wp.static(writer_func)(contact, writer_data, -1)

    def make(streaming):
        arena = _arena(local_capacity, streaming)
        visitor = _traversal(_query_triangle if streaming else _collect_triangle)

        @wp.kernel(enable_backward=False, module="unique")
        def heightfield_pair_owner(scene: Scene, writer_data: Any, num_groups: int):
            group, lane = wp.tid()
            lanes = wp.block_dim()
            if wp.static(not streaming):
                if lanes > 1:
                    lanes = 16
            elif lane != 0:
                return
            for i in range(group, wp.min(scene.pair_count[0], scene.pairs.shape[0]), num_groups):
                if wp.static(streaming):
                    if scene.overflow[i] == 0:
                        continue
                s = Scratch()
                s.ptr = wp.static(arena)()
                s.capacity = wp.static(local_capacity)
                s.streaming = wp.static(int(streaming))
                s.a = scene.pairs[i][0]
                s.b = scene.pairs[i][1]
                s.inverse_a = wp.transform_inverse(scene.transforms[s.a])
                s.lower = scene.lower[s.a]
                s.upper = scene.upper[s.a]
                s.voxels = scene.voxels[s.a]
                s.beta = wp.static(BETA_THRESHOLD) * wp.length(s.upper - s.lower)
                if wp.static(streaming):
                    lanes = 1
                _clear(s, lane, lanes)
                _sync(s.streaming)
                if lane == 0:
                    hfd = scene.heightfield[scene.heightfield_index[s.a]]
                    projection = _prepare_pair(
                        s.a,
                        s.b,
                        hfd,
                        scene.types,
                        scene.transforms,
                        scene.data,
                        scene.sources,
                        scene.gaps,
                        scene.lower,
                        scene.upper,
                        scene.bounds,
                        scene.bound_source,
                    )
                    wp.static(visitor)(
                        s.a,
                        s.b,
                        hfd,
                        projection,
                        scene.transforms,
                        scene.lower,
                        scene.upper,
                        scene.data,
                        scene.gaps,
                        scene.elevations,
                        scene,
                        s,
                    )
                    if wp.static(not streaming):
                        triangles = _count(s, 0)
                        start = wp.atomic_add(scene.triangle_count, 0, triangles)
                        scene.overflow[i] = int(triangles > 64)
                        if start + triangles > scene.triangle_capacity:
                            wp.atomic_or(scene.status, 0, 1)
                _sync(s.streaming)
                if wp.static(not streaming):
                    if _count(s, 0) > 64:
                        _release(s.ptr)
                        continue
                    for t in range(lane, _count(s, 0), lanes):
                        _query_triangle(_triangle(s, t), scene, s)
                    _sync(s.streaming)
                    if _count(s, 1) > wp.static(local_capacity):
                        if lane == 0:
                            scene.overflow[i] = 1
                        _sync(s.streaming)
                        _release(s.ptr)
                        continue
                if lane == 0:
                    callbacks = _count(s, 1)
                    start = wp.atomic_add(scene.callback_count, 0, callbacks)
                    if start + callbacks > scene.callback_capacity:
                        wp.atomic_or(scene.status, 0, 2)
                for bin_id in range(lane, 35, lanes):
                    _suppress_bin(s, bin_id)
                _sync(s.streaming)
                for slot in range(lane, 245, lanes):
                    publish(s, scene, slot, writer_data)
                _sync(s.streaming)
                _release(s.ptr)

        return heightfield_pair_owner

    return make(False), make(True)


def bind(narrow, pairs_np, shape_types):
    """Activate only for immutable unique explicit pairs and the admitted scene."""
    if not narrow._heightfield_pair_reducer_requested:
        return
    terrain = int(GeoType.HFIELD)
    normalized = []
    for a, b in np.asarray(pairs_np).reshape(-1, 2):
        if shape_types[a] == terrain or shape_types[b] == terrain:
            normalized.append((min(int(a), int(b)), max(int(a), int(b))))
    if len(set(normalized)) != len(normalized):
        return
    fast, exceptional = create_pair_kernels(narrow._convex_writer_func)
    narrow._pair_terrain = SimpleNamespace(
        fast=fast,
        exceptional=exceptional,
        overflow=wp.zeros(narrow.max_candidate_pairs, dtype=int, device=narrow.device),
        callback_count=wp.zeros(1, dtype=int, device=narrow.device),
        status=wp.zeros(1, dtype=int, device=narrow.device),
    )
    narrow._heightfield_pair_reducer = True


def launch(narrow, scene, writer_data, device):
    owner = narrow._pair_terrain
    scene.overflow, scene.callback_count, scene.status = owner.overflow, owner.callback_count, owner.status
    scene.pairs, scene.pair_count = narrow.shape_pairs_mesh, narrow.shape_pairs_mesh_count
    scene.triangle_count = narrow.triangle_pairs_count
    scene.triangle_capacity = narrow.max_triangle_pairs
    scene.callback_capacity = narrow.global_contact_reducer.capacity
    owner.callback_count.zero_()
    # Status is sticky like the existing public capacity status. Active pair
    # overflow slots are overwritten by fast traversal on every call.
    groups = 2 * narrow.num_tile_blocks
    width = 1 if wp.get_device(device).is_cpu else 16
    wp.launch(
        owner.fast,
        dim=(groups, width),
        inputs=[scene, writer_data, groups],
        device=device,
        block_dim=1 if width == 1 else 32,
        record_tape=False,
    )
    wp.launch(
        owner.exceptional,
        dim=(narrow.num_tile_blocks, 1 if width == 1 else 32),
        inputs=[scene, writer_data, narrow.num_tile_blocks],
        device=device,
        block_dim=1 if width == 1 else 32,
        record_tape=False,
    )
