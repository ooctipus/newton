# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optional raw-witness pair CSR and adaptive four-contact publication.

The packed finite and generic queries remain separate. Only integer pair tags
are added to their original callback records. Unsupported pairs keep the old
hash reducer; admitted pairs select from all writer-accepted raw witnesses,
not from the old reducer's winners. No pair-local candidate cap is imposed.
"""

import functools
import inspect
import linecache
import textwrap
from typing import Any

import numpy as np
import warp as wp

from .contact_data import ContactData
from .contact_reduction import float_flip
from .contact_reduction_global import (
    BETA_THRESHOLD,
    GlobalContactReducerData,
    compute_effective_radius,
    create_export_reduced_contacts_kernel,
    export_contact_to_buffer,
    mesh_triangle_contacts_to_reducer_kernel,
    reduce_contact_in_hashtable,
    unpack_contact,
)
from .heightfield_manifold import _selection_score
from .types import GeoType

# The error interval below assumes round-to-nearest IEEE arithmetic, not
# fast-math reassociation/approximation. It bounds the represented witness's
# stock endpoint arithmetic, not collision-query or body-local roundtrip error.
wp.set_module_options({"fast_math": False})
_UNIT_ROUNDOFF = 2.0**-24
_GAMMA16 = 16.0 * _UNIT_ROUNDOFF / (1.0 - 16.0 * _UNIT_ROUNDOFF)
_GAMMA8 = 8.0 * _UNIT_ROUNDOFF / (1.0 - 8.0 * _UNIT_ROUNDOFF)
_INTERVAL_FACTOR = wp.constant(float(np.nextafter(np.float32(_GAMMA16 / (1.0 - _GAMMA8)), np.float32(np.inf))))
_INTERVAL_UNDERFLOW = wp.constant(64.0 * 2.0**-126)


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return nextafterf(x, INFINITY);
#else
// Warp's WP_NO_CRT CPU module does not declare nextafterf. This is the
// identical binary32 successor, including signed zero, infinities and NaNs.
union { float f; unsigned int u; } value;
value.f=x;
if ((value.u&0x7fffffffu)>0x7f800000u || value.u==0x7f800000u) return x;
if ((value.u&0x7fffffffu)==0u) value.u=1u;
else if (value.u&0x80000000u) --value.u;
else ++value.u;
return value.f;
#endif
""")
def _next_up(x: float) -> float: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return nextafterf(x, -INFINITY);
#else
union { float f; unsigned int u; } value;
value.f=x;
if ((value.u&0x7fffffffu)>0x7f800000u || value.u==0xff800000u) return x;
if ((value.u&0x7fffffffu)==0u) value.u=0x80000001u;
else if (value.u&0x80000000u) ++value.u;
else --value.u;
return value.f;
#endif
""")
def _next_down(x: float) -> float: ...


@wp.func
def _writer_separation_interval(
    position: wp.vec3, normal: wp.vec3, depth: float, ra: float, rb: float, ma: float, mb: float
) -> wp.vec3:
    """Return literal stock separation and an outward arithmetic interval.

    ``normal`` is unpack_contact's already-normalized output. The second
    normalization below matches write_contact exactly. Expanded endpoint/dot
    terms have at most sixteen rounding factors. The positive absolute scale
    has at most eight sequential factors, so gamma16/(1-gamma8) also covers
    its downward rounding. The constants are rounded upward explicitly.
    """
    n = wp.normalize(normal)
    total = ra + rb + ma + mb
    point_a = position - n * (0.5 * depth + ra)
    point_b = position + n * (0.5 * depth + rb)
    distance = wp.dot(point_b - point_a, n)
    phi = distance - total
    amplitude = wp.abs(depth) + wp.abs(ra) + wp.abs(rb)
    scale = wp.abs(ra) + wp.abs(rb) + wp.abs(ma) + wp.abs(mb)
    for axis in range(3):
        component = wp.abs(n[axis])
        scale += component * (2.0 * wp.abs(position[axis]) + component * amplitude)
    error = _next_up(_next_up(wp.static(_INTERVAL_FACTOR) * scale) + wp.static(_INTERVAL_UNDERFLOW))
    lower = _next_down(phi - error)
    upper = _next_up(phi + error)
    # Finite endpoints catch intermediate overflow even if later arithmetic
    # could otherwise hide it. Invalid intervals cause the caller's sticky
    # failure, never an implicit geometry drop or a broadened admission gap.
    for axis in range(3):
        if not wp.isfinite(n[axis]) or not wp.isfinite(point_a[axis]) or not wp.isfinite(point_b[axis]):
            lower = wp.inf
    if not wp.isfinite(total) or not wp.isfinite(distance) or not wp.isfinite(amplitude) or not wp.isfinite(scale):
        lower = wp.inf
    return wp.vec3(phi, lower, upper)


@wp.struct
class PairCSRData:
    reducer: GlobalContactReducerData
    triangle_pair: wp.array[int]
    raw_pair: wp.array[int]
    counts: wp.array[int]
    offsets: wp.array[int]
    cursors: wp.array[int]
    ids: wp.array[int]
    separation: wp.array[wp.vec2]
    status: wp.array[int]
    pair_index: int


@wp.func
def _write_pair(contact: ContactData, data: PairCSRData, ignored: int):
    contact_id = export_contact_to_buffer(
        contact.shape_a,
        contact.shape_b,
        contact.contact_point_center,
        contact.contact_normal_a_to_b,
        contact.contact_distance,
        contact.sort_sub_key,
        data.reducer,
    )
    if contact_id < 0:
        # The original allocator undoes a failed reservation. Its counter alone
        # therefore cannot certify that the complete raw witness pool fitted.
        wp.atomic_or(data.status, 0, 1)
    else:
        data.raw_pair[contact_id] = data.pair_index


@wp.kernel(enable_backward=False)
def count_contacts(
    data: PairCSRData,
    pairs: wp.array[wp.vec2i],
    pair_count: wp.array[int],
    types: wp.array[int],
    shape_data: wp.array[wp.vec4],
    gaps: wp.array[float],
    total_threads: int,
):
    count = data.reducer.contact_count[0]
    npairs = pair_count[0]
    if count < 0 or count > data.reducer.capacity or npairs < 0 or npairs > pairs.shape[0]:
        wp.atomic_or(data.status, 0, 2)
        return
    for index in range(wp.tid(), count, total_threads):
        contact_id = index + 1
        pair = data.raw_pair[contact_id]
        if pair < 0 or pair >= npairs or pair >= data.cursors.shape[0]:
            data.raw_pair[contact_id] = -2
            wp.atomic_or(data.status, 0, 4)
            continue
        shapes = data.reducer.shape_pairs[contact_id]
        expected_shapes = pairs[pair]
        if shapes[0] != expected_shapes[0] or shapes[1] != expected_shapes[1]:
            data.raw_pair[contact_id] = -2
            wp.atomic_or(data.status, 0, 4)
            continue
        pd = data.reducer.position_depth[contact_id]
        normal = data.reducer.normal[contact_id]
        if not (
            wp.isfinite(pd[0])
            and wp.isfinite(pd[1])
            and wp.isfinite(pd[2])
            and wp.isfinite(pd[3])
            and wp.isfinite(normal[0])
            and wp.isfinite(normal[1])
        ):
            data.raw_pair[contact_id] = -2
            wp.atomic_or(data.status, 0, 16)
            continue
        a, b = shapes[0], shapes[1]
        if a < 0 or b < 0 or a >= types.shape[0] or b >= types.shape[0]:
            data.raw_pair[contact_id] = -2
            wp.atomic_or(data.status, 0, 4)
            continue
        if types[a] != GeoType.HFIELD or (types[b] != GeoType.BOX and types[b] != GeoType.CONVEX_MESH):
            data.raw_pair[contact_id] = -1
        else:
            point, n, depth = unpack_contact(contact_id, data.reducer.position_depth, data.reducer.normal)
            interval = _writer_separation_interval(
                point,
                n,
                depth,
                compute_effective_radius(types[a], shape_data[a]),
                compute_effective_radius(types[b], shape_data[b]),
                shape_data[a][3],
                shape_data[b][3],
            )
            if not wp.isfinite(interval[0]) or not wp.isfinite(interval[1]) or not wp.isfinite(interval[2]):
                data.raw_pair[contact_id] = -2
                wp.atomic_or(data.status, 0, 32)
            elif not (interval[0] > gaps[a] + gaps[b]):
                data.separation[contact_id] = wp.vec2(interval[1], interval[2])
                wp.atomic_add(data.counts, pair, 1)
            else:
                data.raw_pair[contact_id] = -2


@wp.kernel(enable_backward=False)
def scatter_contacts(data: PairCSRData, pair_count: wp.array[int], total_threads: int):
    if data.status[0] != 0:
        return
    count = data.reducer.contact_count[0]
    npairs = pair_count[0]
    if data.offsets[data.offsets.shape[0] - 1] > data.ids.shape[0]:
        wp.atomic_or(data.status, 0, 8)
        return
    for index in range(wp.tid(), count, total_threads):
        contact_id = index + 1
        pair = data.raw_pair[contact_id]
        if pair >= 0:
            if pair >= npairs:
                wp.atomic_or(data.status, 0, 4)
                continue
            local = wp.atomic_add(data.cursors, pair, 1)
            destination = data.offsets[pair] + local
            if local >= data.counts[pair] or destination < 0 or destination >= data.ids.shape[0]:
                wp.atomic_or(data.status, 0, 8)
            else:
                data.ids[destination] = contact_id


@wp.kernel(enable_backward=False)
def reduce_fallback_contacts(
    data: PairCSRData,
    shape_transform: wp.array[wp.transform],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    voxels: wp.array[wp.vec3i],
    total_threads: int,
):
    if data.status[0] != 0:
        return
    for index in range(wp.tid(), data.reducer.contact_count[0], total_threads):
        if data.raw_pair[index + 1] == -1:
            reduce_contact_in_hashtable(
                index + 1, data.reducer, wp.static(BETA_THRESHOLD), shape_transform, lower, upper, voxels
            )


@functools.cache
def create_export_kernel(writer_func):
    """Create one warp per pair's variable-length raw-ID bucket."""

    @wp.kernel(enable_backward=False, module=f"heightfield_pair_csr_{writer_func.__name__}")
    def export_heightfield_pair_csr(
        data: PairCSRData,
        pair_count: wp.array[int],
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_gap: wp.array[float],
        writer_data: Any,
        total_blocks: int,
        parallel_pairs: int,
    ):
        block, lane = wp.tid()
        if data.status[0] != 0:
            return
        width = int(1)
        if parallel_pairs != 0:
            width = 32
        scores = wp.tile_zeros(shape=32, dtype=wp.uint64, storage="shared")
        for pair in range(block, pair_count[0], total_blocks):
            begin, end = data.offsets[pair], data.offsets[pair + 1]
            if begin == end:
                continue
            # One additional membership scan; intervals were decoded once in
            # count_contacts. The min-upper cohort contains all potentially
            # active witnesses, or the closest represented positive footprint.
            nearest = wp.uint64(0)
            for index in range(begin + lane, end, width):
                contact_id = data.ids[index]
                upper = data.separation[contact_id][1]
                key = (wp.uint64(float_flip(-upper)) << wp.uint64(32)) | (wp.uint64(0xFFFFFFFF) - wp.uint64(contact_id))
                nearest = wp.max(nearest, key)
            wp.tile_scatter_masked(scores, lane, nearest, True)
            nearest_key = wp.tile_reduce(wp.max, scores)[0]
            nearest_id = int(wp.uint64(0xFFFFFFFF) - (nearest_key & wp.uint64(0xFFFFFFFF)))
            cutoff = wp.max(0.0, data.separation[nearest_id][1])
            selected = wp.vec4i(0)
            p0, p1, p2, n0 = wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0)
            for stage in range(4):
                best = wp.uint64(0)
                for index in range(begin + lane, end, width):
                    contact_id = data.ids[index]
                    if (
                        contact_id == selected[0]
                        or contact_id == selected[1]
                        or contact_id == selected[2]
                        or contact_id == selected[3]
                    ):
                        continue
                    pd = data.reducer.position_depth[contact_id]
                    interval = data.separation[contact_id]
                    score = -interval[1]
                    if stage > 0 and interval[0] <= cutoff:
                        # Spread scores are nonnegative; all outside-cohort
                        # upper bounds are positive. The existing full32-bit
                        # score therefore gives exact priority without adding
                        # another reduction or dropping score precision.
                        score = _selection_score(stage, wp.vec3(pd[0], pd[1], pd[2]), pd[3], p0, p1, p2, n0)
                    key = (wp.uint64(float_flip(score)) << wp.uint64(32)) | (
                        wp.uint64(0xFFFFFFFF) - wp.uint64(contact_id)
                    )
                    best = wp.max(best, key)
                wp.tile_scatter_masked(scores, lane, best, True)
                winner_key = wp.tile_reduce(wp.max, scores)[0]
                winner = int(0)
                if winner_key != wp.uint64(0):
                    winner = int(wp.uint64(0xFFFFFFFF) - (winner_key & wp.uint64(0xFFFFFFFF)))
                selected[stage] = winner
                if winner != 0:
                    point, normal, depth = unpack_contact(winner, data.reducer.position_depth, data.reducer.normal)
                    if stage == 0:
                        p0, n0 = point, normal
                    elif stage == 1:
                        p1 = point
                    elif stage == 2:
                        p2 = point
                    if lane == 0:
                        shapes = data.reducer.shape_pairs[winner]
                        a, b = shapes[0], shapes[1]
                        contact = ContactData()
                        contact.contact_point_center = point
                        contact.contact_normal_a_to_b = normal
                        contact.contact_distance = depth
                        contact.radius_eff_a = compute_effective_radius(shape_types[a], shape_data[a])
                        contact.radius_eff_b = compute_effective_radius(shape_types[b], shape_data[b])
                        contact.margin_a = shape_data[a][3]
                        contact.margin_b = shape_data[b][3]
                        contact.shape_a = a
                        contact.shape_b = b
                        contact.gap_sum = shape_gap[a] + shape_gap[b]
                        contact.sort_sub_key = data.reducer.contact_fingerprints[winner]
                        wp.static(writer_func)(contact, writer_data, -1)
            wp.tile_scatter_masked(scores, lane, wp.uint64(0), True)

    return export_heightfield_pair_csr


def _clone(original, name, replacements, extra, *, kernel=False):
    """Change only checked producer ownership seams in existing source."""
    function = original.func
    source = textwrap.dedent(inspect.getsource(function))
    source = source[source.index("def ") :]
    # Factory keys can rename __name__ without changing the source definition.
    source_name = source[4 : source.index("(")]
    if not source_name.isidentifier():
        raise RuntimeError(f"Pair-CSR definition seam changed: {name}")
    source = source.replace("def " + source_name + "(", "def " + name + "(", 1)
    for before, after in replacements:
        if source.count(before) != 1:
            raise RuntimeError(f"Pair-CSR source seam changed: {name}: {before!r}")
        source = source.replace(before, after, 1)
    namespace = dict(function.__globals__)
    namespace.update(inspect.getclosurevars(function).nonlocals)
    namespace.update(extra)
    filename = f"<heightfield_pair_csr_{name}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), namespace)
    if kernel:
        return wp.kernel(namespace[name], enable_backward=False, module="unique")
    return wp.func(namespace[name])


@functools.cache
def get_query_kernels(shell_support: bool = False):
    """Tag the original packed triangle stream and both disjoint query routes."""
    from .heightfield_cells import _heightfield_cell_midphase  # noqa: PLC0415
    from .heightfield_finite import create_query_kernel, marked_fallback  # noqa: PLC0415
    from .heightfield_geometric import _geometric_midphase, heightfield_geometric_overlaps_kernel  # noqa: PLC0415

    geometric = _clone(
        _geometric_midphase,
        "pair_csr_geometric_midphase",
        [
            (
                "    count: wp.array[int],",
                "    count: wp.array[int],\n    pair_tags: wp.array[int],\n    pair_index: int,",
            ),
            (
                "triples[out] = wp.vec3i(a, b, tri)",
                "triples[out] = wp.vec3i(a, b, tri)\n                    pair_tags[out] = pair_index",
            ),
        ],
        {},
    )
    cell = _clone(
        _heightfield_cell_midphase,
        "pair_csr_cell_midphase",
        [
            (
                "    triangle_pairs_count: wp.array[int],",
                "    triangle_pairs_count: wp.array[int],\n    pair_tags: wp.array[int],\n    pair_index: int,",
            ),
            (
                "triangle_pairs[out_idx] = wp.vec3i(hfield_shape, other_shape, tri_idx)",
                "triangle_pairs[out_idx] = wp.vec3i(hfield_shape, other_shape, tri_idx)\n                    pair_tags[out_idx] = pair_index",
            ),
        ],
        {},
    )
    midphase = _clone(
        heightfield_geometric_overlaps_kernel,
        "heightfield_pair_csr_overlaps",
        [
            ("    count: wp.array[int],", "    count: wp.array[int],\n    pair_tags: wp.array[int],"),
            (
                "range(tid, pair_count[0], total_num_threads)",
                "range(tid, wp.min(pair_count[0], pairs.shape[0]), total_num_threads)",
            ),
            (
                "_geometric_midphase(a, b, hfd, projection, transforms, lower, upper, data, gaps, elevations, triples, count)",
                "_geometric_midphase(a, b, hfd, projection, transforms, lower, upper, data, gaps, elevations, triples, count, pair_tags, i)",
            ),
            (
                "_heightfield_cell_midphase(a, b, hfd, transforms, lower, upper, data, gaps, elevations, triples, count)",
                "_heightfield_cell_midphase(a, b, hfd, transforms, lower, upper, data, gaps, elevations, triples, count, pair_tags, i)",
            ),
        ],
        {"_geometric_midphase": geometric, "_heightfield_cell_midphase": cell},
        kernel=True,
    )
    finite = _clone(
        create_query_kernel(_write_pair, shell_support),
        "heightfield_pair_csr_shell_contacts" if shell_support else "heightfield_pair_csr_finite_contacts",
        [
            ("    writer_data: Any,", "    writer_data: PairCSRData,"),
            (
                "        triple = triangle_pairs[i]",
                "        local_writer = writer_data\n        local_writer.pair_index = writer_data.triangle_pair[i]\n        triple = triangle_pairs[i]",
            ),
            ("wp.static(writer_func)(contact, writer_data, -1)", "wp.static(writer_func)(contact, local_writer, -1)"),
        ],
        {"PairCSRData": PairCSRData},
        kernel=True,
    )
    generic = _clone(
        marked_fallback(mesh_triangle_contacts_to_reducer_kernel),
        "heightfield_pair_csr_generic_contacts",
        [
            ("    reducer_data: GlobalContactReducerData,", "    reducer_data: PairCSRData,"),
            (
                "        triple = triangle_pairs[i]",
                "        local_writer = reducer_data\n        local_writer.pair_index = reducer_data.triangle_pair[i]\n        triple = triangle_pairs[i]",
            ),
            ("                write_contact_to_reducer,", "                _write_pair,"),
            ("            reducer_data,", "            local_writer,"),
        ],
        {"PairCSRData": PairCSRData, "_write_pair": _write_pair},
        kernel=True,
    )
    return midphase, finite, generic


class PairCSR:
    """Own membership and arithmetic intervals for the calibrated raw pool."""

    def __init__(self, reducer, pair_capacity, triangle_capacity, device, *, shell_support=False):
        self.data = data = PairCSRData()
        data.reducer = reducer.get_data_struct()
        data.triangle_pair = wp.empty(triangle_capacity, dtype=int, device=device)
        data.raw_pair = wp.empty(reducer.capacity + 1, dtype=int, device=device)
        data.ids = wp.empty(reducer.capacity, dtype=int, device=device)
        data.separation = wp.empty(reducer.capacity + 1, dtype=wp.vec2, device=device)
        data.counts = wp.zeros(pair_capacity + 1, dtype=int, device=device)
        data.offsets = wp.empty(pair_capacity + 1, dtype=int, device=device)
        data.cursors = wp.zeros(pair_capacity, dtype=int, device=device)
        data.status = wp.zeros(1, dtype=int, device=device)
        for name in ("triangle_pair", "raw_pair", "ids", "separation", "counts", "offsets", "cursors", "status"):
            setattr(self, name, getattr(data, name))
        self.device = device
        self.shell_support = shell_support
        self.midphase, self.finite, self.generic = get_query_kernels(shell_support)

    def build(self, pairs, pair_count, types, shape_data, gaps, total_threads):
        """Count, scan and scatter once after both original query launches."""
        self.data.counts.zero_()
        self.data.cursors.zero_()
        wp.launch(
            count_contacts,
            dim=total_threads,
            inputs=[self.data, pairs, pair_count, types, shape_data, gaps, total_threads],
            device=self.device,
            record_tape=False,
        )
        wp.utils.array_scan(self.data.counts, self.data.offsets, inclusive=False)
        wp.launch(
            scatter_contacts,
            dim=total_threads,
            inputs=[self.data, pair_count, total_threads],
            device=self.device,
            record_tape=False,
        )


def bind(narrow, pairs_np, shape_types, *, unique_generated=False):
    """Admit only internally generated or checked unique canonical pair lists."""
    if not narrow._heightfield_pair_csr_requested:
        return
    if not unique_generated:
        if pairs_np is None:
            return
        pairs = np.asarray(pairs_np).reshape(-1, 2)
        if np.any(pairs < 0) or np.any(pairs >= len(shape_types)):
            return
        canonical = np.sort(pairs, axis=1)
        if np.any(canonical[:, 0] == canonical[:, 1]) or len(np.unique(canonical, axis=0)) != len(canonical):
            return
    owner = PairCSR(
        narrow.global_contact_reducer,
        narrow.max_candidate_pairs,
        narrow.max_triangle_pairs,
        narrow.device,
        shell_support=narrow._heightfield_pair_csr_shell_requested,
    )
    owner.export_kernel = create_export_kernel(narrow._convex_writer_func)
    owner.fallback_export = create_export_reduced_contacts_kernel(narrow._convex_writer_func)
    narrow._pair_csr = owner
    narrow._heightfield_pair_csr = True
    narrow._heightfield_pair_csr_shell = owner.shell_support
    narrow.export_reduced_contacts_kernel = owner.export_kernel
