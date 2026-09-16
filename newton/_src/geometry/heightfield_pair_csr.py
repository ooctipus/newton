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
from .heightfield_manifold import _selection_score, _stock_writer_accepts
from .types import GeoType


@wp.struct
class PairCSRData:
    reducer: GlobalContactReducerData
    triangle_pair: wp.array[int]
    raw_pair: wp.array[int]
    counts: wp.array[int]
    offsets: wp.array[int]
    cursors: wp.array[int]
    ids: wp.array[int]
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
        elif _stock_writer_accepts(
            contact_id,
            data.reducer.position_depth,
            data.reducer.normal,
            data.reducer.shape_pairs,
            types,
            shape_data,
            gaps,
        ):
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
    source = source.replace("def " + function.__name__ + "(", "def " + name + "(", 1)
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
def get_query_kernels():
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
        create_query_kernel(_write_pair),
        "heightfield_pair_csr_finite_contacts",
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
    """Own integer membership while retaining the calibrated raw geometry pool."""

    def __init__(self, reducer, pair_capacity, triangle_capacity, device):
        self.data = data = PairCSRData()
        data.reducer = reducer.get_data_struct()
        data.triangle_pair = wp.empty(triangle_capacity, dtype=int, device=device)
        data.raw_pair = wp.empty(reducer.capacity + 1, dtype=int, device=device)
        data.ids = wp.empty(reducer.capacity, dtype=int, device=device)
        data.counts = wp.zeros(pair_capacity + 1, dtype=int, device=device)
        data.offsets = wp.empty(pair_capacity + 1, dtype=int, device=device)
        data.cursors = wp.zeros(pair_capacity, dtype=int, device=device)
        data.status = wp.zeros(1, dtype=int, device=device)
        for name in ("triangle_pair", "raw_pair", "ids", "counts", "offsets", "cursors", "status"):
            setattr(self, name, getattr(data, name))
        self.device = device
        self.midphase, self.finite, self.generic = get_query_kernels()

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
    owner = PairCSR(narrow.global_contact_reducer, narrow.max_candidate_pairs, narrow.max_triangle_pairs, narrow.device)
    owner.export_kernel = create_export_kernel(narrow._convex_writer_func)
    owner.fallback_export = create_export_reduced_contacts_kernel(narrow._convex_writer_func)
    narrow._pair_csr = owner
    narrow._heightfield_pair_csr = True
    narrow.export_reduced_contacts_kernel = owner.export_kernel
