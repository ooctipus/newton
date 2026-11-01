# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""NxN (all-pairs) broad phase collision detection.

Provides O(N^2) broad phase using AABB overlap tests. Simple and effective
for small scenes (<100 shapes). For larger scenes, use SAP broad phase.

See Also:
    :class:`BroadPhaseSAP` in ``broad_phase_sap.py`` for O(N log N) performance.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import warp as wp

from ..core.types import Devicelike
from .broad_phase_common import (
    BODY_FLAG_KINEMATIC,
    FORECAST_HYDROELASTIC,
    FORECAST_REDUCED,
    FORECAST_SLOTS_SHIFT,
    FORECAST_TERRAIN,
    ShapeForecast,
    check_aabb_overlap_moving,
    forecast_add,
    forecasting,
    is_pair_excluded,
    is_shape_pair_immovable_filtered,
    precompute_world_map,
    test_world_and_group_pair,
    write_pair,
)

_NO_FORECAST = ShapeForecast()
"""A forecast with no arrays: the pass forecasts nothing."""


@wp.kernel(enable_backward=False)
def _nxn_broadphase_precomputed_pairs(
    # Input arrays
    shape_bounding_box_lower: wp.array[wp.vec3],
    shape_bounding_box_upper: wp.array[wp.vec3],
    shape_gap: wp.array[float],  # Optional per-shape effective gaps (can be empty if AABBs pre-expanded)
    shape_displacement: wp.array[wp.vec3],  # Optional displacement over the collision-update interval [m]
    nxn_shape_pair: wp.array[wp.vec2i],
    shape_body: wp.array[int],
    body_flags: wp.array[int],
    include_static_kinematic_pairs: bool,
    # Output arrays
    candidate_pair: wp.array[wp.vec2i],
    candidate_pair_count: wp.array[int],  # Size one array
    max_candidate_pair: int,
):
    elementid = wp.tid()

    pair = nxn_shape_pair[elementid]
    shape1 = pair[0]
    shape2 = pair[1]

    # Check if gaps are provided (empty array means AABBs are pre-expanded)
    gap1 = 0.0
    gap2 = 0.0
    if shape_gap.shape[0] > 0:
        gap1 = shape_gap[shape1]
        gap2 = shape_gap[shape2]

    if check_aabb_overlap_moving(
        shape1, shape2, shape_bounding_box_lower, shape_bounding_box_upper, gap1, gap2, shape_displacement
    ):
        # Only overlapping pairs gather their bodies for the immovable test; most listed pairs are apart.
        if is_shape_pair_immovable_filtered(shape1, shape2, shape_body, body_flags, include_static_kinematic_pairs):
            return
        write_pair(
            pair,
            candidate_pair,
            candidate_pair_count,
            max_candidate_pair,
        )


@wp.kernel(enable_backward=False)
def _nxn_broadphase_precomputed_pairs_forecast(
    # Input arrays
    shape_bounding_box_lower: wp.array[wp.vec3],
    shape_bounding_box_upper: wp.array[wp.vec3],
    shape_gap: wp.array[float],  # Optional per-shape effective gaps (can be empty if AABBs pre-expanded)
    shape_displacement: wp.array[wp.vec3],  # Optional displacement over the collision-update interval [m]
    nxn_shape_pair: wp.array[wp.vec2i],
    shape_body: wp.array[int],
    body_flags: wp.array[int],
    include_static_kinematic_pairs: bool,
    # Output arrays
    candidate_pair: wp.array[wp.vec2i],
    candidate_pair_count: wp.array[int],  # Size one array
    max_candidate_pair: int,
    forecast: ShapeForecast,
):
    """:func:`_nxn_broadphase_precomputed_pairs` on a forecasting pass, which adds each pair's forecast."""
    elementid = wp.tid()

    pair = nxn_shape_pair[elementid]
    shape1 = pair[0]
    shape2 = pair[1]
    forecast_add(
        forecast,
        elementid,
        shape1,
        shape2,
        shape_bounding_box_lower,
        shape_bounding_box_upper,
        shape_body,
        body_flags,
        include_static_kinematic_pairs,
    )

    # Check if gaps are provided (empty array means AABBs are pre-expanded)
    gap1 = 0.0
    gap2 = 0.0
    if shape_gap.shape[0] > 0:
        gap1 = shape_gap[shape1]
        gap2 = shape_gap[shape2]

    if check_aabb_overlap_moving(
        shape1, shape2, shape_bounding_box_lower, shape_bounding_box_upper, gap1, gap2, shape_displacement
    ):
        # Only overlapping pairs gather their bodies for the immovable test; most listed pairs are apart.
        if is_shape_pair_immovable_filtered(shape1, shape2, shape_body, body_flags, include_static_kinematic_pairs):
            return
        write_pair(
            pair,
            candidate_pair,
            candidate_pair_count,
            max_candidate_pair,
        )


@wp.struct
class _PairTemplateData:
    # Global-global pairs: they lead the explicit list and are read from it.
    global_count: int
    # Leading blocks that test the global-global pairs, ``global_chunk`` consecutive pairs each.
    global_blocks: int
    global_chunk: int
    # First shape of each world with template pairs (a "slot"), in explicit-list order.
    slot_shape_start: wp.array[int]
    # Template index of each slot.
    slot_template: wp.array[int]
    # Start of each template's staged shapes in ``stage``, shape [template_count + 1].
    template_stage_start: wp.array[int]
    # Start of each template's entries in ``entries``, shape [template_count + 1].
    template_entry_start: wp.array[int]
    # Staged shapes per template: a world-local id, or ``-(global id + 1)``.
    stage: wp.array[int]
    # Pairs of staged-shape indices, packed ``a | (b << 16)``, in explicit-list order.
    entries: wp.array[int]


def _immovable(shape: str) -> str:
    """C++ for ``is_shape_pair_immovable_filtered``'s per-shape test (static, or kinematic when flags exist)."""
    return (
        f"(shape_body.data[{shape}] < 0 || (body_flags.shape[0] > 0 && "
        f"(body_flags.data[shape_body.data[{shape}]] & {int(BODY_FLAG_KINEMATIC)}) != 0))"
    )


# The global-global pairs, tested from the explicit list itself: the leading
# global_blocks blocks take global_chunk consecutive pairs each, so a long
# global list spreads over the GPU as in the flat kernel. Blocks run in
# order on CPU, which keeps the list order there.
_TEMPLATE_GLOBAL = f"""
    if (block < data.global_blocks) {{
        const int begin = block * data.global_chunk;
        const int remaining = data.global_count - begin;
        const int end = begin + (remaining < data.global_chunk ? remaining : data.global_chunk);
        for (int i0 = begin; i0 < end; i0 += STEP) {{
            const int i = i0 + LANE;
            bool hit = false;
            int a = 0, b = 0;
            if (i < end) {{
                const wp::vec2i pair = pairs.data[i];
                a = pair[0];
                b = pair[1];
                const wp::vec3 la = lower.data[a];
                const wp::vec3 ua = upper.data[a];
                const wp::vec3 lb = lower.data[b];
                const wp::vec3 ub = upper.data[b];
                hit = la[0] <= ub[0] && ua[0] >= lb[0] && la[1] <= ub[1] && ua[1] >= lb[1] && la[2] <= ub[2]
                    && ua[2] >= lb[2];
                if (hit && immovable_filter) hit = !({_immovable("a")} && {_immovable("b")});FORECAST_GLOBAL
            }}
APPEND
        }}FLUSH
        return;
    }}
    const int world_block = block - data.global_blocks;
"""

# One world group: its slot, template and the base of its shared segment.
_TEMPLATE_WORLD = """
        const int slot = world_block * WORLDS + group;
        int start = 0, stage_begin = 0, stage_count = 0, entry_begin = 0, entry_count = 0;
        if (slot < data.slot_shape_start.shape[0]) {
            const int t = data.slot_template.data[slot];
            start = data.slot_shape_start.data[slot];
            stage_begin = data.template_stage_start.data[t];
            stage_count = data.template_stage_start.data[t + 1] - stage_begin;
            entry_begin = data.template_entry_start.data[t];
            entry_count = data.template_entry_start.data[t + 1] - entry_begin;
        }
        const int base = group * SLOTS;
"""

# Stage the world's referenced AABBs, shape ids and immovable flags once.
_TEMPLATE_STAGE = f"""
        for (int i = LANE; i < stage_count; i += STEP) {{
            const int code = data.stage.data[stage_begin + i];
            const int shape = code >= 0 ? start + code : -code - 1;
            const wp::vec3 lo = lower.data[shape];
            const wp::vec3 hi = upper.data[shape];
            s_lx[base + i] = lo[0];
            s_ly[base + i] = lo[1];
            s_lz[base + i] = lo[2];
            s_ux[base + i] = hi[0];
            s_uy[base + i] = hi[1];
            s_uz[base + i] = hi[2];
            s_shape[base + i] = shape;
            s_fixed[base + i] = immovable_filter && {_immovable("shape")};FORECAST_STAGE
        }}
"""

# Test the template's entries in explicit-list order. The overlap test is
# check_aabb_overlap's with zero gaps (x + 0 compares like x) and is symmetric
# in its two boxes; the pair is written as (min, max), as the list stores it.
_TEMPLATE_TEST = """
        for (int e0 = 0; e0 < entry_count; e0 += STEP) {
            const int e = e0 + LANE;
            bool hit = false;
            int a = 0, b = 0;
            if (e < entry_count) {
                const unsigned int packed = (unsigned int)data.entries.data[entry_begin + e];
                const int ia = base + (int)(packed & 0xffffu);
                const int ib = base + (int)(packed >> 16);
                hit = s_lx[ia] <= s_ux[ib] && s_ux[ia] >= s_lx[ib] && s_ly[ia] <= s_uy[ib] && s_uy[ia] >= s_ly[ib]
                    && s_lz[ia] <= s_uz[ib] && s_uz[ia] >= s_lz[ib];
                if (hit && immovable_filter) hit = !(s_fixed[ia] && s_fixed[ib]);FORECAST_TEST
                if (hit) {
                    a = s_shape[ia];
                    b = s_shape[ib];
                    if (a > b) {
                        const int swap = a;
                        a = b;
                        b = swap;
                    }
                }
            }
APPEND
        }
"""

# write_pair's contract with one atomic per warp: the count advances by every
# hit, and only indices below the capacity are written. Loops around it are
# warp-uniform (a world group is whole warps; a global block strides by the block).
_TEMPLATE_APPEND_CUDA = """
            {
                const unsigned int mask = __ballot_sync(0xffffffffu, hit);
                if (mask != 0u) {
                    const int warp_lane = thread & 31;
                    const int leader = __ffs(mask) - 1;
                    int first = 0;
                    if (warp_lane == leader) first = atomicAdd(candidate_pair_count.data, __popc(mask));
                    first = __shfl_sync(0xffffffffu, first, leader);
                    if (hit) {
                        const int index = first + __popc(mask & ((1u << warp_lane) - 1u));
                        if (index < max_candidate_pair) candidate_pair.data[index] = wp::vec2i(a, b);
                    }
                }
            }
"""

_TEMPLATE_APPEND_CPU = """
            if (hit) {
                const int index = candidate_pair_count.data[0]++;
                if (index < max_candidate_pair) candidate_pair.data[index] = wp::vec2i(a, b);
            }
"""


# The forecasting variant's additions to the template kernel (forecast=True), at the placeholders above, which run
# when the pool's flag says the pass forecasts; the plain kernel is the one without them. The forecast
# (broad_phase_common.forecast_pair) runs the same tests in the same order: the boxes, each shape's forecast word and
# reach from the world's stage (or memory, for a global pair), the capsules (placed by the shapes' poses) and the
# terrain's top, which only a pair within reach of its boxes reads, from memory.
_FORECAST_PAIR = f"""
    const bool fc_on = forecast.on.shape[0] > 0 && forecast.on.data[0] != 0;
    long long fc_total = 0;
    auto fc_pair = [&](int a, int b, int wa, int wb, float reach, float alx, float aly, float alz, float aux, float auy,
                       float auz, float blx, float bly, float blz, float bux, float buy, float buz) -> int {{
        if (((wa | wb) & {int(FORECAST_TERRAIN)}) != 0) {{
            if ((wb & {int(FORECAST_TERRAIN)}) != 0)
                return alz - forecast.top.data[a] > reach ? 0 : (wa >> {int(FORECAST_SLOTS_SHIFT)});
            return blz - forecast.top.data[b] > reach ? 0 : (wb >> {int(FORECAST_SLOTS_SHIFT)});
        }}
        const float gap = wp::max(
            wp::max(wp::max(alx - bux, blx - aux), wp::max(aly - buy, bly - auy)), wp::max(alz - buz, blz - auz));
        if (gap > reach) return 0;
        const wp::vec4 xa = forecast.capsule.data[a];
        const wp::vec4 xb = forecast.capsule.data[b];
        const float ra = xa[3];
        const float rb = xb[3];
        if (ra >= 0.0f && rb >= 0.0f) {{
            const wp::transform pa = forecast.pose.data[a];
            const wp::transform pb = forecast.pose.data[b];
            const wp::vec4 ca = forecast.middle.data[a];
            const wp::vec4 cb = forecast.middle.data[b];
            const wp::vec3 ma = wp::transform_point(pa, wp::vec3(ca[0], ca[1], ca[2]));
            const wp::vec3 ha = wp::transform_vector(pa, wp::vec3(xa[0], xa[1], xa[2]));
            const wp::vec3 mb = wp::transform_point(pb, wp::vec3(cb[0], cb[1], cb[2]));
            const wp::vec3 hb = wp::transform_vector(pb, wp::vec3(xb[0], xb[1], xb[2]));
            const wp::vec3 closest = wp::closest_point_edge_edge(
                wp::sub(ma, ha), wp::add(ma, ha), wp::sub(mb, hb), wp::add(mb, hb), 1.0e-9f);
            if (closest[2] - ra - rb > reach) return 0;
        }}
        if ((wa & wb & {int(FORECAST_HYDROELASTIC)}) != 0) return forecast.bound_hydroelastic;
        if (((wa | wb) & {int(FORECAST_REDUCED)}) != 0) return forecast.bound_reduced;
        return forecast.bound_convex;
    }};
"""

_FORECAST_GLOBAL = f"""
                if (fc_on) {{
                    int contacts = fc_pair(a, b, forecast.word.data[a], forecast.word.data[b],
                        forecast.reach.data[a] + forecast.reach.data[b], la[0], la[1], la[2], ua[0], ua[1], ua[2],
                        lb[0], lb[1], lb[2], ub[0], ub[1], ub[2]);
                    if (contacts > 0 && immovable_filter && {_immovable("a")} && {_immovable("b")}) contacts = 0;
                    fc_total += contacts;
                }}"""

_FORECAST_STAGE = """
            if (fc_on) {
                s_fw[base + i] = forecast.word.data[shape];
                s_fv[base + i] = forecast.reach.data[shape];
            }"""

_FORECAST_TEST = """
                if (fc_on) {
                    int contacts = fc_pair(s_shape[ia], s_shape[ib], s_fw[ia], s_fw[ib], s_fv[ia] + s_fv[ib], s_lx[ia],
                        s_ly[ia], s_lz[ia], s_ux[ia], s_uy[ia], s_uz[ia], s_lx[ib], s_ly[ib], s_lz[ib], s_ux[ib],
                        s_uy[ib], s_uz[ib]);
                    if (contacts > 0 && immovable_filter && s_fixed[ia] && s_fixed[ib]) contacts = 0;
                    fc_total += contacts;
                }"""

# A warp adds its lanes' forecast in one atomic, into the counter of its warp; on CPU the block adds its own.
_FORECAST_FLUSH_CUDA = """
        if (fc_on) {
            for (int offset = 16; offset > 0; offset >>= 1) fc_total += __shfl_down_sync(0xffffffffu, fc_total, offset);
            if ((thread & 31) == 0 && fc_total != 0) {
                const int counter = (block * 32 + (thread >> 5)) % forecast.slots.shape[0];
                atomicAdd((unsigned long long*)(forecast.slots.data + counter), (unsigned long long)fc_total);
            }
        }"""

_FORECAST_FLUSH_CPU = """
        if (fc_total != 0) forecast.slots.data[block % forecast.slots.shape[0]] += fc_total;"""


def _pair_template_source(slots: int, threads: int, worlds: int, forecast: bool = False) -> str:
    """CUDA and CPU source of the world-template kernel, with the contact forecast on a forecasting pass.

    CUDA runs ``worlds`` world groups of ``threads`` lanes per block; CPU runs
    the block's groups, and every loop in them, serially in explicit-list order.
    """

    def fill(text: str, lane: str, step: str, append: str, flush: str = "") -> str:
        pieces = {
            "FORECAST_GLOBAL": _FORECAST_GLOBAL if forecast else "",
            "FORECAST_STAGE": _FORECAST_STAGE if forecast else "",
            "FORECAST_TEST": _FORECAST_TEST if forecast else "",
            "FLUSH": flush if forecast else "",
        }
        for name, piece in pieces.items():
            text = text.replace(name, piece)
        return text.replace("LANE", lane).replace("STEP", step).replace("APPEND", append)

    def arrays(qualifier: str) -> str:
        capacity = slots * worlds
        names = ("s_lx", "s_ly", "s_lz", "s_ux", "s_uy", "s_uz") + (("s_fv",) if forecast else ())
        lines = [f"    {qualifier}float {name}[{capacity}];" for name in names]
        lines += [f"    {qualifier}int s_shape[{capacity}];", f"    {qualifier}int s_fixed[{capacity}];"]
        if forecast:
            lines += [f"    {qualifier}int s_fw[{capacity}];"]
        return "\n".join(lines) + "\n" + (_FORECAST_PAIR if forecast else "")

    world = _TEMPLATE_WORLD.replace("WORLDS", str(worlds)).replace("SLOTS", str(slots))
    return (
        "#if defined(__CUDA_ARCH__)\n"
        + arrays("__shared__ ")
        + "    const int thread = threadIdx.x;\n"
        + fill(_TEMPLATE_GLOBAL, "thread", "(int)blockDim.x", _TEMPLATE_APPEND_CUDA, _FORECAST_FLUSH_CUDA)
        + f"    const int group = thread / {threads};\n"
        + f"    const int lane = thread - group * {threads};\n"
        + "    {\n"
        + world
        + fill(_TEMPLATE_STAGE, "lane", str(threads), "")
        + "        __syncthreads();\n"
        + fill(_TEMPLATE_TEST, "lane", str(threads), _TEMPLATE_APPEND_CUDA)
        + (_FORECAST_FLUSH_CUDA + "\n" if forecast else "")
        + "    }\n"
        + "#else\n"
        + "    if (logical_lane != 0) return;\n"
        + arrays("")
        + fill(_TEMPLATE_GLOBAL, "0", "1", _TEMPLATE_APPEND_CPU, _FORECAST_FLUSH_CPU)
        + f"    for (int group = 0; group < {worlds}; ++group) {{\n"
        + world
        + fill(_TEMPLATE_STAGE, "0", "1", "")
        + fill(_TEMPLATE_TEST, "0", "1", _TEMPLATE_APPEND_CPU)
        + "    }\n"
        + (_FORECAST_FLUSH_CPU + "\n" if forecast else "")
        + "#endif\n"
    )


@functools.cache
def _pair_template_kernel(slots: int, threads: int, worlds: int, forecast: bool = False):
    """World-template broad-phase kernel; launch tiled with ``block_dim = threads * worlds``.

    Args:
        slots: Staged shapes per world group (shared-memory capacity).
        threads: Lanes per world group, a multiple of 32 so every warp serves one world.
        worlds: World groups per block.
        forecast: The forecasting pass's kernel, which adds the contact forecast and takes it as its last argument;
            every other pass runs the kernel without it.
    """
    source = _pair_template_source(slots, threads, worlds, forecast)
    if forecast:

        @wp.func_native(source)
        def test_templates(
            block: int,
            logical_lane: int,
            lower: wp.array[wp.vec3],
            upper: wp.array[wp.vec3],
            pairs: wp.array[wp.vec2i],
            data: _PairTemplateData,
            shape_body: wp.array[int],
            body_flags: wp.array[int],
            immovable_filter: bool,
            candidate_pair: wp.array[wp.vec2i],
            candidate_pair_count: wp.array[int],
            max_candidate_pair: int,
            forecast: ShapeForecast,
        ): ...

        def kernel(
            lower: wp.array[wp.vec3],
            upper: wp.array[wp.vec3],
            pairs: wp.array[wp.vec2i],
            data: _PairTemplateData,
            shape_body: wp.array[int],
            body_flags: wp.array[int],
            include_static_kinematic_pairs: bool,
            candidate_pair: wp.array[wp.vec2i],
            candidate_pair_count: wp.array[int],
            max_candidate_pair: int,
            forecast: ShapeForecast,
        ):
            block, lane = wp.tid()
            # is_shape_pair_immovable_filtered: no filtering without shape metadata.
            immovable_filter = (not include_static_kinematic_pairs) and shape_body.shape[0] > 0
            test_templates(
                block,
                lane,
                lower,
                upper,
                pairs,
                data,
                shape_body,
                body_flags,
                immovable_filter,
                candidate_pair,
                candidate_pair_count,
                max_candidate_pair,
                forecast,
            )

        kernel.__name__ = kernel.__qualname__ = f"broad_phase_pair_templates_forecast_{slots}_{threads}_{worlds}"
        return wp.kernel(enable_backward=False, module="unique", grid_stride=False)(kernel)

    @wp.func_native(source)
    def test_templates(
        block: int,
        logical_lane: int,
        lower: wp.array[wp.vec3],
        upper: wp.array[wp.vec3],
        pairs: wp.array[wp.vec2i],
        data: _PairTemplateData,
        shape_body: wp.array[int],
        body_flags: wp.array[int],
        immovable_filter: bool,
        candidate_pair: wp.array[wp.vec2i],
        candidate_pair_count: wp.array[int],
        max_candidate_pair: int,
    ): ...

    def kernel(
        lower: wp.array[wp.vec3],
        upper: wp.array[wp.vec3],
        pairs: wp.array[wp.vec2i],
        data: _PairTemplateData,
        shape_body: wp.array[int],
        body_flags: wp.array[int],
        include_static_kinematic_pairs: bool,
        candidate_pair: wp.array[wp.vec2i],
        candidate_pair_count: wp.array[int],
        max_candidate_pair: int,
    ):
        block, lane = wp.tid()
        # is_shape_pair_immovable_filtered: no filtering without shape metadata.
        immovable_filter = (not include_static_kinematic_pairs) and shape_body.shape[0] > 0
        test_templates(
            block,
            lane,
            lower,
            upper,
            pairs,
            data,
            shape_body,
            body_flags,
            immovable_filter,
            candidate_pair,
            candidate_pair_count,
            max_candidate_pair,
        )

    kernel.__name__ = kernel.__qualname__ = f"broad_phase_pair_templates_{slots}_{threads}_{worlds}"
    # The CUDA snippet stages one block's worlds in shared memory behind a single
    # __syncthreads(), so each thread must run exactly one logical block: no
    # grid-stride loop, and Warp refuses a launch that caps the grid.
    return wp.kernel(enable_backward=False, module="unique", grid_stride=False)(kernel)


@wp.kernel(enable_backward=False)
def _verify_pair_templates(
    data: _PairTemplateData,
    slot_pair_start: wp.array[int],
    pairs: wp.array[wp.vec2i],
    stride: int,
    mismatch: wp.array[int],
):
    """Count template entries whose decoded pair differs from the explicit list at its position."""
    slot, lane = wp.tid()
    template = data.slot_template[slot]
    start = data.slot_shape_start[slot]
    stage_begin = data.template_stage_start[template]
    entry_begin = data.template_entry_start[template]
    entry_count = data.template_entry_start[template + 1] - entry_begin
    offset = slot_pair_start[slot]
    entry = lane
    while entry < entry_count:
        packed = data.entries[entry_begin + entry]
        code_a = data.stage[stage_begin + (packed & 0xFFFF)]
        code_b = data.stage[stage_begin + (packed >> 16)]
        shape_a = start + code_a
        if code_a < 0:
            shape_a = -code_a - 1
        shape_b = start + code_b
        if code_b < 0:
            shape_b = -code_b - 1
        pair = pairs[offset + entry]
        if wp.min(shape_a, shape_b) != pair[0] or wp.max(shape_a, shape_b) != pair[1]:
            wp.atomic_add(mismatch, 0, 1)
        entry += stride


@wp.func
def _get_lower_triangular_indices(index: int, matrix_size: int) -> tuple[int, int]:
    total = (matrix_size * (matrix_size - 1)) >> 1
    if index >= total:
        # In Warp, we can't throw, so return an invalid pair
        return -1, -1

    low = int(0)
    high = matrix_size - 1
    while low < high:
        mid = (low + high) >> 1
        count = (mid * (2 * matrix_size - mid - 1)) >> 1
        if count <= index:
            low = mid + 1
        else:
            high = mid
    r = low - 1
    f = (r * (2 * matrix_size - r - 1)) >> 1
    c = (index - f) + r + 1
    return r, c


@wp.func
def _find_world_and_local_id(
    tid: int,
    world_cumsum_lower_tri: wp.array[int],
):
    """Binary search to find world ID and local ID from thread ID.

    Args:
        tid: Global thread ID
        world_cumsum_lower_tri: Cumulative sum of lower triangular elements per world

    Returns:
        tuple: (world_id, local_id) - World ID and local index within that world
    """
    world_count = world_cumsum_lower_tri.shape[0]

    # Find world_id using binary search
    # Declare as dynamic variables for loop mutation
    low = int(0)
    high = int(world_count - 1)
    world_id = int(0)

    while low <= high:
        mid = (low + high) >> 1
        if tid < world_cumsum_lower_tri[mid]:
            high = mid - 1
            world_id = mid
        else:
            low = mid + 1

    # Calculate local index within this world
    local_id = tid
    if world_id > 0:
        local_id = tid - world_cumsum_lower_tri[world_id - 1]

    return world_id, local_id


@wp.kernel(enable_backward=False)
def _nxn_broadphase_kernel(
    # Input arrays
    shape_bounding_box_lower: wp.array[wp.vec3],
    shape_bounding_box_upper: wp.array[wp.vec3],
    shape_gap: wp.array[float],  # Optional per-shape effective gaps (can be empty if AABBs pre-expanded)
    shape_displacement: wp.array[wp.vec3],  # Optional displacement over the collision-update interval [m]
    collision_group: wp.array[int],  # per-shape
    shape_world: wp.array[int],  # per-shape world indices
    shape_body: wp.array[int],
    body_flags: wp.array[int],
    include_static_kinematic_pairs: bool,
    world_cumsum_lower_tri: wp.array[int],  # Cumulative sum of lower tri elements per world
    world_slice_ends: wp.array[int],  # End indices of each world slice
    world_index_map: wp.array[int],  # Index map into source geometry
    num_regular_worlds: int,  # Number of regular world segments (excluding dedicated -1 segment)
    filter_pairs: wp.array[wp.vec2i],  # Sorted excluded pairs (empty if none)
    num_filter_pairs: int,
    # Output arrays
    candidate_pair: wp.array[wp.vec2i],
    candidate_pair_count: wp.array[int],  # Size one array
    max_candidate_pair: int,
    forecast: ShapeForecast,
):
    tid = wp.tid()

    # Find which world this thread belongs to and the local index within that world
    world_id, local_id = _find_world_and_local_id(tid, world_cumsum_lower_tri)

    # Get the slice boundaries for this world in the index map
    world_slice_start = 0
    if world_id > 0:
        world_slice_start = world_slice_ends[world_id - 1]
    world_slice_end = world_slice_ends[world_id]

    # Number of geometries in this world
    num_shapes_in_world = world_slice_end - world_slice_start

    # Convert local_id to pair indices within the world
    local_shape1, local_shape2 = _get_lower_triangular_indices(local_id, num_shapes_in_world)

    # Map to actual geometry indices using the world_index_map
    shape1_tmp = world_index_map[world_slice_start + local_shape1]
    shape2_tmp = world_index_map[world_slice_start + local_shape2]

    # Ensure canonical ordering (smaller index first)
    # After mapping, the indices might not preserve local_shape1 < local_shape2 ordering
    shape1 = wp.min(shape1_tmp, shape2_tmp)
    shape2 = wp.max(shape1_tmp, shape2_tmp)

    # Get world and collision groups
    world1 = shape_world[shape1]
    world2 = shape_world[shape2]
    collision_group1 = collision_group[shape1]
    collision_group2 = collision_group[shape2]

    # Skip pairs where both geometries are global (world -1), unless we're in the dedicated -1 segment
    # The dedicated -1 segment is the last segment (world_id >= num_regular_worlds)
    is_dedicated_minus_one_segment = world_id >= num_regular_worlds
    if world1 == -1 and world2 == -1 and not is_dedicated_minus_one_segment:
        return

    # Check both world and collision groups
    if not test_world_and_group_pair(world1, world2, collision_group1, collision_group2):
        return

    if forecasting(forecast) and (
        num_filter_pairs == 0 or not is_pair_excluded(wp.vec2i(shape1, shape2), filter_pairs, num_filter_pairs)
    ):
        forecast_add(
            forecast,
            tid,
            shape1,
            shape2,
            shape_bounding_box_lower,
            shape_bounding_box_upper,
            shape_body,
            body_flags,
            include_static_kinematic_pairs,
        )

    # Check if gaps are provided (empty array means AABBs are pre-expanded)
    gap1 = 0.0
    gap2 = 0.0
    if shape_gap.shape[0] > 0:
        gap1 = shape_gap[shape1]
        gap2 = shape_gap[shape2]

    if check_aabb_overlap_moving(
        shape1, shape2, shape_bounding_box_lower, shape_bounding_box_upper, gap1, gap2, shape_displacement
    ):
        # Only overlapping pairs gather their bodies for the immovable test.
        if is_shape_pair_immovable_filtered(shape1, shape2, shape_body, body_flags, include_static_kinematic_pairs):
            return
        # Skip explicitly excluded pairs (e.g. shape_collision_filter_pairs)
        if num_filter_pairs > 0 and is_pair_excluded(wp.vec2i(shape1, shape2), filter_pairs, num_filter_pairs):
            return
        write_pair(
            wp.vec2i(shape1, shape2),
            candidate_pair,
            candidate_pair_count,
            max_candidate_pair,
        )


class BroadPhaseAllPairs:
    """A broad phase collision detection class that performs N x N collision checks between all geometry pairs.

    This class performs collision detection between all possible pairs of geometries by checking for
    axis-aligned bounding box (AABB) overlaps. It uses a lower triangular matrix approach to avoid
    checking each pair twice.

    The collision checks take into account per-geometry cutoff distances and collision groups. Two geometries
    will only be considered as a candidate pair if:
    1. Their AABBs overlap when expanded by their cutoff distances
    2. Their collision groups allow interaction (determined by test_group_pair())

    The class outputs an array of candidate collision pairs that need more detailed narrow phase collision
    checking.
    """

    def __init__(
        self,
        shape_world: wp.array[wp.int32] | np.ndarray,
        shape_flags: wp.array[wp.int32] | np.ndarray | None = None,
        device: Devicelike | None = None,
    ) -> None:
        """Initialize the broad phase with world ID information.

        Args:
            shape_world: Array of world IDs (numpy or warp array).
                Positive/zero values represent distinct worlds, negative values represent
                shared entities that belong to all worlds.
            shape_flags: Optional array of shape flags (numpy or warp array). If provided,
                only shapes with the COLLIDE_SHAPES flag will be included in collision checks.
                This efficiently filters out visual-only shapes.
            device: Device to store the precomputed arrays on. If None, uses CPU for numpy
                arrays or the device of the input warp array.
        """
        # Convert to numpy if it's a warp array
        if isinstance(shape_world, wp.array):
            shape_world_np = shape_world.numpy()
            if device is None:
                device = shape_world.device
        else:
            shape_world_np = shape_world
            if device is None:
                device = "cpu"

        # Convert shape_flags to numpy if provided
        shape_flags_np = None
        if shape_flags is not None:
            if isinstance(shape_flags, wp.array):
                shape_flags_np = shape_flags.numpy()
            else:
                shape_flags_np = shape_flags

        # Precompute the world map (filters out non-colliding shapes if flags provided)
        index_map_np, slice_ends_np = precompute_world_map(shape_world_np, shape_flags_np)

        # Calculate number of regular worlds (excluding dedicated -1 segment at end)
        # Must be derived from filtered slices since precompute_world_map applies flags
        # slice_ends_np has length (num_filtered_worlds + 1), where +1 is the dedicated -1 segment
        num_regular_worlds = max(0, len(slice_ends_np) - 1)

        # Calculate cumulative sum of lower triangular elements per world
        # For each world, compute n*(n-1)/2 where n is the number of geometries in that world
        world_count = len(slice_ends_np)
        world_cumsum_lower_tri_np = np.zeros(world_count, dtype=np.int32)

        start_idx = 0
        cumsum = 0
        for world_idx in range(world_count):
            end_idx = slice_ends_np[world_idx]
            # Number of geometries in this world (including shared geometries)
            num_geoms_in_world = end_idx - start_idx
            # Number of lower triangular elements for this world
            num_lower_tri = (num_geoms_in_world * (num_geoms_in_world - 1)) // 2
            cumsum += num_lower_tri
            world_cumsum_lower_tri_np[world_idx] = cumsum
            start_idx = end_idx

        # Store as warp arrays
        self.world_index_map = wp.array(index_map_np, dtype=wp.int32, device=device)
        self.world_slice_ends = wp.array(slice_ends_np, dtype=wp.int32, device=device)
        self.world_cumsum_lower_tri = wp.array(world_cumsum_lower_tri_np, dtype=wp.int32, device=device)

        # Store total number of kernel threads needed (last element of cumsum)
        self.num_kernel_threads = int(world_cumsum_lower_tri_np[-1]) if world_count > 0 else 0

        # Store number of regular worlds (for distinguishing dedicated -1 segment)
        self.num_regular_worlds = int(num_regular_worlds)

    def launch(
        self,
        shape_lower: wp.array[wp.vec3],  # Lower bounds of shape bounding boxes
        shape_upper: wp.array[wp.vec3],  # Upper bounds of shape bounding boxes
        shape_gap: wp.array[float] | None,  # Optional per-shape effective gaps
        shape_collision_group: wp.array[int],  # Collision group ID per box
        shape_world: wp.array[int],  # World index per box
        shape_count: int,  # Number of active bounding boxes
        # Outputs
        candidate_pair: wp.array[wp.vec2i],  # Array to store overlapping shape pairs
        candidate_pair_count: wp.array[int],
        device: Devicelike | None = None,  # Device to launch on
        filter_pairs: wp.array[wp.vec2i] | None = None,  # Sorted excluded pairs
        num_filter_pairs: int | None = None,
        skip_count_zero: bool = False,  # Skip candidate_pair_count.zero_() if already zeroed by the caller
        *,
        shape_body: wp.array[int] | None = None,
        body_flags: wp.array[int] | None = None,
        include_static_kinematic_pairs: bool = True,
        shape_displacement: wp.array[wp.vec3] | None = None,
        forecast: ShapeForecast | None = None,
    ) -> None:
        """Launch the N x N broad phase collision detection.

        This method performs collision detection between all possible pairs of geometries by checking for
        axis-aligned bounding box (AABB) overlaps. It uses a lower triangular matrix approach to avoid
        checking each pair twice.

        Args:
            shape_lower: Array of lower bounds for each shape's AABB
            shape_upper: Array of upper bounds for each shape's AABB
            shape_gap: Optional array of per-shape effective gaps. If None or empty array,
                assumes AABBs are pre-expanded (gaps = 0). If provided, gaps are added during overlap checks.
            shape_collision_group: Array of collision group IDs for each shape. Positive values indicate
                groups that only collide with themselves (and with negative groups). Negative values indicate
                groups that collide with everything except their negative counterpart. Zero indicates no collisions.
            shape_world: Array of world indices for each shape. Index -1 indicates global entities
                that collide with all worlds. Indices 0, 1, 2, ... indicate world-specific entities.
            shape_count: Number of active bounding boxes to check
            candidate_pair: Output array to store overlapping shape pairs
            candidate_pair_count: Output array to store number of overlapping pairs found
            device: Device to launch on. If None, uses the device of the input arrays.
            filter_pairs: Optional sorted shape pairs to exclude.
            num_filter_pairs: Number of valid entries in ``filter_pairs``. If None, uses ``filter_pairs.shape[0]``.
            skip_count_zero: If True, skip the internal ``candidate_pair_count.zero_()``.
                The caller guarantees ``candidate_pair_count[0] == 0`` on entry (e.g. when
                the counter was zeroed by a preceding fused kernel).  Defaults to False so
                the launch remains self-contained.
            shape_body: Optional array mapping each shape to its body index. Negative body indices are static shapes.
                Omitting this array disables immovable-pair filtering for expert callers.
            body_flags: Optional body flag array used to identify kinematic bodies. An empty array is valid for
                an all-static model when ``shape_body`` is provided.
            include_static_kinematic_pairs: Whether to include pairs where both shapes are immovable. Set to
                ``False`` to filter static-static, static-kinematic, and kinematic-kinematic pairs.
            shape_displacement: Optional world-space displacement of each shape over the collision-update interval
                ``dt``,
                used for speculative-contact swept-AABB tests [m]. See
                :ref:`Speculative contacts <speculative-contacts>`. :class:`CollisionPipeline` computes it as the
                shape-origin velocity, including the angular contribution from its COM offset, times ``dt``; angular
                travel expands the supplied AABB separately.
            forecast: The contact forecast a collision pass adds the tested pairs into
                (:class:`~newton._src.geometry.broad_phase_common.ShapeForecast`); ``None`` forecasts nothing.

        The method will populate candidate_pair with the indices of shape pairs (i,j) where i < j whose AABBs overlap
        (with optional margin expansion), whose collision groups allow interaction, and whose world indices are
        compatible (same world or at least one is global). Pairs in filter_pairs (if provided) are excluded.
        The number of pairs found will be written to candidate_pair_count[0].
        """
        max_candidate_pair = candidate_pair.shape[0]

        if not skip_count_zero:
            candidate_pair_count.zero_()

        if device is None:
            device = shape_lower.device

        # If no gaps provided, pass empty array (kernel will use 0.0 gaps)
        if shape_gap is None:
            shape_gap = wp.empty(0, dtype=wp.float32, device=device)
        if shape_body is None:
            shape_body = wp.empty(0, dtype=wp.int32, device=device)
        if body_flags is None:
            body_flags = wp.empty(0, dtype=wp.int32, device=device)
        if shape_displacement is not None and shape_displacement.shape[0] != shape_lower.shape[0]:
            raise ValueError(
                "shape_displacement length must match the shape bounds "
                f"({shape_lower.shape[0]}), got {shape_displacement.shape[0]}"
            )

        # Exclusion filter: empty array and 0 when not provided or empty
        if filter_pairs is None or filter_pairs.shape[0] == 0:
            filter_pairs_arr = wp.empty(0, dtype=wp.vec2i, device=device)
            n_filter = 0
        else:
            filter_pairs_arr = filter_pairs
            n_filter = num_filter_pairs if num_filter_pairs is not None else filter_pairs.shape[0]

        # Launch with the precomputed number of kernel threads
        wp.launch(
            _nxn_broadphase_kernel,
            dim=self.num_kernel_threads,
            inputs=[
                shape_lower,
                shape_upper,
                shape_gap,
                shape_displacement,
                shape_collision_group,
                shape_world,
                shape_body,
                body_flags,
                include_static_kinematic_pairs,
                self.world_cumsum_lower_tri,
                self.world_slice_ends,
                self.world_index_map,
                self.num_regular_worlds,
                filter_pairs_arr,
                n_filter,
            ],
            outputs=[
                candidate_pair,
                candidate_pair_count,
                max_candidate_pair,
                _NO_FORECAST if forecast is None else forecast,
            ],
            device=device,
            record_tape=False,
        )


class BroadPhaseExplicit:
    """A broad phase collision detection class that only checks explicitly provided geometry pairs.

    This class performs collision detection only between geometry pairs that are explicitly specified,
    rather than checking all possible pairs. This can be more efficient when the set of potential
    collision pairs is known ahead of time.

    The class checks for axis-aligned bounding box (AABB) overlaps between the specified geometry pairs,
    taking into account per-geometry cutoff distances.
    """

    def launch(
        self,
        shape_lower: wp.array[wp.vec3],  # Lower bounds of shape bounding boxes
        shape_upper: wp.array[wp.vec3],  # Upper bounds of shape bounding boxes
        shape_gap: wp.array[float] | None,  # Optional per-shape effective gaps
        shape_pairs: wp.array[wp.vec2i],  # Precomputed pairs to check
        shape_pair_count: int,
        # Outputs
        candidate_pair: wp.array[wp.vec2i],  # Array to store overlapping shape pairs
        candidate_pair_count: wp.array[int],
        device: Devicelike | None = None,  # Device to launch on
        skip_count_zero: bool = False,  # Skip candidate_pair_count.zero_() if already zeroed
        *,
        shape_body: wp.array[int] | None = None,
        body_flags: wp.array[int] | None = None,
        include_static_kinematic_pairs: bool = True,
        shape_displacement: wp.array[wp.vec3] | None = None,
        forecast: ShapeForecast | None = None,
    ) -> None:
        """Launch the explicit pairs broad phase collision detection.

        This method checks for AABB overlaps only between explicitly specified shape pairs,
        rather than checking all possible pairs. It populates the candidate_pair array with
        indices of shape pairs whose AABBs overlap.

        Args:
            shape_lower: Array of lower bounds for shape bounding boxes
            shape_upper: Array of upper bounds for shape bounding boxes
            shape_gap: Optional array of per-shape effective gaps. If None or empty array,
                assumes AABBs are pre-expanded (gaps = 0). If provided, gaps are added during overlap checks.
            shape_pairs: Array of precomputed shape pairs to check
            shape_pair_count: Number of shape pairs to check
            candidate_pair: Output array to store overlapping shape pairs
            candidate_pair_count: Output array to store number of overlapping pairs found
            device: Device to launch on. If None, uses the device of the input arrays.
            skip_count_zero: If True, skip the internal ``candidate_pair_count.zero_()``.
                The caller guarantees ``candidate_pair_count[0] == 0`` on entry (e.g. when
                the counter was zeroed by a preceding fused kernel).  Defaults to False so
                the launch remains self-contained.
            shape_body: Optional array mapping each shape to its body index. Negative body indices are static shapes.
                Omitting this array disables immovable-pair filtering for expert callers.
            body_flags: Optional body flag array used to identify kinematic bodies. An empty array is valid for
                an all-static model when ``shape_body`` is provided.
            include_static_kinematic_pairs: Whether to include pairs where both shapes are immovable. Set to
                ``False`` to filter static-static, static-kinematic, and kinematic-kinematic pairs.
            shape_displacement: Optional world-space displacement of each shape over the collision-update interval
                ``dt``,
                used for speculative-contact swept-AABB tests [m]. See
                :ref:`Speculative contacts <speculative-contacts>`. :class:`CollisionPipeline` computes it as the
                shape-origin velocity, including the angular contribution from its COM offset, times ``dt``; angular
                travel expands the supplied AABB separately.
            forecast: The contact forecast a collision pass adds the tested pairs into
                (:class:`~newton._src.geometry.broad_phase_common.ShapeForecast`); ``None`` forecasts nothing.

        The method will populate candidate_pair with the indices of shape pairs whose AABBs overlap
        (with optional margin expansion), but only checking the explicitly provided pairs.
        """

        max_candidate_pair = candidate_pair.shape[0]

        if not skip_count_zero:
            candidate_pair_count.zero_()

        if device is None:
            device = shape_lower.device

        # If no gaps provided, pass empty array (kernel will use 0.0 gaps)
        if shape_gap is None:
            shape_gap = wp.empty(0, dtype=wp.float32, device=device)
        if shape_body is None:
            shape_body = wp.empty(0, dtype=wp.int32, device=device)
        if body_flags is None:
            body_flags = wp.empty(0, dtype=wp.int32, device=device)
        if shape_displacement is not None and shape_displacement.shape[0] != shape_lower.shape[0]:
            raise ValueError(
                "shape_displacement length must match the shape bounds "
                f"({shape_lower.shape[0]}), got {shape_displacement.shape[0]}"
            )

        inputs = [
            shape_lower,
            shape_upper,
            shape_gap,
            shape_displacement,
            shape_pairs,
            shape_body,
            body_flags,
            include_static_kinematic_pairs,
            candidate_pair,
            candidate_pair_count,
            max_candidate_pair,
        ]
        wp.launch(
            kernel=_nxn_broadphase_precomputed_pairs
            if forecast is None
            else _nxn_broadphase_precomputed_pairs_forecast,
            dim=shape_pair_count,
            inputs=inputs if forecast is None else [*inputs, forecast],
            device=device,
            record_tape=False,
        )


def replay_shape_pair_templates(
    global_pairs: np.ndarray,
    templates: Sequence[tuple[np.ndarray, np.ndarray]],
    runs: Sequence[tuple[int, np.ndarray]],
) -> np.ndarray:
    """Expand per-world pair templates into the explicit pair list they describe.

    The list holds ``global_pairs`` first. Then, run by run and world by world, it
    holds the template's global-local pairs, ordered ``(min, max)`` after the
    world-local id is shifted by the world's first shape, followed by the
    template's local pairs shifted the same way.

    Args:
        global_pairs: Pairs between global shapes, shape [global_pair_count, 2], int32.
        templates: Per template, its global-local pairs (absolute global id, world-local
            id), shape [n, 2], and its local pairs, shape [m, 2], both int32.
        runs: Consecutive worlds that share a template: the template index and the first
            shape of each world, shape [world_count], int32.

    Returns:
        The explicit pair list, shape [pair_count, 2], int32.
    """
    chunks = []
    if global_pairs.shape[0]:
        chunks.append(global_pairs)
    for template, starts in runs:
        global_local_pairs, local_pairs = templates[template]
        offsets = np.asarray(starts, dtype=np.int32)
        global_count = global_local_pairs.shape[0]
        pairs_per_world = global_count + local_pairs.shape[0]
        if pairs_per_world == 0:
            continue
        replay = np.empty((offsets.shape[0], pairs_per_world, 2), dtype=np.int32)
        if global_count:
            global_replay = replay[:, :global_count, :]
            global_replay[:, :, 0] = global_local_pairs[:, 0]
            global_replay[:, :, 1] = global_local_pairs[:, 1] + offsets[:, None]
            global_replay.sort(axis=2)
        if pairs_per_world > global_count:
            replay[:, global_count:, :] = local_pairs[None, :, :] + offsets[:, None, None]
        chunks.append(replay.reshape((-1, 2)))
    if chunks:
        return np.concatenate(chunks, axis=0)
    return np.empty((0, 2), dtype=np.int32)


@dataclass(frozen=True, eq=False)
class ShapePairTemplates:
    """Per-world pair templates that replay into an explicit pair list.

    :meth:`~newton.ModelBuilder.finalize` keeps them on the model next to the
    ``shape_contact_pairs`` array it replayed them into, so
    :class:`~newton.CollisionPipeline` can test the templates instead of
    streaming that array.
    """

    global_pairs: np.ndarray
    """Pairs between global shapes, which lead the list, shape [global_pair_count, 2], int32."""
    templates: tuple[tuple[np.ndarray, np.ndarray], ...]
    """Per template, its global-local pairs (absolute global id, world-local id), shape [n, 2],
    and its local pairs, shape [m, 2], both int32."""
    runs: tuple[tuple[int, np.ndarray], ...]
    """Consecutive worlds that share a template: the template index and each world's first
    shape, shape [world_count], int32."""
    pairs_ptr: int = 0
    """Address of the array the replay was uploaded to; 0 while unbound."""
    pair_count: int = 0
    """Length of that array."""
    pairs_device: str = ""
    """Device of that array."""

    def replay(self) -> np.ndarray:
        """Return the explicit pair list these templates describe, shape [pair_count, 2], int32."""
        return replay_shape_pair_templates(self.global_pairs, self.templates, self.runs)

    def bind(self, pairs: wp.array[wp.vec2i]) -> ShapePairTemplates:
        """Return a copy that describes ``pairs``, the array :meth:`replay` was uploaded to."""
        return replace(
            self, pairs_ptr=int(pairs.ptr or 0), pair_count=int(pairs.shape[0]), pairs_device=str(pairs.device)
        )

    def describes(self, pairs: object) -> bool:
        """Return whether ``pairs`` is the array this record was bound to (same device, address and length).

        A pipeline still checks the contents on the device before it relies on them.
        """
        return (
            isinstance(pairs, wp.array)
            and str(pairs.device) == self.pairs_device
            and int(pairs.ptr or 0) == self.pairs_ptr
            and int(pairs.shape[0]) == self.pair_count
        )


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, int(value) - 1).bit_length()


def _template_layout(global_local: np.ndarray, local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stage list and packed entries of one template, entries in replay order.

    The stage list holds the referenced world-local ids, ascending, then the
    referenced global ids encoded as ``-(id + 1)``. An entry packs the stage
    indices of its two shapes as ``a | (b << 16)``.
    """
    local_ids = np.unique(np.concatenate((global_local[:, 1], local.reshape(-1)))).astype(np.int64)
    global_ids = np.unique(global_local[:, 0]).astype(np.int64)
    stage = np.concatenate((local_ids, -global_ids - 1)).astype(np.int32)
    if stage.shape[0] > 1 << 16:
        return stage, np.empty(0, dtype=np.int32)
    first = np.concatenate(
        (local_ids.shape[0] + np.searchsorted(global_ids, global_local[:, 0]), np.searchsorted(local_ids, local[:, 0]))
    )
    second = np.concatenate((np.searchsorted(local_ids, global_local[:, 1]), np.searchsorted(local_ids, local[:, 1])))
    return stage, (first | (second << 16)).astype(np.int32)


class BroadPhaseExplicitTemplates:
    """Explicit broad phase over per-world pair templates instead of the flat pair list.

    Each world stages the AABBs its template references once (in shared memory on
    CUDA) and tests the template's entries against them, so the explicit list is
    not streamed. The global-global pairs that lead the list are read from it, a
    chunk per leading block. Built from :class:`ShapePairTemplates` and verified
    against the list they replay into, it tests exactly that list's pairs with the
    overlap and immovable-pair rules of :class:`BroadPhaseExplicit` and writes them
    as the list stores them. Only the order of the output differs on CUDA, where it
    is set by atomics in both kernels; on CPU the output equals
    :class:`BroadPhaseExplicit`'s. AABBs must be pre-expanded: per-shape gaps and
    speculative displacement are not supported.
    """

    AUTO_MIN_PAIRS = 1 << 20
    """Template pairs below which automatic selection keeps the flat kernel (the stream it saves is small).

    Global-global pairs do not count: they are streamed from the list as in the flat kernel."""
    AUTO_MIN_PAIRS_PER_WORLD = 128
    """Mean pairs per world below which automatic selection keeps the flat kernel.

    Measured on an RTX PRO 6000 at 16384 worlds: at 64 pairs per world the template kernel only ties
    the flat one (0.998x, and 1.002x with 1000 global shapes); from 137 it wins (0.92x, 0.78x at 166)."""
    MAX_STAGED_SHAPES = 1024
    """Most shapes a world's template may reference."""
    _SHARED_BYTES = 32 * 1024
    _STAGED_SHAPE_BYTES = 32  # six bounds, the shape id and the immovable flag
    _GLOBAL_PAIRS_PER_LANE = 8  # global-global pairs per lane of a leading block

    def __init__(self, templates: ShapePairTemplates, *, device: Devicelike = None):
        """Lay the templates out for the device.

        Args:
            templates: Templates of the explicit pair list this broad phase replaces.
            device: Device of the explicit pair list and of the AABBs.
        """
        self.device = wp.get_device(device)
        self.global_count = int(templates.global_pairs.shape[0])
        stages: list[np.ndarray] = []
        entries: list[np.ndarray] = []
        compact: dict[int, int] = {}
        slot_starts: list[np.ndarray] = []
        slot_templates: list[np.ndarray] = []
        slot_counts: list[np.ndarray] = []
        for template, starts in templates.runs:
            global_local, local = templates.templates[template]
            entry_count = int(global_local.shape[0] + local.shape[0])
            if entry_count == 0:
                continue
            index = compact.get(template)
            if index is None:
                index = compact[template] = len(stages)
                stage, packed = _template_layout(global_local, local)
                stages.append(stage)
                entries.append(packed)
            world_starts = np.asarray(starts, dtype=np.int32)
            slot_starts.append(world_starts)
            slot_templates.append(np.full(world_starts.shape[0], index, dtype=np.int32))
            slot_counts.append(np.full(world_starts.shape[0], entry_count, dtype=np.int64))

        def joined(parts: list[np.ndarray], dtype) -> np.ndarray:
            return np.concatenate(parts).astype(dtype) if parts else np.empty(0, dtype=dtype)

        counts = joined(slot_counts, np.int64)
        self.template_count = len(stages)
        self.slot_count = int(counts.shape[0])
        self.template_pair_count = int(counts.sum())
        self.pair_count = self.global_count + self.template_pair_count
        self.max_pairs_per_world = max((packed.shape[0] for packed in entries), default=0)
        self.mean_pairs_per_world = float(counts.mean()) if counts.shape[0] else 0.0
        self.staged_shapes = max((stage.shape[0] for stage in stages), default=0)
        self.threads_per_world = min(256, max(32, _next_power_of_two(math.ceil(self.max_pairs_per_world / 8))))
        self.staged_capacity = max(32, -(-self.staged_shapes // 32) * 32)
        self.worlds_per_block = max(
            1,
            min(256 // self.threads_per_world, self._SHARED_BYTES // (self.staged_capacity * self._STAGED_SHAPE_BYTES)),
        )
        self.block_dim = self.threads_per_world * self.worlds_per_block
        self.global_chunk = self.block_dim * self._GLOBAL_PAIRS_PER_LANE
        self.global_blocks = -(-self.global_count // self.global_chunk)
        self.unsupported_reason = None
        self.data = None
        if self.staged_shapes > self.MAX_STAGED_SHAPES:
            self.unsupported_reason = (
                f"A world template references {self.staged_shapes} shapes; the template kernel stages at most "
                f"{self.MAX_STAGED_SHAPES}."
            )
            return
        if self.pair_count > np.iinfo(np.int32).max - self.block_dim:
            self.unsupported_reason = "The explicit pair list exceeds 32-bit indexing."
            return

        stage_start = np.zeros(len(stages) + 1, dtype=np.int64)
        stage_start[1:] = np.cumsum([stage.shape[0] for stage in stages])
        entry_start = np.zeros(len(entries) + 1, dtype=np.int64)
        entry_start[1:] = np.cumsum([packed.shape[0] for packed in entries])
        slot_pair_start = np.zeros(self.slot_count, dtype=np.int64)
        slot_pair_start[1:] = np.cumsum(counts)[:-1]
        # Host layout, uploaded by verify() (or the first launch), so a pipeline that keeps the
        # flat kernel without verifying allocates nothing.
        self._host = {
            "slot_shape_start": joined(slot_starts, np.int32),
            "slot_template": joined(slot_templates, np.int32),
            "template_stage_start": stage_start.astype(np.int32),
            "template_entry_start": entry_start.astype(np.int32),
            "stage": joined(stages, np.int32),
            "entries": joined(entries, np.int32),
            "slot_pair_start": (self.global_count + slot_pair_start).astype(np.int32),
        }

    def _upload(self) -> None:
        """Upload the layout and load the kernel for this launch shape, once."""
        if self.data is not None:
            return

        def device_array(name: str) -> wp.array[int]:
            return wp.array(self._host[name], dtype=wp.int32, device=self.device)

        data = _PairTemplateData()
        data.global_count = self.global_count
        data.global_blocks = self.global_blocks
        data.global_chunk = self.global_chunk
        for name in (
            "slot_shape_start",
            "slot_template",
            "template_stage_start",
            "template_entry_start",
            "stage",
            "entries",
        ):
            setattr(data, name, device_array(name))
        self._slot_pair_start = device_array("slot_pair_start")
        self._empty = wp.empty(0, dtype=wp.int32, device=self.device)
        self._kernel = _pair_template_kernel(self.staged_capacity, self.threads_per_world, self.worlds_per_block)
        self._forecast_kernel = _pair_template_kernel(
            self.staged_capacity, self.threads_per_world, self.worlds_per_block, True
        )
        # Load the variants launch() runs (Warp launches CPU kernels with block_dim 1) outside any
        # capture: drivers older than CUDA 12.3 cannot load a module during graph capture, and
        # capture_begin then preloads only the variants already loaded, else block_dim 256.
        for kernel in (self._kernel, self._forecast_kernel):
            wp.load_module(kernel.module, device=self.device, block_dim=1 if self.device.is_cpu else self.block_dim)
        self.data = data

    @property
    def automatic(self) -> bool:
        """Whether automatic selection uses these templates: enough template pairs, and enough per world."""
        return (
            self.unsupported_reason is None
            and self.template_pair_count >= self.AUTO_MIN_PAIRS
            and self.mean_pairs_per_world >= self.AUTO_MIN_PAIRS_PER_WORLD
        )

    def describe(self) -> dict:
        """Host-only, JSON-compliant summary of the layout and launch shape."""
        return {
            "templates": self.template_count,
            "worlds": self.slot_count,
            "global_pairs": self.global_count,
            "global_blocks": self.global_blocks,
            "pairs": self.pair_count,
            "max_pairs_per_world": self.max_pairs_per_world,
            "mean_pairs_per_world": self.mean_pairs_per_world,
            "staged_shapes": self.staged_shapes,
            "threads_per_world": self.threads_per_world,
            "worlds_per_block": self.worlds_per_block,
        }

    @staticmethod
    def flat_kernel_orders_writes() -> bool:
        """Whether Warp's deterministic mode orders :class:`BroadPhaseExplicit`'s candidate writes.

        The flat kernel appends each hit with a counter atomic, which a deterministic module
        runs in ordered passes; this kernel's per-warp atomics cannot be ordered that way.
        """
        return int(_nxn_broadphase_precomputed_pairs.module.options.get("deterministic") or 0) != 0

    def verify(self, pairs: wp.array[wp.vec2i]) -> bool:
        """Return whether the templates enumerate ``pairs`` exactly, position by position.

        Global-global pairs are read from ``pairs`` itself; every template entry is
        decoded on the device and compared with the pair at its position. Launches,
        allocates the device layout, loads the template kernel and synchronizes, so
        call it at construction, outside graph capture; a verified layout neither
        allocates nor loads a module in :meth:`launch`.
        """
        if (
            self.unsupported_reason is not None
            or not isinstance(pairs, wp.array)
            or pairs.dtype != wp.vec2i
            or pairs.ndim != 1
            or not pairs.is_contiguous
            or str(pairs.device) != str(self.device)
            or pairs.shape[0] != self.pair_count
        ):
            return False
        self._upload()
        if self.slot_count == 0:
            return True
        mismatch = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch_tiled(
            _verify_pair_templates,
            dim=[self.slot_count],
            inputs=[self.data, self._slot_pair_start, pairs, 1 if self.device.is_cpu else 256, mismatch],
            block_dim=256,
            device=self.device,
        )
        return int(mismatch.numpy()[0]) == 0

    def launch(
        self,
        shape_lower: wp.array[wp.vec3],
        shape_upper: wp.array[wp.vec3],
        shape_pairs: wp.array[wp.vec2i],
        candidate_pair: wp.array[wp.vec2i],
        candidate_pair_count: wp.array[int],
        *,
        shape_body: wp.array[int] | None = None,
        body_flags: wp.array[int] | None = None,
        include_static_kinematic_pairs: bool = True,
        forecast: ShapeForecast | None = None,
    ) -> None:
        """Test the templates against pre-expanded AABBs.

        The caller zeroes ``candidate_pair_count``, as with ``skip_count_zero=True``.
        Without a prior :meth:`verify`, the first call allocates the device layout and
        loads the kernel.

        Args:
            shape_lower: Lower AABB bounds, expanded by the effective gaps, shape [shape_count].
            shape_upper: Upper AABB bounds, expanded by the effective gaps, shape [shape_count].
            shape_pairs: The explicit pair list these templates were verified against.
            candidate_pair: Output pairs; its length is the capacity.
            candidate_pair_count: Number of overlapping pairs found, shape [1].
            shape_body: Body of each shape; omitting it disables immovable-pair filtering.
            body_flags: Body flags used to identify kinematic bodies.
            include_static_kinematic_pairs: Whether to keep pairs whose shapes are both immovable.
            forecast: On a forecasting pass, the contact forecast the tested pairs add into, which the kernel's
                forecasting variant reads from the staged shapes
                (:class:`~newton._src.geometry.broad_phase_common.ShapeForecast`); ``None`` runs the plain kernel.
        """
        if self.unsupported_reason is not None:
            raise ValueError(self.unsupported_reason)
        blocks = self.global_blocks + -(-self.slot_count // self.worlds_per_block)
        if blocks == 0:
            return
        self._upload()
        inputs = [
            shape_lower,
            shape_upper,
            shape_pairs,
            self.data,
            self._empty if shape_body is None else shape_body,
            self._empty if body_flags is None else body_flags,
            include_static_kinematic_pairs,
            candidate_pair,
            candidate_pair_count,
            candidate_pair.shape[0],
        ]
        kernel = self._kernel
        if forecast is not None:
            kernel = self._forecast_kernel
            inputs.append(forecast)
        wp.launch_tiled(
            kernel,
            dim=[blocks],
            inputs=inputs,
            block_dim=self.block_dim,
            device=self.device,
            record_tape=False,
        )
