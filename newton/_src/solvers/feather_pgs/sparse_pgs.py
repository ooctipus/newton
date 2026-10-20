# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Ordered PGS over ancestor-sparse, mass-whitened constraint rows."""

import textwrap
from functools import cache

import warp as wp

from .device_preludes import cuda_native
from .friction import FRICTION_PAIR, FRICTION_PAIR_CACHED
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
)
from .row_inputs import RowArgument, make_row_kernel
from .solve_policy import EXACT_POLICY, SolvePolicy, sweep_exit, vote
from .sparse_contact import CONTACT_RECORD_FLOATS, CONTACT_RING_FLOATS
from .sparse_mass_matrix import _SparseRowSupport
from .velocity_limit_law import accumulated_impulse, accumulated_update, check_velocity_limit_law

_SPARSE_CONTACT_METRIC_CACHE_SIZE = 12
_SPARSE_LOCAL_WORKSPACE_BYTES = 8 * 1024
CONTACT_TRANSACTIONS = ("auto", "single", "staged")
"""Triplet transactions of the sparse sweep; ``auto`` is ``single`` for linked rows, else ``staged``."""


def contact_transaction_launch_bounds(max_constraints: int, transaction_blocks: int = 16) -> tuple[int, int]:
    """Return the launch bounds of a single-stage triplet sweep: 64 threads, minimum blocks per SM.

    Above 128 rows, shared memory admits 12 blocks of two worlds; at or below,
    ``transaction_blocks`` (14 or 16) sets the register budget.

    Args:
        max_constraints: Row capacity per world.
        transaction_blocks: Minimum resident blocks requested at or below 128 rows.
    """
    if transaction_blocks not in (14, 16):
        raise ValueError("transaction_blocks must be 14 or 16")
    return 64, 12 if int(max_constraints) > 128 else int(transaction_blocks)


@cache
def _get_pgs_solve_sparse_kernel(
    max_constraints: int,
    max_world_dofs: int,
    max_row_dofs: int,
    *,
    mf_max_constraints: int = 0,
    free_row_dofs: int = 0,
    active_rows: bool = False,
    use_support_table: bool = False,
    contact_triplets: bool = False,
    velocity_limits: bool = False,
    lanes_per_world: int = 32,
    local_factors: bool = False,
    warm_start: bool = False,
    linked_rows: bool = False,
    velocity_limit_law: str = "stateless",
    solve_policy: SolvePolicy = EXACT_POLICY,
    contact_transaction: str = "auto",
    transaction_blocks: int = 16,
) -> wp.Kernel:
    """Build a CUDA sweep over factored articulations and physical free bodies.

    Rows contain unique factor-coordinate indices, padded with -1. Both
    tangent rows of a contact must have identical, ordered support. The caller
    supplies ``row_incident = J v_hat``. Factor blocks store ``Z = J P^T L^-T``
    once; physical free blocks store J and their response separately. All
    coordinates accumulate deltas from v_hat; only factor blocks need decoding.

    This uses the existing interleaved row order and paired friction law,
    including linked patch normal loads. The caller admits only augmented
    drives, immediate response, and no warm start or torsion. ``velocity_limits``
    adds a final articulated limit pass to each sweep, whose unrelaxed
    corrections under ``velocity_limit_law`` accumulate in the factor delta.
    It requires full ordered rows and no independent free-body response.
    Seeded patch impulses retain the ordinary delta semantics.
    Independent free-body rows follow the articulated rows in every sweep,
    retaining their own response, capacity, friction law, and impulse buffer.
    ``active_rows`` selects point-contact normal/limit rows at sweep entry,
    then retains their ordered GS updates. Newly violated zero rows wait until
    the next sweep; mixed responses and linked patches retain the full sweep.

    ``linked_rows`` admits ring-linked patch normals to the active-row scan.
    A first tangent is then selected while any normal of its patch ring is
    selected at sweep entry, or while either of its tangents holds impulse.
    A normal unselected at entry keeps zero impulse through the sweep, so a
    ring without a selected normal keeps zero load, and the ordinary update
    of its zero-impulse pair would be a no-op. Unlinked normals select as
    without ``linked_rows``.

    ``use_support_table`` reads identical padded indices from a shared
    topology table. It requires a single factored articulation per world,
    at offset zero, with no physical free-body response.

    ``contact_triplets`` optionally retains tangent metrics within this call
    and publishes an admitted normal/tangent transaction once. It requires
    active rows and support tables; unsupported owners retain the ordinary
    sweep. The bounded metric cache recomputes uncached pairs when full;
    runtime metadata guards retain the ordinary update for other rows. With
    ``linked_rows`` the transaction also admits ring-linked patch anchors, in
    the active or the full sweep: its radius pools the ring from the new
    normal on, in the order the ordinary pair update pools it, and its early
    exit tests each change as the ordinary updates do.

    ``contact_transaction`` selects how an admitted triplet updates under that
    law. ``"staged"`` is the transaction above: the normal's load and
    reduction, then its tangents', with the metric cache. ``"single"`` loads
    the three rows, the triplet's record and its friction coefficient in one
    group and reduces the three residuals together. The record, which the
    sparse contact builder writes (``contact_records``, see
    :data:`~.sparse_contact.CONTACT_RECORD_FLOATS`), supplies ``1/W_nn`` for the normal step,
    ``W_nt`` to move the tangent residuals by the normal change, and the
    pair's eigenbasis; there is no metric cache or capacity. Its iterates
    differ from ``"staged"`` by rounding only. With ``linked_rows`` one pass
    per call stores each anchor's other ring members in its record, and a
    visit pools the ring from them; rings of more than four members keep
    the walk. ``"auto"`` selects ``"single"`` for linked rows. The kernel
    then takes ``contact_records`` of shape ``(world_count, max_constraints
    // 3, CONTACT_RECORD_FLOATS (+ CONTACT_RING_FLOATS when linked))`` after
    ``iteration_offset`` and launch bounds from
    :func:`contact_transaction_launch_bounds` with ``transaction_blocks``.

    Small complete articulated sweeps may use 8 or 16 lanes per world, as in
    the local response/solve owner. Launch with ``ceil(world_count / (64 /
    lanes_per_world))`` tiles and 64 threads per tile. Active-row bitsets and
    independent free-body work retain whole warps.

    ``local_factors`` retains small factor rows in private shared storage,
    like the local response/solve owner. It expands sparse indices once per
    call; it neither applies a second factor transform nor stores physical Y.

    ``warm_start`` seeds the factor delta with the response of the impulses
    already stored in the row buffer before the first sweep, so carried
    contact impulses act on the velocity exactly as the ordinary route
    installs them in ``v_out``. Warm start seeds no velocity-limit row. It
    requires no independent free-body rows.

    ``velocity_limit_law="accumulated"`` makes the final velocity-limit pass
    and independent free-body velocity-limit rows accumulate unrelaxed
    unilateral impulses (:mod:`.velocity_limit_law`); velocity-limit rows enter
    every solve with zero impulse, so warm starting still seeds none of them.
    The stateless default keeps the generated source byte-identical, and a
    sweep without velocity-limit rows ignores the law.

    ``solve_policy`` compiles the sweep's stopping rule and readout
    (:mod:`solve_policy`); a readout appends its operand after ``support``.
    Under a velocity tolerance an active-row sweep also bounds, before it
    stops, the velocity change a visit would make to every zero-impulse
    unilateral row (its violation times ``omega``), which its scan may have
    left unvisited.
    """
    M, D, S = int(max_constraints), int(max_world_dofs), int(max_row_dofs)
    F = int(mf_max_constraints)
    P = int(free_row_dofs)
    if M <= 0 or D <= 0 or S <= 0 or S > D or F < 0 or P < 0 or P > S:
        raise ValueError("sparse PGS requires positive capacities and max_row_dofs <= max_world_dofs")
    if use_support_table and (F or P or D > 64):
        raise ValueError("sparse support tables require at most 64 factored DOFs and no free rows")
    if velocity_limits and (F or P or active_rows or contact_triplets):
        raise ValueError("sparse joint velocity limits require full articulated row sweeps")
    L = int(lanes_per_world)
    if L not in (8, 16, 32):
        raise ValueError("sparse PGS requires 8, 16 or 32 lanes per world")
    if L < 32 and (D > L or F or P or active_rows or contact_triplets):
        raise ValueError("subwarp sparse PGS requires a small complete articulated sweep")
    if local_factors and (D > 16 or M > 64 or F or P or active_rows or contact_triplets):
        raise ValueError("local factor storage requires a small complete articulated sweep")
    if warm_start and F:
        raise ValueError("sparse warm starting requires no independent free-body rows")
    if contact_transaction not in CONTACT_TRANSACTIONS:
        raise ValueError(f"contact_transaction must be one of {CONTACT_TRANSACTIONS}")
    launch_bounds = contact_transaction_launch_bounds(M, transaction_blocks)
    # Only a sweep that can hold velocity-limit rows has a law to choose; the others share one variant.
    accumulated_limits = check_velocity_limit_law(velocity_limit_law) == "accumulated" and bool(velocity_limits or F)
    index_read = (
        f"support.class_dof.data[support.row_class.data[offset + row] * {S} + entry]"
        if use_support_table
        else f"row_dof.data[row_base + row * {S} + entry]"
    )
    W = 64 // L
    row_width = D if local_factors else S
    Q = (row_width + L - 1) // L
    factor_read = f"row_factor.data[row_base + row * {S} + entry]"
    sibling_read = f"row_factor.data[row_base + sibling * {S} + entry]"
    factor_storage = factor_initialize = ""
    if local_factors:
        factor_storage = f"""
    __shared__ float local_factor_storage[{W * M * D}];
    float* local_factor = local_factor_storage + slot * {M * D};
"""
        factor_initialize = f"""
    for (int i = lane; i < count * {D}; i += {L}) local_factor[i] = 0.0f;
    __syncwarp(MASK);
    for (int i = lane; i < count * {S}; i += {L}) {{
        const int row = i / {S}, entry = i % {S};
        const int index = {index_read};
        if (index >= 0) local_factor[row * {D} + index] = {factor_read};
    }}
"""
        index_read = "entry"
        factor_read = f"local_factor[row * {D} + entry]"
        sibling_read = f"local_factor[sibling * {D} + entry]"
    contact_type = int(PGS_CONSTRAINT_TYPE_CONTACT)
    friction_type = int(PGS_CONSTRAINT_TYPE_FRICTION)
    limit_type = int(PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
    velocity_limit_type = int(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
    active_rows = bool(active_rows) and F == 0 and P == 0
    # Linked triplets pool the ring as the pair update does, so they fuse in either sweep.
    contact_triplets = bool(contact_triplets) and (active_rows or bool(linked_rows)) and use_support_table and M >= 3
    linked_rows = bool(linked_rows) and (active_rows or contact_triplets)
    # Patch friction takes the single-stage transaction by default; point friction keeps the staged one.
    if contact_transaction == "auto":
        contact_transaction = "single" if linked_rows else "staged"
    single_stage = contact_triplets and contact_transaction == "single"
    active_storage = active_initialize = active_scan = ""
    row_loop = "        for (int row = 0; row < count; ++row) {"
    row_loop_end = "        }"
    if active_rows:
        words = (M + 31) // 32
        active_storage = f"""
    __shared__ unsigned normal_words_storage[{W * words}];
    __shared__ unsigned selected_words_storage[{W * words}];
    unsigned* normal_words = normal_words_storage + slot * {words};
    unsigned* selected_words = selected_words_storage + slot * {words};
"""
        active_initialize = f"""
    bool linked = false;
    for (int row = lane; row < count; row += 32)
        if (type[row] == {contact_type} && world_row_parent.data[offset + row] >= 0) linked = true;
    const bool select_active = __ballot_sync(MASK, linked) == 0u && isfinite(omega) && omega >= 0.0f;
"""
        tangent_selection = """selected = row == parent + 1 && (
                        ((normal_words[parent / 32] >> (parent & 31)) & 1u) != 0u
                        || impulse[row] != 0.0f || impulse[parent + 2] != 0.0f);"""
        if linked_rows:
            # Rings select by their pooled load below; no linked world falls back.
            active_initialize = """
    const bool select_active = isfinite(omega) && omega >= 0.0f;
"""
            # Walk the ring as the pair update pools its load, from the pair's own normal.
            tangent_selection = """selected = row == parent + 1 && (impulse[row] != 0.0f || impulse[parent + 2] != 0.0f);
                    for (int next = parent; row == parent + 1 && !selected && next >= 0;) {
                        selected = ((normal_words[next / 32] >> (next & 31)) & 1u) != 0u;
                        next = world_row_parent.data[offset + next];
                        if (next == parent) break;
                    }"""
        active_scan = f"""
        for (int first = 0; first < count; first += 32) {{
            const int row = first + lane;
            bool selected = false;
            if (row < count) {{
                const int kind = static_cast<int>(type[row]);
                if (kind != {friction_type}) {{
                    selected = !select_active || (kind != {contact_type} && kind != {limit_type})
                        || impulse[row] != 0.0f;
                    if (!selected && diagonal[row] > 0.0f) {{
                        float dot = 0.0f;
                        // Delta starts at zero even when the impulse buffer is seeded,
                        // unless warm starting seeded it before the first sweep.
                        if (iteration != 0 || {str(bool(warm_start)).lower()}) {{
                            #pragma unroll
                            for (int entry = 0; entry < {S}; ++entry) {{
                                const int index = {index_read};
                                if (index >= 0)
                                    dot += row_factor.data[row_base + row * {S} + entry] * delta[index];
                            }}
                        }}
                        const float residual = dot + rhs[row];
                        selected = residual < 0.0f || !isfinite(residual);
                    }}
                }} else if (select_active && global_iteration < friction_start_iteration) {{
                    // The active list visits only first tangents; clear both delayed rows here.
                    impulse[row] = 0.0f;
                }}
            }}
            const unsigned mask = __ballot_sync(MASK, selected);
            if (lane == 0) normal_words[first / 32] = mask;
        }}
        __syncwarp(MASK);
        for (int first = 0; first < count; first += 32) {{
            const int row = first + lane;
            bool selected = false;
            if (row < count) {{
                selected = !select_active || ((normal_words[first / 32] >> lane) & 1u) != 0u;
                if (select_active && type[row] == {friction_type}
                    && global_iteration >= friction_start_iteration) {{
                    const int parent = world_row_parent.data[offset + row];
                    {tangent_selection}
                }}
            }}
            const unsigned mask = __ballot_sync(MASK, selected);
            if (lane == 0) selected_words[first / 32] = mask;
        }}
        __syncwarp(MASK);
"""
        row_loop = """
        for (int word = 0; word < (count + 31) / 32; ++word) {
            unsigned pending = selected_words[word];
            while (pending != 0u) {
                const int row = word * 32 + __ffs(pending) - 1;
                pending &= pending - 1u;"""
        row_loop_end = "            }\n        }"
    impulse_slack = "change_tolerance + relative_change_tolerance * fabsf(new_impulse)"
    mf_storage = mf_initialize = mf_sweep = mf_writeback = ""
    if accumulated_limits:
        free_velocity_limit = (
            "                // Velocity limits accumulate an unrelaxed unilateral impulse.\n"
            f"                new_impulse = {accumulated_impulse('old_impulse', 'residual * inverse_diagonal')};"
        )
        free_change = "const float change = new_impulse - old_impulse;"
    else:
        free_velocity_limit = """                // Velocity limits apply a stateless, unrelaxed correction.
                new_impulse = residual < 0.0f ? -residual * inverse_diagonal : 0.0f;"""
        free_change = f"const float change = kind == {velocity_limit_type} ? new_impulse : new_impulse - old_impulse;"
    if F:
        sibling_inverse = "__int_as_float(mf_meta.data[(mf_offset + sibling) * 4 + 1])"
        mf_vote = vote(
            solve_policy,
            """                if (!(fabsf(change) <= change_tolerance + relative_change_tolerance * fabsf(new_impulse)
                      && fabsf(sibling_change) <= change_tolerance + relative_change_tolerance * fabsf(new_impulse))) changed = 1;""",
            flag="changed",
            terms=(
                ("change", "inverse_diagonal > 0.0f ? 1.0f / inverse_diagonal : 0.0f", "kind"),
                ("sibling_change", f"{sibling_inverse} > 0.0f ? 1.0f / {sibling_inverse} : 0.0f", str(friction_type)),
            ),
            slack=(impulse_slack, impulse_slack),
        )
        mf_storage = f"""
    __shared__ float mf_impulse_storage[{W * F}];
    __shared__ float mf_incident_storage[{W * F}];
    const int mf_offset = world * {F};
    const int mf_count = min(mf_constraint_count.data[world], {F});
    float* mf_impulse = mf_impulse_storage + slot * {F};
    float* mf_incident = mf_incident_storage + slot * {F};
"""
        mf_initialize = f"""
    for (int i = lane; i < mf_count; i += 32) {{
        const int global_row = mf_offset + i;
        mf_impulse[i] = mf_impulses.data[global_row];
        const int dofs = mf_meta.data[global_row * 4];
        const int first = dofs >> 16;
        const int second = (dofs << 16) >> 16;
        float incident = 0.0f;
        for (int k = 0; k < 6; ++k) {{
            if (first >= 0)
                incident += mf_J_a.data[global_row * 6 + k]
                    * v_hat.data[world_dof_indices.data[world * {D} + first + k]];
            if (second >= 0)
                incident += mf_J_b.data[global_row * 6 + k]
                    * v_hat.data[world_dof_indices.data[world * {D} + second + k]];
        }}
        mf_incident[i] = incident;
    }}
"""
        mf_sweep = f"""
        for (int row = 0; row < mf_count; ++row) {{
            const int global_row = mf_offset + row;
            const int4 meta = *reinterpret_cast<const int4*>(&mf_meta.data[global_row * 4]);
            const int kind = meta.w & 0xffff;
            const bool friction = kind == {friction_type};
            if (friction && global_iteration < friction_start_iteration) {{
                if (lane == 0) mf_impulse[row] = 0.0f;
                __syncwarp(MASK);
                continue;
            }}
            const int parent = meta.w >> 16;
            if (friction && row != parent + 1) continue;
            const int sibling = friction ? parent + 2 : -1;
            const float inverse_diagonal = __int_as_float(meta.y);
            if (!friction && inverse_diagonal <= 0.0f) continue;
            const int first = meta.x >> 16;
            const int second = (meta.x << 16) >> 16;
            int index = -1;
            if (lane < 6 && first >= 0) index = first + lane;
            if (lane >= 6 && lane < 12 && second >= 0) index = second + lane - 6;
            float left = 0.0f, right = 0.0f, sibling_left = 0.0f, sibling_right = 0.0f;
            if (index >= 0 && lane < 6) {{
                left = mf_J_a.data[global_row * 6 + lane];
                right = mf_MiJt_a.data[global_row * 6 + lane];
                if (friction) {{
                    sibling_left = mf_J_a.data[(mf_offset + sibling) * 6 + lane];
                    sibling_right = mf_MiJt_a.data[(mf_offset + sibling) * 6 + lane];
                }}
            }}
            if (index >= 0 && lane >= 6 && lane < 12) {{
                left = mf_J_b.data[global_row * 6 + lane - 6];
                right = mf_MiJt_b.data[global_row * 6 + lane - 6];
                if (friction) {{
                    sibling_left = mf_J_b.data[(mf_offset + sibling) * 6 + lane - 6];
                    sibling_right = mf_MiJt_b.data[(mf_offset + sibling) * 6 + lane - 6];
                }}
            }}
            // A coordinate has one writer, including coincident endpoint blocks.
            if (first >= 0 && first == second) {{
                if (lane < 6) {{
                    left += mf_J_b.data[global_row * 6 + lane];
                    right += mf_MiJt_b.data[global_row * 6 + lane];
                    if (friction) {{
                        sibling_left += mf_J_b.data[(mf_offset + sibling) * 6 + lane];
                        sibling_right += mf_MiJt_b.data[(mf_offset + sibling) * 6 + lane];
                    }}
                }} else {{
                    index = -1;
                    left = right = sibling_left = sibling_right = 0.0f;
                }}
            }}
            float dot = index >= 0 ? left * delta[index] : 0.0f;
            float sibling_dot = index >= 0 ? sibling_left * delta[index] : 0.0f;
            float cross = left * sibling_right;
            for (int shift = 16; shift > 0; shift >>= 1) {{
                dot += __shfl_down_sync(MASK, dot, shift);
                sibling_dot += __shfl_down_sync(MASK, sibling_dot, shift);
                cross += __shfl_down_sync(MASK, cross, shift);
            }}
            const float residual = __shfl_sync(MASK, dot, 0)
                + mf_incident[row] + __int_as_float(meta.z);
            const float old_impulse = mf_impulse[row];
            float new_impulse = old_impulse;
            float sibling_change = 0.0f;
            if (friction) {{
                float normal_load = mf_impulse[parent];
                for (int next = mf_meta.data[(mf_offset + parent) * 4 + 3] >> 16;
                     next >= 0 && next != parent; next = mf_meta.data[(mf_offset + next) * 4 + 3] >> 16)
                    normal_load += mf_impulse[next];
                const float radius = fmaxf(mf_row_mu.data[global_row] * normal_load, 0.0f);
                const float old_sibling = mf_impulse[sibling];
                float2 pair = make_float2(0.0f, 0.0f);
                if (radius > 0.0f) {{
                    const float sibling_inverse = __int_as_float(mf_meta.data[(mf_offset + sibling) * 4 + 1]);
                    const float sibling_residual = __shfl_sync(MASK, sibling_dot, 0)
                        + mf_incident[sibling]
                        + __int_as_float(mf_meta.data[(mf_offset + sibling) * 4 + 2]);
                    pair = friction_pair_candidate(inverse_diagonal > 0.0f ? 1.0f / inverse_diagonal : 0.0f,
                        __shfl_sync(MASK, cross, 0), sibling_inverse > 0.0f ? 1.0f / sibling_inverse : 0.0f,
                        residual, sibling_residual, old_impulse, old_sibling, radius, omega);
                }}
                const float magnitude = sqrtf(pair.x * pair.x + pair.y * pair.y);
                const float scale = magnitude > radius ? radius / magnitude : 1.0f;
                new_impulse = pair.x * scale;
                const float new_sibling = pair.y * scale;
                sibling_change = new_sibling - old_sibling;
                if (lane == 0) mf_impulse[sibling] = new_sibling;
            }} else if (kind == {velocity_limit_type}) {{
{free_velocity_limit}
            }} else {{
                new_impulse = old_impulse - omega * residual * inverse_diagonal;
                if (kind == {contact_type}) new_impulse = fmaxf(new_impulse, 0.0f);
            }}
            {free_change}
            if (lane == 0) mf_impulse[row] = new_impulse;
            if (change != 0.0f || sibling_change != 0.0f) {{
{mf_vote}
                if (index >= 0) {{
                    float value = delta[index];
                    if (sibling_change != 0.0f) value += sibling_right * sibling_change;
                    if (change != 0.0f) value += right * change;
                    delta[index] = value;
                }}
            }}
            __syncwarp(MASK);
        }}
"""
        mf_writeback = """
    for (int i = lane; i < mf_count; i += 32) mf_impulses.data[mf_offset + i] = mf_impulse[i];
"""
    setup = f"""    __shared__ float delta_storage[{W * D}];
    __shared__ float impulse_storage[{W * M}];
    __shared__ float rhs_storage[{W * M}];
    __shared__ float diagonal_storage[{W * M}];
    __shared__ unsigned char type_storage[{W * M}];
    const unsigned MASK = {hex((1 << L) - 1)}u << (threadIdx.x & {32 - L});
    const int lane = threadIdx.x & {L - 1};
    const int slot = threadIdx.x / {L};
    const int offset = world * {M};
    const int row_base = offset * {S};
    const int count = min(world_constraint_count.data[world], {M});
    float* delta = delta_storage + slot * {D};
    float* impulse = impulse_storage + slot * {M};
    float* rhs = rhs_storage + slot * {M};
    float* diagonal = diagonal_storage + slot * {M};
    unsigned char* type = type_storage + slot * {M};
{mf_storage}
{active_storage}
{factor_storage}

    for (int d = lane; d < {D}; d += {L}) delta[d] = 0.0f;
    for (int i = lane; i < count; i += {L}) {{
        impulse[i] = world_impulses.data[offset + i];
        rhs[i] = rhs_bias.data[offset + i] + row_incident.data[offset + i];
        diagonal[i] = world_diag.data[offset + i];
        type[i] = static_cast<unsigned char>(world_row_type.data[offset + i]);
    }}
{mf_initialize}
{factor_initialize}
    __syncwarp(MASK);
{active_initialize}
"""
    if warm_start:
        # Carried impulses already sit in the row buffer; give the factor delta their
        # response so the first residual sees them, as the ordinary route installs them
        # in v_out. Rows own unique indices and lanes own disjoint entries, so one row's
        # updates never collide; rows are serialized by the warp barrier.
        setup += f"""
    for (int row = 0; row < count; ++row) {{
        const float seed = impulse[row];
        if (seed != 0.0f && type[row] != {velocity_limit_type}) {{
            #pragma unroll
            for (int q = 0; q < {Q}; ++q) {{
                const int entry = lane + q * {L};
                const int index = entry < {row_width} ? {index_read} : -1;
                if (index >= 0) {{
                    float response = {factor_read};
                    if ({str(bool(P)).lower()} && world_free_dof_mask.data[world * {D} + index] != 0)
                        response = row_free_response.data[(offset + row) * row_free_response.shape[2] + entry];
                    delta[index] += response * seed;
                }}
            }}
        }}
        __syncwarp(MASK);
    }}
"""
    velocity_filter = velocity_sweep = ""
    if velocity_limits:
        velocity_vote = vote(
            solve_policy,
            "                if (!(fabsf(correction) <= change_tolerance)) changed = 1;",
            flag="changed",
            terms=(("correction", "denominator", str(velocity_limit_type)),),
            slack=("change_tolerance",),
        )
        # Storage puts velocity rows before contacts, but their original law
        # owns the final pass. Scalar rows need only one residual reduction.
        velocity_filter = f"""
            if (row_type == {velocity_limit_type}) continue;
"""
        if accumulated_limits:
            final_velocity_limit = (
                "            // An accumulated total: its change applies to the delta.\n"
                "            const float old_vlim = impulse[row];\n"
                + accumulated_update(
                    "old_vlim",
                    "residual / denominator",
                    new="new_vlim",
                    delta="correction",
                    indent=" " * 12,
                    declare="const float ",
                )
                + "            __syncwarp(MASK);\n"
                + "            if (lane == 0) impulse[row] = new_vlim;"
            )
        else:
            final_velocity_limit = """            // Limits report this correction without undoing earlier ones.
            const float correction = residual < 0.0f ? -residual / denominator : 0.0f;
            if (lane == 0) impulse[row] = correction;"""
        velocity_sweep = f"""
        for (int row = 0; row < count; ++row) {{
            if (type[row] != {velocity_limit_type}) continue;
            const float denominator = diagonal[row];
            if (denominator <= 0.0f) continue;
            int indices[{Q}];
            float responses[{Q}];
            float dot = 0.0f;
            #pragma unroll
            for (int q = 0; q < {Q}; ++q) {{
                const int entry = lane + q * {L};
                const int index = entry < {row_width} ? {index_read} : -1;
                const float response = index >= 0 ? {factor_read} : 0.0f;
                indices[q] = index;
                responses[q] = response;
                if (index >= 0) dot += response * delta[index];
            }}
            for (int shift = {L // 2}; shift > 0; shift >>= 1)
                dot += __shfl_down_sync(MASK, dot, shift, {L});
            const float residual = __shfl_sync(MASK, dot, 0, {L}) + rhs[row];
{final_velocity_limit}
            if (correction != 0.0f) {{
{velocity_vote}
                #pragma unroll
                for (int q = 0; q < {Q}; ++q)
                    if (indices[q] >= 0) delta[indices[q]] += responses[q] * correction;
            }}
            __syncwarp(MASK);
        }}
"""
    row_vote = vote(
        solve_policy,
        """                if (!(fabsf(change) <= change_tolerance + relative_change_tolerance * fabsf(new_impulse)
                      && fabsf(sibling_change) <= change_tolerance + relative_change_tolerance * fabsf(new_impulse))) changed = 1;""",
        flag="changed",
        terms=(("change", "denominator", "row_type"), ("sibling_change", "diagonal[sibling]", str(friction_type))),
        slack=(impulse_slack, impulse_slack),
    )
    row_update = f"""
            const int row_type = static_cast<int>(type[row]);
{velocity_filter}
            const bool friction = row_type == {friction_type};
            if (friction && global_iteration < friction_start_iteration) {{
                if (lane == 0) impulse[row] = 0.0f;
                __syncwarp(MASK);
                continue;
            }}
            const int parent = friction ? world_row_parent.data[offset + row] : -1;
            if (friction && row != parent + 1) continue;
            const int sibling = friction ? parent + 2 : -1;
            const float denominator = diagonal[row];
            if (!friction && denominator <= 0.0f) continue;
            const float old_impulse = impulse[row];
            float old_sibling = 0.0f;
            float radius = 0.0f;
            if (friction) {{
                float normal_load = impulse[parent];
                for (int next = world_row_parent.data[offset + parent]; next >= 0 && next != parent;
                     next = world_row_parent.data[offset + next])
                    normal_load += impulse[next];
                radius = fmaxf(world_row_mu.data[offset + row] * normal_load, 0.0f);
                old_sibling = impulse[sibling];
                // A zero-radius, zero-impulse pair cannot change the velocity.
                if (radius == 0.0f && old_impulse == 0.0f && old_sibling == 0.0f) continue;
            }}

            int indices[{Q}];
            float responses[{Q}];
            float sibling_responses[{Q}];
            float dot = 0.0f;
            float sibling_dot = 0.0f;
            float cross = 0.0f;
            #pragma unroll
            for (int q = 0; q < {Q}; ++q) {{
                const int entry = lane + q * {L};
                const int index = entry < {row_width} ? {index_read} : -1;
                const float value = index >= 0 ? {factor_read} : 0.0f;
                const float sibling_value = friction && index >= 0
                    ? {sibling_read} : 0.0f;
                float response = value;
                float sibling_response = sibling_value;
                if ({str(bool(P)).lower()} && index >= 0 && world_free_dof_mask.data[world * {D} + index] != 0) {{
                    const int width = row_free_response.shape[2];
                    response = row_free_response.data[(offset + row) * width + entry];
                    if (friction)
                        sibling_response = row_free_response.data[(offset + sibling) * width + entry];
                }}
                indices[q] = index;
                responses[q] = response;
                sibling_responses[q] = sibling_response;
                if (index >= 0) {{
                    const float velocity = delta[index];
                    dot += value * velocity;
                    sibling_dot += sibling_value * velocity;
                    cross += value * sibling_response;
                }}
            }}
            for (int shift = {L // 2}; shift > 0; shift >>= 1) {{
                dot += __shfl_down_sync(MASK, dot, shift, {L});
                sibling_dot += __shfl_down_sync(MASK, sibling_dot, shift, {L});
                cross += __shfl_down_sync(MASK, cross, shift, {L});
            }}
            const float residual = __shfl_sync(MASK, dot, 0, {L}) + rhs[row];
            float new_impulse = old_impulse;
            float sibling_change = 0.0f;
            if (friction) {{
                float2 pair = make_float2(0.0f, 0.0f);
                if (radius > 0.0f) {{
                    const float sibling_residual = __shfl_sync(MASK, sibling_dot, 0, {L}) + rhs[sibling];
                    pair = friction_pair_candidate(denominator, __shfl_sync(MASK, cross, 0, {L}),
                        diagonal[sibling], residual, sibling_residual, old_impulse, old_sibling, radius, omega);
                }}
                const float magnitude = sqrtf(pair.x * pair.x + pair.y * pair.y);
                const float scale = magnitude > radius ? radius / magnitude : 1.0f;
                new_impulse = pair.x * scale;
                const float new_sibling = pair.y * scale;
                sibling_change = new_sibling - old_sibling;
                if (lane == 0) impulse[sibling] = new_sibling;
            }} else {{
                new_impulse = old_impulse - omega * residual / denominator;
                if (row_type == {contact_type} || row_type == {limit_type})
                    new_impulse = fmaxf(new_impulse, 0.0f);
            }}
            const float change = new_impulse - old_impulse;
            if (lane == 0) impulse[row] = new_impulse;
            if (change != 0.0f || sibling_change != 0.0f) {{
{row_vote}
                #pragma unroll
                for (int q = 0; q < {Q}; ++q) {{
                    if (indices[q] >= 0) {{
                        // Match the existing paired-tangent response order.
                        float value = delta[indices[q]];
                        if (sibling_change != 0.0f) value += sibling_responses[q] * sibling_change;
                        if (change != 0.0f) value += responses[q] * change;
                        delta[indices[q]] = value;
                    }}
                }}
            }}
            __syncwarp(MASK);
"""
    violation = None
    if active_rows:
        # Rows unselected at sweep entry are not visited: a zero-impulse unilateral row's violation
        # times omega is the velocity change its visit would make, so a tolerance also bounds it.
        violation = (
            f"""float sweep_violation = 0.0f;
int sweep_violation_kind = -1;
for (int first = 0; first < count; first += 32) {{
    const int row = first + lane;
    const int kind = row < count ? static_cast<int>(type[row]) : -1;
    if ((kind == {contact_type} || kind == {limit_type}) && impulse[row] == 0.0f && diagonal[row] > 0.0f) {{
        float dot = 0.0f;
        #pragma unroll
        for (int entry = 0; entry < {S}; ++entry) {{
            const int index = {index_read};
            if (index >= 0) dot += row_factor.data[row_base + row * {S} + entry] * delta[index];
        }}
        const float residual = dot + rhs[row];
        const float excess = isfinite(residual) ? omega * fmaxf(-residual, 0.0f) : __int_as_float(0x7f800000);
        if (excess > sweep_violation) {{
            sweep_violation = excess;
            sweep_violation_kind = kind;
        }}
    }}
}}
for (int offset = 16; offset > 0; offset >>= 1) {{
    const float other = __shfl_xor_sync(MASK, sweep_violation, offset);
    const int other_kind = __shfl_xor_sync(MASK, sweep_violation_kind, offset);
    if (other > sweep_violation || (other == sweep_violation && other_kind > sweep_violation_kind)) {{
        sweep_violation = other;
        sweep_violation_kind = other_kind;
    }}
}}""",
            "sweep_violation",
            "sweep_violation_kind",
        )
    sweep_finish = "\n" + sweep_exit(
        solve_policy,
        "        if (global_iteration >= friction_start_iteration && __ballot_sync(MASK, changed != 0) == 0u) break;\n",
        gate="global_iteration >= friction_start_iteration",
        quiet="__ballot_sync(MASK, changed != 0) == 0u",
        violation=violation,
        lanes=L,
    )
    laws = (FRICTION_PAIR,)
    if single_stage:
        triplet_setup, triplet_update = _single_stage_triplet_code(
            M, S, Q, contact_type, friction_type, linked=linked_rows, active=active_rows, solve_policy=solve_policy
        )
        setup += triplet_setup
        active_scan += "\n        int consumed_tangent = -1;\n"
        # Rows the transaction declines keep the ordinary update and its own pair metric.
        row_update = triplet_update + row_update
        laws = (FRICTION_PAIR, FRICTION_PAIR_CACHED)
    elif contact_triplets:
        triplet_setup, triplet_update = _contact_triplet_code(
            M, S, Q, contact_type, friction_type, linked=linked_rows, active=active_rows, solve_policy=solve_policy
        )
        setup += triplet_setup
        active_scan += "\n        int consumed_tangent = -1;\n"
        row_update = row_update.replace(
            "            int indices[",
            "            const int metric_slot = friction ? contact_metric_slot(parent) : -1;\n"
            "            const bool metric_ready = metric_slot >= 0 && metric_map[metric_slot] >= 0;\n"
            "            int indices[",
            1,
        ).replace("cross += value * sibling_response;", "if (!metric_ready) cross += value * sibling_response;")
        row_update = row_update.replace(
            "cross += __shfl_down_sync(MASK, cross, shift, 32);",
            "if (!metric_ready) cross += __shfl_down_sync(MASK, cross, shift, 32);",
        ).replace(
            "pair = friction_pair_candidate(denominator,",
            "pair = friction_pair_cached(metric_slot, denominator,",
        )
        row_update = triplet_update + row_update
        laws = (FRICTION_PAIR_CACHED,)
    writeback = f"""
    for (int d = lane; d < {D}; d += {L}) factor_velocity_delta.data[world * {D} + d] = delta[d];
    for (int i = lane; i < count; i += {L}) world_impulses.data[offset + i] = impulse[i];
{mf_writeback}
"""
    snippet = cuda_native(
        setup
        + f"""
    for (int iteration = 0; iteration < iterations; ++iteration) {{
        const int global_iteration = iteration_offset + iteration;
        int changed = 0;
{active_scan}
{row_loop}
{row_update}
{row_loop_end}
{mf_sweep}
{velocity_sweep}
{sweep_finish}
    }}
{writeback}
#endif
""",
        preludes=(*laws, *solve_policy.preludes),
        lead="",
    )

    @wp.func_native(snippet)
    def solve_native(
        world: int,
        world_constraint_count: wp.array[int],
        rhs_bias: wp.array2d[float],
        world_diag: wp.array2d[float],
        world_impulses: wp.array2d[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_free_response: wp.array3d[float],
        row_incident: wp.array2d[float],
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        world_free_dof_mask: wp.array2d[int],
        world_dof_indices: wp.array2d[int],
        v_hat: wp.array[float],
        mf_constraint_count: wp.array[int],
        mf_meta: wp.array2d[int],
        mf_impulses: wp.array2d[float],
        mf_J_a: wp.array3d[float],
        mf_J_b: wp.array3d[float],
        mf_MiJt_a: wp.array3d[float],
        mf_MiJt_b: wp.array3d[float],
        mf_row_mu: wp.array2d[float],
        iterations: int,
        omega: float,
        change_tolerance: float,
        relative_change_tolerance: float,
        friction_start_iteration: int,
        iteration_offset: int,
        factor_velocity_delta: wp.array2d[float],
        support: _SparseRowSupport,
    ): ...

    def solve(
        world_count: int,
        world_constraint_count: wp.array[int],
        rhs_bias: wp.array2d[float],
        world_diag: wp.array2d[float],
        world_impulses: wp.array2d[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_free_response: wp.array3d[float],
        row_incident: wp.array2d[float],
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        world_free_dof_mask: wp.array2d[int],
        world_dof_indices: wp.array2d[int],
        v_hat: wp.array[float],
        mf_constraint_count: wp.array[int],
        mf_meta: wp.array2d[int],
        mf_impulses: wp.array2d[float],
        mf_J_a: wp.array3d[float],
        mf_J_b: wp.array3d[float],
        mf_MiJt_a: wp.array3d[float],
        mf_MiJt_b: wp.array3d[float],
        mf_row_mu: wp.array2d[float],
        iterations: int,
        omega: float,
        change_tolerance: float,
        relative_change_tolerance: float,
        friction_start_iteration: int,
        iteration_offset: int,
        factor_velocity_delta: wp.array2d[float],
        support: _SparseRowSupport,
    ):
        block, lane = wp.tid()
        world = block * W + lane // L
        if world < world_count:
            solve_native(
                world,
                world_constraint_count,
                rhs_bias,
                world_diag,
                world_impulses,
                row_dof,
                row_factor,
                row_free_response,
                row_incident,
                world_row_type,
                world_row_parent,
                world_row_mu,
                world_free_dof_mask,
                world_dof_indices,
                v_hat,
                mf_constraint_count,
                mf_meta,
                mf_impulses,
                mf_J_a,
                mf_J_b,
                mf_MiJt_a,
                mf_MiJt_b,
                mf_row_mu,
                iterations,
                omega,
                change_tolerance,
                relative_change_tolerance,
                friction_start_iteration,
                iteration_offset,
                factor_velocity_delta,
                support,
            )

    suffix = "_active" if active_rows else ""
    if use_support_table:
        suffix += "_support_table"
    if contact_triplets:
        suffix += "_contact_triplets"
    if single_stage:
        # The minimum resident blocks of the launch bounds tell the A/B variants apart.
        suffix += f"_single{launch_bounds[1]}"
    if linked_rows:
        suffix += "_linked"
    if velocity_limits:
        suffix += "_velocity_limits"
    if accumulated_limits:
        suffix += "_vacc"
    if L != 32:
        suffix += f"_lanes{L}"
    if local_factors:
        suffix += "_local_factors"
    if warm_start:
        suffix += "_warm_start"
    if single_stage:
        name = f"pgs_solve_sparse_{M}_{D}_{S}_{F}_{P}{suffix}{solve_policy.suffix}"
        records = RowArgument("contact_records", wp.array3d[float], after="iteration_offset")
        kernel = make_row_kernel(
            name,
            snippet,
            solve_native.func,
            (records, *solve_policy.arguments),
            dispatch="tiles",
            groups=W,
            lanes=L,
            launch_bounds=launch_bounds,
        )
        if solve_policy.counts:
            kernel._fpgs_solve_readout = solve_policy.readout
        return kernel
    if solve_policy.counts:
        name = f"pgs_solve_sparse_{M}_{D}_{S}_{F}_{P}{suffix}{solve_policy.suffix}"
        kernel = make_row_kernel(
            name, snippet, solve_native.func, solve_policy.arguments, dispatch="tiles", groups=W, lanes=L
        )
        kernel._fpgs_solve_readout = solve_policy.readout
        return kernel
    solve.__name__ = f"pgs_solve_sparse_{M}_{D}_{S}_{F}_{P}{suffix}{solve_policy.suffix}"
    solve.__qualname__ = solve.__name__
    return wp.kernel(enable_backward=False, module="unique")(solve)


def _contact_triplet_code(
    M: int,
    S: int,
    Q: int,
    contact_type: int,
    friction_type: int,
    *,
    linked: bool = False,
    active: bool = True,
    solve_policy: SolvePolicy = EXACT_POLICY,
) -> tuple[str, str]:
    """Native additions for one-call metrics and ordered register-held updates.

    ``linked`` admits ring-linked patch normals: the fused radius pools the ring
    from the new normal on, in the order the ordinary pair update pools it, and
    the normal and the pair bound their changes as their ordinary updates do,
    so the early exit matches under a relative change tolerance too. Unlinked
    transactions bound every change by the new normal.
    ``active`` fuses selected rows of the active-row scan; otherwise the
    transaction runs inside the full ordered sweep. ``solve_policy`` compiles
    the transaction's change vote.
    """
    pairs = M // 3
    cache_size = min(_SPARSE_CONTACT_METRIC_CACHE_SIZE, pairs)
    entry_guard = "!select_active || " if active else ""
    linked_guard = "" if linked else "\n            || world_row_parent.data[offset + normal] >= 0"
    selected_guard = "\n                && ((selected_words[(row + 1) / 32] >> ((row + 1) & 31)) & 1u) != 0u"
    selected_guard = selected_guard if active else ""
    ring_load, load = "", "new_normal"
    if linked:
        ring_load = """
                float load = new_normal;
                for (int next = world_row_parent.data[offset + row]; next >= 0 && next != row;
                     next = world_row_parent.data[offset + next])
                    load += impulse[next];"""
        load = "load"
    tangent_state, tangent_keep, change_test = _triplet_change_vote(
        contact_type, friction_type, linked=linked, solve_policy=solve_policy
    )
    setup = f"""
    __shared__ float metric_storage[{2 * cache_size * 5}];
    __shared__ signed char metric_map_storage[{2 * pairs}];
    __shared__ int metric_count_storage[2];
    float* metrics = metric_storage + slot * {cache_size * 5};
    signed char* metric_map = metric_map_storage + slot * {pairs};
    int* metric_count = metric_count_storage + slot;
    for (int id = lane; id < {pairs}; id += 32) metric_map[id] = -1;
    if (lane == 0) *metric_count = 0;
    __syncwarp(MASK);
    const auto contact_metric_slot = [&](int normal) {{
        if ({entry_guard}normal < 0 || normal + 2 >= count
            || type[normal] != {contact_type}
            || type[normal + 1] != {friction_type} || type[normal + 2] != {friction_type}{linked_guard}
            || world_row_parent.data[offset + normal + 1] != normal
            || world_row_parent.data[offset + normal + 2] != normal
            || support.row_class.data[offset + normal + 1]
                != support.row_class.data[offset + normal + 2]) return -1;
        // Full, disjoint three-row records make normal/3 injective even when
        // another row kind precedes a contact or a triplet crosses a mask word.
        return normal / 3;
    }};
    const auto friction_pair_cached = [&](int id, float a, float c, float d,
        float r0, float r1, float old0, float old1, float radius, float relaxation) {{
        FrictionPairMetric metric;
        const int cached = id >= 0 ? metric_map[id] : -1;
        if (cached >= 0) {{
            metric = FrictionPairMetric{{metrics[cached * 5], metrics[cached * 5 + 1], metrics[cached * 5 + 2],
                metrics[cached * 5 + 3], metrics[cached * 5 + 4]}};
        }} else {{
            metric = friction_pair_metric(a, c, d);
            if (id >= 0 && lane == 0 && *metric_count < {cache_size}) {{
                // One warp owns this world's ordered visits. A full cache
                // leaves the map unset, retaining exact recomputation.
                const int entry = *metric_count;
                metrics[entry * 5] = metric.scale;
                metrics[entry * 5 + 1] = metric.largest;
                metrics[entry * 5 + 2] = metric.smallest;
                metrics[entry * 5 + 3] = metric.vx;
                metrics[entry * 5 + 4] = metric.vy;
                metric_map[id] = static_cast<signed char>(entry);
                *metric_count = entry + 1;
            }}
            __syncwarp(MASK);
        }}
        return friction_pair_from_metric(metric, r0, r1, old0, old1, radius, relaxation);
    }};
"""
    update = f"""
            if (row == consumed_tangent) continue;
            const int triplet_slot = contact_metric_slot(row);
            const bool fuse = triplet_slot >= 0 && global_iteration >= friction_start_iteration
                && isfinite(diagonal[row]) && diagonal[row] > 0.0f
                && support.row_class.data[offset + row] == support.row_class.data[offset + row + 1]{selected_guard};
            if (fuse) {{
                const int tangent = row + 1, sibling = row + 2;
                int indices[{Q}];
                float velocities[{Q}], normal_responses[{Q}];
                float tangent_responses[{Q}], sibling_responses[{Q}];
                float normal_dot = 0.0f;
                #pragma unroll
                for (int q = 0; q < {Q}; ++q) {{
                    const int entry = lane + q * 32;
                    const int index = entry < {S}
                        ? support.class_dof.data[support.row_class.data[offset + row] * {S} + entry] : -1;
                    indices[q] = index;
                    normal_responses[q] = index >= 0 ? row_factor.data[row_base + row * {S} + entry] : 0.0f;
                    velocities[q] = index >= 0 ? delta[index] : 0.0f;
                    if (index >= 0) normal_dot += normal_responses[q] * velocities[q];
                }}
                for (int shift = 16; shift > 0; shift >>= 1)
                    normal_dot += __shfl_down_sync(MASK, normal_dot, shift, 32);
                const float normal_residual = __shfl_sync(MASK, normal_dot, 0, 32) + rhs[row];
                const float old_normal = impulse[row];
                const float new_normal = fmaxf(old_normal - omega * normal_residual / diagonal[row], 0.0f);
                const float normal_change = new_normal - old_normal;{ring_load}
                const float radius = fmaxf(world_row_mu.data[offset + tangent] * {load}, 0.0f);
                const float old_tangent = impulse[tangent], old_sibling = impulse[sibling];
                {tangent_state}
                if (lane == 0) impulse[row] = new_normal;
                #pragma unroll
                for (int q = 0; q < {Q}; ++q)
                    if (indices[q] >= 0 && normal_change != 0.0f)
                        velocities[q] += normal_responses[q] * normal_change;
                if (!(radius == 0.0f && old_tangent == 0.0f && old_sibling == 0.0f)) {{
                    const bool ready = metric_map[triplet_slot] >= 0;
                    float dot = 0.0f, sibling_dot = 0.0f, cross = 0.0f;
                    #pragma unroll
                    for (int q = 0; q < {Q}; ++q) {{
                        const int entry = lane + q * 32;
                        const int index = indices[q];
                        const float value = index >= 0 ? row_factor.data[row_base + tangent * {S} + entry] : 0.0f;
                        const float sibling_value = index >= 0
                            ? row_factor.data[row_base + sibling * {S} + entry] : 0.0f;
                        tangent_responses[q] = value;
                        sibling_responses[q] = sibling_value;
                        if (index >= 0) {{
                            // Use the actual post-normal coordinate, preserving
                            // the original rounding and sibling-then-first order.
                            dot += value * velocities[q];
                            sibling_dot += sibling_value * velocities[q];
                            if (!ready) cross += value * sibling_value;
                        }}
                    }}
                    for (int shift = 16; shift > 0; shift >>= 1) {{
                        dot += __shfl_down_sync(MASK, dot, shift, 32);
                        sibling_dot += __shfl_down_sync(MASK, sibling_dot, shift, 32);
                        if (!ready) cross += __shfl_down_sync(MASK, cross, shift, 32);
                    }}
                    const float residual = __shfl_sync(MASK, dot, 0, 32) + rhs[tangent];
                    float2 pair = make_float2(0.0f, 0.0f);
                    if (radius > 0.0f) {{
                        const float sibling_residual = __shfl_sync(MASK, sibling_dot, 0, 32) + rhs[sibling];
                        pair = friction_pair_cached(triplet_slot, diagonal[tangent], __shfl_sync(MASK, cross, 0, 32),
                            diagonal[sibling], residual, sibling_residual, old_tangent, old_sibling, radius, omega);
                    }}
                    const float magnitude = sqrtf(pair.x * pair.x + pair.y * pair.y);
                    const float scale = magnitude > radius ? radius / magnitude : 1.0f;
                    const float new_tangent = pair.x * scale, new_sibling = pair.y * scale;
                    tangent_change = new_tangent - old_tangent;
                    sibling_change = new_sibling - old_sibling;{tangent_keep}
                    if (lane == 0) {{ impulse[tangent] = new_tangent; impulse[sibling] = new_sibling; }}
                    #pragma unroll
                    for (int q = 0; q < {Q}; ++q) {{
                        if (indices[q] >= 0) {{
                            if (sibling_change != 0.0f) velocities[q] += sibling_responses[q] * sibling_change;
                            if (tangent_change != 0.0f) velocities[q] += tangent_responses[q] * tangent_change;
                        }}
                    }}
                }}
                if (normal_change != 0.0f || tangent_change != 0.0f || sibling_change != 0.0f) {{
{change_test}
                    #pragma unroll
                    for (int q = 0; q < {Q}; ++q)
                        if (indices[q] >= 0) delta[indices[q]] = velocities[q];
                }}
                consumed_tangent = tangent;
                __syncwarp(MASK);
                continue;
            }}
"""
    return setup, update


def _triplet_change_vote(
    contact_type: int, friction_type: int, *, linked: bool, solve_policy: SolvePolicy
) -> tuple[str, str, str]:
    """Return a triplet transaction's tangent state, its first-tangent capture and its change vote.

    Unlinked transactions bound every change by the new normal. Linked ones bound
    the normal's change by its new impulse and the pair's by its new first
    tangent, as their ordinary updates do, so the early exit matches under a
    relative change tolerance too.
    """
    tangent_state = "float tangent_change = 0.0f, sibling_change = 0.0f;"
    tangent_keep = ""
    change_test = """                    if (!(fabsf(normal_change) <= change_tolerance + relative_change_tolerance * fabsf(new_normal)
                          && fabsf(tangent_change) <= change_tolerance + relative_change_tolerance * fabsf(new_normal)
                          && fabsf(sibling_change) <= change_tolerance + relative_change_tolerance * fabsf(new_normal))) changed = 1;"""
    normal_slack = tangent_slack = "change_tolerance + relative_change_tolerance * fabsf(new_normal)"
    if linked:
        tangent_state = "float tangent_change = 0.0f, sibling_change = 0.0f, tangent_value = 0.0f;"
        tangent_keep = "\n                    tangent_value = new_tangent;"
        # Rows r and r + 1 bound their changes by their own new impulses, the pair by its first tangent.
        change_test = """                    if (normal_change != 0.0f
                        && !(fabsf(normal_change) <= change_tolerance + relative_change_tolerance * fabsf(new_normal)))
                        changed = 1;
                    if ((tangent_change != 0.0f || sibling_change != 0.0f)
                        && !(fabsf(tangent_change) <= change_tolerance + relative_change_tolerance * fabsf(tangent_value)
                             && fabsf(sibling_change) <= change_tolerance + relative_change_tolerance * fabsf(tangent_value)))
                        changed = 1;"""
        tangent_slack = "change_tolerance + relative_change_tolerance * fabsf(tangent_value)"
    change_test = vote(
        solve_policy,
        change_test,
        flag="changed",
        terms=(
            ("normal_change", "diagonal[row]", str(contact_type)),
            ("tangent_change", "diagonal[tangent]", str(friction_type)),
            ("sibling_change", "diagonal[sibling]", str(friction_type)),
        ),
        slack=(normal_slack, tangent_slack, tangent_slack),
    )
    return tangent_state, tangent_keep, change_test


def _single_stage_triplet_code(
    M: int,
    S: int,
    Q: int,
    contact_type: int,
    friction_type: int,
    *,
    linked: bool = False,
    active: bool = True,
    solve_policy: SolvePolicy = EXACT_POLICY,
) -> tuple[str, str]:
    """Native additions for the single-stage triplet transaction under the paired friction law.

    A contact normal followed by its two tangents in shared metadata issues one
    load group: the three rows' guards, coordinates and responses, the triplet's
    record and its friction coefficient, with the shared impulses, right-hand
    sides and tangent diagonals of the visit. One interleaved reduction gives the
    three residuals. The normal takes its relaxed step with the record's
    ``1/W_nn``; the pair then solves at the new load, its residuals moved by the
    normal change through the record's ``W_nt`` and its eigenbasis read from the
    record. Lane 0 writes the three impulses after a warp barrier, since every
    lane read them. The ordinary update keeps any row whose global guards decline.

    With ``linked`` a pass at sweep entry stores each anchor's other ring
    members, at most three, behind its record in the order the pair update
    pools them, and a visit sums their shared impulses; a longer ring is marked
    ``-1`` and keeps the walk. ``active`` fuses selected rows of the active-row
    scan; otherwise the transaction runs inside the full ordered sweep.
    ``solve_policy`` compiles the change vote.
    """
    width = CONTACT_RECORD_FLOATS + (CONTACT_RING_FLOATS if linked else 0)
    record = f"contact_records.data + (static_cast<size_t>(world) * contact_records.shape[1] + {{row}} / 3) * {width}"
    entry_guard = "select_active && " if active else ""
    selected_guard = "\n                    & (((selected_words[(row + 1) / 32] >> ((row + 1) & 31)) & 1u) != 0u)"
    selected_guard = selected_guard if active else ""
    tangent_state, tangent_keep, change_test = _triplet_change_vote(
        contact_type, friction_type, linked=linked, solve_policy=solve_policy
    )
    # The shared vote sits one block deeper here.
    tangent_keep = tangent_keep.replace("\n", "\n    ")
    change_test = textwrap.indent(change_test, "    ")
    setup = ""
    normal_parent, unlinked_guard, ring_read = "", " && normal_parent < 0", ""
    load = "new_normal"
    if linked:
        setup = f"""
    // Each anchor's other ring members, at most three, in the order the pair update pools
    // them; -1 marks a longer ring, which keeps the walk. Visits read them with the record.
    for (int normal = lane; normal + 2 < count; normal += 32) {{
        if (type[normal] != {contact_type} || type[normal + 1] != {friction_type}
            || type[normal + 2] != {friction_type}) continue;
        const int first = world_row_parent.data[offset + normal];
        int second = -1, third = -1, others = 0;
        if (first >= 0 && first != normal) {{
            others = 1;
            second = world_row_parent.data[offset + first];
            if (second >= 0 && second != normal) {{
                others = 2;
                third = world_row_parent.data[offset + second];
                if (third >= 0 && third != normal) {{
                    const int fourth = world_row_parent.data[offset + third];
                    others = fourth >= 0 && fourth != normal ? -1 : 3;
                }}
            }}
        }}
        *reinterpret_cast<int4*>({record.format(row="normal")} + {CONTACT_RECORD_FLOATS})
            = make_int4(first, second, third, others);
    }}
    __syncwarp(MASK);
"""
        normal_parent, unlinked_guard = "", ""
        ring_read = f"""
                    const int4 ring = *reinterpret_cast<const int4*>(record + {CONTACT_RECORD_FLOATS});"""
        load = "load"
    else:
        normal_parent = """
                const int normal_parent = world_row_parent.data[offset + row];"""
    ring_load = ""
    if linked:
        ring_load = """
                    // The ring pools as the pair update walks it, from the new normal on.
                    float load = new_normal;
                    if (ring.w >= 0) {
                        if (ring.w > 0) load += impulse[ring.x];
                        if (ring.w > 1) load += impulse[ring.y];
                        if (ring.w > 2) load += impulse[ring.z];
                    } else {
                        for (int next = world_row_parent.data[offset + row]; next >= 0 && next != row;
                             next = world_row_parent.data[offset + next])
                            load += impulse[next];
                    }"""
    update = f"""
            if (row == consumed_tangent) continue;
            // Shared metadata first, read together: a contact normal with its two tangents.
            if ({entry_guard}row + 2 < count && global_iteration >= friction_start_iteration
                && ((type[row] == {contact_type}) & (type[row + 1] == {friction_type}) & (type[row + 2] == {friction_type})
                    & (isfinite(diagonal[row]) != 0) & (diagonal[row] > 0.0f){selected_guard})) {{
                const int tangent = row + 1, sibling = row + 2;
                // The guards load together; the rows, the record and the coordinate indices
                // then load together.
                const int row_class = support.row_class.data[offset + row];
                const int tangent_class = support.row_class.data[offset + tangent];
                const int sibling_class = support.row_class.data[offset + sibling];
                const int tangent_parent = world_row_parent.data[offset + tangent];
                const int sibling_parent = world_row_parent.data[offset + sibling];{normal_parent}
                if (tangent_parent == row && sibling_parent == row && tangent_class == sibling_class
                    && row_class == tangent_class{unlinked_guard}) {{
                    const float* record = {record.format(row="row")};
                    const float4 coupling = *reinterpret_cast<const float4*>(record);
                    const float4 basis = *reinterpret_cast<const float4*>(record + 4);{ring_read}
                    const float mu = world_row_mu.data[offset + tangent];
                    int indices[{Q}];
                    float velocities[{Q}], normal_responses[{Q}];
                    float tangent_responses[{Q}], sibling_responses[{Q}];
                    float normal_dot = 0.0f, tangent_dot = 0.0f, sibling_dot = 0.0f;
                    #pragma unroll
                    for (int q = 0; q < {Q}; ++q) {{
                        const int entry = lane + q * 32;
                        const bool stored = entry < {S};
                        const float normal_entry = stored ? row_factor.data[row_base + row * {S} + entry] : 0.0f;
                        const float tangent_entry = stored ? row_factor.data[row_base + tangent * {S} + entry] : 0.0f;
                        const float sibling_entry = stored ? row_factor.data[row_base + sibling * {S} + entry] : 0.0f;
                        const int index = stored ? support.class_dof.data[row_class * {S} + entry] : -1;
                        indices[q] = index;
                        normal_responses[q] = index >= 0 ? normal_entry : 0.0f;
                        tangent_responses[q] = index >= 0 ? tangent_entry : 0.0f;
                        sibling_responses[q] = index >= 0 ? sibling_entry : 0.0f;
                        velocities[q] = index >= 0 ? delta[index] : 0.0f;
                        if (index >= 0) {{
                            normal_dot += normal_responses[q] * velocities[q];
                            tangent_dot += tangent_responses[q] * velocities[q];
                            sibling_dot += sibling_responses[q] * velocities[q];
                        }}
                    }}
                    // The shared values of the visit load with the rows, off the reduction's chain.
                    const float old_normal = impulse[row];
                    const float old_tangent = impulse[tangent], old_sibling = impulse[sibling];
                    const float normal_rhs = rhs[row], tangent_rhs = rhs[tangent], sibling_rhs = rhs[sibling];
                    const float tangent_scale = fmaxf(fmaxf(diagonal[tangent], diagonal[sibling]), 1.0e-20f);
                    for (int shift = 16; shift > 0; shift >>= 1) {{
                        normal_dot += __shfl_down_sync(MASK, normal_dot, shift, 32);
                        tangent_dot += __shfl_down_sync(MASK, tangent_dot, shift, 32);
                        sibling_dot += __shfl_down_sync(MASK, sibling_dot, shift, 32);
                    }}
                    const float normal_residual = __shfl_sync(MASK, normal_dot, 0, 32) + normal_rhs;
                    const float tangent_residual = __shfl_sync(MASK, tangent_dot, 0, 32) + tangent_rhs;
                    const float sibling_residual = __shfl_sync(MASK, sibling_dot, 0, 32) + sibling_rhs;
                    const float new_normal = fmaxf(old_normal - omega * normal_residual * coupling.w, 0.0f);
                    const float normal_change = new_normal - old_normal;{ring_load}
                    const float radius = fmaxf(mu * {load}, 0.0f);
                    {tangent_state}
                    // Shuffles order execution, not memory: every lane has read the three
                    // impulses before lane 0 writes them.
                    __syncwarp(MASK);
                    if (lane == 0) impulse[row] = new_normal;
                    if (!(radius == 0.0f && old_tangent == 0.0f && old_sibling == 0.0f)) {{
                        float2 pair = make_float2(0.0f, 0.0f);
                        if (radius > 0.0f) {{
                            // W_nt moves the tangent residuals by the normal change. The eigenbasis
                            // is relative to the larger tangent diagonal, as the pair law scales it.
                            const FrictionPairMetric metric{{tangent_scale, basis.x, basis.y, basis.z, basis.w}};
                            pair = friction_pair_from_metric(metric, tangent_residual + coupling.x * normal_change,
                                sibling_residual + coupling.y * normal_change, old_tangent, old_sibling, radius, omega);
                        }}
                        const float magnitude = sqrtf(pair.x * pair.x + pair.y * pair.y);
                        const float projection = magnitude > radius ? radius / magnitude : 1.0f;
                        const float new_tangent = pair.x * projection, new_sibling = pair.y * projection;
                        tangent_change = new_tangent - old_tangent;
                        sibling_change = new_sibling - old_sibling;{tangent_keep}
                        if (lane == 0) {{ impulse[tangent] = new_tangent; impulse[sibling] = new_sibling; }}
                    }}
                    if (normal_change != 0.0f || tangent_change != 0.0f || sibling_change != 0.0f) {{
{change_test}
                        // The staged transaction's order: normal, then sibling, then first tangent.
                        #pragma unroll
                        for (int q = 0; q < {Q}; ++q) {{
                            if (indices[q] >= 0) {{
                                float value = velocities[q];
                                if (normal_change != 0.0f) value += normal_responses[q] * normal_change;
                                if (sibling_change != 0.0f) value += sibling_responses[q] * sibling_change;
                                if (tangent_change != 0.0f) value += tangent_responses[q] * tangent_change;
                                delta[indices[q]] = value;
                            }}
                        }}
                    }}
                    consumed_tangent = tangent;
                    __syncwarp(MASK);
                    continue;
                }}
            }}
"""
    return setup, update
