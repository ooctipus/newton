# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The free-body rows' headers and addressing, the bilateral closure and the pairs' qualification.

The rows are the walk's (:mod:`.walk_rows`: the one writer walks every row's seeds into its whitened form ``Z``). This
module holds what is built around them:

- the free-body rows (:class:`.response_operator.FreeBodyZ`, rows over at most two free bodies): each row's header
  (:func:`free_body_row_header`: ``A_ii``, ``rhs`` and the row's weight from its incident velocity) and the packed row
  word (:func:`free_body_meta_kernel`, which :meth:`.FreeBodyZ.consume` decodes: its sides' coordinates, ``1 /
  A_ii``);
- the bilateral closure (:func:`close_block`, :func:`close_row`, :func:`seed_blocks_kernel`), the Schur transform the
  one writer's block phase applies to fold a world's bilateral block into the responses of its other rows;
- the paired factor rows' qualification (:func:`paired_components_kernel`).

The Warp kernels call Warp functions, one expression tree on both devices; the native kernels call the operators'
emitters.
"""

from __future__ import annotations

from functools import cache

import warp as wp

from . import kernel_template
from .dense_rows import MfRows
from .device_preludes import CPU_MATH, dual_native, lane_group
from .kernel_template import Dispatch, KernelTemplate
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_SPEED,
)
from .paired_components import FREE_BODY_DOFS, PAIRED_COMPONENT_THREADS, PairedComponentData
from .projection_laws import BILATERAL_PIVOT
from .response_operator import FreeBodyZ
from .row_laws import (
    LINK_ROWS,
    contact_capped_beta,
    contact_depenetration_cap,
    contact_restitution_rhs,
    contact_rhs,
    row_rhs,
)
from .row_metadata import _validate_mf_metadata_encoding
from .solver_fields import Preelim

# ---------------------------------------------------------------------------------------------------------------------
# Free-body rows (FreeBodyZ)
# ---------------------------------------------------------------------------------------------------------------------
#
# A free-body row acts on at most two free rigid bodies, ``a`` and ``b`` (a negative body for none), through six
# entries each. A free body is a one-node tree; the one writer walks each side's seed into ``Z = L^-1 J^T`` (its walk's
# factor, armature included), so ``A_ii = cfm + |Z_a|^2 + |Z_b|^2`` and the sweeps hold the change of the bodies'
# velocities in factor coordinates ``u = L^T dv``. A body's six velocity coordinates start at its articulation's first
# coordinate, in the addressing the mapping reads: global (``articulation_dof_start``) or world
# (``articulation_world_dof_offset``).


@wp.func
def free_body_row_rhs(
    body_a: int,
    body_b: int,
    row_type: int,
    phi: float,
    target: float,
    restitution: float,
    incident: float,
    max_depenetration_velocity: wp.array[float],
    beta: float,
    contact_w: float,
    dt: float,
    speculative_scale: float,
    restitution_threshold: float,
):
    """A free-body row's position-pass ``rhs`` and regularization weight, by the row laws (:mod:`.row_laws`).

    A contact takes :func:`.row_laws.contact_rhs` at its bodies' capped beta, and, when restitutive, the
    rebound of :func:`.row_laws.contact_restitution_rhs` from its incident velocity ``J v_hat`` less its target
    (weight 1); a friction row its anchor law (the row builder folded the anchor gain into ``phi``); a speed row
    ``phi - target`` (its limit); any other row ``-target``. The sweeps recompute ``J v``: ``rhs`` holds the bias
    only.
    """
    inv_dt = 1.0 / dt
    rhs = -target
    row_w = float(1.0)
    if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
        cap = contact_depenetration_cap(body_a, body_b, max_depenetration_velocity)
        rhs, row_w = contact_rhs(
            phi, contact_capped_beta(phi, beta, cap, dt), target, inv_dt, 1.0, speculative_scale, contact_w
        )
        if restitution > 0.0:
            fires, bounce = contact_restitution_rhs(
                phi, incident - target, target, restitution, dt, restitution_threshold
            )
            if fires:
                rhs = bounce
                # An impact is impulsive, not a spring: keep the rebound exact.
                row_w = 1.0
    elif row_type == PGS_CONSTRAINT_TYPE_FRICTION:
        rhs, _w = row_rhs(row_type, phi, 1.0, target, inv_dt, 1.0, 0.0, 0.0, 1.0)
    elif row_type == PGS_CONSTRAINT_TYPE_SPEED:
        rhs = phi - target
    return rhs, row_w


@wp.func
def free_body_row_header(
    mf: MfRows,
    line: int,
    i: int,
    body_a: int,
    body_b: int,
    incident: float,
    diagonal: float,
    has_target_velocity: int,
    rigid_body_max_depenetration_velocity: wp.array[float],
    pgs_beta: float,
    contact_w: float,
    dt: float,
    contact_speculative_scale: float,
    restitution_velocity_threshold: float,
    weighted: int,
    material: int,
    weights: wp.array2d[float],
):
    """Free-body row ``i``'s header (its responding bodies ``body_a``, ``body_b``) from its incident velocity ``J v_hat``
    and its diagonal ``A_ii``: both stored, then its ``rhs`` (:func:`free_body_row_rhs`). ``weighted`` rows store their relative regularization
    weight (in ``weights``: the row store's free-body region); a material contact (``material``: its writer stored its
    rhs and compliance ``eps``) keeps its rhs."""
    row_type = mf.row_type[line, i]
    mf.incident[line, i] = incident
    mf.diag[line, i] = diagonal
    if material != 0 and row_type == PGS_CONSTRAINT_TYPE_CONTACT:
        if mf.eps[line, i] > 0.0:
            return
    target = float(0.0)
    if has_target_velocity != 0:
        target = mf.target[line, i]
    rhs, row_w = free_body_row_rhs(
        body_a,
        body_b,
        row_type,
        mf.phi[line, i],
        target,
        mf.restitution[line, i],
        incident,
        rigid_body_max_depenetration_velocity,
        pgs_beta,
        contact_w,
        dt,
        contact_speculative_scale,
        restitution_velocity_threshold,
    )
    mf.rhs[line, i] = rhs
    if weighted != 0:
        weights[line, i] = row_w


@cache
def free_body_meta_kernel(mf_max_constraints: int, pitch: int, base: int, device_arch: str) -> wp.Kernel:
    """Pack each world's free-body rows into the 128-bit word the matrix-free sweeps load (:meth:`.FreeBodyZ.pack`).

    A warp per world (``wp.launch_tiled`` over the worlds, 32 threads; one thread on the CPU), a lane per row; the
    word holds the rows' sides' first world coordinates (``world_dof_offset`` of their articulations), the inverse of
    their stored diagonal ``A_ii`` (the step's form of the law, cached in the header), right-hand side (the law's plus
    the row's incident velocity: its residual at ``du = 0``), type and parent. The records are the row store's
    (``pitch`` rows a line, the free bodies' from ``base``). ``device_arch`` keys the variant.
    """
    _validate_mf_metadata_encoding(mf_max_constraints)
    word = FreeBodyZ().pack("packed", "da", "db", "inverse", "rhs_val", "rt", "par", "        ")
    body = f"""    int m_mf = mf_constraint_count.data[world];
    const int mf_line = first.data[world];
    __builtin_assume(mf_line >= 0);
    int off_meta = mf_line * {mf_max_constraints * 4};
    const int off_body = mf_line * {pitch} + {base};

    for (int i = LANE; i < m_mf; i += STRIDE) {{
        const wp::vec2i sides = row_sides.data[off_body + i];
        int da = sides.c[0] >= 0 ? world_dof_offset.data[sides.c[0]] : -1;
        int db = sides.c[1] >= 0 ? world_dof_offset.data[sides.c[1]] : -1;
        const float A = diag.data[off_body + i];
        float inverse = 0.0f;
        if (A > 0.0f) inverse = 1.0f / A;
        float rhs_val = rhs.data[off_body + i] + row_incident.data[off_body + i];
        int rt = row_type.data[off_body + i];
        int par = row_parent.data[off_body + i];

{word}
        *reinterpret_cast<int4*>(&mf_meta.data[off_meta + i * 4]) = packed;
    }}
"""
    return KernelTemplate(
        f"free_body_meta_{mf_max_constraints}_{pitch}",
        kernel_template.params(
            ("mf_constraint_count", wp.array[int]),
            ("first", wp.array[int]),
            ("world_dof_offset", wp.array[int]),
            ("row_sides", wp.array2d[wp.vec2i]),
            ("diag", wp.array2d[float]),
            ("rhs", wp.array2d[float]),
            ("row_incident", wp.array2d[float]),
            ("row_type", wp.array2d[int]),
            ("row_parent", wp.array2d[int]),
            ("mf_meta", wp.array2d[int]),
        ),
        dual_native(body, preludes=(lane_group(32), CPU_MATH)),
        dispatch=kernel_template.WORLDS,
        composed=True,
        native_name="free_body_meta_native",
    ).kernel()


# ---------------------------------------------------------------------------------------------------------------------
# Bilateral elimination: the exact Schur transform of the packed rows
# ---------------------------------------------------------------------------------------------------------------------
#
# The bilateral rows of a linked component of trees (:mod:`.body_cut`), its mimics and its links, form a block ``B``:
# the block primitive's second phase, the closure, after the slips' release (FL3_design §1.4), in the one writer's
# block phase (:mod:`.walk_rows`). On the packed rows (:mod:`.packed_rows`, ``Z_i = L^-1 P J_i^T`` on each side), wide
# over the world's coordinates, the transform makes the block's rows orthonormal in place by modified Gram-Schmidt, the right-hand sides following:
# ``Q^T Q = I``, a linear combination of the same constraints. A row whose remainder falls to
# :data:`.projection_laws.BILATERAL_PIVOT` of its own norm depends on the rows before it (a planar loop's out-of-plane
# axis, a mimic a closure fixes): it leaves the block, inert. With ``Pi = I - Q Q^T`` the exact projector onto the
# closures:
#
# - correct: every other row touching the block becomes ``Pi Z_i = Z_i - Q Q^T Z_i``, which reaches every tree of the
#   block, its diagonal ``|Pi Z_i|^2 + cfm`` (the closed-loop effective mass); its right-hand side takes back
#   ``(Q^T Z_i)^T r_Q``, ``r_Q = J_Q v_hat + b_Q`` being the block rows' residual at zero change;
# - project: the velocity change starts at ``du = -Q r_Q``, the block rows' impulses at ``-r_Q`` (the sweeps' seed,
#   :func:`.sweep.impulse_seed`).
#
# Then ``Q^T (u_hat + du) = -b_Q`` and a sweep's impulse, ``du += Pi Z_i delta``, keeps it. A corrected row's residual
# at zero change is ``(Pi Z_i)^T u_hat`` and its right-hand side, which the transform folds: its ``J_i v_hat + b_i``
# less ``(Q^T Z_i)^T r_Q``. ``Pi`` is idempotent, so the
# contacts solve on the closed-loop Delassus ``Z^T Pi Z`` exactly (no compliance). The block's rows stay in the sweep:
# they hold in exact arithmetic, and their visits absorb the corrected rows' float32 leakage, without which collinear
# contact rows of a jammed loop drift in their null space and never meet the exit law. Being orthonormal, each absorbs
# its share in one visit, where rows that couple would pass it among themselves at their Gauss-Seidel rate.


@wp.func
def wide_product(Z: wp.array3d[float], world: int, row_a: int, row_b: int, entries: int) -> float:
    """``Z_a . Z_b`` over a world's ``entries`` wide entries, in order."""
    acc = float(0.0)
    for e in range(entries):
        acc += Z[world, row_a, e] * Z[world, row_b, e]
    return acc


@wp.func
def subtract_row(Z: wp.array3d[float], world: int, row: int, scale: float, other: int, entries: int):
    """``Z_row -= scale Z_other`` over a world's ``entries`` wide entries."""
    for e in range(entries):
        Z[world, row, e] -= scale * Z[world, other, e]


@wp.func
def scale_row(Z: wp.array3d[float], world: int, row: int, scale: float, entries: int):
    """``Z_row *= scale`` over a world's ``entries`` wide entries."""
    for e in range(entries):
        Z[world, row, e] *= scale


@wp.func
def bilateral_rows(
    block: int,
    base: int,
    preelim: Preelim,
    mimic_slot: wp.array[int],
    link_slot: wp.array[int],
) -> int:
    """Select the block's reserved rows, in its constraints' order, into ``preelim.slots[base:]`` and return their
    count."""
    n = int(0)
    for i in range(preelim.start[block], preelim.start[block + 1]):
        entry = preelim.entry[i]
        if entry >= 0:
            if mimic_slot[entry] >= 0:
                preelim.slots[base + n] = mimic_slot[entry]
                n += 1
        elif link_slot[-entry - 1] >= 0:
            for a in range(LINK_ROWS):
                preelim.slots[base + n] = link_slot[-entry - 1] + a
                n += 1
    return n


@wp.func
def close_block(
    pe: int,
    capacity: int,
    line: int,
    preelim: Preelim,
    mimic_slot: wp.array[int],
    link_slot: wp.array[int],
    world_row_cfm: wp.array2d[float],
    packed_Z: wp.array3d[float],
    world_rhs: wp.array2d[float],
    world_incident: wp.array2d[float],
    world_diag: wp.array2d[float],
):
    """Block ``pe``'s closure (the one writer's block phase, once its rows are written): modified Gram-Schmidt over the
    block's rows, in its constraints' order, in place on line ``line``. Each row takes out the rows kept before it, then
    joins them unit, or leaves the block inert where nothing of it remains; its right-hand side follows, its incident
    velocity joined to it first, and its diagonal is ``|Q_i|^2 + cfm``. ``preelim.slots`` ends with the kept rows
    (``preelim.nB`` of them)."""
    base = pe * capacity
    entries = packed_Z.shape[2]
    m = bilateral_rows(pe, base, preelim, mimic_slot, link_slot)
    kept = int(0)
    for p in range(m):
        row = preelim.slots[base + p]
        # The row's incident velocity joins its right-hand side: both follow the row through the transform.
        world_rhs[line, row] += world_incident[line, row]
        world_incident[line, row] = 0.0
        own = wide_product(packed_Z, line, row, row, entries)
        for q in range(kept):
            other = preelim.slots[base + q]
            along = wide_product(packed_Z, line, other, row, entries)
            subtract_row(packed_Z, line, row, along, other, entries)
            world_rhs[line, row] -= along * world_rhs[line, other]
        rest = wide_product(packed_Z, line, row, row, entries)
        scale = float(0.0)
        if rest > BILATERAL_PIVOT * own:
            scale = 1.0 / wp.sqrt(rest)
            preelim.slots[base + kept] = row
            kept += 1
        scale_row(packed_Z, line, row, scale, entries)
        world_rhs[line, row] *= scale
        world_diag[line, row] = scale * scale * rest + world_row_cfm[line, row]
    for p in range(kept, capacity):
        preelim.slots[base + p] = -1
    preelim.nB[pe] = kept


@wp.func
def close_row(
    line: int,
    row: int,
    sides: wp.vec2i,
    capacity: int,
    preelim: Preelim,
    world_row_cfm: wp.array2d[float],
    packed_Z: wp.array3d[float],
    world_rhs: wp.array2d[float],
    world_diag: wp.array2d[float],
):
    """Correct row ``row`` (not a block row) on the blocks its sides touch, once they are closed: ``Pi Z_i = Z_i - Q
    Q^T Z_i``, its right-hand side taking back ``(Q^T Z_i) . r_Q``, its diagonal ``|Pi Z_i|^2 + cfm`` (the squares summed
    in entry order, the padding's zeros included). A row the block does not see (``Q^T Z_i`` exactly zero) keeps its
    entries. Blocks hold different trees, so their projectors commute."""
    entries = packed_Z.shape[2]
    touched = int(0)
    last = int(-1)
    for side in range(2):
        art = sides[side]
        if art < 0:
            continue
        pe = preelim.art_to_preelim[art]
        if pe < 0 or pe == last:
            continue
        last = pe
        touched = 1
        base = pe * capacity
        for p in range(preelim.nB[pe]):
            other = preelim.slots[base + p]
            along = wide_product(packed_Z, line, other, row, entries)
            if along != 0.0:
                subtract_row(packed_Z, line, row, along, other, entries)
                world_rhs[line, row] -= along * world_rhs[line, other]
    if touched != 0:
        world_diag[line, row] = wide_product(packed_Z, line, row, row, entries) + world_row_cfm[line, row]


@wp.kernel(enable_backward=False)
def seed_blocks_kernel(
    dense_first: wp.array[int],
    capacity: int,
    preelim: Preelim,
    world_rhs: wp.array2d[float],
    # outputs
    world_impulses: wp.array2d[float],
):
    """The closure's seed, after the warm start's gather: the kept rows' impulses ``-r_Q``, their residual at zero
    change (the sweeps' :func:`.sweep.impulse_seed` installs them), a thread per block."""
    pe = wp.tid()
    base = pe * capacity
    line = dense_first[preelim.world[pe]]
    for p in range(preelim.nB[pe]):
        slot = preelim.slots[base + p]
        world_impulses[line, slot] = -world_rhs[line, slot]


# ---------------------------------------------------------------------------------------------------------------------
# Paired factor rows (FactorRows)
# ---------------------------------------------------------------------------------------------------------------------
#
# A paired world holds one articulation (``P`` coordinates, its primary group) and one free body (``S = 6``, its
# secondary group), each a diagonal block of the leaf-first ``P H P^T = L L^T``. Its rows' factor rows
# ``z = L^-1 P j`` are its packed rows at their world coordinates (:mod:`.packed_rows`, ``A_ii = |z|^2 + cfm``).
#
# A paired world's qualification as independent components: every current row of the world is valid and touches
# only its own component (dense rows the articulation's coordinates, free-body rows the body's), its pair structure
# intact; a qualified world's selector turns negative (zero: no free-body rows, positive: the mixed fallback).
_PAIRED_COMPONENTS = r"""
#if !defined(__CUDA_ARCH__)
    if (logical_lane != 0) return;
#endif
    const int M = data.Y.shape[1];
    const int F = data.mf_meta.shape[1] / 4;
    const int pitch = data.row_type.shape[1];
    for (int world = worker; world < data.world_count; world += workers) {
        const int m = data.counts.data[world];
        const int mf = data.mf_counts.data[world];
        if (LANE == 0) data.selector.data[world] = mf == 0 ? 0 : wp::max(mf, 1);
        if (mf == 0) continue;
        if (m < 0 || m > M || mf < 1 || mf > F) continue;
        const int po = data.primary_offset.data[world];
        const int so = data.secondary_offset.data[world];
        const int art = data.primary_art.data[world];
        if (!((po == 0 && so == $PRIMARY) || (po == $SECONDARY && so == 0)) || art < 0
            || art >= data.dof_start.shape[0]) continue;
        // The articulation's factor is its walk: every record of its coordinates finite and bounded.
        int valid = 1;
        const int records = data.walk.shape[1];
        const float* walk = data.walk.data + (size_t)data.dof_start.data[art] * records;
        for (int k = LANE; k < $PRIMARY * records; k += STRIDE) {
            const float x = walk[k];
            valid &= wp::isfinite(x) && wp::abs(x) <= 1.0e15f;
        }
        for (int row = LANE; row < m; row += STRIDE) {
            const int r = data.first.data[world] * M + row;
            const int at = data.first.data[world] * pitch + row;
            const int type = data.row_type.data[at];
            const float diag = data.diag.data[at];
            const float mu = data.mu.data[at];
            valid &= (type == 0 || type == 2 || type == 3)
                && wp::isfinite(diag) && diag > 0.0f
                && wp::isfinite(mu) && mu >= 0.0f && wp::isfinite(data.rhs.data[at]);
            if (type == 2) {
                const int p = data.row_parent.data[at];
                if (p < 0 || p + 2 >= m || (row != p + 1 && row != p + 2)) valid = 0;
                else {
                    const int base = data.first.data[world] * pitch + p;
                    valid &= data.row_type.data[base] == 0
                        && data.row_type.data[base + 1] == 2 && data.row_type.data[base + 2] == 2
                        && data.row_parent.data[base + 1] == p && data.row_parent.data[base + 2] == p;
                }
            }
            for (int d = 0; d < $WORLD; ++d) {
                const float y = data.Y.data[r * $WORLD + d];
                valid &= wp::isfinite(y) && wp::abs(y) <= 1.0e15f;
                if (d >= so && d < so + $SECONDARY) valid &= y == 0.0f;
            }
        }
        for (int row = LANE; row < mf; row += STRIDE) {
            const int r = data.first.data[world] * F + row;
            const int packed = data.mf_meta.data[r * 4];
            const int a = packed >> 16, b = static_cast<int>(static_cast<short>(packed & 65535));
            const int tp = data.mf_meta.data[r * 4 + 3], type = tp & 65535;
            union { int bits; float value; } diag, rhs;
            diag.bits = data.mf_meta.data[r * 4 + 1];
            rhs.bits = data.mf_meta.data[r * 4 + 2];
            const float mu = data.mu.data[data.first.data[world] * pitch + data.body_base + row];
            valid &= (a == -1 || a == so) && (b == -1 || b == so) && (a == so || b == so)
                && (type == 0 || type == 2 || type == $SPEED)
                && wp::isfinite(diag.value) && diag.value > 0.0f && wp::isfinite(rhs.value)
                && wp::isfinite(mu) && mu >= 0.0f;
            if (type == 2) {
                const int encoded_parent = static_cast<unsigned int>(tp) >> 16;
                const int p = encoded_parent == 65535 ? -1 : encoded_parent;
                if (p < 0 || p + 2 >= mf || (row != p + 1 && row != p + 2)) valid = 0;
                else {
                    const int base = (data.first.data[world] * F + p) * 4 + 3;
                    const int t1 = data.mf_meta.data[base + 4], t2 = data.mf_meta.data[base + 8];
                    valid &= (data.mf_meta.data[base] & 65535) == 0
                        && (t1 & 65535) == 2 && (t2 & 65535) == 2
                        && static_cast<int>(static_cast<unsigned int>(t1) >> 16) == p
                        && static_cast<int>(static_cast<unsigned int>(t2) >> 16) == p;
                }
            }
            for (int d = 0; d < $SECONDARY; ++d) {
                valid &= wp::isfinite(data.mf_Z_a.data[r * $SECONDARY + d])
                    && wp::isfinite(data.mf_Z_b.data[r * $SECONDARY + d]);
            }
        }
        if (ALL(valid != 0) && LANE == 0) data.selector.data[world] = -1;
    }
"""


@cache
def paired_components_kernel(primary_size: int, secondary_size: int = FREE_BODY_DOFS) -> wp.Kernel:
    """Qualify each paired world's rows as independent components. A block of
    :data:`.paired_components.PAIRED_COMPONENT_THREADS` threads per worker strides over the worlds; the signed selector
    is the dispatch's private metadata."""
    if not 1 <= primary_size <= 32 or secondary_size != FREE_BODY_DOFS:
        raise ValueError("Paired components take an articulation of at most 32 coordinates and one free rigid body")
    P = int(primary_size)
    source = kernel_template.substitute(
        _PAIRED_COMPONENTS,
        PRIMARY=P,
        SECONDARY=secondary_size,
        WORLD=P + secondary_size,
        SPEED=int(PGS_CONSTRAINT_TYPE_SPEED),
    )
    return KernelTemplate(
        f"paired_components_{P}_{secondary_size}",
        (("workers", int), ("data", PairedComponentData)),
        source,
        # A lane group per worker: its worker index and the thread's lane in it.
        Dispatch("worker, lane = wp.tid()\n$CALL", ("worker", "lane"), (("worker", int), ("logical_lane", int))),
        preludes=(lane_group(PAIRED_COMPONENT_THREADS),),
        dual=True,
        native_name="paired_components_native",
    ).kernel()
