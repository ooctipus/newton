# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build the row responses of the free-body and factor-row representations, and publish their velocities.

A :class:`.response_operator.ResponseOperator` owns the arithmetic a sweep runs on a row's response. Its *build*
forms that response from the row's Jacobian and the mass operator. This module holds the builds of three
representations, and the publication of the one whose sweep leaves its velocity change outside the kernel:

- :class:`.response_operator.FreeBodyZ` (rows over at most two free bodies): each row whitened per body, ``Z =
  L^-1 J^T`` through the body's held factor, and ``1 / A_ii``, and the rows' body coordinates in each addressing a
  mapping reads (a world's local body slots, world and global coordinates, the packed row word);
- :class:`.response_operator.SparseFactor` (rows ``Z = L^-1 P J^T`` packed by support): the factor and its inverse
  that its predictor reads; its rows and their leave are the walk's (:mod:`.walk_rows`);
- :class:`.response_operator.FactorRows` (a pair's rows in factor coordinates, a lane per coordinate): the packed
  rows at their world coordinates (:mod:`.packed_rows`);

and the bilateral closure (:func:`close_block`, :func:`close_row`, :func:`seed_blocks_kernel`), the Schur transform the
one writer's block phase applies to fold a world's bilateral block into the responses of its other rows.

**API.** Each operator class returns its build from ``build()`` (``FreeBodyZ.build()``, ``SparseFactor.build()``,
``FactorRows.build()``; the sparse diagonal's rows are the walk's, :mod:`.walk_rows`): a frozen descriptor whose
members are the Warp kernels (or cached factories of kernel variants, or a Warp function) of each step of the build
and of the publication; the solver's launch sites call them directly. The arithmetic of a step is one function per representation and mapping, called by
every kernel that runs the step:

- the Warp kernels call Warp functions (``free_body_response``, ``packed_product``...),
  one expression tree on both devices;
- the native kernels call the operator's emitters (:meth:`.FreeBodyZ.pack` writes the row word that
  :meth:`.FreeBodyZ.consume` decodes; :func:`factor_rows` packs factor rows from their support, the layout
  :class:`.SparseFactor` reads; :meth:`.FactorRows.transposed` is the factor product of a lane per coordinate).

Where two forms of a step differ only by accident (the order of an operation, a guard), the step keeps each form,
chosen by a compile-time argument, and its documentation names the difference for their unification.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from functools import cache

import warp as wp

from . import kernel_template
from .dense_rows import MfRows
from .device_preludes import CPU_MATH, dual_native, lane_group
from .kernel_template import Dispatch, KernelTemplate
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    BodyFactors,
    factor_whiten,
    free_body_factor,
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
from .sparse_mass_matrix import _SparseMassMatrixIndices

# ---------------------------------------------------------------------------------------------------------------------
# Free-body rows (FreeBodyZ)
# ---------------------------------------------------------------------------------------------------------------------
#
# A free-body row acts on at most two free rigid bodies, ``a`` and ``b`` (a negative body for none), through six
# entries each. A free body is a one-node tree whose mass matrix ``H = L L^T`` the dynamics factor (its held factor,
# armature included: :class:`.kernels.BodyFactors`), and its rows are whitened: the writers store each side's ``J``
# and the build replaces it with ``Z = L^-1 J^T``, so ``A_ii = cfm + |Z_a|^2 + |Z_b|^2`` and the sweeps hold the
# change of the bodies' velocities in factor coordinates ``u = L^T dv``. A body's six velocity coordinates start at its
# articulation's first coordinate, in the addressing the mapping reads: global (``articulation_dof_start``) or world
# (``articulation_world_dof_offset``).


@wp.func
def body_coordinate(body: int, body_to_articulation: wp.array[int], first: wp.array[int]) -> int:
    """The first velocity coordinate of ``body`` in the addressing ``first`` (per articulation), -1 for no body."""
    if body < 0:
        return -1
    return first[body_to_articulation[body]]


@wp.func
def free_body_whiten(Z: wp.array3d[float], world: int, row: int, body: int, factors: BodyFactors, store: bool):
    """Whiten one side of row ``row``, ``Z = L^-1 J^T`` (in place when ``store``), and return its term ``|Z|^2`` of
    the diagonal (zero for no body)."""
    if body < 0:
        return 0.0
    jacobian = wp.spatial_vector(
        Z[world, row, 0], Z[world, row, 1], Z[world, row, 2], Z[world, row, 3], Z[world, row, 4], Z[world, row, 5]
    )
    z = factor_whiten(factors.L, free_body_factor(body, factors), jacobian)
    if store:
        for k in range(6):
            Z[world, row, k] = z[k]
    return wp.dot(z, z)


@wp.func
def free_body_response(
    Z_a: wp.array3d[float],
    Z_b: wp.array3d[float],
    world: int,
    row: int,
    body_a: int,
    body_b: int,
    row_type: int,
    factors: BodyFactors,
    cfm: float,
) -> float:
    """Whiten row ``row``'s sides and return its stored diagonal ``1 / A_ii``.

    ``A_ii = cfm + |Z_a|^2 + |Z_b|^2`` adds each present body's term in turn to ``cfm``; a non-positive ``A_ii``
    stores zero (the row does not move). A speed row keeps its axis ``J = e_k`` unwhitened (its law reads the body's
    velocity through the factor, not a row) and takes ``A_ii = cfm + |L^-1 e_k|^2``.
    """
    store = row_type != PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT
    diagonal = cfm
    diagonal += free_body_whiten(Z_a, world, row, body_a, factors, store)
    diagonal += free_body_whiten(Z_b, world, row, body_b, factors, store)
    if diagonal > 0.0:
        return 1.0 / diagonal
    return 0.0


@wp.func
def free_body_jacobian_velocity(
    J_a: wp.array3d[float],
    J_b: wp.array3d[float],
    world: int,
    row: int,
    body_a: int,
    body_b: int,
    body_to_articulation: wp.array[int],
    first: wp.array[int],
    velocity: wp.array[float],
) -> float:
    """``J_row v`` of the writers' (unwhitened) rows over a velocity in the addressing ``first``: from zero, body
    ``a``'s six products, then ``b``'s."""
    value = float(0.0)
    if body_a >= 0:
        start = body_coordinate(body_a, body_to_articulation, first)
        for k in range(6):
            value += J_a[world, row, k] * velocity[start + k]
    if body_b >= 0:
        start = body_coordinate(body_b, body_to_articulation, first)
        for k in range(6):
            value += J_b[world, row, k] * velocity[start + k]
    return value


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
    elif row_type == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
        rhs = phi - target
    return rhs, row_w


@wp.func
def free_body_row_header(
    mf: MfRows,
    line: int,
    i: int,
    incident: float,
    inverse: float,
    has_target_velocity: int,
    rigid_body_max_depenetration_velocity: wp.array[float],
    pgs_beta: float,
    contact_w: float,
    dt: float,
    contact_speculative_scale: float,
    restitution_velocity_threshold: float,
    weighted: int,
    material: int,
    mf_eff_mass_inv: wp.array2d[float],
    mf_row_w: wp.array2d[float],
):
    """Free-body row ``i``'s header from its incident velocity ``J v_hat`` and stored inverse diagonal ``1 / A_ii``:
    both stored, then its ``rhs`` (:func:`free_body_row_rhs`). ``weighted`` rows store their relative regularization
    weight; a material contact (``material``: its writer stored its rhs and compliance ``eps``) keeps its rhs."""
    row_type = mf.row_type[line, i]
    mf.incident[line, i] = incident
    mf_eff_mass_inv[line, i] = inverse
    if material != 0 and row_type == PGS_CONSTRAINT_TYPE_CONTACT:
        if mf.eps[line, i] > 0.0:
            return
    target = float(0.0)
    if has_target_velocity != 0:
        target = mf.target[line, i]
    rhs, row_w = free_body_row_rhs(
        mf.body_a[line, i],
        mf.body_b[line, i],
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
        mf_row_w[line, i] = row_w


@wp.kernel(enable_backward=False)
def free_body_rows_kernel(
    mf: MfRows,
    mf_constraint_count: wp.array[int],
    factors: BodyFactors,
    has_target_velocity: int,
    incident_velocity: wp.array[float],
    rigid_body_max_depenetration_velocity: wp.array[float],
    pgs_cfm: float,
    pgs_beta: float,
    contact_w: float,
    dt: float,
    contact_speculative_scale: float,
    restitution_velocity_threshold: float,
    mf_max_constraints: int,
    weighted: int,
    material: int,
    # outputs
    mf_eff_mass_inv: wp.array2d[float],
    mf_row_w: wp.array2d[float],
):
    """Whiten each free-body row (:func:`free_body_response`) and store its header (:func:`free_body_row_header`).

    One thread per row slot (``world * mf_max_constraints + row``). A row's incident velocity ``J v_hat`` comes from
    the writers' ``J`` (a speed row, which carries no ``J``, zero). A class whose placement walks writes its rows by
    the walk and builds them with :func:`.walk_rows.free_body_headers_kernel`.
    """
    tid = wp.tid()
    world = tid // mf_max_constraints
    i = tid % mf_max_constraints
    if i >= mf_constraint_count[world]:
        return
    mf_line = mf.first[world]
    body_a = mf.body_a[mf_line, i]
    body_b = mf.body_b[mf_line, i]
    row_type = mf.row_type[mf_line, i]
    incident = float(0.0)
    if row_type != PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT:
        incident = free_body_jacobian_velocity(
            mf.Z_a,
            mf.Z_b,
            mf_line,
            i,
            body_a,
            body_b,
            factors.body_to_articulation,
            factors.dof_start,
            incident_velocity,
        )
    inverse = free_body_response(mf.Z_a, mf.Z_b, mf_line, i, body_a, body_b, row_type, factors, pgs_cfm)
    free_body_row_header(
        mf,
        mf_line,
        i,
        incident,
        inverse,
        has_target_velocity,
        rigid_body_max_depenetration_velocity,
        pgs_beta,
        contact_w,
        dt,
        contact_speculative_scale,
        restitution_velocity_threshold,
        weighted,
        material,
        mf_eff_mass_inv,
        mf_row_w,
    )


@wp.kernel(enable_backward=False)
def free_body_coordinates_kernel(
    mf_first: wp.array[int],
    mf_constraint_count: wp.array[int],
    mf_body_a: wp.array2d[int],
    mf_body_b: wp.array2d[int],
    body_to_articulation: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    mf_max_constraints: int,
    # outputs
    mf_dof_a: wp.array2d[int],
    mf_dof_b: wp.array2d[int],
):
    """Give each row its bodies' first world coordinates (:func:`body_coordinate`), one thread per row slot."""
    tid = wp.tid()
    world = tid // mf_max_constraints
    c = tid % mf_max_constraints
    if c >= mf_constraint_count[world]:
        return
    mf_line = mf_first[world]
    mf_dof_a[mf_line, c] = body_coordinate(mf_body_a[mf_line, c], body_to_articulation, articulation_world_dof_offset)
    mf_dof_b[mf_line, c] = body_coordinate(mf_body_b[mf_line, c], body_to_articulation, articulation_world_dof_offset)


@cache
def free_body_meta_kernel(mf_max_constraints: int, device_arch: str) -> wp.Kernel:
    """Pack each world's free-body rows into the 128-bit word the matrix-free sweeps load (:meth:`.FreeBodyZ.pack`).

    A warp per world (``wp.launch_tiled`` over the worlds, 32 threads; one thread on the CPU), a lane per row; the
    word holds the rows' world coordinates, stored inverse diagonal, right-hand side (the law's plus the row's incident
    velocity: its residual at ``du = 0``), type and parent. ``device_arch`` keys the variant.
    """
    _validate_mf_metadata_encoding(mf_max_constraints)
    word = FreeBodyZ().pack("packed", "da", "db", "diag", "rhs_val", "rt", "par", "        ")
    body = f"""    int m_mf = mf_constraint_count.data[world];
    const int mf_line = mf_first.data[world];
    __builtin_assume(mf_line >= 0);
    int off_mf = mf_line * {mf_max_constraints};
    int off_meta = off_mf * 4;

    for (int i = LANE; i < m_mf; i += STRIDE) {{
        int da = mf_dof_a.data[off_mf + i];
        int db = mf_dof_b.data[off_mf + i];
        float diag = mf_eff_mass_inv.data[off_mf + i];
        float rhs_val = mf_rhs.data[off_mf + i] + mf_row_incident.data[off_mf + i];
        int rt = mf_row_type.data[off_mf + i];
        int par = mf_row_parent.data[off_mf + i];

{word}
        *reinterpret_cast<int4*>(&mf_meta.data[off_meta + i * 4]) = packed;
    }}
"""
    return KernelTemplate(
        f"free_body_meta_{mf_max_constraints}",
        kernel_template.params(
            ("mf_constraint_count", wp.array[int]),
            ("mf_first", wp.array[int]),
            ("mf_dof_a", wp.array2d[int]),
            ("mf_dof_b", wp.array2d[int]),
            ("mf_eff_mass_inv", wp.array2d[float]),
            ("mf_rhs", wp.array2d[float]),
            ("mf_row_incident", wp.array2d[float]),
            ("mf_row_type", wp.array2d[int]),
            ("mf_row_parent", wp.array2d[int]),
            ("mf_meta", wp.array2d[int]),
        ),
        dual_native(body, preludes=(lane_group(32), CPU_MATH)),
        dispatch=kernel_template.WORLDS,
        composed=True,
        native_name="free_body_meta_native",
    ).kernel()


@dataclasses.dataclass(frozen=True)
class FreeBodyZBuild:
    """The build of free-body rows (:class:`.response_operator.FreeBodyZ`).

    Attributes:
        rows: Each row's whitened entries, stored inverse diagonal, incident velocity and ``rhs``
            (:func:`free_body_rows_kernel`).
        coordinates: Each row's bodies' first world coordinates (:func:`free_body_coordinates_kernel`).
        meta: The factory of the row-word pack (:func:`free_body_meta_kernel`).
    """

    rows: wp.Kernel
    coordinates: wp.Kernel
    meta: Callable[[int, str], wp.Kernel]


FREE_BODY_Z = FreeBodyZBuild(
    rows=free_body_rows_kernel, coordinates=free_body_coordinates_kernel, meta=free_body_meta_kernel
)
"""The free-body rows' build (:meth:`.FreeBodyZ.build`)."""


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
# Factor rows (SparseFactor): the factor
# ---------------------------------------------------------------------------------------------------------------------
#
# The factor of an articulation's mass matrix: ``P H P^T = L L^T`` in packed ancestor-closed storage, then
# ``L^-1`` in the same pattern. One warp per articulation (the CPU runs it on one thread): the mass entries are
# read from the held tree's geometric entries, with the joint regularization and the drive's stiffness on the
# diagonal, then factored left-looking level by level and inverted column by column along each column's ancestor
# chain, both with plain division. A nonpositive or nonfinite pivot sets the articulation's status and publishes
# NaN in place of the inverse.


def sparse_cholesky_source(dof_count: int, nonzero_count: int) -> str:
    """Factor ``factor`` in place, left-looking by the topology's levels: each level's pivots (``sqrt`` of the
    diagonal less its row's squares, in order), then its entries (less their prefix products, divided by the
    pivot). The caller binds ``indices``, ``factor``, ``status``, ``group``, ``lane``/``stride`` and a lane group."""
    if not 1 <= dof_count <= 64 or not dof_count <= nonzero_count <= dof_count * (dof_count + 1) // 2:
        raise ValueError("Invalid packed sparse factor dimensions")
    return r"""    for (int level = 0; level < indices.factor_level_count; ++level) {
        const int level_begin = indices.factor_level_offsets.data[level];
        const int level_end = indices.factor_level_offsets.data[level + 1];
        for (int pivot = level_begin + lane; pivot < level_end; pivot += stride) {
            const int col = indices.factor_level_columns.data[pivot];
            const int begin = indices.row_offsets.data[col];
            const int diagonal = indices.row_offsets.data[col + 1] - 1;
            float value = factor[diagonal];
            for (int entry = begin; entry < diagonal; ++entry) value -= factor[entry] * factor[entry];
            if (!(value > 0.0f) || !wp::isfinite(value)) {
#if defined(__CUDA_ARCH__)
                atomicExch(status.data + group, 1);
#else
                status.data[group] = 1;
#endif
            }
            factor[diagonal] = sqrtf(value);
        }
        SYNC();
        const int entry_begin = indices.factor_level_entry_offsets.data[level];
        const int entry_end = indices.factor_level_entry_offsets.data[level + 1];
        for (int work = entry_begin + lane; work < entry_end; work += stride) {
            const int target = indices.factor_level_entries.data[work];
            const int row = indices.entry_rows.data[target];
            const int col = indices.columns.data[target];
            const int begin = indices.row_offsets.data[col];
            const int diagonal = indices.row_offsets.data[col + 1] - 1;
            // Ancestor closure makes the column's whole prefix present in this
            // row. Both prefixes are contiguous under the reversed DFS order.
            const int row_begin = target - (diagonal - begin);
            float value = factor[target];
            for (int entry = begin; entry < diagonal; ++entry) {
                value -= factor[row_begin + entry - begin] * factor[entry];
            }
            factor[target] = value / factor[diagonal];
        }
        SYNC();
    }
"""


def sparse_inverse_source(dof_count: int) -> str:
    """Write ``inverse = L^-1`` column by column, each lane a column along its ancestor chain (the column's CSC
    entries in order: ``(e_col - sum L x) / L_rr``). The caller binds ``indices``, ``factor``, ``inverse`` and
    ``lane``/``stride``; synchronization and publication are its own."""
    if not 1 <= dof_count <= 64:
        raise ValueError("Sparse inverse dimensions must fit the topology's 64-coordinate bound")
    return f"""
    // Each inverse column follows only its ancestor chain.
    for (int col = lane; col < {dof_count}; col += stride) {{
        const int begin = indices.column_offsets.data[col];
        const int end = indices.column_offsets.data[col + 1];
        for (int position = begin; position < end; ++position) {{
            const int target = indices.column_entries.data[position];
            const int row = indices.entry_rows.data[target];
            const int diagonal = indices.row_offsets.data[row + 1] - 1;
            float value = row == col ? 1.0f : 0.0f;
            for (int previous = begin; previous < position; ++previous) {{
                const int other = indices.column_entries.data[previous];
                const int factor_entry = diagonal - row + indices.entry_rows.data[other];
                value -= factor[factor_entry] * inverse[other];
            }}
            value /= factor[diagonal];
            inverse[target] = value;
        }}
    }}
"""


_FACTOR_ASSEMBLY = """
    const int group = tid >> 5;
    if (group >= group_to_art.shape[0]) return;
    const int art = group_to_art.data[group];
    if (mass_update_mask.data[art] == 0) return;
    const int lane = LANE, stride = STRIDE;
#if defined(__CUDA_ARCH__)
    __shared__ float factors[$FACTORS];
    __shared__ float inverses[$FACTORS];
    float* factor = factors + SUB * $NNZ;
    float* inverse = inverses + SUB * $NNZ;

#else
    if ((tid & 31) != 0) return;
    float factor[$NNZ];
    float inverse[$NNZ];
#endif
    const int start = articulation_dof_start.data[art];
$BODY
    SYNC();
    const bool valid = status.data[group] == 0;
    for (int entry = lane; entry < $NNZ; entry += stride) {
        Linv_group.data[group * $NNZ + entry] = valid ? inverse[entry] : NAN;
    }
"""

SPARSE_FACTOR_BODY = """
    if (lane == 0) status.data[group] = 0;
    SYNC();
    for (int entry = lane; entry < $NNZ; entry += stride) {
        const int row = indices.entry_rows.data[entry];
        const int col = indices.columns.data[entry];
        const int physical_row = indices.permutation.data[row];
        float value = geometric.data[group * $NNZ + entry];
        if (row == col) {
            value += R_group.data[group * $N + physical_row];
            const int drive = drive_row_by_dof.data[start + physical_row];
            if (drive >= 0) {
                row_H.data[drive] = value;
                if (row_K.data[drive] > 0.0f) value += row_K.data[drive];
            }
        }
        factor[entry] = value;
    }
    SYNC();
    $FACTOR
    $INVERSE
"""
"""The sparse factor of one tree, ``L^-1`` included: the caller binds ``indices``, ``status``, ``geometric``, ``R_group``,
``drive_row_by_dof``, ``row_K``, ``row_H``, ``factor`` and ``inverse``, and synchronizes after it."""


def sparse_factor_body(dof_count: int, nonzero_count: int) -> str:
    """:data:`SPARSE_FACTOR_BODY` at a tree's sizes."""
    return kernel_template.substitute(
        SPARSE_FACTOR_BODY,
        NNZ=nonzero_count,
        N=dof_count,
        FACTOR=sparse_cholesky_source(dof_count, nonzero_count),
        INVERSE=sparse_inverse_source(dof_count),
    )


@cache
def sparse_factor_kernel(dof_count: int, nonzero_count: int, *, warps_per_block: int = 4) -> wp.Kernel:
    """Read each articulation's mass matrix from the held tree's packed geometric entries and publish its packed
    ``L^-1`` (see the section's comment).

    ``dof_count * 32`` threads per group of articulations, ``warps_per_block * 32`` per block; value arrays must be
    contiguous. An articulation without a mass update keeps its factor and status; ``status`` is 0 after a refresh
    and 1 for a nonpositive or nonfinite pivot (no pivot floor).
    """
    n, nnz, warps = int(dof_count), int(nonzero_count), int(warps_per_block)
    if not 1 <= n <= 64 or not n <= nnz <= n * (n + 1) // 2 or not 1 <= warps <= 32:
        raise ValueError("Invalid sparse factor dimensions or warp count")
    body = kernel_template.substitute(_FACTOR_ASSEMBLY, FACTORS=warps * nnz, NNZ=nnz, BODY=sparse_factor_body(n, nnz))
    return KernelTemplate(
        f"sparse_factor_{n}_{nnz}_{warps}",
        (
            ("group_to_art", wp.array[int]),
            ("mass_update_mask", wp.array[int]),
            ("articulation_dof_start", wp.array[int]),
            ("geometric", wp.array2d[float]),
            ("R_group", wp.array2d[float]),
            ("drive_row_by_dof", wp.array[int]),
            ("row_K", wp.array[float]),
            ("row_H", wp.array[float]),
            ("indices", _SparseMassMatrixIndices),
            ("Linv_group", wp.array2d[float]),
            ("status", wp.array[int]),
        ),
        dual_native(body, preludes=(lane_group(32),)),
        kernel_template.THREADS,
        composed=True,
        native_name="sparse_factor_native",
    ).kernel()


@dataclasses.dataclass(frozen=True)
class SparseFactorBuild:
    """The factor of the factor rows (:class:`.response_operator.SparseFactor`): its rows and their leave are the
    walk's (:func:`.walk_rows.support_rows_kernel`, :func:`.walk_rows.support_publish_kernel`); the factor and its
    inverse serve the predictor until the walk becomes the factor (Z4).

    Attributes:
        factor: The factory of the factor and its inverse (:func:`sparse_factor_kernel`).
    """

    factor: Callable[..., wp.Kernel]


SPARSE_FACTOR = SparseFactorBuild(factor=sparse_factor_kernel)
"""The factor rows' factor (:meth:`.SparseFactor.build`)."""


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
    for (int world = worker; world < data.world_count; world += workers) {
        const int m = data.counts.data[world];
        const int mf = data.mf_counts.data[world];
        if (LANE == 0) data.selector.data[world] = mf == 0 ? 0 : wp::max(mf, 1);
        if (mf == 0) continue;
        if (m < 0 || m > M || mf < 1 || mf > F) continue;
        const int po = data.primary_offset.data[world];
        const int so = data.secondary_offset.data[world];
        const int group = data.primary_group.data[world];
        if (!((po == 0 && so == $PRIMARY) || (po == $SECONDARY && so == 0))
            || group < 0 || group >= data.L.shape[0]) continue;
        int valid = 1;
        for (int k = LANE; k < $PRIMARY * $PRIMARY; k += STRIDE) {
            const float x = data.L.data[group * $PRIMARY * $PRIMARY + k];
            valid &= wp::isfinite(x) && wp::abs(x) <= 1.0e15f;
        }
        for (int row = LANE; row < m; row += STRIDE) {
            const int r = data.first.data[world] * M + row;
            const int type = data.row_type.data[r];
            const float diag = data.diag.data[r];
            const float mu = data.mu.data[r];
            valid &= (type == 0 || type == 2 || type == 3)
                && wp::isfinite(diag) && diag > 0.0f
                && wp::isfinite(mu) && mu >= 0.0f && wp::isfinite(data.rhs.data[r]);
            if (type == 2) {
                const int p = data.row_parent.data[r];
                if (p < 0 || p + 2 >= m || (row != p + 1 && row != p + 2)) valid = 0;
                else {
                    const int base = data.first.data[world] * M + p;
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
            const int r = data.mf_first.data[world] * F + row;
            const int packed = data.mf_meta.data[r * 4];
            const int a = packed >> 16, b = static_cast<int>(static_cast<short>(packed & 65535));
            const int tp = data.mf_meta.data[r * 4 + 3], type = tp & 65535;
            union { int bits; float value; } diag, rhs;
            diag.bits = data.mf_meta.data[r * 4 + 1];
            rhs.bits = data.mf_meta.data[r * 4 + 2];
            const float mu = data.mf_mu.data[r];
            valid &= (a == -1 || a == so) && (b == -1 || b == so) && (a == so || b == so)
                && (type == 0 || type == 2 || type == 4)
                && wp::isfinite(diag.value) && diag.value > 0.0f && wp::isfinite(rhs.value)
                && wp::isfinite(mu) && mu >= 0.0f;
            if (type == 2) {
                const int encoded_parent = static_cast<unsigned int>(tp) >> 16;
                const int p = encoded_parent == 65535 ? -1 : encoded_parent;
                if (p < 0 || p + 2 >= mf || (row != p + 1 && row != p + 2)) valid = 0;
                else {
                    const int base = (data.mf_first.data[world] * F + p) * 4 + 3;
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
        _PAIRED_COMPONENTS, PRIMARY=P, SECONDARY=secondary_size, WORLD=P + secondary_size
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


@dataclasses.dataclass(frozen=True)
class FactorRowsBuild:
    """The build of a pair's factor rows (:class:`.response_operator.FactorRows`): the rows themselves are the packed
    rows at their world coordinates (:mod:`.packed_rows`).

    Attributes:
        components: The factory of the qualification (:func:`paired_components_kernel`).
    """

    components: Callable[..., wp.Kernel]


FACTOR_ROWS = FactorRowsBuild(components=paired_components_kernel)
"""The paired factor rows' build (:meth:`.FactorRows.build`)."""
