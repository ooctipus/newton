# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build the row responses of the free-body and factor-row representations, and publish their velocities.

A :class:`.response_operator.ResponseOperator` owns the arithmetic a sweep runs on a row's response. Its *build*
forms that response from the row's Jacobian and the mass operator. This module holds the builds of four
representations, and the publication of the one whose sweep leaves its velocity change outside the kernel:

- :class:`.response_operator.FreeBodyZ` (rows over at most two free bodies): each row whitened per body, ``Z =
  L^-1 J^T`` through the body's held factor, and ``1 / A_ii``, and the rows' body coordinates in each addressing a
  mapping reads (a world's local body slots, world and global coordinates, the packed row word);
- :class:`.response_operator.SparsePairZ` (a compact dense articulation plus two scalar coordinates): a row's dense
  part whitened through the compact factor, its two sparse slots ``J / sqrt(m)`` merged when they share a coordinate,
  and its diagonal ``|Z|^2 + cfm``;
- :class:`.response_operator.SparseFactor` (rows ``Z = L^-1 P J^T`` packed by support): the internal rows' unit
  columns of ``L^-1`` (position limits, velocity limits, mimic), and the publication ``v = v_hat + P^T L^-T u`` of a
  world's factor-coordinate velocity change;
- :class:`.response_operator.FactorRows` (a pair's rows in factor coordinates, a lane per coordinate): the packed
  rows at their world coordinates (:mod:`.packed_rows`);

and :class:`BilateralElimination`, the Schur transform of a dense response build that folds a
world's bilateral block into the responses of its other rows.

**API.** Each operator class returns its build from ``build()`` (``FreeBodyZ.build()``, ``SparsePairZ.build()``,
``SparseFactor.build()``, ``FactorRows.build()``): a frozen descriptor whose members are the Warp kernels (or cached
factories of kernel variants, or a Warp function) of each step of the build and of the publication; the solver's
launch sites call them directly. The arithmetic of a step is one function per representation and mapping, called by
every kernel that runs the step:

- the Warp kernels call Warp functions (``free_body_response``, ``sparse_pair_slots``, ``packed_product``...),
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

from . import field_group, kernel_template
from .contact_input import ContactInput, contact_frame
from .contact_laws import (
    CONTACT_SPIN_ROW,
    contact_friction,
    contact_restitution,
    contact_row_entry,
    contact_row_frame,
    contact_seed,
    prescribed_spin_target,
    prescribed_target,
)
from .dense_rows import DenseRows, MfRows
from .device_preludes import CPU_MATH, dual_native, lane_group
from .field_group import Field
from .friction_patches import PatchRowLaw
from .kernel_template import Dispatch, KernelTemplate, python_kernel
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    PGS_CONSTRAINT_TYPE_MIMIC,
    PGS_CONSTRAINT_TYPE_TORSION,
    PGS_LOCAL_SOLVE_OWNER_GENERAL,
    BodyFactors,
    factor_whiten,
    free_body_factor,
)
from .paired_components import FREE_BODY_DOFS, PAIRED_COMPONENT_THREADS, PairedComponentData
from .prefix_rows import PrefixLaw, limits, prefix_rows
from .projection_laws import BILATERAL_PIVOT, PROJECTION_LAW
from .response_builds_dense import _indent, _Vec6, whiten_six
from .response_operator import FreeBodyZ
from .row_build import contact_normal_beta, contact_torsion_mu
from .row_laws import (
    LINK_ROWS,
    ROW_LAW,
    contact_capped_beta,
    contact_depenetration_cap,
    contact_material_rhs,
    contact_restitution_rhs,
    contact_rhs,
    row_rhs,
)
from .row_metadata import _validate_mf_metadata_encoding
from .solver_fields import Preelim
from .sparse_mass_matrix import _SparseMassMatrixIndices, _SparseRowSupport
from .sparse_row_projection import _sparse_contact_projection_source

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
# Sparse-diagonal rows (SparsePairZ)
# ---------------------------------------------------------------------------------------------------------------------
#
# A sparse-diagonal row acts on a world's compact dense articulation (its dense part) and on at most two scalar
# coordinates of diagonal-mass articulations (its slots), as a whitened row in factor coordinates: the dense part
# ``Z = L^-1 P J^T`` through the articulation's leaf-first factor (stored in place of the producers' ``J``), and a slot
# ``Z_s = J_s / sqrt(m_s)``, a star's factor being diagonal (``s = m^-1/2``, the square root of the tree factor's
# ``diagonal_inverse_mass``). An endpoint has a slot when its articulation has the sparse size and its body a single
# response coordinate; two slots on one coordinate merge into the first. ``A_ii = |Z|^2 + sum_s Z_s^2 + cfm``.


COMPACT_BOUNDARY = (
    Field("body_single_dof", wp.int32, ("body",), owner="forest", attr="body_single_response_dof"),
    Field("dof_offset", wp.int32, ("articulation",), owner="forest", attr="articulation_world_dof_offset"),
    Field("inverse_mass", wp.float32, ("dof",), owner="factor", attr="diagonal_inverse_mass"),
    Field("dense_group", wp.int32, ("world",), attr="sparse_diagonal_dense_groups", owner="placement"),
    Field("factor", wp.float32, ("group", 6, 6), owner="six"),
    Field("sparse_size", wp.int32, attr="sparse_diagonal_response_size", owner="placement"),
    Field("Z", wp.float32, ("group", "row", 6), owner="six"),
    Field("sparse_dof", wp.int32, ("world", "row", 2), attr="sparse_diagonal_row_dof", owner="placement"),
    Field("sparse_z", wp.float32, ("world", "row", 2), attr="sparse_diagonal_row_z", owner="placement"),
    Field("first", wp.int32, ("world",), attr="dense_first", owner="placement"),
    Field("row_incident", wp.float32, ("world", "row"), owner="placement"),
)
"""What the compact boundary producer writes besides the contact input's row records: each row's whitened dense-six
part (by the six-DOF size group's factor), its two whitened sparse slots and its incident velocity."""

CompactBoundary = field_group.struct("CompactBoundary", COMPACT_BOUNDARY, __name__)


@wp.func
def sparse_pair_slot(
    row: int,
    art: int,
    body: int,
    negate: bool,
    size: int,
    response_dofs: wp.array[int],
    single_dof: wp.array[int],
    dof_start: wp.array[int],
    dof_offset: wp.array[int],
    inverse_mass: wp.array[float],
    motion: wp.array[wp.spatial_vector],
    origin: wp.array[wp.vec3],
    direction: wp.vec3,
    point: wp.vec3,
):
    """One endpoint's sparse slot of contact row ``row``: its world coordinate (-1 for none) and whitened entry
    ``Z = J / sqrt(m)`` (:func:`.contact_laws.contact_row_entry`).

    The entry of the second endpoint is negated (``negate``): a row pushes its endpoints apart. A body's coordinate
    holds a slot only inside its articulation's window.
    """
    coordinate = int(-1)
    value = float(0.0)
    if art >= 0 and body >= 0 and response_dofs[art] == size:
        dof = single_dof[body]
        if dof >= 0:
            local = dof - dof_start[art]
            if local >= 0 and local < size:
                coordinate = dof_offset[art] + local
                value = contact_row_entry(row, motion[dof], direction, point, origin[art]) * wp.sqrt(inverse_mass[dof])
                if negate:
                    value = -value
    return coordinate, value


@wp.func
def sparse_pair_slots(
    row: int,
    art_a: int,
    body_a: int,
    art_b: int,
    body_b: int,
    point: wp.vec3,
    direction: wp.vec3,
    size: int,
    response_dofs: wp.array[int],
    single_dof: wp.array[int],
    dof_start: wp.array[int],
    dof_offset: wp.array[int],
    inverse_mass: wp.array[float],
    motion: wp.array[wp.spatial_vector],
    origin: wp.array[wp.vec3],
):
    """Contact row ``row``'s two sparse slots (:func:`sparse_pair_slot`), merged into the first when they share a
    coordinate: ``(coordinate, Z)`` of slot 0, then of slot 1."""
    coord_a, value_a = sparse_pair_slot(
        row,
        art_a,
        body_a,
        False,
        size,
        response_dofs,
        single_dof,
        dof_start,
        dof_offset,
        inverse_mass,
        motion,
        origin,
        direction,
        point,
    )
    coord_b, value_b = sparse_pair_slot(
        row,
        art_b,
        body_b,
        True,
        size,
        response_dofs,
        single_dof,
        dof_start,
        dof_offset,
        inverse_mass,
        motion,
        origin,
        direction,
        point,
    )
    if coord_a >= 0 and coord_a == coord_b:
        value_a += value_b
        coord_b = -1
        value_b = 0.0
    return coord_a, value_a, coord_b, value_b


@wp.func
def sparse_pair_diagonal(coord_0: int, value_0: float, coord_1: int, value_1: float):
    """The sparse slots' term of ``A_ii``: from zero, each present slot's ``Z_s^2`` in slot order."""
    diagonal = float(0.0)
    if coord_0 >= 0:
        diagonal += value_0 * value_0
    if coord_1 >= 0:
        diagonal += value_1 * value_1
    return diagonal


@wp.func
def sparse_pair_dense(factor: wp.array3d[float], group: int, jacobian: _Vec6):
    """The dense part whitened, ``Z = L^-1 P J^T``, through the group's compact six-coordinate factor
    (:func:`.response_builds_dense.whiten_six`), and its term ``|Z|^2`` of the diagonal, from zero in coordinate
    order."""
    whitened = whiten_six(factor, group, jacobian)
    diagonal = float(0.0)
    for dof in range(6):
        diagonal += whitened[dof] * whitened[dof]
    return whitened, diagonal


@wp.kernel(enable_backward=False)
def sparse_pair_contact_rows_kernel(
    dense_first: wp.array[int],
    data: ContactInput,
    total_num_workers: int,
    target_size: int,
    articulation_world_dof_offset: wp.array[int],
    body_single_response_dof: wp.array[int],
    diagonal_inverse_mass: wp.array[float],
    law: PatchRowLaw,
    # outputs
    sparse_row_dof: wp.array3d[int],
    sparse_row_z: wp.array3d[float],
):
    """Write each contact row's sparse slots (:func:`sparse_pair_slots`); a worker strides over the contacts.

    A row's geometry is the contact front end's (:func:`.contact_input.contact_frame`): tangent rows act at the
    row law's friction location, as in the dense Jacobian builders.
    """
    worker = wp.tid()
    total_contacts = data.count[0]
    for c in range(worker, total_contacts, total_num_workers):
        slot = data.slot[c]
        if data.path[c] != 0 or slot < 0:
            continue
        body_a, body_b, normal, point, _phi, tangent0, tangent1, friction_point = contact_frame(data, law, c)
        world = data.world[c]
        line = dense_first[world]
        art_a = data.art0[c]
        art_b = data.art1[c]
        rows = data.slots_needed[c]
        # The normal, its tangents and, under contact torsion, its spin row.
        for row in range(4):
            if row >= rows:
                continue
            direction, row_point = contact_row_frame(row, normal, tangent0, tangent1, point, friction_point)
            coord_a, value_a, coord_b, value_b = sparse_pair_slots(
                row,
                art_a,
                body_a,
                art_b,
                body_b,
                row_point,
                direction,
                target_size,
                data.response_dofs,
                body_single_response_dof,
                data.dof_start,
                articulation_world_dof_offset,
                diagonal_inverse_mass,
                data.motion,
                data.origin,
            )
            output_row = slot + row
            sparse_row_dof[line, output_row, 0] = coord_a
            sparse_row_dof[line, output_row, 1] = coord_b
            sparse_row_z[line, output_row, 0] = value_a
            sparse_row_z[line, output_row, 1] = value_b


@wp.func
def sparse_pair_slot_velocity(
    coordinate: int,
    value: float,
    body: int,
    single_dof: wp.array[int],
    inverse_mass: wp.array[float],
    velocity: wp.array[float],
):
    """A slot's term of a row's ``J v``, ``Z_s (v / s)`` on its body's coordinate; zero for an empty slot."""
    term = float(0.0)
    if coordinate >= 0:
        dof = single_dof[body]
        scale = wp.sqrt(inverse_mass[dof])
        if scale > 0.0:
            term = value * (velocity[dof] / scale)
    return term


@wp.func
def sparse_pair_row_response(
    data: ContactInput,
    boundary: CompactBoundary,
    world: int,
    row: int,
    output_row: int,
    group: int,
    has_dense: bool,
    art0: int,
    body0: int,
    art1: int,
    body1: int,
    direction: wp.vec3,
    point: wp.vec3,
):
    """Write contact row ``row`` of the compact boundary whitened: its dense-six part, its sparse slots, its diagonal
    ``(|Z|^2 + sum_s Z_s^2) + cfm`` (:func:`.contact_laws.contact_row_entry`) and its incident velocity ``J v_hat``.

    The dense Jacobian sums both endpoints' entries per coordinate of the six-coordinate articulation (each where
    the body's coordinate mask holds the coordinate), then is whitened in registers.
    """
    line = boundary.first[world]
    diagonal = float(0.0)
    incident = float(0.0)
    if has_dense:
        jacobian = _Vec6()
        for dof in range(6):
            value0 = float(0.0)
            value1 = float(0.0)
            bit = wp.uint32(1) << wp.uint32(dof)
            if art0 >= 0 and body0 >= 0 and data.response_dofs[art0] == 6:
                if (data.body_mask[body0] & bit) != wp.uint32(0):
                    value0 = contact_row_entry(
                        row, data.motion[data.dof_start[art0] + dof], direction, point, data.origin[art0]
                    )
                    incident += value0 * data.incident[data.dof_start[art0] + dof]
            if art1 >= 0 and body1 >= 0 and data.response_dofs[art1] == 6:
                if (data.body_mask[body1] & bit) != wp.uint32(0):
                    value1 = -contact_row_entry(
                        row, data.motion[data.dof_start[art1] + dof], direction, point, data.origin[art1]
                    )
                    incident += value1 * data.incident[data.dof_start[art1] + dof]
            jacobian[dof] = value0 + value1
        whitened, diagonal = sparse_pair_dense(boundary.factor, group, jacobian)
        for dof in range(6):
            boundary.Z[group, output_row, dof] = whitened[dof]
    coord0, value0, coord1, value1 = sparse_pair_slots(
        row,
        art0,
        body0,
        art1,
        body1,
        point,
        direction,
        boundary.sparse_size,
        data.response_dofs,
        boundary.body_single_dof,
        data.dof_start,
        boundary.dof_offset,
        boundary.inverse_mass,
        data.motion,
        data.origin,
    )
    boundary.sparse_dof[line, output_row, 0] = coord0
    boundary.sparse_dof[line, output_row, 1] = coord1
    boundary.sparse_z[line, output_row, 0] = value0
    boundary.sparse_z[line, output_row, 1] = value1
    sparse = sparse_pair_diagonal(coord0, value0, coord1, value1)
    data.diag[line, output_row] = (diagonal + sparse) + data.cfm
    single_dof, inverse_mass = boundary.body_single_dof, boundary.inverse_mass
    incident += sparse_pair_slot_velocity(coord0, value0, body0, single_dof, inverse_mass, data.incident)
    incident += sparse_pair_slot_velocity(coord1, value1, body1, single_dof, inverse_mass, data.incident)
    boundary.row_incident[line, output_row] = incident


@cache
def sparse_pair_whiten_kernel() -> wp.Kernel:
    """Whiten the sparse-diagonal rows' dense parts in place and store their diagonals, one thread per world row
    (``dim = worlds * rows``, ``rows`` at most ``max_constraints``).

    The dense part, ``Z = L^-1 P J^T`` through the world's compact dense factor (:class:`.CholeskySolve`'s forward
    solve at the leaf-first positions, Z0), overwrites the producers' ``J``; ``A_ii = (|Z|^2 + sum_s Z_s^2) + cfm``
    adds the row's whitened slots; the row's incident velocity ``J v_hat`` sums the dense part's terms while ``J`` is
    in hand, then the slots' ``Z_s (v / s)``. Every row below the world's count, or with ``prefix_only`` below its
    limit prefix's bound (the compact boundary whitens its contact rows as it writes them).
    """
    from .response_builds_dense import _element  # noqa: PLC0415
    from .response_operator import CholeskySolve, Reversed  # noqa: PLC0415

    solve = CholeskySolve("n_dofs", lambda r, c: _element("L_group", "group", r, c))
    # The row's last coordinates, read backwards: factor coordinate k at physical position n_dofs - 1 - k.
    entries = Reversed(lambda k: f"whitened_row[-{k}]")
    body = f"""    const int world = tid / rows;
    const int sparse_row = tid - world * rows;
    int end = world_constraint_count.data[world];
    if (end > max_constraints) end = max_constraints;
    if (prefix_only != 0 && dense_phase_bounds.data[world * 2 + 1] < end) end = dense_phase_bounds.data[world * 2 + 1];
    if (sparse_row >= end) return;
    const int line = dense_first.data[world];
    __builtin_assume(line >= 0);
    const int group = dense_groups.data[world];
    float* whitened_row = &{_element("Z_group", "group", "sparse_row", "n_dofs - 1")};
    float incident = 0.0f;
    const int dense_start = world * world_dof_indices.shape[1] + dense_offsets.data[world];
    for (int i = 0; i < n_dofs; ++i) {{
        const int global_dof = world_dof_indices.data[dense_start + i];
        if (global_dof >= 0) incident += {_element("Z_group", "group", "sparse_row", "i")} * v_hat.data[global_dof];
    }}
    {{
{_indent(solve.whiten(entries, entries), "        ")}
    }}
    float diagonal = 0.0f;
    for (int i = 0; i < n_dofs; ++i) {{
        const float z = {_element("Z_group", "group", "sparse_row", "i")};
        diagonal += z * z;
    }}
    for (int slot = 0; slot < 2; ++slot) {{
        const int world_dof = {_element("sparse_row_dof", "line", "sparse_row", "slot")};
        if (world_dof >= 0) {{
            const float z = {_element("sparse_row_z", "line", "sparse_row", "slot")};
            diagonal += z * z;
            const int global_dof = world_dof_indices.data[world * world_dof_indices.shape[1] + world_dof];
            if (global_dof >= 0) {{
                const float scale = wp::sqrt(diagonal_inverse_mass.data[global_dof]);
                if (scale > 0.0f) incident += z * (v_hat.data[global_dof] / scale);
            }}
        }}
    }}
    {_element("world_diag", "line", "sparse_row")} = diagonal + {_element("row_cfm", "line", "sparse_row")};
    {_element("row_incident", "line", "sparse_row")} = incident;
"""
    return KernelTemplate(
        "sparse_pair_whiten",
        kernel_template.params(
            ("L_group", wp.array3d[float]),
            ("dense_groups", wp.array[int]),
            ("dense_offsets", wp.array[int]),
            ("world_dof_indices", wp.array2d[int]),
            ("world_constraint_count", wp.array[int]),
            ("dense_first", wp.array[int]),
            ("dense_phase_bounds", wp.array2d[int]),
            ("prefix_only", int),
            ("rows", int),
            ("max_constraints", int),
            ("n_dofs", int),
            ("sparse_row_dof", wp.array3d[int]),
            ("sparse_row_z", wp.array3d[float]),
            ("row_cfm", wp.array2d[float]),
            ("diagonal_inverse_mass", wp.array[float]),
            ("v_hat", wp.array[float]),
            ("Z_group", wp.array3d[float]),
            ("world_diag", wp.array2d[float]),
            ("row_incident", wp.array2d[float]),
        ),
        dual_native(body, preludes=()),
        kernel_template.THREADS,
        composed=True,
        native_name="sparse_pair_whiten_native",
    ).kernel()


@dataclasses.dataclass(frozen=True)
class SparsePairBuild:
    """The build of sparse-diagonal rows (:class:`.response_operator.SparsePairZ`).

    Attributes:
        contact_rows: Each contact row's whitened sparse slots (:func:`sparse_pair_contact_rows_kernel`); the dense
            part is the producers' ``J``, whitened by :attr:`whiten`.
        row_response: One compact-boundary row whitened, its dense part, slots and diagonal
            (:func:`sparse_pair_row_response`), the response of the compact contact producer.
        whiten: The rows' dense parts whitened in place and their diagonals (:func:`sparse_pair_whiten_kernel`).
    """

    contact_rows: wp.Kernel
    row_response: wp.Function
    whiten: Callable[[], wp.Kernel]


SPARSE_PAIR = SparsePairBuild(
    contact_rows=sparse_pair_contact_rows_kernel,
    row_response=sparse_pair_row_response,
    whiten=sparse_pair_whiten_kernel,
)
"""The sparse-diagonal rows' build (:meth:`.SparsePairZ.build`)."""


# ---------------------------------------------------------------------------------------------------------------------
# Bilateral elimination: the exact Schur transform of the packed rows
# ---------------------------------------------------------------------------------------------------------------------
#
# The bilateral rows of a linked component of trees (:mod:`.body_cut`), its mimics and its links, form a block ``B``.
# On the packed rows (:mod:`.packed_rows`, ``Z_i = L^-1 P J_i^T`` on each side), wide over the world's coordinates, the
# transform makes the block's rows orthonormal in place by modified Gram-Schmidt, the right-hand sides following:
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


@dataclasses.dataclass(frozen=True)
class BilateralElimination:
    """The exact Schur transform of the packed rows by bilateral blocks of up to ``rows`` rows, and its projection.

    Attributes:
        rows: The block capacity: the model's largest block (its mimic rows plus six per link).
        setup: Make each block's rows orthonormal, once the right-hand sides are final.
        correct: Correct the other rows and their right-hand sides and diagonals.
        weights: The projection: the block rows' impulses, which the sweep seeds.
    """

    rows: int
    setup: wp.Kernel
    correct: wp.Kernel
    weights: wp.Kernel


@cache
def elimination(rows: int) -> BilateralElimination:
    """The bilateral elimination of the packed rows for blocks of up to ``rows`` rows (see the section's comment)."""

    def setup(
        dense_first: wp.array[int],
        preelim: Preelim,
        mimic_slot: wp.array[int],
        link_slot: wp.array[int],
        local_solve_owner: wp.array[int],
        world_row_cfm: wp.array2d[float],
        # outputs
        packed_Z: wp.array3d[float],
        world_rhs: wp.array2d[float],
        world_incident: wp.array2d[float],
        world_diag: wp.array2d[float],
    ):
        # Modified Gram-Schmidt over a block's rows, in order, in place: each takes out the rows kept before it, then
        # joins them unit, or leaves the block inert where nothing of it remains. preelim.slots ends with the kept rows
        # (preelim.nB of them). One thread per block.
        pe = wp.tid()
        base = pe * wp.static(rows)
        world = preelim.world[pe]
        line = dense_first[world]
        entries = packed_Z.shape[2]
        m = bilateral_rows(pe, base, preelim, mimic_slot, link_slot)
        kept = int(0)
        for p in range(m):
            row = preelim.slots[base + p]
            # The row's incident velocity joins its right-hand side: both follow the row through the transform.
            world_rhs[line, row] += world_incident[line, row]
            world_incident[line, row] = 0.0
            own = wide_product(packed_Z, world, row, row, entries)
            for q in range(kept):
                other = preelim.slots[base + q]
                along = wide_product(packed_Z, world, other, row, entries)
                subtract_row(packed_Z, world, row, along, other, entries)
                world_rhs[line, row] -= along * world_rhs[line, other]
            rest = wide_product(packed_Z, world, row, row, entries)
            scale = float(0.0)
            if rest > BILATERAL_PIVOT * own:
                scale = 1.0 / wp.sqrt(rest)
                preelim.slots[base + kept] = row
                kept += 1
            scale_row(packed_Z, world, row, scale, entries)
            world_rhs[line, row] *= scale
            if local_solve_owner[world] == PGS_LOCAL_SOLVE_OWNER_GENERAL:
                world_diag[line, row] = scale * scale * rest + world_row_cfm[line, row]
        for p in range(kept, wp.static(rows)):
            preelim.slots[base + p] = -1
        preelim.nB[pe] = kept

    def correct(
        dense_first: wp.array[int],
        preelim: Preelim,
        row_sides: wp.array2d[wp.vec2i],
        constraint_count: wp.array[int],
        local_solve_owner: wp.array[int],
        world_row_cfm: wp.array2d[float],
        # outputs
        packed_Z: wp.array3d[float],
        world_rhs: wp.array2d[float],
        world_diag: wp.array2d[float],
    ):
        # Correct every other row on the blocks its sides touch, Pi Z_i = Z_i - Z_B y_i with y_i = G^-1 Z_B^T Z_i, take
        # y_i^T b_B back from its right-hand side and store its diagonal |Pi Z_i|^2 + cfm (the squares summed in entry
        # order, the padding's zeros included; in the worlds the general solver owns). One thread per world row, after
        # the right-hand sides are final. A row the block does not see (Z_B^T Z_i exactly zero) keeps its entries; the
        # block's rows, kept or dropped, keep theirs. Blocks hold different trees, so their projectors commute.
        tid = wp.tid()
        capacity = row_sides.shape[1]
        world = tid // capacity
        row = tid - world * capacity
        if world >= constraint_count.shape[0] or row >= constraint_count[world]:
            return
        line = dense_first[world]
        sides = row_sides[line, row]
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
            m = preelim.nB[pe]
            base = pe * wp.static(rows)
            for p in range(wp.static(rows)):
                if preelim.slots[base + p] == row:
                    return
            for p in range(m):
                other = preelim.slots[base + p]
                along = wide_product(packed_Z, world, other, row, entries)
                if along != 0.0:
                    subtract_row(packed_Z, world, row, along, other, entries)
                    world_rhs[line, row] -= along * world_rhs[line, other]
        if touched != 0 and local_solve_owner[world] == PGS_LOCAL_SOLVE_OWNER_GENERAL:
            world_diag[line, row] = wide_product(packed_Z, world, row, row, entries) + world_row_cfm[line, row]

    def weights(
        dense_first: wp.array[int],
        preelim: Preelim,
        world_rhs: wp.array2d[float],
        # outputs
        world_impulses: wp.array2d[float],
    ):
        # The projection: the kept rows' impulses -r_Q, their residual at zero change, one thread per block.
        pe = wp.tid()
        base = pe * wp.static(rows)
        world = preelim.world[pe]
        line = dense_first[world]
        for p in range(preelim.nB[pe]):
            slot = preelim.slots[base + p]
            world_impulses[line, slot] = -world_rhs[line, slot]

    return BilateralElimination(
        rows,
        python_kernel(setup, f"bilateral_setup_{rows}"),
        python_kernel(correct, f"bilateral_correct_{rows}"),
        python_kernel(weights, f"bilateral_weights_{rows}"),
    )


# ---------------------------------------------------------------------------------------------------------------------
# Factor rows (SparseFactor): publication
# ---------------------------------------------------------------------------------------------------------------------


@wp.func
def factor_transpose_column(
    factor_lookup: wp.array2d[int],
    inverse_factor: wp.array2d[float],
    group: int,
    column: int,
    u: wp.array2d[float],
    world: int,
    first: int,
) -> float:
    """``(L^-T u)_column`` in factor coordinates: from zero, ``L^-1[row, column] u_row`` for the rows at and below
    ``column`` in order, through the factor's lookup grid (``u_row`` at ``u[world, first + row]``)."""
    delta = float(0.0)
    for row in range(column, factor_lookup.shape[0]):
        entry = factor_lookup[row, column]
        if entry >= 0:
            delta += inverse_factor[group, entry] * u[world, first + row]
    return delta


@wp.kernel(enable_backward=False)
def factor_publish_kernel(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_world_dof_offset: wp.array[int],
    articulation_dof_start: wp.array[int],
    permutation: wp.array[int],
    factor_lookup: wp.array2d[int],
    inverse_factor: wp.array2d[float],
    factor_velocity_delta: wp.array2d[float],
    v_hat: wp.array[float],
    v_out: wp.array[float],
):
    """Publish ``v = v_hat + P^T L^-T u`` from a world's factor-coordinate velocity change ``u`` once after its
    sweeps: one thread per (articulation, factor coordinate), written at its physical coordinate."""
    group, column = wp.tid()
    art = group_to_art[group]
    world = art_to_world[art]
    delta = factor_transpose_column(
        factor_lookup,
        inverse_factor,
        group,
        column,
        factor_velocity_delta,
        world,
        articulation_world_dof_offset[art],
    )
    dof = articulation_dof_start[art] + permutation[column]
    v_out[dof] = v_hat[dof] + delta


# ---------------------------------------------------------------------------------------------------------------------
# Factor rows (SparseFactor): packed rows from their support
# ---------------------------------------------------------------------------------------------------------------------
#
# A factor row ``Z = L^-1 P J^T`` is nonzero only on the factor coordinates of its support (the ancestors of the
# row's coordinates). It is packed in support order: entry ``index`` is the row's value at the support's coordinate
# of rank ``index`` (the count of support coordinates below it), padded to the row capacity with zero values and,
# where rows carry explicit coordinates (no support table), coordinate -1; ``A_ii = |Z|^2`` (the row law adds
# ``cfm``). One warp packs one row (or the rows that share one support), a lane per support coordinate, and sums the
# squares with ``SUM``. An internal row's Jacobian is a signed unit vector or a difference of two (a mimic row), so
# its ``Z`` is a signed column of ``L^-1``, or a combination of two, read through the factor's lookup grid.


def store_entry(offset: str, index: str, coordinate: str, value: str, explicit: str) -> list[str]:
    """Statements that store a packed factor row's entry ``index`` (from ``offset``): its value, and its coordinate
    where the row carries explicit coordinates (the native condition ``explicit``; empty: always)."""
    store = f"row_dof.data[{offset} + {index}] = {coordinate};"
    return [f"if ({explicit}) {store}" if explicit else store, f"row_factor.data[{offset} + {index}] = {value};"]


def pad_entries(offset: str, explicit: str) -> list[str]:
    """Statements that pad entry ``index`` of a packed factor row (from ``offset``) past its support: value zero,
    coordinate -1 where the row carries explicit coordinates."""
    return store_entry(offset, "index", "-1", "0.0f", explicit)


@dataclasses.dataclass(frozen=True)
class FactorRow:
    """One packed row a support walk writes.

    Attributes:
        offset: The row's first entry in ``row_dof``/``row_factor``.
        value: An entry's stored value (native, in the walk's ``value`` and ``entry``).
        coordinate: An entry's explicit coordinate (native, in ``node``).
        guard: A condition for the row's presence (native), or empty.
        explicit: The condition under which the row carries explicit coordinates (native; empty: always).
    """

    offset: str
    value: str = "value"
    coordinate: str = "articulation_world_dof_offset.data[art] + node"
    guard: str = ""
    explicit: str = "!support.enabled"


def _guarded(guard: str, statements: list[str], indent: str) -> list[str]:
    if not guard:
        return [indent + line for line in statements]
    return [f"{indent}if ({guard}) {{", *(f"{indent}    {line}" for line in statements), f"{indent}}}"]


def factor_rows(support: str, value: str, rows: tuple[FactorRow, ...], indent: str) -> str:
    """Pack ``rows`` over the support ``support`` (a mask of factor coordinates), lane-strided, and sum ``norm``.

    ``value`` declares ``const float value`` (the row's ``Z`` at factor coordinate ``node``; with the lookup's
    ``entry``); each row stores it at the coordinate's rank (:func:`store_entry`), then the walk pads each row past
    the support (:func:`pad_entries`). Every lane ends holding ``norm = |Z|^2``, its entries' squares summed in walk
    order and reduced with ``SUM``.
    """
    walk = [
        "float norm = 0.0f;",
        "for (int node = LANE; node < dofs; node += STRIDE) {",
        "    const unsigned long long bit = 1ull << node;",
        f"    if (({support} & bit) == 0) continue;",
        f"    const int index = COUNT({support} & (bit - 1ull));",
        *(f"    {line}" for line in value.split("\n")),
    ]
    for row in rows:
        walk += _guarded(row.guard, store_entry(row.offset, "index", row.coordinate, row.value, row.explicit), "    ")
    walk += [
        "    norm += value * value;",
        "}",
        f"const int support_count = COUNT({support});",
        "for (int index = support_count + LANE; index < support_capacity; index += STRIDE) {",
    ]
    for row in rows:
        walk += _guarded(row.guard, pad_entries(row.offset, row.explicit), "    ")
    walk += ["}", "SUM(norm);"]
    return "\n".join(indent + line for line in walk)


def _column_value(column: str, scale: str = "") -> str:
    """``const float value``: column ``column`` of ``L^-1`` at factor coordinate ``node`` (times ``scale``)."""
    factor = "inverse_factor.data[factor_base + entry]"
    return (
        f"const int entry = factor_lookup.data[node * dofs + {column}];\n"
        f"const float value = entry >= 0 ? {f'{scale} * ' if scale else ''}{factor} : 0.0f;"
    )


_GROUP_ARTICULATION = """
    const int group = tid >> 5;
    if (group >= group_to_art.shape[0]) return;
#if !defined(__CUDA_ARCH__)
    if ((tid & 31) != 0) return;
#endif
    const int art = group_to_art.data[group];
    const int world = art_to_world.data[art];
    const int start = articulation_dof_start.data[art];
    const int dofs = inverse_permutation.shape[0];
    const int capacity = ${ROWS}row_type.shape[1];
    const int line = ${ROWS}first.data[world];
    __builtin_assume(line >= 0);
    const int support_capacity = row_factor.shape[2];
    const int factor_base = group * inverse_factor.shape[1];"""
"""A warp per articulation of the factor group (``tid >> 5``; the CPU runs a warp's work on its first thread), and
the group's sizes: coordinates, row and support capacities (of the records ``$ROWS``), the articulation's factor."""


def _record(row: str, fields: dict[str, str], indent: str) -> str:
    """Write row ``row``'s dense record fields (``rows.<field>.data[row] = value``), in order."""
    return "\n".join(f"{indent}rows.{name}.data[{row}] = {value};" for name, value in fields.items())


_LIMIT_WALK = """        // Visit the ballot in lane order: the rows' slots, in the articulation's lower/upper coordinate order.
        unsigned pending = votes;
        while (pending != 0) {
            const int source = FIRST(pending);
            const int row = READ(slot, source);
            // The reservation still counts every over-capacity row.
            if (row >= capacity) break;
            const int selected_dof = READ(d0, source);
            const float sign = READ(w0, source);
            const int column = inverse_permutation.data[selected_dof];
            const unsigned long long support_mask = dof_mask.data[selected_dof];
            const int row_offset = line * capacity + row;
            const int response_offset = row_offset * support_capacity;
$WALK
            if (LANE == 0) {
                if (support.enabled)
                    support.row_class.data[row_offset] = support.limit_class.data[selected_dof];
                row_incident.data[row_offset] = sign * v_hat.data[start + selected_dof];
                law.diag.data[row_offset] = norm;
            }
            pending &= pending - 1;
        }
"""
"""A position-limit row's packed ``L^-1`` column, signed by side, and its ``|Z|^2`` diagonal: the whole warp walks
each row the chunk reserved."""

_VELOCITY_LIMIT_ROWS = """$ARTICULATION
    for (int local_dof = 0; local_dof < dofs; ++local_dof) {
        const int dof = start + local_dof;
        const int lower = velocity_limit_slot.data[2 * dof];
        const int upper = velocity_limit_slot.data[2 * dof + 1];
        // The existing allocator owns eligibility, ordering and overflow demand.
        const bool write_lower = lower >= 0 && lower < capacity;
        const bool write_upper = upper >= 0 && upper < capacity;
        if (!write_lower && !write_upper) continue;
        const float lower_sign = write_lower ? velocity_limit_sign.data[2 * dof] : 0.0f;
        const float upper_sign = write_upper ? velocity_limit_sign.data[2 * dof + 1] : 0.0f;
        const int column = inverse_permutation.data[local_dof];
        const unsigned long long support_mask = dof_mask.data[local_dof];
        const int lower_offset = (line * capacity + lower) * support_capacity;
        const int upper_offset = (line * capacity + upper) * support_capacity;
        const int dof_offset = articulation_world_dof_offset.data[art];
$WALK
        if (LANE == 0) {
            const float incident = v_hat.data[dof];
            const float target = -joint_velocity_limit.data[dof];
            for (int side = 0; side < 2; ++side) {
                const int row = side == 0 ? lower : upper;
                if (row < 0 || row >= capacity) continue;
                const float sign = side == 0 ? lower_sign : upper_sign;
                const int row_offset = line * capacity + row;
                if (support.enabled)
                    support.row_class.data[row_offset] = support.limit_class.data[local_dof];
                row_incident.data[row_offset] = sign * incident;
                // The prefix finalizer adds CFM and forms RHS from target_velocity.
$RECORD
            }
        }
    }
"""

_MIMIC_ROWS = """$ARTICULATION
    for (int m = mimic_art_start.data[art]; m < mimic_art_start.data[art + 1]; ++m) {
        const int k = mimic_art_list.data[m];
        const int row = mimic_slot.data[k];
        if (row < 0 || row >= capacity) continue;
        const int dof0 = mimic_dof0.data[k];
        const int dof1 = mimic_dof1.data[k];
        const int local0 = dof0 - start;
        const int local1 = dof1 - start;
        const int column0 = inverse_permutation.data[local0];
        const int column1 = inverse_permutation.data[local1];
        const float coefficient = mimic_coef1.data[k];
        // Keep both structural supports even when the live coefficient is zero.
        const unsigned long long support_mask = dof_mask.data[local0] | dof_mask.data[local1];
        const int row_offset = line * capacity + row;
        const int response_offset = row_offset * support_capacity;
$WALK
        if (LANE == 0) {
            row_incident.data[row_offset] = v_hat.data[dof0] - coefficient * v_hat.data[dof1];
$RECORD
        }
    }
"""

_MIMIC_VALUE = """const int entry0 = factor_lookup.data[node * dofs + column0];
const int entry1 = factor_lookup.data[node * dofs + column1];
const float value0 = entry0 >= 0 ? inverse_factor.data[factor_base + entry0] : 0.0f;
const float value1 = entry1 >= 0 ? inverse_factor.data[factor_base + entry1] : 0.0f;
const float value = value0 - coefficient * value1;"""
"""A mimic row's ``Z`` at factor coordinate ``node``: ``L^-1 (e_0 - c e_1)``."""


def _internal_row_record(kind: int, beta: str, phi: str, target: str) -> dict[str, str]:
    """An internal row's dense record: its ``|Z|^2`` as the diagonal, no parent or friction, the row's ``cfm``."""
    return {
        "diag": "norm",
        "row_type": str(kind),
        "row_parent": "-1",
        "row_mu": "0.0f",
        "row_beta": beta,
        "row_cfm": "pgs_cfm",
        "phi": phi,
        "target": target,
    }


_LIMIT_ROWS_SOURCE = kernel_template.substitute(_GROUP_ARTICULATION + "\n", ROWS="law.") + prefix_rows(
    ((limits("dofs", "start + d", "law."),),),
    law="law.",
    ordered=False,
    rhs=False,
    after=kernel_template.substitute(
        _LIMIT_WALK,
        WALK=factor_rows("support_mask", _column_value("column", "sign"), (FactorRow("response_offset"),), " " * 12),
    ),
)

_VELOCITY_LIMIT_ROWS_SOURCE = kernel_template.substitute(
    _VELOCITY_LIMIT_ROWS,
    ARTICULATION=kernel_template.substitute(_GROUP_ARTICULATION, ROWS="rows."),
    WALK=factor_rows(
        "support_mask",
        _column_value("column"),
        tuple(
            FactorRow(
                f"{side}_offset", f"entry >= 0 ? {side}_sign * value : 0.0f", "dof_offset + node", f"write_{side}"
            )
            for side in ("lower", "upper")
        ),
        " " * 8,
    ),
    RECORD=_record(
        "row_offset", _internal_row_record(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT, "0.0f", "0.0f", "target"), " " * 16
    ),
)

_MIMIC_ROWS_SOURCE = kernel_template.substitute(
    _MIMIC_ROWS,
    ARTICULATION=kernel_template.substitute(_GROUP_ARTICULATION, ROWS="rows."),
    WALK=factor_rows("support_mask", _MIMIC_VALUE, (FactorRow("response_offset", explicit=""),), " " * 8),
    RECORD=_record(
        "row_offset",
        _internal_row_record(
            PGS_CONSTRAINT_TYPE_MIMIC,
            "pgs_beta",
            "RowLaw::mimic_phi(joint_q.data[mimic_q0.data[k]], joint_q.data[mimic_q1.data[k]], mimic_coef0.data[k], "
            "coefficient)",
            "0.0f",
        ),
        " " * 12,
    ),
)


_FACTOR_GROUP = kernel_template.params(
    ("group_to_art", wp.array[int]),
    ("art_to_world", wp.array[int]),
    ("articulation_world_dof_offset", wp.array[int]),
    ("articulation_dof_start", wp.array[int]),
)
_FACTOR_COLUMNS = kernel_template.params(
    ("dof_mask", wp.array[wp.uint64]),
    ("inverse_permutation", wp.array[int]),
    ("factor_lookup", wp.array2d[int]),
    ("inverse_factor", wp.array2d[float]),
    ("v_hat", wp.array[float]),
)
_FACTOR_OUTPUTS = kernel_template.params(
    ("rows", DenseRows),
    ("row_dof", wp.array3d[int]),
    ("row_factor", wp.array3d[float]),
    ("row_incident", wp.array2d[float]),
)


def _internal_rows_kernel(name: str, params, source: str, preludes) -> wp.Kernel:
    """A warp-per-articulation internal row builder: ``32 * group_count`` threads in blocks divisible by 32."""
    return KernelTemplate(
        name,
        params,
        source,
        kernel_template.THREADS,
        preludes=(lane_group(32), *preludes),
        dual=True,
        native_name=f"_{name}",
        module=__name__,
    ).kernel()


factor_limit_rows_kernel = _internal_rows_kernel(
    "factor_limit_rows",
    _FACTOR_GROUP
    + kernel_template.params(("law", PrefixLaw))
    + _FACTOR_COLUMNS
    + _FACTOR_OUTPUTS[1:]
    + kernel_template.params(("support", _SparseRowSupport)),
    _LIMIT_ROWS_SOURCE,
    ROW_LAW.preludes(),
)
"""The position-limit rows of each articulation (:func:`.prefix_rows.prefix_rows`) and their packed columns of
``L^-1``, signed by side; a reservation past the row capacity still counts, and writes stay within it."""

factor_velocity_limit_rows_kernel = _internal_rows_kernel(
    "factor_velocity_limit_rows",
    _FACTOR_GROUP
    + kernel_template.params(
        ("joint_velocity_limit", wp.array[float]),
        ("velocity_limit_slot", wp.array[int]),
        ("velocity_limit_sign", wp.array[float]),
        ("pgs_cfm", float),
    )
    + _FACTOR_COLUMNS
    + _FACTOR_OUTPUTS
    + kernel_template.params(("support", _SparseRowSupport)),
    _VELOCITY_LIMIT_ROWS_SOURCE,
    (),
)
"""The allocated signed velocity-limit rows of each coordinate: the two sides share one column and its norm.

The allocator owns the slots, activation and capacity; infinite limits are kept, and no position bias is added
(:func:`.sparse_contact.finalize_sparse_joint_limit_rows` adds ``cfm`` and forms the ``rhs``)."""

factor_mimic_rows_kernel = _internal_rows_kernel(
    "factor_mimic_rows",
    _FACTOR_GROUP
    + kernel_template.params(
        ("mimic_slot", wp.array[int]),
        ("mimic_art_start", wp.array[int]),
        ("mimic_art_list", wp.array[int]),
        ("mimic_dof0", wp.array[int]),
        ("mimic_dof1", wp.array[int]),
        ("mimic_q0", wp.array[int]),
        ("mimic_q1", wp.array[int]),
        ("mimic_coef0", wp.array[float]),
        ("mimic_coef1", wp.array[float]),
        ("joint_q", wp.array[float]),
        ("pgs_beta", float),
        ("pgs_cfm", float),
    )
    + _FACTOR_COLUMNS
    + _FACTOR_OUTPUTS,
    _MIMIC_ROWS_SOURCE,
    ROW_LAW.preludes(),
)
"""The allocated same-articulation mimic rows: ``Z = L^-1 (e_0 - c e_1)`` over the union of both supports (kept
when the live coefficient is zero), with explicit coordinates."""


# A contact's rows (its normal, then its tangents) share their endpoints' support. A lane group per contact evaluates
# each endpoint's Jacobian once into shared scratch (``J = d . (S_lin + S_ang x (p - o))`` per coordinate, with the
# row's angular wrench about the articulation origin), with the incident velocity ``J v_hat``, then packs the rows:
# an articulation's coordinates as factor rows ``Z = L^-1 P J^T`` through the support's columns of ``L^-1``
# (:func:`.sparse_row_projection._sparse_contact_projection_source`, the factor-product law), a free body's six as
# its physical Jacobian with the response ``H^-1 J^T`` through its own inverse factor (the free body keeps its
# mass metric). ``A_ii = |Z|^2 + J H^-1 J^T + cfm``. With contact records, a full triplet also writes its block
# record for the single-stage sweep: the cross terms ``W_nt1, W_nt2, W_t1t2``, ``1 / W_nn`` and the friction pair's
# tangent metric (``ProjectionLaw::friction_pair_metric``) of the diagonals as written.

_TRIPLET_RECORD = """
    if (row_count == $ROWS) {
        SUM($CHANNELS);
        if (LANE == 0) {
            float* record = row_free_response.data
                + (line * row_free_response.shape[1] + slot / 3) * row_free_response.shape[2];
            record[0] = cross.c[0];
            record[1] = cross.c[1];
            record[2] = cross.c[2];
            record[3] = 1.0f / (norm.c[0] + cfm);
$TAIL
        }
    }
"""
"""A full triplet's block record (:data:`.sparse_contact.CONTACT_RECORD_FLOATS` floats), from the lanes' cross terms
reduced in one ladder and the diagonals as written: then the pair's metric (:data:`_PAIR_METRIC_RECORD`), or a
quad's spin couplings (:data:`_SPIN_RECORD`)."""

_PAIR_METRIC_RECORD = """            // The tangent pair's metric (ProjectionLaw::friction_pair_metric), which the triplet sweeps reuse.
            const auto metric = ProjectionLaw::friction_pair_metric(norm.c[1] + cfm, cross.c[2], norm.c[2] + cfm);
            record[4] = metric.v1;
            record[5] = metric.v2;
            record[6] = metric.v3[0];
            record[7] = metric.v3[1];"""
_SPIN_RECORD = """            // The spin row's couplings W_ns, W_t1s, W_t2s (contact torsion); the friction triple's law forms its metric.
            record[4] = spin_cross.c[0];
            record[5] = spin_cross.c[1];
            record[6] = spin_cross.c[2];"""

_CONTACT_ROWS = """
#if defined(__CUDA_ARCH__)
    __shared__ float storage[$STORAGE];
    float* jacobian = storage + SUB * $SCRATCH;
    $GEOMETRY_CUDA
#else
    if ((tid & ($LANES - 1)) != 0) return;
    float jacobian[$SCRATCH];
    $GEOMETRY_CPU
#endif
    float* free_factor = jacobian + $JACOBIAN;
    const auto jacobian_value = [&](const auto& motion, int side, int row) {
        const auto linear = wp::vec3(motion.c[0], motion.c[1], motion.c[2]);
        const auto angular = wp::vec3(motion.c[3], motion.c[4], motion.c[5]);
        ${SPIN_VALUE}$GEOMETRY_DOT
        return wp::dot(direction, linear) + wp::dot(angular_wrench, angular);
    };
    unsigned long long mask_a = 0, mask_b = 0;
    if (body_a >= 0 && art_a >= 0) mask_a = body_dof_mask.data[body_a];
    if (body_b >= 0 && art_b >= 0) mask_b = body_dof_mask.data[body_b];
    const int row_offset = line * row_factor.shape[1] + slot;
    const int support_capacity = row_factor.shape[2];
    int count = 0;
    auto norm = $VEC(0.0f);
${CROSS_DECL}    auto incident = $VEC(0.0f);
    // Physical blocks remain a short prefix; ordinary free/free contacts use
    // native MF storage, so production rows have at most one such block.
    int first_endpoint = 0;
    if (art_b >= 0 && is_free_rigid.data[art_b] != 0
        && (art_a < 0 || is_free_rigid.data[art_a] == 0)) first_endpoint = 1;
    for (int endpoint = 0; endpoint < 2; ++endpoint) {
        const int side = (first_endpoint + endpoint) % 2;
        const int art = side == 0 ? art_a : art_b;
        if (art < 0 || (side == 1 && art == art_a)
            || articulation_response_dof_count.data[art] <= 0) continue;
        const int group = art_group_idx.data[art];
        const int start = articulation_dof_start.data[art];
        const int offset = articulation_world_dof_offset.data[art];
        if (is_free_rigid.data[art] != 0) {
            for (int node = LANE; node < 6; node += STRIDE) {
                const auto motion = joint_S_s.data[start + node];
                const float velocity = v_hat.data[start + node];
#pragma unroll
                for (int row = 0; row < $ROWS; ++row) {
                    if (row >= row_count) continue;
                    float value = 0.0f;
                    if (art == art_a) value += jacobian_value(motion, 0, row);
                    if (art == art_b) value -= jacobian_value(motion, 1, row);
                    jacobian[row * $PLANE + node] = value;
                    incident.c[row] += value * velocity;
                }
            }
            SYNC();
            const int factor_base = group * 36;
            for (int node = LANE; node < 6; node += STRIDE) {
                auto value = $VEC(0.0f);
                for (int column = 0; column <= node; ++column) {
                    const float factor = free_inverse_factor.data[factor_base + node * 6 + column];
#pragma unroll
                    for (int row = 0; row < $ROWS; ++row) {
                        if (row < row_count) value.c[row] += factor * jacobian[row * $PLANE + column];
                    }
                }
#pragma unroll
                for (int row = 0; row < $ROWS; ++row) {
                    if (row < row_count) free_factor[row * $PLANE + node] = value.c[row];
                }
            }
            SYNC();
            for (int node = LANE; node < 6; node += STRIDE) {
                auto response = $VEC(0.0f);
                for (int column = node; column < 6; ++column) {
                    const float factor = free_inverse_factor.data[factor_base + column * 6 + node];
#pragma unroll
                    for (int row = 0; row < $ROWS; ++row) {
                        if (row < row_count) response.c[row] += factor * free_factor[row * $PLANE + column];
                    }
                }
#pragma unroll
                for (int row = 0; row < $ROWS; ++row) {
                    if (row >= row_count) continue;
                    const int response_offset = (row_offset + row) * support_capacity;
                    const float value = jacobian[row * $PLANE + node];
$FREE_STORE
${FREE_RESPONSE}                    norm.c[row] += value * response.c[row];
                }
${FREE_CROSS}            }
            count += 6;
        } else {
            unsigned long long mask = 0;
            if (art == art_a) mask |= mask_a;
            if (art == art_b) mask |= mask_b;
            for (int node = LANE; node < $N; node += STRIDE) {
                const unsigned long long bit = 1ull << node;
                if ((mask & bit) == 0) continue;
                const int physical = permutation.data[node];
                const auto motion = joint_S_s.data[start + physical];
                const float velocity = v_hat.data[start + physical];
#pragma unroll
                for (int row = 0; row < $ROWS; ++row) {
                    if (row >= row_count) continue;
                    float value = 0.0f;
                    if (art == art_a && (mask_a & bit) != 0)
                        value += jacobian_value(motion, 0, row);
                    if (art == art_b && (mask_b & bit) != 0)
                        value -= jacobian_value(motion, 1, row);
                    jacobian[row * $PLANE + node] = value;
                    incident.c[row] += value * velocity;
                }
            }
            SYNC();
            const int factor_base = group * inverse_factor.shape[1];
            for (int node = LANE; node < $N; node += STRIDE) {
                const unsigned long long bit = 1ull << node;
                if ((mask & bit) == 0) continue;
                $SPARSE_PROJECTION
                const int index = count + COUNT(mask & (bit - 1ull));
#pragma unroll
                for (int row = 0; row < $ROWS; ++row) {
                    if (row >= row_count) continue;
                    const int response_offset = (row_offset + row) * support_capacity;
$FACTOR_STORE
                    norm.c[row] += value.c[row] * value.c[row];
                }
${FACTOR_CROSS}            }
            count += COUNT(mask);
        }
        // The next endpoint reuses the same group-local Jacobian storage.
        SYNC();
    }
#pragma unroll
    for (int row = 0; row < $ROWS; ++row) {
        if (row >= row_count) continue;
        const int response_offset = (row_offset + row) * support_capacity;
        for (int index = count + LANE; index < support_capacity; index += STRIDE) {
$PAD
        }
        SUM(norm.c[row], incident.c[row]);
        if (LANE == 0) {
            if ($SUPPORT_TABLE) {
                const int class_a = mask_a != 0ull && art_a >= 0 && articulation_response_dof_count.data[art_a] > 0
                    ? support.body_class.data[body_a] : 0;
                const int class_b = mask_b != 0ull && art_b >= 0 && articulation_response_dof_count.data[art_b] > 0
                    ? support.body_class.data[body_b] : 0;
                support.row_class.data[row_offset + row] = support.pair_class.data[
                    class_a * support.pair_class.shape[1] + class_b];
            }
            row_incident.data[row_offset + row] = incident.c[row];
            diagonal.data[row_offset + row] = norm.c[row] + cfm;
        }
    }
${RECORD}"""

_GEOMETRY_CUDA = """
    auto angular_a_shared = angular_a;
    auto angular_b_shared = angular_b;
    auto directions_shared = directions;
#pragma unroll
    for (int row = 0; row < 3; ++row) {
        if (row >= row_count) continue;
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            angular_a_shared.data[row][k] = READ(angular_a.data[row][k], 0);
            angular_b_shared.data[row][k] = READ(angular_b.data[row][k], 0);
            directions_shared.data[row][k] = READ(directions.data[row][k], 0);
        }
    }
"""
_GEOMETRY_CPU = """
    const auto angular_a_shared = angular_a;
    const auto angular_b_shared = angular_b;
    const auto directions_shared = directions;
"""
_GEOMETRY_DOT = """
        const auto& angular_wrenches = side == 0 ? angular_a_shared : angular_b_shared;
        const auto direction = wp::vec3(directions_shared.data[row][0],
            directions_shared.data[row][1], directions_shared.data[row][2]);
        const auto angular_wrench = wp::vec3(angular_wrenches.data[row][0],
            angular_wrenches.data[row][1], angular_wrenches.data[row][2]);
"""
# With prepared rows the geometry is shared by the contact, not live in every lane throughout the factor
# contraction: volatile reads keep loop-invariant loads from restoring the matrix triplet to lane registers.
_GEOMETRY_COPY = """
#pragma unroll
        for (int row = 0; row < 3; ++row) {
            if (row >= row_count) continue;
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                geometry[row * 9 + k] = directions.data[row][k];
                geometry[row * 9 + 3 + k] = angular_a.data[row][k];
                geometry[row * 9 + 6 + k] = angular_b.data[row][k];
            }
        }
"""
_PREPARED_GEOMETRY_DOT = """
        const int angular_offset = row * 9 + 3 + side * 3;
        const auto direction = wp::vec3(geometry[row * 9], geometry[row * 9 + 1], geometry[row * 9 + 2]);
        const auto angular_wrench = wp::vec3(geometry[angular_offset],
            geometry[angular_offset + 1], geometry[angular_offset + 2]);
"""


def _contact_rows_source(
    n: int, lanes: int, warps: int, prepare_rows: bool, table: bool, records: bool, torsion: bool
) -> str:
    """The native projection of one contact's rows (see the section's comment), for ``lanes`` lanes per contact; under
    contact ``torsion`` a fourth row, its spin row ``J = n . S_ang``."""
    groups = warps * 32 // lanes
    scratch = max(n, 6) + 6
    geometry = 27 if prepare_rows else 0
    rows = 4 if torsion else 3
    # The record's Gram couplings: (channel, its component, the two rows).
    couplings = [("cross", k, a, b) for k, (a, b) in enumerate(((0, 1), (0, 2), (1, 2)))]
    if torsion:
        couplings += [("spin_cross", a, a, 3) for a in range(3)]
    explicit = f"!{str(table).lower()}"
    free_store = store_entry("response_offset", "count + node", "offset + node", "value", explicit)
    factor_store = store_entry("response_offset", "index", "offset + node", "value.c[row]", explicit)
    cross = {
        "CROSS_DECL": "",
        "FREE_RESPONSE": "                    row_free_response.data[(row_offset + row) * row_free_response.shape[2]"
        " + count + node] = response.c[row];\n",
        "FREE_CROSS": "",
        "FACTOR_CROSS": "",
        "RECORD": "",
    }
    if records:

        def sums(entry) -> str:
            lines = (f"                    {name}.c[{k}] += {entry(a, b)};" for name, k, a, b in couplings)
            return f"                if (row_count == {rows}) {{\n" + "\n".join(lines) + "\n                }\n"

        free = lambda a, b: f"{'jacobian[node]' if a == 0 else f'jacobian[{a * scratch} + node]'} * response.c[{b}]"  # noqa: E731
        cross = {
            "CROSS_DECL": "    auto cross = wp::vec3(0.0f);\n"
            + ("    auto spin_cross = wp::vec3(0.0f);\n" if torsion else ""),
            "FREE_RESPONSE": "",
            "FREE_CROSS": sums(free),
            "FACTOR_CROSS": sums(lambda a, b: f"value.c[{a}] * value.c[{b}]"),
            "RECORD": kernel_template.substitute(
                _TRIPLET_RECORD,
                ROWS=rows,
                CHANNELS=", ".join(f"{name}.c[{k}]" for name, k, _, _ in couplings),
                TAIL=_SPIN_RECORD if torsion else _PAIR_METRIC_RECORD,
            ),
        }
    geometry_cuda, geometry_cpu, geometry_dot = _GEOMETRY_CUDA, _GEOMETRY_CPU, _GEOMETRY_DOT
    normal = "directions_shared.data[0][0], directions_shared.data[0][1], directions_shared.data[0][2]"
    if prepare_rows:
        geometry_cuda = (
            f"volatile float* geometry = jacobian + {rows * scratch};\nif (LANE == 0) {{\n"
            + _GEOMETRY_COPY
            + "}\nSYNC();\n"
        )
        geometry_cpu = "float geometry[27];\n" + _GEOMETRY_COPY
        geometry_dot = _PREPARED_GEOMETRY_DOT
        normal = "geometry[0], geometry[1], geometry[2]"
    # The spin row turns about the normal, row 0's direction (:func:`.contact_laws.spin_entry`).
    spin = f"if (row == 3) return wp::dot(wp::vec3({normal}), angular);" if torsion else ""
    return kernel_template.substitute(
        _CONTACT_ROWS,
        SPARSE_PROJECTION=_sparse_contact_projection_source(
            "shared_mask", dof_count=n, scratch_stride=scratch, rows=rows
        ),
        GEOMETRY_CUDA=geometry_cuda,
        GEOMETRY_CPU=geometry_cpu,
        GEOMETRY_DOT=geometry_dot,
        SPIN_VALUE=spin,
        ROWS=rows,
        VEC=f"wp::vec{rows}",
        STORAGE=groups * (rows * scratch + geometry),
        LANES=lanes,
        SCRATCH=rows * scratch + geometry,
        PLANE=scratch,
        JACOBIAN=max(n, 6),
        N=n,
        SUPPORT_TABLE=str(table).lower(),
        FREE_STORE="\n".join(" " * 20 + line for line in free_store),
        FACTOR_STORE="\n".join(" " * 20 + line for line in factor_store),
        PAD="\n".join(" " * 12 + line for line in pad_entries("response_offset", explicit)),
        **cross,
    )


@cache
def factor_contact_rows_kernel(
    dof_count: int,
    *,
    lanes_per_contact: int = 8,
    warps_per_block: int = 4,
    prepare_rows: bool = False,
    use_support_table: bool = False,
    contact_records: bool = False,
    torsion: bool = False,
) -> wp.Kernel:
    """Pack each contact's rows in factor coordinates (see the section's comment), a lane group per contact.

    The contact front end (:mod:`.contact_laws`) gives each row its frame and each endpoint's angular wrench; with
    ``prepare_rows`` it also writes the rows' records and right-hand sides by the row laws (bias, ``cfm`` and, after
    the projection, a restitutive normal's rebound from its incident velocity). ``use_support_table`` publishes each
    row's support class instead of its coordinates (one factored articulation per world, no free-body response);
    ``contact_records`` writes each full triplet's block record into ``row_free_response`` (it needs both). Under contact
    ``torsion`` a friction contact's fourth row is its spin row (:func:`.row_build.write_contact_row`'s), and a full
    quad's record holds its spin couplings in place of the pair's metric. A grid of
    workers strides over the contacts: ``total_num_workers * lanes_per_contact`` threads in blocks of
    ``warps_per_block * 32``; the CPU runs a contact's group on one thread. Arrays must be contiguous.
    """
    n, lanes, warps = int(dof_count), int(lanes_per_contact), int(warps_per_block)
    if not 1 <= n <= 64 or lanes not in (8, 16, 32) or not 1 <= warps <= 32:
        raise ValueError("Invalid sparse contact dimensions or lane group")
    if contact_records and not (prepare_rows and use_support_table):
        raise ValueError("Contact records require prepared rows and the row support table")
    snippet = dual_native(
        _contact_rows_source(n, lanes, warps, prepare_rows, use_support_table, contact_records, torsion),
        preludes=(lane_group(lanes), *PROJECTION_LAW.preludes()),
    )

    @wp.func_native(snippet)
    def project_native(
        tid: int,
        body_a: int,
        body_b: int,
        art_a: int,
        art_b: int,
        world: int,
        line: int,
        slot: int,
        row_count: int,
        angular_a: wp.mat33,
        angular_b: wp.mat33,
        directions: wp.mat33,
        art_group_idx: wp.array[int],
        articulation_response_dof_count: wp.array[int],
        is_free_rigid: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        articulation_dof_start: wp.array[int],
        body_dof_mask: wp.array[wp.uint64],
        joint_S_s: wp.array[wp.spatial_vector],
        permutation: wp.array[int],
        factor_row_offsets: wp.array[int],
        inverse_factor: wp.array2d[float],
        free_inverse_factor: wp.array3d[float],
        v_hat: wp.array[float],
        cfm: float,
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_free_response: wp.array3d[float],
        row_incident: wp.array2d[float],
        diagonal: wp.array2d[float],
        support: _SparseRowSupport,
    ): ...

    def contact_kernel(
        data: ContactInput,
        total_num_workers: int,
        art_group_idx: wp.array[int],
        is_free_rigid: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        body_dof_mask: wp.array[wp.uint64],
        law: PatchRowLaw,
        permutation: wp.array[int],
        factor_row_offsets: wp.array[int],
        inverse_factor: wp.array2d[float],
        free_inverse_factor: wp.array3d[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_free_response: wp.array3d[float],
        row_incident: wp.array2d[float],
        support: _SparseRowSupport,
    ):
        tid = wp.tid()
        lane = tid % lanes
        worker = tid // lanes
        total_contacts = data.count[0]
        for c in range(worker, total_contacts, total_num_workers):
            row_count = wp.min(data.slots_needed[c], wp.static(3 + int(torsion)))
            if data.path[c] != 0 or data.slot[c] < 0 or row_count <= 0:
                continue
            art_a = data.art0[c]
            art_b = data.art1[c]
            body_a, body_b, normal, point, gap, tangent0, tangent1, friction_point = contact_frame(data, law, c)
            cfm = float(0.0)
            if wp.static(prepare_rows):
                cfm = data.cfm
            if wp.static(prepare_rows):
                if lane < row_count:
                    world = data.world[c]
                    line = data.first[world]
                    slot = data.slot[c] + lane
                    direction, row_point = contact_row_frame(lane, normal, tangent0, tangent1, point, friction_point)
                    target = prescribed_target(
                        body_a,
                        art_a,
                        body_b,
                        art_b,
                        row_point,
                        direction,
                        data.prescribed,
                        data.origin,
                        data.body_v,
                    )
                    mu = contact_friction(data.shape0[c], data.shape1[c], data.material_mu)
                    beta = data.beta
                    phi = gap
                    row_type = PGS_CONSTRAINT_TYPE_CONTACT
                    parent = int(-1)
                    restitution = float(0.0)
                    stiffness = float(0.0)
                    if data.material != 0:
                        stiffness = data.stiffness[c]
                    if lane == 0:
                        beta = contact_normal_beta(data, c, body_a, body_b, phi)
                        if stiffness <= 0.0:
                            restitution = contact_restitution(data.shape0[c], data.shape1[c], data.material_restitution)
                    else:
                        beta = law.beta
                        # Selected, not indexed by the lane, which would place the pair in local memory.
                        anchor_phi = law.patches.phi[c]
                        phi = anchor_phi[0]
                        if lane == 2:
                            phi = anchor_phi[1]
                        if stiffness > 0.0 and data.friction_weight[c] > 0.0:
                            mu *= data.friction_weight[c]
                        row_type = PGS_CONSTRAINT_TYPE_FRICTION
                        parent = data.slot[c]
                    if wp.static(torsion):
                        if lane == CONTACT_SPIN_ROW:
                            # The spin row: the torsional coefficient, its rhs minus the prescribed bodies' spin.
                            target = prescribed_spin_target(
                                body_a, art_a, body_b, art_b, normal, data.prescribed, data.body_v
                            )
                            mu = contact_torsion_mu(data, c)
                            row_type = PGS_CONSTRAINT_TYPE_TORSION
                    # The position pass's row law (bias scale 1) and its regularization weight; a material normal's
                    # own law.
                    rhs, row_w = row_rhs(
                        row_type, phi, beta, target, 1.0 / data.dt, 1.0, data.speculative_scale, 0.0, data.contact_w
                    )
                    if data.contact_w < 1.0:
                        data.row_w[line, slot] = row_w
                    if data.material != 0:
                        eps = float(0.0)
                        if lane == 0 and stiffness > 0.0:
                            rhs, eps = contact_material_rhs(phi, target, stiffness, data.damping[c], data.dt)
                        data.row_eps[line, slot] = eps
                    data.row_type[line, slot] = row_type
                    data.row_parent[line, slot] = parent
                    data.row_mu[line, slot] = mu
                    # Bias and CFM are already applied; only normal restitution reads row records after the
                    # response projection.
                    if lane == 0:
                        data.phi[line, slot] = phi
                        data.target[line, slot] = target
                        data.restitution[line, slot] = restitution
                    data.rhs[line, slot] = rhs

            # Fixed component constructors avoid materializing dynamically indexed matrices in local memory during
            # geometry assembly.
            angular_a = wp.mat33(0.0)
            angular_b = wp.mat33(0.0)
            directions = wp.mat33(0.0)
            if lane == 0:
                normal_a = wp.vec3(0.0)
                normal_b = wp.vec3(0.0)
                tangent0_a = wp.vec3(0.0)
                tangent1_a = wp.vec3(0.0)
                tangent0_b = wp.vec3(0.0)
                tangent1_b = wp.vec3(0.0)
                # Each row's wrench about an endpoint's articulation origin: the angular part of the free-body
                # Jacobian [d, (p - o) x d], whose product with a motion column is that column's Jacobian entry.
                if art_a >= 0 and data.response_dofs[art_a] > 0:
                    origin_a = data.origin[art_a]
                    normal_a = wp.spatial_bottom(contact_seed(normal, point - origin_a))
                    if row_count > 1:
                        tangent0_a = wp.spatial_bottom(contact_seed(tangent0, friction_point - origin_a))
                        tangent1_a = wp.spatial_bottom(contact_seed(tangent1, friction_point - origin_a))
                if art_b >= 0 and data.response_dofs[art_b] > 0:
                    origin_b = data.origin[art_b]
                    normal_b = wp.spatial_bottom(contact_seed(normal, point - origin_b))
                    if row_count > 1:
                        tangent0_b = wp.spatial_bottom(contact_seed(tangent0, friction_point - origin_b))
                        tangent1_b = wp.spatial_bottom(contact_seed(tangent1, friction_point - origin_b))
                angular_a = wp.matrix_from_rows(normal_a, tangent0_a, tangent1_a)
                angular_b = wp.matrix_from_rows(normal_b, tangent0_b, tangent1_b)
                directions = wp.matrix_from_rows(normal, tangent0, tangent1)
            project_native(
                tid,
                body_a,
                body_b,
                art_a,
                art_b,
                data.world[c],
                data.first[data.world[c]],
                data.slot[c],
                row_count,
                angular_a,
                angular_b,
                directions,
                art_group_idx,
                data.response_dofs,
                is_free_rigid,
                articulation_world_dof_offset,
                data.dof_start,
                body_dof_mask,
                data.motion,
                permutation,
                factor_row_offsets,
                inverse_factor,
                free_inverse_factor,
                data.incident,
                cfm,
                row_dof,
                row_factor,
                row_free_response,
                row_incident,
                data.diag,
                support,
            )
            if wp.static(prepare_rows):
                if lane == 0:
                    world = data.world[c]
                    line = data.first[world]
                    slot = data.slot[c]
                    # Only a restitutive contact reads its row back for the rebound law.
                    row_restitution = data.restitution[line, slot]
                    if row_restitution > 0.0:
                        target = data.target[line, slot]
                        fires, bounce = contact_restitution_rhs(
                            data.phi[line, slot],
                            row_incident[line, slot] - target,
                            target,
                            row_restitution,
                            data.dt,
                            data.restitution_threshold,
                        )
                        if fires:
                            data.rhs[line, slot] = bounce
                            # An impact is impulsive, not a spring: keep the rebound exact.
                            if data.contact_w < 1.0:
                                data.row_w[line, slot] = 1.0

    name = f"factor_contact_rows_{n}_{lanes}_{warps}"
    if prepare_rows:
        name += "_prepared"
    if use_support_table:
        name += "_support_table"
    if contact_records:
        name += "_contact_records"
    if torsion:
        name += "_torsion"
    return python_kernel(contact_kernel, name)


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
    for (int col = lane; col < $N; col += stride) {
        const int drive = drive_row_by_dof.data[start + indices.permutation.data[col]];
        if (drive < 0) continue;
        const int diagonal = indices.row_offsets.data[col + 1] - 1;
        float pivot = row_H.data[drive];
        for (int entry = indices.row_offsets.data[col]; entry < diagonal; ++entry) pivot -= factor[entry] * factor[entry];
        row_P.data[drive] = pivot;
    }
    $INVERSE
"""
"""The sparse factor of one tree, ``L^-1`` included, and each drive's pivot without its ``K`` (``row_P``: its ``H_ii`` less
the squares its pivot subtracts after adding ``K``, so that pivot keeps its order): the caller binds ``indices``,
``status``, ``geometric``, ``R_group``, ``drive_row_by_dof``, ``row_K``, ``row_H``, ``row_P``, ``factor`` and ``inverse``,
and synchronizes after it."""


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
            ("row_P", wp.array[float]),
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
    """The build and publication of factor rows (:class:`.response_operator.SparseFactor`).

    Attributes:
        factor: The factory of the factor and its inverse (:func:`sparse_factor_kernel`).
        contact_rows: The factory of the contact rows' build (:func:`factor_contact_rows_kernel`).
        limit_rows: Position-limit rows (:data:`factor_limit_rows_kernel`).
        velocity_limit_rows: Velocity-limit rows (:data:`factor_velocity_limit_rows_kernel`).
        mimic_rows: Mimic rows (:data:`factor_mimic_rows_kernel`).
        publish: ``v = v_hat + P^T L^-T u`` of a world's factor-coordinate velocity change (:func:`factor_publish_kernel`).
    """

    factor: Callable[..., wp.Kernel]
    contact_rows: Callable[..., wp.Kernel]
    limit_rows: wp.Kernel
    velocity_limit_rows: wp.Kernel
    mimic_rows: wp.Kernel
    publish: wp.Kernel


SPARSE_FACTOR = SparseFactorBuild(
    factor=sparse_factor_kernel,
    contact_rows=factor_contact_rows_kernel,
    limit_rows=factor_limit_rows_kernel,
    velocity_limit_rows=factor_velocity_limit_rows_kernel,
    mimic_rows=factor_mimic_rows_kernel,
    publish=factor_publish_kernel,
)
"""The factor rows' build (:meth:`.SparseFactor.build`)."""


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
