# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The audit: how far each world's solved step stands from the problem its rows state.

A row states ``r_i = J_i v+ + b_i + eps_i lambda_i`` with ``v+`` the published velocity. The audit (``Diagnostics.audit``, off
by default, reading what the step already holds: it steers no route and changes no computed value) reduces the rows
to three numbers per world:

* the energy gap ``G = sum_rows |lambda_i r_i| + sum_blocks |lambda_c . r_c + N_c |C r_c||`` [J], over the rows and
  each contact's friction block, ``C = diag(mu, mu, mu_t)`` and ``N_c`` the block's normal load (a region's pooled
  load, with the anchor's share in ``mu``). Where ``r`` is dual-feasible every term is non-negative, so this is the
  Fenchel gap of the row laws, zero exactly at their solution. Each term counts in magnitude because an unconverged
  solve is not dual-feasible: its closing rows are negative and a signed sum cancels, so it could not rank two solves;
* the dual violation ``D = max_i max(-r_i, 0)`` on unilateral rows, ``|r_i|`` on bilateral ones [m/s];
* the injected work ``W = -sum_rows lambda_i b_i`` [J], what the rows' biases put in.

For any ``lambda`` with ``v+ = v_hat + M^-1 J^T lambda``,
``sum lambda_i r_i = dT + sum lambda_i b_i + sum eps_i lambda_i^2 + |Z^T lambda|^2 / 2``, so ``G`` bounds the energy the
solve failed to account for. It needs no reference, and it cannot catch a wrong law: ``G = 0`` says the stated rows were
solved, not that they are right.

Each row store fills the rows' ``J_i v+ + eps_i lambda_i`` (:func:`packed_rows`, :func:`sparse_rows`,
:func:`diagonal_rows`, :func:`free_rows`), and one kernel applies the law to the stores (:func:`audit_world`). Packed
rows are read as ``Z_i^T u+`` with ``u+`` the walk's coordinates of ``v+`` (:func:`.factor.walk.walk_enter_kernel`) and
the sparse diagonal's likewise, so a defect in leaving factor coordinates shows too, and corrected rows and targets agree
on a closure. Native sparse rows hold no dense factor and
read their own ``J_i v_hat + Z_i^T du``. A world whose rows exist only inside its sweep (keyed rows) has
none to read: it reports NaN and is not summed.
"""

from __future__ import annotations

import warp as wp

from .factor.walk import WalkTrees, walk_enter_kernel
from .field_group import Field, FieldGroup, SolverArrays
from .friction_patches import patch_normal_load
from .packed_rows import WIDE
from .row_laws import (
    ROW_CONTACT,
    ROW_FRICTION,
    ROW_LINK,
    ROW_MIMIC,
    ROW_SPEED,
    ROW_TORSION,
)
from .row_records import DenseRows, MfRows, RowSupport
from .row_records import bind as bind_dense_rows
from .row_records import bind_mf as bind_mf_rows
from .sparse_mass_matrix import _SparseRowSupport

__all__ = ["AUDIT_FIELDS", "AUDIT_SLOTS", "Audit"]

AUDIT_SLOTS = ("gap", "violation", "work", "gap_sum", "violation_max", "work_sum", "steps")
"""The per-world readout: the last step's ``G``, ``D`` and ``W``, and over the read steps since construction the sum of
``G``, the largest ``D``, the sum of ``W`` and their count, so one read of a captured graph covers a run."""

AUDIT_FIELDS = FieldGroup(
    Field("readout", wp.float32, ("world", len(AUDIT_SLOTS))),
    Field("coordinates", wp.float32, ("world", "world_dof")),
    Field("velocity", wp.float32, ("world", "row")),
    Field("mf_velocity", wp.float32, ("world", "mf_row")),
    Field("fused", wp.vec3, ("world",)),
)
"""The readout and the kernels' scratch: each world's coordinates in the form its rows read, each row's ``J_i v+ +
eps_i lambda_i``, and the ``(G, D, W)`` of the rows no store holds, the sparse diagonal's fused limits."""


@wp.struct
class AuditRows:
    """What every audit kernel reads: the dense rows, their impulses and counts, the free-body counts, the skips of the
    rows' build (:func:`.packed_rows._skipped`) and whether keyed rows hold the tiers' rows, and the laws in play."""

    rows: DenseRows
    impulses: wp.array2d[float]
    counts: wp.array[int]
    mf_counts: wp.array[int]
    skip_rows_le: int
    keyed: int
    weighted: int
    material: int
    capacity: int


@wp.func
def row_compliance(kind: int, weight: float, eps: float, diagonal: float) -> float:
    """A row's ``eps_i``: a material contact's, else the relative regularization's ``g d`` from its weight ``1 / (1 +
    g)``; the sweeps regularize contact normals only (:func:`.sweep.row_weight`)."""
    if kind != ROW_CONTACT:
        return 0.0
    if eps > 0.0:
        return eps
    return (1.0 / weight - 1.0) * diagonal


@wp.func
def with_compliance(a: AuditRows, world: int, i: int, term: float) -> float:
    """A dense row's velocity term ``term`` plus ``eps_i lambda_i``."""
    line = a.rows.first[world]
    weight = float(1.0)
    if a.weighted != 0:
        weight = a.rows.row_w[line, i]
    eps = float(0.0)
    if a.material != 0:
        eps = a.rows.row_eps[line, i]
    return term + row_compliance(a.rows.row_type[line, i], weight, eps, a.rows.diag[line, i]) * a.impulses[line, i]


@wp.func
def tier_world(a: AuditRows, world: int) -> bool:
    """Whether the parallel tiers form ``world``'s rows in their kernel."""
    return a.skip_rows_le > 0 and a.counts[world] <= a.skip_rows_le and a.mf_counts[world] == 0


@wp.func
def packed_dot(support: RowSupport, entries: wp.array3d[float], x: wp.array2d[float], world: int, i: int) -> float:
    """One packed row against its world's coordinates, side by side (:mod:`.packed_rows`)."""
    out = float(0.0)
    line = support.first[world]
    if support.dense == WIDE:
        for e in range(entries.shape[2]):
            if e < x.shape[1]:
                out += entries[line, i, e] * x[world, e]
        return out
    sides = support.sides[line, i]
    first = int(0)
    for side in range(2):
        art = sides[side]
        if art >= 0:
            n = support.dofs[art]
            offset = support.offset[art]
            at = first
            if support.dense != 0:
                at = offset
            for e in range(n):
                out += entries[line, i, at + e] * x[world, offset + e]
            first += n
    return out


@wp.kernel(enable_backward=False)
def packed_rows(
    a: AuditRows,
    support: RowSupport,
    J: wp.array3d[float],
    Z: wp.array3d[float],
    coordinates: wp.array2d[float],
    hat: wp.array2d[float],
    incident: wp.array2d[float],
    velocity: wp.array2d[float],
):
    """One thread per world: each packed row's ``J_i v+ + eps_i lambda_i``."""
    world = wp.tid()
    keeps = tier_world(a, world)
    for i in range(wp.min(a.counts[world], a.capacity)):
        term = float(0.0)
        if keeps:
            term = packed_dot(support, J, coordinates, world, i)
        else:
            term = (
                incident[a.rows.first[world], i]
                + packed_dot(support, Z, coordinates, world, i)
                - packed_dot(support, Z, hat, world, i)
            )
        velocity[world, i] = with_compliance(a, world, i, term)


@wp.kernel(enable_backward=False)
def sparse_rows(
    a: AuditRows,
    support: _SparseRowSupport,
    row_dof: wp.array3d[int],
    row_factor: wp.array3d[float],
    row_incident: wp.array2d[float],
    delta: wp.array2d[float],
    velocity: wp.array2d[float],
):
    """One thread per world: each native sparse row's ``J_i v_hat + Z_i^T du + eps_i lambda_i``, entries through the
    shared support table or explicit coordinates (:func:`.sweep_routes_sparse.factor_operator`)."""
    world = wp.tid()
    line = a.rows.first[world]
    for i in range(wp.min(a.counts[world], a.capacity)):
        term = row_incident[line, i]
        for e in range(row_factor.shape[2]):
            coordinate = int(-1)
            if support.enabled != 0:
                coordinate = support.class_dof[support.row_class[line, i], e]
            else:
                coordinate = row_dof[line, i, e]
            if coordinate >= 0:
                term += row_factor[line, i, e] * delta[world, coordinate]
        velocity[world, i] = with_compliance(a, world, i, term)


@wp.kernel(enable_backward=False)
def diagonal_rows(
    a: AuditRows,
    world_dof_indices: wp.array2d[int],
    dense_offsets: wp.array[int],
    dense_groups: wp.array[int],
    dense_dofs: int,
    coordinates: wp.array2d[float],
    dense_Z: wp.array3d[float],
    sparse_row_dof: wp.array3d[int],
    sparse_row_z: wp.array3d[float],
    inverse_mass: wp.array[float],
    v: wp.array[float],
    velocity: wp.array2d[float],
):
    """One thread per world: each sparse-diagonal row's ``Z_i^T u+ + eps_i lambda_i``, ``u`` on the dense part the walk's
    coordinates of ``v`` (``coordinates``, :func:`.tree_factor.walk_enter_kernel`, the frame its rows were whitened in)
    and ``u = v / s``, ``s = m^-1/2``, on a slot's coordinate."""
    world = wp.tid()
    offset = dense_offsets[world]
    group = dense_groups[world]
    line = a.rows.first[world]
    for i in range(wp.min(a.counts[world], a.capacity)):
        term = float(0.0)
        for c in range(dense_dofs):
            term += dense_Z[group, i, c] * coordinates[world, offset + c]
        for slot in range(2):
            world_dof = sparse_row_dof[line, i, slot]
            if world_dof >= 0:
                dof = world_dof_indices[world, world_dof]
                if dof >= 0:
                    scale = wp.sqrt(inverse_mass[dof])
                    if scale > 0.0:
                        term += sparse_row_z[line, i, slot] * (v[dof] / scale)
        velocity[world, i] = with_compliance(a, world, i, term)


@wp.kernel(enable_backward=False)
def fused_limits(
    active: wp.array2d[int],
    lower_rhs: wp.array2d[float],
    upper_rhs: wp.array2d[float],
    lower_lambda: wp.array2d[float],
    upper_lambda: wp.array2d[float],
    world_dof_indices: wp.array2d[int],
    v: wp.array[float],
    fused: wp.array[wp.vec3],
):
    """One thread per world: the ``(G, D, W)`` of the sparse diagonal's fused limits, unilateral rows ``+-v + rhs`` on
    a diagonal coordinate (:func:`.sweep_routes_diag._diagonal_limits`) that no store holds."""
    world = wp.tid()
    total = wp.vec3()
    for d in range(active.shape[1]):
        sides = active[world, d]
        dof = world_dof_indices[world, d]
        if sides != 0 and dof >= 0:
            if (sides & 1) != 0:
                total = limit_row(total, lower_lambda[world, d], v[dof], lower_rhs[world, d])
            if (sides & 2) != 0:
                total = limit_row(total, upper_lambda[world, d], -v[dof], upper_rhs[world, d])
    fused[world] = total


@wp.func
def limit_row(total: wp.vec3, impulse: float, term: float, bias: float) -> wp.vec3:
    """``total``'s ``(G, D, W)`` with one more unilateral row."""
    residual = term + bias
    return wp.vec3(total[0] + wp.abs(impulse * residual), wp.max(total[1], -residual), total[2] - impulse * bias)


@wp.func
def walked_side(
    Z: wp.array3d[float],
    line: int,
    row: int,
    side: int,
    art: int,
    world: int,
    world_offset: wp.array[int],
    coordinates: wp.array2d[float],
) -> float:
    """``Z_side^T u``: one side (its six entries from ``side`` of the row's twelve; articulation ``art``, -1 for none)
    of a walked free-body row over the walk's coordinates of the published velocity
    (:func:`.tree_factor.walk_enter_kernel`)."""
    if art < 0:
        return 0.0
    start = world_offset[art]
    out = float(0.0)
    for k in range(6):
        out += Z[line, row, side + k] * coordinates[world, start + k]
    return out


@wp.kernel(enable_backward=False)
def free_rows(
    a: AuditRows,
    mf: MfRows,
    walk: WalkTrees,
    impulses: wp.array2d[float],
    weights: wp.array2d[float],
    v: wp.array[float],
    world_offset: wp.array[int],
    coordinates: wp.array2d[float],
    velocity: wp.array2d[float],
):
    """One thread per world: each free-body row's ``J_i v+ + eps_i lambda_i``, its sides over the walk's coordinates
    (``coordinates``, by world coordinate). A speed row's term is ``-|v|`` of the coordinates its unwhitened axis marks, linear from ``e_0`` or angular from ``e_3``,
    its bias the limit (:func:`.projection_laws.speed_row`)."""
    world = wp.tid()
    line = mf.first[world]
    for i in range(a.mf_counts[world]):
        # Row ``i``'s ``Z`` there, its records at ``mf.base + i`` of the world's line.
        r = mf.base + i
        kind = mf.row_type[line, r]
        sides = mf.sides[line, r]
        if kind == ROW_SPEED:
            start = walk.dof_start[sides[0]]
            if mf.Z[line, i, 3] != 0.0:
                start += 3
            velocity[world, i] = -wp.length(wp.vec3(v[start], v[start + 1], v[start + 2]))
            continue
        term = walked_side(mf.Z, line, i, 0, sides[0], world, world_offset, coordinates)
        term += walked_side(mf.Z, line, i, 6, sides[1], world, world_offset, coordinates)
        diagonal = mf.diag[line, r]
        if diagonal > 0.0:
            weight = float(1.0)
            if a.weighted != 0:
                weight = weights[line, r]
            eps = float(0.0)
            if a.material != 0:
                eps = mf.eps[line, r]
            term += row_compliance(kind, weight, eps, diagonal) * impulses[line, r]
        velocity[world, i] = term


@wp.func
def audit_store(
    world: int,
    line: int,
    base: int,
    count: int,
    kinds: wp.array2d[int],
    parents: wp.array2d[int],
    mus: wp.array2d[float],
    impulses: wp.array2d[float],
    rhs: wp.array2d[float],
    velocity: wp.array2d[float],
) -> wp.vec3:
    """One row class's ``(G, D, W)``: its ``count`` rows from ``base`` of the world's line, row ``i``'s velocity term
    at ``velocity[world, i]``. A contact's friction block is its pair after its normal, then its spin row if any: its
    first tangent reads it whole, and its other rows add only their work."""
    gap = float(0.0)
    violation = float(0.0)
    work = float(0.0)
    for i in range(count):
        r = base + i
        kind = kinds[line, r]
        impulse = impulses[line, r]
        bias = rhs[line, r]
        residual = velocity[world, i] + bias
        work -= impulse * bias
        if kind == ROW_FRICTION or kind == ROW_TORSION:
            if kind == ROW_FRICTION and parents[line, r] == r - 1:
                second = velocity[world, i + 1] + rhs[line, r + 1]
                spin = float(0.0)
                spin_scale = float(0.0)
                term = impulse * residual + impulses[line, r + 1] * second
                if i + 2 < count:
                    if kinds[line, r + 2] == ROW_TORSION:
                        spin = velocity[world, i + 2] + rhs[line, r + 2]
                        spin_scale = mus[line, r + 2]
                        term += impulses[line, r + 2] * spin
                scaled = wp.vec3(mus[line, r] * residual, mus[line, r + 1] * second, spin_scale * spin)
                term += patch_normal_load(parents, impulses, line, r - 1) * wp.length(scaled)
                gap += wp.abs(term)
            continue
        gap += wp.abs(impulse * residual)
        if kind == ROW_MIMIC or kind == ROW_LINK:
            violation = wp.max(violation, wp.abs(residual))
        else:
            violation = wp.max(violation, -residual)
    return wp.vec3(gap, violation, work)


@wp.kernel(enable_backward=False)
def audit_world(
    a: AuditRows,
    mf: MfRows,
    mf_impulses: wp.array2d[float],
    velocity: wp.array2d[float],
    mf_velocity: wp.array2d[float],
    fused: wp.array[wp.vec3],
    readout: wp.array2d[float],
):
    """One thread per world: the row laws of both stores over the rows' velocity terms, into :data:`AUDIT_SLOTS`."""
    world = wp.tid()
    if tier_world(a, world) and a.keyed != 0:
        readout[world, 0] = wp.nan
        readout[world, 1] = wp.nan
        readout[world, 2] = wp.nan
        return
    rows = a.rows
    count = wp.min(a.counts[world], a.capacity)
    dense = audit_store(
        world, rows.first[world], 0, count, rows.row_type, rows.row_parent, rows.row_mu, a.impulses, rows.rhs, velocity
    )
    free = audit_store(
        world,
        mf.first[world],
        mf.base,
        a.mf_counts[world],
        mf.row_type,
        mf.row_parent,
        mf.row_mu,
        mf_impulses,
        mf.rhs,
        mf_velocity,
    )
    extra = fused[world]
    gap = dense[0] + free[0] + extra[0]
    violation = wp.max(wp.max(dense[1], free[1]), extra[1])
    work = dense[2] + free[2] + extra[2]
    readout[world, 0] = gap
    readout[world, 1] = violation
    readout[world, 2] = work
    readout[world, 3] += gap
    readout[world, 4] = wp.max(readout[world, 4], violation)
    readout[world, 5] += work
    readout[world, 6] += 1.0


class Audit:
    """The audit's arrays and its launches after the sweep (``solver._audit`` under ``Diagnostics.audit``)."""

    def __init__(self, solver):
        self.solver = solver
        forest = solver.forest
        SolverArrays(self, solver.model.device).allocate(
            AUDIT_FIELDS,
            world=forest.world_count,
            world_dof=max(forest.max_world_dofs, 1),
            row=solver.placement.dense_max_constraints,
            mf_row=solver.placement.mf_max_constraints,
        )
        self.coordinates_hat = wp.zeros_like(self.coordinates)

    def record(self) -> dict:
        """Which rows the audit reads, for ``routes.diagnostics.audit``."""
        reason = "Reads every row of every world."
        if self.solver.placement.keyed_rows is not None:
            reason = "Reads NaN in keyed rows' worlds: no store holds their rows after the sweep."
        return {"enabled": True, "slots": list(AUDIT_SLOTS), "reason": reason}

    def _operands(self) -> AuditRows:
        solver = self.solver
        a = AuditRows()
        a.rows = bind_dense_rows(solver.placement)
        a.impulses = solver.placement.impulses
        a.counts = solver.placement.constraint_count
        a.mf_counts = solver.placement.mf_constraint_count
        a.skip_rows_le = int(solver.placement.ink_skip_rows())
        a.keyed = int(solver.placement.keyed_rows is not None)
        a.weighted = int(solver.laws.set.weighted)
        a.material = int(solver.options.contact_compliance)
        a.capacity = solver.placement.dense_max_constraints
        return a

    def _walk_coordinates(self, velocity: wp.array, coordinates: wp.array) -> None:
        """The walked trees' coordinates of ``velocity`` in the walk's frame, at their world coordinates
        (:func:`.factor.walk.walk_enter_kernel`)."""
        solver = self.solver
        factor, forest = solver.factor, solver.forest
        wp.launch(
            walk_enter_kernel(),
            dim=factor.walk_arts.shape[0],
            inputs=[
                factor.walk_arts,
                factor.walk_view(),
                forest.art_to_world,
                forest.articulation_world_dof_offset,
                velocity,
                coordinates,
            ],
            device=solver.model.device,
        )

    def launch(self) -> None:
        """Audit the step: after the solve, on the published velocity ``v_out``."""
        solver, forest = self.solver, self.solver.forest
        device, worlds = solver.model.device, forest.world_count
        a, mf, v = self._operands(), bind_mf_rows(solver.placement), solver.v_out
        route = solver._plan.solve.route.name
        if route == "sparse_pgs":
            inputs = [
                a,
                solver.placement.sparse_row_support,
                solver.placement.sparse_row_dof,
                solver.placement.sparse_row_factor,
                solver.placement.row_incident,
                solver.placement.sparse_factor_velocity_delta,
            ]
            wp.launch(sparse_rows, dim=worlds, inputs=inputs, outputs=[self.velocity], device=device)
        elif route == "sparse_diagonal":
            size = solver.placement.sparse_diagonal_dense_size
            # The dense arms' rows are the walk's (walk_rows): their coordinates are the walk's too.
            self._walk_coordinates(v, self.coordinates)
            inputs = [
                a,
                forest.world_dof_indices,
                solver.placement.sparse_diagonal_dense_offsets,
                solver.placement.sparse_diagonal_dense_groups,
                size,
                self.coordinates,
                solver.placement.J_by_size[size],
                solver.placement.sparse_diagonal_row_dof,
                solver.placement.sparse_diagonal_row_z,
                solver.factor.diagonal_inverse_mass,
                v,
            ]
            wp.launch(diagonal_rows, dim=worlds, inputs=inputs, outputs=[self.velocity], device=device)
            limits = [
                solver.placement.fused_diagonal_limit_active,
                solver.placement.fused_diagonal_limit_lower_rhs,
                solver.placement.fused_diagonal_limit_upper_rhs,
                solver.placement.fused_diagonal_limit_lower_lambda,
                solver.placement.fused_diagonal_limit_upper_lambda,
                forest.world_dof_indices,
                v,
            ]
            wp.launch(fused_limits, dim=worlds, inputs=limits, outputs=[self.fused], device=device)
        else:
            if solver.placement.walks:
                # The one writer's rows are the walk's (walk_rows): their trees' coordinates are the walk's too.
                self._walk_coordinates(v, self.coordinates)
                self._walk_coordinates(solver.v_hat, self.coordinates_hat)
            inputs = [
                a,
                solver.placement.row_support(),
                solver.placement.packed_Z,
                solver.placement.packed_Z,
                self.coordinates,
                self.coordinates_hat,
                solver.placement.row_incident,
            ]
            wp.launch(packed_rows, dim=worlds, inputs=inputs, outputs=[self.velocity], device=device)
        if forest.has_free_rigid_bodies:
            inputs = [
                a,
                mf,
                solver.factor.walk_view(),
                solver.placement.impulses,
                solver.placement.row_w,
                v,
                forest.articulation_world_dof_offset,
                self.coordinates,
            ]
            wp.launch(free_rows, dim=worlds, inputs=inputs, outputs=[self.mf_velocity], device=device)
        inputs = [a, mf, solver.placement.impulses, self.velocity, self.mf_velocity, self.fused]
        wp.launch(audit_world, dim=worlds, inputs=inputs, outputs=[self.readout], device=device)

    def read(self) -> dict:
        """The readout by slot (:data:`AUDIT_SLOTS`), each a per-world array; this synchronizes with the device."""
        values = self.readout.numpy()
        return {name: values[:, slot].copy() for slot, name in enumerate(AUDIT_SLOTS)}
