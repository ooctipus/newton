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

The rows' velocity terms are one kernel over the one ``Z`` operator's ``dot`` (:func:`rows_kernel`), a thread a world,
each row's entries summed in order (:data:`ONE`); a support class is its data (:class:`RowClass`). Packed rows
(:func:`packed_class`) and the sparse diagonal's (:func:`slot_class`) read ``Z_i^T u+`` with ``u+`` the walk's
coordinates of ``v+`` (:func:`.factor.walk.walk_enter_kernel`), so a defect in leaving factor coordinates shows too, and
corrected rows and targets agree on a closure; native sparse rows (:func:`sparse_class`) hold no dense factor and read
their own ``J_i v_hat + Z_i^T du``; the free bodies' (:func:`free_class`) are the 6 + 6 class over the walk's
coordinates, its twelve entries one chain as the sweep's one thread sums them. One law kernel (:func:`audit_world`) then
reads both stores and adds each contact normal's ``eps_i lambda_i``. The sparse diagonal's fused limits, which no store
holds, read their law in place (:func:`fused_limits`). A world whose rows exist only inside its sweep (keyed rows) has
none to read: it reports NaN and is not summed.
"""

from __future__ import annotations

import dataclasses
import functools

import warp as wp

from .device_preludes import substitute
from .factor.walk import walk_enter_kernel
from .field_group import Field, FieldGroup, SolverArrays, bind, struct, values
from .friction_patches import patch_normal_load
from .kernel_template import KernelTemplate, items
from .packed_rows import CLASS_SIDES, PACKED_SIDES, WIDE
from .response_operator import Lanes, SparseFactor, Velocity
from .row_laws import (
    ROW_CONTACT,
    ROW_FRICTION,
    ROW_LINK,
    ROW_MIMIC,
    ROW_SPEED,
    ROW_TORSION,
)
from .row_records import DenseRows
from .row_records import bind as bind_dense_rows
from .sparse_mass_matrix import _SparseRowSupport
from .sweep_routes import free_body_rows, packed_operator
from .sweep_routes_diag import slot_rows

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


AuditRows = struct(
    "AuditRows",
    (
        Field("rows", DenseRows, owner="step"),
        Field("impulses", float, ("line", "row"), owner="placement"),
        Field("counts", int, ("line",), owner="placement", attr="constraint_count"),
        Field("mf_counts", int, ("line",), owner="placement", attr="mf_constraint_count"),
        Field("speed_first", int, ("line",), owner="placement"),
        Field("skip_rows_le", int, owner="classes", attr="tier_rows"),
        Field("keyed", int, owner="step"),
        Field("weighted", int, owner="step"),
        Field("material", int, owner="options", attr="contact_compliance"),
        Field("capacity", int, owner="placement", attr="dense_max_constraints"),
    ),
    __name__,
)
"""What the law reads: the row store's records and impulses, the dense and free-body counts, whether the tiers form
the rows of their worlds in their kernel (:func:`.packed_rows._skipped`) and keyed rows hold them, and the laws in
play."""


@wp.func
def contact_compliance(weight: float, eps: float, diagonal: float) -> float:
    """A contact normal's ``eps_i``: a material contact's, else the relative regularization's ``g d`` from its weight
    ``1 / (1 + g)``; the sweeps regularize contact normals only (:func:`.sweep.row_weight`)."""
    if eps > 0.0:
        return eps
    return (1.0 / weight - 1.0) * diagonal


@wp.func
def tier_world(a: AuditRows, world: int) -> bool:
    """Whether the parallel tiers form ``world``'s rows in their kernel (:func:`.sweep_levels.list_worlds`)."""
    count = a.counts[world]
    return a.skip_rows_le > 0 and count <= a.skip_rows_le and a.mf_counts[world] == 0 and count <= a.speed_first[world]


# ---------------------------------------------------------------------------------------------------------------------
# The rows' velocity terms: one kernel over the one Z operator's dot, a support class its data
# ---------------------------------------------------------------------------------------------------------------------

OPERANDS = FieldGroup(
    Field("row_sides", wp.vec2i, ("line", "row"), owner="placement"),
    Field("response_dofs", int, ("art",), owner="forest", attr="articulation_response_dof_count"),
    Field("articulation_world_dof_offset", int, ("art",), owner="forest"),
    Field("world_offset", int, ("art",), owner="forest", attr="articulation_world_dof_offset"),
    Field("packed_dense", int, owner="placement"),
    Field("world_dof_indices", int, ("world", "world_dof"), owner="forest"),
    Field("packed_Z", float, ("line", "row", "entry"), owner="placement"),
    Field("coordinates", float, ("world", "world_dof"), owner="audit"),
    Field("hat", float, ("world", "world_dof"), owner="audit", attr="coordinates_hat"),
    Field("incident", float, ("line", "row"), owner="placement", attr="row_incident"),
    Field("row_incident", float, ("line", "row"), owner="placement"),
    Field("mf_counts", int, ("world",), owner="placement", attr="mf_constraint_count"),
    Field("speed_first", int, ("world",), owner="placement"),
    Field("tier_rows", int, owner="classes"),
    Field("row_type", int, ("line", "row"), owner="placement"),
    Field("dof_start", int, ("art",), owner="forest", attr="articulation_dof_start"),
    Field("base", int, owner="placement", attr="body_base"),
    Field("v", float, ("dof",), owner="step"),
    Field("support", _SparseRowSupport, owner="placement", attr="sparse_row_support"),
    Field("row_dof", int, ("line", "row", "support"), owner="placement", attr="sparse_row_dof"),
    Field("row_factor", float, ("line", "row", "support"), owner="placement", attr="sparse_row_factor"),
    Field("delta", float, ("world", "world_dof"), owner="placement", attr="sparse_factor_velocity_delta"),
    Field("dense_offsets", int, ("world",), owner="placement", attr="sparse_diagonal_dense_offsets"),
    Field("dense_groups", int, ("world",), owner="placement", attr="sparse_diagonal_dense_groups"),
    Field("dense_Z", float, ("line", "row", "dof"), owner="step"),
    Field("sparse_row_dof", int, ("line", "row", "entry"), owner="placement", attr="sparse_diagonal_row_dof"),
    Field("sparse_row_z", float, ("line", "row", "entry"), owner="placement", attr="sparse_diagonal_row_z"),
    Field("inverse_mass", float, ("dof",), owner="factor", attr="diagonal_inverse_mass"),
    Field("row_classes", int, ("line", "row"), owner="class_table", attr="row_class"),
    Field("class_dof", int, ("class", "entry"), owner="class_table"),
    Field("class_count", int, ("class",), owner="class_table"),
)
"""Every operand a support class reads beside the kernel's own, declared once with where the audit finds it."""

ONE = Lanes(1, "0")
"""The audit's lanes: one thread a world, each row's entries summed in order (:class:`.response_operator.Lanes`)."""


@dataclasses.dataclass(frozen=True)
class Called(Velocity):
    """``u`` at a coordinate through a native function of it, ``array(index)``."""

    def at(self, index: str) -> str:
        return f"{self.array}({index})"


def _sum(operator: SparseFactor, out: str, start: str) -> str:
    """``float out = start``, then ``out += Z_i . u`` over row ``i``'s entries in order: the operator's one-thread
    ``dot`` (its velocity ``u``)."""
    dot = operator.dot(((out, "row", None),), {"row": "i"}, "            ", reduce=False)
    return f"        float {out} = {start};\n        {{\n{dot}\n        }}"


@dataclasses.dataclass(frozen=True)
class RowClass:
    """A support class whose rows the audit reads: its :data:`OPERANDS` beside the kernel's own, a world's names
    (``prologue``, after ``line`` and ``count``) and row ``i``'s statements that declare its velocity term ``term``."""

    name: str
    operands: tuple
    prologue: str
    row: str
    preludes: tuple = ()


_ROWS = """    const int line = first.data[world];
    const int count = counts.data[world] < capacity ? counts.data[world] : capacity;
$prologue
    for (int i = 0; i < count; ++i) {
$row
        velocity.data[(size_t)world * velocity.shape[1] + i] = term;
    }
"""


@functools.cache
def rows_kernel(cls: RowClass) -> wp.Kernel:
    """The rows' velocity terms of support class ``cls``, a thread a world: each row ``i``'s ``term`` into
    ``velocity[world, i]``, over the world's ``min(counts[world], capacity)`` rows from its line ``first[world]``."""
    common = (
        ("first", wp.array[int]),
        ("counts", wp.array[int]),
        ("capacity", int),
        ("velocity", wp.array2d[float]),
    )
    return KernelTemplate(
        f"audit_rows_{cls.name}",
        (*common, *((field.name, field.annotation) for field in cls.operands)),
        substitute(_ROWS, prologue=cls.prologue, row=cls.row),
        items("world", flat=True),
        preludes=cls.preludes,
        dual=True,
        native_name="audit_rows_native",
    ).kernel()


def packed_class(support: int, wide: bool, classes: bool = False) -> RowClass:
    """The packed rows (:func:`.sweep_routes.packed_operator`): ``Z_i^T u+``, ``u`` the walk's coordinates of ``v+``;
    a world whose rows the tiers form reads ``J_i v+`` (``Z`` there holds ``J``), every other
    ``(J_i v_hat + Z_i^T u+) - Z_i^T u_hat``. A free body's speed row, a row of the line, reads its law's own term,
    as the free bodies' run's (:func:`free_class`); with ``classes``, by their support classes."""
    operator = packed_operator(support, ONE, wide=wide, classes=classes)
    at_u = dataclasses.replace(operator, velocity=Velocity("u"))
    at_hat = dataclasses.replace(operator, velocity=Velocity("u_hat"))
    return RowClass(
        f"packed{support}{'_wide' if wide else ''}{'_classes' if classes else ''}",
        OPERANDS.select(
            *(("row_classes", "class_dof", "class_count") if classes else ()),
            "row_sides",
            "response_dofs",
            "articulation_world_dof_offset",
            "packed_dense",
            "world_dof_indices",
            "packed_Z",
            "coordinates",
            "hat",
            "incident",
            "mf_counts",
            "speed_first",
            "tier_rows",
            "row_type",
            "dof_start",
            "v",
        ),
        """    const int off_dense = line * row_sides.shape[1];
    const int packed_base = line * packed_Z.shape[1] * packed_Z.shape[2];
    const int dof_map_base = world * world_dof_indices.shape[1];
    const float* u = coordinates.data + (size_t)world * coordinates.shape[1];
    const float* u_hat = hat.data + (size_t)world * hat.shape[1];
    const bool tier = tier_rows > 0 && counts.data[world] <= tier_rows && mf_counts.data[world] == 0
        && counts.data[world] <= speed_first.data[world];""",
        f"""        float term = 0.0f;
        if (row_type.data[off_dense + i] == {int(ROW_SPEED)}) {{
            const wp::vec2i sides = row_sides.data[off_dense + i];
            const float axis = packed_Z.data[packed_base + i * packed_Z.shape[2] + PACKED_FIRST(sides, sides[0]) + 3];
            const int start = dof_start.data[sides[0]] + (axis != 0.0f ? 3 : 0);
            term = -wp::length(wp::vec3(v.data[start], v.data[start + 1], v.data[start + 2]));
        }} else {{
"""
        + _sum(at_u, "zu", "0.0f")
        + """
        term = zu;
        if (!tier) {
"""
        + _sum(at_hat, "zhat", "0.0f")
        + """
            term = incident.data[(size_t)line * incident.shape[1] + i] + zu - zhat;
        }
        }""",
        (PACKED_SIDES, *((CLASS_SIDES,) if classes else ())),
    )


def sparse_class(support: int, table: bool) -> RowClass:
    """Native sparse factors' rows (case K1, :func:`.sweep_routes_sparse.factor_operator`): ``J_i v_hat + Z_i^T du``,
    summed from the incident, entries through the support table or explicit coordinates."""

    def index(row, entry):
        if table:
            return f"support.class_dof.data[support.row_class.data[offset + {row}] * {support} + {entry}]"
        return f"row_dof.data[row_base + {row} * {support} + {entry}]"

    def factor(row, entry):
        return f"row_factor.data[row_base + {row} * {support} + {entry}]"

    operator = SparseFactor(support, index, factor, ONE, velocity=Velocity("du"))
    return RowClass(
        f"sparse{support}{'_table' if table else ''}",
        OPERANDS.select("support", "row_dof", "row_factor", "row_incident", "delta"),
        """    const int offset = line * support.row_class.shape[1];
    const int row_base = line * row_factor.shape[1] * row_factor.shape[2];
    const float* du = delta.data + (size_t)world * delta.shape[1];""",
        _sum(operator, "term", "row_incident.data[(size_t)line * row_incident.shape[1] + i]"),
    )


def slot_class(dense: int, rows: int) -> RowClass:
    """The sparse diagonal's rows (case K2, :func:`.sweep_routes_diag.slot_rows`): ``Z_i^T u+``, ``u`` on the dense
    group the walk's coordinates of ``v+`` (the frame its rows were whitened in) and ``u = v / s``, ``s = m^-1/2``, on a
    slot's coordinate (zero where the coordinate has no inverse mass)."""
    operator = dataclasses.replace(slot_rows(dense, rows, ONE), velocity=Called("U"))
    return RowClass(
        f"slots{dense}_{rows}",
        OPERANDS.select(
            "world_dof_indices",
            "dense_offsets",
            "dense_groups",
            "coordinates",
            "dense_Z",
            "sparse_row_dof",
            "sparse_row_z",
            "inverse_mass",
            "v",
        ),
        f"""    const int dense_offset = dense_offsets.data[world], dense_group = dense_groups.data[world];
    const int row_base = line * sparse_row_dof.shape[1];
    const float* u = coordinates.data + (size_t)world * coordinates.shape[1];
    const int* dofs = world_dof_indices.data + (size_t)world * world_dof_indices.shape[1];
    const auto U = [&](const int c) -> float {{
        if (c >= dense_offset && c < dense_offset + {dense}) return u[c];
        const int dof = dofs[c];
        if (dof < 0) return 0.0f;
        const float scale = wp::sqrt(inverse_mass.data[dof]);
        return scale > 0.0f ? v.data[dof] / scale : 0.0f;
    }};""",
        _sum(operator, "term", "0.0f"),
    )


@functools.cache
def free_class() -> RowClass:
    """The free bodies' rows, the 6 + 6 class (:func:`.sweep_routes.free_body_rows`): ``Z_i^T u+`` over the walk's
    coordinates of ``v+`` at the sides' world coordinates. A speed row's term is ``-|v|`` of the coordinates its
    unwhitened axis marks, linear from ``e_0`` or angular from ``e_3``, its bias the limit
    (:func:`.projection_laws.speed_row`): the law's own term, not a ``Z`` row."""
    operator = dataclasses.replace(free_body_rows(ONE, "S"), velocity=Velocity("u"))
    return RowClass(
        "free",
        OPERANDS.select("row_type", "row_sides", "base", "packed_Z", "dof_start", "world_offset", "v", "coordinates"),
        """    const int records = line * row_type.shape[1] + base;
    const int S = packed_Z.shape[2];
    const size_t free_base = ((size_t)line * packed_Z.shape[1] + base) * S;
    const float* u = coordinates.data + (size_t)world * coordinates.shape[1];""",
        f"""        const wp::vec2i sides = row_sides.data[records + i];
        float term = 0.0f;
        if (row_type.data[records + i] == {int(ROW_SPEED)}) {{
            const int start = dof_start.data[sides[0]] + (packed_Z.data[free_base + i * S + 3] != 0.0f ? 3 : 0);
            term = -wp::length(wp::vec3(v.data[start], v.data[start + 1], v.data[start + 2]));
        }} else {{
            const int dof_a = sides[0] >= 0 ? world_offset.data[sides[0]] : -1;
            const int dof_b = sides[1] >= 0 ? world_offset.data[sides[1]] : -1;
"""
        + _sum(operator, "z", "0.0f")
        + """
            term = z;
        }""",
    )


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
def audit_store(a: AuditRows, world: int, line: int, base: int, count: int, velocity: wp.array2d[float]) -> wp.vec3:
    """One row class's ``(G, D, W)``: its ``count`` rows from ``base`` of the world's line, row ``i``'s velocity term
    at ``velocity[world, i]``, a contact normal with a positive diagonal plus its ``eps_i lambda_i``. A contact's
    friction block is its pair after its normal, then its spin row if any: its first tangent reads it whole, and its
    other rows add only their work."""
    rows = a.rows
    gap = float(0.0)
    violation = float(0.0)
    work = float(0.0)
    for i in range(count):
        r = base + i
        kind = rows.row_type[line, r]
        impulse = a.impulses[line, r]
        bias = rows.rhs[line, r]
        term = velocity[world, i]
        diagonal = rows.diag[line, r]
        if kind == ROW_CONTACT and diagonal > 0.0:
            weight = float(1.0)
            if a.weighted != 0:
                weight = rows.row_w[line, r]
            eps = float(0.0)
            if a.material != 0:
                eps = rows.row_eps[line, r]
            term += contact_compliance(weight, eps, diagonal) * impulse
        residual = term + bias
        work -= impulse * bias
        if kind == ROW_FRICTION or kind == ROW_TORSION:
            if kind == ROW_FRICTION and rows.row_parent[line, r] == r - 1:
                second = velocity[world, i + 1] + rows.rhs[line, r + 1]
                spin = float(0.0)
                spin_scale = float(0.0)
                term = impulse * residual + a.impulses[line, r + 1] * second
                if i + 2 < count:
                    if rows.row_type[line, r + 2] == ROW_TORSION:
                        spin = velocity[world, i + 2] + rows.rhs[line, r + 2]
                        spin_scale = rows.row_mu[line, r + 2]
                        term += a.impulses[line, r + 2] * spin
                scaled = wp.vec3(rows.row_mu[line, r] * residual, rows.row_mu[line, r + 1] * second, spin_scale * spin)
                term += patch_normal_load(rows.row_parent, a.impulses, line, r - 1) * wp.length(scaled)
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
    base: int,
    velocity: wp.array2d[float],
    mf_velocity: wp.array2d[float],
    fused: wp.array[wp.vec3],
    readout: wp.array2d[float],
):
    """One thread per world: the row laws of both stores over the rows' velocity terms, into :data:`AUDIT_SLOTS`; the
    free bodies' rows from ``base`` of the world's line."""
    world = wp.tid()
    if tier_world(a, world) and a.keyed != 0:
        readout[world, 0] = wp.nan
        readout[world, 1] = wp.nan
        readout[world, 2] = wp.nan
        return
    line = a.rows.first[world]
    dense = audit_store(a, world, line, 0, wp.min(a.counts[world], a.capacity), velocity)
    free = audit_store(a, world, line, base, a.mf_counts[world], mf_velocity)
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

    def _operands(self):
        solver, placement = self.solver, self.solver.placement
        owners = {"placement": placement, "classes": solver.spec.classes, "options": solver.options}
        keyed, weighted = placement.keyed_rows is not None, solver.laws.set.weighted
        return bind(AuditRows, rows=bind_dense_rows(placement), keyed=keyed, weighted=weighted, **owners)

    def _walk_coordinates(self, velocity: wp.array, coordinates: wp.array) -> None:
        """The walked trees' coordinates of ``velocity`` in the walk's frame, at their world coordinates
        (:func:`.factor.walk.walk_enter_kernel`)."""
        solver = self.solver
        factor, forest = solver.factor, solver.forest
        wp.launch(
            walk_enter_kernel(factor.walk_classes),
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

    @functools.cached_property
    def dense_class(self) -> RowClass:
        """The dense rows' support class, by the solve's route (:attr:`.step_plan.StepPlan.solve`), chosen once."""
        solver = self.solver
        placement = solver.placement
        route = solver._plan.solve.route.name
        if route == "sparse_pgs":
            return sparse_class(int(placement.sparse_row_factor.shape[2]), bool(placement.sparse_row_support.enabled))
        if route == "sparse_diagonal":
            size = solver.spec.classes.dense_size
            return slot_class(int(size), int(placement.J_by_size[size].shape[1]))
        return packed_class(int(placement.packed_support), int(placement.packed_dense) == WIDE, placement.classed)

    def _operands_of(self, cls: RowClass, v: wp.array) -> list:
        solver, placement, classes = self.solver, self.solver.placement, self.solver.spec.classes
        owners = {"placement": placement, "forest": solver.forest, "factor": solver.factor, "classes": classes}
        return values(
            cls.operands,
            **owners,
            audit=self,
            v=v,
            dense_Z=placement.J_by_size.get(classes.dense_size),
            class_table=placement.row_classes,
        )

    def launch(self) -> None:
        """Audit the step: after the solve, on the published velocity ``v_out``: each store's velocity terms by the one
        rows kernel of its support class, the sparse diagonal's fused limits, then the law."""
        solver, forest, placement = self.solver, self.solver.forest, self.solver.placement
        device, worlds, v = solver.model.device, forest.world_count, solver.v_out
        first, capacity = placement.dense_first, placement.dense_max_constraints
        # The walked rows (walk_rows) are in the walk's coordinates: their trees' coordinates are the walk's too.
        route = solver._plan.solve.route.name
        if route != "sparse_pgs" and placement.walks:
            self._walk_coordinates(v, self.coordinates)
            if route != "sparse_diagonal":
                self._walk_coordinates(solver.v_hat, self.coordinates_hat)
        operands = self._operands_of(self.dense_class, v)
        wp.launch(
            rows_kernel(self.dense_class),
            dim=worlds,
            inputs=[first, placement.constraint_count, capacity, self.velocity, *operands],
            device=device,
        )
        if route == "sparse_diagonal":
            limits = [
                placement.fused_diagonal_limit_active,
                placement.fused_diagonal_limit_lower_rhs,
                placement.fused_diagonal_limit_upper_rhs,
                placement.fused_diagonal_limit_lower_lambda,
                placement.fused_diagonal_limit_upper_lambda,
                forest.world_dof_indices,
                v,
            ]
            wp.launch(fused_limits, dim=worlds, inputs=limits, outputs=[self.fused], device=device)
        if forest.has_free_rigid_bodies:
            mf = [first, placement.mf_constraint_count, placement.mf_max_constraints, self.mf_velocity]
            free = self._operands_of(free_class(), v)
            wp.launch(rows_kernel(free_class()), dim=worlds, inputs=[*mf, *free], device=device)
        inputs = [self._operands(), int(placement.body_base), self.velocity, self.mf_velocity, self.fused]
        wp.launch(audit_world, dim=worlds, inputs=inputs, outputs=[self.readout], device=device)

    def read(self) -> dict:
        """The readout by slot (:data:`AUDIT_SLOTS`), each a per-world array; this synchronizes with the device."""
        values = self.readout.numpy()
        return {name: values[:, slot].copy() for slot, name in enumerate(AUDIT_SLOTS)}
