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
coordinates, summed as the sweep's one thread sums them; a tier's walked rows by the tier's text (:func:`seeded_class`).
One law kernel (:func:`audit_world`) then adds each contact normal's ``eps_i lambda_i``; the sparse diagonal's fused
limits, which no store holds, read their law in place (:func:`fused_limits`).
"""

from __future__ import annotations

import dataclasses
import functools

import warp as wp

from .device_preludes import CPU_MATH, substitute
from .factor.walk import WalkTrees, walk_enter_kernel, walk_prelude, walk_row_prelude, walk_tables
from .field_group import Field, FieldGroup, SolverArrays, bind, struct, values
from .friction_patches import patch_normal_load
from .kernel_template import Dispatch, KernelTemplate, items
from .law_source import law_preludes
from .packed_rows import CLASS_SIDES, PACKED_SIDES, WIDE
from .response_operator import Lanes, SparseFactor, Velocity, WalkedRows
from .row_laws import (
    ROW_CONTACT,
    ROW_FRICTION,
    ROW_LAW,
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
from .sweep_routes_tiers import _seeded_row
from .tree_scan import classes_half
from .walk_rows import RowSeeds, emits

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
    Field("rhs", wp.float32, ("world", "row")),
    Field("diag", wp.float32, ("world", "row")),
)
"""The readout and scratch: each world's coordinates as its rows read them, each row's ``J_i v+ + eps_i lambda_i`` (and
a walked row's rhs and diagonal, :func:`seeded_class`), and the sparse diagonal's fused limits' ``(G, D, W)``."""


AuditRows = struct(
    "AuditRows",
    (
        Field("rows", DenseRows, owner="step"),
        Field("impulses", float, ("line", "row"), owner="placement"),
        Field("counts", int, ("line",), owner="placement", attr="constraint_count"),
        Field("mf_counts", int, ("line",), owner="placement", attr="mf_constraint_count"),
        Field("speed_first", int, ("line",), owner="placement"),
        Field("weighted", int, owner="step"),
        Field("material", int, owner="options", attr="contact_compliance"),
        Field("capacity", int, owner="placement", attr="dense_max_constraints"),
    ),
    __name__,
)
"""What the law reads: the row store's records and impulses, the dense and free-body counts, and the laws in play."""


@wp.func
def contact_compliance(weight: float, eps: float, diagonal: float) -> float:
    """A contact normal's ``eps_i``: a material contact's, else ``g d`` from its weight ``1 / (1 + g)``."""
    if eps > 0.0:
        return eps
    return (1.0 / weight - 1.0) * diagonal


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
    Field("row_seeds", RowSeeds, owner="placement"),
    Field("rhs_bias", float, ("line", "row"), owner="placement", attr="rhs"),
    Field("world_row_type", int, ("line", "row"), owner="placement", attr="row_type"),
    Field("world_row_eps", float, ("line", "row"), owner="placement", attr="row_eps"),
    Field("ink_meta", int, ("meta",), owner="placement"),
    Field("walk", WalkTrees, owner="step"),
    Field("v_hat", float, ("dof",), owner="step"),
    Field("cfm", float, owner="options", attr="pgs_cfm"),
    Field("rhs", float, ("world", "row"), owner="audit"),
    Field("diag", float, ("world", "row"), owner="audit"),
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
    """``float out = start``, then ``out += Z_i . u`` over row ``i``'s entries in order (the one-thread dot)."""
    dot = operator.dot(((out, "row", None),), {"row": "i"}, "            ", reduce=False)
    return f"        float {out} = {start};\n        {{\n{dot}\n        }}"


WORLDS = items("world", flat=True)


@dataclasses.dataclass(frozen=True)
class RowClass:
    """A support class whose rows the audit reads: its :data:`OPERANDS` beside the kernel's own, a world's names
    (``prologue``, after ``line`` and ``count``) and row ``i``'s statements that declare its velocity term ``term``."""

    name: str
    operands: tuple
    prologue: str
    row: str
    preludes: tuple = ()
    dispatch: Dispatch = WORLDS


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
    """Support class ``cls``'s velocity terms, a world a thread: row ``i``'s ``term`` into ``velocity[world, i]``."""
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
        cls.dispatch,
        preludes=cls.preludes,
        dual=True,
        native_name="audit_rows_native",
    ).kernel()


def packed_class(support: int, wide: bool, classes: bool = False) -> RowClass:
    """The packed rows (:func:`.sweep_routes.packed_operator`): ``Z_i^T u+``, ``u`` the walk's coordinates of ``v+``: a
    tier's world reads ``J_i v+`` (``Z`` holds ``J``), every other ``(J_i v_hat + Z_i^T u+) - Z_i^T u_hat``; a free
    body's speed row its law's own term (:func:`free_class`); with ``classes``, by their support classes."""
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
    coordinates of ``v+``. A speed row's term is ``-|v|`` of the coordinates its unwhitened axis marks (linear from
    ``e_0``, angular from ``e_3``), its bias the limit: the law's own term (:func:`.projection_laws.speed_row`)."""
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


EMITTED = Dispatch(
    "world = wp.tid()\nif emits(row_seeds, world, counts[world]):\n    $CALL",
    ("world",),
    (("world", int),),
    constants=(("emits", emits),),
)
"""A thread a world whose tier walks its rows (:func:`.walk_rows.emits`)."""


_SEEDED = "row_seeds rhs_bias world_row_type ink_meta walk world_dof_indices v_hat coordinates cfm rhs diag"


@functools.cache
def seeded_class(dofs: int, response: tuple, frame: tuple, classes: tuple, material: bool) -> RowClass:
    """The rows a Jacobi tier walks from their seeds (:class:`.walk_rows.RowSeeds`), by the tier's own text
    (:func:`.sweep_routes_tiers._seeded_row`, :meth:`.WalkedRows.seed_walk`): ``Z_i^T u+`` its term; its rhs (a bounce
    from ``J_i v_hat``) and diagonal ``|Z_i|^2 + cfm`` as the tier forms them, into ``rhs`` and ``diag``."""
    NA, NB, OA, OB = response
    read = _seeded_row(False, material)
    walk, release = WalkedRows(NA, NB, OA, OB, dofs, 1, ONE, read, True, frame, len(classes)).seed_walk("    ")
    tables = walk_tables(classes)
    return RowClass(
        f"seeded{dofs}_{NA}_{NB}_{OA}_{OB}_f{frame[0]}_{frame[1]}_{abs(hash(tables)) % (1 << 32):08x}"
        + ("_material" if material else ""),
        OPERANDS.select(*_SEEDED.split(), *(("world_row_eps",) if material else ())),
        f"""    const int off_dense = line * rhs_bias.shape[1];
    const int art_a = ink_meta.data[world * 4], art_b = ink_meta.data[world * 4 + 2];
    const int* dof_map = world_dof_indices.data + (size_t)world * world_dof_indices.shape[1];
    const float* u = coordinates.data + (size_t)world * coordinates.shape[1];
    float s_v[{dofs}], Zi[{dofs}], incident = 0.0f;
    for (int d = 0; d < {dofs}; ++d) s_v[d] = dof_map[d] >= 0 ? v_hat.data[dof_map[d]] : 0.0f;
    float* const s_rhs = rhs.data + (size_t)world * rhs.shape[1];
    constexpr int ZS = 1;
{walk}""",
        f"""        for (int d = 0; d < {dofs}; ++d) Zi[d] = 0.0f;
        incident = 0.0f;
        s_rhs[i] = rhs_bias.data[off_dense + i];
        {{
{read}
        }}
{release}
        float term = 0.0f, norm = 0.0f;
        for (int d = 0; d < {dofs}; ++d) {{
            term += Zi[d] * u[d];
            norm += Zi[d] * Zi[d];
        }}
        diag.data[(size_t)world * diag.shape[1] + i] = norm + cfm;""",
        (CPU_MATH, *law_preludes(ROW_LAW), walk_prelude(tables, 1, classes_half(classes)), walk_row_prelude(1)),
        EMITTED,
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
def row_rhs(a: AuditRows, rhs: wp.array2d[float], walked: bool, world: int, line: int, r: int, i: int) -> float:
    """Row ``r``'s rhs as its sweep read it: the store's, or the audit's where its tier walked it (``walked``)."""
    if walked:
        return rhs[world, i]
    return a.rows.rhs[line, r]


@wp.func
def audit_store(
    a: AuditRows, world: int, line: int, base: int, count: int, velocity: wp.array2d[float], walked: bool,
    rhs: wp.array2d[float], diag: wp.array2d[float],
) -> wp.vec3:  # fmt: skip
    """One row class's ``(G, D, W)``: its ``count`` rows from ``base`` of the world's line (their rhs and diagonals in
    ``rhs`` and ``diag`` where ``walked``), row ``i``'s velocity term at ``velocity[world, i]``, a contact normal with a
    positive diagonal plus its ``eps_i lambda_i``. A contact's friction block is its pair after its normal, then its
    spin row if any: its first tangent reads it whole, and its other rows add only their work."""
    rows = a.rows
    gap = float(0.0)
    violation = float(0.0)
    work = float(0.0)
    for i in range(count):
        r = base + i
        kind = rows.row_type[line, r]
        impulse = a.impulses[line, r]
        bias = row_rhs(a, rhs, walked, world, line, r, i)
        term = velocity[world, i]
        diagonal = rows.diag[line, r]
        if walked:
            diagonal = diag[world, i]
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
                second = velocity[world, i + 1] + row_rhs(a, rhs, walked, world, line, r + 1, i + 1)
                spin = float(0.0)
                spin_scale = float(0.0)
                term = impulse * residual + a.impulses[line, r + 1] * second
                if i + 2 < count:
                    if rows.row_type[line, r + 2] == ROW_TORSION:
                        spin = velocity[world, i + 2] + row_rhs(a, rhs, walked, world, line, r + 2, i + 2)
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
    seeds: RowSeeds,
    base: int,
    velocity: wp.array2d[float],
    mf_velocity: wp.array2d[float],
    fused: wp.array[wp.vec3],
    rhs: wp.array2d[float],
    diag: wp.array2d[float],
    readout: wp.array2d[float],
):
    """One thread per world: the row laws of both stores over the rows' velocity terms, into :data:`AUDIT_SLOTS`; the
    free bodies' rows from ``base`` of the world's line, a tier's walked rows by ``rhs`` and ``diag``."""
    world = wp.tid()
    line = a.rows.first[world]
    walked = emits(seeds, world, a.counts[world])
    dense = audit_store(a, world, line, 0, wp.min(a.counts[world], a.capacity), velocity, walked, rhs, diag)
    free = audit_store(a, world, line, base, a.mf_counts[world], mf_velocity, False, rhs, diag)
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
        return {"enabled": True, "slots": list(AUDIT_SLOTS), "reason": "Reads every row of every world."}

    def _operands(self):
        solver, placement = self.solver, self.solver.placement
        owners = {"placement": placement, "classes": solver.spec.classes, "options": solver.options}
        return bind(AuditRows, rows=bind_dense_rows(placement), weighted=solver.laws.set.weighted, **owners)

    def _walk_coordinates(self, velocity: wp.array, coordinates: wp.array) -> None:
        """The walked trees' walk coordinates of ``velocity`` (:func:`.factor.walk.walk_enter_kernel`)."""
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
        route = solver._plan.solve.name
        if route == "sparse_pgs":
            return sparse_class(int(placement.sparse_row_factor.shape[2]), bool(placement.sparse_row_support.enabled))
        if route == "sparse_diagonal":
            size = solver.spec.classes.dense_size
            return slot_class(int(size), int(placement.J_by_size[size].shape[1]))
        return packed_class(int(placement.packed_support), int(placement.packed_dense) == WIDE, placement.classed)

    @functools.cached_property
    def walked_class(self) -> RowClass | None:
        """The rows the tiers walk from their seeds (:func:`seeded_class`, of the class's tiers), or ``None``."""
        emitted = [launch.config for launch in self.solver.sweep.tiers if launch.config.emit]
        if not emitted:
            return None
        tier = emitted[0]
        return seeded_class(tier.max_world_dofs, tier.inkernel_response, tier.frame, tier.classes, tier.material)

    def _operands_of(self, cls: RowClass, v: wp.array) -> list:
        solver, placement, classes = self.solver, self.solver.placement, self.solver.spec.classes
        owners = {"placement": placement, "forest": solver.forest, "factor": solver.factor, "classes": classes}
        return values(
            cls.operands,
            **owners,
            audit=self,
            options=solver.options,
            v=v,
            v_hat=solver.v_hat,
            walk=solver.factor.walk_view() if solver.factor.walk_arts is not None else None,
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
        route = solver._plan.solve.name
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
        if self.walked_class is not None:
            walked = self._operands_of(self.walked_class, v)
            inputs = [first, placement.constraint_count, capacity, self.velocity, *walked]
            wp.launch(rows_kernel(self.walked_class), dim=worlds, inputs=inputs, device=device)
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
        inputs = [self._operands(), placement.row_seeds, int(placement.body_base)]
        inputs += [self.velocity, self.mf_velocity, self.fused, self.rhs, self.diag]
        wp.launch(audit_world, dim=worlds, inputs=inputs, outputs=[self.readout], device=device)

    def read(self) -> dict:
        """The readout by slot (:data:`AUDIT_SLOTS`), each a per-world array; this synchronizes with the device."""
        values = self.readout.numpy()
        return {name: values[:, slot].copy() for slot, name in enumerate(AUDIT_SLOTS)}
