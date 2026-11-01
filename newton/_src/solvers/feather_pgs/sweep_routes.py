# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""FeatherPGS's sweep routes as configurations of the sweep emitter (:mod:`.sweep`).

Each function here builds one route's kernel from the emitter's parts; every kernel's working set fits the static
budget of the device it is built for (:func:`.staging.arch_budget`).
"""

from __future__ import annotations

import dataclasses
from functools import cache

import warp as wp

from . import kernel_template, sweep_units
from .device_preludes import CPU_MATH, RING, SHARED_STORAGE, cuda_native, dual_native, lane_group
from .kernel_template import KernelTemplate
from .kernels import PGS_LOCAL_SOLVE_OWNER_GENERAL, BodyFactors
from .launch import device_lanes
from .law_source import law_preludes
from .packed_rows import PACKED_SIDES
from .projection_laws import CONE_LAW, PROJECTION_LAW
from .response_operator import CholeskySolve, FreeBodyZ, Lanes, SparseFactor, Velocity
from .row_metadata import _validate_dense_metadata_encoding
from .solve_policy import DEFAULT_POLICY, QUIET_LITERAL, SolvePolicy, block_argmax, sweep_exit
from .staging import Buffer, Staging, arch_budget
from .sweep import (
    CONTACT,
    FRICTION,
    VELOCITY_LIMIT,
    CachedPairMetric,
    FreeBodyRows,
    ReducedPairCoupling,
    RowContext,
    RowLaws,
    UnitOperator,
    WorldRows,
    cached_pair_law,
    contact_unit,
    dense_rows_block,
    end_sweep,
    free_body_rows_block,
    impulse_seed,
    indent,
    material_arguments,
    speed_rows_pass,
    velocity_limit_pass,
)

# The row loops leave the velocity-limit rows to the passes at the end of every sweep (the last word).
_DENSE_ADMISSION = f"if (row_type == {VELOCITY_LIMIT}) continue;"
_MF_ADMISSION = f"if (mf_rt == {VELOCITY_LIMIT}) continue;"

MF_GS_ABI = kernel_template.params(
    ("world", int),
    ("world_constraint_count", wp.array[int]),
    ("dense_first", wp.array[int]),
    ("dense_phase_bounds", wp.array2d[int]),
    ("local_solve_owner", wp.array[int]),
    ("world_dof_indices", wp.array2d[int]),
    ("rhs_bias", wp.array2d[float]),
    ("row_incident", wp.array2d[float]),
    ("world_diag", wp.array2d[float]),
    ("world_row_w", wp.array2d[float]),
    ("world_impulses", wp.array2d[float]),
    ("response_dofs", wp.array[int]),
    ("articulation_world_dof_offset", wp.array[int]),
    ("packed_dense", int),
    ("packed_Z", wp.array3d[float]),
    ("row_sides", wp.array2d[wp.vec2i]),
    ("world_row_type", wp.array2d[int]),
    ("world_row_parent", wp.array2d[int]),
    ("world_row_mu", wp.array2d[float]),
    ("mf_constraint_count", wp.array[int]),
    ("mf_first", wp.array[int]),
    ("mf_contact_rows_end", wp.array[int]),
    ("mf_meta", wp.array2d[int]),
    ("mf_impulses", wp.array2d[float]),
    ("mf_Z_a", wp.array3d[float]),
    ("mf_Z_b", wp.array3d[float]),
    ("factors", BodyFactors),
    ("mf_row_mu", wp.array2d[float]),
    ("mf_row_w", wp.array2d[float]),
    ("iterations", int),
    ("omega", float),
    ("regularize", int),
    ("v_out", wp.array[float]),
)
"""The matrix-free sweep's native ABI: one world per call (the world queue's dispatch)."""


class FactorCoordinates:
    """The articulations a matrix-free sweep's velocity change touches, and their way back (native).

    A sweep runs on the velocity change from the predictor in factor coordinates, ``du`` (``s_v``, from zero): a row's
    residual is its right-hand side, which holds its incident velocity ``J v_hat``, plus ``Z^T du``, and its update
    ``du += Z delta``. The articulations the launch's rows touch are marked at their first world coordinate in a bitmap
    (``s_factor_marks``): each free body of its free-body rows and each side of its dense rows
    (:func:`.dense_rows.row_sides`). Each leaves factor coordinates once, ``dv = P^T L^-T du``
    (:class:`.CholeskySolve`'s ``leave``), and adds to the predictor. An articulation's factor is
    found from its first global coordinate (:class:`.kernels.BodyFactors`' ``first_art``, ``size`` and ``address``); a
    free body's six coordinates take the unrolled solves.
    """

    op = FreeBodyZ()

    @staticmethod
    def words(dofs: int) -> int:
        return (int(dofs) + 31) // 32

    @classmethod
    def buffers(cls, dofs: int) -> tuple[Buffer, ...]:
        return (Buffer("s_factor_marks", "unsigned", cls.words(dofs)),)

    def factor(self, first: str) -> str:
        """The native index in ``factors.L`` of the free body whose first world coordinate is ``first``."""
        return f"{self.op.factors}.dof_factor.data[world_dof_indices.data[dof_map_base + {first}]]"

    @classmethod
    def clear(cls, dofs: int, lane: str) -> str:
        """Clear the bitmap (the first warp, before the loads' barrier)."""
        return f"for (int w = {lane}; w < {cls.words(dofs)}; w += STRIDE) s_factor_marks[w] = 0u;"

    @staticmethod
    def mark(first_row: str, rows: str, thread: str, threads: str) -> str:
        """Mark the bodies of the free-body rows ``[first_row, rows)``."""
        return f"""for (int i = {first_row} + {thread}; i < {rows}; i += {threads}) {{
    const int dofs = mf_meta.data[off_meta + i * 4];
    const int da = dofs >> 16, db = (dofs << 16) >> 16;
    if (da >= 0) atomicOr(&s_factor_marks[da >> 5], 1u << (da & 31));
    if (db >= 0) atomicOr(&s_factor_marks[db >> 5], 1u << (db & 31));
}}"""

    def mark_sides(self, rows: str, thread: str, threads: str) -> str:
        """Mark the sides of the dense rows ``[0, rows)`` at their first world coordinates."""
        return f"""for (int i = {thread}; i < {rows}; i += {threads}) {{
    const wp::vec2i sides = row_sides.data[off_dense + i];
    for (int side = 0; side < 2; ++side) {{
        if (sides[side] < 0) continue;
        const int d = {self.op.factors}.world_offset.data[sides[side]];
        atomicOr(&s_factor_marks[d >> 5], 1u << (d & 31));
    }}
}}"""

    def leave_all(self, dofs: int, thread: str, threads: str, sync: str) -> str:
        """Every marked articulation leaves factor coordinates in place, a thread per bitmap word."""
        factors = self.op.factors
        coordinates = lambda k: f"s_v[d + {k}]"  # noqa: E731
        return f"""for (int w = {thread}; w < {self.words(dofs)}; w += {threads}) {{
    unsigned bits = s_factor_marks[w];
    while (bits != 0u) {{
        const int d = (w << 5) + __ffs(bits) - 1;
        bits &= bits - 1u;
        const int art = {factors}.first_art.data[world_dof_indices.data[dof_map_base + d]];
        const int size = {factors}.size.data[art];
        const float* factor = reinterpret_cast<const float*>({factors}.address.data[art]);
        if (size == 6) {{
{indent(CholeskySolve(6, "factor", unroll=True).leave(coordinates, coordinates), "            ")}
        }} else {{
{indent(CholeskySolve("size", "factor").leave(coordinates, coordinates), "            ")}
        }}
    }}
}}
{sync}"""


FACTOR_COORDINATES = FactorCoordinates()
"""The marks and the way back of a matrix-free sweep's velocity change (:class:`FactorCoordinates`)."""


def packed_operator(support: int, lanes: Lanes, *, wide: bool = False) -> SparseFactor:
    """The dense rows of a matrix-free sweep: packed whitened rows of ``support`` entries (:mod:`.packed_rows`), their
    world coordinates from their sides (``PACKED_COORDINATE``), or with ``wide`` rows every coordinate of the world, and
    values from the world's first row (``packed_base``), over the factor velocity change the sweep stages (``s_v``)."""

    def coordinate(row, entry):
        if wide:
            return f"(world_dof_indices.data[dof_map_base + ({entry})] >= 0 ? ({entry}) : -1)"
        return f"PACKED_COORDINATE(row_sides.data[off_dense + ({row})], {entry})"

    def value(row, entry):
        return f"packed_Z.data[packed_base + ({row}) * {int(support)} + ({entry})]"

    return SparseFactor(int(support), coordinate, value, lanes, velocity=Velocity("s_v"))


MAPPINGS = ("warp", "colored")
"""The matrix-free sweep's thread mappings of a world's free-body rows: one warp visits them in row order, or a block
visits their colors in order and each color's units at once (:mod:`.sweep_units`)."""

COLORED_THREADS = 256
"""Threads of a colored world's block: a color's units beyond them run in turns. The largest color measured holds
237 units (coins) and 139 (hardware), and a unit's staged registers (up to 193) leave room for 256 threads."""

COLOR_CHUNK = 256
"""Color offsets a colored block stages at a time: a world's colors are bounded by its units, not by the block."""

COLORED_WORLDS_ABI = kernel_template.params(*MF_GS_ABI[:-1], ("world_colored", wp.array[int]), MF_GS_ABI[-1])
"""The one-warp mapping's ABI beside the colored one: the matrix-free sweep's, then each world's mapping
(:data:`.sweep_units.UNIT_FIELDS`' ``world_colored``) before ``v_out``."""

COLORED_ABI = kernel_template.params(
    *MF_GS_ABI[:-1],
    ("world_colored", wp.array[int]),
    ("world_slot", wp.array[int]),
    ("color_count", wp.array[int]),
    ("color_offset", wp.array2d[int]),
    ("color_slot", wp.array3d[int]),
    ("slot_data", wp.array3d[float]),
    ("slot_words", wp.array3d[int]),
    ("slot_metric", wp.array3d[float]),
    ("v_out", wp.array[float]),
)
"""The colored mapping's native ABI: the matrix-free sweep's, then each world's mapping and storage slot, the worlds'
color counts, their slots' colors and staged units (:data:`.sweep_units.WORLD_FIELDS`, :data:`.sweep_units.UNIT_FIELDS`,
:data:`.sweep_units.SLOT_FIELDS`) before ``v_out``."""


@dataclasses.dataclass(frozen=True)
class _GramUnitBodyZ(UnitOperator, CachedPairMetric, FreeBodyZ):
    """The colored unit's rows (:class:`.sweep.UnitOperator`): one thread per unit, every row's velocity term reduced
    from the unit's velocity ``u`` before the unit's first row (:func:`_unit_terms`), the Gram couplings and the
    pair's metric from the unit's cache formed once per launch (:func:`_colored_metrics`), one pass over the unit's
    coordinates after its rows (:func:`_unit_apply`)."""


@dataclasses.dataclass(frozen=True)
class _GramSpinUnitBodyZ(UnitOperator, ReducedPairCoupling, FreeBodyZ):
    """The colored unit's rows with a spin row (contact torsion): as :class:`_GramUnitBodyZ`, its rows' Gram couplings
    from the unit's cache (the pair's ``unit_cross`` among them); the friction triple's law forms its metric."""


_UNIT_ROWS = (("n", "unit_jv0", "unit_dn"), ("t1", "unit_jv1", "unit_dt1"), ("t2", "unit_jv2", "unit_dt2"))
"""A colored unit's rows: each one's partial sum, velocity term and deferred change, at ``E``'s ``12 r``."""
_SPIN_ROW = ("spin", "unit_jv3", "unit_ds")
"""Under contact torsion the unit's fourth row, its spin row."""


def _unit_rows(torsion: bool) -> tuple:
    return (*_UNIT_ROWS, _SPIN_ROW) if torsion else _UNIT_ROWS


_PAIR_METRIC = ("metric.v0", "metric.v1", "metric.v2", "metric.v3[0]", "metric.v3[1]")
"""A colored friction unit's cached pair metric (``ProjectionLaw::friction_pair_metric``): scale, the two eigenvalues,
the larger's axis."""
_PAIR_COUPLINGS = (("gram_n1", 0, 1), ("gram_n2", 0, 2))
"""The Gram couplings a colored friction unit caches after its pair's metric: each one's name and rows (``E``'s
``12 r``), the pair's tangents with the normal."""
_SPIN_COUPLINGS = (("gram_n3", 0, 3), ("unit_cross", 1, 2), ("unit_cross13", 1, 3), ("unit_cross23", 2, 3))
"""Under contact torsion the rest of the unit's couplings (the friction triple's law forms its own metric)."""


def _unit_couplings(torsion: bool) -> tuple[int, tuple]:
    """A colored unit's cached Gram couplings and their start in its metric: after the friction pair's metric
    (:data:`_PAIR_METRIC`), or under contact ``torsion`` from the start, with the spin row's."""
    return (0, (*_PAIR_COUPLINGS, *_SPIN_COUPLINGS)) if torsion else (len(_PAIR_METRIC), _PAIR_COUPLINGS)


def _unit_terms(pad: str, *, torsion: bool = False) -> str:
    """Native statements, after a unit's ``E``, ``metric``, ``rows`` and ``dof_a``, ``dof_b`` are declared: the unit's
    velocity on its bodies (``unit_u``), its rows' velocity terms from it (each side's six products, then the sides
    added) and the cached Gram couplings of the pair with the normal; under ``torsion`` the spin row's term too and
    every coupling of the unit's rows (:attr:`.sweep_units.SlotLayout.metric`)."""
    rows = _unit_rows(torsion)
    start, couplings = _unit_couplings(torsion)
    lines = [
        "float unit_u[12];",
        "#pragma unroll",
        "for (int k = 0; k < 6; ++k) {",
        "    unit_u[k] = dof_a >= 0 ? s_v[dof_a + k] : 0.0f;",
        "    unit_u[6 + k] = dof_b >= 0 ? s_v[dof_b + k] : 0.0f;",
        "}",
        f"float {', '.join(f'{term} = 0.0f' for _, term, _ in rows)};",
        "#pragma unroll",
        "for (int side = 0; side < 2; ++side) {",
        "    if ((side == 0 ? dof_a : dof_b) < 0) continue;",
        f"    float {', '.join(f'{part} = 0.0f' for part, _, _ in rows)};",
        "#pragma unroll",
        "    for (int k = 0; k < 6; ++k) {",
        "        const float u = unit_u[side * 6 + k];",
        *(
            f"        {part} += E[{f'{12 * r} + ' if r else ''}side * 6 + k] * u;"
            for r, (part, _, _) in enumerate(rows)
        ),
        "    }",
        *(f"    {term} += {part};" for part, term, _ in rows),
        "}",
        f"const float {', '.join(f'{name} = metric[{start + f}]' for f, (name, _, _) in enumerate(couplings))};",
    ]
    return indent("\n".join(lines), pad)


def _unit_apply(pad: str, *, torsion: bool = False) -> str:
    """The unit's deferred changes in one pass: ``u += Z_n dn + Z_t1 dt1 + Z_t2 dt2`` on each of its bodies (and
    ``Z_s ds`` of its spin row under ``torsion``)."""
    rows = _unit_rows(torsion)
    changes = [change for _, _, change in rows]

    def sum_(side: int) -> str:
        terms = [
            f"E[{f'{12 * r + 6 * side} + ' if 12 * r + 6 * side else ''}k] * {change}"
            for r, change in enumerate(changes)
        ]
        total = f"{terms[0]} + {terms[1]}"
        for term in terms[2:]:
            total = f"({total}) + {term}"
        return total

    lines = [
        f"if ({' || '.join(f'{change} != 0.0f' for change in changes)}) {{",
        "#pragma unroll",
        "    for (int k = 0; k < 6; ++k) {",
        "        if (dof_a >= 0)",
        f"            s_v[dof_a + k] = unit_u[k] + ({sum_(0)});",
        "        if (dof_b >= 0)",
        f"            s_v[dof_b + k] = unit_u[6 + k] + ({sum_(1)});",
        "    }",
        "}",
    ]
    return indent("\n".join(lines), pad)


def _colored_pair_metric(op: FreeBodyZ, pad: str) -> str:
    """Native statements, after a unit's ``E`` and ``W`` and ``dof_a``, ``dof_b`` are declared: its friction pair's
    metric (``ProjectionLaw::friction_pair_metric`` of the first tangent's diagonal, the pair's coupling and the
    second tangent's diagonal), formed exactly as the visit's pair law forms it."""
    lines = [
        "const int r = 1;",
        "const float mf_diag = __int_as_float(W[3]);",
        op.coupling("cross", "i", "sib", ""),
        f"const auto metric = ProjectionLaw::friction_pair_metric({op.value('i')}, cross, {op.value('sib')});",
    ]
    return indent("\n".join(lines), pad)


def _colored_units(free: FreeBodyRows, laws: RowLaws, M_MF: int, units: int) -> str:
    """A color's units, one thread each, at block stride: each loads its slot's fields at once
    (:data:`.sweep_units.SLOT_FIELDS`, field by field across the color's ``units`` slots) and runs the free-body row
    law, unrolled over the unit's rows, on the world's staged velocity. No other unit of the color shares a body."""
    unit_rows = FreeBodyRows(M_MF, stream=free.stream, lanes=Lanes(1))
    arrays = {"mf_Z_a": 0, "mf_Z_b": 1}
    torsion = laws.torsion
    layout = sweep_units.slot_layout(torsion)

    def unit_offset(row: str) -> str:
        offsets = {"i": "r", "sib": "2", **({"spin": "3"} if torsion else {})}
        if row not in offsets:
            raise ValueError(f"the colored unit reads rows {sorted(offsets)}, not {row!r}")
        return offsets[row]

    unit_op = (_GramSpinUnitBodyZ if torsion else _GramUnitBodyZ)(
        Lanes(1),
        velocity=Velocity("s_v"),
        entry=lambda array, row, k: f"E[{unit_offset(row)} * 12 + {arrays[array] * 6} + {k}]",
        inverse=lambda row: "mf_diag" if row == "i" else f"__int_as_float(W[{unit_offset(row)} * 3])",
    )
    ring, members = layout.ring, sweep_units.RING_MEMBERS

    def ring_load(parent: str) -> str:
        # The staged ring's members in ring order (the walk's sum, without its chain of global loads), or the walk.
        words = ", ".join(f"W[{ring + k}]" for k in range(members // 2))
        count = f"W[{ring + members // 2}]"
        return f"({count} < 0 ? {unit_rows.ring_load(parent)} : RING_SUM({unit_rows.load(parent)}, {words}, {count}))"

    unit_ctx = RowContext(
        row="i",
        kind="mf_rt",
        operator=unit_op,
        load=unit_rows.load,
        store=unit_rows.store,
        rhs=lambda row: "__int_as_float(meta.z)" if row == "i" else f"__int_as_float(W[{unit_offset(row)} * 3 + 1])",
        mu=lambda row: f"E[{layout.spin_mu if row == 'spin' else layout.mu}]",
        weight=lambda row: f"E[{layout.weight}]",
        eps=unit_rows.eps,
        parent="parent",
        ring_load=ring_load,
        lanes=Lanes(1),
        contact_kinds=(CONTACT,),
        pair_law="FRICTION_PAIR_CACHED(p, ",
    )
    lam = unit_rows.load
    ring_sum = "\n".join(
        [
            "            // A staged patch ring's pooled load: its members' impulses added in ring order.",
            "            const auto RING_SUM = [&](float load, int w0, int w1, int w2, int w3, int members) -> float {",
            "                // Every member's impulse at once (a member past the count reads nothing), then the sum in order.",
            f"                float member[{members}];",
            "#pragma unroll",
            f"                for (int k = 0; k < {members}; ++k) {{",
            "                    const int word = k < 2 ? w0 : k < 4 ? w1 : k < 6 ? w2 : w3;",
            "                    const int r = (int)(((unsigned)word >> ((k & 1) * 16)) & 0xFFFFu);",
            f"                    member[k] = k < members ? {lam('r')} : 0.0f;",
            "                }",
            "#pragma unroll",
            f"                for (int k = 0; k < {members}; ++k)",
            "                    if (k < members) load += member[k];",
            "                return load;",
            "            };",
            *(
                (
                    "            // The unit's Gram couplings, cached once per launch before the sweeps, which it loads with its",
                    "            // fields; the friction triple's law forms its metric.",
                    f"            float metric[{layout.metric}];",
                )
                if torsion
                else (
                    "            // A friction pair's law from its unit's cached metric (formed once per launch, before the sweeps), which",
                    "            // the unit loads with its fields.",
                    f"            float metric[{layout.metric}];",
                    cached_pair_law(
                        "FRICTION_PAIR_CACHED",
                        "metric[0]",
                        "metric[1]",
                        "metric[2]",
                        ("metric[3]", "metric[4]"),
                        " " * 12,
                    ),
                )
            ),
        ]
    )
    fields, words = layout.fields, layout.words
    declare = f"""const int i = unit_row + r;
int4 meta;
meta.x = 0;
meta.y = W[r * 3];
meta.z = W[r * 3 + 1];
meta.w = W[r * 3 + 2];
const int packed_tp = meta.w;
// The unit builder's rows: a contact, then its friction pair (each copy knows its kind).
const int mf_rt = r == 0 ? {CONTACT} : {FRICTION};
const float mf_diag = __int_as_float(meta.y);"""
    unit = contact_unit(
        unit_ctx,
        laws,
        rows="rows",
        declare=declare,
        parent=unit_rows.parent_of("packed_tp"),
        check="mf_diag",
        pad=" " * 20,
    )
    return f"""{ring_sum}
            const int colors = color_count.data[world];
            for (int color = 0; color < colors; ++color) {{
                // The offsets of the next chunk of colors (the barrier after the last color freed the previous one).
                const int chunk = color % {COLOR_CHUNK};
                if (chunk == 0) {{
                    for (int c = tid; c <= {COLOR_CHUNK} && color + c <= colors; c += blockDim.x)
                        s_color_offset[c] = color_offset.data[off_color + color + c];
                    __syncthreads();
                }}
                const int end_unit = s_color_offset[chunk + 1];
                for (int p = s_color_offset[chunk] + tid; p < end_unit; p += blockDim.x) {{
                    const int4 slot = __ldg(reinterpret_cast<const int4*>(&color_slot.data[(off_unit + p) * 4]));
                    const int unit_row = slot.x, rows = slot.y, dof_a = slot.z, dof_b = slot.w;
                    const int slot_base = unit_slot * {fields * units} + p;
                    const int word_base = unit_slot * {words * units} + p;
                    // Every field of the slot and its metric at once, beside the slot's own load: one round of
                    // memory latency per unit. A unit without a friction pair never reads its pair's fields.
                    float E[{fields}];
                    int W[{words}];
#pragma unroll
                    for (int f = 0; f < {fields}; ++f) E[f] = slot_data.data[slot_base + f * {units}];
#pragma unroll
                    for (int f = 0; f < {words}; ++f) W[f] = slot_words.data[word_base + f * {units}];
#pragma unroll
                    for (int f = 0; f < {layout.metric}; ++f)
                        metric[f] = slot_metric.data[unit_slot * {layout.metric * units} + p + f * {units}];
{_unit_terms(" " * 20, torsion=torsion)}
{unit}
{_unit_apply(" " * 20, torsion=torsion)}
                }}
                __syncthreads();
            }}"""


def _colored_metrics(units: int, *, torsion: bool = False) -> str:
    """Before the sweeps, every thread forms the cached metric of the world's staged friction units at block stride
    and stores it in its slot's metric cache (``slot_metric``, field by field across the units), which the unit's law
    and terms read at every visit: a friction pair's metric (scale, the two eigenvalues and the larger's axis), then
    its Gram couplings (:func:`_unit_couplings`)."""
    layout = sweep_units.slot_layout(torsion)
    fields, words = layout.fields, layout.words
    couplings = _unit_couplings(torsion)[1]
    names = [name for name, _, _ in couplings]
    head = [
        "    {",
        "        const int staged = color_offset.data[off_color + color_count.data[world]];",
        "        for (int p = tid; p < staged; p += blockDim.x) {",
        "            const int4 slot = __ldg(reinterpret_cast<const int4*>(&color_slot.data[(off_unit + p) * 4]));",
        f"            if ((slot.y & {7 if torsion else 3}) != {layout.rows}) continue;",
        "            const int dof_a = slot.z, dof_b = slot.w;",
        f"            const int slot_base = unit_slot * {fields * units} + p;",
    ]
    stage = [
        "#pragma unroll",
        f"            for (int f = 0; f < {fields}; ++f) E[f] = slot_data.data[slot_base + f * {units}];",
    ]
    values = names
    if torsion:
        rows = [*head, f"            float E[{fields}];", *stage]
        rows.append("            // The Gram couplings of the unit's rows: per coordinate, body a's then b's products.")
    else:
        arrays = {"mf_Z_a": 0, "mf_Z_b": 1}
        op = FreeBodyZ(
            Lanes(1),
            velocity=Velocity("s_v"),
            entry=lambda array, row, k: f"E[{ {'i': 'r', 'sib': '2'}[row] } * 12 + {arrays[array] * 6} + {k}]",
            inverse=lambda row: "mf_diag" if row == "i" else "__int_as_float(W[2 * 3])",
        )
        rows = [
            *head,
            f"            const int word_base = unit_slot * {words * units} + p;",
            f"            float E[{fields}];",
            f"            int W[{words}];",
            *stage,
            "#pragma unroll",
            f"            for (int f = 0; f < 9; ++f) W[f] = slot_words.data[word_base + f * {units}];",
            _colored_pair_metric(op, " " * 12),
            "            // The Gram couplings of the pair's tangents with the normal, Z_n . Z_t1 and Z_n . Z_t2: per coordinate,",
            "            // body a's then b's products.",
        ]
        values = [*_PAIR_METRIC, *names]

    def products(side: int) -> list[str]:
        entry = lambda row: f"E[{12 * row + 6 * side} + k]" if 12 * row + 6 * side else "E[k]"  # noqa: E731
        return [f"                    {name} += {entry(a)} * {entry(b)};" for name, a, b in couplings]

    rows += [
        f"            float {', '.join(f'{name} = 0.0f' for name in names)};",
        "            for (int k = 0; k < 6; ++k) {",
        "                if (dof_a >= 0) {",
        *products(0),
        "                }",
        "                if (dof_b >= 0) {",
        *products(1),
        "                }",
        "            }",
        f"            const int at = unit_slot * {layout.metric * units} + p;",
        *(
            f"            slot_metric.data[at{f' + {f * units}' if f else ''}] = {value};"
            for f, value in enumerate(values)
        ),
        "        }",
        "    }",
        "    __syncthreads();",
    ]
    return "\n".join(rows)


def mf_gs_staging(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    *,
    mapping: str = "warp",
    shared_metadata: bool = True,
    resident_row_metadata: bool = False,
    stream_mf_impulses: bool = False,
    traces: bool = False,
    wide_rows: bool = False,
) -> Staging:
    """The shared working set of a world's matrix-free sweep block (:func:`mf_gs_source` declares it): the world's
    velocity, its dense rows' impulses and staged metadata, the free-body impulses unless they stream, and the colored
    mapping's color offsets (and its trace's warp maxima)."""
    dense = WorldRows(
        max_constraints, shared_metadata=shared_metadata, packed_types=shared_metadata or resident_row_metadata
    )
    free = FreeBodyRows(mf_max_constraints, stream=stream_mf_impulses)
    colored = ()
    if mapping == "colored":
        warps = COLORED_THREADS // 32
        colored = (Buffer("s_color_offset", "int", COLOR_CHUNK + 1),)
        if traces:
            colored += block_argmax(warps, warp="warp", lane="lane")[1]
    return Staging(
        (
            Buffer("s_v", "float", max_world_dofs),
            *((Buffer("s_sweep_v", "float", max_world_dofs),) if wide_rows else ()),
            *FACTOR_COORDINATES.buffers(max_world_dofs),
            *dense.buffers(),
            *free.buffers(),
            *colored,
        ),
        threads=COLORED_THREADS if mapping == "colored" else 32,
    )


def mf_gs_source(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    *,
    max_row_dofs: int | None = None,
    mapping: str = "warp",
    colored_worlds: bool = False,
    shared_metadata: bool = True,
    resident_row_metadata: bool = False,
    stream_mf_impulses: bool = False,
    has_dense_velocity_limit_rows: bool = True,
    skip_local_internal_worlds: bool = False,
    contact_torsion: bool = False,
    factor_coordinates: bool = False,
    skip_tier_rows: int = 0,
    independent_components: bool = False,
    solve_policy: SolvePolicy = DEFAULT_POLICY,
    lanes: int = 32,
    material: bool = False,
    seeded: bool = False,
    wide_rows: bool = False,
) -> tuple[str, Staging]:
    """The native body of the matrix-free Gauss-Seidel sweep: a world's dense rows (:func:`packed_operator`, packed
    whitened rows of ``max_row_dofs`` entries, by default as many as the world's coordinates) and its free-body rows
    (:class:`.FreeBodyZ`), then its end-of-sweep passes, on the world's velocity change from the predictor in factor
    coordinates (:class:`FactorCoordinates`), from zero, or with ``seeded`` from the response of the impulses the
    rows carry (:func:`.sweep.impulse_seed`). The launch publishes it onto ``v_out``, which holds the predictor.

    ``mapping`` places the free-body rows on threads (:data:`MAPPINGS`): ``"warp"``, the world's warp visits them in
    row order; ``"colored"``, a block of :data:`COLORED_THREADS` visits their colors in order and a color's units at
    once, the world's first warp running everything else. Both are the same row-order Gauss-Seidel sweep. Each world's
    mapping is the unit builder's (``world_colored``, :class:`.sweep_units.ColoredRule`): the colored kernel returns
    for the warp's worlds and, with ``colored_worlds`` (the solver builds both), the warp's kernel for the colored
    ones.

    ``lanes`` is the warp mapping's lane group (:func:`.launch.device_lanes`): a warp on CUDA, or one thread, which runs
    the world on either device (the CPU's mapping), so that body composes for both (:func:`.dual_native`).

    Returns the body and its staging.
    """
    if mapping not in MAPPINGS:
        raise ValueError(f"mapping must be one of {MAPPINGS}, got {mapping!r}")
    colored = mapping == "colored"
    if resident_row_metadata and shared_metadata:
        raise ValueError("resident_row_metadata applies to streamed metadata (shared_metadata=False)")
    if factor_coordinates and (has_dense_velocity_limit_rows or skip_local_internal_worlds):
        raise ValueError("factor coordinates require the paired contact solve")
    if independent_components and not factor_coordinates:
        raise ValueError("Independent components require the paired factor-coordinate solve")
    if lanes not in (1, 32) or (colored and lanes != 32):
        raise ValueError(f"the warp mapping runs 32 lanes or one thread, the colored mapping 32; got {lanes}")
    _validate_dense_metadata_encoding(max_constraints)
    M_D, M_MF, D = max_constraints, mf_max_constraints, max_world_dofs
    max_row_dofs = int(max_world_dofs if max_row_dofs is None else max_row_dofs)
    row_lanes = Lanes(lanes)
    laws = RowLaws(regularize=True, material=material, torsion=contact_torsion, policy=solve_policy)
    dense = WorldRows(
        M_D, shared_metadata=shared_metadata, packed_types=shared_metadata or resident_row_metadata, lanes=row_lanes
    )
    free = FreeBodyRows(M_MF, stream=stream_mf_impulses, lanes=row_lanes)
    dense_op = packed_operator(max_row_dofs, row_lanes, wide=wide_rows)
    bodies = FACTOR_COORDINATES
    free_op = FreeBodyZ(row_lanes, factor=bodies.factor)
    staging = mf_gs_staging(
        M_D,
        M_MF,
        D,
        mapping=mapping,
        shared_metadata=shared_metadata,
        resident_row_metadata=resident_row_metadata,
        stream_mf_impulses=stream_mf_impulses,
        traces=solve_policy.traces,
        wide_rows=wide_rows,
    )

    # Ownership: the paired owner's signed selector arrives in local_solve_owner under independent components.
    dense_count = "int m_dense = world_constraint_count.data[world];"
    component = owned = ""
    if independent_components:
        dense_count = (
            "const bool split_component = local_solve_owner.data[world] < 0;\n"
            "    int m_dense = split_component ? 0 : world_constraint_count.data[world];"
        )
        component = (
            "\n    const int first_dofs = split_component ? mf_meta.data[off_meta] : 0;"
            "\n    const int component_offset = split_component ? "
            "max(first_dofs >> 16, static_cast<int>(static_cast<short>(first_dofs & 65535))) : 0;"
        )
        owned = " && (!split_component || (d >= component_offset && d < component_offset + 6))"
    skips = []
    if factor_coordinates:
        skips.append("    if (m_mf == 0) return;")
    if skip_local_internal_worlds:
        skips.append(f"    if (local_solve_owner.data[world] != {PGS_LOCAL_SOLVE_OWNER_GENERAL}) return;")
    if skip_tier_rows > 0:
        skips.append(f"    if (m_mf == 0 && m_dense <= {int(skip_tier_rows)}) return;")
    mapped = ""
    if colored or colored_worlds:
        # The worlds of the other mapping, as the unit builder assigned them at the pack.
        mapped = f"\n    if (world_colored.data[world] {'==' if colored else '!='} 0) return;"
    dense_loads = indent(dense.loads("i"), "        ")
    mf_load = free.load_line("i")
    mf_store = free.store_line("i")

    passes = []
    if has_dense_velocity_limit_rows:
        passes.append(velocity_limit_pass(dense, dense_op, laws, pad="        "))
    passes.append(speed_rows_pass(free, free_op, laws, pad="        "))

    if colored:
        # The block's vote: every warp's rows and units (a barrier); the first thread records the solve.
        quiet = "!__syncthreads_or(iteration_changed != 0)"
        sweep_end = sweep_exit(
            solve_policy,
            quiet=quiet,
            iteration="iter",
            lane="tid",
            argmax="BLOCK_ARGMAX",
        )
        head = "    const int lane = LANE;\n    const int tid = (int)threadIdx.x;\n    const int warp = tid >> 5;"
        first_warp, end_first_warp, barrier = "    if (warp == 0) {\n", "    }\n", "__syncthreads();"
        units = sweep_units.unit_capacity(M_MF)
        free_phase = _colored_units(free, laws, M_MF, units)
        trace_reduction = ""
        if solve_policy.traces:
            trace_reduction = block_argmax(COLORED_THREADS // 32, warp="warp", lane="lane")[0]
        # The world's storage slot (a colored world has one): its colors and staged units.
        offsets = (
            "\n    const int unit_slot = world_slot.data[world];"
            f"\n    const int off_unit = unit_slot * {units};"
            f"\n    const int off_color = unit_slot * {units + 1};"
        )
    else:
        sweep_end = end_sweep(solve_policy, iteration="iter")
        head = "    const int lane = LANE;"
        first_warp = end_first_warp = ""
        barrier = "SYNC();"
        free_phase = free_body_rows_block(free, free_op, laws, admission=_MF_ADMISSION, pad="    ")
        offsets = trace_reduction = ""

    thread, threads = ("tid", "(int)blockDim.x") if colored else ("lane", "STRIDE")
    sync = "__syncthreads();" if colored else "SYNC();"
    metric_build = _colored_metrics(sweep_units.unit_capacity(M_MF), torsion=contact_torsion) if colored else ""
    seed = ""
    if seeded:
        seeds = (
            impulse_seed(
                "m_dense",
                dense.peek("i"),
                f"({dense.kind('i', scan=True)})",
                dense_op.apply_row("i", "seed", "        "),
                row_lanes.sync(),
            ),
            impulse_seed(
                "m_mf",
                free.load("i"),
                "(mf_meta.data[off_meta + i * 4 + 3] & 0xFFFF)",
                free_op.apply_row("i", "seed", "        "),
                row_lanes.sync(),
                declare=(
                    "const int packed_dofs = mf_meta.data[off_meta + i * 4];",
                    "const int dof_a = packed_dofs >> 16, dof_b = (packed_dofs << 16) >> 16;",
                ),
            ),
        )
        seed = f"{first_warp}{chr(10).join(seeds)}\n{end_first_warp}    {barrier}\n"
    # Wide rows, which the bilateral block corrected, can be collinear (a jammed loop), and collinear rows can trade
    # impulses forever with no effect on the velocity, each loud alone. The velocity is the solve's answer, so a wide
    # sweep also ends when no coordinate moved past the exit law's floor of the largest one: factor coordinates share
    # the kinetic energy's metric, and a coordinate the rows cancel to zero keeps their rounding, far above its own size.
    sweep_vote = ""
    if wide_rows:
        metric_build += f"""
    for (int d = {thread}; d < {D}; d += {threads}) s_sweep_v[d] = s_v[d];"""
        sweep_vote = f"""        {barrier}
        int velocity_moved = 0;
        float velocity_floor = 0.0f;
        for (int d = 0; d < {D}; ++d) velocity_floor = fmaxf(velocity_floor, fabsf(s_v[d]));
        velocity_floor *= {QUIET_LITERAL};
        for (int d = {thread}; d < {D}; d += {threads}) {{
            const float now = s_v[d];
            if (!(fabsf(now - s_sweep_v[d]) <= velocity_floor)) velocity_moved = 1;
            s_sweep_v[d] = now;
        }}
        if ({"!__syncthreads_or(velocity_moved != 0)" if colored else "BALLOT(velocity_moved != 0) == 0u"}) iteration_changed = 0;
"""
    body = f"""{head}

    {dense_count}
    int m_mf = mf_constraint_count.data[world];
    if (m_dense == 0 && m_mf == 0) return;
    if (m_dense > {M_D}) m_dense = {M_D};
    if (m_mf > {M_MF}) m_mf = {M_MF};
{chr(10).join(skips)}
    int mf_contact_end = mf_contact_rows_end.data[world];
    if (mf_contact_end > m_mf) mf_contact_end = m_mf;{mapped}

    int dof_map_base = world * {D};
    const int line = dense_first.data[world];
    __builtin_assume(line >= 0);
    const int mf_line = mf_first.data[world];
    __builtin_assume(mf_line >= 0);
    int off_dense = line * {M_D};
    int off_mf = mf_line * {M_MF};
    int off_meta = off_mf * 4;{component}
    const size_t packed_base = (size_t)line * {M_D * int(max_row_dofs)};
    int mf6_base = mf_line * {M_MF} * 6;{offsets}

{staging.declarations("    ")}
{free.declaration()}{trace_reduction}

{first_warp}    for (int i = lane; i < m_dense; i += STRIDE) {{
{dense_loads}
    }}
    for (int i = lane; i < m_mf; i += STRIDE) {{
        {mf_load}
    }}
    for (int d = lane; d < {D}; d += STRIDE) s_v[d] = 0.0f;
{indent(bodies.clear(D, "lane"), "    ")}
{end_first_warp}    {barrier}
    // The articulations this launch's rows touch: the bodies of its free-body rows, the sides of every dense row.
{indent(bodies.mark("0", "m_mf", thread, threads), "    ")}
{indent(bodies.mark_sides("m_dense", thread, threads), "    ")}
    {barrier}
{seed}{metric_build}

    for (int iter = 0; iter < iterations; iter++) {{
        int iteration_changed = 0;

{first_warp}{dense_rows_block(dense, dense_op, laws, admission=_DENSE_ADMISSION, pad="        ")}
{end_first_warp}{"        __syncthreads();" + chr(10) if colored else ""}
{indent(free_phase, "    ")}

{first_warp}{chr(10).join(passes)}
{end_first_warp}{sweep_vote}{sweep_end}    }}

    {barrier}
{indent(bodies.leave_all(D, thread, threads, sync), "    ")}
{first_warp}    for (int d = lane; d < {D}; d += STRIDE) {{
        int global_dof = world_dof_indices.data[dof_map_base + d];
        if (global_dof >= 0{owned}) v_out.data[global_dof] += s_v[d];
    }}
    for (int i = lane; i < m_dense; i += STRIDE) {{
        world_impulses.data[off_dense + i] = s_lam_dense[i];
    }}
    for (int i = lane; i < m_mf; i += STRIDE) {{
        {mf_store}
    }}
{end_first_warp}"""
    preludes = (
        lane_group(lanes),
        SHARED_STORAGE,
        RING,
        PACKED_SIDES,
        *law_preludes(PROJECTION_LAW, *((CONE_LAW,) if contact_torsion else ())),
        *solve_policy.preludes,
    )
    if lanes == 1:
        return dual_native(body, preludes=(*preludes, CPU_MATH)), staging
    return cuda_native(body + "#endif\n", preludes=preludes), staging


@cache
def mf_gs_kernel(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    device_arch: str,
    *,
    max_row_dofs: int | None = None,
    mapping: str = "warp",
    colored_worlds: bool = False,
    shared_metadata: bool = True,
    resident_row_metadata: bool = False,
    stream_mf_impulses: bool = False,
    has_dense_velocity_limit_rows: bool = True,
    skip_local_internal_worlds: bool = False,
    local_internal_max_constraints: int = 0,
    contact_torsion: bool = False,
    factor_coordinates: bool = False,
    skip_tier_rows: int = 0,
    independent_components: bool = False,
    solve_policy: SolvePolicy = DEFAULT_POLICY,
    material: bool = False,
    seeded: bool = False,
    wide_rows: bool = False,
) -> wp.Kernel:
    """The matrix-free Gauss-Seidel sweep kernel of a thread mapping (:func:`mf_gs_source`): the warp's launches a
    warp per world item (one thread on the CPU, :func:`.launch.device_lanes`) and takes :data:`MF_GS_ABI`
    (:data:`COLORED_WORLDS_ABI` with ``colored_worlds``), the colored one :data:`COLORED_THREADS` and takes
    :data:`COLORED_ABI`."""
    max_row_dofs = int(max_world_dofs if max_row_dofs is None else max_row_dofs)
    snippet, staging = mf_gs_source(
        max_constraints,
        mf_max_constraints,
        max_world_dofs,
        max_row_dofs=max_row_dofs,
        mapping=mapping,
        colored_worlds=colored_worlds,
        shared_metadata=shared_metadata,
        resident_row_metadata=resident_row_metadata,
        stream_mf_impulses=stream_mf_impulses,
        has_dense_velocity_limit_rows=has_dense_velocity_limit_rows,
        skip_local_internal_worlds=skip_local_internal_worlds,
        contact_torsion=contact_torsion,
        factor_coordinates=factor_coordinates,
        skip_tier_rows=skip_tier_rows,
        independent_components=independent_components,
        solve_policy=solve_policy,
        lanes=32 if mapping == "colored" else device_lanes(device_arch),
        material=material,
        seeded=seeded,
        wide_rows=wide_rows,
    )
    colored = mapping == "colored"
    staging.check(f"the {mapping} matrix-free sweep", arch_budget(device_arch))
    name = (
        f"sweep_mf_{'colored' if colored else 'gs'}_{max_constraints}_{mf_max_constraints}_{max_world_dofs}"
        f"_s{max_row_dofs}"
        f"_densevlim{int(has_dense_velocity_limit_rows)}"
        + ("" if shared_metadata else "_gmeta")
        + ("_smeta" if resident_row_metadata else "")
        + ("_glambda" if stream_mf_impulses else "")
        + (f"_local_fallback{local_internal_max_constraints}" if skip_local_internal_worlds else "")
        + ("_torsion" if contact_torsion else "")
        + ("_factor" if factor_coordinates else "")
        + (f"_skiptier{int(skip_tier_rows)}" if skip_tier_rows > 0 else "")
        + ("_components" if independent_components else "")
        + ("_cw" if colored_worlds and not colored else "")
        + ("_material" if material else "")
        + ("_seeded" if seeded else "")
        + ("_wide" if wide_rows else "")
        + solve_policy.suffix
    )
    return KernelTemplate.from_abi(
        name,
        COLORED_ABI if colored else COLORED_WORLDS_ABI if colored_worlds else MF_GS_ABI,
        snippet,
        kernel_template.WORLD_QUEUE,
        extra=(*material_arguments(material), *solve_policy.arguments),
        composed=True,
        native_name="sweep_mf_colored_native" if colored else "sweep_mf_gs_native",
        launch_bounds=(COLORED_THREADS, 1) if colored else None,
    ).kernel()
