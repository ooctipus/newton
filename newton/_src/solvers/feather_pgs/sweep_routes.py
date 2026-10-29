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
from .contact_torsion import torque_sweep_source
from .device_preludes import RING, cuda_native, lane_group
from .kernel_template import KernelTemplate
from .kernels import PGS_LOCAL_SOLVE_OWNER_GENERAL, BodyFactors
from .law_source import law_preludes
from .packed_rows import PACKED_SIDES
from .projection_laws import PROJECTION_LAW
from .response_operator import CholeskySolve, FreeBodyZ, Lanes, SparseFactor, Velocity
from .row_metadata import _validate_dense_metadata_encoding
from .solve_policy import DEFAULT_POLICY, SolvePolicy, block_argmax, sweep_exit
from .staging import Buffer, Staging, arch_budget
from .sweep import (
    CONTACT,
    DRIVE,
    FRICTION,
    JOINT_LIMIT,
    VELOCITY_LIMIT,
    FreeBodyRows,
    RowContext,
    RowLaws,
    WorldRows,
    dense_rows_block,
    end_sweep,
    free_body_rows_block,
    fused_clamp_pass,
    indent,
    row_visit,
    speed_rows_pass,
    velocity_limit_pass,
)
from .velocity_limit_law import check_velocity_limit_law, velocity_limit_law_suffix

# Dense row types each row phase visits (MF-GS ``row_phase``): 0 interleaved (every row; velocity limits in the end
# pass), 1/2 contact-then-internal, 3/4/5 the PhysX grasp schedule (internal, contacts, velocity limits), 6 dense
# rows only.
_MIMIC, _CONNECT = 5, 6
_PHASE_DENSE_KINDS = {
    1: (CONTACT, FRICTION),
    2: (DRIVE, JOINT_LIMIT, VELOCITY_LIMIT, _MIMIC, _CONNECT),
    3: (DRIVE, JOINT_LIMIT, _MIMIC, _CONNECT),
    4: (CONTACT, FRICTION),
    5: (VELOCITY_LIMIT,),
}


def _phase_admission() -> str:
    lines = []
    for phase, kinds in _PHASE_DENSE_KINDS.items():
        test = " && ".join(f"row_type != {kind}" for kind in kinds)
        lines.append(f"if (row_phase == {phase} && {test}) continue;")
    lines.append(f"if ((row_phase == 0 || row_phase == 2 || row_phase == 6) && row_type == {VELOCITY_LIMIT}) continue;")
    lines.append(f"if (freeze_drive_rows != 0 && row_type == {DRIVE}) continue;")
    return "\n".join(lines)


_MF_PHASE_ADMISSION = (
    f"if ((row_phase == 1 || row_phase == 4) && mf_rt != {CONTACT} && mf_rt != {FRICTION}) continue;\n"
    f"if (row_phase == 0 && mf_rt == {VELOCITY_LIMIT}) continue;"
)

MF_GS_ABI = kernel_template.params(
    ("world", int),
    ("world_constraint_count", wp.array[int]),
    ("dense_phase_bounds", wp.array2d[int]),
    ("local_solve_owner", wp.array[int]),
    ("world_dof_indices", wp.array2d[int]),
    ("rhs_bias", wp.array2d[float]),
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
    ("world_torsion_group", wp.array2d[int]),
    ("contact_torsion_radius", float),
    ("world_drive_target_vel_bias", wp.array2d[float]),
    ("world_drive_vel_multiplier", wp.array2d[float]),
    ("world_drive_impulse_multiplier", wp.array2d[float]),
    ("world_drive_max_impulse", wp.array2d[float]),
    ("world_drive_vel_limit", wp.array2d[float]),
    ("mf_constraint_count", wp.array[int]),
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
    ("row_phase", int),
    ("friction_start_iteration", int),
    ("iteration_offset", int),
    ("freeze_drive_rows", int),
    ("v_out", wp.array[float]),
)
"""The matrix-free sweep's native ABI: one world per call (the world queue's dispatch)."""


class FactorCoordinates:
    """The articulations a matrix-free sweep holds in factor coordinates, and their conversions (native).

    The articulations the launch's rows touch are marked at their first world coordinate in a bitmap
    (``s_factor_marks``): each free body of its free-body rows and each side of its dense rows
    (:func:`.dense_rows.row_sides`). Each enters factor coordinates ``u = L^T P v`` once the velocity is staged and
    leaves them, ``v = P^T L^-T u``, before it is written back (:class:`.CholeskySolve`'s ``enter`` and ``leave``), so
    every row the sweep visits reads and updates ``u``: ``Z^T u`` and ``u += Z delta``. An articulation's factor is
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
        return f"for (int w = {lane}; w < {cls.words(dofs)}; w += 32) s_factor_marks[w] = 0u;"

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

    def convert_all(self, dofs: int, thread: str, threads: str, sync: str, *, enter: bool) -> str:
        """Every marked articulation enters (or leaves) factor coordinates in place, a thread per bitmap word."""
        factors = self.op.factors
        action = "enter" if enter else "leave"

        def solve(size):
            if size == 6:
                return getattr(CholeskySolve(6, "factor", unroll=True), action)
            return getattr(CholeskySolve("size", "factor"), action)

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
{indent(solve(6)(coordinates, coordinates), "            ")}
        }} else {{
{indent(solve(0)(coordinates, coordinates), "            ")}
        }}
    }}
}}
{sync}"""


FACTOR_COORDINATES = FactorCoordinates()
"""Where a matrix-free sweep keeps its articulations' coordinates (:class:`FactorCoordinates`)."""


def packed_operator(support: int, lanes: Lanes) -> SparseFactor:
    """The dense rows of a matrix-free sweep: packed whitened rows of ``support`` entries (:mod:`.packed_rows`), their
    world coordinates from their sides (``PACKED_COORDINATE``) and values from the world's first row
    (``packed_base``), over the factor velocity the sweep stages (``s_v``)."""

    def coordinate(row, entry):
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

METRIC_FIELDS = sweep_units.METRIC_FIELDS

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
class _CachedPairBodyZ(FreeBodyZ):
    """The colored unit's operator: a friction pair's coupling feeds only its metric, which the unit's metric cache
    holds (:func:`_colored_metrics`), so the visit forms none."""

    def coupling(self, out: str, row: str, sibling: str, indent: str) -> str:
        return f"{indent}const float {out} = 0.0f;  // the cached metric replaces the coupling"


@dataclasses.dataclass(frozen=True)
class _GramUnitBodyZ(_CachedPairBodyZ):
    """The colored unit's rows on the unit's cached Gram block ``G = Z^T Z`` of its normal and friction rows.

    Every row's velocity term is reduced from the unit's velocity ``u`` before the unit's first row
    (``unit_jv0``..``unit_jv2``, :func:`_unit_terms`), at once; the friction pair's are then corrected by the normal
    row's change through the cached couplings, ``Z_t^T (u + Z_n dn) = Z_t^T u + G_tn dn``. The applies are deferred
    (``unit_dn``, ``unit_dt1``, ``unit_dt2``) to one pass over the unit's coordinates after its rows
    (:func:`_unit_apply`): in exact arithmetic the sequential sweep, with one dot product round and one apply round
    off the pair's critical path.
    """

    def residual(self, out: str, indent: str) -> str:
        return f"{indent}float {out} = r == 0 ? unit_jv0 : unit_jv1 + gram_n1 * unit_dn;"

    def sibling_residual(self, out: str, sibling: str, indent: str, rhs: str) -> str:
        return f"{indent}float {out} = (unit_jv2 + gram_n2 * unit_dn) + {rhs};"

    def apply(self, delta: str, indent: str) -> str:
        return f"\n{indent}if (r == 0) unit_dn += {delta}; else unit_dt1 += {delta};"

    def apply_row(self, row: str, delta: str, indent: str) -> str:
        return f"\n{indent}unit_dt2 += {delta};"


def _unit_terms(pad: str) -> str:
    """Native statements, after a unit's ``E``, ``metric``, ``rows`` and ``dof_a``, ``dof_b`` are declared: the unit's
    velocity on its bodies (``unit_u``), its rows' velocity terms from it (each side's six products, then the sides
    added), the cached Gram couplings of the pair with the normal, and the deferred changes."""
    lines = [
        "float unit_u[12];",
        "#pragma unroll",
        "for (int k = 0; k < 6; ++k) {",
        "    unit_u[k] = dof_a >= 0 ? s_v[dof_a + k] : 0.0f;",
        "    unit_u[6 + k] = dof_b >= 0 ? s_v[dof_b + k] : 0.0f;",
        "}",
        "float unit_jv0 = 0.0f, unit_jv1 = 0.0f, unit_jv2 = 0.0f;",
        "#pragma unroll",
        "for (int side = 0; side < 2; ++side) {",
        "    if ((side == 0 ? dof_a : dof_b) < 0) continue;",
        "    float n = 0.0f, t1 = 0.0f, t2 = 0.0f;",
        "#pragma unroll",
        "    for (int k = 0; k < 6; ++k) {",
        "        const float u = unit_u[side * 6 + k];",
        "        n += E[side * 6 + k] * u;",
        "        t1 += E[12 + side * 6 + k] * u;",
        "        t2 += E[24 + side * 6 + k] * u;",
        "    }",
        "    unit_jv0 += n;",
        "    unit_jv1 += t1;",
        "    unit_jv2 += t2;",
        "}",
        "const float gram_n1 = metric[5], gram_n2 = metric[6];",
        "float unit_dn = 0.0f, unit_dt1 = 0.0f, unit_dt2 = 0.0f;",
    ]
    return indent("\n".join(lines), pad)


def _unit_apply(pad: str) -> str:
    """The unit's deferred changes in one pass: ``u += Z_n dn + Z_t1 dt1 + Z_t2 dt2`` on each of its bodies."""
    lines = [
        "if (unit_dn != 0.0f || unit_dt1 != 0.0f || unit_dt2 != 0.0f) {",
        "#pragma unroll",
        "    for (int k = 0; k < 6; ++k) {",
        "        if (dof_a >= 0)",
        "            s_v[dof_a + k] = unit_u[k] + ((E[k] * unit_dn + E[12 + k] * unit_dt1) + E[24 + k] * unit_dt2);",
        "        if (dof_b >= 0)",
        "            s_v[dof_b + k] = unit_u[6 + k] + ((E[6 + k] * unit_dn + E[18 + k] * unit_dt1) + E[30 + k] * unit_dt2);",
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

    def unit_offset(row: str) -> str:
        offsets = {"i": "r", "sib": "2"}
        if row not in offsets:
            raise ValueError(f"the colored unit reads rows i and sib, not {row!r}")
        return offsets[row]

    unit_op = _GramUnitBodyZ(
        Lanes(1),
        velocity=Velocity("s_v"),
        entry=lambda array, row, k: f"E[{unit_offset(row)} * 12 + {arrays[array] * 6} + {k}]",
        inverse=lambda row: "mf_diag" if row == "i" else f"__int_as_float(W[{unit_offset(row)} * 3])",
    )
    ring, members = sweep_units.SLOT_RING, sweep_units.RING_MEMBERS

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
        mu=lambda row: f"E[{sweep_units.SLOT_MU}]",
        weight=lambda row: f"E[{sweep_units.SLOT_WEIGHT}]",
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
            "            // A friction pair's law from its unit's cached metric (formed once per launch, before the sweeps), which",
            "            // the unit loads with its fields.",
            f"            float metric[{METRIC_FIELDS}];",
            "            const auto FRICTION_PAIR_CACHED = [&](int, float, float, float, wp::vec2 residual, wp::vec2 old,",
            "                                                  float radius, float relaxation) -> wp::vec2 {",
            "                return ProjectionLaw::friction_pair_from_metric(metric[0], metric[1], metric[2],",
            "                    wp::vec2(metric[3], metric[4]), residual, old, radius, relaxation);",
            "            };",
        ]
    )
    fields, words = sweep_units.SLOT_ENTRIES + 2, sweep_units.SLOT_WORDS
    visit = row_visit(unit_ctx, laws, parent=unit_rows.parent_of("packed_tp"), check="mf_diag", pad=" " * 24)
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
                    for (int f = 0; f < {METRIC_FIELDS}; ++f)
                        metric[f] = slot_metric.data[unit_slot * {METRIC_FIELDS * units} + p + f * {units}];
{_unit_terms(" " * 20)}
#pragma unroll
                    for (int r = 0; r < 2; ++r) {{
                        if (r >= rows) break;
                        // One pass of the row law; its `continue` leaves the row.
                        for (int once = 0; once < 1; ++once) {{
                            const int i = unit_row + r;
                            int4 meta;
                            meta.x = 0;
                            meta.y = W[r * 3];
                            meta.z = W[r * 3 + 1];
                            meta.w = W[r * 3 + 2];
                            const int packed_tp = meta.w;
                            // The unit builder's rows: a contact, then its friction pair (each copy knows its kind).
                            const int mf_rt = r == 0 ? {CONTACT} : {FRICTION};
                            const float mf_diag = __int_as_float(meta.y);
{visit}
                        }}
                    }}
{_unit_apply(" " * 20)}
                    // The pair's second tangent: the first's visit solved it; only the delayed-friction gate holds it.
                    if (rows == 3 && ProjectionLaw::friction_gate(W[2 * 3 + 2] & 0xFFFF, global_iter, friction_start_iteration))
                        s_lam_mf[unit_row + 2] = 0.0f;
                }}
                __syncthreads();
            }}"""


def _colored_metrics(units: int) -> str:
    """Before the sweeps, every thread forms the friction pair metric of the world's staged friction units (a unit of
    three rows) at block stride and stores it in its slot's metric cache (``slot_metric``: scale, the two
    eigenvalues and the larger's axis, then the pair's Gram couplings with the normal, field by field across the
    units), which the pair law and the unit's terms read at every visit."""
    fields, words = sweep_units.SLOT_ENTRIES + 2, sweep_units.SLOT_WORDS
    arrays = {"mf_Z_a": 0, "mf_Z_b": 1}
    op = FreeBodyZ(
        Lanes(1),
        velocity=Velocity("s_v"),
        entry=lambda array, row, k: f"E[{ {'i': 'r', 'sib': '2'}[row] } * 12 + {arrays[array] * 6} + {k}]",
        inverse=lambda row: "mf_diag" if row == "i" else "__int_as_float(W[2 * 3])",
    )
    return f"""    {{
        const int staged = color_offset.data[off_color + color_count.data[world]];
        for (int p = tid; p < staged; p += blockDim.x) {{
            const int4 slot = __ldg(reinterpret_cast<const int4*>(&color_slot.data[(off_unit + p) * 4]));
            if ((slot.y & 3) != 3) continue;
            const int dof_a = slot.z, dof_b = slot.w;
            const int slot_base = unit_slot * {fields * units} + p;
            const int word_base = unit_slot * {words * units} + p;
            float E[{fields}];
            int W[{words}];
#pragma unroll
            for (int f = 0; f < {fields}; ++f) E[f] = slot_data.data[slot_base + f * {units}];
#pragma unroll
            for (int f = 0; f < 9; ++f) W[f] = slot_words.data[word_base + f * {units}];
{_colored_pair_metric(op, " " * 12)}
            // The Gram couplings of the pair's tangents with the normal, Z_n . Z_t1 and Z_n . Z_t2: per coordinate,
            // body a's then b's products.
            float gram_n1 = 0.0f, gram_n2 = 0.0f;
            for (int k = 0; k < 6; ++k) {{
                if (dof_a >= 0) {{
                    gram_n1 += E[k] * E[12 + k];
                    gram_n2 += E[k] * E[24 + k];
                }}
                if (dof_b >= 0) {{
                    gram_n1 += E[6 + k] * E[18 + k];
                    gram_n2 += E[6 + k] * E[30 + k];
                }}
            }}
            const int at = unit_slot * {METRIC_FIELDS * units} + p;
            slot_metric.data[at] = metric.v0;
            slot_metric.data[at + {units}] = metric.v1;
            slot_metric.data[at + {2 * units}] = metric.v2;
            slot_metric.data[at + {3 * units}] = metric.v3[0];
            slot_metric.data[at + {4 * units}] = metric.v3[1];
            slot_metric.data[at + {5 * units}] = gram_n1;
            slot_metric.data[at + {6 * units}] = gram_n2;
        }}
    }}
    __syncthreads();"""


def mf_gs_staging(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    *,
    mapping: str = "warp",
    shared_metadata: bool = True,
    resident_row_metadata: bool = False,
    stream_mf_impulses: bool = False,
    has_drive_rows: bool = True,
    fuse_vel_limits: bool = False,
    traces: bool = False,
) -> Staging:
    """The shared working set of a world's matrix-free sweep block (:func:`mf_gs_source` declares it): the world's
    velocity, its dense rows' impulses and staged metadata, the free-body impulses unless they stream, and the colored
    mapping's color offsets (and its trace's warp maxima)."""
    dense = WorldRows(
        max_constraints,
        shared_metadata=shared_metadata,
        packed_types=shared_metadata or resident_row_metadata,
        drive=has_drive_rows,
        fused_limits=fuse_vel_limits,
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
    has_drive_rows: bool = True,
    has_dense_velocity_limit_rows: bool = True,
    fuse_vel_limits: bool = False,
    skip_local_internal_worlds: bool = False,
    contact_torsion: bool = False,
    factor_coordinates: bool = False,
    skip_tier_rows: int = 0,
    independent_components: bool = False,
    velocity_limit_law: str = "stateless",
    solve_policy: SolvePolicy = DEFAULT_POLICY,
) -> tuple[str, Staging]:
    """The native body of the matrix-free Gauss-Seidel sweep: a world's dense rows (:func:`packed_operator`, packed
    whitened rows of ``max_row_dofs`` entries, by default as many as the world's coordinates) and its free-body rows
    (:class:`.FreeBodyZ`), then its end-of-sweep passes, on the world's velocity in factor coordinates
    (:class:`FactorCoordinates`).

    ``mapping`` places the free-body rows on threads (:data:`MAPPINGS`): ``"warp"``, the world's warp visits them in
    row order; ``"colored"``, a block of :data:`COLORED_THREADS` visits their colors in order and a color's units at
    once, the world's first warp running everything else. Both are the same row-order Gauss-Seidel sweep. Each world's
    mapping is the unit builder's (``world_colored``, :class:`.sweep_units.ColoredRule`): the colored kernel returns
    for the warp's worlds and, with ``colored_worlds`` (the solver builds both), the warp's kernel for the colored
    ones.

    Returns the body and its staging.
    """
    if mapping not in MAPPINGS:
        raise ValueError(f"mapping must be one of {MAPPINGS}, got {mapping!r}")
    colored = mapping == "colored"
    if resident_row_metadata and shared_metadata:
        raise ValueError("resident_row_metadata applies to streamed metadata (shared_metadata=False)")
    if fuse_vel_limits and not has_drive_rows:
        raise ValueError("fuse_vel_limits requires has_drive_rows")
    if factor_coordinates and (
        has_drive_rows or has_dense_velocity_limit_rows or fuse_vel_limits or skip_local_internal_worlds
    ):
        raise ValueError("factor coordinates require the paired augmented-drive contact solve")
    if independent_components and not factor_coordinates:
        raise ValueError("Independent components require the paired factor-coordinate solve")
    _validate_dense_metadata_encoding(max_constraints)
    accumulated = check_velocity_limit_law(velocity_limit_law) == "accumulated"
    M_D, M_MF, D = max_constraints, mf_max_constraints, max_world_dofs
    max_row_dofs = int(max_world_dofs if max_row_dofs is None else max_row_dofs)
    lanes = Lanes(32)
    laws = RowLaws(
        drive=has_drive_rows,
        velocity_limit_law="accumulated" if accumulated else "stateless",
        regularize=True,
        torsion=contact_torsion,
        policy=solve_policy,
    )
    dense = WorldRows(
        M_D,
        shared_metadata=shared_metadata,
        packed_types=shared_metadata or resident_row_metadata,
        drive=has_drive_rows,
        fused_limits=fuse_vel_limits,
        lanes=lanes,
    )
    free = FreeBodyRows(M_MF, stream=stream_mf_impulses, lanes=lanes)
    dense_op = packed_operator(max_row_dofs, lanes)
    bodies = FACTOR_COORDINATES
    free_op = FreeBodyZ(lanes, factor=bodies.factor)
    staging = mf_gs_staging(
        M_D,
        M_MF,
        D,
        mapping=mapping,
        shared_metadata=shared_metadata,
        resident_row_metadata=resident_row_metadata,
        stream_mf_impulses=stream_mf_impulses,
        has_drive_rows=has_drive_rows,
        fuse_vel_limits=fuse_vel_limits,
        traces=solve_policy.traces,
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
    load_lo = "    int load_lo = (row_phase == 5) ? 0 : dense_lo;" if fuse_vel_limits else ""
    empty_limit_phase = (
        "" if fuse_vel_limits else "    if (row_phase == 5 && dense_lo == dense_hi && mf_contact_end == m_mf) return;"
    )
    dense_loads = indent(dense.loads("i"), "        ")
    mf_load = free.load_line("i")
    mf_store = free.store_line("i")

    passes = []
    if has_dense_velocity_limit_rows:
        passes.append(
            "        if (row_phase == 0 || row_phase == 2) {\n"
            + velocity_limit_pass(dense, dense_op, laws, pad="            ")
            + "\n        }"
        )
    if fuse_vel_limits:
        passes.append(
            "        if (row_phase == 0 || row_phase == 2 || row_phase == 5) {\n"
            + fused_clamp_pass(dense, dense_op, laws, pad="            ")
            + "\n        }"
        )
    passes.append(
        "        if (row_phase == 0 || row_phase == 2 || row_phase == 5) {\n"
        + speed_rows_pass(free, free_op, laws, pad="            ")
        + "\n        }"
    )
    if contact_torsion:
        passes.append(torque_sweep_source(dense_op, solve_policy))

    if colored:
        # The block's vote: every warp's rows and units (a barrier); the first thread records the solve.
        quiet = "!__syncthreads_or(iteration_changed != 0)"
        sweep_end = sweep_exit(
            solve_policy,
            quiet=quiet,
            gate="global_iter >= friction_start_iteration",
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
        free_phase = free_body_rows_block(free, free_op, laws, admission=_MF_PHASE_ADMISSION, pad="        ")
        offsets = trace_reduction = ""

    thread, threads = ("tid", "(int)blockDim.x") if colored else ("lane", "32")
    sync = "__syncthreads();" if colored else "SYNC();"
    metric_build = _colored_metrics(sweep_units.unit_capacity(M_MF)) if colored else ""
    # The launch's prologue after the marks: the marked articulations entering factor coordinates, and the colored
    # units' pair metrics.
    prologue = "\n".join(
        part for part in (bodies.convert_all(D, thread, threads, sync, enter=True), metric_build) if part
    )
    body = f"""{head}

    {dense_count}
    int m_mf = mf_constraint_count.data[world];
    if (m_dense == 0 && m_mf == 0) return;
    if (m_dense > {M_D}) m_dense = {M_D};
    if (m_mf > {M_MF}) m_mf = {M_MF};
{chr(10).join(skips)}
    int mf_contact_end = mf_contact_rows_end.data[world];
    if (mf_contact_end > m_mf) mf_contact_end = m_mf;{mapped}

    // Row phases own contiguous dense ranges: 3 [0, bounds[0]), 5 [bounds[0], bounds[1]), 4 [bounds[1], m).
    int dense_lo = 0;
    int dense_hi = m_dense;
    if (row_phase == 3) {{
        dense_hi = min(dense_phase_bounds.data[world * 2 + 0], m_dense);
    }} else if (row_phase == 5) {{
        dense_lo = min(dense_phase_bounds.data[world * 2 + 0], m_dense);
        dense_hi = min(dense_phase_bounds.data[world * 2 + 1], m_dense);
    }} else if (row_phase == 4) {{
        dense_lo = min(dense_phase_bounds.data[world * 2 + 1], m_dense);
    }}
{load_lo}
{empty_limit_phase}

    int dof_map_base = world * {D};
    int off_dense = world * {M_D};
    int off_mf = world * {M_MF};
    int off_meta = off_mf * 4;{component}
    const size_t packed_base = (size_t)world * {M_D * int(max_row_dofs)};
    int mf6_base = world * {M_MF} * 6;
    int mf_load_lo = (row_phase == 5) ? mf_contact_end : 0;{offsets}

{staging.declarations("    ")}
{free.declaration()}{trace_reduction}

{first_warp}    for (int i = {"load_lo" if fuse_vel_limits else "dense_lo"} + lane; i < dense_hi; i += 32) {{
{dense_loads}
    }}
    if (row_phase != 3 && row_phase != 6) {{
        for (int i = mf_load_lo + lane; i < m_mf; i += 32) {{
            {mf_load}
        }}
    }}
    for (int d = lane; d < {D}; d += 32) {{
        int global_dof = world_dof_indices.data[dof_map_base + d];
        s_v[d] = global_dof >= 0{owned} ? v_out.data[global_dof] : 0.0f;
    }}
{indent(bodies.clear(D, "lane"), "    ")}
{end_first_warp}    {barrier}
    // The articulations this launch's rows touch: the bodies of its free-body rows, the sides of every dense row.
    if (row_phase != 3 && row_phase != 6) {{
{indent(bodies.mark("mf_load_lo", "m_mf", thread, threads), "        ")}
    }}
{indent(bodies.mark_sides("m_dense", thread, threads), "    ")}
    {barrier}
{prologue}

    for (int iter = 0; iter < iterations; iter++) {{
        int global_iter = iteration_offset + iter;
        int iteration_changed = 0;

{first_warp}{dense_rows_block(dense, dense_op, laws, admission=_phase_admission(), inline_velocity_limit=True, pad="        ")}
{end_first_warp}{"        __syncthreads();" + chr(10) if colored else ""}
        if (row_phase == 0 || row_phase == 1 || row_phase == 4) {{
{free_phase}
        }}

{first_warp}{chr(10).join(passes)}
{end_first_warp}{sweep_end}    }}

    {barrier}
{indent(bodies.convert_all(D, thread, threads, sync, enter=False), "    ")}
{first_warp}    for (int d = lane; d < {D}; d += 32) {{
        int global_dof = world_dof_indices.data[dof_map_base + d];
        if (global_dof >= 0{owned}) v_out.data[global_dof] = s_v[d];
    }}
    for (int i = dense_lo + lane; i < dense_hi; i += 32) {{
        world_impulses.data[off_dense + i] = s_lam_dense[i];
    }}
    if (row_phase != 3 && row_phase != 6) {{
        for (int i = mf_load_lo + lane; i < m_mf; i += 32) {{
            {mf_store}
        }}
    }}
{end_first_warp}#endif
"""
    preludes = (lane_group(32), RING, PACKED_SIDES, *law_preludes(PROJECTION_LAW), *solve_policy.preludes)
    return cuda_native(body, preludes=preludes), staging


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
    has_drive_rows: bool = True,
    has_dense_velocity_limit_rows: bool = True,
    fuse_vel_limits: bool = False,
    skip_local_internal_worlds: bool = False,
    local_internal_max_constraints: int = 0,
    contact_torsion: bool = False,
    factor_coordinates: bool = False,
    skip_tier_rows: int = 0,
    independent_components: bool = False,
    velocity_limit_law: str = "stateless",
    solve_policy: SolvePolicy = DEFAULT_POLICY,
) -> wp.Kernel:
    """The matrix-free Gauss-Seidel sweep kernel of a thread mapping (:func:`mf_gs_source`): the warp's launches
    32 threads per world item and takes :data:`MF_GS_ABI` (:data:`COLORED_WORLDS_ABI` with ``colored_worlds``), the
    colored one :data:`COLORED_THREADS` and takes :data:`COLORED_ABI`."""
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
        has_drive_rows=has_drive_rows,
        has_dense_velocity_limit_rows=has_dense_velocity_limit_rows,
        fuse_vel_limits=fuse_vel_limits,
        skip_local_internal_worlds=skip_local_internal_worlds,
        contact_torsion=contact_torsion,
        factor_coordinates=factor_coordinates,
        skip_tier_rows=skip_tier_rows,
        independent_components=independent_components,
        velocity_limit_law=velocity_limit_law,
        solve_policy=solve_policy,
    )
    colored = mapping == "colored"
    staging.check(f"the {mapping} matrix-free sweep", arch_budget(device_arch))
    name = (
        f"sweep_mf_{'colored' if colored else 'gs'}_{max_constraints}_{mf_max_constraints}_{max_world_dofs}"
        f"_s{max_row_dofs}"
        f"_drive{int(has_drive_rows)}_densevlim{int(has_dense_velocity_limit_rows)}"
        + velocity_limit_law_suffix(velocity_limit_law)
        + ("_fvl" if fuse_vel_limits else "")
        + ("" if shared_metadata else "_gmeta")
        + ("_smeta" if resident_row_metadata else "")
        + ("_glambda" if stream_mf_impulses else "")
        + (f"_local_fallback{local_internal_max_constraints}" if skip_local_internal_worlds else "")
        + ("_torsion" if contact_torsion else "")
        + ("_factor" if factor_coordinates else "")
        + (f"_skiptier{int(skip_tier_rows)}" if skip_tier_rows > 0 else "")
        + ("_components" if independent_components else "")
        + ("_cw" if colored_worlds and not colored else "")
        + solve_policy.suffix
    )
    return KernelTemplate.from_abi(
        name,
        COLORED_ABI if colored else COLORED_WORLDS_ABI if colored_worlds else MF_GS_ABI,
        snippet,
        kernel_template.WORLD_QUEUE,
        extra=solve_policy.arguments,
        composed=True,
        native_name="sweep_mf_colored_native" if colored else "sweep_mf_gs_native",
        launch_bounds=(COLORED_THREADS, 1) if colored else None,
    ).kernel()
