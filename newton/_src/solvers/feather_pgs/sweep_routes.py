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
from .projection_laws import PROJECTION_LAW
from .response_operator import Block, DenseJY, DenseSupport, FreeBodyZ, Lanes, Velocity
from .row_metadata import _DENSE_META_ROW_TYPE_MASK, _validate_dense_metadata_encoding
from .solve_policy import EXACT_POLICY, SolvePolicy, block_argmax, sweep_exit
from .staging import Buffer, Staging, arch_budget
from .sweep import (
    CONTACT,
    DRIVE,
    DRIVE_DATA,
    FRICTION,
    JOINT_LIMIT,
    VELOCITY_LIMIT,
    FreeBodyRows,
    RowContext,
    RowLaws,
    WorldRows,
    change_vote,
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
    ("J_world", wp.array3d[float]),
    ("Y_world", wp.array3d[float]),
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


DENSE_SUPPORT = DenseSupport()
"""Where a matrix-free sweep keeps its dense rows' support (:class:`.response_operator.DenseSupport`)."""


class FreeBodyCoordinates:
    """The free bodies a matrix-free sweep holds in factor coordinates, and their conversions (native).

    The bodies of the launch's free-body rows are marked at their first world coordinate in a bitmap
    (``s_mf_bodies``), enter factor coordinates ``u = L^T v`` once the velocity is staged and leave them, ``v =
    L^-T u``, before it is written back (:meth:`.FreeBodyZ.enter`, :meth:`.FreeBodyZ.leave`). The world's dense rows
    read physical coordinates, so a body they touch too (``s_mf_shared``) leaves around them in every sweep. A
    body's factor is ``factors.dof_factor`` of its first global coordinate.
    """

    op = FreeBodyZ()

    @staticmethod
    def words(dofs: int) -> int:
        return (int(dofs) + 31) // 32

    @classmethod
    def buffers(cls, dofs: int) -> tuple[Buffer, ...]:
        words = cls.words(dofs)
        return (
            Buffer("s_mf_bodies", "unsigned", words),
            Buffer("s_mf_shared", "unsigned", words),
            Buffer("s_mf_any_shared", "int", 1),
        )

    def factor(self, first: str) -> str:
        """The native factor index of the free body whose first world coordinate is ``first``."""
        return f"{self.op.factors}.dof_factor.data[world_dof_indices.data[dof_map_base + {first}]]"

    @classmethod
    def clear(cls, dofs: int, lane: str) -> str:
        """Clear the bitmaps (the first warp, before the loads' barrier)."""
        return (
            f"for (int w = {lane}; w < {cls.words(dofs)}; w += 32) {{ s_mf_bodies[w] = 0u; s_mf_shared[w] = 0u; }}\n"
            f"if ({lane} == 0) s_mf_any_shared[0] = 0;"
        )

    @staticmethod
    def mark(first_row: str, rows: str, thread: str, threads: str) -> str:
        """Mark the bodies of the free-body rows ``[first_row, rows)``."""
        return f"""for (int i = {first_row} + {thread}; i < {rows}; i += {threads}) {{
    const int dofs = mf_meta.data[off_meta + i * 4];
    const int da = dofs >> 16, db = (dofs << 16) >> 16;
    if (da >= 0) atomicOr(&s_mf_bodies[da >> 5], 1u << (da & 31));
    if (db >= 0) atomicOr(&s_mf_bodies[db >> 5], 1u << (db & 31));
}}"""

    @classmethod
    def _each(cls, bitmap: str, dofs: int, thread: str, threads: str, body: str) -> str:
        """``body`` on each marked first coordinate ``d`` of ``bitmap``, a thread per word."""
        return f"""for (int w = {thread}; w < {cls.words(dofs)}; w += {threads}) {{
    unsigned bits = {bitmap}[w];
    while (bits != 0u) {{
        const int d = (w << 5) + __ffs(bits) - 1;
        bits &= bits - 1u;
{indent(body, "        ")}
    }}
}}"""

    @classmethod
    def shared(cls, dofs: int, touched: str, thread: str, threads: str, sync: str) -> str:
        """Mark each body some dense row touches as shared (``touched``: native, true when coordinate ``c`` is)."""
        body = f"""bool touched = false;
for (int c = d; c < d + 6 && !touched; ++c) touched = {touched};
if (touched) {{
    s_mf_shared[w] |= 1u << (d & 31);
    s_mf_any_shared[0] = 1;
}}"""
        return f"{cls._each('s_mf_bodies', dofs, thread, threads, body)}\n{sync}"

    def convert_all(self, dofs: int, thread: str, threads: str, sync: str, *, enter: bool) -> str:
        """Every marked body enters (or leaves) factor coordinates, a thread per bitmap word."""
        action = self.op.enter if enter else self.op.leave
        body = f"const int factor = {self.factor('d')};\n{action(lambda k: f's_v[d + {k}]', 'factor')}"
        return f"{self._each('s_mf_bodies', dofs, thread, threads, body)}\n{sync}"

    def convert_shared(self, dofs: int, lane: str, *, enter: bool) -> str:
        """The shared bodies enter (or leave) factor coordinates on the first warp, between warp barriers."""
        action = self.op.enter if enter else self.op.leave
        body = f"const int factor = {self.factor('d')};\n{action(lambda k: f's_v[d + {k}]', 'factor')}"
        return f"""if (mf_shared) {{
    SYNC();
{indent(self._each("s_mf_shared", dofs, lane, "32", body), "    ")}
    SYNC();
}}"""


FREE_BODY_COORDINATES = FreeBodyCoordinates()
"""Where a matrix-free sweep keeps its free bodies' coordinates (:class:`FreeBodyCoordinates`)."""


def dense_support(max_world_dofs: int, lanes: Lanes = Lanes(32)) -> DenseSupport | None:
    """The dense rows' support of a sweep over worlds of ``max_world_dofs`` coordinates, or None.

    The support skips a lane's coordinates that no row touches, so it can only pay when a lane owns more than one
    coordinate. A world of at most ``lanes.width`` coordinates keeps the dense loop, whose software pipeline loads the
    next row's entries while a row is solved (the support's loop reads them at use).
    """
    return DENSE_SUPPORT if int(max_world_dofs) > lanes.width else None


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
class _CachedPairBodyZ(FreeBodyZ):
    """The colored unit's operator: a friction pair's coupling feeds only its metric, which the unit's metric cache
    holds (:func:`_colored_metrics`), so the visit forms none."""

    def coupling(self, out: str, row: str, sibling: str, indent: str) -> str:
        return f"{indent}const float {out} = 0.0f;  // the cached metric replaces the coupling"


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


def _colored_units(free: FreeBodyRows, laws: RowLaws, M_MF: int) -> str:
    """A color's units, one thread each, at block stride: each loads its slot's fields at once
    (:data:`.sweep_units.SLOT_FIELDS`, field by field across the color's slots) and runs the free-body row law,
    unrolled over the unit's rows, on the world's staged velocity. No other unit of the color shares a body."""
    unit_rows = FreeBodyRows(M_MF, stream=free.stream, lanes=Lanes(1))
    arrays = {"mf_Z_a": 0, "mf_Z_b": 1}

    def unit_offset(row: str) -> str:
        offsets = {"i": "r", "sib": "2"}
        if row not in offsets:
            raise ValueError(f"the colored unit reads rows i and sib, not {row!r}")
        return offsets[row]

    unit_op = _CachedPairBodyZ(
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
            "            float metric[5];",
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
                    const int slot_base = unit_slot * {fields * M_MF} + p;
                    const int word_base = unit_slot * {words * M_MF} + p;
                    // Every field of the slot and its metric at once, beside the slot's own load: one round of
                    // memory latency per unit. A unit without a friction pair never reads its pair's fields.
                    float E[{fields}];
                    int W[{words}];
#pragma unroll
                    for (int f = 0; f < {fields}; ++f) E[f] = slot_data.data[slot_base + f * {M_MF}];
#pragma unroll
                    for (int f = 0; f < {words}; ++f) W[f] = slot_words.data[word_base + f * {M_MF}];
#pragma unroll
                    for (int f = 0; f < 5; ++f) metric[f] = slot_metric.data[unit_slot * {5 * M_MF} + p + f * {M_MF}];
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
                    // The pair's second tangent: the first's visit solved it; only the delayed-friction gate holds it.
                    if (rows == 3 && ProjectionLaw::friction_gate(W[2 * 3 + 2] & 0xFFFF, global_iter, friction_start_iteration))
                        s_lam_mf[unit_row + 2] = 0.0f;
                }}
                __syncthreads();
            }}"""


def _colored_metrics(M_MF: int) -> str:
    """Before the sweeps, every thread forms the friction pair metric of the world's staged friction units (a unit of
    three rows) at block stride and stores it in its slot's metric cache (``slot_metric``: scale, the two
    eigenvalues and the larger's axis, field by field across the units), which the pair law reads at every visit."""
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
            const int slot_base = unit_slot * {fields * M_MF} + p;
            const int word_base = unit_slot * {words * M_MF} + p;
            float E[{fields}];
            int W[{words}];
#pragma unroll
            for (int f = 0; f < {fields}; ++f) E[f] = slot_data.data[slot_base + f * {M_MF}];
#pragma unroll
            for (int f = 0; f < 9; ++f) W[f] = slot_words.data[word_base + f * {M_MF}];
{_colored_pair_metric(op, " " * 12)}
            const int at = unit_slot * {5 * M_MF} + p;
            slot_metric.data[at] = metric.v0;
            slot_metric.data[at + {M_MF}] = metric.v1;
            slot_metric.data[at + {2 * M_MF}] = metric.v2;
            slot_metric.data[at + {3 * M_MF}] = metric.v3[0];
            slot_metric.data[at + {4 * M_MF}] = metric.v3[1];
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
    support = dense_support(max_world_dofs)
    return Staging(
        (
            Buffer("s_v", "float", max_world_dofs),
            *FREE_BODY_COORDINATES.buffers(max_world_dofs),
            *dense.buffers(),
            *free.buffers(),
            *(support.buffers(max_world_dofs) if support is not None else ()),
            *colored,
        ),
        threads=COLORED_THREADS if mapping == "colored" else 32,
    )


def mf_gs_source(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    *,
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
    skip_incremental_rows: int = 0,
    independent_components: bool = False,
    velocity_limit_law: str = "stateless",
    solve_policy: SolvePolicy = EXACT_POLICY,
) -> tuple[str, Staging]:
    """The native body of the matrix-free Gauss-Seidel sweep: a world's dense rows (:class:`.DenseJY`) and its
    free-body rows (:class:`.FreeBodyZ`, its bodies in factor coordinates: :class:`FreeBodyCoordinates`), then its
    end-of-sweep passes.

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
    support = dense_support(D, lanes)
    dense_op = DenseJY(D, lanes, support=support)
    bodies = FREE_BODY_COORDINATES
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
    if skip_incremental_rows > 0:
        skips.append(f"    if (m_mf == 0 && m_dense <= {int(skip_incremental_rows)}) return;")
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
        # The torsion pass applies dense rows: their bodies leave factor coordinates around it.
        passes.append(
            indent(bodies.convert_shared(D, "lane", enter=False), "        ")
            + torque_sweep_source(D, solve_policy)
            + indent(bodies.convert_shared(D, "lane", enter=True), "        ")
        )

    if colored:
        # The block's vote: every warp's rows and units (a barrier); the first thread records the solve.
        quiet = "!__syncthreads_or(iteration_changed != 0)"
        sweep_end = sweep_exit(
            solve_policy,
            f"""        if (global_iter >= friction_start_iteration) {{
            if ({quiet}) break;
        }}
""",
            quiet=quiet,
            gate="global_iter >= friction_start_iteration",
            iteration="iter",
            lane="tid",
            argmax="BLOCK_ARGMAX",
        )
        head = "    const int lane = LANE;\n    const int tid = (int)threadIdx.x;\n    const int warp = tid >> 5;"
        first_warp, end_first_warp, barrier = "    if (warp == 0) {\n", "    }\n", "__syncthreads();"
        free_phase = _colored_units(free, laws, M_MF)
        trace_reduction = ""
        if solve_policy.traces:
            trace_reduction = block_argmax(COLORED_THREADS // 32, warp="warp", lane="lane")[0]
        # The world's storage slot (a colored world has one): its colors and staged units.
        offsets = (
            "\n    const int unit_slot = world_slot.data[world];"
            f"\n    const int off_unit = unit_slot * {M_MF};"
            f"\n    const int off_color = unit_slot * {M_MF + 1};"
        )
    else:
        sweep_end = end_sweep(
            solve_policy,
            iteration="iter",
            exact="""        if (global_iter >= friction_start_iteration) {
            unsigned changed = BALLOT(iteration_changed != 0);
            if (changed == 0u) break;
        }
""",
        )
        head = "    const int lane = LANE;"
        first_warp = end_first_warp = ""
        barrier = "SYNC();"
        free_phase = free_body_rows_block(free, free_op, laws, admission=_MF_PHASE_ADMISSION, pad="        ")
        offsets = trace_reduction = ""

    thread, threads = ("tid", "(int)blockDim.x") if colored else ("lane", "32")
    sync = "__syncthreads();" if colored else "SYNC();"
    # The dense rows' support: the coordinates their J or Y touch, from rows [0, m_dense) (every pass reads them),
    # which also flags the free bodies the dense rows share.
    if support is not None:
        shared = support.build(
            dofs=D,
            rows="m_dense",
            J="J_world",
            Y="Y_world",
            base="jy_world_base",
            thread=thread,
            threads=threads,
            first_warp="warp == 0" if colored else "true",
            lane="lane",
            sync=sync,
            marked=bodies.shared(D, f"(({support.bits}[c >> 5] >> (c & 31)) & 1u) != 0u", thread, threads, sync),
        )
    else:
        scan = (
            "[&]() { for (int r = 0; r < m_dense; ++r) "
            f"if (J_world.data[jy_world_base + r * {D} + c] != 0.0f || Y_world.data[jy_world_base + r * {D} + c] "
            "!= 0.0f) return true; return false; }()"
        )
        shared = bodies.shared(D, scan, thread, threads, sync)
    metric_build = _colored_metrics(M_MF) if colored else ""
    # The launch's prologue after the loads: the dense rows' support and shared bodies, the free bodies entering
    # factor coordinates, and the colored units' pair metrics.
    prologue = "\n".join(
        part
        for part in (
            shared,
            bodies.convert_all(D, thread, threads, sync, enter=True),
            "    const bool mf_shared = s_mf_any_shared[0] != 0;",
            metric_build,
        )
        if part
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
    int jy_world_base = world * {M_D} * {D};
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
    // The bodies of the free-body rows this launch visits.
    if (row_phase != 3 && row_phase != 6) {{
{indent(bodies.mark("mf_load_lo", "m_mf", thread, threads), "        ")}
    }}
    {barrier}
{prologue}
{dense_op.pipeline_registers("    ")}

    for (int iter = 0; iter < iterations; iter++) {{
        int global_iter = iteration_offset + iter;
        int iteration_changed = 0;

{first_warp}{indent(bodies.convert_shared(D, "lane", enter=False), "        ")}
{dense_rows_block(dense, dense_op, laws, admission=_phase_admission(), inline_velocity_limit=True, pad="        ")}
{indent(bodies.convert_shared(D, "lane", enter=True), "        ")}
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
    preludes = (lane_group(32), RING, *law_preludes(PROJECTION_LAW), *solve_policy.preludes)
    return cuda_native(body, preludes=preludes), staging


@cache
def mf_gs_kernel(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    device_arch: str,
    *,
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
    skip_incremental_rows: int = 0,
    independent_components: bool = False,
    velocity_limit_law: str = "stateless",
    solve_policy: SolvePolicy = EXACT_POLICY,
) -> wp.Kernel:
    """The matrix-free Gauss-Seidel sweep kernel of a thread mapping (:func:`mf_gs_source`): the warp's launches
    32 threads per world item and takes :data:`MF_GS_ABI` (:data:`COLORED_WORLDS_ABI` with ``colored_worlds``), the
    colored one :data:`COLORED_THREADS` and takes :data:`COLORED_ABI`."""
    snippet, staging = mf_gs_source(
        max_constraints,
        mf_max_constraints,
        max_world_dofs,
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
        skip_incremental_rows=skip_incremental_rows,
        independent_components=independent_components,
        velocity_limit_law=velocity_limit_law,
        solve_policy=solve_policy,
    )
    colored = mapping == "colored"
    staging.check(f"the {mapping} matrix-free sweep", arch_budget(device_arch))
    name = (
        f"sweep_mf_{'colored' if colored else 'gs'}_{max_constraints}_{mf_max_constraints}_{max_world_dofs}"
        f"_drive{int(has_drive_rows)}_densevlim{int(has_dense_velocity_limit_rows)}"
        + velocity_limit_law_suffix(velocity_limit_law)
        + ("_fvl" if fuse_vel_limits else "")
        + ("" if shared_metadata else "_gmeta")
        + ("_smeta" if resident_row_metadata else "")
        + ("_glambda" if stream_mf_impulses else "")
        + (f"_local_fallback{local_internal_max_constraints}" if skip_local_internal_worlds else "")
        + ("_torsion" if contact_torsion else "")
        + ("_factor" if factor_coordinates else "")
        + (f"_skipinc{int(skip_incremental_rows)}" if skip_incremental_rows > 0 else "")
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


def incremental_source(
    max_constraints: int,
    max_world_dofs: int,
    *,
    incremental_rows: int,
    has_drive_rows: bool = True,
    has_dense_velocity_limit_rows: bool = True,
    skip_local_internal_worlds: bool = False,
    min_rows: int = 0,
    velocity_limit_law: str = "stateless",
    solve_policy: SolvePolicy = EXACT_POLICY,
) -> tuple[str, Staging]:
    """The Gauss-Seidel sweep of a world's dense rows (at most ``incremental_rows``) over their response block.

    For worlds without free-body rows: the warp builds ``A = J Y`` and the residuals once (:class:`.Block`),
    compacts the rows this phase visits in row order, and solves them with the shared row law; the velocity is
    published once after the sweeps.
    """
    _validate_dense_metadata_encoding(max_constraints)
    accumulated = check_velocity_limit_law(velocity_limit_law) == "accumulated"
    M_D, D, AM, MIN_ROWS = max_constraints, max_world_dofs, int(incremental_rows), int(min_rows)
    if AM <= 0 or AM > M_D or AM > 64:
        raise ValueError("incremental_rows must be in [1, min(max_constraints, 64)]")
    if MIN_ROWS < 0 or MIN_ROWS >= AM:
        raise ValueError("min_rows must be in [0, incremental_rows)")
    lanes = Lanes(32)
    laws = RowLaws(
        drive=has_drive_rows,
        velocity_limit_law="accumulated" if accumulated else "stateless",
        regularize=True,
        policy=solve_policy,
    )
    op = Block(D, AM, lanes, diagonal=lambda row: f"s_diag[{row}]", inverse=lambda row: f"s_inv_diag[{row}]")
    drive = DRIVE_DATA if has_drive_rows else {}
    staging = Staging(
        (
            Buffer("s_v", "float", D),
            *(
                Buffer(name, "float", AM)
                for name in ("s_lam", "s_app", "s_rhs", "s_diag", "s_inv_diag", "s_res", "s_w")
            ),
            Buffer("s_mu", "float", AM),
            Buffer("s_parent", "int", AM),
            Buffer("s_type", "int", AM),
            Buffer("s_A", "float", AM * AM),
            Buffer("s_rows", "int", AM),
            Buffer("s_vlim_rows", "int", AM),
            *(Buffer(name, "float", AM) for name, _ in drive.values()),
        ),
        threads=32,
    )
    ctx = RowContext(
        row="i",
        kind="row_type",
        operator=op,
        load=lambda row: f"s_lam[{row}]",
        store=lambda row, value: lanes.first(f"s_lam[{row}] = {value};"),
        rhs=lambda row: f"s_rhs[{row}]",
        mu=lambda row: f"s_mu[{row}]",
        weight=lambda row: f"s_w[{row}]",
        parent="parent",
        ring_load=lambda parent: (
            f"RING_LOAD(s_lam[{parent}], {parent}, {AM},\n"
            "    [&](int r) { return s_parent[r]; }, [&](int r) { return s_lam[r]; })"
        ),
        lanes=lanes,
        drive_data=(lambda name, row: f"{DRIVE_DATA[name][0]}[{row}]") if has_drive_rows else None,
        inline_velocity_limit=True,
        record=lambda row, delta: lanes.first(f"s_app[{row}] += {delta};"),
    )
    # The compaction admitted the rows on their diagonals.
    row_loop = "\n".join(
        [
            "        for (int k = 0; k < n_active; k++) {",
            "            const int i = s_rows[k];",
            "            const int row_type = s_type[i];",
            row_visit(ctx, laws, parent="s_parent[i]", check=None, pad="            "),
            "        }",
        ]
    )
    vlim_old = "s_lam[i]" if accumulated else "0.0f"
    vlim_pass = ""
    if has_dense_velocity_limit_rows:
        vlim_pass = f"""
        if (row_phase == 0 || row_phase == 2) {{
            for (int k = 0; k < n_vlim; k++) {{
                const int i = s_vlim_rows[k];
                const float residual_vlim = s_res[i];
                const auto vlim = ProjectionLaw::velocity_limit_update({vlim_old}, residual_vlim, {op.scale("residual_vlim", "i")}, {int(accumulated)});
                const float delta_impulse_vlim = vlim.v1;
                SYNC();
                if (lane == 0) {{
                    s_lam[i] = vlim.v0;
                    s_app[i] += delta_impulse_vlim;
                }}
                if (delta_impulse_vlim != 0.0f) {{
{change_vote(laws, (("delta_impulse_vlim", op.value("i"), str(VELOCITY_LIMIT)),), " " * 20)}
{op.apply_row("i", "delta_impulse_vlim", "                    ")}
                }}
                SYNC();
            }}
        }}"""
    sweep_end = end_sweep(solve_policy, iteration="iter")
    local_skip = (
        f"\n    if (local_solve_owner.data[world] != {PGS_LOCAL_SOLVE_OWNER_GENERAL}) return;"
        if skip_local_internal_worlds
        else ""
    )
    drive_loads = "".join(f"\n        {name}[i] = {source}.data[off_dense + i];" for name, source in drive.values())
    admit_drive = "1" if has_drive_rows else f"(row_type != {DRIVE})"
    phase_filters = "\n".join(
        f"            if (row_phase == {phase} && "
        + " && ".join(f"row_type != {kind}" for kind in kinds)
        + ") admit = 0;"
        for phase, kinds in _PHASE_DENSE_KINDS.items()
    )
    body = f"""    const int lane = LANE;

    int m_dense = world_constraint_count.data[world];
    const int m_mf = mf_constraint_count.data[world];
    if (m_dense == 0 || m_mf != 0) return;
    if (m_dense > {AM} || m_dense <= {MIN_ROWS}) return;{local_skip}

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
    if (dense_hi <= dense_lo) return;
    // The block holds rows [0, dense_hi): a friction row's parent precedes it, rows below dense_lo never change.
    const int n_rows_total = dense_hi;
    const int dof_map_base = world * {D};
    const int off_dense = world * {M_D};
    const int jy_world_base = world * {M_D} * {D};

{staging.declarations("    ")}

    for (int i = lane; i < n_rows_total; i += 32) {{
        s_lam[i] = world_impulses.data[off_dense + i];
        s_app[i] = 0.0f;
        s_rhs[i] = rhs_bias.data[off_dense + i];
        const float diag = world_diag.data[off_dense + i];
        s_diag[i] = diag;
        s_inv_diag[i] = diag > 0.0f ? 1.0f / diag : 0.0f;
        s_type[i] = world_row_type.data[off_dense + i] & {_DENSE_META_ROW_TYPE_MASK};
        s_parent[i] = world_row_parent.data[off_dense + i];
        s_mu[i] = world_row_mu.data[off_dense + i];
        s_w[i] = regularize ? world_row_w.data[off_dense + i] : 1.0f;{drive_loads}
    }}
    for (int d = lane; d < {D}; d += 32) {{
        const int global_dof = world_dof_indices.data[dof_map_base + d];
        s_v[d] = global_dof >= 0 ? v_out.data[global_dof] : 0.0f;
    }}
    SYNC();

    // The rows this phase visits, in row order; the phase's velocity-limit rows for the end pass.
    int n_active = 0;
    int n_vlim = 0;
    for (int base = dense_lo; base < dense_hi; base += 32) {{
        const int row = base + lane;
        int admit = 0;
        int vlim = 0;
        if (row < dense_hi && (s_inv_diag[row] > 0.0f || s_type[row] == {FRICTION})) {{
            const int row_type = s_type[row];
            admit = {admit_drive};
{phase_filters}
            if ((row_phase == 0 || row_phase == 2) && row_type == {VELOCITY_LIMIT}) {{
                admit = 0;
                vlim = 1;
            }}
            if (freeze_drive_rows != 0 && row_type == {DRIVE}) admit = 0;
        }}
        const int slot = ORDERED(n_active, admit != 0);
        if (admit) s_rows[slot] = row;
        const int vlim_slot = ORDERED(n_vlim, vlim != 0);
        if (vlim) s_vlim_rows[vlim_slot] = row;
    }}

{op.build()}
    SYNC();

    for (int iter = 0; iter < iterations; iter++) {{
        const int global_iter = iteration_offset + iter;
        int iteration_changed = 0;
{row_loop}
{vlim_pass}
{sweep_end}    }}

{op.publish()}
    for (int i = dense_lo + lane; i < dense_hi; i += 32) {{
        world_impulses.data[off_dense + i] = s_lam[i];
    }}
#endif
"""
    preludes = (lane_group(32), RING, *law_preludes(PROJECTION_LAW), *solve_policy.preludes)
    return cuda_native(body, preludes=preludes), staging


@cache
def incremental_kernel(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    device_arch: str,
    *,
    incremental_rows: int,
    has_drive_rows: bool = True,
    has_dense_velocity_limit_rows: bool = True,
    skip_local_internal_worlds: bool = False,
    min_rows: int = 0,
    velocity_limit_law: str = "stateless",
    solve_policy: SolvePolicy = EXACT_POLICY,
) -> wp.Kernel:
    """The incremental tier's kernel (the matrix-free sweep's ABI); its working set fits the static budget of
    ``device_arch``."""
    snippet, staging = incremental_source(
        max_constraints,
        max_world_dofs,
        incremental_rows=incremental_rows,
        has_drive_rows=has_drive_rows,
        has_dense_velocity_limit_rows=has_dense_velocity_limit_rows,
        skip_local_internal_worlds=skip_local_internal_worlds,
        min_rows=min_rows,
        velocity_limit_law=velocity_limit_law,
        solve_policy=solve_policy,
    )
    staging.check("the incremental tier", arch_budget(device_arch))
    name = (
        f"sweep_incremental_{max_constraints}_{mf_max_constraints}_{max_world_dofs}_rows{int(incremental_rows)}"
        f"_min{int(min_rows)}_drive{int(has_drive_rows)}_densevlim{int(has_dense_velocity_limit_rows)}"
        + velocity_limit_law_suffix(velocity_limit_law)
        + ("_local_fallback" if skip_local_internal_worlds else "")
        + solve_policy.suffix
    )
    return KernelTemplate.from_abi(
        name,
        MF_GS_ABI,
        snippet,
        kernel_template.WORLD_QUEUE,
        extra=solve_policy.arguments,
        composed=True,
        native_name="sweep_incremental_native",
    ).kernel()
