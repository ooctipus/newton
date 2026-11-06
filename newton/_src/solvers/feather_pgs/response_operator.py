# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Author response actions independently of force production and publication.

Vector operands are functions from an index expression to the native expression
of the element at that index. The caller binds storage offsets in them. The rows
are whitened by the walk (:mod:`.walk_rows`), and the velocity change in factor
coordinates leaves by the walk (:func:`.factor.walk.walk_leave_kernel`).

A tree's factor is leaf-first: ``L`` of ``P H P^T`` with ``P`` the reversal of an
articulation's coordinates (factor coordinate ``i`` is physical coordinate
``n - 1 - i``). A vector in factor coordinates that shares a physical vector's
storage sits in physical positions, its factor coordinate ``i`` at position
``n - 1 - i`` (:func:`leaf_first`).

These generators allocate no storage and impose no launch boundary. Owners
supply scratch, lane ownership, synchronization between dependent actions,
held-factor validity and output transactions. Operands are authored bindings,
never extracted from another owner's source. No runtime method is selected here.
"""

from collections.abc import Callable
from dataclasses import dataclass

from ...core.memory import Buffer
from .factor.walk import WALK_COORDINATE, WALK_LINK
from .solve_policy import terms_bound

Vector = Callable[[str], str]
"""A vector operand: the native expression of its element at an index expression."""

Matrix = Callable[[str, str], str]
"""A matrix operand: the native expression of its element at a (row, column) pair of index expressions."""


def _at(vector: Vector, index: str) -> str:
    return vector(f"({index})")


def leaf_first(size: int | str, index: str) -> str:
    """The position of factor coordinate ``index`` in an ``size``-coordinate vector: ``size - 1 - index`` (the
    leaf-first factor's coordinate reversal)."""
    return f"({size}-1-({index}))"


def _operand(expression: str) -> str:
    """``expression`` as an operand: parenthesized unless it is a name or a literal."""
    return expression if expression.replace("_", "x").replace(".", "x").isalnum() else f"({expression})"


def indent_lines(text: str, pad: str) -> str:
    """``text`` with every non-empty line indented by ``pad``."""
    return "\n".join(pad + line if line.strip() else line for line in text.split("\n"))


@dataclass(frozen=True)
class Lanes:
    """The threads that update one row together.

    ``width`` lanes of the sweep's lane group (``LANE`` in ``lane``) split a
    row's sums and reduce them with ``SUM``; every lane then holds the result
    and runs the row's scalar law redundantly. ``width == 1`` is one thread
    per row: sums run in term order and nothing is reduced or synchronized.
    With ``segment``, the lanes are an aligned segment of ``width`` lanes of a
    wider group, one row per segment, and sums reduce over the segment
    (``SEGMENT_SUM``).
    """

    width: int = 32
    lane: str = "lane"
    segment: bool = False

    @property
    def parallel(self) -> bool:
        return self.width > 1

    def reduce(self, *values: str) -> str:
        """Statements that reduce ``values`` over the lanes (nothing for one thread)."""
        if not self.parallel:
            return ""
        if self.segment:
            return f"SEGMENT_SUM({self.width}, {', '.join(values)});"
        return f"SUM({', '.join(values)});"

    def sync(self) -> str:
        """A barrier of the row's lanes (nothing for one thread)."""
        return "SYNC();" if self.parallel else ""

    def first(self, statement: str) -> str:
        """``statement`` run by the row's first lane only."""
        return f"if ({self.lane} == 0) {statement}" if self.parallel else statement

    def broadcast(self, value: str) -> str:
        """``value`` as the first lane reads it, on every lane."""
        return f"READ({self.lane} == 0 ? {value} : 0.0f, 0)" if self.parallel else value


@dataclass(frozen=True)
class Velocity:
    """Where a sweep's rows read and update the velocity they act on: ``array[index]``."""

    array: str = "s_v"

    def at(self, index: str) -> str:
        return f"{self.array}[{index}]"


@dataclass(frozen=True)
class LaneVelocity(Velocity):
    """The velocity in a register a lane: lane ``l`` holds coordinate ``l`` (dense rows, whose entry ``l`` is at
    coordinate ``l``), so a lane reads and updates its own entry's coordinate in its register ``array``."""

    def at(self, index: str) -> str:
        return self.array


class ResponseOperator:
    """A row response representation, which the row law (:mod:`.sweep`) acts through by two actions: ``dot``, sums
    ``Z_row . u`` and Gram entries ``Z_row . Z_other`` reduced in one ladder, and ``axpy``, ``u += sum_k Z_k d_k``.

    The actions are native statements at the indentation ``indent``. A row's diagonal ``A_ii`` is its row store's,
    which the row law reads (:class:`.sweep.Diagonal`), not the operator's.
    """

    applies = True
    """Whether ``axpy`` writes state the row's lanes share (the velocity, a maintained residual), so the lanes
    synchronize after it. An impulse-space operator without it leaves every lane its own copy of the impulses."""
    fused_pair_apply = False
    """Whether a row's apply takes its friction pair's second tangent too (the sibling's change, then the row's, per
    entry, written back once) instead of the sibling's apply, then the row's."""
    pair_with_visit = False
    """Whether the visited row's residual reduces its friction pair's sums in its own ladder (``sibling_jv``,
    ``pair_coupling``: :func:`.sweep.row_sums`), which the pair's terms then read instead of reducing their own."""

    def terms(self, out: str, rhs: str, *, whole: bool) -> str:
        """``S_i`` of a sum this operator reduced into ``out``, the magnitude of the terms its velocity sum adds, or a
        lower bound of it, which the exit law scales a row's floor by (:class:`.solve_policy.Change`).

        By default the reduced velocity term: ``|out|``, or ``|out - rhs|`` for a sum that includes its right-hand
        side ``rhs`` (``whole``). An operator that reduces ``sum_k |t_ik|`` beside the sum returns it.
        """
        return terms_bound(out, rhs) if whole else f"fabsf({out})"


def _applied_sum(out: str, start: str, response: Callable[[str], str], applied: Callable[[str], str], rows: str):
    """Lines of ``float out = start + sum_i response(i) applied(i)`` over rows ``[0, rows)`` in row order, a zero
    applied impulse skipped."""
    return [
        f"float {out} = {start};",
        f"for (int i = 0; i < {rows}; ++i) {{",
        f"    const float a = {applied('i')};",
        f"    if (a != 0.0f) {out} += {response('i')} * a;",
        "}",
    ]


def _lines(lines, indent: str) -> str:
    return "\n".join(indent + line if line.strip() else line for line in "\n".join(lines).split("\n"))


# The last word of a Jacobi sweep's velocity-limit rows (:func:`.sweep.jacobi_limit_pass`) on ``w``, the velocity
# change of the current impulses: ``Y`` (or ``Z``) at ``(row, coordinate)`` through ``response``, the pushed row's ``J``
# at a coordinate through ``row``.


def _push_prologue(response: Callable[[str, str], str], lanes: Lanes, dofs: int, rows: str):
    """Lines of ``w = Y^T x`` (``s_dv``), lane-strided over the coordinates, each a sum over the rows in order."""
    return [
        f"for (int d = {lanes.lane}; d < {dofs}; d += {lanes.width}) {{",
        "    float acc = 0.0f;",
        f"    for (int j = 0; j < {rows}; ++j) acc += {response('j', 'd')} * s_x[j];",
        "    s_dv[d] = acc;",
        "}",
        "SYNC();",
    ]


@dataclass(frozen=True)
class WalkedRows:
    """A world's rows in the walk's coordinates, built in the kernel: each row's seeds walked up its trees' root paths
    (:func:`.factor.walk.walk_prelude`: each tree's path its class's, in constant memory), ``Z_i`` at the row's world
    coordinates.

    A world holds the ``primary`` tree of ``primary`` coordinates from world coordinate ``primary_offset`` and at most
    one ``secondary`` tree from ``secondary_offset``: its articulations ``art_a`` and ``art_b`` (``ink_meta``, ``-1``
    for none). Each lane takes its row of ``rows``: the row input's statements ``read`` write its footprints at
    ``Zi[q * ZS]`` (world coordinate ``q``, a staged row's coordinates ``stride`` apart) and its incident velocity
    ``J_i v_hat`` into ``incident``, which folds into the row's right-hand side. A row input that walks its rows in the
    kernel calls ``TIER_WALK(side, body, k, seed)``: the wrench ``seed[0:6]`` at local ``body`` of the world's tree
    ``side`` (``k < 0``), or the covector ``seed[0] e_k`` at its local coordinate ``k`` (``body`` its body), walked into
    the row and its ``J`` entries against ``v_hat`` into ``incident``; with ``walked`` each slipping tree's release ``R``
    (``walk.release``, :class:`.factor.walk.WalkRelease`) then takes the decided factor's row, as the rows the one
    writer stored hold it already. ``frame`` is each tree's slip capacity, 0 where none of the class's trees of that
    size can slip: its release compiles out. ``classes`` is the number of walking classes the tables hold: past one, a
    tree names its class (:attr:`.factor.walk.WalkTrees.cls`). ``A = Z^T Z``, so ``Z`` serves as both ``J^T`` and ``Y``; the lane keeps
    ``Z_i`` in its registers (``Jr``). The velocity change of the applied impulses, ``u = Z^T delta``, leaves by the
    walk down in the rows' square root, ``P^T L^-T R^T u``, a tree per lane (:meth:`publish`).
    """

    primary: int
    secondary: int
    primary_offset: int
    secondary_offset: int
    dofs: int
    stride: int
    lanes: Lanes
    read: str
    walked: bool = False
    frame: tuple[int, int] = (0, 0)
    classes: int = 1
    rows: str = "n_rows"

    def _tree(self) -> list[str]:
        """The world's trees (every lane): ``art_a``, ``art_b``."""
        return ["const int art_a = ink_meta.data[world * 4];", "const int art_b = ink_meta.data[world * 4 + 2];"]

    def _walk(self) -> str:
        """``TIER_WALK``: one seed of row ``i`` on the world's tree ``side`` (``WALK_ROW``, :func:`.factor.walk.walk_row_prelude`), its
        footprints added at the tree's world coordinates and its ``J`` entries against ``s_v`` into ``incident``."""
        tree = "\n    WALK_TREE(walk.cls.data[art]);" if self.classes > 1 else ""
        return f"""const auto TIER_WALK = [&](const int side, const int body, const int k, const float* seed) {{
    const int art = side == 0 ? art_a : art_b, offset = side == 0 ? {self.primary_offset} : {self.secondary_offset};
    const int j0 = walk.joint_start.data[art], d0 = walk.dof_start.data[art];{tree}
    const float* data = walk.data.data + (size_t)d0 * {WALK_COORDINATE};
    const float* link = walk.link.data + (size_t)j0 * {WALK_LINK};
    // The step's geometry the finish published: its columns and links at the step's poses.
    const float* now = reinterpret_cast<const float*>(walk.now.data + d0);
    const float* now_link = walk.now_link.data + (size_t)j0 * {WALK_LINK};
    const auto sink = [&](int, int c, float z, float j) {{
        Zi[(offset + c) * ZS] += z;
        incident += j * s_v[offset + c];
    }};
    float shift[3];
    WALK_SHIFT(walk, art, shift);
    WALK_ROW(data, link, now, now_link, shift, body, k, seed, sink);
}};"""

    def _sides(self) -> tuple[tuple[int, int, str], ...]:
        """The trees whose slips' release compiles: ``(side, coordinates, articulation)``."""
        trees = ((0, self.primary, self.frame[0], "art_a"), (1, self.secondary, self.frame[1], "art_b"))
        return tuple((side, n, art) for side, n, cap, art in trees if n and cap)

    def _release_rows(self) -> list[str]:
        """The release on the lane's row, each side whose tree slips: ``z <- R z`` (``WALK_RELEASE`` on the row's
        coordinates, ``ZS`` apart), the decided factor's row, inside the row's support."""
        lines = []
        for side, _, art in self._sides():
            offset = self.primary_offset if side == 0 else self.secondary_offset
            lines.append(f"if ({art} >= 0) WALK_RELEASE(walk.release, {art}, &Zi[{offset} * ZS], ZS);")
        return lines

    def stage(self, target: str, *, indent: str) -> str:
        """Walk each lane's row into ``target`` (``Zi = &target[i]``) and its registers, its incident velocity folded
        into its right-hand side."""
        D = self.dofs
        row = [
            f"if ({self.lanes.lane} < {self.rows}) {{",
            *("    " + line for line in self._tree()),
            f"    const int i = {self.lanes.lane};",
            f"    float* Zi = &{target}[i];",
            f"    constexpr int ZS = {self.stride};",
            "    #pragma unroll",
            f"    for (int d = 0; d < {D}; ++d) Zi[d * ZS] = 0.0f;",
            "    // J_i v_hat, folded into the row's right-hand side.",
            "    float incident = 0.0f;",
        ]
        if self.walked:
            row.append(_lines([self._walk()], "    "))
        row += ["    {", self.read, "    }", "    s_rhs[i] += incident;"]
        if self.walked:
            row.append(_lines(self._release_rows(), "    "))
        row.append("}")
        # Each lane's registers take its row only now (a lane past the rows zero), so they are not live during the walk.
        row += [
            "#pragma unroll",
            f"for (int d = 0; d < {D}; ++d) Jr[d] = {self.lanes.lane} < {self.rows} ? {target}[d * {self.stride} + {self.lanes.lane}] : 0.0f;",
        ]
        return _lines(row, indent)

    def publish(self, z: Callable[[str], str], applied: Callable[[str], str], indent: str) -> str:
        """``u = sum_i Z_i applied_i`` (``z(i)``: ``Z_i`` at coordinate ``d``), in row order, in the walk's coordinates
        to the step's one leave (:attr:`.factor.walk.WalkTrees.du`, :func:`.factor.walk.walk_leave_kernel`), which
        walks each tree down: ``v = v_in + P^T L^-T R^T u``."""
        lane, width, D = self.lanes.lane, self.lanes.width, self.dofs
        lines = [
            f"for (int d = {lane}; d < {D}; d += {width}) {{",
            *("    " + line for line in _applied_sum("u", "0.0f", z, applied, self.rows)),
            "    const int global_dof = world_dof_indices.data[dof_map_base + d];",
            "    if (global_dof >= 0) walk.du.data[global_dof] += u;",
            "}",
        ]
        return _lines(lines, indent)

    def operator(self, target: str, lanes: Lanes, velocity: Velocity) -> "SparseFactor":
        """The staged rows as the operator's walked support class: row ``r``'s entry ``d`` at coordinate ``d``, its
        value ``target[d * stride + r]`` (as :meth:`stage` wrote it), on the velocity ``velocity``."""
        return SparseFactor(
            self.dofs,
            lambda row, entry: entry,
            lambda row, entry: f"{target}[{entry} * {self.stride} + {row}]",
            lanes,
            velocity=velocity,
        )


@dataclass(frozen=True)
class TransposedResponse(ResponseOperator):
    """Walked rows ``Z`` in registers and staged transposed: ``A y = Z^T (Z y)``, one lane a row: the simultaneous
    form of the response, every row of a Jacobi level at once (row ``i = lane``).

    A Jacobi sweep (:func:`.sweep.jacobi_sweeps`) keeps each row's residual at zero impulse, ``b'_i = rhs_i + J_i (v -
    Y^T lambda_0)``, in ``s_rhs`` and forms ``r_i(y) = b'_i + (A y)_i`` at the extrapolated impulses ``y`` (``s_y``)
    each sweep. The velocity is published once, from the impulses the sweeps applied; the last word of the
    velocity-limit rows pushes them on ``A x``, maintained as ``w = Y^T x``.

    The rows are walked in the kernel or read from the one writer's store (:class:`WalkedRows`), so ``Z``
    serves as both ``J^T`` and ``Y``. Each lane holds its row in registers
    (``Jr``); the block stages ``Z`` coordinate-major, ``s_Yt[d * stride + i]
    = Z_i[d]`` (``stride = capacity + 4``: past the rows by a float4, so that
    neighbouring coordinates fall in different banks), with zero columns up to
    a multiple of four rows. Each sweep forms ``w = Z y`` once
    (:meth:`gather`): a lane per coordinate and chunk of the rows' float4
    groups (``chunks`` of them, as many as the lanes cover), four accumulators
    joined pairwise, the chunks summed in order; a row's residual is then
    ``b'_i + Z_i w`` in coordinate order. The step scale's bound is the
    column-sum bound ``sum_d |Z_id| sum_j |Z_jd|`` over the rows that can move
    (:meth:`jacobi_steps`), the bound the tiers' sweep budget was fit with
    (:data:`.sweep_routes_tiers.TIER_SWEEPS_PER_ITERATION`).
    """

    dofs: int
    capacity: int
    walked: WalkedRows
    lanes: Lanes = Lanes()
    rows: str = "n_rows"

    def __post_init__(self):
        if self.walked is None:
            raise ValueError("the transposed response holds walked rows")

    @property
    def stride(self) -> int:
        return self.capacity + 4

    @property
    def chunks(self) -> int:
        """Row chunks of :meth:`gather`: lanes per coordinate."""
        return max(self.lanes.width // self.dofs, 1)

    def _yt(self, row: str, coordinate: str) -> str:
        return f"s_Yt[{coordinate} * {self.stride} + {row}]"

    def buffers(self, capacity: int) -> tuple[Buffer, ...]:
        """The staged block, the column sums of the bound, ``w`` and the chunks' partial ``w``."""
        D = self.dofs
        buffers = [Buffer("s_Yt", "float", self.stride * D, align=16), Buffer("s_colsum", "float", D)]
        buffers.append(Buffer("s_dv", "float", (D + 3) & ~3, align=16))
        if self.chunks > 1:
            buffers.append(Buffer("s_dvp", "float", self.chunks * D))
        return tuple(buffers)

    def registers(self, indent: str) -> str:
        """Declare each lane's row registers ``Jr``, which the stage fills (:meth:`WalkedRows.stage`)."""
        return f"{indent}float Jr[{self.dofs}];"

    def jacobi_stage(
        self,
        padding: str,
        scale: Callable[[str, str, str], str],
        indent: str,
        *,
        idle: str,
        counted: Callable[[str], str],
    ) -> str:
        """Walk and stage the rows, mark the rows that cannot move (``idle``) and set each step scale
        (:meth:`jacobi_steps`): ``s_rhs[i]``, which took ``J_i v_hat`` at staging, is the row's residual ``b'_i`` at
        zero impulse; ``padding`` pads the rows' impulses.
        """
        lane, width, D, AM, rows = self.lanes.lane, self.lanes.width, self.dofs, self.capacity, self.rows
        stage = [self.walked.stage("s_Yt", indent=indent)]
        lines = [
            f"for (int e = {lane}; e < {D} * 4; e += {width}) {{",
            "    const int d = e >> 2;",
            f"    const int j = {rows} + (e & 3);",
            f"    if (j < {AM}) {self._yt('j', 'd')} = 0.0f;",
            "}",
            idle,
            "SYNC();",
            *self._jacobi_steps(scale, counted),
            padding,
            "SYNC();",
        ]
        return "\n".join([*stage, _lines(lines, indent)])

    def jacobi_steps(self, scale: Callable[[str, str, str], str], indent: str, *, counted: Callable[[str], str]) -> str:
        """Set each admitted row's step from the column-sum bound over the counted rows; synchronized."""
        return _lines([*self._jacobi_steps(scale, counted), "SYNC();"], indent)

    def _jacobi_steps(self, scale: Callable[[str, str, str], str], counted: Callable[[str], str]) -> list[str]:
        """The column sums over the counted rows, then each admitted row's ``A_ii``, bound and step (lane ``i``'s row
        in its registers)."""
        lane, width, D, rows = self.lanes.lane, self.lanes.width, self.dofs, self.rows
        return [
            f"for (int d = {lane}; d < {D}; d += {width}) {{",
            "    float csum = 0.0f;",
            f"    for (int j = 0; j < {rows}; ++j)",
            f"        if ({counted('j')}) csum += fabsf({self._yt('j', 'd')});",
            "    s_colsum[d] = csum;",
            "}",
            "SYNC();",
            f"for (int i = {lane}; i < {rows}; i += {width}) {{",
            "    const float* Ji = Jr;",
            "    if (s_kind[i] >= 0) {",
            "        float aii = 0.0f, bound = 0.0f;",
            "        #pragma unroll",
            f"        for (int d = 0; d < {D}; ++d) {{",
            f"            aii += Ji[d] * {self._yt('i', 'd')};",
            "            bound += fabsf(Ji[d]) * s_colsum[d];",
            "        }",
            scale("aii", "bound", "        "),
            "    }",
            "}",
        ]

    def gather(self, indent: str) -> str:
        """``w = Y^T y`` (``s_dv``): a lane per coordinate and chunk of float4 row groups, four accumulators joined
        pairwise, the chunks' partial sums (``s_dvp``) added in chunk order; synchronized."""
        lane, width, D, chunks = self.lanes.lane, self.lanes.width, self.dofs, self.chunks
        store = "s_dv[d]" if chunks == 1 else f"s_dvp[ch * {D} + d]"
        lines = [
            "{",
            f"    const int n4 = ({self.rows} + 3) >> 2;",
            f"    const int per = (n4 + {chunks - 1}) / {chunks};",
            f"    if ({lane} < {chunks * D}) {{",
            f"        const int d = {lane} % {D};",
            f"        const int ch = {lane} / {D};",
            "        const float4* y4 = reinterpret_cast<const float4*>(s_y);",
            f"        const float4* Y4 = reinterpret_cast<const float4*>(&s_Yt[d * {self.stride}]);",
            "        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;",
            "        const int j4_end = min(n4, (ch + 1) * per);",
            "        for (int j4 = ch * per; j4 < j4_end; ++j4) {",
            "            const float4 yv = y4[j4];",
            "            const float4 Yv = Y4[j4];",
            "            a0 += Yv.x * yv.x; a1 += Yv.y * yv.y; a2 += Yv.z * yv.z; a3 += Yv.w * yv.w;",
            "        }",
            f"        {store} = (a0 + a1) + (a2 + a3);",
            "    }",
            "    SYNC();",
        ]
        if chunks > 1:
            lines += [
                f"    for (int d = {lane}; d < {D}; d += {width}) {{",
                "        float acc = 0.0f;",
                "        #pragma unroll",
                f"        for (int ch = 0; ch < {chunks}; ++ch) acc += s_dvp[ch * {D} + d];",
                "        s_dv[d] = acc;",
                "    }",
                "    SYNC();",
            ]
        lines.append("}")
        return _lines(lines, indent)

    def jacobi_residual(self, out: str, indent: str) -> str:
        """``float out = b'_i + J_i w``, in coordinate order from ``b'_i``."""
        lines = [
            "const float* Ji = Jr;",
            f"float {out} = s_rhs[i];",
            "#pragma unroll",
            f"for (int d = 0; d < {self.dofs}; ++d) {out} += Ji[d] * s_dv[d];",
        ]
        return _lines(lines, indent)

    def publish(self, indent: str = "    ", *, applied: Callable[[str], str]) -> str:
        """``v = v_hat + P^T L^-T sum_i Z_i applied_i`` in row order from the staged block."""
        return self.walked.publish(lambda row: self._yt(row, "d"), applied, indent)

    def push_prologue(self, indent: str) -> str:
        """``w = Y^T x`` before the pushes of the last word."""
        return _lines(_push_prologue(self._yt, self.lanes, self.dofs, self.rows), indent)

    def staged(self, lanes: Lanes) -> "SparseFactor":
        """The staged rows (``s_Yt``) on ``w`` (``s_dv``), the walked support class: the last word's row actions
        (:func:`.sweep.jacobi_limit_pass`)."""
        return self.walked.operator("s_Yt", lanes, Velocity("s_dv"))


@dataclass(frozen=True)
class SparseFactor(ResponseOperator):
    """Rows of ``Z = L^-1 P J^T`` as packed factor rows, acting on the factor-coordinate velocity delta.

    Row ``r`` packs ``support`` entries: entry ``e`` is ``factor(r, e)`` at the factor coordinate ``index(r, e)``,
    negative for padding, and the row's velocity term is ``Z_r . delta`` over the staged delta (``velocity``). With
    ``dense`` coordinates the packed rows are expanded once per call (:meth:`expand`) into rows of every coordinate
    in private shared storage (``local``): entry ``e`` at coordinate ``e``.

    Two actions: :meth:`dot` reduces sums ``Z_r . u`` and Gram entries ``Z_r . Z_s`` of rows that share a support in
    one ladder, and :meth:`axpy` adds rows' changes ``u += sum_k Z_k d_k``; the row law's actions are these two. With
    lanes, lane ``l`` owns entries ``l + q L`` (``q`` ascending): it keeps their coordinates and values in registers
    (``indices[q]``, ``{name}_z[q]``) for the rows' apply, its partial sums accumulate them from zero, and the lanes
    reduce with ``SUM``. One thread (``Lanes(1, "0")``) sums a row's entries in order. A friction pair's second tangent
    shares its first tangent's support: the first tangent's residual reduces the second's velocity term
    (``sibling_jv``) and the pair's coupling ``Z_t . Z_s`` (``pair_coupling``) in the same ``SUM``, and the apply adds
    the second tangent's change, then the first's, to each entry, written back once.
    """

    support: int
    index: Callable[[str, str], str]
    factor: Callable[[str, str], str]
    lanes: Lanes = Lanes()
    dense: int = 0
    local: str = "s_local_factor"
    velocity: Velocity = Velocity("s_delta")
    registers: bool = False
    """``Z``'s place: each row the visit acts on is in a register the schedule loaded, ``factor(name, entry)`` of its
    name in the visit (``row``, ``sib``, ``spin``, a unit's rows), instead of in memory at its index."""
    magnitudes: bool = False
    """Each velocity sum also reduces its terms' magnitude, which the exit law's vote reads (:meth:`terms`)."""
    fused_pair_apply: bool = True
    """A row's apply takes its friction pair's second tangent too, one :meth:`axpy` of the sibling's change, then the
    row's, written back once; without it the second tangent applies on its own first, then the row."""

    @property
    def width(self) -> int:
        """Entries of a row as the sweeps read it: the packed support, or the dense coordinates."""
        return self.dense or self.support

    @property
    def entries(self) -> int:
        """Entries per lane."""
        return (self.width + self.lanes.width - 1) // self.lanes.width

    def coordinate(self, row: str, entry: str = "entry") -> str:
        """The factor coordinate of row ``row``'s entry ``entry`` (negative for padding)."""
        return entry if self.dense else self.index(_operand(row), entry)

    def element(self, row: str, entry: str = "entry") -> str:
        """Row ``row``'s value at entry ``entry``."""
        if self.dense:
            return f"{self.local}[{_operand(row)} * {self.dense} + {entry}]"
        return self.factor(_operand(row), entry)

    def entries_loop(self, body: str, indent: str, *, entry: bool = True) -> str:
        """``body`` for each entry this lane owns, ``q`` ascending; ``entry`` declares the row entry."""
        first = f"{self.lanes.lane} + q * {self.lanes.width}" if self.lanes.parallel else "q"
        head = ["#pragma unroll", f"for (int q = 0; q < {self.entries}; ++q) {{"]
        if entry:
            head.append(f"    const int entry = {first};")
        lines = [*head, *("    " + line for line in body.split("\n")), "}"]
        return "\n".join(indent + line if line.strip() else line for line in lines)

    def guarded_coordinate(self, row: str, exists: str = "") -> str:
        """The coordinate of this lane's ``entry`` of row ``row``, -1 past the row (with lanes) or where the row's
        condition ``exists`` fails."""
        guards = " && ".join(c for c in (exists, f"entry < {self.width}" if self.lanes.parallel else "") if c)
        return f"{guards} ? {self.coordinate(row)} : -1" if guards else self.coordinate(row)

    # -- the two actions ----------------------------------------------------------------------------------------------

    def _entry(self, name: str, rows: dict[str, str]) -> str:
        """Row ``name``'s value at this lane's entry ``q``: its register (``registers``, by its name), or what a
        :meth:`dot` kept (``z_{name}``)."""
        return self.factor(name, "entry") if self.registers else f"z_{name}[q]"

    def dot(
        self,
        sums: tuple[tuple[str, str, str | None], ...],
        rows: dict[str, str],
        indent: str,
        *,
        exists: dict[str, str] | None = None,
        kept: tuple[str, ...] = (),
        keep_velocity: bool = False,
        magnitudes: tuple[str, ...] = (),
        reduce: bool = True,
        rhs: dict[str, str] | None = None,
        reduced: dict[str, str] | None = None,
    ) -> str:
        """``float out = Z_row . other`` for each ``(out, row, other)`` of ``sums`` (``other`` ``None`` for the velocity,
        else a row: a Gram entry), over the rows' shared support, reduced in one ladder (every lane holds them).

        ``rows`` names each row (its index expression, the first the support's); a row of ``exists`` is zero where its
        condition fails. The support stays in ``indices`` and each row's entries in ``z_{name}`` (with
        ``keep_velocity`` the velocity there in ``velocities``, at the support's entries) for :meth:`axpy`; rows held
        in ``registers`` are read there. The rows of ``kept`` were kept by an earlier ``dot`` on the same support:
        their entries and the support are read again, not loaded. Each velocity sum of an operator with
        ``magnitudes``, and each sum ``magnitudes`` names, also reduces its terms' magnitude ``{out}_terms`` beside
        it (:meth:`terms`). Without ``reduce`` the sums accumulate this lane's entries into ``out``, declared before,
        in order. A sum of ``rhs`` adds its right-hand side after the ladder (``out += rhs``); a sum of ``reduced`` was
        reduced by an earlier ladder under another name (``out`` -> that name), which it is declared from, after the
        ladder, its right-hand side added beside it.
        """
        exists, rhs, reduced = exists or {}, rhs or {}, reduced or {}
        later = []
        for out, _, _ in sums:
            if out in reduced:
                later.append(f"float {out} = {reduced[out]};")
                later += [f"{out} += {rhs[out]};"] if out in rhs else []
        if reduced:
            sums = tuple(s for s in sums if s[0] not in reduced)
            rows = {name: row for name, row in rows.items() if any(name in (r, o) for _, r, o in sums)}
        added = [f"{out} += {rhs[out]};" for out, _, _ in sums if out in rhs]
        tail = "".join(f"\n{indent}{line}" for line in (*added, *later))
        if not sums:
            return tail.removeprefix("\n")
        count, names = self.entries, list(rows)
        kept = () if self.registers else kept
        loaded = [] if self.registers else [name for name in names if name not in kept]
        reads = any(other is None for _, _, other in sums)
        outs = []
        for out, _, other in sums:
            outs.append(out)
            if (other is None and self.magnitudes) or out in magnitudes:
                outs.append(f"{out}_terms")
        keep_support = not (kept or self.registers)
        lines = [f"int indices[{count}];"] if keep_support else []
        lines += [f"float z_{name}[{count}];" for name in loaded]
        if keep_velocity:
            lines.append(f"float velocities[{count}];")
        if reduce:
            lines.append(f"float {', '.join(f'{out} = 0.0f' for out in outs)};")
        # Each row's entry loads beside its coordinate, off the coordinate's chain, and is masked once it is known.
        stored = f"entry < {self.width}" if self.lanes.parallel else ""
        body = []
        for name in loaded:
            condition = " && ".join(c for c in (exists.get(name, ""), stored) if c)
            value = self.element(rows[name])
            body.append(f"const float {name}_value = {f'{condition} ? {value} : 0.0f' if condition else value};")
        if kept:
            body.append("const int index = indices[q];")
        else:
            body.append(f"const int index = {self.guarded_coordinate(rows[names[0]], exists.get(names[0], ''))};")
            body += ["indices[q] = index;"] if keep_support else []
        body += [f"z_{name}[q] = index >= 0 ? {name}_value : 0.0f;" for name in loaded]
        # The sums take this lane's entries where the support has one; padding adds nothing.
        guarded = [f"const float velocity = {self.velocity.at('index')};"] if reads else []
        if keep_velocity:
            guarded.append("velocities[q] = velocity;")
        for out, row, other in sums:
            product = f"{self._entry(row, rows)} * {'velocity' if other is None else self._entry(other, rows)}"
            guarded.append(f"{out} += {product};")
            if f"{out}_terms" in outs:
                guarded.append(f"{out}_terms += fabsf({product});")
        body += ["if (index >= 0) {", *("    " + line for line in guarded), "}"]
        text = "\n".join(indent + line for line in lines) + "\n" + self.entries_loop("\n".join(body), indent)
        ladder = self.lanes.reduce(*outs) if reduce else ""
        return text + "".join(f"\n{indent}{line}" for line in ladder.split("\n") if line) + tail

    def axpy(
        self,
        changes: tuple[tuple[str, str], ...],
        indent: str,
        *,
        row: str | None = None,
        kept: bool = False,
        block: bool = False,
    ) -> str:
        """``delta_v += sum_k Z_k d_k`` over the ``(name, d)`` of ``changes``, from the entries a :meth:`dot` kept, or
        their registers, written back once. One change adds in place. Several read the velocity once (the one a
        ``dot`` kept with ``kept``) and are either one row's sequence of applies, each nonzero change added in the
        order given (a row and its friction sibling: the row-by-row sweep's arithmetic, each add one rounding), or,
        with ``block``, one block's change (a contact unit's rows, solved together): their sum in the order given,
        added once. A block's change rounds once at the velocity's scale, within half an ulp of the exact apply at
        p99, where its changes added in turn round once each (up to 1.2 ulps at p99, and a change below half an ulp
        of the velocity is lost). With ``row`` (one change) the row's entries are read here."""
        if row is not None or self.registers:
            first = row if row is not None else changes[0][0]
            entries = {
                name: self.element(row) if row is not None else self.factor(name, "entry") for name, _ in changes
            }
            head, guard = [f"const int index = {self.guarded_coordinate(first)};"], "index"
        else:
            entries = {name: f"z_{name}[q]" for name, _ in changes}
            head, guard = [], "indices[q]"
        target = self.velocity.at(guard)
        base = "velocities[q]" if kept else target
        if len(changes) == 1:
            ((name, delta),) = changes
            body = [*head, f"if ({guard} >= 0) {target} += {entries[name]} * {delta};"]
        elif block:
            terms = [f"{entries[name]} * {delta}" for name, delta in changes]
            total = f"{terms[0]} + {terms[1]}"
            for term in terms[2:]:
                total = f"({total}) + {term}"
            body = [*head, f"if ({guard} >= 0) {target} = {base} + ({total});"]
        else:
            body = [
                *head,
                f"if ({guard} >= 0) {{",
                f"    float value = {base};",
                *(f"    if ({delta} != 0.0f) value += {entries[name]} * {delta};" for name, delta in changes),
                f"    {target} = value;",
                "}",
            ]
        return self.entries_loop("\n".join(body), indent, entry=bool(head))

    def terms(self, out: str, rhs: str, *, whole: bool) -> str:
        """``sum_k |Z_k u_k|`` of a velocity sum reduced into ``out``, reduced beside it (``magnitudes``)."""
        return f"{out}_terms" if self.magnitudes else super().terms(out, rhs, whole=whole)

    @property
    def pair_with_visit(self) -> bool:
        """Rows in memory: the visited row's residual reduces its pair's sums on the support's one load. Rows in
        registers reduce the pair's own ladder."""
        return not self.registers

    def expand(self, rows: str, indent: str) -> str:
        """Expand the packed rows ``[0, rows)`` into the dense rows (``local``), zero elsewhere; ends unsynchronized."""
        lane, width, dense, support = self.lanes.lane, self.lanes.width, self.dense, self.support
        return (
            f"{indent}for (int i = {lane}; i < {rows} * {dense}; i += {width}) {self.local}[i] = 0.0f;\n"
            f"{indent}{self.lanes.sync()}\n"
            f"{indent}for (int i = {lane}; i < {rows} * {support}; i += {width}) {{\n"
            f"{indent}    const int row = i / {support}, entry = i % {support};\n"
            f"{indent}    const int index = {self.index('row', 'entry')};\n"
            f"{indent}    if (index >= 0) {self.local}[row * {dense} + index] = {self.factor('row', 'entry')};\n"
            f"{indent}}}"
        )
