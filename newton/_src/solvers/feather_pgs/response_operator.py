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


class ResponseOperator:
    """A row response representation: residual, coupling and apply, emitted per row.

    The row actions are native statements at the indentation ``indent``; an
    operator implements the ones its sweeps take. The visited row is ``i``.
    A row's diagonal ``A_ii`` is its row store's, which the row law reads
    (:class:`.sweep.Diagonal`), not the operator's.
    """

    name = "response"
    applies = True
    """Whether :meth:`apply` writes state the row's lanes share (the velocity, a maintained residual), so the lanes
    synchronize after it. An impulse-space operator without apply leaves every lane its own copy of the impulses."""
    fused_pair_apply = False
    """Whether a row's apply takes its friction pair's second tangent too (``apply_pair``: the sibling's change, then
    the row's, per entry, written back once) instead of ``apply_row`` of the sibling then ``apply`` of the row."""

    def terms(self, out: str, rhs: str, *, whole: bool) -> str:
        """``S_i`` of a sum this operator reduced into ``out``, the magnitude of the terms its velocity sum adds, or a
        lower bound of it, which the exit law scales a row's floor by (:class:`.solve_policy.Change`).

        By default the reduced velocity term: ``|out|``, or ``|out - rhs|`` for a sum that includes its right-hand
        side ``rhs`` (``whole``). An operator that reduces ``sum_k |t_ik|`` beside the sum returns it.
        """
        return terms_bound(out, rhs) if whole else f"fabsf({out})"

    # -- row actions ---------------------------------------------------------------------------------------------------

    def _lacks(self, action: str):
        return NotImplementedError(f"{type(self).__name__} has no {action}")

    def residual(self, out: str, indent: str) -> str:
        """``float out = J_i v`` of the visited row (every lane holds it)."""
        raise self._lacks("residual")

    def row_velocity(self, out: str, row: str, indent: str) -> str:
        """``float out = J_row v`` of a row outside the visit (every lane holds it)."""
        raise self._lacks("row velocity")

    def sibling_residual(self, out: str, sibling: str, indent: str, rhs: str) -> str:
        """``float out = J_s v + rhs`` of the friction sibling ``sibling``: its row velocity, then its right-hand
        side ``rhs``."""
        return f"{self.row_velocity(out, sibling, indent)}\n{indent}{out} += {rhs};"

    def coupling(self, out: str, row: str, sibling: str, indent: str) -> str:
        """``float out = J_i Y_s``: the pair's off-diagonal entry."""
        raise self._lacks("pair coupling")

    def pair_entries(self, sibling: str, indent: str) -> str:
        """Declare what a friction pair's terms and its second tangent's apply read of the sibling ``sibling``, once,
        before the pair's disk: nothing by default (:meth:`pair_terms` and :meth:`apply_row` read their own)."""
        return ""

    def pair_terms(self, row: str, sibling: str, rhs: Callable[[str], str], indent: str, *, own: bool = False) -> str:
        """Declare a friction pair's ``sibling_residual`` (``J_s v + rhs_s``) and ``cross`` (``J_i Y_s``).

        With ``own`` the row's residual is reduced here too: ``residual`` (``J_i v + rhs_i``) of the operator's
        current row, for a pair solved as one unit. ``rhs`` maps a row index to its right-hand side. An operator
        whose lanes reduce several sums in one ladder reduces the pair's sums together.
        """
        lines = []
        if own:
            lines.append(self.residual("residual", indent))
            lines.append(f"{indent}residual += {rhs(row)};")
        lines += [
            self.sibling_residual("sibling_residual", sibling, indent, rhs(sibling)),
            self.coupling("cross", row, sibling, indent),
        ]
        return "\n".join(line for line in lines if line != "")

    def spin_terms(self, row: str, sibling: str, spin: str, rhs: Callable[[str], str], indent: str) -> str:
        """Declare a friction triple's spin terms (contact torsion), after :meth:`pair_terms`: ``spin_residual``
        (``J_p v + rhs_p`` of the spin row ``spin``) and its couplings with the tangents, ``spin_cross``
        (``J_i Y_p``) and ``spin_cross_sibling`` (``J_s Y_p``)."""
        lines = (
            self.sibling_residual("spin_residual", spin, indent, rhs(spin)),
            self.coupling("spin_cross", row, spin, indent),
            self.coupling("spin_cross_sibling", sibling, spin, indent),
        )
        return "\n".join(line for line in lines if line != "")

    def apply(self, delta: str, indent: str) -> str:
        """``v += Y_i delta`` of the visited row."""
        raise self._lacks("apply")

    def apply_row(self, row: str, delta: str, indent: str) -> str:
        """``v += Y_row delta`` of a row outside the visit (the friction sibling)."""
        raise self._lacks("row apply")

    def apply_pair(self, sibling_delta: str, delta: str, indent: str) -> str:
        """``v += Y_s sibling_delta + Y_i delta`` in one pass (:attr:`fused_pair_apply`)."""
        raise self._lacks("pair apply")

    def apply_strided(self, row: str, delta: str, indent: str) -> str:
        """``v += Y_row delta`` after :meth:`row_velocity` of the same row (the end-of-sweep passes): by default
        :meth:`apply_row`; an operator may reuse what the row velocity kept."""
        return self.apply_row(row, delta, indent)

    # -- Jacobi actions: every row of a block at once, one lane a row (row ``i = lane``) -------------------------------
    #
    # A Jacobi sweep (:func:`.sweep.jacobi_sweeps`) keeps each row's residual at zero impulse, ``b'_i = rhs_i + J_i
    # (v - Y^T lambda_0)``, in ``s_rhs`` and forms ``r_i(y) = b'_i + (A y)_i`` at the extrapolated impulses ``y``
    # (``s_y``) each sweep. The velocity is published once, from the impulses the sweeps applied; the last word of
    # the velocity-limit rows pushes them on ``A x``, maintained as ``w = Y^T x`` where the operator keeps one.

    def registers(self, indent: str) -> str:
        """Declare the lanes' row registers (nothing by default)."""
        return ""

    def jacobi_stage(
        self,
        padding: str,
        scale: Callable[[str, str, str], str],
        indent: str,
        *,
        idle: str,
        counted: Callable[[str], str],
    ) -> str:
        """Stage the operator, set each row's ``s_rhs`` to ``b'_i``, run ``idle`` (statements that read ``s_rhs``:
        the row store's marks of the rows that cannot move) and set each admitted row's step (:meth:`jacobi_steps`);
        ``padding`` zeroes the rows' padded impulses ahead of the last barrier."""
        raise self._lacks("Jacobi staging")

    def jacobi_steps(self, scale: Callable[[str, str, str], str], indent: str, *, counted: Callable[[str], str]) -> str:
        """Set each admitted row's step, ``scale(aii, bound, indent)``: from ``A_ii`` and a bound of its absolute sum
        ``sum_j |A_ij|`` over the rows ``counted(j)`` names (those that can move); synchronized. Run at staging and
        whenever a row joins the counted rows."""
        raise self._lacks("Jacobi steps")

    def gather(self, indent: str) -> str:
        """Per sweep, before the rows update: the products every row's residual reads (nothing by default)."""
        return ""

    def jacobi_residual(self, out: str, indent: str) -> str:
        """``float out = b'_i + (A y)_i`` of row ``i = lane``."""
        raise self._lacks("Jacobi residual")

    def push_prologue(self, indent: str) -> str:
        """Before the last word's pushes: ``w = Y^T x`` where the operator keeps it (nothing by default)."""
        return ""

    def push_velocity(self, out: str, lanes: Lanes, indent: str) -> str:
        """``float out = (A x)_i`` of the pushed row ``i`` over ``lanes``: the residual's impulse term."""
        raise self._lacks("push velocity")

    def push_apply(self, delta: str, lanes: Lanes, indent: str) -> str:
        """After a push ``delta`` of row ``i``: ``w += Y_i delta`` where the operator keeps ``w`` (nothing by
        default)."""
        return ""

    def push_diagonal(self) -> str:
        """``A_ii`` of the pushed row ``i``."""
        raise self._lacks("push diagonal")


@dataclass(frozen=True)
class FreeBodyZ(ResponseOperator):
    """Rows over at most two free bodies, whitened: ``Z = L^-1 J^T``, six coordinates on each side.

    A free body is a one-node tree; ``L`` is its held factor (``H = L L^T``,
    armature included), and the sweep holds the change of the body's six
    velocity coordinates in factor coordinates, ``u = L^T dv``, until
    :meth:`leave`: a row's velocity term is ``Z^T u``, its update ``u += Z
    delta``. Row ``i``'s twelve entries are one packed row at ``[base + i * 12]``: side ``a``'s six, then side
    ``b``'s; its bodies' first velocity coordinates are ``dof_a``/``dof_b`` (negative for no body).
    With lanes, lane ``l < 12`` owns the row's entry ``l``: lanes 0-5 body ``a``'s coordinates and lanes 6-11 body
    ``b``'s, one product each, and the lanes reduce with ``SUM``; the other
    lanes add zero. With one thread (``Lanes(1)``) the visited row ``i``'s
    entries are read where they are used: the residual adds each body's six
    products as one expression, ``jv += Z_0 u_0 + ... + Z_5 u_5``, body
    ``a``'s then ``b``'s; a friction pair's
    sibling residual starts from its right-hand side and its coupling from
    zero, each adding body ``a``'s then ``b``'s product per coordinate, as does
    the row's own residual where the pair reduces it (``own``). The builder
    stores ``1 / A_ii`` (``A_ii = cfm + |Z|^2``).

    A body's speed row reads its velocity by its walk (``walk_leave``,
    ``walk_enter``).
    """

    lanes: Lanes = Lanes()
    Z: str = "mf_Z"
    base: str = "mf12_base"
    velocity: Velocity = Velocity()
    entry: Callable[[int, str, str], str] | None = None
    """One thread: where entry ``k`` of side ``side``'s six of a row is read, ``entry(side, row, k)``, when a mapping
    stages the rows elsewhere (default: the packed row)."""
    walk_leave: Callable[[str, str], str] | None = None
    """The code mapping the six values ``values`` of the body whose first coordinate is ``first`` back in place,
    ``walk_leave(first, values)``, ``v = P^T L^-T u`` by the walk down (:mod:`.walk_rows`); a speed row needs it."""
    walk_enter: Callable[[str, str, str], str] | None = None
    """With ``walk_leave``: the code giving ``u = L^T P v`` of the six values ``velocity`` in ``out``,
    ``walk_enter(first, velocity, out)``."""
    name = "free_body_z"
    SIDES = ((0, "dof_a"), (1, "dof_b"))
    """Each side and its body's first coordinate, in the row's order."""

    def __post_init__(self):
        if self.entry is not None and self.lanes.parallel:
            raise ValueError("staged entries take the one-thread layout")

    def pack(self, word: str, dof_a: str, dof_b: str, inverse: str, rhs: str, kind: str, parent: str, indent: str):
        """Declare ``int4 word``, a row's packed word as :meth:`consume` decodes it.

        ``x``: the bodies' first world coordinates, ``dof_a`` in the high and ``dof_b`` in the low 16 bits (-1 for
        none); ``y`` and ``z``: the bits of the stored inverse diagonal and of the right-hand side; ``w``: the row
        type in the low and the parent (0xFFFF for none) in the high 16 bits.
        """
        return (
            f"{indent}int4 {word};\n"
            f"{indent}{word}.x = ({dof_a} << 16) | ({dof_b} & 0xFFFF);\n"
            f"{indent}{word}.y = __float_as_int({inverse});\n"
            f"{indent}{word}.z = __float_as_int({rhs});\n"
            f"{indent}{word}.w = static_cast<int>(static_cast<unsigned int>({kind}) | "
            f"(static_cast<unsigned int>({parent}) << 16));"
        )

    # -- software pipeline: the packed row word (bodies, inverse diagonal, rhs, type and parent) and the entries ---

    def pipeline_registers(self, indent: str = "        ") -> str:
        return f"{indent}int4 pre_meta;\n{indent}float pre_z = 0.0f;"

    def _load(self, row_meta: str, row12: str, indent: str) -> str:
        lane = self.lanes.lane
        return (
            f"{indent}pre_meta = *reinterpret_cast<const int4*>(&mf_meta.data[{row_meta}]);\n"
            f"{indent}if ({lane} < 12) pre_z = {self.Z}.data[{row12} + {lane}];"
        )

    def prefetch_first(self, first: str, indent: str) -> str:
        """Load row ``first``'s packed word and entries into the pipeline registers."""
        if first == "0":
            return self._load("off_meta", self.base, indent)
        return self._load(f"off_meta + {first} * 4", f"{self.base} + {first} * 12", indent)

    def consume(self, row: str, end: str, indent: str) -> str:
        """Take row ``row``'s word and entries from the pipeline, load row ``row + 1``'s, and decode the word.

        Declares ``meta``, ``cur_z`` (the lane's entry), ``dof_a``, ``dof_b``, ``mf_diag`` (the stored inverse
        diagonal), ``packed_tp`` and ``mf_rt`` (the row type).
        """
        return (
            f"{indent}int4 meta = pre_meta;\n"
            f"{indent}float cur_z = pre_z;\n"
            f"{indent}if ({row} + 1 < {end}) {{\n"
            f"{indent}    int next_z = {self.base} + ({row} + 1) * 12;\n"
            + self._load(f"off_meta + ({row} + 1) * 4", "next_z", indent + "    ")
            + f"\n{indent}}}\n"
            f"{indent}int packed_dofs = meta.x;\n"
            f"{indent}int dof_a = packed_dofs >> 16;\n"
            f"{indent}int dof_b = (packed_dofs << 16) >> 16;\n"
            f"{indent}float mf_diag = __int_as_float(meta.y);\n"
            f"{indent}int packed_tp = meta.w;\n"
            f"{indent}int mf_rt = packed_tp & 0xFFFF;"
        )

    def _lane_terms(self, out: str, a_value: str, b_value: str, indent: str, assign: str = "=") -> str:
        lane = self.lanes.lane
        return (
            f"{indent}if ({lane} < 6 && dof_a >= 0)\n{indent}    {out} {assign} {a_value};\n"
            f"{indent}if ({lane} >= 6 && {lane} < 12 && dof_b >= 0)\n{indent}    {out} {assign} {b_value};"
        )

    def residual(self, out: str, indent: str) -> str:
        """``float out = Z_i u`` from the pipelined entry ``cur_z`` (one thread: row ``i``'s)."""
        lane = self.lanes.lane
        if self.lanes.parallel:
            return (
                f"{indent}float {out}_partial = 0.0f;\n"
                + self._lane_terms(
                    f"{out}_partial",
                    f"cur_z * {self.velocity.at(f'dof_a + {lane}')}",
                    f"cur_z * {self.velocity.at(f'dof_b + {lane} - 6')}",
                    indent,
                )
                + f"\n{indent}{self.lanes.reduce(f'{out}_partial')}\n{indent}float {out} = {out}_partial;"
            )
        sides = "\n".join(
            f"{indent}if ({body} >= 0) {{\n{indent}    {out} += "
            + " + ".join(f"{self._at(side, 'i', k)} * {self._v(body, k)}" for k in range(6))
            + f";\n{indent}}}"
            for side, body in self.SIDES
        )
        return f"{indent}float {out} = 0.0f;\n{sides}"

    def apply(self, delta: str, indent: str) -> str:
        """``u += Z_i delta`` from the pipelined entry ``cur_z`` (one thread: row ``i``'s)."""
        if not self.lanes.parallel:
            return self.apply_row("i", delta, indent)
        lane = self.lanes.lane
        return (
            f"\n{indent}if ({lane} < 6 && dof_a >= 0) {{\n"
            f"{indent}    {self.velocity.at(f'dof_a + {lane}')} += cur_z * {delta};\n{indent}}}\n"
            f"{indent}if ({lane} >= 6 && {lane} < 12 && dof_b >= 0) {{\n"
            f"{indent}    {self.velocity.at(f'dof_b + {lane} - 6')} += cur_z * {delta};\n{indent}}}"
        )

    def row_velocity(self, out: str, row: str, indent: str) -> str:
        """``float out = Z_row u`` of a row outside the pipeline (lanes 0-11, reduced; one thread: from zero)."""
        if not self.lanes.parallel:
            return self._per_coordinate(out, "0.0f", self._velocity_term(row), indent)
        lane = self.lanes.lane
        entry = f"{self.Z}.data[{self.base} + {row} * 12 + {lane}]"
        return (
            f"{indent}float {out} = 0.0f;\n"
            + self._lane_terms(
                out,
                f"{entry} * {self.velocity.at(f'dof_a + {lane}')}",
                f"{entry} * {self.velocity.at(f'dof_b + {lane} - 6')}",
                indent,
            )
            + f"\n{indent}{self.lanes.reduce(out)}"
        )

    def sibling_residual(self, out: str, sibling: str, indent: str, rhs: str) -> str:
        """``float out = Z_s u + rhs``: the row velocity, then the right-hand side; one thread sums from it."""
        if not self.lanes.parallel:
            return self._per_coordinate(out, rhs, self._velocity_term(sibling), indent)
        return super().sibling_residual(out, sibling, indent, rhs)

    def coupling(self, out: str, row: str, sibling: str, indent: str) -> str:
        """``float out = Z_i Z_s`` (lanes 0-11, reduced; one thread: from zero)."""
        if not self.lanes.parallel:
            return self._per_coordinate(
                out, "0.0f", lambda side, body: f"{self._at(side, row, 'k')} * {self._at(side, sibling, 'k')}", indent
            )
        lane = self.lanes.lane
        product = (
            f"{self.Z}.data[{self.base} + {row} * 12 + {lane}] * {self.Z}.data[{self.base} + {sibling} * 12 + {lane}]"
        )
        return (
            f"{indent}float {out} = 0.0f;\n"
            + self._lane_terms(
                out,
                product,
                product,
                indent,
            )
            + f"\n{indent}{self.lanes.reduce(out)}"
        )

    def pair_terms(self, row: str, sibling: str, rhs: Callable[[str], str], indent: str, *, own: bool = False) -> str:
        """The pair's terms; one thread reduces the row's own residual (``own``) as the sibling's: per coordinate,
        body ``a``'s then ``b``'s product, from its right-hand side."""
        if not own or self.lanes.parallel:
            return super().pair_terms(row, sibling, rhs, indent, own=own)
        return "\n".join(
            (
                self._per_coordinate("residual", rhs(row), self._velocity_term(row), indent),
                self.sibling_residual("sibling_residual", sibling, indent, rhs(sibling)),
                self.coupling("cross", row, sibling, indent),
            )
        )

    def apply_row(self, row: str, delta: str, indent: str) -> str:
        """``u += Z_row delta`` for a row outside the pipeline (lanes 0-11; one thread: each coordinate in order)."""
        if not self.lanes.parallel:
            return "".join(
                f"\n{indent}if ({body} >= 0) {{"
                + "".join(f"\n{indent}    {self._v(body, k)} += {self._at(side, row, k)} * {delta};" for k in range(6))
                + f"\n{indent}}}"
                for side, body in self.SIDES
            )
        lane = self.lanes.lane
        entry = f"{self.Z}.data[{self.base} + {row} * 12 + {lane}]"
        return (
            f"\n{indent}if ({lane} < 6 && dof_a >= 0)\n"
            f"{indent}    {self.velocity.at(f'dof_a + {lane}')} += {entry} * {delta};\n"
            f"{indent}if ({lane} >= 6 && {lane} < 12 && dof_b >= 0)\n"
            f"{indent}    {self.velocity.at(f'dof_b + {lane} - 6')} += {entry} * {delta};"
        )

    # -- one thread per row: entries read where they are used -------------------------------------------------------

    def _at(self, side: int, row: str, k) -> str:
        if self.entry is not None:
            return self.entry(side, row, str(k))
        return f"{self.Z}.data[{self.base} + {row} * 12 + {f'6 + {k}' if side else k}]"

    def _v(self, first: str, k) -> str:
        return self.velocity.at(f"{first} + {k}" if k != 0 else first)

    def _velocity_term(self, row: str):
        return lambda side, body: f"{self._at(side, row, 'k')} * {self._v(body, 'k')}"

    def _per_coordinate(self, out: str, start: str, term, indent: str) -> str:
        """``float out = start``, then per coordinate ``k`` the term ``term(side, body)`` of body ``a``, then of body
        ``b`` (each side when it has a body)."""
        sides = "\n".join(f"{indent}    if ({body} >= 0) {out} += {term(side, body)};" for side, body in self.SIDES)
        return f"{indent}float {out} = {start};\n{indent}for (int k = 0; k < 6; ++k) {{\n{sides}\n{indent}}}"

    # -- free-body speed rows ---------------------------------------------------------------------------------------

    def speed(self, out: str, body: str, axis: str, indent: str) -> str:
        """``const float out``: the magnitude of the body's physical velocity coordinates ``[axis, axis + 3)``.

        The body's coordinates from ``body`` hold ``u``; every lane forms its physical velocity ``speed_v = L^-T u``
        (``walk_leave``) on the predictor, which ``v_out`` holds through the sweep, and the magnitude, the squares added
        in coordinate order. The lanes then synchronize, so :meth:`scale_speed` may write ``u``.
        """
        if self.walk_leave is None:
            raise ValueError("a free body's speed row reads its velocity by its walk (walk_leave)")
        lines = [
            "float speed_v[6];",
            "for (int k = 0; k < 6; ++k) speed_v[k] = " + self._v(body, "k") + ";",
            self.walk_leave(body, "speed_v"),
            f"for (int k = 0; k < 6; ++k) speed_v[k] += v_out.data[world_dof_indices.data[dof_map_base + {body} + k]];",
            f"const float {out} = sqrtf((speed_v[{axis}] * speed_v[{axis}] + speed_v[{axis} + 1] * speed_v[{axis} + 1])"
            f" + speed_v[{axis} + 2] * speed_v[{axis} + 2]);",
        ]
        if self.lanes.parallel:
            lines.append(self.lanes.sync())
        return "\n".join(indent + line for line in "\n".join(lines).split("\n"))

    def scale_speed(self, scale: str, body: str, axis: str, indent: str) -> str:
        """``v += scale v`` on the coordinates :meth:`speed` read, as ``u += L^T dv`` (one thread, or the first lane)."""
        lines = [
            "float speed_dv[6];",
            f"for (int k = 0; k < 6; ++k) speed_dv[k] = k >= {axis} && k < {axis} + 3 ? {scale} * speed_v[k] : 0.0f;",
        ]
        if self.walk_enter is None:
            raise ValueError("a free body's speed row pushes its velocity by its walk (walk_enter)")
        lines += [
            "float speed_du[6];",
            self.walk_enter(body, "speed_dv", "speed_du"),
            "for (int k = 0; k < 6; ++k) " + self._v(body, "k") + " += speed_du[k];",
        ]
        text = "\n".join(lines)
        if self.lanes.parallel:
            text = f"if ({self.lanes.lane} == 0) {{\n{text}\n}}"
        return "\n".join(indent + line for line in text.split("\n"))


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


def _push_velocity(out: str, row: Callable[[str], str], lanes: Lanes, dofs: int):
    """Lines of ``float out = Z_i w`` and ``zz = |Z_i|^2`` over ``lanes``, in one ladder."""
    return [
        f"float {out} = 0.0f, zz = 0.0f;",
        f"for (int d = {lanes.lane}; d < {dofs}; d += {lanes.width}) {{",
        f"    const float row = {row('d')};",
        f"    {out} += row * s_dv[d];",
        "    zz += row * row;",
        "}",
        lanes.reduce(out, "zz"),
    ]


def _push_apply(delta: str, column: Callable[[str], str], lanes: Lanes, dofs: int):
    """Lines of ``w += Y_i delta`` over ``lanes``."""
    return [f"for (int d = {lanes.lane}; d < {dofs}; d += {lanes.width}) s_dv[d] += {column('d')} * {delta};"]


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

    def push_velocity(self, out: str, z: Callable[[str, str], str], lanes: Lanes, indent: str) -> str:
        """``float out = Z_i w`` and ``zz = |Z_i|^2`` of the pushed row ``i`` over ``lanes``, in one ladder."""
        return _lines(_push_velocity(out, lambda d: z("i", d), lanes, self.dofs), indent)

    def push_diagonal(self) -> str:
        """``A_ii = |Z_i|^2 + cfm`` (the rows' law's), from the squares :meth:`push_velocity` reduced."""
        return "zz + cfm"


@dataclass(frozen=True)
class TransposedResponse(ResponseOperator):
    """Walked rows ``Z`` in registers and staged transposed: ``A y = Z^T (Z y)``, one lane a row.

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

    name = "transposed_response"

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

    def push_velocity(self, out: str, lanes: Lanes, indent: str) -> str:
        """``float out = Z_i w`` and ``zz = |Z_i|^2`` over ``lanes``."""
        return self.walked.push_velocity(out, self._yt, lanes, indent)

    def push_apply(self, delta: str, lanes: Lanes, indent: str) -> str:
        """``w += Y_i delta`` after a push of row ``i``."""
        return _lines(_push_apply(delta, lambda d: self._yt("i", d), lanes, self.dofs), indent)

    def push_diagonal(self) -> str:
        """``A_ii`` of a pushed row, formed from its row."""
        return self.walked.push_diagonal()


@dataclass(frozen=True)
class SparsePairZ(ResponseOperator):
    """Whitened rows over a world's dense group of ``dofs`` coordinates plus two scalar sparse coordinates.

    The sparse-diagonal rows: a world pairs one compact dense articulation (its group ``group``, first coordinate
    ``offset``) with diagonal-mass coordinates, and the sweep holds the velocity change in factor coordinates: ``u =
    R^-T L^T P dv`` on the dense articulation (the walk's factor and its slips' release) and ``u = sqrt(m) dv`` on a
    diagonal coordinate (a star's factor is diagonal). Row ``i``'s dense part is its walk's footprints ``Z = F L^-1 P
    J^T`` at ``Z[(group * rows + i) * dofs + l]`` on coordinate ``offset + l`` (the walk's coordinate ``l``); its sparse
    part is two slots ``s``: coordinate ``sparse_dof[(base + i) * 2 + s]`` (``-1`` for none, below ``-1`` an
    independent chain's link) and ``Z_s = J_s / sqrt(m_s)``, a key's one step, at ``sparse_z[(base + i) * 2 + s]``. A row's velocity term is ``Z^T u``, its update
    ``u += Z delta``, and two rows couple through ``Z_i^T Z_s``: ``W = Z^T Z``.

    With lanes (a warp, or a segment of ``width`` lanes), lane ``l < dofs`` owns dense coordinate ``l`` and lanes
    ``dofs`` and ``dofs + 1`` the slots: each lane's one product, zero without a coordinate, is reduced with the
    lanes' sum; the other lanes add zero. A segment of at least ``dofs + 2`` lanes reduces the same products in the
    same order as the warp, up to the sign of a zero sum. The visited row's lane entries are ``coord`` and ``z``
    (:meth:`consume`).

    With one thread (``Lanes(1)``), a row on the single coordinate ``coordinate`` (an independent scalar contact,
    whose dense part is empty), through the slot that holds it: its product is rounded on its own (``__fmul_rn``,
    no contraction into the right-hand side's add), as the lanes' sum of one product rounds it.

    Written with the rows (:func:`.walk_rows.write_slot_rows`, :func:`.compact_contact.produce_contacts`).
    """

    dofs: int
    rows: int
    lanes: Lanes = Lanes()
    Z: str = "dense_Z"
    group: str = "dense_group"
    offset: str = "dense_offset"
    sparse_dof: str = "sparse_row_dof"
    sparse_z: str = "sparse_row_z"
    base: str = "row_base"
    coordinate: str = "scalar_coord"
    velocity: Velocity = Velocity()
    name = "sparse_pair_z"

    def _entries(self, row: str, coord: str, value: str, indent: str, active=None) -> str:
        """Declare this lane's coordinate and entry of row ``row`` (``coord`` negative for none)."""
        lane, P = self.lanes.lane, self.dofs
        guard = f"{active} && " if active else ""
        lines = [
            f"int {coord} = -1;",
            f"float {value} = 0.0f;",
            f"if ({guard}{lane} < {P}) {{",
            f"    {coord} = {self.offset} + {lane};",
            f"    {value} = {self.Z}.data[({self.group} * {self.rows} + {row}) * {P} + {lane}];",
            f"}} else if ({guard}{lane} <= {P + 1}) {{",
            f"    const int sparse_entry = ({self.base} + {row}) * 2 + {lane} - {P};",
            f"    {coord} = {self.sparse_dof}.data[sparse_entry];",
            f"    {value} = {self.sparse_z}.data[sparse_entry];",
            "}",
        ]
        return "\n".join(indent + line for line in lines)

    def _slot(self, row: str, name: str, indent: str) -> str:
        """One thread: declare ``name``, the entry of row ``row``'s slot that holds its coordinate."""
        sparse_row = f"({self.base} + {row})"
        return f"{indent}const int {name} = {sparse_row} * 2 + ({self.sparse_dof}.data[{sparse_row} * 2] >= 0 ? 0 : 1);"

    def _here(self, sibling: str, out: str) -> list[str]:
        """A slot lane's entry of row ``sibling`` on the lane's own coordinate ``coord``: the sibling's slot that
        carries it, zero when none does (``out`` declared zero before)."""
        lane, P = self.lanes.lane, self.dofs
        return [
            f"if ({lane} >= {P} && {lane} <= {P + 1}) {{",
            f"    const int {out}_row = ({self.base} + {sibling}) * 2;",
            f"    {out} = 0.0f;",
            f"    if (coord >= 0 && coord == {self.sparse_dof}.data[{out}_row])",
            f"        {out} = {self.sparse_z}.data[{out}_row];",
            f"    else if (coord >= 0 && coord == {self.sparse_dof}.data[{out}_row + 1])",
            f"        {out} = {self.sparse_z}.data[{out}_row + 1];",
            "}",
        ]

    def consume(self, row: str, indent: str) -> str:
        """Declare the visited row ``row``'s entries: this lane's ``coord`` and ``z``."""
        if not self.lanes.parallel:
            return f"{self._slot(row, 'row_entry', indent)}\n{indent}const float z = {self.sparse_z}.data[row_entry];"
        return self._entries(row, "coord", "z", indent)

    def residual(self, out: str, indent: str) -> str:
        """``float out = Z_i^T u`` from the consumed entries (every lane holds it)."""
        if not self.lanes.parallel:
            return f"{indent}float {out} = __fmul_rn(z, {self.velocity.at(self.coordinate)});"
        return (
            f"{indent}float {out}_partial = coord >= 0 ? z * {self.velocity.at('coord')} : 0.0f;\n"
            f"{indent}{self.lanes.reduce(f'{out}_partial')}\n"
            f"{indent}float {out} = {out}_partial;"
        )

    def row_velocity(self, out: str, row: str, indent: str, *, active: str | None = None) -> str:
        """``float out = Z_row^T u`` of a row outside the visit (``active``: whether this lane's row exists)."""
        if not self.lanes.parallel:
            return (
                f"{self._slot(row, f'{out}_entry', indent)}\n"
                f"{indent}float {out} = __fmul_rn({self.sparse_z}.data[{out}_entry], "
                f"{self.velocity.at(self.coordinate)});"
            )
        return (
            self._entries(row, f"{out}_coord", f"{out}_z", indent, active)
            + f"\n{indent}float {out} = {out}_coord >= 0 ? {out}_z * {self.velocity.at(f'{out}_coord')} : 0.0f;"
            + f"\n{indent}{self.lanes.reduce(out)}"
        )

    def pair_entries(self, sibling: str, indent: str) -> str:
        """Load the sibling's entries once for the pair's terms and its apply: one thread's slot entry (``sib_z``); a
        lane's coordinate and entry of the sibling, and the sibling's entry on the lane's own coordinate
        (``sib_z_here``: a sparse slot's when the sibling carries that coordinate)."""
        if not self.lanes.parallel:
            return (
                f"{self._slot(sibling, 'sib_entry', indent)}\n"
                f"{indent}const float sib_z = {self.sparse_z}.data[sib_entry];"
            )
        lines = [
            self._entries(sibling, "sib_coord", "sib_z", ""),
            "float sib_z_here = sib_z;",
            *self._here(sibling, "sib_z_here"),
        ]
        return "\n".join(indent + line for line in "\n".join(lines).split("\n"))

    def pair_terms(self, row: str, sibling: str, rhs: Callable[[str], str], indent: str, *, own: bool = False) -> str:
        """The pair's terms from the sibling's entries (:meth:`pair_entries`): one thread's two products, a lane's
        two products reduced in one ladder."""
        if own:
            return super().pair_terms(row, sibling, rhs, indent, own=own)
        if not self.lanes.parallel:
            return (
                f"{indent}float sibling_residual = __fmul_rn(sib_z, {self.velocity.at(self.coordinate)});\n"
                f"{indent}sibling_residual += {rhs(sibling)};\n"
                f"{indent}float cross = z * sib_z;"
            )
        return (
            f"{indent}float sibling_residual = sib_coord >= 0 ? sib_z * {self.velocity.at('sib_coord')} : 0.0f;\n"
            f"{indent}float cross = coord >= 0 ? z * sib_z_here : 0.0f;\n"
            f"{indent}{self.lanes.reduce('sibling_residual', 'cross')}\n"
            f"{indent}sibling_residual += {rhs(sibling)};"
        )

    def spin_terms(self, row: str, sibling: str, spin: str, rhs: Callable[[str], str], indent: str) -> str:
        """The spin row's terms (contact torsion): its residual, and its couplings with the row and the sibling, each
        an entry product on the visited row's coordinate, which a contact's rows share (one thread: its slot's)."""
        if not self.lanes.parallel:
            return (
                f"{self._slot(spin, 'spin_entry', indent)}\n"
                f"{indent}const float spin_z = {self.sparse_z}.data[spin_entry];\n"
                f"{indent}float spin_residual = __fmul_rn(spin_z, {self.velocity.at(self.coordinate)});\n"
                f"{indent}spin_residual += {rhs(spin)};\n"
                f"{indent}float spin_cross = z * spin_z;\n"
                f"{indent}float spin_cross_sibling = sib_z * spin_z;"
            )
        lines = [
            self._entries(spin, "spin_coord", "spin_z", ""),
            "float spin_z_here = spin_z;",
            *self._here(spin, "spin_z_here"),
            f"float spin_residual = spin_coord >= 0 ? spin_z * {self.velocity.at('spin_coord')} : 0.0f;",
            "float spin_cross = coord >= 0 ? z * spin_z_here : 0.0f;",
            "float spin_cross_sibling = coord >= 0 ? sib_z_here * spin_z_here : 0.0f;",
            self.lanes.reduce("spin_residual", "spin_cross", "spin_cross_sibling"),
            f"spin_residual += {rhs(spin)};",
        ]
        return "\n".join(indent + line for line in "\n".join(lines).split("\n"))

    def coupling(self, out: str, row: str, sibling: str, indent: str) -> str:
        """``float out = Z_i^T Z_s`` of the visited row ``row``: each lane's entry times the sibling's on its
        coordinate (a sparse slot's, when the sibling carries that coordinate), reduced."""
        if not self.lanes.parallel:
            return (
                f"{self._slot(sibling, f'{out}_entry', indent)}\n"
                f"{indent}float {out} = z * {self.sparse_z}.data[{out}_entry];"
            )
        lane, P = self.lanes.lane, self.dofs
        lines = [
            f"float {out}_here = 0.0f;",
            f"if ({lane} < {P}) {out}_here = {self.Z}.data[({self.group} * {self.rows} + {sibling}) * {P} + {lane}];",
            *self._here(sibling, f"{out}_here"),
            f"float {out} = coord >= 0 ? z * {out}_here : 0.0f;",
            self.lanes.reduce(out),
        ]
        return "\n".join(indent + line for line in lines)

    def apply(self, delta: str, indent: str) -> str:
        """``u += Z_i delta`` from the consumed entries."""
        if not self.lanes.parallel:
            return f"\n{indent}{self.velocity.at(self.coordinate)} += z * {delta};"
        return f"\n{indent}if (coord >= 0) {self.velocity.at('coord')} += z * {delta};"

    def apply_row(self, row: str, delta: str, indent: str) -> str:
        """``u += Z_row delta`` for a row outside the visit; the pair's sibling ``sib`` from its kept entries
        (:meth:`pair_entries`)."""
        if row == "sib":
            if not self.lanes.parallel:
                return f"\n{indent}{self.velocity.at(self.coordinate)} += sib_z * {delta};"
            return f"\n{indent}if (sib_coord >= 0) {self.velocity.at('sib_coord')} += sib_z * {delta};"
        if not self.lanes.parallel:
            return (
                f"\n{self._slot(row, 'apply_entry', indent)}\n"
                f"{indent}{self.velocity.at(self.coordinate)} += {self.sparse_z}.data[apply_entry] * {delta};"
            )
        lines = [
            self._entries(row, "apply_coord", "apply_z", ""),
            f"if (apply_coord >= 0) {self.velocity.at('apply_coord')} += apply_z * {delta};",
        ]
        return "".join(f"\n{indent}{line}" for line in "\n".join(lines).split("\n"))

    def unit_terms(self, head: str, indent: str) -> str:
        """A contact unit's terms (:class:`.sweep.UnitZ`), the normal ``head`` and its tangents ``head + 1``, ``head +
        2``: declare this lane's entries of the unit's rows (``unit_z0``..``unit_z2``) and the velocity on their
        coordinate (``unit_u``), then reduce the rows' velocity terms and the pair's couplings in one ladder.

        A contact's rows act on its two bodies' coordinates, which the build puts in the same slot of each row
        (:func:`.walk_rows.slot_header`), so the lane that owns a coordinate holds all three rows' entries there: the
        unit reads the velocity there once and writes it once (:meth:`unit_apply`)."""
        lane, P, rows = self.lanes.lane, self.dofs, range(3)
        lines = [
            "int unit_coord = -1;",
            f"float {', '.join(f'unit_z{k} = 0.0f' for k in rows)};",
            f"if ({lane} < {P}) {{",
            f"    unit_coord = {self.offset} + {lane};",
            f"    const int unit_dense = ({self.group} * {self.rows} + {head}) * {P} + {lane};",
            *(f"    unit_z{k} = {self.Z}.data[unit_dense + {k * P}];" for k in rows),
            f"}} else if ({lane} <= {P + 1}) {{",
            f"    const int unit_sparse = ({self.base} + {head}) * 2 + {lane} - {P};",
            f"    unit_coord = {self.sparse_dof}.data[unit_sparse];",
            *(f"    unit_z{k} = {self.sparse_z}.data[unit_sparse + {2 * k}];" for k in rows),
            "}",
            f"const float unit_u = unit_coord >= 0 ? {self.velocity.at('unit_coord')} : 0.0f;",
            "float unit_jv0 = unit_z0 * unit_u, unit_jv1 = unit_z1 * unit_u, unit_jv2 = unit_z2 * unit_u;",
            "float gram_n1 = unit_z0 * unit_z1, gram_n2 = unit_z0 * unit_z2, unit_cross = unit_z1 * unit_z2;",
            self.lanes.reduce("unit_jv0", "unit_jv1", "unit_jv2", "gram_n1", "gram_n2", "unit_cross"),
        ]
        return "\n".join(indent + line for line in lines)

    def unit_apply(self, indent: str) -> str:
        """The unit's changes in one pass on this lane's coordinate; the lanes synchronize after it."""
        change = "(unit_z0 * unit_dn + unit_z1 * unit_dt1) + unit_z2 * unit_dt2"
        lines = [
            "if ((unit_dn != 0.0f || unit_dt1 != 0.0f || unit_dt2 != 0.0f) && unit_coord >= 0)",
            f"    {self.velocity.at('unit_coord')} = unit_u + ({change});",
            self.lanes.sync(),
        ]
        return "\n".join(indent + line for line in lines)


@dataclass(frozen=True)
class FactorComponent:
    """One diagonal block of a block-diagonal factor: ``size`` coordinates from lane ``offset``.

    ``lane`` is this lane's coordinate in the block (``LANE - offset``, out of range on other lanes). A lane's coordinate
    is a physical position: factor coordinate ``i`` sits on lane ``size - 1 - i`` (:func:`leaf_first`).
    """

    size: int
    offset: str
    lane: str


@dataclass(frozen=True)
class FactorRows(ResponseOperator):
    """Rows in factor coordinates: ``Z = L^-1 J^T`` over ``dofs`` coordinates (at most a warp), one lane each.

    With ``H = L L^T`` and the velocity change ``u = L^T dv``, a row's velocity
    term is ``J_i dv = Z_i u`` and its response ``u += Z_i delta``: one row array
    (``rows``, row ``i`` from ``base + i * dofs``) is both. Lane ``l`` holds
    coordinate ``l`` of ``u`` in the register ``velocity``, and of the rows it
    acts on in registers: the current row's (``current``), its friction
    sibling's (``sibling``) and, under contact torsion, its contact's spin
    row's (``spin``), which the schedule loads (:meth:`load`, or the pipeline
    :meth:`prefetch` and :meth:`consume`). A sum is one product per
    lane (zero past ``dofs``) reduced with ``SUM``; a pair's sums reduce in one
    ladder (:meth:`pair_terms`). ``components``, the diagonal blocks of ``L``,
    map the velocity change back out of factor coordinates (:meth:`leave`).
    """

    dofs: int
    components: tuple[FactorComponent, ...] = ()
    lanes: Lanes = Lanes()
    rows: str = "factor_rows"
    base: str = "row_base"
    velocity: str = "factor_velocity"
    current: str = "row_z"
    sibling: str = "sibling_z"
    spin: str = "spin_z"
    name = "factor_rows"

    def load(self, row: str) -> str:
        """This lane's coordinate of row ``row`` (zero past ``dofs``)."""
        lane = self.lanes.lane
        return f"{lane} < {self.dofs} ? {self.rows}.data[{self.base} + ({row}) * {self.dofs} + {lane}] : 0.0f"

    def prefetch(self, first: str, indent: str) -> str:
        """Declare the pipeline register ``pre_z`` holding row ``first``."""
        return f"{indent}float pre_z = {self.load(first)};"

    def consume(self, row: str, end: str, indent: str) -> str:
        """Take row ``row`` from the pipeline (``current``) and load row ``row + 1``'s (before ``end``)."""
        lane = self.lanes.lane
        return (
            f"{indent}const float {self.current} = pre_z;\n"
            f"{indent}if ({row} + 1 < {end} && {lane} < {self.dofs})\n"
            f"{indent}    pre_z = {self.rows}.data[{self.base} + ({row} + 1) * {self.dofs} + {lane}];"
        )

    def _product(self, out: str, a: str, b: str) -> str:
        return f"float {out} = {self.lanes.lane} < {self.dofs} ? {a} * {b} : 0.0f;"

    def _sums(self, channels, indent: str, *, magnitudes: tuple[str, ...] = ()) -> str:
        """Reduce each channel ``(out, a, b)``'s products in one ladder; a velocity sum (``b`` the velocity), and each
        channel named in ``magnitudes``, reduces its terms' magnitude beside it, ``{out}_terms`` (:meth:`terms`)."""
        lines, names = [], []
        for out, a, b in channels:
            lines.append(f"{indent}{self._product(out, a, b)}")
            names.append(out)
            if b == self.velocity or out in magnitudes:
                lines.append(f"{indent}float {out}_terms = fabsf({out});")
                names.append(f"{out}_terms")
        return "\n".join([*lines, f"{indent}{self.lanes.reduce(*names)}"])

    def terms(self, out: str, rhs: str, *, whole: bool) -> str:
        """``sum_k |Z_ik u_k|`` of a velocity sum reduced into ``out``, reduced beside it."""
        return f"{out}_terms"

    def residual(self, out: str, indent: str) -> str:
        """``float out = Z_i u`` of the current row, and ``{out}_terms``."""
        return self._sums(((out, self.current, self.velocity),), indent)

    def sibling_residual(self, out: str, sibling: str, indent: str, rhs: str) -> str:
        """``float out = Z_s u + rhs`` of the friction sibling (its entries in the ``sibling`` register)."""
        return f"{self._sums(((out, self.sibling, self.velocity),), indent)}\n{indent}{out} += {rhs};"

    def coupling(self, out: str, row: str, sibling: str, indent: str) -> str:
        """``float out = Z_i Z_s``, the pair's off-diagonal entry."""
        return self._sums(((out, self.current, self.sibling),), indent)

    def pair_terms(self, row: str, sibling: str, rhs: Callable[[str], str], indent: str, *, own: bool = False) -> str:
        """The pair's sums in one ladder: the row's residual (with ``own``), the sibling's, then the coupling."""
        channels = [("residual", self.current, self.velocity)] if own else []
        channels += [("sibling_residual", self.sibling, self.velocity), ("cross", self.current, self.sibling)]
        lines = [self._sums(channels, indent)]
        if own:
            lines.append(f"{indent}residual += {rhs(row)};")
        lines.append(f"{indent}sibling_residual += {rhs(sibling)};")
        return "\n".join(lines)

    def spin_terms(self, row: str, sibling: str, spin: str, rhs: Callable[[str], str], indent: str) -> str:
        """The spin row's sums in one ladder (its entries in the ``spin`` register): its residual, then its couplings
        with the current row and the sibling."""
        channels = (
            ("spin_residual", self.spin, self.velocity),
            ("spin_cross", self.current, self.spin),
            ("spin_cross_sibling", self.sibling, self.spin),
        )
        return f"{self._sums(channels, indent)}\n{indent}spin_residual += {rhs(spin)};"

    def apply(self, delta: str, indent: str) -> str:
        """``u += Z_i delta`` of the current row (lanes past ``dofs`` add zero)."""
        return f"\n{indent}{self.velocity} += {self.current} * {delta};"

    def apply_row(self, row: str, delta: str, indent: str) -> str:
        """``u += Z_s delta`` of the friction sibling ``row`` (its entries in the ``sibling`` register), or of the spin
        row (``spin``, in the ``spin`` register)."""
        register = self.spin if row == "spin" else self.sibling
        return f"\n{indent}{self.velocity} += {register} * {delta};"

    def lanes_of(self, indent: str) -> list[str]:
        """Lines declaring each component's lane (``LANE - offset``)."""
        return [f"{indent}const int {c.lane} = {self.lanes.lane} - {c.offset};" for c in self.components]


@dataclass(frozen=True)
class SparseFactor(ResponseOperator):
    """Rows of ``Z = L^-1 P J^T`` as packed factor rows, acting on the factor-coordinate velocity delta.

    Row ``r`` packs ``support`` entries: entry ``e`` is ``factor(r, e)`` at the factor coordinate ``index(r, e)``,
    negative for padding, and the row's velocity term is ``Z_r . delta`` over the staged delta (``velocity``). With
    ``dense`` coordinates the packed rows are expanded once per call (:meth:`expand`) into rows of every coordinate
    in private shared storage (``local``): entry ``e`` at coordinate ``e``.

    With lanes, lane ``l`` owns entries ``l + q L`` (``q`` ascending): it keeps their coordinates and values in
    registers (``indices[q]``, ``responses[q]``) for the row's apply, its partial sums accumulate them from zero, and
    the lanes reduce with ``SUM``. One thread (``Lanes(1, "0")``) sums a row's entries in order. A friction pair's
    second tangent shares its first tangent's support: the first tangent's residual reduces the second's velocity
    term (``sibling_jv``) and the pair's coupling ``Z_t . Z_s`` (``pair_coupling``) in the same ``SUM``, and the
    apply adds the second tangent's change, then the first's, to each entry, written back once.
    """

    support: int
    index: Callable[[str, str], str]
    factor: Callable[[str, str], str]
    lanes: Lanes = Lanes()
    dense: int = 0
    local: str = "s_local_factor"
    velocity: Velocity = Velocity("s_delta")
    name = "sparse_factor"
    fused_pair_apply = True

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

    def guarded_coordinate(self, row: str) -> str:
        """The coordinate of this lane's ``entry`` of row ``row``, -1 past the row (with lanes)."""
        if not self.lanes.parallel:
            return self.coordinate(row)
        return f"entry < {self.width} ? {self.coordinate(row)} : -1"

    # -- row actions ---------------------------------------------------------------------------------------------------

    def residual(
        self,
        out: str,
        indent: str,
        *,
        row: str = "i",
        sibling: str | None = None,
        paired: str = "false",
    ) -> str:
        """``float out = Z_row . delta`` (every lane holds it), the row's entries kept for :meth:`apply`.

        With ``sibling``, a ``paired`` row (a first tangent) also reduces ``sibling_jv = Z_sibling . delta`` and
        ``pair_coupling = Z_row . Z_sibling`` in the same ``SUM`` and keeps the sibling's entries for
        :meth:`apply_pair`; a contact's spin row, where it has one, reduces its own terms (:meth:`spin_terms`).
        """
        count = self.entries
        lines = [f"int indices[{count}];", f"float responses[{count}];"]
        body = [
            f"const int index = {self.guarded_coordinate(row)};",
            f"const float value = index >= 0 ? {self.element(row)} : 0.0f;",
        ]
        if sibling is None:
            lines.append(f"float {out} = 0.0f;")
            body += [
                "indices[q] = index;",
                "responses[q] = value;",
                f"if (index >= 0) {out} += value * {self.velocity.at('index')};",
            ]
            reduce = self.lanes.reduce(out)
        else:
            lines += [f"float sibling_responses[{count}];", f"float {out} = 0.0f;"]
            lines += ["float sibling_jv = 0.0f;", "float pair_coupling = 0.0f;"]
            body += [
                f"const float sibling_value = {paired} && index >= 0",
                f"    ? {self.element(sibling)} : 0.0f;",
                "indices[q] = index;",
                "responses[q] = value;",
                "sibling_responses[q] = sibling_value;",
                "if (index >= 0) {",
                f"    const float velocity = {self.velocity.at('index')};",
                f"    {out} += value * velocity;",
                "    sibling_jv += sibling_value * velocity;",
                "    pair_coupling += value * sibling_value;",
                "}",
            ]
            reduce = self.lanes.reduce(out, "sibling_jv", "pair_coupling")
        text = "\n".join(indent + line for line in lines) + "\n" + self.entries_loop("\n".join(body), indent)
        return text + "".join(f"\n{indent}{line}" for line in reduce.split("\n") if line)

    def sibling_residual(self, out: str, sibling: str, indent: str, rhs: str) -> str:
        """``float out = Z_s . delta + rhs`` of the pair's second tangent, whose sum :meth:`residual` reduced."""
        return f"{indent}float {out} = sibling_jv;\n{indent}{out} += {rhs};"

    def coupling(self, out: str, row: str, sibling: str, indent: str) -> str:
        """``float out = Z_t . Z_s``: the pair's off-diagonal entry, which :meth:`residual` reduced."""
        return f"{indent}float {out} = pair_coupling;"

    def spin_terms(self, row: str, sibling: str, spin: str, rhs: Callable[[str], str], indent: str) -> str:
        """The spin row's residual and its couplings with the pair, reduced in one ladder over the entries
        :meth:`residual` kept (a contact's rows share its support): only a contact with a spin row pays them."""
        sums = ("spin_residual", "spin_cross", "spin_cross_sibling")
        terms = (
            "    spin_cross += responses[q] * spin_value;",
            "    spin_cross_sibling += sibling_responses[q] * spin_value;",
        )
        declare = f"{indent}float spin_residual = 0.0f, spin_cross = 0.0f, spin_cross_sibling = 0.0f;\n"
        body = "\n".join(
            (
                "if (indices[q] >= 0) {",
                f"    const float spin_value = {self.element(spin)};",
                f"    spin_residual += spin_value * {self.velocity.at('indices[q]')};",
                *terms,
                "}",
            )
        )
        reduce = self.lanes.reduce(*sums)
        return (
            declare
            + self.entries_loop(body, indent)
            + "".join(f"\n{indent}{line}" for line in reduce.split("\n") if line)
            + f"\n{indent}spin_residual += {rhs(spin)};"
        )

    def unit_terms(self, rows: tuple[tuple[str, str], ...], indent: str) -> str:
        """``float {name}_jv = Z_row . delta`` of each ``(row, name)`` of a contact's rows, which share the first row's
        support, reduced in one ladder; the shared coordinates (``indices``), the velocity they read (``velocities``)
        and each row's entries (``{name}_responses``) are kept for :meth:`unit_apply`."""
        count = self.entries
        lines = [
            f"int indices[{count}];",
            f"float velocities[{count}];",
            *(f"float {name}_responses[{count}];" for _, name in rows),
            "float " + ", ".join(f"{name}_jv = 0.0f" for _, name in rows) + ";",
        ]
        body = [
            f"const int index = {self.guarded_coordinate(rows[0][0])};",
            "indices[q] = index;",
            f"velocities[q] = index >= 0 ? {self.velocity.at('index')} : 0.0f;",
            *(f"{name}_responses[q] = index >= 0 ? {self.element(row)} : 0.0f;" for row, name in rows),
            *(f"{name}_jv += {name}_responses[q] * velocities[q];" for _, name in rows),
        ]
        reduce = self.lanes.reduce(*(f"{name}_jv" for _, name in rows))
        text = "\n".join(indent + line for line in lines) + "\n" + self.entries_loop("\n".join(body), indent)
        return text + "".join(f"\n{indent}{line}" for line in reduce.split("\n") if line)

    def unit_apply(self, changes: tuple[tuple[str, str], ...], indent: str) -> str:
        """``delta_v += sum Z_row d`` over the ``(name, d)`` of :meth:`unit_terms`' rows, per entry in the order given,
        each when nonzero, from the velocity :meth:`unit_terms` read, written back once."""
        body = [
            "if (indices[q] >= 0) {",
            "    float value = velocities[q];",
            *(f"    if ({delta} != 0.0f) value += {name}_responses[q] * {delta};" for name, delta in changes),
            f"    {self.velocity.at('indices[q]')} = value;",
            "}",
        ]
        return self.entries_loop("\n".join(body), indent, entry=False)

    def apply(self, delta: str, indent: str) -> str:
        """``delta_v += Z_row delta`` from the entries :meth:`residual` (or :meth:`row_velocity`) kept."""
        target = self.velocity.at("indices[q]")
        return self.entries_loop(f"if (indices[q] >= 0) {target} += responses[q] * {delta};", indent, entry=False)

    def apply_pair(self, sibling_delta: str, delta: str, indent: str) -> str:
        """``delta_v += Z_s sibling_delta + Z_row delta`` from :meth:`residual`'s entries: per entry the sibling's
        change, then the row's, each when nonzero, written back once."""
        target = self.velocity.at("indices[q]")
        body = (
            "if (indices[q] >= 0) {\n"
            f"    float value = {target};\n"
            f"    if ({sibling_delta} != 0.0f) value += sibling_responses[q] * {sibling_delta};\n"
            f"    if ({delta} != 0.0f) value += responses[q] * {delta};\n"
            f"    {target} = value;\n"
            "}"
        )
        return self.entries_loop(body, indent, entry=False)

    def accumulate(self, out: str, row: str, indent: str) -> str:
        """``out += Z_row . delta`` over this lane's entries, in order, unreduced."""
        body = (
            f"const int index = {self.guarded_coordinate(row)};\n"
            f"if (index >= 0) {out} += {self.element(row)} * {self.velocity.at('index')};"
        )
        return self.entries_loop(body, indent)

    def row_velocity(self, out: str, row: str, indent: str) -> str:
        """``float out = Z_row . delta`` (every lane holds it), the row's entries kept for :meth:`apply_strided`."""
        count = self.entries
        body = (
            f"const int index = {self.guarded_coordinate(row)};\n"
            f"const float response = index >= 0 ? {self.element(row)} : 0.0f;\n"
            "indices[q] = index;\n"
            "responses[q] = response;\n"
            f"if (index >= 0) {out} += response * {self.velocity.at('index')};"
        )
        reduce = self.lanes.reduce(out)
        return (
            f"{indent}int indices[{count}];\n{indent}float responses[{count}];\n{indent}float {out} = 0.0f;\n"
            + self.entries_loop(body, indent)
            + (f"\n{indent}{reduce}" if reduce else "")
        )

    def apply_strided(self, row: str, delta: str, indent: str) -> str:
        """``delta_v += Z_row delta`` from the entries :meth:`row_velocity` kept."""
        return self.apply(delta, indent)

    def apply_row(self, row: str, delta: str, indent: str) -> str:
        """``delta_v += Z_row delta``, the row's entries read here."""
        body = (
            f"const int index = {self.guarded_coordinate(row)};\n"
            "if (index >= 0) {\n"
            f"    const float response = {self.element(row)};\n"
            f"    {self.velocity.at('index')} += response * {delta};\n"
            "}"
        )
        return self.entries_loop(body, indent)

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
