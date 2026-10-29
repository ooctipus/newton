# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Author response actions independently of force production and publication.

Vector operands are functions from an index expression to the native expression
of the element at that index. The caller binds storage offsets in them. A forward
action maps force into factor coordinates; its transpose maps back to physical
response. Cholesky solves retain their dependency order; inverse factors retain
products.

Every dense factor is the tree's leaf-first factor: ``L`` of ``P H P^T`` with ``P``
the reversal of an articulation's coordinates (factor coordinate ``i`` is physical
coordinate ``n - 1 - i``). Numbering a tree parent-first makes the reversal
leaf-first, so ``L`` has no fill (:meth:`.TreePlan.factor`'s order for a
depth-first numbering). ``L`` is indexed in factor coordinates; a vector the dense
actions read or write sits in physical positions, its factor coordinate ``i`` at
position ``n - 1 - i`` (:func:`leaf_first`), so whitened rows and factor
velocities share their physical vectors' storage and an action may work in place.

These generators allocate no storage and impose no launch boundary. Owners
supply scratch, lane ownership, synchronization between dependent actions,
held-factor validity and output transactions. Operands are authored bindings,
never extracted from another owner's source. No runtime method is selected here.
"""

from collections.abc import Callable
from dataclasses import dataclass

from .solve_policy import Change, bound, terms_bound
from .staging import Buffer

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


@dataclass(frozen=True)
class Reversed:
    """A vector operand that names factor coordinate ``k`` itself (``at(k)``): its element at physical position
    ``size - 1 - k``, typically a pointer to the vector's last element read backwards, so a solve's address steps
    down with ``k`` instead of recomputing ``size - 1 - k`` per access (the runtime-size loops)."""

    at: Vector


def _operand(expression: str) -> str:
    """``expression`` as an operand: parenthesized unless it is a name or a literal."""
    return expression if expression.replace("_", "x").replace(".", "x").isalnum() else f"({expression})"


def _accumulate(coefficient: str, operand: str) -> str:
    return f"value+={coefficient}*{operand};"


def indent_lines(text: str, pad: str) -> str:
    """``text`` with every non-empty line indented by ``pad``."""
    return "\n".join(pad + line if line.strip() else line for line in text.split("\n"))


def _unrolled(head: str, pad: str, unroll: bool) -> str:
    """The loop ``head``, after an unroll directive on a line of its own (indented by ``pad``) when ``unroll``."""
    return f"#pragma unroll\n{pad}{head}" if unroll else head


@dataclass(frozen=True)
class CholeskyFactor:
    """Factor ``A = L L^T`` by columns on one thread: the small dense kit's Cholesky.

    Column ``col`` takes the pivot ``s = A_cc - sum_k L_ck^2`` (``k < col``
    ascending), ``L_cc = sqrt(s)`` and ``r = 1 / L_cc``, then each entry below
    it ``L_rc = (A_rc - sum_k L_rk L_ck) r``, its terms ascending from ``A_rc``.
    Nothing guards the pivot: a positive definite ``A`` has a positive one, and
    a nonpositive pivot publishes NaN, which the factor's consumers detect.
    ``lower`` addresses ``L``; ``source`` addresses ``A``'s lower triangle
    (``None``: ``A`` lies in ``lower`` and is factored in place) and
    ``diagonal`` its regularized diagonal (``None``: ``source``'s). ``unroll``
    unrolls the loops of a compile-time ``size``.
    """

    size: int | str
    lower: Matrix
    source: Matrix | None = None
    diagonal: Vector | None = None
    unroll: bool = False

    def _pivot_source(self) -> Vector:
        A = self.source or self.lower
        return self.diagonal or (lambda col: A(col, col))

    def factor(self) -> str:
        """The factorization's statements, column by column, the pivot's terms first."""
        n, L, unroll = self.size, self.lower, self.unroll
        A = self.source or L
        return "\n".join(
            (
                _unrolled(f"for (int col = 0; col < {n}; ++col) {{", "", unroll),
                f"    float diagonal = {self._pivot_source()('col')};",
                "    " + _unrolled("for (int k = 0; k < col; ++k) {", "    ", unroll),
                f"        const float value = {L('col', 'k')};",
                "        diagonal -= value * value;",
                "    }",
                "    diagonal = sqrtf(diagonal);",
                f"    {L('col', 'col')} = diagonal;",
                "    const float inverse_diagonal = 1.0f / diagonal;",
                "    " + _unrolled(f"for (int row = col + 1; row < {n}; ++row) {{", "    ", unroll),
                f"        float value = {A('row', 'col')};",
                "        " + _unrolled("for (int k = 0; k < col; ++k)", "        ", unroll),
                f"            value -= {L('row', 'k')} * {L('col', 'k')};",
                f"        {L('row', 'col')} = value * inverse_diagonal;",
                "    }",
                "}",
            )
        )

    def pivot(self, col: str) -> str:
        """``L_cc = sqrt(A_cc)``: the whole factor of a diagonal ``A``, at column ``col``."""
        return f"{self.lower(col, col)} = sqrtf({self._pivot_source()(col)});"


@dataclass(frozen=True)
class CholeskySolve:
    """Apply the leaf-first factor's L^-1 and L^-T with the owner's zero-diagonal convention.

    ``L`` is indexed in factor coordinates and every vector operand in physical
    positions (:func:`leaf_first`): :meth:`whiten` maps a physical vector into
    factor coordinates (``x = L^-1 P b``), :meth:`leave` maps back (``P^T L^-T``)
    and :meth:`enter` maps a velocity in (``L^T P v``): the three actions of the
    whitened rows. Both operands of an action keep the same positions, so an
    action may work in place; ``whiten`` then ``leave`` is the response ``H^-1``.
    One lane owns each complete solve. The output may alias the input; an
    optional post-row statement retains per-owner finite-state reporting.
    A zero diagonal produces zero, as in the retained predictor/fallback law.
    With ``reciprocal`` the factor's diagonal holds ``1 / L_ii`` (zero for a
    zero diagonal, as :meth:`stage` writes it) and each step multiplies by it.
    ``ascending`` counts the transposed solve's steps upward (``row = size -
    1 - step``), the loop form NVCC unrolls like the forward solve's when the
    solution is a register array.

    With ``inverse`` the reciprocal diagonal lives in an array of its own
    (``inverse[row] = 1 / L_ii``, staged by :meth:`stage_inverse` without a
    zero convention: the factor of a positive definite matrix has no zero
    diagonal) and the factor keeps its diagonal. ``masked`` keeps the forward
    solve's row loop rolled and runs each row's terms over every column,
    masked to the columns before the row: NVCC then holds one row's chain at
    a time, the form whose registers stay bounded when the solution lives in
    strided shared memory.

    ``lower`` is the factor's row-major storage (a pointer expression read as
    ``lower[row * size + col]``, which staging needs) or a :data:`Matrix`
    operand. ``unroll`` unrolls both solves' loops (a compile-time ``size``),
    the form of a solution held in a register array.

    The zero-diagonal conventions, by option: a zero diagonal produces zero
    (Z0, the default: ``d != 0 ? v / d : 0``); ``reciprocal`` multiplies by a
    staged ``d != 0 ? 1 / d : 0`` (Zr0); ``inverse`` multiplies by an unguarded
    staged ``1 / d`` (Zr-inf).
    """

    size: int | str
    lower: str | Matrix
    reciprocal: bool = False
    ascending: bool = False
    inverse: str | None = None
    masked: bool = False
    unroll: bool = False

    def _entry(self, row: str, col: str) -> str:
        """``L[row][col]``."""
        if callable(self.lower):
            return self.lower(row, col)
        return f"({self.lower})[{row}*{self.size}+{col}]"

    def _at(self, vector: Vector | Reversed, index: str) -> str:
        """Factor coordinate ``index`` of a vector operand: its element at position ``size - 1 - index``."""
        if isinstance(vector, Reversed):
            return vector.at(f"({index})")
        return vector(leaf_first(self.size, index))

    def _finish(self) -> str:
        if self.inverse is not None:
            return f"value*({self.inverse})[row]"
        return "value*diagonal" if self.reciprocal else "diagonal!=0.0f?value/diagonal:0.0f"

    def _diagonal(self) -> str:
        """The step's read of the factor's diagonal, on a line of its own (none with a separate inverse)."""
        if self.inverse is not None:
            return ""
        return f"\n            const float diagonal={self._entry('row', 'row')};"

    def whiten(self, rhs: Vector, output: Vector) -> str:
        """Whiten: solve ``L x = P rhs`` in increasing row order."""
        if self.masked:
            return f"""#pragma unroll 1
        for(int row=0;row<{self.size};++row){{
            float value={self._at(rhs, "row")};
            #pragma unroll
            for(int k=0;k<{self.size};++k)if(k<row)value-={self._entry("row", "k")}*{self._at(output, "k")};{self._diagonal()}
            {self._at(output, "row")}={self._finish()};
        }}"""
        rows = _unrolled(f"for(int row=0;row<{self.size};++row){{", "        ", self.unroll)
        terms = _unrolled("for(int k=0;k<row;++k)", "            ", self.unroll)
        return f"""{rows}
            float value={self._at(rhs, "row")};
            {terms}value-={self._entry("row", "k")}*{self._at(output, "k")};{self._diagonal()}
            {self._at(output, "row")}={self._finish()};
        }}"""

    def leave(self, rhs: Vector, output: Vector, *, post_row: str = "") -> str:
        """Leave factor coordinates: solve ``L^T P x = rhs`` in decreasing row order."""
        head = (
            f"for(int step=0;step<{self.size};++step){{\n            const int row={self.size}-1-step;"
            if self.ascending
            else f"for(int row={self.size}-1;row>=0;--row){{"
        )
        post = f"\n            {post_row}" if post_row or self.inverse is None else ""
        terms = _unrolled(f"for(int k=row+1;k<{self.size};++k)", "            ", self.unroll)
        return f"""{_unrolled(head, "        ", self.unroll)}
            float value={self._at(rhs, "row")};
            {terms}value-={self._entry("k", "row")}*{self._at(output, "k")};{self._diagonal()}
            {self._at(output, "row")}={self._finish()};{post}
        }}"""

    def enter(self, rhs: Vector, output: Vector) -> str:
        """Enter factor coordinates: ``x = L^T P rhs`` in increasing row order, each row's terms from zero, ascending (the output may alias the
        input: row ``r`` reads rows ``r`` on)."""
        rows = _unrolled(f"for(int row=0;row<{self.size};++row){{", "        ", self.unroll)
        terms = _unrolled(f"for(int k=row;k<{self.size};++k)", "            ", self.unroll)
        return f"""{rows}
            float value=0.0f;
            {terms}value+={self._entry("k", "row")}*{self._at(rhs, "k")};
            {self._at(output, "row")}=value;
        }}"""

    def stage_inverse(self, lane: str) -> str:
        """Stage the reciprocal diagonal of :attr:`inverse`, one row per lane (the lanes cover the rows)."""
        n = self.size
        return f"if ({lane} < {n}) ({self.inverse})[{lane}] = 1.0f / ({self.lower})[{lane} * {n} + {lane}];"

    def stage(self, source: Vector, lane: str, width: int | str) -> str:
        """Copy the factor from ``source`` into ``lower``, lane-strided; a reciprocal solve's diagonal becomes ``1 / L_ii``."""
        n = self.size
        count, step = (n * n, n + 1) if isinstance(n, int) else (f"{n} * {n}", f"{n} + 1")
        reciprocal = (
            f"\n    if (i % {step} == 0) factor = factor != 0.0f ? 1.0f / factor : 0.0f;" if self.reciprocal else ""
        )
        return (
            f"for (int i = {lane}; i < {count}; i += {width}) {{\n"
            f"    float factor = {_at(source, 'i')};{reciprocal}\n"
            f"    ({self.lower})[i] = factor;\n"
            "}"
        )


@dataclass(frozen=True)
class LaneCholeskySolve:
    """Apply L^-1 and L^-T with one lane per coordinate and one shuffle per step.

    The same law as :class:`CholeskySolve` (zero diagonal produces zero, the
    leaf-first factor: lane ``i`` owns factor row ``i``, whose right-hand side
    and solution sit at physical position ``size - 1 - i``), with
    the dependency chain shared across the lanes of one warp group: at step
    ``k`` the lane holding row ``k`` finalizes its coordinate, every lane reads
    it through a shuffle and subtracts its own ``L[i, k]`` term. The forward
    pass runs rows upward, the transposed pass downward. ``size`` is a
    compile-time constant of at most the group's lane count; the caller
    supplies the lane variable, the group's shuffle mask and shuffle width
    (by default the names the tree layout's thread prelude declares), and
    scratch index bases for the right-hand side and the output. Lanes past
    ``size`` take part in every shuffle and write nothing.
    """

    size: int
    lower: str

    def solve(self, rhs_base: str, output_base: str, *, lane: str = "lane", mask: str = "MASK", width: str = "WIDTH"):
        n = self.size
        # Each lane's factor row (forward) and column (transposed) are loaded
        # into registers before the loops so the per-step chain is a shuffle
        # and a multiply-add, not a global load. Indices are static after
        # unrolling; an idle lane loads row 0 and writes nothing.
        return f"""{{
            const float* Lr={self.lower};
            const int i={lane};
            const bool active=i<{n};
            const int ic=active?i:0;
            float row[{n}],col[{n}];
            #pragma unroll
            for(int k=0;k<{n};++k){{row[k]=Lr[ic*{n}+k];col[k]=Lr[k*{n}+ic];}}
            const float diagonal=Lr[ic*{n}+ic];
            float value=active?s[{rhs_base}+{n - 1}-i]:0.0f;
            float x=0.0f;
            #pragma unroll
            for(int k=0;k<{n};++k){{
                if(i==k)x=diagonal!=0.0f?value/diagonal:0.0f;
                const float xk=__shfl_sync({mask},x,k,{width});
                if(active&&i>k)value-=row[k]*xk;
            }}
            value=x;
            float z=0.0f;
            #pragma unroll
            for(int k={n}-1;k>=0;--k){{
                if(i==k)z=diagonal!=0.0f?value/diagonal:0.0f;
                const float zk=__shfl_sync({mask},z,k,{width});
                if(active&&i<k)value-=col[k]*zk;
            }}
            if(active)s[{output_base}+{n - 1}-i]=z;
        }}"""


@dataclass(frozen=True)
class IndexedSupport:
    """Visit packed lower entries through a lookup grid and explicit column supports."""

    size: int | str
    index: str
    column_count: str
    column_nodes: str
    column_width: int | str

    def row(self, row: str, operation: str) -> str:
        return f"""for(int col=0;col<=({row});++col){{
            const int entry={self.index}[({row})*{self.size}+col];
            if(entry>=0){{{operation}}}
        }}"""

    def column(self, col: str, operation: str) -> str:
        return f"""for(int k=0;k<{self.column_count}[{col}];++k){{
            const int row={self.column_nodes}[({col})*{self.column_width}+k];
            const int entry={self.index}[row*{self.size}+({col})];
            {operation}
        }}"""


@dataclass(frozen=True)
class CompressedSupport:
    """Visit the owner's existing CSR rows and CSC entry references."""

    row_offsets: str
    columns: str
    column_offsets: str
    column_entries: str
    entry_rows: str

    def row(self, row: str, operation: str) -> str:
        return f"""for(int entry={self.row_offsets}[{row}];entry<{self.row_offsets}[({row})+1];++entry){{
            const int col={self.columns}[entry];
            {operation}
        }}"""

    def column(self, col: str, operation: str) -> str:
        return f"""for(int position={self.column_offsets}[{col}];position<{self.column_offsets}[({col})+1];++position){{
            const int entry={self.column_entries}[position];
            const int row={self.entry_rows}[entry];
            {operation}
        }}"""


@dataclass(frozen=True)
class SparseInverseFactor:
    """Apply the same sparse factor arithmetic through either retained storage map."""

    factor: str
    support: IndexedSupport | CompressedSupport

    def forward(self, row: str, rhs: Vector) -> str:
        """Compute one sparse forward row, preserving the support's traversal order."""
        operation = _accumulate(f"({self.factor})[entry]", _at(rhs, "col"))
        return "float value=0.0f;" + self.support.row(row, operation)

    def transpose(self, col: str, rhs: Vector) -> str:
        """Compute one sparse transpose column from completed forward coordinates."""
        operation = _accumulate(f"({self.factor})[entry]", _at(rhs, "row"))
        return "float value=0.0f;" + self.support.column(col, operation)


# ---------------------------------------------------------------------------------------------------------------------
# Response operators of the sweeps: one implementation per representation of a row's response
# ---------------------------------------------------------------------------------------------------------------------
#
# A Gauss-Seidel or Jacobi row update needs four things of its row's response, whatever holds it:
#
# * the residual's velocity term, ``J_i v`` (or ``C_i lambda`` in impulse space), of the visited row and of a row
#   outside the visit (a friction sibling, an end-of-sweep row);
# * the diagonal ``A_ii`` the step divides by, and the step ``-r / A_ii`` itself;
# * the coupling of a friction pair, ``J_i Y_s`` (the pair's off-diagonal block entry), with the sibling's residual
#   (the pair's terms);
# * the apply, ``v += Y_i delta`` (or nothing, in impulse space), of the row, of its sibling, or of both in one pass.
#
# A :class:`ResponseOperator` emits each as native statements over one row index, for the lanes that update the row
# together (:class:`Lanes`): the lanes split the sum's terms and reduce them with the lane group's ``SUM``, or one
# thread sums them in order. The lane layout and the representation fix the floating-point order, so an operator
# writes exactly one order per layout: the one the sweeps it replaces compute.


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
    """A row response representation: residual, diagonal, coupling and apply, emitted per row.

    A concrete operator reads a row's diagonal through :attr:`diagonal`
    (``A_ii``) and, when its builder stores or stages it, :attr:`inverse`
    (``1 / A_ii``): callables from a row index to the native expression. The
    step multiplies by the inverse when there is one and divides by the
    diagonal otherwise, as the builders of the sweeps it replaces left them.

    The row actions are native statements at the indentation ``indent``; an
    operator implements the ones its sweeps take. The visited row is ``i``.
    """

    name = "response"
    residual_includes_rhs = False
    """Whether :meth:`residual` gives the whole residual ``J v + rhs`` (a maintained residual) or ``J v``."""
    applies = True
    """Whether :meth:`apply` writes state the row's lanes share (the velocity, a maintained residual), so the lanes
    synchronize after it. An impulse-space operator without apply leaves every lane its own copy of the impulses."""
    fused_pair_apply = False
    """Whether a row's apply takes its friction pair's second tangent too (``apply_pair``: the sibling's change, then
    the row's, per entry, written back once) instead of ``apply_row`` of the sibling then ``apply`` of the row."""
    diagonal: Callable[[str], str] | None = None
    inverse: Callable[[str], str] | None = None

    def value(self, row: str) -> str:
        """``A_ii`` of row ``row`` (zero for a zero inverse)."""
        if self.diagonal is not None:
            return self.diagonal(row)
        inverse = self.inverse(row)
        return f"{inverse} > 0.0f ? 1.0f / {inverse} : 0.0f"

    def bound(self, rhs: str, terms: str, impulse: str, row: str) -> str:
        """The exit law's bound of row ``row`` at ``impulse`` (:func:`.solve_policy.bound`), in the form its storage
        gives without a division: impulse units from a stored inverse, velocity units from a stored diagonal."""
        if self.inverse is not None:
            return bound(rhs, terms, impulse, step=self.inverse(row))
        return bound(rhs, terms, impulse, diagonal=self.diagonal(row))

    def change(self, delta: str, bound: str, kind: str, row: str) -> Change:
        """Row ``row``'s change ``delta`` against its :meth:`bound`, as the exit law votes it."""
        return Change(delta, bound, kind, self.value(row), scaled=self.inverse is None)

    def step(self, residual: str, row: str) -> str:
        """The Gauss-Seidel step ``-r / A_ii``."""
        if self.inverse is not None:
            return f"-{residual} * {self.inverse(row)}"
        return f"-{residual} / {self.diagonal(row)}"

    def scale(self, residual: str, row: str) -> str:
        """``r / A_ii`` (the step's magnitude, as the velocity-limit law takes it)."""
        if self.inverse is not None:
            return f"{residual} * {self.inverse(row)}"
        return f"{residual} / {self.diagonal(row)}"

    def velocity_term(self, residual: str, rhs: str, jv: str) -> str:
        """``J v`` of the row: the reduced ``jv``, or the maintained residual less its right-hand side."""
        return f"{residual} - {rhs}" if self.residual_includes_rhs else jv

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
        """``float out = J_i v`` of the visited row (every lane holds it), or its whole maintained residual
        (:attr:`residual_includes_rhs`)."""
        raise self._lacks("residual")

    def row_velocity(self, out: str, row: str, indent: str) -> str:
        """``float out = J_row v`` of a row outside the visit (every lane holds it)."""
        raise self._lacks("row velocity")

    def sibling_residual(self, out: str, sibling: str, indent: str, rhs: str) -> str:
        """``float out = J_s v + rhs`` of the friction sibling ``sibling``: its row velocity, then its right-hand
        side ``rhs``. An operator that maintains residuals reads the sibling's whole."""
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
            lines.append("" if self.residual_includes_rhs else f"{indent}residual += {rhs(row)};")
        lines += [
            self.sibling_residual("sibling_residual", sibling, indent, rhs(sibling)),
            self.coupling("cross", row, sibling, indent),
        ]
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

    def jacobi_stage(self, padding: str, scale: Callable[[str, str, str], str], indent: str) -> str:
        """Stage the operator, set each row's ``s_rhs`` to ``b'_i`` and each admitted row's step (``scale(aii, bound,
        indent)``: from ``A_ii`` and its row's absolute sum ``sum_j |A_ij|`` or a bound of it); ``padding`` zeroes
        the rows' padded impulses ahead of the last barrier."""
        raise self._lacks("Jacobi staging")

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
    """Rows over at most two free bodies, whitened: ``Z_a, Z_b = L^-1 J^T`` of 6 coordinates each.

    A free body is a one-node tree; ``L`` is its held factor (``H = L L^T``,
    armature included), and the sweep holds the body's six velocity
    coordinates in factor coordinates ``u = L^T v`` between :meth:`enter` and
    :meth:`leave`: a row's velocity term is ``Z^T u``, its update ``u += Z
    delta``. Row ``i``'s entries are at ``[base + i * 6 + k]``; its bodies'
    first velocity coordinates are ``dof_a``/``dof_b`` (negative for no body).
    With lanes, lanes 0-5 own body ``a``'s coordinates and lanes 6-11 body
    ``b``'s, one product each, and the lanes reduce with ``SUM``; the other
    lanes add zero. With one thread (``Lanes(1)``) the visited row ``i``'s
    entries are read where they are used, in the order split mode's sweep sums
    them: the residual adds each body's six products as one expression, ``jv
    += Z_0 u_0 + ... + Z_5 u_5``, body ``a``'s then ``b``'s; a friction pair's
    sibling residual starts from its right-hand side and its coupling from
    zero, each adding body ``a``'s then ``b``'s product per coordinate, as does
    the row's own residual where the pair reduces it (``own``). The free-body
    loop's order (``order="loop"``) adds the residual's products one at a time
    instead, body ``a``'s six then ``b``'s. The builder stores ``1 / A_ii``
    (``A_ii = cfm + |Z_a|^2 + |Z_b|^2``).

    ``factors`` names the native :class:`.kernels.BodyFactors` operand and
    ``factor`` maps a body's first velocity coordinate (in the sweep's
    addressing) to the index of its factor in ``factors.L``.
    """

    lanes: Lanes = Lanes()
    Z_a: str = "mf_Z_a"
    Z_b: str = "mf_Z_b"
    base: str = "mf6_base"
    velocity: Velocity = Velocity()
    diagonal: Callable[[str], str] | None = None
    inverse: Callable[[str], str] | None = None
    order: str = "split"
    """One thread's residual order: ``"split"`` (each body's products as one expression) or ``"loop"`` (one product
    at a time)."""
    entry: Callable[[str, str, str], str] | None = None
    """One thread: where entry ``k`` of a row's ``Z_a`` or ``Z_b`` (the array's name) is read, ``entry(array, row,
    k)``, when a mapping stages the rows elsewhere (default: the array, by row)."""
    factors: str = "factors"
    factor: Callable[[str], str] | None = None
    name = "free_body_z"

    ORDERS = ("split", "loop")

    def __post_init__(self):
        if self.order not in self.ORDERS:
            raise ValueError(f"order must be one of {self.ORDERS}, got {self.order!r}")
        if self.entry is not None and self.lanes.parallel:
            raise ValueError("staged entries take the one-thread layout")

    # -- build (response_builds_rows) ---------------------------------------------------------------------------------

    @staticmethod
    def build():
        """The rows' build and publication (:data:`.response_builds_rows.FREE_BODY_Z`): each row's ``Z`` and
        ``1 / A_ii``, the rows' body coordinates and words (:meth:`pack`), and ``v += L^-T sum Z lambda``."""
        from .response_builds_rows import FREE_BODY_Z  # noqa: PLC0415

        return FREE_BODY_Z

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

    # -- the bodies' factor: entering and leaving factor coordinates ------------------------------------------------

    def lower(self, factor: str) -> str:
        """The native pointer of factor ``factor``'s ``L`` (row-major, lower)."""
        return f"({self.factors}.L.data + ({factor}) * 36)"

    def enter(self, velocity: Vector, factor: str) -> str:
        """Map a body's six coordinates ``velocity`` into factor coordinates in place, ``u = L^T v`` (one thread)."""
        return CholeskySolve(6, self.lower(factor), unroll=True).enter(velocity, velocity)

    def leave(self, velocity: Vector, factor: str) -> str:
        """Map a body's six coordinates ``velocity`` back in place, ``v = L^-T u`` (one thread)."""
        return CholeskySolve(6, self.lower(factor), unroll=True).leave(velocity, velocity)

    # -- software pipeline: the packed row word (bodies, inverse diagonal, rhs, type and parent) and the entries ---

    def pipeline_registers(self, indent: str = "        ") -> str:
        return f"{indent}int4 pre_meta;\n{indent}float pre_Za = 0.0f, pre_Zb = 0.0f;"

    def _load(self, row_meta: str, row6: str, indent: str) -> str:
        lane = self.lanes.lane
        return (
            f"{indent}pre_meta = *reinterpret_cast<const int4*>(&mf_meta.data[{row_meta}]);\n"
            f"{indent}if ({lane} < 6) pre_Za = {self.Z_a}.data[{row6} + {lane}];\n"
            f"{indent}if ({lane} >= 6 && {lane} < 12) pre_Zb = {self.Z_b}.data[{row6} + {lane} - 6];"
        )

    def prefetch_first(self, first: str, indent: str) -> str:
        """Load row ``first``'s packed word and entries into the pipeline registers."""
        if first == "0":
            return self._load("off_meta", self.base, indent)
        return self._load(f"off_meta + {first} * 4", f"{self.base} + {first} * 6", indent)

    def consume(self, row: str, end: str, indent: str) -> str:
        """Take row ``row``'s word and entries from the pipeline, load row ``row + 1``'s, and decode the word.

        Declares ``meta``, ``cur_*``, ``dof_a``, ``dof_b``, ``mf_diag`` (the stored inverse diagonal),
        ``packed_tp`` and ``mf_rt`` (the row type).
        """
        return (
            f"{indent}int4 meta = pre_meta;\n"
            f"{indent}float cur_Za = pre_Za;\n{indent}float cur_Zb = pre_Zb;\n"
            f"{indent}if ({row} + 1 < {end}) {{\n"
            f"{indent}    int next_mf6 = {self.base} + ({row} + 1) * 6;\n"
            + self._load(f"off_meta + ({row} + 1) * 4", "next_mf6", indent + "    ")
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
        """``float out = Z_i u`` from the pipelined entries ``cur_Za``/``cur_Zb`` (one thread: row ``i``'s)."""
        lane = self.lanes.lane
        if self.lanes.parallel:
            return (
                f"{indent}float {out}_partial = 0.0f;\n"
                + self._lane_terms(
                    f"{out}_partial",
                    f"cur_Za * {self.velocity.at(f'dof_a + {lane}')}",
                    f"cur_Zb * {self.velocity.at(f'dof_b + {lane} - 6')}",
                    indent,
                )
                + f"\n{indent}{self.lanes.reduce(f'{out}_partial')}\n{indent}float {out} = {out}_partial;"
            )
        if self.order == "loop":
            sides = "\n".join(
                f"{indent}if ({body} >= 0) {{\n"
                f"{indent}    for (int k = 0; k < 6; ++k) {out} += {self._at(Z, 'i', 'k')} * {self._v(body, 'k')};\n"
                f"{indent}}}"
                for Z, body in ((self.Z_a, "dof_a"), (self.Z_b, "dof_b"))
            )
            return f"{indent}float {out} = 0.0f;\n{sides}"
        sides = "\n".join(
            f"{indent}if ({body} >= 0) {{\n{indent}    {out} += "
            + " + ".join(f"{self._at(Z, 'i', k)} * {self._v(body, k)}" for k in range(6))
            + f";\n{indent}}}"
            for Z, body in ((self.Z_a, "dof_a"), (self.Z_b, "dof_b"))
        )
        return f"{indent}float {out} = 0.0f;\n{sides}"

    def apply(self, delta: str, indent: str) -> str:
        """``u += Z_i delta`` from the pipelined entries ``cur_Za``/``cur_Zb`` (one thread: row ``i``'s)."""
        if not self.lanes.parallel:
            return self.apply_row("i", delta, indent)
        lane = self.lanes.lane
        return (
            f"\n{indent}if ({lane} < 6 && dof_a >= 0) {{\n"
            f"{indent}    {self.velocity.at(f'dof_a + {lane}')} += cur_Za * {delta};\n{indent}}}\n"
            f"{indent}if ({lane} >= 6 && {lane} < 12 && dof_b >= 0) {{\n"
            f"{indent}    {self.velocity.at(f'dof_b + {lane} - 6')} += cur_Zb * {delta};\n{indent}}}"
        )

    def row_velocity(self, out: str, row: str, indent: str) -> str:
        """``float out = Z_row u`` of a row outside the pipeline (lanes 0-11, reduced; one thread: from zero)."""
        if not self.lanes.parallel:
            return self._per_coordinate(out, "0.0f", self._velocity_term(row), indent)
        lane = self.lanes.lane
        entries = f"{self.base} + {row} * 6"
        return (
            f"{indent}float {out} = 0.0f;\n"
            + self._lane_terms(
                out,
                f"{self.Z_a}.data[{entries} + {lane}] * {self.velocity.at(f'dof_a + {lane}')}",
                f"{self.Z_b}.data[{entries} + {lane} - 6] * {self.velocity.at(f'dof_b + {lane} - 6')}",
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
                out, "0.0f", lambda Z, body: f"{self._at(Z, row, 'k')} * {self._at(Z, sibling, 'k')}", indent
            )
        lane = self.lanes.lane
        i_row, s_row = f"{self.base} + {row} * 6", f"{self.base} + {sibling} * 6"
        return (
            f"{indent}float {out} = 0.0f;\n"
            + self._lane_terms(
                out,
                f"{self.Z_a}.data[{i_row} + {lane}] * {self.Z_a}.data[{s_row} + {lane}]",
                f"{self.Z_b}.data[{i_row} + {lane} - 6] * {self.Z_b}.data[{s_row} + {lane} - 6]",
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

    def shared_coupling(self, out: str, row: str, column: str, bodies: tuple[str, str], indent: str) -> str:
        """``float out = Z_row Z_column`` over the bodies two rows share (one thread), ``bodies`` the column's.

        The row's bodies are ``dof_a``/``dof_b``; a body is its first velocity coordinate. From zero, each side of
        the row adds its six products with each side of the column on the same body: ``a`` with ``a``, then with
        ``b``, then ``b`` with ``a``, then with ``b``.
        """
        lines = [f"float {out} = 0.0f;"]
        for Z, side in ((self.Z_a, "dof_a"), (self.Z_b, "dof_b")):
            lines.append(f"if ({side} >= 0) {{")
            for C, other in zip((self.Z_a, self.Z_b), bodies, strict=True):
                lines += [
                    f"    if ({side} == {other}) {{",
                    f"        for (int k = 0; k < 6; ++k) {out} += {self._at(Z, row, 'k')} * {self._at(C, column, 'k')};",
                    "    }",
                ]
            lines.append("}")
        return "\n".join(indent + line for line in lines)

    def apply_row(self, row: str, delta: str, indent: str) -> str:
        """``u += Z_row delta`` for a row outside the pipeline (lanes 0-11; one thread: each coordinate in order)."""
        if not self.lanes.parallel:
            return "".join(
                f"\n{indent}if ({body} >= 0) {{"
                + "".join(f"\n{indent}    {self._v(body, k)} += {self._at(Z, row, k)} * {delta};" for k in range(6))
                + f"\n{indent}}}"
                for Z, body in ((self.Z_a, "dof_a"), (self.Z_b, "dof_b"))
            )
        lane = self.lanes.lane
        s_row = f"{self.base} + {row} * 6"
        return (
            f"\n{indent}if ({lane} < 6 && dof_a >= 0)\n"
            f"{indent}    {self.velocity.at(f'dof_a + {lane}')} += {self.Z_a}.data[{s_row} + {lane}] * {delta};\n"
            f"{indent}if ({lane} >= 6 && {lane} < 12 && dof_b >= 0)\n"
            f"{indent}    {self.velocity.at(f'dof_b + {lane} - 6')} += {self.Z_b}.data[{s_row} + {lane} - 6] * {delta};"
        )

    # -- one thread per row: entries read where they are used -------------------------------------------------------

    def _at(self, array: str, row: str, k) -> str:
        if self.entry is not None:
            return self.entry(array, row, str(k))
        return f"{array}.data[{self.base} + {row} * 6 + {k}]"

    def _v(self, first: str, k) -> str:
        return self.velocity.at(f"{first} + {k}" if k != 0 else first)

    def _velocity_term(self, row: str):
        return lambda Z, body: f"{self._at(Z, row, 'k')} * {self._v(body, 'k')}"

    def _per_coordinate(self, out: str, start: str, term, indent: str) -> str:
        """``float out = start``, then per coordinate ``k`` the term ``term(Z, body)`` of body ``a``, then of body
        ``b`` (each side when it has a body)."""
        sides = "\n".join(
            f"{indent}    if ({body} >= 0) {out} += {term(Z, body)};"
            for Z, body in ((self.Z_a, "dof_a"), (self.Z_b, "dof_b"))
        )
        return f"{indent}float {out} = {start};\n{indent}for (int k = 0; k < 6; ++k) {{\n{sides}\n{indent}}}"

    # -- free-body speed rows ---------------------------------------------------------------------------------------

    def speed(self, out: str, body: str, axis: str, indent: str) -> str:
        """``const float out``: the magnitude of the body's physical velocity coordinates ``[axis, axis + 3)``.

        The body's coordinates from ``body`` hold ``u``; every lane forms its physical velocity ``speed_v = L^-T u``
        (:meth:`leave`) and the magnitude, the squares added in coordinate order. The lanes then synchronize, so
        :meth:`scale_speed` may write ``u``.
        """
        factor = self.factor(body)
        lines = [
            "float speed_v[6];",
            "for (int k = 0; k < 6; ++k) speed_v[k] = " + self._v(body, "k") + ";",
            self.leave(lambda k: f"speed_v[{k}]", factor),
            f"const float {out} = sqrtf((speed_v[{axis}] * speed_v[{axis}] + speed_v[{axis} + 1] * speed_v[{axis} + 1])"
            f" + speed_v[{axis} + 2] * speed_v[{axis} + 2]);",
        ]
        if self.lanes.parallel:
            lines.append(self.lanes.sync())
        return "\n".join(indent + line for line in "\n".join(lines).split("\n"))

    def scale_speed(self, scale: str, body: str, axis: str, indent: str) -> str:
        """``v += scale v`` on the coordinates :meth:`speed` read, as ``u += L^T dv`` (one thread, or the first lane)."""
        factor = self.factor(body)
        lines = [
            "float speed_dv[6];",
            f"for (int k = 0; k < 6; ++k) speed_dv[k] = k >= {axis} && k < {axis} + 3 ? {scale} * speed_v[k] : 0.0f;",
            CholeskySolve(6, self.lower(factor), unroll=True).enter(
                lambda k: f"speed_dv[{k}]", lambda k: f"speed_dv[{k}]"
            ),
            "for (int k = 0; k < 6; ++k) " + self._v(body, "k") + " += speed_dv[k];",
        ]
        text = "\n".join(lines)
        if self.lanes.parallel:
            text = f"if ({self.lanes.lane} == 0) {{\n{text}\n}}"
        return "\n".join(indent + line for line in text.split("\n"))


class ResidualBlock(ResponseOperator):
    """The arithmetic of a staged response block ``A`` (``s_A``) whose rows keep their whole residual (``s_res``).

    Row ``r``'s residual ``rhs + J_r v`` is maintained in ``s_res[r]``: a
    change ``delta`` of row ``i`` adds ``A[i][r] delta`` to every row ``r`` of
    the block (``rows``), lane-strided, and the pair coupling ``J_i Y_s`` is the
    entry ``A[s][i]``. A layout places entry ``A[i][r]`` (:meth:`entry`); a view
    of a later range of the block's rows maps its row indices to block rows
    (:meth:`block_row`).
    """

    residual_includes_rhs = True

    def block_row(self, row: str) -> str:
        """The block row of the view's row ``row``."""
        return row

    def entry(self, source: str, target: str) -> tuple[str, str]:
        """``(statements, index)`` of ``A[source][target]``, the response of row ``target`` to row ``source``."""
        raise NotImplementedError

    def residual(self, out: str, indent: str) -> str:
        return f"{indent}float {out} = s_res[{self.block_row('i')}];"

    def sibling_residual(self, out: str, sibling: str, indent: str, rhs: str) -> str:
        """The sibling's maintained residual, which holds its right-hand side ``rhs`` already."""
        return f"{indent}float {out} = s_res[{self.block_row(sibling)}];"

    def coupling(self, out: str, row: str, sibling: str, indent: str) -> str:
        """``J_i Y_s``, the pair's off-diagonal entry, as the block stores it (``A[s][i]``)."""
        setup, index = self.entry(self.block_row(sibling), self.block_row(row))
        lead = f"{indent}{setup}\n" if setup else ""
        return f"{lead}{indent}float {out} = s_A[{index}];"

    def apply_row(self, row: str, delta: str, indent: str) -> str:
        lane, width = self.lanes.lane, self.lanes.width
        setup, index = self.entry(self.block_row(row), "r")
        head = f"{indent}for (int r = {lane}; r < {self.rows}; r += {width})"
        update = f"s_res[r] += s_A[{index}] * {delta};"
        if not setup:
            return f"{head} {update}"
        return f"{head} {{\n{indent}    {setup}\n{indent}    {update}\n{indent}}}"

    def apply(self, delta: str, indent: str) -> str:
        return self.apply_row("i", delta, indent)


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
class Articulation:
    """One articulation's block of a whitened row: ``size`` coordinates from ``offset`` (a constant named
    ``offset_name``), its factor and Jacobian rows in the size group ``group`` of ``L`` and ``J``, its factor staged
    from ``first`` in ``s_L`` (its reciprocal diagonal from ``first_inverse`` in ``s_Dinv``)."""

    size: int
    offset: int
    offset_name: str
    group: str
    L: str
    J: str
    first: int
    first_inverse: int
    guard: str = ""
    """A condition for the articulation's presence in a world (none for the primary)."""

    @property
    def factor(self) -> CholeskySolve:
        lower = f"s_L + {self.first}" if self.first else "s_L"
        inverse = f"s_Dinv + {self.first_inverse}" if self.first_inverse else "s_Dinv"
        return CholeskySolve(self.size, lower, inverse=inverse, masked=True)

    def guarded(self, text: str, indent: str = "    ") -> str:
        """``text`` under the articulation's presence."""
        return f"if ({self.guard}) {{\n{_lines([text], indent)}\n}}" if self.guard else text


@dataclass(frozen=True)
class WhitenedRows:
    """A world's rows in whitened coordinates, built in the kernel: ``Z_i = L^-1 J_i^T`` per articulation.

    ``H = L L^T`` block by block: the world's ``primary`` articulation from
    coordinate ``primary_offset`` and at most one ``secondary`` from
    ``secondary_offset``, each a size group of the response's factors
    (``ink_L_a``, ``ink_L_b``; the world's group indices in ``ink_meta``),
    staged with their reciprocal diagonals. ``A = Z^T Z``, so ``Z`` serves as
    both ``J^T`` and ``Y``. Each lane whitens its row of ``rows``: it stages
    the row's physical Jacobian, the group rows ``ink_J_a``, ``ink_J_b``
    (``capacity`` rows a world) or the statements ``read`` (which write
    coordinate ``q`` to ``Zi[q * ZS]``), folds ``J_i v`` into the row's
    right-hand side, solves ``L Z_i = J_i^T`` in place per articulation
    (:class:`CholeskySolve`, masked) and keeps ``Z_i`` in its registers
    (``Jr``). A staged row's coordinates lie ``stride`` apart. ``staged``
    replaces the factor staging and the whitening (rows a row input stages
    itself). The diagonal ``A_ii + cfm`` is formed from the rows; the velocity
    change of the applied impulses, ``u = Z^T delta``, returns to physical
    coordinates through ``L^-T`` per articulation (:meth:`publish`).
    """

    primary: int
    secondary: int
    primary_offset: int
    secondary_offset: int
    dofs: int
    capacity: int
    stride: int
    lanes: Lanes
    read: str | None = None
    staged: str | None = None
    rows: str = "n_rows"

    @property
    def articulations(self) -> tuple[Articulation, ...]:
        """The primary articulation, then the secondary (in the worlds that hold one)."""
        na, nb = self.primary, self.secondary
        primary = Articulation(na, self.primary_offset, "ink_oa", "ink_ga", "ink_L_a", "ink_J_a", 0, 0)
        if not nb:
            return (primary,)
        secondary = Articulation(
            nb, self.secondary_offset, "ink_ob", "ink_gb", "ink_L_b", "ink_J_b", na * na, na, guard="ink_gb >= 0"
        )
        return primary, secondary

    def buffers(self, *, velocity: bool) -> tuple[Buffer, ...]:
        """The staged factors and reciprocal diagonals (after ``s_dv`` when ``velocity``: the operator has none)."""
        na, nb, D = self.primary, self.secondary, self.dofs
        dv = (Buffer("s_dv", "float", (D + 3) & ~3, align=16),) if velocity else ()
        return (*dv, Buffer("s_L", "float", na * na + nb * nb), Buffer("s_Dinv", "float", na + nb))

    def registers(self, indent: str) -> str:
        """Declare each lane's row registers ``Jr``, zero."""
        D = self.dofs
        return f"{indent}float Jr[{D}];\n{indent}#pragma unroll\n{indent}for (int d = 0; d < {D}; ++d) Jr[d] = 0.0f;"

    def _read(self) -> str:
        """Stage the row's physical Jacobian from the articulations' group rows, coordinate ``q`` at ``Zi[q * ZS]``."""
        lines = []
        for art in self.articulations:
            read = [
                f"const float* {art.J}_row = &{art.J}.data[((size_t){art.group} * {self.capacity} + i) * {art.size}];",
                "#pragma unroll",
                f"for (int d = 0; d < {art.size}; ++d) Zi[({art.offset_name} + d) * ZS] = {art.J}_row[d];",
            ]
            lines.append(art.guarded("\n".join(read)))
        return _lines(lines, "        ")

    def stage(self, target: str, *, publish: bool, indent: str) -> str:
        """Stage the factors, then whiten each lane's row into ``target`` (``Zi = &target[i]``) and its registers.

        With ``publish`` the rows are also written to the world's packed rows (:mod:`.packed_rows`, at their
        coordinates: the packed build's rows, now in the tier's own rounding), where the response block's build and the
        publication read them.
        """
        if self.staged is not None:
            return self.staged
        lane, width, D = self.lanes.lane, self.lanes.width, self.dofs
        lines = [
            f"const int {art.group} = ink_meta.data[world * 4 + {2 * k}];" for k, art in enumerate(self.articulations)
        ]
        lines += [f"constexpr int {art.offset_name} = {art.offset};" for art in self.articulations]
        for art in self.articulations:
            n = art.size

            def source(index, art=art, n=n):
                return f"{art.L}.data[(size_t){art.group} * {n} * {n} + {index}]"

            lines.append(art.guarded(art.factor.stage(source, lane, width)))
        lines.append("SYNC();")
        lines += [art.guarded(art.factor.stage_inverse(lane)) for art in self.articulations]
        lines.append("SYNC();")
        row = [
            f"if ({lane} < {self.rows}) {{",
            f"    const int i = {lane};",
            f"    float* Zi = &{target}[i];",
            f"    constexpr int ZS = {self.stride};",
            "    #pragma unroll",
            f"    for (int d = 0; d < {D}; ++d) Zi[d * ZS] = 0.0f;",
            "    {",
            self._read() if self.read is None else self.read,
            "    }",
            "    // J_i v, folded into the row's right-hand side.",
            "    float bjv = 0.0f;",
            "    #pragma unroll",
            f"    for (int d = 0; d < {D}; ++d) bjv += Zi[d * ZS] * s_v[d];",
            "    s_rhs[i] += bjv;",
        ]
        for art in self.articulations:

            def z(index, art=art):
                return f"(Zi + {art.offset_name} * ZS)[{index} * ZS]"

            row.append(_lines([art.guarded(art.factor.whiten(z, z))], "    "))
        row += ["    #pragma unroll", f"    for (int d = 0; d < {D}; ++d) Jr[d] = Zi[d * ZS];"]
        if publish:
            row += [
                "    #pragma unroll",
                f"    for (int d = 0; d < {D}; ++d) packed_Z.data[packed_base + (size_t)i * {D} + d] = Jr[d];",
            ]
        row.append("}")
        return _lines([*lines, *row], indent)

    def publish(self, z: Callable[[str], str], applied: Callable[[str], str], indent: str) -> str:
        """``v = v_in + L^-T u`` with ``u = sum_i Z_i applied_i`` (``z(i)``: ``Z_i`` at coordinate ``d``), in row
        order; each articulation's backward solve on one lane (the primary's on lane 0, the secondary's on the next
        warp's first lane, or lane 1 in one warp)."""
        lane, width, D = self.lanes.lane, self.lanes.width, self.dofs
        lines = [
            f"for (int d = {lane}; d < {D}; d += {width}) {{",
            *("    " + line for line in _applied_sum("u", "0.0f", z, applied, self.rows)),
            "    s_dv[d] = u;",
            "}",
            "SYNC();",
        ]
        owners = ("0", "32" if width > 32 else "1")
        for art, owner in zip(self.articulations, owners, strict=False):

            def dv(index, art=art):
                return f"s_dv[{art.offset_name} + {index}]"

            test = f"{art.guard} && {lane} == {owner}" if art.guard else f"{lane} == {owner}"
            lines += [f"if ({test}) {{", _lines([art.factor.leave(dv, dv)], "    "), "}"]
        lines += [
            "SYNC();",
            f"for (int d = {lane}; d < {D}; d += {width}) {{",
            "    const int global_dof = world_dof_indices.data[dof_map_base + d];",
            "    if (global_dof >= 0) v_out.data[global_dof] = s_v[d] + s_dv[d];",
            "}",
        ]
        return _lines(lines, indent)

    def push_prologue(self, z: Callable[[str, str], str], indent: str) -> str:
        """``w = Z^T x`` before the last word's pushes (``z(row, coordinate)``)."""
        return _lines(_push_prologue(z, self.lanes, self.dofs, self.rows), indent)

    def push_velocity(self, out: str, z: Callable[[str, str], str], lanes: Lanes, indent: str) -> str:
        """``float out = Z_i w`` and ``zz = |Z_i|^2`` of the pushed row ``i`` over ``lanes``, in one ladder."""
        return _lines(_push_velocity(out, lambda d: z("i", d), lanes, self.dofs), indent)

    def push_apply(self, delta: str, z: Callable[[str, str], str], lanes: Lanes, indent: str) -> str:
        """``w += Z_i delta`` after a push of row ``i``."""
        return _lines(_push_apply(delta, lambda d: z("i", d), lanes, self.dofs), indent)

    def push_diagonal(self) -> str:
        """``A_ii = |Z_i|^2 + cfm_i``, from the squares :meth:`push_velocity` reduced."""
        return "zz + world_row_cfm.data[off_dense + i]"


@dataclass(frozen=True)
class Block(ResidualBlock):
    """The response block ``A = Z^T Z`` of a world's whitened rows, staged.

    ``layout`` places the block in ``s_A``. ``"column"``: ``s_A[i * stride + r]
    = A[i][r]``, column ``i`` (the response of every row's residual to row
    ``i``'s impulse) contiguous, for a Gauss-Seidel sweep that maintains each
    row's whole residual: ``s_res[r]`` holds ``rhs + J_r v`` at entry and a
    change ``delta`` of row ``i`` adds ``A[i][r] delta`` to every row ``r`` of
    the ``rows`` the block holds (the incremental tier, which builds it from the
    packed rows: :func:`.sweep_routes.packed_block`). ``"row"``: ``s_A[i *
    stride + j] = Z_i . Z_j``, row ``i`` contiguous (``stride`` pads it), for a
    Jacobi sweep that forms each row's residual from the impulses
    (:meth:`jacobi_residual`).

    The row layout holds ``whitened`` rows (:class:`WhitenedRows`), built in
    the kernel: each lane keeps its row ``Z_i`` in registers and publishes it
    to the world's packed rows (:mod:`.packed_rows`, at their coordinates:
    rows of ``dofs`` entries from ``packed_base``), where every entry, a sum
    over the world's coordinates in order from zero (:meth:`assemble`), reads
    the other rows. The velocity is published once, after the sweeps, from the
    impulses each row applied (:meth:`publish`). A Gauss-Seidel sweep stages
    the diagonal and its inverse (``s_diag``, ``s_inv_diag``).
    """

    dofs: int
    stride: int
    lanes: Lanes = Lanes()
    rows: str = "n_rows_total"
    layout: str = "column"
    whitened: WhitenedRows | None = None
    diagonal: Callable[[str], str] | None = None
    inverse: Callable[[str], str] | None = None
    name = "block"

    LAYOUTS = ("column", "row")

    def __post_init__(self):
        if self.layout not in self.LAYOUTS:
            raise ValueError(f"layout must be one of {self.LAYOUTS}")
        if (self.layout == "row") != (self.whitened is not None):
            raise ValueError("the row layout holds whitened rows, the column layout none")

    def entry(self, source: str, target: str) -> tuple[str, str]:
        if self.layout == "row":
            return "", f"{target} * {self.stride} + {source}"
        return "", f"{source} * {self.stride} + {target}"

    def assemble(self, indent: str = "    ") -> str:
        """Build ``A``'s entries of rows ``[0, rows)``: ``Z_i . Z_r`` summed in coordinate order from zero, a lane per
        storage row ``i`` (its ``Z_i`` in registers), the other rows read from the world's packed rows."""
        lane, width, D, rows = self.lanes.lane, self.lanes.width, self.dofs, self.rows
        lines = [
            f"for (int i = {lane}; i < {rows}; i += {width}) {{",
            f"    for (int r = 0; r < {rows}; ++r) {{",
            f"        const float* Zr = &packed_Z.data[packed_base + (size_t)r * {D}];",
            "        float a = 0.0f;",
            "        #pragma unroll",
            f"        for (int d = 0; d < {D}; ++d) a += Jr[d] * Zr[d];",
            f"        s_A[i * {self.stride} + r] = a;",
            "    }",
            "}",
        ]
        return _lines(lines, indent)

    def publish(self, indent: str = "    ", *, applied: Callable[[str], str] = lambda row: f"s_app[{row}]") -> str:
        """``v = v_in + P^T L^-T sum_i Z_i applied_i`` in row order, written through the world's coordinate map."""
        return self.whitened.publish(lambda row: self._z(row, "d"), applied, indent)

    # -- Jacobi actions (row layout): every row at once, one thread per row ---------------------------------------

    def _row(self, row: str, column: str) -> str:
        """``A[row][column]``: the response of row ``row`` to row ``column``'s impulse."""
        return f"s_A[{self.entry(column, row)[1]}]"

    def registers(self, indent: str) -> str:
        """Declare the lanes' row registers (``Jr``), which hold each lane's ``Z_i``."""
        return self.whitened.registers(indent)

    def jacobi_stage(self, padding: str, scale: Callable[[str, str, str], str], indent: str) -> str:
        """Whiten and publish the rows, build ``A``, start every row's residual at zero impulse and set each row's
        step scale.

        ``s_rhs[i]`` becomes ``b'_i = rhs_i + J_i v - (A lambda_0)_i``, the residual at zero impulse (``J_i v`` folded
        in at staging), its product with ``lambda_0`` summed in column order; each row's storage is padded with zeros
        to a multiple of four columns, and ``padding`` pads the rows' impulses. ``scale(aii, bound, indent)`` sets an
        admitted row's step from ``A_ii`` and its absolute sum over the admitted columns, ``sum_j |A_ij|`` (in column
        order).
        """
        lane, width, rows = self.lanes.lane, self.lanes.width, self.rows
        stage = [self.whitened.stage("s_A", publish=True, indent=indent), f"{indent}SYNC();", self.assemble(indent)]
        lines = [
            f"for (int i = {lane}; i < {rows}; i += {width}) {{",
            f"    for (int j = {rows}; j < (({rows} + 3) & ~3); ++j) {self._row('i', 'j')} = 0.0f;",
            "}",
            padding,
            "SYNC();",
            f"for (int i = {lane}; i < {rows}; i += {width}) {{",
            "    float acc = 0.0f;",
            f"    for (int j = 0; j < {rows}; ++j) acc += {self._row('i', 'j')} * s_lam0[j];",
            "    s_rhs[i] -= acc;",
            "}",
            "SYNC();",
            f"for (int i = {lane}; i < {rows}; i += {width}) {{",
            "    if (s_kind[i] >= 0) {",
            "        float rowsum = 0.0f;",
            f"        for (int j = 0; j < {rows}; ++j) if (s_kind[j] >= 0) rowsum += fabsf({self._row('i', 'j')});",
            f"        const float aii = {self._row('i', 'i')};",
            scale("aii", "rowsum", "        "),
            "    }",
            "}",
            "SYNC();",
        ]
        return "\n".join([*stage, _lines(lines, indent)])

    def jacobi_residual(self, out: str, indent: str) -> str:
        """``const float out = b'_i + (A y)_i``: four accumulators over the float4 groups of ``y`` (the rows padded
        with zeros to a multiple of four), the first from ``b'_i``, joined pairwise."""
        lines = [
            f"const float* Ai = &s_A[i * {self.stride}];",
            "const float4* y4 = reinterpret_cast<const float4*>(s_y);",
            "float r0 = s_rhs[i], r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;",
            f"const int n4 = ({self.rows} + 3) >> 2;",
            "for (int j4 = 0; j4 < n4; ++j4) {",
            "    const float4 yv = y4[j4];",
            "    const int j = 4 * j4;",
            "    r0 += Ai[j] * yv.x;",
            "    r1 += Ai[j + 1] * yv.y;",
            "    r2 += Ai[j + 2] * yv.z;",
            "    r3 += Ai[j + 3] * yv.w;",
            "}",
            f"const float {out} = (r0 + r1) + (r2 + r3);",
        ]
        return _lines(lines, indent)

    def push_prologue(self, indent: str) -> str:
        """Before the pushes of the last word: ``w = Z^T x``."""
        return self.whitened.push_prologue(self._z, indent)

    def push_velocity(self, out: str, lanes: Lanes, indent: str) -> str:
        """``float out = Z_i w`` and ``zz = |Z_i|^2`` over ``lanes``: the residual's impulse term."""
        return self.whitened.push_velocity(out, self._z, lanes, indent)

    def push_apply(self, delta: str, lanes: Lanes, indent: str) -> str:
        """After a push of row ``i``: ``w += Z_i delta``."""
        return self.whitened.push_apply(delta, self._z, lanes, indent)

    def push_diagonal(self) -> str:
        """``A_ii`` of a pushed row, formed from its row."""
        return self.whitened.push_diagonal()

    def _z(self, row: str, coordinate: str) -> str:
        """``Z`` at ``(coordinate, row)``: the whitened rows live in the world's packed rows."""
        return f"packed_Z.data[packed_base + (size_t)({row}) * {self.dofs} + ({coordinate})]"

    def buffers(self, capacity: int) -> tuple[Buffer, ...]:
        """The block of ``capacity`` rows (row layout: padded by its stride), then the whitened rows' factors."""
        whitened = self.whitened.buffers(velocity=True) if self.whitened is not None else ()
        return (Buffer("s_A", "float", capacity * self.stride), *whitened)


@dataclass(frozen=True)
class TransposedResponse(ResponseOperator):
    """Whitened rows ``Z`` in registers and staged transposed: ``A y = Z^T (Z y)``, one lane a row.

    Each lane holds its row ``Z_i`` in registers (``Jr``); the block stages
    them coordinate-major, ``s_Yt[d * stride + i] = Z_i[d]`` (``stride =
    capacity + 4``: past the rows by a float4, so that neighbouring
    coordinates fall in different banks), with zero columns up to a multiple
    of four rows. Each sweep forms ``w = Y^T y`` once (:meth:`gather`): a lane
    per coordinate and chunk of the rows' float4 groups (``chunks`` of them,
    as many as the lanes cover), four accumulators joined pairwise, the chunks
    summed in order; a row's residual is then ``b'_i + J_i w`` in coordinate
    order. The step scale's bound is ``A``'s exact absolute row sum
    (``exact``, each entry ``J_i Y_j`` summed in coordinate order) or the
    column-sum bound ``sum_d |J_id| sum_j |Y_jd|``. The rows are ``whitened`` in the kernel
    (:class:`WhitenedRows`): ``Z`` serves as both ``J^T`` and ``Y``.
    """

    dofs: int
    capacity: int
    lanes: Lanes = Lanes()
    exact: bool = False
    whitened: WhitenedRows | None = None
    rows: str = "n_rows"

    def __post_init__(self):
        if self.whitened is None:
            raise ValueError("the transposed response holds whitened rows")

    diagonal: Callable[[str], str] | None = None
    inverse: Callable[[str], str] | None = None
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
        """The staged block, the column sums of the bound, ``w`` and the chunks' partial ``w``, the whitened factors."""
        D = self.dofs
        buffers = [Buffer("s_Yt", "float", self.stride * D, align=16)]
        if not self.exact:
            buffers.append(Buffer("s_colsum", "float", D))
        buffers.append(Buffer("s_dv", "float", (D + 3) & ~3, align=16))
        if self.chunks > 1:
            buffers.append(Buffer("s_dvp", "float", self.chunks * D))
        buffers += self.whitened.buffers(velocity=False)
        return tuple(buffers)

    def registers(self, indent: str) -> str:
        """Declare each lane's row registers ``Jr``, zero."""
        D = self.dofs
        return f"{indent}float Jr[{D}];\n{indent}#pragma unroll\n{indent}for (int d = 0; d < {D}; ++d) Jr[d] = 0.0f;"

    def jacobi_stage(self, padding: str, scale: Callable[[str, str, str], str], indent: str) -> str:
        """Whiten and stage the rows, start every row's residual at zero impulse and set each step scale.

        ``w_0 = -Z^T lambda_0`` (``J_i v`` was folded in at staging) is
        summed per coordinate in row order, then ``s_rhs[i]`` becomes ``b'_i = rhs_i + J_i w_0``; ``padding`` pads the
        rows' impulses. ``scale(aii, bound, indent)`` sets an admitted row's step.
        """
        lane, width, D, AM, rows = self.lanes.lane, self.lanes.width, self.dofs, self.capacity, self.rows
        stage = [self.whitened.stage("s_Yt", publish=False, indent=indent)]
        column = [] if self.exact else ["    if (s_kind[j] >= 0) csum += fabsf(yv);"]
        if self.exact:
            bound = [
                f"        for (int j = 0; j < {rows}; ++j) {{",
                "            if (s_kind[j] < 0) continue;",
                "            float acc = 0.0f;",
                "            #pragma unroll",
                f"            for (int d = 0; d < {D}; ++d) acc += Ji[d] * {self._yt('j', 'd')};",
                "            if (j == i) aii = acc;",
                "            bound += fabsf(acc);",
                "        }",
            ]
        else:
            bound = [
                "        #pragma unroll",
                f"        for (int d = 0; d < {D}; ++d) {{",
                f"            aii += Ji[d] * {self._yt('i', 'd')};",
                "            bound += fabsf(Ji[d]) * s_colsum[d];",
                "        }",
            ]
        lines = [
            f"for (int e = {lane}; e < {D} * 4; e += {width}) {{",
            "    const int d = e >> 2;",
            f"    const int j = {rows} + (e & 3);",
            f"    if (j < {AM}) {self._yt('j', 'd')} = 0.0f;",
            "}",
            "SYNC();",
            f"for (int d = {lane}; d < {D}; d += {width}) {{",
            "    float acc = 0.0f;" if self.exact else "    float acc = 0.0f, csum = 0.0f;",
            f"    for (int j = 0; j < {rows}; ++j) {{",
            f"        const float yv = {self._yt('j', 'd')};",
            "        acc += yv * s_lam0[j];",
            *("    " + line for line in column),
            "    }",
            "    s_dv[d] = 0.0f - acc;",
            *([] if self.exact else ["    s_colsum[d] = csum;"]),
            "}",
            "SYNC();",
            f"for (int i = {lane}; i < {rows}; i += {width}) {{",
            "    const float* Ji = Jr;",
            "    float b = s_rhs[i];",
            "    #pragma unroll",
            f"    for (int d = 0; d < {D}; ++d) b += Ji[d] * s_dv[d];",
            "    s_rhs[i] = b;",
            "    if (s_kind[i] >= 0) {",
            "        float aii = 0.0f, bound = 0.0f;",
            *bound,
            scale("aii", "bound", "        "),
            "    }",
            "}",
            padding,
            "SYNC();",
        ]
        return "\n".join([*stage, _lines(lines, indent)])

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
        """``v = v_in + P^T L^-T sum_i Z_i applied_i`` in row order from the staged block."""
        return self.whitened.publish(lambda row: self._yt(row, "d"), applied, indent)

    def push_prologue(self, indent: str) -> str:
        """``w = Y^T x`` before the pushes of the last word."""
        return _lines(_push_prologue(self._yt, self.lanes, self.dofs, self.rows), indent)

    def push_velocity(self, out: str, lanes: Lanes, indent: str) -> str:
        """``float out = Z_i w`` and ``zz = |Z_i|^2`` over ``lanes``."""
        return self.whitened.push_velocity(out, self._yt, lanes, indent)

    def push_apply(self, delta: str, lanes: Lanes, indent: str) -> str:
        """``w += Y_i delta`` after a push of row ``i``."""
        return _lines(_push_apply(delta, lambda d: self._yt("i", d), lanes, self.dofs), indent)

    def push_diagonal(self) -> str:
        """``A_ii`` of a pushed row, formed from its row."""
        return self.whitened.push_diagonal()


@dataclass(frozen=True)
class SparsePairJY(ResponseOperator):
    """Rows of ``J`` and ``Y`` over a world's dense group of ``dofs`` coordinates plus two scalar sparse coordinates.

    The sparse-diagonal rows: a world pairs one compact dense articulation (its group ``group``, first coordinate
    ``offset``) with diagonal-mass coordinates. Row ``i``'s dense part is ``J[(group * rows + i) * dofs + l]`` (and
    ``Y``) on coordinate ``offset + l``; its sparse part is two slots ``s``: coordinate
    ``sparse_dof[(base + i) * 2 + s]`` (``-1`` for none, below ``-1`` an independent chain's link),
    ``J = sparse_jy[(base + i) * 4 + 2 s]`` and ``Y = sparse_jy[(base + i) * 4 + 2 s + 1]``.

    With lanes (a warp, or a segment of ``width`` lanes), lane ``l < dofs`` owns dense coordinate ``l`` and lanes
    ``dofs`` and ``dofs + 1`` the slots: each lane's one product, zero without a coordinate, is reduced with the
    lanes' sum; the other lanes add zero. A segment of at least ``dofs + 2`` lanes reduces the same products in the
    same order as the warp, up to the sign of a zero sum. The visited row's lane entries are ``coord``,
    ``jacobian`` and ``response`` (:meth:`consume`).

    With one thread (``Lanes(1)``), a row on the single coordinate ``coordinate`` (an independent scalar contact,
    whose dense part is empty), through the slot that holds it: its product is rounded on its own
    (``__fmul_rn``, no contraction into the right-hand side's add), as the lanes' sum of one product rounds it.

    Built by the response's Cholesky solves, ``A_ii`` is stored as a value; the step divides by it.
    """

    dofs: int
    rows: int
    lanes: Lanes = Lanes()
    J: str = "dense_J"
    Y: str = "dense_Y"
    group: str = "dense_group"
    offset: str = "dense_offset"
    sparse_dof: str = "sparse_row_dof"
    sparse_jy: str = "sparse_row_jy"
    base: str = "row_base"
    coordinate: str = "scalar_coord"
    velocity: Velocity = Velocity()
    diagonal: Callable[[str], str] | None = None
    inverse: Callable[[str], str] | None = None
    name = "sparse_pair_jy"

    # -- build (response_builds_rows) ---------------------------------------------------------------------------------

    @staticmethod
    def build():
        """The rows' build (:data:`.response_builds_rows.SPARSE_PAIR`): each row's sparse slots ``(J, J / m)``,
        merged on a shared coordinate, its dense part through the compact factor, and ``A_ii = (dense + sparse) +
        cfm``."""
        from .response_builds_rows import SPARSE_PAIR  # noqa: PLC0415

        return SPARSE_PAIR

    def _entries(self, row: str, coord: str, jacobian: str, response: str | None, indent: str, active=None) -> str:
        """Declare this lane's coordinate and entries of row ``row`` (``coord`` negative for none)."""
        lane, P = self.lanes.lane, self.dofs
        guard = f"{active} && " if active else ""
        lines = [f"int {coord} = -1;", f"float {jacobian} = 0.0f;"]
        if response:
            lines.append(f"float {response} = 0.0f;")
        lines += [
            f"if ({guard}{lane} < {P}) {{",
            f"    {coord} = {self.offset} + {lane};",
            f"    const int dense_index = ({self.group} * {self.rows} + {row}) * {P} + {lane};",
            f"    {jacobian} = {self.J}.data[dense_index];",
        ]
        if response:
            lines.append(f"    {response} = {self.Y}.data[dense_index];")
        lines += [
            f"}} else if ({guard}{lane} <= {P + 1}) {{",
            f"    const int sparse_slot = {lane} - {P};",
            f"    const int sparse_row = {self.base} + {row};",
            f"    {coord} = {self.sparse_dof}.data[sparse_row * 2 + sparse_slot];",
            f"    {jacobian} = {self.sparse_jy}.data[sparse_row * 4 + sparse_slot * 2];",
        ]
        if response:
            lines.append(f"    {response} = {self.sparse_jy}.data[sparse_row * 4 + sparse_slot * 2 + 1];")
        lines.append("}")
        return "\n".join(indent + line for line in lines)

    def _slot(self, row: str, name: str, indent: str) -> str:
        """One thread: declare ``name``, the entry index of row ``row``'s slot that holds its coordinate."""
        sparse_row = f"({self.base} + {row})"
        return f"{indent}const int {name} = {sparse_row} * 4 + ({self.sparse_dof}.data[{sparse_row} * 2] >= 0 ? 0 : 2);"

    def consume(self, row: str, indent: str) -> str:
        """Declare the visited row ``row``'s entries: this lane's ``coord``, ``jacobian`` and ``response``."""
        if not self.lanes.parallel:
            return (
                f"{self._slot(row, 'row_entry', indent)}\n"
                f"{indent}const float jacobian = {self.sparse_jy}.data[row_entry];\n"
                f"{indent}const float response = {self.sparse_jy}.data[row_entry + 1];"
            )
        return self._entries(row, "coord", "jacobian", "response", indent)

    def residual(self, out: str, indent: str) -> str:
        """``float out = J_i v`` from the consumed entries (every lane holds it)."""
        if not self.lanes.parallel:
            return f"{indent}float {out} = __fmul_rn(jacobian, {self.velocity.at(self.coordinate)});"
        return (
            f"{indent}float {out}_partial = coord >= 0 ? jacobian * {self.velocity.at('coord')} : 0.0f;\n"
            f"{indent}{self.lanes.reduce(f'{out}_partial')}\n"
            f"{indent}float {out} = {out}_partial;"
        )

    def row_velocity(self, out: str, row: str, indent: str, *, active: str | None = None) -> str:
        """``float out = J_row v`` of a row outside the visit (``active``: whether this lane's row exists)."""
        if not self.lanes.parallel:
            return (
                f"{self._slot(row, f'{out}_entry', indent)}\n"
                f"{indent}float {out} = __fmul_rn({self.sparse_jy}.data[{out}_entry], "
                f"{self.velocity.at(self.coordinate)});"
            )
        return (
            self._entries(row, f"{out}_coord", f"{out}_jacobian", None, indent, active)
            + f"\n{indent}float {out} = {out}_coord >= 0 ? {out}_jacobian * {self.velocity.at(f'{out}_coord')} : 0.0f;"
            + f"\n{indent}{self.lanes.reduce(out)}"
        )

    def pair_entries(self, sibling: str, indent: str) -> str:
        """Load the sibling's entries once for the pair's terms and its apply: one thread's slot entry (``sib_entry``,
        its ``J`` and ``Y``); a lane's coordinate, ``J`` and ``Y`` of the sibling, and the sibling's ``Y`` on the
        lane's own coordinate (``sib_response_here``: a sparse slot's when the sibling carries that coordinate)."""
        if not self.lanes.parallel:
            return (
                f"{self._slot(sibling, 'sib_entry', indent)}\n"
                f"{indent}const float sib_jacobian = {self.sparse_jy}.data[sib_entry];\n"
                f"{indent}const float sib_response = {self.sparse_jy}.data[sib_entry + 1];"
            )
        lane, P = self.lanes.lane, self.dofs
        lines = [
            self._entries(sibling, "sib_coord", "sib_jacobian", "sib_response", ""),
            "float sib_response_here = sib_response;",
            f"if ({lane} >= {P} && {lane} <= {P + 1}) {{",
            f"    const int sib_row = {self.base} + {sibling};",
            "    sib_response_here = 0.0f;",
            f"    if (coord >= 0 && coord == {self.sparse_dof}.data[sib_row * 2])",
            f"        sib_response_here = {self.sparse_jy}.data[sib_row * 4 + 1];",
            f"    else if (coord >= 0 && coord == {self.sparse_dof}.data[sib_row * 2 + 1])",
            f"        sib_response_here = {self.sparse_jy}.data[sib_row * 4 + 3];",
            "}",
        ]
        return "\n".join(indent + line for line in "\n".join(lines).split("\n"))

    def pair_terms(self, row: str, sibling: str, rhs: Callable[[str], str], indent: str, *, own: bool = False) -> str:
        """The pair's terms from the sibling's entries (:meth:`pair_entries`): one thread's two products, a lane's
        two products reduced in one ladder."""
        if own:
            return super().pair_terms(row, sibling, rhs, indent, own=own)
        if not self.lanes.parallel:
            return (
                f"{indent}float sibling_residual = __fmul_rn(sib_jacobian, {self.velocity.at(self.coordinate)});\n"
                f"{indent}sibling_residual += {rhs(sibling)};\n"
                f"{indent}float cross = jacobian * sib_response;"
            )
        return (
            f"{indent}float sibling_residual = sib_coord >= 0 ? sib_jacobian * {self.velocity.at('sib_coord')} : 0.0f;\n"
            f"{indent}float cross = coord >= 0 ? jacobian * sib_response_here : 0.0f;\n"
            f"{indent}{self.lanes.reduce('sibling_residual', 'cross')}\n"
            f"{indent}sibling_residual += {rhs(sibling)};"
        )

    def coupling(self, out: str, row: str, sibling: str, indent: str) -> str:
        """``float out = J_i Y_s`` of the visited row ``row``: each lane's ``J`` times the sibling's ``Y`` at its
        coordinate (a sparse slot's, when the sibling carries that coordinate), reduced."""
        if not self.lanes.parallel:
            return (
                f"{self._slot(sibling, f'{out}_entry', indent)}\n"
                f"{indent}float {out} = jacobian * {self.sparse_jy}.data[{out}_entry + 1];"
            )
        lane, P = self.lanes.lane, self.dofs
        lines = [
            f"float {out}_response = 0.0f;",
            f"if ({lane} < {P}) {{",
            f"    {out}_response = {self.Y}.data[({self.group} * {self.rows} + {sibling}) * {P} + {lane}];",
            f"}} else if ({lane} <= {P + 1}) {{",
            f"    const int {out}_row = {self.base} + {sibling};",
            f"    const int {out}_coord_0 = {self.sparse_dof}.data[{out}_row * 2];",
            f"    const int {out}_coord_1 = {self.sparse_dof}.data[{out}_row * 2 + 1];",
            f"    if (coord >= 0 && coord == {out}_coord_0)",
            f"        {out}_response = {self.sparse_jy}.data[{out}_row * 4 + 1];",
            f"    else if (coord >= 0 && coord == {out}_coord_1)",
            f"        {out}_response = {self.sparse_jy}.data[{out}_row * 4 + 3];",
            "}",
            f"float {out} = coord >= 0 ? jacobian * {out}_response : 0.0f;",
            self.lanes.reduce(out),
        ]
        return "\n".join(indent + line for line in lines)

    def apply(self, delta: str, indent: str) -> str:
        """``v += Y_i delta`` from the consumed entries."""
        if not self.lanes.parallel:
            return f"\n{indent}{self.velocity.at(self.coordinate)} += response * {delta};"
        return f"\n{indent}if (coord >= 0) {self.velocity.at('coord')} += response * {delta};"

    def apply_row(self, row: str, delta: str, indent: str) -> str:
        """``v += Y_row delta`` for a row outside the visit; the pair's sibling ``sib`` from its kept entries
        (:meth:`pair_entries`)."""
        if row == "sib":
            if not self.lanes.parallel:
                return f"\n{indent}{self.velocity.at(self.coordinate)} += sib_response * {delta};"
            return f"\n{indent}if (sib_coord >= 0) {self.velocity.at('sib_coord')} += sib_response * {delta};"
        if not self.lanes.parallel:
            return (
                f"\n{self._slot(row, 'apply_entry', indent)}\n"
                f"{indent}{self.velocity.at(self.coordinate)} += {self.sparse_jy}.data[apply_entry + 1] * {delta};"
            )
        lane, P = self.lanes.lane, self.dofs
        lines = [
            "int apply_coord = -1;",
            "float apply_response = 0.0f;",
            f"if ({lane} < {P}) {{",
            f"    apply_coord = {self.offset} + {lane};",
            f"    apply_response = {self.Y}.data[({self.group} * {self.rows} + {row}) * {P} + {lane}];",
            f"}} else if ({lane} <= {P + 1}) {{",
            f"    const int sparse_slot = {lane} - {P};",
            f"    const int sparse_row = {self.base} + {row};",
            f"    apply_coord = {self.sparse_dof}.data[sparse_row * 2 + sparse_slot];",
            f"    apply_response = {self.sparse_jy}.data[sparse_row * 4 + sparse_slot * 2 + 1];",
            "}",
            f"if (apply_coord >= 0) {self.velocity.at('apply_coord')} += apply_response * {delta};",
        ]
        return "".join(f"\n{indent}{line}" for line in lines)


@dataclass(frozen=True)
class Assembled(ResponseOperator):
    """The assembled Delassus operator ``C = J H^-1 J^T`` of a world's rows, in impulse space.

    A row's velocity term is ``C_i lambda``: lane ``l`` sums its columns ``l + 32 k`` in ascending ``k`` from
    zero and the lanes reduce with ``SUM``. Every lane computes every impulse and stores it, as every other lane
    does, so each reads only its own stores: nothing is applied or synchronized (:attr:`applies`). ``count`` rows
    of capacity ``capacity`` are staged; row ``i`` of ``C`` is ``world_C[off2 + i * stride + j]``.

    ``staging`` says where ``C`` is read. ``"shared"``: its packed lower triangle ``s_Ctri[i (i + 1) / 2 + j]``
    (``j <= i``), staged by :meth:`stage`, read as ``C(i, j)`` for ``j <= i`` and ``C(j, i)`` otherwise.
    ``"global"``: its rows in global memory, ``C(i, j)`` for every ``j``. ``C`` is assembled in floating point and
    is not exactly symmetric, so above the diagonal the two stagings read different entries. A pair's coupling is
    ``C(s, i)`` (``s = i + 1``), an entry both stagings read. Built with the response, ``A_ii`` is stored as a
    value; the step divides by it.

    With one thread (``Lanes(1)``, the loop sweep's layout) a row's sum starts from its right-hand side ``rhs``,
    so the residual is whole (:attr:`residual_includes_rhs`), and adds ``C(i, j) lambda_j`` for ``j`` ascending;
    the coupling of rows ``i`` and ``s`` is row ``i``'s entry ``C(i, s)``, read as row ``i``'s sum reads it. The
    thread walks the ``count`` rows: ``capacity`` (the lanes' column registers, the shared triangle) is unused.
    """

    capacity: int
    stride: str
    staging: str = "shared"
    lanes: Lanes = Lanes()
    impulse: str = "s_lam"
    count: str = "m"
    rhs: Callable[[str], str] | None = None
    """The rows' right-hand sides, which one thread's sums start from."""
    diagonal: Callable[[str], str] | None = None
    inverse: Callable[[str], str] | None = None
    name = "assembled"
    applies = False

    STAGINGS = ("shared", "global")

    @staticmethod
    def build(stage: str, mapping: str, **options):
        """The launch that builds stage ``stage`` of the operator's data with the thread mapping ``mapping``.

        ``C = J Y^T`` and its diagonal accumulate onto a zeroed ``C`` and diagonal from a size group's response (the
        dense response build, :data:`.response_builds_dense.DENSE_JY_BUILDS`), or the response and ``C`` are built
        fused; then the diagonals' ``cfm``. Returns a
        :class:`.response_builds_dense.Build` (:data:`.response_builds_dense.ASSEMBLED_BUILDS` lists the stages and
        mappings).
        """
        from .response_builds_dense import ASSEMBLED_BUILDS  # noqa: PLC0415

        return ASSEMBLED_BUILDS[stage, mapping](**options)

    def __post_init__(self):
        if self.staging not in self.STAGINGS:
            raise ValueError(f"staging must be one of {self.STAGINGS}, got {self.staging!r}")
        if not self.lanes.parallel and self.rhs is None:
            raise ValueError("one thread sums a row's residual from its right-hand side: give rhs")

    @property
    def residual_includes_rhs(self) -> bool:
        """One thread sums a row's whole residual, from its right-hand side; lanes sum ``C_i lambda``."""
        return not self.lanes.parallel

    @property
    def shared(self) -> bool:
        return self.staging == "shared"

    @property
    def triangle(self) -> int:
        """Entries of the packed lower triangle of ``capacity`` rows."""
        return self.capacity * (self.capacity + 1) // 2

    @property
    def columns(self) -> int:
        """Columns per lane."""
        return (self.capacity + self.lanes.width - 1) // self.lanes.width

    def _column(self, k: int) -> str:
        return f"{self.lanes.lane} + {k * self.lanes.width}"

    def _entry(self, row: str, column: str, triangle_base: str, row_base: str, column_base: str) -> str:
        """``C(row, column)`` as the staging reads it, given the row's and the column's triangle bases (shared) or
        the row's first entry (global)."""
        if self.shared:
            return f"{column} <= {row} ? s_Ctri[{triangle_base} + {column}] : s_Ctri[{column_base} + {row}]"
        return f"world_C.data[{row_base} + {column}]"

    def prepare(self, indent: str) -> str:
        """Declare each lane's columns ``j_k`` (and their triangle bases ``jb_k``), once before the sweeps."""
        lines = []
        for k in range(self.columns):
            lines.append(f"{indent}const int j{k} = {self._column(k)};")
            if self.shared:
                lines.append(f"{indent}const int jb{k} = (j{k} * (j{k} + 1)) >> 1;")
        return "\n".join(lines)

    def stage(self, indent: str) -> str:
        """Stage the lower triangle of the active ``count`` rows (shared), each row's columns lane-strided."""
        if not self.shared:
            return f"{indent}// Delassus rows stay in global memory: each update reads its row, coalesced across lanes."
        lane, width = self.lanes.lane, self.lanes.width
        return (
            f"{indent}for (int i = 0; i < {self.count}; ++i) {{\n"
            f"{indent}    const int base = (i * (i + 1)) >> 1;\n"
            f"{indent}    for (int j = {lane}; j <= i; j += {width}) "
            f"s_Ctri[base + j] = world_C.data[off2 + i * {self.stride} + j];\n"
            f"{indent}}}"
        )

    def _row_sum(self, out: str, row: str, start: str, indent: str) -> str:
        """One thread: ``float out = start``, then ``C(row, j) lambda_j`` added for ``j`` ascending."""
        base = (
            f"const int {out}_base = ({row} * ({row} + 1)) >> 1;"
            if self.shared
            else f"const int {out}_row = off2 + {row} * {self.stride};"
        )
        entry = self._entry(row, "j", f"{out}_base", f"{out}_row", "((j * (j + 1)) >> 1)")
        lines = [
            f"float {out} = {start};",
            base,
            f"for (int j = 0; j < {self.count}; ++j) {{",
            f"    const float c = {entry};",
            f"    {out} += c * {self.impulse}[j];",
            "}",
        ]
        return "\n".join(indent + line for line in lines)

    def residual(self, out: str, indent: str) -> str:
        """``float out = C_i lambda`` of the visited row ``i`` over the prepared columns (every lane holds it); one
        thread: the whole residual ``rhs_i + C_i lambda``."""
        if not self.lanes.parallel:
            return self._row_sum(out, "i", self.rhs("i"), indent)
        base = (
            "const int base_i = (i * (i + 1)) >> 1;" if self.shared else f"const int row_i = off2 + i * {self.stride};"
        )
        lines = [f"float {out}_partial = 0.0f;", base]
        for k in range(self.columns):
            entry = self._entry("i", f"j{k}", "base_i", "row_i", f"jb{k}")
            lines += [
                f"if (j{k} < {self.count}) {{",
                f"    const float c = {entry};",
                f"    {out}_partial += c * {self.impulse}[j{k}];",
                "}",
            ]
        lines += [self.lanes.reduce(f"{out}_partial"), f"float {out} = {out}_partial;"]
        return "\n".join(indent + line for line in lines)

    def sibling_residual(self, out: str, sibling: str, indent: str, rhs: str) -> str:
        """``float out = C_s lambda + rhs``: the sibling's row, lane-strided over its columns, reduced, then its
        right-hand side; one thread sums the row from its right-hand side."""
        if not self.lanes.parallel:
            return self._row_sum(out, sibling, rhs, indent)
        lane, width = self.lanes.lane, self.lanes.width
        base = (
            f"const int sib_base = ({sibling} * ({sibling} + 1)) >> 1;"
            if self.shared
            else f"const int sib_row = off2 + {sibling} * {self.stride};"
        )
        entry = self._entry(sibling, "j", "sib_base", "sib_row", "((j * (j + 1)) >> 1)")
        lines = [
            f"float {out} = 0.0f;",
            base,
            f"for (int j = {lane}; j < {self.count}; j += {width}) {{",
            f"    const float c = {entry};",
            f"    {out} += c * {self.impulse}[j];",
            "}",
            self.lanes.reduce(out),
            f"{out} += {rhs};",
        ]
        return "\n".join(indent + line for line in lines)

    def coupling(self, out: str, row: str, sibling: str, indent: str) -> str:
        """``float out = C(s, i)``: the pair's off-diagonal entry, below the diagonal (``s > i``); one thread reads
        ``C(i, s)``, as row ``i``'s sum reads it (any two rows ``i`` and ``s``)."""
        if not self.lanes.parallel:
            row_base, sibling_base = (f"(({index} * ({index} + 1)) >> 1)" for index in (row, sibling))
            entry = self._entry(row, sibling, row_base, f"off2 + {row} * {self.stride}", sibling_base)
            return f"{indent}float {out} = {entry};"
        if self.shared:
            return f"{indent}float {out} = s_Ctri[(({sibling} * ({sibling} + 1)) >> 1) + {row}];"
        return f"{indent}float {out} = world_C.data[off2 + {sibling} * {self.stride} + {row}];"

    def apply(self, delta: str, indent: str) -> str:
        """Nothing: the impulses are the state."""
        return ""

    def apply_row(self, row: str, delta: str, indent: str) -> str:
        """Nothing: the impulses are the state."""
        return ""


@dataclass(frozen=True)
class LocalBlock(ResidualBlock):
    """The response block of a local owner's rows, built in the kernel and stored as a packed lower triangle.

    The owner's ``dense`` rows act on its ``dofs`` coordinates: one
    articulation's, then a paired free body's, each a diagonal block of ``H``
    with its staged factor (``factors``, in coordinate order). Each row's ``J``
    and response ``Y = H^-1 J^T`` are staged in ``s_J`` and ``s_Y`` (row ``r``
    from ``r * dofs``), the response solved per block by the factor's
    triangular solves (:meth:`respond`). With ``free``, the world's free-body
    rows (``free_count``, :class:`FreeBodyZ` entries read in place, their
    ``mf_meta`` words from ``off_meta``) follow the dense rows in the block:
    they act on the paired free body, whitened by its held factor
    ``free_factor`` (a :data:`Matrix` of ``L``), so a free row's terms read the
    body in factor coordinates, ``u = L^T v`` and ``L^T Y`` of a dense row.

    ``s_A`` holds the lower triangle: ``A[hi][lo] = J_hi Y_lo`` (``hi >= lo``)
    at ``hi (hi + 1) / 2 + lo``, built lane-strided over the packed entries,
    each a sum in coordinate order from zero (:meth:`assemble`); either row of
    an entry reads it. The residuals start at ``rhs + J v``. The velocity
    ``s_v`` is committed once after the sweeps, every row's applied impulse
    (``s_app``) times its response added in row order, the free rows' as
    ``L^-T sum Z app`` (:meth:`commit`). A view of the free-body rows addresses
    them from block row ``first``.
    """

    dofs: int
    factors: tuple[CholeskySolve, ...] = ()
    lanes: Lanes = Lanes()
    dense: str = "row_count"
    free: FreeBodyZ | None = None
    free_factor: Matrix | None = None
    free_count: str = "mf_count"
    first: str = ""
    diagonal: Callable[[str], str] | None = None
    inverse: Callable[[str], str] | None = None
    name = "local_block"

    @property
    def rows(self) -> str:
        """The block's rows: the dense rows, then the free-body rows."""
        return self.dense if self.free is None else f"{self.dense} + {self.free_count}"

    def block_row(self, row: str) -> str:
        return f"{self.first} + {row}" if self.first else row

    def entry(self, source: str, target: str) -> tuple[str, str]:
        a, b = _operand(source), _operand(target)
        return f"const int hi = {a} > {b} ? {a} : {b}, lo = {a} > {b} ? {b} : {a};", "hi * (hi + 1) / 2 + lo"

    def respond(self, reads: tuple[str, ...], indent: str) -> str:
        """Stage row ``row``'s ``J`` and solve its response ``Y``, in the caller's loop over ``row``.

        ``reads[b]`` is coordinate ``i`` of the row's ``J`` on factor block
        ``b`` (native, in ``row`` and ``i``). Declares ``active`` (some entry of
        ``J`` is nonzero) and ``row_diagonal`` (``J Y``, summed block by block
        in coordinate order). The solves' loop variable shadows ``row``, so the
        row's staged offset is bound first.
        """
        lines = [f"const int staged = row * {self.dofs};", "int active = 0;", "float row_diagonal = 0.0f;"]
        offset = 0
        for block, (solve, read) in enumerate(zip(self.factors, reads, strict=True)):
            first = f"staged + {offset}" if offset else "staged"

            def jacobian(i, first=first):
                return f"s_J[{first} + {i}]"

            def response(i, block=block):
                return f"response_{block}[{i}]"

            lines += [
                f"float response_{block}[{solve.size}];",
                f"for (int i = 0; i < {solve.size}; ++i) {{",
                f"    const float jacobian = {read};",
                f"    {jacobian('i')} = jacobian;",
                "    if (jacobian != 0.0f) active = 1;",
                "}",
                solve.whiten(jacobian, response),
                solve.leave(response, response),
                f"for (int i = 0; i < {solve.size}; ++i) {{",
                f"    s_Y[{first} + i] = {response('i')};",
                f"    row_diagonal += {jacobian('i')} * {response('i')};",
                "}",
            ]
            offset += solve.size
        if offset != self.dofs:
            raise ValueError(f"the factor blocks span {offset} coordinates, the block {self.dofs}")
        return "\n".join(indent + line if line.strip() else line for line in "\n".join(lines).split("\n"))

    def _free_row(self, free: str, name: str, indent: str, *, word: bool = False) -> str:
        """Decode free-body row ``free``'s bodies (``{name}_a``, ``{name}_b``) and entry base (``{name}6``).

        With ``word`` the row's whole packed word is read (``{name}_meta``: bodies, inverse diagonal, rhs, type).
        """
        packed = f"mf_meta.data[off_meta + ({free}) * 4]"
        read = (
            f"{indent}const int4 {name}_meta = *reinterpret_cast<const int4*>(&{packed});\n"
            f"{indent}const int {name}_dofs = {name}_meta.x;\n"
            if word
            else f"{indent}const int {name}_dofs = {packed};\n"
        )
        return (
            f"{read}"
            f"{indent}const int {name}_a = {name}_dofs >> 16;\n"
            f"{indent}const int {name}_b = ({name}_dofs << 16) >> 16;\n"
            f"{indent}const int {name}6 = {self.free.base} + ({free}) * 6;"
        )

    def _free_terms(self, out: str, physical: Callable[[str, str], str], indent: str) -> str:
        """Add free-body row ``row``'s products with a physical vector on its bodies to ``out``: each side's ``Z``
        against the vector in factor coordinates, ``L^T x`` (``physical(side, d)``: coordinate ``d`` of the vector on
        the body at ``side``), body ``a``'s side, then ``b``'s, each from its own sum."""
        free = self.free
        lines = []
        for Z, side in ((free.Z_a, "row_a"), (free.Z_b, "row_b")):
            lines += [
                f"if ({side} >= 0) {{",
                "    float side_u[6];",
                indent_lines(
                    CholeskySolve(6, self.free_factor, unroll=True).enter(
                        lambda d, side=side: physical(side, d), lambda d: f"side_u[{d}]"
                    ),
                    "    ",
                ),
                "    float side_term = 0.0f;",
                f"    for (int d = 0; d < 6; ++d) side_term += {Z}.data[row6 + d] * side_u[d];",
                f"    {out} += side_term;",
                "}",
            ]
        return "\n".join(indent + line for line in "\n".join(lines).split("\n"))

    def assemble(self, rhs: Vector, indent: str) -> str:
        """Start every row's residual at ``rhs + J v`` (``s_res``, ``s_app`` cleared) and build ``A``'s lower triangle.

        A dense row's terms run over the block's coordinates, a free-body row's over its bodies' six in factor
        coordinates (:meth:`_free_terms`), body ``a``'s side and then ``b``'s; a free-body row's response to a
        free-body column sums the column's bodies acting on the row's body, ``Z`` against ``Z``.
        """
        lane, width, T, free = self.lanes.lane, self.lanes.width, self.dofs, self.free
        p = indent
        dense_residual = _at(rhs, "r")
        lines = [f"{p}{{", f"{p}    const int block_rows = {self.rows};"]
        lines.append(f"{p}    for (int r = {lane}; r < block_rows; r += {width}) {{")
        if free is None:
            lines += [
                f"{p}        float residual = {dense_residual};",
                f"{p}        for (int d = 0; d < {T}; ++d) residual += s_J[r * {T} + d] * s_v[d];",
            ]
        else:
            lines += [
                f"{p}        float residual;",
                f"{p}        if (r < {self.dense}) {{",
                f"{p}            residual = {dense_residual};",
                f"{p}            for (int d = 0; d < {T}; ++d) residual += s_J[r * {T} + d] * s_v[d];",
                f"{p}        }} else {{",
                self._free_row(f"r - {self.dense}", "row", p + "            ", word=True),
                f"{p}            residual = __int_as_float(row_meta.z);",
                self._free_terms("residual", lambda side, d: f"s_v[{side} + {d}]", p + "            "),
                f"{p}        }}",
            ]
        lines += [
            f"{p}        s_res[r] = residual;",
            f"{p}        s_app[r] = 0.0f;",
            f"{p}    }}",
            f"{p}    const int entries = block_rows * (block_rows + 1) / 2;",
            f"{p}    int hi = 0, start = 0, next = 1;",
            f"{p}    for (int e = {lane}; e < entries; e += {width}) {{",
            f"{p}        while (e >= next) {{",
            f"{p}            start = next;",
            f"{p}            ++hi;",
            f"{p}            next += hi + 1;",
            f"{p}        }}",
            f"{p}        const int lo = e - start;",
            f"{p}        float a = 0.0f;",
        ]
        dense_entry = f"for (int d = 0; d < {T}; ++d) a += s_J[hi * {T} + d] * s_Y[lo * {T} + d];"
        if free is None:
            lines.append(f"{p}        {dense_entry}")
        else:
            q = p + "                "
            lines += [
                f"{p}        if (hi < {self.dense}) {{",
                f"{p}            {dense_entry}",
                f"{p}        }} else {{",
                self._free_row(f"hi - {self.dense}", "row", p + "            ", word=True),
                f"{p}            if (lo < {self.dense}) {{",
                self._free_terms("a", lambda side, d: f"s_Y[lo * {T} + {side} + {d}]", q),
                f"{p}            }} else {{",
                self._free_row(f"lo - {self.dense}", "column", q, word=True),
                f"{q}for (int d = 0; d < 6; ++d) {{",
                f"{q}    if (row_a >= 0) {{",
                f"{q}        float response = 0.0f;",
                f"{q}        if (column_a == row_a) response += {free.Z_a}.data[column6 + d];",
                f"{q}        if (column_b == row_a) response += {free.Z_b}.data[column6 + d];",
                f"{q}        a += {free.Z_a}.data[row6 + d] * response;",
                f"{q}    }}",
                f"{q}    if (row_b >= 0) {{",
                f"{q}        float response = 0.0f;",
                f"{q}        if (column_a == row_b) response += {free.Z_a}.data[column6 + d];",
                f"{q}        if (column_b == row_b) response += {free.Z_b}.data[column6 + d];",
                f"{q}        a += {free.Z_b}.data[row6 + d] * response;",
                f"{q}    }}",
                f"{q}}}",
                f"{p}            }}",
                f"{p}        }}",
            ]
        lines += [f"{p}        s_A[e] = a;", f"{p}    }}", f"{p}}}"]
        return "\n".join(lines)

    def commit(self, indent: str) -> str:
        """``s_v += sum_r Y_r applied_r`` over every row of the block, in row order (zero terms included); the free
        rows' terms on their body as ``L^-T sum_f Z_f applied_f``, after the dense rows'."""
        lane, width, T, free = self.lanes.lane, self.lanes.width, self.dofs, self.free
        p = indent
        lines = [
            f"{p}for (int d = {lane}; d < {T}; d += {width}) {{",
            f"{p}    float v = s_v[d];",
            f"{p}    for (int r = 0; r < {self.dense}; ++r) v += s_Y[r * {T} + d] * s_app[r];",
        ]
        if free is not None:
            # The free rows' change of a body coordinate: L^-T of the body's sum Z app, every lane forming the
            # whole body's (the free rows act on the paired body alone).
            leave = CholeskySolve(6, self.free_factor, unroll=True).leave(
                lambda k: f"body_du[{k}]", lambda k: f"body_du[{k}]"
            )
            lines += [
                f"{p}    int body_first = -1;",
                f"{p}    float body_du[6] = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};",
                f"{p}    for (int f = 0; f < {self.free_count}; ++f) {{",
                self._free_row("f", "row", p + "        "),
                f"{p}        const float applied = s_app[{self.dense} + f];",
                f"{p}        if (row_a >= 0 && d >= row_a && d < row_a + 6) {{",
                f"{p}            body_first = row_a;",
                f"{p}            for (int k = 0; k < 6; ++k) body_du[k] += {free.Z_a}.data[row6 + k] * applied;",
                f"{p}        }}",
                f"{p}        if (row_b >= 0 && d >= row_b && d < row_b + 6) {{",
                f"{p}            body_first = row_b;",
                f"{p}            for (int k = 0; k < 6; ++k) body_du[k] += {free.Z_b}.data[row6 + k] * applied;",
                f"{p}        }}",
                f"{p}    }}",
                f"{p}    if (body_first >= 0) {{",
                indent_lines(leave, p + "        "),
                f"{p}        v += body_du[d - body_first];",
                f"{p}    }}",
            ]
        lines += [f"{p}    s_v[d] = v;", f"{p}}}"]
        return "\n".join(lines)


@dataclass(frozen=True)
class FactorComponent:
    """One diagonal block of a block-diagonal factor ``L``: ``size`` coordinates from lane ``offset``.

    ``lane`` is this lane's coordinate in the block (``LANE - offset``, out of range on other lanes); ``L`` holds the
    block's leaf-first lower factor in factor coordinates, row-major from ``base``: a Warp array, or with ``staged``
    the block's own shared array. A lane's coordinate is a physical position: factor coordinate ``i`` sits on lane
    ``size - 1 - i`` (:func:`leaf_first`).
    """

    size: int
    offset: str
    lane: str
    L: str
    base: str
    staged: bool = False

    def element(self, row: str, col: str) -> str:
        """Entry ``(row, col)`` of the block's ``L``."""
        data = self.L if self.staged else f"{self.L}.data"
        return f"{data}[{self.base} + ({row}) * {self.size} + ({col})]"


@dataclass(frozen=True)
class FactorRows(ResponseOperator):
    """Rows in factor coordinates: ``Z = L^-1 J^T`` over ``dofs`` coordinates (at most a warp), one lane each.

    With ``H = L L^T`` and the factor velocity ``u = L^T v``, a row's velocity
    term is ``J_i v = Z_i u`` and its response ``u += Z_i delta``: one row array
    (``rows``, row ``i`` from ``base + i * dofs``) is both. Lane ``l`` holds
    coordinate ``l`` of ``u`` in the register ``velocity``, and of the rows it
    acts on in registers: the current row's (``current``) and its friction
    sibling's (``sibling``), which the schedule loads (:meth:`load`, or the
    pipeline :meth:`prefetch` and :meth:`consume`). A sum is one product per
    lane (zero past ``dofs``) reduced with ``SUM``; a pair's sums reduce in one
    ladder (:meth:`pair_terms`). The builder stores ``A_ii``; the step divides
    by it. ``components``, the diagonal blocks of ``L``, map the velocity into
    factor coordinates and back (:meth:`enter`, :meth:`leave`).
    """

    dofs: int
    components: tuple[FactorComponent, ...] = ()
    lanes: Lanes = Lanes()
    rows: str = "factor_rows"
    base: str = "row_base"
    velocity: str = "factor_velocity"
    current: str = "row_z"
    sibling: str = "sibling_z"
    diagonal: Callable[[str], str] | None = None
    inverse: Callable[[str], str] | None = None
    name = "factor_rows"

    # -- build (response_builds_rows) ---------------------------------------------------------------------------------

    @staticmethod
    def build():
        """The rows' build (:data:`.response_builds_rows.FACTOR_ROWS`): the rows ``Z = L^-1 J^T`` (``A_ii = |Z|^2 +
        cfm``) are the packed rows at their world coordinates (:mod:`.packed_rows`)."""
        from .response_builds_rows import FACTOR_ROWS  # noqa: PLC0415

        return FACTOR_ROWS

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

    def _sums(self, channels, indent: str) -> str:
        """Reduce each channel ``(out, a, b)``'s products in one ladder; a velocity sum (``b`` the velocity) reduces
        its terms' magnitude beside it, ``{out}_terms`` (:meth:`terms`)."""
        lines, names = [], []
        for out, a, b in channels:
            lines.append(f"{indent}{self._product(out, a, b)}")
            names.append(out)
            if b == self.velocity:
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

    def apply(self, delta: str, indent: str) -> str:
        """``u += Z_i delta`` of the current row (lanes past ``dofs`` add zero)."""
        return f"\n{indent}{self.velocity} += {self.current} * {delta};"

    def apply_row(self, row: str, delta: str, indent: str) -> str:
        """``u += Z_s delta`` of the friction sibling ``row`` (its entries in the ``sibling`` register)."""
        return f"\n{indent}{self.velocity} += {self.sibling} * {delta};"

    def _factor_product(self, source: str, out: str, indent: str, *, transposed: bool, declare: bool) -> str:
        """``out = L source`` or ``L^T source`` per component in leaf-first positions (factor coordinate ``i`` on
        lane ``size - 1 - i``), lane ``l`` summing its terms in increasing lane ``k``."""
        lines = [f"{indent}float {out} = 0.0f;"] if declare else []
        for component in self.components:
            size, lane = component.size, component.lane
            row, col = leaf_first(size, lane), leaf_first(size, "k")
            if transposed:
                terms, entry = f"k <= {lane}", component.element(col, row)
            else:
                terms, entry = f"{lane} <= k", component.element(row, col)
            lines += [
                "#pragma unroll",
                f"{indent}for (int k = 0; k < {size}; ++k) {{",
                f"{indent}    const float xk = READ({source}, {component.offset} + k);",
                f"{indent}    if ({lane} >= 0 && {terms} && {lane} < {size})",
                f"{indent}        {out} += {entry} * xk;",
                f"{indent}}}",
            ]
        return "\n".join(lines)

    def transposed(self, source: str, out: str, indent: str, *, declare: bool = True) -> str:
        """``out = L^T P source`` per component (leaf-first positions): lane ``l`` adds the terms of the lanes ``k <=
        l`` in increasing order, each ``x_k`` read from its lane; ``out`` is declared zero (``declare``) or given so."""
        return self._factor_product(source, out, indent, transposed=True, declare=declare)

    def forward(self, source: str, out: str, indent: str, *, declare: bool = True) -> str:
        """``out = L source`` per component (leaf-first positions): lane ``l`` adds the terms of the lanes ``k >= l``
        in increasing order, each ``x_k`` read from its lane; ``out`` is declared zero (``declare``) or given so."""
        return self._factor_product(source, out, indent, transposed=False, declare=declare)

    def lanes_of(self, indent: str) -> list[str]:
        """Lines declaring each component's lane (``LANE - offset``)."""
        return [f"{indent}const int {c.lane} = {self.lanes.lane} - {c.offset};" for c in self.components]

    def enter(self, physical: str, indent: str) -> str:
        """Declare each component's lane and ``velocity = L^T P physical`` (the factor velocity)."""
        return "\n".join([*self.lanes_of(indent), self.transposed(physical, self.velocity, indent)])

    def leave(self, out: str, indent: str) -> str:
        """Declare ``out = P^T L^-T velocity`` (this lane's physical velocity): the components' back substitutions
        run side by side, one step per factor row from the last (lane 0 of the component) and one shuffle a step:
        the lane that holds the row divides by its pivot (a zero pivot gives zero), every later lane of its
        component subtracts its term."""
        steps = max(c.size for c in self.components)
        lines = [f"{indent}float {out} = 0.0f;", f"{indent}{{", f"{indent}    float value = {self.velocity};"]
        lines += [
            "#pragma unroll",
            f"{indent}    for (int s = 0; s < {steps}; ++s) {{",
            f"{indent}        int source = 0;",
        ]
        for c in self.components:
            mine = f"{c.lane} >= 0 && {c.lane} < {c.size} && s < {c.size}"
            pivot = c.element(leaf_first(c.size, "s"), leaf_first(c.size, "s"))
            lines += [
                f"{indent}        if ({mine}) {{",
                f"{indent}            source = {c.offset} + s;",
                f"{indent}            if ({c.lane} == s) {{",
                f"{indent}                const float pivot = {pivot};",
                f"{indent}                {out} = pivot != 0.0f ? value / pivot : 0.0f;",
                f"{indent}            }}",
                f"{indent}        }}",
            ]
        lines.append(f"{indent}        const float x = READ({out}, source);")
        for c in self.components:
            entry = c.element(leaf_first(c.size, "s"), leaf_first(c.size, c.lane))
            lines.append(
                f"{indent}        if ({c.lane} > s && {c.lane} < {c.size} && s < {c.size}) value -= {entry} * x;"
            )
        lines += [f"{indent}    }}", f"{indent}}}"]
        return "\n".join(lines)


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
    apply adds the second tangent's change, then the first's, to each entry, written back once. The builder stores
    ``A_ii = |Z_i|^2 + cfm`` as a value: the step divides.
    """

    support: int
    index: Callable[[str, str], str]
    factor: Callable[[str, str], str]
    lanes: Lanes = Lanes()
    dense: int = 0
    local: str = "s_local_factor"
    velocity: Velocity = Velocity("s_delta")
    diagonal: Callable[[str], str] | None = None
    inverse: Callable[[str], str] | None = None
    name = "sparse_factor"
    fused_pair_apply = True

    # -- build (response_builds_rows) ---------------------------------------------------------------------------------

    @staticmethod
    def build():
        """The rows' build and publication (:data:`.response_builds_rows.SPARSE_FACTOR`): the factor, rows packed
        from their support with ``A_ii = |Z|^2`` (:func:`.response_builds_rows.factor_rows`), and ``v = v_hat + P^T
        L^-T u``."""
        from .response_builds_rows import SPARSE_FACTOR  # noqa: PLC0415

        return SPARSE_FACTOR

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
        coupling: str | None = None,
    ) -> str:
        """``float out = Z_row . delta`` (every lane holds it), the row's entries kept for :meth:`apply`.

        With ``sibling``, a ``paired`` row (a first tangent) also reduces ``sibling_jv = Z_sibling . delta`` and
        ``pair_coupling = Z_row . Z_sibling`` in the same ``SUM`` and keeps the sibling's entries for
        :meth:`apply_pair`; the coupling is summed only under the condition ``coupling`` when one is given.
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
            guard = f"if ({coupling}) " if coupling else ""
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
                f"    {guard}pair_coupling += value * sibling_value;",
                "}",
            ]
            reduce = self.lanes.reduce(out, "sibling_jv", "pair_coupling")
            if coupling and self.lanes.parallel:
                reduce = f"{self.lanes.reduce(out, 'sibling_jv')}\n{guard}{self.lanes.reduce('pair_coupling')}"
        text = "\n".join(indent + line for line in lines) + "\n" + self.entries_loop("\n".join(body), indent)
        return text + "".join(f"\n{indent}{line}" for line in reduce.split("\n") if line)

    def sibling_residual(self, out: str, sibling: str, indent: str, rhs: str) -> str:
        """``float out = Z_s . delta + rhs`` of the pair's second tangent, whose sum :meth:`residual` reduced."""
        return f"{indent}float {out} = sibling_jv;\n{indent}{out} += {rhs};"

    def coupling(self, out: str, row: str, sibling: str, indent: str) -> str:
        """``float out = Z_t . Z_s``: the pair's off-diagonal entry, which :meth:`residual` reduced."""
        return f"{indent}float {out} = pair_coupling;"

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
