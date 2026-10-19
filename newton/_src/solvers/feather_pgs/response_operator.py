# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Author response actions independently of force production and publication.

Vector operands are native expressions containing ``$index``. The caller binds
physical/factor permutations and storage offsets there. A forward action maps
force into factor coordinates; its transpose maps back to physical response.
Cholesky solves retain their dependency order; inverse factors retain products.

These generators allocate no storage and impose no launch boundary. Owners
supply scratch, lane ownership, synchronization between dependent actions,
held-factor validity and output transactions. Operands are authored bindings,
never extracted from another owner's source. No runtime method is selected here.
"""

from dataclasses import dataclass


def _at(vector: str, index: str) -> str:
    return vector.replace("$index", f"({index})")


def _accumulate(coefficient: str, operand: str, check: str = "") -> str:
    if check:
        return f"const float a={coefficient};{check}value+=a*{operand};"
    return f"value+={coefficient}*{operand};"


@dataclass(frozen=True)
class CholeskySolve:
    """Apply L^-1 and L^-T with the owner's zero-diagonal convention.

    One lane owns each complete solve. The output may alias the input; an
    optional post-row statement retains per-owner finite-state reporting.
    A zero diagonal produces zero, as in the retained predictor/fallback law.
    """

    size: int | str
    lower: str

    def forward(self, rhs: str, output: str) -> str:
        """Solve L x = rhs in increasing row order."""
        return f"""for(int row=0;row<{self.size};++row){{
            float value={_at(rhs, "row")};
            for(int k=0;k<row;++k)value-=({self.lower})[row*{self.size}+k]*{_at(output, "k")};
            const float diagonal=({self.lower})[row*{self.size}+row];
            {_at(output, "row")}=diagonal!=0.0f?value/diagonal:0.0f;
        }}"""

    def transpose(self, rhs: str, output: str, *, post_row: str = "") -> str:
        """Solve L.T x = rhs in decreasing row order."""
        return f"""for(int row={self.size}-1;row>=0;--row){{
            float value={_at(rhs, "row")};
            for(int k=row+1;k<{self.size};++k)value-=({self.lower})[k*{self.size}+row]*{_at(output, "k")};
            const float diagonal=({self.lower})[row*{self.size}+row];
            {_at(output, "row")}=diagonal!=0.0f?value/diagonal:0.0f;
            {post_row}
        }}"""


@dataclass(frozen=True)
class LaneCholeskySolve:
    """Apply L^-1 and L^-T with one lane per coordinate and one shuffle per step.

    The same law as :class:`CholeskySolve` (zero diagonal produces zero), with
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
            float value=active?s[{rhs_base}+i]:0.0f;
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
            if(active)s[{output_base}+i]=z;
        }}"""


@dataclass(frozen=True)
class DenseLowerInverse:
    """Apply an existing row-major lower inverse factor without refactoring it."""

    size: int | str
    factor: str

    def forward(self, row: str, rhs: str, *, check: str = "") -> str:
        """Compute one forward row; check may validate coefficient ``a``."""
        operation = _accumulate(f"({self.factor})[({row})*{self.size}+k]", _at(rhs, "k"), check)
        return f"float value=0.0f;for(int k=0;k<=({row});++k){{{operation}}}"

    def transpose(self, col: str, rhs: str) -> str:
        """Compute one transpose column in increasing contributing-row order."""
        operation = _accumulate(f"({self.factor})[k*{self.size}+({col})]", _at(rhs, "k"))
        return f"float value=0.0f;for(int k=({col});k<{self.size};++k){{{operation}}}"


@dataclass(frozen=True)
class BlockArrowInverse:
    """Apply independent lower branch blocks coupled into a lower central block.

    Coordinates place the central block before all equal-sized branch blocks.
    Packed storage places branch triangles, the central triangle, then dense
    central-by-branch rectangles. This describes storage, not topology admission.
    """

    central: int
    branch: int
    branches: int
    factor: str

    def __post_init__(self):
        if self.central < 1 or self.branch < 1 or self.branches < 1:
            raise ValueError("Block response dimensions must be positive")

    @property
    def branch_triangle(self):
        return self.branch * (self.branch + 1) // 2

    @property
    def central_offset(self):
        return self.branches * self.branch_triangle

    @property
    def coupling_offset(self):
        return self.central_offset + self.central * (self.central + 1) // 2

    def forward(self, index: str, rhs: str, *, check: str = "") -> str:
        """Apply each branch locally and include every branch in the central action."""
        n, b, count = self.central, self.branch, self.branches
        central = _accumulate(
            f"({self.factor})[{self.central_offset}+({index})*(({index})+1)/2+k]", _at(rhs, "k"), check
        )
        coupling = _accumulate(
            f"({self.factor})[{self.coupling_offset}+branch_id*{n * b}+({index})*{b}+k]",
            _at(rhs, f"{n}+branch_id*{b}+k"),
            check,
        )
        branch = _accumulate(
            f"({self.factor})[branch_id*{self.branch_triangle}+branch_row*(branch_row+1)/2+k]",
            _at(rhs, f"{n}+branch_id*{b}+k"),
            check,
        )
        return f"""float value=0.0f;
            if(({index})<{n}){{
                for(int k=0;k<=({index});++k){{{central}}}
                for(int branch_id=0;branch_id<{count};++branch_id)for(int k=0;k<{b};++k){{{coupling}}}
            }}else{{
                const int branch_id=(({index})-{n})/{b},branch_row=(({index})-{n})%{b};
                for(int k=0;k<=branch_row;++k){{{branch}}}
            }}"""

    def transpose(self, index: str, rhs: str) -> str:
        """Apply the central transpose, then each branch and its central coupling."""
        n, b = self.central, self.branch
        central = _accumulate(f"({self.factor})[{self.central_offset}+k*(k+1)/2+({index})]", _at(rhs, "k"))
        branch = _accumulate(
            f"({self.factor})[branch_id*{self.branch_triangle}+k*(k+1)/2+branch_col]", _at(rhs, f"{n}+branch_id*{b}+k")
        )
        coupling = _accumulate(
            f"({self.factor})[{self.coupling_offset}+branch_id*{n * b}+k*{b}+branch_col]", _at(rhs, "k")
        )
        return f"""float value=0.0f;
            if(({index})<{n}){{for(int k=({index});k<{n};++k){{{central}}}}}
            else{{
                const int branch_id=(({index})-{n})/{b},branch_col=(({index})-{n})%{b};
                for(int k=branch_col;k<{b};++k){{{branch}}}
                for(int k=0;k<{n};++k){{{coupling}}}
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

    def forward(self, row: str, rhs: str) -> str:
        """Compute one sparse forward row, preserving the support's traversal order."""
        operation = _accumulate(f"({self.factor})[entry]", _at(rhs, "col"))
        return "float value=0.0f;" + self.support.row(row, operation)

    def transpose(self, col: str, rhs: str) -> str:
        """Compute one sparse transpose column from completed forward coordinates."""
        operation = _accumulate(f"({self.factor})[entry]", _at(rhs, "row"))
        return "float value=0.0f;" + self.support.column(col, operation)
