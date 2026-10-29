# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build the dense response operators' data: the factor of ``H``, ``Y = H^-1 J^T`` and its diagonal, the assembled
Delassus ``C``, and the velocity the impulses publish.

A dense response is built per articulation size group in stages (:data:`STAGES`), each one implementation per
thread mapping, keyed ``(stage, mapping)`` in :data:`DENSE_JY_BUILDS` (the response by size group and the world layout) and
:data:`ASSEMBLED_BUILDS` (the data of an :class:`.Assembled` sweep). Each stage's arithmetic comes from one small
dense kit:

* :class:`.CholeskyFactor`: the leaf-first factor ``P (H + diag(R)) P^T = L L^T`` (``P`` reverses an
  articulation's coordinates, :func:`.leaf_first`) by columns on one thread (``factor`` over articulations, and lane
  0 of the warp-mapped factor that assembles ``H`` from the held tree's geometry);
* :class:`.CholeskySolve`: the triangular solves (physical vectors in, physical vectors out, the reversal bound in
  the kit), with the zero-diagonal conventions as named options: a zero
  diagonal gives zero (Z0: the thread-, lane- and queue-mapped responses, the register response of the six-DOF
  group, the paired response), or the step multiplies by an unguarded staged reciprocal (Zr-inf: the warp-mapped
  response); a diagonal factor divides twice, unguarded (Zdiv, :func:`diagonal_solve`);
* :func:`dot`: a sum of products in ascending order from zero (every diagonal, every Delassus entry, the apply);
* the tile primitives (``tile_cholesky``, ``tile_lower_solve``, ``tile_upper_solve``, ``tile_matmul``) for the
  tiled mappings: their rounding is the primitives' (MathDx's where it is on), so nothing
  re-derives it.

Where two mappings of a stage differ in the order they round in, each keeps its own order (the lane layout, a
register or a global scratch, a staged reciprocal); where two builds of a stage differ only by accident, each keeps
its order and the difference is listed for their unification (see the stage's builder).

A build is a :class:`Build`: its kernel, the stage and mapping it implements, and its grid. The solver's launch sites
call the kernels and kernel factories directly.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable

import warp as wp

from . import kernel_template
from .device_preludes import lane_group, substitute
from .kernel_template import THREADS, KernelTemplate, python_kernel
from .kernels import PGS_LOCAL_SOLVE_OWNER_GENERAL
from .response_operator import CholeskyFactor, CholeskySolve, Reversed, Vector, leaf_first
from .staging import Buffer, Staging

__all__ = [
    "ASSEMBLED_BUILDS",
    "DENSE_JY_BUILDS",
    "STAGES",
    "Build",
    "diagonal_solve",
    "dot",
    "whiten_six",
]

STAGES = ("factor", "response", "diagonal", "cfm", "delassus", "apply")
"""What a build stage writes: the leaf-first factor ``L`` of ``P (H + diag(R)) P^T``; the response ``Y = H^-1 J^T`` (by
size group, optionally in the world layout, optionally with its diagonal ``J_i Y_i``); the world diagonal ``A_ii = J_i
Y_i``; its ``cfm``; the Delassus ``C = J Y^T`` with its diagonal; the velocity ``v_hat + Y^T lambda``."""


# ---------------------------------------------------------------------------------------------------------------------
# The kit's text: index maps, sums, the diagonal factor's solve
# ---------------------------------------------------------------------------------------------------------------------


def _indent(text: str, pad: str) -> str:
    """Prefix every non-empty line of ``text``."""
    return "\n".join(pad + line if line.strip() else line for line in text.split("\n"))


def _element(array: str, *indices: str) -> str:
    """``array[indices]`` of a contiguous array, flattened through the array's own shape (Warp's indexing)."""
    offset = indices[0]
    if len(indices) > 1:
        offset = f"(size_t)({offset})"
        for axis, index in enumerate(indices[1:], start=1):
            offset = f"({offset}) * {array}.shape[{axis}] + ({index})"
    return f"{array}.data[{offset}]"


def dot(out: str, count: str, a: Vector, b: Vector, *, index: str = "k", unroll: bool = False) -> str:
    """``float out = sum a(k) b(k)`` over ``k < count``, the products added in ascending order from zero."""
    loop = f"for (int {index} = 0; {index} < {count}; ++{index}) {out} += {a(index)} * {b(index)};"
    return f"float {out} = 0.0f;\n" + ("#pragma unroll\n" if unroll else "") + loop


def diagonal_solve(value: str, factor: str) -> str:
    """``H^-1`` of a diagonal ``H = L L^T`` at one coordinate: ``(value / l) / l``, unguarded (Zdiv)."""
    return f"({value} / {factor}) / {factor}"


def _skips(rows: str, world: str) -> str:
    """The parallel tiers' worlds, whose sweeps form their response in the kernel: at most ``skip_rows_le`` dense rows
    and no free-body rows (``skip_rows_le`` 0: none)."""
    return f"skip_rows_le > 0 && {rows} <= skip_rows_le && mf_constraint_count.data[{world}] == 0"


def _respond_in_place(L: str, J: str, Y: str, group: str, row: str, size: str) -> str:
    """``Y_row = H^-1 J_row^T`` of a size group's row in global memory: ``L z = P J^T`` into ``Y``, then ``L^T P y =
    z`` in place (the kit's leaf-first positions), each step ascending in its terms from the right-hand side, a zero
    diagonal giving zero (Z0).

    The row is bound first: the solves' loop variable is ``row``.
    """
    solve = CholeskySolve(size, lambda r, c: _element(L, group, r, c))
    # The row's last coordinates, read backwards: factor coordinate k at physical position size - 1 - k.
    jacobian = Reversed(lambda k: f"response_J[-{k}]")
    response = Reversed(lambda k: f"response_Y[-{k}]")
    return "\n".join(
        (
            f"const int response_row = {row};",
            f"const float* response_J = &{_element(J, group, 'response_row', f'{size} - 1')};",
            f"float* response_Y = &{_element(Y, group, 'response_row', f'{size} - 1')};",
            solve.whiten(jacobian, response),
            solve.leave(response, response),
        )
    )


def _publish_row(
    J_world: str, Y_world: str | None, J: str, Y: str, group: str, world: str, row: str, offset: str, size: str
) -> str:
    """Copy a size group's row to the world layout from coordinate ``offset``: ``J`` (and ``Y``, with ``Y_world``)."""
    lines = [f"for (int i = 0; i < {size}; ++i) {{"]
    lines.append(f"    {_element(J_world, world, row, f'{offset} + i')} = {_element(J, group, row, 'i')};")
    if Y_world is not None:
        lines.append(f"    {_element(Y_world, world, row, f'{offset} + i')} = {_element(Y, group, row, 'i')};")
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------------
# Launch descriptors
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Build:
    """One launch of a dense response build: its kernel and its grid.

    Args:
        stage: What it writes (:data:`STAGES`).
        mapping: How its threads cover the stage's items: its key in :data:`DENSE_JY_BUILDS` or
            :data:`ASSEMBLED_BUILDS`.
        kernel: The Warp kernel.
        grid: The launch's dimensions, by item: a thread grid launches their product as threads, a tiled launch
            them as blocks.
        block_dim: Threads per block (0: Warp's default); a kernel that stages per lane group needs its own.
        tiled: A tiled launch (``wp.launch_tiled``) rather than a thread grid.
    """

    stage: str
    mapping: str
    kernel: wp.Kernel
    grid: tuple[str, ...]
    block_dim: int = 0
    tiled: bool = False


def _kernel(name: str, params, body: str, *, dual: bool = True, preludes=(), dispatch=THREADS, **options) -> wp.Kernel:
    return KernelTemplate(
        name, params, body, dispatch=dispatch, preludes=preludes, dual=dual, native_name=f"{name}_native", **options
    ).kernel()


# ---------------------------------------------------------------------------------------------------------------------
# Stage "factor": H + diag(R) = L L^T
# ---------------------------------------------------------------------------------------------------------------------

FACTOR_ABI = kernel_template.params(
    ("H_group", wp.array3d[float]),
    ("R_group", wp.array2d[float]),
    ("group_to_art", wp.array[int]),
    ("mass_update_mask", wp.array[int]),
    ("n_dofs", int),
    ("L_group", wp.array3d[float]),
)
"""A size group's factor from its mass matrices (``H_group``) and their regularization (``R_group``), for the
articulations whose mass changed (``mass_update_mask``)."""


def _group_factor() -> CholeskyFactor:
    """The kit's leaf-first factor of articulation ``group``'s ``H + diag(R)``: ``A = P (H + diag(R)) P^T`` with the
    coordinates reversed (:func:`.leaf_first`), ``A_jj = H_jj + R_j``, ``H`` with its stiffness folded in already, so
    the diagonal rounds as ``(S F + K) + R``."""

    def H(row, col):
        return _element("H_group", "group", leaf_first("n_dofs", row), leaf_first("n_dofs", col))

    def L(row, col):
        return _element("L_group", "group", row, col)

    def diagonal(col):
        return f"{H(col, col)} + {_element('R_group', 'group', leaf_first('n_dofs', col))}"

    return CholeskyFactor("n_dofs", L, source=H, diagonal=diagonal)


@functools.cache
def factor_articulations_kernel() -> wp.Kernel:
    """The factor, one thread per articulation (``dim = articulations``), with the kit's column Cholesky in global
    memory."""
    body = (
        "    const int group = tid;\n"
        "    if (mass_update_mask.data[group_to_art.data[group]] == 0) return;\n"
        f"{_indent(_group_factor().factor(), '    ')}\n"
    )
    return _kernel("dense_factor_articulations", FACTOR_ABI, body)


@functools.cache
def factor_coordinates_kernel() -> wp.Kernel:
    """The factor of a structurally diagonal ``H``, one thread per articulation coordinate (``dim = articulations *
    n_dofs``): the kit's pivot, ``L_jj = sqrt(H_jj + R_j)``."""
    body = (
        "    const int group = tid / n_dofs;\n"
        "    const int dof = tid - group * n_dofs;\n"
        "    if (mass_update_mask.data[group_to_art.data[group]] != 0)\n"
        f"        {_group_factor().pivot('dof')}\n"
    )
    return _kernel("dense_factor_coordinates", FACTOR_ABI, body)


# A warp per articulation reads its leaf-first H (P H P^T, the coordinates reversed) from the held tree's geometry into
# shared memory, lane 0 factors it with the kit, and the lanes store the lower triangle. The diagonal rounds as
# (S F + R) + K.
_WARP_FACTOR = """
    const int group = tid >> 5;
    if (group >= n_arts) return;

    const int articulation = group_to_art.data[group];
    if (mass_update_mask.data[articulation] == 0) return;

$STAGING
    const int dof_start = articulation_dof_start.data[articulation];
    for (int schedule_index = LANE; schedule_index < $LOWER; schedule_index += STRIDE) {
        const int element = lower_schedule.data[schedule_index];
        const int row = element / $DOFS;
        const int col = element - row * $DOFS;
        const int geometry_entry = geometry_lookup.data[element];
        float value = geometry_entry < 0 ? 0.0f : geometric.data[$GEOMETRY_ROW * $GEOMETRY_STRIDE + geometry_entry];
        if (row == col) {
            const int physical = $DOFS - 1 - row;
            value += R_group.data[group * $DOFS + physical];
            const int drive_row = drive_row_by_dof.data[dof_start + physical];
            if (drive_row >= 0) {
                const float K = row_K.data[drive_row];
                if (K > 0.0f) value += K;
            }
        }
        factor[element] = value;
    }
    SYNC();

    if (LANE == 0) {
$FACTOR
    }
    SYNC();

    if (LANE < $DOFS) {
        const int factor_base = group * $ELEMENTS;
#pragma unroll
        for (int col = 0; col <= LANE; ++col)
            L_group.data[factor_base + LANE * $DOFS + col] = factor[LANE * $DOFS + col];
    }
#endif
"""

_WARP_FACTOR_INPUTS = (
    ("n_arts", int),
    ("articulation_dof_start", wp.array[int]),
    ("mass_update_mask", wp.array[int]),
    ("group_to_art", wp.array[int]),
    ("R_group", wp.array2d[float]),
    ("lower_schedule", wp.array[int]),
    ("drive_row_by_dof", wp.array[int]),
    ("row_K", wp.array[float]),
    ("geometric", wp.array2d[float]),
    ("geometry_lookup", wp.array[int]),
)


def leaf_index(physical_index) -> tuple[int, ...]:
    """A symmetric physical ``(row, col)`` index grid (a square array) as the leaf-first factor's, flattened: factor
    ``(row, col)`` reads physical ``(n - 1 - row, n - 1 - col)``."""
    return tuple(map(int, physical_index[::-1, ::-1].ravel()))


@functools.cache
def factor_warps_kernel(
    size: int, geometric_index, *, warps_per_block: int = 4, geometric_stride: int = 0, geometric_group: bool = False
) -> wp.Kernel:
    """The held tree's leaf-first factor, one warp per articulation (``dim = articulations * 32``): each lower entry
    of ``P H P^T`` read from the held geometry through ``geometric_index`` (the factor ``(row, col)`` entry's geometry
    slot, :func:`leaf_index`; ``-1`` where structurally zero; rows of ``geometric_stride``, groups mapped to geometry
    rows with ``geometric_group``), then ``R`` and the augmented drives' ``K`` of the physical coordinate on the
    diagonal; lane 0 factors with the kit (:class:`.CholeskyFactor`, unrolled), the lanes store the lower triangle.
    """
    size, warps_per_block = int(size), int(warps_per_block)
    if not 1 <= size <= 32 or not 1 <= warps_per_block <= 32:
        raise ValueError("Small Cholesky requires 1..32 DOFs and 1..32 warps per block")
    if len(geometric_index) != size * size or min(geometric_index) < -1:
        raise ValueError("Geometric factor requires a complete physical matrix index")
    if geometric_stride and geometric_stride <= max(geometric_index):
        raise ValueError("Geometry stride must cover every indexed entry")
    elements = size * size
    inputs = list(_WARP_FACTOR_INPUTS)
    row = "group"
    if geometric_group:
        inputs.append(("geometry_group", wp.array[int]))
        row = "geometry_group.data[group]"
    inputs.append(("L_group", wp.array3d[float]))
    staging = Staging((Buffer("factor", "float", elements),), groups=warps_per_block, layout="arrays")
    factor = CholeskyFactor(size, lambda row, col: f"factor[{row} * {size} + {col}]", unroll=True)
    snippet = substitute(
        _WARP_FACTOR,
        DOFS=size,
        ELEMENTS=elements,
        LOWER=size * (size + 1) // 2,
        STAGING=staging.declarations(),
        GEOMETRY_ROW=row,
        GEOMETRY_STRIDE=geometric_stride or max(geometric_index) + 1,
        FACTOR=_indent(factor.factor(), "        "),
    )
    return KernelTemplate(
        f"geometric_dense_factor_warps{warps_per_block}_{size}",
        tuple(inputs),
        snippet,
        THREADS,
        preludes=(lane_group(32),),
        native_name="factor_native",
    ).kernel()


@functools.cache
def factor_tiles_kernel(n_dofs: int, tile_threads: int = 64) -> wp.Kernel:
    """The leaf-first factor, one tile per articulation (``launch_tiled(dim=[articulations], block_dim=tile_threads)``):
    ``tile_cholesky(tile_diag_add(P H P^T, P R))``, the coordinates reversed by indexed loads, the primitive's
    rounding."""
    dofs = wp.constant(int(n_dofs))
    last = wp.constant(int(n_dofs) - 1)

    def cholesky_tiled_template(
        H_group: wp.array3d[float],
        R_group: wp.array2d[float],
        group_to_art: wp.array[int],
        mass_update_mask: wp.array[int],
        L_group: wp.array3d[float],
    ):
        idx = wp.tid()
        art = group_to_art[idx]
        if mass_update_mask[art] == 0:
            return
        reverse = wp.tile_arange(last, -1, -1, dtype=int)
        H_tile = wp.tile_load_indexed(H_group[idx], indices=reverse, shape=(dofs, dofs), axis=0)[:, reverse]
        armature = wp.tile_load_indexed(R_group[idx], indices=reverse, shape=(dofs,), axis=0)
        H_tile = wp.tile_diag_add(H_tile, armature)
        L_tile = wp.tile_cholesky(H_tile)
        wp.tile_store(L_group[idx], L_tile)

    return python_kernel(cholesky_tiled_template, f"dense_factor_tiles_{int(n_dofs)}_bd{int(tile_threads)}")


# ---------------------------------------------------------------------------------------------------------------------
# Stage "response": Y = H^-1 J^T, optionally in the world layout and with its diagonal J_i Y_i
# ---------------------------------------------------------------------------------------------------------------------

ROWS_ABI = kernel_template.params(
    ("L_group", wp.array3d[float]),
    ("J_group", wp.array3d[float]),
    ("group_to_art", wp.array[int]),
    ("art_to_world", wp.array[int]),
    ("articulation_world_dof_offset", wp.array[int]),
    ("world_constraint_count", wp.array[int]),
    ("n_dofs", int),
    ("max_constraints", int),
    ("n_arts", int),
    ("write_world", int),
    ("Y_group", wp.array3d[float]),
    ("J_world", wp.array3d[float]),
    ("Y_world", wp.array3d[float]),
)
"""The response of a size group's rows: its factors and Jacobian rows, the world layout's coordinate offsets, and
the switch that also writes the rows to the world layout (``J_world``, ``Y_world``)."""


@functools.cache
def response_rows_kernel() -> wp.Kernel:
    """The response, one thread per articulation row (``dim = articulations * max_constraints``), solved in place in
    ``Y`` (:func:`_respond_in_place`, Z0)."""
    publish = _publish_row("J_world", "Y_world", "J_group", "Y_group", "idx", "world", "c", "dof_offset", "n_dofs")
    body = f"""    const int c = tid % max_constraints;
    const int idx = tid / max_constraints;
    if (idx >= n_arts) return;
    const int art = group_to_art.data[idx];
    const int world = art_to_world.data[art];
    if (c >= world_constraint_count.data[world]) return;
    {{
{_indent(_respond_in_place("L_group", "J_group", "Y_group", "idx", "c", "n_dofs"), "        ")}
    }}
    if (write_world != 0) {{
        const int dof_offset = articulation_world_dof_offset.data[art];
{_indent(publish, "        ")}
    }}
"""
    return _kernel("dense_response_rows", ROWS_ABI, body)


@functools.cache
def response_lanes_kernel() -> wp.Kernel:
    """The response of the worlds the general solver owns, 32 threads per articulation, rows lane-strided (``dim =
    articulations * 32``), each solved in place (Z0). In a world a local owner solves, only the impact rows
    (restitution above zero) copy their Jacobian to the world layout, for the restitution target pass."""
    impact = _publish_row(
        "J_world", None, "J_group", "Y_group", "group_index", "world", "constraint", "dof_offset", "n_dofs"
    )
    publish = _publish_row(
        "J_world", "Y_world", "J_group", "Y_group", "group_index", "world", "constraint", "dof_offset", "n_dofs"
    )
    restitution = _element("world_row_restitution", "world", "constraint")
    respond = _respond_in_place("L_group", "J_group", "Y_group", "group_index", "constraint", "n_dofs")
    body = f"""    const int group_index = tid / 32;
    const int lane = tid % 32;
    if (group_index >= n_arts) return;
    const int art = group_to_art.data[group_index];
    const int world = art_to_world.data[art];
    const int constraint_count = world_constraint_count.data[world];
    if (local_solve_owner.data[world] != {PGS_LOCAL_SOLVE_OWNER_GENERAL}) {{
        if (write_world != 0) {{
            const int dof_offset = articulation_world_dof_offset.data[art];
            for (int constraint = lane; constraint < constraint_count; constraint += 32) {{
                if ({restitution} > 0.0f) {{
{_indent(impact, "                    ")}
                }}
            }}
        }}
        return;
    }}
    for (int constraint = lane; constraint < constraint_count; constraint += 32) {{
        {{
{_indent(respond, "            ")}
        }}
        if (write_world != 0) {{
            const int dof_offset = articulation_world_dof_offset.data[art];
{_indent(publish, "            ")}
        }}
    }}
"""
    params = kernel_template.params(
        *ROWS_ABI[:6],
        ("local_solve_owner", wp.array[int]),
        ("world_row_restitution", wp.array2d[float]),
        *ROWS_ABI[6:],
    )
    return _kernel("dense_response_lanes", params, body)


_Vec6 = wp.types.vector(length=6, dtype=wp.float32)


def _whiten_six_snippet() -> str:
    solve = CholeskySolve(6, lambda row, col: _element("factor", "group", row, col))

    def jacobian_at(i):
        return f"jacobian[{i}]"

    def whitened_at(i):
        return f"whitened[{i}]"

    return f"""
    wp::vec_t<6, float> whitened(0.0f);
{_indent(solve.whiten(jacobian_at, whitened_at), "    ")}
    return whitened;
"""


@wp.func_native(_whiten_six_snippet())
def whiten_six(factor: wp.array3d[float], group: int, jacobian: _Vec6) -> _Vec6:
    """One row of the six-coordinate group whitened in registers, ``Z = L^-1 P J^T`` by the kit's forward solve (Z0):
    the dense part of a compact boundary row (:func:`.response_builds_rows.sparse_pair_dense`)."""


@functools.cache
def response_diagonal_kernel() -> wp.Kernel:
    """The response of a structurally diagonal ``H``, one thread per articulation row (``dim = articulations *
    max_constraints``): ``y = (j / l) / l`` per coordinate, ``l`` its leaf-first pivot (:func:`diagonal_solve`,
    Zdiv), written to the group
    (``write_group``) and the world layout (``write_world``), with the row's diagonal ``sum j y`` in coordinate
    order (``compute_diag``)."""
    J = _element("J_group", "group", "constraint", "dof")
    body = f"""    const int constraint = tid % max_constraints;
    const int group = tid / max_constraints;
    if (group >= n_arts) return;
    const int art = group_to_art.data[group];
    const int world = art_to_world.data[art];
    if (constraint >= world_constraint_count.data[world]) return;
    int dof_offset = 0;
    if (write_world != 0) dof_offset = articulation_world_dof_offset.data[art];
    float diagonal_sum = 0.0f;
    for (int dof = 0; dof < n_dofs; ++dof) {{
        const float jacobian = {J};
        const float factor = {_element("L_group", "group", leaf_first("n_dofs", "dof"), leaf_first("n_dofs", "dof"))};
        const float response = {diagonal_solve("jacobian", "factor")};
        if (write_group != 0) {_element("Y_group", "group", "constraint", "dof")} = response;
        if (write_world != 0) {{
            {_element("J_world", "world", "constraint", "dof_offset + dof")} = jacobian;
            {_element("Y_world", "world", "constraint", "dof_offset + dof")} = response;
        }}
        if (compute_diag != 0) diagonal_sum += jacobian * response;
    }}
    if (compute_diag != 0) {_element("diag_group", "group", "constraint")} = diagonal_sum;
"""
    params = kernel_template.params(
        *ROWS_ABI[:9],
        ("write_group", int),
        ("write_world", int),
        ("compute_diag", int),
        *ROWS_ABI[10:],
        ("diag_group", wp.array2d[float]),
    )
    return _kernel("dense_response_diagonal", params, body)


@functools.cache
def response_warps_kernel(
    n_dofs: int, max_constraints: int, *, write_world: bool, write_group: bool, compute_diag: bool
) -> wp.Kernel:
    """The response, one warp per articulation and one lane per row (``launch_tiled(dim=[ceil(articulations /
    warps_per_block)], block_dim=32 * warps_per_block)``): the factor staged in shared memory with its reciprocal
    diagonal, each row solved in registers by the kit's solves (unrolled, Zr-inf: the steps multiply by the
    unguarded ``1 / L_ii``), every row of a touched 32-row chunk written (inactive rows carry zero Jacobians); the
    group copy, the diagonal ``sum j y`` and the world layout as asked. The parallel tiers' worlds are skipped.
    The staging holds four warps' factors: ``warps_per_block`` must be 4. CUDA only."""
    N, M = int(n_dofs), int(max_constraints)
    solve = CholeskySolve(N, "L", inverse="Dinv", unroll=True)
    staging = Staging((Buffer("L", "float", N * N), Buffer("Dinv", "float", N)), groups=4, layout="arrays")

    def j(i):
        return f"j[{i}]"

    def y(i):
        return f"y[{i}]"

    group_store = (
        "\n#pragma unroll\nfor (int d = 0; d < N_; ++d) Y_group.data[gbase + d] = y[d];" if write_group else ""
    )
    diag_store = (
        f"\n{dot('acc', 'N_', j, y, index='d', unroll=True)}\ndiag_group.data[(size_t)idx * M_ + row] = acc;"
        if compute_diag
        else ""
    )
    world_store = (
        """
const int dof_offset = articulation_world_dof_offset.data[art];
const size_t wbase = (size_t)world * world_rows * world_dofs + (size_t)row * world_dofs + dof_offset;
#pragma unroll
for (int d = 0; d < N_; ++d) {
    J_world.data[wbase + d] = j[d];
    Y_world.data[wbase + d] = y[d];
}"""
        if write_world
        else ""
    )
    body = f"""    constexpr int N_ = {N};
    constexpr int M_ = {M};
    const int idx = block * warps_per_block + SUB;
{staging.declarations()}
    if (idx >= n_arts) return;
    const int art = group_to_art.data[idx];
    const int world = art_to_world.data[art];
    const int n_rows = world_constraint_count.data[world];
    if (n_rows <= 0) return;
    if ({_skips("n_rows", "world")}) return;
{_indent(solve.stage(lambda e: f"L_group.data[(size_t)idx * N_ * N_ + {e}]", "LANE", "STRIDE"), "    ")}
    SYNC();
    {solve.stage_inverse("LANE")}
    SYNC();
    const int world_rows = Y_world.shape[1];
    const int world_dofs = Y_world.shape[2];
    const int n_pad = ((n_rows + 31) / 32) * 32;
    for (int row = LANE; row < n_pad && row < M_; row += STRIDE) {{
        const size_t gbase = ((size_t)idx * M_ + row) * N_;
        float j[N_], y[N_];
        #pragma unroll
        for (int d = 0; d < N_; ++d) {{ j[d] = J_group.data[gbase + d]; y[d] = 0.0f; }}
{_indent(solve.whiten(j, y), "        ")}
{_indent(solve.leave(y, y), "        ")}{_indent(group_store + diag_store + world_store, "        ")}
    }}
#endif
"""
    suffix = (
        ("_world" if write_world else "") + ("_nogroup" if not write_group else "") + ("_diag" if compute_diag else "")
    )
    params = kernel_template.params(
        ("warps_per_block", int),
        ("n_arts", int),
        *ROWS_ABI[:6],
        *ROWS_ABI[10:],
        ("diag_group", wp.array2d[float]),
        ("mf_constraint_count", wp.array[int]),
        ("skip_rows_le", int),
    )
    return KernelTemplate(
        f"dense_response_warps_{N}_{M}{suffix}",
        params,
        body,
        preludes=(lane_group(32),),
        native_name="grouped_hinv_jt_native",
    ).kernel()


@functools.cache
def response_tiles_kernel(
    n_dofs: int,
    max_constraints: int,
    tile_threads: int = 64,
    constraint_chunk_size: int | None = None,
    write_world: bool = False,
    write_group: bool = True,
    compute_diag: bool = False,
) -> wp.Kernel:
    """The response, one tile per articulation and chunk of rows (``launch_tiled(dim=[articulations, chunks],
    block_dim=tile_threads)``): ``tile_lower_solve(L, P J^T)`` then ``Y^T = P^T tile_upper_solve(L^T, Z)`` (the
    leaf-first factor, the coordinates reversed by indexing), the primitives' rounding; the group copy, the world layout and, with ``compute_diag``, the diagonal ``tile_sum(J * Y, axis=1)``
    (the kernel then takes ``diag_group``, which a plain kernel does not declare at all: Warp keeps an unused
    output's work).
    """
    TILE_DOF_LOCAL = wp.constant(int(n_dofs))
    TILE_DOF_LAST = wp.constant(int(n_dofs) - 1)
    chunk_size = max_constraints if constraint_chunk_size is None else int(constraint_chunk_size)
    if chunk_size <= 0 or chunk_size > max_constraints:
        raise ValueError("constraint_chunk_size must be in [1, max_constraints]")
    TILE_CONSTRAINTS_LOCAL = wp.constant(chunk_size)
    BOUNDS_CHECK = max_constraints % chunk_size != 0
    WRITE_WORLD = wp.constant(1 if write_world else 0)
    WRITE_GROUP = wp.constant(1 if write_group else 0)
    suffix = ("_world" if write_world else "") + ("_nogroup" if not write_group else "")
    name = f"dense_response_tiles_{n_dofs}_{max_constraints}_c{chunk_size}_bd{tile_threads}{suffix}"

    if compute_diag:
        COMPUTE_DIAG = wp.constant(1)

        def hinv_jt_tiled_template(
            L_group: wp.array3d[float],
            J_group: wp.array3d[float],
            group_to_art: wp.array[int],
            art_to_world: wp.array[int],
            articulation_world_dof_offset: wp.array[int],
            world_constraint_count: wp.array[int],
            Y_group: wp.array3d[float],
            J_world: wp.array3d[float],
            Y_world: wp.array3d[float],
            diag_group: wp.array2d[float],
        ):
            idx, chunk = wp.tid()
            art = group_to_art[idx]
            world = art_to_world[art]
            n_constraints = world_constraint_count[world]
            row_start = chunk * TILE_CONSTRAINTS_LOCAL

            if row_start >= n_constraints:
                return

            L_tile = wp.tile_load(L_group[idx], shape=(TILE_DOF_LOCAL, TILE_DOF_LOCAL), bounds_check=False)
            J_tile = wp.tile_load(
                J_group[idx],
                shape=(TILE_CONSTRAINTS_LOCAL, TILE_DOF_LOCAL),
                offset=(row_start, 0),
                bounds_check=BOUNDS_CHECK,
            )
            reverse = wp.tile_arange(TILE_DOF_LAST, -1, -1, dtype=int)
            Jt_tile = wp.tile_transpose(J_tile[:, reverse])
            Z_tile = wp.tile_lower_solve(L_tile, Jt_tile)
            Lt_tile = wp.tile_transpose(L_tile)
            X_tile = wp.tile_upper_solve(Lt_tile, Z_tile)
            Y_out_tile = wp.tile_transpose(X_tile[reverse, :])
            if WRITE_GROUP != 0:
                wp.tile_store(Y_group[idx], Y_out_tile, offset=(row_start, 0), bounds_check=BOUNDS_CHECK)

            if COMPUTE_DIAG != 0:
                diag_tile = wp.tile_sum(wp.tile_map(wp.mul, J_tile, Y_out_tile), axis=1)
                wp.tile_store(diag_group[idx], diag_tile, offset=row_start, bounds_check=BOUNDS_CHECK)

            if WRITE_WORLD != 0:
                dof_offset = articulation_world_dof_offset[art]
                wp.tile_store(J_world[world], J_tile, offset=(row_start, dof_offset), bounds_check=BOUNDS_CHECK)
                wp.tile_store(Y_world[world], Y_out_tile, offset=(row_start, dof_offset), bounds_check=BOUNDS_CHECK)

        return python_kernel(hinv_jt_tiled_template, name + "_diag")

    def hinv_jt_plain_template(
        L_group: wp.array3d[float],
        J_group: wp.array3d[float],
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        articulation_world_dof_offset: wp.array[int],
        world_constraint_count: wp.array[int],
        Y_group: wp.array3d[float],
        J_world: wp.array3d[float],
        Y_world: wp.array3d[float],
    ):
        idx, chunk = wp.tid()
        art = group_to_art[idx]
        world = art_to_world[art]
        n_constraints = world_constraint_count[world]
        row_start = chunk * TILE_CONSTRAINTS_LOCAL

        if row_start >= n_constraints:
            return

        L_tile = wp.tile_load(L_group[idx], shape=(TILE_DOF_LOCAL, TILE_DOF_LOCAL), bounds_check=False)
        J_tile = wp.tile_load(
            J_group[idx],
            shape=(TILE_CONSTRAINTS_LOCAL, TILE_DOF_LOCAL),
            offset=(row_start, 0),
            bounds_check=BOUNDS_CHECK,
        )
        reverse = wp.tile_arange(TILE_DOF_LAST, -1, -1, dtype=int)
        Jt_tile = wp.tile_transpose(J_tile[:, reverse])
        Z_tile = wp.tile_lower_solve(L_tile, Jt_tile)
        Lt_tile = wp.tile_transpose(L_tile)
        X_tile = wp.tile_upper_solve(Lt_tile, Z_tile)
        Y_out_tile = wp.tile_transpose(X_tile[reverse, :])

        if WRITE_GROUP != 0:
            wp.tile_store(Y_group[idx], Y_out_tile, offset=(row_start, 0), bounds_check=BOUNDS_CHECK)

        if WRITE_WORLD != 0:
            dof_offset = articulation_world_dof_offset[art]
            wp.tile_store(J_world[world], J_tile, offset=(row_start, dof_offset), bounds_check=BOUNDS_CHECK)
            wp.tile_store(Y_world[world], Y_out_tile, offset=(row_start, dof_offset), bounds_check=BOUNDS_CHECK)

    return python_kernel(hinv_jt_plain_template, name)


# ---------------------------------------------------------------------------------------------------------------------
# Stages "diagonal" and "cfm": A_ii = J_i Y_i (+ cfm) in the world layout
# ---------------------------------------------------------------------------------------------------------------------


@functools.cache
def diagonal_rows_kernel() -> wp.Kernel:
    """The diagonal from a size group's rows, one thread per articulation row (``dim = articulations *
    max_constraints``): ``sum_k J Y`` in coordinate order, added atomically to the world's diagonal when nonzero (a
    row spans at most two articulations: two adds onto a zeroed diagonal, deterministic). The parallel tiers' worlds
    are skipped."""

    def J(k):
        return _element("J_group", "idx", "c", k)

    def Y(k):
        return _element("Y_group", "idx", "c", k)

    body = f"""    const int c = tid % max_constraints;
    const int idx = tid / max_constraints;
    if (idx >= n_arts) return;
    const int art = group_to_art.data[idx];
    const int world = art_to_world.data[art];
    const int m = world_constraint_count.data[world];
    if (c >= m) return;
    if ({_skips("m", "world")}) return;
{_indent(dot("val", "n_dofs", J, Y), "    ")}
    if (val != 0.0f) wp::atomic_add(&{_element("world_diag", "world", "c")}, val);
"""
    params = kernel_template.params(
        ("J_group", wp.array3d[float]),
        ("Y_group", wp.array3d[float]),
        ("group_to_art", wp.array[int]),
        ("art_to_world", wp.array[int]),
        ("world_constraint_count", wp.array[int]),
        ("n_dofs", int),
        ("max_constraints", int),
        ("n_arts", int),
        ("mf_constraint_count", wp.array[int]),
        ("skip_rows_le", int),
        ("world_diag", wp.array2d[float]),
    )
    return _kernel("dense_diagonal_rows", params, body)


@functools.cache
def diagonal_groups_kernel() -> wp.Kernel:
    """The diagonal from the size group's per-row diagonals (the tiled, warp and diagonal responses' ``diag_group``),
    one thread per world row (``dim = worlds * max_constraints``): the world's articulations' entries summed in their
    order, added atomically when nonzero. The parallel tiers' worlds are skipped."""
    body = f"""    const int c = tid % max_constraints;
    const int world = tid / max_constraints;
    const int m = world_constraint_count.data[world];
    if (c >= m) return;
    if ({_skips("m", "world")}) return;
    float value = 0.0f;
    const int end = world_group_art_start.data[world + 1];
    for (int offset = world_group_art_start.data[world]; offset < end; ++offset) {{
        const int art = world_group_to_art.data[offset];
        value += {_element("group_diag", "art_group_idx.data[art]", "c")};
    }}
    if (value != 0.0f) wp::atomic_add(&{_element("world_diag", "world", "c")}, value);
"""
    params = kernel_template.params(
        ("group_diag", wp.array2d[float]),
        ("world_group_art_start", wp.array[int]),
        ("world_group_to_art", wp.array[int]),
        ("art_group_idx", wp.array[int]),
        ("world_constraint_count", wp.array[int]),
        ("max_constraints", int),
        ("mf_constraint_count", wp.array[int]),
        ("skip_rows_le", int),
        ("world_diag", wp.array2d[float]),
    )
    return _kernel("dense_diagonal_groups", params, body)


@functools.cache
def cfm_worlds_kernel() -> wp.Kernel:
    """The diagonals' ``cfm``, one thread per world (``dim = worlds``): each row's added atomically, after the
    Delassus or response diagonal. The parallel tiers' worlds are skipped."""
    body = f"""    const int world = tid;
    const int m = world_constraint_count.data[world];
    if ({_skips("m", "world")}) return;
    for (int i = 0; i < m; ++i) wp::atomic_add(&{_element("world_diag", "world", "i")}, {_element("world_row_cfm", "world", "i")});
"""
    params = kernel_template.params(
        ("world_constraint_count", wp.array[int]),
        ("world_row_cfm", wp.array2d[float]),
        ("mf_constraint_count", wp.array[int]),
        ("skip_rows_le", int),
        ("world_diag", wp.array2d[float]),
    )
    return _kernel("dense_cfm_worlds", params, body)


# ---------------------------------------------------------------------------------------------------------------------
# Stage "delassus": C = J Y^T and its diagonal, accumulated onto a zeroed C and diagonal (S17: array clears)
# ---------------------------------------------------------------------------------------------------------------------


@functools.cache
def delassus_entries_kernel() -> wp.Kernel:
    """The Delassus of a size group, one thread per articulation entry (``dim = articulations * max_constraints^2``):
    ``C_ij = sum_k J_ik Y_jk`` in coordinate order, added atomically when nonzero (and to the diagonal)."""

    def J(k):
        return _element("J_group", "idx", "i", k)

    def Y(k):
        return _element("Y_group", "idx", "j", k)

    body = f"""    const int j = tid % max_constraints;
    const int i = (tid / max_constraints) % max_constraints;
    const int idx = tid / (max_constraints * max_constraints);
    if (idx >= n_arts) return;
    const int art = group_to_art.data[idx];
    const int world = art_to_world.data[art];
    const int n_constraints = world_constraint_count.data[world];
    if (i >= n_constraints || j >= n_constraints) return;
{_indent(dot("val", "n_dofs", J, Y), "    ")}
    if (val != 0.0f) wp::atomic_add(&{_element("world_C", "world", "i", "j")}, val);
    if (i == j && val != 0.0f) wp::atomic_add(&{_element("world_diag", "world", "i")}, val);
"""
    params = kernel_template.params(
        ("J_group", wp.array3d[float]),
        ("Y_group", wp.array3d[float]),
        ("group_to_art", wp.array[int]),
        ("art_to_world", wp.array[int]),
        ("world_constraint_count", wp.array[int]),
        ("n_dofs", int),
        ("max_constraints", int),
        ("n_arts", int),
        ("world_C", wp.array3d[float]),
        ("world_diag", wp.array2d[float]),
    )
    return _kernel("dense_delassus_entries", params, body)


# ---------------------------------------------------------------------------------------------------------------------
# Stage "apply": v = v_hat + Y^T lambda
# ---------------------------------------------------------------------------------------------------------------------


@functools.cache
def apply_coordinates_kernel() -> wp.Kernel:
    """The velocity of a size group's impulses, one thread per articulation coordinate (``dim = articulations *
    n_dofs``): ``v_out = v_hat + sum_c Y_c lambda_c`` over the world's rows in order, zero impulses included."""

    def Y(c):
        return _element("Y_group", "idx", c, "local_dof")

    def impulse(c):
        return _element("world_impulses", "world", c)

    body = f"""    const int local_dof = tid % n_dofs;
    const int idx = tid / n_dofs;
    if (idx >= n_arts) return;
    const int art = group_to_art.data[idx];
    const int world = art_to_world.data[art];
    const int n_constraints = world_constraint_count.data[world];
    const int dof_start = art_dof_start.data[art];
{_indent(dot("delta_v", "n_constraints", Y, impulse, index="c"), "    ")}
    const int global_dof = dof_start + local_dof;
    v_out.data[global_dof] = v_hat.data[global_dof] + delta_v;
"""
    params = kernel_template.params(
        ("group_to_art", wp.array[int]),
        ("art_to_world", wp.array[int]),
        ("art_dof_start", wp.array[int]),
        ("n_dofs", int),
        ("n_arts", int),
        ("world_constraint_count", wp.array[int]),
        ("max_constraints", int),
        ("Y_group", wp.array3d[float]),
        ("world_impulses", wp.array2d[float]),
        ("v_hat", wp.array[float]),
        ("v_out", wp.array[float]),
    )
    return _kernel("dense_apply_coordinates", params, body)


# ---------------------------------------------------------------------------------------------------------------------
# The builds by stage and mapping
# ---------------------------------------------------------------------------------------------------------------------


def _entry(
    stage: str,
    mapping: str,
    factory: Callable[..., wp.Kernel],
    grid: tuple[str, ...],
    block_dim=0,
    *,
    tiled: bool = False,
):
    """A registry entry: ``(**options) -> Build``, the factory's kernel for ``options`` and the launch's grid
    (``block_dim`` a count, or a function of the options)."""

    def build(**options) -> Build:
        block = block_dim(options) if callable(block_dim) else block_dim
        return Build(stage, mapping, factory(**options), grid, block, tiled)

    build.__doc__ = factory.__doc__
    return build


def _tile_threads(options) -> int:
    return int(options.get("tile_threads", 64))


def _warps(options) -> int:
    return 32 * int(options.get("warps_per_block", 4))


_CFM_WORLDS = _entry("cfm", "worlds", cfm_worlds_kernel, ("worlds",))

DENSE_JY_BUILDS: dict[tuple[str, str], Callable[..., Build]] = {
    ("factor", "articulations"): _entry("factor", "articulations", factor_articulations_kernel, ("articulations",)),
    ("factor", "coordinates"): _entry(
        "factor", "coordinates", factor_coordinates_kernel, ("articulations", "coordinates")
    ),
    ("factor", "warps"): _entry("factor", "warps", factor_warps_kernel, ("articulations", "32 lanes"), _warps),
    ("factor", "tiles"): _entry("factor", "tiles", factor_tiles_kernel, ("articulations",), _tile_threads, tiled=True),
    ("response", "rows"): _entry("response", "rows", response_rows_kernel, ("articulations", "rows")),
    ("response", "lanes"): _entry("response", "lanes", response_lanes_kernel, ("articulations", "32 lanes"), 256),
    ("response", "diagonal"): _entry("response", "diagonal", response_diagonal_kernel, ("articulations", "rows")),
    ("response", "warps"): _entry("response", "warps", response_warps_kernel, ("articulation quads",), 128, tiled=True),
    ("response", "tiles"): _entry(
        "response", "tiles", response_tiles_kernel, ("articulations", "row chunks"), _tile_threads, tiled=True
    ),
    ("diagonal", "rows"): _entry("diagonal", "rows", diagonal_rows_kernel, ("articulations", "rows")),
    ("diagonal", "groups"): _entry("diagonal", "groups", diagonal_groups_kernel, ("worlds", "rows")),
    ("cfm", "worlds"): _CFM_WORLDS,
    ("apply", "coordinates"): _entry(
        "apply", "coordinates", apply_coordinates_kernel, ("articulations", "coordinates")
    ),
}
"""The builds of the dense response's data, by ``(stage, mapping)``. Each takes its kernel factory's options
as keywords and returns the :class:`Build`. The mappings: ``articulations``, a thread per articulation; ``coordinates``,
a thread per articulation (or world row) coordinate; ``rows``, a thread per articulation row; ``lanes``, 32 threads
per articulation, rows lane-strided; ``warps``, a warp per articulation, a lane per row (or the lanes assembling and
lane 0 factoring); ``tiles``, a block per articulation (and row chunk) through the tile primitives; ``queue``, warps
walking the local owners' general-world queue; ``pairs``, a block per paired world, a thread per row; ``groups``, a
thread per world row summing its articulations' group diagonals; ``worlds``, a thread per world. ``diagonal`` is the thread-per-row response of a structurally diagonal ``H``.
The register whitening of the six-coordinate group is a device function, :func:`whiten_six`."""

ASSEMBLED_BUILDS: dict[tuple[str, str], Callable[..., Build]] = {
    ("delassus", "entries"): _entry("delassus", "entries", delassus_entries_kernel, ("articulations", "rows", "rows")),
    ("cfm", "worlds"): _CFM_WORLDS,
}
"""The builds of an :class:`.Assembled` operator's data (the CPU's split mode), by ``(stage, mapping)``: ``C = J Y^T``
and its diagonal accumulated onto a zeroed ``C`` and diagonal, a thread per entry, then the diagonals' ``cfm``. The
response itself is a :data:`DENSE_JY_BUILDS` build."""
