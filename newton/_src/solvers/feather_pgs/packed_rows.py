# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dense rows stored packed by support: the matrix-free sweeps' placement of whitened rows.

A dense row touches at most two articulations, its sides (:func:`.dense_rows.row_sides`): the first by index, then
the second or none. Packed, a row holds its sides' coordinates in order, the first side's, then the second's: entry
``e`` of a side of ``n`` coordinates is the side's coordinate ``e``, world coordinate ``offset + e``, so a row's
entries ascend with the world's coordinates. An entry's world coordinate is not stored: it follows from the row's
sides (``row_sides``, :class:`.dense_rows.RowSupport`, :data:`PACKED_SIDES`). A row holds one array of entries,
``packed_Z``: the row producers write its Jacobian ``J`` at the row's sides (:func:`.dense_rows.jacobian_row`), and the
build, which takes the row's incident velocity ``J v_hat`` (``row_incident``) from it, replaces it with its whitened form
``Z = L^-1 P J^T`` through the side's leaf-first factor ``L``, in the kit's positions: factor coordinate ``i`` of a side
at its entry ``n - 1 - i`` (:class:`.response_operator.CholeskySolve`). The worlds a sweep whitens in the kernel keep
``J``.

Whitening keeps a side's support: ``J`` vanishes off the ancestors of the row's bodies, a set closed under ancestors,
so ``Z`` does too and the solve reads ``L`` only there. A row's diagonal is ``A_ii = |Z|^2``, its entries' squares
summed in entry order (the row law adds ``cfm``), and its velocity term at ``v = v_hat + P^T L^-T du`` is
``J_i v_hat + Z_i^T du``: the sweeps hold the velocity change ``du`` on every articulation they touch
(:class:`.sweep_routes.FactorCoordinates`).

The rows' width ``S`` (``support``) comes from the data: the most coordinates two articulations of one world hold
(:func:`support_width`). Where that is every coordinate of the widest world (worlds of one or two articulations), the
rows are laid out ``dense``: entry ``e`` is world coordinate ``e`` and the coordinates outside the row's sides hold
zero (coordinate -1), so a row is also its dense row over the world, the form the routes that hold a world's
coordinates on lanes read. Bilateral elimination widens them (:data:`WIDE`): a corrected row reaches every tree of its
block, past its two sides, so the layout is dense and the sides no longer bound a row's coordinates. The producers
write each row's sides whole, zeros included, and record its sides: the
rows are never cleared. The build (:func:`pack_kernel`, one launch) whitens the rows the routes read, zeroes their
entries past the sides and forms their diagonals (the parallel tiers whiten their worlds' rows in the kernel from
the rows' sides: the build leaves them). Bilateral pre-elimination corrects them
(:func:`.response_builds_rows.elimination`).
"""

from __future__ import annotations

import functools

import numpy as np
import warp as wp

from . import kernel_template
from .device_preludes import DevicePrelude
from .field_group import Field, FieldGroup
from .kernel_template import THREADS, KernelTemplate
from .kernels import PGS_LOCAL_SOLVE_OWNER_GENERAL
from .response_operator import CholeskySolve

__all__ = ["PACKED_ROWS", "PACKED_SIDES", "WIDE", "pack_kernel", "support_width"]

WIDE = 2
"""The ``dense`` layout of rows that bilateral elimination corrected: every entry is a world coordinate's, sides or
not."""

PACKED_ROWS = FieldGroup(Field("packed_Z", wp.float32, ("world", "row", "support"), owner="placement"))
"""Each dense row's packed entries: its Jacobian, then its whitened value."""


_PACKED_SIDES = """    // A packed row's entries (packed_rows): PACKED_COORDINATE(sides, e) is the world coordinate of entry e of the
    // row with these sides, -1 past them. Dense: entry e is coordinate e, held inside a side's coordinates; else the
    // first side's entries, then the second's.
    const auto PACKED_COORDINATE = [&](const wp::vec2i sides, const int e) {
        const int a0 = sides[0], a1 = sides[1];
        const int n0 = a0 >= 0 ? response_dofs.data[a0] : 0, n1 = a1 >= 0 ? response_dofs.data[a1] : 0;
        const int off0 = a0 >= 0 ? articulation_world_dof_offset.data[a0] : 0;
        const int off1 = a1 >= 0 ? articulation_world_dof_offset.data[a1] : 0;
        if (packed_dense != 0) return ((e >= off0 && e < off0 + n0) || (e >= off1 && e < off1 + n1)) ? e : -1;
        return e < n0 ? off0 + e : (e < n0 + n1 ? off1 + e - n0 : -1);
    };
    // PACKED_FIRST(sides, art): the entry where articulation art's side of a row with these sides starts, -1 when it
    // is not one of them (dense_rows.jacobian_row).
    const auto PACKED_FIRST = [&](const wp::vec2i sides, const int art) {
        if (art < 0 || (art != sides[0] && art != sides[1])) return -1;
        if (packed_dense != 0) return articulation_world_dof_offset.data[art];
        return art == sides[0] ? 0 : response_dofs.data[sides[0]];
    };"""

PACKED_SIDES = DevicePrelude("packed_sides", _PACKED_SIDES, _PACKED_SIDES)
"""``PACKED_COORDINATE(sides, e)`` and ``PACKED_FIRST(sides, art)`` for native sources that take ``response_dofs``,
``articulation_world_dof_offset`` and ``packed_dense``: a packed row's coordinates and sides from its sides (no stored
coordinates)."""


def support_width(response_dofs: np.ndarray, art_to_world: np.ndarray, world_count: int) -> int:
    """The packed rows' width: the most coordinates two articulations of one world hold (one, in a world of one);
    the rows are ``dense`` where this is the widest world's coordinates.

    Args:
        response_dofs: Each articulation's coordinates.
        art_to_world: Each articulation's world (-1 for none).
        world_count: The worlds.
    """
    largest = np.zeros((max(int(world_count), 1), 2), dtype=np.int64)
    for size, world in zip(np.asarray(response_dofs).tolist(), np.asarray(art_to_world).tolist(), strict=True):
        if world < 0:
            continue
        if size > largest[world, 0]:
            largest[world, 1] = largest[world, 0]
            largest[world, 0] = size
        elif size > largest[world, 1]:
            largest[world, 1] = size
    return max(int(largest.sum(axis=1).max()), 1)


def _pack_abi(sizes: tuple[int, ...]) -> tuple:
    """The rows' build: each side size's factors (``L_<n>``), every row's sides and the articulations' sizes, group
    indices and first coordinates, the rows per world and their owners (the worlds other sweeps own skipped:
    :func:`_kept`), the rows' ``cfm``, the predictor, the packed rows, the diagonal and the incident velocity."""
    factors = [(f"L_{n}", wp.array3d[float]) for n in sizes]
    return kernel_template.params(
        *factors,
        ("row_sides", wp.array2d[wp.vec2i]),
        ("response_dofs", wp.array[int]),
        ("art_group", wp.array[int]),
        ("articulation_world_dof_offset", wp.array[int]),
        ("articulation_dof_start", wp.array[int]),
        ("world_constraint_count", wp.array[int]),
        ("mf_constraint_count", wp.array[int]),
        ("skip_rows_le", int),
        ("local_solve_owner", wp.array[int]),
        ("rows_elsewhere", int),
        ("world_row_cfm", wp.array2d[float]),
        ("dense", int),
        ("write_diag", int),
        ("v_hat", wp.array[float]),
        ("packed_Z", wp.array3d[float]),
        ("world_diag", wp.array2d[float]),
        ("row_incident", wp.array2d[float]),
    )


def _kept(world: str) -> str:
    """The worlds whose rows a sweep whitens in the kernel: the parallel tiers' (at most ``skip_rows_le`` rows and no
    free-body rows; ``skip_rows_le`` 0: none) and the local owners'. Their rows stay Jacobians."""
    count = f"world_constraint_count.data[{world}]"
    tier = f"(skip_rows_le > 0 && {count} <= skip_rows_le && mf_constraint_count.data[{world}] == 0)"
    return f"{tier} || local_solve_owner.data[{world}] != {PGS_LOCAL_SOLVE_OWNER_GENERAL}"


_UNROLLED_SIDE = 32
"""The largest side a pack thread whitens fully unrolled, its row in registers; wider sides keep a rolled solve on a
thread-local row."""


def _side(n: int) -> str:
    """A side of ``n`` coordinates on its row's thread: load its Jacobian and add ``J v_hat`` to ``incident`` in entry
    order, then, unless the row's world ``keeps`` it, whiten it in registers (:class:`.CholeskySolve`, Z0: a zero
    diagonal gives zero), store it and add its squares to ``norm`` in entry order."""

    def L(row, col):
        return f"L_{n}.data[((size_t)group * {n} + ({row})) * {n} + ({col})]"

    # The row in physical positions: factor coordinate k at position n - 1 - k (CholeskySolve's leaf-first operand).
    solve = CholeskySolve(n, L, unroll=n <= _UNROLLED_SIDE)
    pragma = "#pragma unroll" if n <= _UNROLLED_SIDE else "#pragma unroll 1"
    return f"""        if (n == {n}) {{
            float z[{n}];
            {pragma}
            for (int e = 0; e < {n}; ++e) {{
                z[e] = packed_Z.data[base + e];
                incident += z[e] * v_hat.data[start + e];
            }}
            if (!keeps) {{
{solve.whiten(lambda i: f"z[{i}]", lambda i: f"z[{i}]")}
                {pragma}
                for (int e = 0; e < {n}; ++e) {{
                    packed_Z.data[base + e] = z[e];
                    norm += z[e] * z[e];
                }}
            }}
        }}"""


@functools.cache
def pack_kernel(sizes: tuple[int, ...]) -> wp.Kernel:
    """The rows' pass, one thread per world row (``dim = worlds * rows``): each side's Jacobian, which its producer
    wrote packed, gives the row's incident velocity ``J v_hat`` in entry order and is whitened on the thread (Z0: a
    zero diagonal gives zero) and stored in its place, each entry once; pad the row past its sides (zero values); and,
    in the worlds the general solver owns, store its diagonal by ``write_diag``: 0 none, 1 ``A_ii = |Z|^2`` (an owner
    adds ``cfm`` after), 2 ``|Z|^2 + cfm``, the squares summed in entry order. A world another sweep whitens in the
    kernel (:func:`_kept`) keeps its rows and gets their incident velocity alone; with ``rows_elsewhere`` (keyed rows,
    row packets) its rows are not here at all.

    ``sizes`` are the side sizes with a dense factor (each a ``L_<n>``); a side of another size is left as it is.
    Consecutive threads hold one world's rows, so a side's factor reads are the same address across a warp.
    """
    sizes = tuple(int(n) for n in sizes)
    sides = "\n".join(_side(n) for n in sizes)
    # The solves' loop variable is ``row``: the world's row is ``slot``, its thread ``tid``.
    body = f"""    const int capacity = row_sides.shape[1];
    const int world = tid / capacity;
    const int slot = tid - world * capacity;
    if (world >= world_constraint_count.shape[0] || slot >= world_constraint_count.data[world]) return;
    const bool keeps = {_kept("world")};
    if (keeps && rows_elsewhere != 0) return;
    const wp::vec2i sides = row_sides.data[tid];
    const int support = packed_Z.shape[2];
    // Each side's entries [first, first + n): after the previous side's, or at its coordinates (dense).
    int first[2] = {{0, 0}}, count[2] = {{0, 0}};
    float norm = 0.0f, incident = 0.0f;
    for (int side = 0; side < 2; ++side) {{
        const int art = sides[side];
        if (art < 0) continue;
        const int n = response_dofs.data[art];
        const int group = art_group.data[art];
        const int offset = articulation_world_dof_offset.data[art];
        const int start = articulation_dof_start.data[art];
        count[side] = n;
        first[side] = dense != 0 ? offset : side == 0 ? 0 : count[0];
        const size_t base = (size_t)tid * support + first[side];
{sides}
    }}
    row_incident.data[tid] = incident;
    if (keeps) return;
    for (int e = 0; e < support; ++e) {{
        if ((e >= first[0] && e < first[0] + count[0]) || (e >= first[1] && e < first[1] + count[1])) continue;
        packed_Z.data[(size_t)tid * support + e] = 0.0f;
    }}
    if (write_diag != 0 && local_solve_owner.data[world] == {PGS_LOCAL_SOLVE_OWNER_GENERAL})
        world_diag.data[tid] = write_diag == 2 ? norm + world_row_cfm.data[tid] : norm;
"""
    name = "pack_dense_rows" + "".join(f"_{n}" for n in sizes)
    return KernelTemplate(
        name, _pack_abi(sizes), body, dispatch=THREADS, dual=True, native_name=f"{name}_native"
    ).kernel()
