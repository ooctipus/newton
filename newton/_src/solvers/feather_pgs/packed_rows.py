# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dense rows stored packed by support: the matrix-free sweeps' placement of whitened rows.

A dense row touches at most two articulations, its sides (:func:`.dense_rows.row_sides`): the first by index, then
the second or none. Packed, a row holds its sides' coordinates in order, the first side's, then the second's: entry
``e`` of a side of ``n`` coordinates is the side's coordinate ``e``, world coordinate ``offset + e``, so a row's
entries ascend with the world's coordinates. An entry's world coordinate is not stored: it follows from the row's
sides (``row_sides``, :class:`.dense_rows.RowSupport`, :data:`PACKED_SIDES`). Each entry keeps

* the row's Jacobian ``J`` (``packed_J``), which the row producers write at the row's sides
  (:func:`.dense_rows.jacobian_row`), the build reads for the row's incident velocity ``J v_hat`` (``row_incident``)
  and the diagnostics read with the physical velocity (:func:`.kernels.world_contact_row_dot`);
* its whitened form ``Z = L^-1 P J^T`` (``packed_Z``) through the side's leaf-first factor ``L``, in the kit's
  positions: factor coordinate ``i`` of a side at its entry ``n - 1 - i``
  (:class:`.response_operator.CholeskySolve`).

Whitening keeps a side's support: ``J`` vanishes off the ancestors of the row's bodies, a set closed under ancestors,
so ``Z`` does too and the solve reads ``L`` only there. A row's diagonal is ``A_ii = |Z|^2``, its entries' squares
summed in entry order (the row law adds ``cfm``), and its velocity term ``J_i v = Z_i^T u`` with ``u = L^T P v`` on
each side: the sweeps hold every articulation they touch in factor coordinates
(:class:`.sweep_routes.FactorCoordinates`).

The rows' width ``S`` (``support``) comes from the data: the most coordinates two articulations of one world hold
(:func:`support_width`). Where that is every coordinate of the widest world (worlds of one or two articulations), the
rows are laid out ``dense``: entry ``e`` is world coordinate ``e`` and the coordinates outside the row's sides hold
zero (coordinate -1), so a row is also its dense row over the world, the form the routes that hold a world's
coordinates on lanes read. The producers write each row's sides whole, zeros included, and record its sides: the
rows are never cleared. The build (:func:`pack_kernel`, one launch) whitens the rows the routes read, zeroes their
entries past the sides and forms their diagonals (the parallel tiers whiten their worlds' rows in the kernel from
the rows' sides: the build leaves them without a warm start). The
warm start applies the rows' impulses through them (:func:`apply_kernel`), and bilateral pre-elimination corrects
them (:func:`.response_builds_rows.elimination`) and projects the velocity onto its blocks (:func:`project_kernel`).
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
from .response_operator import CholeskySolve, Reversed

__all__ = ["PACKED_ROWS", "PACKED_SIDES", "apply_kernel", "pack_kernel", "project_kernel", "support_width"]

PACKED_ROWS = FieldGroup(
    Field("packed_J", wp.float32, ("world", "row", "support"), owner="placement"),
    Field("packed_Z", wp.float32, ("world", "row", "support"), owner="placement"),
)
"""Each dense row's packed entries: world coordinate, Jacobian and whitened value."""


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
    :func:`_skipped`), the rows' ``cfm``, the predictor, the packed rows, the diagonal and the incident velocity."""
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
        ("skip_local", int),
        ("rows_elsewhere", int),
        ("world_row_cfm", wp.array2d[float]),
        ("dense", int),
        ("write_diag", int),
        ("v_hat", wp.array[float]),
        ("packed_J", wp.array3d[float]),
        ("packed_Z", wp.array3d[float]),
        ("world_diag", wp.array2d[float]),
        ("row_incident", wp.array2d[float]),
    )


def _skipped(world: str) -> str:
    """The worlds whose rows another sweep forms: the parallel tiers' (at most ``skip_rows_le`` rows and no free-body
    rows; ``skip_rows_le`` 0: none), which whiten them in the kernel, and with ``skip_local`` (the row packets) the
    local owners', which solve their own rows. The packing leaves them, unless a warm start applies their rows'
    impulses through the packed rows (then both are 0)."""
    count = f"world_constraint_count.data[{world}]"
    tier = f"(skip_rows_le > 0 && {count} <= skip_rows_le && mf_constraint_count.data[{world}] == 0)"
    local = f"(skip_local != 0 && local_solve_owner.data[{world}] != {PGS_LOCAL_SOLVE_OWNER_GENERAL})"
    return f"{tier} || {local}"


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
                z[e] = packed_J.data[base + e];
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
    zero diagonal gives zero) and stored, each entry once; pad the row past its sides (zero values); and, in the worlds
    the general solver owns, store its diagonal by ``write_diag``: 0 none, 1 ``A_ii = |Z|^2`` (an owner adds ``cfm``
    after), 2 ``|Z|^2 + cfm``, the squares summed in entry order. A world another sweep whitens in the kernel
    (:func:`_skipped`) keeps its rows and gets their incident velocity alone; with ``rows_elsewhere`` (keyed rows, row
    packets) its rows are not here at all.

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
    const bool keeps = {_skipped("world")};
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
        packed_J.data[(size_t)tid * support + e] = 0.0f;
        packed_Z.data[(size_t)tid * support + e] = 0.0f;
    }}
    if (write_diag != 0 && local_solve_owner.data[world] == {PGS_LOCAL_SOLVE_OWNER_GENERAL})
        world_diag.data[tid] = write_diag == 2 ? norm + world_row_cfm.data[tid] : norm;
"""
    name = "pack_dense_rows" + "".join(f"_{n}" for n in sizes)
    return KernelTemplate(
        name, _pack_abi(sizes), body, dispatch=THREADS, dual=True, native_name=f"{name}_native"
    ).kernel()


_APPLY_ABI = kernel_template.params(
    ("L_group", wp.array3d[float]),
    ("group_to_art", wp.array[int]),
    ("art_to_world", wp.array[int]),
    ("art_dof_start", wp.array[int]),
    ("response_dofs", wp.array[int]),
    ("n_dofs", int),
    ("row_sides", wp.array2d[wp.vec2i]),
    ("world_constraint_count", wp.array[int]),
    ("articulation_world_dof_offset", wp.array[int]),
    ("dense", int),
    ("packed_Z", wp.array3d[float]),
    ("world_impulses", wp.array2d[float]),
    ("v_hat", wp.array[float]),
    ("v_out", wp.array[float]),
)
"""A size group's velocity of the impulses: its factors, articulations, the rows' sides and whitened entries, the
impulses and the velocities."""


@functools.cache
def apply_kernel() -> wp.Kernel:
    """The velocity of the impulses, one thread per articulation of the size group (``dim = articulations``): ``u =
    sum_r Z_r lambda_r`` over its world's rows in row order (the rows it is a side of, zero impulses included), then
    ``v_out = v_hat + P^T L^-T u`` (Z0), built in place in ``v_out``."""

    def L(row, col):
        return f"L_group.data[((size_t)group * n_dofs + ({row})) * n_dofs + ({col})]"

    solve = CholeskySolve("n_dofs", L)
    velocity = Reversed(lambda k: f"v_last[-{k}]")
    body = f"""    const int group = tid;
    if (group >= group_to_art.shape[0]) return;
    const int art = group_to_art.data[group];
    const int world = art_to_world.data[art];
    const int start = art_dof_start.data[art];
    const int capacity = row_sides.shape[1];
    const int support = packed_Z.shape[2];
    const int rows = world_constraint_count.data[world];
    const int count = rows < capacity ? rows : capacity;
    for (int e = 0; e < n_dofs; ++e) v_out.data[start + e] = 0.0f;
    for (int r = 0; r < count; ++r) {{
        const wp::vec2i sides = row_sides.data[world * capacity + r];
        int first = -1;
        if (sides[0] == art) first = 0;
        else if (sides[1] == art) first = response_dofs.data[sides[0]];
        if (first < 0) continue;
        if (dense != 0) first = articulation_world_dof_offset.data[art];
        const float impulse = world_impulses.data[world * world_impulses.shape[1] + r];
        const float* z = &packed_Z.data[((size_t)world * capacity + r) * support + first];
        for (int e = 0; e < n_dofs; ++e) v_out.data[start + e] += z[e] * impulse;
    }}
    float* v_last = &v_out.data[start + n_dofs - 1];
{solve.leave(velocity, velocity)}
    for (int e = 0; e < n_dofs; ++e) v_out.data[start + e] += v_hat.data[start + e];
"""
    return KernelTemplate(
        "apply_dense_rows", _APPLY_ABI, body, dispatch=THREADS, dual=True, native_name="apply_dense_rows_native"
    ).kernel()


_PROJECT_ABI = kernel_template.params(
    ("L_group", wp.array3d[float]),
    ("group_to_art", wp.array[int]),
    ("art_to_world", wp.array[int]),
    ("art_dof_start", wp.array[int]),
    ("articulation_world_dof_offset", wp.array[int]),
    ("n_dofs", int),
    ("dense", int),
    ("art_to_preelim", wp.array[int]),
    ("preelim_slots", wp.array[int]),
    ("preelim_nB", wp.array[int]),
    ("preelim_weight", wp.array[float]),
    ("packed_Z", wp.array3d[float]),
    ("v_out", wp.array[float]),
)
"""A size group's bilateral projection: its factors and articulations, the blocks' rows and weights
(:func:`.response_builds_rows.elimination`), the packed rows and the velocity."""


@functools.cache
def project_kernel(rows: int) -> wp.Kernel:
    """Project the velocity onto each articulation's bilateral block of up to ``rows`` rows, ``v -= P^T L^-T Z_B y``
    with the block's weights ``y`` (:func:`.response_builds_rows.elimination`), one thread per articulation of the
    size group (``dim = articulations``): enter ``u = L^T P v``, subtract ``Z_B y`` in block order, leave (Z0), in
    place in ``v_out``."""

    def L(row, col):
        return f"L_group.data[((size_t)group * n_dofs + ({row})) * n_dofs + ({col})]"

    solve = CholeskySolve("n_dofs", L)
    velocity = Reversed(lambda k: f"v_last[-{k}]")
    body = f"""    const int group = tid;
    if (group >= group_to_art.shape[0]) return;
    const int art = group_to_art.data[group];
    const int pe = art_to_preelim.data[art];
    if (pe < 0) return;
    const int m = preelim_nB.data[pe];
    if (m == 0) return;
    const int world = art_to_world.data[art];
    const int start = art_dof_start.data[art];
    const int first = dense != 0 ? articulation_world_dof_offset.data[art] : 0;
    const int capacity = packed_Z.shape[1];
    const int support = packed_Z.shape[2];
    const int base = pe * {int(rows)};
    float* v_last = &v_out.data[start + n_dofs - 1];
{solve.enter(velocity, velocity)}
    for (int e = 0; e < n_dofs; ++e) {{
        float acc = 0.0f;
        for (int p = 0; p < m; ++p) {{
            const size_t row = (size_t)world * capacity + preelim_slots.data[base + p];
            acc += packed_Z.data[row * support + first + e] * preelim_weight.data[base + p];
        }}
        v_out.data[start + e] -= acc;
    }}
{solve.leave(velocity, velocity)}
"""
    return KernelTemplate(
        f"project_dense_rows_{int(rows)}",
        _PROJECT_ABI,
        body,
        dispatch=THREADS,
        dual=True,
        native_name="project_dense_rows_native",
    ).kernel()
