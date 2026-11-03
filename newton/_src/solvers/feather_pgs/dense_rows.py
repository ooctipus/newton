# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The row records every row producer writes, declared once as fields (:mod:`.field_group`).

One store holds every world line's rows (:data:`.solver_fields.DENSE_ROWS`): the dense class's from row 0, then the
free bodies' from the store's ``body_base``. :data:`DenseRows` is the Warp struct of the dense class's records; the
contact producers' input (:data:`.contact_input.ContactInput`) carries the same fields. :data:`MfRows` reads the free
bodies' rows: the same records at ``base``, and their placement (bodies, coordinates, ``Z``).
"""

import dataclasses

import warp as wp

from . import field_group
from .field_group import Field
from .solver_fields import DENSE_ROWS, MF_ROWS

RECORDS = (
    Field("row_type", wp.int32, ("world", "row"), owner="placement"),
    Field("row_parent", wp.int32, ("world", "row"), owner="placement"),
    Field("row_mu", wp.float32, ("world", "row"), owner="placement"),
    Field("row_beta", wp.float32, ("world", "row"), owner="placement"),
    Field("phi", wp.float32, ("world", "row"), owner="placement"),
    Field("target", wp.float32, ("world", "row"), attr="target_velocity", owner="placement"),
    Field("restitution", wp.float32, ("world", "row"), attr="row_restitution", owner="placement"),
    Field("rhs", wp.float32, ("world", "row"), owner="placement"),
    Field("diag", wp.float32, ("world", "row"), owner="placement"),
    Field("row_w", wp.float32, ("world", "row"), owner="placement"),
    Field("row_eps", wp.float32, ("world", "row"), owner="placement"),
    Field("sides", wp.vec2i, ("world", "row"), attr="row_sides", owner="placement"),
    Field("first", wp.int32, ("world",), attr="dense_first", owner="placement"),
)
"""Each dense row's kind, friction parent, coefficients, gap, targets and law outputs (a material contact's
compliance ``row_eps`` among them), and its sides (the articulations whose response coordinates it touches:
:func:`row_sides`), by world and row. The rows' regularization ``cfm`` is the law's parameter (``pgs_cfm``), one value:
a row's diagonal is ``A_ii = |Z_i|^2 + cfm``."""

DenseRows = field_group.struct(
    "DenseRows", (*RECORDS, Field("cfm", wp.float32, owner="options", attr="pgs_cfm")), __name__
)


@wp.func
def row_side(art: int, response_dofs: wp.array[int]) -> int:
    """``art`` as a row's side: itself where it has response coordinates, else -1.

    An articulation without them (a prescribed free body the forest plan elides, or one without coordinates) has no
    world coordinates (its ``articulation_world_dof_offset`` is -1). Its motion enters a row through the row's target
    alone (:func:`.contact_laws.prescribed_target`).
    """
    side = int(-1)
    if art >= 0:
        if response_dofs[art] > 0:
            side = art
    return side


@wp.func
def row_sides(art_a: int, art_b: int, response_dofs: wp.array[int]) -> wp.vec2i:
    """A row's sides, its support: the articulations of ``art_a`` and ``art_b`` that have response coordinates
    (:func:`row_side`), the first by index, then the second or -1 (none, or the same).

    Every side therefore holds coordinates at a first world coordinate, which every reader of the sides takes as given:
    the packed rows' entries, the sweeps' factor coordinates (:class:`.sweep_routes.FreeBodyCoordinates` marks a side
    at that coordinate). The rows that name one articulation write its size group's articulation, which has them.
    Articulation indices ascend with a world's coordinates, so a packed row's entries (:mod:`.packed_rows`) ascend too.
    """
    a = row_side(art_a, response_dofs)
    b = row_side(art_b, response_dofs)
    if a < 0 or a == b:
        return wp.vec2i(b, -1)
    if b < 0:
        return wp.vec2i(a, -1)
    return wp.vec2i(wp.min(a, b), wp.max(a, b))


UNIT_CONTACT = wp.constant(1)
"""A row's unit source (:func:`unit_source`): contact ``c``'s row ``r``, index ``4 c + r``."""
UNIT_LIMIT = wp.constant(2)
"""A joint limit's row: coordinate ``dof``'s side (0 below, ``J = +1``; 1 above, ``J = -1``), index ``2 dof + side``."""
UNIT_MIMIC = wp.constant(3)
"""A mimic's row: index the mimic."""
UNIT_LINK = wp.constant(4)
"""A link's row: index ``6 link + row``."""
UNIT_VELOCITY = wp.constant(5)
"""A joint velocity limit's row: index ``2 dof + side``, side 0 the lower (``+e_dof``), side 1 the upper (``-e_dof``)."""
UNIT_SHIFT = wp.constant(28)
"""Bits of a unit source's index: its kind sits above them."""


@wp.func
def unit_source(kind: int, index: int) -> int:
    """A row's unit source: what the one writer (:mod:`.walk_rows`) reads to find the row's law and seed, its kind
    (``UNIT_*``) above :data:`UNIT_SHIFT` bits of index."""
    return (kind << UNIT_SHIFT) | index


@wp.struct
class RowSupport:
    """Where a dense row's Jacobian entries sit. In the matrix-free sweeps' packed rows (``packed``,
    :mod:`.packed_rows`) a row of world ``w`` is ``J[w, row, :]``: its sides (``sides``: :func:`row_sides`), each
    articulation's coordinates (``dofs``), first world coordinate (``offset``) and world (``world``), and the layout
    (``dense``: entry ``e`` is world coordinate ``e``; else the first side's entries, then the second's). Otherwise a
    size group's rows hold each articulation's side from entry 0, row ``J[group, row, :]``.

    Where the one writer walks the rows (``walks``, :mod:`.walk_rows`), a producer writes no ``J``: it records each
    row's unit source (``units``, :func:`unit_source`), by world line and row, from which the writer takes its seed."""

    sides: wp.array2d[wp.vec2i]
    dofs: wp.array[int]
    offset: wp.array[int]
    world: wp.array[int]
    dense: int
    packed: int
    first: wp.array[int]
    units: wp.array2d[int]
    walks: int


@wp.func
def jacobian_row(support: RowSupport, group: int, art: int, sides: wp.vec2i) -> wp.vec2i:
    """Where articulation ``art``'s side of a row with ``sides`` starts in the Jacobian rows a producer writes: the
    rows' first index and the side's first entry (``J[at[0], row, at[1] + e]`` is the side's coordinate ``e``). The
    packed rows (:class:`RowSupport`) hold it at its world coordinates (dense) or after the row's first side; a size
    group's rows at its group, from entry 0."""
    if support.packed == 0:
        return wp.vec2i(group, 0)
    line = support.first[support.world[art]]
    if support.dense != 0:
        return wp.vec2i(line, support.offset[art])
    if art == sides[0]:
        return wp.vec2i(line, 0)
    return wp.vec2i(line, support.dofs[sides[0]])


@wp.func
def clear_side(support: RowSupport, J: wp.array3d[float], at: wp.vec2i, row: int, art: int):
    """Zero articulation ``art``'s side of a packed row (at ``at``: :func:`jacobian_row`) before a producer writes
    only its nonzero entries: the packed rows are never cleared, every producer writes its rows' sides whole. A size
    group's rows are cleared each step: nothing to do."""
    if support.packed != 0:
        for e in range(support.dofs[art]):
            J[at[0], row, at[1] + e] = 0.0


MF_RECORDS = (
    *(
        dataclasses.replace(MF_ROWS[attr], name=name, attr=attr)
        for name, attr in (
            ("Z_a", "mf_Z_a"),
            ("Z_b", "mf_Z_b"),
        )
    ),
    *(
        dataclasses.replace(DENSE_ROWS[attr], name=name, attr=attr, owner="body")
        for name, attr in (
            ("row_type", "row_type"),
            ("row_parent", "row_parent"),
            ("row_mu", "row_mu"),
            ("row_beta", "row_beta"),
            ("phi", "phi"),
            ("restitution", "row_restitution"),
            ("incident", "row_incident"),
            ("rhs", "rhs"),
            ("target", "target_velocity"),
            ("diag", "diag"),
            ("sides", "row_sides"),
        )
    ),
    Field("eps", wp.float32, ("world", "row"), attr="row_eps", owner="body"),
    Field("first", wp.int32, ("world",), attr="dense_first", owner="placement"),
)
"""Each free-body row's per-body blocks (the writers' ``J``, whitened in place to ``Z`` by the rows' build), by world
line and free-body row; and its records, the row store's region of the free bodies
(:meth:`.placement.Placement.body_region`, row ``i`` the store's ``body_base + i``): kind, friction parent,
coefficient, the row law's beta, gap, restitution, incident velocity (``J v_hat``, stored by the build), rhs, target,
effective mass ``A_ii``, material compliance and sides (the responding bodies' articulations in ``a``/``b`` order, -1
for none)."""

MfRows = field_group.struct("MfRows", MF_RECORDS, __name__)


def bind_mf(placement):
    """Bind the placement's free-body rows into an :data:`MfRows`: their records in the row store's free-body region."""
    return field_group.bind(MfRows, placement=placement, body=placement.body_rows)


def bind(placement, data=None, options=None):
    """Bind the placement's dense row records and the law's ``cfm`` (``options``, by default the placement's) into a
    :data:`DenseRows` (``data``: an instance to refill)."""
    return field_group.bind(
        DenseRows, data, placement=placement, options=placement.options if options is None else options
    )
