# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The row records every row producer writes, declared once as fields (:mod:`.field_group`).

One store holds every world line's rows (:data:`DENSE_ROWS`): the dense class's from row 0, then the
free bodies' from the store's ``body_base``. :data:`DenseRows` is the Warp struct of the dense class's records; the
contact producers' input (:data:`.contact_input.ContactInput`) carries the same fields. :data:`MfRows` reads the free
bodies' rows: the same records at ``base``. Every row's whitened ``Z`` is the packed rows' (:mod:`.packed_rows`).
"""

import dataclasses

import warp as wp

from . import field_group
from .field_group import Field, FieldGroup

DENSE_ROWS = FieldGroup(
    Field("rhs", wp.float32, ("world", "row"), owner="placement"),
    Field("row_incident", wp.float32, ("world", "row"), owner="placement"),
    Field("impulses", wp.float32, ("world", "row"), owner="placement"),
    Field("diag", wp.float32, ("world", "row"), owner="placement"),
    Field("row_type", wp.int32, ("world", "row"), owner="placement"),
    Field("row_parent", wp.int32, ("world", "row"), fill=-1, owner="placement"),
    Field("row_mu", wp.float32, ("world", "row"), owner="placement"),
    Field("row_restitution", wp.float32, ("world", "row"), owner="placement"),
    Field("row_beta", wp.float32, ("world", "row"), owner="placement"),
    Field("phi", wp.float32, ("world", "row"), owner="placement"),
    Field("target_velocity", wp.float32, ("world", "row"), owner="placement"),
    Field("row_sides", wp.vec2i, ("world", "row"), owner="placement"),
)
"""The row store: every world line's rows, the dense class's (``[0, dense_max_constraints)``) then the free bodies'
(from ``body_base``): the system and the row records (with each row's sides). One row index for both classes: a
contact's slot and its rows' parents count in the line. ``row_incident`` is each row's
incident velocity ``J v_hat``, written where its walk sums it: the one writer (:mod:`.walk_rows`), the free bodies'
writer (:func:`.walk_rows.write_free_body_rows`)."""

ROW_RECORDS = FieldGroup(Field("row_record", wp.int32, ("world", "record"), owner="placement"))
"""The streamed rows' records, which the sweeps' row loops load a row ahead, packed each step from the row store
(:func:`.response_builds_rows.row_record_kernel`, :func:`.row_metadata.pack_row_record`): eight ints a row, the line's
rows from the plan's first streamed row (the dense rows' where a sweep streams their metadata, else the free bodies'
from ``body_base``, which always stream). A free row's record keeps the inverse of its effective mass ``A_ii``; its
rows themselves are the line's: their records the row store's (:data:`DENSE_ROWS`, from ``body_base``), their sides
their responding bodies' articulations in the rows' ``a``/``b`` order (-1 for none), and their whitened
``Z = L^-1 J^T`` (``L`` the body's held factor: two one-node trees of six) the packed rows'
(:data:`.packed_rows.PACKED_ROWS`, the line's rows at the world's support)."""


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
UNIT_SPEED = wp.constant(6)
"""A free body's speed row: index ``2 dof``, ``dof`` the first coordinate of the velocity it bounds (its linear
velocity's or its angular one's), seeded ``+e_dof`` as a velocity limit's lower side; the writer stores the axis
unwhitened (:func:`.walk_rows.speed_axis`)."""
UNIT_SHIFT = wp.constant(28)
"""Bits of a unit source's index: its kind sits above them."""


@wp.func
def unit_source(kind: int, index: int) -> int:
    """A row's unit source: what the one writer (:mod:`.walk_rows`) reads to find the row's law and seed, its kind
    (``UNIT_*``) above :data:`UNIT_SHIFT` bits of index."""
    return (kind << UNIT_SHIFT) | index


@wp.struct
class RowSupport:
    """Where a dense row's entries sit in the packed rows (:mod:`.packed_rows`), and the seed the one writer walks into
    them (:mod:`.walk_rows`). A row of world ``w`` is ``Z[first[w], row, :]``: its sides (``sides``: :func:`row_sides`),
    each articulation's coordinates (``dofs``) and first world coordinate (``offset``), and the layout (``dense``: entry
    ``e`` is world coordinate ``e``; else the first side's entries, then the second's). A producer writes no entry: it
    records each row's unit source (``units``, :func:`unit_source`), by world line and row, from which the writer
    takes its seed."""

    sides: wp.array2d[wp.vec2i]
    dofs: wp.array[int]
    offset: wp.array[int]
    dense: int
    first: wp.array[int]
    units: wp.array2d[int]


MF_RECORDS = (
    *(
        dataclasses.replace(DENSE_ROWS[attr], name=name, attr=attr, owner="placement")
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
    Field("eps", wp.float32, ("world", "row"), attr="row_eps", owner="placement"),
    Field("first", wp.int32, ("world",), attr="dense_first", owner="placement"),
    Field("base", wp.int32, attr="body_base", owner="placement"),
)
"""Each free-body row's records, the row store's at row ``base + i`` of the world's line (one row index
for both classes: a free-body contact's slot, its rows' parents): kind, friction parent, coefficient, the row law's
beta, gap, restitution, incident velocity (``J v_hat``, stored by the build), rhs, target, effective mass ``A_ii``,
material compliance and sides (the responding bodies' articulations in ``a``/``b`` order, -1 for none)."""

MfRows = field_group.struct("MfRows", MF_RECORDS, __name__)


def bind_mf(placement):
    """Bind the placement's free-body rows into an :data:`MfRows`: their placement and the row store's records."""
    return field_group.bind(MfRows, placement=placement)


def bind(placement, data=None, options=None):
    """Bind the placement's dense row records and the law's ``cfm`` (``options``, by default the placement's) into a
    :data:`DenseRows` (``data``: an instance to refill)."""
    return field_group.bind(
        DenseRows, data, placement=placement, options=placement.options if options is None else options
    )
