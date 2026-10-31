# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The row records every row producer writes, declared once as fields (:mod:`.field_group`).

:data:`DenseRows` is the Warp struct of the dense rows; the contact producers' input
(:data:`.contact_input.ContactInput`) carries the same fields. :data:`MfRows` is the struct of the
matrix-free (free-body) rows, under the same record names.
"""

import dataclasses

import warp as wp

from . import field_group
from .field_group import Field
from .solver_fields import MF_ROWS

RECORDS = (
    Field("row_type", wp.int32, ("world", "row"), owner="placement"),
    Field("row_parent", wp.int32, ("world", "row"), owner="placement"),
    Field("row_mu", wp.float32, ("world", "row"), owner="placement"),
    Field("row_beta", wp.float32, ("world", "row"), owner="placement"),
    Field("row_cfm", wp.float32, ("world", "row"), owner="placement"),
    Field("phi", wp.float32, ("world", "row"), owner="placement"),
    Field("target", wp.float32, ("world", "row"), attr="target_velocity", owner="placement"),
    Field("restitution", wp.float32, ("world", "row"), attr="row_restitution", owner="placement"),
    Field("rhs", wp.float32, ("world", "row"), owner="placement"),
    Field("diag", wp.float32, ("world", "row"), owner="placement"),
    Field("row_w", wp.float32, ("world", "row"), owner="placement"),
    Field("row_eps", wp.float32, ("world", "row"), owner="placement"),
    Field("sides", wp.vec2i, ("world", "row"), attr="row_sides", owner="placement"),
)
"""Each dense row's kind, friction parent, coefficients, gap, targets and law outputs (a material contact's
compliance ``row_eps`` among them), and its sides (the articulations whose response coordinates it touches:
:func:`row_sides`), by world and row."""

DenseRows = field_group.struct("DenseRows", RECORDS, __name__)


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
    the packed rows' entries, the sweeps' factor coordinates (:class:`.sweep_routes.FactorCoordinates` marks a side
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


@wp.struct
class RowSupport:
    """Where a dense row's Jacobian entries sit. In the matrix-free sweeps' packed rows (``packed``,
    :mod:`.packed_rows`) a row of world ``w`` is ``J[w, row, :]``: its sides (``sides``: :func:`row_sides`), each
    articulation's coordinates (``dofs``), first world coordinate (``offset``) and world (``world``), and the layout
    (``dense``: entry ``e`` is world coordinate ``e``; else the first side's entries, then the second's). Otherwise a
    size group's rows hold each articulation's side from entry 0, row ``J[group, row, :]``."""

    sides: wp.array2d[wp.vec2i]
    dofs: wp.array[int]
    offset: wp.array[int]
    world: wp.array[int]
    dense: int
    packed: int


@wp.func
def jacobian_row(support: RowSupport, group: int, art: int, sides: wp.vec2i) -> wp.vec2i:
    """Where articulation ``art``'s side of a row with ``sides`` starts in the Jacobian rows a producer writes: the
    rows' first index and the side's first entry (``J[at[0], row, at[1] + e]`` is the side's coordinate ``e``). The
    packed rows (:class:`RowSupport`) hold it at its world coordinates (dense) or after the row's first side; a size
    group's rows at its group, from entry 0."""
    if support.packed == 0:
        return wp.vec2i(group, 0)
    world = support.world[art]
    if support.dense != 0:
        return wp.vec2i(world, support.offset[art])
    if art == sides[0]:
        return wp.vec2i(world, 0)
    return wp.vec2i(world, support.dofs[sides[0]])


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
            ("body_a", "mf_body_a"),
            ("body_b", "mf_body_b"),
            ("dof_a", "mf_dof_a"),
            ("dof_b", "mf_dof_b"),
            ("Z_a", "mf_Z_a"),
            ("Z_b", "mf_Z_b"),
            ("row_type", "mf_row_type"),
            ("row_parent", "mf_row_parent"),
            ("row_mu", "mf_row_mu"),
            ("phi", "mf_phi"),
            ("restitution", "mf_row_restitution"),
            ("incident", "mf_row_incident"),
            ("rhs", "mf_rhs"),
        )
    ),
    Field("target", wp.float32, ("world", "mf_row"), attr="mf_target_velocity", owner="placement"),
    Field("eps", wp.float32, ("world", "mf_row"), attr="mf_row_eps", owner="placement"),
)
"""Each matrix-free row's bodies, coordinates and per-body blocks (the writers' ``J``, whitened in place to ``Z``
by the rows' build), and the dense records' kind, friction parent, coefficient, gap, restitution, incident velocity
(``J v_hat`` of a restitutive contact, stored by the build), rhs, target and material compliance, by world and
row."""

MfRows = field_group.struct("MfRows", MF_RECORDS, __name__)


def bind_mf(placement):
    """Bind the placement's matrix-free row records into an :data:`MfRows`."""
    return field_group.bind(MfRows, placement=placement)


def bind(placement, data=None):
    """Bind the placement's dense row records into a :data:`DenseRows` (``data``: an instance to refill)."""
    return field_group.bind(DenseRows, data, placement=placement)
