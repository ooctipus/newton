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
from .solver_fields import MF_BODY_MAP, MF_ROWS

RECORDS = (
    Field("row_type", wp.int32, ("world", "row")),
    Field("row_parent", wp.int32, ("world", "row")),
    Field("row_mu", wp.float32, ("world", "row")),
    Field("row_beta", wp.float32, ("world", "row")),
    Field("row_cfm", wp.float32, ("world", "row")),
    Field("phi", wp.float32, ("world", "row")),
    Field("target", wp.float32, ("world", "row"), attr="target_velocity"),
    Field("restitution", wp.float32, ("world", "row"), attr="row_restitution"),
    Field("rhs", wp.float32, ("world", "row")),
    Field("diag", wp.float32, ("world", "row")),
    Field("row_w", wp.float32, ("world", "row")),
)
"""Each dense row's kind, friction parent, coefficients, gap, targets and law outputs, by world and row."""

DenseRows = field_group.struct("DenseRows", RECORDS, __name__)


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
    Field("target", wp.float32, ("world", "mf_row"), attr="mf_target_velocity"),
)
"""Each matrix-free row's bodies, coordinates and per-body blocks (the writers' ``J``, whitened in place to ``Z``
by the rows' build), and the dense records' kind, friction parent, coefficient, gap, restitution, incident velocity
(``J v_hat`` of a restitutive contact, stored by the build), rhs and target, by world and row."""

MfRows = field_group.struct("MfRows", MF_RECORDS, __name__)


MfBodyMap = field_group.struct(
    "MfBodyMap",
    (
        dataclasses.replace(MF_BODY_MAP[attr], name=attr.removeprefix("mf_"), attr=attr)
        for attr in ("mf_body_list", "mf_body_dof_start", "mf_body_count", "mf_local_body_a", "mf_local_body_b")
    ),
    __name__,
)
"""Each world's matrix-free bodies in row order, their coordinate starts and count, and every row's two local body
slots (split mode's tiled sweep)."""


def bind_mf_body_map(solver):
    """Bind the solver's matrix-free body map into an :data:`MfBodyMap`."""
    return field_group.bind(MfBodyMap, solver=solver)


def bind_mf(solver, **records):
    """Bind the solver's matrix-free row records into an :data:`MfRows` (``records`` replace records by name)."""
    return field_group.bind(MfRows, solver=solver, **records)


def bind(solver, data=None, **records):
    """Bind the solver's dense row records into a :data:`DenseRows` (``data``: an instance to refill).

    ``records`` replace records by field name, such as another ``rhs`` for the velocity pass.
    """
    return field_group.bind(DenseRows, data, solver=solver, **records)
