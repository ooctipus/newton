# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The dense row records every row producer writes, declared once as fields (:mod:`.field_group`).

:data:`DenseRows` is their Warp struct; the contact producers' input
(:data:`.contact_input.ContactInput`) carries the same fields.
"""

import warp as wp

from . import field_group
from .field_group import Field

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


def bind(solver, data=None):
    """Bind the solver's dense row records into a :data:`DenseRows` (``data``: an instance to refill)."""
    return field_group.bind(DenseRows, data, solver=solver)
