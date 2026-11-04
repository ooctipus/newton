# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Which producer writes a FeatherPGS solver's contact rows.

The plan reads host state only and launches nothing. The row pass
(:meth:`.rows.RowLaws.write_contacts`) dispatches on it and ``get_path_selection()``
records it under ``routes.contact_rows``, so the record names the builders
that launch.
"""

import dataclasses

# Every builder method a plan can name; the route record names each one's kernels.
BUILDER_METHODS = (
    "walk",
    "local_packets",
    "keyed_rows",
    "compact_contact_boundary",
)


@dataclasses.dataclass(frozen=True)
class ContactRowPlan:
    """The contact-row producer of one solver and the builder method of each DOF size."""

    producer: str
    """The producer the class's placement case names: ``walk`` (the one writer: the general placement's, the support
    placement's, case K1, and the sparse diagonal's, case K2), ``local_packets``, ``keyed_rows`` or
    ``compact_contact_boundary``; ``none`` where the class walks no rows."""
    law: str
    """The friction law of the contact rows: ``"point"`` or ``"patch"``."""
    methods: dict[int, str]
    """Builder method by DOF size, in ``size_groups`` order."""


def contact_row_plan(solver) -> ContactRowPlan:
    """Decide the contact-row producer and each DOF size's builder method.

    The class's placement case (``solver.routes.case``) names the producer that writes every contact row: local row
    packets for the local owners (K3), keyed rows for the tiers (K4) and the compact contact boundary for the sparse
    diagonal (K2), each where its own fit admitted it; else the one writer, where the class walks its rows (K6, K1 and
    the rest of the sparse diagonal). A class that walks no rows has no responding coordinates (the placement's
    ``walks``): the allocator admits none of its contacts, and no producer runs (``none``).
    Owners decided at construction admit the producers, so the plan stays fixed for a solver; it is evaluated from
    the solver's current attributes.
    """
    law = "patch" if solver.laws.friction_anchors_enabled else "point"
    sizes = solver.forest.size_groups
    case = solver.routes.case
    owned = {
        "K3": "local_packets" if solver.placement.row_packets is not None else None,
        "K4": "keyed_rows" if solver.placement.keyed_rows is not None else None,
        "K2": "compact_contact_boundary" if solver.placement.compact_contact_boundary else None,
    }.get(case)
    if owned is not None:
        # An in-kernel owner's producer; where its class walks (the local packets) it records the general worlds'
        # unit sources too, which the one writer walks.
        return ContactRowPlan(owned, law, dict.fromkeys(sizes, owned))
    if getattr(solver.placement, "walks", False):
        # The one writer (walk_rows): the row pass writes each contact row's records and unit source, and the writer
        # walks its seed into the class's sink (the general placement's packed rows, the support placement's support,
        # the sparse diagonal's slots).
        return ContactRowPlan("walk", law, dict.fromkeys(sizes, "walk"))
    return ContactRowPlan("none", law, {})
