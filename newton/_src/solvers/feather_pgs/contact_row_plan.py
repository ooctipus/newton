# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Which producer writes a FeatherPGS solver's contact rows, and which builder each DOF size takes.

The plan reads host state only and launches nothing. The row pass
(:meth:`.rows.RowLaws.write`) dispatches on it and ``get_path_selection()``
records it under ``routes.contact_rows``, so the record names the builders
that launch.
"""

import dataclasses

# Producers that write the row metadata with prepare_world_contact_rows and build each size's Jacobian.
SHARED_METADATA_PRODUCERS = ("compact_jacobian",)

# Every builder method a plan can name, by producer; the route record names each one's kernels.
BUILDER_METHODS = (
    "walk",
    "local_packets",
    "keyed_rows",
    "compact_contact_boundary",
    "compact_jacobian",
    "per_contact",
    "masked",
    "masked_combined",
)


@dataclasses.dataclass(frozen=True)
class ContactRowPlan:
    """The contact-row producer of one solver and the builder method of each DOF size."""

    producer: str
    """The producer the class's placement case names: ``walk`` (the one writer: the general placement's, the support
    placement's, case K1, and the sparse diagonal's, case K2), ``local_packets``, ``keyed_rows``,
    ``compact_contact_boundary``, ``compact_jacobian`` or ``general``."""
    law: str
    """The friction law of the contact rows: ``"point"`` or ``"patch"``."""
    methods: dict[int, str]
    """Builder method by DOF size, in ``size_groups`` order."""
    combined: tuple[int, ...] = ()
    """General producer: the sizes whose rows one masked launch builds, in ``size_groups`` order."""
    metadata_launch: bool = False
    """Shared-metadata producers: a separate ``prepare_world_contact_rows`` launch writes the row metadata."""


def contact_row_plan(solver) -> ContactRowPlan:
    """Decide the contact-row producer and each DOF size's builder method.

    The class's placement case (``solver.routes.case``) names the producer that writes every contact row: local row
    packets for the local owners (K3), keyed rows for the tiers (K4) and the compact contact boundary for the sparse
    diagonal (K2), each where its own fit admitted it; then the one writer where the class walks its rows (K6, K1 and
    the rest of the sparse diagonal), the compact Jacobian and last the general builder.
    Owners decided at construction admit the producers, so the plan stays fixed for a solver; it is evaluated from
    the solver's current attributes.

    Builder methods by producer:

    - ``local_packets``, ``keyed_rows``, ``compact_contact_boundary``: the owner's own builder.
    - ``walk``: the row pass writes each contact row's records and unit source; the one writer walks it.
    - Shared-metadata producers: ``compact_jacobian`` for every size.
    - ``general``: sizes of at most 32 DOFs take the masked builder (``masked_combined``, one launch
      for one or two such sizes; ``masked``, one launch per size for three or more); wider sizes take
      ``per_contact``.
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
    if solver._compact_contact_jacobian:
        return ContactRowPlan("compact_jacobian", law, dict.fromkeys(sizes, "compact_jacobian"), metadata_launch=True)
    masked_sizes = tuple(size for size in sizes if size <= 32)
    combined = masked_sizes if 1 <= len(masked_sizes) <= 2 else ()
    methods = {}
    for size in sizes:
        if size in combined:
            methods[size] = "masked_combined"
        elif size in masked_sizes:
            methods[size] = "masked"
        else:
            methods[size] = "per_contact"
    return ContactRowPlan("general", law, methods, combined=combined)
