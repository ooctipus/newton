# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Which producer writes a FeatherPGS solver's contact rows, and which builder each DOF size takes.

The plan reads host state only and launches nothing. The row pass
(:meth:`.rows.RowLaws.write`) dispatches on it and ``get_path_selection()``
records it under ``routes.contact_rows``, so the record names the builders
that launch.
"""

import dataclasses

# Producers that write the row metadata with prepare_world_contact_rows (unless native sparse
# factors prepare their rows in the response kernel) and build each size's response or Jacobian.
SHARED_METADATA_PRODUCERS = ("sparse_contact_response", "sparse_diagonal", "compact_jacobian")

# Every builder method a plan can name, by producer; the route record names each one's kernels.
BUILDER_METHODS = (
    "walk",
    "local_packets",
    "keyed_rows",
    "compact_contact_boundary",
    "sparse_response",
    "sparse_diagonal",
    "compact_jacobian",
    "per_contact",
    "masked",
    "masked_combined",
)


@dataclasses.dataclass(frozen=True)
class ContactRowPlan:
    """The contact-row producer of one solver and the builder method of each DOF size."""

    producer: str
    """The producer the class's placement case names: ``walk`` (the general placement's one writer),
    ``local_packets``, ``keyed_rows``, ``compact_contact_boundary``, ``sparse_contact_response``, ``sparse_diagonal``,
    ``compact_jacobian`` or ``general``."""
    law: str
    """The friction law of the contact rows: ``"point"`` or ``"patch"``."""
    methods: dict[int, str]
    """Builder method by DOF size, in ``size_groups`` order."""
    combined: tuple[int, ...] = ()
    """General producer: the sizes whose rows one masked launch builds, in ``size_groups`` order."""
    metadata_launch: bool = False
    """Shared-metadata producers: a separate ``prepare_world_contact_rows`` launch writes the row metadata.

    False for native sparse factors whose response kernel prepares the rows itself. The sparse response
    factory's ``prepare_rows`` option has the opposite sense: ``True`` selects that kernel (suffix
    ``_prepared``), which needs no metadata launch."""


def contact_row_plan(solver) -> ContactRowPlan:
    """Decide the contact-row producer and each DOF size's builder method.

    The class's placement case (``solver.routes.case``) names the producer that writes every contact row: local row
    packets for the local owners (K3), keyed rows for the tiers (K4) and the compact contact boundary for the
    sparse diagonal (K2), each where its own fit admitted it; then the shared-metadata producers (native sparse
    factors' response for K1, the sparse-diagonal rows for K2, the compact Jacobian) and last the general builder.
    Owners decided at construction admit the producers, so the plan stays fixed for a solver; it is evaluated from
    the solver's current attributes.

    Builder methods by producer:

    - ``local_packets``, ``keyed_rows``, ``compact_contact_boundary``: the owner's own builder.
    - Shared-metadata producers: ``sparse_response`` for the factored size, the only size of a model
      native sparse factors admit; ``sparse_diagonal`` for the diagonal-mass size; ``compact_jacobian``
      for every other size.
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
        # The general placement's one writer (walk_rows): the row pass writes each contact row's records and unit
        # source, and the writer walks its seed.
        return ContactRowPlan("walk", law, dict.fromkeys(sizes, "walk"))
    if case in ("K1", "K2") or solver._compact_contact_jacobian:
        methods = {}
        for size in sizes:
            if case == "K1":
                # Native sparse factors admit one articulated size and no free rigid bodies.
                methods[size] = "sparse_response"
            elif case == "K2" and size == solver.placement.sparse_diagonal_response_size:
                methods[size] = "sparse_diagonal"
            else:
                methods[size] = "compact_jacobian"
        producer = {"K1": "sparse_contact_response", "K2": "sparse_diagonal"}.get(case, "compact_jacobian")
        return ContactRowPlan(producer, law, methods, metadata_launch=not solver.placement.sparse_prepare_contacts)
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
