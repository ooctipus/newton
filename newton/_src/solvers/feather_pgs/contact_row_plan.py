# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Which producer writes a FeatherPGS solver's contact rows, and which builder each DOF size takes.

The plan reads host state only and launches nothing. ``_stage4_build_rows``
dispatches on it and ``get_path_selection()`` records it under
``routes.contact_rows``, so the record names the builders that launch.
"""

import dataclasses

# Producers that write the row metadata with prepare_world_contact_rows (unless native sparse
# factors prepare their rows in the response kernel) and build each size's response or Jacobian.
SHARED_METADATA_PRODUCERS = ("sparse_contact_response", "sparse_diagonal", "compact_jacobian")

# General-builder methods whose one launch builds the rows of up to two sizes.
COMBINED_METHODS = ("masked_combined", "fused_rows")

# Every builder method a plan can name, by producer; the route record names each one's kernels.
BUILDER_METHODS = (
    "local_packets",
    "keyed_rows",
    "compact_contact_boundary",
    "sparse_response",
    "sparse_free_response",
    "sparse_diagonal",
    "compact_jacobian",
    "per_contact",
    "masked",
    "masked_combined",
    "fused_rows",
)


@dataclasses.dataclass(frozen=True)
class ContactRowPlan:
    """The contact-row producer of one solver and the builder method of each DOF size."""

    producer: str
    """The first admitted producer: ``local_packets``, ``keyed_rows``, ``compact_contact_boundary``,
    ``sparse_contact_response``, ``sparse_diagonal``, ``compact_jacobian`` or ``general``."""
    law: str
    """The friction law of the contact rows: ``"point"`` or ``"patch"``."""
    methods: dict[int, str]
    """Builder method by DOF size, in ``size_groups`` order."""
    combined: tuple[int, ...] = ()
    """General producer: the sizes whose rows one masked or fused launch builds, in ``size_groups`` order."""
    check: str | None = None
    """General producer, debug only: the builder a row check compares (``FEATHER_PGS_CHECK_ROWS``:
    ``"masked"`` into copies, then the per-contact rows; ``FEATHER_PGS_CHECK_ROWS_FUSED``:
    ``"fused_rows"`` after the combined masked rows)."""
    metadata_launch: bool = False
    """Shared-metadata producers: a separate ``prepare_world_contact_rows`` launch writes the row metadata.

    False for native sparse factors whose response kernel prepares the rows itself. The sparse response
    factory's ``prepare_rows`` option has the opposite sense: ``True`` selects that kernel (suffix
    ``_prepared``), which needs no metadata launch."""


def contact_row_plan(solver) -> ContactRowPlan:
    """Decide the contact-row producer and each DOF size's builder method.

    The first admitted producer writes every contact row: local row packets, keyed rows, the
    compact contact boundary, then the shared-metadata producers (native sparse factors' response,
    the sparse-diagonal owner, the compact Jacobian) and last the general builder. Owners decided at
    construction admit the producers, so the plan stays fixed for a solver; it is evaluated from the
    solver's current attributes and the solver module's import-time switches.

    Builder methods by producer:

    - ``local_packets``, ``keyed_rows``, ``compact_contact_boundary``: the owner's own builder.
    - Shared-metadata producers: ``sparse_response`` for the factored size, ``sparse_free_response``
      for the free-body size, whose response the sparse kernel writes; ``sparse_diagonal`` for the
      diagonal-mass size; ``compact_jacobian`` for every other size.
    - ``general``: ``fused_rows`` or ``masked_combined`` (one launch for the one or two sizes of at
      most 32 DOFs), ``masked`` (one launch per size), else ``per_contact``.
    """
    from . import solver_feather_pgs as source  # noqa: PLC0415 - the solver module's import-time switches

    law = "patch" if solver._friction_anchors_enabled else "point"
    sizes = solver.size_groups
    if solver._row_packets is not None:
        return ContactRowPlan("local_packets", law, dict.fromkeys(sizes, "local_packets"))
    if solver._keyed_rows is not None:
        return ContactRowPlan("keyed_rows", law, dict.fromkeys(sizes, "keyed_rows"))
    if solver._compact_contact_boundary:
        return ContactRowPlan("compact_contact_boundary", law, dict.fromkeys(sizes, "compact_contact_boundary"))
    sparse_size = solver._sparse_mass_matrix_size
    diagonal = bool(solver._sparse_diagonal_contact_solve)
    if sparse_size is not None and diagonal:
        # _setup_sparse_mass_matrix declines native sparse factors where the sparse-diagonal owner owns the
        # response, so one producer ranks first; the row build launches the diagonal response for that owner only.
        raise RuntimeError("Native sparse factors and the sparse-diagonal contact owner cannot both own contact rows.")
    if sparse_size is not None or diagonal or solver._compact_contact_jacobian:
        methods = {}
        for size in sizes:
            if sparse_size is not None:
                # Native sparse factors admit one articulated size; the other is the free bodies'.
                methods[size] = "sparse_response" if size == sparse_size else "sparse_free_response"
            elif diagonal and size == solver._sparse_diagonal_response_size:
                methods[size] = "sparse_diagonal"
            else:
                methods[size] = "compact_jacobian"
        if sparse_size is not None:
            producer = "sparse_contact_response"
        elif diagonal:
            producer = "sparse_diagonal"
        else:
            producer = "compact_jacobian"
        return ContactRowPlan(producer, law, methods, metadata_launch=not solver._sparse_prepare_contacts)
    check_rows = bool(solver._point_contact_paths and source._CHECK_ROWS)
    masked_law = solver._masked_contact_rows()
    masked_sizes = ()
    if masked_law and source._ROWS_MASKED and not check_rows:
        masked_sizes = tuple(size for size in sizes if size <= 32)
    combined = masked_sizes if 1 <= len(masked_sizes) <= 2 else ()
    methods = {}
    for size in sizes:
        if size in combined:
            methods[size] = "fused_rows" if source._ROWS_FUSED and not source._CHECK_ROWS_FUSED else "masked_combined"
        elif masked_law and source._ROWS_MASKED and size <= 32 and not check_rows:
            methods[size] = "masked"
        else:
            methods[size] = "per_contact"
    if check_rows:
        check = "masked"
    elif combined and source._CHECK_ROWS_FUSED:
        check = "fused_rows"
    else:
        check = None
    return ContactRowPlan("general", law, methods, combined=combined, check=check)
