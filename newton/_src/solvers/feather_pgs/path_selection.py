# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host-only records of the construction-time routes and applied options of SolverFeatherPGS.

Every function here reads host attributes only: no device array is read, no
kernel is launched and nothing synchronizes. The one exception is
:func:`measurement_selection` under an enabled solve readout
(``pgs_solve_readout``), which reads the readout's device counters. Values are
JSON-compliant: non-finite floats become ``"inf"``, ``"-inf"`` or ``"nan"`` and
tuples become lists.
"""

import copy
import functools
import inspect
import math

import numpy as np

from ...sim.contacts import CAPACITY_BLOCKS_PER_SM
from .contact_row_plan import contact_row_plan
from .friction_patches import _CHUNK_SORT_CAPACITY, _LIVE_SORT_CAPACITY, _SORT_SLOT_BITS

# Options whose applied value lives under another attribute after construction.
_APPLIED_ATTRIBUTES = {
    "double_buffer": "_double_buffer",
    "nvtx": "_nvtx",
    "row_watermark": "_row_watermark",
    "mf_warmstart": "_legacy_mf_warmstart",
    "mf_warmstart_decay": "_legacy_mf_warmstart_decay",
    "enable_bilateral_preelimination": "_preelim_active",
}

_BETA_REASONS = {
    "default": "friction_anchor_beta omitted: persistent patch friction at the 0.2 default.",
    "default_point_solver": "friction_anchor_beta omitted with a coupled friction_mode: point friction.",
    "default_compliance": "friction_anchor_beta omitted with contact_compliance: point friction.",
    "argument": "friction_anchor_beta passed explicitly.",
}

# Which claim wins where two patch routes could serve the same world: the order of the
# declines in _setup_sparse_mass_matrix, the paired factor sweep's exclusions (keyed_rows,
# row_packets and the tier gate of _init_tiled_kernels) and the order of the row producers
# in contact_row_plan (which _stage4_build_rows dispatches on); and which native sparse
# sweeps fuse patch anchor triplets (_admit_sparse_contact_triplets). It governs
# law-agnostic solvers only.
_PATCH_ROUTE_PRECEDENCE = (
    " Patch routes, first claim first: a parallel tier that mf_gs_parallel_rows requests (unless"
    " FEATHER_PGS_PATCH_PARALLEL_ROWS=0) declines native sparse factors, worlds with free rigid bodies then keep the"
    " dense route, and native sparse factors take the worlds whose topology"
    " they admit, own their rows and may select patch rows at sweep entry (sparse_mass_matrix.active_rows,"
    " FEATHER_PGS_PATCH_ACTIVE_ROWS). Where native sparse factors admit contact triplets"
    " (sparse_mass_matrix.contact_triplets, FEATHER_PGS_SPARSE_CONTACT_TRIPLETS=1), they fuse patch anchor units,"
    " pooling each ring's normal loads, in the linked active scan and in the full ordered sweep, by default in the"
    " single-stage transaction (sparse_mass_matrix.contact_transaction, FEATHER_PGS_SPARSE_TRANSACTION)."
    " On the dense route an admitted paired factor sweep"
    " (paired_response.factor_coordinates) excludes the parallel tiers, keyed_rows and local_packets, even a tier"
    " whose request declined sparse factors, and walks patch contact units (paired_response.contact_units)."
    " The row producer (contact_rows.producer) is the first admitted of"
    " local_packets, keyed_rows (unless FEATHER_PGS_PATCH_PARALLEL_ROWS=0), the compact and sparse producers and"
    " general, whose masked rows take the patch row law;"
    " contact_rows.builder_by_dof names each size's builder kernels and the law they write."
)
# Contact torsion (patch friction does not admit compliance) takes none of those routes.
_ORDINARY_PATCH_ROUTES = (
    " Contact torsion and compliance keep ordinary factors, rows and serial sweeps, so"
    " FEATHER_PGS_PATCH_ACTIVE_ROWS, FEATHER_PGS_PATCH_PARALLEL_ROWS and"
    " FEATHER_PGS_SPARSE_CONTACT_TRIPLETS select no route."
)


def json_value(value):
    """Return a JSON-compliant host copy of a scalar, sequence or mapping without reading device memory."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    # Never str(): a Warp array's str() copies device memory to the host.
    return f"<{type(value).__name__}>"


@functools.cache
def _declared_defaults(init) -> dict:
    parameters = list(inspect.signature(init).parameters.values())[2:]  # after self and model
    return {parameter.name: parameter.default for parameter in parameters}


def _is_default(value, default) -> bool:
    if value is default:
        return True
    try:
        same = value == default
    except Exception:
        return False
    return bool(same) if isinstance(same, (bool, np.bool_)) else False


def _streams_created(solver) -> bool:
    # Every stream use_parallel_streams gates: size groups, local owners, the compact inertia
    # refresh, asynchronous augmented drives and the paired overlap. CPU builds none.
    names = ("_local_internal_stream", "_global_inertia_stream", "_articulation_dynamics_stream")
    streams = (
        *solver._size_streams.values(),
        *(getattr(solver, name, None) for name in names),
        getattr(solver, "_paired_general_solve_stream", None),
    )
    return any(stream is not None for stream in streams)


def _linked_tier_sweeps(solver) -> int:
    if solver._ink_rows <= 0:
        return 0
    linked = [kernel for kernel in solver._pgs_solve_mf_gs_incremental_kernels if kernel._fpgs_linked_rows]
    return max((int(kernel._fpgs_sweeps) for kernel in linked), default=0)


# Options whose request can resolve to another value, or to no owner or tier at all.
_EFFECTIVE = {
    "sparse_mass_matrix": lambda solver: "on" if solver._sparse_mass_matrix_size is not None else "off",
    "cached_dynamics": lambda solver: "on" if solver._tree_kinetic_state is not None else "off",
    "parallel_tree": lambda solver: solver._tree_plan is not None,
    "grouped_dynamics": lambda solver: solver._grouped_topology is not None,
    # Deferred publication acts only with the fused world-dynamics kernel (K1), and not under patch friction.
    "lazy_kinematics": lambda solver: solver._publication_deferred(),
    # Tier budgets after their gates and clamps.
    "mf_gs_parallel_rows": lambda solver: int(solver._ink_rows),
    "mf_gs_parallel_matrix_free": lambda solver: bool(solver.mf_gs_parallel_matrix_free and solver._ink_rows > 0),
    "mf_gs_incremental_rows": lambda solver: int(solver._incremental_rows),
    # Deprecated: no kernel reads it.
    "pgs_chunk_size": lambda solver: None,
    "use_parallel_streams": _streams_created,
    "propagation_cached_response": lambda solver: bool(solver._propagation_cached_response_active),
    "mf_gs_parallel_linked_velocity_limit_pass": lambda solver: bool(
        solver._ink_rows > 0
        and any(
            getattr(kernel, "_fpgs_velocity_limit_pass", False)
            for kernel in solver._pgs_solve_mf_gs_incremental_kernels
        )
    ),
    # The sweeps' law after the schedule and row gates (routes.velocity_limits.law): None in split mode, which
    # builds no matrix-free sweep and applies no law.
    "velocity_limit_law": lambda solver: solver._velocity_limit_sweep_law()[0],
    # The largest budget of the built linked (patch) tiers; 0 without them.
    "mf_gs_parallel_linked_sweeps": _linked_tier_sweeps,
    # A readout applies only where a sweep kernel carries its counters, the velocity law where a sweep compiles it.
    "pgs_solve_readout": lambda solver: solver.pgs_solve_readout if solver._solve_readout.kernels else "off",
    "pgs_velocity_change_tolerance": lambda solver: (
        solver.pgs_velocity_change_tolerance if solver._solve_policy_honored else 0.0
    ),
    # "unit" where some sweep solves contact units exactly (routes.sweep.contact_update), else "sequential".
    "contact_update": lambda solver: (
        "unit" if solver._dense_contact_unit or solver._free_body_contact_unit else "sequential"
    ),
}


def applied_options(solver, init, arguments: dict) -> dict:
    """Record every constructor option's requested, declared-default and applied value.

    ``arguments`` is ``locals()`` taken as the constructor's first statement.
    Call once, after construction. ``applied`` is the value in effect after
    construction's gates and clamps: an option whose owner, tier or stream was
    not built reports its off value (``False``, ``0`` or ``"off"``), and a row
    budget reports the rows the built tier serves. ``requested`` keeps the
    request, which can still decline other owners when its own mechanism does
    not engage (for example ``grouped_dynamics`` and ``lazy_kinematics``);
    ``routes`` names the owners. ``source`` is ``"environment"`` when an
    ``IL_NEWTON_FPGS_*`` legacy alias or a ``FEATHER_PGS_*`` override changed
    the applied value, ``"argument"`` when the requested value differs from the
    declared default and ``"default"`` otherwise; Python cannot tell an omitted
    option from one passed with its default value.
    """
    record = {}
    environment = getattr(solver, "_environment_options", frozenset())
    for name, default in _declared_defaults(init).items():
        value = arguments.get(name, default)
        effective = _EFFECTIVE.get(name)
        if effective is not None:
            applied = effective(solver)
        else:
            applied = getattr(solver, _APPLIED_ATTRIBUTES.get(name, name), value)
        if name in environment:
            source = "environment"
        else:
            source = "default" if _is_default(value, default) else "argument"
        record[name] = {
            "applied": json_value(applied),
            "requested": json_value(value),
            "default": json_value(default),
            "source": source,
        }
    return record


def options_selection(solver) -> dict:
    """Return a detached copy of the options recorded at construction."""
    return copy.deepcopy(solver._applied_options)


def _key(kernel):
    return None if kernel is None else getattr(kernel, "key", type(kernel).__name__)


def _keys_by_size(kernels) -> dict:
    return {str(int(size)): _key(kernel) for size, kernel in kernels.items() if kernel is not None}


def _counts(counts) -> dict:
    return {str(int(size)): int(count) for size, count in counts.items() if count}


def route_selection(solver) -> dict:
    """Describe every construction-time owner and variant that decides which kernels run."""
    from . import solver_feather_pgs as source  # noqa: PLC0415 - the solver module's import-time flags

    return {
        "articulations_by_dof": {str(int(size)): int(count) for size, count in solver.n_arts_by_size.items()},
        "held_tree": _held_tree(solver),
        "contact_rows": _contact_rows(solver, source),
        "contact_consumers": _contact_consumers(solver),
        "local_solve": _local_solve(solver),
        "paired_response": _paired_response(solver),
        "sparse_diagonal": _sparse_diagonal(solver),
        "sparse_mass_matrix": _sparse_mass_matrix(solver),
        "sweep": _sweep(solver, source),
        "resolved_worlds": _resolved_worlds(solver),
        "publication": _publication(solver, source),
        "velocity_limits": _velocity_limits(solver),
        "free_rigid_bodies": _free_rigid_bodies(solver),
        "friction": _friction(solver),
        "dynamics": _dynamics(solver, source),
        "diagnostics": _diagnostics(solver),
    }


def measurement_selection(solver) -> dict:
    """Measured values, kept out of the routes so that runs of one configuration compare equal.

    ``held_tree_lane_timing_us`` holds the construction-time lane timings. With
    ``pgs_solve_readout`` on, ``solve_readout`` summarizes the sweep counters
    (:meth:`SolveReadoutRecord.summary`); reading them synchronizes the device.
    The counters accumulate, so a window is the difference of two records.
    """
    _, groups = _held_tree_groups(solver)
    record = {
        # Aligned with routes.held_tree.groups; empty for a group whose lane width was not timed.
        "held_tree_lane_timing_us": [
            {str(int(lanes)): round(float(seconds) * 1.0e6, 3) for lanes, seconds in group.lane_timing.items()}
            for group in groups
        ],
    }
    if solver._solve_readout.kernels:
        record["solve_readout"] = json_value(solver._solve_readout.summary())
    return record


def _held_tree_groups(solver) -> tuple:
    owner = solver._tree_kinetic_state
    if owner is None:
        return None, []
    if hasattr(owner, "groups"):
        groups = list(owner.groups)
        # Scalar branch groups hold no tree layout.
        forest = any(getattr(getattr(group, "layout", None), "forest", False) for group in groups)
        return ("forest" if forest else "mixed"), groups
    return "tree", [owner]


def _held_tree(solver) -> dict:
    owner = solver._tree_kinetic_state
    kind, groups = _held_tree_groups(solver)
    return {
        "owner": kind,
        "reason": solver._cached_dynamics_selection_reason,
        "layout_reason": getattr(owner, "forest_reason", None),
        # Whether a diagonal-mass group of world-fixed roots over one-coordinate
        # leaves runs as scalar branches (by default; FEATHER_PGS_SCALAR_BRANCHES=0
        # opts out).
        "scalar_branches": {
            "enabled": getattr(owner, "scalar", None) is not None,
            "reason": getattr(solver, "_scalar_branches_reason", None),
        },
        "groups": [_tree_group(group) for group in groups],
    }


def _tree_group(group) -> dict:
    from .scalar_branches import ScalarBranchState  # noqa: PLC0415

    if isinstance(group, ScalarBranchState):
        return _scalar_group(group)
    layout, operator = group.layout, group.sparse
    return {
        "operator": type(operator).__name__,
        "size": json_value(getattr(operator, "size", None)),
        "group_count": int(group.group_count),
        "bodies": int(layout.bodies),
        "dofs": int(layout.dofs),
        "lanes": int(layout.lanes),
        "trees_per_block": int(layout.pack),
        "block_dim": int(group.block_dim),
        "roots": json_value(layout.roots),
        "root_coordinates": json_value(layout.root_coordinates),
        # Factor family per root: its coordinate count, negated for a diagonal-mass family, 0 when prescribed.
        "root_families": json_value(layout.root_families),
        "chain_scan": bool(group.chain_scan),
        "chain_scan_requested": bool(group.chain_scan_requested),
        "response": group._response,
        # "content": a repair whose input bank is not the bank the cache was published into
        # compares the two banks word for word; "identity": the bank's pointers alone.
        "input_validation": "content" if group.bank_compare else "identity",
        # Within one graph capture, a repair after this group's finish on the bank it published
        # launches nothing (FEATHER_PGS_TREE_REPAIR_SKIP=0 keeps it; motion-velocity layouts always launch).
        "captured_repair_skip": bool(group.skip_captured_repairs),
        # On CUDA, a repair of more blocks than two listed grids lists the blocks that repair (and
        # compare) and repairs over the list: always inside a capture, eagerly when the groups should
        # be stamped with the input bank or compared (FEATHER_PGS_TREE_REPAIR_CLASSIFY=0 keeps one
        # launch per block; motion-velocity layouts always do).
        "classify_first": bool(group.classify_first),
        # How each repair of the latest graph capture was recorded, in order: "skipped", "one_pass" or
        # "classified", with "_compared" when it compares input banks; empty before any capture.
        "captured_repairs": list(group._captured_repairs[1]) if group._captured_repairs else [],
        "motion_velocity": bool(group.motion_velocity),
        "publishes_twists": bool(group.publishes_twists),
        # Several admissible widths are timed once at construction (measurements.held_tree_lane_timing_us);
        # one width is static.
        "lane_choice": "timed" if group.lane_timing else "static",
        "kernels": {
            "repair": _key(group.repair_kernel),
            "finish": _key(group.finish_kernel),
            "predict": _key(group.predictor_kernel),
            # The input-bank comparison: a pass over the blocks a classified repair's classification lists
            # for comparison ("bank") or inside a second one-pass repair ("bank_repair"); None where the
            # group does not compare or classify.
            "bank": _key(group.bank_kernel),
            "bank_repair": _key(group.bank_repair_kernel),
            "classify": _key(group.classify_kernel),
            "listed_repair": _key(group.listed_repair_kernel),
        },
    }


def _scalar_group(group) -> dict:
    """A scalar branch group (``scalar_branches.py``) in the tree groups' schema.

    Its predict and finish run one thread per coordinate (``lanes``) and its
    repair one warp per articulation, so the tree layout's packing fields
    (``trees_per_block``, ``block_dim``) are ``None``. Every repair launches,
    and none is classified or compares input banks (``input_validation`` is
    the group's, "identity"), so the comparison and classification kernels are
    ``None``. ``kinds`` lists its leaf joint kinds; ``drive_dofs`` counts the
    DOFs whose drives the drive kernel still prepares, since the group's
    predict evaluates its own.
    """
    size = int(group.size)
    others = group.other_drive_dofs
    return {
        "operator": type(group).__name__,
        "size": size,
        "group_count": int(group.group_count),
        "bodies": size + 1,
        "dofs": size,
        "lanes": size,
        "trees_per_block": None,
        "block_dim": None,
        "roots": [0],
        "root_coordinates": [0],
        # A diagonal-mass family, negated as for the tree groups.
        "root_families": [-size],
        "chain_scan": False,
        "chain_scan_requested": False,
        "response": "diagonal",
        "input_validation": "content" if group.bank_compare else "identity",
        "captured_repair_skip": bool(group.skip_captured_repairs),
        "classify_first": False,
        "captured_repairs": [],
        "motion_velocity": bool(group.motion_velocity),
        "publishes_twists": False,
        "lane_choice": "static",
        "kinds": list(group.plan.kinds),
        "drive_dofs": 0 if others is None else int(others.shape[0]),
        "kernels": {
            "repair": _key(group.repair_kernel),
            "finish": _key(group.finish_kernel),
            "predict": _key(group.predictor_kernel),
            "bank": None,
            "bank_repair": None,
            "classify": None,
            "listed_repair": None,
        },
    }


def _kernel_names(kernels) -> list:
    return [kernel if isinstance(kernel, str) else _key(kernel) for kernel in kernels]


def _general_kernels(method, patch, source) -> tuple:
    """The general builder's kernels for one method, in launch order (_stage4_build_rows)."""
    if method in ("masked", "masked_combined"):
        if patch:
            return (source.contact_row_prelude_patch, source.populate_world_J_masked_patch)
        return (source.contact_row_prelude, source.populate_world_J_masked)
    if method == "per_contact":
        return (source.populate_world_J_for_size,)
    raise ValueError(f"No contact-row builder kernels are known for the method {method!r}.")


def _row_builder(solver, plan, size, source) -> dict:
    """Name the kernels that build one DOF size's dense contact rows outside the sweep, and their law.

    Mirrors the producers' launches, in launch order: _stage4_build_rows for the shared-metadata
    and general builders, compact_contact.launch_contacts, LocalRowPackets.produce_contacts and
    finish_rows, and KeyedRows.produce_contacts and finish_rows. A launch shared by several sizes
    (the row metadata, a combined masked launch, a producer's contact kernel) is named under each
    of them. Rows built elsewhere are not named: keyed rows' direct rows, which the parallel tiers
    build in the sweep (``direct_rows`` points to routes.sweep.tiers), and the free-body matrix-free
    and propagation rows, which build_mf_contact_rows and build_propagation_contact_rows write for
    the contacts the allocation routes to them (routes.free_rigid_bodies). _row_check names the debug row checks' builders.
    """
    method, law = plan.methods[size], plan.law
    patch = law == "patch"
    prepare = (source.prepare_world_contact_rows,) if plan.metadata_launch else ()
    if method == "local_packets":
        from . import row_packets  # noqa: PLC0415 - the owner's module; loaded with the owner

        # The owner fixes its friction law at construction.
        patch = bool(solver._row_packets.patch_friction)
        kernels = (
            (row_packets.produce_patch_contacts, row_packets.materialize_patch_fallback)
            if patch
            else (row_packets.produce_contacts, row_packets.materialize_fallback)
        )
    elif method == "keyed_rows":
        from . import keyed_rows  # noqa: PLC0415 - the owner's module; loaded with the owner

        owner = solver._keyed_rows
        patch = bool(owner.patch)
        # The parallel tiers (routes.sweep.tiers) build the direct rows in the sweep; the fallback
        # materializes the rows of every other world.
        kernels = (
            (keyed_rows.scatter_patch_contact_keys, keyed_rows.get_patch_fallback_kernel(owner.layout))
            if patch
            else (keyed_rows.scatter_contact_keys, keyed_rows.get_fallback_kernel(owner.layout))
        )
    elif method == "compact_contact_boundary":
        from . import compact_contact  # noqa: PLC0415 - the owner's module; loaded with the owner

        produce = (compact_contact.produce_patch_contacts if patch else compact_contact.produce_contacts,)
        if size == solver._sparse_diagonal_response_size:
            # The marker encodes schedule links into the freshly published sparse DOF slots.
            kernels = (*produce, solver._mark_independent_sparse_contact_candidates_kernel)
        else:
            kernels = (compact_contact.clear_contact_response, *produce)
    elif method == "sparse_response":
        kernels = (*prepare, solver._sparse_contact_response_kernel)
    elif method == "sparse_diagonal":
        kernels = (*prepare, source.populate_sparse_diagonal_contact_response)
        if solver._sparse_diagonal_contact_triples or solver._sparse_diagonal_contact_units:
            kernels = (*kernels, solver._mark_independent_sparse_contact_candidates_kernel)
    elif method == "compact_jacobian":
        kernels = (*prepare, source.populate_world_J_for_compact_size)
    else:
        kernels = _general_kernels(method, patch, source)
    entry = {"method": method, "kernels": _kernel_names(kernels), "law": "patch" if patch else "point"}
    if method == "keyed_rows":
        entry["direct_rows"] = "routes.sweep.tiers"
    return entry


def _row_check(plan, source) -> dict | None:
    """Name, by DOF size, the builders the debug row checks launch besides builder_by_dof's.

    FEATHER_PGS_CHECK_ROWS builds every size's masked rows into copies and compares them
    with the per-contact rows the solve uses. None without a check (ContactRowPlan.check).
    """
    if plan.check is None:
        return None
    kernels = _general_kernels(plan.check, plan.law == "patch", source)
    return {
        str(int(size)): {"method": plan.check, "kernels": _kernel_names(kernels), "law": plan.law}
        for size in plan.methods
    }


def _contact_rows(solver, source) -> dict:
    point = bool(solver._point_contact_paths)
    # The plan _stage4_build_rows dispatches on: the first admitted producer writes every
    # contact row; the general builder's masked patch rows serve only solvers no earlier producer took.
    plan = contact_row_plan(solver)
    methods = None
    if plan.producer == "general":
        methods = {str(int(size)): method for size, method in plan.methods.items()}
    if solver._compact_contact_boundary:
        compact_reason = (
            "FEATHER_PGS_COMPACT_CONTACT_BOUNDARY admitted the six-coordinate sparse-diagonal triples or units."
        )
    elif not source._COMPACT_CONTACT_BOUNDARY:
        compact_reason = "Off: FEATHER_PGS_COMPACT_CONTACT_BOUNDARY is an import-time opt-in."
    elif not solver._law_agnostic_contact_paths:
        compact_reason = (
            "Compact contacts implement the point and patch row laws; torsion and compliance keep ordinary rows."
        )
    else:
        compact_reason = (
            "Declined: requires the six-coordinate sparse-diagonal triple or unit contract and no row checks."
        )
    packets = solver._row_packets
    return {
        "producer": plan.producer,
        "law": plan.law,
        # Every producer, every size: the builder method, its kernels in launch order and the law they write,
        # for the dense contact rows built outside the sweep (_row_builder names the rows built elsewhere).
        "builder_by_dof": {str(int(size)): _row_builder(solver, plan, size, source) for size in plan.methods},
        # The debug row checks only: the builders they launch besides those (None otherwise).
        "row_check_by_dof": _row_check(plan, source),
        "general_method_by_dof": methods,
        "point_contact_paths": point,
        "row_packets": {
            "enabled": packets is not None,
            "reason": solver._row_packets_reason,
            "primary_dofs": None if packets is None else int(packets.primary),
            "secondary_dofs": None if packets is None else int(packets.secondary),
            "mimic_rows": None if packets is None else int(packets.mimics),
            "packets": None if packets is None else int(packets.packets),
        },
        "keyed_rows": {"enabled": solver._keyed_rows is not None, "reason": solver._keyed_rows_reason},
        "compact_contact": {
            "boundary": bool(solver._compact_contact_boundary),
            "reason": compact_reason,
        },
        "compact_jacobian": bool(solver._compact_contact_jacobian),
    }


def _contact_consumers(solver) -> dict:
    blocks = int(solver._contact_capacity_blocks)
    return {
        "max_blocks": blocks,
        "reason": (
            "Launches over the contact capacity (force export, warm-start snapshot and gathers, propagation unit"
            f" collection, keyed-row key scatters) grid-stride over {CAPACITY_BLOCKS_PER_SM} blocks of 256 threads"
            " per SM, about one wave."
            if blocks
            else "Uncapped: a CPU launch, or a kernel compiled without the grid-stride loop, covers every slot."
        ),
    }


def _local_solve(solver) -> dict:
    enabled = bool(solver._local_internal_fast_path)
    if enabled:
        reason = "Worlds with one small non-free articulation (and at most one free body) solve in local shared memory."
    elif solver._sparse_mass_matrix_size is not None:
        reason = "Native sparse factors replace the local owners."
    else:
        reason = (
            "Local owners need CUDA matrix-free immediate interleaved augmented-drive solving with free rigid "
            "bodies present, one small articulation per world, and no velocity limits, velocity passes, warm "
            "start, preelimination, debug or torsion."
        )
    # The candidate counts predate sparse admission, which can retire the owners; report built owners only.
    return {
        "enabled": enabled,
        "internal_articulations_by_dof": _counts(solver._local_internal_art_counts) if enabled else {},
        "pair_articulations_by_dof": _counts(solver._local_pair_art_counts) if enabled else {},
        "residual_articulations_by_dof": _counts(solver._local_residual_art_counts) if enabled else {},
        "reason": reason,
    }


def _paired_response(solver) -> dict:
    primary = solver._paired_response_primary_size
    if primary is None:
        if solver._sparse_diagonal_contact_solve or solver._sparse_mass_matrix_size is not None:
            reason = "A sparse response owner owns response."
        elif solver._explicit_dense_kernels:
            reason = "An explicit dense kernel choice owns response."
        else:
            reason = "No world pairs one tiled-response articulation with exactly one free body."
        return {"enabled": False, "reason": reason}
    factor = bool(solver._paired_factor_coordinates)
    overlap = solver._paired_general_solve_stream is not None
    return {
        "enabled": True,
        "primary_dofs": int(primary),
        "secondary_dofs": int(solver._paired_response_secondary_size),
        "factor_coordinates": factor,
        "contact_triples": bool(solver._factor_coordinate_contact_triples),
        "contact_units": bool(solver._factor_coordinate_contact_units),
        "components": solver._paired_components is not None,
        "components_reason": solver._paired_components_reason,
        "two_stream_overlap": overlap,
        "two_stream_reason": "The paired factor sweep and the general sweep fork onto two streams."
        if overlap
        else "Overlap needs independent components on parallel CUDA streams without debug, warm start or "
        "parallel tiers.",
        "reason": "One articulation factor and one free body per world share one response"
        + (" in factor coordinates." if factor else " with a register inverse."),
    }


def _sparse_diagonal(solver) -> dict:
    enabled = bool(solver._sparse_diagonal_contact_solve)
    if enabled:
        reason = "One diagonal-mass articulation and one compact dense articulation per world share one owner."
    elif solver._contact_torsion_radius > 0.0 or solver._explicit_dense_kernels:
        reason = "Contact torsion or an explicit dense kernel choice keeps the general response."
    else:
        reason = "No world pairs a diagonal-mass articulation with fused limits and one compact dense articulation."
    return {
        "enabled": enabled,
        "response_dofs": json_value(solver._sparse_diagonal_response_size),
        "dense_dofs": int(solver._sparse_diagonal_dense_size),
        "contact_triples": bool(solver._sparse_diagonal_contact_triples),
        "contact_units": bool(solver._sparse_diagonal_contact_units),
        "speculative_contact_batches": bool(solver._sparse_diagonal_speculative_contact_batches),
        "fused_joint_limits": bool(solver._fused_diagonal_joint_limits),
        "reason": reason,
    }


def _sparse_mass_matrix(solver) -> dict:
    size = solver._sparse_mass_matrix_size
    record = {
        "size": json_value(size),
        "reason": solver._sparse_selection_reason,
        "split": bool(solver._sparse_split),
        "velocity_limits": bool(solver._sparse_velocity_limits),
        "active_rows": bool(solver._sparse_active_rows),
        "active_rows_reason": solver._sparse_active_rows_reason,
        "support_table": bool(solver._sparse_row_support.enabled),
        "contact_triplets": bool(solver._sparse_contact_triplets),
        "contact_triplets_reason": solver._sparse_contact_triplets_reason,
        # The admitted triplets' transaction: "single" (one load group and reduction, per-triplet
        # records) or "staged" (normal, then tangents, metric cache); None without triplets.
        "contact_transaction": solver._sparse_contact_transaction,
        "contact_transaction_reason": solver._sparse_contact_transaction_reason,
        "contact_transaction_launch_bounds": json_value(solver._sparse_contact_transaction_launch_bounds),
    }
    if size is not None:
        record.update(
            {
                "max_row_support": json_value(solver._sparse_max_row_support),
                "solve_lanes": int(solver._sparse_solve_lanes),
                "local_factors": bool(solver._sparse_local_factors),
                "factor_warps_per_block": int(solver._sparse_factor_warps_per_block),
                "factor_kernel": _key(solver._crba_sparse_factor_kernel),
                "contact_response_kernel": _key(solver._sparse_contact_response_kernel),
                "sweep_kernel": _key(solver._pgs_solve_sparse_kernel),
            }
        )
    return record


def _sweep(solver, source) -> dict:
    # Split mode sweeps assembled Delassus rows unless native sparse factors own its rows.
    dense_split = solver.pgs_mode != "matrix_free" and not solver._sparse_split
    if dense_split:
        route = f"dense_{solver.pgs_kernel}"
        reason = f"Split mode sweeps the assembled Delassus rows with the {solver.pgs_kernel} kernel."
    elif solver._sparse_mass_matrix_size is not None:
        route, reason = "sparse_pgs", "Native sparse factors own the sweep."
    elif solver._sparse_diagonal_contact_solve:
        route, reason = "sparse_diagonal", "The diagonal-mass contact owner owns the sweep."
    else:
        route = "matrix_free_gs"
        reason = "Matrix-free Gauss-Seidel; local, paired and tiered owners claim their worlds first."
    # What drives the route's sweep each solve: mirrors the stage-6 branches of step(),
    # _launch_matrix_free_position_solve and the schedule branches of _launch_matrix_free_gs_solve.
    if dense_split and solver._has_mixed_contacts:
        driver = "split_interleaved"
        driver_reason = (
            "A world holds free bodies and articulations: each iteration runs a one-iteration dense sweep, "
            "rebuilds the velocity, runs a one-iteration matrix-free sweep and adds its velocity change to "
            "the dense right-hand side."
        )
    elif dense_split and solver.pgs_debug:
        driver = "debug_loop"
        driver_reason = "pgs_debug launches one dense iteration per dispatch and reads the impulses back after each."
    elif dense_split:
        driver = "dense_split"
        driver_reason = "One dense dispatch sweeps all iterations" + (
            "; one matrix-free dispatch then sweeps the free-body rows." if solver._has_free_rigid_bodies else "."
        )
    elif solver.pgs_mode != "matrix_free":
        driver, driver_reason = "direct", "The sparse split rows sweep once per solve."
    elif solver.pgs_debug:
        driver = "debug_loop"
        driver_reason = "pgs_debug launches the position solve one iteration per dispatch with host reads."
    elif solver.contact_compliance:
        driver, driver_reason = (
            "contact_compliance",
            "Compliance updates the right-hand side and repacks metadata before every one-iteration sweep.",
        )
    elif solver._propagation_contacts_enabled():
        driver = "propagation_fused" if solver.propagation_full_fused_iterations else "propagation_phases"
        driver_reason = (
            "One fused kernel runs each propagation iteration."
            if solver.propagation_full_fused_iterations
            else "Each iteration interleaves the route's row phases with the propagation row solve."
        )
    elif route == "matrix_free_gs" and solver.pgs_schedule == "contact_then_internal":
        driver = "contact_then_internal"
        driver_reason = "Contact rows sweep all iterations in one dispatch, then internal rows in a second."
    elif route == "matrix_free_gs" and solver.pgs_schedule == "physx_grasp":
        driver = "physx_grasp"
        driver_reason = "Each iteration launches the drive and limit, contact and velocity-limit row phases in turn."
    else:
        driver, driver_reason = "direct", "The route sweeps all iterations in one dispatch."
    dense = None
    if dense_split:
        if solver.pgs_kernel == "tiled_row":
            budgets = tuple(solver._pgs_solve_tiled_row_budgets)
            dense = {
                "kernels": [_key(kernel) for kernel in budgets or (solver._pgs_solve_tiled_row_kernel,)],
                "row_budgets": bool(budgets),
            }
        else:
            loop = source.pgs_solve_loop_unit if solver._dense_contact_unit else source.pgs_solve_loop
            dense = {"kernels": [_key(loop)]}
    free_body = None
    if dense_split and solver._has_free_rigid_bodies:
        if solver.model.device.is_cuda:
            free_body = _key(solver._pgs_solve_mf_kernel)
        else:
            free_body = _key(
                source.pgs_solve_mf_loop_unit if solver._free_body_contact_unit else source.pgs_solve_mf_loop
            )
    mf_meta_pack = None
    if solver.pgs_mode == "matrix_free":
        # Mirrors _pack_mf_meta: only free rigid bodies produce matrix-free rows.
        packed = bool(solver._has_free_rigid_bodies)
        mf_meta_pack = {
            "launched": packed,
            "kernel": _key(solver._pack_mf_meta_kernel),
            "reason": "Free rigid-body rows pack their metadata each step."
            if packed
            else "Skipped: without free rigid bodies no matrix-free row exists.",
        }
    propagation = None
    if solver._propagation_contacts_enabled():
        propagation = {
            "mode": solver.articulated_contact_response,
            "row_solve": _key(solver._pgs_solve_propagation_kernel),
            "full_iteration": _key(solver._pgs_solve_propagation_full_iteration_kernel),
            "colored_block": _key(solver._pgs_solve_propagation_colored_block_kernel),
            "colored_warp": _key(getattr(solver, "_pgs_solve_propagation_colored_warp_kernel", None)),
            "cached_response": bool(solver._propagation_cached_response_active),
        }
    local = None
    if solver._local_internal_fast_path:
        local = {
            "factory": "row_packets" if solver._row_packets is not None else "local_owned",
            "internal": _keys_by_size(solver._pgs_solve_local_internal_kernels),
            "pair": _keys_by_size(solver._pgs_solve_local_pair_kernels),
            "residual": _keys_by_size(solver._pgs_solve_local_residual_kernels),
        }
    record = {
        "route": route,
        "reason": reason,
        "driver": driver,
        "driver_reason": driver_reason,
        "schedule": solver.pgs_schedule,
        "iterations": int(solver.pgs_iterations),
        "velocity_iterations": int(solver.pgs_velocity_iterations),
        "mf_meta_pack": mf_meta_pack,
        "sparse": _key(solver._pgs_solve_sparse_kernel) if route == "sparse_pgs" else None,
        "sparse_diagonal": _key(solver._pgs_solve_sparse_diagonal_kernel),
        "paired_factor": _key(solver._paired_factor_solve_kernel),
        "mf_gs": _key(solver._pgs_solve_mf_gs_kernel),
        "mf_gs_storage": _mf_gs_storage(solver),
        "tiers": _tiers(solver),
        "local": local,
        "dense": dense,
        "free_body_matrix_free": free_body,
        "propagation": propagation,
    }
    record["readout"] = _readout(solver, record)
    record["exit_law"] = _exit_law(solver, record)
    record["contact_update"] = _contact_update(solver, record)
    return record


def _sweep_kernels(solver, record) -> set:
    """Every sweep kernel key of a sweep record."""
    keys = {record[name] for name in ("sparse", "sparse_diagonal", "paired_factor", "mf_gs", "free_body_matrix_free")}
    keys.update(tier["kernel"] for tier in (record["tiers"] or {}).get("tiers", ()))
    for owner in ("internal", "pair", "residual"):
        keys.update((record["local"] or {}).get(owner, {}).values())
    keys.update((record["dense"] or {}).get("kernels", ()))
    keys.update(value for key, value in (record["propagation"] or {}).items() if key not in ("mode", "cached_response"))
    keys.discard(None)
    return keys


def _uncounted_worlds(record) -> dict:
    """Owners whose sweeps solve rows without counters, with the worlds or rows each one solves.

    A world one owns whole holds rows, so the skip counter leaves it out, and the counted sweeps do
    not enter it: its solves are in no slot and not skipped. The incremental tiers count their own
    worlds.
    """
    owners = {}
    tiers = record["tiers"] or {}
    if tiers.get("family") in ("parallel", "keyed_parallel"):
        rows = max(int(tier["rows"]) for tier in tiers["tiers"])
        owners["parallel_tiers"] = (
            f"Worlds without matrix-free rows and with 1 to {rows} dense rows that no local owner claims: the "
            "parallel (Jacobi) tiers solve them under their own sweep budget and mf_gs_parallel_tol, and the "
            "matrix-free sweep returns before its first sweep."
        )
    if record["propagation"] is not None:
        owners["propagation"] = (
            "Worlds whose rows are all propagation rows: the propagation sweeps solve them, and the other sweeps "
            "return before their first sweep."
        )
    return owners


def _fused_propagation(record) -> bool:
    """Whether one fused propagation kernel runs every solve of a sweep record's route."""
    return (record["propagation"] or {}).get("mode") == "propagation-fused"


def _readout(solver, record) -> dict:
    """Which sweep kernels of the route count their sweeps under ``pgs_solve_readout``, and why the others do not."""
    counted = solver._solve_readout.kernels
    uncounted = sorted(_sweep_kernels(solver, record) - counted.keys())
    uncounted_worlds = _uncounted_worlds(record) if counted else {}
    if solver.pgs_solve_readout == "off":
        reason = "pgs_solve_readout is off."
    elif not counted and _fused_propagation(record):
        reason = (
            "The fused propagation kernel sweeps every world's dense, matrix-free and propagation rows without "
            "counters, and none of the sweeps that count launches, so nothing counts."
        )
    elif not counted:
        reason = "No sweep of this route carries counters: fixed-iteration and Jacobi-tier sweeps do not."
    elif record["driver"] == "direct":
        reason = "Each launch of a counted kernel solves its worlds; uncounted sweeps carry no counters."
    elif record["driver"] == "contact_then_internal":
        reason = "Contact and internal rows sweep in separate launches, so a solve counts once per launch."
    else:
        reason = (
            f"The {record['driver']} driver runs part of a solve per launch, so counts are per launch; a launch "
            "whose row phase holds none of a world's rows can return before its first sweep, which counts that "
            "world in no slot and not as skipped."
        )
    if uncounted_worlds:
        reason += (
            " The owners in uncounted_worlds sweep their rows without counters: the solves of a world one owns "
            "whole are in no slot and not skipped."
        )
    return {
        "mode": solver.pgs_solve_readout if counted else "off",
        "counted": dict(sorted(counted.items())),
        "uncounted": uncounted if counted else [],
        # Owners of whole worlds whose solves neither a slot nor the skip counter counts, with those worlds.
        "uncounted_worlds": uncounted_worlds,
        # Variants that could not compile the requested trace count their sweeps without it.
        "declined": dict(sorted(solver._solve_policy_declined.items())) if solver.pgs_solve_readout == "trace" else {},
        # Passes whose solves of worlds without rows, which the matrix-free sweeps do not enter, count as skipped.
        "skipped_passes": list(solver._solve_readout.skipped_passes) if counted else [],
        "reason": reason,
    }


def _exit_law(solver, record) -> dict:
    """Which sweep kernels of the route stop under the velocity law (``pgs_velocity_change_tolerance``), and why not."""
    tolerance = float(solver.pgs_velocity_change_tolerance)
    honored = dict(sorted(solver._solve_policy_honored.items()))
    declined = dict(sorted(solver._solve_policy_declined.items())) if tolerance > 0.0 else {}
    ignored = sorted(_sweep_kernels(solver, record) - honored.keys() - declined.keys())
    driver_honors = record["driver"] in ("direct", "contact_then_internal")
    if tolerance == 0.0:
        reason = "pgs_velocity_change_tolerance is 0: every sweep keeps the exact stationary-sweep law."
    elif not honored and _fused_propagation(record):
        reason = (
            "The fused propagation kernel runs every solve's fixed iteration count without an exit, and none of "
            "the sweeps that compile the velocity law launches, so no sweep applies it."
        )
    elif not honored:
        reason = "No sweep of this route compiles the velocity law; every sweep keeps the exact law."
    elif not driver_honors:
        reason = (
            f"The {record['driver']} driver runs part of a solve per launch, so a sweep's exit cannot end the solve."
        )
    else:
        reason = (
            "Honored sweeps stop a world once every velocity-scaled change of a sweep is within the tolerance; "
            "declined sweeps keep the exact law, ignored ones their fixed iteration count or own tolerance."
        )
    return {
        "law": "velocity" if tolerance > 0.0 and honored else "exact",
        "velocity_change_tolerance": json_value(tolerance),
        "honored": honored,
        # Variants that could not compile the requested law keep the exact law.
        "declined": declined,
        "ignored": ignored if tolerance > 0.0 else [],
        "driver_honors": driver_honors,
        "reason": reason,
    }


def _contact_update(solver, record) -> dict:
    """Which sweep kernels of the route solve contact units exactly (``contact_update="unit"``), and why not the others."""
    unit = []
    if solver._dense_contact_unit:
        unit.append(record["dense"]["kernels"][0])
    if solver._free_body_contact_unit:
        unit.append(record["free_body_matrix_free"])
    requested = solver.contact_update
    sequential = []
    if requested == "unit":
        sequential = sorted(_sweep_kernels(solver, record) - set(unit))
    if requested == "sequential":
        reason = (
            "contact_update is 'sequential': a sweep steps each contact's normal row, then solves its friction pair "
            "at the held normal load."
        )
    elif not sequential:
        reason = "Every sweep of this route solves each contact unit exactly against its own Delassus block."
    else:
        reason = (
            "Only the loop sweeps solve contact units: the dense loop sweep of split mode (pgs_kernel='loop') on any "
            "device and the split free-body sweep on CPU. The other sweeps keep the sequential law."
        )
    return {
        "requested": requested,
        "law": "unit" if unit else "sequential",
        # Sweep kernels that solve contact units, and those that keep the sequential law under a "unit" request.
        "unit": unit,
        "sequential": sequential,
        # How the unit sweeps solve a patch region: one ring unit, or anchor units with the ring's other load held.
        "patch_units": solver._contact_unit_patch if unit else None,
        "reason": reason,
    }


def contact_update_warnings(sweep) -> list[str]:
    """Return a warning when a requested ``contact_update="unit"`` reaches only some sweeps of a sweep record.

    Args:
        sweep: The ``routes.sweep`` record of :func:`route_selection`.
    """
    update = sweep["contact_update"]
    if update["requested"] != "unit" or not update["sequential"]:
        return []
    return [
        f"contact_update='unit' applies only to some sweeps of this solver: {update['reason']} "
        f"Sequential: {update['sequential']}."
    ]


def policy_warnings(sweep) -> list[str]:
    """Return a warning per requested solve policy that some sweeps of a sweep record do not apply.

    Args:
        sweep: The ``routes.sweep`` record of :func:`route_selection`.
    """
    messages = []
    law = sweep["exit_law"]
    if law["velocity_change_tolerance"] > 0.0 and (law["declined"] or law["ignored"] or not law["driver_honors"]):
        messages.append(
            f"pgs_velocity_change_tolerance applies only to some sweeps of this solver: {law['reason']} "
            f"Declined: {sorted(law['declined'])}; ignored: {law['ignored']}."
        )
    if sweep["readout"]["declined"]:
        messages.append(
            "pgs_solve_readout='trace' applies only to some sweeps of this solver; these count their sweeps "
            f"without the trace: {sorted(sweep['readout']['declined'])}."
        )
    return messages


def _mf_gs_storage(solver) -> dict | None:
    # Storage classes of the MF-GS kernel, chosen by execution_plan._select_mfgs_storage.
    storage = solver._mfgs_storage
    if storage is None:
        return None

    def storage_class(resident):
        return "resident" if resident else "streamed"

    return {
        "metadata": storage_class(storage.shared_metadata),
        "row_types_and_parents": storage_class(storage.shared_metadata or storage.resident_row_metadata),
        "mf_impulses": storage_class(not storage.stream_mf_impulses),
        "reason": storage.reason,
    }


def _tiers(solver) -> dict | None:
    kernels = list(solver._pgs_solve_mf_gs_incremental_kernels)
    if not kernels:
        return None
    if solver._ink_rows > 0:
        return {
            "family": "keyed_parallel" if solver._keyed_rows is not None else "parallel",
            # FEATHER_PGS_INK: the tier kernels build their world's response in-kernel.
            "inkernel_response": solver._ink_sizes is not None,
            # Every built tier projects each friction pair on its patch ring's pooled normal load (patch friction).
            "linked_rows": all(getattr(kernel, "_fpgs_linked_rows", False) for kernel in kernels),
            # How a friction pair's tangent rows step: one isotropic step (the linked tiers' law, the serial pair
            # solve's friction law) or each row's own step. routes.friction.parallel_rows says whether the tiers
            # share the serial sweep's whole fixed point, which also depends on the velocity-limit law.
            "pair_step": "isotropic"
            if all(getattr(kernel, "_fpgs_pair_step", "per_row") == "isotropic" for kernel in kernels)
            else "per_row",
            # The last-word velocity-limit pass after the final sweep (mf_gs_parallel_linked_velocity_limit_pass).
            "velocity_limit_pass": all(getattr(kernel, "_fpgs_velocity_limit_pass", False) for kernel in kernels),
            # The tiers accumulate joint velocity-limit rows whatever the serial sweeps' law
            # (routes.velocity_limits.law), so the two can differ in the worlds the tiers own.
            "velocity_limit_law": _tier_velocity_limit_law(solver, kernels),
            "nesterov": bool(solver.mf_gs_parallel_nesterov),
            "matrix_free": bool(solver.mf_gs_parallel_matrix_free),
            # The largest tier budget; each tier records its own. Point tiers sweep mf_gs_parallel_sweeps, linked
            # tiers mf_gs_parallel_linked_sweeps unless it is None (rule "auto", its default, or a count).
            "sweeps": max(int(kernel._fpgs_sweeps) for kernel in kernels),
            "sweeps_rule": solver._par_sweeps_rule,
            "sweeps_reason": solver._par_sweeps_reason,
            "tolerance": json_value(solver.mf_gs_parallel_tol),
            "tiers": [
                {
                    "rows": int(rows),
                    "min_rows": int(min_rows),
                    "sweeps": int(kernel._fpgs_sweeps),
                    "kernel": _key(kernel),
                }
                for (rows, min_rows), kernel in zip(solver._par_tiers, kernels, strict=False)
            ],
        }
    return {
        "family": "incremental",
        # The incremental tiers are built with the serial sweeps' law.
        "velocity_limit_law": _tier_velocity_limit_law(solver, kernels),
        "tiers": [{"kernel": _key(kernel)} for kernel in kernels],
    }


def _tier_velocity_limit_law(solver, kernels) -> str | None:
    # The law the built tier kernels give joint velocity-limit rows; one construction call builds every tier with
    # it. None where no such row can exist. The tiers serve dense rows only, so free-body limits never reach them.
    if not solver.enable_joint_velocity_limits or np.isinf(solver.velocity_limit_activation_fraction):
        return None
    return kernels[0]._fpgs_velocity_limit_law


def _resolved_worlds(solver) -> dict:
    owner = solver._resolved_worlds_owner
    published = bool(owner is not None and owner.published)
    if owner is None:
        twists = None
    elif published:
        twists = "The held-tree predictor publishes body twists for every responding articulation."
    else:
        twists = "The classifier builds body twists itself: a responding articulation has no twist-publishing tree."
    return {
        "enabled": owner is not None,
        "reason": solver._resolved_worlds_reason,
        "publishes_predicted_twists": published,
        "twists_reason": twists,
    }


def _publication(solver, source) -> dict:
    fused = bool(solver._fused_publication)
    if solver._tree_kinetic_state is not None:
        owner, reason = "cached", "The held-tree owner publishes next-step state."
    elif solver._publication_deferred():
        owner, reason = "deferred", "Fused current dynamics permits deferred kinematics publication."
    elif solver.lazy_kinematics and solver._fused_k1:
        owner, reason = "ordinary", "Patch friction publishes every substep for its release test."
    elif fused:
        owner, reason = "fused_general", "Each articulation publishes in one launch per size group."
    elif not source._FUSED_PUBLISH:
        owner, reason = "ordinary", "Separate launches; FEATHER_PGS_FUSED_PUBLISH is an import-time opt-in."
    else:
        owner = "ordinary"
        reason = (
            "Fused publication declined: an owner, free body, velocity limit or loop joint needs separate launches."
        )
    if not solver._fk_id_cache_enabled:
        cache = "off"
    elif solver._fk_id_cache_uses_snapshot:
        cache = "snapshot"
    else:
        cache = "in_place"
    return {
        "owner": owner,
        "reason": reason,
        "fused_general": fused,
        "fk_id_cache": cache,
        "prismatic_publication": solver._prismatic_publication is not None,
        "prismatic_linear_state": bool(solver._prismatic_linear_state),
    }


def _velocity_limits(solver) -> dict:
    if not solver.enable_joint_velocity_limits:
        joint, reason = "off", "enable_joint_velocity_limits is False."
    elif solver._sparse_velocity_limits:
        joint, reason = "sparse_factor_rows", "Native sparse factors solve velocity-limit rows in factor coordinates."
    elif solver.fuse_joint_velocity_limits:
        joint = "fused_clamp"
        reason = "Driven DOFs take the stateless end-of-iteration clamp; undriven limited DOFs keep dedicated rows."
    else:
        joint, reason = (
            "rows",
            "Dedicated velocity-limit rows (the fused clamp needs matrix-free physx_pgs drives, a finite "
            "activation fraction and model joint velocity limits).",
        )
    free_body_law, free_body_law_reason = _split_free_body_velocity_limit_law(solver)
    limits = bool(solver._has_rigid_body_velocity_limits)
    matrix_free = solver.pgs_mode == "matrix_free"
    model = solver.model
    prescale = solver.enable_joint_velocity_limits and model.joint_count > 0 and model.articulation_count > 0
    # The law the matrix-free sweep kernels were built with; the fused drive clamp keeps PhysX's.
    law, law_reason = solver._velocity_limit_sweep_law()
    return {
        "joint": joint,
        "reason": reason,
        "law": law,
        "law_reason": law_reason,
        # Lanes per articulation of the pre-solve PhysX velocity prescale; 0 when it launches no kernel.
        "prescale_lanes": int(solver._velocity_prescale_lanes) if prescale else 0,
        "activation_fraction": json_value(solver.velocity_limit_activation_fraction),
        "rigid_body_limits": limits,
        # Mirrors _allocate_mf_buffers and the row allocation of _stage4_build_rows (both modes),
        # and _clamp_rigid_velocity_limits for free articulation roots. At an infinite activation
        # fraction the allocator still launches but reserves no row.
        "free_body_rows": bool(limits and solver._free_rigid_body_count > 0),
        # The law of split mode's free-body sweep (sweep.free_body_matrix_free) for these rows.
        "free_body_sweep_law": free_body_law,
        "free_body_sweep_law_reason": free_body_law_reason,
        "root_clamp": bool(
            limits
            and solver._has_root_free
            and not (matrix_free and getattr(solver, "rigid_velocity_limit_slot", None) is not None)
        ),
    }


def _split_free_body_velocity_limit_law(solver) -> tuple[str | None, str]:
    # Mirrors the free_body_matrix_free sweep of _sweep and the kernel selection of _init_tiled_kernels.
    if solver.pgs_mode == "matrix_free" or solver._sparse_split or not solver._has_free_rigid_bodies:
        return None, "No split free-body sweep: matrix-free mode, native sparse split rows or no free rigid bodies."
    if solver._has_free_body_velocity_limit_rows:
        return "stateless", (
            "A violated free-body velocity-limit row pushes by its whole, unrelaxed correction and keeps that "
            "correction instead of an accumulated impulse; the rows end every iteration. The CPU loop applies "
            "this law and CUDA builds the _vlim sweep for it; velocity_limit_law selects the law of the "
            "matrix-free sweeps and does not apply here."
        )
    if solver._has_rigid_body_velocity_limits:
        return None, "velocity_limit_activation_fraction=inf reserves no free-body velocity-limit row."
    return None, "No free-body velocity-limit rows: the rigid-body speed attributes are not registered."


def _free_rigid_bodies(solver) -> dict:
    count = int(solver._free_rigid_body_count)
    owners = []
    if count:
        if solver._route_free_free_contacts:
            owners.append("propagation_rows")
        elif solver.pgs_mode != "matrix_free" and not solver._sparse_split:
            owners.append("split_matrix_free_rows")
        else:
            if solver._sparse_mass_matrix_size is not None:
                owners.append("sparse_free_rows")
            if solver._paired_response_primary_size is not None:
                owners.append("paired_factor" if solver._paired_factor_coordinates else "paired_response")
            # Built local kernels only: sparse admission retires the local owners after their candidates.
            if solver._pgs_solve_local_pair_kernels:
                owners.append("local_pair")
            if solver._pgs_solve_local_residual_kernels:
                owners.append("local_residual")
            owners.append("matrix_free_rows")
    return {
        "count": count,
        "mixed_worlds": bool(solver._has_mixed_contacts),
        "owners": owners,
        "implicit_gyro": solver._free_body_gyro_kernel is not None,
        "reason": "No free rigid bodies."
        if not count
        else "Owners that can solve free-body rows, in claim order; contacts are routed per world at run time.",
    }


def _pair_sort(patches) -> dict:
    """Record the patch pair sort: over the live slots only (``live``), or a radix sort over the whole capacity.

    The choice follows ``_uses_live_sort``: the live sort from ``live_from``
    contact slots on and, with frame reuse, up to ``live_up_to`` slots, while
    the pair keys leave room for its slot bits.
    """
    capacity = int(patches.capacity)
    if patches.live_sort:
        if capacity >= _LIVE_SORT_CAPACITY:
            reason = f"From {_LIVE_SORT_CAPACITY:,} contact slots the pair sort orders only the live slots."
        else:
            reason = (
                f"Up to {_CHUNK_SORT_CAPACITY:,} contact slots the pair sort orders only the live slots, so a frame "
                "without contacts costs no sort."
            )
    elif patches.key_bits + _SORT_SLOT_BITS > 62:
        reason = (
            f"{patches.key_bits}-bit pair keys leave no room for the live sort's {_SORT_SLOT_BITS} slot bits: the "
            "radix sort orders the whole capacity."
        )
    elif capacity > _CHUNK_SORT_CAPACITY:
        reason = (
            f"Between {_CHUNK_SORT_CAPACITY:,} and {_LIVE_SORT_CAPACITY:,} contact slots the radix sort orders the "
            "whole capacity."
        )
    else:
        reason = "Without frame reuse (FEATHER_PGS_PATCH_FRAME_REUSE=0) the radix sort orders the whole capacity."
    return {
        "live": bool(patches.live_sort),
        "capacity": capacity,
        "live_up_to": _CHUNK_SORT_CAPACITY if patches.small_live_sort else None,
        "live_from": _LIVE_SORT_CAPACITY,
        "reason": reason,
    }


def _friction(solver) -> dict:
    patch = bool(solver._friction_anchors_enabled)
    reason = _BETA_REASONS[solver._friction_anchor_beta_source]
    if patch:
        owners = []
        if not solver._parallel_contact_paths:
            owners.append("keyed rows and parallel matrix-free tiers")
        # Patch friction reaches the resolved-world classifier on the dense paths only, not under torsion.
        if not solver._patch_dense_paths:
            owners.append("the resolved-world classifier")
        # FEATHER_PGS_PATCH_ACTIVE_ROWS=0 keeps patch friction off the sparse active-row scan.
        if not solver._patch_active_rows:
            owners.append("sparse active rows")
        if owners:
            reason += f" Point-contact owners ({', '.join(owners)}) need point friction."
        reason += _PATCH_ROUTE_PRECEDENCE if solver._law_agnostic_contact_paths else _ORDINARY_PATCH_ROUTES
    return {
        "law": "patch" if patch else "point",
        "friction_anchor_beta": json_value(solver.friction_anchor_beta),
        "anchor_beta_source": solver._friction_anchor_beta_source,
        "friction_mode": solver.friction_mode,
        "torsion": bool(solver._contact_torsion_enabled),
        "compliance": bool(solver.contact_compliance),
        # Whether the parallel tiers and keyed rows may serve this law; routes.sweep.tiers and
        # routes.contact_rows.keyed_rows report whether the recipe built them.
        "parallel_rows": {
            "admitted": bool(solver._parallel_contact_paths),
            "reason": solver._parallel_contact_paths_reason,
        },
        "reason": reason,
        "frame_reuse": {
            "enabled": solver._friction_patches.reuse_frames,
            "reason": solver._patch_frame_reuse_reason,
        },
        "pair_sort": _pair_sort(solver._friction_patches) if patch else None,
    }


def grouped_response(solver, size) -> bool:
    """Whether a DOF size's stage-4 response launches the grouped H^-1 J^T kernel.

    Mirrors the stage-4 dispatch of ``step()`` and the early return of ``_stage4_hinv_jt_tiled``:
    sparse, diagonal, compact, paired and fused responses own their sizes.
    """
    if not solver._use_grouped_hinv_jt(size):
        return False
    if solver._execution_plan.use_diagonal_mass(size):
        return False
    if solver.pgs_mode == "matrix_free" or solver._sparse_split:
        return not (
            solver._compact_contact_boundary
            or solver._sparse_mass_matrix_size is not None
            or (solver._sparse_diagonal_contact_solve and size == solver._sparse_diagonal_response_size)
            or size in (solver._paired_response_primary_size, solver._paired_response_secondary_size)
        )
    # Split mode: the fused response of one articulation per world replaces every per-size launch.
    return not (
        solver._is_one_solve_art_per_world
        and bool(solver.size_groups)
        and all(solver._execution_plan.use_fused_hinv_jt(other) for other in solver.size_groups)
    )


def _grouped_response_reason(solver, size, source) -> str:
    # Follows the order of _use_grouped_hinv_jt.
    if grouped_response(solver, size):
        if solver._ink_skip_rows() > 0:
            return "Grouped: the in-kernel tiers need the grouped kernel's per-world row skip."
        return f"Grouped: sizes up to FEATHER_PGS_GROUPED_HINV_MAX={source._GROUPED_HINV_MAX} take the grouped kernel."
    if int(size) > 32:
        return "The grouped response solve is verified up to 32 DOFs only."
    if solver._use_grouped_hinv_jt(size):
        return "A sparse, diagonal, compact, paired or fused response owns this size."
    if source._GROUPED_HINV_OFF:
        return "FEATHER_PGS_GROUPED_HINV=0 keeps the planned response kernel."
    if int(size) > source._GROUPED_HINV_MAX:
        return f"Above FEATHER_PGS_GROUPED_HINV_MAX={source._GROUPED_HINV_MAX}."
    return "Free-body sizes keep the per-row kernel; FEATHER_PGS_GROUPED_HINV_FREE=1 opts in."


def _grouped(solver, source) -> dict | None:
    topology = solver._grouped_topology
    if topology is None:
        return None
    held = solver._tree_kinetic_state is not None
    velocity_pass = solver.pgs_mode == "matrix_free" and solver.pgs_velocity_iterations > 0
    # Stage-7 kinematics publication: the held-tree finish and the fused general publication bypass it.
    publishes = solver._fk_id_cache_enabled and (velocity_pass or not (held or solver._fused_publication))
    asynchronous = bool(solver._async_augmented_drives)
    # _launch_rigid_tau: asynchronous drives without fused world dynamics, or drives that do not fuse tau.
    tau = (
        not held
        and solver._direct_branch_tau_kernel is None
        and (
            (asynchronous and not solver._fused_k1)
            or (not asynchronous and not (solver.drive_mode == "augmented" and solver.articulation_max_dofs > 0))
        )
    )
    return {
        "lanes_per_articulation": int(topology.lanes_per_articulation),
        "articulations_per_block": int(topology.articulations_per_block),
        # -1 when the template build failed; template kernels then stay off.
        "template_count": int(topology.template_count),
        "template_fk": bool(publishes and topology.template_count > 0 and not source._GROUPED_FK_OFF),
        "tau": ("grouped_tau_mass" if solver._grouped_tau_mass else "grouped_tau") if tau else None,
        "hinv_jt_by_dof": {
            str(int(size)): {
                "grouped": grouped_response(solver, size),
                "reason": _grouped_response_reason(solver, size, source),
            }
            for size in solver.size_groups
        },
    }


def _dynamics(solver, source) -> dict:
    return {
        "grouped_topology": solver._grouped_topology is not None,
        "grouped": _grouped(solver, source),
        "fused_world_dynamics": bool(solver._fused_k1),
        "grouped_tau_mass": bool(solver._grouped_tau_mass),
        "grouped_mass": bool(solver._grouped_mass),
        "parallel_tree": solver._tree_plan is not None,
        "compact_diagonal_mass_dofs": json_value(solver._compact_diagonal_mass_size),
        "diagonal_mass_dofs": sorted(int(size) for size in solver._diagonal_mass_sizes),
        "parallel_composite_inertia": bool(solver._parallel_composite_inertia),
        "compact_inertia_refresh": bool(solver._compact_inertia_refresh),
        # Response layout: whether J/Y are gathered per world or already world-indexed.
        "jy_world_aliased": bool(solver._jy_world_aliased),
        "hinv_jt_writes_world": bool(solver._hinv_jt_writes_world),
        "async_augmented_drives": bool(solver._async_augmented_drives),
        "parallel_augmented_drive_topology": bool(solver._parallel_augmented_drive_topology),
        "double_buffer": bool(solver._double_buffer),
        # Sizes whose double-buffered mass matrix the end of a refresh step clears on the memset
        # stream: empty without a double buffer, where the ordinary route's stage 1 clears H before
        # CRBA. Sizes a held-tree owner factors from its own geometry have no reader
        # (_unread_mass_matrix_sizes).
        "double_buffer_mass_matrix_clear_dofs": []
        if solver._memset_stream is None
        else sorted(int(size) for size in solver.size_groups if size not in solver._unread_mass_sizes),
        "size_group_streams": any(stream is not None for stream in solver._size_streams.values()),
        "preelimination": bool(solver._preelim_active),
        "mimic_rows": int(solver._mimic_count or 0),
        "connect_rows": int(solver._connect_count or 0),
        "loop_joints": bool(solver._has_loop_joints),
        "prescribed_response": bool(solver._has_prescribed_response),
    }


def _diagnostics(solver) -> dict:
    """Opt-in per-step diagnostics recorded so far: the stage-3 snapshots, which the solver never reads."""
    switches = " and ".join(
        switch
        for switch, on in (
            ("FEATHER_PGS_DEBUG_SNAPSHOTS=1", solver._debug_stage3_snapshot_flag),
            ("pgs_debug=True", solver._debug_buffers_enabled),
        )
        if on
    )
    # Mirrors _allocate_common_buffers: a model without joints has no stage 3 to snapshot.
    snapshots = solver._debug_stage3_qd_work is not None
    if snapshots:
        reason = (
            f"{switches}: every substep copies the stage-3 joint velocity, acceleration and predicted velocity "
            "into _debug_stage3_qd_work, _debug_stage3_joint_qdd and _debug_stage3_v_hat."
        )
    elif switches:
        reason = f"{switches}, but the model has no joints to snapshot."
    else:
        reason = (
            "Off: FEATHER_PGS_DEBUG_SNAPSHOTS=1 or pgs_debug=True opts in to three stage-3 snapshot copies per substep."
        )
    return {"stage3_snapshots": {"enabled": snapshots, "reason": reason}}
