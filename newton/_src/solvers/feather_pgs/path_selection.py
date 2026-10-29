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

from ...core.live_sort import SLOT_BITS
from ...sim.contacts import CAPACITY_BLOCKS_PER_SM
from . import response_builds_rows
from .conditioning import DEEP_CHAIN_JOINTS
from .contact_row_plan import contact_row_plan
from .solve_policy import QUIET_UNITS, UNIT_ROUNDOFF

# Options whose applied value lives under another attribute after construction.
_APPLIED_ATTRIBUTES = {
    "nvtx": "_nvtx",
    "row_watermark": "_row_watermark",
    "enable_bilateral_preelimination": "_preelim_active",
}

_BETA_REASONS = {
    "default": "friction_anchor_beta omitted: persistent patch friction at the 0.2 default.",
    "default_compliance": "friction_anchor_beta omitted with contact_compliance: point friction.",
    "argument": "friction_anchor_beta passed explicitly.",
}


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


# Options whose request can resolve to another value, or to no owner or tier at all.
_EFFECTIVE = {
    # Tier budgets after their gates and clamps.
    "mf_gs_parallel_rows": lambda solver: int(solver._ink_rows),
    "mf_gs_parallel_matrix_free": lambda solver: bool(solver.mf_gs_parallel_matrix_free and solver._ink_rows > 0),
    "mf_gs_incremental_rows": lambda solver: int(solver._incremental_rows),
    "mf_gs_parallel_linked_velocity_limit_pass": lambda solver: bool(
        solver._ink_rows > 0 and any(tier.velocity_limit_pass for tier in solver._mf_gs_tiers)
    ),
    # The sweeps' law after the schedule and row gates (routes.velocity_limits.law): None in the CPU's split mode,
    # which builds no matrix-free sweep and applies no law.
    "velocity_limit_law": lambda solver: solver._velocity_limit_sweep_law()[0],
    # A readout applies only where a sweep kernel carries its counters.
    "pgs_solve_readout": lambda solver: solver.pgs_solve_readout if solver._solve_readout.kernels else "off",
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
    request; ``routes`` names the owners. ``source`` is ``"argument"`` when the
    requested value differs from the declared default and ``"default"``
    otherwise; Python cannot tell an omitted option from one passed with its
    default value.
    """
    record = {}
    for name, default in _declared_defaults(init).items():
        value = arguments.get(name, default)
        effective = _EFFECTIVE.get(name)
        if effective is not None:
            applied = effective(solver)
        else:
            applied = getattr(solver, _APPLIED_ATTRIBUTES.get(name, name), value)
        record[name] = {
            "applied": json_value(applied),
            "requested": json_value(value),
            "default": json_value(default),
            "source": "default" if _is_default(value, default) else "argument",
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
        # The test-only axis pins in force (SolverFeatherPGS._route_pins); empty in production.
        "pins": json_value(dict(solver._route_pins)),
        # The row laws every owner's admission reads (route_table.Laws).
        "laws": solver._laws.record(),
        # The per-stage dispatch tables step() launches through (step_plan.StepPlan).
        "stages": solver._plan.record(),
        "held_tree": _held_tree(solver),
        "contact_rows": _contact_rows(solver, source),
        "contact_consumers": _contact_consumers(solver),
        "local_solve": _local_solve(solver),
        "paired_response": _paired_response(solver),
        "sparse_diagonal": _sparse_diagonal(solver),
        "sparse_mass_matrix": _sparse_mass_matrix(solver),
        "sweep": _sweep(solver),
        "resolved_worlds": _resolved_worlds(solver),
        "publication": _publication(solver),
        "velocity_limits": _velocity_limits(solver),
        "free_rigid_bodies": _free_rigid_bodies(solver),
        "friction": _friction(solver),
        "dynamics": _dynamics(solver),
        "diagnostics": _diagnostics(solver),
    }


def measurement_selection(solver) -> dict:
    """Measured values, kept out of the routes so that runs of one configuration compare equal.

    With ``pgs_solve_readout`` on, ``solve_readout`` summarizes the sweep counters
    (:meth:`SolveReadoutRecord.summary`); reading them synchronizes the device.
    The counters accumulate, so a window is the difference of two records.
    """
    record = {}
    if solver._solve_readout.kernels:
        record["solve_readout"] = json_value(solver._solve_readout.summary())
    return record


def _held_tree_groups(solver) -> tuple:
    owner = solver._tree_kinetic_state
    if owner is None:
        return None, []
    groups = list(owner.groups)
    # Scalar branch groups hold no tree layout.
    if any(getattr(getattr(group, "layout", None), "forest", False) for group in groups):
        return "forest", groups
    return ("tree" if solver._sparse_mass_matrix_size is not None else "mixed"), groups


def _held_tree(solver) -> dict:
    owner = solver._tree_kinetic_state
    kind, groups = _held_tree_groups(solver)
    return {
        "owner": kind,
        "reason": solver._cached_dynamics_selection_reason,
        "layout_reason": getattr(owner, "forest_reason", None),
        # Whether a diagonal-mass group of world-fixed roots over one-coordinate
        # leaves runs as scalar branches.
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
        # The scans' schedules, chosen from the tree's data with its lanes (tree_kinetic_state.held_schedules).
        "schedules": {name: getattr(group.schedules, name).value for name in ("pose", "motion", "reduce", "external")},
        "response": group._response,
        # "content": a repair whose input bank is not the bank the cache was published into
        # compares the two banks word for word; "identity": the bank's pointers alone.
        "input_validation": "content" if group.bank_compare else "identity",
        # Within one graph capture, a repair after this group's finish on the bank it published
        # launches nothing (motion-velocity layouts always launch).
        "captured_repair_skip": bool(group.skip_captured_repairs),
        # On CUDA, a repair of more blocks than two listed grids lists the blocks that repair (and
        # compare) and repairs over the list: always inside a capture, eagerly when the groups should
        # be stamped with the input bank or compared (motion-velocity layouts keep one launch per
        # block).
        "classify_first": bool(group.classify_first),
        # How each repair of the latest graph capture was recorded, in order: "skipped", "one_pass" or
        # "classified", with "_compared" when it compares input banks; empty before any capture.
        "captured_repairs": list(group._captured_repairs[1]) if group._captured_repairs else [],
        "motion_velocity": bool(group.motion_velocity),
        "publishes_twists": bool(group.publishes_twists),
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
    ``None``. ``drive_dofs`` counts the DOFs whose drives the drive kernel
    still prepares, since the group's predict evaluates its own.
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
        "schedules": None,
        "response": "diagonal",
        "input_validation": "content" if group.bank_compare else "identity",
        "captured_repair_skip": bool(group.skip_captured_repairs),
        "classify_first": False,
        "captured_repairs": [],
        "motion_velocity": bool(group.motion_velocity),
        "publishes_twists": False,
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


def _general_kernels(method, source) -> tuple:
    """The general builder's kernels for one method, in launch order (_stage4_build_rows)."""
    if method in ("masked", "masked_combined"):
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
    rows, which build_mf_contact_rows writes for the contacts the allocation routes to them
    (routes.free_rigid_bodies).
    """
    method, law = plan.methods[size], plan.law
    patch = law == "patch"
    prepare = (source.prepare_world_contact_rows,) if plan.metadata_launch else ()
    if method == "local_packets":
        from . import row_packets  # noqa: PLC0415 - the owner's module; loaded with the owner

        kernels = (row_packets.produce_contacts, row_packets.materialize_fallback)
    elif method == "keyed_rows":
        from . import keyed_rows  # noqa: PLC0415 - the owner's module; loaded with the owner

        patch = bool(solver._keyed_rows.patch)
        # The parallel tiers (routes.sweep.tiers) build the direct rows in the sweep; the fallback
        # materializes the rows of every other world.
        kernels = (keyed_rows.scatter_contact_keys, keyed_rows.materialize_general_rows)
    elif method == "compact_contact_boundary":
        from . import compact_contact  # noqa: PLC0415 - the owner's module; loaded with the owner

        produce = (compact_contact.produce_contacts,)
        if size == solver._sparse_diagonal_response_size:
            # The marker encodes schedule links into the freshly published sparse DOF slots.
            kernels = (*produce, solver._mark_independent_sparse_contact_candidates_kernel)
        else:
            kernels = (compact_contact.clear_contact_response, *produce)
    elif method == "sparse_response":
        kernels = (*prepare, solver._sparse_contact_response_kernel)
    elif method == "sparse_diagonal":
        kernels = (*prepare, response_builds_rows.sparse_pair_contact_rows_kernel)
        if solver._sparse_diagonal_contact_triples or solver._sparse_diagonal_contact_units:
            kernels = (*kernels, solver._mark_independent_sparse_contact_candidates_kernel)
    elif method == "compact_jacobian":
        kernels = (*prepare, source.populate_world_J_for_compact_size)
    else:
        kernels = _general_kernels(method, source)
    entry = {"method": method, "kernels": _kernel_names(kernels), "law": "patch" if patch else "point"}
    if method == "keyed_rows":
        entry["direct_rows"] = "routes.sweep.tiers"
    return entry


def _contact_rows(solver, source) -> dict:
    point = bool(solver._laws.point)
    # The plan _stage4_build_rows dispatches on: the first admitted producer writes every
    # contact row; the general builder serves only solvers no earlier producer took.
    plan = contact_row_plan(solver)
    methods = None
    if plan.producer == "general":
        methods = {str(int(size)): method for size, method in plan.methods.items()}
    packets = solver._row_packets
    return {
        "producer": plan.producer,
        "law": plan.law,
        # Every producer, every size: the builder method, its kernels in launch order and the law they write,
        # for the dense contact rows built outside the sweep (_row_builder names the rows built elsewhere).
        "builder_by_dof": {str(int(size)): _row_builder(solver, plan, size, source) for size in plan.methods},
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
            "reason": solver._compact_contact_reason,
        },
        "compact_jacobian": bool(solver._compact_contact_jacobian),
    }


def _contact_consumers(solver) -> dict:
    blocks = int(solver._contact_capacity_blocks)
    return {
        "max_blocks": blocks,
        "reason": (
            "Launches over the contact capacity (force export, warm-start snapshot and gathers, keyed-row key"
            f" scatters) grid-stride over {CAPACITY_BLOCKS_PER_SM} blocks of 256 threads per SM, about one wave."
            if blocks
            else "Uncapped: a CPU launch, or a kernel compiled without the grid-stride loop, covers every slot."
        ),
    }


def _local_solve(solver) -> dict:
    enabled = bool(solver._local_internal_fast_path)
    # The candidate counts predate sparse admission, which can retire the owners; report built owners only.
    return {
        "enabled": enabled,
        "internal_articulations_by_dof": _counts(solver._local_internal_art_counts) if enabled else {},
        "pair_articulations_by_dof": _counts(solver._local_pair_art_counts) if enabled else {},
        "residual_articulations_by_dof": _counts(solver._local_residual_art_counts) if enabled else {},
        "reason": solver._local_solve_reason,
    }


def _paired_response(solver) -> dict:
    primary = solver._paired_response_primary_size
    if primary is None:
        return {"enabled": False, "reason": solver._paired_response_reason}
    factor = bool(solver._paired_factor_coordinates)
    overlap = solver._paired_general_solve_stream is not None
    return {
        "enabled": True,
        "primary_dofs": int(primary),
        "secondary_dofs": int(solver._paired_response_secondary_size),
        "factor_coordinates": factor,
        # Why the paired response keeps its register inverse instead of factor coordinates (None when admitted).
        "factor_coordinates_reason": solver._factor_coordinates_reason,
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
    return {
        "enabled": bool(solver._sparse_diagonal_contact_solve),
        "response_dofs": json_value(solver._sparse_diagonal_response_size),
        "dense_dofs": int(solver._sparse_diagonal_dense_size),
        "contact_triples": bool(solver._sparse_diagonal_contact_triples),
        "contact_units": bool(solver._sparse_diagonal_contact_units),
        "speculative_contact_batches": bool(solver._sparse_diagonal_speculative_contact_batches),
        "fused_joint_limits": bool(solver._fused_diagonal_joint_limits),
        "reason": solver._sparse_diagonal_reason,
    }


def _sparse_mass_matrix(solver) -> dict:
    size = solver._sparse_mass_matrix_size
    record = {
        "size": json_value(size),
        "reason": solver._sparse_selection_reason,
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
    }
    if size is not None:
        record.update(
            {
                "max_row_support": json_value(solver._sparse_max_row_support),
                "solve_lanes": int(solver._sparse_solve_lanes),
                "local_factors": bool(solver._sparse_local_factors),
                "factor_warps_per_block": int(solver._sparse_factor_warps_per_block),
                "contact_response_kernel": _key(solver._sparse_contact_response_kernel),
                "sweep_kernel": _key(solver._pgs_solve_sparse_kernel),
                # The sweep's launch bounds from its shared memory and register budget (threads, minimum blocks).
                "launch_bounds": json_value(solver._pgs_solve_sparse_kernel.options.get("launch_bounds")),
            }
        )
    return record


def _sweep(solver) -> dict:
    solve = solver._plan.solve
    route, driver = solve.route.name, solve.driver.name
    dense_split = solve.assembles_delassus
    dense = None
    if dense_split:
        dense = {"kernels": [_key(solver._pgs_solve_loop_kernel)]}
    free_body = None
    if dense_split and solver._has_free_rigid_bodies:
        free_body = _key(solver._pgs_solve_mf_loop_kernel)
    mf_meta_pack = None
    if solver._matrix_free:
        # Only free rigid bodies produce matrix-free rows: the pack kernel is built where it launches.
        packed = solver._pack_mf_meta_kernel is not None
        mf_meta_pack = {
            "launched": packed,
            "kernel": _key(solver._pack_mf_meta_kernel),
            "reason": "Free rigid-body rows pack their metadata each step."
            if packed
            else "Skipped: without free rigid bodies no matrix-free row exists.",
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
        "reason": solve.route.reason,
        "driver": driver,
        "driver_reason": solve.driver.reason,
        "schedule": solver.pgs_schedule,
        "iterations": int(solver.pgs_iterations),
        "velocity_iterations": int(solver.pgs_velocity_iterations),
        "mf_meta_pack": mf_meta_pack,
        "sparse": _key(solver._pgs_solve_sparse_kernel) if route == "sparse_pgs" else None,
        "sparse_diagonal": _key(solver._pgs_solve_sparse_diagonal_kernel),
        "paired_factor": _key(solver._paired_factor_solve_kernel),
        "mf_gs": _key(solver._pgs_solve_mf_gs_kernel),
        "mf_gs_storage": _mf_gs_storage(solver),
        "mf_colored": _mf_colored(solver),
        "tiers": _tiers(solver),
        "local": local,
        "dense": dense,
        "free_body_matrix_free": free_body,
    }
    record["readout"] = _readout(solver, record)
    record["exit_law"] = _exit_law(solver, record)
    record["contact_update"] = _contact_update(solver, record)
    return record


def _mf_colored(solver) -> dict | None:
    """The colored mapping of the matrix-free sweep: its kernel and the worlds it takes, or ``None`` unbuilt."""
    kernel = solver._pgs_solve_mf_colored_kernel
    if kernel is None:
        return None
    units = solver._colored_units
    rule = units.rule
    return {
        "kernel": _key(kernel),
        "min_units": int(rule.min_units),
        "parallelism": json_value(rule.parallelism),
        # The waves the capable worlds take on the device's SMs, which multiply the units per color a world needs.
        "waves": int(rule.waves),
        "units_per_color": json_value(rule.units_per_color),
        # The worlds whose free bodies admit the rule, each with a storage slot, and that storage's bytes.
        "capable_worlds": int(units.slot_count),
        "storage_bytes": int(units.nbytes),
        "reason": (
            f"At every pack the unit builder gives the colored mapping each capable world (at least "
            f"{rule.units_per_color:g} free bodies: a world's units per color are at most its free bodies) of at least "
            f"{rule.min_units} contact units and {rule.units_per_color:g} units per color: it sweeps a color's units "
            "at once, a block each; the others keep the one-warp mapping (the measured crossover of their solve times, "
            f"{rule.parallelism:g} units per color in one wave of colored worlds on the device, here {rule.waves})."
        ),
    }


def _sweep_kernels(solver, record) -> set:
    """Every sweep kernel key of a sweep record."""
    keys = {record[name] for name in ("sparse", "sparse_diagonal", "paired_factor", "mf_gs", "free_body_matrix_free")}
    keys.add((record["mf_colored"] or {}).get("kernel"))
    keys.update(tier["kernel"] for tier in (record["tiers"] or {}).get("tiers", ()))
    for owner in ("internal", "pair", "residual"):
        keys.update((record["local"] or {}).get(owner, {}).values())
    keys.update((record["dense"] or {}).get("kernels", ()))
    keys.discard(None)
    return keys


def _readout(solver, record) -> dict:
    """Which sweep kernels of the route count their sweeps under ``pgs_solve_readout``, and how a solve counts."""
    counted = solver._solve_readout.kernels
    if solver.pgs_solve_readout == "off":
        reason = "pgs_solve_readout is off."
    elif record["driver"] == "direct":
        reason = "Each launch of a counted kernel solves its worlds."
    elif record["driver"] in ("contact_then_internal", "dense_split"):
        reason = "Two row sets sweep in separate launches, so a solve counts once per launch."
    else:
        reason = (
            f"The {record['driver']} driver runs part of a solve per launch, so counts are per launch; a launch "
            "whose row phase holds none of a world's rows can return before its first sweep, which counts that "
            "world in no slot and not as skipped."
        )
    return {
        "mode": solver.pgs_solve_readout if counted else "off",
        "counted": dict(sorted(counted.items())),
        # Passes whose solves of worlds without rows, which the matrix-free sweeps do not enter, count as skipped.
        "skipped_passes": list(solver._solve_readout.skipped_passes) if counted else [],
        "reason": reason,
    }


def _exit_law(solver, record) -> dict:
    """The exit law every sweep stops a world under (:mod:`.solve_policy`): its floor, in rounding units of float32."""
    return {
        "law": "rounding_floor",
        "quiet_units": QUIET_UNITS,
        "unit_roundoff": json_value(UNIT_ROUNDOFF),
        "reason": (
            "Every sweep stops a world once each row update's change of its own velocity, |dlambda| A_ii, is within "
            f"{QUIET_UNITS} rounding units of the magnitude its update combines: the right-hand side, the terms of "
            "its velocity sum and its own impulse times A_ii."
        ),
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
            "Only the CPU's loop sweeps solve contact units: its assembled dense rows' sweep and its free-body "
            "sweep. The other sweeps keep the sequential law."
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
    tiers = list(solver._mf_gs_tiers)
    if not tiers:
        return None
    if solver._ink_rows > 0:
        return {
            "family": "keyed_parallel" if solver._keyed_rows is not None else "parallel",
            # The tier kernels form their worlds' response in-kernel wherever the model fits, and why.
            "inkernel_response": solver._ink_sizes is not None,
            "inkernel_response_reason": solver._ink_reason,
            # Every built tier projects each friction pair on its patch ring's pooled normal load (patch friction).
            "linked_rows": all(tier.linked_rows for tier in tiers),
            # The last-word velocity-limit pass after the final sweep (mf_gs_parallel_linked_velocity_limit_pass).
            "velocity_limit_pass": all(tier.velocity_limit_pass for tier in tiers),
            # The tiers accumulate joint velocity-limit rows whatever the serial sweeps' law
            # (routes.velocity_limits.law), so the two can differ in the worlds the tiers own.
            "velocity_limit_law": _tier_velocity_limit_law(solver, tiers),
            "nesterov": bool(solver.mf_gs_parallel_nesterov),
            "matrix_free": bool(solver.mf_gs_parallel_matrix_free),
            # The largest tier budget; each tier records its own. Every tier sweeps the automatic budget, at least
            # mf_gs_parallel_sweeps (rule "auto").
            "sweeps": max(tier.sweeps for tier in tiers),
            "sweeps_rule": solver._par_sweeps_rule,
            "sweeps_reason": solver._par_sweeps_reason,
            "tiers": [
                {
                    "rows": int(rows),
                    "min_rows": int(min_rows),
                    "sweeps": tier.sweeps,
                    "kernel": _key(tier.kernel),
                }
                for (rows, min_rows), tier in zip(solver._par_tiers, tiers, strict=False)
            ],
        }
    return {
        "family": "incremental",
        # The incremental tiers are built with the serial sweeps' law.
        "velocity_limit_law": _tier_velocity_limit_law(solver, tiers),
        "tiers": [{"kernel": _key(tier.kernel)} for tier in tiers],
    }


def _tier_velocity_limit_law(solver, tiers) -> str | None:
    # The law the built tier kernels give joint velocity-limit rows; one construction call builds every tier with
    # it. None where no such row can exist. The tiers serve dense rows only, so free-body limits never reach them.
    if not solver.enable_joint_velocity_limits or np.isinf(solver.velocity_limit_activation_fraction):
        return None
    return tiers[0].velocity_limit_law


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


def _publication(solver) -> dict:
    if solver._tree_kinetic_state is not None:
        owner, reason = "cached", "The held-tree owner publishes next-step state."
    else:
        owner, reason = "ordinary", "Separate launches integrate and publish the next state."
    cache = "in_place" if solver._fk_id_cache_enabled else "off"
    return {
        "owner": owner,
        "reason": reason,
        "fk_id_cache": cache,
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
    matrix_free = solver._matrix_free
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
        # Mirrors _allocate_mf_buffers and the row allocation of _stage4_build_rows (both modes). At an
        # infinite activation fraction the allocator still launches but reserves no row.
        "free_body_rows": bool(limits and solver._free_rigid_body_count > 0),
        # The law of split mode's free-body sweep (sweep.free_body_matrix_free) for these rows.
        "free_body_sweep_law": free_body_law,
        "free_body_sweep_law_reason": free_body_law_reason,
        # Split mode's clamp of the solved velocity of free roots (_stage6_clamp_rigid_velocity_limits).
        "root_clamp": bool(limits and solver._has_root_free and not matrix_free),
    }


def _split_free_body_velocity_limit_law(solver) -> tuple[str | None, str]:
    # Mirrors the free_body_matrix_free sweep of _sweep and the kernel selection of _init_tiled_kernels.
    if solver._matrix_free or not solver._has_free_rigid_bodies:
        return None, "No split free-body sweep: CUDA solves the rows matrix-free, or there are no free rigid bodies."
    if solver._has_free_body_velocity_limit_rows:
        return "stateless", (
            "A free body's speed row bounds the magnitude of its linear or angular velocity: past the limit it "
            "pushes, along the velocity, by its whole unrelaxed correction and keeps that correction instead of "
            "an accumulated impulse; the rows end every iteration. The CPU's free-body loop applies this law; "
            "velocity_limit_law selects the law of the matrix-free sweeps and does not apply here."
        )
    if solver._has_rigid_body_velocity_limits:
        return None, "velocity_limit_activation_fraction=inf reserves no free-body velocity-limit row."
    return None, "No free-body velocity-limit rows: the rigid-body speed attributes are not registered."


def _free_rigid_bodies(solver) -> dict:
    count = int(solver._free_rigid_body_count)
    owners = []
    if count:
        if not solver._matrix_free:
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
        "reason": "No free rigid bodies."
        if not count
        else "Owners that can solve free-body rows, in claim order; contacts are routed per world at run time.",
    }


def _pair_sort(patches) -> dict:
    """Record the patch pair sort: over the live slots only (``live``), or a radix sort over the whole capacity.

    The choice follows ``_uses_live_sort``: the live sort whenever the pair keys
    (``key_bits``) leave room for its slot bits.
    """
    capacity = int(patches.capacity)
    if patches.live_sort:
        reason = "The pair sort orders only the live slots, so its cost follows the live contacts."
    else:
        reason = (
            f"{patches.key_bits}-bit pair keys leave no room for the live sort's {SLOT_BITS} slot bits: the "
            "radix sort orders the whole capacity."
        )
    return {
        "live": bool(patches.live_sort),
        "capacity": capacity,
        "key_bits": int(patches.key_bits),
        "reason": reason,
    }


def _friction(solver) -> dict:
    patch = bool(solver._friction_anchors_enabled)
    reason = _BETA_REASONS[solver._friction_anchor_beta_source]
    if patch:
        owners = []
        # Contact regularization keeps them off under either law (parallel_rows.reason).
        if not solver._parallel_contact_paths and not solver._regularization_enabled:
            owners.append("keyed rows and parallel matrix-free tiers")
        # Patch friction reaches the resolved-world classifier on the dense paths only, not under torsion.
        if not solver._laws.patch:
            owners.append("the resolved-world classifier")
        if owners:
            reason += f" Point-contact owners ({', '.join(owners)}) need point friction."
    return {
        "law": "patch" if patch else "point",
        "friction_anchor_beta": json_value(solver.friction_anchor_beta),
        "anchor_beta_source": solver._friction_anchor_beta_source,
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


def _dynamics(solver) -> dict:
    return {
        "compact_diagonal_mass_dofs": json_value(solver._compact_diagonal_mass_size),
        "diagonal_mass_dofs": sorted(int(size) for size in solver._diagonal_mass_sizes),
        # Response layout: whether J/Y are gathered per world or already world-indexed.
        "jy_world_aliased": bool(solver._jy_world_aliased),
        "hinv_jt_writes_world": bool(solver._hinv_jt_writes_world),
        # The matrix-free sweeps' dense rows packed by support (packed_rows), and whether they serve the step alone.
        "packed_rows": {
            "support": int(solver._packed_support) if solver._mf_sweeps else None,
            "dense": bool(solver._packed_dense),
            "response_only": bool(solver._packed_response),
            "reason": solver._packed_response_reason,
        },
        "parallel_augmented_drive_topology": bool(solver._parallel_augmented_drive_topology),
        "double_buffer": bool(solver._double_buffer),
        # Sizes whose double-buffered mass matrix the end of a refresh step clears on the memset
        # stream: empty without a double buffer, where the ordinary route's stage 1 clears H before
        # CRBA. Sizes a held-tree owner factors from its own geometry have no reader
        # (routes.stages.unread_mass_dofs).
        "double_buffer_mass_matrix_clear_dofs": []
        if solver._memset_stream is None
        else sorted(int(size) for size in solver.size_groups if size not in solver._plan.unread_mass_sizes),
        "size_group_streams": any(stream is not None for stream in solver._size_streams.values()),
        "preelimination": bool(solver._preelim_active),
        "mimic_rows": int(solver._mimic_count or 0),
        "connect_rows": int(solver._connect_count or 0),
        "loop_joints": bool(solver._has_loop_joints),
        "prescribed_response": bool(solver._has_prescribed_response),
        "armature_floor": _armature_floor(solver),
    }


def _armature_floor(solver) -> dict:
    # Mirrors _kinematic_state: the armature conditioning_armature added to deep articulations.
    floor = solver._armature_floor
    if floor.evaluated == 0:
        reason = f"No articulation has a root-to-leaf path of {DEEP_CHAIN_JOINTS} joints or more; none is evaluated."
    elif floor.coordinates == 0:
        reason = "Every pivot of the evaluated mass matrices clears 4 n u of its diagonal; no armature is added."
    else:
        reason = (
            "Pivots of the evaluated mass matrices fell below 4 n u of their diagonal, beyond float32 "
            "factorization: the least armature that lifts them there is added."
        )
    return {
        "evaluated_articulations": floor.evaluated,
        "articulations": floor.articulations,
        "coordinates": floor.coordinates,
        "largest": float(floor.largest),
        "reason": reason,
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
