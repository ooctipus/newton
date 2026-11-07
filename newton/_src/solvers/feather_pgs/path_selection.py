# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host-only records of the construction-time routes and applied options of SolverFeatherPGS.

Every function here reads host attributes only: no device array is read, no
kernel is launched and nothing synchronizes. The one exception is
:func:`measurement_selection` under an enabled solve readout
(``Diagnostics.solve_readout``), which reads the readout's device counters. Values are
JSON-compliant: non-finite floats become ``"inf"``, ``"-inf"`` or ``"nan"`` and
tuples become lists.
"""

import copy
import dataclasses
import functools
import inspect
import math

import numpy as np

from ...core.live_sort import SLOT_BITS
from ...core.memory import Place
from ...sim.contacts import CAPACITY_BLOCKS_PER_SM
from .solve_policy import QUIET_UNITS, UNIT_ROUNDOFF
from .step_plan import kernel_key as _key


def _applied(solver, name, requested):
    """An option's applied value: the row laws' resolved friction beta, the factor's preelimination, the solver's
    attribute where it keeps one (capacities), else the options record's value."""
    if name == "friction_anchor_beta":
        return solver.laws.friction_anchor_beta
    if name == "enable_bilateral_preelimination":
        return solver.factor.eliminates
    if hasattr(solver, name):
        return getattr(solver, name)
    return getattr(solver.options, name, requested)


_BETA_REASONS = {
    "default": "friction_anchor_beta omitted: persistent patch friction at the 0.2 default.",
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
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_value(dataclasses.asdict(value))
    # Never str(): a Warp array's str() copies device memory to the host.
    return f"<{type(value).__name__}>"


@functools.cache
def _declared_defaults(init) -> dict:
    """Every option's declared default: the parameters after ``self`` and ``model``, but the test pins, which are the
    spec's (``routes.spec.pins``), not an option."""
    parameters = list(inspect.signature(init).parameters.values())[2:]
    return {parameter.name: parameter.default for parameter in parameters if parameter.name != "pins"}


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
    # A readout applies only where a sweep kernel carries its counters.
    "diagnostics": lambda solver: dataclasses.replace(
        solver.options.diagnostics,
        solve_readout=solver.options.diagnostics.solve_readout if solver.sweep.readout.kernels else "off",
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
            applied = _applied(solver, name, value)
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


def route_selection(solver) -> dict:
    """Describe every construction-time owner and variant that decides which kernels run.

    The owners' records take whether they hold the class, their sizes and their reasons from the solver's route
    (``routes.row``, :class:`.route_table.RouteRow`, held by the step's plan); the rest of each record is the owner's
    own structure. ``routes.spec`` is the spec's own fields (:meth:`.step_plan.StepSpec.record`) and ``routes.stages``
    the plan's (:meth:`.step_plan.StepPlan.record`).
    """
    row = solver._plan.route

    return {
        "articulations_by_dof": {str(int(size)): int(count) for size, count in solver.forest.n_arts_by_size.items()},
        # The step's spec: the device, the test pins in force (none in production), capacities, cadence and
        # diagnostics (step_plan.StepSpec).
        "spec": json_value(solver.spec.record()),
        # The route: the row of the route table the class takes, its axes and each placement case's reason.
        "row": json_value(row.record()),
        # The row laws every owner's admission reads (route_table.Laws, the spec's).
        "laws": solver.spec.laws.record(),
        # The per-stage dispatch tables step() launches through, the sweep's kernels and streams (step_plan.StepPlan).
        "stages": solver._plan.record(),
        "held_tree": _held_tree(solver),
        "contact_rows": _contact_rows(solver),
        "contact_consumers": _contact_consumers(solver),
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

    With ``Diagnostics.solve_readout`` on, ``solve_readout`` summarizes the sweep counters
    (:meth:`SolveReadoutRecord.summary`); reading them synchronizes the device.
    The counters accumulate, so a window is the difference of two records.
    """
    record = {}
    if solver.sweep.readout.kernels:
        record["solve_readout"] = json_value(solver.sweep.readout.summary())
    return record


def _held_tree_groups(solver) -> tuple:
    owner = solver.factor.held
    groups = list(owner.groups)
    # The one-coordinate class holds no tree layout.
    if any(getattr(getattr(group, "layout", None), "forest", False) for group in groups):
        return "forest", groups
    return ("tree" if solver._plan.route.case == "K1" else "mixed"), groups


def _held_tree(solver) -> dict:
    owner = solver.factor.held
    kind, groups = _held_tree_groups(solver)
    return {
        "owner": kind,
        # Each world class's grouping and why (held_tree.HeldClass).
        "classes": [
            {"worlds": int(held.worlds.size), "grouping": held.grouping, "reason": held.reason}
            for held in owner.plan.classes
        ],
        # Whether the compact diagonal family takes the one-coordinate class (its trees rigid below each leaf), and why.
        "coordinate_trees": {
            "enabled": owner.coordinate_trees is not None,
            "reason": owner.plan.coordinate_reason,
        },
        "groups": [
            _coordinate_group(group, owner.drive_dofs) if group is owner.coordinate_trees else _tree_group(group)
            for group in groups
        ],
    }


def _tree_group(group) -> dict:
    layout, operator = group.layout, group.sparse
    return {
        "operator": operator.kind,
        "size": json_value(getattr(operator, "size", None)),
        "group_count": int(group.group_count),
        "bodies": int(layout.bodies),
        "dofs": int(layout.dofs),
        "lanes": int(layout.lanes),
        "trees_per_block": int(layout.pack),
        "block_dim": int(group.block_dim),
        "roots": json_value(layout.roots),
        "root_coordinates": json_value(layout.root_coordinates),
        # Per root, whether it stands on a free joint (a joint root's coordinates are a descendant's).
        "free_roots": json_value(layout.free_roots),
        # Factor family per root: its coordinate count, negated for a diagonal-mass family, 0 when prescribed.
        "root_families": json_value(layout.root_families),
        # The scans' schedules, chosen from the tree's data with its lanes (tree_kinetic_state.held_schedules).
        "schedules": {name: getattr(group.schedules, name).value for name in ("pose", "motion", "reduce")},
        "response": group._response,
        # "content": a repair whose input bank is not the bank the cache was published into
        # compares the two banks word for word; "identity": the bank's pointers alone.
        "input_validation": "content" if group.bank_compare else "identity",
        "motion_velocity": bool(group.motion_velocity),
        "publishes_twists": bool(group.publishes_twists),
        "kernels": {
            "repair": _key(group.repair_kernel),
            "finish": _key(group.finish_kernel),
            "predict": _key(group.predictor_kernel),
        },
    }


def _coordinate_group(group, drive_dofs) -> dict:
    """The one-coordinate class (:mod:`.coordinate_trees`) in the tree groups' schema.

    Its predict and finish run one thread per coordinate (``lanes``) and its
    repair one warp per articulation, so the tree layout's packing fields
    (``trees_per_block``, ``block_dim``) are ``None``. Its repair compares no
    input banks (``input_validation`` is the group's, "identity": the stamp
    alone). ``drive_dofs`` counts the DOFs whose drives the drive kernel
    still prepares (the held plan's, ``drive_dofs``), since the class's predict decides its own.
    """
    size = int(group.size)
    return {
        "operator": type(group).__name__,
        "size": size,
        "group_count": int(group.group_count),
        # The root, a body a leaf and the leaves' welded bodies.
        "bodies": size + 1 + int(group.topology.weld_joint_offset.size),
        "dofs": size,
        "lanes": size,
        "trees_per_block": None,
        "block_dim": None,
        "roots": [0],
        "root_coordinates": [0],
        "free_roots": [False],
        # A diagonal-mass family, negated as for the tree groups.
        "root_families": [-size],
        "schedules": None,
        "response": "diagonal",
        "input_validation": "content" if group.bank_compare else "identity",
        "motion_velocity": bool(group.motion_velocity),
        "publishes_twists": False,
        "drive_dofs": 0 if drive_dofs is None else int(drive_dofs.shape[0]),
        "kernels": {
            "repair": _key(group.repair_kernel),
            "finish": _key(group.finish_kernel),
            "predict": _key(group.predictor_kernel),
        },
    }


def _kernel_names(kernels) -> list:
    return [kernel if isinstance(kernel, str) else _key(kernel) for kernel in kernels]


def _row_builder(solver, method, size) -> dict:
    """Name the kernels that build one DOF size's dense contact rows outside the sweep.

    Mirrors the producers' launches, in launch order: the one writer, and KeyedRows.produce_contacts and
    finish_rows.
    A launch shared by several sizes (a producer's contact kernel) is named under each of them. Rows built elsewhere are not named: keyed rows' direct rows, which the parallel tiers
    build in the sweep (``direct_rows`` points to routes.sweep.tiers), and the free-body matrix-free
    rows, which walk_rows.write_free_body_rows writes for the contacts the allocation routes to them
    (routes.free_rigid_bodies).
    """
    marker = ()
    if size == solver.spec.classes.diagonal_size:
        # The sparse diagonal's marker encodes schedule links into the slots its writer just wrote.
        marker = () if solver.sweep.sparse_contact_marker is None else (solver.sweep.sparse_contact_marker,)
    if method == "walk":
        from . import walk_rows  # noqa: PLC0415 - the one writer's module

        # The allocator writes the contact rows' unit sources; the one writer writes every contact row's records and
        # walks every row (walk_rows): the general placement's into its packed rows, the support placement's (K1)
        # into its support sink, the sparse diagonal's (K2) into its slots sink.
        if solver._plan.route.case == "K1":
            writers = (solver.sweep.support_units_kernel, solver.sweep.support_rows)
        elif solver._plan.route.case == "K2":
            writers = (walk_rows.slot_writer(solver),)
        else:
            # The general placement's rows in two parts: the topology's, walked in blocks, and the rest.
            writers = (walk_rows.writer_kernel(solver, walk_rows.WRITE_TOPOLOGY), walk_rows.writer_kernel(solver))
        kernels = (*writers, *marker)
    elif method == "keyed_rows":
        from . import keyed_rows, walk_rows  # noqa: PLC0415 - the owner's module; loaded with the owner

        # The parallel tiers (routes.sweep.tiers) build the direct rows in the sweep; the one writer walks the rows
        # of every other world (keyed_general_units hands them over).
        kernels = (keyed_rows.scatter_contact_keys, keyed_rows.keyed_general_units, walk_rows.writer_kernel(solver))
    else:
        raise ValueError(f"No contact-row builder kernels are known for the method {method!r}.")
    entry = {"method": method, "kernels": _kernel_names(kernels)}
    if method == "keyed_rows":
        entry["direct_rows"] = "routes.sweep.tiers"
    return entry


def _contact_rows(solver) -> dict:
    row = solver._plan.route
    # The route's producer writes every contact row (the row pass, RowLaws.write_contacts, takes it from the plan).
    sizes = solver.forest.size_groups if row.producer != "none" else ()
    return {
        "producer": row.producer,
        # Every producer, every size: the builder method, its kernels in launch order and the law they write,
        # for the dense contact rows built outside the sweep (_row_builder names the rows built elsewhere).
        "builder_by_dof": {str(int(size)): _row_builder(solver, row.producer, size) for size in sizes},
        "keyed_rows": {"enabled": row.producer == "keyed_rows", "reason": solver.placement.keyed_rows_reason},
    }


def _contact_consumers(solver) -> dict:
    blocks = int(solver.placement.contact_capacity_blocks)
    return {
        "max_blocks": blocks,
        "reason": (
            "Launches over the contact capacity (force export, warm-start snapshot and gathers, keyed-row key"
            f" scatters) grid-stride over {CAPACITY_BLOCKS_PER_SM} blocks of 256 threads per SM, about one wave."
            if blocks
            else "Uncapped: a CPU launch, or a kernel compiled without the grid-stride loop, covers every slot."
        ),
    }


def _paired_response(solver) -> dict:
    row = solver._plan.route
    primary = solver.sweep.paired_response_primary_size
    if primary is None:
        return {"enabled": False, "reason": solver.sweep.paired_response_reason}
    factor = row.case == "K5"
    return {
        "enabled": True,
        "primary_dofs": int(primary),
        "secondary_dofs": int(solver.sweep.paired_response_secondary_size),
        "factor_coordinates": factor,
        # Why the paired response keeps its register inverse instead of factor coordinates (None when admitted).
        "factor_coordinates_reason": row.reasons["K5"],
        "contact_units": bool(solver.sweep.factor_coordinate_contact_units),
        "reason": "One articulation factor and one free body per world share one response"
        + (" in factor coordinates." if factor else " with a register inverse."),
    }


def _sparse_diagonal(solver) -> dict:
    row, diagonal = solver._plan.route, solver.spec.classes.diagonal
    return {
        "enabled": row.case == "K2",
        "response_dofs": json_value(row.sizes["diagonal"]),
        "dense_dofs": int(row.sizes["dense"] or 0),
        "contact_units": bool(diagonal is not None and diagonal.units),
        "speculative_contact_batches": bool(diagonal is not None and diagonal.speculative_batches),
        "fused_joint_limits": bool(solver.spec.classes.fused_sizes),
        "reason": row.reasons["K2"],
    }


def _sparse_mass_matrix(solver) -> dict:
    row = solver._plan.route
    size = row.sizes["packed"]
    record = {
        "size": json_value(size),
        "reason": row.reasons["K1"],
        "velocity_limits": bool(solver.sweep.sparse_velocity_limits),
        "active_rows": bool(solver.sweep.sparse_active_rows),
        "active_rows_reason": solver.sweep.sparse_active_rows_reason,
        "support_table": bool(solver.placement.sparse_row_support.enabled),
        "contact_triplets": bool(solver.sweep.sparse_contact_triplets),
        "contact_triplets_reason": solver.sweep.sparse_contact_triplets_reason,
    }
    if size is not None:
        record.update(
            {
                "max_row_support": json_value(solver.spec.classes.packed.max_support),
                "solve_lanes": int(solver.sweep.sparse_solve_lanes),
                "local_factors": bool(solver.sweep.sparse_local_factors),
                "units_kernel": _key(solver.sweep.support_units_kernel),
                "writer_kernel": _key(solver.sweep.support_rows),
                "leave_kernel": _key(solver.factor.walk_leave),
                "sweep_kernel": _key(solver.sweep.sparse_kernel),
                # The sweep's launch bounds from its shared memory and register budget (threads, minimum blocks).
                "launch_bounds": json_value(solver.sweep.sparse_kernel.options.get("launch_bounds")),
            }
        )
    return record


def _sweep(solver) -> dict:
    solve = solver._plan.solve
    route = solve.route.name
    # The streamed rows' records, which the rows' writers store where a sweep streams a row.
    placement = solver.placement
    records = {
        "written": bool(placement.writes_records),
        "first": int(placement.record_first),
        "reason": "The rows' writers store the streamed rows' records."
        if placement.writes_records
        else "Skipped: no sweep streams a row's metadata.",
    }
    kernels = solver._plan.kernels
    record = {
        "route": route,
        "reason": solve.route.reason,
        "iterations": int(solver.options.pgs_iterations),
        "row_records": records,
        "sparse": kernels["sparse"],
        "sparse_diagonal": kernels["sparse_diagonal"],
        "paired_factor": kernels["paired_factor"],
        "mf_gs": kernels["mf_gs"],
        "mf_gs_storage": _mf_gs_storage(solver),
        "mf_colored": _mf_colored(solver),
        "tiers": _tiers(solver),
    }
    record["readout"] = _readout(solver, record)
    record["exit_law"] = _exit_law(solver, record)
    return record


def _mf_colored(solver) -> dict | None:
    """The colored mapping of the matrix-free sweep: its kernel and the worlds it takes, or ``None`` unbuilt."""
    kernel = solver.sweep.mf_colored_kernel
    if kernel is None:
        return None
    units = solver.sweep.colored_units
    rule = units.rule
    return {
        "kernel": solver._plan.kernels["mf_colored"],
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


def _readout(solver, record) -> dict:
    """Which sweep kernels of the route count their sweeps under ``Diagnostics.solve_readout``, and how a solve
    counts."""
    counted = solver.sweep.readout.kernels
    readout = solver.options.diagnostics.solve_readout
    if readout == "off":
        reason = "The solve readout is off."
    else:
        reason = "Each launch of a counted kernel solves its worlds."
    return {
        "mode": readout if counted else "off",
        "counted": dict(sorted(counted.items())),
        # Whether the solves of worlds without rows, which the matrix-free sweeps do not enter, count as skipped.
        "counts_skipped": bool(counted) and solver.sweep.readout.counts_skipped,
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


def _mf_gs_storage(solver) -> dict | None:
    # Storage classes of the MF-GS kernel, chosen by execution_plan._select_mfgs_storage.
    storage = solver.sweep.mf_gs_storage
    if storage is None:
        return None

    def storage_class(place):
        return "resident" if place is Place.SHARED else "streamed"

    places = storage.places
    return {
        "metadata": storage_class(places.metadata),
        "mf_impulses": storage_class(places.impulses),
        "dense_impulses": storage_class(places.dense_impulses),
        "reason": storage.reason,
    }


def _tiers(solver) -> dict | None:
    row = solver._plan.route
    tiers = list(solver.sweep.tiers)
    if not tiers:
        return None
    return {
        "family": "keyed_parallel" if row.producer == "keyed_rows" else "parallel",
        # Why the class takes the tiers (case K4: the in-kernel response's shape).
        "inkernel_response_reason": row.reasons["K4"],
        # Every built tier projects each friction pair on its patch ring's pooled normal load (patch friction).
        "linked_rows": all(tier.config.linked_rows for tier in tiers),
        # The last-word velocity-limit pass after the final sweep (linked tiers with joint velocity limits).
        "velocity_limit_pass": all(tier.config.velocity_limit_pass for tier in tiers),
        # The largest tier budget; each tier records its own (the equal-accuracy rule "auto", or "pinned").
        "sweeps": max(tier.config.sweeps for tier in tiers),
        "sweeps_rule": solver.sweep.par_sweeps_rule,
        "sweeps_reason": solver.sweep.par_sweeps_reason,
        "tiers": [
            {
                "rows": int(rows),
                "min_rows": int(min_rows),
                "sweeps": tier.config.sweeps,
                "kernel": _key(tier.kernel),
            }
            for (rows, min_rows), tier in zip(solver.sweep.par_tiers, tiers, strict=True)
        ],
    }


def _resolved_worlds(solver) -> dict:
    owner = solver.sweep.resolved_worlds_owner
    published = bool(owner is not None and owner.published)
    if owner is None:
        twists = None
    elif published:
        twists = "The held-tree predictor publishes body twists for every responding articulation."
    else:
        twists = "The classifier builds body twists itself: a responding articulation has no twist-publishing tree."
    return {
        "enabled": owner is not None,
        "reason": solver.sweep.resolved_worlds_reason,
        "publishes_predicted_twists": published,
        "twists_reason": twists,
    }


def _publication(solver) -> dict:
    return {
        "owner": "cached",
        "reason": "The held-tree owner publishes next-step state.",
        "fk_id_cache": "in_place",
    }


def _velocity_limits(solver) -> dict:
    if not solver.options.enable_joint_velocity_limits:
        joint, reason = "off", "enable_joint_velocity_limits is False."
    elif solver.sweep.sparse_velocity_limits:
        joint, reason = "sparse_factor_rows", "Native sparse factors solve velocity-limit rows in factor coordinates."
    else:
        joint, reason = "rows", "Dedicated velocity-limit rows."
    limits = bool(solver.laws.has_rigid_body_velocity_limits)
    model = solver.model
    prescale = solver.options.enable_joint_velocity_limits and model.joint_count > 0 and model.articulation_count > 0
    return {
        "joint": joint,
        "reason": reason,
        # Lanes per articulation of the pre-solve PhysX velocity prescale; 0 when it launches no kernel.
        "prescale_lanes": int(solver.laws.velocity_prescale_lanes) if prescale else 0,
        "activation_fraction": json_value(solver.options.velocity_limit_activation_fraction),
        "rigid_body_limits": limits,
        # Mirrors Placement.allocate_free_body_rows and the row allocation of RowLaws.write_contacts (both modes). At an
        # infinite activation fraction the allocator still launches but reserves no row.
        "free_body_rows": bool(limits and solver.forest.free_rigid_body_count > 0),
    }


def _free_rigid_bodies(solver) -> dict:
    row = solver._plan.route
    count = int(solver.forest.free_rigid_body_count)
    owners = []
    if count:
        if row.case == "K1":
            owners.append("sparse_free_rows")
        if solver.sweep.paired_response_primary_size is not None:
            owners.append("paired_factor" if row.case == "K5" else "paired_response")
        owners.append("matrix_free_rows")
    return {
        "count": count,
        "mixed_worlds": bool(solver.forest.has_mixed_contacts),
        # Whether their contacts keep a run of their own (the colored worlds'), or are rows of the line.
        "contact_run": bool(solver.placement.free_contact_rows),
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
    return {
        "friction_anchor_beta": json_value(solver.laws.friction_anchor_beta),
        "anchor_beta_source": solver.laws.friction_anchor_beta_source,
        "torsion": bool(solver.laws.spin_rows),
        "compliance": bool(solver.options.contact_compliance),
        "reason": _BETA_REASONS[solver.laws.friction_anchor_beta_source],
        "frame_reuse": {
            "enabled": solver.laws.friction_patches.reuse_frames,
            "reason": solver.laws.patch_frame_reuse_reason,
        },
        "pair_sort": _pair_sort(solver.laws.friction_patches),
    }


def _dynamics(solver) -> dict:
    return {
        "compact_diagonal_mass_dofs": json_value(solver.spec.classes.diagonal_size),
        "diagonal_mass_dofs": sorted(int(size) for size in solver.forest.diagonal_mass_sizes),
        # The rows' graph's vertices: how many coordinate trees hold each number of coordinates.
        "tree_sizes": {str(size): int(n) for size, n in enumerate(np.bincount(solver.forest.tree_size)) if n},
        # The matrix-free sweeps' dense rows packed by support (packed_rows), and whether they serve the step alone.
        "packed_rows": {
            "support": int(solver.placement.packed_support) if solver.spec.classes.mf_sweeps else None,
            "dense": bool(solver.placement.packed_dense),
            "response_only": bool(solver.placement.packed_response),
            "reason": solver.placement.packed_response_reason,
        },
        "parallel_augmented_drive_topology": bool(solver.factor.parallel_drives),
        "preelimination": bool(solver.factor.eliminates),
        "mimic_rows": int(solver.laws.mimic_count or 0),
        "link_rows": int(solver.laws.link_count or 0),
        "prescribed_response": bool(solver.forest.has_prescribed_response),
    }


def _diagnostics(solver) -> dict:
    """Opt-in per-step diagnostics recorded so far: the stage-3 snapshots, which the solver never reads."""
    requested = solver.options.diagnostics.snapshots
    # Mirrors _allocate_common_buffers: a model without joints has no stage 3 to snapshot.
    snapshots = solver._debug_stage3_qd_work is not None
    if snapshots:
        reason = (
            "Diagnostics.snapshots: every substep copies the stage-3 joint velocity, acceleration and predicted "
            "velocity into _debug_stage3_qd_work, _debug_stage3_joint_qdd and _debug_stage3_v_hat."
        )
    elif requested:
        reason = "Diagnostics.snapshots, but the model has no joints to snapshot."
    else:
        reason = "Off: Diagnostics.snapshots opts in to three stage-3 snapshot copies per substep."
    audit = solver._audit
    return {
        "stage3_snapshots": {"enabled": snapshots, "reason": reason},
        "audit": {"enabled": False, "reason": "Off: Diagnostics.audit opts in."} if audit is None else audit.record(),
    }
