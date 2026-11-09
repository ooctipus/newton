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

import dataclasses
import math

import numpy as np

from .configuration import declared_options


def kernel_key(kernel) -> str | None:
    """A kernel's key (a factory's product without one: its type's name), or ``None`` for no kernel."""
    return None if kernel is None else getattr(kernel, "key", type(kernel).__name__)


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


def _is_default(value, default) -> bool:
    if value is default:
        return True
    try:
        same = value == default
    except Exception:
        return False
    return bool(same) if isinstance(same, (bool, np.bool_)) else False


def applied_options(init, arguments: dict, applied) -> dict:
    """Record every constructor option's requested, declared-default and applied value.

    ``arguments`` is ``locals()`` taken as the constructor's first statement, ``applied`` the options in effect after
    construction's gates and clamps (:func:`.configuration.check_options`): an option whose owner, tier or stream was not
    built reports its off value (``False``, ``0`` or ``"off"``). ``routes`` names the owners. ``source`` is
    ``"argument"`` when the requested value differs from the declared default and ``"default"`` otherwise; Python
    cannot tell an omitted option from one passed with its default value.
    """
    return {
        name: {
            "applied": json_value(getattr(applied, name)),
            "requested": json_value(arguments[name]),
            "default": json_value(default),
            "source": "default" if _is_default(arguments[name], default) else "argument",
        }
        for name, default in declared_options(init).items()
    }


def route_selection(solver) -> dict:
    """Describe every construction-time owner and variant that decides which kernels run.

    The owners' records take whether they hold the class, their sizes and their reasons from the solver's route
    (``routes.row``, :class:`.route_table.RouteRow`, held by the step's plan); the rest of each record is the owner's
    own structure. ``routes.spec`` is the spec's own fields (:meth:`.step_plan.StepSpec.record`) and ``routes.stages``
    the plan's (:meth:`.step_plan.StepPlan.record`).
    """
    plan = solver._plan
    return {
        "articulations_by_dof": {str(int(size)): int(count) for size, count in solver.forest.n_arts_by_size.items()},
        # The step's spec: the device, the test pins in force (none in production), capacities, cadence and
        # diagnostics (step_plan.StepSpec).
        "spec": solver.spec.record(),
        # The route: the row of the route table the class takes, its axes and each placement case's reason.
        "row": json_value(plan.route.record()),
        # The row laws every owner's admission reads (route_table.Laws, the spec's).
        "laws": solver.spec.laws.record(),
        # The per-stage dispatch tables step() launches through, the sweep's kernels and streams (step_plan.StepPlan).
        "stages": plan.record(),
        **solver.factor.held.record(plan.route.support),
        **solver.laws.record(solver),
        **solver.placement.record(),
        **solver.sweep.record(plan),
        **solver.factor.record(solver.spec.classes, solver.placement),
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
