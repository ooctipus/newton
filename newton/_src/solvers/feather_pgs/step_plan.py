# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The step's plan: what each stage launches, decided once at construction.

Every stage whose work depends on the articulation size (the factor and where stage 4
whitens the rows) has one table: each size's route and the reason for it. The solve of stages 5 and 6 has one
entry: the rows its sweep solves and what drives the sweep. ``step()`` launches through these tables,
the route record (:mod:`.path_selection`) reports them, and construction builds
exactly the kernels they name, so the three cannot disagree.

The plan reads the solver's route (its row of the route table: the placement case
and producer) and the mapping beside it (the
execution plan's resource choices, the kernel choices); it decides nothing that
construction has not, and nothing in it changes after construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ...sim import ModelFlags
from .route_table import RouteRow

__all__ = ["Route", "SolveRoute", "StepPlan", "TopologyPin", "first_failed"]


def first_failed(checks) -> str | None:
    """The reason of the first ``(holds, reason)`` check that fails, or ``None`` when every check holds."""
    return next((reason for holds, reason in checks if not holds), None)


NUMERIC_PROPERTIES = (
    ModelFlags.JOINT_DOF_PROPERTIES
    | ModelFlags.BODY_INERTIAL_PROPERTIES
    | ModelFlags.SHAPE_PROPERTIES
    | ModelFlags.MODEL_PROPERTIES
)
"""The model notifications that change only numbers (coefficients, limits, inertias, shape materials), never a row
owner's topology."""

JOINT_TOPOLOGY = ("joint_type", "joint_parent", "joint_child", "joint_q_start", "joint_qd_start", "body_flags")
"""The structural joint and body arrays every row owner's plan reads."""


@dataclass(frozen=True)
class TopologyPin:
    """The structural model arrays a row owner's plan was built from, read at construction (:meth:`read`)."""

    names: tuple[str, ...]
    data: tuple[bytes, ...]

    @classmethod
    def read(cls, model, names) -> TopologyPin:
        names = tuple(names)
        return cls(names, tuple(getattr(model, name).numpy().tobytes() for name in names))

    def validate(self, model, flags, holds, owner: str) -> None:
        """Refuse a notification beyond :data:`NUMERIC_PROPERTIES` that changes a pinned array or after which the
        owner's own check ``holds()`` fails; a numeric notification reads nothing back."""
        if flags and not (int(flags) & ~int(NUMERIC_PROPERTIES)):
            return
        if TopologyPin.read(model, self.names) != self or not holds():
            raise RuntimeError(f"{owner} topology changed; reconstruct the solver")


@dataclass(frozen=True)
class Route:
    """One stage's route and why it was chosen."""

    name: str
    reason: str

    def record(self) -> dict:
        """The route as a JSON-compliant record."""
        return {"route": self.name, "reason": self.reason}


@dataclass(frozen=True)
class SolveRoute:
    """Stages 5 and 6: the rows the sweep solves, all iterations in one dispatch."""

    route: Route
    """The rows the sweep solves: ``sparse_pgs``, ``sparse_diagonal`` or ``matrix_free_gs``."""


@dataclass(frozen=True)
class StepPlan:
    """The per-stage dispatch tables of one solver, built once by :meth:`build`."""

    response: Mapping[int, Route]
    """Stage 4, per size, where its rows are whitened: ``owner`` (a contact owner publishes the response),
    ``sparse_diagonal`` or ``packed`` (the packed rows' build)."""
    solve: SolveRoute
    jacobian_clear_sizes: tuple[int, ...]
    """The sizes whose rows of the last step are cleared before the producers write the next: none under the
    matrix-free sweeps (their producers write the packed rows' sides whole), native sparse factors or owners that keep
    a private Jacobian (local row packets, keyed rows), and never the sparse-diagonal response size (no dense J) or
    the compact producer's six-coordinate size (complete stores)."""

    @classmethod
    def build(cls, row: RouteRow, solver) -> StepPlan:
        """Read the plan from the solver's route ``row`` and its mapping (sizes, resource rules, kernel choices)."""
        solve = _solve(row)
        return cls(
            response=MappingProxyType(_response(row, solver)),
            solve=solve,
            jacobian_clear_sizes=()
            if row.producer in ("local_packets", "keyed_rows")
            else tuple(size for size in solver.forest.size_groups if solver.placement.clears_jacobian(size)),
        )

    def sizes(self, stage: str, route: str) -> tuple[int, ...]:
        """The sizes a stage's table routes to ``route``."""
        return tuple(size for size, entry in getattr(self, stage).items() if entry.name == route)

    def record(self) -> dict:
        """The tables as a JSON-compliant record (``routes.stages`` of the path selection)."""

        def table(routes):
            return {str(int(size)): route.record() for size, route in routes.items()}

        return {
            "response": table(self.response),
            "jacobian_clear_dofs": [int(size) for size in self.jacobian_clear_sizes],
        }


def _response(row: RouteRow, solver) -> dict[int, Route]:
    """Where each size's rows are whitened: by a contact owner (native sparse factors, case K1, or the compact
    boundary), in place by the sparse diagonal (case K2's pair) or in the packed rows' build (every other class)."""
    owner = row.producer == "compact_contact_boundary" or row.case == "K1"
    routes = {}
    for size in solver.forest.size_groups:
        if owner:
            routes[size] = Route(
                "owner", "The contact owner and the bounded limit prefix publish the complete response."
            )
        elif row.case == "K2" and size in (row.sizes["diagonal"], row.sizes["dense"]):
            routes[size] = Route(
                "sparse_diagonal", "The sparse-diagonal owner whitens its rows in place: no size builds a response."
            )
        elif row.packed:
            routes[size] = Route(
                "packed", "The matrix-free sweeps read the rows packed by support; no route reads this response."
            )
        else:
            raise RuntimeError(f"No placement whitens the rows of size {size}: {row}")
    return routes


def _solve(row: RouteRow) -> SolveRoute:
    if row.case == "K1":
        route = Route("sparse_pgs", "Native sparse factors own the sweep.")
    elif row.case == "K2":
        route = Route("sparse_diagonal", "The diagonal-mass contact owner owns the sweep.")
    else:
        route = Route(
            "matrix_free_gs", "Matrix-free Gauss-Seidel; local, paired and tiered owners claim their worlds first."
        )
    return SolveRoute(route=route)
