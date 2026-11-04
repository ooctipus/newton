# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The step's spec and plan: every unknown that decides the step's path, gathered once, and what each stage launches.

:class:`StepSpec` is the frozen record of the unknowns, built at the end of construction: the device, the forest, the
route (the laws and the class, today the placement case), the capacities, the mass cadence, the sweep's kernels and
streams, and the diagnostics in force. :class:`StepPlan` and the route record (:mod:`.path_selection`) read it; nothing
in it is computed a second time elsewhere.

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
from dataclasses import asdict, dataclass
from types import MappingProxyType

from ...core.execution import Device, device
from ...sim import ModelFlags
from .forest_plan import ForestPlan
from .route_table import RouteRow

__all__ = ["Capacities", "Route", "SolveRoute", "StepPlan", "StepSpec", "TopologyPin", "first_failed", "kernel_key"]


def kernel_key(kernel) -> str | None:
    """A kernel's key (a factory's product without one: its type's name), or ``None`` for no kernel."""
    return None if kernel is None else getattr(kernel, "key", type(kernel).__name__)


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
class Capacities:
    """The row capacities a world holds, fixed at construction; each class's status word records an overflow."""

    dense_rows: int
    """Dense (articulation) rows a world, its link rows included (``dense_max_constraints`` plus the links)."""
    free_body_rows: int
    """Matrix-free (free-body) rows a world (``mf_max_constraints``)."""
    row_pitch: int
    """A world's rows in the row store: the dense rows, then the free-body rows where free bodies exist."""

    def record(self) -> dict:
        """The capacities as a JSON-compliant record."""
        return asdict(self)


@dataclass(frozen=True)
class StepSpec:
    """Every unknown that decides the step's path, gathered once at the end of construction (frozen).

    The step plan builds its tables and stages from it, and the route record reports it (``routes.spec``, with the
    route under ``routes.row`` and ``routes.laws``). A change of any field is a new solver.
    """

    device: Device
    """The device the step runs on (:func:`.core.execution.device`)."""
    forest: ForestPlan
    """The forest after the body cut: topology, groups, world mapping (frozen)."""
    route: RouteRow
    """The route: the row laws and the class (today the placement case), its producer and its sizes."""
    capacities: Capacities
    mass_interval: int
    """Steps between mass refreshes (``update_mass_matrix_interval``)."""
    articulated: bool
    """The forest holds articulations: without them the step only integrates particles."""
    particles: bool
    """The model holds particles, whose forces and integration the step runs."""
    pure_forest: bool
    """The model is its own forest (:mod:`.body_cut`): the step reads and publishes the caller's states directly."""
    kernels: Mapping[str, str | None]
    """The sweep's kernel keys by route: ``sparse``, ``sparse_diagonal``, ``paired_factor``, ``mf_gs`` and
    ``mf_colored`` (``None`` where the class builds none)."""
    streams: tuple[str, ...]
    """The side streams the step forks onto, by owner."""
    diagnostics: tuple[str, ...]
    """The diagnostics in force, each an optional stage or readout: ``nvtx``, ``snapshots``, ``audit``,
    ``row_watermark``, ``readout``, ``sync_timings``."""

    @classmethod
    def build(cls, solver, route: RouteRow, diagnostics: tuple[str, ...]) -> StepSpec:
        """Gather the spec from the constructed ``solver``, its ``route`` and the ``diagnostics`` in force."""
        model, sweep, placement = solver.model, solver.sweep, solver.placement
        streams = (
            ("dynamics", solver.factor.dynamics_stream),
            ("local_internal", sweep.local_internal_stream),
            ("local_residual", sweep.local_residual_stream),
            ("local_pair", sweep.local_pair_stream),
            ("paired_general", sweep.paired_general_solve_stream),
        )
        return cls(
            device=device(model.device),
            forest=solver.forest,
            route=route,
            capacities=Capacities(
                dense_rows=int(placement.dense_max_constraints),
                free_body_rows=int(placement.mf_max_constraints),
                row_pitch=int(placement.row_pitch),
            ),
            mass_interval=int(solver.options.update_mass_matrix_interval),
            articulated=bool(model.articulation_count),
            particles=bool(model.particle_count),
            pure_forest=bool(solver.cut.pure),
            kernels=MappingProxyType(
                {
                    "sparse": kernel_key(sweep.sparse_kernel) if route.case == "K1" else None,
                    "sparse_diagonal": kernel_key(sweep.sparse_diagonal_kernel),
                    "paired_factor": kernel_key(sweep.paired_factor_kernel),
                    "mf_gs": kernel_key(sweep.mf_gs_kernel),
                    "mf_colored": kernel_key(sweep.mf_colored_kernel),
                }
            ),
            streams=tuple(name for name, stream in streams if stream is not None),
            diagnostics=diagnostics,
        )

    @property
    def case(self) -> str:
        """The class's placement case (:data:`.route_table.CASES`, or ``"K6"``)."""
        return self.route.case

    def record(self) -> dict:
        """The spec's own fields as a JSON-compliant record (``routes.spec``; the route is ``routes.row``)."""
        return {
            "device": {"arch": self.device.arch, "sms": self.device.sms},
            "capacities": self.capacities.record(),
            "mass_interval": self.mass_interval,
            "articulated": self.articulated,
            "particles": self.particles,
            "pure_forest": self.pure_forest,
            "kernels": dict(self.kernels),
            "streams": list(self.streams),
            "diagnostics": list(self.diagnostics),
        }


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

    @classmethod
    def build(cls, spec: StepSpec) -> StepPlan:
        """Read the plan from the step's ``spec``: its route and its forest."""
        return cls(response=MappingProxyType(_response(spec.route, spec.forest)), solve=_solve(spec.route))

    def sizes(self, stage: str, route: str) -> tuple[int, ...]:
        """The sizes a stage's table routes to ``route``."""
        return tuple(size for size, entry in getattr(self, stage).items() if entry.name == route)

    def record(self) -> dict:
        """The tables as a JSON-compliant record (``routes.stages`` of the path selection)."""

        def table(routes):
            return {str(int(size)): route.record() for size, route in routes.items()}

        return {
            "response": table(self.response),
        }


def _response(row: RouteRow, forest) -> dict[int, Route]:
    """Where each size's rows are whitened: by a contact owner (native sparse factors, case K1, or the compact
    boundary), in place by the sparse diagonal (case K2's pair) or in the packed rows' build (every other class)."""
    owner = row.producer == "compact_contact_boundary" or row.case == "K1"
    routes = {}
    for size in forest.size_groups:
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
