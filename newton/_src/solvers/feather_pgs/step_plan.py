# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The step's plan: what each stage launches, decided once at construction.

Every stage whose work depends on the articulation size (the stage-2 factor, the
stage-3 solve of the velocity predictor, the stage-4 contact response and the
Delassus assembly) has one table: each size's route and the reason for it. The
solve of stages 5 and 6 has one entry: the branch ``step()`` runs, the rows its
sweep solves and what drives the sweep. ``step()`` launches through these tables,
the route record (:mod:`.path_selection`) reports them, and construction builds
exactly the kernels they name, so the three cannot disagree.

The plan reads construction's decisions (owners, the execution plan's resource
choices, the options); it decides nothing that construction has not, and nothing
in it changes after construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ...sim import ModelFlags

__all__ = ["Route", "SolveRoute", "StepPlan", "TopologyPin", "first_failed"]

_GROUPED_VERIFIED_DOFS = 32


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
    """Stages 5 and 6: the branch of ``step()``, the rows the sweep solves and what drives the sweep."""

    family: str
    """``"matrix_free"`` (CUDA), ``"split_interleaved"`` or ``"dense_split"`` (the CPU): the branch of ``step()``."""
    route: Route
    """The rows the sweep solves: ``dense_loop`` (the CPU), ``sparse_pgs``, ``sparse_diagonal`` or
    ``matrix_free_gs``."""
    driver: Route
    """What drives the route's sweep each solve: ``direct``, ``split_interleaved``, ``dense_split``, ``debug_loop``,
    ``contact_compliance``, ``contact_then_internal`` or ``physx_grasp``."""

    @property
    def assembles_delassus(self) -> bool:
        """Whether stage 4 assembles the Delassus operator (split mode's dense rows) instead of only the response."""
        return self.family in ("split_interleaved", "dense_split")


@dataclass(frozen=True)
class StepPlan:
    """The per-stage dispatch tables of one solver, built once by :meth:`build`."""

    factor: Mapping[int, Route]
    """Stage 2, per size: ``none`` (no dense factor), ``held`` (the held tree refreshes the factor in stage 1),
    ``diagonal``, ``tiles`` or ``loop``."""
    trisolve: Mapping[int, Route]
    """Stage 3, per size: ``sparse``, ``compact_diagonal``, ``diagonal``, ``tiles`` or ``loop``; empty under a
    held-tree owner, whose predictor replaces the stage."""
    response: Mapping[int, Route]
    """Stage 4, per size: ``owner`` (a contact owner publishes the response), ``sparse_diagonal``,
    ``packed`` (only the packed rows' build: none reads the response), ``diagonal``, ``grouped``, ``tiles``,
    ``lanes`` or ``rows``."""
    delassus: Mapping[int, Route]
    """Stage 4, per size, the CPU's Delassus assembly (``entries``); empty where none is assembled."""
    solve: SolveRoute
    unread_mass_sizes: tuple[int, ...]
    """Sizes whose dense mass matrix no stage reads: every size under a held-tree owner, which refreshes the factors
    from its own geometry (stage 2 reads no H). Their end-of-step clear is skipped."""
    jacobian_clear_sizes: tuple[int, ...]
    """End of a double-buffered step: the sizes whose active Jacobian rows are cleared for the buffer's next use;
    none under the matrix-free sweeps (their producers write the packed rows' sides whole) or owners that keep a
    private Jacobian (local row packets, keyed rows), and never the sparse-diagonal response size (no dense J) or the
    compact producer's six-coordinate size (complete stores)."""

    @classmethod
    def build(cls, solver) -> StepPlan:
        """Read the plan from a constructed solver's decisions."""
        factor = _factor(solver)
        held = solver._tree_kinetic_state is not None
        solve = _solve(solver)
        response = _response(solver, solve)
        return cls(
            factor=MappingProxyType(factor),
            trisolve=MappingProxyType({} if held else _trisolve(solver)),
            response=MappingProxyType(response),
            delassus=MappingProxyType(_delassus(solver, solve)),
            solve=solve,
            unread_mass_sizes=tuple(solver.forest.size_groups) if held else (),
            jacobian_clear_sizes=()
            if solver._row_packets is not None or solver._keyed_rows is not None
            else tuple(size for size in solver.forest.size_groups if solver._memset_clears_jacobian(size)),
        )

    def sizes(self, stage: str, route: str) -> tuple[int, ...]:
        """The sizes a stage's table routes to ``route``."""
        return tuple(size for size, entry in getattr(self, stage).items() if entry.name == route)

    def record(self) -> dict:
        """The tables as a JSON-compliant record (``routes.stages`` of the path selection)."""

        def table(routes):
            return {str(int(size)): route.record() for size, route in routes.items()}

        return {
            "factor": table(self.factor),
            "trisolve": table(self.trisolve) if self.trisolve else None,
            "response": table(self.response),
            "delassus": table(self.delassus) if self.delassus else None,
            "solve_family": self.solve.family,
            "unread_mass_dofs": sorted(int(size) for size in self.unread_mass_sizes),
            "jacobian_clear_dofs": [int(size) for size in self.jacobian_clear_sizes],
        }


def _route(solver, size: int, kernel: str, name: str, reason: str) -> Route:
    """A route whose reason leads with the kernel choice (explicit, automatic, private override, CPU) and the size's
    resource rule (execution_plan) for ``kernel``."""
    resource = solver._execution_plan.selection_reasons(size).get(kernel, "")
    return Route(name, " ".join(part for part in (solver._kernel_choice_reasons[kernel], resource, reason) if part))


def _factor(solver) -> dict[int, Route]:
    plan = solver._execution_plan
    held = solver._tree_kinetic_state is not None
    routes = {}
    for size in solver.forest.size_groups:
        if size == solver._compact_diagonal_mass_size:
            route = "none", "The compact diagonal size stores its inverse mass directly; no dense factor."
        elif size == solver._sparse_mass_matrix_size:
            route = "none", solver._sparse_selection_reason
        elif held:
            route = "held", "The held tree refreshes the factor from its geometry in stage 1."
        elif plan.use_diagonal_mass(size):
            route = "diagonal", ""
        elif plan.use_tiled_cholesky(size):
            route = "tiles", ""
        else:
            route = "loop", ""
        routes[size] = _route(solver, size, "cholesky_kernel", *route)
    return routes


def _trisolve(solver) -> dict[int, Route]:
    plan = solver._execution_plan
    choice = solver.trisolve_kernel
    threshold = f"Automatic DOF threshold {solver.small_dof_threshold}."
    routes = {}
    for size in solver.forest.size_groups:
        if size == solver._sparse_mass_matrix_size:
            route = "sparse", "Native sparse factors solve in factor coordinates."
        elif size == solver._compact_diagonal_mass_size:
            route = "compact_diagonal", "The compact diagonal size applies its stored inverse mass."
        elif plan.use_diagonal_mass(size):
            route = "diagonal", "Independent coordinates apply the diagonal inverse."
        elif choice == "tiled" or (choice == "auto" and size > solver.small_dof_threshold):
            route = "tiles", threshold
        else:
            route = "loop", threshold
        routes[size] = _route(solver, size, "trisolve_kernel", *route)
    return routes


def _response(solver, solve: SolveRoute) -> dict[int, Route]:
    plan = solver._execution_plan
    sizes = solver.forest.size_groups
    owner = solver._compact_contact_boundary or solver._sparse_mass_matrix_size is not None
    routes = {}
    for size in sizes:
        route = None
        if not solve.assembles_delassus:
            if owner:
                route = Route("owner", "The contact owner and the bounded limit prefix publish the complete response.")
            elif solver._sparse_diagonal_contact_solve and size in (
                solver._sparse_diagonal_response_size,
                solver._sparse_diagonal_dense_size,
            ):
                route = Route(
                    "sparse_diagonal", "The sparse-diagonal owner whitens its rows in place: no size builds a response."
                )
            elif solver._packed_response:
                route = Route(
                    "packed", "The matrix-free sweeps read the rows packed by support; no route reads this response."
                )
        if route is None and plan.use_diagonal_mass(size):
            route = _route(solver, size, "hinv_jt_kernel", "diagonal", "")
        if route is None:
            grouped, reason = _grouped(solver, size) if solver._grouped_response else (False, "")
            if grouped:
                name = "grouped"
            elif plan.use_tiled_hinv_jt(size):
                name = "tiles"
            elif solver._local_internal_fast_path:
                name, reason = "lanes", f"{reason} A warp per articulation, skipping the worlds local owners solve."
            else:
                name = "rows"
            route = _route(solver, size, "hinv_jt_kernel", name, reason.strip())
        routes[size] = route
    return routes


def _grouped(solver, size: int) -> tuple[bool, str]:
    """Whether the grouped response solve (a warp per articulation, a lane per row) serves a size, and why.

    The data choose it: where the in-kernel tiers own rows, the response must skip each world's tier rows,
    which only the grouped kernel does. It solves the planned kernels' response, summed in its own order.
    """
    if int(size) > _GROUPED_VERIFIED_DOFS:
        return False, f"The grouped response solve is verified up to {_GROUPED_VERIFIED_DOFS} DOFs only."
    if solver._ink_skip_rows() > 0:
        return True, "Grouped: the in-kernel tiers need the grouped kernel's per-world row skip."
    return False, "No in-kernel tier rows to skip: the planned response kernel."


def _delassus(solver, solve: SolveRoute) -> dict[int, Route]:
    if not solve.assembles_delassus:
        return {}
    return dict.fromkeys(
        solver.forest.size_groups, Route("entries", "The CPU assembles each Delassus entry on its own thread.")
    )


def _solve(solver) -> SolveRoute:
    matrix_free = solver._matrix_free
    dense_split = not matrix_free
    if matrix_free:
        family = "matrix_free"
    elif solver.forest.has_mixed_contacts:
        family = "split_interleaved"
    else:
        family = "dense_split"
    # The CPU sweeps its assembled Delassus rows, a thread per world.
    if dense_split:
        route = Route("dense_loop", "The CPU sweeps the assembled Delassus rows on one thread per world.")
    elif solver._sparse_mass_matrix_size is not None:
        route = Route("sparse_pgs", "Native sparse factors own the sweep.")
    elif solver._sparse_diagonal_contact_solve:
        route = Route("sparse_diagonal", "The diagonal-mass contact owner owns the sweep.")
    else:
        route = Route(
            "matrix_free_gs", "Matrix-free Gauss-Seidel; local, paired and tiered owners claim their worlds first."
        )
    if family == "split_interleaved":
        driver = Route(
            "split_interleaved",
            "A world holds free bodies and articulations: each iteration runs a one-iteration dense sweep, "
            "rebuilds the velocity, runs a one-iteration matrix-free sweep and adds its velocity change to "
            "the dense right-hand side.",
        )
    elif dense_split and solver.pgs_debug:
        driver = Route(
            "debug_loop", "pgs_debug launches one dense iteration per dispatch and reads the impulses back after each."
        )
    elif dense_split:
        driver = Route(
            "dense_split",
            "One dense dispatch sweeps all iterations"
            + (
                "; one matrix-free dispatch then sweeps the free-body rows."
                if solver.forest.has_free_rigid_bodies
                else "."
            ),
        )
    elif solver.pgs_debug:
        driver = Route(
            "debug_loop", "pgs_debug launches the position solve one iteration per dispatch with host reads."
        )
    elif solver.contact_compliance:
        driver = Route(
            "contact_compliance",
            "Compliance updates the right-hand side and repacks metadata before every one-iteration sweep.",
        )
    elif route.name == "matrix_free_gs" and solver.pgs_schedule == "contact_then_internal":
        driver = Route(
            "contact_then_internal",
            "Contact rows sweep all iterations in one dispatch, then internal rows in a second.",
        )
    elif route.name == "matrix_free_gs" and solver.pgs_schedule == "physx_grasp":
        driver = Route(
            "physx_grasp", "Each iteration launches the drive and limit, contact and velocity-limit row phases in turn."
        )
    else:
        driver = Route("direct", "The route sweeps all iterations in one dispatch.")
    return SolveRoute(family=family, route=route, driver=driver)
