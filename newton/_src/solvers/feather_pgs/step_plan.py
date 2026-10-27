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

import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

__all__ = ["FACTOR_READS_MASS", "AdmissionFacts", "Route", "SolveRoute", "StepPlan", "first_failed"]

# Import-time switches of the grouped response solve (grouped_dynamics).
_GROUPED_HINV_OFF = os.environ.get("FEATHER_PGS_GROUPED_HINV") == "0"
_GROUPED_HINV_MAX = int(os.environ.get("FEATHER_PGS_GROUPED_HINV_MAX", "32"))
_GROUPED_VERIFIED_DOFS = 32


def first_failed(checks) -> str | None:
    """The reason of the first ``(holds, reason)`` check that fails, or ``None`` when every check holds."""
    return next((reason for holds, reason in checks if not holds), None)


@dataclass(frozen=True)
class AdmissionFacts:
    """The facts the owners' admissions test, computed once from the resolved options, the model and the device."""

    forward_cuda: bool
    """CUDA without gradient recording: native kernels without adjoints may run."""
    matrix_free: bool
    plain_sweep: bool
    """The interleaved schedule with augmented drives and no velocity iterations, warm start, debug rows or bilateral
    pre-elimination: the sweep policy the specialized owners assume."""
    regularized: bool
    velocity_limits: bool

    PLAIN_SWEEP = (
        "the interleaved schedule with augmented drives and no velocity iterations, warm start, debug rows or "
        "bilateral pre-elimination"
    )

    @classmethod
    def build(cls, solver) -> AdmissionFacts:
        """Read the facts from a solver whose options, preelimination plan and regularization are resolved."""
        return cls(
            forward_cuda=bool(solver.model.device.is_cuda and not solver.model.requires_grad),
            matrix_free=solver.pgs_mode == "matrix_free",
            plain_sweep=bool(
                solver.pgs_schedule == "interleaved"
                and solver.drive_mode == "augmented"
                and solver.pgs_velocity_iterations == 0
                and not solver.pgs_warmstart
                and not solver.pgs_debug
                and not solver._preelim_active
            ),
            regularized=bool(solver._regularization_enabled),
            velocity_limits=bool(solver.enable_joint_velocity_limits),
        )


@dataclass(frozen=True)
class Route:
    """One stage's route and why it was chosen."""

    name: str
    reason: str

    def record(self) -> dict:
        """The route as a JSON-compliant record."""
        return {"route": self.name, "reason": self.reason}


FACTOR_READS_MASS = frozenset(("diagonal", "tiles", "tiles_inverse", "loop"))
"""Factor routes that read the size's dense mass matrix H in stage 2."""


@dataclass(frozen=True)
class SolveRoute:
    """Stages 5 and 6: the branch of ``step()``, the rows the sweep solves and what drives the sweep."""

    family: str
    """``"sparse_split"``, ``"matrix_free"``, ``"split_interleaved"`` or ``"dense_split"``: the branch of ``step()``."""
    route: Route
    """The rows the sweep solves: ``dense_<pgs_kernel>``, ``sparse_pgs``, ``sparse_diagonal`` or ``matrix_free_gs``."""
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
    """Stage 2, per size: ``none`` (no dense factor), ``diagonal``, ``tiles``, ``tiles_inverse`` (the paired
    response's primary factor with its inverse), ``loop``, or ``crba_tiles`` / ``crba_warps`` (factored in stage 1)."""
    trisolve: Mapping[int, Route]
    """Stage 3, per size: ``sparse``, ``compact_diagonal``, ``diagonal``, ``tiles`` or ``loop``; empty under a
    held-tree owner, whose predictor replaces the stage."""
    response: Mapping[int, Route]
    """Stage 4, per size: ``owner`` (a contact owner publishes the response), ``sparse_diagonal``,
    ``paired_secondary`` (solved with the primary), ``paired``, ``fused`` (with the Delassus), ``diagonal``,
    ``grouped``, ``tiles``, ``packets``, ``lanes`` or ``rows``."""
    delassus: Mapping[int, Route]
    """Stage 4, per size, split mode's Delassus assembly: ``tiles`` or ``entries``; empty where none is assembled."""
    solve: SolveRoute
    unread_mass_sizes: tuple[int, ...]
    """Sizes whose dense mass matrix no stage reads: under a held-tree owner, the sizes stage 2 factors without
    H. Their end-of-step clear is skipped."""

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
            delassus=MappingProxyType(_delassus(solver, solve, response)),
            solve=solve,
            unread_mass_sizes=_unread_mass_sizes(factor, held=held),
        )

    @property
    def assembles_world_delassus(self) -> bool:
        """Whether split mode assembles the world Delassus operator from per-size blocks after the response."""
        return self.solve.assembles_delassus and not any(route.name == "fused" for route in self.response.values())

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
        }


def _factor(solver) -> dict[int, Route]:
    plan = solver._execution_plan
    routes = {}
    for size in solver.size_groups:
        if size == solver._compact_diagonal_mass_size:
            route = Route("none", "The compact diagonal size stores its inverse mass directly; no dense factor.")
        elif size == solver._sparse_mass_matrix_size:
            route = Route("none", "Native sparse factors refresh with the mass matrix; no dense factor.")
        elif plan.use_diagonal_mass(size):
            route = Route("diagonal", "Independent coordinates: the factor is the diagonal.")
        elif plan.use_tiled_cholesky(size):
            if solver._crba_cholesky_kernels_by_size[size] is not None:
                route = Route(
                    "crba_tiles", "Stage 1 assembles and factors the mass matrix in one tile per articulation."
                )
            elif size == solver._paired_response_primary_size and not solver._paired_factor_coordinates:
                route = Route("tiles_inverse", "The paired response's primary factor also materializes its inverse.")
            else:
                route = Route("tiles", "Tiled factor (the execution plan's resource choice).")
        elif solver._crba_cholesky_warp_kernels_by_size[size] is not None:
            route = Route("crba_warps", "Stage 1 assembles and factors the mass matrix in one warp per articulation.")
        else:
            route = Route("loop", "Loop factor, a thread per articulation (the execution plan's resource choice).")
        routes[size] = route
    return routes


def _unread_mass_sizes(factor: Mapping[int, Route], *, held: bool) -> tuple[int, ...]:
    """Under a held-tree owner, which refreshes the factors from its own geometry and writes no H, the sizes whose
    stage-2 route reads no H either; without an owner, stage 1 writes every H."""
    return tuple(size for size, route in factor.items() if route.name not in FACTOR_READS_MASS) if held else ()


def _trisolve(solver) -> dict[int, Route]:
    plan = solver._execution_plan
    choice = solver.trisolve_kernel
    routes = {}
    for size in solver.size_groups:
        if size == solver._sparse_mass_matrix_size:
            route = Route("sparse", "Native sparse factors solve in factor coordinates.")
        elif size == solver._compact_diagonal_mass_size:
            route = Route("compact_diagonal", "The compact diagonal size applies its stored inverse mass.")
        elif plan.use_diagonal_mass(size):
            route = Route("diagonal", "Independent coordinates apply the diagonal inverse.")
        elif choice == "tiled" or (choice == "auto" and size > solver.small_dof_threshold):
            route = Route("tiles", f"Tiled solve: trisolve_kernel={choice!r}, threshold {solver.small_dof_threshold}.")
        else:
            route = Route("loop", f"Loop solve: trisolve_kernel={choice!r}, threshold {solver.small_dof_threshold}.")
        routes[size] = route
    return routes


def _response(solver, solve: SolveRoute) -> dict[int, Route]:
    plan = solver._execution_plan
    sizes = solver.size_groups
    fused = (
        solve.assembles_delassus
        and solver._is_one_solve_art_per_world
        and bool(sizes)
        and all(plan.use_fused_hinv_jt(size) for size in sizes)
    )
    owner = solver._compact_contact_boundary or solver._sparse_mass_matrix_size is not None
    routes = {}
    for size in sizes:
        route = None
        if fused:
            route = Route("fused", "One articulation per world: the response and the Delassus in one tile.")
        elif not solve.assembles_delassus:
            if owner:
                route = Route("owner", "The contact owner and the bounded limit prefix publish the complete response.")
            elif solver._sparse_diagonal_contact_solve and size == solver._sparse_diagonal_response_size:
                route = Route("sparse_diagonal", "The diagonal-mass contact owner publishes this size's response.")
            elif size == solver._paired_response_secondary_size:
                route = Route("paired_secondary", "The paired response solves this size with its primary.")
            elif size == solver._paired_response_primary_size:
                route = Route("paired", "One articulation and one free body per world share one response.")
        if route is None and plan.use_diagonal_mass(size):
            route = Route("diagonal", "Independent coordinates apply the diagonal inverse.")
        if route is None:
            grouped, reason = _grouped(solver, size) if solver._grouped_response else (False, "")
            if grouped:
                route = Route("grouped", reason)
            elif plan.use_tiled_hinv_jt(size):
                route = Route("tiles", "Tiled response (the execution plan's resource choice).")
            elif solver._row_packets is not None:
                route = Route("packets", "Local row packets solve their worlds' response; the rest per row.")
            elif solver._local_internal_fast_path:
                route = Route("lanes", "A warp per articulation, skipping the worlds local owners solve.")
            else:
                route = Route("rows", "A thread per row (the execution plan's resource choice).")
            if reason and not grouped:
                route = Route(route.name, f"{reason} {route.reason}")
        routes[size] = route
    return routes


def _grouped(solver, size: int) -> tuple[bool, str]:
    """Whether grouped dynamics' response solve serves a size, and why."""
    if int(size) > _GROUPED_VERIFIED_DOFS:
        return False, f"The grouped response solve is verified up to {_GROUPED_VERIFIED_DOFS} DOFs only."
    if solver._ink_skip_rows() > 0:
        return True, "Grouped: the in-kernel tiers need the grouped kernel's per-world row skip."
    if _GROUPED_HINV_OFF:
        return False, "FEATHER_PGS_GROUPED_HINV=0 keeps the planned response kernel."
    if int(size) > _GROUPED_HINV_MAX:
        return False, f"Above FEATHER_PGS_GROUPED_HINV_MAX={_GROUPED_HINV_MAX}."
    if size in solver._free_body_inertia_sizes:
        return False, "Free-body sizes keep the per-row kernel, which is faster there."
    return True, f"Grouped: sizes up to FEATHER_PGS_GROUPED_HINV_MAX={_GROUPED_HINV_MAX} take the grouped kernel."


def _delassus(solver, solve: SolveRoute, response: Mapping[int, Route]) -> dict[int, Route]:
    if not solve.assembles_delassus or any(route.name == "fused" for route in response.values()):
        return {}
    if solver.delassus_kernel == "par_row_col":
        route = Route("entries", "delassus_kernel='par_row_col': a thread per entry.")
    else:
        route = Route("tiles", "A block per articulation streams row chunks of J and Y.")
    return dict.fromkeys(solver.size_groups, route)


def _solve(solver) -> SolveRoute:
    matrix_free = solver.pgs_mode == "matrix_free"
    dense_split = not matrix_free and not solver._sparse_split
    if solver._sparse_split:
        family = "sparse_split"
    elif matrix_free:
        family = "matrix_free"
    elif solver._has_mixed_contacts:
        family = "split_interleaved"
    else:
        family = "dense_split"
    # Split mode sweeps assembled Delassus rows unless native sparse factors own its rows.
    if dense_split:
        route = Route(
            f"dense_{solver.pgs_kernel}",
            f"Split mode sweeps the assembled Delassus rows with the {solver.pgs_kernel} kernel.",
        )
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
            + ("; one matrix-free dispatch then sweeps the free-body rows." if solver._has_free_rigid_bodies else "."),
        )
    elif not matrix_free:
        driver = Route("direct", "The sparse split rows sweep once per solve.")
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
