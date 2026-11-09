# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The step's spec and plan: every unknown that decides the step's path, gathered first, and what each stage launches.

:class:`StepSpec` is the frozen record of the unknowns, built once the forest and the row laws are known and before any
owner of the step: the device, the forest, the laws, the test pins, the capacities, the class plan (the case the class
rules give the class, :mod:`.class_plan`), the mass cadence and the diagnostics in force. Every owner built after it
reads it; nothing in it is computed a second time elsewhere.

:class:`StepPlan` is what the owners built from it: the class's route (the row of the route table: the spec's case, the
producer and the sizes the owners built), the sweep's kernels and streams, and the stages. The
plan's :attr:`StepPlan.stages` are the step: each its owners' calls, bound once, chosen from the spec and the route,
so ``step()`` is the stages run in order on a :class:`Step` with no construction constant re-tested. Every
stage whose work depends on the articulation size (where stage 4 whitens the rows) has one table: each size's route
and the reason for it. The solve of stages 5 and 6 has one entry: the rows its sweep solves and what drives the sweep.
The route record (:mod:`.path_selection`) is the spec's record and the plan's, and construction builds exactly the
kernels they name, so the three cannot disagree.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from functools import partial
from itertools import groupby
from operator import attrgetter, itemgetter
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
import warp as wp

from ...core.execution import Beside
from ...core.memory import L1_LINE_BYTES, Device, device
from . import cached_state_binding
from .class_plan import ClassPlan
from .forest_plan import ForestPlan
from .path_selection import json_value, kernel_key
from .route_table import Laws, Pins, RouteRow
from .walk_rows import WRITE_REST, WRITE_ROWS, WRITE_TOPOLOGY

__all__ = [
    "Capacities",
    "Route",
    "Stage",
    "Step",
    "StepPlan",
    "StepSpec",
]


@dataclass(frozen=True)
class Capacities:
    """The row capacities a world holds, fixed at construction; each class's status word records an overflow."""

    dense_rows: int
    """Dense (articulation) rows a world, its link rows included (``dense_max_constraints`` plus the links)."""
    free_body_rows: int
    """Matrix-free (free-body) rows a world (``mf_max_constraints``)."""
    row_pitch: int
    """A world's rows in the row store: the dense rows, then the free-body rows where free bodies exist."""
    free_contact_rows: bool
    """The free bodies' contacts and speed rows keep a run of their own (the free-body rows): only where a world can
    take the colored mapping, whose units are that run's (:func:`.sweep_units.capable_worlds`)."""
    speed_rows: int
    """The free bodies' speed rows a world holds on the line (two a body under speed limits), where they are rows of
    the line: the line's last rows, from its phase bound 2 (:meth:`.rows.RowLaws.bound_phase`); 0 elsewhere."""

    @classmethod
    def build(cls, options, forest: ForestPlan, link_rows: int, speed_limits: bool) -> Capacities:
        """The capacities from the options and the forest: a link's rows are the closure's, on top of the rows the
        option gives the world's contacts (``link_rows`` a world).

        A free body is a one-node tree: its contacts and speed rows are rows of the line like any other, written by
        the one writer and swept by the class's sweep. They keep a run of their own only where a world can take the
        colored mapping; elsewhere their capacity (``mf_max_constraints``) and the speed rows' join the line's dense
        rows, and the free-body rows hold none."""
        from . import sweep_units  # noqa: PLC0415
        from .row_kernels import RIGID_SPEED_ROWS  # noqa: PLC0415

        dense = int(options.dense_max_constraints) + int(link_rows)
        free = int(options.mf_max_constraints)
        free_bodies = forest.has_free_rigid_bodies
        # The production rule decides the run (the rows' layout); a test that relaxes the rule changes only which of
        # the run's worlds the colored mapping takes.
        run = bool(free_bodies and sweep_units.capable_worlds(forest.free_rigid_per_world, free).size > 0)
        speed_rows = 0
        if free_bodies and not run:
            speed_rows = RIGID_SPEED_ROWS * int(np.max(forest.free_rigid_per_world, initial=0)) if speed_limits else 0
            dense += free + speed_rows
            # The line ends on a whole L1 line, so every world's line starts on one: the free bodies' (empty) rows
            # fill it, one at least.
            line = L1_LINE_BYTES // 4
            free = -(-(dense + 1) // line) * line - dense
        return cls(
            dense_rows=dense,
            free_body_rows=free,
            row_pitch=dense + (free if free_bodies else 0),
            free_contact_rows=run,
            speed_rows=speed_rows,
        )


DIAGNOSTICS = ("nvtx", "snapshots", "audit", "row_watermark", "readout", "sync_timings")
"""The diagnostics a spec may hold, in the record's order."""


@dataclass(frozen=True)
class StepSpec:
    """Every unknown that decides the step's path, gathered once before any owner of the step (frozen).

    Every owner built after it reads it, the step plan builds its tables and stages from it and the class's route,
    and the route record reports it (``routes.spec``, with the laws under ``routes.laws``). A change of any field is a
    new solver.
    """

    device: Device
    """The device the step runs on (:func:`.core.execution.device`)."""
    forest: ForestPlan
    """The forest after the body cut: topology, groups, world mapping (frozen)."""
    laws: Laws
    """The row laws (:class:`.route_table.Laws`), read from the options, the model and the closure's blocks."""
    pins: Pins
    """The test-only route pins in force (:class:`Pins`; none in production)."""
    capacities: Capacities
    classes: ClassPlan
    """The class plan (:class:`.class_plan.ClassPlan`): the case the class rules give the solver's class, with each
    case's reason and the chosen case's plan, which its owners build."""
    mass_interval: int
    """Steps between mass refreshes (``update_mass_matrix_interval``)."""
    articulated: bool
    """The forest holds articulations: without them the step only integrates particles."""
    particles: bool
    """The model holds particles, whose forces and integration the step runs."""
    pure_forest: bool
    """The model is its own forest (:mod:`.body_cut`): the step reads and publishes the caller's states directly."""
    diagnostics: tuple[str, ...]
    """The diagnostics in force, each an optional stage or readout (:data:`DIAGNOSTICS`): ``nvtx``, ``snapshots``,
    ``audit``, ``row_watermark``, ``readout`` (the sweeps' counters), ``sync_timings``."""

    @classmethod
    def build(
        cls,
        model,
        options,
        cut,
        forest: ForestPlan,
        laws: Laws,
        pins: Pins,
        capacities: Capacities,
        classes: ClassPlan,
    ) -> StepSpec:
        """Gather the spec from the ``model``, the checked ``options``, the body ``cut``, the ``forest``, the row
        ``laws``, the test ``pins``, the ``capacities`` and the class plan (``classes``)."""
        diagnostics = options.diagnostics
        requested = {
            "nvtx": diagnostics.nvtx,
            "snapshots": diagnostics.snapshots,
            "audit": diagnostics.audit,
            "row_watermark": diagnostics.row_watermark,
            "readout": diagnostics.solve_readout != "off",
            "sync_timings": diagnostics.sync_timings is not None,
        }
        return cls(
            device=device(model.device),
            forest=forest,
            laws=laws,
            pins=pins,
            capacities=capacities,
            classes=classes,
            mass_interval=int(options.update_mass_matrix_interval),
            articulated=bool(model.articulation_count),
            particles=bool(model.particle_count),
            pure_forest=bool(cut.pure),
            diagnostics=tuple(name for name in DIAGNOSTICS if requested[name]),
        )

    def record(self) -> dict:
        """The spec's own fields as a JSON-compliant record (``routes.spec``; the laws are ``routes.laws``): the
        device's architecture and multiprocessors, and the pins in force."""
        own = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ("forest", "laws", "classes")}
        return json_value(dict(own, device={"arch": self.device.arch, "sms": self.device.sms}, pins=self.pins.record()))


@dataclass(frozen=True)
class Route:
    """One stage's route and why it was chosen."""

    name: str
    reason: str

    def record(self) -> dict:
        """The route as a JSON-compliant record."""
        return {"route": self.name, "reason": self.reason}


@dataclass(frozen=True)
class StepPlan:
    """The step of one solver: what its owners decided from the spec, its stages and per-stage dispatch tables, built
    once by :meth:`build`."""

    route: RouteRow
    """The route: the class's row of the route table (the placement case, its producer and its sizes)."""
    kernels: Mapping[str, str | None]
    """The sweep's kernel keys by route: ``sparse``, ``sparse_diagonal``, ``mf_gs`` and
    ``mf_colored`` (``None`` where the class builds none)."""
    streams: tuple[str, ...]
    """The side streams the step forks onto, by owner."""
    response: Mapping[int, Route]
    """Stage 4, per size, where its rows are whitened: ``owner`` (a contact owner publishes the response),
    ``sparse_diagonal`` or ``packed`` (the packed rows' build)."""
    solve: Route
    """Stages 5 and 6: the rows the sweep solves, ``sparse_pgs``, ``sparse_diagonal`` or ``matrix_free_gs``, all
    iterations in one dispatch."""
    stages: tuple[Stage, ...]
    """The step, in order (:func:`_stages`)."""

    @classmethod
    def build(cls, spec: StepSpec, route: RouteRow, solver) -> StepPlan:
        """Read the plan from the step's ``spec`` and the class's ``route``, and bind its stages to the ``solver``'s
        owners."""
        sweep = solver.sweep
        streams = (
            ("dynamics", getattr(solver.factor.dynamics, "stream", None)),
            *((f"claim{i}", stream) for i, stream in enumerate(sweep.claim_streams)),
        )
        return cls(
            route=route,
            kernels=MappingProxyType(
                {
                    "sparse": kernel_key(sweep.sparse_kernel) if route.support == "factor" else None,
                    "sparse_diagonal": kernel_key(sweep.sparse_diagonal_kernel),
                    "mf_gs": kernel_key(sweep.mf_gs_kernel),
                    "mf_colored": kernel_key(sweep.mf_colored_kernel),
                }
            ),
            streams=tuple(name for name, stream in streams if stream is not None),
            response=MappingProxyType(_response(route, spec.forest)),
            solve=_solve(route),
            stages=_stages(spec, route, solver),
        )

    def run(self, step: Step) -> None:
        """Run the step's stages in order."""
        for stage in self.stages:
            stage.run(step)

    def record(self) -> dict:
        """The tables, the sweep's kernels and streams as a JSON-compliant record (``routes.stages`` of the path
        selection)."""

        def table(routes):
            return {str(int(size)): route.record() for size, route in routes.items()}

        return {
            "response": table(self.response),
            "order": [stage.name for stage in self.stages],
            "kernels": dict(self.kernels),
            "streams": list(self.streams),
        }


def _response(row: RouteRow, forest) -> dict[int, Route]:
    """Where each size's rows are whitened: by native sparse factors' support writer (case K1), in place by the sparse
    diagonal (case K2's pair) or in the packed rows' build (every other class)."""
    routes = {}
    for size in forest.size_groups:
        if row.support == "factor":
            routes[size] = Route("owner", "Native sparse factors' writer whitens every row into its support.")
        elif row.support == "diagonal" and size in (row.sizes["diagonal"], row.sizes["dense"]):
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


def _solve(row: RouteRow) -> Route:
    if row.support == "factor":
        return Route("sparse_pgs", "Native sparse factors own the sweep.")
    if row.support == "diagonal":
        return Route("sparse_diagonal", "The diagonal-mass contact owner owns the sweep.")
    return Route("matrix_free_gs", "Matrix-free Gauss-Seidel; the tiers claim their worlds first.")


class Step:
    """One step's operands: the caller's arguments, the forest's states and what one stage hands the next, with the
    stages that read or set them."""

    __slots__ = (
        "contacts",
        "control",
        "dt",
        "event",
        "index",
        "qd",
        "source_in",
        "source_out",
        "state_in",
        "state_out",
    )

    def __init__(self, state_in, state_out, control, contacts, dt: float, event, index: int):
        # The caller's states, which the leave publishes; the forest's states and control (body_cut), the caller's
        # on a pure forest, from the enter on.
        self.source_in, self.source_out = state_in, state_out
        self.state_in, self.state_out, self.control = state_in, state_out, control
        self.contacts, self.dt, self.index = contacts, dt, index
        # The collision pass's event, waited on once, before the first stage that reads what it produced.
        self.event = event
        # The predictor's velocity (prescaled where velocities are limited), from stage 1.
        self.qd = None

    def enter(self, cut, cache_valid) -> None:
        self.state_in, self.state_out, self.control = cut.enter(
            self.state_in, self.state_out, self.control, cache_valid
        )

    def default_control(self, model) -> None:
        if self.control is None:
            self.control = model.control(clone_variables=False)

    def follow_contacts(self, placement) -> None:
        if self.contacts is not None:
            placement.follow_contacts(self.contacts)

    def warm_start_contract(self) -> None:
        # Warm start accepts body-pair-reduced streams too: matching runs on the written (retained) contacts.
        if self.contacts is not None and getattr(self.contacts, "rigid_contact_match_index", None) is None:
            raise NotImplementedError(
                "FeatherPGS contact warm start requires a Contacts buffer created with "
                "contact_matching enabled (rigid_contact_match_index is None). Build the "
                'CollisionPipeline / Contacts with contact_matching="latest" (or "sticky").'
            )

    def wait_for_collision(self, device) -> None:
        if self.event is not None:
            wp.get_stream(device).wait_event(self.event)
            self.event = None

    def prescale(self, factor, prescaled) -> None:
        factor.prescale_velocity(self.state_in.joint_qd, prescaled)
        self.qd = prescaled

    def velocity(self) -> None:
        self.qd = self.state_in.joint_qd

    def count(self, solver) -> None:
        solver._step += 1
        solver.factor.last_step_dt = self.dt

    def beside(self, beside: Beside, owner, name: str, *arguments) -> None:
        beside.start(partial(getattr(owner, name), *arguments))

    def ranged(self, name: str, stages: tuple[Stage, ...]) -> None:
        """``stages`` inside the NVTX range ``name`` (``Diagnostics.nvtx``)."""
        with wp.ScopedTimer(name, print=False, use_nvtx=True, synchronize=False):
            for stage in stages:
                stage.run(self)

    def timed(self, stage: Stage, timed: Callable[[int], bool]) -> None:
        """``stage``, on a step ``timed`` names (``Diagnostics.sync_timings``) synchronized before and after, its time
        printed."""
        if not timed(self.index):
            stage.run(self)
            return
        wp.synchronize()
        start = time.perf_counter()
        stage.run(self)
        wp.synchronize()
        print(f"[fpgs-sync] step={self.index} {stage.name}={(time.perf_counter() - start) * 1000.0:.3f} ms", flush=True)  # noqa: T201


class Read(NamedTuple):
    """A stage call's argument read from the :class:`Step` when the stage runs: its field ``name``, or the step."""

    name: str | None = None


STEP, STATE_IN, STATE_OUT, CONTROL, CONTACTS, DT, QD, SOURCE_IN, SOURCE_OUT = (
    Read(name) for name in (None, "state_in", "state_out", "control", "contacts", "dt", "qd", "source_in", "source_out")
)


def _itself(step: Step) -> Step:
    return step


class Stage(NamedTuple):
    """One stage of the step: its name and its owners' calls, each ``(owner, arguments, method, reads)``: the method
    read by name when the stage runs, on ``arguments`` with the step's fields in place (``reads``)."""

    name: str
    calls: tuple[tuple, ...]

    @classmethod
    def bind(cls, name: str, calls) -> Stage:
        """The stage ``name`` of ``calls``, each ``(owner, method, *arguments)``: a :class:`Read` argument (or owner,
        the step's own method) is read from the step when the stage runs."""
        bound = []
        for owner, method, *arguments in calls:
            held = (Step, (STEP, *arguments)) if owner is STEP else (owner, tuple(arguments))
            reads = tuple(
                (index, _itself if read.name is None else attrgetter(read.name))
                for index, read in enumerate(held[1])
                if type(read) is Read
            )
            bound.append((*held, method, reads))
        return cls(name, tuple(bound))

    def run(self, step: Step) -> None:
        for owner, arguments, method, reads in self.calls:
            values = list(arguments)
            for index, read in reads:
                values[index] = read(step)
            getattr(owner, method)(*values)


def _stages(spec: StepSpec, route: RouteRow, solver) -> tuple[Stage, ...]:
    """The step's stages: a table of its owners' calls, each with the predicate, decided here once from the spec and
    the class's route, that runs it. A stage then re-tests no construction constant; what it still tests is the call's
    (the contacts and the collision event) or its owner's (``owners.tsv`` of the TG hand-back). Under the NVTX
    diagnostic a named group runs as one stage inside its range; under the timings each stage runs timed."""
    cut, factor, laws, placement, sweep = attrgetter("cut", "factor", "laws", "placement", "sweep")(solver)
    model, options, forest, diagnostics = solver.model, solver.options, spec.forest, spec.diagnostics
    articulated, particles, pure, warm = spec.articulated, spec.particles, spec.pure_forest, spec.laws.warm_start
    v_hat, v_out, colored = solver.v_hat, solver.v_out, sweep.colored_units
    device, limited, preelimination = model.device, options.enable_joint_velocity_limits, spec.laws.preelimination

    # The drives' stage takes the held plan's coordinates (a one-coordinate tree's predict decides its own drive) and
    # overlaps FK/ID beside the step's stream (Beside); the mask (refresh) joins them.
    overlap = factor.dynamics
    drives = bool(factor.parallel_drives and factor.held.prepares_drives)
    forked = drives and overlap is not None
    drive = (factor, "prepare_drives", STATE_IN, CONTROL, DT, v_hat, v_out, factor.held.drive_dofs)
    free_roots = bool(forest.free_root_joint_count and forest.free_rigid_body_count)
    snapshots = ()
    if "snapshots" in diagnostics and articulated:
        qd, qdd, predicted = solver._debug_stage3_qd_work, solver._debug_stage3_joint_qdd, solver._debug_stage3_v_hat
        snapshots = ((wp, "copy", qd, QD), (wp, "copy", qdd, factor.joint_qdd), (wp, "copy", predicted, v_hat))

    # The row pass, its families in the launch order that is the slot order (rows.py): each family a class holds. The
    # joint families' rows are seeds the one writer walks: a class that walks no rows has no responding coordinate, and
    # holds none (no contact is admitted there either).
    holds, run = placement.walks, placement.free_contact_rows
    limit_sizes = laws.limit_sizes(solver) if options.enable_joint_limits and holds else ()
    velocity_limits = laws.velocity_limit_warp_kernels and holds
    speed_rows = placement.speed_rows or spec.classes.tiers is not None
    free_limits = forest.has_free_rigid_bodies and placement.rigid_velocity_limit_slot is not None
    # The rows a world's topology fixes (links, mimics, limits) are known once their families are written, before the
    # contacts: where the one writer walks the general placement's rows and its sink does not wait for the counts (no
    # tier walks its rows), it walks them beside the contact pass (Beside) and the rest after it, the two parts joined
    # before the rest. Their diagonals start from zero before them.
    split = holds and route.support == "dense" and placement.row_seeds.tiers == 0
    topology = Beside(device) if split else None
    clear = ((placement, "diagonals"),) if placement.clears_diagonals else ()
    walk = (laws, "write_walked", solver, STATE_IN, CONTACTS, DT)
    join = ((topology, "join"),) if topology is not None else ()
    # The closure's seed and the colored worlds' units read the row store, which a model without worlds does not hold:
    # each is bound only where its stage runs. The colored worlds' units and colors follow the rows and records their
    # writers published, over the claims' candidates.
    seeds = (factor, "seed_blocks", placement.dense_first, placement.rhs, placement.impulses) if preelimination else ()
    built = (
        "mf_constraint_count dense_first mf_contact_rows_end row_record free_records packed_Z row_mu row_w body_base"
    )
    build = () if colored is None else (colored, "build", *attrgetter(*built.split())(placement))

    # Each row a stage: (group, name, predicate, *calls), a group its NVTX name (or none) and its predicate.
    call, dynamics = (None, True), (None, articulated)
    groups = (
        "S1_FK_ID_CRBA S3_Trisolve_Vhat S4_ContactBuild S4_HinvJt_Diag_RHS S5_PGS_Prep S5_GatherJY S6_PGS_Solve "
        "S7_Integrate S7_Friction_Patch_Carry S7_Warmstart_Carry"
    )
    s1, s3, s4, s4r, s5, s5j, s6, s7, s7p, s7w = ((group, articulated) for group in groups.split())
    rows = (
        # The call's bind: the contacts' material, pool and identity contract.
        (call, "enter", not pure, (STEP, "enter", cut, factor.fk_id_cache_valid)),
        (call, "default_control", True, (STEP, "default_control", model)),
        (call, "bind_material", options.contact_compliance, (solver, "_bind_contact_material", CONTACTS)),
        (call, "follow_contacts", articulated, (STEP, "follow_contacts", placement)),
        (call, "warm_start_contract", warm, (STEP, "warm_start_contract")),
        (call, "wait_for_collision", particles, (STEP, "wait_for_collision", device)),
        (call, "particle_forces", particles, (solver, "_eval_particle_forces", STATE_IN, CONTROL, CONTACTS)),
        # Stages 1-3: the tree factor (the drives, FK/ID, the mass refresh and the factor's, the walk's refresh with the
        # clock), then the predictor v_hat and the free roots' corrections.
        (s1, "check_call", True, (cached_state_binding, "check_call", device, STATE_IN, STATE_OUT, DT)),
        (s1, "drives_on_stream", forked, (STEP, "beside", overlap, *drive)),
        (s1, "drives", drives and overlap is None, drive),
        (s1, "prescale", limited, (STEP, "prescale", factor, solver.qd_work)),
        (s1, "velocity", not limited, (STEP, "velocity")),
        (s1, "fk_id", True, (factor, "fk_id", STATE_IN, STATE_OUT, QD, DT)),
        (s1, "join_drives", forked, (overlap, "join")),
        (s1, "refresh", True, (factor, "refresh", DT)),
        (s1, "advance_clock", factor.walk_arts is None, (factor, "advance_clock", DT)),
        (s1, "walk_step", factor.walk_arts is not None, (factor, "walk_step", STATE_IN, DT)),
        (s3, "predict", True, (factor, "predict", STATE_IN, CONTROL, QD, DT)),
        (s3, "correct_free_roots", free_roots, (factor, "correct_free_roots", STATE_IN, QD, DT, v_hat)),
        (s3, "damp_spin", options.angular_damping != 0.0, (factor, "damp_spin", DT, v_hat)),
        (dynamics, "snapshots", "snapshots" in diagnostics, *snapshots),
        (dynamics, "wait_for_collision", not particles, (STEP, "wait_for_collision", device)),
        # Stage 4: the rows. The row pass (each family's slots and seeds, and each world's acts word), the status and the
        # overflow warning; then the diagonals, the one writer (every walking class's: it walks the rows into their
        # placement's sink after the row pass, each row's rhs by the row law; the free bodies' writer runs in the row
        # pass, after their families' slots, beside the topology part), the patches' link once every contact row's
        # records are written, and the sweep's claims, which read the rows.
        (s4, "begin_rows", True, (laws, "begin", solver, STATE_IN, factor, DT)),
        (s4, "write_mimics", laws.mimic_warp_kernels and holds, (laws, "write_mimics", solver, STATE_IN, DT)),
        (s4, "write_links", laws.link_count > 0 and holds, (laws, "write_links", solver, STATE_IN)),
        (s4, "prepare_fused_limits", spec.classes.fused_sizes, (laws, "prepare_fused_limits", solver, STATE_IN, DT)),
        (s4, "write_limits", limit_sizes, (laws, "write_limits", solver, STATE_IN, DT, limit_sizes)),
        (s4, "bound_limits", True, (laws, "bound_phase", solver, 0)),
        (s4, "write_velocity_limits", velocity_limits, (laws, "write_velocity_limits", solver, STATE_IN, DT)),
        (s4, "bound_velocity_limits", True, (laws, "bound_phase", solver, 1)),
        (s4, "write_topology_rows", topology is not None, *clear, (STEP, "beside", topology, *walk, WRITE_TOPOLOGY)),
        (s4, "write_contacts", True, (laws, "write_contacts", solver, STATE_IN, factor, CONTACTS, DT, route.producer)),
        # The free bodies' run (their contacts and speed rows) where a colored world holds one, else the line's speed
        # rows, its last: the tiers and the sweeps read where they start.
        (s4, "bound_free_body_contacts", run, (laws, "bound_free_body_contacts", solver)),
        (s4, "bound_speed_rows", speed_rows, (laws, "bound_speed_rows", solver)),
        (s4, "allocate_free_body_limits", free_limits, (laws, "allocate_free_body_limits", solver)),
        (s4, "write_free_bodies", run, (laws, "write_free_bodies", solver, STATE_IN, CONTACTS, DT)),
        (s4, "finalize_dense", True, (laws, "finalize_dense", solver)),
        # The claim stage: each world into its claim's bucket once the counts are final; the unit builder and the
        # sweeps walk their buckets.
        (s4, "claim_worlds", sweep.claims is not None, (sweep, "claim_worlds")),
        (s4, "latch_status", True, (placement, "latch_status", CONTACTS)),
        (s4, "warn_overflow", options.diagnostics.overflow_warning, (placement, "warn_overflow")),
        # The one writer writes its contact rows' records after the row pass, so the patches link after it, and the
        # streamed rows' records follow the linked rows; the sparse diagonal's schedule reads the rows it wrote and the
        # marks of their slots.
        (s4r, "diagonals", placement.clears_diagonals and topology is None, (placement, "diagonals")),
        (s4r, "write_walked", holds, *join, (*walk, WRITE_ROWS if topology is None else WRITE_REST)),
        (s4r, "link_patches", True, (laws, "link_patches", solver, CONTACTS)),
        (s4r, "store_records", placement.writes_records, (sweep, "store_records")),
        (s4r, "schedule_diagonal", sweep.sparse_contact_schedule is not None, (sweep, "schedule_diagonal")),
        # Stages 5 and 6: the impulses' start (the warm start's normals, the closures' seeds: the one writer closed the
        # bilateral blocks and corrected the other rows, the blocks' kept rows start at -r_Q, after the warm start's
        # gather), the claims that follow the published rows, and the sweep, which adds its velocity change to the
        # predictor's (a world without rows keeps it).
        (s5, "prepare_impulses", True, (placement, "prepare_impulses")),
        (s5, "warm_start", warm, (sweep, "warm_start", CONTACTS)),
        (s5j, "seed_blocks", preelimination, seeds),
        (s5j, "start_velocity", True, (wp, "copy", v_out, v_hat)),
        (s5j, "build_colored_units", colored is not None, (sweep, "join_claims"), build),
        (s6, "solve", True, (sweep, "solve", v_hat, v_out)),
        (dynamics, "audit_step", "audit" in diagnostics, (solver._audit, "launch")),
        # Stage 7: the finish (qdd, integration, publication), the carry the next step reads, the leave.
        (s7, "finish", True, (factor, "finish", STATE_IN, STATE_OUT, DT)),
        (call, "integrate_particles", particles, (solver, "integrate_particles", model, STATE_IN, STATE_OUT, DT)),
        # The carry: the patches' history, then under the warm start the contacts' normals and normal impulses.
        (s7p, "carry_patches", True, (sweep, "finish_patches", STATE_IN, STATE_OUT, CONTACTS, DT)),
        (s7w, "carry", warm, (sweep, "carry", CONTACTS)),
        (dynamics, "watermarks", "row_watermark" in diagnostics, (placement, "record_watermarks", CONTACTS)),
        # The forest's boundary: a pure forest is the caller's, so it enters and leaves nothing.
        (call, "count", True, (STEP, "count", solver)),
        (call, "leave", not pure, (cut, "leave", SOURCE_IN, SOURCE_OUT)),
        (call, "follow_shadows", not pure and cut.shadow.size, (cut, "follow_shadows", SOURCE_IN, SOURCE_OUT)),
    )
    timed = options.diagnostics.timed if "sync_timings" in diagnostics else None
    stages = []
    for (group, holds_group), named in groupby(rows, key=itemgetter(0)):
        runs = tuple(Stage.bind(name, calls) for _, name, holds_row, *calls in named if holds_group and holds_row)
        if group is not None and runs and "nvtx" in diagnostics:
            runs = (Stage.bind(group, ((STEP, "ranged", group, runs),)),)
        stages += (
            runs if timed is None else [Stage.bind(stage.name, ((STEP, "timed", stage, timed),)) for stage in runs]
        )
    return tuple(stages)
