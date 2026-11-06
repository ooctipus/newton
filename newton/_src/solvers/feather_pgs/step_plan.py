# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The step's spec and plan: every unknown that decides the step's path, gathered first, and what each stage launches.

:class:`StepSpec` is the frozen record of the unknowns, built once the forest and the row laws are known and before any
owner of the step: the device, the forest, the laws, the test pins, the capacities, the class plan (the case the class
rules give the class, :mod:`.class_plan`), the mass cadence and the diagnostics in force. Every owner built after it
reads it; nothing in it is computed a second time elsewhere.

:class:`StepPlan` is what the owners built from it: the class's route (the row of the route table: the spec's case, the
producer and the sizes the owners built), the sweep's kernels and streams, and the stages. The
plan's :attr:`StepPlan.stages` are the step: each a closure bound once to its owners, chosen from the spec and the
route, so ``step()`` is the stages run in order on a :class:`Step` with no construction constant re-tested. Every
stage whose work depends on the articulation size (where stage 4 whitens the rows) has one table: each size's route
and the reason for it. The solve of stages 5 and 6 has one entry: the rows its sweep solves and what drives the sweep.
The route record (:mod:`.path_selection`) is the spec's record and the plan's, and construction builds exactly the
kernels they name, so the three cannot disagree.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
import warp as wp

from ...core.execution import Beside
from ...core.memory import Device, device
from ...sim import ModelFlags
from . import cached_state_binding
from .class_plan import ClassPlan
from .forest_plan import ForestPlan
from .route_table import Laws, Pins, RouteRow
from .walk_rows import WRITE_REST, WRITE_TOPOLOGY

__all__ = [
    "Capacities",
    "Route",
    "SolveRoute",
    "Stage",
    "Step",
    "StepPlan",
    "StepSpec",
    "TopologyPin",
    "kernel_key",
]


def kernel_key(kernel) -> str | None:
    """A kernel's key (a factory's product without one: its type's name), or ``None`` for no kernel."""
    return None if kernel is None else getattr(kernel, "key", type(kernel).__name__)


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
    free_contact_rows: bool
    """The free bodies' contacts keep a run of their own (the free-body rows): only where a world can take the colored
    mapping, whose units are that run's (:func:`.sweep_units.capable_worlds`)."""

    @classmethod
    def build(cls, options, forest: ForestPlan, link_rows: int, speed_limits: bool) -> Capacities:
        """The capacities from the options and the forest: a link's rows are the closure's, on top of the rows the
        option gives the world's contacts (``link_rows`` a world).

        A free body is a one-node tree: its contacts are rows of the line like any other, written by the one writer
        and swept by the class's sweep. They keep a run of their own only where a world can take the colored mapping;
        elsewhere their capacity (``mf_max_constraints``) joins the line's dense rows and the free-body rows hold the
        bodies' speed rows alone (two a body under ``speed_limits``)."""
        from .row_kernels import RIGID_SPEED_ROWS  # noqa: PLC0415
        from .row_metadata import _validate_mf_metadata_encoding  # noqa: PLC0415
        from .sweep_units import capable_worlds  # noqa: PLC0415

        dense = int(options.dense_max_constraints) + int(link_rows)
        free = int(options.mf_max_constraints)
        _validate_mf_metadata_encoding(free)
        free_bodies = forest.has_free_rigid_bodies
        run = bool(free_bodies and capable_worlds(forest.free_rigid_per_world, free).size > 0)
        if free_bodies and not run:
            dense += free
            speed_rows = RIGID_SPEED_ROWS * int(np.max(forest.free_rigid_per_world, initial=0))
            free = max(speed_rows if speed_limits else 0, 1)
        return cls(
            dense_rows=dense,
            free_body_rows=free,
            row_pitch=dense + (free if free_bodies else 0),
            free_contact_rows=run,
        )

    def record(self) -> dict:
        """The capacities as a JSON-compliant record."""
        return asdict(self)


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
        """The spec's own fields as a JSON-compliant record (``routes.spec``; the laws are ``routes.laws``)."""
        return {
            "device": {"arch": self.device.arch, "sms": self.device.sms},
            "pins": self.pins.record(),
            "capacities": self.capacities.record(),
            "mass_interval": self.mass_interval,
            "articulated": self.articulated,
            "particles": self.particles,
            "pure_forest": self.pure_forest,
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
    """The step of one solver: what its owners decided from the spec, its stages and per-stage dispatch tables, built
    once by :meth:`build`."""

    route: RouteRow
    """The route: the class's row of the route table (the placement case, its producer and its sizes)."""
    kernels: Mapping[str, str | None]
    """The sweep's kernel keys by route: ``sparse``, ``sparse_diagonal``, ``paired_factor``, ``mf_gs`` and
    ``mf_colored`` (``None`` where the class builds none)."""
    streams: tuple[str, ...]
    """The side streams the step forks onto, by owner."""
    response: Mapping[int, Route]
    """Stage 4, per size, where its rows are whitened: ``owner`` (a contact owner publishes the response),
    ``sparse_diagonal`` or ``packed`` (the packed rows' build)."""
    solve: SolveRoute
    stages: tuple[Stage, ...]
    """The step, in order (:func:`_stages`)."""

    @classmethod
    def build(cls, spec: StepSpec, route: RouteRow, solver) -> StepPlan:
        """Read the plan from the step's ``spec`` and the class's ``route``, and bind its stages to the ``solver``'s
        owners."""
        sweep = solver.sweep
        streams = (
            ("dynamics", solver.factor.dynamics_stream),
            *((f"claim{i}", stream) for i, stream in enumerate(sweep.claim_streams)),
        )
        return cls(
            route=route,
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
            response=MappingProxyType(_response(route, spec.forest)),
            solve=_solve(route),
            stages=_stages(spec, route, solver),
        )

    def run(self, step: Step) -> None:
        """Run the step's stages in order."""
        for stage in self.stages:
            stage.run(step)

    def sizes(self, stage: str, route: str) -> tuple[int, ...]:
        """The sizes a stage's table routes to ``route``."""
        return tuple(size for size, entry in getattr(self, stage).items() if entry.name == route)

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
        if row.case == "K1":
            routes[size] = Route("owner", "Native sparse factors' writer whitens every row into its support.")
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
        route = Route("matrix_free_gs", "Matrix-free Gauss-Seidel; paired and tiered owners claim their worlds first.")
    return SolveRoute(route=route)


class Step:
    """One step's operands: the caller's arguments, the forest's states and what one stage hands the next."""

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


class Stage(NamedTuple):
    """One stage of the step: its name and the closure, bound at plan time, that runs it on a :class:`Step`."""

    name: str
    run: Callable[[Step], None]


def _timed(stage: Stage, timed: Callable[[int], bool]) -> Stage:
    """``stage`` as a timed stage (``Diagnostics.sync_timings``): on a step ``timed`` names it synchronizes before and
    after the stage and prints its time; on every other step it runs the stage alone."""

    def run(step):
        if not timed(step.index):
            stage.run(step)
            return
        wp.synchronize()
        start = time.perf_counter()
        stage.run(step)
        wp.synchronize()
        print(f"[fpgs-sync] step={step.index} {stage.name}={(time.perf_counter() - start) * 1000.0:.3f} ms", flush=True)

    return Stage(stage.name, run)


def _stages(spec: StepSpec, route: RouteRow, solver) -> tuple[Stage, ...]:
    """The step's stages, each bound once to its owners.

    Every construction constant the step reads is decided here, from the spec and the class's route: which stages
    exist, the writer, the optional claims and the diagnostics. A stage then runs without re-testing one; what it still tests is the call's
    (the contacts and the collision event) or its owner's (``owners.tsv`` of the TG hand-back).
    """
    cut, factor, laws, placement, sweep, model = (
        solver.cut,
        solver.factor,
        solver.laws,
        solver.placement,
        solver.sweep,
        solver.model,
    )
    forest, diagnostics = spec.forest, spec.diagnostics
    stages: list[Stage] = []
    timed = solver.options.diagnostics.timed if "sync_timings" in diagnostics else None

    def add(*runs, nvtx: str | None = None) -> None:
        """Append ``runs`` as stages; under the NVTX diagnostic a named group runs as one stage inside its range, and
        under the timings each stage runs as a timed stage (:func:`_timed`)."""
        if nvtx is not None and "nvtx" in diagnostics:

            def ranged(step, runs=runs, name=nvtx):
                with wp.ScopedTimer(name, print=False, use_nvtx=True, synchronize=False):
                    for run in runs:
                        run(step)

            added = (Stage(nvtx, ranged),)
        else:
            added = tuple(Stage(run.__name__, run) for run in runs)
        stages.extend(added if timed is None else (_timed(stage, timed) for stage in added))

    def enter(step):
        step.state_in, step.state_out, step.control = cut.enter(
            step.state_in, step.state_out, step.control, factor.fk_id_cache_valid
        )

    def default_control(step):
        if step.control is None:
            step.control = model.control(clone_variables=False)

    def bind_material(step):
        solver._bind_contact_material(step.contacts)

    def follow_contacts(step):
        if step.contacts is not None:
            placement.follow_contacts(step.contacts)

    def warm_start_contract(step):
        contacts = step.contacts
        # A reduced stream is valid when it carries retained identities. Preserve the reader lease for an unreduced
        # captured stream: a later producer must not change its layout behind graph replay.
        solver._require_unreduced_contacts(
            contacts,
            supports_body_pair_reduced_contacts=bool(
                contacts is not None and contacts.rigid_contacts_body_pair_reduced
            ),
            configuration="pgs_warmstart=True",
        )
        if contacts is not None and getattr(contacts, "rigid_contact_match_index", None) is None:
            raise NotImplementedError(
                "FeatherPGS contact warm start requires a Contacts buffer created with "
                "contact_matching enabled (rigid_contact_match_index is None). Build the "
                'CollisionPipeline / Contacts with contact_matching="latest" (or "sticky").'
            )

    def wait_for_collision(step):
        if step.event is not None:
            wp.get_stream(model.device).wait_event(step.event)
            step.event = None

    def particle_forces(step):
        solver._eval_particle_forces(step.state_in, step.control, step.contacts)

    def integrate_particles(step):
        solver.integrate_particles(model, step.state_in, step.state_out, step.dt)

    def count(step):
        solver._step += 1
        factor.last_step_dt = step.dt

    def leave(step):
        cut.leave(step.source_in, step.source_out)

    def follow_shadows(step):
        cut.follow_shadows(step.source_in, step.source_out)

    # The forest's boundary: a pure forest is the caller's, so it enters and leaves nothing.
    boundary = () if spec.pure_forest else (leave, follow_shadows) if cut.shadow.size else (leave,)

    # The call's bind: the contacts' material, pool and identity contract.
    add(*(() if spec.pure_forest else (enter,)), default_control)
    if solver.options.contact_compliance:
        add(bind_material)
    if spec.articulated:
        add(follow_contacts)
    if spec.laws.warm_start:
        add(warm_start_contract)
    if spec.particles:
        add(wait_for_collision, particle_forces)
    if not spec.articulated:
        if spec.particles:
            add(integrate_particles)
        add(count, *boundary)
        return tuple(stages)

    # Stages 1-3: the tree factor (the drives, FK/ID, the mass refresh and the factor's, the walk's refresh with the
    # clock), then the predictor v_hat and the free roots' corrections.
    scalar = getattr(factor.held, "scalar", None)
    stream = factor.dynamics_stream

    # Scalar branches evaluate their own drives in their predict: the drives' stage takes the other coordinates.
    drive_dofs = None if scalar is None else scalar.other_drive_dofs

    def check_call(step):
        cached_state_binding.check_call(model.device, step.state_in, step.state_out, step.dt)

    def drives(step):
        factor.prepare_drives(step.state_in, step.control, step.dt, solver.v_hat, solver.v_out, drive_dofs)

    def drives_on_stream(step):
        # The drives overlap FK/ID on the dynamics stream; the mask (refresh) joins them.
        stream.wait_event(wp.get_stream(model.device).record_event())
        with wp.ScopedStream(stream, sync_enter=False):
            drives(step)

    def join_drives(step):
        wp.get_stream(model.device).wait_event(stream.record_event())

    def prescale(step):
        factor.prescale_velocity(step.state_in.joint_qd, solver.qd_work)
        step.qd = solver.qd_work

    def velocity(step):
        step.qd = step.state_in.joint_qd

    def fk_id(step):
        factor.fk_id(step.state_in, step.state_out, step.qd, step.dt)

    def refresh(step):
        factor.refresh(step.dt)

    def walk_step(step):
        factor.walk_step(step.state_in, step.dt)

    def advance_clock(step):
        factor.advance_clock(step.dt)

    def predict(step):
        factor.predict(step.state_in, step.control, step.qd, step.dt)

    def correct_free_roots(step):
        factor.correct_free_roots(step.state_in, step.qd, step.dt, solver.v_hat)

    def damp_spin(step):
        factor.damp_spin(step.dt, solver.v_hat)

    drive_stages = ()
    if factor.parallel_drives and (scalar is None or scalar.other_drive_dofs is not None):
        drive_stages = (drives_on_stream,) if stream is not None else (drives,)
    add(
        check_call,
        *drive_stages,
        prescale if solver.options.enable_joint_velocity_limits else velocity,
        fk_id,
        *((join_drives,) if drive_stages and stream is not None else ()),
        refresh,
        advance_clock if factor.walk_arts is None else walk_step,
        nvtx="S1_FK_ID_CRBA",
    )
    add(
        predict,
        *((correct_free_roots,) if forest.free_root_joint_count and forest.free_rigid_body_count else ()),
        *((damp_spin,) if solver.options.angular_damping != 0.0 else ()),
        nvtx="S3_Trisolve_Vhat",
    )

    def snapshots(step):
        wp.copy(solver._debug_stage3_qd_work, step.qd)
        wp.copy(solver._debug_stage3_joint_qdd, factor.joint_qdd)
        wp.copy(solver._debug_stage3_v_hat, solver.v_hat)

    if "snapshots" in diagnostics:
        add(snapshots)
    if not spec.particles:
        add(wait_for_collision)

    # Stage 4: the rows. The resolved-world claim, the row pass (each family's slots and seeds), the status and the
    # overflow warning; then the diagonals, the one writer (every walking class's: it walks the rows into their
    # placement's sink after the row pass, each row's rhs by the row law; the free bodies' writer runs in the row pass,
    # after their families' slots, beside the topology part), the patches' link once every contact row's records are
    # written, and the sweep's claims, which read the rows.
    resolved = sweep.resolved_worlds_owner

    def classify_resolved_worlds(step):
        resolved.classify(step.state_in, factor, step.contacts, step.dt)

    # The row pass, its families in the launch order that is the slot order (rows.py): each family a class holds. The
    # joint families' rows are seeds the one writer walks: a class that walks no rows has no responding coordinate, and
    # holds none (no contact is admitted there either).
    keyed = placement.keyed_rows is not None
    holds = placement.walks
    limit_sizes = laws.limit_sizes(solver) if solver.options.enable_joint_limits and holds and not keyed else ()

    def begin_rows(step):
        laws.begin(solver, step.state_in, factor, step.dt)

    def write_mimics(step):
        laws.write_mimics(solver, step.state_in, step.dt)

    def write_links(step):
        laws.write_links(solver, step.state_in)

    def prepare_fused_limits(step):
        laws.prepare_fused_limits(solver, step.state_in, step.dt)

    def write_limits(step):
        laws.write_limits(solver, step.state_in, step.dt, limit_sizes)

    def bound_limits(step):
        laws.bound_phase(solver, 0)

    def write_velocity_limits(step):
        laws.write_velocity_limits(solver, step.state_in, step.dt)

    def bound_velocity_limits(step):
        laws.bound_phase(solver, 1)

    def write_contacts(step):
        laws.write_contacts(solver, step.state_in, factor, step.contacts, step.dt, route.producer)

    def bound_free_body_contacts(step):
        laws.bound_free_body_contacts(solver)

    def allocate_free_body_limits(step):
        laws.allocate_free_body_limits(solver)

    def write_free_bodies(step):
        laws.write_free_bodies(solver, step.state_in, step.contacts, step.dt)

    def finalize_dense(step):
        laws.finalize_dense(solver)

    def finish_keyed(step):
        placement.keyed_rows.finish_rows()

    # The dense rows' family bounds exist where the row pass holds the dense rows' order (no keyed rows).
    bounds = not keyed

    # The rows a world's topology fixes (links, mimics, limits) are known once their families are written, before the
    # contacts: where the family bound holds them and the one writer walks the general placement's rows, it walks them
    # beside the contact pass (Beside) and the rest after it, the two parts joined before the rest. Their diagonals
    # start from zero before them.
    topology = Beside(model.device) if holds and bounds and route.case not in ("K1", "K2") else None

    def write_topology_rows(step):
        if placement.clears_diagonals:
            placement.diagonals()
        topology.start(lambda: laws.write_walked(solver, step.state_in, step.contacts, step.dt, WRITE_TOPOLOGY))

    write_rows = (
        begin_rows,
        *((write_mimics,) if laws.mimic_warp_kernels and holds else ()),
        *((write_links,) if laws.link_count > 0 and holds else ()),
        *((prepare_fused_limits,) if spec.classes.fused_sizes else ()),
        *((write_limits,) if limit_sizes else ()),
        *((bound_limits,) if bounds else ()),
        *((write_velocity_limits,) if laws.velocity_limit_warp_kernels and holds and not keyed else ()),
        *((bound_velocity_limits,) if bounds else ()),
        *((write_topology_rows,) if topology is not None else ()),
        write_contacts,
        *((bound_free_body_contacts,) if forest.has_free_rigid_bodies else ()),
        *(
            (allocate_free_body_limits,)
            if forest.has_free_rigid_bodies and placement.rigid_velocity_limit_slot is not None
            else ()
        ),
        *((write_free_bodies,) if forest.has_free_rigid_bodies else ()),
        finalize_dense,
        *((finish_keyed,) if keyed else ()),
    )

    def link_patches(step):
        laws.link_patches(solver, step.contacts)

    def latch_status(step):
        placement.latch_status(step.contacts)

    def claim(step):
        sweep.claim()

    def warn_overflow(step):
        placement.warn_overflow()

    def diagonals(step):
        placement.diagonals()

    def write_walked(step):
        if topology is None:
            laws.write_walked(solver, step.state_in, step.contacts, step.dt)
            return
        topology.join()
        laws.write_walked(solver, step.state_in, step.contacts, step.dt, WRITE_REST)

    # The one writer writes its contact rows' records after the row pass, so the patches link after it; the claims
    # (the sparse diagonal's schedule) read the rows it wrote and the marks of their slots.
    add(
        *((classify_resolved_worlds,) if resolved is not None else ()),
        *write_rows,
        latch_status,
        *((warn_overflow,) if solver.options.diagnostics.overflow_warning else ()),
        nvtx="S4_ContactBuild",
    )
    add(
        *((diagonals,) if placement.clears_diagonals and topology is None else ()),
        *((write_walked,) if holds else ()),
        link_patches,
        claim,
        nvtx="S4_HinvJt_Diag_RHS",
    )

    # Stages 5 and 6: the impulses' start (the warm start's normals, the closures' seeds), the claims that
    # follow the published rows, and the sweep.
    def prepare_impulses(step):
        placement.prepare_impulses()

    def warm_start(step):
        sweep.warm_start(step.contacts)

    def seed_blocks(step):
        # The closure's seed: the one writer closed the bilateral blocks and corrected the other rows; the blocks'
        # kept rows start at -r_Q, after the warm start's gather.
        factor.seed_blocks(first=placement.dense_first, rhs=placement.rhs, impulses=placement.impulses)

    def start_velocity(step):
        # The sweeps add their velocity change to the predictor; a world without rows keeps it.
        wp.copy(solver.v_out, solver.v_hat)

    def pack_free_body_meta(step):
        placement.pack_free_body_meta()

    colored = sweep.colored_units

    def build_colored_units(step):
        # The colored worlds' units and colors follow the rows the pack just published.
        colored.build(
            placement.mf_constraint_count,
            placement.dense_first,
            placement.mf_contact_rows_end,
            placement.mf_meta_packed,
            placement.packed_Z,
            placement.row_mu,
            placement.row_w,
            placement.body_base,
        )

    def solve(step):
        sweep.solve(solver.v_hat, solver.v_out)

    add(prepare_impulses, *((warm_start,) if spec.laws.warm_start else ()), nvtx="S5_PGS_Prep")
    add(
        *((seed_blocks,) if spec.laws.preelimination else ()),
        start_velocity,
        *((pack_free_body_meta,) if forest.has_free_rigid_bodies else ()),
        *((build_colored_units,) if colored is not None else ()),
        nvtx="S5_GatherJY",
    )
    add(solve, nvtx="S6_PGS_Solve")
    if "audit" in diagnostics:
        audit = solver._audit

        def audit_step(step):
            audit.launch()

        add(audit_step)

    # Stage 7: the finish (qdd, integration, publication), the carry the next step reads, the leave.
    def finish(step):
        factor.finish(step.state_in, step.state_out, step.dt)

    def carry_patches(step):
        sweep.finish_patches(step.state_in, step.state_out, step.contacts, step.dt)

    def carry(step):
        sweep.carry(step.contacts)

    def watermarks(step):
        placement.record_watermarks(step.contacts)

    def resolved_watermark(step):
        # The rows the resolved worlds skipped: the diagnostic counts them and leaves the classifier's choice alone.
        resolved.record_watermark(placement.row_watermark_resolved)

    add(finish, nvtx="S7_Integrate")
    if spec.particles:
        add(integrate_particles)
    # The carry: the patches' history, then under the warm start the contacts' normals and normal impulses.
    add(carry_patches, nvtx="S7_Friction_Patch_Carry")
    if spec.laws.warm_start:
        add(carry, nvtx="S7_Warmstart_Carry")
    if "row_watermark" in diagnostics:
        add(watermarks, *((resolved_watermark,) if resolved is not None else ()))
    add(count, *boundary)
    return tuple(stages)
