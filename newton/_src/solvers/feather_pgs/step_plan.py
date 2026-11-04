# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The step's spec and plan: every unknown that decides the step's path, gathered once, and what each stage launches.

:class:`StepSpec` is the frozen record of the unknowns, built at the end of construction: the device, the forest, the
route (the laws and the class, today the placement case), the capacities, the mass cadence, the sweep's kernels and
streams, and the diagnostics in force. :class:`StepPlan` and the route record (:mod:`.path_selection`) read it; nothing
in it is computed a second time elsewhere.

The plan's :attr:`StepPlan.stages` are the step: each a closure bound once to its owners, chosen from the spec, so
``step()`` is the stages run in order on a :class:`Step` with no construction constant re-tested. Every stage whose
work depends on the articulation size (where stage 4 whitens the rows) has one table: each size's route and the
reason for it. The solve of stages 5 and 6 has one entry: the rows its sweep solves and what drives the sweep. The
route record (:mod:`.path_selection`) reports the tables and the stages' order, and construction builds exactly the
kernels they name, so the three cannot disagree.

The plan reads the solver's route (its row of the route table: the placement case
and producer) and the mapping beside it (the
execution plan's resource choices, the kernel choices); it decides nothing that
construction has not, and nothing in it changes after construction.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import NamedTuple

import warp as wp

from ...core.memory import Device, device
from ...sim import ModelFlags
from . import cached_state_binding
from .forest_plan import ForestPlan
from .route_table import RouteRow

__all__ = [
    "Capacities",
    "Route",
    "SolveRoute",
    "Stage",
    "Step",
    "StepPlan",
    "StepSpec",
    "TopologyPin",
    "first_failed",
    "kernel_key",
]


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
    rhs: str
    """Who writes the dense rows' right-hand sides: ``"walk"``, the one writer, which walks the rows after the row pass
    (every walking class but the sparse diagonal's); ``"producers"``, the row pass's own producers (the sparse
    diagonal's slot rows, the keyed rows); ``"bias"``, the rows' bias launch apart."""
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
            *((f"claim{i}", stream) for i, stream in enumerate(sweep.claim_streams)),
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
            rhs=(
                "walk"
                if placement.walks and route.case != "K2"
                else "producers"
                if placement.walks or placement.keyed_rows is not None
                else "bias"
            ),
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
            "rhs": self.rhs,
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
    """The step of one solver: its stages and per-stage dispatch tables, built once by :meth:`build`."""

    response: Mapping[int, Route]
    """Stage 4, per size, where its rows are whitened: ``owner`` (a contact owner publishes the response),
    ``sparse_diagonal`` or ``packed`` (the packed rows' build)."""
    solve: SolveRoute
    stages: tuple[Stage, ...]
    """The step, in order (:func:`_stages`)."""

    @classmethod
    def build(cls, spec: StepSpec, solver) -> StepPlan:
        """Read the plan from the step's ``spec`` and bind its stages to the ``solver``'s owners."""
        return cls(
            response=MappingProxyType(_response(spec.route, spec.forest)),
            solve=_solve(spec.route),
            stages=_stages(spec, solver),
        )

    def run(self, step: Step) -> None:
        """Run the step's stages in order."""
        for stage in self.stages:
            stage.run(step)

    def sizes(self, stage: str, route: str) -> tuple[int, ...]:
        """The sizes a stage's table routes to ``route``."""
        return tuple(size for size, entry in getattr(self, stage).items() if entry.name == route)

    def record(self) -> dict:
        """The tables as a JSON-compliant record (``routes.stages`` of the path selection)."""

        def table(routes):
            return {str(int(size)): route.record() for size, route in routes.items()}

        return {
            "response": table(self.response),
            "order": [stage.name for stage in self.stages],
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


def _stages(spec: StepSpec, solver) -> tuple[Stage, ...]:
    """The step's stages, each bound once to its owners.

    Every construction constant the step reads is decided here, from the spec: which stages exist, the writer, the
    optional claims and the diagnostics. A stage then runs without re-testing one; what it still tests is the call's
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
    route, forest, diagnostics = spec.route, spec.forest, spec.diagnostics
    stages: list[Stage] = []

    def add(*runs, nvtx: str | None = None) -> None:
        """Append ``runs`` as stages; under the NVTX diagnostic a named group runs as one stage inside its range."""
        if nvtx is not None and "nvtx" in diagnostics:

            def ranged(step, runs=runs, name=nvtx):
                with wp.ScopedTimer(name, print=False, use_nvtx=True, synchronize=False):
                    for run in runs:
                        run(step)

            stages.append(Stage(nvtx, ranged))
        else:
            stages.extend(Stage(run.__name__, run) for run in runs)

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
    if route.laws.warm_start:
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

    # Stage 4: the rows. The resolved-world claim, the row pass, the patches' link once every contact row's records are
    # written, the status, the sweep's claims and the overflow warning; then the diagonals and the right-hand sides'
    # writer (spec.rhs: the one writer walks the rows into their placement's sink, the producers wrote them, or the rows'
    # bias is written apart), then the free bodies' rows' rhs.
    resolved = sweep.resolved_worlds_owner

    def classify_resolved_worlds(step):
        resolved.classify(step.state_in, factor, step.contacts, step.dt)

    # The row pass, its families in the launch order that is the slot order (rows.py): each family a class holds.
    keyed = placement.keyed_rows is not None
    limit_sizes = laws.limit_sizes(solver) if solver.options.enable_joint_limits and not keyed else ()

    def begin_rows(step):
        laws.begin(solver, step.state_in, factor, step.dt)

    def write_mimics(step):
        laws.write_mimics(solver, step.state_in, step.dt)

    def write_links(step):
        laws.write_links(solver, step.state_in, factor)

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
        laws.write_contacts(solver, step.state_in, factor, step.contacts, step.dt)

    def bound_free_body_contacts(step):
        laws.bound_free_body_contacts(solver)

    def write_free_body_limits(step):
        laws.write_free_body_limits(solver)

    def finalize_dense(step):
        laws.finalize_dense(solver)

    def finish_keyed(step):
        placement.keyed_rows.finish_rows()

    # The dense rows' family bounds exist where the row pass holds the dense rows' order (no keyed rows).
    bounds = not keyed
    write_rows = (
        begin_rows,
        *((write_mimics,) if laws.mimic_warp_kernels else ()),
        *((write_links,) if laws.link_count > 0 else ()),
        *((prepare_fused_limits,) if placement.fused_diagonal_joint_limits else ()),
        *((write_limits,) if limit_sizes else ()),
        *((bound_limits,) if bounds else ()),
        *((write_velocity_limits,) if laws.velocity_limit_warp_kernels and not keyed else ()),
        *((bound_velocity_limits,) if bounds else ()),
        write_contacts,
        *((bound_free_body_contacts,) if forest.has_free_rigid_bodies else ()),
        *(
            (write_free_body_limits,)
            if forest.has_free_rigid_bodies and placement.rigid_velocity_limit_slot is not None
            else ()
        ),
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
        laws.write_walked(solver, step.state_in, step.contacts, step.dt)

    def write_rhs(step):
        laws.write_rhs(solver, step.dt)

    def write_free_body_rhs(step):
        laws.write_free_body_rhs(solver, step.dt)

    def write_free_body_rhs_timed(step):
        with solver._sync_timed("s4_mf_setup"):
            laws.write_free_body_rhs(solver, step.dt)

    # The one writer writes its contact rows' records after the row pass, so the patches link after it there.
    link = (link_patches,) if laws.friction_anchors_enabled else ()
    add(
        *((classify_resolved_worlds,) if resolved is not None else ()),
        *write_rows,
        *(link if spec.rhs != "walk" else ()),
        latch_status,
        claim,
        *((warn_overflow,) if solver.options.diagnostics.overflow_warning else ()),
        nvtx="S4_ContactBuild",
    )
    rhs = {"walk": (write_walked, *link), "producers": (), "bias": (write_rhs,)}[spec.rhs]
    add(*((diagonals,) if placement.clears_diagonals else ()), *rhs, nvtx="S4_HinvJt_Diag_RHS")
    if forest.has_free_rigid_bodies:
        add(write_free_body_rhs_timed if "sync_timings" in diagnostics else write_free_body_rhs, nvtx="S4_MF_Setup")

    # Stages 5 and 6: the impulses' start (warm start, the patches' seeds, the closures' seeds), the claims that
    # follow the published rows, and the sweep.
    def prepare_impulses(step):
        placement.prepare_impulses()

    def warm_start(step):
        sweep.warm_start(step.contacts, step.dt)

    def seed_patches(step):
        sweep.seed_patches(step.state_in, step.contacts, step.dt)

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
    weighted = int(route.laws.weighted)

    def build_colored_units(step):
        # The colored worlds' units and colors follow the rows the pack just published.
        colored.build(
            placement.mf_constraint_count,
            placement.dense_first,
            placement.mf_contact_rows_end,
            placement.mf_meta_packed,
            placement.mf_Z,
            placement.row_mu,
            placement.row_w,
            placement.body_base,
            weighted,
        )

    def solve(step):
        sweep.solve(solver.v_hat, solver.v_out)

    add(prepare_impulses, *((warm_start,) if route.laws.warm_start else ()), nvtx="S5_PGS_Prep")
    if route.laws.patch:
        add(seed_patches)
    add(
        *((seed_blocks,) if route.laws.preelimination else ()),
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

    def carry(step):
        sweep.carry(step.state_in, step.state_out, step.contacts, step.dt)

    def watermarks(step):
        placement.record_watermarks(step.contacts)

    def resolved_watermark(step):
        # The rows the resolved worlds skipped: the diagnostic counts them and leaves the classifier's choice alone.
        resolved.record_watermark(placement.row_watermark_resolved)

    add(finish, nvtx="S7_Integrate")
    if spec.particles:
        add(integrate_particles)
    add(carry)
    if "row_watermark" in diagnostics:
        add(watermarks, *((resolved_watermark,) if resolved is not None else ()))
    add(count, *boundary)
    return tuple(stages)
