# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physics conformance gate of SolverFeatherPGS.

The gate checks physics, not generated code: it runs small deterministic scenes
(:mod:`newton.tests.feather_pgs_scenes`) through every route the solver admits
for them and requires each route to obey the same laws and reach the converged
reference trajectory, whichever kernels serve it. A change may rewrite or merge
kernels freely as long as the physics holds.

Routes
    A route is a set of constructor options, test-only axis pins
    (``SolverFeatherPGS._route_pins``) and import-time switches, patched for
    the whole run. Every run reads
    ``get_path_selection()["routes"]`` before its first step and asserts the
    record the route expects, so a test cannot silently run another route.
    Owners that claim worlds at run time (sparse factors, the parallel tiers,
    local owners, the paired and sparse-diagonal sweeps) also get an ownership
    test that reads the solve readout. Covered on CUDA: native sparse factors
    (with contact triplets), the matrix-free Gauss-Seidel sweep with its
    parallel and keyed-parallel tiers, local owners (row packets and owned rows), the paired factor sweep
    and the sparse-diagonal sweep (sparse-diagonal rows and the compact contact
    boundary); on CPU, split mode. Dynamics owners: ``TreeSparseOperator``,
    ``TreeHeldOperator``, ``ForestHeldOperator``, scalar branches and ordinary
    dynamics. Contact-row producers: sparse response, compact Jacobian, general
    (per contact and masked), keyed rows, local packets, sparse diagonal and
    compact contact boundary. Friction: point (``friction_anchor_beta=0``) and
    persistent patch friction (0.2).

Scenes and features
    Dropped and stacked boxes, a box sticking and one sliding on an incline, a
    bouncing ball, a box and a sphere on a heightfield, boxes pushed against a
    rigid-body speed limit, a revolute chain on limits and drives, a four-bar
    loop, a mimic pair, a spring hinge, an arm under joint velocity limits, a
    legged robot standing on a free base, a gripper lifting a box, a
    diagonal-mass keyboard pressed by an arm, a 17-coordinate hand pressing a
    free cube, and two articulations above the held-tree owner's caps (a
    66-coordinate humanoid-like tree, a 100-link snake with and without joint
    armature). Articulated contact
    scenes hold two worlds that differ, so a world mix-up shows. Variants: PhysX
    drives, velocity iterations, warm start, contact compliance, contact
    torsion, ``contact_update="unit"``, restitution, bilateral pre-elimination,
    joint and rigid-body velocity limits.

Metrics
    Computed on the host from the model, the states before and after each
    step, the step's contacts and the forces
    :meth:`~newton.solvers.SolverFeatherPGS.update_contacts` exports into
    ``Contacts.rigid_contact_force``, never from route-private buffers. Contact
    laws use the step's post-solve velocities at its pre-step pose, the pose its
    rows linearize:

    - normal force ``f_n >= 0``; Coulomb cone ``|f_t| <= mu f_n`` per contact
      under point friction, per body pair under patch friction (whose friction
      rows act at patch anchors);
    - natural residuals of the normal law ``|min(f_n dt / m, v_n + b)|`` (``b``
      the Baumgarte or speculative bias) and of Coulomb friction with maximum
      dissipation ``|x - Proj_cone(x - v_t)|``, ``x = f_t dt / m``: a sticking
      contact does not slip, a sliding one is on the cone opposing its slip;
    - penetration after a step, energy of passive scenes, joint-limit
      violation, loop-closure gap and mimic error;
    - momentum balance of every floating articulation against the exported
      forces (and the angular balance of free rigid bodies under point
      friction), which checks the force export on every route;
    - scene laws: sliding acceleration ``g (sin a - mu cos a)`` and no creep on
      the incline, rebound ``v+ = -e u``, speed limits, the spring hinge's
      discrete oscillator, the compliant rest height ``r - m g / k``, torsional
      stiction and spin-up, the lifted box following the gripper, and the
      exported force on a pressed key balancing its drive.

    Individual contact forces are not compared between routes: a statically
    indeterminate distribution (a box's four corners, the lateral friction of
    planar legs) may differ while the motion agrees.

Reference and tolerances
    Per scene, friction law and variant, the reference route runs at the gate
    budget ``N`` and at ``8 N`` and ``8 N + 1`` iterations: the plain
    matrix-free Gauss-Seidel sweep over ordinary dynamics on CUDA, split mode on
    CPU. ``8 N`` is the converged reference. Its spread, the larger distance of
    the runs at ``N`` and ``8 N + 1`` (sequential Gauss-Seidel can cycle with
    period two) from it and, on CUDA, of a repeat of the run at ``N`` (atomic
    allocation varies the row order; scenes whose support is statically
    indeterminate sort their contacts instead), sets every route's trajectory
    tolerance. A statically indeterminate support also turns rounding into a
    different motion (the box stack's patch reference ends 1.6e-3 m/s from
    itself with its time step one float32 ulp longer, three times the bound its
    convergence alone sets), so such a scene's spread includes the run at ``N``
    with the time step one ulp longer and shorter: a route's arithmetic differs
    from the reference's by rounding at least:
    ``4 x spread + float32 floor`` in body position, rotation, velocity and
    angular velocity, over the run and at its end (a pose at rest under static
    friction is one of a continuum, so the end pose takes the run's spread). A
    route's natural residuals,
    penetration, energy gain, patch cone excess, limit, loop and mimic errors
    may exceed the reference's at ``N`` by the same factor plus a floor; the
    point cone, the normal force sign and the free-body momentum balance, which
    every iterate satisfies, get a float32 floor only. The parallel (Jacobi)
    tiers run the solver's default budget and exit law, as users run them: the
    automatic budget sizes each tier from ``N`` and its row bound. Each group also
    checks that its routes agree with each other, and on CUDA that the converged
    CUDA and CPU references agree.

Findings
    Routes that fail a check at the gate's baseline are expected failures
    that name the finding.

Runtime
    Small scenes (one or two worlds, 60 to 240 steps): with a warm kernel
    cache, the whole module takes about five minutes on one GPU, the CPU class
    about one. A cold cache adds the compilation of every route's kernels.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import math
import os
import time
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import solver_feather_pgs as solver_module
from newton.solvers import SolverFeatherPGS
from newton.tests import feather_pgs_scenes as scenes

# ---------------------------------------------------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------------------------------------------------

BASE_OPTIONS = {
    # Exact energy and momentum balances: no artificial damping of free roots.
    "angular_damping": 0.0,
}


@dataclasses.dataclass(frozen=True)
class Route:
    """One way the solver can run a scene, forced by options and flags and identified by its route record."""

    name: str
    device: str
    """``"cpu"`` or ``"cuda"``."""
    options: dict = dataclasses.field(default_factory=dict)
    """Constructor options that select the route."""
    pins: dict = dataclasses.field(default_factory=dict)
    """Axis values the route fixes where the rules would choose others (``SolverFeatherPGS._route_pins``)."""
    module: dict = dataclasses.field(default_factory=dict)
    """Attributes patched for the whole run: the solver module's import-time switches, or ``"module.attribute"`` of
    another FeatherPGS module."""
    expect: dict = dataclasses.field(default_factory=dict)
    """Route-record values the run must report: ``"key.path": value`` into ``get_path_selection()["routes"]``."""

    def but(
        self, name: str, *, options: dict | None = None, pins: dict | None = None, expect: dict | None = None
    ) -> Route:
        """This route with more options, pins and expectations (a feature variant served by the same route)."""
        return dataclasses.replace(
            self,
            name=name,
            options={**self.options, **(options or {})},
            pins={**self.pins, **(pins or {})},
            expect={**self.expect, **(expect or {})},
        )


def route_value(routes: dict, path: str):
    """Read ``path`` (dot-separated keys, integer list indices) from a route record."""
    value = routes
    for key in path.split("."):
        if isinstance(value, list):
            value = value[int(key)] if int(key) < len(value) else None
        elif isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value


@contextlib.contextmanager
def route_context(route: Route):
    """Set the route's pins and module switches, clearing every solver variable of the environment."""
    names = [name for name in os.environ if name.startswith(("FEATHER_PGS_", "IL_NEWTON_FPGS_"))]
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ))
        for name in names:
            del os.environ[name]
        stack.enter_context(mock.patch.object(SolverFeatherPGS, "_route_pins", dict(route.pins)))
        for name, value in route.module.items():
            owner, _, attribute = name.rpartition(".")
            target = importlib.import_module(f"newton._src.solvers.feather_pgs.{owner}") if owner else solver_module
            stack.enter_context(mock.patch.object(target, attribute, value))
        yield


# ---------------------------------------------------------------------------------------------------------------------
# Simulation record
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Run:
    """Everything a route's run records, on the host."""

    scene: scenes.Scene
    route: Route
    routes: dict
    options: dict
    body_q: np.ndarray  # [steps + 1, bodies, 7]
    body_qd: np.ndarray  # [steps + 1, bodies, 6]
    joint_q: np.ndarray  # [steps + 1, coords]
    joint_qd: np.ndarray  # [steps + 1, dofs]
    body_qd_solve: np.ndarray  # [steps, bodies, 6] post-solve velocity at the pre-step pose
    contacts: list  # per step: dict of contact arrays (pre-step geometry, exported forces)
    capacity: dict
    measurements: dict | None = None  # get_path_selection()["measurements"] after the last step, with a readout


CONTACT_FIELDS = (
    "rigid_contact_shape0",
    "rigid_contact_shape1",
    "rigid_contact_point0",
    "rigid_contact_point1",
    "rigid_contact_normal",
    "rigid_contact_margin0",
    "rigid_contact_margin1",
    "rigid_contact_force",
)
STATE_FIELDS = ("body_q", "body_qd", "joint_q", "joint_qd")


class _Recorder:
    """Device-side history of a run: each step copies into preallocated buffers; one host read at the end."""

    def __init__(self, arrays: dict, steps: int, device):
        self.buffers = {
            name: wp.empty(steps * array.shape[0], dtype=array.dtype, device=device) for name, array in arrays.items()
        }

    def record(self, step: int, arrays: dict) -> None:
        for name, array in arrays.items():
            size = array.shape[0]
            wp.copy(self.buffers[name], array, dest_offset=step * size, count=size)

    def read(self, steps: int) -> dict:
        history = {}
        for name, buffer in self.buffers.items():
            values = buffer.numpy()
            history[name] = values.reshape(steps, -1, *values.shape[1:])
        return history


def simulate(
    scene: scenes.Scene, route: Route, device, *, options: dict | None = None, steps: int | None = None
) -> Run:
    """Step ``scene`` through ``route`` and record states, contacts and exported contact forces.

    The output state is copied back into the input state, so the solver sees
    one input object, as a captured step loop does. The history is recorded on
    the device and read once at the end. The route record is read before the
    first step and checked against ``route.expect`` by the caller.
    """
    steps = scene.steps if steps is None else steps
    model = scene.build(device)
    pipeline_options = {}
    if scene.rigid_contact_max is not None:
        pipeline_options["rigid_contact_max"] = scene.rigid_contact_max
    if scene.contact_matching is not None:
        pipeline_options["contact_matching"] = scene.contact_matching
    if scene.deterministic_contacts:
        pipeline_options["deterministic"] = True
    pipeline = newton.CollisionPipeline(model, **pipeline_options)
    contacts = pipeline.contacts()
    # The history holds the rows the scene's contacts can fill (its budget), not every row the
    # contact buffers' first pages map, which can be a hundred times more.
    recorded = min(contacts.rigid_contact_max, pipeline.rigid_contact_max)
    if scene.contact_stiffness > 0.0:
        capacity = contacts.rigid_contact_normal.shape[0]
        for name in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
            setattr(contacts, name, wp.zeros(capacity, dtype=float, device=model.device))
    merged = {**BASE_OPTIONS, **scene.solver_options, **(options or {}), **route.options}
    with route_context(route):
        solver = SolverFeatherPGS(model, **merged)
        routes = solver.get_path_selection()["routes"]
        state, out, scratch = model.state(), model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        initial = {name: getattr(state, name).numpy().copy() for name in STATE_FIELDS}

        def step_arrays():
            arrays = {name: getattr(contacts, name)[:recorded] for name in CONTACT_FIELDS}
            arrays.update({name: getattr(state, name) for name in STATE_FIELDS})
            arrays["contact_count"] = contacts.rigid_contact_count
            arrays["body_qd_solve"] = scratch.body_qd
            if model.body_count:
                arrays["body_f"] = state.body_f
            return arrays

        recorder = None
        for step in range(steps):
            state.clear_forces()
            if scene.control is not None:
                scene.control(model, control, state, step)
            pipeline.collide(state, contacts)
            if scene.contact_stiffness > 0.0:
                contacts.rigid_contact_stiffness.fill_(scene.contact_stiffness)
                contacts.rigid_contact_damping.fill_(scene.contact_damping)
                contacts.rigid_contact_friction.fill_(1.0)
            pre = {name: getattr(contacts, name)[:recorded] for name in CONTACT_FIELDS[:-1]}
            pre["contact_count"] = contacts.rigid_contact_count
            if model.body_count:
                pre["body_f"] = state.body_f
            if recorder is None:
                recorder = _Recorder(step_arrays(), steps, model.device)
            recorder.record(step, pre)
            solver.step(state, out, control, contacts, scene.dt)
            solver.update_contacts(contacts)
            newton.eval_fk(model, state.joint_q, out.joint_qd, scratch)
            for name in STATE_FIELDS:
                wp.copy(getattr(state, name), getattr(out, name))
            recorder.record(
                step,
                {
                    "rigid_contact_force": contacts.rigid_contact_force[:recorded],
                    "body_qd_solve": scratch.body_qd,
                    **{name: getattr(state, name) for name in STATE_FIELDS},
                },
            )
        capacity = solver.constraint_capacity_status()
        measurements = None
        if merged.get("pgs_solve_readout", "off") != "off":
            measurements = solver.get_path_selection()["measurements"]
    history = recorder.read(steps)
    counts = history["contact_count"].reshape(steps)
    contact_log = []
    for step in range(steps):
        count = int(counts[step])
        entry = {name: history[name][step, :count] for name in CONTACT_FIELDS}
        entry["force"] = entry.pop("rigid_contact_force")
        entry["body_f"] = history["body_f"][step].astype(np.float64) if "body_f" in history else None
        contact_log.append(entry)

    def trajectory(name):
        return np.concatenate([initial[name][None], history[name]], axis=0).astype(np.float64)

    return Run(
        scene=scene,
        route=route,
        routes=routes,
        options=merged,
        body_q=trajectory("body_q"),
        body_qd=trajectory("body_qd"),
        joint_q=trajectory("joint_q"),
        joint_qd=trajectory("joint_qd"),
        body_qd_solve=history["body_qd_solve"].astype(np.float64),
        contacts=contact_log,
        capacity=capacity,
        measurements=measurements,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Route-independent metrics (host, float64)
# ---------------------------------------------------------------------------------------------------------------------


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` by unit quaternions ``q`` (x, y, z, w), broadcasting over leading axes."""
    axis, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(axis, v)
    return v + w * t + np.cross(axis, t)


def _transform_point(pose: np.ndarray, point: np.ndarray) -> np.ndarray:
    return pose[..., :3] + _quat_rotate(pose[..., 3:7], point)


@dataclasses.dataclass
class ModelData:
    """Host copies of the model arrays the metrics read."""

    shape_body: np.ndarray
    shape_mu: np.ndarray
    shape_restitution: np.ndarray
    body_com: np.ndarray
    body_mass: np.ndarray
    body_inertia: np.ndarray
    body_world: np.ndarray
    gravity: np.ndarray  # per body [bodies, 3]
    free_bodies: np.ndarray  # bodies that are single-body FREE articulations
    floating: np.ndarray  # articulations on a FREE root joint
    floating_bodies: list  # bodies of each floating articulation
    joint_limit_lower: np.ndarray
    joint_limit_upper: np.ndarray
    joint_dof_limited: np.ndarray  # DOFs of revolute/prismatic joints, ordered as joint_qd
    joint_dof_coord: np.ndarray  # coordinate index of each such DOF

    @classmethod
    def of(cls, model) -> ModelData:
        body_world = model.body_world.numpy()
        gravity = model.gravity.numpy().astype(np.float64)
        per_body = gravity[np.where(body_world >= 0, body_world, gravity.shape[0] - 1)]
        joint_type = model.joint_type.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        art_start = model.articulation_start.numpy()
        kinematic = (model.body_flags.numpy() & int(newton.BodyFlags.KINEMATIC)) != 0
        free, floating, floating_bodies = [], [], []
        for art in range(model.articulation_count):
            joints = range(art_start[art], art_start[art + 1])
            if joint_type[joints[0]] != int(newton.JointType.FREE) or joint_parent[joints[0]] >= 0:
                continue
            if kinematic[joint_child[joints[0]]]:
                continue
            floating.append(art)
            floating_bodies.append(np.asarray([joint_child[joint] for joint in joints], dtype=np.int64))
            if len(joints) == 1:
                free.append(joint_child[joints[0]])
        q_start, qd_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        dofs, coords = [], []
        for joint, kind in enumerate(joint_type):
            if kind in (int(newton.JointType.REVOLUTE), int(newton.JointType.PRISMATIC)):
                dofs.append(qd_start[joint])
                coords.append(q_start[joint])
        return cls(
            shape_body=model.shape_body.numpy(),
            shape_mu=model.shape_material_mu.numpy().astype(np.float64),
            shape_restitution=model.shape_material_restitution.numpy().astype(np.float64),
            body_com=model.body_com.numpy().astype(np.float64),
            body_mass=model.body_mass.numpy().astype(np.float64),
            body_inertia=model.body_inertia.numpy().astype(np.float64),
            body_world=body_world,
            gravity=per_body,
            free_bodies=np.asarray(free, dtype=np.int64),
            floating=np.asarray(floating, dtype=np.int64),
            floating_bodies=floating_bodies,
            joint_limit_lower=model.joint_limit_lower.numpy().astype(np.float64),
            joint_limit_upper=model.joint_limit_upper.numpy().astype(np.float64),
            joint_dof_limited=np.asarray(dofs, dtype=np.int64),
            joint_dof_coord=np.asarray(coords, dtype=np.int64),
        )


@dataclasses.dataclass
class ContactFrame:
    """Per-contact quantities of one step: pre-step geometry, post-solve velocities and exported forces."""

    body0: np.ndarray
    body1: np.ndarray
    normal: np.ndarray  # shape 0 toward shape 1
    point0: np.ndarray  # world witness on body 0 (thickness included)
    point1: np.ndarray
    gap: np.ndarray  # signed gap phi [m]
    separation_velocity: np.ndarray  # n . (v1(p1) - v0(p0)) after the solve [m/s]
    slip: np.ndarray  # tangential velocity of body 0 relative to body 1 [m/s]
    force: np.ndarray  # force on body 0 from body 1 [N]
    normal_force: np.ndarray  # -force . n [N]
    friction: np.ndarray  # tangential part of force [N]
    mu: np.ndarray
    restitution: np.ndarray


def _point_velocity(body, point, body_q, body_qd, data: ModelData):
    velocity = np.zeros_like(point)
    moving = body >= 0
    if np.any(moving):
        b = body[moving]
        com = _transform_point(body_q[b], data.body_com[b])
        velocity[moving] = body_qd[b, :3] + np.cross(body_qd[b, 3:6], point[moving] - com)
    return velocity


def contact_frame(entry: dict, body_q: np.ndarray, body_qd_solve: np.ndarray, data: ModelData) -> ContactFrame:
    """Evaluate one step's contacts at the pre-step pose ``body_q`` with post-solve velocities ``body_qd_solve``."""
    shape0 = entry["rigid_contact_shape0"]
    shape1 = entry["rigid_contact_shape1"]
    body0 = np.where(shape0 >= 0, data.shape_body[np.maximum(shape0, 0)], -1)
    body1 = np.where(shape1 >= 0, data.shape_body[np.maximum(shape1, 0)], -1)
    normal = entry["rigid_contact_normal"].astype(np.float64)
    local0 = entry["rigid_contact_point0"].astype(np.float64)
    local1 = entry["rigid_contact_point1"].astype(np.float64)
    point0 = local0.copy()
    point1 = local1.copy()
    if np.any(body0 >= 0):
        point0[body0 >= 0] = _transform_point(body_q[body0[body0 >= 0]], local0[body0 >= 0])
    if np.any(body1 >= 0):
        point1[body1 >= 0] = _transform_point(body_q[body1[body1 >= 0]], local1[body1 >= 0])
    point0 = point0 + entry["rigid_contact_margin0"][:, None] * normal
    point1 = point1 - entry["rigid_contact_margin1"][:, None] * normal
    gap = np.einsum("ij,ij->i", normal, point1 - point0)
    v0 = _point_velocity(body0, point0, body_q, body_qd_solve, data)
    v1 = _point_velocity(body1, point1, body_q, body_qd_solve, data)
    relative = v0 - v1
    separation = -np.einsum("ij,ij->i", normal, relative)
    slip = relative + separation[:, None] * normal
    force = entry["force"].astype(np.float64)
    normal_force = -np.einsum("ij,ij->i", force, normal)
    friction = force + normal_force[:, None] * normal
    count = np.where(shape0 >= 0, 1.0, 0.0) + np.where(shape1 >= 0, 1.0, 0.0)
    mu = (
        np.where(shape0 >= 0, data.shape_mu[np.maximum(shape0, 0)], 0.0)
        + np.where(shape1 >= 0, data.shape_mu[np.maximum(shape1, 0)], 0.0)
    ) / np.maximum(count, 1.0)
    restitution = (
        np.where(shape0 >= 0, np.clip(data.shape_restitution[np.maximum(shape0, 0)], 0.0, 1.0), 0.0)
        + np.where(shape1 >= 0, np.clip(data.shape_restitution[np.maximum(shape1, 0)], 0.0, 1.0), 0.0)
    ) / np.maximum(count, 1.0)
    return ContactFrame(
        body0=body0,
        body1=body1,
        normal=normal,
        point0=point0,
        point1=point1,
        gap=gap,
        separation_velocity=separation,
        slip=slip,
        force=force,
        normal_force=normal_force,
        friction=friction,
        mu=mu,
        restitution=restitution,
    )


def mechanical_energy(body_q: np.ndarray, body_qd: np.ndarray, data: ModelData) -> np.ndarray:
    """Kinetic plus gravitational energy [J] per state, summed over bodies: ``body_q`` [..., bodies, 7]."""
    mass = data.body_mass
    com = _transform_point(body_q, data.body_com)
    linear = body_qd[..., :3]
    omega_world = body_qd[..., 3:6]
    rotation = body_q[..., 3:7]
    conj = rotation * np.array([-1.0, -1.0, -1.0, 1.0])
    omega_body = _quat_rotate(conj, omega_world)
    angular = np.einsum("...i,...ij,...j->...", omega_body, data.body_inertia, omega_body)
    kinetic = 0.5 * mass * np.einsum("...i,...i->...", linear, linear) + 0.5 * angular
    potential = -mass * np.einsum("...i,...i->...", data.gravity, com)
    return np.sum(kinetic + potential, axis=-1)


def _reduced_mass(frame: ContactFrame, data: ModelData) -> np.ndarray:
    inverse = np.zeros(frame.body0.shape[0])
    for body in (frame.body0, frame.body1):
        moving = body >= 0
        inverse[moving] += 1.0 / np.maximum(data.body_mass[body[moving]], 1.0e-12)
    return 1.0 / np.maximum(inverse, 1.0e-12)


def _project_disk(vectors: np.ndarray, radius: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vectors, axis=1)
    scale = np.where(length > radius, radius / np.maximum(length, 1.0e-30), 1.0)
    return vectors * scale[:, None]


@dataclasses.dataclass
class Laws:
    """Worst value over a run of each contact and joint law, in the units noted.

    ``None`` marks a law the run cannot evaluate (no contact, no limited joint).
    """

    normal_force_min: float | None = None
    """Smallest contact normal force [N] (negative: the export or the solve pulls)."""
    cone_excess: float | None = None
    """Largest ``|f_t| - mu f_n`` [N]: per contact under point friction, per body pair under patch friction."""
    normal_residual: float | None = None
    """Largest natural residual ``|min(f_n dt / m, v_n + b)|`` of the normal contact law [m/s]."""
    friction_residual: float | None = None
    """Largest natural residual of the Coulomb law with maximum dissipation [m/s] (point friction only)."""
    penetration: float | None = None
    """Largest penetration ``-phi`` found by a collision pass after a step [m]."""
    energy_gain: float | None = None
    """Largest rise of kinetic plus gravitational energy above its initial value [J] (passive scenes)."""
    momentum_residual: float | None = None
    """Largest ``|m dv - dt (m g + sum F)|`` of a free body over its weight impulse ``m g dt`` (the force export)."""
    torque_residual: float | None = None
    """Largest ``|I dw - dt sum r x F|`` of a free body over ``m g dt`` times its size [m] (point friction only)."""
    limit_violation: float | None = None
    """Largest joint position-limit violation [m or rad]."""


def contact_laws(run: Run, data: ModelData, *, patch: bool, skip: int = 0) -> Laws:
    """Evaluate the contact laws of every step from ``skip`` on, from public state, contacts and exported forces."""
    dt = run.scene.dt
    beta = float(run.options.get("pgs_beta", 0.2))
    speculative = float(run.options.get("contact_speculative_scale", 1.0))
    laws = Laws()
    worst = {}

    def keep(name, value, better):
        current = worst.get(name)
        worst[name] = value if current is None else better(current, value)

    for step, entry in enumerate(run.contacts):
        if step < skip or len(entry["force"]) == 0:
            continue
        frame = contact_frame(entry, run.body_q[step], run.body_qd_solve[step], data)
        keep("normal_force_min", float(frame.normal_force.min()), min)
        mass = _reduced_mass(frame, data)
        weight = dt / mass
        bias = np.where(frame.gap <= 0.0, beta * frame.gap / dt, speculative * frame.gap / dt)
        residual = frame.separation_velocity + bias
        # Restitution replaces the bias of a fast approaching contact: its normal law is checked by the drop test.
        bouncing = frame.restitution > 0.0
        normal = np.abs(np.minimum(np.maximum(frame.normal_force, 0.0) * weight, residual))
        if np.any(~bouncing):
            keep("normal_residual", float(normal[~bouncing].max()), max)
        if patch:
            pairs = {}
            for index, key in enumerate(zip(frame.body0.tolist(), frame.body1.tolist(), strict=True)):
                pairs.setdefault(key, []).append(index)
            excess = max(
                float(
                    np.linalg.norm(frame.friction[indices], axis=1).sum()
                    - frame.mu[indices[0]] * np.maximum(frame.normal_force[indices], 0.0).sum()
                )
                for indices in pairs.values()
            )
            keep("cone_excess", excess, max)
        else:
            radius = frame.mu * np.maximum(frame.normal_force, 0.0)
            keep("cone_excess", float((np.linalg.norm(frame.friction, axis=1) - radius).max()), max)
            impulse = frame.friction * weight[:, None]
            projected = _project_disk(impulse - frame.slip, radius * weight)
            keep("friction_residual", float(np.linalg.norm(impulse - projected, axis=1).max()), max)
        if step > 0:
            keep("penetration", float(max((-frame.gap).max(), 0.0)), max)
    for name, value in worst.items():
        setattr(laws, name, value)
    return laws


def momentum_balance(run: Run, data: ModelData, *, torque: bool, skip: int = 0) -> dict:
    """Largest momentum-balance residuals of the floating articulations against the exported forces.

    For each articulation on a free root, the change of its total linear
    momentum over a step must equal ``dt`` times gravity, body forces and the
    exported contact forces on its bodies: exactly for a free rigid body
    (``"free"``) and up to ``O(dt^2)`` for a jointed tree (``"articulated"``),
    whose momentum also moves with its pose. The angular balance about the
    centre of mass (``"angular"``) is checked for free rigid bodies when
    ``torque`` is set, against the witness points the rows act at (point
    friction; patch friction acts at its anchors). Residuals are relative to the
    weight impulse ``m g dt`` (and to ``m g dt`` times the body's radius of
    gyration for the angular one); ``None`` where no such body exists.
    """
    dt = run.scene.dt
    worst = {"free": None, "articulated": None, "angular": None}

    def keep(kind, value):
        worst[kind] = value if worst[kind] is None else max(worst[kind], value)

    for step, entry in enumerate(run.contacts):
        if step < skip:
            continue
        frame = None
        if len(entry["force"]):
            frame = contact_frame(entry, run.body_q[step], run.body_qd_solve[step], data)
        external = entry.get("body_f")
        for bodies in data.floating_bodies:
            mass = data.body_mass[bodies]
            force = np.zeros(3)
            if frame is not None:
                for side, sign in ((frame.body0, 1.0), (frame.body1, -1.0)):
                    force += sign * frame.force[np.isin(side, bodies)].sum(axis=0)
            if external is not None:
                force += external[bodies, :3].sum(axis=0)
            gravity = (mass[:, None] * data.gravity[bodies]).sum(axis=0)
            change = (mass[:, None] * (run.body_qd[step + 1, bodies, :3] - run.body_qd[step, bodies, :3])).sum(axis=0)
            scale = mass.sum() * np.linalg.norm(data.gravity[bodies[0]]) * dt
            residual = float(np.linalg.norm(change - dt * (gravity + force))) / scale
            keep("free" if bodies.size == 1 else "articulated", residual)
        if not torque:
            continue
        for body in data.free_bodies:
            mass = data.body_mass[body]
            pose = run.body_q[step, body]
            com = _transform_point(pose, data.body_com[body])
            moment = np.zeros(3)
            if frame is not None:
                for side, points, sign in ((frame.body0, frame.point0, 1.0), (frame.body1, frame.point1, -1.0)):
                    mask = side == body
                    moment += sign * np.cross(points[mask] - com, frame.force[mask]).sum(axis=0)
            if external is not None:
                moment += external[body, 3:6]
            rotation = pose[3:7]
            conj = rotation * np.array([-1.0, -1.0, -1.0, 1.0])
            inertia = data.body_inertia[body]
            omega0, omega1 = run.body_qd[step, body, 3:6], run.body_qd[step + 1, body, 3:6]
            change = _quat_rotate(rotation, inertia @ _quat_rotate(conj, omega1 - omega0))
            scale = mass * np.linalg.norm(data.gravity[body]) * dt * np.sqrt(np.trace(inertia) / mass)
            keep("angular", float(np.linalg.norm(change - dt * moment)) / scale)
    return worst


def limit_violation(run: Run, data: ModelData) -> float | None:
    """Largest violation of a finite revolute or prismatic position limit over the run."""
    if data.joint_dof_limited.size == 0:
        return None
    lower = data.joint_limit_lower[data.joint_dof_limited]
    upper = data.joint_limit_upper[data.joint_dof_limited]
    finite = (np.abs(lower) < 1.0e9) & (np.abs(upper) < 1.0e9)
    if not np.any(finite):
        return None
    q = run.joint_q[:, data.joint_dof_coord][:, finite]
    return float(np.maximum(np.maximum(q - upper[finite], lower[finite] - q), 0.0).max())


@dataclasses.dataclass
class Distance:
    """Largest differences between two runs of one scene, over the trajectory and at its end.

    Contact forces are not compared: a statically indeterminate distribution
    (the four corners of a resting box, or the lateral friction of feet that a
    planar leg cannot move sideways) is any member of a set that yields the
    same motion. The momentum balance checks their sum; scene laws check single
    loads that are determinate.
    """

    position: float
    rotation: float
    velocity: float
    angular_velocity: float
    end_position: float
    end_rotation: float
    end_velocity: float
    end_angular_velocity: float

    FIELDS = (
        "position",
        "rotation",
        "velocity",
        "angular_velocity",
        "end_position",
        "end_rotation",
        "end_velocity",
        "end_angular_velocity",
    )

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in self.FIELDS])


def distance(a: Run, b: Run) -> Distance:
    """Compare body poses and twists of two runs of the same scene."""
    position = np.linalg.norm(a.body_q[..., :3] - b.body_q[..., :3], axis=-1)
    qa, qb = a.body_q[..., 3:7], b.body_q[..., 3:7]
    # Vector part of conj(qa) * qb: robust to the float32 normalization error that arccos(qa . qb) amplifies.
    relative = qa[..., 3:4] * qb[..., :3] - qb[..., 3:4] * qa[..., :3] - np.cross(qa[..., :3], qb[..., :3])
    rotation = 2.0 * np.arcsin(np.clip(np.linalg.norm(relative, axis=-1), 0.0, 1.0))
    velocity = np.linalg.norm(a.body_qd[..., :3] - b.body_qd[..., :3], axis=-1)
    angular = np.linalg.norm(a.body_qd[..., 3:6] - b.body_qd[..., 3:6], axis=-1)
    return Distance(
        position=float(position.max()),
        rotation=float(rotation.max()),
        velocity=float(velocity.max()),
        angular_velocity=float(angular.max()),
        end_position=float(position[-1].max()),
        end_rotation=float(rotation[-1].max()),
        end_velocity=float(velocity[-1].max()),
        end_angular_velocity=float(angular[-1].max()),
    )


def loop_gap(run: Run, model_data: dict) -> float | None:
    """Largest distance between the parent and child anchors of a loop (ball) joint over the run [m]."""
    joints = model_data["loop_joints"]
    if not joints:
        return None
    worst = 0.0
    for parent, child, parent_xform, child_xform in joints:
        anchors = []
        for body, xform in ((parent, parent_xform), (child, child_xform)):
            pose = run.body_q[:, body]
            anchors.append(_transform_point(pose, np.broadcast_to(xform[:3], pose[:, :3].shape)))
        worst = max(worst, float(np.linalg.norm(anchors[0] - anchors[1], axis=-1).max()))
    return worst


def mimic_error(run: Run, model_data: dict) -> float | None:
    """Largest violation of ``q_follower = coef0 + coef1 q_leader`` over the run's second half [rad or m].

    The scenes start off the constraint; its first half lets the rows close the initial violation.
    """
    mimics = model_data["mimics"]
    if not mimics:
        return None
    q = run.joint_q[run.joint_q.shape[0] // 2 :]
    return max(
        float(np.abs(q[:, follower] - coef0 - coef1 * q[:, leader]).max()) for follower, leader, coef0, coef1 in mimics
    )


def joint_data(model) -> dict:
    """Host copies of the loop joints and mimic constraints the joint laws read."""
    joint_type = model.joint_type.numpy()
    articulation = model.joint_articulation.numpy()
    parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
    x_p, x_c = model.joint_X_p.numpy().astype(np.float64), model.joint_X_c.numpy().astype(np.float64)
    loops = [
        (int(parent[j]), int(child[j]), x_p[j], x_c[j])
        for j in range(model.joint_count)
        if joint_type[j] == int(newton.JointType.BALL) and articulation[j] < 0
    ]
    mimics = []
    if getattr(model, "constraint_mimic_count", 0):
        q_start = model.joint_q_start.numpy()
        follower, leader = model.constraint_mimic_joint0.numpy(), model.constraint_mimic_joint1.numpy()
        coef0, coef1 = model.constraint_mimic_coef0.numpy(), model.constraint_mimic_coef1.numpy()
        mimics = [
            (int(q_start[follower[i]]), int(q_start[leader[i]]), float(coef0[i]), float(coef1[i]))
            for i in range(model.constraint_mimic_count)
        ]
    return {"loop_joints": loops, "mimics": mimics}


# ---------------------------------------------------------------------------------------------------------------------
# Scene laws: analytic expectations, each a ``(name, value, bound)`` that passes when ``value <= bound``
# ---------------------------------------------------------------------------------------------------------------------


def incline_laws(run: Run) -> list:
    """A box sticks where ``mu > tan(angle)`` and otherwise slides at ``g (sin(angle) - mu cos(angle))``."""
    params, dt = run.scene.params, run.scene.dt
    angle, downhill = params["angle"], np.asarray(params["downhill"])
    start = run.scene.steps // 3
    laws = []
    for world, mu in enumerate(params["mu"]):
        along = run.body_q[:, world, :3] @ downhill
        speed = run.body_qd[:, world, :3] @ downhill
        if mu > math.tan(angle):
            laws.append((f"world {world} creep [m]", abs(along[-1] - along[start]), 1.0e-5))
        else:
            expected = scenes.GRAVITY * (math.sin(angle) - mu * math.cos(angle))
            measured = (speed[-1] - speed[start]) / ((run.scene.steps - start) * dt)
            laws.append((f"world {world} sliding acceleration error [1]", abs(measured / expected - 1.0), 2.0e-3))
    return laws


def drop_laws(run: Run) -> list:
    """At the first rebound of a ball with ``e > 0``, the normal velocity is ``-e`` times the incident one."""
    laws = []
    dt, gravity = run.scene.dt, -scenes.GRAVITY
    for world, restitution in enumerate(run.scene.params["restitution"]):
        if restitution <= 0.0:
            continue
        vz = run.body_qd[:, world, 2]
        rebound = np.flatnonzero((vz[1:] > 0.0) & (vz[:-1] < 0.0))
        if rebound.size == 0:
            laws.append((f"world {world} rebound found", 1.0, 0.0))
            continue
        step = int(rebound[0])
        incident = vz[step] + gravity * dt
        laws.append((f"world {world} rebound ratio error [1]", abs(-vz[step + 1] / incident - restitution), 1.0e-3))
    return laws


def speed_limit_laws(run: Run) -> list:
    """A free body pushed past its speed limit moves at the limit and never faster (one body per world)."""
    limit = run.scene.params["speed_limit"]
    laws = []
    for body in range(run.body_qd.shape[1]):
        speed = np.linalg.norm(run.body_qd[:, body, :3], axis=-1)
        laws.append((f"world {body} speed overshoot [1]", float(speed.max() / limit - 1.0), 1.0e-4))
        laws.append((f"world {body} final speed deficit [1]", float(1.0 - speed[-1] / limit), 1.0e-3))
    return laws


def joint_speed_limit_laws(run: Run) -> list:
    """Joints pushed past their velocity limits never exceed them."""
    limit = run.scene.params["velocity_limit"]
    return [("overshoot [1]", float(np.abs(run.joint_qd).max() / limit - 1.0), 1.0e-3)]


def spring_laws(run: Run) -> list:
    """The hinge follows the discrete oscillator of semi-implicit Euler with an explicit spring and damper."""
    params, dt = run.scene.params, run.scene.dt
    q, qd = params["start"], 0.0
    expected = [q]
    for _ in range(run.scene.steps):
        qd += dt * (-params["stiffness"] * q - params["damping"] * qd) / params["inertia"]
        q += dt * qd
        expected.append(q)
    return [("angle error [rad]", float(np.abs(run.joint_q[:, 0] - np.asarray(expected)).max()), 1.0e-4)]


def compliance_laws(run: Run) -> list:
    """A sphere on a compliant contact rests at ``r - m g / k``."""
    params = run.scene.params
    expected = params["radius"] - params["mass"] * scenes.GRAVITY / params["stiffness"]
    return [("rest height error [m]", abs(float(run.body_q[-1, 0, 2]) - expected), 1.0e-5)]


def torsion_laws(run: Run) -> list:
    """A sphere under a yaw torque below ``mu m g R`` holds; above it spins up at ``(torque - mu m g R) / I``."""
    params, dt = run.scene.params, run.scene.dt
    spin_radius = run.options["contact_torsion_radius"]
    mass, radius = params["mass"], params["radius"]
    inertia = 0.4 * mass * radius * radius
    laws = []
    start = run.scene.steps // 3
    for world, (torque, mu) in enumerate(zip(params["torques"], params["mu"], strict=True)):
        capacity = mu * mass * scenes.GRAVITY * spin_radius
        omega = run.body_qd[:, world, 5]
        if torque < capacity:
            laws.append((f"world {world} spin [rad/s]", float(np.abs(omega[start:]).max()), 1.0e-3))
        else:
            measured = (omega[-1] - omega[start]) / ((run.scene.steps - start) * dt)
            expected = (torque - capacity) / inertia
            laws.append((f"world {world} spin acceleration error [1]", abs(measured / expected - 1.0), 2.0e-2))
    return laws


def grasp_laws(run: Run) -> list:
    """Each squeezed box rises with its palm: no slip between fingers and box once lifted."""
    worlds = run.scene.params["worlds"]
    bodies = run.body_q.shape[1] // worlds  # per world: base, palm, two fingers, box
    laws = []
    for world in range(worlds):
        palm, box = world * bodies + 1, world * bodies + bodies - 1
        lift_palm = run.body_q[-1, palm, 2] - run.body_q[0, palm, 2]
        lift_box = run.body_q[-1, box, 2] - run.body_q[0, box, 2]
        laws.append((f"world {world} box lift below palm lift [m]", float(abs(lift_palm - lift_box)), 1.0e-3))
        laws.append((f"world {world} box lift shortfall [m]", float(0.9 * run.scene.params["lift"] - lift_box), 0.0))
    return laws


def keyboard_laws(run: Run) -> list:
    """At rest, the exported force of the arm's tip on key 0 balances the key's drive and weight along its axis.

    Per world, key 0 is the body after the frame on the first prismatic joint
    (axis ``+z``, the world's first coordinate, drive ``ke`` toward 0), so
    ``f_z = m g + ke q`` with ``q`` the key's coordinate at rest.
    """
    params, worlds = run.scene.params, run.scene.params["worlds"]
    shape_body = model_data(run.scene)[0].shape_body
    bodies, coordinates = run.body_q.shape[1] // worlds, run.joint_q.shape[1] // worlds
    entry = run.contacts[-1]
    laws = []
    for world in range(worlds):
        key = world * bodies + 1
        force = np.zeros(3)
        for side, sign in (("rigid_contact_shape0", 1.0), ("rigid_contact_shape1", -1.0)):
            owners = np.where(entry[side] >= 0, shape_body[np.maximum(entry[side], 0)], -1)
            force += sign * entry["force"][owners == key].sum(axis=0)
        expected = params["key_mass"] * scenes.GRAVITY + params["key_stiffness"] * run.joint_q[-1, world * coordinates]
        bound = 1.0e-3 + 1.0e-3 * abs(expected)
        laws.append((f"world {world} key 0 force balance error [N]", abs(float(force[2]) - expected), bound))
    return laws


SCENE_LAWS = {
    "incline": incline_laws,
    "ball_drop": drop_laws,
    "box_push": speed_limit_laws,
    "pushed_arm": joint_speed_limit_laws,
    "spring_hinge": spring_laws,
    "compliant_ball": compliance_laws,
    "torsion_pad": torsion_laws,
    "gripper": grasp_laws,
    "gripper_table": grasp_laws,
    "keyboard": keyboard_laws,
}
"""Scene name to its analytic laws."""


# ---------------------------------------------------------------------------------------------------------------------
# References and tolerances
# ---------------------------------------------------------------------------------------------------------------------

REFERENCE_FACTOR = 8
"""The converged reference runs ``REFERENCE_FACTOR`` times the gate budget (and one iteration more)."""

TRAJECTORY_FACTOR = 4.0
"""A route may stray from the converged reference by this multiple of the reference's own convergence spread."""

LAW_FACTOR = 4.0
"""A route's law residual may exceed the reference's residual at the same budget by this factor (plus a floor)."""

TRAJECTORY_FLOOR = Distance(
    position=2.0e-6,
    rotation=2.0e-5,
    velocity=2.0e-4,
    angular_velocity=2.0e-3,
    end_position=2.0e-6,
    end_rotation=2.0e-5,
    end_velocity=2.0e-4,
    end_angular_velocity=2.0e-3,
)
"""Float32 floor of each compared quantity [m, rad, m/s, rad/s]."""

FORCE_FLOOR = 1.0e-5
"""Floor of the force laws (negative normal force, cone excess), relative to the model's weight [1]."""
VELOCITY_FLOOR = 2.0e-5
"""Floor of the natural residuals [m/s]."""
PENETRATION_FLOOR = 1.0e-5
"""Floor of the penetration bound [m]."""
ENERGY_FLOOR = 1.0e-5
"""Floor of the energy gain of a passive scene, relative to the model's weight times 1 m [1]."""
LIMIT_FLOOR = 1.0e-4
"""Floor of the joint-limit, loop-closure and mimic violations [m or rad]."""
MOMENTUM_BOUND = {"free": 1.0e-3, "articulated": 1.0e-3, "angular": 1.0e-3}
"""Momentum-balance floor relative to ``m g dt``. A free rigid body balances exactly; a jointed tree up to the
``O(dt^2)`` motion of its momentum with its pose, which the reference's balance at the same budget bounds."""


@dataclasses.dataclass(frozen=True)
class Physics:
    """One physical problem: a scene, a friction law and the options that change its laws."""

    scene: scenes.Scene
    law: str | None
    """``"point"``, ``"patch"`` or ``None`` for a scene without contact."""
    iterations: int
    """Gate budget N of every route."""
    variant: tuple = ()
    """``(option, value)`` pairs every route of this problem shares (velocity iterations, compliance...)."""
    label: str = ""

    @property
    def name(self) -> str:
        parts = [self.scene.name, self.law or "nofriction"]
        if self.label:
            parts.append(self.label)
        return "_".join(parts)

    def options(self, iterations: int) -> dict:
        options = dict(self.variant)
        if self.law is not None:
            options["friction_anchor_beta"] = 0.0 if self.law == "point" else 0.2
        options["pgs_iterations"] = iterations
        return options


_RUNS: dict = {}
_MODEL_DATA: dict = {}


def model_data(scene: scenes.Scene) -> tuple[ModelData, dict, float]:
    """Host model data, joint data and total mass [kg] of a scene (built once on the CPU)."""
    if scene.name not in _MODEL_DATA:
        model = scene.build("cpu")
        _MODEL_DATA[scene.name] = (ModelData.of(model), joint_data(model), float(model.body_mass.numpy().sum()))
    return _MODEL_DATA[scene.name]


def run_physics(physics: Physics, route: Route, device, iterations: int | None = None) -> Run:
    """Run (once per process) ``physics`` through ``route`` at ``iterations`` (default: the gate budget)."""
    iterations = physics.iterations if iterations is None else iterations
    key = (physics.name, route.name, str(device), iterations)
    if key not in _RUNS:
        _RUNS[key] = simulate(physics.scene, route, device, options=physics.options(iterations))
    return _RUNS[key]


@dataclasses.dataclass
class Reference:
    """The converged reference of a problem on one device, and the tolerances its own convergence implies."""

    converged: Run
    budget: Run
    spread: np.ndarray
    tolerance: np.ndarray
    laws: Laws
    momentum: dict
    limit: float | None
    loop: float | None
    mimic: float | None


def reference(physics: Physics, route: Route, device) -> Reference:
    """Run the reference route at the gate budget and converged, and derive the trajectory tolerance.

    On CUDA the budget run repeats: the broad phase, the narrow phase and
    FeatherPGS allocate pairs, contacts and rows with atomics, so the row
    order, and with it a Gauss-Seidel iterate, follows the warps from run to
    run. Where the contact problem is determinate the iterate stays within the
    budget's convergence band, which the spread then includes. A statically
    indeterminate support lets the order pick one of many rigid solutions,
    which more iterations do not remove and one repeat cannot bound, so such
    a scene sorts its contacts (``deterministic_contacts``) and repeats
    bitwise, as FeatherPGS then gives the rows the contacts' order.
    """
    high = REFERENCE_FACTOR * physics.iterations
    converged = run_physics(physics, route, device, high)
    parity = run_physics(physics, route, device, high + 1)
    budget = run_physics(physics, route, device)
    spread = np.maximum(distance(budget, converged).as_array(), distance(parity, converged).as_array())
    if wp.get_device(device).is_cuda:
        repeat = run_physics(physics, dataclasses.replace(route, name=f"{route.name}_repeat"), device)
        spread = np.maximum(spread, distance(repeat, budget).as_array())
    if physics.scene.deterministic_contacts:
        # The support's rounding sensitivity: the run at N with the time step one float32 ulp off either way.
        for toward in (np.inf, -np.inf):
            dt = float(np.nextafter(np.float32(physics.scene.dt), np.float32(toward)))
            nudged = dataclasses.replace(physics, scene=dataclasses.replace(physics.scene, dt=dt))
            rounding = run_physics(nudged, dataclasses.replace(route, name=f"{route.name}_dt{dt!r}"), device)
            spread = np.maximum(spread, distance(rounding, budget).as_array())
    # A pose at rest under static friction is one of a continuum: a perturbation of the transient leaves an offset
    # of its own size, so the end pose is held to the transient's spread. A rest velocity is unique (zero).
    fields = Distance.FIELDS
    for name in ("position", "rotation"):
        end = fields.index(f"end_{name}")
        spread[end] = max(spread[end], spread[fields.index(name)])
    tolerance = TRAJECTORY_FACTOR * spread + TRAJECTORY_FLOOR.as_array()
    data, joints, _ = model_data(physics.scene)
    return Reference(
        converged=converged,
        budget=budget,
        spread=spread,
        tolerance=tolerance,
        laws=contact_laws(budget, data, patch=physics.law == "patch"),
        momentum=momentum_balance(budget, data, torque=physics.law == "point"),
        limit=limit_violation(budget, data),
        loop=loop_gap(budget, joints),
        mimic=mimic_error(budget, joints),
    )


# ---------------------------------------------------------------------------------------------------------------------
# Route catalog
# ---------------------------------------------------------------------------------------------------------------------

_MF = {}
# Dense storage and ordinary dynamics: the route of the general placement's reference.
_ORDINARY = {"storage": "dense", "evaluator": "ordinary"}


REFERENCE_CUDA = Route(
    "reference",
    "cuda",
    # The plain matrix-free Gauss-Seidel sweep over ordinary dynamics and rows: the general placement declines the
    # local, paired and sparse-diagonal owners, and every world is solved (no resolved-world claim).
    options=_MF,
    pins={**_ORDINARY, "hinv_jt_kernel": "tiled", "placement": "general", "claims": ()},
    expect={
        "sweep.route": "matrix_free_gs",
        "sweep.tiers": None,
        "local_solve.enabled": False,
        "paired_response.enabled": False,
        "sparse_diagonal.enabled": False,
        "held_tree.owner": None,
        "resolved_worlds.enabled": False,
    },
)
CPU_SPLIT = Route(
    "cpu_split",
    "cpu",
    options={},
    expect={"sweep.route": "dense_loop", "held_tree.owner": None, "contact_rows.producer": "general"},
)
CPU_SPLIT_UNIT = CPU_SPLIT.but(
    "cpu_split_unit", options={"contact_update": "unit"}, expect={"sweep.contact_update.law": "unit"}
)
MF_DEFAULT = Route("mf_default", "cuda", options=_MF, expect={"sweep.route": "matrix_free_gs"})
MF_GENERAL = Route(
    "mf_general",
    "cuda",
    # The clutter scenes' route: the matrix-free sweep over ordinary dynamics.
    options=_MF,
    pins=_ORDINARY,
    expect={"sweep.route": "matrix_free_gs", "sweep.tiers": None, "held_tree.owner": None},
)
MF_HELD = Route(
    "mf_held",
    "cuda",
    options=_MF,
    pins={"storage": "dense"},
    expect={
        "sweep.route": "matrix_free_gs",
        "sweep.tiers": None,
        "held_tree.owner": "mixed",
        "held_tree.groups.0.operator": "TreeHeldOperator",
    },
)
MF_PARALLEL = Route(
    "mf_parallel",
    "cuda",
    # The Jacobi tiers that a test pin builds on any class whose response they form, over the group rows.
    options=_MF,
    pins={"tiers": 32, "producer": "general"},
    expect={"sweep.route": "matrix_free_gs", "sweep.tiers.family": "parallel"},
)
# The allegro recipe: the tiers take keyed rows on a model they fit (the legged scene).
KEYED_PARALLEL = Route(
    "keyed_parallel",
    "cuda",
    options=_MF,
    pins={"tiers": 32},
    expect={
        "sweep.route": "matrix_free_gs",
        "sweep.tiers.family": "keyed_parallel",
        "contact_rows.producer": "keyed_rows",
        "held_tree.owner": "mixed",
    },
)
SPARSE = Route(
    "sparse",
    "cuda",
    # The unfused sparse sweep, which models without the row support table run (g1).
    options=_MF,
    pins={"contact_units": None},
    expect={
        "sweep.route": "sparse_pgs",
        "contact_rows.producer": "sparse_contact_response",
        "held_tree.owner": "tree",
        "held_tree.groups.0.operator": "TreeSparseOperator",
        "sparse_mass_matrix.contact_triplets": False,
    },
)
SPARSE_TRIPLETS = Route(
    "sparse_triplets",
    "cuda",
    # The default sparse route of a single tree with the row support table: fused contact triplets.
    options=_MF,
    expect={"sweep.route": "sparse_pgs", "sparse_mass_matrix.contact_triplets": True},
)
LOCAL_PACKETS = Route(
    "local_packets",
    "cuda",
    options=_MF,
    expect={
        "local_solve.enabled": True,
        "sweep.local.factory": "row_packets",
        "contact_rows.producer": "local_packets",
        "held_tree.owner": "mixed",
    },
)
LOCAL_FOREST = Route(
    "local_packets_forest",
    "cuda",
    # The franka recipe's route: local packets over one held forest per world.
    options=_MF,
    pins={"grouping": "forest"},
    expect={
        "local_solve.enabled": True,
        "contact_rows.producer": "local_packets",
        "held_tree.owner": "forest",
        "held_tree.groups.0.operator": "ForestHeldOperator",
    },
)
LOCAL_KINEMATIC = Route(
    "local_packets_kinematic_table",
    "cuda",
    # A kinematic table (a prescribed root) makes one held forest per world the default, as on the franka task.
    options=_MF,
    expect={
        "local_solve.enabled": True,
        "contact_rows.producer": "local_packets",
        "held_tree.owner": "forest",
        "held_tree.groups.0.operator": "ForestHeldOperator",
    },
)
LOCAL_OWNED = Route(
    "local_owned",
    "cuda",
    options=_MF,
    pins={"producer": "compact_jacobian"},
    expect={
        "local_solve.enabled": True,
        "sweep.local.factory": "local_owned",
        "contact_rows.producer": "compact_jacobian",
        "held_tree.owner": "mixed",
    },
)
LOCAL_ORDINARY = Route(
    "local_owned_ordinary",
    "cuda",
    # The clutter scenes' route: local owners over ordinary dynamics.
    options=_MF,
    pins={"producer": "compact_jacobian", "evaluator": "ordinary"},
    expect={"local_solve.enabled": True, "sweep.local.factory": "local_owned", "held_tree.owner": None},
)
PAIRED = Route(
    "paired",
    "cuda",
    options=_MF,
    expect={"paired_response.enabled": True, "paired_response.factor_coordinates": True, "held_tree.owner": "mixed"},
)
PAIRED_FOREST = Route(
    "paired_forest",
    "cuda",
    # The kuka recipes' route: the paired factor sweep over one held forest per world.
    options=_MF,
    pins={"grouping": "forest"},
    expect={
        "paired_response.enabled": True,
        "paired_response.factor_coordinates": True,
        "held_tree.owner": "forest",
        "contact_rows.producer": "general",
    },
)


def _unsupported(solver) -> bool:
    return False


# The sparse-diagonal producers write the rows of a dense articulation other than six coordinates (franka-cabinet's
# nine); on the keyboard the compact boundary would own them, so these routes withhold its admission.
_SPARSE_DIAGONAL_ROWS = {"compact_contact.supported": _unsupported}
SPARSE_DIAGONAL = Route(
    "sparse_diagonal_scalar",
    "cuda",
    options=_MF,
    module=_SPARSE_DIAGONAL_ROWS,
    expect={
        "sweep.route": "sparse_diagonal",
        "contact_rows.producer": "sparse_diagonal",
        "held_tree.scalar_branches.enabled": True,
    },
)
SPARSE_DIAGONAL_TREE = Route(
    "sparse_diagonal_tree",
    "cuda",
    # The franka-cabinet route: sparse-diagonal rows with the diagonal-mass group in a held tree.
    options=_MF,
    pins={"star": "held"},
    module=_SPARSE_DIAGONAL_ROWS,
    expect={
        "sweep.route": "sparse_diagonal",
        "contact_rows.producer": "sparse_diagonal",
        "held_tree.scalar_branches.enabled": False,
        "held_tree.groups.0.operator": "TreeHeldOperator",
    },
)
COMPACT_BOUNDARY = Route(
    "compact_contact_boundary",
    "cuda",
    # The keyboard recipe's route: the compact boundary owns a six-coordinate dense articulation's rows.
    options=_MF,
    expect={
        "sweep.route": "sparse_diagonal",
        "contact_rows.producer": "compact_contact_boundary",
        "held_tree.scalar_branches.enabled": True,
    },
)


# ---------------------------------------------------------------------------------------------------------------------
# Catalog: problems, their reference and the routes that serve them
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Group:
    """One physical problem on one device: its reference route and the routes compared against it."""

    physics: Physics
    reference: Route
    routes: tuple
    tag: str = ""
    """Distinguishes groups that share a problem (and its reference runs) but compare other routes."""
    findings: tuple = ()
    """``(route name, finding)`` of routes that fail a check on this problem at the gate's baseline."""

    @property
    def name(self) -> str:
        return f"{self.physics.name}_{self.tag}" if self.tag else self.physics.name

    @property
    def device(self) -> str:
        return self.reference.device


def _physics(scene, law, iterations, label="", **variant):
    return Physics(scene, law, iterations, tuple(sorted(variant.items())), label)


def _laws(*laws):
    return laws or ("point", "patch")


_WARM = {"pgs_warmstart": True}


def cpu_groups() -> list:
    """Problems of the CPU subset: split mode, its unit contact update and warm start."""
    groups = []
    for law in _laws():
        groups += [
            Group(_physics(scenes.box_drop(), law, 32), CPU_SPLIT, (CPU_SPLIT, CPU_SPLIT_UNIT)),
            Group(_physics(scenes.box_stack(), law, 32), CPU_SPLIT, (CPU_SPLIT,)),
            Group(_physics(scenes.incline(), law, 32), CPU_SPLIT, (CPU_SPLIT, CPU_SPLIT_UNIT)),
            Group(_physics(scenes.heightfield(), law, 32), CPU_SPLIT, (CPU_SPLIT,)),
            Group(_physics(scenes.legged(), law, 32), CPU_SPLIT, (CPU_SPLIT,)),
            Group(_physics(scenes.gripper(), law, 16), CPU_SPLIT, (CPU_SPLIT, CPU_SPLIT_UNIT)),
            Group(_physics(scenes.gripper(table=True), law, 16), CPU_SPLIT, (CPU_SPLIT,)),
        ]
    groups += [
        Group(
            _physics(scenes.box_stack(contact_matching="latest"), "point", 32),
            CPU_SPLIT,
            (CPU_SPLIT.but("cpu_split_warm_start", options=_WARM),),
        ),
        Group(_physics(scenes.ball_drop(), "point", 16), CPU_SPLIT, (CPU_SPLIT,)),
        Group(_physics(scenes.box_push(), "point", 16), CPU_SPLIT, (CPU_SPLIT,)),
        Group(_physics(scenes.keyboard(), "point", 16), CPU_SPLIT, (CPU_SPLIT,)),
        Group(_physics(scenes.hand(), "point", 16), CPU_SPLIT, (CPU_SPLIT,)),
        Group(_physics(scenes.sprawl(), "point", 64), CPU_SPLIT, (CPU_SPLIT,)),
        Group(_physics(scenes.long_chain(), "point", 64), CPU_SPLIT, (CPU_SPLIT,)),
        Group(_physics(scenes.long_chain(armature=0.0), "point", 64), CPU_SPLIT, (CPU_SPLIT,)),
    ]
    for scene, iterations in (
        (scenes.chain(), 16),
        (scenes.four_bar(), 16),
        (scenes.mimic_pair(), 16),
        (scenes.spring_hinge(), 8),
    ):
        groups.append(Group(_physics(scene, None, iterations), CPU_SPLIT, (CPU_SPLIT,)))
    return groups


def cuda_groups() -> list:
    """Problems of the CUDA matrix: every route each scene admits, against the CUDA reference."""
    ref = REFERENCE_CUDA
    vlimits = {"enable_joint_velocity_limits": True}
    groups = []
    for law in _laws():
        # Patch friction keeps the packed full sweep of the legged scene's small worlds: only point friction fuses
        # its contact triplets.
        triplets = [SPARSE_TRIPLETS] if law == "point" else []
        legged = [SPARSE, *triplets, MF_HELD, MF_GENERAL]
        legged += [MF_PARALLEL, KEYED_PARALLEL]
        groups += [
            Group(
                _physics(scenes.box_drop(), law, 32),
                ref,
                (MF_HELD, MF_GENERAL),
            ),
            Group(_physics(scenes.box_stack(), law, 32), ref, (MF_HELD, MF_GENERAL)),
            Group(_physics(scenes.incline(), law, 32), ref, (MF_HELD, MF_GENERAL)),
            Group(_physics(scenes.heightfield(), law, 32), ref, (MF_HELD,)),
            Group(_physics(scenes.legged(), law, 32), ref, tuple(legged)),
            # A grasp drifts with the row order at a small budget: the scene sorts its contacts and runs 32 iterations.
            Group(
                _physics(scenes.gripper(), law, 32),
                ref,
                (LOCAL_PACKETS, LOCAL_FOREST, LOCAL_OWNED, LOCAL_ORDINARY),
            ),
            Group(_physics(scenes.gripper(table=True), law, 32), ref, (LOCAL_KINEMATIC,)),
            Group(
                _physics(scenes.keyboard(), law, 16),
                ref,
                (SPARSE_DIAGONAL, SPARSE_DIAGONAL_TREE, COMPACT_BOUNDARY),
            ),
            Group(_physics(scenes.hand(), law, 16), ref, (PAIRED, PAIRED_FOREST)),
            # The tiers form the response in-kernel, which bounds an articulation at 32 coordinates: the sprawl (66)
            # and the snake (106) take no tier, even pinned.
            Group(_physics(scenes.sprawl(), law, 64), ref, (MF_DEFAULT,)),
            Group(_physics(scenes.long_chain(), law, 64), ref, (MF_DEFAULT,)),
            # Without armature the snake's mass matrix is beyond float32 factorization: the solver adds the
            # armature it needs near the head, the same on every route.
            Group(_physics(scenes.long_chain(armature=0.0), law, 64), ref, (MF_DEFAULT,)),
            Group(
                _physics(scenes.torsion_pad(), law, 32, "torsion", contact_torsion_radius=0.05),
                ref,
                (
                    MF_DEFAULT.but(
                        "mf_torsion", pins={"torsion_preparation": "host"}, expect={"friction.torsion": True}
                    ),
                    MF_DEFAULT.but("mf_torsion_device"),
                ),
            ),
        ]
    groups += [
        # Warm start solves the cold problem from another first iterate: compare it with the cold reference.
        Group(
            _physics(scenes.box_stack(contact_matching="latest"), "point", 32),
            ref,
            (MF_HELD.but("mf_held_warm_start", options=_WARM),),
        ),
        Group(
            _physics(scenes.legged(contact_matching="latest"), "point", 32),
            ref,
            (SPARSE.but("sparse_warm_start", options=_WARM),),
        ),
        Group(_physics(scenes.ball_drop(), "point", 16), ref, (MF_HELD, MF_GENERAL)),
        Group(_physics(scenes.box_push(), "point", 16), ref, (MF_HELD,)),
        Group(_physics(scenes.chain(), None, 16), ref, (MF_HELD, MF_GENERAL, MF_PARALLEL)),
        Group(
            _physics(scenes.chain(), None, 16, "physx_drives", drive_mode="physx_pgs"),
            ref,
            (MF_DEFAULT.but("mf_physx_drives", expect={"held_tree.owner": None}),),
        ),
        Group(_physics(scenes.four_bar(), None, 16), ref, (MF_DEFAULT,)),
        Group(
            # Pre-elimination solves the iterative connect rows' problem exactly: compare with the iterative reference.
            _physics(scenes.four_bar(), None, 16),
            ref,
            (
                MF_DEFAULT.but(
                    "mf_preelimination",
                    options={"enable_bilateral_preelimination": True},
                    expect={"dynamics.preelimination": True},
                ),
            ),
            tag="preelimination",
        ),
        Group(_physics(scenes.mimic_pair(), None, 16), ref, (MF_HELD,)),
        Group(
            _physics(scenes.mimic_pair(), None, 16, "velocity_limits", **vlimits),
            ref,
            (SPARSE.but("sparse_velocity_mimic", expect={"sparse_mass_matrix.velocity_limits": True}),),
        ),
        # A one-coordinate arm is diagonal-mass, which the held tree declines.
        Group(_physics(scenes.spring_hinge(), None, 8), ref, (MF_GENERAL,)),
        Group(
            _physics(scenes.pushed_arm(), None, 16, "stateless", **vlimits),
            ref,
            (MF_HELD, SPARSE.but("sparse_velocity_limits", expect={"sparse_mass_matrix.velocity_limits": True})),
        ),
        Group(
            _physics(scenes.pushed_arm(), None, 16, "accumulated", velocity_limit_law="accumulated", **vlimits),
            ref,
            (
                MF_HELD,
                SPARSE.but("sparse_velocity_limits", expect={"sparse_mass_matrix.velocity_limits": True}),
                MF_PARALLEL,
            ),
        ),
        Group(
            _physics(scenes.box_drop(), "point", 32, "velocity_iterations", pgs_velocity_iterations=4),
            ref,
            (MF_GENERAL,),
        ),
        Group(
            _physics(
                scenes.compliant_ball(),
                "point",
                16,
                "compliance",
                contact_compliance=True,
                enable_restitution=False,
            ),
            ref,
            (MF_DEFAULT.but("mf_compliance", expect={"sweep.driver": "contact_compliance"}),),
        ),
    ]
    return groups


# ---------------------------------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------------------------------


def _record_checks(route: Route, routes: dict) -> list:
    rows = []
    for path, expected in route.expect.items():
        actual = route_value(routes, path)
        ok = bool(expected(actual)) if callable(expected) else actual == expected
        label = getattr(expected, "__name__", repr(expected)) if callable(expected) else repr(expected)
        rows.append((f"route {path}={actual!r}, expected {label}", 0.0 if ok else 1.0, 0.0))
    return rows


def _energy_gain(run: Run, data: ModelData) -> float:
    energy = mechanical_energy(run.body_q, run.body_qd, data)
    return float(max((energy - energy[0]).max(), 0.0))


def run_checks(group: Group, route: Route, device) -> list:
    """Every check of one route's run: ``(name, value, bound)``, passing when ``value <= bound``."""
    physics = group.physics
    start = time.perf_counter()
    ref = reference(physics, group.reference, device)
    run = run_physics(physics, route, device)
    elapsed = time.perf_counter() - start
    data, joints, mass = model_data(physics.scene)
    weight = mass * scenes.GRAVITY
    patch = physics.law == "patch"
    rows = _record_checks(route, run.routes)
    lost = sorted(name for name, flag in run.capacity.items() if flag)
    rows.append((f"capacity overflow {lost}", float(len(lost)), 0.0))
    finite = all(np.isfinite(array).all() for array in (run.body_q, run.body_qd, run.joint_q, run.joint_qd))
    rows.append(("non-finite state", 0.0 if finite else 1.0, 0.0))
    if not finite:
        return rows
    laws = contact_laws(run, data, patch=patch)
    if laws.normal_force_min is not None:
        rows.append(("negative normal force [N]", -laws.normal_force_min, FORCE_FLOOR * weight))
        # Point friction projects each contact's pair on its own final normal impulse: every iterate is inside
        # the cone. A patch ring pools normals that later updates of the last sweep still change, so its cone
        # holds at convergence: the reference's excess at the same budget bounds it.
        rows.append(
            (
                "friction cone excess [N]",
                laws.cone_excess,
                LAW_FACTOR * max(ref.laws.cone_excess or 0.0, 0.0) + FORCE_FLOOR * weight,
            )
        )
        rows.append(
            (
                "normal law residual [m/s]",
                laws.normal_residual or 0.0,
                LAW_FACTOR * (ref.laws.normal_residual or 0.0) + VELOCITY_FLOOR,
            )
        )
        if laws.friction_residual is not None:
            rows.append(
                (
                    "friction law residual [m/s]",
                    laws.friction_residual,
                    LAW_FACTOR * (ref.laws.friction_residual or 0.0) + VELOCITY_FLOOR,
                )
            )
        if laws.penetration is not None:
            rows.append(
                (
                    "penetration [m]",
                    laws.penetration,
                    LAW_FACTOR * (ref.laws.penetration or 0.0) + PENETRATION_FLOOR,
                )
            )
    if physics.scene.passive:
        rows.append(
            (
                "energy gain [J]",
                _energy_gain(run, data),
                LAW_FACTOR * _energy_gain(ref.budget, data) + ENERGY_FLOOR * weight,
            )
        )
    # A rigid-body speed clamp removes momentum outside the contact rows, which the export does not report.
    if "rigid_body_velocity_limits" not in physics.scene.tags:
        balance = momentum_balance(run, data, torque=physics.law == "point")
        for kind, value in balance.items():
            if value is not None:
                relative = LAW_FACTOR * (ref.momentum[kind] or 0.0) if kind == "articulated" else 0.0
                rows.append((f"momentum balance ({kind}) [m g dt]", value, relative + MOMENTUM_BOUND[kind]))
    for name, value, bound in (
        ("joint limit violation", limit_violation(run, data), ref.limit),
        ("loop closure gap [m]", loop_gap(run, joints), ref.loop),
        ("mimic error", mimic_error(run, joints), ref.mimic),
    ):
        if value is not None:
            rows.append((name, value, LAW_FACTOR * (bound or 0.0) + LIMIT_FLOOR))
    scene_laws = SCENE_LAWS.get(physics.scene.name)
    if scene_laws is not None:
        rows += [(f"scene law: {name}", value, bound) for name, value, bound in scene_laws(run)]
    rows += [
        (f"distance to converged reference: {field}", value, bound)
        for field, value, bound in zip(
            Distance.FIELDS, distance(run, ref.converged).as_array(), ref.tolerance, strict=True
        )
    ]
    _report(group, route, device, rows, elapsed)
    return rows


def _report(group: Group, route: Route, device, rows: list, seconds: float) -> None:
    """Append a run's checks to the JSON-lines file ``FPGS_CONFORMANCE_REPORT`` names, if set."""
    path = os.environ.get("FPGS_CONFORMANCE_REPORT")
    if not path:
        return
    import json  # noqa: PLC0415

    entry = {
        "group": group.name,
        "route": route.name,
        "device": str(device),
        "seconds": round(seconds, 3),
        "checks": [[name, float(value), float(bound)] for name, value, bound in rows],
    }
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry) + "\n")


def _failures(rows: list) -> list:
    return [(name, value, bound) for name, value, bound in rows if not value <= bound]


def _message(rows: list) -> str:
    return "\n".join(f"  {name}: {value:.4g} > {bound:.4g}" for name, value, bound in rows)


# ---------------------------------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------------------------------


def _cuda_device():
    return wp.get_device("cuda:0") if wp.is_cuda_available() else None


class _ConformanceCase(unittest.TestCase):
    device = None

    def setUp(self):
        if self.device is None:
            self.skipTest("No CUDA device")

    def check_route(self, group: Group, route: Route):
        failures = _failures(run_checks(group, route, self.device))
        self.assertFalse(failures, f"{group.name} through {route.name}:\n{_message(failures)}")

    def check_agreement(self, group: Group, routes: tuple):
        ref = reference(group.physics, group.reference, self.device)
        runs = [run_physics(group.physics, route, self.device) for route in routes]
        runs.append(ref.budget)
        rows = []
        for first in range(len(runs)):
            for second in range(first + 1, len(runs)):
                pair = f"{runs[first].route.name} vs {runs[second].route.name}"
                gap = distance(runs[first], runs[second]).as_array()
                rows += [
                    (f"{pair}: {field}", value, 2.0 * bound)
                    for field, value, bound in zip(Distance.FIELDS, gap, ref.tolerance, strict=True)
                ]
        failures = _failures(rows)
        self.assertFalse(failures, f"{group.name} routes disagree:\n{_message(failures)}")


def _add_group_tests(cls, groups: list) -> None:
    for group in groups:
        findings = dict(group.findings)
        for route in group.routes:

            def test(self, group=group, route=route):
                self.check_route(group, route)

            test.__doc__ = f"Check {group.name} through the {route.name} route: route record, laws and trajectory."
            if route.name in findings:
                test.__doc__ += f"\n\n        Known failure at the gate's baseline: {findings[route.name]}"
                test = unittest.expectedFailure(test)
            setattr(cls, f"test_{group.name}_{route.name}", test)
        # Routes with a known failure stay out of the agreement of the others with each other and the reference.
        agreeing = tuple(route for route in group.routes if route.name not in findings)
        if agreeing and group.reference.name not in findings:

            def agreement(self, group=group, routes=agreeing):
                self.check_agreement(group, routes)

            agreement.__doc__ = f"Check that the routes of {group.name} and its reference agree with each other."
            setattr(cls, f"test_{group.name}_routes_agree", agreement)


class TestFeatherPGSConformanceCPU(_ConformanceCase):
    """Split mode on CPU against its converged reference."""

    device = wp.get_device("cpu")


class TestFeatherPGSConformanceCUDA(_ConformanceCase):
    """Every CUDA route against the converged CUDA reference, which must match the CPU reference."""

    device = _cuda_device()


_add_group_tests(TestFeatherPGSConformanceCPU, cpu_groups())
_add_group_tests(TestFeatherPGSConformanceCUDA, cuda_groups())


OWNERSHIP = (
    # (scene, law, iterations, route, owners that together solve every world and each solve one, counted owners that
    # must solve none). A sweep without counters (the parallel tiers) owns a world when no counted sweep solves it.
    (scenes.legged, "point", 32, MF_HELD, ("mf_gs",), ()),
    (scenes.legged, "point", 32, SPARSE, ("sparse_pgs",), ("mf_gs",)),
    (scenes.legged, "point", 32, MF_PARALLEL, (), ("mf_gs",)),
    (scenes.legged, "patch", 32, KEYED_PARALLEL, (), ("mf_gs",)),
    (scenes.gripper, "point", 32, LOCAL_PACKETS, ("local_pair|local_residual|local_internal",), ()),
    (scenes.gripper, "patch", 32, LOCAL_OWNED, ("local_pair|local_residual|local_internal",), ()),
    # The paired factor sweep solves worlds of independent components; coupled worlds take the mixed fallback.
    (scenes.hand, "point", 16, PAIRED, ("paired_factor", "mf_gs"), ()),
    (scenes.keyboard, "point", 16, SPARSE_DIAGONAL, ("sparse_diagonal",), ("mf_gs",)),
    (scenes.keyboard, "patch", 16, COMPACT_BOUNDARY, ("sparse_diagonal",), ("mf_gs",)),
)
"""Routes whose owners claim worlds at run time, and the sweeps the solve readout must see serve them."""


def _add_ownership_tests() -> None:
    for factory, law, iterations, route, owners, absent in OWNERSHIP:
        physics = _physics(factory(), law, iterations)

        def test(self, physics=physics, route=route, owners=owners, absent=absent):
            counted = route.but(f"{route.name}_readout", options={"pgs_solve_readout": "sweeps"})
            run = run_physics(physics, counted, self.device)
            failures = _failures(_record_checks(route, run.routes))
            self.assertFalse(failures, _message(failures))
            worlds = run.scene.params.get("worlds", len(run.scene.params.get("mu", [None])))
            slots = run.measurements["solve_readout"]["slots"]
            solved = {}
            for slot in slots:
                solved[slot["route"]] = max(solved.get(slot["route"], 0), slot["worlds"] if slot["solves"] else 0)
            served = [max(solved.get(label, 0) for label in owner.split("|")) for owner in owners]
            for owner, count in zip(owners, served, strict=True):
                self.assertGreater(count, 0, f"{owner} solved no world: {solved}")
            if owners:
                self.assertGreaterEqual(sum(served), worlds, f"{owners} solved {served} of {worlds} worlds: {solved}")
            for owner in absent:
                self.assertEqual(solved.get(owner, 0), 0, f"{owner} solved worlds its route's owner claims: {solved}")

        test.__doc__ = f"Check with the solve readout that {route.name}'s owner solves the worlds of {physics.name}."
        setattr(TestFeatherPGSConformanceCUDA, f"test_{physics.name}_{route.name}_owns_its_worlds", test)


_add_ownership_tests()


def _add_reference_tests() -> None:
    cpu = {group.name: group for group in cpu_groups()}
    for group in cuda_groups():
        twin = cpu.get(group.name)
        if twin is None or group.tag:
            continue

        def test(self, group=group, twin=twin):
            cuda = reference(group.physics, group.reference, self.device)
            host = reference(twin.physics, twin.reference, wp.get_device("cpu"))
            gap = distance(cuda.converged, host.converged).as_array()
            rows = list(zip(Distance.FIELDS, gap, cuda.tolerance + host.tolerance, strict=True))
            failures = _failures(rows)
            self.assertFalse(failures, f"CUDA and CPU references of {group.name} differ:\n{_message(failures)}")

        test.__doc__ = f"Check that the converged CUDA and CPU references of {group.name} agree."
        finding = dict(group.findings).get("reference_matches_cpu")
        if finding is not None:
            test.__doc__ += f"\n\n        Known failure at the gate's baseline: {finding}"
            test = unittest.expectedFailure(test)
        setattr(TestFeatherPGSConformanceCUDA, f"test_{group.name}_reference_matches_cpu", test)


_add_reference_tests()


if __name__ == "__main__":
    unittest.main(verbosity=2)
