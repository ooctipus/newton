# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Small deterministic FeatherPGS scenes built in code, shared by the solver tests.

Each :class:`Scene` holds a model builder, its time step and step count, the
control it applies each step and the physical parameters that analytic checks
read (friction coefficients, incline angle, restitution, stiffness). The scenes
need no Isaac Lab asset and are small enough to step on CPU. Every builder
places its bodies a few millimetres off their supports, so the first steps
also exercise impact and settling.

Scenes by topology:

- free rigid bodies: :func:`box_drop`, :func:`box_stack`, :func:`incline`,
  :func:`ball_drop`, :func:`heightfield`, :func:`box_push`;
- fixed-base articulations without contact: :func:`chain`, :func:`four_bar`,
  :func:`mimic_pair`, :func:`spring_hinge`, :func:`pushed_arm`;
- articulations in contact: :func:`legged` (a free-floating base on legs),
  :func:`gripper` (an articulation holding a free box), :func:`keyboard` (a
  diagonal-mass articulation pressed by a six-coordinate arm), :func:`hand`
  (a 17-coordinate articulation pressing a free cube), :func:`torsion_pad`
  and :func:`compliant_ball`;
- articulations above the held-tree owner's cap of 64 bodies and 64
  coordinates: :func:`sprawl` (a humanoid-like tree with two hands, 66
  coordinates) and :func:`long_chain` (a 100-link snake), both resting on the
  ground through one contact per link.

Fixed-base articulations hang from a shapeless base welded to the world, as
robot descriptions root them; articulation shapes collide with everything but
their own articulation (collision group :data:`SELF`).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.route_table import Laws

GRAVITY = 9.81
"""Magnitude of the default gravity [m/s²]."""

RIGID_GAP = 0.01
"""Contact detection gap of a shape [m]; a pair detects contacts within the sum of its shapes' gaps. It exceeds a
step's travel in every scene but the drop, whose shapes widen it, and bounds the speculative rows."""

SELF = -1
"""Collision group of an articulation's shapes: they collide with everything except each other."""


def plain_laws(**changes) -> Laws:
    """The row laws of a CUDA matrix-free solver with joint limits and point friction (the plain sweep), with
    ``changes``: the laws a stand-in solver of the admission tests carries."""
    laws = {
        "contact": "point",
        "regularized": False,
        "torsion": False,
        "compliance": False,
        "limits": True,
        "velocity_limits": False,
        "drives": "augmented",
        "schedule": "interleaved",
        "velocity_pass": False,
        "warm_start": False,
        "debug_rows": False,
        "preelimination": False,
        "mimic": False,
        "connect": False,
        "loops": False,
        "iterations": True,
        "forward_cuda": True,
    }
    return Laws(**{**laws, **changes})


@dataclasses.dataclass(frozen=True)
class Scene:
    """One deterministic scene: how to build it, how long to step it and what its checks assume."""

    name: str
    """Unique scene name."""
    build: Callable[[str], newton.Model]
    """Return the finalized model on a device."""
    dt: float
    """Step size [s]."""
    steps: int
    """Number of steps."""
    params: dict = dataclasses.field(default_factory=dict)
    """Physical parameters of the analytic checks, per world where a list."""
    solver_options: dict = dataclasses.field(default_factory=dict)
    """Solver options every route of the scene shares (part of the scene's physics)."""
    control: Callable[[newton.Model, newton.Control, newton.State, int], None] | None = None
    """Write a step's control and body forces before it is taken (after ``clear_forces``); ``None`` keeps the
    model's targets."""
    rigid_contact_max: int | None = None
    """Contact capacity of the collision pipeline; ``None`` uses the pipeline's estimate."""
    contact_matching: str | None = None
    """Contact matching mode of the collision pipeline (``"latest"`` for warm start)."""
    deterministic_contacts: bool = False
    """Whether the collision pipeline sorts its contacts (``deterministic=True``), so repeats of a route agree
    bitwise and the reference's run-to-run spread holds no contact or row order."""
    contact_stiffness: float = 0.0
    """Per-contact stiffness [N/m] written after each collision pass; 0 keeps hard contact."""
    contact_damping: float = 0.0
    """Per-contact damping [N·s/m] written with the stiffness."""
    passive: bool = False
    """Whether no drive, external force or spring injects energy, so mechanical energy cannot grow."""
    tags: frozenset = frozenset()
    """Features the scene exercises."""


def _cfg(
    *, mu: float = 0.5, density: float = 1000.0, restitution: float = 0.0, gap: float | None = None, group: int = 1
):
    """A shape configuration; ``gap`` ``None`` keeps the builder's :data:`RIGID_GAP`."""
    cfg = newton.ModelBuilder.ShapeConfig(mu=mu, density=density, restitution=restitution, collision_group=group)
    if gap is not None:
        cfg.gap = gap
    return cfg


def _builder() -> newton.ModelBuilder:
    """A builder with the scenes' contact gap."""
    builder = newton.ModelBuilder()
    builder.rigid_gap = RIGID_GAP
    return builder


def _finalize(builder: newton.ModelBuilder, device) -> newton.Model:
    """Finalize with joint targets in the coordinate layout, the only layout of future releases."""
    with mock.patch.object(newton, "use_coord_layout_targets", True):
        return builder.finalize(device=device)


def _replicate(template: newton.ModelBuilder, worlds: int, ground: newton.ModelBuilder.ShapeConfig | None):
    """Copy a one-world template into ``worlds`` worlds 1 m apart, beside a global ground plane if ``ground``."""
    builder = _builder()
    if ground is not None:
        builder.add_ground_plane(cfg=ground)
    builder.replicate(template, worlds, spacing=(1.0, 0.0, 0.0))
    return builder


def _set_targets(model: newton.Model, control: newton.Control, values: dict) -> None:
    """Write position targets: ``values`` maps a joint index to its target; every other target is zero."""
    start = model.joint_target_q_start.numpy()
    targets = np.zeros(model.joint_target_q.shape[0], dtype=np.float32)
    for joint, value in values.items():
        targets[start[joint]] = value
    control.joint_target_q.assign(targets)


def _box_inertia(mass: float, hx: float, hy: float, hz: float) -> wp.mat33:
    """Inertia of a solid box of half extents ``hx``, ``hy``, ``hz`` about its centre [kg·m²]."""
    return wp.mat33(
        np.diag(
            [
                mass * (hy * hy + hz * hz) / 3.0,
                mass * (hx * hx + hz * hz) / 3.0,
                mass * (hx * hx + hy * hy) / 3.0,
            ]
        ).astype(np.float32)
    )


def _sphere_inertia(mass: float, radius: float) -> wp.mat33:
    """Inertia of a solid sphere about its centre [kg·m²]."""
    return wp.mat33(np.eye(3, dtype=np.float32) * np.float32(0.4 * mass * radius * radius))


def _free_box(builder, position, rotation=None, *, half=0.1, mass=1.0, mu=0.5, restitution=0.0):
    """Add a free cube of half extent ``half`` and explicit mass; returns its body."""
    body = builder.add_body(
        xform=wp.transform(wp.vec3(*position), rotation or wp.quat_identity()),
        mass=mass,
        inertia=_box_inertia(mass, half, half, half),
    )
    builder.add_shape_box(body, hx=half, hy=half, hz=half, cfg=_cfg(mu=mu, density=0.0, restitution=restitution))
    return body


# ---------------------------------------------------------------------------------------------------------------------
# Free rigid bodies
# ---------------------------------------------------------------------------------------------------------------------


def box_drop(*, mu: float = 0.5) -> Scene:
    """Two worlds of a 0.2 m, 1 kg box dropped 1 cm onto the ground: flat, and yawed and tilted onto an edge."""
    tilt = 0.08

    def build(device):
        builder = _builder()
        builder.add_ground_plane(cfg=_cfg(mu=mu))
        builder.begin_world()
        _free_box(builder, (0.0, 0.0, 0.11), mu=mu)
        builder.end_world()
        builder.begin_world()
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5) * wp.quat_from_axis_angle(
            wp.vec3(1.0, 0.0, 0.0), tilt
        )
        _free_box(builder, (0.05, -0.02, 0.01 + 0.1 * (math.cos(tilt) + math.sin(tilt))), rotation, mu=mu)
        builder.end_world()
        return _finalize(builder, device)

    return Scene("box_drop", build, dt=1.0 / 240.0, steps=120, params={"mu": [mu, mu]}, passive=True)


def box_stack(*, mu: float = 0.6, contact_matching: str | None = None) -> Scene:
    """Three 0.2 m, 1 kg boxes stacked with 1 mm gaps, the middle one 2 cm off-centre."""

    def build(device):
        builder = _builder()
        builder.add_ground_plane(cfg=_cfg(mu=mu))
        for level, offset in enumerate((0.0, 0.02, 0.0)):
            _free_box(builder, (offset, 0.0, 0.101 + 0.201 * level), mu=mu)
        return _finalize(builder, device)

    name = "box_stack" if contact_matching is None else f"box_stack_{contact_matching}"
    return Scene(
        name,
        build,
        dt=1.0 / 240.0,
        steps=120,
        params={"mu": [mu]},
        passive=True,
        contact_matching=contact_matching,
    )


def incline(*, angle: float = 0.35, mus: tuple[float, ...] = (0.6, 0.15)) -> Scene:
    """A 0.2 m, 1 kg box on a plane tilted by ``angle`` [rad], one world per friction coefficient.

    Worlds with ``mu > tan(angle)`` stick; the others slide down the plane's
    local ``+x`` axis, ``(cos(angle), 0, -sin(angle))``, at
    ``g * (sin(angle) - mu * cos(angle))``.
    """
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)
    normal = np.array([math.sin(angle), 0.0, math.cos(angle)])

    def build(device):
        builder = _builder()
        for mu in mus:
            builder.begin_world()
            builder.add_shape_plane(xform=wp.transform(wp.vec3(0.0), rotation), width=0.0, length=0.0, cfg=_cfg(mu=mu))
            _free_box(builder, tuple(normal * 0.102), rotation, mu=mu)
            builder.end_world()
        return _finalize(builder, device)

    downhill = (math.cos(angle), 0.0, -math.sin(angle))
    return Scene(
        "incline",
        build,
        dt=1.0 / 240.0,
        steps=120,
        params={"mu": list(mus), "angle": angle, "normal": tuple(normal), "downhill": downhill},
        passive=True,
    )


def ball_drop(*, restitutions: tuple[float, ...] = (0.5, 0.0), height: float = 0.5) -> Scene:
    """A 0.1 m, 1 kg sphere dropped from ``height`` [m] onto a plane, one world per restitution coefficient."""

    def build(device):
        builder = _builder()
        for restitution in restitutions:
            builder.begin_world()
            builder.add_shape_plane(width=0.0, length=0.0, cfg=_cfg(mu=0.5, restitution=restitution, gap=0.05))
            body = builder.add_body(
                xform=wp.transform(wp.vec3(0.0, 0.0, height), wp.quat_identity()),
                mass=1.0,
                inertia=_sphere_inertia(1.0, 0.1),
            )
            builder.add_shape_sphere(body, radius=0.1, cfg=_cfg(mu=0.5, density=0.0, restitution=restitution, gap=0.05))
            builder.end_world()
        return _finalize(builder, device)

    return Scene(
        "ball_drop",
        build,
        dt=1.0 / 240.0,
        steps=150,
        params={"mu": [0.5] * len(restitutions), "restitution": list(restitutions), "radius": 0.1},
        passive=True,
        tags=frozenset({"restitution"}),
    )


def heightfield(*, mu: float = 0.6) -> Scene:
    """A 0.2 m box settling into a dimple of a sinusoidal heightfield.

    The terrain is ``0.02 sin(2 pi x) cos(2 pi y)`` [m] on a 33 x 33 grid over
    2 m. The box drops 2 mm onto the gentle bowl of the dimple at
    ``(-0.25, 0)`` and settles on the rims of its triangles. Scenes that
    tumble, roll or slowly tip are avoided: a sphere keeps rolling to and fro
    (rigid contacts have no rolling resistance), and a smaller box balanced on
    the facets of the other dimple tips over within a second, both of which
    make the end state depend on tiny differences.
    """
    amplitude, wavelength = 0.02, 1.0

    def build(device):
        builder = _builder()
        count = 33
        x = np.linspace(-1.0, 1.0, count)
        # Rows run along y and columns along x (Heightfield data is [nrow, ncol]).
        grid_x, grid_y = np.meshgrid(x, x, indexing="xy")
        data = amplitude * np.sin(2.0 * np.pi * grid_x / wavelength) * np.cos(2.0 * np.pi * grid_y / wavelength)
        field = newton.Heightfield(
            data=data.astype(np.float32), nrow=count, ncol=count, hx=1.0, hy=1.0, min_z=-amplitude, max_z=amplitude
        )
        builder.add_shape_heightfield(heightfield=field, cfg=_cfg(mu=mu))
        _free_box(builder, (-0.25, 0.0, 0.09), mu=mu)
        return _finalize(builder, device)

    return Scene(
        "heightfield",
        build,
        dt=1.0 / 240.0,
        steps=120,
        params={"mu": [mu]},
        passive=True,
        rigid_contact_max=256,
        tags=frozenset({"heightfield"}),
    )


def box_push(*, speed_limit: float = 0.5, force: float = 20.0) -> Scene:
    """A 1 kg box on a frictionless plane pushed by ``force`` [N] under a rigid-body speed limit [m/s].

    World 0 pushes along ``+x``, world 1 along the diagonal ``(1, 1, 0) / sqrt(2)``.
    PhysX's ``maxLinearVelocity`` bounds the speed, the magnitude of the linear
    velocity, and FeatherPGS bounds it on the body's solved velocity, so both
    boxes settle at ``speed_limit``. (PhysX clamps the unconstrained velocity,
    gravity's increment included, before its solve, which leaves a box pushed
    along the ground slightly below the limit.)
    """
    directions = ((1.0, 0.0, 0.0), (math.sqrt(0.5), math.sqrt(0.5), 0.0))

    def build(device):
        from newton.solvers import SolverFeatherPGS  # noqa: PLC0415

        builder = _builder()
        SolverFeatherPGS.register_custom_attributes(builder)
        for _ in directions:
            builder.begin_world()
            builder.add_shape_plane(width=0.0, length=0.0, cfg=_cfg(mu=0.0))
            body = builder.add_body(
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.101), wp.quat_identity()),
                mass=1.0,
                inertia=_box_inertia(1.0, 0.1, 0.1, 0.1),
                custom_attributes={
                    "rigid_body_max_linear_velocity": speed_limit,
                    "rigid_body_max_angular_velocity": 10.0,
                },
            )
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=_cfg(mu=0.0, density=0.0))
            builder.end_world()
        return _finalize(builder, device)

    def control(model, control, state, step):
        forces = np.zeros((model.body_count, 6), dtype=np.float32)
        forces[:, :3] = force * np.asarray(directions, dtype=np.float32)
        state.body_f.assign(forces)

    return Scene(
        "box_push",
        build,
        dt=1.0 / 240.0,
        steps=120,
        params={"mu": [0.0, 0.0], "speed_limit": speed_limit, "force": force},
        control=control,
        tags=frozenset({"rigid_body_velocity_limits"}),
    )


# ---------------------------------------------------------------------------------------------------------------------
# Fixed-base articulations without contact
# ---------------------------------------------------------------------------------------------------------------------


def _link_box(builder, xform, *, half, mass, mu=0.5, group=SELF):
    """Add an articulation link carrying a box of half extents ``half``; returns its body."""
    body = builder.add_link(xform=xform, mass=mass, inertia=_box_inertia(mass, *half))
    builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2], cfg=_cfg(mu=mu, density=0.0, group=group))
    return body


def _fixed_base(builder) -> tuple[int, int]:
    """Weld a shapeless base body to the world at the origin, as robot descriptions root a fixed-base arm.

    Joints whose parent is this base take world-frame parent transforms.
    Returns the base body and its fixed joint, which leads the articulation.
    """
    base = builder.add_link(mass=1.0, inertia=_box_inertia(1.0, 0.05, 0.05, 0.05))
    return base, builder.add_joint_fixed(-1, base)


def chain(*, links: int = 3) -> Scene:
    """A fixed-base chain of 0.3 m, 1 kg links on revolute ``y`` joints, released horizontally.

    Gravity drives every joint toward positive angles. Joint 0 is driven past
    its upper limit (the limit holds it against the drive), joint 1 holds a
    target inside its limits and the last joint is undriven and falls onto its
    upper limit.
    """
    lowers = (-0.5, -0.6, -0.4, -0.4)[:links]
    uppers = (0.5, 0.6, 0.4, 0.4)[:links]
    targets = (0.9, -0.2, 0.0, 0.0)[:links]
    gains = ((200.0, 10.0), (100.0, 5.0), (0.0, 0.0), (0.0, 0.0))[:links]

    def build(device):
        builder = _builder()
        parent, root = _fixed_base(builder)
        joints = [root]
        for index in range(links):
            body = _link_box(
                builder,
                wp.transform(wp.vec3(0.15 + 0.3 * index, 0.0, 1.0), wp.quat_identity()),
                half=(0.15, 0.03, 0.03),
                mass=1.0,
            )
            joints.append(
                builder.add_joint_revolute(
                    parent,
                    body,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0) if index == 0 else wp.vec3(0.15, 0.0, 0.0)),
                    child_xform=wp.transform(wp.vec3(-0.15, 0.0, 0.0)),
                    limit_lower=lowers[index],
                    limit_upper=uppers[index],
                    target_ke=gains[index][0],
                    target_kd=gains[index][1],
                )
            )
            parent = body
        builder.add_articulation(joints)
        return _finalize(builder, device)

    def control(model, control, state, step):
        if step == 0:
            _set_targets(model, control, {index + 1: target for index, target in enumerate(targets)})

    return Scene(
        "chain",
        build,
        dt=1.0 / 240.0,
        steps=240,
        params={"limit_lower": lowers, "limit_upper": uppers, "targets": targets},
        solver_options={"enable_joint_limits": True},
        control=control,
        tags=frozenset({"joint_limits", "drives"}),
    )


def four_bar(*, crank_target: float = 0.6) -> Scene:
    """A planar four-bar: two grounded revolute links closed by a ball loop joint, only the crank driven."""

    def build(device):
        builder = _builder()
        base, root = _fixed_base(builder)
        z0 = 0.6
        crank = _link_box(builder, wp.transform(wp.vec3(0.0, 0.0, z0 - 0.1)), half=(0.02, 0.02, 0.1), mass=0.32)
        j_crank = builder.add_joint_revolute(
            base,
            crank,
            axis=newton.Axis.Y,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, z0)),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1)),
            target_ke=50.0,
            target_kd=5.0,
        )
        coupler = _link_box(builder, wp.transform(wp.vec3(0.2, 0.0, z0 - 0.2)), half=(0.2, 0.02, 0.02), mass=0.64)
        j_coupler = builder.add_joint_revolute(
            crank,
            coupler,
            axis=newton.Axis.Y,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1)),
            child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0)),
        )
        rocker = _link_box(builder, wp.transform(wp.vec3(0.4, 0.0, z0 - 0.1)), half=(0.02, 0.02, 0.1), mass=0.32)
        j_rocker = builder.add_joint_revolute(
            base,
            rocker,
            axis=newton.Axis.Y,
            parent_xform=wp.transform(wp.vec3(0.4, 0.0, z0)),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1)),
        )
        builder.add_articulation([root, j_crank, j_coupler, j_rocker])
        builder.add_joint_ball(
            coupler,
            rocker,
            parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0)),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1)),
        )
        return _finalize(builder, device)

    def control(model, control, state, step):
        if step == 0:
            _set_targets(model, control, {1: crank_target})

    return Scene(
        "four_bar",
        build,
        dt=1.0 / 240.0,
        steps=240,
        params={"crank_target": crank_target},
        control=control,
        tags=frozenset({"loop_joints"}),
    )


def mimic_pair(*, coef0: float = 0.1, coef1: float = -0.5, leader_target: float = 0.5) -> Scene:
    """Two revolute ``z`` links: the leader driven to ``leader_target``, the follower held by a mimic constraint."""

    def build(device):
        builder = _builder()
        base, root = _fixed_base(builder)
        link_a = _link_box(builder, wp.transform(wp.vec3(0.2, 0.0, 0.5)), half=(0.1, 0.02, 0.02), mass=0.32)
        leader = builder.add_joint_revolute(
            base,
            link_a,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5)),
            child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0)),
            target_ke=50.0,
            target_kd=5.0,
        )
        link_b = _link_box(builder, wp.transform(wp.vec3(0.6, 0.0, 0.5)), half=(0.1, 0.02, 0.02), mass=0.32)
        follower = builder.add_joint_revolute(
            link_a,
            link_b,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0)),
            child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0)),
        )
        builder.add_articulation([root, leader, follower])
        builder.add_constraint_mimic(joint0=follower, joint1=leader, coef0=coef0, coef1=coef1)
        return _finalize(builder, device)

    def control(model, control, state, step):
        if step == 0:
            _set_targets(model, control, {1: leader_target})

    return Scene(
        "mimic_pair",
        build,
        dt=1.0 / 240.0,
        steps=240,
        params={"coef0": coef0, "coef1": coef1, "leader_target": leader_target},
        control=control,
        tags=frozenset({"mimic"}),
    )


def spring_hinge(*, stiffness: float = 2.0, damping: float = 0.02, start: float = 0.5) -> Scene:
    """A 0.3 m, 1 kg link on a revolute ``z`` hinge, released at ``start`` [rad] against a passive spring and damper.

    Gravity applies no torque about the vertical axis, so the hinge is a damped
    oscillator of inertia ``m (2 L)² / 3`` about the pivot.
    """
    mass, half = 1.0, 0.15

    def build(device):
        builder = _builder()
        base, root = _fixed_base(builder)
        link = _link_box(builder, wp.transform(wp.vec3(half, 0.0, 0.5)), half=(half, 0.03, 0.03), mass=mass)
        joint = builder.add_joint_revolute(
            base,
            link,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5)),
            child_xform=wp.transform(wp.vec3(-half, 0.0, 0.0)),
            spring_stiffness=stiffness,
            spring_ref=0.0,
            damping=damping,
        )
        builder.add_articulation([root, joint])
        builder.joint_q[0] = start
        return _finalize(builder, device)

    inertia = mass * (half * half + 0.03 * 0.03) / 3.0 + mass * half * half
    return Scene(
        "spring_hinge",
        build,
        dt=1.0 / 240.0,
        steps=240,
        params={"stiffness": stiffness, "damping": damping, "start": start, "inertia": inertia},
        tags=frozenset({"springs"}),
    )


def pushed_arm(*, velocity_limit: float = 1.0, torque: float = 2.0) -> Scene:
    """A fixed-base two-link revolute ``z`` arm pushed by joint torques beyond its joint velocity limits."""

    def build(device):
        builder = _builder()
        parent, root = _fixed_base(builder)
        joints = [root]
        for index in range(2):
            body = _link_box(
                builder,
                wp.transform(wp.vec3(0.1 + 0.2 * index, 0.0, 0.5)),
                half=(0.1, 0.02, 0.02),
                mass=0.5,
            )
            joints.append(
                builder.add_joint_revolute(
                    parent,
                    body,
                    axis=newton.Axis.Z,
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5) if index == 0 else wp.vec3(0.1, 0.0, 0.0)),
                    child_xform=wp.transform(wp.vec3(-0.1, 0.0, 0.0)),
                    velocity_limit=velocity_limit,
                )
            )
            parent = body
        builder.add_articulation(joints)
        return _finalize(builder, device)

    def control(model, control, state, step):
        if step == 0:
            control.joint_f.assign(np.array([torque, -0.5 * torque], dtype=np.float32))

    return Scene(
        "pushed_arm",
        build,
        dt=1.0 / 240.0,
        steps=120,
        params={"velocity_limit": velocity_limit},
        control=control,
        tags=frozenset({"joint_velocity_limits"}),
    )


# ---------------------------------------------------------------------------------------------------------------------
# Articulations in contact
# ---------------------------------------------------------------------------------------------------------------------


def legged(
    *, knees: bool = True, legs: int = 4, mu: float = 0.8, worlds: int = 2, contact_matching: str | None = None
) -> Scene:
    """A free-floating 4 kg base standing on ``legs`` legs with sphere feet, joints held by PD drives.

    With ``knees`` each leg is a hip and a knee (14 coordinates for four legs)
    crouched at hip 0.5 and knee -1.0 rad; without, a straight hip-only leg
    (10 coordinates). Only the base and the feet carry shapes, and feet start
    5 mm above the ground. World ``w`` targets a hip 0.05 ``w`` rad straighter.
    """
    hip_sites = ((0.15, 0.1), (0.15, -0.1), (-0.15, 0.1), (-0.15, -0.1))[:legs]
    hip, knee = (0.5, -1.0) if knees else (0.0, 0.0)
    segment = 0.16 if knees else 0.24
    foot = 0.025
    depth = (2.0 * segment * math.cos(hip)) if knees else segment
    base_z = 0.05 + depth + foot + 0.005

    def add_leg(builder, base, site):
        joints = []
        shapes_parent = base
        count = 2 if knees else 1
        for index in range(count):
            mass = 0.4 if index == 0 else 0.3
            body = builder.add_link(mass=mass, inertia=_box_inertia(mass, 0.02, 0.02, 0.5 * segment))
            joints.append(
                builder.add_joint_revolute(
                    shapes_parent,
                    body,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform(
                        wp.vec3(site[0], site[1], -0.05) if index == 0 else wp.vec3(0.0, 0.0, -0.5 * segment)
                    ),
                    child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * segment)),
                    target_ke=40.0,
                    target_kd=1.5,
                    limit_lower=-1.2 if index == 0 else -2.0,
                    limit_upper=1.2 if index == 0 else 0.2,
                )
            )
            shapes_parent = body
        builder.add_shape_sphere(
            shapes_parent,
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.5 * segment)),
            radius=foot,
            cfg=_cfg(mu=mu, density=0.0, group=SELF),
        )
        return joints

    def build(device):
        template = _builder()
        base = template.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, base_z)), mass=4.0, inertia=_box_inertia(4.0, 0.2, 0.12, 0.05)
        )
        template.add_shape_box(base, hx=0.2, hy=0.12, hz=0.05, cfg=_cfg(mu=mu, density=0.0, group=SELF))
        joints = [template.add_joint_free(base)]
        for site in hip_sites:
            joints.extend(add_leg(template, base, site))
        template.add_articulation(joints)
        coordinates = np.array(template.joint_q, dtype=np.float64)
        pose = [hip, knee] if knees else [hip]
        for leg in range(legs):
            for index, angle in enumerate(pose):
                coordinates[7 + leg * len(pose) + index] = angle
        template.joint_q = coordinates.tolist()
        builder = _builder()
        builder.add_ground_plane(cfg=_cfg(mu=mu))
        builder.replicate(template, worlds, spacing=(2.0, 0.0, 0.0))
        return _finalize(builder, device)

    pose = [hip, knee] * legs if knees else [hip] * legs

    def control(model, control, state, step):
        if step == 0:
            # Per world: the free root joint, then each leg's joints in order.
            joints = model.joint_count // worlds
            targets = {}
            for world in range(worlds):
                for index, angle in enumerate(pose):
                    hip_joint = index % (2 if knees else 1) == 0
                    targets[world * joints + 1 + index] = angle - (0.05 * world if hip_joint else 0.0)
            _set_targets(model, control, targets)

    name = "legged" if knees else "legged_straight"
    if contact_matching is not None:
        name = f"{name}_{contact_matching}"
    return Scene(
        name,
        build,
        dt=1.0 / 240.0,
        steps=160,
        params={"mu": [mu] * worlds},
        solver_options={"enable_joint_limits": True, "dense_max_constraints": 64},
        control=control,
        contact_matching=contact_matching,
        tags=frozenset({"articulated_contact", "joint_limits"}),
    )


def gripper(
    *, mu: float = 1.0, squeeze: float = 0.005, lift: float = 0.04, worlds: int = 2, table: bool = False
) -> Scene:
    """A lift and two prismatic fingers squeeze a free 0.2 kg box on the ground, then lift it by ``lift`` [m].

    The fingers close for 60 steps, then the lift target rises at 0.2 m/s. Each
    finger presses about ``400 N/m * squeeze``, so friction can hold twice the
    box's weight; world ``w`` squeezes ``w`` mm more, so worlds differ. With
    ``table`` the box rests on a kinematic table instead of the ground: a
    prescribed root, as a manipulation task's table is.
    """
    half = 0.04
    lift_start, lift_speed = 60, 0.2

    def build(device):
        builder = _builder()
        base, root = _fixed_base(builder)
        palm = _link_box(builder, wp.transform(wp.vec3(0.0, 0.0, 0.13)), half=(0.08, 0.03, 0.01), mass=0.5, mu=mu)
        joints = [
            root,
            builder.add_joint_prismatic(
                base,
                palm,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.13)),
                target_ke=4000.0,
                target_kd=100.0,
                limit_lower=-0.05,
                limit_upper=0.1,
            ),
        ]
        for side in (-1.0, 1.0):
            finger = builder.add_link(mass=0.1, inertia=_box_inertia(0.1, 0.01, 0.02, 0.025))
            builder.add_shape_box(finger, hx=0.01, hy=0.02, hz=0.025, cfg=_cfg(mu=mu, density=0.0, group=SELF))
            joints.append(
                builder.add_joint_prismatic(
                    palm,
                    finger,
                    axis=wp.vec3(-side, 0.0, 0.0),
                    parent_xform=wp.transform(wp.vec3(side * (half + 0.012), 0.0, -0.08)),
                    target_ke=400.0,
                    target_kd=10.0,
                    limit_lower=-0.02,
                    limit_upper=0.03,
                )
            )
        builder.add_articulation(joints)
        if table:
            top = builder.add_body(
                xform=wp.transform(wp.vec3(0.0, 0.0, -0.01)),
                mass=5.0,
                inertia=_box_inertia(5.0, 0.15, 0.15, 0.01),
                is_kinematic=True,
            )
            builder.add_shape_box(top, hx=0.15, hy=0.15, hz=0.01, cfg=_cfg(mu=mu, density=0.0))
        _free_box(builder, (0.0, 0.0, half + 0.001), half=half, mass=0.2, mu=mu)
        return _finalize(_replicate(builder, worlds, None if table else _cfg(mu=mu)), device)

    def control(model, control, state, step):
        # Per world: the base weld, the lift, the two fingers, then the box's free joint.
        height = min(max(step - lift_start, 0) * lift_speed / 240.0, lift)
        joints = model.joint_count // worlds
        targets = {}
        for world in range(worlds):
            close = 0.002 + squeeze + 0.001 * world
            targets.update({world * joints + 1: height, world * joints + 2: close, world * joints + 3: close})
        _set_targets(model, control, targets)

    return Scene(
        "gripper_table" if table else "gripper",
        build,
        dt=1.0 / 240.0,
        steps=140,
        params={"mu": [mu] * worlds, "lift": lift, "worlds": worlds},
        solver_options={"enable_joint_limits": True, "dense_max_constraints": 64},
        control=control,
        # A squeezed box is a statically indeterminate support, so the grasp drifts with the row order between CUDA
        # repeats (the table's by more than its routes' bands allowed): sorted contacts fix the order.
        deterministic_contacts=True,
        tags=frozenset({"articulated_contact", "grasp"}),
    )


def keyboard(*, keys: int = 8, press: float = 0.003, worlds: int = 2) -> Scene:
    """A fixed frame of ``keys`` prismatic keys (a diagonal-mass articulation) and a six-coordinate arm pressing key 0.

    Keys hang on ``z`` prismatic joints held by PD drives inside a 1 cm travel.
    The arm stacks prismatic ``z``, ``x``, ``y`` and revolute ``z``, ``y``,
    ``x`` joints at one point and carries a sphere tip that starts 1 mm above
    key 0; its ``z`` drive presses ``press`` [m] past contact, ``w`` mm more in
    world ``w``.
    """
    tip = 0.01

    def build(device):
        builder = _builder()
        cfg = _cfg(mu=0.5, density=0.0, gap=0.004, group=SELF)
        frame = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.1)), mass=1.0, inertia=_box_inertia(1.0, 0.1, 0.02, 0.01)
        )
        joints = [builder.add_joint_fixed(-1, frame, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1)))]
        for index in range(keys):
            key = builder.add_link(mass=0.01, inertia=_box_inertia(0.01, 0.01, 0.01, 0.005))
            builder.add_shape_box(key, hx=0.01, hy=0.01, hz=0.005, cfg=cfg)
            joints.append(
                builder.add_joint_prismatic(
                    frame,
                    key,
                    axis=newton.Axis.Z,
                    parent_xform=wp.transform(wp.vec3(0.03 * index, 0.0, 0.0)),
                    target_ke=400.0,
                    target_kd=2.0,
                    limit_lower=-0.008,
                    limit_upper=0.002,
                )
            )
        builder.add_articulation(joints)
        parent, root = _fixed_base(builder)
        arm_joints = [root]
        axes = (
            ("prismatic", wp.vec3(0.0, 0.0, -1.0)),
            ("prismatic", wp.vec3(1.0, 0.0, 0.0)),
            ("prismatic", wp.vec3(0.0, 1.0, 0.0)),
            ("revolute", wp.vec3(0.0, 0.0, 1.0)),
            ("revolute", wp.vec3(0.0, 1.0, 0.0)),
            ("revolute", wp.vec3(1.0, 0.0, 0.0)),
        )
        anchor = wp.vec3(0.0, 0.0, 0.1 + 0.005 + tip + 0.001)
        for index, (kind, axis) in enumerate(axes):
            body = builder.add_link(xform=wp.transform(anchor), mass=0.02, inertia=_sphere_inertia(0.02, 0.02))
            add = builder.add_joint_prismatic if kind == "prismatic" else builder.add_joint_revolute
            arm_joints.append(
                add(
                    parent,
                    body,
                    axis=axis,
                    parent_xform=wp.transform(anchor if index == 0 else wp.vec3(0.0)),
                    target_ke=2000.0 if kind == "prismatic" else 20.0,
                    target_kd=20.0 if kind == "prismatic" else 0.2,
                    limit_lower=-0.05,
                    limit_upper=0.05,
                )
            )
            parent = body
        builder.add_shape_sphere(parent, radius=tip, cfg=_cfg(mu=0.5, density=0.0, gap=0.004, group=SELF - 1))
        builder.add_articulation(arm_joints)
        return _finalize(_replicate(builder, worlds, None), device)

    def control(model, control, state, step):
        if step == 0:
            # Per world: the frame weld, the keys, the arm's base weld, then its prismatic z joint.
            joints = model.joint_count // worlds
            _set_targets(
                model, control, {world * joints + keys + 2: 0.001 + press + 0.001 * world for world in range(worlds)}
            )

    return Scene(
        "keyboard",
        build,
        dt=1.0 / 240.0,
        steps=160,
        params={"mu": [0.5] * worlds, "key_mass": 0.01, "key_stiffness": 400.0, "worlds": worlds},
        solver_options={"enable_joint_limits": True},
        control=control,
        tags=frozenset({"articulated_contact", "diagonal_mass"}),
    )


def hand(*, fingers: int = 4, segments: int = 4, press: float = 0.002, worlds: int = 2) -> Scene:
    """A 17-coordinate hand (a ``z`` lift under four 4-joint fingers) balancing a free 0.5 kg cube on its fingertips.

    The fingers stand straight up from the palm's corners, their sphere tips
    1 mm under the cube's bottom face, and their PD drives hold them upright;
    the lift drive raises the palm ``press`` [m] past contact, ``w`` mm more in
    world ``w``. The cube touches only the fingertips, so hand and cube are one
    coupled component without matrix-free rows. The last world sets its cube
    on the ground 0.3 m aside instead: there the hand touches nothing and the
    cube only the ground.
    """
    cube, tip, link, palm_z = 0.05, 0.008, 0.03, 0.05
    corners = ((0.03, 0.03), (0.03, -0.03), (-0.03, 0.03), (-0.03, -0.03))[:fingers]
    cube_z = palm_z + 0.01 + segments * link + tip + 0.001 + cube

    def build(device):
        builder = _builder()
        base, root = _fixed_base(builder)
        palm = _link_box(builder, wp.transform(wp.vec3(0.0, 0.0, palm_z)), half=(0.04, 0.04, 0.01), mass=0.3, mu=0.8)
        joints = [
            root,
            builder.add_joint_prismatic(
                base,
                palm,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, palm_z)),
                target_ke=1000.0,
                target_kd=30.0,
            ),
        ]
        for corner in corners:
            parent = palm
            for segment in range(segments):
                body = builder.add_link(mass=0.02, inertia=_box_inertia(0.02, 0.006, 0.006, 0.5 * link))
                joints.append(
                    builder.add_joint_revolute(
                        parent,
                        body,
                        axis=newton.Axis.Y if segment % 2 == 0 else newton.Axis.X,
                        parent_xform=wp.transform(
                            wp.vec3(corner[0], corner[1], 0.01) if segment == 0 else wp.vec3(0.0, 0.0, 0.5 * link)
                        ),
                        child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.5 * link)),
                        target_ke=4.0,
                        target_kd=0.05,
                    )
                )
                parent = body
            builder.add_shape_sphere(
                parent,
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * link)),
                radius=tip,
                cfg=_cfg(mu=0.8, density=0.0, group=SELF),
            )
        builder.add_articulation(joints)
        _free_box(builder, (0.0, 0.0, cube_z), half=cube, mass=0.5, mu=0.8)
        replicated = _replicate(builder, worlds, _cfg(mu=0.8))
        # The last joint is the last world's cube: set it on the ground beside the hand.
        start = replicated.joint_q_start[-1]
        replicated.joint_q[start] += 0.3
        replicated.joint_q[start + 2] = cube + 0.001
        return _finalize(replicated, device)

    def control(model, control, state, step):
        if step == 0:
            # Per world: the base weld, then the lift.
            joints = model.joint_count // worlds
            _set_targets(model, control, {world * joints + 1: 0.001 + press + 0.001 * world for world in range(worlds)})

    return Scene(
        "hand",
        build,
        dt=1.0 / 240.0,
        steps=120,
        params={"mu": [0.8] * worlds, "worlds": worlds},
        solver_options={"dense_max_constraints": 64},
        control=control,
        tags=frozenset({"articulated_contact"}),
    )


def torsion_pad(*, torques: tuple[float, ...] = (0.1, 0.4), mu: float = 0.5, radius: float = 0.1) -> Scene:
    """A 1 kg sphere on a six-coordinate D6 joint resting on the ground under a yaw torque, one world per torque.

    A sphere touches the ground at one point, so tangential friction resists
    no spin: only the torsion row of an effective spin radius ``R`` does, and
    it holds the sphere while ``|torque| <= mu * m * g * R``. Above that the
    sphere spins up at ``(torque - mu m g R) / (0.4 m r^2)``.
    """

    def build(device):
        builder = _builder()
        builder.add_ground_plane(cfg=_cfg(mu=mu))
        axis = newton.ModelBuilder.JointDofConfig
        for _ in torques:
            builder.begin_world()
            body = builder.add_link(
                xform=wp.transform(wp.vec3(0.0, 0.0, radius + 0.001)), mass=1.0, inertia=_sphere_inertia(1.0, radius)
            )
            builder.add_shape_sphere(body, radius=radius, cfg=_cfg(mu=mu, density=0.0))
            joint = builder.add_joint_d6(
                -1,
                body,
                linear_axes=[axis(axis=newton.Axis.X), axis(axis=newton.Axis.Y), axis(axis=newton.Axis.Z)],
                angular_axes=[axis(axis=newton.Axis.X), axis(axis=newton.Axis.Y), axis(axis=newton.Axis.Z)],
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, radius + 0.001)),
            )
            builder.add_articulation([joint])
            builder.end_world()
        return _finalize(builder, device)

    def control(model, control, state, step):
        if step == 0:
            forces = np.zeros(model.joint_f.shape[0], dtype=np.float32)
            for world, torque in enumerate(torques):
                forces[world * 6 + 5] = torque
            control.joint_f.assign(forces)

    return Scene(
        "torsion_pad",
        build,
        dt=1.0 / 240.0,
        steps=120,
        params={"mu": [mu] * len(torques), "torques": torques, "mass": 1.0, "radius": radius},
        control=control,
        tags=frozenset({"torsion"}),
    )


def compliant_ball(*, stiffness: float = 3000.0, damping: float = 20.0, mass: float = 0.3) -> Scene:
    """A 0.05 m free sphere resting on a compliant contact of ``stiffness`` [N/m] and ``damping`` [N·s/m]."""
    radius = 0.05

    def build(device):
        builder = _builder()
        builder.rigid_gap = 0.005
        builder.add_ground_plane(cfg=_cfg(mu=0.5))
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, radius)), mass=mass, inertia=_sphere_inertia(mass, radius)
        )
        builder.add_shape_sphere(body, radius=radius, cfg=_cfg(mu=0.5, density=0.0))
        return _finalize(builder, device)

    return Scene(
        "compliant_ball",
        build,
        dt=0.005,
        steps=200,
        params={"mu": [0.5], "stiffness": stiffness, "mass": mass, "radius": radius},
        rigid_contact_max=32,
        contact_stiffness=stiffness,
        contact_damping=damping,
        tags=frozenset({"contact_compliance"}),
    )


# ---------------------------------------------------------------------------------------------------------------------
# Large articulations (above the held-tree owner's 64-body, 64-coordinate cap)
# ---------------------------------------------------------------------------------------------------------------------


def _limb(
    builder,
    parent,
    anchor,
    heading,
    *,
    links,
    length,
    radius,
    mass,
    mu,
    axes=(newton.Axis.Z,),
    ke=4.0,
    kd=0.1,
    armature=0.0,
):
    """Append a serial limb of ``links`` revolute links lying along ``heading`` [rad, about z] in the ground plane.

    Each link's body origin sits on its joint and carries a ``radius`` sphere at
    its far end; joint ``k`` turns about ``axes[k % len(axes)]`` of the link
    frame and a PD drive holds it at zero. Returns the joints and the last body.
    """
    joints, body = [], parent
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), heading)
    for index in range(links):
        child = builder.add_link(
            com=wp.vec3(0.5 * length, 0.0, 0.0),
            mass=mass,
            inertia=_box_inertia(mass, 0.5 * length, radius, radius),
        )
        builder.add_shape_sphere(
            child,
            xform=wp.transform(wp.vec3(length, 0.0, 0.0)),
            radius=radius,
            cfg=_cfg(mu=mu, density=0.0, group=SELF),
        )
        joints.append(
            builder.add_joint_revolute(
                body,
                child,
                axis=axes[index % len(axes)],
                parent_xform=wp.transform(anchor, rotation) if index == 0 else wp.transform(wp.vec3(length, 0.0, 0.0)),
                target_ke=ke,
                target_kd=kd,
                armature=armature,
            )
        )
        body = child
    return joints, body


def sprawl(*, mu: float = 0.8, fingers: int = 5, finger_links: int = 4) -> Scene:
    """A humanoid-like free torso lying on the ground with two hands, two legs and a neck: 66 coordinates.

    Arms (4 links) end in ``fingers`` fingers of ``finger_links`` links fanned
    in the ground plane, legs have 5 links and the neck 2. Every link carries a
    3 cm sphere and every joint lies at the spheres' height, so the whole tree
    rests on the ground through one contact per link plus the torso's box. The
    tree exceeds the held-tree owner's cap of 64 coordinates.
    """
    radius = 0.03

    def build(device):
        builder = _builder()
        builder.add_ground_plane(cfg=_cfg(mu=mu))
        torso = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, radius + 0.002)),
            mass=3.0,
            inertia=_box_inertia(3.0, 0.2, 0.12, radius),
        )
        builder.add_shape_box(torso, hx=0.2, hy=0.12, hz=radius, cfg=_cfg(mu=mu, density=0.0, group=SELF))
        joints = [builder.add_joint_free(torso)]
        axes = (newton.Axis.Z, newton.Axis.Y, newton.Axis.X)
        for side in (-1.0, 1.0):
            arm, wrist = _limb(
                builder,
                torso,
                wp.vec3(0.08, side * 0.12, 0.0),
                side * 0.5 * math.pi,
                links=4,
                length=0.1,
                radius=radius,
                mass=0.15,
                mu=mu,
                axes=axes,
            )
            joints += arm
            for finger in range(fingers):
                offset = (finger - 0.5 * (fingers - 1)) * 0.035
                spread = (finger - 0.5 * (fingers - 1)) * 0.25
                anchor = wp.vec3(0.1 + 0.02, offset, 0.0)
                chain, _ = _limb(
                    builder,
                    wrist,
                    anchor,
                    spread,
                    links=finger_links,
                    length=0.06,
                    radius=radius,
                    mass=0.03,
                    mu=mu,
                    axes=(newton.Axis.Z, newton.Axis.Y),
                )
                joints += chain
            leg, _ = _limb(
                builder,
                torso,
                wp.vec3(-0.2, side * 0.07, 0.0),
                math.pi,
                links=5,
                length=0.12,
                radius=radius,
                mass=0.3,
                mu=mu,
                axes=axes,
            )
            joints += leg
        neck, _ = _limb(
            builder, torso, wp.vec3(0.2, 0.0, 0.0), 0.0, links=2, length=0.08, radius=radius, mass=0.2, mu=mu
        )
        builder.add_articulation(joints + neck)
        return _finalize(builder, device)

    return Scene(
        "sprawl",
        build,
        dt=1.0 / 240.0,
        steps=60,
        params={"mu": [mu]},
        solver_options={"dense_max_constraints": 256},
        rigid_contact_max=512,
        # The tree rests on one contact per link, a statically indeterminate support: sorted contacts fix the row
        # order (see long_chain).
        deterministic_contacts=True,
        tags=frozenset({"articulated_contact", "large_articulation"}),
    )


def long_chain(*, links: int = 100, mu: float = 0.6, pull: float = 0.0, armature: float = 1.0e-3) -> Scene:
    """A free-floating snake of ``links`` 4 cm revolute ``z`` links dropped 2 mm in an arc onto the ground.

    Every link carries a 2 cm sphere, so the chain lands on and rests along
    its whole length; an optional ``pull`` [N] on the head body along ``+x``
    loads friction (the gate leaves it at zero). The chain has ``links + 1``
    bodies and ``links + 6`` coordinates, above the held-tree owner's caps.
    Joint ``armature`` [kg·m²] (a geared joint's rotor inertia) bounds the
    condition number of the generalized mass matrix: a passive 100-link cable
    without it reaches about 4e8, beyond float32, and the solver adds the
    armature its factorization needs near the head (the scene is then named
    ``long_chain_bare``).

    The support is statically indeterminate: the ``z`` joints leave the chain
    rigid out of its plane, so ``links + 1`` contacts carry a body with three
    such freedoms. The rigid contact problem then has a family of solutions,
    and the order in which Gauss-Seidel visits the rows picks the member: at
    rest, a sweep in the contacts' sorted order holds the chain to about
    1e-6 rad/s, other orders leave up to 2e-3 rad/s and 2e-4 rad of drift by
    the end, and more iterations do not remove it. The scene therefore sorts
    its contacts, so every run and every route sweeps the same order;
    otherwise the order would follow the warps of the atomic broad-phase,
    contact and row allocations.
    """
    radius = 0.02

    def build(device):
        builder = _builder()
        builder.add_ground_plane(cfg=_cfg(mu=mu))
        head = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, radius + 0.002)), mass=0.05, inertia=_sphere_inertia(0.05, radius)
        )
        builder.add_shape_sphere(head, radius=radius, cfg=_cfg(mu=mu, density=0.0, group=SELF))
        joints = [builder.add_joint_free(head)]
        chain, _ = _limb(
            builder,
            head,
            wp.vec3(0.0),
            math.pi,
            links=links,
            length=0.04,
            radius=radius,
            mass=0.05,
            mu=mu,
            ke=0.0,
            kd=0.0,
            armature=armature,
        )
        builder.add_articulation(joints + chain)
        coordinates = np.array(builder.joint_q, dtype=np.float64)
        coordinates[7:] = 0.02
        builder.joint_q = coordinates.tolist()
        return _finalize(builder, device)

    def control(model, control, state, step):
        forces = np.zeros((model.body_count, 6), dtype=np.float32)
        forces[0, 0] = pull
        state.body_f.assign(forces)

    return Scene(
        "long_chain" if armature else "long_chain_bare",
        build,
        dt=1.0 / 240.0,
        steps=40,
        params={"mu": [mu], "pull": pull},
        solver_options={"dense_max_constraints": 384},
        rigid_contact_max=512,
        control=control,
        deterministic_contacts=True,
        tags=frozenset({"articulated_contact", "large_articulation"}),
    )
