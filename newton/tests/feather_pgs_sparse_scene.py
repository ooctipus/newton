# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Branched floating or fixed trees with one contacting foot, on native sparse factors or dense storage: the scene,
the paired solvers and their stepping, shared by the sparse-route tests; and the host probes of the sparse rule
(:class:`ForestProbe`, :func:`eligible_inputs`), which stop it where topology planning starts."""

from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_patch_parity import standing_model

DT = 1.0 / 400.0
STATE_FIELDS = ("joint_q", "joint_qd", "body_q", "body_qd")


def branched_model(*, floating=True, branches=3, depth=1, foot="sphere", device="cuda:0"):
    """Build branched floating or fixed trees with one contacting foot.

    A sphere foot touches the ground at one point. A box foot touches it at
    several, so patch friction links their normals in one ring and divides
    the friction budget between its anchors.
    """
    scene = newton.ModelBuilder()
    root = scene.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.16), wp.quat_identity()),
        mass=2.0,
        inertia=wp.mat33(np.diag([0.025, 0.03, 0.04])),
    )
    joints = [
        scene.add_joint_free(root)
        if floating
        else scene.add_joint_fixed(-1, root, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.16), wp.quat_identity()))
    ]
    for branch in range(branches):
        parent = root
        for level in range(depth):
            child = scene.add_link(mass=0.5, inertia=wp.mat33(np.diag([0.002, 0.003, 0.004])))
            joints.append(
                scene.add_joint_revolute(
                    parent,
                    child,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform(
                        wp.vec3(0.1 if level == 0 else 0.0, 0.12 * branch if level == 0 else 0.0, -0.11 / depth),
                        wp.quat_identity(),
                    ),
                    target_pos=0.03,
                    target_ke=15.0,
                    target_kd=1.0,
                    armature=0.05,
                    limit_lower=-0.2,
                    limit_upper=0.2,
                    effort_limit=10.0,
                )
            )
            if branch == 0 and level == depth - 1:
                cfg = scene.default_shape_cfg.copy()
                cfg.density, cfg.mu, cfg.restitution = 0.0, 0.6, 0.4
                xform = wp.transform(wp.vec3(0.02, 0.0, 0.0), wp.quat_identity())
                if foot == "box":
                    scene.add_shape_box(child, xform=xform, hx=0.04, hy=0.04, hz=0.055, cfg=cfg)
                else:
                    scene.add_shape_sphere(child, xform=xform, radius=0.055, cfg=cfg)
            parent = child
    scene.add_articulation(joints)
    scene.joint_q[7 if floating else 0] = 0.23
    scene.joint_qd[:] = [0.7] * len(scene.joint_qd)
    if floating:
        scene.joint_qd[:6] = [0.12, -0.08, -0.7, 0.03, -0.04, 0.02]
    builder = newton.ModelBuilder()
    builder.replicate(scene, 2)
    ground = builder.default_shape_cfg.copy()
    ground.restitution = 0.4
    builder.add_ground_plane(cfg=ground)
    return builder.finalize(device=device)


def make_solver(model, enabled, pins=None, **overrides):
    """Change only representation admission between the paired solvers; ``pins`` are more test-only route pins."""
    options = {
        "pgs_iterations": 8,
        "pgs_warmstart": False,
        "enable_joint_limits": True,
        "enable_joint_velocity_limits": False,
        "enable_restitution": True,
        "restitution_velocity_threshold": 0.01,
        "dense_max_constraints": 32,
        "mf_max_constraints": 32,
        "update_mass_matrix_interval": 2,
    }
    options.update(overrides)
    pins = {**({} if enabled else {"storage": "dense"}), **(pins or {})}
    return SolverFeatherPGS(model, pins=SolverFeatherPGS.Pins(**pins), **options)


def make_case(enabled, *, options=None, pins=None, contact_matching=None, deterministic=False, standing=False, **shape):
    model = standing_model(**shape) if standing else branched_model(**shape)
    solver = make_solver(model, enabled, pins=pins, **(options or {}))
    state, out, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    control.joint_f.assign(np.linspace(-0.08, 0.11, model.joint_dof_count, dtype=np.float32))
    matching = {} if contact_matching is None else {"contact_matching": contact_matching}
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, deterministic=deterministic, **matching)
    return model, solver, state, out, control, pipeline, pipeline.contacts()


@wp.kernel
def _external_forces(force: wp.array[wp.spatial_vector]):
    body = wp.tid()
    force[body] = wp.spatial_vector(0.03, -0.02, -0.07, 0.002, -0.003, 0.001)


def step(case):
    model, solver, state, out, control, pipeline, contacts = case
    state.clear_forces()
    wp.launch(_external_forces, dim=model.body_count, inputs=[state.body_f], device=model.device)
    pipeline.collide(state, contacts)
    solver.step(state, out, control, contacts, DT)
    for field in STATE_FIELDS:
        wp.copy(getattr(state, field), getattr(out, field))


class TopologyReached(Exception):
    """Raised by :class:`ForestProbe` once the sparse admission reaches topology planning."""


class ForestProbe(SimpleNamespace):
    @property
    def response_dof_count(self):
        # Stop after admission, before any factor planning or device allocation.
        raise TopologyReached


def eligible_inputs():
    """The class rules' inputs (:class:`.class_plan.ClassInputs`) the sparse rule reads before topology planning
    starts: a CUDA model, plain options, one mimic-free tree without free bodies, per-coordinate drives."""
    forest = ForestProbe(has_free_rigid_bodies=False)
    options = SimpleNamespace(
        pgs_iterations=8,
        enable_joint_velocity_limits=False,
        pgs_warmstart=False,
        contact_compliance=False,
    )
    return SimpleNamespace(
        model=SimpleNamespace(device=SimpleNamespace(is_cuda=True)),
        pins=SolverFeatherPGS.Pins(),
        factor=SimpleNamespace(parallel_drives=True, eliminates=False),
        dense_rows=32,
        forest=forest,
        options=options,
        laws=SimpleNamespace(options=options, forest=forest, mimic_count=0),
        packed_decline=None,
    )


def change_probe(inputs, name, value):
    """Set a probe's field: an option in its options record, its forest plan in both places that read it."""
    if name == "forest":
        inputs.forest = inputs.laws.forest = value
    elif hasattr(inputs.options, name):
        setattr(inputs.options, name, value)
    else:
        setattr(inputs, name, value)


def equal_dof_trees(branches, depth, device):
    """Build equal-DOF trees with different widths, without loading task assets."""
    scene = newton.ModelBuilder()
    root = scene.add_link(mass=2.0, inertia=wp.mat33(np.eye(3)))
    joints = [scene.add_joint_free(root)]
    for branch in range(branches):
        parent = root
        for level in range(depth):
            child = scene.add_link(mass=0.5, inertia=wp.mat33(np.eye(3) * 0.1))
            joints.append(
                scene.add_joint_revolute(
                    parent,
                    child,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform(wp.vec3(0.1, 0.1 * branch, -0.1), wp.quat_identity()),
                    target_ke=20.0,
                    target_kd=2.0,
                    armature=0.1,
                )
            )
            if level == depth - 1:
                scene.add_shape_sphere(child, radius=0.04)
            parent = child
    scene.add_articulation(joints)
    builder = newton.ModelBuilder()
    builder.replicate(scene, 2)
    return builder.finalize(device=device)
