# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FeatherPGS contact regularizer (``pgs_contact_regularization``)."""

import inspect
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import field_group
from newton._src.solvers.feather_pgs.dense_rows import DenseRows
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    compute_world_contact_bias,
)
from newton.tests.feather_pgs_free_body_rows import sweep_free_body_rows
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, get_test_devices

DT = 1.0 / 60.0
PATH_DENSE = 0  # contact routing id of the dense Delassus path


def _build_scissor_model(device: str) -> newton.Model:
    """One articulation: base link + two sibling links whose boxes overlap.

    The siblings are not directly jointed, so collision produces a
    link-link contact within a single articulation (common ancestor = base).
    """
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 1.0e3
    builder.default_shape_cfg.mu = 0.75
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    base = builder.add_link()
    builder.add_shape_box(base, hx=0.05, hy=0.05, hz=0.05)
    j_base = builder.add_joint_revolute(
        parent=-1,
        child=base,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform_identity(),
    )

    joints = [j_base]
    for side in (1.0, -1.0):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.12, hy=0.03, hz=0.04)
        joints.append(
            builder.add_joint_revolute(
                parent=base,
                child=link,
                axis=wp.vec3(0.0, 0.0, 1.0),
                parent_xform=wp.transform(wp.vec3(0.06, side * 0.028, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.12, 0.0, 0.0), wp.quat_identity()),
            )
        )
    builder.add_articulation(joints)
    return builder.finalize(device=device)


def _contact_rows(solver, path_id: int, contact_count: int) -> dict[int, int]:
    """contact index -> row slot for world-0 contacts routed to the given path."""
    contact_path = solver.placement.contact_path.numpy().astype(np.int32)
    contact_slot = solver.placement.contact_slot.numpy().astype(np.int32)
    contact_world = solver.placement.contact_world.numpy().astype(np.int32)
    out = {}
    for c in range(contact_count):
        if int(contact_world[c]) == 0 and int(contact_path[c]) == path_id and int(contact_slot[c]) >= 0:
            out[c] = int(contact_slot[c])
    return out


def _solve_rows(device, n_rows, g, iterations=300):
    """Solve ``n_rows`` identical unit contact rows on one 6-DOF body with unit mass and rhs = -1 through the general
    sweep's free-body rows, regularized with the weight ``1 / (1 + g)``."""
    Z = np.zeros((n_rows, 6))
    Z[:, 2] = 1.0
    return sweep_free_body_rows(
        device,
        Z,
        np.full(n_rows, -1.0),
        np.full(n_rows, PGS_CONSTRAINT_TYPE_CONTACT),
        np.full(n_rows, -1),
        np.zeros(n_rows),
        weights=np.full(n_rows, 1.0 / (1.0 + g)),
        regularize=True,
        iterations=iterations,
    )


def test_regularization_fixed_points(test: unittest.TestCase, device):
    """Analytic contracts of the regularized update on unit rows:

    - one determined row converges to ``1/(1+g)`` of the rigid impulse (the
      documented load deficit);
    - two redundant identical rows converge to the unique symmetric split
      ``lambda_1 = lambda_2 = 1/(2+g)``;
    - ``g = 0`` reproduces the exact rigid law (total impulse 1).
    """
    with wp.ScopedDevice(device):
        g = 0.5
        lam = _solve_rows(device, 1, g)
        test.assertAlmostEqual(float(lam[0]), 1.0 / (1.0 + g), places=5)
        lam = _solve_rows(device, 2, g)
        test.assertAlmostEqual(float(lam[0]), 1.0 / (2.0 + g), places=5)
        test.assertAlmostEqual(float(lam[1]), 1.0 / (2.0 + g), places=5)
        lam = _solve_rows(device, 2, 0.0)
        test.assertAlmostEqual(float(lam.sum()), 1.0, places=5)


def _stack_scene(device, g):
    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.003
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
    bodies = []
    for k in range(3):
        b = builder.add_body(xform=wp.transform(wp.vec3(0.002 * k, 0.0, 0.0505 + 0.101 * k), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
        bodies.append(b)
    model = builder.finalize()
    pipeline = newton.CollisionPipeline(
        model,
        reduce_contacts=True,
        rigid_contact_max=128,
        broad_phase="nxn",
        deterministic=True,
        contact_matching="latest",
    )
    solver = newton.solvers.SolverFeatherPGS(
        model,
        pgs_iterations=8,
        pgs_contact_regularization=g,
        pgs_warmstart=True,
    )
    return model, pipeline, solver, bodies


def _run(model, pipeline, solver, frames, dt=DT):
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    zs = []
    for _k in range(frames):
        pipeline.collide(s0, contacts)
        s0.clear_forces()
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
        zs.append(float(s0.body_q.numpy()[-1][2]))
    return s0.body_q.numpy(), zs


def _resting_box(device, g, rate, frames, **solver_kwargs):
    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.01
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()))
    builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    model = builder.finalize()
    pipeline = newton.CollisionPipeline(
        model,
        reduce_contacts=True,
        rigid_contact_max=32,
        broad_phase="nxn",
        deterministic=True,
        contact_matching="latest",
    )
    solver = newton.solvers.SolverFeatherPGS(
        model,
        pgs_iterations=12,
        pgs_contact_regularization=g,
        **solver_kwargs,
    )
    _, zs = _run(model, pipeline, solver, frames, dt=1.0 / rate)
    return 0.05 - zs[-1]


def _sag_formula(g, rate):
    """Resting sag of a body under gravity: ``g * a * dt^2 / pgs_beta`` with the default beta."""
    return g * 9.81 / (rate * rate) / 0.2


def test_regularization_documented_sag(test: unittest.TestCase, device):
    """The regularizer is a numerical damped compliance: a resting box sags by
    ``g * a * dt^2 / beta`` (6.8 mm at g = 0.5 and 60 Hz, 0.43 mm at 240 Hz), on the
    matrix-free route."""
    with wp.ScopedDevice(device):
        for rate in (60, 240):
            sag = _resting_box(device, 0.5, rate, 3 * rate, pgs_warmstart=True)
            expected = _sag_formula(0.5, rate)
            test.assertAlmostEqual(sag, expected, delta=0.15 * expected, msg=f"{rate} Hz: sag {sag * 1000:.2f} mm")


def test_regularization_cpu_route_documented_sag(test: unittest.TestCase, device):
    """The CPU's free-body loop sweep carries the same law as the CUDA sweeps, so its
    resting box sags by the same formula."""
    with wp.ScopedDevice(device):
        sag = _resting_box(device, 0.5, 60, 180, pgs_warmstart=True)
        expected = _sag_formula(0.5, 60)
        test.assertAlmostEqual(sag, expected, delta=0.15 * expected, msg=f"split sag {sag * 1000:.2f} mm")


def test_regularization_indeterminate_split(test: unittest.TestCase, device):
    """A plank on three identical supports has no unique rigid force split;
    the regularizer must select the symmetric one (outer supports equal)."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.003
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
        supports = []
        for k in range(3):
            b = builder.add_body(xform=wp.transform(wp.vec3(0.15 * (k - 1), 0.0, 0.0255), wp.quat_identity()))
            s = builder.add_shape_box(b, hx=0.025, hy=0.025, hz=0.025, cfg=cfg)
            supports.append(s)
        plank = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0605), wp.quat_identity()))
        builder.add_shape_box(plank, hx=0.2, hy=0.03, hz=0.01, cfg=cfg)
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(
            model,
            reduce_contacts=True,
            rigid_contact_max=256,
            broad_phase="nxn",
            deterministic=True,
            contact_matching="latest",
        )
        solver = newton.solvers.SolverFeatherPGS(
            model,
            pgs_iterations=8,
            pgs_contact_regularization=0.02,
            pgs_warmstart=True,
        )
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        for _k in range(120):
            pipeline.collide(s0, contacts)
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, DT)
            s0, s1 = s1, s0
        solver.update_contacts(contacts)

        count = int(contacts.rigid_contact_count.numpy()[0])
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        force = contacts.rigid_contact_force.numpy()[:count]
        loads = []
        for s in supports:
            fz = 0.0
            for i in range(count):
                # plank-support pairs only (exclude support-ground)
                if s in (shape0[i], shape1[i]) and 0 not in (shape0[i], shape1[i]):
                    fz += abs(float(force[i][2]))
            loads.append(fz)
        test.assertGreater(min(loads), 0.0, f"a support carries no load: {loads}")
        test.assertAlmostEqual(
            loads[0],
            loads[2],
            delta=0.1 * max(loads),
            msg=f"outer supports asymmetric: {loads}",
        )


def test_regularization_validation(test: unittest.TestCase, device):
    """Parameter contract: finite non-negative values only."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05)
        model = builder.finalize()
        for bad in (-0.1, float("nan"), float("inf"), 1.0e7):
            with test.assertRaises(ValueError):
                newton.solvers.SolverFeatherPGS(model, pgs_contact_regularization=bad)
        newton.solvers.SolverFeatherPGS(model, pgs_contact_regularization=0.0)
        newton.solvers.SolverFeatherPGS(model, pgs_contact_regularization=1.0e6)


def test_zero_regularization_shares_one_weight_slot(test: unittest.TestCase, device):
    """The exact-rigid path carries no capacity-sized per-row weight buffers."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
        model = builder.finalize()
        solver = newton.solvers.SolverFeatherPGS(model)
        test.assertFalse(solver.laws.regularization_enabled)
        test.assertIs(solver.placement.row_w, solver.placement.contact_row_w_dummy)
        test.assertIs(solver.placement.mf_row_w, solver.placement.contact_row_w_dummy)
        test.assertEqual(solver.placement.row_w.shape, (1, 1))


def test_zero_regularization_split_runtime_uses_no_weights(test: unittest.TestCase, device):
    """The standalone MF route must not read its one-element inactive weight argument."""
    with wp.ScopedDevice(device):
        model, pipeline, solver, _bodies = _stack_scene(device, g=0.0)
        body_q, _zs = _run(model, pipeline, solver, 1)
        test.assertGreater(int(solver.placement.mf_constraint_count.numpy().sum()), 1)
        test.assertTrue(np.isfinite(body_q).all())
        test.assertIs(solver.placement.mf_row_w, solver.placement.contact_row_w_dummy)
        test.assertEqual(float(solver.placement.mf_row_w.numpy()[0, 0]), 1.0)


def test_exact_surface_contact_is_regularized(test: unittest.TestCase, device):
    """A zero-gap contact is active; only a strictly positive gap is speculative."""
    with wp.ScopedDevice(device):
        row_w = wp.zeros((1, 1), dtype=wp.float32, device=device)
        rows = field_group.upload(
            DenseRows,
            device,
            phi=[[0.0]],
            row_beta=[[0.2]],
            row_type=[[PGS_CONSTRAINT_TYPE_CONTACT]],
            target=[[0.0]],
            restitution=[[0.0]],
            rhs=[[0.0]],
            row_w=row_w,
        )
        wp.launch(
            compute_world_contact_bias,
            dim=1,
            inputs=[
                rows,
                wp.zeros_like(row_w),
                wp.array([1], dtype=wp.int32, device=device),
                1,
                DT,
                1.0,
                0.5,
                0.0,
                1,
                0,
            ],
            device=device,
        )
        test.assertEqual(float(row_w.numpy()[0, 0]), 0.5)


def test_compliance_parameters_removed(test: unittest.TestCase, device):
    """The no-op dense compliance knobs are gone; passing them fails loudly."""
    names = tuple(inspect.signature(newton.solvers.SolverFeatherPGS).parameters)
    test.assertNotIn("dense_contact_compliance", names)
    test.assertNotIn("speculative_dense_contact_compliance", names)
    start = names.index("pgs_beta")
    test.assertEqual(names[start : start + 4], ("pgs_beta", "pgs_cfm", "pgs_omega", "pgs_contact_regularization"))
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05)
        model = builder.finalize()
        with test.assertRaises(TypeError):
            newton.solvers.SolverFeatherPGS(model, dense_contact_compliance=1.0e-4)


def test_restitution_rows_stay_rigid(test: unittest.TestCase, device):
    """A row whose rebound target fires is solved rigid, so the rebound is
    e * v_in whatever the regularizer. Without the exemption the regularized
    fixed point would be (e - g)/(1 + g) * v_in and vanish at g = e."""

    def rebound(g):
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.003
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7, restitution=0.8)
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7, restitution=0.8))
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.25), wp.quat_identity()))
        builder.add_shape_sphere(b, radius=0.05, cfg=cfg)
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(
            model,
            reduce_contacts=True,
            rigid_contact_max=32,
            broad_phase="nxn",
            deterministic=True,
            contact_matching="latest",
        )
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverFeatherPGS(
            model,
            pgs_iterations=16,
            pgs_contact_regularization=g,
            pgs_warmstart=True,
        )
        s0, s1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        max_up = 0.0
        for _k in range(90):
            pipeline.collide(s0, contacts)
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, DT)
            s0, s1 = s1, s0
            max_up = max(max_up, float(s0.body_qd.numpy()[b][2]))
        return max_up

    with wp.ScopedDevice(device):
        v_ref = rebound(0.0)
        test.assertGreater(v_ref, 1.0, "no bounce measured")
        for g in (0.02, 0.5):
            ratio = rebound(g) / v_ref
            test.assertAlmostEqual(ratio, 1.0, delta=0.05, msg=f"g={g}: rebound ratio {ratio:.3f}")


def test_regularization_takes_the_weighted_tiers(test: unittest.TestCase, device):
    """Admit the parallel tiers and keyed rows to a regularized solver as to a rigid one, under either friction law.

    The tiers' Jacobi law steps each contact row by its weight, the serial sweeps' row law, and keyed rows write the
    weights, so the worlds they own keep ``pgs_contact_regularization``. Admission is decided at construction on
    every device; only the matrix-free route, which needs CUDA, builds the tiers it admits. A test pin (``tiers``)
    asks for them on this scene.
    """
    with wp.ScopedDevice(device), mock.patch.dict(newton.solvers.SolverFeatherPGS._route_pins, {"tiers": 32}):
        model = _build_scissor_model(device)
        for beta in (0.0, 0.2):
            records = {}
            for g in (0.0, 0.02):
                solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=beta, pgs_contact_regularization=g)
                routes = solver.get_path_selection()["routes"]
                with test.subTest(friction_anchor_beta=beta, g=g):
                    test.assertNotIn("parallel matrix-free tiers", routes["friction"]["reason"])
                records[g] = (
                    routes["contact_rows"]["keyed_rows"]["enabled"],
                    (routes["sweep"]["tiers"] or {}).get("family"),
                    solver.placement.ink_rows,
                )
            with test.subTest(friction_anchor_beta=beta):
                test.assertEqual(records[0.02], records[0.0])


class TestFeatherPGSRegularization(unittest.TestCase):
    pass


add_function_test(
    TestFeatherPGSRegularization,
    "test_regularization_fixed_points",
    test_regularization_fixed_points,
    devices=get_test_devices(),
)
add_function_test(
    TestFeatherPGSRegularization,
    "test_regularization_documented_sag",
    test_regularization_documented_sag,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestFeatherPGSRegularization,
    "test_regularization_cpu_route_documented_sag",
    test_regularization_cpu_route_documented_sag,
    devices=["cpu"],
)
add_function_test(
    TestFeatherPGSRegularization,
    "test_regularization_indeterminate_split",
    test_regularization_indeterminate_split,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestFeatherPGSRegularization,
    "test_regularization_validation",
    test_regularization_validation,
    devices=get_test_devices(),
)
add_function_test(
    TestFeatherPGSRegularization,
    "test_zero_regularization_shares_one_weight_slot",
    test_zero_regularization_shares_one_weight_slot,
    devices=get_test_devices(),
)
add_function_test(
    TestFeatherPGSRegularization,
    "test_zero_regularization_split_runtime_uses_no_weights",
    test_zero_regularization_split_runtime_uses_no_weights,
    devices=get_test_devices(),
)
add_function_test(
    TestFeatherPGSRegularization,
    "test_exact_surface_contact_is_regularized",
    test_exact_surface_contact_is_regularized,
    devices=get_test_devices(),
)
add_function_test(
    TestFeatherPGSRegularization,
    "test_compliance_parameters_removed",
    test_compliance_parameters_removed,
    devices=get_test_devices(),
)
add_function_test(
    TestFeatherPGSRegularization,
    "test_restitution_rows_stay_rigid",
    test_restitution_rows_stay_rigid,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestFeatherPGSRegularization,
    "test_regularization_takes_the_weighted_tiers",
    test_regularization_takes_the_weighted_tiers,
    devices=["cpu"],
)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
