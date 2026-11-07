# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Integration checks for the sparse dynamics and contact path."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.core.memory import arch_budget
from newton._src.solvers.feather_pgs import sweep_routes_sparse
from newton._src.solvers.feather_pgs.row_laws import ROW_CONTACT, ROW_JOINT_LIMIT, ROW_JOINT_VELOCITY_LIMIT
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_law_harness import unpacked_rows, walk_inverse

DT = 1.0 / 400.0
STATE_FIELDS = ("joint_q", "joint_qd", "body_q", "body_qd")
# Worlds and revolute links of the floating chain (_build_chain), read when it is built.
CHAIN_WORLDS = 4
CHAIN_LINKS = 6


def _make_floating_chain_world(*, base_z: float, contact_sphere: str | None) -> newton.ModelBuilder:
    """A floating box base (free joint) trailing CHAIN_LINKS revolute links.

    ``contact_sphere`` places a single collision sphere on the base (``"base"``) or on the
    deepest link (``"deep"``); all other shapes carry no collision geometry, so each world
    produces at most one contact point. ``None`` builds density-only box shapes with
    collision enabled.
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.mu = 0.7

    no_collision = builder.default_shape_cfg.copy()
    no_collision.has_shape_collision = False
    no_collision.collision_group = 0

    base = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, base_z), wp.quat_identity()))
    cfg = builder.default_shape_cfg if contact_sphere is None else no_collision
    builder.add_shape_box(base, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
    if contact_sphere == "base":
        builder.add_shape_sphere(base, radius=0.1)
    joints = [builder.add_joint_free(base)]
    prev = base
    for i in range(CHAIN_LINKS):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.12, hy=0.03, hz=0.03, cfg=cfg)
        if contact_sphere == "deep" and i == CHAIN_LINKS - 1:
            builder.add_shape_sphere(link, radius=0.05)
        offset = 0.22 if i == 0 else 0.12
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(wp.vec3(offset, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.12, 0.0, 0.0), wp.quat_identity()),
            )
        )
        prev = link
    builder.add_articulation(joints)
    return builder


def _build_chain(device: str, *, base_z: float, contact_sphere: str | None, ground: bool) -> newton.Model:
    """Replicate the floating chain in CHAIN_WORLDS worlds, optionally over a ground plane."""
    scene = newton.ModelBuilder()
    for _ in range(CHAIN_WORLDS):
        scene.add_world(_make_floating_chain_world(base_z=base_z, contact_sphere=contact_sphere))
    if ground:
        scene.add_ground_plane()
    return scene.finalize(device=device)


def _build_model(
    *, free_count=0, robot_count=1, prescribed=False, heterogeneous=False, free_heights=None, device="cuda:0"
):
    """Replicate a contacting tripod, optionally surrounded by separate free bodies.

    Native sparse factors decline worlds with free rigid bodies, so ``free_count`` builds
    models that keep the dense route. ``heterogeneous`` puts one tripod world beside a world
    holding a single free sphere instead of replicating the tripod world. ``free_heights``
    builds a world per height instead, its free bodies' centers at that height.
    """
    if free_heights is not None:
        builder = newton.ModelBuilder()
        for height in free_heights:
            builder.add_world(_tripod_env(free_count, robot_count, prescribed, height))
        if not prescribed:
            builder.add_ground_plane()
        return builder.finalize(device=device)
    env = _tripod_env(free_count, robot_count, prescribed, 0.05)
    builder = newton.ModelBuilder()
    if heterogeneous:
        builder.add_world(env)
        free_world = newton.ModelBuilder()
        body = free_world.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()))
        free_world.add_shape_sphere(body, radius=0.05)
        builder.add_world(free_world)
    else:
        builder.replicate(env, 2)
    if not prescribed:
        builder.add_ground_plane()
    return builder.finalize(device=device)


def _tripod_env(free_count, robot_count, prescribed, free_height):
    """One world of :func:`_build_model`: the tripods and the free bodies, their centers at ``free_height``."""
    env = newton.ModelBuilder()
    if prescribed:
        support = env.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, -0.02), wp.quat_identity()), is_kinematic=True)
        env.add_shape_box(support, hx=0.6, hy=0.4, hz=0.02)
        env.joint_qd[:6] = [0.2, 0.0, 0.04, 0.0, 0.0, 0.15]

    for robot in range(robot_count):
        robot_q_start = len(env.joint_q)
        root = env.add_link(
            xform=wp.transform(wp.vec3(robot * 0.105, 0, 0.16), wp.quat_identity()),
            mass=2.0,
            inertia=wp.mat33(np.eye(3) * 0.03),
        )
        joints = [env.add_joint_free(root)]
        for x, y in ((0.16, 0.0), (-0.08, 0.14), (-0.08, -0.14)):
            child = env.add_link(mass=0.5, inertia=wp.mat33(np.eye(3) * 0.002))
            cfg = env.default_shape_cfg.copy()
            cfg.density, cfg.mu = 0.0, 0.6
            env.add_shape_sphere(child, radius=0.055, cfg=cfg)
            joints.append(
                env.add_joint_revolute(
                    root,
                    child,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform(wp.vec3(x, y, -0.11), wp.quat_identity()),
                    target_pos=0.05,
                    target_ke=15.0,
                    target_kd=1.0,
                    armature=0.05,
                    limit_lower=-0.2,
                    limit_upper=0.2,
                    effort_limit=10.0,
                )
            )
        env.add_articulation(joints)
        env.joint_q[robot_q_start + 7] = 0.23  # Exercise an actual unilateral limit on the first step.
    for index in range(free_count):
        body = env.add_link(
            xform=wp.transform(
                wp.vec3(0.26 + index * 0.095, 0.0, free_height),
                wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 0.4, 0.5)), 0.6 + index * 0.2),
            ),
            mass=0.4 + index * 0.2,
            com=wp.vec3(0.01, -0.015, 0.007),
            inertia=wp.mat33(np.diag([0.003, 0.004, 0.005])),
        )
        cfg = env.default_shape_cfg.copy()
        cfg.density, cfg.mu = 0.0, 0.6
        env.add_shape_sphere(body, radius=0.05, cfg=cfg)
        env.add_articulation([env.add_joint_free(body)])
    return env


def _over_budget_rows(model) -> int:
    """The smallest dense row capacity whose sparse sweep outgrows its device's static budget.

    The working set is the one the admission sizes (:attr:`.sweep_routes_sparse.SparseSweep.staging`), for the
    sweep a small capacity admits.
    """
    probe = _make_solver(model, True)

    def staging(rows):
        return sweep_routes_sparse.SparseSweep(
            rows,
            probe.forest.max_world_dofs,
            probe.spec.classes.packed.max_support,
            active_rows=probe.spec.classes.packed.sweep["active_rows"],
            linked_rows=True,
        ).staging

    budget = arch_budget(model.device.arch)
    low, high = 1, 1 << 16
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (middle, high) if staging(middle).fits(budget) else (low, middle)
    return high


def _make_solver(model, sparse, *, pins=None, **overrides):
    """Change only the representation selection between comparison solvers (and the test-only route ``pins``)."""
    options = {
        "pgs_iterations": 32,
        "enable_joint_limits": True,
        "enable_joint_velocity_limits": False,
        "pgs_warmstart": False,
        "dense_max_constraints": 32,
        "update_mass_matrix_interval": 4,
    }
    options.update(overrides)
    pins = {**({} if sparse else {"storage": "dense"}), **(pins or {})}
    return SolverFeatherPGS(model, pins=SolverFeatherPGS.Pins(**pins), **options)


def _case(sparse, *, robot_count=1, prescribed=False, pins=None):
    """Allocate a stable input state for eager and captured cache reuse."""
    model = _build_model(robot_count=robot_count, prescribed=prescribed)
    # Row-level comparisons need every world's rows: the resolved-world classifier, which skips the
    # rows of worlds that all separate, stays off on the dense reference.
    solver = _make_solver(
        model,
        sparse,
        pins={"claims": (), **(pins or {})},
        **({"dense_max_constraints": 96, "mf_max_constraints": 64} if robot_count > 1 else {}),
    )
    state, out = model.state(), model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    return model, solver, state, out, model.control(), pipeline, pipeline.contacts()


def _step(case):
    """Advance without swapping the input object used by the dynamics cache."""
    _, solver, state, out, control, pipeline, contacts = case
    state.clear_forces()
    pipeline.collide(state, contacts)
    solver.step(state, out, control, contacts, DT)
    for name in STATE_FIELDS:
        wp.copy(getattr(state, name), getattr(out, name))


def _tree(case, group: int, records: np.ndarray) -> np.ndarray:
    """The walk records of the sparse group's ``group``-th articulation."""
    solver = case[1]
    art = int(solver.forest.group_to_art[solver.spec.classes.packed_size].numpy()[group])
    d0 = int(solver.forest.articulation_dof_start.numpy()[art])
    return records[d0 : d0 + int(solver.forest.articulation_dof_count.numpy()[art])]


@unittest.skipUnless(wp.is_cuda_available(), "Sparse integrated solve requires CUDA")
class TestFeatherPGSSparseSolver(unittest.TestCase):
    def test_shared_row_support_matches_explicit_indices(self):
        """Remove per-world indices while preserving contacts, limits and reset behavior."""
        explicit = _case(True, pins={"row_support": False})
        shared = _case(True)
        self.assertFalse(explicit[1].placement.sparse_row_support.enabled)
        self.assertTrue(shared[1].placement.sparse_row_support.enabled)
        self.assertEqual(shared[1].placement.sparse_row_dof.shape, (1, 1, 1))
        for step in range(8):
            for case in (explicit, shared):
                if step == 4:
                    model, solver, state, *_ = case
                    q = state.joint_q.numpy()
                    q[2] += 0.002
                    state.joint_q.assign(q)
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                    solver.reset(state, wp.array([True, False], dtype=bool, device=model.device))
                _step(case)
            self.assert_states_close(explicit[2], shared[2])
            self.assert_sparse_state(shared)
            counts = explicit[1].placement.constraint_count.numpy()
            np.testing.assert_array_equal(counts, shared[1].placement.constraint_count.numpy())
            support = shared[1].placement.sparse_row_support
            classes, indices = support.row_class.numpy(), support.class_dof.numpy()
            for world, count in enumerate(counts):
                np.testing.assert_array_equal(
                    indices[classes[world, :count]], explicit[1].placement.sparse_row_dof.numpy()[world, :count]
                )

    def test_moving_prescribed_supports_keep_dense_factors(self):
        """Keep moving prescribed supports on dense factors: native sparse factors are the held tree's, whose
        sparse group holds no prescribed articulations."""
        _, solver, *_ = _case(True, prescribed=True)
        self.assertTrue(solver.forest.has_prescribed_response)
        self.assertIsNone(solver.spec.classes.packed_size)
        self.assertIn("holds no prescribed articulations", solver._plan.route.reasons["packed"])

    def assert_states_close(self, first, second):
        """Allow float32 factor-order roundoff, not a changed physics recipe."""
        for name in STATE_FIELDS:
            # Independent elimination/reduction orders accumulate over 32 contact steps.
            tolerance = 2.0e-3 if name.endswith("qd") else 2.0e-4
            np.testing.assert_allclose(
                getattr(first, name).numpy(), getattr(second, name).numpy(), rtol=2.0e-4, atol=tolerance, err_msg=name
            )

    def assert_sparse_state(self, case):
        """Check storage elimination, row support, impulses, and residual coordinates."""
        _, solver, state, *_ = case
        size = solver.spec.classes.packed_size
        self.assertEqual(size, 9)
        self.assertEqual(solver.placement.J_by_size[size].shape, (1, 1, 1))  # Only an argument stand-in remains.
        self.assertFalse(solver.placement.constraint_overflow.numpy().any())
        for name in STATE_FIELDS:
            self.assertTrue(np.isfinite(getattr(state, name).numpy()).all(), name)
        counts, impulses = solver.placement.constraint_count.numpy(), solver.placement.impulses.numpy()
        incidence, rhs = solver.placement.row_incident.numpy(), solver.placement.rhs.numpy()
        v_hat, v_out = solver.v_hat.numpy(), solver.v_out.numpy()
        for world in range(solver.forest.world_count):
            count = int(counts[world])
            self.assertLessEqual(count, solver.placement.dense_max_constraints)
            self.assertTrue(np.isfinite(impulses[world, :count]).all())
            kinds = solver.placement.row_type.numpy()[world, :count]
            unilateral = (kinds == ROW_CONTACT) | (kinds == ROW_JOINT_LIMIT)
            self.assertTrue(np.all(impulses[world, :count][unilateral] >= -1.0e-7))
            z, physical_j = self._sparse_rows_physical(solver, world, count)
            velocity = solver.forest.world_dof_indices.numpy()[world]
            np.testing.assert_allclose(physical_j @ v_hat[velocity], incidence[world, :count], atol=2.0e-5, rtol=2.0e-5)
            factor_residual = (
                z @ solver.placement.sparse_factor_velocity_delta.numpy()[world]
                + incidence[world, :count]
                + rhs[world, :count]
            )
            np.testing.assert_allclose(
                physical_j @ v_out[velocity] + rhs[world, :count], factor_residual, atol=3.0e-5, rtol=3.0e-5
            )

    def _sparse_rows_physical(self, solver, world, count):
        """Recover physical Jacobians independently from the stored inverse factors."""
        plan = solver.factor.sparse_pattern
        size = plan.dof_count
        row_dof, values = solver.placement.sparse_row_dof.numpy(), solver.placement.sparse_row_factor.numpy()
        shared = solver.placement.sparse_row_support
        if shared.enabled:
            row_dof = shared.class_dof.numpy()[shared.row_class.numpy()[world, :count]][None, :, :]
        z = np.zeros((count, solver.forest.max_world_dofs))
        for row in range(count):
            support = row_dof[0 if shared.enabled else world, row]
            valid = support >= 0
            self.assertTrue(np.all(support[valid] < solver.forest.max_world_dofs))
            self.assertEqual(len(set(support[valid])), int(valid.sum()))
            z[row, support[valid]] = values[world, row, valid]
        physical_j = np.zeros_like(z)
        for art in solver.forest.group_to_art[size].numpy():
            if solver.forest.art_to_world.numpy()[art] != world:
                continue
            inverse = walk_inverse(solver, int(art))[np.ix_(plan.permutation, plan.permutation)]
            offset = int(solver.forest.articulation_world_dof_offset.numpy()[art])
            physical_j[:, offset + plan.permutation] = np.linalg.solve(inverse, z[:, offset : offset + size].T).T
        if solver.forest.has_free_rigid_bodies:
            for art in solver.forest.group_to_art[6].numpy():
                if solver.forest.art_to_world.numpy()[art] == world:
                    offset = int(solver.forest.articulation_world_dof_offset.numpy()[art])
                    physical_j[:, offset : offset + 6] = z[:, offset : offset + 6]
        return z, physical_j

    def _order_multi_articulation_rows(self, case):
        """Give this fixture identical semantic row order without changing production allocation."""
        _, solver, _, _, _, _, contacts = case
        sparse = solver.spec.classes.packed_size is not None
        names = [
            "rhs",
            "diag",
            "impulses",
            "row_type",
            "row_parent",
            "row_mu",
            "row_beta",
            "phi",
            "target_velocity",
            "row_restitution",
            "row_incident",
        ]
        # The dense route's rows are packed (packed_rows): they move with their row slots.
        packed = ["packed_Z", "row_sides"]
        names += ["sparse_row_dof", "sparse_row_factor"] if sparse else packed
        arrays = {name: getattr(solver.placement, name).numpy() for name in names}
        dense_jacobian = None if sparse else unpacked_rows(solver)[0]
        worlds, slots, needed = (
            getattr(solver.placement, name).numpy()
            for name in ("contact_world", "contact_slot", "contact_slots_needed")
        )
        shape_a, shape_b = contacts.rigid_contact_shape0.numpy(), contacts.rigid_contact_shape1.numpy()
        identities = []
        for world, count in enumerate(solver.placement.constraint_count.numpy()):
            jacobian = self._sparse_rows_physical(solver, world, count)[1] if sparse else dense_jacobian[world, :count]
            bundles = []
            prefix = np.isin(
                arrays["row_type"][world, :count],
                [ROW_JOINT_LIMIT, ROW_JOINT_VELOCITY_LIMIT],
            )
            for row in np.flatnonzero(prefix):
                dof = int(np.argmax(np.abs(jacobian[row])))
                bundles.append(((0, int(arrays["row_type"][world, row]), dof, int(np.sign(jacobian[row, dof]))), [row]))
            active = [
                c for c in range(int(contacts.rigid_contact_count.numpy()[0])) if worlds[c] == world and slots[c] >= 0
            ]
            for c in active:
                bundles.append(((1, int(shape_a[c]), int(shape_b[c])), list(range(slots[c], slots[c] + needed[c]))))
            bundles.sort(key=lambda item: item[0])
            keys = [key for key, _ in bundles]
            self.assertEqual(len(keys), len(set(keys)), "Fixture must have unique limits and shape-pair contacts")
            order = np.array([row for _, rows in bundles for row in rows], dtype=int)
            np.testing.assert_array_equal(np.sort(order), np.arange(count))
            inverse = np.argsort(order)
            for values in arrays.values():
                values[world, :count] = values[world, order]
            parent = arrays["row_parent"][world, :count]
            parent[parent >= 0] = inverse[parent[parent >= 0]]
            slots[active] = inverse[slots[active]]
            identities.append([(key, len(rows)) for key, rows in bundles])
        for name, values in arrays.items():
            getattr(solver.placement, name).assign(values)
        solver.placement.contact_slot.assign(slots)
        return identities

    def test_two_articulations_share_contact_solve(self):
        """Distinguish factor groups from worlds when two branched robots contact each other.

        The dense route sweeps every row, so the sparse arm keeps patch friction's
        full ordered sweep and the comparison isolates the representation. On this
        fixture the active-row law alone moves joint velocities by up to 0.07 for
        either friction law (test_feather_pgs_sparse_patch_rows covers it).
        """
        dense = _case(False, robot_count=2)
        sparse = _case(True, robot_count=2, pins={"active_rows": False})
        self.assertFalse(sparse[1].spec.classes.packed.sweep["active_rows"])
        self.assertEqual(sparse[1].spec.classes.packed_size, 9)
        self.assertEqual(sparse[1].forest.group_to_art[9].shape[0], 4)
        self.assertEqual(sparse[1].forest.world_count, 2)
        contact_seen = False
        for _ in range(16):
            row_orders = []
            for case in (dense, sparse):
                solve = case[1].sweep.solve

                def ordered_solve(*args, _case=case, _solve=solve, _orders=row_orders, **kwargs):
                    # Both production GPU paths solve the same semantic order;
                    # atomic allocation order is not a physics-equivalence invariant.
                    _orders.append(self._order_multi_articulation_rows(_case))
                    return _solve(*args, **kwargs)

                with mock.patch.object(case[1].sweep, "solve", side_effect=ordered_solve):
                    _step(case)
            self.assertEqual(len(row_orders), 2)
            self.assertEqual(*row_orders)
            self.assert_states_close(dense[2], sparse[2])
            model, solver, _, _, _, _, contacts = sparse
            active = int(contacts.rigid_contact_count.numpy()[0])
            shape_body, body_art = model.shape_body.numpy(), solver.forest.body_to_articulation.numpy()
            for shape_a, shape_b in zip(
                contacts.rigid_contact_shape0.numpy()[:active],
                contacts.rigid_contact_shape1.numpy()[:active],
                strict=True,
            ):
                body_a, body_b = shape_body[shape_a], shape_body[shape_b]
                contact_seen |= body_a >= 0 and body_b >= 0 and body_art[body_a] != body_art[body_b]
        self.assertTrue(contact_seen, "Fixture must couple separate responsive articulations")
        self.assert_sparse_state(sparse)

    def test_trajectory_masked_reset_and_inertial_notify(self):
        """Match 32 contact steps and refresh only the requested cached factors."""
        dense, sparse = _case(False), _case(True)
        self.assertIsNone(dense[1].spec.classes.packed_size)
        self.assertEqual(sparse[1].spec.classes.packed_size, 9)
        contact_seen = limit_seen = False
        for _ in range(32):
            for case in (dense, sparse):
                _step(case)
            contact_seen |= bool(sparse[-1].rigid_contact_count.numpy()[0])
            limit_seen |= bool(np.any(sparse[1].placement.row_type.numpy() == ROW_JOINT_LIMIT))
            self.assert_states_close(dense[2], sparse[2])
        self.assertTrue(contact_seen and limit_seen, "Fixture must exercise contacts and joint limits")
        self.assert_sparse_state(sparse)
        for case in (dense, sparse):
            _step(case)  # Step32 refreshes globally; step33 is a held-mass step.
        held = sparse[1].factor.walk_data.numpy().copy()
        for model, solver, state, *_ in (dense, sparse):
            q = state.joint_q.numpy()
            q[2] += 0.02
            state.joint_q.assign(q)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            solver.reset(state, wp.array([True, False], dtype=bool, device=model.device))
        for case in (dense, sparse):
            _step(case)
        np.testing.assert_array_equal(sparse[1].factor.mass_update_mask.numpy(), [1, 0])
        np.testing.assert_array_equal(_tree(sparse, 1, sparse[1].factor.walk_data.numpy()), _tree(sparse, 1, held))
        self.assert_states_close(dense[2], sparse[2])
        for model, solver, *_ in (dense, sparse):
            mass = model.body_mass.numpy()
            mass[0] *= 1.1
            model.body_mass.assign(mass)
            solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        for case in (dense, sparse):
            _step(case)
        np.testing.assert_array_equal(sparse[1].factor.mass_update_mask.numpy(), [1, 1])
        self.assertFalse(np.array_equal(_tree(sparse, 0, sparse[1].factor.walk_data.numpy()), _tree(sparse, 0, held)))
        self.assert_states_close(dense[2], sparse[2])
        self.assert_sparse_state(sparse)

    def test_graph_replay_preserves_sparse_problem(self):
        """Replay a complete four-step mass-refresh cadence without dense scratch."""
        eager, graph = _case(True), _case(True)
        for case in (eager, graph):
            _step(case)  # Finish allocation/compilation outside capture.
        with wp.ScopedCapture("cuda:0") as capture:
            for _ in range(4):
                _step(graph)
        for _ in range(8):
            wp.capture_launch(capture.graph)
            for _ in range(4):
                _step(eager)
        self.assert_states_close(eager[2], graph[2])
        self.assert_sparse_state(graph)

    def test_unsupported_configurations_keep_existing_path(self):
        """Reject full-support chains and oversized dense rows, and take contact torsion's spin rows.

        Joint velocity limits are not among them: they compose with sparse
        factors under either friction law (the route probe records it).
        """
        chain = _build_chain("cuda:0", base_z=1.2, contact_sphere="deep", ground=False)
        self.assertIsNone(_make_solver(chain, True).spec.classes.packed_size)
        model = _build_model()
        self.assertIsNotNone(_make_solver(model, True, contact_torsion=True).spec.classes.packed_size)
        for options in ({"dense_max_constraints": _over_budget_rows(model)},):
            with self.subTest(options=options):
                solver = _make_solver(model, True, **options)
                self.assertIsNone(solver.spec.classes.packed_size)
                # The existing path: the matrix-free sweeps, whose producers write the packed rows.
                self.assertTrue(solver.spec.classes.mf_sweeps)
                self.assertGreater(solver.placement.packed_Z.size, 1)

    def test_warm_start_is_admitted(self):
        """Warm start seeds the sparse sweeps from carried impulses on models without free rigid bodies."""
        self.assertEqual(_make_solver(_build_model(), True, pgs_warmstart=True).spec.classes.packed_size, 9)


if __name__ == "__main__":
    unittest.main()
