# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import execution_plan, sweep_routes, tree_passes
from newton._src.solvers.feather_pgs import kernels as feather_pgs_kernels
from newton._src.solvers.feather_pgs.contact_row_plan import contact_row_plan
from newton._src.solvers.feather_pgs.solver_feather_pgs import (
    _DENSE_META_MAX_PARENT,
    _DENSE_META_ROW_TYPE_MASK,
    _estimate_cholesky_shared_memory,
    _estimate_hinv_jt_shared_memory,
    _FeatherPGSExecutionPlan,
    _select_hinv_jt_chunk_size,
    _validate_dense_metadata_encoding,
)
from newton.solvers import SolverFeatherPGS


def _build_chain_model(num_links=3, num_worlds=2, *, with_free_body=False, device=None):
    chain = newton.ModelBuilder()
    hx = 0.3
    joints = []
    parent = -1
    root_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.45 * wp.pi)
    for _ in range(num_links):
        link = chain.add_link()
        chain.add_shape_box(link, hx=hx - 0.08, hy=0.05, hz=0.05)
        if parent == -1:
            parent_xform = wp.transform(p=wp.vec3(0.0, 0.0, 2.5), q=root_rot)
        else:
            parent_xform = wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity())
        joints.append(
            chain.add_joint_revolute(
                parent=parent,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=parent_xform,
                child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
            )
        )
        parent = link
    chain.add_articulation(joints)
    if with_free_body:
        body = chain.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        chain.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        chain.add_articulation([chain.add_joint_free(parent=-1, child=body)])
    main = newton.ModelBuilder()
    main.replicate(chain, num_worlds, spacing=(3.0, 3.0, 0.0))
    return main.finalize(device=device)


def _build_fixed_base_star_model(num_branches=16, num_worlds=2):
    star = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    base = star.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joints = [star.add_joint_fixed(parent=-1, child=base)]
    for branch in range(num_branches):
        child = star.add_link(mass=1.0 + 0.01 * branch, inertia=wp.mat33(np.eye(3)))
        joints.append(
            star.add_joint_prismatic(
                parent=base,
                child=child,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(0.02 * branch, 0.0, 0.0), wp.quat_identity()),
            )
        )
    star.add_articulation(joints)
    main = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    main.replicate(star, num_worlds, spacing=(2.0, 2.0, 0.0))
    return main.finalize()


def _build_sparse_diagonal_pair_model(num_branches=16, num_worlds=2, *, device=None, revolute_branches=False):
    """Build one independent fixed-base star and one compact serial chain per world.

    ``revolute_branches`` hinges the star branches about Y instead of sliding them along Z, so the branch
    response depends on the link inertia and center of mass rather than on the mass alone.
    """
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    base = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    star_joints = [scene.add_joint_fixed(parent=-1, child=base)]
    add_branch_joint = scene.add_joint_revolute if revolute_branches else scene.add_joint_prismatic
    for branch in range(num_branches):
        child = scene.add_link(mass=1.0 + 0.01 * branch, inertia=wp.mat33(np.eye(3)))
        star_joints.append(
            add_branch_joint(
                parent=base,
                child=child,
                axis=newton.Axis.Y if revolute_branches else newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(0.03 * branch, 0.0, 0.0), wp.quat_identity()),
                limit_lower=-0.1,
                limit_upper=0.1,
            )
        )
    scene.add_articulation(star_joints)

    # The chain stands on a fixed base: the held tree, which alone evaluates the compact diagonal, takes fixed or
    # free roots.
    parent = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    chain_joints = [scene.add_joint_fixed(parent=-1, child=parent)]
    for _link_index in range(3):
        child = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        chain_joints.append(
            scene.add_joint_revolute(
                parent=parent,
                child=child,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
                limit_lower=-0.2,
                limit_upper=0.2,
            )
        )
        parent = child
    scene.add_articulation(chain_joints)

    replicated = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    replicated.replicate(scene, num_worlds, spacing=(3.0, 3.0, 0.0))
    return replicated.finalize(device=device)


def _build_sparse_contact_friction_model(num_branches=16, num_worlds=2, *, device=None):
    """Build the sparse pair model with tilted sliding branches resting on a static ground plane.

    Each branch slides along ``(1, 0, 1) / sqrt(2)`` and carries a small box that touches the ground, so every
    contact normal and tangent row acts on that branch's single coordinate and friction is active.
    """
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    scene.default_shape_cfg.ke = 1.0e5
    scene.default_shape_cfg.kd = 1.0e3
    scene.default_shape_cfg.mu = 0.8
    scene.default_shape_cfg.margin = 0.0
    scene.default_shape_cfg.gap = 0.0
    scene.add_ground_plane()
    base = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    star_joints = [
        scene.add_joint_fixed(
            parent=-1, child=base, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity())
        )
    ]
    for branch in range(num_branches):
        child = scene.add_link(mass=1.0 + 0.01 * branch, inertia=wp.mat33(np.eye(3)))
        # Four boxes keep the four-point patches within the dense row capacity of the test solvers.
        if branch % 4 == 0:
            scene.add_shape_box(child, hx=0.01, hy=0.01, hz=0.01)
        star_joints.append(
            scene.add_joint_prismatic(
                parent=base,
                child=child,
                axis=wp.normalize(wp.vec3(1.0, 0.0, 1.0)),
                parent_xform=wp.transform(wp.vec3(0.05 * branch, 0.0, -0.4905), wp.quat_identity()),
                limit_lower=-0.1,
                limit_upper=0.1,
            )
        )
    scene.add_articulation(star_joints)

    # The chain stands on a fixed base: the held tree, which alone evaluates the compact diagonal, takes fixed or
    # free roots.
    parent = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    chain_joints = [scene.add_joint_fixed(parent=-1, child=parent)]
    for _link_index in range(3):
        child = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        chain_joints.append(
            scene.add_joint_revolute(
                parent=parent,
                child=child,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(0.0, 1.0, 1.0), wp.quat_identity()),
                limit_lower=-0.2,
                limit_upper=0.2,
            )
        )
        parent = child
    scene.add_articulation(chain_joints)

    replicated = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    replicated.replicate(scene, num_worlds, spacing=(3.0, 3.0, 0.0))
    return replicated.finalize(device=device)


def _build_heterogeneous_world_model():
    free_template = newton.ModelBuilder()
    free_body = free_template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    free_joint = free_template.add_joint_free(parent=-1, child=free_body)
    free_template.add_articulation([free_joint])

    slider_template = newton.ModelBuilder()
    slider_body = slider_template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    slider_joint = slider_template.add_joint_prismatic(parent=-1, child=slider_body, axis=newton.Axis.X)
    slider_template.add_articulation([slider_joint])

    builder = newton.ModelBuilder()
    builder.add_world(free_template)
    builder.add_world(slider_template)
    return builder.finalize()


class TestFeatherPGSLaunchConfig(unittest.TestCase):
    def test_research_options_preserve_positional_solver_budget(self):
        """Preserve main's positional budget while adding keyword-only research options."""
        parameters = inspect.signature(SolverFeatherPGS).parameters
        names = list(parameters)
        self.assertEqual(names[names.index("velocity_limit_activation_fraction") + 1], "pgs_iterations")
        for name in ("pgs_solve_readout", "contact_compliance"):
            with self.subTest(name=name):
                self.assertEqual(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_point_and_patch_friction_take_the_masked_builder(self):
        """Build a size of at most 32 DOFs with the masked builder under explicit point friction and patch friction.

        Patch friction takes the masked builder's patch-law variant. The row build
        dispatches on contact_row_plan.
        """
        with wp.ScopedDevice("cpu"):  # the general builder; CUDA admits other producers first
            model = _build_chain_model(num_links=12, num_worlds=1)
        for beta, point in ((None, False), (0.0, True)):
            with self.subTest(beta=beta):
                solver = SolverFeatherPGS(model, friction_anchor_beta=beta)
                self.assertEqual(solver.laws.set.point, point)
                plan = contact_row_plan(solver)
                self.assertEqual(plan.producer, "general")
                self.assertEqual(plan.methods, {12: "masked_combined"})
                self.assertEqual(plan.combined, (12,))

    def test_launch_geometry_kernels_use_dedicated_modules(self):
        """Keep custom-block-dimension kernels out of the general modules, one module per launch block size."""
        expected_modules = {
            feather_pgs_kernels: {
                "eval_rigid_tau_and_augmented_drives": "inverse_dynamics",
            },
            tree_passes: {
                "fk_id": "serial",
                "inverse_dynamics_tau": "serial",
                "composite_inertia": "serial",
                "refresh_body_inertia": "mass_dynamics",
                "crba_fill": "mass_dynamics",
            },
        }
        for owner, kernels in expected_modules.items():
            general_module = owner.__name__
            for kernel_name, module_suffix in kernels.items():
                with self.subTest(kernel=kernel_name):
                    kernel_module = getattr(owner, kernel_name).module.name
                    self.assertNotEqual(kernel_module, general_module)
                    self.assertEqual(kernel_module, f"{general_module}.{module_suffix}")

    def test_defaults_preserved(self):
        model = newton.ModelBuilder().finalize()
        solver = SolverFeatherPGS(model)
        self.assertEqual(solver.tile_threads, 64)

    def test_hinv_chunk_selection_accounts_for_tile_threads(self):
        """Include tile scratch space when selecting response chunks."""
        self.assertEqual(_select_hinv_jt_chunk_size(50, 384, 30000, 64), 32)
        self.assertEqual(_select_hinv_jt_chunk_size(50, 384, 30000, 256), 16)

    def test_hinv_chunk_selection_caps_response_tiles(self):
        """Cap response chunks at 32 rows across articulation sizes."""
        self.assertEqual(_select_hinv_jt_chunk_size(20, 384, 101376, 64), 32)
        self.assertEqual(_select_hinv_jt_chunk_size(21, 384, 101376, 64), 32)

    def test_dense_metadata_encoding_bounds(self):
        _validate_dense_metadata_encoding(32)
        self.assertGreaterEqual(_DENSE_META_ROW_TYPE_MASK + 1, 5)
        with self.assertRaisesRegex(ValueError, "packed parent capacity"):
            _validate_dense_metadata_encoding(_DENSE_META_MAX_PARENT + 2)

    def test_hinv_execution_plan_falls_back_or_rejects_safely(self):
        fallback = _FeatherPGSExecutionPlan.build(
            [160],
            max_constraints=384,
            max_shared_memory=101376,
            cholesky_kernel="auto",
            hinv_jt_kernel="auto",
            small_dof_threshold=12,
            tile_threads=64,
        )
        self.assertFalse(fallback.use_tiled_hinv_jt(160))
        self.assertIn("shared memory", fallback.selection_reasons(160)["hinv_jt_kernel"])

        zero_capacity = _FeatherPGSExecutionPlan.build(
            [23],
            max_constraints=0,
            max_shared_memory=101376,
            cholesky_kernel="auto",
            hinv_jt_kernel="tiled",
            small_dof_threshold=12,
            tile_threads=64,
        )
        self.assertFalse(zero_capacity.use_tiled_hinv_jt(23))
        self.assertIn("zero row capacity", zero_capacity.selection_reasons(23)["hinv_jt_kernel"])

        with self.assertRaisesRegex(ValueError, "hinv_jt_kernel='tiled'"):
            _FeatherPGSExecutionPlan.build(
                [160],
                max_constraints=384,
                max_shared_memory=101376,
                cholesky_kernel="auto",
                hinv_jt_kernel="tiled",
                small_dof_threshold=12,
                tile_threads=64,
            )

    def test_hinv_estimate_counts_three_response_tiles(self):
        """Count the factor, three response tiles and the Warp scratch of the tiled response."""
        self.assertEqual(_estimate_hinv_jt_shared_memory(27, 32, tile_threads=64), 13552)

    def test_cholesky_execution_plan_respects_shared_memory(self):
        """Fall back for oversized auto tiles and reject forced invalid launches."""
        self.assertEqual(_estimate_cholesky_shared_memory(108), 140400)
        automatic = _FeatherPGSExecutionPlan.build(
            [23, 108],
            max_constraints=0,
            max_shared_memory=101376,
            cholesky_kernel="auto",
            hinv_jt_kernel="auto",
            small_dof_threshold=12,
            tile_threads=64,
        )
        self.assertTrue(automatic.use_tiled_cholesky(23))
        self.assertFalse(automatic.use_tiled_cholesky(108))
        self.assertIn("140400", automatic.selection_reasons(108)["cholesky_kernel"])
        self.assertIn("101376", automatic.selection_reasons(108)["cholesky_kernel"])

        with self.assertRaisesRegex(ValueError, "cholesky_kernel='tiled'.*140400.*101376"):
            _FeatherPGSExecutionPlan.build(
                [108],
                max_constraints=0,
                max_shared_memory=101376,
                cholesky_kernel="tiled",
                hinv_jt_kernel="auto",
                small_dof_threshold=12,
                tile_threads=64,
            )

    def test_fixed_base_sibling_branches_detect_diagonal_mass_topology(self):
        """Detect independent fixed-base branches on every supported device."""
        solver = SolverFeatherPGS(_build_fixed_base_star_model(), enable_joint_limits=False, dense_max_constraints=16)
        self.assertEqual(solver.forest.diagonal_mass_sizes, frozenset((16,)))

    @unittest.skipUnless(wp.is_cuda_available(), "diagonal mass execution-path parity requires CUDA")
    def test_fixed_base_sibling_branches_use_diagonal_mass_path(self):
        """Match the dense reference while selecting independent branch solves."""
        model = _build_fixed_base_star_model()
        optimized = SolverFeatherPGS(model, enable_joint_limits=False, dense_max_constraints=16)
        self.assertEqual(optimized.forest.diagonal_mass_sizes, frozenset((16,)))
        self.assertTrue(optimized._execution_plan.use_diagonal_mass(16))
        # No dense factor or triangular-solve kernel: the held tree factors in stage 1, or the diagonal route.
        ordinary = optimized.factor.ordinary
        if ordinary is not None:
            self.assertIsNone(ordinary.cholesky_kernels_by_size[16])
            self.assertIsNone(ordinary.triangular_solve_kernels_by_size[16])
        self.assertIsNone(optimized.placement.hinv_jt_kernels_by_size[16])

        forced = {"cholesky_kernel": "loop", "trisolve_kernel": "loop", "hinv_jt_kernel": "par_row"}
        with mock.patch.object(SolverFeatherPGS, "_route_pins", forced):
            reference = SolverFeatherPGS(model, enable_joint_limits=False, dense_max_constraints=16)
        self.assertFalse(reference._execution_plan.use_diagonal_mass(16))

        q = np.tile(np.linspace(-0.04, 0.04, 16, dtype=np.float32), 2)
        qd = np.tile(np.linspace(0.1, -0.1, 16, dtype=np.float32), 2)
        trajectories = []
        for solver in (optimized, reference):
            state_0, state_1 = model.state(), model.state()
            state_0.joint_q.assign(q)
            state_0.joint_qd.assign(qd)
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            control = model.control()
            history = []
            for _ in range(5):
                state_0.clear_forces()
                solver.step(state_0, state_1, control, None, 1.0 / 120.0)
                state_0, state_1 = state_1, state_0
                history.append((state_0.joint_q.numpy().copy(), state_0.joint_qd.numpy().copy()))
            trajectories.append(history)

        for diagonal, dense in zip(*trajectories, strict=True):
            np.testing.assert_allclose(diagonal[0], dense[0], rtol=0.0, atol=2.0e-6)
            np.testing.assert_allclose(diagonal[1], dense[1], rtol=0.0, atol=2.0e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "sparse diagonal response requires CUDA")
    def test_sparse_diagonal_response_matches_dense_joint_limits(self):
        """Match dense position-limit rows while removing the wide response storage."""
        model = _build_sparse_diagonal_pair_model(device="cuda:0")
        optimized = SolverFeatherPGS(
            model,
            dense_max_constraints=64,
            mf_max_constraints=32,
            pgs_iterations=8,
            enable_joint_limits=True,
            # Point friction keeps the contact-triple schedule under test; patch rows are not uniform triples.
            friction_anchor_beta=0.0,
        )
        # The dense reference: the general placement keeps the response and limit rows of every size.
        with mock.patch.dict(SolverFeatherPGS._route_pins, {"placement": "general"}):
            reference = SolverFeatherPGS(
                model,
                dense_max_constraints=64,
                mf_max_constraints=32,
                pgs_iterations=8,
                enable_joint_limits=True,
                # Point friction keeps the contact-triple schedule under test; patch rows are not uniform triples.
                friction_anchor_beta=0.0,
            )

        self.assertTrue(optimized.placement.sparse_diagonal_contact_solve)
        self.assertTrue(optimized.placement.sparse_diagonal_contact_triples)
        self.assertTrue(optimized.placement.sparse_diagonal_speculative_contact_batches)
        self.assertEqual(optimized.forest.direct_diagonal_inertia_sizes, frozenset((16,)))
        self.assertEqual(optimized.placement.sparse_diagonal_response_size, 16)
        self.assertEqual(optimized.placement.sparse_diagonal_dense_size, 3)
        self.assertEqual(optimized.H_by_size[16].shape, (1, 1, 1))
        self.assertEqual(optimized.placement.J_by_size[16].shape, (1, 1, 1))
        self.assertEqual(optimized.placement.J_world.shape, (1, 1, 1))
        self.assertFalse(reference.placement.sparse_diagonal_contact_solve)
        self.assertFalse(reference.placement.sparse_diagonal_contact_triples)

        q = np.tile(np.concatenate((np.full(16, -0.105), np.full(3, 0.205))).astype(np.float32), 2)
        qd = np.tile(np.concatenate((np.full(16, -0.2), np.full(3, 0.1))).astype(np.float32), 2)
        trajectories = []
        original_launch = wp.launch
        cleared_shapes = {id(optimized): [], id(reference): []}

        def checked_launch(kernel, *args, **kwargs):
            if kernel is feather_pgs_kernels.clear_grouped_jacobian_active_rows:
                jacobian = kwargs["outputs"][0]
                group_to_art, _, _, dofs, rows = kwargs["inputs"]
                self.assertEqual(
                    jacobian.shape,
                    (group_to_art.shape[0], rows, dofs),
                    "Dense maintenance must not write the sparse response placeholder",
                )
                cleared_shapes[id(solver)].append(jacobian.shape)
            return original_launch(kernel, *args, **kwargs)

        for solver in (optimized, reference):
            state_in, state_out = model.state(), model.state()
            state_in.joint_q.assign(q)
            state_in.joint_qd.assign(qd)
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            control = model.control()
            history = []
            for _ in range(3):
                state_in.clear_forces()
                # Fail before dispatching an invalid write, so this regression
                # can safely diagnose the old four-byte placeholder bug.
                with mock.patch.object(wp, "launch", side_effect=checked_launch):
                    solver.step(state_in, state_out, control, None, 1.0 / 120.0)
                state_in, state_out = state_out, state_in
                history.append((state_in.joint_q.numpy().copy(), state_in.joint_qd.numpy().copy()))
                if solver is optimized:
                    # Order snapshots after maintenance on its owning stream.
                    with wp.ScopedStream(solver._memset_stream):
                        for buffers in solver.placement.J_bufs:
                            np.testing.assert_array_equal(buffers[16].numpy(), np.zeros((1, 1, 1), dtype=np.float32))
                            self.assertFalse(np.any(buffers[3].numpy()), "Dense Jacobian maintenance was skipped")
            trajectories.append(history)

        self.assertEqual(cleared_shapes[id(optimized)], [(2, 64, 3)] * 3)
        # The reference's matrix-free sweeps read the packed rows, which their producers write whole: no group rows.
        self.assertEqual(cleared_shapes[id(reference)], [])

        first_q, first_qd = trajectories[0][0]
        np.testing.assert_allclose(first_q.reshape(2, 19)[:, :16], -0.104, rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(first_qd.reshape(2, 19)[:, :16], 0.12, rtol=0.0, atol=2.0e-6)
        for sparse, dense in zip(*trajectories, strict=True):
            np.testing.assert_allclose(sparse[0], dense[0], rtol=2.0e-5, atol=2.0e-6)
            np.testing.assert_allclose(sparse[1], dense[1], rtol=2.0e-5, atol=2.0e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "sparse diagonal contact response requires CUDA")
    def test_sparse_diagonal_contact_friction_matches_general_owner(self):
        """Match the general owner's paired friction on sparse contacts with patch and point friction."""
        model = _build_sparse_contact_friction_model(device="cuda:0")

        def make_solver(**kwargs):
            return SolverFeatherPGS(
                model,
                dense_max_constraints=96,
                mf_max_constraints=32,
                pgs_iterations=8,
                enable_joint_limits=True,
                **kwargs,
            )

        def run(solver):
            state_in, state_out = model.state(), model.state()
            joint_qd = state_in.joint_qd.numpy()
            for art in np.flatnonzero(solver.forest.response_dof_count == 16):
                start = int(solver.forest.articulation_dof_start_host[art])
                joint_qd[start : start + 16] = -0.3
            state_in.joint_qd.assign(joint_qd)
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False)
            contacts = pipeline.contacts()
            control = model.control()
            history = []
            for _ in range(4):
                state_in.clear_forces()
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
                state_in, state_out = state_out, state_in
                history.append((state_in.joint_q.numpy().copy(), state_in.joint_qd.numpy().copy()))
            return history

        slick = run(make_solver(enable_contact_friction=False))
        for label, friction_kwargs in (("patch", {}), ("point", {"friction_anchor_beta": 0.0})):
            with self.subTest(friction=label):
                optimized = make_solver(**friction_kwargs)
                with mock.patch.dict(SolverFeatherPGS._route_pins, {"placement": "general"}):
                    reference = make_solver(**friction_kwargs)
                self.assertTrue(optimized.placement.sparse_diagonal_contact_solve)
                self.assertEqual(optimized.laws.friction_anchors_enabled, label == "patch")
                # Point contacts are uniform triples; patch contacts are units of one or three rows.
                self.assertEqual(optimized.placement.sparse_diagonal_contact_triples, label == "point")
                self.assertEqual(optimized.placement.sparse_diagonal_contact_units, label == "patch")
                self.assertTrue(optimized.placement.sparse_diagonal_speculative_contact_batches)
                self.assertEqual(optimized._pgs_solve_sparse_diagonal_kernel.key.endswith("_units"), label == "patch")
                self.assertFalse(reference.placement.sparse_diagonal_contact_solve)

                sparse = run(optimized)
                general = run(reference)
                self.assertGreater(
                    int(optimized.placement.constraint_count.numpy().max()), 0, "no contact rows were generated"
                )
                # Each box's contacts act on its branch's coordinate alone, so they schedule independently.
                # The serial list holds a head per row of capacity for units, and one per three rows for triples.
                self.assertGreater(int(optimized.placement.sparse_contact_group_count.numpy().max()), 0)
                capacity = optimized.placement.dense_max_constraints
                self.assertEqual(
                    optimized.placement.sparse_contact_serial_normals.shape,
                    (2, capacity if label == "patch" else (capacity + 2) // 3),
                )
                for kernel in (
                    optimized._mark_independent_sparse_contact_candidates_kernel,
                    optimized._build_independent_sparse_contact_groups_kernel,
                ):
                    self.assertEqual(kernel.key.endswith("_units"), label == "patch", kernel.key)
                self.assertGreater(
                    float(np.abs(sparse[-1][1] - slick[-1][1]).max()),
                    1.0e-3,
                    "friction did not change the sparse response",
                )
                for step, ((sparse_q, sparse_qd), (general_q, general_qd)) in enumerate(
                    zip(sparse, general, strict=True)
                ):
                    np.testing.assert_allclose(
                        sparse_q, general_q, rtol=2.0e-5, atol=2.0e-6, err_msg=f"joint_q step {step}"
                    )
                    np.testing.assert_allclose(
                        sparse_qd, general_qd, rtol=2.0e-5, atol=2.0e-6, err_msg=f"joint_qd step {step}"
                    )

    @unittest.skipUnless(wp.is_cuda_available(), "sparse diagonal contact response requires CUDA")
    def test_sparse_diagonal_owner_takes_contact_torsion(self):
        """Keep the sparse-diagonal owner and its schedule under contact torsion, which solves the spin rows."""
        model = _build_sparse_diagonal_pair_model(device="cuda:0")
        solvers = {
            torsion: SolverFeatherPGS(
                model,
                dense_max_constraints=64,
                mf_max_constraints=32,
                pgs_iterations=8,
                enable_joint_limits=True,
                contact_torsion=torsion,
            )
            for torsion in (False, True)
        }
        self.assertTrue(solvers[False].placement.sparse_diagonal_contact_solve)
        self.assertTrue(solvers[True].laws.contact_torsion_enabled)
        self.assertTrue(solvers[True].placement.sparse_diagonal_contact_solve)
        self.assertEqual(
            solvers[True].placement.sparse_diagonal_contact_triples,
            solvers[False].placement.sparse_diagonal_contact_triples,
        )

    @unittest.skipUnless(wp.is_cuda_available(), "compact diagonal inertia requires CUDA")
    def test_compact_diagonal_inertia_refreshes_masked_inertial_changes(self):
        """Refresh the compact inertia terms when an inertial model change lands on a mass-reuse step."""
        model = _build_sparse_diagonal_pair_model(device="cuda:0", revolute_branches=True)
        solver = SolverFeatherPGS(
            model,
            dense_max_constraints=64,
            mf_max_constraints=32,
            pgs_iterations=8,
            enable_joint_limits=True,
            update_mass_matrix_interval=4,
        )
        self.assertEqual(solver._compact_diagonal_mass_size, 16)

        # Star branch joints and their child links (skip the fixed root joint of each star).
        articulation_start = model.articulation_start.numpy()
        star_articulations = np.flatnonzero(solver.forest.response_dof_count == 16)
        branch_joints = np.concatenate(
            [np.arange(articulation_start[art] + 1, articulation_start[art + 1]) for art in star_articulations]
        )
        branch_bodies = model.joint_child.numpy()[branch_joints]
        branch_dofs = model.joint_qd_start.numpy()[branch_joints]

        state_in, state_out = model.state(), model.state()
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        control = model.control()

        def step():
            nonlocal state_in, state_out
            state_in.clear_forces()
            solver.step(state_in, state_out, control, None, 1.0 / 120.0)
            state_in, state_out = state_out, state_in
            return solver.factor.diagonal_inverse_mass.numpy()[branch_dofs].copy()

        body_mass = model.body_mass.numpy()
        armature = model.joint_armature.numpy()[branch_dofs]
        # Unit inertia hinged about an axis through the COM: the response is 1 / (1 + armature).
        np.testing.assert_allclose(step(), 1.0 / (1.0 + armature), rtol=1.0e-6, atol=0.0)

        # Double the branch inertia and move the COM 0.1 m off the hinge axis on a mass-reuse step.
        body_inertia = model.body_inertia.numpy()
        body_com = model.body_com.numpy()
        body_inertia[branch_bodies] *= 2.0
        body_com[branch_bodies, 0] += 0.1
        model.body_inertia.assign(body_inertia)
        model.body_com.assign(body_com)
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        self.assertFalse(solver._step % solver.options.update_mass_matrix_interval == 0)
        masked = step()

        # Parallel-axis inertia about the hinge: 2 + m * 0.1^2 (+ armature).
        expected = 1.0 / (2.0 + body_mass[branch_bodies] * 0.1**2 + armature)
        np.testing.assert_allclose(masked, expected, rtol=1.0e-5, atol=0.0)
        # The next global refresh must agree with the masked refresh.
        for _ in range(2):
            step()
        self.assertTrue(solver._step % solver.options.update_mass_matrix_interval == 0)
        np.testing.assert_allclose(step(), masked, rtol=1.0e-6, atol=0.0)

    @unittest.skipUnless(wp.is_cuda_available(), "the CUDA trajectory requires CUDA")
    def test_cuda_ordinary_mass_pipeline_matches_cpu_trajectory(self):
        """Run ordinary dynamics' drives and mass pipeline on CUDA as on the CPU (both pinned to ordinary dynamics)."""
        model = _build_chain_model(num_links=3, num_worlds=2)
        with mock.patch.dict(SolverFeatherPGS._route_pins, {"evaluator": "ordinary"}):
            optimized = SolverFeatherPGS(model, enable_joint_limits=False, dense_max_constraints=16)
            reference = SolverFeatherPGS(
                _build_chain_model(num_links=3, num_worlds=2, device="cpu"),
                enable_joint_limits=False,
                dense_max_constraints=16,
            )
        self.assertIsNone(optimized.factor.held)
        self.assertEqual(optimized._plan.factor[3].name, reference._plan.factor[3].name)
        # Both evaluate the drives with the joint forces in one pass and add K after the assembly.
        self.assertFalse(optimized.factor.parallel_drives)
        self.assertFalse(reference.factor.parallel_drives)

        q = np.tile(np.array([0.15, -0.2, 0.3], dtype=np.float32), 2)
        qd = np.tile(np.array([0.1, -0.05, 0.2], dtype=np.float32), 2)
        trajectories = []
        for solver in (optimized, reference):
            model = solver.model
            state_0, state_1 = model.state(), model.state()
            state_0.joint_q.assign(q)
            state_0.joint_qd.assign(qd)
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            control = model.control()
            control.joint_target_q.assign(np.tile(np.array([0.1, -0.1, 0.2], dtype=np.float32), 2))
            history = []
            for _ in range(5):
                state_0.clear_forces()
                solver.step(state_0, state_1, control, None, 1.0 / 120.0)
                state_0, state_1 = state_1, state_0
                history.append((state_0.joint_q.numpy().copy(), state_0.joint_qd.numpy().copy()))
            trajectories.append(history)

        # The reference runs on the CPU, whose float order differs (velocities about 1e-5 relative after 5 steps).
        for cuda, cpu in zip(*trajectories, strict=True):
            np.testing.assert_allclose(cuda[0], cpu[0], rtol=0.0, atol=5.0e-6)
            np.testing.assert_allclose(cuda[1], cpu[1], rtol=0.0, atol=5.0e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "the matrix-free sweeps require CUDA")
    def test_matrix_free_rows_are_packed_alone(self):
        """Serve a chain's rows, with or without a free body beside it, from the packed rows alone: their build forms
        every diagonal, and no response is written in world layout, aliased to it or reduced by size."""
        for with_free_body in (False, True):
            with self.subTest(with_free_body=with_free_body):
                solver = SolverFeatherPGS(
                    _build_chain_model(num_links=23, num_worlds=1, with_free_body=with_free_body),
                    dense_max_constraints=192,
                )
                self.assertTrue(solver.placement.packed_response)
                self.assertFalse(solver.placement.jy_world_aliased)
                self.assertFalse(solver.placement.hinv_jt_writes_world)
                self.assertEqual(solver.placement.hinv_jt_diag_sizes, frozenset())
                self.assertIsNone(solver.placement.J_world)

    def test_mfgs_metadata_storage_respects_resource_budget(self):
        """Keep resident metadata only when its complete working set fits."""

        def storage(shape, budget):
            staging = functools.partial(sweep_routes.mf_gs_staging, *shape)
            return execution_plan._select_mfgs_storage(staging, budget, block_count=16384, sm_count=188)

        compact = storage((192, 64, 29), 100352).shared_metadata
        large = storage((1024, 4096, 604), 100352).shared_metadata
        overcommitted = storage((192, 64, 29), 4096).shared_metadata

        self.assertTrue(compact)
        self.assertFalse(large)
        self.assertFalse(overcommitted)

    def test_reset_diagnostic_logs_is_deprecated(self):
        """Warn that reset_diagnostic_logs() will be removed, and still clear the pgs_debug logs."""
        solver = SolverFeatherPGS(newton.ModelBuilder().finalize(device="cpu"))
        solver._pgs_convergence_log.append([1.0])
        with self.assertWarnsRegex(DeprecationWarning, "reset_diagnostic_logs"):
            solver.reset_diagnostic_logs()
        self.assertEqual((solver._pgs_convergence_log, solver._pgs_ncp_residual_log), ([], []))

    def test_serial_passes_launch_at_the_serial_block_size(self):
        """A step of ordinary dynamics (pinned) launches every one-articulation-per-thread pass (FK/ID, the joint forces
        and drives, the composite inertia) in blocks of ``tree_passes.SERIAL_BLOCK_DIM`` threads."""
        model = _build_chain_model()
        with mock.patch.dict(SolverFeatherPGS._route_pins, {"evaluator": "ordinary"}):
            solver = SolverFeatherPGS(model, update_mass_matrix_interval=1)
        state_0, state_1, control = model.state(), model.state(), model.control()
        serial = {
            tree_passes.fk_id,
            tree_passes.inverse_dynamics_tau,
            tree_passes.composite_inertia,
            feather_pgs_kernels.eval_rigid_tau_and_augmented_drives,
        }
        blocks = {}
        launch = wp.launch

        def record(kernel, *args, **kwargs):
            if kernel in serial:
                blocks.setdefault(kernel.key, set()).add(kwargs.get("block_dim"))
            return launch(kernel, *args, **kwargs)

        with mock.patch.object(wp, "launch", record):
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        self.assertIn(tree_passes.fk_id.key, blocks)
        self.assertEqual(set().union(*blocks.values()), {tree_passes.SERIAL_BLOCK_DIM}, blocks)


if __name__ == "__main__":
    unittest.main(verbosity=2)
