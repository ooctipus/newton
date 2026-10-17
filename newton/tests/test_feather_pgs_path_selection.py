# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise public automatic choices and forced implementation contracts."""

import inspect
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.path_selection import applied_options
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_patch_parity import assert_history_close, assert_loaded_anchors, standing_model
from newton.tests.test_feather_pgs_sparse_solver import _build_model, _make_solver
from newton.tests.unittest_utils import assert_host_only

STATE_FIELDS = ("joint_q", "joint_qd", "body_q", "body_qd")


def _cpu_model():
    builder = newton.ModelBuilder()
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.add_articulation([builder.add_joint_revolute(-1, body)])
    return builder.finalize(device="cpu")


def _cpu_free_body_model(velocity_limit=None):
    """A one-joint arm and a free box above the ground in one CPU world, optionally with box velocity limits."""
    builder = newton.ModelBuilder()
    limits = {}
    if velocity_limit is not None:
        SolverFeatherPGS.register_custom_attributes(builder)
        limits = {
            "custom_attributes": {
                "rigid_body_max_linear_velocity": velocity_limit,
                "rigid_body_max_angular_velocity": velocity_limit,
            }
        }
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.add_articulation([builder.add_joint_revolute(-1, body)])
    free = builder.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), **limits)
    builder.add_shape_box(free, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()
    return builder.finalize(device="cpu")


def _cpu_separate_worlds_model():
    """The one-joint arm in one world and a free box resting on the ground in another: no world mixes them."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.add_articulation([builder.add_joint_revolute(-1, body)])
    builder.end_world()
    builder.begin_world()
    free = builder.add_body(xform=wp.transform((0.0, 0.0, 0.09), wp.quat_identity()))
    builder.add_shape_box(free, hx=0.1, hy=0.1, hz=0.1)
    builder.end_world()
    builder.add_ground_plane()
    return builder.finalize(device="cpu")


def _graph_and_reset_rollout(model, sparse, cached_dynamics, **options):
    """Step once, then replay a captured contacting step around a partial-world reset."""
    solver = _make_solver(model, sparse, cached_dynamics=cached_dynamics, **options)
    state, output = model.state(), model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model, convex_collision="ordinary")
    contacts, control = pipeline.contacts(), model.control()

    def step():
        pipeline.collide(state, contacts)
        solver.step(state, output, control, contacts, 1.0 / 400.0)
        state.assign(output)

    step()
    with wp.ScopedCapture(device=model.device) as capture:
        step()
    for _ in range(8):
        wp.capture_launch(capture.graph)
    solver.reset(state, world_mask=wp.array([True, False], dtype=wp.bool, device=model.device))
    for _ in range(8):
        wp.capture_launch(capture.graph)
    solver.check_constraint_capacity()
    return solver, contacts, state


def _launched_kernel_keys(model, solver, **pipeline_options):
    """Step once through the collision pipeline and return the key of every kernel the step launched."""
    pipeline = newton.CollisionPipeline(model, **pipeline_options)
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    pipeline.collide(state_in, contacts)
    keys = []

    def recording(launch):
        def record(kernel, *args, **kwargs):
            keys.append(getattr(kernel, "key", None))
            return launch(kernel, *args, **kwargs)

        return record

    with patch.multiple(wp, launch=recording(wp.launch), launch_tiled=recording(wp.launch_tiled)):
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
    return keys


class TestFeatherPGSPathSelection(unittest.TestCase):
    def test_cpu_selection_reasons_are_detached_and_host_only(self):
        """Explain CPU fallbacks without synchronizing or exposing mutable solver state."""
        solver = SolverFeatherPGS(_cpu_model())
        with assert_host_only():
            selected = solver.get_path_selection()
        self.assertIn("CPU", selected["reasons"]["sparse_mass_matrix"])
        self.assertIn("CPU", selected["reasons"]["cached_dynamics"])
        self.assertIn("CPU", selected["reasons"]["kernel_plan_by_dof"][1]["cholesky_kernel"])
        # Benchmark results store this record with json.dumps(..., allow_nan=False).
        json.dumps(selected, allow_nan=False)
        selected["reasons"]["kernel_plan_by_dof"][1]["cholesky_kernel"] = "edited"
        selected["routes"]["friction"]["law"] = "edited"
        selected["options"]["pgs_iterations"]["applied"] = -1
        again = solver.get_path_selection()
        self.assertNotEqual(again["reasons"], selected["reasons"])
        self.assertEqual(again["routes"]["friction"]["law"], "patch")
        self.assertEqual(again["options"]["pgs_iterations"]["applied"], 12)

    def test_cpu_options_record_applied_values_and_sources(self):
        """Record every option's requested, default and applied value once, with non-finite floats as strings."""
        model = _cpu_model()
        options = SolverFeatherPGS(model).get_path_selection()["options"]
        self.assertEqual(set(options), set(inspect.signature(SolverFeatherPGS).parameters) - {"model"})
        self.assertEqual(
            options["friction_anchor_beta"], {"applied": 0.2, "requested": None, "default": None, "source": "default"}
        )
        self.assertEqual(
            options["pgs_kernel"], {"applied": "loop", "requested": "auto", "default": "auto", "source": "default"}
        )
        self.assertEqual(options["pgs_chunk_size"]["applied"], 1)
        self.assertEqual(options["contact_friction_gap_threshold"]["applied"], "inf")
        self.assertEqual(options["joint_limit_activation_gap"]["default"], "inf")
        self.assertFalse(options["fuse_joint_velocity_limits"]["applied"])
        self.assertEqual(options["sparse_mass_matrix"]["applied"], "off")
        self.assertEqual(options["cached_dynamics"]["applied"], "off")
        # Split mode builds no matrix-free tier and CPU builds no stream or cached propagation response.
        self.assertEqual(
            options["mf_gs_incremental_rows"], {"applied": 0, "requested": 32, "default": 32, "source": "default"}
        )
        self.assertEqual(options["mf_gs_response_block_lane_rows"]["applied"], 0)
        for name in ("use_parallel_streams", "propagation_cached_response"):
            self.assertEqual(options[name], {"applied": False, "requested": True, "default": True, "source": "default"})
        solver = SolverFeatherPGS(
            model,
            0.1,
            friction_anchor_beta=0.0,
            pgs_iterations=7,
            parallel_tree=True,
            grouped_dynamics=True,
            lazy_kinematics=True,
            mf_gs_parallel_rows=48,
        )
        selected = solver.get_path_selection()
        options = selected["options"]
        self.assertEqual(
            options["angular_damping"], {"applied": 0.1, "requested": 0.1, "default": 0.05, "source": "argument"}
        )
        self.assertEqual(
            options["friction_anchor_beta"], {"applied": 0.0, "requested": 0.0, "default": None, "source": "argument"}
        )
        # CPU keeps serial traversal and ordinary dynamics: each request resolves to no owner.
        for name in ("parallel_tree", "grouped_dynamics", "lazy_kinematics"):
            self.assertEqual(
                options[name], {"applied": False, "requested": True, "default": False, "source": "argument"}
            )
        self.assertEqual(
            options["mf_gs_parallel_rows"], {"applied": 0, "requested": 48, "default": 0, "source": "argument"}
        )
        self.assertIsNone(selected["routes"]["sweep"]["tiers"])
        self.assertFalse(selected["routes"]["dynamics"]["grouped_topology"])
        self.assertEqual(selected["routes"]["publication"]["owner"], "ordinary")
        self.assertEqual(selected["routes"]["friction"]["law"], "point")
        self.assertEqual(selected["routes"]["friction"]["anchor_beta_source"], "argument")
        solver.pgs_iterations = 3
        self.assertEqual(solver.get_path_selection()["options"]["pgs_iterations"]["applied"], 7)

    def test_cpu_options_record_legacy_alias_environment(self):
        """Attribute an option the IL_NEWTON_FPGS_ warm-start aliases changed to the environment."""
        with patch.dict(os.environ, {"IL_NEWTON_FPGS_MF_WARMSTART": "1", "IL_NEWTON_FPGS_MF_WARMSTART_DECAY": "0.5"}):
            options = SolverFeatherPGS(_cpu_model()).get_path_selection()["options"]
        self.assertEqual(
            options["mf_warmstart"], {"applied": True, "requested": False, "default": False, "source": "environment"}
        )
        self.assertEqual(
            options["mf_warmstart_decay"], {"applied": 0.5, "requested": 1.0, "default": 1.0, "source": "environment"}
        )
        self.assertEqual(
            options["pgs_warmstart"], {"applied": True, "requested": False, "default": False, "source": "environment"}
        )
        self.assertEqual(
            options["pgs_warmstart_decay"],
            {"applied": 0.5, "requested": 1.0, "default": 1.0, "source": "environment"},
        )
        with patch.dict(os.environ, {"IL_NEWTON_FPGS_MF_WARMSTART": "0", "IL_NEWTON_FPGS_MF_WARMSTART_DECAY": "0.5"}):
            options = SolverFeatherPGS(_cpu_model()).get_path_selection()["options"]
        self.assertEqual(options["mf_warmstart_decay"]["source"], "environment")
        # Without the warm-start alias the decay alias leaves the unified decay alone.
        self.assertEqual(
            options["pgs_warmstart_decay"], {"applied": 1.0, "requested": 1.0, "default": 1.0, "source": "default"}
        )

    def test_cpu_route_record_reports_known_selections(self):
        """Report the CPU loop sweep, the split free-body rows and the declined owners with their reasons."""
        selected = SolverFeatherPGS(_cpu_free_body_model()).get_path_selection()
        routes = selected["routes"]
        self.assertEqual(routes["articulations_by_dof"], {"6": 1, "1": 1})
        self.assertEqual(routes["sweep"]["route"], "dense_loop")
        # The arm and the free box share a world: split mode interleaves the dense and free-body sweeps.
        self.assertTrue(routes["free_rigid_bodies"]["mixed_worlds"])
        self.assertEqual(routes["sweep"]["driver"], "split_interleaved")
        self.assertEqual(routes["sweep"]["dense"], {"kernels": ["pgs_solve_loop"]})
        self.assertEqual(routes["sweep"]["free_body_matrix_free"], "pgs_solve_mf_loop")
        # Only the matrix-free step packs row metadata.
        self.assertIsNone(routes["sweep"]["mf_meta_pack"])
        self.assertEqual(routes["free_rigid_bodies"]["count"], 1)
        self.assertEqual(routes["free_rigid_bodies"]["owners"], ["split_matrix_free_rows"])
        self.assertIsNone(routes["held_tree"]["owner"])
        self.assertIn("CPU", routes["held_tree"]["reason"])
        self.assertEqual(routes["contact_rows"]["producer"], "general")
        self.assertEqual(routes["contact_rows"]["general_method_by_dof"], {"6": "per_contact", "1": "per_contact"})
        self.assertFalse(routes["contact_rows"]["row_packets"]["enabled"])
        self.assertIn("point contacts", routes["contact_rows"]["row_packets"]["reason"])
        self.assertFalse(routes["resolved_worlds"]["enabled"])
        self.assertIn("matrix-free route", routes["resolved_worlds"]["reason"])
        self.assertEqual(routes["publication"]["owner"], "ordinary")
        self.assertEqual(routes["publication"]["fk_id_cache"], "off")
        self.assertEqual(routes["velocity_limits"]["joint"], "off")
        self.assertFalse(routes["velocity_limits"]["free_body_rows"])
        self.assertEqual(routes["friction"]["law"], "patch")
        self.assertEqual(routes["friction"]["anchor_beta_source"], "default")
        self.assertIn("need point friction", routes["friction"]["reason"])
        # The compact contact boundary writes the patch row law as well.
        self.assertNotIn("compact contacts", routes["friction"]["reason"])
        self.assertIsNone(routes["sparse_mass_matrix"]["size"])
        self.assertFalse(routes["sparse_mass_matrix"]["active_rows"])
        self.assertIn("CPU", routes["sparse_mass_matrix"]["reason"])
        self.assertIsNone(routes["dynamics"]["grouped"])
        self.assertEqual({plan["hinv_jt_kernel"] for plan in selected["kernel_plan_by_dof"].values()}, {"par_row"})
        self.assertEqual(selected["measurements"], {"held_tree_lane_timing_us": []})
        separate = SolverFeatherPGS(_cpu_separate_worlds_model()).get_path_selection()["routes"]
        self.assertFalse(separate["free_rigid_bodies"]["mixed_worlds"])
        self.assertEqual(separate["sweep"]["driver"], "dense_split")
        self.assertIn("free-body rows", separate["sweep"]["driver_reason"])
        debug = SolverFeatherPGS(_cpu_model(), pgs_debug=True).get_path_selection()["routes"]
        self.assertEqual(debug["sweep"]["driver"], "debug_loop")

    def test_cpu_grouped_record_follows_the_response_dispatch(self):
        """Label the sizes a grouped topology would serve, with their reasons, from host state alone."""
        solver = SolverFeatherPGS(_cpu_free_body_model())
        # A stand-in for the CUDA-only grouped topology: the record reads its layout fields only.
        solver._grouped_topology = SimpleNamespace(
            lanes_per_articulation=8, articulations_per_block=4, template_count=2
        )
        selected = solver.get_path_selection()
        json.dumps(selected, allow_nan=False)
        grouped = selected["routes"]["dynamics"]["grouped"]
        self.assertEqual(
            (grouped["lanes_per_articulation"], grouped["articulations_per_block"], grouped["template_count"]),
            (8, 4, 2),
        )
        # CPU builds no FK/ID cache and no asynchronous drives, so neither template kernel is reached.
        self.assertFalse(grouped["template_fk"])
        self.assertIsNone(grouped["tau"])
        by_dof = grouped["hinv_jt_by_dof"]
        self.assertTrue(by_dof["1"]["grouped"])
        self.assertEqual(selected["kernel_plan_by_dof"][1]["hinv_jt_kernel"], "grouped")
        self.assertIn("routes.dynamics.grouped", selected["reasons"]["kernel_plan_by_dof"][1]["hinv_jt_kernel"])
        # The free box's six-coordinate size keeps its planned kernel unless opted in.
        self.assertFalse(by_dof["6"]["grouped"])
        self.assertIn("FEATHER_PGS_GROUPED_HINV_FREE", by_dof["6"]["reason"])
        self.assertEqual(selected["kernel_plan_by_dof"][6]["hinv_jt_kernel"], "par_row")

    def test_cpu_route_record_matches_launched_kernels(self):
        """Tie the record's sweep driver, row builders and root clamp to the kernels one CPU step launches."""
        for name, model in (
            ("mixed", _cpu_free_body_model()),
            ("mixed_velocity_limits", _cpu_free_body_model(velocity_limit=1.0)),
            ("separate_worlds", _cpu_separate_worlds_model()),
            ("arm", _cpu_model()),
        ):
            with self.subTest(model=name):
                solver = SolverFeatherPGS(model)
                routes = solver.get_path_selection()["routes"]
                sweep = routes["sweep"]
                keys = _launched_kernel_keys(model, solver)
                dense, free_body = sweep["dense"]["kernels"][0], sweep["free_body_matrix_free"]
                sweeps = [key for key in keys if key in (dense, free_body)]
                if sweep["driver"] == "split_interleaved":
                    # One one-iteration dense sweep, then one free-body sweep, per iteration.
                    self.assertEqual(sweeps, [dense, free_body] * solver.pgs_iterations)
                else:
                    self.assertEqual(sweep["driver"], "dense_split")
                    self.assertEqual(sweeps, [dense] if free_body is None else [dense, free_body])
                methods = routes["contact_rows"]["general_method_by_dof"]
                self.assertEqual(keys.count("populate_world_J_for_size"), list(methods.values()).count("per_contact"))
                self.assertEqual(routes["velocity_limits"]["root_clamp"], "clamp_free_root_velocity_limits" in keys)

    def test_cpu_split_route_reports_free_body_velocity_limit_rows(self):
        """Report free-body velocity-limit rows in split mode, where the row builder allocates them as well."""
        solver = SolverFeatherPGS(_cpu_free_body_model(velocity_limit=1.0))
        limits = solver.get_path_selection()["routes"]["velocity_limits"]
        self.assertTrue(limits["rigid_body_limits"])
        self.assertIsNotNone(solver.rigid_velocity_limit_slot)
        self.assertTrue(limits["free_body_rows"])
        # Split mode keeps the direct root clamp of the free box beside the rows.
        self.assertTrue(limits["root_clamp"])

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse admission requires CUDA")
    def test_route_record_reports_sparse_rows_and_tree_owner(self):
        """Report the active-row scan, triplets, sweep kernel and held-tree layout of the sparse route."""
        model = _build_model()
        with patch.dict(os.environ, {"FEATHER_PGS_SPARSE_CONTACT_TRIPLETS": "1"}):
            point = _make_solver(model, True, friction_anchor_beta=0.0)
            patch_friction = _make_solver(model, True)
        with assert_host_only():
            selected = point.get_path_selection()
            patch_selected = patch_friction.get_path_selection()
        json.dumps(selected, allow_nan=False)
        routes = selected["routes"]
        sparse = routes["sparse_mass_matrix"]
        self.assertEqual(sparse["size"], point._sparse_mass_matrix_size)
        self.assertTrue(sparse["active_rows"])
        self.assertEqual(sparse["contact_triplets"], point._sparse_contact_triplets)
        self.assertEqual(routes["sweep"]["route"], "sparse_pgs")
        self.assertEqual(routes["sweep"]["sparse"], point._pgs_solve_sparse_kernel.key)
        self.assertIn("_active", routes["sweep"]["sparse"])
        if sparse["contact_triplets"]:
            self.assertIn("_contact_triplets", routes["sweep"]["sparse"])
        patch_sparse = patch_selected["routes"]["sparse_mass_matrix"]
        self.assertFalse(patch_sparse["active_rows"])
        self.assertFalse(patch_sparse["contact_triplets"])
        if patch_sparse["size"] is not None:
            self.assertIn("friction_anchor_beta", patch_sparse["active_rows_reason"])
            self.assertIn("active-row scan", patch_sparse["contact_triplets_reason"])
        tree = routes["held_tree"]
        self.assertEqual(tree["owner"] is None, point._tree_kinetic_state is None)
        for group in tree["groups"]:
            self.assertIn(group["lanes"], (4, 8, 16, 32, 64))
            self.assertEqual(len(group["roots"]), len(group["root_families"]))
            self.assertIsNotNone(group["kernels"]["finish"])

    @unittest.skipUnless(wp.is_cuda_available(), "Cached dynamics requires CUDA")
    def test_route_record_reports_mixed_owner_and_paired_response(self):
        """Report the mixed held-tree groups and the paired response of an arm with one free box per world."""
        from newton.tests.test_feather_pgs_mixed_tree_free_body import _build  # noqa: PLC0415

        model = _build(wp.get_device("cuda:0"), links=17)
        environment = {
            "FEATHER_PGS_MIXED_TREE_KINETIC_STATE": "1",
            "FEATHER_PGS_TREE_KINETIC_STATE": "0",
            "FEATHER_PGS_TREE_FOREST": "0",
        }
        with patch.dict(os.environ, environment):
            solver = SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                pgs_iterations=8,
                enable_joint_limits=True,
                update_mass_matrix_interval=2,
                friction_anchor_beta=0.0,
                dense_max_constraints=192,
            )
        with assert_host_only():
            selected = solver.get_path_selection()
        json.dumps(selected, allow_nan=False)
        routes = selected["routes"]
        paired = routes["paired_response"]
        self.assertEqual(paired["primary_dofs"], 17)
        self.assertEqual(paired["factor_coordinates"], solver._paired_factor_coordinates)
        self.assertEqual(paired["components"], solver._paired_components is not None)
        self.assertEqual(paired["two_stream_overlap"], solver._paired_general_solve_stream is not None)
        self.assertEqual(routes["sweep"]["paired_factor"], getattr(solver._paired_factor_solve_kernel, "key", None))
        self.assertEqual(routes["held_tree"]["owner"], "mixed")
        self.assertIn("FEATHER_PGS_TREE_FOREST", routes["held_tree"]["layout_reason"])
        self.assertEqual(sorted(group["dofs"] for group in routes["held_tree"]["groups"]), [6, 17])
        self.assertIn(routes["free_rigid_bodies"]["owners"][0], ("paired_factor", "paired_response"))
        self.assertEqual(routes["resolved_worlds"]["enabled"], solver._resolved_worlds_owner is not None)

    @unittest.skipUnless(wp.is_cuda_available(), "Matrix-free routes require CUDA")
    def test_route_record_is_json_across_routes(self):
        """Build a JSON-compliant record for every major admitted construction route."""
        cases = (
            ({}, {}),
            ({"friction_anchor_beta": 0.0}, {}),
            ({"pgs_mode": "split"}, {}),
            ({"sparse_mass_matrix": "off", "friction_anchor_beta": 0.0}, {"free_count": 2}),
            (
                {"sparse_mass_matrix": "off", "mf_gs_parallel_rows": 48, "friction_anchor_beta": 0.0},
                {"free_count": 2},
            ),
            ({"sparse_mass_matrix": "off", "mf_gs_response_block_rows": 64, "friction_anchor_beta": 0.0}, {}),
            ({"articulated_contact_response": "propagation"}, {"free_count": 2}),
            ({"articulated_contact_response": "propagation-colored"}, {"free_count": 2}),
            ({"grouped_dynamics": True}, {}),
            ({"grouped_dynamics": True, "sparse_mass_matrix": "off", "friction_anchor_beta": 0.0}, {}),
            ({"enable_joint_velocity_limits": True}, {}),
            # Patch friction builds no parallel tier, whatever the requested budget.
            ({"sparse_mass_matrix": "off", "mf_gs_parallel_rows": 48}, {"free_count": 2}),
            ({"sparse_mass_matrix": "off", "pgs_schedule": "physx_grasp", "friction_anchor_beta": 0.0}, {}),
            # Lazy publication defers only on the fused world-dynamics kernel, and never under patch friction.
            (
                {
                    "grouped_dynamics": True,
                    "lazy_kinematics": True,
                    "sparse_mass_matrix": "off",
                    "cached_dynamics": "off",
                    "friction_anchor_beta": 0.0,
                },
                {},
            ),
            (
                {
                    "grouped_dynamics": True,
                    "lazy_kinematics": True,
                    "sparse_mass_matrix": "off",
                    "cached_dynamics": "off",
                },
                {},
            ),
        )
        built = 0
        for options, model_options in cases:
            with self.subTest(options=options, model=model_options):
                try:
                    solver = _make_solver(_build_model(**model_options), True, **options)
                except (ValueError, NotImplementedError):
                    continue  # The record describes admitted constructors only.
                built += 1
                with assert_host_only():
                    selected = solver.get_path_selection()
                json.dumps(selected, allow_nan=False)
                applied = {name: option["applied"] for name, option in selected["options"].items()}
                self.assertEqual(set(applied), set(inspect.signature(SolverFeatherPGS).parameters) - {"model"})
                routes = selected["routes"]
                # Owners are reported only where their kernels were built.
                owners = routes["free_rigid_bodies"]["owners"]
                self.assertEqual("local_pair" in owners, bool(solver._pgs_solve_local_pair_kernels))
                self.assertEqual("local_residual" in owners, bool(solver._pgs_solve_local_residual_kernels))
                pack = routes["sweep"]["mf_meta_pack"]
                self.assertEqual(pack is None, solver.pgs_mode != "matrix_free")
                # Applied routing options agree with the routes they request.
                tiers = routes["sweep"]["tiers"]
                parallel = tiers is not None and tiers["family"] in ("parallel", "keyed_parallel")
                self.assertEqual(applied["mf_gs_parallel_rows"], tiers["tiers"][-1]["rows"] if parallel else 0)
                self.assertEqual(applied["grouped_dynamics"], routes["dynamics"]["grouped_topology"])
                self.assertEqual(
                    applied["lazy_kinematics"],
                    selected["options"]["lazy_kinematics"]["requested"]
                    and routes["dynamics"]["fused_world_dynamics"]
                    and routes["friction"]["law"] == "point",
                )
                self.assertEqual(applied["lazy_kinematics"], routes["publication"]["owner"] == "deferred")
                if options.get("pgs_schedule") == "physx_grasp" and routes["sweep"]["route"] == "matrix_free_gs":
                    self.assertEqual(routes["sweep"]["driver"], "physx_grasp")
                # A size whose response the grouped kernel solves is labeled so in the kernel plan.
                grouped = routes["dynamics"]["grouped"]
                self.assertEqual(grouped is None, solver._grouped_topology is None)
                for size, plan in selected["kernel_plan_by_dof"].items():
                    is_grouped = grouped is not None and grouped["hinv_jt_by_dof"][str(size)]["grouped"]
                    self.assertEqual(plan["hinv_jt_kernel"] == "grouped", is_grouped, size)
                if grouped is not None:
                    self.assertEqual(grouped["lanes_per_articulation"], solver._grouped_topology.lanes_per_articulation)
        self.assertGreaterEqual(built, 8)

    def test_private_override_validation_and_cpu_precedence(self):
        """Validate effective overrides while preserving explicit choices and CPU fallbacks."""
        model = _cpu_model()
        with patch.dict(SolverFeatherPGS._kernel_overrides, {"cholesky_kernel": "invalid"}):
            with self.assertRaisesRegex(ValueError, "cholesky_kernel"):
                SolverFeatherPGS(model)
            explicit = SolverFeatherPGS(model, cholesky_kernel="loop").get_path_selection()
            self.assertIn("Explicit", explicit["reasons"]["kernel_plan_by_dof"][1]["cholesky_kernel"])
        with patch.dict(SolverFeatherPGS._kernel_overrides, {"cholesky_kernel": "tiled"}):
            solver = SolverFeatherPGS(model)
            self.assertEqual(solver.cholesky_kernel, "loop")
            reason = solver.get_path_selection()["reasons"]["kernel_plan_by_dof"][1]["cholesky_kernel"]
            self.assertIn("CPU", reason)
            self.assertIn("override", reason)

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse admission requires CUDA")
    def test_sparse_selection_reasons_distinguish_policy_capacity_and_override(self):
        """Explain the actual sparse rejection while retaining the dense fallback."""
        model = _build_model()
        cases = (
            ({"sparse_mass_matrix": "off"}, "Explicit"),
            ({"trisolve_kernel": "loop"}, "explicit dense"),
            ({"dense_max_constraints": 2048}, "shared memory"),
        )
        for options, expected in cases:
            with self.subTest(options=options):
                selected = _make_solver(model, True, **options).get_path_selection()
                self.assertEqual(selected["sparse_mass_matrix"], "off")
                self.assertIn(expected, selected["reasons"]["sparse_mass_matrix"])
        selected = _make_solver(model, True).get_path_selection()
        self.assertEqual(selected["sparse_mass_matrix"], "on")
        self.assertIn("topology", selected["reasons"]["sparse_mass_matrix"])

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse admission requires CUDA")
    def test_sparse_warm_start_installs_no_dense_response(self):
        """Seed warm starts in the native sparse sweep, which keeps no dense response for the Y*lambda install."""
        model = _build_model()
        for beta in (0.0, 0.2):
            for options in ({}, {"enable_joint_velocity_limits": True}):
                with self.subTest(friction_anchor_beta=beta, **options):
                    solver = _make_solver(model, True, pgs_warmstart=True, friction_anchor_beta=beta, **options)
                    routes = solver.get_path_selection()["routes"]
                    self.assertEqual(routes["sweep"]["route"], "sparse_pgs")
                    self.assertTrue(routes["sweep"]["sparse"].endswith("_warm_start"))
                    self.assertEqual(routes["sparse_mass_matrix"]["velocity_limits"], bool(options))
                    # A placeholder stands in for the dense response that the install would read.
                    self.assertEqual(solver.Y_by_size[9].shape, (1, 1, 1))
                    # Contact warm start carries impulses by matched contact identity.
                    keys = _launched_kernel_keys(model, solver, contact_matching="latest")
                    self.assertIn(routes["sweep"]["sparse"], keys)
                    self.assertNotIn("apply_impulses_world_par_dof", keys)

    @unittest.skipUnless(wp.is_cuda_available(), "Cached dynamics requires CUDA")
    def test_cache_selection_reports_environment_and_explicit_precedence(self):
        """Retain construction-time cache decisions after environment overrides change."""
        model = _build_model()
        with patch.dict(os.environ, {"FEATHER_PGS_TREE_KINETIC_STATE": "0"}):
            disabled = _make_solver(model, True)
            forced = _make_solver(model, True, cached_dynamics="on")
        self.assertEqual(disabled.get_path_selection()["cached_dynamics"], "off")
        self.assertIn("FEATHER_PGS_TREE_KINETIC_STATE", disabled.get_path_selection()["reasons"]["cached_dynamics"])
        self.assertEqual(forced.get_path_selection()["cached_dynamics"], "on")
        self.assertIn("Explicit", forced.get_path_selection()["reasons"]["cached_dynamics"])

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse admission requires CUDA")
    def test_sparse_auto_composes_existing_execution_choices(self):
        """Select qualified split, velocity-limit and grouped paths without private switches."""
        model = _build_model()
        flags = (
            "FEATHER_PGS_SPARSE_SPLIT",
            "FEATHER_PGS_SPARSE_VELOCITY_LIMITS",
            "FEATHER_PGS_SPARSE_GROUPED_DYNAMICS",
        )
        cases = (
            {"pgs_mode": "split"},
            {"enable_joint_velocity_limits": True},
            {"grouped_dynamics": True},
        )
        with patch.dict(os.environ):
            for name in flags:
                os.environ.pop(name, None)
            for options in cases:
                with self.subTest(options=options):
                    solver = _make_solver(model, True, friction_anchor_beta=0.0, **options)
                    self.assertEqual(solver.get_path_selection()["sparse_mass_matrix"], "on")
                    disabled = _make_solver(model, True, sparse_mass_matrix="off", friction_anchor_beta=0.0, **options)
                    self.assertEqual(disabled.get_path_selection()["sparse_mass_matrix"], "off")
            for options in ({"pgs_mode": "split"}, {"enable_joint_velocity_limits": True}):
                with self.subTest(patch=options):
                    # Default patch friction composes with the same sparse factors and tree cache.
                    selected = _make_solver(model, True, **options).get_path_selection()
                    self.assertEqual(selected["sparse_mass_matrix"], "on")
                    self.assertEqual(selected["cached_prediction"], "tree")
            for name in flags:
                os.environ[name] = "0"
            for options in cases:
                with self.subTest(legacy_disabled=options):
                    ordinary = _make_solver(model, True, friction_anchor_beta=0.0, **options)
                    self.assertEqual(ordinary.get_path_selection()["sparse_mass_matrix"], "off")
                    forced = _make_solver(model, True, sparse_mass_matrix="on", friction_anchor_beta=0.0, **options)
                    self.assertEqual(forced.get_path_selection()["sparse_mass_matrix"], "on")

    def test_cached_dynamics_cpu_choices(self):
        """Keep CPU auto ordinary and reject a forced unsupported cache."""
        model = _cpu_model()
        for choice in ("auto", "off"):
            selected = SolverFeatherPGS(model, cached_dynamics=choice).get_path_selection()
            self.assertEqual(selected["requested"]["cached_dynamics"], choice)
            self.assertEqual(selected["cached_dynamics"], "off")
            self.assertEqual(selected["publication"], "ordinary")
        for choice in ("on", "typo"):
            with self.subTest(choice=choice), self.assertRaisesRegex(ValueError, "cached_dynamics"):
                SolverFeatherPGS(model, cached_dynamics=choice)

    def test_cpu_patch_publication_is_never_deferred(self):
        """Report every-substep publication for patch friction, even with lazy kinematics on the fused kernel."""
        for beta, publication in ((0.0, "deferred"), (0.2, "ordinary")):
            with self.subTest(friction_anchor_beta=beta):
                solver = SolverFeatherPGS(_cpu_model(), friction_anchor_beta=beta, lazy_kinematics=True)
                # A stand-in for the CUDA-only fused world-dynamics kernel: the record reads host flags only.
                solver._fused_k1 = True
                selected = solver.get_path_selection()
                self.assertEqual(selected["publication"], publication)
                self.assertEqual(selected["routes"]["publication"]["owner"], publication)
                # The options record was captured at construction, before the stand-in: recompute it.
                options = applied_options(
                    solver, SolverFeatherPGS.__init__, {"friction_anchor_beta": beta, "lazy_kinematics": True}
                )
                self.assertEqual(options["lazy_kinematics"]["applied"], publication == "deferred")
                if beta:
                    self.assertIn("release test", selected["reasons"]["publication"])
                    self.assertIn("release test", selected["routes"]["publication"]["reason"])

    @unittest.skipUnless(wp.is_cuda_available(), "Cached dynamics requires CUDA")
    def test_cached_dynamics_overrides_environment_and_dense_conflicts(self):
        """Honor explicit cache choices and retain automatic dense kernel requests."""
        model = _build_model()
        for choice, legacy, expected in (("on", "0", True), ("off", "1", False), ("auto", "0", False)):
            with self.subTest(choice=choice), patch.dict(os.environ, {"FEATHER_PGS_TREE_KINETIC_STATE": legacy}):
                solver = _make_solver(model, True, cached_dynamics=choice)
                selected = solver.get_path_selection()
                self.assertEqual(solver._tree_kinetic_state is not None, expected)
                self.assertEqual(selected["cached_dynamics"], "on" if expected else "off")
                self.assertEqual(selected["publication"], "cached" if expected else "ordinary")
        with self.assertRaisesRegex(ValueError, "cached_dynamics.*conflicts"):
            _make_solver(model, False, cached_dynamics="on", trisolve_kernel="loop")
        solver = _make_solver(model, False, cached_dynamics="auto", trisolve_kernel="loop")
        self.assertEqual(solver.get_path_selection()["cached_dynamics"], "off")

    @unittest.skipUnless(wp.is_cuda_available(), "Cached dynamics requires CUDA")
    def test_cached_dynamics_generic_tree_is_default_and_unsupported_roots_reject(self):
        """A fixed arm with a free body is cached by the generic tree owner by default; a revolute world root is rejected."""
        world = newton.ModelBuilder()
        joints = []
        parent = -1
        for index in range(5):
            body = world.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            if index in (0, 4):
                joint = world.add_joint_fixed(parent, body)
            else:
                joint = world.add_joint_revolute(parent, body, armature=0.01)
            joints.append(joint)
            parent = body
        world.add_articulation(joints)
        body = world.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        world.add_articulation([world.add_joint_free(body)])
        builder = newton.ModelBuilder()
        builder.replicate(world, 2)
        model = builder.finalize(device="cuda:0")
        solver = _make_solver(model, False, cached_dynamics="auto", friction_anchor_beta=0.0)
        self.assertEqual(solver.get_path_selection()["cached_dynamics"], "on")
        self.assertEqual(solver.get_path_selection()["cached_prediction"], "tree")
        # FEATHER_PGS_MIXED_TREE_KINETIC_STATE=0 retains ordinary dynamics.
        with patch.dict(os.environ, {"FEATHER_PGS_MIXED_TREE_KINETIC_STATE": "0"}):
            solver = _make_solver(model, False, cached_dynamics="auto", friction_anchor_beta=0.0)
        self.assertEqual(solver.get_path_selection()["cached_dynamics"], "off")
        # Cached trees require a fixed or free root; a revolute joint at the world is rejected.
        world = newton.ModelBuilder()
        joints, parent = [], -1
        for _ in range(3):
            body = world.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joints.append(world.add_joint_revolute(parent, body, armature=0.01))
            parent = body
        world.add_articulation(joints)
        body = world.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        world.add_articulation([world.add_joint_free(body)])
        builder = newton.ModelBuilder()
        builder.replicate(world, 2)
        model = builder.finalize(device="cuda:0")
        with self.assertRaisesRegex(ValueError, "cached_dynamics"):
            _make_solver(model, False, cached_dynamics="on", friction_anchor_beta=0.0)

    @unittest.skipUnless(wp.is_cuda_available(), "Cached dynamics requires CUDA")
    def test_cached_dynamics_contact_graph_and_reset(self):
        """Preserve contacting states through graph replay and a partial-world reset."""
        model = _build_model()
        outputs = []
        for choice in ("off", "on"):
            _solver, contacts, state = _graph_and_reset_rollout(model, True, choice)
            self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
            outputs.append({name: getattr(state, name).numpy() for name in STATE_FIELDS})
        for name in outputs[0]:
            np.testing.assert_allclose(outputs[0][name], outputs[1][name], atol=3e-4, rtol=3e-4, err_msg=name)

    @unittest.skipUnless(wp.is_cuda_available(), "Cached dynamics requires CUDA")
    def test_cached_dynamics_dense_patch_graph_and_reset(self):
        """Cache the dense route under default patch friction with unchanged states and anchor history.

        The standing fixture's feet keep loaded, ring-linked anchors through the replays and the
        partial reset, so the history comparison covers carried anchors.
        """
        model = standing_model()
        # Without the owner this is the ordinary patch route: the incremental tier, which also serves
        # patch friction, stays off in both arms so that they differ by the owner alone.
        runs = [_graph_and_reset_rollout(model, False, choice, mf_gs_incremental_rows=0) for choice in ("off", "on")]
        self.assertIsNone(runs[0][0].get_path_selection()["cached_prediction"])
        self.assertEqual(runs[1][0].get_path_selection()["cached_prediction"], "tree")
        for solver, contacts, _state in runs:
            self.assertTrue(solver._friction_anchors_enabled)
            self.assertIsNone(solver._sparse_mass_matrix_size)
            self.assertIsNone(solver.get_path_selection()["routes"]["sweep"]["tiers"])
            self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        for name in STATE_FIELDS:
            np.testing.assert_allclose(
                getattr(runs[0][2], name).numpy(), getattr(runs[1][2], name).numpy(), atol=3e-4, rtol=3e-4, err_msg=name
            )
        for solver, contacts, _state in runs:
            assert_loaded_anchors(solver, contacts)
        assert_history_close(runs[0][0], runs[1][0])

    def test_scalar_overrides_reach_solver(self):
        """Honor public choices independently of private test overrides."""
        choices = {
            "cholesky_kernel": "loop",
            "trisolve_kernel": "loop",
            "hinv_jt_kernel": "par_row",
            "delassus_kernel": "par_row_col",
            "pgs_kernel": "loop",
        }
        with patch.dict(SolverFeatherPGS._kernel_overrides, {"cholesky_kernel": "tiled"}):
            solver = SolverFeatherPGS(_cpu_model(), **choices)
        for name, value in choices.items():
            self.assertEqual(getattr(solver, name), value)
        self.assertEqual(solver.get_path_selection()["requested"]["cholesky_kernel"], "loop")

    def test_auto_cpu_and_invalid_forced_choices(self):
        """Resolve CPU auto to scalar kernels and reject unsupported explicit choices."""
        model = _cpu_model()
        self.assertEqual(SolverFeatherPGS(model).get_path_selection()["pgs_kernel"], "loop")
        for name, value in (
            ("cholesky_kernel", "tiled"),
            ("trisolve_kernel", "tiled"),
            ("hinv_jt_kernel", "tiled"),
            ("delassus_kernel", "tiled"),
            ("pgs_kernel", "tiled_row"),
            ("sparse_mass_matrix", "on"),
            ("cholesky_kernel", "typo"),
        ):
            with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                SolverFeatherPGS(model, **{name: value})

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse factors require CUDA")
    def test_public_dense_kernels_override_sparse_and_fused_dispatch(self):
        """Select actual dense kernels while leaving unspecified choices automatic."""
        model = _build_model()
        solver = SolverFeatherPGS(model, pgs_mode="matrix_free", trisolve_kernel="loop")
        selected = solver.get_path_selection()
        self.assertEqual(selected["sparse_mass_matrix"], "off")
        self.assertEqual(selected["kernel_plan_by_dof"][9]["trisolve_kernel"], "loop")
        self.assertEqual(selected["requested"]["hinv_jt_kernel"], "auto")
        split = SolverFeatherPGS(model, pgs_mode="split", delassus_kernel="par_row_col")
        self.assertFalse(split._execution_plan.hinv_jt_fused_sizes)
        for kwargs, message in (
            ({"sparse_mass_matrix": "on", "trisolve_kernel": "loop"}, "conflicts"),
            ({"pgs_kernel": "loop"}, "pgs_kernel"),
            ({"delassus_kernel": "tiled"}, "delassus_kernel"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                SolverFeatherPGS(model, pgs_mode="matrix_free", **kwargs)

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse factors require CUDA")
    def test_force_sparse_and_dense_with_graph_replay(self):
        """Force both representations and preserve a contacting articulated rollout."""
        model = _build_model()
        outputs = []
        for choice in ("on", "off"):
            solver = SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                sparse_mass_matrix=choice,
                enable_joint_limits=True,
                pgs_iterations=32,
                double_buffer=False,
            )
            self.assertEqual(solver.get_path_selection()["sparse_mass_matrix"], choice)
            state, output = model.state(), model.state()
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            pipeline = newton.CollisionPipeline(model, convex_collision="ordinary")
            contacts, control = pipeline.contacts(), model.control()

            def step(pipeline=pipeline, state=state, contacts=contacts, solver=solver, output=output, control=control):
                pipeline.collide(state, contacts)
                solver.step(state, output, control, contacts, 1.0 / 400.0)
                state.assign(output)

            step()
            with wp.ScopedCapture(device=model.device) as capture:
                step()
            for _ in range(8):
                wp.capture_launch(capture.graph)
            solver.check_constraint_capacity()
            outputs.append(state.joint_qd.numpy())
        np.testing.assert_allclose(outputs[0], outputs[1], atol=3e-4, rtol=3e-4)


if __name__ == "__main__":
    unittest.main()
