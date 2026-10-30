# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise public automatic choices and forced implementation contracts."""

import copy
import dataclasses
import inspect
import json
import os
import unittest
import warnings
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import environment_flags, friction_patches, path_selection, sweep_routes_tiers
from newton._src.solvers.feather_pgs import solver_feather_pgs as solver_module
from newton._src.solvers.feather_pgs.contact_row_plan import BUILDER_METHODS, contact_row_plan
from newton._src.solvers.feather_pgs.keyed_row_layout import KeyedLayout
from newton._src.solvers.feather_pgs.step_plan import StepPlan
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_patch_parity import (
    STANDING_REGION_BOUNDS,
    anchor_sites,
    assert_anchor_sites_close,
    assert_history_matches_rows,
    assert_loaded_anchors,
    assert_patch_impulses_close,
    assert_patch_rows_close,
    assert_region_impulses_close,
    assert_ring_coulomb,
    assert_states_close,
    patch_rows,
    standing_model,
)
from newton.tests.feather_pgs_row_builders import (
    assert_row_builders_named,
    install_cpu_keyed_rows,
    is_row_builder,
    launched_kernels,
)
from newton.tests.feather_pgs_scenes import plain_laws
from newton.tests.test_feather_pgs_sparse_solver import _build_model, _make_solver, _over_budget_rows
from newton.tests.unittest_utils import assert_host_only

STATE_FIELDS = ("joint_q", "joint_qd", "body_q", "body_qd")
PATCH_DT = 1.0 / 400.0
# The first step of the cached patch route against the ordinary route, from one state. The owner's forward
# kinematics writes the input poses a few float32 ulps from the ordinary FK, and the resting feet's 1/dt
# position gain turns that into joint velocity and contact impulse. Per ulp of every input pose component
# (CPU split stand-in, 32 iterations, 32 sign draws, linear up to 4 ulps): joint coordinates 2.8e-7, joint
# velocity 1.2e-4, per-contact normal impulse 1.2e-4, region normal load 1.4e-4, region friction 1.5e-5. The
# bounds admit six to eight ulps, as the step's 2e-6 pose bound admits seven, and stay below the step's own
# joint velocity (1.5e-3), so a route that publishes stale velocities fails; re-measure on CUDA. Both routes
# reserve contact rows in one order (sorted contacts, one warp): a random per-world row order alone moves the
# step's joint velocity by 3.3e-3 on CPU.
FIRST_STEP_VELOCITY_ATOL = 7.5e-4
FIRST_STEP_IMPULSE_ATOL = 1.0e-3
FIRST_STEP_REGION_BOUNDS = {"load_atol": 1.0e-3, "friction_atol": 1.0e-4}


def _pinned(model, pins, **options):
    """A solver built under the test-only route pins ``pins``."""
    with patch.dict(SolverFeatherPGS._route_pins, pins):
        return SolverFeatherPGS(model, **options)


def _cpu_model():
    builder = newton.ModelBuilder()
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.add_articulation([builder.add_joint_revolute(-1, body)])
    return builder.finalize(device="cpu")


@contextmanager
def _without_solver_environment():
    """Remove every FEATHER_PGS_ and IL_NEWTON_FPGS_ variable, so construction warns only about its arguments."""
    with patch.dict(os.environ):
        for name in [name for name in os.environ if name.startswith(environment_flags.PREFIXES)]:
            del os.environ[name]
        yield


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


def _resting_arms_and_box(*links, device="cpu"):
    """One world of revolute arms with the given link counts and a free box, all resting on the ground.

    Each arm hangs straight down from its first joint to a sphere foot on the ground, so the world
    holds one size group per link count beside the box's six coordinates.
    """
    builder = newton.ModelBuilder()
    for index, count in enumerate(links):
        parent, joints = -1, []
        for link in range(count):
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.05))
            offset = (0.4 * index, 0.0, 0.05 + 0.1 * (count - 1)) if link == 0 else (0.0, 0.0, -0.1)
            joints.append(
                builder.add_joint_revolute(
                    parent,
                    body,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform(wp.vec3(*offset), wp.quat_identity()),
                    target_ke=50.0,
                    target_kd=5.0,
                )
            )
            parent = body
        builder.add_shape_sphere(parent, radius=0.05)
        builder.add_articulation(joints)
    box = builder.add_body(xform=wp.transform((-0.4, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _plan_stand_in(law, sizes=(108, 6), case="K6", **attributes):
    """A host stand-in with the attributes contact_row_plan and the route record read; no owner is admitted unless
    ``case`` and the attributes name one."""
    values = {
        "laws": SimpleNamespace(friction_anchors_enabled=law == "patch", set=plain_laws(contact=law)),
        "forest": SimpleNamespace(size_groups=list(sizes)),
        "routes": SimpleNamespace(case=case),
        "_row_packets": None,
        "_keyed_rows": None,
        "_compact_contact_boundary": False,
        "_sparse_mass_matrix_size": None,
        "_sparse_prepare_contacts": False,
        "_sparse_diagonal_response_size": None,
        "_sparse_diagonal_contact_triples": False,
        "_sparse_diagonal_contact_units": False,
        "_compact_contact_jacobian": False,
    }
    values.update(attributes)
    return SimpleNamespace(**values)


def _branched_arm_and_box(device="cuda:0", worlds=2, depth=9):
    """A fixed-base arm of two ``depth``-link branches, which sparse factors admit, and a free box per world.

    Nine links per branch give the arm 18 DOFs: the paired factor response pairs it with the box, as
    in the Kuka scenes, and its contact rows come from the general builders rather than the compact
    producer (up to 10 DOFs).
    """
    scene = newton.ModelBuilder()
    base = scene.add_link(mass=2.0, inertia=wp.mat33(np.eye(3) * 0.1))
    joints = [scene.add_joint_fixed(-1, base, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))]
    for side in (-1.0, 1.0):
        parent = base
        for _ in range(depth):
            child = scene.add_link(mass=0.5, inertia=wp.mat33(np.eye(3) * 0.05))
            joints.append(
                scene.add_joint_revolute(
                    parent,
                    child,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform(wp.vec3(0.1, 0.1 * side, -0.1), wp.quat_identity()),
                    target_ke=20.0,
                    target_kd=2.0,
                    armature=0.1,
                )
            )
            parent = child
        scene.add_shape_sphere(parent, radius=0.04)
    scene.add_articulation(joints)
    box = scene.add_body(xform=wp.transform(wp.vec3(0.6, 0.0, 0.06), wp.quat_identity()), mass=0.5)
    scene.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05)
    builder = newton.ModelBuilder()
    builder.replicate(scene, worlds, spacing=(3.0, 3.0, 0.0))
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _graph_and_reset_rollout(model, sparse, cached_dynamics, *, deterministic=False, after_reset=None, **options):
    """Step once, then replay a captured contacting step around a partial-world reset.

    ``after_reset(solver, contacts)`` runs after the first replay that follows
    the reset; seven more replays follow it. ``deterministic`` sorts contacts.
    """
    with patch.dict(SolverFeatherPGS._route_pins, {"evaluator": "ordinary"} if cached_dynamics == "off" else {}):
        solver = _make_solver(model, sparse, **options)
    state, output = model.state(), model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model, convex_collision="ordinary", deterministic=deterministic)
    contacts, control = pipeline.contacts(), model.control()

    def step():
        pipeline.collide(state, contacts)
        solver.step(state, output, control, contacts, PATCH_DT)
        state.assign(output)

    step()
    with wp.ScopedCapture(device=model.device) as capture:
        step()
    for _ in range(8):
        wp.capture_launch(capture.graph)
    solver.reset(state, world_mask=wp.array([True, False], dtype=wp.bool, device=model.device))
    wp.capture_launch(capture.graph)
    if after_reset is not None:
        after_reset(solver, contacts)
    for _ in range(7):
        wp.capture_launch(capture.graph)
    solver.check_constraint_capacity()
    return solver, contacts, state


def _first_patch_step(model, cached_dynamics):
    """Step a fresh standing solver once, eagerly, and copy the poses and twists its release test reads."""
    # The caller compares the opt-in stage-3 snapshot of the predicted velocity.
    with (
        patch.dict(os.environ, {"FEATHER_PGS_DEBUG_SNAPSHOTS": "1"}),
        patch.dict(SolverFeatherPGS._route_pins, {"evaluator": "ordinary"} if cached_dynamics == "off" else {}),
    ):
        solver = _make_solver(model, False)
    state, output = model.state(), model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model, convex_collision="ordinary", deterministic=True)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    launch, released = wp.launch, []

    def release_inputs(kernel, *args, **kwargs):
        if kernel is friction_patches._finish_and_store and kwargs["inputs"][2] is not None:
            released.append((kwargs["inputs"][2].numpy().copy(), kwargs["inputs"][3].numpy().copy()))
        return launch(kernel, *args, **kwargs)

    with patch.object(wp, "launch", release_inputs):
        solver.step(state, output, model.control(), contacts, PATCH_DT)
    return solver, contacts, output, released


def _launched_kernel_keys(model, solver, **pipeline_options):
    """Step once through the collision pipeline and return the key of every kernel the step launched."""
    pipeline = newton.CollisionPipeline(model, **pipeline_options)
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    pipeline.collide(state_in, contacts)
    with launched_kernels() as keys:
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
    return keys


class TestFeatherPGSPathSelection(unittest.TestCase):
    def test_cpu_selection_reasons_are_detached_and_host_only(self):
        """Explain CPU fallbacks without synchronizing or exposing mutable solver state."""
        solver = SolverFeatherPGS(_cpu_model())
        with assert_host_only():
            selected = solver.get_path_selection()
        self.assertIn("CUDA", selected["reasons"]["sparse_mass_matrix"])
        self.assertIn("tree cache admitted", selected["reasons"]["cached_dynamics"])
        self.assertIn("CPU", selected["routes"]["stages"]["factor"]["1"]["reason"])
        # Benchmark results store this record with json.dumps(..., allow_nan=False).
        json.dumps(selected, allow_nan=False)
        selected["reasons"]["cached_dynamics"] = "edited"
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
        self.assertEqual(options["contact_friction_gap_threshold"]["applied"], "inf")
        self.assertEqual(options["joint_limit_activation_gap"]["default"], "inf")
        solver = SolverFeatherPGS(model, angular_damping=0.1, friction_anchor_beta=0.0, pgs_iterations=7)
        selected = solver.get_path_selection()
        options = selected["options"]
        self.assertEqual(
            options["angular_damping"], {"applied": 0.1, "requested": 0.1, "default": 0.05, "source": "argument"}
        )
        self.assertEqual(
            options["friction_anchor_beta"], {"applied": 0.0, "requested": 0.0, "default": None, "source": "argument"}
        )
        # The CPU builds no Jacobi tier.
        self.assertIsNone(selected["routes"]["sweep"]["tiers"])
        self.assertNotIn("grouped", {r["route"] for r in selected["routes"]["stages"]["response"].values()})
        self.assertEqual(selected["routes"]["publication"]["owner"], "cached")
        self.assertEqual(selected["routes"]["friction"]["law"], "point")
        self.assertEqual(selected["routes"]["friction"]["anchor_beta_source"], "argument")
        solver.options = solver.laws.options = solver.factor.options = dataclasses.replace(
            solver.options, pgs_iterations=3
        )
        self.assertEqual(solver.get_path_selection()["options"]["pgs_iterations"]["applied"], 7)

    def test_positional_options_warn_and_build_the_keyword_solver(self):
        """Bind deprecated positional options like keywords, warning only for positional use.

        Every option after model is keyword-only. The deprecation shim binds positional values in
        declaration order, so a positional call over the 34 legacy options must build the solver
        the same keywords build: every requested and applied option, route and reason. A 35th
        positional value would bind an option that was always keyword-only, so it still raises.
        """
        model = _cpu_model()
        parameters = list(inspect.signature(SolverFeatherPGS).parameters.values())
        self.assertEqual(parameters[0].name, "model")
        self.assertEqual({parameter.kind for parameter in parameters[1:]}, {inspect.Parameter.KEYWORD_ONLY})
        legacy = {parameter.name: parameter.default for parameter in parameters[1:35]}
        self.assertEqual((parameters[1].name, parameters[34].name), ("angular_damping", "friction_anchor_beta"))
        legacy.update(
            angular_damping=0.1,
            joint_limit_activation_gap=0.123,
            pgs_iterations=7,
            pgs_beta=0.3,
            dense_max_constraints=16,
            friction_anchor_beta=0.0,
        )
        with self.assertWarnsRegex(
            DeprecationWarning,
            r"^Passing 'angular_damping', 'update_mass_matrix_interval', .*, 'friction_anchor_beta' positionally "
            r"to SolverFeatherPGS\.__init__\(\) is deprecated\. Pass these arguments as keyword arguments instead\.$",
        ) as caught:
            positional = SolverFeatherPGS(model, *legacy.values())
        self.assertEqual(os.path.basename(caught.filename), os.path.basename(__file__))
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            keyword = SolverFeatherPGS(model, **legacy)
            SolverFeatherPGS(model=model)
        self.assertEqual([str(item.message) for item in recorded if "positionally" in str(item.message)], [])
        selected, expected = positional.get_path_selection(), keyword.get_path_selection()
        del selected["measurements"], expected["measurements"]
        self.assertEqual(selected, expected)
        self.assertEqual(selected["options"]["joint_limit_activation_gap"]["applied"], 0.123)
        self.assertEqual(selected["options"]["dense_max_constraints"]["source"], "argument")
        self.assertEqual(parameters[35].name, "pgs_solve_readout")
        with self.assertRaisesRegex(TypeError, r"takes at most 36 positional arguments but 37 were given$"):
            SolverFeatherPGS(model, *legacy.values(), parameters[35].default)

    def test_constructor_warnings_name_the_caller(self):
        """Attribute construction warnings to the constructor's caller, not to the keyword-only shim."""
        model = _cpu_model()
        with _without_solver_environment(), warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            SolverFeatherPGS(model, 0.05, contact_shared_anchor=True)
        messages = {str(item.message).split(" ")[0]: os.path.basename(item.filename) for item in recorded}
        self.assertEqual(messages, dict.fromkeys(("Passing", "Patch"), os.path.basename(__file__)))

    def test_cpu_route_record_reports_known_selections(self):
        """Report the CPU's matrix-free sweep, its free-body rows and the declined owners with their reasons."""
        selected = SolverFeatherPGS(_cpu_free_body_model()).get_path_selection()
        routes = selected["routes"]
        self.assertEqual(routes["articulations_by_dof"], {"6": 1, "1": 1})
        self.assertEqual(routes["sweep"]["route"], "matrix_free_gs")
        # The arm and the free box share a world, which one sweep of the general row solves.
        self.assertTrue(routes["free_rigid_bodies"]["mixed_worlds"])
        self.assertEqual(routes["sweep"]["driver"], "direct")
        self.assertEqual(routes["sweep"]["mf_meta_pack"]["kernel"], "free_body_meta_512")
        # The CPU's one thread keeps the impulses and streams the metadata: no shared memory to stage them in.
        storage = routes["sweep"]["mf_gs_storage"]
        self.assertEqual((storage["metadata"], storage["mf_impulses"]), ("streamed", "resident"))
        self.assertEqual(routes["free_rigid_bodies"]["count"], 1)
        self.assertEqual(routes["free_rigid_bodies"]["owners"], ["matrix_free_rows"])
        # The arm's revolute root is a joint root: the arm and the free box form one mixed held group.
        self.assertEqual(routes["held_tree"]["owner"], "mixed")
        self.assertEqual(routes["contact_rows"]["producer"], "general")
        method = "masked_combined"
        self.assertEqual(routes["contact_rows"]["general_method_by_dof"], {"6": method, "1": method})
        # Every size names its builder and the law it writes; one masked launch pair builds both sizes.
        self.assertEqual(routes["contact_rows"]["law"], "patch")
        masked = {"method": method, "kernels": ["contact_row_prelude", "populate_world_J_masked"], "law": "patch"}
        self.assertEqual(routes["contact_rows"]["builder_by_dof"], {"6": masked, "1": masked})
        self.assertFalse(routes["contact_rows"]["row_packets"]["enabled"])
        # The packet producer implements patch friction; its laws need the joint-limit rows this model lacks.
        self.assertIn("Local packets need joint-limit rows", routes["contact_rows"]["row_packets"]["reason"])
        self.assertIn("CUDA", routes["local_solve"]["reason"])
        self.assertFalse(routes["resolved_worlds"]["enabled"])
        self.assertIn("matrix-free route", routes["resolved_worlds"]["reason"])
        self.assertEqual(routes["publication"]["owner"], "cached")
        self.assertEqual(routes["publication"]["fk_id_cache"], "in_place")
        self.assertEqual([group["free_roots"] for group in routes["held_tree"]["groups"]], [[True], [False]])
        self.assertEqual(routes["velocity_limits"]["joint"], "off")
        self.assertFalse(routes["velocity_limits"]["free_body_rows"])
        self.assertEqual(routes["friction"]["law"], "patch")
        self.assertEqual(routes["friction"]["anchor_beta_source"], "default")
        # Every contact owner of the default route serves patch friction, so the record names none as point-only.
        self.assertNotIn("need point friction", routes["friction"]["reason"])
        # The compact contact boundary and the local packets write the patch row law as well.
        self.assertNotIn("compact contacts", routes["friction"]["reason"])
        self.assertNotIn("local packets", routes["friction"]["reason"])
        self.assertIsNone(routes["sparse_mass_matrix"]["size"])
        self.assertFalse(routes["sparse_mass_matrix"]["active_rows"])
        self.assertIn("CUDA", routes["sparse_mass_matrix"]["reason"])
        self.assertEqual({entry["route"] for entry in routes["stages"]["response"].values()}, {"packed"})
        self.assertEqual(selected["measurements"], {})
        separate = SolverFeatherPGS(_cpu_separate_worlds_model()).get_path_selection()["routes"]
        self.assertFalse(separate["free_rigid_bodies"]["mixed_worlds"])
        self.assertEqual(separate["sweep"]["driver"], "direct")
        self.assertIn("one dispatch", separate["sweep"]["driver_reason"])
        debug = SolverFeatherPGS(_cpu_model(), pgs_debug=True).get_path_selection()["routes"]
        self.assertEqual(debug["sweep"]["driver"], "debug_loop")

    def test_cpu_patch_record_names_the_point_only_owners(self):
        """Name the owners patch friction cannot use, and route the masked rows and the classifier to the patch law.

        Which owner claims a world is recorded where it is decided (routes.stages, each owner's reason); the
        friction record names only the owners that need point friction. Contact torsion keeps the tiers' admission
        whatever the flags say.
        """
        model = _cpu_free_body_model()
        routes = SolverFeatherPGS(model).get_path_selection()["routes"]
        friction = routes["friction"]
        self.assertEqual(friction["law"], "patch")
        # The resolved-world classifier serves patch friction.
        self.assertNotIn("resolved-world classifier", friction["reason"])
        # The general builder's masked rows take the patch row law.
        method = "masked_combined"
        self.assertEqual(routes["contact_rows"]["general_method_by_dof"], {"6": method, "1": method})
        kernels = ["contact_row_prelude", "populate_world_J_masked"]
        builder = {"method": method, "kernels": kernels, "law": "patch"}
        self.assertEqual(routes["contact_rows"]["builder_by_dof"], {"6": builder, "1": builder})
        # Contact torsion keeps ordinary factors and serial sweeps; its rows take the masked builder, as the
        # solver's without torsion do.
        for flags in ({},):
            with self.subTest(torsion=True, **flags), patch.dict(os.environ, flags):
                routes = SolverFeatherPGS(model, contact_torsion=True).get_path_selection()["routes"]
                friction = routes["friction"]
                self.assertEqual((friction["law"], friction["torsion"]), ("patch", True))
                self.assertNotIn("need point friction", friction["reason"])
                self.assertTrue(friction["parallel_rows"]["admitted"])
                self.assertEqual(routes["contact_rows"]["builder_by_dof"], {"6": builder, "1": builder})
        # Point friction's record names no patch route.
        point = SolverFeatherPGS(model, friction_anchor_beta=0.0).get_path_selection()
        self.assertEqual(point["routes"]["friction"]["reason"], "friction_anchor_beta passed explicitly.")
        self.assertEqual(point["routes"]["contact_rows"]["law"], "point")

    def test_cpu_grouped_record_follows_the_response_dispatch(self):
        """Label the sizes the grouped response serves from the data, with their reasons, from the step plan's table:
        every size up to 32 coordinates whose worlds' in-kernel tier rows it skips, and no other."""
        solver = SolverFeatherPGS(_cpu_free_body_model())
        # A stand-in for the CUDA-only grouped response of a row that builds responses (no packed rows): the plan
        # reads host state only, and the record its table.
        solver._grouped_response = True
        for skip, route in ((0, "rows"), (8, "grouped")):
            with self.subTest(skip_rows=skip):
                row = dataclasses.replace(solver.routes.row, tier_rows=skip, packed=False)
                solver._plan = StepPlan.build(row, solver)
                selected = solver.get_path_selection()
                json.dumps(selected, allow_nan=False)
                response = selected["routes"]["stages"]["response"]
                self.assertEqual({entry["route"] for entry in response.values()}, {route})
                reason = "in-kernel tiers need the grouped kernel" if skip else "No in-kernel tier rows to skip"
                self.assertTrue(all(reason in entry["reason"] for entry in response.values()), response)

    def test_cpu_route_record_matches_launched_kernels(self):
        """Tie the record's sweep and row builders to the kernels one CPU step launches."""
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
                # The general row sweeps every iteration of a world's dense and free-body rows in one launch.
                self.assertEqual(sweep["driver"], "direct")
                self.assertEqual([key for key in keys if key.startswith("sweep_")], [sweep["mf_gs"]])
                methods = routes["contact_rows"]["general_method_by_dof"]
                self.assertEqual(keys.count("populate_world_J_for_size"), list(methods.values()).count("per_contact"))
                assert_row_builders_named(self, routes["contact_rows"], keys)

    def test_cpu_record_names_every_launched_row_builder(self):
        """Name in builder_by_dof exactly the contact-row builders one CPU step launches, under both laws.

        The general builder's methods: masked rows in one launch for up to two sizes or per size for
        three, and per-contact rows for a size above the masked builder's 32 DOFs; then keyed rows on
        CPU stand-ins.
        """
        mixed, three, wide = _cpu_free_body_model(), _resting_arms_and_box(1, 2), _resting_arms_and_box(34)
        pair = ("6", "1")
        cases = (
            # (model, friction_anchor_beta, methods by size)
            (mixed, None, dict.fromkeys(pair, "masked_combined")),
            (mixed, 0.0, dict.fromkeys(pair, "masked_combined")),
            (three, None, dict.fromkeys(("6", "2", "1"), "masked")),
            (three, 0.0, dict.fromkeys(("6", "2", "1"), "masked")),
            (wide, None, {"34": "per_contact", "6": "masked_combined"}),
            (wide, 0.0, {"34": "per_contact", "6": "masked_combined"}),
        )
        for model, beta, methods in cases:
            law = "patch" if beta is None else "point"
            with self.subTest(sizes=list(methods), law=law):
                solver = SolverFeatherPGS(model, friction_anchor_beta=beta)
                record = solver.get_path_selection()["routes"]["contact_rows"]
                keys = _launched_kernel_keys(model, solver)
                self.assertEqual((record["producer"], record["law"]), ("general", law))
                self.assertEqual({size: entry["method"] for size, entry in record["builder_by_dof"].items()}, methods)
                self.assertEqual(record["general_method_by_dof"], methods)
                assert_row_builders_named(self, record, keys)
        model = standing_model(device="cpu", worlds=2)
        for law, beta in (("patch", 0.2), ("point", 0.0)):
            with self.subTest(producer="keyed_rows", law=law):
                solver = SolverFeatherPGS(
                    model,
                    friction_anchor_beta=beta,
                    pgs_iterations=4,
                    dense_max_constraints=64,
                    joint_limit_activation_gap=0.1,
                )
                install_cpu_keyed_rows(solver)
                record = solver.get_path_selection()["routes"]["contact_rows"]
                keys = _launched_kernel_keys(model, solver)
                self.assertEqual((record["producer"], record["law"]), ("keyed_rows", law))
                self.assertIsNone(record["general_method_by_dof"])
                self.assertEqual({entry["method"] for entry in record["builder_by_dof"].values()}, {"keyed_rows"})
                # The parallel tiers build the direct rows in the sweep; on CPU every world takes the fallback.
                self.assertEqual(
                    {entry["direct_rows"] for entry in record["builder_by_dof"].values()}, {"routes.sweep.tiers"}
                )
                assert_row_builders_named(self, record, keys)

    def test_host_record_maps_every_producer_plan(self):
        """Map every producer's plan to its builder methods, kernels and law from host attributes alone.

        Stand-ins carry what contact_row_plan and the record read, so the producers that need CUDA are
        pinned on CPU as well: local row packets, keyed rows, the compact contact boundary, native sparse
        factors, the sparse-diagonal owner (with and without its schedule marker) and the compact Jacobian,
        beside the general builder's methods, each under both laws. The cases name every builder method, and every
        kernel they name is one the launch check recognizes.
        """
        layout = KeyedLayout(16, 6, 0, 16, 128, False, 0)
        marker = SimpleNamespace(key="mark_independent_sparse_contact_candidates_stand_in")
        response = SimpleNamespace(key="factor_contact_rows_stand_in")
        prepare, compact = "prepare_world_contact_rows", "populate_world_J_for_compact_size"
        per_contact = ["populate_world_J_for_size"]
        named, methods = set(), set()
        for law in ("point", "patch"):
            point = law == "point"
            masked = ["contact_row_prelude", "populate_world_J_masked"]
            diagonal = {
                "case": "K2",
                "_sparse_diagonal_response_size": 108,
                "_mark_independent_sparse_contact_candidates_kernel": marker,
            }
            schedule = {"_sparse_diagonal_contact_triples": point, "_sparse_diagonal_contact_units": not point}
            sparse = {"case": "K1", "_sparse_mass_matrix_size": 18, "_sparse_contact_response_kernel": response}

            def same(sizes, method, kernels):
                return dict.fromkeys(sizes, (method, kernels))

            cases = (
                # (attributes, sizes, producer, {size: (method, kernels)}); the case is the attributes' "case"
                (
                    {"case": "K3", "_row_packets": SimpleNamespace()},
                    (18, 6),
                    "local_packets",
                    same((18, 6), "local_packets", ["produce_contacts", "materialize_fallback"]),
                ),
                (
                    {"case": "K4", "_keyed_rows": SimpleNamespace(patch=not point, layout=layout)},
                    (16, 6),
                    "keyed_rows",
                    same((16, 6), "keyed_rows", ["scatter_contact_keys", "materialize_general_rows"]),
                ),
                (
                    {"_compact_contact_boundary": True, **diagonal, **schedule},
                    (108, 6),
                    "compact_contact_boundary",
                    {
                        108: ("compact_contact_boundary", ["produce_contacts", marker.key]),
                        6: ("compact_contact_boundary", ["clear_contact_response", "produce_contacts"]),
                    },
                ),
                (
                    {**sparse, "_sparse_prepare_contacts": True},
                    (18,),
                    "sparse_contact_response",
                    {18: ("sparse_response", [response.key])},
                ),
                (
                    {**diagonal, **schedule},
                    (108, 6),
                    "sparse_diagonal",
                    {
                        108: ("sparse_diagonal", [prepare, "sparse_pair_contact_rows_kernel", marker.key]),
                        6: ("compact_jacobian", [prepare, compact]),
                    },
                ),
                (
                    diagonal,
                    (108, 6),
                    "sparse_diagonal",
                    {
                        108: ("sparse_diagonal", [prepare, "sparse_pair_contact_rows_kernel"]),
                        6: ("compact_jacobian", [prepare, compact]),
                    },
                ),
                (
                    {"_compact_contact_jacobian": True},
                    (6, 1),
                    "compact_jacobian",
                    same((6, 1), "compact_jacobian", [prepare, compact]),
                ),
                ({}, (6, 2, 1), "general", same((6, 2, 1), "masked", masked)),
                ({}, (34, 6), "general", {34: ("per_contact", per_contact), 6: ("masked_combined", masked)}),
            )
            for attributes, sizes, producer, expected in cases:
                with self.subTest(producer, law=law, attributes=sorted(attributes)):
                    stand_in = _plan_stand_in(law, sizes, **attributes)
                    plan = contact_row_plan(stand_in)
                    builders = {
                        size: path_selection._row_builder(stand_in, plan, size, solver_module) for size in plan.methods
                    }
                    self.assertEqual(plan.producer, producer)
                    self.assertEqual(
                        {size: (entry["method"], entry["kernels"]) for size, entry in builders.items()}, expected
                    )
                    self.assertEqual({entry["law"] for entry in builders.values()}, {law})
                    pointer = "routes.sweep.tiers" if producer == "keyed_rows" else None
                    self.assertEqual({entry.get("direct_rows") for entry in builders.values()}, {pointer})
                    for entry in builders.values():
                        methods.add(entry["method"])
                        named.update(entry["kernels"])
        self.assertEqual(methods, set(BUILDER_METHODS))
        self.assertEqual({kernel for kernel in named if not is_row_builder(kernel)}, set())

    def test_cpu_row_builder_check_rejects_wrong_order_and_counts(self):
        """Fail the launch check for a record that names the launched kernels in the wrong order or count.

        Three sizes take the masked builder's per-size method, one prelude and one Jacobian launch each.
        A record that lists one size's kernels in reverse, or names one size's launch as part of a combined
        launch, still names the launched set, and the check must reject it.
        """
        model = _resting_arms_and_box(1, 2)
        solver = SolverFeatherPGS(model, friction_anchor_beta=0.0)
        record = solver.get_path_selection()["routes"]["contact_rows"]
        keys = _launched_kernel_keys(model, solver)
        self.assertEqual({entry["method"] for entry in record["builder_by_dof"].values()}, {"masked"})
        assert_row_builders_named(self, record, keys)
        reversed_order = copy.deepcopy(record)
        reversed_order["builder_by_dof"]["2"]["kernels"].reverse()
        combined = copy.deepcopy(record)
        combined["builder_by_dof"]["2"]["method"] = "masked_combined"
        for name, wrong in (("order", reversed_order), ("count", combined)):
            with self.subTest(name), self.assertRaises(AssertionError):
                assert_row_builders_named(self, wrong, keys)

    def test_cpu_row_build_dispatches_on_the_recorded_plan(self):
        """Record and launch the builders of one plan: the row build and the record call the same function.

        Both modules bind one contact_row_plan. A plan that names per-contact rows where the data selects
        combined masked rows, substituted in both, makes the record name the per-contact builder and the row
        build launch it, so neither can follow a decision of its own.
        """
        self.assertIs(path_selection.contact_row_plan, solver_module.contact_row_plan)
        model = _cpu_free_body_model()
        plan_rows = solver_module.contact_row_plan

        def per_contact(solver):
            plan = plan_rows(solver)
            return dataclasses.replace(plan, methods=dict.fromkeys(plan.methods, "per_contact"), combined=())

        solver = SolverFeatherPGS(model)
        record = solver.get_path_selection()["routes"]["contact_rows"]
        masked = _launched_kernel_keys(model, solver)
        with (
            patch.object(solver_module, "contact_row_plan", per_contact),
            patch.object(path_selection, "contact_row_plan", per_contact),
        ):
            forced_record = solver.get_path_selection()["routes"]["contact_rows"]
            forced = _launched_kernel_keys(model, solver)
        self.assertEqual(record["general_method_by_dof"], {"6": "masked_combined", "1": "masked_combined"})
        self.assertEqual(forced_record["general_method_by_dof"], {"6": "per_contact", "1": "per_contact"})
        assert_row_builders_named(self, record, masked)
        assert_row_builders_named(self, forced_record, forced)
        builders = ("contact_row_prelude", "populate_world_J_masked", "populate_world_J_for_size")
        # One combined masked launch for both sizes, then one per-contact launch per size.
        self.assertEqual([key for key in masked if key in builders], list(builders[:2]))
        self.assertEqual([key for key in forced if key in builders], [builders[2]] * 2)

    def test_cpu_route_reports_free_body_velocity_limit_rows(self):
        """Report free-body velocity-limit rows on the CPU, where the row builder allocates them as well."""
        solver = SolverFeatherPGS(_cpu_free_body_model(velocity_limit=1.0))
        limits = solver.get_path_selection()["routes"]["velocity_limits"]
        self.assertTrue(limits["rigid_body_limits"])
        self.assertIsNotNone(solver.rigid_velocity_limit_slot)
        self.assertTrue(limits["free_body_rows"])

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse admission requires CUDA")
    def test_route_record_reports_sparse_rows_and_tree_owner(self):
        """Report the active-row scan, triplets, sweep kernel and held-tree layout of the sparse route."""
        model = _build_model()
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
        self.assertNotIn("contact_transaction", sparse)
        # Every sparse sweep states its occupancy: 64 threads, at most the register budget's 16 blocks.
        threads, blocks = sparse["launch_bounds"]
        self.assertEqual(threads, 64)
        self.assertLessEqual(blocks, 16)
        # The 9-DOF, 32-row tripod is a small world: patch friction keeps its packed full sweep.
        patch_sparse = patch_selected["routes"]["sparse_mass_matrix"]
        self.assertFalse(patch_sparse["active_rows"])
        self.assertFalse(patch_sparse["contact_triplets"])
        if patch_sparse["size"] is not None:
            self.assertIn("friction_anchor_beta", patch_sparse["active_rows_reason"])
            self.assertIn("Small worlds", patch_sparse["contact_triplets_reason"])
        tree = routes["held_tree"]
        self.assertEqual(tree["owner"] is None, point.factor.held is None)
        for group in tree["groups"]:
            self.assertIn(group["lanes"], (4, 8, 16, 32, 64))
            self.assertEqual(len(group["roots"]), len(group["root_families"]))
            self.assertIsNotNone(group["kernels"]["finish"])

    @unittest.skipUnless(wp.is_cuda_available(), "Cached dynamics requires CUDA")
    def test_route_record_reports_mixed_owner_and_paired_response(self):
        """Report the mixed held-tree groups and the paired response of an arm with one free box per world."""
        from newton.tests.test_feather_pgs_mixed_tree_free_body import _build  # noqa: PLC0415

        model = _build(wp.get_device("cuda:0"), links=17)
        with patch.object(SolverFeatherPGS, "_route_pins", {"storage": "dense", "grouping": "per_size"}):
            solver = SolverFeatherPGS(
                model,
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
        self.assertIn("grouping='per_size'", routes["held_tree"]["layout_reason"])
        self.assertEqual(sorted(group["dofs"] for group in routes["held_tree"]["groups"]), [6, 17])
        self.assertIn(routes["free_rigid_bodies"]["owners"][0], ("paired_factor", "paired_response"))
        self.assertEqual(routes["resolved_worlds"]["enabled"], solver._resolved_worlds_owner is not None)

    @unittest.skipUnless(wp.is_cuda_available(), "Matrix-free routes require CUDA")
    def test_route_record_is_json_across_routes(self):
        """Build a JSON-compliant record for every major admitted construction route.

        A tree beside one free body with joint velocity limits takes the Jacobi tiers (case K4) under patch
        friction, as under point friction.
        """
        patch_parallel = ({"enable_joint_velocity_limits": True}, {"free_count": 1})
        cases = (
            ({}, {}),
            ({"friction_anchor_beta": 0.0}, {}),
            ({}, {}),
            ({"sparse_mass_matrix": "off", "friction_anchor_beta": 0.0}, {"free_count": 2}),
            ({"enable_joint_velocity_limits": True, "friction_anchor_beta": 0.0}, {"free_count": 1}),
            ({"sparse_mass_matrix": "off", "friction_anchor_beta": 0.0}, {}),
            ({"enable_joint_velocity_limits": True}, {}),
            # Patch friction builds ring-pooled (linked) parallel tiers, as point friction builds its tiers.
            patch_parallel,
            # 320 dense rows stream the row metadata (5120 bytes); the two-world grid keeps row types and parents
            # resident, under either friction law.
            ({"sparse_mass_matrix": "off", "dense_max_constraints": 320}, {}),
            ({"sparse_mass_matrix": "off", "dense_max_constraints": 320, "friction_anchor_beta": 0.0}, {}),
            # Ordinary dynamics under either friction law.
            ({"sparse_mass_matrix": "off", "cached_dynamics": "off", "friction_anchor_beta": 0.0}, {}),
            ({"sparse_mass_matrix": "off", "cached_dynamics": "off"}, {}),
        )
        built = resident_rows = 0
        for options, model_options in cases:
            with self.subTest(options=options, model=model_options):
                arguments, pins = dict(options), {}
                if arguments.pop("sparse_mass_matrix", None) == "off":
                    pins["storage"] = "dense"
                if arguments.pop("cached_dynamics", None) == "off":
                    pins["evaluator"] = "ordinary"
                try:
                    with patch.dict(SolverFeatherPGS._route_pins, pins):
                        solver = _make_solver(_build_model(**model_options), True, **arguments)
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
                self.assertEqual(pack["launched"], solver._pack_mf_meta_kernel is not None)
                # The MF-GS kernel's name spells the storage the record reports.
                mf_gs, storage = routes["sweep"]["mf_gs"], routes["sweep"]["mf_gs_storage"]
                self.assertEqual(storage is None, mf_gs is None)
                if storage is not None:
                    streamed = storage["metadata"] == "streamed"
                    self.assertEqual("_gmeta" in mf_gs, streamed)
                    self.assertEqual("_smeta" in mf_gs, streamed and storage["row_types_and_parents"] == "resident")
                    self.assertEqual("_glambda" in mf_gs, storage["mf_impulses"] == "streamed")
                    resident_rows += "_gmeta_smeta" in mf_gs
                # Applied routing options agree with the routes they request.
                tiers = routes["sweep"]["tiers"]
                parallel = tiers is not None and tiers["family"] in ("parallel", "keyed_parallel")
                self.assertEqual(solver._ink_rows, tiers["tiers"][-1]["rows"] if parallel else 0)
                # Patch friction is admitted to the parallel tiers, and the case whose class they serve builds them:
                # ring-pooled (linked) tiers, where point friction builds point tiers.
                patch_law = routes["friction"]["law"] == "patch"
                self.assertTrue(routes["friction"]["parallel_rows"]["admitted"])
                if patch_law:
                    self.assertEqual(parallel, options is patch_parallel[0])
                if parallel:
                    self.assertEqual(tiers["linked_rows"], patch_law)
                    # Every tier steps each friction pair isotropically; linked tiers also pool the ring, and with
                    # joint velocity limits they take the last word.
                    last_word = patch_law and bool(applied["enable_joint_velocity_limits"])
                    for tier in tiers["tiers"]:
                        self.assertEqual("_linked" in tier["kernel"], patch_law, tier)
                        self.assertEqual(tier["kernel"].endswith("_lw"), last_word, tier)
                    self.assertEqual(tiers["velocity_limit_pass"], last_word)
                    # Every tier sweeps the equal-accuracy budget.
                    self.assertEqual(tiers["sweeps_rule"], "auto")
                    for tier in tiers["tiers"]:
                        expected = sweep_routes_tiers.tier_sweeps(tier["rows"], applied["pgs_iterations"])
                        self.assertEqual(tier["sweeps"], expected)
                # The grouped kernel serves only sizes whose worlds' in-kernel tier rows it skips.
                if solver._ink_skip_rows() == 0:
                    self.assertNotIn("grouped", {entry["route"] for entry in routes["stages"]["response"].values()})
        self.assertGreaterEqual(built, 8)
        self.assertGreaterEqual(
            resident_rows, 1, "A streamed-metadata route must keep its row types and parents resident"
        )

    def test_kernel_pins_are_validated_on_the_cpu(self):
        """Run the CPU's loop kernels, accept their own pins, and reject an invalid or a CUDA-only pin."""
        model = _cpu_model()
        solver = SolverFeatherPGS(model)
        self.assertEqual(solver.cholesky_kernel, "loop")
        self.assertIn("CPU", solver.get_path_selection()["routes"]["stages"]["factor"]["1"]["reason"])
        with patch.dict(SolverFeatherPGS._route_pins, {"cholesky_kernel": "loop"}):
            self.assertEqual(SolverFeatherPGS(model).cholesky_kernel, "loop")
        for name, value in (
            ("cholesky_kernel", "invalid"),
            ("cholesky_kernel", "tiled"),
            ("trisolve_kernel", "tiled"),
            ("hinv_jt_kernel", "tiled"),
        ):
            with (
                self.subTest(name=name, value=value),
                patch.dict(SolverFeatherPGS._route_pins, {name: value}),
                self.assertRaisesRegex(ValueError, name),
            ):
                SolverFeatherPGS(model)

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse admission requires CUDA")
    def test_sparse_selection_reasons_distinguish_policy_capacity_and_override(self):
        """Explain the actual sparse rejection while retaining the dense fallback."""
        model = _build_model()
        cases = (
            ({}, {"storage": "dense"}, "storage='dense'"),
            ({}, {"placement": "general"}, "placement='general'"),
            ({"dense_max_constraints": _over_budget_rows(model)}, {}, "shared memory"),
        )
        for options, pins, expected in cases:
            with self.subTest(options=options, pins=pins), patch.dict(SolverFeatherPGS._route_pins, pins):
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
                    self.assertNotIn("dense_apply_coordinates", keys)

    @unittest.skipUnless(wp.is_cuda_available(), "Cached dynamics requires CUDA")
    def test_cache_selection_reports_environment_and_explicit_precedence(self):
        """Retain construction-time cache decisions after environment overrides change."""
        model = _build_model()
        with patch.dict(SolverFeatherPGS._route_pins, {"storage": "dense"}):
            disabled = _make_solver(model, True)
        # The pin keeps the native sparse factor out; the dense groups' held tree takes the solver.
        self.assertEqual(disabled.get_path_selection()["sparse_mass_matrix"], "off")
        self.assertIn("storage='dense'", disabled.get_path_selection()["reasons"]["sparse_mass_matrix"])
        self.assertEqual(disabled.get_path_selection()["cached_dynamics"], "on")

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse admission requires CUDA")
    def test_sparse_auto_composes_existing_execution_choices(self):
        """Select qualified split and velocity-limit paths without pins."""
        model = _build_model()
        cases = ({}, {"enable_joint_velocity_limits": True})
        with patch.dict(SolverFeatherPGS._route_pins):
            for options in cases:
                with self.subTest(options=options):
                    solver = _make_solver(model, True, friction_anchor_beta=0.0, **options)
                    self.assertEqual(solver.get_path_selection()["sparse_mass_matrix"], "on")
            for options in ({}, {"enable_joint_velocity_limits": True}):
                with self.subTest(patch=options):
                    # Default patch friction composes with the same sparse factors and tree cache.
                    selected = _make_solver(model, True, **options).get_path_selection()
                    self.assertEqual(selected["sparse_mass_matrix"], "on")
                    self.assertEqual(selected["cached_prediction"], "tree")
            SolverFeatherPGS._route_pins["storage"] = "dense"
            for options in cases:
                with self.subTest(pinned_dense=options):
                    ordinary = _make_solver(model, True, friction_anchor_beta=0.0, **options)
                    self.assertEqual(ordinary.get_path_selection()["sparse_mass_matrix"], "off")

    @unittest.skipUnless(wp.is_cuda_available(), "Cached dynamics requires CUDA")
    def test_cached_dynamics_generic_tree_is_default_on_every_root(self):
        """A fixed arm with a free body is cached by the generic tree owner by default, and so is a revolute world root."""
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
        solver = _make_solver(model, False, friction_anchor_beta=0.0)
        self.assertEqual(solver.get_path_selection()["cached_dynamics"], "on")
        self.assertEqual(solver.get_path_selection()["cached_prediction"], "tree")
        # The evaluator pin retains ordinary dynamics.
        with patch.dict(SolverFeatherPGS._route_pins, {"evaluator": "ordinary"}):
            solver = _make_solver(model, False, friction_anchor_beta=0.0)
        self.assertEqual(solver.get_path_selection()["cached_dynamics"], "off")
        # A revolute joint at the world is a joint root: its coordinates are a descendant's.
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
        solver = _make_solver(model, False, friction_anchor_beta=0.0)
        self.assertIsNotNone(solver.factor.held)
        self.assertEqual(solver.get_path_selection()["cached_dynamics"], "on")

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse admission requires CUDA")
    def test_free_body_worlds_keep_the_dense_route_for_bothplain_laws(self):
        """A branched arm with a free body per world stays dense with the tree cache, under patch friction too.

        Sparse factors admit the arm alone. On the dense route patch friction keeps the bundle that makes
        it worth taking: contact units in the paired factor sweep, masked rows and the resolved-world classifier.
        """
        model = _branched_arm_and_box()
        for law, beta in (("patch", None), ("point", 0.0)):
            with self.subTest(law=law):
                solver = _make_solver(model, True, friction_anchor_beta=beta, dense_max_constraints=96)
                selected = solver.get_path_selection()
                routes = selected["routes"]
                self.assertEqual(routes["friction"]["law"], law)
                self.assertEqual(selected["sparse_mass_matrix"], "off")
                self.assertIn("free rigid bodies", selected["reasons"]["sparse_mass_matrix"])
                self.assertEqual(selected["cached_prediction"], "tree")
                self.assertTrue(routes["paired_response"]["factor_coordinates"])
                # Masked rows build both sizes in one launch pair, under either friction law.
                kernels = ["contact_row_prelude", "populate_world_J_masked"]
                builder = {"method": "masked_combined", "kernels": kernels, "law": law}
                self.assertEqual(routes["contact_rows"]["builder_by_dof"], {"18": builder, "6": builder})
                if law == "patch":
                    self.assertTrue(routes["paired_response"]["contact_units"])
                    self.assertEqual(
                        routes["contact_rows"]["general_method_by_dof"],
                        {"18": "masked_combined", "6": "masked_combined"},
                    )
                    self.assertTrue(routes["resolved_worlds"]["enabled"], routes["resolved_worlds"]["reason"])
                    self.assertIn("patch-friction rows", routes["resolved_worlds"]["reason"])

    @unittest.skipUnless(wp.is_cuda_available(), "The sparse and compact producers require CUDA")
    def test_route_record_names_every_launched_row_builder(self):
        """Name in builder_by_dof exactly the contact-row builders one CUDA step launches, under both laws.

        Native sparse factors, the compact Jacobian and combined masked rows on the dense route. The compact contact
        boundary and the sparse-diagonal owner (both laws), local row packets and keyed rows (both
        laws) are checked in their own modules, keyed rows also on CPU stand-ins
        (test_cpu_record_names_every_launched_row_builder), and every producer's plan-to-record
        mapping on host stand-ins (test_host_record_maps_every_producer_plan).
        """
        tripod, arm, small = _build_model(), _branched_arm_and_box(), _resting_arms_and_box(1, device="cuda:0")
        cases = (
            # (name, model, environment, build(beta), producer, methods by size or None)
            ("sparse", tripod, {}, lambda beta: _make_solver(tripod, True, friction_anchor_beta=beta), None, None),
            (
                "compact_jacobian",
                small,
                {},
                lambda beta: _pinned(small, {"storage": "dense"}, friction_anchor_beta=beta),
                "compact_jacobian",
                dict.fromkeys(("6", "1"), "compact_jacobian"),
            ),
            (
                "masked_combined",
                arm,
                {},
                lambda beta: _make_solver(arm, True, friction_anchor_beta=beta, dense_max_constraints=96),
                "general",
                dict.fromkeys(("18", "6"), "masked_combined"),
            ),
        )
        for name, model, environment, build, producer, methods in cases:
            for law, beta in (("patch", None), ("point", 0.0)):
                with self.subTest(name, law=law), patch.dict(os.environ):
                    os.environ.update(environment)
                    solver = build(beta)
                    record = solver.get_path_selection()["routes"]["contact_rows"]
                    keys = _launched_kernel_keys(model, solver, convex_collision="ordinary")
                    if producer is not None:
                        self.assertEqual(record["producer"], producer)
                        self.assertEqual(
                            {size: entry["method"] for size, entry in record["builder_by_dof"].items()}, methods
                        )
                    self.assertEqual(record["law"], law)
                    assert_row_builders_named(self, record, keys)
                    if name == "sparse" and law == "point":
                        # Point friction takes the tripod's native sparse factors, which prepare its rows.
                        self.assertEqual(record["producer"], "sparse_contact_response")
                        (entry,) = record["builder_by_dof"].values()
                        self.assertEqual(entry["method"], "sparse_response")
                        self.assertEqual(entry["kernels"], [solver._sparse_contact_response_kernel.key])
                    if name == "compact_jacobian":
                        builder = ["prepare_world_contact_rows", "populate_world_J_for_compact_size"]
                        self.assertEqual(
                            {tuple(entry["kernels"]) for entry in record["builder_by_dof"].values()}, {tuple(builder)}
                        )

    @unittest.skipUnless(wp.is_cuda_available(), "The paired factor sweep requires CUDA")
    def test_paired_factor_sweep_serves_the_tree_and_its_body_without_tiers(self):
        """Serve a tree beside a free body with the paired factor sweep (case K5), never the Jacobi tiers.

        Patch friction is admitted to the tiers, but the paired factor sweep admitted on the dense route claims the
        class, so it walks the patch contact units and no tier or keyed row is built.
        """
        model = _branched_arm_and_box()
        solver = _make_solver(model, True, dense_max_constraints=96)
        selected = solver.get_path_selection()
        routes = selected["routes"]
        self.assertEqual(routes["friction"]["law"], "patch")
        self.assertTrue(routes["friction"]["parallel_rows"]["admitted"])
        self.assertEqual(selected["sparse_mass_matrix"], "off")
        self.assertIn("free rigid bodies", selected["reasons"]["sparse_mass_matrix"])
        self.assertTrue(routes["paired_response"]["factor_coordinates"])
        self.assertTrue(routes["paired_response"]["contact_units"])
        self.assertIsNone(routes["sweep"]["tiers"])
        self.assertFalse(routes["contact_rows"]["keyed_rows"]["enabled"])
        self.assertEqual(solver._ink_rows, 0)
        self.assertIn("case K5", solver.routes.row.reasons["K4"])

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
        """Cache the dense route under default patch friction with the ordinary step-0 operators and anchor lifecycle.

        One eager step from the same state isolates the owner: its predicted
        velocity, row Jacobians and biases match the ordinary route's, and each
        route's release test reads, and publishes, forward kinematics of its
        integrated state. The owner's own forward kinematics rewrites the input
        poses by a few ulps, which the resting feet's 1/dt position gain turns
        into about 1e-4 of contact impulse and velocity per ulp, so the step's
        impulses and twists compare within the FIRST_STEP bounds. Every link of
        the standing fixture has its center of mass at its origin, so these
        checks cannot see a center-of-mass twist convention error; the tools
        TestPatchMixedTreeSteps (public-state oracle, offset centers of mass)
        and the mixed-tree free-body tests cover the owner's convention.
        Through graph replay and a partial reset, the routes then keep the same
        stored anchors, each tied to its own solved rows: the standing fixture's
        per-anchor impulses are not determined (its contact rows outnumber its
        degrees of freedom), so they do not compare across routes.
        """
        model = standing_model()
        # Without the owner this is the ordinary patch route, so that the arms differ by the owner alone.
        first = [_first_patch_step(model, choice) for choice in ("off", "on")]
        self.assertIsNone(first[0][0].get_path_selection()["cached_prediction"])
        self.assertEqual(first[1][0].get_path_selection()["cached_prediction"], "tree")
        kinematics = model.state()
        for _solver, _contacts, output, released in first:
            self.assertTrue(released, "The release test must run")
            newton.eval_fk(model, output.joint_q, output.joint_qd, kinematics)
            published = (output.body_q.numpy(), output.body_qd.numpy())
            for label, (body_q, body_qd) in [
                *(("release input", inputs) for inputs in released),
                ("published", published),
            ]:
                np.testing.assert_allclose(body_q, kinematics.body_q.numpy(), rtol=0.0, atol=1e-6, err_msg=label)
                np.testing.assert_allclose(body_qd, kinematics.body_qd.numpy(), rtol=0.0, atol=1e-6, err_msg=label)
        (ordinary, ordinary_contacts, ordinary_output, _), (cached, cached_contacts, cached_output, _) = first
        np.testing.assert_allclose(
            cached._debug_stage3_v_hat.numpy(),
            ordinary._debug_stage3_v_hat.numpy(),
            rtol=1e-5,
            atol=1e-6,
            err_msg="v_hat",
        )
        rows = [
            patch_rows(solver, contacts, PATCH_DT, jacobian=True)
            for solver, contacts in ((ordinary, ordinary_contacts), (cached, cached_contacts))
        ]
        assert_patch_rows_close(*rows)
        assert_patch_impulses_close(*rows, atol=FIRST_STEP_IMPULSE_ATOL, rtol=0.0)
        assert_region_impulses_close(*rows, **FIRST_STEP_REGION_BOUNDS)
        for name in ("joint_q", "body_q"):
            np.testing.assert_allclose(
                getattr(cached_output, name).numpy(),
                getattr(ordinary_output, name).numpy(),
                rtol=0.0,
                atol=2e-6,
                err_msg=name,
            )
        assert_states_close(ordinary_output, cached_output, velocity_atol=FIRST_STEP_VELOCITY_ATOL)

        def reset_restarts_world_0(solver, contacts):
            assert_loaded_anchors(solver, contacts, reset_worlds=(0,))

        runs = [
            _graph_and_reset_rollout(model, False, choice, deterministic=True, after_reset=reset_restarts_world_0)
            for choice in ("off", "on")
        ]
        self.assertIsNone(runs[0][0].get_path_selection()["cached_prediction"])
        self.assertEqual(runs[1][0].get_path_selection()["cached_prediction"], "tree")
        for solver, contacts, _state in runs:
            self.assertTrue(solver.laws.friction_anchors_enabled)
            self.assertIsNone(solver._sparse_mass_matrix_size)
            self.assertIsNone(solver.get_path_selection()["routes"]["sweep"]["tiers"])
            self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
            assert_loaded_anchors(solver, contacts)
            self.assertGreater(assert_ring_coulomb(solver), 0)
            self.assertGreater(assert_history_matches_rows(solver, contacts), 0)
        # Poses that drift by one float32 ulp per step move joint velocity by up to 8.7e-4 over this schedule
        # (CPU split stand-in, 32 seeds).
        assert_states_close(runs[0][2], runs[1][2], velocity_atol=2e-3)
        assert_region_impulses_close(
            *(patch_rows(solver, contacts, PATCH_DT) for solver, contacts, _ in runs), **STANDING_REGION_BOUNDS
        )
        assert_anchor_sites_close(*(anchor_sites(solver, contacts) for solver, contacts, _ in runs), tangent=False)

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse factors require CUDA")
    def test_force_sparse_and_dense_with_graph_replay(self):
        """Run both representations (dense storage pinned) and preserve a contacting articulated rollout."""
        model = _build_model()
        outputs = []
        for choice, pins in (("on", {}), ("off", {"storage": "dense"})):
            solver = _pinned(model, pins, enable_joint_limits=True, pgs_iterations=32)
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
