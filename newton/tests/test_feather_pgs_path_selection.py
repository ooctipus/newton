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
from newton._src.solvers.feather_pgs import environment_flags, friction_patches, path_selection
from newton._src.solvers.feather_pgs import solver_feather_pgs as solver_module
from newton._src.solvers.feather_pgs.configuration import resolve_kernel_choices
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


def _plan_stand_in(law, sizes=(108, 6), **attributes):
    """A host stand-in with the attributes contact_row_plan and the route record read; no owner is admitted."""
    values = {
        "_friction_anchors_enabled": law == "patch",
        "_point_contact_paths": law == "point",
        "_patch_dense_paths": law == "patch",
        "size_groups": list(sizes),
        "_row_packets": None,
        "_keyed_rows": None,
        "_compact_contact_boundary": False,
        "_sparse_mass_matrix_size": None,
        "_sparse_prepare_contacts": False,
        "_sparse_diagonal_contact_solve": False,
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
    solver = _make_solver(model, sparse, cached_dynamics=cached_dynamics, **options)
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
    with patch.dict(os.environ, {"FEATHER_PGS_DEBUG_SNAPSHOTS": "1"}):
        solver = _make_solver(model, False, cached_dynamics=cached_dynamics, mf_gs_incremental_rows=0)
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
        self.assertIn("CPU", selected["reasons"]["sparse_mass_matrix"])
        self.assertIn("CPU", selected["reasons"]["cached_dynamics"])
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
        self.assertEqual(
            options["pgs_kernel"], {"applied": "loop", "requested": "auto", "default": "auto", "source": "default"}
        )
        self.assertIsNone(options["pgs_chunk_size"]["applied"])
        self.assertEqual(options["contact_friction_gap_threshold"]["applied"], "inf")
        self.assertEqual(options["joint_limit_activation_gap"]["default"], "inf")
        self.assertFalse(options["fuse_joint_velocity_limits"]["applied"])
        self.assertEqual(options["sparse_mass_matrix"]["applied"], "off")
        self.assertEqual(options["cached_dynamics"]["applied"], "off")
        # Split mode builds no matrix-free tier, CPU builds no stream, and the deprecated
        # propagation_cached_response selects nothing.
        self.assertEqual(
            options["mf_gs_incremental_rows"], {"applied": 0, "requested": 32, "default": 32, "source": "default"}
        )
        for name in ("use_parallel_streams", "propagation_cached_response"):
            self.assertEqual(options[name], {"applied": False, "requested": True, "default": True, "source": "default"})
        solver = SolverFeatherPGS(
            model, angular_damping=0.1, friction_anchor_beta=0.0, pgs_iterations=7, mf_gs_parallel_rows=48
        )
        selected = solver.get_path_selection()
        options = selected["options"]
        self.assertEqual(
            options["angular_damping"], {"applied": 0.1, "requested": 0.1, "default": 0.05, "source": "argument"}
        )
        self.assertEqual(
            options["friction_anchor_beta"], {"applied": 0.0, "requested": 0.0, "default": None, "source": "argument"}
        )
        self.assertEqual(
            options["mf_gs_parallel_rows"], {"applied": 0, "requested": 48, "default": 0, "source": "argument"}
        )
        self.assertIsNone(selected["routes"]["sweep"]["tiers"])
        self.assertNotIn("grouped", {r["route"] for r in selected["routes"]["stages"]["response"].values()})
        self.assertEqual(selected["routes"]["publication"]["owner"], "ordinary")
        self.assertEqual(selected["routes"]["friction"]["law"], "point")
        self.assertEqual(selected["routes"]["friction"]["anchor_beta_source"], "argument")
        solver.pgs_iterations = 3
        self.assertEqual(solver.get_path_selection()["options"]["pgs_iterations"]["applied"], 7)

    def test_positional_options_warn_and_build_the_keyword_solver(self):
        """Bind deprecated positional options like keywords, warning only for positional use.

        Every option after model is keyword-only. The deprecation shim binds positional values in
        declaration order, so a positional call over the 52 legacy options must build the solver
        the same keywords build: every requested and applied option, route and reason. A 53rd
        positional value would bind an option that was always keyword-only, so it still raises.
        """
        model = _cpu_model()
        parameters = list(inspect.signature(SolverFeatherPGS).parameters.values())
        self.assertEqual(parameters[0].name, "model")
        self.assertEqual({parameter.kind for parameter in parameters[1:]}, {inspect.Parameter.KEYWORD_ONLY})
        legacy = {parameter.name: parameter.default for parameter in parameters[1:53]}
        self.assertEqual((parameters[1].name, parameters[52].name), ("angular_damping", "friction_anchor_beta"))
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
        self.assertEqual(parameters[53].name, "pgs_change_tolerance")
        with self.assertRaisesRegex(TypeError, r"takes at most 54 positional arguments but 55 were given$"):
            SolverFeatherPGS(model, *legacy.values(), parameters[53].default)

    def test_constructor_warnings_name_the_caller(self):
        """Attribute construction warnings to the constructor's caller, not to the keyword-only shim."""
        model = _cpu_model()
        with _without_solver_environment(), warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            SolverFeatherPGS(model, 0.05, contact_friction_anchor_limit=2, contact_shared_anchor=True)
        messages = {str(item.message).split(" ")[0]: os.path.basename(item.filename) for item in recorded}
        self.assertEqual(
            messages,
            dict.fromkeys(("Passing", "contact_friction_anchor_limit", "Patch"), os.path.basename(__file__)),
        )

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
        self.assertEqual(routes["sweep"]["dense"], {"kernels": ["sweep_loop"]})
        self.assertEqual(routes["sweep"]["free_body_matrix_free"], "sweep_mf_loop")
        # Only the matrix-free step packs row metadata, and only its GS kernel chooses a storage.
        self.assertIsNone(routes["sweep"]["mf_meta_pack"])
        self.assertIsNone(routes["sweep"]["mf_gs_storage"])
        self.assertEqual(routes["free_rigid_bodies"]["count"], 1)
        self.assertEqual(routes["free_rigid_bodies"]["owners"], ["split_matrix_free_rows"])
        self.assertIsNone(routes["held_tree"]["owner"])
        self.assertIn("CPU", routes["held_tree"]["reason"])
        self.assertEqual(routes["contact_rows"]["producer"], "general")
        method = "masked_combined"
        self.assertEqual(routes["contact_rows"]["general_method_by_dof"], {"6": method, "1": method})
        # Every size names its builder and the law it writes; one masked launch pair builds both sizes.
        self.assertEqual(routes["contact_rows"]["law"], "patch")
        masked = {"method": method, "kernels": ["contact_row_prelude", "populate_world_J_masked"], "law": "patch"}
        self.assertEqual(routes["contact_rows"]["builder_by_dof"], {"6": masked, "1": masked})
        self.assertFalse(routes["contact_rows"]["row_packets"]["enabled"])
        # Patch friction reaches the packet owner, which serves the local owners' worlds (CUDA only): none here.
        self.assertIn("local owners' worlds; none is built", routes["contact_rows"]["row_packets"]["reason"])
        self.assertIn("CUDA", routes["local_solve"]["reason"])
        self.assertFalse(routes["resolved_worlds"]["enabled"])
        self.assertIn("matrix-free route", routes["resolved_worlds"]["reason"])
        self.assertEqual(routes["publication"]["owner"], "ordinary")
        self.assertEqual(routes["publication"]["fk_id_cache"], "off")
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
        self.assertIn("CPU", routes["sparse_mass_matrix"]["reason"])
        self.assertEqual({entry["route"] for entry in routes["stages"]["response"].values()}, {"rows"})
        self.assertEqual(selected["measurements"], {})
        separate = SolverFeatherPGS(_cpu_separate_worlds_model()).get_path_selection()["routes"]
        self.assertFalse(separate["free_rigid_bodies"]["mixed_worlds"])
        self.assertEqual(separate["sweep"]["driver"], "dense_split")
        self.assertIn("free-body rows", separate["sweep"]["driver_reason"])
        debug = SolverFeatherPGS(_cpu_model(), pgs_debug=True).get_path_selection()["routes"]
        self.assertEqual(debug["sweep"]["driver"], "debug_loop")

    def test_cpu_patch_record_names_the_point_only_owners(self):
        """Name the owners patch friction cannot use, and route the masked rows and the classifier to the patch law.

        Which owner claims a world is recorded where it is decided (routes.stages, each owner's reason); the
        friction record names only the owners that need point friction. Contact torsion takes none of the patch
        routes whatever the flags say.
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
        # Contact torsion keeps ordinary factors and serial sweeps under every patch route flag; its rows take the
        # masked builder, as the zero-radius solver's do.
        for flags in ({}, {"FEATHER_PGS_PATCH_PARALLEL_ROWS": "1"}):
            with (
                self.subTest(torsion=True, **flags),
                patch.dict(os.environ, flags),
                # Construct on the CPU, as TestTorsionNeutralRoute does; nothing is stepped.
                patch("newton._src.solvers.feather_pgs.contact_torsion._validate_torsion_mode"),
            ):
                routes = SolverFeatherPGS(model, contact_torsion_radius=0.01).get_path_selection()["routes"]
                friction = routes["friction"]
                self.assertEqual((friction["law"], friction["torsion"]), ("patch", True))
                self.assertIn("need point friction", friction["reason"])
                self.assertFalse(friction["parallel_rows"]["admitted"])
                self.assertEqual(routes["contact_rows"]["builder_by_dof"], {"6": builder, "1": builder})
        # Point friction's record names no patch route.
        point = SolverFeatherPGS(model, friction_anchor_beta=0.0).get_path_selection()
        self.assertEqual(point["routes"]["friction"]["reason"], "friction_anchor_beta passed explicitly.")
        self.assertEqual(point["routes"]["contact_rows"]["law"], "point")

    def test_cpu_grouped_record_follows_the_response_dispatch(self):
        """Label the sizes the grouped response serves from the data, with their reasons, from the step plan's table:
        every size up to 32 coordinates whose worlds' in-kernel tier rows it skips, and no other."""
        solver = SolverFeatherPGS(_cpu_free_body_model())
        # A stand-in for the CUDA-only grouped response: the plan reads host state only, and the record its table.
        solver._grouped_response = True
        for skip, route in ((0, "rows"), (8, "grouped")):
            with self.subTest(skip_rows=skip), patch.object(solver, "_ink_skip_rows", lambda skip=skip: skip):
                solver._plan = StepPlan.build(solver)
                selected = solver.get_path_selection()
                json.dumps(selected, allow_nan=False)
                response = selected["routes"]["stages"]["response"]
                self.assertEqual({entry["route"] for entry in response.values()}, {route})
                reason = "in-kernel tiers need the grouped kernel" if skip else "No in-kernel tier rows to skip"
                self.assertTrue(all(reason in entry["reason"] for entry in response.values()), response)

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
                assert_row_builders_named(self, routes["contact_rows"], keys)
                self.assertEqual(routes["velocity_limits"]["root_clamp"], "clamp_free_root_velocity_limits" in keys)

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
                "_sparse_diagonal_contact_solve": True,
                "_sparse_diagonal_response_size": 108,
                "_mark_independent_sparse_contact_candidates_kernel": marker,
            }
            schedule = {"_sparse_diagonal_contact_triples": point, "_sparse_diagonal_contact_units": not point}
            sparse = {"_sparse_mass_matrix_size": 18, "_sparse_contact_response_kernel": response}

            def same(sizes, method, kernels):
                return dict.fromkeys(sizes, (method, kernels))

            cases = (
                # (attributes, sizes, producer, {size: (method, kernels)})
                (
                    {"_row_packets": SimpleNamespace()},
                    (18, 6),
                    "local_packets",
                    same((18, 6), "local_packets", ["produce_contacts", "materialize_fallback"]),
                ),
                (
                    {"_keyed_rows": SimpleNamespace(patch=not point, layout=layout)},
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

    def test_plan_refuses_sparse_factors_beside_the_sparse_diagonal_owner(self):
        """Refuse a plan in which native sparse factors and the sparse-diagonal owner both own contact rows.

        _setup_sparse_mass_matrix declines sparse factors where the sparse-diagonal owner owns the
        response. The plan ranks sparse factors first and the row build launches the diagonal response
        for the sparse-diagonal producer only, so both owners together would drop that response.
        """
        for law in ("point", "patch"):
            with self.subTest(law=law):
                both = _plan_stand_in(
                    law,
                    _sparse_mass_matrix_size=6,
                    _sparse_diagonal_contact_solve=True,
                    _sparse_diagonal_response_size=108,
                )
                with self.assertRaisesRegex(RuntimeError, "sparse-diagonal contact owner"):
                    contact_row_plan(both)
                # Either owner alone produces its own rows.
                for producer, owner in (
                    ("sparse_contact_response", {"_sparse_mass_matrix_size": 6}),
                    ("sparse_diagonal", {"_sparse_diagonal_contact_solve": True}),
                ):
                    plan = contact_row_plan(_plan_stand_in(law, _sparse_diagonal_response_size=108, **owner))
                    self.assertEqual(plan.producer, producer)

    def test_cpu_split_route_reports_free_body_velocity_limit_rows(self):
        """Report free-body velocity-limit rows in split mode, where the row builder allocates them as well."""
        solver = SolverFeatherPGS(_cpu_free_body_model(velocity_limit=1.0))
        limits = solver.get_path_selection()["routes"]["velocity_limits"]
        self.assertTrue(limits["rigid_body_limits"])
        self.assertIsNotNone(solver.rigid_velocity_limit_slot)
        self.assertTrue(limits["free_body_rows"])
        # Its free-body sweep clamps them statelessly (test_feather_pgs_split_velocity_limits).
        self.assertEqual(limits["free_body_sweep_law"], "stateless")
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
        # Point friction keeps the staged transaction of admitted triplets by default.
        self.assertEqual(sparse["contact_transaction"], "staged" if sparse["contact_triplets"] else None)
        self.assertIsNone(sparse["contact_transaction_launch_bounds"])
        # The 9-DOF, 32-row tripod is a small world: patch friction keeps its packed full sweep.
        patch_sparse = patch_selected["routes"]["sparse_mass_matrix"]
        self.assertFalse(patch_sparse["active_rows"])
        self.assertFalse(patch_sparse["contact_triplets"])
        self.assertIsNone(patch_sparse["contact_transaction"])
        if patch_sparse["size"] is not None:
            self.assertIn("friction_anchor_beta", patch_sparse["active_rows_reason"])
            self.assertIn("small worlds", patch_sparse["contact_triplets_reason"])
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
        """Build a JSON-compliant record for every major admitted construction route.

        Every case runs with ``FEATHER_PGS_PATCH_PARALLEL_ROWS`` unset, its default (on); the
        patch-friction parallel case also runs opted out (``=0``).
        """
        patch_parallel = ({"sparse_mass_matrix": "off", "mf_gs_parallel_rows": 48}, {"free_count": 2})
        cases = (
            ({}, {}),
            ({"friction_anchor_beta": 0.0}, {}),
            ({"pgs_mode": "split"}, {}),
            ({"sparse_mass_matrix": "off", "friction_anchor_beta": 0.0}, {"free_count": 2}),
            (
                {"sparse_mass_matrix": "off", "mf_gs_parallel_rows": 48, "friction_anchor_beta": 0.0},
                {"free_count": 2},
            ),
            ({"sparse_mass_matrix": "off", "friction_anchor_beta": 0.0}, {}),
            ({"enable_joint_velocity_limits": True}, {}),
            # Patch friction builds ring-pooled (linked) parallel tiers, as point friction builds its tiers; opted
            # out it builds none, whatever the requested budget.
            patch_parallel,
            ({"sparse_mass_matrix": "off", "pgs_schedule": "physx_grasp", "friction_anchor_beta": 0.0}, {}),
            # 320 dense rows stream the row metadata (5120 bytes); the two-world grid keeps row types and parents
            # resident, under either friction law.
            ({"sparse_mass_matrix": "off", "dense_max_constraints": 320}, {}),
            ({"sparse_mass_matrix": "off", "dense_max_constraints": 320, "friction_anchor_beta": 0.0}, {}),
            # Ordinary dynamics under either friction law.
            ({"sparse_mass_matrix": "off", "cached_dynamics": "off", "friction_anchor_beta": 0.0}, {}),
            ({"sparse_mass_matrix": "off", "cached_dynamics": "off"}, {}),
        )
        built = opted_out = resident_rows = 0
        for (options, model_options), switch in [(case, None) for case in cases] + [(patch_parallel, "0")]:
            with self.subTest(options=options, model=model_options, switch=switch):
                try:
                    # Set the switch as the arm says, whatever the exported environment holds.
                    with patch.dict(os.environ):
                        os.environ.pop("FEATHER_PGS_PATCH_PARALLEL_ROWS", None)
                        if switch is not None:
                            os.environ["FEATHER_PGS_PATCH_PARALLEL_ROWS"] = switch
                        solver = _make_solver(_build_model(**model_options), True, **options)
                except (ValueError, NotImplementedError):
                    continue  # The record describes admitted constructors only.
                built += 1
                opted_out += switch is not None
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
                self.assertEqual(applied["mf_gs_parallel_rows"], tiers["tiers"][-1]["rows"] if parallel else 0)
                # Patch friction is admitted to the parallel tiers unless opted out, and the case that requests
                # them builds them: ring-pooled (linked) tiers, where point friction builds point tiers.
                patch_law = routes["friction"]["law"] == "patch"
                self.assertEqual(routes["friction"]["parallel_rows"]["admitted"], not patch_law or switch != "0")
                if patch_law:
                    self.assertEqual(parallel, options is patch_parallel[0] and switch != "0")
                if parallel:
                    self.assertEqual(tiers["linked_rows"], patch_law)
                    # Every tier steps each friction pair isotropically; linked tiers also pool the ring.
                    for tier in tiers["tiers"]:
                        self.assertEqual(tier["kernel"].endswith("_linked"), bool(patch_law), tier)
                        self.assertEqual("_linked" in tier["kernel"], patch_law, tier)
                    # Every tier sweeps the automatic budget, at least mf_gs_parallel_sweeps.
                    self.assertEqual(tiers["sweeps_rule"], "auto")
                    self.assertTrue(all(t["sweeps"] >= applied["mf_gs_parallel_sweeps"] for t in tiers["tiers"]))
                if options.get("pgs_schedule") == "physx_grasp" and routes["sweep"]["route"] == "matrix_free_gs":
                    self.assertEqual(routes["sweep"]["driver"], "physx_grasp")
                # The grouped kernel serves only sizes whose worlds' in-kernel tier rows it skips.
                if solver._ink_skip_rows() == 0:
                    self.assertNotIn("grouped", {entry["route"] for entry in routes["stages"]["response"].values()})
        self.assertGreaterEqual(built, 8)
        self.assertEqual(opted_out, 1, "The opted-out patch-friction parallel arm must build")
        self.assertGreaterEqual(
            resident_rows, 1, "A streamed-metadata route must keep its row types and parents resident"
        )

    def test_private_override_validation_and_cpu_precedence(self):
        """Validate effective overrides while preserving explicit choices and CPU fallbacks."""
        model = _cpu_model()
        with patch.dict(SolverFeatherPGS._kernel_overrides, {"cholesky_kernel": "invalid"}):
            with self.assertRaisesRegex(ValueError, "cholesky_kernel"):
                SolverFeatherPGS(model)
            explicit = SolverFeatherPGS(model, cholesky_kernel="loop").get_path_selection()
            self.assertIn("Explicit", explicit["routes"]["stages"]["factor"]["1"]["reason"])
        with patch.dict(SolverFeatherPGS._kernel_overrides, {"cholesky_kernel": "tiled"}):
            solver = SolverFeatherPGS(model)
            self.assertEqual(solver.cholesky_kernel, "loop")
            reason = solver.get_path_selection()["routes"]["stages"]["factor"]["1"]["reason"]
            self.assertIn("CPU", reason)
            self.assertIn("override", reason)

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse admission requires CUDA")
    def test_sparse_selection_reasons_distinguish_policy_capacity_and_override(self):
        """Explain the actual sparse rejection while retaining the dense fallback."""
        model = _build_model()
        cases = (
            ({"sparse_mass_matrix": "off"}, "Explicit"),
            ({"trisolve_kernel": "loop"}, "explicit dense"),
            ({"dense_max_constraints": _over_budget_rows(model)}, "shared memory"),
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
                    self.assertNotIn("dense_apply_coordinates", keys)

    @unittest.skipUnless(wp.is_cuda_available(), "Cached dynamics requires CUDA")
    def test_cache_selection_reports_environment_and_explicit_precedence(self):
        """Retain construction-time cache decisions after environment overrides change."""
        model = _build_model()
        with patch.dict(os.environ, {"FEATHER_PGS_TREE_KINETIC_STATE": "0"}):
            disabled = _make_solver(model, True)
            forced = _make_solver(model, True, cached_dynamics="on")
        # The switch disables the sparse factor's held group, and with it the native sparse factor, which only the
        # held tree evaluates.
        self.assertEqual(disabled.get_path_selection()["sparse_mass_matrix"], "off")
        self.assertIn("FEATHER_PGS_TREE_KINETIC_STATE", disabled.get_path_selection()["reasons"]["sparse_mass_matrix"])
        self.assertEqual(forced.get_path_selection()["sparse_mass_matrix"], "on")
        self.assertEqual(forced.get_path_selection()["cached_dynamics"], "on")
        self.assertIn("Explicit", forced.get_path_selection()["reasons"]["cached_dynamics"])

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse admission requires CUDA")
    def test_sparse_auto_composes_existing_execution_choices(self):
        """Select qualified split and velocity-limit paths without private switches."""
        model = _build_model()
        flags = ("FEATHER_PGS_SPARSE_SPLIT", "FEATHER_PGS_SPARSE_VELOCITY_LIMITS")
        cases = ({"pgs_mode": "split"}, {"enable_joint_velocity_limits": True})
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

    @unittest.skipUnless(wp.is_cuda_available(), "Cached dynamics requires CUDA")
    def test_cached_dynamics_overrides_environment_and_dense_conflicts(self):
        """Honor explicit cache choices and retain automatic dense kernel requests."""
        model = _build_model()
        # FEATHER_PGS_TREE_KINETIC_STATE=0 under auto declines the native sparse factor, which only the held tree
        # evaluates; the dense groups' held tree takes the solver.
        for choice, legacy, expected, sparse in (
            ("on", "0", True, True),
            ("off", "1", False, False),
            ("auto", "0", True, False),
        ):
            with self.subTest(choice=choice), patch.dict(os.environ, {"FEATHER_PGS_TREE_KINETIC_STATE": legacy}):
                solver = _make_solver(model, True, cached_dynamics=choice)
                selected = solver.get_path_selection()
                self.assertEqual(solver._sparse_mass_matrix_size is not None, sparse)
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

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse admission requires CUDA")
    def test_free_body_worlds_keep_the_dense_route_for_both_laws(self):
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
                lambda beta: SolverFeatherPGS(small, friction_anchor_beta=beta, sparse_mass_matrix="off"),
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
    def test_paired_factor_sweep_excludes_a_requested_patch_tier(self):
        """Serve a free-body world with the paired factor sweep even when a requested patch tier declined sparse factors.

        Patch friction admits a tier that mf_gs_parallel_rows requests by default (the test sets
        FEATHER_PGS_PATCH_PARALLEL_ROWS=1 so that an exported opt-out cannot change the route), and the tier declines
        sparse factors before the free-body rule does, as the friction record says. The paired factor sweep then
        admitted on the dense route excludes the tiers and keyed rows, so it walks the patch contact units and no
        tier is built.
        """
        model = _branched_arm_and_box()
        with patch.dict(os.environ, {"FEATHER_PGS_PATCH_PARALLEL_ROWS": "1"}):
            solver = _make_solver(
                model, True, dense_max_constraints=96, mf_gs_parallel_rows=32, mf_gs_parallel_matrix_free=True
            )
        selected = solver.get_path_selection()
        routes = selected["routes"]
        self.assertEqual(routes["friction"]["law"], "patch")
        self.assertTrue(routes["friction"]["parallel_rows"]["admitted"])
        self.assertEqual(selected["sparse_mass_matrix"], "off")
        self.assertIn("competing", selected["reasons"]["sparse_mass_matrix"])
        self.assertTrue(routes["paired_response"]["factor_coordinates"])
        self.assertTrue(routes["paired_response"]["contact_units"])
        self.assertIsNone(routes["sweep"]["tiers"])
        self.assertFalse(routes["contact_rows"]["keyed_rows"]["enabled"])
        self.assertEqual(selected["options"]["mf_gs_parallel_rows"]["applied"], 0)

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
        # Without the owner this is the ordinary patch route: the incremental tier, which also serves
        # patch friction, stays off in both arms so that they differ by the owner alone.
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
            _graph_and_reset_rollout(
                model, False, choice, deterministic=True, after_reset=reset_restarts_world_0, mf_gs_incremental_rows=0
            )
            for choice in ("off", "on")
        ]
        self.assertIsNone(runs[0][0].get_path_selection()["cached_prediction"])
        self.assertEqual(runs[1][0].get_path_selection()["cached_prediction"], "tree")
        for solver, contacts, _state in runs:
            self.assertTrue(solver._friction_anchors_enabled)
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

    def test_matrix_free_resolution_ignores_an_explicit_pgs_kernel(self):
        """Resolve an explicit matrix-free pgs_kernel like auto on every device, returning its warning.

        Split mode still selects the kernel with its device rules, and matrix-free mode still
        rejects unknown names and an explicit delassus_kernel.
        """
        auto = dict.fromkeys(
            (
                "cholesky_kernel",
                "trisolve_kernel",
                "hinv_jt_kernel",
                "delassus_kernel",
                "pgs_kernel",
                "sparse_mass_matrix",
                "cached_dynamics",
            ),
            "auto",
        )
        for is_cpu, automatic in ((False, "tiled_row"), (True, "loop")):
            for kernel in ("loop", "tiled_row"):
                with self.subTest(is_cpu=is_cpu, pgs_kernel=kernel):
                    resolved, explicit_dense, reasons, ignored = resolve_kernel_choices(
                        {**auto, "pgs_kernel": kernel}, {}, is_cpu=is_cpu, pgs_mode="matrix_free"
                    )
                    reference = resolve_kernel_choices(auto, {}, is_cpu=is_cpu, pgs_mode="matrix_free")
                    self.assertEqual(reference[3], {})
                    self.assertEqual(resolved, reference[0])
                    self.assertEqual(resolved["pgs_kernel"], automatic)
                    self.assertFalse(explicit_dense)
                    self.assertEqual(list(ignored), ["pgs_kernel"])
                    self.assertIn(f"pgs_kernel={kernel!r} is ignored in matrix_free mode", ignored["pgs_kernel"])
                    self.assertTrue(reasons["pgs_kernel"].startswith(f"Explicit pgs_kernel={kernel!r} is ignored"))
        resolved, _, _, ignored = resolve_kernel_choices(
            {**auto, "pgs_kernel": "loop"}, {}, is_cpu=False, pgs_mode="split"
        )
        self.assertEqual((resolved["pgs_kernel"], ignored), ("loop", {}))
        for is_cpu, mode, choice, message in (
            (True, "split", {"pgs_kernel": "tiled_row"}, "requires CUDA"),
            (False, "matrix_free", {"pgs_kernel": "typo"}, "must be one of"),
            (False, "matrix_free", {"delassus_kernel": "tiled"}, "delassus_kernel is unused in matrix_free"),
        ):
            with self.subTest(is_cpu=is_cpu, pgs_mode=mode, choice=choice), self.assertRaisesRegex(ValueError, message):
                resolve_kernel_choices({**auto, **choice}, {}, is_cpu=is_cpu, pgs_mode=mode)

    def test_contact_only_pgs_kernels_are_deprecated(self):
        """Warn that the removed contact-only kernels have no effect and build the automatic solver.

        ``tiled_contact`` and ``streaming`` select the automatic kernel in either mode, with the
        default patch friction and joint limits, and ``pgs_chunk_size``, which tuned the streaming
        kernel, is ignored. CPU construction of matrix-free mode stops at its device check, after
        the warnings; there the request warns only that it is deprecated, not that matrix-free mode
        ignores it.
        """
        model = _cpu_model()
        reference = SolverFeatherPGS(model, enable_joint_limits=True).get_path_selection()
        for kernel in ("tiled_contact", "streaming"):
            with self.subTest(pgs_kernel=kernel):
                with _without_solver_environment(), warnings.catch_warnings(record=True) as recorded:
                    warnings.simplefilter("always")
                    solver = SolverFeatherPGS(model, pgs_kernel=kernel, pgs_chunk_size=16, enable_joint_limits=True)
                deprecated = [item for item in recorded if issubclass(item.category, DeprecationWarning)]
                self.assertEqual(len(deprecated), 2, [str(item.message) for item in recorded])
                self.assertIn(f"pgs_kernel={kernel!r} is deprecated and has no effect", str(deprecated[0].message))
                self.assertIn("pgs_chunk_size is deprecated and has no effect", str(deprecated[1].message))
                for item in deprecated:
                    self.assertEqual(os.path.basename(item.filename), os.path.basename(__file__))
                self.assertEqual(solver.pgs_kernel, "loop")
                self.assertTrue(solver._friction_anchors_enabled)
                selected = solver.get_path_selection()
                for key in ("routes", "pgs_kernel"):
                    self.assertEqual(selected[key], reference[key], key)
                self.assertEqual(
                    selected["options"]["pgs_kernel"],
                    {"applied": "loop", "requested": kernel, "default": "auto", "source": "argument"},
                )
                self.assertIsNone(selected["options"]["pgs_chunk_size"]["applied"])
                with _without_solver_environment(), warnings.catch_warnings(record=True) as recorded:
                    warnings.simplefilter("always")
                    with self.assertRaisesRegex(NotImplementedError, "pgs_mode='matrix_free' requires CUDA"):
                        SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_kernel=kernel)
                self.assertEqual(
                    [item.category for item in recorded if "pgs_kernel" in str(item.message)], [DeprecationWarning]
                )

    @unittest.skipUnless(wp.is_cuda_available(), "Matrix-free mode requires CUDA")
    def test_matrix_free_ignores_an_explicit_pgs_kernel(self):
        """Build the automatic matrix-free solver for an explicit pgs_kernel and warn that it is ignored.

        An explicit friction law keeps a point kernel from selecting point friction, so every route
        and kernel plan must equal the automatic solver's; only the request, its option record and
        the pgs_kernel reason name the ignored kernel.
        """
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_articulation([builder.add_joint_revolute(-1, body)])
        model = builder.finalize(device="cuda:0")
        for beta in (0.0, 0.2):
            reference = SolverFeatherPGS(model, pgs_mode="matrix_free", friction_anchor_beta=beta).get_path_selection()
            for kernel in ("loop", "tiled_row"):
                with self.subTest(friction_anchor_beta=beta, pgs_kernel=kernel):
                    with self.assertWarnsRegex(UserWarning, f"pgs_kernel='{kernel}' is ignored in matrix_free mode"):
                        solver = SolverFeatherPGS(
                            model, pgs_mode="matrix_free", pgs_kernel=kernel, friction_anchor_beta=beta
                        )
                    self.assertEqual(solver.pgs_kernel, "tiled_row")
                    selected = solver.get_path_selection()
                    for key in ("routes", "pgs_kernel", "sparse_mass_matrix", "cached_dynamics"):
                        self.assertEqual(selected[key], reference[key], key)
                    self.assertEqual(selected["requested"], {**reference["requested"], "pgs_kernel": kernel})
                    self.assertEqual(
                        selected["options"]["pgs_kernel"],
                        {"applied": "tiled_row", "requested": kernel, "default": "auto", "source": "argument"},
                    )
                    self.assertEqual(
                        selected["reasons"]["pgs_kernel"],
                        f"{reference['reasons']['pgs_kernel']} Explicit pgs_kernel={kernel!r} is ignored in "
                        "matrix_free mode.",
                    )

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse factors require CUDA")
    def test_public_dense_kernels_override_sparse_and_fused_dispatch(self):
        """Select actual dense kernels while leaving unspecified choices automatic."""
        model = _build_model()
        solver = SolverFeatherPGS(model, pgs_mode="matrix_free", trisolve_kernel="loop")
        selected = solver.get_path_selection()
        self.assertEqual(selected["sparse_mass_matrix"], "off")
        self.assertEqual(selected["routes"]["stages"]["trisolve"]["9"]["route"], "loop")
        self.assertEqual(selected["requested"]["hinv_jt_kernel"], "auto")
        split = SolverFeatherPGS(model, pgs_mode="split", delassus_kernel="par_row_col")
        self.assertFalse(split._execution_plan.hinv_jt_fused_sizes)
        for kwargs, message in (
            ({"sparse_mass_matrix": "on", "trisolve_kernel": "loop"}, "conflicts"),
            ({"delassus_kernel": "tiled"}, "delassus_kernel"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                SolverFeatherPGS(model, pgs_mode="matrix_free", **kwargs)
        # pgs_kernel is no dense factor choice: matrix-free mode ignores it and keeps automatic admission.
        with self.assertWarnsRegex(UserWarning, "pgs_kernel='loop' is ignored in matrix_free mode"):
            ignored = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_kernel="loop")
        self.assertEqual(
            ignored.get_path_selection()["sparse_mass_matrix"],
            SolverFeatherPGS(model, pgs_mode="matrix_free").get_path_selection()["sparse_mass_matrix"],
        )

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
