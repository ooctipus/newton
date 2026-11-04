# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The public API of SolverFeatherPGS: diagnostics, construction, capacity and teardown.

A contract module: :mod:`newton.tests.test_feather_pgs_contract_lint` keeps it on the public API.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch, sentinel

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_test_devices

# ---------------------------------------------------------------------------------------------------------------------
# diagnostics
#
# The ``diagnostics`` option of SolverFeatherPGS: what it reads out, never what it computes.
#
# Every check reads the public API: the constructor, :meth:`~newton.solvers.SolverFeatherPGS.step`, the options record
# of :meth:`~newton.solvers.SolverFeatherPGS.get_path_selection` and the readouts
# (:meth:`~newton.solvers.SolverFeatherPGS.read_audit`,
# :meth:`~newton.solvers.SolverFeatherPGS.constraint_row_watermarks`). The stage-3 snapshots are read by the names
# external NaN dumpers use.
# ---------------------------------------------------------------------------------------------------------------------

Diagnostics = SolverFeatherPGS.Diagnostics
STATE_FIELDS = ("joint_q", "joint_qd", "body_q", "body_qd")
SNAPSHOTS = ("_debug_stage3_qd_work", "_debug_stage3_joint_qdd", "_debug_stage3_v_hat")
DT = 1.0 / 240.0
STEPS = 6
# Separate CUDA solver instances may order float atomic accumulations differently; CPU runs serially.
CUDA_INSTANCE_ATOL = 2.0e-6
# Every diagnostic on but NVTX, which needs the ``nvtx`` package (Warp's ScopedTimer emits the ranges). The sweep
# readout counts where the sweeps carry its counters.
ALL_ON = Diagnostics(
    snapshots=True,
    row_watermark=True,
    overflow_warning=True,
    solve_readout="sweeps",
    audit=True,
    stage_clocks=True,
    sync_timings=(1, 2),
)
ALL_OFF = Diagnostics(overflow_warning=False)


def _scene(device):
    """A two-link arm leaning onto the ground beside a dropped box: joints, limits and contacts in one world."""
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.6
    upper = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01))
    builder.add_shape_box(upper, hx=0.05, hy=0.05, hz=0.2)
    lower = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01))
    builder.add_shape_sphere(lower, radius=0.06, xform=wp.transform((0.0, 0.0, -0.2), wp.quat_identity()))
    builder.add_articulation(
        [
            builder.add_joint_revolute(
                -1,
                upper,
                axis=newton.Axis.Y,
                parent_xform=wp.transform((0.0, 0.0, 0.62), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5)),
                child_xform=wp.transform((0.0, 0.0, 0.2), wp.quat_identity()),
                limit_lower=-1.0,
                limit_upper=1.0,
            ),
            builder.add_joint_revolute(
                upper,
                lower,
                axis=newton.Axis.Y,
                parent_xform=wp.transform((0.0, 0.0, -0.2), wp.quat_identity()),
                child_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
            ),
        ]
    )
    box = builder.add_body(xform=wp.transform((0.6, 0.0, 0.11), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _rollout(model, diagnostics, steps=STEPS):
    """Step ``model`` from rest; return the solver and the published states, one per step."""
    solver = SolverFeatherPGS(model, enable_joint_limits=True, pgs_iterations=16, diagnostics=diagnostics)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    trajectory = []
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        trajectory.append({name: getattr(state_0, name).numpy().copy() for name in STATE_FIELDS})
    return solver, contacts, trajectory


def test_diagnostics_change_no_published_state(test, device):
    """Publish the same states with every diagnostic on as with all of them off: bitwise on the CPU."""
    model = _scene(device)
    with contextlib.redirect_stdout(io.StringIO()):
        _, contacts, on = _rollout(model, ALL_ON)
    _, _, off = _rollout(model, ALL_OFF)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    for step, (expected, actual) in enumerate(zip(off, on, strict=True)):
        for name, values in expected.items():
            test.assertTrue(np.all(np.isfinite(values)), f"step {step} {name}")
            if wp.get_device(device).is_cuda:
                np.testing.assert_allclose(actual[name], values, rtol=0.0, atol=CUDA_INSTANCE_ATOL)
            else:
                np.testing.assert_array_equal(actual[name], values, err_msg=f"step {step} {name}")


def test_snapshots_hold_the_step_by_name(test, device):
    """Allocate the stage-3 snapshots only when asked, and fill them with finite values of every coordinate."""
    model = _scene(device)
    solver, _, _ = _rollout(model, Diagnostics(snapshots=True), steps=2)
    for name in SNAPSHOTS:
        values = getattr(solver, name).numpy()
        test.assertEqual(values.shape, (model.joint_dof_count,), name)
        test.assertTrue(np.all(np.isfinite(values)), name)
    solver, _, _ = _rollout(model, None, steps=1)
    for name in SNAPSHOTS:
        test.assertIsNone(getattr(solver, name), name)
    record = solver.get_path_selection()["routes"]["diagnostics"]["stage3_snapshots"]
    test.assertFalse(record["enabled"])


def test_readouts_answer_only_when_asked(test, device):
    """Read the audit and the row watermarks of a solver that asked for them; refuse the audit otherwise."""
    model = _scene(device)
    solver, _, _ = _rollout(model, Diagnostics(audit=True, row_watermark=True))
    audit = solver.read_audit()
    record = solver.get_path_selection()["routes"]["diagnostics"]["audit"]
    test.assertEqual(set(audit), set(record["slots"]))
    # A world whose rows exist only inside its sweep (row packets, keyed rows) reads NaN, as the record says.
    if record["reason"] == "Reads every row of every world.":
        test.assertEqual(int(audit["steps"][0]), STEPS)
    else:
        test.assertTrue(np.isnan(audit["gap"][0]), record["reason"])
    test.assertGreater(solver.constraint_row_watermarks()["contact_high_water"], 0)
    solver, _, _ = _rollout(model, None, steps=1)
    with test.assertRaisesRegex(RuntimeError, "Diagnostics"):
        solver.read_audit()
    test.assertEqual(solver.constraint_row_watermarks()["contact_high_water"], 0)


def test_sync_timings_print_their_steps(test, device):
    """Print each timed stage of the steps ``sync_timings`` names, and of no other step."""
    model = _scene(device)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        _rollout(model, Diagnostics(sync_timings=(2, 2)))
    steps = {
        int(line.split("step=")[1].split()[0]) for line in printed.getvalue().splitlines() if "[fpgs-sync]" in line
    }
    test.assertEqual(steps, {2, 3})


class TestFeatherPGSDiagnostics(unittest.TestCase):
    def test_record_reports_the_diagnostics_as_applied(self):
        """Record the request and the applied diagnostics: a readout no sweep counts applies as off."""
        model = _scene("cpu")
        requested = Diagnostics(solve_readout="sweeps", audit=True)
        solver = SolverFeatherPGS(model, diagnostics=requested)
        record = solver.get_path_selection()["options"]["diagnostics"]
        self.assertEqual(record["requested"], dataclasses.asdict(requested))
        self.assertEqual(record["source"], "argument")
        self.assertTrue(record["applied"]["audit"])
        counted = bool(solver.get_path_selection()["routes"]["sweep"]["readout"]["counted"])
        self.assertEqual(record["applied"]["solve_readout"], "sweeps" if counted else "off")
        default = SolverFeatherPGS(model).get_path_selection()["options"]["diagnostics"]
        self.assertEqual(default["source"], "default")
        self.assertEqual(default["applied"], dataclasses.asdict(Diagnostics()))

    def test_invalid_diagnostics_are_refused(self):
        """Refuse an unknown readout, a malformed timing window and anything but a Diagnostics record."""
        with self.assertRaisesRegex(ValueError, "solve_readout"):
            Diagnostics(solve_readout="every_sweep")
        for window in ((1,), (-1, 2), (0, -1)):
            with self.subTest(window=window), self.assertRaisesRegex(ValueError, "sync_timings"):
                Diagnostics(sync_timings=window)
        with self.assertRaisesRegex(TypeError, "Diagnostics"):
            SolverFeatherPGS(_scene("cpu"), diagnostics={"audit": True})


devices = get_test_devices()
for _test in (
    test_diagnostics_change_no_published_state,
    test_snapshots_hold_the_step_by_name,
    test_readouts_answer_only_when_asked,
    test_sync_timings_print_their_steps,
):
    add_function_test(TestFeatherPGSDiagnostics, _test.__name__, _test, devices=devices)


# ---------------------------------------------------------------------------------------------------------------------
# hetero guard
#
# Construction of SolverFeatherPGS on heterogeneous multi-world models.
#
# Heterogeneous multi-world models (worlds whose per-world DOF counts differ)
# construct on the matrix-free and split paths, as homogeneous models do.
# ---------------------------------------------------------------------------------------------------------------------


def _make_chain_world(n_links: int) -> newton.ModelBuilder:
    """A fixed-base serial chain of ``n_links`` revolute links (n_links DOFs)."""
    builder = newton.ModelBuilder()
    joints = []
    prev = -1
    for i in range(n_links):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.15, hy=0.03, hz=0.03)
        if prev == -1:
            parent_xform = wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity())
        else:
            parent_xform = wp.transform(p=wp.vec3(0.15, 0.0, 0.0), q=wp.quat_identity())
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=parent_xform,
                child_xform=wp.transform(p=wp.vec3(-0.15, 0.0, 0.0), q=wp.quat_identity()),
                label=f"chain_joint_{i}",
            )
        )
        prev = link
    builder.add_articulation(joints, label="chain")
    return builder


def _build_model(world_link_counts: list[int], device: str) -> newton.Model:
    """One chain world per entry in ``world_link_counts``, plus a ground plane."""
    scene = newton.ModelBuilder()
    for n_links in world_link_counts:
        scene.add_world(_make_chain_world(n_links))
    scene.add_ground_plane()
    return scene.finalize(device=device)


class TestFeatherPGSHeteroGuard(unittest.TestCase):
    """Heterogeneous and homogeneous models construct."""

    @classmethod
    def setUpClass(cls):
        if wp.get_cuda_device_count() == 0:
            raise unittest.SkipTest("SolverFeatherPGS construction tests require cuda:0")
        cls.device = "cuda:0"
        # Worlds alternate a 1-link pendulum and a 3-link chain: per-world DOF
        # counts [1, 3, 1, 3] -> heterogeneous.
        cls.hetero_model = _build_model([1, 3, 1, 3], cls.device)
        # All 3-link chains: per-world DOF counts uniform -> homogeneous.
        cls.homogeneous_model = _build_model([3, 3, 3, 3], cls.device)

    def test_hetero_constructs(self):
        solver = SolverFeatherPGS(self.hetero_model)
        self.assertEqual(solver.get_path_selection()["routes"]["sweep"]["route"], "matrix_free_gs")

    def test_homogeneous_constructs(self):
        solver = SolverFeatherPGS(self.homogeneous_model)
        self.assertEqual(solver.get_path_selection()["routes"]["sweep"]["route"], "matrix_free_gs")


# ---------------------------------------------------------------------------------------------------------------------
# safety
#
# Verify physical invariants needed by persistent FeatherPGS contacts.
# ---------------------------------------------------------------------------------------------------------------------


class TestFeatherPGSCapacityStatus(unittest.TestCase):
    def test_capacity_failure_is_observable_without_watermarks(self):
        """Expose dropped contacts even when optional detailed telemetry is off."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cpu")
        solver = SolverFeatherPGS(model, mf_max_constraints=1)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state_in, state_out = model.state(), model.state()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            solver.check_constraint_capacity()
        self.assertTrue(solver.placement.constraint_overflow.numpy()[0])
        solver.reset(state_out)
        solver.check_constraint_capacity()


# ---------------------------------------------------------------------------------------------------------------------
# teardown
#
# Keep solver teardown independent of cyclically finalized stream objects.
# ---------------------------------------------------------------------------------------------------------------------


class TestFeatherPGSTeardown(unittest.TestCase):
    def test_drain_context_without_dereferencing_finalized_streams(self):
        """Drain queued work without trusting a stream's surviving Python wrapper."""
        owner = SimpleNamespace(
            model=SimpleNamespace(device=sentinel.device),
            sweep=SimpleNamespace(local_pair_stream=sentinel.finalized_stream),
        )
        with (
            patch.object(wp, "synchronize_device") as drain,
            patch.object(wp, "synchronize_stream", side_effect=AssertionError("finalized stream handle")),
        ):
            SolverFeatherPGS.__del__(owner)
        drain.assert_called_once_with(sentinel.device)

    def test_partial_construction_and_interpreter_shutdown(self):
        """Allow missing construction state and already unavailable CUDA runtime."""
        with patch.object(wp, "synchronize_device") as drain:
            SolverFeatherPGS.__del__(SimpleNamespace())
        drain.assert_not_called()
        for error in (AttributeError, RuntimeError, TypeError):
            with self.subTest(error=error), patch.object(wp, "synchronize_device", side_effect=error):
                SolverFeatherPGS.__del__(SimpleNamespace(model=SimpleNamespace(device=sentinel.device)))


# ---------------------------------------------------------------------------------------------------------------------
# reset
#
# ``reset(state, world_mask)`` restarts the masked worlds: each steps from ``state`` as a solver built fresh would, with
# no impulse, friction anchor or cached kinematics carried from before; every other world steps as if no reset
# happened. A model change restarts every world's impulse history (the patch law's anchors are geometry and stay) and
# keeps the capacity status.
# ---------------------------------------------------------------------------------------------------------------------


def _resting_boxes(device, worlds=2):
    """Boxes resting on the ground, one per world, whose warm-started contacts carry impulses step to step."""
    template = newton.ModelBuilder()
    box = template.add_body(xform=wp.transform((0.0, 0.0, 0.0995), wp.quat_identity()))
    template.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    template.add_ground_plane()
    builder = newton.ModelBuilder()
    builder.replicate(template, worlds)
    return builder.finalize(device=device)


class _Run:
    """A solver with its own deterministic, matching collision pipeline, stepping its own state pair."""

    def __init__(self, model, state, **options):
        self.model = model
        self.solver = SolverFeatherPGS(model, pgs_warmstart=True, pgs_iterations=16, **options)
        self.pipeline = newton.CollisionPipeline(model, contact_matching="latest", deterministic=True)
        self.contacts = self.pipeline.contacts()
        self.state, self.out = model.state(), model.state()
        self.state.assign(state)

    def step(self):
        self.pipeline.collide(self.state, self.contacts)
        self.state.clear_forces()
        self.solver.step(self.state, self.out, self.model.control(), self.contacts, DT)
        self.state, self.out = self.out, self.state
        return {name: getattr(self.state, name).numpy().copy() for name in STATE_FIELDS}


def _world(values: dict, model, world: int) -> dict:
    """The coordinates and bodies of ``world`` (the replicated worlds hold equal, contiguous blocks)."""
    worlds = model.world_count
    return {name: np.split(array, worlds)[world] for name, array in values.items()}


def _spun(model, state, world: int):
    """``state`` with ``world``'s resting box spinning about the normal: a state no step of the run produced, still in
    contact, so carried impulses would change its step."""
    spun = model.state()
    spun.assign(state)
    joint_qd = spun.joint_qd.numpy()
    np.split(joint_qd, model.world_count)[world][5] = 3.0
    spun.joint_qd.assign(joint_qd)
    newton.eval_fk(model, spun.joint_q, spun.joint_qd, spun)
    return spun


def test_reset_restarts_only_the_masked_worlds(test, device):
    """Step a reset world as a fresh solver does and the other world as a solver that never reset (bitwise on the
    CPU; on CUDA within the bound between solver instances)."""
    model = _resting_boxes(device)
    start = model.state()
    newton.eval_fk(model, start.joint_q, start.joint_qd, start)
    run, twin = _Run(model, start), _Run(model, start)
    for _ in range(8):
        run.step()
        twin.step()
    test.assertGreater(int(run.contacts.rigid_contact_count.numpy()[0]), 0)
    restart = _spun(model, run.state, world=0)
    run.state.assign(restart)
    twin.state.assign(restart)
    run.solver.reset(run.state, wp.array((True, False), dtype=wp.bool, device=model.device))
    fresh = _Run(model, restart)
    for step in range(3):
        stepped, continued, restarted = run.step(), twin.step(), fresh.step()
        if step == 0:
            # The history the reset drops changes the step: the twin's world 0, which kept it, steps otherwise.
            test.assertFalse(
                np.array_equal(_world(continued, model, 0)["joint_qd"], _world(restarted, model, 0)["joint_qd"])
            )
        for world, expected in ((0, restarted), (1, continued)):
            actual, reference = _world(stepped, model, world), _world(expected, model, world)
            for name in STATE_FIELDS:
                message = f"step {step} world {world} {name}"
                if wp.get_device(device).is_cuda:
                    np.testing.assert_allclose(actual[name], reference[name], rtol=0.0, atol=CUDA_INSTANCE_ATOL)
                else:
                    np.testing.assert_array_equal(actual[name], reference[name], err_msg=message)


def test_reset_is_capturable(test, device):
    """Replay a captured reset and step as the eager calls run them."""
    if not wp.get_device(device).is_cuda:
        test.skipTest("CUDA graphs need CUDA")
    model = _resting_boxes(device)
    start = model.state()
    newton.eval_fk(model, start.joint_q, start.joint_qd, start)
    eager, captured = _Run(model, start), _Run(model, start)
    mask = wp.array((True, False), dtype=wp.bool, device=model.device)
    for run in (eager, captured):
        for _ in range(4):
            run.step()
    eager.solver.reset(eager.state, mask)
    expected = eager.step()
    captured.solver.reset(captured.state, mask)
    wp.synchronize_device(model.device)
    with wp.ScopedCapture(device=model.device) as capture:
        captured.solver.reset(captured.state, mask)
        captured.pipeline.collide(captured.state, captured.contacts)
        captured.solver.step(captured.state, captured.out, model.control(), captured.contacts, DT)
    wp.capture_launch(capture.graph)
    for name in STATE_FIELDS:
        np.testing.assert_allclose(
            getattr(captured.out, name).numpy(), expected[name], rtol=0.0, atol=CUDA_INSTANCE_ATOL, err_msg=name
        )


class TestFeatherPGSReset(unittest.TestCase):
    def test_reset_refuses_a_mask_of_another_world_count(self):
        """Refuse a mask whose length is not the world count, whatever the solver carries."""
        model = _resting_boxes("cpu")
        for warm in (False, True):
            with self.subTest(warm_start=warm):
                solver = SolverFeatherPGS(model, pgs_warmstart=warm)
                with self.assertRaisesRegex(ValueError, "expected 2"):
                    solver.reset(model.state(), wp.ones(3, dtype=wp.bool, device="cpu"))
                solver.reset(model.state(), wp.array((True, False), dtype=wp.bool, device="cpu"))

    def test_model_change_restarts_every_world_and_keeps_the_status(self):
        """Step every world as a fresh solver does after a model change, and keep a capacity failure latched."""
        model = _resting_boxes("cpu")
        start = model.state()
        newton.eval_fk(model, start.joint_q, start.joint_qd, start)
        for flags in (
            newton.ModelFlags.BODY_INERTIAL_PROPERTIES,
            newton.ModelFlags.JOINT_DOF_PROPERTIES,
            newton.ModelFlags.SHAPE_PROPERTIES,
        ):
            with self.subTest(flags=flags):
                run = _Run(model, start, friction_anchor_beta=0.0)
                for _ in range(8):
                    run.step()
                run.solver.notify_model_changed(flags)
                fresh = _Run(model, run.state, friction_anchor_beta=0.0)
                stepped, expected = run.step(), fresh.step()
                for name in STATE_FIELDS:
                    np.testing.assert_array_equal(stepped[name], expected[name], err_msg=name)
        # One matrix-free row a world cannot hold a box's contacts: the failure stays latched through a model change.
        solver = SolverFeatherPGS(model, mf_max_constraints=1)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(start, contacts)
        solver.step(start, model.state(), model.control(), contacts, DT)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            solver.check_constraint_capacity()
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            solver.check_constraint_capacity()


for _test in (test_reset_restarts_only_the_masked_worlds, test_reset_is_capturable):
    add_function_test(TestFeatherPGSReset, _test.__name__, _test, devices=devices)

if __name__ == "__main__":
    unittest.main(verbosity=2)
