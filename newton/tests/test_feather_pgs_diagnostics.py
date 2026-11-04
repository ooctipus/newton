# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The ``diagnostics`` option of SolverFeatherPGS: what it reads out, never what it computes.

Every check reads the public API: the constructor, :meth:`~newton.solvers.SolverFeatherPGS.step`, the options record
of :meth:`~newton.solvers.SolverFeatherPGS.get_path_selection` and the readouts
(:meth:`~newton.solvers.SolverFeatherPGS.read_audit`,
:meth:`~newton.solvers.SolverFeatherPGS.constraint_row_watermarks`). The stage-3 snapshots are read by the names
external NaN dumpers use.
"""

import contextlib
import dataclasses
import io
import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_test_devices

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
    test.assertEqual(int(audit["steps"][0]), STEPS)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
