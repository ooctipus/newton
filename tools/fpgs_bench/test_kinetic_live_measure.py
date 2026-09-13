# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU controls for boundary-only actual graph measurement."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tools.fpgs_bench.kinetic_live_measure import Boundaries, measuring, profile_arguments
from tools.fpgs_bench.kinetic_live_probe import RETIRED


class Array:
    """Expose a small immutable host readback for observer tests."""

    def __init__(self, values):
        self.values = np.asarray(values)

    def numpy(self):
        """Return controlled current data without any device work."""
        return self.values


def fixture():
    """Create only descriptors read by the boundary observer."""

    class Solver:
        """Count the original call independently of observation wrappers."""

        def step(self):
            """Represent one unchanged original step."""
            self.calls += 1
            return "original"

    for name in RETIRED:
        setattr(Solver, name, lambda self: None)
    solver = Solver()
    solver.calls = 0
    solver.model = SimpleNamespace(device=SimpleNamespace(is_capturing=True))
    zero = Array([0])
    slots = SimpleNamespace(
        schedule=SimpleNamespace(status=zero), next_schedule=SimpleNamespace(status=zero), refresh_status=zero
    )
    owner = SimpleNamespace(
        ever_admitted=False,
        last_private=True,
        join=lambda: None,
        calls={1: slots},
        states={1: object(), 2: object()},
        clock=Array([5, 6]),
        held=SimpleNamespace(valid=Array([1, 1])),
        last_call=SimpleNamespace(solve=SimpleNamespace(guard=SimpleNamespace(frame_status=zero))),
    )
    solver._kinetic_world = owner
    solver.constraint_count, solver.mf_constraint_count = Array([3]), zero
    solver.v_hat = solver.v_out = Array([1.0])
    solver._step = 8
    state = SimpleNamespace(**{name: Array([1.0]) for name in ("joint_q", "joint_qd", "body_q", "body_qd")})
    contacts = SimpleNamespace(rigid_contact_count=zero)
    manager = SimpleNamespace(
        _solver=solver, _graph=object(), _state_0=state, _solver_dt=1 / 240, get_contacts=lambda: contacts
    )
    return Solver, solver, owner, manager


class TestLiveMeasure(unittest.TestCase):
    """Guard scope, hook restoration and diagnostic failure semantics."""

    def test_recipe_is_bounded_without_solver_budget_overrides(self):
        """Use exact graph and forty-step discovery sampling only."""
        for mode, worlds, warm, steps in (("graph", 512, "2", "3"), ("performance", 16384, "200", "40")):
            args = profile_arguments(worlds, "out", mode)
            self.assertEqual(args[args.index("--warmup-steps") + 1], warm)
            self.assertEqual(args[args.index("--steps") + 1], steps)
            self.assertNotIn("--no-graph", args)
            self.assertFalse(any("iterations" in value or "capacity" in value for value in args))
        with self.assertRaises(ValueError):
            profile_arguments(16384, "out", "graph")

    def test_hooks_removed_before_first_boundary_and_clock_checked(self):
        """Observe one original call then leave timed calls completely unwrapped."""
        cls, solver, owner, manager = fixture()
        original = cls.step
        report = {}
        observer = Boundaries(cls, report, "performance")
        observer.install()
        self.assertEqual(solver.step(), "original")
        with patch("tools.fpgs_bench.kinetic_live_measure.Observer._admit"):
            observer.check(manager)
            self.assertIs(cls.step, original)
            solver.step()
            self.assertEqual(len(report["pre_timing_calls"]), 1)
            owner.clock = Array([7, 8])
            observer.check(manager)
            self.assertEqual(report["graph_boundaries"][1]["epoch_delta_min"], 2)
        self.assertEqual(solver.calls, 2)

    def test_failure_persisted_without_false_boundary_pass(self):
        """A failed solve status stays a failed record without repair or clearing."""
        cls, solver, owner, manager = fixture()
        report = {}
        observer = Boundaries(cls, report, "performance")
        observer.install()
        solver.step()
        owner.last_call.solve.guard.frame_status = Array([4])
        with (
            patch("tools.fpgs_bench.kinetic_live_measure.Observer._admit"),
            self.assertRaisesRegex(ValueError, "status"),
        ):
            observer.check(manager)
        self.assertFalse(report["graph_boundaries"][0]["passed"])
        self.assertEqual(owner.last_call.solve.guard.frame_status.numpy()[0], 4)
        self.assertFalse(observer.replacements)

    def test_installer_and_solver_restored_after_exception(self):
        """Failed construction does not leave benchmark hooks installed."""
        cls, _, _, _ = fixture()
        original_step = cls.step

        def original_install(*args):
            """Stand in for the unchanged boundary installer."""
            return None

        checked = SimpleNamespace(install_boundary_check=original_install)
        with self.assertRaisesRegex(RuntimeError, "controlled"), measuring(cls, {}, checked, "graph"):
            raise RuntimeError("controlled")
        self.assertIs(cls.step, original_step)
        self.assertIs(checked.install_boundary_check, original_install)


if __name__ == "__main__":
    unittest.main()
