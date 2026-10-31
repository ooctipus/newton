# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for FeatherPGS contact-capacity handling."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS


def _build_model():
    """Build one articulated body for contact-capacity tests."""
    builder = newton.ModelBuilder()
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([joint])
    return builder.finalize(device="cpu")


class TestFeatherPGSContactCapacity(unittest.TestCase):
    def test_step_grows_scratch_to_larger_contacts(self):
        """Grow the solver's per-contact scratch to a larger contact buffer before writing it."""
        model = _build_model()
        solver = SolverFeatherPGS(model)
        rows = solver.placement.contact_pool.ready + 64
        contacts = newton.Contacts(rigid_contact_max=rows, soft_contact_max=0, device=model.device)

        solver.step(model.state(), model.state(), model.control(), contacts, 1.0 / 60.0)
        self.assertEqual(solver.placement.contact_pool.ready, rows)
        self.assertEqual(solver.placement.contact_slot.shape, (rows,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
