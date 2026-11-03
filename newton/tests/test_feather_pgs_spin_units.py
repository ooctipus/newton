# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact units with spin rows (contact torsion): the unit schedules against the serial row loop.

The paired factor sweep (:func:`.sweep_routes_local.paired_factor_kernel`) solves a world's contacts row by row
(the serial row loop: each friction pair at its first tangent, with its contact's spin row as the friction triple),
as normal/tangent/tangent/spin quads after a prefix (``contact_triples``), or as contact units in slot order
(``contact_units``: a loaded unit on its Gram block, one reduction and one apply). The three are the same
Gauss-Seidel sweep in exact arithmetic. On random fixtures, solved to convergence, they reach the same impulses and
velocity to the rounding of their reductions, and every friction triple ends on or inside its elliptic cone.
"""

import inspect
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import sweep_routes_local
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_TORSION,
)
from newton.tests.feather_pgs_law_harness import diagonal_walk, leave_walk, row_lines
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

WORLDS, CONTACTS, PRIMARY, SECONDARY = 8, 3, 7, 6
DOFS = PRIMARY + SECONDARY
CAPACITY = 1 + 4 * CONTACTS
SCHEDULES = ({}, {"contact_triples": True}, {"contact_units": True})


def _fixture(rng):
    """Random worlds of a joint-limit prefix row and ``CONTACTS`` contact quads (normal, tangents, spin) over the
    factor rows ``Z`` (identity factors, so the rows are the Jacobians), with friction and torsional coefficients over
    a decade and start impulses some of which lie outside their cones."""
    types = [PGS_CONSTRAINT_TYPE_JOINT_LIMIT]
    parents = [-1]
    for contact in range(CONTACTS):
        head = 1 + 4 * contact
        types += [PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION]
        types.append(PGS_CONSTRAINT_TYPE_TORSION)
        parents += [-1, head, head, head]
    rows = rng.normal(size=(WORLDS, CAPACITY, DOFS)).astype(np.float32)
    mu = np.zeros((WORLDS, CAPACITY), np.float32)
    for contact in range(CONTACTS):
        head = 1 + 4 * contact
        friction = rng.uniform(0.2, 2.0, WORLDS)
        mu[:, head] = friction
        mu[:, head + 1 : head + 3] = friction[:, None]
        mu[:, head + 3] = friction * rng.uniform(0.005, 0.05, WORLDS)
    return {
        "rows": rows,
        "types": np.tile(np.array(types, np.int32), (WORLDS, 1)),
        "parents": np.tile(np.array(parents, np.int32), (WORLDS, 1)),
        "mu": mu,
        "rhs": rng.normal(scale=0.5, size=(WORLDS, CAPACITY)).astype(np.float32),
        "incident": rng.normal(scale=0.5, size=(WORLDS, CAPACITY)).astype(np.float32),
        "diag": (np.sum(rows.astype(np.float64) ** 2, axis=2) + 1.0e-3).astype(np.float32),
        "impulses": np.abs(rng.normal(scale=0.2, size=(WORLDS, CAPACITY))).astype(np.float32),
        "velocity": rng.normal(scale=0.5, size=WORLDS * DOFS).astype(np.float32),
    }


def _solve(fixture, schedule, device, iterations):
    """Run the paired factor sweep with spin rows under ``schedule`` and the step's leave of its change (identity
    factors: a unit diagonal walk, :func:`.diagonal_walk`); return the impulses and velocity."""
    kernel = sweep_routes_local.paired_factor_kernel(
        CAPACITY, DOFS, PRIMARY, SECONDARY, str(device.arch), torsion=True, **schedule
    )

    def array(value, dtype=None):
        value = np.asarray(value)
        if dtype is None:
            dtype = wp.int32 if value.dtype == np.int32 else wp.float32
        return wp.array(value, dtype=dtype, device=device)

    values = {
        "world_count": WORLDS,
        "world_constraint_count": array(np.full(WORLDS, CAPACITY, np.int32)),
        # The row table: each world's rows on its own line.
        "dense_first": row_lines(device),
        # The prefix is the limit row; the contact quads follow it.
        "dense_phase_bounds": array(np.tile(np.array([1, 1], np.int32), (WORLDS, 1))),
        "world_dof_indices": array(np.arange(WORLDS * DOFS, dtype=np.int32).reshape(WORLDS, DOFS)),
        "rhs_bias": array(fixture["rhs"]),
        "row_incident": array(fixture["incident"]),
        "world_diag": array(fixture["diag"]),
        "world_impulses": array(fixture["impulses"].copy()),
        "factor_rows": array(fixture["rows"]),
        "world_row_type": array(fixture["types"]),
        "world_row_parent": array(fixture["parents"]),
        "world_row_mu": array(fixture["mu"]),
        "world_mf_constraint_count": array(np.zeros(WORLDS, np.int32)),
        "primary_offset_by_world": array(np.zeros(WORLDS, np.int32)),
        "secondary_offset_by_world": array(np.full(WORLDS, PRIMARY, np.int32)),
        "iterations": iterations,
        "omega": 1.0,
        "v_out": array(fixture["velocity"].copy()),
        "walk": diagonal_walk(np.ones((WORLDS, DOFS), np.float32), device),
    }
    arguments = [values[name] for name in inspect.signature(kernel.func).parameters]
    blocks = (WORLDS + 1) // 2
    wp.launch_tiled(kernel, dim=[blocks], inputs=arguments, block_dim=64, device=device)
    leave_walk(values["walk"], values["v_out"], device)
    return values["world_impulses"].numpy(), values["v_out"].numpy()


def test_unit_schedules_match_the_serial_row_loop(test, device):
    """Solve random spin-unit fixtures to convergence by every schedule: the unit and quad schedules reach the serial
    row loop's impulses and velocity to the rounding of their reductions, and every friction triple lies on or inside
    its cone ``|(t1 / mu, t2 / mu, s / mu_t)| <= lambda_n``."""
    rng = np.random.default_rng(2026)
    for trial in range(3):
        fixture = _fixture(rng)
        results = [_solve(fixture, schedule, device, iterations=400) for schedule in SCHEDULES]
        reference_impulses, reference_velocity = results[0]
        scale = max(np.abs(reference_impulses).max(), 1.0)
        for schedule, (impulses, velocity) in zip(SCHEDULES[1:], results[1:], strict=True):
            with test.subTest(trial=trial, schedule=sorted(schedule)):
                np.testing.assert_allclose(impulses, reference_impulses, rtol=0.0, atol=2.0e-4 * scale)
                np.testing.assert_allclose(velocity, reference_velocity, rtol=0.0, atol=2.0e-4 * scale)
        for impulses, _ in results:
            for contact in range(CONTACTS):
                head = 1 + 4 * contact
                load = impulses[:, head]
                mu, mu_t = fixture["mu"][:, head + 1], fixture["mu"][:, head + 3]
                scaled = np.stack(
                    [impulses[:, head + 1] / mu, impulses[:, head + 2] / mu, impulses[:, head + 3] / mu_t], axis=1
                )
                test.assertTrue(np.all(load >= 0.0))
                test.assertTrue(np.all(np.linalg.norm(scaled, axis=1) <= load * (1.0 + 1.0e-5) + 1.0e-7))
        # The spin rows carry torque: the cones bind somewhere, and the spin row moves with its pair.
        test.assertGreater(np.abs(results[0][0][:, 4::4]).max(), 0.0)


class TestSpinUnits(unittest.TestCase):
    pass


add_function_test(
    TestSpinUnits,
    "test_unit_schedules_match_the_serial_row_loop",
    test_unit_schedules_match_the_serial_row_loop,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
