# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The row-activation gates: which rows a step builds, read from the step's data.

A joint-limit row exists where its coordinate is within joint_limit_activation_gap of a finite bound; a row that may act
at the predictor marks its world's acts word. The rows the general route's warp builds are checked against the row law on both devices.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.core.types import MAXVAL
from newton._src.sim.enums import JointType
from newton._src.solvers.feather_pgs.row_laws import ROW_JOINT_LIMIT
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs_law_harness import PREFIX_RECORDS, prefix_warp_rows

# ---------------------------------------------------------------------------------------------------------------------
# joint_limit_activation_gap
# ---------------------------------------------------------------------------------------------------------------------


def _built_rows(q: float, *, gap: float, lower: float = -1.0, upper: float = 1.0):
    """One revolute coordinate's joint-limit rows (the general route's warp, on the CPU): their J and phi."""
    rows = prefix_warp_rows(
        "cpu",
        1,
        1,
        8,
        lower=[lower],
        upper=[upper],
        q=[q],
        activation_gap=gap,
        beta=0.2,
        cfm=0.0,
    )
    count = int(rows["count"][0])
    return rows["J"][0, :count, 0].tolist(), rows["phi"][0, :count].tolist()


def _limit_law(q: np.ndarray, lower: np.ndarray, upper: np.ndarray, gap: float) -> list:
    """The joint-limit rows' phi the row law gives (:func:`.row_laws.joint_limit_side`), in row order: each coordinate's
    lower side (``q - lower``), then its upper side (``upper - q``), where its bound is finite and ``q`` within ``gap``."""
    phi = []
    for value, low, high in zip(q.astype(np.float32), lower.astype(np.float32), upper.astype(np.float32), strict=True):
        if np.isfinite(low) and value <= low + np.float32(gap):
            phi.append(value - low)
        if np.isfinite(high) and value >= high - np.float32(gap):
            phi.append(high - value)
    return phi


def _limit_scene(device, *, pgs_iterations=0):
    """A limited revolute link past its upper limit and moving further into it, clear of the ground, and a free box on
    the ground: joint-limit and contact rows. Nothing caps the link's speed, so its box stays off the ground, where a
    contact's push would carry the joint away from its limit."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.mu = 0.5
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    link = builder.add_link()
    builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(-1.0, 0.0, 0.3), wp.quat_identity()),
        limit_lower=-0.1,
        limit_upper=0.1,
    )
    builder.add_articulation([joint])
    builder.joint_q[0] = 0.2

    free_body = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.05), wp.quat_identity()))
    builder.add_shape_box(free_body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()

    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(
        model,
        enable_joint_limits=True,
        dense_max_constraints=64,
        mf_max_constraints=64,
        pgs_iterations=pgs_iterations,
        pgs_warmstart=False,
    )
    state_0, state_1 = model.state(), model.state()
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[0] = 0.5
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    return model, solver, state_0, state_1, model.control(), model.contacts()


def _run_joint_limit_trajectory():
    """Run a short trajectory; each step's joint-limit rows against the row law at the step's input coordinates."""
    model, solver, state_in, state_out, control, contacts = _limit_scene("cuda:0", pgs_iterations=8)
    revolute = np.flatnonzero(model.joint_type.numpy() == int(JointType.REVOLUTE))
    coordinates = model.joint_q_start.numpy()[revolute]
    dofs = model.joint_qd_start.numpy()[revolute]
    samples = []
    for _ in range(4):
        q = state_in.joint_q.numpy()[coordinates]
        state_in.clear_forces()
        model.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        dense_count = int(solver.placement.constraint_count.numpy()[0])
        row_type = solver.placement.row_type.numpy()[0, :dense_count]
        limit_rows = np.flatnonzero(row_type == ROW_JOINT_LIMIT)
        samples.append(
            {
                "phi": solver.placement.phi.numpy()[0, limit_rows].tolist(),
                "law": _limit_law(
                    q,
                    model.joint_limit_lower.numpy()[dofs],
                    model.joint_limit_upper.numpy()[dofs],
                    solver.options.joint_limit_activation_gap,
                ),
                "impulses": solver.placement.impulses.numpy()[0, limit_rows].copy(),
            }
        )
        state_in, state_out = state_out, state_in
    return samples


class TestFeatherPGSJointLimitActivationGap(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "the matrix-free row families require CUDA")
    def test_limit_rows_follow_the_law_over_a_trajectory(self):
        """Give every step the joint-limit rows the row law gives at its input coordinates, with bitwise phi."""
        samples = _run_joint_limit_trajectory()
        for step, sample in enumerate(samples):
            self.assertEqual(sample["phi"], sample["law"], f"limit rows differed at step {step}")
        self.assertGreater(len(samples[0]["phi"]), 0)
        self.assertGreater(float(np.max(np.abs(samples[0]["impulses"]))), 0.0)

    def test_solver_rejects_invalid_joint_limit_activation_gap(self):
        model = newton.ModelBuilder().finalize()

        for value in (-0.1, float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "joint_limit_activation_gap"):
                    SolverFeatherPGS(model, joint_limit_activation_gap=value)

    def test_finite_gap_builds_only_near_limit_rows(self):
        self.assertEqual(_built_rows(0.0, gap=0.2), ([], []))

        jacobian, phi = _built_rows(-0.85, gap=0.2)
        self.assertEqual(jacobian, [1.0])
        self.assertAlmostEqual(phi[0], 0.15, places=6)

        jacobian, phi = _built_rows(0.85, gap=0.2)
        self.assertEqual(jacobian, [-1.0])
        self.assertAlmostEqual(phi[0], 0.15, places=6)

    def test_finite_gap_does_not_activate_unlimited_sentinel_limits(self):
        self.assertEqual(_built_rows(0.0, gap=0.2, lower=-MAXVAL, upper=MAXVAL), ([], []))

    def test_infinite_gap_preserves_historical_always_allocate_behavior(self):
        jacobian, phi = _built_rows(0.0, gap=float("inf"))

        self.assertEqual(jacobian, [1.0, -1.0])
        self.assertEqual(phi, [1.0, 1.0])


# ---------------------------------------------------------------------------------------------------------------------
# joint_limit_warp
#
# The general route's joint-limit warp (:func:`.rows.joint_limit_warp_kernel`): one text on both devices, a warp per
# articulation on CUDA and a thread per articulation on the CPU (:func:`.rows.prefix_shape`).
# ---------------------------------------------------------------------------------------------------------------------


class TestFeatherPGSJointLimitWarp(unittest.TestCase):
    def test_acts_word_marks_worlds_whose_limits_may_act(self):
        """A world's acts word is set where an active side's residual at the predictor ``w v_hat + rhs`` is not above
        the separation margin (:func:`.row_laws.separates`), and the rows stay those of no word, on every device."""
        arts, size, dt = 257, 6, 0.01
        rng = np.random.default_rng(5)
        q = rng.uniform(-1.2, 1.2, arts * size)
        vhat = rng.uniform(-40.0, 40.0, arts * size) * (rng.random(arts * size) < 0.2)
        bounds = {"lower": np.full(arts * size, -1.0), "upper": np.full(arts * size, 1.0)}
        law = {**bounds, "q": q, "activation_gap": 0.25, "beta": 0.2, "cfm": 1.0e-6, "dt": dt}
        active, phi, sign = np.stack([q <= -0.75, q >= 0.75]), np.stack([q + 1.0, 1.0 - q]), np.array([[1.0], [-1.0]])
        rhs = np.where(phi < 0.0, 0.2 * phi, phi) / dt
        margin = 1.0e-5 + 2.0**-20 * (np.abs(vhat) + np.abs(rhs))
        acting = np.any(active & (sign * vhat + rhs < margin), axis=0).reshape(arts, size).any(axis=1)
        self.assertTrue(acting.any() and not acting.all())
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device):
                plain = prefix_warp_rows(device, arts, size, 2 * size, **law)
                rows = prefix_warp_rows(device, arts, size, 2 * size, vhat=vhat, acts=np.zeros(arts), **law)
                np.testing.assert_array_equal(rows["acts"], acting)
                for name in ("count", "J", *PREFIX_RECORDS):
                    np.testing.assert_array_equal(rows[name], plain[name], err_msg=name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
