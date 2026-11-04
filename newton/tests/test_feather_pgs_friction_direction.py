# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check isotropic friction against the direction and magnitude of Coulomb sliding."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import sweep_routes_tiers
from newton._src.solvers.feather_pgs.row_kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION
from newton._src.solvers.feather_pgs.solve_policy import QUIET
from newton.tests.feather_pgs_free_body_rows import sweep_free_body_rows
from newton.tests.test_feather_pgs_parallel_patch import _launch_parallel

_MU = 0.3
"""The friction coefficient of the unequal-tangent-mass cases."""
_TYPES = (PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION)
_PARENTS = (-1, 0, 0)


def unequal_tangent_cases():
    """Contacts whose friction pairs have rotated, unequal tangent masses, in five regimes, with their solutions.

    One normal row and its friction pair act on a six-coordinate body of unit mass: rows ``Z`` whose block is
    ``Z Z^T = diag(1, T)``, ``T = R diag(small, 7) R^T`` (the normal along coordinate 2, the tangents the symmetric
    square root of ``T`` in coordinates 0 and 1). The normal's right-hand side -1 loads it with 1, so the pair's
    disk has radius ``mu = 0.3``. The regimes: ``sticking`` (the pair holds its velocity's impulse inside the disk),
    ``sliding`` (on the disk opposite the direction), ``fixed`` and ``sliding_fixed`` (started at their solution),
    ``weak_sliding`` (a tiny positive multiplier still on the boundary). Each case gives the rows, the right-hand
    sides, the starting impulses, the expected ones and the check that applies.
    """
    cases = []
    for angle, small in ((0.0, 1.0), (0.37, 1.0), (1.1, 1.0), (0.37, 0.0), (0.0, 0.0), (0.0, 1.0e-4), (0.0, 1.0e-6)):
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        tangent_mass = rotation @ np.diag([small, 7.0]) @ rotation.T
        values, vectors = np.linalg.eigh(tangent_mass)
        root = vectors @ np.diag(np.sqrt(np.maximum(values, 0.0))) @ vectors.T
        Z = np.zeros((3, 6))
        Z[0, 2] = 1.0
        Z[1:3, 0:2] = root
        tangent_mass = Z[1:3] @ Z[1:3].T
        for regime in ("sticking", "sliding", "fixed", "weak_sliding", "sliding_fixed"):
            if regime == "weak_sliding" and small == 0.0:
                continue
            sliding = regime in ("sliding", "weak_sliding", "sliding_fixed")
            direction = np.array([0.6, 0.8])
            expected = -(_MU if sliding else 0.1) * direction
            if regime == "fixed":
                expected = np.array([0.125, 0.0])
            velocity = 2.0 * direction if sliding else np.zeros(2)
            if regime == "weak_sliding":
                # A tiny positive KKT multiplier still requires a boundary impulse.
                velocity = small * direction
            rhs = np.array([-1.0, *(velocity - tangent_mass @ expected)])
            if regime == "fixed":
                rhs[1:3] = -(tangent_mass @ expected)
            initial = np.zeros(3)
            if regime in ("fixed", "sliding_fixed"):
                initial = np.array([1.0, *expected])
            # A singular sticking block has a line of solutions: rounding in its residual moves the pair along the
            # line, which changes no velocity, so only the residual and the disk hold.
            singular = small == 0.0 and not sliding
            check = "singular" if singular else "fixed" if regime in ("fixed", "sliding_fixed") else "solution"
            name = f"angle={angle:g} small={small:g} {regime}"
            cases.append(
                {
                    "name": name,
                    "Z": Z,
                    "tangent_mass": tangent_mass,
                    "rhs": rhs,
                    "initial": initial,
                    "expected": np.array([1.0, *expected]),
                    "check": check,
                }
            )
    return cases


def check_unequal_tangent_solution(test, case, actual, *, atol):
    """Hold ``actual`` (normal, pair) to ``case``'s solution: a singular sticking block to one of its solutions, a
    fixed point re-solves to itself within the exit law's rounding floor, every other case to its solution."""
    if case["check"] == "fixed":
        np.testing.assert_allclose(actual, case["initial"], rtol=QUIET, atol=atol)
    elif case["check"] == "singular":
        # A singular sticking block has multiple impulse solutions.
        np.testing.assert_allclose(case["tangent_mass"] @ actual[1:] + case["rhs"][1:3], 0.0, atol=atol)
        test.assertLessEqual(float(np.linalg.norm(actual[1:])), _MU + 1.0e-6)
    else:
        np.testing.assert_allclose(actual, case["expected"], atol=atol)


TIERS_OPEN = frozenset(
    (
        "angle=0 small=0.0001 sticking",
        "angle=0 small=1e-06 sticking",
        "angle=0 small=1e-06 weak_sliding",
    )
)
"""The unequal tangent cases the Jacobi tiers leave short of their solution: ill-conditioned blocks, ``small=1e-06``
(conditioned 7e6), sticking and weak sliding, and ``small=0.0001`` sticking (7e4; the float64 reference reaches it
within 2.5e-7, the float32 tiers stop 1.3e-4 short on CUDA). The tiers' fixed point is the solution, but a pair steps
isotropically, by the smaller of its steps, so the block's soft direction closes by about the inverse of its
condition number per sweep: after 2000 sweeps the three stop 1.3e-4, 5.6e-2 and 0.13 short, after 20000 1.3e-4,
1.3e-2 and 0. A pair law that solved each block exactly reaches them in one sweep, at a cost per sweep the tiers do
not pay (37% of allegro's step).
"""


def tiers_case(case):
    """``case`` as one world of the Jacobi tiers' fixture (``test_feather_pgs_parallel_patch``): the body's unit mass
    makes the rows' Jacobian, response rows and whitened rows all ``Z``; the normal links itself, a ring of one; the
    velocity is the one the starting impulses publish."""
    Z = case["Z"]
    return {
        "J": Z,
        "Y": Z,
        "Z": Z,
        "L": np.eye(6),
        "cfm": 0.0,
        "diag": np.einsum("ij,ij->i", Z, Z),
        "kinds": np.array((0, 1, 1), np.int32),
        "types": np.array(_TYPES, np.int32),
        "parents": np.zeros(3, np.int32),
        "mu": np.array((0.0, _MU, _MU)),
        "rhs": case["rhs"],
        "impulses": case["initial"],
        "velocity": Z.T @ case["initial"],
    }


def _slide(device, beta, mesh, angle, articulated=False, *, locked_tangent=False):
    """Slide a symmetric, explicitly massed box under a force above the friction limit.

    ``articulated`` builds the box as a D6-jointed articulation instead of a free body.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.3, gap=0.0, restitution=0.0)
    builder.add_shape_box(-1, hx=2.0, hy=2.0, hz=0.05, xform=wp.transform((0, 0, -0.05)), cfg=cfg)
    inertia = np.diag([(0.2**2 + 0.05**2) / 12] * 2 + [2 * 0.2**2 / 12]).astype(np.float32)
    add_body = builder.add_link if articulated else builder.add_body
    body = add_body(xform=wp.transform((0, 0, 0.025)), mass=1.0, inertia=wp.mat33(inertia), lock_inertia=True)
    if articulated:
        joint = builder.add_joint_d6(
            parent=-1,
            child=body,
            parent_xform=wp.transform((0, 0, 0.025)),
            linear_axes=[
                newton.ModelBuilder.JointDofConfig(axis=axis)
                for axis in ((newton.Axis.X, newton.Axis.Z) if locked_tangent else newton.Axis)
            ],
        )
        builder.add_articulation([joint])
    if mesh:
        builder.add_shape_mesh(body, mesh=newton.Mesh.create_box(0.1, 0.1, 0.025), cfg=cfg)
    else:
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.025, cfg=cfg)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverFeatherPGS(
        model,
        friction_anchor_beta=beta,
        pgs_iterations=64,
        pgs_cfm=0.0 if locked_tangent else 1.0e-6,
        angular_damping=0.0,
        mf_max_constraints=128,
        dense_max_constraints=32,
        row_watermark=True,
    )
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=128, deterministic=True)
    contacts = pipeline.contacts()
    state, next_state = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    direction = np.array([np.cos(angle), np.sin(angle)])
    force = np.zeros((1, 6), dtype=np.float32)
    force[0, :2] = direction * 1.2 * 0.3 * 9.81
    hz = 240
    for step in range(hz // 5 + hz // 2):
        state.clear_forces()
        if step >= hz // 5:
            state.body_f.assign(force)
        pipeline.collide(state, contacts)
        solver.step(state, next_state, control, contacts, 1.0 / hz)
        state, next_state = next_state, state
    return state.body_qd.numpy()[0, :2], direction, solver.constraint_row_watermarks()


class TestFeatherPGSFrictionDirection(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_articulated_sliding_direction(self):
        """Preserve sliding direction through the dense articulated route."""
        velocity, direction, _ = _slide("cuda:0", 0.2, False, np.pi / 4, articulated=True)
        np.testing.assert_allclose(velocity, direction * (0.2 * 0.3 * 9.81) * 0.5, atol=0.015, rtol=0.03)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_locked_tangent_preserves_other_friction_direction(self):
        """Retain friction along a slider when its first tangent has zero response."""
        for beta in (0.0, 0.2):
            with self.subTest(beta=beta):
                velocity, direction, watermarks = _slide(
                    "cuda:0", beta, False, 0.0, articulated=True, locked_tangent=True
                )
                np.testing.assert_allclose(velocity, direction * (0.2 * 0.3 * 9.81) * 0.5, atol=0.015, rtol=0.03)
                for key, value in watermarks.items():
                    if "overflow" in key or "dropped" in key:
                        self.assertEqual(value, 0, key)

    def test_unequal_tangent_masses(self):
        """Solve sticking and sliding pairs with rotated, unequal tangent masses in one sweep of the general sweep."""
        devices = ("cpu", "cuda:0") if wp.is_cuda_available() else ("cpu",)
        for device in devices:
            for case in unequal_tangent_cases():
                with self.subTest(device=device, case=case["name"]):
                    actual = sweep_free_body_rows(
                        device,
                        case["Z"],
                        case["rhs"],
                        _TYPES,
                        _PARENTS,
                        (0.0, _MU, _MU),
                        impulses=case["initial"],
                        iterations=1,
                    )
                    check_unequal_tangent_solution(self, case, actual, atol=1.0e-6)

    def _solve_on_the_tiers(self, cases):
        """Hold ``cases`` through the Jacobi tiers: rows formed in the kernel, linked, 2000 sweeps, tolerance 0."""
        device = wp.get_device("cuda:0")
        kernel = sweep_routes_tiers.parallel_kernel(
            64, 32, 6, str(device.arch), rows=32, sweeps=2000, inkernel_response=(6, 0, 0, 0), linked_rows=True
        )
        for case in cases:
            with self.subTest(case=case["name"]):
                impulses, _ = _launch_parallel(kernel, tiers_case(case), device=device)
                check_unequal_tangent_solution(self, case, impulses.astype(np.float64), atol=1.0e-5)

    @unittest.skipUnless(wp.is_cuda_available(), "The Jacobi tiers need CUDA")
    def test_unequal_tangent_masses_on_the_tiers(self):
        """Solve the same pairs through the Jacobi tiers, all but :data:`TIERS_OPEN`."""
        self._solve_on_the_tiers([case for case in unequal_tangent_cases() if case["name"] not in TIERS_OPEN])

    @unittest.skipUnless(wp.is_cuda_available(), "The Jacobi tiers need CUDA")
    @unittest.expectedFailure
    def test_unequal_tangent_masses_on_the_tiers_open(self):
        """Solve :data:`TIERS_OPEN` through the Jacobi tiers.

        Known failure: the ill-conditioned blocks' soft direction, which the isotropic step closes past the budget.
        """
        self._solve_on_the_tiers([case for case in unequal_tangent_cases() if case["name"] in TIERS_OPEN])

    def test_sliding_direction(self):
        """Oppose sliding in every direction for analytic and mesh boxes, with and without patches."""
        devices = ("cpu", "cuda:0") if wp.is_cuda_available() else ("cpu",)
        for device in devices:
            for beta in (0.0, 0.2):
                for mesh in (False, True):
                    for angle in (0.0, np.pi / 6, np.pi / 4, 2 * np.pi / 3):
                        with self.subTest(device=device, beta=beta, mesh=mesh, angle=angle):
                            velocity, direction, watermarks = _slide(device, beta, mesh, angle)
                            expected = direction * (0.2 * 0.3 * 9.81) * 0.5
                            np.testing.assert_allclose(velocity, expected, atol=0.015, rtol=0.03)
                            for key, value in watermarks.items():
                                if "overflow" in key or "dropped" in key:
                                    self.assertEqual(value, 0, key)


if __name__ == "__main__":
    unittest.main()
