# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The conformance reference held to an independent float64 evaluation, on every catalogue model.

The reference route is the held tree on the general placement (:data:`.test_feather_pgs_conformance.REFERENCE_CUDA`).
What keeps it honest is outside the solver: after a few steps of each model,

- the held geometry equals the float64 mass matrix ``H`` at the step's output (:mod:`.feather_pgs_mass_reference`);
- the walk (the factor) holds ``|L L^T - (H + R + K)|`` at the step's input and the response ``|A y - x|``,
  ``y = A^-1 x``;
- the swept rows hold ``|A Y_i - J_i|``, ``Y_i = L^-T Z_i``: their whitening and the walk agree.

Each bound is a multiple of the coordinates times the float32 rounding of the matrix's largest entry.
"""

import dataclasses
import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests import test_feather_pgs_conformance as conformance
from newton.tests.feather_pgs_law_harness import unpacked_rows, walk_inverse
from newton.tests.feather_pgs_mass_reference import MassReference

STEPS = 8
ROUNDING = 2.0**-24
BOUND = 16.0
"""The bounds' multiple of ``n u |A|``: every catalogue model measures under 6 on the CPU."""


def _problems() -> list:
    """Every model of the CUDA catalogue once, with its options."""
    seen, problems = set(), []
    for group in conformance.cuda_groups():
        if group.physics.scene.name not in seen:
            seen.add(group.physics.scene.name)
            problems.append(group.physics)
    return problems


@dataclasses.dataclass
class Evaluated:
    """A solver after its last step, with the coordinates around that step."""

    solver: SolverFeatherPGS
    joint_q_in: np.ndarray
    joint_q_out: np.ndarray


def _run(physics, reference, device) -> Evaluated:
    scene = physics.scene
    model = scene.build(device)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    options = {**conformance.BASE_OPTIONS, **scene.solver_options, **physics.options(physics.iterations)}
    # The rows' material law is the sweep's: these contacts hold no per-contact material.
    options.pop("contact_compliance", None)
    with conformance.route_context(reference):
        solver = SolverFeatherPGS(model, **options)
        state, out, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        for step in range(STEPS):
            state.clear_forces()
            if scene.control is not None:
                scene.control(model, control, state, step)
            pipeline.collide(state, contacts)
            solver.step(state, out, control, contacts, scene.dt)
            # The coordinates as the forest holds them: the reference evaluates the solver's own model.
            joint_q_in = solver.cut.mirror(state).joint_q.numpy().copy()
            state, out = out, state
    return Evaluated(solver, joint_q_in, solver.cut.mirror(state).joint_q.numpy().copy())


def _held_groups(solver):
    """Each held tree group of dense or compact coordinates: its operator, its articulations per group, its host."""
    for group in solver.factor.held.groups:
        operator = group.sparse
        if getattr(operator, "prescribed", False) or not hasattr(operator, "host"):
            continue
        arts = getattr(operator, "forest_arts", None)
        arts = operator.plan.group_to_art.numpy()[:, None] if arts is None else arts
        yield group, operator, arts


def _physical(geometric, host) -> np.ndarray:
    """A group's packed geometry as its symmetric physical matrix."""
    n = host["physical"].size
    H = np.zeros((n, n))
    rows, cols = host["physical"][host["row"]], host["physical"][host["col"]]
    H[rows, cols] = H[cols, rows] = geometric
    return H


def _block(reference, joint_q, arts) -> np.ndarray:
    """The mass matrix of the trees ``arts`` side by side."""
    blocks = [reference.mass_matrix(joint_q, int(art)) for art in arts]
    n = sum(block.shape[0] for block in blocks)
    H, at = np.zeros((n, n)), 0
    for block in blocks:
        H[at : at + block.shape[0], at : at + block.shape[0]] = block
        at += block.shape[0]
    return H


def _bound(A, n=None) -> float:
    return BOUND * (A.shape[0] if n is None else n) * ROUNDING * np.abs(A).max()


def _diagonal(solver, art) -> np.ndarray:
    """The armature and drive stiffness of articulation ``art``'s coordinates."""
    forest = solver.forest
    start, size = int(forest.articulation_dof_start_host[art]), int(forest.articulation_dof_count_host[art])
    added = solver.factor.dof_armature.numpy()[start : start + size].astype(np.float64)
    rows = solver.factor.drive_row_by_dof.numpy()[start : start + size]
    # The factor holds every drive's K: a saturated drive's is released by the predictor's border, not the factor.
    K = solver.factor.aug_row_K.numpy()
    added += np.where(rows >= 0, np.maximum(K[np.maximum(rows, 0)], 0.0), 0.0)
    return added


def check_geometry(test, evaluated, reference) -> None:
    """The held geometry against the float64 mass matrix at the last step's output."""
    for group, operator, arts in _held_groups(evaluated.solver):
        geometric = group.geometric.numpy().astype(np.float64)
        for index in (0, len(arts) - 1):
            H = _block(reference, evaluated.joint_q_out, arts[index])
            error = np.abs(_physical(geometric[index], operator.host) - H).max()
            test.assertLessEqual(error, _bound(H), f"H of group {index}, {type(operator).__name__}")


def check_factor(test, evaluated, reference) -> None:
    """The walk (the factor) against the float64 matrix at the last step's input, and its solve."""
    solver = evaluated.solver
    rng = np.random.default_rng(11)
    for _, operator, arts in _held_groups(solver):
        if getattr(operator, "diagonal", False) or arts.shape[1] != 1:
            continue
        for index in (0, len(arts) - 1):
            art = int(arts[index, 0])
            A = reference.mass_matrix(evaluated.joint_q_in, art) + np.diag(_diagonal(solver, art))
            inverse = walk_inverse(solver, art)
            L = np.linalg.inv(inverse)
            test.assertLessEqual(np.abs(L @ L.T - A).max(), _bound(A), "L L^T")
            x = rng.normal(size=A.shape[0])
            y = inverse.T @ (inverse @ x)
            test.assertLessEqual(
                np.abs(A @ y - x).max(), _bound(A) * np.abs(y).max() + BOUND * ROUNDING * np.abs(x).max(), "A y - x"
            )


def check_rows(test, evaluated, reference) -> None:
    """The swept rows' whitening against the float64 matrix at the last step's input."""
    solver = evaluated.solver
    if solver.placement.packed_Z.shape[2] <= 1 or not int(solver.placement.constraint_count.numpy().sum()):
        return
    forest = solver.forest
    sizes = forest.articulation_response_dof_count.numpy()
    offsets = forest.articulation_world_dof_offset.numpy()
    worlds = forest.art_to_world.numpy()
    J, Y = unpacked_rows(solver)
    for world, count in enumerate(solver.placement.constraint_count.numpy()):
        if not count:
            continue
        n = int(J.shape[2])
        A = np.zeros((n, n))
        for art in np.flatnonzero((worlds == world) & (sizes > 0)):
            at, size = int(offsets[art]), int(sizes[art])
            H = reference.mass_matrix(evaluated.joint_q_in, int(art))
            A[at : at + size, at : at + size] = H + np.diag(_diagonal(solver, int(art)))
        scale = _bound(A) * np.abs(Y[world, : int(count)]).max(axis=1)
        residual = np.abs(Y[world, : int(count)] @ A - J[world, : int(count)]).max(axis=1)
        test.assertTrue(np.all(residual <= scale + _bound(np.abs(J[world, : int(count)]), n)), "A Y - J")


class _Invariants(unittest.TestCase):
    device = None
    reference_route = None

    def check(self, physics):
        evaluated = _run(physics, self.reference_route, self.device)
        reference = MassReference(evaluated.solver.model)
        check_geometry(self, evaluated, reference)
        check_factor(self, evaluated, reference)
        check_rows(self, evaluated, reference)


def _add_tests(cls) -> None:
    for physics in _problems():

        def test(self, physics=physics):
            self.check(physics)

        test.__doc__ = f"Hold {physics.scene.name}'s held tree to the float64 mass matrix."
        setattr(cls, f"test_{physics.scene.name}", test)


class TestTreeInvariantsCPU(_Invariants):
    device = "cpu"
    reference_route = conformance.REFERENCE_CPU


_add_tests(TestTreeInvariantsCPU)


@unittest.skipUnless(wp.is_cuda_available(), "CUDA")
class TestTreeInvariantsCUDA(_Invariants):
    device = "cuda:0"
    reference_route = conformance.REFERENCE_CUDA


_add_tests(TestTreeInvariantsCUDA)


if __name__ == "__main__":
    unittest.main()
