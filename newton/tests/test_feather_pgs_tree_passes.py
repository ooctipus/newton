# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The general route's dynamics passes (``feather_pgs.tree_passes``) and the solver stages that run them.

* :class:`TestTreePassesPhysics`: each pass against Newton's own kinematics and dynamics, within float32
  rounding: poses and center-of-mass twists against :func:`newton.eval_fk`; joint forces against the
  manipulator equation at zero acceleration, ``joint_f + passive - (C qd + g) + J^T f``
  (:func:`newton.eval_inverse_dynamics_passive`, :func:`newton.eval_jacobian`); joint-space inertias against
  the same mass matrix, and exactly symmetric; composite inertia records against subtree sums.
* :class:`TestTreePassesGradients`: Warp's tape through each pass against central differences.
* :class:`TestGeneralRouteStages`: the solver's FK/ID and CRBA stages on the gradient route, through the
  solver's own stage methods: the mass matrices' gradient against central differences, and on CUDA
  deterministic, exactly symmetric mass matrices with multi-coordinate joints. The CRBA kernel the passes
  replaced failed both: its tape gradient was zero, and the column threads of one multi-coordinate joint
  raced on the joint's shared entries.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

EPS = float(np.finfo(np.float32).eps)

PHYSICS_SHAPES = ("anymal13", "humanoid13", "g1_30", "chain64", "random60", "pendulum2")
GRADIENT_SHAPES = ("anymal13", "humanoid13", "pendulum2")


def _gap(value, reference) -> float:
    """``value``'s largest difference from ``reference`` in float32 units of ``reference``'s largest entry."""
    reference = np.asarray(reference, dtype=np.float64)
    return float(np.abs(np.asarray(value, dtype=np.float64) - reference).max() / np.abs(reference).max() / EPS)


def _external_forces(model, state, jacobian) -> np.ndarray:
    """The generalized forces of ``state``'s body wrenches (at the centers of mass, world frame): ``J^T f``."""
    starts, qd_start = model.articulation_start.numpy(), model.joint_qd_start.numpy()
    child, wrench = model.joint_child.numpy(), state.body_f.numpy().astype(np.float64)
    forces = np.zeros(model.joint_dof_count)
    for art in range(model.articulation_count):
        joints = np.arange(starts[art], starts[art + 1])
        first, last = qd_start[joints[0]], qd_start[joints[-1] + 1]
        forces[first:last] = jacobian[art, : 6 * joints.size, : last - first].T @ wrench[child[joints]].reshape(-1)
    return forces


def check_physics(test, device):
    from newton._src.solvers.feather_pgs.kernels import update_body_qd_from_featherstone  # noqa: PLC0415
    from newton.tests.feather_pgs_tree_scan_fixture import (  # noqa: PLC0415
        articulated_model,
        general_route,
        joint_space_inertias,
        run_pass,
    )

    model = articulated_model(PHYSICS_SHAPES, 2, device)
    route = general_route(model)
    solver, state, aug = route.solver, route.state, route.aug
    reference = model.state()
    newton.eval_fk(model, reference.joint_q, reference.joint_qd, reference)

    # Poses, and the center-of-mass twists published from the twists about the origin.
    run_pass(route, "fk_id", write_inertia_records=1)
    test.assertLess(_gap(state.body_q.numpy(), reference.body_q.numpy()), 32.0, "poses")
    wp.launch(
        update_body_qd_from_featherstone,
        model.body_count,
        inputs=[aug.body_v_s, state.body_q, model.body_com, solver.body_to_articulation, solver.articulation_origin],
        outputs=[state.body_qd],
        device=device,
    )
    test.assertLess(_gap(state.body_qd.numpy(), reference.body_qd.numpy()), 32.0, "center-of-mass twists")

    # Joint forces: the manipulator equation at zero acceleration with the external wrenches, generalized
    # forces and passive springs and dampers of the route. The passes take a free root's coordinates as the
    # spatial twist at the origin and hold its spatial acceleration at zero, where Newton holds its center of
    # mass's acceleration at zero; the two agree when the root does not rotate, so the free roots' angular
    # velocities are zeroed here (the step transports the rotating root's twist separately).
    qd = reference.joint_qd.numpy().copy()
    qd_start = model.joint_qd_start.numpy()
    for joint in np.flatnonzero(model.joint_type.numpy() == int(newton.JointType.FREE)):
        qd[qd_start[joint] + 3 : qd_start[joint] + 6] = 0.0
    for target in (reference, state):
        target.joint_qd.assign(qd)
    newton.eval_fk(model, reference.joint_q, reference.joint_qd, reference)
    run_pass(route, "fk_id", write_inertia_records=1)
    width = model.max_dofs_per_articulation
    mass_matrix = wp.zeros((model.articulation_count, width, width), dtype=float, device=device)
    gravity_force = wp.zeros(model.joint_dof_count, dtype=float, device=device)
    coriolis_force = wp.zeros(model.joint_dof_count, dtype=float, device=device)
    newton.eval_inverse_dynamics_passive(
        model, reference, mass_matrix=mass_matrix, gravity_force=gravity_force, coriolis_force=coriolis_force
    )
    q_start = model.joint_q_start.numpy()
    dof_joint = np.repeat(np.arange(model.joint_count), np.diff(qd_start))
    q = reference.joint_q.numpy()[q_start[dof_joint]].astype(np.float64)
    passive = solver._passive_spring_stiffness.numpy() * (solver._passive_spring_ref.numpy() - q)
    passive -= solver._passive_joint_damping.numpy() * qd
    expected = (
        route.control.joint_f.numpy()
        + passive
        - coriolis_force.numpy()
        - gravity_force.numpy()
        + _external_forces(model, state, newton.eval_jacobian(model, reference).numpy())
    )
    run_pass(route, "tau")
    test.assertLess(_gap(aug.joint_tau.numpy(), expected), 64.0, "joint forces")

    # Composite inertia: each body's record plus its subtree's, children after parents in joint order.
    run_pass(route, "composite_inertia")
    composite = solver._body_inertia_terms.numpy().astype(np.float64)
    parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
    for joint in range(model.joint_count - 1, -1, -1):
        if parent[joint] >= 0:
            composite[parent[joint]] += composite[child[joint]]
    test.assertLess(_gap(solver.body_I_c.numpy(), composite), 16.0, "composite inertia")

    # Joint-space inertias: Newton's mass matrix; each entry written once, so exactly symmetric.
    run_pass(route, "crba")
    H = joint_space_inertias(route)
    test.assertLess(_gap(H, mass_matrix.numpy()), 64.0, "joint-space inertia")
    np.testing.assert_array_equal(H, np.swapaxes(H, 1, 2), "joint-space inertia symmetry")


# Per pass and input, the central-difference step: small on the nonlinear inputs, larger on the linear ones.
GRADIENT_CASES = (
    ("fk_id", "joint_q", 1e-3),
    ("fk_id", "joint_qd", 1e-2),
    ("fk_id", "body_mass", 1e-2),
    ("fk_id", "body_inertia", 1e-2),
    ("tau", "joint_q", 1e-2),
    ("tau", "joint_qd", 1e-2),
    ("tau", "joint_f", 1e-1),
    ("tau", "joint_S_s", 1e-2),
    ("tau", "body_f_s", 1e-1),
    ("tau", "body_f", 1e-1),
    ("tau", "body_q", 1e-3),
    ("composite_inertia", "_body_inertia_terms", 1e-1),
    ("crba", "joint_S_s", 1e-2),
    ("crba", "body_I_c", 1e-1),
)


def check_gradients(test, device):
    from newton.tests.feather_pgs_tree_scan_fixture import (  # noqa: PLC0415
        GRADIENT_INPUTS,
        articulated_model,
        pass_difference,
        pass_gradients,
    )

    model = articulated_model(GRADIENT_SHAPES, 1, device, requires_grad=True)
    test.assertEqual(
        {(name, key) for name, key, _ in GRADIENT_CASES}, {(n, k) for n in GRADIENT_INPUTS for k in GRADIENT_INPUTS[n]}
    )
    rng = np.random.default_rng(7)
    tape = {name: pass_gradients(model, name) for name in GRADIENT_INPUTS}
    for name, key, step in GRADIENT_CASES:
        gradient = tape[name][key].reshape(-1)
        picks = rng.choice(gradient.size, min(gradient.size, 24), replace=False)
        difference = pass_difference(model, name, key, step, picks).reshape(-1)[picks]
        test.assertGreater(np.abs(difference).max(), 0.0, f"{name} {key}")
        error = np.abs(gradient[picks] - difference).max() / np.abs(difference).max()
        test.assertLess(error, 1e-3, f"{name} {key}: tape against central differences")


# ---------------------------------------------------------------------------------------------------------------------
# The solver's stages on the gradient route
# ---------------------------------------------------------------------------------------------------------------------


class _Stages:
    """A solver on a model that requires gradients, its FK/ID and CRBA stages run as a step runs them."""

    def __init__(self, model):
        from newton.solvers import SolverFeatherPGS  # noqa: PLC0415

        self.solver = SolverFeatherPGS(model)
        self.state, self.out, control = model.state(), model.state(), model.control()
        self.aug = self.solver._prepare_augmented_state(self.state, self.out, control)

    def mass_matrices(self) -> list:
        self.solver._stage1_fk_id(self.state, self.aug, self.out)
        self.solver._stage1_crba(self.aug, None)
        return [self.solver.H_by_size[size] for size in self.solver.size_groups]


def check_stage_mass_gradient(test, device):
    """The mass matrices' gradient with respect to every joint coordinate, by the tape through the FK/ID and
    CRBA stages, against central differences."""
    from newton.tests.feather_pgs_tree_scan_fixture import articulated_model  # noqa: PLC0415

    stages = _Stages(articulated_model(GRADIENT_SHAPES, 1, device, requires_grad=True))
    rng = np.random.default_rng(11)
    weights = [rng.standard_normal(H.shape) for H in stages.mass_matrices()]
    tape = wp.Tape()
    with tape:
        matrices = stages.mass_matrices()
    tape.backward(
        grads={H: wp.array(w.astype(np.float32), device=device) for H, w in zip(matrices, weights, strict=True)}
    )
    gradient = stages.state.joint_q.grad.numpy().astype(np.float64)
    tape.zero()

    base, step = stages.state.joint_q.numpy().copy(), 1e-3

    def masses(q):
        stages.state.joint_q.assign(q)
        return [H.numpy().astype(np.float64) for H in stages.mass_matrices()]

    difference = np.zeros(base.size)
    for k in range(base.size):
        plus, minus = base.copy(), base.copy()
        plus[k] += step
        minus[k] -= step
        pairs = zip(masses(plus), masses(minus), weights, strict=True)
        difference[k] = sum(float(np.sum((a - b) * w)) for a, b, w in pairs) / (2.0 * step)
    stages.state.joint_q.assign(base)
    test.assertGreater(np.abs(difference).max(), 0.0)
    error = np.abs(gradient - difference).max() / np.abs(difference).max()
    test.assertLess(error, 1e-3, "mass matrices: tape against central differences")


def check_stage_mass_symmetry(test, device):
    """Ball joints and free roots across many articulations: every launch gives the same, exactly symmetric
    matrices. (The replaced kernel's race showed in every few launches at this size: entries changed from
    launch to launch, and some lost their symmetry.)"""
    from newton.tests.feather_pgs_tree_scan_fixture import articulated_model  # noqa: PLC0415

    stages = _Stages(articulated_model(("humanoid13", "anymal13"), 1024, device, requires_grad=True))
    first = [H.numpy().copy() for H in stages.mass_matrices()]
    for H in first:
        np.testing.assert_array_equal(H, np.swapaxes(H, 1, 2), "symmetry")
    for _ in range(8):
        for H, again in zip(first, stages.mass_matrices(), strict=True):
            np.testing.assert_array_equal(again.numpy(), H, "determinism")


def check_up_scans_read_no_stale_slot(test, device):
    """The joint forces and the composite inertia set every slot they read: from scratch holding NaN they give the
    words they give from zeroed scratch, each body's children's sum (zero at a leaf) and each subtree's inertia
    included. (The up scans store a first child's sum and add the others; nothing clears their slots first.)"""
    from newton.tests.feather_pgs_tree_scan_fixture import (  # noqa: PLC0415
        articulated_model,
        general_route,
        run_inputs,
        run_pass,
    )

    route = general_route(articulated_model(("humanoid13", "anymal13", "pendulum2"), 3, device))
    for name, scratch in (("tau", "body_ft_s"), ("composite_inertia", "body_I_c")):
        run_inputs(route, name)
        results = []
        for fill in (0.0, np.nan):
            array = getattr(route.aug if name == "tau" else route.solver, scratch)
            array.assign(np.full(array.numpy().shape, fill, dtype=np.float32))
            route.aug.joint_tau.zero_()
            run_pass(route, name, write_inertia_records=1)
            results.append((array.numpy().copy(), route.aug.joint_tau.numpy().copy()))
        for clean, stale in zip(*results, strict=True):
            np.testing.assert_array_equal(stale, clean, f"{name}: the results depend on what the scratch held")


class TestTreePassesPhysics(unittest.TestCase):
    pass


class TestTreePassesGradients(unittest.TestCase):
    pass


class TestGeneralRouteStages(unittest.TestCase):
    pass


add_function_test(TestTreePassesPhysics, "test_physics", check_physics, devices=get_test_devices())
add_function_test(
    TestTreePassesPhysics,
    "test_up_scans_read_no_stale_slot",
    check_up_scans_read_no_stale_slot,
    devices=get_test_devices(),
)
add_function_test(TestTreePassesGradients, "test_gradients", check_gradients, devices=get_test_devices())
add_function_test(TestGeneralRouteStages, "test_mass_gradient", check_stage_mass_gradient, devices=get_test_devices())
add_function_test(
    TestGeneralRouteStages, "test_mass_symmetry", check_stage_mass_symmetry, devices=get_cuda_test_devices()
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
