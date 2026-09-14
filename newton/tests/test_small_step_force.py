# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the private force-to-predictor fragment before integrating its owner."""

import os
import unittest
from functools import cache
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import small_step_force, sparse_factor
from tools.fpgs_bench.test_sparse_factor import fixture, unpack


def force_fixture(device="cpu"):
    """Reuse the original physical operator fixture without constructing dispatch."""
    with patch.dict(os.environ, {"FEATHER_PGS_SPARSE_SMALL_STEP": "0"}):
        result = fixture(device)
    solver, model, state = (result[name] for name in ("solver", "model", "state"))
    plan = small_step_force.build_force_plan(model)
    control = model.control()
    rng = np.random.default_rng(914922)
    bias, external = rng.normal(scale=0.2, size=(2, 44, 6)).astype(np.float32)
    actuation, stiffness, reference, damping, initial = rng.normal(scale=0.1, size=(5, 43)).astype(np.float32)
    current_qd, stage3_qd = rng.normal(scale=0.4, size=(2, 43)).astype(np.float32)
    solver.body_f_s.assign(bias)
    state.body_f.assign(external)
    control.joint_f.assign(actuation)
    solver._passive_spring_stiffness.assign(stiffness)
    solver._passive_spring_ref.assign(reference)
    solver._passive_joint_damping.assign(damping)
    solver.joint_tau.assign(initial)
    state.joint_qd.assign(current_qd)
    stage3 = wp.array(stage3_qd, dtype=float, device=model.device)
    inputs = small_step_force.make_force_input(solver, state, solver, control, stage3, 1.0 / 240.0, plan=plan)
    result.update(force_plan=plan, force_input=inputs, control=control)
    return result


def force_reference(case):
    """Accumulate the original current-wrench tree without descendant lists."""
    model, f = case["model"], case["force_input"]
    poses, com, origin = f.body_q.numpy(), f.body_com.numpy(), f.origin.numpy()[0]
    external = f.external.numpy().astype(float)
    external[(f.body_flags.numpy() & 2) != 0] = 0
    local = f.body_f_s.numpy().astype(float)
    for body in range(44):
        center = np.asarray(wp.transform_point(wp.transform(*poses[body]), wp.vec3(*com[body])), float) - origin
        local[body, :3] -= external[body, :3]
        local[body, 3:] -= external[body, 3:] + np.cross(center, external[body, :3])
    subtree = local.copy()
    parent = model.joint_parent.numpy()
    for body in range(43, 0, -1):
        subtree[parent[body]] += subtree[body]
    screws = f.joint_S_s.numpy().astype(float)
    joints = case["host"]["dof_joint"]
    tau = -(screws * subtree[joints]).sum(axis=1) + f.actuation.numpy() + f.u0.numpy()
    q_index = model.joint_q_start.numpy()[joints[6:]]
    tau[6:] += f.stiffness.numpy()[6:] * (f.reference.numpy()[6:] - f.q.numpy()[q_index])
    tau[6:] -= f.damping.numpy()[6:] * f.current_qd.numpy()[6:]
    return tau, local


def predictor_reference(case, tau, matrix):
    """Apply the held W twice, then the original live-mask root transport."""
    f = case["force_input"]
    acceleration = (matrix.T @ (matrix @ tau[::-1]))[::-1]
    acceleration[f.kinematic_dof_mask.numpy() != 0] = 0
    qd = f.stage3_qd.numpy().astype(float)
    predicted = qd + f.dt * acceleration
    if not f.kinematic_joint_mask.numpy()[0]:
        predicted[:3] += f.dt * np.cross(qd[3:6], qd[:3])
    return predicted


@cache
def get_force_probe_kernel():
    """Expose the composed fragment outputs only to the assigned native tests."""
    source = (
        r"""
#if defined(__CUDA_ARCH__)
    const int art = p.group_to_art.data[group], world = p.art_to_world.data[art];
    const int start = p.art_dof_start.data[art], lane = threadIdx.x;
"""
        + small_step_force.get_force_source()
        + r"""
    for (int natural = lane; natural < 43; natural += 32) {
        output.data[natural] = small_tau[natural];
        output.data[43+natural] = small_vhat[natural];
    }
#endif
"""
    )

    @wp.func_native(source)
    def native(
        group: int,
        p: sparse_factor.SparsePlan,
        d: sparse_factor.SparseData,
        f: small_step_force.ForceInput,
        output: wp.array2d[float],
    ): ...

    def probe(
        p: sparse_factor.SparsePlan,
        d: sparse_factor.SparseData,
        f: small_step_force.ForceInput,
        output: wp.array2d[float],
    ):
        group, _ = wp.tid()
        native(group, p, d, f, output)

    probe.__name__ = probe.__qualname__ = "small_step_force_probe43"
    return wp.kernel(enable_backward=False, module="unique")(probe)


class TestSmallStepForceCPU(unittest.TestCase):
    def test_source_contract(self):
        """Keep the composed predictor private and in natural coordinate order."""
        source = small_step_force.get_force_source()
        self.assertIn("small_vhat[43]", source)
        self.assertIn("small_root_force[6]", source)
        self.assertNotIn("qdd.data[", source)
        self.assertNotIn("tau.data[", source)

    def test_descendants_match_original_force_bucket(self):
        """Retain weld forces, COM shifts, passive terms and clamped u0 once."""
        case = force_fixture()
        plan, f = case["force_plan"], case["force_input"]
        self.assertEqual(plan.scalar_body_visits + plan.root_body_visits, 250)
        self.assertEqual(plan.descendant_offsets.shape, (38,))
        expected, local = force_reference(case)
        screws = f.joint_S_s.numpy().astype(float)
        offsets, bodies = plan.host["descendant_offsets"], plan.host["descendant_bodies"]
        projected = np.zeros(43)
        projected[:6] = screws[:6] @ local.sum(axis=0)
        for natural in range(6, 43):
            selected = bodies[offsets[natural - 6] : offsets[natural - 5]]
            projected[natural] = screws[natural] @ local[selected].sum(axis=0)
        tree_projection = -(expected - f.actuation.numpy() - f.u0.numpy())
        q = f.q.numpy()[plan.host["q_index"][6:]]
        tree_projection[6:] += f.stiffness.numpy()[6:] * (f.reference.numpy()[6:] - q)
        tree_projection[6:] -= f.damping.numpy()[6:] * f.current_qd.numpy()[6:]
        np.testing.assert_allclose(projected, tree_projection, rtol=2.0e-7, atol=2.0e-7)
        # Exercise the original jcalc_tau producer as a second independent oracle.
        case["solver"]._launch_rigid_tau(case["state"], case["solver"], case["control"], add_to_existing=True)
        np.testing.assert_allclose(case["solver"].joint_tau.numpy(), expected, rtol=2.0e-6, atol=2.0e-6)

    def test_model_and_binding_contract(self):
        """Reject changed layouts and keep distinct current and stage3 velocities."""
        case = force_fixture()
        f, model = case["force_input"], case["model"]
        self.assertIs(f.current_qd, case["state"].joint_qd)
        self.assertIs(f.u0, case["solver"].joint_tau)
        self.assertFalse(np.array_equal(f.current_qd.numpy(), f.stage3_qd.numpy()))
        parent = model.joint_parent.numpy()
        parent[7] = 1
        model.joint_parent.assign(parent)
        with self.assertRaisesRegex(ValueError, "topology"):
            small_step_force.build_force_plan(model)


class TestSmallStepForceCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Use only the parent-assigned visible GPU for actual native checks."""
        wp.init()
        if not wp.get_cuda_devices():
            raise unittest.SkipTest("Native force fragment requires the assigned CUDA device")
        cls.device = wp.get_cuda_devices()[0]

    def test_native_current_held_transport_and_masks(self):
        """Preserve current forces, held response and transport without global writes."""
        case = force_fixture(self.device)
        owner, solver, f = (case[name] for name in ("owner", "solver", "force_input"))
        owner.refresh(solver)
        owner.check()
        matrix = unpack(owner)
        held = owner.data.W.numpy().copy()
        output = wp.zeros((2, 43), dtype=float, device=self.device)
        kernel = get_force_probe_kernel()
        original_u0 = f.u0.numpy().copy()
        untouched = (solver.joint_qdd.numpy().copy(), solver.v_hat.numpy().copy())

        def run():
            wp.launch_tiled(
                kernel, dim=[1], inputs=[owner.plan, owner.data, f, output], block_dim=32, device=self.device
            )

        for current_change in (False, True):
            if current_change:
                screws = f.joint_S_s.numpy()
                screws[6:] += np.random.default_rng(914923).normal(scale=0.01, size=(37, 6)).astype(np.float32)
                f.joint_S_s.assign(screws)
                solver.mass_update_mask.zero_()
            tau, _ = force_reference(case)
            expected = predictor_reference(case, tau, matrix)
            run()
            actual = output.numpy()
            np.testing.assert_allclose(actual[0], tau, rtol=3.0e-6, atol=3.0e-6)
            np.testing.assert_allclose(actual[1], expected, rtol=3.0e-5, atol=3.0e-6)
            qd = f.stage3_qd.numpy().astype(float)
            acceleration = (actual[1] - qd) / f.dt
            acceleration[:3] -= np.cross(qd[3:6], qd[:3])
            mass = case["H"]
            defect = np.linalg.norm(mass @ acceleration - actual[0], np.inf) / (
                1
                + np.linalg.norm(mass, np.inf) * np.linalg.norm(acceleration, np.inf)
                + np.linalg.norm(actual[0], np.inf)
            )
            self.assertLess(defect, 2.0e-6)
            np.testing.assert_array_equal(owner.data.W.numpy(), held)

        # Live masks are preserved even though ordinary admission rejects kinematics.
        dof_mask = np.zeros(43, np.int32)
        dof_mask[:6] = 1
        joint_mask = np.zeros(44, np.int32)
        joint_mask[0] = 1
        f.kinematic_dof_mask.assign(dof_mask)
        f.kinematic_joint_mask.assign(joint_mask)
        flags = f.body_flags.numpy()
        flags[7] = 2
        f.body_flags.assign(flags)
        tau, _ = force_reference(case)
        expected = predictor_reference(case, tau, matrix)
        run()
        np.testing.assert_allclose(output.numpy()[1], expected, rtol=3.0e-5, atol=3.0e-6)
        with wp.ScopedCapture(device=self.device) as capture:
            run()
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(output.numpy()[1], expected, rtol=3.0e-5, atol=3.0e-6)
        np.testing.assert_array_equal(f.u0.numpy(), original_u0)
        np.testing.assert_array_equal(solver.joint_qdd.numpy(), untouched[0])
        np.testing.assert_array_equal(solver.v_hat.numpy(), untouched[1])


if __name__ == "__main__":
    unittest.main()
