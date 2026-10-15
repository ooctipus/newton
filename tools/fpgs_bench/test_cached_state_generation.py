# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check independent state generation across cached publication owners."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import franka_kinetic_state, kinetic_state
from newton._src.solvers.feather_pgs.solver_feather_pgs import _FeatherPGSModelPlan
from tools.fpgs_bench import test_franka_kinetic_state as forest
from tools.fpgs_bench import test_world_scan_publication as publication
from tools.fpgs_bench.test_g1_kinetic_state import physical_mass, public_reference, public_state


def forest_model(device, worlds=5):
    """Build a live revolute/prismatic tree with free and prescribed root services."""
    part = newton.ModelBuilder()
    parents = (-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8)
    joints = []
    for local, parent in enumerate(parents):
        body = part.add_link(
            mass=0.5 + 0.1 * local,
            inertia=wp.mat33(np.diag([0.08, 0.1, 0.12]).astype(np.float32)),
            com=wp.vec3(0.01, -0.02, 0.03),
        )
        options = {
            "parent": parent,
            "child": body,
            "parent_xform": wp.transform(wp.vec3(0.02, -0.01, 0.06), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.09)),
            "child_xform": wp.transform(wp.vec3(-0.01, 0.005, 0), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), -0.07)),
        }
        if local in (0, 8):
            joint = part.add_joint_fixed(**options)
        else:
            add = part.add_joint_prismatic if local >= 9 else part.add_joint_revolute
            joint = add(**options, axis=wp.vec3(0, 0, 1), armature=0.03, limit_lower=-1, limit_upper=1)
        joints.append(joint)
    part.add_articulation(joints)
    part.add_constraint_mimic(joints[9], joints[10], 1.0, 0.0)
    for prescribed in (False, True):
        body = part.add_link(
            mass=0.7,
            inertia=wp.mat33(np.diag([0.08, 0.1, 0.12]).astype(np.float32)),
            com=wp.vec3(-0.01, 0.02, -0.03),
            is_kinematic=prescribed,
        )
        part.add_articulation([part.add_joint_free(body)])
    for body in (9, 11, 12):
        part.add_shape_box(body, hx=0.02, hy=0.02, hz=0.02)
    builder = newton.ModelBuilder()
    builder.replicate(part, worlds, spacing=(2, 0, 0))
    model = builder.finalize(device=device)
    model.rigid_contact_max = 32
    return model


def initialize(model, state):
    """Excite every physical coordinate and rotate both independent free roots."""
    q = state.joint_q.numpy().reshape(model.world_count, 23)
    q[:, :9] = np.linspace(-0.1, 0.15, 9)
    for start in (9, 16):
        q[:, start + 3 : start + 7] = np.asarray(wp.quat_from_axis_angle(wp.normalize(wp.vec3(1, 2, -1)), 0.21))
    state.joint_q.assign(q.reshape(-1))
    state.joint_qd.assign(np.linspace(-0.12, 0.17, model.joint_dof_count, dtype=np.float32))
    state.body_f.assign(np.random.default_rng(817).normal(0, 0.3, (model.body_count, 6)).astype(np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


class TestCachedStateGeneration(unittest.TestCase):
    def test_forest_factory_independence(self):
        """Build both packed state phases without reading another owner's factory."""
        franka_kinetic_state.get_state_kernel.cache_clear()
        with patch.object(kinetic_state, "_get_kernel", side_effect=AssertionError("unrelated factory")):
            for finish in (False, True):
                kernel = franka_kinetic_state.get_state_kernel(finish)
                self.assertTrue(kernel.key.endswith("_p16"))

    def check_forest_publication(self, device, compact):
        """Compare mass and complete publication against independent physical services."""
        model = forest_model(device)
        # Split CPU construction normally retains prescribed response columns.
        # Bind the production CUDA response ABI for this publication-only test;
        # the CPU test does not execute a matrix-free constraint solve.
        original_plan = _FeatherPGSModelPlan.build

        def response_plan(model, kinematic_dof_mask, **kwargs):
            return original_plan(model, kinematic_dof_mask, enable_prescribed_response=True)

        with patch.object(_FeatherPGSModelPlan, "build", side_effect=response_plan):
            solver = forest.make_solver(model, False, friction_anchor_beta=0.0)
        with patch.dict(os.environ, {"FEATHER_PGS_FRANKA_COMPACT_WORKSPACE": str(int(compact))}):
            owner = franka_kinetic_state.FrankaKineticState(solver)
        state, output, reference = model.state(), model.state(), model.state()
        initialize(model, state)
        augmented = solver._prepare_augmented_state(state, output, model.control())
        dt = 1.0 / 240.0
        owner.begin(state, augmented, dt, True)
        public_state(self, model, state)
        mass = physical_mass(model, state)
        for world, dofs in enumerate(owner.host_plan["dof_ids"][:, :9]):
            np.testing.assert_allclose(
                owner.geometric.numpy()[world].reshape(9, 9),
                mass[np.ix_(dofs, dofs)],
                atol=3e-6,
                rtol=3e-5,
            )
        solver.v_out.assign(state.joint_qd.numpy() + np.float32(0.025))
        data = owner._publication(state, augmented, reference, dt)
        values = {name: getattr(data, name) for name in franka_kinetic_state.PublicationData.vars}
        values.update(
            articulation_start=model.articulation_start,
            articulation_joint_end=solver.articulation_joint_end,
            v_new=data.v_out,
            inv_dt=1 / dt,
        )
        publication.original(
            SimpleNamespace(
                values=values,
                device=device,
                dimensions=(
                    model.joint_dof_count,
                    solver._free_root_joint_indices.size,
                    model.joint_count,
                    model.articulation_count,
                    model.body_count,
                ),
            )
        )
        held = owner.geometric.numpy().copy()
        owner.finish(state, augmented, output, dt, False)
        for field in ("joint_q", "joint_qd"):
            np.testing.assert_allclose(
                getattr(output, field).numpy(), getattr(reference, field).numpy(), atol=3e-6, rtol=2e-5
            )
        public_state(self, model, output)
        np.testing.assert_array_equal(owner.geometric.numpy(), held)
        solver._mass_update_requested.zero_()
        generation = owner.data.generation.numpy().copy()
        owner.begin(output, augmented, dt, False)
        np.testing.assert_array_equal(owner.data.generation.numpy(), generation)
        mask = wp.array([True, False, True, False, True], dtype=wp.bool, device=device)
        owner.invalidate(mask)
        owner.begin(output, augmented, dt, False)
        np.testing.assert_array_equal(owner.data.generation.numpy(), np.add(generation, [1, 0, 1, 0, 1]))
        with self.assertRaisesRegex(RuntimeError, "alternate prescaled"):
            owner.predict(output, augmented, model.control(), wp.clone(output.joint_qd), dt)
        with self.assertRaisesRegex(RuntimeError, "disjoint"):
            owner.finish(output, augmented, output, dt, False)
        owner.check()
        if wp.get_device(device).is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                owner.begin(output, augmented, dt, True)
            owner.invalidate(mask)
            wp.capture_launch(capture.graph)
            public_state(self, model, output)
            owner.check()

    def test_live_forest_cpu(self):
        """Check both forest scratch layouts without archived captures."""
        for compact in (False, True):
            with self.subTest(compact=compact):
                self.check_forest_publication("cpu", compact)

    def test_live_forest_cuda(self):
        """Check independent half-warps, an absent tail, and graph repair on GPU0."""
        if not wp.get_cuda_devices():
            self.skipTest("No leased CUDA device")
        for compact in (False, True):
            with self.subTest(compact=compact):
                self.check_forest_publication("cuda:0", compact)

    def test_complete_forest_steps(self):
        """Preserve loaded steps, held factors and masked resets with a live forest."""
        if not wp.get_cuda_devices():
            self.skipTest("No leased CUDA device")
        model = forest_model("cuda:0")
        solvers = [forest.make_solver(model, enabled, friction_anchor_beta=0.0) for enabled in (False, True)]
        self.assertIsNotNone(solvers[1]._franka_kinetic_state)
        states = [[model.state(), model.state()] for _ in solvers]
        for pair in states:
            initialize(model, pair[0])
        controls = [model.control() for _ in solvers]
        contacts = forest.loaded_contacts(model, states[0][0])
        for step in range(8):
            for solver, pair, control in zip(solvers, states, controls, strict=True):
                if step in (3, 6):
                    solver.reset(
                        pair[0],
                        world_mask=wp.array([True, False, True, False, True], dtype=wp.bool, device=model.device),
                    )
                solver.step(pair[0], pair[1], control, contacts, 1.0 / 240.0)
                solver.check_constraint_capacity()
                if solver is solvers[1]:
                    # Integrating this loaded fixture produces ~5e-6 normalized
                    # error on the qualified parent too. Direct publication
                    # above retains the separate, stricter 3e-6 bound.
                    expected = public_reference(model, pair[1])
                    np.testing.assert_allclose(pair[1].body_q.numpy(), expected["body_q"], atol=3e-6, rtol=2e-5)
                    actual = pair[1].body_qd.numpy().astype(float).reshape(-1, 2, 3)
                    reference = expected["body_qd"].reshape(-1, 2, 3)
                    error = np.linalg.norm(actual - reference, axis=2) / (1 + np.linalg.norm(reference, axis=2))
                    self.assertLess(float(np.max(error)), 1e-5, "integrated public linear/angular velocity")
            for field in ("joint_q", "joint_qd", "body_q", "body_qd"):
                np.testing.assert_allclose(
                    getattr(states[1][1], field).numpy(), getattr(states[0][1], field).numpy(), atol=3e-4, rtol=3e-4
                )
            for pair in states:
                pair.reverse()


if __name__ == "__main__":
    unittest.main()
