# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Production lifecycle controls using the saved first Allegro articulation pair.

The model inertias, joint frames, axes, limits and state are captured inputs.
Small box shapes and repeated synthetic witnesses exercise dispatch, not collision
accuracy. The separate kinetic-row tests retain the actual captured contacts.
"""

import copy
import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import allegro_kinetic_rows as rows
from newton._src.solvers.feather_pgs import solver_feather_pgs as source
from tools.fpgs_bench.test_allegro_kinetic_rows import captures


@contextmanager
def mode(enabled):
    """Bind the existing INK recipe and the new flag before construction."""
    with (
        patch.dict(os.environ, {"FEATHER_PGS_ALLEGRO_KINETIC_ROWS": str(int(enabled))}),
        patch.object(source, "_INK_ON", True),
    ):
        yield


def saved_model(device):
    """Reconstruct exactly the first saved world's 22-body joint/inertia model."""
    with np.load(next(captures())) as data:
        builder = newton.ModelBuilder()
        builder.begin_world()
        for body in range(22):
            pose = data["state_body_q"][body]
            builder.add_link(
                xform=wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:])),
                mass=float(data["full_model_body_mass"][body]),
                inertia=wp.mat33(*data["full_model_body_inertia"][body].ravel()),
                com=wp.vec3(*data["full_model_body_com"][body]),
                lock_inertia=True,
            )
        properties = (
            "limit_lower",
            "limit_upper",
            "velocity_limit",
            "armature",
            "target_ke",
            "target_kd",
            "effort_limit",
            "damping",
        )
        for joint in range(22):
            first = int(data["full_model_joint_qd_start"][joint])
            linear, angular = map(int, data["full_model_joint_dof_dim"][joint])
            axes = [
                newton.ModelBuilder.JointDofConfig(
                    axis=wp.vec3(*data["full_model_joint_axis"][dof]),
                    **{name: float(data[f"full_model_joint_{name}"][dof]) for name in properties},
                )
                for dof in range(first, first + linear + angular)
            ]
            xp, xc = data["full_model_joint_X_p"][joint], data["full_model_joint_X_c"][joint]
            builder.add_joint(
                newton.JointType(int(data["full_model_joint_type"][joint])),
                int(data["full_model_joint_parent"][joint]),
                int(data["full_model_joint_child"][joint]),
                linear_axes=axes[:linear],
                angular_axes=axes[linear:],
                parent_xform=wp.transform(wp.vec3(*xp[:3]), wp.quat(*xp[3:])),
                child_xform=wp.transform(wp.vec3(*xc[:3]), wp.quat(*xc[3:])),
            )
        builder.add_articulation(list(range(21)))
        builder.add_articulation([21])
        # Preserve authored inertia despite these deliberately synthetic shapes.
        builder.add_shape_box(5, hx=0.01, hy=0.01, hz=0.01)
        builder.add_shape_box(21, hx=0.025, hy=0.025, hz=0.025)
        builder.end_world()
        builder.joint_q[:] = data["state_joint_q"][:23].tolist()
        builder.joint_qd[:] = data["state_joint_qd"][:22].tolist()
        model = builder.finalize(device=device)
        np.testing.assert_array_equal(model.body_flags.numpy(), data["full_model_body_flags"][:22])
        model.gravity.assign(data["full_model_gravity"][: model.gravity.shape[0]])
        model.rigid_contact_max = 64
        return model


def make_solver(model, enabled, **overrides):
    options = {
        "pgs_mode": "matrix_free",
        "drive_mode": "augmented",
        "friction_mode": "current",
        "pgs_schedule": "interleaved",
        "pgs_iterations": 12,
        "pgs_velocity_iterations": 0,
        "pgs_beta": 0.05,
        "dense_max_constraints": 192,
        "mf_max_constraints": 64,
        "enable_joint_limits": True,
        "joint_limit_activation_gap": 0.2,
        "enable_joint_velocity_limits": True,
        "velocity_limit_activation_fraction": 0.7,
        "fuse_joint_velocity_limits": False,
        "mf_gs_parallel_rows": 128,
        "mf_gs_parallel_matrix_free": True,
        "mf_gs_parallel_sweeps": 24,
        "update_mass_matrix_interval": 2,
        "grouped_dynamics": True,
    }
    options.update(overrides)
    with mode(enabled):
        return source.SolverFeatherPGS(model, **options)


def contacts_for(model):
    contacts = newton.Contacts(64, 0, device=model.device)
    contacts.rigid_contact_shape0.fill_(0)
    contacts.rigid_contact_shape1.fill_(1)
    contacts.rigid_contact_normal.assign(np.tile([0.0, 0.0, 1.0], (64, 1)).astype(np.float32))
    # Coincident world witnesses initially, with a 0.2 mm penetration.
    poses = model.body_q.numpy()
    pose0 = wp.transform(wp.vec3(*poses[5, :3]), wp.quat(*poses[5, 3:]))
    pose1 = wp.transform(wp.vec3(*poses[21, :3]), wp.quat(*poses[21, 3:]))
    world = wp.transform_point(pose0, wp.vec3(0.0, 0.0, 0.01))
    p1 = wp.transform_point(wp.transform_inverse(pose1), world - wp.vec3(0.0, 0.0, 0.0002))
    contacts.rigid_contact_point0.assign(np.tile([0.0, 0.0, 0.01], (64, 1)).astype(np.float32))
    contacts.rigid_contact_point1.assign(np.tile(np.asarray(p1), (64, 1)).astype(np.float32))
    return contacts


class TestAllegroKineticLifecycleCPU(unittest.TestCase):
    def test_saved_model_and_constructor_admission(self):
        model = saved_model("cpu")
        with self.assertRaisesRegex(NotImplementedError, "requires CUDA"):
            make_solver(model, True)
        solver = make_solver(model, True, pgs_mode="split", enable_joint_velocity_limits=False)
        self.assertIsNone(solver._allegro_kinetic_rows)  # CUDA-only owner, never a CPU stand-in.
        self.assertEqual(model.body_count, 22)
        self.assertEqual(model.joint_dof_count, 22)
        self.assertEqual(sorted(solver.size_groups), [6, 16])
        np.testing.assert_array_equal(model.articulation_start.numpy(), [0, 21, 22])
        self.assertEqual(contacts_for(model).rigid_contact_max, 64)
        # Exercise the real admission predicates without creating CUDA arrays.
        admitted = copy.copy(solver)
        admitted.model = SimpleNamespace(device=SimpleNamespace(is_cuda=True), requires_grad=False)
        admitted.pgs_mode = "matrix_free"
        admitted.enable_joint_velocity_limits = True
        admitted._hinv_jt_writes_world = True
        sentinel = object()
        with mode(True), patch.object(rows, "AllegroKineticRows", return_value=sentinel):
            self.assertIs(rows.create_owner(admitted), sentinel)
            for name, value in (
                ("mf_gs_response_block_rows", 32),
                ("dense_max_constraints", 99),
                ("pgs_warmstart", True),
                ("articulated_contact_response", "propagation"),
                ("pgs_velocity_iterations", 1),
                ("_debug_buffers_enabled", True),
            ):
                with self.subTest(unsupported=name), patch.object(admitted, name, value):
                    self.assertIsNone(rows.create_owner(admitted))


class TestAllegroKineticLifecycleCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA required; run only under the root's GPU lease")
        cls.device = wp.get_device("cuda:0")

    def test_production_step_held_refresh_fallback_reset_and_graph(self):
        model = saved_model(self.device)
        original, candidate = make_solver(model, False), make_solver(model, True)
        self.assertIsNone(original._allegro_kinetic_rows)
        self.assertIsNotNone(candidate._allegro_kinetic_rows)
        self.assertTrue(candidate._allegro_kinetic_rows.keyed_rows)
        self.assertEqual(candidate._allegro_kinetic_rows.data.rowkeys.shape, (1, candidate.dense_max_constraints))
        self.assertFalse(hasattr(candidate._allegro_kinetic_rows.data, "coefficients"))
        self.assertFalse(hasattr(candidate._allegro_kinetic_rows.data, "encoding"))
        for buffers in candidate._J_bufs:
            for value in buffers.values():
                value.fill_(float("nan"))
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        contacts = contacts_for(model)
        control = model.control()
        out_original, out_candidate = model.state(), model.state()
        dt = 1.0 / 240.0

        def paired_step(count):
            contacts.rigid_contact_count.fill_(count)
            state.clear_forces()
            original.step(state, out_original, control, contacts, dt)
            candidate.step(state, out_candidate, control, contacts, dt)
            for field in ("joint_q", "joint_qd", "body_q", "body_qd"):
                actual, expected = getattr(out_candidate, field).numpy(), getattr(out_original, field).numpy()
                self.assertTrue(np.isfinite(actual).all(), field)
                np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=3e-5, err_msg=field)
            np.testing.assert_array_equal(candidate.constraint_count.numpy(), original.constraint_count.numpy())
            np.testing.assert_array_equal(candidate._constraint_capacity_status.numpy(), np.zeros(4, dtype=np.int32))

        paired_step(0)
        self.assertTrue(all(np.isnan(value.numpy()).all() for value in candidate.J_by_size.values()))
        held = {n: candidate.L_by_size[n].numpy().copy() for n in (6, 16)}
        # Current q/geometry changes while the second call keeps the held factor.
        q = state.joint_q.numpy()
        q[:16] += np.float32(0.001)
        state.joint_q.assign(q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        paired_step(3)
        for n in (6, 16):
            np.testing.assert_array_equal(candidate.L_by_size[n].numpy(), held[n])
        paired_step(45)
        self.assertGreater(int(candidate.constraint_count.numpy()[0]), 128)
        for n in (6, 16):
            count = int(candidate.constraint_count.numpy()[0])
            # Original end-of-step maintenance has already cleared its J;
            # compare final physics above, and require complete native fallback
            # materialization here (including all previously poisoned entries).
            current_j = candidate.J_by_size[n].numpy()[:, :count]
            self.assertTrue(np.isfinite(current_j).all())
            self.assertGreater(float(np.max(np.abs(current_j))), 0.0)
        paired_step(0)
        paired_step(3)
        for solver in (original, candidate):
            solver.reset(state)
            solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        paired_step(3)

        # Capture two real steps so the interval-2 mass cadence is represented.
        # The same input is intentional: replay tests cache/source ownership, not
        # a long trajectory or a new physical acceptance metric.
        # Match FeatherPGSManager._prepare_cuda_graph_capture: completed eager
        # maintenance may leave events from outside this graph. Drain that work,
        # then seed both buffer waits INSIDE each capture, retaining all streams.
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(device=self.device) as reference_capture:
            original.seed_double_buffer_events()
            original.step(state, out_original, control, contacts, dt)
            original.step(state, out_original, control, contacts, dt)
        with wp.ScopedCapture(device=self.device) as capture:
            candidate.seed_double_buffer_events()
            candidate.step(state, out_candidate, control, contacts, dt)
            candidate.step(state, out_candidate, control, contacts, dt)
        for count in (0, 45, 3, 0, 3):
            contacts.rigid_contact_count.fill_(count)
            wp.capture_launch(reference_capture.graph)
            wp.capture_launch(capture.graph)
            self.assertTrue(np.isfinite(out_candidate.joint_qd.numpy()).all())
            np.testing.assert_allclose(
                out_candidate.joint_qd.numpy(), out_original.joint_qd.numpy(), rtol=2e-4, atol=3e-5
            )
            current = int(candidate.constraint_count.numpy()[0])
            if count == 45:
                self.assertGreater(current, 128)
            else:
                self.assertLessEqual(current, 128)
            np.testing.assert_array_equal(candidate._constraint_capacity_status.numpy(), np.zeros(4, dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
