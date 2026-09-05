# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dormant contact store shared by the collision pipeline and SolverMuJoCo."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo

_ROW_FIELDS = (
    "rigid_contact_shape0",
    "rigid_contact_shape1",
    "rigid_contact_point0",
    "rigid_contact_point1",
    "rigid_contact_offset0",
    "rigid_contact_offset1",
    "rigid_contact_normal",
    "rigid_contact_margin0",
    "rigid_contact_margin1",
)


def _sorted_rows(contacts) -> np.ndarray:
    """Return the live rows as a sorted flat table so buffers of different order compare equal."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    columns = []
    for name in _ROW_FIELDS:
        values = getattr(contacts, name).numpy()[:count]
        columns.append(values.reshape(count, -1).astype(np.float64))
    table = np.concatenate(columns, axis=1)
    order = np.lexsort(table.T[::-1])
    return table[order]


def _build_supported_box_model(device, *, world_count: int = 1) -> tuple[newton.Model, int, int]:
    """Build a dynamic SDF box resting on a kinematic (mocap) SDF plate in every world.

    Returns the model plus the per-world local indices of the dynamic body and shape.
    """
    support_mesh = newton.Mesh.create_box(0.3, 0.3, 0.08, duplicate_vertices=False)
    dynamic_mesh = newton.Mesh.create_box(0.1, 0.1, 0.08, duplicate_vertices=False)
    for mesh in (support_mesh, dynamic_mesh):
        mesh.build_sdf(device=device, max_resolution=16, narrow_band_range=(-0.02, 0.02), margin=0.02)

    template = newton.ModelBuilder()
    template.rigid_gap = 0.005
    # A kinematic articulated fixed root becomes a MuJoCo mocap body without a tree
    # (sleep index -2), the same classification as fixed-base fixtures in Isaac Lab.
    support_body = template.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
        is_kinematic=True,
    )
    template.add_articulation([template.add_joint_fixed(parent=-1, child=support_body)])
    template.add_shape_mesh(body=support_body, mesh=support_mesh)
    dynamic_body = template.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.158), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
        lock_inertia=True,
    )
    dynamic_shape = template.add_shape_mesh(body=dynamic_body, mesh=dynamic_mesh)

    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.005
    for world in range(world_count):
        builder.add_world(template, xform=wp.transform((2.0 * world, 0.0, 0.0), wp.quat_identity()))
    return builder.finalize(device=device), dynamic_body, dynamic_shape


class TestContactDormantStore(unittest.TestCase):
    def setUp(self):
        if not wp.is_cuda_available():
            self.skipTest("Texture SDF construction requires CUDA")
        self.device = wp.get_device()

    def _make_sim(self, model, *, store: bool, slab_rows: int = 64, dormant_contact_filter: bool = True):
        solver = SolverMuJoCo(
            model,
            enable_sleeping=True,
            nvmax=model.joint_dof_count // model.world_count,
            iterations=10,
            ls_iterations=5,
            njmax=128,
            nconmax=64,
            use_mujoco_contacts=False,
            dormant_contact_filter=dormant_contact_filter,
        )
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="sap",
            rigid_contact_max=64,
            max_triangle_pairs=4096,
            deterministic=True,
            verify_buffers=False,
            sdf_contact_replay_max=64 if store else 0,
            sdf_contact_slab_rows=slab_rows,
        )
        shape_sleep_index, tree_asleep = solver.collision_sleep_filter
        pipeline.configure_sleep_filter(shape_sleep_index, tree_asleep)
        self.assertEqual(pipeline.dormant_contact_store is not None, store)
        state_in = model.state()
        state_out = model.state()
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        return solver, pipeline, state_in, state_out, model.control(), pipeline.contacts()

    def _sleep_body(self, solver, state, body: int, world: int) -> None:
        solver.set_body_sleep_state(
            wp.array([[body]], dtype=wp.int32, device=self.device),
            wp.array([[True]], dtype=wp.bool, device=self.device),
            wp.array([world], dtype=wp.int32, device=self.device),
        )
        solver.reset(state, flags=0)

    def _snapshot(self, solver, state_out) -> dict[str, np.ndarray]:
        return {
            "joint_q": state_out.joint_q.numpy(),
            "joint_qd": state_out.joint_qd.numpy(),
            "body_q": state_out.body_q.numpy(),
            "body_qd": state_out.body_qd.numpy(),
            "qacc": solver.mjw_data.qacc.numpy(),
            "qfrc_constraint": solver.mjw_data.qfrc_constraint.numpy(),
        }

    def _run_sleep_wake_scenario(self, *, store: bool, slab_rows: int = 64, dormant_contact_filter: bool = True):
        """Awake seed -> sleep -> quiet substep -> force wake -> one tracked substep."""
        model, dynamic_body, dynamic_shape = _build_supported_box_model(self.device)
        solver, pipeline, state_in, state_out, control, contacts = self._make_sim(
            model, store=store, slab_rows=slab_rows, dormant_contact_filter=dormant_contact_filter
        )
        shape_sleep_index, tree_asleep = solver.collision_sleep_filter
        dynamic_world, dynamic_tree = (int(value) for value in shape_sleep_index.numpy()[dynamic_shape])
        result = {}

        pipeline.collide(state_in, contacts)
        result["awake_count"] = int(contacts.rigid_contact_count.numpy()[0])
        result["awake_pairs"] = int(pipeline.narrow_phase.shape_pairs_mesh_mesh_count.numpy()[0])
        self._sleep_body(solver, state_in, dynamic_body, dynamic_world)
        self.assertGreaterEqual(int(tree_asleep.numpy()[dynamic_world, dynamic_tree]), 0)

        pipeline.collide(state_in, contacts)
        result["asleep_count"] = int(contacts.rigid_contact_count.numpy()[0])
        result["asleep_pairs"] = int(pipeline.narrow_phase.shape_pairs_mesh_mesh_count.numpy()[0])
        dormant_store = pipeline.dormant_contact_store
        if dormant_store is not None:
            slab = int(dormant_store.slab_of_shape.numpy()[dynamic_shape])
            self.assertGreaterEqual(slab, 0)
            result["slab"] = slab
            result["slab_count"] = int(dormant_store.slab_count.numpy()[slab])
            result["slab_overflow"] = int(dormant_store.slab_overflow.numpy()[slab])

        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        self.assertGreaterEqual(int(tree_asleep.numpy()[dynamic_world, dynamic_tree]), 0)
        result["quiet_nacon"] = int(solver.mjw_data.nacon.numpy()[0])

        body_force = np.zeros((model.body_count, 6), dtype=np.float32)
        body_force[dynamic_body, 3] = 20.0
        state_in.body_f.assign(body_force)
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        self.assertLess(int(tree_asleep.numpy()[dynamic_world, dynamic_tree]), 0)
        result["woken_nacon"] = int(solver.mjw_data.nacon.numpy()[0])
        result["woken_count"] = int(contacts.rigid_contact_count.numpy()[0])
        result["woken_rows"] = _sorted_rows(contacts)
        result["woken_nefc"] = int(solver.mjw_data.nefc.numpy()[0])
        result["woken_state"] = self._snapshot(solver, state_out)

        # The fast path must track the injected rows on the next substep.
        state_in, state_out = state_out, state_in
        state_in.body_f.assign(body_force)
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        result["tracked_nacon"] = int(solver.mjw_data.nacon.numpy()[0])
        result["tracked_state"] = self._snapshot(solver, state_out)
        result["handles"] = (solver, pipeline, state_in, state_out, control, contacts, dynamic_body, dynamic_world)
        return result

    def _assert_states_match(self, actual, expected):
        for name, values in expected.items():
            np.testing.assert_allclose(actual[name], values, rtol=1.0e-5, atol=1.0e-6, err_msg=name)

    def test_store_parks_sleeping_rows_masks_pair_and_injects_on_wake(self):
        """Dormant rows leave the live buffer, their pair is skipped, and a wake restores them in-substep."""
        reference = self._run_sleep_wake_scenario(store=False, dormant_contact_filter=False)
        stored = self._run_sleep_wake_scenario(store=True)

        count = reference["awake_count"]
        self.assertGreater(count, 0)
        self.assertGreater(reference["awake_pairs"], 0)
        self.assertEqual(stored["awake_count"], count)
        # (a) live rows exclude the sleeping body; the slab holds exactly its rows.
        self.assertEqual(reference["asleep_count"], count)
        self.assertEqual(stored["asleep_count"], 0)
        self.assertEqual(stored["slab_count"], count)
        self.assertEqual(stored["slab_overflow"], 0)
        # (b) the unchanged sleeping pair is masked from the SDF narrow phase.
        self.assertGreater(reference["asleep_pairs"], 0)
        self.assertEqual(stored["asleep_pairs"], 0)
        # Quiet substep: nothing reaches MJWarp from the store, everything does without it.
        self.assertEqual(reference["quiet_nacon"], count)
        self.assertEqual(stored["quiet_nacon"], 0)
        # (c) the wake substep injects the slab rows before the solve.
        self.assertEqual(stored["woken_nacon"], count)
        self.assertEqual(stored["woken_count"], count)
        self.assertEqual(reference["woken_nacon"], count)
        np.testing.assert_array_equal(stored["woken_rows"], reference["woken_rows"])
        self.assertGreater(reference["woken_nefc"], 0)
        self.assertEqual(stored["woken_nefc"], reference["woken_nefc"])
        self._assert_states_match(stored["woken_state"], reference["woken_state"])
        self.assertEqual(stored["tracked_nacon"], count)
        self._assert_states_match(stored["tracked_state"], reference["tracked_state"])

    def test_store_does_not_inject_twice_within_one_generation(self):
        """A tree that re-sleeps and wakes again before the next collide keeps its live rows once."""
        stored = self._run_sleep_wake_scenario(store=True)
        solver, _pipeline, state_in, state_out, control, contacts, dynamic_body, dynamic_world = stored["handles"]
        count = stored["woken_nacon"]
        self.assertGreater(count, 0)

        # Put the tree back to sleep without a collide: its rows are still live.
        self._sleep_body(solver, state_in, dynamic_body, dynamic_world)
        body_force = np.zeros((solver.model.body_count, 6), dtype=np.float32)
        body_force[dynamic_body, 3] = 20.0
        state_in.body_f.assign(body_force)
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        self.assertEqual(int(solver.mjw_data.nacon.numpy()[0]), count)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), count)

    def test_store_overflow_falls_back_to_live_rows(self):
        """A full slab spills to the live buffer, disables masking, and drops no contact on wake."""
        reference = self._run_sleep_wake_scenario(store=False, dormant_contact_filter=False)
        stored = self._run_sleep_wake_scenario(store=True, slab_rows=1)

        count = reference["awake_count"]
        self.assertGreater(count, 1)
        self.assertEqual(stored["asleep_count"], count - 1)
        self.assertEqual(stored["slab_overflow"], 1)
        # (e) the pair is recomputed, not masked, and the wake sees every row exactly once.
        self.assertGreater(stored["asleep_pairs"], 0)
        self.assertEqual(stored["woken_nacon"], count)
        np.testing.assert_array_equal(stored["woken_rows"], reference["woken_rows"])
        self._assert_states_match(stored["woken_state"], reference["woken_state"])

    def test_reset_contact_history_invalidates_selected_world_slabs_only(self):
        """A masked reset drops the selected world's slabs and recomputes only its pairs."""
        model, dynamic_body, dynamic_shape = _build_supported_box_model(self.device, world_count=2)
        solver, pipeline, state_in, _state_out, _control, contacts = self._make_sim(model, store=True)
        store = pipeline.dormant_contact_store
        shapes_per_world = model.shape_count // 2
        slabs = [int(store.slab_of_shape.numpy()[dynamic_shape + world * shapes_per_world]) for world in range(2)]
        bodies_per_world = model.body_count // 2

        pipeline.collide(state_in, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, 0)
        self.assertEqual(count % 2, 0)
        per_world = count // 2
        solver.set_body_sleep_state(
            wp.array(
                [[dynamic_body + world * bodies_per_world] for world in range(2)], dtype=wp.int32, device=self.device
            ),
            wp.array([[True], [True]], dtype=wp.bool, device=self.device),
            wp.array([0, 1], dtype=wp.int32, device=self.device),
        )
        solver.reset(state_in, flags=0)
        pipeline.collide(state_in, contacts)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertEqual(int(pipeline.narrow_phase.shape_pairs_mesh_mesh_count.numpy()[0]), 0)
        np.testing.assert_array_equal(store.slab_count.numpy()[slabs], [per_world, per_world])

        # (d) invalidate world 0 only: its slab empties now and refills from a recompute.
        mask = wp.array([True, False, False], dtype=wp.bool, device=self.device)
        pipeline.reset_contact_history(mask)
        np.testing.assert_array_equal(store.slab_count.numpy()[slabs], [0, per_world])
        pipeline.collide(state_in, contacts)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertEqual(int(pipeline.narrow_phase.shape_pairs_mesh_mesh_count.numpy()[0]), 1)
        np.testing.assert_array_equal(store.slab_count.numpy()[slabs], [per_world, per_world])
        # World 1 was never touched and stays masked on the following pass as well.
        pipeline.collide(state_in, contacts)
        self.assertEqual(int(pipeline.narrow_phase.shape_pairs_mesh_mesh_count.numpy()[0]), 0)

        # An all-false mask leaves both slabs alone.
        pipeline.reset_contact_history(wp.zeros(3, dtype=wp.bool, device=self.device))
        np.testing.assert_array_equal(store.slab_count.numpy()[slabs], [per_world, per_world])


if __name__ == "__main__":
    unittest.main(verbosity=2)
