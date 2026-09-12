# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent current-owner, complete fallback and actual solve controls."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import compact_contact as compact
from newton._src.solvers.feather_pgs import fused_contact_solve as fused
from newton._src.solvers.feather_pgs import solver_feather_pgs as solver_module
from newton.tests.test_feather_pgs_compact_contact import (
    OUTPUTS,
    build_solver_fixture,
    fixture,
    geometry_reference,
    install_contacts,
    run_original,
)


def routing_fixture(device="cpu"):
    """Allocate stable owners for two worlds and 128 possible triples per world."""
    data, _ = fixture(device)
    worlds = np.repeat(np.arange(2, dtype=np.int32), 128)
    for name, values in {
        "world": worlds,
        "slot": np.full(256, -1, np.int32),
        "path": np.full(256, -1, np.int32),
        "slots_needed": np.full(256, 3, np.int32),
        "art0": np.where(worlds == 0, 1, 4),
        "art1": np.full(256, -1, np.int32),
    }.items():
        setattr(data, name, wp.array(values, dtype=int, device=device))
    data.count.assign(np.array([256], np.int32))
    aux = SimpleNamespace(
        device=device,
        counts=wp.zeros(2, dtype=int, device=device),
        bounds=wp.zeros((2, 2), dtype=int, device=device),
        row_contact=wp.full((2, 704), -777, dtype=int, device=device),
        owner=wp.full(2, -777, dtype=int, device=device),
        fallback_counts=wp.full(2, -777, dtype=int, device=device),
        fallback_bounds=wp.full((2, 2), -777, dtype=int, device=device),
    )
    return data, aux


def set_routes(data, aux, *, prefix=(0, 0), triples=(0, 0), dense=(0, 0)):
    """Change current reservations without clearing the previous inverse map."""
    slot = np.full(256, -1, np.int32)
    art = np.repeat(np.array([1, 4], np.int32), 128)
    for world in range(2):
        base = 128 * world
        slot[base : base + triples[world]] = prefix[world] + 3 * np.arange(triples[world])
        art[base : base + dense[world]] = 3 * world
    data.slot.assign(slot)
    data.path.assign(np.where(slot >= 0, 0, -1).astype(np.int32))
    data.slots_needed.fill_(3)
    data.art0.assign(art)
    data.count.assign(np.array([256], np.int32))
    aux.counts.assign(np.array(prefix, np.int32) + 3 * np.array(triples, np.int32))
    # The first marker is deliberately different from the contact prefix.
    aux.bounds.assign(np.array([[0, prefix[0]], [0, prefix[1]]], np.int32))


def launch_route(data, aux, *, invert=True):
    """Use actual kernels and keep classification separately testable against stale maps."""
    if invert:
        wp.launch(fused.route_contacts, dim=256, inputs=[data, aux.row_contact], device=aux.device)
    wp.launch(
        fused.classify_worlds,
        dim=2,
        inputs=[data, aux.counts, aux.bounds, aux.row_contact, aux.owner, aux.fallback_counts, aux.fallback_bounds],
        device=aux.device,
    )


def wide_fixture(device="cpu", **kwargs):
    """Keep the concrete original geometry but poison the entire canonical 704-row allocation."""
    data, aux = fixture(device, **kwargs)
    for name in OUTPUTS:
        old = getattr(data, name)
        shape = (old.shape[0], 704, *old.shape[2:])
        value = -777 if old.dtype == wp.int32 else float("nan")
        setattr(data, name, wp.full(shape, value, dtype=old.dtype, device=device))
    return data, aux


class TestFusedContactSolve(unittest.TestCase):
    def test_exact_owner_thresholds_and_current_map_transitions(self):
        """Separate 384/385 rows and 32/33 dense contacts through empty and stale-map reuse."""
        data, aux = routing_fixture()
        cases = (
            ((0, 1), (128, 128), (0, 0), [1, 0]),
            ((0, 0), (32, 33), (32, 33), [1, 0]),
            ((12, 0), (0, 0), (0, 0), [1, 1]),
            ((1, 12), (127, 124), (1, 32), [1, 1]),
            ((13, 0), (0, 1), (0, 0), [0, 1]),
            ((0, 0), (0, 0), (0, 0), [1, 1]),
        )
        pointers = (aux.row_contact.ptr, aux.owner.ptr, aux.fallback_counts.ptr, aux.fallback_bounds.ptr)
        for prefix, triples, dense, expected in cases:
            set_routes(data, aux, prefix=prefix, triples=triples, dense=dense)
            before = aux.row_contact.numpy().copy()
            launch_route(data, aux)
            np.testing.assert_array_equal(aux.owner.numpy(), expected)
            after = aux.row_contact.numpy()
            for world in range(2):
                start, end = prefix[world], prefix[world] + 3 * triples[world]
                np.testing.assert_array_equal(after[world, :start], before[world, :start])
                np.testing.assert_array_equal(after[world, end:], before[world, end:])
                np.testing.assert_array_equal(
                    after[world, start:end], np.repeat(128 * world + np.arange(triples[world]), 3)
                )
                if expected[world]:
                    self.assertEqual(aux.fallback_counts.numpy()[world], 0)
                    np.testing.assert_array_equal(aux.fallback_bounds.numpy()[world], [0, 0])
                else:
                    self.assertEqual(aux.fallback_counts.numpy()[world], aux.counts.numpy()[world])
                    np.testing.assert_array_equal(aux.fallback_bounds.numpy()[world], aux.bounds.numpy()[world])
            self.assertEqual(
                pointers, (aux.row_contact.ptr, aux.owner.ptr, aux.fallback_counts.ptr, aux.fallback_bounds.ptr)
            )

    def test_stale_source_slot_path_triple_and_count_refuse_ownership(self):
        """Reject stale in-range mappings before accepting a current complete triple owner."""
        data, aux = routing_fixture()
        set_routes(data, aux, prefix=(2, 3), triples=(2, 2))
        launch_route(data, aux)
        np.testing.assert_array_equal(aux.owner.numpy(), [1, 1])
        for name, value in (("slot", 5), ("path", -1), ("slots_needed", 2), ("world", 1)):
            original = getattr(data, name).numpy().copy()
            changed = original.copy()
            changed[0] = value
            getattr(data, name).assign(changed)
            launch_route(data, aux, invert=False)
            np.testing.assert_array_equal(aux.owner.numpy(), [0, 1], err_msg=name)
            getattr(data, name).assign(original)
        original_map = aux.row_contact.numpy().copy()
        for bad in (-1, 256, 128, 1):
            changed = original_map.copy()
            changed[0, 2] = bad
            aux.row_contact.assign(changed)
            launch_route(data, aux, invert=False)
            np.testing.assert_array_equal(aux.owner.numpy(), [0, 1], err_msg=str(bad))
        aux.row_contact.assign(original_map)
        data.count.assign(np.array([1], np.int32))
        launch_route(data, aux, invert=False)
        np.testing.assert_array_equal(aux.owner.numpy(), [0, 0])

    def test_rejected_raw_counts_clamp_only_fallback_launch_domain(self):
        """Keep 705/706 raw overflow evidence while bounding fallback initialization at 704."""
        data, aux = routing_fixture()
        for counts in ([705, 706], [-1, 704]):
            aux.counts.assign(np.array(counts, np.int32))
            aux.bounds.assign(np.array([[0, 2], [1, 3]], np.int32))
            launch_route(data, aux)
            np.testing.assert_array_equal(aux.owner.numpy(), [0, 0])
            np.testing.assert_array_equal(aux.counts.numpy(), counts)
            np.testing.assert_array_equal(aux.fallback_counts.numpy(), np.clip(counts, 0, 704))
            np.testing.assert_array_equal(aux.fallback_bounds.numpy(), [[0, 2], [1, 3]])

    def test_masked_fallback_complete_original_law_and_float64_action(self):
        """Preserve original geometry, prescribed targets, held-factor action and untouched owned rows."""
        for selected in (0, 1):
            original, aux = fixture(shared=0, friction_shared=1)
            candidate, _ = wide_fixture(shared=0, friction_shared=1)
            reference = geometry_reference(candidate)
            run_original(original, aux)
            owner_host = np.array([selected == 0, selected == 1], np.int32)
            owner = wp.array(owner_host, dtype=int, device="cpu")
            fallback = 1 - selected
            counts = aux.counts.numpy().copy()
            bounds = aux.bounds.numpy().copy()
            counts[selected] = 0
            bounds[selected] = 0
            before = {name: getattr(candidate, name).numpy().copy() for name in OUTPUTS}
            factor_before = candidate.factor.numpy().copy()
            wp.launch(
                compact.clear_contact_response,
                dim=(2, 256),
                inputs=[
                    wp.array(counts, dtype=int, device="cpu"),
                    wp.array(bounds, dtype=int, device="cpu"),
                    candidate.dense_group,
                    candidate.J,
                    candidate.Y,
                ],
                block_dim=256,
                device="cpu",
            )
            wp.launch(fused.produce_fallback, dim=16, inputs=[candidate, owner], device="cpu")
            start, end = aux.bounds.numpy()[fallback, 1], aux.counts.numpy()[fallback]
            for name in OUTPUTS:
                a = getattr(original, name).numpy()
                b = getattr(candidate, name).numpy()
                groups = candidate.dense_group.numpy() if name in ("J", "Y") else np.arange(2)
                index = groups[fallback]
                if a.dtype.kind in "iu":
                    np.testing.assert_array_equal(b[index, start:end], a[index, start:end], err_msg=name)
                else:
                    np.testing.assert_allclose(
                        b[index, start:end], a[index, start:end], rtol=3e-6, atol=3e-6, err_msg=name
                    )
                np.testing.assert_array_equal(b[groups[selected]], before[name][groups[selected]])
                np.testing.assert_array_equal(b[index, :start], before[name][index, :start])
                np.testing.assert_array_equal(b[index, end:], before[name][index, end:])
            np.testing.assert_array_equal(candidate.factor.numpy(), factor_before)
            velocity = np.linspace(-0.7, 0.8, 114)
            for world, row, dense, sparse, target, phi in reference:
                if world != fallback:
                    continue
                group = candidate.dense_group.numpy()[world]
                J = candidate.J.numpy()[group, row].astype(np.float64)
                np.testing.assert_allclose(J, dense, rtol=3e-6, atol=3e-6)
                actual = J @ velocity[:6]
                for slot, dof in enumerate(candidate.sparse_dof.numpy()[world, row]):
                    if dof >= 0:
                        actual += candidate.sparse_jy.numpy()[world, row, 2 * slot] * velocity[dof]
                self.assertAlmostEqual(actual, dense @ velocity[:6] + sparse @ velocity, delta=1e-5)
                self.assertAlmostEqual(candidate.target.numpy()[world, row], target, delta=3e-6)
                self.assertAlmostEqual(candidate.phi.numpy()[world, row], phi, delta=3e-6)
                L = candidate.factor.numpy()[group].astype(np.float64)
                response = np.linalg.solve(L.T, np.linalg.solve(L, J))
                np.testing.assert_allclose(candidate.Y.numpy()[group, row], response, rtol=5e-6, atol=4e-6)
                diagonal = J @ response + float(candidate.row_cfm.numpy()[world, row])
                for slot, dof in enumerate(candidate.sparse_dof.numpy()[world, row]):
                    if dof >= 0:
                        jy = candidate.sparse_jy.numpy()[world, row].astype(np.float64)
                        diagonal += jy[2 * slot] * jy[2 * slot + 1]
                np.testing.assert_allclose(candidate.diag.numpy()[world, row], diagonal, rtol=5e-6, atol=4e-6)
            sparse_before = candidate.sparse_dof.numpy().copy()
            wp.launch(fused.mark_fallback, dim=16, inputs=[candidate, owner], device="cpu")
            marked = candidate.sparse_dof.numpy()
            np.testing.assert_array_equal(marked[selected], sparse_before[selected])
            expected = sparse_before.copy()
            for contact in range(16):
                if candidate.world.numpy()[contact] != fallback or candidate.slot.numpy()[contact] < 0:
                    continue
                arts = (candidate.art0.numpy()[contact], candidate.art1.numpy()[contact])
                dense = any(a >= 0 and candidate.response_dofs.numpy()[a] not in (0, 108) for a in arts)
                row = candidate.slot.numpy()[contact]
                d0, d1 = expected[fallback, row]
                if not dense and (d0 >= 0) != (d1 >= 0):
                    expected[fallback, row, 1 if d0 >= 0 else 0] = -2
            np.testing.assert_array_equal(marked, expected)

    def test_complete_mode_admission_and_cpu_constructor_fallback(self):
        """Reject incompatible physics, row layout and cold-solve modes without constructing an owner."""
        # This admitted mode owns only a (1, 1) dummy weight. Publishing a
        # canonical row-indexed weight was the first actual GPU failure.
        self.assertNotIn("row_w", fused.BiasData.vars)
        self.assertNotIn("row_w", fused._PREPARE)
        valid = {
            "_compact_contact_boundary": True,
            "dense_max_constraints": 704,
            "max_world_dofs": 114,
            "_sparse_diagonal_response_size": 108,
            "pgs_mode": "matrix_free",
            "pgs_schedule": "interleaved",
            "pgs_velocity_iterations": 0,
            "pgs_iterations": 8,
            "friction_mode": "current",
            "pgs_warmstart": False,
            "_mf_warmstart_enabled": False,
            "_regularization_enabled": False,
            "_contact_w": 1.0,
            "_has_free_rigid_bodies": False,
            "model": SimpleNamespace(requires_grad=False),
        }
        with mock.patch.object(compact, "supported", return_value=True):
            self.assertTrue(fused.supported(SimpleNamespace(**valid)))
            for name, value in (
                ("_compact_contact_boundary", False),
                ("dense_max_constraints", 384),
                ("max_world_dofs", 113),
                ("_sparse_diagonal_response_size", 107),
                ("pgs_mode", "split"),
                ("pgs_schedule", "alternating"),
                ("pgs_velocity_iterations", 1),
                ("pgs_iterations", 0),
                ("friction_mode", "lagged"),
                ("pgs_warmstart", True),
                ("_mf_warmstart_enabled", True),
                ("_regularization_enabled", True),
                ("_contact_w", 0.5),
                ("_has_free_rigid_bodies", True),
                ("model", SimpleNamespace(requires_grad=True)),
            ):
                self.assertFalse(fused.supported(SimpleNamespace(**{**valid, name: value})), name)
        model = build_solver_fixture("cpu")
        with mock.patch.object(solver_module, "_FUSED_CONTACT_SOLVE", True):
            solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="split", dense_max_constraints=704)
        self.assertIsNone(solver._fused_contact_solve)
        self.assertIsNone(solver._fused_contact_solve_active)

    def test_cuda_routing_two_graphs_and_rejected_overflow(self):
        """Replay captured current-count classification through opposite and empty ownership states."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA graph execution requires a CUDA device")
        for device in devices:
            data, aux = routing_fixture(device)
            set_routes(data, aux, triples=(32, 33), dense=(32, 33))
            launch_route(data, aux)
            graphs = []
            for _ in range(2):
                with wp.ScopedCapture(device=device) as capture:
                    launch_route(data, aux)
                graphs.append(capture.graph)
            for prefix, triples, dense, expected in (
                ((0, 0), (33, 32), (33, 32), [0, 1]),
                ((0, 1), (128, 128), (0, 0), [1, 0]),
                ((12, 0), (0, 0), (0, 0), [1, 1]),
            ):
                set_routes(data, aux, prefix=prefix, triples=triples, dense=dense)
                for graph in graphs:
                    aux.owner.fill_(-999)
                    aux.fallback_counts.fill_(-999)
                    wp.capture_launch(graph)
                    np.testing.assert_array_equal(aux.owner.numpy(), expected)
            aux.counts.assign(np.array([705, 706], np.int32))
            wp.capture_launch(graphs[0])
            np.testing.assert_array_equal(aux.owner.numpy(), [0, 0])
            np.testing.assert_array_equal(aux.fallback_counts.numpy(), [704, 704])

    def test_cuda_actual_full_solve_mixed_fallback_reset_notify_and_graphs(self):
        """Compare the complete original eight-sweep law with selected and dense-panel fallback worlds."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("The admitted native solver and event graphs require CUDA")
        for device in devices:
            model = build_solver_fixture(device)
            model.rigid_contact_max = 80
            solvers = []
            for enabled in (False, True):
                with (
                    mock.patch.object(solver_module, "_FUSED_CONTACT_SOLVE", enabled),
                    mock.patch.object(solver_module, "_COMPACT_CONTACT_BOUNDARY", True),
                    mock.patch.object(solver_module, "_PRISMATIC_PUBLICATION", True),
                    mock.patch.object(solver_module, "_SPARSE_CONTACT_DIRECT", True),
                ):
                    solvers.append(
                        newton.solvers.SolverFeatherPGS(
                            model,
                            pgs_mode="matrix_free",
                            pgs_iterations=8,
                            update_mass_matrix_interval=2,
                            use_parallel_streams=True,
                            enable_joint_limits=True,
                            dense_max_constraints=704,
                            mf_max_constraints=704,
                        )
                    )
            self.assertIsNone(solvers[0]._fused_contact_solve)
            self.assertIsNotNone(solvers[1]._fused_contact_solve)
            np.testing.assert_array_equal(solvers[1]._sparse_diagonal_dense_offsets.numpy(), [108, 108])
            weights = []
            for solver in solvers:
                self.assertFalse(solver._regularization_enabled)
                self.assertEqual(solver.row_w.shape, (1, 1))
                self.assertEqual(solver.row_w.ptr, solver._contact_row_w_dummy.ptr)
                weights.append(solver.row_w.numpy().copy())
            states = [[model.state(), model.state()] for _ in solvers]
            controls = [model.control() for _ in solvers]
            contacts = [newton.Contacts(80, 0, device=device) for _ in solvers]
            source = newton.Contacts(8, 0, device=device)
            newton.eval_fk(model, states[0][0].joint_q, states[0][0].joint_qd, states[0][0])
            install_contacts(model, states[0][0], source)
            source_fields = (
                "rigid_contact_shape0",
                "rigid_contact_shape1",
                "rigid_contact_point0",
                "rigid_contact_point1",
                "rigid_contact_margin0",
                "rigid_contact_margin1",
                "rigid_contact_normal",
            )
            original = {name: getattr(source, name).numpy() for name in source_fields}

            def publish(indices, contacts=contacts, source_fields=source_fields, original=original):
                for target in contacts:
                    for name in source_fields:
                        array = getattr(target, name)
                        values = np.zeros(array.numpy().shape, dtype=original[name].dtype)
                        if "shape" in name:
                            values.fill(-1)
                        values[: len(indices)] = original[name][indices]
                        array.assign(values)
                    target.rigid_contact_count.assign(np.array([len(indices)], np.int32))

            def compare(states=states, solvers=solvers, weights=weights):
                for field in ("joint_q", "joint_qd", "body_q", "body_qd"):
                    np.testing.assert_allclose(
                        getattr(states[1][0], field).numpy(),
                        getattr(states[0][0], field).numpy(),
                        rtol=3e-5,
                        atol=5e-6,
                        err_msg=field,
                    )
                for arm, solver in enumerate(solvers):
                    np.testing.assert_array_equal(solver.row_w.numpy(), weights[arm])
                    solver.check_constraint_capacity()
                    self.assertTrue(np.all(np.isfinite(solver.v_out.numpy())))
                    active = solver.constraint_count.numpy()
                    for world, count in enumerate(active):
                        self.assertTrue(np.all(np.isfinite(solver.impulses.numpy()[world, :count])))

            for state in states[1]:
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            for step, indices in enumerate(
                (list(range(6)), [0, 1, 2] + [3] * 33, [], list(range(6)), list(range(6)), list(range(6)))
            ):
                publish(indices)
                if step == 3:
                    model.body_mass.assign(model.body_mass.numpy() * 1.1)
                    for solver in solvers:
                        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
                for index, solver in enumerate(solvers):
                    current, following = states[index]
                    if step == 2:
                        solver.reset(current, wp.array([True, False], dtype=bool, device=device))
                    # Prefix rows intentionally have no sparse coordinates. Their
                    # ignored coefficient storage must not become a NaN operand.
                    solver._sparse_diagonal_row_jy.fill_(float("nan"))
                    solver._sparse_diagonal_row_dof.fill_(-25)
                    solver.step(current, following, controls[index], contacts[index], 1 / 240)
                    states[index] = [following, current]
                compare()
                np.testing.assert_array_equal(
                    solvers[1]._fused_contact_solve.owner.numpy(), [1, 0] if step == 1 else [1, 1]
                )
            graphs = []
            for index, solver in enumerate(solvers):
                current, following = states[index]
                solver.reset(current)
                with wp.ScopedCapture(device=device) as capture:
                    solver.seed_double_buffer_events()
                    solver.step(current, following, controls[index], contacts[index], 1 / 240)
                    solver.step(following, current, controls[index], contacts[index], 1 / 240)
                graphs.append(capture.graph)
            for _ in range(2):
                for graph in graphs:
                    wp.capture_launch(graph)
                compare()
            for index, solver in enumerate(solvers):
                current, following = states[index]
                # Captured stream events cannot be reused by an eager launch.
                solver.seed_double_buffer_events()
                solver.step(current, following, controls[index], None, 1 / 240)
                states[index] = [following, current]
            self.assertIsNone(solvers[1]._fused_contact_solve_active)
            compare()


if __name__ == "__main__":
    unittest.main()
