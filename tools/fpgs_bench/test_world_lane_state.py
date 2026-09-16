# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the complete scalar-world owner against retained physical references."""

import importlib
import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.world_lane_state import WorldLaneState
from tools.fpgs_bench import test_franka_kinetic_state as reference


def bind(snapshot, capture, device="cpu", *, changed=False):
    """Reuse the original physical capture binding and its exact current/held inputs."""
    case = reference.bind_saved(snapshot, capture, device, changed=changed)
    case.owner = WorldLaneState(case.solver)
    solver = case.solver
    solver.mass_update_mask = wp.zeros(case.model.articulation_count, dtype=int, device=device)
    solver.R_by_size = {
        size: wp.array(snapshot[f"solver__R_by_size__{size}"], dtype=float, device=device) for size in (9, 6)
    }
    solver._augmented_drive_row_by_dof = wp.array(
        snapshot["solver___augmented_drive_row_by_dof"], dtype=int, device=device
    )
    solver.aug_row_K = wp.array(snapshot["solver__aug_row_K"], dtype=float, device=device)
    return case


def launch_state(case, *, finish=False, refresh=True):
    """Use the real saved-input ABI without claiming a snapshot namespace is a production solver."""
    owner = case.owner
    kernel = owner.finish_refresh_kernel if refresh else owner.finish_held_kernel
    if not finish:
        kernel = owner.repair_kernel
    output = case.output if finish else case.state
    wp.launch(
        kernel,
        dim=case.model.world_count,
        inputs=[
            owner.plan,
            owner._publication(case.state, case.solver, output, case.dt),
            owner.data,
            case.solver._mass_update_requested,
            int(refresh),
        ],
        block_dim=128,
        device=case.device,
    )


def launch_predict(case):
    """Bind current forces/factors directly for the same saved-input native ABI."""
    owner = case.owner
    force, factor = owner._predictor_inputs(case.state, case.solver, case.control, case.state.joint_qd, case.dt)
    wp.launch(
        owner.predictor_kernel,
        dim=case.model.world_count,
        inputs=[owner.plan, owner.data, force, factor],
        block_dim=128,
        device=case.device,
    )


def geometry(case):
    """Decode only the candidate's compact symmetric private representation."""
    lower = case.owner.data.geometric.numpy().T
    result = np.zeros((case.model.world_count, 9, 9), dtype=np.float64)
    for row in range(9):
        for col in range(row + 1):
            result[:, row, col] = result[:, col, row] = lower[:, row * (row + 1) // 2 + col]
    return result


def check_current(test, case, *, refresh=True):
    """Check current body services and dynamics against the inherited FP64 oracle."""
    expected = reference.physical_reference(case, case.state)
    ids = case.owner.host_plan["body_ids"][:, :13]
    reference.scaled(test, case.state.body_q.numpy()[ids], expected["body_q"], name="world-lane body pose")
    for half in (slice(0, 3), slice(3, 6)):
        reference.scaled(
            test,
            case.state.body_qd.numpy()[ids][..., half],
            expected["public"][..., half],
            name="world-lane public velocity",
        )
    reference.scaled(test, case.owner.bias.numpy().T, expected["bias"], name="world-lane bias")
    if refresh:
        reference.scaled(test, geometry(case), expected["H"], name="world-lane physical H")
    free = np.zeros((case.model.world_count, 6, 6))
    free[:, :3, :3] = expected["mass"][:, 11, None, None] * np.eye(3)
    free[:, 3:, 3:] = expected["inertia"][:, 11]
    reference.scaled(test, case.solver.body_I_s.numpy()[ids[:, 11]], free, name="world-lane dynamic free inertia")
    reference.scaled(
        test, case.solver.body_v_s.numpy()[ids[:, 12]], expected["body_v_s"][:, 12], name="world-lane prescribed twist"
    )
    case.owner.check()
    return expected


class TestWorldLaneStateCPU(unittest.TestCase):
    def test_owner_api(self):
        """Require an explicit complete owner without changing the retained p16 factories."""
        module = importlib.import_module("newton._src.solvers.feather_pgs.world_lane_state")
        for name in ("WorldLaneState", "build_schedule", "get_state_kernel", "get_predictor_kernel"):
            self.assertTrue(hasattr(module, name), name)
        for name in ("begin", "predict", "finish", "invalidate", "validate_model"):
            self.assertTrue(callable(getattr(module.WorldLaneState, name)), name)

    def test_current_and_finish_physical_reference(self):
        """Preserve current geometry and original integration across held/refresh publication."""
        capture = next(reference.captures())
        with np.load(capture["path"], allow_pickle=False) as snapshot:
            case = bind(snapshot, capture, changed=True)
            original = reference.bind_saved(snapshot, capture, "cpu", changed=True)
        launch_state(case)
        check_current(self, case)
        held_geometry = case.owner.data.geometric.numpy().copy()
        reference.reference_module().original(reference.publication_values(original, original.output))
        launch_state(case, finish=True, refresh=False)
        for name in ("joint_q", "joint_qd"):
            reference.scaled(
                self,
                getattr(case.output, name).numpy().reshape(case.model.world_count, -1),
                getattr(original.output, name).numpy().reshape(case.model.world_count, -1),
                name="original " + name,
            )
        np.testing.assert_array_equal(case.owner.data.geometric.numpy(), held_geometry)
        case.state = case.output
        check_current(self, case, refresh=False)
        launch_state(case)
        check_current(self, case)
        reference.report_errors(self)

    def test_current_factor_and_held_force_reference(self):
        """Use original held-factor momentum gates and independent current mass/drive assembly."""
        capture = next(reference.captures())
        with np.load(capture["path"], allow_pickle=False) as snapshot:
            case = bind(snapshot, capture)
        launch_state(case)
        physical = check_current(self, case)
        original_launch = wp.launch_tiled

        def launch_predictor(kernel, *args, **kwargs):
            if kernel is case.owner.predictor_kernel:
                launch_predict(case)
                return None
            return original_launch(kernel, *args, **kwargs)

        with patch.object(wp, "launch_tiled", side_effect=launch_predictor):
            reference.check_predictor(self, case, physical)
        case.solver.mass_update_mask.fill_(1)
        launch_predict(case)
        reference.check_factor(self, case)
        case.solver.mass_update_mask.zero_()
        with patch.object(wp, "launch_tiled", side_effect=launch_predictor):
            reference.check_predictor(self, case, physical)
        reference.report_errors(self)

    def test_masked_repair_and_structural_notification(self):
        """Preserve per-world reset validity and reject changed generated topology inputs."""
        capture = next(reference.captures())
        with np.load(capture["path"], allow_pickle=False) as snapshot:
            case = bind(snapshot, capture)
        owner = case.owner
        owner.begin(case.state, case.solver, case.dt, True)
        generation = owner.data.generation.numpy().copy()
        selected = np.arange(case.model.world_count) % 3 == 1
        mask = wp.array(selected, dtype=wp.bool, device=case.device)
        owner.invalidate(mask)
        arts = owner.host_plan["arts"]
        np.testing.assert_array_equal(
            owner.data.current_valid.numpy()[arts], np.repeat((~selected)[:, None], 3, axis=1)
        )
        q = case.state.joint_q.numpy().copy()
        q[owner.host_plan["q_index"][selected, 0]] += np.float32(0.001)
        case.state.joint_q.assign(q)
        owner.begin(case.state, case.solver, case.dt, False)
        np.testing.assert_array_equal(owner.data.generation.numpy(), generation + selected)
        check_current(self, case, refresh=False)
        owner.validate_model(newton.ModelFlags.JOINT_PROPERTIES)
        axes = case.model.joint_axis.numpy().copy()
        changed = axes.copy()
        changed[0, 0] += np.float32(0.01)
        case.model.joint_axis.assign(changed)
        with self.assertRaisesRegex(RuntimeError, "joint_axis changed"):
            owner.validate_model(newton.ModelFlags.JOINT_PROPERTIES)
        case.model.joint_axis.assign(axes)

    def test_unsupported_cpu_constructor_falls_back(self):
        """Keep the default and unsupported CPU production graph on the original owner."""
        model = reference.model_fixture("cpu", worlds=1)
        with patch.dict(os.environ, {"FEATHER_PGS_WORLD_LANE_STATE": "1"}):
            solver = reference.make_solver(model, False)
        self.assertIsNone(solver._world_lane_state)
        self.assertIsNone(solver._franka_kinetic_state)


class TestWorldLaneStateCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Use only the CUDA device explicitly assigned by the root launcher."""
        wp.init()
        if not wp.get_cuda_devices():
            raise unittest.SkipTest("CUDA unavailable")
        cls.device = wp.get_device("cuda:0")

    def test_saved_current_factor_held_force_and_finish(self):
        """Cover saved current/held banks with the unchanged independent physical gates."""
        for capture in reference.captures():
            with self.subTest(path=capture["path"]):
                with np.load(capture["path"], allow_pickle=False) as snapshot:
                    case = bind(snapshot, capture, self.device)
                    original = reference.bind_saved(snapshot, capture, self.device)
                launch_state(case)
                physical = check_current(self, case)
                original_launch = wp.launch_tiled

                def launch_predictor(kernel, *args, case=case, original_launch=original_launch, **kwargs):
                    if kernel is case.owner.predictor_kernel:
                        launch_predict(case)
                        return None
                    return original_launch(kernel, *args, **kwargs)

                with patch.object(wp, "launch_tiled", side_effect=launch_predictor):
                    reference.check_predictor(self, case, physical)
                case.solver.mass_update_mask.fill_(1)
                launch_predict(case)
                reference.check_factor(self, case)
                reference.reference_module().original(reference.publication_values(original, original.output))
                launch_state(case, finish=True, refresh=True)
                for name in ("joint_q", "joint_qd"):
                    reference.scaled(
                        self,
                        getattr(case.output, name).numpy().reshape(case.model.world_count, -1),
                        getattr(original.output, name).numpy().reshape(case.model.world_count, -1),
                        name="saved original " + name,
                    )
                case.state = case.output
                check_current(self, case)
        reference.report_errors(self)

    def test_actual_p16_control_loaded_reset_notification_and_graph(self):
        """Compare complete five-world steps to p16, including live resets and captured replay."""
        model = reference.model_fixture(self.device, worlds=5)
        solvers = []
        for enabled in (False, True):
            with patch.dict(os.environ, {"FEATHER_PGS_WORLD_LANE_STATE": str(int(enabled))}):
                solvers.append(reference.make_solver(model, True))
        original, candidate = solvers
        self.addCleanup(lambda: wp.synchronize_device(self.device))
        self.assertIsNone(original._world_lane_state)
        self.assertIsNotNone(original._franka_kinetic_state)
        self.assertIs(candidate._world_lane_state, candidate._franka_kinetic_state)
        states = [[model.state(), model.state()] for _ in solvers]
        controls = [model.control() for _ in solvers]
        for pair, control in zip(states, controls, strict=True):
            pair[0].joint_qd.assign(np.linspace(-0.12, 0.17, model.joint_dof_count, dtype=np.float32))
            pair[0].body_f.assign(np.random.default_rng(817).normal(0, 0.3, (model.body_count, 6)).astype(np.float32))
            control.joint_f.assign(np.linspace(-0.1, 0.15, model.joint_dof_count, dtype=np.float32))
            newton.eval_fk(model, pair[0].joint_q, pair[0].joint_qd, pair[0])
        contacts = reference.loaded_contacts(model, states[0][0])
        seen = []
        original_launch = wp.launch

        def launch(*args, **kwargs):
            kernel = kwargs.get("kernel", args[0] if args else None)
            seen.append(kernel.key)
            return original_launch(*args, **kwargs)

        def complete(source, destination, refresh):
            for index, (solver, pair, control) in enumerate(zip(solvers, states, controls, strict=True)):
                inputs = {
                    name: getattr(pair[source], name).numpy().copy() for name in ("joint_q", "joint_qd", "body_f")
                }
                if index:
                    with patch.object(wp, "launch", side_effect=launch):
                        solver.step(pair[source], pair[destination], control, contacts, reference.DT)
                else:
                    solver.step(pair[source], pair[destination], control, contacts, reference.DT)
                solver.check_constraint_capacity()
                np.testing.assert_array_equal(solver._mass_update_requested.numpy(), [0])
                np.testing.assert_array_equal(
                    solver.mass_update_mask.numpy(), np.full(model.articulation_count, int(refresh))
                )
                for name, value in inputs.items():
                    np.testing.assert_array_equal(getattr(pair[source], name).numpy(), value)
            if refresh:
                reference.check_factor(
                    self, reference.actual_case(candidate, states[1][source], states[1][destination])
                )
            current = reference.actual_case(candidate, states[1][destination], states[1][source])
            check_current(self, current, refresh=bool(np.all(current.owner.geometry_valid.numpy())))
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                reference.scaled(
                    self,
                    getattr(states[1][destination], name).numpy().reshape(model.world_count, -1),
                    getattr(states[0][destination], name).numpy().reshape(model.world_count, -1),
                    tolerance=7e-4,
                    name="p16 matched8 " + name,
                )

        complete(0, 1, True)
        held = {size: candidate.L_by_size[size].numpy().copy() for size in (9, 6)}
        complete(1, 0, False)
        for size in (9, 6):
            np.testing.assert_array_equal(candidate.L_by_size[size].numpy(), held[size])
        selected = np.array([False, True, True, False, True])
        mask = wp.array(selected, dtype=wp.bool, device=self.device)
        for solver, pair in zip(solvers, states, strict=True):
            q = pair[0].joint_q.numpy().copy()
            q.reshape(model.world_count, -1)[selected, :9] += np.float32(0.001)
            pair[0].joint_q.assign(q)
            solver.reset(pair[0], mask)
            solver._step, solver._force_mass_update = 3, False
            solver._mass_update_requested.fill_(1)
        complete(0, 1, True)
        model.body_mass.assign(model.body_mass.numpy() * np.float32(1.01))
        model.body_inertia.assign(model.body_inertia.numpy() * np.float32(1.02))
        for solver, control in zip(solvers, controls, strict=True):
            solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
            solver._step, solver._force_mass_update = 5, False
            control.joint_f.assign(-control.joint_f.numpy())
        complete(1, 0, True)
        self.assertTrue(any("world_lane_factor_predict" in key for key in seen))
        self.assertTrue(any("world_lane_finish_held" in key for key in seen))
        self.assertTrue(any("world_lane_finish_refresh" in key for key in seen))
        for retired in (
            "crba_cholesky",
            "franka_kinetic",
            "compute_composite_inertia",
            "integrate_generalized_joints",
            "compute_velocity_predictor",
        ):
            self.assertFalse(any(retired in key for key in seen), retired)
        candidate._step = 8
        contacts.rigid_contact_count.zero_()
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            candidate.seed_double_buffer_events()
            candidate.step(states[1][0], states[1][1], controls[1], contacts, reference.DT)
            candidate.step(states[1][1], states[1][0], controls[1], contacts, reference.DT)
        for count in (0, 15, 0, 15):
            candidate.reset(states[1][0], mask)
            candidate._mass_update_requested.fill_(1)
            contacts.rigid_contact_count.fill_(count)
            wp.capture_launch(capture.graph)
            candidate.check_constraint_capacity()
            np.testing.assert_array_equal(candidate._mass_update_requested.numpy(), [0])
            check_current(self, reference.actual_case(candidate, states[1][0], states[1][1]))
        owner, state = candidate._world_lane_state, states[1][0]
        candidate.mass_update_mask.zero_()
        force = state.body_f.numpy().copy()
        invalid = force.copy()
        invalid[owner.host_plan["body_ids"][1, 0], 0] = np.nan
        state.body_f.assign(invalid)
        candidate.v_hat.fill_(123.0)
        owner.predict(state, candidate, controls[1], state.joint_qd, reference.DT)
        np.testing.assert_array_equal(owner.status.numpy(), [0, 2, 0, 0, 0])
        np.testing.assert_array_equal(candidate.v_hat.numpy()[owner.host_plan["dof_ids"][1]], 123.0)
        state.body_f.assign(force)
        owner.status.zero_()
        reference.report_errors(self)


if __name__ == "__main__":
    unittest.main()
