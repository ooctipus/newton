# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Current-input controls for the integrated local prefix/packet boundary."""

import importlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import franka_row_packets as candidate
from newton._src.solvers.feather_pgs.franka_row_packets import (
    LocalRowPackets,
    QueueData,
    topology_signature,
    validate_prefix_topology,
)

CAPTURES = Path("/tmp/fpgs-franka-current-rows-paired512-20260912-01")


def bind_prefix(snapshot, settings, device="cpu"):
    """Bind only current prefix operands and private output buffers from a capture."""
    module = importlib.import_module("newton._src.solvers.feather_pgs.franka_row_packets")
    data = module.PrefixData()
    bindings = {
        "group_to_art": "solver__group_to_art__9",
        "art_to_world": "solver__art_to_world",
        "dof_start": "solver__articulation_dof_start",
        "mimic_start": "solver___mimic_art_start",
        "mimic_list": "solver___mimic_art_list",
        "mimic_valid": "solver___mimic_valid",
        "mimic_enabled": "model__constraint_mimic_enabled",
        "mimic_dof0": "solver___mimic_dof0",
        "mimic_dof1": "solver___mimic_dof1",
        "mimic_q0": "solver___mimic_q0",
        "mimic_q1": "solver___mimic_q1",
        "mimic_coef0": "model__constraint_mimic_coef0",
        "mimic_coef1": "model__constraint_mimic_coef1",
        "limit_q_index": "solver___joint_limit_q_index",
        "lower": "model__joint_limit_lower",
        "upper": "model__joint_limit_upper",
        "q": "state_in__joint_q",
    }
    float_names = {"mimic_coef0", "mimic_coef1", "lower", "upper", "q"}
    for name, key in bindings.items():
        dtype = wp.bool if name == "mimic_enabled" else float if name in float_names else int
        setattr(data, name, wp.array(snapshot[key], dtype=dtype, device=device))
    worlds, rows = snapshot["solver__row_type"].shape
    data.activation_gap = settings["joint_limit_activation_gap"]
    data.beta, data.cfm, data.dt = settings["pgs_beta"], settings["pgs_cfm"], 1 / 240
    data.bias_scale, data.limit_speculative_scale = 1.0, 1.0
    data.dofs = wp.full((worlds, 20, 2), -99, dtype=int, device=device)
    data.weights = wp.full((worlds, 20, 2), 9876.0, dtype=float, device=device)
    data.row_contact = wp.full((worlds, 40), 97, dtype=int, device=device)
    data.mimic_slot = wp.full(len(snapshot["solver__mimic_slot"]), -97, dtype=int, device=device)
    data.slot_counter = wp.full(worlds, 91, dtype=int, device=device)
    data.bounds = wp.full((worlds, 2), 95, dtype=int, device=device)
    for name in ("kind", "parent"):
        setattr(data, name, wp.full((worlds, rows), -97, dtype=int, device=device))
    for name in ("mu", "row_beta", "row_cfm", "phi", "target", "restitution", "rhs", "diag", "impulses"):
        setattr(data, name, wp.full((worlds, rows), 9876.0, dtype=float, device=device))
    return data


def check_prefix(snapshot, data):
    """Compare direct current prefix to actual original count, row laws and Jacobian."""
    bounds = snapshot["solver__dense_phase_bounds"]
    np.testing.assert_array_equal(data.bounds.numpy(), bounds)
    np.testing.assert_array_equal(data.slot_counter.numpy(), bounds[:, 0])
    np.testing.assert_array_equal(data.mimic_slot.numpy(), snapshot["solver__mimic_slot"])
    np.testing.assert_array_equal(data.row_contact.numpy(), -1)
    dofs, weights = data.dofs.numpy(), data.weights.numpy()
    rows = np.arange(snapshot["solver__row_type"].shape[1])[None, :]
    prefix = rows < bounds[:, :1]
    for name, key in (("kind", "row_type"), ("parent", "row_parent"), ("phi", "phi"), ("rhs", "rhs")):
        actual = getattr(data, name).numpy()
        expected = snapshot["solver__" + key]
        np.testing.assert_allclose(actual[prefix], expected[prefix], rtol=3e-6, atol=2e-6)
        np.testing.assert_array_equal(actual[~prefix], -97 if name in ("kind", "parent") else 9876)
    groups = snapshot["solver__group_to_art__9"]
    worlds = snapshot["solver__art_to_world"][groups]
    original = snapshot["solver__J_by_size__9"]
    for group, world in enumerate(worlds):
        count = bounds[world, 0]
        matrix = np.zeros((count, 9), np.float32)
        for row in range(count):
            for component in (0, 1):
                dof = dofs[world, row, component]
                if dof >= 0:
                    matrix[row, dof] += weights[world, row, component]
        np.testing.assert_array_equal(matrix, original[group, :count])
    np.testing.assert_array_equal(data.impulses.numpy()[prefix], 0)


def bind_queue(snapshot, device="cpu"):
    """Bind current allocator results and private queue outputs from one checkpoint."""
    queue = QueueData()
    names = {
        "primary": "_local_primary_articulation",
        "secondary": "_local_pair_articulation",
        "residual_secondary": "_local_residual_pair_articulation",
        "mf_count": "mf_constraint_count",
        "mf_body0": "mf_body_a",
        "mf_body1": "mf_body_b",
        "body_art": "body_to_articulation",
    }
    for name, key in names.items():
        setattr(queue, name, wp.array(snapshot["solver__" + key], dtype=int, device=device))
    worlds = len(snapshot["solver__constraint_count"])
    queue.status = wp.zeros(3, dtype=int, device=device)
    for name in ("count", "owner", "general", "pair", "pair_secondary", "residual", "residual_pair"):
        setattr(queue, name, wp.full(worlds, -19, dtype=int, device=device))
    for name in ("general_count", "pair_count", "residual_count"):
        setattr(queue, name, wp.zeros(1, dtype=int, device=device))
    return queue


class TestFrankaRowPackets(unittest.TestCase):
    @unittest.skipUnless(CAPTURES.exists(), "Local saved Franka input fixture")
    def test_actual_current_prefix_all_captures(self):
        """Match dominant mimic/limit rows in both GPUs' refresh and reuse captures."""
        module = importlib.import_module("newton._src.solvers.feather_pgs.franka_row_packets")
        for gpu in (0, 1):
            directory = CAPTURES / f"gpu{gpu}"
            audit = json.loads((directory / "audit.json").read_text())
            self.assertTrue(audit["complete"] and audit["success"] and audit["source_guard_pass"])
            for capture in audit["captures"]:
                with self.subTest(gpu=gpu, step=capture["solver_step"]), np.load(capture["path"]) as snapshot:
                    data = bind_prefix(snapshot, capture["settings"]["solver"])
                    wp.launch_tiled(module.get_prefix_kernel(), dim=[128], inputs=[data], block_dim=128, device="cpu")
                    check_prefix(snapshot, data)

    def test_prefix_owner_exists(self):
        """Require a direct prefix owner instead of the original full Jacobian passes."""
        module = importlib.import_module("newton._src.solvers.feather_pgs.franka_row_packets")
        self.assertIsNotNone(module.get_prefix_kernel())

    def test_exact_queue_thresholds_and_sticky_overflow(self):
        """Keep 9/20/40-row and 12-MF bounds, empty worlds and raw overflow distinct."""
        prefix = candidate.PrefixData()
        prefix.kind = wp.zeros((8, 192), dtype=int, device="cpu")
        prefix.slot_counter = wp.array([0, 9, 10, 20, 21, 40, 41, 193], dtype=int, device="cpu")
        prefix.bounds = wp.array([[0, 0], [9, 9], [10, 10], *([[6, 6]] * 5)], dtype=int, device="cpu")
        queue = QueueData()
        for name in ("primary", "secondary", "residual_secondary"):
            setattr(queue, name, wp.full(8, 0 if name == "primary" else 1, dtype=int, device="cpu"))
        queue.mf_count = wp.array([0, 0, 0, 0, 0, 12, 0, 0], dtype=int, device="cpu")
        queue.mf_body0 = wp.full((8, 64), 0, dtype=int, device="cpu")
        queue.mf_body1 = wp.full((8, 64), -1, dtype=int, device="cpu")
        queue.body_art = wp.array([1, 2], dtype=int, device="cpu")
        queue.status = wp.zeros(3, dtype=int, device="cpu")
        for name in ("count", "owner", "general", "pair", "pair_secondary", "residual", "residual_pair"):
            setattr(queue, name, wp.full(8, -97, dtype=int, device="cpu"))
        for name in ("general_count", "pair_count", "residual_count"):
            setattr(queue, name, wp.zeros(1, dtype=int, device="cpu"))
        wp.launch(candidate.finalize_and_queue, dim=8, inputs=[prefix, queue], device="cpu")
        np.testing.assert_array_equal(queue.count.numpy(), [0, 9, 10, 20, 21, 40, 41, 192])
        np.testing.assert_array_equal(queue.owner.numpy(), [0, 1, 0, 2, 3, 3, 0, 0])
        np.testing.assert_array_equal(queue.general.numpy()[:3], [2, 6, 7])
        np.testing.assert_array_equal(queue.status.numpy(), [1, 0, 0])
        # A foreign MF endpoint forces complete fallback; overflow remains sticky.
        queue.mf_body0.fill_(1)
        prefix.slot_counter.assign(np.array([0, 9, 10, 20, 21, 40, 41, 40], np.int32))
        for name in ("general_count", "pair_count", "residual_count"):
            getattr(queue, name).zero_()
        wp.launch(candidate.finalize_and_queue, dim=8, inputs=[prefix, queue], device="cpu")
        self.assertEqual(queue.owner.numpy()[5], 0)
        np.testing.assert_array_equal(queue.status.numpy(), [1, 0, 0])

    def test_total_mimic_entries_guard(self):
        """A disabled leading entry cannot hide a third valid mimic from admission."""
        base = [np.array([0]), np.array([0]), np.array([0, 9])]
        self.assertTrue(
            validate_prefix_topology(*base, np.array([0, 2]), np.arange(2), np.array([7, 8]), np.array([8, 7]))
        )
        self.assertFalse(
            validate_prefix_topology(*base, np.array([0, 3]), np.arange(3), np.array([7, 8, 7]), np.array([8, 7, 8]))
        )
        self.assertFalse(
            validate_prefix_topology(
                *base, np.array([0, 2]), np.arange(2), np.array([7, 8]), np.array([8, 7]), np.array([0, 1])
            )
        )

    @unittest.skipUnless(CAPTURES.exists(), "Local saved Franka input fixture")
    def test_constructor_guard_before_late_fields(self):
        """Admit current mode before late compact initialization, rejecting missing readers."""
        audit = json.loads((CAPTURES / "gpu0/audit.json").read_text())
        solver = SimpleNamespace(**audit["captures"][0]["settings"]["solver"])
        solver.model = SimpleNamespace(device=SimpleNamespace(is_cuda=True), requires_grad=False)
        solver.size_groups = [6, 9]
        solver._execution_plan = SimpleNamespace(
            use_tiled_hinv_jt=lambda size: False, use_diagonal_mass=lambda size: False
        )
        del solver._compact_contact_boundary
        sentinel = object()
        with (
            patch.object(candidate, "LocalRowPackets", return_value=sentinel),
            patch.object(candidate, "topology_supported", return_value=True),
        ):
            self.assertIs(candidate.create_owner(solver), sentinel)
            for name in ("_mf_warmstart_enabled", "_debug_buffers_enabled", "_regularization_enabled"):
                old = getattr(solver, name)
                setattr(solver, name, True)
                self.assertIsNone(candidate.create_owner(solver))
                setattr(solver, name, old)
            solver.model.requires_grad = True
            self.assertIsNone(candidate.create_owner(solver))

    @unittest.skipUnless(CAPTURES.exists(), "Local saved Franka input fixture")
    def test_owner_allocation_does_not_read_late_status(self):
        """Allocate the real owner using precisely the pre-tiled constructor fields."""
        with np.load(CAPTURES / "gpu0/rows/step1600_capture0.npz") as snapshot:
            model = SimpleNamespace(device=wp.get_device("cpu"))
            for name in (
                "constraint_mimic_joint0",
                "constraint_mimic_joint1",
                "constraint_mimic_world",
                "joint_type",
                "joint_parent",
                "joint_child",
                "joint_q_start",
                "joint_qd_start",
                "body_flags",
            ):
                setattr(model, name, wp.array(snapshot["model__" + name], dtype=int, device="cpu"))
            solver = SimpleNamespace(model=model, world_count=512)
            for name in (
                "_local_primary_articulation",
                "_local_pair_articulation",
                "_local_residual_pair_articulation",
                "mf_constraint_count",
                "mf_body_a",
                "mf_body_b",
                "body_to_articulation",
                "constraint_count",
                "_local_solve_owner",
                "_local_general_world_count",
                "_local_general_worlds",
            ):
                setattr(solver, name, wp.array(snapshot["solver__" + name], dtype=int, device="cpu"))
            for name in (
                "_local_pair_active_counts",
                "_local_pair_active_candidates",
                "_local_pair_active_secondaries",
                "_local_residual_active_counts",
                "_local_residual_active_candidates",
                "_local_residual_active_secondaries",
            ):
                setattr(solver, name, {9: wp.array(snapshot["solver__" + name + "__9"], dtype=int, device="cpu")})
            self.assertFalse(hasattr(solver, "_constraint_capacity_status"))
            owner = LocalRowPackets(solver)
            self.assertEqual(owner.packet.jacobian.shape, (512, 40, 15))
            self.assertIsNone(owner.queue.status)

    def test_structural_notification_requires_reconstruction(self):
        """A topology edit is rejected before stale private mappings can be captured."""
        model = SimpleNamespace(
            **{
                name: wp.array([0, 1], dtype=int, device="cpu")
                for name in (
                    "constraint_mimic_joint0",
                    "constraint_mimic_joint1",
                    "constraint_mimic_world",
                    "joint_type",
                    "joint_parent",
                    "joint_child",
                    "joint_q_start",
                    "joint_qd_start",
                    "body_flags",
                )
            }
        )
        owner = object.__new__(LocalRowPackets)
        owner.solver = SimpleNamespace(model=model)
        owner._topology_signature = topology_signature(owner.solver)
        model.constraint_mimic_joint0.fill_(9)
        with self.assertRaisesRegex(RuntimeError, "reconstruct"):
            owner.validate_notification()

    @unittest.skipUnless(CAPTURES.exists(), "Local saved Franka input fixture")
    def test_disabled_first_mimic_stays_current(self):
        """Disabled first entries compact safely while the second remains active."""
        module = candidate

        audit = json.loads((CAPTURES / "gpu0/audit.json").read_text())
        capture = audit["captures"][0]
        with np.load(capture["path"]) as snapshot:
            data = bind_prefix(snapshot, capture["settings"]["solver"])
            enabled = data.mimic_enabled.numpy()
            enabled[0] = False
            data.mimic_enabled = wp.array(enabled, dtype=wp.bool, device="cpu")
            wp.launch_tiled(module.get_prefix_kernel(), dim=[128], inputs=[data], block_dim=128, device="cpu")
            self.assertEqual(data.mimic_slot.numpy()[0], -1)
            self.assertEqual(data.mimic_slot.numpy()[1], 0)
            self.assertEqual(data.bounds.numpy()[0, 0], snapshot["solver__dense_phase_bounds"][0, 0] - 1)
            np.testing.assert_array_equal(data.dofs.numpy()[0, 0], [7, 8])

    @unittest.skipUnless(CAPTURES.exists(), "Local saved Franka input fixture")
    def test_fused_current_classification_and_queues(self):
        """Match all original owners and queue sets at both refresh/reuse boundaries."""
        module = candidate

        for gpu in (0, 1):
            audit = json.loads((CAPTURES / f"gpu{gpu}/audit.json").read_text())
            for capture in audit["captures"]:
                with self.subTest(gpu=gpu, step=capture["solver_step"]), np.load(capture["path"]) as snapshot:
                    prefix = bind_prefix(snapshot, capture["settings"]["solver"])
                    prefix.slot_counter = wp.array(snapshot["solver__slot_counter"], dtype=int, device="cpu")
                    prefix.bounds = wp.array(snapshot["solver__dense_phase_bounds"], dtype=int, device="cpu")
                    queue = bind_queue(snapshot)
                    wp.launch(module.finalize_and_queue, dim=512, inputs=[prefix, queue], device="cpu")
                    np.testing.assert_array_equal(queue.count.numpy(), snapshot["solver__constraint_count"])
                    np.testing.assert_array_equal(queue.owner.numpy(), snapshot["solver___local_solve_owner"])
                    np.testing.assert_array_equal(queue.status.numpy(), 0)
                    for name, tier in (("pair", "pair"), ("residual", "residual")):
                        count = int(getattr(queue, name + "_count").numpy()[0])
                        expected_count = int(snapshot[f"solver___local_{tier}_active_counts__9"][0])
                        self.assertEqual(count, expected_count)
                        np.testing.assert_array_equal(
                            np.sort(getattr(queue, name).numpy()[:count]),
                            np.sort(snapshot[f"solver___local_{tier}_active_candidates__9"][:count]),
                        )


if __name__ == "__main__":
    unittest.main()
