# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check ordered contact transactions against physical sparse PGS references."""

import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT as CONTACT,
)
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_FRICTION as FRICTION,
)
from newton._src.solvers.feather_pgs.sparse_mass_matrix import _SparseRowSupport
from newton._src.solvers.feather_pgs.sparse_pgs import _get_pgs_solve_sparse_kernel
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_sparse_pgs import _problem, _sweep
from newton.tests.test_feather_pgs_sparse_research_admission import _branched_model

CANDIDATE_OPTIONS = {"contact_triplets": True}


def factory(capacity, dofs, width, *, candidate=True):
    return _get_pgs_solve_sparse_kernel(
        capacity,
        dofs,
        width,
        active_rows=True,
        use_support_table=True,
        **(CANDIDATE_OPTIONS if candidate else {}),
    )


def fixture(*, offset=0, activity=None, seeded=False, omega=1.0, delay=0, iteration_offset=0, linked=False):
    """Retain the independent physical-coordinate oracle, including its factor."""
    dofs, width = (3, 3) if activity is not None else (18, 12)
    factor, indices, z, initial, bias, kinds, parents, mu, impulses, jacobian, response, incident, diag = _problem(
        dofs,
        width,
        seeded,
        contact_activity=activity,
    )
    if not linked and activity != "linked_load":
        parents[kinds == CONTACT] = -1
    if offset:
        z = np.pad(z, ((offset, 0), (0, 0)))
        jacobian = np.pad(jacobian, ((offset, 0), (0, 0)))
        response = np.pad(response, ((offset, 0), (0, 0)))
        bias = np.pad(bias, (offset, 0), constant_values=1.0)
        incident = np.pad(incident, (offset, 0))
        diag = np.pad(diag, (offset, 0), constant_values=1.0)
        kinds = np.pad(kinds, (offset, 0), constant_values=CONTACT)
        parents = np.pad(np.where(parents >= 0, parents + offset, -1), (offset, 0), constant_values=-1)
        mu, impulses = np.pad(mu, (offset, 0)), np.pad(impulses, (offset, 0))
    count, capacity, worlds = len(kinds), 192, 3
    assert count <= capacity

    def padded(value, dtype=np.float32):
        result = np.zeros((worlds, capacity, *value.shape[1:]), dtype=dtype)
        result[:, : len(value)] = value
        return result

    row_factor = np.zeros((worlds, capacity, width), dtype=np.float32)
    row_factor[:, :count] = z[:, indices]
    data = {
        "world_constraint_count": np.array([count, 0, count], np.int32),
        "rhs_bias": padded(bias),
        "diagonal": padded(diag),
        "initial_impulses": padded(impulses),
        "row_factor": row_factor,
        "row_incident": padded(incident),
        "row_type": padded(kinds, np.int32),
        "row_parent": padded(parents, np.int32),
        "row_mu": padded(mu),
        "support_row_class": np.zeros((worlds, capacity), np.int32),
        "support_class_dof": indices[None].astype(np.int32),
        "iterations": np.array(8),
        "omega": np.array(omega),
        "friction_start_iteration": np.array(delay),
        "iteration_offset": np.array(iteration_offset),
        "reference_factor_delta": np.zeros((worlds, dofs), np.float32),
    }
    expected_velocity, expected_impulse = _sweep(
        jacobian,
        response,
        initial,
        bias,
        diag,
        kinds,
        parents,
        mu,
        impulses,
        omega=omega,
        friction_start=delay,
        iteration_offset=iteration_offset,
        active_rows=True,
        iterations=8,
    )
    physical = {
        "factor": factor,
        "initial": initial,
        "jacobian": jacobian,
        "response": response,
        "expected_velocity": expected_velocity,
        "expected_impulse": expected_impulse,
        "initial_impulse": impulses,
    }
    return data, physical


class DeviceSolve:
    """One ABI adapter shared by synthetic and preserved actual-input tests."""

    def __init__(self, data, *, candidate, device="cuda:0"):
        self.device = wp.get_device(device)
        self.data = data
        self.worlds, self.capacity, self.width = data["row_factor"].shape
        self.dofs = data["reference_factor_delta"].shape[1]
        self.kernel = factory(self.capacity, self.dofs, self.width, candidate=candidate)

        def array(value, dtype=float):
            return wp.array(value, dtype=dtype, device=self.device)

        def zeros(shape, dtype=float):
            return wp.zeros(shape, dtype=dtype, device=self.device)

        self.support = _SparseRowSupport()
        self.support.enabled = 1
        self.support.row_class = array(data["support_row_class"], int)
        self.support.class_dof = array(data["support_class_dof"], int)
        self.impulses = array(data["initial_impulses"])
        self.delta = wp.full((self.worlds, self.dofs), float("nan"), device=self.device)
        self.inputs = [
            self.worlds,
            array(data["world_constraint_count"], int),
            array(data["rhs_bias"]),
            array(data["diagonal"]),
            self.impulses,
            wp.full((1, 1, 1), -9, dtype=int, device=self.device),
            array(data["row_factor"]),
            zeros((self.worlds, self.capacity, 0)),
            array(data["row_incident"]),
            array(data["row_type"], int),
            array(data["row_parent"], int),
            array(data["row_mu"]),
            zeros((self.worlds, self.dofs), int),
            array(np.arange(self.worlds * self.dofs).reshape(self.worlds, self.dofs), int),
            zeros(self.worlds * self.dofs),
            zeros(self.worlds, int),
            zeros((self.worlds, 4), int),
            zeros((self.worlds, 1)),
            zeros((self.worlds, 1, 6)),
            zeros((self.worlds, 1, 6)),
            zeros((self.worlds, 1, 6)),
            zeros((self.worlds, 1, 6)),
            zeros((self.worlds, 1)),
            int(data["iterations"]),
            float(data["omega"]),
            int(data["friction_start_iteration"]),
            int(data["iteration_offset"]),
        ]

    def launch(self):
        wp.launch_tiled(
            self.kernel,
            dim=[(self.worlds + 1) // 2],
            inputs=self.inputs,
            outputs=[self.delta, self.support],
            block_dim=64,
            device=self.device,
        )

    def reset_impulses(self):
        self.impulses.assign(self.data["initial_impulses"])

    def result(self):
        return self.delta.numpy(), self.impulses.numpy()


def assert_output(test, data, delta, impulses, *, physical=None, reference=None):
    """Check independent response action and physical reference, not kernel naming."""
    counts = data["world_constraint_count"]
    live = np.arange(impulses.shape[1])[None] < counts[:, None]
    test.assertTrue(np.isfinite(delta).all())
    test.assertTrue(np.isfinite(impulses[live]).all())
    if reference is not None:
        np.testing.assert_allclose(delta, reference[0], atol=3e-5, rtol=3e-4)
        np.testing.assert_allclose(impulses[live], reference[1][live], atol=3e-5, rtol=3e-4)
    for world, count in enumerate(counts):
        if count == 0:
            np.testing.assert_array_equal(delta[world], 0.0)
            continue
        if physical is not None:
            velocity = physical["initial"] + np.linalg.solve(physical["factor"].T, delta[world])
            np.testing.assert_allclose(velocity, physical["expected_velocity"], atol=3e-5, rtol=3e-4)
            np.testing.assert_allclose(impulses[world, :count], physical["expected_impulse"], atol=3e-5, rtol=3e-4)
        # Delayed friction intentionally discards seeds without response; do not
        # falsely impose the ordinary delta-lambda identity on that old behavior.
        if int(data["iteration_offset"]) >= int(data["friction_start_iteration"]):
            dense = np.zeros((count, delta.shape[1]), dtype=np.float64)
            indices = data["support_class_dof"][data["support_row_class"][world, :count]]
            for row in range(count):
                valid = indices[row] >= 0
                dense[row, indices[row, valid]] = data["row_factor"][world, row, valid]
            expected_delta = dense.T @ (impulses[world, :count] - data["initial_impulses"][world, :count])
            np.testing.assert_allclose(delta[world], expected_delta, atol=3e-5, rtol=3e-4)
        for row in range(count):
            parent = int(data["row_parent"][world, row])
            if data["row_type"][world, row] != FRICTION or row != parent + 1:
                continue
            if parent < 0 or data["row_parent"][world, parent] >= 0:
                continue  # Linked normal loads retain their existing fallback.
            radius = max(float(data["row_mu"][world, row] * impulses[world, parent]), 0.0)
            test.assertLessEqual(float(np.linalg.norm(impulses[world, row : row + 2])), radius + 3e-5)


class TestSparseContactTriplets(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Sparse construction requires CUDA")
    def test_host_admission_and_shared_memory_fallback_cuda(self):
        """Admit compatible triplets and preserve sparse fallback at resource limits."""
        model = _branched_model(4, 3, "cuda:0")
        cases = (
            ("0", 192, {}, False),
            ("1", 192, {}, True),
            ("1", 40, {}, False),  # The tiny batch's support table costs more than explicit indices.
            ("1", 192, {"sparse_active_contacts": False}, False),
            ("1", 192, {"sparse_row_support": False}, False),
            ("1", 1227, {}, False),
            ("1", 1500, {}, False),
        )
        for flag, capacity, overrides, selected in cases:
            with (
                self.subTest(flag=flag, capacity=capacity, overrides=overrides),
                mock.patch.dict(os.environ, {"FEATHER_PGS_SPARSE_CONTACT_TRIPLETS": flag}, clear=True),
                mock.patch.dict(SolverFeatherPGS._kernel_overrides, overrides),
            ):
                solver = SolverFeatherPGS(
                    model,
                    pgs_mode="matrix_free",
                    drive_mode="augmented",
                    articulated_contact_response="immediate",
                    friction_anchor_beta=0.0,
                    enable_joint_velocity_limits=False,
                    pgs_warmstart=False,
                    mf_warmstart=False,
                    dense_max_constraints=capacity,
                    grouped_dynamics=False,
                    mf_gs_parallel_rows=0,
                    mf_gs_response_block_rows=0,
                    use_parallel_streams=True,
                )
                self.assertEqual(solver._sparse_mass_matrix_size, 18)
                self.assertEqual("contact_triplets" in solver._pgs_solve_sparse_kernel.key, selected)

    def test_factory_on_coupled_fixture_cpu(self):
        """Exercise coupled normal and tangent rows before constructing the opt-in kernel."""
        _, physical = fixture(offset=30, seeded=True)
        test_normal, test_tangent = 31, 32
        cross = physical["jacobian"][test_normal] @ physical["response"][test_tangent]
        self.assertGreater(abs(cross), 1e-4)
        self.assertGreater(np.linalg.norm(physical["expected_velocity"] - physical["initial"]), 1e-3)
        self.assertGreater(np.linalg.norm(physical["expected_impulse"] - physical["initial_impulse"]), 1e-3)
        selected = factory(192, 18, 12)
        old = factory(192, 18, 12, candidate=False)
        self.assertIsNot(selected, old)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires root-owned CUDA lease")
    def test_eight_sweep_physical_reference_cuda(self):
        """Preserve the ordered physical solve across contact activation and friction cases."""
        cases = (
            [{"offset": value, "seeded": True} for value in (0, 29, 30, 184)]
            + [
                {"activity": value}
                for value in (
                    "late_activation",
                    "deferred_activation",
                    "near_singular",
                    "seeded_retraction",
                    "linked_load",
                )
            ]
            + [{"omega": value, "seeded": True} for value in (0.0, 0.7, 1.2, -0.5)]
            + [{"delay": 3, "iteration_offset": value, "seeded": True} for value in (0, 1, 4)]
        )
        for settings in cases:
            with self.subTest(settings=settings):
                data, physical = fixture(**settings)
                control, candidate = DeviceSolve(data, candidate=False), DeviceSolve(data, candidate=True)
                control.launch()
                candidate.launch()
                result = candidate.result()
                assert_output(self, data, *result, physical=physical, reference=control.result())

    @unittest.skipUnless(wp.is_cuda_available(), "Requires root-owned CUDA lease")
    def test_finite_fallback_and_distinct_live_worlds_cuda(self):
        """Keep legal fallback rows and isolate each world's cached metric."""
        for mode in (
            "two_live_slots",
            "zero_normal_diag",
            "negative_normal_diag",
            "tail_normal",
            "different_normal_support",
        ):
            with self.subTest(mode=mode):
                data, _ = fixture(seeded=True)
                count = int(data["world_constraint_count"][0])
                # Both warps of block0 must be live and physically different.
                data["world_constraint_count"][1] = count
                data["row_factor"][1] *= 0.55
                data["row_incident"][1] *= 0.55
                data["diagonal"][1] = (data["diagonal"][1] - 1e-6) * 0.55**2 + 0.015
                data["row_mu"][1] *= 0.4
                data["rhs_bias"][1] *= 0.7
                if mode in ("zero_normal_diag", "negative_normal_diag"):
                    # The original kernel skips this normal but still visits
                    # its legal pair at the incoming seeded normal load.
                    data["diagonal"][:, 1] = 0.0 if mode == "zero_normal_diag" else -0.5
                elif mode == "tail_normal":
                    # The last valid row is an ordinary contact normal without
                    # children; do not construct a malformed t1 with no sibling.
                    data["world_constraint_count"][:] = 5
                    self.assertEqual(data["row_type"][0, 4], CONTACT)
                    self.assertEqual(data["row_parent"][0, 4], -1)
                elif mode == "different_normal_support":
                    classes = np.repeat(data["support_class_dof"], 2, axis=0)
                    classes[1, -1] = -1
                    data["support_class_dof"] = classes
                    data["support_row_class"][:, 1] = 1
                    data["row_factor"][:, 1, -1] = 0.0
                    data["diagonal"][:, 1] = np.sum(data["row_factor"][:, 1] ** 2, axis=1) + 1e-6
                    self.assertTrue(np.all(data["support_row_class"][:, 2:4] == 0))
                control, candidate = DeviceSolve(data, candidate=False), DeviceSolve(data, candidate=True)
                control.launch()
                candidate.launch()
                assert_output(self, data, *candidate.result(), reference=control.result())

    @unittest.skipUnless(wp.is_cuda_available(), "Requires root-owned CUDA lease")
    def test_graph_empty_regrow_changed_metric_cuda(self):
        """Reset cached metrics as captured buffers change count, response and CFM."""
        data, _ = fixture(offset=30, seeded=True)
        control, candidate = DeviceSolve(data, candidate=False), DeviceSolve(data, candidate=True)
        control.launch()
        candidate.launch()
        with wp.ScopedCapture(device=candidate.device) as captured:
            candidate.launch()
        original = {
            key: data[key].copy() for key in ("world_constraint_count", "row_factor", "diagonal", "row_incident")
        }
        for loaded, factor_scale, cfm in ((True, 1.0, 1e-6), (False, 1.0, 1e-6), (True, 0.6, 0.01), (True, 1.0, 1e-6)):
            data["world_constraint_count"] = original["world_constraint_count"] if loaded else np.zeros(3, np.int32)
            data["row_factor"] = original["row_factor"] * factor_scale
            data["row_incident"] = original["row_incident"] * factor_scale
            data["diagonal"] = (original["diagonal"] - 1e-6) * factor_scale**2 + cfm
            for runner in (control, candidate):
                for slot, key in (
                    (1, "world_constraint_count"),
                    (3, "diagonal"),
                    (6, "row_factor"),
                    (8, "row_incident"),
                ):
                    runner.inputs[slot].assign(data[key])
                runner.reset_impulses()
            control.launch()
            wp.capture_launch(captured.graph)
            assert_output(self, data, *candidate.result(), reference=control.result())


def check_frozen_payload(path, *, device="cuda:0"):
    """Root-only diagnostic replay of one actual512-world refresh/held NPZ.

    Source/manifest/hash verification belongs to the external runner; this
    reusable adapter never picks a file implicitly or changes a production test.
    """
    with np.load(Path(path), allow_pickle=False) as payload:
        data = {key: payload[key].copy() for key in payload.files}
    assert int(data["iterations"]) == 8 and int(data["iteration_offset"]) == 0
    assert data["row_factor"].shape[0] == len(data["world_ids"]) <= 512
    control = DeviceSolve(data, candidate=False, device=device)
    candidate = DeviceSolve(data, candidate=True, device=device)
    control.launch()
    candidate.launch()
    test = unittest.TestCase()
    reference = (data["reference_factor_delta"], data["reference_impulses"])
    assert_output(test, data, *control.result(), reference=reference)
    assert_output(test, data, *candidate.result(), reference=control.result())
    return {
        "path": str(path),
        "worlds": len(data["world_ids"]),
        "same_input_output_check": "pass",
        "independent_response_and_cone_check": "pass",
    }


if __name__ == "__main__":
    unittest.main()
