# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check endpoint residuals through the existing live Kuka physical fixture."""

import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import kinetic_predictor, kinetic_solve
from tools.fpgs_bench.test_kinetic_current_contact import contact_call
from tools.fpgs_bench.test_kinetic_lazy_response import copy_arrays, original_responses


def physical_call(enabled, *, shared_anchor=False):
    """Reuse actual live chains with opposing contacts and all endpoint families."""
    with patch.dict(
        os.environ,
        {
            "FEATHER_PGS_KUKA_ENDPOINT_RESIDUAL": str(int(enabled)),
            "FEATHER_PGS_KUKA_LAZY_RESPONSE": "0",
        },
    ):
        call = contact_call(True, shared_anchor=shared_anchor)[-1]
    raw = call.rows.raw
    a, b = raw.shape0.numpy(), raw.shape1.numpy()
    # Unlike the two complete finger chains in world0, this arm/finger pair
    # shares only a prefix of the seven arm joints. It cannot use the full-arm
    # cancellation formula for every arm coefficient.
    a[8], b[8] = 32 + 3, 32 + 13
    if shared_anchor:
        # Body14 is the fixed child of body13: common rigid motion at a shared
        # anchor has no relative response, even with active neighbouring rows.
        b[2] = 14
    raw.shape0.assign(a)
    raw.shape1.assign(b)
    # Keep the opposed-normal activation visible with a low-friction material;
    # the other worlds retain their original nontrivial friction disks.
    mu = raw.shape_mu.numpy()
    mu[:32] = 1.0e-4
    raw.shape_mu.assign(mu)
    normal = np.tile([0.6, 0.0, 0.8], (raw.normal.size, 1)).astype(np.float32)
    normal[1] *= -1
    raw.normal.assign(normal)
    raw.margin0.zero_()
    raw.margin1.zero_()
    first = np.tile([0.03, -0.02, 0.01], (raw.point0.size, 1)).astype(np.float32)
    second = raw.point1.numpy()
    poses, bodies = raw.body_q.numpy(), raw.shape_body.numpy()
    # Moving an anchor along its normal controls the physical separation;
    # a tangential offset retains nonzero common-arm rotational response.
    gaps = [-0.001, 0.000025, 0.1, -0.001, -0.001, -0.001, 0.1, -0.001, -0.001]
    if shared_anchor:
        gaps[2] = 0.0
    for index, gap in enumerate(gaps):
        body_a = int(bodies[a[index]]) if a[index] >= 0 else -1
        body_b = int(bodies[b[index]]) if b[index] >= 0 else -1
        point = wp.vec3(first[index])
        if body_a >= 0:
            pose = wp.transform(wp.vec3(poses[body_a, :3]), wp.quat(poses[body_a, 3:]))
            point = wp.transform_point(pose, point)
        point = point + gap * wp.vec3(normal[index]) + wp.vec3(0.016, 0.013, -0.012)
        if body_b >= 0:
            pose = wp.transform(wp.vec3(poses[body_b, :3]), wp.quat(poses[body_b, 3:]))
            point = wp.transform_point(wp.transform_inverse(pose), point)
        second[index] = np.asarray(point)
    raw.point0.assign(first)
    raw.point1.assign(second)
    raw.count.assign([9])
    return call


def prepare_original(call):
    """Build the original complete physical operator without changing solve routing."""
    call.zero.launch()
    call.solve.setup()
    call.solve.allocate()
    counts = call.rows.state.mf_count.numpy().copy()
    # Original MF0 deliberately omits public J. This existing test-only producer
    # request exposes it for the independent oracle, then restores real routes.
    call.rows.state.mf_count.fill_(1)
    call.solve.build_rows()
    call.rows.state.mf_count.assign(counts)
    call.solve.prepare_mf()
    call.solve.qualify()
    call.solve.materialize()
    return call


def module():
    """Keep the feature's regression-first import failure local to this suite."""
    return importlib.import_module("newton._src.solvers.feather_pgs.kinetic_endpoint_residual")


def prepared_call(enabled, **recipe):
    """Execute the production phase aliases, including the early MF route."""
    call = physical_call(enabled, **recipe)
    if not enabled:
        return prepare_original(call)
    call.zero.launch()
    call.solve.setup()
    call.solve.allocate()
    call.solve.build_rows()
    call.solve.prepare_mf()
    call.solve.qualify()
    call.solve.materialize()
    return call


def direct_case(enabled, device, **recipe):
    """Replay actual native solve inputs from the existing live physical model."""
    source = prepared_call(enabled, **recipe)
    device = wp.get_device(device)
    args = copy_arrays(source.solve.arguments["offset_eight"], device, {})
    # Exercise the second physical packing on a real independent dense+MF
    # world. The global DOF identities and the physical model are unchanged.
    po, so = args[2].primary_offset.numpy(), args[2].secondary_offset.numpy()
    po[1], so[1] = 6, 0
    args[2].primary_offset.assign(po)
    args[2].secondary_offset.assign(so)
    if not enabled:
        j = args[3].physical_J.numpy()
        old = j[1].copy()
        j[1, :, 6:29], j[1, :, :6] = old[:, :23], old[:, 23:29]
        args[3].physical_J.assign(j)
        original_responses(args)
    factory = module().get_solve_kernel if enabled else kinetic_solve.get_solve_kernel
    kernel = factory(str(device.arch))
    seed = wp.zeros_like(args[3].impulses)
    state, rows, solve = args[2:5]
    if enabled:
        np.testing.assert_array_equal(args[8].route.numpy()[:2], 1)

    def launch():
        wp.copy(solve.v_out, state.v_hat)
        wp.copy(rows.impulses, seed)
        solve.status.zero_()
        if enabled:
            # Positive fallback owns its eager storage; private rows must not
            # read any stale response/diagonal before their first response.
            for world in (0, 1):
                rows.response[world].fill_(float("nan"))
                rows.diag[world].fill_(float("nan"))
        wp.launch_tiled(kernel, dim=[2], inputs=args, block_dim=64, device=device)

    return SimpleNamespace(source=source, args=args, seed=seed, launch=launch)


def close(test, expected, actual):
    """Reuse the existing kinetic native eight-sweep comparison gate."""
    first, second = np.asarray(expected), np.asarray(actual)
    test.assertTrue(np.isfinite(second).all())
    test.assertLessEqual(float(np.max(np.abs(first - second))) / (1 + float(np.max(np.abs(first)))), 2.0**-17)


def check_direct(test, device, *, shared_anchor=False):
    """Test endpoint residuals, response readiness and original impulse order."""
    first, second = (direct_case(enabled, device, shared_anchor=shared_anchor) for enabled in (False, True))
    reference_j = first.args[3].physical_J.numpy().astype(np.float64)
    original_counts = first.args[2].dense_count.numpy()
    predictor_twists = second.args[2].endpoint_twists.numpy().copy()
    current_axes = second.args[5].axes.numpy().copy()
    graphs = []
    for case in (first, second):
        case.launch()
        if wp.get_device(device).is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                case.launch()
            graphs.append(capture.graph)
    for epoch in range(4):
        for case in (first, second):
            plan, held, state, rows, solve = case.args[:5]
            counts = original_counts.copy()
            if epoch == 1:
                counts[0], counts[1] = 0, 0
            state.dense_count.assign(counts)
            if epoch == 2:
                # Current geometry is unchanged while the held action changes;
                # neither response nor readiness may survive the invocation.
                held.T.assign(held.T.numpy() * np.float32(0.9))
            if epoch == 3:
                incoming = case.seed.numpy()
                incoming[0, :4] = [0.03, 0.1, 0.015, -0.01]
                case.seed.assign(incoming)
                solve.friction_start_iteration = 1
        original_responses(first.args[:5])
        for index, case in enumerate((first, second)):
            if graphs:
                # Struct scalar launch parameters are captured, unlike array
                # contents; delayed friction is checked eagerly below.
                if epoch == 3:
                    case.launch()
                else:
                    wp.capture_launch(graphs[index])
            else:
                case.launch()
        for case in (first, second):
            np.testing.assert_array_equal(case.args[4].status.numpy(), 0)
            np.testing.assert_array_equal(case.args[3].status.numpy(), 0)
        close(test, first.args[4].v_out.numpy(), second.args[4].v_out.numpy())
        for world in (0, 1):
            count = int(second.args[2].dense_count.numpy()[world])
            if not count:
                continue
            close(test, first.args[3].impulses.numpy()[world, :count], second.args[3].impulses.numpy()[world, :count])
            plan, held, state, rows, solve = second.args[:5]
            ids = plan.dof_ids.numpy()[world, :23]
            po = int(state.primary_offset.numpy()[world])
            t = kinetic_predictor.unpack_T(held.T.numpy()[world].astype(np.float64))
            h = np.linalg.inv(t.T @ t)
            dv = solve.v_out.numpy()[ids].astype(np.float64) - state.v_hat.numpy()[ids].astype(np.float64)
            lam = rows.impulses.numpy()[world, :count].astype(np.float64)
            seed = second.seed.numpy()[world, :count].astype(np.float64)
            # Delayed friction literally clears incoming tangents without a
            # velocity update. Its original warm convention is not H dv=J dl.
            if epoch != 3:
                impulse = reference_j[world, :count, po : po + 23].T @ (lam - seed)
                scale = 1 + np.linalg.norm(h, np.inf) * np.linalg.norm(dv, np.inf) + np.linalg.norm(impulse, np.inf)
                test.assertLessEqual(float(np.max(np.abs(h @ dv - impulse)) / scale), 2.0**-17)
            types, parents = rows.row_type.numpy()[world], rows.row_parent.numpy()[world]
            mu = rows.row_mu.numpy()[world]
            for row in range(count):
                if types[row] == 2 and row == parents[row] + 1:
                    radius = max(float(mu[row] * lam[parents[row]]), 0.0)
                    test.assertLessEqual(float(np.linalg.norm(lam[row : row + 2]) - radius), (1 + radius) * 2.0**-17)
        np.testing.assert_array_equal(second.args[2].endpoint_twists.numpy(), predictor_twists)
        np.testing.assert_array_equal(second.args[5].axes.numpy(), current_axes)
        if epoch == 0 and not shared_anchor:
            # This normal starts feasible and activates from earlier physical
            # prefix/contact updates, not from an invented coefficient chain.
            test.assertGreater(float(first.args[3].r0.numpy()[0, 4]), 0)
            test.assertGreater(float(second.args[3].impulses.numpy()[0, 4]), 0)
        if epoch == 0 and shared_anchor:
            row = int(first.source.rows.raw.slot.numpy()[2])
            test.assertLess(float(np.max(np.abs(reference_j[0, row]))), 1.0e-6)
            # A near-zero relative operator is judged by the final physical
            # impulse too. Tiny gap bias divided by CFM already creates a
            # nonzero original impulse; the oracle is that load, not zero.
            close(
                test,
                first.args[3].impulses.numpy()[0, row : row + 3],
                second.args[3].impulses.numpy()[0, row : row + 3],
            )


class TestKineticEndpointResidualCPU(unittest.TestCase):
    def test_factory_api(self):
        """Require the actual current-geometry entry and complete descriptor ABI."""
        kernel = module().get_solve_kernel("cpu")
        self.assertIn("endpoint", kernel.key)
        self.assertEqual(len(kernel.adj.args), 9)

    def test_installed_packet_routes_and_current_metadata(self):
        first, second = prepared_call(False), prepared_call(True)
        self.assertTrue(second.endpoint_residual.active)
        self.assertFalse(getattr(getattr(first, "endpoint_residual", None), "active", False))
        np.testing.assert_array_equal(second.solve.arguments["offset_eight"][8].route.numpy()[:3], [1, 1, 0])
        np.testing.assert_array_equal(second.solve.solve_descriptor.selector.numpy()[:2], [0, -1])
        self.assertGreater(int(second.solve.solve_descriptor.selector.numpy()[2]), 0)
        for call in (first, second):
            for field in (
                call.rows.state.raw_invalid,
                call.rows.state.capacity_status,
                call.rows.out.status,
                call.rows.out.global_status,
                call.solve.solve_descriptor.status,
            ):
                np.testing.assert_array_equal(field.numpy(), 0)
        for world, count in enumerate(first.rows.state.dense_count.numpy()):
            np.testing.assert_array_equal(second.rows.state.dense_count.numpy()[world], count)
            if not count:
                continue
            for name in ("row_type", "row_parent", "valid"):
                np.testing.assert_array_equal(
                    getattr(first.rows.out, name).numpy()[world, :count],
                    getattr(second.rows.out, name).numpy()[world, :count],
                )
            for name in ("r0", "row_mu"):
                close(
                    self,
                    getattr(first.rows.out, name).numpy()[world, :count],
                    getattr(second.rows.out, name).numpy()[world, :count],
                )
        count = int(second.rows.state.dense_count.numpy()[2])
        for name in ("physical_J", "response", "diag"):
            close(
                self,
                getattr(first.rows.out, name).numpy()[2, :count],
                getattr(second.rows.out, name).numpy()[2, :count],
            )

    def test_actual_late_activation_fixture(self):
        call = prepare_original(physical_call(False))
        args = call.solve.arguments["offset_eight"]
        wp.launch_tiled(kinetic_solve.get_solve_kernel("cpu"), dim=[2], inputs=args, block_dim=64, device="cpu")
        self.assertGreater(float(args[3].r0.numpy()[0, 4]), 0)
        self.assertGreater(float(args[3].impulses.numpy()[0, 4]), 0)
        np.testing.assert_array_equal(args[4].status.numpy(), 0)


@unittest.skipUnless(wp.is_cuda_available(), "CUDA required for the actual endpoint owner")
class TestKineticEndpointResidualCUDA(unittest.TestCase):
    def test_actual_current_held_and_captured_readiness(self):
        check_direct(self, wp.get_device("cuda:0"))

    def test_common_rigid_shared_anchor(self):
        check_direct(self, wp.get_device("cuda:0"), shared_anchor=True)


if __name__ == "__main__":
    unittest.main()
