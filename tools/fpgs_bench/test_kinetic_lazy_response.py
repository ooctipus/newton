# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise the full-visit lazy owner with existing Kuka physical fixtures."""

import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import kinetic_guard, kinetic_predictor, kinetic_solve
from tools.fpgs_bench import test_kinetic_live_bindings as live_tests
from tools.fpgs_bench.test_kinetic_current_contact import contact_call


def module():
    """Keep regression-first failure local to the requested feature."""
    return importlib.import_module("newton._src.solvers.feather_pgs.kinetic_lazy_response")


def copy_arrays(value, device, memo):
    """Rebind existing live descriptors while preserving every array alias."""
    if isinstance(value, wp.array):
        key = (value.ptr, value.dtype, value.size)
        if key not in memo:
            memo[key] = wp.array(value.numpy(), dtype=value.dtype, device=device)
        return memo[key].reshape(value.shape)
    if hasattr(value, "_cls"):
        result = value._cls()
        for name in value._cls.vars:
            setattr(result, name, copy_arrays(getattr(value, name), device, memo))
        return result
    if isinstance(value, dict):
        return {name: copy_arrays(item, device, memo) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [copy_arrays(item, device, memo) for item in value]
    return value


def current_rows(enabled):
    """Reuse actual common-arm, prefix, free-MF and coupled current geometry."""
    with patch.dict(os.environ, {"FEATHER_PGS_KUKA_LAZY_RESPONSE": str(int(enabled))}):
        call = contact_call(True)[-1]
    raw = call.rows.raw
    raw.normal.assign(np.tile([0.6, 0.0, 0.8], (raw.normal.size, 1)))
    raw.point0.assign(np.tile([0.03, -0.02, 0.01], (raw.point0.size, 1)))
    # Add a dense self-contact alongside the existing free-body MF contact in
    # world1: independent admission must skip real dense rows, not an empty set.
    shapes = raw.shape0.numpy()
    shapes[8] = 32 + 13
    raw.shape0.assign(shapes)
    shapes = raw.shape1.numpy()
    shapes[8] = 32 + 17
    raw.shape1.assign(shapes)
    raw.count.assign([9])
    call.zero.launch()
    call.solve.setup()
    call.solve.allocate()
    # The original producer normally omits MF0 J publication. Ask only its row
    # producer for complete physical J; restore true routing before any solve.
    counts = call.rows.state.mf_count.numpy().copy()
    if not enabled:
        call.rows.state.mf_count.fill_(1)
    call.rows.out.response.fill_(float("nan"))
    call.rows.out.diag.fill_(float("nan"))
    call.solve.build_rows()
    call.rows.state.mf_count.assign(counts)
    return call


def synthetic_rows():
    """Embed the existing nine-contact late-activation chain in live 29D slots."""
    call = current_rows(False)
    _plan, held, state, rows, solve = call.solve.arguments["offset_eight"]
    worlds = 4
    counts = np.array([27, 4, 0, 27], np.int32)
    po = np.array([0, 0, 0, 6], np.int32)
    state.primary_offset.assign(po)
    state.secondary_offset.assign(np.where(po == 0, 23, 0).astype(np.int32))
    state.active_worlds.assign(np.arange(worlds, dtype=np.int32))
    state.active_count.assign([worlds])
    state.dense_count.assign(counts)
    state.mf_count.zero_()
    state.resolved.zero_()
    state.predictor_status.zero_()
    state.v_hat.zero_()
    state.held_generation.fill_(7)
    held.valid.fill_(1)
    held.generation.fill_(7)
    scales = np.array([1.0, 0.8, 1.0, 1.2], np.float32)
    held.T.assign(kinetic_predictor.pack_augmented(scales[:, None, None] * np.eye(23, dtype=np.float32)))
    held.inverse6.assign(np.tile(np.eye(6, dtype=np.float32), (worlds, 1, 1)))
    solve.selector.zero_()
    solve.status.zero_()
    rows.status.zero_()
    rows.global_status.zero_()
    rows.valid.fill_(1)
    rows.phase_bounds.assign(np.array([[0, 0], [1, 1], [0, 0], [0, 0]], np.int32))
    j = np.zeros((worlds, 192, 29), np.float32)
    types = np.zeros((worlds, 192), np.int32)
    parents = np.full((worlds, 192), -1, np.int32)
    rhs = np.zeros((worlds, 192), np.float32)
    mu = np.zeros((worlds, 192), np.float32)
    for world in (0, 3):
        for contact in range(9):
            row = contact * 3
            j[world, row, po[world] + contact] = 1
            if contact:
                j[world, row, po[world] + contact - 1] = -1
            types[world, row + 1 : row + 3] = 2
            parents[world, row + 1 : row + 3] = row
        rhs[world, 0] = -1
    j[1, :4, :3] = [[1, 0, 0], [0.2, 1, 0], [0.1, 0.3, 1], [-0.2, 0.4, 0.7]]
    types[1, :4], parents[1, :4] = [3, 0, 2, 2], [-1, -1, 1, 1]
    rhs[1, :4], mu[1, :4] = [-0.2, -0.3, -0.8, 0.55], [0, 0.5, 0.5, 0.5]
    rows.physical_J.assign(j)
    rows.row_type.assign(types)
    rows.row_parent.assign(parents)
    rows.row_mu.assign(mu)
    rows.r0.assign(rhs)
    rows.rhs.assign(rhs)
    rows.row_cfm.fill_(1e-6)
    rows.impulses.zero_()
    return call


def original_responses(args):
    """Materialize the original held factor rows outside the tested launch."""
    _plan, held, state, rows, _solve = args
    j = rows.physical_J.numpy()
    t = kinetic_predictor.unpack_T(held.T.numpy())
    z = np.zeros_like(j)
    for world, po in enumerate(state.primary_offset.numpy()):
        so = int(state.secondary_offset.numpy()[world])
        z[world, :, po : po + 23] = j[world, :, po : po + 23] @ t[world].T
        group = int(_plan.secondary_group.numpy()[world])
        z[world, :, so : so + 6] = j[world, :, so : so + 6] @ held.inverse6.numpy()[group].T
    rows.response.assign(z)
    rows.diag.assign(np.einsum("wrd,wrd->wr", z, z) + rows.row_cfm.numpy())


def split_case(enabled, device):
    """Rebind the installed producer/qualifier/bridge and original MF owner."""
    source = current_rows(enabled)
    source.solve.prepare_mf()
    device = wp.get_device(device)
    memo = {}
    row_args, args, guard, mf_impulses = copy_arrays(
        [source.rows.arguments, source.solve.arguments, source.solve.guard, source.services.impulses], device, memo
    )
    state, rows, solve = args["offset_eight"][2:]
    stream = wp.Stream(device) if device.is_cuda else None
    ready = wp.Event(device) if device.is_cuda else None
    done = wp.Event(device) if device.is_cuda else None

    def dispatch(name):
        dim, block = source.solve.dimensions[name]
        wp.launch_tiled(source.solve.kernels[name], dim=[dim], inputs=args[name], block_dim=block, device=device)

    def launch(*, solve_rows=True):
        kinetic_guard.initialize_guard(guard, device)
        wp.copy(solve.v_out, state.v_hat)
        rows.impulses.zero_()
        mf_impulses.zero_()
        rows.response.fill_(float("nan"))
        rows.diag.fill_(float("nan"))
        if enabled:
            rows.physical_J.fill_(float("nan"))
        rows.global_status.zero_()
        for kernel, arguments, dim in zip(
            source.rows.kernels, row_args, (4, 4, source.rows.settings.workers, 4), strict=True
        ):
            wp.launch_tiled(kernel, dim=[dim], inputs=arguments, block_dim=32, device=device)
        dispatch("qualify")
        dispatch("materialize")
        kinetic_guard.check_guard(guard, device)
        if solve_rows:
            if device.is_cuda:
                wp.record_event(ready)
                with wp.ScopedStream(stream):
                    wp.wait_event(ready)
                    dispatch("offset_eight")
                    wp.record_event(done)
                dispatch("general")
                wp.wait_event(done)
            else:
                dispatch("offset_eight")
            kinetic_guard.check_guard(guard, device)

    return SimpleNamespace(launch=launch, state=state, rows=rows, solve=solve, guard=guard, mf_impulses=mf_impulses)


def check_split(test, device, *, solve_rows):
    """Keep independent and coupled branches honest at the installed boundary."""
    reference, candidate = split_case(False, device), split_case(True, device)

    def compare():
        for case in (reference, candidate):
            np.testing.assert_array_equal(case.guard.frame_status.numpy(), 0)
            np.testing.assert_array_equal(case.rows.status.numpy(), 0)
            np.testing.assert_array_equal(case.solve.status.numpy(), 0)
            selector = case.solve.selector.numpy()
            test.assertEqual(int(selector[0]), 0)
            test.assertEqual(int(selector[1]), -1)
            test.assertGreater(int(selector[2]), 0)
            test.assertGreater(int(case.state.dense_count.numpy()[1]), 0)
            test.assertGreater(int(case.state.mf_count.numpy()[1]), 0)
        if not solve_rows:
            for world in (0, 1):
                test.assertTrue(np.isnan(candidate.rows.response.numpy()[world]).all())
                test.assertTrue(np.isnan(candidate.rows.diag.numpy()[world]).all())
            count = int(candidate.state.dense_count.numpy()[2])
            test.assertTrue(np.isfinite(candidate.rows.response.numpy()[2, :count]).all())
        else:
            for first, second in (
                (reference.solve.v_out, candidate.solve.v_out),
                (reference.rows.impulses, candidate.rows.impulses),
                (reference.mf_impulses, candidate.mf_impulses),
            ):
                a, b = first.numpy(), second.numpy()
                test.assertTrue(np.isfinite(b).all())
                test.assertLessEqual(float(np.max(np.abs(a - b))) / (1 + float(np.max(np.abs(a)))), 2.0**-17)

    for case in (reference, candidate):
        case.launch(solve_rows=solve_rows)
    compare()
    if wp.get_device(device).is_cuda:
        graphs = []
        for case in (reference, candidate):
            with wp.ScopedCapture(device=device) as capture:
                case.launch(solve_rows=solve_rows)
            graphs.append(capture.graph)
        for _ in range(2):
            for graph in graphs:
                wp.capture_launch(graph)
            compare()


def check_solve(test, device):
    """Compare actual entries across cold, changed held/J and warm graph calls."""
    device = wp.get_device(device)
    source = synthetic_rows().solve.arguments["offset_eight"]
    first, second = (copy_arrays(source, device, {}) for _ in range(2))
    kernels = (kinetic_solve.get_solve_kernel(str(device.arch)), module().get_solve_kernel(str(device.arch)))
    seeds = [wp.zeros_like(args[3].impulses) for args in (first, second)]

    def launch(index):
        args = (first, second)[index]
        wp.copy(args[4].v_out, args[2].v_hat)
        wp.copy(args[3].impulses, seeds[index])
        args[4].status.zero_()
        if index:
            args[3].response.fill_(float("nan"))
            args[3].diag.fill_(float("nan"))
        wp.launch_tiled(kernels[index], dim=[2], inputs=args, block_dim=64, device=device)

    graphs = []
    for index in range(2):
        if index == 0:
            original_responses(first)
        launch(index)
        if device.is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                launch(index)
            graphs.append(capture.graph)
    for epoch in range(4):
        for args, seed in zip((first, second), seeds, strict=True):
            state, rows = args[2:4]
            count = [27, 4, 0, 27]
            if epoch == 1:
                count[0], count[3] = 0, 3
            state.dense_count.assign(count)
            # Same captured buffers, a new held action and current RHS/J.
            if epoch == 2:
                args[1].T.assign(args[1].T.numpy() * np.float32(0.9))
                rhs = rows.r0.numpy()
                rhs[0, 0], rhs[3, 0] = -0.7, -1.3
                rows.r0.assign(rhs)
                j = rows.physical_J.numpy()
                j[1, 2, 2] += np.float32(0.125)
                rows.physical_J.assign(j)
            if epoch == 3:
                incoming = seed.numpy()
                incoming[1, :4] = [0.1, 0.2, 0.15, -0.13]
                seed.assign(incoming)
        original_responses(first)
        for index in range(2):
            if graphs:
                wp.capture_launch(graphs[index])
            else:
                launch(index)
        for args in (first, second):
            np.testing.assert_array_equal(args[4].status.numpy(), 0)
        for expected, actual in ((first[4].v_out, second[4].v_out), (first[3].impulses, second[3].impulses)):
            a, b = expected.numpy(), actual.numpy()
            test.assertTrue(np.isfinite(b).all())
            test.assertLessEqual(float(np.max(np.abs(a - b))) / (1 + float(np.max(np.abs(a)))), 2.0**-17)
        plan, held, state, rows, solve = second
        j, lam = rows.physical_J.numpy().astype(np.float64), rows.impulses.numpy().astype(np.float64)
        seed = seeds[1].numpy().astype(np.float64)
        for world, count in enumerate(state.dense_count.numpy()):
            po, so = int(state.primary_offset.numpy()[world]), int(state.secondary_offset.numpy()[world])
            ids = np.empty(29, np.int32)
            ids[po : po + 23], ids[so : so + 6] = plan.dof_ids.numpy()[world, :23], plan.dof_ids.numpy()[world, 23:29]
            t = np.zeros((29, 29), np.float64)
            t[po : po + 23, po : po + 23] = kinetic_predictor.unpack_T(held.T.numpy()[world].astype(np.float64))
            t[so : so + 6, so : so + 6] = held.inverse6.numpy()[plan.secondary_group.numpy()[world]]
            h = np.linalg.inv(t.T @ t)
            dv = solve.v_out.numpy()[ids].astype(np.float64) - state.v_hat.numpy()[ids].astype(np.float64)
            impulse = j[world, :count].T @ (lam[world, :count] - seed[world, :count])
            error = np.max(np.abs(h @ dv - impulse))
            scale = 1 + np.linalg.norm(h, np.inf) * np.linalg.norm(dv, np.inf) + np.linalg.norm(impulse, np.inf)
            test.assertLessEqual(float(error / scale), 2.0**-17)
        radius = 0.5 * lam[1, 1]
        test.assertGreaterEqual(radius, 0)
        test.assertLessEqual(float(np.linalg.norm(lam[1, 2:4]) - radius), (1 + radius) * 2.0**-17)
        if epoch == 0:
            # Every normal in the nine-contact chain activates in its canonical
            # first sweep; the old once-per-sweep admission omits the last one.
            test.assertTrue(np.all(second[3].impulses.numpy()[0, :27:3] > 0))
            velocity, impulses = np.zeros(9), np.zeros(9)
            for _ in range(8):
                for row in range(9):
                    j = np.zeros(9)
                    j[row] = 1
                    if row:
                        j[row - 1] = -1
                    old = impulses[row]
                    residual = j @ velocity - float(row == 0)
                    impulses[row] = max(old - residual / (j @ j + 1e-6), 0)
                    velocity += j * (impulses[row] - old)
            ids = second[0].dof_ids.numpy()[0, :9]
            np.testing.assert_allclose(second[4].v_out.numpy()[ids], velocity, rtol=2.0**-17, atol=2.0**-17)
            np.testing.assert_allclose(second[3].impulses.numpy()[0, :27:3], impulses, rtol=2.0**-17, atol=2.0**-17)
        test.assertTrue(np.isnan(second[3].response.numpy()[2]).all())


class TestKineticLazyResponseCPU(unittest.TestCase):
    def test_factory_api(self):
        """Expose the distinct production entry without changing its ABI."""
        kernel = module().get_solve_kernel("cpu")
        self.assertTrue(kernel.key.endswith("kinetic_lazy_physical_eight"), kernel.key)
        self.assertEqual(len(kernel.adj.args), 5)

    def test_default_off_and_original_nonpositive_cfm(self):
        """Keep the original complete owner when disabled or CFM is unsupported."""
        with patch.dict(os.environ, {"FEATHER_PGS_KUKA_LAZY_RESPONSE": "0"}):
            call = contact_call(True)[-1]
        self.assertIsNone(getattr(call, "lazy_response", None))
        self.assertIs(call.solve.kernels["offset_eight"], kinetic_solve.get_solve_kernel("0"))
        original = live_tests.live_solver
        for cfm in (0.0, -1e-4):

            def solver(worlds, value=cfm):
                result = original(worlds)
                result.pgs_cfm = value
                return result

            with (
                patch.object(live_tests, "live_solver", solver),
                patch.dict(os.environ, {"FEATHER_PGS_KUKA_LAZY_RESPONSE": "1"}),
            ):
                call = contact_call(True)[-1]
            self.assertFalse(call.lazy_response.active)
            self.assertIs(call.solve.kernels["offset_eight"], kinetic_solve.get_solve_kernel("0"))

    def test_complete_current_j_without_response(self):
        """Publish full common-arm and signed prefix J without eager responses."""
        reference, candidate = current_rows(False), current_rows(True)
        self.assertTrue(candidate.lazy_response.active)
        for call in (reference, candidate):
            np.testing.assert_array_equal(call.rows.out.status.numpy(), 0)
            np.testing.assert_array_equal(call.rows.out.global_status.numpy(), 0)
        counts = candidate.rows.state.dense_count.numpy()
        for world, count in enumerate(counts):
            for name in ("physical_J", "r0", "rhs", "row_cfm", "row_type", "row_parent", "row_mu"):
                a = getattr(reference.rows.out, name).numpy()[world, :count]
                b = getattr(candidate.rows.out, name).numpy()[world, :count]
                np.testing.assert_allclose(b, a, rtol=2.0**-17, atol=2.0**-17, err_msg=name)
        prefix = int(candidate.rows.out.phase_bounds.numpy()[0, 1])
        self.assertGreater(prefix, 0)
        j = candidate.rows.out.physical_J.numpy()[0]
        self.assertTrue(np.any(np.abs(j[prefix : counts[0], :7]) > 1e-5))
        np.testing.assert_array_equal(np.sum(np.abs(j[:prefix]), axis=1), 1)
        self.assertTrue(np.isnan(candidate.rows.out.response.numpy()).all())
        self.assertTrue(np.isnan(candidate.rows.out.diag.numpy()).all())

    def test_original_eight_cpu_control(self):
        """Run the full-visit late chain and changed-cache scalar controls."""
        check_solve(self, "cpu")

    def test_independent_and_positive_materialization(self):
        """Leave independent responses unbuilt and fully prepare positive fallback."""
        check_split(self, "cpu", solve_rows=False)


@unittest.skipUnless(os.environ.get("FPGS_TEST_DEVICE", "cpu").startswith("cuda"), "Explicit root-owned CUDA lease")
class TestKineticLazyResponseCUDA(unittest.TestCase):
    def test_full_visit_and_captured_readiness(self):
        """Execute the actual lazy entry across changed same-buffer graph calls."""
        check_solve(self, os.environ["FPGS_TEST_DEVICE"])

    def test_current_rows_independent_and_coupled(self):
        """Run actual current producers and disjoint dense/MF native owners."""
        check_split(self, os.environ["FPGS_TEST_DEVICE"], solve_rows=True)


if __name__ == "__main__":
    unittest.main()
