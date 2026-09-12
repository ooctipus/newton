# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Control compact active dispatch without changing the paired response law."""

import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp


def fixture(device="cpu"):
    """Bind nonidentity groups and both physical component orders with poisoned tails."""
    module = importlib.import_module("newton._src.solvers.feather_pgs.kuka_active_response")
    worlds = 4
    rng = np.random.default_rng(41)
    primary_groups = np.array([2, 0, 3, 1], np.int32)
    secondary_groups = np.array([1, 3, 0, 2], np.int32)
    primary_arts = np.empty(worlds, np.int32)
    secondary_arts = np.empty(worlds, np.int32)
    pair = np.empty(worlds, np.int32)
    offsets = np.zeros(worlds * 3, np.int32)
    counts = np.array([5, 0, 3, 0], np.int32)
    mf_counts = np.array([0, 0, 3, 0], np.int32)
    matrices, jacobians = {}, {}
    for size in (23, 6):
        lower = np.tril(rng.normal(0, 0.03, (worlds, size, size))).astype(np.float32)
        for group in range(worlds):
            np.fill_diagonal(lower[group], rng.uniform(0.6, 1.4, size))
        lower[:, np.triu_indices(size, 1)[0], np.triu_indices(size, 1)[1]] = np.nan
        matrices[size] = lower
        jacobians[size] = np.full((worlds, 192, size), np.nan, np.float32)
    for world in range(worlds):
        pg, sg = primary_groups[world], secondary_groups[world]
        primary_arts[pg], secondary_arts[sg] = world * 3, world * 3 + 1
        pair[pg] = sg
        offsets[world * 3], offsets[world * 3 + 1] = (0, 23) if world % 3 == 0 else (6, 0)
        if counts[world]:
            jacobians[23][pg, : counts[world]] = rng.normal(0, 0.2, (counts[world], 23))
            jacobians[6][sg, : counts[world]] = rng.normal(0, 0.2, (counts[world], 6))
        else:
            matrices[23][pg] = np.nan
            matrices[6][sg] = np.nan

    def arr(value, dtype=None):
        if dtype is None:
            dtype = wp.float32 if value.dtype.kind == "f" else wp.int32
        return wp.array(value, dtype=dtype, device=device)

    solver = SimpleNamespace(
        _paired_response_primary_size=23,
        _paired_response_secondary_size=6,
        _paired_factor_coordinates=True,
        dense_max_constraints=192,
        max_world_dofs=29,
        world_count=worlds,
        model=SimpleNamespace(device=wp.get_device(device)),
        Hinv_by_size={size: arr(np.full_like(matrices[size], np.nan)) for size in (23, 6)},
        Linv_by_size={size: arr(matrices[size]) for size in (23, 6)},
        J_by_size={size: arr(jacobians[size]) for size in (23, 6)},
        group_to_art={23: arr(primary_arts), 6: arr(secondary_arts)},
        _paired_response_secondary_groups=arr(pair),
        art_to_world=arr(np.repeat(np.arange(worlds, dtype=np.int32), 3)),
        articulation_world_dof_offset=arr(offsets),
        constraint_count=arr(counts),
        mf_constraint_count=arr(mf_counts),
        row_cfm=arr(np.full((worlds, 192), 0.017, np.float32)),
        J_world=arr(np.full((worlds, 192, 29), 123.0, np.float32)),
        Y_world=arr(np.full((worlds, 192, 29), 456.0, np.float32)),
        diag=arr(np.full((worlds, 192), 789.0, np.float32)),
        _paired_factor_primary_groups_by_world=arr(primary_groups),
    )
    active = arr(np.array([2, 0, 1, -1234], np.int32))
    count = arr(np.array([3], np.int32))
    owner = module.ActiveKukaResponse(solver, active, count)
    return solver, owner


def reference(solver, owner):
    """Apply the physical held operator in FP64 to the identical published FP32 J."""
    result = [solver.J_world.numpy(), solver.Y_world.numpy(), solver.diag.numpy()]
    counts, mf = solver.constraint_count.numpy(), solver.mf_constraint_count.numpy()
    pg = solver._paired_factor_primary_groups_by_world.numpy()
    pair = solver._paired_response_secondary_groups.numpy()
    arts = {size: solver.group_to_art[size].numpy() for size in (23, 6)}
    offsets = solver.articulation_world_dof_offset.numpy()
    factors = {size: solver.Linv_by_size[size].numpy() for size in (23, 6)}
    jacobians = {size: solver.J_by_size[size].numpy() for size in (23, 6)}
    cfm = solver.row_cfm.numpy()
    for world in owner.active_worlds.numpy()[: owner.active_count.numpy()[0]]:
        count = min(counts[world], 192)
        if count <= 0:
            continue
        diagonal = np.zeros(count, np.float64)
        for size, group in ((23, pg[world]), (6, pair[pg[world]])):
            offset = offsets[arts[size][group]]
            j = jacobians[size][group, :count].astype(np.float64)
            lower = np.tril(factors[size][group].astype(np.float64))
            z = j @ lower.T
            y = z if mf[world] == 0 else z @ lower
            diagonal += np.sum(z * z if mf[world] == 0 else j * y, axis=1)
            result[0][world, :count, offset : offset + size] = j
            result[1][world, :count, offset : offset + size] = y
        result[2][world, :count] = diagonal + cfm[world, :count]
    return result


def assert_outputs(test, solver, expected):
    """Compare active action and exact untouched output sentinels together."""
    for array, control in zip((solver.J_world, solver.Y_world, solver.diag), expected, strict=True):
        np.testing.assert_allclose(array.numpy(), control, rtol=2e-6, atol=3e-7)


class TestKukaActiveResponse(unittest.TestCase):
    def test_factory_available(self):
        """Require the new active-only response owner before integration."""
        module = importlib.import_module("newton._src.solvers.feather_pgs.kuka_active_response")
        self.assertTrue(callable(module.get_response_kernel))

    def test_original_cuda_source_recovery(self):
        """Recover every original CUDA statement after removing only the empty guard."""
        module = importlib.import_module("newton._src.solvers.feather_pgs.kuka_active_response")

        original, adapted = module.response_sources("cpu")
        recovered = adapted.replace(module._CPU_RESPONSE, "").replace(module._EMPTY_GUARD, "")
        self.assertEqual(original, recovered)
        self.assertLess(adapted.index(module._EMPTY_GUARD), adapted.index("for (int element = threadIdx.x"))
        with patch.object(module, "_FACTORY_SHA256", "unreviewed"):
            with self.assertRaisesRegex(RuntimeError, "factory changed"):
                module.response_sources("cpu")

    def test_current_rows_and_nonidentity_fp64_action(self):
        """Preserve direct Z, mixed physical Y, diagonal CFM and current group maps."""
        solver, owner = fixture()
        expected = reference(solver, owner)
        before = [array.numpy() for array in owner.inputs()[:14]]
        owner.launch()
        assert_outputs(self, solver, expected)
        for array, original in zip(owner.inputs()[:14], before, strict=True):
            np.testing.assert_array_equal(array.numpy(), original)

    def test_empty_growing_shrinking_prefix_and_held_refresh(self):
        """Use current counts and factors without clearing or reading inactive tails."""
        solver, owner = fixture()
        owner.active_count.assign(np.array([0], np.int32))
        expected = reference(solver, owner)
        owner.launch()
        assert_outputs(self, solver, expected)
        owner.active_count.assign(np.array([3], np.int32))
        for rows in (192, 1, 0, 8):
            counts = solver.constraint_count.numpy()
            counts[0] = rows
            solver.constraint_count.assign(counts)
            for size in (23, 6):
                group = 2 if size == 23 else 1
                j = solver.J_by_size[size].numpy()
                j[group] = np.nan
                j[group, :rows] = np.arange(rows * size, dtype=np.float32).reshape(rows, size) * 0.0001
                solver.J_by_size[size].assign(j)
                lower = solver.Linv_by_size[size].numpy()
                lower[group, np.arange(size), np.arange(size)] *= np.float32(1.01)
                solver.Linv_by_size[size].assign(lower)
            expected = reference(solver, owner)
            owner.launch()
            assert_outputs(self, solver, expected)

    def test_binding_and_layout_rejection(self):
        """Keep all original owners and reject unsupported construction without allocation."""
        module = importlib.import_module("newton._src.solvers.feather_pgs.kuka_active_response")

        solver, owner = fixture()
        args = owner.inputs()
        self.assertEqual(len(args), 20)
        self.assertIs(args[14], solver.J_world)
        self.assertIs(args[15], solver.Y_world)
        self.assertIs(args[16], solver.diag)
        self.assertIs(args[17], owner.active_worlds)
        with patch.object(wp, "launch_tiled") as launch:
            owner.launch()
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(launch.call_args.kwargs["dim"], [solver.world_count])
        self.assertEqual(launch.call_args.kwargs["block_dim"], 128)
        for field, value in (
            ("dense_max_constraints", 193),
            ("max_world_dofs", 28),
            ("_paired_factor_coordinates", False),
        ):
            previous = getattr(solver, field)
            setattr(solver, field, value)
            with self.assertRaises(ValueError):
                module.ActiveKukaResponse(solver, owner.active_worlds, owner.active_count)
            setattr(solver, field, previous)

    def test_cuda_original_and_graph_transition(self):
        """Compare actual original CUDA response through current-list graph replay changes."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Root-owned CUDA gate required")
        original_factory = importlib.import_module(
            "newton._src.solvers.feather_pgs.solver_feather_pgs"
        )._get_paired_hinv_jt_kernel

        device = devices[0]
        solver, owner = fixture(device)
        original = original_factory(23, 6, 192, 29, str(device.arch), 4, factor_coordinates=True)
        outputs = [wp.clone(array) for array in (solver.J_world, solver.Y_world, solver.diag)]
        args = owner.inputs()[:14] + outputs

        def control():
            wp.launch_tiled(original, dim=[4], inputs=args, block_dim=128, device=device)

        owner.launch()
        control()
        for new, old in zip((solver.J_world, solver.Y_world, solver.diag), outputs, strict=True):
            np.testing.assert_allclose(new.numpy(), old.numpy(), rtol=2e-6, atol=3e-7)
        with wp.ScopedCapture(device=device) as capture:
            owner.launch()
            control()
        for counts, active in (([1, 0, 1, 0], [2, 0, 1, -1234]), ([0, 0, 3, 0], [2, 1, 0, -1234])):
            solver.constraint_count.assign(np.array(counts, np.int32))
            owner.active_worlds.assign(np.array(active, np.int32))
            wp.capture_launch(capture.graph)
            for new, old in zip((solver.J_world, solver.Y_world, solver.diag), outputs, strict=True):
                np.testing.assert_allclose(new.numpy(), old.numpy(), rtol=2e-6, atol=3e-7)


if __name__ == "__main__":
    unittest.main()
