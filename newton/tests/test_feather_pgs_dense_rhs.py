# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify dense row-parallel RHS accumulation against the original serial rows."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton import JointType
from newton._src.solvers.feather_pgs import kernels
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS, _FeatherPGSModelPlan


class TestFeatherPGSDenseRHS(unittest.TestCase):
    def test_model_plan_preserves_single_owner_groups(self):
        """Derive row ownership from actual response plans, including omitted articulations."""
        cases = (
            # name, physical DOFs, worlds, free arts, kinematic arts, response DOFs, eligible
            ("mixed sizes", (2, 6, 3), (0, 1, 2), (1,), (), (2, 6, 3), True),
            ("same-size double owner", (2, 2), (0, 0), (), (), (2, 2), False),
            ("free plus articulated", (6, 2), (0, 0), (0,), (), (6, 2), False),
            ("prescribed free omitted", (6, 2), (0, 0), (0,), (0,), (0, 2), True),
            ("only kinematic free retained", (6,), (0,), (0,), (0,), (6,), True),
            ("zero-DOF art omitted", (0, 2), (0, 0), (), (), (0, 2), True),
            ("world without response", (0, 2), (0, 1), (), (), (0, 2), False),
            ("missing world ID", (2, 2), (0, 2), (), (), (2, 2), False),
            ("empty model", (), (), (), (), (), False),
        )
        device = wp.get_device("cpu")

        def array(values):
            return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)

        for name, sizes, worlds, free, kinematic, response, eligible in cases:
            with self.subTest(name=name):
                count = len(sizes)
                starts = np.concatenate(([0], np.cumsum(sizes))).astype(np.int32)
                coord_sizes = [7 if art in free else size for art, size in enumerate(sizes)]
                coord_starts = np.concatenate(([0], np.cumsum(coord_sizes))).astype(np.int32)
                # Minimal CPU model metadata: one tree joint per articulation.
                # The plan and both grouping methods run without mocking their logic.
                model = SimpleNamespace(
                    articulation_count=count,
                    joint_count=count,
                    joint_dof_count=int(starts[-1]),
                    articulation_world=array(worlds),
                    articulation_start=array(range(count + 1)),
                    joint_parent=array([-1] * count),
                    joint_qd_start=array(starts),
                    joint_q_start=array(coord_starts),
                    joint_type=array(
                        [
                            JointType.FREE if art in free else JointType.D6 if size else JointType.FIXED
                            for art, size in enumerate(sizes)
                        ]
                    ),
                    joint_child=array(range(count)),
                    joint_dof_dim=array([[min(size, 3), max(size - 3, 0)] for size in sizes]),
                    device=device,
                    requires_grad=False,
                )
                mask = np.zeros(model.joint_dof_count, dtype=np.int32)
                for art in kinematic:
                    mask[starts[art] : starts[art + 1]] = 1
                plan = _FeatherPGSModelPlan.build(model, mask, enable_prescribed_response=True)
                self.assertEqual(plan.response_dof_count.tolist(), list(response))
                state = SimpleNamespace(_model_plan=plan)
                SolverFeatherPGS._setup_size_grouping(state, model)
                SolverFeatherPGS._setup_world_mapping(state, model)
                self.assertEqual(state._is_one_solve_art_per_world, eligible)
                if count:
                    self.assertEqual(state.art_to_world.numpy().tolist(), list(worlds))
                grouped_arts = []
                for size in state.size_groups:
                    arts = state.group_to_art[size].numpy().tolist()
                    self.assertEqual(len(arts), state.n_arts_by_size[size])
                    self.assertTrue(all(response[art] == size for art in arts))
                    grouped_arts.extend(arts)
                self.assertEqual(sorted(grouped_arts), np.flatnonzero(np.asarray(response) > 0).tolist())
                if eligible:
                    self.assertEqual(len({worlds[art] for art in grouped_arts}), len(grouped_arts))

                # Exercise the derived guard for every launched size group without
                # acquiring CUDA: only the launch/device shell is replaced.
                state.model = SimpleNamespace(device=SimpleNamespace(is_cuda=True), requires_grad=False)
                state.dense_max_constraints = 8
                state.J_by_size = dict.fromkeys(state.size_groups, object())
                for attr in ("constraint_count", "articulation_dof_start", "v_hat", "rhs"):
                    setattr(state, attr, object())
                for size in state.size_groups:
                    with patch("newton._src.solvers.feather_pgs.solver_feather_pgs.wp.launch") as launch:
                        SolverFeatherPGS._stage4_accumulate_rhs_world(state, size)
                    expected = kernels.rhs_accum_world_par_row if eligible else kernels.rhs_accum_world_par_art
                    self.assertIs(launch.call_args.args[0], expected)
                    self.assertEqual(
                        launch.call_args.kwargs["dim"], state.n_arts_by_size[size] * (8 if eligible else 1)
                    )

    def test_dispatch_preserves_ordering_domains(self):
        """Use row workers only for non-differentiable CUDA single-owner worlds."""
        state = SimpleNamespace(
            model=SimpleNamespace(device=SimpleNamespace(is_cuda=True), requires_grad=False),
            _is_one_solve_art_per_world=True,
            n_arts_by_size={14: 17},
            dense_max_constraints=64,
            group_to_art={14: object()},
            J_by_size={14: object()},
        )
        for name in ("constraint_count", "art_to_world", "articulation_dof_start", "v_hat", "rhs"):
            setattr(state, name, object())
        for cuda, gradients, single, expected_rows in (
            (True, False, True, True),
            (False, False, True, False),
            (True, True, True, False),
            (True, False, False, False),
        ):
            with self.subTest(cuda=cuda, gradients=gradients, single=single):
                state.model.device.is_cuda = cuda
                state.model.requires_grad = gradients
                state._is_one_solve_art_per_world = single
                velocity = object()
                with patch("newton._src.solvers.feather_pgs.solver_feather_pgs.wp.launch") as launch:
                    SolverFeatherPGS._stage4_accumulate_rhs_world(state, 14, velocity)
                expected = kernels.rhs_accum_world_par_row if expected_rows else kernels.rhs_accum_world_par_art
                self.assertIs(launch.call_args.args[0], expected)
                self.assertEqual(launch.call_args.kwargs["dim"], 17 * 64 if expected_rows else 17)
                self.assertEqual(
                    launch.call_args.kwargs["inputs"],
                    [
                        state.constraint_count,
                        64,
                        state.art_to_world,
                        state.articulation_dof_start,
                        velocity,
                        state.group_to_art[14],
                        state.J_by_size[14],
                        14,
                    ],
                )
                self.assertEqual(launch.call_args.kwargs["outputs"], [state.rhs])

    def _compare(self, device, graph):
        """Exercise permuted worlds, nonzero RHS, strided aliases and count changes."""
        rng = np.random.default_rng(7184)
        for capacity, dofs in ((1, 1), (7, 14), (64, 27), (96, 65)):
            worlds = 19
            art_world = rng.permutation(worlds).astype(np.int32)
            groups = rng.permutation(worlds).astype(np.int32)
            starts = (np.arange(worlds + 1) * dofs).astype(np.int32)
            velocity = rng.normal(size=worlds * dofs).astype(np.float32)
            raw_j = rng.normal(size=(worlds, capacity, dofs * 2)).astype(np.float32)
            raw_rhs = rng.normal(size=(worlds, capacity * 2)).astype(np.float32)
            # Exercise CUDA atomic denormal/signed-zero behavior without changing
            # the serial DOF reduction or replacing the atomic add by a plain sum.
            velocity[:dofs] = np.float32(0.0)
            raw_j[0, :, ::2] = np.float32(-0.0)
            raw_rhs[:, 1::2][:, 0] = np.nextafter(np.float32(0.0), np.float32(1.0))
            j_owner = wp.array(raw_j, device=device)
            jacobian = wp.array(
                ptr=j_owner.ptr,
                shape=(worlds, capacity, dofs),
                strides=(j_owner.strides[0], j_owner.strides[1], 8),
                dtype=wp.float32,
                device=device,
                copy=False,
            )
            jacobian._ref = j_owner
            mappings = [wp.array(value, device=device) for value in (art_world, starts, velocity, groups)]
            readonly = [j_owner, *mappings]
            before = [value.numpy().tobytes() for value in readonly]
            rhs_owners = [wp.array(raw_rhs, device=device) for _ in range(2)]
            initial_rhs = wp.array(raw_rhs, device=device)
            rhs = [
                wp.array(
                    ptr=value.ptr + 4,
                    shape=(worlds, capacity),
                    strides=(value.strides[0], 8),
                    dtype=wp.float32,
                    device=device,
                    copy=False,
                )
                for value in rhs_owners
            ]
            for view, owner in zip(rhs, rhs_owners, strict=True):
                view._ref = owner
            counts = wp.array(rng.integers(0, capacity + 1, worlds, dtype=np.int32), device=device)
            stream = wp.Stream(device) if graph else None
            for sweep in range(3):
                if sweep:
                    changed = np.full(worlds, capacity if sweep == 1 else 0, dtype=np.int32)
                    counts.assign(changed)
                if graph:
                    stream.wait_stream(wp.get_stream(device))
                for i, kernel in enumerate((kernels.rhs_accum_world_par_art, kernels.rhs_accum_world_par_row)):
                    wp.copy(rhs_owners[i], initial_rhs, stream=stream)
                    config = {
                        "dim": worlds if i == 0 else worlds * capacity,
                        "inputs": [
                            counts,
                            capacity,
                            mappings[0],
                            mappings[1],
                            mappings[2],
                            mappings[3],
                            jacobian,
                            dofs,
                        ],
                        "outputs": [rhs[i]],
                        "device": device,
                    }
                    if graph:
                        config["stream"] = stream
                    wp.launch(kernel, **config)
                    if graph:
                        wp.synchronize_device(device)
                        wp.copy(rhs_owners[i], initial_rhs, stream=stream)
                        with wp.ScopedCapture(device=device, stream=stream) as capture:
                            wp.launch(kernel, **config)
                        wp.capture_launch(capture.graph, stream=stream)
                if graph:
                    wp.get_stream(device).wait_stream(stream)
                self.assertEqual(rhs_owners[0].numpy().tobytes(), rhs_owners[1].numpy().tobytes())
                self.assertEqual(before, [value.numpy().tobytes() for value in readonly])
                self.assertEqual(rhs_owners[1].numpy()[:, ::2].tobytes(), raw_rhs[:, ::2].tobytes())

    def test_cpu_full_buffers(self):
        """Match the original full arrays for varied sizes and ownership permutations."""
        self._compare(wp.get_device("cpu"), False)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is unavailable")
    def test_cuda_full_buffers(self):
        """Match the original full arrays on the active CUDA device."""
        self._compare(wp.get_device("cuda:0"), False)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is unavailable")
    def test_cuda_nondefault_graph(self):
        """Preserve strided owners and count transitions in a non-default-stream graph."""
        self._compare(wp.get_device("cuda:0"), True)


if __name__ == "__main__":
    unittest.main()
