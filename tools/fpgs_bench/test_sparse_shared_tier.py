# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check private row capacity and the original sparse owner's fork/join boundary.

The author's pre-change control (session92387) rejected shared_row_capacity
with TypeError. These tests retain public704 and the original eight sweeps.
"""

import ast
import inspect
import os
import types
import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import solver_feather_pgs as implementation
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
)
from newton.tests.test_feather_pgs_response_diagonal import _build_mixed_response_model


def factory(**kwargs):
    """Construct the actual public704 sparse owner without launching it."""
    return implementation._get_pgs_solve_sparse_diagonal_kernel(
        704, 114, 6, "120", contact_triples=True, speculative_contact_batches=True, **kwargs
    )


def native_source(**kwargs):
    """Capture the generated native text at the existing decorator boundary."""
    implementation._get_pgs_solve_sparse_diagonal_kernel.cache_clear()
    with mock.patch.object(wp, "func_native", wraps=wp.func_native) as decorator:
        kernel = factory(**kwargs)
    return kernel, decorator.call_args.args[0]


class TestSparseSharedTier(unittest.TestCase):
    def test_native_fixture_binds_required_production_arguments(self):
        """Bind the actual native fixture call before a GPU launch is needed."""
        tree = ast.parse(inspect.getsource(TestSparseSharedTierCUDA))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_launch_matrix_free_gs_solve"
        ]
        self.assertEqual(len(calls), 1)
        signature = inspect.signature(implementation.SolverFeatherPGS._launch_matrix_free_gs_solve)
        signature.bind(None, **{keyword.arg: None for keyword in calls[0].keywords})

    def test_private_capacity_preserves_public_strides_and_original_math(self):
        """Change only private arrays and disjoint admission in emitted code."""
        old, original = native_source()
        for capacity, minimum in ((384, 0), (704, 385)):
            kernel, source = native_source(shared_row_capacity=capacity, min_row_count=minimum)
            self.assertTrue(kernel.key.endswith(f"_shared{capacity}_rows{minimum}_{capacity}"))
            self.assertEqual([arg.label for arg in old.adj.args], [arg.label for arg in kernel.adj.args])
            guard = f"    if (row_count > {capacity}) return;"
            if minimum:
                guard += f"\n    if (row_count < {minimum}) return;"
            self.assertLess(source.index(guard), source.index("const int row_base"))
            restored = source.replace("\n" + guard, "", 1)
            for name in ("s_lambda", "s_rhs", "s_diag", "s_meta", "s_mu"):
                restored = restored.replace(f"{name}[{capacity}]", f"{name}[704]")
            self.assertEqual(restored, original)
            self.assertIn("world * 704", source)
            self.assertIn("world * 235 + candidate_index", source)
        self.assertNotIn("_shared", old.key)
        for kwargs in (
            {"shared_row_capacity": 0},
            {"shared_row_capacity": 705},
            {"shared_row_capacity": 384, "min_row_count": 385},
            {"min_row_count": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                factory(**kwargs)

    def test_default_off_and_explicit_unsupported_constructor(self):
        """Keep the ordinary constructor off and reject unsupported opt-in."""
        model = _build_mixed_response_model("cpu", dof_count=3)
        with mock.patch.dict(os.environ, {"FEATHER_PGS_SPARSE_SHARED_TIER": "0"}):
            solver = implementation.SolverFeatherPGS(model, dense_max_constraints=32)
        self.assertFalse(solver._sparse_shared_tier)
        self.assertIsNone(solver._pgs_solve_sparse_diagonal_tail_kernel)
        with (
            mock.patch.dict(os.environ, {"FEATHER_PGS_SPARSE_SHARED_TIER": "1"}),
            self.assertRaisesRegex(ValueError, "CUDA 704-row"),
        ):
            implementation.SolverFeatherPGS(model, dense_max_constraints=32)

    def test_actual_dispatch_forwards_aliases_and_joins_on_failure(self):
        """Order ready, two owners and done even when a launch raises."""
        method = implementation.SolverFeatherPGS._launch_sparse_shared_tier
        for failure in (None, "tail", "bulk"):
            with self.subTest(failure=failure):
                events = []

                class Stream:
                    def __init__(self, name, event_log):
                        self.name = name
                        self.events = event_log

                    def record_event(self, event):
                        self.events.append((self.name, "record", event))

                    def wait_event(self, event):
                        self.events.append((self.name, "wait", event))

                main, side = Stream("main", events), Stream("side", events)
                owner = types.SimpleNamespace(
                    _size_streams={6: side},
                    _size_events={6: "ready", 108: "done"},
                    _pgs_solve_sparse_diagonal_tail_kernel="tail",
                )
                inputs, outputs = [object()], [object()]

                def launch(
                    kernel, _inputs=inputs, _outputs=outputs, _side=side, _events=events, _failure=failure, **kwargs
                ):
                    self.assertIs(kwargs["inputs"], _inputs)
                    self.assertIs(kwargs["outputs"], _outputs)
                    self.assertEqual(kwargs["dim"], [5])
                    self.assertEqual(kwargs["block_dim"], 32)
                    self.assertIs(kwargs.get("stream"), _side if kernel == "tail" else None)
                    _events.append(("launch", kernel))
                    if kernel == _failure:
                        raise RuntimeError("authored launch failure")

                with (
                    mock.patch.object(wp, "get_stream", return_value=main),
                    mock.patch.object(wp, "launch_tiled", side_effect=launch),
                ):
                    args = {"dim": [5], "inputs": inputs, "outputs": outputs, "block_dim": 32, "device": "test"}
                    if failure:
                        with self.assertRaisesRegex(RuntimeError, "authored"):
                            method(owner, "bulk", **args)
                    else:
                        method(owner, "bulk", **args)
                expected = [("main", "record", "ready"), ("side", "wait", "ready"), ("launch", "tail")]
                if failure != "tail":
                    expected.append(("launch", "bulk"))
                expected += [("side", "record", "done"), ("main", "wait", "done")]
                self.assertEqual(events, expected)
        source = inspect.getsource(implementation.SolverFeatherPGS._launch_matrix_free_gs_solve)
        self.assertIn("self._launch_sparse_shared_tier if self._sparse_shared_tier else wp.launch_tiled", source)


def fixture_arrays(counts):
    """Embed original scalar/coupled contact laws in the actual704/114/6 ABI."""
    worlds, rows, dofs, dense = 5, 704, 114, 6
    counts = np.asarray(counts, dtype=np.int32)
    prefix = counts % 3
    inverse = np.linspace(0.5, 1.0, worlds * dofs, dtype=np.float32)
    initial = np.linspace(-0.2, 0.2, worlds * dofs, dtype=np.float32)
    arrays = {
        "constraint_count": counts,
        "world_dof_indices": np.arange(worlds * dofs, dtype=np.int32).reshape(worlds, dofs),
        "rhs": np.zeros((worlds, rows), np.float32),
        "diag": np.ones((worlds, rows), np.float32),
        "impulses": np.zeros((worlds, rows), np.float32),
        "row_type": np.full((worlds, rows), PGS_CONSTRAINT_TYPE_JOINT_LIMIT, np.int32),
        "row_parent": np.full((worlds, rows), -1, np.int32),
        "row_mu": np.zeros((worlds, rows), np.float32),
        "dense_phase_bounds": np.column_stack((np.zeros(worlds, np.int32), prefix)),
        "_sparse_contact_group_count": np.zeros(worlds, np.int32),
        "_sparse_contact_group_heads": np.full((worlds, dofs), -1, np.int32),
        "_sparse_contact_serial_count": np.zeros(worlds, np.int32),
        "_sparse_contact_serial_normals": np.full((worlds, 235), -1, np.int32),
        "_sparse_diagonal_dense_offsets": np.zeros(worlds, np.int32),
        "_sparse_diagonal_dense_groups": np.arange(worlds - 1, -1, -1, dtype=np.int32),
        "J": np.zeros((worlds, rows, dense), np.float32),
        "Y": np.zeros((worlds, rows, dense), np.float32),
        "_sparse_diagonal_row_dof": np.full((worlds, rows, 2), -1, np.int32),
        "_sparse_diagonal_row_jy": np.zeros((worlds, rows, 4), np.float32),
        "_fused_diagonal_limit_active": np.zeros((worlds, dofs), np.int32),
        "_fused_diagonal_limit_lower_rhs": np.zeros((worlds, dofs), np.float32),
        "_fused_diagonal_limit_upper_rhs": np.zeros((worlds, dofs), np.float32),
        "_diagonal_inverse_mass": inverse,
        "_fused_diagonal_limit_lower_lambda": np.full((worlds, dofs), -19.0, np.float32),
        "_fused_diagonal_limit_upper_lambda": np.full((worlds, dofs), -23.0, np.float32),
        "v_out": initial,
    }
    for world, count in enumerate(counts):
        group = worlds - world - 1
        h = inverse[world * dofs : (world + 1) * dofs]
        arrays["_fused_diagonal_limit_active"][world, (6, 112, 113)] = (1, 1, 2)
        arrays["_fused_diagonal_limit_lower_rhs"][world, (6, 112)] = (-0.15, -0.4)
        arrays["_fused_diagonal_limit_upper_rhs"][world, 113] = -0.4
        for row in range(int(prefix[world])):
            arrays["J"][group, row, row] = 1.0
            arrays["Y"][group, row, row] = h[row]
            arrays["rhs"][world, row] = -0.1
            arrays["diag"][world, row] = h[row] + 1.0e-6
        for normal in range(int(prefix[world]), int(count), 3):
            independent = normal < prefix[world] + 6
            jacobian = np.zeros((3, dofs), np.float32)
            if independent:
                jacobian[:, 13] = (1.0, 0.2, -0.3)
                arrays["_sparse_diagonal_row_dof"][world, normal : normal + 3, 0] = 13
                arrays["_sparse_diagonal_row_dof"][world, normal, 1] = -2
            else:
                jacobian[:, (0, 1, 6, 7)] = (
                    (0.6, -0.4, 0.2, 0.0),
                    (0.1, 0.5, -0.3, 0.4),
                    (-0.2, 0.25, 0.1, -0.35),
                )
                arrays["_sparse_diagonal_row_dof"][world, normal : normal + 3] = (6, 7)
            response = jacobian * h
            arrays["J"][group, normal : normal + 3] = jacobian[:, :dense]
            arrays["Y"][group, normal : normal + 3] = response[:, :dense]
            for component in range(3):
                row = normal + component
                for slot, coord in enumerate(arrays["_sparse_diagonal_row_dof"][world, row]):
                    if coord >= 0:
                        arrays["_sparse_diagonal_row_jy"][world, row, 2 * slot : 2 * slot + 2] = (
                            jacobian[component, coord],
                            response[component, coord],
                        )
            arrays["diag"][world, normal : normal + 3] = np.sum(jacobian * response, axis=1) + 1.0e-6
            arrays["row_type"][world, normal : normal + 3] = (
                PGS_CONSTRAINT_TYPE_CONTACT,
                PGS_CONSTRAINT_TYPE_FRICTION,
                PGS_CONSTRAINT_TYPE_FRICTION,
            )
            arrays["row_parent"][world, normal + 1 : normal + 3] = normal
            arrays["row_mu"][world, normal + 1 : normal + 3] = 0.7
            # Load the final complete triple as well as the independent chain.
            arrays["rhs"][world, normal] = -0.1 if independent or normal == count - 3 else 2.0
    return arrays


def bind_owner(host, device, tiered):
    """Bind the real solver launch method to only its required production fields."""
    arrays = {name: wp.array(value, device=device) for name, value in host.items()}
    owner = types.SimpleNamespace(**arrays)
    owner.model = types.SimpleNamespace(device=device)
    owner.world_count = 5
    owner._sparse_factor = None
    owner._sparse_diagonal_contact_solve = True
    owner._sparse_diagonal_dense_size = 6
    owner._sparse_shared_tier = tiered
    owner.pgs_cfm = 1.0e-6
    owner.J_by_size, owner.Y_by_size = {6: arrays["J"]}, {6: arrays["Y"]}
    owner._pgs_solve_sparse_diagonal_kernel = factory(**({"shared_row_capacity": 384} if tiered else {}))
    owner._pgs_solve_sparse_diagonal_tail_kernel = factory(shared_row_capacity=704, min_row_count=385)
    owner._size_streams = {6: wp.Stream(device)}
    owner._size_events = {6: wp.Event(device), 108: wp.Event(device)}
    owner._launch_sparse_shared_tier = types.MethodType(
        implementation.SolverFeatherPGS._launch_sparse_shared_tier, owner
    )
    return owner, arrays


@unittest.skipUnless(wp.is_cuda_available(), "native shared tier requires CUDA")
class TestSparseSharedTierCUDA(unittest.TestCase):
    def test_actual_fork_join_boundary_counts_and_captured_growth(self):
        """Match original704 on both tier boundaries and replay current counts."""
        device = wp.get_device("cuda:0")
        initial_counts = (0, 383, 384, 385, 704)
        host = fixture_arrays(initial_counts)
        owners = [bind_owner(host, device, tiered) for tiered in (False, True)]
        written = ("impulses", "_fused_diagonal_limit_lower_lambda", "_fused_diagonal_limit_upper_lambda", "v_out")
        for owner, arrays in owners:
            owner.observed = {name: wp.empty_like(arrays[name]) for name in written}
        builder = implementation._get_build_independent_sparse_contact_groups_kernel(
            704, 114, str(device.arch), build_serial_contacts=True
        )

        def execute(owner):
            wp.launch(
                builder,
                dim=5 * 32,
                inputs=[owner.constraint_count, owner.dense_phase_bounds],
                outputs=[
                    owner._sparse_diagonal_row_dof,
                    owner._sparse_contact_group_count,
                    owner._sparse_contact_group_heads,
                    owner._sparse_contact_serial_count,
                    owner._sparse_contact_serial_normals,
                ],
                device=device,
            )
            implementation.SolverFeatherPGS._launch_matrix_free_gs_solve(
                owner, dense_rhs=owner.rhs, mf_meta=None, iterations=8, omega=1.0, friction_start_iteration=2
            )
            # A real same-stream consumer belongs after the tail join, including
            # inside the graph; a later host readback must not hide a missing join.
            for name in written:
                wp.copy(owner.observed[name], getattr(owner, name))

        schedule = (
            "_sparse_diagonal_row_dof",
            "_sparse_contact_group_count",
            "_sparse_contact_group_heads",
            "_sparse_contact_serial_count",
            "_sparse_contact_serial_normals",
        )

        def compare(current):
            for name in schedule:
                np.testing.assert_array_equal(owners[1][1][name].numpy(), owners[0][1][name].numpy(), err_msg=name)
            for name in written:
                # Retain the original scalar/speculative physical tolerances;
                # private storage need not imply bit-identical floating codegen.
                np.testing.assert_allclose(
                    owners[1][1][name].numpy(),
                    owners[0][1][name].numpy(),
                    rtol=2.0e-5,
                    atol=2.0e-6,
                    err_msg=name,
                )
                np.testing.assert_allclose(
                    owners[1][0].observed[name].numpy(),
                    owners[0][1][name].numpy(),
                    rtol=2.0e-5,
                    atol=2.0e-6,
                    err_msg=f"joined {name}",
                )
            for _, arrays in owners:
                for name, value in current.items():
                    if name not in written + schedule:
                        np.testing.assert_array_equal(arrays[name].numpy(), value, err_msg=f"readonly {name}")
            velocity = owners[1][1]["v_out"].numpy().reshape(5, 114)
            self.assertTrue(np.isfinite(velocity).all())
            # Count zero must still execute lower/upper fused-limit updates.
            np.testing.assert_allclose(velocity[:, 112], 0.4, rtol=2.0e-5, atol=2.0e-6)
            np.testing.assert_allclose(velocity[:, 113], -0.4, rtol=2.0e-5, atol=2.0e-6)
            for world, count in enumerate(current["constraint_count"]):
                if count:
                    impulse = float(owners[1][1]["impulses"].numpy()[world, count - 3])
                    if world == 1 and tuple(current["constraint_count"]) == (704, 385, 384, 383, 0):
                        # Prefix1 leaves v1 negative: the CPU-proven normal
                        # residual is +0.03766, so this reversed case stays cold.
                        self.assertEqual(impulse, 0.0)
                    else:
                        self.assertGreater(impulse, 0.0)

        for owner, _ in owners:
            execute(owner)
        compare(host)
        # Warm both kernels/events before capturing the same production fork.
        wp.synchronize_device(device)
        graphs = []
        for owner, arrays in owners:
            for name, value in host.items():
                arrays[name].assign(value)
            with wp.ScopedCapture(device=device) as capture:
                execute(owner)
            graphs.append(capture.graph)
        for counts in ((704, 385, 384, 383, 0), (0, 0, 0, 0, 0), initial_counts):
            current = fixture_arrays(counts)
            for (_owner, arrays), graph in zip(owners, graphs, strict=True):
                for name, value in current.items():
                    arrays[name].assign(value)
                wp.capture_launch(graph)
            compare(current)
        # Explicitly release all graph/stream owners while their context lives.
        wp.synchronize_device(device)


if __name__ == "__main__":
    unittest.main()
