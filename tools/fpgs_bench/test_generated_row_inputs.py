# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generated operands must not depend on robot dimensions or factory source text."""

import inspect
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import keyed_rows, franka_row_packets
from newton._src.solvers.feather_pgs.franka_contact_packet import ContactPacketData
from newton._src.solvers.feather_pgs.row_inputs import CompactLocalRows, RowArgument
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_pgs_solve_parallel_kernel


class TestGeneratedRowInputs(unittest.TestCase):
    def test_packet_factory_without_source_inspection(self):
        franka_row_packets.get_local_factory.cache_clear()
        with patch.object(inspect, "getsource", side_effect=AssertionError("factory source is not an API")):
            self.assertTrue(callable(franka_row_packets.get_local_factory()))

    def test_keyed_factory_without_source_inspection(self):
        keyed_rows.get_parallel_factory.cache_clear()
        with patch.object(inspect, "getsource", side_effect=AssertionError("factory source is not an API")):
            self.assertTrue(callable(keyed_rows.get_parallel_factory(_get_pgs_solve_parallel_kernel)))

    @unittest.skipUnless(wp.is_cuda_available(), "native local solving requires CUDA")
    def test_non_robot_layouts_against_independent_solve_oracles(self):
        """Reuse physical GS oracles with sparse prefix/packet operands and poisoned J.

        These fixtures cover 2, 3, and paired coordinates, held nonidentity mass,
        bilateral/limit rows, coupled friction, queue tails, subwarps and MF work.
        The decoder's prefix deliberately repeats a coordinate to check addition.
        """
        from newton.tests import test_feather_pgs_local_internal_solve as oracle  # noqa: PLC0415

        ordinary_factory = oracle._get_pgs_solve_local_owned_kernel
        ordinary_launch = wp.launch_tiled

        def factory(max_rows, local_rows, dofs, arch, paired_dof_count=0, **kwargs):
            layout = CompactLocalRows(
                dofs,
                paired_dof_count,
                max_rows,
                max_rows,
                dofs + 1,
                (RowArgument("prefix", franka_row_packets.PrefixData), RowArgument("packets", ContactPacketData)),
            )
            kernel = ordinary_factory(max_rows, local_rows, dofs, arch, paired_dof_count, row_inputs=layout, **kwargs)
            kernel._test_row_layout = layout
            return kernel

        def launch(kernel, **kwargs):
            layout = getattr(kernel, "_test_row_layout", None)
            if layout is None:
                return ordinary_launch(kernel, **kwargs)
            args = list(kwargs.pop("inputs", [])) + list(kwargs.pop("outputs", []))
            names = [arg.label for arg in kernel.adj.args][:-2]
            data = dict(zip(names, args, strict=True))
            dev = kwargs["device"]
            worlds = len(data["world_constraint_count"])
            capacity, dofs, secondary = layout.packet_capacity, layout.primary_dofs, layout.secondary_dofs
            packet = np.zeros((worlds, capacity, dofs + secondary), np.float32)
            groups, art_world = data["articulation_group_index"].numpy(), data["articulation_world"].numpy()
            ja, jb = data["J_group"].numpy(), data["secondary_J_group"].numpy()
            for candidate, art in enumerate(data["candidate_articulations"].numpy()):
                world = art_world[art]
                packet[world, :, :dofs] = ja[groups[art]]
                if secondary:
                    other = data["candidate_secondary_articulations"].numpy()[candidate]
                    packet[world, :, dofs:] = jb[groups[other]]
            prefix_mask = np.broadcast_to(np.arange(capacity) % 2 == 0, (worlds, capacity)).copy()
            if secondary:
                prefix_mask &= np.all(packet[:, :, dofs:] == 0, axis=2)
            prefix, packets = franka_row_packets.PrefixData(), ContactPacketData()
            coordinates = np.broadcast_to(np.r_[np.arange(dofs), 0], (worlds, capacity, dofs + 1)).astype(np.int32)
            coefficients = np.concatenate((packet[:, :, :dofs].copy(), packet[:, :, :1] * 0.5), axis=2)
            coefficients[:, :, 0] *= 0.5
            prefix.dofs = wp.array(coordinates, dtype=int, device=dev)
            prefix.weights = wp.array(coefficients, dtype=float, device=dev)
            packets.row_contact = wp.array(np.where(prefix_mask, -1, 0), dtype=int, device=dev)
            packets.jacobian = wp.array(packet, dtype=float, device=dev)
            for name in ("J_group", "secondary_J_group"):
                args[names.index(name)] = wp.full(data[name].shape, float("nan"), dtype=float, device=dev)
            return ordinary_launch(kernel, inputs=[*args, prefix, packets], **kwargs)

        methods = (
            "test_local_kernel_matches_sequential_pgs_and_respects_world_ownership",
            "test_local_kernel_waits_for_delayed_friction_activation",
            "test_local_kernel_uses_patch_load_and_coupled_tangent_response",
            "test_local_residual_queue_matches_sequential_dense_and_matrix_free_pgs",
        )
        with (
            patch.object(oracle, "_get_pgs_solve_local_owned_kernel", factory),
            patch.object(wp, "launch_tiled", launch),
        ):
            for method in methods:
                with self.subTest(oracle=method):
                    getattr(oracle.TestFeatherPGSLocalInternalSolve(), method)()


if __name__ == "__main__":
    unittest.main()
