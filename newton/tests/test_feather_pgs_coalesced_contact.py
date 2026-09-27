# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test current-contact zero ownership without changing canonical row readers."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import compact_contact as compact
from newton.tests import test_feather_pgs_compact_contact as baseline
from newton.tests.test_feather_pgs_compact_contact import OUTPUTS, fixture, install_prefix, run_original


def clear(data, aux):
    """Invoke the coalesced current-contact clear with the production launch mapping."""
    wp.launch(
        compact.clear_contact_response,
        dim=(2, 256),
        inputs=[aux.counts, aux.bounds, data.dense_group, data.J, data.Y],
        block_dim=256,
        device=aux.device,
    )


def publish(data, aux):
    """Run all new output owners without host readbacks inside the captured sequence."""
    clear(data, aux)
    wp.launch(compact.produce_contacts, dim=16, inputs=[data], device=aux.device)
    wp.launch(
        compact.produce_limit_response,
        dim=(2, 12),
        inputs=[aux.bounds, aux.counts, data.dense_group, data.factor, data.J, data.Y, data.row_cfm, data.diag],
        device=aux.device,
    )


class TestCoalescedContact(unittest.TestCase):
    def test_current_contact_range_preserves_prefix_tail_and_groups(self):
        """Clear both outputs only in current contact ranges through growth, shrink and empty cases."""
        data, aux = fixture()
        pointers = (data.J.ptr, data.Y.ptr, aux.counts.ptr, aux.bounds.ptr)
        for prefix, counts in (((2, 3), (23, 24)), ((12, 1), (18, 25)), ((1, 12), (25, 15)), ((3, 0), (3, 0))):
            aux.bounds.assign(np.array([[p, p] for p in prefix], dtype=np.int32))
            aux.counts.assign(np.array(counts, dtype=np.int32))
            before_j = np.arange(480, dtype=np.float32).reshape(2, 40, 6) + 0.5
            before_y = -before_j - 1.0
            data.J.assign(before_j)
            data.Y.assign(before_y)
            clear(data, aux)
            for actual, before in ((data.J, before_j), (data.Y, before_y)):
                expected = before.copy()
                for world, group in enumerate(data.dense_group.numpy()):
                    expected[group, prefix[world] : counts[world]] = 0.0
                np.testing.assert_array_equal(actual.numpy(), expected)
            self.assertEqual(pointers, (data.J.ptr, data.Y.ptr, aux.counts.ptr, aux.bounds.ptr))
            np.testing.assert_array_equal(aux.counts.numpy(), counts)
            np.testing.assert_array_equal(aux.bounds.numpy()[:, 1], prefix)

    def test_raw_key_only_owner_does_not_publish_dense_zeros(self):
        """Leave key-only J/Y to the preceding clear while dense endpoints still own all coefficients."""
        data, aux = fixture(shared=1)
        data.J.fill_(713.0)
        data.Y.fill_(-919.0)
        wp.launch(compact.produce_contacts, dim=16, inputs=[data], device=aux.device)
        response_dofs = data.response_dofs.numpy()
        arts = (data.art0.numpy(), data.art1.numpy())
        for contact, slot in enumerate(data.slot.numpy()):
            if slot < 0:
                continue
            world = data.world.numpy()[contact]
            group = data.dense_group.numpy()[world]
            dense = any(art[contact] >= 0 and response_dofs[art[contact]] == 6 for art in arts)
            if not dense:
                np.testing.assert_array_equal(data.J.numpy()[group, slot : slot + 3], 713.0)
                np.testing.assert_array_equal(data.Y.numpy()[group, slot : slot + 3], -919.0)
            else:
                self.assertTrue(np.isfinite(data.J.numpy()[group, slot : slot + 3]).all())
                self.assertFalse(np.any(data.J.numpy()[group, slot : slot + 3] == 713.0))
            self.assertTrue(np.isfinite(data.diag.numpy()[world, slot : slot + 3]).all())

    def test_multi_tile_ranges_and_capacity_clamp(self):
        """Cover every coefficient across repeated cooperative tiles without changing the raw overflow count."""
        data, aux = fixture()
        before_j = np.arange(2 * 704 * 6, dtype=np.float32).reshape(2, 704, 6) + 0.25
        before_y = -before_j - 2.0
        data.J = wp.array(before_j, dtype=float, device="cpu")
        data.Y = wp.array(before_y, dtype=float, device="cpu")
        aux.bounds.assign(np.array([[11, 11], [9, 9]], dtype=np.int32))
        aux.counts.assign(np.array([803, 670], dtype=np.int32))
        clear(data, aux)
        for actual, before in ((data.J, before_j), (data.Y, before_y)):
            expected = before.copy()
            expected[1, 11:704] = 0.0
            expected[0, 9:670] = 0.0
            np.testing.assert_array_equal(actual.numpy(), expected)
        # The allocator's original sticky status/finalizer still owns rejection;
        # this bounded writer must neither write out of bounds nor hide demand.
        np.testing.assert_array_equal(aux.counts.numpy(), [803, 670])

    def test_complete_clear_and_producer_match_original_chain(self):
        """Compare every active canonical field against the unchanged seven-kernel control."""
        compare = baseline.TestCompactContact().compare
        for shared, friction_shared in ((0, 0), (0, 1), (1, 0)):
            original, aux = fixture(shared=shared, friction_shared=friction_shared)
            data, _ = fixture(shared=shared, friction_shared=friction_shared)
            run_original(original, aux)
            install_prefix(data, aux)
            publish(data, aux)
            compare(original, data, aux)

    def test_dense_tail_and_cancelling_dense_endpoint(self):
        """Retain complete dense ownership for a contiguous tail, including exactly cancelling endpoints."""
        original, aux = fixture(shared=1)
        data, _ = fixture(shared=1)
        response = original.response_dofs.numpy()
        art0, art1 = original.art0.numpy(), original.art1.numpy()
        dense = ((art0 >= 0) & (response[np.maximum(art0, 0)] == 6)) | (
            (art1 >= 0) & (response[np.maximum(art1, 0)] == 6)
        )
        dense &= original.slot.numpy() >= 0
        order = np.argsort(dense, kind="stable")
        names = (
            "point0",
            "point1",
            "normal",
            "shape0",
            "shape1",
            "margin0",
            "margin1",
            "world",
            "slot",
            "art0",
            "art1",
            "path",
            "slots_needed",
        )
        for name in names:
            values = getattr(original, name).numpy()[order].copy()
            getattr(original, name).assign(values)
            getattr(data, name).assign(values)
        first = int(np.flatnonzero(dense[order])[0])
        self.assertTrue(np.all(dense[order][first:]))
        for target in (original, data):
            shapes = target.shape1.numpy()
            shapes[first] = target.shape0.numpy()[first]
            target.shape1.assign(shapes)
            arts = target.art1.numpy()
            arts[first] = target.art0.numpy()[first]
            target.art1.assign(arts)
        run_original(original, aux)
        install_prefix(data, aux)
        publish(data, aux)
        baseline.TestCompactContact().compare(original, data, aux)
        world, slot = data.world.numpy()[first], data.slot.numpy()[first]
        group = data.dense_group.numpy()[world]
        np.testing.assert_array_equal(data.J.numpy()[group, slot : slot + 3], 0.0)
        np.testing.assert_array_equal(data.Y.numpy()[group, slot : slot + 3], 0.0)

    def test_launch_binding_uses_current_counts_before_original_marker(self):
        """Bind the live slot counter, not stale finalized counts, and retain same-stream publication order."""
        data, aux = fixture()
        mapping = {
            "contact_world": "world",
            "contact_slot": "slot",
            "contact_art_a": "art0",
            "contact_art_b": "art1",
            "contact_path": "path",
            "contact_slots_needed": "slots_needed",
            "body_response_dof_mask": "body_mask",
            "body_single_response_dof": "body_single_dof",
            "articulation_response_dof_count": "response_dofs",
            "articulation_dof_start": "dof_start",
            "articulation_world_dof_offset": "dof_offset",
            "articulation_origin": "origin",
            "_prescribed_articulation": "prescribed",
            "shape_material_mu": "material_mu",
            "shape_material_restitution": "material_restitution",
            "_diagonal_inverse_mass": "inverse_mass",
            "_sparse_diagonal_dense_groups": "dense_group",
            "_sparse_diagonal_response_size": "sparse_size",
            "contact_friction_scale": "friction_scale",
            "pgs_beta": "beta",
            "pgs_cfm": "cfm",
            "target_velocity": "target",
            "row_restitution": "restitution",
            "_sparse_diagonal_row_dof": "sparse_dof",
            "_sparse_diagonal_row_jy": "sparse_jy",
            "contact_shared_anchor": "shared_anchor",
            "contact_friction_shared_anchor": "friction_shared_anchor",
        }
        values = {name: getattr(data, source) for name, source in mapping.items()}
        values.update(
            {
                name: getattr(data, name)
                for name in ("row_type", "row_parent", "row_mu", "row_beta", "row_cfm", "phi", "diag")
            }
        )
        marker = object()
        solver = SimpleNamespace(
            **values,
            world_count=2,
            slot_counter=aux.counts,
            dense_phase_bounds=aux.bounds,
            constraint_count=wp.zeros(2, dtype=int, device="cpu"),
            model=SimpleNamespace(device="cpu", shape_body=data.shape_body),
            L_by_size={6: data.factor},
            J_by_size={6: data.J},
            Y_by_size={6: data.Y},
            _mark_independent_sparse_contact_candidates_kernel=marker,
        )
        contacts = SimpleNamespace(
            rigid_contact_max=16,
            **{
                "rigid_contact_" + name: getattr(data, name)
                for name in ("count", "point0", "point1", "normal", "shape0", "shape1", "margin0", "margin1")
            },
        )
        with mock.patch.object(compact.wp, "launch") as launch:
            compact.launch_contacts(
                solver,
                SimpleNamespace(body_q=data.body_q),
                SimpleNamespace(body_v_s=data.body_v, joint_S_s=data.motion),
                contacts,
                16,
            )
        self.assertEqual(
            [call.args[0] for call in launch.call_args_list],
            [compact.clear_contact_response, compact.produce_contacts, marker],
        )
        call = launch.call_args_list[0]
        self.assertIs(call.kwargs["inputs"][0], aux.counts)
        self.assertIsNot(call.kwargs["inputs"][0], solver.constraint_count)
        self.assertIs(call.kwargs["inputs"][1], aux.bounds)
        self.assertEqual(call.kwargs["dim"], (2, 256))
        self.assertEqual(call.kwargs["block_dim"], 256)
        self.assertTrue(all(call.kwargs["device"] == "cpu" for call in launch.call_args_list))

    def test_cuda_poisoned_growing_ranges_and_two_graphs(self):
        """Replay both current owners across poisoned changing prefixes without replacing graph outputs."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Actual graph replay requires CUDA")
        compare = baseline.TestCompactContact().compare
        for device in devices:
            data, aux = fixture(device)
            install_prefix(data, aux)
            publish(data, aux)
            graphs = []
            for _ in range(2):
                with wp.ScopedCapture(device=device) as capture:
                    publish(data, aux)
                graphs.append(capture.graph)
            owners = {name: getattr(data, name).ptr for name in OUTPUTS}
            for prefix, kept in (
                ((2, 3), (7, 7)),
                ((0, 12), (0, 2)),
                ((12, 0), (7, 0)),
                ((1, 1), (7, 7)),
                ((0, 0), (0, 0)),
            ):
                original, other = fixture(device, prefix=prefix, kept=kept)
                data.slot.assign(original.slot)
                data.path.assign(original.path)
                aux.counts.assign(other.counts)
                aux.bounds.assign(other.bounds)
                run_original(original, other)
                for graph in graphs:
                    data.J.fill_(float("nan"))
                    data.Y.fill_(float("nan"))
                    data.sparse_dof.fill_(-35)
                    install_prefix(data, aux)
                    wp.capture_launch(graph)
                    compare(original, data, aux)
                    self.assertEqual(owners, {name: getattr(data, name).ptr for name in OUTPUTS})


if __name__ == "__main__":
    unittest.main()
