# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test current-contact zero ownership without changing canonical row readers."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

from newton._src.core.contact_pool import ContactPool
from newton._src.solvers.feather_pgs import compact_contact as compact
from newton._src.solvers.feather_pgs import contact_input, response_builds_rows
from newton.tests import test_feather_pgs_compact_contact as baseline
from newton.tests.feather_pgs_patch_parity import point_row_law
from newton.tests.test_feather_pgs_compact_contact import (
    _FACTOR_FIELDS,
    _FOREST_FIELDS,
    _LAUNCH_FIELDS,
    _LAWS_FIELDS,
    _OPTIONS_FIELDS,
    OUTPUTS,
    fixture,
    install_prefix,
    run_original,
    structs,
)


def clear(data, aux):
    """Invoke the coalesced current-contact clear with the production launch mapping."""
    wp.launch(
        compact.clear_contact_response,
        dim=(2, 256),
        inputs=[aux.counts, aux.bounds, data.dense_group, data.Z],
        block_dim=256,
        device=aux.device,
    )


def produce(data, aux):
    """Publish the contact rows under point friction (the patch law with beta 0 and a disabled view)."""
    law = point_row_law(aux.device, data.point0.shape[0])
    # One thread per row: three per contact.
    wp.launch(compact.produce_contacts, dim=(16, 3), inputs=[*structs(data), law], device=aux.device)


def publish(data, aux):
    """Run all new output owners without host readbacks inside the captured sequence."""
    clear(data, aux)
    produce(data, aux)
    wp.launch(
        response_builds_rows.sparse_pair_whiten_kernel(),
        dim=2 * 12,
        inputs=[
            data.factor,
            data.dense_group,
            aux.counts,
            aux.bounds,
            1,
            12,
            40,
            6,
            data.sparse_dof,
            data.sparse_z,
            data.row_cfm,
        ],
        outputs=[data.Z, data.diag],
        device=aux.device,
    )


class TestCoalescedContact(unittest.TestCase):
    def test_current_contact_range_preserves_prefix_tail_and_groups(self):
        """Clear the dense rows only in current contact ranges through growth, shrink and empty cases."""
        data, aux = fixture()
        pointers = (data.Z.ptr, aux.counts.ptr, aux.bounds.ptr)
        for prefix, counts in (((2, 3), (23, 24)), ((12, 1), (18, 25)), ((1, 12), (25, 15)), ((3, 0), (3, 0))):
            aux.bounds.assign(np.array([[p, p] for p in prefix], dtype=np.int32))
            aux.counts.assign(np.array(counts, dtype=np.int32))
            before = np.arange(480, dtype=np.float32).reshape(2, 40, 6) + 0.5
            data.Z.assign(before)
            clear(data, aux)
            expected = before.copy()
            for world, group in enumerate(data.dense_group.numpy()):
                expected[group, prefix[world] : counts[world]] = 0.0
            np.testing.assert_array_equal(data.Z.numpy(), expected)
            self.assertEqual(pointers, (data.Z.ptr, aux.counts.ptr, aux.bounds.ptr))
            np.testing.assert_array_equal(aux.counts.numpy(), counts)
            np.testing.assert_array_equal(aux.bounds.numpy()[:, 1], prefix)

    def test_raw_key_only_owner_does_not_publish_dense_zeros(self):
        """Leave key-only rows to the preceding clear while dense endpoints still own all coefficients."""
        data, aux = fixture(shared=1)
        data.Z.fill_(713.0)
        produce(data, aux)
        response_dofs = data.response_dofs.numpy()
        arts = (data.art0.numpy(), data.art1.numpy())
        for contact, slot in enumerate(data.slot.numpy()):
            if slot < 0:
                continue
            world = data.world.numpy()[contact]
            group = data.dense_group.numpy()[world]
            dense = any(art[contact] >= 0 and response_dofs[art[contact]] == 6 for art in arts)
            if not dense:
                np.testing.assert_array_equal(data.Z.numpy()[group, slot : slot + 3], 713.0)
            else:
                self.assertTrue(np.isfinite(data.Z.numpy()[group, slot : slot + 3]).all())
                self.assertFalse(np.any(data.Z.numpy()[group, slot : slot + 3] == 713.0))
            self.assertTrue(np.isfinite(data.diag.numpy()[world, slot : slot + 3]).all())

    def test_multi_tile_ranges_and_capacity_clamp(self):
        """Cover every coefficient across repeated cooperative tiles without changing the raw overflow count."""
        data, aux = fixture()
        before = np.arange(2 * 704 * 6, dtype=np.float32).reshape(2, 704, 6) + 0.25
        data.Z = wp.array(before, dtype=float, device="cpu")
        aux.bounds.assign(np.array([[11, 11], [9, 9]], dtype=np.int32))
        aux.counts.assign(np.array([803, 670], dtype=np.int32))
        clear(data, aux)
        expected = before.copy()
        expected[1, 11:704] = 0.0
        expected[0, 9:670] = 0.0
        np.testing.assert_array_equal(data.Z.numpy(), expected)
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
        np.testing.assert_array_equal(data.Z.numpy()[group, slot : slot + 3], 0.0)

    def test_launch_binding_uses_current_counts_before_original_marker(self):
        """Bind the live slot counter, not stale finalized counts, and retain same-stream publication order."""
        data, aux = fixture()
        # Every solver attribute the producer's structs bind, from the fixture's fields.
        values = {name: getattr(data, field) for name, field in _LAUNCH_FIELDS.items()}
        forest = {name: getattr(data, field) for name, field in _FOREST_FIELDS.items()}
        marker = object()
        solver = SimpleNamespace(
            **values,
            forest=SimpleNamespace(**forest, world_count=2),
            slot_counter=aux.counts,
            dense_phase_bounds=aux.bounds,
            constraint_count=wp.zeros(2, dtype=int, device="cpu"),
            model=SimpleNamespace(device="cpu", shape_body=data.shape_body),
            factor=SimpleNamespace(
                **{name: getattr(data, field) for name, field in _FACTOR_FIELDS.items()}, L_by_size={6: data.factor}
            ),
            J_by_size={6: data.Z},
            _mark_independent_sparse_contact_candidates_kernel=marker,
            # Point friction: the patch row law with beta 0 and a disabled view.
            laws=SimpleNamespace(
                **{name: getattr(data, field) for name, field in _LAWS_FIELDS.items()},
                friction_anchors_enabled=False,
                contact_torsion_enabled=False,
                friction_anchor_beta=0.0,
                friction_patches=SimpleNamespace(view=point_row_law("cpu", 16).patches),
            ),
            options=SimpleNamespace(**{name: getattr(data, field) for name, field in _OPTIONS_FIELDS.items()}),
            # Launches sized by contact rows go through the pool, eagerly here.
            _contact_pool=ContactPool(device="cpu", rows=16),
        )
        contacts = SimpleNamespace(
            rigid_contact_max=16,
            **{
                "rigid_contact_" + name: getattr(data, name)
                for name in ("count", "point0", "point1", "normal", "shape0", "shape1", "margin0", "margin1")
            },
        )
        with mock.patch.object(compact.wp, "launch") as launch:
            bound = contact_input.bind(
                solver,
                SimpleNamespace(body_q=data.body_q),
                SimpleNamespace(body_v_s=data.body_v, joint_S_s=data.motion),
                contacts,
                0.0,
                1.0,
            )
            compact.launch_contacts(solver, bound, 16)
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
                    data.Z.fill_(float("nan"))
                    data.sparse_dof.fill_(-35)
                    install_prefix(data, aux)
                    wp.capture_launch(graph)
                    compare(original, data, aux)
                    self.assertEqual(owners, {name: getattr(data, name).ptr for name in OUTPUTS})


if __name__ == "__main__":
    unittest.main()
