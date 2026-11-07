# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve buffered cylinder contacts across split convex-query stages."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.geometry.contact_reduction_global import decode_oct
from newton._src.geometry.narrow_phase import NarrowPhase, write_contact_simple
from newton._src.geometry.support_function import SupportMapDataProvider
from newton.tests import test_heightfield_primitive_query as primitive
from newton.tests.test_heightfield_primitive_cull import primitive_model

_BASE_PIPELINE = primitive.pipeline


def split_pipeline(model, enabled):
    """Compare split and fused cylinders with the same typed-query owner."""
    with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_CYLINDER_SPLIT": str(int(enabled))}):
        return _BASE_PIPELINE(model, True)


class TestHeightfieldCylinderSplit(unittest.TestCase):
    def test_admission_cpu(self):
        """Require the original typed owner and an explicit valid split option."""
        flags = {**primitive.FLAGS, "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "1"}

        def create(**options):
            options = {"has_heightfields": True, "has_meshes": False, **options}
            return NarrowPhase(max_candidate_pairs=4, max_triangle_pairs=33, device="cpu", **options)

        with patch.dict(os.environ, {**flags, "NEWTON_HEIGHTFIELD_CYLINDER_SPLIT": "1"}):
            owner = create()
            self.assertTrue(owner._heightfield_cylinder_split)
            self.assertEqual(len(owner._heightfield_cylinder_query_results), len(owner.triangle_pairs))
            self.assertEqual(len(owner._heightfield_cylinder_gjk_indices), len(owner.triangle_pairs))
            for options in (
                {"has_heightfields": False},
                {"reduce_contacts": False},
                {"speculative": True},
                {"deterministic": True},
                {"contact_writer_warp_func": write_contact_simple},
            ):
                with self.subTest(options=options):
                    self.assertFalse(create(**options)._heightfield_cylinder_split)
            for option in ("NEWTON_HEIGHTFIELD_FINITE_QUERY", "NEWTON_HEIGHTFIELD_PAIR_CSR"):
                with patch.dict(os.environ, {option: "1", "NEWTON_HEIGHTFIELD_FINITE_QUERY": "1"}):
                    self.assertFalse(create()._heightfield_cylinder_split)
            with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "0"}):
                self.assertFalse(create()._heightfield_cylinder_split)
        with patch.dict(os.environ, flags):
            os.environ.pop("NEWTON_HEIGHTFIELD_CYLINDER_SPLIT", None)
            self.assertFalse(create()._heightfield_cylinder_split)
        with patch.dict(os.environ, {**flags, "NEWTON_HEIGHTFIELD_CYLINDER_SPLIT": "invalid"}):
            with self.assertRaises(ValueError):
                create()

    def assert_raw_equal(self, before, after):
        """Compare fingerprint ownership and physical geometry independently of allocation order."""
        self.assertEqual(before.keys(), after.keys())
        for key in before:
            np.testing.assert_allclose(before[key][:4], after[key][:4], rtol=2e-6, atol=2e-4)
            old_normal = np.asarray(decode_oct(wp.vec2(*before[key][4:])))
            new_normal = np.asarray(decode_oct(wp.vec2(*after[key][4:])))
            self.assertLess(float(np.linalg.norm(old_normal - new_normal)), 2e-3)

    def check_loaded_cylinder(self, device):
        """Retain loaded geometry and unsupported shapes through the original fallback."""
        checker = primitive.TestHeightfieldPrimitiveQuery()
        cases = [("cylinder", opts) for opts in ({}, {"slope": True}, {"border": True}, {"deep": True})]
        cases += [("cylinder", {"barrel": True, "slope": True}), ("capsule", {}), ("sphere", {})]
        cases += [
            ("cylinder", {"dimensions": (0.0175, 0.19786), "angle": 87.99}),
            ("cylinder", {"dimensions": (0.07, 0.081565), "angle": 88.01}),
        ]
        for kind, options in cases:
            with self.subTest(device=device, kind=kind, options=options):
                model = primitive_model(
                    device, kind, **{k: v for k, v in options.items() if k not in ("dimensions", "angle")}
                )
                state = model.state()
                if "dimensions" in options:
                    dimensions = model.shape_scale.numpy()
                    dimensions[1, :2] = options["dimensions"]
                    model.shape_scale.assign(dimensions)
                    poses = state.body_q.numpy()
                    poses[:, 3:] = np.asarray(
                        wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(np.deg2rad(options["angle"])))
                    )
                    state.body_q.assign(poses)
                reference, actual = split_pipeline(model, False), split_pipeline(model, True)
                self.assertTrue(actual.narrow_phase._heightfield_cylinder_split)
                before, after = reference.contacts(), actual.contacts()
                reference.collide(state, before)
                actual.collide(state, after)
                checker.assert_current_contacts(reference, actual, state, before, after)
                if kind != "cylinder" or options.get("barrel"):
                    self.assertEqual(int(actual.narrow_phase._heightfield_cylinder_gjk_count.numpy()[0]), 0)

    def test_loaded_cylinder_cpu(self):
        """Run the admitted split owner and retain loaded cylinder geometry."""
        self.check_loaded_cylinder("cpu")

    def check_direct_queries(self, device):
        """Exercise far queries without storing them, and bound the failed-MPR queue."""
        model = primitive_model(device, "cylinder")
        state = model.state()
        reference, actual = split_pipeline(model, False), split_pipeline(model, True)
        captured = []
        launch = wp.launch

        def observe(*args, **kwargs):
            if kwargs.get("kernel") == reference.narrow_phase._heightfield_primitive_kernels[1]:
                captured.append(list(kwargs["inputs"]))
            return launch(*args, **kwargs)

        with patch.object(wp, "launch", side_effect=observe):
            reference.collide(state, reference.contacts())
        self.assertEqual(len(captured), 1)
        common = captured[0]
        common[-1] = 128
        owner = actual.narrow_phase
        checker = primitive.TestHeightfieldPrimitiveQuery()
        seen_mpr = seen_gjk = seen_far = False
        # Include far cells that broad phase would omit, so rejection is tested
        # after querying rather than hidden by broad-phase admission.
        triangles = np.column_stack((np.zeros(512, np.int32), np.ones(512, np.int32), np.arange(512)))
        for margin_sum in (0.0, 5e-5, 1e-4, 0.002):
            data = common[1].numpy().copy()
            data[:, 3] = margin_sum * 0.5
            common[1].assign(data)
            for count in (0, 1, 31, 32, 33, 512, len(owner._heightfield_cylinder_query_results)):
                with self.subTest(device=device, margin_sum=margin_sum, count=count):
                    # Include unsupported holes and a late eligible position, without
                    # changing fingerprint identity for the admitted triangles.
                    pairs = triangles.copy()
                    if count > len(triangles):
                        pairs = np.tile(np.array([1, 0, 0], dtype=np.int32), (count, 1))
                        pairs[-1] = (0, 1, 511)
                    if count == 33:
                        pairs[::2, :2] = (1, 0)
                        pairs[32, :2] = (0, 1)
                    common[8] = wp.array(pairs, dtype=wp.vec3i, device=device)
                    common[9] = wp.array([count], dtype=int, device=device)
                    reference.narrow_phase.global_contact_reducer.contact_count.zero_()
                    common[10] = reference.narrow_phase.global_contact_reducer.get_data_struct()
                    launch(
                        reference.narrow_phase._heightfield_primitive_kernels[1],
                        dim=128,
                        inputs=common,
                        device=device,
                        block_dim=128,
                    )
                    expected = checker.raw_contacts(reference)
                    owner.global_contact_reducer.contact_count.zero_()
                    owner._heightfield_cylinder_gjk_count.zero_()
                    common[10] = owner.global_contact_reducer.get_data_struct()
                    inputs = [
                        *common,
                        owner._heightfield_cylinder_query_results,
                        owner._heightfield_cylinder_gjk_indices,
                        owner._heightfield_cylinder_gjk_count,
                        SupportMapDataProvider(),
                    ]
                    for kernel in owner._heightfield_cylinder_split_kernels:
                        launch(kernel, dim=128, inputs=inputs, device=device, block_dim=128)
                    result = checker.raw_contacts(actual)
                    self.assert_raw_equal(expected, result)
                    n = int(owner._heightfield_cylinder_gjk_count.numpy()[0])
                    indices = owner._heightfield_cylinder_gjk_indices.numpy()[:n]
                    eligible = np.flatnonzero((pairs[:count, 0] == 0) & (pairs[:count, 1] == 1))
                    self.assertLessEqual(n, len(eligible))
                    self.assertEqual(len(set(indices.tolist())), n)
                    self.assertTrue(set(indices.tolist()).issubset(set(eligible.tolist())))
                    seen_gjk |= n > 0
                    seen_mpr |= n < len(eligible)
                    threshold = float(common[4].numpy().sum()) + margin_sum
                    queries = owner._heightfield_cylinder_query_results.numpy()
                    seen_far |= bool(np.any(queries["signed_distance"][eligible] > threshold + 1e-3))
                    self.assertFalse(
                        any(value[3] > threshold + 1e-3 for value in expected.values()),
                        "Clearly separated witnesses must not consume raw contact storage",
                    )
        self.assertTrue(seen_mpr and seen_gjk and seen_far, "Exercise MPR, GJK and rejected far witnesses")

    def test_direct_queries_cpu(self):
        """Compare fixed queries, far rejection and margin-dependent inflation."""
        self.check_direct_queries("cpu")

    def collide_and_check_route(self, owner, state, contacts, fallback):
        """Observe the writer that owns callbacks after a current shape-type change."""
        narrow = owner.narrow_phase
        kernels = (narrow._heightfield_cylinder_split_kernels[-1], narrow._heightfield_primitive_kernels[-1])
        counts = {}
        launch = wp.launch

        def observe(*args, **kwargs):
            kernel = kwargs.get("kernel", args[0] if args else None)
            if kernel not in kernels:
                return launch(*args, **kwargs)
            before = int(narrow.global_contact_reducer.contact_count.numpy()[0])
            result = launch(*args, **kwargs)
            counts[kernels.index(kernel)] = int(narrow.global_contact_reducer.contact_count.numpy()[0]) - before
            return result

        with patch.object(wp, "launch", side_effect=observe):
            owner.collide(state, contacts)
        self.assertEqual(counts[int(not fallback)], 0)
        self.assertGreater(counts[int(fallback)], 0)

    def check_lifecycle(self, device):
        """Refresh results and queue ownership through graph-empty and geometry mutation transitions."""
        model = primitive_model(device, "cylinder")
        state = model.state()
        reference, actual = split_pipeline(model, False), split_pipeline(model, True)
        before, after = reference.contacts(), actual.contacts()
        reference.collide(state, before)
        actual.collide(state, after)
        checker = primitive.TestHeightfieldPrimitiveQuery()
        graph = None
        if wp.get_device(device).is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                actual.collide(state, after)
            graph = capture.graph

        def replay():
            if graph is None:
                actual.collide(state, after)
            else:
                wp.capture_launch(graph)

        poses = state.body_q.numpy().copy()
        types = model.shape_type.numpy().copy()
        scales = model.shape_scale.numpy().copy()
        for epoch in range(5):
            with self.subTest(device=device, epoch=epoch):
                away = poses.copy()
                away[:, 2] += 3.0
                state.body_q.assign(away)
                replay()
                self.assertEqual(int(after.rigid_contact_count.numpy()[0]), 0)
                self.assertEqual(int(actual.narrow_phase._heightfield_cylinder_gjk_count.numpy()[0]), 0)
                current = poses.copy()
                current[:, 0] += 0.005 * epoch
                current[:, 3:] = np.asarray(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.3 * epoch))
                state.body_q.assign(current)
                kinds, dimensions = types.copy(), scales.copy()
                dimensions[1, :2] *= 0.9
                if epoch == 1:
                    # Every primitive now has a typed query; only a barrel cylinder stays on
                    # the generic fallback.
                    dimensions[1, 2] = 0.2
                if epoch == 2:
                    dimensions[1, 2] = 0.2
                model.shape_type.assign(kinds)
                model.shape_scale.assign(dimensions)
                heights = model.heightfield_elevations.numpy()
                heights[8 * 17 + 8] += 0.001
                model.heightfield_elevations.assign(heights)
                model.shape_gap.fill_(0.001 + 0.0001 * epoch)
                model.shape_margin.fill_(0.0005 * epoch)
                reference.collide(state, before)
                replay()
                checker.assert_current_contacts(reference, actual, state, before, after)
                captured_raw = checker.raw_contacts(actual)
                self.collide_and_check_route(actual, state, after, fallback=epoch in (1, 2))
                self.assert_raw_equal(captured_raw, checker.raw_contacts(actual))
                if epoch in (1, 2):
                    self.assertEqual(int(actual.narrow_phase._heightfield_cylinder_gjk_count.numpy()[0]), 0)

    def test_lifecycle_cpu(self):
        """Refresh current geometry after empty, barrel fallback frames."""
        self.check_lifecycle("cpu")

    def test_loaded_support_cpu(self):
        """Retain the existing loaded XPBD sphere and cylinder support outcomes."""
        with patch.object(primitive, "pipeline", side_effect=split_pipeline):
            primitive.TestHeightfieldPrimitiveQuery().test_loaded_support_cpu()

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_loaded_cylinder_cuda(self):
        """Retain loaded typed and fallback geometry on CUDA."""
        self.check_loaded_cylinder("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_direct_queries_cuda(self):
        """Reject far CUDA witnesses and retain bounded partial work queues."""
        self.check_direct_queries("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_graph_lifecycle_cuda(self):
        """Replay current geometry and empty/regrowing queues in a CUDA graph."""
        self.check_lifecycle("cuda:0")


if __name__ == "__main__":
    unittest.main()
