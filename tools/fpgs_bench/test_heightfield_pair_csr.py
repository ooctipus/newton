"""Focused routing and physical controls for raw-witness pair CSR."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.geometry.contact_data import ContactData
from newton._src.geometry.contact_reduction_global import (
    GlobalContactReducer,
    GlobalContactReducerData,
    export_contact_to_buffer,
)
from newton._src.geometry.heightfield_pair_csr import (
    PairCSRData,
    _next_down,
    _next_up,
    _write_pair,
    _writer_separation_interval,
)
from newton._src.geometry.types import GeoType
from tools.fpgs_bench import test_heightfield_adaptive_manifold as adaptive_tests


@wp.kernel
def seed_raw_pool(
    reducer: GlobalContactReducerData,
    raw_pair: wp.array[int],
    pairs: wp.array[wp.vec2i],
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    depths: wp.array[float],
    tags: wp.array[int],
):
    for index in range(pairs.shape[0]):
        pair = pairs[index]
        identifier = export_contact_to_buffer(
            pair[0], pair[1], points[index], normals[index], depths[index], index + 1, reducer
        )
        if identifier > 0:
            raw_pair[identifier] = tags[index]


@wp.kernel
def overflow_raw_writer(data: PairCSRData):
    local = data
    local.pair_index = 0
    contact = ContactData()
    contact.shape_a = 0
    contact.shape_b = 1
    contact.contact_point_center = wp.vec3(0.0)
    contact.contact_normal_a_to_b = wp.vec3(0.0, 0.0, 1.0)
    contact.contact_distance = -0.001
    for index in range(2):
        contact.sort_sub_key = index + 1
        _write_pair(contact, local, -1)


@wp.kernel
def evaluate_intervals(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    depths: wp.array[float],
    parameters: wp.array[wp.vec4],
    result: wp.array[wp.vec3],
    represented_normal: wp.array[wp.vec3],
):
    index = wp.tid()
    p = parameters[index]
    result[index] = _writer_separation_interval(points[index], normals[index], depths[index], p[0], p[1], p[2], p[3])
    represented_normal[index] = wp.normalize(normals[index])


@wp.kernel
def evaluate_successors(values: wp.array[float], result: wp.array[wp.vec2]):
    index = wp.tid()
    result[index] = wp.vec2(_next_down(values[index]), _next_up(values[index]))


class TestHeightfieldPairCSR(unittest.TestCase):
    loaded_feature_env = "NEWTON_HEIGHTFIELD_PAIR_CSR"
    loaded_feature_attribute = "_heightfield_pair_csr"
    loaded_report_label = "PAIR_CSR_QUALIFICATION"

    def _interval_arithmetic(self, device):
        from newton._src.geometry.heightfield_pair_csr import count_contacts  # noqa: PLC0415

        self.assertIs(count_contacts.module.options["fast_math"], False)
        values = np.array(
            [
                0.0,
                -0.0,
                1.0,
                -1.0,
                np.finfo(np.float32).tiny,
                -np.finfo(np.float32).tiny,
                np.finfo(np.float32).max,
                -np.finfo(np.float32).max,
                np.inf,
                -np.inf,
                np.nan,
            ],
            dtype=np.float32,
        )
        successors = wp.empty(len(values), dtype=wp.vec2, device=device)
        wp.launch(
            evaluate_successors, dim=len(values), inputs=[wp.array(values, device=device), successors], device=device
        )
        with np.errstate(over="ignore", invalid="ignore"):
            expected = np.column_stack(
                (np.nextafter(values, np.float32(-np.inf)), np.nextafter(values, np.float32(np.inf)))
            )
        np.testing.assert_array_equal(successors.numpy(), expected)

        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, -0.1, 0.0025],
                [1e5, -1e5, 1e5],
                [1e5, -1e5, 1e5],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0.6, 0.8, 0], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        depths = np.array([0.005, 0.008, 0.005, 0.012, 0.001, -0.001], dtype=np.float32)
        # Includes zero/negative authored margins and nonzero radii, without
        # changing the stock separation algebra or widening the detection gap.
        parameters = np.array(
            [
                [0, 0, 0.0025, 0.0025],
                [0, 0, 0.0025, 0.0025],
                [0, 0, 0.0025, 0.0025],
                [0.01, 0.02, 0.0025, 0.0025],
                [0, 0, 0, 0],
                [0, 0, -0.002, -0.003],
            ],
            dtype=np.float32,
        )
        intervals = wp.empty(len(points), dtype=wp.vec3, device=device)
        represented = wp.empty(len(points), dtype=wp.vec3, device=device)
        wp.launch(
            evaluate_intervals,
            dim=len(points),
            inputs=[
                wp.array(points, dtype=wp.vec3, device=device),
                wp.array(normals, dtype=wp.vec3, device=device),
                wp.array(depths, device=device),
                wp.array(parameters, dtype=wp.vec4, device=device),
                intervals,
                represented,
            ],
            device=device,
        )
        actual, n = intervals.numpy().astype(float), represented.numpy().astype(float)
        p = parameters.astype(float)
        exact = (depths.astype(float) + p[:, 0] + p[:, 1]) * np.sum(n * n, axis=1) - p.sum(axis=1)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(np.all(actual[:, 1] <= exact), (actual, exact))
        self.assertTrue(np.all(exact <= actual[:, 2]), (actual, exact))
        self.assertTrue(np.all(actual[:, 1] < actual[:, 0]))
        self.assertTrue(np.all(actual[:, 0] < actual[:, 2]))
        # Large-center cancellation broadens the arithmetic interval; it must
        # not fabricate precision or silently become a geometric certificate.
        self.assertGreater(actual[2, 2] - actual[2, 1], actual[0, 2] - actual[0, 1])
        np.testing.assert_allclose(actual[[4, 5], 0], [0.001, 0.004], rtol=1e-6, atol=1e-9)

        from newton._src.geometry.heightfield_pair_csr import PairCSR  # noqa: PLC0415

        reducer = GlobalContactReducer(4, device=device)
        owner = PairCSR(reducer, 1, 1, device=device)
        pairs = wp.array([(0, 1)], dtype=wp.vec2i, device=device)
        pair_count = wp.array([1], dtype=int, device=device)
        types = wp.array([GeoType.HFIELD, GeoType.BOX], dtype=int, device=device)
        gaps = wp.full(2, 0.01, dtype=float, device=device)
        for margins, center, expected_count, expected_status in (
            ((0.0, 0.0), 0.0, 1, 0),
            ((-0.002, -0.003), 0.0, 0, 0),
            # All source fields are finite, but 2*|center| overflows the
            # interval scale. A broad/NaN cohort must not hide this failure.
            ((0.0, 0.0), 3e38, 0, 32),
        ):
            reducer.contact_count.zero_()
            owner.status.zero_()
            wp.launch(
                seed_raw_pool,
                dim=1,
                inputs=[
                    reducer.get_data_struct(),
                    owner.raw_pair,
                    pairs,
                    wp.array([(center, 0.0, 0.0)], dtype=wp.vec3, device=device),
                    wp.array([(1.0, 0.0, 0.0)], dtype=wp.vec3, device=device),
                    wp.array([0.018], dtype=float, device=device),
                    wp.array([0], dtype=int, device=device),
                ],
                device=device,
            )
            shape_data = wp.array([(0.0, 0.0, 0.0, m) for m in margins], dtype=wp.vec4, device=device)
            owner.build(pairs, pair_count, types, shape_data, gaps, 32)
            self.assertEqual(int(owner.counts.numpy()[0]), expected_count)
            self.assertEqual(int(owner.status.numpy()[0]), expected_status)

    def test_interval_arithmetic_cpu(self):
        """Bound represented-witness arithmetic, including margins and cancellation."""
        self._interval_arithmetic("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Root owns CUDA execution")
    def test_interval_arithmetic_cuda(self):
        """Check native outward rounding and the same finite arithmetic controls."""
        self._interval_arithmetic("cuda:0")

    def test_module_exists(self):
        """Require the independently owned pair-CSR implementation."""
        from newton._src.geometry import heightfield_pair_csr  # noqa: PLC0415

        self.assertIsNotNone(heightfield_pair_csr)

    def test_dispatch_guards_cpu(self):
        """Require stock writer and preserve unsupported mode ownership."""
        from newton._src.geometry.narrow_phase import NarrowPhase  # noqa: PLC0415
        from newton._src.sim.collide import write_contact  # noqa: PLC0415

        for flag, writer, speculative, deterministic, expected in (
            ("0", write_contact, False, False, False),
            ("1", write_contact, False, False, True),
            ("1", None, False, False, False),
            ("1", adaptive_tests.record_writer, False, False, False),
            ("1", write_contact, True, False, False),
            ("1", write_contact, False, True, False),
        ):
            with self.subTest(flag=flag, writer=writer, speculative=speculative, deterministic=deterministic):
                with patch.dict(
                    os.environ,
                    {
                        "NEWTON_HEIGHTFIELD_PAIR_CSR": flag,
                        "NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD": "0",
                        "NEWTON_HEIGHTFIELD_CELL_REJECT": "1",
                        "NEWTON_HEIGHTFIELD_FINITE_QUERY": "1",
                        "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1",
                    },
                ):
                    owner = NarrowPhase(
                        max_candidate_pairs=8,
                        max_triangle_pairs=32,
                        has_meshes=False,
                        has_heightfields=True,
                        contact_writer_warp_func=writer,
                        speculative=speculative,
                        contact_writer_supports_speculative=True,
                        deterministic=deterministic,
                        device="cpu",
                    )
                self.assertEqual(owner._heightfield_pair_csr_requested, expected)
                # Actual activation additionally needs pipeline pair uniqueness.
                self.assertFalse(owner._heightfield_pair_csr)
                self.assertIsNone(owner._pair_csr)

    def _raw_variable_pool(self, device):
        from newton._src.geometry.heightfield_pair_csr import PairCSR  # noqa: PLC0415

        reducer = GlobalContactReducer(400, device=device)
        owner = PairCSR(reducer, 3, 2, device=device)
        # A deliberately larger-than-old252 pool plus a rejected farther
        # point, a second mixed-normal pair, and an unsupported callback.
        pairs = [(0, 1)] * 301 + [(0, 2)] * 4 + [(0, 3)]
        points = [(float(i) / 300.0, 0.0, 0.0) for i in range(300)] + [(100.0, 0.0, 0.0)] + [(0.0, 0.0, 0.0)] * 5
        normals = (
            [(0.0, 0.0, 1.0)] * 301
            + [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)]
            + [(0.0, 0.0, 1.0)]
        )
        depths = [-0.001] * 299 + [-0.01, 0.2] + [-0.001] * 5
        tags = [0] * 301 + [1] * 4 + [2]
        wp.launch(
            seed_raw_pool,
            dim=1,
            inputs=[
                reducer.get_data_struct(),
                owner.data.raw_pair,
                wp.array(pairs, dtype=wp.vec2i, device=device),
                wp.array(points, dtype=wp.vec3, device=device),
                wp.array(normals, dtype=wp.vec3, device=device),
                wp.array(depths, dtype=float, device=device),
                wp.array(tags, dtype=int, device=device),
            ],
            device=device,
        )
        active = wp.array([(0, 1), (0, 2), (0, 3)], dtype=wp.vec2i, device=device)
        active_count = wp.array([3], dtype=int, device=device)
        types = wp.array([GeoType.HFIELD, GeoType.BOX, GeoType.CONVEX_MESH, GeoType.SPHERE], dtype=int, device=device)
        shape_data = wp.zeros(4, dtype=wp.vec4, device=device)
        gaps = wp.full(4, 0.01, dtype=float, device=device)
        owner.build(active, active_count, types, shape_data, gaps, 256)
        self.assertEqual(int(owner.data.status.numpy()[0]), 0)
        np.testing.assert_array_equal(owner.data.counts.numpy()[:2], [300, 4])
        np.testing.assert_array_equal(owner.data.offsets.numpy()[:3], [0, 300, 304])
        ids = owner.data.ids.numpy()[:304]
        np.testing.assert_array_equal(np.sort(ids[:300]), np.arange(1, 301))
        np.testing.assert_array_equal(np.sort(ids[300:]), np.arange(302, 306))
        self.assertEqual(len(set(ids.tolist())), 304)
        raw_tags = owner.data.raw_pair.numpy()
        self.assertEqual(raw_tags[301], -2)
        self.assertEqual(raw_tags[306], -1)
        from newton._src.geometry.heightfield_pair_csr import create_export_kernel  # noqa: PLC0415

        writer = adaptive_tests.WriterData()
        writer.count = wp.zeros(1, dtype=int, device=device)
        writer.ids = wp.empty(16, dtype=int, device=device)
        writer.points = wp.empty(16, dtype=wp.vec3, device=device)
        writer.normals = wp.empty(16, dtype=wp.vec3, device=device)
        writer.depths = wp.empty(16, dtype=float, device=device)
        width = 1 if wp.get_device(device).is_cpu else 32
        wp.launch_tiled(
            create_export_kernel(adaptive_tests.record_writer),
            dim=3,
            inputs=[owner.data, active_count, types, shape_data, gaps, writer, 3, int(width > 1)],
            device=device,
            block_dim=width,
        )
        published = writer.ids.numpy()[: int(writer.count.numpy()[0])]
        self.assertEqual(len(published), 8)
        self.assertEqual(len(set(published.tolist())), 8)
        self.assertIn(300, published)  # Deepest raw witness lies beyond old252.
        self.assertNotIn(301, published)  # Rejected farthest cannot win.
        self.assertTrue(set(range(302, 306)).issubset(published))
        self.assertNotIn(306, published)  # Unsupported route belongs to old export.
        # A new query re-tags its complete raw cohort before the next build.
        raw_tags[1:307] = tags
        owner.data.raw_pair.assign(raw_tags)
        owner.build(active, active_count, types, shape_data, gaps, 256)
        np.testing.assert_array_equal(owner.data.offsets.numpy()[:3], [0, 300, 304])
        reducer.contact_count.zero_()
        owner.build(active, active_count, types, shape_data, gaps, 256)
        np.testing.assert_array_equal(owner.data.offsets.numpy()[:3], [0, 0, 0])
        self.assertEqual(int(owner.data.status.numpy()[0]), 0)

    def test_variable_pool_cpu(self):
        """Route every accepted raw ID without a252-candidate per-pair cap."""
        self._raw_variable_pool("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Root owns CUDA execution")
    def test_variable_pool_cuda(self):
        """Check native count/scan/scatter completeness and reuse beyond252 IDs."""
        self._raw_variable_pool("cuda:0")

    def test_stock_nxn_reset_cpu(self):
        """Preserve live geometry and clear stale CSR membership after empty frames."""
        self._stock_nxn_reset("cpu")

    def _nearest_footprint(self, device):
        q = adaptive_tests.qualification_module()
        for name in ("support", "rebound"):
            with self.subTest(case=name):
                case = next(case for case in q.CASES if case.name == name)
                with patch.dict(
                    os.environ,
                    {
                        "NEWTON_HEIGHTFIELD_PAIR_CSR": "1",
                        "NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD": "0",
                        "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1",
                    },
                ):
                    scene = q.build_scene(case, True, device, make_solver=False)
                scene.pipeline.collide(scene.states[0], scene.contacts)
                observed = q.geometry(scene, scene.states[0])
                self.assertEqual(observed["contacts"], 4)
                # The actual raw pool includes side witnesses 3--4 cm away.
                # A resting or uniformly approaching planar footprint must not
                # lose three near-surface constraints to those outer witnesses.
                normal = scene.contacts.rigid_contact_normal.numpy()[:4]
                np.testing.assert_allclose(normal, np.tile(scene.normal, (4, 1)), rtol=0.0, atol=3e-5)
                np.testing.assert_allclose(scene.distance.numpy()[:4], case.gap, rtol=0.0, atol=3e-6)
                points = scene.point0.numpy()[:4, :2].astype(float)
                self.assertGreater(float(np.ptp(points[:, 0])), 0.15)
                self.assertGreater(float(np.ptp(points[:, 1])), 0.12)
                # Origin lies inside the selected support polygon. This is a
                # geometry regression, not a replacement for loaded dynamics.
                points = points[np.argsort(np.arctan2(points[:, 1], points[:, 0]))]
                following = np.roll(points, -1, axis=0)
                cross = points[:, 0] * following[:, 1] - points[:, 1] * following[:, 0]
                self.assertTrue(np.all(cross >= -1e-7), points)

    def test_nearest_footprint_cpu(self):
        """Retain active and all-positive nearest footprints from actual query pools."""
        self._nearest_footprint("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Root owns CUDA execution")
    def test_nearest_footprint_cuda(self):
        """Check the native selector without changing the loaded fixture or gates."""
        self._nearest_footprint("cuda:0")

    def _stock_nxn_reset(self, device):
        q = adaptive_tests.qualification_module()
        case = next(case for case in q.CASES if case.name == "support")
        with patch.dict(
            os.environ,
            {
                "NEWTON_HEIGHTFIELD_PAIR_CSR": "1",
                "NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD": "0",
                "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1",
            },
        ):
            scene = q.build_scene(case, True, device, make_solver=False)
        narrow = scene.pipeline.narrow_phase
        self.assertIs(narrow._heightfield_pair_csr, True)
        self.assertIsNotNone(narrow._pair_csr)
        state = scene.states[0]
        original_q = state.joint_q.numpy().copy()
        counts = []
        for raised in (False, True, False):
            coordinates = original_q.copy()
            if raised:
                coordinates[2] += 1.0
            state.joint_q.assign(coordinates)
            q.newton.eval_fk(scene.model, state.joint_q, state.joint_qd, state)
            scene.pipeline.collide(state, scene.contacts)
            narrow.check_buffer_capacity()
            self.assertEqual(int(narrow._pair_csr.data.status.numpy()[0]), 0)
            count = int(scene.contacts.rigid_contact_count.numpy()[0])
            counts.append(count)
            self.assertTrue(np.isfinite(scene.contacts.rigid_contact_normal.numpy()[:count]).all())
        self.assertGreater(counts[0], 0)
        self.assertLessEqual(counts[0], 4)
        self.assertEqual(counts[1], 0)
        self.assertGreater(counts[2], 0)
        self.assertLessEqual(counts[2], 4)
        if not wp.get_device(device).is_cpu:
            # Warmed producer/scan workspace above; capture only the existing
            # collision path, then vary consumed state without reallocating.
            with wp.ScopedCapture(device=device) as capture:
                scene.pipeline.collide(state, scene.contacts)
            for raised in (True, False):
                coordinates = original_q.copy()
                coordinates[2] += float(raised)
                state.joint_q.assign(coordinates)
                q.newton.eval_fk(scene.model, state.joint_q, state.joint_qd, state)
                wp.capture_launch(capture.graph)
                narrow.check_buffer_capacity()
                self.assertEqual(int(narrow._pair_csr.data.status.numpy()[0]), 0)
                count = int(scene.contacts.rigid_contact_count.numpy()[0])
                if raised:
                    self.assertEqual(count, 0)
                else:
                    self.assertGreater(count, 0)
                    self.assertLessEqual(count, 4)

    def test_duplicate_explicit_fallback_cpu(self):
        """Keep nonunique explicit-pair inputs on the unchanged owner."""
        q = adaptive_tests.qualification_module()
        case = next(case for case in q.CASES if case.name == "support")
        with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_PAIR_CSR": "0", "NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD": "0"}):
            scene = q.build_scene(case, True, "cpu", make_solver=False)
        duplicate = wp.array([(0, 1), (1, 0)], dtype=wp.vec2i, device="cpu")
        with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_PAIR_CSR": "1", "NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD": "0"}):
            pipeline = q.newton.CollisionPipeline(
                scene.model,
                broad_phase="explicit",
                shape_pairs_filtered=duplicate,
                reduce_contacts=True,
                rigid_contact_max=100,
                max_triangle_pairs=128,
            )
        self.assertIs(pipeline.narrow_phase._heightfield_pair_csr, False)
        contacts = pipeline.contacts()
        pipeline.collide(scene.states[0], contacts)
        pipeline.narrow_phase.check_buffer_capacity()
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)

    def _sticky_overflow(self, device):
        q = adaptive_tests.qualification_module()
        case = next(case for case in q.CASES if case.name == "support")
        with patch.dict(
            os.environ,
            {
                "NEWTON_HEIGHTFIELD_PAIR_CSR": "1",
                "NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD": "0",
                "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1",
            },
        ):
            scene = q.build_scene(case, True, device, make_solver=False)
        narrow = scene.pipeline.narrow_phase
        self.assertTrue(narrow._heightfield_pair_csr)
        owner = narrow._pair_csr
        # Tighten only the logical raw reservation bound, retaining actual
        # allocated storage. Exercise the real callback rather than fabricating
        # a status flag; the failed reservation must undo count AND latch loss.
        capacity = owner.data.reducer.capacity
        owner.data.reducer.capacity = 1
        owner.data.reducer.contact_count.zero_()
        wp.launch(overflow_raw_writer, dim=1, inputs=[owner.data], device=device)
        self.assertEqual(int(owner.data.reducer.contact_count.numpy()[0]), 1)
        self.assertEqual(int(owner.status.numpy()[0]) & 1, 1)
        with self.assertRaisesRegex(RuntimeError, "heightfield_pair_csr"):
            narrow.check_buffer_capacity()
        owner.data.reducer.capacity = capacity
        scene.pipeline.collide(scene.states[0], scene.contacts)
        with self.assertRaisesRegex(RuntimeError, "heightfield_pair_csr"):
            narrow.check_buffer_capacity()
        cleared = narrow.buffer_capacity_status(clear=True)
        self.assertIs(cleared["heightfield_pair_csr_raw_or_membership"], True)
        self.assertIs(narrow.buffer_capacity_status()["heightfield_pair_csr_raw_or_membership"], False)
        scene.pipeline.collide(scene.states[0], scene.contacts)
        narrow.check_buffer_capacity()
        self.assertEqual(int(owner.status.numpy()[0]), 0)
        self.assertGreater(int(scene.contacts.rigid_contact_count.numpy()[0]), 0)

    def test_sticky_overflow_cpu(self):
        """Expose actual raw callback overflow through the public capacity check."""
        self._sticky_overflow("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Root owns CUDA execution")
    def test_sticky_overflow_cuda(self):
        """Keep raw loss sticky across native resets until explicit acknowledgement."""
        self._sticky_overflow("cuda:0")

    def _mixed_scene(self, device):
        q = adaptive_tests.qualification_module()
        builder = q.newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        config = builder.ShapeConfig(density=0.0, mu=0.4, margin=0.0025, gap=0.02)
        builder.add_shape_heightfield(
            heightfield=q.newton.Heightfield(data=np.zeros((9, 9), dtype=np.float32), nrow=9, ncol=9, hx=1.0, hy=1.0),
            cfg=config,
        )
        for sphere, x in ((False, -0.3), (True, 0.3)):
            link = builder.add_link(
                xform=wp.transform(wp.vec3(x, 0.0, 0.054), wp.quat_identity()),
                mass=1.0,
                inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01),
                lock_inertia=True,
            )
            joint = builder.add_joint_free(child=link)
            builder.add_articulation([joint])
            if sphere:
                builder.add_shape_sphere(body=link, radius=0.05, cfg=config)
            else:
                builder.add_shape_box(body=link, hx=0.08, hy=0.07, hz=0.05, cfg=config)
        model = builder.finalize(device=device)
        state = model.state()
        q.newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        outputs = []
        for enabled in (False, True):
            with patch.dict(
                os.environ,
                {
                    "NEWTON_HEIGHTFIELD_PAIR_CSR": str(int(enabled)),
                    "NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD": "0",
                    "NEWTON_HEIGHTFIELD_CELL_REJECT": "1",
                    "NEWTON_HEIGHTFIELD_FINITE_QUERY": "1",
                    "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1",
                },
            ):
                pipeline = q.newton.CollisionPipeline(
                    model, broad_phase="nxn", reduce_contacts=True, rigid_contact_max=128, max_triangle_pairs=256
                )
            self.assertIs(pipeline.narrow_phase._heightfield_pair_csr, enabled)
            contacts = pipeline.contacts()
            pipeline.collide(state, contacts)
            pipeline.narrow_phase.check_buffer_capacity()
            count = int(contacts.rigid_contact_count.numpy()[0])
            shape0 = contacts.rigid_contact_shape0.numpy()[:count]
            shape1 = contacts.rigid_contact_shape1.numpy()[:count]
            sphere_rows = np.flatnonzero((shape0 == 2) | (shape1 == 2))
            box_rows = np.flatnonzero((shape0 == 1) | (shape1 == 1))
            self.assertGreater(len(sphere_rows), 0)
            self.assertGreater(len(box_rows), 0)
            if enabled:
                self.assertLessEqual(len(box_rows), 4)
                self.assertEqual(int(pipeline.narrow_phase._pair_csr.data.status.numpy()[0]), 0)
            # The unsupported sphere must still traverse its original query,
            # reduction and stock writer; compare its complete witness set.
            geometry = np.column_stack(
                (
                    contacts.rigid_contact_point0.numpy()[:count],
                    contacts.rigid_contact_point1.numpy()[:count],
                    contacts.rigid_contact_normal.numpy()[:count],
                )
            )[sphere_rows]
            order = np.lexsort(tuple(geometry[:, i] for i in reversed(range(geometry.shape[1]))))
            outputs.append(geometry[order])
        np.testing.assert_allclose(outputs[0], outputs[1], rtol=3e-5, atol=3e-6)

    def test_mixed_fallback_cpu(self):
        """Publish admitted boxes and unchanged unsupported sphere witnesses once."""
        self._mixed_scene("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Root owns CUDA execution")
    def test_mixed_fallback_cuda(self):
        """Exercise simultaneous admitted and fallback producer callbacks natively."""
        self._mixed_scene("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "Root owns CUDA execution")
    def test_stock_nxn_reset_cuda(self):
        """Exercise reset/empty/regrow on the root-owned native device."""
        self._stock_nxn_reset("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "Root owns CUDA execution")
    def test_loaded_cuda(self):
        """Reuse the unchanged nine-case paired physical controls and budgets."""
        with patch.dict(
            os.environ, {"NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD": "0", "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1"}
        ):
            adaptive_tests.TestAdaptiveManifold.test_loaded_cuda(self)


if __name__ == "__main__":
    unittest.main()
