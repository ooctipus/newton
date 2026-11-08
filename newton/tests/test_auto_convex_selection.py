# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check automatic collision choice without task-specific dispatch."""

import json
import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.coherent_convex import _ConvexQueryCache
from newton._src.geometry.coherent_convex_rejection import _create_rejection_query_kernels
from newton._src.geometry.convex_query_cost import _ConvexMethodChoice, _nonnegative_fit, _QueryCostModel
from newton.tests.test_coherent_convex_rejection import (
    _accepted,
    _box_inputs,
    _buffers,
    _original,
    _run_original,
    _selector_model,
)
from newton.tests.unittest_utils import add_function_test, assert_host_only, get_test_devices

# Terrain route flags the cases flip; every other NEWTON_ variable is cleared, so a
# developer's shell cannot change the expected routes.
_TERRAIN_ENVIRONMENT = {
    "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "0",
    "NEWTON_HEIGHTFIELD_ANALYTIC_PRIMITIVES": "1",
    "NEWTON_HEIGHTFIELD_ANALYTIC_CONVEX": "1",
    "NEWTON_HEIGHTFIELD_CYLINDER_SPLIT": "0",
    "NEWTON_HEIGHTFIELD_FINITE_QUERY": "0",
    "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "0",
    "NEWTON_HEIGHTFIELD_PAIR_CSR": "0",
    "NEWTON_REDUCER_EXPORT_CLEANUP": "0",
    # The hull routes are the terrain route's: keep it for every heightfield pair (the terrain law's opt-out).
    "NEWTON_TERRAIN_LAW": "none",
}


def _terrain_model(device):
    """Place one cube hull (8 vertices, 12 triangles) over a flat heightfield."""
    vertices = np.array([[x, y, z] for x in (-0.25, 0.25) for y in (-0.25, 0.25) for z in (-0.25, 0.25)], np.float32)
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ],
        np.int32,
    )
    builder = newton.ModelBuilder()
    builder.add_shape_heightfield(
        heightfield=newton.Heightfield(data=np.zeros((5, 5), np.float32), nrow=5, ncol=5, hx=0.4, hy=0.4)
    )
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.25), wp.quat_identity()))
    builder.add_shape_convex_hull(body, mesh=newton.Mesh(vertices, faces.ravel()))
    return builder.finalize(device=device)


def _terrain_pipeline(model, choice, environment=None):
    """Build a pipeline from Newton's defaults with the terrain flags pinned and the convex split left automatic."""
    with patch.dict(os.environ):
        for name in [name for name in os.environ if name.startswith("NEWTON_")]:
            del os.environ[name]
        os.environ.update({**_TERRAIN_ENVIRONMENT, **(environment or {})})
        return newton.CollisionPipeline(model, rigid_contact_max=4096, max_triangle_pairs=4096, convex_collision=choice)


def test_graph_choice_preserves_queries(test, device):
    """Change methods inside a captured graph as geometry changes; preserve cold outputs."""
    n, threads = 512, 128
    pairs, pair_array, values, inputs = _box_inputs(n, device)
    count = wp.array([n], dtype=int, device=device)
    baseline, candidate = _buffers(n, device), _buffers(n, device)
    original, kernels = _original(), _create_rejection_query_kernels()
    cache = _ConvexQueryCache(
        pairs=pairs,
        shape_types=values[0].numpy(),
        shape_world=np.zeros(2 * n, np.int32),
        world_count=1,
        query_capacity=n,
        device=device,
    )
    # Fixed independent-machine costs exercise both decisions reproducibly.
    model = _QueryCostModel(0, 100, 0, 1, 0, 1, 10, 0, ())
    with patch("newton._src.geometry.convex_query_cost._calibrate_query_cost", return_value=model):
        owner = _ConvexMethodChoice(device=device, slots=n)
    owner.period = 2

    def ordinary(buffers, query_count=None):
        query_count = count if query_count is None else query_count
        _run_original(original, pair_array, query_count, values, device, threads, buffers)

    def rejection(query_count, gjk_count):
        cache.begin()
        rejection_buffers = [candidate[0], candidate[1], gjk_count, candidate[3], candidate[4]]
        for kernel in kernels:
            wp.launch(
                kernel,
                threads,
                [pair_array, query_count, inputs, cache.data, threads, *rejection_buffers],
                device=device,
                block_dim=128,
            )

    def run():
        for buffer in candidate:
            buffer.zero_()
        event = wp.get_stream(device).record_event()
        owner.launch(pair_array, count, inputs, lambda query_count: ordinary(candidate, query_count), rejection)
        wp.get_stream(device).wait_event(event)

    poses = values[2].numpy()
    run()  # Compile both branch factories before capture.
    with wp.ScopedCapture(device=device) as capture:
        run()
    for distance, expected, reset in ((1.2, 1, False), (0.8, 0, False), (1.2, 1, True), (1.002, 0, True)):
        with test.subTest(distance=distance, reset=reset):
            poses[1::2, 0] = distance
            values[2].assign(poses)
            if reset:
                cache.reset(wp.array([True, False], dtype=wp.bool, device=device))
            for buffer in baseline:
                buffer.zero_()
            ordinary(baseline)
            for _ in range(4):
                wp.capture_launch(capture.graph)
            test.assertEqual(int(owner.choice.numpy()[0]), expected)
            ids, results = _accepted(candidate)
            reference_ids, reference = _accepted(baseline)
            np.testing.assert_array_equal(np.sort(ids), np.sort(reference_ids))
            test.assertEqual(results[np.sort(ids)].tobytes(), reference[np.sort(ids)].tobytes())
            np.testing.assert_array_equal(values[2].numpy(), poses)
    count.zero_()
    for _ in range(2):
        wp.capture_launch(capture.graph)
    test.assertEqual(int(owner.choice.numpy()[0]), 0)
    test.assertEqual(len(_accepted(candidate)[0]), 0)


def test_dense_pipeline_uses_ordinary(test, device):
    """An admitted dense scene uses the real calibrated model and ordinary contacts."""
    model = _selector_model(device)
    pipelines = []
    for mode in ("ordinary", "auto", "rejection"):
        with patch.dict(
            os.environ,
            {
                "NEWTON_NARROW_PHASE_COHERENT_CONVEX": "0",
                "NEWTON_NARROW_PHASE_COHERENT_STATS": "0",
                "NEWTON_NARROW_PHASE_PAIR_SHAPE_PREP": "0",
            },
        ):
            pipelines.append(
                newton.CollisionPipeline(model, broad_phase="explicit", rigid_contact_max=300000, convex_collision=mode)
            )
    state = model.state()
    counts = []
    for pipeline in pipelines:
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        with wp.ScopedCapture(device=device) as capture:
            pipeline.collide(state, contacts)
        wp.capture_launch(capture.graph)
        counts.append(int(contacts.rigid_contact_count.numpy()[0]))
    test.assertEqual(counts[0], counts[1])
    test.assertEqual(counts[0], counts[2])
    test.assertEqual(pipelines[2].get_collision_selection()["method"], "rejection")
    test.assertGreater(counts[0], 0)
    owner = pipelines[1].narrow_phase._convex_method_choice
    test.assertIsNotNone(owner)
    test.assertEqual(owner.describe()["method"], "ordinary")
    test.assertTrue(np.isfinite(owner.coefficients.numpy()).all())


class TestAutoConvexSelection(unittest.TestCase):
    def test_public_collision_choice_overrides_environment(self):
        """Keep explicit instance choices independent of legacy process settings."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body=body, hx=0.5, hy=0.5, hz=0.5)
        builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        with patch.dict(os.environ, {"NEWTON_NARROW_PHASE_COHERENT_CONVEX": "reject_only"}):
            ordinary = newton.CollisionPipeline(model, convex_collision="ordinary")
            automatic = newton.CollisionPipeline(model, convex_collision="auto")
        self.assertEqual(ordinary.get_collision_selection()["method"], "ordinary")
        self.assertEqual(automatic.get_collision_selection()["requested"], "auto")
        self.assertEqual(automatic.get_collision_selection()["reason"], "unsupported_pipeline")
        self.assertEqual(
            ordinary.get_collision_selection()["narrow_phase"]["coherent_convex"],
            {"requested": "off", "source": "argument", "mode": "off", "reason": "explicit_ordinary"},
        )
        self.assertEqual(
            automatic.get_collision_selection()["narrow_phase"]["coherent_convex"],
            {"requested": "auto", "source": "argument", "mode": "off", "reason": "unsupported_pipeline"},
        )
        self.assertIsNone(automatic.get_collision_selection()["heightfield"])
        for choice in ("rejection", "convex_cells", "typo"):
            with self.subTest(choice=choice), self.assertRaisesRegex(ValueError, "convex_collision"):
                newton.CollisionPipeline(model, convex_collision=choice)

    def test_collision_record_reports_terrain_routes_cpu(self):
        """Report the typed terrain table, the convex-hull route and opt-in routes without device reads."""
        model = _terrain_model("cpu")
        geometric = {
            "NEWTON_HEIGHTFIELD_FINITE_QUERY": "1",
            "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1",
            "NEWTON_HEIGHTFIELD_PAIR_CSR": "1",
        }
        cases = (
            ("auto", {}, "closed_form_hull_row", True),
            ("ordinary", {}, "generic_gjk", False),
            ("ordinary", {"NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "1"}, "closed_form_hull_row", True),
            ("ordinary", geometric, "pair_csr", False),
        )
        for choice, environment, hull, table in cases:
            with self.subTest(choice=choice, environment=environment):
                pipeline = _terrain_pipeline(model, choice, environment)
                # No chooser exists on CPU, so the whole record is host-only here.
                with assert_host_only():
                    selection = pipeline.get_collision_selection()
                # Benchmark results store this record with json.dumps(..., allow_nan=False).
                json.dumps(selection, allow_nan=False)
                terrain = selection["heightfield"]
                self.assertEqual(terrain["hull_route"], hull)
                self.assertEqual(terrain["primitive_query"]["enabled"], table)
                self.assertEqual(
                    [row["kernel"] for row in terrain["primitive_query"]["rows"]],
                    [kernel.key for kernel in pipeline.narrow_phase._heightfield_primitive_kernels],
                )
                self.assertEqual(selection["narrow_phase"]["coherent_convex"]["mode"], "off")
        automatic_selection = _terrain_pipeline(model, "auto").get_collision_selection()
        # The pipeline's typed-table request superseded the pinned NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY=0.
        self.assertEqual(automatic_selection["environment"]["overridden"], {"NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "0"})
        self.assertNotIn("NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY", automatic_selection["environment"]["applied"])
        automatic = automatic_selection["heightfield"]
        self.assertEqual(automatic["primitive_query"]["source"], "argument")
        self.assertIn("convex_collision: admitted", automatic["primitive_query"]["reason"])
        self.assertEqual(
            [(row["row"], row.get("closed_form", row.get("hull"))) for row in automatic["primitive_query"]["rows"]],
            [("sphere", True), ("cylinder", True), ("capsule", True), ("box", True), ("fallback", "closed_form")],
        )
        environment = {"NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "1", "NEWTON_HEIGHTFIELD_CYLINDER_SPLIT": "1"}
        split_selection = _terrain_pipeline(model, "ordinary", environment).get_collision_selection()
        # "ordinary" requests no typed table, so the environment's value is the one applied.
        self.assertEqual(split_selection["environment"]["applied"]["NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY"], "1")
        self.assertNotIn("NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY", split_selection["environment"]["overridden"])
        split = split_selection["heightfield"]
        self.assertFalse(split["cylinder_split"]["enabled"])
        self.assertIn("closed-form cylinder row", split["cylinder_split"]["reason"])
        self.assertEqual(split["primitive_query"]["source"], "environment")
        self.assertEqual(split["primitive_query"]["reason"], "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY=1: admitted.")
        pair = _terrain_pipeline(model, "ordinary", geometric).get_collision_selection()["heightfield"]
        self.assertEqual(pair["midphase"], "pair_csr")
        self.assertTrue(pair["pair_csr"]["enabled"])
        self.assertTrue(pair["finite_query"]["enabled"] and pair["geometric_cull"]["enabled"])
        self.assertIn("NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY is not 1", pair["primitive_query"]["reason"])

    @unittest.skipUnless(wp.is_cuda_available(), "requires CUDA")
    def test_collision_record_reports_automatic_terrain_routes_cuda(self):
        """Send hulls through the closed-form hull row on CUDA too, with every typed row of the CPU record."""
        model = _terrain_model("cuda:0")
        pipeline = _terrain_pipeline(model, "auto")
        with assert_host_only():
            pipeline._narrow_phase_selection()
            pipeline._heightfield_selection()
        automatic = pipeline.get_collision_selection()
        json.dumps(automatic, allow_nan=False)
        terrain = automatic["heightfield"]
        self.assertEqual(terrain["hull_route"], "closed_form_hull_row")
        self.assertEqual(
            [row["kernel"] for row in terrain["primitive_query"]["rows"]],
            [kernel.key for kernel in pipeline.narrow_phase._heightfield_primitive_kernels],
        )
        self.assertEqual(
            [row["row"] for row in terrain["primitive_query"]["rows"]],
            ["sphere", "cylinder", "capsule", "box", "fallback"],
        )
        ordinary = _terrain_pipeline(model, "ordinary", {"NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "1"})
        self.assertEqual(ordinary.get_collision_selection()["heightfield"]["hull_route"], "closed_form_hull_row")

    def test_nonnegative_cost_fit(self):
        # Constraining a negative slope requires refitting the intercept.
        result = _nonnegative_fit([[1, 1], [1, 2], [1, 3]], [3, 2, 1])
        np.testing.assert_allclose(result, [2, 0])

    def test_unsupported_auto_preserves_collision(self):
        """Fall back to ordinary collision on a CPU pipeline without losing contacts."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body=body, hx=0.5, hy=0.5, hz=0.5)
        builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        with patch.dict(os.environ, {"NEWTON_NARROW_PHASE_COHERENT_CONVEX": "auto"}):
            pipeline = newton.CollisionPipeline(model)
        self.assertIsNone(pipeline.narrow_phase._coherent_cache)
        self.assertEqual(pipeline._convex_selection_reason, "unsupported_pipeline")
        contacts = pipeline.contacts()
        pipeline.collide(model.state(), contacts)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


for device in get_test_devices():
    if wp.get_device(device).is_cuda:
        add_function_test(
            TestAutoConvexSelection,
            "test_graph_choice_preserves_queries",
            test_graph_choice_preserves_queries,
            devices=[device],
        )
        add_function_test(
            TestAutoConvexSelection,
            "test_dense_pipeline_uses_ordinary",
            test_dense_pipeline_uses_ordinary,
            devices=[device],
        )

if __name__ == "__main__":
    unittest.main()
