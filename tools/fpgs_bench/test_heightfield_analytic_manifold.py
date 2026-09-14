# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check certified face witnesses feeding the retained contact manifold."""

import inspect
import json
import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry import heightfield_finite as finite
from newton._src.geometry.collision_convex import ConvexQueryResult
from newton._src.geometry.contact_data import ContactData
from newton._src.geometry.narrow_phase import create_narrow_phase_process_mesh_triangle_contacts_kernel
from newton._src.utils.heightfield import HeightfieldData
from tools.fpgs_bench.test_heightfield_finite_geometry import (
    current_fixture,
    rotation,
    snapshot,
    stream_query_kernel,
    triangle_box_distance,
)


@wp.kernel(enable_backward=False)
def evaluate_witness(
    e1: wp.array[wp.vec3],
    e2: wp.array[wp.vec3],
    center: wp.array[wp.vec3],
    rotation: wp.array[wp.quat],
    half: wp.array[wp.vec3],
    shell: wp.array[wp.vec2],
    accepted: wp.array[int],
    result: wp.array[ConvexQueryResult],
):
    """Evaluate the actual certificate, independently of manifold publication."""
    i = wp.tid()
    ok, value = finite.query_top_face_witness(e1[i], e2[i], center[i], rotation[i], half[i], shell[i][0], shell[i][1])
    accepted[i] = int(ok)
    result[i] = value


@wp.struct
class CollectedContacts:
    count: wp.array[int]
    values: wp.array[ContactData]


@wp.func
def collect_contact(contact: ContactData, data: CollectedContacts, output_index: int):
    index = wp.atomic_add(data.count, 0, 1)
    if index < data.values.shape[0]:
        data.values[index] = contact


def witnesses(cases, device):
    """Run a bounded list of geometry controls through the production function."""
    inputs = [
        wp.array(
            [case[k] for case in cases], dtype=(wp.vec3, wp.vec3, wp.vec3, wp.quat, wp.vec3, wp.vec2)[k], device=device
        )
        for k in range(6)
    ]
    accepted = wp.empty(len(cases), dtype=int, device=device)
    result = wp.empty(len(cases), dtype=ConvexQueryResult, device=device)
    wp.launch(evaluate_witness, len(cases), inputs=inputs, outputs=[accepted, result], device=device)
    return accepted.numpy(), result.numpy()


def check_manifold(device):
    """Exercise the actual writer, marker skip and retained uncertain fallback."""
    hfd = HeightfieldData()
    hfd.nrow = hfd.ncol = 2
    hfd.hx = hfd.hy = 1.0
    hfd.max_z = 1.0
    data = CollectedContacts()
    data.count = wp.zeros(1, dtype=int, device=device)
    data.values = wp.zeros(16, dtype=ContactData, device=device)
    triples = wp.array([[0, 1, 0]], dtype=wp.vec3i, device=device)
    transforms = wp.array(
        [wp.transform_identity(), wp.transform(wp.vec3(0.3, -0.4, 0.13), wp.quat_identity())],
        dtype=wp.transform,
        device=device,
    )
    inputs = [
        wp.array([newton.GeoType.HFIELD, newton.GeoType.BOX], dtype=int, device=device),
        wp.array([[1, 1, 1, 0.01], [0.1, 0.1, 0.1, 0.01]], dtype=wp.vec4, device=device),
        transforms,
        wp.zeros(2, dtype=wp.uint64, device=device),
        wp.full(2, 0.01, dtype=float, device=device),
        wp.array([0, -1], dtype=int, device=device),
        wp.array([hfd], dtype=HeightfieldData, device=device),
        wp.zeros(4, dtype=float, device=device),
        triples,
        wp.ones(1, dtype=int, device=device),
        data,
        1,
    ]
    bounds = wp.array([[[0, 0, 0], [-1, -1, -1]], [[0, 0, 0], [1, 1, 1]]], dtype=wp.vec3, ndim=2, device=device)
    kernel = finite.create_query_kernel(collect_contact, analytic_manifold=True)
    if kernel.key != "heightfield_analytic_manifold_contacts":
        raise AssertionError(kernel.key)
    wp.launch(kernel, 1, inputs=[*inputs, bounds, inputs[3]], device=device)
    count = int(data.count.numpy()[0])
    candidate = data.values.numpy()[:count].copy()
    if count != 4 or int(triples.numpy()[0, 2]) != -1:
        raise AssertionError(("Original four-point manifold not consumed", count, triples.numpy()))
    fallback = finite.marked_fallback(create_narrow_phase_process_mesh_triangle_contacts_kernel(collect_contact))
    wp.launch(fallback, 1, inputs=inputs, device=device)
    if int(data.count.numpy()[0]) != count:
        raise AssertionError("Accepted analytic entry still reached MPR/GJK")
    # Compare this simple, nondegenerate original manifold, not arbitrary raw counts.
    data.count.zero_()
    triples.assign(np.array([[0, 1, 0]], dtype=np.int32))
    wp.launch(fallback, 1, inputs=inputs, device=device)
    original = data.values.numpy()[: int(data.count.numpy()[0])]
    for field in ("contact_point_center", "contact_distance", "sort_sub_key"):
        order_a = np.argsort(candidate["sort_sub_key"])
        order_b = np.argsort(original["sort_sub_key"])
        np.testing.assert_allclose(candidate[field][order_a], original[field][order_b], rtol=0, atol=2e-6)
    # The authored flat plane, not GJK's approximate normal, is the geometric
    # oracle. Keep the same tolerance and record the original approximation.
    plane = np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(candidate["contact_normal_a_to_b"], np.tile(plane, (count, 1)), rtol=0, atol=2e-6)
    print(
        "ANALYTIC_MANIFOLD_NORMAL_DIAGNOSTIC "
        + json.dumps(
            {
                "device": str(device),
                "candidate_plane_max_error": float(
                    np.max(np.linalg.norm(candidate["contact_normal_a_to_b"] - plane, axis=1))
                ),
                "original_plane_max_error": float(
                    np.max(np.linalg.norm(original["contact_normal_a_to_b"] - plane, axis=1))
                ),
            }
        ),
        flush=True,
    )
    for field in ("shape_a", "shape_b", "margin_a", "margin_b", "gap_sum", "radius_eff_a", "radius_eff_b"):
        np.testing.assert_array_equal(candidate[field], original[field])
    # The supporting face center is exactly on the diagonal: retain old fallback.
    transforms.assign([wp.transform_identity(), wp.transform(wp.vec3(0.0, 0.0, 0.13), wp.quat_identity())])
    data.count.zero_()
    triples.assign(np.array([[0, 1, 0]], dtype=np.int32))
    wp.launch(kernel, 1, inputs=[*inputs, bounds, inputs[3]], device=device)
    if int(data.count.numpy()[0]) != 0 or int(triples.numpy()[0, 2]) != 0:
        raise AssertionError("Uncertain face boundary bypassed the old query")
    wp.launch(fallback, 1, inputs=inputs, device=device)
    if int(data.count.numpy()[0]) == 0:
        raise AssertionError("Uncertain entry lost its original manifold")


class TestAnalyticManifold(unittest.TestCase):
    def test_witness_api(self):
        """Expose a guarded witness and explicit original-manifold dispatch."""
        self.assertTrue(callable(finite.query_top_face_witness))
        self.assertTrue(callable(finite.create_query_kernel))

    def test_certified_closest_and_uncertain_cpu(self):
        """Check distance, true surfaces, slopes and conservative boundaries."""
        identity = (0, 0, 0, 1)
        flat = ((2, 0, 0), (0, 2, 0), (0.4, 0.4, 0.13), identity, (0.1, 0.1, 0.1), (0.04, 0.02))
        cases = [flat]
        for center in ((0.0, 0.4, 0.13), (1.0, 1.0, 0.13), (1.5, 1.5, 0.13), (0.4, 0.4, 0.12), (0.4, 0.4, 0.14)):
            cases.append((*flat[:2], center, *flat[3:]))
        # Construct a sloped, rotated box with its support .03m above an interior point.
        e1, e2 = np.array([2.0, 0, 0.2]), np.array([0, 2.0, 0.3])
        normal = np.cross(e1, e2)
        normal /= np.linalg.norm(normal)
        quat = (0, np.sin(0.17), 0, np.cos(0.17))
        rot = rotation(quat)
        support = rot @ (-np.sign(rot.T @ normal) * 0.1)
        center = 0.2 * (e1 + e2) + 0.03 * normal - support
        cases.append((e1, e2, center, quat, flat[4], flat[5]))
        accepted, values = witnesses(cases, "cpu")
        np.testing.assert_array_equal(accepted, [1, 0, 0, 0, 0, 0, 1])
        for index in np.flatnonzero(accepted):
            e1, e2, center, quat, half, _shell = cases[index]
            value = values[index]
            pa, pb, n = (value[name].astype(float) for name in ("point_a", "point_b", "normal"))
            d = float(value["signed_distance"])
            np.testing.assert_allclose(pb - pa, d * n, atol=2e-6)
            self.assertAlmostEqual(float(n @ pa), 0, delta=2e-6)
            local = rotation(quat).T @ (pb - center)
            self.assertLessEqual(float(np.max(np.abs(local) - half)), 2e-6)
            self.assertLess(float(np.min(np.abs(np.abs(local) - half))), 2e-6)
            independent = triangle_box_distance(
                np.array([[0, 0, 0], e1, e2]), np.array(center), rotation(quat), np.array(half)
            )
            self.assertAlmostEqual(d, independent, delta=2e-6)

    def test_original_manifold_and_fallback_cpu(self):
        """Require real original manifold publication and no duplicate generic call."""
        source = inspect.getsource(finite.create_query_kernel.__wrapped__)
        self.assertIn("create_write_convex_query_result(support_map", source)
        self.assertIn("wp.static(write_manifold)", source)
        self.assertIn("flat_seam_query_allowed(geom, witness.point_a, witness.normal", source)
        check_manifold("cpu")

    def test_saved96_witness_admission_cpu(self):
        """Count actual certified entries without assigning query counts to time."""
        reports = []
        for gpu in (0, 1):
            model, state, pairs = current_fixture(gpu, "cpu")
            with patch.dict(os.environ, NEWTON_HEIGHTFIELD_CELL_REJECT="1", NEWTON_HEIGHTFIELD_FINITE_QUERY="1"):
                pipeline = newton.CollisionPipeline(
                    model,
                    shape_pairs_filtered=pairs,
                    reduce_contacts=False,
                    rigid_contact_max=32768,
                    max_triangle_pairs=32768,
                )
            contacts = pipeline.contacts()
            pipeline.collide(state, contacts)
            initial = snapshot(pipeline, state, contacts)
            triples = initial["triples"]
            count = len(triples)
            kernel, result_type = stream_query_kernel(analytic_manifold=False)
            output = wp.empty(count, dtype=result_type, device="cpu")
            vertices = wp.empty((count, 3), dtype=wp.vec3, device="cpu")
            centers = wp.empty(count, dtype=wp.vec3, device="cpu")
            quats = wp.empty(count, dtype=wp.quat, device="cpu")
            wp.launch(
                kernel,
                count,
                inputs=[
                    wp.array(triples, dtype=wp.vec3i, device="cpu"),
                    pipeline.geom_transform,
                    model.heightfield_data,
                    model.heightfield_elevations,
                    model.shape_scale,
                    model.shape_gap,
                    model.shape_margin,
                ],
                outputs=[output, vertices, centers, quats],
                device="cpu",
            )
            margins, gaps = model.shape_margin.numpy(), model.shape_gap.numpy()
            half = model.shape_scale.numpy()[triples[:, 1]] * 0.5
            tri = vertices.numpy()
            cases = [
                (
                    tri[i, 1],
                    tri[i, 2],
                    center,
                    quat,
                    half[i],
                    (gaps[0] + gaps[shape] + margins[0] + margins[shape], margins[0] + margins[shape]),
                )
                for i, (center, quat, shape) in enumerate(
                    zip(centers.numpy(), quats.numpy(), triples[:, 1], strict=True)
                )
            ]
            accepted, _values = witnesses(cases, "cpu")
            fallback = output.numpy()[:, 0] < 0
            self.assertFalse(np.any((accepted != 0) & ~fallback), "Certificate changed a non-fallback finite query")
            reports.append(
                {
                    "fixture": gpu,
                    "triangles": count,
                    "original_fallback": int(fallback.sum()),
                    "certified": int(accepted.sum()),
                    "certified_shapes": int(len(np.unique(triples[accepted != 0, 1]))),
                }
            )
        print("ANALYTIC_WITNESS_ADMISSION " + json.dumps(reports), flush=True)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires the root-owned paired GPU lease")
    def test_native_original_manifold_and_fallback_cuda(self):
        """Execute certified manifold and uncertain fallback on the leased device."""
        check_manifold("cuda:0")


if __name__ == "__main__":
    unittest.main()
