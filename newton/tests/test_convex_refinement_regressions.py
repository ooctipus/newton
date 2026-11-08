# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check convex collisions with non-hull source triangles, triangle seams, wide grounds and resting stacks."""

import unittest
from itertools import product
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _contacts(
    test,
    device,
    half,
    vertices,
    indices,
    pose,
    scale,
    *,
    swap=False,
    split=False,
    margin=0.0,
    triangle=False,
    hull_box=False,
    ground=None,
):
    """Return physical witnesses and box-to-mesh normals through the public pipeline.

    With ``hull_box``, the static box is an eight-corner convex hull instead of a box primitive.
    ``ground`` is the static box's pose (identity by default).
    """
    ground = wp.transform_identity() if ground is None else ground
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3)))
    cfg = builder.ShapeConfig(margin=margin, gap=1.0e-3, density=0.0)
    mesh = newton.Mesh(vertices, np.asarray(indices, dtype=np.int32).ravel(), compute_inertia=False)
    box_index = -1
    for box in (False, True) if swap else (True, False):
        if box and hull_box:
            hull = newton.Mesh.create_box(
                *half, duplicate_vertices=False, compute_normals=False, compute_uvs=False, compute_inertia=False
            )
            box_index = builder.add_shape_convex_hull(-1, xform=ground, mesh=hull, cfg=cfg)
        elif box:
            box_index = builder.add_shape_box(-1, xform=ground, hx=half[0], hy=half[1], hz=half[2], cfg=cfg)
        elif triangle:
            builder.add_shape_mesh(body, mesh=mesh, scale=scale, cfg=cfg)
        else:
            builder.add_shape_convex_hull(body, mesh=mesh, scale=scale, cfg=cfg)
    model = builder.finalize(device)
    threshold = 0 if split else 10**9
    with patch("newton._src.sim.collide._SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD", threshold):
        pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
    test.assertEqual(pipeline.narrow_phase.split_gjk_mpr, split)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    normals = contacts.rigid_contact_normal.numpy()[:count]
    points0 = contacts.rigid_contact_point0.numpy()[:count]
    points1 = contacts.rigid_contact_point1.numpy()[:count]
    box_points, mesh_points, directions = [], [], []
    for i in range(count):
        box_first = shape0[i] == box_index
        box_points.append(points0[i] if box_first else points1[i])
        local = points1[i] if box_first else points0[i]
        mesh_points.append(np.asarray(wp.transform_point(pose, wp.vec3(local))))
        directions.append(normals[i] if box_first else -normals[i])
    return np.asarray(box_points), np.asarray(mesh_points), np.asarray(directions)


def test_refinement_source_topology(test, device):
    """Keep correct contacts when the same support vertices have concave or incomplete triangles."""
    from scipy.spatial import ConvexHull

    vertices = np.array(
        [
            [-0.4000000059604645, -0.30000001192092896, -0.20000000298023224],
            [0.6000000238418579, -0.20000000298023224, -0.25],
            [0.15000000596046448, 0.699999988079071, -0.30000001192092896],
            [0.05000000074505806, 0.10000000149011612, 0.699999988079071],
            [0.10916666686534882, 0.07041666656732559, -0.1431249976158142],
            [0.09083333611488342, -0.03958333283662796, 0.04020833224058151],
            [0.008333333767950535, 0.12541666626930237, 0.031041666865348816],
            [0.19166666269302368, 0.14374999701976776, 0.02187499962747097],
        ],
        dtype=np.float32,
    )
    hull = ConvexHull(vertices)
    half = np.array([0.45, 0.35, 0.3])
    pose = wp.transform(
        wp.vec3([-0.5678544640541077, -0.4294249415397644, -0.5421493649482727]),
        wp.quat([-0.7031238079071045, 0.21531128883361816, 0.2050347626209259, -0.6459246873855591]),
    )
    partner = np.array([wp.transform_point(pose, wp.vec3(vertex)) for vertex in vertices])
    corners = np.array(list(product(*[(-value, value) for value in half])))
    planes = ConvexHull((corners[:, None] - partner[None, :]).reshape(-1, 3)).equations
    plane = planes[np.argmax(planes[:, 3])]
    topologies = (
        hull.simplices,
        np.array(
            [
                [2, 1, 4],
                [1, 0, 4],
                [0, 2, 4],
                [3, 1, 5],
                [1, 0, 5],
                [0, 3, 5],
                [3, 2, 6],
                [2, 0, 6],
                [0, 3, 6],
                [3, 2, 7],
                [2, 1, 7],
                [1, 3, 7],
            ]
        ),
        hull.simplices[:1],
    )
    for split, swap, index in product(
        (False, True) if device.is_cuda else (False,), (False, True), range(len(topologies))
    ):
        with test.subTest(split=split, swap=swap, topology=index):
            pa, pb, normals = _contacts(
                test,
                device,
                half,
                vertices,
                topologies[index],
                pose,
                np.ones(3),
                swap=swap,
                split=split,
            )
            test.assertGreater(float(np.min(normals @ plane[:3])), 0.99)
            test.assertAlmostEqual(float(np.max(np.sum((pa - pb) * normals, axis=1))), -plane[3], delta=1.0e-5)


def test_refinement_triangle_box_seam(test, device):
    """Keep seam contacts on the mesh front side with the expected normals and depth."""
    half = np.array([0.085, 0.03, 0.005])
    height, pitch = 0.002, 0.1
    vertices = np.array([[-0.2, -0.2, 0.0], [0.2, -0.2, 0.0], [0.2, 0.2, 0.0], [-0.2, 0.2, 0.0]], dtype=np.float32)
    box_pose = wp.transform(wp.vec3(0.0, 0.0, height), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), pitch))
    expected_depth = half[0] * np.sin(pitch) + half[2] * np.cos(pitch) - height
    # Besides the mesh face, a triangle's seam can contact the tilted box edge.
    # Its normal is perpendicular to the diagonal and the box's rotated X edge.
    edge_normal = wp.normalize(wp.vec3(-np.sin(pitch), np.sin(pitch), -np.cos(pitch)))
    frames = (
        wp.transform_identity(),
        wp.transform(wp.vec3(0.3, -0.2, 0.4), wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.4, 0.2)), 0.35)),
    )
    # Reverse shape insertion while also rotating the scene, so the front side
    # cannot accidentally be inferred from world Z or contact shape order.
    for swap, frame in enumerate(frames):
        with test.subTest(swap=bool(swap)):
            pose = wp.transform_multiply(frame, box_pose)
            pa, pb, normals = _contacts(
                test,
                device,
                half,
                vertices,
                [0, 1, 2, 0, 2, 3],
                frame,
                np.ones(3),
                swap=bool(swap),
                triangle=True,
                ground=pose,
            )
            rotation = wp.transform_get_rotation(frame)
            expected_normals = np.array(
                [
                    wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, -1.0)),
                    wp.quat_rotate(rotation, edge_normal),
                ]
            )
            error = np.linalg.norm(normals[:, None, :] - expected_normals[None, :, :], axis=2)
            np.testing.assert_allclose(np.min(error, axis=1), 0.0, atol=1.0e-5)
            depths = np.sum((pa - pb) * normals, axis=1)
            test.assertAlmostEqual(float(np.max(depths)), expected_depth, delta=1.0e-5)
            local_mesh = np.array([wp.transform_point(wp.transform_inverse(frame), wp.vec3(point)) for point in pb])
            np.testing.assert_allclose(local_mesh[:, 2], 0.0, atol=1.0e-5)
            local_box = np.array([wp.transform_point(wp.transform_inverse(pose), wp.vec3(point)) for point in pa])
            np.testing.assert_allclose(np.max(np.abs(local_box) - half, axis=1), 0.0, atol=1.0e-5)


def test_refinement_triangle_above_wide_ground(test, device):
    """Report the true gap for a small triangle hovering above a wide ground.

    The witness points sum support coordinates hundreds of meters long, so a
    normal rebuilt from them tilted by a few degrees for a millimeter gap. The
    manifold then measured depth along that normal to a far ground corner and
    reported contacts meters deep, launching objects resting on the ground.
    The split GJK/MPR kernels take only generic convex pairs, so they see the
    triangle as a flat convex hull.
    """
    # Mesh triangles captured from assets resting on a wide ground, relative to its top face [m].
    cases = {
        "toaster": (
            [
                [-0.116287656, 0.057376586, 0.000993134],
                [-0.116287656, 0.058698855, 0.000993129],
                [-0.115889244, 0.058659613, 0.000993134],
            ],
            np.array([500.0, 500.0, 5.0]),
        ),
        "hammer": (
            [
                [-1.2480534e-02, 2.1553008e-01, 1.3521127e-04],
                [-6.1373673e-03, 2.0449370e-01, 1.2469478e-04],
                [-1.2529907e-02, 2.0449370e-01, 1.2479909e-04],
            ],
            np.array([50.0, 50.0, 1.0]),
        ),
    }
    level = wp.transform_identity()
    tilted = wp.transform(wp.vec3(0.0), wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.4, 0.0)), 0.35))
    # (mesh triangle, split kernels, hull_box, swap, ground): the mesh path sweeps both
    # ground shapes and insertion orders; the flat-hull paths and the tilted ground run once.
    variants = [(True, False, hull_box, swap, level) for hull_box, swap in product((False, True), (False, True))]
    variants += [(False, split, False, False, level) for split in ((False, True) if device.is_cuda else (False,))]
    variants.append((True, False, True, True, tilted))
    for (name, (points, half)), (mesh, split, hull_box, swap, ground) in product(cases.items(), variants):
        with test.subTest(triangle=name, mesh=mesh, split=split, hull_box=hull_box, swap=swap, tilted=ground != level):
            vertices = np.array(points, dtype=np.float32)
            pose = wp.transform_multiply(ground, wp.transform(wp.vec3(0.0, 0.0, half[2]), wp.quat_identity()))
            pa, pb, normals = _contacts(
                test,
                device,
                half,
                vertices,
                [0, 1, 2],
                pose,
                np.ones(3),
                swap=swap,
                split=split,
                triangle=mesh,
                hull_box=hull_box,
                ground=ground,
            )
            up = np.array(wp.quat_rotate(ground.q, wp.vec3(0.0, 0.0, 1.0)))
            np.testing.assert_allclose(normals, np.tile(up, (len(normals), 1)), atol=1.0e-4)
            # Witness sums round at the ground's float32 spacing; on a tilted ground
            # part of that rounding lies along the normal.
            slack = 1.0e-5 + 2.0 * float(np.spacing(np.float32(np.max(half))))
            gaps = np.sum((pb - pa) * normals, axis=1)
            test.assertGreaterEqual(float(np.min(gaps)), float(np.min(vertices[:, 2])) - slack)
            test.assertLessEqual(float(np.max(gaps)), float(np.max(vertices[:, 2])) + slack)


def test_refinement_resting_box_stack(test, device):
    """Report face contacts for centered, nearly aligned resting cubes.

    Centered support ties make MPR's region tests evaluate to float32 noise and
    accept a sliver portal that misses the origin ray, which used to report a
    single tilted contact 0.155 m deep.
    """
    lower = wp.transform(
        wp.vec3(-1.2568341e-03, -4.8650557e-04, 1.2503407e00),
        wp.quat(1.9415199e-04, -4.9863686e-04, 2.3975765e-05, 9.9999988e-01),
    )
    upper = wp.transform(
        wp.vec3(-1.7543929e-03, -6.8072625e-04, 1.7503396e00),
        wp.quat(1.9580426e-04, -4.9785688e-04, 2.5186999e-05, 9.9999988e-01),
    )
    for split, swap in product((False, True) if device.is_cuda else (False,), (False, True)):
        with test.subTest(split=split, swap=swap):
            builder = newton.ModelBuilder()
            for pose in (upper, lower) if swap else (lower, upper):
                body = builder.add_body(xform=pose)
                builder.add_shape_box(body, hx=0.25, hy=0.25, hz=0.25)
            model = builder.finalize(device)
            threshold = 0 if split else 10**9
            with patch("newton._src.sim.collide._SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD", threshold):
                pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
            test.assertEqual(pipeline.narrow_phase.split_gjk_mpr, split)
            contacts = pipeline.contacts()
            state = model.state()
            pipeline.collide(state, contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            test.assertGreaterEqual(count, 4)
            body_q = state.body_q.numpy()
            shape0 = contacts.rigid_contact_shape0.numpy()[:count]
            shape1 = contacts.rigid_contact_shape1.numpy()[:count]
            points0 = contacts.rigid_contact_point0.numpy()[:count]
            points1 = contacts.rigid_contact_point1.numpy()[:count]
            normals = contacts.rigid_contact_normal.numpy()[:count]
            for i in range(count):
                pa = np.asarray(wp.transform_point(wp.transform(*body_q[shape0[i]]), wp.vec3(points0[i])))
                pb = np.asarray(wp.transform_point(wp.transform(*body_q[shape1[i]]), wp.vec3(points1[i])))
                # Normals point from shape 0 to shape 1, so the lower cube decides the sign.
                up = 1.0 if body_q[shape0[i]][2] < body_q[shape1[i]][2] else -1.0
                test.assertGreater(float(normals[i][2] * up), 0.99)
                test.assertAlmostEqual(float(np.dot(pb - pa, normals[i])), 0.0, delta=1.0e-4)


class TestConvexRefinementRegressions(unittest.TestCase):
    pass


for function in (
    test_refinement_source_topology,
    test_refinement_triangle_box_seam,
    test_refinement_triangle_above_wide_ground,
    test_refinement_resting_box_stack,
):
    add_function_test(TestConvexRefinementRegressions, function.__name__, function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
