# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cover convex contacts at sampled lines, near-planar faces, resting boxes and shallow cubes."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.mpr import create_solve_mpr
from newton._src.geometry.support_function import (
    GenericShapeData,
    SupportMapDataProvider,
    support_map,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestConvexContactWitness(unittest.TestCase):
    pass


def test_convex_contact_sampled_line(test, device):
    """Keep the deepest contact when sampled endpoints lie below the supporting plane."""
    vertices = np.array([[-1, 0, 0], [1, 0, 0], [0, 0, 0.001], [0, -0.1, -0.1], [0, 0.1, -0.1]], dtype=np.float32)
    indices = np.array([[0, 4, 3], [1, 3, 4], [0, 3, 2], [1, 2, 3], [0, 2, 4], [1, 4, 2]], dtype=np.int32)
    mesh = newton.Mesh(vertices, indices.reshape(-1))
    for swap in (False, True):
        builder = newton.ModelBuilder()
        cfg = builder.ShapeConfig(margin=0.0, gap=0.0)
        hull_body = builder.add_body()
        box_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1005), wp.quat_identity()))
        if swap:
            builder.add_shape_box(box_body, hx=2.0, hy=1.0, hz=0.1, cfg=cfg)
        builder.add_shape_convex_hull(hull_body, mesh=mesh, cfg=cfg)
        if not swap:
            builder.add_shape_box(box_body, hx=2.0, hy=1.0, hz=0.1, cfg=cfg)
        model = builder.finalize(device=device)
        for split in (False, True) if device.is_cuda else (False,):
            with test.subTest(swap=swap, split=split):
                threshold = 0 if split else 1_000_000_000
                with patch("newton._src.sim.collide._SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD", threshold):
                    pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
                contacts = pipeline.contacts()
                pipeline.collide(model.state(), contacts)
                count = int(contacts.rigid_contact_count.numpy()[0])
                test.assertEqual(count, 1)
                shape0 = int(contacts.rigid_contact_shape0.numpy()[0])
                points = [contacts.rigid_contact_point0.numpy()[0], contacts.rigid_contact_point1.numpy()[0]]
                hull_point, box_point = points if shape0 == int(swap) else points[::-1]
                np.testing.assert_allclose(hull_point, [0.0, 0.0, 0.001], atol=1.0e-5)
                np.testing.assert_allclose(box_point, [0.0, 0.0, -0.1], atol=1.0e-5)
                normal = contacts.rigid_contact_normal.numpy()[0]
                test.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0, delta=1.0e-6)
                test.assertAlmostEqual(abs(float(normal[2])), 1.0, delta=1.0e-6)


def test_convex_contact_near_planar_face(test, device):
    """Report every corner of a resting face that is not planar to EPS, at its exact depth."""
    # A sole-sized box hull whose first bottom corner sits 20 um above the
    # other three. All four corners penetrate a slab, so each must get a
    # contact on both surfaces instead of the deepest witness alone.
    half = np.array([0.0205, 0.027, 0.0065])
    vertices = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)]) * half
    bump = 2.0e-5
    vertices[0, 2] += bump
    mesh = newton.Mesh(vertices, [0, 1, 2], compute_inertia=False).compute_convex_hull()
    depth = 2.0e-4
    corners = vertices[:4]
    corner_depths = depth - (corners[:, 2] + half[2])
    for swap in (False, True):
        builder = newton.ModelBuilder()
        cfg = builder.ShapeConfig(margin=0.0, gap=0.0)
        hull_body = builder.add_body()
        slab_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, depth - half[2] - 0.1), wp.quat_identity()))
        if swap:
            builder.add_shape_box(slab_body, hx=0.5, hy=0.5, hz=0.1, cfg=cfg)
        builder.add_shape_convex_hull(hull_body, mesh=mesh, cfg=cfg)
        if not swap:
            builder.add_shape_box(slab_body, hx=0.5, hy=0.5, hz=0.1, cfg=cfg)
        model = builder.finalize(device=device)
        state = model.state()
        for split in (False, True) if device.is_cuda else (False,):
            with test.subTest(swap=swap, split=split):
                threshold = 0 if split else 1_000_000_000
                with patch("newton._src.sim.collide._SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD", threshold):
                    pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
                contacts = pipeline.contacts()
                pipeline.collide(state, contacts)
                count = int(contacts.rigid_contact_count.numpy()[0])
                distances_array = wp.empty(contacts.rigid_contact_max, dtype=float, device=device)
                newton.eval_rigid_contact_kinematics(model, state, contacts, out_distance=distances_array)
                shape0 = contacts.rigid_contact_shape0.numpy()[:count]
                points = [contacts.rigid_contact_point0.numpy()[:count], contacts.rigid_contact_point1.numpy()[:count]]
                hull_points = np.where((shape0 == int(swap))[:, None], points[0], points[1])
                distances = distances_array.numpy()[:count]
                for corner, corner_depth in zip(corners, corner_depths, strict=True):
                    nearest = int(np.argmin(np.linalg.norm(hull_points - corner, axis=1)))
                    np.testing.assert_allclose(hull_points[nearest], corner, atol=1.0e-5)
                    test.assertAlmostEqual(float(distances[nearest]), -corner_depth, delta=1.0e-6)
                test.assertTrue(np.all(distances < 0.0))

                # Reuse the pair across touching, separated, and out-of-gap
                # states. The writer must apply the same gap policy as for a
                # planar manifold, including rejecting just the raised corner.
                gap = 1.0e-4
                model.shape_gap.fill_(gap)
                for separation in (0.0, 5.0e-5, 1.9e-4, 2.1e-4):
                    with test.subTest(separation=separation):
                        poses = state.body_q.numpy()
                        poses[slab_body, 2] = -separation - half[2] - 0.1
                        state.body_q.assign(poses)
                        pipeline.collide(state, contacts)
                        count = int(contacts.rigid_contact_count.numpy()[0])
                        expected = separation + corners[:, 2] + half[2] < 2.0 * gap
                        if not np.any(expected):
                            test.assertEqual(count, 0)
                            continue
                        test.assertGreaterEqual(count, int(np.sum(expected)))
                        test.assertLessEqual(count, 5)
                        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
                        points = [
                            contacts.rigid_contact_point0.numpy()[:count],
                            contacts.rigid_contact_point1.numpy()[:count],
                        ]
                        hull_points = np.where((shape0 == int(swap))[:, None], points[0], points[1])
                        newton.eval_rigid_contact_kinematics(model, state, contacts, out_distance=distances_array)
                        measured = distances_array.numpy()[:count]
                        test.assertTrue(np.all(measured < 2.0 * gap))
                        for corner in corners[expected]:
                            nearest = int(np.argmin(np.linalg.norm(hull_points - corner, axis=1)))
                            np.testing.assert_allclose(hull_points[nearest], corner, atol=1.0e-5)
                            test.assertAlmostEqual(
                                float(measured[nearest]), separation + corner[2] + half[2], delta=1.0e-6
                            )
                # Restore the penetrating state before testing the other kernel.
                poses[slab_body, 2] = depth - half[2] - 0.1
                state.body_q.assign(poses)
                model.shape_gap.zero_()


for function in (test_convex_contact_sampled_line, test_convex_contact_near_planar_face):
    add_function_test(TestConvexContactWitness, function.__name__, function, devices=get_test_devices())


@wp.kernel(enable_backward=False)
def _resting_box_kernel(
    half_a: wp.vec3,
    half_b: wp.vec3,
    rotation: wp.quat,
    position: wp.vec3,
    normal: wp.array[wp.vec3],
    depth: wp.array[float],
):
    shape_a = GenericShapeData()
    shape_a.shape_type = int(newton.GeoType.BOX)
    shape_a.scale = half_a
    shape_a.shape_index = -1
    shape_b = shape_a
    shape_b.scale = half_b
    _collision, _pa, _pb, n, penetration = wp.static(create_solve_mpr(support_map).core)(
        shape_a, shape_b, rotation, position, 1.0e-4, SupportMapDataProvider()
    )
    normal[0] = n
    depth[0] = penetration - 1.0e-4


def test_convex_contact_resting_box(test, device):
    """Keep a centered face contact on its face instead of a retry's distant side face."""
    # Pose captured from a box resting on a long slab in the friction-ramp test.
    normal = wp.zeros(1, dtype=wp.vec3, device=device)
    depth = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        _resting_box_kernel,
        dim=1,
        inputs=[
            wp.vec3(0.5, 2.5, 0.05),
            wp.vec3(0.2, 0.2, 0.05),
            wp.quat(3.069639205932617e-06, 5.119977219969485e-10, -4.247564105996915e-12, 1.0),
            wp.vec3(0.0, -0.00018496252596378326, 0.09996524453163147),
        ],
        outputs=[normal, depth],
        device=device,
    )
    np.testing.assert_allclose(normal.numpy()[0], (0.0, 0.0, 1.0), atol=1.0e-4)
    test.assertAlmostEqual(float(depth.numpy()[0]), 0.1 - 0.09996524453163147, delta=1.0e-5)


add_function_test(
    TestConvexContactWitness,
    "test_convex_contact_resting_box",
    test_convex_contact_resting_box,
    devices=get_test_devices(),
)


def _box_minimum_depth(half, rotation, position):
    """Return the FP64 separating-axis depth of two equal boxes, B posed in A's frame."""
    axes_b = np.array([wp.quat_rotate(rotation, wp.vec3(*axis)) for axis in np.eye(3)], dtype=np.float64)
    axes = [*np.eye(3), *axes_b, *(np.cross(a, b) for a in np.eye(3) for b in axes_b)]
    depth = np.inf
    for axis in axes:
        length = np.linalg.norm(axis)
        if length > 1.0e-12:
            unit = axis / length
            radius = half * (np.sum(np.abs(unit)) + np.sum(np.abs(axes_b @ unit)))
            depth = min(depth, radius - abs(float(np.dot(unit, position))))
    return depth


def test_convex_contact_shallow_cubes(test, device, *, split=False):
    """Report no contact deeper than the minimum for shallow cube contacts that need refinement."""
    # Relative poses captured from a collapsed cube pile where refinement once
    # reported a far Minkowski corner, or a normal tilted by the ray shift, and
    # launched cubes. The portal misses the minimum certificate by its
    # convergence tolerance, so refinement runs.
    poses = (
        (
            wp.quat(-0.0011170628713443875, 1.7762567949830554e-05, -0.004657822661101818, 0.9999886751174927),
            wp.vec3(0.4209620952606201, -6.29425048828125e-05, 0.8006205558776855),
        ),
        (
            wp.quat(-0.0001483643427491188, -0.006294019520282745, 1.2986361980438232e-05, 0.9999802112579346),
            wp.vec3(0.8049442768096924, -6.67572021484375e-06, 0.005084991455078125),
        ),
    )
    half = 0.4
    for case, (rotation, position) in enumerate(poses):
        with test.subTest(case=case):
            builder = newton.ModelBuilder()
            builder.add_shape_box(-1, hx=half, hy=half, hz=half)
            body = builder.add_body(xform=wp.transform(position, rotation))
            builder.add_shape_box(body, hx=half, hy=half, hz=half)
            model = builder.finalize(device)
            pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
            test.assertEqual(pipeline.narrow_phase.split_gjk_mpr, split)
            contacts = pipeline.contacts()
            state = model.state()
            pipeline.collide(state, contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            test.assertGreater(count, 0)
            pose = wp.transform(*state.body_q.numpy()[body])
            depths = []
            for contact in range(count):
                points = [
                    contacts.rigid_contact_point0.numpy()[contact],
                    contacts.rigid_contact_point1.numpy()[contact],
                ]
                if int(contacts.rigid_contact_shape0.numpy()[contact]) == 1:
                    points.reverse()
                points[1] = np.asarray(wp.transform_point(pose, wp.vec3(points[1])))
                sign = 1.0 if int(contacts.rigid_contact_shape0.numpy()[contact]) == 0 else -1.0
                normal = sign * contacts.rigid_contact_normal.numpy()[contact]
                depths.append(-float(np.dot(points[1] - points[0], normal)))
            # The deepest contact must match the minimum penetration depth.
            test.assertAlmostEqual(max(depths), _box_minimum_depth(half, rotation, position), delta=1.0e-5)


def test_convex_contact_shallow_cubes_split(test, device):
    """Report no contact deeper than the minimum through the split CUDA collision kernels."""
    if not device.is_cuda:
        test.skipTest("Split collision kernels run only on CUDA")
    with patch("newton._src.sim.collide._SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD", 0):
        test_convex_contact_shallow_cubes(test, device, split=True)


for function in (test_convex_contact_shallow_cubes, test_convex_contact_shallow_cubes_split):
    add_function_test(TestConvexContactWitness, function.__name__, function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
