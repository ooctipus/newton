# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cover contact normals and footprints for convex shapes."""

import gc
import unittest
from itertools import product
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.collision_core import post_process_minkowski_only
from newton._src.geometry.contact_data import ContactData
from newton._src.geometry.multicontact import create_build_manifold
from newton._src.geometry.support_function import GenericShapeData, SupportMapDataProvider, support_map
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestBoxContactNormal(unittest.TestCase):
    pass


def test_box_contact_split_reuse(test, device):
    """Reuse captured box work across overlap, separation, and empty queues."""
    if not device.is_cuda:
        test.skipTest("Split collision kernels run only on CUDA")
    half = (0.0126257, 0.0126257, 0.0252514)
    for use_mesh in (False, True):
        builder = newton.ModelBuilder()
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.585, 0.0, 0.215)),
            hx=0.3675,
            hy=0.61,
            hz=0.015,
            cfg=builder.ShapeConfig(gap=0.01),
        )
        body = builder.add_body(xform=wp.transform(wp.vec3(0.735, -0.35, 0.2326)))
        if use_mesh:
            builder.add_shape_convex_hull(
                body, mesh=newton.Mesh.create_box(*half, duplicate_vertices=False), cfg=builder.ShapeConfig(gap=0.001)
            )
        else:
            builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2], cfg=builder.ShapeConfig(gap=0.001))
        model = builder.finalize(device)
        with patch("newton._src.sim.collide._SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD", 0):
            pipeline = newton.CollisionPipeline(model, broad_phase="sap")
        state = model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        # Cyclic GC can destroy SDF textures from earlier scenes during capture.
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            with wp.ScopedCapture(device=device) as capture:
                pipeline.collide(state, contacts)
        finally:
            if gc_was_enabled:
                gc.enable()
        poses = state.body_q.numpy()
        for height in (0.2326, 0.256, 0.27, 0.2326):
            with test.subTest(mesh=use_mesh, height=height):
                poses[body, 2] = height
                state.body_q.assign(poses)
                wp.capture_launch(capture.graph)
                count = int(contacts.rigid_contact_count.numpy()[0])
                if height == 0.27:
                    test.assertEqual(count, 0)
                    np.testing.assert_array_equal(pipeline.narrow_phase.split_gjk_work_count.numpy(), (0, 0))
                    continue
                test.assertGreater(count, 0)
                normals = contacts.rigid_contact_normal.numpy()[:count]
                # Normalizing a submillimeter GJK gap amplifies float32
                # cancellation in the world-space witness subtraction.
                normal_tolerance = 3.0e-4 if height == 0.256 else 1.0e-4
                np.testing.assert_allclose(
                    normals, np.broadcast_to((0.0, 0.0, 1.0), normals.shape), atol=normal_tolerance
                )
                table = contacts.rigid_contact_point0.numpy()[:count]
                partner = contacts.rigid_contact_point1.numpy()[:count] + poses[body, :3]
                depth = np.sum((table - partner) * normals, axis=1)
                np.testing.assert_allclose(depth, 0.23 - height + half[2], atol=1.0e-5)


add_function_test(
    TestBoxContactNormal, "test_box_contact_split_reuse", test_box_contact_split_reuse, devices=get_test_devices()
)


def test_small_convex_contact_footprint(test, device, *, split=False):
    """Preserve both sides of submillimeter contact faces in either shape order."""
    for scale, geometry, swap in product((1.0, 10.0), ("box", "cylinder_face", "cylinder_edge"), (False, True)):
        with test.subTest(scale=scale, geometry=geometry, swap=swap):
            builder = newton.ModelBuilder()
            cylinder = geometry != "box"
            phase = wp.pi / 64 if geometry == "cylinder_face" else 0.0
            rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -wp.pi / 2)
            rotation *= wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), phase)
            radius = 0.00725 * scale
            half_width = radius * np.sin(phase) if cylinder else 0.0004 * scale
            half_height = 0.024375 * scale
            bottom = radius * np.cos(phase) if cylinder else radius
            pose = wp.transform(wp.vec3(0.585, -0.22, 0.23 + bottom - 1.0e-5 * scale), rotation)
            body = builder.add_body(xform=pose)
            mesh = (
                newton.Mesh.create_cylinder(radius, half_height, up_axis=newton.Axis.Z, segments=64)
                if cylinder
                else newton.Mesh.create_box(half_width, radius, half_height, duplicate_vertices=False)
            )
            for is_table in (False, True) if swap else (True, False):
                if is_table:
                    table_shape = builder.add_shape_box(
                        -1, xform=wp.transform(wp.vec3(0.585, 0.0, 0.215)), hx=0.3675, hy=0.61, hz=0.015
                    )
                else:
                    builder.add_shape_convex_hull(body, mesh=mesh)
            model = builder.finalize(device)
            pipeline = newton.CollisionPipeline(model)
            test.assertEqual(pipeline.narrow_phase.split_gjk_mpr, split)
            contacts = pipeline.contacts()
            pipeline.collide(model.state(), contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            if geometry == "cylinder_edge":
                test.assertEqual(count, 2)
            else:
                test.assertGreaterEqual(count, 4)
            shape0 = contacts.rigid_contact_shape0.numpy()[:count]
            point0 = contacts.rigid_contact_point0.numpy()[:count]
            point1 = contacts.rigid_contact_point1.numpy()[:count]
            object_points = np.where((shape0 == table_shape)[:, None], point1, point0)
            world_points = np.array([wp.transform_point(pose, wp.vec3(p)) for p in object_points])
            # The complete footprint must bracket the center of mass on both axes.
            np.testing.assert_allclose(
                [world_points[:, 0].min(), world_points[:, 0].max()],
                [0.585 - half_width, 0.585 + half_width],
                atol=max(2.0e-6 * scale, 1.1e-5 if geometry == "cylinder_edge" else 0.0),
                rtol=0.0,
            )
            np.testing.assert_allclose(
                [world_points[:, 1].min(), world_points[:, 1].max()],
                [-0.22 - half_height, -0.22 + half_height],
                atol=2.0e-6 * scale,
                rtol=0.0,
            )
            np.testing.assert_allclose(world_points[:, 2], 0.23 - 1.0e-5 * scale, atol=2.0e-6 * scale)


add_function_test(
    TestBoxContactNormal,
    "test_small_convex_contact_footprint",
    test_small_convex_contact_footprint,
    devices=get_test_devices(),
)


def test_small_convex_contact_footprint_split(test, device):
    """Preserve small contact faces through the split CUDA kernels."""
    if not device.is_cuda:
        test.skipTest("Split collision kernels run only on CUDA")
    with patch("newton._src.sim.collide._SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD", 0):
        test_small_convex_contact_footprint(test, device, split=True)


add_function_test(
    TestBoxContactNormal,
    "test_small_convex_contact_footprint_split",
    test_small_convex_contact_footprint_split,
    devices=get_test_devices(),
)


def test_capsule_box_edge_contact(test, device, *, split=False):
    """Keep one capsule endpoint contact and preserve an overlapping segment."""
    # Regression for #4756, resolved by the GJK convergence changes in #4771.
    for center_x, expected_x in ((-0.075, [-0.05]), (-0.025, [-0.05, 0.0])):
        with test.subTest(center_x=center_x):
            builder = newton.ModelBuilder()
            rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * wp.pi)
            builder.add_shape_capsule(
                -1,
                xform=wp.transform((center_x, 0.0, 0.012), rotation),
                radius=0.012,
                half_height=0.025,
            )
            box = builder.add_body(xform=wp.transform((0.0, 0.0, 0.074 - 1.0e-5), wp.quat_identity()))
            box_shape = builder.add_shape_box(box, hx=0.05, hy=0.08, hz=0.05)
            model = builder.finalize(device)
            pipeline = newton.CollisionPipeline(model)
            test.assertEqual(pipeline.narrow_phase.split_gjk_mpr, split)
            contacts = pipeline.contacts()
            pipeline.collide(model.state(), contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            test.assertEqual(count, len(expected_x))
            shape0 = contacts.rigid_contact_shape0.numpy()[:count]
            point0 = contacts.rigid_contact_point0.numpy()[:count]
            point1 = contacts.rigid_contact_point1.numpy()[:count]
            box_points = np.where((shape0 == box_shape)[:, None], point0, point1)
            np.testing.assert_allclose(np.sort(box_points[:, 0]), expected_x, atol=1.0e-5, rtol=0.0)
            np.testing.assert_allclose(box_points[:, 1], 0.0, atol=1.0e-5, rtol=0.0)
            np.testing.assert_allclose(box_points[:, 2], -0.05, atol=1.0e-5, rtol=0.0)


add_function_test(
    TestBoxContactNormal,
    "test_capsule_box_edge_contact",
    test_capsule_box_edge_contact,
    devices=get_test_devices(),
)


def test_capsule_box_edge_contact_split(test, device):
    """Check capsule endpoint and segment contacts through the split CUDA kernels."""
    if not device.is_cuda:
        test.skipTest("Split collision kernels run only on CUDA")
    with patch("newton._src.sim.collide._SPLIT_GJK_MPR_FULL_PAIR_COUNT_THRESHOLD", 0):
        test_capsule_box_edge_contact(test, device, split=True)


add_function_test(
    TestBoxContactNormal,
    "test_capsule_box_edge_contact_split",
    test_capsule_box_edge_contact_split,
    devices=get_test_devices(),
)


def test_capsule_cylinder_tangent_contact(test, device, *, split=False):
    """Use a single contact where a capsule is tangent to a straight cylinder's side."""
    rotations = (wp.quat_identity(), wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), 0.7))
    for swap, rotation, angle in product((False, True), rotations, (wp.pi / 2, wp.pi / 6, wp.pi / 12)):
        with test.subTest(swap=swap, rotation=rotation, angle=angle):
            radius, capsule_radius, overlap = 0.025, 0.003, 0.0001
            frame = wp.transform(wp.vec3(0.585, -0.22, 0.046), rotation)
            capsule_pose = frame * wp.transform(
                wp.vec3(0.0, radius + capsule_radius - overlap, 0.0),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle),
            )
            builder = newton.ModelBuilder()
            body = builder.add_body(xform=capsule_pose)
            for is_cylinder in (False, True) if swap else (True, False):
                if is_cylinder:
                    cylinder_shape = builder.add_shape_cylinder(-1, xform=frame, radius=radius, half_height=0.00465)
                else:
                    builder.add_shape_capsule(body, radius=capsule_radius, half_height=0.0075)
            model = builder.finalize(device)
            pipeline = newton.CollisionPipeline(model)
            test.assertEqual(pipeline.narrow_phase.split_gjk_mpr, split)
            contacts = pipeline.contacts()
            pipeline.collide(model.state(), contacts)
            test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
            cylinder_first = int(contacts.rigid_contact_shape0.numpy()[0]) == cylinder_shape
            normal = contacts.rigid_contact_normal.numpy()[0]
            expected_normal = np.asarray(wp.quat_rotate(rotation, wp.vec3(0.0, 1.0, 0.0)))
            np.testing.assert_allclose(normal, expected_normal if cylinder_first else -expected_normal, atol=1.0e-4)
            point0, point1 = contacts.rigid_contact_point0.numpy()[0], contacts.rigid_contact_point1.numpy()[0]
            cylinder_point = point0 if cylinder_first else point1
            capsule_point = point1 if cylinder_first else point0
            # Oblique crossings amplify axial error in the deepest GJK witness.
            witness_tolerance = 1.0e-5 if angle == wp.pi / 2 else 1.0e-4
            np.testing.assert_allclose(
                cylinder_point, np.asarray(wp.transform_point(frame, wp.vec3(0.0, radius, 0.0))), atol=witness_tolerance
            )
            # The capsule's radius is stored as a separate friction-anchor offset.
            np.testing.assert_allclose(capsule_point[[0, 2]], [0.0, 0.0], atol=witness_tolerance)
            offset0, offset1 = contacts.rigid_contact_offset0.numpy()[0], contacts.rigid_contact_offset1.numpy()[0]
            capsule_offset = offset1 if cylinder_first else offset0
            np.testing.assert_allclose(capsule_offset, [0.0, -capsule_radius, 0.0], atol=1.0e-6)


add_function_test(
    TestBoxContactNormal,
    "test_capsule_cylinder_tangent_contact",
    test_capsule_cylinder_tangent_contact,
    devices=get_test_devices(),
)


def test_capsule_cylinder_tangent_contact_split(test, device):
    """Check tangent capsule-cylinder contacts through the split CUDA kernels."""
    if not device.is_cuda:
        test.skipTest("Split collision kernels run only on CUDA")
    with patch("newton._src.sim.collide._SPLIT_GJK_MPR_FULL_PAIR_COUNT_THRESHOLD", 0):
        test_capsule_cylinder_tangent_contact(test, device, split=True)


add_function_test(
    TestBoxContactNormal,
    "test_capsule_cylinder_tangent_contact_split",
    test_capsule_cylinder_tangent_contact_split,
    devices=get_test_devices(),
)


def test_resized_primitive_contact(test, device, *, split=False):
    """Keep shallow contact normals after resizing either primitive at runtime."""
    for resized, swap in product(("box", "cylinder"), (False, True)):
        with test.subTest(resized=resized, swap=swap):
            builder = newton.ModelBuilder()
            body = builder.add_body(xform=wp.transform(wp.vec3(0.63, 0.0, 0.049)))
            for is_box in (False, True) if swap else (True, False):
                if is_box:
                    box = builder.add_shape_box(
                        -1,
                        hx=0.6 / 1.05 if resized == "box" else 0.6,
                        hy=0.15,
                        hz=0.025,
                        cfg=builder.ShapeConfig(gap=0.0),
                    )
                else:
                    cylinder = builder.add_shape_cylinder(
                        body,
                        xform=wp.transform(wp.vec3(0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi / 2)),
                        radius=0.025 / 1.05 if resized == "cylinder" else 0.025,
                        half_height=0.05,
                        cfg=builder.ShapeConfig(gap=0.0),
                    )
            model = builder.finalize(device)
            pipeline = newton.CollisionPipeline(model)
            test.assertEqual(pipeline.narrow_phase.split_gjk_mpr, split)
            state, contacts = model.state(), pipeline.contacts()
            for factor in (1.0, 0.5, 1.0):
                scales = model.shape_scale.numpy()
                scales[box, 0] = 0.6 * factor if resized == "box" else 0.6
                scales[cylinder, 0] = 0.025 * factor if resized == "cylinder" else 0.025
                model.shape_scale.assign(scales)
                pipeline.collide(state, contacts)
                count = int(contacts.rigid_contact_count.numpy()[0])
                if factor == 0.5:
                    test.assertEqual(count, 0)
                    continue
                test.assertGreater(count, 0)
                shape0 = contacts.rigid_contact_shape0.numpy()[:count]
                normals = contacts.rigid_contact_normal.numpy()[:count]
                expected = np.where((shape0 == box)[:, None], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0])
                np.testing.assert_allclose(normals, expected, atol=1.0e-4)


add_function_test(
    TestBoxContactNormal, "test_resized_primitive_contact", test_resized_primitive_contact, devices=get_test_devices()
)


def test_resized_primitive_contact_split(test, device):
    """Check runtime primitive resizing through the split CUDA kernels."""
    if not device.is_cuda:
        test.skipTest("Split collision kernels run only on CUDA")
    with patch("newton._src.sim.collide._SPLIT_GJK_MPR_FULL_PAIR_COUNT_THRESHOLD", 0):
        test_resized_primitive_contact(test, device, split=True)


add_function_test(
    TestBoxContactNormal,
    "test_resized_primitive_contact_split",
    test_resized_primitive_contact_split,
    devices=get_test_devices(),
)


@wp.struct
class _ManifoldWriterData:
    contacts: wp.array[ContactData]
    count: wp.array[int]


@wp.func
def _write_manifold_contact(contact: ContactData, writer: _ManifoldWriterData, slot: int):
    index = wp.atomic_add(writer.count, 0, 1)
    writer.contacts[index] = contact


_build_cylinder_manifold = create_build_manifold(support_map, _write_manifold_contact, post_process_minkowski_only)


@wp.kernel(enable_backward=False)
def _cylinder_manifold_kernel(axial_error: float, swap: bool, cap: bool, writer: _ManifoldWriterData):
    shape_a = GenericShapeData()
    shape_a.shape_type = int(newton.GeoType.CYLINDER)
    shape_a.scale = wp.vec3(0.025, 0.05, 0.0)
    shape_b = shape_a
    shape_b.scale = wp.vec3(0.003, 0.0075, 0.0)
    if swap:
        shape_a.scale = wp.vec3(0.003, 0.0075, 0.0)
        shape_b.scale = wp.vec3(0.025, 0.05, 0.0)
    sign = -1.0 if swap else 1.0
    position = wp.vec3(sign * 0.0279, 0.0, 0.0)
    point_a = wp.vec3(sign * shape_a.scale[0], 0.0, 0.0)
    point_b = position - wp.vec3(sign * shape_b.scale[0], 0.0, 0.0)
    normal = wp.normalize(wp.vec3(sign, sign * 0.006, axial_error))
    if cap:
        position = wp.vec3(0.001, 0.0, 0.0574)
        point_a = wp.vec3(0.001, 0.0, shape_a.scale[1])
        point_b = position - wp.vec3(0.0, 0.0, shape_b.scale[1])
        normal = wp.vec3(0.0, 0.0, 1.0)
    _build_cylinder_manifold(
        shape_a,
        shape_b,
        wp.quat_identity(),
        wp.vec3(0.0),
        wp.quat_identity(),
        position,
        point_a,
        point_b,
        normal,
        SupportMapDataProvider(),
        writer,
        ContactData(),
    )


def test_parallel_cylinder_manifold_normal_error(test, device):
    """Preserve side endpoints under normal error while retaining cylinder cap faces."""
    with wp.ScopedDevice(device):
        writer = _ManifoldWriterData()
        writer.contacts = wp.empty(5, dtype=ContactData)
        writer.count = wp.zeros(1, dtype=int)
        for error, swap, cap in product((0.0, 5.0e-5, -2.0e-4, 1.0e-3), (False, True), (False, True)):
            with test.subTest(error=error, swap=swap, cap=cap):
                writer.count.zero_()
                wp.launch(_cylinder_manifold_kernel, 1, inputs=[error, swap, cap, writer])
                count = int(writer.count.numpy()[0])
                if cap:
                    test.assertGreaterEqual(count, 3)
                else:
                    test.assertEqual(count, 2)
                    points = writer.contacts.numpy()["contact_point_center"][:count]
                    np.testing.assert_allclose(np.sort(points[:, 2]), [-0.0075, 0.0075], atol=1.0e-5)


add_function_test(
    TestBoxContactNormal,
    "test_parallel_cylinder_manifold_normal_error",
    test_parallel_cylinder_manifold_normal_error,
    devices=get_test_devices(),
)


def test_parallel_cylinder_contact_endpoints(test, device, *, split=False):
    """Retain both endpoints of the contact line between parallel cylinders."""
    rotations = (wp.quat_identity(), wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), 0.7))
    for swap, rotation in product((False, True), rotations):
        with test.subTest(swap=swap, rotation=rotation):
            frame = wp.transform(wp.vec3(0.0), rotation)
            builder = newton.ModelBuilder()
            body = builder.add_body(xform=frame * wp.transform(wp.vec3(0.0279, 0.0, 0.0)))
            for is_large in (False, True) if swap else (True, False):
                if is_large:
                    large = builder.add_shape_cylinder(-1, xform=frame, radius=0.025, half_height=0.05)
                else:
                    builder.add_shape_cylinder(body, radius=0.003, half_height=0.0075)
            model = builder.finalize(device)
            pipeline = newton.CollisionPipeline(model)
            test.assertEqual(pipeline.narrow_phase.split_gjk_mpr, split)
            contacts = pipeline.contacts()
            pipeline.collide(model.state(), contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            test.assertEqual(count, 2)
            shape0 = contacts.rigid_contact_shape0.numpy()[:count]
            point0 = contacts.rigid_contact_point0.numpy()[:count]
            point1 = contacts.rigid_contact_point1.numpy()[:count]
            points = np.where((shape0 == large)[:, None], point1, point0)
            np.testing.assert_allclose(np.sort(points[:, 2]), [-0.0075, 0.0075], atol=1.0e-5)


add_function_test(
    TestBoxContactNormal,
    "test_parallel_cylinder_contact_endpoints",
    test_parallel_cylinder_contact_endpoints,
    devices=get_test_devices(),
)


def test_parallel_cylinder_contact_endpoints_split(test, device):
    """Retain parallel-cylinder endpoints through the split CUDA kernels."""
    if not device.is_cuda:
        test.skipTest("Split collision kernels run only on CUDA")
    with patch("newton._src.sim.collide._SPLIT_GJK_MPR_FULL_PAIR_COUNT_THRESHOLD", 0):
        test_parallel_cylinder_contact_endpoints(test, device, split=True)


add_function_test(
    TestBoxContactNormal,
    "test_parallel_cylinder_contact_endpoints_split",
    test_parallel_cylinder_contact_endpoints_split,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
