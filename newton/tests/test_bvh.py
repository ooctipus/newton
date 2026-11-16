# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for cooperative shape BVH local-bounds reduction."""

import gc
import unittest
import weakref
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.geometry.bvh import SHAPE_BOUNDS_BLOCK_DIM, compute_shape_local_bounds
from newton.tests.unittest_utils import add_function_test, get_test_devices

# Straddle the reduction block size: fewer points than lanes (idle lanes reduce only the
# sentinel), one lane short of full, exactly one iteration per lane, a single strided tail
# element, and several iterations per lane.
POINT_COUNTS = (
    3,
    SHAPE_BOUNDS_BLOCK_DIM - 1,
    SHAPE_BOUNDS_BLOCK_DIM,
    SHAPE_BOUNDS_BLOCK_DIM + 1,
    4 * SHAPE_BOUNDS_BLOCK_DIM + 1,
)


class TestShapeBvhBounds(unittest.TestCase):
    pass


def _make_points(rng: np.random.Generator, num_points: int, extent: float) -> np.ndarray:
    """Sample points in a cube, planting the extremes at the strided-tail and midpoint indices."""
    points = rng.uniform(-extent, extent, size=(num_points, 3)).astype(np.float32)
    # Index num_points - 1 is the final iteration of the last active lane, so a dropped tail
    # moves the bounds no matter how num_points relates to the block size.
    points[num_points - 1] = (2.0 * extent, 4.0 * extent, -6.0 * extent)
    points[num_points // 2] = (-4.0 * extent, 2.0 * extent, 1.0 * extent)
    return points


def _make_mesh(vertices: np.ndarray) -> newton.Mesh:
    """Wrap vertices in a Mesh; only the point set matters for bounds, so one triangle suffices."""
    return newton.Mesh(vertices, np.array([0, 1, 2], dtype=np.int32), compute_inertia=False)


def _gaussian_bounds(gaussian: newton.Gaussian, positions: np.ndarray, scales: np.ndarray) -> tuple[np.ndarray, ...]:
    """Reference AABB over Gaussian ellipsoids, given default identity rotations and unit opacities."""
    response_scale = np.sqrt(np.log(gaussian.min_response) / -0.5)
    return (
        np.min(positions - scales * response_scale, axis=0),
        np.max(positions + scales * response_scale, axis=0),
    )


def test_tiled_local_bounds(test: TestShapeBvhBounds, device: str):
    """Verify tiled local bounds match exact mesh and Gaussian AABBs across reduction block boundaries."""
    rng = np.random.default_rng(1234)

    for num_points in POINT_COUNTS:
        with test.subTest(num_points=num_points):
            vertices = _make_points(rng, num_points, 1.0)
            mesh = _make_mesh(vertices)

            positions = _make_points(rng, num_points, 2.0)
            scales = rng.uniform(0.05, 0.5, size=(num_points, 3)).astype(np.float32)
            gaussian = newton.Gaussian(positions=positions, scales=scales, min_response=0.1)

            builder = newton.ModelBuilder()
            mesh_shape = builder.add_shape_mesh(body=-1, mesh=mesh)
            convex_shape = builder.add_shape_convex_hull(body=-1, mesh=mesh)
            gaussian_shape = builder.add_shape_gaussian(body=-1, gaussian=gaussian)
            model = builder.finalize(device=device)

            bounds = model.bvh_shape_bounds.numpy()
            mesh_bounds = (vertices.min(axis=0), vertices.max(axis=0))
            np.testing.assert_allclose(bounds[mesh_shape], mesh_bounds, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(bounds[convex_shape], mesh_bounds, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(
                bounds[gaussian_shape], _gaussian_bounds(gaussian, positions, scales), rtol=1.0e-6, atol=1.0e-6
            )


def test_tiled_local_bounds_rebuild(test: TestShapeBvhBounds, device: str):
    """Verify a tiled bounds rebuild re-reads mutated mesh points rather than reusing the first result."""
    rng = np.random.default_rng(1234)
    num_points = POINT_COUNTS[-1]
    vertices = _make_points(rng, num_points, 1.0)
    mesh = _make_mesh(vertices)

    builder = newton.ModelBuilder()
    mesh_shape = builder.add_shape_mesh(body=-1, mesh=mesh)
    # All vertices are distinct, so finalize's convex dedup returns this same Mesh and both
    # shapes share one wp.Mesh; that is what lets the assign() below reach the convex shape too.
    convex_shape = builder.add_shape_convex_hull(body=-1, mesh=mesh)

    with mock.patch.object(wp, "launch_tiled", wraps=wp.launch_tiled) as launch_tiled:
        model = builder.finalize(device=device)

    local_bounds_launches = [
        call for call in launch_tiled.call_args_list if call.kwargs.get("kernel") is compute_shape_local_bounds
    ]
    test.assertEqual(len(local_bounds_launches), 1)
    test.assertEqual(local_bounds_launches[0].kwargs["block_dim"], SHAPE_BOUNDS_BLOCK_DIM)

    changed_vertices = vertices.copy()
    changed_vertices[num_points // 3] = (-8.0, 9.0, -10.0)
    mesh.mesh.points.assign(changed_vertices)
    model.bvh_build_shapes(model.state())

    rebuilt_bounds = model.bvh_shape_bounds.numpy()
    changed_bounds = (changed_vertices.min(axis=0), changed_vertices.max(axis=0))
    for shape, label in ((mesh_shape, "mesh"), (convex_shape, "convex")):
        with test.subTest(shape=label):
            np.testing.assert_allclose(rebuilt_bounds[shape], changed_bounds, rtol=0.0, atol=0.0)


def test_model_owns_gaussian_data(test: TestShapeBvhBounds, device: str):
    """A model owns the arrays and the BVH that its packed Gaussian structs address.

    :attr:`newton.Model.gaussians_data` packs each Gaussian's finalized record into a device array of structs,
    which holds only the addresses of the record's arrays and the id of its BVH, and the shape BVH's bounds read
    them on every build and refit. A Gaussian replaces its own record and BVH on its next ``finalize()`` (a second
    model from the same builder): unless the first model owns them, they are freed there and the allocator hands
    their memory to later objects. The owners are watched through weak references, so a freed address is never
    read.
    """
    gaussian = newton.Gaussian(positions=_make_points(np.random.default_rng(7), 16, 1.0))
    builder = newton.ModelBuilder()
    builder.add_shape_gaussian(builder.add_body(), gaussian=gaussian)
    model = builder.finalize(device=device)

    # The owners while the Gaussian still holds its first record: the live arrays at the packed addresses.
    packed = model.gaussians_data.numpy()[0]
    owners = {}
    for name in ("transforms", "scales", "opacities", "sh_coeffs"):
        address = int(packed[name]["data"])
        found = [
            array
            for array in gc.get_objects()
            if isinstance(array, wp.array) and getattr(array, "deleter", None) is not None and array.ptr == address
        ]
        test.assertEqual(len(found), 1)
        owners[name] = weakref.ref(found[0])
        del found
    bvh = weakref.ref(gaussian.bvh)
    test.assertEqual(bvh().id, int(packed["bvh_id"]))

    # The same Gaussian finalized again replaces its own record and BVH.
    other = builder.finalize(device=device)
    gc.collect()

    for name, owner in owners.items():
        with test.subTest(array=name):
            test.assertIsNotNone(owner(), f"nothing owns the {name} array the packed Gaussian struct addresses")
    test.assertIsNotNone(bvh(), "nothing owns the BVH the packed Gaussian struct names")
    test.assertNotEqual(other.gaussians_data.numpy()[0]["bvh_id"], packed["bvh_id"])


add_function_test(TestShapeBvhBounds, "test_tiled_local_bounds", test_tiled_local_bounds, devices=get_test_devices())
add_function_test(
    TestShapeBvhBounds, "test_tiled_local_bounds_rebuild", test_tiled_local_bounds_rebuild, devices=get_test_devices()
)
add_function_test(
    TestShapeBvhBounds, "test_model_owns_gaussian_data", test_model_owns_gaussian_data, devices=get_test_devices()
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
