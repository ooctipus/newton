# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check one force law across ordinary and guarded publication."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import kernels, kinetic_public_force


class TestSharedContactForce(unittest.TestCase):
    def test_direct_shared_factory(self):
        """Construct guarded owners through explicit shared force functions."""
        from newton._src.solvers.feather_pgs import contact_force  # noqa: PLC0415

        self.assertTrue(hasattr(kernels, "_contact_linear_force"))
        self.assertTrue(hasattr(kernels, "_pack_contact_linear_force"))
        for name in kinetic_public_force.NAMES:
            self.assertIs(kinetic_public_force.get_kernel(name), contact_force.get_kernel(name))

    def _check(self, device, *, graph=False):
        """Exercise every impulse route, tangent admission, and frame transition."""
        from newton._src.solvers.feather_pgs import contact_force  # noqa: PLC0415

        count, capacity = 9, 16
        paths = np.array([0, 1, 2, -1, 0, 1, 2, 0, 0] + [-1] * 7, np.int32)
        slots = np.array([0, 3, 6, -1, 9, 0, 3, 6, 9] + [-1] * 7, np.int32)
        worlds = np.arange(capacity, dtype=np.int32) % 2
        normals = np.tile(np.eye(3, dtype=np.float32), (6, 1))[:capacity]
        impulses = [(np.arange(24).reshape(2, 12) + 1).astype(np.float32) * scale for scale in (0.1, 0.2, 0.3)]
        types = [np.tile([0, 2, 2] * 4, (2, 1)).astype(np.int32) for _ in range(3)]
        parents = [np.tile([-1, 0, 0, -1, 3, 3, -1, 6, 6, -1, 9, 9], (2, 1)).astype(np.int32) for _ in range(3)]
        parents[0][1, 7] = 0
        types[2][0, 5] = 0
        counts = [np.array([12, 11], np.int32) for _ in range(3)]

        def array(value, dtype=float):
            return wp.array(value, dtype=dtype, device=device)

        linear = wp.full(capacity, wp.vec3(77.0), dtype=wp.vec3, device=device)
        spatial = wp.full(capacity, wp.spatial_vector(88.0), dtype=wp.spatial_vector, device=device)
        flag = wp.zeros(1, dtype=int, device=device)
        args = [
            array([count], int),
            array(normals, wp.vec3),
            array(worlds, int),
            array(slots, int),
            array(paths, int),
            *[array(x) for x in impulses],
            *[array(x, int) for x in counts],
            *[array(x, int) for pair in zip(types, parents, strict=True) for x in pair],
            1,
            100.0,
            linear,
        ]
        first, second = (contact_force.get_kernel(name) for name in kinetic_public_force.NAMES)

        def launch():
            wp.launch(first, dim=capacity, inputs=[*args, flag], device=device)
            wp.launch(second, dim=capacity, inputs=[args[0], linear, spatial, flag], device=device)

        expected = np.full((capacity, 3), 77.0, np.float32)
        for c in range(count):
            expected[c] = 0
            path, slot, world = paths[c], slots[c], worlds[c]
            if path < 0 or slot < 0:
                continue
            normal = -normals[c]
            force = impulses[path][world, slot] * normal
            if (
                slot + 2 < counts[path][world]
                and np.all(types[path][world, slot + 1 : slot + 3] == 2)
                and np.all(parents[path][world, slot + 1 : slot + 3] == slot)
            ):
                t0 = np.cross(normal, [1.0, 0.0, 0.0])
                if np.dot(t0, t0) < 1e-12:
                    t0 = np.cross(normal, [0.0, 1.0, 0.0])
                t0 /= np.linalg.norm(t0)
                t1 = np.cross(normal, t0)
                force += impulses[path][world, slot + 1] * t0 + impulses[path][world, slot + 2] * t1
            expected[c] = 100 * force
        launch()
        np.testing.assert_allclose(linear.numpy(), expected, rtol=2e-6, atol=1e-5)
        np.testing.assert_array_equal(spatial.numpy()[:count, 3:], 0)
        np.testing.assert_allclose(spatial.numpy()[:count, :3], expected[:count], rtol=2e-6, atol=1e-5)
        np.testing.assert_array_equal(spatial.numpy()[count:], 88)
        ordinary = wp.full(capacity, wp.vec3(77.0), dtype=wp.vec3, device=device)
        wp.launch(
            kernels.compute_contact_linear_force_from_impulses,
            dim=capacity,
            inputs=[*args[:-1], ordinary],
            device=device,
        )
        np.testing.assert_array_equal(ordinary.numpy(), linear.numpy())

        if graph:
            with wp.ScopedCapture(device=device) as capture:
                launch()

            def launch_graph():
                wp.capture_launch(capture.graph)
        else:
            launch_graph = launch
        flag.fill_(1)
        linear.fill_(wp.vec3(77.0))
        spatial.fill_(wp.spatial_vector(88.0))
        launch_graph()
        np.testing.assert_array_equal(linear.numpy(), 77)
        np.testing.assert_array_equal(spatial.numpy(), 88)
        flag.zero_()
        launch_graph()
        np.testing.assert_allclose(linear.numpy(), expected, rtol=2e-6, atol=1e-5)
        args[-3] = 0
        launch()
        normal_only = np.zeros((count, 3), np.float32)
        for c in range(count):
            if paths[c] >= 0 and slots[c] >= 0:
                normal_only[c] = -100 * impulses[paths[c]][worlds[c], slots[c]] * normals[c]
        np.testing.assert_allclose(linear.numpy()[:count], normal_only, rtol=2e-6, atol=1e-5)
        args[-2] = 0.0
        launch()
        np.testing.assert_array_equal(linear.numpy()[:count], 0)
        args[0].zero_()
        linear.fill_(wp.vec3(77.0))
        spatial.fill_(wp.spatial_vector(88.0))
        launch()
        np.testing.assert_array_equal(linear.numpy(), 77)
        np.testing.assert_array_equal(spatial.numpy(), 88)

    def test_force_routes_cpu(self):
        """Preserve force conversion and completed-frame publication on CPU."""
        self._check("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph publication requires CUDA")
    def test_force_routes_graph_replay(self):
        """Preserve invalid no-write and valid publication across captured replay."""
        self._check("cuda:0", graph=True)


if __name__ == "__main__":
    unittest.main()
