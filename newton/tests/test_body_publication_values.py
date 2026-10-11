# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check shared body publication against independent spatial mechanics."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import kernels


class TestBodyPublicationValues(unittest.TestCase):
    def check_mechanics(self, device):
        rng = np.random.default_rng(81)
        count = 4
        q = rng.normal(size=(count, 4))
        q /= np.linalg.norm(q, axis=1)[:, None]
        rotation = np.empty((count, 3, 3))
        for i, (x, y, z, w) in enumerate(q):
            rotation[i] = [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        position = rng.uniform(-3, 3, (count, 3))
        offset = rng.uniform(-0.2, 0.2, (count, 3))
        center = position + np.einsum("bij,bj->bi", rotation, offset)
        origin = rng.uniform(-2, 2, (3, 3))
        mass = rng.uniform(0.4, 4, count)
        tensor = rng.normal(size=(count, 3, 3))
        tensor = tensor @ tensor.transpose(0, 2, 1) + 0.3 * np.eye(3)
        velocity = rng.normal(size=(count, 6))
        acceleration = rng.normal(size=(count, 6))
        gravity = np.array([0.2, -0.4, -9.81])
        art = np.array([-1, 0, 1, 2])
        free = np.array([0, 1, 0])
        expected_i = np.empty((count, 6, 6))
        expected_terms = np.empty((count, 12))
        expected_f = np.empty((count, 6))
        expected_v = np.empty((count, 6))
        for body in range(count):
            c = center[body] - (origin[art[body]] if art[body] >= 0 else 0)
            v, w = velocity[body, :3], velocity[body, 3:]
            expected_v[body] = np.r_[v + np.cross(w, c), w]
            cross = np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]])
            inertia = rotation[body] @ tensor[body] @ rotation[body].T
            inertia += mass[body] * ((c @ c) * np.eye(3) - np.outer(c, c))
            spatial = np.block([[mass[body] * np.eye(3), -mass[body] * cross], [mass[body] * cross, inertia]])
            momentum = spatial @ velocity[body]
            bias = np.r_[np.cross(w, momentum[:3]), np.cross(v, momentum[:3]) + np.cross(w, momentum[3:])]
            expected_i[body] = spatial
            expected_terms[body] = np.r_[c, inertia.ravel()]
            expected_f[body] = (
                spatial @ acceleration[body] + bias - np.r_[mass[body] * gravity, np.cross(c, mass[body] * gravity)]
            )
        values = {
            "body_to_articulation": wp.array(art, dtype=int, device=device),
            "body_q": wp.array(np.c_[position, q], dtype=wp.transform, device=device),
            "body_q_com": wp.array(np.c_[center, q], dtype=wp.transform, device=device),
            "body_com": wp.array(offset, dtype=wp.vec3, device=device),
            "body_mass": wp.array(mass, dtype=float, device=device),
            "body_inertia": wp.array(tensor, dtype=wp.mat33, device=device),
            "is_free_rigid": wp.array(free, dtype=int, device=device),
            "articulation_origin": wp.array(origin, dtype=wp.vec3, device=device),
            "gravity": wp.array([gravity], dtype=wp.vec3, device=device),
            "body_v_s": wp.array(velocity, dtype=wp.spatial_vector, device=device),
            "body_a_s": wp.array(acceleration, dtype=wp.spatial_vector, device=device),
        }
        for full, compact in ((0, 0), (1, 0), (0, 1)):
            with self.subTest(device=device, full=full, compact=compact):
                values.update(
                    materialize_all_body_inertia=full,
                    materialize_body_inertia_terms=compact,
                    body_I_s=wp.full(count, wp.spatial_matrix(17.0), dtype=wp.spatial_matrix, device=device),
                    body_inertia_terms=wp.full((count, 12), 17.0, dtype=float, device=device),
                    body_f_s=wp.full(count, wp.spatial_vector(17.0), dtype=wp.spatial_vector, device=device),
                    body_qd=wp.zeros(count, dtype=wp.spatial_vector, device=device),
                )
                wp.launch(
                    kernels.finalize_body_dynamics,
                    count,
                    [values[a.label] for a in kernels.finalize_body_dynamics.adj.args],
                    device=device,
                )
                np.testing.assert_allclose(values["body_qd"].numpy(), expected_v, atol=2e-5, rtol=2e-6)
                np.testing.assert_allclose(values["body_f_s"].numpy()[1:], expected_f[1:], atol=5e-5, rtol=3e-6)
                np.testing.assert_array_equal(values["body_f_s"].numpy()[0], 17.0)
                for body in range(count):
                    materialize = art[body] >= 0 and (full or free[art[body]])
                    np.testing.assert_allclose(
                        values["body_I_s"].numpy()[body],
                        expected_i[body] if materialize else 17.0,
                        atol=2e-5,
                        rtol=3e-6,
                    )
                    np.testing.assert_allclose(
                        values["body_inertia_terms"].numpy()[body],
                        expected_terms[body] if compact and art[body] >= 0 else 17.0,
                        atol=2e-5,
                        rtol=3e-6,
                    )

    def test_mechanics_cpu(self):
        self.check_mechanics("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_mechanics_cuda(self):
        self.check_mechanics("cuda:0")


if __name__ == "__main__":
    unittest.main()
