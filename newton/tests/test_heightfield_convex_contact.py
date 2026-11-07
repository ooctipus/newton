# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the closed-form convex-hull-vs-prism contact against the generic GJK/MPR manifold."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry import support_function as support
from newton._src.geometry.collision_core import create_compute_gjk_mpr_contacts
from newton._src.geometry.contact_data import ContactData
from newton._src.geometry.contact_reduction_global import create_mesh_triangle_contacts_to_reducer_kernel
from newton._src.geometry.heightfield_primitive_contact import create_convex_prism_contacts
from newton._src.geometry.support_function import create_triangle_prism_penetration_refiner
from newton.tests.test_heightfield_cylinder_contact import _quat_from_to, _quat_mul
from newton.tests.test_heightfield_primitive_query import FLAGS


@wp.struct
class _Deepest:
    count: wp.array[int]
    distance: wp.array[float]


@wp.func
def _record_deepest(contact: ContactData, data: _Deepest, output_index: int):
    i = contact.sort_sub_key >> 3
    wp.atomic_add(data.count, i, 1)
    if contact.contact_distance < data.distance[i]:
        data.distance[i] = contact.contact_distance


_generic = create_compute_gjk_mpr_contacts(
    _record_deepest,
    support_func=support.support_map,
    use_precomputed_center=True,
    penetration_refiner=create_triangle_prism_penetration_refiner(support.support_map),
)
_analytic = create_convex_prism_contacts(_record_deepest)


@wp.kernel(enable_backward=False)
def _compare_pairs(
    prisms: wp.array[support.GenericShapeData],
    hulls: wp.array[support.GenericShapeData],
    rot_a: wp.array[wp.quat],
    rot_b: wp.array[wp.quat],
    pos_a: wp.array[wp.vec3],
    pos_b: wp.array[wp.vec3],
    gaps: wp.array[float],
    generic: _Deepest,
    analytic: _Deepest,
):
    tid = wp.tid()
    if generic.count.shape[0] > 0:
        _generic(
            prisms[tid],
            hulls[tid],
            rot_a[tid],
            rot_b[tid],
            pos_a[tid],
            pos_b[tid],
            gaps[tid],
            0,
            1,
            0.0,
            0.0,
            support.SupportMapDataProvider(),
            generic,
            tid,
            0,
        )
    _analytic(
        prisms[tid],
        hulls[tid],
        rot_a[tid],
        rot_b[tid],
        pos_a[tid],
        pos_b[tid],
        gaps[tid],
        0,
        1,
        0.0,
        0.0,
        analytic,
        tid,
        0,
    )


def foot_hull_vertices():
    """A chamfered box of 16 vertices, the shape of a typical foot collider."""
    hx, hy, hz, c = 0.09, 0.04, 0.025, 0.012
    vertices = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            vertices.append([sx * hx, sy * hy, hz])
            vertices.append([sx * hx, sy * hy, -hz + c])
            vertices.append([sx * (hx - c), sy * (hy - c), -hz])
            vertices.append([sx * (hx - c), sy * hy, hz - c])
    return np.array(vertices, dtype=np.float32)


def hull_mesh(device):
    vertices = foot_hull_vertices()
    return wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=device),
        indices=wp.array(hull_faces(vertices).reshape(-1), dtype=wp.int32, device=device),
    )


def packed_mesh_ptr(mesh_id):
    """Python twin of support_function.pack_mesh_ptr (22/22/20-bit float chunks)."""
    mesh_id = int(mesh_id)
    return wp.vec3(float(mesh_id & 0x3FFFFF), float((mesh_id >> 22) & 0x3FFFFF), float((mesh_id >> 44) & 0xFFFFF))


def _rotation(q):
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _xy_inside(p, e1, e2):
    det = e1[0] * e2[1] - e1[1] * e2[0]
    u = (p[0] * e2[1] - p[1] * e2[0]) / det
    v = (e1[0] * p[1] - e1[1] * p[0]) / det
    return u >= 0.0 and v >= 0.0 and u + v <= 1.0


def random_configurations(rng, count, mesh_id):
    """Random cell/hull pairs; ``inside`` marks hull centers over the cell, ``owner`` lowest vertices over it."""
    prisms, hulls, rot_a, rot_b, pos_a, pos_b, gaps, inside, owner = [], [], [], [], [], [], [], [], []
    extent = np.array([0.09, 0.04, 0.025])
    vertices = foot_hull_vertices().astype(np.float64)
    for _ in range(count):
        dx, dy = rng.uniform(0.1, 0.4, size=2)
        slope = 0.0 if rng.random() < 0.3 else rng.uniform(0.0, 0.25)
        h1, h2 = rng.uniform(-slope, slope, size=2)
        if rng.random() < 0.5:
            e1, e2 = np.array([dx, 0.0, h1]), np.array([dx, dy, h2])
        else:
            e1, e2 = np.array([dx, dy, h1]), np.array([0.0, dy, h2])
        n = np.cross(e1, e2)
        n /= np.linalg.norm(n)
        if n[2] < 0.0:
            n = -n
        gap = rng.choice([0.0, 0.002, 0.01])
        q_align = _quat_from_to(np.array([0.0, 0.0, 1.0]), n)
        mode = rng.random()
        if mode < 0.5:  # sole on the face with a small tilt
            tilt_axis = rng.normal(size=3)
            tilt_axis -= (tilt_axis @ n) * n
            tilt = rng.uniform(0.0, 0.02)
            lift = extent[2]
        elif mode < 0.8:  # toe or heel down
            tilt_axis = np.cross(n, rng.normal(size=3))
            tilt = rng.uniform(0.2, 0.7)
            lift = extent[2] * np.cos(tilt) + extent[0] * np.sin(tilt)
        else:
            tilt_axis = rng.normal(size=3)
            tilt = rng.uniform(-np.pi, np.pi)
            lift = float(np.linalg.norm(extent))
        q = _quat_mul(np.asarray(wp.quat_from_axis_angle(wp.normalize(wp.vec3(*tilt_axis)), float(tilt))), q_align)
        q = _quat_mul(np.asarray(wp.quat_from_axis_angle(wp.vec3(*n), float(rng.uniform(-np.pi, np.pi)))), q)
        u, v = rng.uniform(-0.2, 1.2, size=2)
        c = u * e1 + v * e2 + n * (lift + rng.uniform(-0.01, gap + 0.004))
        qa = wp.quat_identity()
        if rng.random() < 0.3:
            qa = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rng.uniform(-np.pi, np.pi)))
        pa = rng.uniform(-5.0, 5.0, size=3)
        pb = pa + np.asarray(wp.quat_rotate(qa, wp.vec3(*c)))
        prism = support.GenericShapeData()
        prism.shape_type = int(support.GeoTypeEx.TRIANGLE_PRISM)
        prism.scale = wp.vec3(*e1)
        prism.auxiliary = wp.vec3(*e2)
        prism.center = wp.vec3(0.0)
        hull = support.GenericShapeData()
        hull.shape_type = int(newton.GeoType.CONVEX_MESH)
        hull.scale = wp.vec3(1.0, 1.0, 1.0)
        hull.auxiliary = packed_mesh_ptr(mesh_id)
        hull.center = wp.vec3(0.0)
        prisms.append(prism)
        hulls.append(hull)
        rot_a.append(qa)
        rot_b.append(qa * wp.quat(*[float(x) for x in q]))
        pos_a.append(pa)
        pos_b.append(pb)
        gaps.append(gap)
        inside.append(u >= 0.0 and v >= 0.0 and u + v <= 1.0)
        local = vertices @ _rotation(q).T + c
        owner.append(_xy_inside(local[np.argmin(local @ n)], e1, e2))
    return (
        prisms,
        hulls,
        rot_a,
        rot_b,
        np.array(pos_a),
        np.array(pos_b),
        np.array(gaps),
        np.array(inside),
        np.array(owner),
    )


class TestHeightfieldConvexContact(unittest.TestCase):
    def check_random_pairs(self, device, count=800, seed=13):
        rng = np.random.default_rng(seed)
        mesh = hull_mesh(device)
        prisms, hulls, rot_a, rot_b, pos_a, pos_b, gaps, inside, owner = random_configurations(rng, count, mesh.id)

        def deepest(size):
            data = _Deepest()
            data.count = wp.zeros(size, dtype=int, device=device)
            data.distance = wp.full(size, 1.0e9, dtype=float, device=device)
            return data

        with_generic = not wp.get_device(device).is_cuda
        generic, analytic = deepest(count if with_generic else 0), deepest(count)
        wp.launch(
            _compare_pairs,
            dim=count,
            inputs=[
                wp.array(prisms, dtype=support.GenericShapeData, device=device),
                wp.array(hulls, dtype=support.GenericShapeData, device=device),
                wp.array(rot_a, dtype=wp.quat, device=device),
                wp.array(rot_b, dtype=wp.quat, device=device),
                wp.array(pos_a, dtype=wp.vec3, device=device),
                wp.array(pos_b, dtype=wp.vec3, device=device),
                wp.array(gaps, dtype=float, device=device),
                generic,
                analytic,
            ],
            device=device,
        )
        wp.synchronize_device(device)
        a_count, a_distance = analytic.count.numpy(), analytic.distance.numpy()
        a_admitted = (a_count > 0) & (a_distance <= gaps)
        self.assertGreater(int(a_admitted.sum()), count // 6)
        self.assertLessEqual(int(a_count.max()), 8)
        self.assertTrue(np.isfinite(a_distance[a_count > 0]).all())
        if not with_generic:
            return
        g_count, g_distance = generic.count.numpy(), generic.distance.numpy()
        g_admitted = (g_count > 0) & (g_distance <= gaps)
        both = g_admitted & a_admitted
        self.assertGreater(int(both.sum()), count // 5)
        # The cell under the lowest vertex reports its exact plane depth.
        error = np.abs(g_distance - a_distance)[both & owner]
        self.assertGreater(error.size, count // 10)
        self.assertLess(float(np.percentile(error, 90)), 2.0e-3)
        # Neighbouring cells see the touching polygon's chord at their boundary,
        # which is never deeper than the hull surface the generic query refines.
        shallower = (a_distance - g_distance)[both & ~owner]
        self.assertLess(int((shallower < -2.0e-3).sum()), max(2, int(0.02 * shallower.size)))
        self.assertEqual(int((a_admitted & ~g_admitted).sum()), 0)
        disagree = np.flatnonzero(inside & (g_admitted != a_admitted))
        near = np.abs((np.where(a_count > 0, a_distance, g_distance) - gaps)[disagree]) < 5.0e-3
        self.assertLess(int((~near).sum()), max(2, int(0.02 * inside.sum())))

    def test_random_pairs_cpu(self):
        """Match the generic manifold's deepest contact for sole, toe and arbitrary poses (CPU)."""
        self.check_random_pairs("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_random_pairs_cuda(self):
        """The closed form stays finite and bounded on CUDA."""
        self.check_random_pairs("cuda:0")

    def test_kernel_selection(self):
        """Convex hulls get a typed analytic query inside the fallback scan, or the generic query by default."""
        analytic = create_mesh_triangle_contacts_to_reducer_kernel(int(newton.GeoType.CONVEX_MESH), analytic=True)
        generic = create_mesh_triangle_contacts_to_reducer_kernel(int(newton.GeoType.CONVEX_MESH), analytic=False)
        self.assertIsNot(analytic, generic)
        self.assertTrue(analytic.key.endswith("convex_contacts_to_reducer_kernel_analytic"))
        # In the pipeline the row runs inside the generic fallback scan (no sixth kernel).
        fallback = create_mesh_triangle_contacts_to_reducer_kernel(exclude_primitives=True, analytic=True)
        self.assertTrue(fallback.key.endswith("fallback_kernel_analytic_convex"))
        # Opt-in: by default the fallback keeps the generic query for hulls.
        with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_ANALYTIC_CONVEX": "0"}):
            default = create_mesh_triangle_contacts_to_reducer_kernel(exclude_primitives=True)
        self.assertTrue(default.key.endswith("mesh_triangle_contacts_to_reducer_fallback_kernel"))
        with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_ANALYTIC_CONVEX": "1"}):
            enabled = create_mesh_triangle_contacts_to_reducer_kernel(exclude_primitives=True)
        self.assertIs(enabled, fallback)

    def check_resting(self, device):
        """A foot hull settles sole-down at the same height and place through both queries, also over small cells."""
        for start, cells in (("sole", 9), ("toe", 9), ("sole", 33)):
            with self.subTest(device=device, start=start, cells=cells):
                rng = np.random.default_rng(2)
                builder = newton.ModelBuilder()
                cfg = builder.ShapeConfig(margin=0.001, gap=0.002, mu=0.8)
                heights = rng.uniform(-0.001, 0.001, size=(cells, cells)).astype(np.float32)
                terrain = newton.Heightfield(data=heights, nrow=cells, ncol=cells, hx=0.4, hy=0.4)
                builder.add_shape_heightfield(heightfield=terrain, cfg=cfg)
                q = wp.quat_identity()
                z = 0.025 + 0.01
                if start == "toe":
                    q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.3)
                    z = 0.06
                body = builder.add_body(xform=wp.transform(wp.vec3(0.03, 0.02, z), q), mass=1.0)
                vertices = foot_hull_vertices()
                builder.add_shape_convex_hull(
                    body, mesh=newton.Mesh(vertices=vertices, indices=hull_faces(vertices)), cfg=cfg
                )
                model = builder.finalize(device=device)
                outcomes = []
                for analytic in ("0", "1"):
                    env = {
                        **FLAGS,
                        "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "1",
                        "NEWTON_HEIGHTFIELD_ANALYTIC_PRIMITIVES": analytic,
                        "NEWTON_HEIGHTFIELD_ANALYTIC_CONVEX": analytic,
                    }
                    with patch.dict(os.environ, env):
                        owner = newton.CollisionPipeline(
                            model, reduce_contacts=True, rigid_contact_max=4096, max_triangle_pairs=4096
                        )
                    solver = newton.solvers.SolverXPBD(model, iterations=10)
                    state, next_state = model.state(), model.state()
                    contacts, control = owner.contacts(), model.control()
                    history = []
                    for step in range(400):
                        state.clear_forces()
                        owner.collide(state, contacts)
                        solver.step(state, next_state, control, contacts, 0.002)
                        state, next_state = next_state, state
                        if step >= 300:
                            history.append(state.body_q.numpy()[0].copy())
                    owner.narrow_phase.check_buffer_capacity()
                    history = np.array(history)
                    pose, velocity = history[-1], state.body_qd.numpy()[0]
                    self.assertTrue(np.isfinite(history).all() and np.isfinite(velocity).all())
                    self.assertAlmostEqual(float(pose[2]), 0.025, delta=0.006)
                    self.assertGreater(abs(float(pose[6])), 0.99)
                    # Settled: the pose moves by less than a few millimetres over the last 100 steps.
                    self.assertLess(float(np.ptp(history[:, :3], axis=0).max()), 0.003)
                    outcomes.append(pose.copy())
                # Same resting height; the slide after a toe landing is a lively
                # event under either query, so the place is compared loosely.
                self.assertAlmostEqual(float(outcomes[0][2]), float(outcomes[1][2]), delta=0.003)
                np.testing.assert_allclose(outcomes[0][:2], outcomes[1][:2], atol=0.02, rtol=0.0)

    def test_resting_cpu(self):
        """Rest a foot hull on fine and coarse terrain cells through both queries on CPU."""
        self.check_resting("cpu")


def hull_faces(vertices):
    """Triangle faces of the hull (scipy when available; a brute-force fallback otherwise)."""
    try:
        from scipy.spatial import ConvexHull

        return ConvexHull(vertices).simplices.astype(np.int32)
    except Exception:  # pragma: no cover
        faces = []
        count = len(vertices)
        for i in range(count):
            for j in range(i + 1, count):
                for k in range(j + 1, count):
                    normal = np.cross(vertices[j] - vertices[i], vertices[k] - vertices[i])
                    if np.linalg.norm(normal) < 1e-9:
                        continue
                    side = (vertices - vertices[i]) @ normal
                    if np.all(side <= 1e-9) or np.all(side >= -1e-9):
                        faces.append((i, j, k))
        return np.array(faces, dtype=np.int32)


if __name__ == "__main__":
    unittest.main()
