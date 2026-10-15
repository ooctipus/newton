# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the closed-form box-vs-prism contact against the generic GJK/MPR manifold."""

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
from newton._src.geometry.heightfield_primitive_contact import create_box_prism_contacts
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
    support_func=support.support_map_heightfield_box,
    use_precomputed_center=True,
    penetration_refiner=create_triangle_prism_penetration_refiner(support.support_map_heightfield_box),
)
_analytic = create_box_prism_contacts(_record_deepest)


@wp.kernel(enable_backward=False)
def _compare_pairs(
    prisms: wp.array[support.GenericShapeData],
    boxes: wp.array[support.GenericShapeData],
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
            boxes[tid],
            rot_a[tid],
            rot_b[tid],
            pos_a[tid],
            pos_b[tid],
            gaps[tid],
            0,
            1,
            0.0,
            0.0,
            generic,
            tid,
            0,
        )
    _analytic(
        prisms[tid],
        boxes[tid],
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


def random_configurations(rng, count):
    """Cells with a box resting flat, on an edge or on a corner near the face."""
    prisms, boxes, rot_a, rot_b, pos_a, pos_b, gaps, inside = [], [], [], [], [], [], [], []
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
        half = rng.uniform(0.015, 0.08, size=3)
        gap = rng.choice([0.0, 0.002, 0.01])
        q_align = _quat_from_to(np.array([0.0, 0.0, 1.0]), n)
        mode = rng.random()
        if mode < 0.4:  # flat with a small tilt
            tilt_axis = rng.normal(size=3)
            tilt_axis -= (tilt_axis @ n) * n
            tilt = rng.uniform(0.0, 0.02)
            lift = half[2]
        elif mode < 0.7:  # on an edge
            tilt_axis = n.copy()
            tilt_axis = np.cross(n, rng.normal(size=3))
            tilt = rng.uniform(0.2, 0.8)
            lift = half[2] * np.cos(tilt) + max(half[0], half[1]) * np.sin(tilt)
        else:  # arbitrary
            tilt_axis = rng.normal(size=3)
            tilt = rng.uniform(-np.pi, np.pi)
            lift = float(np.linalg.norm(half))
        q_tilt = np.asarray(wp.quat_from_axis_angle(wp.normalize(wp.vec3(*tilt_axis)), float(tilt)))
        q = _quat_mul(q_tilt, q_align)
        yaw = np.asarray(wp.quat_from_axis_angle(wp.vec3(*n), float(rng.uniform(-np.pi, np.pi))))
        q = _quat_mul(yaw, q)
        u, v = rng.uniform(-0.2, 1.2, size=2)
        target = u * e1 + v * e2
        offset = rng.uniform(-0.01, gap + 0.004)
        c = target + n * (lift + offset)
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
        box = support.GenericShapeData()
        box.shape_type = int(newton.GeoType.BOX)
        box.scale = wp.vec3(*half)
        box.auxiliary = wp.vec3(0.0)
        box.center = wp.vec3(0.0)
        prisms.append(prism)
        boxes.append(box)
        rot_a.append(qa)
        rot_b.append(qa * wp.quat(*[float(x) for x in q]))
        pos_a.append(pa)
        pos_b.append(pb)
        gaps.append(gap)
        inside.append(u >= 0.0 and v >= 0.0 and u + v <= 1.0)
    return prisms, boxes, rot_a, rot_b, np.array(pos_a), np.array(pos_b), np.array(gaps), np.array(inside)


class TestHeightfieldBoxContact(unittest.TestCase):
    def check_random_pairs(self, device, count=800, seed=9):
        rng = np.random.default_rng(seed)
        prisms, boxes, rot_a, rot_b, pos_a, pos_b, gaps, inside = random_configurations(rng, count)

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
                wp.array(boxes, dtype=support.GenericShapeData, device=device),
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
        self.assertGreater(int(a_admitted.sum()), count // 4)
        self.assertLessEqual(int(a_count.max()), 7)
        self.assertTrue(np.isfinite(a_distance[a_count > 0]).all())
        if not with_generic:
            return
        g_count, g_distance = generic.count.numpy(), generic.distance.numpy()
        g_admitted = (g_count > 0) & (g_distance <= gaps)
        both = g_admitted & a_admitted
        self.assertGreater(int(both.sum()), count // 5)
        error = np.abs(g_distance - a_distance)[both]
        self.assertLess(float(np.percentile(error, 90)), 3.0e-3)
        self.assertEqual(int((a_admitted & ~g_admitted).sum()), 0)
        disagree = np.flatnonzero(inside & (g_admitted != a_admitted))
        near = np.abs((np.where(a_count > 0, a_distance, g_distance) - gaps)[disagree]) < 5.0e-3
        self.assertLess(int((~near).sum()), max(2, int(0.02 * inside.sum())))

    def test_random_pairs_cpu(self):
        """Match the generic manifold's deepest contact on flat, edge and corner poses (CPU)."""
        self.check_random_pairs("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_random_pairs_cuda(self):
        """The closed form stays finite and bounded on CUDA."""
        self.check_random_pairs("cuda:0")

    def test_kernel_selection(self):
        """Boxes get a typed analytic query behind the shared switch."""
        analytic = create_mesh_triangle_contacts_to_reducer_kernel(int(newton.GeoType.BOX), analytic=True)
        generic = create_mesh_triangle_contacts_to_reducer_kernel(int(newton.GeoType.BOX), analytic=False)
        self.assertIsNot(analytic, generic)
        self.assertTrue(analytic.key.endswith("box_contacts_to_reducer_kernel_analytic"))

    def check_resting(self, device):
        """Flat and edge-started boxes settle at the same place through both queries, also over small cells."""
        for start, cells in (("flat", 9), ("edge", 9), ("flat", 33)):
            with self.subTest(device=device, start=start, cells=cells):
                rng = np.random.default_rng(1)
                builder = newton.ModelBuilder()
                cfg = builder.ShapeConfig(margin=0.001, gap=0.002, mu=0.8)
                heights = rng.uniform(-0.001, 0.001, size=(cells, cells)).astype(np.float32)
                terrain = newton.Heightfield(data=heights, nrow=cells, ncol=cells, hx=0.4, hy=0.4)
                builder.add_shape_heightfield(heightfield=terrain, cfg=cfg)
                q = wp.quat_identity()
                z = 0.04 + 0.01
                if start == "edge":
                    q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.35)
                    z = 0.09
                body = builder.add_body(xform=wp.transform(wp.vec3(0.03, 0.02, z), q), mass=1.0)
                builder.add_shape_box(body, hx=0.06, hy=0.05, hz=0.04, cfg=cfg)
                model = builder.finalize(device=device)
                outcomes = []
                for analytic in ("0", "1"):
                    env = {
                        **FLAGS,
                        "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "1",
                        "NEWTON_HEIGHTFIELD_ANALYTIC_PRIMITIVES": analytic,
                    }
                    with patch.dict(os.environ, env):
                        owner = newton.CollisionPipeline(
                            model, reduce_contacts=True, rigid_contact_max=4096, max_triangle_pairs=4096
                        )
                    solver = newton.solvers.SolverXPBD(model, iterations=10)
                    state, next_state = model.state(), model.state()
                    contacts, control = owner.contacts(), model.control()
                    for _ in range(400):
                        state.clear_forces()
                        owner.collide(state, contacts)
                        solver.step(state, next_state, control, contacts, 0.002)
                        state, next_state = next_state, state
                    owner.narrow_phase.check_buffer_capacity()
                    pose, velocity = state.body_q.numpy()[0], state.body_qd.numpy()[0]
                    self.assertTrue(np.isfinite(pose).all() and np.isfinite(velocity).all())
                    # Settles flat: one half-height above the surface, upright, at rest.
                    self.assertAlmostEqual(float(pose[2]), 0.04, delta=0.006)
                    self.assertGreater(abs(float(pose[6])), 0.995)
                    self.assertLess(float(np.abs(velocity).max()), 0.3)
                    outcomes.append(pose.copy())
                np.testing.assert_allclose(outcomes[0][:3], outcomes[1][:3], atol=0.01, rtol=0.0)

    def test_resting_cpu(self):
        """Rest boxes on fine and coarse terrain cells through both queries on CPU."""
        self.check_resting("cpu")


if __name__ == "__main__":
    unittest.main()
