# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the closed-form cylinder-vs-prism contact against the generic GJK/MPR manifold."""

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
from newton._src.geometry.heightfield_primitive_contact import create_cylinder_prism_contacts
from newton._src.geometry.support_function import create_triangle_prism_penetration_refiner
from newton.tests.test_heightfield_primitive_query import FLAGS


@wp.struct
class _Deepest:
    count: wp.array[int]
    distance: wp.array[float]
    normal: wp.array[wp.vec3]


@wp.func
def _record_deepest(contact: ContactData, data: _Deepest, output_index: int):
    i = contact.sort_sub_key >> 3
    wp.atomic_add(data.count, i, 1)
    if contact.contact_distance < data.distance[i]:
        data.distance[i] = contact.contact_distance
        data.normal[i] = contact.contact_normal_a_to_b


_generic = create_compute_gjk_mpr_contacts(
    _record_deepest,
    support_func=support.support_map_heightfield_cylinder,
    use_precomputed_center=True,
    penetration_refiner=create_triangle_prism_penetration_refiner(support.support_map_heightfield_cylinder),
)
_analytic = create_cylinder_prism_contacts(_record_deepest)


@wp.kernel(enable_backward=False)
def _compare_pairs(
    prisms: wp.array[support.GenericShapeData],
    cylinders: wp.array[support.GenericShapeData],
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
            cylinders[tid],
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
        cylinders[tid],
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
    """Cells with a cylinder resting upright, lying, or tilted near the face."""
    prisms, cylinders, rot_a, rot_b, pos_a, pos_b, gaps, inside = [], [], [], [], [], [], [], []
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
        radius = rng.uniform(0.015, 0.05)
        half_height = rng.uniform(0.02, 0.15)
        gap = rng.choice([0.0, 0.002, 0.01])
        mode = rng.random()
        if mode < 0.4:  # upright on the face (small tilt)
            tilt_axis = rng.normal(size=3)
            tilt_axis -= (tilt_axis @ n) * n
            tilt = rng.uniform(0.0, 0.03) if rng.random() < 0.7 else rng.uniform(0.05, 0.6)
            q_align = _quat_from_to(np.array([0.0, 0.0, 1.0]), n)
            q_tilt = np.asarray(wp.quat_from_axis_angle(wp.normalize(wp.vec3(*tilt_axis)), float(tilt)))
            q = _quat_mul(q_tilt, q_align)
            lift = half_height
        elif mode < 0.75:  # lying (rolling) with small misalignment
            side = rng.normal(size=3)
            side -= (side @ n) * n
            side /= np.linalg.norm(side)
            q_align = _quat_from_to(np.array([0.0, 0.0, 1.0]), side)
            q_tilt = np.asarray(wp.quat_from_axis_angle(wp.vec3(*np.cross(side, n)), float(rng.uniform(-0.03, 0.03))))
            q = _quat_mul(q_tilt, q_align)
            lift = radius
        else:  # arbitrary
            axis = rng.normal(size=3)
            q = np.asarray(wp.quat_from_axis_angle(wp.normalize(wp.vec3(*axis)), float(rng.uniform(-np.pi, np.pi))))
            lift = radius + half_height
        u, v = rng.uniform(-0.2, 1.2, size=2)
        target = u * e1 + v * e2
        offset = rng.uniform(-0.01, gap + 0.004)
        c = target + n * (lift + offset)
        qa = wp.quat_identity()
        if rng.random() < 0.3:
            qa = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rng.uniform(-np.pi, np.pi)))
        pa = rng.uniform(-5.0, 5.0, size=3)
        pb = pa + np.asarray(wp.quat_rotate(qa, wp.vec3(*c)))
        qb = wp.quat(*[float(x) for x in q])
        prism = support.GenericShapeData()
        prism.shape_type = int(support.GeoTypeEx.TRIANGLE_PRISM)
        prism.scale = wp.vec3(*e1)
        prism.auxiliary = wp.vec3(*e2)
        prism.center = wp.vec3(0.0)
        cylinder = support.GenericShapeData()
        cylinder.shape_type = int(newton.GeoType.CYLINDER)
        cylinder.scale = wp.vec3(radius, half_height, 0.0)
        cylinder.auxiliary = wp.vec3(0.0)
        cylinder.center = wp.vec3(0.0)
        prisms.append(prism)
        cylinders.append(cylinder)
        rot_a.append(qa)
        rot_b.append(qa * qb)
        pos_a.append(pa)
        pos_b.append(pb)
        gaps.append(gap)
        inside.append(u >= 0.0 and v >= 0.0 and u + v <= 1.0)
    return prisms, cylinders, rot_a, rot_b, np.array(pos_a), np.array(pos_b), np.array(gaps), np.array(inside)


def _quat_from_to(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = np.cross(a, b)
    d = float(a @ b)
    if d < -1.0 + 1e-9:
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        return np.asarray(wp.quat_from_axis_angle(wp.normalize(wp.vec3(*axis)), float(np.pi)))
    q = np.array([c[0], c[1], c[2], 1.0 + d])
    return q / np.linalg.norm(q)


def _quat_mul(p, q):
    return np.asarray(wp.mul(wp.quat(*[float(x) for x in p]), wp.quat(*[float(x) for x in q])))


class TestHeightfieldCylinderContact(unittest.TestCase):
    def check_random_pairs(self, device, count=800, seed=5):
        rng = np.random.default_rng(seed)
        prisms, cylinders, rot_a, rot_b, pos_a, pos_b, gaps, inside = random_configurations(rng, count)

        def deepest(size):
            data = _Deepest()
            data.count = wp.zeros(size, dtype=int, device=device)
            data.distance = wp.full(size, 1.0e9, dtype=float, device=device)
            data.normal = wp.zeros(size, dtype=wp.vec3, device=device)
            return data

        with_generic = not wp.get_device(device).is_cuda
        generic, analytic = deepest(count if with_generic else 0), deepest(count)
        wp.launch(
            _compare_pairs,
            dim=count,
            inputs=[
                wp.array(prisms, dtype=support.GenericShapeData, device=device),
                wp.array(cylinders, dtype=support.GenericShapeData, device=device),
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
        self.assertLessEqual(int(a_count.max()), 4)
        self.assertTrue(np.isfinite(a_distance[a_count > 0]).all())
        if not with_generic:
            return
        g_count, g_distance = generic.count.numpy(), generic.distance.numpy()
        g_admitted = (g_count > 0) & (g_distance <= gaps)
        both = g_admitted & a_admitted
        self.assertGreater(int(both.sum()), count // 5)
        # The generic manifold's deepest point and the closed form's agree to a few
        # millimetres on the loaded pairs (MPR depth vs the exact surface point).
        error = np.abs(g_distance - a_distance)[both]
        self.assertLess(float(np.percentile(error, 90)), 3.0e-3)
        # The closed form never admits a pair the generic query rejects, and with the
        # cylinder center over the cell the two admit the same pairs away from the gap
        # threshold. Over a neighbouring cell the generic query still reports the plane
        # depth of its MPR witness (the sphere test documents the same artifact); the
        # closed form leaves those contacts to the cell that owns them.
        self.assertEqual(int((a_admitted & ~g_admitted).sum()), 0)
        disagree = np.flatnonzero(inside & (g_admitted != a_admitted))
        near = np.abs((np.where(a_count > 0, a_distance, g_distance) - gaps)[disagree]) < 5.0e-3
        self.assertLess(int((~near).sum()), max(2, int(0.02 * inside.sum())))

    def test_random_pairs_cpu(self):
        """Match the generic manifold's deepest contact on upright, lying and tilted cylinders (CPU)."""
        self.check_random_pairs("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_random_pairs_cuda(self):
        """The closed form stays finite and bounded on CUDA."""
        self.check_random_pairs("cuda:0")

    def test_kernel_selection(self):
        """Cylinders get the typed analytic query and the GJK split owner steps aside."""
        analytic = create_mesh_triangle_contacts_to_reducer_kernel(int(newton.GeoType.CYLINDER), analytic=True)
        generic = create_mesh_triangle_contacts_to_reducer_kernel(int(newton.GeoType.CYLINDER), analytic=False)
        self.assertIsNot(analytic, generic)
        self.assertTrue(analytic.key.endswith("cylinder_contacts_to_reducer_kernel_analytic"))
        builder = newton.ModelBuilder()
        terrain = newton.Heightfield(data=np.zeros((5, 5), np.float32), nrow=5, ncol=5, hx=0.4, hy=0.4)
        builder.add_shape_heightfield(heightfield=terrain)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_cylinder(body, radius=0.03, half_height=0.05)
        model = builder.finalize(device="cpu")
        for analytic_flag, split in (("1", False), ("0", True)):
            env = {
                **FLAGS,
                "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "1",
                "NEWTON_HEIGHTFIELD_CYLINDER_SPLIT": "1",
                "NEWTON_HEIGHTFIELD_ANALYTIC_PRIMITIVES": analytic_flag,
            }
            with patch.dict(os.environ, env):
                pipeline = newton.CollisionPipeline(model, reduce_contacts=True, rigid_contact_max=256)
            self.assertEqual(pipeline.narrow_phase._heightfield_cylinder_split, split)
            self.assertEqual(
                pipeline.narrow_phase._heightfield_primitive_kernels[1].key.endswith("_analytic"), not split
            )

    def check_resting(self, device):
        """Upright and lying cylinders rest at the same height through both queries on flat and sloped cells."""
        for pose_kind in ("upright", "lying"):
            for slope in (0.0, 0.12):
                with self.subTest(device=device, pose=pose_kind, slope=slope):
                    builder = newton.ModelBuilder()
                    cfg = builder.ShapeConfig(margin=0.001, gap=0.002, mu=0.8)
                    grid = np.linspace(-0.4, 0.4, 9, dtype=np.float32)
                    heights = np.ascontiguousarray(np.broadcast_to(slope * grid[None, :], (9, 9)), dtype=np.float32)
                    terrain = newton.Heightfield(data=heights, nrow=9, ncol=9, hx=0.4, hy=0.4)
                    builder.add_shape_heightfield(heightfield=terrain, cfg=cfg)
                    normal = np.array([-slope, 0.0, 1.0])
                    normal /= np.linalg.norm(normal)
                    if pose_kind == "upright":
                        q = _quat_from_to(np.array([0.0, 0.0, 1.0]), normal)
                        rest = 0.06
                    else:
                        # Axis down the slope so the cylinder rests instead of rolling.
                        q = _quat_from_to(np.array([0.0, 0.0, 1.0]), np.array([normal[2], 0.0, -normal[0]]))
                        rest = 0.03
                    body = builder.add_body(
                        xform=wp.transform(wp.vec3(0.03, 0.02, rest + 0.01), wp.quat(*[float(x) for x in q])), mass=1.0
                    )
                    builder.add_shape_cylinder(body, radius=0.03, half_height=0.06, cfg=cfg)
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
                        for _ in range(300):
                            state.clear_forces()
                            owner.collide(state, contacts)
                            solver.step(state, next_state, control, contacts, 0.002)
                            state, next_state = next_state, state
                        owner.narrow_phase.check_buffer_capacity()
                        pose, velocity = state.body_q.numpy(), state.body_qd.numpy()
                        self.assertTrue(np.isfinite(pose).all() and np.isfinite(velocity).all())
                        outcomes.append((pose.copy(), velocity.copy()))
                        # Height above the local surface: rest distance along the surface normal.
                        surface = slope * float(pose[0, 0])
                        height = (float(pose[0, 2]) - surface) * normal[2]
                        self.assertAlmostEqual(height, rest, delta=0.012)
                        self.assertLess(float(np.abs(velocity[0, :3]).max()), 0.3)
                    np.testing.assert_allclose(outcomes[0][0][:, :3], outcomes[1][0][:, :3], atol=0.01, rtol=0.0)

    def test_resting_cpu(self):
        """Rest upright and lying cylinders on flat and sloped terrain through both queries on CPU."""
        self.check_resting("cpu")

    def test_rough_terrain_support_cpu(self):
        """A long lying cylinder over small random cells rests on both queries without sinking or drifting."""
        rng = np.random.default_rng(3)
        builder = newton.ModelBuilder()
        cfg = builder.ShapeConfig(margin=0.001, gap=0.002, mu=0.8)
        # A shallow channel along x keeps the cylinder from rolling away; millimetre noise
        # still splits its contact line across cells.
        grid = np.linspace(-0.4, 0.4, 17, dtype=np.float32)
        heights = (0.1 * np.abs(grid)[:, None] + 0.0 * grid[None, :]).astype(np.float32)
        heights += rng.uniform(-0.002, 0.002, size=(17, 17)).astype(np.float32)
        terrain = newton.Heightfield(data=heights, nrow=17, ncol=17, hx=0.4, hy=0.4)
        builder.add_shape_heightfield(heightfield=terrain, cfg=cfg)
        q = _quat_from_to(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.01, 0.02, 0.05), wp.quat(*[float(x) for x in q])), mass=1.0
        )
        builder.add_shape_cylinder(body, radius=0.025, half_height=0.12, cfg=cfg)
        model = builder.finalize(device="cpu")
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
            for _ in range(300):
                state.clear_forces()
                owner.collide(state, contacts)
                solver.step(state, next_state, control, contacts, 0.002)
                state, next_state = next_state, state
            pose, velocity = state.body_q.numpy()[0], state.body_qd.numpy()[0]
            self.assertTrue(np.isfinite(pose).all() and np.isfinite(velocity).all())
            # Rolls into the channel bottom, rests one radius above the surface and stays
            # put (it still rocks slowly in the channel at this point).
            self.assertAlmostEqual(float(pose[2]), 0.025 + 0.1 * abs(float(pose[1])), delta=0.008)
            self.assertLess(abs(float(pose[1])), 0.03)
            self.assertLess(float(np.abs(velocity[:3]).max()), 0.1)
            self.assertLess(float(np.abs(velocity[3:]).max()), 1.5)
            outcomes.append(pose.copy())
        np.testing.assert_allclose(outcomes[0][:3], outcomes[1][:3], atol=0.002, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
