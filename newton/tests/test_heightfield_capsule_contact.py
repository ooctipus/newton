# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the closed-form capsule-vs-prism contact against the generic GJK/MPR manifold."""

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
from newton._src.geometry.heightfield_primitive_contact import create_capsule_prism_contacts
from newton._src.geometry.support_function import create_triangle_prism_penetration_refiner
from newton.tests.test_heightfield_primitive_query import FLAGS
from newton.tests.test_heightfield_sphere_contact import _angle, _quat_matrix_np, closest_point_on_prism_np


@wp.struct
class _Deepest:
    count: wp.array[int]
    distance: wp.array[float]
    normal: wp.array[wp.vec3]


@wp.func
def _record_deepest(contact: ContactData, data: _Deepest, output_index: int):
    i = contact.sort_sub_key >> 3
    wp.atomic_add(data.count, i, 1)
    # Record the deepest contact of the pair (atomic min on a float through a CAS loop is not
    # needed: pairs are owned by one thread, contacts arrive in sequence).
    if contact.contact_distance < data.distance[i]:
        data.distance[i] = contact.contact_distance
        data.normal[i] = contact.contact_normal_a_to_b


_generic = create_compute_gjk_mpr_contacts(
    _record_deepest,
    support_func=support.support_map_heightfield_capsule,
    use_precomputed_center=True,
    penetration_refiner=create_triangle_prism_penetration_refiner(support.support_map_heightfield_capsule),
)
_analytic = create_capsule_prism_contacts(_record_deepest)


@wp.kernel(enable_backward=False)
def _compare_pairs(
    prisms: wp.array[support.GenericShapeData],
    capsules: wp.array[support.GenericShapeData],
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
            capsules[tid],
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
        capsules[tid],
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
    prisms, capsules, rot_a, rot_b, pos_a, pos_b, gaps = [], [], [], [], [], [], []
    for _ in range(count):
        dx, dy = rng.uniform(0.1, 0.4, size=2)
        slope = 0.0 if rng.random() < 0.2 else rng.uniform(0.0, 0.3)
        h1, h2 = rng.uniform(-slope, slope, size=2)
        if rng.random() < 0.5:
            e1, e2 = np.array([dx, 0.0, h1]), np.array([dx, dy, h2])
        else:
            e1, e2 = np.array([dx, dy, h1]), np.array([0.0, dy, h2])
        n = np.cross(e1, e2)
        n /= np.linalg.norm(n)
        if n[2] < 0.0:
            n = -n
        radius = rng.uniform(0.01, 0.05)
        half_height = rng.uniform(0.02, 0.2)
        gap = rng.choice([0.0, 0.002, 0.01])
        u, v = rng.uniform(-0.3, 1.3, size=2)
        target = u * e1 + v * e2
        offset = rng.uniform(-radius * 0.5, gap + 0.03)
        c = target + n * (radius + offset) + rng.normal(0.0, 0.02, size=3)
        axis = rng.normal(size=3)
        if rng.random() < 0.4:
            axis[2] *= 0.05  # mostly lying capsules
        qb = wp.quat_from_axis_angle(wp.normalize(wp.vec3(*axis)), float(rng.uniform(-np.pi, np.pi)))
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
        capsule = support.GenericShapeData()
        capsule.shape_type = int(newton.GeoType.CAPSULE)
        capsule.scale = wp.vec3(radius, half_height, 0.0)
        capsule.auxiliary = wp.vec3(0.0)
        capsule.center = wp.vec3(0.0)
        prisms.append(prism)
        capsules.append(capsule)
        rot_a.append(qa)
        rot_b.append(qb)
        pos_a.append(pa)
        pos_b.append(pb)
        gaps.append(gap)
    return prisms, capsules, rot_a, rot_b, np.array(pos_a), np.array(pos_b), np.array(gaps)


def _segment_prism_reference(p0, p1, e1, e2, samples=2001):
    """Float64 reference: deepest/nearest point of the capsule axis against the prism."""
    best = None
    for t in np.linspace(0.0, 1.0, samples):
        p = p0 + t * (p1 - p0)
        n = np.cross(e1, e2)
        n /= np.linalg.norm(n)
        if n[2] < 0.0:
            n = -n
        s = p @ n
        det = e1[0] * e2[1] - e1[1] * e2[0]
        u = (p[0] * e2[1] - p[1] * e2[0]) / det
        v = (e1[0] * p[1] - e1[1] * p[0]) / det
        inside = s < 0.0 and -s <= n[2] and u >= 0 and v >= 0 and u + v <= 1.0
        if inside:
            d = s
        else:
            _q, d = closest_point_on_prism_np(p, e1, e2)
        if best is None or d < best:
            best = d
    return best


class TestHeightfieldCapsuleContact(unittest.TestCase):
    def check_random_pairs(self, device, count=600, seed=11):
        rng = np.random.default_rng(seed)
        prisms, capsules, rot_a, rot_b, pos_a, pos_b, gaps = random_configurations(rng, count)

        def deepest(size):
            data = _Deepest()
            data.count = wp.zeros(size, dtype=int, device=device)
            data.distance = wp.full(size, 1.0e9, dtype=float, device=device)
            data.normal = wp.zeros(size, dtype=wp.vec3, device=device)
            return data

        # The generic multi-contact manifold is compared on CPU only; against this test's
        # recording writer it faults on CUDA (it is not the production reducer writer).
        with_generic = not wp.get_device(device).is_cuda
        generic, analytic = deepest(count if with_generic else 0), deepest(count)
        wp.launch(
            _compare_pairs,
            dim=count,
            inputs=[
                wp.array(prisms, dtype=support.GenericShapeData, device=device),
                wp.array(capsules, dtype=support.GenericShapeData, device=device),
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
        a_count, a_distance, a_normal = analytic.count.numpy(), analytic.distance.numpy(), analytic.normal.numpy()
        radii = np.array([c.scale[0] for c in capsules])
        a_admitted = (a_count > 0) & (a_distance <= gaps)
        self.assertGreater(int(a_admitted.sum()), count // 5)
        self.assertLessEqual(int(a_count.max()), 3)
        self.assertTrue(np.isfinite(a_distance[a_count > 0]).all())

        # Float64 reference for the deepest axis point against the prism.
        reference = np.full(count, np.nan)
        for i in range(count):
            e1 = np.array(prisms[i].scale, dtype=np.float64)
            e2 = np.array(prisms[i].auxiliary, dtype=np.float64)
            ra = _quat_matrix_np(np.array(rot_a[i], dtype=np.float64))
            rb = _quat_matrix_np(np.array(rot_b[i], dtype=np.float64))
            c = ra.T @ (pos_b[i] - pos_a[i])
            axis = ra.T @ (rb @ np.array([0.0, 0.0, capsules[i].scale[1]]))
            reference[i] = _segment_prism_reference(c - axis, c + axis, e1, e2) - 1.0e-4 - radii[i]
        # The closed form reproduces the reference deepest distance (the reference samples the
        # axis at 0.5 mm of a 0.4 m span, so allow its own discretization).
        self.assertLess(float(np.abs(a_distance - reference)[a_admitted].max()), 5.0e-4)
        # Pairs the reference admits are admitted; pairs it rejects by a margin are rejected.
        self.assertTrue(bool(np.all(a_admitted[reference <= gaps - 5.0e-4])))
        self.assertFalse(bool(np.any(a_admitted[reference > gaps + 5.0e-4])))
        if not with_generic:
            return

        g_count, g_distance, g_normal = generic.count.numpy(), generic.distance.numpy(), generic.normal.numpy()
        g_admitted = (g_count > 0) & (g_distance <= gaps)
        both = g_admitted & a_admitted
        self.assertGreater(int(both.sum()), count // 5)
        # The generic manifold reports the plane depth of its deepest end cap when that cap
        # projects onto the face, and otherwise falls back to MPR's witness, which is
        # shallower than the closed form's interior contact when the deepest axis point
        # inside the footprint is not an end cap. That is the case the closed form fixes;
        # it stays a minority of the loaded pairs.
        shallower = (g_distance - a_distance)[both] > 2.0e-3
        self.assertLess(int(shallower.sum()), int(0.15 * both.sum()))
        for i in np.flatnonzero(a_admitted & ~g_admitted):
            self.assertLess(abs(float(reference[i] - gaps[i])), 2.0e-3, f"pair {i}")
        # Normals of the deepest contact agree for the clear majority (the deepest point of a
        # lying capsule across an edge is ambiguous between the two queries).
        angles = _angle(g_normal[both], a_normal[both])
        self.assertLess(float(np.percentile(angles, 80)), 2.0e-2)

    def test_random_pairs_cpu(self):
        """Match the generic manifold's deepest contact on random cells and capsule poses (CPU)."""
        self.check_random_pairs("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_random_pairs_cuda(self):
        """Match the float64 reference on CUDA."""
        self.check_random_pairs("cuda:0")

    def test_kernel_selection(self):
        """Capsules get a typed analytic query behind the shared switch."""
        analytic = create_mesh_triangle_contacts_to_reducer_kernel(int(newton.GeoType.CAPSULE), analytic=True)
        generic = create_mesh_triangle_contacts_to_reducer_kernel(int(newton.GeoType.CAPSULE), analytic=False)
        self.assertIsNot(analytic, generic)
        self.assertTrue(analytic.key.endswith("capsule_contacts_to_reducer_kernel_analytic"))
        with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_ANALYTIC_PRIMITIVES": "0"}):
            self.assertIs(create_mesh_triangle_contacts_to_reducer_kernel(int(newton.GeoType.CAPSULE)), generic)

    def check_resting(self, device):
        """A capsule lying on flat and sloped terrain rests at the same height through both queries."""
        for slope in (0.0, 0.15):
            with self.subTest(device=device, slope=slope):
                builder = newton.ModelBuilder()
                cfg = builder.ShapeConfig(margin=0.001, gap=0.002)
                grid = np.linspace(-0.4, 0.4, 9, dtype=np.float32)
                heights = (slope * grid[None, :] + 0.0 * grid[:, None]).astype(np.float32)
                terrain = newton.Heightfield(data=heights, nrow=9, ncol=9, hx=0.4, hy=0.4)
                builder.add_shape_heightfield(heightfield=terrain, cfg=cfg)
                body = builder.add_body(
                    xform=wp.transform(
                        wp.vec3(0.03, 0.02, 0.12), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * np.pi)
                    ),
                    mass=1.0,
                )
                builder.add_shape_capsule(body, radius=0.03, half_height=0.08, cfg=cfg)
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
                    # The axis lies along x; the body rests one radius above the surface
                    # under its center (plus the slope of the terrain under it).
                    expected = 0.03 + slope * float(pose[0, 0])
                    self.assertAlmostEqual(float(pose[0, 2]), expected, delta=0.01)
                    self.assertLess(float(np.abs(velocity).max()), 0.3)
                np.testing.assert_allclose(outcomes[0][0][:, :3], outcomes[1][0][:, :3], atol=0.005, rtol=0.0)

    def test_resting_cpu(self):
        """Rest a lying capsule on flat and sloped terrain through both queries on CPU."""
        self.check_resting("cpu")


if __name__ == "__main__":
    unittest.main()
