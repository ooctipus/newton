# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the closed-form sphere-vs-prism contact against the generic GJK/MPR query."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry import support_function as support
from newton._src.geometry.collision_core import MINKOWSKI_CORE_RADIUS, create_compute_gjk_mpr_contacts
from newton._src.geometry.contact_data import ContactData
from newton._src.geometry.contact_reduction_global import create_mesh_triangle_contacts_to_reducer_kernel
from newton._src.geometry.heightfield_sphere_contact import (
    _closest_point_on_prism_sides,
    _xy_inside_triangle,
    create_sphere_prism_contacts,
)
from newton._src.geometry.support_function import (
    TRIANGLE_PRISM_EXTRUSION,
    closest_point_on_triangle,
    create_triangle_prism_penetration_refiner,
)
from newton.tests.test_heightfield_primitive_cull import primitive_model
from newton.tests.test_heightfield_primitive_query import FLAGS, TestHeightfieldPrimitiveQuery

# Case codes recorded by the comparison kernel.
CASE_SEPARATED = 0
CASE_INSIDE_ON_FACE = 1
CASE_INSIDE_OFF_FACE = 2
CASE_NEAR_SURFACE = 3


@wp.struct
class _Capture:
    written: wp.array[int]
    center: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    distance: wp.array[float]
    radius_eff_b: wp.array[float]


@wp.func
def _capture_contact(contact: ContactData, data: _Capture, output_index: int):
    i = contact.sort_sub_key >> 3
    data.written[i] = 1
    data.center[i] = contact.contact_point_center
    data.normal[i] = contact.contact_normal_a_to_b
    data.distance[i] = contact.contact_distance
    data.radius_eff_b[i] = contact.radius_eff_b


_generic = create_compute_gjk_mpr_contacts(
    _capture_contact,
    support_func=support.support_map_heightfield_sphere,
    use_precomputed_center=True,
    penetration_refiner=create_triangle_prism_penetration_refiner(support.support_map_heightfield_sphere),
)
_analytic = create_sphere_prism_contacts(_capture_contact)


@wp.kernel(enable_backward=False)
def _compare_pairs(
    prisms: wp.array[support.GenericShapeData],
    spheres: wp.array[support.GenericShapeData],
    rot_a: wp.array[wp.quat],
    rot_b: wp.array[wp.quat],
    pos_a: wp.array[wp.vec3],
    pos_b: wp.array[wp.vec3],
    gaps: wp.array[float],
    margins: wp.array[wp.vec2],
    generic: _Capture,
    analytic: _Capture,
    cases: wp.array[int],
    plane_depth: wp.array[float],
):
    tid = wp.tid()
    prism = prisms[tid]
    sphere = spheres[tid]
    margin = margins[tid]
    _generic(
        prism,
        sphere,
        rot_a[tid],
        rot_b[tid],
        pos_a[tid],
        pos_b[tid],
        gaps[tid],
        0,
        1,
        margin[0],
        margin[1],
        support.SupportMapDataProvider(),
        generic,
        tid,
        0,
    )
    _analytic(
        prism,
        sphere,
        rot_a[tid],
        rot_b[tid],
        pos_a[tid],
        pos_b[tid],
        gaps[tid],
        0,
        1,
        margin[0],
        margin[1],
        analytic,
        tid,
        0,
    )

    # Classify the configuration for tolerance selection.
    e1 = prism.scale
    e2 = prism.auxiliary
    n = wp.normalize(wp.cross(e1, e2))
    if n[2] < 0.0:
        n = -n
    c = wp.quat_rotate_inv(rot_a[tid], pos_b[tid] - pos_a[tid])
    s = wp.dot(c, n)
    projected = c - s * n
    top = closest_point_on_triangle(projected, wp.vec3(0.0), e1, e2)
    on_face = wp.length_sq(projected - top) < 1.0e-10
    inside = s < 0.0 and -s <= TRIANGLE_PRISM_EXTRUSION * n[2] and _xy_inside_triangle(c, e1, e2)
    case = CASE_SEPARATED
    if inside:
        case = CASE_INSIDE_ON_FACE
        if not on_face:
            case = CASE_INSIDE_OFF_FACE
    else:
        _best, best_sq = _closest_point_on_prism_sides(c, e1, e2, top, wp.length_sq(c - top), 1.0e6)
        if best_sq < 16.0 * MINKOWSKI_CORE_RADIUS * MINKOWSKI_CORE_RADIUS:
            case = CASE_NEAR_SURFACE
    cases[tid] = case
    plane_depth[tid] = s


def _closest_on_triangle_np(p, a, b, c):
    """Float64 closest point on a triangle (Ericson, Real-Time Collision Detection 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + ab * (d1 / (d1 - d3))
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + ac * (d2 / (d2 - d6))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))
    denom = 1.0 / (va + vb + vc)
    return a + ab * (vb * denom) + ac * (vc * denom)


def _angle(a, b):
    """Angle between row vectors, robust near zero (arccos loses float32 precision there)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a * b, axis=1))


def _quat_matrix_np(q):
    """Rotation matrix of a Warp (x, y, z, w) quaternion in float64."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def closest_point_on_prism_np(c, e1, e2):
    """Float64 closest point on the prism (triangle extruded by 1 m along -Z) to an outside point."""
    down = np.array([0.0, 0.0, TRIANGLE_PRISM_EXTRUSION])
    a0 = np.zeros(3)
    faces = [
        (a0, e1, e2),
        (a0 - down, e1 - down, e2 - down),
        (a0, e1, e1 - down),
        (a0, e1 - down, a0 - down),
        (e1, e2, e2 - down),
        (e1, e2 - down, e1 - down),
        (e2, a0, a0 - down),
        (e2, a0 - down, e2 - down),
    ]
    best, best_sq = None, np.inf
    for face in faces:
        q = _closest_on_triangle_np(c, *face)
        d_sq = float((c - q) @ (c - q))
        if d_sq < best_sq:
            best, best_sq = q, d_sq
    return best, np.sqrt(best_sq)


def random_configurations(rng, count):
    """Sample heightfield cells, spheres and poses around contact."""
    prisms, spheres, rot_a, rot_b, pos_a, pos_b, gaps, margins = [], [], [], [], [], [], [], []
    for _ in range(count):
        dx, dy = rng.uniform(0.05, 0.4, size=2)
        slope = 0.0 if rng.random() < 0.15 else rng.uniform(0.0, 0.4)
        h1, h2 = rng.uniform(-slope, slope, size=2)
        if rng.random() < 0.5:
            e1, e2 = np.array([dx, 0.0, h1]), np.array([dx, dy, h2])
        else:
            e1, e2 = np.array([dx, dy, h1]), np.array([0.0, dy, h2])
        n = np.cross(e1, e2)
        n /= np.linalg.norm(n)
        if n[2] < 0.0:
            n = -n
        radius = rng.uniform(0.01, 0.08)
        gap = rng.choice([0.0, 0.001, 0.005, 0.02])
        margin = np.zeros(2) if rng.random() < 0.5 else rng.uniform(0.0, 0.003, size=2)
        mode = rng.random()
        if mode < 0.6:
            u, v = rng.uniform(0.0, 1.0, size=2)
            if u + v > 1.0:
                u, v = 1.0 - u, 1.0 - v
            target = u * e1 + v * e2
        elif mode < 0.85:
            u, v = rng.uniform(-0.5, 1.5, size=2)
            target = u * e1 + v * e2
        else:
            u, v = rng.uniform(0.0, 1.0, size=2)
            if u + v > 1.0:
                u, v = 1.0 - u, 1.0 - v
            target = u * e1 + v * e2 - np.array([0.0, 0.0, rng.uniform(0.0, 0.3)])
        offset = rng.uniform(-radius - 0.05, gap + 0.05)
        c = target + n * (radius + offset) + rng.normal(0.0, 0.03, size=3)
        kind = rng.random()
        if kind < 0.5:
            qa = wp.quat_identity()
        elif kind < 0.8:
            qa = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rng.uniform(-np.pi, np.pi)))
        else:
            axis = rng.normal(size=3)
            qa = wp.quat_from_axis_angle(wp.normalize(wp.vec3(*axis)), float(rng.uniform(-np.pi, np.pi)))
        axis = rng.normal(size=3)
        qb = wp.quat_from_axis_angle(wp.normalize(wp.vec3(*axis)), float(rng.uniform(-np.pi, np.pi)))
        pa = rng.uniform(-20.0, 20.0, size=3)
        pb = pa + np.asarray(wp.quat_rotate(qa, wp.vec3(*c)))
        prism = support.GenericShapeData()
        prism.shape_type = int(support.GeoTypeEx.TRIANGLE_PRISM)
        prism.scale = wp.vec3(*e1)
        prism.auxiliary = wp.vec3(*e2)
        prism.center = wp.vec3(0.0)
        sphere = support.GenericShapeData()
        sphere.shape_type = int(newton.GeoType.SPHERE)
        sphere.scale = wp.vec3(radius, 0.0, 0.0)
        sphere.auxiliary = wp.vec3(0.0)
        sphere.center = wp.vec3(0.0)
        prisms.append(prism)
        spheres.append(sphere)
        rot_a.append(qa)
        rot_b.append(qb)
        pos_a.append(pa)
        pos_b.append(pb)
        gaps.append(gap)
        margins.append(margin)
    return prisms, spheres, rot_a, rot_b, np.array(pos_a), np.array(pos_b), np.array(gaps), np.array(margins)


class TestHeightfieldSphereContact(unittest.TestCase):
    raw_contacts = TestHeightfieldPrimitiveQuery.raw_contacts
    assert_current_contacts = TestHeightfieldPrimitiveQuery.assert_current_contacts

    def check_random_pairs(self, device, count=4000, seed=7):
        rng = np.random.default_rng(seed)
        prisms, spheres, rot_a, rot_b, pos_a, pos_b, gaps, margins = random_configurations(rng, count)

        def capture():
            data = _Capture()
            data.written = wp.zeros(count, dtype=int, device=device)
            data.center = wp.zeros(count, dtype=wp.vec3, device=device)
            data.normal = wp.zeros(count, dtype=wp.vec3, device=device)
            data.distance = wp.zeros(count, dtype=float, device=device)
            data.radius_eff_b = wp.zeros(count, dtype=float, device=device)
            return data

        generic, analytic = capture(), capture()
        cases = wp.zeros(count, dtype=int, device=device)
        plane_depth = wp.zeros(count, dtype=float, device=device)
        wp.launch(
            _compare_pairs,
            dim=count,
            inputs=[
                wp.array(prisms, dtype=support.GenericShapeData, device=device),
                wp.array(spheres, dtype=support.GenericShapeData, device=device),
                wp.array(rot_a, dtype=wp.quat, device=device),
                wp.array(rot_b, dtype=wp.quat, device=device),
                wp.array(pos_a, dtype=wp.vec3, device=device),
                wp.array(pos_b, dtype=wp.vec3, device=device),
                wp.array(gaps, dtype=float, device=device),
                wp.array(margins, dtype=wp.vec2, device=device),
                generic,
                analytic,
                cases,
                plane_depth,
            ],
            device=device,
        )
        wp.synchronize_device(device)
        cases = cases.numpy()
        plane_depth = plane_depth.numpy()
        g_written, a_written = generic.written.numpy() == 1, analytic.written.numpy() == 1
        g_distance, a_distance = generic.distance.numpy(), analytic.distance.numpy()
        g_center, a_center = generic.center.numpy(), analytic.center.numpy()
        g_normal, a_normal = generic.normal.numpy(), analytic.normal.numpy()
        radii = np.array([s.scale[0] for s in spheres])
        np.testing.assert_array_equal(analytic.radius_eff_b.numpy()[a_written], radii[a_written])

        # The reducer admits a contact when the separation is within the gap sum.
        margin_sum = margins.sum(axis=1)
        g_excess = g_distance - margin_sum - gaps
        a_excess = a_distance - margin_sum - gaps
        g_admitted = g_written & (g_excess <= 0.0)
        a_admitted = a_written & (a_excess <= 0.0)
        self.assertGreater(int(g_admitted.sum()), count // 4, "The sample must load many pairs")
        disagree = g_admitted != a_admitted
        for i in np.flatnonzero(disagree):
            excess = g_excess[i] if g_written[i] else a_excess[i]
            self.assertLess(abs(float(excess)), 1.0e-4, f"admission differs away from the threshold at pair {i}")
        both = g_admitted & a_admitted
        self.assertGreater(int(both.sum()), count // 4)

        distance_error = np.abs(g_distance - a_distance)
        center_error = np.linalg.norm(g_center - a_center, axis=1)
        normal_error = _angle(g_normal, a_normal)
        strict = both & np.isin(cases, [CASE_SEPARATED, CASE_INSIDE_ON_FACE])
        resting = strict & (cases == CASE_INSIDE_ON_FACE)
        separated = strict & (cases == CASE_SEPARATED)
        self.assertGreater(int(resting.sum()), 0)
        self.assertGreater(int(separated.sum()), 0)
        # Penetrating through the face: the refiner's plane depth, float32 roundoff only
        # (world positions up to 20 m).
        self.assertLess(float(distance_error[resting].max()), 2.0e-5)
        self.assertLess(float(center_error[resting].max()), 3.0e-5)
        self.assertLess(float(normal_error[resting].max()), 1.0e-3)
        # Separated: GJK converges to COLLIDE_EPSILON = 1e-4, so the generic query may stop
        # before resolving its 0.1 mm core radius and its witness direction is accurate to
        # about COLLIDE_EPSILON / distance. The closed form matches a float64 reference of the
        # true sphere surface.
        exact_distance = np.zeros(count)
        exact_center = np.zeros((count, 3))
        exact_normal = np.zeros((count, 3))
        for i in np.flatnonzero(separated):
            e1, e2 = np.array(prisms[i].scale, dtype=np.float64), np.array(prisms[i].auxiliary, dtype=np.float64)
            rotation = _quat_matrix_np(np.array(rot_a[i], dtype=np.float64))
            c = rotation.T @ (pos_b[i] - pos_a[i])
            q, d = closest_point_on_prism_np(c, e1, e2)
            normal = (c - q) / d
            center = 0.5 * (q + c) - normal * (radii[i] * 0.5)
            exact_distance[i] = d - radii[i]
            exact_center[i] = rotation @ center + pos_a[i]
            exact_normal[i] = rotation @ normal
        analytic_distance_error = np.abs(a_distance - exact_distance)
        analytic_center_error = np.linalg.norm(a_center - exact_center, axis=1)
        analytic_normal_error = _angle(a_normal, exact_normal)
        self.assertLess(float(analytic_distance_error[separated].max()), 1.0e-5)
        self.assertLess(float(analytic_center_error[separated].max()), 3.0e-5)
        # Direction accuracy is bounded by float32 roundoff of the 20 m poses over the
        # separation (about 4e-6 / d).
        separation = np.maximum(exact_distance + radii, 1.0e-6)
        normal_bound = 1.0e-4 + 8.0e-6 / separation
        self.assertTrue(bool(np.all(analytic_normal_error[separated] < normal_bound[separated])))
        generic_distance_error = np.abs(g_distance - exact_distance)
        generic_normal_error = _angle(g_normal, exact_normal)
        self.assertLess(float(generic_distance_error[separated].max()), 1.2e-4)
        generic_bound = 2.0e-3 + 2.0e-4 / separation
        self.assertTrue(bool(np.all(generic_normal_error[separated] < generic_bound[separated])))
        self.assertLess(float(center_error[separated].max()), 1.0e-3)

        # Inside the prism off the face both queries use the top-face normal and the plane
        # depth; the generic witness differs by its 0.1 mm core. MPR occasionally fails to
        # detect a deep interior center and the generic query then publishes a zero-depth
        # contact; the closed form never does.
        inside = both & (cases == CASE_INSIDE_OFF_FACE)
        truth = plane_depth - radii
        self.assertGreater(int(inside.sum()), 0)
        np.testing.assert_allclose(a_distance[inside], truth[inside], atol=1.0e-5, rtol=0.0)
        generic_failed = inside & (np.abs(g_distance - truth) > 1.0e-3)
        self.assertLess(int(generic_failed.sum()), max(1, int(0.02 * inside.sum())))
        relaxed = inside & ~generic_failed
        self.assertLess(float(distance_error[relaxed].max()), 4.0e-4)
        self.assertLess(float(center_error[relaxed].max()), 4.0e-4)
        self.assertLess(float(normal_error[relaxed].max()), 1.0e-3)

        # Within 0.4 mm outside the prism the core sphere still overlaps it and the generic
        # query reports a top-face penetration of arbitrary depth; the closed form reports
        # the surface distance. The band is measure-small and must stay rare in the sample.
        band = both & (cases == CASE_NEAR_SURFACE)
        self.assertLess(int(band.sum()), max(1, int(0.02 * both.sum())))
        return {
            "strict": int(strict.sum()),
            "inside_off_face": int(inside.sum()),
            "generic_failed": int(generic_failed.sum()),
            "band": int(band.sum()),
            "max_distance": float(distance_error[strict].max()),
            "max_center": float(center_error[strict].max()),
            "max_normal": float(normal_error[strict].max()),
            "relaxed_distance": float(distance_error[relaxed].max()),
            "relaxed_center": float(center_error[relaxed].max()),
            "relaxed_normal": float(normal_error[relaxed].max()),
        }

    def test_random_pairs_cpu(self):
        """Match the generic query on random cells, slopes, borders and penetrations on CPU."""
        self.check_random_pairs("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_random_pairs_cuda(self):
        """Match the generic query on CUDA."""
        self.check_random_pairs("cuda:0")

    def test_kernel_selection(self):
        """Select the analytic query by default and keep the generic one behind the opt-out."""
        analytic = create_mesh_triangle_contacts_to_reducer_kernel(int(newton.GeoType.SPHERE), analytic=True)
        generic = create_mesh_triangle_contacts_to_reducer_kernel(int(newton.GeoType.SPHERE), analytic=False)
        self.assertIsNot(analytic, generic)
        self.assertTrue(analytic.key.endswith("_analytic"))
        with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_ANALYTIC_PRIMITIVES": "1"}):
            self.assertIs(create_mesh_triangle_contacts_to_reducer_kernel(int(newton.GeoType.SPHERE)), analytic)
        with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_ANALYTIC_PRIMITIVES": "0"}):
            self.assertIs(create_mesh_triangle_contacts_to_reducer_kernel(int(newton.GeoType.SPHERE)), generic)
        with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_ANALYTIC_PRIMITIVES": "yes"}):
            with self.assertRaises(ValueError):
                create_mesh_triangle_contacts_to_reducer_kernel(int(newton.GeoType.SPHERE))
        # The cylinder query ignores the flag.
        cylinder = create_mesh_triangle_contacts_to_reducer_kernel(int(newton.GeoType.CYLINDER), analytic=True)
        self.assertIs(cylinder, create_mesh_triangle_contacts_to_reducer_kernel(int(newton.GeoType.CYLINDER)))

    def check_loaded_terrain(self, device):
        """Compare whole-pipeline contacts of the typed sphere query with and without the closed form."""
        for options in ({}, {"slope": True}, {"border": True}, {"deep": True}):
            with self.subTest(device=device, options=options):
                model = primitive_model(device, "sphere", **options)
                state = model.state()
                owners = []
                for analytic in ("0", "1"):
                    env = {
                        **FLAGS,
                        "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "1",
                        "NEWTON_HEIGHTFIELD_ANALYTIC_PRIMITIVES": analytic,
                    }
                    with patch.dict(os.environ, env):
                        owners.append(
                            newton.CollisionPipeline(
                                model, reduce_contacts=True, rigid_contact_max=4096, max_triangle_pairs=4096
                            )
                        )
                old, new = owners
                self.assertTrue(new.narrow_phase._heightfield_primitive_kernels[0].key.endswith("_analytic"))
                self.assertFalse(old.narrow_phase._heightfield_primitive_kernels[0].key.endswith("_analytic"))
                old_contacts, new_contacts = old.contacts(), new.contacts()
                old.collide(state, old_contacts)
                new.collide(state, new_contacts)
                self.assert_current_contacts(old, new, state, old_contacts, new_contacts)

    def test_loaded_terrain_cpu(self):
        """Keep terrain contacts of a loaded sphere across slopes, borders and deep overlap on CPU."""
        self.check_loaded_terrain("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_loaded_terrain_cuda(self):
        """Keep terrain contacts of a loaded sphere on CUDA."""
        self.check_loaded_terrain("cuda:0")


if __name__ == "__main__":
    unittest.main()
