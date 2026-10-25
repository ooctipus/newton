# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check typed heightfield queries against the original geometry path."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry import support_function as support
from newton._src.geometry.contact_reduction_global import decode_oct
from newton._src.geometry.narrow_phase import NarrowPhase, write_contact_simple
from newton.tests.test_heightfield_packed_pairs import logical
from newton.tests.test_heightfield_primitive_cull import primitive_model

FLAGS = {
    "NEWTON_HEIGHTFIELD_CELL_REJECT": "1",
    "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1",
    "NEWTON_HEIGHTFIELD_FINITE_QUERY": "0",
    "NEWTON_HEIGHTFIELD_PAIR_CSR": "0",
    # This module checks the typed GJK/MPR queries against the generic one; the
    # closed-form rows have their own tests (test_heightfield_*_contact).
    "NEWTON_HEIGHTFIELD_ANALYTIC_PRIMITIVES": "0",
    # Both are terrain-route queries: keep the route for every heightfield pair.
    "NEWTON_TERRAIN_LAW": "none",
}


@wp.kernel(enable_backward=False)
def _evaluate_supports(
    shapes: wp.array[support.GenericShapeData],
    routes: wp.array[int],
    directions: wp.array[wp.vec3],
    rotations: wp.array[wp.quat],
    original: wp.array[wp.vec3],
    typed: wp.array[wp.vec3],
):
    tid = wp.tid()
    shape = shapes[tid]
    rotation = rotations[tid]
    direction = wp.quat_rotate_inv(rotation, directions[tid])
    provider = support.SupportMapDataProvider()
    original[tid] = wp.quat_rotate(rotation, support.support_map(shape, direction, provider))
    point = wp.vec3(0.0)
    if routes[tid] == int(newton.GeoType.SPHERE):
        point = support.support_map_heightfield_sphere(shape, direction, provider)
    else:
        point = support.support_map_heightfield_cylinder(shape, direction, provider)
    typed[tid] = wp.quat_rotate(rotation, point)


def pipeline(model, enabled):
    """Toggle query specialization while preserving append and reduction settings."""
    with patch.dict(os.environ, {**FLAGS, "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": str(int(enabled))}):
        return newton.CollisionPipeline(model, reduce_contacts=True, rigid_contact_max=4096, max_triangle_pairs=4096)


class TestHeightfieldPrimitiveQuery(unittest.TestCase):
    def test_typed_query_admission_cpu(self):
        """Admit explicit typed queries for the stock buffered heightfield owner."""

        def create(**options):
            options = {"has_heightfields": True, "has_meshes": False, **options}
            return NarrowPhase(max_candidate_pairs=4, max_triangle_pairs=16, device="cpu", **options)

        with patch.dict(os.environ, {**FLAGS, "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "1"}):
            self.assertTrue(create()._heightfield_primitive_query)
            for options in (
                {"has_heightfields": False},
                {"reduce_contacts": False},
                {"speculative": True},
                {"deterministic": True},
                {"contact_writer_warp_func": write_contact_simple},
            ):
                with self.subTest(options=options):
                    self.assertFalse(create(**options)._heightfield_primitive_query)
        with patch.dict(os.environ, {**FLAGS, "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "0"}):
            self.assertFalse(create()._heightfield_primitive_query)
        with patch.dict(os.environ, FLAGS):
            os.environ.pop("NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY", None)
            self.assertFalse(create()._heightfield_primitive_query)
        with patch.dict(os.environ, {**FLAGS, "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "yes"}):
            with self.assertRaises(ValueError):
                create()
        with patch.dict(
            os.environ,
            {**FLAGS, "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "1", "NEWTON_HEIGHTFIELD_FINITE_QUERY": "1"},
        ):
            self.assertFalse(create()._heightfield_primitive_query)

    def check_supports(self, device):
        """Preserve native support tie, epsilon, barrel, and rotated direction rules."""
        directions = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (0.0, 1.0, 1.0),
            (0.3, -0.4, 0.8),
            (1e-7, 0.0, 0.0),
            (1e-6, 0.0, 0.0),
            (1.001e-6, 0.0, 0.0),
            (1.001e-6, 0.0, -1e-12),
            (1.001e-6, 0.0, 1e-12),
        ]
        rotations = [wp.quat_identity(), wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, -0.7, 0.2)), 1.1)]
        configurations = [
            (newton.GeoType.SPHERE, newton.GeoType.SPHERE, (radius, 0.0, 0.0), (0.0, 0.0, 0.0))
            for radius in (0.0001, 0.01, 0.031455, 1.0)
        ]
        configurations += [
            (newton.GeoType.CYLINDER, newton.GeoType.CYLINDER, scale, (0.0, 0.0, 0.0))
            for scale in ((0.0175, 0.19786, 0.0), (0.07, 0.081565, 0.0), (0.025, 0.1, 0.2), (0.025, 0.1, 0.1))
        ]
        configurations += [
            (route, support.GeoTypeEx.TRIANGLE_PRISM, (0.1, 0.0, 0.01), (0.1, 0.1, 0.02))
            for route in (newton.GeoType.SPHERE, newton.GeoType.CYLINDER)
        ]
        shapes, routes, query_directions, query_rotations = [], [], [], []
        for route, kind, scale, auxiliary in configurations:
            for rotation in rotations:
                for direction in directions:
                    shape = support.GenericShapeData()
                    shape.shape_type = int(kind)
                    shape.scale = wp.vec3(*scale)
                    shape.auxiliary = wp.vec3(*auxiliary)
                    shapes.append(shape)
                    routes.append(int(route))
                    query_directions.append(direction)
                    query_rotations.append(rotation)
        original = wp.empty(len(shapes), dtype=wp.vec3, device=device)
        typed = wp.empty_like(original)
        wp.launch(
            _evaluate_supports,
            dim=len(shapes),
            inputs=[
                wp.array(shapes, dtype=support.GenericShapeData, device=device),
                wp.array(routes, dtype=int, device=device),
                wp.array(query_directions, dtype=wp.vec3, device=device),
                wp.array(query_rotations, dtype=wp.quat, device=device),
                original,
                typed,
            ],
            device=device,
        )
        self.assertTrue(np.isfinite(typed.numpy()).all())
        np.testing.assert_allclose(typed.numpy(), original.numpy(), rtol=2e-6, atol=2e-7)

    def test_supports_cpu(self):
        """Compare typed primitive and prism support against generic support on CPU."""
        self.check_supports("cpu")

    def raw_contacts(self, owner):
        """Identify unreduced callbacks independently of atomic allocation order."""
        reducer = owner.narrow_phase.global_contact_reducer
        count = int(reducer.contact_count.numpy()[0])
        self.assertLess(count, reducer.capacity)
        pair = reducer.shape_pairs.numpy()[1 : count + 1]
        fingerprints = reducer.contact_fingerprints.numpy()[1 : count + 1]
        positions = reducer.position_depth.numpy()[1 : count + 1]
        normals = reducer.normal.numpy()[1 : count + 1]
        self.assertTrue(np.isfinite(positions).all() and np.isfinite(normals).all())
        result = {}
        for p, fingerprint, position, normal in zip(pair, fingerprints, positions, normals, strict=True):
            key = (int(p[0]), int(p[1]), int(fingerprint))
            self.assertNotIn(key, result, "A triangle callback must be emitted by exactly one query pass")
            result[key] = np.r_[position, normal]
        return result

    def assert_current_contacts(self, old, new, state, old_contacts, new_contacts):
        """Retain callback ownership and loaded finite support without reduction tie identity."""
        for owner in (old, new):
            owner.narrow_phase.check_buffer_capacity()
            self.assertFalse(owner.narrow_phase._heightfield_finite_query)
        self.assertEqual(logical(old.narrow_phase), logical(new.narrow_phase))
        before, after = self.raw_contacts(old), self.raw_contacts(new)
        self.assertTrue(before, "The fixture must exercise triangle callbacks")
        self.assertEqual(before.keys(), after.keys())
        for key in before:
            np.testing.assert_allclose(before[key][:4], after[key][:4], rtol=2e-6, atol=2e-4)
            # Octahedral coordinates are not an isotropic direction metric. Use the
            # physical 3D normal limit from TestHeightfieldCellReject instead.
            old_normal = np.asarray(decode_oct(wp.vec2(*before[key][4:])))
            new_normal = np.asarray(decode_oct(wp.vec2(*after[key][4:])))
            # GJK witness directions of the generic query are accurate to about
            # COLLIDE_EPSILON / distance for side-wall contacts of a loaded sphere.
            self.assertLess(float(np.linalg.norm(old_normal - new_normal)), 8e-3)
        observations = []
        for owner, contacts in ((old, old_contacts), (new, new_contacts)):
            count = int(contacts.rigid_contact_count.numpy()[0])
            self.assertGreater(count, 0, "A loaded primitive must retain physical support")
            self.assertLess(count, contacts.rigid_contact_max)
            distance = wp.empty(contacts.rigid_contact_max, dtype=float, device=state.body_q.device)
            point = wp.empty(contacts.rigid_contact_max, dtype=wp.vec3, device=state.body_q.device)
            newton.eval_rigid_contact_kinematics(
                owner.model, state, contacts, out_distance=distance, out_point0_world=point
            )
            d, p = distance.numpy()[:count], point.numpy()[:count]
            normals = contacts.rigid_contact_normal.numpy()[:count]
            self.assertTrue(np.isfinite(d).all() and np.isfinite(p).all() and np.isfinite(normals).all())
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=2e-5)
            self.assertLess(float(d.min()), 0.005)
            shape_pairs = np.column_stack(
                (
                    contacts.rigid_contact_shape0.numpy()[:count],
                    contacts.rigid_contact_shape1.numpy()[:count],
                )
            )
            pairs = set(map(tuple, shape_pairs.tolist()))
            observations.append((float(d.min()), pairs, d, p, normals, shape_pairs))
        self.assertAlmostEqual(observations[0][0], observations[1][0], delta=2e-4)
        self.assertEqual(observations[0][1], observations[1][1])
        # Match public geometry in both directions without imposing reduction row order.
        for source, target in (observations, observations[::-1]):
            for pair in source[1]:
                src = np.all(source[5] == pair, axis=1)
                dst = np.all(target[5] == pair, axis=1)
                error = np.maximum.reduce(
                    (
                        np.abs(source[2][src][:, None] - target[2][dst][None, :]) / 2e-4,
                        np.linalg.norm(source[3][src][:, None] - target[3][dst][None, :], axis=2) / 2e-4,
                        np.linalg.norm(source[4][src][:, None] - target[4][dst][None, :], axis=2) / 8e-3,
                    )
                )
                self.assertLess(float(np.max(np.min(error, axis=1))), 1.0)

    def check_loaded_geometry(self, device):
        """Retain separated and deep contacts across slopes, borders and fallback types."""
        cases = [
            (kind, options)
            for kind in ("sphere", "cylinder")
            for options in ({}, {"slope": True}, {"border": True}, {"deep": True})
        ]
        # Capsules now have their own typed closed-form query; see test_heightfield_capsule_contact.
        cases += [("cylinder", {"barrel": True, "slope": True})]
        for kind, options in cases:
            with self.subTest(device=device, kind=kind, options=options):
                model = primitive_model(device, kind, **options)
                state = model.state()
                old, new = pipeline(model, False), pipeline(model, True)
                self.assertTrue(new.narrow_phase._heightfield_primitive_query)
                old_contacts, new_contacts = old.contacts(), new.contacts()
                old.collide(state, old_contacts)
                if options.get("barrel"):
                    self.collide_barrel_fallback(new, state, new_contacts)
                else:
                    new.collide(state, new_contacts)
                self.assert_current_contacts(old, new, state, old_contacts, new_contacts)

    def collide_barrel_fallback(self, owner, state, contacts):
        """Verify only the generic query publishes a current barrel-cylinder pair."""
        kernels = owner.narrow_phase._heightfield_primitive_kernels
        reducer = owner.narrow_phase.global_contact_reducer
        allocations = []
        launch = wp.launch

        def observe(*args, **kwargs):
            kernel = kwargs.get("kernel", args[0] if args else None)
            if kernel not in kernels:
                return launch(*args, **kwargs)
            before = int(reducer.contact_count.numpy()[0])
            result = launch(*args, **kwargs)
            allocations.append((kernels.index(kernel), int(reducer.contact_count.numpy()[0]) - before))
            return result

        with patch.object(wp, "launch", side_effect=observe):
            owner.collide(state, contacts)
        self.assertEqual([index for index, count in allocations], [0, 1, 2, 3, 4])
        self.assertEqual([count for index, count in allocations if index != 4], [0, 0, 0, 0])
        self.assertGreater(allocations[4][1], 0)

    def test_loaded_geometry_cpu(self):
        """Compare real typed and generic terrain contacts and capsule fallback on CPU."""
        self.check_loaded_geometry("cpu")

    def check_mesh_fallback(self, device):
        """Keep mesh triangles on the generic query despite the admitted primitive owner."""
        builder = newton.ModelBuilder()
        terrain = newton.Heightfield(data=np.zeros((5, 5), np.float32), nrow=5, ncol=5, hx=0.4, hy=0.4)
        builder.add_shape_heightfield(heightfield=terrain, xform=wp.transform((10.0, 0.0, 0.0), wp.quat_identity()))
        mesh = newton.Mesh(
            vertices=np.array([[-0.2, -0.2, 0.0], [0.2, -0.2, 0.0], [0.2, 0.2, 0.0], [-0.2, 0.2, 0.0]], np.float32),
            indices=np.array([0, 1, 2, 0, 2, 3], np.int32),
        )
        builder.add_shape_mesh(body=-1, mesh=mesh)
        body = builder.add_body(xform=wp.transform((0.014, 0.009, 0.03), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.045)
        model = builder.finalize(device=device)
        state = model.state()
        old, new = pipeline(model, False), pipeline(model, True)
        self.assertTrue(new.narrow_phase._heightfield_primitive_query)
        old_contacts, new_contacts = old.contacts(), new.contacts()
        old.collide(state, old_contacts)
        new.collide(state, new_contacts)
        self.assert_current_contacts(old, new, state, old_contacts, new_contacts)
        types = model.shape_type.numpy()
        self.assertTrue(all(types[a] == newton.GeoType.MESH for a, _, _ in logical(new.narrow_phase)))

    def test_mesh_fallback_cpu(self):
        """Retain loaded mesh-sphere callbacks in the separate generic fallback."""
        self.check_mesh_fallback("cpu")

    def test_loaded_support_cpu(self):
        """Rest a sphere and cylinder under gravity through the typed query owner."""
        for kind in ("sphere", "cylinder"):
            with self.subTest(kind=kind):
                builder = newton.ModelBuilder()
                cfg = builder.ShapeConfig(margin=0.001, gap=0.002)
                terrain = newton.Heightfield(data=np.zeros((9, 9), np.float32), nrow=9, ncol=9, hx=0.4, hy=0.4)
                builder.add_shape_heightfield(heightfield=terrain, cfg=cfg)
                body = builder.add_body(xform=wp.transform((0.014, 0.009, 0.10), wp.quat_identity()), mass=1.0)
                if kind == "sphere":
                    builder.add_shape_sphere(body, radius=0.045, cfg=cfg)
                    rest_height = 0.045
                else:
                    builder.add_shape_cylinder(body, radius=0.045, half_height=0.06, cfg=cfg)
                    rest_height = 0.06
                model = builder.finalize(device="cpu")
                outcomes = []
                for enabled in (False, True):
                    owner = pipeline(model, enabled)
                    solver = newton.solvers.SolverXPBD(model, iterations=10)
                    state, next_state = model.state(), model.state()
                    contacts, control = owner.contacts(), model.control()
                    for _ in range(200):
                        state.clear_forces()
                        owner.collide(state, contacts)
                        solver.step(state, next_state, control, contacts, 0.002)
                        state, next_state = next_state, state
                    owner.narrow_phase.check_buffer_capacity()
                    pose, velocity = state.body_q.numpy(), state.body_qd.numpy()
                    self.assertTrue(np.isfinite(pose).all() and np.isfinite(velocity).all())
                    self.assertAlmostEqual(float(pose[body, 2]), rest_height, delta=0.01)
                    self.assertLess(abs(float(velocity[body, 2])), 0.1)
                    outcomes.append((pose, velocity))
                np.testing.assert_allclose(outcomes[0][0], outcomes[1][0], atol=0.002, rtol=0.0)
                np.testing.assert_allclose(outcomes[0][1], outcomes[1][1], atol=0.05, rtol=0.0)

    def check_lifecycle(self, device):
        """Read current type, dimensions and terrain through empty and regrowing work."""
        for kind in ("sphere", "cylinder"):
            with self.subTest(device=device, kind=kind):
                model = primitive_model(device, kind)
                state = model.state()
                old, new = pipeline(model, False), pipeline(model, True)
                old_contacts, new_contacts = old.contacts(), new.contacts()
                old.collide(state, old_contacts)
                new.collide(state, new_contacts)
                self.assert_current_contacts(old, new, state, old_contacts, new_contacts)
                captured = None
                if wp.get_device(device).is_cuda:
                    with wp.ScopedCapture(device=device) as captured:
                        new.collide(state, new_contacts)

                def replay(captured=captured, new=new, state=state, new_contacts=new_contacts):
                    if captured is None:
                        new.collide(state, new_contacts)
                    else:
                        wp.capture_launch(captured.graph)

                poses = state.body_q.numpy().copy()
                away = poses.copy()
                away[:, 2] += 3.0
                state.body_q.assign(away)
                replay()
                self.assertEqual(logical(new.narrow_phase), [])
                self.assertEqual(int(new_contacts.rigid_contact_count.numpy()[0]), 0)
                types = model.shape_type.numpy().copy()
                scale = model.shape_scale.numpy().copy()
                for epoch in range(4):
                    current = poses.copy()
                    current[:, 0] += 0.007 * epoch
                    current[:, 3:] = np.array(
                        wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.8, 0.1, -0.3)), 0.4 + epoch * 0.2)
                    )
                    state.body_q.assign(current)
                    current_types = types.copy()
                    if epoch == 1:
                        # A barrel cylinder stays on the generic fallback in both pipelines.
                        current_types[1] = int(newton.GeoType.CYLINDER)
                    model.shape_type.assign(current_types)
                    current_scale = scale.copy()
                    current_scale[1, :2] *= 0.9
                    if epoch == 1:
                        current_scale[1, 1] = 0.05
                        current_scale[1, 2] = 0.2
                    if kind == "cylinder" and epoch == 2:
                        current_scale[1, 2] = 0.2
                    model.shape_scale.assign(current_scale)
                    heights = model.heightfield_elevations.numpy().copy()
                    heights[8 * 17 + 8] += 0.001
                    model.heightfield_elevations.assign(heights)
                    old.collide(state, old_contacts)
                    replay()
                    self.assert_current_contacts(old, new, state, old_contacts, new_contacts)

    def test_lifecycle_cpu(self):
        """Exercise current geometry and primitive-to-fallback type mutation on CPU."""
        self.check_lifecycle("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_supports_cuda(self):
        """Compare support tie and tiny-direction behavior on CUDA."""
        self.check_supports("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_loaded_geometry_cuda(self):
        """Compare loaded typed and fallback terrain contacts on CUDA."""
        self.check_loaded_geometry("cuda:0")
        self.check_mesh_fallback("cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_graph_lifecycle_cuda(self):
        """Replay empty, regrowing and type-mutated terrain work in a CUDA graph."""
        self.check_lifecycle("cuda:0")


if __name__ == "__main__":
    unittest.main()
