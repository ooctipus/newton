# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for Kamino's rigid contact capacity estimate."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.flags import ShapeFlags
from newton._src.geometry.types import GeoType
from newton._src.solvers.kamino.contact_capacity import (
    _CONVEX_PAIR_MAX_CONTACTS,
    _HYDRO_PAIR_MAX_CONTACTS,
    _MESH_PAIR_MAX_CONTACTS,
    estimate_rigid_contact_capacity,
)


class TestRigidContactCapacityEstimate(unittest.TestCase):
    """Kamino's estimate of the most rigid contacts a model's shapes may produce, for its fixed containers."""

    def test_visual_only_meshes_do_not_inflate_estimate(self):
        """Visual meshes should not affect rigid contact capacity estimates."""
        model = newton.Model()
        model.world_count = 1
        model.shape_contact_pair_count = 0

        shape_type = np.array(
            [int(GeoType.BOX)] * 4 + [int(GeoType.MESH)] * 100,
            dtype=np.int32,
        )
        shape_world = np.zeros(len(shape_type), dtype=np.int32)
        shape_flags = np.array(
            [int(ShapeFlags.COLLIDE_SHAPES)] * 4 + [int(ShapeFlags.VISIBLE)] * 100,
            dtype=np.int32,
        )

        model.shape_type = wp.array(shape_type, dtype=wp.int32)
        model.shape_world = wp.array(shape_world, dtype=wp.int32)
        model.shape_flags = wp.array(shape_flags, dtype=wp.int32)

        estimate = estimate_rigid_contact_capacity(model)
        self.assertEqual(estimate, 1000)

    def test_heuristic_caps_large_pair_count(self):
        """When pair count is huge, the heuristic provides a tighter bound."""
        model = newton.Model()
        model.world_count = 1
        model.shape_contact_pair_count = 999999

        # 4 primitives (CPP=5), 3 meshes (CPP=40), 2 planes, all in world 0.
        # non-plane: (4*20*5 + 3*20*40) // 2 = (400 + 2400) // 2 = 1400
        # weighted_plane_cpp: (4*5 + 3*40) // 7 = 140 // 7 = 20
        # plane (per-world): 2*7 pairs * 20 = 280
        # heuristic = 1680, pair = huge => min = 1680
        shape_type = np.array(
            [int(GeoType.BOX)] * 4 + [int(GeoType.MESH)] * 3 + [int(GeoType.PLANE)] * 2,
            dtype=np.int32,
        )
        shape_world = np.zeros(len(shape_type), dtype=np.int32)

        model.shape_type = wp.array(shape_type, dtype=wp.int32)
        model.shape_world = wp.array(shape_world, dtype=wp.int32)

        estimate = estimate_rigid_contact_capacity(model)
        self.assertEqual(estimate, 1680)

    def test_world_aware_plane_estimate(self):
        """Per-world plane computation avoids quadratic cross-world overcount."""
        model = newton.Model()
        model.world_count = 4
        model.shape_contact_pair_count = 0

        # 4 worlds, each with 10 boxes (CPP=5) and 10 planes.
        # non-plane: (40*20*5) // 2 = 2000
        # weighted_plane_cpp: (40*5) // 40 = 5
        # plane (per-world): 4*(10*10) pairs * 5 = 2000
        # total = 4000
        shape_type = np.array(
            ([int(GeoType.BOX)] * 10 + [int(GeoType.PLANE)] * 10) * 4,
            dtype=np.int32,
        )
        shape_world = np.repeat(np.arange(4, dtype=np.int32), 20)

        model.shape_type = wp.array(shape_type, dtype=wp.int32)
        model.shape_world = wp.array(shape_world, dtype=wp.int32)

        estimate = estimate_rigid_contact_capacity(model)
        self.assertEqual(estimate, 4000)

    def test_pair_count_tighter_than_heuristic(self):
        """When precomputed pair count is tighter than the heuristic, it is used."""
        model = newton.Model()
        model.world_count = 4
        model.shape_contact_pair_count = 300

        # 40 boxes (CPP=5) across 4 worlds, no planes.
        # heuristic: (40*20*5) // 2 = 2000
        # weighted_cpp: max(5, 5) = 5
        # pair-based: 300 * 5 = 1500
        # min(2000, 1500) = 1500
        shape_type = np.array(
            [int(GeoType.BOX)] * 40,
            dtype=np.int32,
        )
        shape_world = np.repeat(np.arange(4, dtype=np.int32), 10)

        model.shape_type = wp.array(shape_type, dtype=wp.int32)
        model.shape_world = wp.array(shape_world, dtype=wp.int32)

        estimate = estimate_rigid_contact_capacity(model)
        self.assertEqual(estimate, 1500)

    # ------------------------------------------------------------------
    # Pair-aware estimator tests (builder-based scenes, CPU)
    # ------------------------------------------------------------------

    @staticmethod
    def _cube_mesh(hs: float = 0.1) -> newton.Mesh:
        vertices = np.array(
            [
                [-hs, -hs, -hs],
                [hs, -hs, -hs],
                [hs, hs, -hs],
                [-hs, hs, -hs],
                [-hs, -hs, hs],
                [hs, -hs, hs],
                [hs, hs, hs],
                [-hs, hs, hs],
            ],
            dtype=np.float32,
        )
        # fmt: off
        indices = np.array(
            [
                0, 3, 2, 0, 2, 1,  # -z
                4, 5, 6, 4, 6, 7,  # +z
                0, 1, 5, 0, 5, 4,  # -y
                2, 3, 7, 2, 7, 6,  # +y
                0, 4, 7, 0, 7, 3,  # -x
                1, 2, 6, 1, 6, 5,  # +x
            ],
            dtype=np.int32,
        )
        # fmt: on
        return newton.Mesh(vertices, indices)

    @classmethod
    def _build_multi_world_scene(
        cls, num_worlds: int, boxes_per_world: int, mesh_per_world: bool = False
    ) -> newton.Model:
        """Global ground plane + ``num_worlds`` worlds of free-body boxes (and optionally one colliding mesh)."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for _ in range(num_worlds):
            builder.begin_world()
            for i in range(boxes_per_world):
                b = builder.add_body(xform=wp.transform(wp.vec3(0.5 * float(i), 0.0, 0.5), wp.quat_identity()))
                builder.add_shape_box(body=b, hx=0.1, hy=0.1, hz=0.1)
            if mesh_per_world:
                b = builder.add_body(xform=wp.transform(wp.vec3(-1.0, 0.0, 0.5), wp.quat_identity()))
                builder.add_shape_mesh(body=b, mesh=cls._cube_mesh())
            builder.end_world()
        return builder.finalize(device="cpu")

    @classmethod
    def _build_box_scene(cls, num_boxes: int, mesh_mode: str | None = None) -> newton.Model:
        """Ground plane + ``num_boxes`` free-body boxes, optionally plus a mesh shape.

        mesh_mode:
            None         -- no mesh shape
            "colliding"  -- mesh shape that participates in collision
            "visual"     -- mesh shape with has_shape_collision=False (render-only)
        """
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for i in range(num_boxes):
            b = builder.add_body(xform=wp.transform(wp.vec3(0.5 * float(i), 0.0, 0.5), wp.quat_identity()))
            builder.add_shape_box(body=b, hx=0.1, hy=0.1, hz=0.1)
        if mesh_mode is not None:
            cfg = None
            if mesh_mode == "visual":
                cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
            b = builder.add_body(xform=wp.transform(wp.vec3(-1.0, 0.0, 0.5), wp.quat_identity()))
            builder.add_shape_mesh(body=b, mesh=cls._cube_mesh(), cfg=cfg)
        return builder.finalize(device="cpu")

    def test_pair_aware_sum_of_budgets(self):
        """For sparse multi-world scenes the estimate is the exact per-pair budget sum."""
        model = self._build_multi_world_scene(num_worlds=100, boxes_per_world=2)
        # Per world: 1 box-box + 2 plane-box pairs = 3 pairs, all convex.
        self.assertEqual(model.shape_contact_pair_count, 300)
        # Locality cap (plane pairs full + 200 boxes * 20 neighbors * 5 // 2)
        # is far larger, so the provable pair sum binds.
        self.assertEqual(estimate_rigid_contact_capacity(model), 300 * _CONVEX_PAIR_MAX_CONTACTS)

    def test_pair_aware_mesh_pair_budget(self):
        """Mesh-involved pairs are budgeted with the contact-reduction per-pair cap."""
        model = self._build_multi_world_scene(num_worlds=50, boxes_per_world=2, mesh_per_world=True)
        # Per world: 1 box-box + 2 plane-box (convex) + 2 mesh-box + 1 plane-mesh (mesh) = 6 pairs.
        self.assertEqual(model.shape_contact_pair_count, 300)
        expected = 50 * (3 * _CONVEX_PAIR_MAX_CONTACTS + 3 * _MESH_PAIR_MAX_CONTACTS)
        self.assertEqual(estimate_rigid_contact_capacity(model), expected)

    def test_locality_cap_binds_for_dense_single_world(self):
        """Dense single-world pair graphs are capped by the spatial-locality estimate."""
        model = self._build_box_scene(num_boxes=25)
        # Pair sum: C(25,2)=300 box-box + 25 plane-box = 325 pairs * 5 = 1625.
        self.assertEqual(model.shape_contact_pair_count, 325)
        # Locality cap: plane pairs keep their budget (25 * 5) and the 25 boxes
        # are budgeted for 20 simultaneous neighbors each (halved per pair):
        # 125 + 25 * 5 * 20 // 2 = 1375 < 1625, so the cap binds.
        plane_term = 25 * _CONVEX_PAIR_MAX_CONTACTS
        locality = 25 * _CONVEX_PAIR_MAX_CONTACTS * 20 // 2
        self.assertEqual(estimate_rigid_contact_capacity(model), plane_term + locality)

    def test_visual_mesh_does_not_inflate_estimate(self):
        """A non-colliding (visual-only) mesh shape must not change the estimate."""
        model_plain = self._build_box_scene(num_boxes=25)
        model_visual = self._build_box_scene(num_boxes=25, mesh_mode="visual")
        # The visual mesh adds a shape but no contact pairs.
        self.assertEqual(model_visual.shape_count, model_plain.shape_count + 1)
        self.assertEqual(model_visual.shape_contact_pair_count, model_plain.shape_contact_pair_count)
        self.assertEqual(
            estimate_rigid_contact_capacity(model_visual),
            estimate_rigid_contact_capacity(model_plain),
        )

    def test_hydroelastic_pair_budget(self):
        """Hydro-hydro pairs get the anchor-augmented per-pair budget."""
        model = self._build_multi_world_scene(num_worlds=50, boxes_per_world=2)
        flags = model.shape_flags.numpy()
        flags |= int(ShapeFlags.HYDROELASTIC)
        model.shape_flags = wp.array(flags, dtype=wp.int32, device=model.device)
        # All 150 pairs become hydro-hydro: budgeted at MAX_CONTACTS_PER_PAIR
        # plus one anchor contact per normal bin; the locality cap (which also
        # uses the hydro budget per shape) stays larger, so the sum binds.
        self.assertEqual(estimate_rigid_contact_capacity(model), 150 * _HYDRO_PAIR_MAX_CONTACTS)

    def test_minimum_allocation_floor(self):
        """Small pair sums are floored at the 1000-contact minimum allocation."""
        model = self._build_box_scene(num_boxes=2)
        # 2 boxes + plane: 3 pairs * 5 = 15 -> floored to 1000.
        self.assertEqual(estimate_rigid_contact_capacity(model), 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
