# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise the opt-in heightfield cell rejection without changing contact laws."""

import ast
import inspect
import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry import heightfield_cells
from newton._src.geometry.heightfield_cells import _cell_is_above
from newton._src.geometry.narrow_phase import NarrowPhase
from newton._src.utils import heightfield as original_heightfield
from newton._src.utils.heightfield import HeightfieldData


@wp.kernel
def check_cells(lower: wp.array[float], heights: wp.array[float], result: wp.array[int]):
    """Exercise finite/top-surface admission without full pipeline setup."""
    i = wp.tid()
    hfd = HeightfieldData()
    hfd.data_offset = i * 4
    hfd.nrow = 2
    hfd.ncol = 2
    hfd.hx = 1.0
    hfd.hy = 1.0
    hfd.min_z = 0.0
    hfd.max_z = 1.0
    result[i] = int(_cell_is_above(lower[i], 0.001, hfd, heights, 0, 0))


def make_model(
    device="cpu",
    *,
    z=0.2,
    heights=None,
    rotation=None,
    reverse=False,
    margin=0.01,
    gap=0.01,
    heightfield_scale=(1.0, 1.0, 1.0),
):
    """Create one nonuniform cuboid and the exact finite heightfield surface."""
    builder = newton.ModelBuilder()
    config = builder.ShapeConfig(margin=margin, gap=gap)
    values = np.zeros((9, 9), dtype=np.float32) if heights is None else np.asarray(heights, dtype=np.float32)
    if heights is None:
        # A distant hill keeps the whole terrain AABB overlapping an airborne
        # cuboid; only current per-cell geometry can reject this pair's region.
        values[0, 0] = 1.0
    heightfield = newton.Heightfield(data=values, nrow=9, ncol=9, hx=0.4, hy=0.4)
    q = wp.quat_identity() if rotation is None else rotation
    terrain_x = wp.transform((20.0, -40.0, 0.0), q)
    body_x = wp.transform_multiply(terrain_x, wp.transform((0.0, 0.0, z), wp.quat_identity()))

    def terrain():
        builder.add_shape_heightfield(heightfield=heightfield, xform=terrain_x, cfg=config, scale=heightfield_scale)

    def box():
        body = builder.add_body(xform=body_x)
        builder.add_shape_box(body=body, hx=0.10155461, hy=0.03273462, hz=0.00925394, cfg=config)

    if reverse:
        box()
        terrain()
    else:
        terrain()
        box()
    return builder.finalize(device=device)


def collide(model, enabled, *, reduce_contacts=False, speculative=False):
    """Run the complete original collision pipeline with one constructor flag."""
    with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_CELL_REJECT": str(int(enabled))}):
        pipeline = newton.CollisionPipeline(
            model,
            reduce_contacts=reduce_contacts,
            rigid_contact_max=2048,
            max_triangle_pairs=4096,
            speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(max_speculative_extension=0.5)
            if speculative
            else None,
        )
    contacts = pipeline.contacts()
    state = model.state()
    if speculative:
        velocity = np.zeros((model.body_count, 6), dtype=np.float32)
        velocity[:, 2] = -20.0
        state.body_qd.assign(velocity)
    pipeline.collide(state, contacts, dt=0.01 if speculative else None)
    pipeline.narrow_phase.check_buffer_capacity()
    count = int(contacts.rigid_contact_count.numpy()[0])
    distance = wp.empty(2048, dtype=float, device=model.device)
    point = wp.empty(2048, dtype=wp.vec3, device=model.device)
    newton.eval_rigid_contact_kinematics(model, state, contacts, out_distance=distance, out_point0_world=point)
    return (
        pipeline,
        contacts,
        (distance.numpy()[:count], contacts.rigid_contact_normal.numpy()[:count], point.numpy()[:count]),
    )


class TestHeightfieldCellReject(unittest.TestCase):
    """Keep constructor admission and complete surface geometry independently visible."""

    device = "cpu"

    def _assert_geometry_coverage(self, a, b):
        """Require bidirectional full-contact geometry coverage, independent of ordering."""
        self.assertTrue(all(np.isfinite(value).all() for values in (a, b) for value in values))
        for source, target in ((a, b), (b, a)):
            distance = np.abs(source[0][:, None] - target[0][None, :])
            normal = np.linalg.norm(source[1][:, None] - target[1][None, :], axis=2)
            position = np.linalg.norm(source[2][:, None] - target[2][None, :], axis=2)
            score = np.maximum.reduce((distance / 2e-4, normal / 2e-3, position / 2e-4))
            self.assertLess(float(np.max(np.min(score, axis=1))), 1.0)

    def test_reject_above_cells_and_keep_default(self):
        """Remove separated triangle work only under explicit constructor admission."""
        model = make_model(self.device)
        old, _, old_geometry = collide(model, False)
        new, _, new_geometry = collide(model, True)
        self.assertFalse(old.narrow_phase._heightfield_cell_reject)
        self.assertTrue(new.narrow_phase._heightfield_cell_reject)
        self.assertGreater(int(old.narrow_phase.triangle_pairs_count.numpy()[0]), 0)
        self.assertEqual(int(new.narrow_phase.triangle_pairs_count.numpy()[0]), 0)
        self.assertEqual(len(old_geometry[0]), 0)
        self.assertEqual(len(new_geometry[0]), 0)

    def test_near_below_rotated_and_reversed_keep_physical_contacts(self):
        """Retain surface distances and normals across penetration and transformed endpoints."""
        for z, rotation, reverse in (
            (0.015, None, False),
            (-0.02, None, True),
            (0.005, wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.45), True),
        ):
            with self.subTest(z=z, reverse=reverse):
                model = make_model(self.device, z=z, rotation=rotation, reverse=reverse)
                old, _, a = collide(model, False)
                new, _, b = collide(model, True)
                self.assertGreater(len(a[0]), 0)
                self.assertGreater(len(b[0]), 0)
                self._assert_geometry_coverage(a, b)
                self.assertAlmostEqual(float(a[0].min()), float(b[0].min()), delta=2e-5)
                self.assertLess(float(np.min(b[0])), 0.05)
                self.assertEqual(
                    int(old.narrow_phase.triangle_pairs_count.numpy()[0]),
                    int(new.narrow_phase.triangle_pairs_count.numpy()[0]),
                )
                # No dependence on scheduling/contact ordering in this geometry control.
                self.assertLess(
                    float(np.max(np.min(np.linalg.norm(b[1][:, None] - a[1][None, :], axis=2), axis=1))), 2e-4
                )

    def test_threshold_nonflat_nonfinite_and_current_height(self):
        """Retain ties, negative-depth prisms and unknown corner heights."""
        lower = wp.array([0.001, 0.002, -0.5, 0.5, 0.5, 0.5], dtype=float, device=self.device)
        values = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.6], [0, np.nan, 0, 0], [0, 0, 0, 0]],
            dtype=np.float32,
        )
        heights = wp.array(values.reshape(-1), dtype=float, device=self.device)
        out = wp.empty(6, dtype=int, device=self.device)
        wp.launch(check_cells, dim=6, inputs=[lower, heights, out], device=self.device)
        np.testing.assert_array_equal(out.numpy(), [0, 1, 0, 0, 0, 1])
        values[5, 3] = 0.7
        heights.assign(values.reshape(-1))
        wp.launch(check_cells, dim=6, inputs=[lower, heights, out], device=self.device)
        self.assertEqual(int(out.numpy()[5]), 0)

    def test_speculative_current_search_gap_is_retained(self):
        """Do not cull an approaching shape using the smaller base contact shell."""
        model = make_model(self.device, z=0.15)
        old, _, a = collide(model, False, speculative=True)
        new, _, b = collide(model, True, speculative=True)
        self.assertGreater(float(new._shape_search_gap.numpy().max()), float(model.shape_gap.numpy().max()))
        self.assertGreater(len(a[0]), 0)
        self.assertGreater(len(b[0]), 0)
        self._assert_geometry_coverage(a, b)
        self.assertEqual(
            int(old.narrow_phase.triangle_pairs_count.numpy()[0]), int(new.narrow_phase.triangle_pairs_count.numpy()[0])
        )
        self.assertAlmostEqual(float(a[0].min()), float(b[0].min()), delta=2e-5)

    def test_mixed_meshes_fallback_and_invalid_flag(self):
        """Keep the original mixed-scene owner and reject misspelled feature values."""
        with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_CELL_REJECT": "1"}):
            narrow = NarrowPhase(
                max_candidate_pairs=4,
                max_triangle_pairs=16,
                device=self.device,
                has_meshes=True,
                has_heightfields=True,
                reduce_contacts=False,
            )
        self.assertTrue(narrow._heightfield_cell_reject_requested)
        self.assertFalse(narrow._heightfield_cell_reject)
        with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_CELL_REJECT": "yes"}), self.assertRaises(ValueError):
            newton.CollisionPipeline(make_model(self.device))

    def test_scaled_heightfield_and_full_margin_gap(self):
        """Use baked nonuniform heightfield scale and both endpoint detection shells."""
        model = make_model(self.device, z=0.12, margin=0.025, gap=0.05, heightfield_scale=(1.3, 0.7, 2.0))
        old, _, a = collide(model, False)
        new, _, b = collide(model, True)
        self.assertGreater(len(a[0]), 0)
        self.assertGreater(len(b[0]), 0)
        self._assert_geometry_coverage(a, b)
        self.assertEqual(
            int(old.narrow_phase.triangle_pairs_count.numpy()[0]), int(new.narrow_phase.triangle_pairs_count.numpy()[0])
        )
        self.assertAlmostEqual(float(a[0].min()), float(b[0].min()), delta=2e-5)

    def test_original_aabb_and_emission_source_recovery(self):
        """Preserve every original range operation and surviving triangle write."""

        def function(module, name):
            return next(
                n
                for n in ast.parse(inspect.getsource(module)).body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )

        old = function(original_heightfield, "heightfield_vs_convex_midphase")
        new = function(heightfield_cells, "_heightfield_cell_midphase")

        def bounds(fn):
            statements = []
            for node in fn.body:
                if isinstance(node, ast.Assign):
                    statements.append(ast.dump(node))
                    if isinstance(node.targets[0], ast.Name) and node.targets[0].id == "cols":
                        break
            return statements

        self.assertEqual(bounds(old), bounds(new))
        old_emit = old.body[-1].body[0].body[0]
        new_emit = new.body[-1].body[0].body[-1]
        self.assertEqual(ast.dump(old_emit), ast.dump(new_emit))


if __name__ == "__main__":
    unittest.main()
