# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise direct terrain ownership without changing the original contact law."""

import ast
import inspect
import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry import heightfield_cells, heightfield_direct
from newton._src.geometry.narrow_phase import NarrowPhase
from newton.tests import test_heightfield_cell_reject as original


class TestHeightfieldDirect(unittest.TestCase):
    """Keep explicit feature selection and unsupported original fallback visible."""

    device = "cpu"
    _assert_geometry_coverage = original.TestHeightfieldCellReject._assert_geometry_coverage

    def compare(self, model, **kwargs):
        """Compare full current geometry and logical demand against the cell stream."""
        with patch.dict(os.environ, NEWTON_HEIGHTFIELD_DIRECT="0"):
            before, _, a = original.collide(model, True, **kwargs)
        with patch.dict(os.environ, NEWTON_HEIGHTFIELD_DIRECT="1"):
            after, _, b = original.collide(model, True, **kwargs)
        self.assertTrue(after.narrow_phase._heightfield_direct)
        self.assertEqual(
            int(before.narrow_phase.triangle_pairs_count.numpy()[0]),
            int(after.narrow_phase.triangle_pairs_count.numpy()[0]),
        )
        self.assertEqual(bool(len(a[0])), bool(len(b[0])))
        if len(a[0]):
            self._assert_geometry_coverage(a, b)
        return after

    def test_constructor_requires_explicit_cell_mode_and_mixed_falls_back(self):
        """Reject a partial recipe and retain the old mixed mesh owner."""
        kwargs = {
            "max_candidate_pairs": 4,
            "max_triangle_pairs": 16,
            "device": self.device,
            "has_heightfields": True,
            "has_meshes": False,
        }
        with patch.dict(os.environ, NEWTON_HEIGHTFIELD_DIRECT="1", NEWTON_HEIGHTFIELD_CELL_REJECT="1"):
            selected = NarrowPhase(**kwargs)
            mixed = NarrowPhase(**{**kwargs, "has_meshes": True})
        self.assertTrue(selected._heightfield_direct)
        self.assertFalse(mixed._heightfield_direct)
        with patch.dict(os.environ, NEWTON_HEIGHTFIELD_DIRECT="0", NEWTON_HEIGHTFIELD_CELL_REJECT="1"):
            self.assertFalse(NarrowPhase(**kwargs)._heightfield_direct)
        for direct, cell in (("yes", "1"), ("1", "0")):
            with patch.dict(os.environ, NEWTON_HEIGHTFIELD_DIRECT=direct, NEWTON_HEIGHTFIELD_CELL_REJECT=cell):
                with self.assertRaises(ValueError):
                    NarrowPhase(**kwargs)

    def test_original_range_and_padding_recovery(self):
        """Reuse the exact admitted current geometry operations, not a new bound."""

        def assignments(module, name, last):
            tree = ast.parse(inspect.getsource(module))
            fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            output = []
            for node in fn.body:
                if isinstance(node, ast.Assign):
                    output.append(ast.dump(node))
                    if isinstance(node.targets[0], ast.Name) and node.targets[0].id == last:
                        break
            return output

        self.assertEqual(
            assignments(heightfield_cells, "_heightfield_cell_midphase", "can_reject"),
            assignments(heightfield_direct, "_current_cell_range", "can_reject"),
        )

    def test_current_geometry_shell_and_reversed_endpoints(self):
        """Preserve near, below, rotated, scaled, separated and speculative queries."""
        for kwargs in (
            {},
            {"z": 0.015},
            {"z": -0.02, "reverse": True},
            {"z": 0.005, "rotation": wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.45), "reverse": True},
            {"z": 0.12, "margin": 0.025, "gap": 0.05, "heightfield_scale": (1.3, 0.7, 2.0)},
        ):
            with self.subTest(kwargs=kwargs):
                self.compare(original.make_model(self.device, **kwargs))
        self.compare(original.make_model(self.device, z=0.15), speculative=True)

    def test_retired_triangle_stream_reduction_and_current_height(self):
        """Keep logical counts, current heights and complete public contacts without old producers."""
        model = original.make_model(self.device, z=0.01)
        pipeline = self.compare(model, reduce_contacts=True)
        triangles = pipeline.narrow_phase.triangle_pairs
        triangles.fill_(-9876)
        original_launch = wp.launch
        with patch.object(wp, "launch", wraps=original_launch) as launches:
            pipeline.collide(model.state(), pipeline.contacts())
        pipeline.narrow_phase.check_buffer_capacity()
        np.testing.assert_array_equal(triangles.numpy(), -9876)
        retired = (
            "heightfield_cell_overlaps_kernel",
            "narrow_phase_find_mesh_triangle_overlaps_kernel",
            "mesh_triangle_contacts_to_reducer_kernel",
        )
        names = [call.kwargs["kernel"].key for call in launches.call_args_list if "kernel" in call.kwargs]
        self.assertFalse(any(name.endswith(key) for name in names for key in retired), names)
        heights = model.heightfield_elevations.numpy()
        heights[35:45] = 0.15
        model.heightfield_elevations.assign(heights)
        self.compare(model, reduce_contacts=False)

    def test_sphere_capsule_query_inputs_do_not_leak_between_cells(self):
        """Minkowski radius treatment remains local to each original query call."""
        for primitive in ("sphere", "capsule"):
            builder = newton.ModelBuilder()
            values = np.zeros((9, 9), dtype=np.float32)
            values[0, 0] = 0.3
            field = newton.Heightfield(values, nrow=9, ncol=9, hx=0.3, hy=0.3)
            cfg = builder.ShapeConfig(margin=0.01, gap=0.01)
            builder.add_shape_heightfield(heightfield=field, cfg=cfg)
            body = builder.add_body(xform=wp.transform((0, 0, 0.05), wp.quat_identity()))
            if primitive == "sphere":
                builder.add_shape_sphere(body=body, radius=0.05, cfg=cfg)
            else:
                builder.add_shape_capsule(body=body, radius=0.05, half_height=0.02, cfg=cfg)
            with self.subTest(primitive=primitive):
                self.compare(builder.finalize(device=self.device))

    def test_logical_triangle_overflow_remains_observable(self):
        """No query is silently clipped to the retired stream's capacity."""
        model = original.make_model(self.device, z=0.01)
        with patch.dict(os.environ, NEWTON_HEIGHTFIELD_DIRECT="1", NEWTON_HEIGHTFIELD_CELL_REJECT="1"):
            pipeline = newton.CollisionPipeline(
                model, reduce_contacts=False, rigid_contact_max=2048, max_triangle_pairs=1
            )
        pipeline.collide(model.state(), pipeline.contacts())
        self.assertGreater(int(pipeline.narrow_phase.triangle_pairs_count.numpy()[0]), 1)
        self.assertTrue(pipeline.narrow_phase.buffer_capacity_status()["triangle"])
        with self.assertRaisesRegex(RuntimeError, "triangle"):
            pipeline.narrow_phase.check_buffer_capacity()


if __name__ == "__main__":
    unittest.main()
