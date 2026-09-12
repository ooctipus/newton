# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Portable regressions for early ownership, source reuse and lifecycle guards."""

import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import early_franka as early
from newton._src.solvers.feather_pgs import kernels


def cohort():
    """Build a partial four-world cohort with primary, free and prescribed bodies."""
    data = early.CohortData()
    for name in ("incidence", "early_owner", "early_count", "zero_mf", "late_owner", "early_arts"):
        setattr(data, name, wp.zeros(4, dtype=int, device="cpu"))
    data.invalid_raw = wp.zeros(1, dtype=int, device="cpu")
    data.counts = wp.zeros(2, dtype=int, device="cpu")
    data.art_mask = wp.zeros(12, dtype=int, device="cpu")
    data.late_arts = wp.full(12, -123, dtype=int, device="cpu")
    data.primary = wp.array([0, 3, 6, 9], dtype=int, device="cpu")
    data.art_world = wp.array(np.repeat(np.arange(4), 3), dtype=int, device="cpu")
    data.body_art = wp.array(np.arange(12), dtype=int, device="cpu")
    return data


class TestEarlyFrankaGuards(unittest.TestCase):
    def test_publication_equations_are_original(self):
        """Recover every original statement after removing the index-only adapter."""
        names = (
            "update_qdd_from_velocity",
            "integrate_generalized_joints",
            "eval_rigid_fk_kinematics",
            "finalize_body_dynamics",
            "prepare_world_impulses",
        )
        for name in names:
            original = ast.parse(textwrap.dedent(inspect.getsource(getattr(kernels, name).func))).body[0]
            for selected in (False,) if name == "prepare_world_impulses" else (False, True):
                adapted = ast.parse(early.publication_source(name, selected)).body[0]

                class RestoreIndex(ast.NodeTransformer):
                    def visit_Name(self, node):
                        if node.id == "mapped_index":
                            return ast.Call(
                                func=ast.Attribute(value=ast.Name(id="wp", ctx=ast.Load()), attr="tid", ctx=ast.Load()),
                                args=[],
                                keywords=[],
                            )
                        return node

                tail = ast.Module(body=adapted.body[-len(original.body) :], type_ignores=[])
                tail = RestoreIndex().visit(tail)
                expected = ast.Module(body=original.body, type_ignores=[])
                self.assertEqual(ast.dump(tail), ast.dump(expected), (name, selected))

    def test_raw_incidence_and_exact_partition(self):
        """Reject primary raw endpoints and MF/long rows without rejecting a free-only raw pair."""
        data = cohort()
        count = wp.array([2], dtype=int, device="cpu")
        shapes = wp.array([0, 4], dtype=int, device="cpu")  # primary0; free body of world1
        other = wp.array([-1, -1], dtype=int, device="cpu")
        shape_body = wp.array(np.arange(12), dtype=int, device="cpu")
        wp.launch(early.mark_raw_incidence, dim=7, inputs=[count, shapes, other, shape_body, 7, data], device="cpu")
        np.testing.assert_array_equal(data.incidence.numpy(), [1, 0, 0, 0])
        rows = wp.array([6, 7, 10, 6], dtype=int, device="cpu")
        bounds = wp.array(np.array([[6, 6], [7, 7], [10, 10], [6, 6]]), dtype=int, device="cpu")
        mf = wp.array([0, 0, 0, 1], dtype=int, device="cpu")
        wp.launch(early.classify_early, dim=4, inputs=[rows, bounds, mf, data], device="cpu")
        wp.launch(early.compact_publication, dim=12, inputs=[data], device="cpu")
        np.testing.assert_array_equal(data.early_owner.numpy(), [0, 1, 0, 0])
        np.testing.assert_array_equal(data.counts.numpy(), [1, 11])
        self.assertEqual(data.early_arts.numpy()[0], 3)
        self.assertEqual(set(data.late_arts.numpy()[:11]), set(range(12)) - {3})
        self.assertEqual(data.late_arts.numpy()[11], -123)
        # An incomplete raw list never certifies absence, even on unrelated worlds.
        count.assign(np.array([3], dtype=np.int32))
        wp.launch(early.mark_raw_incidence, dim=7, inputs=[count, shapes, other, shape_body, 7, data], device="cpu")
        wp.launch(early.classify_early, dim=4, inputs=[rows, bounds, mf, data], device="cpu")
        np.testing.assert_array_equal(data.early_owner.numpy(), 0)

    def test_late_impulse_and_owner_protection(self):
        """Retain nonzero early impulses and remove exactly one original late solve."""
        data = cohort()
        data.early_owner.assign(np.array([1, 0, 1, 0], dtype=np.int32))
        owner = wp.array([1, 2, 1, 3], dtype=int, device="cpu")
        wp.launch(early.prepare_late_owner, dim=4, inputs=[owner, data], device="cpu")
        np.testing.assert_array_equal(owner.numpy(), [1, 2, 1, 3])
        np.testing.assert_array_equal(data.late_owner.numpy(), [0, 2, 0, 3])
        rows = wp.array([6, 3, 8, 1], dtype=int, device="cpu")
        impulses = wp.full((4, 40), 2.5, dtype=float, device="cpu")
        wp.launch(
            early.get_publication_kernel("prepare_world_impulses", False),
            dim=4,
            inputs=[rows, 40, 0, impulses, data],
            device="cpu",
        )
        values = impulses.numpy().copy()
        np.testing.assert_array_equal(values[[0, 2]], 2.5)
        np.testing.assert_array_equal(values[1, :3], 0)
        np.testing.assert_array_equal(values[1, 3:], 2.5)
        wp.launch(kernels.prepare_world_impulses, dim=4, inputs=[rows, 40, 0, impulses], device="cpu")
        self.assertFalse(np.array_equal(impulses.numpy()[0, :6], values[0, :6]))

    def test_begin_waits_and_rejects_alias(self):
        """Keep prior work behind its event and reject reused state storage."""
        owner = object.__new__(early.EarlyFranka)
        owner.active = True
        owner.done = object()
        owner.solver = SimpleNamespace(model=SimpleNamespace(device="cpu"))
        state = SimpleNamespace(
            **{name: wp.zeros(4, dtype=float, device="cpu") for name in ("joint_q", "joint_qd", "body_q", "body_qd")}
        )
        stream = Mock()
        with patch.object(wp, "get_stream", return_value=stream):
            with self.assertRaisesRegex(ValueError, "non-overlapping"):
                owner.begin(state, state)
        stream.wait_event.assert_called_once_with(owner.done)

    def test_join_ends_event_epoch(self):
        """Do not carry a completed graph's done event into the next capture."""
        owner = object.__new__(early.EarlyFranka)
        owner.active = True
        owner.done = object()
        owner.solver = SimpleNamespace(model=SimpleNamespace(device="cpu"))
        state = object()
        stream = Mock()
        with patch.object(wp, "get_stream", return_value=stream):
            owner.join(state)
            self.assertFalse(owner.active)
            self.assertIs(owner.solver._fk_id_cache_source_state, state)
            owner.wait()
        stream.wait_event.assert_called_once_with(owner.done)


if __name__ == "__main__":
    unittest.main()
