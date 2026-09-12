# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host binding, cache epoch and unsupported-state controls for late publication."""

import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

from newton._src.sim import ModelFlags
from newton._src.solvers.feather_pgs import world_scan_owner as owner
from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS


def array(name="joint_q"):
    """Return a fresh contiguous descriptor without requiring a GPU."""
    annotation = owner.world_scan_publication.PublicationData.vars[name].type
    return wp.zeros((8,) * annotation.ndim, dtype=annotation.dtype, device="cpu")


def fixture():
    """Build distinct state/cache identities for host-only launch inspection."""
    model = SimpleNamespace(device=wp.get_device("cpu"), requires_grad=False, particle_count=0)
    for name in owner.MODEL_FIELDS:
        setattr(model, name, array(name))
    solver = SimpleNamespace(
        model=model,
        world_count=2,
        _fk_id_cache_enabled=True,
        _joint_world=None,
        _step=0,
        update_mass_matrix_interval=2,
        _global_inertia_stream=object(),
        angular_damping=0.03,
        _fk_id_cache_source_state=object(),
    )
    for name, source in owner.SOLVER_FIELDS.items():
        setattr(solver, source, array(name))
    cache = SimpleNamespace()
    aug = SimpleNamespace(joint_qdd=array())
    for name in owner.CACHE_FIELDS:
        setattr(cache, name, array(name))
        setattr(aug, name, array(name))
    solver._fk_id_cache = cache
    solver.articulation_origin, solver._body_inertia_terms = array("articulation_origin"), array("body_inertia_terms")
    states = [
        SimpleNamespace(
            joint_q=array(), joint_qd=array(), body_q=array("body_q"), body_qd=array("body_qd"), requires_grad=False
        )
        for _ in range(2)
    ]
    current = object.__new__(owner.WorldScanOwner)
    current.solver, current.plan, current.kernel = solver, object(), object()
    return current, states[0], aug, states[1]


class TestWorldScanOwner(unittest.TestCase):
    def test_bind_current_cache_and_refresh_epochs(self):
        """Bind every canonical output and retain the original next-refresh predicates."""
        current, state_in, aug, state_out = fixture()
        solver = current.solver
        for step, parallel, expected in ((0, True, (0, 0)), (1, True, (0, 1)), (1, False, (1, 0))):
            solver._step = step
            solver._global_inertia_stream = object() if parallel else None
            with patch.object(owner, "supported", return_value=True), patch.object(owner.wp, "launch_tiled") as launch:
                self.assertTrue(current.try_publish(state_in, aug, state_out, 0.02))
            data = launch.call_args.kwargs["inputs"][1]
            self.assertEqual((data.materialize_all_body_inertia, data.materialize_body_inertia_terms), expected)
            for name in owner.CACHE_FIELDS:
                self.assertIs(getattr(data, name), getattr(solver._fk_id_cache, name), name)
            self.assertIs(data.joint_q, state_in.joint_q)
            self.assertIs(data.joint_qd, state_in.joint_qd)
            self.assertIs(data.joint_q_new, state_out.joint_q)
            self.assertIs(data.joint_qd_new, state_out.joint_qd)
            self.assertIs(data.joint_qdd, aug.joint_qdd)
            self.assertIs(data.body_q, state_out.body_q)
            self.assertIs(data.body_qd, state_out.body_qd)
            self.assertIs(solver._fk_id_cache_source_state, state_out)
            self.assertEqual(launch.call_args.kwargs["block_dim"], 32)

    def test_without_separate_cache_uses_original_augmented_fields(self):
        """Preserve the original augmented-state fallback when no cache object exists."""
        current, state_in, aug, state_out = fixture()
        current.solver._fk_id_cache = None
        with patch.object(owner, "supported", return_value=True), patch.object(owner.wp, "launch_tiled") as launch:
            self.assertTrue(current.try_publish(state_in, aug, state_out, 0.02))
        data = launch.call_args.kwargs["inputs"][1]
        for name in owner.CACHE_FIELDS:
            expected = getattr(aug, name)
            if name == "articulation_origin":
                expected = current.solver.articulation_origin
            elif name == "body_inertia_terms":
                expected = current.solver._body_inertia_terms
            self.assertIs(getattr(data, name), expected, name)

    def test_alias_noncontiguous_and_gradient_leave_original_owner(self):
        """Reject source/output overlaps including partial body/cache aliases before launching."""
        for case in ("same_q", "partial_q", "body_to_cache", "noncontiguous", "gradient", "unsupported"):
            current, state_in, aug, state_out = fixture()
            if case == "same_q":
                state_out.joint_q = state_in.joint_q
            elif case == "partial_q":
                state_out.joint_q = state_in.joint_q[2:6]
            elif case == "body_to_cache":
                current.solver._fk_id_cache.body_a_s = state_in.body_qd
            elif case == "noncontiguous":
                state_in.joint_q = wp.zeros((8, 2), dtype=float, device="cpu")[:, 0]
            elif case == "gradient":
                state_out.requires_grad = True
            before = current.solver._fk_id_cache_source_state
            with (
                patch.object(owner, "supported", return_value=case != "unsupported"),
                patch.object(owner.wp, "launch_tiled") as launch,
            ):
                self.assertFalse(current.try_publish(state_in, aug, state_out, 0.02), case)
            launch.assert_not_called()
            self.assertIs(current.solver._fk_id_cache_source_state, before)

    def test_failed_launch_does_not_publish_source_identity(self):
        """Leave the prior host cache identity intact if the complete launch raises."""
        current, state_in, aug, state_out = fixture()
        before = current.solver._fk_id_cache_source_state
        with (
            patch.object(owner, "supported", return_value=True),
            patch.object(owner.wp, "launch_tiled", side_effect=RuntimeError("injected")),
            self.assertRaisesRegex(RuntimeError, "injected"),
        ):
            current.try_publish(state_in, aug, state_out, 0.02)
        self.assertIs(current.solver._fk_id_cache_source_state, before)

    def test_numeric_notifications_do_not_read_back_static_arrays(self):
        """Keep numeric reset notifications cheap and reject changed static ownership."""
        current, _, _, _ = fixture()
        current.model_plan_values = {}
        for name in owner.PLAN_FIELDS:
            setattr(current.solver.model, name, Mock())
            getattr(current.solver.model, name).numpy.return_value = np.array([1, 2])
            current.model_plan_values[name] = np.array([1, 2])
        numeric = (
            ModelFlags.JOINT_DOF_PROPERTIES
            | ModelFlags.BODY_INERTIAL_PROPERTIES
            | ModelFlags.SHAPE_PROPERTIES
            | ModelFlags.MODEL_PROPERTIES
        )
        for flags in (ModelFlags.MODEL_PROPERTIES, numeric):
            current.validate_notification(flags)
        for name in owner.PLAN_FIELDS:
            getattr(current.solver.model, name).numpy.assert_not_called()
        for flags in (0, ModelFlags.JOINT_PROPERTIES, ModelFlags.BODY_PROPERTIES, -1):
            current.validate_notification(flags)
        current.solver.model.joint_parent.numpy.return_value = np.array([2, 1])
        with self.assertRaisesRegex(RuntimeError, "reconstruct"):
            current.validate_notification(ModelFlags.JOINT_PROPERTIES)

    def test_unsupported_constructor_and_joint_world_composition_fall_back(self):
        """Keep unsupported topology and the unvalidated light composition on the original owner."""
        current, _, _, _ = fixture()
        solver = current.solver
        with patch.object(owner.simple_world, "_supported", return_value=True):
            self.assertFalse(owner.supported(solver))  # CPU is not a runtime target.
        with (
            patch.object(owner, "supported", return_value=True),
            patch.object(owner.kuka_joint_world, "bind_plan", side_effect=ValueError("unsupported")),
        ):
            self.assertIsNone(owner.create_owner(solver))
        solver.model.device = SimpleNamespace(is_cuda=True)
        solver._joint_world = object()
        with patch.object(owner.simple_world, "_supported", return_value=True):
            self.assertFalse(owner.supported(solver))

    def test_solver_dispatch_replaces_the_complete_late_owner(self):
        """Require the opt-in path before both generalized integration and original FK dispatch."""
        source = inspect.getsource(SolverFeatherPGS)
        self.assertIn("FEATHER_PGS_WORLD_SCAN_PUBLICATION", source)
        self.assertIn("self._world_scan_publication.validate_notification(flags)", source)
        tree = ast.parse(textwrap.dedent(source))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertEqual(
            sum(isinstance(node.func, ast.Attribute) and node.func.attr == "try_publish" for node in calls), 1
        )
        marker = source.index('with wp.ScopedTimer("S7_Integrate"')
        segment = source[marker : source.index("# ── MF warm-start", marker)]
        self.assertLess(segment.index("try_publish"), segment.index("finish_active"))
        self.assertLess(segment.index("try_publish"), segment.index("_stage6_update_qdd"))


if __name__ == "__main__":
    unittest.main()
