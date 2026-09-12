# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent current-world publication and lifecycle controls for the light owner."""

import ast
import inspect
import os
import textwrap
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

from newton._src.sim import ModelFlags
from newton._src.solvers.feather_pgs import kernels
from newton._src.solvers.feather_pgs import kuka_joint_owner as owner
from newton._src.solvers.feather_pgs import kuka_joint_world as light
from tools.fpgs_bench.test_kuka_joint_world import plan_inputs, snapshots

DEVICE = os.environ.get("FPGS_TEST_DEVICE", "cpu")
CHAIN = tuple(getattr(kernels, name) for name in owner.PUBLICATION_WIDTH)
OUTPUT_KEYS = {
    "v_new": "post_solve_v_out",
    "joint_qdd": "post3_aug_joint_qdd",
    "joint_q_new": "pre_state_joint_q",
    "joint_qd_new": "pre_state_joint_qd",
}


def bind(snapshot, device):
    """Bind actual current inputs and separate writable publication arrays."""
    values = {"inv_dt": 240.0, "dt": 1.0 / 240.0, "angular_damping": 0.0}
    aliases = {
        "joint_q": "pre_state_joint_q",
        "joint_qd": "pre_state_joint_qd",
        "kinematic_dof_mask": "post3_solver__kinematic_dof_mask",
        "kinematic_joint_mask": "post3_solver__kinematic_joint_mask",
        "free_root_joint_indices": "post3_solver__free_root_joint_indices",
    }
    for kernel in CHAIN:
        for arg in kernel.adj.args:
            name = arg.label
            if name in values:
                continue
            key = OUTPUT_KEYS.get(name, aliases.get(name))
            if key is None:
                key = next(key for key in ("full_model_" + name, "post3_solver_" + name) if key in snapshot)
            values[name] = wp.array(snapshot[key], dtype=arg.type.dtype, device=device)
    host = light.build_plan(**plan_inputs(snapshot))
    model = SimpleNamespace(device=wp.get_device(device))
    for name in (
        "joint_type",
        "joint_parent",
        "joint_child",
        "joint_q_start",
        "joint_qd_start",
        "joint_dof_dim",
        "body_com",
        "joint_X_c",
    ):
        setattr(model, name, values[name])
    state_in = SimpleNamespace(joint_q=values["joint_q"], joint_qd=values["joint_qd"])
    state_aug = SimpleNamespace(joint_qdd=values["joint_qdd"])
    state_out = SimpleNamespace(joint_q=values["joint_q_new"], joint_qd=values["joint_qd_new"])
    solver = SimpleNamespace(
        model=model,
        world_count=512,
        v_out=values["v_new"],
        _kinematic_dof_mask=values["kinematic_dof_mask"],
        _kinematic_joint_mask=values["kinematic_joint_mask"],
        _free_root_joint_indices=values["free_root_joint_indices"],
        angular_damping=values["angular_damping"],
        _stage7_update_kinematics=Mock(),
        integrate_particles=Mock(),
    )
    current = object.__new__(owner.JointWorldOwner)
    current.solver, current.plan = solver, host.device_data(device)
    current.output = SimpleNamespace(
        active_worlds=wp.empty(512, dtype=int, device=device),
        active_count=wp.zeros(1, dtype=int, device=device),
    )
    current.publication = {name: owner.get_publication_kernel(name) for name in owner.PUBLICATION_WIDTH}
    dof_world = np.empty(512 * 35, dtype=np.int32)
    dof_world[host.dof_ids] = np.arange(512)[:, None]
    joint_world = snapshot["post3_solver_art_to_world"][snapshot["full_model_joint_articulation"]]
    q_world = np.repeat(joint_world, np.diff(snapshot["full_model_joint_q_start"]))
    return SimpleNamespace(
        values=values,
        owner=current,
        host=host,
        device=device,
        state_in=state_in,
        state_aug=state_aug,
        state_out=state_out,
        worlds={"v_new": dof_world, "joint_qdd": dof_world, "joint_q_new": q_world, "joint_qd_new": dof_world},
    )


def seed(bundle, *, solved=False):
    """Reuse the same array identities while poisoning unpublished output fields."""
    values = bundle.values
    if not solved:
        values["joint_qdd"].fill_(37.0)
        values["joint_q_new"].fill_(-43.0)
        values["joint_qd_new"].fill_(51.0)
    return {name: values[name].numpy().copy() for name in OUTPUT_KEYS}


def set_active(bundle, ids):
    """Write a current unordered queue and poison every unused capacity slot."""
    ids = np.asarray(ids, dtype=np.int32)
    queue = np.full(512, -71, dtype=np.int32)
    queue[: len(ids)] = ids
    bundle.owner.output.active_worlds.assign(queue)
    bundle.owner.output.active_count.assign(np.asarray([len(ids)], dtype=np.int32))


def original(bundle):
    """Run the exact original generalized equation chain on all physical owners."""
    dims = (512 * 35, 512 * 2, 512 * 32)
    for kernel, dim in zip(CHAIN, dims, strict=True):
        wp.launch(kernel, dim=dim, inputs=[bundle.values[a.label] for a in kernel.adj.args], device=bundle.device)


def candidate(bundle):
    """Exercise the actual host publication method, including its late FK handoff."""
    bundle.owner.finish_active(bundle.state_in, bundle.state_aug, bundle.state_out, 1.0 / 240.0)


def check(test, bundle, reference, before, ids):
    """Compare active math to original and preserve every excluded ZERO output exactly."""
    world_mask = np.zeros(512, dtype=bool)
    world_mask[np.asarray(ids, dtype=int)] = True
    for name in OUTPUT_KEYS:
        selected = world_mask[bundle.worlds[name]]
        actual, expected = bundle.values[name].numpy(), reference.values[name].numpy()
        np.testing.assert_allclose(actual[selected], expected[selected], atol=3e-6, rtol=3e-6, err_msg=name)
        np.testing.assert_array_equal(actual[~selected], before[name][~selected], err_msg=f"ZERO overwrite: {name}")
        test.assertTrue(np.isfinite(actual).all(), name)


class TestKukaJointOwner(unittest.TestCase):
    def test_unsupported_topology_keeps_original_owner(self):
        """Keep a recipe-compatible but differently shaped model on the original solver."""
        solver = SimpleNamespace(
            _simple_world_classifier=SimpleNamespace(enabled=True),
            model=SimpleNamespace(particle_count=0, device="cpu"),
            _fk_id_cache_enabled=True,
            dense_max_constraints=192,
            _has_rigid_body_velocity_limits=False,
        )
        with (
            patch.object(owner.simple_world, "_supported", return_value=True),
            patch.object(light, "bind_plan", side_effect=ValueError("unsupported topology")),
        ):
            self.assertIsNone(owner.create_owner(solver))

    def test_original_publication_statement_identity(self):
        """Recover every original equation and guard the queue before any mapped read."""
        for name in owner.PUBLICATION_WIDTH:
            source = ast.parse(textwrap.dedent(inspect.getsource(getattr(kernels, name).func))).body[0]
            adapted = ast.parse(owner.publication_source(name)).body[0]

            class Restore(ast.NodeTransformer):
                def visit_Name(self, node):
                    if node.id == "mapped_index":
                        return ast.Call(
                            func=ast.Attribute(value=ast.Name(id="wp", ctx=ast.Load()), attr="tid", ctx=ast.Load()),
                            args=[],
                            keywords=[],
                        )
                    return node

            restored = Restore().visit(ast.Module(body=adapted.body[-len(source.body) :], type_ignores=[]))
            self.assertEqual(ast.dump(restored), ast.dump(ast.Module(body=source.body, type_ignores=[])))
            prelude = adapted.body[: -len(source.body)]
            self.assertIsInstance(prelude[2], ast.If)
            self.assertIsInstance(prelude[2].body[0], ast.Return)
            self.assertLess(
                owner.publication_source(name).index("if slot >= active_count[0]"),
                owner.publication_source(name).index("world = active_worlds[slot]"),
            )

    def test_actual_current512_zero_partial_all_and_reentry(self):
        """Preserve current physical35 publication across queue withdrawal and reentry."""
        for snapshot in snapshots():
            reference, actual = bind(snapshot, DEVICE), bind(snapshot, DEVICE)
            original(reference)
            for ids in ([], [511, 0, 257, 3], np.arange(512)[::-1], [2, 19, 256, 510]):
                before = seed(actual)
                set_active(actual, ids)
                candidate(actual)
                check(self, actual, reference, before, ids)
            # Already-completed ZERO outputs are also valid physical results,
            # not only sentinels; an active-only replay must preserve them.
            for name in OUTPUT_KEYS:
                wp.copy(actual.values[name], reference.values[name])
            before = seed(actual, solved=True)
            ids = [511, 1, 10]
            set_active(actual, ids)
            candidate(actual)
            check(self, actual, reference, before, ids)
            actual.owner.solver._stage7_update_kinematics.assert_called_with(actual.state_out, actual.state_aug)
            actual.owner.solver.integrate_particles.assert_called_with(
                actual.owner.solver.model, actual.state_in, actual.state_out, 1.0 / 240.0
            )

    def test_unmasked_negative_control_overwrites_zero(self):
        """Reject the original unmasked chain when used as an active-only replacement."""
        iterator = snapshots()
        snapshot = next(iterator)
        actual, reference = bind(snapshot, "cpu"), bind(snapshot, "cpu")
        original(reference)
        before = seed(actual)
        original(actual)
        with self.assertRaises(AssertionError):
            check(self, actual, reference, before, [0, 1])

    def test_numeric_notification_has_no_topology_readback(self):
        """Keep all nonzero numeric subsets cheap while retaining structural assertions."""
        current = object.__new__(owner.JointWorldOwner)
        current.pending = False
        current.solver = SimpleNamespace(model=SimpleNamespace())
        current.model_plan_values = {}
        arrays = []
        for name in current._MODEL_PLAN_FIELDS:
            array = Mock()
            array.numpy.return_value = np.asarray([1, 2], dtype=np.int32)
            setattr(current.solver.model, name, array)
            current.model_plan_values[name] = np.asarray([1, 2], dtype=np.int32)
            arrays.append(array)
        bits = [
            ModelFlags.JOINT_DOF_PROPERTIES,
            ModelFlags.BODY_INERTIAL_PROPERTIES,
            ModelFlags.SHAPE_PROPERTIES,
            ModelFlags.MODEL_PROPERTIES,
        ]
        for subset in range(1, 16):
            flags = sum(int(bit) for i, bit in enumerate(bits) if subset & (1 << i))
            current.validate_notification(flags)
        for array in arrays:
            array.numpy.assert_not_called()
        for flags in (
            0,
            ModelFlags.BODY_PROPERTIES,
            ModelFlags.JOINT_PROPERTIES,
            ModelFlags.CONSTRAINT_PROPERTIES,
            ModelFlags.ALL,
            1 << 29,
        ):
            current.validate_notification(flags)
        for array in arrays:
            self.assertEqual(array.numpy.call_count, 6)
        current.solver.model.body_flags.numpy.return_value = np.asarray([1, 3], dtype=np.int32)
        with self.assertRaisesRegex(RuntimeError, "reconstruct"):
            current.validate_notification(ModelFlags.BODY_PROPERTIES)

    def test_begin_alias_and_unsupported_fallback(self):
        """Return to the original path before scheduling unsupported or aliased states."""
        current = object.__new__(owner.JointWorldOwner)
        current.pending = False
        current.solver = SimpleNamespace(model=SimpleNamespace(device="cpu"))
        current.buckets = SimpleNamespace(contact_capacity=10, build=Mock())
        current.stream, current.ready, current.done = Mock(), object(), object()
        a = wp.zeros(12, dtype=float, device="cpu")
        b = wp.zeros(8, dtype=float, device="cpu")
        source = SimpleNamespace(joint_q=a, joint_qd=b)
        same = SimpleNamespace(joint_q=a, joint_qd=wp.zeros(8, dtype=float, device="cpu"))
        partial = SimpleNamespace(
            joint_q=wp.array(ptr=a.ptr + 4, shape=(8,), capacity=32, dtype=float, device="cpu"),
            joint_qd=wp.zeros(8, dtype=float, device="cpu"),
        )
        body_storage = wp.zeros(14, dtype=float, device="cpu")
        source.body_q = wp.array(ptr=body_storage.ptr, shape=(2,), capacity=56, dtype=wp.transform, device="cpu")
        body_alias = SimpleNamespace(
            joint_q=wp.array(ptr=body_storage.ptr + 4, shape=(8,), capacity=32, dtype=float, device="cpu"),
            joint_qd=wp.zeros(8, dtype=float, device="cpu"),
        )
        contact = SimpleNamespace(rigid_contact_max=10)
        with (
            patch.object(owner.simple_world, "_supported", return_value=True),
            patch.object(owner, "raw_inputs") as raw,
        ):
            self.assertFalse(current.begin(source, same, contact, None))
            self.assertFalse(current.begin(source, partial, contact, None))
            self.assertFalse(current.begin(source, body_alias, contact, None))
            self.assertFalse(current.begin(source, same, None, None))
            raw.assert_not_called()
        with patch.object(owner.simple_world, "_supported", return_value=False):
            self.assertFalse(current.begin(source, same, contact, None))
        current.buckets.build.assert_not_called()

    def test_pending_raw_join_lifecycle(self):
        """Join an unfinished raw stream exactly once before releasing its pending flag."""
        current = object.__new__(owner.JointWorldOwner)
        current.solver = SimpleNamespace(model=SimpleNamespace(device="cpu"))
        current.pending, current.done = True, object()
        stream = Mock()
        stream.wait_event.side_effect = lambda event: self.assertTrue(current.pending)
        with patch.object(wp, "get_stream", return_value=stream):
            current.join_raw()
            self.assertFalse(current.pending)
            current.join_raw()
        stream.wait_event.assert_called_once_with(current.done)

    def test_failed_raw_build_records_recoverable_event(self):
        """Record and join prior raw readers even when construction raises after enqueue."""
        current = object.__new__(owner.JointWorldOwner)
        current.pending = False
        current.solver = SimpleNamespace(
            model=SimpleNamespace(device="cpu"), body_to_articulation=object(), art_to_world=object()
        )
        current.buckets = SimpleNamespace(contact_capacity=10, build=Mock(side_effect=RuntimeError("injected")))
        current.stream, current.ready, current.done = Mock(), object(), object()
        source = SimpleNamespace(
            joint_q=wp.zeros(8, dtype=float, device="cpu"), joint_qd=wp.zeros(8, dtype=float, device="cpu")
        )
        destination = SimpleNamespace(
            joint_q=wp.zeros(8, dtype=float, device="cpu"), joint_qd=wp.zeros(8, dtype=float, device="cpu")
        )
        raw = SimpleNamespace(count=object(), shape_body=object())
        for name in ("shape0", "shape1", "point0", "point1", "normal", "margin0", "margin1"):
            setattr(raw, name, SimpleNamespace(shape=(10,)))
        main, collision_done = Mock(), object()
        with (
            patch.object(owner.simple_world, "_supported", return_value=True),
            patch.object(owner, "raw_inputs", return_value=raw),
            patch.object(wp, "get_stream", return_value=main),
            patch.object(wp, "ScopedStream", return_value=nullcontext()),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                current.begin(source, destination, SimpleNamespace(rigid_contact_max=10), collision_done)
            self.assertTrue(current.pending)
            current.stream.record_event.assert_called_once_with(current.done)
            current.stream.wait_event.assert_any_call(collision_done)
            current.join_raw()
            self.assertFalse(current.pending)
        main.wait_event.assert_called_once_with(current.done)

    @unittest.skipUnless(DEVICE.startswith("cuda"), "Set FPGS_TEST_DEVICE to request actual CUDA graph controls")
    def test_two_actual_graphs_read_current_queue(self):
        """Replay two captures with current counts and preserve excluded publication owners."""
        iterator = snapshots()
        snapshot = next(iterator)
        actual, reference = bind(snapshot, DEVICE), bind(snapshot, DEVICE)
        original(reference)
        set_active(actual, np.arange(512))
        candidate(actual)  # Compile all original/adapted kernels before capture.
        with wp.ScopedCapture(device=DEVICE) as first:
            candidate(actual)
        with wp.ScopedCapture(device=DEVICE) as second:
            candidate(actual)
        for graph, ids in (
            (first.graph, []),
            (second.graph, [511, 0, 17]),
            (first.graph, np.arange(512)),
            (second.graph, [3, 4]),
        ):
            before = seed(actual)
            set_active(actual, ids)
            wp.capture_launch(graph)
            check(self, actual, reference, before, ids)


if __name__ == "__main__":
    unittest.main()
