# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepared rigid-model replication: ownership, indexing, and no reconstruction."""

import gc
import unittest
import weakref
from itertools import pairwise
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.sim import model_replication
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _prototype():
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    mesh = newton.Mesh.create_box(0.02, 0.02, 0.01, compute_inertia=False)
    arm = []
    parent = -1
    for index in range(3):
        body = builder.add_link(label=f"robot/link{index}")
        builder.add_shape_box(body, hx=0.03, hy=0.02, hz=0.1)
        arm.append(builder.add_joint_revolute(parent, body, axis=(0.0, 1.0, 0.0)))
        parent = body
    builder.add_articulation(arm)
    for partition in range(2):
        joints = []
        for key in range(6):
            body = builder.add_link(label=f"keyboard/partition{partition}/key{key}")
            builder.add_shape_mesh(body, mesh=mesh)
            joints.append(
                builder.add_joint_prismatic(-1, body, axis=(0.0, 0.0, 1.0), limit_lower=0.0, limit_upper=0.006)
            )
        builder.add_articulation(joints)
    # Different coordinate and DOF strides catch accidental interchange.
    body = builder.add_link(label="free_object")
    builder.add_shape_sphere(body, radius=0.04)
    builder.add_articulation([builder.add_joint_free(body)])
    builder.add_shape_box(-1, hx=0.5, hy=0.3, hz=0.01)
    builder.shape_collision_filter_pairs.append((0, 4))
    builder.add_custom_attribute(
        newton.ModelBuilder.CustomAttribute(
            name="owner",
            namespace="test",
            frequency=newton.Model.AttributeFrequency.SHAPE,
            dtype=wp.int32,
            references="body",
            default=-1,
            values={0: 2},
        )
    )
    return builder


def _build(prototype, count, device):
    builder = newton.ModelBuilder()
    builder.replicate(prototype, count)
    return builder.finalize(device=device)


def test_prepared_replication(test, device):
    """Match builder output without a host readback, finalization, or pair discovery."""
    with wp.ScopedDevice(device):
        prototype = _prototype()
        source = _build(prototype, 1, device)
        reference = _build(prototype, 3, device)
        before = source.joint_q.numpy().copy()
        source_arrays = []
        for name, _spec in source._iter_attribute_specs():
            owner, field = source, name
            if ":" in name:
                namespace, field = name.split(":", 1)
                owner = getattr(source, namespace)
            value = getattr(owner, field, None)
            if isinstance(value, wp.array):
                source_arrays.append((name, value, value.numpy()))
        with (
            mock.patch.object(wp.array, "numpy", side_effect=AssertionError("device readback")),
            mock.patch.object(newton.ModelBuilder, "finalize", side_effect=AssertionError("finalize")),
            mock.patch.object(newton.Model, "bvh_build_shapes", side_effect=AssertionError("geometry preparation")),
            mock.patch.object(
                newton.ModelBuilder, "_find_shape_contact_pairs", side_effect=AssertionError("pair discovery")
            ),
        ):
            result = source.replicate(3)
        test.assertIs(result._replication_source, source)
        test.assertNotIn("body_shapes", result.__dict__)
        test.assertEqual(result.world_count, 3)
        test.assertIsNot(result.bvh_shapes, source.bvh_shapes)
        test.assertIs(result.mesh_edge_indices, source.mesh_edge_indices)
        for name, array, initial in source_arrays:
            np.testing.assert_array_equal(array.numpy(), initial, err_msg=f"prototype mutated: {name}")
        for name, spec in source._iter_attribute_specs():
            if spec.compaction_policy == "passthrough":
                continue
            actual_owner, expected_owner, source_owner, field = result, reference, source, name
            if ":" in name:
                namespace, field = name.split(":", 1)
                actual_owner, expected_owner = getattr(result, namespace), getattr(reference, namespace)
                source_owner = getattr(source, namespace)
            if not hasattr(actual_owner, field):
                continue
            actual, expected = getattr(actual_owner, field), getattr(expected_owner, field)
            with test.subTest(attribute=name):
                if isinstance(actual, wp.array):
                    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-7)
                    if actual.size:
                        test.assertNotEqual(actual.ptr, getattr(source_owner, field).ptr)
                elif name != "shape_source":
                    test.assertEqual(actual, expected)
        for name in (
            "_fk_articulation_level_start",
            "_fk_level_joint_start",
            "_fk_level_joints",
            "_fk_level_parent_pos",
            "shape_contact_pairs",
            "gravity",
        ):
            np.testing.assert_array_equal(getattr(result, name).numpy(), getattr(reference, name).numpy(), err_msg=name)
        test.assertEqual(result.body_shapes, reference.body_shapes)
        test.assertIs(result.body_shapes, result.__dict__["body_shapes"])
        test.assertEqual(result.shape_collision_filter_pairs, reference.shape_collision_filter_pairs)
        np.testing.assert_array_equal(result.mujoco.equality_constraint_world_start.numpy(), [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(source.joint_q.numpy(), before)
        second = source.replicate(2)
        result.joint_q.fill_(0.125)
        np.testing.assert_array_equal(source.joint_q.numpy(), before)
        np.testing.assert_array_equal(second.joint_q.numpy(), np.tile(before, 2))
        # Fresh FK topology can be consumed normally, including the free joint.
        expected_model = _build(prototype, 2, device)
        actual_state, expected_state = second.state(), expected_model.state()
        newton.eval_fk(second, second.joint_q, second.joint_qd, actual_state)
        newton.eval_fk(expected_model, expected_model.joint_q, expected_model.joint_qd, expected_state)
        np.testing.assert_allclose(actual_state.body_q.numpy(), expected_state.body_q.numpy(), atol=1e-6)


def test_plain_replication_storage(test, device):
    """Pack only canonical plain fields, retaining bytes, metadata, and independent storage."""
    builder = _prototype()
    for name, frequency, dtype, default in (
        ("odd", "WORLD", wp.uint8, 7),
        ("enabled", "BODY", wp.bool, True),
        ("once", "ONCE", wp.uint16, 513),
    ):
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name=name,
                namespace="test",
                frequency=getattr(newton.Model.AttributeFrequency, frequency),
                dtype=dtype,
                default=default,
            )
        )
    source = _build(builder, 1, device)
    # An odd byte count exercises the byte arena, independently of boolean fields.
    source.test.odd = wp.array([1, 128, 255], dtype=wp.uint8, device=device)
    source._set_attribute_spec(
        "test:odd", newton.Model.AttributeSpec(newton.Model.AttributeFrequency.WORLD, row_width=3)
    )
    with mock.patch.object(model_replication, "_packed_plain_arrays", return_value=({}, ())):
        ordinary = source.replicate(3)
    packed = source.replicate(3)
    plan = source._replication_copy_plan
    test.assertIs(packed._replication_storage[0], plan)
    test.assertIsNot(packed.attribute_specs, source.attribute_specs)
    test.assertIsNot(packed.test, source.test)
    spans = []
    for _, _, records, _, _, _ in plan.groups:
        for name, value, _, _ in records:
            target, reference = packed, ordinary
            field = name
            if ":" in name:
                namespace, field = name.split(":", 1)
                target, reference = getattr(target, namespace), getattr(reference, namespace)
            a, b = getattr(target, field), getattr(reference, field)
            test.assertEqual((a.shape, a.dtype), (b.shape, b.dtype))
            np.testing.assert_array_equal(a.numpy().view(np.uint8), b.numpy().view(np.uint8), err_msg=name)
            test.assertNotEqual(a.ptr, value.ptr)
            test.assertEqual(a.ptr % 256, 0)
            spans.append((a.ptr, a.ptr + a.size * a.strides[-1]))
    spans.sort()
    test.assertTrue(all(end <= start for (_, end), (start, _) in pairwise(spans)))
    test.assertEqual(packed.test.once.shape, (1,))
    np.testing.assert_array_equal(packed.test.odd.numpy(), np.tile([1, 128, 255], 3))
    packed.test.odd.fill_(17)
    np.testing.assert_array_equal(source.test.odd.numpy(), [1, 128, 255])
    np.testing.assert_array_equal(ordinary.test.odd.numpy(), np.tile([1, 128, 255], 3))
    # Population sizes never create cached plans or allocations.
    for count in (1, 4, 2, 3):
        result = source.replicate(count)
        test.assertIs(source._replication_copy_plan, plan)
        test.assertEqual(result.world_count, count)
        np.testing.assert_array_equal(result.test.odd.numpy(), np.tile([1, 128, 255], count))
        del result
    test.assertEqual(set(vars(plan)), {"signature", "groups"})
    source.test.odd.fill_(19)
    result = source.replicate(2)
    test.assertIs(source._replication_copy_plan, plan)
    np.testing.assert_array_equal(result.test.odd.numpy(), np.full(6, 19))
    source.test.odd = wp.array([21, 22, 23], dtype=wp.uint8, device=device)
    result = source.replicate(2)
    test.assertIsNot(source._replication_copy_plan, plan)
    np.testing.assert_array_equal(result.test.odd.numpy(), [21, 22, 23, 21, 22, 23])
    # A semantic spec change invalidates even with the same source array and pointer.
    plan = source._replication_copy_plan
    source._set_attribute_spec("test:once", newton.Model.AttributeSpec(newton.Model.AttributeFrequency.WORLD))
    result = source.replicate(2)
    test.assertIsNot(source._replication_copy_plan, plan)
    np.testing.assert_array_equal(result.test.once.numpy(), [513, 513])
    # Layout changes, including an offset pointer, are read through the current layout.
    plan = source._replication_copy_plan
    source.test.odd = wp.array([0, 25, 26, 27], dtype=wp.uint8, device=device)[1:]
    result = source.replicate(2)
    test.assertIsNot(source._replication_copy_plan, plan)
    np.testing.assert_array_equal(result.test.odd.numpy(), [25, 26, 27, 25, 26, 27])


def test_plain_replication_lifetime(test, device):
    """An extracted field retains its backing allocation without retaining the model or plan."""

    def extract():
        source = _build(_prototype(), 1, device)
        result = source.replicate(3)
        return result.body_mass, weakref.ref(source), weakref.ref(result), weakref.ref(source._replication_copy_plan)

    field, source_ref, result_ref, plan_ref = extract()
    wp.synchronize_device(device)
    expected = field.numpy().copy()
    gc.collect()
    test.assertIsNone(source_ref())
    test.assertIsNone(result_ref())
    test.assertIsNone(plan_ref())
    np.testing.assert_array_equal(field.numpy(), expected)
    field.fill_(42.0)
    np.testing.assert_array_equal(field.numpy(), np.full(field.shape, 42.0))


def test_plain_replication_fallback(test, device):
    """Decline unsupported raw layouts before any arena allocation, preserving ordinary replication."""
    source = _build(_prototype(), 1, device)
    original = source.body_mass
    values = np.repeat(original.numpy(), 2)
    source.body_mass = wp.array(values, dtype=original.dtype, device=device)[::2]
    with mock.patch.object(model_replication, "_PlainCopyPlan", side_effect=AssertionError("packed strided array")):
        result = source.replicate(2)
    np.testing.assert_array_equal(result.body_mass.numpy(), np.tile(original.numpy(), 2))
    test.assertFalse(hasattr(source, "_replication_copy_plan"))
    # A metadata-only large span must be rejected before making a raw int32 view or launching.
    source.body_mass = wp.array(ptr=original.ptr, shape=2**29, dtype=wp.vec3, device=device)
    counts = {key: source._attribute_frequency_count(key) for key in source._ATTRIBUTE_FREQUENCY_COUNT_ATTRS}
    counts.update(source.custom_frequency_counts)
    counts[newton.Model.AttributeFrequency.ONCE] = 1
    with (
        mock.patch.object(model_replication, "_PlainCopyPlan", side_effect=AssertionError("overflowed raw span")),
        mock.patch.object(wp, "launch", side_effect=AssertionError("overflowed launch")),
    ):
        arrays, storage = model_replication._packed_plain_arrays(
            source, tuple(source._iter_attribute_specs()), counts, 2
        )
    test.assertEqual((arrays, storage), ({}, ()))


def test_plain_replication_fallback_releases_source_plan(test, device):
    """Fallback evicts only the source cache; existing replicas keep their upload owners alive."""
    source = _build(_prototype(), 1, device)
    original = source.body_mass.numpy().copy()
    first = source.replicate(2)
    plan_ref = weakref.ref(source._replication_copy_plan)
    array_ref = weakref.ref(source.body_mass)
    source.body_mass = wp.array(np.repeat(original, 2), dtype=float, device=device)[::2]
    fallback = source.replicate(3)
    wp.synchronize_device(device)
    test.assertFalse(hasattr(source, "_replication_copy_plan"))
    gc.collect()
    test.assertIsNotNone(plan_ref())
    test.assertIsNotNone(array_ref())
    np.testing.assert_array_equal(first.body_mass.numpy(), np.tile(original, 2))
    np.testing.assert_array_equal(fallback.body_mass.numpy(), np.tile(original, 3))
    del first
    gc.collect()
    test.assertIsNone(plan_ref())
    test.assertIsNone(array_ref())
    source.body_mass = wp.array(original, dtype=float, device=device)
    next_replica = source.replicate(4)
    test.assertTrue(hasattr(source, "_replication_copy_plan"))
    np.testing.assert_array_equal(next_replica.body_mass.numpy(), np.tile(original, 4))


class TestModelReplication(unittest.TestCase):
    def test_transfer_layout_validation_precedes_device_work(self):
        """Reject a strided registered world field before launching any transfer kernel."""
        source_model = _build(_prototype(), 1, "cpu")
        source_model.custom_values = wp.array([[2.0, 3.0]], dtype=float, device="cpu")
        source_model.attribute_specs["custom_values"] = newton.Model.AttributeSpec(
            newton.Model.AttributeFrequency.WORLD
        )
        source_model.attribute_frequency["custom_values"] = newton.Model.AttributeFrequency.WORLD
        source, target = source_model.replicate(2), source_model.replicate(3)
        source.body_mass.fill_(42.0)
        target.custom_values = wp.zeros((6, 2), dtype=float, device="cpu")[::2]
        before = target.body_mass.numpy().copy()
        states, controls = ((source.state(), target.state()),), ((source.control(), target.control()),)
        with (
            mock.patch.object(wp, "launch", side_effect=AssertionError("transfer started before validation")),
            self.assertRaisesRegex(ValueError, "must be contiguous: custom_values"),
        ):
            model_replication._world_copy_arrays(source, target, states, controls)
        np.testing.assert_array_equal(target.body_mass.numpy(), before)

    def test_plain_copy_kernel_bytes(self):
        """Exercise the byte-preserving kernels on CPU, including odd widths and NaN payloads."""
        values = {
            "odd": wp.array([1, 128, 255], dtype=wp.uint8, device="cpu"),
            "bool": wp.array([True, False, True], dtype=wp.bool, device="cpu"),
            "short": wp.array([1, 512, 65535], dtype=wp.uint16, device="cpu"),
            "double": wp.array(
                np.array([0, 2**63, 0x7FF8000000000042], dtype=np.uint64).view(np.float64), device="cpu"
            ),
            "once": wp.array([0xDEADBEEF], dtype=wp.uint32, device="cpu"),
        }
        fields = []
        for name, value in values.items():
            size = value.size * value.strides[-1]
            fields.append((name, value, name != "once", size, 4 if size % 4 == 0 else 1))
        plan = model_replication._PlainCopyPlan(fields, ())
        for count in (1, 4, 2):
            arrays, storage = plan.copy(count)
            self.assertIs(storage[0], plan)
            for name, value in values.items():
                expected = np.tile(value.numpy(), count if name != "once" else 1)
                np.testing.assert_array_equal(arrays[name].numpy().view(np.uint8), expected.view(np.uint8))
                self.assertNotEqual(arrays[name].ptr, value.ptr)
                self.assertEqual(arrays[name].ptr % 16, 0)

    def test_reject_unsupported_layouts(self):
        """Reject unsupported domains and unknown storage instead of rebuilding or aliasing."""
        prototype = newton.ModelBuilder()
        prototype.add_body()
        source = _build(prototype, 1, "cpu")
        for count in (0, -1, True):
            with self.assertRaises(ValueError):
                source.replicate(count)
        with self.assertRaisesRegex(ValueError, "explicit local world"):
            prototype.finalize("cpu").replicate(2)
        globals_builder = newton.ModelBuilder()
        globals_builder.add_body()
        globals_builder.replicate(prototype, 1)
        with self.assertRaisesRegex(ValueError, "global entities"):
            globals_builder.finalize("cpu").replicate(2)
        for name, value in (
            ("particle_count", 1),
            ("heightfield_count", 1),
            ("requires_grad", True),
            ("_has_rod_joints", True),
            ("actuators", [object()]),
            ("custom_frequency_counts", {"test:extra": 1}),
        ):
            old = getattr(source, name)
            setattr(source, name, value)
            with self.assertRaises(ValueError):
                source.replicate(2)
            setattr(source, name, old)
        source.unregistered = wp.ones(1, device="cpu")
        with self.assertRaisesRegex(ValueError, "lacks metadata"):
            source.replicate(2)

    def test_compact_filter_set(self):
        """Answer membership from the prototype and materialize only explicit host views."""
        prototype = _prototype()
        source = _build(prototype, 1, "cpu")
        result = source.replicate(4)
        filters = result._shape_collision_filter_pairs
        self.assertIsNone(filters._packed_data)
        self.assertEqual(len(filters), 4 * len(source.shape_collision_filter_pairs))
        self.assertTrue(filters)
        for world in range(4):
            offset = world * source.shape_count
            self.assertIn((offset, offset + 4), filters)
            self.assertTrue(result.shape_collision_filter_contains(offset + 4, offset))
        self.assertNotIn((0, source.shape_count + 4), filters)
        self.assertNotIn((-1, 4), filters)
        self.assertNotIn((0, 4 * source.shape_count), filters)
        self.assertIsNone(filters._packed_data)
        reference = _build(prototype, 4, "cpu")
        np.testing.assert_array_equal(filters.pairs_array(), reference._shape_collision_filter_pairs.pairs_array())
        self.assertFalse(filters.pairs_array().flags.writeable)

    def test_empty_world(self):
        """Retain sentinels and per-world gravity when the prototype has no entities."""
        builder = newton.ModelBuilder()
        builder.begin_world()
        builder.end_world()
        source = builder.finalize("cpu")
        result = source.replicate(4)
        self.assertEqual(result.world_count, 4)
        np.testing.assert_array_equal(result.body_world_start.numpy(), np.zeros(6))
        np.testing.assert_array_equal(result.joint_q_start.numpy(), [0])
        self.assertEqual(result.gravity.shape, (5,))

    def test_host_topology_is_lazy(self):
        """Keep prepared replication independent of the host body-to-shape map."""
        source = _build(_prototype(), 1, "cpu")
        mapping = source.body_shapes
        source.body_shapes = mock.Mock()
        source.body_shapes.items.side_effect = AssertionError("host body topology expansion")
        with mock.patch.object(wp.array, "numpy", side_effect=AssertionError("device readback")):
            result = source.replicate(1)
            chained = result.replicate(3)
        self.assertNotIn("body_shapes", result.__dict__)
        self.assertNotIn("body_shapes", chained.__dict__)
        source.body_shapes = mapping
        reference = _build(_prototype(), 3, "cpu")
        self.assertEqual(chained.body_shapes, reference.body_shapes)

    def test_prepared_bvh_selection(self):
        """Preserve a prototype's explicit empty BVH selection without rediscovering geometry."""
        builder = newton.ModelBuilder()
        builder.begin_world()
        body = builder.add_body()
        builder.add_shape_box(body)
        builder.end_world()
        source = builder.finalize("cpu")
        source.bvh_build_shapes(source, shape_flags=0)
        with mock.patch.object(wp.array, "numpy", side_effect=AssertionError("device readback")):
            result = source.replicate(3)
        self.assertEqual(result.bvh_shape_count_enabled, 0)
        self.assertIsNone(result.bvh_shapes)

    def test_bvh_outputs_overwrite_uninitialized_storage(self):
        """Write every BVH output, including the global group, without clearing it first."""
        builder = newton.ModelBuilder()
        builder.replicate(_prototype(), 2)
        builder.add_shape_box(-1, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize("cpu")
        expected = [model.bvh_shapes.lowers.numpy(), model.bvh_shapes.uppers.numpy(), model.bvh_shapes.groups.numpy()]
        empty = wp.empty

        def poisoned_empty(*args, **kwargs):
            array = empty(*args, **kwargs)
            array.fill_(-12345 if array.dtype == wp.int32 else float("nan"))
            return array

        with (
            mock.patch.object(wp, "empty", side_effect=poisoned_empty),
            mock.patch.object(wp, "zeros", side_effect=AssertionError("redundant BVH initialization")),
        ):
            model._build_shape_bvh(model)
        for actual, expected_array in zip(
            (model.bvh_shapes.lowers, model.bvh_shapes.uppers, model.bvh_shapes.groups), expected, strict=True
        ):
            np.testing.assert_array_equal(actual.numpy(), expected_array)
        self.assertTrue(np.all(model.bvh_shapes_group_roots.numpy() >= 0))

    def test_joint_target_layout_snapshot(self):
        """Use the prepared model's target layout even if the process default changes."""
        original = newton.use_coord_layout_targets
        try:
            for layout in (False, True):
                newton.use_coord_layout_targets = layout
                source = _build(_prototype(), 1, "cpu")
                newton.use_coord_layout_targets = not layout
                result = source.replicate(3)
                self.assertEqual(result.use_coord_layout_targets, layout)
                np.testing.assert_array_equal(result.joint_target_q.numpy(), np.tile(source.joint_target_q.numpy(), 3))
                self.assertEqual(result._attribute_spec("joint_target_q"), source._attribute_spec("joint_target_q"))
        finally:
            newton.use_coord_layout_targets = original


for device in get_test_devices():
    add_function_test(TestModelReplication, "test_prepared_replication", test_prepared_replication, devices=[device])
    if device.is_cuda:
        for function in (
            test_plain_replication_storage,
            test_plain_replication_lifetime,
            test_plain_replication_fallback,
            test_plain_replication_fallback_releases_source_plan,
        ):
            add_function_test(TestModelReplication, function.__name__, function, devices=[device])


if __name__ == "__main__":
    unittest.main(verbosity=2)
