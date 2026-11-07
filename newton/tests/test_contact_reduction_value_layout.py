# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the entry-major layout of the global contact reducer's winner slots."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_reduction_global import (
    VALUES_PER_KEY,
    VALUES_STRIDE,
    GlobalContactReducer,
    reduction_finalize_slot,
    reduction_insert_slot,
    reduction_rollback_slot,
    reduction_try_update_slot,
    reduction_update_slot,
    reduction_value_index,
    reduction_value_index_np,
)
from newton._src.geometry.hashtable import HASHTABLE_EMPTY_KEY
from newton.tests.unittest_utils import add_function_test, get_test_devices

_UPDATE, _TRY_UPDATE, _FINALIZE, _ROLLBACK, _INSERT = range(5)
_BACKGROUND = 1
_PROVISIONAL = 3
_WRITTEN = 5


@wp.kernel(enable_backward=False)
def _index_every_cell(hits: wp.array[int], indices: wp.array2d[int]):
    entry, slot = wp.tid()
    index = reduction_value_index(entry, slot)
    indices[entry, slot] = index
    wp.atomic_add(hits, index, 1)


@wp.kernel(enable_backward=False)
def _apply_accessor(
    mode: int,
    entries: wp.array[int],
    slots: wp.array[int],
    values: wp.array2d[wp.uint64],
    keys: wp.array2d[wp.uint64],
    active_slots: wp.array2d[wp.int32],
    returned: wp.array[wp.uint64],
):
    case = wp.tid()
    row = values[case]
    entry = entries[case]
    slot = slots[case]
    written = wp.uint64(_WRITTEN)
    provisional = wp.uint64(_PROVISIONAL)
    if mode == _UPDATE:
        reduction_update_slot(entry, slot, written, row)
    elif mode == _TRY_UPDATE:
        returned[case] = reduction_try_update_slot(entry, slot, written, row)
    elif mode == _FINALIZE:
        if reduction_finalize_slot(entry, slot, provisional, written, row):
            returned[case] = wp.uint64(1)
    elif mode == _ROLLBACK:
        reduction_rollback_slot(entry, slot, provisional, written, row)
    else:
        reduction_insert_slot(wp.uint64(entry + 1), slot, written, keys[case], row, active_slots[case])


def test_value_index_is_a_bijection(test, device):
    """Map each (entry, slot) of the table to its own value, covering every value once."""
    capacity = 64
    hits = wp.zeros(capacity * VALUES_STRIDE, dtype=int, device=device)
    indices = wp.zeros((capacity, VALUES_STRIDE), dtype=int, device=device)
    wp.launch(_index_every_cell, dim=(capacity, VALUES_STRIDE), inputs=[hits, indices], device=device)
    np.testing.assert_array_equal(hits.numpy(), 1)
    entries, slots = np.meshgrid(np.arange(capacity), np.arange(VALUES_STRIDE), indexing="ij")
    np.testing.assert_array_equal(indices.numpy(), reduction_value_index_np(entries, slots))

    reducer = GlobalContactReducer(capacity=4096, device=device)
    test.assertEqual(reducer.values_per_key, VALUES_PER_KEY)
    test.assertEqual(reducer.values_per_key, 7)
    test.assertEqual(reducer.values_stride, VALUES_STRIDE)
    test.assertEqual(reducer.ht_values.shape[0], reducer.hashtable.capacity * VALUES_STRIDE)


def test_slot_accessors_touch_only_their_cell(test, device):
    """Write each accessor's value into the one cell its (entry, slot) names.

    Accessors name winner slots only, so every row's padding word keeps its value.
    """
    capacity = 4
    entries, slots = (
        grid.ravel() for grid in np.meshgrid(np.arange(capacity), np.arange(VALUES_PER_KEY), indexing="ij")
    )
    cases = len(entries)
    cells = reduction_value_index_np(entries, slots)
    for mode in (_UPDATE, _TRY_UPDATE, _FINALIZE, _ROLLBACK, _INSERT):
        with test.subTest(mode=mode):
            values = np.full((cases, capacity * VALUES_STRIDE), _BACKGROUND, dtype=np.uint64)
            if mode in (_FINALIZE, _ROLLBACK):
                values[np.arange(cases), cells] = _PROVISIONAL
            values = wp.array(values, dtype=wp.uint64, device=device)
            keys = wp.full((cases, capacity), HASHTABLE_EMPTY_KEY, dtype=wp.uint64, device=device)
            active_slots = wp.zeros((cases, capacity + 1), dtype=wp.int32, device=device)
            returned = wp.zeros(cases, dtype=wp.uint64, device=device)
            wp.launch(
                _apply_accessor,
                dim=cases,
                inputs=[
                    mode,
                    wp.array(entries, dtype=int, device=device),
                    wp.array(slots, dtype=int, device=device),
                    values,
                    keys,
                    active_slots,
                    returned,
                ],
                device=device,
            )
            expected_cells = cells
            if mode == _INSERT:
                # The inserted key's entry comes from the hash; locate it in each case's table.
                found = keys.numpy() == (entries + 1)[:, None].astype(np.uint64)
                np.testing.assert_array_equal(found.sum(axis=1), 1)
                expected_cells = reduction_value_index_np(np.argmax(found, axis=1), slots)
            expected = np.full((cases, capacity * VALUES_STRIDE), _BACKGROUND, dtype=np.uint64)
            expected[np.arange(cases), expected_cells] = _WRITTEN
            np.testing.assert_array_equal(values.numpy(), expected)
            if mode == _TRY_UPDATE:
                np.testing.assert_array_equal(returned.numpy(), _BACKGROUND)
            elif mode == _FINALIZE:
                np.testing.assert_array_equal(returned.numpy(), 1)


_TERRAIN_FLAGS = {
    "NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY": "1",
    "NEWTON_HEIGHTFIELD_CELL_REJECT": "1",
    "NEWTON_HEIGHTFIELD_GEOMETRIC_CULL": "1",
    "NEWTON_HEIGHTFIELD_FINITE_QUERY": "0",
    "NEWTON_HEIGHTFIELD_PAIR_CSR": "0",
    # The reducer holds the terrain route's contacts: keep the route for every heightfield pair.
    "NEWTON_TERRAIN_LAW": "none",
}


def _reducer_scene():
    """Primitives and a hull on sloped terrain plus a mesh pair stacked on a plane."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = builder.ShapeConfig(margin=0.001, gap=0.01)
    builder.add_ground_plane(cfg=cfg)
    rng = np.random.default_rng(7)
    grid = np.linspace(-1.0, 1.0, 11, dtype=np.float32)
    heights = 0.06 * grid[None, :] + 0.04 * grid[:, None] + rng.uniform(-0.02, 0.02, size=(11, 11))
    terrain = newton.Heightfield(data=heights.astype(np.float32), nrow=11, ncol=11, hx=1.0, hy=1.0)
    builder.add_shape_heightfield(heightfield=terrain, xform=wp.transform((0.0, 0.0, 0.0)), cfg=cfg)
    for index, kind in enumerate(("sphere", "capsule", "cylinder", "box", "hull")):
        rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(*rng.normal(size=3))), float(rng.uniform(0.0, 1.0)))
        body = builder.add_body(xform=wp.transform((-0.6 + 0.3 * index, 0.1, 0.03), rotation))
        if kind == "sphere":
            builder.add_shape_sphere(body, radius=0.05, cfg=cfg)
        elif kind == "capsule":
            builder.add_shape_capsule(body, radius=0.03, half_height=0.08, cfg=cfg)
        elif kind == "cylinder":
            builder.add_shape_cylinder(body, radius=0.04, half_height=0.06, cfg=cfg)
        elif kind == "box":
            builder.add_shape_box(body, hx=0.07, hy=0.05, hz=0.04, cfg=cfg)
        else:
            hull = newton.Mesh.create_box(0.06, 0.05, 0.04, duplicate_vertices=False)
            builder.add_shape_convex_hull(body, mesh=hull, cfg=cfg)
    box = newton.Mesh.create_box(0.1, 0.08, 0.06, duplicate_vertices=False)
    for pose in ((3.0, 0.0, 0.058), (3.05, 0.01, 0.17)):
        body = builder.add_body(xform=wp.transform(pose, wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.2)))
        builder.add_shape_mesh(body, mesh=box, cfg=cfg)
    return builder.finalize(device="cpu")


class TestContactReductionValueLayout(unittest.TestCase):
    def test_pass_output_is_independent_of_table_size_cpu(self):
        """Export the same contacts in the same order wherever each key's row lands.

        Two table sizes place every key in a different row. Collisions at a pose,
        clear of contact, and back at the first pose then reproduce the first
        output, so clearing retires every winner of every active row.
        """
        model = _reducer_scene()
        state = model.state()
        initial = state.body_q.numpy().copy()
        lifted = initial.copy()
        # Distinct heights also separate the stacked mesh pair.
        lifted[:, 2] += 5.0 + np.arange(len(lifted))
        outputs, capacities = [], []
        for factor in (0.25, 4.0):
            with patch.dict(os.environ, _TERRAIN_FLAGS):
                pipeline = newton.CollisionPipeline(
                    model,
                    broad_phase="nxn",
                    rigid_contact_max=2048,
                    max_triangle_pairs=4096,
                    contact_reduction_hashtable_size_factor=factor,
                )
            reducer = pipeline.narrow_phase.global_contact_reducer
            capacities.append(reducer.hashtable.capacity)
            self.assertTrue(pipeline.narrow_phase._heightfield_primitive_query)
            contacts = pipeline.contacts()
            passes = []
            for body_q in (initial, lifted, initial):
                state.body_q.assign(body_q)
                pipeline.collide(state, contacts)
                count = int(contacts.rigid_contact_count.numpy()[0])
                passes.append(
                    {
                        name: getattr(contacts, name).numpy()[:count].copy()
                        for name in (
                            "rigid_contact_shape0",
                            "rigid_contact_shape1",
                            "rigid_contact_point0",
                            "rigid_contact_point1",
                            "rigid_contact_normal",
                            "rigid_contact_offset0",
                            "rigid_contact_offset1",
                            "rigid_contact_margin0",
                            "rigid_contact_margin1",
                        )
                    }
                )
                # Isaac Lab's suite check: retained export claims fit the active keys' slots.
                raw = int(reducer.contact_count.numpy()[0])
                active = int(reducer.hashtable.active_slots.numpy()[-1])
                claims = int(np.sum(reducer.exported_flags.numpy()[1 : raw + 1], dtype=np.int64))
                self.assertLessEqual(claims, reducer.values_per_key * active)
            self.assertGreater(len(passes[0]["rigid_contact_shape0"]), 0)
            self.assertEqual(len(passes[1]["rigid_contact_shape0"]), 0)
            for name, array in passes[0].items():
                np.testing.assert_array_equal(passes[2][name], array, err_msg=name)
            outputs.append(passes)
        self.assertLess(capacities[0], capacities[1])
        for name, array in outputs[0][0].items():
            np.testing.assert_array_equal(outputs[1][0][name], array, err_msg=name)


devices = get_test_devices()
add_function_test(
    TestContactReductionValueLayout,
    "test_value_index_is_a_bijection",
    test_value_index_is_a_bijection,
    devices=devices,
)
add_function_test(
    TestContactReductionValueLayout,
    "test_slot_accessors_touch_only_their_cell",
    test_slot_accessors_touch_only_their_cell,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
