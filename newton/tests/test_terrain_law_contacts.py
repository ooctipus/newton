# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the terrain law writing its partners' contacts (``NEWTON_TERRAIN_LAW``).

With partner classes listed, the narrow phase drops their heightfield pairs and the law writes those pairs' contacts
from its rows: one contact per row, through the stock contact writer, with the heightfield as shape A, the row's
normal, its terrain point and the partner's core point, and the row's feature id as sort key. Every other contact
stays what the terrain route and the narrow phase write without the flag.
"""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.tests import terrain_law_test_utils as tu
from newton.tests.test_terrain_law_shadow import PIPELINE, _mixed_scene
from newton.tests.unittest_utils import (
    add_function_test,
    get_cuda_test_devices,
    get_test_devices,
    scoped_capture_after_gc,
)


def _pipeline(model, environ: dict, **kwargs):
    """A collision pipeline built with only the given ``NEWTON_TERRAIN_LAW*`` variables set."""
    clean = {k: v for k, v in os.environ.items() if not k.startswith("NEWTON_TERRAIN_LAW")}
    clean.update(environ)
    with patch.dict(os.environ, clean, clear=True):
        return newton.CollisionPipeline(model, **kwargs)


def _collide(pipeline, state):
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    return contacts


def _pairs(contacts) -> dict:
    """Contact indices per (shape0, shape1) of the last collide."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    out = {}
    for i in range(count):
        out.setdefault((int(shape0[i]), int(shape1[i])), []).append(i)
    return out


def test_law_writes_its_rows_as_contacts(test, device):
    """Write one contact per law row for the listed partners' heightfield pairs, and leave every other contact.

    The terrain route writes these pairs' contacts per triangle and selects some afterwards (118 against the law's 81
    rows here); with the flag the contacts are the law's rows: the same terrain points, normals and core distances,
    within float32 frame changes.
    """
    model, terrain = _mixed_scene(device)
    state = model.state()
    for deterministic in (False, True):
        with test.subTest(deterministic=deterministic):
            today = _collide(_pipeline(model, {}, deterministic=deterministic, **PIPELINE), state)
            shadow = _pipeline(model, {"NEWTON_TERRAIN_LAW_SHADOW": "all"}, deterministic=deterministic, **PIPELINE)
            _collide(shadow, state)
            law = _pipeline(model, {"NEWTON_TERRAIN_LAW": "all"}, deterministic=deterministic, **PIPELINE)
            contacts = _collide(law, state)
            rows = tu.shadow_rows(shadow)
            got = tu.contact_rows(model, state, contacts, terrain)
            test.assertGreater(sum(len(r) for r in rows.values()), 20)
            test.assertEqual(sorted(p for p, r in rows.items() if r), sorted(got))
            for partner, expected in rows.items():
                written = got.get(partner, [])
                test.assertEqual(len(written), len(expected), msg=f"partner {partner}")
                missing = tu.missing_rows(expected, written, q_tol=2.0e-6, angle_tol=1.0e-5, d_tol=2.0e-6)
                test.assertEqual(missing, [], msg=f"partner {partner}")
            # Every contact without the heightfield is the narrow phase's, unchanged.
            before, after = _pairs(today), _pairs(contacts)
            other = sorted(p for p in before if terrain not in p)
            test.assertEqual(other, sorted(p for p in after if terrain not in p))
            for pair in other:
                for name in ("rigid_contact_point0", "rigid_contact_point1", "rigid_contact_normal"):
                    a = getattr(today, name).numpy()[before[pair]]
                    b = getattr(contacts, name).numpy()[after[pair]]
                    np.testing.assert_array_equal(np.sort(a, axis=0), np.sort(b, axis=0), err_msg=f"{pair} {name}")
            # Shape A is the heightfield; the margins and the sort key are the writer's.
            terrain_pairs = [p for p in after if terrain in p]
            test.assertTrue(all(p[0] == terrain for p in terrain_pairs))
            index = np.concatenate([after[p] for p in terrain_pairs])
            margin0 = contacts.rigid_contact_margin0.numpy()[index]
            np.testing.assert_allclose(margin0, model.shape_margin.numpy()[terrain], rtol=0, atol=1e-7)
            if deterministic:
                keys = law._sort_key_array.numpy()[: int(contacts.rigid_contact_count.numpy()[0])]
                test.assertEqual(len(np.unique(keys)), len(keys))


def test_law_contacts_replay_under_graph_capture(test, device):
    """Capture the collide with the law writing contacts and replay it to the contacts of an uncaptured collide."""
    model, _terrain = _mixed_scene(device)
    pipeline = _pipeline(model, {"NEWTON_TERRAIN_LAW": "all"}, deterministic=True, **PIPELINE)
    state = model.state()
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    names = ("rigid_contact_shape0", "rigid_contact_shape1", "rigid_contact_point0", "rigid_contact_normal")
    expected = {name: getattr(contacts, name).numpy()[:count].copy() for name in names}
    moved = model.state()
    q = moved.body_q.numpy()
    q[:, 2] += 0.5
    moved.body_q.assign(q)
    pipeline.collide(moved, contacts)
    test.assertLess(int(contacts.rigid_contact_count.numpy()[0]), count)
    with scoped_capture_after_gc(device=device) as capture:
        pipeline.collide(state, contacts)
    for _replay in range(2):
        wp.capture_launch(capture.graph)
        test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), count)
        for name in names:
            np.testing.assert_array_equal(getattr(contacts, name).numpy()[:count], expected[name], err_msg=name)


class TestTerrainLawContacts(unittest.TestCase):
    def test_flag_values_cpu(self):
        """Refuse both flags at once, an unknown class and speculative contacts; report the route."""
        model, _terrain = _mixed_scene(wp.get_device("cpu"))
        with self.assertRaisesRegex(ValueError, "both set"):
            _pipeline(model, {"NEWTON_TERRAIN_LAW": "all", "NEWTON_TERRAIN_LAW_SHADOW": "point"}, **PIPELINE)
        with self.assertRaisesRegex(ValueError, "NEWTON_TERRAIN_LAW names unknown partner class 'sphere'"):
            _pipeline(model, {"NEWTON_TERRAIN_LAW": "sphere"}, **PIPELINE)
        with self.assertRaisesRegex(ValueError, "speculative"):
            _pipeline(
                model,
                {"NEWTON_TERRAIN_LAW": "point"},
                speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(),
                **PIPELINE,
            )
        pipeline = _pipeline(model, {"NEWTON_TERRAIN_LAW": "point"}, **PIPELINE)
        self.assertIn("NEWTON_TERRAIN_LAW:", pipeline._terrain_law_shadow_reason)
        # The narrow phase drops the heightfield pairs of spheres and capsules only.
        partner = pipeline.narrow_phase._terrain_law_partner.numpy()
        kind = model.shape_type.numpy()
        point = (kind == int(newton.GeoType.SPHERE)) | (kind == int(newton.GeoType.CAPSULE))
        np.testing.assert_array_equal(partner != 0, point)

    def test_contact_overflow_raises_contacts_cpu(self):
        """Raise the narrow phase's ``contacts`` bit when the law's contacts overflow the contact buffer.

        The narrow phase checks the buffer before the law writes; with every terrain pair the law's, it sees no
        overflow of its own.
        """
        model, _terrain = _mixed_scene(wp.get_device("cpu"))
        state = model.state()
        pipeline = _pipeline(
            model, {"NEWTON_TERRAIN_LAW": "all"}, rigid_contact_max=60, max_triangle_pairs=1 << 16, verify_buffers=True
        )
        contacts = _collide(pipeline, state)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 60)
        self.assertTrue(pipeline.narrow_phase.buffer_capacity_status()["contacts"])
        roomy = _pipeline(model, {"NEWTON_TERRAIN_LAW": "all"}, verify_buffers=True, **PIPELINE)
        _collide(roomy, state)
        self.assertFalse(roomy.narrow_phase.buffer_capacity_status()["contacts"])

    def test_listed_classes_only_cpu(self):
        """With one class listed, only its pairs' contacts are the law's; the other pairs keep the terrain route."""
        model, terrain = _mixed_scene(wp.get_device("cpu"))
        state = model.state()
        today = tu.contact_rows(model, state, _collide(_pipeline(model, {}, **PIPELINE), state), terrain)
        shadow = _pipeline(model, {"NEWTON_TERRAIN_LAW_SHADOW": "box"}, **PIPELINE)
        _collide(shadow, state)
        rows = tu.shadow_rows(shadow)
        law = _pipeline(model, {"NEWTON_TERRAIN_LAW": "box"}, **PIPELINE)
        got = tu.contact_rows(model, state, _collide(law, state), terrain)
        kind = model.shape_type.numpy()
        for partner in sorted(set(today) | set(got)):
            if kind[partner] == int(newton.GeoType.BOX):
                self.assertEqual(len(got.get(partner, [])), len(rows.get(partner, [])), msg=f"box {partner}")
            else:
                self.assertEqual(len(got.get(partner, [])), len(today.get(partner, [])), msg=f"partner {partner}")


devices = get_test_devices()
add_function_test(
    TestTerrainLawContacts,
    "test_law_writes_its_rows_as_contacts",
    test_law_writes_its_rows_as_contacts,
    devices=devices,
)
add_function_test(
    TestTerrainLawContacts,
    "test_law_contacts_replay_under_graph_capture",
    test_law_contacts_replay_under_graph_capture,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
