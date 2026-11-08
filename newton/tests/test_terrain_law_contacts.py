# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the terrain law writing its partners' heightfield contacts: the default, and ``NEWTON_TERRAIN_LAW``.

By default, and for the partner classes ``NEWTON_TERRAIN_LAW`` lists, the narrow phase drops the heightfield pairs and
the law writes those pairs' contacts from its rows: one contact per row, through the stock contact writer, with the
heightfield as shape A, the row's normal, its terrain point and the partner's core point, and the row's feature id as
sort key. Every other contact stays what the terrain route and the narrow phase write. A configuration the law does
not support yet (speculative contacts, the shadow's comparison mode, a prebuilt narrow phase with its own contact
writer) keeps the terrain route by default and raises when the law is requested; ``NEWTON_TERRAIN_LAW=none`` keeps
the terrain route (a temporary opt-out).
"""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_data import ContactData
from newton._src.geometry.narrow_phase import NarrowPhase
from newton._src.sim.collide import ContactWriterData, write_contact
from newton.geometry import BroadPhaseExplicit
from newton.tests import terrain_law_test_utils as tu
from newton.tests.test_terrain_law_shadow import PIPELINE, _mixed_scene
from newton.tests.unittest_utils import (
    add_function_test,
    get_cuda_test_devices,
    get_test_devices,
    scoped_capture_after_gc,
)

LAW_CLASSES = ["point", "box", "cylinder", "hull"]
"""Every partner class with a kernel: the classes the law writes by default."""

TERRAIN_ROUTE = {"NEWTON_TERRAIN_LAW": "none"}
"""The law's temporary opt-out: the terrain route writes every heightfield contact."""


@wp.func
def _forwarding_writer(contact_data: ContactData, writer_data: ContactWriterData, output_index: int):
    """A contact writer of its own (another function than the stock writer it forwards to)."""
    write_contact(contact_data, writer_data, output_index)


def _pipeline(model, environ: dict, **kwargs):
    """A collision pipeline built with only the given ``NEWTON_TERRAIN_LAW*`` variables set."""
    clean = {k: v for k, v in os.environ.items() if not k.startswith("NEWTON_TERRAIN_LAW")}
    clean.update(environ)
    with patch.dict(os.environ, clean, clear=True):
        return newton.CollisionPipeline(model, **kwargs)


def _collide(pipeline, state, **kwargs):
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts, **kwargs)
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


def _arrays(contacts) -> dict:
    """The per-contact geometry of the last collide, cut to the contact count."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    names = (
        "rigid_contact_shape0",
        "rigid_contact_shape1",
        "rigid_contact_point0",
        "rigid_contact_point1",
        "rigid_contact_offset0",
        "rigid_contact_offset1",
        "rigid_contact_normal",
        "rigid_contact_margin0",
        "rigid_contact_margin1",
    )
    return {name: getattr(contacts, name).numpy()[:count] for name in names}


def _canonical(arrays: dict) -> np.ndarray:
    """The contacts of :func:`_arrays` as rows of their bits, sorted: the same set in any order compares equal."""
    count = len(arrays["rigid_contact_shape0"])
    columns = [np.ascontiguousarray(arrays[name]).view(np.int32).reshape(count, -1) for name in sorted(arrays)]
    stacked = np.concatenate(columns, axis=1)
    return stacked[np.lexsort(stacked.T[::-1])]


def _assert_same_contacts(test, a: dict, b: dict):
    """Assert that two collides wrote the same contacts bit for bit, in any order."""
    test.assertEqual(sorted(a), sorted(b))
    test.assertGreater(len(a["rigid_contact_shape0"]), 0)
    np.testing.assert_array_equal(_canonical(a), _canonical(b))


def _law_record(pipeline) -> dict:
    return pipeline.get_collision_selection()["heightfield"]["law"]


def _expert_narrow_phase(model, writer):
    """A prebuilt narrow phase for ``model``'s pairs that writes through ``writer``."""
    device = model.device
    return NarrowPhase(
        max_candidate_pairs=int(model.shape_contact_pair_count),
        max_triangle_pairs=PIPELINE["max_triangle_pairs"],
        device=device,
        shape_aabb_lower=wp.zeros(model.shape_count, dtype=wp.vec3, device=device),
        shape_aabb_upper=wp.zeros(model.shape_count, dtype=wp.vec3, device=device),
        contact_writer_warp_func=writer,
        has_meshes=False,
        has_heightfields=True,
        contact_max=1 << 14,
    )


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
            today = _collide(_pipeline(model, TERRAIN_ROUTE, deterministic=deterministic, **PIPELINE), state)
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


def test_default_writes_with_the_law(test, device):
    """Write every partner class's heightfield contacts with the law by default, as ``NEWTON_TERRAIN_LAW=all`` does.

    Without the variable the pipeline builds the law for every class it has a kernel for: its contacts are those of
    an explicit ``all`` bit for bit (both deterministic), the narrow phase drops exactly the law's partners' pairs, and
    the route record says that the law writes by default. The terrain route's contacts of those pairs differ (118
    against the law's 81 rows on the CPU).
    """
    model, terrain = _mixed_scene(device)
    state = model.state()
    default = _pipeline(model, {}, deterministic=True, **PIPELINE)
    explicit = _pipeline(model, {"NEWTON_TERRAIN_LAW": "all"}, deterministic=True, **PIPELINE)
    route = _pipeline(model, TERRAIN_ROUTE, deterministic=True, **PIPELINE)
    record = _law_record(default)
    test.assertTrue(record["writes"])
    test.assertEqual(record["decision"], "default")
    test.assertEqual(record["classes"], LAW_CLASSES)
    test.assertTrue(record["reason"].startswith("Default: the terrain law writes"))
    test.assertEqual(sum(sum(lanes.values()) for lanes in record["pairs"].values()), 24)
    selection = default.get_collision_selection()["heightfield"]
    test.assertFalse(selection["law_shadow"]["enabled"])
    test.assertEqual(_law_record(explicit)["decision"], "explicit")
    partner = default.narrow_phase._terrain_law_partner.numpy()
    kind = model.shape_type.numpy()
    np.testing.assert_array_equal(partner != 0, kind != int(newton.GeoType.HFIELD))
    contacts = _collide(default, state)
    _assert_same_contacts(test, _arrays(contacts), _arrays(_collide(explicit, state)))
    law_rows = sum(len(r) for r in tu.contact_rows(model, state, contacts, terrain).values())
    route_rows = sum(len(r) for r in tu.contact_rows(model, state, _collide(route, state), terrain).values())
    test.assertGreater(law_rows, 20)
    test.assertNotEqual(law_rows, route_rows)


def test_law_contacts_replay_under_graph_capture(test, device):
    """Capture the collide with the law writing contacts and replay it to the contacts of an uncaptured collide."""
    model, _terrain = _mixed_scene(device)
    pipeline = _pipeline(model, {}, deterministic=True, **PIPELINE)
    test.assertTrue(_law_record(pipeline)["writes"])
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
        """Refuse both flags at once, an unknown class, ``none`` with classes and speculative contacts; report the
        route."""
        model, _terrain = _mixed_scene(wp.get_device("cpu"))
        with self.assertRaisesRegex(ValueError, "both set"):
            _pipeline(model, {"NEWTON_TERRAIN_LAW": "all", "NEWTON_TERRAIN_LAW_SHADOW": "point"}, **PIPELINE)
        with self.assertRaisesRegex(ValueError, "NEWTON_TERRAIN_LAW names unknown partner class 'sphere'"):
            _pipeline(model, {"NEWTON_TERRAIN_LAW": "sphere"}, **PIPELINE)
        with self.assertRaisesRegex(ValueError, "'none' opts out of the terrain law and stands alone"):
            _pipeline(model, {"NEWTON_TERRAIN_LAW": "none, point"}, **PIPELINE)
        with self.assertRaisesRegex(ValueError, "does not write speculative contacts yet"):
            _pipeline(
                model,
                {"NEWTON_TERRAIN_LAW": "point"},
                speculative_contact_gap_max=0.1,
                **PIPELINE,
            )
        pipeline = _pipeline(model, {"NEWTON_TERRAIN_LAW": " point "}, **PIPELINE)
        record = _law_record(pipeline)
        self.assertTrue(record["writes"])
        self.assertEqual(record["decision"], "explicit")
        self.assertEqual(record["classes"], ["point"])
        self.assertTrue(record["reason"].startswith("NEWTON_TERRAIN_LAW=point: the terrain law writes"))
        # The narrow phase drops the heightfield pairs of spheres and capsules only.
        partner = pipeline.narrow_phase._terrain_law_partner.numpy()
        kind = model.shape_type.numpy()
        point = (kind == int(newton.GeoType.SPHERE)) | (kind == int(newton.GeoType.CAPSULE))
        np.testing.assert_array_equal(partner != 0, point)
        # A blank value is the default.
        self.assertEqual(
            _law_record(_pipeline(model, {"NEWTON_TERRAIN_LAW": " , "}, **PIPELINE))["decision"], "default"
        )

    def test_opt_out_keeps_the_terrain_route_cpu(self):
        """Keep the terrain route for every heightfield pair with ``NEWTON_TERRAIN_LAW=none``: no law is built, the
        narrow phase drops no pair, and the record says it is the temporary opt-out.

        A narrow phase that a law-writing pipeline configured, handed to an opted-out pipeline, routes every pair
        again: that pipeline writes the terrain route's contacts.
        """
        model, terrain = _mixed_scene(wp.get_device("cpu"))
        state = model.state()
        route = _pipeline(model, TERRAIN_ROUTE, **PIPELINE)
        self.assertIsNone(route._terrain_law_shadow)
        self.assertEqual(route.narrow_phase._terrain_law_partner.shape[0], 0)
        record = _law_record(route)
        self.assertEqual(
            {k: record[k] for k in ("writes", "decision", "classes")},
            {"writes": False, "decision": "opted_out", "classes": []},
        )
        self.assertIn("temporary opt-out", record["reason"])
        self.assertEqual(route.get_collision_selection()["environment"]["applied"]["NEWTON_TERRAIN_LAW"], "none")
        contacts = _collide(route, state)
        expected = _arrays(contacts)
        # 83 rows: the route's seam witnesses take the face beneath their partner, which retires 31 oblique rows
        # of 114 and keeps every partner's deepest row.
        self.assertEqual(sum(len(r) for r in tu.contact_rows(model, state, contacts, terrain).values()), 83)
        law = _pipeline(model, {}, **PIPELINE)
        self.assertTrue(_law_record(law)["writes"])
        expert = _pipeline(
            model,
            TERRAIN_ROUTE,
            broad_phase=BroadPhaseExplicit(),
            narrow_phase=law.narrow_phase,
        )
        self.assertEqual(_law_record(expert)["decision"], "opted_out")
        self.assertEqual(law.narrow_phase._terrain_law_partner.shape[0], 0)
        _assert_same_contacts(self, expected, _arrays(_collide(expert, state)))

    def test_fallbacks_keep_the_terrain_route_cpu(self):
        """Keep the terrain route by default where the law does not support the configuration yet, and say why:
        speculative contacts, the shadow's comparison mode and a prebuilt narrow phase with its own contact writer.

        Each writes the terrain route's contacts (those of ``NEWTON_TERRAIN_LAW=none`` in the same configuration);
        requested explicitly, the law raises instead (the shadow refuses ``NEWTON_TERRAIN_LAW`` in
        :meth:`test_flag_values_cpu`).
        """
        model, terrain = _mixed_scene(wp.get_device("cpu"))
        state = model.state()
        speculative = {"speculative_contact_gap_max": 0.1, **PIPELINE}
        cases = (
            ("speculative", {}, speculative, {"dt": 0.01}, "Speculative contacts: the terrain law does not write"),
            ("shadow", {"NEWTON_TERRAIN_LAW_SHADOW": "all"}, PIPELINE, {}, "NEWTON_TERRAIN_LAW_SHADOW runs the law"),
        )
        for name, environ, kwargs, collide, reason in cases:
            with self.subTest(fallback=name):
                pipeline = _pipeline(model, environ, deterministic=True, **kwargs)
                record = _law_record(pipeline)
                self.assertEqual(
                    {k: record[k] for k in ("writes", "decision", "classes")},
                    {"writes": False, "decision": "unsupported", "classes": []},
                )
                self.assertTrue(record["reason"].startswith(reason), msg=record["reason"])
                self.assertEqual(pipeline.narrow_phase._terrain_law_partner.shape[0], 0)
                self.assertEqual(pipeline._terrain_law_shadow is not None, name == "shadow")
                self.assertEqual(
                    pipeline.get_collision_selection()["heightfield"]["law_shadow"]["enabled"], name == "shadow"
                )
                route = _pipeline(model, {**environ, **TERRAIN_ROUTE}, deterministic=True, **kwargs)
                self.assertEqual(_law_record(route)["decision"], "opted_out")
                _assert_same_contacts(
                    self, _arrays(_collide(route, state, **collide)), _arrays(_collide(pipeline, state, **collide))
                )
        # A prebuilt narrow phase with its own contact writer: the law would write past it.
        expert = _pipeline(
            model,
            {},
            broad_phase=BroadPhaseExplicit(),
            narrow_phase=_expert_narrow_phase(model, _forwarding_writer),
        )
        record = _law_record(expert)
        self.assertEqual((record["writes"], record["decision"]), (False, "unsupported"))
        self.assertTrue(record["reason"].startswith("A prebuilt narrow phase with its own contact writer"))
        self.assertIsNone(expert._terrain_law_shadow)
        rows = tu.contact_rows(model, state, _collide(expert, state), terrain)
        self.assertGreater(sum(len(r) for r in rows.values()), 20)
        with self.assertRaisesRegex(ValueError, "writes through the stock contact writer"):
            _pipeline(
                model,
                {"NEWTON_TERRAIN_LAW": "all"},
                broad_phase=BroadPhaseExplicit(),
                narrow_phase=_expert_narrow_phase(model, _forwarding_writer),
            )
        # With the stock writer, a prebuilt narrow phase is supported: the law writes by default.
        stock = _pipeline(
            model,
            {},
            broad_phase=BroadPhaseExplicit(),
            narrow_phase=_expert_narrow_phase(model, write_contact),
        )
        self.assertEqual(_law_record(stock)["decision"], "default")

    def test_no_partner_keeps_the_terrain_route_cpu(self):
        """Build no law where no heightfield pair has a partner it writes, and say so: a triangle mesh's only pair (the
        mesh path's), and a listed class without pairs."""
        builder = newton.ModelBuilder()
        cfg = builder.ShapeConfig(margin=0.01, gap=0.01)
        terrain = newton.Heightfield(data=np.zeros((9, 9), np.float32), nrow=9, ncol=9, hx=0.4, hy=0.4)
        builder.add_shape_heightfield(heightfield=terrain, cfg=cfg)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.2, 0.0, 0.05)))
        mesh = newton.Mesh.create_box(0.03, 0.03, 0.03, compute_inertia=False)
        builder.add_shape_mesh(body, mesh=mesh, cfg=cfg)
        model = builder.finalize(device="cpu")
        pipeline = _pipeline(model, {}, **PIPELINE)
        record = _law_record(pipeline)
        self.assertEqual((record["writes"], record["decision"]), (False, "no_partner"))
        self.assertIsNone(pipeline._terrain_law_shadow)
        self.assertEqual(pipeline.narrow_phase._terrain_law_partner.shape[0], 0)
        builder = newton.ModelBuilder()
        builder.add_shape_heightfield(heightfield=terrain, cfg=cfg)
        builder.add_shape_box(builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.05))), hx=0.03, hy=0.03, hz=0.03)
        record = _law_record(_pipeline(builder.finalize(device="cpu"), {"NEWTON_TERRAIN_LAW": "point"}, **PIPELINE))
        self.assertEqual((record["writes"], record["decision"]), (False, "no_partner"))
        self.assertIn("of the listed classes", record["reason"])

    def test_barrels_and_finite_planes_are_the_laws_cpu(self):
        """Write the heightfield contacts of barrel cylinders (the cylinder class) and of finite planes on bodies (the box
        class, a box of no thickness) with the law by default; refuse an infinite plane on a body, which meets the
        whole heightfield, at construction (``NEWTON_TERRAIN_LAW=none`` keeps the terrain route for it)."""
        builder = newton.ModelBuilder()
        cfg = builder.ShapeConfig(margin=0.01, gap=0.01)
        terrain = newton.Heightfield(data=np.zeros((9, 9), np.float32), nrow=9, ncol=9, hx=0.4, hy=0.4)
        hfield = builder.add_shape_heightfield(heightfield=terrain, cfg=cfg)
        barrel = builder.add_shape_cylinder(
            builder.add_body(xform=wp.transform(wp.vec3(0.2, 0.0, 0.035))),
            radius=0.03,
            half_height=0.04,
            barrel_radius=0.2,
            cfg=cfg,
        )
        plate = builder.add_shape_plane(
            body=builder.add_body(xform=wp.transform(wp.vec3(-0.2, 0.0, -0.002))), width=0.1, length=0.06, cfg=cfg
        )
        model = builder.finalize(device="cpu")
        pipeline = _pipeline(model, {}, **PIPELINE)
        self.assertEqual(_law_record(pipeline)["decision"], "default")
        partner = pipeline.narrow_phase._terrain_law_partner.numpy()
        self.assertEqual((int(partner[barrel]), int(partner[plate])), (1, 1))
        state = model.state()
        rows = tu.contact_rows(model, state, _collide(pipeline, state), hfield)
        self.assertEqual(len(rows[barrel]), 3)
        self.assertEqual(len(rows[plate]), 4)
        builder.add_shape_plane(body=builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.3, 0.1))), width=0.0, cfg=cfg)
        model = builder.finalize(device="cpu")
        with self.assertRaisesRegex(ValueError, "a plane of infinite extent on body 2, collides with heightfield"):
            _pipeline(model, {}, **PIPELINE)
        self.assertEqual(_law_record(_pipeline(model, TERRAIN_ROUTE, **PIPELINE))["decision"], "opted_out")

    def test_contact_overflow_raises_contacts_cpu(self):
        """Raise the narrow phase's ``contacts`` bit when the law's contacts overflow the contact buffer.

        The narrow phase checks the buffer before the law writes; with every terrain pair the law's, it sees no
        overflow of its own.
        """
        model, _terrain = _mixed_scene(wp.get_device("cpu"))
        state = model.state()
        pipeline = _pipeline(model, {}, max_triangle_pairs=1 << 16, verify_buffers=True)
        # Outside capture a pass grows a buffer it overflows; a graph replay cannot, so it flags.
        _collide(pipeline, state)
        contacts = newton.Contacts(60, 0, device=model.device)
        with wp.ScopedCapture(device=model.device) as capture:
            pipeline.collide(state, contacts)
        wp.capture_launch(capture.graph)
        # The pass keeps the rows the fixed buffer holds; its count keeps every contact found, as Newton's buffers do.
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 60)
        self.assertEqual(contacts.rigid_contact_max, 60)
        self.assertTrue(pipeline.narrow_phase.buffer_capacity_status()["contacts"])
        roomy = _pipeline(model, {}, verify_buffers=True, **PIPELINE)
        _collide(roomy, state)
        self.assertFalse(roomy.narrow_phase.buffer_capacity_status()["contacts"])

    def test_listed_classes_only_cpu(self):
        """With one class listed, only its pairs' contacts are the law's; the other pairs keep the terrain route."""
        model, terrain = _mixed_scene(wp.get_device("cpu"))
        state = model.state()
        today = tu.contact_rows(model, state, _collide(_pipeline(model, TERRAIN_ROUTE, **PIPELINE), state), terrain)
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
for _name, _func in (
    ("test_law_writes_its_rows_as_contacts", test_law_writes_its_rows_as_contacts),
    ("test_default_writes_with_the_law", test_default_writes_with_the_law),
):
    add_function_test(TestTerrainLawContacts, _name, _func, devices=devices)
add_function_test(
    TestTerrainLawContacts,
    "test_law_contacts_replay_under_graph_capture",
    test_law_contacts_replay_under_graph_capture,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
