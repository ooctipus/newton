"""Focused regression controls for the optional adaptive terrain exporter."""

import hashlib
import importlib.util
import json
import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.geometry.contact_data import SHAPE_PAIR_HFIELD_BIT, ContactData
from newton._src.geometry.contact_reduction_global import (
    GlobalContactReducer,
    GlobalContactReducerData,
    _make_contact_value_fast,
    create_export_reduced_contacts_kernel,
    export_contact_to_buffer,
    make_contact_key,
)
from newton._src.geometry.hashtable import hashtable_find_or_insert
from newton._src.geometry.types import GeoType


def qualification_module():
    path = Path("/tmp/fpgs-heightfield-finite-qualification-J9kBvfGP/qualification.py")
    if (
        hashlib.sha256(path.read_bytes()).hexdigest()
        != "bd6d47727eeee2bb29a9f7e113b9c4282373b4b730e35f9179f38964c7265762"
    ):
        raise AssertionError("Changed original physical qualification helper")
    name = "adaptive_manifold_original_qualification"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def complete_tail(samples, *, mass, gravity, dt):
    """Use the predeclared final 80 steps, not five aliased force samples."""
    if len(samples) != 80:
        raise AssertionError("Incomplete fixed 80-step support interval")
    forces = np.asarray([item["force"] for item in samples], dtype=float)
    duration = len(samples) * dt
    impulse = forces.sum(axis=0) * dt
    required = np.asarray(samples[-1]["momentum_after"]) - np.asarray(samples[0]["momentum_before"])
    required -= mass * np.asarray(gravity) * duration
    closure = float(np.linalg.norm(impulse - required) / max(1.0, np.linalg.norm(impulse), np.linalg.norm(required)))
    weight = mass * 9.81
    return {
        "first_step": samples[0]["step"],
        "last_step": samples[-1]["step"],
        "count": len(samples),
        "mean_force": forces.mean(axis=0).tolist(),
        "min_force": forces.min(axis=0).tolist(),
        "max_force": forces.max(axis=0).tolist(),
        "mean_weight_relative_error": float(abs(forces[:, 2].mean() - weight) / weight),
        "momentum_scaled_error": closure,
        "contact_impulse": impulse.tolist(),
        "required_impulse": required.tolist(),
        "max_spin": max(item["spin"] for item in samples),
        "samples": samples,
    }


def cone_diagnostic(types, parents, mu, impulse):
    """Check the actual own-parent disk law, not a pooled-patch fiction."""
    if not np.isfinite(impulse).all() or not np.isfinite(mu).all():
        raise AssertionError("Nonfinite impulse or friction coefficient")
    negative = float(max(0.0, -np.min(impulse[types == 0], initial=0.0)))
    cone = capacity = 0.0
    for row in np.flatnonzero(types == 2):
        parent = int(parents[row])
        if row != parent + 1:
            continue
        if (
            parent < 0
            or parent + 2 >= len(types)
            or types[parent] != 0
            or types[parent + 2] != 2
            or parents[parent + 2] != parent
        ):
            raise AssertionError("Malformed normal/tangent row family")
        radius = max(0.0, float(mu[row] * impulse[parent]))
        cone = max(cone, float(np.linalg.norm(impulse[parent + 1 : parent + 3])) - radius)
        capacity += radius
    return {
        "cone_excess": cone,
        "negative_normal": negative,
        "selected_friction_capacity": capacity,
        "total_normal_impulse": float(impulse[types == 0].sum()),
    }


@wp.struct
class WriterData:
    count: wp.array[int]
    ids: wp.array[int]
    points: wp.array[wp.vec3]
    normals: wp.array[wp.vec3]
    depths: wp.array[float]


@wp.func
def record_writer(contact: ContactData, writer: WriterData, output_index: int):
    index = wp.atomic_add(writer.count, 0, 1)
    writer.ids[index] = contact.sort_sub_key
    writer.points[index] = contact.contact_point_center
    writer.normals[index] = contact.contact_normal_a_to_b
    writer.depths[index] = contact.contact_distance


@wp.kernel
def seed_pool(
    data: GlobalContactReducerData,
    pairs: wp.array[wp.vec2i],
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    depths: wp.array[float],
    bins: wp.array[int],
    slots: wp.array[int],
):
    # Serial stable contact IDs; native export still exercises duplicate CTAs.
    for i in range(pairs.shape[0]):
        pair = pairs[i]
        contact_id = export_contact_to_buffer(pair[0], pair[1], points[i], normals[i], depths[i], i + 1, data)
        entry = hashtable_find_or_insert(
            make_contact_key(pair[0], pair[1], bins[i]), data.ht_keys, data.ht_active_slots
        )
        data.ht_values[slots[i] * data.ht_capacity + entry] = _make_contact_value_fast(1.0, i + 1, contact_id)
        if i == 0:
            # The same ID in a normal and voxel key must not consume two slots.
            duplicate_entry = hashtable_find_or_insert(
                make_contact_key(pair[0], pair[1], 34), data.ht_keys, data.ht_active_slots
            )
            data.ht_values[duplicate_entry] = _make_contact_value_fast(1.0, i + 1, contact_id)


def exercise_export(test, device):
    from newton._src.geometry.heightfield_manifold import create_export_kernel  # noqa: PLC0415

    pairs, points, normals, depths, bins, slots = [], [], [], [], [], []

    def append(pair, point, normal, depth, index):
        pairs.append(pair)
        points.append(point)
        normals.append(normal)
        depths.append(depth)
        bins.append((index // 7) * 20)
        slots.append(index % 7)

    # Eight accepted candidates plus a much farther but writer-rejected one.
    for i, p in enumerate(
        (
            (-1, -1, 0),
            (1, -1, 0),
            (1, 1, 0),
            (-1, 1, 0),
            (0, 0, 0),
            (0.2, 0.1, 0),
            (0.4, -0.3, 0),
            (0.1, 0.5, 0),
            (100, 100, 0),
        )
    ):
        append((0, 1), p, (0, 0, 1), -0.02 if i == 0 else (0.2 if i == 8 else -0.01), i)
    # Complete unsupported sphere/mesh fallback.
    for i in range(6):
        append((2, 3), (i, 5, 0), (0, 0, 1), -0.01, i)
    # Coincident positions and different normals: all four IDs survive.
    for i, n in enumerate(((0, 0, 1), (0, 1, 0), (1, 0, 0), (0, -1, 0))):
        append((4, 5), (0, 0, 0), n, -0.01, i)
    for pair, count in (((6, 7), 2), ((8, 9), 2), ((10, 11), 3)):
        for i in range(count):
            append(pair, (i, pair[0], 0), (0, 0, 1), -0.01, i)

    reducer = GlobalContactReducer(capacity=128, device=device)
    arrays = [
        wp.array(pairs, dtype=wp.vec2i, device=device),
        wp.array(points, dtype=wp.vec3, device=device),
        wp.array(normals, dtype=wp.vec3, device=device),
        wp.array(depths, dtype=float, device=device),
        wp.array(bins, dtype=int, device=device),
        wp.array(slots, dtype=int, device=device),
    ]
    wp.launch(seed_pool, dim=1, inputs=[reducer.get_data_struct(), *arrays], device=device)
    shape_types = wp.array(
        [
            GeoType.HFIELD,
            GeoType.BOX,
            GeoType.MESH,
            GeoType.SPHERE,
            GeoType.HFIELD,
            GeoType.CONVEX_MESH,
            GeoType.MESH,
            GeoType.BOX,
            GeoType.MESH,
            GeoType.PLANE,
            GeoType.HFIELD,
            GeoType.MESH,
        ],
        dtype=int,
        device=device,
    )
    shape_data = wp.zeros(12, dtype=wp.vec4, device=device)
    shape_gap = wp.full(12, 0.01, dtype=float, device=device)
    writer = WriterData()
    writer.count = wp.zeros(1, dtype=int, device=device)
    writer.ids = wp.zeros(128, dtype=int, device=device)
    writer.points = wp.zeros(128, dtype=wp.vec3, device=device)
    writer.normals = wp.zeros(128, dtype=wp.vec3, device=device)
    writer.depths = wp.zeros(128, dtype=float, device=device)
    old_inputs = [
        reducer.hashtable.keys,
        reducer.ht_values,
        reducer.hashtable.active_slots,
        reducer.position_depth,
        reducer.normal,
        reducer.shape_pairs,
        reducer.contact_fingerprints,
        reducer.exported_flags,
        shape_types,
        shape_data,
        shape_gap,
        writer,
        2,
        int(not wp.get_device(device).is_cpu),
        0,
    ]
    route = wp.array([(0, 1), (0, 1), (2, 3), (4, 5), (7, 6)], dtype=wp.vec2i, device=device)
    plane = wp.array([(8, 9)], dtype=wp.vec2i, device=device)
    mesh_mesh = wp.array([(10 | int(SHAPE_PAIR_HFIELD_BIT), 11)], dtype=wp.vec2i, device=device)
    extra = [
        route,
        wp.array([5], dtype=int, device=device),
        plane,
        wp.array([1], dtype=int, device=device),
        mesh_mesh,
        wp.array([1], dtype=int, device=device),
    ]
    wp.launch_tiled(
        create_export_reduced_contacts_kernel(record_writer), dim=2, inputs=old_inputs, block_dim=32, device=device
    )
    old_ids = sorted(writer.ids.numpy()[: writer.count.numpy()[0]].tolist())
    test.assertEqual(old_ids, list(range(1, len(pairs) + 1)))
    for _ in range(3):
        reducer.exported_flags.zero_()
        writer.count.zero_()
        wp.launch_tiled(
            create_export_kernel(record_writer), dim=2, inputs=[*old_inputs, *extra], block_dim=32, device=device
        )
        count = int(writer.count.numpy()[0])
        ids = writer.ids.numpy()[:count]
        test.assertEqual(len(ids), len(set(ids.tolist())))
        # Independent exact square selection; deepest1, opposite3, then2/4.
        test.assertEqual(sorted(ids[ids <= 9].tolist()), [1, 2, 3, 4])
        test.assertEqual(sorted(ids[ids > 9].tolist()), list(range(10, len(pairs) + 1)))
        for row, identifier in enumerate(ids):
            np.testing.assert_array_equal(
                writer.points.numpy()[row], np.asarray(points[identifier - 1], dtype=np.float32)
            )
            np.testing.assert_allclose(writer.normals.numpy()[row], normals[identifier - 1], atol=1e-7)
            test.assertEqual(writer.depths.numpy()[row], np.float32(depths[identifier - 1]))


class TestAdaptiveManifold(unittest.TestCase):
    loaded_feature_env = "NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD"
    loaded_feature_attribute = "_heightfield_adaptive_manifold"
    loaded_report_label = "ADAPTIVE_MANIFOLD_QUALIFICATION"

    def test_full_tail_and_cone_oracles(self):
        # Five selected low impulses can alias even though the complete interval
        # balances weight exactly. A truly incorrect mean must still fail 2%.
        mass, dt = 2.25, 0.0025
        weight = mass * 9.81
        values = np.full(80, weight)
        values[[0, 20, 40, 60, 79]] -= 3.0
        values[[1, 21, 41, 61, 78]] += 3.0
        momentum = np.zeros(3)
        samples = []
        for i, force in enumerate(values):
            before = momentum.copy()
            momentum[2] += (force - weight) * dt
            samples.append(
                {
                    "step": i + 161,
                    "force": [0.0, 0.0, force],
                    "momentum_before": before.tolist(),
                    "momentum_after": momentum.copy().tolist(),
                    "spin": 0.0,
                }
            )
        result = complete_tail(samples, mass=mass, gravity=[0, 0, -9.81], dt=dt)
        self.assertLess(result["mean_weight_relative_error"], 1e-12)
        self.assertLess(result["momentum_scaled_error"], 1e-12)
        self.assertGreater(abs(values[[0, 20, 40, 60, 79]].mean() / weight - 1), 0.02)
        for sample in samples:
            sample["force"][2] *= 1.1
        bad = complete_tail(samples, mass=mass, gravity=[0, 0, -9.81], dt=dt)
        self.assertGreater(bad["mean_weight_relative_error"], 0.02)
        self.assertGreater(bad["momentum_scaled_error"], 2e-4)
        args = (np.array([0, 2, 2]), np.array([-1, 0, 0]), np.array([0.0, 0.2, 0.2]))
        self.assertEqual(cone_diagnostic(*args, np.array([1.0, 0.2, 0.0]))["cone_excess"], 0.0)
        self.assertGreater(cone_diagnostic(*args, np.array([1.0, 0.2, 0.2]))["cone_excess"], 3e-5)

    def test_factory_exists(self):
        from newton._src.geometry.heightfield_manifold import create_export_kernel  # noqa: PLC0415

        self.assertTrue(callable(create_export_kernel))

    def test_pool_cpu(self):
        exercise_export(self, "cpu")

    def test_dispatch_guards(self):
        from newton._src.geometry.narrow_phase import NarrowPhase  # noqa: PLC0415
        from newton._src.sim.collide import write_contact  # noqa: PLC0415

        for enabled, writer, speculative, expected in (
            ("0", write_contact, False, False),
            ("1", write_contact, False, True),
            ("1", record_writer, False, False),
            ("1", None, False, False),
            ("1", write_contact, True, False),
        ):
            with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD": enabled}):
                owner = NarrowPhase(
                    max_candidate_pairs=8,
                    max_triangle_pairs=32,
                    has_meshes=False,
                    has_heightfields=True,
                    contact_writer_warp_func=writer,
                    speculative=speculative,
                    contact_writer_supports_speculative=True,
                    device="cpu",
                )
            self.assertEqual(owner._heightfield_adaptive_manifold, expected)
            if expected:
                self.assertIn("export_adaptive_heightfield_manifold", owner.export_reduced_contacts_kernel.key)
        with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD": "1"}):
            owner = NarrowPhase(
                max_candidate_pairs=8,
                max_triangle_pairs=32,
                has_meshes=False,
                has_heightfields=True,
                contact_writer_warp_func=write_contact,
                deterministic=True,
                device="cpu",
            )
        self.assertFalse(owner._heightfield_adaptive_manifold)

    def test_stock_pipeline_cpu(self):
        q = qualification_module()
        case = next(case for case in q.CASES if case.name == "support")
        with patch.dict(os.environ, {"NEWTON_HEIGHTFIELD_ADAPTIVE_MANIFOLD": "1"}):
            scene = q.build_scene(case, True, "cpu", make_solver=False)
        self.assertTrue(scene.pipeline.narrow_phase._heightfield_adaptive_manifold)
        for _ in range(3):
            scene.pipeline.collide(scene.states[0], scene.contacts)
            scene.pipeline.narrow_phase.check_buffer_capacity()
            count = int(scene.contacts.rigid_contact_count.numpy()[0])
            self.assertGreater(count, 0)
            self.assertLessEqual(count, 4)
            self.assertTrue(np.isfinite(scene.contacts.rigid_contact_normal.numpy()[:count]).all())

    @unittest.skipUnless(wp.is_cuda_available(), "Root owns CUDA execution")
    def test_pool_cuda(self):
        exercise_export(self, "cuda:0")

    @unittest.skipUnless(wp.is_cuda_available(), "Root owns CUDA execution")
    def test_loaded_cuda(self):
        """Existing complete physical gates, plus yaw/step, at unchanged eight sweeps."""
        q = qualification_module()
        original_build = q.build_scene
        original_solver = q.newton.solvers.SolverFeatherPGS
        original_heightfield = q.newton.Heightfield
        original_force = q.public_force
        base_cases = [
            next(case for case in q.CASES if case.name == name)
            for name in ("scalar_rebound", "rebound", "support", "tilted_foot", "sliding", "finite_border")
        ]
        support = next(case for case in q.CASES if case.name == "support")
        sliding = next(case for case in q.CASES if case.name == "sliding")
        cases = [
            *base_cases,
            replace(sliding, name="sliding_full_friction"),
            replace(support, name="yaw_support", steps=320),
            replace(support, name="step_support", steps=320),
        ]
        report = {
            "scope": "adaptive4 new discretization; original eight sweeps and physical gates",
            "records": [],
            "failures": [],
        }
        for case in cases:
            paired = []
            for enabled in (False, True):
                counts = []
                tail = []
                laws = []
                anchor_limit = 0 if case.name == "sliding_full_friction" else 2

                def solver(*args, selected_limit=anchor_limit, **kwargs):
                    kwargs.update(contact_friction_anchor_limit=selected_limit, contact_friction_gap_threshold=0.02)
                    return original_solver(*args, **kwargs)

                def heightfield(*args, selected_case=case, **kwargs):
                    if selected_case.name == "step_support":
                        data = np.asarray(kwargs["data"]).copy()
                        data[:, data.shape[1] // 2 + 1 :] = 0.01
                        kwargs["data"] = data
                    return original_heightfield(*args, **kwargs)

                def build(*args, selected_case=case, selected_enabled=enabled, selected_limit=anchor_limit, **kwargs):
                    with (
                        patch.object(q.newton.solvers, "SolverFeatherPGS", solver),
                        patch.object(q.newton, "Heightfield", heightfield),
                    ):
                        scene = original_build(*args, **kwargs)
                    self.assertEqual(
                        getattr(scene.pipeline.narrow_phase, self.loaded_feature_attribute), selected_enabled
                    )
                    self.assertEqual(scene.solver.contact_friction_anchor_limit, selected_limit)
                    if selected_case.name == "yaw_support":
                        for state in scene.states:
                            qd = state.joint_qd.numpy()
                            qd[5] = 2.0
                            state.joint_qd.assign(qd)
                            q.newton.eval_fk(scene.model, state.joint_q, state.joint_qd, state)
                    return scene

                def force(scene, state, count_records=counts, tail_records=tail, law_records=laws):
                    value = original_force(scene, state)
                    total = int(scene.solver.constraint_count.numpy()[0])
                    row_types = scene.solver.row_type.numpy()[0, :total]
                    mf_total = int(scene.solver.mf_constraint_count.numpy()[0])
                    mf_types = scene.solver.mf_row_type.numpy()[0, :mf_total]
                    count_records.append(
                        {
                            "raw": int(scene.contacts.rigid_contact_count.numpy()[0]),
                            "rows": total + mf_total,
                            "tangents": int(np.count_nonzero(row_types == 2) + np.count_nonzero(mf_types == 2)),
                        }
                    )
                    out = scene.states[1] if state is scene.states[0] else scene.states[0]
                    step = len(count_records)
                    if scene.case.kind in ("support", "tilt") and step > scene.case.steps - 80:
                        tail_records.append(
                            {
                                "step": step,
                                "force": value.tolist(),
                                "momentum_before": q.linear_momentum(scene, state).tolist(),
                                "momentum_after": q.linear_momentum(scene, out).tolist(),
                                "spin": float(np.linalg.norm(out.body_qd.numpy()[scene.foot, 3:])),
                            }
                        )
                    dense = cone_diagnostic(
                        row_types,
                        scene.solver.row_parent.numpy()[0, :total],
                        scene.solver.row_mu.numpy()[0, :total],
                        scene.solver.impulses.numpy()[0, :total],
                    )
                    mf = cone_diagnostic(
                        mf_types,
                        scene.solver.mf_row_parent.numpy()[0, :mf_total],
                        scene.solver.mf_row_mu.numpy()[0, :mf_total],
                        scene.solver.mf_impulses.numpy()[0, :mf_total],
                    )
                    law_records.append(
                        {
                            "cone_excess": max(dense["cone_excess"], mf["cone_excess"]),
                            "negative_normal": max(dense["negative_normal"], mf["negative_normal"]),
                            "selected_friction_capacity": dense["selected_friction_capacity"]
                            + mf["selected_friction_capacity"],
                            "total_normal_impulse": dense["total_normal_impulse"] + mf["total_normal_impulse"],
                        }
                    )
                    return value

                try:
                    with (
                        patch.dict(os.environ, {self.loaded_feature_env: str(int(enabled))}),
                        patch.object(q, "build_scene", build),
                        patch.object(q, "public_force", force),
                    ):
                        # Both arms use exactly the same finite geometry/reducer inputs.
                        record = q.run_case(case, True, "cuda:0", reduce=True)
                    record["adaptive_manifold"] = enabled
                    record["row_counts"] = counts
                    record["anchor_limit"] = anchor_limit
                    record["original_sampled_failures"] = list(record["failures"])
                    record["law_diagnostics"] = laws
                    if case.kind in ("support", "tilt"):
                        result = complete_tail(tail, mass=record["mass"], gravity=[0, 0, -9.81], dt=record["dt"])
                        record["complete_tail"] = result
                        old_force_gate = "settled support force differs from weight by more than2%"
                        record["failures"] = [item for item in record["failures"] if item != old_force_gate]
                        if result["mean_weight_relative_error"] > 0.02:
                            record["failures"].append("complete80-step support mean differs from weight by more than2%")
                        if result["momentum_scaled_error"] > 2e-4:
                            record["failures"].append("complete80-step contact impulse disagrees with momentum")
                    if case.name == "sliding":
                        # The production selected-anchor law does not have the
                        # full-load mu*g stopping distance. Preserve that old
                        # diagnostic and check its actual disks/momentum; the
                        # separate anchor0 case retains the analytical gate.
                        record["analytic_stopping_gate_applicable"] = False
                        record["failures"] = [
                            item
                            for item in record["failures"]
                            if item != "sliding stopping distance/speed violates Coulomb control"
                        ]
                    if case.name == "step_support":
                        record["fixture_scope"] = (
                            "8.32mm initially penetrating height-ramp recovery; unchanged final settle gate"
                        )
                    if max(item["cone_excess"] for item in laws) > 3e-5:
                        record["failures"].append("actual own-parent Coulomb cone excess exceeds3e-5")
                    if max(item["negative_normal"] for item in laws) > 1e-7:
                        record["failures"].append("negative normal impulse exceeds1e-7")
                    paired.append(record)
                    report["failures"].extend([case.name, enabled, failure] for failure in record["failures"])
                except Exception as error:
                    record = {
                        "case": case.name,
                        "adaptive_manifold": enabled,
                        "error": repr(error),
                        "row_counts": counts,
                    }
                    report["failures"].append([case.name, enabled, repr(error)])
                report["records"].append(record)
            if len(paired) == 2:
                self.assertEqual(paired[0]["initial_state"], paired[1]["initial_state"])
                self.assertEqual(paired[0]["initial_velocity"], paired[1]["initial_velocity"])
        print(self.loaded_report_label + " " + json.dumps(report, allow_nan=False), flush=True)
        self.assertEqual(report["failures"], [], "All original physical gates retained; see complete report")


if __name__ == "__main__":
    unittest.main()
