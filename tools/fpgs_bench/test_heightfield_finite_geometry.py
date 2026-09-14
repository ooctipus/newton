# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent finite witnesses and complete saved-current G1 pipeline checks.

The 96-pair fixtures are local, pinned diagnostic inputs, not task throughput or
trajectory acceptance. Contact manifolds may change; finite geometry may not.
"""

import hashlib
import json
import os
import unittest
from collections import Counter
from functools import cache
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton

CAPTURE = Path("/tmp/fpgs-g1-geometry-paired16k-20260913-01")
PINS = (
    "09a540b0e88f3411cb26e8cbf6288ea89f521fa552ec939870795b4debd0952d",
    "4751dd0fc1d1dddaccc66087df73c36b8f356173be1a79516ad17395fba60eb0",
)
RECORDS = []


def rotation(q):
    """Independent normalized rigid rotation, with xyzw input."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    v = np.eye(3)
    return (v + 2 * np.cross(q[:3], np.cross(q[:3], v) + q[3] * v)).T


def triangle_box_distance(tri, center, rot, half):
    """Independent convex quadratic program, not native feature enumeration."""
    from scipy.optimize import minimize

    local = (tri - center) @ rot
    edges = (local[1:] - local[0]).T

    def objective(x):
        delta = local[0] + edges @ x[:2] - x[2:]
        return delta @ delta

    def jacobian(x):
        delta = local[0] + edges @ x[:2] - x[2:]
        return np.r_[2 * edges.T @ delta, -2 * delta]

    result = minimize(
        objective,
        np.r_[1 / 3, 1 / 3, np.clip(local.mean(axis=0), -half, half)],
        jac=jacobian,
        bounds=[(0, 1), (0, 1), *zip(-half, half, strict=True)],
        constraints={
            "type": "ineq",
            "fun": lambda x: 1 - x[0] - x[1],
            "jac": lambda x: np.array([-1.0, -1.0, 0.0, 0.0, 0.0]),
        },
        method="SLSQP",
        options={"ftol": 1e-13, "maxiter": 200},
    )
    if not result.success:
        raise AssertionError("Independent distance control failed: " + result.message)
    return float(np.sqrt(max(0.0, result.fun)))


def audit_query(result, tri, center, rot, half, tolerance=3e-5):
    """Check finite supported witnesses; allow a different physical manifold."""
    result = np.asarray(result, dtype=float)
    if not np.isfinite(result).all():
        raise AssertionError("Nonfinite query output")
    count = int(result[0])
    if count == -1:
        return {"fallback": True}
    if count != result[0] or not 1 <= count <= 5:
        raise AssertionError("Invalid finite contact cardinality")
    normal = result[1:4]
    if abs(np.linalg.norm(normal) - 1) > 2e-5 or normal[2] < -2e-5:
        raise AssertionError("Invalid finite top/prism normal")
    edges = (tri[1:] - tri[0]).T
    errors, distances = [], []
    for k in range(count):
        midpoint, distance = result[4 + 4 * k : 7 + 4 * k], result[7 + 4 * k]
        a, b = midpoint - 0.5 * distance * normal, midpoint + 0.5 * distance * normal
        uv = np.linalg.lstsq(edges, a - tri[0], rcond=None)[0]
        plane_error = np.linalg.norm(a - tri[0] - edges @ uv)
        # Convert barycentric edge excursions to metres, not unitless tolerance.
        edge_error = max(0.0, -uv[0], -uv[1], uv.sum() - 1) * max(np.linalg.norm(edges, axis=0))
        local = rot.T @ (b - center)
        box_outside = max(0.0, float(np.max(np.abs(local) - half)))
        box_surface = float(np.min(np.abs(np.abs(local) - half)))
        errors.append(max(plane_error, edge_error, box_outside, box_surface))
        distances.append(float(distance))
    if max(errors) > tolerance:
        raise AssertionError(("Witness off finite surfaces", max(errors), count, result.tolist()))
    closest = triangle_box_distance(tri, center, rot, half)
    if min(distances) >= 0 and abs(min(distances) - closest) > tolerance:
        raise AssertionError(("Wrong separated minimum", min(distances), closest))
    # Deep finite-prism overlap can have a separated top triangle. SAT/side/
    # bottom negatives live in the independent synthetic query tests; top
    # triangle distance alone cannot establish or reject prism penetration.
    return {
        "fallback": False,
        "count": count,
        "surface_error": float(max(errors)),
        "minimum_distance": min(distances),
        "top_triangle_distance": closest,
    }


def current_fixture(gpu, device):
    """Reproduce the exact existing stratified 96-pair selection and materials."""
    path = CAPTURE / f"gpu{gpu}/audit.geometry.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINS[gpu]:
        raise RuntimeError("Changed saved current geometry")
    z = dict(np.load(path, allow_pickle=False))
    pairs = z["mesh_pairs"][np.argsort(z["mesh_pairs"][:, 1])]
    selected = pairs[np.linspace(0, len(pairs) - 1, 96, dtype=np.int64)]
    terrain = int(selected[0, 0])
    hfd = z["heightfield_data"][z["shape_heightfield_index"][terrain]]
    heightfield = newton.Heightfield(
        z["heightfield_elevations"].reshape(int(hfd["nrow"]), int(hfd["ncol"])),
        nrow=int(hfd["nrow"]),
        ncol=int(hfd["ncol"]),
        hx=float(hfd["hx"]),
        hy=float(hfd["hy"]),
        min_z=float(hfd["min_z"]),
        max_z=float(hfd["max_z"]),
    )
    mesh = newton.Mesh(z["hull_0_vertices"], z["hull_0_indices"])
    builder = newton.ModelBuilder()

    def cfg(i):
        return builder.ShapeConfig(
            margin=float(z["shape_margin"][i]),
            gap=float(z["shape_gap"][i]),
            mu=float(z["shape_material_mu"][i]),
            restitution=float(z["shape_material_restitution"][i]),
        )

    pose = z["geom_transform"][terrain]
    builder.add_shape_heightfield(heightfield=heightfield, xform=wp.transform(pose[:3], pose[3:]), cfg=cfg(terrain))
    for _, shape in selected:
        pose = z["geom_transform"][shape]
        body = builder.add_body(xform=wp.transform(pose[:3], pose[3:]))
        builder.add_shape_convex_hull(body=body, mesh=mesh, scale=z["shape_scale"][shape], cfg=cfg(shape))
    model = builder.finalize(device=device)
    np.testing.assert_array_equal(model.heightfield_elevations.numpy(), z["heightfield_elevations"])
    filtered = wp.array([[0, i + 1] for i in range(96)], dtype=wp.vec2i, device=device)
    return model, model.state(), filtered


@cache
def stream_query_kernel(*, analytic_manifold=None):
    """Use the production ABI on the exact decoded current triangle stream."""
    from newton._src.geometry.heightfield_finite import (  # noqa: PLC0415
        ANALYTIC_MANIFOLD,
        QueryResult,
        query_contacts,
        query_top_face_witness,
    )
    from newton._src.utils.heightfield import HeightfieldData, get_triangle_shape_from_heightfield  # noqa: PLC0415

    if analytic_manifold is None:
        analytic_manifold = ANALYTIC_MANIFOLD

    @wp.kernel(enable_backward=False, module="unique")
    def evaluate(
        triples: wp.array[wp.vec3i],
        transforms: wp.array[wp.transform],
        hfd: wp.array[HeightfieldData],
        elevations: wp.array[float],
        scales: wp.array[wp.vec3],
        gaps: wp.array[float],
        margins: wp.array[float],
        output: wp.array[QueryResult],
        vertices: wp.array2d[wp.vec3],
        centers: wp.array[wp.vec3],
        quaternions: wp.array[wp.quat],
    ):
        i = wp.tid()
        triple = triples[i]
        tri, origin = get_triangle_shape_from_heightfield(hfd[0], elevations, transforms[0], triple[2])
        inverse = wp.transform_inverse(transforms[0])
        center = wp.transform_vector(inverse, wp.transform_get_translation(transforms[triple[1]]) - origin)
        quat = wp.quat_inverse(wp.transform_get_rotation(transforms[0])) * wp.transform_get_rotation(
            transforms[triple[1]]
        )
        threshold = gaps[0] + gaps[triple[1]] + margins[0] + margins[triple[1]]
        output[i] = query_contacts(
            tri.scale,
            tri.auxiliary,
            center,
            quat,
            scales[triple[1]] * 0.5,
            threshold,
            margins[0] + margins[triple[1]],
        )
        if wp.static(analytic_manifold):
            certified, witness = query_top_face_witness(
                tri.scale,
                tri.auxiliary,
                center,
                quat,
                scales[triple[1]] * 0.5,
                threshold,
                margins[0] + margins[triple[1]],
            )
            if certified:
                # Audit the closest witness and logical ownership. The full
                # original manifold is checked separately at public output.
                value = QueryResult()
                value[0] = 1.0
                middle = 0.5 * (witness.point_a + witness.point_b)
                for k in range(3):
                    value[k + 1] = witness.normal[k]
                    value[k + 4] = middle[k]
                value[7] = witness.signed_distance
                output[i] = value
        vertices[i, 0] = wp.vec3()
        vertices[i, 1] = tri.scale
        vertices[i, 2] = tri.auxiliary
        centers[i] = center
        quaternions[i] = quat

    return evaluate, QueryResult


@cache
def synthetic_query_kernel():
    """Exercise native FP32 CUDA arithmetic independently of the pipeline caller."""
    from newton._src.geometry.heightfield_finite import QueryResult, query  # noqa: PLC0415

    @wp.kernel(enable_backward=False, module="unique")
    def evaluate(
        e1: wp.array[wp.vec3],
        e2: wp.array[wp.vec3],
        center: wp.array[wp.vec3],
        quat: wp.array[wp.quat],
        half: wp.array[wp.vec3],
        output: wp.array[QueryResult],
    ):
        i = wp.tid()
        output[i] = query(e1[i], e2[i], center[i], quat[i], half[i], 0.04)

    return evaluate, QueryResult


def check_synthetic_native(device):
    """Cover finite edge, zero gap, rotation, overlap and required generic fallback."""
    q = (0.0, float(np.sin(0.26)), 0.0, float(np.cos(0.26)))
    identity = (0.0, 0.0, 0.0, 1.0)
    # e1, e2, center, quaternion, box half extent, expected generic fallback
    cases = [
        ((2, 0, 0), (0, 2, 0), (0.4, 0.4, 0.13), identity, (0.1, 0.1, 0.1), False),
        ((2, 0, 0), (0, 2, 0), (0.4, 0.4, -0.6), q, (0.1, 0.1, 0.1), False),
        ((2, 0, 0), (0, 2, 0), (0.4, 0.4, -0.02), q, (0.1, 0.1, 0.1), False),
        ((2, 0, 0), (0, 2, 0), (0.4, 0.4, 0.1), identity, (0.1, 0.1, 0.1), False),
        ((2, 0, 0), (0, 2, 0), (2, 2, 0.13), identity, (0.1, 0.1, 0.1), False),
        ((0.98, 0.5, -0.98), (0, 1, 0), (-1.02, 0.5, 1.02), identity, (1, 1, 1), False),
        ((2, 0, 0), (0, 2, 0), (0.4, 0.4, 0.16), tuple(1.00001 * np.array(q)), (0.1, 0.1, 0.1), False),
        ((1, 0, 1), (0, 1, 0), (-0.03, 0.3, -0.2), identity, (0.01, 0.01, 0.01), True),
        ((2, 0, 0), (0, 2, 0), (0.4, 0.4, -1.3), identity, (0.1, 0.1, 0.1), True),
        ((0, 0, 0), (0, 2, 0), (0.4, 0.4, 0.1), identity, (0.1, 0.1, 0.1), True),
        ((0, 2, 0), (2, 0, 0), (0.4, 0.4, 0.1), identity, (0.1, 0.1, 0.1), True),
    ]
    kernel, dtype = synthetic_query_kernel()
    inputs = [
        wp.array([case[k] for case in cases], dtype=wp.quat if k == 3 else wp.vec3, device=device) for k in range(5)
    ]
    output = wp.empty(len(cases), dtype=dtype, device=device)
    wp.launch(kernel, dim=len(cases), inputs=inputs, outputs=[output], device=device)
    reports = []
    for index, (case, result) in enumerate(zip(cases, output.numpy(), strict=True)):
        e1, e2, center, quat, half, expected_fallback = case
        report = audit_query(
            result, np.array([[0.0, 0.0, 0.0], e1, e2]), np.array(center), rotation(quat), np.array(half)
        )
        if report["fallback"] != expected_fallback:
            raise AssertionError(("Wrong synthetic admission", index, result.tolist()))
        reports.append({"case": index, **report})
    return {"scope": "Native synthetic finite geometry", "device": str(device), "records": reports}


def snapshot(pipeline, state, contacts):
    """Read and own current output bytes only at an untimed test boundary."""
    pipeline.narrow_phase.check_buffer_capacity()
    n = int(contacts.rigid_contact_count.numpy()[0])
    nt = int(pipeline.narrow_phase.triangle_pairs_count.numpy()[0])
    if not 0 <= n <= contacts.rigid_contact_max or not 0 <= nt <= pipeline.narrow_phase.triangle_pairs.shape[0]:
        raise AssertionError("Truncated diagnostic buffer")
    triples = pipeline.narrow_phase.triangle_pairs.numpy()[:nt].copy()
    marked = triples[:, 2] < 0
    triples[marked, 2] = ~triples[marked, 2]
    p0, p1 = (getattr(contacts, "rigid_contact_point" + side).numpy()[:n].copy() for side in ("0", "1"))
    shape0 = contacts.rigid_contact_shape0.numpy()[:n]
    shapes = contacts.rigid_contact_shape1.numpy()[:n].copy()
    poses = state.body_q.numpy()
    if np.any(shape0 != 0) or np.any(shapes < 1) or np.any(shapes > len(poses)):
        raise AssertionError("Invalid public contact shape routing")
    world1 = np.array(
        [poses[s - 1, :3] + rotation(poses[s - 1, 3:]) @ v for s, v in zip(shapes, p1, strict=True)]
    ).reshape((-1, 3))
    normal = contacts.rigid_contact_normal.numpy()[:n].copy()
    distance = np.einsum("ij,ij->i", world1 - p0, normal)
    if not all(np.isfinite(v).all() for v in (p0, world1, normal, distance)):
        raise AssertionError("Nonfinite public contacts")
    if n and np.max(np.abs(np.linalg.norm(normal, axis=1) - 1.0)) > 2e-4:
        raise AssertionError("Invalid public contact normals")
    return {
        "triples": triples,
        "marked_triples": triples[marked].copy(),
        "marked": int(marked.sum()),
        "shape": shapes,
        "point": p0,
        "normal": normal,
        "distance": distance,
        "count": n,
    }


def logical_keys(triples):
    """Sort the decoded logical multiset without depending on atomic output order."""
    return sorted(tuple(int(value) for value in row) for row in triples)


def key_comparison(before, after):
    """Record exact multiset identity with bounded mismatch examples."""
    left, right = Counter(logical_keys(before)), Counter(logical_keys(after))
    missing, added = list((left - right).elements()), list((right - left).elements())
    return {
        "equal": left == right,
        "missing_count": len(missing),
        "added_count": len(added),
        "missing_examples": missing[:8],
        "added_examples": added[:8],
    }


def summarize_snapshot(value):
    """Keep counts, exact-key digests and all per-shape minima in the diagnostic."""

    def digest(rows):
        return hashlib.sha256(np.asarray(logical_keys(rows), dtype=np.int32).tobytes()).hexdigest()

    shapes = sorted(int(shape) for shape in np.unique(value["shape"]))
    return {
        "contact_count": value["count"],
        "marked_count": value["marked"],
        "triangle_count": len(value["triples"]),
        "logical_multiset_sha256": digest(value["triples"]),
        "marked_logical_keys_sha256": digest(value["marked_triples"]),
        "contacting_shapes": shapes,
        "minimum_separation_by_shape": {
            str(shape): float(value["distance"][value["shape"] == shape].min()) for shape in shapes
        },
    }


def compare_snapshots(before, after, *, compare_marked=True):
    """Separate legacy cardinality diagnostics from unchanged ownership/physics gates."""
    logical = key_comparison(before["triples"], after["triples"])
    marked = key_comparison(before["marked_triples"], after["marked_triples"])
    left, right = set(map(int, before["shape"])), set(map(int, after["shape"]))
    missing, added = sorted(left - right), sorted(right - left)
    changes = {
        str(shape): abs(
            float(before["distance"][before["shape"] == shape].min())
            - float(after["distance"][after["shape"] == shape].min())
        )
        for shape in sorted(left & right)
    }
    maximum = max(changes.values(), default=0.0)
    failures = []
    if not logical["equal"]:
        failures.append("logical_multiset")
    if compare_marked and not marked["equal"]:
        failures.append("marked_logical_keys")
    if missing:
        failures.append("lost_contacting_shapes")
    if maximum > 2e-4:
        failures.append("minimum_separation")
    return {
        "legacy_count_gate_pass": before["count"] == after["count"] and before["marked"] == after["marked"],
        "contact_count_delta": after["count"] - before["count"],
        "marked_count_delta": after["marked"] - before["marked"],
        "logical_multiset": logical,
        "marked_logical_keys": marked,
        "marked_identity_required": compare_marked,
        "missing_contacting_shapes": missing,
        "added_contacting_shapes": added,
        "per_shape_minimum_separation_change": changes,
        "maximum_minimum_separation_change": maximum,
        "hard_failures": failures,
    }


def collect_query_audit(pipeline, model, initial, device):
    """Reach every native query/independent QP even after a lifecycle-count mismatch."""
    triples = wp.array(initial["triples"], dtype=wp.vec3i, device=device)
    kernel, result_type = stream_query_kernel()
    count = len(initial["triples"])
    output = wp.empty(count, dtype=result_type, device=device)
    vertices = wp.empty((count, 3), dtype=wp.vec3, device=device)
    centers = wp.empty(count, dtype=wp.vec3, device=device)
    quaternions = wp.empty(count, dtype=wp.quat, device=device)
    wp.launch(
        kernel,
        dim=count,
        inputs=[
            triples,
            pipeline.geom_transform,
            model.heightfield_data,
            model.heightfield_elevations,
            model.shape_scale,
            model.shape_gap,
            model.shape_margin,
        ],
        outputs=[output, vertices, centers, quaternions],
        device=device,
    )
    half = model.shape_scale.numpy()[initial["triples"][:, 1]] * 0.5
    values = output.numpy()
    reports, errors = [], []
    for index, (result, tri, center, quat, h) in enumerate(
        zip(values, vertices.numpy(), centers.numpy(), quaternions.numpy(), half, strict=True)
    ):
        try:
            reports.append(audit_query(result, tri, center, rotation(quat), h))
        except Exception as error:
            errors.append({"index": index, "logical_key": initial["triples"][index].tolist(), "error": repr(error)})
    admitted = [item for item in reports if not item["fallback"]]
    marker_match = key_comparison(initial["marked_triples"], initial["triples"][values[:, 0] >= 0])
    failures = (["native_query_geometry"] if errors else []) + ([] if marker_match["equal"] else ["query_marker_keys"])
    return {
        "query_cases": count,
        "query_cases_audited": len(reports) + len(errors),
        "admitted": len(admitted),
        "maximum_surface_error": max((item["surface_error"] for item in admitted), default=0.0),
        "query_error_count": len(errors),
        "query_error_examples": errors[:12],
        "query_marker_keys": marker_match,
        "hard_failures": failures,
    }


def check_current_pipeline(gpu, device):
    """Collect every variant/lifecycle case, preserving legacy failures as diagnostics."""
    model, state, pairs = current_fixture(gpu, device)
    original_pose = state.body_q.numpy().copy()
    records, comparisons, failures = [], [], []
    initial_by_variant = {}
    for reduce in (False, True):
        for enabled in (False, True):
            case = {"reduce": reduce, "enabled": enabled, "snapshots": [], "hard_failures": []}
            records.append(case)
            try:
                state.body_q.assign(original_pose)
                with patch.dict(
                    os.environ, NEWTON_HEIGHTFIELD_CELL_REJECT="1", NEWTON_HEIGHTFIELD_FINITE_QUERY=str(int(enabled))
                ):
                    pipeline = newton.CollisionPipeline(
                        model,
                        shape_pairs_filtered=pairs,
                        reduce_contacts=reduce,
                        rigid_contact_max=32768,
                        max_triangle_pairs=32768,
                    )
                contacts = pipeline.contacts()
                pipeline.collide(state, contacts)
                initial = snapshot(pipeline, state, contacts)
                initial_by_variant[reduce, enabled] = initial
                case["snapshots"].append({"stage": "initial", **summarize_snapshot(initial)})
                if (enabled and initial["marked"] == 0) or (not enabled and initial["marked"] != 0):
                    case["hard_failures"].append("initial_activation")

                # The original baseline now undergoes exactly the same transition.
                away = original_pose.copy()
                away[:, 2] += 1000.0
                state.body_q.assign(away)
                pipeline.collide(state, contacts)
                empty = snapshot(pipeline, state, contacts)
                case["snapshots"].append({"stage": "empty", **summarize_snapshot(empty)})
                if empty["count"] or len(empty["triples"]) or empty["marked"]:
                    case["hard_failures"].append("empty_prefix_or_contacts")
                state.body_q.assign(original_pose)
                pipeline.collide(state, contacts)
                restored = snapshot(pipeline, state, contacts)
                comparison = compare_snapshots(initial, restored)
                case["snapshots"].append(
                    {"stage": "restored", **summarize_snapshot(restored), "vs_initial": comparison}
                )
                case["hard_failures"].extend("restored:" + error for error in comparison["hard_failures"])

                if device != "cpu":
                    with wp.ScopedCapture(device=device) as captured:
                        pipeline.collide(state, contacts)
                    for index in range(3):
                        wp.capture_launch(captured.graph)
                        replay = snapshot(pipeline, state, contacts)
                        comparison = compare_snapshots(initial, replay)
                        case["snapshots"].append(
                            {"stage": f"replay{index}", **summarize_snapshot(replay), "vs_initial": comparison}
                        )
                        case["hard_failures"].extend(f"replay{index}:" + error for error in comparison["hard_failures"])
                if enabled and not reduce:
                    case["query_audit"] = collect_query_audit(pipeline, model, initial, device)
                    case["hard_failures"].extend(case["query_audit"]["hard_failures"])
            except Exception as error:
                case["collection_error"] = repr(error)
                case["hard_failures"].append("collection_error")
            finally:
                state.body_q.assign(original_pose)
            failures.extend(f"reduce{int(reduce)}_finite{int(enabled)}:{error}" for error in case["hard_failures"])

        if all((reduce, enabled) in initial_by_variant for enabled in (False, True)):
            comparison = compare_snapshots(
                initial_by_variant[reduce, False], initial_by_variant[reduce, True], compare_marked=False
            )
            comparisons.append({"reduce": reduce, **comparison})
            failures.extend(f"variant_reduce{int(reduce)}:{error}" for error in comparison["hard_failures"])
        else:
            failures.append(f"variant_reduce{int(reduce)}:missing_initial")
    return {
        "gpu_fixture": gpu,
        "device": str(device),
        "input_sha256": PINS[gpu],
        "records": records,
        "variant_comparisons": comparisons,
        "hard_failures": failures,
        "legacy_count_failure_count": sum(
            not item["vs_initial"]["legacy_count_gate_pass"]
            for case in records
            for item in case["snapshots"]
            if "vs_initial" in item
        ),
        "scope": "Complete lifecycle diagnostic; historical exact-count failures retained",
    }


class TestIndependentGeometry(unittest.TestCase):
    def test_lifecycle_collector_controls(self):
        """Reject stale ownership/lost shapes while recording benign count variation."""
        initial = {
            "triples": np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
            "marked_triples": np.array([[0, 1, 2]], dtype=np.int32),
            "marked": 1,
            "shape": np.array([1, 2]),
            "distance": np.array([0.01, -0.02]),
            "count": 2,
        }
        changed = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in initial.items()}
        changed.update(shape=np.array([2, 1, 1]), distance=np.array([-0.02, 0.01, 0.02]), count=3)
        changed["triples"] = changed["triples"][::-1]
        report = compare_snapshots(initial, changed)
        self.assertFalse(report["legacy_count_gate_pass"])
        self.assertEqual(report["hard_failures"], [])
        for field, value, failure in (
            ("marked_triples", np.array([[0, 2, 3]], dtype=np.int32), "marked_logical_keys"),
            ("triples", np.array([[0, 1, 2], [0, 2, 4]], dtype=np.int32), "logical_multiset"),
            ("shape", np.array([1, 1, 1]), "lost_contacting_shapes"),
            ("distance", np.array([-0.021, 0.01, 0.02]), "minimum_separation"),
        ):
            injected = dict(changed)
            injected[field] = value
            self.assertIn(failure, compare_snapshots(initial, injected)["hard_failures"])

    def test_surface_checker_rejects_border_and_sign_errors(self):
        """Fail malformed finite witnesses rather than accepting finite numbers alone."""
        tri = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        center, half = np.array([0.4, 0.4, 0.2]), np.full(3, 0.1)
        result = np.zeros(24)
        result[:8] = [1, 0, 0, 1, 0.4, 0.4, 0.05, 0.1]
        self.assertFalse(audit_query(result, tri, center, np.eye(3), half)["fallback"])
        for changed in ((4, 3.0), (3, -1.0), (7, -0.1), (0, 6.0)):
            bad = result.copy()
            bad[changed[0]] = changed[1]
            with self.assertRaises(AssertionError):
                audit_query(bad, tri, center, np.eye(3), half)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires the root-owned paired GPU lease")
    def test_actual96_complete_pipeline_and_query(self):
        """Check both saved GPU scenes through each complete current CUDA pipeline."""
        failures = []
        for gpu in (0, 1):
            try:
                record = check_current_pipeline(gpu, "cuda:0")
            except Exception as error:
                record = {"gpu_fixture": gpu, "hard_failures": ["fixture_collection"], "error": repr(error)}
            RECORDS.append(record)
            failures.extend(f"fixture{gpu}:{error}" for error in record["hard_failures"])
        print("FINITE_LIFECYCLE " + json.dumps(RECORDS, allow_nan=False), flush=True)
        self.assertEqual(failures, [], "All cases collected; see FINITE_LIFECYCLE and current_geometry")

    def test_synthetic_native_cpu(self):
        """Check actual native CPU geometry against independent physical witnesses."""
        check_synthetic_native("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires the root-owned paired GPU lease")
    def test_synthetic_native_cuda(self):
        """Exercise finite edge/penetration/fallback arithmetic on the owned GPU."""
        RECORDS.append(check_synthetic_native("cuda:0"))

    @unittest.skipUnless(wp.is_cuda_available(), "Requires the root-owned paired GPU lease")
    def test_reduced_loaded_flat_seam_cuda(self):
        """Collect reduced loaded seam cases under the unchanged eight-sweep physical checks."""
        import importlib.util  # noqa: PLC0415
        import sys  # noqa: PLC0415

        from newton._src.geometry.heightfield_features import WELD_FLAT_SEAMS  # noqa: PLC0415

        if os.environ.get("NEWTON_HEIGHTFIELD_WELD_FLAT_SEAMS") != "1":
            self.skipTest("Requires explicit flat-seam activation before importing Newton")
        self.assertTrue(WELD_FLAT_SEAMS, "The imported query must actually enable flat-seam filtering")
        path = Path("/tmp/fpgs-heightfield-finite-qualification-J9kBvfGP/qualification.py")
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "bd6d47727eeee2bb29a9f7e113b9c4282373b4b730e35f9179f38964c7265762",
            "The original qualification and physical failures must remain unchanged",
        )
        q = sys.modules.get("qualification")
        if q is None:
            spec = importlib.util.spec_from_file_location("qualification", path)
            q = importlib.util.module_from_spec(spec)
            sys.modules["qualification"] = q
            spec.loader.exec_module(q)
        self.assertEqual(Path(q.__file__).resolve(), path.resolve(), "Mixed qualification helper import")
        report = {
            "scope": "Reduced support/tilt/sliding/border at original eight sweeps; not trajectory or timing acceptance",
            "records": [],
            "failures": [],
            "immutable_hull_only": True,
            "weld_flat_seams": True,
        }
        # Select this method after the separate rebound diagnostic to retain
        # that earlier readback in the unchanged runner's final report.
        if hasattr(q, "QUALIFICATION_REPORT"):
            report["prior_qualification"] = q.QUALIFICATION_REPORT
        q.QUALIFICATION_REPORT = report

        def support_tail_diagnostic(record, tail):
            """Compare every actual tail impulse without replacing the original sampled gate."""
            result = {"steps": [161, 240], "samples": tail, "complete_80_steps": len(tail) == 80}
            if not result["complete_80_steps"]:
                return result
            forces = np.asarray([item["force"] for item in tail])
            weight = record["mass"] * 9.81
            dt = record["dt"]
            duration = len(tail) * dt
            before = np.asarray(tail[0]["momentum_before"])
            after = np.asarray(tail[-1]["momentum_after"])
            contact_impulse = forces.sum(axis=0) * dt
            required_impulse = after - before + np.array([0.0, 0.0, weight * duration])
            closure = float(
                np.linalg.norm(contact_impulse - required_impulse)
                / max(1.0, np.linalg.norm(contact_impulse), np.linalg.norm(required_impulse))
            )
            force_departure = np.abs(forces[:, 2] - weight) > 0.02 * weight
            longest = current = 0
            for departed in force_departure:
                current = current + 1 if departed else 0
                longest = max(longest, current)
            result.update(
                duration_s=duration,
                weight_n=weight,
                mean_force_n=forces.mean(axis=0).tolist(),
                mean_vertical_relative_error=float(abs(forces[:, 2].mean() - weight) / weight),
                mean_within_original_2_percent=bool(abs(forces[:, 2].mean() - weight) <= 0.02 * weight),
                contact_impulse_ns=contact_impulse.tolist(),
                required_impulse_ns=required_impulse.tolist(),
                momentum_scaled_error=closure,
                momentum_within_original_2e_4=bool(closure <= 2e-4),
                force_departure_total_s=float(force_departure.sum() * dt),
                force_departure_longest_s=longest * dt,
                three_contact_total_s=sum(item["contacts"] == 3 for item in tail) * dt,
                peak_spin_rad_s=max(float(np.linalg.norm(item["twist"][3:])) for item in tail),
                original_five_sample_mean_n=float(np.mean([item["force"][2] for item in record["samples"][-5:]])),
                original_failures=list(record["failures"]),
            )
            return result

        for name in ("support", "tilted_foot", "sliding", "finite_border"):
            case = next(case for case in q.CASES if case.name == name)
            pair = []
            for enabled in (False, True):
                tail = []
                step = 0
                original_public_force = q.public_force

                def observe_support_force(scene, state, original=original_public_force, samples=tail):
                    """Read the just-solved output bank after the one original force publication."""
                    nonlocal step
                    force = original(scene, state)
                    step += 1
                    if scene.case.name == "support" and 161 <= step <= 240:
                        out = scene.states[1] if state is scene.states[0] else scene.states[0]
                        samples.append(
                            {
                                "step": step,
                                "force": force.tolist(),
                                "contacts": int(scene.contacts.rigid_contact_count.numpy()[0]),
                                "pose": out.body_q.numpy()[scene.foot].astype(float).tolist(),
                                "twist": out.body_qd.numpy()[scene.foot].astype(float).tolist(),
                                "momentum_before": q.linear_momentum(scene, state).tolist(),
                                "momentum_after": q.linear_momentum(scene, out).tolist(),
                            }
                        )
                    return force

                try:
                    with patch.object(q, "public_force", observe_support_force):
                        record = q.run_case(case, enabled, "cuda:0", reduce=True)
                    if name == "support":
                        record["support_tail_diagnostic"] = support_tail_diagnostic(record, tail)
                    pair.append(record)
                    report["failures"].extend([name, enabled, error] for error in record["failures"])
                except Exception as error:  # Retain all eight cases before reporting failure.
                    record = {"case": {"name": name}, "enabled": enabled, "reduce": True, "error": repr(error)}
                    if name == "support":
                        record["support_tail_diagnostic"] = {"samples": tail, "complete_80_steps": False}
                    q.RECORDS.append(record)
                    report["failures"].append([name, enabled, repr(error)])
                report["records"].append(record)
            if len(pair) == 2 and (
                pair[0]["initial_state"] != pair[1]["initial_state"]
                or pair[0]["initial_velocity"] != pair[1]["initial_velocity"]
            ):
                report["failures"].append([name, "mismatched authored initial state"])
        print("QUALIFICATION_REPORT " + json.dumps(report, allow_nan=False), flush=True)
        self.assertEqual(len(report["records"]), 8)
        self.assertEqual(report["failures"], [], msg="All reduced cases collected; see QUALIFICATION_REPORT")


if __name__ == "__main__":
    unittest.main()
