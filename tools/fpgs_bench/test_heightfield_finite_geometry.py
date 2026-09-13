# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent finite witnesses and complete saved-current G1 pipeline checks.

The 96-pair fixtures are local, pinned diagnostic inputs, not task throughput or
trajectory acceptance. Contact manifolds may change; finite geometry may not.
"""

import hashlib
import os
import unittest
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
def stream_query_kernel():
    """Use the production ABI on the exact decoded current triangle stream."""
    from newton._src.geometry.heightfield_finite import QueryResult, query  # noqa: PLC0415
    from newton._src.utils.heightfield import HeightfieldData, get_triangle_shape_from_heightfield  # noqa: PLC0415

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
        output[i] = query(tri.scale, tri.auxiliary, center, quat, scales[triple[1]] * 0.5, threshold)
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
    """Read complete current output only at an untimed test boundary."""
    pipeline.narrow_phase.check_buffer_capacity()
    n = int(contacts.rigid_contact_count.numpy()[0])
    nt = int(pipeline.narrow_phase.triangle_pairs_count.numpy()[0])
    if not 0 <= n <= contacts.rigid_contact_max or not 0 <= nt <= pipeline.narrow_phase.triangle_pairs.shape[0]:
        raise AssertionError("Truncated diagnostic buffer")
    triples = pipeline.narrow_phase.triangle_pairs.numpy()[:nt].copy()
    marked = triples[:, 2] < 0
    triples[marked, 2] = ~triples[marked, 2]
    p0, p1 = (getattr(contacts, "rigid_contact_point" + side).numpy()[:n] for side in ("0", "1"))
    shapes = contacts.rigid_contact_shape1.numpy()[:n]
    poses = state.body_q.numpy()
    world1 = np.array([poses[s - 1, :3] + rotation(poses[s - 1, 3:]) @ v for s, v in zip(shapes, p1, strict=True)])
    normal = contacts.rigid_contact_normal.numpy()[:n]
    distance = np.einsum("ij,ij->i", world1 - p0, normal)
    if not all(np.isfinite(v).all() for v in (p0, world1, normal, distance)):
        raise AssertionError("Nonfinite public contacts")
    return {
        "triples": triples,
        "marked": int(marked.sum()),
        "shape": shapes,
        "point": p0,
        "normal": normal,
        "distance": distance,
        "count": n,
    }


def check_current_pipeline(gpu, device):
    """Complete direct/reducer controls, exact stream ownership, and graph replays."""
    model, state, pairs = current_fixture(gpu, device)
    records = []
    for reduce in (False, True):
        variants = []
        for enabled in (False, True):
            with patch.dict(
                os.environ, NEWTON_HEIGHTFIELD_CELL_REJECT="1", NEWTON_HEIGHTFIELD_FINITE_QUERY=str(int(enabled))
            ):
                # Preserve the inherited96 fixture bounds; these are NOT16K
                # benchmark allocations or extra candidate-only buffers.
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
            if enabled and initial["marked"] == 0:
                raise AssertionError("Requested finite query silently fell back")
            if not enabled and initial["marked"] != 0:
                raise AssertionError("Baseline unexpectedly marked its stream")
            if enabled:
                poses = state.body_q.numpy().copy()
                away = poses.copy()
                away[:, 2] += 1000.0
                state.body_q.assign(away)
                pipeline.collide(state, contacts)
                pipeline.narrow_phase.check_buffer_capacity()
                if int(contacts.rigid_contact_count.numpy()[0]) != 0:
                    raise AssertionError("Raised boxes retained stale contacts")
                if int(pipeline.narrow_phase.triangle_pairs_count.numpy()[0]) != 0:
                    raise AssertionError("Raised boxes retained a stale triangle prefix")
                state.body_q.assign(poses)
                pipeline.collide(state, contacts)
                restored = snapshot(pipeline, state, contacts)
                if restored["count"] != initial["count"] or restored["marked"] != initial["marked"]:
                    raise AssertionError("Empty-to-regrown prefix retained stale markers")
            if device != "cpu":
                with wp.ScopedCapture(device=device) as captured:
                    pipeline.collide(state, contacts)
                for _ in range(3):
                    wp.capture_launch(captured.graph)
                    replay = snapshot(pipeline, state, contacts)
                    if replay["count"] != initial["count"] or replay["marked"] != initial["marked"]:
                        raise AssertionError("Query marker/manifold lifecycle changed on replay")
            variants.append(initial)
            if enabled and not reduce:
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
                audits = [
                    audit_query(result, tri, center, rotation(quat), h)
                    for result, tri, center, quat, h in zip(
                        output.numpy(), vertices.numpy(), centers.numpy(), quaternions.numpy(), half, strict=True
                    )
                ]
                admitted = [item for item in audits if not item["fallback"]]
                if len(admitted) != initial["marked"]:
                    raise AssertionError("Pipeline and independent query disagree about handled stream entries")
                records.append(
                    {
                        "query_cases": count,
                        "admitted": len(admitted),
                        "maximum_surface_error": max(item["surface_error"] for item in admitted),
                    }
                )
        before, after = variants

        def key(row):
            return tuple(int(v) for v in row)

        if sorted(map(key, before["triples"])) != sorted(map(key, after["triples"])):
            raise AssertionError("Analytical query changed the logical midphase stream")
        missing = sorted(set(before["shape"]) - set(after["shape"]))
        if missing:
            raise AssertionError(("Previously contacting shapes lost all contacts", missing))
        minimum_errors = [
            abs(
                float(before["distance"][before["shape"] == shape].min())
                - float(after["distance"][after["shape"] == shape].min())
            )
            for shape in np.unique(before["shape"])
        ]
        if max(minimum_errors, default=0.0) > 2e-4:
            raise AssertionError(("Changed actual minimum separation", max(minimum_errors)))
        records.append(
            {
                "reduce": reduce,
                "original_contacts": before["count"],
                "candidate_contacts": after["count"],
                "marked": after["marked"],
                "maximum_minimum_separation_change": max(minimum_errors, default=0.0),
            }
        )
    return {"gpu_fixture": gpu, "device": str(device), "input_sha256": PINS[gpu], "records": records}


class TestIndependentGeometry(unittest.TestCase):
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
        for gpu in (0, 1):
            RECORDS.append(check_current_pipeline(gpu, "cuda:0"))

    def test_synthetic_native_cpu(self):
        """Check actual native CPU geometry against independent physical witnesses."""
        check_synthetic_native("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires the root-owned paired GPU lease")
    def test_synthetic_native_cuda(self):
        """Exercise finite edge/penetration/fallback arithmetic on the owned GPU."""
        RECORDS.append(check_synthetic_native("cuda:0"))


if __name__ == "__main__":
    unittest.main()
