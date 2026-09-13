# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Same-input G1 convex fixtures for the complete direct terrain query owner.

Large geometry captures remain local. This is a physical query/replay control,
not dynamics/convergence or throughput evidence.
"""

import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton

CAPTURES = Path("/tmp/fpgs-g1-geometry-paired16k-20260913-01")
PINS = (
    "09a540b0e88f3411cb26e8cbf6288ea89f521fa552ec939870795b4debd0952d",
    "4751dd0fc1d1dddaccc66087df73c36b8f356173be1a79516ad17395fba60eb0",
)


def make_fixture(gpu, device):
    """Use the same 96 actual pairs as the accepted cell-rejection control."""
    path = CAPTURES / f"gpu{gpu}/audit.geometry.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINS[gpu]:
        raise RuntimeError("Wrong current geometry capture")
    z = np.load(path)
    original_pairs = z["mesh_pairs"][np.argsort(z["mesh_pairs"][:, 1])]
    selected = original_pairs[np.linspace(0, len(original_pairs) - 1, 96, dtype=np.int64)]
    terrain = int(selected[0, 0])
    hfd = z["heightfield_data"][z["shape_heightfield_index"][terrain]]
    values = z["heightfield_elevations"].reshape(int(hfd["nrow"]), int(hfd["ncol"]))
    heightfield = newton.Heightfield(
        values,
        nrow=int(hfd["nrow"]),
        ncol=int(hfd["ncol"]),
        hx=float(hfd["hx"]),
        hy=float(hfd["hy"]),
        min_z=float(hfd["min_z"]),
        max_z=float(hfd["max_z"]),
    )
    mesh = newton.Mesh(z["hull_0_vertices"], z["hull_0_indices"])
    builder = newton.ModelBuilder()

    def config(index):
        return builder.ShapeConfig(
            margin=float(z["shape_margin"][index]),
            gap=float(z["shape_gap"][index]),
            mu=float(z["shape_material_mu"][index]),
            restitution=float(z["shape_material_restitution"][index]),
        )

    pose = z["geom_transform"][terrain]
    builder.add_shape_heightfield(heightfield=heightfield, xform=wp.transform(pose[:3], pose[3:]), cfg=config(terrain))
    for _, index in selected:
        pose = z["geom_transform"][index]
        body = builder.add_body(xform=wp.transform(pose[:3], pose[3:]))
        builder.add_shape_convex_hull(body=body, mesh=mesh, scale=z["shape_scale"][index], cfg=config(index))
    model = builder.finalize(device=device)
    np.testing.assert_array_equal(model.heightfield_elevations.numpy(), z["heightfield_elevations"])
    np.testing.assert_array_equal(model.heightfield_data.numpy(), z["heightfield_data"])
    pairs = wp.array([[0, i + 1] for i in range(len(selected))], dtype=wp.vec2i, device=device)
    z.close()
    return model, pairs


def coverage(a, b):
    """Require bidirectional full tuples within the original physical tolerances."""
    maxima = np.zeros(3)
    if set(a["shape"]) != set(b["shape"]):
        raise AssertionError("A physical shape-pair contact set disappeared")
    for source, target in ((a, b), (b, a)):
        for shape in np.unique(source["shape"]):
            si, ti = source["shape"] == shape, target["shape"] == shape
            position = np.linalg.norm(source["point"][si, None] - target["point"][ti][None, :], axis=2)
            distance = np.abs(source["distance"][si, None] - target["distance"][ti][None, :])
            normal = np.linalg.norm(source["normal"][si, None] - target["normal"][ti][None, :], axis=2)
            score = np.maximum.reduce((position / 2e-4, distance / 2e-4, normal / 2e-3))
            match = np.argmin(score, axis=1)
            row = np.arange(len(match))
            maxima = np.maximum(maxima, [value[row, match].max() for value in (position, distance, normal)])
    if not (maxima < (2e-4, 2e-4, 2e-3)).all():
        raise AssertionError(("Changed physical contacts", maxima.tolist()))
    return maxima.tolist()


def check(gpu, device):
    """Compare both complete collision owners and three same-input graph replays."""
    model, pairs = make_fixture(gpu, device)
    state = model.state()
    records = []
    for direct in (False, True):
        with patch.dict(os.environ, NEWTON_HEIGHTFIELD_CELL_REJECT="1", NEWTON_HEIGHTFIELD_DIRECT=str(int(direct))):
            pipeline = newton.CollisionPipeline(
                model,
                shape_pairs_filtered=pairs,
                reduce_contacts=False,
                rigid_contact_max=4096,
                max_triangle_pairs=4096,
            )
        if pipeline.narrow_phase._heightfield_direct != direct:
            raise AssertionError("Wrong current collision owner")
        contacts = pipeline.contacts()
        distance = wp.empty(4096, dtype=float, device=device)
        point = wp.empty(4096, dtype=wp.vec3, device=device)

        def read(pipeline=pipeline, contacts=contacts, distance=distance, point=point):
            pipeline.narrow_phase.check_buffer_capacity()
            count = int(contacts.rigid_contact_count.numpy()[0])
            newton.eval_rigid_contact_kinematics(model, state, contacts, out_distance=distance, out_point0_world=point)
            result = {
                "shape": contacts.rigid_contact_shape1.numpy()[:count],
                "distance": distance.numpy()[:count],
                "normal": contacts.rigid_contact_normal.numpy()[:count],
                "point": point.numpy()[:count],
                "triangles": int(pipeline.narrow_phase.triangle_pairs_count.numpy()[0]),
            }
            if not all(np.isfinite(result[name]).all() for name in ("distance", "normal", "point")):
                raise AssertionError("Nonfinite contact geometry")
            return result

        pipeline.collide(state, contacts)
        initial = read()
        if device != "cpu":
            with wp.ScopedCapture(device=device) as captured:
                pipeline.collide(state, contacts)
            for _ in range(3):
                wp.capture_launch(captured.graph)
            replay = read()
            if replay["triangles"] != initial["triangles"]:
                raise AssertionError("Graph changed logical triangle demand")
            coverage(initial, replay)
        records.append(initial)
    before, after = records
    if before["triangles"] != after["triangles"]:
        raise AssertionError("Direct owner changed logical triangle demand")
    return {
        "capture_gpu": gpu,
        "capture_sha256": PINS[gpu],
        "pairs": 96,
        "triangles": before["triangles"],
        "contacts_before": len(before["shape"]),
        "contacts_after": len(after["shape"]),
        "max_position_distance_normal_error": coverage(before, after),
    }
