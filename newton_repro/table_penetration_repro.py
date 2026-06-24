# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Standalone Newton repro for Kuka-Allegro fingertip/table penetration.

This captures env 25 from the IsaacLab Kuka-Allegro play run that hit the
5x velocity trigger at episode length 2.  The script embeds the pre-physics
body poses and shape-local collision transforms from the debug NPZ, then runs
Newton's collision pipeline without IsaacLab.

Expected fixed/analytical behavior reports vertical table contacts with about
3.3 cm maximum penetration.  On unfixed generic MPR paths this same setup may
instead report a much deeper lateral contact for thin table box pairs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton


@dataclass(frozen=True)
class ShapeSpec:
    name: str
    kind: str
    scale: tuple[float, float, float]
    body_pose: tuple[float, float, float, float, float, float, float]
    shape_pose: tuple[float, float, float, float, float, float, float]


# Pose tuple convention is (x, y, z, qx, qy, qz, qw), matching wp.transform.
# These are the pre-physics poses from:
# /tmp/isaaclab_newton_thin_debug_5x/debug_replay_velocity_spike_20260623_051539_098789.npz
_CAPTURED_SHAPES: tuple[ShapeSpec, ...] = (
    ShapeSpec(
        name="ring_link_2_box",
        kind="box",
        scale=(0.01587500050663948, 0.012249999679625034, 0.013000000268220901),
        body_pose=(
            -4.826506614685059,
            -4.887167453765869,
            0.29723361134529114,
            0.2506120502948761,
            -0.03361557424068451,
            -0.9588756561279297,
            0.12892676889896393,
        ),
        shape_pose=(
            0.029046397656202316,
            -0.002373478841036558,
            -3.552713678800501e-15,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    ),
    ShapeSpec(
        name="ring_link_3_box",
        kind="box",
        scale=(0.00912499986588955, 0.008750000037252903, 0.008663489483296871),
        body_pose=(
            -4.858806610107422,
            -4.897308826446533,
            0.27911096811294556,
            0.252519428730011,
            -0.013051452115178108,
            -0.966198742389679,
            0.050244636833667755,
        ),
        shape_pose=(
            0.019999999552965164,
            -2.0887805374236734e-14,
            -1.7763568394002505e-15,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    ),
    ShapeSpec(
        name="middle_link_3_capsule",
        kind="capsule",
        scale=(0.007499999832361937, 0.009999999776482582, 0.0),
        body_pose=(
            -4.881906986236572,
            -4.889946460723877,
            0.3132691979408264,
            0.3113357722759247,
            0.18293151259422302,
            -0.8513144254684448,
            0.3806189000606537,
        ),
        shape_pose=(
            0.04500000178813934,
            0.004003198351711035,
            5.329070518200751e-15,
            -0.11670627444982529,
            0.6974092721939087,
            0.11670627444982529,
            0.6974092721939087,
        ),
    ),
    ShapeSpec(
        name="ring_link_3_capsule",
        kind="capsule",
        scale=(0.007499999832361937, 0.009999999776482582, 0.0),
        body_pose=(
            -4.858806610107422,
            -4.897308826446533,
            0.27911096811294556,
            0.252519428730011,
            -0.013051452115178108,
            -0.966198742389679,
            0.050244636833667755,
        ),
        shape_pose=(
            0.04500000178813934,
            0.004003198351711035,
            1.7763568394002505e-15,
            -0.11670627444982529,
            0.6974092721939087,
            0.11670627444982529,
            0.6974092721939087,
        ),
    ),
    ShapeSpec(
        name="table_box",
        kind="box",
        scale=(0.3999999761581421, 0.75, 0.019999999552965164),
        body_pose=(-5.050000190734863, -4.5, 0.23499999940395355, 0.0, 0.0, 0.0, 1.0),
        shape_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    ),
)


def _wp_transform(pose: tuple[float, float, float, float, float, float, float]) -> wp.transform:
    return wp.transform(
        wp.vec3(float(pose[0]), float(pose[1]), float(pose[2])),
        wp.quat(float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])),
    )


def _add_shape(builder: newton.ModelBuilder, spec: ShapeSpec) -> None:
    body = builder.add_body(xform=_wp_transform(spec.body_pose))
    shape_xform = _wp_transform(spec.shape_pose)
    if spec.kind == "box":
        builder.add_shape_box(body, xform=shape_xform, hx=spec.scale[0], hy=spec.scale[1], hz=spec.scale[2])
    elif spec.kind == "capsule":
        builder.add_shape_capsule(body, xform=shape_xform, radius=spec.scale[0], half_height=spec.scale[1])
    else:
        raise ValueError(f"Unsupported shape kind: {spec.kind}")


def _build_model(shapes: tuple[ShapeSpec, ...], device: str) -> newton.Model:
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.01
    for spec in shapes:
        _add_shape(builder, spec)
    return builder.finalize(device=device)


def _contact_distance(contacts: newton.Contacts, index: int) -> float:
    normal = contacts.rigid_contact_normal.numpy()[index]
    point0 = contacts.rigid_contact_point0.numpy()[index]
    point1 = contacts.rigid_contact_point1.numpy()[index]
    margin0 = float(contacts.rigid_contact_margin0.numpy()[index])
    margin1 = float(contacts.rigid_contact_margin1.numpy()[index])
    return float(np.dot(normal, point1 - point0) - (margin0 + margin1))


def _run_case(name: str, shapes: tuple[ShapeSpec, ...], device: str, broad_phase: str) -> None:
    model = _build_model(shapes, device)
    state = model.state()
    pipeline = newton.CollisionPipeline(model, broad_phase=broad_phase)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)

    count = int(contacts.rigid_contact_count.numpy()[0])
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    normals = contacts.rigid_contact_normal.numpy()[:count]

    rows: list[tuple[float, int, int, int]] = []
    for i in range(count):
        rows.append((_contact_distance(contacts, i), i, int(shape0[i]), int(shape1[i])))

    print(f"\n{name}: {count} contacts, broad_phase={broad_phase}")
    table_contacts: list[float] = []
    for distance, i, a, b in sorted(rows, key=lambda row: row[0]):
        is_table = shapes[a].name == "table_box" or shapes[b].name == "table_box"
        if is_table:
            table_contacts.append(distance)
        tag = "TABLE" if is_table else "     "
        print(
            f"  {i:2d} {tag} dist={distance: .9f} normal={normals[i]} "
            f"{shapes[a].name} | {shapes[b].name}"
        )

    if table_contacts:
        deepest = min(table_contacts)
        print(f"  deepest table penetration: {-deepest:.6f} m ({-100.0 * deepest:.2f} cm)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--broad-phase", default="explicit", choices=("explicit", "nxn", "sap"))
    parser.add_argument("--pairs", action="store_true", help="Also run each fingertip/table pair in isolation.")
    parser.add_argument("--keep-cache", action="store_true", help="Do not clear the Warp kernel cache before running.")
    args = parser.parse_args()

    if not args.keep_cache:
        wp.clear_kernel_cache()

    with wp.ScopedDevice(args.device):
        _run_case("captured_pre_step_cluster", _CAPTURED_SHAPES, args.device, args.broad_phase)
        if args.pairs:
            table = _CAPTURED_SHAPES[-1]
            for spec in _CAPTURED_SHAPES[:-1]:
                _run_case(f"pair_{spec.name}_vs_table", (spec, table), args.device, args.broad_phase)


if __name__ == "__main__":
    main()
