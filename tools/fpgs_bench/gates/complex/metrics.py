# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physics and timing metrics for the patch-vs-point friction study (pure NumPy, no Warp).

Conventions (Newton): ``body_q`` rows are ``(px, py, pz, qx, qy, qz, qw)``; ``body_qd`` rows are
``(vx, vy, vz, wx, wy, wz)`` with the COM linear velocity and the angular velocity in world
coordinates. Sample arrays are shaped ``(samples, worlds, bodies, ...)``. Lengths are reported in
millimetres, angles in degrees, energies in joules.
"""

import warnings

import numpy as np


def quat_conj(q):
    return np.concatenate((-q[..., :3], q[..., 3:]), axis=-1)


def quat_mul(a, b):
    av, aw = a[..., :3], a[..., 3:]
    bv, bw = b[..., :3], b[..., 3:]
    v = aw * bv + bw * av + np.cross(av, bv)
    w = aw * bw - np.sum(av * bv, axis=-1, keepdims=True)
    return np.concatenate((v, w), axis=-1)


def quat_rotate(q, v):
    u, w = q[..., :3], q[..., 3:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_angle_deg(q):
    """Rotation angle of unit quaternions, in [0, 180] degrees (atan2 form: accurate near zero)."""
    return np.degrees(2.0 * np.arctan2(np.linalg.norm(q[..., :3], axis=-1), np.abs(q[..., 3])))


def yaw_deg(q):
    """Angle about world z of a rotation that is (close to) a pure z rotation."""
    return np.degrees(2.0 * np.arctan2(q[..., 2], q[..., 3]))


def stats(values, mask=None) -> dict:
    """Pooled distribution summary of a metric.

    ``mask`` (broadcast to ``values``) selects the entries that belong to the metric, e.g. the objects
    still inside a container. Selected entries that are not finite are counted in ``nonfinite`` and
    left out of the statistics, so a partly NaN arm is visible instead of looking normal.
    """
    v = np.asarray(values, dtype=np.float64)
    if mask is not None:
        v = v[np.broadcast_to(np.asarray(mask, dtype=bool), v.shape)]
    v = v.ravel()
    finite = np.isfinite(v)
    out = {"n": int(finite.sum())}
    if not finite.all():
        out["nonfinite"] = int((~finite).sum())
    v = v[finite]
    if v.size == 0:
        return out
    out.update(
        {
            "median": float(np.median(v)),
            "mean": float(v.mean()),
            "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)),
            "p95": float(np.percentile(v, 95)),
            "max": float(v.max()),
            "min": float(v.min()),
        }
    )
    return out


def per_world_stats(per_world_object, mask=None) -> dict:
    """Pooled summary plus the spread (IQR) of per-world medians.

    The spread across worlds means something only when the worlds start from different states
    (the runner's seeded per-world perturbation); for exact replicas it is atomics noise.
    """
    a = np.asarray(per_world_object, dtype=np.float64)
    out = stats(a, mask)
    if a.ndim == 2 and a.shape[0] > 1 and a.size:
        selected = np.isfinite(a) if mask is None else np.isfinite(a) & np.broadcast_to(np.asarray(mask, bool), a.shape)
        masked = np.where(selected, a, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # worlds with no selected entry
            medians = np.nanmedian(masked, axis=1)
        medians = medians[np.isfinite(medians)]
        if medians.size > 1:
            out["world_median_q25"] = float(np.percentile(medians, 25))
            out["world_median_q75"] = float(np.percentile(medians, 75))
            out["worlds"] = int(medians.size)
    return out


def window_indices(times, t0, t1, tolerance):
    """Sample indices inside [t0, t1] and whether the samples cover the window from start to end."""
    times = np.asarray(times)
    idx = np.flatnonzero((times >= t0 - tolerance) & (times <= t1 + tolerance))
    complete = bool(times.size) and times[-1] >= t1 - tolerance and times[0] <= t0 + tolerance
    return idx, complete


def com_positions(q, com_local):
    """World COM positions: q (S, W, B, 7), com_local (B, 3) -> (S, W, B, 3)."""
    return q[..., :3] + quat_rotate(q[..., 3:], np.broadcast_to(com_local, q[..., :3].shape))


def in_support_frame(points, rotations, support):
    """Express world COM points (S, W, B, 3) and body rotations (S, W, B, 4) in a support body's
    frame; ``support`` is that body's transform (S, W, 7), or None for the world frame."""
    if support is None:
        return points, rotations
    qc_inv = np.broadcast_to(quat_conj(support[..., 3:])[..., None, :], rotations.shape)
    return quat_rotate(qc_inv, points - support[..., None, :3]), quat_mul(qc_inv, rotations)


def lsq_slope(t, p):
    """Least-squares velocity of each point: t (S,), p (S, ...) -> (...) [units of p per s]."""
    tc = t - t.mean()
    return np.tensordot(tc, p - p.mean(axis=0), axes=(0, 0)) / np.sum(tc * tc)


def rest_metrics(times, points, rotations, t0, t1, tolerance):
    """Drift, LSQ creep speed and rotation drift of each object over a quiet window.

    points: (S, W, B, 3) support-frame COM positions [m]; rotations: (S, W, B, 4).
    Returns per-world, per-object arrays: drift [mm], creep [mm/s] over the whole window and over
    its first and second halves (a still-decaying settle shows as first >> second), jitter [mm]
    and rotation drift [deg]. None when fewer than three samples fall in the window.
    """
    idx, complete = window_indices(times, t0, t1, tolerance)
    if idx.size < 3:
        return None
    t = np.asarray(times, dtype=np.float64)[idx]
    p = points[idx]
    drift = np.linalg.norm(p[-1] - p[0], axis=-1) * 1e3
    slope = lsq_slope(t, p)
    tc = t - t.mean()
    residual = p - (p.mean(axis=0) + tc[:, None, None, None] * slope[None])
    jitter = np.sqrt(np.mean(np.sum(residual * residual, axis=-1), axis=0)) * 1e3
    rotation = quat_angle_deg(quat_mul(quat_conj(rotations[idx[0]]), rotations[idx[-1]]))
    out = {
        "t0": float(t[0]),
        "t1": float(t[-1]),
        "samples": int(idx.size),
        "complete": complete,
        "drift_mm": drift,
        "creep_mm_s": np.linalg.norm(slope, axis=-1) * 1e3,
        "jitter_mm": jitter,
        "rotation_deg": rotation,
    }
    half = idx.size // 2
    if idx.size >= 5:  # each half keeps >= 3 samples (they share the middle one)
        out["creep_first_half_mm_s"] = np.linalg.norm(lsq_slope(t[: half + 1], p[: half + 1]), axis=-1) * 1e3
        out["creep_second_half_mm_s"] = np.linalg.norm(lsq_slope(t[half:], p[half:]), axis=-1) * 1e3
    return out


def float32_ulp_mm(points):
    """Coarsest float32 position step [mm] over the axes of world points (..., 3) -> (...).

    Body poses are float32 on the device: a drift no larger than this step is below resolution.
    """
    ulp = np.spacing(np.abs(np.asarray(points, dtype=np.float64)).astype(np.float32)).astype(np.float64)
    return ulp.max(axis=-1) * 1e3


def deadband_mm_s(ulp_mm, substep_dt):
    """Speed below which a float32 position update rounds away: half a step per substep [mm/s]."""
    return 0.5 * np.asarray(ulp_mm, dtype=np.float64) / substep_dt


def velocity_split(v, normal):
    """|normal| and |tangential| parts of velocities v (..., 3) against unit normals (..., 3)."""
    vn = np.sum(v * normal, axis=-1)
    vt = np.linalg.norm(v - vn[..., None] * normal, axis=-1)
    return np.abs(vn), vt


def container_local(points, support_q):
    """World points (W, B, 3) in the container frame given its transforms (W, 7)."""
    conj = np.broadcast_to(quat_conj(support_q[:, 3:])[:, None, :], (*points.shape[:-1], 4))
    return quat_rotate(conj, points - support_q[:, None, :3])


def containment(local, half_extents, floor_top, *, margin=0.005, below_floor=0.0005, off_table=0.02):
    """Classify container-frame COM points (W, B, 3).

    outside: beyond the interior footprint (half extents + ``margin``) or more than ``off_table``
    below the container's bottom plane (fell off the bench). below_floor: inside the footprint but
    with the COM more than ``below_floor`` under the floor's top surface, i.e. tunnelled into or
    through the floor (an object resting on the floor keeps its COM above it).
    """
    hx, hy = half_extents
    in_footprint = (np.abs(local[..., 0]) <= hx + margin) & (np.abs(local[..., 1]) <= hy + margin)
    fell = local[..., 2] < -off_table
    outside = ~in_footprint | fell
    under = in_footprint & ~fell & (local[..., 2] < floor_top - below_floor)
    return outside, under


def kinetic_energy(q, qd, mass, inertia):
    """Per-body kinetic energy [J]: q (..., B, 7), qd (..., B, 6), mass (B,), inertia (B, 3, 3) body frame."""
    v, w = qd[..., :3], qd[..., 3:]
    linear = 0.5 * mass * np.sum(v * v, axis=-1)
    w_body = quat_rotate(quat_conj(q[..., 3:]), w)
    angular = 0.5 * np.einsum("...i,...ij,...j->...", w_body, inertia, w_body)
    return linear + angular


def transform_points(body_q, bodies, points):
    """World positions of contact points given in body frames (static shapes: bodies < 0, world frame)."""
    out = np.array(points, dtype=np.float64, copy=True)
    mask = bodies >= 0
    if np.any(mask):
        q = body_q[bodies[mask]]
        out[mask] = quat_rotate(q[:, 3:], points[mask]) + q[:, :3]
    return out


def contact_gap(body_q, shape_body, s0, s1, p0, p1, normal, margin0, margin1):
    """The solver's contact gap phi = n.(X1 p1 - X0 p0) - (margin0 + margin1) [m]; negative = penetration."""
    w0 = transform_points(body_q, shape_body[s0], p0)
    w1 = transform_points(body_q, shape_body[s1], p1)
    return np.einsum("ij,ij->i", normal, w1 - w0) - (margin0 + margin1)


def timing_stats(ms) -> dict:
    v = np.asarray(ms, dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "median_ms": float(np.median(v)),
        "mean_ms": float(v.mean()),
        "p10_ms": float(np.percentile(v, 10)),
        "p90_ms": float(np.percentile(v, 90)),
        "min_ms": float(v.min()),
        "max_ms": float(v.max()),
        "std_ms": float(v.std()),
    }
