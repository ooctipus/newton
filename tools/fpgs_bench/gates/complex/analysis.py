# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Turn one run's raw data into the result sections and summary rows (pure NumPy; no Warp, no Newton).

``analyze(ctx, raw, run, horizon_frames=None)`` is called twice in the study's life:

* by ``run_complex.py`` at the end of a run, over every frame it ran, to fill the result JSON;
* by ``compare.py`` for both arms of a pair whose runs stopped at different frames (a budget
  truncation), over the frames *both* arms completed, so that windows, drift and creep are compared
  over the same time range. The raw arrays come from the ``.npz`` written next to each JSON.

``ctx`` is the JSON-safe ``analysis_context`` (scene layout, windows, metric bodies), ``raw`` holds
the per-frame series, the sampled states and the snapshots, and ``run`` the run-level records that
cannot be clipped to a horizon (construction, watermarks, capacity flags, the all-world end check).
"""

import metrics
import numpy as np

DEEP_MM = 0.5  # contact penetration counted as "deep"
MOVED_MM = 10.0  # pyramid box displacement within the rest window counted as "moved"
COLLAPSE_M = 0.5  # example_pyramid.test_final: a top box displaced >= 0.5 m from its build pose collapsed
POST_IMPACT_S = 0.5  # pyramid: the levels have dropped onto each other by then
LOST_GRASP_MM = 15.0  # TCP slip relative to the container counted as a lost grasp
TABLE_PLANE_M = -0.005  # below this a clutter body has left the table's top
ROLLED_OFF_SPEED = 0.3  # m/s: a body leaving the table slower than this rolled off; faster, it left fast
LAUNCH_RISE_M = 0.030  # a rise over this within LAUNCH_WINDOW_S after settle is a launch
LAUNCH_WINDOW_S = 0.25
GRAVITY = 9.81


def window_frames(frames, fps, t0, t1):
    end_times = (np.arange(frames) + 1) / fps
    return (end_times >= t0 - 1e-9) & (end_times <= t1 + 1e-9)


def coverage(frames, fps, t0, t1) -> dict:
    """Which part of [t0, t1] the frames reached; complete when the run got to the window's end."""
    ends = (np.arange(frames) + 1) / fps
    inside = ends[(ends >= t0 - 1e-9) & (ends <= t1 + 1e-9)]
    return {
        "window_s": [float(t0), float(t1)],
        "covered_s": [float(inside[0]), float(inside[-1])] if inside.size else None,
        "complete": bool(frames / fps >= t1 - 0.5 / fps),
    }


def timing_section(ctx, raw, frames):
    ms = np.asarray(raw["frame_ms"][:frames], dtype=np.float64)
    after_sync = np.asarray(raw["after_sync"][:frames], dtype=bool)
    fps, substeps = ctx["fps"], ctx["substeps"]
    timed = ~after_sync & (np.arange(frames) >= ctx["warmup"]) & np.isfinite(ms)
    windows = {"all": {**metrics.timing_stats(ms[timed]), **coverage(frames, fps, 0.0, ctx["frames_planned"] / fps)}}
    for name, (t0, t1) in ctx["windows"].items():
        mask = timed & window_frames(frames, fps, t0, t1)
        windows[name] = {**metrics.timing_stats(ms[mask]), **coverage(frames, fps, t0, t1)}
    for entry in windows.values():
        if entry.get("n"):
            entry["median_ms_per_substep"] = entry["median_ms"] / substeps
    return {"frames": frames, "frames_timed": int(timed.sum()), "windows": windows}


def counts_section(ctx, raw, frames):
    fps = ctx["fps"]
    per_world = np.asarray(raw["contacts_total"][:frames], dtype=np.float64) / ctx["world_count"]
    rows = {
        name: np.asarray(raw[f"rows_{name}"][:frames], dtype=np.float64)
        for name in ("mf", "dense")
        if raw.get(f"rows_{name}") is not None and len(raw[f"rows_{name}"])
    }
    out = {}
    for name, (t0, t1) in {"all": (0.0, np.inf), **ctx["windows"]}.items():
        mask = window_frames(frames, fps, t0, t1)
        if not mask.any():
            continue
        entry = {
            "contacts_per_world_median": float(np.median(per_world[mask])),
            "contacts_per_world_max": float(per_world[mask].max()),
        }
        for row_name, values in rows.items():  # (frames, metric worlds): per-world row counts
            entry[f"{row_name}_rows_per_world_median"] = float(np.median(values[mask]))
            entry[f"{row_name}_rows_per_world_max"] = float(values[mask].max())
        out[name] = entry
    return out


def _picked(snapshots, t0, t1, t_end):
    hi = min(t1, t_end) + 1e-9
    return [s for s in snapshots if t0 - 1e-9 <= s["t"] <= hi]


def penetration_section(snapshots, windows, t_end):
    """Penetration over contacts touching a metric object or the container (``objects``); robot and
    fixture contacts (e.g. the fixed Franka base on the static bench) are kept apart in ``other``."""
    out = {}
    for name, (t0, t1) in {"all": (0.0, np.inf), **windows}.items():
        picked = [s for s in _picked(snapshots, t0, t1, t_end) if s.get("penetration", {}).get("objects", {}).get("n")]
        if not picked:
            continue
        objects = [s["penetration"]["objects"] for s in picked]
        other = [s["penetration"]["other"]["max_mm"] for s in picked if s["penetration"].get("other", {}).get("n")]
        out[name] = {
            "snapshots": len(picked),
            "max_mm": max(o["max_mm"] for o in objects),
            "p99_mm_median": float(np.median([o["p99_mm"] for o in objects])),
            "deep_count_max": max(o["deep_count"] for o in objects),
            "other_max_mm": max(other) if other else None,
        }
    return out


def census_section(snapshots, law, t_end):
    snaps = _picked(snapshots, 0.0, np.inf, t_end)
    out = {}
    for name in ("mf", "dense"):
        ratios = [
            s["census"][name]["friction"] / s["census"][name]["normal"]
            for s in snaps
            if name in s.get("census", {}) and s["census"][name]["normal"]
        ]
        if ratios:
            out[f"{name}_friction_per_normal_median"] = float(np.median(ratios))
    if law == "patch":
        carried = [
            s["anchors"]["carried"] / s["anchors"]["with_rows"] for s in snaps if s.get("anchors", {}).get("with_rows")
        ]
        if carried:
            out["anchors_carried_fraction_median"] = float(np.median(carried))
    return out


def physics_section(ctx, raw, t_end):
    """Quality metrics from the sampled states of the metric worlds; see README for definitions."""
    fps = ctx["fps"]
    tol = 0.5 / fps
    times_all = np.asarray(raw["times"], dtype=np.float64)
    keep = times_all <= t_end + tol
    times = times_all[keep]
    Q = np.asarray(raw["body_q"], dtype=np.float64)[keep]
    QD = np.asarray(raw["body_qd"], dtype=np.float64)[keep]
    com, mass, inertia = (np.asarray(raw[key], dtype=np.float64) for key in ("com", "mass", "inertia"))
    n_samples, n_worlds = Q.shape[:2]
    substep_dt = 1.0 / (fps * ctx["substeps"])
    out = {
        "samples": int(n_samples),
        "sample_times": [float(times[0]), float(times[-1])],
        "worlds": int(n_worlds),
        "worlds_independent": bool(ctx.get("perturbed")),
        "finite_samples": bool(np.isfinite(Q).all() and np.isfinite(QD).all()),
    }
    support = ctx.get("support_body")
    support_q = Q[:, :, support] if support is not None else None
    objects = sorted({b for members in ctx["groups"].values() for b in members})
    column = {b: i for i, b in enumerate(objects)}
    points = metrics.com_positions(Q[:, :, objects], com[objects])  # (S, W, O, 3) world COMs

    def closest(t):
        return int(np.argmin(np.abs(times - t)))

    rest = ctx["windows"].get("rest")
    rest_idx = metrics.window_indices(times, *rest, tol)[0] if rest else np.zeros(0, dtype=int)
    have_rest = rest_idx.size >= 3

    # Containment (container scenes): objects outside the interior footprint, and objects whose COM sits
    # under the floor's top surface inside the footprint (tunnelled into or through the floor).
    container = support is not None and ctx.get("container_half_extents") is not None
    in_container = np.ones((n_worlds, len(objects)), dtype=bool)  # objects the rest metrics cover

    def classify(i):
        local = metrics.container_local(points[i], support_q[i])
        return metrics.containment(local, ctx["container_half_extents"], ctx["floor_top"])

    if container:
        if have_rest:
            for label, i in (("rest_start", int(rest_idx[0])), ("rest_end", int(rest_idx[-1]))):
                outside, under = classify(i)
                out[f"containment.{label}"] = {
                    "t": float(times[i]),
                    "outside": metrics.stats(outside.sum(axis=1)),
                    "below_floor": metrics.stats(under.sum(axis=1)),
                }
                if label == "rest_start":
                    in_container = ~outside  # spilled before the quiet window: reported apart
                else:
                    outside_rest_end = outside.sum(axis=1)
        outside, under = classify(n_samples - 1)
        end = {
            "t": float(times[-1]),
            "outside": metrics.stats(outside.sum(axis=1)),
            "below_floor": metrics.stats(under.sum(axis=1)),
        }
        if have_rest:
            end["worlds_lost_objects_after_rest"] = int(np.count_nonzero(outside.sum(axis=1) > outside_rest_end))
        out["containment.end"] = end

    # Quiet-window drift and creep per group, in the support frame; float32 resolution floors beside them.
    groups = {"objects": objects, **{k: v for k, v in ctx["groups"].items() if sorted(v) != objects}}
    if have_rest:
        i0 = int(rest_idx[0])
        support_ulp = metrics.float32_ulp_mm(support_q[i0][:, None, :3]) if support is not None else 0.0
        for group, members in groups.items():
            cols = [column[b] for b in members]
            local_points, local_rot = metrics.in_support_frame(points[:, :, cols], Q[:, :, members, 3:], support_q)
            result = metrics.rest_metrics(times, local_points, local_rot, *rest, tol)
            mask = in_container[:, cols]
            ulp = metrics.float32_ulp_mm(points[i0][:, cols]) + support_ulp
            entry = {
                "window_s": [result["t0"], result["t1"]],
                "complete": result["complete"],
                "samples": result["samples"],
                "objects_per_world_median": float(np.median(mask.sum(axis=1))),
                "drift_floor_mm": metrics.stats(ulp, mask),
                "creep_deadband_mm_s": metrics.stats(metrics.deadband_mm_s(ulp, substep_dt), mask),
            }
            for key in (
                "drift_mm",
                "creep_mm_s",
                "creep_first_half_mm_s",
                "creep_second_half_mm_s",
                "jitter_mm",
                "rotation_deg",
            ):
                if key in result:
                    entry[key] = metrics.per_world_stats(result[key], mask)
            out[f"rest.{group}"] = entry
            if ctx["kind"] == "pyramid" and group == "objects":
                moved = np.count_nonzero(result["drift_mm"] > MOVED_MM, axis=1)
        if container and not in_container.all():  # spilled parts, on the bench: world frame
            result = metrics.rest_metrics(times, points, Q[:, :, objects, 3:], *rest, tol)
            spilled = ~in_container
            out["rest.spilled"] = {
                "window_s": [result["t0"], result["t1"]],
                "objects_per_world_median": float(np.median(spilled.sum(axis=1))),
                "drift_mm": metrics.per_world_stats(result["drift_mm"], spilled),
                "creep_mm_s": metrics.per_world_stats(result["creep_mm_s"], spilled),
            }

        # Window averages, not one frame: KE, and object speed split along/against the support normal.
        idx = rest_idx
        v = QD[idx][:, :, objects, :3]
        if support is not None:
            normal = metrics.quat_rotate(
                support_q[idx][..., 3:], np.broadcast_to([0.0, 0.0, 1.0], (*support_q[idx].shape[:-1], 3))
            )
        else:
            normal = np.broadcast_to([0.0, 0.0, 1.0], (*v.shape[:2], 3))
        vn, vt = metrics.velocity_split(v, np.broadcast_to(normal[:, :, None, :], v.shape))
        ke = metrics.kinetic_energy(Q[idx][:, :, objects], QD[idx][:, :, objects], mass[objects], inertia[objects])
        ke = np.where(in_container[None], ke, 0.0).sum(axis=-1)  # (s, W)
        out["rest.motion"] = {
            "window_s": [float(times[idx[0]]), float(times[idx[-1]])],
            "samples": int(idx.size),
            "ke_J_per_world_mean": metrics.stats(ke.mean(axis=0)),
            "speed_tangential_mm_s": metrics.per_world_stats(vt.mean(axis=0) * 1e3, in_container),
            "speed_normal_mm_s": metrics.per_world_stats(vn.mean(axis=0) * 1e3, in_container),
            "mean_velocity_mm_s": metrics.per_world_stats(np.linalg.norm(v.mean(axis=0), axis=-1) * 1e3, in_container),
        }
        if support is not None:
            i_end = int(rest_idx[-1])
            move = np.linalg.norm(support_q[i_end, :, :3] - support_q[i0, :, :3], axis=-1) * 1e3
            turn = metrics.quat_angle_deg(
                metrics.quat_mul(metrics.quat_conj(support_q[i0, :, 3:]), support_q[i_end, :, 3:])
            )
            out["rest.container"] = {"drift_mm": metrics.stats(move), "rotation_deg": metrics.stats(turn)}

    strokes = ctx["windows"].get("strokes")
    if strokes and ctx.get("tcp_body") is not None and support is not None:
        idx, complete = metrics.window_indices(times, *strokes, tol)
        if idx.size >= 3:
            c = support_q[idx]
            tcp = Q[idx][:, :, ctx["tcp_body"], :3]
            tcp_local = metrics.quat_rotate(metrics.quat_conj(c[..., 3:]), tcp - c[..., :3])
            slip = np.linalg.norm(tcp_local - tcp_local[0], axis=-1) * 1e3
            axis = ctx["shake_axis"]
            travel_c = np.ptp(c[..., axis], axis=0) * 1e3
            travel_t = np.ptp(tcp[..., axis], axis=0) * 1e3
            out["strokes"] = {
                "window_s": [float(times[idx[0]]), float(times[idx[-1]])],
                "complete": complete,
                "tcp_slip_in_container_max_mm": metrics.stats(slip.max(axis=0)),
                "tcp_slip_in_container_end_mm": metrics.stats(slip[-1]),
                "grasp_lost_worlds": int(np.count_nonzero(slip.max(axis=0) > LOST_GRASP_MM)),
                "container_travel_shake_axis_mm": metrics.stats(travel_c),
                "container_travel_x_mm": metrics.stats(np.ptp(c[..., 0], axis=0) * 1e3),
                "container_travel_y_mm": metrics.stats(np.ptp(c[..., 1], axis=0) * 1e3),
                "tracking_ratio": metrics.stats(travel_c / np.maximum(travel_t, 1e-9)),
            }

    special = ctx.get("special", {})
    if "top_center" in special:
        top = special["top_center"]
        yaw = (metrics.yaw_deg(Q[:, :, top, 3:]) - metrics.yaw_deg(Q[0, :, top, 3:]) + 180.0) % 360.0 - 180.0
        cube = {"top_yaw_end_deg": metrics.stats(yaw[-1])}
        turn = ctx["windows"].get("turn")
        if turn and times[-1] >= turn[1] - tol:
            cube["top_yaw_turn_end_deg"] = metrics.stats(yaw[closest(turn[1])])
        lower = special["lower_layer"]
        rotation = metrics.quat_angle_deg(
            metrics.quat_mul(metrics.quat_conj(Q[0][:, lower, 3:])[None], Q[:, :, lower, 3:])
        )
        cube["lower_layer_rotation_max_deg"] = metrics.stats(rotation.max(axis=(0, 2)))
        out["cube"] = cube

    if ctx["kind"] == "pyramid":
        # Displacement from the build pose and from the post-impact pose, kept apart from the in-window
        # "moved" count (creep, not collapse). Survival uses example_pyramid.test_final's 0.5 m top-box
        # criterion against the *settled* pose (build x/y, level gaps closed): the example tests 5 rows,
        # whose top box drops 0.16 m closing the 0.04 m build gaps; a 14-row top drops 0.52 m doing so.
        def displacement(i, ref):
            return np.linalg.norm(points[i] - points[ref], axis=-1) * 1e3

        top_cols = [column[b] for b in special.get("top", [])]
        last = n_samples - 1
        pyramid = {}
        if have_rest:
            impact = closest(POST_IMPACT_S)
            for label, i in (("rest_start", int(rest_idx[0])), ("rest_end", int(rest_idx[-1]))):
                pyramid[f"from_build_{label}_mm"] = metrics.stats(displacement(i, 0))
                pyramid[f"from_impact_{label}_mm"] = metrics.stats(displacement(i, impact))
            pyramid["impact_t"] = float(times[impact])
            pyramid["moved_gt_10mm_in_rest"] = metrics.stats(moved)
        if top_cols:
            settled = points[0].copy()
            settled[..., 2] = np.asarray(special["settled_z"], dtype=np.float64)[objects]
            from_settled = np.linalg.norm(points[last] - settled, axis=-1)[:, top_cols] * 1e3
            pyramid["top_from_build_end_mm"] = metrics.stats(displacement(last, 0)[:, top_cols])
            pyramid["top_from_settled_end_mm"] = metrics.stats(from_settled)
            pyramid["top_height_deficit_end_mm"] = metrics.stats(
                (settled[:, top_cols, 2] - points[last][:, top_cols, 2]) * 1e3
            )
            pyramid["collapsed_worlds"] = int(np.count_nonzero(from_settled.max(axis=1) >= COLLAPSE_M * 1e3))
        out["pyramid"] = pyramid

    q_end = Q[-1]
    out["horizon_end"] = {
        "t": float(times[-1]),
        "finite": bool(np.isfinite(q_end).all() and np.isfinite(QD[-1]).all()),
        **scene_validity(ctx, times, Q, QD),
    }
    return out


def off_table(ctx, q_first, q_end):
    """The bodies gone at the end, by Example.test_post_step's tabletop bounds (pyramid: fell through the ground or
    left the stack's footprint)."""
    if ctx["kind"] == "clutter":
        return (np.max(np.abs(q_end[..., :3]), axis=-1) > 3.0) | (q_end[..., 2] < -0.10)
    return (q_end[..., 2] < -0.10) | (np.linalg.norm(q_end[..., :3] - q_first[..., :3], axis=-1) > 3.0)


def scene_validity(ctx, times, Q, QD, *, after=None):
    """Whether a run's scene stayed valid: what left it, and how (see README, "Scene validity").

    A clutter scene's workbench is finite with no floor, so a body rolling off its edge is geometry, not an ejection:
    ``rolled_off`` counts the bodies gone at the end that left the table's top below ROLLED_OFF_SPEED, ``left_fast`` the
    others. ``launched`` counts objects that rose over LAUNCH_RISE_M within LAUNCH_WINDOW_S after settle, or, while the
    robot holds still (clutter), kept that rise's energy for two samples running. ``energy_spikes`` counts objects
    whose energy jumped that far for a single sample: a body that reads 0.8 m/s once and has barely moved was not
    launched, though the solve gave it the energy. All are relative to the support body, and sampled states can miss
    what happens between samples. ``after`` overrides the settle time. ``G`` comes from a run's own audit.
    """
    windows = ctx["windows"]
    rest = windows.get("rest")
    if after is None:
        after = rest[0] if rest else windows.get("settle", (0.0, 0.0))[1]
    objects = sorted({b for members in ctx["groups"].values() for b in members})
    gone = off_table(ctx, Q[0], Q[-1])
    rolled, fast = 0, 0
    speeds = []
    for world, body in zip(*np.nonzero(gone), strict=True):
        if ctx["kind"] != "clutter":
            fast += 1
            continue
        on_top = np.flatnonzero(Q[:, world, body, 2] > TABLE_PLANE_M)
        speed = float(np.linalg.norm(QD[on_top[-1] if on_top.size else 0, world, body, :2]))
        speeds.append(speed)
        if speed < ROLLED_OFF_SPEED:
            rolled += 1
        else:
            fast += 1
    step = float(np.median(np.diff(times))) if times.size > 1 else 0.0
    lag = max(round(LAUNCH_WINDOW_S / step), 1) if step > 0.0 else 1
    first = int(np.searchsorted(times, after - 1e-9))
    launched_mask = np.zeros(Q.shape[1:3], dtype=bool)
    spike_mask = np.zeros(Q.shape[1:3], dtype=bool)
    largest_rise = None
    if times.size > first + lag:
        support = ctx.get("support_body")
        base = Q[:, :, support, 2][:, :, None] if support is not None else 0.0
        height = Q[:, :, :, 2] - base
        rise = (height[first + lag :] - height[first:-lag]).max(axis=0)
        launched_mask = rise > LAUNCH_RISE_M
        largest_rise = float(rise[:, objects].max())
        if rest and ctx["kind"] == "clutter":
            still = (times >= rest[0] - 1e-9) & (times <= rest[1] + 1e-9)
            indices = np.flatnonzero(still)
            if indices.size > lag + 1:
                energy = GRAVITY * height[indices] + 0.5 * np.sum(QD[indices, :, :, :3] ** 2, axis=-1)
                above = energy[lag:] - energy[:-lag] > GRAVITY * LAUNCH_RISE_M
                held = (above[:-1] & above[1:]).any(axis=0)
                launched_mask |= held
                spike_mask = above.any(axis=0) & ~held
    return {
        "off_table": int(np.count_nonzero(gone)),
        "rolled_off": rolled,
        "left_fast": fast,
        "launched": int(np.count_nonzero(launched_mask[:, objects])),
        "energy_spikes": int(np.count_nonzero(spike_mask[:, objects])),
        "edge_speed_max": max(speeds) if speeds else None,
        "largest_rise_m": largest_rise,
    }


def audit_section(raw):
    """A run's own audit (``pgs_audit``): the mean energy gap per step and the largest dual violation, per world."""
    gap = raw.get("audit_gap_sum")
    if gap is None:
        return None
    steps = np.maximum(np.asarray(raw["audit_steps"], dtype=np.float64), 1.0)
    readable = np.asarray(raw["audit_steps"]) > 0
    if not readable.any():
        return None
    per_step = np.asarray(gap, dtype=np.float64)[readable] / steps[readable]
    violation = np.asarray(raw["audit_violation_max"], dtype=np.float64)[readable]
    return {
        "worlds": int(readable.sum()),
        "gap_per_step_j": {"p50": float(np.median(per_step)), "p99": float(np.percentile(per_step, 99))},
        "violation_max_m_s": {"p50": float(np.median(violation)), "p99": float(np.percentile(violation, 99))},
    }


def _get(node, *path):
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def physics_flags(physics, penetration, run, horizon_own) -> list:
    """Physical failures (results, not invalid runs): shown next to the verdict, never hidden."""
    flags = []
    end = physics.get("horizon_end", {})
    if not physics.get("finite_samples", True) or end.get("finite") is False:
        flags.append("nonfinite")
    if horizon_own and run.get("final", {}).get("finite_end") is False:
        flags.append("nonfinite_all_worlds")
    if end.get("launched"):
        flags.append(f"launched:{end['launched']}")
    if end.get("left_fast"):
        flags.append(f"left_fast:{end['left_fast']}")
    for label in ("rest_end", "end"):
        below = _get(physics, f"containment.{label}", "below_floor", "max")
        if below:
            flags.append(f"below_floor@{label}:{below:g}")
    lost_after = _get(physics, "containment.end", "worlds_lost_objects_after_rest")
    if lost_after:
        flags.append(f"lost_after_rest:{lost_after}w")
    strokes = physics.get("strokes", {})
    if strokes.get("grasp_lost_worlds"):
        flags.append(f"grasp_lost:{strokes['grasp_lost_worlds']}/{physics.get('worlds')}w")
    deep = _get(penetration, "all", "deep_count_max")
    if deep:
        flags.append(f"deep_contacts:{deep}")
    moved = _get(physics, "pyramid", "moved_gt_10mm_in_rest", "max")
    if moved:
        flags.append(f"moved>10mm:{moved:g}")
    collapsed = _get(physics, "pyramid", "collapsed_worlds")
    if collapsed:
        flags.append(f"collapsed:{collapsed}w")
    return flags


def summary_rows(ctx, sections, run) -> list:
    """Flat, ordered rows that compare.py prints side by side (patch vs point).

    ``compare``: ratio (patch/point), diff (patch - point; counts) or none. ``window``: the window
    the value belongs to (compare.py marks rows of incomplete windows). ``floor``: the float32
    resolution under which both arms' values are noise. ``run_level``: covers the arm's whole run,
    so it cannot be clipped to a common horizon.
    """
    rows = []

    def add(key, label, value, compare="ratio", **extra):
        if value is None:
            return
        row = {"key": key, "label": label, "value": value, "compare": compare}
        row.update({k: v for k, v in extra.items() if v is not None})
        rows.append(row)

    def stat(key, label, entry, name, compare="ratio", **extra):
        if isinstance(entry, dict) and name in entry:
            # IQR of per-world medians: printed next to median rows only (it is not a spread of p95 or max).
            spread = (
                [entry["world_median_q25"], entry["world_median_q75"]]
                if name == "median" and "world_median_q25" in entry
                else None
            )
            add(
                key,
                label,
                entry[name],
                compare,
                n=entry.get("n"),
                nonfinite=entry.get("nonfinite"),
                spread=spread,
                **extra,
            )

    timing = sections["timing"]["windows"]
    for name, entry in timing.items():
        if entry.get("n"):
            add(f"time.{name}.median_ms", f"ms/frame median [{name}]", entry["median_ms"], n=entry["n"], window=name)
    add("time.all.mean_ms", "ms/frame mean [all]", _get(timing, "all", "mean_ms"), window="all")
    add(
        "time.all.ms_per_substep", "ms/substep median [all]", _get(timing, "all", "median_ms_per_substep"), window="all"
    )
    c = run.get("construction", {})
    add("construct.scene_s", "construction s (scene + solver)", c.get("scene_s"), run_level=True)
    add("construct.solver_init_s", "solver __init__ s", c.get("solver_init_s"), run_level=True)
    add("construct.capture_s", "graph capture s (module loads incl.)", c.get("capture_s"), run_level=True)
    add(
        "construct.new_kernel_files",
        "kernel cache files written (cold-cache indicator)",
        c.get("new_kernel_files"),
        "none",
        run_level=True,
    )
    for name, entry in sections["counts"].items():
        add(
            f"contacts.{name}.median",
            f"contacts/world median [{name}]",
            entry["contacts_per_world_median"],
            window=name,
        )
        for row_name in ("mf", "dense"):
            add(
                f"rows.{name}.{row_name}",
                f"{row_name} rows/world median [{name}]",
                entry.get(f"{row_name}_rows_per_world_median"),
                window=name,
            )
            add(
                f"rows.{name}.{row_name}_max",
                f"{row_name} rows/world max [{name}]",
                entry.get(f"{row_name}_rows_per_world_max"),
                window=name,
            )
    w = run.get("watermarks", {})
    for key in ("mf_high_water", "dense_high_water", "contact_high_water"):
        add(f"rows.watermark.{key}", f"watermark {key} (whole run)", w.get(key), run_level=True)
    for key, value in sections["census"].items():
        add(f"rows.census.{key}", key.replace("_", " "), value)

    physics = sections["physics"]
    for key, entry in physics.items():
        if not (key.startswith("rest.") and key not in ("rest.container", "rest.motion", "rest.spilled")):
            continue
        group = key.split(".", 1)[1]
        drift_floor, creep_floor = entry.get("drift_floor_mm", {}), entry.get("creep_deadband_mm_s", {})
        stat(
            f"{key}.drift_mm.median",
            f"rest {group}: drift mm median",
            entry.get("drift_mm"),
            "median",
            window="rest",
            floor=drift_floor.get("median"),
        )
        stat(
            f"{key}.drift_mm.p95",
            f"rest {group}: drift mm p95",
            entry.get("drift_mm"),
            "p95",
            window="rest",
            floor=drift_floor.get("max"),
        )
        stat(
            f"{key}.drift_mm.max",
            f"rest {group}: drift mm max",
            entry.get("drift_mm"),
            "max",
            window="rest",
            floor=drift_floor.get("max"),
        )
        stat(
            f"{key}.creep_mm_s.median",
            f"rest {group}: creep mm/s median",
            entry.get("creep_mm_s"),
            "median",
            window="rest",
            floor=creep_floor.get("median"),
        )
        stat(
            f"{key}.creep_mm_s.p95",
            f"rest {group}: creep mm/s p95",
            entry.get("creep_mm_s"),
            "p95",
            window="rest",
            floor=creep_floor.get("max"),
        )
        stat(
            f"{key}.creep_first_half.median",
            f"rest {group}: creep 1st half mm/s median",
            entry.get("creep_first_half_mm_s"),
            "median",
            window="rest",
            floor=creep_floor.get("median"),
        )
        stat(
            f"{key}.creep_second_half.median",
            f"rest {group}: creep 2nd half mm/s median",
            entry.get("creep_second_half_mm_s"),
            "median",
            window="rest",
            floor=creep_floor.get("median"),
        )
        stat(
            f"{key}.rotation_deg.p95",
            f"rest {group}: rotation deg p95",
            entry.get("rotation_deg"),
            "p95",
            window="rest",
        )
        stat(
            f"{key}.drift_floor_mm",
            f"rest {group}: float32 drift floor mm (median)",
            drift_floor,
            "median",
            "none",
            window="rest",
        )
        stat(
            f"{key}.creep_deadband_mm_s",
            f"rest {group}: float32 creep dead band mm/s (median)",
            creep_floor,
            "median",
            "none",
            window="rest",
        )
    spilled = physics.get("rest.spilled", {})
    stat(
        "rest.spilled.drift_mm.median",
        "rest spilled (on bench): drift mm median",
        spilled.get("drift_mm"),
        "median",
        window="rest",
    )
    stat(
        "rest.container.drift_mm.max",
        "rest container drift mm max",
        _get(physics, "rest.container", "drift_mm"),
        "max",
        window="rest",
    )
    motion = physics.get("rest.motion", {})
    stat(
        "rest.motion.ke",
        "objects KE, rest-window mean, J/world median",
        motion.get("ke_J_per_world_mean"),
        "median",
        window="rest",
    )
    stat(
        "rest.motion.speed_t",
        "object tangential speed, rest mean, mm/s median",
        motion.get("speed_tangential_mm_s"),
        "median",
        window="rest",
    )
    stat(
        "rest.motion.speed_n",
        "object normal speed, rest mean, mm/s median",
        motion.get("speed_normal_mm_s"),
        "median",
        window="rest",
    )
    stat(
        "rest.motion.mean_velocity",
        "object |mean velocity| over rest, mm/s median",
        motion.get("mean_velocity_mm_s"),
        "median",
        window="rest",
    )

    pyramid = physics.get("pyramid", {})
    stat(
        "pyramid.from_build_rest_end",
        "boxes: displacement from build pose at rest end, mm median",
        pyramid.get("from_build_rest_end_mm"),
        "median",
        window="rest",
    )
    stat(
        "pyramid.from_impact_rest_end",
        "boxes: displacement from post-impact pose at rest end, mm median",
        pyramid.get("from_impact_rest_end_mm"),
        "median",
        window="rest",
    )
    stat(
        "pyramid.top_from_build",
        "top box: displacement from build pose at end, mm max",
        pyramid.get("top_from_build_end_mm"),
        "max",
    )
    stat(
        "pyramid.top_from_settled",
        "top box: displacement from settled pose at end, mm max",
        pyramid.get("top_from_settled_end_mm"),
        "max",
    )
    stat(
        "pyramid.top_height_deficit",
        "top box: height below settled pose at end, mm median",
        pyramid.get("top_height_deficit_end_mm"),
        "median",
    )
    add(
        "pyramid.collapsed_worlds",
        f"worlds whose top box is >= {COLLAPSE_M} m from its settled pose (collapsed)",
        pyramid.get("collapsed_worlds"),
        "diff",
    )
    stat(
        "pyramid.moved_gt_10mm",
        f"boxes moved > {MOVED_MM:g} mm within rest (max/world)",
        pyramid.get("moved_gt_10mm_in_rest"),
        "max",
        "diff",
        window="rest",
    )

    for name in ("all", "rest"):
        pen = sections["penetration"].get(name, {})
        add(
            f"penetration.{name}.max_mm",
            f"object penetration max mm [{name}]",
            pen.get("max_mm"),
            window=None if name == "all" else name,
        )
        add(
            f"penetration.{name}.p99_mm",
            f"object penetration p99 mm (median snapshot) [{name}]",
            pen.get("p99_mm_median"),
            window=None if name == "all" else name,
        )
    add(
        "penetration.all.deep",
        f"object contacts deeper than {DEEP_MM} mm (max snapshot)",
        _get(sections["penetration"], "all", "deep_count_max"),
        "diff",
    )
    add(
        "penetration.all.other_max_mm",
        "robot/fixture penetration max mm (not objects)",
        _get(sections["penetration"], "all", "other_max_mm"),
        "none",
    )

    s = physics.get("strokes", {})
    stat(
        "strokes.slip.max",
        "TCP slip in container, max mm (median world)",
        s.get("tcp_slip_in_container_max_mm"),
        "median",
        window="strokes",
    )
    add(
        "strokes.lost",
        f"worlds with slip > {LOST_GRASP_MM:g} mm (of {physics.get('worlds')})",
        s.get("grasp_lost_worlds"),
        "diff",
        window="strokes",
    )
    stat(
        "strokes.travel",
        "container travel along shake mm (median)",
        s.get("container_travel_shake_axis_mm"),
        "median",
        window="strokes",
    )
    stat("strokes.tracking", "container/TCP travel ratio (median)", s.get("tracking_ratio"), "median", window="strokes")
    for label in ("rest_start", "rest_end", "end"):
        entry = physics.get(f"containment.{label}", {})
        stat(
            f"containment.{label}.outside",
            f"objects outside container at {label} (max/world)",
            entry.get("outside"),
            "max",
            "diff",
        )
        stat(
            f"containment.{label}.below_floor",
            f"objects below container floor at {label} (max/world)",
            entry.get("below_floor"),
            "max",
            "diff",
        )
    add(
        "containment.lost_after_rest",
        "worlds with more objects outside at end than at rest end",
        _get(physics, "containment.end", "worlds_lost_objects_after_rest"),
        "diff",
    )
    cube = physics.get("cube", {})
    stat(
        "cube.yaw_turn",
        "cube top yaw at turn end deg (median)",
        cube.get("top_yaw_turn_end_deg"),
        "median",
        "diff",
        window="turn",
    )
    stat("cube.yaw_end", "cube top yaw at end deg (median)", cube.get("top_yaw_end_deg"), "median", "diff")
    stat("cube.lower", "cube lower layer rotation max deg", cube.get("lower_layer_rotation_max_deg"), "max", "diff")
    end = physics.get("horizon_end", {})
    add(
        "end.finite",
        "finite state (metric worlds, samples and end)",
        bool(physics.get("finite_samples")) and end.get("finite", False),
        "none",
    )
    add("end.rolled_off", "bodies rolled off the table (metric worlds)", end.get("rolled_off"), "diff")
    add("end.left_fast", "bodies that left the table fast (metric worlds)", end.get("left_fast"), "diff")
    add("end.launched", "objects launched after settle (metric worlds)", end.get("launched"), "diff")
    add(
        "end.energy_spikes",
        "objects with a one-sample energy spike at rest (metric worlds)",
        end.get("energy_spikes"),
        "diff",
    )
    audit = sections.get("audit") or {}
    add("end.audit_gap_p50", "audit gap per step, p50 [J]", _get(audit, "gap_per_step_j", "p50"), "ratio")
    add("end.audit_gap_p99", "audit gap per step, p99 [J]", _get(audit, "gap_per_step_j", "p99"), "ratio")
    add("end.audit_violation_p99", "audit dual violation, p99 [m/s]", _get(audit, "violation_max_m_s", "p99"), "ratio")
    final = run.get("final", {})
    add(
        "run.off_table_all_worlds",
        "bodies off the table at run end (all worlds, not classified: only metric worlds are sampled)",
        final.get("escaped_end"),
        "diff",
        run_level=True,
    )
    add(
        "run.finite_all_worlds", "finite state at run end (all worlds)", final.get("finite_end"), "none", run_level=True
    )
    flags = run.get("flags", {})
    add("flags.law_verified", "law verified", flags.get("law_verified"), "none", run_level=True)
    add("flags.capacity_overflow", "capacity overflow flagged", flags.get("capacity_overflow"), "none", run_level=True)
    add("flags.truncated", "truncated by budget", flags.get("truncated"), "none", run_level=True)
    add(
        "flags.gpu_contention",
        "foreign GPU compute process seen during the run",
        flags.get("gpu_contention"),
        "none",
        run_level=True,
    )
    return rows


def analyze(ctx, raw, run, horizon_frames=None) -> dict:
    """All result sections for frames [0, horizon_frames) of a run (default: every frame it ran)."""
    frames_run = len(raw["frame_ms"])
    frames = frames_run if horizon_frames is None else max(0, min(frames_run, int(horizon_frames)))
    t_end = frames / ctx["fps"]
    snapshots = raw.get("snapshots", [])
    sections = {
        "horizon": {
            "frames": frames,
            "t": t_end,
            "frames_run": frames_run,
            "frames_planned": ctx["frames_planned"],
            "own": frames == frames_run,
            "complete": frames >= ctx["frames_planned"],
        },
        "timing": timing_section(ctx, raw, frames),
        "counts": counts_section(ctx, raw, frames),
        "penetration": penetration_section(snapshots, ctx["windows"], t_end),
        "census": census_section(snapshots, ctx["law"], t_end),
        "physics": physics_section(ctx, raw, t_end),
        "audit": audit_section(raw),
    }
    sections["physics_flags"] = physics_flags(
        sections["physics"], sections["penetration"], run, sections["horizon"]["own"]
    )
    sections["summary"] = summary_rows(ctx, sections, run)
    return sections


def load_raw(npz_path, snapshots) -> dict:
    """The raw arrays a run saved (``run_complex.save_raw``) plus its JSON snapshots."""
    with np.load(npz_path) as data:
        raw = {key: data[key] for key in data.files}
    for key in ("rows_mf", "rows_dense"):
        if key in raw and raw[key].size == 0:
            raw[key] = None
    raw["snapshots"] = snapshots
    return raw
