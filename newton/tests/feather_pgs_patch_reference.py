# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Full-capacity reference of the persistent patch-friction mechanism.

The kernels below are verbatim copies of the patch kernels at 98401301: pair
keys ``(ka << 32) | kb``, every launch over the whole contact capacity, one row
kernel launch per contact route, and the reset kernel over the whole history.
Later versions launch over the live contacts instead; tests compare them with
this reference, decoding pair keys through :func:`pair_codes`: bitwise on the
CPU, and on CUDA with computed float payloads within the FMA-contraction bound
of ``assert_same_patches``, because NVRTC may contract this module's
multiply-adds differently from the patch module's. Device
functions that the mechanism work leaves unchanged are imported. ``_build``
follows the builder's current anchor choice (the canonical farthest pair, the
footprint tolerance, the support-edge members and their carriers), which is
not mechanism work. This module holds shared test helpers; it defines no tests.
"""

from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.contact_filters import contact_normal_gap_limit
from newton._src.solvers.feather_pgs.contact_laws import contact_tangent_basis
from newton._src.solvers.feather_pgs.friction_patches import (
    _FOOTPRINT_TOLERANCE,
    FrictionPatches,
    _carried_displacement,
    _compatible,
    _geometry_adjacent,
    _local_point,
    _PatchFrame,
    _point_velocity,
    _pose_motion,
    _world_point,
    patch_normal_load,
    patch_routes,
)

SENTINEL = 0x7FFFFFFFFFFFFFFF
# The solved state a finish without patch rows takes: with every route disabled it
# reads neither the state nor the row arrays.
_UNSOLVED = SimpleNamespace(body_q=None, body_qd=None)


def store_frame(patches, state):
    """Store the patch state's built frame without finishing an anchor, as a solver step without patch rows does.

    The step finishes with every route disabled (``patch_routes(())``), which stores
    the frame as built at the poses of ``state``.
    """
    patches.finish(state, _UNSOLVED, None, None, None, None, None, patch_routes(()), 0.0)


def pair_codes(keys, stride=None):
    """Decode sorted pair keys into ``(ka, kb)`` codes; empty slots decode to ``(-1, -1)``.

    ``stride`` selects compressed keys ``ka * stride + kb``; ``None`` selects this
    reference's shifted keys.
    """
    keys = np.asarray(keys, dtype=np.int64)
    live = keys != SENTINEL
    codes = np.full((len(keys), 2), -1, dtype=np.int64)
    if stride is None:
        codes[live, 0], codes[live, 1] = keys[live] >> 32, keys[live] & 0xFFFFFFFF
    else:
        codes[live, 0], codes[live, 1] = np.divmod(keys[live], stride)
    return codes


@wp.kernel(enable_backward=False)
def _prepare(
    count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    shape_body: wp.array[int],
    shape_mu: wp.array[float],
    shape_gap: wp.array[float],
    body_radius: wp.array[float],
    q: wp.array[wp.transform],
    body_to_articulation: wp.array[int],
    is_free_rigid: wp.array[int],
    contact_gap_gate: float,
    same_articulation_gap_gate: float,
    articulation_pair_gap_gate: float,
    friction_gap: float,
    frame: _PatchFrame,
    patches: FrictionPatches,
):
    c = wp.tid()
    frame.indices[c] = c
    frame.keys[c] = wp.int64(0x7FFFFFFFFFFFFFFF)
    frame.owner[c] = -1
    frame.valid[c] = 0
    frame.used[c] = 0
    frame.source[c] = -1
    frame.eligible[c] = 0
    frame.tangent_impulse[c] = wp.vec3(0.0)
    patches.weight[c] = 0.0
    patches.next_contact[c] = -1
    patches.phi[c] = wp.vec2(0.0)
    if count[0] > shape0.shape[0] or c >= count[0] or c >= shape0.shape[0]:
        return
    sa = shape0[c]
    sb = shape1[c]
    a = int(-1)
    b = int(-1)
    if sa >= 0:
        a = shape_body[sa]
    if sb >= 0:
        b = shape_body[sb]
    ka = a + 1
    kb = b + 1
    if a < 0:
        ka = q.shape[0] + (sa if sa >= 0 else shape_body.shape[0]) + 1
    if b < 0:
        kb = q.shape[0] + (sb if sb >= 0 else shape_body.shape[0]) + 1
    n = -normal[c]
    pa = _world_point(q, a, point0[c]) - margin0[c] * n
    pb = _world_point(q, b, point1[c]) + margin1[c] * n
    gap = wp.dot(n, pa - pb)
    art_a = int(-1)
    art_b = int(-1)
    if a >= 0 and a < body_to_articulation.shape[0]:
        art_a = body_to_articulation[a]
    if b >= 0 and b < body_to_articulation.shape[0]:
        art_b = body_to_articulation[b]
    a_non_free = art_a >= 0 and is_free_rigid[art_a] == 0
    b_non_free = art_b >= 0 and is_free_rigid[art_b] == 0
    support_gap_limit = float(0.0)
    if sa >= 0:
        support_gap_limit += shape_gap[sa]
    if sb >= 0:
        support_gap_limit += shape_gap[sb]
    normal_gap_limit = contact_normal_gap_limit(
        a_non_free,
        b_non_free,
        art_a == art_b,
        contact_gap_gate,
        articulation_pair_gap_gate,
        same_articulation_gap_gate,
    )
    support_gap_limit = wp.min(support_gap_limit, normal_gap_limit)
    eligible = gap <= normal_gap_limit and gap <= friction_gap
    # Friction-only filters suppress rows without ending contact support;
    # filtered regions must still be able to carry their anchor history.
    frame.support_gap_limit[c] = wp.max(support_gap_limit, 0.0)
    frame.eligible[c] = int(eligible)
    flip = int(ka > kb)
    if flip != 0:
        t = a
        a = b
        b = t
        t = ka
        ka = kb
        kb = t
        t = sa
        sa = sb
        sb = t
        n = -n
    frame.keys[c] = (wp.int64(ka) << wp.int64(32)) | wp.int64(kb)
    frame.body_a[c] = a
    frame.body_b[c] = b
    frame.shape_a[c] = sa
    frame.shape_b[c] = sb
    frame.flipped[c] = flip
    frame.surface_a[c] = _local_point(q, a, pb if flip != 0 else pa)
    frame.surface_b[c] = _local_point(q, b, pa if flip != 0 else pb)
    frame.center[c] = 0.5 * (pa + pb)
    frame.normal[c] = n
    mu_a = float(0.0)
    mu_b = float(0.0)
    if sa >= 0:
        mu_a = shape_mu[sa]
    if sb >= 0:
        mu_b = shape_mu[sb]
    frame.mu[c] = wp.vec2(mu_a, mu_b)
    r = float(1.0e10)
    if a >= 0:
        r = wp.min(r, body_radius[a])
    if b >= 0:
        r = wp.min(r, body_radius[b])
    frame.radius[c] = wp.max(r, 1.0e-4)


@wp.kernel(enable_backward=False)
def _build(
    count: wp.array[int],
    q: wp.array[wp.transform],
    previous_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_radius: wp.array[float],
    frame: _PatchFrame,
    prev: _PatchFrame,
    patches: FrictionPatches,
):
    """One thread per sorted body pair; writes disjoint current/previous records."""
    start = wp.tid()
    end = wp.min(count[0], frame.center.shape[0])
    if start >= end or (start > 0 and frame.keys[start - 1] == frame.keys[start]):
        return
    key = frame.keys[start]
    if key == wp.int64(0x7FFFFFFFFFFFFFFF):
        return
    stop = start + 1
    while stop < end and frame.keys[stop] == key:
        stop += 1

    # Binary search the previous pair, avoiding a scan over other worlds/pairs.
    lo = int(0)
    hi = prev.center.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if prev.keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    prev_start = lo
    prev_stop = lo
    # Index valid history once per pair rather than rescanning all old contacts
    # for every new friction location. Keep sorted order for matching ties.
    prev_anchor_stop = prev_start
    while prev_stop < prev.center.shape[0] and prev.keys[prev_stop] == key:
        p = prev.indices[prev_stop]
        prev.used[p] = 0
        if prev.valid[p] != 0:
            prev.members[prev_anchor_stop] = p
            prev_anchor_stop += 1
        prev_stop += 1

    member_stop = start
    remaining = stop - start
    for index in range(start, stop):
        seed = frame.indices[index]
        if frame.owner[seed] >= 0:
            continue
        frame.owner[seed] = seed
        remaining -= 1
        tail = seed
        first = int(-1)
        if frame.eligible[seed] != 0:
            first = seed
        cursor = seed
        expanded_a = int(-2)
        expanded_b = int(-2)
        # Flood through adjacent convex pieces. The queue is the contact list
        # itself; no allocation, atomics, or fixed patch-count limit is needed.
        while cursor >= 0 and remaining > 0:
            sa = frame.shape_a[cursor]
            sb = frame.shape_b[cursor]
            if sa != expanded_a or sb != expanded_b:
                for j in range(index + 1, stop):
                    c = frame.indices[j]
                    if frame.owner[c] < 0 and _compatible(frame, seed, c):
                        if _geometry_adjacent(
                            sa, sb, frame.shape_a[c], frame.shape_b[c], shape_transform, shape_radius
                        ):
                            frame.owner[c] = seed
                            remaining -= 1
                            patches.next_contact[tail] = c
                            tail = c
                            # Sorted order within a pair is contact order, so the
                            # lowest index is the first eligible member in it.
                            if frame.eligible[c] != 0 and (first < 0 or c < first):
                                first = c
                expanded_a = sa
                expanded_b = sb
            cursor = patches.next_contact[cursor]
        patches.next_contact[tail] = seed
        carry_only = int(0)
        if first < 0:
            # No member may carry friction rows this step (gap filters). Keep the
            # region's anchor history on its members with zero weight so a single
            # filtered step does not forget the accumulated positional correction.
            carry_only = int(1)
            first = seed
        # The canonical farthest pair: the member farthest from the first
        # eligible one, then the member farthest from that. Member spans and
        # contact order define it, so neither roundoff in world coordinates nor
        # the footprint's pose can swap its diagonal; reordering the contacts
        # can. A choice independent of contact order kicked resting boxes whose
        # contacts were reordered every step harder (measured).
        reference = first
        separation = float(0.0)
        member_start = member_stop
        # Preserve sorted contact order for footprint sums and tie breaking.
        # The collision list follows flood order and must remain unchanged.
        for j in range(index, stop):
            c = frame.indices[j]
            if frame.owner[c] != seed:
                continue
            frame.members[member_stop] = c
            member_stop += 1
            if carry_only != 0 or frame.eligible[c] != 0:
                distance = wp.length_sq(frame.center[c] - frame.center[reference])
                if distance > separation * (1.0 + _FOOTPRINT_TOLERANCE):
                    separation = distance
                    first = c
        anchors = int(1)
        if separation > 1.0e-8 * frame.radius[seed] * frame.radius[seed]:
            anchors = 2
        else:
            first = reference
        second = first
        location0 = frame.center[first]
        location1 = location0
        if anchors == 2:
            # Use the footprint's principal extent, averaging its support edges.
            # Picking opposite corners of a narrow rectangle introduces an
            # artificial diagonal friction couple even when the footprint is symmetric.
            origin = location0
            t0, t1 = contact_tangent_basis(frame.normal[seed])
            # Accumulate moments in double precision: contact-order roundoff
            # must not give an isotropic footprint a spurious principal axis.
            # Raw moments also avoid a separate pass to compute the centroid.
            sum_x = wp.float64(0.0)
            sum_y = wp.float64(0.0)
            sum_xx = wp.float64(0.0)
            sum_xy = wp.float64(0.0)
            sum_yy = wp.float64(0.0)
            members = int(0)
            reach = float(0.0)
            for j in range(member_start, member_stop):
                c = frame.members[j]
                if carry_only != 0 or frame.eligible[c] != 0:
                    offset = frame.center[c] - origin
                    distance = wp.length_sq(offset)
                    if distance > reach * (1.0 + _FOOTPRINT_TOLERANCE):
                        reach = distance
                        second = c
                    x = wp.float64(wp.dot(offset, t0))
                    y = wp.float64(wp.dot(offset, t1))
                    sum_x += x
                    sum_y += y
                    sum_xx += x * x
                    sum_xy += x * y
                    sum_yy += y * y
                    members += 1
            location1 = frame.center[second]
            member_count = wp.float64(members)
            xx = float(sum_xx - sum_x * sum_x / member_count)
            xy = float(sum_xy - sum_x * sum_y / member_count)
            yy = float(sum_yy - sum_y * sum_y / member_count)
            spread = wp.sqrt((xx - yy) * (xx - yy) + 4.0 * xy * xy)
            # An isotropic footprint has no distinguished axis; retain its
            # canonical farthest pair instead of amplifying roundoff. Carry-only
            # regions keep that pair too.
            if carry_only == 0 and spread > _FOOTPRINT_TOLERANCE * (xx + yy):
                largest = 0.5 * (xx + yy + spread)
                axis2 = wp.vec2(xy, largest - xx)
                if xx >= yy:
                    axis2 = wp.vec2(largest - yy, xy)
                axis2 = wp.normalize(axis2)
                axis = axis2[0] * t0 + axis2[1] * t1
                low = float(1.0e30)
                high = float(-1.0e30)
                for j in range(member_start, member_stop):
                    c = frame.members[j]
                    if frame.eligible[c] != 0:
                        projection = wp.dot(frame.center[c] - origin, axis)
                        low = wp.min(low, projection)
                        high = wp.max(high, projection)
                sum0 = wp.vec3d(0.0)
                sum1 = wp.vec3d(0.0)
                count0 = int(0)
                count1 = int(0)
                # Tied corners differ along the axis by the quantization of
                # world coordinates and by the axis tilt it causes, which
                # outgrow a fixed fraction of the body radius a few meters
                # from the origin.
                edge_tolerance = wp.max(1.0e-5 * frame.radius[seed], _FOOTPRINT_TOLERANCE * (high - low))
                for j in range(member_start, member_stop):
                    c = frame.members[j]
                    if frame.eligible[c] != 0:
                        point = frame.center[c]
                        offset = point - origin
                        projection = wp.dot(offset, axis)
                        precise_point = wp.vec3d(wp.float64(point[0]), wp.float64(point[1]), wp.float64(point[2]))
                        # Each edge's last member in contact order carries its
                        # rows, not the strict extreme, which roundoff picks
                        # among tied corners. Where rows follow contact order, a
                        # Gauss-Seidel sweep reaches the anchor's friction rows
                        # after every normal row of its edge; the first member
                        # splits them, and the fixed lag in the anchors' pooled
                        # budgets yaws sliding boxes at low iteration counts.
                        if projection <= low + edge_tolerance:
                            first = c
                            sum0 += precise_point
                            count0 += 1
                        if projection >= high - edge_tolerance:
                            second = c
                            sum1 += precise_point
                            count1 += 1
                # Preserve constant coordinates and single-point edges exactly;
                # avoid subtracting and adding the origin around their average.
                mean0 = sum0 / wp.float64(count0)
                mean1 = sum1 / wp.float64(count1)
                location0 = wp.vec3(float(mean0[0]), float(mean0[1]), float(mean0[2]))
                location1 = wp.vec3(float(mean1[0]), float(mean1[1]), float(mean1[2]))
        for aidx in range(anchors):
            c = first
            if aidx == 1:
                c = second
            location = location0
            if aidx == 1:
                location = location1
            a = frame.body_a[c]
            b = frame.body_b[c]
            n = frame.normal[c]
            r = frame.radius[c]
            surface_a = frame.surface_a[c]
            surface_b = frame.surface_b[c]
            current_gap = wp.dot(_world_point(q, a, surface_a) - _world_point(q, b, surface_b), n)
            chosen = int(-1)
            nearest = float(1.0e30)
            motion = wp.vec3(0.0)
            if prev_start < prev_stop:
                motion = _pose_motion(q, previous_q, a, location) - _pose_motion(q, previous_q, b, location)
            for j in range(prev_start, prev_anchor_stop):
                p = prev.members[j]
                if (
                    prev.valid[p] == 0
                    or prev.used[p] != 0
                    or prev.mu[p][0] != frame.mu[c][0]
                    or prev.mu[p][1] != frame.mu[c][1]
                ):
                    continue
                old_n = prev.normal[p]
                if a >= 0:
                    old_n = wp.transform_vector(q[a], old_n)
                # Reject incompatible normals and positions before transporting
                # the spring or evaluating support witnesses. These are the same
                # gates as below, ordered to avoid work for rejected history.
                if not (wp.dot(old_n, n) >= 0.995):
                    continue
                pa = _world_point(q, a, prev.anchor_a[p])
                pb = _world_point(q, b, prev.anchor_b[p])
                delta = pa - pb
                offset = 0.5 * (pa + pb) - location
                distance = wp.length_sq(offset)
                if not (
                    distance < nearest
                    and distance <= 4.0 * r * r
                    and wp.abs(wp.dot(delta, n)) <= 0.1 * r
                    and wp.abs(wp.dot(offset, n)) <= 0.05 * r
                ):
                    continue
                # Honor the existing contact envelope: a zero-gap test drops
                # valid speculative contacts during a squeeze. Surface
                # witnesses retain penetration during decompression and
                # reject rocking beyond the same envelope as fresh contacts.
                support_gap = wp.dot(_world_point(q, a, prev.surface_a[p]) - _world_point(q, b, prev.surface_b[p]), n)
                if not (support_gap <= wp.max(current_gap, frame.support_gap_limit[c]) + 1.0e-5 * r):
                    continue
                error = _carried_displacement(q, prev, p, n)
                error += motion
                tangent_delta = error - n * wp.dot(error, n)
                # The tangential gate bounds uncorrected slip of a carried pair at a
                # tenth of the smaller body's radius, about three times the measured
                # Baumgarte equilibrium separation of a held box at 60 Hz. Tighter
                # gates re-anchor below that equilibrium and turn the correction
                # into creep; genuine sliding is released by ``finish_patch_impulses``.
                if not (wp.length_sq(tangent_delta) <= 0.01 * r * r):
                    continue
                connected = _geometry_adjacent(
                    prev.shape_a[p], prev.shape_b[p], frame.shape_a[c], frame.shape_b[c], shape_transform, shape_radius
                )
                if not connected:
                    for k in range(member_start, member_stop):
                        member = frame.members[k]
                        if _geometry_adjacent(
                            prev.shape_a[p],
                            prev.shape_b[p],
                            frame.shape_a[member],
                            frame.shape_b[member],
                            shape_transform,
                            shape_radius,
                        ):
                            connected = True
                            break
                if connected:
                    chosen = p
                    nearest = distance
            pa = location
            pb = pa
            anchor_a = _local_point(q, a, pa)
            anchor_b = _local_point(q, b, pb)
            carried_impulse = wp.vec3(0.0)
            error = wp.vec3(0.0)
            if chosen >= 0:
                prev.used[chosen] = 1
                error = _carried_displacement(q, prev, chosen, n)
                old_center = 0.5 * (
                    _world_point(q, a, prev.anchor_a[chosen]) + _world_point(q, b, prev.anchor_b[chosen])
                )
                # The two spring samples describe a planar rigid displacement:
                # translation plus twist. Evaluate that field at the new points
                # instead of transferring old lever-arm errors independently.
                # A one-point region carries translation only.
                for j in range(prev_start, prev_anchor_stop):
                    partner = prev.members[j]
                    if partner != chosen and prev.valid[partner] != 0 and prev.owner[partner] == prev.owner[chosen]:
                        other_normal = prev.normal[partner]
                        if a >= 0:
                            other_normal = wp.transform_vector(q[a], other_normal)
                        other_gap = wp.dot(
                            _world_point(q, a, prev.surface_a[partner]) - _world_point(q, b, prev.surface_b[partner]), n
                        )
                        if (
                            wp.dot(other_normal, n) < 0.995
                            or other_gap > wp.max(current_gap, frame.support_gap_limit[c]) + 1.0e-5 * r
                        ):
                            break
                        other_center = 0.5 * (
                            _world_point(q, a, prev.anchor_a[partner]) + _world_point(q, b, prev.anchor_b[partner])
                        )
                        span = other_center - old_center
                        span -= n * wp.dot(span, n)
                        span_sq = wp.length_sq(span)
                        if span_sq > 1.0e-8 * r * r:
                            other_error = _carried_displacement(q, prev, partner, n)
                            spin = wp.dot(other_error - error, wp.cross(n, span)) / span_sq
                            error = 0.5 * (error + other_error) + spin * wp.cross(
                                n, pa - 0.5 * (old_center + other_center)
                            )
                        break
                error += motion
                error -= n * wp.dot(error, n)
                carried_impulse = prev.tangent_impulse[chosen]
            stored_error = error
            if a >= 0:
                stored_error = wp.transform_vector(wp.transform_inverse(q[a]), error)
            frame.displacement[c] = stored_error
            frame.source[c] = chosen
            if carry_only != 0 and chosen < 0:
                # Filtered members only carry existing history; they never start
                # it, because no friction row held the pair during this step.
                continue
            frame.anchor_a[c] = anchor_a
            frame.anchor_b[c] = anchor_b
            frame.surface_a[c] = surface_a
            frame.surface_b[c] = surface_b
            # Default for anchors without friction rows this step; solved rows
            # overwrite it in ``finish_patch_impulses``.
            frame.tangent_impulse[c] = carried_impulse
            frame.valid[c] = 1
            patches.weight[c] = 1.0 / float(anchors)
            if carry_only != 0:
                patches.weight[c] = 0.0
            if frame.flipped[c] != 0:
                n = -n
            patches.point[c] = pa
            t0, t1 = contact_tangent_basis(n)
            if frame.flipped[c] != 0:
                error = -error
            patches.phi[c] = wp.vec2(wp.dot(t0, error), wp.dot(t1, error))


@wp.kernel(enable_backward=False)
def _store_history(
    q: wp.array[wp.transform],
    body_world: wp.array[int],
    frame: _PatchFrame,
    prev: _PatchFrame,
    previous_world: wp.array[int],
    previous_q: wp.array[wp.transform],
):
    """Carry contact history and input poses, independent of solver velocity passes."""
    c = wp.tid()
    if c < previous_q.shape[0]:
        previous_q[c] = q[c]
    if c >= frame.valid.shape[0]:
        return
    # Pair searches need the complete sorted index, but only live anchors own
    # geometric history. Retire stale validity and reset ownership before exit.
    prev.keys[c] = frame.keys[c]
    prev.indices[c] = frame.indices[c]
    prev.valid[c] = frame.valid[c]
    previous_world[c] = -1
    if frame.valid[c] == 0:
        return
    normal = frame.normal[c]
    if frame.body_a[c] >= 0:
        normal = wp.transform_vector(wp.transform_inverse(q[frame.body_a[c]]), normal)
    prev.normal[c] = normal
    prev.displacement[c] = frame.displacement[c]
    prev.mu[c] = frame.mu[c]
    prev.body_a[c] = frame.body_a[c]
    prev.body_b[c] = frame.body_b[c]
    prev.owner[c] = frame.owner[c]
    prev.shape_a[c] = frame.shape_a[c]
    prev.shape_b[c] = frame.shape_b[c]
    prev.anchor_a[c] = frame.anchor_a[c]
    prev.anchor_b[c] = frame.anchor_b[c]
    prev.surface_a[c] = frame.surface_a[c]
    prev.surface_b[c] = frame.surface_b[c]
    prev.tangent_impulse[c] = frame.tangent_impulse[c]
    world = int(-1)
    if frame.body_a[c] >= 0:
        world = body_world[frame.body_a[c]]
    if world < 0 and frame.body_b[c] >= 0:
        world = body_world[frame.body_b[c]]
    # The solver maps the default, unpartitioned world (-1) to world zero.
    world = wp.max(world, 0)
    previous_world[c] = world


@wp.kernel(enable_backward=False)
def _invalidate_geometry_history(prev: _PatchFrame, bodies: wp.array[int], shapes: wp.array[int]):
    """Discard material points whose supporting geometry changed."""
    c = wp.tid()
    if prev.valid[c] == 0:
        return
    changed = bool(False)
    if prev.shape_a[c] >= 0:
        changed = shapes[prev.shape_a[c]] != 0
    if prev.shape_b[c] >= 0:
        changed = changed or shapes[prev.shape_b[c]] != 0
    if prev.body_a[c] >= 0:
        changed = changed or bodies[prev.body_a[c]] != 0
    if prev.body_b[c] >= 0:
        changed = changed or bodies[prev.body_b[c]] != 0
    if changed:
        prev.valid[c] = 0


@wp.kernel(enable_backward=False)
def link_patch_rows(
    count: wp.array[int],
    patches: FrictionPatches,
    world: wp.array[int],
    slot: wp.array[int],
    path: wp.array[int],
    slots_needed: wp.array[int],
    route: int,
    parents: wp.array2d[int],
    mu: wp.array2d[float],
):
    """Link only allocated rows and divide the Coulomb budget among surviving anchors.

    Row builders write the unscaled friction coefficient; this kernel divides it by
    the number of anchors that actually received rows, so the pooled normal load
    is shared exactly once.
    """
    c = wp.tid()
    if c >= count[0] or slot[c] < 0 or path[c] != route:
        return
    row = patches.next_contact[c]
    if row < 0:
        return
    next_slot = slot[c]
    anchors = int(patches.weight[c] > 0.0 and slots_needed[c] == 3)
    while row != c:
        if slot[row] >= 0 and path[row] == route and world[row] == world[c]:
            if next_slot == slot[c]:
                next_slot = slot[row]
            if patches.weight[row] > 0.0 and slots_needed[row] == 3:
                anchors += 1
            if patches.weight[c] == 0.0:
                break
        row = patches.next_contact[row]
    parents[world[c], slot[c]] = next_slot
    if patches.weight[c] > 0.0 and slots_needed[c] == 3 and anchors > 0:
        for t in range(1, 3):
            mu[world[c], slot[c] + t] /= float(anchors)


@wp.kernel(enable_backward=False)
def seed_patch_impulses(
    count: wp.array[int],
    frame: _PatchFrame,
    prev: _PatchFrame,
    q: wp.array[wp.transform],
    world: wp.array[int],
    slot: wp.array[int],
    path: wp.array[int],
    slots_needed: wp.array[int],
    route: int,
    parents: wp.array2d[int],
    mu: wp.array2d[float],
    impulses: wp.array2d[float],
    scale: float,
):
    """Transport cached patch impulses after all contact normals have been seeded.

    Anchors without patch history keep whatever the contact-matched warm start
    seeded; only carried anchors overwrite their friction rows.
    """
    c = wp.tid()
    if c >= count[0] or path[c] != route or slot[c] < 0 or slots_needed[c] != 3:
        return
    source = frame.source[c]
    if source < 0:
        return
    tangent = prev.tangent_impulse[source] * scale
    if frame.body_a[c] >= 0:
        tangent = wp.transform_vector(q[frame.body_a[c]], tangent)
    n = frame.normal[c]
    if frame.flipped[c] != 0:
        n = -n
        tangent = -tangent
    t0, t1 = contact_tangent_basis(n)
    value = wp.vec2(wp.dot(tangent, t0), wp.dot(tangent, t1))
    w = world[c]
    s = slot[c]
    radius = wp.max(mu[w, s + 1] * patch_normal_load(parents, impulses, w, s), 0.0)
    magnitude = wp.length(value)
    if magnitude > radius and magnitude > 0.0:
        value *= radius / magnitude
    impulses[w, s + 1] = value[0]
    impulses[w, s + 2] = value[1]


@wp.kernel(enable_backward=False)
def finish_patch_impulses(
    count: wp.array[int],
    frame: _PatchFrame,
    q: wp.array[wp.transform],
    q_out: wp.array[wp.transform],
    qd_out: wp.array[wp.spatial_vector],
    com: wp.array[wp.vec3],
    world: wp.array[int],
    slot: wp.array[int],
    path: wp.array[int],
    slots_needed: wp.array[int],
    route: int,
    parents: wp.array2d[int],
    mu: wp.array2d[float],
    impulses: wp.array2d[float],
    dt: float,
):
    """Cache solved friction and release saturated anchors with tangential motion.

    The tolerance is relative to geometry and timestep. Merely resting at the
    Coulomb limit does not prove sliding, so it must not erase static history.
    """
    c = wp.tid()
    if c >= count[0] or frame.valid[c] == 0:
        return
    if slot[c] < 0:
        # Capacity rejection provides no solved normal load; preserve the
        # geometrically supported history until rows are available again.
        return
    if path[c] != route:
        return
    w = world[c]
    s = slot[c]
    load = patch_normal_load(parents, impulses, w, s)
    if load <= 0.0:
        # A speculative region cannot accumulate a spring while unsupported
        # and apply that error when it later lands (common on curved meshes).
        frame.valid[c] = 0
        return
    if slots_needed[c] != 3:
        # Friction was filtered, but its normal rows still prove support.
        return
    n = frame.normal[c]
    t0, t1 = contact_tangent_basis(n)
    if frame.flipped[c] != 0:
        t0, t1 = contact_tangent_basis(-n)
    tangent = impulses[w, s + 1] * t0 + impulses[w, s + 2] * t1
    if frame.flipped[c] != 0:
        tangent = -tangent
    stored = tangent
    a = frame.body_a[c]
    b = frame.body_b[c]
    if a >= 0:
        stored = wp.transform_vector(wp.transform_inverse(q[a]), tangent)
    frame.tangent_impulse[c] = stored
    radius = wp.max(mu[w, s + 1] * load, 0.0)
    speed = _point_velocity(q_out, qd_out, com, a, frame.anchor_a[c]) - _point_velocity(
        q_out, qd_out, com, b, frame.anchor_b[c]
    )
    # Check opposing tangential motion to distinguish slipping from impending slip.
    if radius > 0.0 and wp.length(tangent) >= (1.0 - 1.0e-5) * radius:
        if wp.dot(speed, tangent) < -radius * (1.0e-5 * frame.radius[c] / dt):
            frame.valid[c] = 0


@wp.kernel
def reset_friction_anchor_history(
    world_mask: wp.array[wp.bool],
    prev_world: wp.array[int],
    prev_valid: wp.array[int],
):
    """Drop the carried friction anchors of reset worlds (all worlds for an empty mask)."""
    c = wp.tid()
    if world_mask:
        w = prev_world[c]
        if w >= 0 and w < world_mask.shape[0] and not world_mask[w]:
            return
    prev_valid[c] = 0


class ReferencePatches:
    """The 98401301 patch state, driven through the solver's launch sequence.

    Construction, geometry updates, :meth:`build` and :meth:`store` copy
    ``_FrictionPatchState`` with patches enabled. :meth:`link`, :meth:`seed`,
    :meth:`finish` and :meth:`reset` launch the per-route kernels and the history
    reset as the solver did. ``routes`` holds ``(route, parents, mu, impulses,
    scale)`` tuples; only :meth:`seed` reads ``scale``, its warm-start decay.
    """

    def __init__(self, model, capacity, phi):
        self.view = FrictionPatches()
        self.view.enabled = 1
        self.view.phi = phi
        self.capacity = capacity
        device = model.device
        self.view.weight = wp.zeros(capacity, dtype=float, device=device)
        self.view.next_contact = wp.full(capacity, -1, dtype=int, device=device)
        self.view.point = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self.body_world = model.body_world
        self.body_radius = wp.zeros(model.body_count, dtype=float, device=device)
        self.update_geometry(model)
        self.current = self._frame(capacity, device)
        self.previous = self._frame(capacity, device)
        self.previous_world = wp.full(capacity, -1, dtype=int, device=device)
        self.previous_q = wp.zeros(model.body_count, dtype=wp.transform, device=device)

    def update_geometry(self, model):
        """Refresh scales and retire affected history after explicit geometry edits."""
        geometry = {
            name: getattr(model, name).numpy().copy()
            for name in (
                "shape_body",
                "shape_transform",
                "shape_scale",
                "shape_type",
                "shape_source_ptr",
                "shape_margin",
                "shape_is_solid",
            )
        }
        radii = np.zeros(model.body_count, dtype=np.float32)
        for body, radius, transform in zip(
            geometry["shape_body"], model.shape_collision_radius.numpy(), geometry["shape_transform"], strict=True
        ):
            if body >= 0:
                radii[body] = max(radii[body], radius + np.linalg.norm(transform[:3]))
        self.body_radius.assign(radii)
        if hasattr(self, "_geometry"):
            changed = np.zeros(len(geometry["shape_body"]), dtype=bool)
            for name, values in geometry.items():
                difference = values != self._geometry[name]
                if difference.ndim > 1:
                    difference = np.any(difference, axis=tuple(range(1, difference.ndim)))
                changed |= difference
            if np.any(changed):
                bodies = np.zeros(model.body_count, dtype=np.int32)
                for shape_body in (self._geometry["shape_body"], geometry["shape_body"]):
                    affected = shape_body[changed]
                    bodies[affected[affected >= 0]] = 1
                wp.launch(
                    _invalidate_geometry_history,
                    dim=self.capacity,
                    inputs=[
                        self.previous,
                        wp.array(bodies, dtype=int, device=model.device),
                        wp.array(changed, dtype=int, device=model.device),
                    ],
                    device=model.device,
                )
        self._geometry = geometry

    @staticmethod
    def _frame(n, device):
        frame = _PatchFrame()
        frame.keys = wp.full(2 * n, 0x7FFFFFFFFFFFFFFF, dtype=wp.int64, device=device)
        frame.indices = wp.zeros(2 * n, dtype=int, device=device)
        for field in (
            "center",
            "normal",
            "displacement",
            "anchor_a",
            "anchor_b",
            "surface_a",
            "surface_b",
            "tangent_impulse",
        ):
            setattr(frame, field, wp.zeros(n, dtype=wp.vec3, device=device))
        frame.mu = wp.zeros(n, dtype=wp.vec2, device=device)
        frame.radius = wp.zeros(n, dtype=float, device=device)
        frame.support_gap_limit = wp.zeros(n, dtype=float, device=device)
        for field in (
            "body_a",
            "body_b",
            "shape_a",
            "shape_b",
            "flipped",
            "owner",
            "valid",
            "used",
            "source",
            "eligible",
            "members",
        ):
            setattr(frame, field, wp.zeros(n, dtype=int, device=device))
        return frame

    def build(
        self,
        model,
        state,
        contacts,
        *,
        body_to_articulation=None,
        is_free_rigid=None,
        contact_gap_gate=0.0,
        same_articulation_gap_gate=0.0,
        articulation_pair_gap_gate=0.0,
        friction_gap=float("inf"),
    ):
        wp.launch(
            _prepare,
            dim=self.capacity,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                model.shape_body,
                model.shape_material_mu,
                model.shape_gap,
                self.body_radius,
                state.body_q,
                body_to_articulation,
                is_free_rigid,
                contact_gap_gate,
                same_articulation_gap_gate,
                articulation_pair_gap_gate,
                friction_gap,
                self.current,
                self.view,
            ],
            device=model.device,
        )
        wp.utils.radix_sort_pairs(self.current.keys, self.current.indices, self.capacity)
        wp.launch(
            _build,
            dim=self.capacity,
            inputs=[
                contacts.rigid_contact_count,
                state.body_q,
                self.previous_q,
                model.shape_transform,
                model.shape_collision_radius,
                self.current,
                self.previous,
                self.view,
            ],
            device=model.device,
        )

    def store(self, state):
        wp.launch(
            _store_history,
            dim=max(self.capacity, self.previous_q.shape[0]),
            inputs=[
                state.body_q,
                self.body_world,
                self.current,
                self.previous,
                self.previous_world,
                self.previous_q,
            ],
            device=state.body_q.device,
        )

    def link(self, contacts, world, slot, path, slots_needed, routes):
        """Link each route's normal rows, one launch per route."""
        for route, parents, mu, _impulses, _scale in routes:
            wp.launch(
                link_patch_rows,
                dim=contacts.rigid_contact_max,
                inputs=[contacts.rigid_contact_count, self.view, world, slot, path, slots_needed, route, parents, mu],
                device=world.device,
            )

    def seed(self, contacts, state, world, slot, path, slots_needed, routes):
        """Seed carried tangent impulses on the routes with a warm-start decay."""
        for route, parents, mu, impulses, scale in routes:
            if scale > 0.0:
                wp.launch(
                    seed_patch_impulses,
                    dim=contacts.rigid_contact_max,
                    inputs=[
                        contacts.rigid_contact_count,
                        self.current,
                        self.previous,
                        state.body_q,
                        world,
                        slot,
                        path,
                        slots_needed,
                        route,
                        parents,
                        mu,
                        impulses,
                        scale,
                    ],
                    device=world.device,
                )

    def finish(self, contacts, state_in, state_out, com, world, slot, path, slots_needed, routes, dt):
        """Release or cache solved anchors per route, then store the frame; ``None`` contacts drop history."""
        if contacts is None:
            self.previous.valid.zero_()
            return
        for route, parents, mu, impulses, _scale in routes:
            wp.launch(
                finish_patch_impulses,
                dim=contacts.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    self.current,
                    state_in.body_q,
                    state_out.body_q,
                    state_out.body_qd,
                    com,
                    world,
                    slot,
                    path,
                    slots_needed,
                    route,
                    parents,
                    mu,
                    impulses,
                    dt,
                ],
                device=world.device,
            )
        self.store(state_in)

    def reset(self, world_mask):
        """Drop the stored anchors of the masked worlds over the whole history."""
        wp.launch(
            reset_friction_anchor_history,
            dim=self.previous.valid.shape[0],
            inputs=[world_mask, self.previous_world, self.previous.valid],
            device=self.previous.valid.device,
        )
