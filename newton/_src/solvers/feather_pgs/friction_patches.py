# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver-owned friction regions, independent of collision contact identities.

Normal contacts are never reduced here. Each compatible region selects up to two
friction locations, with an equal share of the region's total normal impulse.
Locations average the support edges along the footprint's principal extent,
avoiding a diagonal friction couple on symmetric narrow footprints; each edge's
last member in contact order carries its friction rows. Isotropic footprints
retain a canonical farthest pair because they have no unique axis; member spans
and contact order choose it, so float32 roundoff in world coordinates cannot
swap it between steps. Both choices are canonical relative to contact order: a
collision pipeline that reorders a pair's contacts between steps moves an edge's
rows to another member, and can swap an isotropic pair, whose locations are its
members' contact centres.
Regions require matching friction materials, nearly aligned normals, nearby
contact planes, and connected shape bounding spheres on each body. The latter
is a conservative test across convex seams, not an exact surface-connectivity
query. Length tolerances scale with the smaller body's bounding radius; angular
tolerance is about six degrees. Normals remain individual unilateral rows.

Tangential displacement and impulses persist across frames; friction locations
follow the current contact region. Displacement accumulates incremental rigid
motion at those locations, using the pose increment's SE(3) logarithm rather
than a rotating material point's chord. Saved poses also capture externally
imposed motion and position-only solver passes. Pair identity and geometric
compatibility correlate history without a collision matcher or shape-type rules.
Unloaded regions and saturated friction with opposing slip release history;
saturation alone does not. All matching tolerances are internal so correction
strength is the only new user control.

Sorting body pairs makes construction local to a pair and preserves CUDA graph
capture: all buffers and the radix-sort workspace have fixed capacity. Launches
stride over the live contacts and the slots earlier frames used, so their work
follows the live contact count rather than the capacity. With frame reuse, the
solver's default, a captured build skips the pair sort when the previous build
was captured in the same graph over the same contacts and the host issued no
contact write in between, which is the case on every solver substep that shares
a collision frame (see :class:`_FrictionPatchState`). The solver's experimental
opt-in ``FEATHER_PGS_PATCH_CONDITIONAL_NODES=1`` sorts the other captured builds
under a conditional node that also skips empty frames and the frame whose pairs
the slots already hold sorted.
"""

import itertools
import weakref
from contextlib import nullcontext

import numpy as np
import warp as wp

from ...math.spatial import velocity_at_point
from ...sim.contacts import GENERATION_SENTINEL, capture_scope, contact_frame_write
from .capture_joins import capture_has_unjoined_work
from .contact_filters import contact_friction_eligible, contact_normal_gap_limit

# Grids are fixed at construction so CUDA graphs replay them. Their sizes count
# _BLOCK_THREADS threads per SM (one SM on the CPU); launches stride over the
# live slots, so no grid changes any output.
_BLOCK_THREADS = 256
# The light launches (prepare, link, seed, finish and store, reset) handle one
# slot per loop iteration at 40 to 48 registers, which fit five or six blocks
# per SM, and their reads through slots and rings wait on memory; blocks past
# the live slots only read the counts. Measured on a 188-SM GPU against two
# blocks per SM: the finish 0.88x, the link 0.80x and a reused frame's prepare
# 0.77x per call with 512k live contacts, the link 0.67x with 346k, a new
# frame's prepare 0.98x to 1.05x, and no change with 1.6k.
_LIGHT_BLOCKS_PER_SM = 6
# Each of the live sort's chunk blocks sorts 2048 slots in shared memory.
_SORT_BLOCKS_PER_SM = 2
# A _build thread walks one body pair's run, a chain of dependent loads at 96
# registers. One-warp blocks fit 21 warps per SM instead of two 256-thread
# blocks' 16, free each warp's slot when it finishes rather than with its
# slowest block-mate, and spread a short frame's runs over every SM; the wider
# grid leaves no thread a second run below 8 * SMs * 256 live slots. Measured
# on a 188-SM GPU: 0.82x the time per build with 512k live contacts, 0.85x
# with 346k, 0.81x with 1.6k of a 4M capacity.
_BUILD_BLOCK_DIM = 32
_BUILD_BLOCKS_PER_SM = 8
# _merge_runs keeps the former build grid: a live sort launches it for every
# merge round its capacity allows, mostly over few live slots.
_MERGE_BLOCKS_PER_SM = 4
# From this contact capacity on, the pair sort orders only the live slots
# instead of the whole capacity. Provisional: calibrate against the full radix
# sort on the GPU at large capacities and at high occupancy.
_LIVE_SORT_CAPACITY = 2_000_000
# Up to this contact capacity the pair sort orders only the live slots too: its
# launches do no work without live contacts, where a radix sort still passes over
# the whole capacity (52-55 us at 16,384 slots on the RTX PRO 6000). Provisional:
# tune against the radix sort in a captured graph at partial occupancy.
_CHUNK_SORT_CAPACITY = 16_384
# The live sort orders chunks of _SORT_CHUNK slots in shared memory by
# (key, slot) and then merges sorted runs pairwise.
_SORT_CHUNK = wp.constant(2048)
_SORT_SLOT_BITS = wp.constant(11)
_SORT_PAD = wp.constant(wp.int64(1 << 62))
# Contact centres carry the narrow phase's float32 world-space quantization. On
# a resting 0.1 m cube it reads as a relative anisotropy of the footprint's
# second moments of up to 1.2e-6 at 1.5 m from the origin, 9.5e-6 at 10 m and
# 5.4e-5 at 80 m. Second moments and squared member spans that differ by less
# than this fraction count as equal, and members within this fraction of the
# footprint's extent of a support edge belong to it, so roundoff neither gives
# an isotropic footprint an axis, chooses between equally distant members, nor
# splits an edge. The anisotropy grows as the footprint shrinks: one-ulp moves
# of every corner of a square of half-width h reach 2 ulp(|x|) / h, so the
# fraction covers squares wider than about 3 cm at 80 m from the origin and
# 6 cm at 190 m, and narrower ones nearer to it.
_FOOTPRINT_TOLERANCE = wp.constant(1.0e-3)
# Contact paths 0, 1 and 2 hold dense, matrix-free and propagation rows, and -1 holds none.
# CONTACT_PATH_RESOLVED marks a contact without rows in a world the resolved-world classifier
# skipped, where the complete path would have reserved them: its patch anchor releases.
CONTACT_PATH_RESOLVED = wp.constant(-2)
# The contact generation no frame has: a recorded frame with it never matches.
_FRAME_SENTINEL = wp.constant(wp.int64(GENERATION_SENTINEL))
# Modes of _prepare_frame, the host's admission of a frame-reusing build: the
# pair sort follows unconditionally, a conditional node sorts if the frame is
# new, or the slots keep the sorted pairs of the frame an earlier build recorded.
_FRAME_SORT = wp.constant(0)
_FRAME_DETECT = wp.constant(1)
_FRAME_KEEP = wp.constant(2)


@wp.struct
class FrictionPatches:
    """Per-contact view consumed by the existing contact row builders."""

    enabled: int
    weight: wp.array[float]
    next_contact: wp.array[int]
    point_a: wp.array[wp.vec3]
    point_b: wp.array[wp.vec3]
    phi: wp.array[wp.vec2]


@wp.struct
class PatchRowLaw:
    """Tangent-row law of persistent patches, for producers that write their own friction rows.

    A tangent row acts at its patch's friction location, carries the anchor's tangential
    displacement as phi and corrects it with ``beta`` (``friction_anchor_beta``). Producers write
    the undivided friction coefficient; the patch link (``_FrictionPatchState.link``, which
    launches ``link_patch_routes``) divides it and links the ring. The compact contact boundary
    and the local row packets apply the law through :func:`patch_row_frame` and
    :func:`patch_tangent_row`; keyed rows write it in their native row source
    (``keyed_rows._patch_row_law``), and the masked row builder's patch variants
    (``row_build.contact_row_prelude_patch``) as the per-contact builder does.
    """

    patches: FrictionPatches
    beta: float


@wp.func
def patch_friction_location(patches: FrictionPatches, c: int):
    """Return contact ``c``'s friction location on each side, in world coordinates."""
    return patches.point_a[c], patches.point_b[c]


@wp.func
def patch_row_frame(
    row: int,
    shared_anchor: int,
    normal: wp.vec3,
    tangent0: wp.vec3,
    tangent1: wp.vec3,
    point0: wp.vec3,
    point1: wp.vec3,
    friction0: wp.vec3,
    friction1: wp.vec3,
):
    """Return contact row ``row``'s direction and endpoints under the patch row law.

    The normal row (row 0) acts at the contact points ``point0`` and ``point1``, or at their
    midpoint with ``shared_anchor``. Tangent rows 1 and 2 act along ``tangent0`` and ``tangent1``
    at the friction location ``friction0`` and ``friction1`` (:func:`patch_friction_location`),
    which takes the place of both shared anchors.
    """
    direction = normal
    p0 = point0
    p1 = point1
    if row == 0:
        if shared_anchor != 0:
            p0 = 0.5 * (point0 + point1)
            p1 = p0
    else:
        direction = tangent0
        if row == 2:
            direction = tangent1
        p0 = friction0
        p1 = friction1
    return direction, p0, p1


@wp.func
def patch_tangent_row(law: PatchRowLaw, c: int, row: int, mu: float, friction_scale: float):
    """Return contact ``c``'s tangent row ``row`` (1 or 2): its phi, beta and friction coefficient.

    The coefficient is the undivided ``mu * friction_scale``; the patch link divides it among the
    ring's anchors.
    """
    return law.patches.phi[c][row - 1], law.beta, mu * friction_scale


@wp.struct
class _PatchFrame:
    keys: wp.array[wp.int64]
    indices: wp.array[int]
    center: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    displacement: wp.array[wp.vec3]
    mu: wp.array[wp.vec2]
    radius: wp.array[float]
    body_a: wp.array[int]
    body_b: wp.array[int]
    shape_a: wp.array[int]
    shape_b: wp.array[int]
    flipped: wp.array[int]
    owner: wp.array[int]
    anchor_a: wp.array[wp.vec3]
    anchor_b: wp.array[wp.vec3]
    surface_a: wp.array[wp.vec3]
    surface_b: wp.array[wp.vec3]
    valid: wp.array[int]
    used: wp.array[int]
    source: wp.array[int]
    eligible: wp.array[int]
    # Scratch indices: patch members in the current frame, valid anchors in the previous frame.
    members: wp.array[int]
    support_gap_limit: wp.array[float]
    tangent_impulse: wp.array[wp.vec3]


@wp.func
def _world_point(q: wp.array[wp.transform], body: int, point: wp.vec3):
    result = point
    if body >= 0:
        result = wp.transform_point(q[body], point)
    return result


@wp.func
def _local_point(q: wp.array[wp.transform], body: int, point: wp.vec3):
    result = point
    if body >= 0:
        result = wp.transform_point(wp.transform_inverse(q[body]), point)
    return result


@wp.func
def _prepare_contact(
    c: int,
    live: int,
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
    friction_articulation_pairs_only: int,
    key_stride: int,
    frame: _PatchFrame,
    patches: FrictionPatches,
):
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
    if c >= live:
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
    eligible = gap <= normal_gap_limit and contact_friction_eligible(
        gap, a_non_free, b_non_free, friction_articulation_pairs_only, friction_gap
    )
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
    # ka and kb are below key_stride, so the key orders pairs as (ka, kb) does
    # and occupies only the low bits that build() sorts.
    frame.keys[c] = wp.int64(ka) * wp.int64(key_stride) + wp.int64(kb)
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


@wp.func
def _prepare_contact_fields(
    c: int,
    live: int,
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
    friction_articulation_pairs_only: int,
    key_stride: int,
    frame: _PatchFrame,
    patches: FrictionPatches,
):
    """Prepare contact ``c`` as :func:`_prepare_contact` does, except slot ``c``'s pair key and index.

    Returns the pair key :func:`_prepare_contact` stores in slot ``c``, the
    empty key past ``live``, so a build that reuses sorted pairs can leave the
    slots' keys and indices untouched. Keep the arithmetic in step with
    :func:`_prepare_contact`; the frame-reuse tests compare the two bitwise.
    """
    frame.owner[c] = -1
    frame.valid[c] = 0
    frame.used[c] = 0
    frame.source[c] = -1
    frame.eligible[c] = 0
    frame.tangent_impulse[c] = wp.vec3(0.0)
    patches.weight[c] = 0.0
    patches.next_contact[c] = -1
    patches.phi[c] = wp.vec2(0.0)
    if c >= live:
        return wp.int64(0x7FFFFFFFFFFFFFFF)
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
    eligible = gap <= normal_gap_limit and contact_friction_eligible(
        gap, a_non_free, b_non_free, friction_articulation_pairs_only, friction_gap
    )
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
    return wp.int64(ka) * wp.int64(key_stride) + wp.int64(kb)


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
    friction_articulation_pairs_only: int,
    key_stride: int,
    stored: wp.array[int],
    built: wp.array[int],
    threads: int,
    frame: _PatchFrame,
    patches: FrictionPatches,
    extent: wp.array[int],
):
    """Fill this frame's contact slots and clear every slot an earlier frame used.

    The narrow phase counts contacts before checking its capacity, so an
    overflowing count describes an incomplete frame and leaves no contact live.
    Slots past the stored and the last built frame already hold the empty
    record. Thread 0 publishes the slots to visit and the live count; this
    launch reads neither.
    """
    thread = wp.tid()
    live = count[0]
    if live < 0 or live > shape0.shape[0]:
        live = 0
    live = wp.min(live, frame.valid.shape[0])
    end = wp.max(live, wp.max(stored[0], built[0]))
    if thread == 0:
        extent[0] = end
        extent[1] = live
    for c in range(thread, end, threads):
        _prepare_contact(
            c,
            live,
            shape0,
            shape1,
            point0,
            point1,
            normal,
            margin0,
            margin1,
            shape_body,
            shape_mu,
            shape_gap,
            body_radius,
            q,
            body_to_articulation,
            is_free_rigid,
            contact_gap_gate,
            same_articulation_gap_gate,
            articulation_pair_gap_gate,
            friction_gap,
            friction_articulation_pairs_only,
            key_stride,
            frame,
            patches,
        )


# The frame-reuse variants compile in modules of their own, so the base patch
# kernels keep their module and generated code. Pairs are built by the base
# _build kernel itself: a second kernel that inlines _build_pair rounds the
# carried displacements differently on CUDA.
@wp.kernel(enable_backward=False, module="unique")
def _prepare_frame(
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
    friction_articulation_pairs_only: int,
    key_stride: int,
    stored: wp.array[int],
    built: wp.array[int],
    threads: int,
    frame: _PatchFrame,
    patches: FrictionPatches,
    generation: wp.array[wp.int32],
    stream: wp.int64,
    mode: int,
    sorted_frame: wp.array[wp.int64],
    extent: wp.array[int],
    sort_pairs: wp.array[wp.int32],
    frame_status: wp.array[wp.int32],
):
    """Prepare as :func:`_prepare` does, in the host's admission ``mode`` of a frame-reusing build.

    ``sorted_frame`` records the contact frame whose pairs the slots hold
    sorted: its generation, raw count and contact stream. ``_FRAME_SORT``
    prepares a frame the pair sort follows unconditionally, so thread 0
    records it. ``_FRAME_KEEP`` reuses the pairs an earlier build sorted: live
    slots keep their keys and indices, which other modes overwrite. Its frame
    must be the recorded one. A frame rewritten since then, without a write
    the host noted, is left empty instead: no contact is live, the record is
    dropped and ``frame_status`` latches, because stale pairs would group
    other contacts. ``_FRAME_DETECT`` keeps the recorded frame's pairs the same
    way and publishes in ``sort_pairs`` whether another frame's live contacts
    need sorting; it drops the record of such a frame and records a new frame
    without live contacts, which needs no sort. Threads agree on the frame
    whether they read the record before or after thread 0 writes it.
    """
    thread = wp.tid()
    raw = count[0]
    live = raw
    if live < 0 or live > shape0.shape[0]:
        live = 0
    live = wp.min(live, frame.valid.shape[0])
    end = wp.max(live, wp.max(stored[0], built[0]))
    recorded = int(0)
    if (
        mode != _FRAME_SORT
        and sorted_frame[0] == wp.int64(generation[0])
        and sorted_frame[1] == wp.int64(raw)
        and sorted_frame[2] == stream
    ):
        recorded = 1
    if mode == _FRAME_KEEP and recorded == 0:
        live = 0
    if thread == 0:
        extent[0] = end
        extent[1] = live
        # The slots hold a frame without live contacts sorted as soon as it is
        # prepared; a thread that reads its record keeps no slot.
        if mode == _FRAME_SORT or (mode == _FRAME_DETECT and recorded == 0 and live == 0):
            sorted_frame[0] = wp.int64(generation[0])
            sorted_frame[1] = wp.int64(raw)
            sorted_frame[2] = stream
        elif recorded == 0:
            sorted_frame[0] = _FRAME_SENTINEL
            if mode == _FRAME_KEEP:
                frame_status[0] = 1
        if mode == _FRAME_DETECT:
            sort_pairs[0] = 0
            if recorded == 0 and live > 0:
                sort_pairs[0] = 1
    for c in range(thread, end, threads):
        key = _prepare_contact_fields(
            c,
            live,
            shape0,
            shape1,
            point0,
            point1,
            normal,
            margin0,
            margin1,
            shape_body,
            shape_mu,
            shape_gap,
            body_radius,
            q,
            body_to_articulation,
            is_free_rigid,
            contact_gap_gate,
            same_articulation_gap_gate,
            articulation_pair_gap_gate,
            friction_gap,
            friction_articulation_pairs_only,
            key_stride,
            frame,
            patches,
        )
        # A kept slot holds the sorted pair at position c, not contact c's key.
        if recorded == 0 or c >= live:
            frame.indices[c] = c
            frame.keys[c] = key


@wp.func
def _chunk_key(key: wp.int64, local: int, start: int, live: int):
    """Order a chunk's slots by (pair key, slot); slots past the live count sort last, in slot order."""
    if start + local < live:
        return (key << wp.int64(_SORT_SLOT_BITS)) | wp.int64(local)
    return _SORT_PAD + wp.int64(local)


@wp.func
def _chunk_pair_key(order: wp.int64):
    if order >= _SORT_PAD:
        return wp.int64(0x7FFFFFFFFFFFFFFF)
    return order >> wp.int64(_SORT_SLOT_BITS)


@wp.func
def _merge_rounds(live: int):
    """Return the pairwise merges that join sorted chunks into one run of ``live`` slots."""
    rounds = int(0)
    width = int(_SORT_CHUNK)
    while width < live and rounds < 30:
        width *= 2
        rounds += 1
    return rounds


@wp.kernel(enable_backward=False)
def _sort_chunks(
    slot_keys: wp.array[wp.int64],
    slot_indices: wp.array[int],
    keys: wp.array[wp.int64],
    indices: wp.array[int],
    extent: wp.array[int],
    capacity: int,
    blocks: int,
):
    """Sort each chunk of the live slots by (key, slot) in shared memory.

    The composite keys are unique, so even an unstable sort orders a chunk as a
    stable sort by key would. Each block strides over chunks; the output goes to
    the half from which the merges end in the first. Chunks load through
    ``slot_keys`` and ``slot_indices``, views of the first half that read zeros
    past the capacity, so no block reads what another block of this launch
    writes.
    """
    block = wp.tid()
    live = extent[1]
    output = capacity * (_merge_rounds(live) % 2)
    for start in range(block * _SORT_CHUNK, live, blocks * _SORT_CHUNK):
        loaded = wp.tile_load(slot_keys, shape=_SORT_CHUNK, offset=start)
        local = wp.tile_arange(_SORT_CHUNK, dtype=int)
        first = wp.tile_full(_SORT_CHUNK, start, dtype=int)
        count = wp.tile_full(_SORT_CHUNK, live, dtype=int)
        order = wp.tile_zeros(shape=_SORT_CHUNK, dtype=wp.int64, storage="shared")
        # Warp hashes a function passed to a builtin only through wp.static();
        # otherwise edits to the key maps would reuse a stale cached kernel.
        wp.tile_assign(order, wp.tile_map(wp.static(_chunk_key), loaded, local, first, count))
        values = wp.tile_load(slot_indices, shape=_SORT_CHUNK, offset=start, storage="shared")
        wp.tile_sort(order, values)
        wp.tile_store(keys, wp.tile_map(wp.static(_chunk_pair_key), order), offset=output + start)
        wp.tile_store(indices, values, offset=output + start)


@wp.kernel(enable_backward=False)
def _merge_runs(
    keys: wp.array[wp.int64],
    indices: wp.array[int],
    extent: wp.array[int],
    capacity: int,
    merge: int,
    threads: int,
):
    """Merge sorted runs pairwise; each slot moves past the smaller slots of its partner run.

    Slots compare by (key, index). Indices are unique, so no two slots tie and
    every target position is distinct. Launches past the needed merges return.
    """
    thread = wp.tid()
    live = extent[1]
    rounds = _merge_rounds(live)
    if merge >= rounds:
        return
    source = capacity * ((rounds + merge) % 2)
    target = capacity * ((rounds + merge + 1) % 2)
    width = int(_SORT_CHUNK) << merge
    for i in range(thread, live, threads):
        key = keys[source + i]
        index = indices[source + i]
        run = i // width
        partner = run ^ 1
        lo = partner * width
        hi = wp.min(lo + width, live)
        first = lo
        while lo < hi:
            mid = (lo + hi) // 2
            other = keys[source + mid]
            if other < key or (other == key and indices[source + mid] < index):
                lo = mid + 1
            else:
                hi = mid
        position = wp.min(run, partner) * width + (i - run * width) + (lo - first)
        keys[target + position] = key
        indices[target + position] = index


@wp.func
def _compatible(frame: _PatchFrame, seed: int, c: int):
    d = frame.center[c] - frame.center[seed]
    r = frame.radius[seed]
    return (
        wp.dot(frame.normal[seed], frame.normal[c]) >= 0.995
        and frame.mu[seed][0] == frame.mu[c][0]
        and frame.mu[seed][1] == frame.mu[c][1]
        and wp.abs(wp.dot(d, frame.normal[seed])) <= 0.02 * r
        and wp.length_sq(d) <= 4.0 * r * r
    )


@wp.func
def _shapes_adjacent(a: int, b: int, transforms: wp.array[wp.transform], radii: wp.array[float]):
    if a == b:
        return True
    if a < 0 or b < 0:
        return False
    radius = radii[a] + radii[b]
    delta = wp.transform_get_translation(transforms[a]) - wp.transform_get_translation(transforms[b])
    return wp.length_sq(delta) <= (1.0 + 1.0e-5) * radius * radius


@wp.func
def _geometry_adjacent(sa: int, sb: int, ta: int, tb: int, transforms: wp.array[wp.transform], radii: wp.array[float]):
    """Conservative adjacency in each body's local frame, including convex seams."""
    return _shapes_adjacent(sa, ta, transforms, radii) and _shapes_adjacent(sb, tb, transforms, radii)


@wp.func
def _pose_motion(q: wp.array[wp.transform], previous_q: wp.array[wp.transform], body: int, point: wp.vec3):
    """Evaluate the rigid pose increment's SE(3) logarithm at a world point.

    The midpoint form avoids subtracting large world-origin moments. It is
    exact for a constant screw motion: a fixed pivot has zero displacement,
    and rolling does not mistake a material point's chord for tangential slip.
    """
    motion = wp.vec3(0.0)
    if body >= 0:
        current = q[body]
        previous = previous_q[body]
        x0 = wp.transform_get_translation(previous)
        x1 = wp.transform_get_translation(current)
        rotation = wp.transform_get_rotation(current) * wp.quat_inverse(wp.transform_get_rotation(previous))
        if rotation[3] < 0.0:
            rotation = -rotation
        vector = wp.vec3(rotation[0], rotation[1], rotation[2])
        sine = wp.length(vector)
        angular = 2.0 * vector
        translation = x1 - x0
        linear = translation
        if sine > 1.0e-6:
            half_angle = wp.atan2(sine, rotation[3])
            axis = vector / sine
            angular = (2.0 * half_angle) * axis
            scale = half_angle * rotation[3] / sine
            linear = scale * translation + (1.0 - scale) * wp.dot(axis, translation) * axis
        motion = linear + wp.cross(angular, point - 0.5 * (x0 + x1))
    return motion


@wp.func
def _carried_displacement(q: wp.array[wp.transform], prev: _PatchFrame, p: int, normal: wp.vec3):
    """Rotate stored spring displacement into the current tangent plane."""
    a = prev.body_a[p]
    error = prev.displacement[p]
    old_normal = prev.normal[p]
    if a >= 0:
        error = wp.transform_vector(q[a], error)
        old_normal = wp.transform_vector(q[a], old_normal)
    # Parallel transport preserves spring length when the tangent plane turns;
    # projection alone would dissipate history on every small rocking motion.
    cosine = wp.dot(old_normal, normal)
    if cosine > 0.0:
        axis = wp.cross(old_normal, normal)
        cross_error = wp.cross(axis, error)
        error += cross_error + wp.cross(axis, cross_error) / (1.0 + cosine)
    return error


@wp.func
def _build_pair(
    start: int,
    end: int,
    previous_end: int,
    q: wp.array[wp.transform],
    previous_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_radius: wp.array[float],
    frame: _PatchFrame,
    prev: _PatchFrame,
    patches: FrictionPatches,
):
    """Build the body pair whose sorted run starts at ``start``; other positions return."""
    if start > 0 and frame.keys[start - 1] == frame.keys[start]:
        return
    key = frame.keys[start]
    if key == wp.int64(0x7FFFFFFFFFFFFFFF):
        return
    stop = start + 1
    while stop < end and frame.keys[stop] == key:
        stop += 1

    # Binary search the previous pair, avoiding a scan over other worlds/pairs.
    # Positions from previous_end on hold no stored record.
    lo = int(0)
    hi = previous_end
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
    while prev_stop < previous_end and prev.keys[prev_stop] == key:
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
                temp = pa
                pa = pb
                pb = temp
                n = -n
            patches.point_a[c] = pa
            patches.point_b[c] = pb
            t0, t1 = contact_tangent_basis(n)
            if frame.flipped[c] != 0:
                error = -error
            patches.phi[c] = wp.vec2(wp.dot(t0, error), wp.dot(t1, error))


@wp.kernel(enable_backward=False, module="unique")
def _record_sorted_frame(
    generation: wp.array[wp.int32],
    count: wp.array[int],
    stream: wp.int64,
    sorted_frame: wp.array[wp.int64],
):
    """Record the contact frame whose pairs the slots now hold sorted.

    One thread runs it right after the sort, and _prepare_frame read the record
    in an earlier launch.
    """
    sorted_frame[0] = wp.int64(generation[0])
    sorted_frame[1] = wp.int64(count[0])
    sorted_frame[2] = stream


@wp.kernel(enable_backward=False)
def _build(
    extent: wp.array[int],
    stored: wp.array[int],
    q: wp.array[wp.transform],
    previous_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_radius: wp.array[float],
    threads: int,
    frame: _PatchFrame,
    prev: _PatchFrame,
    patches: FrictionPatches,
    built: wp.array[int],
):
    """Stride over the live sorted positions; body pairs write disjoint current/previous records.

    Thread 0 records the live count, so the next _prepare clears these slots even
    when no store follows this build.
    """
    thread = wp.tid()
    live = extent[1]
    if thread == 0:
        built[0] = live
    previous_end = stored[0]
    for start in range(thread, live, threads):
        _build_pair(start, live, previous_end, q, previous_q, shape_transform, shape_radius, frame, prev, patches)


@wp.func
def _store_contact(
    c: int,
    q: wp.array[wp.transform],
    body_world: wp.array[int],
    frame: _PatchFrame,
    prev: _PatchFrame,
    previous_world: wp.array[int],
    previous_q: wp.array[wp.transform],
):
    """Carry contact history and input poses, independent of solver velocity passes."""
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
    # _build reads pose increments only through valid stored anchors, so only
    # their bodies need the input pose.
    if frame.body_a[c] >= 0:
        previous_q[frame.body_a[c]] = q[frame.body_a[c]]
    if frame.body_b[c] >= 0:
        previous_q[frame.body_b[c]] = q[frame.body_b[c]]


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


def _pair_key_layout(bodies, shapes):
    """Return the pair key stride and the number of bits a pair key occupies.

    _prepare codes bodies as 1..bodies and static or missing shapes above them,
    so every code is below the stride. The stride reaches _prepare as a 32-bit
    int, which bounds it and keeps pair keys below 2**62.
    """
    stride = bodies + shapes + 2
    if stride > 2**31 - 1:
        raise ValueError(
            f"Patch friction supports at most 2**31 - 3 bodies and shapes, got {bodies} bodies and {shapes} shapes"
        )
    return stride, (stride * stride - 1).bit_length()


def _uses_live_sort(capacity, key_bits):
    """Whether the pair sort orders only the live slots instead of the whole capacity.

    It does up to _CHUNK_SORT_CAPACITY and from _LIVE_SORT_CAPACITY on, unless a
    chunk key, the pair key with an 11-bit slot appended, would reach the pad bit.
    Both sorts leave the unique (key, slot) order.
    """
    sized = capacity <= _CHUNK_SORT_CAPACITY or capacity >= _LIVE_SORT_CAPACITY
    return sized and key_bits + _SORT_SLOT_BITS <= 62


class _FrictionPatchState:
    """Own preallocated patch frames; never read device counters on the host.

    With ``reuse_frames``, the host admits each captured build statically. A
    build keeps the sorted pairs the previous build left in the slots when that
    build was captured in the same scope (one graph, or one conditional body;
    :func:`~newton._src.sim.contacts.capture_scope`) over the same
    ``contact_generation`` array, and no write of that contact stream was noted
    since (:func:`~newton._src.sim.contacts.note_contact_frame_write`) nor
    :meth:`forget_frame` called. Every other build sorts. The device checks each
    kept build against the frame the last sort recorded: a frame rewritten by a
    writer the host never saw leaves that build empty and latches
    ``frame_status``, so stale pairs never group contacts.

    With ``conditional_nodes``, captured builds that do not keep their pairs sort
    under a conditional node that skips empty frames and the recorded frame;
    eager builds then record the frames they sort for such a node. Contact frames
    are identified by ``Contacts.contact_generation``, which every collision
    pass and :meth:`newton.Contacts.clear` advance, together with the raw contact
    count and the contact stream, a token of the generation counter array that
    is never reused (see :meth:`stream_token`). Eager builds always sort.
    """

    def __init__(self, model, capacity, enabled, phi, *, reuse_frames=False, conditional_nodes=False):
        self.view = FrictionPatches()
        self.view.enabled = int(enabled)
        self.view.phi = phi
        self.capacity = capacity
        device = model.device
        n = capacity if enabled else 0
        self.view.weight = wp.zeros(n, dtype=float, device=device)
        self.view.next_contact = wp.full(n, -1, dtype=int, device=device)
        self.view.point_a = wp.zeros(n, dtype=wp.vec3, device=device)
        self.view.point_b = wp.zeros(n, dtype=wp.vec3, device=device)
        self.reuse_frames = bool(reuse_frames and enabled)
        self.conditional_nodes = bool(conditional_nodes and enabled)
        # (capture scope, weak generation array, frame-write tick) of the previous
        # captured build, which a build in the same scope may keep the pairs of.
        self._last_build = None
        if not enabled:
            return
        self.device = device
        # The pair sort skips the key bits no pair key can set.
        self.key_stride, self.key_bits = _pair_key_layout(model.body_count, model.shape_body.shape[0])
        self.body_world = model.body_world
        self.body_radius = wp.zeros(model.body_count, dtype=float, device=device)
        # The contact frame whose pairs the current slots hold sorted, as
        # (generation, raw count, contact stream); the sentinel generation makes
        # the next build sort. _prepare_frame records a frame the sort follows,
        # compares a reused one and publishes whether a node sorts, and
        # _record_sorted_frame records the frame a node sorted.
        self.sorted_frame = wp.full(3, GENERATION_SENTINEL, dtype=wp.int64, device=device)
        self.sort_pairs = wp.zeros(1, dtype=wp.int32, device=device)
        # Latched when a build that reuses sorted pairs finds another contact
        # frame than the recorded one and is left empty; only the host clears it.
        self.frame_status = wp.zeros(1, dtype=wp.int32, device=device)
        self._stream_tokens = weakref.WeakKeyDictionary()
        self._next_stream_token = itertools.count(1)
        self.update_geometry(model)
        self.current = self._frame(capacity, device)
        self.previous = self._frame(capacity, device)
        self.previous_world = wp.full(capacity, -1, dtype=int, device=device)
        self.previous_q = wp.zeros(model.body_count, dtype=wp.transform, device=device)
        # Live contact counts on the device. The finish epilogue alone writes the
        # stored frame's count and _build the last built frame's; _prepare
        # publishes the slots this frame visits and its live count. No launch
        # reads a count it writes, so blocks never race on them.
        self.stored = wp.zeros(1, dtype=int, device=device)
        self.built = wp.zeros(1, dtype=int, device=device)
        self.extent = wp.zeros(2, dtype=int, device=device)
        # Fixed grids keep CUDA graph replay valid; threads stride over live slots.
        blocks = device.sm_count if device.is_cuda else 1
        self.threads = max(1, min(capacity, _LIGHT_BLOCKS_PER_SM * blocks * _BLOCK_THREADS))
        self.build_threads = max(1, min(capacity, _BUILD_BLOCKS_PER_SM * blocks * _BLOCK_THREADS))
        self.build_block_dim = _BUILD_BLOCK_DIM
        self.merge_threads = max(1, min(capacity, _MERGE_BLOCKS_PER_SM * blocks * _BLOCK_THREADS))
        self.live_sort = _uses_live_sort(capacity, self.key_bits)
        self.sort_blocks = max(1, min(-(-capacity // _SORT_CHUNK), _SORT_BLOCKS_PER_SM * blocks))
        self.merges = 0
        while _SORT_CHUNK << self.merges < capacity:
            self.merges += 1

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
                # An anchor may have crossed a convex seam since it was created.
                # Invalidate the affected body's history, including other carrier
                # shapes, while keeping unrelated bodies and static shapes intact.
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
        # Pair keys follow the shapes' bodies.
        self.forget_frame()

    def forget_frame(self):
        """Make the next build sort its pairs, as it does for a new contact frame."""
        self._last_build = None
        if self.reuse_frames or self.conditional_nodes:
            self.sorted_frame.fill_(GENERATION_SENTINEL)

    def stream_token(self, generation):
        """Return the token that names the contact stream whose generation counter is ``generation``.

        A token is never reused, unlike the address of a freed counter, which the
        allocator can hand to the next ``Contacts``: a stream allocated in its place
        is a new stream. Graphs bake the token of the stream they were captured with.
        """
        token = self._stream_tokens.get(generation)
        if token is None:
            token = self._stream_tokens[generation] = next(self._next_stream_token)
        return token

    @staticmethod
    def _frame(n, device):
        frame = _PatchFrame()
        # Every slot starts as the empty record _prepare writes: the sentinel key,
        # its own index, and no owner or source. Bounded launches never revisit
        # slots that still hold it.
        frame.keys = wp.full(2 * n, 0x7FFFFFFFFFFFFFFF, dtype=wp.int64, device=device)
        frame.indices = wp.zeros(2 * n, dtype=int, device=device)
        wp.launch(_arange, dim=n, inputs=[frame.indices], device=device)
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
        frame.owner.fill_(-1)
        frame.source.fill_(-1)
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
        friction_articulation_pairs_only=False,
        capture_pause=nullcontext,
    ):
        """Build the current patch frame from ``contacts``.

        With frame reuse, a captured build whose contact frame the previous
        build of its capture scope sorted keeps those pairs instead of sorting
        (see :class:`_FrictionPatchState`). ``capture_pause`` returns a context
        manager entered around a build sorted under the opt-in conditional node,
        which pauses the graph capture to record its body, so the caller joins
        its forked streams on entry and records again the events it waits on
        later. A capture that still holds forked work nobody joined cannot
        pause; that build sorts unconditionally, as an eager build does.
        """
        bodies = 0 if state.body_q is None else state.body_q.shape[0]
        if bodies + model.shape_body.shape[0] + 2 != self.key_stride:
            raise ValueError(
                "Friction patch pair keys were sized for another model: "
                f"{bodies} body poses and {model.shape_body.shape[0]} shapes"
            )
        prepare_inputs = self._prepare_inputs(
            model,
            state,
            contacts,
            body_to_articulation=body_to_articulation,
            is_free_rigid=is_free_rigid,
            contact_gap_gate=contact_gap_gate,
            same_articulation_gap_gate=same_articulation_gap_gate,
            articulation_pair_gap_gate=articulation_pair_gap_gate,
            friction_gap=friction_gap,
            friction_articulation_pairs_only=friction_articulation_pairs_only,
        )
        tracked = self.reuse_frames or self.conditional_nodes
        generation = getattr(contacts, "contact_generation", None) if tracked else None
        scope = None
        tick = None
        if generation is None:
            # The slots receive a frame no record can name.
            self.forget_frame()
            self._prepare_and_sort(prepare_inputs)
        else:
            # The token identifies the contact stream, also across graphs captured
            # with other Contacts instances.
            stream = self.stream_token(generation)
            write = contact_frame_write(generation)
            tick = None if write is None else write.tick
            scope = self._capture_scope()
            if self._keeps_pairs(scope, generation, tick):
                # The previous build of this capture scope sorted this stream's frame
                # and the host issued no write of it since.
                self._launch_prepare_frame(prepare_inputs, generation, stream, _FRAME_KEEP)
            elif scope is None and not self.conditional_nodes:
                # Eager builds sort as without frame reuse: no later build keeps
                # their pairs, so their frame needs no record.
                self._prepare_and_sort(prepare_inputs)
            else:
                self._sort_frame(prepare_inputs, contacts, generation, stream, scope, capture_pause)
                # A node pauses the capture, after which Warp names the main graph's
                # raw handle; later builds of the graph compare with that scope.
                scope = self._capture_scope()
        self._last_build = None if scope is None else (scope, weakref.ref(generation), tick)
        self._launch_build(model, state)

    def _keeps_pairs(self, scope, generation, tick):
        """Whether a build in capture ``scope`` keeps the sorted pairs the previous build left in the slots.

        The previous build must be captured in the same scope over the same
        generation array, with ``tick``, the stream's last noted frame write,
        unchanged since and no :meth:`forget_frame` in between.
        """
        last = self._last_build
        return (
            self.reuse_frames
            and scope is not None
            and last is not None
            and last[0] == scope
            and last[1]() is generation
            and last[2] == tick
        )

    def _prepare_and_sort(self, prepare_inputs):
        """Prepare and sort a frame that no record names, as builds without frame reuse do."""
        wp.launch(_prepare, dim=self.threads, inputs=prepare_inputs, outputs=[self.extent], device=self.device)
        self._order_pairs()

    def _sort_frame(self, prepare_inputs, contacts, generation, stream, scope, capture_pause):
        """Prepare and sort a frame-tracking build, under a conditional node where the opt-in allows one."""
        branch = self._capture_branch() if self.conditional_nodes and scope is not None else None
        # Only _order_pairs runs under the node: _prepare_frame computes the
        # condition, and _build, the link, the seed and the finish stride over
        # the frame's extent at other stages of the step, where each node
        # would pause the capture again to skip microseconds of work.
        with nullcontext() if branch is None else capture_pause():
            if branch is not None and not self._capture_can_pause():
                branch = None
            if branch is None:
                # The prepare records the frame that the sort then orders.
                self._launch_prepare_frame(prepare_inputs, generation, stream, _FRAME_SORT)
                self._order_pairs()
            else:
                self._launch_prepare_frame(prepare_inputs, generation, stream, _FRAME_DETECT)
                record = (generation, contacts.rigid_contact_count, stream)
                branch(self.sort_pairs, on_true=self._order_pairs, record=record)

    def _launch_build(self, model, state):
        """Build the pairs of the prepared, sorted slots."""
        # Warp compiles this module once more for _build's block size; the
        # variants differ only in the tile block size, which no _build
        # arithmetic reads. That variant also tiles _sort_chunks for 32
        # threads, which no launch uses and which about doubles the variant's
        # compile time on a cold kernel cache.
        wp.launch(
            _build,
            dim=self.build_threads,
            inputs=[
                self.extent,
                self.stored,
                state.body_q,
                self.previous_q,
                model.shape_transform,
                model.shape_collision_radius,
                self.build_threads,
                self.current,
                self.previous,
                self.view,
            ],
            outputs=[self.built],
            device=model.device,
            block_dim=self.build_block_dim,
        )

    def _prepare_inputs(
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
        friction_articulation_pairs_only=False,
    ):
        """Return the inputs that :func:`_prepare` and :func:`_prepare_frame` share, with :meth:`build`'s gates."""
        return [
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
            int(friction_articulation_pairs_only),
            self.key_stride,
            self.stored,
            self.built,
            self.threads,
            self.current,
            self.view,
        ]

    def _launch_prepare_frame(self, prepare_inputs, generation, stream, mode):
        """Prepare a frame-reusing build in ``mode`` (``_FRAME_SORT``, ``_FRAME_DETECT`` or ``_FRAME_KEEP``)."""
        wp.launch(
            _prepare_frame,
            dim=self.threads,
            inputs=[*prepare_inputs, generation, stream, int(mode), self.sorted_frame],
            outputs=[self.extent, self.sort_pairs, self.frame_status],
            device=self.device,
        )

    def _capture_scope(self):
        """Return the capture scope that this device's current stream records into, ``None`` when eager."""
        return capture_scope(self.device)

    def _capture_branch(self):
        """Return :func:`warp.capture_if` while Warp captures this device's stream, else ``None``.

        Only a graph replay evaluates the condition on the device; eager builds
        sort unconditionally, so no condition is read back on the host.
        """
        device = self.device
        if device.is_cuda and device.captures.get(device.stream) is not None:
            return wp.capture_if
        return None

    def _capture_can_pause(self):
        """Whether the capture can pause for a conditional node: no forked stream holds unjoined work."""
        return not capture_has_unjoined_work(self.device)

    def _order_pairs(self, record=None):
        """Sort the prepared slots by pair key, the work that only a new contact frame needs.

        A captured build runs this as the body of its conditional node, so this is
        the one place that lists what an empty or reused frame skips; work added
        here must leave outputs that stay valid while later builds reuse the frame.
        ``record``, the contact frame's ``(generation, count array, stream token)``,
        is stored as the frame the slots now hold sorted.
        """
        if self.live_sort:
            self._sort_live()
        else:
            # Empty slots hold INT64_MAX, whose low key_bits are all ones: no pair
            # key exceeds them, and the stable sort keeps live slots ahead of a tie.
            wp.utils.radix_sort_pairs(self.current.keys, self.current.indices, self.capacity, end_bit=self.key_bits)
        if record is not None:
            wp.launch(_record_sorted_frame, dim=1, inputs=[*record], outputs=[self.sorted_frame], device=self.device)

    def _sort_live(self):
        """Stably sort the live slots by pair key and leave every other slot untouched.

        A stable sort of slots [0, live) from indices c is the unique order by
        (key, c), which the full radix sort produces; slots past live already
        hold the empty record that sort leaves there.
        """
        keys, indices = self.current.keys, self.current.indices
        n = self.capacity
        wp.launch_tiled(
            _sort_chunks,
            dim=[self.sort_blocks],
            inputs=[keys[:n], indices[:n], keys, indices, self.extent, n, self.sort_blocks],
            block_dim=_BLOCK_THREADS,
            device=self.device,
        )
        for merge in range(self.merges):
            wp.launch(
                _merge_runs,
                dim=self.merge_threads,
                inputs=[keys, indices, self.extent, self.capacity, merge, self.merge_threads],
                device=self.device,
            )

    def link(self, world, slot, path, slots_needed, routes):
        """Link the normal rows of each patch and share its Coulomb budget on the enabled routes."""
        wp.launch(
            link_patch_routes,
            dim=self.threads,
            inputs=[self.extent, self.view, world, slot, path, slots_needed, routes, self.threads],
            device=self.device,
        )

    def seed(self, state, world, slot, path, slots_needed, routes):
        """Transport cached patch impulses on the enabled routes, scaled by each route's decay."""
        wp.launch(
            seed_patch_routes,
            dim=self.threads,
            inputs=[
                self.extent,
                self.current,
                self.previous,
                state.body_q,
                world,
                slot,
                path,
                slots_needed,
                routes,
                self.threads,
            ],
            device=self.device,
        )

    def finish(self, state_in, state_out, com, world, slot, path, slots_needed, routes, dt):
        """Cache solved friction and release anchors on the enabled routes, then store the frame."""
        self._finish(
            state_in.body_q, state_out.body_q, state_out.body_qd, com, world, slot, path, slots_needed, routes, dt
        )

    def store(self, state):
        """Store the current frame as built, without finishing any anchor."""
        self._finish(state.body_q, None, None, None, None, None, None, None, _PatchRoutes(), 0.0)

    def _finish(self, q, q_out, qd_out, com, world, slot, path, slots_needed, routes, dt):
        wp.launch(
            _finish_and_store,
            dim=self.threads,
            inputs=[
                self.extent,
                q,
                q_out,
                qd_out,
                com,
                self.body_world,
                world,
                slot,
                path,
                slots_needed,
                routes,
                dt,
                self.threads,
                self.current,
                self.previous,
                self.previous_world,
                self.previous_q,
            ],
            outputs=[self.stored],
            device=self.device,
        )

    def reset(self, world_mask):
        """Drop the stored anchors of the masked worlds (all worlds without a mask)."""
        wp.launch(
            _reset_history,
            dim=self.threads,
            inputs=[world_mask, self.stored, self.threads, self.previous_world, self.previous.valid],
            device=self.device,
        )


@wp.func
def patch_normal_load(parents: wp.array2d[int], impulses: wp.array2d[float], world: int, first: int):
    """Sum a region's live normal impulses; -1 terminates ordinary point contacts.

    A contact row's parent field links the next normal in its region (a circular
    list). Friction rows still point to their adjacent normal row. Other row
    families retain their original parent semantics.
    """
    load = impulses[world, first]
    row = parents[world, first]
    while row >= 0 and row != first:
        load += impulses[world, row]
        row = parents[world, row]
    return load


@wp.struct
class _PatchRoutes:
    """Row arrays of the contact routes that hold patch rows; ``enabled`` gates each route.

    ``scale`` holds each route's warm-start decay, which only seeding reads.
    """

    enabled: wp.vec3i
    scale: wp.vec3
    parents0: wp.array2d[int]
    parents1: wp.array2d[int]
    parents2: wp.array2d[int]
    mu0: wp.array2d[float]
    mu1: wp.array2d[float]
    mu2: wp.array2d[float]
    impulses0: wp.array2d[float]
    impulses1: wp.array2d[float]
    impulses2: wp.array2d[float]


def patch_routes(rows):
    """Pack ``(route, parents, mu, impulses, scale)`` rows into one launch argument.

    Routes absent from ``rows`` stay disabled, and no launch reads their arrays.
    """
    routes = _PatchRoutes()
    enabled = [0, 0, 0]
    scale = [0.0, 0.0, 0.0]
    for route, parents, mu, impulses, decay in rows:
        enabled[route] = 1
        scale[route] = decay
        setattr(routes, f"parents{route}", parents)
        setattr(routes, f"mu{route}", mu)
        setattr(routes, f"impulses{route}", impulses)
    routes.enabled = wp.vec3i(*enabled)
    routes.scale = wp.vec3(*scale)
    return routes


@wp.func
def _contact_route(routes: _PatchRoutes, slot: wp.array[int], path: wp.array[int], c: int):
    """Return the enabled route that holds contact ``c``'s rows, or -1 without rows."""
    route = int(-1)
    if routes.enabled[0] + routes.enabled[1] + routes.enabled[2] > 0 and slot[c] >= 0:
        route = path[c]
        if route < 0 or route > 2 or routes.enabled[route] == 0:
            route = -1
    return route


@wp.func
def _route_parents(routes: _PatchRoutes, route: int):
    if route == 1:
        return routes.parents1
    if route == 2:
        return routes.parents2
    return routes.parents0


@wp.func
def _route_mu(routes: _PatchRoutes, route: int):
    if route == 1:
        return routes.mu1
    if route == 2:
        return routes.mu2
    return routes.mu0


@wp.func
def _route_impulses(routes: _PatchRoutes, route: int):
    if route == 1:
        return routes.impulses1
    if route == 2:
        return routes.impulses2
    return routes.impulses0


@wp.func
def _link_contact(
    c: int,
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

    Row builders write the unscaled friction coefficient; linking divides it by
    the number of anchors that actually received rows, so the pooled normal load
    is shared exactly once.
    """
    row = patches.next_contact[c]
    if row < 0:
        return
    if row == c:
        # A one-contact region's normal row closes its own ring, and its budget
        # stays whole: the general path divides it by one anchor or not at all.
        parents[world[c], slot[c]] = slot[c]
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
    """Link the allocated rows of one route; see :func:`_link_contact`."""
    c = wp.tid()
    if c >= count[0] or slot[c] < 0 or path[c] != route:
        return
    _link_contact(c, patches, world, slot, path, slots_needed, route, parents, mu)


@wp.kernel(enable_backward=False)
def link_patch_routes(
    extent: wp.array[int],
    patches: FrictionPatches,
    world: wp.array[int],
    slot: wp.array[int],
    path: wp.array[int],
    slots_needed: wp.array[int],
    routes: _PatchRoutes,
    threads: int,
):
    """Link the rows of every enabled route in one launch over the live contacts.

    A contact writes only its own rows, on the route that holds them, so this
    matches one launch per route.
    """
    thread = wp.tid()
    for c in range(thread, extent[1], threads):
        route = _contact_route(routes, slot, path, c)
        if route >= 0:
            parents = _route_parents(routes, route)
            mu = _route_mu(routes, route)
            _link_contact(c, patches, world, slot, path, slots_needed, route, parents, mu)


@wp.func
def contact_tangent_basis(n: wp.vec3):
    """Return the deterministic tangent pair every FeatherPGS friction row uses for ``n``."""
    tangent0 = wp.cross(n, wp.vec3(1.0, 0.0, 0.0))
    if wp.length_sq(tangent0) < 1.0e-12:
        tangent0 = wp.cross(n, wp.vec3(0.0, 1.0, 0.0))
    tangent0 = wp.normalize(tangent0)
    tangent1 = wp.normalize(wp.cross(n, tangent0))
    return tangent0, tangent1


@wp.func
def _seed_contact(
    c: int,
    frame: _PatchFrame,
    prev: _PatchFrame,
    q: wp.array[wp.transform],
    world: wp.array[int],
    slot: wp.array[int],
    parents: wp.array2d[int],
    mu: wp.array2d[float],
    impulses: wp.array2d[float],
    scale: float,
):
    """Transport cached patch impulses after all contact normals have been seeded.

    Anchors without patch history keep whatever the contact-matched warm start
    seeded; only carried anchors overwrite their friction rows.
    """
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
    """Seed the friction rows of one route; see :func:`_seed_contact`."""
    c = wp.tid()
    if c >= count[0] or path[c] != route or slot[c] < 0 or slots_needed[c] != 3:
        return
    _seed_contact(c, frame, prev, q, world, slot, parents, mu, impulses, scale)


@wp.kernel(enable_backward=False)
def seed_patch_routes(
    extent: wp.array[int],
    frame: _PatchFrame,
    prev: _PatchFrame,
    q: wp.array[wp.transform],
    world: wp.array[int],
    slot: wp.array[int],
    path: wp.array[int],
    slots_needed: wp.array[int],
    routes: _PatchRoutes,
    threads: int,
):
    """Seed every enabled route in one launch over the live contacts, scaled by its decay.

    A contact writes only its own friction rows and reads normal rows, which
    seeding never writes, so this matches one launch per route.
    """
    thread = wp.tid()
    for c in range(thread, extent[1], threads):
        route = _contact_route(routes, slot, path, c)
        if route >= 0 and slots_needed[c] == 3:
            parents = _route_parents(routes, route)
            mu = _route_mu(routes, route)
            impulses = _route_impulses(routes, route)
            _seed_contact(c, frame, prev, q, world, slot, parents, mu, impulses, routes.scale[route])


@wp.func
def _point_velocity(
    q: wp.array[wp.transform], qd: wp.array[wp.spatial_vector], com: wp.array[wp.vec3], body: int, anchor: wp.vec3
):
    velocity = wp.vec3(0.0)
    if body >= 0:
        velocity = velocity_at_point(qd[body], wp.transform_vector(q[body], anchor - com[body]))
    return velocity


@wp.func
def _finish_contact(
    c: int,
    frame: _PatchFrame,
    q: wp.array[wp.transform],
    q_out: wp.array[wp.transform],
    qd_out: wp.array[wp.spatial_vector],
    com: wp.array[wp.vec3],
    world: wp.array[int],
    slot: wp.array[int],
    slots_needed: wp.array[int],
    parents: wp.array2d[int],
    mu: wp.array2d[float],
    impulses: wp.array2d[float],
    dt: float,
):
    """Cache solved friction and release saturated anchors with tangential motion.

    The tolerance is relative to geometry and timestep. Merely resting at the
    Coulomb limit does not prove sliding, so it must not erase static history.
    """
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
    """Finish the anchors of one route; see :func:`_finish_contact`."""
    c = wp.tid()
    if c >= count[0] or frame.valid[c] == 0:
        return
    if slot[c] < 0:
        # A contact of a resolved world would have solved zero load on the complete
        # path, which releases its anchor. Capacity rejection provides no solved
        # normal load; preserve the geometrically supported history until rows are
        # available again.
        if path[c] == CONTACT_PATH_RESOLVED:
            frame.valid[c] = 0
        return
    if path[c] != route:
        return
    _finish_contact(c, frame, q, q_out, qd_out, com, world, slot, slots_needed, parents, mu, impulses, dt)


@wp.kernel(enable_backward=False)
def _finish_and_store(
    extent: wp.array[int],
    q: wp.array[wp.transform],
    q_out: wp.array[wp.transform],
    qd_out: wp.array[wp.spatial_vector],
    com: wp.array[wp.vec3],
    body_world: wp.array[int],
    world: wp.array[int],
    slot: wp.array[int],
    path: wp.array[int],
    slots_needed: wp.array[int],
    routes: _PatchRoutes,
    dt: float,
    threads: int,
    frame: _PatchFrame,
    prev: _PatchFrame,
    previous_world: wp.array[int],
    previous_q: wp.array[wp.transform],
    stored: wp.array[int],
):
    """Finish each live anchor on its route, then store every slot this frame visits.

    Finishing writes only the contact's own validity and tangent impulse and
    reads other contacts only through the row arrays, so finishing and storing
    contact by contact matches finishing every route before storing. A contact
    the resolved-world classifier left without rows (CONTACT_PATH_RESOLVED)
    releases its anchor, as its zero load does on the complete path. Other
    anchors without rows (capacity rejection, or a disabled route) keep their
    history. Thread 0 publishes the stored count, which this launch never reads.
    """
    thread = wp.tid()
    live = extent[1]
    if thread == 0:
        stored[0] = live
    finishing = routes.enabled[0] + routes.enabled[1] + routes.enabled[2] > 0
    for c in range(thread, extent[0], threads):
        if finishing and c < live and frame.valid[c] != 0:
            if slot[c] < 0 and path[c] == CONTACT_PATH_RESOLVED:
                frame.valid[c] = 0
            else:
                route = _contact_route(routes, slot, path, c)
                if route >= 0:
                    parents = _route_parents(routes, route)
                    mu = _route_mu(routes, route)
                    impulses = _route_impulses(routes, route)
                    _finish_contact(
                        c, frame, q, q_out, qd_out, com, world, slot, slots_needed, parents, mu, impulses, dt
                    )
        _store_contact(c, q, body_world, frame, prev, previous_world, previous_q)


@wp.kernel(enable_backward=False)
def _reset_history(
    world_mask: wp.array[wp.bool],
    stored: wp.array[int],
    threads: int,
    prev_world: wp.array[int],
    prev_valid: wp.array[int],
):
    """Drop the stored anchors of reset worlds (all worlds for an empty mask).

    Slots from the stored count on hold no anchor.
    """
    thread = wp.tid()
    for c in range(thread, stored[0], threads):
        if world_mask:
            w = prev_world[c]
            if w >= 0 and w < world_mask.shape[0] and not world_mask[w]:
                continue
        prev_valid[c] = 0


@wp.kernel(enable_backward=False)
def _arange(values: wp.array[int]):
    i = wp.tid()
    values[i] = i
