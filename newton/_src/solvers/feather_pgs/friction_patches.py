# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver-owned friction regions, independent of collision contact identities.

Normal contacts are never reduced here. A region is one friction wrench (core-docs/WRENCH/VERDICT.md): one anchor, a
tangent pair and a spin row on the ball ``|(t1 / mu, t2 / mu, s / mu_t)| <= lambda_n`` of the region's pooled normal
impulse, with ``mu_t = mu rho`` and ``rho`` 0.94 times the polar radius of gyration of the members' convex hull (a
uniform patch's mean radius). The anchor sits at the region's centre of pressure, carried from the last solve, or at
the hull's centroid; the region's last eligible member in contact order carries its rows, so a Gauss-Seidel sweep
reaches them after every normal of the region. A one-contact region has no footprint: its friction is a pair, and a
spin row only under contact torsion, with the authored torsional coefficient (:func:`patch_spin_rows`).
Regions require matching friction materials, nearly aligned normals, nearby
contact planes, and connected shape bounding spheres on each body. The latter
is a conservative test across convex seams, not an exact surface-connectivity
query. Length tolerances scale with the smaller body's bounding radius; angular
tolerance is about six degrees. Normals remain individual unilateral rows.

Tangential and twist displacement persist across frames: they are the friction's
history, and a warm-started solve carries no friction impulse beside them
(:mod:`.warm_start`). Friction locations follow the current contact region.
Displacement accumulates incremental rigid motion at those locations, using the
pose increment's SE(3) logarithm rather than a rotating material point's chord.
Saved poses also capture externally imposed motion and position-only solver
passes. Pair identity and geometric compatibility correlate history without a
collision matcher or shape-type rules. Unloaded regions and saturated friction
with opposing slip release history; saturation alone does not. All matching
tolerances are internal so correction strength is the only new user control.

Sorting body pairs makes construction local to a pair and preserves CUDA graph
capture: all buffers and the radix-sort workspace have a fixed capacity, the
ready rows of the solver's contact pool, which only change between captures.
Growth keeps the frames' contents and gives the new slots the empty record.
Launches
stride over the live contacts and the slots earlier frames used, so their work
follows the live contact count rather than the capacity. With frame reuse, the
solver's default, a captured build skips the pair sort when the previous build
was captured in the same graph over the same contact stream (its
``contact_generation`` array) and the host issued no write of that stream in
between, which is the case on every solver substep that shares
a collision frame (see :class:`_FrictionPatchState`).
"""

import itertools
import weakref

import numpy as np
import warp as wp

from ...core.contact_pool import ContactField, ContactPool
from ...core.live_sort import (
    EXTENT_SIZE,
    SLOT_BITS,
    SORT_CHUNK,
    launch_live_sort,
    publish_sort_extent,
    slot_keys_fit,
    sort_grid,
)
from ...sim.contacts import GENERATION_SENTINEL, capture_scope, contact_frame_write
from . import field_group
from .contact_filters import contact_normal_gap_limit
from .contact_laws import contact_bodies, contact_point, contact_points, contact_tangent_basis
from .field_group import Field
from .spatial_laws import point_velocity

# Grids are fixed at construction so CUDA graphs replay them. Their sizes count
# _BLOCK_THREADS threads per SM (one SM on the CPU); launches stride over the
# live slots, so no grid changes any output.
_BLOCK_THREADS = 256
# The light launches (prepare, link, finish and store, reset) handle one
# slot per loop iteration at 40 to 48 registers, which fit five or six blocks
# per SM, and their reads through slots and rings wait on memory; blocks past
# the live slots only read the counts.
_LIGHT_BLOCKS_PER_SM = 6
# A _build thread walks one body pair's run, a chain of dependent loads at 96
# registers. One-warp blocks fit 21 warps per SM instead of two 256-thread
# blocks' 16, free each warp's slot when it finishes rather than with its
# slowest block-mate, and spread a short frame's runs over every SM; the wider
# grid leaves no thread a second run below 8 * SMs * 256 live slots.
_BUILD_BLOCK_DIM = 32
_BUILD_BLOCKS_PER_SM = 8
# The patch extent: the slots to visit, then the live sort's extent (live_sort.publish_sort_extent).
_SORT_EXTENT = wp.constant(1)
# Contact paths 0 and 1 hold dense and matrix-free rows, and -1 holds none.
# CONTACT_PATH_RESOLVED marks a contact without rows in a world the resolved-world classifier
# skipped, where the complete path would have reserved them: its patch anchor releases.
CONTACT_PATH_RESOLVED = wp.constant(-2)
# A region's spin arm over the polar radius of gyration of its contacts' hull: the mean radius of a convex patch under
# uniform pressure over its gyration radius is 0.937 on a square, 0.943 on a disc and 0.866 on a line, and 0.94 holds
# every convex footprint within 4% of quadrature (core-docs/WRENCH/VERDICT.md section 6).
_SPIN_ARM = wp.constant(0.94)
# The contact generation no frame has: a recorded frame with it never matches.
_FRAME_SENTINEL = wp.constant(wp.int64(GENERATION_SENTINEL))
# Modes of _prepare_frame, the host's admission of a frame-reusing build: the
# pair sort follows unconditionally, or the slots keep the sorted pairs of the
# frame an earlier build recorded. No build launches the kernel's detect mode.
_FRAME_SORT = wp.constant(0)
_FRAME_DETECT = wp.constant(1)
_FRAME_KEEP = wp.constant(2)


@wp.struct
class FrictionPatches:
    """Per-contact view consumed by the existing contact row builders."""

    weight: wp.array[float]
    next_contact: wp.array[int]
    point: wp.array[wp.vec3]
    phi: wp.array[wp.vec2]
    spin_arm: wp.array[float]
    spin_phi: wp.array[float]


PATCH_ROW_LAW = (
    Field("patches", FrictionPatches, owner="friction_patches", attr="view"),
    Field("beta", wp.float32, owner="laws", attr="friction_anchor_beta"),
)

PatchRowLaw = field_group.struct("PatchRowLaw", PATCH_ROW_LAW, __name__)
"""The friction row law, for producers that write their own friction rows.

A tangent row acts at its region's friction location, carries the anchor's tangential displacement as phi and
corrects it with ``beta`` (``friction_anchor_beta``, the stick's gain, positive). Producers write the undivided friction coefficient; the patch
link (``_FrictionPatchState.link``, which launches ``link_patch_routes``) divides it and links the
ring. The Warp producers apply the law through :func:`.contact_input.contact_frame` and
:func:`patch_tangent_row`.
"""


def patch_row_law(solver) -> "PatchRowLaw":
    """The solver's patch row law for this step. Bound every step: contact-pool growth replaces the patch view."""
    return field_group.bind(PatchRowLaw, laws=solver.laws, friction_patches=solver.laws.friction_patches)


@wp.func
def patch_friction_location(patches: FrictionPatches, c: int):
    """Return contact ``c``'s friction location, in world coordinates."""
    return patches.point[c]


@wp.func
def patch_tangent_row(law: PatchRowLaw, c: int, row: int):
    """Return contact ``c``'s tangent row ``row`` (1 or 2): its phi and beta. Its coefficient is the contact's ``mu``,
    whole: a region has one anchor."""
    return law.patches.phi[c][row - 1], law.beta


@wp.func
def patch_spin_rows(patches: FrictionPatches, c: int, spin: int):
    """Return the spin rows (0 or 1) of contact ``c``'s friction where the plan lets ``spin`` contacts carry one (0
    none, 1 those whose region has a footprint, 2 every one: :attr:`.rows.RowLaws.spin_rows_where`): one where its
    region has a footprint (a positive spin arm) or contact torsion gives every friction contact one, none on a
    one-contact region otherwise, and none where the model admits no footprint (contacts it cannot make carry none)."""
    rows = int(0)
    if spin == 2 or (spin == 1 and patches.spin_arm[c] > 0.0):
        rows = 1
    return rows


@wp.func
def patch_spin_row(law: PatchRowLaw, c: int, mu: float, authored: float):
    """Return contact ``c``'s spin row law: its phi (the region's twist displacement), beta and coefficient ``mu_t``,
    the friction coefficient ``mu`` times the region's spin arm, or the ``authored`` torsional coefficient where that
    is larger (contact torsion; 0 without it)."""
    return law.patches.spin_phi[c], law.beta, wp.max(authored, mu * law.patches.spin_arm[c])


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
    pressure: wp.array[wp.vec3]
    pressured: wp.array[int]
    twist: wp.array[float]


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
def _surface_gap(q: wp.array[wp.transform], a: int, point_a: wp.vec3, b: int, point_b: wp.vec3, normal: wp.vec3):
    """The gap along ``normal`` between two points held in their bodies' frames (a static shape's in world
    coordinates), formed as :func:`.contact_laws.contact_points` forms a contact's: translations apart from the rotated
    points, so it carries no rounding of world points."""
    translation_a = point_a
    offset_a = wp.vec3(0.0)
    translation_b = point_b
    offset_b = wp.vec3(0.0)
    if a >= 0:
        translation_a = wp.transform_get_translation(q[a])
        offset_a = wp.quat_rotate(wp.transform_get_rotation(q[a]), point_a)
    if b >= 0:
        translation_b = wp.transform_get_translation(q[b])
        offset_b = wp.quat_rotate(wp.transform_get_rotation(q[b]), point_b)
    return wp.dot(normal, (translation_a - translation_b) + (offset_a - offset_b))


@wp.func
def _local_surface(q: wp.array[wp.transform], body: int, translation: wp.vec3, offset: wp.vec3):
    """A contact's surface point, ``translation + offset`` (:func:`.contact_laws.contact_points`), in its body's frame
    (a static shape's in world coordinates): the offset rotated back, which carries no rounding of the translation."""
    result = translation + offset
    if body >= 0:
        result = wp.quat_rotate_inv(wp.transform_get_rotation(q[body]), offset)
    return result


@wp.struct
class _PatchContacts:
    """The contacts a frame build reads and the gates it applies, which :func:`_prepare` and :func:`_prepare_frame`
    share (the host's :meth:`_FrictionPatchState._prepare_inputs`)."""

    shape0: wp.array[int]
    shape1: wp.array[int]
    point0: wp.array[wp.vec3]
    point1: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    margin0: wp.array[float]
    margin1: wp.array[float]
    shape_body: wp.array[int]
    shape_mu: wp.array[float]
    shape_gap: wp.array[float]
    body_radius: wp.array[float]
    q: wp.array[wp.transform]
    body_to_articulation: wp.array[int]
    is_free_rigid: wp.array[int]
    contact_gap_gate: float
    same_articulation_gap_gate: float
    articulation_pair_gap_gate: float
    friction_gap: float
    key_stride: int


@wp.func
def _prepare_contact_fields(c: int, live: int, inputs: _PatchContacts, frame: _PatchFrame, patches: FrictionPatches):
    """Prepare contact ``c``'s frame record, all but slot ``c``'s pair key and index; return the pair key.

    Past ``live`` the record is empty and the key the empty key. A build stores the key and the index in
    slot ``c``; a build that reuses sorted pairs can leave the slots' keys and indices untouched.
    """
    frame.owner[c] = -1
    frame.valid[c] = 0
    frame.used[c] = 0
    frame.source[c] = -1
    frame.eligible[c] = 0
    frame.pressured[c] = 0
    patches.weight[c] = 0.0
    patches.next_contact[c] = -1
    patches.phi[c] = wp.vec2(0.0)
    patches.spin_phi[c] = 0.0
    if c >= live:
        return wp.int64(0x7FFFFFFFFFFFFFFF)
    sa = inputs.shape0[c]
    sb = inputs.shape1[c]
    a, b = contact_bodies(sa, sb, inputs.shape_body)
    ka = a + 1
    kb = b + 1
    if a < 0:
        ka = inputs.q.shape[0] + (sa if sa >= 0 else inputs.shape_body.shape[0]) + 1
    if b < 0:
        kb = inputs.q.shape[0] + (sb if sb >= 0 else inputs.shape_body.shape[0]) + 1
    n, translation_a, offset_a, translation_b, offset_b, separation, gap = contact_points(
        inputs.normal[c], inputs.point0[c], inputs.point1[c], inputs.margin0[c], inputs.margin1[c], a, b, inputs.q
    )
    surface_a = _local_surface(inputs.q, a, translation_a, offset_a)
    surface_b = _local_surface(inputs.q, b, translation_b, offset_b)
    art_a = int(-1)
    art_b = int(-1)
    if a >= 0 and a < inputs.body_to_articulation.shape[0]:
        art_a = inputs.body_to_articulation[a]
    if b >= 0 and b < inputs.body_to_articulation.shape[0]:
        art_b = inputs.body_to_articulation[b]
    a_non_free = art_a >= 0 and inputs.is_free_rigid[art_a] == 0
    b_non_free = art_b >= 0 and inputs.is_free_rigid[art_b] == 0
    support_gap_limit = float(0.0)
    if sa >= 0:
        support_gap_limit += inputs.shape_gap[sa]
    if sb >= 0:
        support_gap_limit += inputs.shape_gap[sb]
    normal_gap_limit = contact_normal_gap_limit(
        a_non_free,
        b_non_free,
        art_a == art_b,
        inputs.contact_gap_gate,
        inputs.articulation_pair_gap_gate,
        inputs.same_articulation_gap_gate,
    )
    support_gap_limit = wp.min(support_gap_limit, normal_gap_limit)
    eligible = gap <= normal_gap_limit and gap <= inputs.friction_gap
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
    frame.body_a[c] = a
    frame.body_b[c] = b
    frame.shape_a[c] = sa
    frame.shape_b[c] = sb
    frame.flipped[c] = flip
    frame.surface_a[c] = surface_b if flip != 0 else surface_a
    frame.surface_b[c] = surface_a if flip != 0 else surface_b
    frame.center[c] = contact_point(translation_a, offset_a, separation)
    frame.normal[c] = n
    mu_a = float(0.0)
    mu_b = float(0.0)
    if sa >= 0:
        mu_a = inputs.shape_mu[sa]
    if sb >= 0:
        mu_b = inputs.shape_mu[sb]
    frame.mu[c] = wp.vec2(mu_a, mu_b)
    r = float(1.0e10)
    if a >= 0:
        r = wp.min(r, inputs.body_radius[a])
    if b >= 0:
        r = wp.min(r, inputs.body_radius[b])
    frame.radius[c] = wp.max(r, 1.0e-4)
    # ka and kb are below key_stride, so the key orders pairs as (ka, kb) does
    # and occupies only the low bits that build() sorts.
    return wp.int64(ka) * wp.int64(inputs.key_stride) + wp.int64(kb)


# Defined ahead of _prepare_frame: a kernel of a unique module is named after the functions
# defined when it is decorated, and a later definition would rename it after its first load.
@wp.func
def _publish_extent(extent: wp.array[int], end: int, live: int):
    """Publish the slots to visit, then the live sort's extent of the live slots.

    ``extent[0]`` holds the slots to visit and ``extent[1:]`` the sort extent
    (:func:`~newton._src.core.live_sort.publish_sort_extent`), whose first entry
    is the live count: a captured live sort follows it, launching no work past
    the live slots.
    """
    extent[0] = end
    publish_sort_extent(extent, _SORT_EXTENT, live)


@wp.kernel(enable_backward=False)
def _prepare(
    count: wp.array[int],
    inputs: _PatchContacts,
    stored: wp.array[int],
    built: wp.array[int],
    threads: int,
    frame: _PatchFrame,
    patches: FrictionPatches,
    extent: wp.array[int],
):
    """Fill this frame's contact slots and clear every slot an earlier frame used.

    A collision pass clamps the count to the rows it wrote, so every slot below
    it is live. Slots past the stored and the last built frame already hold the empty
    record. Thread 0 publishes the slots to visit and the live count; this
    launch reads neither.
    """
    thread = wp.tid()
    live = count[0]
    if live < 0:
        live = 0
    end = wp.max(live, wp.max(stored[0], built[0]))
    if thread == 0:
        _publish_extent(extent, end, live)
    for c in range(thread, end, threads):
        frame.indices[c] = c
        frame.keys[c] = _prepare_contact_fields(c, live, inputs, frame, patches)


# The frame-reuse variants compile in modules of their own, so the base patch
# kernels keep their module and generated code. Pairs are built by the base
# _build kernel itself: a second kernel that inlines _build_pair rounds the
# carried displacements differently on CUDA.
@wp.kernel(enable_backward=False, module="unique")
def _prepare_frame(
    count: wp.array[int],
    inputs: _PatchContacts,
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
    if live < 0:
        live = 0
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
        _publish_extent(extent, end, live)
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
        key = _prepare_contact_fields(c, live, inputs, frame, patches)
        # A kept slot holds the sorted pair at position c, not contact c's key.
        if recorded == 0 or c >= live:
            frame.indices[c] = c
            frame.keys[c] = key


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
def _pose_twist(q: wp.array[wp.transform], previous_q: wp.array[wp.transform], body: int, normal: wp.vec3):
    """The pose increment's rotation about ``normal``: a body's twist since the stored pose, as the carried spin
    displacement reads it."""
    twist = float(0.0)
    if body >= 0:
        rotation = wp.transform_get_rotation(q[body]) * wp.quat_inverse(wp.transform_get_rotation(previous_q[body]))
        if rotation[3] < 0.0:
            rotation = -rotation
        twist = 2.0 * wp.dot(wp.vec3(rotation[0], rotation[1], rotation[2]), normal)
    return twist


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
def _footprint_point(frame: _PatchFrame, c: int, origin: wp.vec3, t0: wp.vec3, t1: wp.vec3):
    """Member ``c``'s contact centre in the region's tangent plane, relative to ``origin``."""
    d = frame.center[c] - origin
    return wp.dot(d, t0), wp.dot(d, t1)


@wp.func
def _region_footprint(
    frame: _PatchFrame,
    member_start: int,
    member_stop: int,
    carry_only: int,
    origin: wp.vec3,
    t0: wp.vec3,
    t1: wp.vec3,
):
    """The region's footprint in its tangent plane: the centroid of its members' convex hull relative to ``origin``,
    and its spin arm, :data:`_SPIN_ARM` times the hull's polar radius of gyration (core-docs/WRENCH/VERDICT.md U6).

    The hull is gift-wrapped over the members, and its moments accumulate over a triangle fan from the first hull
    vertex, so nothing is stored. A hull of no area is a line: its arm is a uniform segment's. Returns ``(x, y, arm)``;
    a region of one point has arm 0.
    """
    count = int(0)
    sum_x = float(0.0)
    sum_y = float(0.0)
    sum_rr = float(0.0)
    start_x = float(1.0e30)
    start_y = float(1.0e30)
    for j in range(member_start, member_stop):
        c = frame.members[j]
        if carry_only != 0 or frame.eligible[c] != 0:
            x, y = _footprint_point(frame, c, origin, t0, t1)
            count += 1
            sum_x += x
            sum_y += y
            sum_rr += x * x + y * y
            if x < start_x or (x == start_x and y < start_y):
                start_x = x
                start_y = y
    mean_x = sum_x / float(wp.max(count, 1))
    mean_y = sum_y / float(wp.max(count, 1))
    spread = wp.max(sum_rr / float(wp.max(count, 1)) - mean_x * mean_x - mean_y * mean_y, 0.0)
    if count < 2 or spread <= 0.0:
        return mean_x, mean_y, 0.0
    tolerance = 1.0e-10 * spread
    x_now = start_x
    y_now = start_y
    area = float(0.0)
    moment_x = float(0.0)
    moment_y = float(0.0)
    polar = float(0.0)
    perimeter = float(0.0)
    far_x = float(0.0)
    far_y = float(0.0)
    far = float(0.0)
    for _step in range(count + 1):
        # The next hull vertex: the member every other lies left of (the farther one on a tie).
        best_x = float(0.0)
        best_y = float(0.0)
        found = int(0)
        for j in range(member_start, member_stop):
            c = frame.members[j]
            if carry_only != 0 or frame.eligible[c] != 0:
                x, y = _footprint_point(frame, c, origin, t0, t1)
                x -= x_now
                y -= y_now
                length = x * x + y * y
                if length > tolerance:
                    cross = best_x * y - best_y * x
                    if found == 0 or cross < 0.0 or (cross == 0.0 and length > best_x * best_x + best_y * best_y):
                        best_x = x
                        best_y = y
                        found = 1
        if found == 0:
            break
        next_x = x_now + best_x
        next_y = y_now + best_y
        perimeter += wp.sqrt(best_x * best_x + best_y * best_y)
        ax = x_now - start_x
        ay = y_now - start_y
        bx = next_x - start_x
        by = next_y - start_y
        triangle = 0.5 * (ax * by - ay * bx)
        area += triangle
        moment_x += triangle * (ax + bx) / 3.0
        moment_y += triangle * (ay + by) / 3.0
        polar += triangle * (ax * ax + ay * ay + bx * bx + by * by + ax * bx + ay * by) / 6.0
        if bx * bx + by * by > far:
            far = bx * bx + by * by
            far_x = bx
            far_y = by
        x_now = next_x
        y_now = next_y
        if bx * bx + by * by <= tolerance:
            break
    span = 0.5 * perimeter
    if wp.abs(area) <= 1.0e-6 * span * span:
        # A line: the walk goes out and back.
        return start_x + 0.5 * far_x, start_y + 0.5 * far_y, _SPIN_ARM * span / wp.sqrt(12.0)
    centre_x = moment_x / area
    centre_y = moment_y / area
    gyration = wp.max(polar / area - centre_x * centre_x - centre_y * centre_y, 0.0)
    return start_x + centre_x, start_y + centre_y, _SPIN_ARM * wp.sqrt(gyration)


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
        # One anchor per region, at its footprint's hull centroid at the members' mean height, whose rows the last
        # eligible member in contact order carries: a Gauss-Seidel sweep reaches them after every normal row of the
        # region, and contact order alone picks the member.
        member_start = member_stop
        for j in range(index, stop):
            c = frame.indices[j]
            if frame.owner[c] != seed:
                continue
            frame.members[member_stop] = c
            member_stop += 1
            if carry_only != 0 or frame.eligible[c] != 0:
                first = c
        normal = frame.normal[seed]
        plane_t0, plane_t1 = contact_tangent_basis(normal)
        origin = frame.center[first]
        foot_x, foot_y, spin_arm = _region_footprint(
            frame, member_start, member_stop, carry_only, origin, plane_t0, plane_t1
        )
        height = float(0.0)
        height_count = int(0)
        for j in range(member_start, member_stop):
            c = frame.members[j]
            if carry_only != 0 or frame.eligible[c] != 0:
                height += wp.dot(frame.center[c] - origin, normal)
                height_count += 1
        location = origin + foot_x * plane_t0 + foot_y * plane_t1 + normal * (height / float(wp.max(height_count, 1)))
        c = first
        a = frame.body_a[c]
        b = frame.body_b[c]
        n = frame.normal[c]
        r = frame.radius[c]
        surface_a = frame.surface_a[c]
        surface_b = frame.surface_b[c]
        current_gap = _surface_gap(q, a, surface_a, b, surface_b, n)
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
            offset = 0.5 * (pa + pb) - location
            distance = wp.length_sq(offset)
            if not (
                distance < nearest
                and distance <= 4.0 * r * r
                and wp.abs(_surface_gap(q, a, prev.anchor_a[p], b, prev.anchor_b[p], n)) <= 0.1 * r
                and wp.abs(wp.dot(offset, n)) <= 0.05 * r
            ):
                continue
            # Honor the existing contact envelope: a zero-gap test drops
            # valid speculative contacts during a squeeze. Surface
            # witnesses retain penetration during decompression and
            # reject rocking beyond the same envelope as fresh contacts.
            support_gap = _surface_gap(q, a, prev.surface_a[p], b, prev.surface_b[p], n)
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
        # The anchor sits at the region's centre of pressure, carried from the last solve: the nearest stored one of
        # this pair, sliding or stuck, since the anchor may have released.
        nearest_pressure = float(1.0e30)
        shift = wp.vec3(0.0)
        for j in range(prev_start, prev_stop):
            p = prev.indices[j]
            if prev.pressured[p] != 0:
                candidate = _world_point(q, a, prev.pressure[p]) - location
                candidate -= n * wp.dot(candidate, n)
                if wp.length_sq(candidate) < nearest_pressure:
                    nearest_pressure = wp.length_sq(candidate)
                    shift = candidate
        if nearest_pressure < 1.0e29:
            # A stale centre stays on the footprint: no further along its direction than the footprint's support.
            shift_length = wp.length(shift)
            if shift_length > 0.0:
                direction = shift / shift_length
                support = float(0.0)
                for j in range(member_start, member_stop):
                    member = frame.members[j]
                    if carry_only != 0 or frame.eligible[member] != 0:
                        support = wp.max(support, wp.dot(frame.center[member] - location, direction))
                if shift_length > support:
                    shift = direction * support
            pa = location + shift
            if chosen >= 0:
                motion = _pose_motion(q, previous_q, a, pa) - _pose_motion(q, previous_q, b, pa)
        anchor_a = _local_point(q, a, pa)
        anchor_b = _local_point(q, b, pa)
        error = wp.vec3(0.0)
        if chosen >= 0:
            prev.used[chosen] = 1
            error = _carried_displacement(q, prev, chosen, n)
            error += motion
            error -= n * wp.dot(error, n)
        stored_error = error
        if a >= 0:
            stored_error = wp.transform_vector(wp.transform_inverse(q[a]), error)
        frame.displacement[c] = stored_error
        # The twist displacement: the spin the anchor has turned through since it stuck, its spin row's phi.
        twist = float(0.0)
        if chosen >= 0:
            twist = prev.twist[chosen] + _pose_twist(q, previous_q, a, n) - _pose_twist(q, previous_q, b, n)
        frame.twist[c] = twist
        frame.source[c] = chosen
        if carry_only != 0 and chosen < 0:
            # Filtered members only carry existing history; they never start
            # it, because no friction row held the pair during this step.
            continue
        frame.anchor_a[c] = anchor_a
        frame.anchor_b[c] = anchor_b
        frame.pressure[c] = anchor_a
        frame.surface_a[c] = surface_a
        frame.surface_b[c] = surface_b
        frame.valid[c] = 1
        patches.weight[c] = 1.0
        patches.spin_arm[c] = spin_arm
        if carry_only != 0:
            patches.weight[c] = 0.0
        if frame.flipped[c] != 0:
            n = -n
        patches.point[c] = pa
        t0, t1 = contact_tangent_basis(n)
        if frame.flipped[c] != 0:
            error = -error
        patches.phi[c] = wp.vec2(wp.dot(t0, error), wp.dot(t1, error))
        patches.spin_phi[c] = twist


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
    # The centre of pressure of the last solve, in body a's frame, sliding or stuck: the next build's anchor.
    prev.pressured[c] = frame.pressured[c]
    if frame.pressured[c] != 0:
        prev.pressure[c] = frame.pressure[c]
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
    prev.twist[c] = frame.twist[c]
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
    prev.pressured[c] = 0
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


def _uses_live_sort(key_bits):
    """Whether the pair sort orders only the live slots instead of the whole capacity.

    It does whenever a chunk key, the pair key with an 11-bit slot appended,
    stays below the pad bit (pair keys up to 51 bits: about 47 million bodies
    and shapes); wider keys take the radix sort of the whole capacity. Both sorts
    leave the unique (key, slot) order. The live sort's cost follows the live
    contacts, the radix sort's the capacity.
    """
    return slot_keys_fit(key_bits)


_BUILD_FIELDS = frozenset(("center", "radius", "support_gap_limit", "flipped", "source", "eligible"))
"""Patch frame fields only a build reads: the stored frame (:attr:`_FrictionPatchState.previous`) holds none."""


class _FrictionPatchState:
    """Own preallocated patch frames; never read device counters on the host.

    With ``reuse_frames``, the host admits each captured build statically. A
    build keeps the sorted pairs the previous build left in the slots when that
    build was captured in the same scope (one graph, or one conditional body;
    :func:`~newton._src.sim.contacts.capture_scope`) over the same
    ``contact_generation`` array, and no write of that contact stream was noted
    since (:func:`~newton._src.sim.contacts.note_contact_frame_write`) nor
    :meth:`forget_frame` called. Every other build sorts. The device checks each
    kept build's contact generation and raw count against the frame the last
    sort recorded: a writer the host never saw that changed either leaves that
    build empty and latches ``frame_status``. One that kept both but changed the
    shapes of a contact slot, or the contacts' order, goes unseen, and the kept
    build groups its contacts by the stale pairs; its kept indices still lie
    below the live count. Rewrites of points, normals or margins alone are
    exact: pair keys follow the shapes, and a kept build prepares the rest anew.

    Contact frames are identified by ``Contacts.contact_generation``, which every
    collision pass and :meth:`newton.Contacts.clear` advance, together with the
    raw contact count and the contact stream, a token of the generation counter
    array that is never reused (see :meth:`stream_token`). Eager builds always
    sort.
    """

    def __init__(self, model, capacity, phi, *, reuse_frames=False, pool=None):
        self.view = FrictionPatches()
        self.view.phi = phi
        device = model.device
        if pool is None:
            pool = ContactPool(device=device, rows=capacity)
        self.pool = pool
        self._rows_used = None
        self.capacity = pool.ready
        self.reuse_frames = bool(reuse_frames)
        # (capture scope, weak generation array, frame-write tick) of the previous
        # captured build, which a build in the same scope may keep the pairs of.
        self._last_build = None
        self.device = device
        # The pair sort skips the key bits no pair key can set.
        self.key_stride, self.key_bits = _pair_key_layout(model.body_count, model.shape_body.shape[0])
        self.body_world = model.body_world
        self.body_radius = wp.zeros(model.body_count, dtype=float, device=device)
        # The contact frame whose pairs the current slots hold sorted, as
        # (generation, raw count, contact stream); the sentinel generation makes
        # the next build sort. _prepare_frame records a frame the sort follows
        # and compares a reused one. sort_pairs receives its sort flag, which
        # only its detect mode writes and no build launches.
        self.sorted_frame = wp.full(3, GENERATION_SENTINEL, dtype=wp.int64, device=device)
        self.sort_pairs = wp.zeros(1, dtype=wp.int32, device=device)
        # Latched when a build that reuses sorted pairs finds another contact
        # frame than the recorded one and is left empty; only the host clears it.
        self.frame_status = wp.zeros(1, dtype=wp.int32, device=device)
        self._stream_tokens = weakref.WeakKeyDictionary()
        self._next_stream_token = itertools.count(1)
        # Per-contact slots hold the contact pool's ready rows and grow with it.
        self.current = _PatchFrame()
        self.previous = _PatchFrame()
        self._view_rows = pool.rows(
            (
                ContactField("weight", wp.float32),
                ContactField("next_contact", wp.int32, -1),
                ContactField("point", wp.vec3),
                ContactField("spin_arm", wp.float32),
                ContactField("spin_phi", wp.float32),
            ),
            on_rows=self._on_view_rows,
        )
        self._current_rows = pool.rows(self._frame_fields(), on_rows=self._on_current_rows)
        self._previous_rows = pool.rows(
            (*self._frame_fields(stored=True), ContactField("previous_world", wp.int32, -1)),
            on_rows=self._on_previous_rows,
        )
        self.update_geometry(model)
        self.previous_q = wp.zeros(model.body_count, dtype=wp.transform, device=device)
        # Live contact counts on the device. The finish epilogue alone writes the
        # stored frame's count and _build the last built frame's; _prepare
        # publishes the slots this frame visits and its live count. No launch
        # reads a count it writes, so blocks never race on them.
        self.stored = wp.zeros(1, dtype=int, device=device)
        self.built = wp.zeros(1, dtype=int, device=device)
        self.extent = wp.zeros(_SORT_EXTENT + EXTENT_SIZE, dtype=int, device=device)
        self._load_capture_modules()

    def _on_view_rows(self, rows):
        rows.publish(self.view)

    def _on_current_rows(self, rows):
        rows.publish(self.current)

    def _on_previous_rows(self, rows):
        for field in rows.fields:
            if field.name != "previous_world":
                setattr(self.previous, field.name, rows.views[field.name])
        self.previous_world = rows.views["previous_world"]
        # The last rows to grow: the slots' grids follow the new capacity.
        self._set_capacity(rows.rows)

    def _set_capacity(self, capacity):
        """Hold the slots' capacity: the rows the radix fallback sorts and eager grids follow."""
        self.capacity = capacity
        self.build_block_dim = _BUILD_BLOCK_DIM
        self.live_sort = _uses_live_sort(self.key_bits)
        self._grids(capacity)

    def record(self) -> dict:
        """The pair sort's record (``routes.friction.pair_sort``): the live slots' sort, or the radix sort's."""
        if self.live_sort:
            reason = "The pair sort orders only the live slots, so its cost follows the live contacts."
        else:
            reason = (
                f"{self.key_bits}-bit pair keys leave no room for the live sort's {SLOT_BITS} slot bits: the "
                "radix sort orders the whole capacity."
            )
        return {
            "live": bool(self.live_sort),
            "capacity": int(self.capacity),
            "key_bits": int(self.key_bits),
            "reason": reason,
        }

    def _grids(self, rows):
        """Size the grids for ``rows`` slots: the ready rows, or the bound of a capture whose launches bind.

        Every launch strides over its device count, so a grid only has to fill the device.
        """
        blocks = self.device.sm_count if self.device.is_cuda else 1
        self.threads = max(1, min(rows, _LIGHT_BLOCKS_PER_SM * blocks * _BLOCK_THREADS))
        self.build_threads = max(1, min(rows, _BUILD_BLOCKS_PER_SM * blocks * _BLOCK_THREADS))
        self.sort_blocks, self.merges, self.merge_threads = sort_grid(self.device, rows)
        return self

    def _load_capture_modules(self):
        """Load the module of the kernel that captured builds launch first, on a CUDA device.

        With frame reuse only captured builds launch _prepare_frame, so its module
        would otherwise first load inside a graph capture. The kernel has a module
        of its own: loading it compiles nothing else. CPU captures build as eager
        ones do.
        """
        if not self.device.is_cuda:
            return
        if self.reuse_frames:
            _prepare_frame.module.load(self.device)

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
        # A body's radius bounds its shapes: the largest collision radius plus
        # offset from the body origin, zero for a body without shapes.
        shape_body = geometry["shape_body"]
        offset = geometry["shape_transform"][:, None, :3]
        # matmul takes each offset's dot product with BLAS, as np.linalg.norm does,
        # so the extents match per-shape norms bit for bit.
        extent = model.shape_collision_radius.numpy() + np.sqrt(np.matmul(offset, offset.swapaxes(1, 2))).reshape(-1)
        # Only a positive extent raises a radius from zero; NaN never does.
        reaching = (shape_body >= 0) & (extent > 0.0)
        radii = np.zeros(model.body_count, dtype=np.float32)
        np.maximum.at(radii, shape_body[reaching], extent[reaching])
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
        """Make the next build sort its pairs, as it does for a new contact frame.

        The device record stays: a kept build follows a sort of its own capture
        scope, which records its frame.
        """
        self._last_build = None

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
    def _frame_fields(stored=False):
        """Return the per-slot fields of a patch frame, or of the stored frame with ``stored``.

        Every slot starts as the empty record _prepare writes: the sentinel key,
        its own index, and no owner or source. Bounded launches never revisit
        slots that still hold it. A built frame's keys and indices hold a second,
        scratch half for the radix sort. The stored frame keeps what the next
        build reads (_build_pair, _seed_contact; _store_contact writes it): one
        key and index a slot, which no sort visits, and none of the fields only
        a build reads (:data:`_BUILD_FIELDS`).
        """
        halves = 1 if stored else 2
        slack = 0 if stored else int(SORT_CHUNK)
        fields = [
            # One chunk of slack past the scratch half takes the live sort's padded tile stores.
            ContactField("keys", wp.int64, 0x7FFFFFFFFFFFFFFF, per_contact=halves, extra=slack),
            ContactField("indices", wp.int32, None, per_contact=halves, extra=slack),
            ContactField("mu", wp.vec2),
            ContactField("radius", wp.float32),
            ContactField("support_gap_limit", wp.float32),
            ContactField("twist", wp.float32),
        ]
        fields += [
            ContactField(name, wp.vec3)
            for name in (
                "center",
                "normal",
                "displacement",
                "anchor_a",
                "anchor_b",
                "surface_a",
                "surface_b",
                "pressure",
            )
        ]
        fields += [
            ContactField(name, wp.int32, -1 if name in ("owner", "source") else 0)
            for name in (
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
                "pressured",
            )
        ]
        if stored:
            fields = [field for field in fields if field.name not in _BUILD_FIELDS]
        return fields

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
        """Build the current patch frame from ``contacts``.

        With frame reuse, a captured build whose contact frame the previous
        build of its capture scope sorted keeps those pairs instead of sorting
        (see :class:`_FrictionPatchState`).
        """
        bodies = 0 if state.body_q is None else state.body_q.shape[0]
        if bodies + model.shape_body.shape[0] + 2 != self.key_stride:
            raise ValueError(
                "Friction patch pair keys were sized for another model: "
                f"{bodies} body poses and {model.shape_body.shape[0]} shapes"
            )
        self._grids(self.pool.launch_rows())
        source = getattr(contacts, "_rigid_contact_pool", None)
        self._rows_used = source.rows_used if source is self.pool else None
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
        )
        generation = getattr(contacts, "contact_generation", None) if self.reuse_frames else None
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
            elif scope is None:
                # Eager builds sort as without frame reuse: no later build keeps
                # their pairs, so their frame needs no record.
                self._prepare_and_sort(prepare_inputs)
            else:
                # The prepare records the frame that the sort then orders.
                self._launch_prepare_frame(prepare_inputs, generation, stream, _FRAME_SORT)
                self._order_pairs()
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
        self._launch_prepare(_prepare, prepare_inputs, [self.extent])
        self._order_pairs()

    def _launch_prepare(self, kernel, inputs, outputs):
        """Prepare the slots, a pass over every slot an earlier frame used, then follow its published extent.

        The pass visits the live slots and those of the stored and the last built
        frame, all among the rows the contact stream ever used (``rows_used``,
        when the slots share the stream's pool). It publishes the live count and
        the sort's work (:func:`_publish_extent`); in a capture whose launches
        bind, the launches after it follow them through a new updater.
        """
        used = self._rows_used if self._rows_used is not None else self.pool.ready_rows
        self.pool.launch(kernel, self.threads, inputs=inputs, outputs=outputs, extent=used, clamp=True)
        self.pool.begin_updates()

    def _launch_build(self, model, state):
        """Build the pairs of the prepared, sorted slots."""
        # Warp compiles this module once more for _build's block size; the
        # variants differ only in the tile block size, which no _build
        # arithmetic reads.
        self.pool.launch(
            _build,
            self.build_threads,
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
            # The slots to visit, not the live ones: thread 0 publishes the live count, also when it drops to zero.
            extent=self.extent[0:1],
            clamp=True,
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
    ):
        """Return the inputs that :func:`_prepare` and :func:`_prepare_frame` share, with :meth:`build`'s gates."""
        inputs = _PatchContacts()
        inputs.shape0, inputs.shape1 = contacts.rigid_contact_shape0, contacts.rigid_contact_shape1
        inputs.point0, inputs.point1 = contacts.rigid_contact_point0, contacts.rigid_contact_point1
        inputs.normal = contacts.rigid_contact_normal
        inputs.margin0, inputs.margin1 = contacts.rigid_contact_margin0, contacts.rigid_contact_margin1
        inputs.shape_body, inputs.shape_mu, inputs.shape_gap = (
            model.shape_body,
            model.shape_material_mu,
            model.shape_gap,
        )
        inputs.body_radius, inputs.q = self.body_radius, state.body_q
        inputs.body_to_articulation, inputs.is_free_rigid = body_to_articulation, is_free_rigid
        inputs.contact_gap_gate = contact_gap_gate
        inputs.same_articulation_gap_gate = same_articulation_gap_gate
        inputs.articulation_pair_gap_gate = articulation_pair_gap_gate
        inputs.friction_gap = friction_gap
        inputs.key_stride = self.key_stride
        return [
            contacts.rigid_contact_count,
            inputs,
            self.stored,
            self.built,
            self.threads,
            self.current,
            self.view,
        ]

    def _launch_prepare_frame(self, prepare_inputs, generation, stream, mode):
        """Prepare a frame-reusing build in ``mode`` (``_FRAME_SORT`` or ``_FRAME_KEEP``)."""
        self._launch_prepare(
            _prepare_frame,
            [*prepare_inputs, generation, stream, int(mode), self.sorted_frame],
            [self.extent, self.sort_pairs, self.frame_status],
        )

    def _capture_scope(self):
        """Return the capture scope that this device's current stream records into, ``None`` when eager."""
        return capture_scope(self.device)

    def _order_pairs(self):
        """Sort the prepared slots by pair key, the work that only a new contact frame needs.

        A kept build skips exactly this, so work added here must leave outputs that
        stay valid while later builds reuse the frame.
        """
        if self.live_sort:
            self._sort_live()
        else:
            # Empty slots hold INT64_MAX, whose low key_bits are all ones: no pair
            # key exceeds them, and the stable sort keeps live slots ahead of a tie.
            # The radix sort covers today's rows: a capture of it is recaptured after growth.
            self.pool.note_unbound()
            wp.utils.radix_sort_pairs(self.current.keys, self.current.indices, self.capacity, end_bit=self.key_bits)

    def _sort_live(self):
        """Stably sort the live slots by pair key and leave every other slot untouched.

        A stable sort of slots [0, live) from indices c is the unique order by
        (key, c), which the full radix sort produces; slots past live already
        hold the empty record that sort leaves there. The launches follow the
        sort's work that the prepare published: a capture launches the chunk
        sort over the live chunks and only the merges the live slots need. The
        scratch half starts at the ready rows, which a capture's updater
        rewrites on every replay, and the kernels address the whole
        reservation, of which the live slots, the scratch half and one chunk of
        slack are mapped.
        """
        launch_live_sort(
            self.pool,
            self._current_rows.reserved["keys"],
            self._current_rows.reserved["indices"],
            self.extent[_SORT_EXTENT:],
            self.capacity,
            blocks=self.sort_blocks,
            merges=self.merges,
            merge_threads=self.merge_threads,
            slot_keys=True,
        )

    def link(self, world, slot, path, slots_needed, routes):
        """Link the normal rows of each patch and share its Coulomb budget on the enabled routes."""
        self._grids(self.pool.launch_rows())
        self.pool.launch(
            link_patch_routes,
            self.threads,
            inputs=[self.extent, self.view, world, slot, path, slots_needed, routes, self.threads],
            extent=self.extent[1:2],
            clamp=True,
        )

    def finish(self, state_in, state_out, com, world, slot, path, slots_needed, routes, dt):
        """Place the next anchors at the solved centres of pressure and release sliding anchors on the enabled routes,
        then store the frame."""
        self._finish(
            state_in.body_q, state_out.body_q, state_out.body_qd, com, world, slot, path, slots_needed, routes, dt
        )

    def _finish(self, q, q_out, qd_out, com, world, slot, path, slots_needed, routes, dt):
        self._grids(self.pool.launch_rows())
        self.pool.launch(
            _finish_and_store,
            self.threads,
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
                self.view,
                dt,
                self.threads,
                self.current,
                self.previous,
                self.previous_world,
                self.previous_q,
            ],
            outputs=[self.stored],
            extent=self.extent[0:1],
            clamp=True,
        )

    def reset(self, world_mask):
        """Drop the stored anchors of the masked worlds (all worlds without a mask)."""
        self._grids(self.pool.launch_rows())
        self.pool.launch(
            _reset_history,
            self.threads,
            inputs=[
                world_mask,
                self.stored,
                self.threads,
                self.previous_world,
                self.previous.valid,
                self.previous.pressured,
            ],
            extent=self.stored,
            clamp=True,
        )


@wp.func
def patch_normal_load(parents: wp.array2d[int], impulses: wp.array2d[float], world: int, first: int):
    """Sum a region's live normal impulses; -1 terminates ordinary point contacts.

    A contact row's parent field links the next normal in its region (a circular
    list). Friction rows still point to their adjacent normal row. Other row
    families retain their original parent semantics. The Warp form of the native
    ``RING_LOAD`` (:mod:`.device_preludes`): the same order, the same hop guard.
    """
    load = impulses[world, first]
    row = parents[world, first]
    hop = int(0)
    while row >= 0 and row != first and hop < parents.shape[1]:
        load += impulses[world, row]
        row = parents[world, row]
        hop += 1
    return load


@wp.struct
class _PatchRoutes:
    """Row arrays of the contact routes that hold patch rows; ``enabled`` gates each route."""

    enabled: wp.vec3i
    parents0: wp.array2d[int]
    parents1: wp.array2d[int]
    parents2: wp.array2d[int]
    mu0: wp.array2d[float]
    mu1: wp.array2d[float]
    mu2: wp.array2d[float]
    impulses0: wp.array2d[float]
    impulses1: wp.array2d[float]
    impulses2: wp.array2d[float]
    first0: wp.array[int]
    first1: wp.array[int]
    first2: wp.array[int]


def patch_routes(rows):
    """Pack ``(route, parents, mu, impulses, first)`` rows into one launch argument.

    Routes absent from ``rows`` stay disabled, and no launch reads their arrays.
    """
    routes = _PatchRoutes()
    enabled = [0, 0, 0]
    for route, parents, mu, impulses, first in rows:
        enabled[route] = 1
        setattr(routes, f"parents{route}", parents)
        setattr(routes, f"mu{route}", mu)
        setattr(routes, f"impulses{route}", impulses)
        setattr(routes, f"first{route}", first)
    routes.enabled = wp.vec3i(*enabled)
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
def _route_first(routes: _PatchRoutes, route: int):
    if route == 1:
        return routes.first1
    if route == 2:
        return routes.first2
    return routes.first0


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
    route: int,
    parents: wp.array2d[int],
    first: wp.array[int],
):
    """Link contact ``c``'s normal row into its region's ring: its parent is the next member's row on the route.

    A region has one anchor, so its friction rows' coefficients are whole: the pooled normal load of the ring bounds
    them once.
    """
    row = patches.next_contact[c]
    if row < 0:
        return
    next_slot = slot[c]
    while row != c:
        if slot[row] >= 0 and path[row] == route and world[row] == world[c]:
            next_slot = slot[row]
            break
        row = patches.next_contact[row]
    parents[first[world[c]], slot[c]] = next_slot


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
            first = _route_first(routes, route)
            _link_contact(c, patches, world, slot, path, route, parents, first)


@wp.func
def _point_velocity(
    q: wp.array[wp.transform], qd: wp.array[wp.spatial_vector], com: wp.array[wp.vec3], body: int, anchor: wp.vec3
):
    velocity = wp.vec3(0.0)
    if body >= 0:
        velocity = point_velocity(qd[body], wp.transform_vector(q[body], anchor - com[body]))
    return velocity


@wp.func
def _finish_contact(
    c: int,
    frame: _PatchFrame,
    patches: FrictionPatches,
    q: wp.array[wp.transform],
    q_out: wp.array[wp.transform],
    qd_out: wp.array[wp.spatial_vector],
    com: wp.array[wp.vec3],
    world: wp.array[int],
    slot: wp.array[int],
    path: wp.array[int],
    slots_needed: wp.array[int],
    parents: wp.array2d[int],
    mu: wp.array2d[float],
    impulses: wp.array2d[float],
    dt: float,
    first: wp.array[int],
):
    """Record the region's centre of pressure and release saturated anchors with tangential motion.

    The tolerance is relative to geometry and timestep. Merely resting at the
    Coulomb limit does not prove sliding, so it must not erase static history.
    """
    w = first[world[c]]
    s = slot[c]
    load = patch_normal_load(parents, impulses, w, s)
    if load <= 0.0:
        # A speculative region cannot accumulate a spring while unsupported
        # and apply that error when it later lands (common on curved meshes).
        frame.valid[c] = 0
        return
    if slots_needed[c] < 3:
        # Friction was filtered, but its normal rows still prove support.
        return
    n = frame.normal[c]
    t0, t1 = contact_tangent_basis(n)
    if frame.flipped[c] != 0:
        t0, t1 = contact_tangent_basis(-n)
    tangent = impulses[w, s + 1] * t0 + impulses[w, s + 2] * t1
    if frame.flipped[c] != 0:
        tangent = -tangent
    a = frame.body_a[c]
    b = frame.body_b[c]
    # The moment of the normal impulses is determinate even where they are not: the ring's centre of pressure, where
    # the next build places the anchor.
    moment = wp.vec3(0.0)
    moment_load = float(0.0)
    member = c
    hop = int(0)
    while hop < parents.shape[1]:
        if slot[member] >= 0 and path[member] == path[c] and world[member] == world[c]:
            weight = wp.max(impulses[w, slot[member]], 0.0)
            moment += weight * frame.center[member]
            moment_load += weight
        member = patches.next_contact[member]
        hop += 1
        if member == c or member < 0:
            break
    if moment_load > 0.0:
        frame.pressure[c] = _local_point(q, a, moment / moment_load)
        frame.pressured[c] = 1
    radius = wp.max(mu[w, s + 1] * load, 0.0)
    speed = _point_velocity(q_out, qd_out, com, a, frame.anchor_a[c]) - _point_velocity(
        q_out, qd_out, com, b, frame.anchor_b[c]
    )
    # Check opposing motion to distinguish slipping from impending slip. The anchor saturates on the ball of the pair
    # and the spin row, not on the pair's disk; tangential slip releases it, and opposing spin its twist alone.
    spin = float(0.0)
    if slots_needed[c] > 3 and mu[w, s + 3] > 0.0:
        spin = impulses[w, s + 3] / mu[w, s + 3]
    tangent_load = float(0.0)
    if mu[w, s + 1] > 0.0:
        tangent_load = wp.length(tangent) / mu[w, s + 1]
    used = wp.sqrt(tangent_load * tangent_load + spin * spin)
    if radius > 0.0 and used >= (1.0 - 1.0e-5) * load:
        if wp.dot(speed, tangent) < -wp.length(tangent) * (1.0e-5 * frame.radius[c] / dt):
            frame.valid[c] = 0
        if spin != 0.0:
            rate = float(0.0)
            if a >= 0:
                rate += wp.dot(wp.spatial_bottom(qd_out[a]), frame.normal[c])
            if b >= 0:
                rate -= wp.dot(wp.spatial_bottom(qd_out[b]), frame.normal[c])
            if rate * spin < 0.0:
                # The spin slips: the twist at the step's end is zero. The next build adds this step's turn from the
                # stored pose, so the stored twist cancels it, as a released anchor carries no slip.
                frame.twist[c] = _pose_twist(q_out, q, b, frame.normal[c]) - _pose_twist(q_out, q, a, frame.normal[c])


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
    patches: FrictionPatches,
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
                        c,
                        frame,
                        patches,
                        q,
                        q_out,
                        qd_out,
                        com,
                        world,
                        slot,
                        path,
                        slots_needed,
                        parents,
                        mu,
                        impulses,
                        dt,
                        _route_first(routes, route),
                    )
        _store_contact(c, q, body_world, frame, prev, previous_world, previous_q)


@wp.kernel(enable_backward=False)
def _reset_history(
    world_mask: wp.array[wp.bool],
    stored: wp.array[int],
    threads: int,
    prev_world: wp.array[int],
    prev_valid: wp.array[int],
    prev_pressured: wp.array[int],
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
        prev_pressured[c] = 0
