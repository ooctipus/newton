# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Global GPU contact reduction using hashtable-based tracking.

This module provides a global contact reduction system that uses a hashtable
to track the best contacts across shape pairs, normal bins, and scan directions.
Unlike the shared-memory based approach in ``contact_reduction.py``, this works
across the entire GPU without block-level synchronization constraints.

**When to Use:**

- Used for mesh-mesh (SDF) collisions where contacts span multiple GPU blocks
- The shared-memory approach in ``contact_reduction.py`` is used for mesh-plane
  and mesh-convex where all contacts for a pair fit in one block

**Contact Reduction Strategy:**

The same three-strategy approach as the shared-memory reduction in
``contact_reduction.py``.  Slot counts depend on the configuration in that
module (``NUM_NORMAL_BINS``, ``NUM_SPATIAL_DIRECTIONS``,
``NUM_VOXEL_DEPTH_SLOTS``, ``MAX_CONTACTS_PER_PAIR``).

1. **Spatial Extreme Slots** (``NUM_SPATIAL_DIRECTIONS`` per normal bin)
   - Builds support polygon boundary for stable stacking
   - Only contacts with depth < beta participate

2. **Per-Bin Max-Depth Slots** (1 per normal bin)
   - Tracks deepest contact per normal direction
   - Critical for gear-like contacts with varied normal orientations
   - Participates unconditionally (not gated by beta)

3. **Voxel-Based Depth Slots** (``NUM_VOXEL_DEPTH_SLOTS`` total per pair)
   - Tracks deepest contact per mesh-local voxel region
   - Ensures early detection of contacts at mesh centers
   - Prevents sudden contact jumps between frames

**Under speculation** a pass predicts the window until the next one, and the slots keep what supports the pair over
it. A candidate stands by its closest approach over the window: its gap moves linearly, so that is its depth now or its
depth less its closing speed (rotation included) times the window, whichever is deeper. Within a normal bin the normal
displacement of a rigid motion is affine in a point's tangent position, so the candidates whose (tangent position,
standing) lie on the lower convex hull of the bin's candidates imply the others' non-penetration for any motion; a
direction slot keeps the support of that hull in its direction, scoring a candidate by its projection less its
standing (the hull's facets up to 45 degrees, steeper than any rotation a window makes), among the candidates that touch
by the window's end. The max-depth and voxel slots keep the deepest by standing. In the buffered reduction the
candidates touching now outrank the ones that only reach the window's end for the extremes: those rows bear the pair's
load from the first substep, and under a soft contact a load on fewer rows sinks deeper, while a reaching row the slots
miss costs at most what the window's reach brings. Without speculation the standing is the depth and the scores are
the projections.

**Implementation Details:**

- Contacts stored in global buffer (struct of arrays: position_depth, normal, shape_pairs)
- Hashtable key: ``(shape_a, shape_b, bin_id)`` where ``bin_id`` is
  ``0..NUM_NORMAL_BINS-1`` for normal bins and higher indices for voxel groups
- Each normal bin entry has ``NUM_SPATIAL_DIRECTIONS + 1`` value slots
- Voxels are grouped by ``NUM_SPATIAL_DIRECTIONS + 1``:
  ``bin_id = NUM_NORMAL_BINS + (voxel_idx // group_size)``
- A slot is won by the full score, ties broken by the candidate's fingerprint
  (see the reduction slots section), never by the order threads arrive in

See Also:
    ``contact_reduction.py`` for shared utility functions, configuration
    constants, and detailed algorithm documentation.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import warp as wp

from newton._src.geometry.hashtable import (
    HASHTABLE_EMPTY_KEY,
    HashTable,
    hashtable_find,
    hashtable_find_or_insert,
)

from ..utils.heightfield import HeightfieldData, get_triangle_shape_from_heightfield
from .collision_convex import mpr_inflate
from .collision_core import (
    create_compute_face_contacts,
    create_compute_gjk_mpr_contacts,
    triangle_face_case,
    triangle_facet_case,
)
from .contact_data import ContactData, compute_contact_approach_speed
from .contact_reduction import (
    MAX_CONTACTS_PER_PAIR,
    NUM_NORMAL_BINS,
    NUM_SPATIAL_DIRECTIONS,
    NUM_VOXEL_DEPTH_SLOTS,
    compute_voxel_index,
    float_flip,
    get_slot,
    get_spatial_direction_2d,
    project_point_to_plane,
)
from .heightfield_features import HeightfieldFeatureContext, seam_surface_witness
from .heightfield_primitive_contact import (
    ANALYTIC_PRIMITIVE_QUERIES,
    analytic_convex_default,
    analytic_primitives_default,
)
from .mesh_topology import (
    MeshTopologyData,
    create_mesh_capsule_contacts,
    create_mesh_sphere_contact,
    create_surface_penetration_refiner,
    mesh_contact_filter,
    mesh_deepest_once,
    mesh_facet_interior,
    mesh_manifold_triangle,
    mesh_surface_context,
    mesh_surface_planes,
    mesh_triangle_corners,
    surface_query_filter,
    triangle_plane_clears,
    triangle_shape_from_corners,
)
from .support_function import (
    AcceleratedSupportMapDataProvider,
    SupportMapDataProvider,
    create_triangle_prism_penetration_refiner,
    extract_shape_data,
    support_map,
    support_map_accelerated_inline,
    support_map_heightfield_box,
    support_map_heightfield_capsule,
    support_map_heightfield_cylinder,
    support_map_heightfield_sphere,
)
from .types import GeoType

# Fixed beta threshold for contact reduction - small positive value to avoid flickering
# from numerical noise while effectively selecting only near-penetrating contacts for
# the support polygon.
BETA_THRESHOLD = 0.0001  # 0.1mm

VALUES_PER_KEY = NUM_SPATIAL_DIRECTIONS + 1
# Slots are stored entry-major: each key owns one row of VALUES_STRIDE slots,
# of which the first VALUES_PER_KEY hold its winners. The last slot only pads
# the row: no writer addresses it, so it stays zero. Eight uint64 slots make a
# 64-byte row, so reading or retiring a key touches two sectors instead of one
# sector in each of seven slot planes ``capacity`` entries apart.
VALUES_STRIDE = 8
assert VALUES_STRIDE >= VALUES_PER_KEY
# Normal bins occupy 0..19 and 15 voxel groups occupy 20..34. Reserve
# bin 35 for a pair-level predictive manifold without changing the regular
# seven-slot value layout: six fingerprint-sharded clearance contacts and one
# earliest time-of-impact guard.
PREDICTIVE_BIN_ID = NUM_NORMAL_BINS + (NUM_VOXEL_DEPTH_SLOTS + VALUES_PER_KEY - 1) // VALUES_PER_KEY
_GLOBAL_TOTAL_SLOTS = NUM_NORMAL_BINS * VALUES_PER_KEY + NUM_VOXEL_DEPTH_SLOTS + VALUES_PER_KEY
assert _GLOBAL_TOTAL_SLOTS <= MAX_CONTACTS_PER_PAIR
PAIRS_PER_KEY = VALUES_PER_KEY * (VALUES_PER_KEY - 1) // 2
EXPORT_REDUCED_CONTACTS_BLOCK_DIM = 32
EXPORT_REDUCED_CONTACTS_THREAD_BUDGET_MULTIPLIER = 4

# Preserve slot-level parallelism for small active sets; larger sets
# assign one thread to each entry's value row.
CLEAR_ACTIVE_ENTRY_PARALLEL_THRESHOLD = 1024

# Open-addressed linear probing gets expensive at high load and failed inserts
# scan the whole table.
HASHTABLE_WARN_LOAD_PERCENT = 80

# Vector type for tracking exported contact IDs (used in export kernels)
exported_ids_vec_type = wp.types.vector(length=VALUES_PER_KEY, dtype=wp.int32)
replaced_values_vec_type = wp.types.vector(length=VALUES_PER_KEY + 1, dtype=wp.uint64)


@wp.func
def is_contact_already_exported(
    contact_id: int,
    exported_ids: wp.types.vector(length=VALUES_PER_KEY, dtype=wp.int32),
    num_exported: int,
) -> bool:
    """Check if a contact_id is already in the exported list.

    Args:
        contact_id: The contact ID to check
        exported_ids: Vector of already exported contact IDs
        num_exported: Number of valid entries in exported_ids

    Returns:
        True if contact_id is already in the list, False otherwise
    """
    j = int(0)
    while j < num_exported:
        if exported_ids[j] == contact_id:
            return True
        j = j + 1
    return False


@wp.func
def _floats_are_near_ulps(a: float, b: float) -> bool:
    """Return whether two floats are close in ULPs and absolute magnitude."""
    if wp.abs(a - b) > 1.0e-8:
        return False
    a_bits = float_flip(a)
    b_bits = float_flip(b)
    if a_bits > b_bits:
        return a_bits - b_bits <= wp.uint32(16)
    return b_bits - a_bits <= wp.uint32(16)


@wp.func
def _contacts_are_numerically_equivalent(
    contact_a: int,
    contact_b: int,
    position_depth: wp.array[wp.vec4],
    normal: wp.array[wp.vec2],
) -> bool:
    """Return whether two buffered contacts differ only by float rounding."""
    pd_a = position_depth[contact_a]
    pd_b = position_depth[contact_b]
    if not _floats_are_near_ulps(pd_a[0], pd_b[0]):
        return False
    if not _floats_are_near_ulps(pd_a[1], pd_b[1]):
        return False
    if not _floats_are_near_ulps(pd_a[2], pd_b[2]):
        return False
    if not _floats_are_near_ulps(pd_a[3], pd_b[3]):
        return False

    n_a = normal[contact_a]
    n_b = normal[contact_b]
    if not _floats_are_near_ulps(n_a[0], n_b[0]):
        return False
    return _floats_are_near_ulps(n_a[1], n_b[1])


@wp.func
def compute_effective_radius(shape_type: int, shape_scale: wp.vec4) -> float:
    """Compute effective radius for a shape based on its type.

    For shapes that can be represented as Minkowski sums with a sphere (sphere, capsule),
    the effective radius is the sphere radius component. For other shapes, it's 0.

    Args:
        shape_type: The GeoType of the shape
        shape_scale: Shape scale data (vec4, xyz are scale components)

    Returns:
        Effective radius (scale[0] for sphere/capsule, 0 otherwise)
    """
    if shape_type == GeoType.SPHERE or shape_type == GeoType.CAPSULE:
        return shape_scale[0]
    return 0.0


# =============================================================================
# Reduction slots
# =============================================================================
# Each hashtable entry owns one row of VALUES_STRIDE slots, entry-major:
# [entry0_slot0, ..., entry0_slot7, entry1_slot0, ..., entry1_slot7, ...]
# Slots 0 to VALUES_PER_KEY - 1 hold winners; slot 7 pads the row to 64
# bytes and stays zero. Every access addresses a slot through
# reduction_value_index (device) or reduction_value_index_np (host).
#
# A slot holds one uint64 key while candidates compete for it:
#
#   key = priority << 63 | float_flip(score) << 31 | fingerprint
#
# (priority: 1 for an inner candidate of a prioritized directional slot, else
# 0; fingerprint: the candidate's geometric feature, unique among the
# candidates of its shape pair, below 2**31). A candidate wins a slot by an
# atomic max of its key: by its full score (and priority), ties broken by its
# fingerprint, so a slot's holder after a pass is decided by the geometry
# alone, never by the order in which threads arrive.
#
# The key names no buffer row. After every candidate of a pass has competed,
# a resolve pass (:func:`resolve_reduced_contacts`) visits the buffered
# candidates: the one whose fingerprint a slot's key holds is its winner (the
# only candidate of its pair with that fingerprint) and replaces the key by its
# contact ID (a value whose score field is zero, which no key has). Every winner
# is buffered: it raised its slot when it competed, and a candidate that raises
# a slot is buffered.

SLOT_FINGERPRINT_BITS = 31
SLOT_FINGERPRINT_MASK = wp.constant(wp.uint64((1 << 31) - 1))


@wp.func
def reduction_value_index(entry_idx: int, slot_id: int) -> int:
    """Return the index of one entry's slot in the entry-major slot rows.

    Args:
        entry_idx: Entry index from hashtable_find_or_insert()
        slot_id: Winner slot within the entry's row (0 to ``VALUES_PER_KEY - 1``),
            or the padding slot ``VALUES_STRIDE - 1`` when clearing the whole row
    """
    return entry_idx * wp.static(VALUES_STRIDE) + slot_id


def reduction_value_index_np(entry_idx, slot_id):
    """Host counterpart of :func:`reduction_value_index` for integers and NumPy index arrays."""
    return entry_idx * VALUES_STRIDE + slot_id


@wp.func
def contact_key(score: float, fingerprint: int) -> wp.uint64:
    """The key of a candidate scoring ``score`` (higher wins) with ``fingerprint`` in an unprioritized slot."""
    return (wp.uint64(float_flip(score)) << wp.uint64(31)) | (wp.uint64(fingerprint) & SLOT_FINGERPRINT_MASK)


@wp.func
def spatial_contact_key(score: float, is_inner: bool, fingerprint: int) -> wp.uint64:
    """The key of a candidate in a prioritized directional slot: inner candidates first, then the full score."""
    key = contact_key(score, fingerprint)
    if is_inner:
        key = key | (wp.uint64(1) << wp.uint64(63))
    return key


@wp.func
def slot_contact_id(value: wp.uint64) -> int:
    """The contact ID a resolved slot holds; zero for an empty slot or a key no buffered candidate resolved."""
    if (value >> wp.uint64(31)) != wp.uint64(0):
        return 0
    return int(value)


@wp.func
def competed_voxel_word(voxel_entry_idx: int, predictive: bool) -> int:
    """A buffered candidate's voxel word (:attr:`GlobalContactReducerData.contact_slots`): the voxel entry in which it
    raised a slot (-1: none), or ``-3 - entry`` (``-2`` for none) when it also raised its pair's predictive slots."""
    if predictive:
        return -3 - voxel_entry_idx
    return voxel_entry_idx


@wp.func
def competed_voxel_entry(word: int) -> int:
    """The voxel entry a voxel word names (-1: none)."""
    if word < -1:
        return -3 - word
    return word


@wp.func
def reduction_update_slot(entry_idx: int, slot_id: int, key: wp.uint64, values: wp.array[wp.uint64]) -> bool:
    """Raise one of an entry's slots to ``key`` where it is higher (an atomic max, its result not awaited); returns
    whether the slot was open to ``key`` when read (every raise among them).

    Use this after hashtable_find_or_insert() to offer several slots of the same entry without repeated hash lookups.
    """
    value_idx = reduction_value_index(entry_idx, slot_id)
    # Check before the atomic to reduce contention: a slot only rises during a pass.
    if values[value_idx] < key:
        wp.atomic_max(values, value_idx, key)
        return True
    return False


@wp.func
def reduction_try_update_slot(entry_idx: int, slot_id: int, key: wp.uint64, values: wp.array[wp.uint64]) -> bool:
    """Raise one of an entry's slots to ``key`` where it is higher; returns whether this offer raised it."""
    value_idx = reduction_value_index(entry_idx, slot_id)
    if values[value_idx] >= key:
        return False
    return wp.atomic_max(values, value_idx, key) < key


@wp.func
def reduction_slot_open(entry_idx: int, slot_id: int, key: wp.uint64, values: wp.array[wp.uint64]) -> bool:
    """Whether a candidate of ``key`` would still raise one of an entry's slots (a probe before buffering it). A slot only
    rises during a pass, so a candidate a stale read turns away stays beaten."""
    return values[reduction_value_index(entry_idx, slot_id)] < key


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    wp::uint64 value;
    asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(value) : "l"(values.data + idx) : "memory");
    return value;
#else
    return values.data[idx];
#endif
"""
)
def reduction_slot_key(values: wp.array[wp.uint64], idx: int) -> wp.uint64:
    """Slot ``idx``'s key as the atomics hold it now: a coherent load, never a cached copy older than a raise."""
    ...


@wp.func
def reduction_slot_holds(value_idx: int, key: wp.uint64, values: wp.array[wp.uint64]) -> bool:
    """Whether slot ``value_idx`` holds ``key`` now (:func:`reduction_slot_key`)."""
    return reduction_slot_key(values, value_idx) == key


@wp.func
def reduction_resolve_slot(value_idx: int, fingerprint: int, contact_id: int, values: wp.array[wp.uint64]) -> bool:
    """Give slot ``value_idx`` the contact ID of the buffered candidate ``contact_id`` if the slot's key is its
    (its fingerprint, which no other candidate of its pair has); a resolved or empty slot is left alone. Returns
    whether the slot is the candidate's, so a repeated resolve names the same slots."""
    value = values[value_idx]
    if value == wp.uint64(contact_id):
        return True
    if (value >> wp.uint64(31)) != wp.uint64(0) and (value & SLOT_FINGERPRINT_MASK) == (
        wp.uint64(fingerprint) & SLOT_FINGERPRINT_MASK
    ):
        values[value_idx] = wp.uint64(contact_id)
        return True
    return False


@wp.func
def reduction_insert_slot(
    key: wp.uint64,
    slot_id: int,
    value: wp.uint64,
    keys: wp.array[wp.uint64],
    values: wp.array[wp.uint64],
    active_slots: wp.array[wp.int32],
) -> bool:
    """Insert a hashtable key and raise one of its slots to ``value``.

    Convenience function that combines hashtable_find_or_insert() and reduction_update_slot().

    Returns:
        True if insertion/update succeeded, False if the table is full
    """
    entry_idx = hashtable_find_or_insert(key, keys, active_slots)
    if entry_idx < 0:
        return False
    reduction_update_slot(entry_idx, slot_id, value, values)
    return True


# =============================================================================
# Contact key packing
# =============================================================================

# Bit layout for hashtable key (63 bits used, bit 63 kept 0 for signed/unsigned safety):
# Key is (shape_a, shape_b, bin_id) - NO slot_id (slots are handled via values_per_key)
# - Bits 0-26:   shape_a (27 bits, up to ~134M shapes)
# - Bits 27-54:  shape_b (28 bits, up to ~268M shapes)
# - Bits 55-62:  bin_id (8 bits, 0-255, supports normal bins + voxel groups)
# - Bit 63:      unused (kept 0 for signed/unsigned compatibility)
# Total: 63 bits used

SHAPE_A_BITS = wp.constant(wp.uint64(27))
SHAPE_A_MASK = wp.constant(wp.uint64((1 << 27) - 1))
SHAPE_B_BITS = wp.constant(wp.uint64(28))
SHAPE_B_MASK = wp.constant(wp.uint64((1 << 28) - 1))
BIN_BITS = wp.constant(wp.uint64(8))
BIN_MASK = wp.constant(wp.uint64((1 << 8) - 1))


@wp.func
def make_contact_key(shape_a: int, shape_b: int, bin_id: int) -> wp.uint64:
    """Create a hashtable key from shape pair and bin.

    Args:
        shape_a: First shape index
        shape_b: Second shape index
        bin_id: Bin index (``0..NUM_NORMAL_BINS-1`` for normal bins, higher for voxel groups)

    Returns:
        64-bit key for hashtable lookup (only 63 bits used)
    """
    key = wp.uint64(shape_a) & SHAPE_A_MASK
    key = key | ((wp.uint64(shape_b) & SHAPE_B_MASK) << SHAPE_A_BITS)
    # bin_id goes at bits 55-62 (after 27 + 28 = 55 bits for shape IDs)
    key = key | ((wp.uint64(bin_id) & BIN_MASK) << wp.uint64(55))
    return key


@wp.func
def encode_oct(n: wp.vec3) -> wp.vec2:
    """Encode a unit normal into octahedral 2D representation.

    Projects the unit vector onto an octahedron and flattens to 2D.
    Near-uniform precision, stable numerics, no trig needed.
    """
    l1 = wp.abs(n[0]) + wp.abs(n[1]) + wp.abs(n[2])
    if l1 < 1.0e-20:
        return wp.vec2(0.0, 0.0)
    inv_l1 = 1.0 / l1
    ox = n[0] * inv_l1
    oy = n[1] * inv_l1
    oz = n[2] * inv_l1

    if oz < 0.0:
        sign_x = 1.0
        if ox < 0.0:
            sign_x = -1.0
        sign_y = 1.0
        if oy < 0.0:
            sign_y = -1.0
        new_x = (1.0 - wp.abs(oy)) * sign_x
        new_y = (1.0 - wp.abs(ox)) * sign_y
        ox = new_x
        oy = new_y

    return wp.vec2(ox, oy)


@wp.func
def decode_oct(e: wp.vec2) -> wp.vec3:
    """Decode octahedral 2D representation back to a unit normal.

    Inverse of encode_oct.  Lossless within float precision.
    """
    nz = 1.0 - wp.abs(e[0]) - wp.abs(e[1])
    nx = e[0]
    ny = e[1]

    if nz < 0.0:
        sign_x = 1.0
        if nx < 0.0:
            sign_x = -1.0
        sign_y = 1.0
        if ny < 0.0:
            sign_y = -1.0
        new_x = (1.0 - wp.abs(ny)) * sign_x
        new_y = (1.0 - wp.abs(nx)) * sign_y
        nx = new_x
        ny = new_y

    return wp.normalize(wp.vec3(nx, ny, nz))


@wp.struct
class GlobalContactReducerData:
    """Struct for passing GlobalContactReducer arrays to kernels.

    This struct bundles all the arrays needed for global contact reduction
    so they can be passed as a single argument to warp kernels/functions.
    """

    # Contact buffer arrays
    position_depth: wp.array[wp.vec4]
    normal: wp.array[wp.vec2]  # Octahedral-encoded unit normal (see encode_oct/decode_oct)
    shape_pairs: wp.array[wp.vec2i]
    contact_count: wp.array[wp.int32]
    reclaimed_contact_bits: wp.array[wp.uint32]
    reclaimed_contact_count: wp.array[wp.int32]
    capacity: int

    # Fingerprint per contact: its geometric feature (triangle/edge/vertex index), unique among its pair's
    # candidates. It breaks a slot's score ties (the key's low bits), so the winner never depends on thread order.
    contact_fingerprints: wp.array[wp.int32]
    # The entries in which each buffered candidate raised a slot, which the resolve pass reads: its normal bin's (-1:
    # none) and a voxel word, its voxel group's entry (-1: none) marked when it also raised its pair's predictive
    # slots (competed_voxel_word); the predictive entry itself is found by its key. The resolve then writes the
    # candidate's home entry into the first word (-1: it holds no slot): the first entry, in the order normal bin,
    # voxel, predictive, in which it holds a slot. The export writes a winner only from its home entry.
    contact_slots: wp.array[wp.vec2i]

    # Optional hydroelastic data
    # contact_area: force-bearing area when penetrating, full face area when speculative
    contact_area: wp.array[wp.float32]
    # contact_pressure: current pressure evaluated once during contact generation
    contact_pressure: wp.array[wp.float32]

    # Cached normal-bin hashtable entry index per contact
    contact_nbin_entry: wp.array[wp.int32]

    # Aggregate force per hashtable entry (indexed by ht_capacity)
    # Used for hydroelastic stiffness calculation: c_stiffness = |agg_force| / total_depth
    # Accumulates sum(area * pressure * normal) for all penetrating contacts per entry
    agg_force: wp.array[wp.vec3]

    # Aggregate geometric depth-volume per hashtable entry: sum(area * |depth| * normal)
    # for all penetrating contacts. Unlike ``agg_force`` this is independent of the
    # pressure law, so its magnitude is used as the pressure-law-agnostic
    # direction-reliability gate for normal matching / anchor placement.
    agg_depth_volume: wp.array[wp.vec3]

    # Weighted position sum per hashtable entry (for anchor contact computation)
    # Accumulates sum(area * pressure * position) for penetrating contacts
    # Divide by weight_sum to get center of pressure (anchor position)
    weighted_pos_sum: wp.array[wp.vec3]

    # Weight sum per hashtable entry (for anchor contact normalization)
    # Accumulates sum(area * pressure) for penetrating contacts
    weight_sum: wp.array[wp.float32]

    # Total depth of reduced (winning) contacts per normal bin entry.
    total_depth_reduced: wp.array[wp.float32]

    # Total depth-weighted normal of reduced (winning) contacts per normal bin entry.
    total_normal_reduced: wp.array[wp.vec3]

    # Hashtable arrays
    ht_keys: wp.array[wp.uint64]
    ht_values: wp.array[wp.uint64]  # Slot rows (see reduction_value_index)
    ht_active_slots: wp.array[wp.int32]
    ht_insert_failures: wp.array[wp.int32]
    buffer_overflows: wp.array[wp.int32]
    ht_capacity: int
    ht_values_per_key: int


@wp.kernel(enable_backward=False)
def _clear_active_kernel(
    # Hashtable arrays
    ht_keys: wp.array[wp.uint64],
    ht_values: wp.array[wp.uint64],
    ht_active_slots: wp.array[wp.int32],
    # Hydroelastic per-entry arrays
    agg_force: wp.array[wp.vec3],
    agg_depth_volume: wp.array[wp.vec3],
    weighted_pos_sum: wp.array[wp.vec3],
    weight_sum: wp.array[wp.float32],
    total_depth_reduced: wp.array[wp.float32],
    total_normal_reduced: wp.array[wp.vec3],
    agg_moment_unreduced: wp.array[wp.float32],
    agg_moment_reduced: wp.array[wp.float32],
    agg_moment2_reduced: wp.array[wp.float32],
    # Counter arrays to zero (merged from _zero_count_and_contacts_kernel)
    contact_count: wp.array[wp.int32],
    reclaimed_contact_bits: wp.array[wp.uint32],
    reclaimed_contact_count: wp.array[wp.int32],
    ht_insert_failures: wp.array[wp.int32],
    buffer_overflows: wp.array[wp.int32],
    ht_capacity: int,
    num_threads: int,
):
    """Clear active hashtable entries, values, hydroelastic aggregates, and counters.

    Small active sets assign one thread per value of each entry's row to expose
    enough work for the GPU. Large active sets assign one thread per entry, which
    clears that entry's whole row. The active count is shared by the launch, so
    every thread follows the same branch.

    Thread 0 also zeros contact_count, ht_insert_failures and buffer_overflows (no other thread in this
    kernel reads them, so there is no race). The active-slots count stored at
    ``ht_active_slots[ht_capacity]`` must NOT be reset here: every thread reads it
    at the top of the kernel and we have no cross-block barrier, so a follow-up
    kernel launch (``_zero_active_count_kernel``) is used to zero it safely.

    Memory layout for values is entry-major, ``VALUES_STRIDE`` values per entry:
    [entry0_slot0, ..., entry0_slot7, entry1_slot0, ...]
    """
    tid = wp.tid()

    if tid == 0:
        contact_count[0] = 0
        if reclaimed_contact_count.shape[0] > 0:
            reclaimed_contact_count[0] = 0
        ht_insert_failures[0] = 0
        buffer_overflows[0] = 0

    # Read count from GPU - stored at active_slots[capacity].
    # All threads read this before it is modified by the follow-up zeroing kernel.
    count = ht_active_slots[ht_capacity]

    if count < wp.static(CLEAR_ACTIVE_ENTRY_PARALLEL_THRESHOLD):
        total_work = count * wp.static(VALUES_STRIDE)
        i = tid
        while i < total_work:
            active_idx = i / wp.static(VALUES_STRIDE)
            local_idx = i % wp.static(VALUES_STRIDE)
            entry_idx = ht_active_slots[active_idx]

            if local_idx == 0:
                ht_keys[entry_idx] = HASHTABLE_EMPTY_KEY
                if agg_force.shape[0] > 0:
                    agg_force[entry_idx] = wp.vec3(0.0, 0.0, 0.0)
                    agg_depth_volume[entry_idx] = wp.vec3(0.0, 0.0, 0.0)
                    weighted_pos_sum[entry_idx] = wp.vec3(0.0, 0.0, 0.0)
                    weight_sum[entry_idx] = 0.0
                    total_depth_reduced[entry_idx] = 0.0
                    total_normal_reduced[entry_idx] = wp.vec3(0.0, 0.0, 0.0)
                    if agg_moment_unreduced.shape[0] > 0:
                        agg_moment_unreduced[entry_idx] = 0.0
                        agg_moment_reduced[entry_idx] = 0.0
                        agg_moment2_reduced[entry_idx] = 0.0

            ht_values[reduction_value_index(entry_idx, local_idx)] = wp.uint64(0)
            i += num_threads
    else:
        active_idx = tid
        while active_idx < count:
            entry_idx = ht_active_slots[active_idx]

            ht_keys[entry_idx] = HASHTABLE_EMPTY_KEY
            if agg_force.shape[0] > 0:
                agg_force[entry_idx] = wp.vec3(0.0, 0.0, 0.0)
                agg_depth_volume[entry_idx] = wp.vec3(0.0, 0.0, 0.0)
                weighted_pos_sum[entry_idx] = wp.vec3(0.0, 0.0, 0.0)
                weight_sum[entry_idx] = 0.0
                total_depth_reduced[entry_idx] = 0.0
                total_normal_reduced[entry_idx] = wp.vec3(0.0, 0.0, 0.0)
                if agg_moment_unreduced.shape[0] > 0:
                    agg_moment_unreduced[entry_idx] = 0.0
                    agg_moment_reduced[entry_idx] = 0.0
                    agg_moment2_reduced[entry_idx] = 0.0

            for local_idx in range(wp.static(VALUES_STRIDE)):
                ht_values[reduction_value_index(entry_idx, local_idx)] = wp.uint64(0)

            active_idx += num_threads

    # Reclaimed IDs need not correspond to active hashtable entries.
    i = tid
    while i < reclaimed_contact_bits.shape[0]:
        reclaimed_contact_bits[i] = wp.uint32(0)
        i += num_threads


@wp.kernel(enable_backward=False)
def _zero_active_count_kernel(
    ht_active_slots: wp.array[wp.int32],
    ht_capacity: int,
):
    """Zero the active-slots count after ``_clear_active_kernel`` has finished.

    Launched as a separate kernel to obtain a grid-wide ordering guarantee:
    every thread of ``_clear_active_kernel`` has retired before this kernel
    starts, so we cannot race with any in-flight reads of
    ``ht_active_slots[ht_capacity]``.
    """
    ht_active_slots[ht_capacity] = 0


@wp.kernel(enable_backward=False)
def _reset_after_export_kernel(
    ht_active_slots: wp.array[wp.int32],
    ht_capacity: int,
    contact_count: wp.array[wp.int32],
    ht_insert_failures: wp.array[wp.int32],
    buffer_overflows: wp.array[wp.int32],
):
    """Reset counters after the preceding export retired all active entries."""
    ht_active_slots[ht_capacity] = 0
    contact_count[0] = 0
    ht_insert_failures[0] = 0
    buffer_overflows[0] = 0


class GlobalContactReducer:
    """Global contact reduction using hashtable-based tracking.

    This class manages:

    1. A global contact buffer storing contact data (struct of arrays)
    2. A hashtable tracking the best contact per (shape_pair, bin, slot)

    Slot counts depend on the configuration in ``contact_reduction.py``.

    **Hashtable Structure:**

    - Key: ``(shape_a, shape_b, bin_id)`` packed into 64 bits
    - bin_id ``0..NUM_NORMAL_BINS-1``: Normal bins (polyhedron faces)
    - Higher bin_ids: Voxel groups

    **Slot Layout per Normal Bin Entry** (``NUM_SPATIAL_DIRECTIONS + 1`` slots):

    - Slots ``0..NUM_SPATIAL_DIRECTIONS-1``: Spatial direction extremes (depth < beta)
    - Last slot: Maximum depth contact for the bin (unconditional)

    **Slot Layout per Voxel Group Entry** (``NUM_SPATIAL_DIRECTIONS + 1`` slots):

    - Each slot tracks the deepest contact for one voxel in the group
    - ``bin_id = NUM_NORMAL_BINS + (voxel_idx // group_size)``

    **Contact Data Storage:**

    Packed for efficient memory access:

    - position_depth: vec4(position.x, position.y, position.z, depth)
    - normal: vec2(octahedral-encoded unit normal)
    - shape_pairs: vec2i(shape_a, shape_b)
    - contact_area: force-bearing area when penetrating, full face area when speculative
      (optional, per hydroelastic contact)
    - contact_pressure: current face pressure (optional, per hydroelastic contact)

    Attributes:
        capacity: Maximum number of contacts that can be stored
        values_per_key: Number of value slots per hashtable entry (``NUM_SPATIAL_DIRECTIONS + 1``)
        values_stride: Length of each entry's row in ``ht_values`` (``VALUES_STRIDE``)
        position_depth: vec4 array storing position.xyz and depth
        normal: vec2 array storing octahedral-encoded contact normal
        shape_pairs: vec2i array storing (shape_a, shape_b) per contact
        contact_area: float array storing force-bearing area for penetrating
            contacts and full face area for speculative contacts (for hydroelastic)
        contact_pressure: float array storing current face pressure per contact (for hydroelastic)
        contact_count: Atomic counter for allocated contacts
        hashtable: HashTable for tracking best contacts (keys only)
        ht_values: Slot rows for the hashtable (managed here, not by HashTable),
            entry-major: see :func:`reduction_value_index`
    """

    def __init__(
        self,
        capacity: int,
        device: str | None = None,
        store_hydroelastic_data: bool = False,
        store_moment_data: bool = False,
        hashtable_size_factor: float = 0.25,
        enable_contact_reclamation: bool = False,
        enable_reduction: bool = True,
    ):
        """Initialize the global contact reducer.

        Args:
            capacity: Maximum number of contacts to store
            device: Warp device (e.g., "cuda:0", "cpu")
            store_hydroelastic_data: If True, allocate hydroelastic contact and aggregate arrays.
            store_moment_data: If True, allocate moment accumulator arrays for friction
                moment matching. Only needed when ``moment_matching=True``.
            hashtable_size_factor: Multiplier applied to ``capacity`` when sizing
                the reduction hashtable. Must be positive.
            enable_contact_reclamation: Allocate the reservation-reuse stack used
                by predictive contact reduction.
            enable_reduction: Allocate hashtable values and aggregate arrays.
                Disable when the contact buffer is decoded without reduction.
        """
        hashtable_size_factor = float(hashtable_size_factor)
        if not hashtable_size_factor > 0.0:
            raise ValueError(f"hashtable_size_factor must be > 0.0, got {hashtable_size_factor}")

        self.capacity = capacity
        self.device = device
        self.store_hydroelastic_data = store_hydroelastic_data
        self.hashtable_size_factor = hashtable_size_factor
        self.enable_contact_reclamation = enable_contact_reclamation
        self.enable_reduction = enable_reduction

        self.values_per_key = NUM_SPATIAL_DIRECTIONS + 1
        self.values_stride = VALUES_STRIDE

        # Contact buffer (struct of arrays)
        # ID zero is reserved for provisional winners; real contacts use the
        # remaining ``capacity`` elements with one-based IDs.
        buffer_size = capacity + 1
        self.position_depth = wp.zeros(buffer_size, dtype=wp.vec4, device=device)
        self.normal = wp.zeros(buffer_size, dtype=wp.vec2, device=device)  # Octahedral-encoded normals
        self.shape_pairs = wp.zeros(buffer_size, dtype=wp.vec2i, device=device)
        self.contact_fingerprints = wp.zeros(buffer_size, dtype=wp.int32, device=device)
        self.contact_slots = wp.full(
            buffer_size if enable_reduction else 0, wp.vec2i(-1, -1), dtype=wp.vec2i, device=device
        )

        # Optional hydroelastic data arrays
        if store_hydroelastic_data:
            self.contact_area = wp.zeros(buffer_size, dtype=wp.float32, device=device)
            self.contact_pressure = wp.zeros(buffer_size, dtype=wp.float32, device=device)
            self.contact_nbin_entry = wp.zeros(buffer_size if enable_reduction else 0, dtype=wp.int32, device=device)
        else:
            self.contact_area = wp.zeros(0, dtype=wp.float32, device=device)
            self.contact_pressure = wp.zeros(0, dtype=wp.float32, device=device)
            self.contact_nbin_entry = wp.zeros(0, dtype=wp.int32, device=device)

        # Atomic counter for contact allocation
        self.contact_count = wp.zeros(1, dtype=wp.int32, device=device)
        # Atomic bits avoid publishing a linked-list pointer separately from its head.
        if enable_contact_reclamation:
            self.reclaimed_contact_bits = wp.zeros(capacity // 32 + 1, dtype=wp.uint32, device=device)
            self.reclaimed_contact_count = wp.zeros(1, dtype=wp.int32, device=device)
        else:
            self.reclaimed_contact_bits = wp.zeros(0, dtype=wp.uint32, device=device)
            self.reclaimed_contact_count = wp.zeros(0, dtype=wp.int32, device=device)
        # Count failed hashtable inserts (e.g., table full)
        self.ht_insert_failures = wp.zeros(1, dtype=wp.int32, device=device)
        # Count contacts dropped because the contact buffer was full. The reservation in
        # export_contact_to_buffer is rolled back, so contact_count alone never shows the loss.
        self.buffer_overflows = wp.zeros(1, dtype=wp.int32, device=device)

        # Hashtable sizing: keep the historical default at capacity / 4 for
        # memory compatibility, while exposing a factor for dense batched scenes.
        # A full open-addressed table can turn failed inserts into whole-table probes.
        hashtable_size = max(int(capacity * hashtable_size_factor), 1024) if enable_reduction else 1
        self.hashtable = HashTable(hashtable_size, device=device)

        # Values array for hashtable - managed here, not by HashTable
        # This is contact-reduction-specific (entry-major rows of values_stride
        # values, the first values_per_key of which hold slots)
        self.ht_values = wp.zeros(
            self.hashtable.capacity * self.values_stride if enable_reduction else 0,
            dtype=wp.uint64,
            device=device,
        )

        # Aggregate force per hashtable entry (for hydroelastic stiffness calculation)
        # Accumulates sum(area * pressure * normal) for all penetrating contacts per entry
        if store_hydroelastic_data and enable_reduction:
            self.agg_force = wp.zeros(self.hashtable.capacity, dtype=wp.vec3, device=device)
            self.agg_depth_volume = wp.zeros(self.hashtable.capacity, dtype=wp.vec3, device=device)
            self.weighted_pos_sum = wp.zeros(self.hashtable.capacity, dtype=wp.vec3, device=device)
            self.weight_sum = wp.zeros(self.hashtable.capacity, dtype=wp.float32, device=device)
            # Total depth of reduced contacts per normal bin (accumulated from all winning contacts)
            self.total_depth_reduced = wp.zeros(self.hashtable.capacity, dtype=wp.float32, device=device)
            # Total depth-weighted normal of reduced contacts per normal bin
            self.total_normal_reduced = wp.zeros(self.hashtable.capacity, dtype=wp.vec3, device=device)
            # Moment accumulators for moment matching (friction scale adjustment)
            if store_moment_data:
                self.agg_moment_unreduced = wp.zeros(self.hashtable.capacity, dtype=wp.float32, device=device)
                self.agg_moment_reduced = wp.zeros(self.hashtable.capacity, dtype=wp.float32, device=device)
                self.agg_moment2_reduced = wp.zeros(self.hashtable.capacity, dtype=wp.float32, device=device)
            else:
                self.agg_moment_unreduced = wp.zeros(0, dtype=wp.float32, device=device)
                self.agg_moment_reduced = wp.zeros(0, dtype=wp.float32, device=device)
                self.agg_moment2_reduced = wp.zeros(0, dtype=wp.float32, device=device)
        else:
            self.agg_force = wp.zeros(0, dtype=wp.vec3, device=device)
            self.agg_depth_volume = wp.zeros(0, dtype=wp.vec3, device=device)
            self.weighted_pos_sum = wp.zeros(0, dtype=wp.vec3, device=device)
            self.weight_sum = wp.zeros(0, dtype=wp.float32, device=device)
            self.total_depth_reduced = wp.zeros(0, dtype=wp.float32, device=device)
            self.total_normal_reduced = wp.zeros(0, dtype=wp.vec3, device=device)
            self.agg_moment_unreduced = wp.zeros(0, dtype=wp.float32, device=device)
            self.agg_moment_reduced = wp.zeros(0, dtype=wp.float32, device=device)
            self.agg_moment2_reduced = wp.zeros(0, dtype=wp.float32, device=device)

    def clear(self):
        """Clear all contacts and reset the reducer (full clear)."""
        self.contact_count.zero_()
        self.reclaimed_contact_bits.zero_()
        self.reclaimed_contact_count.zero_()
        self.ht_insert_failures.zero_()
        self.buffer_overflows.zero_()
        self.hashtable.clear()
        self.ht_values.zero_()

    def clear_active(self):
        """Clear only the active entries (efficient for sparse usage).

        Uses two kernel launches (mirroring ``HashTable.clear_active``):

        1. ``_clear_active_kernel`` clears hashtable keys, values, hydroelastic
           aggregates, and per-step counters (``contact_count``,
           ``ht_insert_failures``, ``buffer_overflows``).
        2. ``_zero_active_count_kernel`` zeroes ``ht_active_slots[ht_capacity]``.

        The second kernel is needed because every thread of the first kernel
        reads ``ht_active_slots[ht_capacity]`` to size its grid-stride loop,
        and CUDA provides no intra-launch grid-wide barrier. Zeroing that slot
        from inside the first kernel would race with threads in
        later-scheduled blocks (or even later-issued warps/lanes under
        independent thread scheduling), causing some entries to be skipped.
        """
        # The clear is a grid-stride loop over the active entries with several
        # scattered stores each, so it needs many resident warps to hide the
        # store latency; the active count is only known on the device, and
        # surplus threads exit immediately.
        num_threads = min(65536, self.hashtable.capacity)

        wp.launch(
            _clear_active_kernel,
            dim=num_threads,
            inputs=[
                self.hashtable.keys,
                self.ht_values,
                self.hashtable.active_slots,
                self.agg_force,
                self.agg_depth_volume,
                self.weighted_pos_sum,
                self.weight_sum,
                self.total_depth_reduced,
                self.total_normal_reduced,
                self.agg_moment_unreduced,
                self.agg_moment_reduced,
                self.agg_moment2_reduced,
                self.contact_count,
                self.reclaimed_contact_bits,
                self.reclaimed_contact_count,
                self.ht_insert_failures,
                self.buffer_overflows,
                self.hashtable.capacity,
                num_threads,
            ],
            device=self.device,
            record_tape=False,
        )
        # Zero the active-slots count in a separate kernel so the write is
        # ordered (by the CUDA kernel-launch boundary) after every read in
        # `_clear_active_kernel`.
        wp.launch(
            _zero_active_count_kernel,
            dim=1,
            inputs=[self.hashtable.active_slots, self.hashtable.capacity],
            device=self.device,
            record_tape=False,
        )

    def reset_after_export(self):
        """Reset counters after an explicitly destructive export.

        Every active key and winner slot must already have been retired by
        ``create_export_reduced_contacts_kernel(clear_after_export=True)``.
        Call at the next collision boundary, after consumers have inspected the
        previous active count and overflow evidence. Raw payload and home
        entries remain untouched; the next contact allocations reset them.
        """
        if self.store_hydroelastic_data or self.enable_contact_reclamation:
            raise ValueError("Counter-only reset requires a non-hydroelastic reducer without contact reclamation")
        wp.launch(
            _reset_after_export_kernel,
            dim=1,
            inputs=[
                self.hashtable.active_slots,
                self.hashtable.capacity,
                self.contact_count,
                self.ht_insert_failures,
                self.buffer_overflows,
            ],
            device=self.device,
            record_tape=False,
        )

    def get_data_struct(self) -> GlobalContactReducerData:
        """Get a GlobalContactReducerData struct for passing to kernels.

        Returns:
            A GlobalContactReducerData struct containing all arrays.
        """
        data = GlobalContactReducerData()
        data.position_depth = self.position_depth
        data.normal = self.normal
        data.shape_pairs = self.shape_pairs
        data.contact_count = self.contact_count
        data.reclaimed_contact_bits = self.reclaimed_contact_bits
        data.reclaimed_contact_count = self.reclaimed_contact_count
        data.capacity = self.capacity
        data.contact_fingerprints = self.contact_fingerprints
        data.contact_slots = self.contact_slots
        data.contact_area = self.contact_area
        data.contact_pressure = self.contact_pressure
        data.contact_nbin_entry = self.contact_nbin_entry
        data.agg_force = self.agg_force
        data.agg_depth_volume = self.agg_depth_volume
        data.weighted_pos_sum = self.weighted_pos_sum
        data.weight_sum = self.weight_sum
        data.total_depth_reduced = self.total_depth_reduced
        data.total_normal_reduced = self.total_normal_reduced
        data.ht_keys = self.hashtable.keys
        data.ht_values = self.ht_values
        data.ht_active_slots = self.hashtable.active_slots
        data.ht_insert_failures = self.ht_insert_failures
        data.buffer_overflows = self.buffer_overflows
        data.ht_capacity = self.hashtable.capacity
        data.ht_values_per_key = self.values_per_key
        return data


@wp.func
def reclaim_contact_id(contact_id: int, reducer_data: GlobalContactReducerData):
    """Mark an unreferenced materialized contact ID as available for reuse."""
    if reducer_data.reclaimed_contact_bits.shape[0] == 0:
        return
    word = contact_id / 32
    mask = wp.uint32(1) << wp.uint32(contact_id % 32)
    old_bits = wp.atomic_or(reducer_data.reclaimed_contact_bits, word, mask)
    # Count only newly set bits so every reservation is backed by a bit.
    if old_bits & mask == wp.uint32(0):
        wp.atomic_add(reducer_data.reclaimed_contact_count, 0, 1)


@wp.func
def _pop_reclaimed_contact_id(reducer_data: GlobalContactReducerData, ticket: int) -> int:
    """Claim a reclaimed contact ID, returning zero when none is available.

    A single bitmap pass can miss bits published behind it, so reserve against
    the count first; a successful reservation guarantees the scan finds a bit.
    """
    while wp.atomic_add(reducer_data.reclaimed_contact_count, 0, -1) <= 0:
        # Other failed pops' transient debits can hide IDs; whoever undoes
        # last sees the true count, so retry while it is positive.
        if wp.atomic_add(reducer_data.reclaimed_contact_count, 0, 1) < 0:
            return 0

    word_count = reducer_data.reclaimed_contact_bits.shape[0]
    word = ticket % word_count
    while True:
        bits = wp.atomic_or(reducer_data.reclaimed_contact_bits, word, wp.uint32(0))
        bit = int(0)
        while bit < 32:
            mask = wp.uint32(1) << wp.uint32(bit)
            if bits & mask != wp.uint32(0):
                keep_mask = ~mask
                old_bits = wp.atomic_and(reducer_data.reclaimed_contact_bits, word, keep_mask)
                if old_bits & mask != wp.uint32(0):
                    return word * 32 + bit
            bit += 1
        word = (word + 1) % word_count


@wp.func
def export_contact_to_buffer(
    shape_a: int,
    shape_b: int,
    position: wp.vec3,
    normal: wp.vec3,
    depth: float,
    fingerprint: int,
    reducer_data: GlobalContactReducerData,
) -> int:
    """Store a contact in the buffer without reduction.

    Args:
        shape_a: First shape index
        shape_b: Second shape index
        position: Contact position in world space
        normal: Contact normal
        depth: Penetration depth (negative = penetrating)
        fingerprint: Deterministic contact identifier (e.g. triangle/edge/vertex index)
        reducer_data: GlobalContactReducerData with all arrays

    Returns:
        Contact ID if successfully stored, -1 if buffer full
    """
    # Prefer fresh IDs so the default path never probes the optional
    # reclamation stack. Once fresh capacity is exhausted, speculative
    # reducers can reuse reservations that lost every slot claim.
    buffer_idx = wp.atomic_add(reducer_data.contact_count, 0, 1)
    if buffer_idx < reducer_data.capacity:
        contact_id = buffer_idx + 1  # ID zero is reserved for provisional winners.
    else:
        contact_id = int(0)
        if reducer_data.reclaimed_contact_bits.shape[0] > 0:
            contact_id = _pop_reclaimed_contact_id(reducer_data, buffer_idx - reducer_data.capacity)
        if contact_id == 0:
            # Undo the reservation because no buffer slot was available.
            wp.atomic_add(reducer_data.contact_count, 0, -1)
            wp.atomic_add(reducer_data.buffer_overflows, 0, 1)
            return -1
        # A successful reuse is not a dropped contact.
        wp.atomic_add(reducer_data.contact_count, 0, -1)
    # Store contact data (packed into vec4, normal octahedral-encoded into vec2)
    reducer_data.position_depth[contact_id] = wp.vec4(position[0], position[1], position[2], depth)
    reducer_data.normal[contact_id] = encode_oct(normal)
    reducer_data.shape_pairs[contact_id] = wp.vec2i(shape_a, shape_b)
    reducer_data.contact_fingerprints[contact_id] = fingerprint
    if reducer_data.contact_slots.shape[0] > 0:
        reducer_data.contact_slots[contact_id] = wp.vec2i(-1, -1)

    return contact_id


@wp.func
def support_score(pos_2d: wp.vec2, dir_i: int, end_depth: float, speculative: bool) -> float:
    """A candidate's score for its normal bin's direction slot ``dir_i``.

    The projection of its tangent position on the slot's direction; under speculation less its standing (its closest
    approach over the window, ``end_depth``), so the slot keeps the support of the lower hull of the bin's (tangent
    position, standing) points: a point higher than it is further out cannot bear the pair within any rotation a window
    makes, and loses the slot to a lower one. See the module's notes.
    """
    score = wp.dot(pos_2d, get_spatial_direction_2d(dir_i))
    if speculative:
        score -= end_depth
    return score


@wp.func
def reduce_contact_in_hashtable(
    contact_id: int,
    reducer_data: GlobalContactReducerData,
    beta: float,
    shape_transform: wp.array[wp.transform],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    shape_voxel_resolution: wp.array[wp.vec3i],
    closing: float,
    speculative: bool,
):
    """Register a buffered contact in the reduction hashtable.

    Uses single beta threshold for contact reduction with two strategies:

    1. **Normal-binned slots** (``NUM_NORMAL_BINS`` x ``NUM_SPATIAL_DIRECTIONS + 1``):
       - Spatial direction slots for contacts with depth < beta
       - 1 max-depth slot per normal bin (always participates)

    2. **Voxel-based depth slots** (``NUM_VOXEL_DEPTH_SLOTS`` voxels, grouped):
       - Voxels are grouped by ``NUM_SPATIAL_DIRECTIONS + 1``
       - Each slot tracks the deepest contact in that voxel region
       - Provides spatial coverage independent of contact normal

    Args:
        contact_id: Index of contact in buffer
        reducer_data: Reducer data
        beta: Depth threshold (contacts with depth < beta participate in spatial competition)
        shape_transform: Per-shape world transforms (for transforming position to local space)
        shape_collision_aabb_lower: Per-shape local AABB lower bounds
        shape_collision_aabb_upper: Per-shape local AABB upper bounds
        shape_voxel_resolution: Per-shape voxel grid resolution
        closing: The depth [m] the contact's points close over the pass's window (closing speed times the window,
            rotation included; zero without speculation). The contact stands by its closest approach over the window
            (its depth less the closing when it closes): it competes for the bin's extremes when it touches by the
            window's end, and the depth slots rank it by that standing.
        speculative: Whether the pass predicts its window: the extremes then keep the support of the lower hull in
            (tangent position, standing) (:func:`support_score`), contacts touching now ahead of the ones that only
            reach the window's end.
    """
    # Read contact data from buffer (normal is octahedral-encoded)
    pd = reducer_data.position_depth[contact_id]
    normal = decode_oct(reducer_data.normal[contact_id])
    pair = reducer_data.shape_pairs[contact_id]
    fingerprint = reducer_data.contact_fingerprints[contact_id]

    position = wp.vec3(pd[0], pd[1], pd[2])
    depth = pd[3]
    shape_a = pair[0]  # Mesh shape
    shape_b = pair[1]  # Convex shape

    aabb_lower = shape_collision_aabb_lower[shape_a]
    aabb_upper = shape_collision_aabb_upper[shape_a]

    # === Part 1: Normal-binned reduction (spatial extremes + max-depth per bin) ===
    # Get normal bin from polyhedron face matching
    bin_id = get_slot(normal)

    # Project position to 2D plane of the polyhedron face
    pos_2d = project_point_to_plane(bin_id, position)

    # Key is (shape_a, shape_b, bin_id)
    key = make_contact_key(shape_a, shape_b, bin_id)

    # Find or create the hashtable entry ONCE, then write directly to slots
    entry_idx = hashtable_find_or_insert(key, reducer_data.ht_keys, reducer_data.ht_active_slots)
    # The candidate's deepest over the window: its gap moves linearly, so its closest approach is at one end.
    end_depth = depth - wp.max(closing, 0.0)
    raised_entry = False
    if entry_idx >= 0:
        touch = beta * wp.length(aabb_upper - aabb_lower)
        use_beta = end_depth < touch
        # Under speculation the rows that bear the pair now outrank the ones that only reach the window's end.
        touching_now = depth < touch
        for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
            if use_beta:
                score = support_score(pos_2d, dir_i, end_depth, speculative)
                if speculative:
                    key = spatial_contact_key(score, touching_now, fingerprint)
                else:
                    key = contact_key(score, fingerprint)
                if reduction_update_slot(entry_idx, dir_i, key, reducer_data.ht_values):
                    raised_entry = True

        if reduction_update_slot(
            entry_idx, wp.static(NUM_SPATIAL_DIRECTIONS), contact_key(-end_depth, fingerprint), reducer_data.ht_values
        ):
            raised_entry = True
    else:
        wp.atomic_add(reducer_data.ht_insert_failures, 0, 1)

    # === Part 2: Voxel-based reduction (deepest contact per voxel) ===
    # Transform contact position from world space to shape_a's local space
    X_shape_ws = shape_transform[shape_a]
    X_ws_shape = wp.transform_inverse(X_shape_ws)
    position_local = wp.transform_point(X_ws_shape, position)

    # Compute voxel index using shape_a's local AABB
    voxel_res = shape_voxel_resolution[shape_a]
    voxel_idx = compute_voxel_index(position_local, aabb_lower, aabb_upper, voxel_res)

    # Clamp voxel index to valid range
    voxel_idx = wp.clamp(voxel_idx, 0, wp.static(NUM_VOXEL_DEPTH_SLOTS - 1))

    voxels_per_group = wp.static(NUM_SPATIAL_DIRECTIONS + 1)
    voxel_group = voxel_idx // voxels_per_group
    voxel_local_slot = voxel_idx % voxels_per_group

    voxel_bin_id = wp.static(NUM_NORMAL_BINS) + voxel_group
    voxel_key = make_contact_key(shape_a, shape_b, voxel_bin_id)

    voxel_entry_idx = hashtable_find_or_insert(voxel_key, reducer_data.ht_keys, reducer_data.ht_active_slots)
    raised_voxel = False
    if voxel_entry_idx >= 0:
        # Key by -depth so the most penetrating (most negative depth) at the window's end wins
        raised_voxel = reduction_update_slot(
            voxel_entry_idx, voxel_local_slot, contact_key(-end_depth, fingerprint), reducer_data.ht_values
        )
    else:
        wp.atomic_add(reducer_data.ht_insert_failures, 0, 1)
    # The entries in which it offered a slot its key (every raise among them), for the resolve pass.
    reducer_data.contact_slots[contact_id] = wp.vec2i(
        wp.where(raised_entry, entry_idx, -1), wp.where(raised_voxel, voxel_entry_idx, -1)
    )


@wp.func
def export_and_reduce_contact(
    shape_a: int,
    shape_b: int,
    position: wp.vec3,
    normal: wp.vec3,
    depth: float,
    fingerprint: int,
    reducer_data: GlobalContactReducerData,
    beta: float,
    shape_transform: wp.array[wp.transform],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    shape_voxel_resolution: wp.array[wp.vec3i],
) -> int:
    """Export contact to buffer and register in hashtable for reduction."""
    contact_id = export_contact_to_buffer(shape_a, shape_b, position, normal, depth, fingerprint, reducer_data)

    if contact_id >= 0:
        reduce_contact_in_hashtable(
            contact_id,
            reducer_data,
            beta,
            shape_transform,
            shape_collision_aabb_lower,
            shape_collision_aabb_upper,
            shape_voxel_resolution,
            0.0,
            False,
        )

    return contact_id


@wp.func
def export_and_reduce_contact_centered(
    shape_a: int,
    shape_b: int,
    position: wp.vec3,
    normal: wp.vec3,
    depth: float,
    fingerprint: int,
    centered_position: wp.vec3,
    X_ws_voxel_shape: wp.transform,
    aabb_lower_voxel: wp.vec3,
    aabb_upper_voxel: wp.vec3,
    voxel_res: wp.vec3i,
    reducer_data: GlobalContactReducerData,
) -> int:
    """Export contact to buffer and register in hashtable matching thread-block behavior.

    Differs from :func:`export_and_reduce_contact` in three ways that match
    the shared-memory reduction used for mesh-mesh contacts:

    - Spatial projection uses *centered_position* (midpoint-centered)
    - Beta threshold is fixed at 0.0001 m (not scale-relative)
    - Voxel grid uses the caller-specified AABB/transform (tri_shape's)

    Pre-prunes contacts by non-atomically reading current slot values before
    allocating a buffer slot. Contacts that cannot beat any existing winner
    are skipped entirely, reducing buffer pressure.

    Args:
        shape_a: First shape index (for hashtable key)
        shape_b: Second shape index (for hashtable key)
        position: World-space contact position (stored in buffer)
        normal: Contact normal (a-to-b)
        depth: Penetration depth
        fingerprint: Deterministic contact identifier (e.g. edge index)
        centered_position: Midpoint-centered position for spatial projection
        X_ws_voxel_shape: World-to-local transform for voxel computation
        aabb_lower_voxel: Local AABB lower for voxel grid
        aabb_upper_voxel: Local AABB upper for voxel grid
        voxel_res: Voxel grid resolution
        reducer_data: Global reducer data
    """
    use_beta = depth < wp.static(BETA_THRESHOLD)

    # === Normal bin: find/create hashtable entry ===
    bin_id = get_slot(normal)
    pos_2d = project_point_to_plane(bin_id, centered_position)
    key = make_contact_key(shape_a, shape_b, bin_id)

    entry_idx = hashtable_find_or_insert(key, reducer_data.ht_keys, reducer_data.ht_active_slots)

    # === Pre-prune normal bin: probe the slots before buffering ===
    might_win = False

    if entry_idx >= 0:
        if reduction_slot_open(
            entry_idx, wp.static(NUM_SPATIAL_DIRECTIONS), contact_key(-depth, fingerprint), reducer_data.ht_values
        ):
            might_win = True

        if not might_win and use_beta:
            for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
                if not might_win:
                    dir_2d = get_spatial_direction_2d(dir_i)
                    score = wp.dot(pos_2d, dir_2d)
                    if reduction_slot_open(entry_idx, dir_i, contact_key(score, fingerprint), reducer_data.ht_values):
                        might_win = True

    # === Voxel bin: only look up if normal bin didn't already show a win ===
    position_local = wp.transform_point(X_ws_voxel_shape, position)
    voxel_idx = compute_voxel_index(position_local, aabb_lower_voxel, aabb_upper_voxel, voxel_res)
    voxel_idx = wp.clamp(voxel_idx, 0, wp.static(NUM_VOXEL_DEPTH_SLOTS - 1))

    voxels_per_group = wp.static(NUM_SPATIAL_DIRECTIONS + 1)
    voxel_group = voxel_idx // voxels_per_group
    voxel_local_slot = voxel_idx % voxels_per_group
    voxel_bin_id = wp.static(NUM_NORMAL_BINS) + voxel_group
    voxel_key = make_contact_key(shape_a, shape_b, voxel_bin_id)

    voxel_entry_idx = -1
    if not might_win:
        voxel_entry_idx = hashtable_find_or_insert(voxel_key, reducer_data.ht_keys, reducer_data.ht_active_slots)
        if voxel_entry_idx >= 0:
            if reduction_slot_open(
                voxel_entry_idx, voxel_local_slot, contact_key(-depth, fingerprint), reducer_data.ht_values
            ):
                might_win = True

    if not might_win:
        return -1

    # === Allocate buffer slot (only for contacts that might win) ===
    contact_id = export_contact_to_buffer(shape_a, shape_b, position, normal, depth, fingerprint, reducer_data)
    if contact_id < 0:
        return -1

    # === Raise the slots: the score wins, the fingerprint breaks ties ===
    raised_entry = False
    if entry_idx >= 0:
        for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
            if use_beta:
                dir_2d = get_spatial_direction_2d(dir_i)
                score = wp.dot(pos_2d, dir_2d)
                if reduction_update_slot(entry_idx, dir_i, contact_key(score, fingerprint), reducer_data.ht_values):
                    raised_entry = True

        if reduction_update_slot(
            entry_idx, wp.static(NUM_SPATIAL_DIRECTIONS), contact_key(-depth, fingerprint), reducer_data.ht_values
        ):
            raised_entry = True
    else:
        wp.atomic_add(reducer_data.ht_insert_failures, 0, 1)

    # Deferred voxel entry lookup for contacts that won via normal bin
    if voxel_entry_idx < 0:
        voxel_entry_idx = hashtable_find_or_insert(voxel_key, reducer_data.ht_keys, reducer_data.ht_active_slots)
    raised_voxel = False
    if voxel_entry_idx >= 0:
        raised_voxel = reduction_update_slot(
            voxel_entry_idx, voxel_local_slot, contact_key(-depth, fingerprint), reducer_data.ht_values
        )
    else:
        wp.atomic_add(reducer_data.ht_insert_failures, 0, 1)
    reducer_data.contact_slots[contact_id] = wp.vec2i(
        wp.where(raised_entry, entry_idx, -1), wp.where(raised_voxel, voxel_entry_idx, -1)
    )

    return contact_id


@wp.func
def _two_depth_value_index(slot: int, entry_idx: int, voxel_entry_idx: int, voxel_local_slot: int) -> int:
    """The value index of a mesh-SDF candidate's slot ``slot``: its normal bin's 0..6, its voxel's 7."""
    if slot == wp.static(NUM_SPATIAL_DIRECTIONS + 1):
        return reduction_value_index(voxel_entry_idx, voxel_local_slot)
    return reduction_value_index(entry_idx, slot)


@wp.func
def _two_depth_holds(
    raised: int,
    keys: replaced_values_vec_type,
    entry_idx: int,
    voxel_entry_idx: int,
    voxel_local_slot: int,
    values: wp.array[wp.uint64],
) -> bool:
    """Whether a mesh-SDF candidate still holds any slot it raised (:func:`reduction_slot_key`, the reads issued back
    to back)."""
    holds = False
    for slot in range(wp.static(NUM_SPATIAL_DIRECTIONS + 2)):
        if (raised & (1 << slot)) != 0:
            if (
                reduction_slot_key(values, _two_depth_value_index(slot, entry_idx, voxel_entry_idx, voxel_local_slot))
                == keys[slot]
            ):
                holds = True
    return holds


@wp.func
def _export_and_reduce_contact_centered_two_spatial_depths(
    shape_a: int,
    shape_b: int,
    position: wp.vec3,
    normal: wp.vec3,
    depth: float,
    step_end_depth: float,
    fingerprint: int,
    centered_position: wp.vec3,
    inner_spatial_depth: float,
    outer_spatial_depth: float,
    position_local: wp.vec3,
    aabb_lower_voxel: wp.vec3,
    aabb_upper_voxel: wp.vec3,
    voxel_res: wp.vec3i,
    reducer_data: GlobalContactReducerData,
    speculative: bool,
) -> int:
    """Export contact with touching-first spatial winners.

    Directional slots accept contacts out to ``outer_spatial_depth``, but
    candidates that touch over the step, their closest approach inside
    ``inner_spatial_depth``, outrank the others. ``step_end_depth`` is the
    depth less the candidate's closing speed times the step (its current
    ``depth`` when the motion is unknown); the closest approach is the deeper
    of the two, so the far side of a support that a rocking body swings onto
    this step keeps rows and a touching point that separates keeps its own.
    Within a tier the furthest spatial projection wins, under speculation
    less the closest approach (:func:`support_score`); the max-depth and
    voxel slots take the contacts touching over the step and rank them by
    their closest approach. Ties go to the higher fingerprint (the key's low bits).

    The candidate raises its slots first (atomic maxima of its keys) and is buffered only where it raised one; the
    resolve pass then names each slot's winner (:func:`resolve_reduced_contacts`).
    """
    # The candidate's deepest over the step: its gap moves linearly, so its closest approach is at one end.
    step_end_depth = wp.min(step_end_depth, depth)
    use_inner = step_end_depth < inner_spatial_depth
    step_end_inner = use_inner
    use_outer = depth < outer_spatial_depth

    if not use_outer:
        return -1

    # === Normal bin: prioritized spatial slots and inner max-depth slot ===
    bin_id = get_slot(normal)
    pos_2d = project_point_to_plane(bin_id, centered_position)
    key = make_contact_key(shape_a, shape_b, bin_id)

    # === Voxel bin: inner depth coverage ===
    voxel_idx = compute_voxel_index(position_local, aabb_lower_voxel, aabb_upper_voxel, voxel_res)
    voxel_idx = wp.clamp(voxel_idx, 0, wp.static(NUM_VOXEL_DEPTH_SLOTS - 1))

    voxels_per_group = wp.static(NUM_SPATIAL_DIRECTIONS + 1)
    voxel_group = voxel_idx // voxels_per_group
    voxel_local_slot = voxel_idx % voxels_per_group
    voxel_bin_id = wp.static(NUM_NORMAL_BINS) + voxel_group
    voxel_key = make_contact_key(shape_a, shape_b, voxel_bin_id)

    values = reducer_data.ht_values
    # Find both entries up front so their probes and the slot reads below overlap; a missing voxel entry is inserted
    # only for a candidate that would raise its slot.
    entry_idx = hashtable_find_or_insert(key, reducer_data.ht_keys, reducer_data.ht_active_slots)
    if entry_idx < 0:
        wp.atomic_add(reducer_data.ht_insert_failures, 0, 1)
    voxel_entry_idx = -1
    if use_inner:
        voxel_entry_idx = hashtable_find(voxel_key, reducer_data.ht_keys)
    depth_key = contact_key(-step_end_depth, fingerprint)
    keys = replaced_values_vec_type()
    for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
        keys[dir_i] = spatial_contact_key(
            support_score(pos_2d, dir_i, step_end_depth, speculative), step_end_inner, fingerprint
        )
    keys[wp.static(NUM_SPATIAL_DIRECTIONS)] = depth_key
    keys[wp.static(NUM_SPATIAL_DIRECTIONS + 1)] = depth_key

    # Read every slot before comparing so the loads issue back to back: a slot only rises, so a candidate a read
    # turns away stays beaten.
    held = replaced_values_vec_type()
    if entry_idx >= 0:
        for slot in range(wp.static(NUM_SPATIAL_DIRECTIONS + 1)):
            held[slot] = values[reduction_value_index(entry_idx, slot)]
    if voxel_entry_idx >= 0:
        held[wp.static(NUM_SPATIAL_DIRECTIONS + 1)] = values[reduction_value_index(voxel_entry_idx, voxel_local_slot)]
    open_slots = int(0)
    if entry_idx >= 0:
        for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
            if held[dir_i] < keys[dir_i]:
                open_slots |= 1 << dir_i
        if use_inner and held[wp.static(NUM_SPATIAL_DIRECTIONS)] < depth_key:
            open_slots |= 1 << wp.static(NUM_SPATIAL_DIRECTIONS)
    if use_inner and held[wp.static(NUM_SPATIAL_DIRECTIONS + 1)] < depth_key:
        open_slots |= 1 << wp.static(NUM_SPATIAL_DIRECTIONS + 1)
    if open_slots == 0:
        return -1
    if use_inner and voxel_entry_idx < 0:
        voxel_entry_idx = hashtable_find_or_insert(voxel_key, reducer_data.ht_keys, reducer_data.ht_active_slots)
        if voxel_entry_idx < 0:
            wp.atomic_add(reducer_data.ht_insert_failures, 0, 1)
            open_slots &= ~(1 << wp.static(NUM_SPATIAL_DIRECTIONS + 1))

    # Raise the open slots; the atomics are independent, so their round trips overlap.
    raised = int(0)
    for slot in range(wp.static(NUM_SPATIAL_DIRECTIONS + 2)):
        if (open_slots & (1 << slot)) != 0:
            value_idx = _two_depth_value_index(slot, entry_idx, voxel_entry_idx, voxel_local_slot)
            if wp.atomic_max(values, value_idx, keys[slot]) < keys[slot]:
                raised |= 1 << slot

    # A candidate that raised no slot can never hold one (a slot only rises): it is not buffered, nor is one that holds
    # none of its raised slots any more (beaten for good). A buffered one found beaten right after returns its row, so
    # the buffer holds the slots' holders (the winners among them) however many candidates raised a slot on the way.
    if raised == 0:
        return -1
    if not _two_depth_holds(raised, keys, entry_idx, voxel_entry_idx, voxel_local_slot, values):
        return -1
    contact_id = export_contact_to_buffer(shape_a, shape_b, position, normal, depth, fingerprint, reducer_data)
    if contact_id < 0:
        return -1
    voxel_bit = 1 << wp.static(NUM_SPATIAL_DIRECTIONS + 1)
    reducer_data.contact_slots[contact_id] = wp.vec2i(
        wp.where((raised & ~voxel_bit) != 0, entry_idx, -1), wp.where((raised & voxel_bit) != 0, voxel_entry_idx, -1)
    )
    if not _two_depth_holds(raised, keys, entry_idx, voxel_entry_idx, voxel_local_slot, values):
        reclaim_contact_id(contact_id, reducer_data)
        return -1
    return contact_id


@wp.func
def export_and_reduce_contact_centered_two_spatial_depths(
    shape_a: int,
    shape_b: int,
    position: wp.vec3,
    normal: wp.vec3,
    depth: float,
    step_end_depth: float,
    fingerprint: int,
    centered_position: wp.vec3,
    inner_spatial_depth: float,
    outer_spatial_depth: float,
    position_local: wp.vec3,
    aabb_lower_voxel: wp.vec3,
    aabb_upper_voxel: wp.vec3,
    voxel_res: wp.vec3i,
    reducer_data: GlobalContactReducerData,
) -> int:
    """Export a contact without speculation (:func:`_export_and_reduce_contact_centered_two_spatial_depths`)."""
    return _export_and_reduce_contact_centered_two_spatial_depths(
        shape_a,
        shape_b,
        position,
        normal,
        depth,
        step_end_depth,
        fingerprint,
        centered_position,
        inner_spatial_depth,
        outer_spatial_depth,
        position_local,
        aabb_lower_voxel,
        aabb_upper_voxel,
        voxel_res,
        reducer_data,
        False,
    )


@wp.func
def export_and_reduce_predictive_contact(
    shape_a: int,
    shape_b: int,
    position: wp.vec3,
    normal: wp.vec3,
    depth: float,
    surface_offset_sum: float,
    radius_eff_a: float,
    radius_eff_b: float,
    fingerprint: int,
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    collision_update_dt: float,
    max_speculative_extension: float,
    existing_contact_id: int,
    reducer_data: GlobalContactReducerData,
) -> int:
    """Register an exact predictive candidate in a bounded shape-pair manifold.

    Args:
        shape_a: First shape index.
        shape_b: Second shape index.
        position: Contact midpoint in world space [m].
        normal: Contact normal from the first shape to the second. A record this call buffers stores it as given,
            the bits the regular export stores for the same candidate (:func:`export_contact_to_buffer`): which of
            the two buffers a candidate depends on the slot races, so its data must not.
        depth: Centerline contact distance [m].
        surface_offset_sum: Combined collision margins [m].
        radius_eff_a: Effective radius of the first shape [m].
        radius_eff_b: Effective radius of the second shape [m].
        fingerprint: Deterministic contact identifier.
        shape_transform: World-space shape transforms.
        shape_linear_velocity: Shape-origin linear velocities [m/s].
        shape_angular_velocity: Shape angular velocities [rad/s].
        collision_update_dt: Collision prediction horizon [s].
        max_speculative_extension: Maximum speculative clearance [m].
        existing_contact_id: The candidate's buffered contact (a positive one-based ID), or ``-1``: it is buffered
            here if it raises a slot.
        reducer_data: Global contact-reducer storage.
    """
    clearance = depth - surface_offset_sum
    if clearance <= 0.0 or clearance > max_speculative_extension:
        return -1

    contact_normal = wp.normalize(normal)
    point_a = position - contact_normal * (0.5 * depth + radius_eff_a)
    point_b = position + contact_normal * (0.5 * depth + radius_eff_b)
    closing_speed = compute_contact_approach_speed(
        shape_a,
        shape_b,
        point_a,
        point_b,
        contact_normal,
        shape_transform,
        shape_linear_velocity,
        shape_angular_velocity,
    )
    if closing_speed <= 0.0 or clearance > closing_speed * collision_update_dt:
        return -1

    key = make_contact_key(shape_a, shape_b, wp.static(PREDICTIVE_BIN_ID))
    entry_idx = hashtable_find_or_insert(key, reducer_data.ht_keys, reducer_data.ht_active_slots)
    if entry_idx < 0:
        wp.atomic_add(reducer_data.ht_insert_failures, 0, 1)
        return -1

    clearance_score = -clearance
    impact_time_score = collision_update_dt - clearance / closing_speed
    shard_hash = wp.uint32(fingerprint)
    shard_hash = shard_hash ^ (shard_hash >> wp.uint32(16))
    shard_hash = shard_hash * wp.uint32(0x7FEB352D)
    shard_hash = shard_hash ^ (shard_hash >> wp.uint32(15))
    shard_hash = shard_hash * ((wp.uint32(0x846C) << wp.uint32(16)) | wp.uint32(0xA68B))
    shard_hash = shard_hash ^ (shard_hash >> wp.uint32(16))
    clearance_slot = int(shard_hash % wp.uint32(wp.static(NUM_SPATIAL_DIRECTIONS)))
    values = reducer_data.ht_values
    clearance_key = contact_key(clearance_score, fingerprint)
    impact_time_key = contact_key(impact_time_score, fingerprint)
    clearance_idx = reduction_value_index(entry_idx, clearance_slot)
    impact_time_idx = reduction_value_index(entry_idx, wp.static(NUM_SPATIAL_DIRECTIONS))
    raised_clearance = reduction_try_update_slot(entry_idx, clearance_slot, clearance_key, values)
    raised_impact_time = reduction_try_update_slot(
        entry_idx, wp.static(NUM_SPATIAL_DIRECTIONS), impact_time_key, values
    )
    # A candidate that raised no slot can never hold one (a slot only rises); one that holds none of its raised slots
    # any more is beaten for good. Buffered here and found beaten, it returns its row (as the mesh-SDF export does).
    if not raised_clearance and not raised_impact_time:
        return -1
    if existing_contact_id >= 0:
        # The resolve pass visits its pair's predictive entry for it.
        word = reducer_data.contact_slots[existing_contact_id][1]
        reducer_data.contact_slots[existing_contact_id][1] = competed_voxel_word(competed_voxel_entry(word), True)
        return existing_contact_id
    holds = (raised_clearance and reduction_slot_holds(clearance_idx, clearance_key, values)) or (
        raised_impact_time and reduction_slot_holds(impact_time_idx, impact_time_key, values)
    )
    if not holds:
        return -1
    contact_id = export_contact_to_buffer(shape_a, shape_b, position, normal, depth, fingerprint, reducer_data)
    if contact_id < 0:
        return -1
    holds = (raised_clearance and reduction_slot_holds(clearance_idx, clearance_key, values)) or (
        raised_impact_time and reduction_slot_holds(impact_time_idx, impact_time_key, values)
    )
    if not holds:
        reclaim_contact_id(contact_id, reducer_data)
        return -1
    reducer_data.contact_slots[contact_id] = wp.vec2i(-1, competed_voxel_word(-1, True))
    return contact_id


# A module of its own: it launches at the narrow phase's block, the module's other kernels at the default.
@wp.kernel(enable_backward=False, module="unique")
def reduce_buffered_contacts_kernel(
    reducer_data: GlobalContactReducerData,
    shape_transform: wp.array[wp.transform],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    shape_voxel_resolution: wp.array[wp.vec3i],
    total_num_threads: int,
):
    """Register buffered contacts in the hashtable for reduction.

    Uses the fixed BETA_THRESHOLD (0.1mm) for spatial competition.
    Contacts with depth < beta participate in spatial extreme competition.
    """
    tid = wp.tid()

    # Get total number of contacts written
    num_contacts = reducer_data.contact_count[0]

    # Early exit if no contacts (fast path for empty work)
    if num_contacts == 0:
        return

    # Cap at capacity
    num_contacts = wp.min(num_contacts, reducer_data.capacity)

    # Grid stride loop over contacts
    for i in range(tid, num_contacts, total_num_threads):
        reduce_contact_in_hashtable(
            i + 1,
            reducer_data,
            wp.static(BETA_THRESHOLD),
            shape_transform,
            shape_collision_aabb_lower,
            shape_collision_aabb_upper,
            shape_voxel_resolution,
            0.0,
            False,
        )


# A module of its own: it launches at the narrow phase's block, the module's other kernels at the default.
@wp.kernel(enable_backward=False, module="unique")
def reduce_buffered_contacts_speculative_kernel(
    reducer_data: GlobalContactReducerData,
    shape_types: wp.array[int],
    shape_data: wp.array[wp.vec4],
    shape_gap: wp.array[float],
    shape_transform: wp.array[wp.transform],
    shape_linear_velocity: wp.array[wp.vec3],
    shape_angular_velocity: wp.array[wp.vec3],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    shape_voxel_resolution: wp.array[wp.vec3i],
    collision_update_dt: float,
    max_speculative_extension: float,
    total_num_threads: int,
):
    """Reduce buffered contacts and retain an exact predictive manifold per pair."""
    tid = wp.tid()
    num_contacts = wp.min(reducer_data.contact_count[0], reducer_data.capacity)
    for i in range(tid, num_contacts, total_num_threads):
        contact_id = i + 1
        pair = reducer_data.shape_pairs[contact_id]
        shape_a = pair[0]
        shape_b = pair[1]
        pd = reducer_data.position_depth[contact_id]
        radius_eff_a = compute_effective_radius(shape_types[shape_a], shape_data[shape_a])
        radius_eff_b = compute_effective_radius(shape_types[shape_b], shape_data[shape_b])
        surface_offset_sum = shape_data[shape_a][3] + shape_data[shape_b][3]
        clearance = pd[3] - surface_offset_sum
        base_gap_sum = shape_gap[shape_a] + shape_gap[shape_b]
        if clearance <= base_gap_sum:
            normal = wp.normalize(decode_oct(reducer_data.normal[contact_id]))
            position = wp.vec3(pd[0], pd[1], pd[2])
            closing_speed = compute_contact_approach_speed(
                shape_a,
                shape_b,
                position - normal * (0.5 * pd[3] + radius_eff_a),
                position + normal * (0.5 * pd[3] + radius_eff_b),
                normal,
                shape_transform,
                shape_linear_velocity,
                shape_angular_velocity,
            )
            reduce_contact_in_hashtable(
                contact_id,
                reducer_data,
                wp.static(BETA_THRESHOLD),
                shape_transform,
                shape_collision_aabb_lower,
                shape_collision_aabb_upper,
                shape_voxel_resolution,
                closing_speed * collision_update_dt,
                True,
            )
        else:
            export_and_reduce_predictive_contact(
                shape_a,
                shape_b,
                wp.vec3(pd[0], pd[1], pd[2]),
                decode_oct(reducer_data.normal[contact_id]),
                pd[3],
                surface_offset_sum,
                radius_eff_a,
                radius_eff_b,
                reducer_data.contact_fingerprints[contact_id],
                shape_transform,
                shape_linear_velocity,
                shape_angular_velocity,
                collision_update_dt,
                max_speculative_extension,
                contact_id,
                reducer_data,
            )


@wp.func
def _resolve_entry(entry_idx: int, fingerprint: int, contact_id: int, values: wp.array[wp.uint64]) -> bool:
    """Resolve the candidate's slots of one entry; returns whether it holds any of them."""
    holds = False
    for slot in range(wp.static(VALUES_PER_KEY)):
        if reduction_resolve_slot(reduction_value_index(entry_idx, slot), fingerprint, contact_id, values):
            holds = True
    return holds


# A module of its own: it launches at the narrow phase's block, the module's other kernels at the default.
@wp.kernel(enable_backward=False, module="unique")
def resolve_reduced_contacts_kernel(reducer_data: GlobalContactReducerData, speculative: int, total_num_threads: int):
    """Name every slot's winner once the pass's candidates have competed: each buffered candidate gives its contact ID
    to the slots whose keys hold its fingerprint (:func:`reduction_resolve_slot`) in the entries where it raised one
    (:attr:`GlobalContactReducerData.contact_slots`; its pair's predictive entry, found by its key, where its voxel
    word marks a raise there). Every slot's winner raised it, so it is buffered and names it; a slot no buffered
    candidate resolves (its winner found no buffer row) exports nothing. A candidate that raised nothing reads two
    words and stops. Each candidate's home entry (the first, in that order, in which it holds a slot; -1: none)
    replaces its normal-bin word, so the export writes every winner once, from its home, with no claim."""
    tid = wp.tid()
    num_contacts = wp.min(reducer_data.contact_count[0], reducer_data.capacity)
    for i in range(tid, num_contacts, total_num_threads):
        contact_id = i + 1
        entries = reducer_data.contact_slots[contact_id]
        if entries[0] < 0 and entries[1] == -1:
            continue
        fingerprint = reducer_data.contact_fingerprints[contact_id]
        home = int(-1)
        if entries[0] >= 0:
            if _resolve_entry(entries[0], fingerprint, contact_id, reducer_data.ht_values):
                home = entries[0]
        voxel_entry_idx = competed_voxel_entry(entries[1])
        if voxel_entry_idx >= 0:
            if _resolve_entry(voxel_entry_idx, fingerprint, contact_id, reducer_data.ht_values) and home < 0:
                home = voxel_entry_idx
        if speculative != 0 and entries[1] < -1:
            pair = reducer_data.shape_pairs[contact_id]
            predictive = hashtable_find(
                make_contact_key(pair[0], pair[1], wp.static(PREDICTIVE_BIN_ID)), reducer_data.ht_keys
            )
            if predictive >= 0:
                if _resolve_entry(predictive, fingerprint, contact_id, reducer_data.ht_values) and home < 0:
                    home = predictive
        if home != entries[0]:
            reducer_data.contact_slots[contact_id] = wp.vec2i(home, entries[1])


# =============================================================================
# Helper functions for contact unpacking and writing
# =============================================================================


@wp.func
def unpack_contact(
    contact_id: int,
    position_depth: wp.array[wp.vec4],
    normal: wp.array[wp.vec2],
):
    """Unpack contact data from the buffer.

    Normal is stored as octahedral-encoded vec2 and decoded back to vec3.

    Args:
        contact_id: Index into the contact buffer
        position_depth: Contact buffer for position.xyz + depth
        normal: Contact buffer for octahedral-encoded normal

    Returns:
        Tuple of (position, normal, depth)
    """
    pd = position_depth[contact_id]
    n = decode_oct(normal[contact_id])

    position = wp.vec3(pd[0], pd[1], pd[2])
    depth = pd[3]

    return position, n, depth


@wp.func
def write_contact_to_reducer(
    contact_data: Any,  # ContactData struct
    reducer_data: GlobalContactReducerData,
    output_index: int,  # Unused, kept for API compatibility with write_contact_simple
):
    """Store contacts within the query gap in GlobalContactReducer for reduction.

    This follows the same signature as write_contact_simple in narrow_phase.py,
    so it can be used with create_compute_gjk_mpr_contacts and other contact
    generation functions.

    Note: Beta threshold is applied later in create_reduce_buffered_contacts_kernel,
    not at write time. This reduces register pressure on contact generation kernels.

    Args:
        contact_data: ContactData struct from contact computation
        reducer_data: GlobalContactReducerData struct with all reducer arrays
        output_index: Unused, kept for API compatibility
    """
    # The final writer subtracts the two reconstructed world points to test
    # this same separation. Retain its near-threshold inputs, including normal
    # packing and world-coordinate roundoff, but avoid allocating clearly far
    # witnesses. The query's gap includes any speculative search extension.
    separation = contact_data.contact_distance - contact_data.margin_a - contact_data.margin_b
    excess = separation - contact_data.gap_sum
    if excess > 0.0:
        center = contact_data.contact_point_center
        scale = (
            1.0
            + wp.max(wp.abs(center[0]), wp.max(wp.abs(center[1]), wp.abs(center[2])))
            + wp.abs(contact_data.contact_distance)
            + wp.abs(contact_data.radius_eff_a)
            + wp.abs(contact_data.radius_eff_b)
            + wp.abs(contact_data.margin_a)
            + wp.abs(contact_data.margin_b)
            + wp.abs(contact_data.gap_sum)
        )
        # A conservative 32-float-epsilon allowance; the final writer still checks admission.
        if excess > 3.814697265625e-6 * scale and wp.length_sq(contact_data.contact_normal_a_to_b) > 0.0:
            return

    # Extract contact info from ContactData
    position = contact_data.contact_point_center
    normal = contact_data.contact_normal_a_to_b
    depth = contact_data.contact_distance
    shape_a = contact_data.shape_a
    shape_b = contact_data.shape_b

    # Store contact ONLY (registration to hashtable happens in a separate kernel)
    # This reduces register pressure on the contact generation kernel
    export_contact_to_buffer(
        shape_a=shape_a,
        shape_b=shape_b,
        position=position,
        normal=normal,
        depth=depth,
        fingerprint=contact_data.sort_sub_key,
        reducer_data=reducer_data,
    )


@wp.func
def _duplicate_bit_for_winners(
    slot_a: int,
    slot_b: int,
    contact_a: int,
    contact_b: int,
    position_depth: wp.array[wp.vec4],
    normal: wp.array[wp.vec2],
    contact_fingerprints: wp.array[wp.int32],
) -> int:
    """Suppress a repeated ID or the existing roundoff-equivalent loser (contact ID zero: no winner)."""
    if contact_a == 0 or contact_b == 0:
        return 0

    if contact_a == contact_b:
        return 1 << slot_b
    if not _contacts_are_numerically_equivalent(contact_a, contact_b, position_depth, normal):
        return 0

    if contact_fingerprints[contact_b] < contact_fingerprints[contact_a]:
        return 1 << slot_a
    return 1 << slot_b


@wp.func
def _roundoff_duplicate_bit_for_slot_pair(
    pair_idx: int,
    entry_idx: int,
    ht_values: wp.array[wp.uint64],
    position_depth: wp.array[wp.vec4],
    normal: wp.array[wp.vec2],
    contact_fingerprints: wp.array[wp.int32],
) -> int:
    """Return the suppression bit for one triangularly indexed slot pair."""
    slot_b = int(1)
    while pair_idx >= slot_b:
        pair_idx = pair_idx - slot_b
        slot_b = slot_b + 1
    return _duplicate_bit_for_winners(
        pair_idx,
        slot_b,
        slot_contact_id(ht_values[reduction_value_index(entry_idx, pair_idx)]),
        slot_contact_id(ht_values[reduction_value_index(entry_idx, slot_b)]),
        position_depth,
        normal,
        contact_fingerprints,
    )


def create_export_reduced_contacts_kernel(writer_func: Any, *, clear_after_export: bool = False, compact: bool = False):
    """Create a tiled kernel that exports globally reduced contacts.

    One thread block processes each active hashtable entry. Its first 21 lanes
    compare the seven possible winner pairs in parallel, preserving the
    lower-fingerprint representative for roundoff-equivalent geometry. CUDA
    lanes export the seven surviving slots concurrently; the CPU streams them
    serially from lane zero. A winner is written only from its home entry (the
    resolve's, in :attr:`GlobalContactReducerData.contact_slots`), so each is
    written once without a claim; one suppressed in its home as a roundoff copy
    of a lower-fingerprint winner is represented by that winner.

    Args:
        writer_func: Writer for contacts selected by the reducer.
        clear_after_export: Explicitly retire consumed keys and winner slots.
            This destructive option preserves raw payload, active indices/count
            and failure counters until the next collision resets them. Callers
            must finish reduction before export and must not reuse its winners.
            Standalone export remains nondestructive by default.
        compact: Use one ordinary thread per active key, reusing its seven
            winners locally. This experimental mapping requires ``wp.launch``
            and a thread count in place of the tiled mapping's block count.
    """

    @wp.func(module="unique")
    def export_contact_id(
        contact_id: int,
        entry_idx: int,
        position_depth: wp.array[wp.vec4],
        normal: wp.array[wp.vec2],
        shape_pairs: wp.array[wp.vec2i],
        contact_fingerprints: wp.array[wp.int32],
        contact_slots: wp.array[wp.vec2i],
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_gap: wp.array[float],
        writer_data: Any,
    ):
        if contact_slots[contact_id][0] != entry_idx:
            return

        position, contact_normal, depth = unpack_contact(contact_id, position_depth, normal)
        pair = shape_pairs[contact_id]
        shape_a = pair[0]
        shape_b = pair[1]
        margin_offset_a = shape_data[shape_a][3]
        margin_offset_b = shape_data[shape_b][3]
        radius_eff_a = compute_effective_radius(shape_types[shape_a], shape_data[shape_a])
        radius_eff_b = compute_effective_radius(shape_types[shape_b], shape_data[shape_b])
        gap_sum = shape_gap[shape_a] + shape_gap[shape_b]

        contact_data = ContactData()
        contact_data.contact_point_center = position
        contact_data.contact_normal_a_to_b = contact_normal
        contact_data.contact_distance = depth
        contact_data.radius_eff_a = radius_eff_a
        contact_data.radius_eff_b = radius_eff_b
        contact_data.margin_a = margin_offset_a
        contact_data.margin_b = margin_offset_b
        contact_data.shape_a = shape_a
        contact_data.shape_b = shape_b
        contact_data.gap_sum = gap_sum
        contact_data.sort_sub_key = contact_fingerprints[contact_id]
        writer_func(contact_data, writer_data, -1)

    if compact:
        winner_ids = wp.types.vector(length=VALUES_PER_KEY, dtype=wp.int32)

        @wp.kernel(enable_backward=False, module="unique")
        def export_compact_contacts_kernel(
            ht_keys: wp.array[wp.uint64],
            ht_values: wp.array[wp.uint64],
            ht_active_slots: wp.array[wp.int32],
            position_depth: wp.array[wp.vec4],
            normal: wp.array[wp.vec2],
            shape_pairs: wp.array[wp.vec2i],
            contact_fingerprints: wp.array[wp.int32],
            contact_slots: wp.array[wp.vec2i],
            shape_types: wp.array[int],
            shape_data: wp.array[wp.vec4],
            shape_gap: wp.array[float],
            writer_data: Any,
            total_num_threads: int,
            parallel_pairs: int,
        ):
            tid = wp.tid()
            ht_capacity = ht_keys.shape[0]
            num_active = ht_active_slots[ht_capacity]
            for active_idx in range(tid, num_active, total_num_threads):
                entry_idx = ht_active_slots[active_idx]
                ids = winner_ids()
                for slot in range(wp.static(VALUES_PER_KEY)):
                    ids[slot] = wp.int32(slot_contact_id(ht_values[reduction_value_index(entry_idx, slot)]))

                duplicate_mask = int(0)
                for slot_b in range(1, wp.static(VALUES_PER_KEY)):
                    # An early continue keeps this fixed loop dynamic in Warp,
                    # forcing the winner vector into local memory.
                    if ids[slot_b] != 0:
                        for slot_a in range(slot_b):
                            duplicate_mask = duplicate_mask | _duplicate_bit_for_winners(
                                slot_a,
                                slot_b,
                                int(ids[slot_a]),
                                int(ids[slot_b]),
                                position_depth,
                                normal,
                                contact_fingerprints,
                            )

                for slot in range(wp.static(VALUES_PER_KEY)):
                    contact_id = int(ids[slot])
                    if contact_id != 0 and duplicate_mask & (1 << slot) == 0:
                        export_contact_id(
                            contact_id,
                            entry_idx,
                            position_depth,
                            normal,
                            shape_pairs,
                            contact_fingerprints,
                            contact_slots,
                            shape_types,
                            shape_data,
                            shape_gap,
                            writer_data,
                        )

                if wp.static(clear_after_export):
                    # Active keys are unique; this thread has finished all
                    # winner reads. Other keys' winners retain their raw payload.
                    for slot in range(wp.static(VALUES_PER_KEY)):
                        ht_values[reduction_value_index(entry_idx, slot)] = wp.uint64(0)
                    ht_keys[entry_idx] = HASHTABLE_EMPTY_KEY

        return export_compact_contacts_kernel

    @wp.kernel(enable_backward=False, module="unique")
    def export_reduced_contacts_kernel(
        ht_keys: wp.array[wp.uint64],
        ht_values: wp.array[wp.uint64],
        ht_active_slots: wp.array[wp.int32],
        position_depth: wp.array[wp.vec4],
        normal: wp.array[wp.vec2],
        shape_pairs: wp.array[wp.vec2i],
        contact_fingerprints: wp.array[wp.int32],
        contact_slots: wp.array[wp.vec2i],
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_gap: wp.array[float],
        writer_data: Any,
        total_num_blocks: int,
        parallel_pairs: int,
    ):
        block_id, lane = wp.tid()
        ht_capacity = ht_keys.shape[0]
        num_active = ht_active_slots[ht_capacity]
        duplicate_bits = wp.tile_zeros(shape=wp.static(EXPORT_REDUCED_CONTACTS_BLOCK_DIM), dtype=int, storage="shared")

        for active_idx in range(block_id, num_active, total_num_blocks):
            entry_idx = ht_active_slots[active_idx]
            duplicate_bit = int(0)

            if parallel_pairs != 0:
                if lane < wp.static(PAIRS_PER_KEY):
                    duplicate_bit = _roundoff_duplicate_bit_for_slot_pair(
                        lane,
                        entry_idx,
                        ht_values,
                        position_depth,
                        normal,
                        contact_fingerprints,
                    )
            elif lane == 0:
                for pair_idx in range(wp.static(PAIRS_PER_KEY)):
                    duplicate_bit = duplicate_bit | _roundoff_duplicate_bit_for_slot_pair(
                        pair_idx,
                        entry_idx,
                        ht_values,
                        position_depth,
                        normal,
                        contact_fingerprints,
                    )

            wp.tile_scatter_masked(duplicate_bits, lane, duplicate_bit, True)
            duplicate_mask = wp.tile_reduce(wp.bit_or, duplicate_bits)[0]

            if parallel_pairs != 0:
                if lane < wp.static(VALUES_PER_KEY) and duplicate_mask & (1 << lane) == 0:
                    contact_id = slot_contact_id(ht_values[reduction_value_index(entry_idx, lane)])
                    if contact_id != 0:
                        export_contact_id(
                            contact_id,
                            entry_idx,
                            position_depth,
                            normal,
                            shape_pairs,
                            contact_fingerprints,
                            contact_slots,
                            shape_types,
                            shape_data,
                            shape_gap,
                            writer_data,
                        )
            elif lane == 0:
                for slot in range(wp.static(VALUES_PER_KEY)):
                    if duplicate_mask & (1 << slot) != 0:
                        continue
                    contact_id = slot_contact_id(ht_values[reduction_value_index(entry_idx, slot)])
                    if contact_id == 0:
                        continue
                    # A repeated winner within this entry is suppressed above;
                    # one held in several entries is written from its home.
                    export_contact_id(
                        contact_id,
                        entry_idx,
                        position_depth,
                        normal,
                        shape_pairs,
                        contact_fingerprints,
                        contact_slots,
                        shape_types,
                        shape_data,
                        shape_gap,
                        writer_data,
                    )

            # Clear cooperatively so lane 0 finishes exporting before any warp
            # can reuse the shared tile for the next active entry.
            wp.tile_scatter_masked(duplicate_bits, lane, int(0), True)

            if wp.static(clear_after_export):
                # The tile barrier above finishes all reads of this entry.
                # Other entries read the raw payload and home entries, which
                # remain live. Keep counts and failures for later inspection.
                if parallel_pairs != 0:
                    if lane < wp.static(VALUES_PER_KEY):
                        value_idx = reduction_value_index(entry_idx, lane)
                        if ht_values[value_idx] != wp.uint64(0):
                            ht_values[value_idx] = wp.uint64(0)
                elif lane == 0:
                    for slot in range(wp.static(VALUES_PER_KEY)):
                        value_idx = reduction_value_index(entry_idx, slot)
                        if ht_values[value_idx] != wp.uint64(0):
                            ht_values[value_idx] = wp.uint64(0)
                if lane == 0:
                    ht_keys[entry_idx] = HASHTABLE_EMPTY_KEY

    return export_reduced_contacts_kernel


MESH_ROUND_ALL = 0
MESH_ROUND_ONLY = 1
MESH_ROUND_NONE = 2

SURFACES_ALL = 0
SURFACES_MESH = 1
SURFACES_HEIGHTFIELD = 2


def create_mesh_triangle_contacts_to_reducer_kernel(
    primitive_type: int = 0,
    exclude_primitives: bool = False,
    *,
    skip_marked: bool = False,
    writer_func=write_contact_to_reducer,
    writer_binding=None,
    writer_data_type=GlobalContactReducerData,
    kernel_name=None,
    analytic: bool | None = None,
    mesh_round: int = MESH_ROUND_ALL,
    plane_cull: bool = True,
    surfaces: int = SURFACES_ALL,
):
    """Create the original query or a filtered heightfield primitive variant.

    Args:
        surfaces: Which surfaces' triangle pairs the query takes: all (``SURFACES_ALL``), meshes only
            (``SURFACES_MESH``) or heightfields only (``SURFACES_HEIGHTFIELD``).
        plane_cull: Skip a mesh triangle whose plane the partner clears (off for the queue the face kernel already
            culled, :func:`create_mesh_face_contacts_kernel`).
        mesh_round: Which mesh-triangle pairs the generic query takes: all (``MESH_ROUND_ALL``), only those
            with a sphere or capsule (``MESH_ROUND_ONLY``: their closed form, in a lean kernel), or all but those
            (``MESH_ROUND_NONE``: GJK/MPR, beside the lean kernel). Heightfield pairs are in every generic
            variant but the lean one.
        analytic: Use the closed-form cell contact of ``ANALYTIC_PRIMITIVE_QUERIES``
            for a typed primitive query that has one, or, in the generic
            fallback that still serves heightfield convex pairs, for those
            pairs inside the same kernel (every typed kernel is one more scan
            of the whole candidate list, so convex hulls do not get their own).
            ``None`` reads ``NEWTON_HEIGHTFIELD_ANALYTIC_PRIMITIVES`` (default on)
            and, for the convex row, ``NEWTON_HEIGHTFIELD_ANALYTIC_CONVEX``
            (default off); an explicit ``True`` enables both.
    """
    analytic_convex = analytic if analytic is not None else analytic_convex_default()
    if analytic is None:
        analytic = analytic_primitives_default()
    analytic = bool(analytic)
    convex_in_fallback = exclude_primitives
    return _create_mesh_triangle_contacts_to_reducer_kernel(
        primitive_type,
        exclude_primitives,
        skip_marked=skip_marked,
        writer_func=writer_func,
        writer_binding=writer_binding,
        writer_data_type=writer_data_type,
        kernel_name=kernel_name,
        analytic=analytic and primitive_type in ANALYTIC_PRIMITIVE_QUERIES,
        analytic_convex=bool(analytic_convex)
        and convex_in_fallback
        and int(GeoType.CONVEX_MESH) in ANALYTIC_PRIMITIVE_QUERIES,
        mesh_round=mesh_round,
        plane_cull=plane_cull,
        surfaces=surfaces,
    )


HEIGHTFIELD_PRIMITIVE_SUPPORTS = {
    int(GeoType.SPHERE): support_map_heightfield_sphere,
    int(GeoType.CAPSULE): support_map_heightfield_capsule,
    int(GeoType.CYLINDER): support_map_heightfield_cylinder,
    int(GeoType.BOX): support_map_heightfield_box,
    int(GeoType.CONVEX_MESH): support_map,
}
"""Typed heightfield primitive queries: each type owns a reduced support dispatch."""


@wp.func
def heightfield_typed_primitive(shape_type: int, shape_scale: wp.vec4) -> bool:
    """Whether a heightfield pair with this convex shape is owned by a typed primitive query."""
    if shape_type == GeoType.SPHERE or shape_type == GeoType.CAPSULE or shape_type == GeoType.BOX:
        return True
    return shape_type == GeoType.CYLINDER and shape_scale[2] == 0.0


@cache
def _create_mesh_triangle_contacts_to_reducer_kernel(
    primitive_type: int = 0,
    exclude_primitives: bool = False,
    *,
    skip_marked: bool = False,
    writer_func=write_contact_to_reducer,
    writer_binding=None,
    writer_data_type=GlobalContactReducerData,
    kernel_name=None,
    analytic: bool = False,
    analytic_convex: bool = False,
    mesh_round: int = MESH_ROUND_ALL,
    plane_cull: bool = True,
    surfaces: int = SURFACES_ALL,
):
    if mesh_round != MESH_ROUND_ALL and (primitive_type != 0 or analytic):
        raise ValueError("Only the generic query splits its mesh round-shape pairs")
    if primitive_type not in (0, *HEIGHTFIELD_PRIMITIVE_SUPPORTS):
        raise ValueError("Only sphere, capsule, cylinder, box and convex-mesh heightfield queries can be specialized")
    if primitive_type and exclude_primitives:
        raise ValueError("A typed query cannot also exclude primitive pairs")
    # Convex hulls take the cooked accelerated support (directional seeds and vertex adjacency);
    # the typed primitive queries keep their reduced support dispatch.
    accelerated = primitive_type in (0, int(GeoType.CONVEX_MESH))
    query_support = support_map_accelerated_inline if accelerated else HEIGHTFIELD_PRIMITIVE_SUPPORTS[primitive_type]
    query_filter = seam_surface_witness
    if analytic:
        pair_query = ANALYTIC_PRIMITIVE_QUERIES[primitive_type](writer_func, query_filter=query_filter)
    elif primitive_type == 0:
        # Mesh and heightfield triangles: the surface's feature rule on every mesh contact.
        pair_query = create_compute_gjk_mpr_contacts(
            writer_func,
            support_func=query_support,
            use_precomputed_center=True,
            penetration_refiner=create_surface_penetration_refiner(query_support),
            query_filter=surface_query_filter,
            point_filter=mesh_contact_filter,
            manifold_shape=mesh_manifold_triangle,
            deepest_once=mesh_deepest_once,
        )
    else:
        pair_query = create_compute_gjk_mpr_contacts(
            writer_func,
            support_func=query_support,
            use_precomputed_center=True,
            penetration_refiner=create_triangle_prism_penetration_refiner(query_support),
            query_filter=query_filter,
        )
    mesh_sphere_contact = create_mesh_sphere_contact(writer_func)
    mesh_capsule_contacts = create_mesh_capsule_contacts(writer_func)
    # The fallback keeps heightfield convex pairs in its own scan; with the
    # closed-form row on, those pairs take the row here instead of GJK/MPR.
    convex_query = (
        ANALYTIC_PRIMITIVE_QUERIES[int(GeoType.CONVEX_MESH)](writer_func, query_filter=query_filter)
        if analytic_convex
        else pair_query
    )

    def mesh_triangle_contacts_to_reducer_kernel(
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_gap: wp.array[float],
        shape_heightfield_index: wp.array[wp.int32],
        heightfield_data: wp.array[HeightfieldData],
        heightfield_elevations: wp.array[wp.float32],
        triangle_pairs: wp.array[wp.vec3i],
        triangle_pairs_count: wp.array[int],
        reducer_data: writer_data_type,
        total_num_threads: int,
        # Convex-hull support acceleration and interior seeds (MESH_TRIANGLE_SUPPORT_INPUTS order).
        shape_support_data: wp.array[wp.vec4i],
        support_lut: wp.array[int],
        support_vertex_offsets: wp.array[int],
        support_neighbors: wp.array[int],
        shape_interior_point: wp.array[wp.vec3],
        mesh_topology: MeshTopologyData,
    ):
        """Process mesh/heightfield-triangle contacts and store them in GlobalContactReducer.

        This kernel processes triangle pairs (mesh-or-hfield shape, convex-shape, triangle_index)
        and computes contacts using exact sphere-triangle geometry or GJK/MPR, storing results
        in the GlobalContactReducer for subsequent reduction and export.

        Uses grid stride loop over triangle pairs.
        """
        tid = wp.tid()

        num_triangle_pairs = triangle_pairs_count[0]

        for i in range(tid, num_triangle_pairs, total_num_threads):
            if i >= triangle_pairs.shape[0]:
                break

            # Preserve the original argument in the unbound specialization. The
            # binding is a plain closure call: the dead branch is never compiled,
            # while wp.static(None) would be unresolvable, and Warp builds a kernel
            # with an unresolvable static at declaration (at import, for this one).
            if wp.static(writer_binding is not None):
                reducer_data = writer_binding(reducer_data, i)
            triple = triangle_pairs[i]
            shape_a = triple[0]  # Mesh or heightfield shape
            shape_b = triple[1]  # Convex shape
            tri_idx = triple[2]
            if wp.static(skip_marked):
                if tri_idx < 0:
                    continue

            type_a = shape_types[shape_a]
            if wp.static(surfaces == SURFACES_MESH):
                if type_a != GeoType.MESH:
                    continue
            elif wp.static(surfaces == SURFACES_HEIGHTFIELD):
                if type_a != GeoType.HFIELD:
                    continue
            if wp.static(mesh_round != MESH_ROUND_ALL):
                round_pair = type_a == GeoType.MESH and (
                    shape_types[shape_b] == GeoType.SPHERE or shape_types[shape_b] == GeoType.CAPSULE
                )
                if wp.static(mesh_round == MESH_ROUND_ONLY):
                    if not round_pair:
                        continue
                elif round_pair:
                    continue
            if wp.static(primitive_type != 0):
                if type_a != GeoType.HFIELD or shape_types[shape_b] != primitive_type:
                    continue
                if wp.static(primitive_type == int(GeoType.CYLINDER)):
                    if shape_data[shape_b][2] != 0.0:
                        continue
                type_a = wp.static(int(GeoType.HFIELD))
            elif wp.static(exclude_primitives):
                if type_a == GeoType.HFIELD and heightfield_typed_primitive(shape_types[shape_b], shape_data[shape_b]):
                    continue

            corner_0 = wp.vec3(0.0)
            corner_1 = wp.vec3(0.0)
            corner_2 = wp.vec3(0.0)
            if wp.static(primitive_type != 0):
                hfd = heightfield_data[shape_heightfield_index[shape_a]]
                X_ws_a = shape_transform[shape_a]
                shape_data_a, v0_world = get_triangle_shape_from_heightfield(
                    hfd, heightfield_elevations, X_ws_a, tri_idx
                )
            elif type_a == GeoType.HFIELD:
                # Heightfield triangle
                hfd = heightfield_data[shape_heightfield_index[shape_a]]
                X_ws_a = shape_transform[shape_a]
                shape_data_a, v0_world = get_triangle_shape_from_heightfield(
                    hfd, heightfield_elevations, X_ws_a, tri_idx
                )
            else:
                # Mesh triangle (mesh_id already validated by midphase)
                mesh_id_a = shape_source[shape_a]
                scale_data_a = shape_data[shape_a]
                mesh_scale_a = wp.vec3(scale_data_a[0], scale_data_a[1], scale_data_a[2])
                X_ws_a = shape_transform[shape_a]
                corner_0, corner_1, corner_2 = mesh_triangle_corners(mesh_id_a, mesh_scale_a, X_ws_a, tri_idx)
                shape_data_a = triangle_shape_from_corners(corner_0, corner_1, corner_2, mesh_scale_a)
                v0_world = corner_0

            # Extract shape B data
            pos_b, quat_b, shape_data_b, _scale_b, margin_offset_b = extract_shape_data(
                shape_b,
                shape_transform,
                shape_types,
                shape_data,
                shape_source,
            )
            if wp.static(primitive_type != 0):
                # The current pair passed the exact type filter above. Keep
                # that fact visible when compiling the shared query math.
                shape_data_b.shape_type = wp.static(primitive_type)
            if shape_data_b.shape_type == GeoType.CONVEX_MESH:
                # MPR/GJK need an interior seed. A hull's AABB center can lie
                # outside it and flip the contact against thin triangles.
                shape_data_b.center = shape_interior_point[shape_b]

            # Triangle position is vertex A in world space.
            # For heightfield prisms, edges are in heightfield-local space
            # so we pass the heightfield rotation to let MPR/GJK work in
            # that frame (where -Z is always the down axis).
            pos_a = v0_world
            if type_a == GeoType.HFIELD:
                quat_a = wp.transform_get_rotation(shape_transform[shape_a])
            else:
                quat_a = wp.quat_identity()

            # Mesh midphase already applies topology-aware back-face admission.
            # Repeating a center-only test would discard closed-surface overlaps.

            # Extract margin offset for shape A (signed distance padding)
            margin_offset_a = shape_data[shape_a][3]

            # Use additive per-shape contact gap for detection threshold
            gap_a = shape_gap[shape_a]
            gap_b = shape_gap[shape_b]
            gap_sum = gap_a + gap_b

            if wp.static(accelerated):
                data_provider = AcceleratedSupportMapDataProvider()
                data_provider.shape_support_data = shape_support_data
                data_provider.support_neighbors = support_neighbors
                data_provider.support_lut = support_lut
                data_provider.support_vertex_offsets = support_vertex_offsets
            else:
                data_provider = SupportMapDataProvider()

            feature_context = HeightfieldFeatureContext()
            if type_a == GeoType.HFIELD:
                feature_context.elevations = heightfield_elevations
                feature_context.triangle = tri_idx
                feature_context.heightfield = heightfield_data[shape_heightfield_index[shape_a]]

            if wp.static(primitive_type == 0 and mesh_round != MESH_ROUND_ONLY and plane_cull):
                # Mesh triangles against other shapes: skip a triangle whose plane the shape clears by more than the
                # contact threshold, before anything else is read (spheres and capsules have their own exits).
                if type_a == GeoType.MESH and shape_data_b.shape_type != GeoType.SPHERE:
                    if shape_data_b.shape_type != GeoType.CAPSULE:
                        face = wp.cross(shape_data_a.scale, shape_data_a.auxiliary)
                        support_b = wp.static(query_support)(
                            shape_data_b, wp.quat_rotate_inv(quat_b, -face), data_provider
                        )
                        if triangle_plane_clears(
                            shape_data_a,
                            pos_a,
                            wp.quat_rotate(quat_b, support_b) + pos_b,
                            gap_sum + margin_offset_a + margin_offset_b,
                        ):
                            continue

            if wp.static(primitive_type == 0):
                # Heightfield triangles have no mesh: their context reads no mesh data.
                surface_mesh = wp.uint64(0)
                if type_a == GeoType.MESH:
                    surface_mesh = shape_source[shape_a]
                surface = mesh_surface_context(
                    mesh_topology,
                    shape_a,
                    tri_idx,
                    surface_mesh,
                    wp.vec3(shape_data[shape_a][0], shape_data[shape_a][1], shape_data[shape_a][2]),
                    shape_transform[shape_a],
                    corner_0,
                    corner_1,
                    corner_2,
                    pos_b + wp.quat_rotate(quat_b, shape_data_b.center),
                    shape_b,
                )
                surface.heightfield = feature_context
                if type_a != GeoType.HFIELD:
                    # Mesh triangles: spheres and capsules in closed form.
                    threshold = gap_sum + margin_offset_a + margin_offset_b
                    if wp.static(mesh_round != MESH_ROUND_NONE):
                        if shape_data_b.shape_type == GeoType.SPHERE:
                            mesh_sphere_contact(
                                surface,
                                pos_b,
                                shape_data_b.scale[0],
                                threshold,
                                margin_offset_a,
                                margin_offset_b,
                                shape_a,
                                shape_b,
                                gap_sum,
                                (tri_idx << 1) | 1,
                                reducer_data,
                            )
                            continue
                        if shape_data_b.shape_type == GeoType.CAPSULE:
                            core = wp.quat_rotate(quat_b, wp.vec3(0.0, 0.0, shape_data_b.scale[1]))
                            mesh_capsule_contacts(
                                surface,
                                pos_b - core,
                                pos_b + core,
                                shape_data_b.scale[0],
                                threshold,
                                margin_offset_a,
                                margin_offset_b,
                                shape_a,
                                shape_b,
                                gap_sum,
                                (tri_idx << 1) | 1,
                                reducer_data,
                            )
                            continue
                    if wp.static(mesh_round == MESH_ROUND_ONLY):
                        continue

            if wp.static(mesh_round != MESH_ROUND_ONLY):
                if wp.static(analytic_convex):
                    if type_a == GeoType.HFIELD and shape_data_b.shape_type == GeoType.CONVEX_MESH:
                        # The closed-form convex row only exists for heightfields; this branch compiles
                        # when it is on, and it takes no support provider.
                        wp.static(convex_query)(
                            shape_data_a,
                            shape_data_b,
                            quat_a,
                            quat_b,
                            pos_a,
                            pos_b,
                            gap_sum,
                            shape_a,
                            shape_b,
                            margin_offset_a,
                            margin_offset_b,
                            reducer_data,
                            (tri_idx << 1) | 1,
                            feature_context,
                        )
                        continue
                if wp.static(analytic):
                    wp.static(pair_query)(
                        shape_data_a,
                        shape_data_b,
                        quat_a,
                        quat_b,
                        pos_a,
                        pos_b,
                        gap_sum,
                        shape_a,
                        shape_b,
                        margin_offset_a,
                        margin_offset_b,
                        reducer_data,
                        (tri_idx << 1) | 1,
                        feature_context,
                    )
                elif wp.static(primitive_type == 0):
                    surface = mesh_surface_planes(surface)
                    wp.static(pair_query)(
                        shape_data_a,
                        shape_data_b,
                        quat_a,
                        quat_b,
                        pos_a,
                        pos_b,
                        gap_sum,
                        shape_a,
                        shape_b,
                        margin_offset_a,
                        margin_offset_b,
                        data_provider,
                        reducer_data,
                        (tri_idx << 1) | 1,
                        surface,
                    )
                else:
                    wp.static(pair_query)(
                        shape_data_a,
                        shape_data_b,
                        quat_a,
                        quat_b,
                        pos_a,
                        pos_b,
                        gap_sum,
                        shape_a,
                        shape_b,
                        margin_offset_a,
                        margin_offset_b,
                        data_provider,
                        reducer_data,
                        (tri_idx << 1) | 1,
                        feature_context,
                    )

    if primitive_type == int(GeoType.SPHERE):
        mesh_triangle_contacts_to_reducer_kernel.__name__ = "heightfield_sphere_contacts_to_reducer_kernel"
    elif primitive_type == int(GeoType.CAPSULE):
        mesh_triangle_contacts_to_reducer_kernel.__name__ = "heightfield_capsule_contacts_to_reducer_kernel"
    elif primitive_type == int(GeoType.CYLINDER):
        mesh_triangle_contacts_to_reducer_kernel.__name__ = "heightfield_cylinder_contacts_to_reducer_kernel"
    elif primitive_type == int(GeoType.BOX):
        mesh_triangle_contacts_to_reducer_kernel.__name__ = "heightfield_box_contacts_to_reducer_kernel"
    elif primitive_type == int(GeoType.CONVEX_MESH):
        mesh_triangle_contacts_to_reducer_kernel.__name__ = "heightfield_convex_contacts_to_reducer_kernel"
    if analytic:
        mesh_triangle_contacts_to_reducer_kernel.__name__ += "_analytic"
    elif exclude_primitives:
        mesh_triangle_contacts_to_reducer_kernel.__name__ = "mesh_triangle_contacts_to_reducer_fallback_kernel"
    if analytic_convex:
        mesh_triangle_contacts_to_reducer_kernel.__name__ += "_analytic_convex"
    if skip_marked:
        mesh_triangle_contacts_to_reducer_kernel.__name__ += "_finite_fallback"
    if mesh_round == MESH_ROUND_ONLY:
        mesh_triangle_contacts_to_reducer_kernel.__name__ = "mesh_round_contacts_to_reducer_kernel"
    elif mesh_round == MESH_ROUND_NONE:
        mesh_triangle_contacts_to_reducer_kernel.__name__ += "_without_mesh_round"
    if not plane_cull:
        mesh_triangle_contacts_to_reducer_kernel.__name__ += "_queued"
    if surfaces == SURFACES_MESH:
        mesh_triangle_contacts_to_reducer_kernel.__name__ += "_meshes"
    elif surfaces == SURFACES_HEIGHTFIELD:
        mesh_triangle_contacts_to_reducer_kernel.__name__ += "_heightfields"
    if kernel_name is not None:
        mesh_triangle_contacts_to_reducer_kernel.__name__ = kernel_name
    mesh_triangle_contacts_to_reducer_kernel.__qualname__ = mesh_triangle_contacts_to_reducer_kernel.__name__
    mesh_triangle_contacts_to_reducer_kernel.__annotations__["reducer_data"] = writer_data_type
    return wp.kernel(mesh_triangle_contacts_to_reducer_kernel, enable_backward=False, module="unique")


mesh_triangle_contacts_to_reducer_kernel = create_mesh_triangle_contacts_to_reducer_kernel()


FACE_CASE_TYPES = (int(GeoType.BOX), int(GeoType.CYLINDER), int(GeoType.CONVEX_MESH))
"""Partners whose mesh-triangle contacts the face kernel decides in closed form where the face case holds."""


@wp.func
def face_case_partner(shape_type: int) -> bool:
    """Whether the face kernel takes this partner's mesh-triangle pairs (see ``FACE_CASE_TYPES``)."""
    return shape_type == GeoType.BOX or shape_type == GeoType.CYLINDER or shape_type == GeoType.CONVEX_MESH


@cache
def create_mesh_face_contacts_kernel(
    writer_func=write_contact_to_reducer,
    writer_data_type=GlobalContactReducerData,
    queue_heightfields: bool = True,
    kernel_name: str = "mesh_face_contacts_kernel",
):
    """Create the first pass over mesh triangle pairs: the plane cull and the closed-form face cases.

    A box, cylinder or convex hull whose support point along minus the face normal has its foot inside the
    triangle (:func:`triangle_face_case`) gets the face-normal manifold here, as GJK/MPR would give it; one on the
    outer side of the plane of a triangle inside a facet (:func:`triangle_facet_case`) gets the facet plane's
    manifold, of which the triangle keeps the points over it. Every
    other pair that survives the cull, and every heightfield pair, is compacted into ``queued_pairs`` for the
    GJK/MPR kernel (the generic query with ``plane_cull=False``), so the closed-form majority does not wait on
    GJK/MPR in a warp. Sphere and capsule pairs are the lean round-shape kernel's. With ``queue_heightfields`` off,
    heightfield pairs are left to a query of their own.
    """
    face_contacts = create_compute_face_contacts(
        writer_func,
        support_map_accelerated_inline,
        mesh_contact_filter,
        mesh_manifold_triangle,
        deepest_once=mesh_deepest_once,
    )

    def mesh_face_contacts_kernel(
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_gap: wp.array[float],
        shape_heightfield_index: wp.array[wp.int32],
        heightfield_data: wp.array[HeightfieldData],
        heightfield_elevations: wp.array[wp.float32],
        triangle_pairs: wp.array[wp.vec3i],
        triangle_pairs_count: wp.array[int],
        writer_data: writer_data_type,
        total_num_threads: int,
        shape_support_data: wp.array[wp.vec4i],
        support_lut: wp.array[int],
        support_vertex_offsets: wp.array[int],
        support_neighbors: wp.array[int],
        shape_interior_point: wp.array[wp.vec3],
        mesh_topology: MeshTopologyData,
        queued_pairs: wp.array[wp.vec3i],
        queued_count: wp.array[int],
    ):
        tid = wp.tid()
        num_triangle_pairs = triangle_pairs_count[0]
        for i in range(tid, num_triangle_pairs, total_num_threads):
            if i >= triangle_pairs.shape[0]:
                break
            triple = triangle_pairs[i]
            shape_a = triple[0]
            shape_b = triple[1]
            tri_idx = triple[2]
            type_b = shape_types[shape_b]
            if shape_types[shape_a] == GeoType.MESH:
                if type_b == GeoType.SPHERE or type_b == GeoType.CAPSULE:
                    continue
                scale_data_a = shape_data[shape_a]
                mesh_scale_a = wp.vec3(scale_data_a[0], scale_data_a[1], scale_data_a[2])
                X_ws_a = shape_transform[shape_a]
                corner_0, corner_1, corner_2 = mesh_triangle_corners(
                    shape_source[shape_a], mesh_scale_a, X_ws_a, tri_idx
                )
                triangle = triangle_shape_from_corners(corner_0, corner_1, corner_2, mesh_scale_a)
                pos_b, quat_b, geom_b, _scale_b, margin_b = extract_shape_data(
                    shape_b, shape_transform, shape_types, shape_data, shape_source
                )
                if geom_b.shape_type == GeoType.CONVEX_MESH:
                    geom_b.center = shape_interior_point[shape_b]
                margin_a = scale_data_a[3]
                gap_sum = shape_gap[shape_a] + shape_gap[shape_b]
                threshold = gap_sum + margin_a + margin_b
                provider = AcceleratedSupportMapDataProvider()
                provider.shape_support_data = shape_support_data
                provider.support_neighbors = support_neighbors
                provider.support_lut = support_lut
                provider.support_vertex_offsets = support_vertex_offsets
                face = wp.cross(triangle.scale, triangle.auxiliary)
                support_point = (
                    wp.quat_rotate(
                        quat_b, support_map_accelerated_inline(geom_b, wp.quat_rotate_inv(quat_b, -face), provider)
                    )
                    + pos_b
                )
                if triangle_plane_clears(triangle, corner_0, support_point, threshold):
                    continue
                if face_case_partner(type_b):
                    center = pos_b + wp.quat_rotate(quat_b, geom_b.center)
                    is_face, unit, distance = triangle_face_case(triangle, corner_0, support_point, center, threshold)
                    surface = mesh_surface_context(
                        mesh_topology,
                        shape_a,
                        tri_idx,
                        shape_source[shape_a],
                        mesh_scale_a,
                        X_ws_a,
                        corner_0,
                        corner_1,
                        corner_2,
                        center,
                        shape_b,
                    )
                    if is_face or triangle_facet_case(
                        mesh_facet_interior(surface), unit, distance, mpr_inflate(margin_a + margin_b)
                    ):
                        surface = mesh_surface_planes(surface)
                        face_contacts(
                            triangle,
                            geom_b,
                            quat_b,
                            corner_0,
                            pos_b,
                            unit,
                            support_point,
                            distance,
                            gap_sum,
                            shape_a,
                            shape_b,
                            margin_a,
                            margin_b,
                            provider,
                            writer_data,
                            (tri_idx << 1) | 1,
                            surface,
                        )
                        continue
            elif wp.static(not queue_heightfields):
                continue
            slot = wp.atomic_add(queued_count, 0, 1)
            if slot < queued_pairs.shape[0]:
                queued_pairs[slot] = triple

    mesh_face_contacts_kernel.__name__ = kernel_name
    mesh_face_contacts_kernel.__qualname__ = kernel_name
    mesh_face_contacts_kernel.__annotations__["writer_data"] = writer_data_type
    return wp.kernel(mesh_face_contacts_kernel, enable_backward=False, module="unique")
