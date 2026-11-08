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

**Implementation Details:**

- Contacts stored in global buffer (struct of arrays: position_depth, normal, shape_pairs)
- Hashtable key: ``(shape_a, shape_b, bin_id)`` where ``bin_id`` is
  ``0..NUM_NORMAL_BINS-1`` for normal bins and higher indices for voxel groups
- Each normal bin entry has ``NUM_SPATIAL_DIRECTIONS + 1`` value slots
- Voxels are grouped by ``NUM_SPATIAL_DIRECTIONS + 1``:
  ``bin_id = NUM_NORMAL_BINS + (voxel_idx // group_size)``
- Atomic max on packed (score, contact_id) selects winners

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
from .collision_core import (
    create_compute_gjk_mpr_contacts,
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
    mesh_manifold_triangle,
    mesh_surface_context,
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
    support_map_accelerated,
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
# Values are stored entry-major: each key owns one row of VALUES_STRIDE words,
# of which the first VALUES_PER_KEY hold its winners. The last word only pads
# the row: no writer addresses it, so it stays zero. Eight uint64 words make a
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
# Reduction slot functions (specific to contact reduction)
# =============================================================================
# These functions handle the entry-major value storage used for contact
# reduction. Each hashtable entry owns one row of VALUES_STRIDE values:
# [entry0_slot0, ..., entry0_slot7, entry1_slot0, ..., entry1_slot7, ...]
# Slots 0 to VALUES_PER_KEY - 1 hold winners; slot 7 is padding that stays
# zero. Every access addresses a value through reduction_value_index (device)
# or reduction_value_index_np (host).


@wp.func
def reduction_value_index(entry_idx: int, slot_id: int) -> int:
    """Return the position of one entry's value slot in the entry-major values array.

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
def reduction_update_slot(
    entry_idx: int,
    slot_id: int,
    value: wp.uint64,
    values: wp.array[wp.uint64],
):
    """Update a reduction slot using atomic max.

    Use this after hashtable_find_or_insert() to write multiple values
    to the same entry without repeated hash lookups.

    Args:
        entry_idx: Entry index from hashtable_find_or_insert()
        slot_id: Which value slot to write to (0 to ``VALUES_PER_KEY - 1``)
        value: The uint64 value to max with existing value
        values: Values array in entry-major layout
    """
    value_idx = reduction_value_index(entry_idx, slot_id)
    # Check before atomic to reduce contention
    if values[value_idx] < value:
        wp.atomic_max(values, value_idx, value)


@wp.func
def reduction_try_update_slot(
    entry_idx: int,
    slot_id: int,
    value: wp.uint64,
    values: wp.array[wp.uint64],
) -> wp.uint64:
    """Atomically update a slot and return the value that occupied it."""
    value_idx = reduction_value_index(entry_idx, slot_id)
    previous_value = values[value_idx]
    if previous_value >= value:
        return previous_value
    return wp.atomic_max(values, value_idx, value)


@wp.func
def reduction_rollback_slot(
    entry_idx: int,
    slot_id: int,
    provisional_value: wp.uint64,
    previous_value: wp.uint64,
    values: wp.array[wp.uint64],
):
    """Restore a displaced value without overwriting a newer winner."""
    value_idx = reduction_value_index(entry_idx, slot_id)
    wp.atomic_cas(values, value_idx, provisional_value, previous_value)


@wp.func
def reduction_finalize_slot(
    entry_idx: int,
    slot_id: int,
    provisional_value: wp.uint64,
    final_value: wp.uint64,
    values: wp.array[wp.uint64],
) -> bool:
    """Atomically transfer a provisional slot claim to its materialized contact."""
    value_idx = reduction_value_index(entry_idx, slot_id)
    return wp.atomic_cas(values, value_idx, provisional_value, final_value) == provisional_value


@wp.func
def reduction_insert_slot(
    key: wp.uint64,
    slot_id: int,
    value: wp.uint64,
    keys: wp.array[wp.uint64],
    values: wp.array[wp.uint64],
    active_slots: wp.array[wp.int32],
) -> bool:
    """Insert or update a value in a specific reduction slot.

    Convenience function that combines hashtable_find_or_insert()
    and reduction_update_slot(). For inserting multiple values to
    the same key, prefer using those functions separately.

    Args:
        key: The uint64 key to insert
        slot_id: Which value slot to write to (0 to ``VALUES_PER_KEY - 1``)
        value: The uint64 value to insert or max with
        keys: The hash table keys array (length must be power of two)
        values: Values array in entry-major layout, ``VALUES_STRIDE`` values per key
        active_slots: Array of size (capacity + 1) tracking active entry indices.

    Returns:
        True if insertion/update succeeded, False if the table is full
    """
    entry_idx = hashtable_find_or_insert(key, keys, active_slots)
    if entry_idx < 0:
        return False
    reduction_update_slot(entry_idx, slot_id, value, values)
    return True


# =============================================================================
# Contact key/value packing
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


# ---------------------------------------------------------------------------
# Contact value packing
# ---------------------------------------------------------------------------
# Two packing modes exist — **fast** (default) and **deterministic**.
# Each variant is a standalone ``@wp.func``.  The dispatching wrappers
# ``make_contact_value`` and ``unpack_contact_id`` select the variant
# at runtime based on a ``deterministic`` flag passed from
# ``GlobalContactReducerData.deterministic``, making the packing mode
# a per-reducer property instead of process-global state.
#
# **Fast** — ``(float_flip(score) << 32) | contact_id``.
#   Full 32-bit score precision, no fingerprint. Contact_id in low 32 bits.
#
# **Deterministic** — ``(float_flip(score)>>10 << 42) | (fp << 20) | (id & 0xFFFFF)``.
#   22-bit score, 22-bit fingerprint tiebreaker, 20-bit contact_id.
# ---------------------------------------------------------------------------

# 22-bit fingerprint is wide enough to distinguish any two contacts that share
# the same truncated score within a single reduction slot.  The remaining 20
# bits for contact_id support up to 1,048,575 buffered contacts.
FINGERPRINT_BITS = wp.constant(wp.uint64(22))
CONTACT_ID_BITS = wp.constant(wp.uint64(20))
CONTACT_ID_MASK = wp.constant(wp.uint64((1 << 20) - 1))
FINGERPRINT_MASK = wp.constant(wp.uint64((1 << 22) - 1))
# Plain Python int (not wp.constant) because it is used inside wp.static()
# which requires a Python-level value for compile-time evaluation.
SCORE_SHIFT = 10


# -- Fast (non-deterministic) variants -------------------------------------


@wp.func
def _make_contact_value_fast(score: float, fingerprint: int, contact_id: int) -> wp.uint64:
    """Pack score and contact_id into a uint64 for ``atomic_max`` (fast path).

    ::

        63                  32 31                 0
        ┌─────────────────────┬────────────────────┐
        │  float_flip(score)  │    contact_id      │
        │     (32 bits)       │    (32 bits)        │
        └─────────────────────┴────────────────────┘

    Full 32-bit IEEE-754 precision for the score.  The fingerprint argument
    is accepted for signature compatibility but ignored — ties are broken
    by contact_id (non-deterministic, but correct).

    Args:
        score: Spatial projection score or negated depth [m]. Higher is better.
        fingerprint: Ignored in this variant (kept for signature compatibility).
        contact_id: Index into the contact buffer (from ``atomic_add``).
    """
    return (wp.uint64(float_flip(score)) << wp.uint64(32)) | wp.uint64(contact_id)


@wp.func
def _make_preprune_probe_fast(score: float, fingerprint: int) -> wp.uint64:
    """Pre-prune ceiling probe (fast path).

    Packs the full-precision score with ``0xFFFFFFFF`` in the contact_id
    field, creating the maximum possible value for this score.  The
    comparison ``stored < probe`` is true whenever the stored value can
    be beaten, regardless of what contact_id the new contact receives.
    """
    return (wp.uint64(float_flip(score)) << wp.uint64(32)) | wp.uint64(0xFFFFFFFF)


@wp.func
def _make_spatial_contact_value_fast(score: float, is_inner: bool, fingerprint: int, contact_id: int) -> wp.uint64:
    """Pack inner/outer priority, spatial score, and contact id."""
    priority = wp.uint64(0)
    if is_inner:
        priority = wp.uint64(1)
    score_bits = wp.uint64(float_flip(score) >> wp.uint32(1))
    return (priority << wp.uint64(63)) | (score_bits << wp.uint64(32)) | wp.uint64(contact_id)


@wp.func
def _make_spatial_preprune_probe_fast(score: float, is_inner: bool, fingerprint: int) -> wp.uint64:
    """Pre-prune probe for prioritized spatial contact slots."""
    priority = wp.uint64(0)
    if is_inner:
        priority = wp.uint64(1)
    score_bits = wp.uint64(float_flip(score) >> wp.uint32(1))
    return (priority << wp.uint64(63)) | (score_bits << wp.uint64(32)) | wp.uint64(0xFFFFFFFF)


@wp.func_native("""
return static_cast<int32_t>(packed & 0xFFFFFFFFull);
""")
def _unpack_contact_id_fast(packed: wp.uint64) -> int:
    """Extract contact_id (low 32 bits) — fast variant."""
    ...


# -- Deterministic variants ------------------------------------------------


@wp.func
def _make_contact_value_det(score: float, fingerprint: int, contact_id: int) -> wp.uint64:
    """Pack score, fingerprint, and contact_id into a uint64 for ``atomic_max``.

    This packing enables **deterministic contact reduction**: multiple GPU
    threads propose contacts for the same reduction slot via ``atomic_max``.
    By encoding a deterministic fingerprint (derived from geometry) above
    the non-deterministic contact_id, the ``atomic_max`` winner is always
    the same regardless of thread scheduling.

    ::

        63        42 41        20 19          0
        ┌──────────┬────────────┬──────────────┐
        │  score   │ fingerprint│  contact_id  │
        │ (22 bit) │  (22 bit)  │   (20 bit)   │
        └──────────┴────────────┴──────────────┘

    **Score (bits 63-42, 22 bits)** — ``float_flip(score) >> 10``.
    ``float_flip`` reinterprets the IEEE-754 float as an order-preserving
    uint32 (see http://stereopsis.com/radix.html).  The right-shift by
    ``SCORE_SHIFT`` (10) discards the 10 least-significant bits of the
    mantissa, keeping 1 sign-equivalent + 8 exponent + 13 mantissa = 22
    bits.  This gives ~2^-13 ≈ 1.2e-4 relative precision — sufficient to
    distinguish contacts whose spatial projection scores or negated depths
    differ by more than ~0.1 mm at 1 m scale.

    **Fingerprint (bits 41-20, 22 bits)** — deterministic tiebreaker
    derived from geometry (edge index | mode/source tag bits).  When two
    contacts have the same truncated score, the fingerprint breaks the tie
    so that ``atomic_max`` always picks the same winner regardless of
    thread scheduling.  Effective limits depend on upstream bit consumption:

    - Mesh-triangle contacts: ``(tri_idx << 1) | 1`` — 21 effective bits
      for ``tri_idx`` (~2M triangles).
    - SDF contacts: ``(edge_idx << 2) | (mode << 1)`` — 20 effective bits
      for ``edge_idx`` (~1M edges).

    Meshes exceeding these limits will overflow the fingerprint field,
    causing non-deterministic tiebreaking for those contacts.

    **Contact ID (bits 19-0, 20 bits)** — buffer slot assigned by
    ``atomic_add``.  20 bits supports up to 1,048,575 buffered contacts.
    Non-deterministic, but only matters when both score and fingerprint
    are identical, which requires two geometrically identical contacts —
    an impossible case.

    The cascade ``score > fingerprint > contact_id`` means ``atomic_max``
    on this uint64 selects the contact with the best score, breaking ties
    deterministically via fingerprint.

    Args:
        score: Spatial projection score or negated depth [m]. Higher is better.
        fingerprint: Deterministic contact identifier (e.g. ``(edge_idx << 2) | (mode << 1)``).
        contact_id: Index into the contact buffer (from ``atomic_add``).
    """
    return (
        (wp.uint64(float_flip(score) >> wp.uint32(wp.static(SCORE_SHIFT))) << wp.uint64(42))
        | ((wp.uint64(fingerprint) & FINGERPRINT_MASK) << CONTACT_ID_BITS)
        | (wp.uint64(contact_id) & CONTACT_ID_MASK)
    )


@wp.func
def _make_preprune_probe_det(score: float, fingerprint: int) -> wp.uint64:
    """Deterministic pre-prune probe for ``export_and_reduce_contact_centered``.

    Packs the score and fingerprint with ``CONTACT_ID_MASK`` (all 1s) in the
    contact_id field, creating the *ceiling* value for this (score, fingerprint)
    pair.  The pre-prune comparison ``stored < probe`` is then true whenever
    the stored value can be beaten by a contact with this score and fingerprint,
    regardless of what ``contact_id`` it receives from ``atomic_add``.

    This makes the pre-prune decision depend only on deterministic quantities
    (score and fingerprint), never on the non-deterministic contact_id.
    """
    return (
        (wp.uint64(float_flip(score) >> wp.uint32(wp.static(SCORE_SHIFT))) << wp.uint64(42))
        | ((wp.uint64(fingerprint) & FINGERPRINT_MASK) << CONTACT_ID_BITS)
        | CONTACT_ID_MASK
    )


@wp.func
def _make_spatial_contact_value_det(score: float, is_inner: bool, fingerprint: int, contact_id: int) -> wp.uint64:
    """Pack inner/outer priority, spatial score, fingerprint, and contact id."""
    priority = wp.uint64(0)
    if is_inner:
        priority = wp.uint64(1)
    score_bits = wp.uint64(float_flip(score) >> wp.uint32(wp.static(SCORE_SHIFT + 1)))
    return (
        (priority << wp.uint64(63))
        | (score_bits << wp.uint64(42))
        | ((wp.uint64(fingerprint) & FINGERPRINT_MASK) << CONTACT_ID_BITS)
        | (wp.uint64(contact_id) & CONTACT_ID_MASK)
    )


@wp.func
def _make_spatial_preprune_probe_det(score: float, is_inner: bool, fingerprint: int) -> wp.uint64:
    """Deterministic pre-prune probe for prioritized spatial slots."""
    priority = wp.uint64(0)
    if is_inner:
        priority = wp.uint64(1)
    score_bits = wp.uint64(float_flip(score) >> wp.uint32(wp.static(SCORE_SHIFT + 1)))
    return (
        (priority << wp.uint64(63))
        | (score_bits << wp.uint64(42))
        | ((wp.uint64(fingerprint) & FINGERPRINT_MASK) << CONTACT_ID_BITS)
        | CONTACT_ID_MASK
    )


@wp.func_native("""
return static_cast<int32_t>(packed & 0xFFFFFull);
""")
def _unpack_contact_id_det(packed: wp.uint64) -> int:
    """Extract contact_id (low 20 bits) — deterministic variant."""
    ...


# -- Per-reducer dispatching functions -------------------------------------
# These functions dispatch between fast and deterministic variants based on
# a ``deterministic`` flag, making the packing mode a per-reducer property
# instead of process-global state.


@wp.func
def make_contact_value(score: float, fingerprint: int, contact_id: int, deterministic: int) -> wp.uint64:
    """Pack score, fingerprint, and contact_id into a uint64 for ``atomic_max``.

    Dispatches between fast and deterministic packing based on the
    ``deterministic`` flag.  See :func:`_make_contact_value_fast` and
    :func:`_make_contact_value_det` for the two packing layouts.

    Args:
        score: Spatial projection score or negated depth [m]. Higher is better.
        fingerprint: Deterministic contact identifier (ignored in fast mode).
        contact_id: Index into the contact buffer (from ``atomic_add``).
        deterministic: Non-zero to use deterministic packing.
    """
    if deterministic != 0:
        return _make_contact_value_det(score, fingerprint, contact_id)
    return _make_contact_value_fast(score, fingerprint, contact_id)


@wp.func
def make_spatial_contact_value(
    score: float,
    is_inner: bool,
    fingerprint: int,
    contact_id: int,
    deterministic: int,
) -> wp.uint64:
    """Pack a directional contact value where inner contacts outrank outer contacts."""
    if deterministic != 0:
        return _make_spatial_contact_value_det(score, is_inner, fingerprint, contact_id)
    return _make_spatial_contact_value_fast(score, is_inner, fingerprint, contact_id)


@wp.func
def make_spatial_preprune_probe(score: float, is_inner: bool, fingerprint: int, deterministic: int) -> wp.uint64:
    """Build a pre-prune probe for prioritized directional slots."""
    if deterministic != 0:
        return _make_spatial_preprune_probe_det(score, is_inner, fingerprint)
    return _make_spatial_preprune_probe_fast(score, is_inner, fingerprint)


@wp.func
def unpack_contact_id(packed: wp.uint64, deterministic: int) -> int:
    """Extract contact_id from a packed value.

    Dispatches between fast (low 32 bits) and deterministic (low 20 bits)
    unpacking based on the ``deterministic`` flag.

    Args:
        packed: Packed uint64 value from ``make_contact_value``.
        deterministic: Non-zero to use deterministic unpacking.
    """
    if deterministic != 0:
        return _unpack_contact_id_det(packed)
    return _unpack_contact_id_fast(packed)


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
    exported_flags: wp.array[wp.int32]
    reclaimed_contact_bits: wp.array[wp.uint32]
    reclaimed_contact_count: wp.array[wp.int32]
    capacity: int

    # Deterministic fingerprint per contact (triangle/edge/vertex index).
    # Used as a deterministic tiebreaker in make_contact_value so that
    # atomic_max picks the same winner regardless of thread scheduling.
    contact_fingerprints: wp.array[wp.int32]

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
    ht_values: wp.array[wp.uint64]
    ht_active_slots: wp.array[wp.int32]
    ht_insert_failures: wp.array[wp.int32]
    buffer_overflows: wp.array[wp.int32]
    ht_capacity: int
    ht_values_per_key: int

    # When non-zero, replace the speculative pre-prune probe with a
    # deterministic variant (make_preprune_probe) so that the prune
    # decision depends only on score and fingerprint, never on the
    # non-deterministic contact_id.
    deterministic: int


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
        ht_values: Values array for hashtable (managed here, not by HashTable),
            entry-major: see :func:`reduction_value_index`
    """

    def __init__(
        self,
        capacity: int,
        device: str | None = None,
        store_hydroelastic_data: bool = False,
        store_moment_data: bool = False,
        deterministic: bool = False,
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
            deterministic: If True, use deterministic fingerprint-based tiebreaking
                in contact reduction and replace the pre-prune probe with a
                deterministic variant.
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

        # Predictive entries reserve packed ID zero for provisional winners.
        max_det_contacts = 1 << int(CONTACT_ID_BITS)
        if deterministic and capacity >= max_det_contacts:
            raise ValueError(
                f"Deterministic contact packing supports at most {max_det_contacts - 1} "
                f"buffered contacts ({int(CONTACT_ID_BITS)}-bit contact_id), "
                f"but capacity={capacity}. Reduce max_triangle_pairs or disable "
                f"deterministic mode."
            )
        self.capacity = capacity
        self.device = device
        self.store_hydroelastic_data = store_hydroelastic_data
        self.deterministic = deterministic
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

        # Optional hydroelastic data arrays
        if store_hydroelastic_data:
            self.contact_area = wp.zeros(buffer_size, dtype=wp.float32, device=device)
            self.contact_pressure = wp.zeros(buffer_size, dtype=wp.float32, device=device)
            self.contact_nbin_entry = wp.zeros(buffer_size if enable_reduction else 0, dtype=wp.int32, device=device)
        else:
            self.contact_area = wp.zeros(0, dtype=wp.float32, device=device)
            self.contact_pressure = wp.zeros(0, dtype=wp.float32, device=device)
            self.contact_nbin_entry = wp.zeros(0, dtype=wp.int32, device=device)

        # Generic reduction deduplicates cross-entry winners during export.
        # Hydroelastic reduction intentionally preserves and source-tags them.
        self.exported_flags = wp.zeros(0 if store_hydroelastic_data else buffer_size, dtype=wp.int32, device=device)

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
        previous active count and overflow evidence. Raw payload and exported
        flags remain untouched; the next contact allocations initialize flags.
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
        data.exported_flags = self.exported_flags
        data.reclaimed_contact_bits = self.reclaimed_contact_bits
        data.reclaimed_contact_count = self.reclaimed_contact_count
        data.capacity = self.capacity
        data.contact_fingerprints = self.contact_fingerprints
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
        data.deterministic = 1 if self.deterministic else 0
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
    if reducer_data.exported_flags.shape[0] > 0:
        reducer_data.exported_flags[contact_id] = 0

    # Store contact data (packed into vec4, normal octahedral-encoded into vec2)
    reducer_data.position_depth[contact_id] = wp.vec4(position[0], position[1], position[2], depth)
    reducer_data.normal[contact_id] = encode_oct(normal)
    reducer_data.shape_pairs[contact_id] = wp.vec2i(shape_a, shape_b)
    reducer_data.contact_fingerprints[contact_id] = fingerprint

    return contact_id


@wp.func
def reduce_contact_in_hashtable(
    contact_id: int,
    reducer_data: GlobalContactReducerData,
    beta: float,
    shape_transform: wp.array[wp.transform],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    shape_voxel_resolution: wp.array[wp.vec3i],
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
    if entry_idx >= 0:
        use_beta = depth < beta * wp.length(aabb_upper - aabb_lower)
        for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
            if use_beta:
                dir_2d = get_spatial_direction_2d(dir_i)
                score = wp.dot(pos_2d, dir_2d)
                value = make_contact_value(score, fingerprint, contact_id, reducer_data.deterministic)
                slot_id = dir_i
                reduction_update_slot(entry_idx, slot_id, value, reducer_data.ht_values)

        max_depth_value = make_contact_value(-depth, fingerprint, contact_id, reducer_data.deterministic)
        reduction_update_slot(entry_idx, wp.static(NUM_SPATIAL_DIRECTIONS), max_depth_value, reducer_data.ht_values)
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
    if voxel_entry_idx >= 0:
        # Use -depth so atomic_max selects most penetrating (most negative depth)
        voxel_value = make_contact_value(-depth, fingerprint, contact_id, reducer_data.deterministic)
        reduction_update_slot(voxel_entry_idx, voxel_local_slot, voxel_value, reducer_data.ht_values)
    else:
        wp.atomic_add(reducer_data.ht_insert_failures, 0, 1)


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

    # === Pre-prune normal bin: non-atomic reads ===
    # In deterministic mode we use _make_preprune_probe_det (score + fingerprint +
    # max contact_id) so the prune decision never depends on the non-deterministic
    # contact_id.  In non-deterministic mode we use the cheaper floor probe.
    might_win = False

    if entry_idx >= 0:
        if reducer_data.deterministic != 0:
            max_depth_probe = _make_preprune_probe_det(-depth, fingerprint)
        else:
            max_depth_probe = _make_contact_value_fast(-depth, 0, 0)
        if (
            reducer_data.ht_values[reduction_value_index(entry_idx, wp.static(NUM_SPATIAL_DIRECTIONS))]
            < max_depth_probe
        ):
            might_win = True

        if not might_win and use_beta:
            for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
                if not might_win:
                    dir_2d = get_spatial_direction_2d(dir_i)
                    score = wp.dot(pos_2d, dir_2d)
                    if reducer_data.deterministic != 0:
                        probe = _make_preprune_probe_det(score, fingerprint)
                    else:
                        probe = _make_contact_value_fast(score, 0, 0)
                    if reducer_data.ht_values[reduction_value_index(entry_idx, dir_i)] < probe:
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
            if reducer_data.deterministic != 0:
                voxel_probe = _make_preprune_probe_det(-depth, fingerprint)
            else:
                voxel_probe = _make_contact_value_fast(-depth, 0, 0)
            if reducer_data.ht_values[reduction_value_index(voxel_entry_idx, voxel_local_slot)] < voxel_probe:
                might_win = True

    if not might_win:
        return -1

    # === Allocate buffer slot (only for contacts that might win) ===
    contact_id = export_contact_to_buffer(shape_a, shape_b, position, normal, depth, fingerprint, reducer_data)
    if contact_id < 0:
        return -1

    # === Register in hashtable with fingerprint for deterministic tiebreaking ===
    if entry_idx >= 0:
        for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
            if use_beta:
                dir_2d = get_spatial_direction_2d(dir_i)
                score = wp.dot(pos_2d, dir_2d)
                value = make_contact_value(score, fingerprint, contact_id, reducer_data.deterministic)
                reduction_update_slot(entry_idx, dir_i, value, reducer_data.ht_values)

        max_depth_value = make_contact_value(-depth, fingerprint, contact_id, reducer_data.deterministic)
        reduction_update_slot(entry_idx, wp.static(NUM_SPATIAL_DIRECTIONS), max_depth_value, reducer_data.ht_values)
    else:
        wp.atomic_add(reducer_data.ht_insert_failures, 0, 1)

    # Deferred voxel entry lookup for contacts that won via normal bin
    if voxel_entry_idx < 0:
        voxel_entry_idx = hashtable_find_or_insert(voxel_key, reducer_data.ht_keys, reducer_data.ht_active_slots)
    if voxel_entry_idx >= 0:
        voxel_value = make_contact_value(-depth, fingerprint, contact_id, reducer_data.deterministic)
        reduction_update_slot(voxel_entry_idx, voxel_local_slot, voxel_value, reducer_data.ht_values)
    else:
        wp.atomic_add(reducer_data.ht_insert_failures, 0, 1)

    return contact_id


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
    deterministic: int,
) -> int:
    """Export contact with touching-first spatial winners.

    Directional slots accept contacts out to ``outer_spatial_depth``, but
    candidates that touch by the end of the step, ``step_end_depth`` inside
    ``inner_spatial_depth``, outrank the others. ``step_end_depth`` is the
    depth less the candidate's closing speed times the step (its current
    ``depth`` when the motion is unknown), so the far side of a support that
    a rocking body swings onto this step keeps rows. Within a tier the
    furthest spatial projection wins; the max-depth and voxel slots take
    contacts touching now (``depth`` inside ``inner_spatial_depth``) only.
    """
    use_inner = depth < inner_spatial_depth
    step_end_inner = step_end_depth < inner_spatial_depth
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

    # Resolve both keys up front so their probes and the slot reads below can
    # overlap. Missing voxel keys are published only after a contact ID is
    # available; deleting a speculative key after publication would race with
    # concurrent threads that have already found it.
    entry_idx = hashtable_find_or_insert(key, reducer_data.ht_keys, reducer_data.ht_active_slots)
    voxel_entry_idx = -1
    if use_inner:
        voxel_entry_idx = hashtable_find(voxel_key, reducer_data.ht_keys)

    might_win = False
    if entry_idx >= 0:
        # Read every slot before comparing so the loads issue back to back.
        slot_values = replaced_values_vec_type()
        for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS + 1)):
            slot_values[dir_i] = reducer_data.ht_values[reduction_value_index(entry_idx, dir_i)]
        voxel_slot_value = wp.uint64(0)
        if voxel_entry_idx >= 0:
            voxel_slot_value = reducer_data.ht_values[reduction_value_index(voxel_entry_idx, voxel_local_slot)]

        if use_inner:
            if deterministic != 0:
                max_depth_probe = _make_preprune_probe_det(-depth, fingerprint)
            else:
                max_depth_probe = _make_contact_value_fast(-depth, 0, 0)
            if slot_values[wp.static(NUM_SPATIAL_DIRECTIONS)] < max_depth_probe:
                might_win = True
            if voxel_entry_idx >= 0 and voxel_slot_value < max_depth_probe:
                might_win = True
            if voxel_entry_idx < 0:
                might_win = True

        for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
            dir_2d = get_spatial_direction_2d(dir_i)
            score = wp.dot(pos_2d, dir_2d)
            probe = make_spatial_preprune_probe(score, step_end_inner, fingerprint, deterministic)
            if slot_values[dir_i] < probe:
                might_win = True
    else:
        wp.atomic_add(reducer_data.ht_insert_failures, 0, 1)
        if voxel_entry_idx >= 0:
            if deterministic != 0:
                voxel_probe = _make_preprune_probe_det(-depth, fingerprint)
            else:
                voxel_probe = _make_contact_value_fast(-depth, 0, 0)
            if reducer_data.ht_values[reduction_value_index(voxel_entry_idx, voxel_local_slot)] < voxel_probe:
                might_win = True

    if not might_win:
        return -1

    won_mask = int(0)
    replaced_values = replaced_values_vec_type()
    if entry_idx >= 0:
        for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
            dir_2d = get_spatial_direction_2d(dir_i)
            score = wp.dot(pos_2d, dir_2d)
            provisional_value = make_spatial_contact_value(score, step_end_inner, fingerprint, 0, deterministic)
            previous_value = reduction_try_update_slot(entry_idx, dir_i, provisional_value, reducer_data.ht_values)
            if previous_value < provisional_value:
                won_mask |= 1 << dir_i
                replaced_values[dir_i] = previous_value

        if use_inner:
            provisional_value = make_contact_value(-depth, fingerprint, 0, deterministic)
            previous_value = reduction_try_update_slot(
                entry_idx,
                wp.static(NUM_SPATIAL_DIRECTIONS),
                provisional_value,
                reducer_data.ht_values,
            )
            if previous_value < provisional_value:
                won_mask |= 1 << wp.static(NUM_SPATIAL_DIRECTIONS)
                replaced_values[wp.static(NUM_SPATIAL_DIRECTIONS)] = previous_value

    if use_inner and voxel_entry_idx >= 0:
        provisional_value = make_contact_value(-depth, fingerprint, 0, deterministic)
        previous_value = reduction_try_update_slot(
            voxel_entry_idx, voxel_local_slot, provisional_value, reducer_data.ht_values
        )
        if previous_value < provisional_value:
            won_mask |= 1 << wp.static(NUM_SPATIAL_DIRECTIONS + 1)
            replaced_values[wp.static(NUM_SPATIAL_DIRECTIONS + 1)] = previous_value

    voxel_entry_missing = use_inner and voxel_entry_idx < 0
    if won_mask == 0 and not voxel_entry_missing:
        return -1

    # Avoid allocating candidates superseded during their own slot updates.
    still_wins = False
    if entry_idx >= 0:
        for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
            if not still_wins and (won_mask & (1 << dir_i)) != 0:
                dir_2d = get_spatial_direction_2d(dir_i)
                score = wp.dot(pos_2d, dir_2d)
                provisional_value = make_spatial_contact_value(score, step_end_inner, fingerprint, 0, deterministic)
                if reducer_data.ht_values[reduction_value_index(entry_idx, dir_i)] == provisional_value:
                    still_wins = True

        if not still_wins and use_inner and (won_mask & (1 << wp.static(NUM_SPATIAL_DIRECTIONS))) != 0:
            provisional_value = make_contact_value(-depth, fingerprint, 0, deterministic)
            if (
                reducer_data.ht_values[reduction_value_index(entry_idx, wp.static(NUM_SPATIAL_DIRECTIONS))]
                == provisional_value
            ):
                still_wins = True

    if (
        not still_wins
        and use_inner
        and voxel_entry_idx >= 0
        and (won_mask & (1 << wp.static(NUM_SPATIAL_DIRECTIONS + 1))) != 0
    ):
        provisional_value = make_contact_value(-depth, fingerprint, 0, deterministic)
        if reducer_data.ht_values[reduction_value_index(voxel_entry_idx, voxel_local_slot)] == provisional_value:
            still_wins = True

    # Without a surviving slot win, a contact may still claim the voxel slot of
    # an entry that is not published yet. That claim happens after the contact
    # ID exists, so a losing claimant returns its ID below.
    voxel_only = voxel_entry_missing and not still_wins
    if not still_wins and not voxel_only:
        return -1
    contact_id = export_contact_to_buffer(shape_a, shape_b, position, normal, depth, fingerprint, reducer_data)
    if contact_id < 0:
        if entry_idx >= 0:
            for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
                if (won_mask & (1 << dir_i)) != 0:
                    dir_2d = get_spatial_direction_2d(dir_i)
                    score = wp.dot(pos_2d, dir_2d)
                    provisional_value = make_spatial_contact_value(score, step_end_inner, fingerprint, 0, deterministic)
                    reduction_rollback_slot(
                        entry_idx, dir_i, provisional_value, replaced_values[dir_i], reducer_data.ht_values
                    )
            if use_inner and (won_mask & (1 << wp.static(NUM_SPATIAL_DIRECTIONS))) != 0:
                provisional_value = make_contact_value(-depth, fingerprint, 0, deterministic)
                reduction_rollback_slot(
                    entry_idx,
                    wp.static(NUM_SPATIAL_DIRECTIONS),
                    provisional_value,
                    replaced_values[wp.static(NUM_SPATIAL_DIRECTIONS)],
                    reducer_data.ht_values,
                )
        if use_inner and voxel_entry_idx >= 0 and (won_mask & (1 << wp.static(NUM_SPATIAL_DIRECTIONS + 1))) != 0:
            provisional_value = make_contact_value(-depth, fingerprint, 0, deterministic)
            reduction_rollback_slot(
                voxel_entry_idx,
                voxel_local_slot,
                provisional_value,
                replaced_values[wp.static(NUM_SPATIAL_DIRECTIONS + 1)],
                reducer_data.ht_values,
            )
        return -1

    if entry_idx >= 0:
        for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
            dir_2d = get_spatial_direction_2d(dir_i)
            score = wp.dot(pos_2d, dir_2d)
            value = make_spatial_contact_value(score, step_end_inner, fingerprint, contact_id, deterministic)
            reduction_update_slot(entry_idx, dir_i, value, reducer_data.ht_values)

        if use_inner:
            max_depth_value = make_contact_value(-depth, fingerprint, contact_id, deterministic)
            reduction_update_slot(entry_idx, wp.static(NUM_SPATIAL_DIRECTIONS), max_depth_value, reducer_data.ht_values)

    if use_inner:
        if voxel_entry_idx < 0:
            voxel_entry_idx = hashtable_find_or_insert(voxel_key, reducer_data.ht_keys, reducer_data.ht_active_slots)
        if voxel_entry_idx >= 0:
            voxel_value = make_contact_value(-depth, fingerprint, contact_id, deterministic)
            if voxel_only:
                previous_value = reduction_try_update_slot(
                    voxel_entry_idx, voxel_local_slot, voxel_value, reducer_data.ht_values
                )
                if previous_value >= voxel_value:
                    reclaim_contact_id(contact_id, reducer_data)
                    return -1
            else:
                reduction_update_slot(voxel_entry_idx, voxel_local_slot, voxel_value, reducer_data.ht_values)
        else:
            wp.atomic_add(reducer_data.ht_insert_failures, 0, 1)
            if voxel_only:
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
    """Export a contact using the reduction mode stored in ``reducer_data``."""
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
        reducer_data.deterministic,
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
        existing_contact_id: ``-1`` requests a provisional claim; a positive one-based ID updates an existing
            buffered contact.
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
    clearance_idx = reduction_value_index(entry_idx, clearance_slot)
    impact_time_idx = reduction_value_index(entry_idx, wp.static(NUM_SPATIAL_DIRECTIONS))

    # ID zero is reserved so provisional winners can be claimed atomically
    # before allocating buffer storage.
    contact_id = existing_contact_id
    if contact_id >= 0:
        clearance_value = make_contact_value(clearance_score, fingerprint, contact_id, reducer_data.deterministic)
        impact_time_value = make_contact_value(impact_time_score, fingerprint, contact_id, reducer_data.deterministic)
        reduction_update_slot(entry_idx, clearance_slot, clearance_value, reducer_data.ht_values)
        reduction_update_slot(
            entry_idx,
            wp.static(NUM_SPATIAL_DIRECTIONS),
            impact_time_value,
            reducer_data.ht_values,
        )
        if (
            reducer_data.ht_values[clearance_idx] == clearance_value
            or reducer_data.ht_values[impact_time_idx] == impact_time_value
        ):
            return contact_id
        return -1

    provisional_clearance_value = make_contact_value(clearance_score, fingerprint, 0, reducer_data.deterministic)
    provisional_impact_time_value = make_contact_value(impact_time_score, fingerprint, 0, reducer_data.deterministic)
    previous_clearance_value = reduction_try_update_slot(
        entry_idx, clearance_slot, provisional_clearance_value, reducer_data.ht_values
    )
    previous_impact_time_value = reduction_try_update_slot(
        entry_idx,
        wp.static(NUM_SPATIAL_DIRECTIONS),
        provisional_impact_time_value,
        reducer_data.ht_values,
    )
    clearance_won = previous_clearance_value < provisional_clearance_value
    impact_time_won = previous_impact_time_value < provisional_impact_time_value
    if clearance_won:
        clearance_won = reducer_data.ht_values[clearance_idx] == provisional_clearance_value
    if impact_time_won:
        impact_time_won = reducer_data.ht_values[impact_time_idx] == provisional_impact_time_value
    if not clearance_won and not impact_time_won:
        return -1

    contact_id = export_contact_to_buffer(shape_a, shape_b, position, normal, depth, fingerprint, reducer_data)
    if contact_id < 0:
        if clearance_won:
            reduction_rollback_slot(
                entry_idx,
                clearance_slot,
                provisional_clearance_value,
                previous_clearance_value,
                reducer_data.ht_values,
            )
        if impact_time_won:
            reduction_rollback_slot(
                entry_idx,
                wp.static(NUM_SPATIAL_DIRECTIONS),
                provisional_impact_time_value,
                previous_impact_time_value,
                reducer_data.ht_values,
            )
        return -1

    retained = bool(False)
    if clearance_won:
        clearance_value = make_contact_value(clearance_score, fingerprint, contact_id, reducer_data.deterministic)
        if reduction_finalize_slot(
            entry_idx,
            clearance_slot,
            provisional_clearance_value,
            clearance_value,
            reducer_data.ht_values,
        ):
            retained = True
    if impact_time_won:
        impact_time_value = make_contact_value(impact_time_score, fingerprint, contact_id, reducer_data.deterministic)
        if reduction_finalize_slot(
            entry_idx,
            wp.static(NUM_SPATIAL_DIRECTIONS),
            provisional_impact_time_value,
            impact_time_value,
            reducer_data.ht_values,
        ):
            retained = True
    if retained:
        return contact_id
    reclaim_contact_id(contact_id, reducer_data)
    return -1


@wp.kernel(enable_backward=False)
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
        )


@wp.kernel(enable_backward=False)
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
            reduce_contact_in_hashtable(
                contact_id,
                reducer_data,
                wp.static(BETA_THRESHOLD),
                shape_transform,
                shape_collision_aabb_lower,
                shape_collision_aabb_upper,
                shape_voxel_resolution,
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
    value_a: wp.uint64,
    value_b: wp.uint64,
    position_depth: wp.array[wp.vec4],
    normal: wp.array[wp.vec2],
    contact_fingerprints: wp.array[wp.int32],
    deterministic: int,
) -> int:
    """Suppress a repeated ID or the existing roundoff-equivalent loser."""
    if value_a == wp.uint64(0) or value_b == wp.uint64(0):
        return 0

    contact_a = unpack_contact_id(value_a, deterministic)
    contact_b = unpack_contact_id(value_b, deterministic)
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
    deterministic: int,
) -> int:
    """Return the suppression bit for one triangularly indexed slot pair."""
    slot_b = int(1)
    while pair_idx >= slot_b:
        pair_idx = pair_idx - slot_b
        slot_b = slot_b + 1
    return _duplicate_bit_for_winners(
        pair_idx,
        slot_b,
        ht_values[reduction_value_index(entry_idx, pair_idx)],
        ht_values[reduction_value_index(entry_idx, slot_b)],
        position_depth,
        normal,
        contact_fingerprints,
        deterministic,
    )


def create_export_reduced_contacts_kernel(writer_func: Any, *, clear_after_export: bool = False, compact: bool = False):
    """Create a tiled kernel that exports globally reduced contacts.

    One thread block processes each active hashtable entry. Its first 21 lanes
    compare the seven possible winner pairs in parallel, preserving the
    lower-fingerprint representative for roundoff-equivalent geometry. CUDA
    lanes export the seven surviving slots concurrently in fast mode; CPU and
    deterministic paths stream them serially from lane zero.

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
    _module = f"export_reduced_contacts_{writer_func.__name__}"
    if clear_after_export:
        _module += "_cleanup"
    if compact:
        _module += "_compact"

    @wp.func
    def export_contact_id(
        contact_id: int,
        position_depth: wp.array[wp.vec4],
        normal: wp.array[wp.vec2],
        shape_pairs: wp.array[wp.vec2i],
        contact_fingerprints: wp.array[wp.int32],
        exported_flags: wp.array[wp.int32],
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_gap: wp.array[float],
        writer_data: Any,
    ):
        old_flag = wp.atomic_add(exported_flags, contact_id, 1)
        if old_flag > 0:
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
        winner_values = wp.types.vector(length=VALUES_PER_KEY, dtype=wp.uint64)

        @wp.kernel(enable_backward=False, module=_module)
        def export_compact_contacts_kernel(
            ht_keys: wp.array[wp.uint64],
            ht_values: wp.array[wp.uint64],
            ht_active_slots: wp.array[wp.int32],
            position_depth: wp.array[wp.vec4],
            normal: wp.array[wp.vec2],
            shape_pairs: wp.array[wp.vec2i],
            contact_fingerprints: wp.array[wp.int32],
            exported_flags: wp.array[wp.int32],
            shape_types: wp.array[int],
            shape_data: wp.array[wp.vec4],
            shape_gap: wp.array[float],
            writer_data: Any,
            total_num_threads: int,
            parallel_pairs: int,
            deterministic: int,
        ):
            tid = wp.tid()
            ht_capacity = ht_keys.shape[0]
            num_active = ht_active_slots[ht_capacity]
            for active_idx in range(tid, num_active, total_num_threads):
                entry_idx = ht_active_slots[active_idx]
                values = winner_values()
                for slot in range(wp.static(VALUES_PER_KEY)):
                    values[slot] = ht_values[reduction_value_index(entry_idx, slot)]

                duplicate_mask = int(0)
                for slot_b in range(1, wp.static(VALUES_PER_KEY)):
                    # An early continue keeps this fixed loop dynamic in Warp,
                    # forcing the winner vector into local memory.
                    if values[slot_b] != wp.uint64(0):
                        for slot_a in range(slot_b):
                            duplicate_mask = duplicate_mask | _duplicate_bit_for_winners(
                                slot_a,
                                slot_b,
                                values[slot_a],
                                values[slot_b],
                                position_depth,
                                normal,
                                contact_fingerprints,
                                deterministic,
                            )

                for slot in range(wp.static(VALUES_PER_KEY)):
                    value = values[slot]
                    if value != wp.uint64(0) and duplicate_mask & (1 << slot) == 0:
                        contact_id = unpack_contact_id(value, deterministic)
                        if contact_id != 0:
                            export_contact_id(
                                contact_id,
                                position_depth,
                                normal,
                                shape_pairs,
                                contact_fingerprints,
                                exported_flags,
                                shape_types,
                                shape_data,
                                shape_gap,
                                writer_data,
                            )

                if wp.static(clear_after_export):
                    # Active keys are unique; this thread has finished all
                    # winner reads. Cross-key claims retain their raw payload.
                    for slot in range(wp.static(VALUES_PER_KEY)):
                        if values[slot] != wp.uint64(0):
                            ht_values[reduction_value_index(entry_idx, slot)] = wp.uint64(0)
                    ht_keys[entry_idx] = HASHTABLE_EMPTY_KEY

        return export_compact_contacts_kernel

    @wp.kernel(enable_backward=False, module=_module)
    def export_reduced_contacts_kernel(
        ht_keys: wp.array[wp.uint64],
        ht_values: wp.array[wp.uint64],
        ht_active_slots: wp.array[wp.int32],
        position_depth: wp.array[wp.vec4],
        normal: wp.array[wp.vec2],
        shape_pairs: wp.array[wp.vec2i],
        contact_fingerprints: wp.array[wp.int32],
        exported_flags: wp.array[wp.int32],
        shape_types: wp.array[int],
        shape_data: wp.array[wp.vec4],
        shape_gap: wp.array[float],
        writer_data: Any,
        total_num_blocks: int,
        parallel_pairs: int,
        deterministic: int,
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
                        deterministic,
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
                        deterministic,
                    )

            wp.tile_scatter_masked(duplicate_bits, lane, duplicate_bit, True)
            duplicate_mask = wp.tile_reduce(wp.bit_or, duplicate_bits)[0]

            if parallel_pairs != 0 and deterministic == 0:
                if lane < wp.static(VALUES_PER_KEY) and duplicate_mask & (1 << lane) == 0:
                    value = ht_values[reduction_value_index(entry_idx, lane)]
                    if value != wp.uint64(0):
                        contact_id = unpack_contact_id(value, deterministic)
                        if contact_id != 0:
                            export_contact_id(
                                contact_id,
                                position_depth,
                                normal,
                                shape_pairs,
                                contact_fingerprints,
                                exported_flags,
                                shape_types,
                                shape_data,
                                shape_gap,
                                writer_data,
                            )
            elif lane == 0:
                for slot in range(wp.static(VALUES_PER_KEY)):
                    if duplicate_mask & (1 << slot) != 0:
                        continue
                    value = ht_values[reduction_value_index(entry_idx, slot)]
                    if value == wp.uint64(0):
                        continue

                    contact_id = unpack_contact_id(value, deterministic)
                    if contact_id == 0:
                        continue
                    # The global flag also suppresses repeated winners within
                    # this entry, so no per-entry ID list is needed.
                    export_contact_id(
                        contact_id,
                        position_depth,
                        normal,
                        shape_pairs,
                        contact_fingerprints,
                        exported_flags,
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
                # Other entries deduplicate through raw payload/flags, which
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
):
    """Create the original query or a filtered heightfield primitive variant.

    Args:
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
    query_support = support_map_accelerated if accelerated else HEIGHTFIELD_PRIMITIVE_SUPPORTS[primitive_type]
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

            if wp.static(primitive_type == 0 and mesh_round != MESH_ROUND_ONLY):
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
    if kernel_name is not None:
        mesh_triangle_contacts_to_reducer_kernel.__name__ = kernel_name
    mesh_triangle_contacts_to_reducer_kernel.__qualname__ = mesh_triangle_contacts_to_reducer_kernel.__name__
    mesh_triangle_contacts_to_reducer_kernel.__annotations__["reducer_data"] = writer_data_type
    return wp.kernel(mesh_triangle_contacts_to_reducer_kernel, enable_backward=False, module="unique")


mesh_triangle_contacts_to_reducer_kernel = create_mesh_triangle_contacts_to_reducer_kernel()
