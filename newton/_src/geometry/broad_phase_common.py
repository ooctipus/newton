# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Common utilities for broad phase collision detection.

Provides shared functions for AABB overlap tests, world/group filtering,
and pair output used by both NxN and SAP broad phase implementations.
"""

from typing import Any

import numpy as np
import warp as wp

from .flags import ShapeFlags

BODY_FLAG_KINEMATIC = wp.constant(1 << 1)


@wp.func
def check_aabb_overlap(
    box1_lower: wp.vec3,
    box1_upper: wp.vec3,
    box1_cutoff: float,
    box2_lower: wp.vec3,
    box2_upper: wp.vec3,
    box2_cutoff: float,
) -> bool:
    cutoff_combined = box1_cutoff + box2_cutoff
    return (
        box1_lower[0] <= box2_upper[0] + cutoff_combined
        and box1_upper[0] >= box2_lower[0] - cutoff_combined
        and box1_lower[1] <= box2_upper[1] + cutoff_combined
        and box1_upper[1] >= box2_lower[1] - cutoff_combined
        and box1_lower[2] <= box2_upper[2] + cutoff_combined
        and box1_upper[2] >= box2_lower[2] - cutoff_combined
    )


@wp.func
def check_aabb_overlap_moving(
    shape1: int,
    shape2: int,
    box_lower: wp.array[wp.vec3],
    box_upper: wp.array[wp.vec3],
    cutoff1: float,
    cutoff2: float,
    shape_displacement: wp.array[wp.vec3],
) -> bool:
    """Return static or continuous overlap according to the optional sweep array."""
    if shape_displacement.shape[0] == 0:
        return check_aabb_overlap(
            box_lower[shape1], box_upper[shape1], cutoff1, box_lower[shape2], box_upper[shape2], cutoff2
        )

    cutoff_combined = cutoff1 + cutoff2
    relative_displacement = shape_displacement[shape1] - shape_displacement[shape2]
    lower_box1 = box_lower[shape1]
    upper_box1 = box_upper[shape1]
    lower_box2 = box_lower[shape2]
    upper_box2 = box_upper[shape2]
    enter = float(0.0)
    exit_time = float(1.0)
    for axis in range(3):
        lower1 = lower_box1[axis]
        upper1 = upper_box1[axis]
        lower2 = lower_box2[axis] - cutoff_combined
        upper2 = upper_box2[axis] + cutoff_combined
        delta = relative_displacement[axis]
        if delta == 0.0:
            if lower1 > upper2 or upper1 < lower2:
                return False
        else:
            axis_enter = (lower2 - upper1) / delta
            axis_exit = (upper2 - lower1) / delta
            if axis_enter > axis_exit:
                tmp = axis_enter
                axis_enter = axis_exit
                axis_exit = tmp
            enter = wp.max(enter, axis_enter)
            exit_time = wp.min(exit_time, axis_exit)
            if enter > exit_time:
                return False
    return True


@wp.func
def binary_search(values: wp.array[Any], value: Any, lower: int, upper: int) -> int:
    while lower < upper:
        mid = (lower + upper) >> 1
        if values[mid] > value:
            upper = mid
        else:
            lower = mid + 1

    return upper


@wp.func
def _vec2i_less(p: wp.vec2i, q: wp.vec2i) -> bool:
    """Lexicographic less-than for vec2i.

    Args:
        p: First vector to compare.
        q: Second vector to compare.

    Returns:
        True if p < q lexicographically, i.e. p[0] < q[0] or (p[0] == q[0] and p[1] < q[1]).
    """
    if p[0] < q[0]:
        return True
    if p[0] > q[0]:
        return False
    return p[1] < q[1]


@wp.func
def _vec2i_equal(p: wp.vec2i, q: wp.vec2i) -> bool:
    """Element-wise equality for vec2i.

    Args:
        p: First vector to compare.
        q: Second vector to compare.

    Returns:
        True if p[0] == q[0] and p[1] == q[1].
    """
    return p[0] == q[0] and p[1] == q[1]


@wp.func
def is_pair_excluded(
    pair: wp.vec2i,
    filter_pairs: wp.array[wp.vec2i],
    num_filter_pairs: int,
) -> bool:
    """Check whether a shape pair is in the sorted exclusion list via binary search.

    Args:
        pair: Canonical shape pair (min, max) to look up.
        filter_pairs: Lexicographically sorted array of excluded shape pairs.
            Each entry must be canonical (min, max).
        num_filter_pairs: Number of valid entries in ``filter_pairs``.

    Returns:
        True if ``pair`` is found in ``filter_pairs``, False otherwise.
        Returns False immediately when ``num_filter_pairs`` is 0.
    """
    if num_filter_pairs <= 0:
        return False
    low = int(0)
    high = num_filter_pairs - 1
    while low <= high:
        mid = (low + high) >> 1
        m = filter_pairs[mid]
        if _vec2i_equal(pair, m):
            return True
        if _vec2i_less(pair, m):
            high = mid - 1
        else:
            low = mid + 1
    return False


@wp.func
def is_shape_pair_immovable_filtered(
    shape_a: int,
    shape_b: int,
    shape_body: wp.array[int],
    body_flags: wp.array[int],
    include_static_kinematic_pairs: bool,
) -> bool:
    """Return whether a shape pair should be skipped by immovable-body filtering."""
    # Empty shape metadata is the expert-call opt-out. An empty body array,
    # however, is valid for an all-static model and must still filter the pair.
    if include_static_kinematic_pairs or shape_body.shape[0] == 0:
        return False

    body_a = shape_body[shape_a]
    body_b = shape_body[shape_b]

    static_a = body_a < 0
    static_b = body_b < 0

    if static_a and static_b:
        return True

    # Without body metadata we cannot distinguish dynamic from kinematic.
    if body_flags.shape[0] == 0:
        return False

    kinematic_a = False
    kinematic_b = False
    if not static_a:
        kinematic_a = (body_flags[body_a] & BODY_FLAG_KINEMATIC) != 0
    if not static_b:
        kinematic_b = (body_flags[body_b] & BODY_FLAG_KINEMATIC) != 0

    immovable_a = static_a or kinematic_a
    immovable_b = static_b or kinematic_b
    return immovable_a and immovable_b


@wp.func
def write_pair(
    pair: wp.vec2i,
    candidate_pair: wp.array[wp.vec2i],
    candidate_pair_count: wp.array[int],  # Size one array
    max_candidate_pair: int,
):
    pairid = wp.atomic_add(candidate_pair_count, 0, 1)

    if pairid >= max_candidate_pair:
        return

    candidate_pair[pairid] = pair


# -- The contact forecast ---------------------------------------------------------------------------------------
#
# A graph replay cannot map contact rows and the host reads a replay's demand a few replays late, so the contact pool
# needs a bound on the demand within that lag (``ContactPool.horizon``). On a forecasting pass (the pool's flag, which
# its record sets on the device: one pass in ``FORECAST_PASSES``) the bounding-box pass states each shape's reach over
# the horizon (forecast_shape), and the broad phase adds, over the pairs it tests, the most contacts a pair can make
# when its shapes' boxes and bounding capsules can meet within their reaches (forecast_pair). It is a predicate of the
# pair test the broad phase runs anyway: no launch of its own.

FORECAST_REDUCED = wp.constant(1)
"""A shape's forecast bit: a mesh or a heightfield, whose pairs contact reduction bounds."""

FORECAST_HYDROELASTIC = wp.constant(2)
"""A shape's forecast bit: hydroelastic, so a pair of two takes the SDF-SDF path."""

FORECAST_FLOATS = wp.constant(4)
"""A shape's forecast bit: on a body gravity moves, which no fixed-base articulation holds."""

FORECAST_TERRAIN = wp.constant(8)
"""A shape's forecast bit: an upright static heightfield, whose height under a partner the forecast reads by tiles."""

FORECAST_SLOTS_SHIFT = wp.constant(8)
"""A shape's forecast word holds, from this bit on, the rows the terrain law's walk holds for a pair of it."""


@wp.struct
class ShapeForecast:
    """What a collision pass's contact forecast reads and writes. Empty arrays forecast nothing."""

    on: wp.array[wp.int32]
    """Whether this pass forecasts (the contact pool's flag, shape 1)."""
    slots: wp.array[wp.int64]
    """Counters the broad phase adds the forecast into (the contact pool's)."""
    horizon: wp.array[wp.vec2]
    """The horizon [s] and the fall gravity adds over it [m] (the contact pool's, shape 1)."""
    word: wp.array[wp.int32]
    """Per shape: its forecast bits, and its terrain walk rows from :data:`FORECAST_SLOTS_SHIFT` on."""
    capsule: wp.array[wp.vec4]
    """Per shape: its bounding capsule's half axis in the shape frame, and its radius [m], -1 for a plane or a
    heightfield, which have none."""
    middle: wp.array[wp.vec4]
    """Per shape: its bounding capsule's center in the shape frame, and its farthest point from its origin [m]."""
    pose: wp.array[wp.transform]
    """Per shape: its world transform, which the bounding-box pass writes on every pass."""
    body_com: wp.array[wp.vec3]
    """Per body: its center of mass in the body frame."""
    reach: wp.array[wp.float32]
    """Per shape, written by a forecasting pass: how far any point of the shape may travel over the horizon [m]."""
    top: wp.array[wp.float32]
    """Per shape, written by a forecasting pass of a model with terrain: the highest terrain under its box [m]."""
    terrain_count: int
    terrain_origin: wp.array[wp.vec3]
    """Per terrain: its corner in the world and its height origin [m]."""
    terrain_scale: wp.array[wp.vec2]
    """Per terrain: tiles per meter along x and y."""
    terrain_layout: wp.array[wp.vec3i]
    """Per terrain: its first tile, tile rows and tile columns."""
    terrain_tiles: wp.array[wp.float32]
    """The highest height of each tile [m]."""
    bound_convex: int
    """The most contacts a convex pair makes."""
    bound_reduced: int
    """The most contacts contact reduction keeps for a pair with a mesh or a heightfield."""
    bound_hydroelastic: int
    """The most contacts a hydroelastic pair makes."""


@wp.func
def forecasting(forecast: ShapeForecast) -> bool:
    """Whether this pass forecasts: the pool's flag says so (a forecast without arrays forecasts nothing)."""
    return forecast.on.shape[0] > 0 and forecast.on[0] != 0


@wp.func
def _terrain_top(
    origin: wp.vec3, scale: wp.vec2, layout: wp.vec3i, tiles: wp.array[wp.float32], x: float, y: float
) -> float:
    """Return the highest terrain [m] of the tile under world point ``(x, y)``."""
    column = wp.clamp(int((x - origin[0]) * scale[0]), 0, layout[2] - 1)
    row = wp.clamp(int((y - origin[1]) * scale[1]), 0, layout[1] - 1)
    return origin[2] + tiles[layout[0] + row * layout[2] + column]


@wp.func
def forecast_shape(
    forecast: ShapeForecast,
    shape: int,
    body: int,
    pose: wp.transform,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    lower: wp.vec3,
    upper: wp.vec3,
):
    """State a shape's reach over the horizon on a forecasting pass, and the highest terrain under its box.

    The reach bounds how far any point of the shape travels: its body's center-of-mass speed plus its spin times
    the lever from the center of mass to the shape's farthest point, over the horizon, plus gravity's fall if the
    body floats. The highest terrain under the shape's box (``lower``, ``upper``) gives a heightfield pair its gap.
    """
    horizon = forecast.horizon[0]
    speed = float(0.0)
    fall = float(0.0)
    word = forecast.word[shape]
    if body >= 0:
        twist = body_qd[body]
        com = wp.transform_point(body_q[body], forecast.body_com[body])
        lever = wp.length(wp.transform_get_translation(pose) - com) + forecast.middle[shape][3]
        speed = wp.length(wp.spatial_top(twist)) + wp.length(wp.spatial_bottom(twist)) * lever
        if word & FORECAST_FLOATS != 0:
            fall = horizon[1]
    forecast.reach[shape] = speed * horizon[0] + fall
    if forecast.top.shape[0] > 0:
        top = float(-1.0e30)
        if word & FORECAST_TERRAIN == 0:
            for terrain in range(forecast.terrain_count):
                origin = forecast.terrain_origin[terrain]
                scale = forecast.terrain_scale[terrain]
                layout = forecast.terrain_layout[terrain]
                tiles = forecast.terrain_tiles
                top = wp.max(
                    top,
                    wp.max(
                        wp.max(
                            _terrain_top(origin, scale, layout, tiles, lower[0], lower[1]),
                            _terrain_top(origin, scale, layout, tiles, lower[0], upper[1]),
                        ),
                        wp.max(
                            _terrain_top(origin, scale, layout, tiles, upper[0], lower[1]),
                            _terrain_top(origin, scale, layout, tiles, upper[0], upper[1]),
                        ),
                    ),
                )
        forecast.top[shape] = top


@wp.func
def forecast_pair(
    shape_a: int,
    shape_b: int,
    box_lower: wp.array[wp.vec3],
    box_upper: wp.array[wp.vec3],
    forecast: ShapeForecast,
) -> int:
    """Return the most contacts a pair can make within the horizon, or 0 when its shapes cannot meet in it.

    The pair's gap is the larger of its bounding boxes' and its bounding capsules' (a long link's box overlaps much
    that the link does not reach); a heightfield's is the partner box's height over the highest terrain under it.
    A gap within the two shapes' reaches counts the pair's worst case: the terrain law's walk rows for a heightfield
    pair, contact reduction's bound for a mesh pair, the convex bound otherwise.
    """
    reach = forecast.reach[shape_a] + forecast.reach[shape_b]
    word_a = forecast.word[shape_a]
    word_b = forecast.word[shape_b]
    if (word_a | word_b) & FORECAST_TERRAIN != 0:
        partner = shape_b
        rows = word_b >> FORECAST_SLOTS_SHIFT
        if word_b & FORECAST_TERRAIN != 0:
            partner = shape_a
            rows = word_a >> FORECAST_SLOTS_SHIFT
        if box_lower[partner][2] - forecast.top[partner] > reach:
            return 0
        return rows
    apart_a = box_lower[shape_a] - box_upper[shape_b]
    apart_b = box_lower[shape_b] - box_upper[shape_a]
    gap = wp.max(wp.max(wp.max(apart_a[0], apart_b[0]), wp.max(apart_a[1], apart_b[1])), wp.max(apart_a[2], apart_b[2]))
    if gap > reach:
        return 0
    capsule_a = forecast.capsule[shape_a]
    capsule_b = forecast.capsule[shape_b]
    radius_a = capsule_a[3]
    radius_b = capsule_b[3]
    if radius_a >= 0.0 and radius_b >= 0.0:
        pose_a = forecast.pose[shape_a]
        pose_b = forecast.pose[shape_b]
        local_a = forecast.middle[shape_a]
        local_b = forecast.middle[shape_b]
        middle_a = wp.transform_point(pose_a, wp.vec3(local_a[0], local_a[1], local_a[2]))
        half_a = wp.transform_vector(pose_a, wp.vec3(capsule_a[0], capsule_a[1], capsule_a[2]))
        middle_b = wp.transform_point(pose_b, wp.vec3(local_b[0], local_b[1], local_b[2]))
        half_b = wp.transform_vector(pose_b, wp.vec3(capsule_b[0], capsule_b[1], capsule_b[2]))
        closest = wp.closest_point_edge_edge(
            middle_a - half_a, middle_a + half_a, middle_b - half_b, middle_b + half_b, 1.0e-9
        )
        if closest[2] - radius_a - radius_b > reach:
            return 0
    if word_a & word_b & FORECAST_HYDROELASTIC != 0:
        return forecast.bound_hydroelastic
    if (word_a | word_b) & FORECAST_REDUCED != 0:
        return forecast.bound_reduced
    return forecast.bound_convex


@wp.func
def forecast_add(
    forecast: ShapeForecast,
    thread: int,
    shape_a: int,
    shape_b: int,
    box_lower: wp.array[wp.vec3],
    box_upper: wp.array[wp.vec3],
    shape_body: wp.array[int],
    body_flags: wp.array[int],
    include_static_kinematic_pairs: bool,
):
    """On a forecasting pass, add a pair's forecast contacts (:func:`forecast_pair`) to the counter of ``thread``'s
    warp (one address a warp, which the hardware adds once), under the immovable-pair rule its broad phase applies to
    an overlapping pair."""
    if not forecasting(forecast):
        return
    contacts = forecast_pair(shape_a, shape_b, box_lower, box_upper, forecast)
    if contacts == 0:
        return
    if is_shape_pair_immovable_filtered(shape_a, shape_b, shape_body, body_flags, include_static_kinematic_pairs):
        return
    wp.atomic_add(forecast.slots, (thread >> 5) % forecast.slots.shape[0], wp.int64(contacts))


# Collision filtering
@wp.func
def test_group_pair(group_a: int, group_b: int) -> bool:
    """Test if two collision groups should interact.

    Args:
        group_a: First collision group ID. Positive values indicate groups that only collide with themselves (and with negative groups).
                Negative values indicate groups that collide with everything except their negative counterpart.
                Zero indicates no collisions.
        group_b: Second collision group ID. Same meaning as group_a.

    Returns:
        bool: True if the groups should collide, False if they should not.
    """
    if group_a == 0 or group_b == 0:
        return False
    if group_a > 0:
        return group_a == group_b or group_b < 0
    if group_a < 0:
        return group_a != group_b


@wp.func
def test_world_and_group_pair(world_a: int, world_b: int, collision_group_a: int, collision_group_b: int) -> bool:
    """Test if two entities should collide based on world indices and collision groups.

    World indices define which simulation world an entity belongs to:
    - Index -1: Global entities that collide with all worlds
    - Indices 0, 1, 2, ...: World-specific entities

    Collision rules:
    1. Entities from different worlds (except -1) do not collide
    2. Global entities (index -1) collide with all worlds
    3. Within the same world, collision groups determine interactions

    Args:
        world_a: World index of first entity
        world_b: World index of second entity
        collision_group_a: Collision group of first entity
        collision_group_b: Collision group of second entity

    Returns:
        bool: True if the entities should collide, False otherwise
    """
    # Check world indices first
    if world_a != -1 and world_b != -1 and world_a != world_b:
        return False

    # If same world or at least one is global (-1), check collision groups
    return test_group_pair(collision_group_a, collision_group_b)


def precompute_world_map(shape_world: np.ndarray | list[int], shape_flags: np.ndarray | list[int] | None = None):
    """Precompute an index map that groups shapes by world ID with shared shapes.

    This method creates an index mapping where shapes belonging to the same world
    (non-negative world ID) are grouped together, and shared shapes
    (world ID -1) are appended to each world's slice.

    A dedicated segment at the end contains only world -1 objects for handling
    -1 vs -1 collisions without duplication.

    Optionally filters out shapes that should not participate in collision detection
    based on their flags (e.g., visual-only shapes without COLLIDE_SHAPES flag).

    Args:
        shape_world: Array of world IDs. Must contain only:
            - World ID -1: Global/shared entities that collide with all worlds
            - World IDs >= 0: World-specific entities (0, 1, 2, ...)
            World IDs < -1 are not supported and will raise ValueError.
        shape_flags: Optional array of shape flags. If provided, only shapes with the
            COLLIDE_SHAPES flag (bit 1) set will be included in the output map. This allows
            efficient filtering of visual-only shapes that shouldn't participate in collision.

    Raises:
        ValueError: If shape_flags is provided and lengths don't match shape_world, or if
            any world IDs are < -1.

    Returns:
        tuple: (index_map, slice_ends)
            - index_map: 1D array of indices into shape_world, arranged such that:
                * Each regular world's indices are followed by all world -1 (shared) indices
                * A final segment contains only world -1 (shared) indices
                Only includes shapes that pass the collision flag filter.
            - slice_ends: 1D array containing the end index (exclusive) of each world's slice
                in the index_map (including the dedicated -1 segment at the end)
    """
    # Ensure shape_world is a numpy array (might be a list from builder)
    if not isinstance(shape_world, np.ndarray):
        shape_world = np.array(shape_world)

    # Filter out non-colliding shapes if flags are provided
    if shape_flags is not None:
        # Ensure shape_flags is also a numpy array
        if not isinstance(shape_flags, np.ndarray):
            shape_flags = np.array(shape_flags)
        if shape_flags.shape[0] != shape_world.shape[0]:
            raise ValueError("shape_flags and shape_world must have the same length")
        colliding_mask = (shape_flags & ShapeFlags.COLLIDE_SHAPES) != 0
    else:
        colliding_mask = np.ones(len(shape_world), dtype=bool)

    # Apply collision filter to get valid indices
    valid_indices = np.where(colliding_mask)[0]

    # Work with filtered world IDs
    filtered_world_ids = shape_world[valid_indices]

    # Validate world IDs: only -1, 0, 1, 2, ... are allowed
    invalid_worlds = shape_world[(shape_world < -1)]
    if len(invalid_worlds) > 0:
        unique_invalid = np.unique(invalid_worlds)
        raise ValueError(
            f"Invalid world IDs detected: {unique_invalid.tolist()}. "
            f"Only world ID -1 (global/shared) and non-negative IDs (0, 1, 2, ...) are supported."
        )

    # Count world -1 (global entities) in filtered set -> num_shared
    # Only world -1 is treated as shared; kernels special-case -1 for deduplication
    negative_mask = filtered_world_ids == -1
    num_shared = np.sum(negative_mask)

    # Get indices of world -1 (shared) entries in the valid set
    shared_local_indices = np.where(negative_mask)[0]
    # Map back to original shape indices
    shared_indices = valid_indices[shared_local_indices]

    # Count how many distinct positive (or zero) world IDs are in filtered set -> world_count
    # Get unique positive/zero world IDs
    positive_mask = filtered_world_ids >= 0
    positive_world_ids = filtered_world_ids[positive_mask]
    unique_worlds = np.unique(positive_world_ids)
    world_count = len(unique_worlds)

    # Calculate total size of result
    # Each world gets its own indices + all shared indices
    # Plus one additional segment at the end with only shared indices
    num_positive = np.sum(positive_mask)
    total_size = num_positive + (num_shared * world_count) + num_shared

    # Allocate output arrays (world_count + 1 to include dedicated -1 segment)
    index_map = np.empty(total_size, dtype=np.int32)
    slice_ends = np.empty(world_count + 1, dtype=np.int32)

    # Build the index map
    current_pos = 0
    for world_idx, world_id in enumerate(unique_worlds):
        # Get indices for this world in the filtered set
        world_local_indices = np.where(filtered_world_ids == world_id)[0]
        # Map back to original shape indices
        world_indices = valid_indices[world_local_indices]
        world_shape_count = len(world_indices)

        # Copy world-specific indices (using original shape indices)
        index_map[current_pos : current_pos + world_shape_count] = world_indices
        current_pos += world_shape_count

        # Append shared (negative) indices (using original shape indices)
        index_map[current_pos : current_pos + num_shared] = shared_indices
        current_pos += num_shared

        # Store the end position of this slice
        slice_ends[world_idx] = current_pos

    # Add dedicated segment at the end with only world -1 objects
    index_map[current_pos : current_pos + num_shared] = shared_indices
    current_pos += num_shared
    slice_ends[world_count] = current_pos

    return index_map, slice_ends
