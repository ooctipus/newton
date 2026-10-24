# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid contact capacity estimates for Kamino's fixed-capacity contact containers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...geometry.contact_reduction import MAX_CONTACTS_PER_PAIR, NUM_NORMAL_BINS
from ...geometry.flags import ShapeFlags
from ...geometry.types import GeoType
from ...sim.collide import _shape_collide_mask

if TYPE_CHECKING:
    from ...sim.model import Model

__all__ = ["estimate_rigid_contact_capacity"]

# Per-pair worst-case contact counts established by the narrow phase:
#
# * Convex-convex pairs (all primitive GeoTypes, CONVEX_MESH, and planes) are
#   handled either by the analytic primitive fast path, which writes at most 4
#   contacts per pair (``narrow_phase.py::narrow_phase_primitive_kernel``,
#   contact slots 0-3), or by the GJK/MPR multi-contact path, which writes at
#   most 4 manifold points plus 1 deepest contact = 5 per pair
#   (``multicontact.py::build_manifold``: ``count_out = min(num_manifold_points, 4)``
#   followed by an optional deepest-contact write).
_CONVEX_PAIR_MAX_CONTACTS = 5
#
# * Mesh/heightfield-involved pairs (mesh-convex, mesh-plane, mesh-mesh,
#   heightfield-*) and hydroelastic SDF-SDF pairs route their per-triangle /
#   per-vertex contacts through contact reduction, which retains at most
#   ``NUM_NORMAL_BINS * (NUM_SPATIAL_DIRECTIONS + 1) + NUM_VOXEL_DEPTH_SLOTS``
#   slots per pair (240 with the default icosahedron configuration).
#   ``contact_reduction.py`` asserts at import time that the slot count never
#   exceeds ``MAX_CONTACTS_PER_PAIR`` (255, the hard architectural limit that
#   keeps per-pair contact indices representable in 8 bits), so
#   MAX_CONTACTS_PER_PAIR bounds every reduced mesh path. The bound assumes
#   contact reduction is enabled (the default); with ``reduce_contacts=False``
#   mesh paths write unreduced per-triangle/per-vertex contacts and no static
#   estimate is possible.
_MESH_PAIR_MAX_CONTACTS = MAX_CONTACTS_PER_PAIR
#
# * Hydroelastic SDF-SDF pairs additionally export one synthetic anchor
#   contact PER ACTIVE NORMAL BIN when ``anchor_contact`` (or
#   ``moment_matching``, which implies it) is enabled
#   (``contact_reduction_hydroelastic.py::export_hydroelastic_reduced_contacts_kernel``),
#   on top of the reduced slots — worst case ``MAX_CONTACTS_PER_PAIR +
#   NUM_NORMAL_BINS`` per pair.
_HYDRO_PAIR_MAX_CONTACTS = MAX_CONTACTS_PER_PAIR + NUM_NORMAL_BINS


# Primitive pairs (GJK/MPR) produce up to 5 manifold contacts.
# Mesh-involved pairs (SDF + contact reduction) typically retain about 40.
_RIGID_CONTACTS_PER_PRIMITIVE_PAIR = 5
_RIGID_CONTACTS_PER_MESH_PAIR = 40
_RIGID_CONTACT_MAX_NEIGHBORS_PER_SHAPE = 20
_RIGID_CONTACT_MIN_CAPACITY = 1000


def estimate_rigid_contact_capacity(model: Model) -> int:
    """
    Estimate the most rigid contacts a model's shapes may produce at once.

    Kamino sizes its fixed-capacity contact containers by this estimate; Newton's
    collision pipeline grows its contact buffers from measured demand instead.

    When precomputed contact pairs are available (``model.shape_contact_pairs``,
    produced by ``ModelBuilder.find_shape_contact_pairs`` for every finalized
    model), the estimate is the exact sum of per-pair worst-case contact counts
    over the actual pair list: each pair is classified by the GeoTypes (and
    hydroelastic flags) of its two shapes and assigned the narrow phase's hard
    per-pair cap. Every broad phase mode (explicit, nxn, sap) applies the same
    world/group/filter-pair logic as ``find_shape_contact_pairs``, so any
    candidate pair reaching the narrow phase is contained in the precomputed
    list and the sum is a provable worst case. For dense pair graphs the sum is
    capped by a spatial-locality estimate (see the inline comment); where that
    cap binds, capacity is physically motivated rather than provable, matching
    the pre-pair-aware behavior. Crucially, shapes that participate in no
    contact pair (e.g. visual-only meshes) contribute nothing to either term.
    The per-pair caps assume contact reduction is enabled (the default for
    mesh/hydroelastic paths); with ``reduce_contacts=False`` no static bound
    exists and overflow remains detectable via ``contact_count > contact_max``.

    Otherwise falls back to a linear neighbor-budget estimate assuming each
    non-plane shape contacts at most ``_RIGID_CONTACT_MAX_NEIGHBORS_PER_SHAPE`` others (spatial
    locality).  The non-plane term is additive across independent worlds so a
    single-pool computation is correct.  The plane term (each plane vs all
    non-planes in its world) would be quadratic if computed globally, so it is
    evaluated per world when metadata is available.

    Args:
        model: The simulation model.

    Returns:
        Estimated maximum number of rigid contacts.
    """
    if not hasattr(model, "shape_type") or model.shape_type is None:
        return 1000  # Fallback

    shape_types = model.shape_type.numpy()
    colliding_mask = _shape_collide_mask(model, len(shape_types))

    mesh_mask = colliding_mask & ((shape_types == int(GeoType.MESH)) | (shape_types == int(GeoType.HFIELD)))

    # ------------------------------------------------------------------
    # Pair-aware exact bound: sum of per-pair worst cases over the actual
    # precomputed contact pairs.
    # ------------------------------------------------------------------
    shape_contact_pairs = getattr(model, "shape_contact_pairs", None)
    if getattr(model, "shape_contact_pair_count", 0) > 0 and shape_contact_pairs is not None:
        # One-time host transfer at pipeline/solver init (can be large for many
        # worlds, e.g. ~655k pairs; init-only cost is acceptable).
        pairs = shape_contact_pairs.numpy().reshape(-1, 2)

        # Mesh/heightfield-involved pairs go through contact reduction.
        pair_is_mesh = mesh_mask[pairs[:, 0]] | mesh_mask[pairs[:, 1]]

        # Pairs where both shapes are hydroelastic route to the SDF-SDF
        # hydroelastic path, whose anchor export can exceed the reduction
        # slots by one contact per normal bin.
        shape_flags = getattr(model, "shape_flags", None)
        if shape_flags is not None:
            hydro_mask = (shape_flags.numpy() & int(ShapeFlags.HYDROELASTIC)) != 0
            pair_is_hydro = hydro_mask[pairs[:, 0]] & hydro_mask[pairs[:, 1]]
        else:
            pair_is_hydro = np.zeros(len(pairs), dtype=bool)
        pair_is_mesh = pair_is_mesh & ~pair_is_hydro

        pair_cap = np.full(len(pairs), _CONVEX_PAIR_MAX_CONTACTS, dtype=np.int64)
        pair_cap[pair_is_mesh] = _MESH_PAIR_MAX_CONTACTS
        pair_cap[pair_is_hydro] = _HYDRO_PAIR_MAX_CONTACTS
        pair_contacts = int(pair_cap.sum())

        # The pair sum is the combinatorial worst case, but for dense pair
        # graphs (e.g. thousands of mutually collidable shapes piled in one
        # world) it explodes quadratically while only a bounded number of
        # neighbors can touch a shape simultaneously. Cap it with a
        # spatial-locality estimate, mirroring the pre-pair-aware behavior:
        # plane-involved pairs keep their full per-pair budget (a ground plane
        # really can contact every shape resting on it at once), and each
        # non-plane shape is budgeted for at most MAX_NEIGHBORS_PER_SHAPE
        # simultaneous neighbors at its own per-pair cap (halved to avoid
        # double-counting both shapes of a pair). Where this cap binds the
        # capacity is physically motivated rather than a provable bound -
        # exactly as before this estimator existed; where the pair sum binds
        # (typical multi-world robot scenes) it is a provable worst case.
        plane_shape = shape_types == int(GeoType.PLANE)
        pair_has_plane = plane_shape[pairs[:, 0]] | plane_shape[pairs[:, 1]]
        plane_pair_contacts = int(pair_cap[pair_has_plane].sum())
        nonplane_pairs = pairs[~pair_has_plane]
        if len(nonplane_pairs) > 0:
            active_shapes = np.unique(nonplane_pairs)
            shape_cap = np.full(len(active_shapes), _CONVEX_PAIR_MAX_CONTACTS, dtype=np.int64)
            shape_cap[mesh_mask[active_shapes]] = _MESH_PAIR_MAX_CONTACTS
            if shape_flags is not None:
                shape_cap[hydro_mask[active_shapes]] = _HYDRO_PAIR_MAX_CONTACTS
            nonplane_locality = int(shape_cap.sum()) * _RIGID_CONTACT_MAX_NEIGHBORS_PER_SHAPE // 2
        else:
            nonplane_locality = 0
        locality_cap = plane_pair_contacts + nonplane_locality

        return max(1000, min(pair_contacts, locality_cap))

    # ------------------------------------------------------------------
    # Fallback: neighbor-budget heuristic (no precomputed pairs available).
    # ------------------------------------------------------------------
    plane_mask = colliding_mask & (shape_types == int(GeoType.PLANE))
    non_plane_mask = colliding_mask & ~plane_mask
    num_meshes = int(np.count_nonzero(mesh_mask))
    num_non_planes = int(np.count_nonzero(non_plane_mask))
    num_primitives = num_non_planes - num_meshes
    num_planes = int(np.count_nonzero(plane_mask))

    # Weighted contacts from non-plane shape types.
    # Each shape's neighbor pairs are weighted by its type's contacts-per-pair.
    # Divide by 2 to avoid double-counting pairs.
    non_plane_contacts = (
        num_primitives * _RIGID_CONTACT_MAX_NEIGHBORS_PER_SHAPE * _RIGID_CONTACTS_PER_PRIMITIVE_PAIR
        + num_meshes * _RIGID_CONTACT_MAX_NEIGHBORS_PER_SHAPE * _RIGID_CONTACTS_PER_MESH_PAIR
    ) // 2

    # Weighted average contacts-per-pair based on the scene's shape mix.
    avg_cpp = (
        (num_primitives * _RIGID_CONTACTS_PER_PRIMITIVE_PAIR + num_meshes * _RIGID_CONTACTS_PER_MESH_PAIR)
        // max(num_non_planes, 1)
        if num_non_planes > 0
        else 0
    )

    # Plane contacts: each plane contacts all non-plane shapes *in its world*.
    # The naive global formula (num_planes * num_non_planes) is O(worlds²) when
    # both counts grow with the number of worlds.  Use per-world counts instead.
    plane_contacts = 0
    if num_planes > 0 and num_non_planes > 0:
        has_world_info = (
            hasattr(model, "shape_world")
            and model.shape_world is not None
            and hasattr(model, "world_count")
            and model.world_count > 0
        )
        shape_world = model.shape_world.numpy() if has_world_info else None

        if shape_world is not None and len(shape_world) == len(shape_types):
            global_mask = shape_world == -1
            local_mask = ~global_mask
            n_worlds = model.world_count

            global_planes = int(np.count_nonzero(global_mask & plane_mask))
            global_non_planes = int(np.count_nonzero(global_mask & non_plane_mask))

            local_plane_counts = np.bincount(shape_world[local_mask & plane_mask], minlength=n_worlds)[:n_worlds]
            local_non_plane_counts = np.bincount(shape_world[local_mask & non_plane_mask], minlength=n_worlds)[
                :n_worlds
            ]

            per_world_planes = local_plane_counts + global_planes
            per_world_non_planes = local_non_plane_counts + global_non_planes

            # Global-global pairs appear in every world slice; keep one copy.
            plane_pair_count = int(np.sum(per_world_planes * per_world_non_planes))
            if n_worlds > 1:
                plane_pair_count -= (n_worlds - 1) * global_planes * global_non_planes
            plane_contacts = plane_pair_count * avg_cpp
        else:
            # Fallback: exact type-weighted sum (correct for single-world models).
            plane_contacts = num_planes * (
                num_primitives * _RIGID_CONTACTS_PER_PRIMITIVE_PAIR + num_meshes * _RIGID_CONTACTS_PER_MESH_PAIR
            )

    total_contacts = non_plane_contacts + plane_contacts

    # Legacy fallback: a pair count is known but the pair list itself is not
    # available (only possible for hand-assembled models; ModelBuilder.finalize
    # always populates shape_contact_pairs). Use the count as a tighter bound.
    if hasattr(model, "shape_contact_pair_count") and model.shape_contact_pair_count > 0:
        weighted_cpp = max(avg_cpp, _RIGID_CONTACTS_PER_PRIMITIVE_PAIR)
        pair_contacts = int(model.shape_contact_pair_count) * weighted_cpp
        total_contacts = min(total_contacts, pair_contacts)

    # Ensure minimum allocation
    return max(_RIGID_CONTACT_MIN_CAPACITY, total_contacts)
