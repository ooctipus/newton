# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic contact sorting on per-contact keys.

Provides the machinery to reorder contact arrays into a canonical,
deterministic order after the narrow-phase collision pipeline has
written contacts in GPU-scheduling-dependent order.

The sort orders the live contacts only (:mod:`~newton._src.core.live_sort`):
a chunk sort in shared memory and pairwise merges whose launches follow
the live count, published on the device. In a CUDA graph capture whose
contact launches bind (:meth:`~newton._src.core.contact_pool.ContactPool.launch`),
every replay sorts the contacts of that replay, so the capture follows
the contact pool's growth.
"""

from __future__ import annotations

import warp as wp

from ..core.contact_pool import ContactField, ContactPool
from ..core.live_sort import EXTENT_SIZE, SORT_CHUNK, launch_live_sort, publish_live, sort_grid
from ..core.types import Devicelike

# Sentinel key for unused contact slots: the max positive int64 sorts last.
SORT_KEY_SENTINEL = wp.constant(wp.int64(0x7FFFFFFFFFFFFFFF))

# -----------------------------------------------------------------------
# Structs + fused backup/gather kernels for the two contact layouts.
#
# Each layout has ONE struct holding both the live arrays and their
# scratch-buffer counterparts, plus two kernels:
#   _backup_*  — copies only active contacts into the scratch buffers
#   _gather_*  — permuted write-back from scratch into the live arrays
# This replaces the previous N x wp.copy + N x kernel-launch pattern
# with 2 kernel launches total (backup + gather) regardless of how
# many arrays are being sorted.
# -----------------------------------------------------------------------


@wp.struct
class _SimpleContactArrays:
    """Live + scratch arrays for the simplified narrow-phase contact layout."""

    pair: wp.array[wp.vec2i]
    position: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    penetration: wp.array[float]
    tangent: wp.array[wp.vec3]
    match_index: wp.array[wp.int32]
    pair_buf: wp.array[wp.vec2i]
    position_buf: wp.array[wp.vec3]
    normal_buf: wp.array[wp.vec3]
    penetration_buf: wp.array[float]
    tangent_buf: wp.array[wp.vec3]
    match_index_buf: wp.array[wp.int32]
    has_tangent: int
    has_match_index: int


@wp.kernel(enable_backward=False)
def _backup_simple_kernel(
    data: _SimpleContactArrays,
    count: wp.array[int],
    sort_keys_src: wp.array[wp.int64],
    sort_keys_dst: wp.array[wp.int64],
    sort_indices: wp.array[wp.int32],
):
    """Prepare sort keys and copy active contacts into scratch buffers."""
    i = wp.tid()
    sort_indices[i] = wp.int32(i)
    if i >= count[0]:
        sort_keys_dst[i] = SORT_KEY_SENTINEL
        return
    sort_keys_dst[i] = sort_keys_src[i]
    data.pair_buf[i] = data.pair[i]
    data.position_buf[i] = data.position[i]
    data.normal_buf[i] = data.normal[i]
    data.penetration_buf[i] = data.penetration[i]
    if data.has_tangent != 0:
        data.tangent_buf[i] = data.tangent[i]
    if data.has_match_index != 0:
        data.match_index_buf[i] = data.match_index[i]


@wp.kernel(enable_backward=False)
def _gather_simple_kernel(data: _SimpleContactArrays, perm: wp.array[wp.int32], count: wp.array[int]):
    """Permuted write-back from scratch into live arrays."""
    i = wp.tid()
    if i >= count[0]:
        return
    p = perm[i]
    data.pair[i] = data.pair_buf[p]
    data.position[i] = data.position_buf[p]
    data.normal[i] = data.normal_buf[p]
    data.penetration[i] = data.penetration_buf[p]
    if data.has_tangent != 0:
        data.tangent[i] = data.tangent_buf[p]
    if data.has_match_index != 0:
        data.match_index[i] = data.match_index_buf[p]


@wp.struct
class _FullContactArrays:
    """Live + scratch arrays for the full CollisionPipeline contact layout."""

    shape0: wp.array[wp.int32]
    shape1: wp.array[wp.int32]
    point0: wp.array[wp.vec3]
    point1: wp.array[wp.vec3]
    offset0: wp.array[wp.vec3]
    offset1: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    margin0: wp.array[float]
    margin1: wp.array[float]
    tids: wp.array[wp.int32]
    stiffness: wp.array[float]
    damping: wp.array[float]
    friction: wp.array[float]
    match_index: wp.array[wp.int32]
    shape0_buf: wp.array[wp.int32]
    shape1_buf: wp.array[wp.int32]
    point0_buf: wp.array[wp.vec3]
    point1_buf: wp.array[wp.vec3]
    offset0_buf: wp.array[wp.vec3]
    offset1_buf: wp.array[wp.vec3]
    normal_buf: wp.array[wp.vec3]
    margin0_buf: wp.array[float]
    margin1_buf: wp.array[float]
    tids_buf: wp.array[wp.int32]
    stiffness_buf: wp.array[float]
    damping_buf: wp.array[float]
    friction_buf: wp.array[float]
    match_index_buf: wp.array[wp.int32]
    has_shape_props: int
    has_match_index: int


@wp.kernel(enable_backward=False)
def _backup_full_kernel(
    data: _FullContactArrays,
    count: wp.array[int],
    sort_keys_src: wp.array[wp.int64],
    sort_keys_dst: wp.array[wp.int64],
    sort_indices: wp.array[wp.int32],
):
    """Prepare sort keys and copy active contacts into scratch buffers."""
    i = wp.tid()
    sort_indices[i] = wp.int32(i)
    if i >= count[0]:
        sort_keys_dst[i] = SORT_KEY_SENTINEL
        return
    sort_keys_dst[i] = sort_keys_src[i]
    data.shape0_buf[i] = data.shape0[i]
    data.shape1_buf[i] = data.shape1[i]
    data.point0_buf[i] = data.point0[i]
    data.point1_buf[i] = data.point1[i]
    data.offset0_buf[i] = data.offset0[i]
    data.offset1_buf[i] = data.offset1[i]
    data.normal_buf[i] = data.normal[i]
    data.margin0_buf[i] = data.margin0[i]
    data.margin1_buf[i] = data.margin1[i]
    data.tids_buf[i] = data.tids[i]
    if data.has_shape_props != 0:
        data.stiffness_buf[i] = data.stiffness[i]
        data.damping_buf[i] = data.damping[i]
        data.friction_buf[i] = data.friction[i]
    if data.has_match_index != 0:
        data.match_index_buf[i] = data.match_index[i]


@wp.kernel(enable_backward=False)
def _gather_full_kernel(data: _FullContactArrays, perm: wp.array[wp.int32], count: wp.array[int]):
    """Permuted write-back from scratch into live arrays."""
    i = wp.tid()
    if i >= count[0]:
        return
    p = perm[i]
    data.shape0[i] = data.shape0_buf[p]
    data.shape1[i] = data.shape1_buf[p]
    data.point0[i] = data.point0_buf[p]
    data.point1[i] = data.point1_buf[p]
    data.offset0[i] = data.offset0_buf[p]
    data.offset1[i] = data.offset1_buf[p]
    data.normal[i] = data.normal_buf[p]
    data.margin0[i] = data.margin0_buf[p]
    data.margin1[i] = data.margin1_buf[p]
    data.tids[i] = data.tids_buf[p]
    if data.has_shape_props != 0:
        data.stiffness[i] = data.stiffness_buf[p]
        data.damping[i] = data.damping_buf[p]
        data.friction[i] = data.friction_buf[p]
    if data.has_match_index != 0:
        data.match_index[i] = data.match_index_buf[p]


class ContactSorter:
    """Sort contact arrays into a deterministic canonical order.

    Pre-allocates double-buffer arrays and permutation indices at construction
    time so that the per-frame :meth:`sort_simple` / :meth:`sort_full` calls
    are allocation-free and fully CUDA-graph-capturable (no host synchronization).

    The buffers hold the ready rows of the contact pool they grow with. Each
    sort publishes the live contacts on the device, at most those rows, and
    sorts only them; every launch follows that count.

    Set ``allocate_simple_scratch=False`` for a sorter used only by
    :meth:`sort_full` to avoid allocating the unused simple-layout scratch.
    Scratch allocation is fixed at construction time.
    """

    def __init__(
        self,
        capacity: int,
        *,
        per_contact_shape_properties: bool = False,
        allocate_simple_scratch: bool = True,
        device: Devicelike = None,
        pool: ContactPool | None = None,
    ):
        """
        Args:
            capacity: Initial contact slots to sort; ignored when ``pool`` is given.
            per_contact_shape_properties: Provision scratch for per-contact stiffness, damping and friction.
            allocate_simple_scratch: Provision the simple-layout scratch :meth:`sort_simple` needs.
            device: Device to allocate on.
            pool: Contact pool whose ready rows the buffers hold and grow with; by default a pool
                starting at ``capacity`` rows.
        """
        with wp.ScopedDevice(device):
            self._has_shape_props = per_contact_shape_properties
            if pool is None:
                pool = ContactPool(device=wp.get_device(), rows=capacity)
            self._has_simple_scratch = allocate_simple_scratch
            fields = [
                # The live sort merges through a second half and loads whole chunks past the first.
                ContactField("_sort_indices", wp.int32, per_contact=2, extra=int(SORT_CHUNK)),
                ContactField("_sort_keys_copy", wp.int64, SORT_KEY_SENTINEL, per_contact=2, extra=int(SORT_CHUNK)),
            ]
            if allocate_simple_scratch:
                # Scratch buffers for the simple gather (NarrowPhase.launch path).
                fields += [
                    ContactField("_simple_pair_buf", wp.vec2i),
                    ContactField("_simple_position_buf", wp.vec3),
                    ContactField("_simple_normal_buf", wp.vec3),
                    ContactField("_simple_penetration_buf", wp.float32),
                    ContactField("_simple_tangent_buf", wp.vec3),
                ]
            # Scratch buffers for the full gather (CollisionPipeline.collide path).
            fields += [
                ContactField("_full_shape0_buf", wp.int32),
                ContactField("_full_shape1_buf", wp.int32),
                ContactField("_full_point0_buf", wp.vec3),
                ContactField("_full_point1_buf", wp.vec3),
                ContactField("_full_offset0_buf", wp.vec3),
                ContactField("_full_offset1_buf", wp.vec3),
                ContactField("_full_normal_buf", wp.vec3),
                ContactField("_full_margin0_buf", wp.float32),
                ContactField("_full_margin1_buf", wp.float32),
                ContactField("_full_tids_buf", wp.int32),
                ContactField("_full_match_index_buf", wp.int32),
            ]
            if per_contact_shape_properties:
                fields += [
                    ContactField(name, wp.float32)
                    for name in ("_full_stiffness_buf", "_full_damping_buf", "_full_friction_buf")
                ]
            else:
                self._full_stiffness_buf = wp.zeros(0, dtype=float)
                self._full_damping_buf = wp.zeros(0, dtype=float)
                self._full_friction_buf = wp.zeros(0, dtype=float)
            self._simple_match_index_buf = wp.zeros(1, dtype=wp.int32)
            # The sort extent: the live contacts first (live_sort.publish_live).
            self._extent = wp.zeros(EXTENT_SIZE, dtype=int)
            self._pool = pool
            self._capacity = pool.ready
            self._rows = pool.rows(fields, on_rows=self._on_rows)

    def _on_rows(self, rows) -> None:
        """Assign the buffers at the pool's ready rows, on creation and after each growth."""
        rows.publish(self)
        self._capacity = rows.rows

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def borrow_full_scratch(self) -> dict:
        """Expose the full-layout scratch buffers for reuse by a sequential stage.

        The buffers are only live INSIDE :meth:`sort_full`; between calls their
        contents are dead, so a pipeline stage that runs strictly after the sort
        (and finishes before the next one) may reuse them instead of allocating
        a second full-size copy of every contact array. The borrower must treat
        the contents as clobbered by every :meth:`sort_full` call.

        Returns:
            Mapping from contact field name to the scratch array. The
            ``stiffness``/``damping``/``friction`` entries are zero-length
            unless the sorter was built with ``per_contact_shape_properties``.
        """
        return {
            "shape0": self._full_shape0_buf,
            "shape1": self._full_shape1_buf,
            "point0": self._full_point0_buf,
            "point1": self._full_point1_buf,
            "offset0": self._full_offset0_buf,
            "offset1": self._full_offset1_buf,
            "normal": self._full_normal_buf,
            "margin0": self._full_margin0_buf,
            "margin1": self._full_margin1_buf,
            "tids": self._full_tids_buf,
            "stiffness": self._full_stiffness_buf,
            "damping": self._full_damping_buf,
            "friction": self._full_friction_buf,
        }

    def sort_simple(
        self,
        sort_keys: wp.array,
        contact_count: wp.array,
        *,
        contact_pair: wp.array,
        contact_position: wp.array,
        contact_normal: wp.array,
        contact_penetration: wp.array,
        contact_tangent: wp.array | None = None,
        match_index: wp.array | None = None,
        device: Devicelike = None,
    ) -> None:
        """Sort contacts written through the simplified narrow-phase writer.

        Fully graph-capturable — no host synchronization.

        Args:
            sort_keys: Per-contact int64 sort keys (filled by the writer).
            contact_count: Single-element int array with the active contact count.
            contact_pair: vec2i shape pair array.
            contact_position: vec3 contact positions.
            contact_normal: vec3 contact normals.
            contact_penetration: float penetration depths.
            contact_tangent: Optional vec3 tangent array.
            match_index: Optional int32 array of per-contact match indices
                from :class:`ContactMatcher`.  When provided, the array is
                permuted alongside the other contact fields during sorting.
            device: Device to launch on.
        """
        if not self._has_simple_scratch:
            raise ValueError("sort_simple requires allocate_simple_scratch=True")
        has_tangent = contact_tangent is not None and contact_tangent.shape[0] > 0
        has_match = match_index is not None and match_index.shape[0] > 0

        data = _SimpleContactArrays()
        data.pair = contact_pair
        data.position = contact_position
        data.normal = contact_normal
        data.penetration = contact_penetration
        data.tangent = contact_tangent if has_tangent else self._simple_tangent_buf
        data.match_index = match_index if has_match else self._simple_match_index_buf
        data.pair_buf = self._simple_pair_buf
        data.position_buf = self._simple_position_buf
        data.normal_buf = self._simple_normal_buf
        data.penetration_buf = self._simple_penetration_buf
        data.tangent_buf = self._simple_tangent_buf
        data.match_index_buf = self._simple_match_index_buf
        data.has_tangent = 1 if has_tangent else 0
        data.has_match_index = 1 if has_match else 0

        self._sort(_backup_simple_kernel, _gather_simple_kernel, data, sort_keys, contact_count)

    def sort_full(
        self,
        sort_keys: wp.array,
        contact_count: wp.array,
        *,
        shape0: wp.array,
        shape1: wp.array,
        point0: wp.array,
        point1: wp.array,
        offset0: wp.array,
        offset1: wp.array,
        normal: wp.array,
        margin0: wp.array,
        margin1: wp.array,
        tids: wp.array,
        stiffness: wp.array | None = None,
        damping: wp.array | None = None,
        friction: wp.array | None = None,
        match_index: wp.array | None = None,
        device: Devicelike = None,
    ) -> None:
        """Sort contacts written through the full collide.py writer.

        Fully graph-capturable — no host synchronization.

        Args:
            sort_keys: Per-contact int64 sort keys (filled by the writer).
            contact_count: Single-element int array with the active contact count.
            shape0: int array of first shape indices.
            shape1: int array of second shape indices.
            point0: vec3 body-frame contact points on shape 0.
            point1: vec3 body-frame contact points on shape 1.
            offset0: vec3 body-frame friction anchor offsets for shape 0.
            offset1: vec3 body-frame friction anchor offsets for shape 1.
            normal: vec3 contact normals.
            margin0: float surface thickness for shape 0.
            margin1: float surface thickness for shape 1.
            tids: int tid array.
            stiffness: Optional float per-contact stiffness.
            damping: Optional float per-contact damping.
            friction: Optional float per-contact friction.
            match_index: Optional int32 array of per-contact match indices
                from :class:`ContactMatcher`.  When provided, the array is
                permuted alongside the other contact fields during sorting.
            device: Device to launch on.
        """
        # Scratch provisioning and the current buffer schema are separate.
        # A reducer-enabled deterministic pipeline provisions material scratch
        # so a later external rich Contacts buffer is safe, but its ordinary
        # property-less buffer must not run an in-place permutation over the
        # scratch placeholders themselves.
        has_props = (
            self._has_shape_props
            and stiffness is not None
            and damping is not None
            and friction is not None
            and stiffness.shape[0] > 0
            and damping.shape[0] > 0
            and friction.shape[0] > 0
        )
        has_match = match_index is not None and match_index.shape[0] > 0

        data = _FullContactArrays()
        data.shape0 = shape0
        data.shape1 = shape1
        data.point0 = point0
        data.point1 = point1
        data.offset0 = offset0
        data.offset1 = offset1
        data.normal = normal
        data.margin0 = margin0
        data.margin1 = margin1
        data.tids = tids
        data.stiffness = (
            stiffness if has_props and stiffness is not None and stiffness.shape[0] > 0 else self._full_stiffness_buf
        )
        data.damping = damping if has_props and damping is not None and damping.shape[0] > 0 else self._full_damping_buf
        data.friction = (
            friction if has_props and friction is not None and friction.shape[0] > 0 else self._full_friction_buf
        )
        data.match_index = match_index if has_match else self._full_match_index_buf
        data.shape0_buf = self._full_shape0_buf
        data.shape1_buf = self._full_shape1_buf
        data.point0_buf = self._full_point0_buf
        data.point1_buf = self._full_point1_buf
        data.offset0_buf = self._full_offset0_buf
        data.offset1_buf = self._full_offset1_buf
        data.normal_buf = self._full_normal_buf
        data.margin0_buf = self._full_margin0_buf
        data.margin1_buf = self._full_margin1_buf
        data.tids_buf = self._full_tids_buf
        data.stiffness_buf = self._full_stiffness_buf
        data.damping_buf = self._full_damping_buf
        data.friction_buf = self._full_friction_buf
        data.match_index_buf = self._full_match_index_buf
        data.has_shape_props = 1 if has_props else 0
        data.has_match_index = 1 if has_match else 0

        self._sort(_backup_full_kernel, _gather_full_kernel, data, sort_keys, contact_count)

    def _sort(self, backup, gather, data, sort_keys, contact_count):
        """Back up the live contacts, sort their keys, and gather them in key order.

        The live contacts are published first, at most the ready rows, and
        every launch follows them; in a capture whose launches bind, each replay
        sorts the contacts of that replay.
        """
        pool = self._pool
        publish_live(pool, contact_count, self._extent)
        live = self._extent[0:1]
        rows = pool.launch_rows()
        keys = self._rows.reserved["_sort_keys_copy"]
        indices = self._rows.reserved["_sort_indices"]
        pool.launch(backup, rows, inputs=[data, live, sort_keys, keys, indices], extent=live)
        blocks, merges, merge_threads = sort_grid(pool.device, rows)
        launch_live_sort(
            pool,
            keys,
            indices,
            self._extent,
            self._capacity,
            blocks=blocks,
            merges=merges,
            merge_threads=merge_threads,
            slot_keys=False,
        )
        pool.launch(gather, rows, inputs=[data, indices, live], extent=live)

    @property
    def sorted_keys_view(self) -> wp.array:
        """View of sorted keys (first half of internal buffer).

        Valid only after :meth:`sort_simple` or :meth:`sort_full` returns.
        The array has ``capacity`` elements; active entries are
        ``sorted_keys_view[:contact_count]``.
        """
        return self._sort_keys_copy[: self._capacity]

    @property
    def scratch_pos_world(self) -> wp.array:
        """Transient position scratch available to sequential pipeline stages.

        Sized ``capacity`` :class:`wp.vec3`.  :meth:`sort_full` overwrites this
        storage on every call, so it cannot hold state across frames;
        :class:`~newton._src.geometry.contact_match.ContactMatcher` keeps its
        previous-frame history in buffers it owns.

        .. note::
            Contents are only meaningful to the stage that wrote them, until
            the next :meth:`sort_full` call.  Coordinate any reuse with the
            pipeline's per-frame call order.
        """
        return self._full_point0_buf

    @property
    def scratch_normal(self) -> wp.array:
        """Transient normal scratch available to sequential pipeline stages.

        Sized ``capacity`` :class:`wp.vec3`.  Companion to
        :attr:`scratch_pos_world`; see that property for usage constraints.
        """
        return self._full_normal_buf
