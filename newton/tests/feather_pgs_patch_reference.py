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
functions that the mechanism work leaves unchanged are imported, and so is the
per-pair and per-contact work of the one-wrench law (``_build_pair``,
``_finish_contact``), which is law, not mechanism: this module keeps the launches,
their extents and the history's storage of the old mechanism. ``_prepare`` forms
the gaps from body-relative points (``contact_laws.contact_points``,
``_surface_gap``), which is not mechanism work either. This module holds shared test helpers; it defines no tests.
"""

from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.contact_filters import contact_normal_gap_limit
from newton._src.solvers.feather_pgs.contact_laws import contact_point, contact_points
from newton._src.solvers.feather_pgs.friction_patches import (
    FrictionPatches,
    _build_pair,
    _finish_contact,
    _local_surface,
    _PatchFrame,
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
    frame.pressured[c] = 0
    patches.weight[c] = 0.0
    patches.next_contact[c] = -1
    patches.phi[c] = wp.vec2(0.0)
    patches.spin_phi[c] = 0.0
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
    n, translation_a, offset_a, translation_b, offset_b, separation, gap = contact_points(
        normal[c], point0[c], point1[c], margin0[c], margin1[c], a, b, q
    )
    surface_a = _local_surface(q, a, translation_a, offset_a)
    surface_b = _local_surface(q, b, translation_b, offset_b)
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
    frame.surface_a[c] = surface_b if flip != 0 else surface_a
    frame.surface_b[c] = surface_a if flip != 0 else surface_b
    frame.center[c] = contact_point(translation_a, offset_a, separation)
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
    """One thread per sorted body pair, over the whole capacity; the pair's build is the module's (the one-wrench law),
    its previous pairs searched over the whole history."""
    start = wp.tid()
    end = wp.min(count[0], frame.center.shape[0])
    if start >= end:
        return
    _build_pair(start, end, prev.center.shape[0], q, previous_q, shape_transform, shape_radius, frame, prev, patches)


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
    """Link only allocated rows: each normal row's parent is the next member's row of its region on the route. A
    region has one anchor, so the friction coefficients stay whole."""
    c = wp.tid()
    if c >= count[0] or slot[c] < 0 or path[c] != route:
        return
    row = patches.next_contact[c]
    if row < 0:
        return
    next_slot = slot[c]
    while row != c:
        if slot[row] >= 0 and path[row] == route and world[row] == world[c]:
            next_slot = slot[row]
            break
        row = patches.next_contact[row]
    parents[world[c], slot[c]] = next_slot


@wp.kernel(enable_backward=False)
def finish_patch_impulses(
    count: wp.array[int],
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
    route: int,
    parents: wp.array2d[int],
    mu: wp.array2d[float],
    impulses: wp.array2d[float],
    dt: float,
    first: wp.array[int],
):
    """Cache solved friction and release saturated anchors with slip, one launch per route over the whole capacity:
    the contact's finish is the module's (the one-wrench law). A contact without rows (capacity rejection) keeps its
    geometrically supported history until rows are available again."""
    c = wp.tid()
    if c >= count[0] or frame.valid[c] == 0 or slot[c] < 0 or path[c] != route:
        return
    _finish_contact(
        c, frame, patches, q, q_out, qd_out, com, world, slot, path, slots_needed, parents, mu, impulses, dt, first
    )


@wp.kernel
def reset_friction_anchor_history(
    world_mask: wp.array[wp.bool],
    prev_world: wp.array[int],
    prev_valid: wp.array[int],
    prev_pressured: wp.array[int],
):
    """Drop the carried friction anchors and centres of pressure of reset worlds (all worlds for an empty mask)."""
    c = wp.tid()
    if world_mask:
        w = prev_world[c]
        if w >= 0 and w < world_mask.shape[0] and not world_mask[w]:
            return
    prev_valid[c] = 0
    prev_pressured[c] = 0


class ReferencePatches:
    """The 98401301 patch state's launches, on the one-wrench law's per-contact work, driven through the solver's launch
    sequence.

    Construction, geometry updates, :meth:`build` and :meth:`store` copy
    ``_FrictionPatchState`` with patches enabled. :meth:`link`, :meth:`finish` and
    :meth:`reset` launch the per-route kernels and the history reset as the solver
    did. ``routes`` holds ``(route, parents, mu, impulses, first)`` tuples.
    """

    def __init__(self, model, capacity, phi):
        self.view = FrictionPatches()
        self.view.phi = phi
        self.capacity = capacity
        device = model.device
        self.view.weight = wp.zeros(capacity, dtype=float, device=device)
        self.view.next_contact = wp.full(capacity, -1, dtype=int, device=device)
        self.view.point = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self.view.spin_arm = wp.zeros(capacity, dtype=float, device=device)
        self.view.spin_phi = wp.zeros(capacity, dtype=float, device=device)
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
            "pressure",
        ):
            setattr(frame, field, wp.zeros(n, dtype=wp.vec3, device=device))
        frame.mu = wp.zeros(n, dtype=wp.vec2, device=device)
        frame.radius = wp.zeros(n, dtype=float, device=device)
        frame.support_gap_limit = wp.zeros(n, dtype=float, device=device)
        frame.twist = wp.zeros(n, dtype=float, device=device)
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
            "pressured",
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
        for route, parents, mu, _impulses, _first in routes:
            wp.launch(
                link_patch_rows,
                dim=contacts.rigid_contact_max,
                inputs=[contacts.rigid_contact_count, self.view, world, slot, path, slots_needed, route, parents, mu],
                device=world.device,
            )

    def finish(self, contacts, state_in, state_out, com, world, slot, path, slots_needed, routes, dt):
        """Release or cache solved anchors per route, then store the frame; ``None`` contacts drop history."""
        if contacts is None:
            self.previous.valid.zero_()
            return
        for route, parents, mu, impulses, first in routes:
            wp.launch(
                finish_patch_impulses,
                dim=contacts.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    self.current,
                    self.view,
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
                    first,
                ],
                device=world.device,
            )
        self.store(state_in)

    def reset(self, world_mask):
        """Drop the stored anchors of the masked worlds over the whole history."""
        wp.launch(
            reset_friction_anchor_history,
            dim=self.previous.valid.shape[0],
            inputs=[world_mask, self.previous_world, self.previous.valid, self.previous.pressured],
            device=self.previous.valid.device,
        )
