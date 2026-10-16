# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bind complete cached publication and reset ownership to live solver arrays."""

import functools

import warp as wp

from .publication_data import CACHE_FIELDS, MODEL_FIELDS, SOLVER_FIELDS, PublicationData

_DUMMY_CLOCKS = {}


def stage_clocks(solver):
    """The owner's stage-clock buffer when instrumentation is on, else a shared 1x1 dummy."""
    clocks = getattr(solver, "_stage_clocks", None)
    if clocks is not None:
        return clocks
    key = str(solver.model.device)
    if key not in _DUMMY_CLOCKS:
        _DUMMY_CLOCKS[key] = wp.zeros((1, 1), dtype=wp.int64, device=solver.model.device)
    return _DUMMY_CLOCKS[key]


def publication(solver, state_in, state_aug, state_out, dt):
    """Rebind numeric descriptors on each call, preserving original free finalization."""
    data = PublicationData()
    for name in MODEL_FIELDS:
        setattr(data, name, getattr(solver.model, name))
    for name, source in SOLVER_FIELDS.items():
        setattr(data, name, getattr(solver, source))
    data.joint_q, data.joint_qd = state_in.joint_q, state_in.joint_qd
    data.joint_q_new, data.joint_qd_new = state_out.joint_q, state_out.joint_qd
    data.joint_qdd = state_aug.joint_qdd
    data.body_q, data.body_qd = state_out.body_q, state_out.body_qd
    for name in CACHE_FIELDS:
        if name == "articulation_origin":
            value = solver.articulation_origin
        elif name == "body_inertia_terms":
            value = solver._body_inertia_terms
        else:
            value = getattr(state_aug, name)
        setattr(data, name, value)
    data.dt, data.angular_damping = dt, solver.angular_damping
    # Free finalization still materializes its original I_s on held steps.
    data.materialize_all_body_inertia = 0
    data.materialize_body_inertia_terms = 0
    data.stage_clocks = stage_clocks(solver)
    return data


def disjoint_states(state_in, state_out):
    """Check all generalized input/output byte ranges before any integration write."""
    return not any(
        a.ptr < b.ptr + b.capacity and b.ptr < a.ptr + a.capacity
        for a in (state_in.joint_q, state_in.joint_qd)
        for b in (state_out.joint_q, state_out.joint_qd)
    )


@functools.cache
def _invalidate_kernel(articulations_per_cache):
    """Specialize only the number of articulations owned by one current cache."""

    @wp.kernel(module="unique", enable_backward=False)
    def invalidate(
        world_mask: wp.array[wp.bool],
        cache_to_art: wp.array2d[int],
        art_to_world: wp.array[int],
        current_valid: wp.array[int],
        geometry_valid: wp.array[int],
    ):
        cache = wp.tid()
        if not world_mask or world_mask[art_to_world[cache_to_art[cache, 0]]]:
            for index in range(wp.static(articulations_per_cache)):
                current_valid[cache_to_art[cache, index]] = 0
            geometry_valid[cache] = 0

    return invalidate


def invalidate(world_count, world_mask, cache_to_art, art_to_world, current_valid, geometry_valid):
    """Map world reset masks through the owner's complete, prevalidated forest."""
    if world_mask is not None and (
        world_mask.dtype != wp.bool
        or world_mask.shape != (world_count,)
        or world_mask.device != cache_to_art.device
        or not world_mask.is_contiguous
    ):
        raise ValueError("Cached state reset requires the original contiguous per-world bool mask")
    wp.launch(
        _invalidate_kernel(cache_to_art.shape[1]),
        dim=cache_to_art.shape[0],
        inputs=[world_mask, cache_to_art, art_to_world, current_valid, geometry_valid],
        device=cache_to_art.device,
    )
