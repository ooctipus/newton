# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bind complete cached publication and reset ownership to live solver arrays."""

import functools

import warp as wp

from . import field_group
from .publication_data import PublicationData


def stage_clocks(solver):
    """The owner's stage-clock buffer when instrumentation is on, else the solver's 1x1 placeholder."""
    clocks = getattr(solver, "_stage_clocks", None)
    if clocks is not None:
        return clocks
    return solver._arrays.placeholder(wp.int64, (1, 1))


def publication(solver, state_in, state_aug, state_out, dt):
    """Rebind numeric descriptors on each call, preserving original free finalization."""
    return field_group.bind(
        PublicationData,
        solver=solver,
        forest=solver.forest,
        options=solver.options,
        model=solver.model,
        state_in=state_in,
        state_aug=state_aug,
        state_out=state_out,
        dt=dt,
        stage_clocks=stage_clocks(solver),
    )


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
