# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Explicit guarded publication around the shared contact-force law."""

import warp as wp

from .kernels import _contact_linear_force, _pack_contact_linear_force


@wp.kernel(module="unique", enable_backward=False)
def guarded_compute_contact_linear_force_from_impulses(
    contact_count: wp.array[wp.int32],
    contact_normal: wp.array[wp.vec3],
    contact_world: wp.array[wp.int32],
    contact_slot: wp.array[wp.int32],
    contact_path: wp.array[wp.int32],
    world_impulses: wp.array2d[wp.float32],
    mf_impulses: wp.array2d[wp.float32],
    propagation_impulses: wp.array2d[wp.float32],
    world_constraint_count: wp.array[wp.int32],
    mf_constraint_count: wp.array[wp.int32],
    propagation_constraint_count: wp.array[wp.int32],
    world_row_type: wp.array2d[wp.int32],
    world_row_parent: wp.array2d[wp.int32],
    mf_row_type: wp.array2d[wp.int32],
    mf_row_parent: wp.array2d[wp.int32],
    propagation_row_type: wp.array2d[wp.int32],
    propagation_row_parent: wp.array2d[wp.int32],
    enable_friction: int,
    inv_dt: float,
    # outputs
    rigid_contact_force: wp.array[wp.vec3],
    frame_status: wp.array[int],
):
    """Publish the shared force law only after the frame guard passes."""
    if frame_status[0] != 0:
        return
    _contact_linear_force(
        wp.tid(),
        contact_count,
        contact_normal,
        contact_world,
        contact_slot,
        contact_path,
        world_impulses,
        mf_impulses,
        propagation_impulses,
        world_constraint_count,
        mf_constraint_count,
        propagation_constraint_count,
        world_row_type,
        world_row_parent,
        mf_row_type,
        mf_row_parent,
        propagation_row_type,
        propagation_row_parent,
        enable_friction,
        inv_dt,
        rigid_contact_force,
    )


@wp.kernel(module="unique", enable_backward=False)
def guarded_pack_contact_linear_force_as_spatial(
    contact_count: wp.array[wp.int32],
    rigid_contact_force: wp.array[wp.vec3],
    # outputs
    contact_force: wp.array[wp.spatial_vector],
    frame_status: wp.array[int],
):
    """Publish the shared force law only after the frame guard passes."""
    if frame_status[0] != 0:
        return
    _pack_contact_linear_force(
        wp.tid(),
        contact_count,
        rigid_contact_force,
        contact_force,
    )


def get_kernel(name):
    """Return one of the two supported guarded publication owners."""
    if name == "compute_contact_linear_force_from_impulses":
        return guarded_compute_contact_linear_force_from_impulses
    if name == "pack_contact_linear_force_as_spatial":
        return guarded_pack_contact_linear_force_as_spatial
    raise ValueError("Only original public contact force kernels are allowed")
