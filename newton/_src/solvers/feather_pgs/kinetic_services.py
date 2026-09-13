# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unchanged original current-MF producers and mixed consumer for kinetic rows.

No saved MF inverse, coefficients, RHS or solved state is a candidate input.
Allocation is an external service until the separate current allocator is bound.
"""

import inspect
import json
from types import SimpleNamespace

import numpy as np
import warp as wp

from . import kernels as k
from . import solver_feather_pgs as solver
from .kinetic_source import checked_definitions


@wp.func_native("union { float value; int bits; } v; v.value = value; return v.bits;")
def _float_bits(value: float) -> int: ...


@wp.kernel(enable_backward=False)
def pack_mf_cpu_reference(
    counts: wp.array[int],
    dof_a: wp.array2d[int],
    dof_b: wp.array2d[int],
    diagonal: wp.array2d[float],
    rhs: wp.array2d[float],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    meta: wp.array2d[int],
):
    """CPU-only literal bit layout; the original native pack has an empty CPU body."""
    world, row = wp.tid()
    if row < counts[world]:
        meta[world, 4 * row] = (dof_a[world, row] << 16) | (dof_b[world, row] & 65535)
        meta[world, 4 * row + 1] = _float_bits(diagonal[world, row])
        meta[world, 4 * row + 2] = _float_bits(rhs[world, row])
        meta[world, 4 * row + 3] = row_type[world, row] | (row_parent[world, row] << 16)


def bind_mf(rows, *, v_out=None, selector=None):
    """Use original free-body order/current public inertia and exact MF capacities."""
    checked_definitions(
        "kernels.py",
        (
            "compute_mf_body_Hinv",
            "build_mf_contact_rows",
            "populate_rigid_velocity_limit_rows",
            "compute_mf_world_dof_offsets",
            "compute_mf_effective_mass_and_rhs",
        ),
    )
    checked_definitions("solver_feather_pgs.py", ("_get_pack_mf_meta_kernel", "_get_pgs_solve_mf_gs_kernel"))
    if rows.bound is None:
        raise ValueError("Current MF requires the produced canonical free-body service")
    a = rows.bundle["host"]["snapshot"]
    raw, state, out, settings = rows.raw, rows.state, rows.out, rows.settings
    publication = rows.bound["publication"]
    worlds, device = rows.worlds, rows.device
    meta = json.loads(a["metadata_json"].item())

    def array(value, dtype=wp.float32):
        return wp.array(value, dtype=dtype, device=device)

    def zeros(shape, dtype=wp.float32):
        return wp.zeros(shape, dtype=dtype, device=device)

    mf = SimpleNamespace()
    for name in ("body_a", "body_b", "dof_a", "dof_b", "row_type", "row_parent"):
        setattr(mf, name, wp.full((worlds, 64), -1, dtype=int, device=device))
    for name in (
        "row_mu",
        "phi",
        "target_velocity",
        "row_restitution",
        "rhs",
        "eff_mass_inv",
        "impulses",
    ):
        setattr(mf, name, zeros((worlds, 64)))
    for name in ("J_a", "J_b", "MiJt_a", "MiJt_b"):
        setattr(mf, name, zeros((worlds, 64, 6)))
    mf.meta = zeros((worlds, 64 * 4), int)
    mf.row_w = zeros((1, 1))
    mf.contact_end = zeros(worlds, int)
    mf.free_bodies = array(a["solve_free_rigid_body_indices"], int)
    mf.max_linear = array(a["post3_solver_rigid_body_max_linear_velocity"])
    mf.max_angular = array(a["post3_solver_rigid_body_max_angular_velocity"])
    mf.max_depenetration = array(a["solve_rigid_body_max_depenetration_velocity"])
    mf.limit_slot = wp.full(12 * mf.free_bodies.size, -1, dtype=int, device=device)
    mf.limit_sign = zeros(12 * mf.free_bodies.size)
    mf.body_Hinv = wp.full(
        publication.body_I_s.shape,
        wp.spatial_matrix(float("nan")),
        dtype=wp.spatial_matrix,
        device=device,
    )
    mf.identity = array(np.arange(worlds, dtype=np.int32), int)
    mf.selector = zeros(worlds, int) if selector is None else selector
    mf.v_out = wp.clone(state.v_hat) if v_out is None else v_out
    art_dof_start = array(a["solver_articulation_dof_start"], int)
    art_offsets = array(a["solver_articulation_world_dof_offset"], int)
    world_dofs = array(a["solve_world_dof_indices"], int)
    body_flags = rows.bundle["inputs"].body_flags
    has_prescribed = int(bool(meta["_has_prescribed_response"]))
    workers = min(
        settings.raw_capacity,
        solver._CONTACT_BUILD_THREAD_CAP * solver._CONTACT_THREADS_X,
    )
    mf.inverse_args = [
        mf.free_bodies,
        publication.body_I_s,
        raw.is_free_rigid,
        raw.body_to_articulation,
        body_flags,
        mf.body_Hinv,
    ]
    mf.contact_args = [
        raw.count,
        workers,
        raw.point0,
        raw.point1,
        raw.normal,
        raw.shape0,
        raw.shape1,
        raw.margin0,
        raw.margin1,
        raw.world,
        raw.slot,
        raw.path,
        raw.art_a,
        raw.art_b,
        raw.response_count,
        publication.articulation_origin,
        raw.shape_body,
        raw.body_q,
        publication.body_v_s,
        raw.prescribed_articulation,
        has_prescribed,
        raw.shape_mu,
        raw.shape_restitution,
        settings.enable_friction,
        settings.friction_gap_threshold,
        settings.friction_shared_anchor,
        settings.friction_anchor_limit,
        settings.friction_articulation_pairs_only,
        settings.friction_scale,
        settings.shared_anchor,
        mf.body_a,
        mf.body_b,
        mf.J_a,
        mf.J_b,
        mf.row_type,
        mf.row_parent,
        mf.row_mu,
        mf.phi,
        mf.target_velocity,
        mf.row_restitution,
    ]
    mf.limit_args = [
        mf.free_bodies,
        raw.body_to_articulation,
        raw.art_to_world,
        raw.is_free_rigid,
        mf.max_linear,
        mf.max_angular,
        mf.limit_slot,
        mf.limit_sign,
        mf.body_a,
        mf.body_b,
        mf.J_a,
        mf.J_b,
        mf.row_type,
        mf.row_parent,
        mf.row_mu,
        mf.phi,
    ]
    mf.offset_args = [
        state.mf_count,
        mf.body_a,
        mf.body_b,
        raw.body_to_articulation,
        art_offsets,
        64,
        mf.dof_a,
        mf.dof_b,
    ]
    mf.mass_args = [
        state.mf_count,
        mf.body_a,
        mf.body_b,
        mf.J_a,
        mf.J_b,
        mf.body_Hinv,
        mf.phi,
        mf.row_type,
        mf.target_velocity,
        mf.row_restitution,
        has_prescribed,
        raw.body_to_articulation,
        art_dof_start,
        state.v_hat,
        mf.max_depenetration,
        settings.cfm,
        settings.beta,
        settings.contact_w,
        settings.dt,
        settings.contact_speculative_scale,
        settings.restitution_velocity_threshold,
        64,
        mf.eff_mass_inv,
        mf.MiJt_a,
        mf.MiJt_b,
        mf.rhs,
        mf.row_w,
    ]
    mf.pack = solver._get_pack_mf_meta_kernel(64, str(device.arch))
    mf.pack_args = [
        state.mf_count,
        mf.dof_a,
        mf.dof_b,
        mf.eff_mass_inv,
        mf.rhs,
        mf.row_type,
        mf.row_parent,
        mf.meta,
    ]
    mf.general = solver._get_pgs_solve_mf_gs_kernel(
        192,
        64,
        29,
        str(device.arch),
        has_drive_rows=False,
        has_dense_velocity_limit_rows=False,
        factor_coordinates=True,
        independent_components=True,
    )
    dummy = zeros((1, 1))
    values = {
        "general_world_count": array([worlds], int),
        "general_worlds": mf.identity,
        "general_world_grid_stride": worlds,
        "use_general_world_queue": 0,
        "rb_class": zeros(worlds, int),
        "world_constraint_count": state.dense_count,
        "dense_phase_bounds": out.phase_bounds,
        "local_solve_owner": mf.selector,
        "world_dof_indices": world_dofs,
        "world_deferred_dof_mask": zeros((worlds, 29), int),
        "rhs_bias": out.rhs,
        "world_diag": out.diag,
        "world_row_w": out.row_w,
        "world_impulses": out.impulses,
        "J_world": out.physical_J,
        "Y_world": out.response,
        "world_row_type": out.row_type,
        "world_row_parent": out.row_parent,
        "world_row_mu": out.row_mu,
        "mf_constraint_count": state.mf_count,
        "mf_contact_rows_end": mf.contact_end,
        "mf_meta": mf.meta,
        "mf_impulses": mf.impulses,
        "mf_J_a": mf.J_a,
        "mf_J_b": mf.J_b,
        "mf_MiJt_a": mf.MiJt_a,
        "mf_MiJt_b": mf.MiJt_b,
        "mf_row_mu": mf.row_mu,
        "mf_row_w": mf.row_w,
        "iterations": 8,
        "omega": 1.0,
        "regularize": 0,
        "row_phase": 0,
        "friction_start_iteration": 0,
        "iteration_offset": 0,
        "freeze_drive_rows": 0,
        "defer_dense_response": 0,
        "v_out": mf.v_out,
    }
    for name in (
        "target_vel_bias",
        "vel_multiplier",
        "impulse_multiplier",
        "max_impulse",
        "vel_limit",
    ):
        values["world_drive_" + name] = dummy
    parameters = inspect.signature(mf.general.func).parameters
    if set(parameters) != set(values):
        raise ValueError("Original general consumer signature changed")
    mf.general_args = [values[name] for name in parameters]

    def prepare():
        """Charge original current inverse and every original consumed MF coefficient."""
        mf.target_velocity.zero_()
        mf.impulses.zero_()
        mf.rhs.zero_()
        mf.eff_mass_inv.zero_()
        wp.launch(
            k.compute_mf_body_Hinv,
            dim=mf.free_bodies.size,
            inputs=mf.inverse_args,
            device=device,
        )
        wp.launch(k.build_mf_contact_rows, dim=workers, inputs=mf.contact_args, device=device)
        wp.launch(
            k.populate_rigid_velocity_limit_rows,
            dim=mf.free_bodies.size,
            inputs=mf.limit_args,
            device=device,
        )
        wp.launch(
            k.compute_mf_world_dof_offsets,
            dim=worlds * 64,
            inputs=mf.offset_args,
            device=device,
        )
        wp.launch(
            k.compute_mf_effective_mass_and_rhs,
            dim=worlds * 64,
            inputs=mf.mass_args,
            device=device,
        )
        if device.is_cuda:
            wp.launch_tiled(mf.pack, dim=[worlds], inputs=mf.pack_args, block_dim=32, device=device)
        else:
            wp.launch(
                pack_mf_cpu_reference,
                dim=(worlds, 64),
                inputs=mf.pack_args,
                device=device,
            )

    def seed_velocity():
        """Copy this call's complete current predictor, including prescribed DOFs."""
        wp.copy(mf.v_out, state.v_hat)

    def solve_general():
        """Original eight-sweep general/MF consumer; selector must already be current."""
        wp.launch_tiled(
            mf.general,
            dim=[worlds],
            inputs=mf.general_args,
            block_dim=32,
            device=device,
        )

    mf.prepare, mf.seed_velocity, mf.solve_general = (
        prepare,
        seed_velocity,
        solve_general,
    )
    mf.art_dof_start, mf.art_offsets, mf.world_dofs = (
        art_dof_start,
        art_offsets,
        world_dofs,
    )
    mf.metadata = {
        "current_inverse_producer_included": True,
        "no_saved_coefficients": True,
        "allocator_included": False,
        "complete_boundary": False,
        "cpu_pack_is_reference": device.is_cpu,
    }
    return mf
