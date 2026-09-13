# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental forward-only contact response for single articulated worlds.

Physical J and the canonical held Cholesky remain authoritative. Existing Y
storage holds Z during an admitted call; no private representation survives it.
"""

from functools import cache

import warp as wp


def supported(solver) -> bool:
    """Admit the first measured topology without changing any solver allowance."""
    return bool(
        solver.model.device.is_cuda
        and not solver.model.requires_grad
        and solver.world_count > 0
        and solver.model.articulation_count == solver.world_count
        and tuple(solver.size_groups) == (43,)
        and solver.max_world_dofs == 43
        and solver.n_arts_by_size[43] == solver.world_count
        and solver._execution_plan.use_tiled_hinv_jt(43)
        and not solver._execution_plan.use_diagonal_mass(43)
        and solver._is_one_solve_art_per_world
        and not solver._has_free_rigid_bodies
        and not solver._preelim_active
        and not solver._regularization_enabled
        and not solver._local_internal_fast_path
        and not solver._paired_factor_coordinates
        and not solver._sparse_diagonal_contact_solve
        and not solver._mf_warmstart_enabled
        and (solver._jy_world_aliased or solver._hinv_jt_writes_world)
        and solver.pgs_mode == "matrix_free"
        and solver.pgs_schedule == "interleaved"
        and solver.pgs_iterations > 0
        and solver.pgs_velocity_iterations == 0
        and not solver.pgs_warmstart
        and not solver.pgs_debug
        and solver.drive_mode == "augmented"
        and solver.friction_mode == "current"
        and not solver.enable_joint_velocity_limits
        and not solver.fuse_joint_velocity_limits
        and solver.articulated_contact_response == "immediate"
        and solver.mf_gs_response_block_rows == 0
        and solver.mf_gs_incremental_rows == 0
        and solver.mf_gs_parallel_rows == 0
        and not solver._propagation_contacts_enabled()
    )


@cache
def get_tiled_response_kernel(dofs: int, capacity: int, chunk: int):
    """Form only L-inverse J-transpose and its squared norm, in existing storage."""
    if not 0 < chunk <= capacity or not 0 < dofs <= 64:
        raise ValueError("Invalid forward-response tile")
    n = wp.constant(dofs)
    rows = wp.constant(chunk)
    bounds = capacity % chunk != 0

    def response(
        L: wp.array3d[float],
        J: wp.array3d[float],
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        counts: wp.array[int],
        J_world: wp.array3d[float],
        Z_world: wp.array3d[float],
        diag: wp.array2d[float],
    ):
        group, row_chunk = wp.tid()
        world = art_to_world[group_to_art[group]]
        first = row_chunk * rows
        if first >= counts[world]:
            return
        factor = wp.tile_load(L[group], shape=(n, n), bounds_check=False)
        jacobian = wp.tile_load(J[group], shape=(rows, n), offset=(first, 0), bounds_check=bounds)
        forward = wp.tile_lower_solve(factor, wp.tile_transpose(jacobian))
        z = wp.tile_transpose(forward)
        diagonal = wp.tile_sum(wp.tile_map(wp.mul, z, z), axis=1)
        wp.tile_store(J_world[world], jacobian, offset=(first, 0), bounds_check=bounds)
        wp.tile_store(Z_world[world], z, offset=(first, 0), bounds_check=bounds)
        wp.tile_store(diag[world], diagonal, offset=first, bounds_check=bounds)

    response.__name__ = response.__qualname__ = f"single_factor_response_{dofs}_{capacity}_c{chunk}"
    return wp.kernel(enable_backward=False, module="unique")(response)


@cache
def get_response_kernel(dofs: int, capacity: int, chunk: int):
    """Share one held factor across all current rows; one warp owns each row.

    The first library implementation retained its 255-register TRSM and three
    shared matrix copies despite removing the backward solve. This owner uses
    one shared L, two forward-response registers per lane, and no tile copies.
    ``chunk`` remains an explicit validated comparison parameter, not a cap.
    """
    if not 0 < chunk <= capacity or not 0 < dofs <= 64:
        raise ValueError("Invalid forward-response tile")
    source = f"""
#if defined(__CUDA_ARCH__)
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int world = art_to_world.data[group_to_art.data[group]];
    const int count = counts.data[world];
    if (count == 0) return;
    __shared__ float factor[{dofs * dofs}];
    const int ls0 = L.strides[0] / sizeof(float);
    const int ls1 = L.strides[1] / sizeof(float);
    const int js0 = J.strides[0] / sizeof(float);
    const int js1 = J.strides[1] / sizeof(float);
    const int ws0 = J_world.strides[0] / sizeof(float);
    const int ws1 = J_world.strides[1] / sizeof(float);
    const int zs0 = Z_world.strides[0] / sizeof(float);
    const int zs1 = Z_world.strides[1] / sizeof(float);
    for (int k = threadIdx.x; k < {dofs * dofs}; k += blockDim.x)
        factor[k] = L.data[group * ls0 + (k / {dofs}) * ls1 + k % {dofs}];
    __syncthreads();
    for (int row = warp_id; row < count && row < {capacity}; row += warps) {{
        const int jbase = group * js0 + row * js1;
        const int wbase = world * ws0 + row * ws1;
        const int zbase = world * zs0 + row * zs1;
        float z0 = lane < {dofs} ? J.data[jbase + lane] : 0.0f;
        float z1 = lane + 32 < {dofs} ? J.data[jbase + lane + 32] : 0.0f;
        if (lane < {dofs}) J_world.data[wbase + lane] = z0;
        if (lane + 32 < {dofs}) J_world.data[wbase + lane + 32] = z1;
        #pragma unroll
        for (int pivot = 0; pivot < {dofs}; ++pivot) {{
            float value = 0.0f;
            if (lane == (pivot & 31)) {{
                if (pivot < 32) {{
                    z0 /= factor[pivot * {dofs} + pivot];
                    value = z0;
                }} else {{
                    z1 /= factor[pivot * {dofs} + pivot];
                    value = z1;
                }}
            }}
            value = __shfl_sync(0xffffffff, value, pivot & 31);
            if (lane > pivot && lane < {dofs})
                z0 -= factor[lane * {dofs} + pivot] * value;
            if (lane + 32 > pivot && lane + 32 < {dofs})
                z1 -= factor[(lane + 32) * {dofs} + pivot] * value;
        }}
        if (lane < {dofs}) Z_world.data[zbase + lane] = z0;
        if (lane + 32 < {dofs}) Z_world.data[zbase + lane + 32] = z1;
        float norm = z0 * z0 + z1 * z1;
        for (int shift = 16; shift > 0; shift >>= 1)
            norm += __shfl_down_sync(0xffffffff, norm, shift);
        if (lane == 0) diag.data[world * (diag.strides[0] / sizeof(float)) + row] = norm;
    }}
#endif
"""

    @wp.func_native(source)
    def native_response(
        group: int,
        L: wp.array3d[float],
        J: wp.array3d[float],
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        counts: wp.array[int],
        J_world: wp.array3d[float],
        Z_world: wp.array3d[float],
        diag: wp.array2d[float],
    ):
        """Apply the original held triangular operator with warp-local rows."""
        ...

    def response(
        L: wp.array3d[float],
        J: wp.array3d[float],
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        counts: wp.array[int],
        J_world: wp.array3d[float],
        Z_world: wp.array3d[float],
        diag: wp.array2d[float],
    ):
        group, _lane = wp.tid()
        native_response(group, L, J, group_to_art, art_to_world, counts, J_world, Z_world, diag)

    response.__name__ = response.__qualname__ = f"single_factor_warp_response_{dofs}_{capacity}"
    return wp.kernel(enable_backward=False, module="unique")(response)


@cache
def get_decode_kernel(dofs: int):
    """Apply one transpose solve to the accumulated kinetic delta per world."""
    n = wp.constant(dofs)

    def decode(
        L: wp.array3d[float],
        group_to_art: wp.array[int],
        art_to_world: wp.array[int],
        articulation_dof_start: wp.array[int],
        counts: wp.array[int],
        v_hat: wp.array[float],
        v_out: wp.array[float],
    ):
        group = wp.tid()
        art = group_to_art[group]
        if counts[art_to_world[art]] == 0:
            return
        start = articulation_dof_start[art]
        factor = wp.tile_load(L[group], shape=(n, n), bounds_check=False)
        delta = wp.tile_load(v_out, shape=(n,), offset=(start,), bounds_check=False)
        physical = wp.tile_upper_solve(wp.tile_transpose(factor), wp.tile_reshape(delta, shape=(n, 1)))
        prediction = wp.tile_load(v_hat, shape=(n,), offset=(start,), bounds_check=False)
        result = prediction + wp.tile_reshape(physical, shape=(n,))
        wp.tile_store(v_out, result, offset=start, bounds_check=False)

    decode.__name__ = decode.__qualname__ = f"single_factor_decode_{dofs}"
    return wp.kernel(enable_backward=False, module="unique")(decode)


class SingleFactor:
    """Replace the complete response/diagonal/solve boundary for admitted calls."""

    def __init__(self, solver):
        from .solver_feather_pgs import _get_pgs_solve_mf_gs_kernel  # noqa: PLC0415

        self.solver = solver
        self.size = 43
        self.chunk = solver._execution_plan.hinv_jt_chunk_size(self.size)
        self.response_kernel = get_response_kernel(self.size, solver.dense_max_constraints, self.chunk)
        self.decode_kernel = get_decode_kernel(self.size)
        self.solve_kernel = _get_pgs_solve_mf_gs_kernel(
            solver.dense_max_constraints,
            solver.mf_max_constraints,
            self.size,
            str(solver.model.device.arch),
            has_drive_rows=False,
            has_dense_velocity_limit_rows=False,
            shared_metadata=True,
            single_factor_coordinates=True,
        )

    def response(self):
        """Retire both the row-wise backward solve and the separate J-dot-Y pass."""
        solver, size = self.solver, self.size
        wp.launch_tiled(
            self.response_kernel,
            dim=(solver.n_arts_by_size[size],),
            inputs=[
                solver.L_by_size[size],
                solver.J_by_size[size],
                solver.group_to_art[size],
                solver.art_to_world,
                solver.constraint_count,
            ],
            outputs=[solver.J_world, solver.Y_world, solver.diag],
            block_dim=solver.tile_threads,
            device=solver.model.device,
        )

    def decode(self):
        """Return physical velocity before clamping, integration or publication."""
        solver, size = self.solver, self.size
        wp.launch_tiled(
            self.decode_kernel,
            dim=(solver.n_arts_by_size[size],),
            inputs=[
                solver.L_by_size[size],
                solver.group_to_art[size],
                solver.art_to_world,
                solver.articulation_dof_start,
                solver.constraint_count,
                solver.v_hat,
            ],
            outputs=[solver.v_out],
            block_dim=solver.tile_threads,
            device=solver.model.device,
        )


def create_owner(solver):
    """Leave unsupported models on their original producers and consumers."""
    return SingleFactor(solver) if supported(solver) else None
