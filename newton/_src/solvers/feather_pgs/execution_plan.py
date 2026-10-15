# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Plan FPGS launch resources without allocating arrays or initializing a device.

These construction-time rules describe storage feasibility and existing routing.
They do not change the numerical method, row capacity, or iteration budget.
"""

from dataclasses import dataclass

_MFGS_RESIDENT_METADATA_MAX_BYTES = 4096
_MFGS_TILE_SHARED_STORAGE_BYTES = 128


def _use_resident_mfgs_metadata(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    max_shared_memory: int,
    *,
    has_drive_rows: bool,
    fuse_vel_limits: bool,
) -> bool:
    """Select resident MF-GS metadata for compact solver shapes.

    Resident row metadata removes repeated global loads across PGS iterations,
    but larger shapes lose more throughput from the resulting occupancy drop.
    Keep at most one 4-KiB metadata working set resident and stream larger sets.
    """
    metadata_arrays = 4 + 4 * int(has_drive_rows) + int(fuse_vel_limits)
    metadata_bytes = 4 * max_constraints * metadata_arrays
    base_shared_bytes = 4 * (max_world_dofs + max_constraints + mf_max_constraints)
    total_shared_bytes = _MFGS_TILE_SHARED_STORAGE_BYTES + base_shared_bytes + metadata_bytes
    return metadata_bytes <= _MFGS_RESIDENT_METADATA_MAX_BYTES and total_shared_bytes <= max_shared_memory


def _select_mfgs_storage(
    max_constraints: int,
    mf_max_constraints: int,
    max_world_dofs: int,
    max_shared_memory: int,
    *,
    has_drive_rows: bool,
    fuse_vel_limits: bool,
) -> tuple[bool, bool]:
    """Select metadata and impulse storage without changing row capacity or order.

    Stream MF impulses only when they cannot fit alongside the shared world
    velocity and dense state. This is a construction-time capacity fallback,
    not a claim that global storage is faster for resident-sized workloads.
    """
    shared_metadata = _use_resident_mfgs_metadata(
        max_constraints,
        mf_max_constraints,
        max_world_dofs,
        max_shared_memory,
        has_drive_rows=has_drive_rows,
        fuse_vel_limits=fuse_vel_limits,
    )
    metadata_arrays = (4 + 4 * int(has_drive_rows) + int(fuse_vel_limits)) if shared_metadata else 0
    fixed_bytes = _MFGS_TILE_SHARED_STORAGE_BYTES + 4 * (max_world_dofs + max_constraints * (1 + metadata_arrays))
    if fixed_bytes > max_shared_memory:
        raise ValueError(
            f"MF-GS world velocity and dense state require {fixed_bytes} bytes of shared memory "
            f"even with streamed MF impulses, but the device exposes {max_shared_memory} bytes"
        )
    return shared_metadata, fixed_bytes + 4 * mf_max_constraints > max_shared_memory


_HINV_JT_MAX_CHUNK_SIZE = 32
_FUSED_ROW_CHUNK = 16


def _fused_response_chunk_size(max_constraints: int) -> int:
    """Select the streamed fused response width or its nondivisible fallback."""
    chunk = min(int(max_constraints), _FUSED_ROW_CHUNK)
    return int(max_constraints) if int(max_constraints) % chunk else chunk


def _align_shared_memory(size: int) -> int:
    """Align a shared-memory allocation to Warp's 16-byte tile boundary."""
    return (size + 15) // 16 * 16


def _estimate_cholesky_shared_memory(n_dofs: int) -> int:
    """Estimate the complete tiled Cholesky shared-memory footprint [B]."""
    matrix_bytes = _align_shared_memory(4 * n_dofs * n_dofs)
    vector_bytes = _align_shared_memory(4 * n_dofs)
    return 3 * matrix_bytes + vector_bytes


def _estimate_hinv_jt_shared_memory(n_dofs: int, constraint_count: int, *, fused: bool, tile_threads: int) -> int:
    """Estimate the complete tiled H-inverse shared-memory footprint [B]."""
    if fused:
        constraint_count = _fused_response_chunk_size(constraint_count)
    footprint = _align_shared_memory(4 * n_dofs * n_dofs)
    # Fusion holds J_i, Z_i, X_i and J_j; the unfused solve has three tiles.
    footprint += (4 if fused else 3) * _align_shared_memory(4 * n_dofs * constraint_count)
    if fused:
        footprint += _align_shared_memory(4 * constraint_count * constraint_count)
    return footprint + 4 * tile_threads


def _select_hinv_jt_chunk_size(
    n_dofs: int, max_constraints: int, max_shared_memory: int, tile_threads: int
) -> int | None:
    """Select a validated H-inverse chunk from articulation and device resources.

    Keep response tiles at 32 rows or fewer. Larger tiles waste triangular-solve
    work on the partially occupied tail common to per-world constraint systems;
    inactive follow-on chunks return before loading the factor.
    """
    if n_dofs <= 0 or max_constraints <= 0:
        return None
    candidates = (_HINV_JT_MAX_CHUNK_SIZE, 16, 8, 4, 2, 1)
    tried: set[int] = set()
    for candidate in candidates:
        chunk_size = min(candidate, max_constraints)
        if chunk_size in tried:
            continue
        tried.add(chunk_size)
        if (
            _estimate_hinv_jt_shared_memory(n_dofs, chunk_size, fused=False, tile_threads=tile_threads)
            <= max_shared_memory
        ):
            return chunk_size
    return None


@dataclass(frozen=True)
class _FeatherPGSExecutionPlan:
    """Immutable device-resource choices for one solver shape."""

    diagonal_mass_sizes: frozenset[int]
    cholesky_tiled_sizes: frozenset[int]
    hinv_jt_tiled_sizes: frozenset[int]
    hinv_jt_chunk_sizes: tuple[tuple[int, int], ...]
    hinv_jt_fused_sizes: frozenset[int]
    reasons: tuple[tuple[int, str, str], ...] = ()

    @classmethod
    def build(
        cls,
        size_groups: list[int],
        *,
        max_constraints: int,
        max_shared_memory: int,
        cholesky_kernel: str,
        hinv_jt_kernel: str,
        small_dof_threshold: int,
        tile_threads: int,
        diagonal_mass_sizes: frozenset[int] = frozenset(),
        fuse_delassus: bool = True,
    ) -> "_FeatherPGSExecutionPlan":
        """Resolve H-inverse implementations from solver shape and device limits."""
        planned_diagonal_sizes = (
            frozenset(size for size in size_groups if size in diagonal_mass_sizes)
            if hinv_jt_kernel == "auto"
            else frozenset()
        )
        cholesky_tiled_sizes: set[int] = set()
        tiled_sizes: set[int] = set()
        chunk_sizes: list[tuple[int, int]] = []
        fused_sizes: set[int] = set()
        reasons: list[tuple[int, str, str]] = []
        for size in size_groups:
            if size in planned_diagonal_sizes:
                for name in ("cholesky_kernel", "hinv_jt_kernel"):
                    reasons.append((size, name, "Independent-coordinate topology uses diagonal generalized inertia."))
                continue
            cholesky_requested = cholesky_kernel == "tiled" or (
                cholesky_kernel == "auto" and size > small_dof_threshold
            )
            if cholesky_requested:
                required = _estimate_cholesky_shared_memory(size)
                if required <= max_shared_memory:
                    cholesky_tiled_sizes.add(size)
                    factor_reason = f"Tiled factor fits shared memory: {required} <= {max_shared_memory} bytes."
                elif cholesky_kernel == "tiled":
                    raise ValueError(
                        f"cholesky_kernel='tiled' requires {required} bytes of shared memory for DOF size {size}, "
                        f"but the device exposes {max_shared_memory} bytes; use cholesky_kernel='auto' or 'loop'."
                    )
                else:
                    factor_reason = (
                        f"Loop fallback: tiled factor needs {required} bytes of shared memory; "
                        f"device limit is {max_shared_memory}."
                    )
            else:
                factor_reason = (
                    "Loop factor explicitly selected."
                    if cholesky_kernel == "loop"
                    else f"Loop factor: {size} DOFs do not exceed the automatic threshold {small_dof_threshold}."
                )
            reasons.append((size, "cholesky_kernel", factor_reason))

            if max_constraints <= 0:
                reasons.append((size, "hinv_jt_kernel", "No tiled response with zero row capacity."))
                continue
            requested = hinv_jt_kernel == "tiled" or (hinv_jt_kernel == "auto" and size > small_dof_threshold)
            if not requested:
                reasons.append(
                    (
                        size,
                        "hinv_jt_kernel",
                        "Parallel-row response explicitly selected."
                        if hinv_jt_kernel == "par_row"
                        else f"Parallel-row response: {size} DOFs do not exceed threshold {small_dof_threshold}.",
                    )
                )
                continue
            chunk_size = _select_hinv_jt_chunk_size(size, max_constraints, max_shared_memory, tile_threads)
            if chunk_size is None:
                if hinv_jt_kernel == "tiled":
                    required = _estimate_hinv_jt_shared_memory(size, 1, fused=False, tile_threads=tile_threads)
                    raise ValueError(
                        f"hinv_jt_kernel='tiled' requires at least {required} "
                        f"bytes of shared memory for DOF size {size}, but the device exposes "
                        f"{max_shared_memory} bytes; use hinv_jt_kernel='auto' or 'par_row'."
                    )
                required = _estimate_hinv_jt_shared_memory(size, 1, fused=False, tile_threads=tile_threads)
                reasons.append(
                    (
                        size,
                        "hinv_jt_kernel",
                        f"Parallel-row fallback: one tiled row needs {required} bytes of shared memory; "
                        f"device limit is {max_shared_memory}.",
                    )
                )
                continue

            tiled_sizes.add(size)
            chunk_sizes.append((size, chunk_size))
            reasons.append(
                (size, "hinv_jt_kernel", f"Tiled response uses {chunk_size}-row chunks within shared-memory limits.")
            )
            if fuse_delassus and (
                _estimate_hinv_jt_shared_memory(size, max_constraints, fused=True, tile_threads=tile_threads)
                <= max_shared_memory
            ):
                fused_sizes.add(size)

        return cls(
            planned_diagonal_sizes,
            frozenset(cholesky_tiled_sizes),
            frozenset(tiled_sizes),
            tuple(chunk_sizes),
            frozenset(fused_sizes),
            tuple(reasons),
        )

    def selection_reasons(self, size: int) -> dict[str, str]:
        """Return detached construction-time reasons for one articulation size."""
        return {name: reason for planned_size, name, reason in self.reasons if planned_size == size}

    def use_diagonal_mass(self, size: int) -> bool:
        """Return whether an articulation group has diagonal generalized inertia."""
        return size in self.diagonal_mass_sizes

    def use_tiled_cholesky(self, size: int) -> bool:
        """Return whether an articulation group uses tiled Cholesky."""
        return size in self.cholesky_tiled_sizes

    def hinv_jt_chunk_size(self, size: int) -> int | None:
        """Return the tiled H-inverse chunk size for a response group."""
        for planned_size, chunk_size in self.hinv_jt_chunk_sizes:
            if planned_size == size:
                return chunk_size
        return None

    def use_tiled_hinv_jt(self, size: int) -> bool:
        """Return whether a response group uses tiled H-inverse application."""
        return size in self.hinv_jt_tiled_sizes

    def use_fused_hinv_jt(self, size: int) -> bool:
        """Return whether a response group may fuse H-inverse and Delassus."""
        return size in self.hinv_jt_fused_sizes
