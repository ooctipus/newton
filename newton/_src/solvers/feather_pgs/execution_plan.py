# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Plan FPGS launch resources without allocating arrays or initializing a device.

These construction-time rules describe storage feasibility and existing routing.
They do not change the numerical method, row capacity, or iteration budget.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

from .staging import Staging

_MFGS_RESIDENT_METADATA_MAX_BYTES = 4096
"""Row metadata a matrix-free sweep keeps resident at most [B]: larger working sets cost more occupancy than their
repeated global loads (one 4-KiB set)."""
# Static shared memory a block may declare to keep its packed row metadata resident [B].
_MFGS_STATIC_SHARED_MAX_BYTES = 48 * 1024


class _MFGSStorage(NamedTuple):
    """Storage classes of one MF-GS kernel specialization and why they were chosen.

    ``shared_metadata`` keeps every per-row metadata array in shared memory;
    otherwise the arrays stream from global memory and ``resident_row_metadata``
    says whether the packed dense row types and parents stay in shared memory.
    ``stream_mf_impulses`` keeps the matrix-free impulses in global memory.
    """

    shared_metadata: bool
    stream_mf_impulses: bool
    resident_row_metadata: bool
    reason: str


def _select_mfgs_storage(
    staging: Callable[..., Staging],
    budget: int,
    *,
    block_count: int,
    sm_count: int,
    force_stream_mf_impulses: bool = False,
) -> _MFGSStorage:
    """Select metadata and impulse storage without changing row capacity or order.

    ``staging(shared_metadata=, resident_row_metadata=, stream_mf_impulses=)`` is the sweep block's own declaration
    (``sweep_routes.mf_gs_staging``) and ``budget`` the device's static shared memory per block
    (:func:`.staging.static_budget`): every choice below is a fit of the declared working set, not a mirrored count.

    Row metadata stays resident when it takes at most :data:`_MFGS_RESIDENT_METADATA_MAX_BYTES` and the block fits.
    MF impulses stream only when they cannot fit alongside the world velocity and the dense state (a capacity
    choice, not a claim that global storage is faster), or on ``force_stream_mf_impulses`` (the solver's
    test pin ``mf_storage='streamed'``). Streamed metadata keeps the packed dense row types and
    parents resident when the launch grid of ``block_count`` blocks holds at most one block per SM, so they cannot
    cost concurrency, and the block then declares at most :data:`_MFGS_STATIC_SHARED_MAX_BYTES`. Both storage
    classes hold identical values, so results are bitwise identical.
    """
    resident = staging(shared_metadata=True, resident_row_metadata=False, stream_mf_impulses=False)
    streamed = staging(shared_metadata=False, resident_row_metadata=False, stream_mf_impulses=False)
    metadata_bytes = resident.bytes - streamed.bytes
    metadata_budget = _MFGS_RESIDENT_METADATA_MAX_BYTES
    shared_metadata = metadata_bytes <= metadata_budget and resident.fits(budget)
    if shared_metadata:
        reason = f"Row metadata is resident: {metadata_bytes} bytes fit the {metadata_budget}-byte budget."
    elif metadata_bytes > metadata_budget:
        reason = f"Row metadata streams: {metadata_bytes} bytes exceed the {metadata_budget}-byte budget."
    else:
        reason = (
            f"Row metadata streams: resident, the block would declare {resident.bytes} bytes of static shared "
            f"memory; the device's budget is {budget}."
        )
    fixed = staging(shared_metadata=shared_metadata, resident_row_metadata=False, stream_mf_impulses=True)
    if not fixed.fits(budget):
        raise ValueError(
            f"MF-GS world velocity and dense state require {fixed.bytes} bytes of static shared memory even with "
            f"streamed MF impulses, but the device's budget is {budget} bytes"
        )
    with_impulses = staging(shared_metadata=shared_metadata, resident_row_metadata=False, stream_mf_impulses=False)
    stream_mf_impulses = not with_impulses.fits(budget)
    if stream_mf_impulses:
        reason += f" MF impulses stream: {with_impulses.bytes} bytes exceed the device's {budget}."
    elif force_stream_mf_impulses:
        stream_mf_impulses = True
        reason += " MF impulses stream on request (test pin mf_storage='streamed')."
    resident_row_metadata = False
    if not shared_metadata:
        candidate = staging(shared_metadata=False, resident_row_metadata=True, stream_mf_impulses=stream_mf_impulses)
        limit = min(_MFGS_STATIC_SHARED_MAX_BYTES, budget)
        if block_count > sm_count:
            reason += (
                f" Row types and parents stream too: the {block_count}-block grid exceeds one block per SM "
                f"({sm_count} SMs)."
            )
        elif not candidate.fits(limit):
            reason += (
                f" Row types and parents stream too: resident, the block would declare {candidate.bytes} bytes of "
                f"static shared memory, over the {limit}-byte budget."
            )
        else:
            resident_row_metadata = True
            reason += (
                f" Row types and parents stay resident: the {block_count}-block grid fits one block per SM "
                f"({sm_count} SMs) and the block declares {candidate.bytes} <= {limit} bytes of static shared memory."
            )
    return _MFGSStorage(shared_metadata, stream_mf_impulses, resident_row_metadata, reason)


_HINV_JT_MAX_CHUNK_SIZE = 32


def _align_shared_memory(size: int) -> int:
    """Align a shared-memory allocation to Warp's 16-byte tile boundary."""
    return (size + 15) // 16 * 16


def _estimate_hinv_jt_shared_memory(n_dofs: int, constraint_count: int, *, tile_threads: int) -> int:
    """Estimate the complete tiled H-inverse shared-memory footprint [B]: the factor and three row tiles."""
    footprint = _align_shared_memory(4 * n_dofs * n_dofs)
    footprint += 3 * _align_shared_memory(4 * n_dofs * constraint_count)
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
        if _estimate_hinv_jt_shared_memory(n_dofs, chunk_size, tile_threads=tile_threads) <= max_shared_memory:
            return chunk_size
    return None


@dataclass(frozen=True)
class _FeatherPGSExecutionPlan:
    """Immutable device-resource choices for one solver shape."""

    diagonal_mass_sizes: frozenset[int]
    hinv_jt_tiled_sizes: frozenset[int]
    hinv_jt_chunk_sizes: tuple[tuple[int, int], ...]
    reasons: tuple[tuple[int, str, str], ...] = ()

    @classmethod
    def build(
        cls,
        size_groups: list[int],
        *,
        max_constraints: int,
        max_shared_memory: int,
        hinv_jt_kernel: str,
        small_dof_threshold: int,
        tile_threads: int,
        diagonal_mass_sizes: frozenset[int] = frozenset(),
    ) -> "_FeatherPGSExecutionPlan":
        """Resolve H-inverse implementations from solver shape and device limits."""
        planned_diagonal_sizes = (
            frozenset(size for size in size_groups if size in diagonal_mass_sizes)
            if hinv_jt_kernel == "auto"
            else frozenset()
        )
        tiled_sizes: set[int] = set()
        chunk_sizes: list[tuple[int, int]] = []
        reasons: list[tuple[int, str, str]] = []
        for size in size_groups:
            if size in planned_diagonal_sizes:
                reasons.append(
                    (size, "hinv_jt_kernel", "Independent-coordinate topology uses diagonal generalized inertia.")
                )
                continue

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
                    required = _estimate_hinv_jt_shared_memory(size, 1, tile_threads=tile_threads)
                    raise ValueError(
                        f"hinv_jt_kernel='tiled' requires at least {required} "
                        f"bytes of shared memory for DOF size {size}, but the device exposes "
                        f"{max_shared_memory} bytes; use hinv_jt_kernel='auto' or 'par_row'."
                    )
                required = _estimate_hinv_jt_shared_memory(size, 1, tile_threads=tile_threads)
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

        return cls(
            planned_diagonal_sizes,
            frozenset(tiled_sizes),
            tuple(chunk_sizes),
            tuple(reasons),
        )

    def selection_reasons(self, size: int) -> dict[str, str]:
        """Return detached construction-time reasons for one articulation size."""
        return {name: reason for planned_size, name, reason in self.reasons if planned_size == size}

    def use_diagonal_mass(self, size: int) -> bool:
        """Return whether an articulation group has diagonal generalized inertia."""
        return size in self.diagonal_mass_sizes

    def hinv_jt_chunk_size(self, size: int) -> int | None:
        """Return the tiled H-inverse chunk size for a response group."""
        for planned_size, chunk_size in self.hinv_jt_chunk_sizes:
            if planned_size == size:
                return chunk_size
        return None

    def use_tiled_hinv_jt(self, size: int) -> bool:
        """Return whether a response group uses tiled H-inverse application."""
        return size in self.hinv_jt_tiled_sizes
