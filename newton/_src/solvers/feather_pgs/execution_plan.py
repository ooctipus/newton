# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Plan FPGS launch resources without allocating arrays or initializing a device.

These construction-time rules describe storage feasibility and existing routing.
They do not change the numerical method, row capacity, or iteration budget.
"""

from collections.abc import Callable
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


def _align_shared_memory(size: int) -> int:
    """Align a shared-memory allocation to Warp's 16-byte tile boundary."""
    return (size + 15) // 16 * 16


def retired_tiled_response_fits(n_dofs: int, max_shared_memory: int) -> bool:
    """Whether the retired tiled H^-1 J^T kernel's one-row footprint fits ``max_shared_memory`` [B]: the factor, three
    row tiles and 64 threads' tile storage. Only the local packets' admission reads it (:func:`.row_packets.decline`):
    on CUDA it declines a class with a size above 12 coordinates whose footprint fits, as that kernel's plan did."""
    footprint = _align_shared_memory(4 * n_dofs * n_dofs) + 3 * _align_shared_memory(4 * n_dofs) + 4 * 64
    return footprint <= max_shared_memory
