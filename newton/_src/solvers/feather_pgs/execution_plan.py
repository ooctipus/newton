# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Plan FPGS launch resources without allocating arrays or initializing a device.

These construction-time rules describe storage feasibility and existing routing.
They do not change the numerical method, row capacity, or iteration budget.
"""

from collections.abc import Callable
from typing import NamedTuple

from ...core.memory import Place, Staging
from .sweep_routes import RowPlaces

_MFGS_RESIDENT_METADATA_MAX_BYTES = 4096
"""Row metadata a matrix-free sweep keeps resident at most [B]: larger working sets cost more occupancy than their
repeated global loads (one 4-KiB set)."""
# Static shared memory a block may declare to keep its packed row metadata resident [B].
_MFGS_STATIC_SHARED_MAX_BYTES = 48 * 1024


class _MFGSStorage(NamedTuple):
    """The places of one MF-GS kernel specialization's row operands (:class:`.sweep_routes.RowPlaces`) and why they
    were chosen."""

    places: RowPlaces
    reason: str


def _select_mfgs_storage(
    staging: Callable[..., Staging],
    budget: int,
    *,
    block_count: int,
    sm_count: int,
    force_stream_mf_impulses: bool = False,
) -> _MFGSStorage:
    """Place the sweep's row operands without changing row capacity or order.

    ``staging(places=)`` is the sweep block's own declaration (``sweep_routes.mf_gs_staging``) and ``budget`` the
    device's static shared memory per block (:func:`.memory.static_budget`): every choice below is a fit of the
    declared working set, not a mirrored count.

    The rows' metadata is shared when it takes at most :data:`_MFGS_RESIDENT_METADATA_MAX_BYTES` and the block fits.
    The free bodies' impulses are global only when they cannot fit alongside the world velocity and the dense state (a
    capacity choice, not a claim that global storage is faster), or on ``force_stream_mf_impulses`` (the solver's
    test pin ``mf_storage='streamed'``). Global metadata keeps the rows' kinds and parents shared when the launch grid
    of ``block_count`` blocks holds at most one block per SM, so they cannot cost concurrency, and the block then
    declares at most :data:`_MFGS_STATIC_SHARED_MAX_BYTES`. Both places hold identical values, so results are bitwise
    identical.
    """
    SHARED, GLOBAL = Place.SHARED, Place.GLOBAL
    resident = staging(places=RowPlaces())
    streamed = staging(places=RowPlaces(GLOBAL, GLOBAL))
    metadata_bytes = resident.bytes - streamed.bytes
    metadata_budget = _MFGS_RESIDENT_METADATA_MAX_BYTES
    metadata = SHARED if metadata_bytes <= metadata_budget and resident.fits(budget) else GLOBAL
    if metadata is SHARED:
        reason = f"Row metadata is resident: {metadata_bytes} bytes fit the {metadata_budget}-byte budget."
    elif metadata_bytes > metadata_budget:
        reason = f"Row metadata streams: {metadata_bytes} bytes exceed the {metadata_budget}-byte budget."
    else:
        reason = (
            f"Row metadata streams: resident, the block would declare {resident.bytes} bytes of static shared "
            f"memory; the device's budget is {budget}."
        )
    fixed = staging(places=RowPlaces(metadata, metadata, GLOBAL))
    if not fixed.fits(budget):
        raise ValueError(
            f"MF-GS world velocity and dense state require {fixed.bytes} bytes of static shared memory even with "
            f"streamed MF impulses, but the device's budget is {budget} bytes"
        )
    with_impulses = staging(places=RowPlaces(metadata, metadata))
    impulses = SHARED
    if not with_impulses.fits(budget):
        impulses = GLOBAL
        reason += f" MF impulses stream: {with_impulses.bytes} bytes exceed the device's {budget}."
    elif force_stream_mf_impulses:
        impulses = GLOBAL
        reason += " MF impulses stream on request (test pin mf_storage='streamed')."
    kinds = metadata
    if metadata is GLOBAL:
        candidate = staging(places=RowPlaces(GLOBAL, SHARED, impulses))
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
            kinds = SHARED
            reason += (
                f" Row types and parents stay resident: the {block_count}-block grid fits one block per SM "
                f"({sm_count} SMs) and the block declares {candidate.bytes} <= {limit} bytes of static shared memory."
            )
    return _MFGSStorage(RowPlaces(metadata, kinds, impulses), reason)


def _align_shared_memory(size: int) -> int:
    """Align a shared-memory allocation to Warp's 16-byte tile boundary."""
    return (size + 15) // 16 * 16


def retired_tiled_response_fits(n_dofs: int, max_shared_memory: int) -> bool:
    """Whether the retired tiled H^-1 J^T kernel's one-row footprint fits ``max_shared_memory`` [B]: the factor, three
    row tiles and 64 threads' tile storage. Only the local packets' admission reads it (:func:`.row_packets.decline`):
    on CUDA it declines a class with a size above 12 coordinates whose footprint fits, as that kernel's plan did."""
    footprint = _align_shared_memory(4 * n_dofs * n_dofs) + 3 * _align_shared_memory(4 * n_dofs) + 4 * 64
    return footprint <= max_shared_memory
