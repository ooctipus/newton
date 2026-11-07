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


class _MFGSStorage(NamedTuple):
    """The places of one MF-GS kernel specialization's row operands (:class:`.sweep_routes.RowPlaces`) and why they
    were chosen."""

    places: RowPlaces
    reason: str


def _select_mfgs_storage(
    staging: Callable[..., Staging],
    budget: int,
    *,
    force_stream_mf_impulses: bool = False,
    force_stream_line: bool = False,
) -> _MFGSStorage:
    """Place the sweep's row operands without changing row capacity or order.

    ``staging(places=)`` is the sweep block's own declaration (``sweep_routes.mf_gs_staging``) and ``budget`` the
    device's static shared memory per block (:func:`.memory.static_budget`): every choice below is a fit of the
    declared working set, not a mirrored count.

    The rows' metadata is shared when it takes at most :data:`_MFGS_RESIDENT_METADATA_MAX_BYTES` and the block fits;
    otherwise the rows' records stream, a row ahead (:class:`.sweep.WorldRows`). The free bodies' impulses are global
    only when they cannot fit alongside the world velocity and the dense state (a capacity choice, not a claim that
    global storage is faster), or on ``force_stream_mf_impulses`` (the solver's test pin ``mf_storage='streamed'``);
    the dense rows' impulses, the line's, only when they cannot fit beside the world velocity either (a line past the
    stage is read and written in place), or on ``force_stream_line`` (the test pin ``mf_storage='line'``). Both places
    hold identical values, so results are bitwise identical.
    """
    SHARED, GLOBAL = Place.SHARED, Place.GLOBAL
    resident = staging(places=RowPlaces())
    streamed = staging(places=RowPlaces(GLOBAL))
    metadata_bytes = resident.bytes - streamed.bytes
    metadata_budget = _MFGS_RESIDENT_METADATA_MAX_BYTES
    metadata = SHARED if metadata_bytes <= metadata_budget and resident.fits(budget) else GLOBAL
    if metadata is SHARED:
        reason = f"Row metadata is resident: {metadata_bytes} bytes fit the {metadata_budget}-byte budget."
    elif metadata_bytes > metadata_budget:
        reason = f"Row records stream: {metadata_bytes} bytes of metadata exceed the {metadata_budget}-byte budget."
    else:
        reason = (
            f"Row records stream: resident, the block would declare {resident.bytes} bytes of static shared "
            f"memory; the device's budget is {budget}."
        )
    fixed = staging(places=RowPlaces(metadata, GLOBAL))
    if force_stream_line:
        return _MFGSStorage(RowPlaces(metadata, GLOBAL, GLOBAL), reason + " Every impulse streams on request.")
    if not fixed.fits(budget):
        line = staging(places=RowPlaces(metadata, GLOBAL, GLOBAL))
        if not line.fits(budget):
            raise ValueError(
                f"MF-GS world velocity requires {line.bytes} bytes of static shared memory even with every impulse "
                f"in place, but the device's budget is {budget} bytes"
            )
        reason += f" Every impulse streams: the dense state takes {fixed.bytes} bytes of the device's {budget}."
        return _MFGSStorage(RowPlaces(metadata, GLOBAL, GLOBAL), reason)
    with_impulses = staging(places=RowPlaces(metadata))
    impulses = SHARED
    if not with_impulses.fits(budget):
        impulses = GLOBAL
        reason += f" MF impulses stream: {with_impulses.bytes} bytes exceed the device's {budget}."
    elif force_stream_mf_impulses:
        impulses = GLOBAL
        reason += " MF impulses stream on request (test pin mf_storage='streamed')."
    return _MFGSStorage(RowPlaces(metadata, impulses), reason)
