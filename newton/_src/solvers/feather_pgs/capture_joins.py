# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Ask which graph capture records a stream.

:func:`capture_id` names the capture a host decision was recorded in.
"""

import warp as wp


def capture_id(device) -> int:
    """Warp's id of the graph capture recording ``device``'s current stream, or 0 outside one.

    Warp registers each capture it begins, external ones announced with
    ``wp.capture_begin(external=True)`` included, under its stream in
    ``device.captures``; the graph keeps the driver's capture id, which a
    resumed capture renews, and the driver never reuses an id within a process.
    Both are Warp internals: without them, or for a capture Warp does not know,
    the result is 0, which no recorded capture matches.

    Args:
        device: The Warp device whose current stream may be capturing.
    """
    if not device.is_cuda:
        return 0
    graph = getattr(device, "captures", {}).get(wp.get_stream(device))
    return int(getattr(graph, "capture_id", 0) or 0)
