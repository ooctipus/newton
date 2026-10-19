# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compose native CUDA bodies from device preludes they declare.

A sweep's native body calls device helpers, such as the paired friction law,
that it does not define itself. The body declares those helpers as preludes,
and :func:`cuda_native` places them after the body's one architecture guard,
in the declared order. Only this module joins prelude text to a body.
"""

from dataclasses import dataclass

ARCHITECTURE_GUARD = "#if defined(__CUDA_ARCH__)\n"


@dataclass(frozen=True)
class DevicePrelude:
    """Device code that a native body declares ahead of its own statements."""

    name: str
    """Name of the prelude in tests and diagnostics."""
    source: str
    """Device code placed after the architecture guard, without a trailing newline."""


def cuda_native(body: str, *, preludes: tuple[DevicePrelude, ...] = (), lead: str = "\n") -> str:
    """Return a ``func_native`` snippet that opens with one architecture guard and the body's preludes.

    The snippet is ``lead``, the guard, each prelude followed by a newline in
    the declared order, then ``body``.

    Args:
        body: Native statements after the preludes, including the closing ``#endif``.
        preludes: Device preludes the body declares, in the order they are defined.
        lead: Text before the architecture guard.
    """
    return lead + ARCHITECTURE_GUARD + "".join(prelude.source + "\n" for prelude in preludes) + body
