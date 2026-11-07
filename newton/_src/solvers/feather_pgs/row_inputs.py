# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compile-time row operands of the generated sweeps.

Readers describe storage, not a solve policy: parallel consumers may stage Z plus live metadata. Their recurrences
stay in the ordinary factories. Extra operands (:class:`RowArgument`) extend a sweep's declared native ABI
(``kernel_template.operands``); no solver source is inspected, copied or patched.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RowArgument:
    name: str
    dtype: object
    after: str | None = None


@dataclass(frozen=True)
class ParallelRowInputs:
    """Walked rows and initial metadata for the parallel policy.

    :attr:`read` walks the lane's row into ``Zi`` (``TIER_WALK``, :class:`.response_operator.WalkedRows`), its
    incident velocity into ``incident``, and completes the admitted row's metadata. Prefix metadata may already exist;
    deferred contacts must not read stale fields.
    """

    read: str
    arguments: tuple[RowArgument, ...]
    row_type: str
    rhs: str
    parent: str
    mu: str
    preludes: tuple = ()
    """Device preludes (``device_preludes.DevicePrelude``) that :attr:`read` calls into."""
