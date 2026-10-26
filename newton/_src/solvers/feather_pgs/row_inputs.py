# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compile-time row operands of the generated sweeps.

Readers describe storage, not a solve policy. Local ordered consumers read J;
parallel consumers may stage Z plus live metadata. Their recurrences stay in
the ordinary factories. Extra operands (:class:`RowArgument`) extend a sweep's
declared native ABI (``kernel_template.operands``); no solver source is
inspected, copied or patched.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RowArgument:
    name: str
    dtype: object
    after: str | None = None


@dataclass(frozen=True)
class CompactLocalRows:
    """Dense contact packets plus a bounded sparse prefix in physical coordinates.

    A prefix row may have repeated coordinate indices; coefficients add. The
    prefix belongs to the primary articulation. Contact packets concatenate
    primary and secondary coordinates, even for a primary-only consumer.
    """

    primary_dofs: int
    secondary_dofs: int
    packet_capacity: int
    prefix_capacity: int
    prefix_width: int
    arguments: tuple[RowArgument, ...]

    def __post_init__(self):
        if min(self.primary_dofs, self.packet_capacity, self.prefix_capacity, self.prefix_width) <= 0:
            raise ValueError("compact row dimensions and capacities must be positive")
        if self.secondary_dofs < 0 or self.prefix_capacity > self.packet_capacity:
            raise ValueError("compact row prefix must fit the packet capacity")

    def read(self, primary_dofs, secondary_dofs):
        if primary_dofs != self.primary_dofs or secondary_dofs not in (0, self.secondary_dofs):
            raise ValueError("local consumer dimensions do not match its row operands")
        width = self.primary_dofs + self.secondary_dofs
        packet = f"packets.row_contact.data[world * {self.packet_capacity} + row] >= 0"
        address = f"(world * {self.packet_capacity} + row) * {width}"
        prefix = f"(world * {self.prefix_capacity} + row) * {self.prefix_width}"
        terms = []
        for entry in range(self.prefix_width):
            offset = "" if entry == 0 else f" + {entry}"
            terms.append(f"(prefix.dofs.data[{prefix}{offset}] == i ? prefix.weights.data[{prefix}{offset}] : 0.0f)")
        primary = f"({packet} ? packets.jacobian.data[{address} + i] : ({' + '.join(terms)}))"
        secondary = f"({packet} ? packets.jacobian.data[{address} + {self.primary_dofs} + i] : 0.0f)"
        return primary, secondary


@dataclass(frozen=True)
class ParallelRowInputs:
    """Factor-coordinate staging and initial metadata for the parallel policy.

    The stage must initialize Z, incident velocity in factor coordinates,
    factor scratch for decoding, and complete admitted row metadata. Prefix
    metadata may already exist; deferred contacts must not read stale fields.
    """

    stage: str | None
    arguments: tuple[RowArgument, ...]
    row_type: str
    rhs: str
    parent: str
    mu: str
    physical_row: str | None = None
    preludes: tuple = ()
    """Device preludes (``device_preludes.DevicePrelude``) that :attr:`physical_row` calls into."""
