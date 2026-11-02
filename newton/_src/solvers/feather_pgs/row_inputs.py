# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compile-time row operands of the generated sweeps.

Readers describe storage, not a solve policy. Local ordered consumers walk
their rows' seeds in the kernel or read the one writer's Z; parallel consumers
may stage Z plus live metadata. Their recurrences stay in
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


SEED_NONE = -1
"""A packet row's side without a seed: no body, or a body without response coordinates."""
SEED_SECONDARY = -2
"""A packet row's side on the world's paired free body (a one-body tree of the walk)."""


@dataclass(frozen=True)
class CompactLocalRows:
    """Seeds of a bounded sparse prefix and of contact packets, which the local kernel walks (FL3.6).

    A prefix row (a limit, a mimic) is covectors at the primary's coordinates, ``prefix_width`` (coordinate,
    weight) pairs; a pair repeats or a weight is zero where the row has fewer, and coefficients add. A contact packet
    row is a wrench at each of its two sides' walk frames at the step's poses (six floats a side) with the side's
    code: the body's index in the primary's tree, :data:`SEED_SECONDARY` for the paired free body or
    :data:`SEED_NONE`. :meth:`stage` walks a row's seeds into the response block's entries, as
    :func:`.walk_rows.write_walked_rows` walks a stored row's, then releases the tree's slipping drives along their
    paths (``R``). ``bodies`` is the most bodies of a primary tree, which the kernel stages the walk of.
    """

    primary_dofs: int
    secondary_dofs: int
    packet_capacity: int
    prefix_capacity: int
    prefix_width: int
    arguments: tuple[RowArgument, ...]
    bodies: int = 1

    def __post_init__(self):
        if min(self.primary_dofs, self.packet_capacity, self.prefix_capacity, self.prefix_width, self.bodies) <= 0:
            raise ValueError("compact row dimensions and capacities must be positive")
        if self.secondary_dofs < 0 or self.prefix_capacity > self.packet_capacity:
            raise ValueError("compact row prefix must fit the packet capacity")

    def stage(self, op) -> str:
        """Native lines walking row ``row``'s seeds into ``s_Z[staged + c]`` (cleared first) and declaring
        ``incident``, its ``J v_hat`` summed along the walks (the response block's walks,
        :meth:`.response_operator.LocalBlock.prologue`): the prefix's covectors and a contact's wrench on the primary
        tree, a contact's wrench on the paired free body's; then the tree's release ``R``."""
        P, S = op.primary, op.secondary
        if P != self.primary_dofs or S not in (0, self.secondary_dofs):
            raise ValueError("local consumer dimensions do not match its row operands")
        W = self.prefix_width
        contact = ""
        if op.wrenches:
            free = ""
            if S:
                free = f"""
        }} else if (code == {SEED_SECONDARY}) {{
            walk_free();
            incident += walk_wrench(&s_Z[staged + {P}], s_v + {P}, free_data, free_link, free_now, walk_now_link, 0,
                                    wrench);"""
            contact = f""" else {{
    // A contact row: a wrench seed on each side.
    for (int side = 0; side < 2; ++side) {{
        const int code = packets.seed_body.data[packet * 2 + side];
        const float* wrench = &packets.seed.data[(size_t)(packet * 2 + side) * 6];
        if (code >= 0) {{
            walk_primary();
            incident += walk_wrench(&s_Z[staged], s_v, walk_data, walk_link, walk_now, walk_now_link, code, wrench);{free}
        }}
    }}
}}"""
        release = "\nWALK_RELEASE(walk.release, art, &s_Z[staged]);" if op.release else ""
        return f"""for (int c = 0; c < {P + S}; ++c) s_Z[staged + c] = 0.0f;
float incident = 0.0f;
const int packet = world * {self.packet_capacity} + row;
if (packets.row_contact.data[packet] < 0) {{
    // A prefix row: covector seeds at the primary's coordinates.
    walk_primary();
    for (int k = 0; k < {W}; ++k) {{
        const int dof = prefix.dofs.data[(world * {self.prefix_capacity} + row) * {W} + k];
        const float weight = prefix.weights.data[(world * {self.prefix_capacity} + row) * {W} + k];
        if (dof < 0 || weight == 0.0f) continue;
        incident += walk_covector(&s_Z[staged], dof, weight);
    }}
}}{contact}{release}"""

    def diagonal(self, row_diagonal: str) -> str:
        """The row's effective diagonal: its walked ``|Z|^2`` and the producer's CFM seed in ``world_diag``."""
        return f"{row_diagonal} + world_diag.data[world_row_base + row]"


@dataclass(frozen=True)
class StoredLocalRows:
    """The one writer's rows of the local owners' worlds (FL3.6): the class walks through the writer
    (:mod:`.walk_rows`, the packed sink, its free bodies' sides and rows included, with the release ``R``), so the
    local kernel reads each row's ``Z``, incident velocity and diagonal where the general sweep reads them, and leaves
    in the walk's frame."""

    arguments: tuple[RowArgument, ...] = ()

    def stage(self, op) -> str:
        """Native lines copying row ``row``'s stored ``Z`` (each side's entries, :data:`.packed_rows.PACKED_SIDES`) into
        ``s_Z[staged + c]`` and declaring its stored ``incident``."""
        P, S = op.primary, op.secondary
        lines = [
            "const wp::vec2i sides_r = row_sides.data[world_row_base + row];",
            "const float* Z_row = &packed_Z.data[(size_t)(world_row_base + row) * packed_Z.shape[2]];",
            "const int first_p = PACKED_FIRST(sides_r, art);",
            f"for (int c = 0; c < {P}; ++c) s_Z[staged + c] = first_p >= 0 ? Z_row[first_p + c] : 0.0f;",
        ]
        if S:
            lines += [
                "const int first_s = PACKED_FIRST(sides_r, secondary_art);",
                f"for (int c = 0; c < {S}; ++c) s_Z[staged + {P} + c] = first_s >= 0 ? Z_row[first_s + c] : 0.0f;",
            ]
        lines.append("const float incident = row_incident.data[world_row_base + row];")
        return "\n".join(lines)

    def diagonal(self, row_diagonal: str) -> str:
        """The row's effective diagonal: the writer's ``|Z|^2 + cfm``."""
        return "world_diag.data[world_row_base + row]"


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
