# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compile-time storage and coordinate contracts for cached tree dynamics."""

from dataclasses import dataclass
from string import Template


@dataclass(frozen=True)
class TreeLayout:
    bodies: int
    dofs: int
    entries: int
    inverse_width: int
    max_depth: int
    root_dofs: int
    permutation: tuple[int, ...]

    def __post_init__(self):
        if not (1 <= self.bodies <= 64 and 1 <= self.dofs <= 64 and 0 <= self.max_depth < self.bodies):
            raise ValueError("Cached trees require at most 64 bodies and coordinates")
        if self.root_dofs not in (0, 6) or self.root_dofs > self.dofs:
            raise ValueError("Cached trees require a fixed or free root")
        if sorted(self.permutation) != list(range(self.dofs)):
            raise ValueError("Cached trees require a complete factor-coordinate permutation")
        if not (self.dofs <= self.entries <= self.dofs * (self.dofs + 1) // 2):
            raise ValueError("Invalid packed lower operator size")
        if not 1 <= self.inverse_width <= self.dofs:
            raise ValueError("Invalid inverse-column support width")

    @property
    def axis_offset(self):
        # Collection reuses the dead pose/motion scratch for I*S. A root-only
        # free body has more coordinates than bodies and needs extra room.
        return max(19 * self.bodies, 6 * self.dofs)

    @property
    def wrench_offset(self):
        return self.axis_offset + 6 * self.dofs

    @property
    def moment_offset(self):
        return self.wrench_offset + 6 * self.bodies

    @property
    def origin_offset(self):
        return self.moment_offset + 13 * self.bodies

    @property
    def rounds(self):
        return max(1, self.max_depth.bit_length())

    def render(self, source):
        """Resolve named layout operands; never search for source statements."""
        reverse = self.permutation == tuple(range(self.dofs - 1, -1, -1))

        def physical(index):
            return f"{self.dofs - 1}-{index}" if reverse else f"physical[{index}]"

        values = {
            "BODIES": self.bodies,
            "DOFS": self.dofs,
            "NNZ": self.entries,
            "INVERSE_WIDTH": self.inverse_width,
            "ROOT_DOFS": self.root_dofs,
            "ROUNDS": self.rounds,
            "LAST_PARENT": self.max_depth - 1,
            "MOTION": 7 * self.bodies,
            "SIX_BODIES": 6 * self.bodies,
            "SIX_BODIES_PLUS3": 6 * self.bodies + 3,
            "AXIS": self.axis_offset,
            "WRENCH": self.wrench_offset,
            "MOMENT": self.moment_offset,
            "ORIGIN": self.origin_offset,
            "TOTAL": self.origin_offset + 6,
            "WHITE": 6 * self.bodies + self.dofs,
            "PRED_SCRATCH": 6 * self.bodies + 2 * self.dofs,
            "PHYSICAL_DECL": ""
            if reverse
            else f"constexpr int physical[{self.dofs}]={{{','.join(map(str, self.permutation))}}};",
            "PHYSICAL_ROW": physical("p.row.data[e]"),
            "PHYSICAL_COL": physical("p.col.data[e]"),
            "PHYSICAL_INDEX": physical("col"),
            "FREE_ROOT": int(self.root_dofs == 6),
        }
        return Template(source).substitute(values)


G1_LAYOUT = TreeLayout(44, 43, 434, 18, 10, 6, tuple(range(42, -1, -1)))
