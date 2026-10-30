# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compile-time storage and coordinate contracts for cached tree dynamics."""

from dataclasses import dataclass

from .device_preludes import dual_native, lane_group, substitute


@dataclass(frozen=True)
class TreeLayout:
    bodies: int
    dofs: int
    entries: int
    inverse_width: int
    max_depth: int
    root_dofs: int
    permutation: tuple[int, ...]
    # Threads per tree. Small trees share a 32-thread block (two 16-lane
    # trees, eight 4-lane trees, ...) with warp-level barriers; 64 gives a tree
    # two warps (more than 32 bodies, or heavy chains packed into two warps);
    # 0 selects one tree per block of the warps its bodies need.
    lanes: int = 0
    # A forest: local body index of every root (one tree: (0,)), each root's
    # coordinate count (0 fixed, 6 free, or a joint's), its coordinate offset
    # within the group, its factor family size (0 for a prescribed root) and
    # whether the whole group is one articulation.
    roots: tuple[int, ...] = (0,)
    root_coordinates: tuple[int, ...] = ()
    root_offsets: tuple[int, ...] = (0,)
    root_families: tuple[int, ...] = ()
    # Per root, whether it stands on a free joint. Any other root's coordinates
    # are a joint's, which the stages treat as a descendant's.
    free_roots: tuple[bool, ...] = ()
    # A joint with its own bias acceleration: a D6 whose axes form a chain
    # (linear and angular axes, or several angular ones). The motion scan
    # then reads each joint's own bias instead of taking it as zero.
    own_bias: bool = False

    def __post_init__(self):
        if not (1 <= self.bodies <= 64 and 1 <= self.dofs <= 64 and 0 <= self.max_depth < self.bodies):
            raise ValueError("Cached trees require at most 64 bodies and coordinates")
        if not self.root_coordinates:
            object.__setattr__(self, "root_coordinates", (self.root_dofs,))
        if not self.root_families:
            object.__setattr__(self, "root_families", (self.dofs,))
        if not self.free_roots:
            object.__setattr__(self, "free_roots", tuple(c == 6 for c in self.root_coordinates))
        if not (
            len(self.roots)
            == len(self.root_coordinates)
            == len(self.root_offsets)
            == len(self.root_families)
            == len(self.free_roots)
        ):
            raise ValueError("Cached forest roots need one coordinate count, offset, family and kind each")
        if self.roots[0] != 0 or any(
            c != 6 if free else not 0 <= c <= self.dofs
            for c, free in zip(self.root_coordinates, self.free_roots, strict=True)
        ):
            raise ValueError("Cached forest roots start at body 0 and are fixed, free or a joint's coordinates")
        if self.lanes == 0:
            object.__setattr__(self, "lanes", 64 if self.bodies > 32 else 32)
        if self.lanes not in (4, 8, 16, 32, 64) or self.lanes < self.bodies:
            raise ValueError("Cached tree lanes must be a power of two covering every body")
        if self.root_dofs != self.root_coordinates[0] or self.root_dofs > self.dofs:
            raise ValueError("Cached trees require their root's coordinate count")
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

    @property
    def forest(self):
        return len(self.roots) > 1

    @property
    def free_root_count(self):
        return sum(self.free_roots)

    @property
    def free_root(self):
        """Whether the first root stands on a free joint."""
        return self.free_roots[0]

    @property
    def pack(self):
        """Trees per 32-thread block."""
        return 32 // self.lanes if self.lanes < 32 else 1

    @property
    def block(self):
        """Threads per block."""
        return 64 if self.lanes == 64 else 32

    @property
    def reverse(self):
        """Whether factor coordinates run in reverse physical order."""
        return self.permutation == tuple(range(self.dofs - 1, -1, -1))

    def physical(self, index):
        """Native expression of the physical coordinate of factor coordinate ``index``.

        A permutation other than the reverse order reads the table that
        :meth:`physical_table` declares.
        """
        return f"{self.dofs - 1}-{index}" if self.reverse else f"physical[{index}]"

    def physical_table(self):
        """Native declaration of the factor-to-physical coordinate table :meth:`physical` reads."""
        if self.reverse:
            return ""
        return f"constexpr int physical[{self.dofs}]={{{','.join(map(str, self.permutation))}}};"

    def sizes(self):
        """Every size a native template names, by its ``$NAME``."""
        return {
            "LANES": self.lanes,
            # Window sums of 1, 2, 4, ... lanes a lane group can hold (interval reductions).
            "WINDOWS": self.lanes.bit_length() if self.lanes <= 32 else 6,
            "PACK": self.pack,
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
            "TOTAL": self.origin_offset + 6 * len(self.roots),
            "ROOT_COUNT": len(self.roots),
            "FREE_ROOTS": self.free_root_count,
            "WHITE": 6 * self.bodies + self.dofs,
            "PRED_SCRATCH": 6 * self.bodies + 2 * self.dofs,
            "FREE_ROOT": int(self.free_root),
        }

    def render(self, source, **fragments):
        """Fill a native template with the layout's sizes and the caller's code fragments.

        The layout supplies sizes only. Each fragment is code its owner
        composed, placed at the template's ``$NAME`` slot of the same name
        after its own sizes are filled; one substitution pass places them all,
        so no fragment is searched for, reordered or rewritten afterwards.
        """
        sizes = self.sizes()
        clashes = sizes.keys() & fragments.keys()
        if clashes:
            raise ValueError(f"Native fragments shadow layout sizes: {sorted(clashes)}")
        return substitute(source, sizes, **{name: substitute(text, sizes) for name, text in fragments.items()})

    def native(self, source, *, preludes=(), **fragments):
        """Render a native template (see :meth:`render`) after the tree's lane group of :attr:`lanes` threads and
        the ``preludes`` its fragments call.

        A tree of fewer than 32 lanes shares a warp with others (one per ``SUB``); a tree of 32 lanes owns its
        warp and a tree of 64 its block (:func:`~.device_preludes.lane_group`).
        """
        return dual_native(self.render(source, **fragments), preludes=(lane_group(self.lanes), *preludes))


# The 44-body, 43-coordinate free-root chain the kernels were first written for;
# the default layout argument and the unpacked helpers' reference.
REFERENCE_LAYOUT = TreeLayout(44, 43, 434, 18, 10, 6, tuple(range(42, -1, -1)))
