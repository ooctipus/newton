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
    # Threads per tree. Small trees share a 32-thread block (two 16-lane
    # trees, eight 4-lane trees, ...) with warp-level barriers; 0 selects the
    # unpacked default of one tree per block.
    lanes: int = 0
    # A forest: local body index of every root (one tree: (0,)), each root's
    # coordinate count (0 fixed, 6 free), its coordinate offset within the
    # group, its factor family size (0 for a prescribed root) and whether the
    # whole group is one articulation.
    roots: tuple[int, ...] = (0,)
    root_coordinates: tuple[int, ...] = ()
    root_offsets: tuple[int, ...] = (0,)
    root_families: tuple[int, ...] = ()

    def __post_init__(self):
        if not (1 <= self.bodies <= 64 and 1 <= self.dofs <= 64 and 0 <= self.max_depth < self.bodies):
            raise ValueError("Cached trees require at most 64 bodies and coordinates")
        if not self.root_coordinates:
            object.__setattr__(self, "root_coordinates", (self.root_dofs,))
        if not self.root_families:
            object.__setattr__(self, "root_families", (self.dofs,))
        if not (len(self.roots) == len(self.root_coordinates) == len(self.root_offsets) == len(self.root_families)):
            raise ValueError("Cached forest roots need one coordinate count, offset and family each")
        if self.roots[0] != 0 or any(c not in (0, 6) for c in self.root_coordinates):
            raise ValueError("Cached forest roots start at body 0 and are fixed or free")
        if self.lanes == 0:
            object.__setattr__(self, "lanes", 64 if self.bodies > 32 else 32)
        if self.lanes not in (4, 8, 16, 32, 64) or self.lanes < self.bodies or (self.lanes == 64) != (self.bodies > 32):
            raise ValueError("Cached tree lanes must be a power of two covering every body")
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

    @property
    def forest(self):
        return len(self.roots) > 1

    @property
    def free_roots(self):
        return sum(1 for c in self.root_coordinates if c == 6)

    @property
    def pack(self):
        """Trees per 32-thread block."""
        return 32 // self.lanes if self.lanes < 32 else 1

    @property
    def block(self):
        """Threads per block."""
        return 64 if self.lanes == 64 else 32

    def render(self, source):
        """Resolve named layout operands; never search for source statements."""
        reverse = self.permutation == tuple(range(self.dofs - 1, -1, -1))

        def physical(index):
            return f"{self.dofs - 1}-{index}" if reverse else f"physical[{index}]"

        if self.lanes < 32:
            base = 31 & ~(self.lanes - 1)
            mask = f"((0xffffffffu>>(32-{self.lanes}))<<(threadIdx.x&{base}))"
            threads = {
                "LANE": f"(threadIdx.x&{self.lanes - 1})",
                "STRIDE": str(self.lanes),
                "SUB": f"(threadIdx.x/{self.lanes})",
                "MASK": mask,
                "SYNC": f"__syncwarp({mask})",
                "ALL": f"__all_sync({mask},",
                "ANY": f"__any_sync({mask},",
            }
        else:
            threads = {
                "LANE": "threadIdx.x",
                "STRIDE": "blockDim.x",
                "SUB": "0",
                "MASK": "0xffffffffu",
                "SYNC": "__syncthreads()",
                "ALL": "__syncthreads_and(",
                "ANY": "__syncthreads_or(",
            }
        pred = 6 * self.bodies + 2 * self.dofs
        total = self.origin_offset + 6 * len(self.roots)
        if self.pack > 1:
            scratch = {
                "PRED_SCRATCH_DECL": f"__shared__ float s_all[{pred * self.pack}];__shared__ int nonzero_all[{self.pack}];"
                f"float* s=s_all+{threads['SUB']}*{pred};int& nonzero=nonzero_all[{threads['SUB']}];",
                "STATE_SCRATCH_DECL": f"__shared__ float values[{total * self.pack}];"
                f"return reinterpret_cast<uint64_t>(values+{threads['SUB']}*{total});",
            }
        else:
            # One tree per block keeps the original direct declarations (a
            # pointer into a per-block array costs a measurable few percent).
            scratch = {
                "PRED_SCRATCH_DECL": f"__shared__ float s[{pred}];__shared__ int nonzero;",
                "STATE_SCRATCH_DECL": f"__shared__ float values[{total}];return reinterpret_cast<uint64_t>(values);",
            }
        values = {
            **threads,
            **scratch,
            "LANES": self.lanes,
            # Warp-shuffle scans: the shuffle width of a packed group (a full warp needs none).
            "SHFL": f",{self.lanes}" if self.lanes < 32 else "",
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
            "FREE_ROOTS": self.free_roots,
            # Passive springs act on no root coordinate.
            "SPRING_GUARD": f"i>={self.root_dofs}" if not self.forest else "plan.dof_passive.data[i]!=0",
            "CURRENT_VALID": (
                "cache.current_valid.data[p.group_to_art.data[group]]=cache.status.data[group]==0;"
                if not self.forest
                else f"for(int r=0;r<{len(self.roots)};++r)"
                f"cache.current_valid.data[plan.arts.data[group*{len(self.roots)}+r]]=cache.status.data[group]==0;"
            ),
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


# The 44-body, 43-coordinate free-root chain the kernels were first written for;
# the default layout argument and the unpacked helpers' reference.
REFERENCE_LAYOUT = TreeLayout(44, 43, 434, 18, 10, 6, tuple(range(42, -1, -1)))
