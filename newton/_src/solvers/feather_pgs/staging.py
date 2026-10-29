# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Declared shared-memory working sets: a block's buffers declared once, its byte budget derived from them.

A native sweep keeps its hot working set (a world's velocity, its impulses,
its row metadata) in shared memory. :class:`Staging` declares that set once,
buffer by buffer, and everything a launch site needs follows from the
declaration instead of being mirrored by hand:

- :meth:`Staging.declarations` writes the declarations in ``SHARED`` storage
  (CUDA's shared memory, the stack of the CPU's one thread), one array per buffer, or, for several lane groups per block, one struct per group
  with an alias per buffer (``#define name scratch[group].name``), or (the
  ``arrays`` layout) one array per buffer of every group's copies with a
  pointer to the group's own, so the body reads every buffer by its own name;
- :attr:`Staging.bytes` is the block's static shared memory: the buffers of
  every group, laid out with their alignment, plus the tile storage Warp's
  tiled launches add (four bytes per thread, :data:`TILE_BYTES_PER_THREAD`);
- :meth:`Staging.fits` and :meth:`Staging.check` compare it with a static
  budget: by default :data:`STATIC_SHARED_BYTES`, what an offline build
  accepts, or a device's (:func:`static_budget`).

Static shared memory at run time is not the offline limit. Offline, ptxas 13.2
rejects a block that declares more than 48 KiB of static shared memory on every
architecture (0xc000). At run time Warp's path (NVRTC's PTX, compiled by the
driver) loads up to the opt-in maximum per block less the driver's 1 KiB
reservation (measured 2026-10-27: the RTX PRO 6000, sm_120, loads 100,352 B and
refuses 101,376 B; the GB300, sm_103, loads 231,424 B and refuses 232,448 B).
The budget of a kernel built for a device is therefore a property of the
device, read in one place, :func:`static_budget`: that opt-in maximum less the
reservation unless a probe records less (:data:`STATIC_SHARED_BUDGETS`). A
kernel sized to a budget above 48 KiB compiles only through that runtime path,
not offline.

A choice between stagings (a resident working set or one that streams part of
it from global memory) is a choice between two declarations: the caller builds
both and keeps the first that fits, so every byte count it compares is the one
the kernel declares.
"""

from __future__ import annotations

import dataclasses

import warp as wp

__all__ = [
    "RESERVED_SHARED_BYTES",
    "SM_SHARED_BYTES",
    "STATIC_SHARED_BUDGETS",
    "STATIC_SHARED_BYTES",
    "TILE_BYTES_PER_THREAD",
    "Buffer",
    "Staging",
    "arch_budget",
    "resident_blocks",
    "static_budget",
]

STATIC_SHARED_BYTES = 48 * 1024
"""Static shared memory a block may declare in an offline build [B]: ptxas 13.2 rejects more on every architecture
(0xc000)."""

CPU_STAGING_BYTES = 4 * 1024 * 1024
"""The working set a CPU block may declare [B]: the CPU's one thread runs the block and keeps it on its stack (an
automatic array), so no shared-memory budget applies; this bound keeps a block well inside a thread's stack."""

RESERVED_SHARED_BYTES = 1024
"""Shared memory the driver reserves in every block [B] (sm_80 and later), out of the opt-in maximum per block."""

STATIC_SHARED_BUDGETS: dict[int, int] = {}
"""Static shared memory budgets [B] by architecture (``Device.arch``) that a probe measured below the opt-in
maximum less the driver's reservation; :func:`static_budget` reads them first. None is recorded: the RTX PRO 6000
(sm_120, opt-in 101,376 B) loads a kernel declaring 100,352 B and refuses one declaring 101,376 B."""

TILE_BYTES_PER_THREAD = 4
"""Shared memory Warp's tiled launches declare per block thread [B] (``tile_shared_storage_t``'s offsets)."""

SM_SHARED_BYTES = 100 * 1024
"""Shared memory one SM gives its resident blocks [B] when no device of the architecture is at hand (an offline
build, the CPU): the RTX PRO 6000's (sm_120), the smallest of the suite's GPUs."""

SHARED_ALLOCATION_BYTES = 128
"""Granule in which an SM allocates a block's shared memory [B] (``cuda_occupancy.h``, sm_80 and later)."""

_SIZES = {
    "float": 4,
    "int": 4,
    "unsigned": 4,
    "unsigned int": 4,
    "short": 2,
    "unsigned short": 2,
    "char": 1,
    "signed char": 1,
    "unsigned char": 1,
    "double": 8,
    "long long": 8,
    "unsigned long long": 8,
    "int2": 8,
    "float2": 8,
    "int4": 16,
    "float4": 16,
}
_ALIGNMENTS = {"int2": 8, "float2": 8, "int4": 16, "float4": 16}


@dataclasses.dataclass(frozen=True)
class Buffer:
    """One shared array of a lane group's working set.

    Args:
        name: The array's name in the body.
        ctype: Its CUDA element type (one of the scalar or vector types of :data:`_SIZES`).
        count: Its element count (a compile-time constant).
        align: Byte alignment of the array (default: its element type's).
    """

    name: str
    ctype: str
    count: int
    align: int | None = None

    def __post_init__(self):
        if self.ctype not in _SIZES:
            raise ValueError(f"buffer {self.name}: unknown element type {self.ctype!r}")
        if int(self.count) < 1:
            raise ValueError(f"buffer {self.name}: needs at least one element, got {self.count}")
        object.__setattr__(self, "count", int(self.count))

    @property
    def alignment(self) -> int:
        """Byte alignment of the array."""
        return int(self.align or _ALIGNMENTS.get(self.ctype, _SIZES[self.ctype]))

    @property
    def bytes(self) -> int:
        """Bytes of the array's elements."""
        return self.count * _SIZES[self.ctype]

    def declaration(self) -> str:
        """The array's C declaration, without storage class."""
        align = f"__align__({self.align}) " if self.align else ""
        return f"{align}{self.ctype} {self.name}[{self.count}];"


def static_budget(arch: int) -> int:
    """Static shared memory a block of a kernel built for the CUDA architecture ``arch`` may declare [B].

    The one place a device's static budget is read: a probe's measured budget of the architecture
    (:data:`STATIC_SHARED_BUDGETS`), else the opt-in maximum per block that Warp reports for a CUDA device of the
    architecture (``Device.max_shared_memory_per_block``) less the driver's reservation
    (:data:`RESERVED_SHARED_BYTES`), else, with no such device (an offline build, the CPU),
    :data:`STATIC_SHARED_BYTES`.
    """
    if arch in STATIC_SHARED_BUDGETS:
        return int(STATIC_SHARED_BUDGETS[arch])
    for device in wp.get_cuda_devices():
        if device.arch == arch:
            return int(device.max_shared_memory_per_block) - RESERVED_SHARED_BYTES
    return STATIC_SHARED_BYTES


def factor_budget(device) -> int:
    """The static working set a block of a factor build may declare on ``device`` [B]: the offline limit, or less on a
    CUDA device that loads less; :data:`CPU_STAGING_BYTES` on the CPU."""
    if not device.is_cuda:
        return CPU_STAGING_BYTES
    return min(STATIC_SHARED_BYTES, int(device.max_shared_memory_per_block))


def arch_budget(device_arch) -> int:
    """The working set a block of a kernel factory's ``device_arch`` may declare [B]: :func:`static_budget` of a CUDA
    architecture number, :data:`CPU_STAGING_BYTES` for the CPU (architecture 0) and :data:`STATIC_SHARED_BYTES` for an
    offline build that names none."""
    try:
        arch = int(str(device_arch).removeprefix("sm_"))
    except (TypeError, ValueError):
        return STATIC_SHARED_BYTES
    return static_budget(arch) if arch > 0 else CPU_STAGING_BYTES


def resident_blocks(block_bytes: int, device_arch=None) -> int:
    """Blocks of ``block_bytes`` static shared memory each that one SM of ``device_arch`` keeps resident.

    A block takes its bytes and the driver's reservation, in 128-byte granules, out of the SM's shared memory: the
    opt-in maximum per block plus that reservation (every architecture since sm_80), read from a CUDA device of the
    architecture as :func:`static_budget` reads it, else :data:`SM_SHARED_BYTES`.
    """
    sm_bytes = SM_SHARED_BYTES
    try:
        arch = int(str(device_arch).removeprefix("sm_"))
    except (TypeError, ValueError):
        arch = 0
    for device in wp.get_cuda_devices() if arch > 0 else ():
        if device.arch == arch:
            sm_bytes = int(device.max_shared_memory_per_block) + RESERVED_SHARED_BYTES
            break
    granules = -(-(int(block_bytes) + RESERVED_SHARED_BYTES) // SHARED_ALLOCATION_BYTES)
    return sm_bytes // (granules * SHARED_ALLOCATION_BYTES)


def _layout(buffers) -> int:
    """Bytes of ``buffers`` laid out in order, each at its alignment (the C rules of a struct's members)."""
    offset, widest = 0, 1
    for buffer in buffers:
        alignment = buffer.alignment
        widest = max(widest, alignment)
        offset = (offset + alignment - 1) // alignment * alignment + buffer.bytes
    return (offset + widest - 1) // widest * widest


@dataclasses.dataclass(frozen=True)
class Staging:
    """A block's shared-memory working set: the buffers of each lane group, declared once.

    Args:
        buffers: Each lane group's buffers, in declaration order.
        groups: Lane groups per block; each holds its own copy of the buffers.
        threads: Threads per block, for the tile storage of a tiled launch (0: the launch is not tiled).
        name: Name of the per-group struct when ``groups > 1``.
        layout: How several groups' copies lie: ``"struct"``, one struct of the buffers per group, or
            ``"arrays"``, one array per buffer that holds every group's copy in group order (a buffer's
            elements of the groups that share a warp then fall in distinct banks).
    """

    buffers: tuple[Buffer, ...] = ()
    groups: int = 1
    threads: int = 0
    name: str = "Scratch"
    layout: str = "struct"

    def __post_init__(self):
        names = [buffer.name for buffer in self.buffers]
        if len(set(names)) != len(names):
            raise ValueError(f"staging buffers need distinct names: {names}")
        if self.groups < 1:
            raise ValueError("a staging needs at least one lane group")
        if self.layout not in ("struct", "arrays"):
            raise ValueError(f"staging layout must be 'struct' or 'arrays', got {self.layout!r}")
        object.__setattr__(self, "buffers", tuple(self.buffers))

    def __iter__(self):
        return iter(self.buffers)

    def __contains__(self, name: str) -> bool:
        return any(buffer.name == name for buffer in self.buffers)

    def plus(self, *buffers: Buffer) -> Staging:
        """This staging with ``buffers`` appended."""
        return dataclasses.replace(self, buffers=(*self.buffers, *buffers))

    @property
    def group_bytes(self) -> int:
        """Shared memory of one lane group's buffers [B]: separate arrays or a struct's members, each at its own
        alignment."""
        return _layout(self.buffers)

    @property
    def tile_bytes(self) -> int:
        """Shared memory Warp's tiled launch declares for this block [B]."""
        return TILE_BYTES_PER_THREAD * int(self.threads)

    @property
    def bytes(self) -> int:
        """Static shared memory the block declares [B]: every group's buffers and the tile storage."""
        if self.layout == "arrays" and self.groups > 1:
            arrays = (dataclasses.replace(buffer, count=buffer.count * self.groups) for buffer in self.buffers)
            return _layout(arrays) + self.tile_bytes
        return self.groups * self.group_bytes + self.tile_bytes

    def fits(self, limit: int = STATIC_SHARED_BYTES) -> bool:
        """Whether the block's static shared memory fits the budget ``limit`` [B] (by default an offline build's;
        a device's is :func:`static_budget`)."""
        return self.bytes <= int(limit)

    def check(self, what: str, limit: int = STATIC_SHARED_BYTES) -> None:
        """Raise ``ValueError`` naming ``what`` when the block does not fit the budget ``limit``."""
        if not self.fits(limit):
            raise ValueError(
                f"{what} declares {self.bytes} bytes of static shared memory per block "
                f"({self.groups} x {self.group_bytes} + {self.tile_bytes} tile storage), over the "
                f"{int(limit)}-byte limit"
            )

    def declarations(self, indent: str = "    ", *, group: str = "SUB") -> str:
        """The declarations of the working set in ``SHARED`` storage: CUDA's shared memory, or the stack of the CPU's
        one thread. A body that declares them declares the prelude :data:`.device_preludes.SHARED_STORAGE`.

        One array per buffer; with several lane groups per block, one struct of
        the buffers per group, indexed by ``group``, and an alias per buffer
        (:meth:`undefine` removes the aliases at the end of the body), or in the
        ``arrays`` layout one array of every group's copies per buffer
        (``name_storage``) and a pointer ``name`` to the group's copy.
        """
        if not self.buffers:
            return ""
        if self.groups == 1:
            return "\n".join(f"{indent}SHARED {buffer.declaration()}" for buffer in self.buffers)
        if self.layout == "arrays":
            arrays = (
                dataclasses.replace(buffer, name=f"{buffer.name}_storage", count=buffer.count * self.groups)
                for buffer in self.buffers
            )
            views = (
                f"{indent}{buffer.ctype}* {buffer.name} = {buffer.name}_storage + {group} * {buffer.count};"
                for buffer in self.buffers
            )
            return "\n".join((*(f"{indent}SHARED {array.declaration()}" for array in arrays), *views))
        members = "\n".join(f"{indent}    {buffer.declaration()}" for buffer in self.buffers)
        aliases = "\n".join(
            f"#define {buffer.name} {self.name.lower()}[{group}].{buffer.name}" for buffer in self.buffers
        )
        return (
            f"{indent}struct {self.name} {{\n{members}\n{indent}}};\n"
            f"{indent}SHARED {self.name} {self.name.lower()}[{self.groups}];\n{aliases}"
        )

    def undefine(self) -> str:
        """Remove the aliases :meth:`declarations` defines for several lane groups in a struct."""
        if self.groups == 1 or self.layout == "arrays":
            return ""
        return "\n".join(f"#undef {buffer.name}" for buffer in self.buffers)
