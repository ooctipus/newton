# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One mechanism for FeatherPGS's generated kernels: source slots, preludes, a signature declared once, launches.

Every generated kernel is a :class:`KernelTemplate`:

- its source fills ``$NAME`` slots with :func:`substitute`, which refuses a slot without a value and a value
  without a slot, so no text is edited after it is generated;
- a native kernel's body takes its device preludes through :mod:`.device_preludes` (``cuda_native``, or
  ``dual_native`` for a body that also compiles for the CPU);
- its parameters are declared once, as ``(name, type)`` pairs, :class:`~.field_group.Field` declarations or a
  generated struct's fields (:func:`params`). The native function's ABI and the Warp kernel's signature are both
  derived from them, with the :class:`Dispatch` that maps the launch's threads to native calls;
- :meth:`KernelTemplate.kernel` builds the Warp kernel. What a launch site needs to know about a kernel travels
  with it in a typed descriptor (:class:`SweepTier` for the matrix-free tiers: block size, the operands it takes
  besides its signature's, its velocity-limit law), and a solve readout is registered with the kernel by
  :meth:`.solve_policy.SolveReadoutRecord.register`, in place of attributes set on the kernel.

A Python kernel (no native body) is a function renamed per variant: :func:`python_kernel`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import linecache
from typing import Any

import warp as wp

from . import field_group
from .device_preludes import DevicePrelude, cuda_native, dual_native, substitute

__all__ = [
    "BLOCKS",
    "THREADS",
    "WORLDS",
    "WORLD_QUEUE",
    "Dispatch",
    "KernelTemplate",
    "SweepTier",
    "candidate_queue",
    "candidates",
    "operands",
    "params",
    "python_kernel",
    "substitute",
    "tiles",
]


def operands(base, extra=()) -> tuple[tuple[str, Any], ...]:
    """Parameters ``base`` with each operand of ``extra`` (``name``, ``dtype``, ``after``) inserted after the
    parameter it names (at the end for ``None``): a variant's extra operands, such as a solve readout."""
    result = list(base)
    for operand in extra:
        names = [name for name, _ in result]
        if operand.name in names or not operand.name.isidentifier():
            raise ValueError(f"operand {operand.name!r} needs a distinct identifier")
        index = len(result) if operand.after is None else names.index(operand.after) + 1
        result.insert(index, (operand.name, operand.dtype))
    return tuple(result)


def params(*items) -> tuple[tuple[str, Any], ...]:
    """Kernel parameters, in order, from ``(name, type)`` pairs, :class:`~.field_group.Field` declarations
    (their name and annotation) and generated structs (each of their fields, by :func:`.field_group.fields_of`).
    """
    result = []
    for item in items:
        if isinstance(item, field_group.Field):
            result.append((item.name, item.annotation))
        elif isinstance(item, tuple):
            name, annotation = item
            result.append((name, annotation))
        else:
            result.extend((field.name, field.annotation) for field in field_group.fields_of(item))
    names = [name for name, _ in result]
    if len(set(names)) != len(names) or not all(name.isidentifier() for name in names):
        raise ValueError(f"kernel parameters need distinct identifiers: {names}")
    return tuple(result)


@dataclasses.dataclass(frozen=True)
class Dispatch:
    """How a native kernel's Warp wrapper maps the launch's threads to calls of the native function.

    ``body`` is the wrapper's statements; its slot ``$CALL`` is the call of the native function with
    ``arguments`` then every kernel parameter. ``leading`` are the native parameters those arguments fill, and
    ``extra`` the wrapper's own parameters (for example a world queue), ahead of the kernel's.
    """

    body: str
    arguments: tuple[str, ...] = ()
    leading: tuple[tuple[str, Any], ...] = ()
    extra: tuple[tuple[str, Any], ...] = ()
    constants: tuple[tuple[str, Any], ...] = ()
    """Values of the names ``body`` reads besides its parameters (compile-time constants of the variant)."""
    private: tuple[str, ...] = ()
    """Kernel parameters only the wrapper reads (for example a bound check); the native function does not take them."""
    extra_at: int = 0
    """Where :attr:`extra` goes among the kernel's parameters (an index into them)."""


BLOCKS = Dispatch("block, _lane = wp.tid()\n$CALL", ("block",), (("block", int),))
"""One native call per thread of a 2D ``(blocks, lanes)`` launch, with the block index."""

WORLDS = Dispatch("world, _lane = wp.tid()\n$CALL", ("world",), (("world", int),))
"""One native call per thread of a 2D ``(worlds, lanes)`` launch, with the world index."""

THREADS = Dispatch("$CALL", ("wp.tid()",), (("tid", int),))
"""One native call per thread of a 1D launch, with the thread index."""


def tiles(groups: int, lanes: int) -> Dispatch:
    """World tiles: ``groups`` worlds of ``lanes`` lanes per block, guarded by a leading ``world_count``."""
    return Dispatch(
        f"block, lane = wp.tid()\nworld = block * {groups} + lane // {lanes}\nif world < world_count:\n    $CALL",
        ("world",),
        (("world", int),),
        (("world_count", int),),
    )


def candidates(groups: int, lanes: int) -> Dispatch:
    """One local candidate per lane group: ``groups`` candidates of ``lanes`` lanes per block."""
    return Dispatch(
        f"candidate, lane = wp.tid()\ncandidate = candidate * {groups} + lane // {lanes}\n$CALL",
        ("candidate",),
        (("candidate", int),),
    )


def candidate_queue(groups: int, lanes: int) -> Dispatch:
    """Lane groups that walk the active candidates (``active_candidate_count``) with a grid stride.

    The queue's operands follow the kernel's first two parameters.
    """
    return Dispatch(
        "candidate, lane = wp.tid()\ncandidate_count = active_candidate_count[0]\n"
        f"candidate = candidate * {groups} + lane // {lanes}\n"
        "while candidate < candidate_count:\n    $CALL\n    candidate += candidate_grid_stride",
        ("candidate",),
        (("candidate", int),),
        (("active_candidate_count", wp.array[int]), ("candidate_grid_stride", int)),
        extra_at=2,
    )


# Matrix-free sweeps that serve the world queue: every launch item walks the general worlds it owns.
WORLD_QUEUE = Dispatch(
    "candidate, _lane = wp.tid()\n"
    "general_index = candidate\n"
    "general_count = general_world_count[0]\n"
    "if use_general_world_queue == 0:\n"
    "    general_count = candidate + 1\n"
    "while general_index < general_count:\n"
    "    world = candidate\n"
    "    if use_general_world_queue != 0:\n"
    "        world = general_worlds[general_index]\n"
    "    $CALL\n"
    "    general_index += general_world_grid_stride",
    ("world",),
    (("world", int),),
    (
        ("general_world_count", wp.array[int]),
        ("general_worlds", wp.array[int]),
        ("general_world_grid_stride", int),
        ("use_general_world_queue", int),
    ),
)


@dataclasses.dataclass(frozen=True)
class KernelTemplate:
    """A generated native kernel, declared once.

    Args:
        name: The kernel's name (its Warp key).
        params: The kernel's parameters (:func:`params`); the native function takes the dispatch's leading
            parameters, then these.
        source: The native body: statements after the preludes (with the closing ``#endif`` of a
            ``cuda_native`` body).
        dispatch: How the wrapper calls the native function (:class:`Dispatch`).
        preludes: Device preludes of the body (:mod:`.device_preludes`).
        dual: Compose the body for both devices (``dual_native``) instead of CUDA only (``cuda_native``).
        native_name: The native function's name.
        launch_bounds: CUDA ``(threads, minimum blocks per SM)`` of the kernel, or ``None``.
        cluster_dim: Blocks per thread-block cluster (sm_90 and later): a launch's consecutive blocks in groups of
            this many share a cluster barrier and each other's shared memory; 1 for none.
        composed: ``source`` is a complete snippet whose preludes are already composed.
        module: The Python module whose Warp module holds the kernel (``__name__`` of a module-level kernel), or
            ``"unique"`` for a variant in a module of its own.
    """

    name: str
    params: tuple[tuple[str, Any], ...]
    source: str
    dispatch: Dispatch = BLOCKS
    preludes: tuple[DevicePrelude, ...] = ()
    dual: bool = False
    native_name: str = "native"
    launch_bounds: tuple[int, int] | None = None
    cluster_dim: int = 1
    composed: bool = False
    module: str = "unique"

    @classmethod
    def from_abi(cls, name: str, abi, source: str, dispatch: Dispatch, extra=(), **options) -> KernelTemplate:
        """The template of a native ABI declared whole: its first parameters are the dispatch's leading ones.

        Args:
            name: The kernel's name.
            abi: The native function's parameters, ``(name, type)`` pairs or an annotation-only function.
            source: The native body.
            dispatch: The wrapper's dispatch; the ABI must open with its leading parameters.
            extra: Operands (``name``, ``dtype``, ``after``) inserted into the ABI (:func:`operands`).
            options: Further :class:`KernelTemplate` fields.
        """
        if callable(abi):
            abi = tuple((key, value.annotation) for key, value in inspect.signature(abi).parameters.items())
        abi = operands(abi, extra)
        leading = len(dispatch.leading)
        if tuple(name for name, _ in abi[:leading]) != tuple(name for name, _ in dispatch.leading):
            raise ValueError(f"{name}: the ABI must open with the dispatch's {dispatch.leading}")
        return cls(name, abi[leading:], source, dispatch, **options)

    @property
    def snippet(self) -> str:
        """The native function's complete source: the body after its preludes."""
        if self.composed:
            if self.preludes:
                raise ValueError("a composed snippet takes no further preludes")
            return self.source
        if self.dual:
            return dual_native(self.source, preludes=self.preludes)
        return cuda_native(self.source, preludes=self.preludes)

    @property
    def native_params(self) -> tuple[tuple[str, Any], ...]:
        """The native function's ABI: the dispatch's leading parameters, then the kernel's (except private ones)."""
        return (*self.dispatch.leading, *(item for item in self.params if item[0] not in self.dispatch.private))

    @property
    def signature(self) -> tuple[tuple[str, Any], ...]:
        """The Warp kernel's signature: the kernel's parameters with the dispatch's own at its place."""
        at = self.dispatch.extra_at
        return (*self.params[:at], *self.dispatch.extra, *self.params[at:])

    def declaration(self) -> str:
        """The Python source of the native function and its Warp wrapper (annotations by ``type_<name>`` alias)."""
        operands = (name for name, _ in self.params if name not in self.dispatch.private)
        call = f"{self.native_name}({', '.join((*self.dispatch.arguments, *operands))})"
        body = substitute(self.dispatch.body, CALL=call)
        return (
            f"@wp.func_native(snippet)\ndef {self.native_name}({_annotated(self.native_params)}): ...\n\n"
            f"def kernel_template({_annotated(self.signature)}):\n"
            + "".join(f"    {line}\n" for line in body.splitlines())
        )

    def kernel(self) -> wp.Kernel:
        """Build the Warp kernel (backward disabled)."""
        source, snippet = self.declaration(), self.snippet
        # The functions belong to the kernel's Python module, or to this one for a variant in a module of its own.
        owner = __name__ if self.module == "unique" else self.module
        namespace = {"__name__": owner, "wp": wp, "snippet": snippet, **dict(self.dispatch.constants)}
        namespace.update({f"type_{name}": annotation for name, annotation in (*self.native_params, *self.signature)})
        digest = hashlib.sha256((source + snippet).encode()).hexdigest()
        filename = f"<fpgs-kernel-{digest}>"
        linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
        exec(compile(source, filename, "exec"), namespace)
        function = namespace["kernel_template"]
        function.__name__ = function.__qualname__ = self.name
        options = {"launch_bounds": self.launch_bounds} if self.launch_bounds is not None else {}
        if self.cluster_dim > 1:
            options["cluster_dim"] = self.cluster_dim
        if self.module == "unique":
            options["module"] = "unique"
        return wp.kernel(enable_backward=False, **options)(function)


def _annotated(items) -> str:
    return ", ".join(f"{name}: type_{name}" for name, _ in items)


def python_kernel(function, name: str, **options) -> wp.Kernel:
    """A Python kernel variant: ``function`` renamed ``name``, built in a module of its own.

    Args:
        function: The kernel's Python function (a closure over the variant's constants).
        name: The variant's name (its Warp key).
        options: Options of ``wp.kernel`` besides the module; backward is disabled unless given.
    """
    function.__name__ = function.__qualname__ = name
    return wp.kernel(**{"enable_backward": False, **options}, module="unique")(function)


@dataclasses.dataclass(frozen=True)
class SweepTier:
    """A tier sweep of the matrix-free solve: its kernel and what its launches and the route record read.

    The typed launch descriptor of a tier, in place of attributes set on its kernel.

    Args:
        kernel: The tier's Warp kernel.
        block_dim: Threads per block of its tiled launches.
        tier: Its index among the parallel tiers' world lists, or ``None`` for a tier on the general queue.
        response_inputs: It takes the in-kernel response inputs after ``omega``.
        kinetic_rows: It reads the keyed rows' operands after those.
        linked_rows: It projects each friction pair on its patch ring's pooled normal load.
        velocity_limit_pass: It runs the last-word velocity-limit pass after its final sweep.
        sweeps: Its sweep budget (0 for an incremental tier).
        colored_worlds: It reads each world's mapping before ``v_out`` and leaves the colored worlds
            (:data:`.sweep_routes.COLORED_WORLDS_ABI`).
    """

    kernel: wp.Kernel
    block_dim: int = 32
    tier: int | None = None
    response_inputs: bool = False
    kinetic_rows: bool = False
    linked_rows: bool = False
    velocity_limit_pass: bool = False
    sweeps: int = 0
    colored_worlds: bool = False
