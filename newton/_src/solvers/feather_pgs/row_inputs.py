# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compile-time row operands and explicit Warp native launch generation.

Readers describe storage, not a solve policy. Local ordered consumers read J;
parallel consumers may stage Z plus live metadata. Their recurrences stay in
the ordinary factories. Extra operands extend the declared native signature;
no solver source is inspected, copied or patched.
"""

import hashlib
import inspect
import linecache
from dataclasses import dataclass

import warp as wp


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


def make_row_kernel(name, snippet, signature, arguments=(), *, dispatch, groups=1, lanes=32, launch_bounds=None):
    """Generate a typed native function and one explicit launch wrapper.

    ``signature`` is an annotation-only declaration of the base native ABI.
    Ordinary consumers retain exactly that ABI. The launch schedules below
    match the existing disjoint local queues and parallel world queue, and the
    native sweeps' world tiles (``groups`` worlds of ``lanes`` lanes per block,
    guarded by a leading ``world_count``) and one world per block, whose
    first native operand is the world. ``launch_bounds`` is the CUDA
    ``(threads, minimum blocks per SM)`` pair of the kernel, or ``None``.
    """
    params = [(key, value.annotation) for key, value in inspect.signature(signature).parameters.items()]
    for operand in arguments:
        names = [key for key, _ in params]
        if operand.name in names or not operand.name.isidentifier():
            raise ValueError("row operands must have distinct Python argument names")
        index = len(params) if operand.after is None else names.index(operand.after) + 1
        params.insert(index, (operand.name, operand.dtype))
    launch = params[1:]
    call = "native(" + ", ".join(key for key, _ in params) + ")"
    if dispatch == "local":
        body = f"candidate, lane = wp.tid()\ncandidate = candidate * {groups} + lane // {lanes}\n{call}"
    elif dispatch == "local_queue":
        launch = [*launch[:2], ("active_candidate_count", wp.array[int]), ("candidate_grid_stride", int), *launch[2:]]
        body = (
            "candidate, lane = wp.tid()\ncandidate_count = active_candidate_count[0]\n"
            f"candidate = candidate * {groups} + lane // {lanes}\n"
            f"while candidate < candidate_count:\n    {call}\n    candidate += candidate_grid_stride"
        )
    elif dispatch in ("tiles", "worlds"):
        if params[0][0] != "world":
            raise ValueError(f"{dispatch} dispatch requires the world as the first native operand")
        if dispatch == "tiles":
            launch = [("world_count", int), *launch]
            body = f"block, lane = wp.tid()\nworld = block * {groups} + lane // {lanes}\nif world < world_count:\n    {call}"
        else:
            body = f"world, _lane = wp.tid()\n{call}"
    elif dispatch == "parallel":
        launch = [
            ("general_world_count", wp.array[int]),
            ("general_worlds", wp.array[int]),
            ("general_world_grid_stride", int),
            ("use_general_world_queue", int),
            *launch,
        ]
        body = (
            "candidate, _lane = wp.tid()\ngeneral_index = candidate\ngeneral_count = general_world_count[0]\n"
            "if use_general_world_queue == 0:\n    general_count = candidate + 1\n"
            "while general_index < general_count:\n    world = candidate\n"
            "    if use_general_world_queue != 0:\n        world = general_worlds[general_index]\n"
            f"    {call}\n    general_index += general_world_grid_stride"
        )
    else:
        raise ValueError(f"unknown row dispatch: {dispatch}")
    namespace = {"wp": wp, "snippet": snippet}

    def declaration(items):
        annotations = []
        for key, dtype in items:
            alias = f"type_{key}"
            namespace[alias] = dtype
            annotations.append(f"{key}: {alias}")
        return ", ".join(annotations)

    source = (
        f"@wp.func_native(snippet)\ndef native({declaration(params)}): ...\n\n"
        f"def kernel_template({declaration(launch)}):\n" + "\n".join("    " + line for line in body.splitlines()) + "\n"
    )
    filename = f"<fpgs-row-kernel-{hashlib.sha256((source + snippet).encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    exec(compile(source, filename, "exec"), namespace)
    template = namespace["kernel_template"]
    template.__name__ = template.__qualname__ = name
    kernel = wp.kernel(enable_backward=False, module="unique", launch_bounds=launch_bounds)(template)
    kernel._fpgs_row_native_source = snippet
    return kernel
