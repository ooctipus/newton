# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU world identities, admission, membership and relocation planning.

Prepared domains own state, reset semantics and physics. The directory never
imports a domain, allocates physical state or assumes a state width.
"""

import weakref
from enum import IntEnum

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

NOOP, CREATE, RESET, DESTROY = 0, 1, 2, 3
OK, INVALID, STALE, NOT_ALIVE, BAD_PROTOTYPE, CONFLICT = range(6)
NO_IDS, NO_SLOTS, GENERATION_EXHAUSTED = 6, 7, 8
BAD_COUNT, STALE_BATCH, INITIALIZATION_MISSING, COMPACTION_INVALID, PHASE_INVALID = 9, 10, 11, 12, 13
UNBACKED, READY, LIVE = 0, 1, 2
IDLE, VALIDATED, ADMITTED, MOVING, COPIED = range(5)
STATUS_NAMES = (
    "ok",
    "invalid",
    "stale",
    "not_alive",
    "bad_prototype",
    "conflict",
    "no_ids",
    "no_slots",
    "generation_exhausted",
    "bad_count",
    "stale_batch",
    "initialization_missing",
    "compaction_invalid",
    "phase_invalid",
)


class WorldOperation(IntEnum):
    """Experimental lifecycle commands accepted by :class:`newton.worlds.WorldDirectory`."""

    NOOP = NOOP
    CREATE = CREATE
    RESET = RESET
    DESTROY = DESTROY


class WorldStatus(IntEnum):
    """Experimental per-request results; unsuccessful replacement preserves the old lifetime."""

    OK = OK
    INVALID = INVALID
    STALE = STALE
    NOT_ALIVE = NOT_ALIVE
    BAD_PROTOTYPE = BAD_PROTOTYPE
    CONFLICT = CONFLICT
    NO_IDS = NO_IDS
    NO_SLOTS = NO_SLOTS
    GENERATION_EXHAUSTED = GENERATION_EXHAUSTED
    BAD_COUNT = BAD_COUNT
    STALE_BATCH = STALE_BATCH
    INITIALIZATION_MISSING = INITIALIZATION_MISSING
    COMPACTION_INVALID = COMPACTION_INVALID
    PHASE_INVALID = PHASE_INVALID


class WorldPhase(IntEnum):
    """Experimental transaction phases used by caller-recorded validation and initialization."""

    IDLE = IDLE
    VALIDATED = VALIDATED
    ADMITTED = ADMITTED
    MOVING = MOVING
    COPIED = COPIED


@wp.struct
class WorldCommands:
    """Experimental caller-owned command arrays and monotonically increasing batch sequence."""

    sequence: wp.array[wp.uint64]
    count: wp.array[int]
    op: wp.array[int]
    id: wp.array[int]
    generation: wp.array[wp.uint64]
    prototype: wp.array[int]


@wp.struct
class WorldResults:
    """Experimental per-request status, assigned identity and generation arrays."""

    status: wp.array[int]
    id: wp.array[int]
    generation: wp.array[wp.uint64]


@wp.struct
class WorldDirectoryData:
    """Experimental GPU identity, slot and membership arrays owned by WorldDirectory."""

    prototype: wp.array[int]
    slot: wp.array[int]  # prototype-local row; global slot arrays use starts[prototype] + slot
    generation: wp.array[wp.uint64]
    free_ids: wp.array[int]
    id_count: wp.array[int]
    conflicts: wp.array[int]
    starts: wp.array[int]
    slot_prototype: wp.array[int]
    slot_state: wp.array[int]
    slot_id: wp.array[int]
    slot_rank: wp.array[int]
    active: wp.array[int]
    ready: wp.array[int]
    active_count: wp.array[int]
    ready_count: wp.array[int]
    demand: wp.array[int]
    sequence: wp.array[wp.uint64]
    flags: wp.array[int]  # consume, advancement permitted, batch error


@wp.struct
class WorldTransaction:
    """Experimental one-batch admission records and explicit initialization acknowledgements."""

    phase: wp.array[int]
    permit: wp.array[int]
    initialized: wp.array[wp.uint64]  # acknowledged batch sequence, per request
    status: wp.array[int]
    destination_id: wp.array[int]
    destination_slot: wp.array[int]  # prototype-local row, including before publication
    observed_generation: wp.array[wp.uint64]
    claims: wp.array[int]  # one ticket counter per prototype, followed by the ID counter
    dirty: wp.array[int]  # any successfully published lifetime change in this batch
    request_rank: wp.array[int]
    accepted_requests: wp.array[int]
    group_starts: wp.array[int]  # accepted request ranges; length prototypes + 1


@wp.struct
class WorldCompaction:
    """Experimental source/destination row plans and domain-copy acknowledgements."""

    source: wp.array[int]
    destination: wp.array[int]
    sources: wp.array[int]
    destinations: wp.array[int]
    copied: wp.array[int]  # completed domain row copies per prototype


@wp.kernel
def _begin(d: WorldDirectoryData, t: WorldTransaction, c: WorldCommands, command_capacity: int):
    d.flags[0] = 0
    if t.phase[0] != IDLE:
        d.flags[1] = 0
        d.flags[2] = PHASE_INVALID
        t.phase[0] = IDLE
        return
    if c.sequence[0] == wp.uint64(0) or c.sequence[0] < d.sequence[0]:
        d.flags[1] = 0
        d.flags[2] = STALE_BATCH
    elif c.sequence[0] > d.sequence[0]:
        d.sequence[0] = c.sequence[0]
        d.flags[2] = 0
        if c.count[0] < 0 or c.count[0] > command_capacity:
            d.flags[1] = 0
            d.flags[2] = BAD_COUNT
        else:
            d.flags[0] = 1
            d.flags[1] = 0
            t.permit[0] = 1
            t.phase[0] = VALIDATED


@wp.kernel
def _clear_transaction(
    d: WorldDirectoryData, t: WorldTransaction, directory_capacity: int, command_capacity: int, p: int
):
    i = wp.tid()
    if d.flags[0] != 0:
        if i < directory_capacity:
            d.conflicts[i] = 0
        if i < command_capacity:
            t.status[i] = OK
            t.initialized[i] = wp.uint64(0)
            t.destination_id[i] = -1
            t.destination_slot[i] = -1
            t.request_rank[i] = -1
        if i < p:
            d.demand[i] = 0
            t.claims[i] = 0
        if i == 0:
            t.claims[p] = 0
            t.dirty[0] = 0


@wp.kernel
def _validate(d: WorldDirectoryData, c: WorldCommands, t: WorldTransaction, directory_capacity: int, prototypes: int):
    i = wp.tid()
    if d.flags[0] == 0 or i >= c.count[0]:
        return
    op, identity, prototype = int(c.op[i]), int(c.id[i]), int(c.prototype[i])
    status = int(OK)
    t.observed_generation[i] = wp.uint64(0)
    if identity >= 0 and identity < directory_capacity:
        t.observed_generation[i] = d.generation[identity]
    if op < NOOP or op > DESTROY:
        status = INVALID
    elif op == RESET or op == DESTROY:
        if identity < 0 or identity >= directory_capacity:
            status = INVALID
        else:
            wp.atomic_add(d.conflicts, identity, 1)
            if d.prototype[identity] < 0:
                status = NOT_ALIVE
            elif d.generation[identity] != c.generation[i]:
                status = STALE
            elif d.generation[identity] == wp.uint64(18446744073709551615):
                status = GENERATION_EXHAUSTED
    if status == OK and (op == CREATE or op == RESET):
        if prototype < 0 or prototype >= prototypes:
            status = BAD_PROTOTYPE
    t.status[i] = status


@wp.kernel
def _admit(d: WorldDirectoryData, c: WorldCommands, t: WorldTransaction, directory_capacity: int):
    i = wp.tid()
    if d.flags[0] == 0 or i >= c.count[0]:
        return
    op, identity, prototype = int(c.op[i]), int(c.id[i]), int(c.prototype[i])
    status = int(t.status[i])
    if (op == RESET or op == DESTROY) and identity >= 0 and identity < directory_capacity:
        if d.conflicts[identity] > 1:
            status = CONFLICT
    if status == OK and (op == CREATE or op == RESET):
        if op == CREATE:
            ticket = wp.atomic_add(t.claims, t.claims.shape[0] - 1, 1)
            available = d.id_count[0]
            if ticket >= available:
                status = NO_IDS
            else:
                identity = int(d.free_ids[available - ticket - 1])
                t.destination_id[i] = identity
        else:
            t.destination_id[i] = identity
        if status == OK:
            ticket = wp.atomic_add(t.claims, prototype, 1)
            available = d.ready_count[prototype]
            if ticket >= available:
                status = NO_SLOTS
                wp.atomic_add(d.demand, prototype, 1)
            else:
                t.destination_slot[i] = d.ready[d.starts[prototype] + available - ticket - 1]
                t.request_rank[i] = ticket
    t.status[i] = status
    if status != OK:
        wp.atomic_min(t.permit, 0, 0)


@wp.kernel
def _group_starts(d: WorldDirectoryData, t: WorldTransaction, prototypes: int):
    if d.flags[0] == 0:
        return
    # The tiny prototype prefix is O(P); never walk requests serially.
    total = int(0)
    t.group_starts[0] = 0
    for prototype in range(prototypes):
        total += wp.min(t.claims[prototype], d.ready_count[prototype])
        t.group_starts[prototype + 1] = total


@wp.kernel
def _group_requests(d: WorldDirectoryData, c: WorldCommands, t: WorldTransaction):
    request = wp.tid()
    if d.flags[0] != 0 and request < c.count[0]:
        rank = t.request_rank[request]
        if rank >= 0:
            t.accepted_requests[t.group_starts[c.prototype[request]] + rank] = request


@wp.kernel
def _publish(d: WorldDirectoryData, c: WorldCommands, t: WorldTransaction, r: WorldResults):
    i = wp.tid()
    if d.flags[0] == 0 or i >= c.count[0]:
        return
    if t.phase[0] != ADMITTED:
        return
    status, op, identity = int(t.status[i]), int(c.op[i]), int(c.id[i])
    if status == OK and (op == CREATE or op == RESET) and t.initialized[i] != d.sequence[0]:
        status = INITIALIZATION_MISSING
        t.status[i] = status
        wp.atomic_min(t.permit, 0, 0)
    if status == OK and op != NOOP:
        wp.atomic_max(t.dirty, 0, 1)
        if op == CREATE:
            identity = int(t.destination_id[i])
        if op == RESET or op == DESTROY:
            source = d.starts[d.prototype[identity]] + d.slot[identity]
            d.slot_state[source] = READY
            d.slot_id[source] = -1
            d.slot_rank[source] = -1
        d.generation[identity] = d.generation[identity] + wp.uint64(1)
        if op == DESTROY:
            d.prototype[identity] = -1
            d.slot[identity] = -1
        else:
            prototype, slot = c.prototype[i], t.destination_slot[i]
            d.prototype[identity] = prototype
            d.slot[identity] = slot
            destination = d.starts[prototype] + slot
            d.slot_state[destination] = LIVE
            d.slot_id[destination] = identity
    r.status[i] = status
    r.id[i] = identity
    if status == OK and op != NOOP:
        r.generation[i] = d.generation[identity]
    else:
        r.generation[i] = t.observed_generation[i]


@wp.kernel
def _clear_membership(d: WorldDirectoryData, t: WorldTransaction, prototypes: int, conditional: int):
    i = wp.tid()
    if conditional == 0 or (d.flags[0] != 0 and t.dirty[0] != 0):
        if i < prototypes:
            d.active_count[i] = 0
            d.ready_count[i] = 0
        if i == 0:
            d.id_count[0] = 0


@wp.kernel
def _rebuild(d: WorldDirectoryData, t: WorldTransaction, directory_capacity: int, slot_capacity: int, conditional: int):
    i = wp.tid()
    if conditional != 0 and (d.flags[0] == 0 or t.dirty[0] == 0):
        return
    if i < directory_capacity:
        if d.prototype[i] < 0 and d.generation[i] < wp.uint64(18446744073709551615):
            rank = wp.atomic_add(d.id_count, 0, 1)
            d.free_ids[rank] = i
    if i < slot_capacity:
        prototype = d.slot_prototype[i]
        local_slot = i - d.starts[prototype]
        if d.slot_state[i] == LIVE:
            rank = wp.atomic_add(d.active_count, prototype, 1)
            d.active[d.starts[prototype] + rank] = local_slot
            d.slot_rank[i] = rank
        elif d.slot_state[i] == READY:
            rank = wp.atomic_add(d.ready_count, prototype, 1)
            d.ready[d.starts[prototype] + rank] = local_slot
            d.slot_rank[i] = -1


@wp.kernel
def _clear_compaction(moves: WorldCompaction):
    prototype = wp.tid()
    moves.sources[prototype] = 0
    moves.destinations[prototype] = 0
    moves.copied[prototype] = 0


@wp.kernel
def _plan_compaction(d: WorldDirectoryData, t: WorldTransaction, moves: WorldCompaction):
    index = wp.tid()
    if t.phase[0] != MOVING:
        return
    prototype = d.slot_prototype[index]
    start = d.starts[prototype]
    slot = index - start
    count = d.active_count[prototype]
    if slot < count and d.slot_state[index] == READY:
        rank = wp.atomic_add(moves.destinations, prototype, 1)
        moves.destination[start + rank] = slot
    elif slot >= count and d.slot_state[index] == LIVE:
        rank = wp.atomic_add(moves.sources, prototype, 1)
        moves.source[start + rank] = slot


@wp.kernel
def _validate_compaction(
    d: WorldDirectoryData, t: WorldTransaction, moves: WorldCompaction, prototypes: int, require_copy: int
):
    if t.phase[0] != MOVING:
        if require_copy != 0:
            d.flags[1] = 0
            d.flags[2] = PHASE_INVALID
        return
    complete = int(1)
    for prototype in range(prototypes):
        if moves.sources[prototype] != moves.destinations[prototype]:
            complete = 0
        if require_copy != 0 and moves.copied[prototype] != moves.sources[prototype]:
            complete = 0
    if complete == 0:
        d.flags[2] = COMPACTION_INVALID
        t.permit[0] = 0
        t.phase[0] = IDLE
    elif require_copy != 0:
        t.phase[0] = COPIED


@wp.kernel
def _publish_compaction(d: WorldDirectoryData, t: WorldTransaction, moves: WorldCompaction):
    index = wp.tid()
    if t.phase[0] != COPIED:
        return
    prototype = d.slot_prototype[index]
    start = d.starts[prototype]
    rank = index - start
    if rank < moves.sources[prototype]:
        source, destination = moves.source[start + rank], moves.destination[start + rank]
        identity = d.slot_id[start + source]
        d.slot_state[start + destination] = LIVE
        d.slot_id[start + destination] = identity
        d.slot[identity] = destination
        d.slot_state[start + source] = READY
        d.slot_id[start + source] = -1
        d.slot_rank[start + source] = -1


@wp.kernel
def _check_shrink(d: WorldDirectoryData, ends: wp.array[int], failure: wp.array[int]):
    slot = wp.tid()
    prototype = d.slot_prototype[slot]
    if slot - d.starts[prototype] >= ends[prototype] and d.slot_state[slot] == LIVE:
        wp.atomic_max(failure, 0, 1)


@wp.kernel
def _publish_capacity(d: WorldDirectoryData, ends: wp.array[int]):
    slot = wp.tid()
    prototype = d.slot_prototype[slot]
    if slot - d.starts[prototype] < ends[prototype] and d.slot_state[slot] != LIVE:
        d.slot_state[slot] = READY
        d.slot_rank[slot] = -1


def _validate_arrays(record, schema, capacity, device, scalars=()):
    for name, field in schema.vars.items():
        array = getattr(record, name)
        shape = (1,) if name in scalars else (capacity,)
        if (
            array is None
            or array.shape != shape
            or array.device != device
            or array.dtype != field.type.dtype
            or not array.is_contiguous
        ):
            raise ValueError(f"{schema.key}.{name} must be contiguous {field.type.dtype}, shape {shape}, on {device}")


def create_world_commands(capacity: int, *, device=None) -> WorldCommands:
    """Allocate caller-owned, capture-stable inputs; sequence 0 is deliberately invalid."""
    if type(capacity) is not int or not 1 <= capacity < 2**31:
        raise ValueError("Command capacity must be a positive int32 integer")
    c = WorldCommands()
    c.sequence = wp.zeros(1, dtype=wp.uint64, device=device)
    c.count = wp.zeros(1, dtype=int, device=device)
    for name in ("op", "id", "prototype"):
        setattr(c, name, wp.zeros(capacity, dtype=int, device=device))
    c.generation = wp.zeros(capacity, dtype=wp.uint64, device=device)
    return c


def create_world_results(capacity: int, *, device=None) -> WorldResults:
    """Allocate caller-owned results for a prepared request capacity."""
    if type(capacity) is not int or not 1 <= capacity < 2**31:
        raise ValueError("Result capacity must be a positive int32 integer")
    r = WorldResults()
    r.status = wp.zeros(capacity, dtype=int, device=device)
    r.id = wp.full(capacity, -1, dtype=int, device=device)
    r.generation = wp.zeros(capacity, dtype=wp.uint64, device=device)
    return r


@wp.kernel
def _transition(d: WorldDirectoryData, t: WorldTransaction, c: WorldCommands, expected: int, target: int):
    if d.flags[0] == 0:
        return
    if t.phase[0] != expected or c.sequence[0] != d.sequence[0]:
        d.flags[1] = 0
        d.flags[2] = PHASE_INVALID
        t.permit[0] = 0
        t.phase[0] = IDLE
        d.flags[0] = 0
    else:
        t.phase[0] = target


@wp.kernel
def _finish(d: WorldDirectoryData, t: WorldTransaction):
    if d.flags[0] != 0 and t.phase[0] == ADMITTED:
        t.phase[0] = IDLE
        d.flags[1] = t.permit[0]


@wp.kernel
def _begin_moves(d: WorldDirectoryData, t: WorldTransaction):
    if t.phase[0] != IDLE:
        d.flags[2] = PHASE_INVALID
        t.permit[0] = 0
        t.phase[0] = IDLE
    else:
        t.permit[0] = d.flags[1]
        t.phase[0] = MOVING
    d.flags[1] = 0


@wp.kernel
def _finish_moves(d: WorldDirectoryData, t: WorldTransaction):
    if t.phase[0] == COPIED:
        t.phase[0] = IDLE
        d.flags[1] = t.permit[0]


@wp.kernel
def _withdraw_ready(d: WorldDirectoryData, ends: wp.array[int]):
    slot = wp.tid()
    prototype = d.slot_prototype[slot]
    if slot - d.starts[prototype] >= ends[prototype] and d.slot_state[slot] == READY:
        d.slot_state[slot] = UNBACKED
        d.slot_rank[slot] = -1


class WorldDirectory:
    """Experimental identity, admission and membership owner without simulation state.

    Args:
        slot_limits: Prepared row limits for each prototype, with an int32 total.
        id_capacity: Maximum simultaneous logical identities.
        command_capacity: Maximum requests in a captured batch.
        device: Warp device owning the directory records.

    Domains initialize admitted rows and acknowledge complete relocation before
    publication. Callers must not mutate the lifetime records directly.
    """

    def __init__(self, slot_limits, *, id_capacity=4096, command_capacity=4096, device=None):
        slot_limits = tuple(slot_limits)
        if not slot_limits or any(type(n) is not int or not 1 <= n < 2**31 for n in slot_limits):
            raise ValueError("One positive int32 slot limit is required per prototype")
        if sum(slot_limits) >= 2**31:
            raise ValueError("Combined prototype slots must fit int32 indices")
        if any(type(n) is not int or not 1 <= n < 2**31 for n in (id_capacity, command_capacity)):
            raise ValueError("Identity and command capacities must be positive int32 integers")
        self.slot_limits = slot_limits
        self.id_capacity, self.command_capacity = id_capacity, command_capacity
        self.slot_capacity, self.device = sum(slot_limits), wp.get_device(device)
        self._graphs, self._closed = weakref.WeakSet(), False
        self.d = WorldDirectoryData()
        d = self.d
        for name in ("prototype", "slot"):
            setattr(d, name, wp.full(id_capacity, -1, dtype=int, device=self.device))
        d.generation = wp.zeros(id_capacity, dtype=wp.uint64, device=self.device)
        d.free_ids = wp.zeros(id_capacity, dtype=int, device=self.device)
        d.id_count = wp.zeros(1, dtype=int, device=self.device)
        d.conflicts = wp.zeros(id_capacity, dtype=int, device=self.device)
        starts = np.concatenate(([0], np.cumsum(slot_limits))).astype(np.int32)
        d.starts = wp.array(starts, dtype=int, device=self.device)
        d.slot_prototype = wp.array(
            np.repeat(np.arange(len(slot_limits)), slot_limits).astype(np.int32), dtype=int, device=self.device
        )
        d.slot_state = wp.full(self.slot_capacity, UNBACKED, dtype=int, device=self.device)
        for name in ("slot_id", "slot_rank"):
            setattr(d, name, wp.full(self.slot_capacity, -1, dtype=int, device=self.device))
        for name in ("active", "ready"):
            setattr(d, name, wp.zeros(self.slot_capacity, dtype=int, device=self.device))
        for name in ("active_count", "ready_count", "demand"):
            setattr(d, name, wp.zeros(len(slot_limits), dtype=int, device=self.device))
        d.sequence = wp.zeros(1, dtype=wp.uint64, device=self.device)
        d.flags = wp.zeros(3, dtype=int, device=self.device)
        self.t = WorldTransaction()
        for name in ("status", "destination_id", "destination_slot", "request_rank", "accepted_requests"):
            setattr(self.t, name, wp.zeros(command_capacity, dtype=int, device=self.device))
        for name in ("observed_generation", "initialized"):
            setattr(self.t, name, wp.zeros(command_capacity, dtype=wp.uint64, device=self.device))
        self.t.claims = wp.zeros(len(slot_limits) + 1, dtype=int, device=self.device)
        self.t.group_starts = wp.zeros(len(slot_limits) + 1, dtype=int, device=self.device)
        for name in ("dirty", "phase", "permit"):
            setattr(self.t, name, wp.zeros(1, dtype=int, device=self.device))
        self.moves = WorldCompaction()
        for name in ("source", "destination"):
            setattr(self.moves, name, wp.zeros(self.slot_capacity, dtype=int, device=self.device))
        for name in ("sources", "destinations", "copied"):
            setattr(self.moves, name, wp.zeros(len(slot_limits), dtype=int, device=self.device))
        self._capacity_failure = wp.zeros(1, dtype=int, device=self.device)
        self._capacity_ends = wp.zeros(len(slot_limits), dtype=int, device=self.device)
        self._rebuild(conditional=False)

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("WorldDirectory is closed")

    def _rebuild(self, *, conditional):
        prototypes = len(self.slot_limits)
        wp.launch(
            _clear_membership, max(1, prototypes), [self.d, self.t, prototypes, int(conditional)], device=self.device
        )
        wp.launch(
            _rebuild,
            max(self.id_capacity, self.slot_capacity),
            [self.d, self.t, self.id_capacity, self.slot_capacity, int(conditional)],
            device=self.device,
        )

    def begin(self, commands: WorldCommands):
        """Record batch validation; domain payload validation may follow before admission."""
        self._ensure_open()
        _validate_arrays(commands, WorldCommands, self.command_capacity, self.device, ("sequence", "count"))
        prototypes = len(self.slot_limits)
        wp.launch(_begin, 1, [self.d, self.t, commands, self.command_capacity], device=self.device)
        wp.launch(
            _clear_transaction,
            max(self.id_capacity, self.command_capacity, prototypes),
            [self.d, self.t, self.id_capacity, self.command_capacity, prototypes],
            device=self.device,
        )
        wp.launch(
            _validate,
            self.command_capacity,
            [self.d, commands, self.t, self.id_capacity, prototypes],
            device=self.device,
        )

    def admit(self, commands: WorldCommands):
        """Assign destination tickets without publishing or altering old lifetimes."""
        self._ensure_open()
        _validate_arrays(commands, WorldCommands, self.command_capacity, self.device, ("sequence", "count"))
        wp.launch(_transition, 1, [self.d, self.t, commands, VALIDATED, ADMITTED], device=self.device)
        wp.launch(_admit, self.command_capacity, [self.d, commands, self.t, self.id_capacity], device=self.device)
        wp.launch(_group_starts, 1, [self.d, self.t, len(self.slot_limits)], device=self.device)
        wp.launch(_group_requests, self.command_capacity, [self.d, commands, self.t], device=self.device)

    def publish(self, commands: WorldCommands, results: WorldResults):
        """Publish successfully initialized requests and gate advancement on batch errors."""
        self._ensure_open()
        _validate_arrays(commands, WorldCommands, self.command_capacity, self.device, ("sequence", "count"))
        _validate_arrays(results, WorldResults, self.command_capacity, self.device)
        wp.launch(_transition, 1, [self.d, self.t, commands, ADMITTED, ADMITTED], device=self.device)
        wp.launch(_publish, self.command_capacity, [self.d, commands, self.t, results], device=self.device)
        self._rebuild(conditional=True)
        wp.launch(_finish, 1, [self.d, self.t], device=self.device)

    def plan_moves(self):
        """Plan nonoverlapping retained-row moves into each prototype's live prefix."""
        self._ensure_open()
        wp.launch(_begin_moves, 1, [self.d, self.t], device=self.device)
        wp.launch(_clear_compaction, len(self.slot_limits), [self.moves], device=self.device)
        wp.launch(_plan_compaction, self.slot_capacity, [self.d, self.t, self.moves], device=self.device)
        wp.launch(_validate_compaction, 1, [self.d, self.t, self.moves, len(self.slot_limits), 0], device=self.device)

    def publish_moves(self):
        """Update membership only after the domain acknowledges every planned copy."""
        self._ensure_open()
        wp.launch(_validate_compaction, 1, [self.d, self.t, self.moves, len(self.slot_limits), 1], device=self.device)
        wp.launch(_publish_compaction, self.slot_capacity, [self.d, self.t, self.moves], device=self.device)
        self._rebuild(conditional=False)
        wp.launch(_finish_moves, 1, [self.d, self.t], device=self.device)

    def _capacity_args(self, ends):
        self._ensure_open()
        ends = tuple(ends)
        if len(ends) != len(self.slot_limits) or any(
            type(end) is not int or not 0 <= end <= capacity
            for end, capacity in zip(ends, self.slot_limits, strict=True)
        ):
            raise ValueError("Ready prefix is outside the prepared slot limits")
        if self.t.phase.numpy()[0] != IDLE:
            raise RuntimeError("Capacity service cannot interrupt an open transaction")
        self._capacity_ends.assign(np.asarray(ends, dtype=np.int32))

    def withdraw_ready(self, ends: tuple[int, ...]):
        """Withdraw every requested tail together before domain storage is retired."""
        self._capacity_args(ends)
        self._capacity_failure.zero_()
        wp.launch(
            _check_shrink, self.slot_capacity, [self.d, self._capacity_ends, self._capacity_failure], device=self.device
        )
        if self._capacity_failure.numpy()[0]:
            raise RuntimeError("Cannot withdraw a range containing live worlds; compact or destroy them first")
        wp.launch(_withdraw_ready, self.slot_capacity, [self.d, self._capacity_ends], device=self.device)
        self._rebuild(conditional=False)

    def publish_ready(self, ends: tuple[int, ...]):
        """Publish jointly usable prefixes with one membership rebuild for the batch."""
        self._capacity_args(ends)
        wp.launch(_publish_capacity, self.slot_capacity, [self.d, self._capacity_ends], device=self.device)
        self._rebuild(conditional=False)

    def retain_graph(self, graph, *buffers):
        """Keep the directory and explicit external buffers alive with one captured graph."""
        self._ensure_open()
        if graph.device != self.device:
            raise ValueError("Graph and directory must use the same device")
        graph.world_owners = (*getattr(graph, "world_owners", ()), self, *buffers)
        self._graphs.add(graph)
        return graph

    def close(self, *, streams):
        """Join consumers and reject future operations after all graph borrowers are gone."""
        if self._graphs:
            raise RuntimeError("Destroy retained graphs before closing their directory")
        if self.device.is_cuda:
            if not tuple(streams):
                raise ValueError("CUDA directory closure requires explicit consumer dependencies")
            wp.synchronize_device(self.device)
        self._closed = True

    def memory_report(self):
        """Report owned directory metadata, excluding every domain and graph allocation."""
        return {
            "directory_metadata_bytes": sum(
                getattr(record, name).capacity for record in (self.d, self.t, self.moves) for name in record._cls.vars
            )
            + self._capacity_failure.capacity
            + self._capacity_ends.capacity
        }
