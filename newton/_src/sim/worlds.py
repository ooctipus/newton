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
    """Experimental request and batch outcomes; rejected replacement preserves the old lifetime."""

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
    """Experimental caller-owned inputs for one positive, increasing batch sequence.

    ``sequence`` and ``count`` have length one; all other arrays have prepared
    command capacity. Only ``[0, count[0])`` is consumed. Keep these inputs stable
    from begin through publication and while a recorded replay is in flight.
    CREATE ignores the supplied handle. RESET and DESTROY require a live handle.
    Equal-sequence replays ignore input edits; retry with a higher sequence.
    """

    sequence: wp.array[wp.uint64]
    """Positive batch sequence, shape [1]."""
    count: wp.array[int]
    """Number of requests, shape [1]."""
    op: wp.array[int]
    """WorldOperation values, indexed by request ordinal."""
    id: wp.array[int]
    """Logical identity to reset or destroy, indexed by request ordinal."""
    generation: wp.array[wp.uint64]
    """Expected lifetime generation, indexed by request ordinal."""
    prototype: wp.array[int]
    """Numeric destination prototype for CREATE or RESET, indexed by request ordinal."""


@wp.struct
class WorldResults:
    """Experimental caller-owned outputs, each of shape [command_capacity].

    Publication writes the consumed request prefix. Unconsumed entries retain
    their previous values. Inspect WorldBatch.consumed and WorldBatch.status
    before associating results with a batch; WorldBatch.advance separately gates
    physics. Independent requests can succeed even when advancement is denied.
    Only a successful CREATE or RESET returns a newly usable lifetime handle.
    """

    status: wp.array[int]
    """Per-request WorldStatus; batch errors are reported in WorldBatch.status."""
    id: wp.array[int]
    """Result identity, indexed by request ordinal, never by world ID."""
    generation: wp.array[wp.uint64]
    """Result generation on success; observed pre-batch generation on rejection."""


@wp.struct
class WorldBatch:
    """Experimental named batch outcome; each array has shape [1].

    The directory owns these values. A domain composition root may suppress
    consumed/advance and publish a batch error to quarantine its own failed
    physical storage; it must never grant consumption or advancement itself.
    Equal-sequence replays retain the previous advancement and status values.
    """

    sequence: wp.array[wp.uint64]
    """Last consumed sequence, including a sequence rejected for invalid count."""
    consumed: wp.array[int]
    """One when this begin accepted a new request prefix, otherwise zero."""
    advance: wp.array[int]
    """One only when the batch permits physical advancement; any request failure denies it."""
    status: wp.array[int]
    """Batch-level WorldStatus, independently of individual request failures."""


@wp.struct
class WorldDirectoryData:
    """Experimental read-only numeric relations owned by WorldDirectory.

    Let I be identity capacity, P prototype count, and S the sum of slot limits.
    Values are coherent after publication, outside an open transaction or move.
    Callers must not write these arrays. Physical readiness and actor participation
    belong to the domain and task; these relations establish logical lifetimes.
    """

    prototype: wp.array[int]
    """Prototype of each identity, shape [I]; -1 means dead."""
    slot: wp.array[int]
    """Prototype-local row of each identity, shape [I]; -1 means dead."""
    generation: wp.array[wp.uint64]
    """Lifetime generation of each live or dead identity, shape [I]."""
    starts: wp.array[int]
    """Prefix offsets of prepared slot partitions, shape [P + 1]."""
    slot_id: wp.array[int]
    """Inverse identity at starts[p] + local_row, shape [S]; -1 means unoccupied."""
    slot_rank: wp.array[int]
    """Inverse active-list rank at each global slot, shape [S]; -1 means unoccupied."""
    active: wp.array[int]
    """Live local rows at starts[p]:starts[p]+active_count[p], shape [S]; order unspecified."""
    active_count: wp.array[int]
    """Number of live rows per prototype, shape [P]."""
    free_count: wp.array[int]
    """Number of free admissible rows per prototype, shape [P]; excludes live rows."""
    demand: wp.array[int]
    """Requests rejected for lack of destination rows in the last consumed batch, shape [P]."""


@wp.struct
class WorldTransaction:
    """Experimental domain protocol for validation and complete initialization.

    Request arrays have command capacity; group_starts has prototype count + 1.
    The directory owns phase, destinations and grouping. During VALIDATED,
    validators may replace OK status with rejection. During ADMITTED, initializers
    acknowledge the current batch sequence only after the complete destination
    payload is written; failure leaves the acknowledgement unset. Never turn a rejection into OK.
    These arrays are transaction scratch, not current-replay activity indicators.
    """

    phase: wp.array[int]
    """WorldPhase, shape [1]; a quarantining composition root may reset it to IDLE."""
    status: wp.array[int]
    """Per-request validation/admission status; validators may reject during VALIDATED only."""
    initialized: wp.array[wp.uint64]
    """Per-request sequence acknowledged after complete destination initialization."""
    destination_id: wp.array[int]
    """Admitted destination identity by request; -1 when unavailable."""
    destination_slot: wp.array[int]
    """Admitted prototype-local destination row by request; -1 when unavailable."""
    accepted_requests: wp.array[int]
    """Admitted CREATE/RESET request ordinals, grouped by destination prototype."""
    group_starts: wp.array[int]
    """Prefix offsets into accepted_requests, shape [P + 1]; within-group order unspecified."""


@wp.struct
class WorldCompaction:
    """Experimental same-prototype disjoint move plan and domain acknowledgements.

    source/destination have total slot capacity. Prototype p uses the range
    starts[p]:starts[p]+count[p], whose entries are prototype-local rows.
    The directory owns the plan and count. During MOVING, the domain writes
    copied[p] only after all retained fields for those moves are complete.
    Plans and acknowledgements are valid only in their current lifecycle stage.
    """

    source: wp.array[int]
    """Local source rows, segmented by prepared prototype offsets."""
    destination: wp.array[int]
    """Local destination holes, segmented by prepared prototype offsets."""
    count: wp.array[int]
    """Planned moves per prototype, shape [P]."""
    copied: wp.array[int]
    """Completed domain copies per prototype, shape [P]; must equal count before publication."""


@wp.struct
class _WorldScratch:
    free_ids: wp.array[int]
    id_count: wp.array[int]
    conflicts: wp.array[int]
    slot_prototype: wp.array[int]
    slot_state: wp.array[int]
    ready: wp.array[int]
    claims: wp.array[int]
    dirty: wp.array[int]
    request_rank: wp.array[int]
    observed_generation: wp.array[wp.uint64]
    permit: wp.array[int]
    move_destinations: wp.array[int]


@wp.func
def world_location(data: WorldDirectoryData, identity: int, generation: wp.uint64):
    """Resolve a live handle to (prototype, local row, valid), without reading physical state.

    Invalid handles return (-1, -1, False). Read outside directory mutation stages.
    This validates numeric lifetime and inverse membership, not actor participation
    or domain backing. All selectors are numeric IDs, never names or paths.
    """
    if identity < 0 or identity >= data.prototype.shape[0]:
        return int(-1), int(-1), False
    prototype = data.prototype[identity]
    if prototype < 0 or prototype >= data.starts.shape[0] - 1 or data.generation[identity] != generation:
        return int(-1), int(-1), False
    row = data.slot[identity]
    start = data.starts[prototype]
    if row < 0 or row >= data.starts[prototype + 1] - start:
        return int(-1), int(-1), False
    if data.slot_id[start + row] != identity:
        return int(-1), int(-1), False
    return prototype, row, True


@wp.func
def world_handle_at(data: WorldDirectoryData, prototype: int, row: int):
    """Resolve (prototype, local row) to (identity, generation, valid).

    Invalid or unoccupied locations return (-1, uint64(0), False). Read outside
    directory mutation stages. Compaction changes locations without changing
    handles; resets change generations even when the prototype is unchanged.
    """
    if prototype < 0 or prototype >= data.starts.shape[0] - 1:
        return int(-1), wp.uint64(0), False
    start = data.starts[prototype]
    if row < 0 or row >= data.starts[prototype + 1] - start:
        return int(-1), wp.uint64(0), False
    identity = data.slot_id[start + row]
    if identity < 0 or identity >= data.prototype.shape[0]:
        return int(-1), wp.uint64(0), False
    if data.prototype[identity] != prototype or data.slot[identity] != row:
        return int(-1), wp.uint64(0), False
    return identity, data.generation[identity], True


@wp.kernel
def _begin(t: WorldTransaction, c: WorldCommands, command_capacity: int, b: WorldBatch, s: _WorldScratch):
    b.consumed[0] = 0
    if t.phase[0] != IDLE:
        b.advance[0] = 0
        b.status[0] = PHASE_INVALID
        t.phase[0] = IDLE
        return
    if c.sequence[0] == wp.uint64(0) or c.sequence[0] < b.sequence[0]:
        b.advance[0] = 0
        b.status[0] = STALE_BATCH
    elif c.sequence[0] > b.sequence[0]:
        b.sequence[0] = c.sequence[0]
        b.status[0] = 0
        if c.count[0] < 0 or c.count[0] > command_capacity:
            b.advance[0] = 0
            b.status[0] = BAD_COUNT
        else:
            b.consumed[0] = 1
            b.advance[0] = 0
            s.permit[0] = 1
            t.phase[0] = VALIDATED


@wp.kernel
def _clear_transaction(
    d: WorldDirectoryData,
    t: WorldTransaction,
    directory_capacity: int,
    command_capacity: int,
    p: int,
    b: WorldBatch,
    s: _WorldScratch,
):
    i = wp.tid()
    if b.consumed[0] != 0:
        if i < directory_capacity:
            s.conflicts[i] = 0
        if i < command_capacity:
            t.status[i] = OK
            t.initialized[i] = wp.uint64(0)
            t.destination_id[i] = -1
            t.destination_slot[i] = -1
            s.request_rank[i] = -1
        if i < p:
            d.demand[i] = 0
            s.claims[i] = 0
        if i == 0:
            s.claims[p] = 0
            s.dirty[0] = 0


@wp.kernel
def _validate(
    d: WorldDirectoryData,
    c: WorldCommands,
    t: WorldTransaction,
    directory_capacity: int,
    prototypes: int,
    b: WorldBatch,
    s: _WorldScratch,
):
    i = wp.tid()
    if b.consumed[0] == 0 or i >= c.count[0]:
        return
    op, identity, prototype = int(c.op[i]), int(c.id[i]), int(c.prototype[i])
    status = int(OK)
    s.observed_generation[i] = wp.uint64(0)
    if identity >= 0 and identity < directory_capacity:
        s.observed_generation[i] = d.generation[identity]
    if op < NOOP or op > DESTROY:
        status = INVALID
    elif op == RESET or op == DESTROY:
        if identity < 0 or identity >= directory_capacity:
            status = INVALID
        else:
            wp.atomic_add(s.conflicts, identity, 1)
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
def _admit(
    d: WorldDirectoryData,
    c: WorldCommands,
    t: WorldTransaction,
    directory_capacity: int,
    b: WorldBatch,
    s: _WorldScratch,
):
    i = wp.tid()
    if b.consumed[0] == 0 or i >= c.count[0]:
        return
    op, identity, prototype = int(c.op[i]), int(c.id[i]), int(c.prototype[i])
    status = int(t.status[i])
    if (op == RESET or op == DESTROY) and identity >= 0 and identity < directory_capacity:
        if s.conflicts[identity] > 1:
            status = CONFLICT
    if status == OK and (op == CREATE or op == RESET):
        if op == CREATE:
            ticket = wp.atomic_add(s.claims, s.claims.shape[0] - 1, 1)
            available = s.id_count[0]
            if ticket >= available:
                status = NO_IDS
            else:
                identity = int(s.free_ids[available - ticket - 1])
                t.destination_id[i] = identity
        else:
            t.destination_id[i] = identity
        if status == OK:
            ticket = wp.atomic_add(s.claims, prototype, 1)
            available = d.free_count[prototype]
            if ticket >= available:
                status = NO_SLOTS
                wp.atomic_add(d.demand, prototype, 1)
            else:
                t.destination_slot[i] = s.ready[d.starts[prototype] + available - ticket - 1]
                s.request_rank[i] = ticket
    t.status[i] = status
    if status != OK:
        wp.atomic_min(s.permit, 0, 0)


@wp.kernel
def _group_starts(d: WorldDirectoryData, t: WorldTransaction, prototypes: int, b: WorldBatch, s: _WorldScratch):
    if b.consumed[0] == 0:
        return
    # The tiny prototype prefix is O(P); never walk requests serially.
    total = int(0)
    t.group_starts[0] = 0
    for prototype in range(prototypes):
        total += wp.min(s.claims[prototype], d.free_count[prototype])
        t.group_starts[prototype + 1] = total


@wp.kernel
def _group_requests(c: WorldCommands, t: WorldTransaction, b: WorldBatch, s: _WorldScratch):
    request = wp.tid()
    if b.consumed[0] != 0 and request < c.count[0]:
        rank = s.request_rank[request]
        if rank >= 0:
            t.accepted_requests[t.group_starts[c.prototype[request]] + rank] = request


@wp.kernel
def _publish(
    d: WorldDirectoryData, c: WorldCommands, t: WorldTransaction, r: WorldResults, b: WorldBatch, s: _WorldScratch
):
    i = wp.tid()
    if b.consumed[0] == 0 or i >= c.count[0]:
        return
    if t.phase[0] != ADMITTED:
        return
    status, op, identity = int(t.status[i]), int(c.op[i]), int(c.id[i])
    if status == OK and (op == CREATE or op == RESET) and t.initialized[i] != b.sequence[0]:
        status = INITIALIZATION_MISSING
        t.status[i] = status
        wp.atomic_min(s.permit, 0, 0)
    if status == OK and op != NOOP:
        wp.atomic_max(s.dirty, 0, 1)
        if op == CREATE:
            identity = int(t.destination_id[i])
        if op == RESET or op == DESTROY:
            source = d.starts[d.prototype[identity]] + d.slot[identity]
            s.slot_state[source] = READY
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
            s.slot_state[destination] = LIVE
            d.slot_id[destination] = identity
    r.status[i] = status
    r.id[i] = identity
    if status == OK and op != NOOP:
        r.generation[i] = d.generation[identity]
    else:
        r.generation[i] = s.observed_generation[i]


@wp.kernel
def _clear_membership(d: WorldDirectoryData, prototypes: int, conditional: int, b: WorldBatch, s: _WorldScratch):
    i = wp.tid()
    if conditional == 0 or (b.consumed[0] != 0 and s.dirty[0] != 0):
        if i < prototypes:
            d.active_count[i] = 0
            d.free_count[i] = 0
        if i == 0:
            s.id_count[0] = 0


@wp.kernel
def _rebuild(
    d: WorldDirectoryData,
    directory_capacity: int,
    slot_capacity: int,
    conditional: int,
    b: WorldBatch,
    s: _WorldScratch,
):
    i = wp.tid()
    if conditional != 0 and (b.consumed[0] == 0 or s.dirty[0] == 0):
        return
    if i < directory_capacity:
        if d.prototype[i] < 0 and d.generation[i] < wp.uint64(18446744073709551615):
            rank = wp.atomic_add(s.id_count, 0, 1)
            s.free_ids[rank] = i
    if i < slot_capacity:
        prototype = s.slot_prototype[i]
        local_slot = i - d.starts[prototype]
        if s.slot_state[i] == LIVE:
            rank = wp.atomic_add(d.active_count, prototype, 1)
            d.active[d.starts[prototype] + rank] = local_slot
            d.slot_rank[i] = rank
        elif s.slot_state[i] == READY:
            rank = wp.atomic_add(d.free_count, prototype, 1)
            s.ready[d.starts[prototype] + rank] = local_slot
            d.slot_rank[i] = -1


@wp.kernel
def _clear_compaction(moves: WorldCompaction, s: _WorldScratch):
    prototype = wp.tid()
    moves.count[prototype] = 0
    s.move_destinations[prototype] = 0
    moves.copied[prototype] = 0


@wp.kernel
def _plan_compaction(d: WorldDirectoryData, t: WorldTransaction, moves: WorldCompaction, s: _WorldScratch):
    index = wp.tid()
    if t.phase[0] != MOVING:
        return
    prototype = s.slot_prototype[index]
    start = d.starts[prototype]
    slot = index - start
    count = d.active_count[prototype]
    if slot < count and s.slot_state[index] == READY:
        rank = wp.atomic_add(s.move_destinations, prototype, 1)
        moves.destination[start + rank] = slot
    elif slot >= count and s.slot_state[index] == LIVE:
        rank = wp.atomic_add(moves.count, prototype, 1)
        moves.source[start + rank] = slot


@wp.kernel
def _validate_compaction(
    t: WorldTransaction, moves: WorldCompaction, prototypes: int, require_copy: int, b: WorldBatch, s: _WorldScratch
):
    if t.phase[0] != MOVING:
        if require_copy != 0:
            b.advance[0] = 0
            b.status[0] = PHASE_INVALID
        return
    complete = int(1)
    for prototype in range(prototypes):
        if moves.count[prototype] != s.move_destinations[prototype]:
            complete = 0
        if require_copy != 0 and moves.copied[prototype] != moves.count[prototype]:
            complete = 0
    if complete == 0:
        b.status[0] = COMPACTION_INVALID
        s.permit[0] = 0
        t.phase[0] = IDLE
    elif require_copy != 0:
        t.phase[0] = COPIED


@wp.kernel
def _publish_compaction(d: WorldDirectoryData, t: WorldTransaction, moves: WorldCompaction, s: _WorldScratch):
    index = wp.tid()
    if t.phase[0] != COPIED:
        return
    prototype = s.slot_prototype[index]
    start = d.starts[prototype]
    rank = index - start
    if rank < moves.count[prototype]:
        source, destination = moves.source[start + rank], moves.destination[start + rank]
        identity = d.slot_id[start + source]
        s.slot_state[start + destination] = LIVE
        d.slot_id[start + destination] = identity
        d.slot[identity] = destination
        s.slot_state[start + source] = READY
        d.slot_id[start + source] = -1
        d.slot_rank[start + source] = -1


@wp.kernel
def _check_shrink(d: WorldDirectoryData, ends: wp.array[int], failure: wp.array[int], s: _WorldScratch):
    slot = wp.tid()
    prototype = s.slot_prototype[slot]
    if slot - d.starts[prototype] >= ends[prototype] and s.slot_state[slot] == LIVE:
        wp.atomic_max(failure, 0, 1)


@wp.kernel
def _publish_capacity(d: WorldDirectoryData, ends: wp.array[int], s: _WorldScratch):
    slot = wp.tid()
    prototype = s.slot_prototype[slot]
    if slot - d.starts[prototype] < ends[prototype] and s.slot_state[slot] != LIVE:
        s.slot_state[slot] = READY
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
def _transition(t: WorldTransaction, c: WorldCommands, expected: int, target: int, b: WorldBatch, s: _WorldScratch):
    if b.consumed[0] == 0:
        return
    if t.phase[0] != expected or c.sequence[0] != b.sequence[0]:
        b.advance[0] = 0
        b.status[0] = PHASE_INVALID
        s.permit[0] = 0
        t.phase[0] = IDLE
        b.consumed[0] = 0
    else:
        t.phase[0] = target


@wp.kernel
def _finish(t: WorldTransaction, b: WorldBatch, s: _WorldScratch):
    if b.consumed[0] != 0 and t.phase[0] == ADMITTED:
        t.phase[0] = IDLE
        b.advance[0] = s.permit[0]


@wp.kernel
def _begin_moves(t: WorldTransaction, b: WorldBatch, s: _WorldScratch):
    if t.phase[0] != IDLE:
        b.status[0] = PHASE_INVALID
        s.permit[0] = 0
        t.phase[0] = IDLE
    else:
        s.permit[0] = b.advance[0]
        t.phase[0] = MOVING
    b.advance[0] = 0


@wp.kernel
def _finish_moves(t: WorldTransaction, b: WorldBatch, s: _WorldScratch):
    if t.phase[0] == COPIED:
        t.phase[0] = IDLE
        b.advance[0] = s.permit[0]


@wp.kernel
def _withdraw_ready(d: WorldDirectoryData, ends: wp.array[int], s: _WorldScratch):
    slot = wp.tid()
    prototype = s.slot_prototype[slot]
    if slot - d.starts[prototype] >= ends[prototype] and s.slot_state[slot] == READY:
        s.slot_state[slot] = UNBACKED
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

    data: WorldDirectoryData
    """Borrowed numeric relation view; only directory operations mutate it."""
    batch: WorldBatch
    """Named outcome of the current batch/replay."""
    transaction: WorldTransaction
    """Domain validation and initialization protocol for the current transaction."""
    compaction: WorldCompaction
    """Domain relocation plan and copy acknowledgements for the current move phase."""

    def __init__(
        self, slot_limits: tuple[int, ...], *, id_capacity: int = 4096, command_capacity: int = 4096, device=None
    ):
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
        self.data, self.batch = WorldDirectoryData(), WorldBatch()
        self.transaction, self.compaction, self._scratch = WorldTransaction(), WorldCompaction(), _WorldScratch()
        d, b, t, moves, s = self.data, self.batch, self.transaction, self.compaction, self._scratch
        prototypes = len(slot_limits)
        for name in ("prototype", "slot"):
            setattr(d, name, wp.full(id_capacity, -1, dtype=int, device=self.device))
        d.generation = wp.zeros(id_capacity, dtype=wp.uint64, device=self.device)
        starts = np.concatenate(([0], np.cumsum(slot_limits))).astype(np.int32)
        d.starts = wp.array(starts, dtype=int, device=self.device)
        for name in ("slot_id", "slot_rank"):
            setattr(d, name, wp.full(self.slot_capacity, -1, dtype=int, device=self.device))
        d.active = wp.zeros(self.slot_capacity, dtype=int, device=self.device)
        for name in ("active_count", "free_count", "demand"):
            setattr(d, name, wp.zeros(prototypes, dtype=int, device=self.device))
        b.sequence = wp.zeros(1, dtype=wp.uint64, device=self.device)
        for name in ("consumed", "advance", "status"):
            setattr(b, name, wp.zeros(1, dtype=int, device=self.device))
        for name in ("status", "destination_id", "destination_slot", "accepted_requests"):
            setattr(t, name, wp.zeros(command_capacity, dtype=int, device=self.device))
        t.initialized = wp.zeros(command_capacity, dtype=wp.uint64, device=self.device)
        t.group_starts = wp.zeros(prototypes + 1, dtype=int, device=self.device)
        t.phase = wp.zeros(1, dtype=int, device=self.device)
        for name in ("source", "destination"):
            setattr(moves, name, wp.zeros(self.slot_capacity, dtype=int, device=self.device))
        for name in ("count", "copied"):
            setattr(moves, name, wp.zeros(prototypes, dtype=int, device=self.device))
        s.free_ids = wp.zeros(id_capacity, dtype=int, device=self.device)
        s.id_count = wp.zeros(1, dtype=int, device=self.device)
        s.conflicts = wp.zeros(id_capacity, dtype=int, device=self.device)
        s.slot_prototype = wp.array(
            np.repeat(np.arange(prototypes), slot_limits).astype(np.int32), dtype=int, device=self.device
        )
        s.slot_state = wp.full(self.slot_capacity, UNBACKED, dtype=int, device=self.device)
        s.ready = wp.zeros(self.slot_capacity, dtype=int, device=self.device)
        s.claims = wp.zeros(prototypes + 1, dtype=int, device=self.device)
        for name in ("dirty", "permit"):
            setattr(s, name, wp.zeros(1, dtype=int, device=self.device))
        s.request_rank = wp.zeros(command_capacity, dtype=int, device=self.device)
        s.observed_generation = wp.zeros(command_capacity, dtype=wp.uint64, device=self.device)
        s.move_destinations = wp.zeros(prototypes, dtype=int, device=self.device)
        self._capacity_failure = wp.zeros(1, dtype=int, device=self.device)
        self._capacity_ends = wp.zeros(prototypes, dtype=int, device=self.device)
        self._rebuild(conditional=False)

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("WorldDirectory is closed")

    def _rebuild(self, *, conditional):
        data, batch, scratch = self.data, self.batch, self._scratch
        prototypes = len(self.slot_limits)
        wp.launch(
            _clear_membership,
            max(1, prototypes),
            [data, prototypes, int(conditional), batch, scratch],
            device=self.device,
        )
        wp.launch(
            _rebuild,
            max(self.id_capacity, self.slot_capacity),
            [data, self.id_capacity, self.slot_capacity, int(conditional), batch, scratch],
            device=self.device,
        )

    def begin(self, commands: WorldCommands):
        """Record batch validation; domain payload validation may follow before admission."""
        data, batch, transaction, scratch = self.data, self.batch, self.transaction, self._scratch
        self._ensure_open()
        _validate_arrays(commands, WorldCommands, self.command_capacity, self.device, ("sequence", "count"))
        prototypes = len(self.slot_limits)
        wp.launch(
            _begin,
            1,
            [transaction, commands, self.command_capacity, batch, scratch],
            device=self.device,
        )
        wp.launch(
            _clear_transaction,
            max(self.id_capacity, self.command_capacity, prototypes),
            [
                data,
                transaction,
                self.id_capacity,
                self.command_capacity,
                prototypes,
                batch,
                scratch,
            ],
            device=self.device,
        )
        wp.launch(
            _validate,
            self.command_capacity,
            [data, commands, transaction, self.id_capacity, prototypes, batch, scratch],
            device=self.device,
        )

    def admit(self, commands: WorldCommands):
        """Assign destination tickets without publishing or altering old lifetimes."""
        data, batch, transaction, scratch = self.data, self.batch, self.transaction, self._scratch
        self._ensure_open()
        _validate_arrays(commands, WorldCommands, self.command_capacity, self.device, ("sequence", "count"))
        wp.launch(
            _transition,
            1,
            [transaction, commands, VALIDATED, ADMITTED, batch, scratch],
            device=self.device,
        )
        wp.launch(
            _admit,
            self.command_capacity,
            [data, commands, transaction, self.id_capacity, batch, scratch],
            device=self.device,
        )
        wp.launch(
            _group_starts,
            1,
            [data, transaction, len(self.slot_limits), batch, scratch],
            device=self.device,
        )
        wp.launch(
            _group_requests,
            self.command_capacity,
            [commands, transaction, batch, scratch],
            device=self.device,
        )

    def publish(self, commands: WorldCommands, results: WorldResults):
        """Publish successfully initialized requests and gate advancement on batch errors."""
        data, batch, transaction, scratch = self.data, self.batch, self.transaction, self._scratch
        self._ensure_open()
        _validate_arrays(commands, WorldCommands, self.command_capacity, self.device, ("sequence", "count"))
        _validate_arrays(results, WorldResults, self.command_capacity, self.device)
        wp.launch(
            _transition,
            1,
            [transaction, commands, ADMITTED, ADMITTED, batch, scratch],
            device=self.device,
        )
        wp.launch(
            _publish,
            self.command_capacity,
            [data, commands, transaction, results, batch, scratch],
            device=self.device,
        )
        self._rebuild(conditional=True)
        wp.launch(_finish, 1, [transaction, batch, scratch], device=self.device)

    def plan_moves(self):
        """Plan nonoverlapping retained-row moves into each prototype's live prefix."""
        data, batch, transaction, moves, scratch = (
            self.data,
            self.batch,
            self.transaction,
            self.compaction,
            self._scratch,
        )
        self._ensure_open()
        wp.launch(_begin_moves, 1, [transaction, batch, scratch], device=self.device)
        wp.launch(_clear_compaction, len(self.slot_limits), [moves, scratch], device=self.device)
        wp.launch(
            _plan_compaction,
            self.slot_capacity,
            [data, transaction, moves, scratch],
            device=self.device,
        )
        wp.launch(
            _validate_compaction,
            1,
            [transaction, moves, len(self.slot_limits), 0, batch, scratch],
            device=self.device,
        )

    def publish_moves(self):
        """Update membership only after the domain acknowledges every planned copy."""
        data, batch, transaction, moves, scratch = (
            self.data,
            self.batch,
            self.transaction,
            self.compaction,
            self._scratch,
        )
        self._ensure_open()
        wp.launch(
            _validate_compaction,
            1,
            [transaction, moves, len(self.slot_limits), 1, batch, scratch],
            device=self.device,
        )
        wp.launch(
            _publish_compaction,
            self.slot_capacity,
            [data, transaction, moves, scratch],
            device=self.device,
        )
        self._rebuild(conditional=False)
        wp.launch(_finish_moves, 1, [transaction, batch, scratch], device=self.device)

    def _capacity_args(self, ends):
        self._ensure_open()
        ends = tuple(ends)
        if len(ends) != len(self.slot_limits) or any(
            type(end) is not int or not 0 <= end <= capacity
            for end, capacity in zip(ends, self.slot_limits, strict=True)
        ):
            raise ValueError("Ready prefix is outside the prepared slot limits")
        if self.transaction.phase.numpy()[0] != IDLE:
            raise RuntimeError("Capacity service cannot interrupt an open transaction")
        self._capacity_ends.assign(np.asarray(ends, dtype=np.int32))

    def withdraw_ready(self, ends: tuple[int, ...]):
        """Withdraw every requested tail before storage retirement; reject any live tail.

        This cold operation validates the entire batch before changing admission.
        The domain must then join consumers before unmapping physical storage.
        """
        self._capacity_args(ends)
        self._capacity_failure.zero_()
        wp.launch(
            _check_shrink,
            self.slot_capacity,
            [self.data, self._capacity_ends, self._capacity_failure, self._scratch],
            device=self.device,
        )
        if self._capacity_failure.numpy()[0]:
            raise RuntimeError("Cannot withdraw a range containing live worlds; compact or destroy them first")
        wp.launch(
            _withdraw_ready, self.slot_capacity, [self.data, self._capacity_ends, self._scratch], device=self.device
        )
        self._rebuild(conditional=False)

    def publish_ready(self, ends: tuple[int, ...]):
        """Enable certified usable prefixes without withdrawing any previously enabled tail.

        The domain must complete physical backing/initialization before this call.
        For shrink, call withdraw_ready before retiring physical rows. Exclude
        concurrent submissions during either cold capacity-service operation.
        """
        self._capacity_args(ends)
        wp.launch(
            _publish_capacity, self.slot_capacity, [self.data, self._capacity_ends, self._scratch], device=self.device
        )
        self._rebuild(conditional=False)

    def retain_graph(self, graph, *buffers):
        """Keep the directory and explicit external buffers alive with one captured graph."""
        self._ensure_open()
        if graph.device != self.device:
            raise ValueError("Graph and directory must use the same device")
        graph.world_owners = (*getattr(graph, "world_owners", ()), self, *buffers)
        self._graphs.add(graph)
        return graph

    def close(self, *, streams: tuple[wp.Stream, ...]):
        """Retire submission rights after joining the supplied same-device Warp streams.

        Destroy all retained graphs first and exclude new submissions during close.
        CUDA callers must supply every consumer stream; raw handles are rejected.
        CPU callers may pass an empty tuple. Python references retain the metadata
        arrays themselves after closure; this method does not revoke borrowed views.
        """
        if self._graphs:
            raise RuntimeError("Destroy retained graphs before closing their directory")
        streams = tuple(streams)
        if self.device.is_cuda and not streams:
            raise ValueError("CUDA directory closure requires explicit consumer streams")
        if any(not isinstance(stream, wp.Stream) or stream.device != self.device for stream in streams):
            raise ValueError("Directory closure requires same-device Warp Stream objects")
        for stream in streams:
            wp.synchronize_stream(stream)
        self._closed = True

    def memory_report(self):
        """Report owned directory metadata, excluding every domain and graph allocation."""
        return {
            "directory_metadata_bytes": sum(
                getattr(record, name).capacity
                for record in (self.data, self.batch, self.transaction, self.compaction, self._scratch)
                for name in record._cls.vars
            )
            + self._capacity_failure.capacity
            + self._capacity_ends.capacity
        }
