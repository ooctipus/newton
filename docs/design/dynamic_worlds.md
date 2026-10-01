<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Growable world populations

`newton.solvers.MuJoCoWorlds` runs changing populations of prepared native world
prototypes. A keyboard reset can move a logical world from a six-key prototype to
a 108-key prototype while continuing worlds keep their physical state and solver
history. Preparation fixes topology, geometry, inertia and the CUDA program;
GPU commands change instance membership and the work counts used by that program.
This is an experimental implementation, with the admitted native features below.

## Level 0 The mechanism

```python
# Conceptual lifecycle; no model construction in the replay path.
prepare_each_native_prototype_once()
reserve_stable_virtual_arrays()
back_initial_rows_and_spare_capacity()
graph = capture_lifecycle_and_physics_once()

while running:
    write_gpu_create_reset_destroy_requests()
    replay(graph)  # initialize, publish, compact, run live native worlds

# At an explicitly joined cold boundary, when physical capacity needs to change:
map_or_unmap_backing_then_publish_ready_rows()
```

```text
Immutable native prototypes        GPU directory             Shared backing
6 keys / 108 keys / ...             ID + generation           physical budget
         |                               |                         |
         +---- initialize destination <--+---- ready rows --------+
         |                               |
         +---- copy retained state ------+---- publish membership
         |                               |
         +---- prepared native program <-+---- live work counts
```

The array's virtual base stays fixed. Physical backing may be returned and mapped
again only after all readers join. Moving a world between prototypes changes its
row and generation, not the pointer recorded for either prototype's arrays.
Compaction can move continuing rows without changing their identities or
generations. A failed replacement preserves the old lifetime and blocks physical
advancement; ready capacity and live population are separate quantities.

## Level 1 Ownership and the composition root

The boundary standard is the cloner's composition algebra: asset prototypes,
ordered asset occurrences inside world prototypes, and instances of those world
prototypes. A relation taking `prototype_id: int | str` crosses two boundaries.
A path utility must first resolve names to integer IDs; the numeric relation then
consumes those IDs without interpreting names. Repeated asset occurrences remain
distinct even when they reference the same asset prototype.

For example, the user's Banana/Franka plan has asset prototype IDs `0, 1`,
world contents `[0,1, 0,1,1, 0,0,1, 1]` and offsets `[0,2,5,8,9]`.
World prototype 1 contains two distinct Franka occurrences. A path utility maps
`"Franka"` to asset prototype 1; numeric occurrence queries preserve both
occurrences, while an explicitly unique-world query returns that world once.
Accepting `1 | "Franka"` in the relation itself would mix name resolution with
the algebra. That is the boundary breach this implementation must prevent.

The task's concrete equivalent is preparation followed by numeric binding:

```python
# Before: the numeric owner also interpreted a symbolic selector.
# selection = selections.resolve(NewtonSelectorCfg(BODY, ".*/Robot.*"))

# After: separate symbolic query and numeric relation; managers retain selection.
query = NewtonSelectorCfg(BODY, ".*/Robot.*")
ids = query_selection_indices(selections.model, query)  # selection_paths.py
selection = selections.bind(BODY, ids)                # newton_selection.py
```

The authored environment configuration retains `query`; the manager's copy owns
the bound `selection`. Numeric bindings neither serialize paths nor re-run name
resolution. `world_selection_counts` and `prototype_selection_counts` explicitly identify
their different axes. The bound selection's `width` (configured by `policy_width`)
and current `active_counts()` describe other
relations. Equal count tuples do not establish matching owners or ordered joints;
MDP preparation validates those relationships before pairing coordinates and DOFs.

An index is meaningful only inside its domain. A free joint has seven coordinates
but six velocity DOFs: coordinate 6 and DOF 6 can refer to different joints. The
selection therefore checks each field against Newton's authoritative attribute
frequency before binding it. It does not infer compatibility from array length,
scalar dtype or matching integer values. The prepared model's schema also decides
whether `joint_target_q` uses coordinates or DOFs; a later process-wide default
cannot change the meaning of an existing model.

```python
coordinates = selections.bind(JOINT_COORD, coordinate_indices)
velocities = selections.bind(JOINT_DOF, dof_indices)
q = coordinates.read_state("joint_q")
qd = velocities.read_state("joint_qd")
# coordinates.read_state("joint_qd") raises before gathering any values.
```

The same rule applies here. A world lifetime `(id, generation)` maps to exactly
one `(prototype, local_row)`; the inverse returns that lifetime. An environment's
participation mask is a task fact, not part of lifetime validity. Backing readiness
is a storage fact, not a second definition of identity. Each owner exposes the
relations necessary to compose these facts, without importing the other owner's
policy or duplicating its authority.

The implementation boundary is:

```text
Authoring/path utility  -> integer prototype-local entity selections
WorldDirectory         -> lifetime, placement, membership and transactions
MuJoCoWorlds           -> native physics composition and prepared recording
FieldStorage/MemoryBacking -> typed rows / physical allocation and retirement
Task                   -> environment-to-handle bindings, participation and episode intent
```

The standalone directory separates numeric membership (`data`), named batch
outcomes (`batch_result`), domain initialization (`transaction`) and compaction
(`compaction`). The physics composition root exposes `directory` as the borrowed
`WorldDirectoryData` and `batch_result` separately; its mutable directory owner
is private. Last-batch `slot_rejection_count` belongs to the batch result, not
the membership relation or a desired-population controller.
Allocation tickets, conflict counters and rebuild scratch are private. The native
root exposes a borrowed population recording interface, not its private storage
and graph-owner record. Task selection binding accepts integer IDs; path matching
is a separate preparation utility. Native scalar conversion maps are prepared
once by the scalar-task owner and borrowed by both selection and reset payloads.

Completeness is scoped to a prepared prototype set. For every live handle,
`world_handle_at(data, *world_location(data, id, generation)[:2])` returns the
same ID and generation. Prototype memberships partition the live population;
compaction preserves handles and state; successful reset invalidates the old
generation; failed replacement preserves it. The Banana/Franka occurrence and
unique-world queries above are the architectural reference, not additional APIs
implemented by this subsystem. Every supported array states its index domain and writer, and every
operation states when its results are valid. These laws are architecture and
behavior gates; passing a keyboard benchmark alone does not establish them.

The file tree below remains the owner map. This cleanup adds no backend factory,
compatibility aliases, generic query manager or second lifetime directory.

The production composition root is `newton.solvers.MuJoCoWorlds`. It composes
the experimental `newton.worlds` contracts: `WorldDirectory`, `FieldStorage`,
`MemoryBacking` and explicit graph bindings. Public namespace grouping does not
merge the implementation owners or their authorities. A new consumer can use
these contracts without importing private modules. Native Data is authoritative:
there is no replicated Newton State mirror, solver adapter or second lifecycle owner.

```text
Caller                         owns actions, reset decisions, request/output buffers
  |
  v
MuJoCoWorlds                   concrete physics composition and preparation
  +-> WorldDirectory          identities, admission, membership, relocation plans
  +-> prepared populations    immutable topology + authoritative mutable Data
  |    +-> FieldStorage W       world state and per-world solver scratch
  |    +-> FieldStorage C       contacts and broadphase candidate scratch
  |    +-> FieldStorage D       CCD/EPA scratch
  |    +-> graph bindings     prepared program; GPU work/count inputs
  +-> MemoryBacking             one budget; reservations, handles, mappings, retirement
```

`WorldDirectory` does not import a solver, allocator, Torch or IsaacLab. It plans
changes and requires explicit initialization/copy acknowledgements before
publishing them. It never copies physical state. `FieldStorage` knows field layouts
and typed row operations; it does not know native Data, world identities, default
values or physics policy. `MemoryBacking` knows bytes, mappings and completion
dependencies. The concrete native root supplies the missing semantics and owns
the execution order. Other solvers can compose these same mechanisms without
inheriting a MuJoCo-shaped generic backend interface.

Existing one-world models and builder plans remain the source of geometry,
inertia, composition and names. Prepared native models own their lowered
representation, charged separately. Selection/query manifests are deliberately
left to callers; they cannot become a second importer or lifetime authority.

The native solver declares field layouts and complete initialization rules.
The CUDA owner knows bytes, mappings and completion dependencies, not keyboards,
world IDs or reset policy. Keeping this owner separate is necessary because views
and in-flight graphs can retain backing after the pool itself is no longer used.

```python
import warp as wp
from newton.solvers import MuJoCoWorlds
from newton.worlds import create_world_commands, create_world_results

# Each pair is an immutable MJWarp Model and one-world default Data.
# Warm the admitted native program on separate Data before capture.
worlds = MuJoCoWorlds(
    prepared_model_data_pairs,
    world_capacities=virtual_slot_limits,
    id_capacity=max_live_worlds,
    command_capacity=max_requests,
    contact_capacities=contact_limits,
    ccd_capacities=ccd_limits,
    memory_budget_bytes=budget,
    initial_world_ready_capacities=initial_ready_capacities,
)
commands = create_world_commands(max_requests, device=worlds.device)
results = create_world_results(max_requests, device=worlds.device)
# Application callbacks record payload/control kernels using the contracts below.
graph = worlds.capture(
    commands, results, substeps=2, initialize=record_reset_payload,
    before_step=record_controls, retain=(reset_payload, control_inputs),
)
# Caller kernels write commands, reset payloads and controls before replay.
wp.capture_launch(graph)

# Cold capacity service joins all readers before changing physical mappings.
worlds.resize_backing(tuple(ready_rows_per_prototype), streams=(stream,))
# Drop every graph borrower before closing storage.
del graph
worlds.close(streams=(stream,))
```

`memory_budget_bytes=None` selects fixed backing for the full prepared capacities.
With VMM, `initial_world_ready_capacities=None` also backs every prototype to its full capacity. Set
both the physical budget and explicit initial ready capacities (including zero for unused
prototypes) to start with a smaller physical population. Contact/CCD limits default
to each one-world template quota multiplied by that prototype's virtual capacity.

The experimental native runtime requires Python 3.11+, CUDA, and the pinned
MJWarp prepared-workspace API. Generic world directory records remain usable
without importing a native engine. CUDA graph preparation compiles one small
bridge into the Warp cache, outside the package source; `CUDACXX`/`CUDA_HOME`
select the toolkit, or `NEWTON_CUDA_GRAPH_LIBRARY` selects a prepared library.

`capture` has preparation-only hooks with caller-owned payload buffers. These
hooks record allocation-free GPU operations using storage prepared before capture.
`validate` and `initialize` record transaction work, not per-frame callbacks:
healthy equal-sequence replays may skip that work. Initialization/move counts,
request lists and acknowledgements are transaction scratch, not current-frame
activity signals; consume them only within their recorded lifecycle stage.

- `validate(commands, request_status, consumed)` rejects invalid payloads before
  admission. Check `consumed` before reading the command prefix; change only OK
  requests to rejection. The callback does not receive mutable placement authority.
- `initialize(group, requests, destinations, count, base_status, initialized_sequence,
  sequence)` overrides full default rows, then writes the supplied per-request
  acknowledgement. It does not receive the rest of the transaction owner.
- `before_step(group)` records controls once before the ordered native substeps.
- `after_substep(group)` records contact consumers and diagnostics after each step.
- `retain=(...)` keeps all external arrays used by those recorded kernels alive.

`group` is a supported `newton.solvers.MuJoCoWorldPopulation` borrowed view. Its
integer `prototype_index` identifies the prepared topology; `model` and `data`
expose the native schema. Live counts, certified ready prefixes and reserved capacities have
separate attributes. Application physics callbacks use `group.record_launch(...)`
to launch and bind a kernel's count semantics together. Private field stores,
transfers, updater bindings and retirement methods are not part of this view.
Invalid application count declarations fail before the launch is emitted. A
failure after emission makes that prepared program unusable even if a callback
catches the exception; a partially bound graph must never be published.
Initializer kernels instead use their explicit admitted request/destination list;
they must not treat a newly admitted row as a published live row.

`permit` is a device scalar that suppresses physical advancement without suppressing
valid resets. `refresh_kinematics=True` updates body, geometry and site poses after
a reset-only frame or the final native substep. Any failed request gates both
advancement and pose refresh; there is no optional-request mode. Per-prototype
branches update their own poses before the common graph join. Setting
`refresh_kinematics=False` is explicit and useful for physics-only measurements;
derived transforms then describe the last refresh.

The `MuJoCoWorlds.directory` property exposes numeric observations without the
mutating directory methods. Submit lifecycle commands through the captured graph
and change backing through `resize_backing`. Borrowed Warp arrays are read-only
by contract; consumers must not write membership arrays independently. Relation
and batch diagnostics remain readable after quarantine or closure; observation
does not grant submission rights or certify physical readiness. A standalone
`WorldDirectory` remains available to other domain composition roots that own the
complete initialization, relocation and publication sequence.

The directory is the only lifetime authority. `WorldCommands` carries operation,
identity, expected generation and target prototype; `WorldResults` carries status
and resulting identity/generation. A caller must treat the directory's advancement
permit as authoritative after a partly successful batch. The root only publishes
rows after default/payload initialization or complete retained Data relocation has
acknowledged completion. Current replacement admission needs a spare destination
row, including same-prototype reset; this temporary demand is part of the backing
budget and is not hidden by the API.

Preparation lowers topology, collision exclusions and shared physical defaults
once per prototype. Model parameters have one shared broadcast row; slot-dependent
parameter batches are rejected before allocation because compaction must not change
a continuing world's physical definition. It must not first allocate a maximum-sized dense population
and then repack it. The native layout is planned from a real one-world template;
W/C/D capacities select virtual extents and physical readiness independently.
Graph addresses and typed descriptors remain stable. A shared budget backs the
sum of ready ranges, rather than every prototype's full virtual capacity.

Persistent state and caller inputs belong to W. The correctness baseline copies
every declared native Data row during compaction: nominally derived arrays can
contain structural zeros that a later kernel only partly overwrites. A smaller
copy set requires a first-read/initialization proof, not only a list of intuitive
state fields. Contact/solver scratch has explicit execution lifetimes. Consumers
must finish before it is reused or its backing is withdrawn.

## Level 2 Source ownership

```text
newton/worlds.py                         canonical generic public API
newton/_src/sim/worlds.py                directory, records and lifecycle kernels
newton/_src/utils/cuda_vmm.py            bytes, VA, handles, maps, retirement fences
newton/_src/utils/field_storage.py       typed field layout, readiness and transfers
newton/_src/utils/cuda_graph.py + .cu     prepared CUDA node bindings and fork/join
newton/_src/solvers/mujoco/worlds.py      native schema and physics composition
newton/tests/test_worlds.py              generic semantics and architecture gates
newton/tests/test_cuda_vmm.py            failure-injected mechanical ownership
newton/tests/test_field_storage.py       typed transfer, lifetime and alignment gates
newton/tests/test_cuda_graph.py          independent counts, CUDA grid and graph gates
newton/tests/test_mujoco_worlds.py        independent dense physics and pose oracle
newton/tests/test_world_runtime_composition.py  independent generated-work consumer
```

Documentation has its own composition root: `docs/conf.py` loads
`docs/_ext/autodoc_warp.py`. It reads Warp's underlying Python definitions so public
record fields retain their source annotations and descriptions. It neither copies
the field schema nor changes Warp metadata; runtime modules never import it.
Documentation gates require supported owner operations and record fields to remain
visible in the generated reference.

The native root uses the existing one-world preparation path. It does not derive
from `SolverBase`: a changing native population has no matching dense Newton
`State`, and exposing a fabricated state would hide its real ownership. The
existing homogeneous solver remains unchanged. There is no backend factory,
compatibility alias, source-code inspection, global Warp launch interception or
second lifecycle map.

## Level 3 A complete transaction

The public directory records are `WorldDirectoryData`, `WorldBatchResult`,
`WorldTransaction` and `WorldCompaction`, accessed as `data`, `batch_result`,
`transaction` and `compaction`. Private allocation scratch is not exported.
Caller-owned `WorldCommands` contains a monotonically increasing
batch sequence, operation, identity, expected generation and target prototype.
`WorldResults` returns status, identity and generation. Operation/status/phase
values have canonical public enums in `newton.worlds`. `CREATE` ignores input
identity/generation and returns an allocated handle; `RESET` and `DESTROY` validate
the supplied handle. Reset keeps the identity and increments its generation, even
within the same prototype. Destroy retires that identity and increments its
generation unless it is already `UINT64_MAX`; terminal destruction leaves a dead,
nonreusable tombstone at that generation. Generations never wrap. Results are
indexed by request ordinal, not world ID; accept a new handle only from a
successful create/reset result.

Start with a positive sequence. A new sequence consumes the batch even if some
requests fail; retry only the rejected requests with a higher sequence. Equal-sequence replays retain the
previous permit and may advance physics again, but ignore lifecycle-buffer edits.
A stale sequence or invalid batch count may leave result arrays unchanged. Check
`batch_result.consumed`, `batch_result.status` (batch error) and
`batch_result.advance_allowed` (advancement permit), not only per-request status. Results beyond the consumed
request count are not refreshed.

```text
begin -> caller validation -> admit -> full defaults + caller payload -> publish
                                                                |
                          per-request initialization acknowledgement

plan_compaction -> complete Data copies -> acknowledge -> publish_compaction

health + ready capacity + transaction permit -> native step -> pose refresh
```

Admission reserves destination tickets without exposing incomplete rows. Failed
requests do not invalidate independent successful requests, but any failed
request disables the batch's physics permit. Admission uses the pre-batch free
IDs and rows; resources retired by this batch become available to later batches.
Resource contention does not promise request-order winners. Replaying the same
accepted sequence does not apply the lifecycle command again. Reused IDs increment
their generation. Stale references and resets at generation exhaustion are
rejected; terminal identities remain destroyable but cannot be reused.

Compaction copies complete native Data rows in the correctness baseline. It does
not copy transient contact/CCD queues. Lifecycle publication invalidates their
counts before consumers can confuse a reused local slot with its old lifetime.
Task contact caches also carry the world generation; continuing worlds preserve
their cached forces across compaction. Structural zeros in partly written fields
are part of initialization: reducing relocation to intuitive position/velocity
fields failed a VMM regrowth test and is intentionally not the implementation.

The composition root records all GPU count updaters, then checks every updater's
error buffer before computing any prototype's execution conditions. An error
latches the existing runtime health and named directory batch error, suppressing all
physics, pose refresh and later raw replays without a host readback. Prototype
recording must not launch its own updater or validate only its local error buffer:
a healthy prototype must not run concurrently with an unchecked failing updater.

Native steps of different prototypes form independent branches of one graph.
Each complete branch encloses its physics permit and optional pose refresh under
one outer conditional. Preparation checks the parent DAG and rejects accidental
sibling dependencies after Warp conditional resume. All branches join before
replay completion. This removes host iteration
over separate executable graphs. CUDA node scalar arguments and launch extents
have explicit independent W/C/D count sources. Fixed worker grids remain fixed.
Every registered node and recorded updater must be bound before the graph can be
instantiated; a missing binding is a preparation error. Valid zero-work nodes are
disabled without rewriting their unused launch parameters. Invalid count sources
still fail even when another source disables the node, and active nodes refresh
every parameter. This avoids a second cache of applied graph state.

## Level 4 Storage and failure boundaries

`WorldDirectoryData.live_count[p]` counts live worlds, while its
`free_slot_count[p]` counts free rows available for admission and excludes live rows.
Each `FieldStorage.ready_count` instead certifies a safe-access prefix, including
live rows during healthy operation. It can become zero during quarantine even
while physical memory remains mapped. `world_ready_capacity` is the minimum
usable capacity across world, contact and CCD readiness and quotas. Reserved
`capacity` is the fixed addressable limit; none of these counts measures bytes.

`FieldStorage` receives `FieldSpec(name, inner_shape, dtype, packed, alignment_bytes)` and
a borrowed `protected_count` prefix that cannot be reclaimed. It has no native
schema or defaults. World storage borrows the live count as its protection;
transient contact/CCD storage has zero protection at joined maintenance boundaries
and uses explicit ready-count bounds for recorded work. The optional
`protected_count_host` resize argument is a joined snapshot of that protection.
Packed rows share a reservation; optional dense fields remain ordinary Warp arrays. Natural scalar
alignment is the default. Native world and CCD packing conservatively requests
16-byte alignment; blocked Cholesky matrix consumers specifically require it.
Candidate/contact fields use scalar alignment. A packed field element starts at
`reservation_base + field_offset + row * row_stride_bytes`; its typed view is strided
across rows, while each field's inner row stays contiguous. Field payload and
padding are reported separately.

`FieldTransfer` validates source/destination indices, readiness, duplicates and
unsupported overlap before writing. Its GPU status must be successful before the
directory acknowledges a copy. Plans retain source and destination owners; a
retained executable retains its plans. Graphs and transfer plans block explicit
close. Packed VMM views also block reservation release, preserving the mapping
ledger for retry after those views are dropped. Ordinary Warp allocations can
outlive close through external array references; those bytes then belong to the
external borrowers and are outside the closed storage owner's memory report.
Prefix copies reject unsafe partial aliases before recording; exact self-copy
does no work. Disjoint fields of one packed allocation remain valid operands.

`MemoryBacking` owns one physical budget, stable virtual reservations, mapped
physical handles and an unmapped reserve pool. Mapping failure rolls back or
reports the surviving authoritative ledger. Unmapping requires explicit joined
stream/event dependencies; no owner silently changes a caller's CUDA context.
Interruption is an exceptional exit too: an acquired handle cannot disappear from
the ledger merely because cleanup was interrupted or itself failed. Partial
retirement reports which subowners were released and preserves the remaining
ledgers for diagnosis and retry.
Its budget counts mapped and retained physical blocks. Immutable models, dense
metadata, graph storage and caller data are separate ledger entries rather than
unreported free memory.

`resize_backing` first withdraws the unused directory tail, joins all readers,
services W/C/D and publishes their minimum usable world prefix. A normal budget
rejection leaves a recoverable ready prefix. Unexpected driver or publication
failure quarantines the runtime with a GPU health latch: replaying a retained
raw graph cannot initialize, relocate, advance or refresh unsafe physical rows.
The caller must exclude concurrent submissions during this maintenance boundary.
`MuJoCoWorlds.resize_backing` and `MuJoCoWorlds.close` accept actual same-device
Warp `Stream` objects and convert them at the backing boundary. The reusable
`MemoryBacking.maintenance` and `FieldStorage` maintenance methods accept raw CUDA
stream handles; the concrete composition root owns that conversion and submission
exclusion. These two interfaces must not be confused.
Standalone directory closure joins exactly its supplied streams. Borrowed metadata
remains alive while Python references retain it; closure retires submission rights.

Graph preparation has one graph-wide failure boundary. Pure declaration errors
remain correctable before that operation mutates the captured program, including
binding validation after capture. Failure during node emission or after
irreversible marking invalidates the graph for every updater owner, including a
healthy peer. Catching that exception cannot make the program safe to instantiate.
Preparation callbacks are released on success and failure;
captured execution retains explicit resource owners, not arbitrary task closures.

## Level 5 A second domain without a generic backend interface

`test_world_runtime_composition.py` supplies a small, independent composition
root. Its two immutable prototypes contain two and five moving primitives. It
imports the directory, typed storage, backing and graph mechanisms without any
MJWarp or task dependency. The state schema is explicit:

```python
from newton.worlds import FieldSpec, FieldStorage

# Mechanical field storage receives a schema, not a physics object.
state_fields = (
    FieldSpec("position", (primitive_count,), wp.vec3),
    FieldSpec("velocity", (primitive_count,), wp.vec3),
    FieldSpec("seed", (), wp.int32),
)
state = FieldStorage(capacity, live_count, fields=state_fields, backing=backing)

# Generated samples are transient. Their count is not the live-world count.
samples = FieldStorage(
    sample_capacity, zero_protected_count,
    fields=(FieldSpec("primitive_location", (), wp.vec2i), FieldSpec("value", (), wp.float32)),
    backing=backing,
)
```

The concrete root alone decides what survives compaction and what is rebuilt.
It copies position, velocity and seed when moving a lifetime. Generated samples
are derived work, regenerated after the state update, not retained world state.
This smaller copy set is valid for this consumer's full-write kernels; it is not
evidence that MJWarp's partly written Data can use the same copy policy.

```text
Lifecycle: admit -> initialize -> publish -> copy survivors -> publish compaction
                                                       |
                     +---------------------------------+----------------------+
                     |                                                        |
                 prototype 0                                              prototype 1
             update world launch counts                               update world launch counts
             advance retained state                                   advance retained state
             generate samples + count                                 generate samples + count
             validate sample capacity                                 validate sample capacity
             update sample launch count                               update sample launch count
             consume valid samples                                    consume valid samples
                     |                                                        |
                     +--------------------- explicit join --------------------+
```

Each prototype uses two `DeviceGraphUpdates` owners: the first borrows the live
world count; the second borrows the sample count produced by the preceding stage.
Sample overflow rejects that derived frame without corrupting retained state.
Physics may require a different domain-wide failure policy. The mechanisms do
not select that policy for either domain.

The test covers fixed and virtual backing, reset, destroy, retained state,
compaction, a stale handle, generated-work overflow and retry, rejected budget
growth, shared backing reuse, and graph-retained lifetime. It checks unchanged
array bases and executable identity. Its memory report counts owned metadata,
fixed payload and shared physical backing separately, without recounting borrowed
live-count or compaction views. This demonstrates the extension boundary with a
raster-like workload; it is not an integrated renderer or another physics solver.

## Qualification and admitted scope

Prepared workspaces reject isolated free-body implicit solves when
`body_freeadr.size > 0` unless all `ACTUATION | SPRING | DAMPER` stages are disabled;
this does not reject every free joint.

The native feature boundary is the prepared MJWarp workspace: native NxN contacts,
sleeping articulated rigid bodies, Newton solver, implicit-fast integration and
pyramidal cones. Cameras/lights, sensors, flex, tendons, actuator history, fluid,
SDF and energy computation are rejected until their native programs have explicit
count and memory semantics. Python 3.11+, Warp 1.17 and the pinned MJWarp branch
are the tested stack. CUDA graph bindings are validated during preparation.
Fixed MJWarp workspaces supply neither `world_live_count` nor a launch recorder.
Dynamic workspaces supply both; admission and recording reject an incomplete pairing.
`make_step_workspace` is the sole public preparation entry point. Its implementation
class and derived compact solver/collision views are private: callers configure the
original Model/Data, declared scratch fields and recording contract, never a second
set of native aliases. Pure `step_workspace_layout` describes allocation requirements
without performing device operations.

Numerical qualification explicitly rejects nonfinite values and poisons reused
scratch. This found two representation-independent first-write rules: a world
without constraints must start with zero constraint force regardless of sparse or
compact storage, and disabling actuation must not also skip enabled damping (or
vice versa). Analytical derivative cases and allocation-reuse regressions enforce
these rules for eager and captured execution.

Prepared execution also excludes static-only models, enabled ball limits, contact
surface velocity/passive adhesion, requested postconstraint inverse dynamics,
dense full Jacobians wider than 50 padded DOFs, and derivative-enabled
gathered/sparse inertia factorization. Each rejected branch lacks complete
prepared count semantics; disabling a feature so its branch is not executed
remains allowed. The exact admission predicates belong to MJWarp's workspace.

The production numerical test uses independent dense native worlds with six and
108 sliders, mocap bodies and nonempty sites. It covers create, step, retype,
retained-row compaction, invalid payloads, complete retirement, regrowth, reset-only
poses and final-substep poses. Both fixed and VMM storage keep one executable and
fixed array bases. Per-substep overflow, callback ordering and final physical
backing retirement are checked. Fake-driver tests cover resource failures;
architecture gates reject discarded owners and forbidden dependencies. The
unmapped-tail test exposed and fixes native early-return full-array clears that
fixed backing could not reveal.

This runtime does not yet define training-buffer policy, desired-distribution
controllers, task reset distributions, render bindings or solver-independent
query manifests. Those are callers or later concrete compositions. Numerical
parity alone does not establish a performance win: compare matched native options,
contact budgets, actions, reset cadence and complete maintenance cost.

### Joined backing service

`MuJoCoWorlds.resize_backing(world_ready_capacities, streams=...)` accepts one target prefix per
prototype. The caller must exclude new submissions until it returns. The native
composition root owns the W/C/D service order; the mechanical owners and file
tree remain unchanged. It validates and withdraws the whole directory batch,
joins consumers once, returns all safe shrinking ranges before any growth, then
publishes jointly backed prefixes with one final directory rebuild. One joined
live-count readback supplies protected-prefix checks. Same-granule requests still
validate liveness but do not republish readiness or synchronize each row store.

Newly mapped contact and CCD suffixes are initialized by one packed-row clear
per owner plus the explicit contact-address sentinel fill. Existing prefixes
remain untouched. `MemoryBacking` retains one authoritative page ledger, partitioned
by reservation; no second mapping cache or memory-accounting owner exists.
A clean budget rejection may preserve and publish safe partial progress. Driver,
initialization or readiness-publication failures quarantine the whole native
population so an externally retained graph cannot access retired storage.

An optional `spare_bytes` argument trims unused physical handles inside that same
joined service scope after successful readiness publication. `None` preserves
the existing retain-all behavior; zero releases all spare handles. A positive
reserve rounds up to the CUDA allocation granularity and never allocates handles
to fill an undersized reserve. It changes neither mapped ranges nor virtual
addresses, so captured graphs remain valid. The mechanical
`MemoryBacking.trim(keep_bytes=...)` operation owns release and retry accounting;
no task-level per-reset synchronization or second allocator is introduced.
Release errors retain ownership of failed handles and quarantine the population
through the existing service failure path. No spare-reserve policy is selected
by default; callers should measure maintenance cost and subsequent regrowth.

Both `WorldDirectoryData.slot[id]` and `WorldTransaction.destination_slot[request]`
are prototype-local rows. Slot metadata arrays such as `slot_id` and `slot_live_rank`
use `slot_starts[prototype] + local_row`. Reset payloads write directly to
the local admitted destination before publication.
