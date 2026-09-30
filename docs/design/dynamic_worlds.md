<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Dynamic native worlds

`newton.solvers.MuJoCoWorlds` runs changing populations of prepared native world
prototypes. A keyboard reset can move a logical world from a six-DOF prototype to
a 108-DOF prototype while continuing worlds keep their physical state and solver
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
generations. A failed replacement preserves the old lifetime and blocks mandatory
advancement; ready capacity and live population are separate quantities.

## Level 1 Ownership and the composition root

The production composition root is `newton.solvers.MuJoCoWorlds`. It composes
`newton.worlds.WorldDirectory`, mechanical `RowStorage` and `CudaBacking` owners,
prepared graph updates, and native MJWarp programs. Native Data is authoritative:
there is no replicated Newton State mirror, solver adapter or second lifecycle owner.

```text
Caller                         owns actions, reset decisions, request/output buffers
  |
  v
MuJoCoWorlds                   concrete physics composition and preparation
  +-> WorldDirectory          identities, admission, membership, relocation plans
  +-> prepared native groups  immutable topology + authoritative mutable Data
  |    +-> RowStorage W       world state and per-world solver scratch
  |    +-> RowStorage C       contacts and broadphase candidate scratch
  |    +-> RowStorage D       CCD/EPA scratch
  |    +-> graph bindings     prepared program; GPU work/count inputs
  +-> CudaBacking             one budget; reservations, handles, mappings, retirement
```

`WorldDirectory` does not import a solver, allocator, Torch or IsaacLab. It plans
changes and requires explicit initialization/copy acknowledgements before
publishing them. It never copies physical state. `RowStorage` knows field layouts
and typed row operations; it does not know native Data, world identities, default
values or physics policy. `CudaBacking` knows bytes, mappings and completion
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
    capacities=virtual_slot_limits,
    id_capacity=max_live_worlds,
    command_capacity=max_requests,
    contact_capacities=contact_limits,
    ccd_capacities=ccd_limits,
    memory_budget_bytes=budget,
    initial_rows=initial_ready_rows,
)
commands = create_world_commands(max_requests, device=worlds.device)
results = create_world_results(max_requests, device=worlds.device)
graph = worlds.capture(commands, results, substeps=2)
# Caller kernels write commands, reset payloads and controls before replay.
wp.capture_launch(graph)

# Cold capacity service joins all readers before changing physical mappings.
worlds.resize_backing(tuple(ready_rows_per_prototype), streams=(stream.cuda_stream,))
# Drop every graph borrower before closing storage.
del graph
worlds.close(streams=(stream.cuda_stream,))
```

The experimental native runtime requires Python 3.11+, CUDA, and the pinned
MJWarp prepared-workspace API. Generic world directory records remain usable
without importing a native engine. CUDA graph preparation compiles one small
bridge into the Warp cache, outside the package source; `CUDACXX`/`CUDA_HOME`
select the toolkit, or `NEWTON_CUDA_GRAPH_LIBRARY` selects a prepared library.

`capture` has preparation-only hooks with caller-owned payload buffers:

- `validate(commands, transaction)` rejects invalid payloads before admission.
- `initialize(group, requests, destinations, count, base_status, transaction,
  sequence)` overrides full default rows and explicitly acknowledges initialization.
- `before_step(group)` records controls once before the ordered native substeps.
- `after_substep(group)` records contact consumers and diagnostics after each step.
- `retain=(...)` keeps all external arrays used by those recorded kernels alive.

`permit` is a device scalar that suppresses physical advancement without suppressing
valid resets. `refresh_kinematics=True` updates body, geometry and site poses after
a reset-only frame or the final native substep. Failed mandatory requests gate both
advancement and pose refresh. Per-prototype branches update their own poses before
the common graph join. Setting `refresh_kinematics=False` is explicit and useful for
physics-only measurements; derived transforms then describe the last refresh.

The directory is the only lifetime authority. `WorldCommands` carries operation,
identity, expected generation and target prototype; `WorldResults` carries status
and resulting identity/generation. A caller must treat the directory's advancement
permit as authoritative after a partly successful batch. The root only publishes
rows after default/payload initialization or complete retained Data relocation has
acknowledged completion. Current replacement admission needs a spare destination
row, including same-prototype reset; this temporary demand is part of the backing
budget and is not hidden by the API.

Preparation lowers topology, collision exclusions and shared physical defaults
once per prototype. It must not first allocate a maximum-sized dense population
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
newton/_src/utils/row_storage.py         typed field layout, readiness and transfers
newton/_src/utils/cuda_graph.py + .cu    prepared CUDA node bindings and fork/join
newton/_src/solvers/mujoco/worlds.py      native schema and physics composition
newton/tests/test_worlds.py              generic semantics and architecture gates
newton/tests/test_cuda_vmm.py            failure-injected mechanical ownership
newton/tests/test_row_storage.py         typed transfer, lifetime and alignment gates
newton/tests/test_cuda_graph.py          independent counts, CUDA grid and graph gates
newton/tests/test_mujoco_worlds.py        independent dense physics and pose oracle
```

The native root uses the existing one-world preparation path. It does not derive
from `SolverBase`: a changing native population has no matching dense Newton
`State`, and exposing a fabricated state would hide its real ownership. The
existing homogeneous solver remains unchanged. There is no backend factory,
compatibility alias, source-code inspection, global Warp launch interception or
second lifecycle map.

## Level 3 A complete transaction

The sole directory records are `WorldDirectoryData`, `WorldTransaction` and
`WorldCompaction`. Caller-owned `WorldCommands` contains a monotonically increasing
batch sequence, operation, identity, expected generation and target prototype.
`WorldResults` returns status, identity and generation. Operation/status/phase
values have canonical public enums in `newton.worlds`.

```text
begin -> caller validation -> admit -> full defaults + caller payload -> publish
                                                                |
                          per-request initialization acknowledgement

plan moves -> complete native Data copies -> acknowledge -> publish moves

health + ready capacity + transaction permit -> native step -> pose refresh
```

Admission reserves destination tickets without exposing incomplete rows. Failed
requests do not invalidate independent successful requests, but a mandatory failed
request disables the batch's physics permit. Replaying the same accepted sequence
does not apply the lifecycle command again. Reused IDs increment their generation;
stale references and generation exhaustion are rejected.

Compaction copies complete native Data rows in the correctness baseline. It does
not copy transient contact/CCD queues. Structural zeros in partly written fields
are part of initialization: reducing relocation to intuitive position/velocity
fields failed a VMM regrowth test and is intentionally not the implementation.

Native steps of different prototypes form independent branches of one graph.
Each branch has one GPU count updater, a physics permit and an optional pose
refresh; all branches join before replay completion. This removes host iteration
over separate executable graphs. CUDA node scalar arguments and launch extents
have explicit independent W/C/D count sources. Fixed worker grids remain fixed.

## Level 4 Storage and failure boundaries

`RowStorage` receives `FieldSpec(name, inner_shape, dtype, packed, alignment)` and
one authoritative count. It has no native schema or defaults. Packed rows share
a reservation; optional dense fields remain ordinary Warp arrays. Natural scalar
alignment is the default. Native world and CCD fields explicitly request 16-byte
alignment because aligned tile consumers require it. Candidate/contact fields use
scalar alignment. Field payload and padding are reported separately.

`RowTransfer` validates source/destination indices, readiness, duplicates and
unsupported overlap before writing. Its GPU status must be successful before the
directory acknowledges a copy. Plans retain source and destination owners; a
retained executable retains its plans. Explicit close fails while borrowers exist.

`CudaBacking` owns one physical budget, stable virtual reservations, mapped
physical handles and an unmapped reserve pool. Mapping failure rolls back or
reports the surviving authoritative ledger. Unmapping requires explicit joined
stream/event dependencies; no owner silently changes a caller's CUDA context.
Its budget counts mapped and retained physical blocks. Immutable models, dense
metadata, graph storage and caller data are separate ledger entries rather than
unreported free memory.

`resize_backing` first withdraws the unused directory tail, joins all readers,
services W/C/D and publishes their minimum usable world prefix. A normal budget
rejection leaves a recoverable ready prefix. Unexpected driver or publication
failure quarantines the runtime with a GPU health latch: replaying a retained
raw graph cannot initialize, relocate, advance or refresh unsafe physical rows.
The caller must exclude concurrent submissions during this maintenance boundary.

## Qualification and admitted scope

The native feature boundary is the prepared MJWarp workspace: native NxN contacts,
sleeping articulated rigid bodies, Newton solver, implicit-fast integration and
pyramidal cones. Cameras/lights, sensors, flex, tendons, actuator history, fluid,
SDF and energy computation are rejected until their native programs have explicit
count and memory semantics. Python 3.11+, Warp 1.17 and the pinned MJWarp branch
are the tested stack. CUDA graph bindings are validated during preparation.

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

`MuJoCoWorlds.resize_backing(rows, streams=...)` accepts one target prefix per
prototype and excludes new submissions until it returns. The existing native
composition root owns the W/C/D service order; the mechanical owners and file
tree remain unchanged. It validates and withdraws the whole directory batch,
joins consumers once, returns all safe shrinking ranges before any growth, then
publishes jointly backed prefixes with one final directory rebuild. One joined
active-count readback supplies row liveness checks. Same-granule requests still
validate liveness but do not republish readiness or synchronize each row store.

Newly mapped contact and CCD suffixes are initialized by one packed-row clear
per owner plus the explicit contact-address sentinel fill. Existing prefixes
remain untouched. `CudaBacking` retains one authoritative page ledger, partitioned
by reservation; no second mapping cache or memory-accounting owner exists.
A clean budget rejection may preserve and publish safe partial progress. Driver,
initialization or readiness-publication failures quarantine the whole native
population so an externally retained graph cannot access retired storage.

Both `WorldDirectoryData.slot[id]` and `WorldTransaction.destination_slot[request]`
are prototype-local rows. Only slot metadata arrays such as `slot_id` and
`slot_state` use `starts[prototype] + local_row`. Reset payloads write directly to
the local admitted destination before publication.
