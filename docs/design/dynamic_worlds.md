# Growable-memory native world populations

`newton.solvers.MuJoCoWorlds` composes homogeneous MJWarp prototypes into a changing set of world instances. Each prototype keeps its topology, geometry, inertia and solver constants. A replacement initializes another prototype's instance and publishes a new lifetime; it does not rewrite an articulation's topology in place.

This API is experimental. It requires CUDA and Python 3.11 or newer. Install Newton's optional `sim` dependencies, including the Apache-2.0 `gpu-components` package. Importing base Newton does not load this optional composition.

## Ownership

The consuming application owns prototype preparation, controls, initialization payloads, observations, reset policy, graph submission and maintenance scheduling. Newton's `MuJoCoWorlds` owns the native population composition:

- The standalone `gpu_components.directory` operations own numeric identities, generations, placement and admission. Their records live in `gpu_components.directory_data`.
- `gpu_components.fields` owns typed allocations and transfer operations. `gpu_components.backing` owns virtual reservations and physical byte mappings. These components have no simulation knowledge.
- `gpu_components.graph` binds declared counts to captured CUDA kernel nodes. It does not infer field domains or choose physics stages.
- MJWarp owns `StepBindings` and the prepared native workspace. One shared bindings record borrows the world, candidate and CCD storage domains and owns the stage-binding ledger. Newton does not duplicate that ledger or failure state.
- Native MJWarp `Data` is the authoritative physical state. There is no dense Newton `State` mirror.

`mjw.array_fields(model_or_data)` yields native paths, borrowed arrays and their declared
dimensions. `mjw.replace_arrays(template, arrays)` binds replacement arrays without copying
payload or assigning readiness. MJWarp owns this representation; Newton consumes its
`nworld`/`naconmax` axes to allocate storage, then changes scalar capacities explicitly with
`dataclasses.replace`. Numerical extents never determine capacity domains, and an array
appearing at two native paths does not acquire a second allocation owner.

MJWarp declares native scratch element types and dimensions. GPU Components supplies
`StridedLayout`, packing, binding and descriptor validation for both fixed scratch
and growable `FieldStorage`. The workspace is a passive record; MJWarp's preparation,
validation and reporting functions operate on it. Solver alignment requirements
remain beside the solver. Ordinary `mjw.step(model, data)` still needs no workspace.

Newton supplies concrete capacity `Data`. MJWarp workspace preparation owns a
borrowed execution descriptor with symbolic count identities and the same arrays;
validation checks both descriptors. This projection does not own another physical
state or certify that every reserved row is backed.

The former `newton.worlds` namespace and Newton's generic allocator, directory and graph implementations are removed. Import each standalone concept directly; there are no compatibility aliases.

## Public native operations

`MuJoCoWorlds` constructs the concrete physics composition from immutable one-world Model/Data pairs. `capture()` prepares its single executable, `grow_backing()` admits more physical capacity, `resize_backing()` performs joined memory maintenance including shrinkage, `memory_report()` reports each owner's retained bytes, and `close()` retires resources after graph borrowers are gone.

The `directory` property exposes borrowed `InstanceDirectoryData`, not the mutable directory owner. `batch_result` exposes the outcome and advancement permission. Both remain readable after quarantine or closure; reading metadata does not authorize replay or certify physical readiness.

`populations` returns `MuJoCoWorldPopulation` views. A view exposes its numeric prototype index, immutable model, authoritative Data, reserved capacities, live count, accessible-prefix counts and joint world readiness. Applications record ordinary Warp kernels; a separate preparation operation binds their exact captured records to numeric count sources. It exposes no backing-service or retirement authority.

A handle is `(instance_id, generation)`. A slot is a prototype-local storage position. Compaction may move a handle to another slot without changing its generation. Symbolic names and scene paths must be resolved outside this numeric relation API.

## Prepare and capture

The following integration fragment assumes `prepared` contains warmed one-world native Model/Data pairs. The application supplies capacities, initial accessible prefixes and a byte budget appropriate to those prototypes; this is not a standalone asset-loading example.

```python
from gpu_components import directory
from gpu_components.directory_data import InstanceOperation
from newton.solvers import MuJoCoWorlds

runtime = MuJoCoWorlds(
    prepared,
    world_capacities=world_capacities,
    id_capacity=id_capacity,
    command_capacity=command_capacity,
    initial_world_ready_capacities=initial_world_ready_capacities,
    memory_budget_bytes=memory_budget_bytes,
)
commands = directory.allocate_commands(command_capacity, device=runtime.device)
results = directory.allocate_results(command_capacity, device=runtime.device)
graph = runtime.capture(commands, results, substeps=substeps, retain=application_buffers)
```

Omitting `memory_budget_bytes` selects fixed backing. Both paths use the same instance protocol and native solver program. With virtual backing, reserved capacity is not physically committed capacity. The byte budget includes mapped and reusable spare physical backing. Initial prefixes must be backed before they are admitted.

Preparation validates Model batch parameters, complete Data and scratch layouts, storage alignment, buffers and launch declarations before publication. Every nonempty batched immutable Model parameter must broadcast from one row: moving Data must not change a world's constants. Unsupported MJWarp feature combinations are rejected by its prepared workspace admission.

A captured graph retains the exact directory, storage, transfer, workspace, binding and explicitly supplied application resources. Preparation callbacks are released afterward. Each runtime prepares one graph; a failed irreversible recording invalidates that preparation rather than silently retrying partially marked nodes.

## Replace a world

The opcode is `InstanceOperation.REPLACE`, with numeric value 2. Episode reset remains application meaning. A replacement keeps the identity, initializes a spare destination and increments the generation only when publication succeeds. The old lifetime survives admission or initialization failure. Same-prototype replacement also needs a spare destination under the current protocol.

```python
# Populate the existing command buffers before submitting the graph.
# These writes are schematic host-side setup, not a GPU reset kernel.
commands.sequence.fill_(next_sequence)
commands.count.fill_(request_count)
# Application code writes operation, instance_id, generation and prototype
# for exactly the consumed request prefix. A replacement uses:
operation = int(InstanceOperation.REPLACE)
```

Newton records validation, admission, complete default-state transfer, optional domain initialization, lifetime publication and same-prototype compaction. It relocates every retained world field, including solver history. Validation receives only command inputs, rejection statuses and the accepted-prefix flag. Initialization receives only its admitted indices, destination rows, copy outcome, sequence and acknowledgement output. A callback acknowledges a destination only after its complete payload is ready.

These operations run on the GPU inside the prepared graph. A reset that fits the
already accessible prefixes needs no host allocation, model reconstruction or
graph capture. Physical capacity service is a separate operation, not a required
stage of every reset. Applications decide where they consume the GPU outcome;
the runtime does not require a CPU demand calculation before submission.

Prototype initialization branches access disjoint destination rows and join
before lifetime publication. Compaction copies likewise branch by prototype and
join before placement publication. After successful compaction, live rows form a
dense prefix. Admission can then select the next free rows directly rather than
choosing distant free slots that require avoidable relocation. This does not make
request-to-row assignment deterministic or remove copies needed to fill holes
left by departing lifetimes.

Independent requests may succeed even when another request denies physics advancement. Inspect per-request results together with the batch outcome. Equal-sequence replay does not consume edited commands. Retry a rejected batch with a fresh sequence. Terminal generations reject replacement but remain destroyable without wrapping; their dead identities are never reused.

Every consumed lifecycle batch invalidates transient contact indices. Until native collision runs again, zero `nacon`/`ncollision` means there are no current contact records, not that the geometry is separated. Read contact forces in `after_substep`.

## Record controls and observations

Application callbacks use ordinary Warp launches. Count semantics belong to a separate preparation operation:

```python
def before_step(population):
    wp.launch(
        apply_controls, (population.world_capacity, control_width),
        inputs=(population.data.ctrl, controls),
    )


def application_bindings(population, launches):
    if any(record.kernel is not apply_controls for record in launches):
        raise ValueError("Unexpected application kernel")
    extents = tuple((index, 0, population.world_live_count) for index in range(len(launches)))
    return extents, (), ()


graph = runtime.capture(
    commands, results, before_step=before_step,
    application_bindings=application_bindings, retain=(controls,),
)
```

`application_bindings` receives exactly this population's callback records, in recording order. It returns the numeric `extents`, `parameters`, and `fixed` relations accepted by `gpu_components.graph.adopt_launches`. Every record needs an explicit declaration. Repeated uses of the same kernel may have different count sources; kernel identity and shape never imply ownership. Scalar bindings use `KernelParameterBinding` with explicit int32 argument slots. Fixed records are declared by local index and must guard their own accesses. Application fills/copies are rejected until their storage admission is explicit.

`before_step` records controls once; `after_substep` records consumers after each native substep. Preparation validates native descriptors before each step and after the final callback. Optional final kinematics also covers valid reset-only frames. Updater completion and global error guards precede every prototype's conditional program. Compilation operations and numerical callbacks are released after preparation; only explicit data buffers and numeric bindings remain retained.

## Borrow live state and export

Population views borrow live arrays; they are not snapshots. A consumer must use
the current directory placement and wait for the producing work. A generation
identifies a lifetime, but does not preserve its slot across compaction. The
application must keep the accessed ranges valid and exclude conflicting replay,
reset, compaction or retirement until that consumer completes. Include every
consumer stream when joining maintenance or closure. Retaining Python references
or a graph alone does not establish this ordering.

A renderer may instead consume an application-owned snapshot: copy only admitted,
accessible rows after their producer completes, and release the live borrow once
the copy finishes. Rendering can then proceed independently from the snapshot.
The application owns its layout and any conversion from physics state.

Raw array `.numpy()` and MJWarp `get_data_into` copy complete array descriptors,
even when they subsequently select one world. Concrete capacity metadata does not
prove that these full ranges are physically backed. Such readback is unsupported
for partially backed population arrays; first copy the required live range into
an accessible, caller-owned snapshot. No raw foreign-pointer consumer acquires
readiness, initialization or retirement guarantees merely by receiving a pointer.

## Resize backing and retire

For monotone growth, use `grow_backing()`. It maps fresh suffixes, then queues
accessible-count updates, contact/CCD scratch initialization and directory
admission in that order on the current stream. Subsequent consumers must be ordered after that publication.
The caller excludes concurrent submissions and supplies every consumer stream.

```python
runtime.grow_backing(
    larger_world_ready_capacities,
    streams=consumer_streams,
)
```

Fresh suffix mapping does not need to wait for readers of disjoint existing
storage. Reusing a previously mapped virtual range is different: `grow_backing()`
uses joined maintenance when historical address reuse requires it, completing
publication before returning. Both paths only add mappings: growth never releases
backing and preserves any existing cap on unmapped spares. CUDA mapping
is still a host driver operation and may itself stall; this API does not promise
asynchronous driver execution. Field readiness certifies accessible addresses;
the later directory publication admits destinations, and captured initialization
completes their world payloads before they become live instances. A clean mapping
budget failure leaves readiness unchanged and retains any newly mapped headroom
for a later retry.

The application must exclude new submissions and supply every consumer stream. `resize_backing()` joins them, withdraws inadmissible tails, returns all safe ranges before any growth, services world/contact/CCD storage, and publishes only coherent accessible prefixes. It cannot retire a live tail; compact or destroy those instances first.

```python
runtime.resize_backing(
    requested_world_ready_capacities,
    streams=consumer_streams,
    spare_bytes=0,
)
```

`spare_bytes=None` keeps all reusable spare handles. A nonnegative value trims available unmapped backing after successful service; it does not preallocate reserve. Unlike fresh growth, a clean budget rejection during joined resize can leave a safely backed partial ready result and be retried. Driver, initialization or publication failure quarantines dependent execution and preserves surviving resources for diagnostics and retirement.

Destroy every graph reference and join consumers before closing the runtime. Closure remains retryable after partial resource retirement. Reports identify retired subowners explicitly rather than fabricating zero usage for resources still owned elsewhere.

## Qualification and limits

The standalone package owns generic relation-oracle, byte-ledger, graph, transfer and failure tests. Newton's `test_mujoco_worlds` covers native composition, retained state, finite dense-physics parity, partial sleep, contact invalidation, failure quarantine and preparation lifetimes. It also rejects duplicate generic implementations and duplicate native binding authorities.

The prepared native path currently admits the keyboard-oriented MJWarp feature subset enforced by `step_workspace_layout`; it is not a universal adapter for all MJWarp options or every Newton solver. Host VMM maintenance is outside graph replay. Existing mapped capacity and dynamic counts can be reused without rebuilding the graph, but a new immutable topology or an unsupported native branch requires new preparation. Task observations and learning buffers remain application-owned.
