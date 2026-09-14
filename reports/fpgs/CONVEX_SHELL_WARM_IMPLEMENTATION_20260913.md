# Current warm shell implementation — closed whole-task loss

## September14,02:40: valid full cost and decisive diagnosis

Runtime `8a45444bf3a61c36560e9d3a008988b31697ed8d` fixes the shared polygon
epoch: peer lanes capture count/bank before evaluating vertices, and all
lanes finish before lane0 clips or changes either shared value. The new
source regression fails without the fix. Twelve CPU tests/pre-commit pass;
18 physical methods/card pass with0 skips/errors/failures. Both full warmups
then finish without the earlier shared-read crash. This is not a claim that
the earlier allocation-order memcheck report was resolved: baseline repeats
that non-discriminating report, and the original diagnostic logs remain.

The first post-fix full run was correctly rejected by capacity checks:
raw peaks390708/390594 exceed286720, and RTX also exceeds row capacity192.
No timing was accepted from that run. Read-only ownership review finds no
duplicate warm/cold queue publication. The shell publishes a wide3–4 point
patch plus deepest witness, whereas the original fitted manifold can reject
its patch at the2-degree gate and retain just its deepest point. More
contacts are not themselves proof of better or worse physics.

Existing demand calibration first uses raw393216 with rows192. It records
attempted row peaks214 RTX/218 GB, retaining failure and dropped telemetry.
One measured retry with rows224 completes200/40/40 with all1120 collision
checks and all row-drop counters clean: raw376215/375786 and rows223/218.
Because223 leaves only one row of margin, the final comparison uses rows240,
raw393216, original MF64 and broad524288 in **both arms**; propagation240 is
constructor-derived. No timestep, iteration, material or contact threshold
changes. These are measured capacities, not guaranteed maxima for every run.

| GPU | Current10a baseline | Complete8a shell | Baseline/candidate |
|---|---:|---:|---:|
| RTX |17.326773ms|130.477195ms|0.132795x|
| GB300 |16.987419ms|105.718710ms|0.160685x|

One16K seed0,200 warm/40 wall/40 physics discovery. All four children,
original two-boundary capacity/finite/source checks, actual shell activation,
equal allocation checks and final source/idle guards pass. Whole manifest
`/tmp/fpgs-convex-shell-allegro-variants-paired16k-20260914-03/manifest.json`,
SHA256 `afe6400323063f786fb5ab1115d1c2a8ae0389d3147bd9078eb0d53a28ab7132`.
The existing original owner command from READY.md uses only the compatibility
folder `/tmp/fpgs-shell-measured-capacity-pbb508cK` and new output03. It
executes the original capacity helper with four explicit allocation-source
substitutions (SHA cd737b4c); the files and actual contract are recorded by
the parent. Original helpers, Lab and installed packages remain unchanged.

Three-step nodes identify102.488358ms RTX/76.867317ms GB in
`current_shell_commit` alone. Runtime uses3008/2432 blocks of32, registers
124/128,10496B shared and reported local memory0. Complete cap reconstruction,
edge/slab intersection, serial sorting/clipping and repeated surface-plane
evaluations are new work; they overwhelm the historical~2.925ms covered cold
query envelope. This is an expensive representation, not a small mapping
defect. PGS also costs15.6131/16.9802ms with the new contact output. Even
subtracting the entire shell kernel from the captured whole span leaves
about29.22/29.88ms, already above the current baseline. This optimistic
fixed-trace subtraction is not a measured zero-shell implementation or a
hardware-counter attribution.

Both node children and the original process-scoped physics-graph analyzer
pass, retaining12 physics roots and all2088 graph kernel/memory nodes per
card. The backend parent subsequently fails its final ratio calculation
because this deliberately single-FPGS diagnostic has no MJ arm; that failed
status is preserved. Independent original capacity/source/idle checks pass.
Node manifest `/tmp/fpgs-convex-shell-allegro-nodes-paired16k-20260914-01/manifest.json`,
SHA256 `5477c3569b4d9d0755c1e7e2601f869c2825db055b3ec42ca31bb7f4cf451204`.
Demand retry manifest SHA256
`435d22e9d7a1b1b52dae7641c1d0816c35aea5b6316345ec63cd4c7ba189a482`;
physical fix manifest SHA256
`62ccb96cd7579e887b35ad93af4edc16e628013b7816ead60546d0e128a0224a`.

Decision: keep default-off and do not promote or tune this representation.
A replacement needs inexpensive feasible contact support without building
the complete shell polygon, and must account for its downstream row/solve
cost. No new such algorithm has yet met that cost model. The earlier
physical holds, capacity failures and original handoff remain preserved.

## Original implementation checkpoint

This isolated experiment starts from `10a0b5242109000c699cc92f8c884254ac329ddf`
on branch `ooctipus/convex-shell-warm-20260913`. The measured predecessor and
all saved inputs remain unchanged. The approved design is retained verbatim
in `CONVEX_SHELL_WARM_CARD_20260913.md` (original SHA 170ba4d4).

## Ownership and admission

`NEWTON_NARROW_PHASE_COHERENT_SHELL=1` is a new default-off selector. It
requires the existing `COHERENT_CONVEX=reject_only`, `CONVEX_BSP=1` flags and
an actually admitted immutable BSP provider. All existing explicit-pair,
lean CUDA split, non-speculative, no-particle/mesh/heightfield/hydroelastic,
no-body-pair-reduction and no-gradient restrictions remain. The default
branch retains its original three query factories and original manifold.

Only `sim/collide.py` and `geometry/narrow_phase.py` are changed among the
existing sources. The original BSP support selection, simplex module,
rejection-only factories, manifold and writer remain untouched. The new
private simplex copy restores the old full-coherent callback seam. A source
recovery test strips exactly those callbacks and recovers the current cold
simplex factory. The new cold call preserves the current default iteration
and cutoff arguments; it does not inherit the old early cutoff argument.

The old full-coherent cache stores current feature IDs, masks and its warm
queue, with the original source/type, immediately previous generation,
world epoch, reset and graph-lease law. There is no new KKT cache. Current
BSP support selection supplies the same feature IDs; unsupported shapes
retain the original support callback. Penetrating or uncertain warm results
go to the complete original cold MPR/GJK plus manifold path.

After a positive warm query, one 32-thread owner transforms the actual
vertices, constructs the two current support-shell caps and intersects their
projected domains. It evaluates the actual upper/lower hull surfaces and
lazily clips violated active face-pair halfspaces. Unchanged polygon
vertices retain their evaluated gaps; only new edge intersections are
evaluated again. No all-face-pair product or global polygon panel is built.

The shell result is private until every selected current contact and the
retained deepest witness pass the actual world-space strict writer gap
test. Success writes at most four wide-span shell contacts plus that witness,
within the existing five-contact reservation. Failure appends the pair once
to original cold work, with no partial public contact writes. Cached warm
witnesses never substitute for current transformed surface points.

## Bounds, geometry and selection

The immutable descriptor keeps original triangles and exact unique edges;
it aliases original mesh points and the BSP invalidation flag. Admitted
mesh limits are 64 vertices and 128 original triangles. Larger/unsupported
meshes use complete original queries; boxes have their exact eight-vertex
geometry. No approximate face merging or extra public maximum is used.

Per owner, both caps and the two polygon banks are bounded at 64 vertices,
with 64 lazy cuts. Every append and final publication is checked. Overflow,
unsafe planes, nonfinite data, degeneracy or nonconvergence selects the full
cold path. The saved hard RTX case requires 33 cuts and remains supported;
it is not given a special threshold.

To avoid rejecting mathematically boundary contacts solely through FP32
writer rounding, selected points move by a tiny convex combination toward
the polygon vertex centroid. The weight includes a tiny spatial guard and
the convex-gap interpolation needed to gain four existing guards of gap
headroom, and must be at most one percent. Both current surface witnesses
are reconstructed at that
interior point. Actual positive gap must remain strictly inside the authored
shell before any write. No gap clamp or shell widening is used. Strict
point/count/vertex identity is not a physical acceptance criterion.

## Completed CPU and source checks

The missing native API failed before implementation. Focused checks now
cover current boxes; both orientations and nonuniform scales; separated,
penetrating and invalidated inputs; local-capacity refusal; both original
saved hull-patch failures; cold/warm/source-change/subset-reset/empty work;
single publication ownership; graph lease/reset validation; and source
recovery. Existing BSP complete-factory and rejection-only source controls
also pass. CUDA graph execution is not implied by their CPU execution.

The two saved current patches retain broad spans of 0.0259576 m (RTX) and
0.0965003 m (GB), with strict gaps and current original-hull surface and
containment checks. The polygon counts are not constrained to old counts.
The metre-scale synthetic warm case correctly refused the unchanged old
1e-4 m query envelope; the warm test uses hand-scale authored geometry.
An initial aligned-prism overflow control had only 64 distinct cap points;
the corrected twisted-prism input genuinely exceeds the unchanged bound.
Neither finding changed runtime bounds or numerical tolerances.

The loaded control adapts the existing two-body impact construction and
oracle to actual finite hulls with explicit masses/inertia. It retains the
task's 12 serial sweeps / eligible 24 parallel maximum, current friction,
current contact pipeline and two Newton substeps per 1/120 s simulation
step. It records actual route and warm commit, checks momentum, energy,
relative rebound and changed patch spin, and evaluates both original and
candidate authored states. CPU checks cover the model and real constructor
signature only: original matrix-free dynamics require CUDA. This small
control is not complete Allegro trajectory/convergence acceptance.

## Compiler resources and budget

All controlled-writer factories compile and assemble offline for sm120 and
sm103. Shell commit uses 110/107 registers, 10,496 bytes shared storage and
544 bytes stack, with zero register spills. Warm queries use 120 registers
and 384 bytes stack; cold GJK uses 128 registers and 352 bytes stack; cold
MPR uses 128 registers with no stack. These are compiler facts, not timing.
The actual production writer also compiles and assembles: 126/128 registers,
10,496 bytes shared and 544 bytes stack, with zero spills. It is recorded
separately in the qualification scratch, rather than pretending the
controlled writer is the full live ABI.

The complete warm query, current transforms/planes, shell construction,
validation, queue, fallback, writer and existing reducer are all charged in
the eventual whole run. The target remains at least 1.7 ms net RTX saving
against the source-identical current rejection-only/BSP baseline. Historical
full-coherent gains are physically held and nonadditive. There is no current
GPU physical or timing result for this implementation at this checkpoint.

Exact reproducible commands, full source/input pins, offline artifacts and
the explicit root-owned paired lease command live in
`/tmp/fpgs-convex-shell-qualification-9EDYH9XH/READY.md`.

## First paired gate and bounded correction

The first frozen source `c5ea3b8a` ran all 17 methods on both GPUs at
`/tmp/fpgs-convex-shell-physical-paired-20260914-01`. Both had two failures,
no skips or errors; final source/idle checks passed. Manifest SHA:
`9b5fede72629b717d664cd896b00551b8dd6649f893396212151227894ebd440`.
The failed evidence and original pins are preserved. No whole timing ran.

The saved GB patch safely refused with reason 8 on both GPUs. This was a
warm-coverage failure, not publication of invalid contacts: complete cold
fallback remains authoritative. CPU selected gap was 0.002999998629 m versus
shell 0.003000000142 m, only 1.5134e-9 m headroom, below one FP32 ulp of its
approximately 0.039 m surface witness. A spatial-distance interpolation alone
does not guarantee gap headroom on a shallow face.

The correction retains the same centroid, one-percent movement cap and four
existing guards. For convex current gap g, interpolation obeys
g((1-alpha)p+alpha*c) <= (1-alpha)g(p)+alpha*g(c). Each selected point uses
the larger of the spatial weight and the weight required for gap headroom.
Already-safe vertices avoid division. A missing interior centroid, invalid
denominator or excessive weight refuses; both surfaces and the actual strict
gap are still reconstructed and checked before publication. No shell or
physical tolerance is widened. The added work is one current centroid
surface evaluation; selected vertex gaps reuse the existing per-call cache.

The loaded test executed a real warm commit, but both original and candidate
read zeros from unpublished body velocities. With the task's lazy mode,
`_stage7_update_kinematics` intentionally returns before publishing body_qd.
The test now calls the documented `publish_kinematics(state_out)` once after
the existing solve/capacity check. It adds no solve, timestep or iteration.
Raw records from the failed test are not treated as physical solver results.
The unchanged loaded momentum/energy/rebound/spin assertions await the
corrected paired gate. No physical gain is promoted from this first attempt.
