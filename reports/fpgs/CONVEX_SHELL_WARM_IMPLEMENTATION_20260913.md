# Current warm shell implementation — not GPU qualified

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
the polygon vertex centroid. The weight is the greater of 8 FP32 epsilons
and four scale-aware guards divided by patch diameter, and must be at most
one percent. Both current surface witnesses are reconstructed at that
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
