# Structural window: integrated discoveries

Checkpoint: 2026-09-14 00:17 UTC. Work continues toward the 24-hour window in
`STRUCTURAL_FOURX_20260913.md`; the 4x target is **not achieved**. These are
discoveries, not replacements for accepted baselines except where repeat
evidence is explicitly given. No candidate has met the complete 4x target.
No Isaac Lab source, timestep, substep or iteration allowance was changed.

**Current status:** the original finite-terrain manifold remains rejected.
Its corrected successor now passes rebound and sliding/support/border checks,
including full-history support-force readback. One earlier decimated-force
failure and the unreduced fixture-capacity failure remain preserved, not
erased. The latest finite/weld source remains default-off pending fresh
shared-MJ comparison and repeated timing. Do not use old rejected variants
as performance denominators.

## Latest completed experiments (September14,00:17)

G1 finite/WELD geometry is composed onto current sparse/packed bd1, branch
`ooctipus/g1-finite-seam-compose-20260913`, pushed to ooctipus/newton through
report-only `4a0bf739778041d5f833abfd1d077703677eb12d`. Runtime7e6cb1bd,
timed source6e08, diagnostic-only successor f375. Original31cf ancestry and
the inherited handoff are retained; no parent pointer or Lab source changes.

The flat internal-triangle-edge fix changes premature sliding stops
.337–.408m to approximately.5072m, consistent with the original timestep's
Coulomb prediction. Composed saved96 geometry/lifecycle/rebound passes on
both cards. An initial GB support check failed by2.126% against2%, using only
five decimated samples. Direct80-step tail force readback then passes with
<=0.001508% average-weight error and<=6.01e-8 momentum mismatch. A genuine
2.5ms three-contact/5ms force transient is recorded, not hidden; the original
failure and original test assertion/tolerance remain unchanged.

Complete same-source FPGS A/B, finite0->1 and SPARSE/CELL/PACK/WELD1 on BOTH:

| GPU | Corrected original physics | Finite physics | Incremental ratio |
|---|---:|---:|---:|
| RTX |24.714743ms|22.905959ms|1.078966x|
| GB300 |38.007732ms|27.702692ms|1.371987x|

All four children/activation/capacity/source/idle checks pass. One discovery
round, not4x or a fresh MJWarp ratio. Full wall numbers, exact commands,
manifests and numerical caveats are in that branch's
`reports/fpgs/G1_FINITE_SEAM_COMPOSE_20260913.md`. Fresh same-collision MJWarp
comparison is next; historical1.654/1.481 ratios below are not current new-tip
denominators.

G1 coordinate-register corrective experiment e8fa7955 also fails performance,
despite four physical controls/card passing: currentbd1 25.533741->28.504674ms
RTX and34.814945->46.843395ms GB. Strict nodes attribute the loss to solve,
8.869/17.880ms versus recent original5.867/5.764ms; other owners stay near flat.
It removes the >64 tail but adds dense43 dot work, replicated projection and
large static code. No hardware-counter attribution or third register mapping.
Evidence: `/tmp/fpgs-g1-coordinate-register-AodDkeuk/NODE_CAUSE.md`.

ANYmal complete directL117/compact-root-leg source eec1 passes current/held,
two-leg/warm and actual constructor/fallback/graph/reset controls on both
cards, after preserving and correcting test API setup errors. Whole loss:
9.244660->14.167829ms RTX;9.521509->16.000075ms GB. Actual owner/all16384 valid/
status0/capacity/source/idle checks pass. Strict paired node attribution finds
compact solve4.079->8.641ms RTX/4.455->10.849ms GB. Repeated FFS/tag gathers
replace the original regular float4 schedule; factor/predictor saves nothing.
Retiring buffers also revived redundant full-J clears. One costed correction
is authorized: contiguous per-leg panels once/CTA, regular24-sweep contractions
and original active-J owner, not another small register/sync tuning.
Full evidence: `/tmp/fpgs-anymal-branch-physical-m2e7L8gI/NODE_FINDINGS.md`.

Allegro full warm shell c5ea is implemented and has its first paired GPU
result, not a speedup. One saved patch safely falls back because the old
spatial inward movement leaves sub-ulp gap slack; loaded test also omitted
the original lazy-publication API in BOTH arms. One gap-aware interior
selection correction and the test API correction are underway. All complete
warm/patch/cold-fallback costs must be measured. No achieved4x claim.

## Current comparison and diagnosed losses (18:13 checkpoint)

### Resumed until-4x work (22:40 checkpoint)

The user explicitly resumed optimization until4x across all representative
tasks. The current G1 backend reference remains the original-query19099
comparison below; the new register-Gram candidate is slower and not promoted.
Whole physics, environment wall, incremental FPGS gain and corrected-MJ ratio
remain separate denominators. No new backend sweep is implied by this entry.

Frozen G1 candidate `5bb57dbbf84bbb60677a80acab7f9bc2ec4e79bb` replaces sparse
GS with on-chip32/64-row Gram/residual owners and original >64 fallback.
Four selected CUDA tests per card pass, including full-step/graph lifecycle,
actual current/held operators, reservation and0/31/32/33/64/65 row boundaries.
An explicit source-pinned in-memory parent adapter adds REGISTER_RESIDUAL=1
to the recorded child environments; the old paired parent's environment
cleaner would otherwise silently remove an inherited outer flag. Its source
file is unchanged, but execution is not claimed byte-original. Physical
manifest: `/tmp/fpgs-g1-register-residual-physical-paired-20260913-01/manifest.json`,
SHA `5c3fa283213aeb31daa56834be5c47ca3d2133f51b66a478d39e80bbbe80043b`.

The complete paired A/B, basebd1cc095 versus5bb57, retains SPARSE/CELL/PACK1,
16K seed0,200 warmup,40 wall and40 whole-physics steps, fixed Lab53ee and
calibrated capacities. Only REGISTER_RESIDUAL changes0 to1:

| GPU | Baseline physics | Register Gram | Baseline/candidate |
| --- | ---: | ---: | ---: |
| RTX PRO6000 |25.499776025ms|27.726071850ms|0.919704x|
| GB300 |34.869894625ms|44.424335450ms|0.784928x|

All four captures exit0, eight boundaries pass, sparse valid/status and
capacity checks pass, and final source/idle guards pass. Manifest:
`/tmp/fpgs-g1-register-residual-live-paired16k-20260913-01/manifest.json`,
SHA `4dd41f0133cb1bbed2ceeca99323c8fb7dd9a4d18cc823954b7da9aec2f3598c`.
This is a measured loss, not a numerical rejection. Existing paired node
captures are complete; exclusive attribution is being read before selecting
a correction. No hardware-counter stall/occupancy explanation is asserted.

Allegro FP32-only BSP e25992 and subtree10a0b5 completed earlier physical and
whole discovery gates. The later subtree result is17.037384900/17.031463175ms
FPGS against63.797036100/82.124776150ms corrected native-MJ,3.7445322/4.8219449x.
Its difference from the separate FP32 round is-0.0779/+0.1043ms, not a material
gain or matched incremental A/B. No promotion. Detailed source/checks:
`/tmp/fpgs-convex-bsp-subtree-allegro-backends-paired16k-20260913-01/RESULTS.md`.

Independent source review corrects the subtree cost model: the historical CPU
GJK corpus is query-major, so consecutive32 support calls are not simultaneous
warp lanes. Its old contiguous32 proxy cannot predict GPU savings. Subtree
fallback removes about92% of fallback dots, but only12--13% of an optimistic
plane-plus-dot scalar proxy; ordinary tree traversal and GJK/MPR/manifold work
remain. Current source/resources do not establish an occupancy cliff or exact
timing attribution. No further support-mask tuning is funded from that census.

The new ANYmal branch-response candidate is being implemented independently:
packed reverse L117, compact root/leg contact coordinates, unchanged complete
EX1/parallel24 law, and explicit original8 fallback. It must retire dense
factor production rather than convert/duplicate it. Full card and physical
contract: `/home/octi/Projects/newton-fpgs-anymal-branch-response-20260913/reports/fpgs/ANYMAL_BRANCH_RESPONSE_20260913.md`.
This is a hypothesis, not an implemented or measured gain. The separate
Franka/Kuka public-state consumer review found that skipping intermediate
publication alone cannot clear the10% whole screen even under an unrealistically
free-owner assumption; no publication micro-optimization is being implemented.

The earlier **1.403934x RTX / 1.934297x GB** sparse-plus-finite result is
FPGS versus cell-only FPGS, not FPGS versus MJWarp; that finite version remains
on physical hold. It must not be presented as another improvement in the
backend ratio or included in a promoted baseline.

A fresh G1 comparison uses clean `19099e91` on both backends, with shared
current-height rejection and packed original terrain queries, without finite
queries. FPGS additionally enables sparse43; MJWarp retains its corrected
solver and external Newton collision. Fixed Lab53ee, substeps and iteration
allowances are unchanged. This is one 16K seed0, 200-warmup, 40-profile-step
discovery round, not a repeated promotion result or full RL throughput.

| GPU | FPGS physics | Corrected MJWarp physics | MJWarp / FPGS |
| --- | ---: | ---: | ---: |
| RTX PRO 6000 | 25.563279200 ms | 42.293261875 ms | 1.654453701x |
| GB300 | 34.889251200 ms | 51.687804450 ms | 1.481482195x |

Separate environment-wall times are FPGS/MJ 39.507002401/56.667819526 ms RTX
and 48.788600626/65.323011951 ms GB. All four children exit zero; both checked
boundaries pass, states are finite and MJ warning masks are zero. The original
backend owner completes after its final source guard; this older schema does
not contain a separate final-guard Boolean. Fresh compute-app queries are
empty. Source and results: `/tmp/fpgs-g1-packed-original-shared-mj-paired16k-20260913-01`,
manifest `f523aa0701b0d93bd5a39cab7e8fdd7e2ba2706d67f68915ce798142d670497f`.

The same-GPU September11 handoff capture `compare_gpus_r0rvbs8o` records
clean Newton31cf87f46 FPGS at 40.961346667/67.864682667 ms, physics-only,
with the same two GPU UUIDs. This is a lower historical absolute-time trend,
NOT a matched cumulative gain: the older Lab, capacities, three versus40
profile samples and capacity-check coverage differ, and there is no matched
old MJ arm. It neither establishes the user's recalled fpgs-main1.5x pin nor
licenses multiplying incremental gains. Current G1 remains about1.5--1.65x
corrected MJWarp, far below4x. Shared collision gains benefit MJWarp too.

Two sparse-only successors are physically tested but not promoted:

- Level factor scheduling `a8f8c62d` reduces barriers and indexing while
  retaining all factor products. Four controls/card pass, but whole physics
  improves only 28.971685025 to28.696025300 ms RTX and40.883787725
  to40.436258075 ms GB (about1%). No mapping sweep is funded.
- Implicit limit prefix `95bdb5ae` also passes four controls/card, but whole
  physics loses on RTX: 29.088434225 to30.159742475 ms; GB changes
  40.926988300 to40.513590675 ms. Source-identical three-step node attribution
  shows prefix exclusive cost dropping2.243477 to0.171733 ms RTX, while
  complete GS/decode grows5.916565 to9.244272 ms. Retained work changes only
  +0.007991 ms. Removing stored Z adds repeated dependent inverse-index/shared-W
  gathers inside GS; that is a source-supported mechanism, not isolated
  hardware-stall attribution. Full40 timing remains the throughput authority.
  Node-owner auxiliary-analyzer failures are preserved; all four captures
  pass independent strict correlation and final source/idle checks. Details:
  `/tmp/fpgs-g1-implicit-limit-prefix-L6rKzH/NODE_CAUSE.md`.

Allegro exact BSP `c26ac5fa` passes seven native geometry/factory controls per
card but is slower than the prior accepted discovery: full FPGS is
20.736645200/19.332049075 ms, versus prior17.531317325/18.018165700 ms.
Source-matched nodes localize the loss to covered MPR/GJK/manifold work:
2.924756 to6.402575 ms RTX and3.689429 to4.942409 ms GB. PTX loads six double
limbs and stores nine local doubles even before the FP32 fast decision.
Corrective `e25992ce` removes the FP64 device predicate/descriptors entirely,
retains the conservative FP32 filter and uses the unchanged original support
scan on ambiguity. Ten CPU controls and all18 offline target/factory builds
pass. Paired CUDA controls and complete timing are next, not a claimed gain.
The existing capacity helper adds only this reviewed provider pin; allocation,
constructor checks, runtime gates and benchmark protocol are unchanged.

Finite-face correction `caece007` falls back to original queries for a narrow
positive-clearance top-face class before the common writer/reducer. Four
targeted controls/card pass, including three reduced rebound repetitions.
Whole cell-only FPGS changes37.941748175 to36.886608050 ms RTX (1.028605x)
and52.068057850 to41.787966100 ms GB (1.246006x), all capacity/source/idle
checks passing. It is not stacked with sparse/packing and is not promoted.
Remaining sliding impulses suggest an unsampled non-upward contact or another
impulse defect; internal triangle-edge timing is a hypothesis, not a proven
fault. No friction clamp, tolerance relaxation or blind normal override was
made. Full cost and limitations:
`/tmp/fpgs-heightfield-finite-qualification-J9kBvfGP/RESULT_FACE_COST_AND_SLIDING.md`.

The sections below preserve earlier checkpoints and their original pending
states; this section supersedes their current-status wording.

## Complete physics results so far

Times are milliseconds per batched environment step, including all four
physics graphs. They are means over 40 profiled steps following 200 warmup
steps, with 16,384 worlds and seed zero. Independent 40-step synchronized
environment-wall windows are separate; neither measures full RL training.

| Candidate | RTX accepted / candidate | RTX speedup | GB accepted / candidate | GB speedup |
| --- | ---: | ---: | ---: | ---: |
| G1 single-factor native response | 43.409484 / 41.665184 | 1.041865x | 72.285541 / 70.348522 | 1.027535x |
| G1 current-height cell rejection | 43.513016 / 37.867363 | 1.149090x | 72.398981 / 52.165458 | 1.387872x |
| Kuka complete live kinetic path | 12.778658 / 11.274877 | 1.133374x | 12.183984 / 11.740153 | 1.037805x |
| Kuka kinetic path, shared notification reads | 12.836523 / 11.305150 | 1.13546x | 12.148953 / 11.984799 | 1.01370x |

These candidates are separate, not stacked. Do not multiply their speedups.
Physics values come from graph profiling, not sums of separately optimized
sections. The single-factor experiment's first implementation was slower;
node attribution and one targeted response-kernel replacement recovered the
small gain shown. It remains default-off and below the structural milestone.
See `G1_SINGLE_FACTOR_20260913.md` for the complete loss diagnosis and controls.

Cell rejection removes separated heightfield-triangle work before expensive
contact queries; it adds current corner-height reads but no cache, queue,
buffer or launch. Its environment wall times improve 58.465564 to 50.948728
ms on RTX and 85.902860 to 67.300094 ms on GB. Seven focused tests pass on CPU
and each GPU. Two actual 96-pair convex-mesh fixtures on each GPU, including
three graph replays, preserve complete bidirectional contact tuples: 700/878
contacts with zero observed point/distance/normal error.

Three balanced alternating live rounds now reproduce the terrain gain. The
following are medians of three independent 40-step means, not pooled samples:

| Device | Baseline physics | Cell rejection physics | Physics gain | Baseline wall | Cell rejection wall | Wall gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX PRO 6000 | 43.61139235 | 38.01942965 | 1.147081709x | 58.017999775 | 51.641378400 | 1.123478915x |
| GB300 | 72.339710575 | 52.1157923 | 1.388057389x | 86.1398189998 | 66.6752308243 | 1.291931320x |

All 12 repeat captures passed the original capacity, source and final-idle
checks. This establishes repeatability for this workload, not complete
behavioral qualification or a new MJWarp ratio.

Kuka's complete path now covers real eager calls, per-world gravity changes,
reset-before-first-graph-replay and live MF rows. Its graph-reset failure was
traced by memcheck to private banks allocated during capture, then corrected
by constructor allocation of the exact two state/two directed-call banks.
No extra physics call or Lab workaround was added. However, synchronized
environment wall time **regresses**: 36.684009 to 37.468095 ms RTX and
35.243540 to 38.937888 ms GB. Increased reset host ranges are a correlation,
not disjoint causal attribution. Both the regression and small GB physics
gain are under investigation. Boundary status/finite checks do not establish
live contact convergence or per-replay fallback counts.

The reset attribution identified duplicate topology downloads in the native
notification owners. Correction `ad42fea95` shares one callback-local set of
eight current topology reads between owners, retaining every comparison and
all gravity checks. It does not cache across notifications or change physics.
All 40 candidate reset steps drop from three topology-read sequences to one:
16 copies and 33.947672 MB removed per reset. In fresh whole A/B, RTX wall time
improves 36.113675 to 34.227748 ms (1.05510x), while GB wall still regresses
35.674105 to 36.905660 ms (0.96663x). The GB physics gain remains near flat;
the candidate is not promoted. Current node attribution is diagnostic work,
not another claimed gain.

## Complete-path losses and next structural tests

Franka's complete contact-kinetic replacement passed physical tests on both
GPUs, including loaded current/held operators, actual eight-velocity
decoding and graph reset controls. But its complete restored-state cost
test failed: 4.513856 ms RTX / 4.693696 ms GB, against absolute 10%-gain
acceptance caps of 3.164793929 / 2.813527747 ms. These are replay fixtures,
not fresh 16K task throughput or MJWarp ratios.

Node attribution explains the failure: contact preparation costs
1.004384 / 1.151264 ms and the joined solve/finish costs
1.499605 / 1.759104 ms; the unchanged paired branch is not inflated.
An optimistic correction restoring the older solve cost with zero bridge
cost still misses both caps. This representation is stopped: a solve-only
tuning pass cannot recover the necessary complete-path gain. Physical
success is not performance success.

Two independent G1 structural paths have reached different gates:

- Direct pair-owned terrain queries retire the midphase launch and global
  triangle-triple writes/reads, retaining the same current-height rejection,
  MPR/GJK, manifold writer and reducer. Six tests on each GPU and both actual
  96-pair fixtures plus graph replay pass; 700/878 contacts are retained,
  maximum point/distance error 1.86e-9 / 3.73e-9 and normal error zero.
  The first full-task A/B fails the performance gate: RTX
  37.975026650 to 44.056320125 ms (0.861965469x), GB
  51.958921700 to 86.469124850 ms (0.600895658x). All four captures pass
  the capacity/source/idle checks. This is a work-elimination hypothesis
  defeated by its current execution layout, not an accepted optimization.
  Complete node attribution localizes the loss to the merged query itself:
  13.346504 / 56.375385 ms, versus the old midphase plus query
  7.264690 / 22.088587 ms. Exact current-cell enumeration on the saved actual
  geometry finds only 37.05% / 37.02% active query lanes, versus the old
  globally compacted stream: 2.699 / 2.702 times as many warp query entries.
  Registers also grow from 168 to 216 / 224. Removing global compaction
  removed useful work batching. These counts explain a concrete mechanism,
  not measured hardware utilization or a complete causal time decomposition.
  The path is stopped; there is no credible correction within its complete
  RTX replacement budget. Diagnosis is preserved in branch commit d05476a3.
- Sparse G1 dynamics replaces dense factor storage, current-force prediction,
  dense contact rows/responses, GS and velocity decoding as one representation.
  The static G1 graph needs 434 factor entries rather than 946 dense lower
  entries; a possible two-endpoint contact needs at most 18 kinetic entries.
  Those are structural counts, not runtime evidence. The first actual full
  solver test exposed omitted current drive-mass terms in serial mode:
  its immutable drive map is deliberately absent, while the isolated fixture
  had supplied it. Correction 8e8146 reads the existing compact current drive
  rows and removes the stale old augmented-H tail. The same unchanged
  tolerances now pass both physical methods on both GPUs, including actual
  full-step serial and parallel configurations. Test-only 3f539fae then fixes
  the continuing two-state graph test; both serial and parallel continuing
  loops pass on both cards. The complete warmed assessment now reaches all
  final metrics in all 16 selected cases: predictor, final velocity, impulse,
  momentum and friction-cone gates pass. Two RTX world7410 cases fail a
  stricter reconstructed net-response component comparison. An independent
  original-row CPU control fails that same comparison too; this is evidence
  about the comparison's conditioning, not an original CUDA solve replay or
  a waived numerical gate. Earlier tests incorrectly stopped on intermediate
  coefficient rounding before reaching physical checks; the successor records
  those comparisons and completes every finite supported case before deciding.
  Crucially, the first full 16K live attempt fails during warmup with sticky
  sparse status5 (factor bit1 plus row-topology bit4). No candidate timing or
  gain existed at that point. Eager execution with post-refresh synchronization does not
  reproduce the failure over the complete warmup. A graph-preserving observer
  then establishes the first failure as **only row-topology bit4**: RTX graph
  433, world6097; GB graph131, world14531. The later factor failure is downstream:
  the predictor deliberately propagates NaNs after any sticky error. Thus
  factor conditioning is not the first fault. Allocation records a 1/3-row
  decision, while the original metadata producer separately recomputes friction
  admission. A first-error device snapshot now proves that seam corrupts rows
  on both cards: contact slot30 reserves one row but metadata writes three,
  overwriting the next three-row packet at slot31. The damaged parent/type
  topology triggers status4. Allocated intervals cover the prefix correctly,
  all capacity flags are zero, and inputs are finite. This is not undersizing.
  The shared producer fix54fc1e09 consumes the allocator's existing recorded
  extent; sparse caller3bae8f77 binds that argument. Geometry, friction law,
  thresholds and buffers are unchanged. Both opposite-admission regressions
  fail on the old producer and pass after correction. Captured raw geometry
  and neighbor-order CPU replays pass, with controlled materials explicitly
  scoped as extent tests. Subsequent paired CUDA physical checks and live cost
  are recorded below. Accepted G1's ROWS_MASKED path uses a different producer; this does
  not establish corruption in its frozen reference. No pivot clamp, status
  relaxation or tolerance change is authorized.

Post-cell G1 node attribution locates the remaining work (exclusive diagnostic
milliseconds, RTX / GB): collision 9.364628 / 24.030058; dynamics
6.821106 / 6.542038; row/response 11.792834 / 11.849121; GS
6.257073 / 7.560128; public-state publication 2.406199 / 1.611733.
The generic triangle-query owner alone costs 3.922660 / 17.342261 ms.
Its captured register count is 168 on both devices. A profiler local-memory
field of zero is not proof of no local traffic: cached PTX declares a
432-byte local depot. No memory-bound/compute-bound claim is established.

## Corrected shared-collision MJWarp comparison

The terrain change is generic Newton collision work. A fresh matched run
enables it on **both** FPGS and corrected MJWarp, on the same candidate commit.
It speeds up MJWarp too; do not use the older, slower MJ denominator.

| Device | FPGS physics | Corrected MJWarp physics | MJWarp / FPGS |
| --- | ---: | ---: | ---: |
| RTX PRO 6000 | 37.881833475 ms | 45.697852650 ms | 1.206326x |
| GB300 | 52.083977625 ms | 57.559506800 ms | 1.105129x |

Environment wall times are 52.234619399 / 59.487608250 ms on RTX and
66.320000250 / 72.140569401 ms on GB (FPGS / MJ). This is one fresh round,
not a repeated promotion result. G1 still needs major FPGS-specific work
elimination, as well as further shared collision improvement, to approach 4x.
Other tasks' historical corrected-MJ ratios are not silently refreshed by
these new G1/Kuka FPGS-only discoveries.

The finite-query successor has now also been compared fairly with both
backends using CELL1/FINITE1 on clean70f17773, whose native code is882e468a:

| Device | FPGS physics | Corrected MJWarp physics | MJWarp / FPGS |
| --- | ---: | ---: | ---: |
| RTX PRO 6000 | 35.598116925 ms | 43.155616100 ms | 1.212301x |
| GB300 | 38.107064225 ms | 42.935425875 ms | 1.126705x |

Environment wall times are49.500687650 /57.684546450 ms RTX and
52.346774825 /56.868676926 ms GB. Actual finite dispatch and handled live
triangle entries are observed on both backends at the existing untimed
boundaries. MJ uses external Newton contacts, the correction guard passes,
warning mask is zero, and all original capacity checks pass. This is one
discovery round, not evolving-task physical qualification or promotion.
The large GB collision saving benefits MJ too; it cannot be credited as an
equally large improvement in the FPGS/MJ ratio.

## Fresh corrected Allegro comparison and next work screen

One fresh paired round uses the existing rejection-only collision candidate
fba9fead, compact calibrated FPGS allocation, and corrected native MJWarp
collision with njmax=112 / nconmax=22. No new solver optimization is included.
Both backends retain their existing substeps and iteration allowances.

| Device | FPGS physics | Corrected MJWarp physics | MJWarp / FPGS |
| --- | ---: | ---: | ---: |
| RTX PRO 6000 | 17.531317325 ms | 63.336101650 ms | 3.612741x |
| GB300 | 18.018165700 ms | 82.315187375 ms | 4.568455x |

Environment wall times are 25.493010526 / 71.681469025 ms on RTX and
25.901357926 / 90.487337676 ms on GB (FPGS / MJ). All four captures pass
the capacity/source/final-idle checks. This is a discovery round, not a
new repeated 4x-across-tasks result. RTX needs a further 1.697292 ms net
whole-physics reduction to reach its current 4x denominator.

A separate node capture accounts for all 12 physics roots and 2,064 nodes
per GPU, with no auxiliary roots or unproven correlations. Exclusive
row/response plus solve cost is 8.445006 / 8.959829 ms; collision is
3.955239 / 4.871369 ms. Broad phase alone is only .189877 / .219360 ms.
These diagnostic intervals do not replace the clean throughput means.
Existing Allegro finger-block reduction was already tested at roughly 3%
of solver time and withdrawn; it is not a new structural opportunity.
The complete row-production study is now a no-go for implementation on the
available evidence. Plain contact-row work totals only 1.656873 ms exclusive
on RTX, less than the 1.697292 ms needed even if it became free. Including
velocity/position prefix construction and necessary clears expands the full
removable union to 2.389896 ms, leaving only .692604 ms for all new production,
implicit-prefix consumption and fallback work. No demonstrated complete path
fits that allowance. Prescaling joint velocity cannot be replaced by a final
clamp: it affects FK, bias forces and prediction. This is a bounded feasibility
decision, not a proof that Allegro cannot reach 4x.

Kuka's first-hit successor has completed physical and first live testing: current endpoint
wrenches remain compact packets until a changed residual first demands a
held physical response. MF-positive worlds retain the existing complete
path. Its planned saving must include response preparation, solving and
all retained allocation/CSR/prefix/fallback work. CPU physical success and
the fraction of demanded rows are not a speedup claim.

Its first CUDA test found a small real GB velocity drift. Same-input diagnosis
isolated repeated endpoint-motion residual accumulation, not the cached inverse
action. Correction e96aaab3 reuses the existing contact packet panel for
physical J only after first demand; never-demanded rows still avoid J/Y.
All packet reads finish before overlapping stores and valid2 publishes last.
Under unchanged physical bounds, both GPU suites now pass. The previously
failing world210 error drops from 3.206e-5 to 9.596e-7; across four historical
512-world cases, maximum original-current-geometry velocity error is 5.359e-6.

The complete live performance result nevertheless loses:

| Device | Original kinetic physics | First-hit physics | Original / first-hit | Original wall | First-hit wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX PRO 6000 | 11.299988 ms | 12.386436 ms | 0.912287x | 34.424091 ms | 35.182237 ms |
| GB300 | 11.666766 ms | 12.728927 ms | 0.916555x | 36.337624 ms | 35.553964 ms |

All four captures pass source, actual kernel-owner activation, capacity and
final-idle checks. This is not a new MJWarp ratio or a promoted optimization.
The completed source-identical node trace explains the loss: exclusive row
and response work saves only .032820 / .235318 ms, while the joined solve
grows 1.043361 to 2.484227 ms RTX and .945280 to 2.577408 ms GB. The original
triplet already prunes joint support and cancels common-arm terms; it did not
perform dense 29-by-29 work for every row. The new packet retains raw geometry
and three-row metadata, reducing writes from 87 Z floats to 48 packet floats,
while its physical-velocity consumer adds first-demand actions and recurring
endpoint-motion work. No credible consumer-only 10% whole-physics correction
fits the measured allowance. This mapping is closed, with numerical findings
and complete causal evidence pushed in experimental branch commit13f17b07;
its runtime is identical to measured e96. A larger producer/row-state ownership
study is separate, CPU/source-only work, not a funded implementation yet.

## Finite terrain query: physical diagnostics and first whole-task gain

An isolated successor of cell-only a994 preserves the globally compacted
triangle stream and replaces generic GJK/MPR plus iterative manifold discovery
for recognized immutable cuboids. Separated distance uses three segment/AABB
minimizations and eight vertex/triangle face tests. A nonnegative terrain-local
separation-vector Z certifies the same witness for the finite downward prism;
side/bottom or ambiguous cases retain the complete original query. Penetration
uses finite-prism SAT and clipped finite top patches, never an infinite plane.

At most five contacts per triangle flow through the unchanged writer/reducer.
Far queries retain a witness for original reducer semantics. Handled entries
encode their own triangle index reversibly; the separate original fallback
scans the full live prefix. No new maximum-sized queue or counter is added.
Static descriptors are exactly per shape. The extra scan and both dispatches
must be charged in complete timing. The changed manifold law needs physical
qualification; original contact-count equality is not imposed.

Independent CPU checks on both saved 96-pair scenes pass, including complete
direct and reduced pipelines and nonempty-to-empty-to-regrown prefixes:

| Fixture | Triangle queries / analytically handled | Direct contacts old / new | Reduced contacts old / new | Maximum minimum-separation change |
| --- | ---: | ---: | ---: | ---: |
| RTX saved geometry | 1208 / 1205 | 700 / 648 | 199 / 198 | 1.479e-6 m |
| GB saved geometry | 1190 / 1190 | 878 / 742 | 239 / 223 | 7.066e-7 m |

Every previously contacting shape remains represented. Independent convex-QP
distance and finite-surface witness checks pass; maximum surface error is
1.895e-7 m. Eleven additional native synthetic cases cover finite edges,
rotation, zero separation, deep overlap and required generic fallbacks.
These are CPU diagnostic results, not convergence over an evolving task or
measured speedup. The first paired GPU run passes six control methods and
native synthetic geometry on each card, but fails actual-scene restore/replay
assertions that conflate equal manifold counts with marker lifetime. The
failure is preserved. Test-only successor70f17773 separates logical stream
and marked-entry identity from manifold cardinality, applying the same
empty/regrow and three continuing graph replays to the original query too.
Both GPUs pass all eight methods with no skips. Direct contacts stay exactly
700/878 original and660/761 candidate through every restore/replay. Only the
reducer's redundant counts vary, including for the original query itself;
the original count-gate failures remain recorded. Logical triangle sets,
marked-entry sets and contacting-shape coverage remain exact. Maximum
lifecycle minimum-separation change is1.001e-7 m; all2398 native queries/card
reach the independent convex-QP/witness audit, with maximum finite-surface
error2.063e-7 m. This diagnoses a test assumption, not a runtime marker fix.

The first complete 16K A/B passes source, actual activation, capacity and
final-idle checks on all four captures:

| Device | Cell-only physics | Finite-query physics | Physics gain | Cell-only wall | Finite-query wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX PRO 6000 | 37.955425075 | 35.672949900 | 1.063983x | 51.614142 | 50.134362 |
| GB300 | 52.037375725 | 38.089826200 | 1.366175x | 66.959269 | 52.011439 |

Net savings2.282475 /13.947550 ms exceed the discovery budgets2 /10 ms.
This is a real complete-path discovery, not repeated gain or full evolving
contact-convergence qualification. Later composition is recorded below.
Same-source diagnostic nodes now account for all retained work: query family
falls from3.911321 to1.776897 ms RTX and17.331656 to3.377930 ms GB, including
the new fallback scan. Collision union falls9.370526 to7.562649 ms RTX and
24.071890 to10.351359 ms GB. Both original node parents exit1 on auxiliary
graphs; separate strict readers account for all1260/1272 physics nodes plus
30 auxiliary nodes without converting those parent failures into passes.
The experimental replacement allowance is 1.922660 ms RTX for
2 ms net saving, or 7.342261 ms GB for 10 ms net saving, including fallback
scan and downstream cost changes. Do not add these separate savings to G1's
unqualified sparse solver or reuse an uncorrected shared-MJ denominator.

## Sparse solver, composition and packed-pair discoveries

After the recorded reservation fix, unchanged sparse physical controls pass
four methods/card, including both captured reservation faults. The direct
whole-task sparse-only discovery is37.784922400 to30.598367400 ms RTX
(1.234867269x) and52.137139700 to40.946308525 ms GB (1.273305008x).
Original capture/capacity/source/idle checks pass; sparse operators are
actually valid for all16,384 worlds with zero sticky status. Historical
warmed component-response caveats remain; no numerical tolerance changed.
Pushed sparse report tip954895ec preserves measured runtime3bae8f77.

Exact sparse+finite composition8ecd28b1 passes all12 existing methods/card.
One complete A/B against cell-only gives37.862898900 to26.969148400 ms RTX
(1.403933796x),52.027578975 to26.897414575 ms GB (1.934296653x).
Wall52.043408675 to41.402623575 /66.504235224 to41.987810076 ms.
These are FPGS-to-FPGS speedups, **not MJWarp ratios**. The separate report is
pushed atd27575ef, branchooctipus/g1-sparse-finite-report-20260913.

The fresh same8ecd/CELL1/FINITE1 corrected-MJ comparison gives:

| Device | FPGS physics ms | MJWarp physics ms | MJWarp / FPGS |
| --- | ---: | ---: | ---: |
| RTX PRO 6000 | 26.810854525 | 43.243381150 | 1.612905740x |
| GB300 | 26.988005350 | 42.908563300 | 1.589912361x |

Both actual collision owners and external Newton-to-MJ conversion pass;
MJ's correction guard passes and all warning/capacity flags are clear.
This is one discovery, now subject to the finite-manifold physical hold.
Generic collision gains also accelerate MJ; do not compare to the old slower
denominator. A historical fpgs-main1.5x recollection is not a matched source
or protocol and is being traced without inventing a cumulative speedup.

The packed-pair candidate6a56b017 changes only pure-heightfield execution:
old actual launch[384,128]/stride384 becomes[196608,1]/stride196608 under
the existing THREADS_X4 recipe. The global compacted triangle stream and
all queries/reduction/geometry/capacities remain. No tile-size sweep occurs.
Ten physical/geometry methods/card pass, including exact logical coverage
for nonmultiple pair counts, actual mapping, withdrawal/regrowth and graphs.
The first complete finite-only baseline A/B is35.627923625 to32.286811550 ms
RTX (1.103482255x) and38.122727300 to33.412042700 ms GB (1.140987627x).
Wall49.959500025 to45.380039851 /51.306246975 to46.402043401 ms.
All capture/activation/source/idle checks pass. This prices packing only;
it does not qualify the unchanged finite query's loaded physical behavior.
Exact packed compositionc76384a4 is prepared but further GPU stacking is
paused. Transfer to the original-query sparse path remains independent.

### Packing with the original query: independent gain

Clean candidate `19099e91b8de83b9fdbab3a59b54d52f93e502cd` transfers only the
packed launch to sparse `954895ec`, without the finite-query manifold code.
Eleven unchanged/targeted physical and geometry methods pass on each GPU,
with no skips, failures or errors. Both sparse operators and continuing
graphs are exercised. The fresh paired whole-task discovery gives:

| Device | Sparse + original query | Add packed launch | FPGS baseline / candidate | Absolute saving |
| --- | ---: | ---: | ---: | ---: |
| RTX PRO 6000 | 28.968070275 ms | 25.548476325 ms | 1.133847276x | 3.419593950 ms |
| GB300 | 40.855120725 ms | 34.689812850 ms | 1.177726755x | 6.165307875 ms |

Environment wall means are 43.598839825 to 39.787609050 ms RTX and
55.151633776 to 49.279069225 ms GB. All four captures retain clear original
capacity checks, 16,384 valid sparse operators and zero sparse status at
both recorded boundaries. Sources and final GPU idle pass. These are one
paired round, not repeated timing or full physical qualification. The RTX
baseline differs from the earlier 30.598 ms sparse discovery: the paired
28.968 ms baseline above is the authority for this incremental gain.

Physical manifest SHA256:
`2de80d65eb4b9160c785de69fa27d7af714912d700ea257da826a9ab231847a5`.
Live manifest SHA256:
`5f171630dec4a06346ffd2b4cbf33210a2a404b3a225aa6a40ec20815df7dd6b`.
Directories are `/tmp/fpgs-g1-packed-original-physical-paired-20260913-01`
and `/tmp/fpgs-g1-packed-original-live-paired16k-20260913-01`.
The existing physical owner and sparse checked timing runner are unchanged;
no benchmark framework was added. The same-source, shared-collision MJWarp
comparison remains pending and must not reuse a slower old MJ denominator.

### Keep starting, incremental and backend denominators distinct

The quoted G1 1.403934x / 1.934297x is **cell-only FPGS divided by composed
FPGS**, not MJWarp divided by FPGS. Its matched MJ ratios are 1.612906x /
1.589912x, and those results are on the finite-query physical hold above.
Do not present either pair as accepted cumulative improvement from handoff.

The handoff record at Lab `2129e562` / Newton `31cf87f4` gives G1 39.014977 ms
FPGS and 50.002362 ms MJWarp, or 1.281620x, on RTX 5090. Its G1 graph scope
was mixed/unaudited and included an observation/sensor graph. It is not a
matched RTX PRO 6000 or GB300 physics-only denominator. The user's recalled
`fpgs-main` result near 1.5x has not been resolved to a matching source and
protocol. Neither a regression nor a cumulative qualified gain follows by
comparing that recollection with an incremental FPGS-to-FPGS ratio.

## Loaded-law finding and remaining execution studies

Finite loaded GateA01 failed setup before any physics: the small fixture's
automatic grouped lane count2 is unsupported. The identical-source A02 uses
the existing G1 recipe's original --fpgs lane environment, not altered physics.
All26 cases are attempted:18 records/card, eight unreduced impact cases fail
the authored dense100 capacity because50/46 raw contacts require150/138 rows.
Those are test-capacity failures, not a reason to inflate live G1 capacities.
Original and finite sliding cases also miss the authored Coulomb control;
neither is waived. Reduced scalar restitution, support, tilt, rotated-terrain
support and finite-border controls pass.

Crucially, reduced GB free-foot rebound gives finite spin9.670105 rad/s versus
original0.000001948, at identical authored q/qd, normal speed1.800000191 m/s,
ten contacts, mu0 and restitution0.6. Kinetic-energy ratios are0.391037452
versus0.360000076. Linear impulse/public-force closure remains good. The
equivalent force center lies inside the foot, so this establishes a large
asymmetric finite-eight response, not yet a geometric impossibility or its
cause. Contact distribution, residual and angular impulse are being diagnosed
using this same fixture. No changed tolerance, extra sweeps or promotion.

The targeted same-state rebound collection now localizes the discrepancy:
raw finite clipping already supplies near points on both sides of the foot,
but the reduced finite near-contact subset can cover only one side. The
retained reducer's spatial support slots exclude positive-depth contacts;
depth/voxel selection and packet ordering then matter at eight sweeps.
Captured-order CPU replay reproduces the bad response, and recovered
Jacobian/angular impulse agree with the actual solve and public force.
This is not evidence that force export or raw clipping is missing. A narrow
successor `caece007` hands positive-clearance, in-shell top-face manifolds to
the original query before writing or marking them handled. It preserves
the reducer, capacities and eight sweeps; additional query/fallback work
must be charged. CPU tests and offline compilation pass, CUDA pending.

Both NCU counter attempts fail withERR_NVGPUCTRPERM; jobs are reaped and
source/idle checks pass. No counters were collected and no hardware ceiling,
memory-bound or spill-traffic claim follows. No permission workaround is used.

Allegro exact normal-fan BSP CPU feasibility reaches122,926 geometric support
queries without failures; actual40/64-vertex hulls have maximum6/9 decisions.
Static exact-plane/tree storage is90,108 bytes. Adaptive predicates still
need genuine exact final-tier implementation, and changed support winners
require physical checks. A bounded complete split/coherent/manifold native
implementation is under way on a fresh fork branch, not a measured gain.

The recurring scaffolding problem is explicitly constrained: no further
benchmark-framework development, no new GateB runner work, and reuse existing
paired timing and physical tests. Added diagnostics must address a concrete
changed-law discrepancy. Preparation is not performance progress. Kuka's
earlier1.13546x RTX/1.01370x GB physics result remains a mixed discovery, not
zero work and not a promoted cross-task win; GB environment wall regresses.

Current local authorities: sparse live02 manifestb9e104c4; composed live01
b9542b7d; composed shared-MJ012aa1662c; packed physical015cd3a2fa and live01
c11dd204; loaded GateA02 is failed and preserved. Exact commands, hashes and
per-card checks are in the corresponding /tmp/fpgs-* manifests. All parents
for these completed runs are reaped with source and final-idle checks.

## Exact sources and completed owners

All runs use fixed Lab backend
`53ee6b44c2334341305dbdf385a3916c6b140799`, retaining prepared core/tasks/venv
`1d8feb82d17dbfab8f0772de56f84deae2cb7974`. Startup checks verify actual
Newton and fixed manager import paths. Original worktrees and captures remain
unchanged. Each listed parent and its children are reaped, with source and
GPU-idle checks passed. Capacity/sticky-overflow checks pass at all measured
boundaries; this is not a claim of complete trajectory equivalence.

- Accepted Newton: `50dfa28d3aabe51f1b5b75721450efac2c35c5f1`.
- Single-factor native response: `a4093d57`, subsequently documented with an
  additional unsupported-partial-warp admission guard in `0ad92797`. The
  experimental root branch through `0ad92797` is pushed to `ooctipus/newton`.
  Live parent `/tmp/fpgs-g1-single-factor-live-paired16k-20260913-02`.
- Heightfield candidate: `a994b1b24b6073dcb10e855c10e8197ee116ef14`, isolated
  `ooctipus/g1-heightfield-cells-20260913` branch, now pushed to ooctipus/newton
  and included default-off in the root branch as f0025014. The root branch
  was pushed through 0a16fb67 before this checkpoint; original handoff ancestry is retained.
  First A/B parent `/tmp/fpgs-g1-heightfield-cells-live-paired16k-20260913-01`,
  manifest `04d9b8f632e84677c577dc4516f464b362fafbd2f2258dac69b3fc2bfb283b7b`.
  Shared-MJ parent `/tmp/fpgs-g1-heightfield-cells-shared-mj-paired16k-20260913-01`.
  Physical parent `/tmp/fpgs-g1-heightfield-cells-checks-paired-20260913-01`,
  manifest `eed8c778a64dd85e97d49632837612352f888d48d1fb1d76502ef71b4cf44e18`.
  Balanced repeat parent
  `/tmp/fpgs-g1-heightfield-cells-repeat-paired16k-20260913-01`, manifest
  `1d92e218432ee0b2f0200ad7914aca9b2a2342d5bd80e804c890fa9502efa068`.
  Diagnostic node parent
  `/tmp/fpgs-g1-heightfield-cells-nodes-paired16k-20260913-01`, manifest
  `a4ab94f3d263523513bd4e771d784e468fc6042f5021c69159bd97c1ff43e2ad`.
  This node parent exited 1 solely because the old analyzer rejected the
  auxiliary graph. Both simulations passed; an independent process/correlation
  reader accounted for all 12 physics and three auxiliary graph roots, and
  final source/idle checks passed after the parent was reaped. Do not present
  the diagnostic parent as an exit-zero throughput run.
- Kuka candidate: `4f76552da4f8966f96c0d2270168613142c9bd14`; benchmark-only
  current-owner observers `c9b06e9e1b86c7afbe6e34c69c17944291db9404` in a fresh
  benchmark worktree. Parent
  `/tmp/fpgs-kuka-kinetic-live-discovery-paired16k-20260913-01`, manifest
  `365d1db9fc8a254af28e7395461f030a63da30e3b3f4a049e29b2923fb9fa1fa`.
  The 25 unique pinned source/artifact files were independently rehashed.
- Corrected Kuka candidate: `ad42fea95fd44cd5ec72afa1be1cce570b45109a`,
  benchmark `a3a5dfead5026efabae1a35f030a4fd0e7182e26`. Fresh A/B parent
  `/tmp/fpgs-kuka-kinetic-live-discovery-paired16k-20260913-02`, manifest
  `9b7562872914f4a6748ab03941468d4e259c826120fdf692dd0a6256a02cfe1f`.
- Franka complete-path candidate `0e0fc27e9bd04c419066e3a587235085b4b9af42`.
  Cost parent `/tmp/fpgs-franka-contact-complete-cost-paired16k-20260913-01`,
  manifest `2114a6cc16d1306d405bf19a72bb526aac496b8f9f7c06b0aa3d88cec20c5532`.
  Node parent `/tmp/fpgs-franka-contact-complete-nodes-paired16k-20260913-01`;
  diagnosis `/tmp/fpgs-franka-contact-complete-nodes-0QOS0f/DIAGNOSIS.md`,
  SHA256 `2b3466d9322ca5e1ae6a106827c275b7ab30eb2eb1da20a4f2aafd38243361a8`.
- Direct terrain candidate `c41f46e74a63fc0e1bb79a96636c42260dc63576`.
  Physical parent `/tmp/fpgs-g1-terrain-direct-physical-paired-20260913-01`,
  manifest `f9cbc7a5c0650a14bbcc04169bf465b75762381240d1bd933a29a7595c9b0c15`.
  Complete A/B parent `/tmp/fpgs-g1-terrain-direct-live-paired16k-20260913-01`
  exited zero and is reaped; the performance gate failed as reported above.
- Sparse physical correction parent
  `/tmp/fpgs-g1-sparse-factor-physical-paired-20260913-02` exited zero;
  both GPUs ran exactly two CUDA methods with zero skips/failures/errors.
  Its initial graph wiring was not a continuing state loop; test-only
  successor 3f539fae fixes that wiring and passes continuing-loop CUDA checks
  on both GPUs, but the later full live warmup fails as described above.
- Complete sparse warmed assessment parent
  `/tmp/fpgs-g1-sparse-warmed-assessment-paired-20260913-01`, manifest
  `19c3428105c0bef4c6c7d91923be4ea74557b37190b8b0259bc77ee321678e5b`,
  is reaped exit1 with final source/idle guards passed. All16 cases reach
  final metrics; the two net-response failures are retained.
  Whole live parent `/tmp/fpgs-g1-sparse-live-paired16k-20260913-01` is also
  reaped exit1/source-idle passed: both candidates fail the original warmup
  checker before timing, while both cell-only baselines complete normally.
- Sparse eager first-refresh diagnostic
  `/tmp/fpgs-g1-sparse-first-refresh-paired16k-20260913-01`, manifest
  `fd25a5191b9ee83fd0bc2b769297263bea7dd7048cf7090b67dd5dd560cb65ed`,
  is reaped exit1 because no failure was reproduced; source/idle checks pass.
  Graph-preserving successor
  `/tmp/fpgs-g1-sparse-graph-failure-paired16k-20260913-01`, manifest
  `2da8651b901db4dbefd2ee86f02340f38df3b8304fceda4f4e18dd559add5738`,
  is reaped exit0/source-idle passed: both cards observe the first status4.
  Diagnostic success means observing a fault, not physical/performance acceptance.
- Corrected first-hit physical parent
  `/tmp/fpgs-kuka-first-hit-corrected-paired512-20260913-03`, manifest
  `e55f826101eeb07ad949c6c9cb16f3d55762b4e6f2f3db433f65ece435dcc7de`,
  is reaped exit0/source-idle passed. Complete live parent
  `/tmp/fpgs-kuka-first-hit-live-discovery-paired16k-20260913-01`, manifest
  `d2357eaedc2563a0b06be327e99dc66cb4a040674cdfd067c185410560fa6c83`,
  is reaped exit0/source-idle passed, but the performance gate fails.
  Actual native e96aaab35bd5b61578bc8a77dc8633f7bc80c68d is compared with
  ad42fea95fd44cd5ec72afa1be1cce570b45109a, both kinetic1; first-hit0/1 is
  the sole differing feature flag. The source-bound untimed observer verifies
  actual packet, prefix and solve kernels rather than trusting the flag.
  Same-source node parent
  `/tmp/fpgs-kuka-first-hit-live-nodes-paired16k-20260913-01`, manifest
  `4819e3cb5fe2f26e6d5314ca26539bba50867351829093954d2160331973efd4`,
  is reaped exit0/source-idle passed. Four independent audits account for
  exactly12 physics roots and1920 nodes per arm/card. Full causal report is
  `reports/fpgs/KUKA_FIRST_HIT_RESULT_20260913.md` in experimental branch
  `ooctipus/fpgs-kinetic-first-hit-20260913`, report-only commit13f17b07.
- Finite-query physical parent
  `/tmp/fpgs-heightfield-finite-physical-paired-20260913-01` is reaped exit1,
  with final source/idle checks passed. Runtime and tests were clean882e468a;
  both cards run eight methods with zero skips, retaining the lifecycle
  assertion failures described above. No performance window was launched in
  that failed physical run. Test-only successor70f17773 passes paired physical
  parent `/tmp/fpgs-heightfield-finite-physical-paired-20260913-02`, manifest
  `29fceb8c974ba12bbb60c0ba7e83cda0645f7319338c8a70f9bfaba6f4049624`.
  Complete live parent
  `/tmp/fpgs-g1-finite-query-live-paired16k-20260913-01`, manifest
  `aaa5ea37c797b52b9a0b009db996125940e5102300ef3f92a73a7233372d2f31`,
  is reaped exit0/source-idle passed. The shared-MJ parent
  `/tmp/fpgs-g1-finite-query-shared-mj-paired16k-20260913-01` is likewise
  reaped exit0/source-idle passed. Both use unchanged native882e468a.
- Sparse first-row diagnostic
  `/tmp/fpgs-g1-sparse-row-failure-paired16k-20260913-01`, manifest
  `31816f1c7e3cac17a7abfca02f9705135c1296262408a4925eba720bb64979d7`,
  is reaped exit0/source-idle passed. RTX world2817 at graph410 and GB
  world5335 at graph368 reproduce the slot30/31 overwrite. Diagnostics retain
  raw IDs, neighbor order, current geometry and first-error row bytes, without
  host synchronization inside the physics graph. Fix source3bae8f77 is frozen;
  paired physical parent43599 is active at this checkpoint, not completed.
- Fresh Allegro parent
  `/tmp/fpgs-allegro-fixed-backends-paired16k-20260913-02` exited zero.
  Attempt01 is preserved: MJ construction failed because the wrapper tried
  to set a Newton collision capacity on the native-MJ None configuration.
  Removing that unsupported override is the only retry change.
- Allegro node parent `/tmp/fpgs-allegro-fixed-nodes-paired16k-20260913-01`,
  manifest `8819292cae82b902055dbb63bb611b4b65d0af1d4fb64f5c356d0d71c969918f`,
  exited 1 in the final ratio formatter because this was an FPGS-only
  capture. Both simulations and original analyzers passed; root independently
  verified final source/idle and disjoint node membership with
  `/tmp/fpgs-allegro-fixed-node-owner-EE86jIM8/audit.py`. The failure is
  preserved; no GPU rerun or rewritten exit status was used for formatting.

All feature modes remain explicit/default-off. Next bounded work is corrected
sparse G1's complete cost, finite-query attribution and physical qualification,
and Allegro's exact static-seed convex support feasibility. Kuka's larger
producer ownership study is conditional, not an implemented gain. Keyboard's residual termination
discrepancy remains parked at the user's request.
