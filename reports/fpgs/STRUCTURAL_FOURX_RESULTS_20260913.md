# Structural window: integrated discoveries

Checkpoint: 2026-09-13 14:12 UTC. Work continues toward the 24-hour window in
`STRUCTURAL_FOURX_20260913.md`; the 4x target is **not achieved**. These are
discoveries, not replacements for accepted baselines except where repeat
evidence is explicitly given. No candidate has met the complete 4x target.
No Isaac Lab source, timestep, substep or iteration allowance was changed.

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
  full-step serial and parallel configurations. Continuing graph replay and
  warmed task-state operator gates remain before complete-path timing.

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
The current study is a complete row-production/consumer replacement, not
a broad-phase or finger-dot-product polishing sweep.

Kuka's first-hit successor is also in physical testing: current endpoint
wrenches remain compact packets until a changed residual first demands a
held physical response. MF-positive worlds retain the existing complete
path. Its planned saving must include response preparation, solving and
all retained allocation/CSR/prefix/fallback work. CPU physical success and
the fraction of demanded rows are not a speedup claim.

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
  was pushed through 7a3b9abc; original handoff ancestry is retained.
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
  successor 3f539fae fixes that wiring and is not yet GPU-qualified here.
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

All feature modes remain explicit/default-off. Next bounded work is Kuka
first-hit physical/live gates, G1 complete sparse physical/live gates,
and Allegro's complete row-production feasibility. Keyboard's residual termination
discrepancy remains parked at the user's request.
