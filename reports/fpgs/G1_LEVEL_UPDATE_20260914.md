# G1 right-looking level updates

Design checkpoint before code, 2026-09-14. Experimental, default off:
`FEATHER_PGS_SPARSE_LEVEL_UPDATE=1`, with the original sparse owner enabled.
Base is frozen `da85acb1d1a547ef60a4bc6b45bd8096610f6a42`; retain its measured
parallel-limit prefix, original global rows and all current collision owners.
This is one funded structural correction, not an inverse or solver rewrite.

## New cause versus the closed level experiment

Prior left-looking `a8f8c62d` passed physical controls but saved only 0.275660 ms
RTX / 0.447530 ms GB whole physics. Its 434 final-entry gathers retained 2,242
products and introduced long dependent accumulations: maximum per-phase chain
lengths sum to 576, with 37--42 terms on late root entries. The saved result
does not isolate the unchanged inverse tail, so it does not prove inverse
construction dominated. Preserve that result and its qualification at
`/tmp/fpgs-g1-sparse-level-schedule-tR9MaSnB/RESULT.md` and the source/PTX
diagnosis `/tmp/fpgs-g1-implicit-limit-prefix-L6rKzH/LEVEL_CAUSE.md`.

The proposed correction is right-looking. Complete independent diagonals,
scale their columns, then give each remaining Schur destination one owner
which gathers only this level's contributing pivots. No factor atomics or
long all-predecessor accumulation is introduced. Current 43-pivot code visits
18,662 packed entries for 2,242 actual products and uses 129 factor barriers.
The actual USD/make_plan topology has 15 levels, widths
`8,8,4,4,4,4,2,2,1,1,1,1,1,1,1` (not the 18-coordinate contact-union bound).
The new exact schedule visits 1,142 Schur destinations, retaining all 2,242
products, 391 divisions and 43 square roots, with 45 factor barriers.
Maximum gather lengths per level sum to 42; no gather exceeds eight terms.
This addresses the previous long-chain cause, not a block-size tuning sweep.

## Complete cost and preserved ownership

The current complete refresh owner costs about 3.185 ms RTX per environment
step, including source contraction, augmentation, factorization and W
publication. Target at least 1 ms net whole-physics saving: an approximately
2.185 ms complete refresh screen, with no downstream growth. Fewer inactive
visits, short gathers and 84 fewer barriers make this plausible, not proved;
the unchanged work may still dominate. The root explicitly funds this
bounded correction below the usual ten-percent prioritization threshold.

Added work is immutable exact-size topology metadata and its indexed reads,
shared across worlds. Retain the original 434-entry scratch and held W,
43 source contractions, 434 projections, current R/drive K, interval/request
mask, serial/parallel drive readiness, sticky validity/status and notifications.
Retain the inverse tail's 2,242 products, 434 divisions and 434 W stores.
Predictor, Z/incident/support, current geometry, restitution, original finite
eight sweeps, friction/sibling/early-stop law, decode and publication are
unchanged. Unsupported constructors keep their original representation.
No second world-scaled matrix, queue, extra kernel or inverse mapping.

## Gates and stop condition

Add a regression first to the existing sparse-factor tests: exact schedule
coverage and dependencies, current actual-USD augmented H reconstruction,
inverse action and flag/owner admission. Reuse existing native current/held
geometry/action and complete two-step continuing-graph selectors on both
GPUs. Keep all original numerical and physical tolerances and failure flags.
CPU/native readiness precedes any root-owned GPU lease.

Then promptly run the original paired whole 16K, seed 0, 200 warm / 40 wall /
40 graph protocol against frozen da85, fixed Lab53ee, unchanged G1 capacities
and physics. Baseline and candidate both enable the proven parallel-limit
prefix; only the candidate enables level updates. Whole cost decides the
result. If the approximately 1 ms hypothesis misses, inspect this complete
refresh owner once; no further mapping or inverse campaign without new cause.

CPU study input: cached `g1_minimal.usd` SHA256
`9dfe7a710aa791e49abf2d9ea74ad3163e291f02f21f59bda9bfcc40f3fab428`;
base sparse source SHA256
`4f9e2e250e3c08d0ce03a870817ce4137100d9267b1bdd2a4ad8e1bc803c7f7d`.
No timing or physical acceptance is claimed at this design checkpoint.

## CPU and offline readiness

The missing schedule API regression failed on da85 before implementation.
Thirteen original/new CPU tests pass, including schedule ownership, actual
augmented H and inverse action, original/packet row controls and parallel
limit extent. Seven current tests also pass with level updates and parallel
limits both enabled. Full pre-commit checks pass. The new switch binds once in the constructor and exposes
`level_update=True` with actual kernel `sparse_factor_level_update43_434`.
The ordinary kernel remains `sparse_factor_refresh43_434`. The six optional
schedule arrays total 29,004 bytes and are not allocated when disabled.
The additional plan descriptors enlarge kernel arguments by 336 bytes,
but do not add a world-scaled buffer or change W/Z shapes.

Independent source review finds no ownership, barrier or lifecycle issue;
the flag-off native body is unchanged, and only the original factor loop is
replaced when enabled. The old inverse body and all other kernel factories
are unchanged. Both kernels compile offline for SM120 and SM100 at the
existing 128-thread block size, with CUDA devices hidden. Artifacts:
`/tmp/fpgs-g1-level-update-native-20260914-9ny8_d8k`.
Compiler registers are old/new 44/47 (SM120), 38/40 (SM100); both have
4,308 bytes static shared allocation including compiler overhead, 72 bytes
stack and no reported local allocation. These are compilation facts, not
runtime occupancy or timing measurements. The inherited inverse stack is
not claimed removed.

Reuse the original paired physical parent `155eba000c50e28e5e88522897f89aae519b92d9b6de0b7c97ddb2a95e156e0c`
and child `027da7a49d25bad8e83e94e1ecad87f9a773c5cf663f0d19841a22d2d7d6859e`,
with the same in-memory pinned-input adapter as the parallel-limit gate.
Record SPARSE_FACTOR=1, SPARSE_PACKETS=0, SPARSE_PARALLEL_LIMITS=1 and
SPARSE_LEVEL_UPDATE=1 explicitly in each child environment (all have the
`FEATHER_PGS_` prefix). Use these four existing-module selectors:

- `TestSparseFactorCUDA.test_actual_operator_current_rows_and_held_reuse`
- `TestSparseFactorCUDA.test_complete_owner_two_steps_and_graph`
- `TestSparseLevelUpdate.test_level_schedule_owns_exact_schur_terms`
- `TestSparseLevelUpdate.test_level_factor_actual_operator_and_owner_admission`

All selectors are in `tools.fpgs_bench.test_sparse_factor`. The first two
exercise the actual native operator, held/current geometry and original
serial/parallel continuing graph controls; the last two are host-side
schedule/action and constructor controls. No new physical runner is added.

## Completed whole-cost discovery and bounded diagnosis

Runtime `3ca99cc3633c60fbe6b179837fd62c20fe2520e9` passed all four selected
physical controls on each GPU, zero skips/failures/errors, including current
rows, held-factor reuse and the continuing complete-owner graph. Physical
manifest `/tmp/fpgs-g1-level-update-physical-paired-20260914-01/manifest.json`
has SHA256 `4da8eaaf0f01376e0166419bafbf2f68970661a68f7c005a4eec7b5bc4c2e8d9`.

The unchanged paired 16K seed0/200 warm/40 wall/40 physics protocol compared
da85 against3ca, with parallel limits enabled in both and only level updates
differing. Fixed Lab53ee, original two substeps/eight sweeps, F100/raw294912,
broad49152/tri1769472 and CELL/FINITE/WELD/PACK1 remained identical.

| GPU | Baseline physics | Level updates | Incremental ratio | Saved |
|---|---:|---:|---:|---:|
| RTX PRO6000 |21.148316ms|20.354962ms|1.038976x|0.793355ms|
| GB300 |26.345063ms|25.414346ms|1.036622x|0.930717ms|

Environment wall35.605495->35.832308ms RTX and40.211606->39.569081ms GB.
These are one-round discoveries, not repeated acceptance, training throughput
or ratios versus MJWarp. All four children exit0; eight original capacity
and actual-owner boundaries, source and final-idle checks pass. Whole manifest
`/tmp/fpgs-g1-level-update-live-paired16k-20260914-01/manifest.json`, SHA256
`075bc336f15c491d8202f77874e5bebb5296a8846146faa02598d5c92af99bee`.
Exact launch: `/tmp/fpgs-g1-level-update-checked-Dh40MvE1/LAUNCH.md`.

One bounded three-step node diagnosis gives refresh2.334110ms RTX/1.770069ms
GB, compared with the preserved current4a refresh3.184972/2.816128ms. The
factor column schedule is unchanged between4a and baseline da85; their limit
producer differs. Current GS4.840935/4.763809ms stays close to4a
4.923319/4.773600ms; collision and publication likewise do not grow materially.
The approximately0.85/1.05ms refresh reduction reaches whole physics. This is
not a large hidden saving erased by downstream conversion. Unchanged source
contraction, inverse and W publication still belong to the complete refresh;
no claim isolates their individual costs or proves hardware saturation.

Node manifest `/tmp/fpgs-g1-level-update-nodes-paired16k-20260914-01/manifest.json`,
SHA256 `da472575d3d91bf3d534e63e8c14f51b0a4b669a9302514bd9e8821b8f5bdc4e`.
The original analyzer refuses G1 auxiliary graph roots; parent1768194 and
both child wrappers exit1 after completed captures. This status is preserved.
The existing strict74c5928 reader, adding only the two finite-collision names,
parallel-limit name and new factor name, accounts for12 physics roots,
3 auxiliary roots and1020 physics nodes per card, zero unproven nodes.
Original source/capacity/actual-owner checks independently pass after reaping,
with fresh device-idle verification. No throughput result is taken from nodes.

The approximately1ms target is narrowly missed on primary RTX. Retain this
default-off experimental improvement, but do not launch another level/inverse
tuning campaign or present it as4x. Further work must retire larger complete
producer/consumer contracts; no repeated timing or cross-task promotion yet.
