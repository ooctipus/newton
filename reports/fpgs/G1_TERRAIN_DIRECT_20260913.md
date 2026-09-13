# Direct terrain query: complete producer replacement experiment

Funded 2026-09-13 13:10 UTC, first checkpoint 14:40 UTC. Default-off Newton
collision experiment on top of repeated current-height cell rejection, not a
new solver or Lab/asset/physics-budget change. No measured direct-path gain yet.

The remaining actual G1 collision boundary is 9.364628 ms RTX / 24.030058 ms
GB300 exclusive. Midphase is 3.342030 / 4.746325 ms, triangle-query producer
3.922660 / 17.342261 ms, reduction .435093 / .627285 ms, export .465141 /
.526805 ms. The two replaced serial stages total 7.264690 / 22.088587 ms.
Those stages are serial on the same stream in the captured graph. The full
reader preserves all graph memory and overlap; whole physics remains the
separate balanced 38.019430 / 52.115792 ms reference.

Replace midphase triangle emission followed by global triangle queries with
one warp-owned current heightfield/convex pair traversal that directly calls
the original GJK/MPR/manifold writer for surviving cells. Retire every global
triangle triple write/read, per-triangle emission reservation and separate
midphase launch. Reuse pair descriptor extraction across that warp's cell
traversal. Reduce the existing logical triangle demand counter once per block;
retain its original calibrated capacity check during this experiment. The
existing triangle allocation remains reserved but untouched, not an added
cache/panel. No extra queue or launch is introduced.

Keep current scaled AABB, full speculative/contact shell, conservative current
height rejection, finite downward prism, current transformed triangles, exact
query/refiner/manifold functions, all reducer/raw/public-contact output laws,
sticky checks and original numerical tolerances. Mixed mesh scenes fall back
before the new owner. Cell rejection must be explicitly enabled too. No box
approximation, plane replacement, query projection or dropped contacts.

Added work: uniform range calculation per participating lane, a warp reduction
of the logical count and more pair-local loop state. The GPU mapping changes
from a global compacted triangle stream to lanes sharing one pair. Variable
cell counts and longer register lifetimes can erase the theoretical saving;
there is no claimed occupancy/roofline result. A first loss gets one node-level
diagnosis and one structurally motivated correction, not a block-size sweep.

The >=10% whole-time screen permits this entire replacement at <=3.462747 ms
RTX / 16.877008 ms GB, including all added setup/query work. The RTX requirement
is demanding; merely deleting a small counter kernel would not pass it. A
smaller gain will not justify a new micro-optimization project. Shared Newton
collision must be retimed on corrected MJWarp before claiming backend ratios.

First tests: fail-first constructor admission; original/current near, below,
rotated, reversed, scaled, speculative and actual convex-mesh contact coverage;
exact logical triangle demand; poison old triangle triples and prove no old
midphase/query launch in the selected owner; graph replays/current heights;
overflow still detected and mixed/default fallback unchanged. Then paired live
16K A/B against a994 with cell rejection on both, unchanged calibrated buffers,
200 warm / 40 wall / 40 physics. No stand-alone fused-kernel win can promote it.

Node authority: `/tmp/fpgs-g1-heightfield-cells-nodes-paired16k-20260913-01`,
manifest `a4ab94f3d263523513bd4e771d784e468fc6042f5021c69159bd97c1ff43e2ad`.
The original auxiliary-graph analyzer failed; both actual simulations/checks
completed, parent68374 was reaped, and root independently passed final
source/idle. Preserved interval reader plus one reviewed collision name passes
12 physics / 3 auxiliary roots on each GPU; diagnostic results are in
`/tmp/fpgs-heightfield-node-owner-0sD2F2Ly/gpu{0,1}.json`. No recapture or repaired
original parent status is claimed. Original and candidate node sources/caps
are pinned in their manifests; profile window is three steps, not throughput.

## First implementation gate, 13:24 UTC

Missing-mode constructor regression failed before implementation (session63997).
The first test fixture then needed explicit `has_meshes=False`, since standalone
NarrowPhase defaults to mixed meshes; this was a test-admission correction.
Six focused CPU tests now pass, including sphere/capsule local radius handling,
all range/padding AST operations, complete contact coverage, untouched poisoned
triangle storage, original producer retirement, current heights and expected
logical triangle-overflow failure. Both actual 96-pair CPU fixtures retain all
700/878 contacts and identical 1,208/1,190 logical triangle queries, with zero
observed full-tuple error. Scoped pre-commit passes. No GPU or timing result yet.

## Completed physical and performance gates: stop this execution map

The paired GPU physical parent
`/tmp/fpgs-g1-terrain-direct-physical-paired-20260913-01` exited zero, was reaped,
and passed final source/idle checks. Each card ran six tests, no skips, and both
actual 96-pair fixtures with three graph replays. All 700/878 contacts remain;
maximum point/distance errors are 1.86e-9/3.73e-9, normal error zero. Candidate
runtime pin is `c41f46e74a63fc0e1bb79a96636c42260dc63576`.

The complete actual16K A/B then **failed** the performance gate despite all four
capture/capacity/source/idle checks passing. Parent
`/tmp/fpgs-g1-terrain-direct-live-paired16k-20260913-01` is reaped, exit zero.

| GPU | Cell-only physics ms | Direct physics ms | Throughput ratio |
| --- | ---: | ---: | ---: |
| RTX PRO 6000 | 37.975026650 | 44.056320125 | 0.861965469x |
| GB300 | 51.958921700 | 86.469124850 | 0.600895658x |

Environment wall also regresses, 50.838382 to 58.597690 ms RTX and 67.173939 to
99.248259 ms GB. No direct-mode MJWarp comparison or promotion is justified.

The complete three-step node attribution isolates the penalty in the merged
owner: **13.346504 / 56.375385 ms**, versus the retired midphase plus query
**7.264690 / 22.088587 ms**. Other physics categories remain near identical.
New query register use is 216 RTX / 224 GB versus old168; block32/grid6144 are
unchanged. The exact direct entry's cached PTX declares480 bytes of local state
versus old432; reported localMemoryPerThread=0 does not establish zero traffic.
No hardware-counter bottleneck or measured occupancy claim is made.

Diagnostic parent `/tmp/fpgs-g1-terrain-direct-nodes-paired16k-20260913-01`
was reaped with exit1: both simulations/checks completed, but the inherited
analyzer rejects the auxiliary graph. Independent source/idle checks pass;
the reviewed process/correlation reader accounts for all12 physics roots with
1,248 nodes and three auxiliary roots with30 nodes on each GPU. Original
failed parent status is retained. Reader/output:
`/tmp/fpgs-terrain-direct-node-owner-JJQ8Yj6u/{audit.py,gpu0.json,gpu1.json}`.
Manifest SHA256:
`8f71dcf7f0726193e032f25d0d9ec6cb8d2b4d5867a72f1e046896e48955a4d2`.

### Why deleting the triangle stream lost useful work organization

A separate CPU count uses the same FP32 Warp current-range/rejection functions
on both unchanged actual G1 geometry snapshots. It counts active entries into
the expensive query, not executed instructions, hardware occupancy, or exact
GPU-FMA outcomes. These are saved states, not an asserted replay of the timed
three-step trajectory.

| Snapshot | Live triangles | Pair-owned query warp entries | Active lanes | Compact-stream warp entries (minimum) | Entry inflation |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX | 608,272 | 51,306 | 37.05% | 19,009 | 2.6990x |
| GB | 609,402 | 51,448 | 37.02% | 19,044 | 2.7015x |

Most nonempty pair chunks contain only12 surviving cells; each is visited
twice, once per triangle. The original global stream packs surviving queries
across pairs and triangles. Direct traversal removes that compaction along
with its writes, substantially increasing query-warp entries, while keeping
more range/loop state live through the generic query. This supplies a concrete
mechanism compatible with the measured loss; it does not prove a complete
hardware causal decomposition or assign all runtime to inactive lanes.

Even pairing the two triangles across lanes would leave fragmented pair
cohorts and the enlarged query lifetime. There is no demonstrated correction
that closes the strict complete3.463 ms RTX target. Restoring a compact query
stream largely restores the retired work rather than delivering the promised
deletion. Stop this representation after the causal diagnosis; do not sweep
block sizes or polish the losing kernel. Keep the repeated cell-only win.

CPU diagnostic source/result:
`/tmp/fpgs-terrain-direct-lane-audit-ylTx2xH6/{audit.py,result.json}`.
Both inputs are SHA-pinned in that result and checked unchanged after reading.
No original capture was modified. The distinct finite-feature contact algorithm
remains an unimplemented hypothesis, not a correction credited to this path.
