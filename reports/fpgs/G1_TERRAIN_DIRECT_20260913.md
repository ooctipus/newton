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
