# Packed G1 pairs with the original collision query

This isolated successor of sparse954895ec excludes the finite-query manifold
candidate, which is on physical hold after an asymmetric loaded rebound.
It carries the exact independent-pair launch semantics from6a56b017: scalar
heightfield cells use all existing worker threads with matching stride, while
global triangle compaction, original GJK/MPR, writer and reducer remain intact.
No changes to sparse solver code, geometry, timestep, iterations or capacities.

The explicit selector is `NEWTON_NARROW_PHASE_PACKED_HEIGHTFIELD_PAIRS=1`, using the
existing benchmark's accepted narrow-phase namespace. This avoids another
parser/benchmark wrapper. This experimental branch packs the existing admitted
pure-heightfield CELL1 path by default; explicit0 restores its old launch.
CELL itself stays default-off. All other collision paths remain original.
Thus an existing shared-CELL backend benchmark on this same source also
enables packing for MJ without a new environment-injection wrapper.

Reuse the current sparse paired runner9df and its unchanged capacity/source/
UUID/idle/actual-sparse checks. Both arms use CELL1/SPARSE1; only the explicit
packed flag changes0/1. Use seed0,16K,200warm/40wall/40physics, unchanged
dense100/raw294912/broad49152/triangles1769472. Existing physical tests check
actual packed dimensions, exact triangle multiset and continuing graphs.
No new benchmark framework or physical tolerance is introduced.

## Completed discovery measurements

Runtime/test commit `19099e91` passes all 11 selected CUDA tests per card,
with no skips or failures. The original physical parent exits zero with its
source and idle guards passing. Physical artifact:
`/tmp/fpgs-g1-packed-original-physical-paired-20260913-01`.

The matched sparse954895 to packed19099 whole-physics discovery gives:

| GPU | Original mapping | Packed mapping | Incremental gain |
| --- | ---: | ---: | ---: |
| RTX PRO 6000 | 28.968070275 ms | 25.548476325 ms | 1.133847276x |
| GB300 | 40.855120725 ms | 34.689812850 ms | 1.177726755x |

All four captures pass both original capacity boundaries and final source/idle
checks. The independent complete-graph audit reproduces all 40-step intervals.
Artifact: `/tmp/fpgs-g1-packed-original-live-paired16k-20260913-01`, manifest
`5f171630dec4a06346ffd2b4cbf33210a2a404b3a225aa6a40ec20815df7dd6b`.

A separate fresh backend comparison enables shared CELL and packing on both
backends using this same source; only FPGS enables sparse43. Corrected MJWarp
and fixed Lab53ee retain their current budgets. Results are FPGS/MJ physics
25.563279200/42.293261875 ms RTX (1.654453701x) and
34.889251200/51.687804450 ms GB (1.481482195x). Artifact:
`/tmp/fpgs-g1-packed-original-shared-mj-paired16k-20260913-01`, manifest
`f523aa0701b0d93bd5a39cab7e8fdd7e2ba2706d67f68915ce798142d670497f`.
All four children and checked boundaries pass, with zero MJ warning masks.
The original backend parent completes after its final source check; that
older schema has no separate final-guard Boolean. Fresh compute queries are
empty after reap.

Each result is one discovery round, not a repeated promotion result or full
RL training throughput. No finite-query timing is substituted here. These
incremental mapping gains are not cumulative gains over the initial handoff;
the current backend advantage remains about 1.5--1.65x, not the 4x target.
