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

CPU mapping/admission and all seven existing cell tests pass; CUDA mapping
and complete paired whole timing are pending. The earlier finite-only packing
result is not substituted for timing on this original-query source.
