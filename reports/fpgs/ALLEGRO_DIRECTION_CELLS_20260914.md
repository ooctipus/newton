# Allegro immutable direction-cell support

Experimental, default off; no measured gain or promotion.

## Budget and contract

Funded 2026-09-14 19:18 UTC; first native/physical readiness checkpoint 20:45
UTC. Baseline is integrated `baf0c57ac5ef22bef452acd625b1727dd73a50b5`,
which retains the accepted rejection-only Allegro route from `fba9fead`.
No Isaac Lab, capacity, timestep, substep, or iteration allowance changes.
The user requires at least 4x corrected MJWarp on RTX across all six tasks;
this candidate is an Allegro contribution, not completion of that objective.

Today's accepted Allegro whole physics is 17.3286801 ms RTX / 17.8261976 ms
GB300 against corrected MJWarp 63.433607275 / 82.050660375 ms. The desired
first contribution is at least 1.47 ms RTX whole. Original classifier/MPR/GJK/
manifold together cost about 3.548 ms: a replacement near 2.078 ms is needed
if other work stays flat. These ownership costs are not independently
additive critical-path promises.

## Representation and eliminated work

`NEWTON_NARROW_PHASE_CONVEX_CELLS=1` replaces support searches in all four
existing rejection-route owners. For each immutable validated cooked convex
hull of 4--64 vertices, store six faces by 16 by 16 uint64 candidate masks.
Each mask conservatively intersects the entire relaxed vertex normal cone
with the direction cell, not merely its corner winners. Runtime uses the
original rounded scaled direction, one table lookup, and ascending candidate
dot products. Original primitive policies, feature value/point/magnitude,
MPR/GJK, manifold, query filters and contact publication remain.

This removes full vertex scans on admitted bounded directions. It does not
retain the failed hill walk, CSR traversal, terminal certificate, cold seeds,
pair-history words, or callback hint traffic. The CSR is host-only construction
input. Unsupported hulls/directions and invalidated geometry use original
scans. Zero, nonfinite, and extreme directions are explicitly unsupported.
No ambiguity-dependent secondary scan is added for admitted cell queries.

The fixed16 CPU study `/tmp/fpgs-allegro-direction-cells-ssrJU2` builds
15,360 nonempty masks for the ten cooked Allegro hulls: 122,880 bytes plus
small immutable descriptors, independent of world/pair count. Build time was
4.539 s excluding model loading. Uniform cell candidate mean/p95/max was
1.660/3/31; old saved original GJK rays averaged about 3.95 candidates.
Those saved rays are not current population timing. Counts cannot predict
the whole gain; query arithmetic, lookup latency, descriptor liveness,
remaining manifold work and fallback must all be charged.

## Numerical and lifecycle gates

Reuse the exact cooked-hull validator and exact rational offline clipping.
Normal-cone score slack and expanded cell rectangles cover FP32 score,
normalization and boundary errors for the declared immutable coordinate and
direction ranges. Native normalization uses explicit RN operations. Original
support and classifier feature contracts have separate fallbacks.
Invalidate graph-visible descriptors before any point/topology edit; retain
arrays for existing graphs, rebuild/recapture to re-enable. Poses and current
signed/nonuniform scale remain live. The original cache provenance is unchanged.

Regression first: owner admission, all ten cooked hulls, cell boundaries and
transformed directions, original primitive policies, all four complete query
owners, source/epoch/reset/empty transitions, and actual production graph.
Physical quality, not arbitrary intermediate bit identity, is authoritative.
CPU and offline SM120/SM100 resources precede clean freeze. Reuse the existing
paired physical and whole owners; root owns GPU leases. No new benchmark
framework. If the complete path loses, first localize actual owners and permit
only one cause-specific correction; otherwise close this representation.

## Prior art

The previous BSP path paid per-query predicate/tree work and lost; both exact
hill variants paid callback walks/certificates/hints and lost. The full-shell
manifold changed geometry and suffered a much larger loss. This candidate
precomputes an immutable support candidate representation while keeping the
original query/manifold law. None of those measured losses is its baseline.

Study source SHA256:
`b0a56c31ace53ddc403d3139146af6f85d1b1513531804b78215619879bd62be`.
Study result SHA256:
`3ee96848596556ae1d850bd5ad4a06b18ab9805e7b4565949c52612d69152707`.

## CPU/native readiness (19:38 UTC)

The owner regression first failed with ModuleNotFoundError. Six CPU tests now
pass; the four explicit CUDA selectors are not claimed by their CPU skips.
All ten actual cooked hulls admit 55,724 supported random/face/cell-boundary/
adjacent-FP32 queries with the original support and feature scores retained.
Masks occupy 122,880 bytes; complete immutable descriptors 123,004 bytes;
per-pair hint allocation is zero. Full four-owner CPU tests pass cold/current,
source changes, epoch gaps, reset, empty/regrow and invalidated geometry.

The original be47d960 compiler was reused in memory with only the actual four
kernel instances/modules and SM100 target substituted. All eight native entries
compile; report and exact source pins are at
`/tmp/fpgs-allegro-direction-cells-offline-CfrkAN/offline/report.json`.
Classifier/MPR/GJK/manifold registers are 74/124/125/166 on SM120 and
75/128/121/168 on SM100. Their stack frames are 24/24/352/296 bytes;
ptxas reports zero spills and the original 512-byte shared allocation each.
Stack/local traffic and manifold register residency remain real costs, not
evidence of a win. No GPU execution or whole timing has occurred yet.

Physical selectors in `newton.tests.test_convex_cells.TestConvexCells`:
`test_support_and_cells_cuda`, `test_actual_cooked_hulls_cuda`,
`test_complete_queries_cuda`, `test_production_dispatch_graph_cuda`.
Reuse the original paired physical/whole recipe and fixed Lab53ee. Whole
comparison is integrated baf0 versus this default-off successor, not either
failed hill prototype.
