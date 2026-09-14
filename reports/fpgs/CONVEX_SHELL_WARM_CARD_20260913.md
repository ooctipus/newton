# Retained warm query plus complete current shell patch

2026-09-13. Bounded CPU/source study; no runtime edits or GPU execution.

## Decision

GO for one bounded, default-off integrated experiment, subject to root approval:
reuse the OLD full-coherent positive-distance warm query and its current full-hull
bounds, but replace its retained-contact manifold with a complete, depth-aware
current shell polygon. Do not simultaneously introduce a new KKT cache encoding,
change support search, or tune several capacities/mappings. This is a physical
manifold correction plus query-work removal, not reuse of a previous speedup.

The target is at least 1.7 ms NET whole Allegro RTX saving against the CURRENT
source-identical rejection-only/BSP baseline. Historical covered cold
GJK/MPR/manifold cost is 2.924756 ms per environment step; treating that as the
planning envelope leaves only 1.224756 ms for the complete warm replacement,
remaining cold fallback, and any newly added work. It is not a current e259/10a
node measurement or a hardware prediction. Existing full-task timing decides.

## What the old experiment actually did

Old `coherent_convex_queries.warm_queries` reconstructs at most three cached
simplex vertices from current geometry, runs at most four GJK iterations, and
does two current complete-hull support-bound scans. Retained warm answers require
positive separation above 1e-4 m, feasible witnesses, an inside-shell upper bound,
and the original duality-gap bound. Penetration and uncertainty remain cold.
Every accepted warm result still enters the original ten-support manifold.

The old three-round whole medians were 19.071279275 -> 17.083149250 ms RTX and
20.745794450 -> 17.345010225 ms GB. They remain physically held, are from an older
source/protocol, and are NOT additive with today's rejection-only or BSP changes.
The current rejection-only path already benefits from far-separated reuse.

The old physical problem is concrete: the manifold fits planes to five tilted
supports per shape and rejects its whole polygon if either fitted normal exceeds
the existing 2-degree gate, then emits only the deepest point. Valid closest
witnesses do not prevent substantial torque-span changes. Point/count equality
is not proposed as a new acceptance gate; actual shell geometry and loaded
physical behavior are the relevant obligations.

## One complete algorithm

1. Keep current pair preparation, source/type/generation/world-epoch checks,
   the existing positive-distance warm query and its current full-hull bounds.
2. In the current warm normal's orthogonal coordinates, compute the two complete
   hull caps within the available shell slack. Include original vertices and
   actual edge/slab intersections; use current poses, scale, gaps and margins.
   Project the caps and intersect their convex polygons to obtain an outer P.
3. Projection alone is insufficient: both sides could consume the same slack.
   For projected point u, compute A's front surface a(u)=min_i a_i(u) and B's
   back surface b(u)=max_j b_j(u). The exact ray gap is g(u)=b(u)-a(u).
4. Evaluate g at P's vertices. If a vertex violates g<=shell, its active pair
   supplies one valid linear cut b_j(u)-a_i(u)<=shell. Clip P and repeat.
   Retain gap/active-pair values for UNCHANGED vertices within this collision
   call; evaluate only newly created edge intersections. This does not cache
   an answer across changes to geometry.
5. When all vertices satisfy the bound, convexity gives the complete polygon:
   every cut retained the feasible set, and the final convex polygon lies in it.
   Reconstruct actual current surface witnesses, then reduce/publish through
   the existing contact postprocessor, shell checks, material and public writer
   contract. Preserve the original maximum contact reservation, not every
   intermediate polygon vertex as a new public contact.
6. Unsafe/near-vertical plane cases, unsupported shapes, nonfinite values,
   penetration, failed convergence, or bounded local-storage exhaustion take
   the unchanged COMPLETE cold query/manifold path before any contact write.
   Never publish an unfinished polygon or roll back atomic contact writes.

This is not all-axis SAT. It constructs one current positive-distance shell
patch. Its normal comes from the existing warm query. Changing the manifold
representation still requires the existing geometry and loaded physical gates.

## Same two saved failure cases: CPU evidence

Only the two already-recorded frame-0 failures were examined. No new census,
rollout, benchmark, or test framework was created. These are FP64 host geometry
results from saved FP32 inputs, not native-CUDA or physical acceptance.

| Quantity | RTX pair 1067/1077 | GB pair 2851/2853 |
| --- | ---: | ---: |
| Actual pair shell | 4.000000190 mm | 3.000000026 mm |
| Current full-support lower gap | 1.572395153 mm | 0.505101462 mm |
| Cap vertices plus source-edge intersections, A/B | 25 / 31 | 12 / 37 |
| Projected cap polygon vertices, A/B | 17 / 18 | 4 / 20 |
| Relevant triangulated faces, A/B | 24 / 32 | 2 / 37 |
| Optimistically coplanar-grouped face planes, A/B | 22 / 28 | 1 / 25 |
| Initial intersection vertices | 19 | 11 |
| Full face-pair inequalities | 616 | 25 |
| Full clipping vertex/constraint visits | 15,837 | 275 |
| Lazy active cuts | 33 | 0 |
| Maximum/final polygon vertices | 29 / 29 | 11 / 11 |
| New edge-intersection vertices evaluated | 66 | 0 |
| Face-plane evaluations, unchanged vertices reused | 4,250 | 286 |
| Lazy clipping vertex visits | 649 | 0 |
| Full-reference boundary distance | 1.40e-17 m | 7.16e-18 m |
| Final maximum gap residual | 6.94e-17 m | 1.12e-16 m |
| Complete shell polygon diameter | 25.960801 mm | 96.502416 mm |

Naively rescanning every polygon vertex after every cut cost 33,900 face-plane
evaluations on the RTX case. Reusing unchanged vertices is necessary, not an
optional micro-optimization. Counts are not GPU cycle weights. Plane grouping
above used a numerical host grouping for the study; native static coalescing
must preserve the actual geometry or retain original triangles. Do not claim
the grouped count as an already-proved runtime descriptor bound.

The first host prototype formed unused vertical-plane coefficients and emitted
NumPy divide warnings; no such coefficient participated in the result. The final
incremental evaluation selected applicable finite planes BEFORE division and
produced no warnings. Minimum selected denominator magnitudes were 0.163/0.169
on RTX and 1.0/0.007997 on GB. This does not license ignoring unsafe cases.

The GB failure has eleven EXACTLY coplanar source triangles (111..121) forming
a 13-vertex face (21..33). Exact rational arithmetic on the stored unscaled
coordinates confirms coplanarity. Current box-reference clipping retains broad
in-shell geometry; widening the old angle gate is unnecessary. However a simple
single-face correction is insufficient in general: the most-aligned triangle/
quad of the RTX case clips to only a 0.523 mm patch, while the complete adjacent
shell patch above spans 25.961 mm. This is why the proposal includes the shell,
not merely a new reference-face selector.

## Complete costs and ownership that must remain visible

- Current old-full feature-cache allocation, generation maintenance, queue
  clears, lookup and reset handling are real added costs versus rejection-only.
  Do not add a second persistent clipped-polygon panel. Reuse the old warm queue
  and existing result capacity if a separate prepare/commit phase is required.
- Warm GJK and full-hull bounds remain charged. Each refused warm patch pays
  its attempted work PLUS original cold MPR/GJK/manifold.
- Cap discovery scans current <=64-vertex hulls and their actual edges (186 for
  each 64-vertex triangulated hull in these cases), constructs projections,
  transforms face planes, clips, reduces, and writes current contacts.
- The original local manifold has ten 2D entries. This algorithm needs larger
  bounded query-local polygons, active-pair/gap state and face data. Register,
  local/shared storage, occupancy, extra launch and queue costs must be charged
  in the first complete native measurement. No world-sized maximum panel.
- No fallback may append contacts twice. Prepare first; commit once. Original
  raw identity, counts, capacities, material, margin, friction, solver sweeps,
  public force conversion and collision cadence stay authoritative.
- Restore/retain the old cache's actual source/type/previous-generation and
  world-epoch checks, reset and graph-lease obligations. Current geometry is
  reconstructed every collision invocation; no stored normal or polygon may
  become an answer merely because the shape pair survived.

## First integrated gate, if funded

Use one bounded local implementation and the existing old full-coherent query,
not a simultaneous new feature solver. CPU/reference checks first cover these
two saved failures, original geometry controls, margin/scale/transform changes,
unsafe planes and no-partial-publication fallback. Then reuse existing GPU
geometry/loaded physical controls and exact current whole Allegro protocol.
Do not impose witness-ID, point-count or old-patch equality as an independent
quality gate. Preserve complete finite/convergence/contact-law checks and report
changed torque span/normal/impulses explicitly. No gain is accepted before those
physical checks and the complete source-identical whole-cost comparison.

Allegro's corrected MJWarp comparator uses native collision, so this Newton
collision change currently affects FPGS only there. G1 uses external Newton
collision on both backends; any later G1 transfer must retime the shared MJ arm.

## Readset / existing authority

- `/home/octi/Projects/newton-fpgs-coherent-queues-20260911/newton/_src/geometry/coherent_convex_queries.py`
  (`warm_queries`, lines 148..220) and `coherent_convex.py` (current feature
  reconstruction and cached GJK, lines 305..392).
- `/home/octi/Projects/newton-fpgs-convex-bsp-subtree-20260913/newton/_src/geometry/multicontact.py`
  (ten-support manifold and no-polygon/deepest-point transition, lines 651..955).
- `/tmp/fpgs-coherent-live-dmLltB/RESULTS_AND_LIVE_PLAN.md`,
  `whole_raw_audit.json`, `WARM_MANIFOLD_GATE_STUDY.md`,
  `MANIFOLD_TAIL_REVIEW_V1.md`, `manifold_front_v2.json`, `manifold_tails_v1.json`.
- Saved actual inputs:
  `/tmp/fpgs-coherent-live-quality-gpu0-512-20260911-01/frame_00.npz`
  SHA `f6cdc2365126624618e0d68e40875eb7435f0dd4c0f608cc0f34516727ff497e`;
  GPU1 counterpart SHA `909a284adbacf299f71817123a90c9cc794dd65ca54a29809e147f570caa7053`.
  Both `model.npz` SHA `e4a21145f3571549471b38453cb7bca729a818702c6ad2524cae30afa7901d2e`.
- Current timing-family authority:
  `/tmp/fpgs-convex-bsp-allegro-candidate-nodes-paired16k-20260913-01/DIAGNOSIS.md`;
  `/tmp/fpgs-convex-bsp-subtree-allegro-backends-paired16k-20260913-01/RESULTS.md`.

Inline CPU calculations in this task produced the table; no reusable helper,
new runtime, source mutation, or benchmark was created by this study.

