# Finite cuboid/prism query: complete experimental owner

## Paired physical01 failure and diagnostic successor

The original physical run is preserved at
`/tmp/fpgs-heightfield-finite-physical-paired-20260913-01`: parent42126 reaped
exit1, final source and idle checks passed. All eight selectors ran on each
card; six CPU controls and native synthetic CUDA geometry passed. Actual96
failed RTX's exact replay count/marked-count assertion and GB's exact restored
count/marked-count assertion. The old test did not record which value or
variant changed, and stopped before finishing all geometric checks. This is
not a physical pass, a diagnosed native fix, or timing acceptance.

The diagnostic-only successor changes this report and the independent test,
not runtime. Both original and finite variants now undergo the same
initial→empty→restored→three-graph-replay sequence, for both reducer modes and
both saved fixtures. It records every count, preserves each old count gate as
`legacy_count_gate_pass`, and requires exact decoded logical multisets and
marked logical-key identity, finite valid public geometry, no lost contacting
shape, and the unchanged 2e-4 m per-shape minimum-separation bound. Native
query/independent FP64 QP checks remain and run after count/ownership mismatches.
Failures aggregate only after collecting both fixtures; count-only failures
remain diagnostic evidence, not a claim they are harmless in advance.

Fail-first missing-collector control preceded implementation. Injected stale
marker keys, altered logical stream, lost shape, and excessive separation all
fail the CPU checker; a reordered manifold with one redundant contact changes
the legacy count diagnostic without changing ownership/physical gates. Three
CPU methods pass, two CUDA methods are registered/skipped CPU-side. Both full
96-pair CPU collectors completed with no hard failures or legacy count changes
(parent82742 reaped0). This does not explain the paired GPU difference yet.
Diagnostic test SHA256 is
`9492ef4e126504b5ba0092497635f5ad3aae2126f533ee864485b524da624217`.
All three actual runtime hashes below remain byte-identical to commit882e468a.

Status (2026-09-13): CPU complete-path and offline compilation ready; no GPU
physical, trajectory, or timing acceptance yet. Isolated branch
`ooctipus/g1-finite-query-20260913`, based on cell owner
`a994b1b24b6073dcb10e855c10e8197ee116ef14`. This is complementary to sparse43;
their savings must not be added without an actual composed measurement.

## Exact scope

`NEWTON_HEIGHTFIELD_FINITE_QUERY=1` selects the new owner at collision-pipeline
construction. Default zero leaves the original query. The existing compacted
triangle producer, its count/capacity status, and its launch mapping remain.
For each eligible triangle, the new query replaces generic MPR/GJK plus generic
manifold searches. The same writer consumes at most five contacts. Only after
all writes, that worker marks its own triple's index `~tri_idx`. A second full
original query scan skips these markers and handles every unsupported entry.
The reducer clear, writer, selection, export and public contact arrays remain
original. No exception queue, new maximum-capacity buffer, sorting, worker sweep,
or per-step host readback is added. The next midphase overwrites the live prefix.

Construction recognizes BOX and convex meshes with exactly eight finite cuboid
corners. This creates two vec3 bounds plus one source-pointer stamp per shape,
not a triangle-sized buffer. Poses, positive scales, heightfield elevations,
materials, margins and gaps are current inputs. Hull geometry is immutable for
this experimental pipeline: replacement pointers fall back; in-place hull
refitting requires rebuilding it. Mixed triangle-mesh scenes and speculative
contact mode retain the original full owner. Zero/ambiguous or downward-top
triangles, invalid geometry, tiny-margin inflation cases, and invalid prism
support witnesses use the original fallback.

## Finite contact law

Work is in terrain-local coordinates, with the exact original finite extrusion
of one metre along local negative Z. A support witness projected inside the
finite top triangle handles the simple positive-shell case. Otherwise the
query uses a closed prism/box SAT for overlap and a complete finite
triangle/box distance calculation when separated: three segment/AABB
piecewise-quadratic minimizations plus eight vertex/triangle-face projections.
There is no 108-edge-pair enumeration and no infinite-plane approximation.

For separated closest witnesses `d = b - a`, nonnegative terrain-local `d.z`
proves the same witness supports the downward prism: extrusion cannot increase
the prism's support in direction d. A negative component is an explicit generic
fallback, not a top-contact assertion. Parallel supporting edges preserve their
clipped interval endpoints. Positive distances outside the detection shell
still send one finite witness to the original writer/reducer; there is no new
far-distance rejection before reducer selection.

Overlap/top-face manifolds clip up to three lower cuboid faces by the finite
triangle side planes. At most 21 candidates stream through deepest plus four
tangent extrema, then deduplicate to at most five output contacts. This is an
explicit physical manifold selection change, not coefficient/contact-count
identity. Every selected witness remains on the finite triangle and box.
The local buffers and selection work are part of the query being timed.

The elided original postprocessor is identity for this admitted shape family:
`collision_core.py:216` and `:221` modify only sphere/capsule endpoints;
`:229`–`:233` require cylinder/cone for axial rolling. HFIELD plus BOX/CONVEX
meets neither condition. Effective radii stay zero, matching original
`narrow_phase.py:277`–`:284`. Margins/gaps and original writer conversion remain.
The direct and reducer fallback clones recover every original statement after
removing only their function rename and the negative-index skip; they are not
interchanged because the original kernels have distinct details.

## Evidence and limitations

The approved mathematical card is
`/tmp/fpgs-g1-finite-query-2XZCAc/CARD.md`, SHA256
`53612ce31fcf34f9eed4dd5af22cb6712729dca3fe3b937cd81698cfd92ebfe5`.
Missing-module regressions preceded implementation. The six native CPU tests
cover positive .04-shell contact, finite borders, rotated and deep overlap,
parallel edge intervals, invalid/side/bottom fallback, exact original fallback
source recovery, and a complete mixed BOX/SPHERE stream through both direct
and reducer writers with real prefix shrink/regrowth. Combined with the
independent geometry-checker negative control, eleven native synthetic cases
and seven unchanged cell-owner regressions: 15 CPU passes and two root-only
CUDA methods skipped. Five existing heightfield surface/prism/border and
unsupported-sphere regressions also pass with the candidate explicitly enabled.

Root's independent complete 96-pair CPU controls use the two pinned saved G1
geometries, native query witnesses versus an independent FP64 constrained
distance solve, and current full pipelines with original/finite query:

| Fixture | Queries/admitted | Max witness error | Direct contacts old/new | Reduced old/new | Max pair minimum change |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX capture | 1208 / 1205 | 1.895e-7 m | 700 / 648 | 199 / 198 | 1.479e-6 m |
| GB capture | 1190 / 1190 | 1.529e-7 m | 878 / 742 | 239 / 223 | 7.066e-7 m |

No previously contacting pair was lost; decoded logical triangle sets match.
Root also exercises nonempty→raise-all→empty→restore and registers three actual
graph replays for its CUDA method. These are saved-current geometric checks,
not live G1 convergence or throughput acceptance. The inherited 96-fixture
32768 buffers are diagnostic-only, not a proposed change to calibrated 16K
task capacities. Its current captured terrain rotations are identity; native
synthetic controls include rotated boxes, while inherited neighbor tests cover
original transformed-terrain behavior.

Offline CUDA compilation (no device opened) completed all four dispatch
kernels for SM120 and SM103. `/tmp/fpgs-heightfield-finite-offline-q7SKDc/report.json`
SHA256 `ee0a6e4995b5c41a26dcc74de22b9878d371edbd0adb069d57f640069fa6dec0`:

| Kernel | Registers SM120/SM103 | Stack bytes | Static shared | Spill loads/stores |
| --- | ---: | ---: | ---: | ---: |
| Finite direct | 127 / 128 | 496 | 128 | 0 / 0 |
| Finite reducer writer | 133 / 128 | 496 | 128 | 0 / 0 |
| Retained direct fallback | 168 / 168 | 528 | 128 | 0 / 0 |
| Retained reducer fallback | 168 / 168 | 432 | 128 | 0 / 0 |

These are static compilation resources, not measured occupancy, memory traffic
or runtime. In particular, the new query still has a local stack. There is no
timing result yet and no resource-based speedup claim.

## Charged decision boundary

The prior post-cell diagnostic query+buffered-writer owner was 3.922660 ms RTX
and 17.342261 ms GB. Original reducer/export/active-clear (2.017303/1.864149 ms)
and cell midphase (3.342030/4.746325 ms) remain. For a 2 ms RTX query-family net
removal, all analytical queries plus full fallback scan and changed overlap
must fit 1.922660 ms; a 3 ms stretch requires .922660 ms. More than 10 ms GB
removal requires less than 7.342261 ms. These are hypotheses, not predictions
derived from register counts. Actual complete live task A/B, including all
writers/clears/joins and unchanged solver work, decides promotion. Preserve any
failed timing; do not repeat the rejected direct pair-owned serial mapping.

The initial checkpoint is 16:25 UTC and stop/review ceiling 16:55 UTC. Next gate
is root-owned paired CUDA finite geometry/current pipeline/graph controls,
then one complete calibrated G1 live A/B if physical checks pass. No sparse43
composition before this standalone result. Finite/capacity/warning/source/idle
checks remain; physical contact changes do not require bitwise equality.

## Reproduction

With CUDA hidden and this worktree first on PYTHONPATH, use the existing Lab
interpreter through uv to run:

```text
python -m unittest -v tools.fpgs_bench.test_heightfield_finite tools.fpgs_bench.test_heightfield_finite_geometry newton.tests.test_heightfield_cell_reject
```

Root's GPU selector is
`tools.fpgs_bench.test_heightfield_finite_geometry.TestIndependentGeometry.test_actual96_complete_pipeline_and_query`.
It evaluates both saved captures on each leased card. Candidate native and
dispatch hashes at CPU freeze:

```text
289e89e5f1f9e65787f38c5be7768676b7cf15eb9983d2b3a246e547a2a31197  heightfield_finite.py
2b2b3d927a2e0cf1a896a8da96f7622c0553ce20325d02c299960474db29abf8  narrow_phase.py
b3788e277d3b3d8460605a1695a98eeea9aab72bcb1e7e123d955d37ed657f48  collide.py
15cb84594cb92468ffe90d032501ab61cdf51bac8bd61aa901d4e0f623d3f18a  test_heightfield_finite.py
989c283e34fd1034a30629453ffd5537c85fc64f31161adf2559cdb4b2138059  test_heightfield_finite_geometry.py
```

The last native edit only renamed the local numerator variable for the spelling
hook. All ten candidate/independent methods reran (eight CPU passes, two CUDA
skips) and all eight offline targets were rebuilt afterward. Scoped precommit
passes on exactly the seven candidate files; no global formatting was applied.
The launch wrapper is `/tmp/fpgs-heightfield-finite-physical-kzU2tu/run_checks.py`
(SHA256 `027da7a49d25bad8e83e94e1ecad87f9a773c5cf663f0d19841a22d2d7d6859e`),
a thin successor of the existing checked first-hit runner. Its three CPU
strict-result/import regressions pass. `pins01.json` records 887 source/input
identities (SHA256 `ddd306cf103808467effb12e5b77ffbbc91b8bfdd93b5c0d66fbebf3956235d3`);
actual selected-tree CPU imports and all pre/post hashes pass. The original
`run_pair.py` remains the sole device/process/source/idle owner.
