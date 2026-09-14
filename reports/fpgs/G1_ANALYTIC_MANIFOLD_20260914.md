# Supply certified face witnesses to the original manifold

Pre-code card, 2026-09-14 10:49 UTC. First implementation checkpoint 11:34;
no extension beyond 12:19 without a recorded cause and renewed authority.
Base: `7df75c46bd468b64c17958190fc800cfeb62201e`. Research-only default-off
`NEWTON_HEIGHTFIELD_ANALYTIC_MANIFOLD=1`, used with the existing finite query.
Fixed Isaac Lab53ee, timestep, substeps, iterations and capacities are unchanged.

## Exact boundary and work

For recognized cuboids, a minimum support point whose projection along the
upward triangle normal is inside that finite triangle is a closest witness
for the entire downward prism. Admit only positive clearance strictly inside
the original margin/gap interval, with conservative roundoff fallback. Feed
the true-surface witnesses and normal to the original manifold builder,
support function, query filter, postprocessor, writer and reducer. Preserve
the original triangle sort key and handled-entry lifecycle.

Accepted entries retire the discarded analytical face clipping and the later
MPR/GJK query plus duplicate geometry setup. They still perform the original
ten perturbed support queries, projector construction, polygon clipping,
deepest-point rule, contact writes and full reduction/export. Unsupported or
uncertain entries retain the original finite/generic path. No result buffer,
new queue, launch, reducer law, analytical replacement manifold or capacity
change is introduced.

## Cost and falsification

The old direct-fallback experiment d3cceb only removed discarded clipping:
its finite producer saved .167 ms RTX/.491 ms GB. Current entire fallback
cost is about .95/3.35 ms, including required manifold work and other queries.
Thus this witness replacement alone is not a >=2 ms RTX claim. It is funded
as part of a collision composition: an unmeasured compact geometric cull
could save about1.4 ms RTX and this query retirement about.6 ms. Both are
hypotheses; charge admission, support/manifold work, fallback scans and all
remaining graph owners before claiming any integrated saving. Do not add
overlapping owner times or contact percentages as whole-step gains.

First test the smallest integrated witness path, then existing paired physical
checks, then root-owned paired whole timing. If savings miss the composed
budget, use one existing node attribution to identify the actual cause; no
mapping, register or tolerance sweep is funded.

## Physical controls and prior cause

The preserved finite rebound diagnosis found one-sided close-contact coverage
and poor eight-sweep order robustness despite valid witnesses and momentum
closure. This experiment therefore retains the original manifold, not merely
the old reducer. Changed exact witnesses/normals can still affect support
features and deepest fallback; physics must be tested, not inferred from
geometry validity.

Regression first: missing witness/dispatch API, certified closest geometry,
uncertain triangle/margin boundaries, and actual original-manifold invocation.
Reuse existing saved96 current geometry/direct/reduced/empty/regrow/graph
controls and the reduced loaded rebound plus current seam/sliding/border
controls. Keep original numerical/physical tolerances and eight sweeps; do not
require bitwise rows or identical contact counts. GPU work requires root lease.

## CPU and source checkpoint, 11:05 UTC

The initial API regression failed on7df with missing `query_top_face_witness`
before runtime changes. The implementation now passes11 CPU tests with the
flag and flat-seam filtering enabled; the CUDA selector is explicitly skipped
with devices hidden. Both saved96 fixtures additionally pass the existing
complete direct/reduced/empty/regrow CPU lifecycle and independent witness
audits. The old query audit has a flag-only witness/marker extension; its
original off-mode meaning and all physical tolerances are unchanged.

The simple actual writer regression produces the original four-point face
manifold, keeps true surfaces/material fields/fingerprints, and confirms the
marked accepted entry adds no writes through the retained MPR/GJK kernel.
An uncertain supporting-face center on a triangle diagonal remains unmarked
and gets the original fallback manifold. Rotated/sloped closest distances are
checked against the existing independent convex distance control.

Admission on the existing pinned96 selections is34/53 original fallback
entries on fixture0 and35/45 on fixture1 (64.15%/77.78%), from1208/1190 total
triangles and33/32 distinct shapes. This is a selected geometry census, not
16K population coverage or a time fraction. No broader supporting-face
intersection algorithm was added.

Actual opt-in kernel key: `heightfield_analytic_manifold_contacts` for direct
and reducer writer overloads. Off mode retains the original finite key and
the original fallback kernels are unchanged.

Production offline SM120/100 compilation passed with CUDA hidden, using the
original finite compiler7dc3fb937efbe28a69c4728b3ae2897d1c4e217f0a85af0ef964e60b68ef09af
with only output directory and target103->100 changed in memory; executed
source SHA256 cad6d96bd30e43f15ed6e52d2b9d85508c3769350c7d5feb1559b446d3fff9f1.
Outputs: `/tmp/fpgs-g1-analytic-manifold-offline-WyLZcJ/report.json`.
Direct uses182/181 registers and768B stack; reducer184/184 and688B stack;
all have zero register spills. The current finite reducer was139/133
registers and592B local storage in the earlier runtime evidence. This larger
allocation affects all finite lanes, including uncertified ones, and is a
real integration risk to charge in whole timing, not an occupancy claim or
a reason to start a register/mapping grid.

Prepared original physical selector set (not launched):

- `tools.fpgs_bench.test_heightfield_analytic_manifold:TestAnalyticManifold.test_native_original_manifold_and_fallback_cuda`
- `tools.fpgs_bench.test_heightfield_finite_geometry:TestIndependentGeometry.test_actual96_complete_pipeline_and_query`
- `tools.fpgs_bench.test_heightfield_finite_geometry:TestIndependentGeometry.test_reduced_loaded_flat_seam_cuda`
- `test_rebound_diagnostic:TestReboundDiagnostic.test_reduced_rebound_three_same_state_runs`

The rebound collector's execution success alone is not physical acceptance;
review its original spin, residual, energy and momentum diagnostics. No GPU,
whole-step performance or physical-promotion claim is made at this checkpoint.

## First paired physical run and test-oracle correction

Frozenef7 physical01 is preserved at
`/tmp/fpgs-g1-analytic-manifold-physical-paired-20260914-01`, manifest
`f027b574a444fb53ff377a7084a3f3edd898d69eaa69e6b21d0be6917303acab`.
Parent2002870 and both children were reaped; final source/idle guards pass.
All four selectors ran without skips on both cards. The new simple writer
test failed; saved96, rebound and loaded seam controls passed.

The exact failure is a wrong bilateral normal oracle: the candidate returns
the authored plane normal(0,0,1), while original CUDA GJK returns
(0,-3.973643742938293e-6,1). Requiring these two normals to agree within2e-6
rejects the exact geometric answer. Correct only that test to compare the
candidate with the independent authored plane at the same2e-6 tolerance,
and record the original approximate normal error. Point/distance/material/
sort ownership, real manifold invocation and uncertain-fallback gates remain.
There is no tolerance relaxation or native change. Native SHA remains
`c50c889e2b93643ab63a59ef3d1a2e5ce65e15450e0f3bb13a54c715f27fdbaa`.

Root independently reapplied original `qualification.physical_failures` to
all28 saved seam/rebound records with no failures. Candidate rebound spin is
.00920686rad/s RTX and .00984394GB; energy ratios .36000011/.35994361 and
maximum foot-corner residual .00140013 remain in the retained original
physical envelope. This does not turn the failed execution into a PASS.
A fresh paired test-only correction repeat is authorized; whole timing is not.

## Completed physical and whole-cost decision, 11:57 UTC

Test/report-only `122eb87e63123698ab6aaa33e19727597cbbec36` passes all four
selectors on both cards without skips. No native code or physical tolerance
changed. Physical02 manifest SHA256
`b8fe52080784323eac071ee054ec2e37563e34a513ca4ca0c6bd82e6a2490ed3`,
under `/tmp/fpgs-g1-analytic-manifold-physical-paired-20260914-02`.
Original source, capacity and final-idle checks pass.

The authorized original16K/200warm/40wall/40physics whole test then loses:

| GPU | Original7df ms | Analytic manifold ms | Incremental ratio |
| --- | ---: | ---: | ---: |
| RTX PRO6000 |19.458857175|20.587984375|0.945156x|
| GB300 |25.1015952|30.004724475|0.836588x|

Artifact `/tmp/fpgs-g1-analytic-manifold-paired16k-20260914-02`.
All original source, actual-feature, capacity and idle checks pass. Contacts
and rows change approximately0.4--1.3%, not enough alone to explain the loss.
Whole attempt01 is preserved: its new untimed observer expected a short old
kernel key instead of actual qualified key; both baseline children failed
before measured physics. Only that observer literal was repaired for02.

One bounded node diagnosis at the matched200warm/40wall/3profile recipe
localizes the loss:

| Four collision calls per environment step | Original RTX ms | New RTX ms | Original GB ms | New GB ms |
| --- | ---: | ---: | ---: | ---: |
| Finite or fused analytic query |2.207010|3.700568|7.408693|14.509192|
| Retained generic fallback |0.937473|0.475883|3.364981|1.146581|

Fallback retirement is real, but the fused producer grows more. Solver
3.950146/4.424095ms and publication2.416790/1.621748ms stay close to the
source-matched reference. This is not a downstream solver regression hiding
a fast query. The new producer uses184 registers across its entire6144x32
launch, versus original finite139/133. Register allocation and changed kernel
composition are observed; occupancy or memory-bandwidth causality is NOT
established without unavailable hardware counters.

Original node parent/wrappers retain their auxiliary-graph analyzer failure.
The complete captures are preserved in
`/tmp/fpgs-g1-analytic-manifold-nodes-paired16k-20260914-02`. Reapplying original
source/capacity/activation checks passes; strict reader74c5928 accounts for
12 physics roots,3 auxiliary roots, all memory nodes and zero unproven nodes.
Only the observed analytic kernel alias is added to the established sparse
classification. `NODE_AUDIT_MATCHED.json` SHA256
`be73b6a3ce2f22e1ab49085e3594e56e0ddcfc32a3b09e2b32d42cb66127f305`.
Node01 used only3 wall steps and is retained as an earlier-trajectory
diagnostic, not substituted for the matched40-wall-step attribution.

Decision: keep default-off and OUT of the good baseline. No mapping/register
sweep. A separate two-stage dispatch could reduce the observed fused-producer
tax, but it must pay for admission and remaining manifold work; the original
entire RTX fallback is under1ms and discarded-clipping saving only.167ms.
No new >=2ms primary-GPU boundary is demonstrated by splitting this path.
Prioritize the measured geometric cull and larger complete representations;
do not label this numerical implementation incorrect or its scalar work
retirement imaginary merely because the integrated timing loses.
