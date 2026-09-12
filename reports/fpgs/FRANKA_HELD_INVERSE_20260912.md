# Franka held inverse-action checkpoint

2026-09-12, 18:20 UTC. The 4x corrected-MJWarp target remains unmet for both
tasks. This is a scratch component experiment, not a Newton runtime change or
a new accepted Franka speedup. Isaac Lab remains unchanged at 1d8feb82d17d.
The accepted whole-physics numbers remain those in
[FOURX_PROGRESS2_20260912.md](FOURX_PROGRESS2_20260912.md).

## One representation/dependency retry

The preceding [kinetic-cache experiment](FRANKA_KINETIC_CACHE_20260912.md)
spent about 1.14 ms/environment in its consumer. This retry preserves current
forces, augmented held L, original prefix and rejected-world eight-sweep
fallback. It forms G = L^-T L^-1 on the original refresh, shares it into reuse,
and replaces repeated triangular actions by G times current tau, sparse mimic
action and sparse kinetic diagonals. Canonical L remains available to contacts.
An early force/factor/predictor/prefix kernel and a late current raw/MF admission
kernel remove the prefix-to-MF-allocation dependency cycle of the first sketch.

There is one fixed 32-thread world mapping. Exact new storage is 81 floats plus
one epoch integer per world: 5.125 MiB at 16K. Public contact and row capacities,
dt, substeps, GS iterations, mass refresh cadence and physical laws are unchanged.
No live allocation, reset/cache fallback or overlap improvement is claimed.

## Numerical tests passed, performance screen did not

Four actual saved 512-world refresh/reuse inputs pass independent CPU physical
action, held-cache, current-force, mimic/bounds/momentum and diagonal controls.
Early poisoned raw/MF inputs do not affect prediction; late current raw and MF
growth reject the affected worlds. Stale early and late epochs reject before
action publication. A finite 1% wrong inverse fails the independent oracle.

Both actual GPUs pass the frozen CUDA test: original and new complete component
outputs, unchanged native local9 fallback, shared refresh/reuse cache, and two
fixed graph replays with all numeric/readonly ownership checked. Each runs one
test with zero skips, errors or failures. This is not a continued trajectory or
full MJWarp physical-parity test.

The timing input is 32 independently materialized copies of each saved scene,
not an actual new 16K rollout. Forty event samples follow ten warmups. Each event
contains producer/early/late/original fallback twice. Eight solver substeps per
environment therefore require multiplication by four, not two.

| Serial partial core, ms/environment equivalent | RTX PRO6000 | GB300 |
| --- | ---: | ---: |
| First kinetic core | 2.154624 | 2.032256 |
| Held inverse-action core | 2.064512 | 1.945984 |
| Reduction | 0.090112 (4.18%) | 0.086272 (4.25%) |
| Proposed planning screen | 1.300000 | 1.300000 |

Both miss the screen. These are separate component-run comparisons, not an
accepted whole-physics gain or a universal wall-time lower bound. Retained free6,
generalized finish, current raw/MF production, live cache repair and scheduling
costs are still absent. All source/capture/output/readonly and parent idle guards
pass; analytic/fallback/other-owner counts match the first prototype exactly.

## Diagnose the missed advantage

A separate, complete node trace shows the remaining early stage is expensive:

| Additive node-window mean, ms/environment equivalent | RTX | GB |
| --- | ---: | ---: |
| Producer | 0.863294 | 0.709077 |
| Early force/factor/inverse/predictor/prefix | 0.857451 | 0.838080 |
| Late current admission/rank/diagonal | 0.207359 | 0.232704 |
| Original local9 | 0.124179 | 0.137024 |
| Internal gaps | 0.006443 | 0.004352 |
| First-to-last span | 2.058725 | 1.921237 |

Early refresh/reuse calls average 117.019/97.344 us on RTX and
116.117/93.403 us on GB. Reuse is still costly; inverse construction alone
cannot explain it. Different current inputs are used in the two epochs, so
their difference is not an isolated factorization measurement. The within-early
diagnostic below resolves the remaining source-phase attribution.

Every graph has eight ordered nodes, with no discarded transfers or work.
RTX has actual process/correlation joins. GB supplies correlation zero for all
128 nodes; strict same-process/graph/stream, repeated ordered eight-node IDs,
serialized launch windows and successful same-host-thread completion fences
prove ownership. This alternative is explicitly not a CUPTI correlation join.

The live source audit also identifies unpriced obligations: an unexpected
reuse-step mass notification requires current Hgeo repair; reset refreshes
current bias but must not silently force held mass; early prefix counters cannot
be cleared again; late output follows the original impulse clear and global
velocity seed. Stale primary full spatial inertia prevents a casual return into
the old full-cache path. No live integration is authorized from these results.

## Source-phase diagnosis closes local polish

One source-isolated diagnostic adds five lane-zero cycle stamps, with exact
recovery of original arithmetic/barrier source. All numeric outputs agree
exactly against the uninstrumented eager reference and graph replay on both
GPUs, including held G/L and original fallback. All current/plan readonly arrays
are unchanged. All 16K worlds are stamped in each device/epoch case.

Six alternating original/instrumented event pairs follow ten warm pairs:
RTX 2.068544 to 2.076800 ms and GB 1.945984 to 1.954240 ms. Added instrumentation
cost is 0.008256 ms/environment equivalent, or 0.40%/0.42%. Original and traced
compiler resources are identical. Raw stamps were validated before aggregation;
only summaries from one final untimed instrumented replay were retained.

| Mean-cycle share, percent | Force | Factor | Inverse action | Prefix |
| --- | ---: | ---: | ---: | ---: |
| RTX refresh | 45.28 | 23.95 | 14.97 | 15.80 |
| RTX reuse | 57.41 | 19.98 | 4.58 | 18.04 |
| GB refresh | 44.04 | 27.39 | 14.76 | 13.82 |
| GB reuse | 58.12 | 20.02 | 4.47 | 17.40 |

These are lane-zero source intervals, not elapsed critical-path fractions.
Unfinished other-lane work may enter the following interval; do not multiply
these fractions into promised wall savings. Reuse factor work includes retained
L staging/validation and synchronization, not a new Cholesky. Inverse action is
small on reuse, explaining why counting removed triangular divisions overstated
the opportunity in this consumer.

A separate exact-input census finds all external body wrenches and direct
joint-force controls zero in the four saved cases. The consumer nevertheless
visits 51 ancestry-qualified wrench projections/world: about 6.68 million at
16K across eight substeps. This is not zero total force: current drives, gravity
and bias remain essential. The census does not assign time to those projections.

The measured fixed-other-owner model requires about 88.5% RTX early-stage
removal to reach the 1.30 ms core screen. These diagnostics do not demonstrate a
credible route to that removal through another zero-wrench/inverse adjustment.
Decision: do not pursue local polish or live integration of this prototype
family. Preserve its correct mathematics and measured failure. A genuinely
broader ownership/dependency change may be studied separately; this is not an
impossibility claim about Franka or a differently overlapped formulation.

## Exact local evidence

- Source: `/tmp/fpgs-franka-kinetic-prototype-euL5lYJq/kinetic_inverse_consumer.py`,
  `ab122c217318e1caffb8f25049cacac77059100422d5b66dcdf6994e6b4b365a`.
- CUDA test: same directory, `test_kinetic_inverse_consumer.py`,
  `043ec6cfa731ec988169b9fe977f510332ce75a3dfb974ed6590dfd2a3a38a8a`.
- Numerical parent: `/tmp/fpgs-franka-inverse-correctness-paired512-20260912-01`,
  manifest `d217007ce5f4bb13ae40961b73c34fa971153d21fca316c9963f3831abb0465d`.
- Cost runner: prototype directory, `run_inverse_cost.py`,
  `7fee97b1961f69e8023030bda464d180b23e55930db692ce1ed935233fdd1ddc`.
- Cost parent: `/tmp/fpgs-franka-inverse-cost-paired16k-20260912-01`,
  manifest `d15bc93e789cddba5eed69ed029d080478784e183e872a4c8ea0c9ef24fdebbb`.
- Node parent: `/tmp/fpgs-franka-inverse-nodes-paired16k-20260912-01`,
  manifest `4407a1d43e4d2a1dafddd02823e29f53b875173cd909795386638c4448f58082`.
- Node reader: `/tmp/fpgs-franka-inverse-nodes-f4I5BZgG/audit.py`,
  `dd501953df060adcd891f74afe9061e62e00b5090e3dcd22b126aa191a3fbf17`;
  use its `evidence_final.json` and `FINDINGS.md`.
- Live cache audit: `/tmp/fpgs-franka-live-cache-review-DdCGhHLk/FINDINGS.md`,
  `1ee6ee7c198ef3b3b900aac0359697957bb99b8913f83d016bffcc0208b054fa`.
- Phase parent: `/tmp/fpgs-franka-early-phase-paired16k-20260912-01`,
  manifest `eef8851dc06d3be0540204cd30b0541df27b6342200ebae55d25bf986484f27d`.
- Phase runner: `/tmp/fpgs-franka-early-phase-IW4sJ2pf/run_phase.py`,
  `51455fad9aac35b7c1fcb80eadd0347ee498d8a3212c1878c7cfc8ae0d08fba5`;
  factory `5104e90faa1977f8f6a3f0dd473285ae7d83874df87f2e76e0d573d9725b0bcc`.
- Independent phase audit: `/tmp/fpgs-franka-early-phase-audit-TmJOzOYX`,
  reader `24aad4c5e7d65611e8743b87a75ea69f52688e6fa42998c4e5b5e56708e0f84c`,
  evidence `6c976decd4119f0e1644c544443c85566cb3f8ae06fa8ec2812cedb3fb012dd5`.
- Current-force census: `/tmp/fpgs-franka-current-force-census-D1MEqZ1D`,
  evidence `401ecfb11fee33142bffae4cbc89dbd5c724a1fdbbf65271e6ebb249d309b70b`.

Frozen original helpers, original inputs, fork ancestry and reproduction
dependencies are listed in the preceding report and the parent manifests.
Both actual targets compile at 40 registers for early and late, 960/272 shared
bytes and zero stack/spills. Compiler resources do not predict a speedup.
