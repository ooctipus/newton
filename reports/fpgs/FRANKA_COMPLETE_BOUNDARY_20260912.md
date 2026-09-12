# Franka complete warm boundary: physical-law controls pass, cost gate misses

Checkpoint23:45 UTC. This is a saved-state experiment, not a live runtime
promotion. The [accepted whole-task table](FOURX_PROGRESS2_20260912.md) remains
unchanged. No Isaac Lab code, physics budgets, capacities or dependency pointer
changed. The complete warm mapping is not selected for live integration.

## What was measured

The earlier fixed two-world/16-lane arrow owner reduced its matched complete
core from2.080768 to1.884160ms RTX and2.277888 to1.569280ms GB300. This was a
component comparison against the fastest previous cooperative prototype, not
against accepted whole physics. The paired owner retained current forces,
held7+2 mass action, free-body dynamics, original eight-sweep prefix fallback,
all generalized/public body outputs and next private geometry/bias.

The new boundary additionally rebuilds current raw incidence and routes,
compact complement forces/predictor, demanded canonical contact geometry and
held factors, current dense/MF contact rows, velocity-limit allocation,
current free-body MF inverse, response/RHS/metadata, original local/general
eight-GS solves and complete complement integration/publication. Both response
and solver stream forks and all joins are timed. First output feeds actual
second input, including public state and next private caches.

Actual512 snapshots are materialized32times with distinct, rebased model/state
and raw-contact indices, yielding16,384 worlds. This is not a fresh live16K
contact distribution. Raw capacity stays4,000,000 and dense/MF/propagation
capacities192/64/192. All active routes are reconstructed; saved derived rows
and solved complement velocities are not free inputs.

## Numerical evidence and scope

Physical test01 is preserved as failed. Its original contact oracle compared
CPU geometry to GPU geometry and failed at absolute RHS difference7.15e-6.
Its GB graph check also failed on individual raw-contact impulse values.

Test02 changes only the oracle: identical original primitives on the same GPU,
plus original eight-sweep replay on each current row ordering, net generalized
impulse and public state checks. No tolerance was widened or runtime changed.
All four actual-CUDA tests pass on each GPU, including complement services,
two-graph restoration, original current rows and graph/poison controls.

- Current packet J matches the same-device original control exactly;
  maximum RHS difference is2.384e-7 RTX /1.192e-7 GB.
- Complement velocity agreement with independent FP64 original-eight is
  at most9.130e-8 RTX /2.724e-7 GB; impulse agreement2.734e-7 /5.464e-7.
- Maximum scaled physical action defects are6.353e-8 /1.001e-7.
- GB graph replay changes29 raw allocation slots and can change per-point
  impulses by0.001944/0.002242, while net impulse, public state and each
  ordering's original-eight law pass the existing gates. Per-point allocation
  is not a unique physical solution for redundant manifolds.

**This is finite-eight law compatibility, not proof of convergence.** The GB
second-state maximum unilateral residual violation is0.16523984m/s, also
present in the own-current-row original-eight reference. The metric combines
contact normals and signed joint limits; it must not be called a contact-only
failure before world/row/type localization. RTX maxima are0.00221038/0.00145537
and GB refresh0.00226284. This tail remains a numerical-quality limitation.

## Complete cost outcome

Forty event samples after ten warmups; two chained physical calls per graph
scaled by four to the environment cadence. All mutable state is restored
outside the timed graph. Source and read-only guards, complete world partition,
finite public state, capacity checks and final process/idle checks pass.

| GPU | Complete warm boundary median | Min--max | Final selected/complement per call |
| --- | ---: | ---: | ---: |
| RTX PRO6000 | **4.009984ms** | 4.001664--4.034432ms | 15,552 /832 |
| GB300 | **4.032256ms** | 4.007808--4.056960ms | 15,424 /960 |

Graph-versus-eager maximum scaled public-state differences are1.529e-6/2.629e-6.
These are repeatability diagnostics, separate from the independent law checks.
RTX dense totals107104/107072 and MF288; GB109856/109824 and MF384.

The RTX10%-whole-gain planning allowance is3.164803ms, derived from the
earlier accepted graph's3.739403ms enlarged exclusive replacement family.
The new boundary misses it by**0.845181ms**. This is a failed planning-cost
screen, not a matched-current whole-task regression measurement. Do not report
4.010ms as new whole Franka physics time, or compare it directly with MJWarp.

Live collision, external intercall callbacks, contact-force/sensor conversion,
cold/reset/odd-request repair and live generation/readiness protocol are still
outside this experiment. They are not credited as savings. Initial private
state seeding is outside timing. Inherited core metadata calls canonical L
readonly; that description is obsolete for this complete harness, which
correctly tracks demanded L9/L6 conversions as mutable restored state.

The next action is one bounded node attribution to explain where the additional
services cost time. It is not permission for another owner-layout sweep or
an uncosted live integration. The [24-hour plan](DATAFLOW_24H_20260912.md)
prioritizes broader work elimination, with Kuka first.

## Exact evidence

All parents are complete/reaped, with source and final idle guards passing:

- Paired core cost: `/tmp/fpgs-franka-paired-arrow-cost-paired16k-20260912-01/manifest.json`,
  `68b94bc7dadd28c2826072cb91939dad426cab6ec3c2301dd8f60d516fee97e2`.
- Complete physical02: `/tmp/fpgs-franka-complete-boundary-physical-paired512-20260912-02/manifest.json`,
  `9925530a9609ab1a3760c4bf5e1cbc1310234b5f016aaf96cb30d98aff2b7c94`.
- Complete cost: `/tmp/fpgs-franka-complete-boundary-cost-paired16k-20260912-01/manifest.json`,
  `15e4b5c293c09d7623539ab67964cb2be142912ff3dc3fc220fc0a47e79fa792`.
- Cost RTX audit SHA256
  `515e40cb8cd57220dceaad8a15c3e6e26d4d8ecdb4455464357a0feeca25a5e4`;
  GB `986e94ddc6a3dbdae249226179a91ea3556efb1a7e19add6b9290c94c3b373a3`.
- Frozen source and input inventories are adjacent to
  `/tmp/fpgs-franka-complete-rows-bj3qpP68/pins_cost01.json`.
  Physical test01 is separately preserved byte-for-byte at SHA256
  `1f77da8c92e5912da246090363329d04798b06fae54b3e2688e0bd4644077ea1`;
  test02 is `546ad7e08fc9fa8210c30cf324a13e1b997f45514d52fa5fe3d76260241bce56`.

Independent audit recomputed both40-sample medians and rehashed288 unique
source/input/artifact pins without mismatch. No new runtime is installed.
