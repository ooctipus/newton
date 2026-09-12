# Structural continuation: measured results and open gates

This is the continuation after the correctness checkpoint, not a completion of
the additional 2–4x target. The earlier
[structural plan](STRUCTURAL_V2_20260912.md) records the pre-implementation
hypotheses. This report supersedes its statement that no new timing gain exists.
Isaac Lab is unchanged at `1d8feb82d17dbfab8f0772de56f84deae2cb7974`.
All Newton branches retain the original handoff `31cf87f46` and use the
`ooctipus/newton` fork. Experimental flags remain default-off.

## Keyboard SO101: repeated whole-physics timing

The baseline is the capacity-corrected Newton
`108459ec7420fb62f79bc2b2728c974548142105`; candidate runtime is
`654894cb9ee623c9d516a535eb4de7bbb87ae285`. Both use 4,096 worlds, seed 0,
200 warmup steps, 1,000 synchronized environment steps and 40 graph-profile
steps per arm. Three rounds alternate baseline/candidate, candidate/baseline,
baseline/candidate. Each arm starts RTX PRO 6000 and GB300 simultaneously,
with one owned process per device. These are per-device medians, not pooled
cross-hardware samples.

| Hardware | Baseline physics ms | Candidate physics ms | Baseline/candidate | Baseline wall ms | Candidate wall ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX PRO 6000 | 11.258526 | 7.871787 | **1.43024x** | 38.377142 | 35.700507 |
| GB300 | 10.749849 | 7.124362 | **1.50889x** | 34.095171 | 31.407152 |

Physics samples in round order, in milliseconds:

- RTX baseline: 11.1186283, 11.2827297, 11.25852635;
  candidate: 7.683744825, 8.079784575, 7.871786775.
- GB300 baseline: 10.7663184, 10.5390982, 10.7498492;
  candidate: 7.124362225, 7.0904303, 7.485207275.

The graph speedup is not end-to-end training throughput. Measured environment
wall improvement is only about 1.08x/1.09x; reset/IK, observation, host and other
non-physics work remain. No Lab changes are included. Do not divide these new
samples by an old, unpaired MJWarp sample and call that a fresh measured ratio.

All 12 captures completed with finite states, zero four FPGS sticky capacity
flags and zero 13 Newton narrow-phase flags at both checked boundaries. All
source/idle guards passed and children were reaped. The original timestep,
two substeps, effective eight sparse PGS sweeps and solver law remain unchanged.
Both arms use the same calibrated capacities: 704 rows, 147,456 contacts and
57,344 broad-phase output pairs. The full input pair list is not reduced.

**This is repeated timing evidence, not automatic physical-quality acceptance.**
Actual loaded-input producer controls and physical residual/rollout diagnostics
are a separate pending gate. Finite states and zero capacity flags alone do not
establish numerical quality. Bit-identical independent trajectories are not a
requirement; assess differences at their physical scale.

## What changed, and why it was worth implementing

`FEATHER_PGS_SPARSE_CONTACT_DIRECT=1` assigns one bounded worker to each sparse
scalar-contact slot. The old scalar kernel accidentally inherited the dense
warp-per-contact producer's 4,096-worker limit, giving it only 16 CTAs. The
existing native arithmetic and output ownership are unchanged; there is no new
allocation, host contact-count read or kernel launch. The separate first graph
A/B measured 1.126x RTX / 1.142x GB whole-physics speedup, not just a kernel gain.

`FEATHER_PGS_PRISMATIC_PUBLICATION=1` admits exact fixed-root, direct-prismatic-
leaf stars. Original FK handles the root; the following body-parallel publisher
produces current canonical key kinematics/dynamics. Unsupported articulations
retain their original path. The mass/solver/integration algorithms are unchanged.

The combined actual node profile confirms these owners on the real task:

| Kernel, per environment step | Original RTX ms | Combined RTX ms | Combined GB ms |
| --- | ---: | ---: | ---: |
| Final FK kinematics | 2.824845 | 0.183211 | 0.176191 |
| Sparse diagonal contact response | 1.508556 | 0.190133 | 0.155359 |
| Body finalizer / replacement | 0.419627 | 0.831435 | 0.360107 |

The replacement finalizer is more expensive than the old finalizer; this cost
is included, not hidden behind the faster FK producer. The combined graph still
has 344 kernel and 116 memory nodes per environment step. Node profiling is
attribution, not a substitute for the three-round graph-only timing above.
The remaining RTX sparse solve costs 1.125313 ms in that node capture.

Twenty-one targeted implementation tests pass on **each** actual GPU: sparse
response ownership, prismatic publication and mass-interval/reset behavior.
The wider response suite also exposes one inherited test-signature error:
`test_tiled_response_diagonal_matches_dense_reference` supplies seven arguments
to a nine-argument kernel. It was reproduced on both GPUs at clean baseline
108459ec; it is not hidden or counted as a passing test.

## Next substantial opportunities and stop rules

The additional 2x target against the fixed RTX baseline means approximately
5.629263 ms. The current 7.871787 ms still needs about 2.242524 ms net removal.
Do not restart the target from a slow prototype, count restored dropped work as
an optimization gain, or assume one more producer specialization will reach it.

The next Keyboard experiment changes the admitted keys' representation across
both producers and real scalar mass/force consumers. It must exclude legacy
masked spatial-inertia/composite readers before omitting their inputs, retain
current public poses/velocities and sparse-contact geometry, and preserve cold
cache, reset, notification, held-mass and stream-event contracts. Merely skipping
matrix stores or tuning the finalizer alone is not the proposed experiment.
The six-DOF arm's selected-tau owner is not removable key work.

ANYmal's 18-DOF parallel solver owns about 4.075/4.459 ms in the fresh RTX/GB
profiles. A source-isolated diagnostic records **every executed sweep's** fresh
positive parent/friction-radius history and incoming Nesterov momentum. A final
zero impulse is not proof that a row was unnecessary earlier. Only after this
census will a lazy-row algorithm be considered, charging detection, construction,
changed preconditioning, convergence and fallback costs. This diagnostic has no
timing claim and changes no live solver statement.

Allegro's current collision work owns 5.815/7.837 ms of 19.256/20.864 ms. A
rejection-only temporal support-axis algorithm is being carried forward onto the
correctness-fixed source; it is an old algorithm under new validation, not a new
gain until the new integrated A/B completes. Uncertified current geometry always
falls back to the original cold MPR/GJK/manifold path. Native and actual loaded
contact tests precede whole timing.

The Keyboard broad-phase neighbor-list idea was closed before implementation:
its measured 0.370/0.434 ms owner cannot meet a 10% whole-step milestone even if
free. This is the systematic prevention of unproductive micro-optimization:
write the ownership/cost hypothesis first, run an early integrated causal test,
diagnose a theory/timing mismatch, allow one retry only for a new concrete cause,
and time-box work rather than repeatedly polishing a sub-percent opportunity.

## Local evidence and reproduction identities

Large captures remain local. The measured runtime/source commits and the exact
commands, environment flags, GPU UUIDs, source hashes, results and checked
capacity reports are retained in these manifests:

- `/tmp/fpgs-keyboard-structural-balanced-20260912-01/ab_manifest.json`:
  three alternating long rounds, all six paired arm manifests.
- `/tmp/fpgs-keyboard-structural-nodes-20260912-01/manifest.json`:
  combined node attribution, all checks complete.
- `/tmp/fpgs-response-owner-keyboard4k-ab-20260912-01/ab_manifest.json`:
  response-only first causal A/B.
- `/tmp/fpgs-prismatic-keyboard4k-ab-20260912-01/ab_manifest.json`:
  publication-only node A/B.
- `/tmp/fpgs-keyboard-structural-gpu-tests-20260912-01`:
  21 targeted tests per GPU.
- `/tmp/fpgs-response-diagonal-baseline-control-20260912-01`:
  clean-baseline reproduction of the inherited test error.

The paired benchmark helper and handoff report are retained in this branch;
no Lab dependency pin, original worktree or installed backend package is changed.
The overall structural optimization mission remains open.
