# Keyboard wall gap: timing scope and reset workload

The measured physics-graph advantage is real within its scope, but comparable
contact behavior has not been established. Shared Lab and collision code does
not imply identical workloads or reset schedules. No Lab source, task setting,
termination rule or solver allowance changed for this investigation.

## Verified same-window timing

All twelve original balanced captures pass a new CPU-only audit joining every
physics graph and eager kernel/memcpy/memset to its successful process-scoped
host launch. Each window covers the same forty actions through the successful
final Torch/Warp drain. The original graph ratios remain 3.55014x RTX and
3.90098x GB, using medians of per-round means. This is collision, both solver
substeps, final publication and Newton sensors, not an isolated solver. There
are four physics graph launches per environment step.

Additive averages over three equal forty-step windows, milliseconds per step:

| Hardware/backend | Profiled wall | Physics graph | Reset eager GPU | Other eager GPU | Uncovered timeline |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX FPGS | 40.596935 | 7.839719 | 7.326885 | 0.652209 | 24.778122 |
| RTX MJWarp | 32.084654 | 27.776400 | 0 | 0.643444 | 3.664810 |
| GB FPGS | 36.116111 | 7.134495 | 3.594166 | 0.748295 | 24.639156 |
| GB MJWarp | 32.442802 | 27.988197 | 0 | 0.754197 | 3.700409 |

No graph/eager overlap occurred; explicit overlap categories and synthetic tests
guard that accounting. Uncovered timeline is neither a graph span nor an eager
device operation: it is NOT automatically CPU execution, Lab cost or removable
overhead. Graph intervals also include internal dependencies, not just SM busy
time. Final drain is only 0.051-0.119 ms per complete forty-step window.

These profiled walls are separate from the earlier 1,000-step synchronized wall
medians (RTX FPGS/MJ 35.465/32.027 ms; GB 31.160/32.640 ms). Tracing particularly
affects eager reset work. Do not subtract values from different windows, add
host waits to device time, or assume independent component medians add up.

In the original RTX timed window, inclusive host cost per reset call was
24.305/24.373/24.108 ms FPGS versus 23.191/23.690/23.568 ms MJ across three rounds.
Per-call costs are similar; frequency differs by about fortyfold. Reset sizes
differ, so this is not a matched-work per-call efficiency comparison.

## Actual reset causes

A fresh diagnostic records original termination bits, exact force/velocity
operands, episode ages and actual reset selections before reset. Original
compute/reset each execute once; no extra sensor refresh, per-step host read,
random draw or behavioral change is introduced. Diagnostic timing is not accepted.
All four children exit zero with exact recording consistency, finite state,
two capacity/warning check boundaries, clean sources, owned groups reaped and
final devices idle. Each observes 4,096 worlds for 1,000 steps:

| Hardware/backend | World resets | Steps with reset | Timeouts | Contact >20 N | Velocity limit | Success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX FPGS | 28,787 | 980 | 24,568 | 4,269 | 0 | 1 |
| RTX MJWarp | 28,672 | 24 | 28,666 | 1 | 0 | 5 |
| GB FPGS | 28,814 | 976 | 24,592 | 4,267 | 0 | 0 |
| GB MJWarp | 28,672 | 24 | 28,666 | 1 | 0 | 5 |

The world-reset count is a union: FPGS has 51/45 RTX/GB timeout/contact overlaps.
Nearly equal numbers of worlds reset overall, but FPGS spreads them across
almost every batch step. This is not 98% of worlds failing. FPGS contact-trigger
median episode age is 117/116, minimum 24/27; this is not an immediate-reset loop.
Median triggering force is 23.495/23.468 N, maximum 85.322/71.917 N. MJ's one
contact event has age 73 and force 24.133/24.151 N, so its sensor is not universally
zero. Normal timeout age remains 150 steps.

The unchanged normal-reset routine runs four full-population IK iterations,
model-wide Jacobians, repeated forwarding and synchronization; only final writes
select resetting worlds. Small subsets can incur substantial batch work. Neither
backend ordinarily rebuilds its model or recaptures its graph on reset. Frequent
resets already occur before A+B in corrected baseline108: round-one RTX/GB
0.974/0.957 calls per step. This is not newly introduced by the optimizations.

The contact disparity is a behavioral-quality concern, not permission to raise
the threshold, suppress resets or assume either solver is the correct reference.
Source inspection identified a missing fixed-root reset-pose propagation into
MJWarp mocap state, now reproduced in a small actual MJWarp-on-Warp-CPU test
(CUDA hidden, not the native MuJoCo CPU solver). Both FIXED and fully locked D6
articulation roots retain stale internal mocap/xpos after a root write, masked
flags-zero reset, generic FK, and both interval-skipped and interval-refreshed
solver steps. Internal root position error is 0.04288191 m. Returned Newton
body poses have zero error because final publication runs generic Newton FK
against the current root transform, concealing the internal mismatch.

The positive control explicitly sends JOINT_PROPERTIES notification: mocap
updates immediately and internal root xpos agrees after the next step; the
other world remains unchanged. Root independently reread and reran both cases,
producing byte-identical successful evidence.

A subsequent actual Keyboard GPU readback confirms the mismatch in all sixteen
sampled roots per card, after 200 original warmup steps in a 512-world run.
The live articulation mapping identifies FIXED roots (not inferred locked D6),
each mapping to MJ mocap 1 and body 8. Current intended root and Newton public
pose are exactly equal. MJ mocap and internal root instead disagree with them:

| Sampled error, identical statistics on both cards | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| Root position, meters | 0.0154801 | 0.0277712 | 0.0480247 |
| Root orientation, radians | 0.0519354 | 0.4792413 | 0.7180327 |

The maximum rotation error is about 41 degrees. Queued model-change flags are
empty; model and graph identities are preserved. The observer performs no extra
forward, notification, reset, sensor refresh or physics call. Both children pass
recording, finite/capacity, source and process-cleanup checks. Independent FP64
recomputation of the saved transforms confirms exact Newton pose agreement;
the observer's tiny Newton angle (at most 1.64e-7 rad) was mixed-host-dtype
normalization error, not a runtime discrepancy.

This proves an actual bridge inconsistency in the sampled 512-world run, not in
every world of the earlier 4K capture. It does not quantify how much of the
4,269-versus-one termination disparity it causes. The earlier timings remain
measured costs, but are not accepted equivalent-physics backend comparisons.
The FK dirty mechanism itself correctly refreshes Newton body state; the missing
propagation concerns the separate MJ representation. No solver or Lab fix has
been applied. Extending the earlier FPGS/collision-only change scope to the
Newton-side MJWarp reset bridge has been requested; implementation awaits the
user's answer. A corrected reset-distribution and timing run is still required.

## Evidence and reproduction

- Original balanced captures: `/tmp/fpgs-mj-keyboard-balanced-20260912-01`;
  manifest `2c567eaf412e3b16c16afe68281d941d64777b051d44bab95eb9def67c51b043`.
- Same-window helper: `/tmp/fpgs-keyboard-same-window-Afvp9L/analyze.py`, SHA
  `cf8708aa40cc35a397fac5d1df170853404dd2759662f637e114262501939bc4`.
  Root reread analyzer/tests and reran six CPU tests and all twelve captures.
  Root output is byte-identical to `evidence.json`, SHA
  `08308ae5adc6585761531c66ed265b3f19e9f19b092c3af4170089bf16dfb8b0`.
- Successful diagnostic: `/tmp/fpgs-keyboard-termination-4096-20260912-02`;
  paired manifest `be4605f10c278ed9ca2174979fede536573a6841d4eee5bd06e099a8df97a403`.
- Recorder: `/tmp/fpgs-keyboard-termination-v2-6OMNnO/READY.md`; observer SHA
  `e64b545dab823a4df78facb7d5130e07ab6f518e7c84bb933ac7c97ed766451b`, parent SHA
  `115a09bdfde52572df0561c65cb474d9c9f8f445b70c3a751286dc9b8b18ca78`.
  Eight CPU tests pass independently twice.
- Tiny runtime reproducer: `/tmp/fpgs-mj-root-reset-cpu-EeDChS/repro.py`, SHA
  `5e70f87ea7158517fe3060e895bd13f62dcfdd31465409560acb2ed8ddba4782`.
  Agent `result02.json` and independent `root_result03.json` are identical,
  SHA `43074f353af6752959ff413c15329a9e585853ca4ed780198824ebe05dfe12b6`.
  Both cases and source guards pass. The earlier failed adapter expectation
  (`attempt01.json`) is retained: it mistakenly expected returned Newton poses
  to be stale too, before the final generic-FK publication was accounted for.
- Actual task readback: `/tmp/fpgs-keyboard-root-readback-paired-512-01`;
  paired manifest `6adcc915830b65dc0895f567f825099f64768ce5d0b6ef814b78fa9f7bc30f6a`.
  Helper `/tmp/fpgs-keyboard-root-readback-KwwjYQ/observe.py`, SHA
  `4180cccb9b912942050059531cc48eb1d791c1065fee7c075dace9a70bc00aa9`;
  parent `ff51bd270667db19f2956a4d0cc4c8faba33e0eb1ed1c0519294dfd0ae547378`.
  Five CPU tests pass independently twice. The independent saved-transform
  audit is `RESULT.md` in that helper directory, SHA
  `1a2c7a12f782187f776864b8265ccdf043a8f50dd3f722e10b56b8422cedb220`.
- First attempt `...termination-4096-20260912-01` remains FAILED and preserved:
  both FPGS raw histories were saved, but cleanup accessed a manager already
  deleted by unchanged Lab close. No MJ run or successful parent is claimed.
  The new regression fails old source with that exact error; v2 only retains
  the manager reference for cleanup. A fresh four-child rerun passes.

Use fresh output paths and no other GPU owners:

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-project --python /home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910/.venv/bin/python python /tmp/fpgs-keyboard-same-window-Afvp9L/analyze.py /tmp/fpgs-mj-keyboard-balanced-20260912-01 --output /tmp/fpgs-keyboard-same-window-new.json
CUDA_VISIBLE_DEVICES='' uv run --no-project --python /home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910/.venv/bin/python python /tmp/fpgs-keyboard-termination-v2-6OMNnO/run_pair.py --output /tmp/fpgs-keyboard-termination-new --timeout 1800
CUDA_VISIBLE_DEVICES='' uv run --no-project --python /home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910/.venv/bin/python python /tmp/fpgs-keyboard-root-readback-KwwjYQ/run_pair.py --output /tmp/fpgs-keyboard-root-readback-new --timeout 1200
```

Large captures and source-pinned diagnostic helpers remain local. This report
does not imply a portable artifact bundle, accepted cross-backend physical
parity, training throughput, or completion of the additional 2-4x target.
