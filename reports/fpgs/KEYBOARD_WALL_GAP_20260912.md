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
Source inspection suggests a missing fixed-root reset-pose propagation into
MJWarp mocap state. Runtime verification is pending: Newton's published body
pose alone cannot settle it because that publication uses generic Newton FK,
while internal MJ geometry may differ. Force publication, contact laws and
trajectory differences remain alternatives until direct checks distinguish them.

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
- First attempt `...termination-4096-20260912-01` remains FAILED and preserved:
  both FPGS raw histories were saved, but cleanup accessed a manager already
  deleted by unchanged Lab close. No MJ run or successful parent is claimed.
  The new regression fails old source with that exact error; v2 only retains
  the manager reference for cleanup. A fresh four-child rerun passes.

Use fresh output paths and no other GPU owners:

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-project --python /home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910/.venv/bin/python python /tmp/fpgs-keyboard-same-window-Afvp9L/analyze.py /tmp/fpgs-mj-keyboard-balanced-20260912-01 --output /tmp/fpgs-keyboard-same-window-new.json
CUDA_VISIBLE_DEVICES='' uv run --no-project --python /home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910/.venv/bin/python python /tmp/fpgs-keyboard-termination-v2-6OMNnO/run_pair.py --output /tmp/fpgs-keyboard-termination-new --timeout 1800
```

Large captures and source-pinned diagnostic helpers remain local. This report
does not imply a portable artifact bundle, accepted cross-backend physical
parity, training throughput, or completion of the additional 2-4x target.
