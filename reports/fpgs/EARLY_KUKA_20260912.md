# Experimental Kuka ZERO-world early publication

Default-off `FEATHER_PGS_EARLY_KUKA=1`, based on exact combined fb0f075935c4abcfcf8f3932ae615774551a2d95.
No task, dt, substep, eight-sweep, contact law, capacity or original GS-equation
changes. This is a scheduling experiment, not an accepted speedup.

The measured whole publication boundary costs2.229548ms RTX /1.896704ms GB per
environment step, including qdd, free-root transport, integration, FK and
inertia/bias/COM-velocity finalization. Current actual two-call16K ZERO masks
select83.954–84.967% of worlds and the same homogeneous body/joint/DOF fractions.
An optimistic uniform-cost RTX ceiling is about1.87ms /13.4% of current13.974ms,
before extra launches, list construction, waits and contention. This is not a
20% or4x claim; the root requested one meaningful whole-graph screen.

## Work and ownership

Retain the current classifier, raw geometry, row producers, paired/general
independent fork and all GS equations. Move the original full v_out seed copy
before the new fork. Compact selected/complement articulation IDs using only
the authoritative current resolved mask, then run original selected
qdd/transport/integration/FK equations on a private stream. Global35 coordinates
include prescribed6, not just the29 response coordinates. Every original
publication statement is retained by tested index-only source adapters.

Original raw allocation checks resolved before geometry; metadata/J/MF producers
check the resulting inactive path/flags before shared S/origin reads. Original
zero-count native paired/general kernels return before velocity writes.
Inactive old impulse tails remain inactive; public raw path=-1 publishes zero
force without consuming them. No additional full-capacity impulse clear is
introduced or claimed.

The unmasked original MF body inverse reads old free-body inertia. Selected FK
does not write inertia. Immediately after the original inverse launch the main
stream records its reader event; only then is early finalization queued behind
that event. No wait relies on an event that has not yet been recorded. Original
Stage7 publishes the late complement and joins early done before declaring the
whole-state cache source ready. The existing paired/general fork is unchanged.
Host active is cleared only after queuing the mandatory join; subsequent graph
or eager calls do not inherit an external captured-event wait.

Added cost includes one bounded articulation compaction, one two-int count clear,
early/late publication dispatches, three event dependencies and original-equation
index guards. Early qdd/joint/body grids are bounded by articulation count times
maximum actual DOFs/joints; unused tails are count-guarded before reading IDs.
List storage is constructor-owned, no new per-step GPU allocation. Any contention
or underfilled heterogeneous launch cost is charged to the whole experiment.

## Gate and scope

Unsupported constructors keep the full original path. Require the existing
CUDA ZERO Kuka law, direct non-snapshot cache publication, no grouped/prismatic
replacement, no particles/propagation, and original MF speed-limit ownership.
Validate distinct input/output storage and articulation-local parent/body/DOF
ownership. Resets/notifications between completed steps retain original cache
invalidation; mid-step reentrant mutation is rejected.

Tests compare all five original publication kernels on exact retained Kuka512
refresh/reuse snapshots, mixed/full/empty/growing/shrinking cohorts and poisoned
list tails. They cover prescribed DOFs, transport, held/full/compact inertia,
public q/qd/body poses/velocities, current S/origin/bias/cache and the original
MF inverse reader. CUDA stream/graph controls are prepared for root-only launch.
Component tests are not identical-trajectory or universal convergence proof.
After native controls, one paired16K whole-graph screen decides whether this
mapping delivers a meaningful gain; a percent-selected argument cannot accept it.

Exact prior ownership/card artifacts:
`/tmp/fpgs-kuka-zero-publication-mANgf9CK/{CARD.md,evidence.json,audit.py}`.
Exact completed current mask capture:
`/tmp/fpgs-kuka-zero-cohort-paired16k-20260912-01`.
Source/CPU/offline readiness and mask audit:
`/tmp/fpgs-early-kuka-ready-e9Edn0yC`.

## Completed device controls and rejected timing

The frozen runtime is `fb32ce78d1e0d4242d274f67e492d8ed7478af2e`.
Root completes and reaps **65 actual GPU controls per card**, all passing:
old 45 plus new 10, then the separate 10 original simple-world controls. The
publication/stream/graph and original ZERO laws are tested without skips;
this is bounded correctness evidence, not a universal trajectory guarantee.

The first complete 16K paired node screen is **rejected**, not promoted:

| Disjoint graph-span accounting, ms/env-step | RTX baseline | RTX early | GB baseline | GB early |
| --- | ---: | ---: | ---: | ---: |
| Publication-exclusive busy | 2.228693 | 1.712545 | 1.890346 | 1.559239 |
| All non-publication work union | 11.513141 | 12.960725 | 12.048548 | 13.165468 |
| Graph span uncovered by device activities | .196203 | .273014 | .182628 | .306055 |
| Complete graph span | 13.938037 | 14.946284 | 14.121523 | 15.030761 |

These disjoint categories add to graph span. Concurrent owner durations must
not be added again; uncovered time is not an attribution to CPU or Lab.
This is one paired three-profile-step node window (12 correlated graph roots,
24 Newton substeps per arm), not balanced repeated performance acceptance.

Actual selected-world boundary counts are 12,647/12,584 RTX and 12,529/12,632 GB,
of 16,384: about 76–77%, not the earlier 84–85% two-call sampler. The selected
cohort still owns all three articulations and all 35 global DOFs, including
prescribed 6; response width remains 29. All four candidate boundaries pass
disjoint/exhaustive actual-buffer checks.

All child checks complete/pass; all four solver and 13 narrow/collision sticky
flags are zero at both boundaries in every arm. Public contacts 4,000,000,
dense 192, MF 64, propagation 192, seed 0, 200 warmup steps, 40 wall steps,
sim_dt 1/120, two Newton substeps, decimation 4 and maximum eight GS sweeps are
unchanged. Parent completion, source and idle guards pass. No Lab, task/model
parameter or capacity changes were made.

### Why intended overlap did not yield a gain

Early publication overlaps other work for 71.2%/75.4% of its duration, but
complete publication union grows 2.231147 → 3.480141 ms RTX and
1.892885 → 3.394185 ms GB. Early selected FK alone costs 1.575873/1.680565 ms,
more than original all-world FK 1.437109/1.374922 ms. Including late FK gives
2.243318/2.292192 ms. This is not the .026731/.031808 ms compaction kernel.

The original allocator/contact-prelude/masked-J chain grows from
.897045 → 1.723489 ms RTX and 1.134175 → 1.961203 ms GB. The completed-ZERO to
completed-MF-inverse dependency window grows 2.305365 → 3.692709 and
2.634986 → 4.076747 ms, while ZERO and inverse kernels themselves are essentially
unchanged. This localizes the dilation to newly concurrent publication and
row preparation. It supports contention without uniquely establishing
bandwidth, occupancy or cache behavior.

Early finalization has zero exposed tail after late finalization in all 24
sampled windows on both cards. It finishes 213–309 us RTX / 225–332 us GB before
late publication even starts. The mandatory final join is not the loss.
The FK-to-finalizer wait protects the original MF inertia reader and is not
safe or useful to remove merely because it appears long.

The scalar publication mapping launches articulation-count times maximum-width
grids: early qdd 1.971 times and early integration/finalization 2.8125 times the
original all-world slot counts, followed by original-sized masked late grids.
Early and late FK each launch the full padded articulation capacity. Guards
prevent invalid accesses, not launch cost or mixed-tree iteration tails.

Some solve costs also change with the trajectory. MF active worlds/max rows
differ across boundaries; RTX general duration increases, whereas GB general
duration slightly falls. Therefore the full changed solve tail cannot be
assigned to scheduling, and this timing capture is not a same-input physical
comparison.

Even deleting all candidate publication-exclusive busy, with other work
unchanged, cannot deliver 10% over this paired baseline: the candidate would
need 2.402 ms RTX /2.321 ms GB removed. A tiny compaction/event change does not
address the inflated publication and concurrent row-chain costs. This mapping
is closed; no layout grid or narrow publication retry is selected.

### Frozen provenance

Baseline `064ec8ac455fc4cde557a3b54a1a62624cf56441`; tools
`5a9d8d76dc060b11fbd9672803c902595488caed`; unchanged Lab
`1d8feb82d17dbfab8f0772de56f84deae2cb7974`.

- Runtime early_kuka.py SHA256:
  `2f598b15aea10a01e371f0970a165bafeb93680ce26dc6d530c9351c41ccd176`.
- Runtime solver SHA256:
  `f1cecef6b9986ad5301ff1b4acd7261f03b84e95de766d624fcb9d7c499f9a13`.
- Completed parent:
  `/tmp/fpgs-fourx-early-kuka16k-20260912-01`; manifest SHA256
  `015fc6dad4501f6f5e03e29284752e0c64effa7685c86241ada5f6948dc6fdf7`.
- Frozen causal artifacts:
  `/tmp/fpgs-early-kuka-node-audit-soIL49Ox/{FINDINGS.md,evidence.json,audit.py}`.
  Findings SHA256:
  `f00b876bd4977a58ef89dd7771df30bbc50338de6372916431cef9fbc1bcd019`;
  evidence SHA256:
  `1db6f0ff49b67b3415c63695b031e6d9700514d9c1b779b3ded5e196e474659d`;
  audit SHA256:
  `3574a8ef93d88b82ad91d26d89cc173fff53b0844000750ef93ce07c520fe661`.

The pinned audit includes all 16 input hashes and retains signed timestamp
adjacency discrepancies instead of claiming nanosecond event-order proof.
Actual CUDA controls establish the source dependency separately.
This checkpoint changes only this report; runtime, captures and parent
dependency pointers remain unchanged. The fixed fourfold goal is not achieved.
