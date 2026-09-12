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
