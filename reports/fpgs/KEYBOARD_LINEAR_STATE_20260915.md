# Keyboard translation-only state: pre-code decision

Decision: 2026-09-15 09:36 UTC. Base `28f93014520504e90be34a304507368c7c82c4cc`
(retained Franka integration, accepted Keyboard runtime unchanged from b1bad).
Branch `ooctipus/fpgs-keyboard-linear-state-20260915`. No shared-tier,
read-only-row or stationary-limit experiment is included.

## Work to replace

The existing Stage7 prismatic plan already admits fixed roots and independent
translation-only leaves, but its leaf publisher still invokes the complete
general spatial-body dynamics law. Cold Stage1 still traverses all leaves
serially. The direct scalar mass owner reads twelve inertia terms to project
them against a motion vector with identically zero angular part.

Replace that complete state-production boundary for admitted leaves:

- Reuse the topology-only plan and current frames, axes, q and passed Stage1
  velocity (including any existing prescaling). Compute leaf poses/motion in
  parallel during cold repair and next-state publication.
- For zero angular velocity and zero bias acceleration, Coriolis and inertial
  bias vanish. Publish the full gravity wrench, current public pose and COM
  velocity directly; do not rotate inertia merely to multiply it by zeros.
- The mass projection is `m * dot(axis_world, axis_world) + armature + max(K,0)`.
  Preserve non-unit axes, current drive coefficients and held-mask behavior.
  Remove ordinary key inertia materialization only when no retained consumer
  needs it. Preserve all generic articulated-arm and other unsupported paths.

This is not a contact algorithm or capacity change. Every contact, impulse,
force callback, timestep, substep and original iteration allowance remains.
No Isaac Lab or parent dependency pointer changes.

## Consumer and lifetime requirements

The global parallel composite schedule already excludes the direct-diagonal
key articulation. Exceptional masked refresh/composite paths do not. Retain
their original complete materialization for this first implementation; do not
silently feed them absent buffers. The scalar mass path must cease reading the
ordinary key compact-inertia terms before those terms are retired.

Cold root traversal currently stamps cache validity. Preserve pre-launch
validity in one small persistent device buffer before shortened root repair;
the leaf pass must read that snapshot, not the newly stamped validity. Charge
its copy and launch. Finish all leaf writes on the current stream before any
inertia-ready event or force consumer. Keep reset and model-notification
invalidation, cache source identity and snapshot-backed modes correct; reject
unsupported opt-in combinations rather than silently changing their owners.
Prefer the existing cache contract over a new state machine or proof system.

## Cost and stop rule

Source-equivalent RTX node evidence is the baseline under
`/tmp/fpgs-keyboard-shared-tier-nodes-paired4k-20260915-01`:
next publication 0.867157 ms, cold/cache FK 0.712224 ms and direct mass
0.124224 ms per environment step. Only the first cold FK launch is expensive
(about 0.67472 ms); later cache hits are already cheap. Publication runs eight
times. The 1.703605 ms sum is an upper envelope, not an exclusive saving;
mass overlaps other work and required pose/state production remains.

Fund one integrated candidate targeting at least 0.7 ms RTX whole-physics
saving (roughly 9-10% at the current 7.4-7.9 ms baseline). This needs a large
cut across the cold and next-state producers, not a scalar-mass micro-tune.
All new copies, leaf launches, fallback materialization and joins are charged.
First integrated screen within 90 minutes; no silent extension beyond two
hours. If the full screen misses, use one existing node diagnosis to locate
the gap; no launch-size grid or follow-on polish without a new costed cause.

Use the unchanged paired4K driver, 200 warmup and 40 wall/physics steps,
fixed 704 rows, 147456 contacts and 57344 broad-phase pairs. Baseline is the
accepted runtime, not a failed prototype. A first screen is discovery only.

## Minimal qualification

Extend existing prismatic-publication and scalar-mass tests, regression-first.
Use the generic solver and independent `newton.eval_fk` oracle. Cover poisoned
cold buffers, warm Stage7-to-Stage1 reuse, partial reset and captured replay;
actual changes to root/child frames and non-unit axes; COM, mass, inertia and
gravity notification; external/passive/actuator forces; held and requested
mass refresh; and unsupported revolute/free articulations. Check retained
physical/public outputs, not obsolete unread inertia buffers or bit identity.
No tolerance widening or separate test framework. Default off until repeated
paired timing and physical qualification justify retention.

## First implementation and early-screen gate

The first implementation adds `prismatic_linear_state.py` and narrow host
hooks. `FEATHER_PGS_PRISMATIC_LINEAR_STATE=1` requires cached matrix-free direct
diagonal response, with every direct-size articulation exactly covered by the
existing prismatic plan. Unsupported/grouped/fused/snapshot owners reject the
opt-in. The default remains off; the old publication module is unchanged.

Cold repair copies the original articulation validity before the shortened
root launch and completes its body-parallel leaf pass before the existing
inertia/force events. The persistent addition is one int per articulation
(32 KiB for the 4K two-articulation recipe), copied each substep. Warm leaf
repair returns from that snapshot. Next publication retains the original
body-parallel mapping. Exceptional masked full inertia and composite work,
all drives and force consumers, and all constraint owners remain unchanged.

Source checks confirm the default direct-mass native function is unchanged,
the new scalar mass retains its argument ABI, and the new next-state finalizer
retains the original prismatic finalizer ABI. Independent runtime and lifetime
reviews found no concrete blocker. Native source SHA256:

- `prismatic_linear_state.py`:
  `f0cd1d10a516b182630f6fd97137e8d6514f53accb4d8bffcc06733ebd687da7`.
- `solver_feather_pgs.py`:
  `f5fe22374ec9f60a9400a5e73cb8e334195cc6833afb028b0b460d6c27493d3e`.

Regression-first CPU session 80969 failed on the missing
`_PRISMATIC_LINEAR_STATE` API before implementation. Focused tests reuse the
existing prismatic-publication and scalar-response modules; small CPU forced
state-boundary fixtures do not feed elided inertia into generic mass consumers.
Focused four CPU controls passed (session 34821). The complete existing
prismatic module plus new scalar-mass oracle passed seven tests with one
explicit CUDA skip (session 57668, eight tests total). These include cold/warm
repair, partial reset, notified current frames/inertias/gravity, and scalar
mass with poisoned retired terms. Existing external/passive full-step coverage
remains a legacy-path control, not qualification of the new CUDA owner.
Full `uvx pre-commit run -a` passed (session 97622); the first pass only
formatted the two test files and did not change runtime source.

After focused CPU/source checks, the first clean paired whole run is expressly
an **unqualified cost screen**, not promotion. Full native cold/reset/current
mass and captured lifecycle controls remain a retention gate. This sequencing
does not claim that source review or CPU compilation proves CUDA physics.

## First screen: constructor failure, no candidate timing

The first paired screen at runtime `cd3ec68d5a5c65eb54abc1faaef028fcbd4f794c`
failed before either candidate was constructed. Root parent 58617 was reaped
with exit 1; both baseline captures completed, but neither candidate produced
a timing report. Preserve the original artifacts at
`/tmp/fpgs-keyboard-linear-state-whole-paired4k-20260915-01` (manifest SHA256
`f0319b699acf1285ccab3b246d80a544a6fe2655520b7d58889061d2ca3a9f91`).

The admission check incorrectly boolean-indexed the N+1 articulation-start
array with an N-articulation mask (8193 versus 8192 in the real task). The
intended comparison is against `starts[:-1][admitted]`: the final entry is
the end sentinel, not another articulation. Correct only this host slice and
add a real-plan CPU admission regression. No native math, cache ordering,
physical tolerance or solver budget changes are involved.

The new actual-plan CPU regression first reproduced the same `IndexError`
(session 36463, four starts versus three articulation flags). After the slice
fix, all eight CPU controls passed with one explicit CUDA skip (session 69208,
nine tests total, 0.364 s). An earlier invocation used a nonexistent scalar
test-class name; its loader error was corrected without source changes.
AST comparison confirms every native function is unchanged from `cd3ec68d`;
only `configure_linear_state` differs. Corrected module SHA256:
`e3bfb42113b651aabcce92902771b025109e1d7186a31edeae02b225de733e1c`.
The solver-hook hash remains `f5fe22374ec9f60a9400a5e73cb8e334195cc6833afb028b0b460d6c27493d3e`.
