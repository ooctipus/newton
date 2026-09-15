# General FPGS sleeping: first integrated checkpoint

Base: retained `ca0d427af809571bb5501f644c1a6e03990cd2a8` from
`ooctipus/newton`. No Isaac Lab changes or dependency-pointer updates.

## Candidate card (2026-09-15 20:07 UTC)

User direction: implement generalizable sleeping, referencing MJWarp.
Derive independent dynamic components from model topology, not task names,
body names or a particular DOF count. Static anchored skeletons do not couple
their dynamic branches; responsive ancestors do. Unsupported couplings remain
awake until their complete consumers and wake propagation are supported.

Reference: installed MJWarp `_src/sleep.py`, `_src/island.py`, and their
forward/collision/solver consumers. Borrow minimum-awake counters, explicit
sleep/wake ordering and device-owned masks. Do not borrow capacity truncation.
MJWarp still runs some full producers; active-list creation alone is not a gain.

First causal integration: contact-free, independent components supported by
the existing linear prismatic state owner. All current-contact incident
components wake before dynamics. Keep full collision and original contacts,
row capacities, iteration budgets and force reporting. This deliberately does
not yet sleep resting contact stacks or dynamically connected contact islands.
The topology/lifecycle is shared; the initial consumer capability is limited.

Remove repeated force evaluation, joint integration arithmetic and body-state
publication for sleeping components. Copy frozen public state across output
buffers rather than trusting stale buffer contents. Preserve complete awake
fallback. A newly sleeping body gets a final current publication after velocity
is zeroed. Any applied force, changed drive target, authored state change, reset or
relevant model notification keeps it awake or wakes it. Passive equilibrium is
admitted only after multiple quiet steps, with finite checks.

Existing Keyboard owner exposure is approximately 0.72 ms publication,
0.27 ms direct forces, plus integration/limit handling. Not all is removable.
Hypothesis: retiring most inactive-branch work can save about 0.5–1.0 ms
of the retained ~6.9 ms RTX whole-physics step. Added costs are current-contact
wake classification, per-component quiet/force checks, frozen-state copies,
and the earlier collision-event wait. This is an initial structural milestone,
not a demonstrated additional 2–4x gain or a forecast for active robot tasks.

Use existing benchmark recipes and paired RTX/GB300 execution, one job/device.
Initial checkpoint by 21:30 UTC: integrated correctness and first whole timing,
or a concrete diagnosed blocker and revised test. No silent mapping sweep.
Required tests cover non-Keyboard branch counts, shared responsive ancestry,
contact/force/reset/model-change wake, A/B and in-place states, finite resting
behavior and unchanged awake fallbacks. Unsupported dispatch is not measured
generalization. Promotion requires repeated paired whole timing and relevant
cross-task checks; no speedup is claimed by this card.

### 20:14 refinement from actual imported spring representation

The Keyboard USD authors its springs with `DriveAPI`; Newton imports these as
PD target gains, not necessarily passive `joint_spring_stiffness`. Rejecting
every nonzero PD gain would therefore remove the very population under study.
Before GPU timing, extend the general admission contract to a stable finite
position target with zero target velocity. Remember the live target and wake
on any change; model notifications invalidate gains. Nonzero applied forces
and moving targets remain awake. The same rule applies to any admitted model,
not a Keyboard exception. Add a static-PD equilibrium/changed-target regression.

## Reference and implementation scope

Reference source: [MJWarp sleeping](https://github.com/google-deepmind/mujoco_warp/blob/main/mujoco_warp/_src/sleep.py)
and [island/active-DOF mapping](https://github.com/google-deepmind/mujoco_warp/blob/main/mujoco_warp/_src/island.py).
The installed files used for the design have these SHA256 values:

- `sleep.py`: `5b50688a8949afa30df1ae6ce2195132fdf50cf6ed821746badd8e06b92d91b0`
- `island.py`: `e8ed03ca36d5bf2f28335c35ac3c71785c45449154c8cb6258a9274263c55473`
- `forward.py`: `7b7ffa3d914b6adf65f228261e207998d5bed1e59d4fb17f5bc6ac796aacf490`
- `solver.py`: `509b43da297dc6e49efeacdbb59df675213e84afbdb26770471a5ba1855a24b3`

This first integration uses device masks, not yet compact awake work queues.
It has no contact-island sleep/collision filtering. Full input capacities are
retained. Unsupported consumer configurations use the original GPU path.
`FEATHER_PGS_SLEEPING=1` is experimental and default-off. Optional
`FEATHER_PGS_SLEEPING_DIAGNOSTICS=1` reports eligible/asleep counts only at the
existing host capacity-check boundary, outside the timed graph. It does not
change physics, buffers or sampling. No benchmark framework is added.

## Native correctness checkpoint

The first four selectors passed numerical assertions on both GPUs. The graph
fixture emitted capture-event-to-eager wait warnings; it now joins completed
graph work and restores ordinary eager events before changing execution mode.
This is a test ownership correction, not a solver synchronization bypass.

Independent review then found a genuine new-policy bug: at tiny positive dt,
16 low-velocity steps can occur during strong initial PD acceleration. A real
solver regression with dt=1e-8 and a changed fixed target reproduced premature
sleep on both GPUs (`body_awake=0`, expected 1). Before benchmark, require the
current solved acceleration also to remain within the velocity tolerance over
a 0.05-second physical horizon. This distinguishes equilibrium from short
elapsed motion without changing any task timestep or iteration allowance.
Pre-fix controller SHA256:
`1d04cb243fd2a89634f7928d2a0bb8a435159a45c293ae63c37a29198a1bad94`.

After that correction, all four selectors pass on RTX and GB300 (2.55/2.70 s
test time), including genuine first sphere-pair collision, loaded contact-force
comparison, PD equilibrium/target/applied-force wake, tiny-dt acceleration,
captured masked reset, graph-to-eager transitions, in-place states, public FK,
and inertial/drive-gain notification. No CUDA event errors remain. The ordinary
target-layout deprecation warning is inherited. Original CPU prismatic tests
also pass (9 selectors, 2 CUDA-only skips). This is focused qualification, not
a resting-stack sleeping implementation or a full training-quality study.

## First whole-task discovery

Runtime `20ce737c7c75dd0bc2d85441dcab69a04ad2ad01`, retained base `ca0d427a`.
Both paired jobs completed with finite states and passing capacity checks.
One round, fixed Keyboard 4K recipe, 200 warmup, 40 wall and 40 graph steps:

| GPU | Retained physics ms | Sleeping physics ms | Retained / candidate |
| --- | ---: | ---: | ---: |
| RTX PRO 6000 | 6.926593 | 7.277697 | 0.95176x |
| GB300 | 5.991970 | 6.412042 | 0.93449x |

This is a measured loss, not promoted. Both snapshots show 442,368 eligible
components out of 446,464, but **zero asleep**. Only about 27–29K components
are contact-incident. The prototype pays controller overhead without observed
sleep admission. Before rejecting the mechanism or adjusting tolerances,
inspect can-sleep counts, quiet counters, solved motion and current force/target
inputs at the same existing host boundary. No GPU kernel changes are needed
for that diagnosis. Source/control budgets and all capacities stay fixed.

Raw discovery: `/tmp/fpgs-general-sleeping-paired4k-20260915-01/manifest.json`.

## Diagnosed wake-scope loss and targeted correction

Same-physics diagnostic captures at `e4164aa8`:
`/tmp/fpgs-general-sleeping-gates-YkVfGXlI/gpu{0,1}/capture.log`.
More than 438K of 442,368 eligible components pass the output acceleration
gate, and roughly 415–417K pass current input/contact admission. Quiet counters
are all zero at three of four boundaries; the remaining boundary has 416,158
components at exactly eight steps, never the required sixteen. Fixed Lab
performs eight solver substeps per environment step. A partial fixed-root
reset issues unmasked JOINT_PROPERTIES; our initial controller invalidated
every world on that notification. This erases unrelated quiet history.

Correct in Newton without changing Lab or motion thresholds: snapshot the
complete documented JOINT_PROPERTIES set (model joint_q, joint_X_p, joint_X_c)
and compare actual device values on notification. Include immobile roots via
joint-to-world mapping. Invalidate only changed worlds for this flag alone;
nonfinite/global changes and all other property flags remain conservative.
Snapshots and masks are persistent graph-safe arrays, not host readbacks.

Regression-first: an unchanged notification incorrectly woke 215/218 native
components on both GPUs before the correction. Afterward all four native
selectors pass on both cards (3.67/3.78 s), including unchanged notification,
changed fixed-root wake confined to one world, current public FK, and existing
contact/force/target/reset/in-place/graph controls. Full pre-commit passes.
This fixes missing task admission; task-level benefit is not yet established.

## Corrected whole-task discovery

Pinned runtime `2c3d532d3261daf2f063d828b1f4bd408e30dbd4`, unchanged retained
base/recipe/budgets. Raw `/tmp/fpgs-general-sleeping-paired4k-20260915-02/manifest.json`.
All capacity flags pass and observed states are finite. This single discovery
is still **not a performance win or promotion**:

| GPU | Retained physics ms | Sleeping physics ms | Retained / candidate |
| --- | ---: | ---: | ---: |
| RTX PRO 6000 | 7.028155 | 7.073679 | 0.99356x |
| GB300 | 6.065000 | 6.286362 | 0.96479x |

RTX boundary asleep counts are 413,841 and 406,747; GB counts are 415,711 and
406,804, out of 442,368 eligible. Admission now works, but 92–94% sleeping
does not imply comparable work removal. Wall-step measurements are
31.401/34.189 ms retained/candidate on RTX and 29.381/29.495 ms on GB;
these are environment stepping, not full training. Do not update MJWarp ratios
or retained runtime based on this candidate.

Review also found that the inherited response plan normalizes global
articulations to world zero. Sleeping must retain raw global ownership:
global components remain ineligible, and changed global joints broadcast wake.
A CPU topology regression fails before preserving raw articulation worlds in
both sleeping maps. This conservative edge fix does not change the replicated
Keyboard map. Next diagnostic: reuse node tracing to separate retired owner
work from controller cost and still-full producers, not another mapping sweep.

## Kernel-level cost closure (no performance promotion)

Runtime `b00df921`; paired diagnostic manifest:
`/tmp/fpgs-general-sleeping-nodes-20260915-01/manifest.json`.
Same task, control budgets and capacities; 200 warmup, 40 wall steps and ten
node-traced physics steps. All four runs finish with passing capacity flags
and finite observations. Node instrumentation is diagnostic, not a replacement
for the preceding uninstrumented-node whole-graph comparison. Raw manifests
contain the exact commands, environment, pins and check outputs.

Summed graph-kernel time per environment step (milliseconds):

| GPU | Original force/integrate/publication | Awake versions | Removed | Added sleep controller |
| --- | ---: | ---: | ---: | ---: |
| RTX PRO 6000 | 1.143610 | 0.480852 | 0.662759 | 0.541249 |
| GB300 | 0.532205 | 0.364003 | 0.168202 | 0.398493 |

Controller total includes current-contact marking, begin checks, finish checks
and in-graph awake-mask publication, excluding eager reset/model notification
work and memory nodes. Kernel sums overlap and are not critical-path savings.
Thus real work retirement occurs, but its controller cost nearly consumes the
RTX saving and exceeds the GB saving. GS also varies across these trajectories
(RTX 1.516/1.208 ms, GB 1.369/1.562 ms baseline/candidate); do not attribute those
differences to a changed GS implementation, which this candidate does not have.

Drive preparation, scalar inverse mass/solve, prediction and limit preparation
still execute across the full domain. Their entire baseline bundle is only
0.331 ms RTX / 0.269 ms GB in this capture, so a separate awake-queue conversion
of those kernels alone cannot fund an additional 2–4x whole-physics gain.
Collision, contact production and contact scheduling remain full and correct.
No capacity changes, artificial contact filtering or shortened iterations are
introduced to manufacture a saving.

The larger next architecture is a complete awake-component owner, with shared
queues replacing producers rather than supplementing them, followed by actual
resting-contact island support and collision wake/incremental processing.
MJWarp's masks/compact maps and collision-wake ordering are useful references;
this first implementation does not yet provide that full lifecycle. A compact
active-limit schedule is possible without changing global row/DOF strides, but
there is no evidence that limits dominate enough of GS to justify another
mapping sweep. Retain this branch as a tested default-off foundation, not a
new accepted performance baseline. Representative-task MJWarp ratios stay
unchanged; unsupported consumer dispatch is not evidence of general speedup.

Final focused tests after the global-world guard: all four selectors pass on
both cards (2.45/3.29 s), including the regression-first global topology check.
Full pre-commit passes. Full sustained-contact/training qualification remains
required before any production promotion. No Isaac Lab files were modified.
