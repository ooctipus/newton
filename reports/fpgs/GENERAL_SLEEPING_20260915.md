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
is zeroed. Any applied force, actuator drive, authored state change, reset or
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
