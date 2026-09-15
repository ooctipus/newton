# Franka complete current/next kinetic state — bounded prototype

Start: 2026-09-15 00:29 UTC. First source/native checkpoint: 01:59 UTC.
Base: qualified `1a9efc33efbc0f23e1e7676a5edded795f224c97`.
Branch: `ooctipus/fpgs-franka-kinetic-state-20260915`.
Opt-in only: `FEATHER_PGS_FRANKA_KINETIC_STATE=1`. No GPU lease is held by
the implementation agent. No timestep, capacity, iteration, contact, or PGS-law changes.

## Complete boundary and budget (written before code)

Replace current/next primary FK/dynamics caches, composite construction,
generalized bias-force reduction and predictor with one complete compact state
owner plus current-control/held-factor predictor and bounded invalidation repair.
Keep all existing packet/local/general/MF constraint owners and their inputs.
Keep original L9/L6 factor arithmetic, R+K, inverse preparation, refresh masks,
current geometry, and actual public state for every world.

The source-correlated earlier-today Franka current/next dynamics, force, factor,
prediction and publication exclusive union is 1.923489333 ms RTX / 1.554799 ms
GB. Reserve the entire original factor9+factor6 sum, 0.316598 ms RTX. A 1 ms
complete saving therefore requires new finish+predict+repair <=0.606891333 ms
RTX; this is a falsifiable prototype allowance, not an established runtime.
Current complete physics reference is 5.48441275 ms RTX. No world/count fraction
is converted into a timing prediction.

Source: `/tmp/fpgs-tenhour-franka-current-nodes-paired16k-20260914-01`.
G1 mechanism reference: complete 44-body state/force owner, whose measured RTX
finish+predict+repair sum is 3.148854333 ms. Franka has 11 primary bodies (7
revolutes, 2 prismatic joints, 2 welds) plus dynamic and prescribed free roots:
13 bodies, 21 physical coordinates, 15 responding coordinates, 81 primary mass
entries. Four pose/motion scan rounds cover the ten primary edges. All nine
rotated inertia entries, current external COM force shifts, live passive/control
forces, augmented u0, free-root transport and damping remain required.

Retire primary global V/A/F/I/composite producers and their force/predictor
consumers together. Publish current S/origins, complete joint/body state, the
dynamic free body's full 36-entry I_s for the unchanged MF consumer, and the
prescribed body's six-entry body_v_s for unchanged RHS consumers. Do not keep
the retired primary matrices merely for observers. Free-root original body
finalizer remains inside the state owner. Only geometric H9 replaces original
CRBA assembly; original factorization and held lifetime remain.

## Distinction from closed prior work

This is not private-PGS, contact-complete, or paired-late publication. Those
designs altered or incompletely priced the solve boundary. No private Gram,
rows, local solve, active-world compaction, or per-sweep mapping is added here.
The G1 compact moment/current-next mechanism transfers independently of PGS.

## Required correctness and lifecycle

One checked primary9/free6/prescribed6 owner per world; reject unsupported or
changed topology before retired caches are read. Current state and geometric
validity are separate from held L. Preserve scalar device mass request[0],
original mask consumption, force/dt invalidations, and device source-bank tags.
Reset invalidates current/geometry without silently requesting held refresh.
Read live model/control descriptors at each call; retain sticky native status.
External forces are recomputed every substep, shifted from current COM to the
current articulation origin. Root screw basis, integration and transport reuse
the original operations. Bias for next reuse uses the published post-damping qd.

Regression-first tests reuse four saved current/held Franka payloads and original
publication/force oracles. Test complete public state, free/MF and prescribed
service, changed live frames/COM/control, held factors and input immutability,
reset/model notification and graph lifecycle. Compare physical numerical error,
not bitidentity with cancellation-sensitive FP32 FK. CPU and offline SM120/SM100
resource checks precede frozen paired native tests; root owns all GPU jobs.
An integrated original whole A/B, not isolated NumPy or kernel arithmetic counts,
decides whether this complete boundary is useful. No mapping/tolerance grid.

## Source/offline checkpoint, 00:54 UTC

The complete native/host owner and root-owned solver/factor hooks are present.
The new module was absent in the regression-first import (the first environment
attempt lacked Warp and is not counted as that regression). All four pinned
512-world actual Franka ownership maps admit, including primary source schedules
and every physical body/coordinate. Full physical fixture tests are in progress;
no CUDA execution or performance claim is made here.

The producer reuses the existing original integration and free-body finalizer
through checked source seams. The primary subtree owner reduces all six bias
components and all thirteen mass/first-moment/full-nine-inertia components,
then forms H81 only when requested. The predictor reads mandatory canonical
current S directly: unused duplicated private S/origin stores were removed
before native testing. Private storage is 580 bytes/world current+geometry,
660 bytes/world immutable mapping plus360 fixed map bytes: 20,316,520 bytes at
16K. Canonical fallback allocations remain; this is not a net-memory claim.

Existing offline compiler reused at
`/tmp/fpgs-franka-kinetic-state-offline-lWugyWyp/compile.py`.
Offline01 preserves a C++ mixed-auto declaration failure, corrected without
changing arithmetic. Final Offline04 report SHA256
`540cb6341418a7455aec9153421d871d77f2442fcaeaafc4e77a66986c9e1fcf`
compiles all four actual entries on SM120 and SM100 without a CUDA device.
It pins native/host source `8c6ab89df31161e21f9485e5efd3dbff130355126068d264a137d98314299589`
and factor source `c04c1c64c8ca25e71f41bce9cd76f20511a477d9ee699b33a8f3ef1dfd506012`.
Repair uses72/72 registers; finish80/90; both3712 bytes shared,72-byte stack,
zero spills. Current-force predictor40/40 registers,608 shared,zero stack;
factor9 40/40 registers,1808 shared,zero stack. These are compiler resources,
not achieved occupancy or time. Actual finish/predict/repair cost remains the
falsification target, including repair on reset and held geometry requests.

Before freezing, the live external-force lever arm was corrected to the
original public `body_com` convention, distinct from the inertial frame used
for bias and H. Admission now requires immediate articulated response: the
propagation route still consumes primary full spatial inertias and is not
covered by their retirement. Neither change expands the experiment.

## Physical readiness and root hook review, 01:19 UTC

Final formatted source passes nine CPU tests (1.468 s) and all three CUDA
physical/lifecycle selectors on each card (4.902 s RTX /2.039 s GB, clean
exit0, no skips). The only CUDA-run warning is the inherited target-layout
deprecation. The G1-qualified device-drain teardown repair is included here;
it changes no per-step synchronization. Retained kinetic/notification/
publication/single-factor controls pass29 executed tests with four explicit
CUDA skips. Full pre-commit passes after formatting root hooks/tests.

The new factor seam changes only geometric H9 assembly. Source controls verify
the unchanged mask guards, R/positiveK augmentation, complete Cholesky arithmetic
and L output, plus unchanged original six-DOF ABI. Independent hook review
confirms drive-event and size-stream dependencies, original inverse producers,
free/MF and prescribed services, no retired primary-cache consumers, and one
complete final publication. SIMPLE_WORLD_ZERO is configured by the benchmark
but not active on Franka; no ZERO-path saving is claimed.

GPU tests cover four saved512-world current/held states, changed live frames/
COM/mass/gravity, current controls/body forces, original integration, actual
two-world loaded eight-sweep stepping, held L preservation, device mass-request
consumption, masked reset, inertial notification and two-bank graph0/6/0/6.
Maximum normalized errors agree on both GPUs: current H1.339639e-6, bias2.084246e-6,
held momentum defect1.730141e-7, actual refreshed L9 mass4.964072e-7,
public velocity7.119308e-7, held-force predictor8.721723e-8. These use independent
physical references. The secondary matched-original trajectory ceiling7e-4
was borrowed from the existing G1 control before native testing, not adjusted
after failure; observed joint/body velocity differences are8.268198e-5 and
5.717671e-5. This is not bit-identical stepping or a standalone convergence
certificate inferred from trajectory distance. Required PGS consumers and
their iteration allowances remain unchanged.

Whole16K timing follows these checks; no performance gain is established yet.
