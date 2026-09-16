# Ten-hour awake-component structural work

User-requested window: 2026-09-15 23:44 UTC to 2026-09-16 09:44 UTC.
Start: `fe24f9a64c0f0dbe91f4470ae91ac795fd8c2cb1` (default-off sleeping
foundation). Accepted performance reference remains `ca0d427a`, not this
slower prototype. Fork `ooctipus/newton`; original worktrees are unchanged.

## Contract and checkpoints

Newton collision/FPGS only; preserve Lab, physical budgets, capacities and
handoff ancestry. RTX primary, paired GB300, one root-owned job per GPU.
No sudo or hardware-counter bypass. Use existing tests/captures/bench drivers.
Do not treat a high asleep fraction, component sum or unsupported fallback as
an integrated gain. No dependency-pointer changes or production promotion
without numerical/physical qualification and repeated paired whole timing.

First checkpoint 01:14 UTC: integrated owner test and task timing, or a
specific diagnosed mismatch. One targeted correction per identified cause;
no silent extension past two hours. Later checkpoints pursue the largest
remaining owned work, cross-task effects and final qualification. Record
measured gains, diagnosed losses and unvalidated proposals separately.

## Candidate 1: complete independent-component owner

Replace, rather than supplement, the current independent prismatic pipeline:
sleep begin/finish scans, actuator preparation, direct force conversion,
scalar prediction/inertia, integration and current state publication. Share
one model-derived component representation and current awake classification;
keep unsupported/coupled components on intact existing producers. Preserve
current contacts, original GS and all input/output/reset contracts.

Cost evidence: the previous RTX node trace spent 1.144 ms in the original
force/integration/publication bundle, 0.331 ms in the additional scalar
producer bundle, and the prototype added 0.541 ms of sleep-controller work.
These are overlapping diagnostic sums, not additive critical-path savings.
Replacing duplicated passes and intermediates together plausibly exposes
0.7–1.0 ms of the approximately 7.0 ms whole physics step on RTX. This is a
first structural milestone, not a claim that this limited owner yields 2–4x.
GB has a smaller original owner cost and must be measured independently.

Falsification: if a complete replacement cannot save at least roughly 10%
whole physics or produces disproportionate fallback cost, diagnose the exact
remaining duplication/mapping problem once; do not pursue mask-level polish.
Numerical tests cover actual dynamics/targets/forces, loaded contacts and
limits, authored states, mixed reset/model changes, A/B and in-place states,
public poses/velocities and fallback. Retain timestep/substeps/iterations.

## Parallel architectural study

Reference installed MJWarp sleep/island and forward/collision code for a
general contact-island lifecycle: static ground does not couple independent
responders; authored inputs wake before skipped producers; new contacts wake
the entire affected old island and require complete incremental processing.
Do not filter away the contacts needed to wake a component, omit contact
forces, or copy buffer truncation. Quantify the still-removable collision,
contact-production and solve work before implementing the next owner.

Reuse latest representative-task references to select subsequent owners.
Sleeping alone is not expected to accelerate an always-active robot. The
longer-term target requires eliminating redundant representation and contact
work across those active paths as well.

## First integrated owner (00:15 UTC)

The complete scalar owner replaces the old eligible-component drive, force,
inverse-mass, predictor, integration and publication producers. Compact
fallback index sets preserve ordinary articulated execution. Admission is
topological and requires complete scalar-articulation coverage; unsupported
configurations retain the original path. No task-name or 108-DOF gate.

Four new unittest groups passed on RTX and GB300, including actual admission
at 37 and 108 branches, non-unit rotated axes, displaced COMs, applied forces,
effort limits, active joint limits, held articulated mass, real contacts,
tiny-dt drive wake, captured resets and model changes. These are native
physical controls, not a performance result. CPU-only discovery passed the
API test and explicitly skipped the three native groups. The API regression
failed before the new switch existed.

While tracing ownership, a separate defect was found in the default-off
sleep prototype: authored wake indexed the one-element mass-refresh request
by articulation. A guarded backing-array regression failed on both GPUs
(guard overwritten from -17 to 1) and passed on both after changing the
atomic target to slot zero. Accepted baseline `ca0d427a` is unaffected.
Earlier sleeping tests had not exposed this out-of-bounds write; those tests
alone were therefore insufficient qualification of that prototype.

The existing sleeping, compact-contact and prismatic-publication suites also
passed (22 tests on each GPU). An initial command named a nonexistent
`test_feather_pgs_prismatic_linear_state` module and failed collection for
that selector; the corrected three-module command passed. Linear-state
controls live in the prismatic-publication module.

Independent review found a deliberate scalar-response difference: this owner
uses the exact current scalar inverse, including current `ke*dt^2+kd*dt`,
rather than retaining an old coefficient between matrix refreshes. The
scalar physical mass is coordinate-independent. For ordinary fixed-dt runs
the response is unchanged; sub-threshold timestep changes can differ from
the old held approximation. Articulated fallback mass cadence is preserved.
The tiny-changing-dt case is checked against its analytical response, not
required to reproduce the old stale coefficient. No timestep or iteration
budget in the benchmark is changed.

The completed five-group native owner suite passed on both GPUs, including
that independent tiny-dt analytical check and unchanged held serial factors.
`uvx pre-commit run -a` passed with all new source/test files tracked before
the first owner commit. First paired graph timing uses the included
`REPRO_AWAKE_PIPELINE_20260916.sh`, unchanged 4K task/capacities and the
retained `ca0d427a` reference; it is a discovery round, not promotion.
