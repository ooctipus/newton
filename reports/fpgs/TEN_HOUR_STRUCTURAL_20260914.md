# Complete-step structural work, September 14

User requested the next ten hours at approximately 08:58 UTC, extending the
previous window to approximately 18:58 UTC on September 14. Work remains
Newton collision/FPGS only, on the ooctipus fork with inherited handoff intact.
Lab remains `53ee6b44c2334341305dbdf385a3916c6b140799`; no parent pointer,
timestep, substep count or maximum iteration allowance changes.

## Objective and comparison contract

Seek at least 4x corrected MJWarp whole-physics throughput across Franka,
KukaAllegro, Allegro, ANYmal-D, G1 Rough and SO101 Keyboard. RTX PRO6000 is
primary; run paired RTX/GB300 batches with one owned process per device.
Report current-FPGS improvement separately from the MJWarp ratio, and whole
environment wall time separately from physics. Neither is full RL training.
Shared collision improvements must be enabled for both backends. Preserve
calibrated capacity and warning gates; do not drop contacts or inflate maxima.

Current best G1 candidate is Newton
`7df75c46bd468b64c17958190fc800cfeb62201e`, with metric tangents, parallel
limit prefix, level-scheduled sparse factor and corrected finite terrain
queries. Approximately 19.5 ms RTX / 25.1 ms GB is a discovery reference, not
a new six-task validation. Relative to the last corrected MJWarp reference
40.869609525 / 44.941774125 ms, the 4x ceilings are 10.217402381 /
11.235443531 ms. Refresh the denominator rather than treating it as immutable.

The corrected implicit spatial replacement at `5c90b0c6` measured
21.762360575 / 27.4409904 ms against same-run controls
19.507380075 / 25.078293 ms. It is default-off, unpromoted, and is not the
baseline for future claimed gains. Its lower-looking algebra did not compensate
for producer dependency depth and consumer costs.

## First decision block

Initial checkpoint: 09:28 UTC. No new native owner is funded merely by a high
percentage of low-contact worlds.

1. Recover existing six-task recipes and current best per-task source pins.
   Use existing benchmark drivers and adapters, not a new framework. Begin
   corrected paired triage, then repeat the promising integrated changes.
2. Measure actual contact-plus-active-limit cohorts using existing saved arrays
   where available. Missing per-world fields should be collected at the next
   existing benchmark boundary, outside timing. Count 1-row normal contacts
   separately from 3-row friction triplets; solved-zero impulses are not an
   early admission criterion. World fractions are not cost fractions.
3. Cost a small-contact complete-step owner that retains the useful held factor
   but retires selected-world intermediate force/predictor/row/decode producers.
   All required state publication, joint limits, fallback work, compaction and
   synchronization remain charged. A packet/layout change by itself is not
   the hypothesis. Record a credible multi-millisecond whole-step budget and
   physical regression before implementing.
4. Keep the complete collision-manifold boundary as a separate possibility.
   The existing pair-local census found winner replay would repeat MORE generic
   fallback queries than the original stream. Do not price replay using only
   total triangle percentage or retain the original reducer after replacement.

## Early readiness finding

The generic solver signature accepts `collide_done_event` and can defer waiting
until after prediction. This capability is NOT the fixed Lab workload's
schedule: `NewtonManager._step_solver` calls without that event and
`_simulate_physics_only` / `_simulate_full` call collision first. No Lab edit
is required to select a cohort from the already available contact geometry.

Read-only analysis of the existing metric node captures, preserving the
strict reader's process/correlation and physics-versus-auxiliary scope law,
found 12 physics roots/card. Collision/dynamics union overlap per environment
step was 0 ms RTX and 0.000821333 ms GB; the small GB boundary overlap is not
a meaningful millisecond scheduling benefit. Internal dynamics streams still
overlap and must remain charged when moving force/predictor ownership.

Evidence: `/tmp/fpgs-g1-metric-tangent-nodes-paired16k-20260914-01`, interpreted
with `/tmp/fpgs-g1-sparse-finite-compose-ATSRjnR2/audit_sparse_nodes.py` (SHA256
`74c5928beff9685e341135728d6213d9febbdd8fd821e2e91dc6ae020191e589`).

## Decision discipline

Each implementation receives a 90-minute checkpoint, exact producer-retirement
list, added-cost budget, numerical checks and a falsifiable integrated test.
Diagnose theory/timing discrepancies and allow a targeted correction for a
specific cause. Do not repeatedly tune an unsupported mapping or substitute
benchmark scaffolding for implementation. No silent extension beyond two
hours without new causal evidence and a revised measurable hypothesis.

Physical convergence, complementarity, momentum and stability matter, not
bit-identical trajectories. Include held/refresh, reset, empty/regrowing contact
and fallback transitions where relevant. Classify measured gains, diagnosed
losses and unvalidated ideas separately. Promotion requires repeated paired
whole timings and representative-task checks.

The existing session goal controller reports `paused`; the available goal API
cannot resume or replace that unfinished goal. Work continues in this active
session. This document does not assert an unattended background schedule has
been installed, nor that the 4x objective has been achieved.

## Fresh six-task baseline, approximately 09:44 UTC

One paired round per task, 200 warmup /40 environment wall /40 physics steps.
16,384 worlds except Keyboard4,096. Fixed corrected Lab imports and MJWarp
line-search correction. Every capture passed the original two sticky capacity
boundaries. These are refreshed current baselines, NOT small-step gains.

| Task | RTX FPGS /MJWarp ms | Physics ratio | GB300 FPGS /MJWarp ms | Physics ratio |
| --- | ---: | ---: | ---: | ---: |
| Franka | 5.4844 /10.6914 | 1.95x | 4.9773 /10.3265 | 2.07x |
| KukaAllegro | 12.4332 /27.5043 | 2.21x | 11.8373 /25.9162 | 2.19x |
| Allegro | 17.3287 /63.4336 | 3.66x | 17.8262 /82.0507 | 4.60x |
| ANYmal-D | 9.3885 /36.6712 | 3.91x | 9.5033 /42.2882 | 4.45x |
| G1 Rough | 19.4711 /40.8763 | 2.10x | 25.0667 /44.9423 | 1.79x |
| SO101 Keyboard | 7.6387 /28.2247 | 3.69x | 6.7055 /28.5058 | 4.25x |

Newton FPGS/MJ `7df75c46` except accepted Allegro FPGS
`fba9fead70d17728f842954d66cf1f05f4617d40` (coherent convex reject-only),
MJ still7df. G1 shares all four accepted terrain features between backends.
Allegro uses its established compact adapter; MJWarp native contacts are not
given Newton-only capacity overrides.

Franka additionally qualifies raw32,768 /broad7,680 instead of its inherited
4,000,000 raw default; Kuka qualifies raw311,296 /broad442,368. Their existing
dense192/MF64/prop192 owners remain unchanged. No algorithmic gain is attributed
to these capacity trials. No drops observed; longer loaded testing is still
required before calling a smaller capacity universally safe.

Whole-environment wall time is separate. In particular Franka FPGS is
63.95/62.92 ms versus MJ27.53/29.21 ms (RTX/GB), despite faster physics.
Keyboard wall is30.46/27.93 versus31.34/32.03 ms. Do not claim RL training
speedups or a resolution of the parked termination/reset discrepancy.

Exact artifact directories:

- `/tmp/fpgs-tenhour-franka-paired16k-20260914-01`
- `/tmp/fpgs-tenhour-kuka-paired16k-20260914-01`
- `/tmp/fpgs-tenhour-allegro-paired16k-20260914-02`
- `/tmp/fpgs-tenhour-anymald-paired16k-20260914-02`
- `/tmp/fpgs-tenhour-g1-shared-paired16k-20260914-01`
- `/tmp/fpgs-tenhour-keyboard-paired4k-20260914-01`

Preserve failed startup attempts: ANYmal01 used the wrong unadapted import
entry and was cancelled; Allegro01 rejected the old Lab path before launching
GPU work. Neither contributes a number. Corrected runs02 use fixed Lab
`/home/octi/Projects/IsaacLab.wt/contact-reset-20260913`.

The active structural candidate is `ooctipus/fpgs-g1-small-step-20260914`,
based on7df. Candidate card is `reports/fpgs/G1_SMALL_STEP_20260914.md` in
that worktree. It targets current <=8-contact/<=32-row worlds, retaining all
active limits and required publication, while retiring selected force,
prediction and global contact-response producers. The following completed
checkpoint supersedes its initial readiness status.

## First structural candidate: diagnosed loss, approximately10:32 UTC

Measured runtime `d9beb746fb416c3b1db0f591a97c6d53d102b269` passes paired
native physical controls and actual Solver.step lifecycle transitions, but
loses whole physics: RTX19.471282->21.260241ms (0.915854x),
GB25.111638->27.752758ms (0.904834x). It remains default-off/unpromoted;
current good baseline is still7df. All original capacity/source/actual-owner
checks pass. Whole artifact `/tmp/fpgs-g1-small-step-paired16k-20260914-01`.

The profile and one low-perturbation in-place phase diagnostic expose the
miss: the approximately17% fallback worlds still retain54% of original RTX
GS time. New early layout offsets part of row-producer savings. The complete
new owner costs5.578584/6.199157ms RTX/GB; per-world clocks put approximately
three quarters of its work in row formation and GS, not force calculation.
Geometry/W already serve all three contact directions together. There is no
evidence of an accidentally retained full selected producer or pathological
cohort to repair. Recovering the loss AND saving2ms now needs roughly3.79ms
additional whole-step reduction. No such correction is funded; close this
candidate before its90-minute checkpoint, not after a mapping sweep.

The detailed candidate card retains physical/lifecycle results, original
failed auxiliary-node analyzer manifests, checked independent interval reads
and the explicit instrumented-kernel overhead (1.92%RTX/2.31%GB).
Neither per-world cycle fractions nor diagnostic nodes are throughput claims.

## Next distinct boundary: stronger current terrain separation

A read-only saved-geometry study finds46.6613%/46.6277% of the remaining
RTX/GB triangle queries geometrically separable using projected box axes,
upward box axes and the actual triangle top plane. This is distinct from
earlier flat-run merging/replay proposals: no terrain approximation, bottom
cull, cached-height assumption or manifold coalescing is proposed.

Counts alone are not a timing decision. Separated queries may take the
expensive closest-distance branch; surviving penetrating queries may be
cheaper. Fund ONE time-weighted native diagnostic, charging the separator and
retaining the original finite-query/generic-fallback/reducer writer work.
Every omitted original callback must be checked outside timing against the
current contact shell AND terrain-size-scaled reducer spatial threshold.
The old reducer buffers far witnesses, so geometric separation alone is not
a sufficient implementation argument. All original capacities remain fixed.

Study `/tmp/fpgs-g1-geometric-separation-UYfUwYR8/RESULTS.md`, final result
SHA256`3e15be7c639d8da489b8498189ea6d3ff22c439d794b90bb327c350c45d2a8e4`.
No new terrain speedup or native physical acceptance is claimed yet.

## Native terrain query decision, approximately11:09 UTC

The first native mask-only test passed the omitted-callback safety audit but
saved no time: retaining holes in the old stream retained expensive mixed
warps and the original launch footprint. Original plus screening changed the
four-collision query boundary by -0.152064ms RTX /-0.132992ms GB (a loss).
Preserve `/tmp/fpgs-g1-geometric-query-native-paired-20260914-PV7oXt`.

One causal correction used the compact surviving prefix that the proposed
original midphase append would naturally produce. It used the ORIGINAL
unmasked finite/generic query kernels and original reducer writer, with live
launch AND stride196,608 (fixed Lab's existing thread multiplier). The stable
diagnostic prefix was built outside timing; production must obtain compactness
from its original append, not add an uncharged compaction pass. Screening was
charged, while reduction/export/clear savings were not credited.

| Four-collision query boundary | RTX ms | GB300 ms |
| --- | ---: | ---: |
| Original | 3.224064 | 11.502592 |
| Compact survivors plus screening | 2.247424 | 7.336320 |
| Saving | 0.976640 | 4.166272 |

These are saved-input native query timings, NOT live whole-physics gains.
Both cards pass full callback accounting:608,272/609,402 original triangles,
269,395/269,222 safely omitted, and161,820/161,188 omitted original callbacks.
No omitted callback crossed max(current shell, scaled reducer beta)+pad;
minimum remaining slack1.357mm/1.255mm. Preserved packed artifact:
`/tmp/fpgs-g1-geometric-packed-native-paired-20260914-YoLrRO`.
Frozen diagnostic SHA256
`68239bb36a9bb96499732ab99cb013574b82e6dbed42600fc9edd0e2514064cb`.

Funded production integration in `ooctipus/fpgs-g1-geometric-cull-20260914`,
base7df, with11:45 checkpoint. It must retain uncertain/predictive/custom
writer geometry and current scaled reducer semantics, use no new queue or
launch, and pass existing physical plus paired whole-cost checks. Because
collision is shared, a successful change must also reach the MJWarp control.

In parallel, `ooctipus/fpgs-g1-analytic-manifold-20260914` at
`ef7e22587e7fca00f601e32e6803010510487e32` supplies certified separated
supporting-face witnesses directly to the ORIGINAL manifold tail, retiring
discarded finite clipping and MPR/GJK for that class. CPU geometry/lifecycle
and offline compilation pass; larger allocation184 registers/688B stack in
the reducer overload is a charged risk. Saved96 admission64.15%/77.78% of
old fallback entries is not a population timing claim. Root independently
reviewed native/test source; paired physical checks are next. No promotion
or whole-step gain is claimed for this implementation yet.

## Production terrain result, approximately11:39 UTC

Frozen `7790c35b348dd1885a90f6c2def9b336cde80399` integrates the conservative
separator into the original triangle append. Unsupported/current-uncertain
pairs retain the original cell midphase; no new queue, launch or compaction
pass. Default-off `NEWTON_HEIGHTFIELD_GEOMETRIC_CULL=1` admits only the
qualified nonpredictive stock-writer path. The complete runtime and geometry
proof are in its `G1_GEOMETRIC_CULL_20260914.md` card.

All four original/new physical selectors pass on both GPUs, including
actual emitted-stream subset and every original reducer-buffer callback,
current geometry/guard changes, empty/regrowing graph capture, repeated
same-state rebound and loaded flat/seam response. No skips or capacity
failures. Physical artifact
`/tmp/fpgs-g1-geometric-cull-physical-paired-20260914-01`, manifest SHA256
`1810c0d55e34f25f64dc4dc04461dd31f0c877d7dfcd8baa1d7465ffb93269ce`.

| One whole-physics discovery | Baseline7df ms | Cull7790 ms | Incremental ratio |
| --- | ---: | ---: | ---: |
| RTX PRO6000 |19.449285075|18.189770725|1.069243x|
| GB300 |25.086320725|20.896425375|1.200508x|

Whole wall32.730027->31.666569ms RTX,39.539466->34.807193ms GB. Original
source, actual-feature, sticky capacity and final-idle guards all pass.
Artifact `/tmp/fpgs-g1-geometric-cull-paired16k-20260914-01`.
Two alternating paired repeat rounds are running; the MJWarp denominator
must also enable the shared cull before reporting a new backend ratio.
This is a measured structural gain, not4x or cross-task promotion.

## Analytic-manifold whole-cost loss

The separate certified-witness/original-manifold runtime is unchanged at
SHA256`c50c889e2b93643ab63a59ef3d1a2e5ce65e15450e0f3bb13a54c715f27fdbaa`.
Final test/report-only tip is `122eb87e63123698ab6aaa33e19727597cbbec36`.
The first physical attempt exposed an invalid new oracle: the original CUDA
GJK normal differed from the authored flat plane by3.97364e-6, exceeding
the new bilateral2e-6 comparison. The candidate was exact. Corrected the
test to compare the candidate to the authored plane with the SAME2e-6
tolerance, retained original point/distance/material/fallback and physical
gates, and preserved the failed attempt. No physics or tolerance relaxation.
All four selectors/card pass in physical attempt02, manifest SHA256
`b8fe52080784323eac071ee054ec2e37563e34a513ca4ca0c6bd82e6a2490ed3`.

Whole physics loses:19.458857175->20.587984375ms RTX (0.945156x),
25.1015952->30.004724475ms GB (0.836588x). Original caps/source/activation
checks pass. Contacts/rows changed approximately0.4--1.3%, insufficient by
itself to explain the complete loss. Artifact
`/tmp/fpgs-g1-analytic-manifold-paired16k-20260914-02`.
Keep this candidate default-off and OUT of the good baseline. One original
node capture is next to diagnose added owner cost; larger register allocation
is a hypothesis, not a demonstrated occupancy cause. Do not start a sweep.

Preserve whole attempt01: its untimed observer expected the short old kernel
key instead of actual `create_query_kernel__locals__heightfield_finite_contacts`.
Both baseline children failed before measured physics, so no timing is valid.
The corrected derivative changes only that literal and retains the old
parent/metric/capacity/source laws; successful02 uses the independently
checked wrapper `/tmp/fpgs-g1-collision-redesign-checked-rn5lvjV1`.

## Next bounded owner: exact convex adjacency

Funded at11:40, first checkpoint12:25 and hard90-minute checkpoint13:10 UTC.
Branch `ooctipus/fpgs-allegro-exact-hill-20260914` starts at accepted Allegro
rejection-only `fba9fead70d17728f842954d66cf1f05f4617d40`.
Prior previous-winner hill climbing reduced certified scalar support work
2.305x/2.306x but had NO CUDA timing; it is unmeasured, not a diagnosed
native loss. Existing sampled directions contain zero complete original
warps, so no SIMT throughput claim follows from that scalar work reduction.

The bounded implementation must replace the coherent classifier's full
support/feature/magnitude contract AND cold MPR, cold GJK and manifold support.
Their source-matched RTX owner sum3.564852ms leaves at most2.094574ms for
the replacement to close today's1.470278ms Allegro gap to4x. This is high
risk, but measurable. Charge the actual2,523,136-key persistent pair domain:
two endpoint words, four packed eight-bit family seeds each, cost19.25MiB.
By-value provider scalar mutation is not persistent state. Preserve current
source/epoch ownership, immutable valid cooked convex adjacency, primitive
specializations and original full-scan fallback for uncertain certificates.
Use existing physical and integrated16K paired benchmark owners, not a new
census/framework. Whole cost is authoritative; one evidence-backed causal
correction may follow a miss.
