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

At the start of this window, the best G1 candidate was Newton
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

## Repeated shared-collision result, 12:22 UTC

The geometric cull holds over three FPGS A/B rounds. Two alternating repeat
medians are RTX19.499192->18.176309ms (1.072781x),
GB25.119965->20.957624ms (1.198608x), with all original guards passing.
The detailed repeated physical/cost card is pushed on
`ooctipus/fpgs-g1-geometric-cull-20260914`, report tip `2f84ae50`;
the measured runtime remains clean `7790c35b`.

The fair shared comparison discovered that earlier G1 MJWarp omitted
THREADS_X and inherited Lab's default1 while FPGS used4. New paired runs
explicitly set4 for BOTH backends plus all five shared collision improvements.
This changes MJWarp's actual launch AND stride49152->196608. Denominator
improvement cannot be attributed solely to the cull or called an FPGS gain.
MJWarp's actual shared Newton contact path remains verified.

| Latest repeated G1 physics | FPGS ms | Corrected MJWarp ms | Ratio |
| --- | ---: | ---: | ---: |
| RTX PRO6000 |18.175791|37.818479|2.080706x|
| GB300 |20.927221|38.226997|1.826664x|

Artifacts are `/tmp/fpgs-g1-geometric-cull-paired16k-20260914-repeat02`
and `/tmp/fpgs-g1-shared-geometric-paired16k-20260914-repeat02`.
These replace the earlier G1 denominator, not the other five task values.

## Analytic-manifold causal closure

Matched200warm/40wall/3profile nodes localize the loss to the fused collision
producer, not GS or publication. Finite query grows2.207010->3.700568ms RTX
and7.408693->14.509192ms GB. Its generic fallback genuinely shrinks
.937473->.475883ms RTX /3.364981->1.146581ms GB, but the larger producer
erases that saving. Its184-register allocation is observed; occupancy or
memory-bandwidth causation is NOT proven without hardware counters.

Artifact `/tmp/fpgs-g1-analytic-manifold-nodes-paired16k-20260914-02`, strict
matched audit SHA256
`be73b6a3ce2f22e1ab49085e3594e56e0ddcfc32a3b09e2b32d42cb66127f305`.
The original auxiliary-graph analyzer failure is retained; independent strict
attribution proves12physics+3auxiliary roots/card, all device-memory nodes and
zero unproven owners. All original capacity/source/idle checks pass.
The proposed split-witness correction lacks a new >=2ms RTX removable-work
budget because the entire old fallback is under1ms. Close rather than tune.
Report-only tip `39a39c54` is pushed; runtime remains unchanged/default-off.

## Two active large-cost boundaries, 12:29 UTC

Allegro hill runtime `99a23652bde24ca61946be90ccc7841466a89557` passes CPU,
actual four-owner SM120/SM100 compilation and four physical selectors on each
GPU. Its first integrated result is a loss: RTX17.330418->19.351608ms and
GB17.742004->20.072446ms. All original checks pass and actual four-owner
dispatch plus19.25MiB hint/15164B descriptor costs are present. Keep it OUT
of the baseline. Matched node attribution is running before one causal
correction decision; do not confuse scalar work reduction with SIMT speed.

G1 `ooctipus/fpgs-g1-publication-compact-20260914` replaces complete
publication and current-force/composite production with two owners. It retains
the original W434 factor and all consumers, uses one current combined external
plus intrinsic wrench reduction, and compact ten-moment composites. It does
not revive the closed implicit-spatial consumer path or cache future forces.
Old complete boundary exclusive time4.125572ms RTX gives a2.125572ms replacement
allowance for a2ms whole saving. CPU/AOT review is progressing; no timing gain
claimed. Checkpoint13:00, hard13:45. Card is
`reports/fpgs/G1_PUBLICATION_COMPACT_20260914.md` in that separate worktree.

## Franka large environment-step repair: first measured result

The corrected Lab root-pose notification exposes complete row-packet topology
validation inside resets. Preserve that correctness fix and every current
structural check; replace only the per-world Python predicate with array
operations. No solver/kernel/capacity/notification-law change. Original
regression fails with28735 Python line events at4096worlds versus the351bound;
the replacement and all18 current prefix/notification CPU tests pass with no
skips. Independent source review and full pre-commit pass.

Frozen runtime `99c796ca` in `ooctipus/fpgs-notification-validation-20260914`
gives the following first complete paired result:

| Franka,16K worlds | Old wall ms | New wall ms | Wall ratio | Old/new physics ms |
| --- | ---: | ---: | ---: | ---: |
| RTX PRO6000 |62.723041|29.532334|2.123877x|5.495731/5.478576|
| GB300 |63.690820|28.592554|2.227532x|4.997234/5.031553|

This is a substantial environment-throughput improvement, NOT a physics or
RL-training speedup. It does not meet the4x physics objective. All original
capacity/source/final-idle checks pass and physics budgets are unchanged.
Artifact `/tmp/fpgs-franka-notification-validation-paired16k-20260914-01`,
manifest SHA256
`ed3a7098d2b8c7207ce0a38a80f14e0de910ccd183e96fd57b2afd35215cf743`.
Two alternating paired repeats are queued after the current Allegro node run.

## Mid-window checkpoint, approximately13:20 UTC

Franka's two alternating repeats preserve the large host-side result:
RTX wall median62.881182->28.905997ms (2.175368x),
GB63.050704->28.939305ms (2.178722x). Physics remains effectively unchanged
at5.527632->5.483634ms RTX /5.036061->5.012089ms GB. All eight captures'
original capacity/source/final-idle checks pass, with unchanged physics budgets.
Runtime99c796ca and report-only04b2328e are pushed on the notification branch.
Artifact `/tmp/fpgs-franka-notification-validation-paired16k-20260914-repeat02`,
manifest01994506d028ffeff980c8738b8d89daefbfe86dd147df5eda94ffd956fedbb0.
This is not a4x physics result or evidence of faster RL training.

### Compact G1: numerical pass, diagnosed complete-step miss

Frozen compact runtime `dd05a1e53b2c795db4e812931eeb3e38abfc7cf2` passes
all three actual CUDA physical/lifecycle selectors on each GPU, including
current forces, compact mass, original integration, held factor, reset and
empty/regrowing graph transitions. Whole physics is18.157995->17.883307ms
RTX (1.015360x),20.875017->23.143626ms GB (0.901977x). Do not promote.
Whole artifact `/tmp/fpgs-g1-compact-paired16k-20260914-01`.

Strict matched attribution includes12physics+3auxiliary roots and all device
memory/correlations, preserving the generic analyzer's original auxiliary-root
rejections. New publication increases2.439264->3.129909ms RTX and
1.621152->3.781076ms GB. The retained factor is faster, not the cause of
the loss. Complete touched producers plus retained factor exclusive time is
6.674709->6.590537ms RTX /4.536299->6.987038ms GB. Original tau/factor and
tau/composite overlap is also partly lost; summed savings are not additive.
Audit `/tmp/fpgs-g1-compact-checked-oyNUm1/NODE_AUDIT_COMPLETE.json`.

The source exposes a specific mapping mismatch: new publication uses one
world per64threads, whereas the old template owner packs two worlds into32.
The eleven tree levels have at most seven bodies each. Fund one correction
using four independent eight-lane worlds perwarp, charging its larger shared
storage and preserving all44 current public bodies and compact moments.
No factor retuning or mapping sweep. Source/CPU/AOT checkpoint14:00,
whole-cost decision14:25. This correction is unmeasured.

### Allegro: valid causal counts, one ownership correction

Matched original four-owner collision time grows3.547650->5.613530ms RTX,
4.531648->6.819508ms GB. MPR and GJK each contribute approximately1ms of
RTX loss; unrelated owners remain near-flat. Production remains99a23652,
default-off and unpromoted.

The first diagnostic histogram is INVALID: it allocated for the constructor's
thread count before Lab applied its existing4x multiplier. Preserve artifact01
but withdraw every count from it. Corrected probe02 asserts the final worker
domain before enabling its counters and passes original source/capacity/final
idle checks. The valid RTX sample contains267.385M MPR callbacks with98.373%
terminal certificates and278.110M GJK callbacks with92.547%. Neighbor loads
remain17.674/15.909 percallback across3.213/2.844 dependent hill rounds.
Instrumentation perturbs owner times; these are counts, NOT performance.
Evidence `/tmp/fpgs-allegro-hill-callback-counts-fixed-UXhOGD/RESULT.md`;
valid artifact `/tmp/fpgs-allegro-hill-callback-counts-paired16k-20260914-02`.

Fund one distinct correction: keep support hints in per-query shared cells,
loading/flushing persistent hints once perquery instead of publishing every
callback to global memory. Keep original geometry, terminal proof and full
fallback. Charge1024B per cold/manifold block and all query lifetime seams.
This removes intermediate hint traffic but retains dependent CSR traversal;
the required approximately3.49ms candidate reduction is NOT established by
callback counts. Source/CPU/AOT checkpoint14:00, whole decision14:25.

### Recover and qualify the actual Kuka win

The old live kinetic runtime `ad42fea95fd44cd5ec72afa1be1cce570b45109a`
was a measured RTX gain with incomplete live-contact qualification, NOT a
closed whole-cost loss. Do not confuse it with its failed first-hit successor.
One fresh current-capacity comparison against7df uses the same six established
Kuka flags, with only KINETIC_WORLD added to the candidate. Raw311296,
broad442368, dense192/MF64/prop192 and original eight sweeps/two substeps
are unchanged between arms.

RTX physics12.357853->10.909528ms (1.132758x), GB11.934729->11.218563ms
(1.063838x). RTX whole wall36.390353->35.781557ms, but GB worsens
34.022117->36.469050ms. This is one discovery, not promotion or a fresh
MJWarp ratio. Original checks pass; artifact
`/tmp/fpgs-kuka-kinetic-current-capacity-paired16k-20260914-01`.
The next check samples real post-warm refresh/held Solver.step calls and
compares current contact/control inputs to the original eight-sweep numerical
law before original finish mutates current state. No copied saved population
substitutes for that live qualification. Missing cohorts remain unqualified.

## September14 21:20: four-times minimum, current implementation checkpoint

The user reaffirmed **at least4x corrected MJWarp physics on RTX for every
representative task**, not4x on average or a cap on further gains. The target
remains unmet. Original Lab53ee and accepted Newtonbaf0c57a remain unchanged.
No newly failed candidate below replaces an accepted comparison baseline.

### Kuka: physical qualification recovered; accepted-source composition wins

The4K live current/held readsets passed the unchanged physical allowances after
correcting a test-only signed-limit metric: a smaller positive safe velocity
is not increased limit violation. The old failed diagnostic remains visible in
`/tmp/fpgs-kuka-limit-violation-20260914-dnHZ5Nwl/FINDINGS.md`. Loaded actual
CUDA coupled/type-4 controls also pass on both GPUs; these are not assertions
that the sampled live4K worlds contained that cohort.

Clean6c2297ca composes ad42's byte-identical native owners onto acceptedbaf,
retaining its existing features. CPU68 executed checks and249 frozen native/
descriptor records pass;19 CUDA-only cases explicitly skipped in the CPU batch.
Original paired16K200/40/40 discovery, identical six existing flags plus only
candidate KUKA_KINETIC_WORLD=1:

|GPU|baf physics ms|composed physics ms|incremental ratio|wall old/new ms|
|---|---:|---:|---:|---:|
|RTX PRO6000|12.379235|10.963438|1.129138x|36.788461 /35.906270|
|GB300|11.983935|11.270214|1.063328x|35.482980 /38.251759|

All original source, capacity and idle checks pass. Artifact:
`/tmp/fpgs-kuka-kinetic-integrated-paired16k-20260914-01`.
Three alternating RTX rounds completed21:24: median physics12.506653->
11.119767ms (1.124723x), wall36.993766->33.680188ms (1.098384x).
Artifact `/tmp/fpgs-kuka-kinetic-integrated-repeat-rtx16k-20260914-01`;
all original source/capacity/idle guards pass. Deferred GB-only repeats also
pass: median physics12.033355->11.339816ms (1.061160x), wall36.255405->
35.576291ms (1.019089x). Artifact
`/tmp/fpgs-kuka-kinetic-integrated-repeat-gb16k-20260914-01`.
These repeats are not simultaneous.
The paired repeat stopped before launching because GB300 became externally
occupied; that failed artifact remains. A source-pinned scheduling-only
derivatives admit explicit RTX GPU0 or GB GPU1 alone; no physics/timing checks change.

### Failed mappings: retain causes, do not polish small remnants

- G1 lazy compliance4c2a22cf: whole RTX18.149754->21.314393ms and
  GB20.915280->23.350513ms. Exact node ownership isolates GS3.914287->
  7.159739ms RTX. Cache column construction/residual updates add real work;
  shared storage increases1116->8316B and theoretical resident blocks24->10
  on RTX. This is not achieved-occupancy evidence. No cache/block grid follows.
  Full result and cause are retained in its report-only tipa476bbb5.
- Franka paired late publication5bc2d73d: whole RTX5.480331->5.211907ms
  (1.051502x), GB4.992435->4.875338ms; environment wall time worsens.
  Below the funded whole-task milestone. Preserved, not promoted or retuned;
  report-only tip7a772365 records the complete boundary and failed predecessor.
- Allegro direct kinetic rows55a0259e: four CUDA selectors/card and all
  original whole checks pass, but whole RTX16.499407->18.444526ms and
  GB16.421763->19.581437ms versus direction-cell92507273 with cells on
  BOTH arms. Original12/24 sweeps and calibrated capacities are unchanged.
  Artifact `/tmp/fpgs-allegro-kinetic-rows-paired16k-20260914-01`.
  Complete strict node02 proves192 graph nodes removed, but new contact
  publication costs3.422595ms RTX and fallback scans0.502646ms. The solve
  FAMILY appears0.513706ms faster RTX and0.928817ms slower GB, but this is
  partly intermittent serial fallback: RTX baseline has two881.984/856.928us
  calls while candidate has none; GB baseline has one814.656us call versus
  candidate two825.632/833.440us calls. The actual four parallel tiers cost
  4.906914->4.965593ms RTX and5.446110->6.096678ms GB. Do not credit the
  apparent RTX family gain to faster parallel whitening. Two untimed row
  snapshots do not prove an entire capture had no fallback. Full boundary grows
  9.034745->10.739260ms RTX. Failed node01 records external GB contention;
  uncontended02 passes all source/process/correlation/capacity checks.

### One funded Allegro correction, not another launch sweep

New worktree `newton-fpgs-allegro-keyed-kinetic-rows-20260914` preserves55a0259e.
Replace global contact Z/metadata publication with typed raw-row keys and
current geometry-to-register ingestion inside the actual parallel tier. Keep
the efficient full22-coordinate float4 sweep and original24-iteration law.
Use complete world-centric original-J fallback for >128/MF worlds; no contact
is discarded. The complete row/solve budget is **<=7.034745ms RTX**, including
every key, prefix, fallback, map and solve cost. That is not yet demonstrated.
Early native/resource and existing physical tests precede whole timing. No
bit-identical trajectory requirement, new benchmark framework or Lab edit.

At21:37 keyed candidate7fca2fdf is clean, CPU8 and both cards' unchanged four
CUDA selectors pass. Actual current/held momentum defects3.39e-8--3.90e-8;
full production lifecycle includes current geometry, held factor, crowded
fallback, reset and seeded graph grow/shrink. No global contact coefficient
buffer remains. Final source-matched AOT03 reports no stack/spills. Source/
admission review confirms the preceding response and diagonal stages skip
selected worlds; metadata is created in the actual row lane before solve.
Original guarded whole16K200/40/40 comparison against925 cells is starting,
with cells enabled on both arms. There is no performance result yet.
First launch01 stopped before children because the required local launch-card
file was missing; its failed manifest is retained. Corrected02 includes it.

## September14 22:10: Allegro crosses the RTX four-times threshold

Corrected keyed runtime7fca2fdf whole discovery against925 cells on both arms:
RTX16.340675->15.201010ms (1.074973x); GB16.517619->16.719848ms
(0.987905x). The planned2ms saving was missed. Actual strict node attribution
shows row/setup retirement offset partly by slower parallel tiers, not a
hidden removal of required contact work. RTX parallel4.889749->5.852930ms;
GB5.508978->7.396244ms. GB's one821.568us fallback spike adds only0.270368ms,
not most of the solve growth. Actual executed sweeps are not captured, so
geometry cost versus rounding-sensitive early stopping remains unresolved.

Fresh three-round same-source corrected MJWarp comparison completes at21:55:

|GPU|FPGS physics ms|MJ physics ms|median physics ratio|FPGS/MJ wall ms|
|---|---:|---:|---:|---:|
|RTX PRO6000|15.103509|63.536818|4.206759x|23.131495 /71.449856|
|GB300|16.664395|82.023917|4.922106x|24.920978 /91.547777|

All three same-round RTX ratios exceed4x. All12 children and original source,
capacity, idle, finite-state and line-search checks pass. GB MJ's third round
is94.850148ms, an explicit retained outlier; the median above is82.023917ms.
These physics ratios are not environment throughput or full RL training.
Allegro's original MJ recipe uses native MuJoCo contacts: no Newton collision
pipeline exists there, so direction cells are genuinely inapplicable. Do not
change that contact mode to manufacture a common-pipeline comparison.
Artifact `/tmp/fpgs-allegro-keyed-corrected-backends-paired16k-20260914-01`;
full qualification/reproduction report commit a08f7f19 is pushed to
`ooctipus/fpgs-allegro-keyed-kinetic-rows-20260914` on the user's fork.

Kuka's report-only tip298c038e and prior ledger531a9294 are also pushed and
remote-verified. A new source composition1921e214 retains both qualified
ancestors, original31cf handoff ancestry and all existing accepted features.
Its249 Kuka records and eight Allegro native hashes/ordered ABIs are unchanged;
91 executed CPU checks pass,28 actual-CUDA cases explicitly skipped in that
CPU batch. Existing GPU integration controls are running, not a fresh timing
claim for the composed tree. Original Lab and all prior worktrees remain intact.

### Next bounded work: ANYmal reuse of already-computed exact Gram products

Current ANY18/WR/EX1 already forms full admitted pairwise products for its
step-size row sums, then discards them. New default-off prototype retains
those coefficients in fixed-index registers and reuses the original dense
four-accumulator product inside the same24-step projected/Nesterov law. It
retires the repeated MF transpose/reduction and its synchronization, without
global A/Z publication or reactivating the legacy producers. The historical
shared-A/global-Z path already lost to MF and is not being rebranded as new.
Original current/held fixtures and physical quality gates are reused.

First offline compile has no stack/spills but registers rise from86/80 to
172/204 on RTX32/48 tiers; GB80/80->170/202. Shared storage is unchanged5408B
at32 and7492->7276B at48. This is a major resource risk, NOT a measured loss.
Inspect one code-generation/lifetime cause before any correction; no launch
grid or arbitrary register cap. Whole saving target is at least1ms on ANY,
not merely the0.22ms required to round its current3.91x over4x.

Read-only boundaries closed without another prototype: Keyboard's historical
source-equivalent full contact/solve boundary2.764919ms already had losing
private and packed replacements; lazy-publication plus following FK has only
0.491777ms zero-cost ceiling. G1 corresponding publication/FK ceiling0.931126ms
already excludes its cache-hit next FK. Do not count a nonexistent second FK
traversal as removable work. A separate check of16 pinned selected G1 current/
held operator cases finds no exact duplicate contact triples or coplanar group
with more than two distinct contacts, so it does not fund exact redundant
contact-hull elimination. These selected cases are not a population census.

The all-task RTX4x target remains unmet. Latest known RTX ratios are Franka
1.95x, Kuka about2.47x versus the previous MJ reference, Allegro4.21x fresh,
ANY3.91x, G12.08x and Keyboard3.69x. Do not relabel old rows as new measurements.

## September14 22:29: retained-Gram numerical diagnosis before promotion

Combined qualified tree1a9efc33 and ledger281d50a7 are pushed and remote-verified.
Its six existing integration CUDA selectors pass on each card; these do not
constitute a new whole benchmark of the combined source.

ANYmal prototype acdb168e is frozen. One causal compiler correction prevents
cross-column Gram loads from keeping a72-value temporary tile live. Final AOT
register counts are111/121 on both architectures, with no stack/spills. This
reduces the original resource risk, not proof of a timing gain.

Actual three-selector CUDA suites fail the saved-case selector on BOTH cards;
warm/padding and production graph/fallback selectors pass. Full diagnostic
replay preserves the failed gate and evaluates all2,048 cases including29
limit rows: one case fails, capture GPU1/current/tier32/world357. Its relative
cross-variant velocity difference is8.998622e-5, normal-contact residual rises
from5.838641e-7 to1.217565e-4m/s, and complementarity3.572787e-7 to5.299579e-5.
Momentum and cones pass; aggregate natural and friction residuals slightly
improve. Other cases' maximum velocity difference is3.140934e-6. This is not
just a bit-identity failure, nor evidence of dropped contacts.

Existing FP64 dense/factor references stop at16 of24 and agree with the old
native result. Ordinary CPU arithmetic does not reproduce the new native
deviation; native setup/restart/stopping attribution remains open. Keep the
original physical thresholds, source and failed results unchanged. Artifact:
`/tmp/fpgs-anymal-register-physical-diagnostic-rtNJ6c/actual/gpu{0,1}.json`.
The lifecycle also emits two stream-wait invalid-argument warnings per card
around graph-to-eager transition; a passing assertion is not a clean-warning
claim. That event lifetime is being checked separately.

One original16K paired whole A/B is running as EXPLORATORY performance only,
while a native trace diagnoses the numerical case. It is not a promotion or
physical-qualification claim. This avoids spending hours on a numerical detail
before learning whether the structural change has worthwhile performance.
No timestep, substep, maximum iteration or capacity change is authorized by
this diagnostic. The all-task4x minimum remains unmet.

## September14 22:57: close two attempts; fund a different complete G1 boundary

ANYmal retained Gram is closed, not promoted. Whole16K original checked A/B:
RTX9.332332->9.525339ms, GB9.530262->10.399400ms. Strict nodes attribute
parallel32/48 total4.067974->4.388095ms RTX and4.391071->5.202762ms GB.
RTX32 alone saves0.080915ms; that does not fund a tier-only tuning branch.
Fewer products did not offset increased live register state. Actual registers
rise86/80->110/121 RTX and80/80->111/121 GB, with zero local memory; these
resource counts do not establish achieved occupancy or a hardware ceiling.

The numerical failure now has an exact native trace on both GPUs: setup is
bit-identical, but residual reassociation changes sweep15's restart dot from
+1.189393e-11 to-2.446598e-12. Old restarts/stops16; candidate extrapolates,
then stops17 with the larger normal defect documented above. Traced outputs
exactly match both uninstrumented owners. FP64 dot-only does not repair the
opposite signs of the different operands. No contact-drop explanation or
tolerance relaxation is used. A stabilization change is not funded for this
losing performance path. Report-only tip e5cccc54 preserves runtime acdb168e;
native trace `/tmp/fpgs-anymal-register-trace-Z7z4dJ/actual/gpu{0,1}.json`.
The graph-to-eager stream warnings remain disclosed, not silently qualified.

Allegro endpoint-wrench reuse a73fb4ba passes the existing four CUDA physical/
lifecycle selectors on each card. Whole discovery against qualified1a9:
RTX15.289529->14.985817ms, GB16.584617->16.714202ms. The0.303712ms RTX
saving misses the1ms gate. Existing strict node audits find complete rows+
solve7.524767->7.624923ms RTX and9.053549->10.732662ms GB. Importantly,
the parallel tiers themselves improve0.178437/0.361761ms; unchanged serial
fallback adds0.266037/1.946549ms in those short windows. The64 tier grows
0.043354/0.101301ms, with registers82->108 RTX and83->110 GB. Do not call
all solve growth a register effect. Both graph and node records are retained
in report-only7b29937d; no promotion, retuning or new accepted MJ ratio.

Next prototype starts from qualified1a9, not either losing experiment:
`newton-fpgs-g1-kinetic-state-20260914`. Keep W434 factor/rows/GS, replace the
publication->spatial-body-cache->force/composite source chain with compact
subtree moments and current/next bias. Unlike the earlier losing502 spatial
operator, no contact action or final sparse decode is replaced. Conservative
old dynamics+publication exclusive7.309204ms reserves ALL2.342925ms of the
current factor node,0.154144ms drive preparation and0.027829ms mask work.
The complete new finish+predict+repair allowance is2.784305ms for a2ms RTX
saving. This is a falsifiable budget, not a predicted or measured speedup.
Original root inertial/transport terms, current external/control forces,
current geometry versus held-factor cadence, and public state remain required.

## September15 01:18: two structural gains reproduced; Franka implemented

The complete G1 current/next moment-and-bias owner removes2.531360 ms RTX
physics in three alternating paired16K rounds:18.188705->15.657344 ms,
1.161672x. GB20.916696->20.503290 ms,1.020163x. No old current dynamics,
composite, predictor or publication producer remains in its selected boundary.
The new finish/predict/repair sum3.148854 ms RTX exceeds its initial2.70 ms
allocation, but geometric-input factorization additionally saves0.552298 ms.
The smaller GB gain is explained by its cheaper original boundary and more
expensive new owner, not a hidden duplicated producer. Physical current/held,
loaded contact, reset/graph and retained Allegro/Kuka CUDA checks pass on both.
The reproducible cyclic stream-destruction crash was fixed in Newton teardown;
there is no per-step synchronization change. Code/report is pushed on
`ooctipus/fpgs-g1-kinetic-state-20260914`, report tip8cd56886.

Kuka's combined current geometry/ZERO/allocation cache removes0.519343 ms RTX
physics over qualified1a9 in three paired rounds:10.999563->10.480220 ms,
1.049555x. GB11.276758->10.689198 ms,1.054968x. Old CSR/ZERO/count/allocation/
triplet entries are replaced, not retained alongside it; the remaining response
contraction costs1.476172/1.860778 ms. The original1ms saving plan is missed,
but the measured whole gain is useful. New-boundary CUDA and actual current/
held4K1608-call physical checks pass. Natural trajectories lack a coupled
cohort; the retained loaded coupled/type-4 CUDA fixture passes separately on
both cards. This compositional qualification is explicit, not relabeled as
complete naturally coupled coverage. Runtimec1c678a and detailed results are
pushed on `ooctipus/fpgs-kuka-current-contact-20260915`.

Latest RTX physics ratios using each task's LAST corrected MJWarp denominator
(not a new all-backend benchmark) are Franka1.95x, Kuka2.62x, Allegro4.21x,
ANYmal-D3.91x, G12.42x, Keyboard3.69x. Only Kuka/G1 changed in this paragraph.
The user requires at least4x for every task; that minimum remains unmet.

Franka now implements the same complete current/next representation retirement
for13 bodies, retaining original9/6 factors and every contact/PGS consumer.
Saved current H/bias/held-force/publication and actual loaded eight-sweep reset/
notification/two-bank graph tests pass on both cards. The final formatted full
suite is being rerun, followed immediately by original16K whole A/B. There is
no Franka timing claim yet. Native checkpoint is before its01:59 deadline.

Next Kuka work is one distinct MF-count-zero private response/eight-sweep owner
forked BEFORE current MF preparation, retaining positive-MF fallback and the
complete post-join guard/publication. Current complete affected region3.642265
ms gives a2.595454 ms allowance for a10% whole gain, including reverse-map,
allocator, fallback, overlap and resource costs. Prototype checkpoint02:40 UTC;
no mapping grid or new benchmark framework. Its isolated branch is
`ooctipus/fpgs-kuka-private-response-20260915`; it has no measured result yet.

## September15 01:57: two new prototypes miss; accepted gains unchanged

Franka's first complete owner screen passed capacity/source guards but measured
5.464454->5.369242 ms RTX and5.032037->5.412950 ms GB. Worse, wall time grew
from29/28 ms to1694/1693 ms. The new immutable-plan validation rebuilt all16K
worlds on every unchanged root-property notification. Regression-first host
fix `edd49d0b` fingerprints ALL proof inputs and avoids that redundant loop;
no notification, numerical flag, solver allowance or Lab source changed.
Three physical/lifecycle CUDA selectors still pass on both cards. Fresh whole
screen02 now measures5.457961->5.466557 ms RTX and5.025921->5.383826 ms GB:
no physics improvement. Wall is44.972803/44.296596 ms, still worse than accepted
28.784147/27.884037 ms; added full fingerprint checks remain costly. No promotion.
Manifest `/tmp/fpgs-franka-kinetic-state-paired16k-20260915-02/manifest.json`
SHA256 `06a30f25fd7e441d4781df9ef1daa9d1e1660cf716f4e30098c76778689ff255`.
All four children, eight capacity boundaries,13 helper rehashes and source/idle
guards pass. A matched node profile is in progress to diagnose the GPU miss.

Kuka private-response native boundary and corrected actual two-bank stream/
fallback/failure-publication tests pass on both cards at `867e6b77`. Its first
graph screen is preserved as INVALID comparison evidence because a test-only
fixture correction crossed the source freeze; runtime bytes did not change.
The clean node comparison confirms the loss:10.530701->14.054306 ms RTX and
10.841696->12.144554 ms GB. All source/capacity/idle checks pass. Original strict
reader finds12 physics roots, no unproven nodes, and1728->1752 total nodes.
The complete affected fork union grows3.564513->7.128285 ms RTX and
4.095402->5.439198 ms GB. New private response/eight alone costs6.153778/4.750538
ms; allocation adds only0.005323/0.003093 ms. This is a concurrency/serial-work
problem, not hidden global panel storage or spills. A four-warp contact build
alone has little RTX budget headroom because positive-MF offset remains queued
behind private on the same stream. A single corrective hypothesis is being
costed for cooperative contact construction and removal of that false stream
dependency, charging any lost overlap of the original mixed-world solves.
No correction or new gain is claimed yet. The clean node manifest is
`/tmp/fpgs-kuka-private-response-nodes-paired16k-20260915-01/manifest.json`,
SHA256 `f66f23db48331c457fa46c5119a5f5ac2d1c981f02334864156f59a6488481dc`.

The accepted RTX ratios above remain unchanged; the at-least4x-every-task
target is still unmet. These losing prototypes have not replaced accepted
runtime or been used as new improvement baselines.

## September15 02:55: retained composition, closed Kuka correction, Franka gain candidate

The independently repeated G1 and Kuka current-contact gains are now composed
without semantic edits on `ooctipus/fpgs-kinetic-qualified-20260915`, merge
`ded651f83ce4539e4f7ab13bebb5edde9d4dcc4d`, report tip
`b1bad06ae0cd5b86ac07b2f2844ef62770e99075`. The branch is pushed and its remote
hash verified. All106 selected CPU controls pass with two explicit CUDA skips;
root's ten actual CUDA selectors pass on each card. These are integration
checks, not a new all-task timing sweep. The qualified report includes exact
parent/source pins and retains handoff ancestry. No Lab pointer changed.

Kuka's one causal cooperative correction is closed as a loss, not discarded
without diagnosis. Four warps construct separate contacts and the false
private-to-positive-solve host dependency is removed. Clean whole graph costs
are10.411883->11.834728 ms RTX and10.678865->11.665263 ms GB. The strict complete
affected union grows3.595766->5.092613 ms RTX and3.967386->4.763133 ms GB.
Private response/eight improves6.153778->3.146638 ms RTX versus the FIRST
PROTOTYPE, but still exceeds its entire2.595454 ms replacement allowance before
retained work. Only0.614892 ms of retained-main sums overlaps private; positive
solves remain fully exposed in the measured interval. Physical controls pass;
no promotion or further mapping grid. Both original/private source-guard
failure and clean corrected evidence remain in pushed report tips4affa72a and
60a8a454, respectively. Accepted Kuka remains the current-contact branch.

Franka's complete kinetic owner received one all-three-owner paired16 mapping
correction. Independent source review and the original three CUDA selectors
pass on both cards, including five-world odd-tail/mixed-cache/error isolation.
One clean whole round at91f505f5 measures5.453217->5.032845 ms RTX (1.083526x)
and5.060164->4.729254 ms GB (1.069971x). This positive result misses the declared
0.55 ms RTX saving by0.129628 ms and is not repeated or promoted. Wall remains
28.026434->44.762043 ms RTX and28.648983->44.353130 ms GB. Detailed physical,
whole and matched-node closure is pushed atc643e157 on its isolated branch.

The actual three-owner RTX sum is1.463051 ms, not the proposed<=1.052093 ms;
full matched state/factor/drive/mask exclusive falls2.049643->1.675650 ms.
Lost original overlap is not recovered. Static driver queries additionally
show RTX102400 shared bytes/SM and24 maximum CTAs/SM: doubled state scratch
7296 bytes/block limits nominal shared-memory CTAs to14, rather than the old
24-block combined ceiling at3712 bytes/block. This weakens the ideal2x packing
premise; it is not achieved occupancy or a throughput/stall proof. No third
packing variant is funded.

A separate bounded Newton-only repair now targets the remaining approximately
16 ms host regression, not a new physics milestone. Unchanged notifications
copy/hash22 arrays,26,017,792 bytes at16K. The isolated device-proof candidate
retains all checks with exact-sized constructor snapshots and word comparisons
before one four-byte readback. The original null-stream read envelope is
preserved. Root CUDA and whole-cost checks are pending; checkpoint03:20 UTC,
no silent extension beyond03:35. Accepted ratios remain those at01:18 until
qualification is complete. The goal controller still reports paused; it has
not been marked complete, and at-least4x on every RTX task remains unmet.

## September15 03:26: Franka repeat closed without promotion

Franka's exact device-side notification proof passes eleven CPU controls and
four existing/new CUDA selectors on each card. It removes the prototype's
large host-copy regression without weakening static-value checks or the
original null-stream read envelope. Runtime remains865dbd6c; report-only
tip106a149fe5db78d876f834e1f60ff8bfc7709118 is pushed and remotely verified
on `ooctipus/fpgs-franka-device-proof-20260915`. The first whole attempt is
preserved: its GB baseline exited139 during scene replication, before CUDA
initialization/results, and no candidate arm ran. The exact fresh-directory
retry and subsequent alternating three-round comparison complete cleanly.

Against the accepted1a9 baseline, median physics is5.545875575->5.070682400 ms
RTX (1.093714x,0.475193175 ms saved) and4.999355200->4.721928800 ms GB
(1.058753x,0.277426400 ms saved). All12 repeated children exit0 and all24
capacity/collision boundaries, finite-state, source and final-idle guards pass.
The repeat manifest is
`/tmp/fpgs-franka-device-proof-repeat-paired16k-20260915-01/manifest.json`,
SHA256 `01527229b579d1353e5b670340b4dd78307f69f4424d80442aa523b90f8df45d`.

No round reaches the declared0.55 ms physics milestone. RTX paired wall deltas
are+0.366682,-0.630973,-0.004096 ms, approximately unchanged. GB deltas are
+1.195283,+0.048059,+0.235423 ms: slower in all three pairs, mean+0.492922 ms.
The almost-neutral ratio of GB wall medians must not hide that paired result.
There is no established environment-step win or promotion. This candidate
stays separate from acceptedb1bad, whose six-task MJWarp ratios are unchanged.
No third packing/host-tuning round or new all-task benchmark is funded here.

Read-only follow-up studies distinguish untested ideas from measured losses.
Full G1 body-plus-present-limit ports remain algebraically plausible but were
never implemented; prior sparse packets and lazy compliance are different,
measured losing representations. Warm-starting the second Solver.step did
not reduce sweeps on the selected hard cases. A new conservative within-solve
zero-row expiry bound is being assessed on the existing current/held payloads;
there is no native implementation or speedup claim for it at this checkpoint.
The at-least4x target is not achieved and the paused controller is unchanged.

## September15 04:09: first zero-row expiry loses on RTX; one correction

G1 solve-local zero-row expiry at clean6c2e1fa9 passes nine CPU controls and
eight original/focused CUDA selectors on each card. It preserves the original
eight-sweep law, current/held geometry, all row/collision capacity checks and
default-off native source. All four original whole16K children pass: accepted
b1bad15.608484 -> candidate15.894541 ms RTX, a0.286058 ms loss; GB20.468370 ->
20.230434 ms, a0.237936 ms gain. No promotion or accepted-ratio change.
Whole manifest SHA256:
`d25e107a5cd590a926c7a144110bb89ef78e93e2256c75bcc022a5e2e961862c`.

Both node parents exit1 at the inherited auxiliary-graph analyzer rejection;
the original failures remain. Existing strict ownership reader accepts both
saved baseline and candidate pairs:12 physics roots,3 auxiliary roots,
768 physics nodes,0 unproven nodes, with source/process/correlation checks.
RTX complete GS exclusive grows3.943349 ->4.175872 ms; every other family
changes by at most0.004181 ms. GB GS falls4.363572 ->4.113962 ms. This isolates
the timing change to GS but does not establish which internal check dominates
or that selected CPU omission counts represent the full population.

One correction is funded before the original05:00 checkpoint: remember the
certified scalar/triplet kind and test expiry before repeated immutable row
metadata and impulse guards. Keep upfront complete row validation, the same
conservative norm/travel/rounding bound and all producer/setup costs. No initial
certificate seeding, new stopping law, mapping grid or lowered milestone:
require at least1.5 ms whole RTX saving against acceptedb1bad, not against the
losing6c2e prototype. This correction is not yet measured.

## September15 04:58: close expiry; fund present-port replacement

The one zero-expiry correction is measured and closed, not promoted. Clean
runtime `bc0cfd161c30631ac1cdc633e35b944a8bd507fd` passes ten CPU controls,
eight actual CUDA selectors per card, and the unchanged paired whole16K
protocol. All four children and eight finite/capacity boundaries pass; source,
requested/observed owner and final-idle guards pass. Acceptedb1bad physics is
15.604870425->15.305637050 ms RTX (1.019550534x,0.299233375 ms saved) and
20.516338400->19.752022575 ms GB (1.038695573x,0.764315825 ms saved).
RTX wall is29.290500650->29.393717099 ms, slightly slower; GB wall is
35.519032949->33.621285800 ms. These are not training measurements.

The correction recovers the first RTX loss but misses the declared1.5 ms
milestone. No third tuning round, corrected node capture, repeated comparison
or accepted MJ-ratio change is claimed. The first loss and original auxiliary
analyzer failures are retained. Full closure is pushed and remotely verified
at `de2e288274550abe4c7ce3ed2bfba93ac777eca6` on
`ooctipus/fpgs-g1-zero-row-expiry-20260915`. Corrected whole manifest:
`/tmp/fpgs-g1-zero-row-expiry-whole-paired16k-20260915-02/manifest.json`,
SHA256 `27a20e538216bb86e635e5b13734711f434d7a5d724decda6ab8c50585ddcd49`.

A distinct G1 present-port replacement is funded after a complete pre-code
card, commit `11d5ee6d6f43f60fa75d14c9fd47602c35e1ab99`, on
`ooctipus/fpgs-g1-present-ports-20260915`, directly from acceptedb1bad rather
than expiry. The card is `reports/fpgs/G1_PRESENT_PORTS_20260915.md` in that
worktree. It replaces contact Z, limit response and metric GS with one compact
body/limit response representation, maintaining only actually committed
impulse deltas and decoding once. Up to32 present ports are supported;
unsupported bodies, two responding bodies and larger sets take the complete
original fallback before public writes. Current geometry is refreshed on held
factor calls. The ordered eight-sweep metric law is unchanged.

The full affected RTX boundary is5.743082666 ms; all new geometry, response
construction, solve, decode, guards and retained fallback must fit within
4.243082666 ms to save1.5 ms whole physics. Explicit shared planning is5636
bytes plus compiler overhead. Six geometry scalars per already-reserved row
cost39,321,600 bytes at16K/dense100; row capacities are not increased and old
Z allocation remains for fallback. This is not an allocation-saving claim.
Selected CPU algebra passes, but residual-dot savings are substantially offset
by port-update arithmetic. Native performance must establish whether removing
global response traffic and reductions wins; no operation-count-to-time
prediction is made. Independent tests/review run alongside runtime work.
First integrated readiness checkpoint06:15 UTC, no silent extension past06:45.

ANYmal has a separate CPU-only active-dual semismooth Newton study under
`/tmp/fpgs-anymal-semismooth-cpu-ioOhLY`. The existing2048 saved cases pass
unchanged physical gates in FP64 and guarded FP32 controls. FP64 median outer
iterations are6, maximum13, active linear dimension at most18; the original
executes all24 iterations in2026/2048 cases. Raw FP32 Euclidean stopping has
long tails (median7,maximum24; residual evaluations up to171), so these FP64
counts are not native cost evidence. A baseline-independent physical stopping
control is pending. Same-active-ID Gram reuse can avoid repeated matrix builds;
all setup, pivoted LU, trials, final factor reload and fallback must be charged.
There is no ANYmal native funding or GPU gain yet. The original24 total outer
allowance and separate original eight-iteration fallback remain unchanged.

Read-only follow-ups also prevent reimplementing existing work: Franka already
builds local response Gram matrices and updates residuals; Keyboard already
solves independent key chains in parallel. A simple Keyboard scalar feasible
interval excludes almost all loaded key impulse because opposing finite limits
make those selected intervals inconsistent. A direct ANYmal port of G1's
current/next state has no demonstrated0.94 ms removable-work budget. These
closures are source/algebra studies, not new timings or general impossibility
proofs. Accepted runtime remainsb1bad and the at-least4x-every-RTX-task target
remains unmet; the paused goal controller is unchanged.

## September15 05:59: close present ports; qualify active-dual CUDA

G1 present ports is closed without promotion. Its final test-only tip00ac8a1
passes all nine native selectors per GPU, including all sixteen saved physical
cases. Root sessions45004/28609 exit0 (8.315/8.522 seconds). The initial two
failures were newly added incident/diagonal component barriers that accepted
b1bad also fails on the same physical inputs; their exact controls and failed
logs remain. Saved components now use the inherited diagnostic contract;
synthetic and final physical tolerances and all runtime bytes are unchanged.

The original whole16K screen remains a loss: RTX15.607145525->18.255871400 ms,
GB20.457185200->24.529586400 ms. All four children and eight capacity/owner
boundaries pass. Each observed candidate boundary has16,384 fast worlds and
zero fallback. This is not evidence that every unobserved substep is identical.

Matched strict node attribution locates the loss in the replacement itself.
Complete changed-family exclusive time is6.052297->8.768441 ms RTX and
6.591892->10.723241 ms GB. Main ports alone sums7.754851/9.808511 ms versus
old contact-Z, limit-prefix and GS5.716383/6.312212 ms. All retained owner sums
change by only-0.037314/-0.060689 ms. Metadata, packet prefix and all guarded
fallback launches are charged; deleting every fallback tail cannot recover
the main-owner loss. A>=1.5 ms complete RTX gain would require over54% cut
inside the new main owner with other costs held. No such work retirement is
established, so no launch cleanup, mapping grid or runtime correction follows.

The original node parents73715/90992 exit1 at the inherited auxiliary-graph
analyzer rejection. The unchanged strict reader proves all physics, auxiliary,
memory and process/correlation ownership; it excludes no unknown nodes.
Derived `strict_present_ports_node_audit.json` files live under the paired
baseline and candidate node capture directories. The complete closure is
pushed and remotely verified on ooctipus at86d421c7ade81e709bd59601f703045e64599911,
in `newton-fpgs-g1-present-ports-20260915/reports/fpgs/G1_PRESENT_PORTS_20260915.md`.

ANYmal's separate active-dual owner now has native physical qualification,
not a timing win yet. First native10dfa823 failed a missing host import and
a real rank-deficient saved-case normal-feasibility gate. The CPU terminal
fallback control reproduces the GPU defect; simply taking one original-majorizer
step and resuming Newton still fails two of four hard records. Both negative
controls are retained. One principled correction backtracks along the projected
direction after a failed Newton direction, requiring the same all-row merit
decrease. Each direction has at most eight trials, exceptionally sixteen total,
with only one committed correction and the unchanged24-outer budget. Terminal
unsupported/active-overflow paths retain only the remaining original budget.

Corrected CPU control passes all2,048 cases and singular/initial/late-budget
controls; mean residual evaluations7.72217, maximum110. These are not GPU-time
predictions. Clean16f114edfc913eaf0b761c4f72067c27055845b0 passes five CUDA groups
per card, including all2,048 actual saved cases and29 limits, current/held state,
public FK, graph/reset, warm/self and exact remaining-budget fallback checks.
Root78230/6434 exit0 (23.310/23.312 seconds). The two inherited invalid-stream-event
warnings per card and target-layout deprecation remain explicitly documented.
No physical tolerance was relaxed. Original whole16K paired session69505 exits0
but loses substantially: RTX9.400964600->21.632151675 ms and
GB9.514574625->25.799906300 ms. Both arms retain acceptedb1bad/fixed Lab53ee
comparison settings and unchanged capacities/allowances. No promotion. Matched
node attribution and actual contact-work/source checks are next: selected CPU
iteration counts do not locate this large native-cost failure. All candidate
files remain frozen. Accepted ratios and runtime are unchanged.

## September15 06:45: close the active-dual loss with causal evidence

The physically qualified ANYmal implementation is closed without promotion at
report tip `eb3bc634dbf10ce88f332d87b264146179ce25ab`, pushed and remotely
verified on `ooctipus/fpgs-anymal-active-dual-20260915`. Runtime remains16f114ed;
accepted runtime remainsb1bad. Its full closure is in that worktree's
`reports/fpgs/ANYMAL_ACTIVE_DUAL_20260915.md`.

Strict complete-node attribution places the loss in BOTH replacement solve
owners: their combined exclusive time rises4.091680->16.000468 ms RTX and
4.399958->20.390717 ms GB. All retained work is approximately unchanged.
The matched saved-input control also loses3.1--9.7x, ruling out divergent
trajectories as the sole explanation. Its first helper failed before timing
because it used live72/32 strides with saved192/1 arrays; that failed helper,
logs and the corrected actual-array-stride control are preserved separately.

Clock-only diagnostics pass all2,048 saved cases/card, byte-exact against the
frozen candidate, with <=2.55% instrumentation timing distortion. Combined
Jacobian construction and LU account for54.52% RTX/54.89% GB of summed
within-world cycles. These are NOT live wall fractions or removable-time
bounds. Every sampled case finishes without original fallback. Trial counts
match the CPU prediction, including the hard110-residual-evaluation tail.
Fewer arithmetic products did not remove dependent pivots, shared row updates,
divisions, repeated projections and synchronization. A register-LU-only repair
has no evidence for the required80.45% complete-owner reduction; no such rescue
or mapping grid is funded.

The distinct cold natural-map transfer to G1 also closes at its CPU gate:
13/16 current/held cases pass. The two difficult RTX cases exhaust the fixed
eight corrections with worse normal/complementarity/friction defects. Across
all sixteen,135 map evaluations visit7,144 rows versus the original eight-sweep
upper bound3,616, before other costs. Tightening the stopping test cannot repair
the exhausted-budget cases. No G1 native implementation follows. Evidence:
`/tmp/fpgs-g1-natural-transfer-control-20260915.md` and its pinned JSON.

The next bounded CPU test is ANYmal history-four Anderson acceleration of the
physical natural map, not the old Ant/Humanoid eight-pass block-map experiment.
It removes the full active Jacobian/Gram/LU and weights its history fit in
velocity units. It retains24 total corrections, the uniform physical stop,
all rows and the remaining-budget original safeguard. All history formation,
small solves, trial evaluations and fallback work are charged. The prior AA
tail failures are explicitly reviewed before this test; no history/tolerance
grid or GPU funding follows merely from a smaller iteration count.

No new speedup is claimed in this section. The accepted RTX ratios remain
Franka1.95x, Kuka2.62x, Allegro4.21x, ANYmal3.91x, G1 2.42x and Keyboard3.69x
against the previously corrected MJWarp references, not a fresh backend run.

## September 15 07:30: close AA4 tails; fund stationary Keyboard work retirement

The one fixed ANYmal history-four natural-map control passes 2,033/2,048 saved
physical cases and 3/3 synthetic safeguards, not full qualification. Five
failures stop near the rounded-operator physical floor; six exhaust 24 allowed
corrections; four finish worse after a budget-correct original fallback from
an already changed state. A fixed-regularizer FP64 augmented-QR replay repairs
zero of the 15, with the same correction/trial/fallback counts. This diagnoses
the easy history-solve-precision hypothesis rather than rejecting the method
without investigation. It does not rule out every Anderson algorithm.

Literal AA4 costs 33.374 residual evaluations/world on average. Reusing accepted
trial residuals would reduce that to 17.7075, but that carry was neither native
implemented nor timed. Full residual/history/fallback arithmetic is 88.897212M
products versus 84.268494M original; hypothetical carry lowers it to 55.326528M,
before all remaining reductions, small solves, projections and storage. Those
counts are not milliseconds. Physical tails close this exact candidate with
no GPU funding, parameter/history grid or claimed speedup. Evidence is
`/tmp/fpgs-anymal-aa4-natural-wcfB7tcL/RESULTS.md`; result SHA256
`fa27b4d1dbe39f0fcbc642b27bbb9082e39bcbf2941f4699c209c03538da57c0`,
QR diagnostic `02649c65b76f67f329d6ce2e0972e26aa1b3e5f92eb770ba2dbf2af05a7b59dc`.
All failures and the initial empty-quantile reporting error are preserved.

Keyboard now has a distinct, bounded integrated candidate. The finite-eight
limit study found 1,661,952 contactless-key observations across four existing
4K checkpoints; every one starts inside its enabled limits and has exactly
zero lower/upper impulse in literal eight-sweep and saved native outputs.
Unlike the closed separate private-key owner, the accepted world-wide owner
continues evaluating those no-op limits while any other coordinate changes.

The implementation therefore uses only the simpler stationary fixed-point
proof, not the nonzero finite-eight closed form. After a complete current-row
ownership mask, clear local limit flags only for row-isolated, nondense
coordinates with supported finite inputs and nonnegative enabled residuals.
All other coordinates keep the exact original loop. Preserve public masks,
both zero impulse outputs, all contact schedules and the exact changed ballot.
Signed-zero normalization need not be bit-identical; there is no physical
approximation or contact deletion. The mask scans both endpoints of every
active row after contact linking, touches only empty heads, and is published
before head compaction. Its 16 bytes/world and complete builder cost are charged.

Root created branch `ooctipus/fpgs-keyboard-stationary-limits-20260915` from
accepted b1bad. Its pre-code card is
`reports/fpgs/KEYBOARD_STATIONARY_LIMITS_20260915.md`. The ambitious 0.7 ms RTX
whole-physics milestone would require about 64% of the historical 1.087489 ms
complete sparse owner before mask/check overhead. No limit-phase timing or
GPU gain is claimed. One implementation/paired screen is funded, with no
tile grid or new benchmark framework. Accepted runtime and ratios remain
unchanged until actual integrated evidence warrants promotion.

## September 15 08:14: diagnose stationary limits; test immutable-row residency

Keyboard stationary-limit runtime `fa0b41f8e1bc270a501c294ccc2734408982b0d1`
is not promoted. All three new test methods pass on both cards; the wider
response-diagonal module is 14/15, with the documented inherited tiled
response test passing seven arguments to a nine-argument kernel. Do not call
that full module a pass. The actual GPU compilation succeeded; there was no
separate completed offline-AOT run for this candidate.

One whole-step paired screen, with all capacity/source/idle checks passing:

| GPU | Accepted physics ms | Stationary candidate ms | Ratio |
| --- | ---: | ---: | ---: |
| RTX | 7.443361925 | 7.662634675 | 0.971x |
| GB300 | 6.654806375 | 6.591330550 | 1.010x |

The existing three-step node diagnostic locates both changed RTX costs:
GS 1.078966 -> 1.129783 ms and grouping 0.265622 -> 0.285675 ms. Thus a
large solver gain is not merely hidden by the small mask producer. GB300 GS
falls 1.274048 -> 0.993770 ms while grouping rises 0.283883 -> 0.309387 ms;
that separate node sample is not a repeated whole-step speedup. Actual GS
registers remain 94/88 and shared memory remains 14,664 B, with zero local
memory. Source-level retired branch percentages did not remove the resource
ceiling or establish the number of executed heavy-world sweeps. There are no
hardware-counter grounds for a definitive instruction-stall explanation.

Raw artifacts are `/tmp/fpgs-keyboard-stationary-native-qGrY3s3m`,
`/tmp/fpgs-keyboard-stationary-limits-whole-paired4k-20260915-01` and
`/tmp/fpgs-keyboard-stationary-limits-nodes-paired4k-20260915-01`. All root
sessions are reaped. The candidate branch retains its full physical tests
and negative results; accepted runtime remains b1bad.

One distinct representation experiment is now funded from b1bad: remove the
sparse owner's four read-only shared row copies, keeping shared mutable
velocity/impulses and reading the same authoritative inputs at use. No mask,
row tier, public capacity change, additional producer or launch is introduced.
The source/cache audit finds no production mutation or output-alias blocker;
unbiased passes must read the actually passed RHS and every graph invocation
must observe newly written inputs. Private metadata retains the old encoding.

The resource case is conditional: shared storage would fall from 14,664 B to
about 3,400 B. At unchanged physical registers, 256-register warp allocation
and four register subpartitions bound both cards at 20 one-warp blocks/SM,
versus old shared ceilings of six RTX and fifteen GB300. The previously rough
21/23 register ceilings were incorrect. Reserved block shared memory and
carveout may constrain this further. Neither ceiling is achieved occupancy.
The ideal RTX owner saving is about 0.755 ms; repeated global reads, packing,
address arithmetic and changed registers can erase it. One whole-step screen
will decide whether that narrow large-gain opportunity exists in practice.

Pre-code card and branch:
`newton-fpgs-keyboard-readonly-rows-20260915/reports/fpgs/KEYBOARD_READONLY_ROWS_20260915.md`,
`ooctipus/fpgs-keyboard-readonly-rows-20260915`. The ambition is 0.7 ms RTX,
with repeated qualification also allowed for a smaller substantial gain that
actually reaches the user's 4x task goal. No new speedup is claimed here.

## September 15 08:36: retain a real Franka gain; readonly Keyboard screen misses

The original 0.55 ms Franka funding threshold should not have acted as a binary
correctness/promotion criterion. The existing balanced three-round result is
a real structural physics improvement: 5.545875575 -> 5.070682400 ms RTX,
1.093714x, and 4.999355200 -> 4.721928800 ms GB300, 1.058753x. This is an
explicit policy reconsideration, not newly created performance or a changed
physical tolerance. RTX environment wall time remains approximately neutral;
GB wall remains worse in every pair, mean +0.492922 ms (about 1.73%). Retain
the default-off option with that tradeoff; do not call it training speedup.

New merge `b9ba04a471aa659206d4e67518e421d527e41aad` combines accepted b1bad
and the existing qualified Franka report tip 106a149fe5 on
`ooctipus/fpgs-franka-retained-20260915`. Eight host-hook conflicts preserve
both disjoint G1/Franka owners, current/held mass request handling, validation
before mutation, force/size-stream dependencies, complete publication and
Kuka's accepted current-contact path. Native Franka files and tests are exact
865dbd copies. Independent source review, 77 CPU test executions and pre-commit
pass. Root's eight CUDA selectors pass on each card without skips or errors:
four Franka, three G1 and one Kuka current-contact boundary, 17.882/18.021 s.
Logs: `/tmp/fpgs-franka-retained-native-jLfTopU1/gpu0.log` and `gpu1.log`.
This qualifies the opt-in composition; its report/push is being finalized.
No fresh all-backend table is inferred from these integration tests.

The distinct Keyboard read-only-row candidate is frozen at
`46f24d32d6548ab6b8a550ade04874972cbf2cb9`. Original scalar/speculative
physical oracles and the new two-world changed-input graph replay pass on
both cards. The full response-diagonal module is 15/16 per card, with only
the unchanged inherited seven-versus-nine-argument tiled test error. Eleven
CPU controls and independent source review pass. Default-off generated
native source is byte-exact to b1bad for all three existing factory modes.

Actual CUDA driver queries confirm the intended resource change: static
shared 14,664 -> 3,400 B, registers 94 -> 88 RTX and 88 -> 86 GB300, zero
dynamic shared. Driver active-block ceilings are six -> twenty RTX and
fourteen -> twenty GB300; the earlier shared-only GB estimate of fifteen
omitted additional driver allocation constraints. This is not measured
achieved occupancy or a guarantee of proportional throughput.

The whole-step screen nevertheless misses: RTX physics 7.428472925 ->
7.777718350 ms (0.95510x); GB 6.843528050 -> 6.826663150 ms (1.00247x).
All four children and eight capacity boundaries pass, with unchanged budgets
and successful final source/idle guards. Wall times are 30.422374 -> 29.885452
ms RTX and 28.021014 -> 26.822576 ms GB, a separate one-round measurement
that does not establish a solver speedup or repair the physics regression.
The candidate is not promoted. Existing node attribution is running to
locate the failed throughput prediction; no blind cache-policy/tile grid.

Artifacts: `/tmp/fpgs-keyboard-readonly-resource-4rw2va19`,
`/tmp/fpgs-keyboard-readonly-native-IicjyVO3`, and
`/tmp/fpgs-keyboard-readonly-rows-whole-paired4k-20260915-01`.
The accepted b1bad worktree and all original worktrees remain unchanged.

### 08:44 follow-through: published opt-in; localized loss; one shared tier

Franka opt-in integration is now pushed and remotely verified at report tip
`28f93014520504e90be34a304507368c7c82c4cc`; runtime remains b9ba04a4.
The full report is `reports/fpgs/FRANKA_RETAINED_INTEGRATION_20260915.md` in
`newton-fpgs-franka-retained-20260915`. It retains the original measured
physics gain, GB wall caveat and unchanged default-off option. No old negative
report was rewritten and no new all-backend timing was invented.

Readonly Keyboard's original node analyzer completes successfully on all four
children, with all boundaries/source/idle guards passing. Each capture has
12 physics roots and 1,260 graph nodes: 936 kernels, 204 memsets, 120 copies.
RTX graph span grows 7.607929 -> 7.779795 ms; the changed GS itself grows
1.287766 -> 1.448033 ms, explaining most of the +0.171866 ms in this sample.
GB GS is 1.426954 -> 1.435936 ms, essentially unchanged, while graph span
is 6.897206 -> 6.854346 ms. These three-step node windows are not the separate
40-step whole result. Actual resources match the driver query, with no spills
or new launches. Additional global rereads remain the source-level new cost;
there is no hardware-counter attribution to a particular cache/stall cause.
Full unpromoted closure is pushed at `daf0c407432c8fb08ee630c92327fbf7ba5653c6`
on `ooctipus/fpgs-keyboard-readonly-rows-20260915`.

One distinct, costed alternative is funded from b1bad: private shared rows384
with the original public704/S235 strides, all original shared reuse, and a
disjoint original full704 tail on the existing size6 stream. Reuse the existing
otherwise-unused size events; join before any output consumer. This introduces
no global classifier/queue/JY panel/new stream or public-capacity reduction.
No stationary mask or read-only-row code is carried into it. The single
pre-code card is `newton-fpgs-keyboard-shared-tier-20260915/reports/fpgs/KEYBOARD_SHARED_TIER_20260915.md`.

Its route is narrow, not promised: at an expected ten RTX resident blocks,
ideal scaling leaves only about 0.059 ms overhead/tail allowance using the
earlier1.079 ms owner, or 0.143 ms using the newer1.288 ms node sample, to reach
the approximately0.372 ms saving implied by the recent whole baseline and
historical4x ceiling. Actual compiled resources are checked early. Full cost
is max(bulk, tail) plus fork/join and interference, not bulk time alone. There
is no tier-size grid, and historical world coverage is not a timing claim.
