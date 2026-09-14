# G1 bounded force-to-velocity owner

Candidate card, 2026-09-14 09:21 UTC. First implementation checkpoint 10:51 UTC;
no silent extension past 11:21 without new causal evidence. Based on current
good `7df75c46bd468b64c17958190fc800cfeb62201e`, not the slower spatial branch.
Default off; Newton only; all original timestep/substeps/max-eight allowance,
capacity and physical-quality constraints remain.

## Hypothesis and measured demand

Admit from current geometry and active joint limits before force prediction:
at most eight contact reservations and 32 total rows. Other worlds use the
existing complete sparse path. In four fresh 16K boundary snapshots, this
class covers 83.23–83.91% of worlds, 69.91–70.90% of rows, 70.19–71.17% of
supported row entries, but only 59.90–61.20% of nominal contact-W products.
Every admitted world still averages approximately 7.3 active joint limits.
No zero-impulse admission or omitted physics is allowed. Raising local storage
to 40 rows adds fewer than 0.5 percentage points of worlds, so it is not funded.

One owner will consume current body forces, control and the held W434 factor,
predict velocity, form only local supported rows, run the metric-tangent
updates, and decode final generalized velocity. The selected-world original
force recurrence, predictor and global Z/incident/diagonal/RHS producers must
be removed, not retained alongside it. Retain minimal canonical row identity
and final impulses for existing contact-force export.

Retain and charge mass/composite refresh, current/next body dynamics caches,
generalized integration and required public FK/publication. This first
replacement is an explicit force-to-final-velocity boundary, NOT a claim
that state publication is free or already eliminated. Keep the useful held
factor; no new implicit ABA, dense43 row padding or Gram matrix. The previous
full-capacity 100-row shared packet is not the new hypothesis.

## Cost model and failure conditions

Current RTX diagnostic owners cost grouped force 1.436 ms, predictor 0.410 ms,
rows/response 2.516 ms and solve/decode 3.944 ms. These are owner intervals,
not measured per-world costs. Applying the distinct uniform-versus-row work
fractions suggests an approximately six-ms selected old boundary as a working
model, NOT an observed timing cut. Funding hypothesis: a complete replacement
can execute that selected work in approximately four ms including classification,
fallback masking, any lost internal dynamics overlap and final decode, yielding
at least two ms whole-physics saving on primary RTX. Test this assumption early.

Adverse costs are explicit: direct static descendant force accumulation removes
11 tree levels but performs roughly five times the tree-edge vector additions;
local rows can lose the old per-contact parallelism; shared/register lifetimes
increase; the remaining 16–17% of worlds own approximately 40% of contact
formation work. No time credit is assigned merely for cohort membership.
Fixed Lab already runs collision before the solver, so this workload does not
lose a material collision/dynamics overlap when selecting early. Internal
force/mass-stream overlap still matters.

Stop/go is complete paired physics timing against 7df, not a local-kernel win.
Require no extra/missing canonical producers or stale fallback values, then
check momentum/cone/complementarity/current-versus-held response on existing
small fixtures and actual reset/refresh/empty/regrowing/32-to33-row transitions.
Finite trajectories or unchanged row count alone are not physical acceptance.
On a miss, diagnose which producer/consumer/selection cost defeated the model
and allow one costed correction for that cause. Do not start a layout sweep.

Fresh demand source:
`/tmp/fpgs-tenhour-g1-shared-paired16k-20260914-01`, summarized independently at
`/tmp/fpgs-g1-current-cohort-summary-MpGi6XOa/RESULTS.md`. Fresh whole reference
FPGS/MJWarp: RTX 19.4710592/40.876258325 ms, GB 25.06668425/44.9422667 ms,
one corrected shared-collision comparison. No new optimization gain yet.

## Measured outcome and closure, 10:32 UTC

Runtime `d9beb746fb416c3b1db0f591a97c6d53d102b269` is a diagnosed loss,
default off and unpromoted. It is not the baseline for future claimed gains.
The unchanged paired 16K/200 warm/40 wall/40 physics recipe gives:

| GPU | Good7df physics | Small-step physics | Good/small | Wall good/small |
| --- | ---: | ---: | ---: | ---: |
| RTX PRO6000 |19.471282ms|21.260241ms|0.915854x|33.733745/35.303672ms|
| GB300 |25.111638ms|27.752758ms|0.904834x|38.514887/41.600056ms|

All four processes exit0. Original capacity, actual-mode, fixed-Lab import,
source and final-idle checks pass. Artifact:
`/tmp/fpgs-g1-small-step-paired16k-20260914-01`.

Six native numerical selectors pass on each GPU, including current force and
geometry against held W, momentum/cone/residual controls, stale incoming
impulses and empty/regrowing/fallback boundaries. An actual stale-impulse bug
was reproduced before timing: the private solver must initialize lambda to
zero because selected worlds deliberately skip the old canonical clear.
The regression and correction are in the measured runtime. Native logs:
`/tmp/fpgs-g1-small-step-native-paired-GLPaHLPt`.

Additional full-Solver.step lifecycle tests pass: two CPU guards and one
actual native case/card, without skips. The existing G1 USD fixture exercises
32 selected ->33 fallback ->empty ->32 regrowing rows, original masked
launcher identities, final velocity against metric control, public FK,
current/held W, request consumption, reset and repeated two-step graphs.
The test is retained as `newton/tests/test_small_step_lifecycle.py`; external
measured source SHA256 was
`ef331753ebce4fc8d685707f0d70f502bc97378f2661568e96e1a9a5756e9de5`
before addition of the repository copyright header and formatting. Evidence:
`/tmp/fpgs-g1-small-step-lifecycle-6KbcJvxA/RESULTS.md`. These are bounded
physical/lifecycle checks, not sustained task qualification.

### Why the cost model missed

The selected old work was overestimated. The remaining approximately17% of
worlds still cost2.137185ms RTX /2.094538ms GB in fallback GS. On RTX this is
54% of the original3.944015ms GS, not17%. Original contact work similarly
falls only1.253346->0.759360ms, while the early layout prefix adds0.474037ms.
Real selected force and predictor retirement is observed; no duplicate full
selected producer was found. The new complete owner itself costs5.578584ms
RTX /6.199157ms GB, offsetting the savings.

One bounded in-place six-clock diagnostic explains the owner, without replaying
stale ForceInput after Solver.step. Current per-world mean phase fractions at
both original untimed boundaries are approximately force9%, predictor12%,
rows30%, GS46%, decode3% on RTX; GB9.5%,9%,29%,50%,3%. These are SM cycle
observations, NOT additive whole-step millisecond attribution. Cost grows
regularly with contacts; geometry and W are already shared across the three
friction directions. Serial contact-to-contact formation loses parallelism,
but force-only tuning cannot fund a recovery. A revised owner needs roughly
3.79ms further whole-step reduction merely to recover the loss AND deliver
the original2ms target. No such causal correction is currently cost-qualified.

The diagnostic adds six lane-zero stores and one terminal warp barrier;
actual kernel5.685616/6.342314ms is1.92%/2.31% above production. Actual
resources remain72 registers,4972B shared,zero reported local memory, one
32-thread block/world. Do not infer a hardware ceiling or memory/compute bound.

Both three-step node parents preserve their original auxiliary-graph analyzer
failure after completed captures. The unchanged strict74c5928 interval reader
with explicit observed owner-name additions independently passes all source,
capacity and process/correlation checks:12 physics roots plus3 auxiliary/card,
zero unproven nodes. It does not convert failed manifests into successful
throughput runs. Node evidence:

- `/tmp/fpgs-g1-small-step-nodes-paired16k-20260914-01`
- `/tmp/fpgs-g1-small-step-checked-pBUU65Fi/gpu0_node_audit.json` (andgpu1)
- `/tmp/fpgs-g1-small-step-phase-paired16k-20260914-01`
- `/tmp/fpgs-g1-small-step-phase-4VsF97/LAUNCH.md` and `gpu0_node_audit.json`

Close this implementation at the diagnosed-loss stage before its90-minute
checkpoint. No force-only, block-size or local-row mapping sweep is funded.
The broader ten-hour optimization effort continues from good7df with distinct
work-eliminating hypotheses; closure of this owner is not closure of the goal.
