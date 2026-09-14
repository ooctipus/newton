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
