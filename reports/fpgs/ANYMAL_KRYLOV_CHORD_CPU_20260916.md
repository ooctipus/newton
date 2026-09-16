# ANY18 coupled Krylov/chord CPU falsification

## Status

CPU-only, unpromoted; no runtime or GPU implementation. The initial frozen
policy improves aggregate physical quality, but its retracted Newton chord
can destroy descent. A separately approved projected-direction safeguard is
pending. Preserve this initial result; do not attribute later results to it.

The complete map is the old simultaneous normal-first Coulomb map: physical
`r=b+ZZ^T x`, denominator-only CFM, normal clamp followed by the own-normal
tangent disk. Analytic `Jv` includes the sliding radius derivative. The helper
uses a cached active physical Gram, local nonsymmetric right preconditioner,
GMRES at most eight steps with CGS2, true full-equation forcing verification,
and one physical response per cone-feasible chord. It does not form or factor
a global Jacobian. At most 24 committed corrections, eight line trials per
direction, original physical gates, fresh final response, and literal original
continuation from the current state with the remaining allowance are retained.

## Frozen provenance and regressions

- Worktree base: `2f70a1f7b41eed945806f7b73633babe198d6263`.
- Helper: `88a2cc806b618deebc9c39a76f7fd2d987cdc0f9a3c2e70e4b906af8663efd5e`.
- Tests: `301764615f82e07474f28e019690adb95c55fb699b7eb34d2efbe845da4b2d68`.
- Approved card: `/tmp/fpgs-anymal-krylov-chord-card-nJhiwwNd/CARD.md`,
  `783517c6b3153af5564447c26a8cb0179ea2e1df4ec0421ec2a0d326e70f914a`.
  Relative to approved `79da...`, this only clarified full-map forcing,
  inherited Armijo, and false carried-stop/budget semantics before evaluation.
- Reused adapter: `/tmp/fpgs-krylov-chord-cpu-622baUWd/qualify.py`,
  `59e9a4f3f8eca4656b66e1b3307733c3bbd12fb08a7ac4f46c2bfeb65ca3c729`.

Missing-module regression failed before implementation. Nine focused controls
plus the inherited original-recurrence control passed: analytic open/stick/
slide/radius derivative, chord-versus-projected-line counterexamples, coupled
and inconsistent normals, inactive coupling, true full-map forcing, current
rows/held factor, nonzero incoming fallback, late remaining-one handoff,
zero allowance, false carried stop, and an independent sticking solution.
Independent source review found no mathematical or budget blocker.

## Initial results

The existing saved reference adapter verified original inputs and converged
reference archives before and after evaluation; no new reference was solved.
First eight: all hard/pointwise checks passed and all H-reference distances
improved, with 3–6 corrections. Counted work was 17.13565 original action-pair
equivalents, above the planned 10–12 gate. Root explicitly authorized the cheap
unchanged full cohort because the coupling hypothesis remained promising;
this did not authorize native implementation.

Full 2,048: zero hard failures; 2,017 original pointwise-gate passes and 31
component tradeoffs; 2,032 closer H-reference distances and 12 genuine scaled
H-regressions; 2,029 improved natural residuals. These categories are distinct.
All 12 H-regressions followed a line-search fallback.

| Physical diagnostic | Original median / p99 / max | Initial candidate median / p99 / max |
| --- | --- | --- |
| Scaled H-reference distance | .006390 / .031996 / .055738 | .00000263 / .010277 / .030266 |
| Scaled natural residual | .002593 / .020372 / .039640 | .000000838 / .002952 / .016109 |
| Biased negative normal velocity | .000491 / .014564 / .034372 | .000000666 / .000908 / .034344 |
| Complementarity | .000887 / .009734 / .044360 | .000000591 / .001133 / .015645 |
| MDP gap | .002092 / .035477 / .079436 | .000000650 / .002066 / .089709 |

The worst H regression is RTX held/tier48/world232: absolute H distance
.01928136 to .02552857, scaled .00835215 to .01105826. The worst normal
regression is RTX current/tier48/world21, .00435972 to .03434400. GB current/
tier48/world168 has MDP .07943639 to .08970913 and complementarity .01281264
to .01564473. These are genuine finite-budget tradeoffs, not hard cone or
momentum failures and not automatic evidence of a universal improvement.

Known world142 uses three candidate corrections then exactly 21 original
passes in all four device/current-held cases. All four miss the old normal
component gate. Scaled H distances are RTX current .0135820 to .0136481,
RTX held .0218130 to .0239260, GB current .0145330 to .0135167, and GB held
.0251839 to .0219298. No fallback resets the state or allowance.

## Complete counted work and limitations

There are 7,345 accepted candidate corrections plus 1,454 original fallback
passes, 1,982 fresh physical stops, and 67 fallbacks: 58 line searches and
nine true linear-forcing failures. Actual Krylov counts total 9,876 and
squared counts 18,332, with 7,412 true linear checks. The record charges 2,229
Gram builds/5,183 reuses, 56 inactive-coupling actions, 7,403 chord responses,
7,971 trial maps/626 rejected trials, 11,441 physical scores, and 2,048 fresh
physical reconstructions. No failed trial is free.

Counted products: 34,404,153 versus original EX1 plus recurrence 84,268,494;
one full-cohort original action pair is 2,154,060. Thus the initial proxy is
15.97177 equivalents, not the historical card subtotal. Additional recorded
services are 2,875,225 divisions, 411,912 square roots and 87,618 dot/norm
reductions; these are not instruction-exact totals or synchronization counts.
In particular retraction is only recorded as disk visits, and physical max
reductions are not represented by the dot/norm count.

Independent post-run accounting identifies at least 277,462 omitted products
(three per retraction disk and two per sliding derivative mode), plus 72,546
retraction square roots. The corrected lower bound is 34,681,615 products,
16.10058 action equivalents, before other unpriced scalar/control services.
Map disk norms also imply square-root work beyond the explicit counters.
The card's historical CGS2 q=4 count was 24 reductions; the implementation
correctly charges the pre-CGS norm as well, giving 28. No cohort was rerun or
numerical result changed for this accounting clarification. Retained geometry,
whitening and decode remain outside both operator-only product totals.

The original native dense active-dual experiment lost badly, despite a useful
CPU method. Product retirement is not a GPU speed prediction: sequential
Krylov reductions, physical gates, active staging, fallback and resources still
need complete timing if a native prototype is subsequently authorized.

## Cause, not another parameter search

Frozen replay of all 67 rejection states reproduces each complete work record
exactly. All 58 rejected Newton directions satisfy true relative merit slopes
between -1.000 and -.920. Endpoint retraction changes 24–95% of the direction
norm: 50 resulting chords are ascent directions; eight descend locally but
all eight permitted step lengths are too long. One-sided differences confirm
the analytic slopes. All nine forcing guards reach the fixed eight-step cap;
there is no rank, nonfinite or derivative bug in these records.

The old corrected active-dual safeguard uses exactly `d=pg-x`, where `pg` is
the current complete-map endpoint. Both endpoints are cone feasible, so its
old re-projection of `x+alpha*d` is redundant in exact arithmetic. One physical
`Gd` can serve all eight merit trials. At the 58 frozen line-failure states,
this direction accepts in 16/28/13/1 cases at alpha 1/.5/.25/.0625. This costs
71,568 response products and 116 map trials for those FIRST retries only;
it does not predict subsequent trajectories or complete corrected cost.
Root approved one separately frozen CPU correction on that evidence, not a
new merit-gradient policy or a parameter grid.

## Durable artifacts

All paths are local retained artifacts; summaries pin complete input/reference
and source dependencies, and compressed files preserve full traces.

- First eight, `/tmp/fpgs-krylov-chord-cpu-622baUWd/first8/summary.json`:
  `ad73e3515fc77df41953eabd7cd9616c498412ba653d564c0a4d7ca56b2fd97b`;
  trace `a434d912365eb1872dccb80f08f98dcd19a254ae1caa8a2f516d7312cd284e52`.
- All2,048, same root `all2048/summary.json`:
  `e5b622fdb05fb02b1000ca68eebf95d84ff4b3b70bacd0eddf4694ccf7ab14fc`;
  trace `8d7fb438fdc7f1b5894082c7fed526c3fb333df5893f5921ddf2ba4c935eee35`.
- Cause `/tmp/fpgs-krylov-chord-cause-HseE5K9Z/check.py`:
  `5d580413c8026bd18b7d05fc753be6954dda8ac04bc4e1280fdbf7745e4acd83`;
  result `3754e203c3bf029ae01ac75a5cb4c2e028e6dcd605ffeb13b64644a3071a3e5f`.
- Prior safeguarded direction source:
  `/tmp/fpgs-anymal-semismooth-cpu-ioOhLY/safeguard_projected_line.py`,
  `f2b4d5052dbbe9f933435f287ca8198574a17400608c2203b4a3e2d24ab7eeab`.
