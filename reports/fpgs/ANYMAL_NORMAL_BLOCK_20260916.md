# ANY18 normal-block CPU falsification

2026-09-16. CPU-only, unqualified and not connected to production. This work
starts from `e7729d105b339e4ee6ac780b04af34fcc06feaa2` in the separate branch
`ooctipus/fpgs-normal-block-cpu-20260916`. No GPU, new converged reference,
parameter grid, iteration increase or runtime hook is included.

## Fixed policy before evaluation

The complete approved pre-code card is
`/tmp/fpgs-anymal-normal-block-card-kDpI7T/CARD.md`, SHA256
`c868df8c396a035b5fe538d5c0d29fc38dc4ca86f789078337551ab409ed917d`.
Normal/limit count is 2–15 (median10) in the existing2048 inputs. Full normal
Gram blocks are often singular; their observed positive normal working sets
are much smaller. Saved reference active IDs are diagnostic only and never
enter this implementation.

Each outer correction solves the physical normal-only nonnegative QP at
fixed tangents, using deterministic active-set transitions and lazily cached
complete physical Gram columns. A first-entered column costs N*18 products.
The initial free set uses current positive impulses or negative residuals;
omitted negative rows enter by smallest row ID. Boundary ties use row ID.
Factor reuse requires identical ordered IDs. The normalized Cholesky pivot
guard is the inherited `active_wrench_control.RANK_TOL = 32*float32eps`, fixed
before any candidate trajectory. Rank failure is not regularized. The fixed
2m transition guard counts additions/drops after initial free-set seeding.
Every successful normal endpoint passes all-normal negative-velocity,
complementarity and scaled-natural gates (3e-5,3e-5,1e-5 respectively).

Inner pivots update normal residuals only. The successful net normal change
updates every row once from cached complete columns, including dropped IDs.
One simultaneous tangent spectral disk action then uses corrected physical
residuals and new individual normal radii. Only changed tangent rows enter
the transpose action; its result acts on all rows. CFM enters only tangent
step denominators and inherited metric scaling, never the physical Gram or
normal LCP. Current J and held L are unchanged.

The initial/full-trial physical merit is exactly the frozen spectral helper.
Its first increase permanently halves relaxation. The affine midpoint is of
the COMPLETE feasible normal+tangent proposal; it receives another full
score but no additional operator. All later whole proposals use half
relaxation. Stops retain all inherited physical gates, not a proxy. Initial,
trial, midpoint and fallback-final full scores are counted. Ideal fixed
points satisfy normal complementarity and each individual Coulomb disk MDP
law; this does not prove global or finite24 convergence.

## Literal remaining-budget fallback

On rank, guard, nonfinite normal direction or failed normal certificate, the
uncommitted outer trial is discarded. Already spent columns, factors,
pivots and rechecks remain charged. The existing
`coulomb_block_hybrid.parallel_continue` receives current impulses and the
physical ZERO-impulse RHS with exactly `24 - candidate_sweeps` allowance.
It initializes original extrapolation from the current impulses and executes
the original EX1/Nesterov, restart and delta-exit recurrence. It does not use
saved original outputs or reset the allowance to24.

Nonzero incoming impulses are unsupported by the new map and directly use
that continuation. Supplied velocity already includes incoming response;
the zero-impulse RHS is therefore reconstructed by subtracting G*incoming,
with both actions charged. Final publication is based on final-minus-incoming
impulses. The CPU composition conservatively charges the continuation's
own final publication AND the outer final transpose reconstruction. No claim
of free fallback or hypothetical native optimization is made.

## Regression-first controls and pins

The new test module was run before implementation and failed with missing
`normal_block_control`. After implementation, eight tests pass: seven new
controls plus the inherited exact-original translation regression.

- Initially open normal activation, independently known coupled solution,
  and large denominator CFM that must not perturb the normal LCP.
- Singular duplicate, unequal-bias duplicate and inconsistent opposing
  normals: rank fallback equals literal original continuation.
- Dropped positive normal ID contributes to the net ALL-row update.
- Current rows/held factor preserve disks, residual carry and momentum.
- Nonzero incoming response is neither reapplied nor discarded.
- Failure after23 committed corrections leaves exactly one original pass;
  failed trial work is retained, and final impulses match continuation from
  the recorded current state.
- Exact2m transition cap and zero outer allowance.

Frozen sources:

| Source | SHA256 |
| --- | --- |
| `normal_block_control.py` | `46f67562b7339c5239245cbb6882759e1010ad2095e866703e690926008a9bb0` |
| `test_normal_block_control.py` | `6161cd65a34b647c9ab9d084e296c8c12520dfb446f8496cac6078beb8911a10` |
| Original continuation `coulomb_block_hybrid.py` | `77556360533771968e1e1918487033325fe8a7f78b86ec29131dacb134991c4a` |
| Physical metrics `coulomb_block_control.py` | `0fc44df283d5284763f7fd4889b4b8cc475b197bb6314714cf1952bd0a3d5295` |
| Full-merit helper `spectral_jacobi_control.py` | `f3cb2e5e15de8ff7d5b876bbf966d69139c134226a2338d15072ffa998385a70` |
| Rank constant `active_wrench_control.py` | `fcbb861769a2654a73d6e73f3666b9f998413c95e08034192e93a22126dd335d` |
| Pinned original recurrence `/tmp/fpgs-anymal-lazy-reference-zE5yUe/method.py` | `553c12619971e9cbeb950e1940797d91057950d70d789c1202861f1e6fcf485f` |

CPU command (CUDA hidden; no device execution):

```sh
CUDA_VISIBLE_DEVICES='' UV_PROJECT_ENVIRONMENT=/home/octi/Projects/newton-fpgs-spectral-residual-20260916/.venv uv run --extra dev python -m unittest tools.fpgs_bench.test_normal_block_control tools.fpgs_bench.test_coulomb_block_hybrid
```

## First prescribed eight

`/tmp/fpgs-normal-block-cpu-oGrTia7y/qualify.py` minimally substitutes the new
helper into the pinned existing2048 reference adapter; no reference is solved.
All source, original input and selected saved-reference hashes pass before
and after. Complete traces and numeric work are compressed from the outset
under `first8/result.json.zlib.base64`, with `first8/summary.json` beside it.

Eight cases: no hard finite/cone/sign/momentum failure; all eight closer in
held-H velocity distance; seven pass every original pointwise diagnostic;
no genuine H-reference regression. Median/max scaled H-error change from
.0102300/.0148167 original to .0000153080/.000735952 candidate. These are
selected-case scientific observations, not whole-cohort qualification.

The cost is unfavorable against the proposed native gate:166 committed
passes, four physical stops, no fallback/rank rejection, one drop,36 column
builds, nine factor builds and158 reuses. One case permanently halves at6.
There are175 full scores with5220 row visits,7135 full-score divisions and
10440 full-score square roots. Counted products are220089 versus original
338904 including EX1, but equal25.47 full original action pairs for this
sample, already well above the proposed10–12 bound before division/reduction
latency. This is chiefly remaining normal/tangent coupling, not active-set
or factor churn: normal elimination certifies the pre-tangent endpoint, but
the tangent action changes normal residuals again. Seven final merits are
dominated by negative normal velocity; the half-latched case is dominated by
complementarity. At pass12, the full merit is still1.18–363 times its bound.
No native authority follows from improved quality.

## Full2048 result and closure

Independent source review found no concrete algebra, rollback or budget
blocker. The exact unchanged source was evaluated on all2048 saved cases.
All source/input/archive guards pass twice. Artifacts:

| Artifact | SHA256 |
| --- | --- |
| `/tmp/fpgs-normal-block-cpu-oGrTia7y/qualify.py` | `de921a6d1e9833ed584f0d270d0fe4c8fe3ffe5a6e1f207f83b9831f3b34d830` |
| `first8/summary.json` | `32c3c71e34126b96cc2b7c10f05816217d5ed51af5607b9197a60957d5126be2` |
| `first8/result.json.zlib.base64` | `8588d0533cb6f1a28c5321a46c79c0fb736fdb82210149f0d9b78b039e80cd00` |
| `all2048/summary.json` | `3815ff8907af25310766c6491752d8f3f2e21f49dee2e03122b8736a1ab7be72` |
| `all2048/result.json.zlib.base64` | `9a3574c7a6fe14c466199281b3db17921f82d9e05d43d048abc1ce9b399e72ce` |

Relative artifact paths above share `/tmp/fpgs-normal-block-cpu-oGrTia7y/`.
The existing converged-reference manifest is
`/tmp/fpgs-anymal-lazy-reference-zE5yUe/all2048_01/report.json`, SHA256
`3da3d2d216c21e30c249861aebc809b35b274526c85db82c47d8e1713947fab8`.
The adapter command is the CPU command environment above, replacing the
unittest invocation with `python /tmp/fpgs-normal-block-cpu-oGrTia7y/qualify.py`
(add `--first8` for the prescribed initial sample).

All2048 pass hard finite, cone, nonnegative-normal-impulse and independently
reconstructed momentum checks. This does NOT mean all have converged: only613
meet the complete strict physical stop. There are1397 pointwise diagnostic
passes and651 component tradeoffs.1981 are closer to the saved held-H
reference and67 have genuine scaled H-distance regressions beyond3e-5.
Natural residual improves in1995. None of these categories is substituted
for another.

| Metric (median / p99 / maximum) | Original24 | Normal-block24 |
| --- | --- | --- |
| Scaled held-H velocity error | .00638992 / .0319958 / .0557385 | .000117915 / .00623906 / .0281947 |
| Scaled natural residual | .00259317 / .0203720 / .0396400 | .0000444159 / .00206515 / .00474159 |
| Negative normal velocity | .000491438 / .0145640 / .0343720 | .000200314 / .0124359 / .0303902 |
| Complementarity | .000886633 / .00973418 / .0443596 | .0000949662 / .00641532 / .0220370 |
| Disk MDP gap | .00209226 / .0354769 / .0794364 | .0000162729 / .00165115 / .0616380 |

The largest H-regression is GB/current/tier48/world168:
.0186851→.0281947 (excess .00950963). RTX held/tier32/world375 is
.00341094→.0113212; its current counterpart is .00434038→.0113234.
These are real finite-budget tails, not hidden by improved aggregate maxima.

### Complete counted work

| Work | Count |
| --- | ---: |
| Original actual passes / candidate committed passes | 49059 / 45474 |
| Candidate passes7–23 / exactly24 worlds | 552 / 1496 |
| Physical stops / permanent-half latches | 613 / 420 |
| First-entered columns / column products | 8116 / 4351860 |
| Factor builds / exact-ID reuses | 2460 / 43234 |
| Factor products / divisions / square roots | 28796 / 100026 / 19966 |
| Equality solves / solve products / divisions | 45694 / 557324 / 722644 |
| Pivot transitions: enters / drops | 18 / 202 |
| All-normal rechecks / rows checked | 45492 / 441438 |
| Normal-inner products / all-row normal publication products | 1787931 / 5323882 |
| Tangent disks / changed tangent rows | 439995 / 359484 |
| Tangent transpose / all-row response products | 6470712 / 23781978 |
| Full scores / rows visited / midpoint scores | 47942 / 1394092 / 420 |
| Full-score products / divisions / square roots | 7430824 / 1906298 / 2786884 |
| Rank rejects / guard failures / actual fallback calls | 0 / 0 / 0 |

All declared counters, curves, exact dropped/active IDs, final physical
metrics and fallback fields remain in the compressed artifact. None of the
actual saved cases exercises fallback: its state/budget contract is covered
by the separate regressions, not claimed from this cohort. Column count per
world is2–6;1954 worlds build only one normal factor. The maximum residual
carry discrepancy against final independent operator reconstruction is
1.42109e-13.

Counted major-product total is56,207,637 versus original84,268,494
(32,662,962 EX1 plus51,605,532 recurrence). It equals26.0938 original full
action-pair equivalents, versus the proposed10–12 feasibility gate. This
proxy includes the listed full-score scalar products but is not a complete
instruction count: divisions, square roots, reductions, branching, proposal
scalar operations and retained geometry/whitening/publication are not free.
The9,154,755 retained whitening triangular products are separately recorded.
No GPU execution, timing, occupancy or stall inference exists. The25.24s CPU
adapter elapsed time includes archive loading and independent qualification
and is not a native performance estimate.

| Partition | Committed passes | Full scores | Strict stops |
| --- | ---: | ---: | ---: |
| RTX1024 | 22701 | 23931 | 313 |
| GB1024 | 22773 | 24011 | 300 |
| Tier32,1304 worlds | 29397 | 30901 | 336 |
| Tier48,744 worlds | 16077 | 17041 | 277 |

### Causal trace, not a new policy

RTX/current/tier32/world1 keeps normal IDs `[6,9,12,15]` for all24 passes,
builds exactly one factor and never latches. Every pre-tangent normal endpoint
passes its LCP certificate. The following tangent action changes normal
residual by `G_nt * delta_t`, so the final normal law needs another outer
correction:

| Pass | Post-tangent negative normal velocity | Full merit / bound |
| --- | ---: | ---: |
| 1 | .145444 | 4848.15 |
| 4 | .0594034 | 1980.11 |
| 8 | .0233466 | 778.221 |
| 12 | .0108945 | 363.151 |
| 24 | .00140951 | 46.9837 |

Across2048 final merits,1795 are normal-dominated,175 complementarity,
62 natural, and16 MDP. Worst-tail GB/current48/world168 similarly keeps its
six normal IDs and one factor; it latches at4 and ends with MDP .0616380 and
normal defect .0303902. Thus the unresolved coupled friction/radius dynamics,
not dense-factor churn or fallback, prevents the hoped-for10–12-action work.

Root decision: **close CPU-only; no native prototype and no new policy/grid**.
The ideal fixed-point and aggregate quality improvements are retained as
scientific evidence, but do not satisfy the complete cost criterion. Generic
post-tangent overflow rollback is not separately implemented or qualified;
the finite saved cohort and its hard result checks do not justify broader
native admission. No production behavior is changed.
