# Rank-aware active contact owner: bounded CPU falsification

Funded at 02:52 UTC during the user's 23:44--09:44 UTC structural window.
Start from retained `ca0d427a`, not the unpromoted sleeping prototypes. Isaac
Lab, timestep, substeps, original iteration allowances and capacities remain
unchanged. This card funds a numerical CPU control, not native implementation
or a claimed speedup.

## Distinct complete boundary

Cold-screen current normal/limit residuals using predicted endpoint motion;
discover a feasible working set rather than assuming final active membership.
Materialize contact response vectors only for that working set, including
complete friction triples. Replace repeated PGS visits with a rank-aware
active-contact solve. Preserve every individual impulse, friction disk and
public force; a lower-dimensional endpoint wrench is not permission to discard
contact constraints. Ordinary unsupported/failure paths remain charged.

The candidate must address the prior failure directly: dependent contact
equations can produce enormous unbounded corrections which activate previously
feasible omitted normals. Use a rank-revealed physical response basis and
feasibility/event-limited corrections, with sliding directions represented on
their original friction disks. Do not rerun the closed damped natural-map/LU
method under a new name, add physical compliance, or tune a tolerance/history
grid. Retained CFM changes update denominators only; it is not a regularizer
in the physical residual. Circular non-associated friction is not an SPD cone
quadratic program.

## Evidence and complete cost

Source-matched G1 RTX contact/limit response plus GS costs 5.715755 ms of
15.685799 ms whole physics. A 10% whole saving requires the complete replacement
to fit 4.147175 ms, including screening, active-response formation, every
omitted-row check, state conversion, pivot/probe and fallback. A 1.5x whole
result from this boundary alone would require 0.487155 ms; current evidence
does not support that stretch target.

Saved full 16K final impulses have active dimension mean 4.64/4.66, median 4,
p95 12 and maximum 28/27 on RTX/GB. About 78% of positive-radius contacts lie
on their friction-disk boundary. These are NOT initial working sets, proven
stick/slip states or evidence of convergence: full-batch snapshots omit the
operator residual inputs. Do not use final impulses as an admission oracle.

Original sparse factor/inverse production is already present. Current endpoint
motion can be formed from existing screw ancestry without another whole-tree
scan. Nevertheless every proposed response needs held-response decoding,
endpoint-motion updates and all omitted normal/limit tests. These costs cannot
be hidden by counting only the small active matrix.

## Early falsification gate

Reuse all 16 existing current/held full-law operator cases, including RTX 7410
current/held and GB 4625 held, and the existing singular/slip/denominator-CFM
controls. Use the original physical diagnostics and maximum eight committed
corrections. Any ordinary fallback receives only the unconsumed original
allowance. Charge failed probes, construction and fallback separately; do not
substitute a count reduction for a native speedup.

First complete CPU result or explicit mathematical blocker by 03:45 UTC;
hard decision by 04:15 UTC without a new, recorded causal hypothesis. No new
benchmark framework, GPU capture or native implementation is funded merely
because a final active matrix is small. Native work requires physical quality
and a plausible complete-work result on these controls, followed by an early
whole-task screen using the existing paired driver.

Existing references: `/tmp/fpgs-g1-natural-transfer-control-20260915.md` and
its pinned replay script/results; full-batch impulse-only data at
`/tmp/fpgs-g1-paired-replay-run-wQfIcNQ1/gpu{0,1}/replay.npz`.
The inherited handoff and closed experiments remain in the retained ancestry.

## First CPU result: all-closing equality sets are not funded for native

The complete sixteen-case FP64 control passes the original physical defect,
cone, nonnegative-impulse and uncancelled-momentum gates. This is largely
fallback correctness, not successful numerical acceleration: six initially
open cases need no solve, four nonempty cases converge in six total committed
corrections, and six difficult cases immediately retain all eight original
sweeps. RTX world7410 current/held rejects an inconsistent rank-reduced
equality system. GB worlds4625 and8192 current/held obtain negative normal
directions at zero normal impulse and stop at a zero-length positivity event.

The counted implementation retains48 of78 original executed sweeps,
1508 of1927 original residual row dots and59 of69 original sliding roots.
It adds1580 full-operator row dots,414 diagnostic row dots,35 feasibility
probes and12 rank rebuilds. Total residual row dots increase to3502, about
82 percent above the original1927. Unique response rows decrease342/452,
but that saving does not establish a favorable complete replacement. Dense
CPU triangular/Gram actions and the CPU oracle's80-probe disk roots are
explicitly distinguished from native retained work.

Preserved source: `tools/fpgs_bench/active_wrench_control.py`, SHA256
`fcbb861769a2654a73d6e73f3666b9f998413c95e08034192e93a22126dd335d`.
Exact counted output: `/tmp/fpgs-active-wrench-counted16-final-20260916.log`,
SHA256 `bc1333f567ce0c908277b7d052865b3990baaf641fe733cea7aaa3f82513f1b4`.
No runtime source or accepted performance result changes.

## One funded causal correction: incremental processed-set driver

The specific formulation error is treating all initially closing normals as
final active equalities. Initially violated constraints may become inactive
when another contact changes the motion. A feasibility-limited equality
Newton method cannot simply drop such a row while also insisting every
omitted row remain feasible from the cold state.

The separate bounded CPU follow-up partitions constraints into processed
active, processed inactive and unprocessed sets. A newly admitted normal is
a driving impulse: solve compensating equations only for existing active
contacts and the physical friction modes, then pivot at the incoming residual
reaching zero, an old normal impulse reaching zero, or a friction-mode event.
Processed inactive rows must remain feasible; still-unprocessed violations
are allowed until admission. A one-at-a-time equality Newton solve is not an
implementation of this correction. Actual nonlinear polar reconstruction,
all probes/pivots, fallback and response work remain charged. No final active
set, artificial physical compliance, tolerance grid or expanded correction
budget is allowed.

First sixteen-case output or a precise mathematical blocker is due03:55 UTC;
the original04:15 decision checkpoint remains. Preserve the all-closing
control and result unchanged. This funds no native implementation or GPU job.

Reference distinction: [Baraff's original contact-force paper, sections4--6](https://www.cs.cmu.edu/~baraff/papers/sig94.pdf)
describes incremental driving and contact pivots, but its frictionless
rank/termination arguments have assumptions that these biased frictional
systems need not satisfy. Its three-dimensional static-friction construction
also freezes a boundary direction and permits a weaker opposition condition;
that is not a replacement for this experiment's antiparallel sliding law.
[MuJoCo's constraint-solver documentation](https://mujoco.readthedocs.io/en/3.6.0/computation/)
instead describes a regularized convex formulation used by Newton/CG. Those
convergence properties cannot be borrowed for an unregularized FPGS physical
residual merely by adopting similar matrix machinery.

## Processed-driver first result and causal review,03:34 UTC

The three new focused controls pass, and all three fail when bound to the
preserved all-closing method. Besides initially inconsistent/negative-force
sets, they cover a legitimate zero-wrench transfer that releases an old
normal before the new driver can make progress. The first saved sixteen-case
run passes13/16, not a numerical qualification. Exact output is
`/tmp/fpgs-active-driver-first16-20260916.log`.

RTX7410 current consumes all eight slots: four advances and four nonlinear
restorations. Its normal defect is0.005258 versus the original0, natural
defect0.000506 versus0.000172, and complementarity0.003909 versus0.000997.
RTX7410 held passes with three remaining-budget fallback sweeps. Both GB8192
epochs stop after four corrections but fail the raw-normal gate:0.000144
current and0.0000528 held. Their energy-scaled stopping test does not imply
the already-declared raw physical acceptance criterion.

The new method discovers54 response rows and adds61 in fallback, versus452
original rows. It uses39 committed slots,13 restorations,26 admissions,
40 active rebuilds and115 trial evaluations. Read-only callsite attribution
shows that75 extra trials are exactly three fixed24-probe incoming-normal
root brackets plus their final evaluations. There are NO protected-normal
or friction-cone bracket invocations in this run. The39 stop helpers also
repeat5697 row dots despite already having current residuals and cold scale.
Those are avoidable CPU-control costs, not mandatory native algorithm work.
After excluding only those identified duplications, residual work would be
2525 row dots including fallback versus1927 original dots, still31 percent
more; response construction is74.6 percent lower. Neither proxy establishes
native timing. Crucially, removing probes alone does not fix the exhausted
physical failure.

A single source-pinned, in-memory safety check is funded before any further
runtime work: require the early stop to satisfy baseline-independent raw
physical bounds already used by acceptance, and hand off a genuinely negative
incoming normal-response direction before committing it. The existing rank
scale defines negative, while a zero-response direction with a real finite
release remains admissible. RTX7410 current first encounters this negative
direction at slot7; handing off there leaves two original sweeps. No fresh
budget, physical regularization, reference-solution oracle or threshold grid
is allowed. Run the same three regressions and sixteen saved cases once.

## Safety result and hidden producer dependency

The one in-memory safety check passes the three regressions and15/16 saved
cases. Both GB8192 cases now consume five slots and pass. RTX7410 current
still fails after six corrections plus two original sweeps: natural defect
0.000227 versus0.000172 and normal defect0.001622 versus0, despite improved
complementarity. This is a physical miss under the fixed budget, not merely
an inefficient inner root search. Exact source substitutions and output:
`/tmp/fpgs-active-driver-safeguard16-memory-20260916.log`, SHA256
`689eae2e19739e1d9cf4b492aab8b11c5e26f50b0739f7be1eacbe29bf7b67e2`.
The preserved solve-function variant SHA256 is
`009a5fe4bd755b567ef4eb1ab6ede61dd4cc46f64293096aa8a9372b5649ddf5`.

Root's native-dataflow review finds another decisive dependency: the CPU
control consumes ALL supplied diagonals to compute its cold amplitude. In
production those diagonals are outputs of the very all-row response producer
being proposed for retirement. Normalized admission and natural-map checks
also demand diagonals, though not necessarily for every row. The reported
response-cache counts therefore are NOT achieved producer retirement. They
must not be converted to an expected speedup while that dependency survives.

One bounded contract-removal correction is funded, with first sixteen-case
output by04:05 and the unchanged04:15 decision checkpoint:

- Freeze a metadata/current-residual preflight cost heuristic: one slot per
  closing simple normal/limit and two per closing frictional contact. If its
  estimate exceeds the original eight-slot budget, use the original solver
  cold, before active-response work. This is post-hoc, NOT a convergence
  bound. It rejects both7410 epochs, not just the failing one, and leaves
  roughly69 percent of original residual-row transactions in the admitted
  selected cohort. Test unseen existing full-law data without retuning it.
- Lazily form/count every response diagonal actually needed by admission,
  active equations or a nonzero projected correction. A zero-impulse open
  normal and a zero-radius zero-tangent pair need no diagonal for their
  identically zero correction. Do not receive all-row scaling for free.
- Drop all-row cold-amplitude dependence. Require the weighted natural-map
  numerator itself to be at most1e-5: the existing diagnostic denominator is
  at least1, so this is conservative. Use the same fixed1e-5 active-equation
  accuracy and the already-required raw physical3e-5 bounds. Reuse current
  residuals rather than reevaluating the physical operator in the stop helper.
- Bound incoming-normal root refinement by that declared physical equation
  accuracy, rather than always spending24 probes. Keep genuine cone and
  processed-inactive feasibility guards, count every probe and preserve the
  negative-direction remaining-budget handoff and valid zero-wrench pivots.

This funds CPU code only. No native implementation or new capture is funded
by recomposing known successful cases or by the preflight routing percentages.

## Frozen dataflow correction and unseen falsification, 04:14 UTC

Driver SHA256 `4a72680bc222b3e404d7b9b72aa1e86955444511b4a2dd0aa299e9bbf936210d`
passes all ten focused CPU tests and the selected sixteen G1 cases. The two
root-owned dataflow regressions failed before implementation: an unused open
diagonal poisoned global scaling, and an over-budget closing set exhausted
eight admissions instead of taking the original cold route. Review also
removed an empty-set shortcut which incorrectly treated a tiny raw residual
as a zero weighted correction.

The corrected selected cohort uses 29 commits, 16 original fallback sweeps,
179 of 452 response/diagonal rows, 45 full residual scans, 29 probes and 26
rank solves. Stopping adds no operator applications. Both RTX 7410 epochs
take the frozen cold-budget route. Exact log:
`/tmp/fpgs-active-driver-dataflow16-final-20260916.log`, SHA256
`7429fb705218121016360a1b1d631811180028ce7d0fd8f5332b952090ac1fe5`.
All-row CPU `J @ v` scans remain explicit; this is not an implemented endpoint
residual path or a GPU speedup.

The source was then frozen before testing 2,048 previously unused ANYmal
cases, reconstructed from the existing full-law 512-world captures on both
cards and two consecutive steps. The kinetic-coordinate adapter uses saved
Z as J, identity L, zero predictor, and current incident plus original bias
as the right-hand side. The reference is a fresh cold-eight metric solve,
not the original ANYmal 24-iteration output. This is cross-model algorithm
falsification, not native ANYmal parity or performance qualification.

Result: **766 pass, 1,282 fail**. All 308 cold-budget bypasses pass. Of the
1,740 admitted cases, only 458 pass. Every failure exhausts the eight slots;
there is no remaining fallback allowance. All preserve finite state, cone
feasibility, nonnegative normal impulses and momentum, but physical residuals
fail: natural 1,253, normal 1,194, complementarity 423, dissipation 645
(overlapping counts). Failure is not confined to a boundary estimate:
one of 60 score-four cases, 181 of 364 score-six cases and 1,100 of 1,316
score-eight cases fail. No gate or tolerance was retuned.

Nonlinear restoration consumes 6,044 of the 10,256 failed-case commits.
In 1,023 failed cases, at least one initially closing normal is never selected
as driver. That is a trace fact, not a final negative-normal count: coupling
can separate an unselected row. Low physical active rank does not cure the
problem. There are no rank-inconsistency or negative-response exits in this
cohort. First failure, RTX step 1600 tier 32 world 1: three driver advances
and five restorations worsen natural residual from 0.002721 to 0.052610 and
normal defect from 0.027654 to 0.494790.

Exact frozen run: `/tmp/fpgs-active-driver-unseen2048-YY6giIc2/run.log`, SHA256
`6699289d43d09932d4e304922e50bfc525dddd3dbf4d9ca66a3b1963ec15f1e0`.
All 2,048 cases completed in 105.291 seconds, process exit zero, source and
input guards passing. The algorithm is **not qualified; no native funding**.

The native cost review independently limits the representation-only claim.
Current G1 RTX contact response 1.241589 ms, limit response 0.555381 ms and
metric GS 3.918784 ms total 5.715755 ms. Removing 1.5 ms requires the entire
replacement, including metadata, selection, response generation, residuals,
QR, decode and any fallback, to fit 4.215755 ms. Selected response retirement
alone cannot establish that. A direct lazy-PGS alternative is distinct from
the prior all-row packet and full-port Gram experiments, but introduces
demanded endpoint-response columns and updates. No large net cost advantage
is established, so that alternative is not funded either.

## Next bounded alternative: coupled local contact blocks

CPU-only checkpoint 04:45 UTC. Retain the original ordered eight sweeps, but
solve normal and both friction components together for one contact. This is
different from the previous sticking-only block and from the accepted scalar
normal followed by a two-dimensional metric disk. It introduces no global
working-set factorization. The hypothesis is fewer repeated corrective
passes, not faster bookkeeping or a promised speedup.

For local proximal matrix H and b = r - H lambda_old, eliminate the normal
equation into the two-dimensional Schur complement Q. The sliding candidate
has t(gamma) = -(Q + gamma I)^-1 beta and
n(gamma) = -(b_n + H_nt t(gamma)) / H_nn. Solve
norm(t) - mu n = 0 with a safeguarded scalar root; open and sticking cases
have direct solutions. Existing diagonal regularization belongs only to
the update, never to the physical residual. Unsafe local blocks retain the
original metric transaction. Fixed-eight physical quality, root work and
required global passes decide whether any native implementation is warranted.
