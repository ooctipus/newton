# Structural FPGS study: 2026-09-16

## Handoff status

**No accepted additional speedup over `ca0d427a`.** The user's ten-hour
window is 2026-09-15 23:44 through 2026-09-16 09:44 UTC. Numerical controls,
native implementations, paired whole timings and cause-directed corrections
below are research results, not a completed performance target. The accepted
runtime remains unchanged; no new MJWarp ratio or across-task gain is claimed.

The native body-basis producer loses about 5% physics throughput on both cards.
Paired-world GS loses about 9% RTX/3% GB; its one literal-mask correction also
loses. The original and corrected versions are retained separately in
`45491fa7` and `9f01e841`. Earlier local-block, parallel-block and numerical
controls have their own complete cost/convergence closures below. Do not
enable an experiment merely because its correctness controls pass.

All experiment switches are opt-in and default off. Independent final review
finds only 18 opt-in dispatch/attribute lines changed in existing solver
files relative to the accepted base; new runtime modules are otherwise
unreferenced. Isaac Lab remains `53ee6b44c2334341305dbdf385a3916c6b140799`
with no source edits, and the accepted baseline worktree remains clean at
`ca0d427af809571bb5501f644c1a6e03990cd2a8`. No parent dependency pointer,
substep, maximum iteration allowance or capacity was changed. No sudo was
used. Branch: `ooctipus/newton:ooctipus/fpgs-active-wrench-20260916`.

Final CPU rerun covers all 19 test modules added or changed since the base:
58 tests pass, four CUDA tests skip intentionally with CUDA hidden. The
body-basis and both paired-GS versions have their separate native checks on
RTX PRO6000 and GB300 recorded below. Full pre-commit passes. Runtime/test
sources and this report are in the branch; large captures, checked temporary
adapters and diagnostic artifacts remain at their pinned local paths.

## Initial candidate: rank-aware active contact owner

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

## Coupled-block checkpoint, 04:43 UTC

This is not new contact mathematics. The September 11 investigation already
implemented the same full Coulomb Schur equation and tested 4,069 Ant/Humanoid
states. Its preserved native helper is
`newton-fpgs-tree-operator-20260911/newton/_src/solvers/feather_pgs/contact_block.py`
(SHA256 `313534d6b603f287b4ea87240ea54b8a2fe02fedb53bf8d8ed4366685615eff5`).
That helper was not integrated or timed in a production owner; its Humanoid
tails were mixed. The present question is current-task convergence with the
same physical stopping rule, and replacement of ANYmal's complete all-pairs
majorizer plus parallel iteration boundary. It is not a second claim of
inventing the local root solver.

Frozen CPU control SHA256
`0fc44df283d5284763f7fd4889b4b8cc475b197bb6314714cf1952bd0a3d5295`:
six focused tests pass. The root-owned independent-contact and incoming-impulse
controls fail with the old one-pass metric transaction and pass with the
coupled transaction. The incoming-impulse test also checks uncancelled momentum;
it does not establish an incoming-state bug in the old solver.

Selected G1 fixed-eight physical comparison: **15/16 pass**. RTX world 7410
with the held factor fails natural residual (1.458e-4 versus 9.563e-5),
complementarity (9.078e-4 versus 2.829e-4), and dissipation
(5.799e-4 versus 1.330e-4). No local fallback occurs. A trace isolates the cause:
the final row-33 sticking block has immediate physical residual about 1e-9,
but subsequent contacts change it to (0.001560, -0.002011, 0.000394).
Thus accurate local roots do not cure global contact coupling. This blocks
G1 promotion; neither thresholds nor routing were retuned.

Original exact-stationary stopping totals 78 sweeps over these selected cases.
Applying the identical declared physical stop to the old metric method gives
58; the coupled method needs 36. The fair same-stop row-dot counts are
1,502 to 943 for solving and 2,516 to 1,599 for stopping, plus 2,193 coupled
setup products and 403 candidate root probes. Without the physical stop,
actual sweeps are 78 versus 76. These are CPU work counts, not GPU timing.
The legacy CPU metric helper's 80 bisection probes are explicitly excluded
from native savings claims: today's accepted native metric root is bounded
Newton, not that high-precision reference implementation.

Final sixteen-case log:
`/tmp/fpgs-coulomb-block-final16-20260916.log`, SHA256
`ae28acd5440d62595d0fbfcd25bd9b7f8ac1b628b94a2536fa15c5645cab1c69`.
ANYmal qualification now compares coupled24 and metric24, both with the same
physical stop, separately against the actual saved native24 output. This
population informed the self-coupling diagnosis, so it is cross-model
qualification, not a previously unseen population.

## ANYmal full cohort and bounded hybrid card, 05:01 UTC

The frozen coupled24 control passes 2,034/2,048 cases against saved native24;
same-stop metric24 passes 1,819. Coupled versus metric24 passes 2,030.
All fourteen native-reference failures exhaust 24 passes without satisfying
the physical stop. None is an erroneous early-stop admission or a finite,
cone, momentum or negative-normal-impulse failure. Some physical failures
are substantial, not roundoff: GB current world 504 dissipation rises from
0.026855 to 0.179779. No tolerance or route was changed to admit these cases.

Coupled sweeps have median 2, p90 5, p99 24, total 6,332; the same-stop metric
total is 42,143 with median 23. Coupled root probes total 132,109, maximum
805 per case; eight metric fallbacks occur in three cases. The 590.760-second
CPU run exits zero with input/source guards passing. Result:
`/tmp/fpgs-coupled24-qualification-Pab7PnYA/full_qualification.json`, SHA256
`876f6788cbea270158283a11d9a90733ce019c4e0a9a385ce5d7fdb362abdf14`.

Alternating full forward/reverse passes does not repair the G1 issue:
14/16 pass, both RTX 7410 epochs fail, and same-stop sweeps increase from
36 to 38. It moves the directional tail into negative normal residual.
This deterministic policy is closed without further order tuning.

**Revised native funding decision:** the median-two full-cohort result is a
credible large-cost hypothesis, despite unqualified tails. Build the smallest
default-off complete owner while checking a bounded hybrid independently.
Spend at most six of the original 24 passes on coupled blocks; if they do not
physically converge, continue the original scaled Jacobi/Nesterov method with
exactly the unspent allowance and current impulses. The usual remainder is
18, but an earlier exact-stationary exit must retain all actually unspent
iterations. No cold restart, extra passes, task-name or case-ID dispatch.

Keep row-parallel current geometry, restitution and held-factor whitening.
Remove the all-pairs majorizer and original iteration loop entirely for
physically stopped worlds; construct them only on continuation. Charge local
self blocks, roots, end-pass physical scans, synchronization, continuation,
and unchanged decode/publication. Preserve original ABI, row capacities,
phase fallback and unsupported configurations; add no Lab source change or
persistent queue. The measured RTX owner is 4.091680 ms within 9.388518 ms
whole physics. First milestone: replace that complete owner within
3.091680 ms and demonstrate at least 1 ms whole-physics saving, not infer it
from operation counts. Initial integrated checkpoint: 06:30 UTC; no promotion
without native physical checks and repeated paired whole timings.

## Hybrid closure and native cost diagnosis, 05:49 UTC

The fixed six-pass prefix plus original remaining-budget continuation does
not qualify: 2,012/2,048 pass. It rescues six of the fourteen original coupled
misses but introduces 28 others. All 36 failures use continuation; none is a
false physical stop. There are 151 continuations, 5,294 coupled passes and
2,708 parallel passes. An independent original-recurrence control confirms
correct incoming impulses, zero-impulse bias and remaining allowance; semantic
negative controls detect a fresh24 tail, cold reset and doubled incoming bias.
The failure is not an implementation of those three bugs.

A fixed merit guard (rollback on the first non-improving prefix pass, then
choose the lower-merit handoff/final-tail endpoint) passes 2,017/2,048. It
rescues 22 fixed-hybrid failures but introduces 17 others. All charges include
the 19 rejected attempts, rollback, endpoint projections and continuation.
This policy is closed; a lower scalar merit does not ensure every physical
component improves. Artifacts in
`/tmp/fpgs-coupled24-qualification-Pab7PnYA/`:

- `hybrid_qualification.json`, SHA256
  `0663601ccf5c2a45b5702dc21b88c95dc334497103da2449c78cd0e67ad39950`.
- `guarded_qualification.json`, SHA256
  `5a7d5d4fd4d2d12f85aec8aeeb7172b4e49fb581bbeefe54c6d75ddc3fb49432`.

The native default-off hybrid is frozen at source SHA256
`3e972df7b7e2659716718595a49164a3a7ed9f94c2d874c0e29855dca3384015`.
Root replay of the original 512-world saved inputs preserves captured192/1
strides, current geometry/held factors, read-only inputs, metadata and unowned
outputs. All hard finite/cone/momentum/nonnegative-impulse checks pass, but
physical nonworsening fails on 45 RTX and 27 GB cases. No promotion.

| Restored512-owner replay | RTX original | RTX hybrid | GB original | GB hybrid |
| --- | ---: | ---: | ---: | ---: |
| Tier32, current step, ms | 0.049915 | 0.213707 | 0.051487 | 0.231947 |
| Tier48, current step, ms | 0.054305 | 0.258812 | 0.056468 | 0.276488 |
| Tier32, held step, ms | 0.049776 | 0.217928 | 0.051312 | 0.227069 |
| Tier48, held step, ms | 0.053782 | 0.260418 | 0.056484 | 0.282307 |

These are complete saved-owner launches INCLUDING identical output-restoration
copies, not pure kernel or16K environment timings. Both arms alternate over
six rounds, with32 restores/launches per graph and16 graph executions per
sample. Register whitening is disabled, matching the accepted recipe. The
hybrid is 4.28--5.00 times slower in this limited replay, not a task speedup.
Logs: `/tmp/fpgs-coupled-native-paired512-20260916-MT8Jk2/gpu{0,1}-run03.log`.
Earlier logs are preserved: first missing import path, then runner allocation
lifetime loss during captured replay. The latter is fixed by retaining every
buffer, reset seed, argument list and kernel, not by modifying runtime physics.

An output-only counter clone strips byte-exactly to the frozen native source.
All2,048 outputs match uninstrumented physics byte-for-byte; read-only and
unowned controls pass. It uses more registers and IS NOT timed. Results:

- 1,714 worlds physically stop; another183 fail only the1e-10 cone check.
  Their sum exactly matches the CPU1,897 worlds stopping by six passes.
- The remaining151 worlds fail another physical component as well.
- Only ten local metric fallbacks occur in67,645 block calls; no admission,
  Schur, analytic-bracket, local-law or scalar-fallback storm explains cost.
- The nonlinear work remains serialized in lane zero; the original row-lane
  solver does not have this contact-by-contact execution dependency.

Diagnostic source SHA256
`fcd2fa22288e72cad0f93b9f27ed92879e9c4f44026879225d78edc7b43b6ee6`;
logs `/tmp/fpgs-coupled-native-counters-20260916-QcUc5V/gpu{0,1}-run02.log`.
Initial counter runs supplied the captured unused1x1 dummy instead of a private
world-count by192 bank and are invalid; corrected runs explicitly allocate the
documented diagnostic shape. No production allocation/capacity changes.

## Cause-directed correction: concurrent contact blocks

Frozen CPU Jacobi SHA256
`0ca4b931aec62f2ea7bc5587898e20c8ffd76473038fe5983f25d4fd5852ab00`:
all contact proposals read one common residual and impulse state, then one
transpose/forward operator pair commits their combined response. The resulting
residual is also the physical-stop input and next iteration's input. No second
full stopping scan, all-pairs Gram or original majorizer is needed.

The policy is frozen before qualification: relaxation starts at one and latches
permanently to one half on the first physical-merit increase, using the midpoint
of the just-computed proposal. No further halving or old global tail. Original
8/24 pass maxima remain unchanged. The same-state test fails when bound to the
ordered implementation, distinguishing real concurrent dataflow from renaming.

ANYmal: **2,030/2,048 pass; eighteen remain unqualified**. All eighteen latch
and exhaust24; there are no false physical stops or hard physical failures.
The policy rescues eleven ordered-coupled misses but introduces fifteen.
There are1,939 physical stops,159 latches,9,291 passes (median3,p90=10),
199,049 local root probes,10,615,824 operator products,437,832 self-block
setup products,294,884 stopping projections and5,491 midpoint projections.
These are incomplete arithmetic/work counts, not GPU costs. G1 at8 passes
passes only10/16; no G1 promotion or native specialization is funded.

Result `jacobi_qualification.json` in the same qualification directory,
SHA256 `f083139859e513c1a6f14ed0f33eca5349f6e89ce49d8aed86d944fc024a4411`.

Explicit precision alignment: the next FP32 native candidate uses cone3e-5
for both stopping and merit, matching the unchanged independent physical
allowance, instead of an FP64-scale1e-10 test. An in-memory CPU control with
this exact change has IDENTICAL failures, work counts and physical outputs;
it is not a rescue of the eighteen tails. Artifact
`jacobi_precision_qualification.json`, SHA256
`c2de15eee8323b3e8918ed4fcfe0e6c92f01bd2d7dccedd60611fe8b57b77e7c`.

Fund one concurrent native mapping as the cause-directed correction to the
measured serialization loss, despite the disclosed numerical misses. Keep
original producer/decode, ABI/capacities and local root; use normal-row lanes
for independent triple roots, a CTA join across triple boundaries, and two
matrix-free actions per pass. First source checkpoint06:35 UTC. Early complete
owner timing decides whether further convergence work can have a large payoff;
the old solver remains accepted and no new whole-task gain is claimed.

## Concurrent result and numerical interpretation, 06:20 UTC

The frozen native concurrent owner is `coupled_jacobi.py`, SHA256
`3da318644dad50d0f56955bd72ad042ab07ca21cfc7dc85d3b13f0098c75cf3d`.
It is selected only by `FEATHER_PGS_COUPLED_JACOBI=1`. Five focused CPU checks
and four offline architecture/tier compilations pass. Original producers,
decode and unsupported fallback remain; the ordered module stays frozen.

Saved512 replay remains slower: original/candidate RTX0.1544--0.2098x,
GB0.1757--0.1817x. Hard physical, metadata, unowned and read-only checks pass;
eleven RTX and nine GB states fail the old pointwise nonworsening comparisons.
Logs `/tmp/fpgs-jacobi-native-paired512-20260916-jrrZT5/gpu{0,1}.log`.
The same restore-copy charge and small-batch limitations apply as above.

One full-size discovery confirms that this is not just a512-world tail or
underfill artifact. Original unchanged driver, fixed Lab,16,384 environments,
200 warmup,40 wall/40 graph steps, original24/8 iteration allowances,
two substeps and decimation4; identical public capacities72 dense/32 MF,
raw212,992 and broad294,912. All four runs and eight boundaries pass the
finite/capacity/source checks; these do not prove numerical equivalence.

| ms/environment step | RTX original | RTX concurrent | GB original | GB concurrent |
| --- | ---: | ---: | ---: | ---: |
| Whole physics |9.408004|17.399814|9.624115|20.775221|
| Environment wall |17.286177|24.850491|17.749275|28.347603|

Whole physics original/candidate is0.5407x RTX/0.4632x GB. This is a measured
loss, not an accepted runtime regression: the candidate remains default-off.
Manifest `/tmp/fpgs-coupled-jacobi-whole-paired16k-20260916-01/manifest.json`;
root81722 reaped exit0 and both devices verified idle. No repeated timing of
this losing implementation is funded. A three-step node diagnosis and a
source-exact clock-only observer locate the cause separately from graph timing.

### Physical comparison must not become a stronger contract than requested

The pointwise comparison is not a complete ordering of physical accuracy.
Using already-saved, converged full-law references (no new production budget),
the permanent-latch CPU result is closer in held-mass velocity norm on
**all2,048 cases**. Scaled error uses
norm(L^T(v-v_ref))/(1+norm(L^T v_ref)); median improves0.00638992 to8.17796e-6,
p99 0.0319958 to0.00226889, maximum0.0557385 to0.0164769.
No case exceeds the original scaled error plus3e-5. Natural error improves
in2,047 cases; the remaining increase is1.09e-5, below the existing3e-5
absolute allowance. The eighteen original component failures remain recorded,
not silently relabeled as passes or automatically called physical bugs.

The eighteen also improve against the independently qualified semismooth
reference: candidate/original velocity-error ratio median0.18123, max0.62948.
This supports genuinely better overall velocity convergence despite component
tradeoffs. It does not prove uniqueness, long-horizon behavior or native
performance. Exact input/source/reference guards pass. Artifacts:

- `jacobi_reference_comparison.json`, SHA256
  `a47918f0ad3a4192a7abeb75a2d81d6b88e1239989ebd1dc8716e18c645ea460`.
- `jacobi_all_reference_comparison.json`, SHA256
  `6cee6f322321131158fb81860a7c4789f3519d80d445df8306d2c89979c77487`.

Local Jacobian diagnosis at passes23/24 finds contractive half-step maps in
all eighteen cases, final spectral radii0.525--0.964. Finite differences at
h/2h agree to1.60e-9 relative; an isolated sticking derivative agrees with
H^-1 CFM to1.32e-10. These are finite-budget slow modes, not a final local
half-step instability. A single cause-directed per-pass relaxation-recovery
policy improves old pointwise counts to2,041/2,048, but introduces **five
genuine reference-distance regressions**. It is therefore closed without
native work. This is direct evidence that optimizing the pointwise pass count
alone would select the wrong numerical policy. Artifact
`jacobi_recover_qualification.json`, SHA256
`2d2a505ead7ddb8e708810e4648affb284d8dba482f69ed0e047dd50ba217426`.

## Next bounded structural card: joint impulse/multiplier iteration

CPU-only funding at06:15; native funding remains conditional on measured
phase costs and numerical evidence. No warm-start parameter grid is funded.
The frozen full-root algorithm spends80% of its199,049 scalar evaluations
after the first global pass. Merely improving seven-probe roots to roughly
three evaluations would cut about44% of total probes, insufficient by itself
to promise the required complete-owner saving. Do not call that a2x result.

Instead remove the nested local convergence loop: retain contact multiplier
gamma as solver state alongside lambda and take one Newton multiplier action
per global pass. Evaluate the clamped current gamma once, derive its update
from that same2x2 inverse, and publish that current trial. Rebuild the analytic
upper bound from current bias; never reuse stale bracket signs. Exact open,
frictionless and gamma-zero sticking branches remain. Unsafe matrices,
nonfinite actions or nonnegative derivatives take the original local metric
transaction. An unfinished safe local solve is not mislabeled as unsafe.

Only inward coupled cone/normal repair is permitted; no outward repair.
Damp BOTH lambda and gamma with the same permanent-half policy. The resulting
global physical residual controls early stopping, with the same3e-5 FP32 cone
criterion and unchanged other bounds. Original8/24 maximum global passes,
current rows, held factors and two factor-space operator actions/pass remain. At a genuine
joint fixed point, proximal CFM times delta-lambda vanishes and the original
Coulomb law is recovered; this is not a convergence proof for finite passes.

The hypothesis is a substantially smaller, one-evaluation contact update with
less nested work and shorter native state lifetimes, not cheaper proof code.
Extra outer passes, multiplier state, repair, safeguards, physical projections
and unchanged producers/publication must all be charged. Success still means
at least1ms saved against accepted whole physics, not against17.40ms. First
CPU/source checkpoint06:45, hard07:00 without a new diagnosed cause. Freeze
before full2048 reference/physical checks; do not fund native if that fails
or measured remaining cost makes the complete-owner target implausible.

## Complete cost closure, 06:38 UTC

The source-matched three-step node audit includes all12 physics roots,
process-correlated kernels and memory, and interval overlap. Actual live
candidate keys end in `_ccj24` at72/32 capacities; each32/48 tier executes
eight times per environment step. Exclusive complete solver cost rises
**4.079856 to11.838243ms RTX**, **4.431955 to15.500263ms GB**. Retained work
is5.054821 to5.021173ms RTX and5.050209 to5.232479ms GB. Thus the large loss
is inside the replacement owner, not missing activation or collateral
collision work. RTX registers are122/129 versus86/80; GB124/128 versus80/80.
No local-memory spill is reported. These allocations do not establish
achieved occupancy or a hardware ceiling.

Node evidence:
`/tmp/fpgs-coupled-jacobi-nodes-paired16k-20260916-01/strict_audit02.json`.
It reuses the unchanged strict reader, SHA256
`e76d53f685871f8d3e38a221ce89485d3d87014b9c8af1f6ac22e10af8e74dc6`.
The initial audit command had a mistyped hash and exited before analysis;
the corrected02 output is the evidence, not its empty predecessor.

The output-only clock observer strips to the frozen native source. All2,048
physics outputs match byte-exactly on both cards; immutable inputs and unowned
counter rows pass. Actual9,291 passes match CPU, with1,939 physical stops and
no admission failures. Root probes199,061 versus CPU199,049 are separately
recorded, not assumed identical. Within-world clock fractions are:

| Phase | RTX | GB |
| --- | ---: | ---: |
| Original row staging plus local preparation |23.79%|24.06%|
| Concurrent proposals and join |48.67%|49.43%|
| Two matrix-free actions |3.11%|3.11%|
| Physical merit/stop and commit |17.55%|17.24%|
| Decode/publication |6.88%|6.16%|

These are sums of diagnostic within-world cycle windows, NOT GPU-wall
fractions or an Amdahl proof. The observer changes register allocation and
is not used for latency claims. It nonetheless falsifies the idea that almost
all work consists of repeated local root iterations. Its preparation and
merit work remain substantial, and the true whole owner needs roughly74%
retirement to meet the original one-millisecond saving milestone.
Source `/tmp/fpgs-jacobi-phase-uhgzy9Lf/jacobi_phase.py`, SHA256
`28cf624ff6956b0e8e08692123d4a45578c3a1eba590de6d4c09d67c67c363da`;
logs `/tmp/fpgs-jacobi-phase-paired512-20260916-FeC9sJ/gpu{0,1}.log`.
Root1092/51175 reaped0; all exact64-bit phase sums pass.

The joint-state CPU source freezes at
`40fe61ee36418748a5c49fceb9c32c198fafb6bbcff8d98e81e86b5402f1dba6`.
Five focused controls pass, including one-evaluation semantics, unfinished
feasibility, exact stick/slip fixed points and matched auxiliary-state damping.
Full2048:2,032 old pointwise passes; all2,048 remain closer to the converged
reference in held-mass velocity error, with no hard physical failure or unsafe
fallback. Local evaluations fall199,049 to63,510, but passes rise9,291 to15,056
(median6,p90=14), operator products10.616M to16.501M and physical projections
294,884 to458,358. G1 remains10/16 pointwise with63 versus62 passes.
Result `joint_state_qualification.json`, SHA256
`eb2ea3f2aef58d6e1a1a3c9e06137ac1a0d9fde5ebfa7498da267f218bb57502`.

**No native funding for this joint-state mapping.** It retains all current
row whitening, Schur/eigenvalue preparation, physical merit and decode, while
increasing the non-root iterative work. A lower register peak is possible but
unmeasured; it is not sufficient evidence for the required further acceleration
of all retained phases. No parameter or launch-grid sweep is funded.

Before choosing another representation, prior-art review also closes a blind
repeat of ANYmal leaf-first/root-last9-support whitening. It already replaced
canonical dense H/L, predictor, whitening, exact majorizer, recurrence and
decode with direct L117. Its contiguous-panel correction removed repeated
tag/membership walks but still lost9.295346 to11.759834ms RTX and9.508051
to11.425918ms GB. It was not merely a duplicate conversion layered on top of
the old producer. Exact closure and manifest remain in
`newton-fpgs-anymal-contiguous-response-20260914/reports/fpgs/ANYMAL_CONTIGUOUS_RESPONSE_20260914.md`
and `/tmp/fpgs-anymal-contiguous-live-paired16k-20260914-01/manifest.json`.

## Original-owner cost and reference-based reopening, 06:55 UTC

The output-only original-owner observer preserves all 2,048 native outputs
byte-exactly, immutable inputs and unowned rows. Its exact-limb phase sums pass.
Within-world cycle fractions (not GPU-wall fractions) are:

| Original phase | RTX | GB |
| --- | ---: | ---: |
| Current row staging, geometry and whitening | 31.325% | 33.976% |
| Exact all-pair majorizer and bias preparation | 10.563% | 10.540% |
| Original Nesterov loop | 48.588% | 46.444% |
| Decode/publication | 9.524% | 9.040% |

Mean executed passes are 23.96289/23.94629 of the unchanged maximum 24.
Source `/tmp/fpgs-original-any-phase-WN0VI0hM/original_phase.py`, SHA256
`5f6834e9259bdbfc9d3812953f8e00dc915dec62eed37122ba935183b7de93ae`;
logs `/tmp/fpgs-original-phase-paired512-20260916-An6F0B/gpu{0,1}.log`.
Both GPU sessions exited zero. These measurements do not support an EX1-only
structural experiment: its operation count overstates its cost ownership.

The reference-distance finding justified reopening the exact older AA4 CPU
result, not rerunning or modifying that algorithm. Its saved impulses improve
held-H reference distance in 2,046/2,048 cases, including 14 of the original
15 pointwise misses. Two genuine scaled-error regressions remain: GB current
tier48 world168 rises 0.0186851 to 0.0441590, and world494 rises 0.0101604 to
0.0103660. The latter passed the original component gates. All original
2,033/15 labels remain recorded. Median error improves 0.00638992 to
0.0000121666; p99 improves 0.0319958 to 0.0000933164. This is not an
all-cases accuracy win like permanent-latch Jacobi.

AA4 was CPU-only, not a native measured loss. The exact source is
`/tmp/fpgs-anymal-aa4-natural-wcfB7tcL/control.py`; the new reconstruction uses
existing saved candidate impulses and converged references, without new
candidate or reference solves. Artifact
`/tmp/fpgs-anymal-aa4-reference-9DWofRRJ/reference_comparison.json`, SHA256
`2444ec35de66f91121f365c153aaf32693c56bb8576af77b164598f1e6b6c0de`.

Cost review also rejects describing AA4 as cheap history attached to the
original EX1 recurrence. It uses its own diagonal-normal/spectral-tangent
natural map; original EX1 is built only for nine fallback cases. Literal work
averages 33.374 residual evaluations/world with 30,089 small solves. Exact
accepted-map carry, not implemented, could reduce evaluations to 17.7075/world;
its modeled 55.327M products already charge history and fallback. Retaining
EX1 up front adds 32.478M products, exceeding the original 84.268M before
additional projections, history traffic, reductions and tiny solves. There
is no credible one-millisecond complete-owner case for that combination.
No native AA4 implementation or history-size grid is funded.

The terrain planar-patch screen found real whole-footprint coverage, not
whole-mesh coverage: 22.76%/22.96% of saved post-height-cull triangles belong
to wholly contained, bit-equal horizontal patches. This predates current
geometric/finite-path filtering and is not query-time-weighted evidence.
Current finite-query cost alone is below the 1.57ms milestone; retiring that
milestone from all queries would require about 72% time-weighted coverage
before metadata and replacement-plane cost. Current-height mutation also
requires ordered invalidation or charged metadata rebuilding. No such
funding case was established; no new terrain implementation was started.

Next bounded checks target repeated original iteration work: actual temporal
warm-start support and prior art; physical stopping using an already available
residual; and whether the multi-warp owner has a previously untested way to
remove synchronization. These are independent read-only/CPU screens, not
permission for a parameter or layout grid. Preserve current producers unless
their replacement is the explicit, costed hypothesis.

## Cheap natural-map recurrence: bounded CPU card, 07:00 UTC

CPU feasibility was authorized before implementation. Retain original current
rows, held factors, one pair of factor-space actions per pass, adaptive restart
and maximum 24 passes. Use AA4's diagonal normal step and common spectral
tangent step, but no history solve, Armijo search, contact root or all-pair
majorizer. Common tangent scaling has the correct radial Coulomb fixed point;
the inherited unequal tangent steps generally do not. This is a previously
documented solver-parity limitation, not a newly discovered contact bug.

Recover the current feasible iterate's residual from the already-computed
extrapolated residual: if y=(1+beta)x-beta*x_previous, then
r(x)=(r(y)+beta*r(x_previous))/(1+beta). Bias is included on both sides.
Store one residual vector and the beta that actually formed y. Check the
existing AA4 physical bounds on current x before proposing another update;
no extra operator is donated to the 24-pass allowance. The previous residual
error multiplier is below one half, but fresh FP32 cancellation still needs
independent physical checks. The original impulse-change exit is separately
recorded, not treated as a physical certificate.

The cost hypothesis is removal of quadratic majorizer preparation plus many
of the original nearly 24 passes, without nested solves or history products.
Added residual carry, physical projection/vote, local spectral preparation
and unchanged row producers/decode must be charged. The same at-least-1ms
whole-physics milestone applies; majorizer removal alone cannot fund it.

There is a known general stability counterexample: three duplicate normal
rows with unit coupling and diagonal step one cycle even without momentum.
Original adaptive restart is not a substitute for AA4's globalization. Thus
this is a bounded feasibility test, not a convergence claim or promotion
proposal. Include the counterexample, residual-history identity and isolated
anisotropic sliding in regression-first controls. Start with eight saved
cases; expand only if useful. Stop on instability before funding native work;
no damping, step-size or line-search grid follows. First CPU checkpoint within
25 minutes; root continues owning all GPU execution.

### Warm-start and natural-map falsification, 07:08 UTC

Same-collision-generation warm starting was a real, previously unmeasured
ANYmal policy, not the dormant nearest-point WR warm cache. The saved adjacent
1600/1601 calls share all raw contact payloads and held factors, but thousands
of dense slots move. Remapping by unchanged raw contact identity carries all
29,878 contact rows while keeping 15 limits cold and the residual reference
at zero. Geometry and response are still rebuilt from current inputs.

Across 1,024 donor/receiver transitions, sweeps fall only 24,531 to 23,797
(2.99%), and 855 cases still use 24. There are 811 original pointwise passes
and 105 genuine held-H reference regressions beyond the existing allowance.
This does not support a structural gain or native implementation. Artifact
`/tmp/fpgs-anymal-same-generation-warm-oBqhqJ/SUMMARY.md`, SHA256
`399f17f8a6b32068941cdec13c6c36feb619ef3ab0a7075aebe0255f29dfad3a`.
The identical CPU run was repeated once solely because its initial stdout
exceeded the tool limit; the compressed full result and both source guards
are preserved. No different policy or runtime change was tested.

The new natural-Nesterov screen fails all eight preselected cases: no physical
stops, all 192 passes consumed, and every held-H reference distance worsens
(10.77--183.28x). Residual carry agrees with the direct operator to 3.55e-15;
this is not a carry bug. All eight have zero adaptive restarts despite large
oscillation. The duplicate-normal test reproduces the predicted instability:
local diagonal/spectral scaling lacks a global coupling bound. No full cohort
or native implementation is funded. Frozen helper SHA256
`a87ab5b6716846516d81799bd42cf66b97b9e0b2fd70411b42674e80a4363613`;
three regression-first tests and targeted pre-commit pass.

### One cause-directed correction: sequential spectral contact updates

Fund a bounded CPU control that removes the simultaneous coupling error by
using original contact-order Gauss-Seidel updates. Update and apply a normal
first; then evaluate both tangential residuals at that changed velocity,
take one common spectral gradient step, project the pair onto the current
normal's disk, and apply both tangential changes together. It is not the
old metric disk root or the expensive coupled three-dimensional root. It
has the correct Coulomb fixed point, but no finite-budget convergence claim.

Keep the 24/8 original ANYmal/G1 maximum allowances, current rows and held
response. Remove all-pair EX1, nested roots, AA history and momentum; charge
ordered residual/update transactions, local tangent spectral setup and the
initially explicit end-sweep physical scan. The measured paired owner target
remains at least 1ms below accepted whole physics, not below a slow prototype.
Root-owned native work is conditional on both physical quality and a credible
complete cost: ordered synchronization may erase the arithmetic saving.

Check the same eight ANYmal cases plus existing sixteen G1 cases first, then
only expand a frozen promising implementation to the existing 2,048-reference
cohort. Regression-first controls must include duplicate normals and
anisotropic sliding. No contact-order, damping or step-size grid is funded;
first checkpoint within 25 minutes. This is the single ordering correction
to the diagnosed simultaneous-map instability, not evidence of performance.

### Sequential correction: full-cohort result, 07:17 UTC

The eight-case screen passed, but the frozen 2,048-case qualification did not:
1,554 pass the original pointwise gates; 2,035 improve held-H reference
distance and 13 genuinely regress beyond original plus 3e-5. All thirteen
are tier32 cases consuming the complete 24-pass allowance. These are not
merely stricter residual gates rejecting a more accurate solution.

Total passes fall 49,059 to 44,632 (9.02%), with median 24 and 1,372 cases
still using 24. There are 739 physical stops, including 63 on the last pass.
The sixteen-case G1 screen passes eleven, not sixteen. There are no hard
cone, impulse-sign, momentum or nonfinite failures. No native implementation
or ordering/step-size grid is funded for this policy.

Counted products are 57.506M versus original EX1 plus recurrence 84.268M,
including 1,293,603 additional stop row dots. This is not a timing estimate:
ordered updates serialize dependencies, and current geometry, whitening and
publication remain. The original producer function is unchanged.

Frozen helper `a39eb83e4f5b240a24416f96683df0cc5e7453c8f180cbf539f9d752141256c7`;
three regression-first tests and targeted pre-commit pass. Full evidence:
`/tmp/fpgs-anymal-spectral-gs-qualification-6Nxtdj/README.md`, SHA256
`1a502cfaab3ef47520b8f6c33cb210b1766ff23672ac885870766bd0eae834b6`.
All 2,048 existing reference payloads and source/input guards pass. The CPU
cohort took 23.39 seconds; no new reference solve or GPU job was needed.

## Fixed augmented operator: bounded PADMM card, 07:35 UTC

The distinct hypothesis is amortizing one augmented-operator factor over
closed-form cone/dual updates, instead of nested contact roots or rebuilding
an active Jacobian. Use the dual-corrected updates in
[Carpentier et al., Algorithm 1](https://arxiv.org/html/2405.17020v1), and
reference this checkout's Kamino preconditioner and PADMM implementation.
This is not a claim that Kamino's default maximum 200 converges in our 24/8.

CPU policy: cold start, fixed rho=1, proximal eta=1e-5, no acceleration,
unchanged maximum 24 ANYmal / 8 G1 updates. Use one scale per contact triple,
`S=clamp(1/sqrt(max physical diagonal + float32 epsilon),.02,50)`; scalar
unilateral rows use their own diagonal. Let `U=S Z`, `b'=S b`, `lambda=S y`.
Physical compliance remains zero: inherited CFM is not added to the operator.
Factor `K=(rho+eta)I+U^T U` once and use Woodbury. The De Saxce correction
uses the previous dual variable, not the unconstrained physical residual.
Publish the projected impulse and its actual momentum-consistent velocity;
the unconverged dual velocity is not a physical certificate.

Cheap normalized primal/dual infinity norms at 1e-6 only trigger a separately
charged original-law physical scan. The actual physical gate must pass to
stop; final diagnostics are counted separately. Include independent cone,
Woodbury, fixed-point, and update-order controls before the eight-case ANYmal
and sixteen-case G1 screen. No parameter grid is authorized.

Prior `/tmp/fpgs-desaxce-splitting-XMQOPf` already tested once-factored
Woodbury with isotropic contact scaling on Allegro. Its lagged Douglas--Rachford
trajectory failed, reaching only 300/2,048 original pointwise passes at 24.
Thus Woodbury is not new; dual-corrected PADMM is the bounded new hypothesis.
The prior result was CPU-only and cannot be described as a native timing loss.

For D=18, setup costs 171N lower-Gram products plus a small factor. Each
ordinary Woodbury application costs twice N times D plus 306 triangular products and
36 divisions, before correction/projection/checks. An alternative native
representation, not implemented, forms `W=U C^-T` once for `K=C C^T`:
the application becomes `(h-W(W^T h))/(rho+eta)`, removing per-pass dependent
triangular solves. It adds 153N products and 18N divisions during setup.
It is a latency trade, not automatically fewer products. Correct publication
requires `U^T y=C(W^T y)`; any physical scan and retained producers are charged.
Native funding requires useful accuracy and a complete-owner saving case,
not just the phrase linear in contacts.

### Fixed and accelerated PADMM: diagnosis, not native funding

Frozen fixed helper SHA256
`71ef26bde2eefa3a32edbaf017dd9122689df7c62f17113d5646c04a80e6717b`
passes four regression-first controls and independent actual-source review.
It passes zero of eight ANYmal pointwise cases, improves five held-H reference
distances, and consumes all 192 allowed updates without a cheap or physical
stop. G1 passes only the six initially zero-solution cases of sixteen and
consumes all 128 updates. No hard cone or momentum failure occurs. Screen:
`/tmp/fpgs-padmm-screen-71ef26bd.json`, SHA256
`c58803ef3f7a90e9550cefece39519ff707419a30eafc1831cf8ddcb580b8508`.

The first prescribed failing case, RTX current world1, was diagnosed without
changing its policy. At update24 the scaled physical-minus-dual velocity
mismatch is 0.00440378. The exact decomposition is dominated by the negative
dual-update term (0.00444623), not the consensus response (0.00020146) or
changing correction (0.00019776). The augmented factor has condition 4.986;
shifted-linear backward error stays below 2.55e-16 and independent cone/dual
identities below 5.44e-16. This is slow outer convergence, not a poor linear
solve or wrong fixed point. One diagnostic-only continuation reaches the
physical bounds at update98 and the actual cheap-plus-physical stop at117.
Neither expanded count is admitted as a performance candidate. Artifact:
`/tmp/fpgs-padmm-failure-trace-LiY1bJ/result.json`, SHA256
`49c6d0898eb626077197c47ad124fca9257faa4c9827531b7ad6459237fcb8d2`.

One cause-directed correction then implements the actual Kamino accelerated
policy, not a momentum-parameter grid. It retains the fixed factor, 24/8
budgets, cone law and actual physical publication. Accelerated dual/impulse
hats enter the update; the proximal impulse remains unaccelerated. On restart
the hats rewind to previous unaccelerated states, while current states remain
committed, matching Kamino's source. Three regression-first dense-oracle tests
and independent source-order review pass. The extra merit/history operations
are counted; no additional response action or factor is hidden.

Frozen accelerated helper SHA256
`d00bfff57e4e4a2c81495edc78fd45c67848360a976421c0e89a1484b2799782`
still passes zero of eight ANYmal pointwise cases, improves six reference
distances and consumes all 192 updates, with 55 restarts. G1 again passes only
the six zero-solution cases and consumes all 128 updates, with 14 restarts.
There are no physical stops. Screen:
`/tmp/fpgs-accelerated-padmm-screen-d00bfff5.json`, SHA256
`95aa2982363e5a36c687f081111a359600334dbf792c6b4766173aa0b5f20735`.
Neither variant funds a full cohort or native implementation. No accepted
physics improvement follows from these CPU experiments.

A separate primary-source screen of
[Song et al.'s FBF architecture](https://arxiv.org/html/2607.19599v1)
does not supply a short-budget replacement: its reported contact-rich runs
allow 200 outer updates with 10 or 30 inner sweeps, and its matrix-free mass
application is for independent rigid bodies. Its timing claims are not
evidence for this articulated 24/8-update workload. No FBF prototype is funded.

## Materialized body-basis rows: one native experiment, 08:00 UTC

Preserve the original solver and change only G1 sparse contact-row production.
This is an extension of the existing Allegro body-map representation, not a
new friction law or a rerun of the closed G1 present-port solver. Keep held W,
current S, original materialized sparse Z, incident velocity, diagonals and
all per-contact impulses. Do not construct a port Gram or alter eight-sweep
GS, metric tangent roots, limit prefixes, restitution or publication.

The population screen rejects an ANYmal per-body cache justification: across
2,048 saved cases, 17,936 of 18,944 responding body instances have one contact.
A proposed five-mask in-CTA cache covers only 47.40% of contact directions and
no tier48 worlds. Root/common-prefix reuse is mathematically valid but adds
an inverse and current maps; no large complete-saving case was established.
Its within-world cycle fractions must not be called attributed GPU wall time.

G1 has more reuse. Full saved RTX data contain 77,662 normal contacts and
209,424 contact directions across 24,865 present normal-template/world pairs:
3.123 contacts per template, not the selected sixteen cases' 7.56 per body.
GB gives 3.144 contacts/template. The full archive lacks raw endpoints, so
these are template counts, not a fabricated body/static census. Independently,
the existing admitted SparsePlan proves three responsive endpoint classes.

The replacement uses that proved plan bound. Extend the existing metadata
writer with a raw-contact key in each normal row's support slot. A 128-thread
world CTA consumes all current normal keys before replacing them with original
support templates. Shared current body maps contract the existing held W once
per present endpoint class. Four lanes per contact produce adjacent kinetic
coefficients, with original padded tails zeroed. Current incident motion must
use the predictor, not the public pre-predictor body velocity. Two-endpoint
contacts use the original pair-support union and signs. Uniform per-world
fallback retains complete direct reconstruction; no capacity truncation or
additional global matrix/dispatch is authorized.

This targets repeated warp-uniform geometry, W loads and contact-local
shuffles as well as response arithmetic. Four-lane grouping is fixed before
measurement to limit strided output stores; it is not a tile-size grid. Shared
body maps die at the end of the producer, not after GS. The earlier measured
present-port loss included a new Gram and solver; the earlier packet loss
retained a large shared row cache throughout GS. Neither is the same boundary.

Original contact production costs 1.241589ms RTX in the retained source-bound
node attribution. This leaves little margin: a 0.7ms complete saving would
require replacement plus added metadata cost below 0.541589ms. Arithmetic
alone does not establish that. One native falsification is funded to test
the source-visible execution advantage; the one-millisecond whole-physics
acceptance milestone is unchanged. No gain or promotion is implied by funding.

Default-off `FEATHER_PGS_BODY_BASIS_ROWS=1`; unsupported constructors retain
the original owner. First missing-module test failed before implementation.
Reuse existing current/held raw-contact fixtures and independent physical
rows, then root-owned paired native checks and the existing full16K 200/40/40
whole-step driver. First source/AOT checkpoint 08:40 UTC; window ends 09:44.
Freeze sources before GPU work. One measured loss receives cause attribution,
not another layout grid. Isaac Lab and all numerical/capacity settings remain
unchanged.

### Body-basis producer: paired loss and closure, 08:35 UTC

Frozen runtime `body_basis_rows.py` SHA256
`2eed3b6241200ae357911eb9b13e45a6926a2bd5eab2d5e611ce16829b9afaa0`
passes three CPU controls and both native selectors on each GPU. The native
checks include held factors, changed contact positions, forced complete
fallback, same-articulation dynamic pairs, graph empty/regrow and sixteen
saved current/held cases. The measured test SHA256 is
`63e53003413fb76bd97ffe9155b50240975334b207a760f25ffbd750b96e1efc`;
its later docstring clarification changes no test mathematics.

Qualification scope is important: fixture coefficients/diagonal/incident
are independently checked. The saved sixteen-case recurrence oracle consumes
candidate-produced rows; independent cone and momentum gates pass, while
natural/normal/complementarity/dissipation defects are reported, not bounded
against an independently rebuilt original solve. This is not a claim of
original finite-eight convergence parity. No promotion follows from these
checks. A same-input original/independent physical comparator would still be
required for a successful candidate; the timing loss does not fund that
additional qualification campaign.

Offline compilation for both SM120 and SM103 reports 64 registers/thread,
4,116 bytes shared, zero stack and zero spills. Native resources agree.
Original contact production used 56 registers, 128 shared bytes and a
32-thread block; the replacement uses 128 threads. No hardware-counter or
achieved-occupancy claim follows from these compiler/launch observations.

| One complete 16K discovery pair | RTX PRO6000 | GB300 |
| --- | ---: | ---: |
| Accepted baseline physics | 15.626304 ms | 20.439429 ms |
| Body-basis physics | 16.448546 ms | 21.517904 ms |
| Baseline / candidate | 0.950011x | 0.949880x |
| Baseline environment wall | 29.757961 ms | 34.777078 ms |
| Candidate environment wall | 30.623391 ms | 35.502077 ms |

All four children, eight capacity boundaries and final source/idle guards
pass. Root session95636 exits0. Baseline is accepted `ca0d427a`, not the
previous losing port implementation. All collision flags, 200 warmup,
40 wall/40 profile steps, eight solver sweeps, substeps and capacities remain
fixed. The original manifest deliberately records physical/performance
acceptance false. Artifact
`/tmp/fpgs-g1-body-basis-whole-paired16k-20260916-01/manifest.json`, SHA256
`fc3ad2d7b3d61c15fdb9fe1f18e966dc74618ea3fac06efddc5df460f1943b06`.

One three-step node diagnosis identifies the complete changed family:

| Exclusive GPU owner, ms/env step | RTX old -> new | GB old -> new |
| --- | ---: | ---: |
| Contact producer | 1.253388 -> 2.085837 | 1.324139 -> 2.401643 |
| Contact metadata | 0.263851 -> 0.285974 | 0.192437 -> 0.202219 |
| Complete changed family | 1.517239 -> 2.371811 | 1.517589 -> 2.604885 |
| Retained GS | 3.907990 -> 4.000164 | 4.344299 -> 4.395978 |
| Retained dynamics | 2.724823 -> 2.723992 | 2.409245 -> 2.407879 |
| Retained collision | 3.777175 -> 3.733091 | 8.272319 -> 8.277855 |

These are preserved process-correlated interval/exclusive measurements, not
component sums presented as whole-step savings. Every old metadata/producer
call disappears in the candidate, and each new owner executes eight times.
The loss belongs to the replacement producer, not a duplicated old pipeline.
All four strict audits retain twelve physics and three separate auxiliary
roots, all memory nodes and actual resource records. No source/capacity guard
fails. The old Lab node analyzer rejects auxiliary graphs as expected; its
two parent failures are preserved, not relabeled as successful throughput
runs. Candidate nodes are in a directory labeled `round_01_baseline` because
the unchanged failing parent was invoked separately with the candidate source
and flag as its first arm; actual source/owner checks identify it explicitly.

Strict reports: `/tmp/fpgs-g1-body-basis-strict-1ZgOqX94/`.
`baseline_gpu0.json` SHA256
`0d05d7639c34120826e5ffdc0e7bf877f4084296bc7ea1b16a856ee3aeaf9ecb`;
`baseline_gpu1.json`
`6a0518c1ff4ddbf9a334ada33e9df2784acb85059212170c7d30f23be70f67c3`;
`candidate_gpu0.json`
`60e85aa4f6fefbfed8abb49429caedf32884dc92c113adaa800816971ccda3d6`;
`candidate_gpu1.json`
`6fa50c46534e56fd3aafb7113259bb924e519d2922ec51c87942b8a6eba6321d`.

The source work model explains why theoretical reuse alone was insufficient.
For a static contact with twelve support coefficients, original whitening
uses 234 products/contact; a six-component body map costs 468/body plus
216/contact for contraction. That portion alone amortizes only above 26
contacts/body. Removing current-J construction gives a more favorable but
still modest complete arithmetic estimate, before key discovery, shared
traffic and synchronization. Full captured data average 5.2386 normals per
nonempty world; most of a 128-thread CTA can be inactive during four-lane
contact emission. Those are source/population explanations, not an isolated
measurement of stalls. This implementation is closed, default off, without
a tile-size or admission-threshold grid. No accepted physics gain results.

## Fixed paired-world GS ownership: bounded decision, 08:41 UTC

Fund one execution-ownership experiment, not another numerical law or a
mapping grid. The retained original G1 metric GS costs 3.907990ms RTX in the
new strict attribution. Its twelve-to-eighteen support coefficients and
lane-zero scalar roots leave substantial warp lanes idle. Two independent
sixteen-lane worlds in a single warp can execute those serial roots together
and shorten reductions. This is a plausible utilization opportunity, not
proof of a two-times kernel gain. It changes no required arithmetic, contact
law, order, early-exit rule, factor lifetime or eight-sweep allowance.

Keep the accepted original contact producer; body-basis rows are disabled.
Retain all eighteen coefficients: lanes zero and one additionally own the
16/17 tails in residuals, self-cross terms, finite checks, velocity updates
and scalar sibling fallback. Decode all 43 coordinates. Shared velocity,
impulse, tangent-cross and readiness storage must have disjoint world banks;
bitmaps still use row/32, not row/16. Every shuffle, vote and synchronization
uses the correct half-warp mask and shuffle width. Odd final packs return
before indexing their nonexistent group. No block-wide barrier or new global
scratch is permitted. Group/articulation/world remapping remains explicit.

Default off: `FEATHER_PGS_SPARSE_PAIRED_GS=1`. The missing-module regression
fails before implementation. Reuse existing native law and physical fixtures,
adding a five-world original-versus-paired comparison with permuted mappings,
both halves active, empty/regrown rows, support tails and captured replay.
Unlike the body-basis saved-row oracle, this comparator uses the unchanged
original solver on identical input buffers. Only reduction ordering may differ.

First source/native checkpoint 09:05 UTC; session ends 09:44. A meaningful
complete-physics target remains approximately one millisecond RTX, including
the original producer, all tails and publication. Use the existing paired16K
200/40/40 driver immediately after bounded native checks. A loss receives one
cause diagnosis; do not try eight/four-lane variants or change the root solver.
No Isaac Lab, capacities, parent pointer or accepted-default change is funded.

### Paired-world original mapping: measured regression, 09:00 UTC

Runtime SHA256 `ca0e74a8858c8fa9ce13b3ade2010319b24abd3df2f340d0f5d4a383a58d5ae0`
and test SHA256 `9cd9e449b4128dc6e4d61071c427d156c7561232da46a922a742a5a297b22f5e`
pass both native selectors on both cards. These compare the original solver
on identical saved rows, plus five permuted worlds with support tails,
different half-warp exits, scalar fallback, graph withdrawal/regrowth and an
odd final world. CPU controls and independent source/AOT review also pass.

| Complete 16K discovery pair | RTX PRO6000 | GB300 |
| --- | ---: | ---: |
| Accepted baseline physics | 15.619313 ms | 20.441393 ms |
| Paired-world physics | 17.163453 ms | 20.977020 ms |
| Baseline / candidate | 0.910033x | 0.974466x |
| Baseline environment wall | 30.138453 ms | 35.299111 ms |
| Candidate environment wall | 31.946619 ms | 34.174200 ms |

All source/capacity/idle checks pass; root63319 exits0. The unchanged checked
driver uses accepted `ca0d427a`, original contact production, the same kinetic
and collision paths, 200/40/40 steps and unchanged numerical budgets.
Manifest `/tmp/fpgs-g1-paired-gs-whole-paired16k-20260916-01/manifest.json`
SHA256 `f1f52e60a81f81b663caadcb7899850037fd6a2808efd43e05338e569d671d73`.

One strict three-step node attribution locates the regression in GS:

| Exclusive GPU owner, ms/env step | RTX old -> new | GB old -> new |
| --- | ---: | ---: |
| GS | 3.933331 -> 5.387362 | 4.350613 -> 4.853354 |
| Complete rows | 2.503861 -> 2.509356 | 2.551466 -> 2.560970 |
| Dynamics | 2.722474 -> 2.727532 | 2.410344 -> 2.408157 |
| Collision | 3.710943 -> 3.723117 | 8.375347 -> 8.413361 |
| Publication | 2.326730 -> 2.335862 | 2.495914 -> 2.498165 |

The only kernel-family substitution is original metric GS to paired GS.
All source/file/capacity guards pass, with twelve physics and three auxiliary
roots preserved. Sampled trajectories differ slightly; this is not a
byte-identical workload claim. Parents94434/52238 retain their expected
exit1 from the old Lab auxiliary-graph analyzer. Strict reports are in
`/tmp/fpgs-g1-paired-gs-strict-HXr2FSz7/`: baseline GPU0/GPU1 SHA256
`974a31415ac437029e03b8858074621b7c0175bc29a95b091ce6c6ef20b198a0` /
`bbca5acaf967a43753d2683fe3cca7a547e11196d4d917f8f5a1c9f081f910a2`;
candidate GPU0/GPU1
`52196ad003dad6318789c3598dca3c5229ad706d33445a8a560f6640a4715c60` /
`61fe255aaea165d36a4846043aae56bd122fd5c767bf9923ad34dfb4059c9124`.

Actual resource records show 72 -> 85 registers/thread, 1116 -> 2104 shared
bytes and 16384 -> 8192 blocks, each with 32 threads. Per-world registers and
shared memory actually decrease; these counts alone do not prove an
occupancy loss. Same-compiler source inspection identifies added bookkeeping
for divergent half masks: MATCH.ANY/VOTEU.ANY/REDUX.OR sites 0 -> 13/13/16 and
BSSY/BSYNC sites 24 -> 59. Original/current native source matches exactly.
Old full16K populations also show unequal adjacent row counts in over96% of
pairs, with about25% logical row padding. Neither static instruction counts
nor that historical population is a hardware-counter timing attribution.

### One cause-directed correction: literal collective masks, 09:03 UTC

Direct PTX cannot remove the identified machinery: the current PTX already
uses the native instructions and the extra work appears in ptxas lowering.
An off-tree compile of per-collective literal-low/high mask branches removes
the MATCH/VOTEU/REDUX sites but increases static SHFL/WARPSYNC sites twofold,
BSSY/BSYNC59 ->106, registers85 ->89 and static instructions2520 ->3256 on
RTX. No physics, width, root, row order or budget changes. All shared-memory
synchronization remains. NVIDIA's [documented participant-mask requirements](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-sync-intrinsic-constraints)
must still hold separately for each half; a full-warp mask inside divergent
world loops would not be a safe shortcut.

Root funds one native corrective measurement despite the unfavorable static
counts: static duplication alone does not establish dynamic cost relative
to the old runtime mask partitioning. This is the one targeted correction
for the measured mismatch, not an eight/four-lane or threshold search. Native
same-input parity precedes one unchanged whole pair; a regression closes the
mapping, and a meaningful gain requires balanced repetition. No default or
parent-pointer promotion is implied. Offline evidence:
`/tmp/fpgs-g1-paired-literal-offline-Ewx2goKW/offline02/report.json`, SHA256
`4275098a3be171c95f71f6bf8fadb1f555ba3bde895ae3bf2e9f2330336453c9`.

### Literal-mask corrective run: regression confirmed, 09:12 UTC

The exact offline-tested correction is frozen at module SHA256
`b7485ab38babd9af21efc11ad6a6bad3d4db4717011f54ec6c6c072a419b1734`.
Generated native source SHA256
`81746dacfd257eb9cb05af262ff328677e1bc57c7d2f00b68144bf0ed7e93ecd`
matches offline02 byte-for-byte. Independent review confirms half-uniform
literal branches at each individual collective, not duplicated world
functions. Both native selectors pass on both GPUs (root10372/88318), and
the changed Warp native module actually recompiles. Existing CPU controls
and targeted pre-commit pass. No tolerances or numerical settings change.

| Complete 16K corrective discovery pair | RTX PRO6000 | GB300 |
| --- | ---: | ---: |
| Accepted baseline physics | 15.608934 ms | 20.508753 ms |
| Literal-mask paired physics | 17.616157 ms | 21.599840 ms |
| Baseline / candidate | 0.886058x | 0.949486x |
| Baseline environment wall | 30.416604 ms | 35.352970 ms |
| Candidate environment wall | 32.107146 ms | 36.062266 ms |

All source, capacity and idle guards pass; root73652 exits0. Same checked
driver, original producers, numerical allowance and 200/40/40 protocol.
Artifact `/tmp/fpgs-g1-paired-gs-literal-whole-paired16k-20260916-01/manifest.json`,
SHA256 `757a7102aca3adfaf8fe747ebe9cff632ab382fb816af667437e24d3a568f547`.
This is a separate matched baseline pair, not a formal balanced comparison
between the two losing implementations. It establishes that removing dynamic
mask regrouping does not recover a whole-physics win. No further mapping
grid, new numerical approximation or validation campaign is funded here.
The prototype remains default off; accepted runtime is unchanged.

The original mapping and native controls are preserved in `45491fa7`, pushed
to `ooctipus/newton:ooctipus/fpgs-active-wrench-20260916`. The correction is
retained as a separately reproducible failed experiment, not as a production
optimization. No fresh MJWarp or cross-task speedup is claimed from these
G1-only measurements.

## Remaining tree-dataflow screen, 09:30 UTC: not a measured gain

A final read-only audit finds no redundant complete next-state producer to
delete. Current bias43, screws and COM offsets feed live force/contact
consumers. Public poses/COM velocities are required. Rotated inertia,
subtree moments and geometric434 already run only for the upcoming original
factor refresh; matching valid current state skips repair. Existing finish
launches alternate about 250/332us on RTX and 271/353us on GB. Equal-count
cadence groups differ by 0.328ms/env on each GPU. This is observational
cadence evidence, not isolated phase timing or permission to remove refresh.

A general additive-only chain representation is distinct from another tile
size. The actual 44-body G1 tree has 12 heavy chains, maximum light depth 2,
and maximum chain lengths by stage 11/9/2. Each complete-chain warp could
scan poses, construct existing world-origin screws, scan the original
motion/bias law, and publish both before its light-child chains advance.
Reverse additive suffix/gather stages would form subtree wrenches/moments.
It needs no extra numeric global cache and preserves every public output.

| Source-derived work, not GPU timing | Current | Interleaved chains |
| --- | ---: | ---: |
| Pose compositions | 124 | 87 |
| Common-frame motion compositions | 124 | 87 |
| Forward critical-path scan rounds | 8 | 18 |
| Forward/reverse CTA stage joins | 27 | approximately 6 |
| Subtree additions per component | 43 | 87 |

The longer staged critical path is material: a light child of an early
node waits for a complete heavy chain, even when its own ancestry was short.
Other integration, root-origin, publication, projection and status barriers
remain. No phase evidence establishes a one-millisecond whole saving.
Replacing the separate local pose/motion scans with an associative
SE(3) motion/bias tuple is algebraically valid, but adds up to 348 vector
rotations and 174 translation crosses per world and increases live tuple
storage. It is not assumed to be the faster implementation.

Global prefix-end subtraction is not the proposed subtree algorithm. A
prior controlled counterexample at
`/tmp/fpgs-franka-cooperative-state-FMucSlN7/test_source.py:74` places 1e20
outside a subtree and 1.25 inside: additive projection retains 1.25 while
FP32 prefix subtraction returns 0. This is a regression counterexample, not
a claim that current G1 traces contain that mass/force ratio.

No new native variant is funded from this source-only screen. The next
decision requires the actual traversal cost and a complete replacement
budget; reduced joins or operation counts must not again be substituted
for whole-physics improvement. Existing Newton serial FK combines pose and
motion, but the inspected installed MJWarp branch kernels keep kinematics,
COM motion/acceleration and reverse force propagation separate. Neither
inspection supplies measured evidence for this particular chain design.
