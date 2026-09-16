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
