# Spectral proposal residual / lookahead closure — 2026-09-16

Status: closed, default-off and unpromoted. Native hard checks pass all 2048
saved cases, but original-controlled whole physics loses on both cards:
RTX 9.225785400 → 10.124493750 ms; GB 9.564807650 → 10.536172725 ms.
The aggregate CPU reference-quality improvement and 74 genuine reference
regressions remain separate from this cost failure. No third policy,
instruction/mapping grid, new oracle, or further native correction is funded.

The following proof and policy were written before evaluation. Parent is
cached spectral Jacobi `72b3bb0302c885cf3db3c3ec9e62f306438aded5`, whose
whole result was already a loss: RTX 9.214019375 → 10.193371200 ms.
The pre-evaluation target was another 1.979351825 ms reduction to reach
original minus 1 ms, not merely improve that losing candidate.

## Admission and unchanged physical map

Use the same current rows, held kinetic operator, diagonal, normal-first local
spectral proposal T, incoming-applied velocity convention, and maximum 24
committed corrections. Admit finite, positive physical/augmented row diagonals,
finite CFM, nonnegative friction, canonical indivisible contact triples,
and finite cone-feasible incoming impulses. Unsupported inputs take the frozen
full-merit spectral-Jacobi control, charged and labeled as fallback. No CFM is
added to the physical residual. All matrix quantities below are physical Gram
entries, except the explicitly augmented denominators.

For one triple, write current impulses (n,t), residuals (r_n,r_t), physical
normal/tangent cross vector h, and tangent Gram G_tt. Define

    a = G_nn + cfm_n > 0
    d = max(G_11 + cfm_1, G_22 + cfm_2) > 0
    s = lambda_max(G_tt) + max(cfm_1, cfm_2) >= d
    n* = max(0, n - r_n/a), delta_n = n* - n
    t* = P_(mu n*)[t - (r_t + h delta_n)/s], delta_t = t* - t.

G_tt is positive semidefinite, so its largest eigenvalue is at least either
diagonal; max(cfm) is at least either CFM. Thus s >= d. Native nonnegative CFM
and positive physical diagonals suffice for positivity, but nonnegative CFM
is not necessary for the bound: signed reconstructed CFM still obeys the
max inequality. Before any cohort evaluation, root review therefore narrowed
the admission requirements to the actual finite positive a,d,s and s>=d.
This avoids falsely rejecting diagonal-minus-recomputed-Gram rounding; no CFM
clamp or tolerance is introduced. These inequalities are checked in setup,
not assumed from a model name. Floating-point setup that does not satisfy
admission falls back without widening a physical tolerance.

The cheap unrelaxed proposal residual is

    Q_n = sqrt(a) abs(delta_n)
    Q_t = [s (abs(delta_t[0]) + abs(delta_t[1]))
           + (mu s + abs(h[0]) + abs(h[1])) abs(delta_n)] / sqrt(d)
    Q = max(all Q_n, Q_t) / cold_scale.

Singleton normal/limit rows contribute their Q_n. Coefficients are cached once.
This is always the UNRELAXED T(x)-x; multiplying by the current relaxation
would incorrectly make a half step look more converged.

## Bound on the existing mixed-scale natural residual

At fixed radius R=mu n let e_s=t-P_R(t-r_t/s). The actual proposal changes
both residual and radius. Nonexpansiveness of the disk projection, followed by
its radius-Lipschitz bound, gives

    ||e_s||_2 <= ||delta_t||_2 + (||h||_2/s + mu) abs(delta_n).

This explicitly charges the normal cross correction AND the changed radius;
omitting either is not a valid proxy for this T.

The existing physical check uses d, not s: e_d=t-P_R(t-r_t/d). For any closed
convex set and alpha >= beta > 0, projection residuals satisfy
||e_alpha|| <= (alpha/beta)||e_beta||, where
e_alpha=x-P_C(x-alpha r). To see this, the two projected points have normal
cone elements e_alpha/alpha-r and e_beta/beta-r. Monotonicity of the normal
cone gives, with A=||e_alpha|| and B=||e_beta||,

    beta A^2 + alpha B^2 <= (alpha+beta) A B,
    (A-B)(beta A-alpha B) <= 0,

and consequently A <= (alpha/beta)B (also valid when B=0). Taking alpha=1/d,
beta=1/s yields ||e_d|| <= (s/d)||e_s||. Each of the existing separately
weighted tangent components is at most sqrt(d)||e_d||. Replacing Euclidean
norms of delta_t and h by their L1 upper bounds proves that Q_t bounds BOTH
sqrt(G_ii+cfm_i)|e_d,i|, despite their unequal weights. Q_n is exactly the
existing normal weighted projection residual. Therefore

    existing_scaled_natural_residual <= Q.

Q=0 is the physical fixed-point law: delta_n=0 establishes unilateral normal
complementarity; unchanged normal then makes the tangent radius and cross
correction identical, and delta_t=0 is the standard disk Coulomb
maximum-dissipation fixed point. This is NOT a proof that Q decreases, that a
half latch always stabilizes the iteration, or that 24 passes suffice.

Small Q does not imply the absolute normal/complementarity/MDP gates. For
example, a singleton with a=1e12, n=1, r=1e-3 has weighted proposal defect
about 1e-9 while complementarity is 1e-3. Full physical checks remain mandatory.

## One fixed workflow, locked before evaluation

1. For a positive allowance, compute T(x_0) and Q_0 once. Do not stop at the
   initial state: preserve the old minimum-one-correction behavior.
2. Apply the cached unrelaxed proposal with current alpha, initially 1.
   Perform exactly one Z-transpose delta and one Z delta product and carry
   the resulting physical residual, as in the frozen control.
3. Compute T(x_trial), Q_trial. If alpha=1 and Q_trial>Q_old, permanently
   latch alpha=1/2 and use the same affine midpoint of impulse, residual,
   and kinetic correction. Recompute T(midpoint), Q_midpoint, charging this
   replacement proposal but no additional operator application.
4. Commit the state. Retain its unrelaxed lookahead T for the next pass.
   Run the unchanged full physical score only when Q<=1e-5 or this is the
   last allowed correction. Stop only if that full score passes ALL original
   thresholds: natural 1e-5 and normal/complementarity/MDP/cone 3e-5.
5. Charge the final lookahead even when the allowance is exhausted. It is
   required by the latch/proxy; it does not constitute a 25th correction.
   An allowance of zero runs exactly one full final score, no proposal or
   operator. Nonfinite intermediates cannot pass a stop gate.

With early_stop=False, apply the same admission/check policy but continue to
the allowance. Every return includes a complete final physical score; when
the stopping gate already computed it, reuse it rather than charging twice.
Final independent qualification keeps finite/cone/sign/momentum and reference
distance comparisons. Pointwise component tradeoffs are diagnostics, not an
automatic declaration of physical invalidity.

Count initial, ordinary lookahead, midpoint replacement and final lookahead
proposals separately; the final counter is a subset, not extra work. Count
all operator products, Q visits, full checks (including unsuccessful Q gates),
midpoint affine work, setup coefficients and persistent state. Added logical
persistence is one N-float unrelaxed proposal; scratch for old/trial states
already exists in the old midpoint workflow. CPU allocation is not a native
memory or timing claim.

## Comparison and prior art

First run the prescribed first-owned world in each of eight saved partitions,
then the unchanged 2048-case/reference adapter. Use only the existing converged
held-H reference artifacts, with source/input hashes checked. Report native
original, frozen cached-map CPU control, and this workflow separately, including
median/p99/max scaled velocity distance and tails. No new converged solves.

Natural-merit AA4 and natural Nesterov used different proposals/history;
ordered spectral GS changed global ordering; rooted block-Jacobi used local
root solves. None of the inspected controls implements this unrelaxed
normal-first cross-corrected proposal reuse with the permanent Q half latch.
The full-score latch itself changes here: this is an algorithmic workflow
experiment, not an instruction-equivalent optimization.

## Initial frozen CPU result and diagnosed failure

Initial helper SHA256 `60756db53bad57cd1d80e71cdf79c30ac32656917cc1f58601cc6605ccc2a04d`;
test `1e13a26053f7d8c842412e0041124412c70c9ce7ef45057aceb1dd5b53aed3a4`.
Missing-module regression was observed; five focused controls then passed.
The first eight passed all hard and pointwise checks, with 160 corrections
versus frozen cached-map 163 / native original 192.

The unchanged all2048 run has zero hard failures/fallbacks, 1501 pointwise
passes and 547 component tradeoffs. 2026 outputs are closer than original to
the existing reference; 22 exceed original by more than 3e-5 in scaled kinetic
distance, versus 66 in the cached-map CPU control. However a serious tail
prevents treating the aggregate improvement as qualification:

| Scaled held-H error | Original native | Frozen cached CPU | Q workflow |
| --- | ---: | ---: | ---: |
| Median | .006389918 | .000118792 | .000077728 |
| p99 | .031995761 | .006623684 | .004805623 |
| Maximum | .055738454 | .018130025 | .119926633 |

Work: 45,237 corrections versus 45,557 cached / 49,059 native; 132 permanent
latches versus 431 cached. Initial/lookahead/midpoint proposals are
2048/45,237/132 (the 2048 final lookaheads are included in that count).
Full physical checks are 3591, including 1692 unsuccessful Q-permitted checks.
Full metric row visits fall 1,397,617 → 109,843 (92.14%); proposal row visits
rise 1,324,383 → 1,378,068 (4.05%); operator products fall 47,677,788 →
47,299,104 (0.79%). These are source-work counts, not measured speed.

Worst case is RTX held tier48 world42. Q is dominated by contact normal row30
and decreases from .126109 to .107413 without a latch. Actual error oscillates
near .12. At pass3 the original full merit rises 5981.77 → 10614.64, driven by
absolute negative normal velocity .140569 → .318439; Q/1e-5 instead falls
12026.43 → 11692.50. The original cached map latches at this pass and ends at
scaled H error .001571600, compared with Q workflow .119926633 and native
.006766429. Residual carry error is 2.7e-15. This is genuine finite-budget
convergence loss, not a momentum/implementation error or harmless component
tradeoff. The 22 tails contain 17 tier32 and five tier48 cases; seven latched,
ten have excess above .001, and this one exceeds the original global maximum.

The full artifact and exact failed trajectory are preserved under
`/tmp/fpgs-spectral-residual-cpu-XDbx0r/`:

- `all2048/summary.json`: SHA256 `0801b28c92e1f4a49aef8afe30778c578fbeb3dacce8acc4b5c7f7ec36296290`.
- `all2048/result.json.zlib.base64`: SHA256 `357db38a53fbd8f46ef3f68b0013b917dcacc2e4d8be009925644ebad34ae8b3`.
- `worst_trajectory.json`: both actual unchanged trajectories, all metrics and
  per-contact Q terms; no oracle solve or algorithm variant.

## One authorized correction, locked before cohort evaluation

Keep the initial helper/result untouched. The separate helper
`spectral_residual_normal_control.py` adds exactly one stream

    N(x) = max(max_n max(-r_n, 0), max_n abs(lambda_n r_n)) / 3e-5.

It uses CURRENT trial impulses and carried physical residuals, not proposed
normal impulses or locally cross-corrected tangent residuals. Permanently
halve on the first increase in **Q OR N**, with strict comparisons and no
deadband. Recompute both at a midpoint. Q still only permits the unchanged
full-stop gate; N is a latch diagnostic, not a convergence certificate.

Taking max(Q/1e-5,N) would not fix the measured failure because the loose Q
term still dominates both passes. Root explicitly selected the independent
OR streams, not a weighted sum or a fixed-half policy sweep. Added work is
one multiplication/abs/max per normal plus another scalar reduction during
each proposal/lookahead, including midpoint and final proposals. Added
persistence is one scalar N; no new projection, norm, operator or pass.
Strict floating-point increases may latch early and will be reported, not
hidden by a new tolerance. An independent peer source/trajectory review
confirmed this dataflow and limitation.

## Corrected frozen CPU result

Corrected helper SHA256 `e550c62dbbab2aab3743e2cc0e32406ab51701cc8a0dff83df624cf212171a07`;
test `370e4e0768203ee650f30ae097c11b4052910b79b5c039e83c76450273bc03bb`.
Three controls pass after the missing-module regression, including exact
pass3 repair of the diagnosed saved case. Peer A read the complete corrected
source and found no concrete state/map/accounting blocker. Initial helper and
its artifacts remain untouched.

First8: zero hard failures, all eight closer to original reference, seven
pointwise passes/one diagnostic tradeoff, 163 corrections. Full2048: zero hard
failures and fallbacks; 1974 closer than native, 74 reference regressions
(all 66 cached-map regressions remain, plus eight new); 1378 pointwise passes,
670 component tradeoffs. This is not universal numerical improvement.

| Corrected metric | Median | p99 | Maximum |
| --- | ---: | ---: | ---: |
| Scaled held-H error | .000120582 | .006974387 | .024961323 |
| Own natural residual | .000045824 | .002065615 | .006862707 |
| Absolute normal defect | .000211755 | .012902916 | .029015560 |
| Complementarity | .000101422 | .006212440 | .022036940 |
| MDP gap | .000017914 | .001840288 | .056629353 |

The severe world42 tail now matches the cached-map impulses exactly. Corrected
maximum H error belongs to GB current tier48 world168: original .018685122 →
.024961323. Maximum H error is below original .055738454 but above cached
.018130025. Versus cached, 86 cases worsen by more than 3e-5 and 20 improve by
that margin; 94 improve at all. Maximum scaled excess over cached is
.006831298, and over original is .007910961. All are retained diagnostics.

| Complete CPU work | Cached map | Corrected workflow |
| --- | ---: | ---: |
| Committed corrections | 45,557 | 45,846 |
| Permanent latches | 431 | 476 |
| Full physical checks | 48,036 | 3,456 |
| Full metric row visits | 1,397,617 | 105,172 |
| Proposal row visits | 1,324,383 | 1,407,565 |
| ZT/Z products | 47,677,788 | 47,987,640 |
| Proposal/full-metric dynamic 2D norms | 1,837,366 | 573,844 |

The final row counts include all 2048 initial proposals, 45,846 lookaheads,
476 midpoint replacements, and all final lookaheads (subset of those counts).
There are 2094 Q-permitted full checks, of which 1525 fail an unchanged physical
gate, plus 1362 final-only checks. 597 final states pass the complete stop.
New normal-stream work is 470,057 normal visits and 48,370 scalar reductions;
new logical persistence is one proposal (59,835 floats across this cohort)
and one N scalar/world. Q setup prepares 2 coefficients/contact plus existing
normal weights; no new dense operator. Final decode and original producers
remain. The 92.47% full-metric row retirement costs 6.28% more proposal visits,
.65% more operator products, and extra Q/N reductions/state. The norm count is
an operation proxy, not an instruction/cycle or whole-time prediction.

Relative to cached, latch timing is identical in 1926 cases, earlier in 70,
new in 47, later in three, and absent in two. No latch occurs at pass1; the
largest early bin is pass2 (145 cases). 109 cases take more corrections, two
take fewer. Strict comparisons were retained without a tolerance adjustment.
Residual carry mismatch is at most 2.85e-14.

Corrected artifacts are under the same directory, `normal_all2048/`:

- `summary.json`: SHA256 `687a3f79c691669ebf21fa3a4c281c6a6d29ca161598ac731e941ebedcf3dc51`.
- `result.json.zlib.base64`: SHA256 `c6685f1b25ce8ae168bdd8a95e365282c9ca1d08d291e7ee154d4813e21ed9f8`.

Source/input/reference pre/post checks pass. No new reference solve or GPU
execution occurred. Root funds one subsequent default-off native cost screen
of this exact corrected policy, with original solver as control. The required
1.979351825 ms saving from cached remains a target, not a conclusion; cached
merit wall share was not measured and the older instrumented 53% share is not
a valid cached exclusive-wall budget. No third policy or parameter grid.

## Native implementation and source checkpoint

Default-off `FEATHER_PGS_SPECTRAL_RESIDUAL=1` selects marker
`_fpgs_spectral_residual`, suffix `_sr24`; both earlier spectral flags remain
explicitly zero. Current row staging/whitening, original 32/64-thread owners,
24 maximum corrections, raw CFM denominators, dynamic original fallback and
original final physical decode are retained. The unqualified cached loser is
not the timing control: paired timing uses the same original baseline as the
previous whole experiment.

Two Q coefficients and the next three-value contact proposal persist in
normal-owner registers; no new global or shared allocation. Q and N remain
separate arithmetic reductions but share cross-warp joins. Their four warp
partial values reuse `s_dv[0:4]` only after all row-action readers synchronize;
the next transpose action and original final decode overwrite this scratch.
The full cached physical measure runs only on Q permission or the final pass.
One initial/final lookahead and every midpoint replacement remain real work.

Eleven CPU controls pass, including 256 actual native Q/N expression checks
against the frozen CPU definition, exact factory ABI/block ownership and
unsupported-factory fallback. Both new test modules first failed on missing
implementation. Independent source review found no concrete lifetime, padding,
31/32-crossing triple, midpoint, or pre-mutation admission blocker. Full and
targeted pre-commit passed. No agent GPU launch occurred.

CUDA-hidden live72/32 AOT passes all four owners:

| Architecture / tier | Cached registers | Residual registers | Shared bytes |
| --- | ---: | ---: | ---: |
| SM120 / 32 | 85 | 87 | 5792 |
| SM120 / 48 | 91 | 132 | 8068 |
| SM103 / 32 | 85 | 91 | 4492 |
| SM103 / 48 | 91 | 96 | 8068 |

All have zero stack/spill loads/spill stores. RTX tier48 register growth is a
material liveness cost, not a free transformation or evidence of a measured
stall. Report `/tmp/fpgs-spectral-residual-cpu-XDbx0r/native_aot/report.json`
SHA256 `22becb2eb8624f3afd32f04c08a647ee5280ede2e40964f766b96022385a4946`.

Native checkpoint SHA256 pins:

- Runtime: `e6107d6c8ad074a3a89762cce5dbe79496a35f0cada1c36ea08e0272ea3d8598`.
- Solver hook: `c5153cc3195c94e52dfee69261046c8cfb80f45ec670a777abd5bf9d5109774c`.
- Native tests: `16a544ae836594653f738fa60642009ba1bc9b3dcb90530d0f8efacd69bbc03b`.
- Existing probe: `a8e85830b8cfbad2b7595b5b1c24c25347c5777261adf442f1e21174e3998599`.
- Checked observer: `2d55d11172bf3787c3aff9afa4046fd6d852c3f047c7e898447f3fb51730d8d6`.
- Paired runner: `98f424bd8310978943aafd2131ffa78dc7abf1ef15d1475cf47a024ae0ddf7cf`.
- Unchanged shell: `0e5c5b3153be3be947c8fa3f297bec20fd1a18b8e4a604a73ae3a2a93d7bb30f`.

The existing production probe has no direct per-world admission counter.
Zero CPU fallbacks do not prove zero native fallback, especially with the new
strict FP32 `s>=d` check. Factory/source observation and the saved translation
controls remain separate from dynamic admission evidence.

## Native saved-input result

Root alone ran both devices. All 2048 cases pass the unchanged hard
finite/sign/cone/momentum checks. Eight CPU-to-native translation controls
have maximum scaled held-H velocity difference 1.210523107e-5. These are
translation and hard-law controls, not a claim of converged physical residuals
in every finite-24 result. Native pointwise component diagnostics number 675
(341 RTX, 334 GB), versus CPU 670. The sets share 666 cases: nine are
native-only and four CPU-only. All nine new diagnostics are near-threshold
normal defects, with excess beyond the original-plus-3e-5 gate between
3.708e-8 and 3.042e-6. No tolerance was widened. These labels are distinct
from the 74 CPU held-H reference regressions.

Median saved-partition replay times, milliseconds:

| Card / state / tier | Original | Residual workflow | Original / candidate |
| --- | ---: | ---: | ---: |
| RTX / current / 32 | .048114093 | .053533187 | .898771 |
| RTX / current / 48 | .052132657 | .051803032 | 1.006363 |
| RTX / held / 32 | .048099186 | .054511312 | .882371 |
| RTX / held / 48 | .051732937 | .051945373 | .995910 |
| GB / current / 32 | .050226219 | .056284156 | .892369 |
| GB / current / 48 | .054424999 | .054560687 | .997513 |
| GB / held / 32 | .050139125 | .056832813 | .882221 |
| GB / held / 48 | .054398781 | .054172844 | 1.004171 |

The residual workflow is faster than the cached loser in these replays, but
original tier32 remains faster and tier48 is approximately flat. No replay
speedup is substituted for whole-task evidence. Logs are
`/tmp/fpgs-spectral-residual-native-20260916-X4pTezH6/gpu{0,1}.log`:
SHA256 `cd0101c18eb2e77758588bed7961f1df1e7450a033011d8b3ad655f600130b7c`
and `4d153ec426feba2f7ac32156f174e2d2c1fccf2ee730707317c8b69c030e1e6c`.

## Original-controlled whole result

Existing checked paired recipe: ANYmal D, 16384 worlds, seed0, 200 warmup
steps, 40 measured/profile steps, graph timing, one round. Original control
is clean `ca0d427af809571bb5501f644c1a6e03990cd2a8` in
`newton-fpgs-keyboard-linear-state-20260915`; the candidate is the frozen
native checkpoint above, not the cached-map loser. Both arms explicitly use
SIMPLE_WORLD_ZERO=1, SPECTRAL_GS=0 and SPECTRAL_JACOBI=0; only
SPECTRAL_RESIDUAL changes 0→1. Dense/raw/broad capacities remain
72/212992/294912. Lab, timestep, substeps and iteration allowances are unchanged.

| Whole milliseconds | RTX original | RTX candidate | GB original | GB candidate |
| --- | ---: | ---: | ---: | ---: |
| Physics graph | 9.225785400 | 10.124493750 | 9.564807650 | 10.536172725 |
| Environment wall | 17.774052449 | 18.714352575 | 18.574036774 | 19.152949899 |

Physics ratios are .911234243 RTX / .907806649 GB: losses of .898708350 /
.971365075 ms. Source, idle, actual-owner and capacity guards pass; parent
71540 exits zero. Manifest is
`/tmp/fpgs-anymal-spectral-residual-whole-paired16k-20260916-01/manifest.json`,
SHA256 `86f9cad577d37983e9c649ef30a24558c12a5206940e39ec5e7a20f3fcbedc79`.
Its physical/performance acceptance fields remain false, with no repeated
timing claim. Reaching original minus 1 ms would still require removing
1.898708350 ms from this candidate. Close the line without another policy or
instruction correction; retain the better aggregate numerical result as a
separate finding.

## Bounded original-dataflow diagnosis

The actual original matrix-free EX1/Nesterov loop is in
`solver_feather_pgs.py` around lines26280–26630. The candidate replacement is
`spectral_residual.py:_CACHE_PREP/_LOOKAHEAD/_OPERATORS/_COMMIT`, reusing the
pinned cached local projection and full-stop fragments.

| Producer → consumer | Original production | Residual workflow |
| --- | --- | --- |
| Current geometry/J → incident bias and held-L whitening | Per-row WR producer, register J/Z and shared d-major Z | Unchanged; current predictor incident and public row fields retained |
| Setup → update coefficients | Once-only incoming-reference correction and all-pair EX1 bound/diagonal | No EX1; diagonal, three contact cross products, spectral step, cold scale, reciprocals and Q weights |
| Iteration operator → proposal | ZT of one extrapolated y stream, then row dot immediately consumed in row-lane update | ZT of new-minus-old streams, then row action materialized in carried-residual scratch |
| Proposal → next state | Scalar row-lane update; tangent siblings project together using fresh normal | Normal-owner lane processes all three values, normal cross correction and common spectral disk, retaining next unrelaxed proposal |
| Control → commit/exit | One restart-dot reduction, Nesterov scalar update, cheap change ballot/flag | Two Q/N reductions per lookahead, permanent half latch/midpoint, occasional full physical score |
| Final impulse → velocity/publication | Fresh ZT(x−incoming), held-L transpose solve and output | Same original decode; not removed by residual reuse |

Crucially, original production performs **zero full physical merit scans**.
The 92.47% full-metric visit reduction is against the cached loser, not against
original. Corrected CPU work still includes 2048 initial proposals, 45,846
lookaheads, 476 midpoint replacements, 48,370 Q/N reduction pairs and 3456
full scores. Original's delta exit is cheaper and not the same convergence
test. The native grouped proposal uses robust `hypotf` and serial work within
each normal-owner lane; original tangent lanes use `sqrtf(a*a+b*b)` and
sibling exchange. These are source-derived execution differences, not
measured hardware stalls or an instruction-tuning recommendation.

The existing product proxy counts 50.141M local-setup/operator products versus
original 84.268M EX1/recurrence products; it excludes common geometry,
whitening/decode and does not price projections, reductions, joins, extra
state or dependencies. Original EX1 is cheap relative to this arithmetic
count: the preserved original observer reports 10.563% RTX within-world
cycles for EX1/bias preparation, 48.588% loop, 31.325% row preparation and
9.524% decode. See `ACTIVE_WRENCH_20260916.md`, "Original-owner cost and
reference-based reopening," and
`/tmp/fpgs-original-phase-paired512-20260916-An6F0B/gpu{0,1}.log`.
Those diagnostic windows are not exclusive GPU-wall shares. Likewise the
older uncached-Jacobi ~53% merit fraction cannot be assigned to the cached or
residual candidate. There is no new residual phase profile or hardware-counter
evidence. Zero AOT spills does not prove that liveness/occupancy has no cost.

No distinct exact-dataflow retirement in original production is supported by
a >=1 ms complete budget in this screen:

- A cold-zero specialization could remove its once-only incoming-reference
  ZT/Z correction and first zero transpose. This saves O(N D) work per call,
  not either O(N D) action in every one of 24 passes. Moving Nesterov momentum
  into DOF space also still needs one transpose per projected iterate; it can
  reuse final kinetic output but does not eliminate a per-pass action.
- Symmetric EX1 reuse would need row-sum communication or a new Gram producer.
  It cannot remove more than EX1 itself, whose complete wall budget is not
  established. Held H alone does not make current-J EX1 invariant.
- WR repeats contact geometry across three row lanes, but removing that
  repetition does not retire held whitening or the loop. The prior
  common-body/mask-basis cost screen did not establish the required whole
  bound, and branch/contiguous-response replacements already lost despite
  replacing their old producers. This is not a new funded representation.

This is a no-evidence conclusion for a substantial next original-dataflow
candidate, not a claim that every such optimization is impossible. No new
profiling framework, capture, GPU run or runtime edit was made for diagnosis.
