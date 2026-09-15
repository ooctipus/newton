# G1 repeated zero-row expiry experiment

## Pre-code contract — 2026-09-15 03:31 UTC

Status: funded, unmeasured native hypothesis. Base is accepted composed kinetic
`b1bad06ae0cd5b86ac07b2f2844ef62770e99075`; original trees stay unchanged.
Branch: `ooctipus/fpgs-g1-zero-row-expiry-20260915`.
First native/CPU readiness checkpoint is 05:00 UTC, with no silent extension
beyond 05:20. Root owns every GPU launch.

The only candidate is repeated-row expiry inside the existing capacity-100
metric-tangent sparse GS owner, default off under
`FEATHER_PGS_SPARSE_ZERO_EXPIRY=1`. It does not seed certificates in the initial
prefix. It retains every current row/Z producer, held W434, original row order,
eight-sweep allowance, exact changed ballot, restitution, friction disk root,
scalar fallback, allocation/capacity guard, and final physical publication.
There is no new launch, queue, persistent cache, or world-scaled maximum buffer.

## Complete budget and differentiator

The current complete GS owner costs 3.969 ms on RTX. The hypothesis must save
at least 1.5 ms in original whole physics, with a complete new GS allowance of
approximately 2.469 ms including setup, checks, clock updates, fallback and
unchanged disk roots/decode. Operation counts below are not measured time or
population exposure. Early paired physical smoke and the original integrated
16K whole protocol precede any expanded validation. A loss requires causal node
attribution; no parameter or mapping grid is authorized.

Unlike closed whole-world ZERO, stopping, warm-start, active Newton, ordered
secant, or Gram studies, this retires proven-zero row transactions *inside*
loaded worlds without changing their trajectory or dropping a sweep.

For a row with all applicable old impulses zero, a fresh original residual
read r > 0 certifies the original projected normal/limit update is zero.
Let N bound the actual stored sparse row norm and T bound cumulative changes
in the stored kinetic velocity. A later visit remains zero while its residual
lower bound is positive. Save a conservative expiry threshold for T, then skip
only until it expires; a miss performs the original row and may refresh the
certificate. A full triplet skip additionally requires every original metric
admission guard and all three old impulses zero. Unsupported rows retain the
complete scalar path. No certificate may skip a nonzero old impulse.

Implementation will use one norm bound and one expiry float per row in shared
memory, with a solve-local clock. Norm initialization and all arithmetic are
charged. The bound must include FP32 norm/reduction error, future residual-dot
error, and rounding of every actually committed combined metric or scalar
sibling/own velocity update. Nonfinite/unprovable cases disable certificates;
they do not change the original numerical fallback. Negative compliance must
not be treated as evidence that diagonal D bounds the physical row norm.
Conservative bounds, not a new physical stopping tolerance, determine expiry.

## Existing CPU replay evidence

An in-memory trace observer called the unchanged SHA-pinned helper
`/tmp/fpgs-g1-ordered-secant-cpu-HpvXKyo6/study.py`, SHA256
`cb22a3f6cc90c0ec1a0ad7f98d29326015678a4d92a4603c8cf088d09703cb2d`,
on its existing sixteen current-1600/held-1601 payloads. No helper or runtime
was changed. Full stdout is retained in tool chunk `306987`.

| Cohort | Original residual dots | Repeated-only omitted dots | Omitted complete triplets |
| --- | ---: | ---: | ---: |
| All sixteen | 1931 | 1198 (62.04%) | 522 |
| Eight loaded eight-sweep cases | 1852 | 1180 (63.71%) | 522 |
| Four hard 7410/4625 cases | 1172 | 761 (64.93%) | 410 |

RTX world 7410 current/held omits 156/290 and 185/306 dots; GB world 4625
omits 214/288 and 206/288. Ordinary loaded world 8192 omits 101/176,
113/168, 102/168 and 103/168. All claimed omissions had exactly zero original
updates and positive residual. Original max-eight and exact early exit remain.

Original totals: 764 complete metric visits, 650 zero-radius visits, 194 disk
root probes, 426 changed coefficients in 192 kinetic clock transactions.
Repeated-only bookkeeping: 452 norm square roots, 1537 eligible checks,
310 expiry writes. All 194 root probes remain. The trace includes all combined
metric coefficients and scalar sibling changes. An additional 128-epsilon
roundoff-margin sensitivity check changed no decisions, but is not by itself
a native floating-point proof. The FP64 helper gives one ordinary limit case
five exact sweeps rather than the earlier independent helper's four, explaining
1931 rather than the historical 1927 dots; neither is a native timing claim.

Initial-prefix seeding would omit 1381/1931 dots, but is explicitly not part
of this funded implementation. The native norm setup may cost more than the
diagonal-derived exploratory count and must be included in whole timing.

## Regression and release gates

Regression-first coverage must establish default-off original identity,
metric-capacity admission, fresh expiry/miss and zero-only skipping, mixed
normal/limit/scalar fallback, nonzero incoming tangents, negative/nonfinite
inputs, conservative rounding, and solve-local reset. Reuse the original
current/held physical residual/complementarity/momentum helpers and actual
odd-world/reset/graph lifecycle controls; keep all physical tolerances.
Inspect emitted native/AOT code and resources without inferring achieved
occupancy. Root runs paired CUDA then the unchanged original whole driver.
No promotion or speedup is claimed before those gates and repeated timings.

## First complete native draft — 03:45 UTC

The selected implementation uses the actual eighteen stored Z coefficients
per row, not D, to establish its norm bound. Positive FP32 squared norms are
inflated by 128 epsilons, an absolute squared floor covers underflow/FTZ, and
the square root is upward rounded. A zero/nonfinite squared norm disables the
row certificate. This charges 18 loads/products plus one square root per live
row, including rows that never receive a certificate; no new ABI inputs or
compliance assumption are required. Negative compliance does not establish a
norm bound and does not change the original positive-denominator row law.

Write c = 128 float32 epsilons, N for that norm bound, and
C = 1 + |incident| + |rhs|. The residual error allowance is
e(T) = c(C + NT). After a fresh unchanged zero transaction at T0, the stored
threshold is rounded downward from
E = T0 + (r - 2e(T0)) / (N(1 + c)). Only T < E can omit a later zero
transaction. This includes the saved error and growth of the future error,
not merely the first residual's margin. Scalar admission additionally requires
finite nonnegative omega and finite positive D. A metric triplet omission sits
inside every original metric guard and checks all three old impulses are zero.

Each committed update forms an upward triangle bound B from its actual
rounded coefficient deltas and cached norms. The replicated warp-uniform
clock advances upward by B + c(1 + T + B), covering arithmetic error in the
stored du update and clock itself. Combined metric updates advance once after
acceptance; scalar sibling and own updates advance separately. Delayed
friction clearing does not change du and does not advance the clock. No added
state survives a kernel invocation, and no original changed bit is suppressed
except for a certified transaction whose original update is zero.

Independent source review found no concrete bound or warp-lifetime blocker.
The original default scalar/metric/contact-block native snippets match accepted
b1bad exactly; the original metric SHA256 is
`eda6abf58c026c3abf9ebe6196c7790ed90444a0e5ce99f061d09fbd8a364352`.
The candidate retains the original argument list and types.

Offline compilation reused the existing sparse compiler and both SM120/SM100
targets without a CUDA device. Four actual original/candidate entries passed
at `/tmp/fpgs-g1-zero-expiry-offline-Kd5Qumz3/offline01/report.json`, SHA256
`d486c4ee1093006772928a778c512bf584f52a6e8f020570ede86bf34909a50b`.
Original: 72 registers and 1116 shared bytes on both targets. Candidate:
71/72 registers and 1916 shared bytes on RTX/GB. All have zero stack and
zero spills. These static resources do not establish achieved occupancy.
Candidate PTX contains the directed bound operations and no FP64 arithmetic
instructions; inherited unused SVD double globals are not executed work.

Regression-first test session 25132 failed with the missing `zero_expiry`
keyword before the factory hook landed. The first eight CPU controls passed:
two new expiry tests, one retained metric admission test and five G1 state
tests. No GPU or whole-step result is claimed by this checkpoint.

### First paired CUDA smoke: invocation setup failure preserved

Root's first six-selector invocation enabled expiry globally but did not enable
metric tangents globally. Five tests passed on each card, including the actual
five-world state/graph and saved-row physical controls. The remaining
`TestG1KineticStateCUDA.test_current_force_predictor_and_publication` constructed
its fixture outside `METRIC_ENV` and correctly raised exclusive metric100
admission before running the numerical test. Parents 41399/31059 were reaped
with exit 1. No candidate physical assertion failed. This is an incomplete
smoke result, not a six-test pass. Root will rerun with both explicit flags;
no runtime, physical tolerance or assertion is changed to address setup.

### Source readiness

Final CPU controls pass: three expiry, one original metric admission and five
G1 state tests, nine total in 1.779 s (parent 28711 reaped 0). This includes
mixed combined/separate FP32 clock updates with nearly cancelling terms.
Full `uvx pre-commit run -a` passes after its two formatting-only changes.
The formatted source was recompiled at
`/tmp/fpgs-g1-zero-expiry-offline-Kd5Qumz3/offline02/report.json`, SHA256
`ce5e01033dc4941e66d0b43aa41a82bbe68926cee50f86160b17990b5f2cb96d`:
the same four actual entries pass with identical resource counts and native
entry hashes. The new native module remains SHA256
`1d9f4998d62d4286c52fbdd1977b8df91a3a0f7a28ae26dd7c7900a8dbd16ee9`.
The original metric helper changes only its expected-key assertion according
to actual `owner.zero_expiry`; no numerical assertion is weakened.

## First integrated result and one corrective hypothesis — 04:11 UTC

The corrected eight-selector CUDA run passed on both cards without skips or
CUDA errors: RTX 6.117 s, GB 6.238 s, parents 88482/75542 reaped 0. This
includes fresh zero reactivation, negative compliance/omega, nonfinite and
metric fallback controls, saved sixteen rows, and actual five-world current,
held, reset and graph tests. Only the inherited target-layout deprecation
warning remained. Aggregate Python source guard:
`014cca03469f414e3b273cafa52bda3eb5de803161766eec6000609aa3b416cb`.

The original whole protocol then compared accepted b1bad against clean
`6c2e1fa94e06e91b263b8c944d5d6e1df5a258c3`, changing only the expiry flag.
All four children and root parent 23383 returned zero, with original capacity,
source and actual old/new owner checks passing. Artifacts:
`/tmp/fpgs-g1-zero-row-expiry-whole-paired16k-20260915-01`.

| Whole physics, ms | Accepted baseline | First expiry | Difference |
| --- | ---: | ---: | ---: |
| RTX | 15.608483850 | 15.894541375 | +0.286057525 |
| GB | 20.468369600 | 20.230433600 | -0.237936000 |

The 1.5 ms RTX target was missed. This is one discovery comparison, not a
promotion or a repeated throughput claim. Wall time was
29.777891550 to 30.028116098 ms on RTX and 34.092557302 to 34.434227750 ms
on GB; these environment timings are not training throughput.

The original paired node parent 17564 stopped after its two baseline captures
because the inherited Lab analyzer rejects the already-known auxiliary graph
roots. The unchanged original parent therefore never launched its candidate
arm. Root preserved that failed result, then captured the candidate explicitly
with both source slots at 6c2 and both expiry flags on; parent 86626 likewise
returned 1 from that analyzer after its two complete captures. Those second
directories are labelled `round_01_baseline`, but contain the explicitly
observed candidate source/key and must not be relabelled as accepted baseline.

Existing strict reader 74c5928b, with only owner-name classification additions,
validated each capture's source/correlation scope, twelve physics roots and
three separate auxiliary roots, with zero unproven nodes. Its
`strict_zero_expiry_node_audit.json` files are preserved under:

- `/tmp/fpgs-g1-zero-row-expiry-nodes-paired16k-20260915-01` (accepted baseline).
- `/tmp/fpgs-g1-zero-row-expiry-candidate-nodes-paired16k-20260915-01` (first expiry).

| GS exclusive diagnostic, ms | Accepted baseline | First expiry | Difference |
| --- | ---: | ---: | ---: |
| RTX | 3.943349333 | 4.175872000 | +0.232522667 |
| GB | 4.363572333 | 4.113962000 | -0.249610333 |

Other RTX family deltas are at most 0.0042 ms. On GB collision changes by
-0.047913 ms, and other families by at most 0.00182 ms. This isolates the
whole discrepancy to GS but does **not** prove whether invariant guard reads,
norm setup, certificate arithmetic, or lower actual hit exposure dominates.
The selected CPU 62% figure is not a measured population exposure, and the
original zero-radius metric path already skips both tangent residual dots.

### Approved correction, recorded before runtime edits

Only move a certified lookup ahead of per-row metadata and lambda reads.
Add four shared uint32 words (16 bytes) recording whether a certificate owns
one normal/limit row or the complete validated metric triplet. A hit reads
the existing expiry and clock, then advances by the certified transaction
width. It no longer repeats immutable type/parent/support/friction admission
or the already-proven zero-lambda checks. Initial norm work, inward record
division, directed clock updates, all producers, eight sweeps and exact exit
remain charged and unchanged. No initial-prefix seeding, new queue or second
algorithm is included.

Safety rests on existing full upfront row/parent validation. Normal/limit
lambda has no foreign writer; a validated triplet owns both its tangent
siblings. Kind-three is issued only after original metric admission and an
unchanged all-zero transaction. Arguments and metadata are immutable within
the solve, and the friction-start predicate remains true after admission.
Monotone clock advancement prevents an expired certificate from reviving.
A delayed-friction kind-one certificate skips only its normal row and leaves
the original tangent visits in place; it is not promoted to a triplet without
an original accepted metric transaction. Expiry/kind state resets per launch.

Regression-first CPU parent 6660 failed on original 6c2 because its first
certificate check still followed row metadata reads. Focused tests will cover
the hoisted source boundary, delayed kind transitions, reactivation, siblings,
incoming impulses and unchanged upfront invalid-row rejection. The target
remains at least 1.5 ms original whole RTX savings against accepted b1bad,
not recovery relative to the slower first expiry attempt. The 05:00 readiness
checkpoint and no-silent-extension-after-05:20 limit remain in force.

### Corrective source readiness — 04:16 UTC

The correction changes only the expiry insertion module, its focused tests
and this report. Kernel arguments, key, producers, mathematical bounds and
original numerical paths remain unchanged. Independent source reviews found
no concrete lifetime or bound blocker. The focused regression now passes;
four expiry CPU tests plus the six retained metric/G1 controls pass (ten tests
in 1.632 s, parent 98476 reaped 0). CUDA cases additionally cover all four
bitmap words, bit31, a final97–99 triplet, delayed kind transitions and invalid
metadata on a continuing graph. Root's review added start70 to explicitly
cover bitmap word2; the focused CPU controls passed again afterward.

Both original/candidate entries compile on SM120/SM100 at
`/tmp/fpgs-g1-zero-expiry-offline-Kd5Qumz3/offline03/report.json`, SHA256
`7c3730c1136762c826b23ec49686da0f7d6a92cae21387da24b9014ba188202a`.
The correction uses 72 registers and 1932 shared bytes on each target, exactly
16 additional shared bytes over first expiry; stack and spills remain zero.
These are static resources, not an achieved-occupancy claim. Native module
SHA256 is `3103c2697b245ba144411c7b94c55b604e25edd278593efed89ce9da8134bdac`.

## Corrective whole result: measured small gain, target missed

The one approved correction was frozen at
`bc0cfd161c30631ac1cdc633e35b944a8bd507fd`. All eight existing/focused CUDA
selectors passed on each card: RTX 6.776 s and GB 6.828 s, parents
4486/11753 reaped with exit 0. These include the saved sixteen current/held
cases, five-world graph lifecycle, reactivation, delayed transaction kinds,
all four bitmap words and the original fallback/error controls. Aggregate
Python source SHA256 was unchanged before and after:
`8a6d85b7883400eef6aab9c230bcf83718cdf511dcce2e30a5344de9b56aaf7d`.
The tool retained test names, summaries and exit status, but truncated the
middle of verbose stdout; this is not a claim of full saved unit-test logs.

Root then ran the unchanged paired 16K protocol, 200 warmup, 40 wall and
40 graph steps, against accepted `b1bad06ae0cd5b86ac07b2f2844ef62770e99075`.
Parent 85494 and all four children returned zero. The manifest and every
child artifact hash were independently checked. All eight untimed capacity
boundaries, original collision checks, requested/observed expiry flags and
actual old/new solve keys pass; final source and device-idle guards pass.
Both Newton sources remained clean throughout. Fixed Lab remains
`53ee6b44c2334341305dbdf385a3916c6b140799`, with only its inherited untracked
`.venv` entry. No physics budget changed: eight sweeps, two Newton substeps,
solver dt 0.0025, sim dt 0.005 and decimation four remain original. Capacities
remain dense100, MF32, raw294912, broad49152 and triangle1769472.

| Whole physics, ms | Accepted baseline | Corrected expiry | Saving |
| --- | ---: | ---: | ---: |
| RTX | 15.604870425 | 15.305637050 | 0.299233375 |
| GB | 20.516338400 | 19.752022575 | 0.764315825 |

The paired physics ratios are 1.019550534 on RTX and 1.038695573 on GB.
Environment wall time is 29.290500650 to 29.393717099 ms on RTX (slower),
and 35.519032949 to 33.621285800 ms on GB. Wall time is not training
throughput. This is one checked discovery comparison, not repeated evidence.

Artifacts: `/tmp/fpgs-g1-zero-row-expiry-whole-paired16k-20260915-02`.
Manifest SHA256:
`27a20e538216bb86e635e5b13734711f434d7a5d724decda6ab8c50585ddcd49`.
Reproduction must retain the existing adapter
`/tmp/fpgs-g1-zero-expiry-checked-CR7edpTQ/run_live_sparse_checked.py`
(`81edb4daeb5a3e0ef0ce8dca421d422bfbc6b7dd25a9382cc74140caba395273`),
its checked child `d7dbc9cbeddb1b17e70a12a88dfc968601b3467e47d77a31cd332d4c7657cb0d`,
and the original 484 variants / 798 owner / 42b capture / c062 fixed-import
drivers recorded in the manifest. The manifest contains the exact flags and
four child commands; a current repository driver is not an interchangeable
replacement for these pinned files.

The correction recovers the first RTX loss and measures a small positive
whole-physics gain, but does not meet the 1.5 ms RTX milestone. No repeated
comparison, third tuning round, promotion, new MJ ratio or four-times claim
is authorized. Keep this isolated measured result and the first loss and
auxiliary-analyzer failures above. No corrected node capture is claimed;
the first node comparison isolates GS, not the correction's internal cause.
The broader all-task target remains unmet. Runtime stays frozen at bc0c;
this closure changes only the report.
