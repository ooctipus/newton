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
