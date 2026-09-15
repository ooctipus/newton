# ANYmal active-dual semismooth solve: one bounded native proposal

Status: CPU feasibility only; no native implementation, GPU timing, or qualification.
Source baseline: `newton-fpgs-kinetic-qualified-20260915`,
`b1bad06ae0cd5b86ac07b2f2844ef62770e99075`.

## Decision boundary

Fund at most one integrated default-off owner, preserving the existing ANY18
WR/INK/MF current-row construction, held factor, original 32/48 tiers, and final
velocity publication. Replace the complete EX1 majorizer plus projected/Nesterov
recurrence on admitted worlds; do not execute that old solve alongside the new
one. Existing parallel solve is 4.067974 ms RTX. The falsification milestone is
new complete solve <=3.127974 ms AND >=0.94 ms whole-physics saving, with unchanged
physical gates. A 2 ms solve is a stretch target, not a count-derived prediction.
Other dynamics, factor, row production, collision and publication remain charged.

## Evidence, including negative controls

All 2,048 actual saved cases cover both capture cards, current/held steps and
32/48 tiers, including 29 position-limit rows. Source/input pins are retained by
the existing `rows.py` loader and `result.json`. This is not a population timing
claim and does not cover all unsupported MF/velocity-limit paths.

Original measured recurrence counts on these same cases average 23.95459;
2,026/2,048 execute all 24. The independent 65,536-case census averages 23.94612;
64,568 execute all 24. Thus this is not comparison against an unexecuted maximum.

Final FP32 partial-pivot-LU control with direct computed-physical stopping:

| Quantity | min | median | p90 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Outer evaluations | 2 | 4 | 5 | 7 | 11 |
| All residual/Jacobian/line evaluations | 3 | 7 | 9 | 18 | 47 |
| Exact same-ID Gram builds | 1 | 1 | 1 | 3 | 4 |
| Maximum active dimension | 6 | 12 | 15 | 18 | 18 |

All 2,048 pass the unchanged independent FP64 physical allowances. There are two
guarded LU rejections and 16 cases using a projected safe step, within the same
24-outer budget. Mean outer/evaluation counts are 4.22363/7.64648. Maximum active
dimensions histogram: 6:60, 9:352, 12:1265, 15:313, 18:58. The earlier FP64 study
found two transient rank-17/18 blocks; worst nonzero-spectrum condition ~12,327,
median ~20.4 and p99 ~8,509. No native pseudoinverse or positive regularizer is
proposed to conceal these cases.

Negative controls remain intact: raw FP32 residual stopping at 1e-7 passed
physical quality but cost median 7/max24 outer and median14/max171 evaluations.
A Cauchy worst-case residual-roundoff certificate passed physical quality but
cost median24/max24 outer and approximately126 mean evaluations. Those stopping
rules are rejected, not silently relabelled successes.

Final control: `physical_metric_control.py`, SHA
`8af4557bce9e45510e927cabea8dd9a141e62369f23729df51457be329d9947b`.
Result: `physical_metric_result.json`, SHA
`aedbfc6251b5fb676b1a5e47b44f5760bf1ba4a77961e1ee97efcc102e08e65e`.

## Physics and iteration contract

Use current rows with held response: A=Z Z^T, r=b+A lambda. CFM remains ONLY in
the original update denominator; never add CFM*lambda to the physical residual.
Keep every individual normal, tangent pair, and admitted unilateral limit impulse.
The map projects the normal first and tangents onto the disk of radius mu times
that projected normal. The tangent pair uses its common spectral step to express
maximum dissipation; this is not the closed standalone preconditioner proposal.

Inactive projection equations are identity equations, eliminated algebraically
and rechecked on every outer/trial. Active sliding tangents retain the derivative
of their normal-dependent disk radius. The active Jacobian is nonsymmetric:
row-scaled partial-pivot LU, not Cholesky. A small pivot <=64*FP32 epsilon after
row scaling rejects the Newton action; no invented physical compliance/SVD.

At each outer, compute all-row physical residual and check the uniform existing
quality floors: scaled natural defect, negative normal velocity,
complementarity, maximum-dissipation gap and cone excess <3e-5; negative
unilateral impulse <1e-7. No original impulse or original-quality result enters
this predicate. This is a meaningful computed physical stop, NOT a universal
FP32 error certificate. Actual independent FP64 gates remain mandatory. Native
may check this before constructing an unnecessary final Gram/Jacobian/LU; CPU
counts include that final construction, so do not infer a timing credit.

At most 24 outer correction opportunities TOTAL. Each Newton action has at most
eight projected line trials (alpha=1 through 1/128), using all-row natural-map
merit decrease. Failed LU/globalization uses one projected safe step in that
same outer. A late original-recurrence handoff receives only the remaining
24-minus-consumed budget, never a fresh24 after Newton. Existing unsupported
initial paths use their original 24 or ordinary8 fallback budget unchanged.

## Admission and reference/fallback correctness

First prototype requires the original stored incoming s_lam0 to be exactly zero
and finite, checked before modifying impulse/solver scratch. WW=0 does not prove
this. Nonzero incoming lambda falls back unchanged: original b'=b-A*lambda0,
s_lam0 and final decode of lambda-lambda0 remain original. Include an explicit
nonzero-incoming control; do not infer coverage from these saved zero cases.
Unsupported row_phase, frozen/delayed rows, warm/debug/shadow modes, invalid
metadata/nonfinite values and unsupported self/MF/velocity-limit routing retain
original admission/fallback. Active dimension >18 is safe original fallback,
not truncation. Native must test late growth and remaining-budget transfer.

## Native representation and complete charges

Retain original shared Z, physical metadata and lambda vectors. Retire the full
R-by-R EX1 rowsum-majorizer construction and old projected/Nesterov recurrence
only on admitted paths. Construct active Gram entries from stored Z; cache the
packed 18*19/2=171 floats (684 B) with ordered active IDs (72 B), reusing only
while those IDs are identical within this solve. No cross-step/held-cache reuse.
The projection derivative and LU change each outer even when Gram does not.

The nonsymmetric 18*18 Jacobian/LU needs 324 floats (1,296 B). A principled first
implementation may overlay original s_L[325] AFTER whitening, then reload all
324 held-L floats before original final decode. That is approximately170 MB
extra reads per16K*8-call environment step and is explicitly charged. Preserve
the original L path for early fallback and reload before a late fallback if
needed. Otherwise allocate/charge all 1,296 B; neither placement is free.
Original shared footprints are 5,408/7,492 B for the32/48 tiers. Cache/IDs add
about756 B plus masks/scalars/alignment; LU overlay avoids an additional1,296 B.
No uncharged register arrays or claimed occupancy gain. Keep original tier
mapping; no lane/block sweep. Pivot exchanges, per-stage synchronization,
whole-CTA joins and shared reloads all belong to the measured owner.

Each residual/trial costs Z^T lambda then Z times that18-vector, all-row
projection and reductions. Each Jacobian also needs the inactive-delta coupling;
compute via Z_I^T delta_I and active rows or an exactly equivalent reuse of the
current residual and active Gram. Charge up to another2*R*18 products/outer if
not reused; never omit that work from the budget. Gram reconstruction, LU row
scales/pivots, direction projection, every rejected trial and physical stopping
check are charged. Native accumulation differs from the CPU denseA reference;
existing FP64 gates, not algebraic equivalence alone, decide acceptance.

Arithmetic evidence, not timing: final control totals16,687,152 residual
products +5,963,868 cached-Gram products +10,960,128 LU flop proxy, before
inactive coupling, Jacobian assembly, trial projection, memory and reductions.
Original actual totals84,268,494 logical products including32,662,962 EX1.
The candidate can remove substantial recurrence/setup work, but serial LU,
extra factor reads and tail cases can erase it. This justifies one complete
early test, not a speedup assertion.

## Minimal qualification and closure

Reuse existing ANY saved-input native/physical and actual-constructor graph
controls. Cover current geometry/heldL, nonzero incoming state, all individual
cones/position limits, late-active growth, singular LU, unsupported/fallback,
grow/shrink/empty graph and immutable inputs. Preserve numerical tolerances;
different converged impulses are not rejected for bit identity. Run early whole
16K A/B with unchanged task parameters,24/8 budgets and original checked driver.
If complete owner/whole milestone misses, one source-matched attribution closes
the experiment; no retained-Gram revival, mapping grid or tolerance rescue.

Final stored-FP32 spectral-step control also passes all2,048, with identical
counts. Result `/tmp/fpgs-anymal-semismooth-cpu-ioOhLY/physical_native_step_result.json`,
SHA `78874fdae81286096ca70c759e45ebf43251d1684e73259993c069b0f74e3a53`.
Root funded one prototype at05:05 UTC, first integratedCPU/AOT checkpoint06:30,
hard07:00 without explicit new authority. GPU ownership remains root-only.

### Pre-GPU safeguard correction,05:26UTC

Independent test review supplied a dependent-row counterexample: two identical
unilateral rows with negative bias make the diagonal projected update oscillate,
although the saved16 projected-safeguard cases passed. Preserve the CPU results
as feasibility evidence, not proof of a general safe projected action. Native
rejected LU or exhausted globalization now hands off to the literal original
EX1/Nesterov solve with only24-minus-completed-Newton corrections remaining.
No impulse correction was committed by the rejected trial, so it does not buy
or consume another physical update. Initial rejection is exact original24.
Same-articulation self contacts explicitly take the untouched original path.
The test suite includes the duplicate-row and late-active-growth counterexamples.
The historical CPU iteration counts include16 cases taking that rejected
projected safeguard; they are not predictions of the corrected native path's
iteration counts or fallback cost.

First successful offline draft compile (before this safeguard correction) charged
6,836/9,176 shared bytes for32/48 tiers, versus original5,408/7,492. Registers
were96/128 RTX and99/96 GB, no stack/spills. These are static allocations, not
achieved occupancy or timing. Final source is recompiled after corrections.

### Corrected source-ready checkpoint,05:30UTC

The missing-owner regression failed with ImportError before runtime coding
(session56870, exit1). The new factory preserves the actual original ABI and
adds only `_fpgs_active_dual=True` and suffix `_ad1`; the production constructor
requires explicit `FEATHER_PGS_ACTIVE_DUAL=1`. Flag-off selects the unchanged
original owner. The full original factory source is hash-guarded.

Corrected offline03 compiles original+active32/48 entries for bothSM120/100,
session41772 exit0. Actual active resources: RTX96/128 registers, GB100/96,
shared6,708/8,984B, zero stack/spill loads/spill stores. Original entries use
RTX86/80, GB80/80 registers and5,408/7,492B shared. Only the64-thread entry uses
one CTA barrier resource; the32-thread entry uses warp barriers. This is not
achieved occupancy. The compiler removes the unused projected-vector storage
after the safeguard change; the live Gram/IDs/direction/LU scratch remains charged.

Offline artifact `/tmp/fpgs-anymal-active-dual-offline-0U7hfeeg/offline03/report.json`,
SHA `46d43e5e1911e4ae65203f6ec0ed61f5baad3d1a08ac02b93b3003c557f3fd47`.
Runtime SHA `deb6ede56d3e75f4d0297e0f30717a621f0bb5ac4153f62d664083bf422031fe`;
solver-hook SHA `e911476e839a7241d5cf3eccfe475e3ae4daf5264e3fa19f5b1a14df4d4c70e8`.
No GPU or whole-timing result is claimed at this checkpoint.
Five focused CPU tests pass (session49672,0.999s, exit0). The full-tree
`uvx pre-commit run -a` passes; all native/solver/test bytes remain AOT03-exact.

### First native failures and causal safeguard correction,05:48UTC

Root's native01 sessions54777/9444 both exit1. Warm/self/padding, unsupported
flags, singular/initial-overflow/late23-budget controls pass. Actual construction
fails on a missing `sys` import, fixed without changing numerical kernels.
The saved gate stops at capturegpu0/currentstep0/tier48/world142: independent
normal-negative velocity1.2046871e-4 exceeds original6.21345e-6 plus3e-5.
Logs are preserved at `/tmp/fpgs-anymal-active-dual-native-1RnAUUn0/gpu{0,1}.log`.
No all2,048 native qualification or whole cost is claimed from this run.

The four world142 CPU controls isolate the cause. Terminal original-recurrence
handoff reproduces the GPU defect (CPU1.2044581e-4) and fails allfour records:
it abandons Newton after the transient rejected step. One original-majorizer
projected update then retry also fails twoheld records; do not implement it.
Both results remain preserved as `safeguard_terminal_four_result.json` and
`safeguard_retry_four_result.json` in the original CPU-study directory.

Root approved one principled safeguard: on rejected LU/Newton line search,
backtrack along the projected direction with the SAME all-row merit-decrease
test. This is not the unstable unconditional diagonal update. Each direction
has at most8 trials; exceptional outers can therefore cost16 trial evaluations.
Only one accepted correction counts, total outer allowance stays24. If both
directions fail, or the active dimension exceeds18, use the original remaining
budget. No new cutoff grid, compliance, or acceptance-tolerance change.

This exact CPU action passes all2,048 plus duplicate-singular and exact initial
and late23-budget controls (session53044 exit0). Outer min/median/p90/p99/max
2/4/5/7/15; all residual evaluations3/7/9/19/110, mean7.72217. There are2 rejected
LU actions and16 projected-direction retries. Gram cache counts are unchanged.
Result `safeguard_projected_all_result.json`, SHA
`191937445b15503ac875ec574a4b0ff2308cb63a61c6cd3f15dc216514842ba9`.
These are CPU observations, not native timing predictions. The projected
direction's storage, additional trials and final/terminal L reload are charged;
no original majorizer is built unless the terminal original fallback executes.

Corrected actual-source offline04 passes all eight original/active entries on
SM120/100 (session83449, exit0). Active32 uses99 registers on both cards and
6,836B shared; active48 uses128 RTX/96 GB registers and9,176B shared. All have
zero stack and spill loads/stores. Original32/48 use86/80 RTX and80/80 GB
registers,5,408/7,492B shared. The64-thread entry has one CTA barrier resource;
the32-thread entry has none. The restored projected-direction lifetime is
therefore included in these allocations, not hidden behind the earlier03 data.
These are static resource facts, not achieved occupancy or cost evidence.

Offline04 report SHA
`4499098b63c94473a0a2d4b91ed464a2a4c1d16a1467ff9b88e095508d64f813`;
compiler derivative SHA
`2806d5a7bc47a104cbdd16725c31b932e3fe9393e934c6ff271b4f24957348c2`.
Runtime SHA `945fecad7da140c1d993d0300bc69d3c23602734cdf8c1054decfcfa3b192695`;
solver SHA `597d74fd6e39c41850a38fb2bc603f048861a00ddddc90b0e2401c7b8104a5a4`.
Independent safeguard review found no blocker. Five corrected CPU tests pass
(session54264,0.969s); test SHA
`2ecb3a34afb09d3759a066bf5f815d2a55cd27dd03db2d5a3349b324ab961960`.
The singular fixture now checks the unchanged physical thresholds rather than
requiring the original impulse solution; its normal-feasibility gate is also
explicit. Initial/late-overflow, warm and unsupported exact fallback gates are
unchanged. Corrected native qualification and whole timing remain pending.
