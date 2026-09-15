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

## Final closure: physically qualified performance loss, no promotion

The runtime at `16f114edfc913eaf0b761c4f72067c27055845b0` is closed after
paired physical qualification, whole-environment timing, strict node attribution,
same-input owner timing, and a source-exact clock-only diagnostic. No register-LU
rescue or further mapping/parameter variant is funded. This closes this native
representation, not every possible contact-solver algorithm. Runtime and test
bytes remain the 945fec/597d74/2ecb3a pins recorded above.

### Corrected physical qualification

Root native02 sessions78230/6434 both reaped exit0: five test groups per card,
23.310 s RTX and23.312 s GB. All 2,048 actual current/held saved cases and their
29 limit rows pass the unchanged independent physical gates. The singular
projected safeguard passes physical quality; initial-overflow, late23-budget,
nonzero incoming/warm, self-contact and unsupported paths retain their exact
original controls. The actual two-world lifecycle covers counts12/54, two
graphs, reset and regrowth. No skipped test or relaxed numerical threshold.
Two inherited invalid-stream-event warnings per card and the legacy target-layout
deprecation remain in the logs; this is not a warning-free claim.

Logs `/tmp/fpgs-anymal-active-dual-native-1RnAUUn0/native02_gpu{0,1}.log`, SHA256:

- RTX: `a6382d81b019dcc6f0dc8dbc812891a4fc3b4fee4a379a4fc6371647e30fa910`.
- GB: `95e399b3adf8dddefe754c43b5772bbbf9257b6f537f80fa51168a715b2da1f2`.

### Whole environment and complete node attribution

Root whole session69505 reaped0 using the original paired protocol: 16,384 worlds,
seed0, 200 warmup,40 wall and40 graph-profile steps, unchanged24 parallel/8
fallback budgets, two substeps and decimation4. Both arms use the same accepted
INK/WR/EX1/MF/lazy/simple-zero recipe, dense72/MF32, raw212,992 and broad294,912.
Only the requested active-dual owner and candidate source differ. Actual `_ad1`
keys, capacities, source and idle checks pass at both original boundaries.

| Complete timing, ms/env step | RTX baseline | RTX active | GB baseline | GB active |
| --- | ---: | ---: | ---: | ---: |
| Physics graph | 9.400965 | 21.632152 | 9.514575 | 25.799906 |
| Unprofiled environment wall | 17.169826 | 28.962356 | 17.893967 | 33.830285 |

Physics regresses by12.231187 ms RTX and16.285332 ms GB; baseline/candidate is
0.434583x/0.368783x. Artifact directory
`/tmp/fpgs-anymal-active-dual-whole-paired16k-20260915-01`, manifest SHA
`ba06fed4c9faf3ef72d275d05807d0857e702d3f59170baa5443ed01c85a867c`.
The trajectory has modestly fewer contacts and more GJK queries, but that is not
the measured source of the large regression.

Root matched node session93896 reaped0 with the same200 warm/40 wall and3
profile steps. Each of four strict audits covers all12 physics graph roots,
all memory, original process/correlation/source laws and both checked boundaries.
The original strict reader is `/tmp/fpgs-franka-articulated-attribution-Y8sTiAgS/audit.py`,
SHA `e76d53f685871f8d3e38a221ce89485d3d87014b9c8af1f6ac22e10af8e74dc6`.
Each tier executes eight calls/env step,24 calls over the captured three steps.

| Exclusive busy time, ms/env step | RTX baseline | RTX active | GB baseline | GB active |
| --- | ---: | ---: | ---: | ---: |
| Complete32 owner | 1.709216 | 6.501955 | 1.797216 | 9.082132 |
| Complete48 owner | 2.382464 | 9.498512 | 2.602283 | 11.308190 |
| Parallel-owner union | 4.091680 | 16.000468 | 4.399958 | 20.390717 |
| All retained owners, including collision/memory | 5.062068 | 5.027597 | 5.056388 | 5.047635 |
| Collision subset | 1.029141 | 1.026731 | 0.936116 | 0.938698 |

Thus both actual complete solve owners regress while retained work is nearly
flat; it is not an added external producer or collision-cost explanation.
The table uses interval union/exclusive cost, not summed kernels with overlap
double-counted. Node graph spans9.335700→21.215393 ms RTX and
9.708524→25.690066 ms GB are attribution observations, not the whole timing
authority. The required3.127974 ms complete owner would now require an80.45%
cut from measured RTX16.000468 ms.

Node directory `/tmp/fpgs-anymal-active-dual-nodes-paired16k-20260915-01`,
manifest SHA `572af4ca9ac38aae51c581875cf2a2bf8c229c75e682a700d119cdf04435734c`.
Each arm/card folder contains `strict_active_dual_node_audit.json`; SHA256 in
baselineRTX, baselineGB, candidateRTX, candidateGB order:

- `af2079c477216ace703fa7130d90b34e8012ca455eb0f43e7e7a1470871a8036`.
- `0f75587a094000d39c1ff26e1f35b3690c1b8aa3d556be7e40806f3a54f670ba`.
- `7e80eb9527699dfaccde6a2b7bb8c17a4bd732dd82b4e891b0e99a69828734fa`.
- `316de3f57d93affc18faa36f1966ea097f3ec70ece9b6c2d414ebee47d58be76`.

### Saved-input timing and preserved helper failure

The first external timing helper incorrectly hardcoded live72/32 capacities
against saved192/1 arrays, changing the native dense stride. Sessions35113/20471
both failed their eager-output check before timing. This was a helper defect,
not a solver failure. Failed bytes remain as
`/tmp/fpgs-anymal-active-dual-same-input-jLn9EWCC/time_saved_failed01.py`, SHA
`f7b1db22466385c79ffde7f56a25dba3d5e7d4a5f794ace0aaea9f65e64fd4e1`;
original logs and incomplete gpu0/gpu1 results remain untouched.

The corrected helper derives both capacities from the saved arrays exactly as
the frozen physical helper does; no repacking. Its CPU expected-key and negative
old-key controls pass. Root89699/58981 both reap0 on all eight saved512 groups,
covering2,048 tier-owned cases per card. Every timed launch restores all five
written arrays before the CUDA-event interval. The existing50 alternating
rounds/first10-discard graph recipe measures the complete selected owner only.
Eager, graph and last-timed outputs agree within each arm; readonly inputs remain
unchanged. Active/original GPU-duration ratios are3.0982–9.5200 RTX and
3.1266–9.7106 GB on identical inputs: a large native execution loss independent
of trajectory/contact changes. Saved512/192/1 latency is not live16K/72/32
throughput or a live population census.

Corrected script SHA `e5a120feb9989deda890428e0f9380a746989904b5cead151bdea918ff4aa683`.
Results in that directory at `gpu{0,1}_02/result.json`, SHA256:

- RTX: `fe058d47ad2e13498eb96b1977bbaac4108ecfae7c39fb798ee50fd5e86b2df7`.
- GB: `d24d718613d15e2f295ceaaacb6701b288b1e08d7d4094b5af5f43a8c41d457a`.

### Clock-only attribution and final causal verdict

The external phase factory strips back to the exact frozen native source,
preserves its barriers, and appends only a26-field uint64 diagnostic array.
The saved runner compares all five written arrays and readonly inputs byte-exact
against frozen active, resets diagnostics outside timing, saves every world's
raw fields, and checks cycle conservation and24-budget/trial counters. All
eight groups pass on both cards. Instrumentation/frozen timing distortion is
1.0002–1.0254 RTX and1.0118–1.0243 GB; its higher130/130 RTX and128/132 GB
register allocations are charged, not claimed to preserve achieved occupancy.

Combined Jacobian construction plus LU consumes54.518% RTX/54.894% GB of summed
within-world cycles, with per-world median fractions53.563%/53.687%. This is
neither a GPU-wall fraction nor a claim that those cycles are removable. All
2,048 worlds are admitted/done, with zero original fallback sweeps. Non-timing
counter fields agree exactly across the two execution GPUs. Residual calls agree
with the CPU work model except eight separately charged inactive couplings and
one earlier world331 stop, net six extra calls. Gram-ID reuse already avoids
66.17% of rebuilds relative to LU attempts. Unexpected fallback or an unmodeled
iteration explosion is not the observed explanation.

The slow held captured-RTX world142 has the predicted15 outer evaluations,
110 residual calls,83 Newton trials,12 projected trials,14 active18 LU attempts
and one Gram build. On RTX its within-world split is61.76% Jacobian/LU and
23.14% trials. Raw per-world tails and p50/p95/p99/max statistics are preserved;
one such world can dominate a few-wave512-world kernel, so averages or summed
cycles are not converted into live16K wall time.

The source explains the expensive execution structure. A successful active12 LU
has12 dependent pivots,120 pivot shuffle operations,37 warp joins, shared
Jacobian row traversals and12 single-lane reverse solves. Every line trial adds
eight warp joins at tier32 or eleven CTA joins at tier48, all-row products and
projections. Gram reuse does not retire Jacobian/LU or globalization. Any LU
also requires the324-float held-factor reload. These costs were only arithmetic
proxies in the CPU feasibility study; fewer products did not predict GPU speed.

Phase directory `/tmp/fpgs-anymal-active-dual-phase-KYE5yRY9`; factory SHA
`c9fcc9b50d4634d562758cceea6e66265e31ddffb7be31cd0e7756b80b9790b1`,
runner SHA `522ab40e7a6327097c52a4e1722e77c7cf4020d5f35450495aac1fd78bfae58c`.
Results `gpu{0,1}/result.json` pin every raw diagnostic NPZ; SHA256:

- RTX: `19ee32b83d998dc2f2bc6c0dbf36d6a0432437cca51ca0e7db02866cbf0e41a5`.
- GB: `04725b74c871ab02fad3a426b6c8d9536209ae7f78165957cde67f72b594958b`.

Conclusion: independent physics passes, but the complete native solve is far
outside the funded cost envelope. The phase data do not support the80.45%
complete-owner reduction required for a register-LU-only rescue. No promotion,
no rescue grid, and no physics/iteration/tolerance change. Preserve this complete
loss and all diagnostic failures for future algorithm selection.
