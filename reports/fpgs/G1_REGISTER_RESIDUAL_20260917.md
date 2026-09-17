# G1 register-resident residual experiment

Experimental, default off, not promoted. The initial unrolled version is
measured slower; one compact-control correction is under study. Funded 2026-09-17
03:20 UTC; initial complete checkpoint by 04:50 UTC. Base is corrected
limit-Jacobi checkpoint `f6ea5f2f`, not the slower ordered-limit predecessor.
No Isaac Lab, physics allowance, contact capacity or parent pointer changes.

## New evidence and hypothesis

Current shell/CSR observer histograms show worlds with at most 32 rows contain
89.9--90.7% of all rows/support entries on both cards. The September 13
static-register design's roughly 55% row eligibility is no longer current.
These counts do not measure executed sweeps or time shares. The original
shared-A/incremental32 losses remain closed; this tests statically named
register coefficients and row residuals using the present sparse producer.

Replace repeated Z-dot-du reductions with maintained physical row residuals.
Build a local Gram once, update residual registers for committed impulses,
then reconstruct physical response from actual original Z and accumulated
applied deltas. Keep denominator-only CFM, the current simultaneous-limit
transaction and its actual-Z energy guard, ordered normal-first spectral
contacts, incoming-impulse convention and maximum eight outer iterations.
Do not require identical rounded iterates; assess physical residuals, cone,
momentum, stability and fallback on the unchanged problem.

The small kernel and separately compiled original fallback use the original
15 solve arguments plus int[world_count] routing (64 KiB at 16K). Every call
rewrites each mapped world's route before admission; only fully published
small results set it. Large, unsupported or rejected worlds retain the
original corrected-limit owner. No global Gram or added row capacity exists.

## Complete cost gate

Current corrected RTX GS is 2.5853 ms in the strict diagnostic window. Target
at least 1 ms WHOLE physics saved against corrected limits after local Gram
setup, staging, prefix guard, register pressure, physical decode, the extra
launch/returning CTAs and fallback. Keeping the old producer or converting a
second global matrix is not this design. The unchanged large kernel must not
inherit the small kernel's resource maximum.

Historical ef9481 replay predicts 3.5--3.6x more scalar products for eager
Gram plus decode. The proposed win is shorter dependent warp work, not fewer
products; no throughput bound is inferred. Initial scalar FP32 setup is the
single implementation choice. TF32x3 local Gram is only a separately recorded
possible cause-directed correction, not an authorized tuning/precision grid.

Before timing, check AOT registers/shared/local memory, actual current/held
physical controls, 31/32/33-row transitions, cold zero/nonzero incoming
impulses, accepted/rejected limit transactions, friction delay, non-unit
omega and graph replay. Reuse the existing checked whole driver and record
actual small/fallback factories. A whole loss gets one bounded attribution;
no silent extension or mapping grid. The all-task 4x goal remains unmet.

Full cost card and census remain local:
`/tmp/fpgs-g1-register-gram-rescreen-rezSha/CARD.md`, SHA256
`572df946ea33207b0be4dd7e48cd1ac5a835c138c1078acdfef7057441cc8b43`.

## Initial implementation checkpoint, 03:42 UTC

Runtime `eb1076ba55eea08d7626a9a27c3668b6110eb316f3e2927f2bfed0c55d0b8197`
uses scalar FP32 Gram setup. Prefix physical response is accumulated by
43 coordinate owners in original prefix order from retained actual Z, rather
than the original changed-row sparse scatter. Charge 43 products per changed
row, not at most 18; it removes the extra step scratch and row barriers but
does not eliminate the physical energy guard. Applied deltas are accumulated
explicitly, excluding delayed-friction lambda clearing.

Independent source review found no remaining blocker after bounding the cold
zero shortcut to finite nonnegative omega. The initial offline compiler
report has 96 registers / 5760 shared bytes for the small owner on both
architectures, no stack or spills; filtered fallback is separate. These are
compiler resources, not throughput or hardware-counter measurements.

CPU checks: four new native-owner controls and twelve capture/limit/chain
controls passed. The actual-owner capture regression failed against pristine
f6ea (missing new owner), then passed against the candidate. GB300 native
five test groups passed, including independent saved-sixteen current/held
geometry, physical momentum, contact laws, 31/32/33 transitions, graph reuse,
accepted/rejected prefixes, nonzero incoming impulses and fallback. Log:
`/tmp/fpgs-g1-register-residual-native-gb-20260917-02.log`.
The first native attempt failed before kernel invocation because the new
test helper looked up a nonexistent host dictionary key; its corrected
group count uses the actual plan array. No tolerance or runtime math was
changed for that retry. Earlier geometry-coefficient diagnostics are retained
by the inherited physical tests, not silently treated as bitwise equality.

RTX native, the larger historical corpus, and whole-physics timing remain
pending at this checkpoint. No new speedup is claimed.

## Complete initial falsification, 03:50 UTC

The checkpoint above is preserved as pushed `8998bb98`. RTX subsequently
passes the same five native groups. Native logs are
`/tmp/fpgs-g1-register-residual-native-{gb,rtx}-20260917-{02,01}.log`, with
SHA256 GB `70ac6e117d1e358d031e15f9bb97bd73d3d12acb13f728b610407cefc1e42292`
and RTX `4aab295e1abec7f738964fb6f6dab762d98a349c5499ba60930ea84eb0e1484c`.
The first GB helper failure remains preserved separately.

Historical ef9481 native comparison reuses the original 635 RTX-source and
633 GB-source selected cases, private identical cold inputs, corrected-limit
baseline, and unchanged actual-Z/W physical diagnostics. All 1268 pass finite,
cone, nonnegative unilateral impulse and momentum checks. There are zero
pointwise diagnostic regressions at the inherited finite-budget thresholds;
these labels are not a converged oracle. Accepted small/fallback counts are
556/79 and 557/76. Momentum maximum tolerance ratios baseline/candidate are
0.231561/0.231561 RTX and 0.409866/0.213938 GB. Archived original metric-owner
outputs reproduce exactly before comparison, and immutable input/repeat/source
and selected-device idle guards pass. This corpus is NOT the current shell/CSR
population; independent current/held geometry remains the separate saved16 gate.
Artifacts: `/tmp/fpgs-g1-register-residual-historical635-rtx-20260917-01` and
`/tmp/fpgs-g1-register-residual-historical633-gb-20260917-01`.

Full 16K whole-graph timing against clean corrected-limit f6ea is slower:

| Card | Corrected baseline ms | Initial register ms | Baseline/candidate |
| --- | ---: | ---: | ---: |
| RTX | 12.548266150 | 14.130901925 | 0.888001786x |
| GB300 | 14.447917775 | 26.571799825 | 0.543731244x |

Unprofiled environment wall changes 26.092287 to 28.020672 ms RTX and
28.707518 to 39.834984 ms GB. Whole artifacts are
`/tmp/fpgs-g1-register-residual-whole-{rtx,gb}16k-20260917-01`.
Independent audits replay original source, artifact, capacity, budget and
activation guards. Only the explicit register-owner flag differs; dt,
substeps, eight-iteration maximum and calibrated capacities are unchanged.
Approximately 93% of worlds use the small owner on both cards. GB boundary
contacts and rows differ by at most 0.57%/0.43%, not a population explosion.
Reset call rates stay the same. Increased host termination time coincides
with a CUDA synchronization boundary, not evidence of extra termination work.

Strict node captures preserve the parent's auxiliary-label analyzer error,
while the source-pinned reader independently validates 48 physics/12 auxiliary
roots, zero unproven nodes, capacities and actual dispatch. Small/fallback
times are 3.236040500/0.935499417 ms RTX and 14.033757417/0.997253167 ms GB.
The GS interval union is 4.171539917/15.030661250 ms; all other major families
remain close. The earlier corrected-limit GS was 2.585308250/2.790901167 ms.
Thus the new small owner explains the whole loss. Node artifacts are
`/tmp/fpgs-g1-register-residual-node-candidate-{rtx,gb}16k-20260917-01`;
strict reader `/tmp/fpgs-g1-register-residual-strict-LqaEB5/read_register.py`,
SHA `464b1e415d6b62922a8df5516435b382b211912d12166b7fb7e5aa5ba7a56327`.
An independent replay produces byte-identical strict results on both cards.

## One cause-directed correction, funded 03:57 UTC

Compiled small-owner code is about 7.5 times the baseline footprint: roughly
375.8 kB versus 50.2 kB (375808 versus 50176 bytes on SM120).
The exact text-slot counts are 23488 versus 3136 on SM120 and 23496 versus
3144 on SM103, including operand-less padding/exit instructions. Static
instruction counts are not executed counts or instruction-cache stall evidence.
Both architectures still have no local-memory Gram array or spills. The large
GB/RTX cost split is measured, but its hardware mechanism is not established.

Fund one compact ordered loop with one shared spectral/scalar transaction body
and a uniform switch selecting four values from the constant named Gram
registers. No dynamically indexed Gram array, tensor setup, precision change or
mapping grid is authorized. Keep original prefix, physical decode, fallback,
projection, incoming-delta and maximum-iteration policies. Charge switch/moves,
loop control and any register increase; scalar Gram setup remains unchanged.
Require a smaller compiled body, inspect actual selector lowering and no spills,
then reuse physical tests and measure whole physics early. The approximately
1 ms target is against f6ea, not against the losing 8998 version. Card:
`/tmp/fpgs-g1-register-residual-strict-LqaEB5/COMPACT_LOOP_CARD.md`, SHA
`1bbb9ccd59aa2901292b9df3d5d7e1377e74949ecd207cb90b46c268d45c1c5a`.

### Compact source/native checkpoint, 04:05 UTC

Runtime `4adb4a3b2d9622afaca026f980143116978b423e9a3f3a1204d5de67f4f202c5`
implements only that compact ordered body. The new source regression failed
against 8998 (30 bodies instead of one), then passed. Independent semantic
review, five CPU controls, thirteen capture/limit/owner controls and full
repository precommit pass. All five native groups pass again on each GPU.
Logs: `/tmp/fpgs-g1-register-residual-compact-native-{rtx,gb}-20260917-01.log`.

Executable text shrinks to 62976 bytes/3936 slots on SM120 and 62848/3928
on SM103, about 83.25% smaller. Both retain 96 registers/5760 ptxas shared
bytes, no stack/spills/local loads or stores. The selector lowers to a
balanced compare/branch tree with four range comparisons and one or two
equality tests plus register moves and a common join. It is neither free
nor constant-time indirect register indexing; some leaves execute extra moves.
The numerical prefix, Gram setup, physical decode and fallback are unchanged.
The source-bound AOT report is
`/tmp/fpgs-register-residual-offline-POBpNKXt/offline03/report.json`, SHA
`4b4132eed9f83c7dfd4ee486b1ed733241321459da8af6e0df12eceb8029c04a`;
`CODE_SIZE_COMPACT.md` records the disassembly evidence. Full-pipeline timing
and compact-version historical replay are pending; no gain is claimed yet.
