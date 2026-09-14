# ANYmal register Gram ownership — September 14

Pre-code card, 22:01 UTC. Source/AOT checkpoint 22:25 UTC; parent owns GPU
allocation and original paired physical/whole drivers. Experimental, default off.
Base `baf0c57a`, isolated `ooctipus/fpgs-anymal-register-gram-20260914`.

## Hypothesis and complete cost

Current ANYmal's 18-coordinate INK/WR/EX1 owner already forms every admitted
`A_ij = Z_i dot Z_j` once for the exact row-sum majorizer, then discards it.
Keep those coefficients as compile-time-indexed per-lane registers and consume
them with the existing dense four-accumulator matrix-vector product. Retire
the two factor-space products and their intermediate reduction/synchronization
on every projected/Nesterov sweep. Keep original current geometry, held L,
prefix rows, restitution, CFM, projection, restart, stopping and maximum24;
ordinary fallback remains maximum8. No new global A/Z or separate producer.
Shared Z remains for final `L^-T Z^T (lambda-lambda0)`; Jr need not remain live
after Gram construction. No shape, capacity, Lab or physics-budget changes.

Target: at least1ms RTX whole-physics reduction from current9.3885ms. The
existing complete parallel family is approximately4.075ms RTX, so this requires
roughly25% of that entire family, not merely a faster multiply. Full paired
whole timing is decisive; counts below are not converted to milliseconds.

Saved September12 original source-bound census contains65,536 world-calls
(two GPUs, adjacent current/held16K calls): mean29.518951 rows, range6–48,
98.52295% execute24 sweeps. Only790 prefix rows (one in each of790 worlds),
so an analytic-prefix hybrid is not useful here. All separating rows remain.
At most32 rows:39,815 worlds/60.75287%,52.63332% of rows. At most24:
21,164/32.29370%. At most36:59,940/91.46118%.

Counting logical products at actual executed sweeps, existing EX1 setup is
1,068,777,180 products; recurrence is1,668,191,148. Retained Gram recurrence
is1,422,539,255 (14.72564% fewer), total2,491,316,435 versus2,736,968,328
(8.9746% fewer). The potential larger benefit is removing the transpose/partial
reduction dependency chain and one warp barrier (tier32) or two CTA barriers
(tier48) per sweep. Projection/restart/stop and all row construction remain.
Additional32/48 live register values and generated instruction size may erase
the benefit: AOT must report registers, stack/spills and shared allocation.
Fixed names/indices avoid a dynamically indexed register array spilling by
construction intent, not by assumption; generated code is authoritative.

## Prior art and the distinct boundary

Fixed Lab `reports/fpgs/fpgs_solver_shape_and_spirit_20260909.md` section17.17
already measured ANYmal shared-A INK: tiers48/32 were300/191us before EX1 MF
improved them to234/157us; shared memory fell12.7KB to approximately7KB.
That A-based path publishes Z to global Y_world and reloads it for Gram and
decode. Public `matrix_free=False` still selects that path (`ZG=1`) and is NOT
this experiment. WR was added afterwards in section18. No retained-register
A/current-WR experiment was found in this bounded history audit. Old branch,
contiguous and lazy-active-row experiments changed different boundaries and
are preserved as losses; they do not establish this candidate's speed.

This candidate keeps existing shared allocations, performs no A shared/global
publication, and uses the already-required EX1 dot products. It is not a dense
matrix parameter sweep or a claim that the old shared-A path was untested.

## Admission, validation and stop

Flag `FEATHER_PGS_REGISTER_GRAM=1`; owner only18 coordinates, INK(18,0,0,0),
WR, matrix-free EX1, ordinary32/48 tiers, Nesterov24, no warm/debug/shadow or
register-whitening mode. Original factory ABI and default native source remain
unchanged; unsupported factory calls return the original kernel. Selected native
keys carry `_rg1` and `_fpgs_register_gram=True`.

Regression-first missing factory test precedes runtime implementation. Reuse
original current/held captures, raw-J/held-H physical oracle, both tiers,
warm-input algebra, empty/regrowth, fallback and actual eager/graph lifecycle.
Use physical momentum/residual/constraint quality, not per-operation bit identity.
After CPU/AOT/source checks, run original paired native controls and early
whole-owner A/B. If loss or below target, one causal attribution only before
any correction; no layout/register/unroll grid.

Inputs: `/tmp/fpgs-anymal-census-paired16k-20260912-01/gpu{0,1}/report.json`,
SHA256 `d7873c8b9e31788a95f36b1b8e2b58cad7b8cccd4cd3cc41a86db30e1a13b2ce`
and `256a37477d621be9fd4f64deb5c53cac2a22de0084adf4048ce4b2224efbe118`.
Historical report SHA256
`4ec51734cb58194a157043f908729f6d8e41376c9bba1245231c59160d47b2ab`.
Current source audited before branching:
`3c2a5f05ac4580904078866f0df9a6bb87b5bddd6483abaa4ac2b5f245d7917d`.

## First AOT and one causal compiler-lifetime correction, 22:10 UTC

`/tmp/fpgs-anymal-register-gram-offline-r5Dqm9/offline01` preserves eight
original/candidate SM120/SM100 entries. RTX registers32/48:86/80 to172/204;
GB80/80 to170/202. All stack/spill counts zero. Shared32 remains5408B;
shared48 falls7492 to7276B after the unused partial-dv array disappears.
This is resource evidence, not a measured performance failure.

Offline nvdisasm live ranges locate the peak in Gram construction:162/194
live registers. Adjacent compile-time j values are vectorized into LDS.128
loads at18 dimension offsets (144B/208B strides), keeping a four-column,
72-value Z tile live with the retained coefficients and Jr. Baseline's dynamic
j loop did not require that tile. Generated SASS instruction counts also grow
5488 to6436 (32) and5335 to6814 (48), including helper bodies.

One authorized correction makes only the Gram-build Z view volatile read-only,
preventing this unintended cross-column load grouping. Coefficients, scalar
dot accumulation order, sweep count, layout and law are unchanged. Preserve
the first Python source as `source01_register_gram.py`; compile a fresh02
resource artifact before judging the correction. No register-cap or grid sweep.

Correction confirmed by `offline02/report.json`, eight entries: retained-Gram
registers111/121 for tiers32/48 on BOTH architectures, versus original86/80
on SM120 and80/80 on SM100. All stack/spill counts remain zero; shared memory
is5408/7276B. This resolves the diagnosed extra tile lifetime, not the whole
performance question. No more compiler/layout variants are planned before
integrated physical and whole-owner timing.

## Source-ready checkpoint, 22:14 UTC

Regression-first missing module failed before implementation. Five CPU tests
pass (1.610s); complete `uvx pre-commit run -a` passes. Independent parent
review found no blocker in runtime hooks, generated arithmetic or tests.
No GPU was used for preparation. Native source pins:

- `register_gram.py`: `3469d83f37ee85d325fdb74291d786be4ae7dcb00d9da3bf8aef4d8e01c041fb`
- `solver_feather_pgs.py`: `a730b87761fbc009d5c71cf6dd3af7025a9a9e6b90cc2b39fbe3422b26fc9e72`
- AOT02 report: `676c3bad5728cd0739f88873292328472f9406ceab2c100e0fd612d7b8d09c23`

Root-owned paired CUDA selectors in `tools.fpgs_bench.test_anymal_register_gram.TestRegisterGramCUDA`:
`test_native_saved_current_held_both_tiers`,
`test_native_warm_self_contact_and_padding`, and
`test_native_constructor_fallback_current_held_graph`.
Set `FPGS_TEST_DEVICE=cuda:0` with one UUID-isolated process per GPU. The
tests bind original/retained factories explicitly and preserve ordinary source
arguments; the production lifecycle toggles only the register-Gram flag.
Whole testing reuses the existing fixed-Lab ANYmal paired variants owner,
base `baf0c57a`, candidate flag0→1, original16K/200/40/40 sampling and
0.005 sim_dt, two0.0025 substeps, fallback8/parallel24 allowances.

## Closed integrated result, 22:40 UTC

Runtime `acdb168e12e12f0907c9b13fdeee37f416d8d53a` is a measured loss,
not promoted. Root confirmed every physical, diagnostic, whole and node process
for this source reaped before this report-only amendment. Runtime and the
original failed physical test remain unchanged. No32-only micro-fork or further
compiler, layout or numerical correction is funded.

One exploratory paired whole round used the original checked variants owner,
fixed Lab and16K/200/40/40 settings above. Identical allocations were dense72,
raw contacts212992 and broad-phase294912. Original source/capacity/budget and
final idle checks passed. Physical qualification was explicitly still open;
these are valid measured costs, not accepted performance or convergence.

| Per environment step, ms | RTX original | RTX retained | GB original | GB retained |
| --- | ---: | ---: | ---: | ---: |
| Complete physics | 9.332332 | 9.525339 | 9.530262 | 10.399400 |
| Synchronized environment wall | 16.8238 | 18.0500 | 17.8829 | 19.1133 |
| Node diagnostic: tier32 | 1.685203 | 1.604288 | 1.791466 | 1.985803 |
| Node diagnostic: tier48 | 2.382771 | 2.783807 | 2.599605 | 3.216960 |
| Node diagnostic: both tiers | 4.067974 | 4.388095 | 4.391071 | 5.202762 |

Whole speed ratios are0.97974 RTX and0.91642 GB. The separate three-step
node capture retains four physics graphs/step; all four arms have936 kernel,
372 memset and120 copy nodes with successful process-scoped correlation.
Actual candidate owner names end in `_rg1`. Tier32 saves only.080915ms RTX;
tier48 adds.401036ms. Both GB tiers regress. Thus no measured path to the
funded1ms saving remains in this representation.

Actual native resources: tier32 registers RTX86→110, GB80→111; tier48
80→121 on both. Shared32 remains5408B; shared48 changes7492→7276B;
no local memory was reported. The retained products and removed transpose/
reduction are real, but the saved census gives tier48 only about.54% fewer
recurrence products, while tier32 has about27.5% fewer. Gram setup, current
WR/whitening and final decode remain. Extra coefficient lifetime and static
Gram-load/code expansion are costs, not proven achieved-occupancy or hardware-
counter diagnoses. Resource ceilings alone do not establish the timing cause;
the measured complete owners establish that this mapping does not win.

Artifacts (all preserved):

- Whole: `/tmp/fpgs-anymal-register-gram-paired16k-20260914-01`, manifest
  `418fcc4b1925072a79e9fb735ac1d0a8cdea99fde9f7b5ae664ee77839b9f67a`.
- Nodes: `/tmp/fpgs-anymal-register-gram-nodes-paired16k-20260914-01`, manifest
  `8bd1faacdbd57bd1aa472191cbdf1d0f18a01de8a2d3ec2ac3606185e3291578`;
  each arm's original `capture_analysis.json` records membership and owners.

## Physical failure and exact native numerical explanation

Both GPUs passed warm/self-contact/padding and the production lifecycle's
assertions. The saved selector failed at captured GPU1/step0/tier32/world357:
scaled cross-variant velocity8.9986223e-5 exceeds its unchanged3e-5 gate.
An external diagnostic then evaluated all2048 cases without stopping at that
assertion: exactly one case fails, including normal velocity and complementarity,
not merely the cross-variant distance. Momentum/cone checks pass. Its nine
contact triplets contain no limits; the normal metric in this specific case is
contact velocity, not a mixed contact/limit aggregate.

Old/new negative normal velocity is5.83864e-7/1.217565e-4 and complementarity
3.5728e-7/5.29958e-5. Natural residual and MDP slightly improve. Existing
FP64 references stop at16 and agree closely with the original. Neither a
non-FMA FP32 replay nor the FP64 replay reproduced the native divergence, so
neither was used to assume the difference harmless.

The narrowly instrumented native trace matches its uninstrumented outputs
EXACTLY on both GPUs. Initial current Z, b-prime, preconditioner, metadata and
impulses match between owners. Through sweep15 impulse differences stay below
8.95e-8. At15, residual reassociation changes the restart dot from original
+1.189393e-11 to retained−2.446598e-12: original drops momentum, retained
keeps beta.7185. Original stops at16; retained overshoots, restarts at16 and
passes the inherited relative-impulse stopping vote at17 with the larger
physical defect. FP64 summation of the traced FP32 restart operands preserves
those signs: higher precision for the dot reduction alone would not fix it.
This diagnoses residual-rounding-sensitive momentum plus the existing stopping
rule, not missing geometry, stale held factors or dropped rows.

The separate lifecycle emits two invalid-argument stream-event warnings/card.
Source review identifies a graph→eager transition after reset that retains
capture-created double-buffer event handles; initial in-capture seeding is
present. This location was not experimentally isolated. Do not call that
selector warning-free or the candidate physically qualified. The same warning
affects both owners and no additional warning-only GPU experiment was run.

External diagnostic scope/commands reuse the frozen fixture/oracle only, with
one UUID-isolated process/device and `uv run --no-project --python` pointing
to the existing fixed-Lab Python. They are not a replacement test or benchmark:

```text
python /tmp/fpgs-anymal-register-physical-diagnostic-rtNJ6c/diagnose.py --device cuda:0 --output <fresh-gpu-result.json>
python /tmp/fpgs-anymal-register-trace-Z7z4dJ/trace.py --device cuda:0 --output <fresh-gpu-trace.json>
```

All-case source SHA256 `809f9becd8611d0d09e2a9fc213f018216f43a3de8e4be22f3a0f43c62861790`;
actual/gpu0.json `4dae55734dac8c63c791522f16c564c270abe872077ba10e3337536f315cde42`,
gpu1.json `6514dba16eebf3d55e5fdcfecd1469ebd9641b48f9102075653f9d08dde5ce08`.
Trace source SHA256 `837aaa248fdf0168009742d4d4d6202f164d3758d5364bee6265ebe85c7c3140`;
both output hashes `e8b0281ece597bfafc8d62807e56ad812b3b4211d1483e8c0257aee9850cc4a8`.
The trace derives actual saved dense192/MF1 capacities and checks the ordered
ABI. An initial hardcoded72/32 draft was caught and corrected before any GPU
launch and supplies no numerical evidence. Full explanation remains in
`/tmp/fpgs-anymal-register-trace-Z7z4dJ/FINDINGS.md`.

A near-stationary projected-gradient momentum restart was suggested after the
trace, but is unimplemented and not funded for this slower representation.
No physics tolerance was relaxed and no original acceptance gate was removed.
