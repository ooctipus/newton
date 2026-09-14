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
