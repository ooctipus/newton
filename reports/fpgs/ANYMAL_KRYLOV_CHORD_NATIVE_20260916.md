# ANYmal Krylov/chord native cost screen — 2026-09-16

Experimental, default-off, unpromoted and not fully qualified. The saved native
owner is substantially slower than the original on both GPUs, despite passing
hard physical controls and closely translating the frozen CPU policy. There is
no new accepted cross-task speedup. The complete RTX physics boundary loses
9.222293025 → 27.762799075 ms. Instrumented controls confirm that this loss is
not caused by excess native iterations or mistranslated fallback budgets.

## Boundary and frozen policy

Native branch `ooctipus/fpgs-krylov-chord-native-20260916` starts at CPU checkpoint
`c1b00ae5b4fe682481f574569f2307fa57b0020c`. The CPU policy and its earlier failure
are documented in `ANYMAL_KRYLOV_CHORD_CPU_20260916.md`.

`FEATHER_PGS_KRYLOV_CHORD=1` selects `_fpgs_krylov_chord` / `_kc24`; the feature
is otherwise off. Original current-row construction, held-factor whitening,
32/64-thread owners, canonical publication, capacities and physical time budgets
remain. The admitted cold solve replaces EX1/Nesterov with the simultaneous
individual-Coulomb map, an active physical-Gram cache, local right-preconditioned
GMRES8/CGS2, and a feasible Newton chord. Eight exhausted finite Newton trials
permit the existing projected-map chord with one additional response and at most
eight trials. Every accepted outer correction counts against the original 24;
guards continue literally from the current state with only the remaining budget.
CFM affects step denominators, not the physical operator. No dense Jacobian/LU,
factor overlay, new warm start, extra physical pass or contact-law replacement is
introduced. Active IDs are packed by contact group, including partial groups.

The corrected CPU cohort has zero hard failures, 2,017 fresh physical stops,
19 component-wise tradeoffs and five genuine H-reference regressions. Those
finite-budget tails remain explicit; improved aggregate convergence is not an
all-pointwise gate pass or a performance claim.

## Controls and compilation

- Missing-module regression was observed before creating the outer module.
  Actual current-factory ABI/admission controls pass for both tiers/architectures
  with register whitening enabled; inherited CPU physical/budget controls pass.
  The combined CPU invocation ran 15 tests: 14 passed, one CUDA test skipped.
- Root ran all three exact-fragment CUDA tests on RTX. GB then passed those three
  plus the actual owner fallback control, four tests in 9.039 seconds. The latter
  exercises nonzero incoming impulses and delayed friction at both tiers using
  unchanged saved geometry/operators. RTX also passed that fallback selector.
- Independent outer/linear reviews found no remaining source blocker. The first
  full-owner AOT attempt exposed Warp's `int`/`float` macro expansion in the packed
  Gram decoder; explicit casts fixed it before native execution. Review also
  added explicit nonfinite-map propagation so `fmaxf` cannot hide a bad trial.
  The finite policy did not change. Failed `offline01` is retained.
- Full-owner hidden AOT `offline02` passed at actual factory capacities 72/32:

| Architecture | Tier / threads | Registers | Shared bytes |
| --- | ---: | ---: | ---: |
| SM120 | 32 / 32 | 96 | 9,572 |
| SM120 | 48 / 64 | 128 | 11,912 |
| SM103 | 32 / 32 | 97 | 8,272 |
| SM103 | 48 / 64 | 128 | 11,912 |

All four have zero stack and spill loads/stores. The linear scratch is 3,496
bytes, retained alongside the original state. These resource facts do not prove
an occupancy or hardware-stall explanation. The saved replay retains its actual
captured 192/1 strides; it is not a change to live task capacities.

## Saved native results

The existing restored-input replay measures the complete original/candidate
owner, including unchanged production/whitening/decode and identical output
restoration. It uses six balanced AB/BA rounds, not task-level timing. Current
and held inputs, immutable buffers, unowned outputs and source pins are checked.

| GPU | Input | Tier | Cases | Original ms | Candidate ms | Pointwise records |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RTX | current | 32 | 322 | 0.047642 | 0.233335 | 1 |
| RTX | current | 48 | 190 | 0.051130 | 0.364657 | 2 |
| RTX | held | 32 | 324 | 0.047135 | 0.758528 | 1 |
| RTX | held | 48 | 188 | 0.050709 | 0.297499 | 4 |
| GB | current | 32 | 329 | 0.050219 | 0.227130 | 1 |
| GB | current | 48 | 183 | 0.054536 | 0.565464 | 3 |
| GB | held | 32 | 329 | 0.050224 | 0.269453 | 2 |
| GB | held | 48 | 183 | 0.054528 | 0.356765 | 4 |

All 2,048 cases pass hard finite, nonnegative-normal, disk-feasibility and
momentum/backward-error checks. The frozen 3e-5 / 1e-7 hard thresholds were not
widened. Eight predeclared first-world CPU translation controls have maximum
scaled held-H velocity error 8.962e-6 on RTX and 1.342e-5 on GB. They are not a
full-cohort exact-trajectory assertion. GB's ten pointwise records are a subset
of its eleven CPU records; no new GB diagnostic ID appeared. Native pointwise
counts and CPU H-reference counts are different tests, not interchangeable.

Artifacts: `/tmp/fpgs-krylov-chord-native-probe-t9F4yYPG/{rtx,gb}-probe.log`.
Complete replay is 4.90–16.09 times slower on RTX and 4.52–10.37 times slower on
GB. No replay ratio is extrapolated into a whole-task result.

## Intrinsic work and measured diagnosis

The CPU product proxy was a genuine reduction: 34,318,395 counted products,
34,611,197 after known lower-bound additions, versus 84,268,494 original EX1 plus
recurrence products. Both omit retained geometry/whitening/decode and neither is
a cycle model. The same CPU trace already carries 98,862 recorded norm/dot
reductions plus 11,721 physical scale reductions, versus 49,059 original restart
dot reductions. Divisions, square roots, branches and synchronization are not
equivalent to those retired products.

Original iterations perform one response action, projection, restart reduction
and cheap change vote. They do not assemble active sets, build/factor local
blocks, execute Arnoldi orthogonalization, or evaluate full-map Armijo trials.
For q Krylov steps, the linear region alone requires roughly
`2 + q*q + 3*q` dot/norm reductions, plus projection shuffles and warp fences.
The native active packing is lane-zero serial; Gram construction is cooperative,
and local blocks are factored by one leader each. Tier48's second warp waits
during the warp-zero linear solve. Full-row maps, physical checks and operators
still require CTA communication outside that region.

The preserved GB CPU populations illustrate the tails (maxima can be different
worlds):

| Input/tier | Sum outer / q / line trials | Max outer / q / q-squared / line trials |
| --- | --- | --- |
| current32 | 1,177 / 1,422 / 1,236 | 7 / 28 / 150 / 29 |
| current48 | 756 / 1,442 / 1,231 | 15 / 80 / 490 / 97 |
| held32 | 1,170 / 1,428 / 1,272 | 8 / 35 / 197 / 32 |
| held48 | 713 / 1,201 / 990 | 13 / 42 / 242 / 81 |

Current48 world494 already needs 650 orthogonalization reductions in the CPU
policy. Thus mean q≈1.44 per direction and mean 3.72 committed candidate steps
do not bound a world's critical path. RTX held32 world224 already needs 24
candidate corrections, 104 Krylov steps, q-squared 474, 212 line trials and 682
orthogonalization reductions in the CPU policy. Its replay partition loses
0.047135 → 0.758528 ms. The 512-world replay also has only 183
active GB tier48 CTAs for 152 SMs; tail sensitivity is plausible, not a measured
stall explanation or a prediction for 16K worlds. The measured 16K complete
boundary independently confirms the loss, so small-cohort tail sensitivity is
not an adequate explanation of the full result.

The external diagnostic in `/tmp/fpgs-krylov-diagnostic-HKxyZtZT` records outer,
q/q-squared, Gram, line, physical/fresh, fallback and clock phases. Source
stripping restores exact production bytes and ABI checks pass. RTX diagnostic
66709 completed with bit-identical velocity, impulses and all three metadata
outputs across 1,024 cases. A per-world join against the frozen CPU trace finds
zero mismatches in committed corrections, linear calls, q, q-squared, Gram
builds, projected trials, original sweeps or total line trials. Native totals
are 3,809 / 3,823 / 5,468 / 11,792 / 1,107 / 186 / 283 / 4,676 respectively.
No FP32 iteration-count explosion, changed fallback budget or admission shortcut
explains the cost loss.

Instrumented within-world elapsed cycles aggregate to preparation 26.807%,
linear 38.335%, map/physical 9.255%, line 7.801%, original continuation 0.535%,
staging 12.654% and decode 4.613%. Preparation includes admission/step setup,
active-set grouping, Gram/cache work and inactive-coupling preparation; linear
includes local preconditioner factor/apply, Gram Jv, CGS2, Givens/backsolve and
true forcing checks. The top 1% / top 5% of worlds account for 4.04% / 11.39% of
summed instrumented cycles: tails matter, but the population-wide service cost
cannot be attributed only to a few difficult worlds. The linear source ledger
alone gives 35,842 dot/norm reductions, or 179,210 shuffle stages, for the measured
RTX trajectory, before full-row outer reductions.

Instrumentation raises registers to
168 for tier32 and 167/168 for tier48; even successful phase fractions are
diagnostic only, not production wall-time shares or a hardware-stall diagnosis.
The supported cause is intrinsic control and linear-service overhead, not a
failure to reproduce the CPU outer trajectory. No isolated packing,
coefficient-cache or mapping change has a demonstrated complete saving sufficient
to recover the 18.54050605 ms whole loss and then deliver the 1 ms target. There is
no additional performance variant or policy grid in this checkpoint.

## Whole-task boundary

The first GB attempt, `/tmp/fpgs-anymal-krylov-chord-whole-gb16k-20260916-01`,
stopped at the idle guard before any capture (`runs=[]`): an unrelated Kit process
occupied GB after preparation. Source guard passed; idle guard failed. No process
was killed and no timing exists. It preserved original ca0d baseline, seed0,
16,384 worlds, 200 warmups, 40 measured/profile steps, dense capacity72,
raw212,992 and broadphase294,912. Only single-GPU cardinality was admitted to
reuse an available device; no physics or timing recipe changed.

RTX subsequently became available. Root completed
`/tmp/fpgs-anymal-krylov-chord-whole-rtx16k-20260916-01` at 23:46 UTC, before the
00:30 native/early-cost checkpoint. Parent 92062 exited 0 and all source, idle and
capacity checks passed. Original physics 9.222293025 → candidate 27.762799075 ms
(0.332181672×); unprofiled environment wall 17.281787 → 34.908291 ms. This is an
actual complete-boundary loss, not a replay extrapolation. The entire native tree
is preserved as a default-off, unpromoted experiment. No promotion,
cross-task speedup, hardware bottleneck claim or additional policy variant is
supported by the current evidence.

## Source and evidence pins

SHA256 values below identify measured bytes, not merely the uncommitted branch:

| File | SHA256 |
| --- | --- |
| `krylov_chord.py` | `bca38cf5a2b624533f59f478d564831f1655a976d6bf89efb21b1974b8b2e0de` |
| `krylov_linear.py` | `d68534d5a8abe82901f63707d3d27a661296aec5344dcf75fa3b1debab964249` |
| `solver_feather_pgs.py` | `678cef63949e47879f3b5b34d3ba5271a0ff4f466878415da85c23321184aed2` |
| `test_krylov_chord_native.py` | `221ed39f13411da346117e2e6487a68a6aff38b689c16124c8350ec9b13df56b` |
| `test_krylov_linear.py` | `678c8106c8074b54887ca88ffbfd937bfc260eb38bf7ed6cccfe6f9c5a86482d` |
| frozen CPU helper | `05d97ac7c04e13b835235ceaf1cf2e9d2c97ff2d7191d58852f85f9aff5b269b` |
| capture `checked_capture.py` | `be0222b7b1d14219a8f9056b375929df6e91eb2f987784ffa16544e1d2031abb` |
| capture `run.py` | `da92b6ee90c54b1566a39f7443d6ae9785ba7cf7ccfa761ffb98321c9c9cfcee` |
| capture `nsys_checked.sh` | `0e5c5b3153be3be947c8fa3f297bec20fd1a18b8e4a604a73ae3a2a93d7bb30f` |
| full-owner `offline02/report.json` | `c224ce8c1a2fa9a063699704cd5cd8f52acdc800db53129592f3391b9bebfaee` |
| RTX probe log | `a94cd4be98d0863b7e55be1057dd3a08aea7368298ace370910a2afff1a86372` |
| GB probe log | `c37d992706d3cac5d1a4699e3b6e8d825c35a607a19498de5958b95eeae0e4e1` |
| blocked GB manifest | `53374f6116da3737af61676d53afd08f9832fe8605adc095d0266897352a5131` |
| completed RTX manifest | `23f76e015d80d0a24b072d0adb0c667f2ee874ea44584ca21754e1d8b6207f50` |
| diagnostic `rtx01/result.json` | `4de29b0310b6f2c3563905ca90bd17d4b1c4932cc44afc66541365f25ca50107` |

Full-owner AOT root: `/tmp/fpgs-krylov-owner-offline-bjJGDrjL`.
Frozen corrected CPU trace:
`/tmp/fpgs-krylov-chord-safeguard-HV6mVO8t/all2048/result.json.zlib.base64`
(`815d93d4e78865c9c23bcd8f463a47898d29a21dc89c5f9601836673208b837a`).
No measured runtime/test/observer bytes were edited while preparing this report.
