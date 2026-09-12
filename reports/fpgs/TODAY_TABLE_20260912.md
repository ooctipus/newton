# Fresh corrected backend table — 12 September 2026

Requested quick refresh of the four-task comparison. This is one discovery
round, not repeated performance or solver/trajectory-equivalence acceptance.
Every task uses 16,384 worlds, seed 0, 200 warmup steps, 40 synchronized
environment steps and 40 profiled environment steps. Both cards start together
for each backend; backends run sequentially with one owned process per GPU.
All task timesteps, substeps and original backend iteration allowances remain
unchanged. No Isaac Lab code or configuration objects are edited.

## Fresh physics measurements

Physics milliseconds per batched environment step; ratio = MJWarp/FPGS.

| Task | GPU | FPGS ms | Corrected MJWarp ms | Ratio |
| --- | --- | ---: | ---: | ---: |
| Franka | RTX PRO6000 | 6.246390100 | 10.708024775 | 1.714274x |
| Franka | GB300 | 5.793840575 | 10.348531100 | 1.786126x |
| Allegro | RTX PRO6000 | 17.210517325 | Rejected before timing | — |
| Allegro | GB300 | 18.087479375 | 82.142188050 | 4.541384x |
| KukaAllegro | RTX PRO6000 | 15.769091000 | 27.464576525 | 1.741672x |
| KukaAllegro | GB300 | 15.484580000 | 25.954643650 | 1.676161x |
| ANYmal-D | RTX PRO6000 | 9.937255750 | 36.641471700 | 3.687283x |
| ANYmal-D | GB300 | 10.164583775 | 42.350414100 | 4.166468x |

Franka, Kuka and ANYmal-D form a complete checked batch. Allegro RTX MJWarp
failed its post-warmup warning check; there is no RTX MJ timing or ratio.
Allegro GB is an independently checked complete device pair inside a failed
paired-device batch, not acceptance of that whole batch. Details follow below.

These replace the older warning-affected comparison, including Kuka's apparent
11.05x RTX ratio. They do not establish how much of the old 170.364-ms sample
was failed line search, warning overhead, or changed capacity/dispatch. This
refresh uses the corrected search and calibrated MJ capacities together.

Both backends and the driver use Newton
`42f1492ab99e2e8eedd157bd9893ed9b2fd4de61`; Isaac Lab remains
`1d8feb82d17dbfab8f0772de56f84deae2cb7974`. FPGS retains the previously
reported task recipes, including Kuka's `FEATHER_PGS_SIMPLE_WORLD_ZERO=1`
(the same flag has no supported dispatch on the other two tasks). This is not
a new optimization-flag search or new FPGS capacity calibration.

MJWarp explicitly uses the process-local `--mjwarp-linesearch-fix` helper,
SHA256 `7506361f334566dfb51a3d46880bf7a4f919c1fd45513fcfa44f7a4d3e60a6eb`.
The installed package is unchanged; iteration limits and tolerances are not
increased and warnings are not disabled. Explicit MJ capacities:

| Task | Rows/world | nconmax | Newton pipeline contact request | Broad output |
| --- | ---: | ---: | ---: | ---: |
| Franka | 56 | 1 | 2048 | 7680 |
| KukaAllegro | 84 | 19 | 311296 | 442368 |
| ANYmal-D | 32 | 9 | 147456 | 294912 |

Franka's actual public/MJ contact pool is 16384 despite the smaller pipeline
request. Other MJ pools are nconmax times 16384. FPGS retains its prior
allocations for these three tasks; do not describe them as newly right-sized
or apply the MJ calibration to FPGS implicitly.

All 12 captures exit successfully and pass the two existing untimed metadata
checks: zero available solver warning/capacity bits, zero 13 Newton narrow-phase
flags and finite state. An independent read-only audit recomputes all 480
physics samples from correlated graph spans, checks 24 metadata boundaries,
verifies source/driver hashes, and confirms the 12 recorded processes are gone.
The root parent is reaped and both GPUs are compute-idle afterward.

Warning/capacity cleanliness does not establish numerical equivalence. The
documented elliptic numerical-tail qualification for ANYmal-D remains. Physics
graph ratios are not whole-environment wall ratios or RL training throughput.

Artifacts include exact runnable commands and realized settings:

- `/tmp/fpgs-today-three-task-20260912-01/manifest.json`, SHA256
  `9600ce35ffcd7cf87162b9f8c7f7ca9e042f9cf3ff470707df5a57c83bf76d57`.
- Sibling `ratios.json`, SHA256
  `87be4d48e57999bc36b3805c11a691962db85ff8165ac0a594057c0e708ca621`.
- Per-run `capture_analysis.json`, `capture_checks.json`, original Nsight
  captures and separate synchronized environment wall timings are retained.

## Allegro: GB pair checked; RTX MJ warning retained

FPGS uses the actually implemented collision carry-forward source
`fba9fead70d17728f842954d66cf1f05f4617d40`, not the handoff runtime above:
the handoff contains C's report, not its runtime. MJWarp uses the same corrected
`42f1492a` source/helper as the other tasks. Benchmark tool commit is
`85fdedae` on `ooctipus/fpgs-today-bench-20260912`.

The benchmark-only `--allegro-compact-capacity` adapter reuses the validated
286720 public-contact / 524288 broad-query recipe while preserving sparse GJK,
the original worker grid, full input pairs, C cache-key domain and row budgets.
It adds construction and existing-boundary checks only, with no per-step
diagnostics. C is explicitly enabled by
`NEWTON_NARROW_PHASE_COHERENT_CONVEX=reject_only` on both FPGS children.
MJWarp retains native collision, `njmax=112`, `nconmax=22` and the resulting
360448-slot shared native pair/contact pool; no Newton collision cap is applied
to that backend. The two backends do not share collision ownership on this task.

Both FPGS captures pass with exact requested public/query allocations and
unchanged dispatch, zero row/narrow overflow flags and finite state. GB MJWarp
also passes both checks with zero warning/capacity flags. Independent audit
executes the actual parent's `check_overflow_result` and original Lab
`_read_result` on these three completed children, recomputes all 120 samples,
and verifies six successful metadata boundaries, all source/driver hashes,
constructor restoration, recorded process cleanup and both idle GPUs.

RTX MJWarp instead records **mask 1024, LS_ITERATIONS in one world** at its
first post-warmup boundary and exits before timing. There is no `capture.json`
or valid physics sample for that arm. This is a new observed exhaustion case
outside the earlier warning-free seed-1 calibration; its physical harm and
cause have not been diagnosed. Do not call it harmless or claim the line-search
correction prevents all future warnings. The iteration allowance was not raised,
the warning was not suppressed and no seed/retry was selected to hide it.

The root parent is reaped with exit 1. Its manifest remains **failed**, with no
parent `ratios.json`; the successful GB pair above is explicitly a partial
result, not a rewritten successful batch. Its separate environment wall times
are 26.085375 ms FPGS / 90.230826 ms MJWarp (3.459058x), not the 4.541384x
physics ratio. RTX FPGS wall time is 25.286839 ms, with no MJ counterpart.

The adapter's missing-option regression failed before implementation. Seventy
CPU tests, including actual pinned Newton constructors and the real Lab recipe
forwarding, pass independently twice. Eight shared variant-driver tests and
23 actual CPU MJWarp helper tests also pass. One root test invocation from the
wrong working directory correctly rejects the wrong Newton source; repeating
the documented tools-directory invocation imports the pinned FPGS source and
passes. Full pre-commit passes. No Newton runtime or Isaac Lab source is changed.

Artifacts:

- `/tmp/fpgs-today-allegro-20260912-01/manifest.json`, SHA256
  `b54049c3435bcf2cb97070e2ab3c96b74938ae131f6ee157b31886b7992372f4`.
- GB MJ `round_01_allegro_mjwarp_gpu1/capture_analysis.json`, SHA256
  `02a5fac124aa7c543eb5b4183c569102f0db3a9dfc631fe6b5454ed4e586ba0b`.
- RTX MJ `round_01_allegro_mjwarp_gpu0/capture_checks.json`, SHA256
  `81eb4cff2057c43d9b65c2bf1196baab9bcbf976d2ec02923c65661b11fbe87d`.
- Capacity helper SHA256
  `b1b5143021d06144bc52fa3821b94036a6d7fbd8c4449994180b2056c888d171`.

Exact commands are retained in both manifests; the Allegro invocation and
constructor-test command are also in `tools/fpgs_bench/README.md`.
