# Fresh corrected backend table — 12 September 2026

Requested quick refresh of the four-task comparison. This is one discovery
round, not repeated performance or solver/trajectory-equivalence acceptance.
Every task uses 16,384 worlds, seed 0, 200 warmup steps, 40 synchronized
environment steps and 40 profiled environment steps. Both cards start together
for each backend; backends run sequentially with one owned process per GPU.
All task timesteps, substeps and original backend iteration allowances remain
unchanged. No Isaac Lab code or configuration objects are edited.

## First three tasks: complete, checked

Physics milliseconds per batched environment step; ratio = MJWarp/FPGS.

| Task | GPU | FPGS ms | Corrected MJWarp ms | Ratio |
| --- | --- | ---: | ---: | ---: |
| Franka | RTX PRO6000 | 6.246390100 | 10.708024775 | 1.714274x |
| Franka | GB300 | 5.793840575 | 10.348531100 | 1.786126x |
| KukaAllegro | RTX PRO6000 | 15.769091000 | 27.464576525 | 1.741672x |
| KukaAllegro | GB300 | 15.484580000 | 25.954643650 | 1.676161x |
| ANYmal-D | RTX PRO6000 | 9.937255750 | 36.641471700 | 3.687283x |
| ANYmal-D | GB300 | 10.164583775 | 42.350414100 | 4.166468x |

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

## Allegro: pending the compact-capacity timing capture

Use the actually implemented collision carry-forward source `fba9fead`, not
the handoff source above: the handoff contains C's report, not its runtime.
The benchmark-only compact-capacity adapter reuses the already validated
286720 public-contact / 524288 broad-query recipe while preserving sparse GJK
and the original worker grid. It adds construction and existing-boundary
checks only, with no per-step diagnostics. No Allegro timing is claimed until
the new paired capture and its strict checks complete.
