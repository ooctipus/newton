# Current-light integration and complete late publication

This is a default-off, **unpromoted** Kuka experiment. The target remains
Kuka Allegro and Franka at least 4× corrected MJWarp whole-physics throughput.
No new measured gain is claimed at this implementation checkpoint.

Branch `ooctipus/fpgs-composed-world-20260912` starts at preserved
`fe639895ea6a1ddc4b7393c103c5c17fa6f08f2d`. It retains the original handoff,
accepted `064ec8ac455fc4cde557a3b54a1a62624cf56441` progress, the joint-world
experiment and the standalone scan in its ancestry. Push target is exclusively
`ooctipus/newton`; no PR or dependency-pointer change is authorized.
Isaac Lab stays clean at `1d8feb82d17dbfab8f0772de56f84deae2cb7974`.
Timestep, substeps, decimation, eight GS sweeps, zero additional velocity
passes, held-mass interval and public capacities stay unchanged. The inherited
capacities pass the existing checks; they have not been newly right-sized.

## Accepted reference remains unchanged

Milliseconds per 16K environment step; physics is not full training throughput:

| Task / GPU | Accepted FPGS | Corrected MJWarp | MJWarp/FPGS |
| --- | ---: | ---: | ---: |
| Kuka RTX PRO6000 | 13.973622 | 27.464577 | 1.9655× |
| Kuka GB300 | 13.884428 | 25.954644 | 1.8693× |
| Franka RTX PRO6000 | 5.746087 | 10.708025 | 1.8635× |
| Franka GB300 | 5.195039 | 10.348531 | 1.9920× |

Accepted FPGS uses three balanced rounds. MJWarp is the earlier-today corrected
single-round discovery reference, not a fresh repeated accuracy-matched run.
The old warning-heavy Kuka 170 ms/11.05× comparison remains withdrawn.
The 4× ceilings are Kuka 6.866144/6.488661 ms and Franka 2.677006/2.587133 ms
(RTX/GB). No experiment below achieves that objective.

## Standalone scan: completed measured miss and causal diagnosis

The predecessor report describes its implementation checkpoint. Subsequent
paired complete runs at unchanged `fe639895` passed source/idle guards,
finite checks and all four solver/thirteen collision flags:

| Worlds / GPU | Accepted064 baseline | Standalone scan | Speedup |
| --- | ---: | ---: | ---: |
| 512 / RTX | 4.811160 | 4.352506 | 1.1054× |
| 512 / GB | 5.340021 | 4.910005 | 1.0876× |
| 16384 / RTX | 14.008275 | 13.493651 | 1.0381× |
| 16384 / GB | 14.499741 | 13.113686 | 1.1057× |

These are one-round, three-node-step screens, not repeated promotion evidence.
At16K complete publication exclusive cost fell 2.233227→1.327295 ms RTX and
1.907925→.955520 ms GB: .905932/.952405 ms savings, about 6.5% of matched
whole physics on each card. The replacement itself misses its 10% whole-time
budget by .494896/.497569 ms. Whole-environment wall time was roughly unchanged.

Disjoint additive attribution of complete graph changes (ms):

| GPU | Publication | Other device union | Uncovered span | Whole change |
| --- | ---: | ---: | ---: | ---: |
| RTX | -.905932 | +.401727 | -.010420 | -.514625 |
| GB | -.952405 | -.433143 | -.000507 | -1.386055 |

Changed current MF tails explain much of the other-device difference:
paired/general union rose .271574 ms RTX and fell .424277 ms GB. This is not
an attributable .93 ms RTX solver regression from summing overlapping kernel
durations. Different contact counts alone are not a physical-parity failure.

To explain the expensive replacement, root captured actual16K late-publication
operands at substeps1600/1601 on both GPUs, including all15 outputs, exact
descriptor bindings and real damping .05. Capturing arrays was untimed and
is not wall-performance evidence. An exact-source five-clock adapter then
replayed the same inputs with 20 alternating event-timed original/traced pairs.
All15 outputs were byte-identical; inputs and plans immutable. Instrumented
event/original ratios were .9878/1.0052 RTX and .9915/.9925 GB. Loaded shared
2624 B, local72 B and scheduling bound24 CTAs/SM were unchanged.

Summed CTA cycle shares were generalized 39.5–42.5% RTX /42.7–44.8% GB,
pose 16.9–18.1% /20.6–21.7%, motion 11.5–11.9% /13.2–13.3%, and finalizer
28.0–31.7% /20.3–23.3%. These are **not additive whole-GPU elapsed fractions**.
They identify duplicated generalized work as a substantial candidate, not a
measured savings promise. Standalone four-world body packing compiled at64
registers but only increased the resource-bound world-warps24→32. Even an
optimistic phase-weighted estimate missed the complete .83 ms budget. It was
not implemented in production. COM-only finalizer algebra was also checked
but lacks the whole-time ceiling; no micro-tuning sequence was authorized.

## Composition and ownership

Enable the common accepted ZERO/independent-components/paired-overlap/local-row-
packet flags, plus both `FEATHER_PGS_KUKA_JOINT_WORLD=1` and
`FEATHER_PGS_WORLD_SCAN_PUBLICATION=1`. Unlike the standalone predecessor,
this branch explicitly supports their guarded composition.

Keep the existing raw CSR, light predictor, current ZERO classifier, active
response, original row preparation and eight-sweep solve unchanged. Only
the **current** `_joint_world_active` handoff permits the late kernel to skip
qdd conversion, free-root transport removal and generalized integration for
resolved worlds. The ordinary classifier writes the same resolved array
without integration, so stale mask bits or merely constructing an owner
never permit the skip. The unselected complement performs the original
generalized law. Every world still performs complete late pose/S/V/A,
inertia/bias/COM-velocity/cache publication after the existing solve joins.

The standalone two-argument kernel source is unchanged. A checked source
adapter adds one mask operand and wraps exactly the three generalized loops
and joins; all body statements remain outside. The controller reuses the
light owner's validated static host/device maps, adds no allocation, stream,
event or per-step readback, and preserves all alias/gradient/numeric-notify
and original next-refresh guards. Rejected states retain `finish_active`
and original all-world FK/finalization. Cache source identity changes only
after successful complete publication; current ownership clears afterward.

No CSR/dynamics contention is credited as eliminated. In the light candidate's
own16K window, remaining late-publication exclusive cost is2.270583 ms RTX
and1.798515 GB. With every other light cost held fixed, a new late owner
≤1.399220/.947986 ms would reach10% versus that window's accepted064;
≤.878405/.412077 would instead be needed for10% over the unpromoted light
candidate itself. The phase-weighted .894–.925 RTX /.630–.645 GB hypothesis
is unmeasured and not a GPU-span prediction. A direct fresh complete comparison
against accepted064 decides whether the combination makes progress.

## Initial validation and next gate

Regression-first source and host dispatch tests failed before implementation.
The eleven host tests and four source-adapter tests now pass with CUDA hidden.
Offline original→resolved registers are77→76 on sm120 and79→77 on sm103;
both retain2624 B shared,72 B stack and zero spills. No occupancy gain is
inferred. Independent all15-field same-input/current-mask numerical tests
then passed root-owned execution on both actual GPUs. Each initial77-test
combined suite passed76 with one older explicitly device-selected graph test
skipped; rerunning its nine-test owner module with `FPGS_TEST_DEVICE=cuda:0`
passed all nine on each card, including that graph control. Initial durations
were35.696/35.737 seconds; supplemental1.146/1.128 seconds (RTX/GB). Some
registered controls are CPU tests, not distinct GPU kernels. All children were
reaped and compute-idle verified. Logs are
`/tmp/fpgs-composed-world-native-{rtx,gb}-20260912-01.log` and
`/tmp/fpgs-composed-world-owner-native-{rtx,gb}-20260912-01.log`.
This establishes component ownership/numerical checks, not long-rollout or
backend physical parity. Whole-physics timing remains pending.

Root owns one child per GPU and paired512 then16K integrated comparisons.
Source pins remain clean until both jobs are reaped and idle checks pass.
A credible complete gain proceeds to balanced repeated40-graph timing, plus
current physical/reset checks. Missing the checkpoint requires causal analysis,
not a packing/lane sweep or an unqualified speedup claim.

## Evidence pins (large reference captures stay local)

- Composition card `/tmp/fpgs-world-scan-composition-yFgnLwNO/CARD.md`,
  SHA256 `aec789ef02a32e7a8d487fa946f47fdee986807ca733c8b9a3cafc6bb54dbcd1`.
- Standalone512 parent `c30ef5e9c3e1f1758a2b85dd438d5044374e9d535da253bdf0578e932585da98`;
  standalone16K parent `6da283b661bc7aeea67cfe005021b89d076596f028ea084281cef60f6663d1a4`.
- Node evidence `/tmp/fpgs-world-scan-node-audit-1KjQcyW3/evidence16k.json`,
  `883d23ea5b13b10e1549a48349151311dbc25824895c9085d3ca4f32d881582c`.
- Current publication capture parent `/tmp/fpgs-world-scan-current-paired16k-20260912-01/manifest.json`,
  `e93a2d7d3a6f4c34449d9c8adb0177425bf29ed3b72b614b8889e0b6663259ec`.
- Phase parent `/tmp/fpgs-world-scan-phase-paired16k-20260912-01/manifest.json`,
  `37afb94d41a350fbde1f5dd3cf90c8582a252e1bcc22d831bd8c70d7911807c9`.
- Standalone packing NO-GO `/tmp/fpgs-world-scan-body-0XF0DBmR/CARD.md`,
  `9e3e84067a0e599d9dbf03290b13c3d7be29afbd1786c3d59857461200feae23`.
- Composition offline resources `/tmp/fpgs-world-scan-composed-ready-aaRz2mpm/offline01/report.json`.
- Accepted progress and reproduction: `FOURX_PROGRESS_20260912.md`; original
  handoff and workstream rules: `CROSS_TASK_20260911.md` and inherited reports.
