# Repeatable Kuka gain; the 4x objective remains open

This report-only successor preserves runtime `50dfa28d3aabe51f1b5b75721450efac2c35c5f1`
on branch `ooctipus/fpgs-fourx-progress2-20260912` in the `ooctipus/newton` fork.
The original handoff, accepted064 progress and all intervening diagnoses remain
in the ancestry. Original worktrees are unchanged. Isaac Lab remains clean at
`1d8feb82d17dbfab8f0772de56f84deae2cb7974`; no Lab source, task settings,
substeps, eight-sweep allowance, dependency pointer or capacity changed.

The current Kuka path is a **repeatable opt-in physics improvement**, not a
default-on release, proof of MJWarp physical parity, or achievement of 4x.
The separate Franka kinetic-cache prototype is not included in this runtime.
The subsequent [complete private-state experiment](FRANKA_PRIVATE_STATE_20260912.md)
also remains scratch: numerical controls pass, but its measured owner cost
does not qualify for the proposed large whole-physics gain.
Its [cooperative successor](FRANKA_COOPERATIVE_STATE_20260912.md) passes bounded
physical controls and improves that complete private-owner component by
1.25296x RTX / 1.17312x GB in a matched window. Missing live/fallback services
remain unmeasured; none of these scratch numbers updates the task table below.

## Balanced paired measurements

Three rounds AB/BA/AB, both GPUs concurrently, 16,384 worlds, seed0,
200 warmup steps,40 synchronized environment steps and40 physics-graph steps
per run. Twelve children completed and were reaped; all source/idle, finite,
four solver and thirteen collision checks passed. The independent audit
recomputed1,920 raw graph records and24 before/after metadata boundaries.

Milliseconds per batched environment step:

| Task / GPU | Previous accepted FPGS | Current measured FPGS | Additional speedup | Fixed corrected MJWarp | MJWarp/current |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kuka RTX PRO6000 | 13.957654 | 12.862500 | **1.08514x** | 27.464577 | **2.13524x** |
| Kuka GB300 | 13.771896 | 12.177994 | **1.13088x** | 25.954644 | **2.13127x** |
| Franka RTX PRO6000 | 5.746087 | unchanged | — | 10.708025 | 1.86353x |
| Franka GB300 | 5.195039 | unchanged | — | 10.348531 | 1.99200x |

Kuka's baseline column is the new matched064 median, not a retimed starting
baseline. Franka retains its prior accepted three-round number. The MJWarp
denominators are earlier-today corrected single-round discovery references,
not fresh repeated accuracy-matched runs. The old170ms/11.05x Kuka claim stays
withdrawn. Whole-environment throughput is separate and is not RL training.

**Collision-scope correction,19:30 UTC:** both tasks' quoted corrected MJWarp
captures use **external Newton collision**, as does FPGS. The unchanged lift
configuration explicitly sets `use_mujoco_contacts=False`; the MJWarp manager
therefore sets `_needs_collision_pipeline=True`. More importantly, all eight
saved MJ boundary checks read the actual solver flag and report
`external_newton_prefix`, with conversion/collision/warning checks clear.
An earlier scratch Franka study incorrectly used the generic native-contact
default instead of the task override. Missing collision child-kernel names
in these `graph:host-only` traces do not prove collision absence: none of
their graph child nodes are individually traced.

The independent exact-reference audit is
`/tmp/fpgs-corrected-mj-scope-3o6CabAP/FINDINGS.md`, SHA256
`3f8ee6dc95690381a7adffd86fbc5cf935d7daa7d9f22d68dff646ca39ed8ea8`.
It recomputes all four40-sample means from completed parent
`/tmp/fpgs-today-three-task-20260912-01/manifest.json`, SHA256
`9600ce35ffcd7cf87162b9f8c7f7ca9e042f9cf3ff470707df5a57c83bf76d57`.
Shared collision source does not imply identical evolving contacts or cost.
A general collision improvement enabled for both backends would also require
a newly timed MJ denominator; the fixed historical comparison must stay labeled.

All three Kuka physics rounds improved:

| Round | RTX baseline → current ms | GB baseline → current ms |
| --- | ---: | ---: |
| 1 AB | 13.957654 → 12.560392 | 13.771896 → 12.177994 |
| 2 BA | 14.109428 → 12.862500 | 13.672339 → 12.117067 |
| 3 AB | 13.888974 → 12.877621 | 14.015651 → 12.215608 |

Median physics time removed is7.846% RTX /11.573% GB. This does **not** establish
the proposed10% RTX milestone. Synchronized wall medians32.720166→31.764297ms
RTX and33.521395→32.666721ms GB give only1.03009x/1.02616x throughput.

The fixed4x Kuka ceilings remain6.866144/6.488661ms. Another46.62%/46.72%
reduction in current candidate physics time is required. Franka still needs
approximately half its current time removed as well. Small launch/layout
tuning is not a credible route to closing this gap.

## What changed, and why the forecast missed

The original complete light predictor/current ZERO screen and active response
are composed with late world-scan publication. Only the current active light
invocation's resolved worlds skip already-completed generalized integration.
All worlds publish all current body fields and caches after the original solve
joins. The original eight GS sweeps, collision law and public force conversion
remain. See `COMPOSED_WORLD_20260912.md` for source, lifecycle and numerical gates.

One16K node screen measured14.029917→12.854580ms RTX and
13.825774→12.235723ms GB. Exact disjoint attribution of whole-graph change:

| GPU | Changed-family exclusive | Other device union | Uncovered span | Whole change ms |
| --- | ---: | ---: | ---: | ---: |
| RTX | -1.634110 | +.432095 | +.026678 | -1.175337 |
| GB | -1.939782 | +.293601 | +.056130 | -1.590051 |

Current raw-CSR/dynamics union grows3.008148→3.414029ms RTX and
2.553035→2.729888ms GB; this is charged, not dismissed as noise. Complete late
publication costs1.314998/.829760ms, missing the conditional.9/.63–.65ms
forecast. Paired/general solve union changes only1.156874→1.130807ms RTX and
1.121750→1.199424ms GB. No overlapping durations are added as independent savings.

A same-input probe on actual16K saved publication states then isolated the
generalized skip. It compares unchanged standalone and composed kernels with
synthetic all0/all1/77%-selected masks, correctly seeding selected generalized
outputs from the recorded original result. All15 outputs match exactly;
original inputs/plans/masks remain immutable. Twenty alternating event pairs
per case exclude every restore/readback. No phase clocks or new mapping are used.

On RTX, skipping generalized work for **all** worlds removes only10–12% of
publication elapsed time; mixed77 removes1–2%. GB removes31–32% for all1 and
12–13% for mixed77. All0 is approximately unchanged on both. Loaded scheduling
bound24 blocks/SM, shared2624B and local72B stay unchanged. This disproves use
of earlier40–45% summed per-block phase cycles as whole-GPU elapsed savings.
It supports a mixed-work scheduling/latency hypothesis but does not identify
a particular memory/occupancy mechanism. No small packing/partition sweep is
authorized from this probe; the confirmed combined gain is retained.

## Quality and cross-task scope

Native/source/host controls include actual light outputs, invalid CSR and
nonfinite-inverse fallback, all/no/mixed ownership, deliberately poisoned old
selected inputs, prescribed/free bodies, original damping, all15 canonical
fields, held/refresh epochs, numeric resets and two graph buffers. Physical
geometry controls permit reassociation and use independent FP64/current-state
bounds rather than demanding original FP32 bit identity. Earlier gate failures
and their diagnoses remain recorded in the predecessor reports.

Repeated metadata has no dropped-row/nonfinite/log blocker. Current dense maxima
119RTX/130GB are below192; MF maxima15/12 are below64. All12 candidate boundary
snapshots have complete current ZERO/active partitions and predictor fallback0.
These are current snapshots, not a complete admission or convergence history.
Retain one unresolved scalar tail: RTX candidate round2 before-profile has
`joint_qd_abs_max=8.10656`, versus baseline six-boundary max6.49441. Other
candidate maxima are generally smaller. This mixed-coordinate metadata on
different trajectories neither locates a physical violation nor proves the tail
harmless. No stronger trajectory-equivalence claim is made.

A separate unchanged-recipe Franka16K fallback screen passes all checks:
5.704609→5.814849ms RTX and5.170227→5.193990ms GB. New Kuka owners are ineligible
for the Franka recipe. This single screen establishes no Franka speedup or
statistical regression. Wall is23.455702→23.488596/23.318563→23.915029ms.
Unsupported fallback is not evidence that the Kuka optimization generalizes.

## Reproduction and next structural work

Use clean benchmark tools `32164ded363de6d79e65371f8c89ffad411e8895` with
`tools/fpgs_bench/compare_variants.py`: accepted064 versus this runtime,
`--task kuka --gpus 0 1 --rounds 3 --num-envs 16384 --warmup-steps 200
--steps 40 --profile-steps 40 --trace-mode graph`, and a fresh output directory.
Set the same four flags in both arms:

```text
FEATHER_PGS_SIMPLE_WORLD_ZERO=1
FEATHER_PGS_INDEPENDENT_COMPONENTS=1
FEATHER_PGS_PAIRED_GENERAL_OVERLAP=1
FEATHER_PGS_LOCAL_ROW_PACKETS=1
```

Add only `FEATHER_PGS_KUKA_JOINT_WORLD=1` and
`FEATHER_PGS_WORLD_SCAN_PUBLICATION=1` to the candidate. No capacity or physics
override is needed. The full exact commands and import/source hashes are in the
manifests. `FOURX_PROGRESS_20260912.md` preserves the preceding comparisons.

The Franka experiments now include complete private state and geometry handoff,
current free-body response, generalized integration and next kinetic state.
The earlier .990811 ms allowance applied to a narrower proposal and is superseded
for this expanded family by 1.634704 ms RTX. Even the matched cooperative owner
alone costs 2.072576 ms RTX before live repair/demand/free-factor services, so
the proposed 20% whole-physics gain is not established. The follow-up
[typed free-state owner](FRANKA_TYPED_FREE_STATE_20260912.md) passes full GPU
physical checks but costs2.158656/2.327040ms versus its matched cooperative
2.064448/2.257536ms. Candidate held-free refresh is included while the old
factor refresh is excluded, conservatively; even candidate owner time alone
still exceeds the entire family allowance. It is not promoted. The next study
targets a complete primary dynamics/response/state representation, not isolated
factor or block-size tuning.

Kuka's current-normal certificate family also remains scratch. Its first paired
GPU gate stopped before timing on rare output disagreements (one RTX velocity
coordinate, two GB coordinates; maximum 8.17e-5). Current same-order coefficient,
trial-versus-full-eight and stream-ownership diagnosis subsequently localizes
the differences to J/Y/diag construction in retained rejected worlds. All8649
accepted world-observations exactly match original eight sweeps on their own
coefficients; serial and overlapped outputs also match. Metadata/RHS/selector/MF
coefficients are exact. Completion/source/idle pass is not numerical acceptance:
the diagnostic keeps its original-output acceptance flags false and reports
focus-only physical metrics separately. The subsequent
[physical, complete-cost and node diagnosis](ARROW_AND_NORMAL_DIAGNOSIS_20260912.md)
finds small current-geometry reassociation, then an 8.99% RTX / 11.40% GB
complete-family slowdown. Added exposed transactions and little actual solver
work removed explain the miss. Narrow normal-only corrections cannot meet the
large-gain bound; the family is closed on measured cost, not a bit-identity gate.
The same checkpoint records Franka's complete arrow-state experiment and its
four-arm mass-collector isolation. No new whole-physics gain is promoted.

Evidence (large captures remain local):

- Kuka repeated parent `71fd67f250a1d4517f3cc4b36610abe6a2c7630c366c3fcb31400fd31af483e6`;
  audit `/tmp/fpgs-composed-repeat-audit-Sh5nmclA/FINDINGS.md`,
  `ae455b71` prefix; complete evidence `fb2ac80e966c737b107fca58ba4bdc0f0e09e4df4aad61b404c1163cd6fd90bc`.
- Kuka512 parent `035ac0b8dd17b8786764d4abbd6f46f6f72ccd72814b76abb0e4a7cdb865fe49`;
  Kuka16K node parent `cea5d8df7013ed651aa592e944bd11cd2104d388549f11a2c0cac74273cee43d`.
- Node diagnosis `/tmp/fpgs-composed-world-node-audit-kvaoLH9P/FINDINGS16K.md`.
- Mask-probe parent `5f6e2bb13439d7a8eecadbd2945fdfc0e23be3d28a369566d8a301f5449d0213`;
  wrapper `/tmp/fpgs-world-scan-mask-replay-tFR3ZVbB/run_masks.py`,
  `30f6dc6d39eed31b542c45b1df8f5de134944a2b8bbc0dc3daaa19a3ef725e99`.
- Franka fallback parent `ccc1950f163c4b0629cd53394e87dafaceeb020255651d7742d9a09588833ad2`.
- Franka architecture `/tmp/fpgs-franka-kinetic-cache-card-TPpqTaLU/CARD.md`;
  Kuka next-row NO-GO `/tmp/fpgs-kuka-active-row-card-sL9hPPjS/CARD.md`.
