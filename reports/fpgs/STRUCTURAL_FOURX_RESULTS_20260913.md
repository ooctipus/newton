# Structural window: integrated discoveries

Checkpoint: 2026-09-13 12:50 UTC. Work continues toward the 24-hour window in
`STRUCTURAL_FOURX_20260913.md`; the 4x target is **not achieved**. These are
single-round discoveries, not replacements for accepted repeated baselines.
No Isaac Lab source, timestep, substep or iteration allowance was changed.

## Complete physics results so far

Times are milliseconds per batched environment step, including all four
physics graphs. They are means over 40 profiled steps following 200 warmup
steps, with 16,384 worlds and seed zero. Independent 40-step synchronized
environment-wall windows are separate; neither measures full RL training.

| Candidate | RTX accepted / candidate | RTX speedup | GB accepted / candidate | GB speedup |
| --- | ---: | ---: | ---: | ---: |
| G1 single-factor native response | 43.409484 / 41.665184 | 1.041865x | 72.285541 / 70.348522 | 1.027535x |
| G1 current-height cell rejection | 43.513016 / 37.867363 | 1.149090x | 72.398981 / 52.165458 | 1.387872x |
| Kuka complete live kinetic path | 12.778658 / 11.274877 | 1.133374x | 12.183984 / 11.740153 | 1.037805x |

These candidates are separate, not stacked. Do not multiply their speedups.
Physics values come from graph profiling, not sums of separately optimized
sections. The single-factor experiment's first implementation was slower;
node attribution and one targeted response-kernel replacement recovered the
small gain shown. It remains default-off and below the structural milestone.
See `G1_SINGLE_FACTOR_20260913.md` for the complete loss diagnosis and controls.

Cell rejection removes separated heightfield-triangle work before expensive
contact queries; it adds current corner-height reads but no cache, queue,
buffer or launch. Its environment wall times improve 58.465564 to 50.948728
ms on RTX and 85.902860 to 67.300094 ms on GB. Seven focused tests pass on CPU
and each GPU. Two actual 96-pair convex-mesh fixtures on each GPU, including
three graph replays, preserve complete bidirectional contact tuples: 700/878
contacts with zero observed point/distance/normal error. Balanced live repeats
and broader physical/behavior qualification are still pending.

Kuka's complete path now covers real eager calls, per-world gravity changes,
reset-before-first-graph-replay and live MF rows. Its graph-reset failure was
traced by memcheck to private banks allocated during capture, then corrected
by constructor allocation of the exact two state/two directed-call banks.
No extra physics call or Lab workaround was added. However, synchronized
environment wall time **regresses**: 36.684009 to 37.468095 ms RTX and
35.243540 to 38.937888 ms GB. Increased reset host ranges are a correlation,
not disjoint causal attribution. Both the regression and small GB physics
gain are under investigation. Boundary status/finite checks do not establish
live contact convergence or per-replay fallback counts.

## Corrected shared-collision MJWarp comparison

The terrain change is generic Newton collision work. A fresh matched run
enables it on **both** FPGS and corrected MJWarp, on the same candidate commit.
It speeds up MJWarp too; do not use the older, slower MJ denominator.

| Device | FPGS physics | Corrected MJWarp physics | MJWarp / FPGS |
| --- | ---: | ---: | ---: |
| RTX PRO 6000 | 37.881833475 ms | 45.697852650 ms | 1.206326x |
| GB300 | 52.083977625 ms | 57.559506800 ms | 1.105129x |

Environment wall times are 52.234619399 / 59.487608250 ms on RTX and
66.320000250 / 72.140569401 ms on GB (FPGS / MJ). This is one fresh round,
not a repeated promotion result. G1 still needs major FPGS-specific work
elimination, as well as further shared collision improvement, to approach 4x.
Other tasks' historical corrected-MJ ratios are not silently refreshed by
these new G1/Kuka FPGS-only discoveries.

## Exact sources and completed owners

All runs use fixed Lab backend
`53ee6b44c2334341305dbdf385a3916c6b140799`, retaining prepared core/tasks/venv
`1d8feb82d17dbfab8f0772de56f84deae2cb7974`. Startup checks verify actual
Newton and fixed manager import paths. Original worktrees and captures remain
unchanged. Each listed parent and its children are reaped, with source and
GPU-idle checks passed. Capacity/sticky-overflow checks pass at all measured
boundaries; this is not a claim of complete trajectory equivalence.

- Accepted Newton: `50dfa28d3aabe51f1b5b75721450efac2c35c5f1`.
- Single-factor native response: `a4093d57`, subsequently documented with an
  additional unsupported-partial-warp admission guard in `0ad92797`. The
  experimental root branch through `0ad92797` is pushed to `ooctipus/newton`.
  Live parent `/tmp/fpgs-g1-single-factor-live-paired16k-20260913-02`.
- Heightfield candidate: `a994b1b24b6073dcb10e855c10e8197ee116ef14`, isolated
  `ooctipus/g1-heightfield-cells-20260913` branch; not yet promoted or pushed.
  First A/B parent `/tmp/fpgs-g1-heightfield-cells-live-paired16k-20260913-01`,
  manifest `04d9b8f632e84677c577dc4516f464b362fafbd2f2258dac69b3fc2bfb283b7b`.
  Shared-MJ parent `/tmp/fpgs-g1-heightfield-cells-shared-mj-paired16k-20260913-01`.
  Physical parent `/tmp/fpgs-g1-heightfield-cells-checks-paired-20260913-01`,
  manifest `eed8c778a64dd85e97d49632837612352f888d48d1fb1d76502ef71b4cf44e18`.
- Kuka candidate: `4f76552da4f8966f96c0d2270168613142c9bd14`; benchmark-only
  current-owner observers `c9b06e9e1b86c7afbe6e34c69c17944291db9404` in a fresh
  benchmark worktree. Parent
  `/tmp/fpgs-kuka-kinetic-live-discovery-paired16k-20260913-01`, manifest
  `365d1db9fc8a254af28e7395461f030a63da30e3b3f4a049e29b2923fb9fa1fa`.
  The 25 unique pinned source/artifact files were independently rehashed.

All feature modes remain explicit/default-off. Next bounded work is Franka's
complete contact-path CUDA physical gate, balanced terrain repeats, Kuka
whole-path/reset attribution, and a G1 sparse tree-factor feasibility study.
The latter's operation counts are a hypothesis, not a measured gain.
