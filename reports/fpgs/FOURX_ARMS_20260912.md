# KukaAllegro and Franka: 4x corrected-MJWarp target

Active user target, 2026-09-12. This branch starts from `15158f3a`, whose
runtime is the accepted `42f1492a` handoff. The original `31cf87f4` handoff
remains an ancestor. Runtime changes belong only to Newton FPGS/collision;
Isaac Lab source, configuration objects and dependency pin remain unchanged.

## Fixed baseline and acceptance

These are today's checked, **single-round discovery** physics times, not
repeated acceptance or a claim of physical equivalence. MJWarp uses the
documented line-search correction and calibrated non-overflowing capacities.
See [the source-bound table](TODAY_TABLE_20260912.md) for exact provenance.

| Task / GPU | Current FPGS ms | Corrected MJWarp ms | FPGS ceiling for 4x, ms |
| --- | ---: | ---: | ---: |
| KukaAllegro / RTX PRO6000 | 15.769091 | 27.464577 | 6.866144 |
| KukaAllegro / GB300 | 15.484580 | 25.954644 | 6.488661 |
| Franka / RTX PRO6000 | 6.246390 | 10.708025 | 2.677006 |
| Franka / GB300 | 5.793841 | 10.348531 | 2.587133 |

The target is whole physics, not an isolated solve kernel or full training.
Report reset-inclusive environment wall time separately. Keep timestep,
substeps, decimation and effective iteration budget unchanged. Internal
representation/algorithm may change: numerical convergence and physical
behavior matter, not bit-identical finite Gauss-Seidel iterates. Do not relax
capacity checks, overallocate maximum buffers to conceal overflow, or hide
warnings. Final acceptance needs repeated paired-GPU comparisons and quality
gates; the target is not achieved by this plan.

## Work plan

1. Refresh both tasks' node attribution on exact current runtime and flags,
   simultaneously sampling RTX and GB300. Use two identical-source control
   windows to expose measurement variation. Account for interval overlap;
   summed kernel durations are not removable critical-path time.
2. Implement one shared articulated-factor owner for fixed-base 0/1-DOF
   trees. First use the existing articulation-origin spatial frame: parent
   inertia reductions need no per-edge 6x6 frame congruence. Retain original
   current FK, inertia and inverse-dynamics producers initially. Store held
   motion axes and factors on the original mass-update mask, including
   armature and augmented-drive diagonal. Apply the held operator to the
   current force RHS.
3. Service current constraint Jacobians through that same factor, replacing
   selected-arm CRBA, H/L/inverse materialization and old triangular solves.
   Keep original contact/mimic law, iteration budget, free-rigid owner and
   public force/velocity outputs. Franka's local solve must consume the new
   response without secretly retaining its old Cholesky reader. Canonical
   J/Y retention is an explicit transitional cost, not a claimed elimination.
4. Test actual Kuka snapshots and synthetic branched/mimic cases against
   input-scaled FP64 momentum/operator residuals, then current live tasks.
   Cover refreshed and held mass, changing control, full/partial resets,
   model notifications, graph replay and fallback. Compare residuals and
   physical behavior, not merely warning flags or contact counts.
5. Measure the complete integrated path on both cards. An initial goal is
   at least 10% whole-physics reduction, as a meaningful structural step
   toward the fixed 4x target. If measured costs contradict the expected
   saving, retain the failure, attribute it, and permit one cause-directed
   architectural retry. Do not launch lane/block-size grids or accumulate
   sub-percent tweaks. Subsequent body-port/producer elimination must have
   a separately measured complete-cost argument.

## Guardrails against another unproductive search

The normal warm path already reuses FK/ID: deleting an imagined duplicate
is not a saving. Existing propagation also has articulated factors, but
builds them after the dense predictor; enabling it is not this replacement.
Old grouped/fused mappings lost to serialization and duplicated producers.
The new experiment must actually retire selected old owners and charge
factor construction, reuse, row response, reconstruction, fallback and
publication. Sparse asymptotic complexity alone is not a GPU time estimate
for 9- and 23-DOF arms.

Root owns GPU scheduling and integration. Separate agents own factor kernels,
response consumers, and independent numerical diagnostics. Original source
trees and benchmark tools stay frozen throughout each checked capture.

Current node-capture destinations (local raw artifacts):

- `/tmp/fpgs-fourx-kuka-nodes-20260912-01`
- `/tmp/fpgs-fourx-franka-nodes-20260912-01`

No new speedup or physical-quality acceptance is claimed yet.

## First mapping: measured loss, retained rather than promoted

Frozen prototype `0056983b091afae67f34900913bd79ef17431b4f` versus accepted
`42f1492a`, using tool source `15158f3a`. Both arms use
`FEATHER_PGS_SIMPLE_WORLD_ZERO=1`; only the candidate enables
`FEATHER_PGS_ARTICULATED_FACTOR=1`. Full-scale runs use 16,384 environments,
seed 0, 200 warmup steps, 40 synchronized wall steps, and three node-profiled
steps. These are single paired discovery rounds, not repeated acceptance.

| Task / GPU | Baseline physics ms | Prototype physics ms | Baseline / prototype |
| --- | ---: | ---: | ---: |
| KukaAllegro / RTX | 16.367130 | 17.430860 | 0.939x |
| KukaAllegro / GB300 | 15.645259 | 16.712926 | 0.936x |
| Franka / RTX | 6.375587 | 9.700894 | 0.657x |
| Franka / GB300 | 6.122442 | 9.278238 | 0.660x |

The four children per task pass capacity/finite-state checks; parent source
and post-run idle guards pass. These guards do not establish physical or
convergence equivalence. Full512 component diagnostics retain the original
small basis-residual tail failures; no tolerance was relaxed to accept them.
The candidate is default-off and is **not an accepted optimization**.

Kuka's new factor, predictor and whitening bridge cost respectively
1.706/1.394/1.019 ms RTX and 1.355/.967/.934 ms GB in summed node duration,
while original current torque accumulation remains. Repeated global factor
reads and separate dependency traversals defeat the lower arithmetic count.
These sums overlap other work and are not removable critical-path estimates.
The general MF node also changes (2.766 to 1.834 ms on RTX), so trajectory-
dependent workload changes must not be credited as a producer optimization.

Franka's new row-response producer contributes 2.783/2.785 ms interval union,
2.770/2.764 ms exclusive occupancy. It serializes all current row RHSs through
the 11-link tree, while the original classified local owners already prepare
rows concurrently. Their old combined union is only .509/.609 ms, not the
1.191/1.407 ms sum. Prepared-Y local consumers do not reduce that union.
This is a diagnosed ownership failure, not evidence that articulated methods
can never win. Removing the added row pass alone would still leave a loss.

Local source-bound manifests:

- `/tmp/fpgs-fourx-kuka-aba512-20260912-01/manifest.json`,
  SHA256 `aa3e8e87ec7c38dc254f6e454b8ca70789a78dd9ac138de7cbc810123dee23dc`.
- `/tmp/fpgs-fourx-kuka-aba16k-20260912-01/manifest.json`,
  SHA256 `d2afa6f5642ea01b0067dcbbe7d40d6cc523b13d02c4b24927666f66283d605c`.
- `/tmp/fpgs-fourx-franka-aba16k-20260912-01/manifest.json`,
  SHA256 `11582872ba449396bda0a575634cb8cdbc3251315ac0decded584609a094a4de`.
- Franka correlated interval audit:
  `/tmp/fpgs-franka-articulated-attribution-Y8sTiAgS/FINDINGS.md`;
  numerical/source evidence and rehashes are retained alongside it.

## Cause-directed joint-owner retry

This successor keeps the first mapping selectable and adds default-off
`FEATHER_PGS_ARTICULATED_FUSED=1` (requires the factor flag). It merges current
inverse dynamics, held-factor refresh and predictor into one reverse tree
traversal plus one forward traversal. Current and held axes/origins remain
distinct; old selected torque/factor/predictor producers are excluded. Qdd
clearing moves before this owner, without changing any integration allowance.

For Franka, derive only the held upper kinetic encoding once per refresh.
The original local owners apply its upper/transpose triangular action to
current rows in-place. Delete the added global per-row tree-response pass;
retain complete general fallback and J-only local restitution publication.
Kuka retains its whitened recurrence; the transitional bridge keeps dependent
inverse work in private shared storage before publishing its original buffers.
All conversion, fallback and publication costs remain part of the live gate.

This is one architectural repair of the measured loss, not a lane/grid search
or a claim to reach 4x by itself. Even an ideal factor-only change cannot
remove enough of Franka's total budget. Further progress must retire broader
row/collision/dynamics ownership, not polish this boundary for sub-percent
gains. A separate current-input Kuka study is checking exact independence of
the rare MF object worlds and their dense hand rows; coupled worlds must keep
a complete fallback, not be dropped or reclassified by impulse outcome.

Pre-live retry gates: the five fused-dynamics tests pass with actual CUDA on
both RTX PRO6000 and GB300, including the original current-torque producer and
held-operator checks. Seven response CPU tests pass, covering all three local
source seams, upper-factor algebra, general fallback and local restitution
publication. All 16 selected offline builds (four fused dynamics plus twelve
response/consumer builds across the two architectures) have zero reported
stack/spills. Full pre-commit passes. These are component/integration readiness
checks, not a task-level trajectory acceptance or measured speedup.

## Joint-owner retry: measured loss and bounded mapping test

Frozen fused retry `3674ac60bad931dd4df8d460f8eca0fea065c3ee` uses the same
accepted baseline, tool source and full-scale discovery protocol above. Only
the candidate adds the factor and fused flags. Both tasks' source, idle,
capacity and finite-state gates pass; physical acceptance remains pending.

| Task / GPU | Baseline physics ms | Fused physics ms | Baseline / fused |
| --- | ---: | ---: | ---: |
| KukaAllegro / RTX | 16.047575 | 17.523972 | 0.916x |
| KukaAllegro / GB300 | 15.716480 | 16.944491 | 0.928x |
| Franka / RTX | 6.354392 | 6.729150 | 0.944x |
| Franka / GB300 | 6.063712 | 6.397770 | 0.948x |

Franka's local upper-response repair removes most of the first regression,
but does not beat the accepted baseline. Kuka's fused owner now costs
3.8974/2.6002 ms RTX/GB in summed node duration; its whitening bridge costs
.4938/.5127 ms. These overlapping durations are not additive critical-path
savings. The algorithmic reduction in tree traversals has not overcome its
serial dependency chain and hardware mapping. Neither retry is promoted.

The next bounded test is a single four-tree-per-warp mapping, enabled only
by `FEATHER_PGS_ARTICULATED_SUBWARP=1`, which requires the fused flag. Each
tree receives eight lanes for six-dimensional spatial arithmetic, with
private shared storage and subgroup synchronization. Current equations,
held-factor masks, bridge and fallback are unchanged. No lane/block grid
is part of this experiment.

This reduces inactive-lane instructions but quadruples per-CTA shared
storage: 29,904 B for Kuka and 11,056 B for Franka, with 40 registers and
zero stack/spills on both offline architectures. It is **not** an occupancy
improvement: the Kuka estimate is about twelve resident trees versus
thirteen before, and fewer resident warps. Lower latency hiding can erase
the benefit. Only the complete live pipeline can establish a gain.

All nine fused/subwarp controls pass on actual RTX PRO6000 and GB300,
including a five-tree partial CTA, mixed current/held state, invalid
neighbors, graph replay and recovery. The combined CPU suite reports
22 passes and three CUDA-only skips, covered by those device runs. These
are readiness checks, not numerical trajectory or performance acceptance.

Local source-bound fused manifests:

- `/tmp/fpgs-fourx-kuka-fused16k-20260912-01/manifest.json`,
  SHA256 `c252d7ccae0eab05d206800c10bdc90328398117b62ff88cc9c658d259c34c9c`.
- `/tmp/fpgs-fourx-franka-fused16k-20260912-01/manifest.json`,
  SHA256 `d8cd7a7642e244d41e835239d6ce5c5ae3565860dd100dc483fa0dfa8376b63a`.
- `/tmp/fpgs-articulated-subwarp-ready-GpSZuPAO/READY.md` records frozen
  kernel/test hashes and offline resource evidence.
