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
