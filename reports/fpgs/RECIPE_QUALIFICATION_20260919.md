# Qualify suitable cross-task recipes

## Contract and scope

The user requested suitable, physically validated execution/solver recipes for
all tasks that were measured with generic settings. Original presets are a
reference, not an optimization constraint. Preserve task physics, timestep,
substeps and declared iteration allowances; numerical algorithms need not be
bit-identical. Do not change Isaac Lab sources. Any shared collision changes
must also be enabled for MJWarp before reporting a new backend ratio.

The user additionally requests promotion to defaults when a common configuration
works across the task set. Qualify such a proposal on the already-tuned tasks as
well; do not replace their faster owners blindly. Prefer general capability-based
dispatch and safe fallback over per-task-name flags when no single setting wins.
Default changes require a separate measured runtime revision, with this survey's
source pins preserved as the before reference.

Frozen survey: `/home/octi/Projects/fpgs-survey20-evidence-20260919`.
Newton runtime: `4e42ebe2d9edeb630814da51cb3d4a2c41e99fa6`.
Lab runtime: `53ee6b44c2334341305dbdf385a3916c6b140799`.
New evidence: `/home/octi/Projects/fpgs-recipes-evidence-20260919`.
RTX PRO6000 is primary; run the same task concurrently on RTX and GB300.

Thirteen generic tasks: Ant, ANYmal-D rough, Cartpole, Cassie flat/rough,
Franka reach, G1 flat, Go2 flat/rough, H1 flat/rough, Humanoid and UR10 reach.
Drawer is excluded. Franka lift remains blocked before physics; changing its
reset criteria is not part of recipe qualification. Existing tuned tasks remain
references, not evidence that a recipe generalizes to these thirteen tasks.

## Candidate cards and decision gates

1. **Grouped dynamics plus lazy publication (12 tasks, excluding G1 flat).**
   Replace serial per-articulation execution and avoid intermediate public FK
   publication where fused dynamics and existing manager publication hooks admit
   it. Keep solver mode, constraints and iteration allowance unchanged. Added
   grouping/synchronization and any fallback count in total time. First measure
   actual activation, then compare loaded states and reset/publication transitions
   against the current recipe. This is a screening hypothesis, not a promised
   gain: require substantial whole-physics improvement to promote a new recipe;
   retain current settings on regressions or inactive paths.
2. **G1 flat sparse/kinetic recipe.** Use only paths admitted with its measured
   56-row capacity. Do not inflate to 100 to enable specialized tangent/limit
   kernels. Sparse factor, level update, kinetic state and chain scan are initial
   candidates; confirm activation and loaded-state behavior. Reuse the existing
   sparse and kinetic tests. Savings must include preparation and publication.
3. **Rough-terrain structural work.** First collect a node profile of the current
   ANYmal-D rough recipe. Select applicable existing heightfield paths from the
   observed costly stages and supported geometry; no assumption that the G1
   convex-only benefit covers cylinders. Check physical contact behavior and
   capacities and apply an accepted shared configuration to both backends.
4. **Parallel matrix-free projection, only as a separate numeric candidate.**
   The 48-row flat recipe is not universally suitable. Rough rows exceed 150,
   while the existing parallel matrix-free implementation caps admission at 128;
   larger worlds keep serial fallback. Record actual admission and parallel
   sweep allowance separately from fallback PGS iterations. Require convergence,
   loaded-contact and reset qualification before calling a recipe validated.

Reuse existing capture owners, physical tests and evidence rather than building
a new benchmark framework. One discovery round screens candidates; accepted
performance requires balanced repeated graph timing. Node sums are diagnostic,
not additive critical-path budgets. Clean capacity flags and finite state alone
are not physical validation. Timebox each theory/timing mismatch to one focused
diagnosis and evidence-backed corrective test; do not spend this pass tuning
sub-percent differences.

## Status

Planning and source eligibility audit complete. No new recipe is accepted yet.

### First terrain diagnosis and trial

The old node analyzer rejected a mixed physics/auxiliary graph capture. The raw
SQLite trace remains useful for diagnosis: join successful host graph launches
to their unique `physics_graph` NVTX range and device kernels by process-scoped
correlation ID. Ten profiled environment steps show summed kernel durations:

| Current ANYmal-D rough | RTX ms/step | GB300 ms/step |
| --- | ---: | ---: |
| Triangle contact to reducer | 9.350 | 44.767 |
| Triangle overlap query | 5.701 | 9.163 |
| PGS sweep | 5.914 | 7.040 |

These are instrumented node sums, not accepted whole-step timings or additive
critical-path savings. No gain is claimed from the failed analyzer run.

Initial shared terrain trial: `NEWTON_HEIGHTFIELD_CELL_REJECT=1` and
`NEWTON_NARROW_PHASE_PACKED_HEIGHTFIELD_PAIRS=1`. Unlike finite-query/CSR paths,
these cover primitive shapes as well as boxes/convex hulls. They remove separated
cell work and remap independent pairs while keeping the contact-law writer.
This targets a measured multi-millisecond cost with a credible >10% whole-step
opportunity. Retain existing capacities for the first controlled comparison;
then right-size from new demand if promoted. Existing cell/packed tests plus
matched-state capsule/cylinder coverage are required. Any new MJ comparison
must use the same admitted terrain flags. No full G1 finite-query/CSR bundle is
part of this initial candidate.
