FPGS framework cleanup and next steps
===================================

Direction
---------

Use the existing V22 prediction model and measured profiles to guide engineering
priorities. Continue code cleanup and framework development now. Whole-model
15/30/50 percent error targets are deferred research goals, not prerequisites
for refactoring, exposing existing decisions, or testing a compatible method.
This supersedes the prediction-first sequence in the six-hour session plan.

Keep V22 coefficients, observations and qualification limits as a fixed reference.
Use estimates to identify likely expensive work and compare candidate designs;
use measured whole-step results to accept performance changes. Mark incomplete
estimates and uncertain rankings. A missing model is an unknown cost, not zero.
Investigate prediction error only when it prevents a concrete engineering
decision; timebox that investigation instead of reopening a general fitting loop.

The retained implementation at 1cd88b12 is the starting point. Work proceeds in
the separate octi/framework-cleanup-20261014 branch. The original source trees,
six-hour evidence and inherited edits remain available for comparison.

Framework boundaries
--------------------

The existing solver already has topology plans, execution plans, shared numerical
factories and resolved-path inspection. Develop those boundaries in place:

* Public options describe physical policy and supported implementation overrides.
* Topology and capacity checks determine which existing methods are applicable.
* Execution planning selects storage and launches within device resource limits.
* Shared row, factor, solve and publication implementations own numerical work.
* Workspace and cache code own allocation, lifetime, invalidation and reset.
* Path inspection explains the requested choice, active method and fallback.

Cost estimates guide choices among compatible implementations. They must not
silently change friction laws, iteration budgets, capacities or physical settings.
Keep model data and research scripts outside the runtime dependency graph.

Ordered implementation packages
-------------------------------

1. Separate launch-resource planning -- first slice implemented
   Move the existing metadata/impulse storage rules, response chunk selection,
   shared-memory estimates and immutable execution plan into execution_plan.py.
   The module uses only the standard library and performs no device allocation.
   Preserve current thresholds, decisions, exceptions and internal import paths.
   Run existing launch, path-selection and storage tests. This is a structural
   cleanup; it adds no performance claim or new method admission.

2. Consolidate configuration and explain routing -- first slice implemented
   configuration.py now resolves the seven existing public implementation
   options, private dense-kernel overrides and CPU choices in one place.
   Existing signatures, override precedence and numerical routing are preserved.
   get_path_selection includes a detached reasons mapping for sparse admission,
   cache selection, publication and per-DOF kernel choices. Resource and sparse
   fallback reasons are recorded at the construction-time decision itself.
   Cache factories retain their existing admission checks; a declined factory
   is described at the topology/representation/lifecycle level, without an
   exhaustive internal rejection trace. No new public configuration object,
   numerical method or performance-based automatic choice is introduced.
   Validation: 85 focused tests pass; retained-source comparisons match 29,792
   configuration cases and 3,456 launch-resource cases. See
   ../../../output/fpgs-route-selection-20261014/RESULTS.rst.

   Follow-up: remove the patch-friction exclusion from held Cholesky dynamics
   caching. This owner already publishes complete state for the ordinary patch
   row pipeline; topology, lifecycle, torsion and compliance checks remain.
   Six new patch tests extend the four loaded-forest controls, including solved
   impulses, sticking history, held factors, model edits and graph/reset cycles.
   Broader friction checks reproduce the retained baseline's two explicit-kernel
   errors and mesh-rolling tolerance failure. See the separate removal evidence
   in ../../../output/fpgs-fallback-removal-20261014/RESULTS.rst.
   Three matched RTX pairs at 16,384 worlds measure 11.688 ms ordinary versus
   11.054 ms cached whole physics per environment step, 5.43% less time, with
   patch friction in both arms. This gain is specific to the measured Franka
   configuration; the historical point-friction explorer timings are unchanged.

3. Consolidate ownership and workspace lifetime -- first contact slice implemented
   Use the existing shared-construction consumer map to remove repeated adapter
   setup and clarify which component owns each row buffer, factor and output.
   Keep allocation out of stepping. Make cache repair and invalidation follow
   existing model-change and reset boundaries. Validate masked resets, changed
   model properties and graph replay for each affected route.

   The native Kuka owner now separates begin/finish dynamics lifetime from its
   contact pipeline. Ordinary and cached routes share patch construction,
   normal-load linking and persistent history publication. Native contact rows
   consume the same patch view; the paired recurrence selects general metadata
   for nonuniform patch rows. Compact held factors remain native throughout the
   step, including mixed articulated/free contacts. The point-only ZERO proof
   and experimental current-contact staging remain restricted to point laws.
   Explicit cached_dynamics='on' now reserves native factor storage before
   automatic sparse selection, using the same policy and topology plan as the
   final owner. Automatic patch selection retains the faster sparse route:
   native patch caching measured 24.958 ms versus 19.081 ms whole physics in the
   first corrected 16,384-world Kuka comparison. This is supported composition,
   not a new performance win. The slower generic patch producer was removed;
   point and patch contacts share the existing support-aware row template.
   Six new full CUDA tests cover loaded patches, mixed dense/MF coupling,
   prescribed bodies, held factors, numeric notifications, masked resets, graph
   replay, demotion/re-entry and explicit/automatic selection. A 91-test shared
   regression suite passes. Default point geometry, rows and solve source remain
   byte-identical to the preceding local implementation.
   The wider friction suite reproduces the same retained three failing subcases.
   Matched performance qualification and the rejected automatic route are in
   ../../../output/fpgs-cache-contact-20261014/RESULTS.rst.
   Three matched pairs preserve the default routes within 0.20% mean physics
   time: Kuka patch 19.093/19.069 ms, Franka patch 11.035/11.042 ms, and Kuka point
   10.596/10.617 ms (previous/final). This slice adds reusable composition and
   explicit compatibility without claiming a new speedup.

4. Retire demonstrably replaced code
   Remove duplicate factories and source-rewriting machinery only after checking
   callers, explicit overrides, tests and fallback paths. Retain numerically
   distinct methods and supported experiments. Historical worktrees and reports
   are evidence; deleting them is outside this cleanup. Prefer small reviewable
   changes over a wholesale rewrite of the solver.

5. Improve one measured framework bottleneck at a time
   Combine current predictions with original profile attribution. Keyed solves
   on Allegro/ANYmal flat, mixed diagonal work on Keyboard, and collision on rough
   terrain are useful candidate families. Select a change by expensive repeated
   work and a credible whole-step benefit, not by largest prediction error.
   Record removed work, added preparation/fallback cost and the physical test.
   Compare compatible choices with fresh balanced whole-step measurements.

6. Integrate and qualify the supported framework
   Run focused tests per change and broader task coverage when routing, buffers
   or numerical behavior change. Before promoting an integrated runtime, run the
   fixed-budget 21-scene controls and appropriate policy/learning checks. Publish
   active routes, supported structures, measured effects and remaining limits.
   General cost prediction remains a separate research milestone.

Acceptance and scope
--------------------

Judge cleanup by clearer ownership, less duplicated policy, preserved public
behavior and passing relevant tests. Judge optimization by numerical/physical
behavior and measured whole-physics performance. Neither depends on meeting a
global prediction-error target first. Exact code comparisons may support a pure
move, but do not replace physical tests for an algorithm change.

The known patch-friction limitation remains visible. Ordinary structural cleanup
can proceed independently; a change to the friction law still requires joint
normal/tangent quality checks. Existing learning evidence retains its original
source and task scope.

Use the authorized RTX only, one GPU worker. New scene/performance comparisons
use 16,384 worlds and unchanged task settings; numerical fixtures keep their
original sizes. No new speedup or full-roadmap completion is claimed by the first
extraction. Parent adoption remains separate from this local implementation.
