# Shared heightfield default candidate

Base: `4e42ebe2d9edeb630814da51cb3d4a2c41e99fa6`. Candidate, not accepted.

Subsequent qualification: the shared default passed paired geometry tests and
three-round timing on all four additional rough tasks. See the
[selection report](RECIPE_SELECTION_20260919.md) for measured gains, the same
collision improvement applied to MJWarp, and the bounded coverage. The card
below preserves the pre-experiment hypothesis and original status.

- Remove separated terrain-cell triangle queries and pack independent pairs across lanes in the shared collision owner.
- Apply to heightfield scenes without triangle meshes, regardless of robot, task or solver. Keep explicit opt-out and original mixed-mesh fallback.
- Keep contact generation, reduction, solver budgets and buffer capacities unchanged. Ordering and redundant manifold selection may differ.
- Discovery opt-ins saved 10.15/26.45 ms on ANYmal rough, 8.58/16.17 ms on Go2 rough, 6.29/16.43 ms on Cassie rough and 5.52/14.51 ms on H1 rough (RTX/GB300); these are parent-reported single-round whole-physics observations, not default-candidate qualification.
- Added work is the conservative live-height bound per visited cell; surviving cells retain original triangle/contact processing.
- Require geometry/capacity checks, API validation/opt-out controls and repeated paired whole-physics timing with the same change enabled for MJWarp. Check previously tuned tasks for regressions.
- Reject promotion for missing contacts, invalid geometry, capacity failures or material whole-step regressions; diagnose any mismatch before changing the recipe.

Regression-first controls: default-admission and six missing-input cases fail at the base; nonheightfield omission already passes. Missing elevations are not a supported contacting-heightfield fallback: the old triangle contact kernel also dereferences them. Validate required inputs before dispatch for both midphases instead of claiming a safe fallback. The separate primitive qualification suite passed on CPU and, according to the paired GPU owner, both GPUs; its results qualify the opt-in geometry, not this default candidate's performance.

Candidate CPU checks (CUDA hidden): cell-rejection/packed suites ran 13 tests with one CUDA skip; the general heightfield suite ran 34 tests with five CUDA/experimental skips. All executed tests passed. `uvx pre-commit run -a` and changed-file checks passed. No GPU timing or qualification was run in this candidate worktree before commit.
