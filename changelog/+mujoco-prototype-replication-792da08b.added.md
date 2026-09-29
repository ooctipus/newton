Add experimental `SolverMuJoCo.replicate()` to create independent homogeneous GPU populations from a pristine one-world solver without rebuilding collision exclusions or recomputing prepared MuJoCo parameters.

Add experimental `SolverMuJoCo.copy_worlds_from()` to transfer continuing worlds between replicas of the same prototype, preserving model properties, state, control, sleeping and warm-start state, and contact references without resetting episodes.
