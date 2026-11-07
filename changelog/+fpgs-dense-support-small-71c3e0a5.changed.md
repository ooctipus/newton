Keep the dense loop of `SolverFeatherPGS`'s matrix-free sweep for worlds of at most 32 coordinates, where each lane owns one coordinate and the support cannot skip work.
