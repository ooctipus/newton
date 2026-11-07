Reduce register allocation in fused dense FeatherPGS response publication without changing its arithmetic.
Add the default-off experimental `FEATHER_PGS_DENSE_ROW_BUDGETS=1` storage layout for CUDA C64 tiled-row solves,
preserving the existing row order and projection. Benchmark the task, batch size, and device before enabling
the layout; no configuration migration is required.
