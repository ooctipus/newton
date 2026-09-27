# Corrected sparse G1: first complete live cost

The first complete corrected sparse A/B shows a whole-physics saving against the same-collision cell-only baseline. This is one discovery round, not repeated promotion or a 4× result. The benchmark manifest keeps `performance_accepted=false`, `physical_quality_accepted=false`, and `repeated_timing_evidence=false`.

| Per environment step, 16,384 worlds | RTX baseline a994 | RTX sparse 3bae | Baseline/candidate | GB baseline a994 | GB sparse 3bae | Baseline/candidate |
|---|---:|---:|---:|---:|---:|---:|
| Whole physics, ms | 37.784922400 | 30.598367400 | 1.234867269× | 52.137139700 | 40.946308525 | 1.273305008× |
| Whole environment wall, ms | 52.697644825 | 45.101763601 | 1.168416501× | 66.613864625 | 54.747906451 | 1.216738117× |

Against the cell-only baseline, measured whole-physics removal is 7.186555000 ms RTX and 11.190831175 ms GB. These are whole-window differences, not sums of overlapping node durations. No standalone factor timing or multiplied speedup is substituted for this measurement.

## Recipe and scope

- Baseline: `a994b1b24b6073dcb10e855c10e8197ee116ef14`, heightfield cell reject enabled, sparse factor explicitly disabled.
- Candidate: `3bae8f7719648dc4ed45a9cff821e79c5a565629`, same cell reject enabled, sparse factor explicitly enabled. No direct-terrain or old single-factor mode stacked.
- Fixed Lab backend `53ee6b44c2334341305dbdf385a3916c6b140799`; unchanged old Lab core `1d8feb82d17dbfab8f0772de56f84deae2cb7974`.
- Seed 0, one A/B round, 200 warmup environment steps, 40 wall steps and 40 whole-physics graph steps. Original checked benchmark and frozen `run_live_sparse_checked.py` (9dfac0fc) activation adapter.
- Same capacities: dense 100, configured MF 32/propagation 100, raw 294912, explicit broad 49152, triangle 1769472. No capacity inflation.
- Same physics: sim dt 0.005 s, decimation 4, two 0.0025 s Newton substeps, matrix-free finite-eight maximum. `grouped_dynamics=True`. No Lab/runtime budget changes.
- Every original capture check passed. Both candidate boundaries on both cards show actual sparse owner, 16,384 valid private operators and zero sticky status. Baseline owner absence is explicitly checked.

## Physical and causal evidence

The reservation bug was captured before correction: a one-row reservation's independently recomputed metadata admission wrote two tangents into the next packet. `ROW_FAILURE_RESULT.md` and its pinned NPZ files retain the exact evidence. This was not repaired by widening guards, adding synchronization or changing pivot tolerances.

Shared fix `54fc1e090b1433a5d85254d0bd3645623552c84d` makes compact metadata consume original `contact_slots_needed`, retaining current phi/material/anchor math. Sparse-binding-only commit `3bae8f77` follows it. Accepted G1 ROWS_MASKED does not use that compact producer, so its frozen baseline was not silently refreshed.

The new CPU regressions fail on the old producer in both threshold-disagreement directions. Corrected CPU controls, existing contact/compact/Franka metadata tests and captured raw-geometry replay pass. Full pre-commit passes.

Paired CUDA physical parent 43599 completed all four methods/card with no failures/errors/skips: unchanged actual-tree operator/current-row/held-action tests, complete real Solver continuing-state serial/parallel graph tests, adjacent reservation regression, and both captured failing raw geometries. Physical manifest `2498c5af50f781432074afcfb82868a92612972e5374054a1cc66cbc99c137bb` in `/tmp/fpgs-g1-sparse-reserved-physical-paired-20260913-01`.

Historical warmed component net-response misses remain separately documented in `ASSESSMENT_RESULT.md`/`ASSESSMENT_ORIGINAL_CAUSE.json`; the original captured response also misses that component criterion. No tolerance change or renewed intermediate gate was used for this correction. The narrow captured-input replay controls metadata extent, not a new broad physical-law certificate.

## Completion authority

Whole A/B parent 17359 started 16:03:33 and was reaped exit 0 by 16:06:20 UTC. Final source and idle guards passed; fresh compute query was empty, and both GPUs were released to root. No source or tolerance edits occurred between physical gates and timing.

- Output: `/tmp/fpgs-g1-sparse-live-paired16k-20260913-02`.
- Manifest: `b9e104c4b3b41b0b2cc69bd8c59e18386c6e387a616dd0b8967e81b28ca674e5`.
- Runtime and helpers remain frozen. Repeated balanced timing and any remaining acceptance decision belong to the next root-authorized gate.

## Complete representation boundary

The opt-in owner admits the exact current G1 43-DOF tree and its supported contact endpoints, with no MF or unsupported auxiliary producers. Reverse tree elimination produces a held inverse-whitener with 434 packed entries; exact constructor support bounds allow up to 18 entries per current row. The path replaces the dense mass/factor, predictor triangular solve, dense J/response/diagonal production, and dense residual/update representation together. It retains original current force/composite/transport production, row allocation, finite-eight constraint law, generalized integration and public state.

This is not the earlier forward-only dense single-factor experiment. Current rows use current geometry against the held operator; original refresh/reuse masks and model notifications remain part of the owner. Unsupported constructors keep the original representation before private ownership replaces canonical arrays. These are source-bound implementation facts, not measured per-stage savings.

## Reproduce the accepted gates and discovery recipe

The paired parent is `/tmp/fpgs-capacity-20260911-O7dqFC/run_pair.py`, SHA `155eba000c50e28e5e88522897f89aae519b92d9b6de0b7c97ddb2a95e156e0c`. The physical wrapper is `/tmp/fpgs-g1-tree-factor-OzkV0Het/run_reserved_physical.py`, SHA `08790b12f9b04db2ed79c2bfbce2123e25db3e664302c3b019df5e213aade563`. Captured-geometry helper SHA is `4c1a5e657233476982fb5e381b1918dbfeffbe7f5ff94444a6c1ef8962e2294a`. Use fresh output directories; do not overwrite the evidence above.

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 uv run --no-project \
  --python /home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910/.venv/bin/python python \
  /tmp/fpgs-capacity-20260911-O7dqFC/run_pair.py \
  --newton /home/octi/Projects/newton-fpgs-g1-sparse-20260913 \
  --wrapper /tmp/fpgs-g1-tree-factor-OzkV0Het/run_reserved_physical.py \
  --output /tmp/fpgs-g1-sparse-reserved-physical-paired-20260913-01
```

The unmodified timing adapter is `run_live_sparse_checked.py`, SHA `9dfac0fcd32befc7cbddf2df06e4909386a5efff01d12a608119fbd255d70cca`. It uses benchmark a3a and fixed-Lab adapter c062, checks actual sparse activation at the original untimed boundaries, and retains original source/capacity/budget guards.

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 uv run --no-project \
  --python /home/octi/Projects/IsaacLab.wt/fpgs-opt-20260910/.venv/bin/python python \
  /tmp/fpgs-g1-tree-factor-OzkV0Het/run_live_sparse_checked.py \
  --isaaclab /home/octi/Projects/IsaacLab.wt/contact-reset-20260913 \
  --baseline /home/octi/Projects/newton-fpgs-heightfield-cells-20260913 \
  --candidate /home/octi/Projects/newton-fpgs-g1-sparse-20260913 \
  --output-dir /tmp/fpgs-g1-sparse-live-paired16k-20260913-02 \
  --task g1 --gpus 0 1 --seed 0 --rounds 1 --num-envs 16384 \
  --warmup-steps 200 --steps 40 --profile-steps 40 --trace-mode graph \
  --baseline-env NEWTON_HEIGHTFIELD_CELL_REJECT=1 \
  --baseline-env FEATHER_PGS_SPARSE_FACTOR=0 \
  --candidate-env NEWTON_HEIGHTFIELD_CELL_REJECT=1 \
  --candidate-env FEATHER_PGS_SPARSE_FACTOR=1 \
  --capacity g1:fpgs:dense_max_constraints=100 \
  --capacity g1:fpgs:rigid_contact_max=294912 \
  --capacity g1:fpgs:broad_phase_output_max=49152 \
  --capacity g1:fpgs:max_triangle_pairs=1769472
```

Before promotion, repeat balanced whole timing and retain numerical caveats explicitly. A same-collision corrected MJ retime is a separate denominator; neither a separate-window ratio nor the shared collision gain is multiplied into this sparse measurement.
